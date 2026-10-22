"""OpenAI-compatible HTTP server for GLM-5.3-Flash on the B70 (one request at a time).

  in_rootfs.sh --gpu python3 /w/xpu/serve.py [--port 8053] [--cache 49152] [--ram-gb 27]
  (normally started with scripts/serve.sh, which wraps it in scripts/guarded.sh)

Endpoints
  GET  /health, /v1/models
  POST /v1/chat/completions   stream or not; system / user / assistant / tool messages rendered with the model's own
                              chat template (/models/chat_template.jinja); tools -> tool_calls; thinking goes to
                              reasoning_content. chat_template_kwargs.enable_thinking (default true) switches thinking,
                              reasoning_effort (top level or in chat_template_kwargs) picks GLM's Low / High / Max.
                              Without one, GLM_SERVE_REASONING_EFFORT (default low; the template's own default, Max,
                              thinks for thousands of tokens). clear_thinking defaults to GLM_SERVE_CLEAR_THINKING
                              (default 1: earlier turns' reasoning is not resent). A reply that opens a think block
                              although thinking was off still goes to reasoning_content, never to content.
                              Sampling: temperature, top_p, top_k, min_p, max_tokens / max_completion_tokens, stop, seed.
  POST /v1/completions        raw prompt text, stream or not
  POST /quit                  stop the server (127.0.0.1 only)

The model is loaded exactly as xpu/devserver.py loads it (same warm order, tier, block graphs, live speed log in
run/live.log). The generator's page table is kept between requests, so a chat follow-up that repeats the previous
prompt reuses its cached pages and recurrent checkpoint and only prefills what is new.

All GPU work runs on the main thread; HTTP requests are parsed on their own threads and hand jobs to the main thread
through a queue. A client that disconnects (also during the prompt) cancels its job.
"""
import argparse
import json
import os
import queue
import re
import select
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROFILE = "/w/run/route_profile.npy"   # shared with devserver.py: our own decode pick counts per expert
# OpenAI reasoning_effort -> GLM template value; the template knows low and high, anything else renders "Max"
EFFORTS = {"none": "low", "minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}
# Defaults when a request names no effort / clear_thinking. Max makes the Game Master think for thousands of tokens.
DEFAULT_EFFORT = EFFORTS.get(os.environ.get("GLM_SERVE_REASONING_EFFORT", "low").strip().lower(), "low")
# clear_thinking: reasoning of assistant turns before the last user message is dropped from the prompt (the current
# turn's tool rounds keep theirs), so old thinking is not resent and re-read every turn
DEFAULT_CLEAR_THINKING = os.environ.get("GLM_SERVE_CLEAR_THINKING", "1") not in ("0", "false", "no", "")
TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
TOOL_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def log(*a):
    print(" -- serve:", *a, flush=True)


# ---------------------------------------------------------------------------------------------- prompt rendering

class Renderer:
    """The model's chat template via transformers (it needs HF's tojson(ensure_ascii=...)); a plain GLM rendering as
    the fallback when the HF tokenizer cannot load."""

    def __init__(self, model_dir, template=None):
        self.hf = None
        try:
            from transformers import AutoTokenizer
            self.hf = AutoTokenizer.from_pretrained(model_dir)
            if not getattr(self.hf, "chat_template", None):
                tpl = os.path.join(model_dir, "chat_template.jinja")
                if os.path.exists(tpl):
                    self.hf.chat_template = open(tpl).read()
            if template:                    # --chat-template: e.g. the model's own plus a client's prefill hook
                self.hf.chat_template = open(template).read()
        except Exception as e:  # noqa: BLE001
            log(f"HF tokenizer unavailable ({e}); using the built-in GLM prompt format")
            self.hf = None

    def render(self, messages, thinking=True, tools=None, kwargs=None):
        kw = dict(kwargs or {})
        kw.pop("enable_thinking", None)
        effort = kw.pop("reasoning_effort", None)
        effort = DEFAULT_EFFORT if effort is None else EFFORTS.get(str(effort).strip().lower(), DEFAULT_EFFORT)
        if effort != "max":                 # the template renders Max for any value it does not know
            kw["reasoning_effort"] = effort
        kw.setdefault("clear_thinking", DEFAULT_CLEAR_THINKING)
        msgs = normalize_messages(messages)
        if self.hf is not None and getattr(self.hf, "chat_template", None):
            p = self.hf.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True,
                                            **kw)
        else:
            p = "[gMASK]<sop>"
            for m in msgs:
                c = m.get("content")
                if not isinstance(c, str):
                    c = "".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
                p += f"<|{m['role']}|>{c}"
            p += "<|assistant|><think>"
            p += str(kw.get("nq_prefill") or "")
        # the template's generation prompt opens the think block; thinking off = an empty block
        if not thinking and p.endswith("<think>"):
            p += "</think>"
        return p


def normalize_messages(messages):
    """OpenAI -> GLM template: assistant tool-call arguments as dicts, content never None."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", tc)
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {"arguments": fn["arguments"]}
                if not isinstance(fn.get("arguments"), dict):
                    fn["arguments"] = {}
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def parse_tool_calls(content, tools):
    """GLM tool calls: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>. String-typed
    parameters (per the tool's JSON schema) stay raw text; everything else is parsed as JSON when it parses."""
    if not tools or "<tool_call>" not in content:
        return content, []
    schema = {}
    for t in tools:
        fn = t.get("function", t)
        schema[fn.get("name")] = (fn.get("parameters") or {}).get("properties") or {}
    calls = []
    for body in TOOL_CALL.findall(content):
        name = body.split("<arg_key>", 1)[0].strip()
        if not name:
            continue
        args = {}
        for k, v in TOOL_ARG.findall(body):
            k, v = k.strip(), v.strip()
            if (schema.get(name, {}).get(k) or {}).get("type") == "string":
                args[k] = v
            else:
                try:
                    args[k] = json.loads(v)
                except ValueError:
                    args[k] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return TOOL_CALL.sub("", content).strip(), calls


def partial_suffix(text, marker):
    """Length of the longest end of `text` that could be the start of `marker`."""
    for k in range(min(len(marker) - 1, len(text)), 0, -1):
        if marker.startswith(text[-k:]):
            return k
    return 0


class Splitter:
    """Splits streamed model text into reasoning (until </think>), visible content, and a held tool-call tail."""
    TOOL_OPEN = "<tool_call>"     # where the held tool-call tail starts (another model's server sets its own)

    def __init__(self, thinking, tools):
        self.in_think = thinking
        self.tools = bool(tools)
        self.buf = ""            # text not yet classified (a possible partial marker)
        self.held = ""           # from the first TOOL_OPEN on
        self.started = False     # leading whitespace of the content is dropped
        self.lead = not thinking  # thinking off: a reply that still opens with <think> is reasoning, not story

    def feed(self, text, final=False):
        """Returns [(kind, text)] with kind 'reasoning' or 'content'."""
        out = []
        self.buf += text
        if self.lead:
            head = self.buf.lstrip()
            if head.startswith("<think>"):
                self.in_think, self.lead, self.buf = True, False, head[len("<think>"):]
            elif not final and "<think>".startswith(head):
                return out                  # empty or a partial "<think>": wait for more
            else:
                self.lead = False
        while self.buf:
            if self.in_think:
                r, sep, rest = self.buf.partition("</think>")
                if not sep:
                    hold = 0 if final else partial_suffix(r, "</think>")
                    emit = r[:len(r) - hold]
                    self.buf = r[len(r) - hold:]
                    if emit:
                        out.append(("reasoning", emit.replace("<think>", "")))
                    break
                if r:
                    out.append(("reasoning", r.replace("<think>", "")))
                self.in_think, self.buf = False, rest
                continue
            if self.held:
                self.held += self.buf
                self.buf = ""
                break
            if self.tools:
                c, sep, rest = self.buf.partition(self.TOOL_OPEN)
                if sep:
                    self._content(c, out)
                    self.held, self.buf = sep + rest, ""
                    break
                hold = 0 if final else partial_suffix(c, self.TOOL_OPEN)
                self._content(c[:len(c) - hold], out)
                self.buf = c[len(c) - hold:]
                break
            self._content(self.buf, out)
            self.buf = ""
            break
        return out

    def _content(self, c, out):
        if not c:
            return
        if not self.started:
            c = c.lstrip()
            if not c:
                return
            self.started = True
        out.append(("content", c))


# ---------------------------------------------------------------------------------------------- the GPU side

class Request:
    def __init__(self, ids, sampling, stops, max_new):
        self.ids, self.sampling, self.stops, self.max_new = ids, sampling, stops, max_new
        self.out = queue.Queue()     # ("text", str) ... then ("done", info) or ("error", msg)
        self.cancelled = threading.Event()


class Engine:
    def __init__(self, a):
        import numpy as np
        import torch
        from glm_xpu.runner import load, log as glog
        self.torch = torch
        self.a = a
        # ---- the same load as xpu/devserver.py -------------------------------------------------------------------
        meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
        li_of = {l["key"]: l["li"] for l in meta["layers"]}
        order = []
        if not a.no_warm:
            stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
            E = meta["experts_per_layer"]
            up = np.zeros((meta["n_layers"], E))
            for k, v in stats.items():
                if k in li_of:
                    up[li_of[k]] = np.asarray(v, np.float64) / max(sum(v), 1e-9)
            own = np.load(PROFILE).reshape(up.shape) if os.path.exists(PROFILE) else np.zeros_like(up)
            n_own = own.sum(1, keepdims=True)
            w = np.minimum(n_own.sum() / (8 * 42 * 2000), 0.8)
            score = (1 - w) * up + w * own / np.maximum(n_own, 1e-9)
            order = [(int(li), int(e)) for li, e in zip(*np.unravel_index(np.argsort(-score, axis=None), score.shape))
                     if score[li, e] > 0]
            glog(f"warm order: {len(order)} experts, own routing profile weight {w:.2f}")
        mtp = int(os.environ.get("GLM_XPU_MTP", "0"))
        self.model, self.config, self.cache, self.tok, self.tier = load(
            a.model_dir, a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order, mtp=mtp)
        from exllamav3 import Generator
        gkw = {}
        if mtp:
            gkw = dict(draft_model=self.model.mtp[0], draft_cache=self.model.mtp[1], num_draft_tokens=mtp,
                       dynamic_draft_tokens=True)
        self.gen = Generator(model=self.model, cache=self.cache, tokenizer=self.tok,
                             max_chunk_size=int(os.environ.get("GLM_XPU_CHUNK", "8192")), **gkw)
        from glm_xpu import block_graphs
        block_graphs.install(self.tier)
        from glm_xpu import live
        live.install(self.gen, "/w/run/live.log")
        # ------------------------------------------------------------------------------------------------------------
        eos = set()
        try:
            gc = json.load(open(os.path.join(a.model_dir, "generation_config.json")))
            e = gc.get("eos_token_id")
            eos.update(int(x) for x in (e if isinstance(e, list) else [e]) if x is not None)
            self.default_temp = float(gc.get("temperature", 1.0))
            self.default_top_p = float(gc.get("top_p", 1.0))
        except Exception:  # noqa: BLE001
            self.default_temp, self.default_top_p = 1.0, 1.0
        if self.tok.eos_token_id is not None:
            eos.add(int(self.tok.eos_token_id))
        self.eos = sorted(eos)
        self.renderer = Renderer(a.model_dir, a.chat_template)
        self.queue = queue.Queue()
        self.broken = None

    def encode(self, text):
        return self.tok.encode(text, add_bos=False, encode_special_tokens=True)

    def sampler(self, sp):
        from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
        t = sp.get("temperature")
        t = self.default_temp if t is None else float(t)
        if t <= 0:
            return ArgmaxSampler()
        top_p = sp.get("top_p")
        return ComboSampler(temperature=t, top_p=float(self.default_top_p if top_p is None else top_p),
                            top_k=int(sp.get("top_k") or 0), min_p=float(sp.get("min_p") or 0.0),
                            rep_p=float(sp.get("repetition_penalty") or 1.0),
                            pres_p=float(sp.get("presence_penalty") or 0.0),
                            freq_p=float(sp.get("frequency_penalty") or 0.0))

    def run_forever(self):
        torch = self.torch
        while True:
            req = self.queue.get()
            if req is None:
                break
            try:
                with torch.inference_mode():
                    self.run(req)
            except BaseException as e:  # noqa: BLE001
                traceback.print_exc()
                req.out.put(("error", f"{type(e).__name__}: {e}"))
            try:
                self.tier.save_profile(PROFILE)
            except Exception as e:  # noqa: BLE001
                log(f"routing profile not saved: {e}")

    def run(self, req):
        from exllamav3.generator import Job
        gen, cache = self.gen, self.cache
        if cache is not None and not gen.active_jobs and not gen.pending_jobs \
                and len(cache.free_list) < cache.num_slots:
            log(f"recurrent-state slot leak: {cache.num_slots - len(cache.free_list)} slot(s) reclaimed")
            cache.reset_states()
        P = req.ids.shape[-1]
        job = Job(input_ids=req.ids, max_new_tokens=req.max_new, sampler=self.sampler(req.sampling),
                  stop_conditions=list(self.eos) + list(req.stops), decode_special_tokens=False,
                  seed=req.sampling.get("seed"))
        t0 = time.perf_counter()
        gen.enqueue(job)
        t_first, n, info = None, 0, None
        while gen.num_remaining_jobs():
            if req.cancelled.is_set():
                gen.cancel(job)
                log(f"cancelled after {n} tokens ({P} prompt tokens)")
                return
            for r in gen.iterate():
                stage = r.get("stage")
                if stage == "error":
                    self.broken = f"{r.get('error')!r}"[:300]
                    raise RuntimeError(f"generator error: {self.broken}")
                if stage != "streaming":
                    continue
                if t_first is None:
                    t_first = time.perf_counter() - t0
                ids = r.get("token_ids")
                n += int(ids.numel()) if ids is not None else 0
                if r.get("text"):
                    req.out.put(("text", r["text"]))
                if r.get("eos"):
                    reason = r.get("eos_reason")
                    n = max(n, int(r.get("new_tokens", 0)))
                    dt = time.perf_counter() - t0
                    dec = dt - (t_first or dt)
                    info = {"finish_reason": "length" if reason == "max_new_tokens" else "stop",
                            "prompt_tokens": P, "completion_tokens": n,
                            "cached_tokens": int(r.get("cached_tokens", 0) or 0),
                            "prompt_s": round(t_first or 0.0, 3), "total_s": round(dt, 3),
                            "decode_tok_s": round(max(n - 1, 0) / dec, 2) if dec > 0 else None}
        if info is None:
            info = {"finish_reason": "stop", "prompt_tokens": P, "completion_tokens": n, "cached_tokens": 0}
        log(f"{P} prompt tokens ({info['cached_tokens']} cached), first token {info.get('prompt_s')} s, "
            f"{n} tokens at {info.get('decode_tok_s')} tok/s, {info['finish_reason']}")
        req.out.put(("done", info))


# ---------------------------------------------------------------------------------------------- HTTP

def sse(obj):
    return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()


def make_handler(engine, a):
    # an engine for another model may bring its own tool-call format (dsv41/xpu/serve.py)
    split = getattr(engine, "Splitter", Splitter)
    parse_calls = getattr(engine, "parse_tool_calls", parse_tool_calls)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"     # responses end when the connection closes (streams have no length)

        def log_message(self, fmt, *args):   # quiet; one line per request goes to the log from Engine.run
            pass

        def send_json(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def error(self, code, msg):
            self.send_json(code, {"error": {"message": msg, "type": "invalid_request_error" if code < 500
                                            else "server_error", "code": code}})

        def client_gone(self):
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def events(self, req):
            """Yields ("text", str) and finally ("done", info); raises on error. Cancels on client disconnect."""
            while True:
                try:
                    ev = req.out.get(timeout=0.5)
                except queue.Empty:
                    if self.client_gone():
                        req.cancelled.set()
                        raise BrokenPipeError("client went away")
                    continue
                if ev[0] == "error":
                    raise RuntimeError(ev[1])
                yield ev
                if ev[0] == "done":
                    return

        def do_GET(self):
            if self.path.rstrip("/") in ("/health", ""):
                return self.send_json(200 if not engine.broken else 503,
                                      {"ok": not engine.broken, "error": engine.broken})
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self.send_json(200, {"object": "list", "data": [
                    {"id": a.served_name, "object": "model", "owned_by": "local", "max_model_len": a.cache}]})
            self.error(404, f"no route {self.path}")

        def do_POST(self):
            path = self.path.rstrip("/")
            if path == "/quit":
                if self.client_address[0] not in ("127.0.0.1", "::1"):
                    return self.error(403, "quit only from this machine")
                self.send_json(200, {"ok": True})
                engine.queue.put(None)
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self.error(400, "body is not JSON")
            if engine.broken:
                return self.error(503, f"the model failed earlier ({engine.broken}); restart the server")
            try:
                if path in ("/v1/chat/completions", "/chat/completions"):
                    return self.chat(body)
                if path in ("/v1/completions", "/completions"):
                    return self.completions(body)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                try:
                    return self.error(500, f"{type(e).__name__}: {e}")
                except OSError:
                    return
            self.error(404, f"no route {self.path}")

        def submit(self, ids, body):
            P = ids.shape[-1]
            room = a.cache - P - 8
            if room < 1:
                raise ValueError(f"prompt is {P} tokens; the server's context is {a.cache}")
            mx = body.get("max_completion_tokens") or body.get("max_tokens")
            max_new = room if mx is None else max(1, min(int(mx), room))
            s = body.get("stop")
            stops = [s] if isinstance(s, str) else [x for x in (s or []) if x]
            sp = {k: body.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                                           "presence_penalty", "frequency_penalty")}
            req = Request(ids, sp, stops, max_new)
            engine.queue.put(req)
            return req

        def usage(self, info):
            P, n = info["prompt_tokens"], info["completion_tokens"]
            return {"prompt_tokens": P, "completion_tokens": n, "total_tokens": P + n,
                    "prompt_tokens_details": {"cached_tokens": info.get("cached_tokens", 0)}}

        def timings(self, info):
            return {k: info.get(k) for k in ("cached_tokens", "prompt_s", "total_s", "decode_tok_s")}

        def start_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

        def chat(self, body):
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                return self.error(400, "messages must be a non-empty list")
            kw = dict(body.get("chat_template_kwargs") or {})
            if body.get("reasoning_effort") is not None:
                kw.setdefault("reasoning_effort", body["reasoning_effort"])
            thinking = kw.get("enable_thinking", True) is not False
            tools = body.get("tools") if body.get("tool_choice") != "none" else None
            prompt = engine.renderer.render(msgs, thinking, tools, kw)
            ids = engine.encode(prompt)
            req = self.submit(ids, body)
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": a.served_name}
            # the reply starts inside a thought when the prompt leaves one open (thinking on, or a client prefill
            # that continues a thought); a prefill that closes it makes the reply plain content
            sp = split(thinking and prompt.rfind("<think>") > prompt.rfind("</think>"), tools)
            if body.get("stream"):
                def chunk(delta, fin=None, **extra):
                    return sse(dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": fin}], **extra))
                self.start_stream()
                self.wfile.write(chunk({"role": "assistant", "content": ""}))
                self.wfile.flush()
                info = None
                for ev in self.events(req):
                    if ev[0] == "done":
                        info = ev[1]
                        parts = sp.feed("", final=True)
                    else:
                        parts = sp.feed(ev[1])
                    for kind, t in parts:
                        self.wfile.write(chunk({"reasoning_content" if kind == "reasoning" else "content": t}))
                    try:
                        self.wfile.flush()
                    except OSError:
                        req.cancelled.set()
                        raise
                reason = info["finish_reason"]
                if sp.held:
                    content, calls = parse_calls(sp.held, tools)
                    if content:
                        self.wfile.write(chunk({"content": content}))
                    if calls:
                        reason = "tool_calls"
                        self.wfile.write(chunk({"tool_calls": [dict(c, index=i) for i, c in enumerate(calls)]}))
                self.wfile.write(chunk({}, reason, usage=self.usage(info), timings=self.timings(info)))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            reasoning, content, info = [], [], None
            for ev in self.events(req):
                if ev[0] == "done":
                    info = ev[1]
                    parts = sp.feed("", final=True)
                else:
                    parts = sp.feed(ev[1])
                for kind, t in parts:
                    (reasoning if kind == "reasoning" else content).append(t)
            text, calls = "".join(content), []
            if sp.held:
                rest, calls = parse_calls(sp.held, tools)
                text = text.strip() + ("\n" + rest if rest else "")
            msg = {"role": "assistant", "content": text.strip()}
            if thinking:
                msg["reasoning_content"] = "".join(reasoning).strip()
            if calls:
                msg["tool_calls"] = calls
            self.send_json(200, {"id": cid, "object": "chat.completion", "created": created, "model": a.served_name,
                                 "choices": [{"index": 0, "message": msg,
                                              "finish_reason": "tool_calls" if calls else info["finish_reason"]}],
                                 "usage": self.usage(info), "timings": self.timings(info)})

        def completions(self, body):
            prompt = body.get("prompt")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if not isinstance(prompt, str):
                return self.error(400, "prompt must be a string")
            req = self.submit(engine.encode(prompt), body)
            cid, created = "cmpl-" + uuid.uuid4().hex[:24], int(time.time())
            base = {"id": cid, "object": "text_completion", "created": created, "model": a.served_name}
            if body.get("stream"):
                self.start_stream()
                info = None
                for ev in self.events(req):
                    if ev[0] == "done":
                        info = ev[1]
                        continue
                    self.wfile.write(sse(dict(base, choices=[{"index": 0, "text": ev[1], "finish_reason": None}])))
                    self.wfile.flush()
                self.wfile.write(sse(dict(base, choices=[{"index": 0, "text": "", "finish_reason": info["finish_reason"]}],
                                          usage=self.usage(info), timings=self.timings(info))))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            text, info = [], None
            for ev in self.events(req):
                if ev[0] == "done":
                    info = ev[1]
                else:
                    text.append(ev[1])
            self.send_json(200, dict(base, choices=[{"index": 0, "text": "".join(text),
                                                     "finish_reason": info["finish_reason"]}],
                                     usage=self.usage(info), timings=self.timings(info)))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8053)
    ap.add_argument("--model-dir", default="/models")
    ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
    ap.add_argument("--ram-gb", type=float, default=27.0)
    ap.add_argument("--cache", type=int, default=49152, help="context length in tokens (KV cache size)")
    ap.add_argument("--served-name", default="glm-5.3-flash")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--chat-template", help="Jinja chat template to use instead of the model's own")
    a = ap.parse_args()

    # bind first, so a taken port fails before the ~1 minute load
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64      # clients may open several requests at once; they queue for the one GPU job

    srv = Server((a.host, a.port), BaseHTTPRequestHandler)
    engine = Engine(a)
    srv.RequestHandlerClass = make_handler(engine, a)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"server ready on http://{a.host}:{a.port}/v1 (model {a.served_name}, context {a.cache}, default reasoning "
        f"effort {DEFAULT_EFFORT}, clear_thinking {DEFAULT_CLEAR_THINKING})")
    try:
        engine.run_forever()
    except KeyboardInterrupt:
        pass
    log("server stopped")
    # Skip interpreter teardown (see devserver.py): Level Zero can spin for many minutes freeing ~30 GB pinned memory.
    try:
        engine.torch.xpu.synchronize()
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
