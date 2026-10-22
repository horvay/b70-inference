"""OpenAI-compatible HTTP server for DeepSeek-V4.1-Flash on the B70 (one request at a time), plus a chat page.

  in_rootfs.sh --gpu python3 /w/dsv41/xpu/serve.py [--port 8054] [--cache 32768] [--ram-gb 24]
  (started by `b70 serve` or scripts/serve.sh, which queue it on the B70 through b70q)

Endpoints
  GET  /                      a chat page (streams from /v1/chat/completions)
  GET  /health, /v1/models
  POST /v1/chat/completions   stream or not. Prompts are rendered with DeepSeek's reference encoder (dsv_encoding.py:
                              system / user / assistant / tool messages, tools in DSML, mid-conversation system turns);
                              the model's DSML tool calls come back as tool_calls. Thinking is off unless the request
                              sets chat_template_kwargs.enable_thinking (default: DSV_SERVE_THINKING=0); it goes to
                              reasoning_content. reasoning_effort (top level or in chat_template_kwargs): low / high /
                              max or 1-100 (OpenAI's none / minimal -> low, medium -> high, xhigh -> max).
                              chat_template_kwargs.nq_prefill: text after the generation prompt (NQ's thinking prefill;
                              ending in </think> it closes the thought). Any --chat-template is ignored.
                              Sampling: temperature, top_p, top_k, min_p, max_tokens / max_completion_tokens, stop, seed.
  POST /v1/completions        raw prompt text, stream or not
  POST /quit                  stop the server (127.0.0.1 only)

The HTTP side (request parsing, streaming, reasoning split, cancel on disconnect) is GLM's (glm/xpu/serve.py); this
file loads DSV4.1 (dsv_xpu.runner, DSpark on by default) and renders and parses its prompts. All GPU work runs on the
main thread.
"""
import argparse
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid

_spec = importlib.util.spec_from_file_location("glm_serve", "/w/glm/xpu/serve.py")
glm_serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm_serve)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

import dsv_encoding as E  # noqa: E402

DEFAULT_THINKING = os.environ.get("DSV_SERVE_THINKING", "0") not in ("0", "false", "no", "")
# OpenAI reasoning_effort -> the encoder's low (50) / high (75) / max (100)
EFFORTS = {"none": "low", "minimal": "low", "low": "low", "medium": "high", "high": "high", "xhigh": "max", "max": "max"}
# a tool-call block starts here (the encoder writes "\n\n" before it; a model that leaves the blank line out still counts)
TOOL_OPEN = f"<{E.dsml_token}{E.tool_calls_block_name}"


def log(*a):
    print(" -- serve:", *a, flush=True)


def reasoning_effort(value):
    """A request's reasoning_effort as the encoder takes it: low / high / max or an int in 1..100; None: its default."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(1, min(100, int(value)))
    text = str(value).strip().lower()
    if text.isdigit():
        return max(1, min(100, int(text)))
    return EFFORTS.get(text, E.DEFAULT_REASONING_EFFORT)


def text_of(content):
    """OpenAI content (a string or text parts) as one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(x.get("text", "") for x in content if isinstance(x, dict))


class Renderer:
    """DeepSeek's reference encoder (dsv_encoding.encode_messages) over OpenAI-shaped requests."""

    def render(self, messages, thinking=True, tools=None, kwargs=None):
        kw = dict(kwargs or {})
        on = kw.get("enable_thinking")
        on = DEFAULT_THINKING if on is None else bool(on)
        msgs = []
        for m in messages:
            m = dict(m)
            if m.get("role") == "developer":
                m["role"] = "system"
            m["content"] = text_of(m.get("content"))
            msgs.append(m)
        if tools:
            # the encoder takes the tool schemas on the leading system message
            if msgs and msgs[0].get("role") == "system":
                msgs[0]["tools"] = tools
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": tools})
        prompt = E.encode_messages(msgs, thinking_mode="thinking" if on else "chat",
                                   reasoning_effort=reasoning_effort(kw.get("reasoning_effort")))
        return prompt + str(kw.get("nq_prefill") or "")


class Splitter(glm_serve.Splitter):
    """GLM's reasoning / content / tool-call split, with DSV4.1's DSML block as the tool-call start."""
    TOOL_OPEN = TOOL_OPEN


def parse_tool_calls(held, tools):
    """The held tail (from TOOL_OPEN on) as OpenAI tool_calls via the reference parser: (leftover content, calls).
    A block the parser rejects stays visible as content."""
    if not held.startswith(TOOL_OPEN):
        return held, []
    try:
        _, _, calls = E.parse_tool_calls(len(TOOL_OPEN), held)
    except (ValueError, AssertionError) as e:
        log(f"tool call not parsed ({e}); returned as text")
        return held, []
    out = []
    for c in calls:
        name = c["name"] if c.get("namespace") is None else f"{c['namespace']}::{c['name']}"
        out.append({"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                    "function": {"name": name, "arguments": c["arguments"]}})
    return "", out


class Engine:
    """The interface glm_serve.make_handler uses: renderer, encode, queue, broken, run_forever."""

    def __init__(self, a):
        import torch
        from dsv_xpu.runner import load
        from exllamav3 import Generator
        self.torch = torch
        self.model, self.config, self.cache, self.tok, self.tier = load(
            a.model_dir, a.store, cache_tokens=a.cache, max_batch_size=1, ram_expert_gb=a.ram_gb,
            vram_reserve_gb=a.vram_reserve_gb)
        self.tier.warm(self.tier.load_profile())
        draft = getattr(self.model, "_dsv_draft", None)
        gkw = dict(draft_model=draft[0], draft_cache=draft[1]) if draft else {}
        self.gen = Generator(model=self.model, cache=self.cache, tokenizer=self.tok, max_chunk_size=a.chunk, **gkw)
        gc = json.load(open(os.path.join(a.model_dir, "generation_config.json")))
        e = gc.get("eos_token_id")
        self.eos = sorted({int(x) for x in (e if isinstance(e, list) else [e]) if x is not None})
        self.default_temp = float(gc.get("temperature", 1.0))
        self.default_top_p = float(gc.get("top_p", 1.0))
        self.renderer = Renderer()
        self.Splitter, self.parse_tool_calls = Splitter, parse_tool_calls
        self.queue = queue.Queue()
        self.broken = None
        self.a = a
        # first-use kernel compiles (prompt and decode shapes) before the first real request
        t0 = time.time()
        with torch.inference_mode():
            self.gen.generate(prompt=self.renderer.render([{"role": "user", "content": "Hello."}]), max_new_tokens=8,
                              add_bos=False, encode_special_tokens=True, completion_only=True)
        log(f"warm-up generation {time.time() - t0:.0f} s")

    encode = glm_serve.Engine.encode
    sampler = glm_serve.Engine.sampler
    run = glm_serve.Engine.run

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


def make_handler(engine, a):
    H = glm_serve.make_handler(engine, a)
    page_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")

    class DH(H):
        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/chat"):
                page = open(page_path, "rb").read()        # read per request: edits show without a restart
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            return H.do_GET(self)

    return DH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8054)
    ap.add_argument("--model-dir", default="/models")
    ap.add_argument("--store", default="/nvx")
    ap.add_argument("--ram-gb", type=float, default=24.0)
    ap.add_argument("--vram-reserve-gb", type=float, default=3.0)
    ap.add_argument("--cache", type=int, default=32768, help="context length in tokens (KV cache size)")
    ap.add_argument("--chunk", type=int, default=2048, help="prompt tokens per prefill chunk")
    ap.add_argument("--served-name", default="deepseek-v4.1-flash")
    a = ap.parse_args()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 64

    srv = Server((a.host, a.port), BaseHTTPRequestHandler)      # bind first: a taken port fails before the load
    engine = Engine(a)
    srv.RequestHandlerClass = make_handler(engine, a)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"server ready on http://{a.host}:{a.port}/ (API /v1, model {a.served_name}, context {a.cache}, "
        f"thinking {'on' if DEFAULT_THINKING else 'off'} by default)")
    try:
        engine.run_forever()
    except KeyboardInterrupt:
        pass
    try:
        engine.tier.save_profile()          # the next start warms from this session's routing
    except Exception as e:  # noqa: BLE001
        log(f"routing profile not saved: {e}")
    log("server stopped")
    # skip interpreter teardown: Level Zero can spin for minutes freeing the pinned RAM tier
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
