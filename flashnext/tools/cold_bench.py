#!/usr/bin/env python3
"""Benchmarks for the cold tier / RAM margin (README "Smaller RAM footprint"). Runs on the host against a live server
started with EXL3_MOE_ROUTE_CTL=/run/b70/route_ctl.json (state/run/flashnext/route_ctl.json on the host).

    cold_bench.py speed <users> <short|long> [max_tokens]   N concurrent streamed requests, each with a unique prompt
    cold_bench.py margins <m1,m2,..> [reps]                 decode speed (1 user, short GM turns) per margin
    cold_bench.py quality <m1,m2,..>                        teacher-forced top-1 / KL of decode routing vs exact

quality: reference replies are generated once (state/run/flashnext/quality_ref.json). For each setting the cache is flushed, every
prompt is prefilled with exact routing (so the prompt is cached as in real use), then prompt + reply is scored with the
decode routing applied to the reply positions (route control "bias_prompts"). "exact" scores the reply with exact
routing too. KL is over the union of both sides' top-20 tokens (approximate).
"""
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import urllib.request

BASE = os.environ.get("API", "http://127.0.0.1:30250")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # flashnext/
RUN = os.path.join(os.path.dirname(HERE), "state", "run", "flashnext")  # /run/b70 inside the sandbox
CTL = f"{RUN}/route_ctl.json"
# the system prompt of the "short" turns: a game-master prompt of your own (GM_PROMPT=<file>), else a stand-in
GM = open(os.environ["GM_PROMPT"]).read() if os.environ.get("GM_PROMPT") else (
    "You are the game master of a text adventure. Narrate what happens in vivid second person, voice the people the "
    "player meets, keep track of the world, and end each reply where the player can act.")
ACTIONS = [   # held-out style player turns (not from any campaign)
    "I offer the goblin chieftain a sack of salt in exchange for safe passage through the hills.",
    "I search the shipwreck on the beach for anything useful, starting with the captain's cabin.",
    "I challenge the arrogant knight to a duel at dawn in front of the whole market square.",
    "I follow the strange child through the fog-filled alleys, keeping my distance so she doesn't notice me.",
    "I read the faded letter we found in the desk aloud to my companions and ask what they make of it.",
    "I bargain with the witch in the swamp for a cure for the sleeping sickness spreading through the village.",
    "I ask the stable boy which road the masked riders took and slip him a copper for his trouble.",
    "I climb onto the roof of the granary to see who is moving through the square after curfew.",
]
LEDGER = json.load(open(f"{HERE}/bench/long_prompt_16k.json"))["messages"][0]["content"]
LEDGER = LEDGER[:LEDGER.rindex("\n\n")]


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def set_ctl(margin, bias_prompts=False):
    with open(CTL + ".tmp", "w") as f:
        json.dump({"margin": margin, "bias_prompts": bias_prompts}, f)
    os.replace(CTL + ".tmp", CTL)
    time.sleep(1.3)      # the server re-reads it at most once a second, at the next forward


def chat(msgs, max_tokens):
    body = {"model": "flashnext-uncensored", "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": msgs}
    t0 = time.perf_counter()
    ttft, usage, n, t_last = None, {}, 0, None
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            delta = d["choices"][0].get("delta", {}) if d.get("choices") else {}
            if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                n += 1
                t_last = time.perf_counter()
                if ttft is None:
                    ttft = t_last - t0
    out = usage.get("completion_tokens", n)
    t_end = (t_last or time.perf_counter()) - t0
    return {"t0": t0, "prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "out": out, "ttft": ttft or 0.0, "end": t_end}


def prompt_for(kind, i, tag):
    if kind == "short":    # a game-master turn: the voice prompt is shared (cached after the first), the action is not
        return [{"role": "system", "content": GM},
                {"role": "user", "content": f"[{tag}-{i}] " + ACTIONS[i % len(ACTIONS)]}]
    return [{"role": "user", "content": f"Ledger copy {tag}-{i}.\n\n" + LEDGER +
             "\n\nDescribe in two paragraphs the patterns you notice in the goods, places and tolls above."}]


def speed(users, kind, max_tokens=200):
    tag = f"{time.time():.0f}"
    with cf.ThreadPoolExecutor(users) as ex:
        t0 = time.perf_counter()
        rs = list(ex.map(lambda i: chat(prompt_for(kind, i, tag), max_tokens), range(users)))
    wall = time.perf_counter() - t0
    for r in rs:
        r["start"] = r["t0"] - t0
        r["first"] = r["start"] + r["ttft"]
        r["done"] = r["start"] + r["end"]
        r["dec"] = (r["out"] - 1) / (r["end"] - r["ttft"]) if r["out"] > 1 and r["end"] > r["ttft"] else 0.0
    new_prompt = sum(r["prompt"] - (r["cached"] or 0) for r in rs)
    last_first = max(r["first"] for r in rs)
    # decode throughput while every request was decoding: tokens from the last first-token to the first finish
    both0, both1 = last_first, min(r["done"] for r in rs)
    res = {"users": users, "kind": kind, "prompt_tokens": [r["prompt"] for r in rs], "cached": [r["cached"] for r in rs],
           "ttft_s": [round(r["ttft"], 2) for r in rs], "prefill_tok_s": round(new_prompt / last_first, 0),
           "decode_tok_s_per_user": [round(r["dec"], 1) for r in rs],
           "decode_tok_s_total": round(sum(r["out"] - 1 for r in rs) / max(1e-9, max(r["done"] for r in rs) - min(r["first"] for r in rs)), 1),
           "out_tokens": [r["out"] for r in rs], "wall_s": round(wall, 1)}
    if users > 1 and both1 > both0:
        res["all_decoding_window_s"] = round(both1 - both0, 1)
    print(json.dumps(res), flush=True)
    return res


def margins(ms, reps=3):
    for m in ms:
        set_ctl(m)
        tag = f"m{m}-{time.time():.0f}"
        decs = []
        for i in range(reps):
            r = chat(prompt_for("short", i, tag), 200)
            decs.append((r["out"] - 1) / (r["end"] - r["ttft"]))
        print(json.dumps({"margin": m, "decode_tok_s": [round(d, 1) for d in decs],
                          "mean": round(sum(decs) / len(decs), 1)}), flush=True)


def gen(text=None, ids=None, **kw):
    body = {"sampling_params": {"temperature": 0, "max_new_tokens": kw.pop("max_new_tokens", 1)}, **kw}
    body["text" if text is not None else "input_ids"] = text if text is not None else ids
    with post("/generate", body) as r:
        return json.loads(r.read())


def template(sys_, user):
    return (f"<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _prompt_of(r):
    return r["text"][:r["text"].index("</think>\n\n") + len("</think>\n\n")]


def quality(ms):
    ref_path = f"{RUN}/quality_ref.json"
    if os.path.exists(ref_path):
        refs = json.load(open(ref_path))
    else:
        set_ctl(0.0)
        refs = []
        for a in ACTIONS:     # no prompt logprobs here: a full-vocabulary score per prompt token ran the card out of VRAM
            text = template(GM, a)
            r = gen(text, max_new_tokens=160)
            refs.append({"text": text + r["text"], "prompt_len": r["meta_info"]["prompt_tokens"]})
        json.dump(refs, open(ref_path, "w"))
    dists = {}
    for m in ["exact"] + ms:
        set_ctl(0.0, False)
        urllib.request.urlopen(urllib.request.Request(BASE + "/flush_cache", method="POST"), timeout=60).read()
        for r in refs:                           # prompts prefilled with exact routing, as in real use
            gen(_prompt_of(r))
        if m != "exact":
            set_ctl(float(m), True)
        t0 = time.perf_counter()
        out = []
        for r in refs:
            res = gen(r["text"], return_logprob=True, logprob_start_len=r["prompt_len"], top_logprobs_num=20)
            tops = [t for t in res["meta_info"]["input_top_logprobs"] if t]
            out.append([{t[1]: t[0] for t in pos} for pos in tops])
        dists[m] = out
        set_ctl(0.0, False)
        if m == "exact":
            print(json.dumps({"setting": "exact", "positions": sum(len(o) for o in out),
                              "score_s": round(time.perf_counter() - t0, 1)}), flush=True)
            continue
        agree = n = 0
        kl = 0.0
        for a, b in zip(dists["exact"], out):
            for pa, pb in zip(a, b):
                n += 1
                agree += max(pa, key=pa.get) == max(pb, key=pb.get)
                fa, fb = min(pa.values()), min(pb.values())
                toks = set(pa) | set(pb)
                la = {t: pa.get(t, fa) for t in toks}
                lb = {t: pb.get(t, fb) for t in toks}
                za = math.log(sum(math.exp(v) for v in la.values()))
                zb = math.log(sum(math.exp(v) for v in lb.values()))
                kl += sum(math.exp(la[t] - za) * ((la[t] - za) - (lb[t] - zb)) for t in toks)
        print(json.dumps({"margin": m, "top1_vs_exact": round(100 * agree / n, 1), "kl": round(kl / n, 4),
                          "positions": n}), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "speed":
        speed(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    elif cmd == "margins":
        margins([float(x) for x in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif cmd == "quality":
        quality(sys.argv[2].split(","))
