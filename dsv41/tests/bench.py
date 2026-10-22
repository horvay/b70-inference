"""Decode / prefill benchmark: N concurrent users (jobs), each with its own prompt of P tokens, greedy, fixed output
length. Reports per-user prefill time and decode speed (per user and total).
  BENCH="1:32,2:32,3:32,1:40000,2:40000,3:40000" NEW=64 python3 tests/bench.py
"""
import os, sys, time, random
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load

CASES = [tuple(int(v) for v in c.split(":")) for c in os.environ.get("BENCH", "1:32").split(",")]
NEW = int(os.environ.get("NEW", "64"))
max_users = max(u for u, _ in CASES)
max_p = max(p for _, p in CASES)
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=-(-max(8192, max_users * (max_p + NEW + 512) + 4096) // 256) * 256,
                                       max_batch_size=max_users, ram_expert_gb=float(os.environ.get("RAM_GB", "24")),
                                       vram_reserve_gb=float(os.environ.get("VRAM_RESERVE_GB", "3")))
if os.environ.get("WARM", "1") == "1":
    tier.warm(tier.load_profile() if hasattr(tier, "load_profile") else None)
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=int(os.environ.get("CHUNK", "8192")))

# one-time kernel JIT (SYCL spir64 + Triton) happens on the first prompt/decode of a process: do it before measuring
t0 = time.time()
gen.generate(prompt="Hello, this is a warm-up run of the model.", max_new_tokens=16, add_bos=True)
_w = open("/w/dsv41/tests/corpus.txt").read().split()
gen.generate(prompt=" ".join(_w[5000:5600]), max_new_tokens=24, add_bos=True)   # prompt path + selection regime
for u in range(2, max_users + 1):        # batched decode steps capture their own graphs
    for i in range(u):
        gen.enqueue(Job(input_ids=tok.encode(f"Warm-up job number {i}.", add_bos=True), max_new_tokens=12,
                        min_new_tokens=12, stop_conditions=[]))
    while gen.num_remaining_jobs():
        gen.iterate()
print(f"warm-up generate: {time.time() - t0:.1f} s", flush=True)

words = open("/w/dsv41/tests/corpus.txt").read().split() if os.path.exists("/w/dsv41/tests/corpus.txt") else None


def prompt_ids(n, seed):
    rng = random.Random(seed)
    if words:
        o = rng.randrange(len(words))
        ws = (words[o:] + words[:o]) * 2
        txt = " ".join(ws[:int(n * 0.95) + 16])
    else:
        txt = " ".join(f"item{rng.randint(0, 99999)}" for _ in range(n))
    ids = tok.encode(txt, add_bos=True)
    return ids[:, :n]


def run(users, plen, tag):
    jobs = []
    for u in range(users):
        ids = prompt_ids(plen, seed=hash((tag, u, plen)) & 0xFFFF)
        j = Job(input_ids=ids, max_new_tokens=NEW, min_new_tokens=NEW, stop_conditions=[], identifier=u)
        jobs.append(j)
        gen.enqueue(j)
    t0 = time.time()
    first, last, count = {}, {}, {}
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") != "streaming":
                continue
            u = r["identifier"]
            n = len(r.get("token_ids", torch.empty(0)).view(-1)) if r.get("token_ids") is not None else 0
            if n == 0:
                continue
            now = time.time()
            if u not in first:
                first[u] = now
                count[u] = 0
            else:
                count[u] += n
            last[u] = now
    tot = time.time() - t0
    pre = [first[u] - t0 for u in range(users)]
    dec = [count[u] / max(last[u] - first[u], 1e-6) for u in range(users)]
    # decode window where all users decode at once
    t_all0 = max(first.values())
    t_all1 = min(last.values())
    print(f"RESULT users={users} prompt={plen}: first-token times {', '.join(f'{p:.1f}' for p in pre)} s "
          f"(prefill {users * plen / max(pre):.0f} tok/s overall); decode per user "
          f"{', '.join(f'{d:.2f}' for d in dec)} tok/s, total {sum(dec):.2f} tok/s; wall {tot:.1f} s", flush=True)
    print("   tier", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tier.stats.items()}, flush=True)


for i, (u, p) in enumerate(CASES):
    run(u, p, f"c{i}")
