"""Expert GEMV split-K (P_GU / P_DN work-item parallelism): decode tok/s on the same prompts (exact routing, so every
setting decodes the same text and touches the same experts), after a warm-up pass.
  CFGS=8:4,32:16 DSV_GRAPH=0 DSV_BOOST=0 python3 tests/t_psplit.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load
from dsv_xpu import moe
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import GreedySampler
gen = Generator(model=model, cache=cache, tokenizer=tok)
N = int(os.environ.get("NTOK", "64"))
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Explain how mRNA vaccines teach the immune system to recognize a virus."]


IDS = {}


def run(p):
    gen.enqueue(Job(input_ids=tok.encode(p, add_bos=True), max_new_tokens=N, min_new_tokens=N, stop_conditions=[],
                    sampler=GreedySampler()))
    first = last = None
    n = 0
    ids = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                ids += r["token_ids"].view(-1).tolist()
                now = time.perf_counter()
                if first is None:
                    first = now
                else:
                    n += r["token_ids"].numel()
                    last = now
    IDS[p] = ids
    return n / (last - first)


REF = {}


def same_as_ref():
    """Tokens equal to the first setting's output, up to its first difference (greedy, exact routing: a wrong GEMV
    shows up as a different text; a near-tie may flip one token late in a run)."""
    out = []
    for p in PROMPTS:
        ref, now = REF.setdefault(p, IDS[p]), IDS[p]
        k = next((i for i, (a, b) in enumerate(zip(ref, now)) if a != b), min(len(ref), len(now)))
        out.append(f"{k}/{len(ref)}")
    return " ".join(out)


for p in PROMPTS:
    run(p)                                     # warm-up: JIT for every setting's shapes comes below; caches fill here
for cfg in os.environ.get("CFGS", "8:4,32:16,16:8,8:4,64:32,32:16").split(","):
    moe.P_GU, moe.P_DN = (int(x) for x in cfg.split(":"))
    run(PROMPTS[0][:20])                       # first use of the new partial-sum shapes (JIT) outside the timing
    tps = [run(p) for p in PROMPTS]
    print(f"PSPLIT gate/up {moe.P_GU} down {moe.P_DN}: {sum(tps) / len(tps):.2f} tok/s ({', '.join(f'{t:.2f}' for t in tps)}); "
          f"same text as the first setting for {same_as_ref()} tokens | {tok.decode(torch.tensor(IDS[PROMPTS[0]][:24]))!r}",
          flush=True)
