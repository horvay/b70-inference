"""GPU kernel profile of decode steps only (prompt excluded) + CPU enqueue vs completion time per forward.
  DSV_NOSYNC=1 NTOK=16 python3 tests/kprof2.py"""
import os, time, statistics
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from torch.profiler import profile, ProfilerActivity
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
tier.promote_budget = int(os.environ.get("BUDGET", tier.promote_budget))
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Once upon a time, in a small village by the sea,", max_new_tokens=32, add_bos=True)
N = int(os.environ.get("NTOK", "16"))
fw = model.forward
cpu_t, all_t = [], []


def timed(*a, **k):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    r = fw(*a, **k)
    t1 = time.perf_counter()
    torch.xpu.synchronize()
    t2 = time.perf_counter()
    cpu_t.append(t1 - t0); all_t.append(t2 - t0)
    return r


model.forward = timed
gen.generate(prompt="A short note about rivers", max_new_tokens=N, add_bos=True)
model.forward = fw
print(f"decode forward: enqueue (CPU) {statistics.median(cpu_t) * 1000:.1f} ms, complete {statistics.median(all_t) * 1000:.1f} ms (median of {len(cpu_t)})", flush=True)
j = Job(input_ids=tok.encode("The old lighthouse keeper looked at the horizon and", add_bos=True), max_new_tokens=N,
        min_new_tokens=N, stop_conditions=[])
gen.enqueue(j)
for r in gen.iterate():      # prompt + first token outside the profile
    pass
steps = 0
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    t0 = time.time()
    while gen.num_remaining_jobs():
        gen.iterate()
        steps += 1
    torch.xpu.synchronize()
    dt = time.time() - t0
print(f"{steps} decode steps in {dt:.2f} s ({dt / steps * 1000:.0f} ms/step under the profiler)", flush=True)
ka = prof.key_averages()
dev = [e for e in ka if e.self_device_time_total > 0]
tot = sum(e.self_device_time_total for e in dev)
print(f"total device time {tot / 1000 / steps:.2f} ms/step", flush=True)
for e in sorted(dev, key=lambda e: -e.self_device_time_total)[:32]:
    print(f"  {e.self_device_time_total / 1000 / steps:8.3f} ms/step {e.count / steps:7.1f}/step  {e.key[:100]}")
