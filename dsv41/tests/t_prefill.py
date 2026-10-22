"""Short-prompt processing time: three ~PLEN-token prompts on different topics (the first pays any JIT), then an op
profile of a fourth.
  PLEN=512 python3 tests/t_prefill.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
os.environ.setdefault("DSV_PREFILL_PROBE", "1")
import torch
from torch.profiler import profile, ProfilerActivity
from dsv_xpu.runner import load
from dsv_xpu import moe
P = int(os.environ.get("PLEN", "512"))
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok)
words = open("/w/dsv41/tests/corpus.txt").read().split()


def run(off):
    ids = tok.encode(" ".join(words[off:off + P]), add_bos=True)[:, :P]
    s0 = dict(tier.stats)
    p0 = dict(moe.PREFILL_STATS)
    t0 = time.time()
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, stop_conditions=[]))
    while gen.num_remaining_jobs():
        gen.iterate()
    dt = time.time() - t0
    d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in tier.stats}
    pd = {k: moe.PREFILL_STATS.get(k, 0) - p0.get(k, 0) for k in moe.PREFILL_STATS}
    print(f"PREFILL {ids.shape[-1]} tokens (offset {off}): {dt:.1f} s | MoE {pd.get('moe', 0):.1f} s, staged waits "
          f"{pd.get('wait', 0):.1f} s, scratch pieces {pd.get('scratch', 0):.0f}, NVMe->RAM pieces {d.get('nvme', 0)} "
          f"({d.get('t_nvme', 0):.1f} s)", flush=True)
    return dt


for off in (5000, 15000):
    run(off)
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
run(25000)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
print(s.getvalue(), flush=True)
if os.environ.get("PROF", "1") == "1":
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        run(35000)
    ka = prof.key_averages()
    print("== by CPU time (self)")
    for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:30]:
        print(f"  {e.self_cpu_time_total / 1e6:8.2f} s cpu  {e.count:8d} calls  {e.key[:90]}")
    print("== by device time")
    for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:25]:
        print(f"  {e.self_device_time_total / 1e6:8.2f} s dev  {e.count:8d} calls  {e.key[:90]}")
