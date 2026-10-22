"""Kernel-level profile of decode steps with torch.profiler (XPU activity): GPU time per step by kernel name."""
import os, time
os.environ.setdefault("DSV_NOSYNC", "1")
import torch
from torch.profiler import profile, ProfilerActivity
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "16")))
tier.warm(tier.load_profile())
from exllamav3 import Generator
gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Once upon a time, in a small village by the sea,", max_new_tokens=24, add_bos=True)
N = 16
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    t0 = time.time()
    gen.generate(prompt="The old lighthouse keeper looked at the horizon and", max_new_tokens=N, add_bos=True)
    dt = time.time() - t0
print(f"{N} tokens in {dt:.2f} s", flush=True)
ka = prof.key_averages()
tot_dev = sum(e.self_device_time_total for e in ka)
print(f"total device time {tot_dev / 1000 / N:.2f} ms/token over {N} tokens (incl. prompt)", flush=True)
rows = sorted(ka, key=lambda e: -e.self_device_time_total)[:30]
for e in rows:
    print(f"  {e.self_device_time_total / 1000 / N:8.3f} ms/tok  {e.count / N:7.1f} calls/tok  {e.key[:90]}")
