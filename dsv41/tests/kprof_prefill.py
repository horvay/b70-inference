"""Kernel/op profile of one long prompt chunk (after a warm-up chunk for the JIT)."""
import os, time
import torch
from torch.profiler import profile, ProfilerActivity
from dsv_xpu.runner import load
P = int(os.environ.get("PLEN", "2048"))
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=-(-(2 * P + 1024) // 256) * 256, max_batch_size=1,
                                       ram_expert_gb=16, vram_reserve_gb=6)
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=8192)
words = open("/w/dsv41/tests/corpus.txt").read().split()
def run(off):
    ids = tok.encode(" ".join(words[off:off + P]), add_bos=True)[:, :P]
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, stop_conditions=[]))
    while gen.num_remaining_jobs():
        gen.iterate()
run(3000)
with profile(activities=[ProfilerActivity.XPU]) as prof:
    t0 = time.time(); run(30000); dt = time.time() - t0
print(f"{P} tokens in {dt:.1f} s", flush=True)
ka = prof.key_averages()
print("== by device time")
for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:25]:
    print(f"  {e.self_device_time_total / 1e6:8.2f} s dev  {e.count:8d} calls  {e.key[:90]}")
print("== by CPU time")
for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:25]:
    print(f"  {e.self_cpu_time_total / 1e6:8.2f} s cpu  {e.count:8d} calls  {e.key[:90]}")
