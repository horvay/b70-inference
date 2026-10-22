"""First end-to-end generation: load, generate a short greedy reply, report speed and tier stats."""
import os, sys, time
os.environ.setdefault("DSV_NOSYNC", "0")
import torch
from dsv_xpu.runner import load
from dsv_xpu import moe

model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "20")))
if os.environ.get("WARM", "1") == "1":
    tier.warm(tier.load_profile())
from exllamav3 import Generator
gen = Generator(model=model, cache=cache, tokenizer=tok)
prompt = os.environ.get("PROMPT", "The capital of France is")
for n in (8, int(os.environ.get("NTOK", "48"))):
    t0 = time.time()
    out = gen.generate(prompt=prompt, max_new_tokens=n, completion_only=True, add_bos=True)
    dt = time.time() - t0
    print(f"--- {n} tokens in {dt:.1f} s ({n / dt:.2f} tok/s incl. prompt): {out!r}", flush=True)
    print("tier stats", tier.stats, flush=True)
tier.save_profile()
