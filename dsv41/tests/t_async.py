"""Async pager smoke test on the real model: a short greedy generation with the pager on (DSV_PAGER_DEBUG=1 logs every
handoff)."""
import os, sys, time, faulthandler
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
faulthandler.dump_traceback_later(int(os.environ.get("DUMP_S", "600")), exit=False)
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1, ram_expert_gb=16.0)
tier.warm(tier.load_profile())
from exllamav3 import Generator
gen = Generator(model=model, cache=cache, tokenizer=tok)
t0 = time.time()
print(gen.generate(prompt="The capital of France is", max_new_tokens=int(os.environ.get("NTOK", "8")), add_bos=True),
      flush=True)
print(f"done in {time.time() - t0:.1f} s", flush=True)
