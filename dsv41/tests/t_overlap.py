"""How many distinct experts does a block of w consecutive tokens need (speculative-decoding verification cost)?
Logs every decode layer's picks for NTOK greedy tokens, then per layer: mean distinct experts over w-token windows
divided by 6 (one token). DSV_PICKLOG=1 NTOK=200 python3 tests/t_overlap.py"""
import os
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
os.environ["DSV_PICKLOG"] = "1"
import numpy as np
from dsv_xpu.runner import load
from dsv_xpu import moe
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
from exllamav3 import Generator
gen = Generator(model=model, cache=cache, tokenizer=tok)
words = open("/w/dsv41/tests/corpus.txt").read().split()
N = int(os.environ.get("NTOK", "200"))
out = gen.generate(prompt=" ".join(words[30000:30300]), max_new_tokens=N, add_bos=True)
log = moe.PICKLOG
per_layer = {}
for li, ids in log:
    per_layer.setdefault(li, []).append(set(int(e) for e in ids))
print(f"{len(log)} layer records, {len(per_layer)} layers, ~{len(log) // max(len(per_layer), 1)} tokens", flush=True)
for w in (1, 2, 3, 4, 5, 6):
    ratios = []
    for li, seq in per_layer.items():
        for i in range(0, len(seq) - w + 1):
            u = set().union(*seq[i:i + w])
            ratios.append(len(u) / 6.0)
    r = float(np.mean(ratios))
    print(f"block of {w} tokens: {r:.2f}x the experts of one token ({r / w * 100:.0f}% per token)", flush=True)
print(repr(out[-200:]), flush=True)
