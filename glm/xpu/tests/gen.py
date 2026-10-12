"""Greedy generation through exllamav3's Generator on the B70: decode speed and tier behaviour.

  in_rootfs.sh --gpu python3 /w/xpu/tests/gen.py [--tokens 200] [--ram-gb 20]
"""
import argparse
import json
import time

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--tokens", type=int, default=200)
ap.add_argument("--ram-gb", type=float, default=20.0)
ap.add_argument("--cache", type=int, default=16384)
ap.add_argument("--warm", default="/w/xpu/data/route_stats_glm53.json")
a = ap.parse_args()

meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
order = []
if a.warm:
    stats = json.load(open(a.warm))
    order = sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of for e, sc in enumerate(v)), reverse=True)
    order = [(li, e) for _, li, e in order]

model, config, cache, tok, tier = load("/models", a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order)

from exllamav3 import Generator
from exllamav3.generator.sampler import ArgmaxSampler

gen = Generator(model=model, cache=cache, tokenizer=tok)
prompts = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "Explain how a refrigerator works, step by step."]
for p in prompts:
    text = model.default_chat_prompt(p)
    s0 = dict(tier.stats)
    t0 = time.time()
    out = gen.generate(prompt=text, max_new_tokens=a.tokens, sampler=ArgmaxSampler(), completion_only=True,
                       stop_conditions=config.eos_token_id_list if hasattr(config, "eos_token_id_list") else None)
    dt = time.time() - t0
    n = len(tok.encode(out)[0]) if isinstance(tok.encode(out), torch.Tensor) else a.tokens
    d = {k: (tier.stats[k] - s0[k]) for k in s0}
    log(f"{n} tokens in {dt:.1f} s ({n / dt:.2f} tok/s incl. prefill); tier delta {d}")
    log("OUTPUT: " + out[:600].replace("\n", " | "))
