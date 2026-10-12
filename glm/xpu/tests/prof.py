"""Per-module decode time breakdown on the B70 (synchronizing timers around module forwards).

  in_rootfs.sh --gpu python3 /w/xpu/tests/prof.py [--tokens 48]
"""
import argparse
import collections
import json
import time

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--tokens", type=int, default=48)
ap.add_argument("--ram-gb", type=float, default=20.0)
a = ap.parse_args()

meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
order = [(li, e) for _, li, e in sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of
                                         for e, sc in enumerate(v)), reverse=True)]
model, config, cache, tok, tier = load("/models", a.store, cache_size=8192, ram_gb=a.ram_gb, warm=order)

from exllamav3 import Generator
from exllamav3.generator.sampler import ArgmaxSampler
from exllamav3.modules import hyperconnections as hcm, rmsnorm, gated_delta_net, mla_attn, block_sparse_mlp, mlp, linear

T = collections.Counter()
N = collections.Counter()
ON = [False]


def wrap(cls, name, label):
    f = getattr(cls, name)

    def g(*args, **kw):
        if not ON[0]:
            return f(*args, **kw)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        r = f(*args, **kw)
        torch.xpu.synchronize()
        T[label] += time.perf_counter() - t0
        N[label] += 1
        return r
    setattr(cls, name, g)


wrap(hcm.HyperConnection, "mix", "hc_mix")
wrap(hcm.HyperConnection, "apply_", "hc_apply")
wrap(rmsnorm.RMSNorm, "forward", "rmsnorm")
wrap(gated_delta_net.GatedDeltaNet, "forward", "kda")
wrap(mla_attn.MLAttention, "forward", "mla")
wrap(block_sparse_mlp.BlockSparseMLP, "forward", "moe(total)")
wrap(mlp.GatedMLP, "forward", "gated_mlp(dense+shared)")
wrap(linear.Linear, "forward", "linear(all)")
wrap(type(tier), "ensure", "tier.ensure")
wrap(type(tier), "moe", "tier.moe")

gen = Generator(model=model, cache=cache, tokenizer=tok)
p = model.default_chat_prompt("Write a short story about a lighthouse keeper.")
gen.generate(prompt=p, max_new_tokens=16, sampler=ArgmaxSampler(), completion_only=True)   # JIT / warm
ON[0] = True
s0 = dict(tier.stats)
t0 = time.perf_counter()
gen.generate(prompt=p + " It was", max_new_tokens=a.tokens, sampler=ArgmaxSampler(), completion_only=True)
wall = time.perf_counter() - t0
ON[0] = False
log(f"wall {wall:.2f} s for {a.tokens} tokens ({a.tokens / wall:.2f} tok/s, timed/synced)")
for k, v in T.most_common():
    log(f"{k:28s} {v:8.2f} s  {1000 * v / a.tokens:8.1f} ms/token  calls {N[k]}")
log(f"tier delta { {k: round(tier.stats[k] - s0[k], 2) for k in s0} }")
