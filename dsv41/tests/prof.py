"""Decode-step time breakdown: times attention, MoE, hyper-connection/norm and head with a device sync around each
(the syncs add a little; the split is what matters)."""
import os, time, collections
import torch
from dsv_xpu.runner import load
from dsv_xpu import moe

model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "16")))
tier.warm(tier.load_profile())
from exllamav3 import Generator
gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Once upon a time", max_new_tokens=8, add_bos=True)

acc = collections.defaultdict(float)
cnt = collections.defaultdict(int)
on = [False]


def wrap(obj, name, tag):
    f = getattr(obj, name)

    def w(*a, **k):
        if not on[0]:
            return f(*a, **k)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        r = f(*a, **k)
        torch.xpu.synchronize()
        acc[tag] += time.perf_counter() - t0
        cnt[tag] += 1
        return r
    setattr(obj, name, w)


from exllamav3.modules.dsv41_hc import DSV41TransformerBlock
for m in model.modules:
    if isinstance(m, DSV41TransformerBlock):
        wrap(m.attn, "forward", "attn")
        wrap(m.mlp, "forward", "moe")
        wrap(m.attn_hc, "mix_full", "hc_mix")
        wrap(m.mlp_hc, "mix_full", "hc_mix")
        wrap(m.attn_norm, "forward", "norm")
        wrap(m.mlp_norm, "forward", "norm")
        if m.mlp.shared_experts is not None:
            wrap(m.mlp.shared_experts, "forward", "  (shared expert, inside moe)")
    elif type(m).__name__ in ("Linear", "RMSNorm", "DSV41HyperHead", "Embedding", "EngramLayer", "ExpandStreams"):
        wrap(m, "forward", type(m).__name__)
wrap(moe, "moe_decode_fused", "  (routed experts, inside moe)")
if os.environ.get("PROF_ATTN") == "1":
    from exllamav3.modules import dsv41 as _d41, dsv4 as _d4
    from exllamav3.modules.dsv41 import DSV41Attention
    from exllamav3.modules.dsv4 import DSV4Attention
    for cls in (DSV4Attention,):
        for nm in ("_project_qkv", "_project_o_grouped"):
            wrap(cls, nm, f"  attn.{nm}")
    for nm in ("_compress_into_pool", "_v41_indexer", "_gather_pool_idx"):
        wrap(DSV41Attention, nm, f"  attn.{nm}")
    wrap(_d41, "dsa_attn", "  attn.dsa_attn (v41)")
    wrap(_d4, "dsa_attn", "  attn.dsa_attn (sliding)")
wrap(moe, "route", "  (router, inside moe)")

N = int(os.environ.get("NTOK", "32"))
for it in range(2):
    acc.clear(); cnt.clear()
    on[0] = it == 1
    torch.xpu.synchronize()
    t0 = time.time()
    gen.generate(prompt="The history of the Roman Empire", max_new_tokens=N, add_bos=True)
    dt = time.time() - t0
    print(f"run {it}: {N} tokens in {dt:.2f} s = {N / dt:.2f} tok/s (incl. prompt)", flush=True)
# CPU-side cost: time to enqueue a decode forward vs time until the GPU finishes it
on[0] = False
fw = model.forward
cpu_t, all_t = [], []
def fwd(*a, **k):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    r = fw(*a, **k)
    t1 = time.perf_counter()
    torch.xpu.synchronize()
    t2 = time.perf_counter()
    cpu_t.append(t1 - t0); all_t.append(t2 - t0)
    return r
model.forward = fwd
gen.generate(prompt="A short note about rivers", max_new_tokens=16, add_bos=True)
model.forward = fw
import statistics
print(f"decode forward: enqueue (CPU) {statistics.median(cpu_t) * 1000:.1f} ms, complete {statistics.median(all_t) * 1000:.1f} ms (median of {len(cpu_t)})", flush=True)
tot = sum(v for k, v in acc.items() if not k.startswith(" "))
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {k:34s} {v * 1000 / N:8.2f} ms/token  ({cnt[k] / N:.0f} calls/token)")
print(f"  {'sum of top-level':34s} {tot * 1000 / N:8.2f} ms/token")
print("tier", tier.stats)
