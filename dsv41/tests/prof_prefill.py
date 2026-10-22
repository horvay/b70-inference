"""Prompt-path breakdown with synchronized timers around attention / MoE / other per block (second run, post-JIT)."""
import os, time, collections
import torch
from dsv_xpu.runner import load
P = int(os.environ.get("PLEN", "4096"))
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=-(-(2 * P + 1024) // 256) * 256, max_batch_size=1,
                                       ram_expert_gb=16, vram_reserve_gb=6)
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
from exllamav3.modules.dsv41_hc import DSV41TransformerBlock
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=8192)
words = open("/w/dsv41/tests/corpus.txt").read().split()
acc = collections.defaultdict(float); on = [False]
def wrap(obj, name, tag):
    f = getattr(obj, name)
    def w(*a, **k):
        if not on[0]:
            return f(*a, **k)
        torch.xpu.synchronize(); t0 = time.perf_counter()
        r = f(*a, **k)
        torch.xpu.synchronize(); acc[tag] += time.perf_counter() - t0
        return r
    setattr(obj, name, w)
for m in model.modules:
    if isinstance(m, DSV41TransformerBlock):
        wrap(m.attn, "forward", "attention"); wrap(m.mlp, "forward", "moe (incl. shared)")
        wrap(m.attn_hc, "apply_", "hc apply"); wrap(m.mlp_hc, "apply_", "hc apply")
        wrap(m.attn_hc, "mix_full", "hc mix"); wrap(m.mlp_hc, "mix_full", "hc mix")
    elif type(m).__name__ == "EngramLayer":
        wrap(m, "forward", "engram")
from exllamav3.modules import dsv41 as _d41, dsv4 as _d4
from exllamav3.modules.dsv41 import DSV41Attention
from exllamav3.modules.dsv4 import DSV4Attention
for nm in ("_project_qkv", "_project_o_grouped"):
    wrap(DSV4Attention, nm, f"  attn.{nm}")
for nm in ("_compress_into_pool", "_v41_indexer", "_gather_pool_idx"):
    wrap(DSV41Attention, nm, f"  attn.{nm}")
wrap(_d41, "dsa_attn", "  attn.dsa_attn (v41)")
wrap(_d4, "dsa_attn", "  attn.dsa_attn (sliding)")
def run(off):
    ids = tok.encode(" ".join(words[off:off + P]), add_bos=True)[:, :P]
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, stop_conditions=[]))
    while gen.num_remaining_jobs():
        gen.iterate()
run(3000)
on[0] = True
t0 = time.time(); run(30000); dt = time.time() - t0
print(f"BREAKDOWN {P} tokens in {dt:.1f} s", flush=True)
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"BREAKDOWN   {k:22s} {v:7.1f} s", flush=True)
