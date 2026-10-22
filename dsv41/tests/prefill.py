"""Prompt-processing timing: a long prompt twice (the first pays the JIT), first-token time."""
import os, time
import torch
from dsv_xpu.runner import load
P = int(os.environ.get("PLEN", "4096"))
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=-(-(2 * P + 1024) // 256) * 256, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "16")),
                                       vram_reserve_gb=float(os.environ.get("VRAM_RESERVE_GB", "6")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=int(os.environ.get("CHUNK", "8192")))
words = open("/w/dsv41/tests/corpus.txt").read().split()
for rep, off in enumerate((3000, 30000)):
    ids = tok.encode(" ".join(words[off:off + P]), add_bos=True)[:, :P]
    t0 = time.time()
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, stop_conditions=[]))
    while gen.num_remaining_jobs():
        gen.iterate()
    dt = time.time() - t0
    print(f"PREFILL run {rep}: {P} tokens -> first token in {dt:.1f} s ({P / dt:.0f} tok/s)", flush=True)
