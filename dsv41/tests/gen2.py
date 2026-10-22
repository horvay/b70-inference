"""Greedy generations + decode-only speed (graphs on/off via DSV_GRAPHS)."""
import os, time
os.environ.setdefault("DSV_NOSYNC", "1")
import torch
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "16")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok)
N = int(os.environ.get("NTOK", "64"))
for prompt in ["The capital of France is", "Write a short poem about the sea.\n", "def fibonacci(n):"]:
    ids = tok.encode(prompt, add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=N, min_new_tokens=N, stop_conditions=[])
    gen.enqueue(job)
    t0 = time.time(); tfirst = None; n = 0; text = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("text"):
                text += r["text"]
                n += 1
                if tfirst is None:
                    tfirst = time.time()
    t1 = time.time()
    print(f"=== {prompt!r}: first token {tfirst - t0:.2f} s, decode {(n - 1) / (t1 - tfirst):.2f} tok/s\n{text!r}", flush=True)
print("tier", tier.stats, flush=True)
if tier.stats.get("exact_picks"):
    print(f"exact picks served: {tier.stats['exact_served'] / tier.stats['exact_picks'] * 100:.1f}%", flush=True)
tier.save_profile()
