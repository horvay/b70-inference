"""Does decode get faster as one conversation goes on (expert caching)? One long greedy generation on one topic; per
window of W tokens: tok/s, NVMe / RAM / VRAM pieces per token, promotions.
  NTOK=400 WIN=50 DSV_BOOST=0.3 python3 tests/t_long.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
if getattr(model, "_dsv_draft", None) is not None:
    draft_, dcache_ = model._dsv_draft
    gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft_, draft_cache=dcache_)
else:
    gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Hello.", max_new_tokens=8, add_bos=True)          # JIT
N, W = int(os.environ.get("NTOK", "400")), int(os.environ.get("WIN", "50"))
words = open("/w/dsv41/tests/corpus.txt").read().split()
off = int(os.environ.get("CORPUS_OFF", "30000"))
prompt = " ".join(words[off:off + 300])
prompt_ids = tok.encode(prompt, add_bos=True)
job = Job(input_ids=prompt_ids, max_new_tokens=N, min_new_tokens=N, stop_conditions=[])
t_enq = time.time()
gen.enqueue(job)
cnt, t0, s0, text = 0, None, None, ""
nxt = 0
acc = rej = 0
last_cnt = 0
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("stage") != "streaming" or r.get("token_ids") is None:
            continue
        text += r.get("text", "")
        if "accepted_draft_tokens" in r:
            acc, rej = r["accepted_draft_tokens"], r.get("rejected_draft_tokens", 0)
        if t0 is None:
            t0, s0 = time.time(), dict(tier.stats)
            print(f"prompt {prompt_ids.shape[-1]} tokens: "
                  f"first token after {t0 - t_enq:.1f} s", flush=True)
            nxt = W
            continue
        cnt += r["token_ids"].numel()
        if cnt >= nxt:
            nxt += W
            wn = cnt - last_cnt
            last_cnt = cnt
            dt = time.time() - t0
            d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in tier.stats}
            print(f"tokens {cnt - W:4d}-{cnt:4d}: {wn / dt:5.2f} tok/s (drafts accepted {acc}/{acc + rej}) | per token: nvme {d.get('nvme', 0) / wn:5.1f}, "
                  f"ram {d.get('ram', 0) / wn:5.1f}, vram {d.get('vram', 0) / wn:5.1f} pieces, promoted "
                  f"{d.get('promoted', 0) / W:4.1f}, NVMe waits {(d.get('t_nvme', 0) + d.get('t_wait_ahead', 0)) / W * 1e3:5.1f} ms",
                  flush=True)
            t0, s0 = time.time(), dict(tier.stats)
print("PROMPT END:", repr(prompt[-200:]), flush=True)
print("TEXT START:", repr(text[:300]), flush=True)
print("TEXT END:", repr(text[-600:]), flush=True)
ws = text.split()
grams = [" ".join(ws[i:i + 8]) for i in range(len(ws) - 8)]
print(f"distinct 8-word sequences: {len(set(grams))} of {len(grams)} (a loop repeats them)", flush=True)
