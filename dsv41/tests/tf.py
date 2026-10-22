"""Teacher-forced decode: feed a fixed continuation token by token through the decode path and save each step's
log-probabilities (top 64) plus the forced token's log-prob. Compare modes with tests/tf_cmp.py.
  TF_OUT=/w/dsv41/run/tf_exact.pt DSV_NOSYNC=0 python3 tests/tf.py
"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
from exllamav3.generator.job import Job as JobCls
words = open("/w/dsv41/tests/corpus.txt").read().split()
N = int(os.environ.get("TF_N", "64"))
texts = [" ".join(words[o:o + 400]) for o in (1000, 20000, 41000)][:int(os.environ.get("TF_TEXTS", "3"))]
gen = Generator(model=model, cache=cache, tokenizer=tok)
rec = []
orig = JobCls.receive_logits


def recv(self, logits):
    forced = self._tf_forced
    i = self._tf_i
    self._tf_i += 1
    lp = torch.log_softmax(logits.float().view(-1)[:config.vocab_size], -1)
    t = forced[i] if i < len(forced) else 0
    top = torch.topk(lp, 64)
    rec.append(dict(text=self._tf_text, i=i, tok=t, lp_tok=lp[t].item(), top_v=top.values.cpu(), top_i=top.indices.cpu()))
    nt, a, b, c = orig(self, logits)
    nt = torch.full_like(nt, t)
    return nt, a, b, c


JobCls.receive_logits = recv
# TF_BOOSTS="0,0.1,0.2": one teacher-forced pass per residency boost in this process (outputs TF_OUT with _b<boost>)
boosts = [float(b) for b in os.environ["TF_BOOSTS"].split(",")] if os.environ.get("TF_BOOSTS") else [None]
for boost in boosts:
    if boost is not None:
        tier.set_boost(boost)
    rec.clear()
    for k in tier.stats:
        tier.stats[k] = 0.0 if isinstance(tier.stats[k], float) else 0
    t0 = time.time()
    for ti, text in enumerate(texts):
        ids = tok.encode(text, add_bos=True)
        prompt, cont = ids[:, :48], ids[0, 48:48 + N].tolist()
        job = Job(input_ids=prompt, max_new_tokens=len(cont), min_new_tokens=len(cont), stop_conditions=[])
        job._tf_forced, job._tf_i, job._tf_text = cont, 0, ti
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            gen.iterate()
    dt = time.time() - t0
    out = os.environ["TF_OUT"] if boost is None else os.environ["TF_OUT"].replace(".pt", f"_b{boost:g}.pt")
    torch.save(list(rec), out)
    nll = -sum(r["lp_tok"] for r in rec) / len(rec)
    print(f"TF {out}: {len(rec)} steps in {dt:.0f} s, mean NLL {nll:.3f} (ppl {torch.tensor(nll).exp():.2f})", flush=True)
    print("tier", tier.stats, flush=True)
    if tier.stats.get("boost_picks"):
        print(f"boost {tier.boost}: changed {tier.stats['boost_changed'] / tier.stats['boost_picks'] * 100:.1f}% of "
              f"picks; NVMe pieces {tier.stats['nvme']}, waits "
              f"{tier.stats['t_nvme'] + tier.stats.get('t_wait_ahead', 0):.1f} s", flush=True)
    if tier.stats.get("exact_picks"):
        print(f"exact picks served: {tier.stats['exact_served'] / tier.stats['exact_picks'] * 100:.1f}%", flush=True)
