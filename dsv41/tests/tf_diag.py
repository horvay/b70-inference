"""Which change moved quality? Teacher-forced NLL on the same texts with today's decode features switched off one at a
time (exact routing, no promotions: results do not depend on cache state), in one process.
  TF_TEXTS=2 TF_N=48 python3 tests/tf_diag.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load
from dsv_xpu import moe, block, attn
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "16")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
from exllamav3.generator.job import Job as JobCls
words = open("/w/dsv41/tests/corpus.txt").read().split()
N = int(os.environ.get("TF_N", "48"))
texts = [" ".join(words[o:o + 400]) for o in (1000, 20000, 41000)][:int(os.environ.get("TF_TEXTS", "2"))]
gen = Generator(model=model, cache=cache, tokenizer=tok)
rec = []
orig = JobCls.receive_logits


def recv(self, logits):
    i = self._tf_i
    self._tf_i += 1
    lp = torch.log_softmax(logits.float().view(-1)[:config.vocab_size], -1)
    t = self._tf_forced[i] if i < len(self._tf_forced) else 0
    rec.append(lp[t].item())
    nt, a, b, c = orig(self, logits)
    return torch.full_like(nt, t), a, b, c


JobCls.receive_logits = recv


def nll():
    rec.clear()
    for text in texts:
        ids = tok.encode(text, add_bos=True)
        prompt, cont = ids[:, :48], ids[0, 48:48 + N].tolist()
        job = Job(input_ids=prompt, max_new_tokens=len(cont), min_new_tokens=len(cont), stop_conditions=[])
        job._tf_forced, job._tf_i = cont, 0
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            gen.iterate()
    return -sum(rec) / len(rec)


def zc(on):
    moe.ZERO_COPY = on
    if not on:
        tier.dirty[:] = True
        tier.upload()


feats = {
    "attn_lean": (lambda v: setattr(attn, "ENABLED", v)),
    "route_fused": (lambda v: setattr(moe, "ROUTE_DECODE", v)),
    "hc_fuse": (lambda v: setattr(block, "FUSE_APPLY", v)),
    "shared_fast": (lambda v: setattr(moe, "SHARED_FAST", v)),
    "zero_copy": zc,
}
print(f"DIAG all new features on: NLL {nll():.3f}", flush=True)
for name, f in feats.items():
    f(False)
    print(f"DIAG without {name}: NLL {nll():.3f}", flush=True)
    f(True)
for f in feats.values():
    f(False)
print(f"DIAG all new features off: NLL {nll():.3f}", flush=True)
