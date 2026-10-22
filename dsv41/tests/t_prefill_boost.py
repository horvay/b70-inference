"""Residency boost for prompt routing (DSV_PREFILL_BOOST): prompt time on fresh texts, and quality = NLL of a
teacher-forced continuation decoded with exact routing after a boosted prompt (only the prompt's KV differs).
  PBS=0,0.3,1.0 PLEN=512 python3 tests/t_prefill_boost.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load
from dsv_xpu import moe
P = int(os.environ.get("PLEN", "512"))
NC = int(os.environ.get("NCONT", "64"))
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
tier.set_boost(0.0)                      # continuation decode: exact routing
from exllamav3 import Generator, Job
from exllamav3.generator.job import Job as JobCls
gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Hello.", max_new_tokens=4, add_bos=True)
words = open("/w/dsv41/tests/corpus.txt").read().split()
rec = []
orig = JobCls.receive_logits


def recv(self, logits):
    i = self._tf_i
    self._tf_i += 1
    if i == 0:
        self._t_first = time.time()
    f = self._tf_forced
    t = f[i] if i < len(f) else 0
    if i < len(f):
        rec.append(torch.log_softmax(logits.float().view(-1)[:config.vocab_size], -1)[t].item())
    nt, a, b, c = orig(self, logits)
    return torch.full_like(nt, t), a, b, c


JobCls.receive_logits = recv


def run(off, score):
    ids = tok.encode(" ".join(words[off:off + P + 200]), add_bos=True)
    prompt, cont = ids[:, :P], ids[0, P:P + NC].tolist()
    if not score:
        cont = cont[:1]
    job = Job(input_ids=prompt, max_new_tokens=len(cont), min_new_tokens=len(cont), stop_conditions=[])
    job._tf_forced, job._tf_i = cont, 0
    s0 = dict(tier.stats)
    t0 = time.time()
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        gen.iterate()
    return job._t_first - t0, tier.stats.get("nvme", 0) - s0.get("nvme", 0)


# Each setting gets its own fresh texts (an earlier prompt on the same text would have made its experts resident). Per
# text: the boosted prompt first (timed, scored), then the same text with an exact prompt (scored): the NLL difference
# is the boost's cost.
for j, pb in enumerate(float(x) for x in os.environ.get("PBS", "0,0.3,1.0").split(",")):
    tt, nb, ne = [], [], []
    for off in (1500 + 6000 * (3 * j + t) for t in range(3)):     # (the corpus is ~55K words)
        moe.PREFILL_BOOST[0] = pb
        rec.clear()
        tt.append(run(off, True))
        nb.append(-sum(rec) / len(rec))
        if pb > 0:
            moe.PREFILL_BOOST[0] = 0.0
            rec.clear()
            run(off, True)
            ne.append(-sum(rec) / len(rec))
    d = f"; NLL {sum(nb) / len(nb):.4f} vs exact prompt {sum(ne) / len(ne):.4f} (diff {(sum(nb) - sum(ne)) / len(ne):+.4f})" \
        if ne else f"; NLL {sum(nb) / len(nb):.4f}"
    print(f"PBOOST {pb}: prompt {sum(t for t, _ in tt) / len(tt):.1f} s (NVMe pieces {sum(n for _, n in tt) / len(tt):.0f})"
          + d, flush=True)
