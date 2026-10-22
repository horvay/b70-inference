"""Is a DSpark verify pass (several rows per forward) the same model as plain one-token decode? Greedy-decode a text
with plain steps (exact routing, no boosts), then decode it again through the DSpark generator with the drafter replaced
by that text (every draft is the true continuation), and compare the target's logits position by position.
  NTOK=40 DSV_BOOST=0 python3 tests/t_verify.py"""
import os
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
os.environ.setdefault("DSV_DRAFT", "1")
import torch
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
tier.set_boost(0.0)
tier.block_boost = 0.0
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import GreedySampler
import exllamav3.generator.generator as G
G.LENIENT_TAU[0] = 1.0
draft, dcache = model._dsv_draft
N = int(os.environ.get("NTOK", "40"))
PROMPT = os.environ.get("PROMPT", "Explain how mRNA vaccines teach the immune system to recognize a virus.")
rec = {}                  # position -> (input id, logits fp32 cpu)
fw = model.forward


def capture(input_ids, params=None, **kw):
    pos0 = int(params["cache_seqlens"].view(-1)[0])
    out = fw(input_ids, params, **kw)
    lg = out.float().view(-1, out.shape[-1]).cpu()
    ids = input_ids.view(-1).tolist()
    for i in range(lg.shape[0]):
        rec[pos0 + i] = (ids[i], lg[i])
    return out


model.forward = capture


def decode(gen):
    ids = []
    gen.enqueue(Job(input_ids=tok.encode(PROMPT, add_bos=True), max_new_tokens=N, min_new_tokens=N, stop_conditions=[],
                    sampler=GreedySampler()))
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                ids += r["token_ids"].view(-1).tolist()
    return ids


gen = Generator(model=model, cache=cache, tokenizer=tok)
decode(gen)                                   # warm-up (JIT, caches)
rec.clear()
plain_ids = decode(gen)
plain = dict(rec)
del gen
print("PLAIN:", repr(tok.decode(torch.tensor(plain_ids))[:200]), flush=True)

gen = Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache)
orig_mtp = gen.iterate_draftmodel_mtp_gen


def forced(results):
    real = orig_mtp(results)                  # keeps the drafter's own state moving; its proposals are replaced
    if real is None:
        return None
    job = next(j for j in gen.active_jobs if j.is_prefill_done())
    n = job.new_tokens
    w = real.shape[-1]
    nxt = (plain_ids[n:n + w] + [0] * w)[:w]
    real[0, :w] = torch.tensor(nxt, dtype=real.dtype)
    return real


gen.iterate_draftmodel_mtp_gen = forced
decode(gen)                                   # warm-up of the verify shapes
rec.clear()
spec_ids = decode(gen)
spec = dict(rec)
print("SPEC: ", repr(tok.decode(torch.tensor(spec_ids))[:200]), flush=True)
same = next((i for i, (a, b) in enumerate(zip(plain_ids, spec_ids)) if a != b), min(len(plain_ids), len(spec_ids)))
print(f"VERIFY first differing token: {same} of {min(len(plain_ids), len(spec_ids))}", flush=True)
rows = 0
for p in sorted(plain):
    if p not in spec or spec[p][0] != plain[p][0]:
        continue
    a, b = plain[p][1], spec[p][1]
    rows += 1
    d = (a - b).abs().max().item()
    top_a, top_b = a.argmax().item(), b.argmax().item()
    gap = (a.topk(2).values[0] - a.topk(2).values[1]).item()
    kl = torch.nn.functional.kl_div(b.log_softmax(-1), a.log_softmax(-1), log_target=True, reduction="sum").item()
    print(f"  pos {p}: max |dlogit| {d:.3f}, KL {kl:.4f}, argmax {'same' if top_a == top_b else 'DIFF'} "
          f"(plain top-2 gap {gap:.2f})", flush=True)
print(f"VERIFY compared {rows} positions", flush=True)
