"""DSpark speculative decoding vs plain decode on the B70 (one process, same prompts): decode tok/s per job and the
draft acceptance rate.
  NTOK=96 MODES=plain,draft DSV_BOOST=0.3 python3 tests/spec.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
os.environ.setdefault("DSV_DRAFT", "1")
import torch
from dsv_xpu.runner import load
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
if os.environ.get("DUMP_S"):
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ["DUMP_S"]), repeat=True)
from exllamav3 import Generator, Job
from exllamav3.generator.sampler import GreedySampler
draft, dcache = model._dsv_draft
PH = {}
PHN = {}
if os.environ.get("PHASES") == "1":
    import collections
    PH = collections.defaultdict(float)
    PHN = collections.defaultdict(int)

    def timed(obj, name, tag):
        f = getattr(obj, name)

        def w(*a, **k):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            r = f(*a, **k)
            torch.xpu.synchronize()
            PH[tag] += time.perf_counter() - t0
            PHN[tag] += 1
            return r
        setattr(obj, name, w)
    timed(draft, "forward", "draft forward")
    timed(draft, "sample_from_state", "draft sample (head, markov, confidence)")
    timed(draft, "update_kv_from_target", "drafter kv update")
    timed(model, "forward", "target forward (verify or plain)")


OUTS = {}


def make_gen(mode):          # one Generator per cache at a time
    if mode == "draft":
        return Generator(model=model, cache=cache, tokenizer=tok, draft_model=draft, draft_cache=dcache)
    return Generator(model=model, cache=cache, tokenizer=tok)
N = int(os.environ.get("NTOK", "96"))
words = open("/w/dsv41/tests/corpus.txt").read().split()
PROMPTS = ["Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "def merge_sort(items):\n    \"\"\"Return a sorted copy of items.\"\"\"\n",
           "Explain how mRNA vaccines teach the immune system to recognize a virus.",
           "The history of the printing press begins"]


def run(gen, prompt):
    gen.enqueue(Job(input_ids=tok.encode(prompt, add_bos=True), max_new_tokens=N, min_new_tokens=N,
                    stop_conditions=[], sampler=GreedySampler()))
    first = last = None
    n = acc = rej = 0
    text = ""
    ids_out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "error":
                raise RuntimeError(str({k: v for k, v in r.items() if not torch.is_tensor(v) and k != "job"}))
            if r.get("stage") != "streaming":
                continue
            text += r.get("text", "")
            if "accepted_draft_tokens" in r:
                acc, rej = r["accepted_draft_tokens"], r.get("rejected_draft_tokens", 0)
            t = r.get("token_ids")
            if t is None or not t.numel():
                continue
            ids_out += t.view(-1).tolist()
            now = time.perf_counter()
            if first is None:
                first = last = now
                continue
            n += t.numel()
            last = now
    return n / max(last - first, 1e-6), acc, rej, text, ids_out


for mode in os.environ.get("MODES", "plain,draft").split(","):
    label = mode
    while label in OUTS:
        label += "'"
    if ":" in mode:                       # draft:<confidence threshold>[:<block boost>[:<lenient tau>]]
        parts = mode.split(":")
        mode = parts[0]
        draft.draft_conf_threshold = float(parts[1])
        tier.block_boost = float(parts[2]) if len(parts) > 2 else 0.0
        import exllamav3.generator.generator as _gm
        _gm.LENIENT_TAU[0] = float(parts[3]) if len(parts) > 3 else 1.0
    gen = make_gen(mode)
    gen.generate(prompt="Hello, this is a warm-up.", max_new_tokens=12, add_bos=True)      # JIT / first shapes
    from dsv_xpu import moe as _moe
    s0 = dict(tier.stats)
    p0 = dict(_moe.PAGER.stats) if _moe.PAGER is not None else {}
    tot_tps, tot_acc, tot_rej = [], 0, 0
    for i, p in enumerate(PROMPTS):
        tps, acc, rej, text, ids_ = run(gen, p)
        OUTS.setdefault(label, []).append((p, ids_))
        tot_tps.append(tps)
        tot_acc += acc
        tot_rej += rej
        a = f", acceptance {acc / max(acc + rej, 1) * 100:.0f}% ({acc}/{acc + rej})" if mode == "draft" else ""
        print(f"  [{mode}] prompt {i}: {tps:.2f} tok/s{a} | {text[:80]!r}", flush=True)
    a = f", acceptance {tot_acc / max(tot_acc + tot_rej, 1) * 100:.0f}%" if mode == "draft" else ""
    print(f"MODE {mode} (conf {draft.draft_conf_threshold}): mean {sum(tot_tps) / len(tot_tps):.2f} tok/s{a}", flush=True)
    d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in tier.stats}
    pd = {k: _moe.PAGER.stats.get(k, 0) - p0.get(k, 0) for k in p0} if p0 else {}
    calls = max(PHN.get("target forward (verify or plain)", 1), 1)
    nl = max(pd.get("layers", 1), 1)
    print(f"   per target forward: NVMe sync {d.get('t_nvme', 0) / calls * 1e3:.1f} ms, ahead waits "
          f"{d.get('t_wait_ahead', 0) / calls * 1e3:.1f} ms, nvme pieces {d.get('nvme', 0) / calls:.0f}, ram "
          f"{d.get('ram', 0) / calls:.0f}, vram {d.get('vram', 0) / calls:.0f}; pager per layer: host "
          f"{pd.get('t_host', 0) / nl * 1e3:.2f} ms, waiting for GPU {pd.get('t_ready', 0) / nl * 1e3:.2f} ms, "
          f"GPU spin polls {pd.get('spin', 0) / nl:.0f}", flush=True)
    for k in list(PH):
        print(f"   {k:42s} {PH[k]:7.2f} s total, {PH[k] / max(PHN[k], 1) * 1e3:7.1f} ms per call ({PHN[k]} calls)",
              flush=True)
        PH[k] = 0.0
        PHN[k] = 0


# Quality: mean NLL per token of each setting's generated continuations under the exact model (teacher-forced decode,
# plain exact routing, no boosts, no drafting)
if os.environ.get("SCORE", "1") == "1":
    from exllamav3.generator.job import Job as JobCls
    import exllamav3.generator.generator as _gm
    _gm.LENIENT_TAU[0] = 1.0
    tier.block_boost = 0.0
    saved_boost = tier.boost
    tier.set_boost(0.0)
    gen = make_gen("plain")
    orig_recv = JobCls.receive_logits
    lp_acc = []

    def recv(self, logits):
        i = self._tf_i
        self._tf_i += 1
        f = self._tf_forced
        t = f[i] if i < len(f) else 0
        if i < len(f):
            lp_acc.append(torch.log_softmax(logits.float().view(-1), -1)[t].item())
        nt, a, b, c = orig_recv(self, logits)
        return torch.full_like(nt, t), a, b, c
    JobCls.receive_logits = recv
    for label, outs in OUTS.items():
        lp_acc.clear()
        for p, ids_ in outs:
            if not ids_:
                continue
            job = Job(input_ids=tok.encode(p, add_bos=True), max_new_tokens=len(ids_), min_new_tokens=len(ids_),
                      stop_conditions=[], sampler=GreedySampler())
            job._tf_forced, job._tf_i = ids_, 0
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                gen.iterate()
        # repetition: a loop scores a low NLL too; distinct 8-token sequences per continuation, pooled
        g_all = g_dis = 0
        for p, ids_ in outs:
            gr = [tuple(ids_[i:i + 8]) for i in range(len(ids_) - 8)]
            g_all += len(gr)
            g_dis += len(set(gr))
        print(f"SCORE {label}: mean NLL {-sum(lp_acc) / max(len(lp_acc), 1):.3f} over {len(lp_acc)} tokens; distinct "
              f"8-token sequences {g_dis / max(g_all, 1) * 100:.0f}%", flush=True)
    JobCls.receive_logits = orig_recv
    tier.set_boost(saved_boost)
