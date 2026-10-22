"""Decode benchmark with a per-token time split. Loads once, warms, then for each case runs NTOK greedy tokens on a
few prompts and prints decode tok/s (first token excluded) and, per token: NVMe pieces read on demand, waits for
prefetched reads, host time blocked on the GPU (layer readbacks), promotions.
  CASES="exact,nosync" NTOK=48 python3 tests/dbench.py"""
import os, time
os.environ.setdefault("DSV_SAVE_PROFILE", "0")     # tests do not move the routing profile
import torch
from dsv_xpu.runner import load
from dsv_xpu import moe
model, config, cache, tok, tier = load("/models", "/nvx", cache_tokens=8192, max_batch_size=1,
                                       ram_expert_gb=float(os.environ.get("RAM_GB", "24")))
tier.warm(tier.load_profile())
from exllamav3 import Generator, Job
gen = Generator(model=model, cache=cache, tokenizer=tok)
gen.generate(prompt="Once upon a time, in a small village by the sea,", max_new_tokens=24, add_bos=True)
N = int(os.environ.get("NTOK", "48"))
PROMPTS = ["The history of the Roman Empire", "Write a short poem about rivers and lakes.\n",
           "def quicksort(arr):", "The old lighthouse keeper looked at the horizon and"]


def run(prompt):
    j = Job(input_ids=tok.encode(prompt, add_bos=True), max_new_tokens=N, min_new_tokens=N, stop_conditions=[])
    gen.enqueue(j)
    t_first, cnt, text = None, 0, ""
    s0 = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                text += r.get("text", "")
                if t_first is None:
                    t_first = time.time()
                    s0 = dict(tier.stats)
                else:
                    cnt += r["token_ids"].numel()
    dt = time.time() - t_first
    d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in tier.stats}
    return cnt, dt, d, text


PAGER = moe.PAGER
for case in os.environ.get("CASES", "exact").split(","):
    # "sync..." cases: the per-layer host-wait path (no pager thread); others use the pager when it is enabled
    moe.PAGER = None if case.startswith("sync") else PAGER
    if case.startswith("sync"):
        case = case[4:].lstrip("_") or "exact"
    from dsv_xpu import moe as _m
    depth_saved = _m.AHEAD_DEPTH[0]
    if case.startswith("d1_"):
        _m.AHEAD_DEPTH[0] = 1
        case = case[3:]
    stage_saved = getattr(tier, "stage_on", False)
    promo_saved = getattr(tier, "promo_on", False)
    if case.startswith("promo"):
        tier.promo_on = True
        case = case[5:].lstrip("_") or "exact"
    if case.startswith("nostage"):
        tier.stage_on = False
        case = case[7:].lstrip("_") or "exact"
    tier.nosync = case == "nosync"
    tier.set_boost(float(case[5:]) if case.startswith("boost") else 0.0)
    if "budget" in case:
        tier.promote_budget = int(case.split("budget")[1])
    tot_n, tot_t, agg = 0, 0.0, {}
    ps0 = dict(moe.PAGER.stats) if moe.PAGER is not None else None
    if ps0 is not None:
        ps0.setdefault("fast", 0)
    for p in PROMPTS[:int(os.environ.get("NPROMPTS", "3"))]:
        n, dt, d, text = run(p)
        tot_n += n; tot_t += dt
        for k, v in d.items():
            agg[k] = agg.get(k, 0) + v
        print(f"  [{case}] {n / dt:.2f} tok/s  {text[:70]!r}", flush=True)
    tier.stage_on = stage_saved
    _m.AHEAD_DEPTH[0] = depth_saved
    tier.promo_on = promo_saved
    pt = lambda k: agg.get(k, 0) / tot_n
    if ps0 is not None:
        P = moe.PAGER
        P.wait_idle()
        d = {k: P.stats[k] - ps0[k] for k in ps0}
        nl = max(d["layers"], 1)
        print(f"   pager ({'on' if moe.PAGER else 'off'}): per layer host {d['t_host'] / nl * 1e3:.2f} ms, waiting for the GPU "
              f"{d['t_ready'] / nl * 1e3:.2f} ms, GPU spin polls {d['spin'] / nl:.0f}, released by the C fast path "
              f"{d.get('fast', 0) / nl * 100:.0f}%", flush=True)
    print(f"CASE {case}{'' if moe.PAGER else ' (sync)'}: {tot_n / tot_t:.2f} tok/s ({tot_t / tot_n * 1000:.0f} ms/token); per token: NVMe sync "
          f"{pt('t_nvme') * 1000:.1f} ms, ahead waits {pt('t_wait_ahead') * 1000:.1f} ms, GPU readback waits "
          f"{pt('t_sync') * 1000:.1f} ms; pieces: nvme {pt('nvme'):.1f}, ram {pt('ram'):.1f}, vram {pt('vram'):.1f}, "
          f"promoted {pt('promoted'):.1f}"
          + (f"; boost changed {agg['boost_changed'] / max(agg['boost_picks'], 1) * 100:.1f}% of picks"
             if agg.get("boost_picks") else "")
          + "".join(f"; {d}-ahead prediction {agg[f'pred_hit{d}'] / agg[f'pred_n{d}'] * 100:.0f}% right"
                    for d in (1, 2, 3) if agg.get(f"pred_n{d}"))
          + (f"; staged {pt('staged'):.0f} pieces/token, used {pt('stage_used'):.0f}" if agg.get("staged") else "")
          + (f"; promotions {pt('promoted'):.1f}/token" if agg.get("promoted") else ""),
          flush=True)
