#!/usr/bin/env python3
"""Group the XPU kernels of a torch-profiler trace (scripts/prof_live.sh, activities CPU + XPU) by step and category.

    prof_cats.py <trace.json.gz> [decode|prefill] [top N]

Steps are the CPU "step[DECODE ...]" / "step[EXTEND ...]" annotations; each kernel is assigned to the step whose CPU
span launched it (by correlation id when present, else by device start time inside the step's span). Prints per step
kind: steps, kernels per step, device time per step, wall time per step, and the categories and top kernels by time."""
import gzip
import json
import re
import sys
from collections import defaultdict

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
ev = json.load(gzip.open(path))["traceEvents"]
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
steps = [e for e in ev if e.get("ph") == "X" and str(e.get("name", "")).startswith("step[")]
print(f"{len(kern)} device ops, {len(steps)} steps, categories present: "
      f"{sorted({e.get('cat') for e in ev if e.get('ph') == 'X'} - {None})}")
if not kern:
    sys.exit("no device kernel events in this trace (was the XPU activity requested?)")

CATS = [  # (category, regex over the kernel name), first match wins
    ("moe_exl3", r"exl3.*moe|moe.*exl3|grouped|expert|_moe_|moe_"),
    ("exl3_linear", r"exl3|trellis|gemv|gemm_exl|had|dequant"),
    ("gdn/linear_attn", r"gdn|gated_delta|delta_rule|chunk_|recurrent|fla|conv1d|causal_conv"),
    ("attention/qsa", r"attn|attention|sdpa|flash|softmax_|qsa|topk|indexer|fmha|rope|rotary"),
    ("norm", r"norm"),
    ("gemm(onednn/mkl)", r"gemm|matmul|onednn|dnnl|mkl|xetla|xe_"),
    ("copy/memset", r"memcpy|memset|copy|Memcpy|Memset"),
    ("elementwise/torch", r"elementwise|vectorized|reduce|index|scatter|gather|cat|fill|unrolled|launch_"),
    ("triton", r"triton|_kernel$"),
]


def cat_of(name):
    n = name.lower()
    for c, rx in CATS:
        if re.search(rx, n):
            return c
    return "other"


steps.sort(key=lambda e: e["ts"])
kinds = defaultdict(list)
for s in steps:
    k = "decode" if "DECODE" in s["name"] else ("extend" if "EXTEND" in s["name"] else s["name"][:20])
    kinds[k].append(s)
# assign kernels to steps by correlation: torch kineto links runtime launch (cpu) -> kernel via "correlation" args
corr_ts = {}
for e in ev:
    if e.get("ph") == "X" and e.get("cat") in ("xpu_runtime", "xpu_driver", "cuda_runtime") and "correlation" in e.get("args", {}):
        corr_ts[e["args"]["correlation"]] = e["ts"]
span = [(s["ts"], s["ts"] + s["dur"], s) for s in steps]


def step_of(k):
    c = k.get("args", {}).get("correlation")
    t = corr_ts.get(c, k["ts"])
    for a, b, s in span:
        if a <= t <= b:
            return s
    return None


per = defaultdict(list)
for k in kern:
    s = step_of(k)
    if s is not None:
        per[id(s)].append(k)
for kind, ss in kinds.items():
    if want and not kind.startswith(want[:6]):
        continue
    ks = [k for s in ss for k in per[id(s)]]
    if not ks:
        print(f"\n== {kind}: {len(ss)} steps, no kernels attributed (graph replay?)")
        continue
    dev = sum(k["dur"] for k in ks)
    wall = sum(s["dur"] for s in ss)
    print(f"\n== {kind}: {len(ss)} steps, {len(ks) / len(ss):.0f} device ops/step, device {dev / len(ss) / 1e3:.2f} ms/step, "
          f"CPU step span {wall / len(ss) / 1e3:.2f} ms/step")
    bc = defaultdict(lambda: [0.0, 0])
    bn = defaultdict(lambda: [0.0, 0])
    for k in ks:
        bc[cat_of(k["name"])][0] += k["dur"]; bc[cat_of(k["name"])][1] += 1
        bn[k["name"]][0] += k["dur"]; bn[k["name"]][1] += 1
    for c, (d, n) in sorted(bc.items(), key=lambda x: -x[1][0]):
        print(f"  {c:20s} {d / len(ss) / 1e3:8.3f} ms/step {100 * d / dev:5.1f}%  {n / len(ss):6.0f} ops/step  "
              f"{d / max(n, 1):7.1f} us/op")
    print("  top kernels:")
    for nme, (d, n) in sorted(bn.items(), key=lambda x: -x[1][0])[:top]:
        print(f"   {d / len(ss) / 1e3:7.3f} ms/step {n / len(ss):5.0f}x {d / n:7.1f}us  [{cat_of(nme)}] {nme[:100]}")

# ---- per-module attribution (server started with EXL3_PROF_SCOPES=1): innermost "M:..." range around the launch
scopes = [e for e in ev if e.get("ph") == "X" and str(e.get("name", "")).startswith("M:")]
if scopes:
    import bisect
    scopes.sort(key=lambda e: e["ts"])
    starts = [e["ts"] for e in scopes]

    def scope_of(t):
        i = bisect.bisect_right(starts, t) - 1
        best = None
        # walk back over ranges that started before t; the innermost containing one has the latest start
        j = i
        while j >= 0 and j > i - 64:
            e = scopes[j]
            if e["ts"] <= t <= e["ts"] + e["dur"]:
                best = e
                break
            j -= 1
        return best["name"] if best else "(no module)"

    for kind, ss in kinds.items():
        if want and not kind.startswith(want[:6]):
            continue
        ks = [k for s in ss for k in per[id(s)]]
        if not ks:
            continue
        bm = defaultdict(lambda: [0.0, 0])
        for k in ks:
            nm = scope_of(corr_ts.get(k.get("args", {}).get("correlation"), k["ts"]))
            nm = re.sub(r"^M:", "", nm)
            bm[nm][0] += k["dur"]; bm[nm][1] += 1
        dev = sum(v[0] for v in bm.values())
        print(f"\n== {kind} by module ({len(ss)} steps): device {dev / len(ss) / 1e3:.2f} ms/step")
        for nm, (d, n) in sorted(bm.items(), key=lambda x: -x[1][0])[:top]:
            print(f"  {d / len(ss) / 1e3:7.3f} ms/step {100 * d / dev:5.1f}% {n / len(ss):6.0f} ops/step  {nm[:90]}")
