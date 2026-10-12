#!/usr/bin/env python3
"""Summarise a torch-profiler chrome trace (.trace.json.gz) from prof_run.sh: device kernel time by name, the busy
share of the device over the traced window, and the CPU ops with the most self time.

    prof_top.py <trace.json.gz> [top N]
"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 30
ev = json.load(gzip.open(path))["traceEvents"]
kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset", "xpu_driver")]
cpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
by = defaultdict(lambda: [0.0, 0])
for e in kern:
    by[e["name"]][0] += e["dur"]
    by[e["name"]][1] += 1
tot = sum(v[0] for v in by.values())
if kern:
    t0 = min(e["ts"] for e in kern); t1 = max(e["ts"] + e["dur"] for e in kern)
    # union of kernel intervals: how much of the window the device was busy
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in kern)
    busy, cs, ce = 0.0, iv[0][0], iv[0][1]
    for s, f in iv[1:]:
        if s > ce:
            busy += ce - cs; cs, ce = s, f
        else:
            ce = max(ce, f)
    busy += ce - cs
    print(f"device: {len(kern)} kernels, {tot / 1e3:.1f} ms summed, window {(t1 - t0) / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms "
          f"({100 * busy / (t1 - t0):.0f}%)")
for name, (d, n) in sorted(by.items(), key=lambda x: -x[1][0])[:top]:
    print(f"{d / 1e3:9.1f} ms {100 * d / tot:5.1f}% x{n:<6} {name[:110]}")
cb = defaultdict(lambda: [0.0, 0])
for e in cpu:
    cb[e["name"]][0] += e["dur"]
    cb[e["name"]][1] += 1
print("\nCPU ops by total (inclusive) time:")
for name, (d, n) in sorted(cb.items(), key=lambda x: -x[1][0])[:15]:
    print(f"{d / 1e3:9.1f} ms x{n:<6} {name[:110]}")
