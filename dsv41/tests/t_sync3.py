"""Sync-point handoff stress test per GPU polling method: 0 atomic load, 1 atomic RMW, 2 ESIMD uncached load.
The host thread answers each layer as soon as it sees ready (like the pager). Reports timeouts and poll counts."""
import os, sys, time, ctypes, threading, faulthandler
import torch
sys.setswitchinterval(0.0002)
sys.path.insert(0, "/w/dsv41/xpu")
from dsv_xpu import moe
faulthandler.dump_traceback_later(240, exit=True)
G, X = moe.G(), moe.X()
dev = torch.device("xpu", 0)
S, L, N, MAXIT = 64, 40, int(os.environ.get("TN", "10")), 200000
lib = ctypes.CDLL("/w/dsv41/xpu/lib/dsv_spin.so")
lib.wait_ge.restype = ctypes.c_int64
lib.wait_ge.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
x = torch.zeros(4096, device=dev)


def run(mode):
    flags = torch.zeros((2 * L + 4) * S, dtype=torch.int64).pin_memory(); f = flags.numpy()
    hp = torch.zeros((L, 144), dtype=torch.int64).pin_memory(); h = hp.numpy()
    waits = torch.zeros(L * S, dtype=torch.int64).pin_memory(); w = waits.numpy()
    ERR = 2 * L * S
    base = flags.data_ptr()
    bad, polls, errs = [0], [], [0]

    def host():
        for step in range(1, N + 1):
            for li in range(L):
                r = lib.wait_ge(base + 8 * li * S, step, int(5e9), 0)
                if r != step or h[li, 0] != step * 1000 + li:
                    bad[0] += 1
                f[(L + li) * S] = step

    th = threading.Thread(target=host); th.start()
    t0 = time.time()
    for step in range(1, N + 1):
        for li in range(L):
            x.add_(1.0)
            p = torch.full((18,), step * 1000 + li, dtype=torch.int64, device=dev)
            G.sync_point(p, hp[li], flags, li * S, (L + li) * S, ERR, waits[li * S], step, MAXIT, mode)
            if mode == 2:
                X.spin_done(flags, (L + li) * S, ERR, waits[li * S], step, MAXIT, x)
    th.join()
    torch.xpu.synchronize()
    dt = time.time() - t0
    print(f"mode {mode}: {N * L} sync points in {dt:.2f} s ({dt / N / L * 1e6:.0f} us each); err flag {f[ERR]}, "
          f"bad {bad[0]}, last polls {sorted(int(w[li * S]) for li in range(L))[-3:]}", flush=True)


for mode in [int(m) for m in os.environ.get("MODES", "2,1,0").split(",")]:
    run(mode)
