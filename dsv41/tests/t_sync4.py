"""Production handoff stress test (sync_point mode 2 + spin_done, no GPU atomics on host memory): N steps x L layers on
fresh pinned buffers, the host thread answering after DELAY seconds; reports timeouts and handoff latency."""
import sys, time, ctypes, threading, os
import torch
sys.setswitchinterval(0.0002)
sys.path.insert(0, "/w/dsv41/xpu")
from dsv_xpu import moe
G, X = moe.G(), moe.X()
dev = torch.device("xpu", 0)
S, L, N, MAXIT = 64, 40, 10, 2000000
lib = ctypes.CDLL("/w/dsv41/xpu/lib/dsv_spin.so")
lib.wait_ge.restype = ctypes.c_int64
lib.wait_ge.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
x = torch.zeros(1 << 16, device=dev)
for delay in (0.0, 5e-5, 2e-3):
    flags = torch.zeros((2 * L + 4) * S, dtype=torch.int64).pin_memory(); f = flags.numpy()
    hp = torch.zeros((L, 240), dtype=torch.int64).pin_memory(); h = hp.numpy()
    waits = torch.zeros(L * S, dtype=torch.int64).pin_memory()
    step_dev = torch.zeros(1, dtype=torch.int64, device=dev)
    step_pin = torch.zeros(8, dtype=torch.int64).pin_memory()
    ERR = 2 * L * S
    base = flags.data_ptr()
    bad = [0]

    def host():
        for step in range(1, N + 1):
            for li in range(L):
                r = lib.wait_ge(base + 8 * li * S, step, int(5e9), 0)
                if r != step or h[li, 0] != step * 1000 + li:
                    bad[0] += 1
                if delay:
                    t = time.perf_counter()
                    while time.perf_counter() - t < delay:
                        pass
                f[(L + li) * S] = step

    th = threading.Thread(target=host); th.start()
    t0 = time.time()
    for step in range(1, N + 1):
        sp = step_pin[step % 8:step % 8 + 1]
        sp.fill_(step)
        step_dev.copy_(sp, non_blocking=True)
        for li in range(L):
            x.mul_(1.0001)
            p = torch.full((18,), step * 1000 + li, dtype=torch.int64, device=dev)
            G.sync_point(p, hp[li], flags, li * S, (L + li) * S, ERR, waits[li * S], step_dev, MAXIT, 2)
            X.spin_done(flags, (L + li) * S, ERR, waits[li * S], step_dev, MAXIT, p)
    th.join()
    torch.xpu.synchronize()
    dt = time.time() - t0
    print(f"delay {delay * 1e6:6.0f} us: {N * L} handoffs in {dt:.2f} s ({dt / N / L * 1e6:.0f} us each); "
          f"err flag {int(f[ERR])}, bad {bad[0]}", flush=True)
