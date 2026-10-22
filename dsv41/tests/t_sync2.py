"""Sync-point stress test with the pager's flag layout (one 512-byte slot per flag): N steps x L layers, the host
thread answers each layer as soon as it sees ready (like the pager), the main thread enqueues everything up front."""
import sys, time, ctypes, threading
import torch
sys.setswitchinterval(0.0002)
sys.path.insert(0, "/w/dsv41/xpu")
from dsv_xpu import moe
G = moe.G()
dev = torch.device("xpu", 0)
import os, faulthandler
faulthandler.dump_traceback_later(15, exit=True)
S, L, N = 64, int(os.environ.get("TL", "40")), int(os.environ.get("TN", "50"))
flags = torch.zeros((2 * L + 4) * S, dtype=torch.int64).pin_memory(); f = flags.numpy()
hp = torch.zeros((L, 144), dtype=torch.int64).pin_memory(); h = hp.numpy()
waits = torch.zeros(L * S, dtype=torch.int64).pin_memory(); w = waits.numpy()
ERR = 2 * L * S
lib = ctypes.CDLL("/w/dsv41/xpu/lib/dsv_spin.so")
lib.wait_ge.restype = ctypes.c_int64
lib.wait_ge.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
base = flags.data_ptr()
bad = [0]
polls = []


def host():
    for step in range(1, N + 1):
        for li in range(L):
            r = lib.wait_ge(base + 8 * li * S, step, int(5e9), 0)
            if r != step or h[li, 0] != step * 1000 + li:
                bad[0] += 1
            f[(L + li) * S] = step
            if os.environ.get("TV"): print(f"host: step {step} layer {li} ready={r} -> done", flush=True)
            polls.append(int(w[li * S]))


th = threading.Thread(target=host)
th.start()
x = torch.zeros(4096, device=dev)
t0 = time.time()
for step in range(1, N + 1):
    for li in range(L):
        x.add_(1.0)                                            # some GPU work between sync points
        p = torch.full((18,), step * 1000 + li, dtype=torch.int64, device=dev)
        G.sync_point(p, hp[li], flags, li * S, (L + li) * S, ERR, waits[li * S], step, 2000000)
        if os.environ.get("TV"): print(f"main: enqueued step {step} layer {li}", flush=True)
t_enq = time.time() - t0
th.join()                       # GIL released while waiting: the host thread can answer every sync point
torch.xpu.synchronize()
dt = time.time() - t0
polls.sort()
print(f"enqueue {t_enq:.2f} s; {N} steps x {L} layers in {dt:.2f} s ({dt / N / L * 1e6:.0f} us per sync point incl. enqueue); err={f[ERR]}, "
      f"bad picks/ready={bad[0]}; GPU polls median {polls[len(polls) // 2]}, p99 {polls[int(len(polls) * .99)]}, "
      f"max {polls[-1]}", flush=True)
