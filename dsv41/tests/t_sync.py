"""Sync-point kernel alone: does the GPU see host writes to pinned memory, how fast does it poll, does the host see
the GPU's ready flag."""
import os, sys, time, ctypes, threading
import torch
sys.path.insert(0, "/w/dsv41/xpu")
from dsv_xpu import moe
G = moe.G()
dev = torch.device("xpu", 0)
L = 4
flags = torch.zeros(2 * L + 8, dtype=torch.int64).pin_memory(); f = flags.numpy()
hp = torch.zeros((L, 16), dtype=torch.int64).pin_memory()
waits = torch.zeros(L, dtype=torch.int64).pin_memory(); w = waits.numpy()
picks = torch.arange(12, dtype=torch.int64, device=dev)
lib = ctypes.CDLL("/w/dsv41/xpu/lib/dsv_spin.so")
lib.wait_ge.restype = ctypes.c_int64
lib.wait_ge.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
base = flags.data_ptr()
ERR = 2 * L

# 1) poll rate: nobody sets done; max_it polls
for mx in (1000, 100000):
    torch.xpu.synchronize()
    t0 = time.time()
    G.sync_point(picks, hp[0], flags, 0, L, ERR, waits[0], 1, mx)
    torch.xpu.synchronize()
    dt = time.time() - t0
    print(f"poll test max_it={mx}: {dt * 1e3:.1f} ms -> {dt / mx * 1e9:.0f} ns/poll; ready={f[0]} err={f[ERR]} "
          f"waits={w[0]} picks ok={hp[0, :12].tolist() == list(range(12))}", flush=True)
    f[:] = 0

# 2) host releases after the GPU reports ready
def pager(step, li, delay):
    r = lib.wait_ge(base + 8 * li, step, int(5e9), 0)
    time.sleep(delay)
    f[L + li] = step
    print(f"  pager saw ready={r} for layer {li}", flush=True)

for delay in (0.0, 0.002):
    f[:] = 0
    th = threading.Thread(target=pager, args=(1, 1, delay)); th.start()
    torch.xpu.synchronize()
    t0 = time.time()
    G.sync_point(picks, hp[1], flags, 1, L + 1, ERR, waits[1], 1, 200000)
    torch.xpu.synchronize()
    th.join()
    print(f"release test delay {delay * 1e3:.0f} ms: {(time.time() - t0) * 1e3:.1f} ms, err={f[ERR]}, polls={w[1]}", flush=True)
