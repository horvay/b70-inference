"""Does a pinned -> VRAM copy on a side stream run while the main stream is busy (copy engine), or wait behind it?
Main stream: a spin kernel parked on a host flag (like a pager sync point) for ~30 ms. Side stream: a 24 MB H2D copy then
a host flag. The host watches when the copy's flag rises relative to releasing the main spin."""
import sys, time, os
import torch
sys.path.insert(0, "/w/dsv41/xpu")
from dsv_xpu import moe
X = moe.X()
dev = torch.device("xpu", 0)
S = 64
flags = torch.zeros(8 * S, dtype=torch.int64).pin_memory(); f = flags.numpy()
waits = torch.zeros(2 * S, dtype=torch.int64).pin_memory()
src = torch.empty(24 << 20, dtype=torch.uint8).pin_memory()
dst = torch.empty(24 << 20, dtype=torch.uint8, device=dev)
side = torch.xpu.Stream(dev)
one = torch.ones(1, dtype=torch.int64, device=dev)
# warm-up / JIT
dst.copy_(src, non_blocking=True)
f[S] = 1
X.spin_done(flags, S, 2 * S, waits[0:2], one, 1000, dst)
with torch.xpu.stream(side):
    X.set_host_flag(flags, 3 * S, 1, dst)
torch.xpu.synchronize()

for trial in range(3):
    f[:] = 0
    step = torch.full((1,), 2, dtype=torch.int64, device=dev)
    torch.xpu.synchronize()
    X.spin_done(flags, S, 2 * S, waits[0:2], step, 3000000, dst)       # main stream parked until f[S] >= 2
    t0 = time.perf_counter()
    with torch.xpu.stream(side):
        dst.copy_(src, non_blocking=True)
        X.set_host_flag(flags, 3 * S, 7, dst)
    t_copy = None
    while time.perf_counter() - t0 < 0.03:
        if f[3 * S] == 7 and t_copy is None:
            t_copy = time.perf_counter() - t0
    f[S] = 2                                                            # release the main stream
    t_rel = time.perf_counter() - t0
    while f[3 * S] != 7:
        pass
    t_end = time.perf_counter() - t0
    torch.xpu.synchronize()
    print(f"trial {trial}: copy flag {'at %.2f ms (while main busy)' % (t_copy * 1e3) if t_copy else 'NOT during main spin'}; "
          f"main released at {t_rel * 1e3:.1f} ms; copy flag at {t_end * 1e3:.2f} ms", flush=True)

# copy bandwidth alone
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    with torch.xpu.stream(side):
        dst.copy_(src, non_blocking=True)
torch.xpu.synchronize()
dt = time.perf_counter() - t0
print(f"side-stream H2D: {10 * 24 / 1024 / dt:.1f} GB/s", flush=True)
