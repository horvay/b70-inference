import sys, time
import numpy as np
import torch
sys.path.insert(0, "/w/dsv41/exllamav3/tests")
from testlib import trellis as ref
from dsv_xpu import exl3_tri as T
s64 = lambda p: p - (1 << 64) if p >= (1 << 63) else p
dev = torch.device("xpu")
g = np.random.default_rng(0)

def bench(kdim, n, Ks, M, host=False, NT=4, KSPLIT=4, reps=50):
    trs = []
    for K in Ks:
        t = torch.from_numpy(ref.random_packed((kdim // 16) * (n // 16), K, g).reshape(kdim // 16, n // 16, 16 * K))
        trs.append(t.pin_memory() if host else t.to(dev))
    x = torch.randn(len(Ks), M, kdim, device=dev).half()
    wptr = torch.tensor([s64(t.data_ptr()) for t in trs], dtype=torch.int64, device=dev)
    kk = torch.tensor(Ks, dtype=torch.int32, device=dev)
    ni = torch.tensor([len(Ks)], dtype=torch.int32, device=dev)
    out = T.moe_gemv(x, wptr, kk, ni, n, len(Ks), NT=NT, KSPLIT=KSPLIT)
    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(reps):
        T.moe_gemv(x, wptr, kk, ni, n, len(Ks), NT=NT, KSPLIT=KSPLIT, out=out)
    torch.xpu.synchronize()
    dt = (time.time() - t0) / reps
    nbytes = sum(t.numel() * 2 for t in trs)
    return dt * 1e6, nbytes / dt / 1e9

Ks = [1, 1, 2, 1, 2, 3, 1, 1, 2, 2, 1, 4]       # 6 experts x (gate, up)
for NT, KS in ((2, 1), (4, 1), (4, 4), (8, 4), (4, 8), (2, 8)):
    us, gbs = bench(5120, 2304, Ks, 1, NT=NT, KSPLIT=KS)
    print(f"gate/up 12 items M=1 NT={NT} KSPLIT={KS}: {us:.0f} us, {gbs:.0f} GB/s")
for M in (2, 3):
    us, gbs = bench(5120, 2304, Ks, M, NT=4, KSPLIT=4)
    print(f"gate/up M={M}: {us:.0f} us, {gbs:.0f} GB/s")
us, gbs = bench(2304, 5120, Ks[:6], 1, NT=4, KSPLIT=4)
print(f"down 6 items M=1: {us:.0f} us, {gbs:.0f} GB/s")
us, gbs = bench(5120, 2304, Ks, 1, host=True, NT=4, KSPLIT=4, reps=10)
print(f"gate/up from pinned host: {us:.0f} us, {gbs:.1f} GB/s")
