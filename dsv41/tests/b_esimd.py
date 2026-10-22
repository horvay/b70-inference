import sys, time
import numpy as np
import torch
sys.path.insert(0, "/w/dsv41/exllamav3/tests")
from testlib import trellis as ref
from dsv_xpu import exl3_tri as T
torch.ops.load_library("/w/dsv41/xpu/lib/dsv_moe.so")
X = torch.ops.dsv_moe
s64 = lambda p: p - (1 << 64) if p >= (1 << 63) else p
dev = torch.device("xpu")
g = np.random.default_rng(0)

def blocked(x):   # [items, M, k] -> [items, k/16, M, 16] viewed as [items, M, k]
    a, M, k = x.shape
    return x.view(a, M, k // 16, 16).transpose(1, 2).contiguous().view(a, M, k)

def run(kdim, n, Ks, M, host=False, P=4, reps=50, check=True):
    trs = []
    for K in Ks:
        t = torch.from_numpy(ref.random_packed((kdim // 16) * (n // 16), K, g).reshape(kdim // 16, n // 16, 16 * K))
        trs.append(t.pin_memory() if host else t.to(dev))
    x = torch.randn(len(Ks), M, kdim, device=dev).half()
    xh = blocked(x)
    wptr = torch.tensor([s64(t.data_ptr()) for t in trs], dtype=torch.int64, device=dev)
    kk = torch.tensor(Ks, dtype=torch.int32, device=dev)
    ni = torch.tensor([len(Ks)], dtype=torch.int32, device=dev)
    part = torch.empty((P, len(Ks), M, n), dtype=torch.float, device=dev)
    mask = sum(1 << K for K in set(Ks))
    X.moe_gemv(xh, wptr, kk, ni, part, mask)
    torch.xpu.synchronize()
    if check:
        out = part.sum(0)
        err = 0
        for a, (K, t) in enumerate(zip(Ks, trs)):
            W = T.dequant(t.to(dev), K).float()
            want = x[a].float() @ W
            err = max(err, ((out[a] - want).abs().max() / want.abs().max()).item())
    t0 = time.time()
    for _ in range(reps):
        X.moe_gemv(xh, wptr, kk, ni, part, mask)
    torch.xpu.synchronize()
    dt = (time.time() - t0) / reps
    nbytes = sum(t.numel() * 2 for t in trs)
    return dt * 1e6, nbytes / dt / 1e9, err if check else -1

Ks = [1, 1, 2, 1, 2, 3, 1, 1, 2, 2, 1, 4]
for K in (1, 2, 3, 4):
    us, gbs, err = run(5120, 2304, [K] * 12, 1)
    print(f"gate/up 12 items all K={K}: {us:.0f} us, {gbs:.0f} GB/s, rel err {err:.2g}")
for P in (1, 2, 4, 8):
    us, gbs, err = run(5120, 2304, Ks, 1, P=P)
    print(f"gate/up mixed M=1 P={P}: {us:.0f} us, {gbs:.0f} GB/s, rel err {err:.2g}")
for M in (2, 3, 4):
    us, gbs, err = run(5120, 2304, Ks, M)
    print(f"gate/up mixed M={M}: {us:.0f} us, {gbs:.0f} GB/s, rel err {err:.2g}")
us, gbs, err = run(2304, 5120, Ks[:6], 1)
print(f"down 6 items M=1: {us:.0f} us, {gbs:.0f} GB/s, rel err {err:.2g}")
us, gbs, err = run(5120, 2304, Ks, 1, host=True, reps=10, check=False)
print(f"gate/up from pinned host: {us:.0f} us, {gbs:.1f} GB/s")
# per K: the same 12 GEMVs with the trellises in VRAM vs in pinned host RAM (read over PCIe), and the bytes each moves
for K in (1, 2, 3, 4):
    us_d, gbs_d, _ = run(5120, 2304, [K] * 12, 1, check=False)
    us_h, gbs_h, _ = run(5120, 2304, [K] * 12, 1, host=True, reps=10, check=False)
    print(f"PERK K={K}: VRAM {us_d:.0f} us ({gbs_d:.0f} GB/s), host {us_h:.0f} us ({gbs_h:.1f} GB/s)", flush=True)
