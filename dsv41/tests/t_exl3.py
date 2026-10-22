"""Check the Triton EXL3 kernels against exllamav3's NumPy trellis reference (tests/testlib/trellis.py)."""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/w/dsv41/exllamav3/tests")
from testlib import trellis as ref
from dsv_xpu import exl3_tri as T
s64 = lambda p: p - (1 << 64) if p >= (1 << 63) else p

dev = torch.device("xpu")
g = np.random.default_rng(0)
for K in (1, 2, 3, 4, 8):
    tk, tn = 8, 24
    tr = torch.from_numpy(ref.random_packed(tk * tn, K, g).reshape(tk, tn, 16 * K))
    want = ref.dequant_rotated(tr, K, "mul1")
    got = T.dequant(tr.to(dev), K).cpu().numpy()
    print(f"dequant K={K}: max abs diff {np.abs(want.astype(np.float32) - got.astype(np.float32)).max():.3g}, "
          f"exact {np.mean(want == got) * 100:.2f}%")

# grouped GEMV: items with mixed K, rows M
for M in (1, 2, 3):
    kdim, n = 512, 768
    items = [1, 2, 3, 4, 1, 2]
    trs = [torch.from_numpy(ref.random_packed((kdim // 16) * (n // 16), K, g).reshape(kdim // 16, n // 16, 16 * K)).to(dev)
           for K in items]
    x = torch.randn(len(items), M, kdim, device=dev).half()
    wptr = torch.tensor([s64(t.data_ptr()) for t in trs], dtype=torch.int64, device=dev)
    kk = torch.tensor(items, dtype=torch.int32, device=dev)
    ni = torch.tensor([len(items)], dtype=torch.int32, device=dev)
    out = T.moe_gemv(x, wptr, kk, ni, n, len(items)).sum(0)
    err = 0
    for a, (K, t) in enumerate(zip(items, trs)):
        W = T.dequant(t, K).float()
        want = x[a].float() @ W
        err = max(err, ((out[a] - want).abs().max() / want.abs().max()).item())
    print(f"moe_gemv M={M}: max rel err {err:.3g}")
