import sys, numpy as np, torch
sys.path.insert(0, "/w/dsv41/exllamav3/tests")
from testlib import trellis as ref
from dsv_xpu import exl3_tri as T
s64 = lambda p: p - (1 << 64) if p >= (1 << 63) else p
dev = torch.device("xpu"); g = np.random.default_rng(0)
kd, n = 512, 768
Ks = [1, 2, 1]
trs = [torch.from_numpy(ref.random_packed((kd // 16) * (n // 16), K, g).reshape(kd // 16, n // 16, 16 * K)).to(dev) for K in Ks]
out = torch.zeros(4, kd, n, dtype=torch.half, device=dev)
for K in set(Ks):
    sel = [i for i, k in enumerate(Ks) if k == K]
    T.dequant_multi(torch.tensor([s64(trs[i].data_ptr()) for i in sel], dtype=torch.int64, device=dev),
                    torch.tensor(sel, dtype=torch.int32, device=dev), K, kd, n, out)
for i, (K, t) in enumerate(zip(Ks, trs)):
    want = T.dequant(t, K)
    print(i, K, (out[i].float() - want.float()).abs().max().item())
