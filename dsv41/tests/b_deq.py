import sys, time, numpy as np, torch
sys.path.insert(0, "/w/dsv41/exllamav3/tests")
from testlib import trellis as ref
from dsv_xpu import exl3_tri as T
dev = torch.device("xpu"); g = np.random.default_rng(0)
kd, n = 5120, 2304
for K in (1, 2, 3):
    t = torch.from_numpy(ref.random_packed((kd // 16) * (n // 16), K, g).reshape(kd // 16, n // 16, 16 * K)).to(dev)
    out = torch.empty(kd, n, dtype=torch.half, device=dev)
    T.dequant(t, K, out); torch.xpu.synchronize(); t0 = time.time()
    for _ in range(10): T.dequant(t, K, out)
    torch.xpu.synchronize(); dt = (time.time() - t0) / 10
    print(f"dequant K={K}: {dt*1e3:.2f} ms per projection, {kd*n/dt/1e9:.1f} G weights/s", flush=True)
