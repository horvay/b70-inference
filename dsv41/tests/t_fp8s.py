import time, torch
torch.ops.load_library("/w/dsv41/xpu/lib/dsv_fp8.so")
F = torch.ops.dsv_fp8
from dsv_xpu.fp8 import FP8Linear
dev = torch.device("xpu")
torch.manual_seed(0)
for (N, Kd, G) in ((1280, 5120, 1), (32768, 1280, 1), (5120, 8192, 1), (2304, 5120, 1), (5120, 2304, 1), (512, 5120, 1), (8192, 4096, 8)):
    wf = torch.randn(N, Kd) * 0.05
    w8 = wf.to(torch.float8_e4m3fn)
    s = torch.randint(120, 130, (N // 32, Kd // 32), dtype=torch.uint8)
    ref = w8.float() * (s.float() - 127).exp2().repeat_interleave(32, 0).repeat_interleave(32, 1)
    wd, sd = w8.view(torch.uint8).to(dev), s.to(dev)
    for M in (1, 2, 3, 4):
        x = torch.randn(G, M, Kd, device=dev).half()
        y = torch.empty(M, N, device=dev)
        F.fp8_gemv(x, wd, sd, y, N // G)
        rpg = N // G
        want = torch.cat([x[g].float().cpu() @ ref[g * rpg:(g + 1) * rpg].t() for g in range(G)], 1)
        err = ((y.cpu() - want).abs().max() / want.abs().max()).item()
        if M == 1 or err > 1e-3:
            print(f"N={N} K={Kd} G={G} M={M} rel err {err:.2g}")
    x = torch.randn(G, 1, Kd, device=dev).half(); y = torch.empty(1, N, device=dev)
    for _ in range(3): F.fp8_gemv(x, wd, sd, y, N // G)
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(200): F.fp8_gemv(x, wd, sd, y, N // G)
    torch.xpu.synchronize(); dt = (time.time() - t0) / 200
    print(f"   M=1 {dt*1e6:.0f} us  {N*Kd/dt/1e9:.0f} GB/s")
