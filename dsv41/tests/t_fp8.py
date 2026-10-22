import time, torch
from dsv_xpu.fp8 import FP8Linear
dev = torch.device("xpu")
torch.manual_seed(0)
for (N, Kd) in ((1280, 5120), (32768, 1280), (5120, 8192), (2304, 5120), (5120, 2304), (512, 5120)):
    wf = torch.randn(N, Kd) * 0.05
    w8 = wf.to(torch.float8_e4m3fn)
    s = torch.randint(120, 130, (N // 32, Kd // 32), dtype=torch.uint8)
    ref = w8.float() * (s.float() - 127).exp2().repeat_interleave(32, 0).repeat_interleave(32, 1)
    lin = FP8Linear(w8.view(torch.uint8).to(dev), s.to(dev), torch.half)
    for M in (1, 2, 3, 4, 33):
        x = torch.randn(M, Kd, device=dev).half()
        y = lin.forward(x, None, torch.float)
        want = x.float().cpu() @ ref.t()
        err = ((y.cpu() - want).abs().max() / want.abs().max()).item()
        if M in (1, 33) or err > 1e-2:
            print(f"N={N} K={Kd} M={M} rel err {err:.2g} split {lin.split}")
    x = torch.randn(1, Kd, device=dev).half()
    for _ in range(3): lin.forward(x, None, torch.float)
    torch.xpu.synchronize(); t0 = time.time()
    for _ in range(100): lin.forward(x, None, torch.float)
    torch.xpu.synchronize(); dt = (time.time() - t0) / 100
    print(f"   M=1 {dt*1e6:.0f} us  {N*Kd/dt/1e9:.0f} GB/s")
