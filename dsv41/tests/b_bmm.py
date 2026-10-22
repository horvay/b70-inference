import time, torch
dev = torch.device("xpu")
for (n, m, k, nn) in ((16, 128, 5120, 2304), (16, 512, 5120, 2304), (16, 128, 2304, 5120)):
    a = torch.randn(n, m, k, device=dev).half(); b = torch.randn(n, k, nn, device=dev).half()
    for f, name in ((lambda: torch.bmm(a, b), "bmm"), (lambda: [a[i] @ b[i] for i in range(n)], "loop")):
        f(); torch.xpu.synchronize(); t0 = time.time()
        for _ in range(5): f()
        torch.xpu.synchronize(); dt = (time.time() - t0) / 5
        print(f"{name} {n}x[{m}x{k}]@[{k}x{nn}]: {dt*1e3:.1f} ms, {2*n*m*k*nn/dt/1e12:.1f} TFLOPS", flush=True)
