import time, torch
dev = torch.device("xpu")
x = torch.randn(1, 5120, device=dev)
def body(n):
    y = x
    for _ in range(n):
        y = y * 1.0001 + 0.0001
    return y
for n in (1, 10, 50):
    body(n); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        out = body(n)
    for _ in range(5): g.replay()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): g.replay()
    t1 = time.perf_counter()
    torch.xpu.synchronize()
    t2 = time.perf_counter()
    # eager equivalent
    t3 = time.perf_counter()
    for _ in range(200): body(n)
    t4 = time.perf_counter(); torch.xpu.synchronize(); t5 = time.perf_counter()
    print(f"{n} kernels: graph replay enqueue {(t1 - t0) / 200 * 1e6:.0f} us, complete {(t2 - t0) / 200 * 1e6:.0f} us; eager enqueue {(t4 - t3) / 200 * 1e6:.0f} us, complete {(t5 - t3) / 200 * 1e6:.0f} us")
