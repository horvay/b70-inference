"""Capture each custom-kernel family alone in an XPUGraph and replay it: which one breaks graph replay?"""
import sys, numpy as np, torch
sys.path.insert(0, "/w/dsv41/exllamav3/tests")
from testlib import trellis as ref
from dsv_xpu import moe, fp8 as F8
from dsv_xpu.ops import _rope_lib
dev = torch.device("xpu")
g = np.random.default_rng(0)
s64 = lambda p: p - (1 << 64) if p >= (1 << 63) else p
which = sys.argv[1]

def cap_replay(fn, n=20):
    fn(); torch.xpu.synchronize()
    gr = torch.xpu.XPUGraph()
    with torch.xpu.graph(gr):
        out = fn()
    for i in range(n):
        gr.replay()
    torch.xpu.synchronize()
    print(which, "ok", flush=True)

if which == "gemv":
    H, I = 5120, 2304
    Ks = [1, 2, 3, 4, 1, 2]
    trs = [torch.from_numpy(ref.random_packed((H // 16) * (I // 16), K, g).reshape(H // 16, I // 16, 16 * K)).to(dev) for K in Ks]
    xh = torch.randn(len(Ks), 1, H, device=dev).half()
    wptr = torch.tensor([s64(t.data_ptr()) for t in trs], dtype=torch.int64, device=dev)
    kk = torch.tensor(Ks, dtype=torch.int32, device=dev)
    ni = torch.tensor([len(Ks)], dtype=torch.int32, device=dev)
    part = torch.empty((8, len(Ks), 1, I), device=dev)
    cap_replay(lambda: moe.X().moe_gemv(xh, wptr, kk, ni, part, 0x1E))
elif which == "gemv_host":
    H, I = 5120, 2304
    Ks = [1, 2]
    trs = [torch.from_numpy(ref.random_packed((H // 16) * (I // 16), K, g).reshape(H // 16, I // 16, 16 * K)).pin_memory() for K in Ks]
    xh = torch.randn(len(Ks), 1, H, device=dev).half()
    wptr = torch.tensor([s64(t.data_ptr()) for t in trs], dtype=torch.int64, device=dev)
    kk = torch.tensor(Ks, dtype=torch.int32, device=dev)
    ni = torch.tensor([len(Ks)], dtype=torch.int32, device=dev)
    part = torch.empty((8, len(Ks), 1, I), device=dev)
    cap_replay(lambda: moe.X().moe_gemv(xh, wptr, kk, ni, part, 0x1E))
elif which == "fp8":
    w = torch.randint(0, 120, (2304, 5120), dtype=torch.uint8, device=dev); sc = torch.full((72, 160), 120, dtype=torch.uint8, device=dev)
    lin = F8.FP8Linear(w, sc, torch.half)
    x = torch.randn(1, 5120, device=dev).half()
    cap_replay(lambda: lin.forward(x, None, torch.float))
elif which == "matmul":
    a = torch.randn(1, 5120, device=dev).half(); b = torch.randn(5120, 384, device=dev).half()
    cap_replay(lambda: torch.matmul(a, b).float())
elif which == "route":
    logits = torch.randn(1, 384, device=dev); bias = torch.zeros(384, device=dev); res = torch.zeros(384, device=dev)
    def f():
        ids = torch.empty((1, 6), dtype=torch.int64, device=dev); w = torch.empty((1, 6), device=dev)
        want = torch.empty((1, 6), dtype=torch.int64, device=dev)
        moe.G().moe_route(logits, bias, res, ids, w, want, 1.5)
        return ids
    cap_replay(f)
elif which == "hc_apply":
    from glm_xpu import k_hc
    x = torch.randn(1, 4, 5120, device=dev); y = torch.randn(1, 5120, device=dev)
    post = torch.rand(1, 4, device=dev); comb = torch.rand(1, 4, 4, device=dev)
    cap_replay(lambda: k_hc.hc_apply(x, y, post, comb))
elif which == "silu":
    from glm_xpu.ops import silu_mul
    gg = torch.randn(1, 2304, device=dev).half(); u = torch.randn(1, 2304, device=dev).half()
    cap_replay(lambda: silu_mul(gg, u, u, 10.0))
if which == "glue":
    H, I, E = 5120, 2304, 384
    vec = torch.randn(E, 22272, device=dev).half(); ptr = torch.randint(0, 1000, (E, 3), device=dev); Kt = torch.randint(1, 4, (E, 3), dtype=torch.int32, device=dev)
    y = torch.randn(1, H, device=dev).half(); ids = torch.randint(0, E, (1, 6), device=dev); w = torch.rand(1, 6, device=dev)
    xh = torch.empty(12, 1, H, device=dev).half(); wp = torch.empty(12, dtype=torch.int64, device=dev); kk = torch.empty(12, dtype=torch.int32, device=dev)
    wpd = torch.empty(6, dtype=torch.int64, device=dev); kkd = torch.empty(6, dtype=torch.int32, device=dev)
    part = torch.randn(8, 12, 1, I, device=dev); xd = torch.empty(6, 1, I, device=dev).half(); part2 = torch.randn(4, 6, 1, H, device=dev)
    def f():
        moe.G().moe_prep(y, ids, vec, ptr, Kt, xh, wp, kk, wpd, kkd)
        moe.G().moe_mid(part, ids, vec, xd, 10.0)
        out = torch.empty(1, H, device=dev)
        moe.G().moe_post(part2, ids, w, vec, out)
        return out
    cap_replay(f)
