"""FP8 (e4m3, E8M0 scale per 32 x 32 block) linears kept in FP8 on the B70.

DeepSeek-V4.1's attention projections and shared experts ship as block-scaled FP8 (6.5 GB). exllamav3 expands them to
fp16 at load (13 GB, read in full every decode step); here they stay FP8: decode rows go through a Triton GEMV that
converts in registers, longer inputs dequantize the weight into an fp16 scratch and run one oneDNN GEMM.

e4m3 -> fp16 exactly: the fp16 whose bits are ((b & 0x7F) << 7) | ((b & 0x80) << 8) is the e4m3 value / 256 (normal and
subnormal alike); the 256 folds into the block scale.
"""
import os

import torch
import triton
import triton.language as tl

GEMV_MAX_M = int(os.environ.get("DSV_FP8_GEMV_MAX_M", "8"))
FP8_SYCL = os.environ.get("DSV_FP8_SYCL", "1") == "1"
WOA_FUSED = os.environ.get("DSV_WOA_FUSED", "1") == "1"


@triton.jit
def _e4m3_x256(b):
    """uint8 e4m3 codes -> fp32 value / 256."""
    b = b.to(tl.uint16)
    h = ((b & 0x7F) << 7) | ((b & 0x80) << 8)
    return h.to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _fp8_gemv_kernel(x_ptr, w_ptr, s_ptr, y_ptr, M, N, Kd, sx, sy,
                     BN: tl.constexpr, BK: tl.constexpr, MR: tl.constexpr, SPLIT: tl.constexpr):
    """y[m, n] (+)= sum_k x[m, k] * w[n, k] * 2^(s[n/32, k/32] - 127). w [N, Kd] uint8, s [N/32, Kd/32] uint8.
    Programs: (N / BN, SPLIT); SPLIT > 1 accumulates with atomics into a zeroed fp32 y."""
    pid = tl.program_id(0)
    ks = tl.program_id(1)
    n = pid * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc0 = tl.zeros((BN,), tl.float32)
    acc1 = tl.zeros((BN,), tl.float32)
    acc2 = tl.zeros((BN,), tl.float32)
    acc3 = tl.zeros((BN,), tl.float32)
    nk = Kd // SPLIT
    k0 = ks * nk
    for kb in range(k0, k0 + nk, BK):
        w = tl.load(w_ptr + n[:, None].to(tl.int64) * Kd + kb + k[None, :])
        sc = tl.load(s_ptr + (n[:, None] // 32) * (Kd // 32) + (kb + k[None, :]) // 32)
        wf = _e4m3_x256(w) * tl.exp2(sc.to(tl.float32) - 119.0)        # 2^(s - 127) * 256
        x0 = tl.load(x_ptr + kb + k).to(tl.float32)
        acc0 += tl.sum(wf * x0[None, :], axis=1)
        if MR > 1:
            x1 = tl.load(x_ptr + sx + kb + k, mask=M > 1, other=0.0).to(tl.float32)
            acc1 += tl.sum(wf * x1[None, :], axis=1)
        if MR > 2:
            x2 = tl.load(x_ptr + 2 * sx + kb + k, mask=M > 2, other=0.0).to(tl.float32)
            acc2 += tl.sum(wf * x2[None, :], axis=1)
        if MR > 3:
            x3 = tl.load(x_ptr + 3 * sx + kb + k, mask=M > 3, other=0.0).to(tl.float32)
            acc3 += tl.sum(wf * x3[None, :], axis=1)
    if SPLIT == 1:
        tl.store(y_ptr + n, acc0)
        if MR > 1:
            tl.store(y_ptr + sy + n, acc1, mask=(n < N) & (M > 1))
        if MR > 2:
            tl.store(y_ptr + 2 * sy + n, acc2, mask=(n < N) & (M > 2))
        if MR > 3:
            tl.store(y_ptr + 3 * sy + n, acc3, mask=(n < N) & (M > 3))
    else:
        tl.atomic_add(y_ptr + n, acc0)
        if MR > 1:
            tl.atomic_add(y_ptr + sy + n, acc1, mask=(n < N) & (M > 1))
        if MR > 2:
            tl.atomic_add(y_ptr + 2 * sy + n, acc2, mask=(n < N) & (M > 2))
        if MR > 3:
            tl.atomic_add(y_ptr + 3 * sy + n, acc3, mask=(n < N) & (M > 3))


@triton.jit
def _fp8_dequant_t_kernel(w_ptr, s_ptr, o_ptr, N, Kd, BN: tl.constexpr, BK: tl.constexpr):
    """o[k, n] = w[n, k] * scale (fp16, transposed: [Kd, N] row-major for x @ o)."""
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN)
    k = pk * BK + tl.arange(0, BK)
    w = tl.load(w_ptr + n[:, None].to(tl.int64) * Kd + k[None, :])
    sc = tl.load(s_ptr + (n[:, None] // 32) * (Kd // 32) + k[None, :] // 32)
    wf = _e4m3_x256(w) * tl.exp2(sc.to(tl.float32) - 119.0)
    tl.store(o_ptr + k[None, :].to(tl.int64) * N + n[:, None], wf.to(tl.float16))


_SCRATCH = {}
_S = None


def S():
    global _S
    if _S is None:
        torch.ops.load_library(os.path.join(os.path.dirname(__file__), "..", "lib", "dsv_fp8.so"))
        _S = torch.ops.dsv_fp8
    return _S


def _scratch(dev, numel):
    t = _SCRATCH.get(dev)
    if t is None or t.numel() < numel:
        t = torch.empty(numel, dtype=torch.half, device=dev)
        _SCRATCH[dev] = t
    return t[:numel]


class FP8Linear:
    """Drop-in `inner` for exllamav3 Linear: forward(x, params, out_dtype)."""
    quant_type = "fp8x"
    bc = None

    def __init__(self, w_u8: torch.Tensor, scale_u8: torch.Tensor, out_dtype, key=""):
        self.w = w_u8.contiguous()                          # [N, Kd] uint8 (checkpoint (out, in) orientation)
        self.s = scale_u8.contiguous()                      # [N/32, Kd/32] uint8 exponent bytes
        self.out_features, self.in_features = self.w.shape
        self.out_dtype = out_dtype
        self.key = key
        self.bias = None
        N, Kd = self.out_features, self.in_features
        assert N % 32 == 0 and Kd % 32 == 0, (key, N, Kd)
        # GEMV launch shape: enough programs to fill the B70 (32 Xe cores)
        self.BN = 32
        self.BK = 128 if Kd % 128 == 0 else 32
        progs = N // self.BN
        split = 1
        while progs * split < 256 and Kd // (split * 2) >= self.BK and (Kd // (split * 2)) % self.BK == 0:
            split *= 2
        self.split = split

    def weight_fp16_t(self) -> torch.Tensor:
        N, Kd = self.out_features, self.in_features
        o = _scratch(self.w.device, N * Kd).view(Kd, N)
        _fp8_dequant_t_kernel[(N // 32, Kd // 64)](self.w, self.s, o, N, Kd, BN=32, BK=64, num_warps=4)
        return o

    def forward(self, x, params=None, out_dtype=None):
        out_dtype = out_dtype or self.out_dtype or torch.half
        shp = x.shape[:-1] + (self.out_features,)
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.half:
            x2 = x2.half()
        x2 = x2.contiguous()
        M = x2.shape[0]
        N, Kd = self.out_features, self.in_features
        if FP8_SYCL and M <= GEMV_MAX_M and Kd % 256 == 0:
            y = torch.empty((M, N), dtype=torch.float if out_dtype == torch.float else torch.half, device=x.device)
            S().fp8_gemv(x2, self.w, self.s, y, N)
            if y.dtype != out_dtype:
                y = y.to(out_dtype)
        elif M <= GEMV_MAX_M:
            y = (torch.zeros if self.split > 1 else torch.empty)((M, N), dtype=torch.float, device=x.device)
            for m0 in range(0, M, 4):
                mm = min(4, M - m0)
                MR = 1 if mm == 1 else (2 if mm == 2 else 4)
                _fp8_gemv_kernel[(N // self.BN, self.split)](
                    x2[m0:], self.w, self.s, y[m0:], mm, N, Kd, Kd, N,
                    BN=self.BN, BK=self.BK, MR=MR, SPLIT=self.split, num_warps=4)
            y = y.to(out_dtype)
        else:
            W = self.weight_fp16_t()
            y = torch.matmul(x2, W)
            if out_dtype != torch.half:
                y = y.to(out_dtype)
        return y.view(shp)

    def get_weight_tensor(self):
        return self.weight_fp16_t().clone()

    def get_bias_tensor(self):
        return None


def install():
    """Patch exllamav3 Linear.load: block-scaled FP8 checkpoint tensors (weight F8_E4M3 + 2D E8M0 scale) become
    FP8Linear, sliced along the output dim when the Linear is a slice (frange on dim 0) of a fused tensor."""
    from exllamav3.modules.linear import Linear
    orig = Linear.load

    def load(self, device, **kwargs):
        self.device = device
        stc = self.config.stc
        src = None
        for k in [self.key] + ([self.alt_key] if self.alt_key else []):
            if stc.has_tensor(k + ".weight") and stc.has_tensor(k + ".scale"):
                src, fr = k, None
                break
        if src is None and self.fkey and stc.has_tensor(self.fkey + ".weight") and stc.has_tensor(self.fkey + ".scale") \
                and (self.frange is None or self.frange_dim == 0):
            src, fr = self.fkey, self.frange
        if src is not None:
            w = stc.get_tensor(src + ".weight", "cpu", no_defer=True, arena=False)
            s = stc.get_tensor(src + ".scale", "cpu", no_defer=True)
            if w.dtype == torch.float8_e4m3fn and s.dim() == 2 and s.dtype == torch.uint8 and \
                    w.shape[0] // s.shape[0] == 32 and w.shape[1] // s.shape[1] == 32:
                w = w.view(torch.uint8)
                if fr is not None:
                    a, b = fr
                    assert a % 32 == 0 and b % 32 == 0
                    w, s = w[a:b], s[a // 32:b // 32]
                assert w.shape == (self.out_features, self.in_features), (self.key, w.shape, self.out_features,
                                                                          self.in_features)
                self.inner = FP8Linear(w.to(device), s.to(device), self.out_dtype, key=self.key)
                self.quant_type = "fp8x"
                return
        return orig(self, device, **kwargs)

    Linear.load = load

    # grouped output projection (wo_a): one FP8 GEMV over the 8 group slices instead of 8 calls
    from exllamav3.modules.dsv4 import DSV4Attention
    orig_og = DSV4Attention._project_o_grouped

    def _project_o_grouped(self, o, params, out_dtype, mgemm_out=False):
        G = self.o_groups
        bsz, seq = o.shape[1], o.shape[2]
        if WOA_FUSED and bsz * seq <= GEMV_MAX_M and all(isinstance(l.inner, FP8Linear) for l in self.wo_a):
            fused = getattr(self, "_woa_fp8", None)
            if fused is None:
                fused = self._woa_fp8 = (torch.cat([l.inner.w for l in self.wo_a], 0).contiguous(),
                                         torch.cat([l.inner.s for l in self.wo_a], 0).contiguous(),
                                         self.wo_a[0].inner.out_features)
                rpg_ = self.wo_a[0].inner.out_features
                for g, l in enumerate(self.wo_a):   # the per-group linears become views of the fused copy
                    l.inner.w = fused[0][g * rpg_:(g + 1) * rpg_]
                    l.inner.s = fused[1][g * rpg_ // 32:(g + 1) * rpg_ // 32]
            w, sc, rpg = fused
            x = o.reshape(G, bsz * seq, o.shape[-1])
            if x.dtype != torch.half:
                x = x.half()
            y = torch.empty((bsz * seq, G * rpg), dtype=torch.half, device=o.device)
            S().fp8_gemv(x.contiguous(), w, sc, y, rpg)
            return self.wo_b.forward(y.view(bsz, seq, G * rpg), params, out_dtype=out_dtype or self.out_dtype)
        if getattr(self, "_woa_fp8", None) is not None:
            w, sc, rpg = self._woa_fp8
            o2 = torch.empty((bsz, seq, G * rpg), dtype=torch.half, device=o.device)
            for g in range(G):
                lin = FP8Linear(w[g * rpg:(g + 1) * rpg], sc[g * rpg // 32:(g + 1) * rpg // 32], torch.half)
                o2[..., g * rpg:(g + 1) * rpg] = lin.forward(o[g], params, torch.half)
            return self.wo_b.forward(o2, params, out_dtype=out_dtype or self.out_dtype)
        return orig_og(self, o, params, out_dtype, mgemm_out)

    DSV4Attention._project_o_grouped = _project_o_grouped
