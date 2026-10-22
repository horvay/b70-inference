"""EXL3 trellis kernels in Triton for any integer K (1..8), mul1 codebook.

Format (exllamav3 tests/testlib/trellis.py): a 16x16 tile is a ring of 256 positions; position i's 16-bit window is
ring bits [(i+1)K - 16, (i+1)K) mod 256K, MSB first, ring bit 0 at the MSB of word 0 (words = int16 pairs, little
endian). Ring position i holds tile element (r, c) with t = i // 8, j = i % 8:
    r = (t % 4) * 2 + (j & 1) + 8 * ((j >> 1) & 1),   c = t // 4 + 8 * ((j >> 2) & 1)
The kernels walk positions in (c, s) order, s = 0..15 the 16 positions of column c:
    t = 4 * (c % 8) + s // 4,   j = s % 4 + 4 * (c // 8)   ->   r = 2 * (s // 4) + (s & 1) + 8 * ((s >> 1) & 1)
so a column's 16 values reduce along one axis.

mul1 decode: x = window * 0x83DCD12D mod 2^32; h = 1024 + sum of x's 4 bytes; value = fp16(h * k_inv + k_bias).
"""
import numpy as np
import torch
import triton
import triton.language as tl

K_INV = float(np.array([0x1EEE], np.uint16).view(np.float16)[0])
K_BIAS = float(np.array([0xC931], np.uint16).view(np.float16)[0])


@triton.jit
def _window(wp, cc, ss, K: tl.constexpr):
    """16-bit windows of ring positions (cc, ss) of the tile whose words start at wp (uint32 view)."""
    t = 4 * (cc % 8) + ss // 4
    j = ss % 4 + 4 * (cc // 8)
    i = t * 8 + j
    R: tl.constexpr = 256 * K
    start = ((i + 1) * K - 16 + R) % R
    wi = start >> 5
    off = start & 31
    w0 = tl.load(wp + wi).to(tl.uint32).to(tl.uint64)
    w1 = tl.load(wp + (wi + 1) % (8 * K)).to(tl.uint32).to(tl.uint64)
    v = (w0 << 32) | w1
    return ((v >> (48 - off).to(tl.uint64)) & 0xFFFF).to(tl.uint32)


@triton.jit
def _mul1(win):
    x = (win.to(tl.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
    h = (bs + 1024).to(tl.float32)
    return h * 0.00676727294921875 + -10.3828125     # K_INV, K_BIAS (asserted below)


@triton.jit
def _dequant_kernel(tr, out, tn, n_out, K: tl.constexpr):
    """One program per tile: W_hat[16 tk + r, 16 tn + c] (fp16)."""
    pid = tl.program_id(0)
    ktile = pid // tn
    ntile = pid % tn
    wp = tr + pid.to(tl.int64) * (8 * K)
    cc = tl.arange(0, 16)[:, None]
    ss = tl.arange(0, 16)[None, :]
    val = _mul1(_window(wp, cc, ss, K))
    r = 2 * (ss // 4) + (ss & 1) + 8 * ((ss >> 1) & 1)
    o = out + (ktile * 16 + r).to(tl.int64) * n_out + ntile * 16 + cc
    tl.store(o, val.to(tl.float16))


@triton.jit
def _dequant_ptr_kernel(addr, out, tn, n_out, K: tl.constexpr):
    pid = tl.program_id(0)
    ktile = pid // tn
    ntile = pid % tn
    wp = addr.to(tl.pointer_type(tl.uint32)) + pid.to(tl.int64) * (8 * K)
    cc = tl.arange(0, 16)[:, None]
    ss = tl.arange(0, 16)[None, :]
    val = _mul1(_window(wp, cc, ss, K))
    r = 2 * (ss // 4) + (ss & 1) + 8 * ((ss >> 1) & 1)
    o = out + (ktile * 16 + r).to(tl.int64) * n_out + ntile * 16 + cc
    tl.store(o, val.to(tl.float16))


@triton.jit
def _dequant_multi_kernel(ptrs, slots, out, tn, n_out, tiles, K: tl.constexpr):
    """Programs (tile, item): item's trellis at address ptrs[item] -> out[slots[item]] ([kdim, n] fp16 W_hat)."""
    pid = tl.program_id(0)
    item = tl.program_id(1)
    ktile = pid // tn
    ntile = pid % tn
    wp = tl.load(ptrs + item).to(tl.pointer_type(tl.uint32)) + pid.to(tl.int64) * (8 * K)
    slot = tl.load(slots + item).to(tl.int64)
    cc = tl.arange(0, 16)[:, None]
    ss = tl.arange(0, 16)[None, :]
    val = _mul1(_window(wp, cc, ss, K))
    r = 2 * (ss // 4) + (ss & 1) + 8 * ((ss >> 1) & 1)
    o = out + slot * tiles * 256 + (ktile * 16 + r).to(tl.int64) * n_out + ntile * 16 + cc
    tl.store(o, val.to(tl.float16))


def dequant_multi(ptrs: torch.Tensor, slots: torch.Tensor, K: int, kdim: int, n: int, out: torch.Tensor):
    """Dequantize several same-K trellises (int64 addresses) into out[slot] ([S, kdim, n] fp16)."""
    tk, tn = kdim // 16, n // 16
    if ptrs.numel() == 0:
        return out
    _dequant_multi_kernel[(tk * tn, ptrs.numel())](ptrs, slots, out, tn, n, tk * tn, K=K, num_warps=1)
    return out


def dequant_ptr(addr: int, K: int, kdim: int, n: int, out: torch.Tensor):
    """Trellis at device/host USM address `addr` ([kdim/16][n/16][16K] int16) -> out (kdim, n) fp16 W_hat."""
    tk, tn = kdim // 16, n // 16
    _dequant_ptr_kernel[(tk * tn,)](addr, out, tn, n, K=K, num_warps=1)
    return out


def dequant(trellis: torch.Tensor, K: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """(k/16, n/16, 16K) int16 trellis -> (k, n) fp16 W_hat (no Hadamards, no scales)."""
    tk, tn, w = trellis.shape
    assert w == 16 * K, (trellis.shape, K)
    if out is None:
        out = torch.empty((tk * 16, tn * 16), dtype=torch.half, device=trellis.device)
    words = trellis.contiguous().view(torch.int32)
    _dequant_kernel[(tk * tn,)](words, out, tn, tn * 16, K=K, num_warps=1)
    return out


# ------------------------------------------------------------------------------------------------ grouped GEMV
#
# Item list (one row per (activation row set, weight)):  item a reads input rows xin[a] (Hadamard domain, fp16,
# [M, k]) and weight wptr[a] (trellis words, K = kk[a]), writes fp32 partials out[a, M, n]. Programs: (item, n-block).
# The weight pointer is an int64 address (device or host USM), so VRAM- and RAM-resident experts look the same.

@triton.jit
def _moe_gemv_kernel(x_ptr, x_stride_item, wptr_ptr, k_ptr, out_ptr, n_items, kdim, n,
                     KMAX: tl.constexpr, M: tl.constexpr, NT: tl.constexpr, KSPLIT: tl.constexpr):
    pid_n = tl.program_id(0)
    item = tl.program_id(1)
    ks = tl.program_id(2)
    if item >= tl.load(n_items):
        return
    K = tl.load(k_ptr + item)
    base = tl.load(wptr_ptr + item).to(tl.pointer_type(tl.uint32))
    tn = n // 16
    tk = kdim // 16
    cc = tl.arange(0, 16)
    ss = tl.arange(0, 16)
    t = 4 * (cc[:, None] % 8) + ss[None, :] // 4
    j = ss[None, :] % 4 + 4 * (cc[:, None] // 8)
    i = t * 8 + j                                                     # [16c, 16s] ring position
    rr = 2 * (ss // 4) + (ss & 1) + 8 * ((ss >> 1) & 1)               # [16s] tile row
    nt = tl.arange(0, NT)
    R = 256 * K
    start = ((i + 1) * K - 16 + R) % R
    wi = start >> 5
    sh = (48 - (start & 31)).to(tl.uint64)
    wi1 = (wi + 1) % (8 * K)
    W = 8 * K
    tile_n0 = pid_n * NT
    acc0 = tl.zeros((NT, 16), dtype=tl.float32)
    acc1 = tl.zeros((NT, 16), dtype=tl.float32)
    acc2 = tl.zeros((NT, 16), dtype=tl.float32)
    acc3 = tl.zeros((NT, 16), dtype=tl.float32)
    rows_per = tl.cdiv(tk, KSPLIT)
    kt0 = ks * rows_per
    kt1 = tl.minimum(kt0 + rows_per, tk)
    xb = x_ptr + item.to(tl.int64) * x_stride_item
    for kt in range(kt0, kt1):
        tbase = base + (kt.to(tl.int64) * tn + tile_n0) * W           # NT tiles of this k-row
        wp = tbase + nt[:, None, None].to(tl.int64) * W
        w0 = tl.load(wp + wi[None, :, :]).to(tl.uint64)
        w1 = tl.load(wp + wi1[None, :, :]).to(tl.uint64)
        win = (((w0 << 32) | w1) >> sh[None, :, :]) & 0xFFFF
        x = (win * 0x83DCD12D) & 0xFFFFFFFF
        bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
        v = (bs + 1024).to(tl.float32) * 0.00676727294921875 + -10.3828125   # [NT, 16c, 16s]
        v = v.to(tl.float16).to(tl.float32)
        xr0 = tl.load(xb + kt * 16 + rr).to(tl.float32)
        acc0 += tl.sum(v * xr0[None, None, :], axis=2)
        if M > 1:
            xr1 = tl.load(xb + kdim + kt * 16 + rr).to(tl.float32)
            acc1 += tl.sum(v * xr1[None, None, :], axis=2)
        if M > 2:
            xr2 = tl.load(xb + 2 * kdim + kt * 16 + rr).to(tl.float32)
            acc2 += tl.sum(v * xr2[None, None, :], axis=2)
        if M > 3:
            xr3 = tl.load(xb + 3 * kdim + kt * 16 + rr).to(tl.float32)
            acc3 += tl.sum(v * xr3[None, None, :], axis=2)
    col = (tile_n0 + nt[:, None]) * 16 + cc[None, :]
    ob = out_ptr + ((ks * tl.num_programs(1) + item).to(tl.int64) * M) * n
    tl.store(ob + col, acc0)
    if M > 1:
        tl.store(ob + n + col, acc1)
    if M > 2:
        tl.store(ob + 2 * n + col, acc2)
    if M > 3:
        tl.store(ob + 3 * n + col, acc3)


def moe_gemv(xin: torch.Tensor, wptr: torch.Tensor, kk: torch.Tensor, n_items: torch.Tensor, n: int,
             max_items: int, NT: int = 4, KSPLIT: int = 4, out: torch.Tensor | None = None) -> torch.Tensor:
    """xin (max_items, M, k) fp16 Hadamard-domain inputs; wptr (max_items,) int64 addresses; kk (max_items,) int32 K;
    n_items (1,) int32 on device. Returns (KSPLIT, max_items, M, n) fp32 partials (sum over axis 0)."""
    _, M, kdim = xin.shape
    if out is None:
        out = torch.empty((KSPLIT, max_items, M, n), dtype=torch.float, device=xin.device)
    grid = (n // (16 * NT), max_items, KSPLIT)
    _moe_gemv_kernel[grid](xin, M * kdim, wptr, kk, out, n_items, kdim, n, KMAX=8, M=M, NT=NT, KSPLIT=KSPLIT,
                           num_warps=4)
    return out


_H = {}


def had128(dev):
    H = _H.get(dev)
    if H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = (h / 128 ** 0.5).to(dev)
        _H[dev] = H
    return H


assert abs(K_INV - 0.00676727294921875) < 1e-12 and K_BIAS == -10.3828125, (K_INV, K_BIAS)
