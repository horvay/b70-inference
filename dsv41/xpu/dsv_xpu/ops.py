"""exllamav3_ext ops added for DeepSeek-V4.1-Flash on XPU (torch reference implementations first)."""
import torch
import torch.nn.functional as F

HAS_GR_MIX_TILED = False
HAS_DET_GEMM = False


import os as _os
_R = None
_ROPE_SYCL = _os.environ.get("DSV_ROPE_SYCL", "1") == "1"


def _rope_lib():
    global _R
    if _R is None:
        torch.ops.load_library(_os.path.join(_os.path.dirname(__file__), "..", "lib", "dsv_rope.so"))
        _R = torch.ops.dsv_rope
    return _R


def _i32(t, dev):
    if t is None:
        return None
    t = t.to(dev)
    return t if t.dtype == torch.int32 else t.int()


def rope(q, out_q, k, out_k, inv_freq, position, positions, position_ids, rope_mode, attn_factor, q_norm, k_norm,
         norm_eps, norm_constant_bias, llama_4_scaling_beta, llama_4_scaling_original, rotate_dims, rotate_offset):
    if (_ROPE_SYCL and q.device.type == "xpu" and rope_mode == 1 and rotate_dims == 1 and llama_4_scaling_beta == 0.0
            and q.dim() == 4 and q.dtype == torch.half and q.shape[-1] % 16 == 0 and q.shape[-1] <= 512
            and q.stride(-1) == 1 and (k is None or (k.dim() == 4 and k.stride(-1) == 1 and k.dtype == torch.half))
            and (q.shape[0] == 1 or q.stride(0) == q.shape[1] * q.stride(1))
            and inv_freq.dim() == 1 and rotate_offset % 2 == 0):
        dev = q.device
        _rope_lib().rope(q, out_q, k, out_k, inv_freq.float().contiguous(), int(position), _i32(positions, dev),
                         _i32(position_ids.reshape(-1) if position_ids is not None else None, dev), 1,
                         float(attn_factor), q_norm, k_norm, float(norm_eps), float(norm_constant_bias),
                         int(rotate_offset))
        return
    return rope_torch(q, out_q, k, out_k, inv_freq, position, positions, position_ids, rope_mode, attn_factor, q_norm,
                      k_norm, norm_eps, norm_constant_bias, llama_4_scaling_beta, llama_4_scaling_original,
                      rotate_dims, rotate_offset)


def rope_torch(q, out_q, k, out_k, inv_freq, position, positions, position_ids, rope_mode, attn_factor, q_norm, k_norm,
               norm_eps, norm_constant_bias, llama_4_scaling_beta, llama_4_scaling_original, rotate_dims, rotate_offset):
    """exllamav3 rope.cu: per head, optional RMS norm (weighted; a head without a weight is left unnormalized), then
    rotation of rd = 2 * len(inv_freq) dims at rotate_offset. GPT-J (interleaved pairs) or NeoX (halves).
    q (bsz, seq, Hq, D), k (bsz, seq, Hk, D) fp16, rotated in place into out_q / out_k."""
    assert rotate_dims == 1 and llama_4_scaling_beta == 0.0
    bsz, seq = q.shape[0], q.shape[1]
    dev = q.device
    if positions is not None:
        pos = positions.to(dev).view(bsz, 1).long() + torch.arange(seq, device=dev).view(1, seq)
    elif position_ids is not None:
        pos = position_ids.to(dev).view(bsz, seq).long()
    else:
        pos = (torch.arange(seq, device=dev) + int(position)).view(1, seq).expand(bsz, seq)
    rd = inv_freq.shape[-1] * 2
    ang = pos.float().unsqueeze(-1) * inv_freq.float().view(1, 1, -1)       # (bsz, seq, rd/2)
    sin = (torch.sin(ang) * attn_factor).unsqueeze(2)
    cos = (torch.cos(ang) * attn_factor).unsqueeze(2)

    def one(x, out, w):
        if x is None:
            return
        xf = x.float()
        if w is not None:
            rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm_eps)
            xf = (xf * rms).half().float() * (w.float() + norm_constant_bias)
        seg = xf[..., rotate_offset:rotate_offset + rd]
        if rope_mode == 1:     # GPTJ: interleaved pairs
            a, b = seg[..., 0::2], seg[..., 1::2]
            r1 = a * cos - b * sin
            r2 = b * cos + a * sin
            seg = torch.stack((r1, r2), -1).flatten(-2)
        else:                  # NEOX
            h = rd // 2
            a, b = seg[..., :h], seg[..., h:]
            seg = torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
        xf = xf.clone() if xf.data_ptr() == x.data_ptr() else xf
        xf[..., rotate_offset:rotate_offset + rd] = seg
        out.copy_(xf.to(out.dtype))

    one(q, out_q, q_norm)
    one(k, out_k, k_norm)


ROUTING_ACT_SIGMOID, ROUTING_ACT_SQRTSP = 0, 1


def routing_ds3_nogroup(hidden, gate, scores, bias, topk_indices, topk_weights, scaling_factor, gate_t, act_fn,
                        gate_i8, gate_sb):
    """noaux_tc top-k routing (no groups): selection by act(logits) + bias, weights from act(logits) of the selected,
    normalized over them and scaled. act: sigmoid (0) or sqrt(softplus) (1)."""
    E = scores.shape[-1]
    k = topk_indices.shape[-1]
    logits = torch.matmul(hidden.float(), gate[:, :E].float())
    scores.copy_(logits)
    s = torch.sigmoid(logits) if act_fn == 0 else F.softplus(logits).sqrt()
    key = s + bias.float() if bias is not None else s
    idx = torch.topk(key, k, dim=-1, sorted=True).indices
    w = s.gather(-1, idx)
    w = w * scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
    topk_indices.copy_(idx)
    topk_weights.copy_(w)


def hc_apply(x, y, post, comb, *extra):
    """x <- post (x) y + comb^T x (fork signature carries two trailing optional args, unused here)."""
    from glm_xpu.ops import hc_apply as _glm_hc_apply
    assert all(e is None for e in extra)
    return _glm_hc_apply(x, y, post, comb)


def dspark_write_rows(rows, kv, block_table, cache_seqlens):
    """DSpark drafter main-kv rows -> paged kv (SYCL kernel in dsv_attn.so)."""
    from .attn import A
    dev = kv.device
    bt = block_table.to(dev)
    sl = cache_seqlens.to(dev)
    A().dspark_write_rows(rows.contiguous(), kv, bt if bt.dtype == torch.int32 else bt.int().contiguous(),
                          sl if sl.dtype == torch.int32 else sl.int().contiguous())
