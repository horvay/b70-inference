"""Lean decode attention for DeepSeek-V4.1 (every layer: sliding, CSA ratio 2, ratio 1) on the B70.

exllamav3's per-job attention path spends ~2 ms of host time per layer (dozens of torch ops and Triton launches per
job). For decode steps (one token per job, B <= MAX_B jobs) this runs each layer as:

  x fan   : ONE FP8 GEMV for q_a | wkv | compressor wkv | compressor wgate (fused weights, all jobs at once)
  q_norm  : SYCL RMS norm
  q fan   : ONE FP8 GEMV for q_b
  rope    : SYCL rope at per-job device positions, kv head norm folded in
  pools   : kv sources write the compressed latent (torch, per job; ratio-2 layers every other token)
  indexer : index sources with more than index_topk entries run exllamav3's indexer per job (long contexts only);
            below that every entry is selected (an identity index row, no scoring)
  attn    : SYCL split + combine over every job's [window ++ selected entries], de-rotation, group-major output and
            the ring append in the same launches
  o proj  : the fused grouped wo_a FP8 GEMV + wo_b

Per-job state that exllamav3 hands between layers through the forward's params dict (kv-source pool layer, index
lists, candidate masks) is keyed per job here (exllamav3 keys it per layer only, so with several jobs a consumer layer
read the LAST job's block-table row and index list).
"""
from __future__ import annotations

import os

import torch

from exllamav3.modules.dsv4 import DSV4Attention
from exllamav3.modules.dsv41 import DSV41Attention
from exllamav3.constants import PAGE_SIZE

from .fp8 import FP8Linear, S as F8

ENABLED = os.environ.get("DSV_ATTN_LEAN", "1") == "1"
CHECK = [int(os.environ.get("DSV_ATTN_CHECK", "0"))]
MAX_B = 8
MAX_S = 8           # tokens per job on the lean path (decode 1, verify blocks up to the draft size + 1)
MAX_ROWS = 16
_A = None
_orig_forward = DSV4Attention.forward


def A():
    global _A
    if _A is None:
        torch.ops.load_library(os.path.join(os.path.dirname(__file__), "..", "lib", "dsv_attn.so"))
        _A = torch.ops.dsv_attn
    return _A


def log(*a):
    print(" -- dsv.attn:", *a, flush=True)


class _Prep:
    """Fused weights and offsets of one attention module."""

    def __init__(self, attn):
        lins = [attn.q_a, attn.wkv]          # (the compressor's wkv / wgate are bf16 in the checkpoint: own GEMMs)
        self.ok = all(isinstance(l.inner, FP8Linear) for l in lins + [attn.q_b])
        if not self.ok:
            return
        self.nqa = attn.q_a.out_features
        self.off_kv = self.nqa
        self.wx = torch.cat([l.inner.w for l in lins], 0).contiguous()
        self.sx = torch.cat([l.inner.s for l in lins], 0).contiguous()
        self.nx = self.wx.shape[0]
        self.wq, self.sq = attn.q_b.inner.w, attn.q_b.inner.s
        self.nq = self.wq.shape[0]
        a = 0
        for l in lins:                       # the per-linear weights become views of the fused copy (prompts use them)
            n = l.inner.w.shape[0]
            l.inner.w, l.inner.s = self.wx[a:a + n], self.sx[a // 32:(a + n) // 32]
            a += n
        self.inv_freq = attn._rope_type().float().contiguous()
        self.derot = attn._rope_type_neg().float().contiguous()
        self.hpg = attn.num_q_heads // attn.o_groups
        self.is41 = isinstance(attn, DSV41Attention)


_WS = {}
_META = {}


def _ws(dev, B, kpad):
    ns = 5 + -(-kpad // 32)
    key = (dev, B, kpad)
    w = _WS.get(key)
    if w is None:
        w = _WS[key] = (torch.empty(B * ns * 64 * 2, dtype=torch.float, device=dev),
                        torch.empty(B * ns * 64 * 512, dtype=torch.float, device=dev))
    return w


ATTNS = []          # every DSV4Attention module of the model in forward order (runner)


MS = 12          # meta fields per query row (dsv_attn.sycl)


def prepare_step(params, B, dev, S=1):
    """Per-step query-row state in fixed device buffers (graph replays read them), rows r = job * S + s (S tokens per
    job: 1 for decode, more for a verify block): meta [3, R, 12] int32 (no pool / ratio 1 / ratio 2; fields in
    dsv_attn.sycl), positions [R] and the block table [B, NPR]. Shifts the rings first when a job's tokens would run
    past the ring's end (every layer, by whole pages: the window rows stay inside; the attention reads the shifted ring
    and appends after it). Enqueues copies: call it before a step, outside graph capture."""
    rsg = params["recurrent_states"]
    sig = (S,) + tuple((rs.position, rs.window_beg, rs.slot) for rs in rsg[:B])
    m = params.get("_dsv_meta")
    if m is not None and m["sig"] == sig:
        return m
    R = B * S
    a0 = ATTNS[0]
    inst = params.get("layer_instance", 0)
    rsl0 = a0._get_rsl(rsg[0], (a0.layer_idx, inst))
    ring_rows, window = rsl0.ring_rows, a0.sliding_window
    kl0 = next((rsg[0].cache.layers[(a.layer_idx, inst)] for a in ATTNS if getattr(a, "is_kv_source", False)), None)
    npr = kl0.num_pages if kl0 is not None else 1
    ent = _META.get((dev, B, S))
    n = 3 * R * MS + R
    if ent is None or ent["npr"] < npr:
        ent = _META[(dev, B, S)] = dict(
            i=0, npr=npr, pins=[torch.zeros(n, dtype=torch.int32).pin_memory() for _ in range(4)],
            d=torch.zeros(n, dtype=torch.int32, device=dev),
            bt=torch.zeros((B, npr), dtype=torch.int32, device=dev))
    pin = ent["pins"][ent["i"] % 4]
    ent["i"] += 1
    hv = pin.numpy()
    for b in range(B):
        rs = rsg[b]
        pos0, beg, slot = rs.position, rs.window_beg, rs.slot
        off = pos0 - beg
        shift = 0
        if off + S > ring_rows:
            shift = -(-(off + S - ring_rows) // PAGE_SIZE) * PAGE_SIZE
            for att in ATTNS:
                ring = att._get_rsl(rs, (att.layer_idx, inst)).ring[slot]
                ring[:ring_rows - shift].copy_(ring[shift:].clone())
            rs.wshift = shift
        for si in range(S):
            qp = pos0 + si
            r = b * S + si
            floor_ = qp - min(window - 1, qp - beg, qp)
            for mi, npool in enumerate((0, min(qp + 1, 512), min((qp + 1) // 2, 512))):
                o = (mi * R + r) * MS
                hv[o:o + MS] = (qp, floor_, beg + shift, slot, off + si - shift, npool, pos0, b * S - pos0, qp, r, b,
                                0)
            hv[3 * R * MS + r] = qp
    d = ent["d"]
    d.copy_(pin, non_blocking=True)
    bt = ent["bt"]
    if kl0 is not None:
        from exllamav3.util.tensor import get_for_device
        if "block_table" in params:
            src = get_for_device(params, "block_table", dev)
        else:
            sbt = kl0.slot_bt(rsg[0].cache.num_slots)
            src = sbt[[rsg[i].slot for i in range(B)]]
        bt[:, :src.shape[1]].copy_(src[:B].to(torch.int32))
    m = dict(meta=d[:3 * R * MS].view(3, R, MS), pos=d[3 * R * MS:], bt=bt, sig=sig, S=S)
    params["_dsv_meta"] = m
    return m


def _fp8(x2, w, sc, out, n):
    """FP8 GEMV over rows in chunks of 8 (the kernel's row limit). x2 [R, K] fp16, out [R, N]."""
    R = x2.shape[0]
    for r0 in range(0, R, 8):
        r1 = min(R, r0 + 8)
        F8().fp8_gemv(x2[r0:r1].view(1, r1 - r0, -1), w, sc, out[r0:r1], n)


def _bt(attn, params, rsg, B, kl):
    from exllamav3.util.tensor import get_for_device
    if "block_table" in params:
        return get_for_device(params, "block_table", attn.device)
    sbt = kl.slot_bt(rsg[0].cache.num_slots)
    return sbt[[rsg[i].slot for i in range(B)]]


def _bt_host(params, bt):
    """The block table on the host (the generator's own copy when it is a CPU tensor; else one readback per forward)."""
    v = params.get("block_table")
    if v is not None and v.device.type == "cpu":
        return v
    h = params.get("_dsv_bt_host")
    if h is None:
        h = params["_dsv_bt_host"] = bt.cpu()
    return h


COMPRESS_DEV = os.environ.get("DSV_COMPRESS_DEV", "1") == "1"
COMPRESS_CHECK = [int(os.environ.get("DSV_COMPRESS_CHECK", "0"))]


def _comp_prep(attn):
    """Weights of the device-side compressor (or None: the python path)."""
    cp = getattr(attn, "_dsv_comp", None)
    if cp is None:
        c = attn.compressor
        try:
            ws = [c.wkv.inner.weight] + ([c.wgate.inner.weight] if c.wgate is not None else [])
            ok = all(w.dim() == 2 and w.shape == (attn.hidden_size, attn.head_dim) for w in ws)
            wk = attn.idx_wk.inner.weight if attn.owns_index_keys else None
            ok = ok and (wk is None or wk.shape == (attn.head_dim, attn.index_head_dim))
        except AttributeError:
            ok = False
        if ok:
            cp = dict(W=torch.cat([w.t() for w in ws], 0).half().contiguous(),
                      wk=wk.half().contiguous() if wk is not None else None,
                      inv=attn.inv_freq_compress.float().contiguous())
        else:
            cp = False
            log(f"L{attn.layer_idx}: compressor weights are not plain fp16/bf16: python compressor")
        attn._dsv_comp = cp
    return cp or None


def _compress_dev(self, cp, x2, rsl, kl, bt, meta, params, rsg, layer_instance, S=1):
    R, D = x2.shape[1], x2.shape[2]
    B = R // S
    proj = torch.empty((R, cp["W"].shape[0]), dtype=torch.float, device=x2.device)
    from .moe import G
    xr = x2.view(R, D)
    for r0 in range(0, R, 8):
        G().gemv_h(xr[r0:r0 + 8], cp["W"], proj[r0:r0 + 8])
    c = self.compressor
    args = (proj, rsl.pend_kv, rsl.pend_gate, c.norm.weight, float(c.norm.rms_norm_eps), cp["wk"],
            self.idx_k_norm.weight if self.owns_index_keys else None,
            float(self.idx_k_norm.rms_norm_eps) if self.owns_index_keys else 0.0, cp["inv"], kl.pool_c, kl.pool_r,
            kl.pool_idx if self.owns_index_keys else None, bt, meta, int(self.compress_rate), int(kl.epp))
    if S > 1:                   # one call per row of each job, in order (the pending-row logic is sequential)
        for si in range(S):
            A().compress41(*args, S, si)
        return
    if COMPRESS_CHECK[0] > 0:
        COMPRESS_CHECK[0] -= 1
        ts = [t for t in (kl.pool_c, kl.pool_r, kl.pool_idx, rsl.pend_kv, rsl.pend_gate) if t is not None]
        before = [t.clone() for t in ts]
        A().compress41(*args)
        dev_out = [t.clone() for t in ts]
        for t, b0 in zip(ts, before):
            t.copy_(b0)
        ckv, cg = c.project(x2, params)
        bth = _bt_host(params, bt)
        for b in range(B):
            rs = rsg[b]
            _compress_one(self, ckv[:, b:b + 1], cg[:, b:b + 1] if cg is not None else None,
                          self._get_rsl(rs, layer_instance), rs.slot, kl, bth[b], rs.position)
        errs = [((d.float() - t.float()).abs().max() / (t.float().abs().max() + 1e-9)).item()
                for d, t in zip(dev_out, ts)]
        log(f"compress check L{self.layer_idx} pos {[rs.position for rs in rsg[:B]]}: rel err (pool_c, pool_r, "
            f"[pool_idx], pend...) {[f'{e:.2g}' for e in errs]}")
        return
    A().compress41(*args)


def _compress_one(attn, kvr, gr, rsl, slot, kl, bt_row_host, pos0):
    """DSV41Attention._compress_into_pool for one token: kvr / gr (1, 1, hd) fp32 projected rows."""
    m = attn.compress_rate
    comp = attn.compressor
    pend = pos0 % m
    kv_rows, gate_rows = kvr, gr
    if pend:
        kv_rows = torch.cat((rsl.pend_kv[slot:slot + 1, :pend], kv_rows), dim=1)
        if gate_rows is not None:
            gate_rows = torch.cat((rsl.pend_gate[slot:slot + 1, :pend], gate_rows), dim=1)
    n_rows = pend + 1
    n_groups = n_rows // m
    rem = n_rows - n_groups * m
    if n_groups:
        latent = comp.pool(kv_rows[:, :n_groups * m], None if gate_rows is None else gate_rows[:, :n_groups * m], {})
        g0 = pos0 // m
        epp = kl.epp
        page = int(bt_row_host[g0 // epp])      # host lookup: indexing with a device scalar syncs (and holds the GIL)
        sl = g0 % epp
        gpos = torch.full((1, 1), g0 * m, dtype=torch.int, device=kvr.device)
        from exllamav3.modules.dsv4 import _ext_rope
        if attn.owns_index_keys:
            k = attn.idx_k_norm.forward(attn.idx_wk.forward(latent, {}), {}, out_dtype=torch.half)
            k = k.view(1, 1, 1, attn.index_head_dim).contiguous()
            _ext_rope(k[..., -attn.rope_head_dim:], attn.inv_freq_compress, position_ids=gpos)
            kl.pool_idx[page, sl] = k.view(attn.index_head_dim).half()
        lat = latent.view(1, 1, 1, attn.head_dim).contiguous()
        _ext_rope(lat[..., -attn.rope_head_dim:], attn.inv_freq_compress, position_ids=gpos)
        lat = lat.view(attn.head_dim)
        kl.pool_c[page, sl] = lat[:kl.D_c].half()
        kl.pool_r[page, sl] = lat[kl.D_c:].half()
    if rsl.pend_kv is not None and rem:
        rsl.pend_kv[slot, :rem].copy_(kv_rows[0, n_groups * m:])
        rsl.pend_gate[slot, :rem].copy_(gate_rows[0, n_groups * m:])


_AR = {}


def _arange_rows(dev, B, kpad):
    t = _AR.get((dev, B, kpad))
    if t is None:
        t = _AR[(dev, B, kpad)] = torch.arange(kpad, dtype=torch.int32, device=dev).repeat(B, 1).contiguous()
    return t


def forward(self, x, params, out_dtype=None):
    B, S, D = x.shape
    prep = getattr(self, "_dsv_prep", None)
    if prep is None:
        prep = self._dsv_prep = _Prep(self) if self.num_q_heads else False
    if (not ENABLED or not prep or not prep.ok or S > MAX_S or B * S > MAX_ROWS or params.get("attn_mode") != "flash_attn"
            or self.tp_mode or getattr(self, "ref_quant", False)
            or any(k in params for k in ("capture", "quant_preserve", "ovr", "reconstruct"))):
        return _orig_forward(self, x, params, out_dtype)
    if CHECK[0] > 0 and S > 1:            # (single-token rows were validated against exllamav3 already)
        return _check(self, x, params, out_dtype)
    return _lean(self, prep, x, params, out_dtype)


def _lean(self, prep, x, params, out_dtype):
    B, S, D = x.shape
    R = B * S
    dev = x.device
    rsg = params["recurrent_states"]
    inst = params.get("layer_instance", 0)
    layer_instance = (self.layer_idx, inst)
    rsl = self._get_rsl(rsg[0], layer_instance)
    st = prepare_step(params, B, dev, S)
    xr = x.reshape(R, D)
    if xr.dtype != torch.half:
        xr = xr.half()
    xr = xr.contiguous()
    x2 = xr.view(1, R, D)
    # x fan + q_norm + q fan
    xo = torch.empty((R, prep.nx), dtype=torch.half, device=dev)
    _fp8(xr, prep.wx, prep.sx, xo, prep.nx)
    q_res = torch.empty((R, prep.nqa), dtype=torch.half, device=dev)
    A().rms_norm(xo[:, :prep.nqa], self.q_norm.weight, q_res, float(self.q_norm.rms_norm_eps),
                 float(self.q_norm.constant_bias))
    qo = torch.empty((R, prep.nq), dtype=torch.half, device=dev)
    _fp8(q_res, prep.wq, prep.sq, qo, prep.nq)
    H, hd = self.num_q_heads, self.head_dim
    # [rows, 1, heads, D] views whose token stride is the row stride (the rope kernel addresses row r at r * stride(1))
    q = qo.as_strided((R, 1, H, hd), (prep.nq, prep.nq, hd, 1), qo.storage_offset())
    kv = xo.as_strided((R, 1, 1, hd), (prep.nx, prep.nx, hd, 1), xo.storage_offset() + prep.off_kv)
    from .ops import _rope_lib
    _rope_lib().rope(q, q, kv, kv, prep.inv_freq, 0, st["pos"], None, 1, 1.0,
                     self.q_ones if self.q_head_norm else None, self.kv_norm_w, float(self.rms_norm_eps), 0.0,
                     int(hd - self.rope_head_dim))
    kv2 = kv.view(R, hd)
    # pools / indices
    kpad = -(-self.index_topk // 32) * 32
    if prep.is41:
        m = self.compress_rate
        meta = st["meta"][1 if m == 1 else 2]
        if self.is_kv_source:
            kl = rsg[0].cache.layers[layer_instance]
            bt = st["bt"]
            cp = _comp_prep(self) if COMPRESS_DEV else None
            if cp is not None:
                _compress_dev(self, cp, x2, rsl, kl, bt, meta, params, rsg, layer_instance, S)
            else:
                assert S == 1, "multi-row decode needs the device compressor"
                ckv, cg = self.compressor.project(x2, params)                  # (1, B, hd) fp32 each
                bth = _bt_host(params, bt)
                for b in range(B):
                    rs = rsg[b]
                    kvr = ckv[:, b:b + 1]
                    gr = cg[:, b:b + 1] if cg is not None else None
                    _compress_one(self, kvr, gr, self._get_rsl(rs, layer_instance), rs.slot, kl, bth[b], rs.position)
            params[("dsv_kl", self.layer_idx, inst)] = (kl, bt)
        else:
            kl, bt = params[("dsv_kl", self.kv_source_layer, inst)]
        if self.is_index_source:
            idx = _arange_rows(dev, R, kpad)
            big = [b for b in range(B) if (rsg[b].position + S) // m > self.index_topk]
            if big:
                idx = idx.clone()
                for b in big:
                    rs = rsg[b]
                    ec = (rs.position + S) // m
                    keys = self._gather_pool_idx(kl, bt[b:b + 1], ec)
                    ind, _ = self._v41_indexer(x2[:, b * S:(b + 1) * S], params, q_res[b * S:(b + 1) * S].view(1, S, -1),
                                               keys, ec, rs.position, S, ("dsv_job", b, inst))
                    idx[b * S:(b + 1) * S] = ind
            params[("dsv_idx", self.layer_idx, inst)] = idx
        else:
            idx = params[("dsv_idx", self.index_source_layer, inst)]
        pool_c, pool_r, epp = kl.pool_c.view(-1, kl.D_c), kl.pool_r.view(-1, kl.D_r), kl.epp
    else:
        meta = st["meta"][0]
        pool_c = pool_r = torch.empty(0, dtype=torch.half, device=dev)
        bt = idx = _arange_rows(dev, R, 32)
        epp = 1
    ws_ml, ws_acc = _ws(dev, R, idx.shape[1])
    out = torch.empty((self.o_groups, R, prep.hpg * hd), dtype=torch.half, device=dev)
    A().attn_decode(q.view(R, H, hd), kv2, rsl.ring, pool_c, pool_r, bt, idx, meta, self.sinks, prep.derot, out,
                    ws_ml, ws_acc, epp, float(self.sm_scale), prep.hpg, True, 0)
    return self._project_o_grouped(out.view(self.o_groups, B, S, prep.hpg * hd), params, out_dtype)


# ------------------------------------------------------------------------------------------------ DSpark drafter

_DMETA = {}


def dspark_forward(self, x, params, out_dtype=None):
    """DSparkV41Attention.forward on the lean kernels: the draft block (S rows per job) attends non-causally over
    [the job's last <= window main-kv rows (paged drafter cache) ++ the whole block] with sinks and de-rotation, in
    one attn_decode call for all jobs (exllamav3: a Triton call per job)."""
    B, S, D = x.shape
    R = B * S
    prep = getattr(self, "_dsv_prep", None)
    if prep is None:
        prep = self._dsv_prep = _Prep(self) if self.num_q_heads else False
    if not ENABLED or not prep or not prep.ok or R > MAX_ROWS:
        return _orig_dspark(self, x, params, out_dtype)
    dev = x.device
    from exllamav3.util.tensor import get_for_device
    kl = params["cache"].layers[(self.layer_idx, params.get("layer_instance", 0))]
    bt = get_for_device(params, "block_table", dev)
    if bt.dtype != torch.int32:
        bt = bt.int()
    bt = bt.contiguous()
    seql = params["cache_seqlens"]                            # host values
    key = ("dsp", tuple(int(seql[b]) for b in range(B)), S)
    st = params.get("_dsv_dmeta")
    if st is None or st[0] != key:
        ent = _DMETA.get((dev, R))
        if ent is None:
            ent = _DMETA[(dev, R)] = dict(i=0, pins=[torch.zeros(R * MS + R, dtype=torch.int32).pin_memory()
                                                     for _ in range(4)],
                                          d=torch.zeros(R * MS + R, dtype=torch.int32, device=dev))
        pin = ent["pins"][ent["i"] % 4]
        ent["i"] += 1
        hv = pin.numpy()
        for b in range(B):
            p0 = int(seql[b])
            hist = min(self.sliding_window, p0)
            for si in range(S):
                r = b * S + si
                hv[r * MS:(r + 1) * MS] = (p0 + si, p0 - hist, 0, 0, -1, 0, p0, b * S - p0, p0 + S - 1, r, b, 0)
                hv[R * MS + r] = p0 + si
        ent["d"].copy_(pin, non_blocking=True)
        st = (key, ent["d"][:R * MS].view(R, MS), ent["d"][R * MS:])
        params["_dsv_dmeta"] = st
    _, meta, pos = st
    xr = x.reshape(R, D)
    if xr.dtype != torch.half:
        xr = xr.half()
    xr = xr.contiguous()
    xo = torch.empty((R, prep.nx), dtype=torch.half, device=dev)
    _fp8(xr, prep.wx, prep.sx, xo, prep.nx)
    q_res = torch.empty((R, prep.nqa), dtype=torch.half, device=dev)
    A().rms_norm(xo[:, :prep.nqa], self.q_norm.weight, q_res, float(self.q_norm.rms_norm_eps),
                 float(self.q_norm.constant_bias))
    qo = torch.empty((R, prep.nq), dtype=torch.half, device=dev)
    _fp8(q_res, prep.wq, prep.sq, qo, prep.nq)
    H, hd = self.num_q_heads, self.head_dim
    q = qo.as_strided((R, 1, H, hd), (prep.nq, prep.nq, hd, 1), qo.storage_offset())
    kv = xo.as_strided((R, 1, 1, hd), (prep.nx, prep.nx, hd, 1), xo.storage_offset() + prep.off_kv)
    from .ops import _rope_lib
    # V4.1 drafter: no q head norm; the kv norm (weighted) is folded into the rope call as for the trunk
    _rope_lib().rope(q, q, kv, kv, prep.inv_freq, 0, pos, None, 1, 1.0, None, self.kv_norm.weight,
                     float(self.rms_norm_eps), 0.0, int(hd - self.rope_head_dim))
    kv2 = kv.view(R, hd)
    idx = _arange_rows(dev, R, 32)
    empty = torch.empty(0, dtype=torch.half, device=dev)
    ws_ml, ws_acc = _ws(dev, R, 32)
    out = torch.empty((self.o_groups, R, prep.hpg * hd), dtype=torch.half, device=dev)
    A().attn_decode(q.view(R, H, hd), kv2, kl.kv, empty, empty, bt, idx, meta, self.sinks, prep.derot, out,
                    ws_ml, ws_acc, 1, float(self.sm_scale), prep.hpg, False, 1)
    y = self._project_o_grouped(out.view(self.o_groups, B, S, prep.hpg * hd), params, out_dtype)
    if DSPARK_CHECK[0] > 0:
        DSPARK_CHECK[0] -= 1
        ref = _orig_dspark(self, x, params, out_dtype)
        e = ((y.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)).item()
        log(f"dspark check L{self.layer_idx} B={B} S={S} p0={[int(seql[b]) for b in range(B)]}: rel err {e:.3g}")
    return y


DSPARK_CHECK = [int(os.environ.get("DSV_DSPARK_CHECK", "0"))]


_orig_dspark = None


def _check(self, x, params, out_dtype):
    """Run the original path and the lean path on the same input and state; report the difference and continue with
    the original result (state written by the lean path is rolled back by running the original second)."""
    CHECK[0] -= 1
    rsg = params["recurrent_states"]
    B = x.shape[0]
    inst = params.get("layer_instance", 0)
    rsl = self._get_rsl(rsg[0], (self.layer_idx, inst))
    ring_save = rsl.ring.clone()
    kl = rsg[0].cache.layers[(self.layer_idx, inst)] if getattr(self, "is_kv_source", False) else None
    pools = [t.clone() for t in (kl.pool_c, kl.pool_r) + ((kl.pool_idx,) if kl.pool_idx is not None else ())] \
        if kl is not None else None
    pend = (rsl.pend_kv.clone(), rsl.pend_gate.clone()) if getattr(rsl, "pend_kv", None) is not None else None
    y = _lean(self, self._dsv_prep, x, params, out_dtype)
    rsl.ring.copy_(ring_save)
    if pools is not None:
        kl.pool_c.copy_(pools[0]); kl.pool_r.copy_(pools[1])
        if len(pools) > 2:
            kl.pool_idx.copy_(pools[2])
    if pend is not None:
        rsl.pend_kv.copy_(pend[0]); rsl.pend_gate.copy_(pend[1])
    for rs in rsg[:B]:
        rs.wshift = 0
    ref = _orig_forward(self, x, params, out_dtype)
    e = ((y.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-9)).item()
    log(f"check L{self.layer_idx} B={B} pos={[rs.position for rs in rsg[:B]]}: rel err {e:.3g}")
    return ref


def install():
    global _orig_dspark
    if ENABLED:
        DSV4Attention.forward = forward
        from exllamav3.modules.arch_specific.dspark_v41 import DSparkV41Attention
        _orig_dspark = DSparkV41Attention.forward
        if os.environ.get("DSV_DSPARK_LEAN", "1") == "1":
            DSparkV41Attention.forward = dspark_forward
