"""Decode steps for DeepSeek-V4.1-Flash on XPU graphs.

The eager decode step is CPU-bound (~100 ms to enqueue ~5,000 small launches for ~20 ms of GPU work). Here every
transformer block of a decode step (B jobs x 1 token) runs as

  graph A: hyper-connection site (mix + collapse + attn_norm), q_a / q_norm / q_b / wkv, rope at device positions
  eager  : the position-dependent attention core per job (compressor, indexer, dsa_attn, sliding-window ring)
  graph B: grouped o projection, hc apply, mlp site, MoE (router, experts, shared expert), hc apply

over one static residual-stream buffer updated in place. Graphs are captured per (block, B) on first use. The MoE is
no-sync (pointer tables and residency only change between steps), so it is position-independent too.
"""
from __future__ import annotations

import os

import torch

from exllamav3.modules.dsv41_hc import DSV41TransformerBlock, PRE_MIX_KEY
from exllamav3.modules.dsv41 import DSV41Attention
from exllamav3.modules.engram import EngramLayer
from exllamav3.modules.attention_fn.dsa_triton import dsa_attn
from exllamav3.constants import PAGE_SIZE
from exllamav3.cache.recurrent_util import advance_recurrent_states

from . import block as blockmod
from . import moe as moemod

ENABLED = os.environ.get("DSV_GRAPHS", "0") == "1"      # experimental: graph-B replay segfaults (see README)
USE_A = os.environ.get("DSV_GRAPH_A", "1") == "1"
USE_B = os.environ.get("DSV_GRAPH_B", "1") == "1"
B_LAYERS = None if not os.environ.get("DSV_GRAPH_B_LAYERS") else \
    {int(v) for v in os.environ["DSV_GRAPH_B_LAYERS"].split(",")}
B_DEBUG = os.environ.get("DSV_GRAPH_B_DEBUG", "0") == "1"
MERGE = os.environ.get("DSV_GRAPH_MERGE", "1") == "1"
MAX_B = 4


def log(*a):
    print(" -- dsv.graphs:", *a, flush=True)


class Step:
    def __init__(self, model):
        self.model = model
        self.blocks = {}          # (block index, B) -> dict(gA, gB, io...)
        self.static = {}          # B -> dict(xs, pre, pos)
        self.disabled = False
        self.ncap = 0

    # ------------------------------------------------------------------ static buffers
    def st(self, B, D, dev):
        s = self.static.get(B)
        if s is None:
            s = self.static[B] = dict(
                xs=torch.zeros((B, 1, 4, D), dtype=torch.float, device=dev),
                pre=torch.zeros((B, 1, 4), dtype=torch.float, device=dev),
                pos=torch.zeros((B,), dtype=torch.int32, device=dev),
                pos_pin=torch.zeros((B,), dtype=torch.int32).pin_memory(),
            )
        return s

    # ------------------------------------------------------------------ graph bodies
    @staticmethod
    def body_a(blk, s, params):
        """-> outputs dict (attn_pre, post, comb, y, q_res, q, kv)."""
        attn = blk.attn
        xs, pre = s["xs"], s["pre"]
        B = xs.shape[0]
        attn_pre, attn_post, attn_comb, y = blockmod.site(blk.attn_hc, blk.attn_norm, xs, pre)
        q_res = attn.q_norm.forward(attn.q_a.forward(y, params), params, out_dtype=torch.half)
        q = attn.q_b.forward(q_res, params).view(B, 1, attn.num_q_heads, attn.head_dim)
        kv = attn.wkv.forward(y, params).view(B, 1, 1, attn.head_dim)
        from .ops import _rope_lib
        _rope_lib().rope(q, q, kv, kv, attn._rope_type().float().contiguous(), 0, s["pos"], None, 1, 1.0,
                         attn.q_ones if attn.q_head_norm else None, attn.kv_norm_w, float(attn.rms_norm_eps), 0.0,
                         int(attn.head_dim - attn.rope_head_dim))
        return dict(attn_pre=attn_pre, post=attn_post, comb=attn_comb, y=y, q_res=q_res, q=q, kv=kv.view(B, 1, -1))

    @staticmethod
    def body_b(blk, s, a, o, params):
        attn = blk.attn
        xs = s["xs"]
        B = xs.shape[0]
        y = attn._project_o_grouped(o, params, None)                      # (B, 1, D) fp32
        blk.attn_hc.apply_(xs, y, a["post"], a["comb"], params)           # in place
        mlp_pre, mlp_post, mlp_comb, y2 = blockmod.site(blk.mlp_hc, blk.mlp_norm, xs, a["attn_pre"])
        y2 = blk.mlp.forward(y2, params)
        blk.mlp_hc.apply_(xs, y2, mlp_post, mlp_comb, params)
        s["pre"].copy_(mlp_pre)
        return dict(pending=getattr(blk.mlp, "_pending", None))

    # ------------------------------------------------------------------ eager attention core (one job)
    @staticmethod
    def core(blk, y, q_res, q, kv, params, rs, bt_row):
        """Attention core of one job (seq 1): y (1, 1, D) normed input, q_res (1, 1, R), q (1, 1, H, Dh) and
        kv (1, 1, Dh) roped. Returns o (G, 1, hpg * Dh) fp16 (group-major, de-rotated)."""
        attn = blk.attn
        inst = params.get("layer_instance", 0)
        layer_instance = (attn.layer_idx, inst)
        rsl = attn._get_rsl(rs, layer_instance)
        if not isinstance(attn, DSV41Attention):
            pre = dict(q_res=q_res, q=q, kv=kv, comp_kv=None, comp_gate=None, idx_kv=None, idx_gate=None, q_idx=None)
            return attn._forward_cached_one(y, params, rs, rsl, None, pre=pre, return_o=True, kl=None, bt_row=None)
        device = y.device
        pos0 = rs.position
        slot = rs.slot
        m = attn.compress_rate
        seq = 1
        ec = (pos0 + seq) // m
        if attn.is_kv_source:
            kl = rs.cache.layers[layer_instance]
            attn._compress_into_pool(y, params, rsl, slot, kl, bt_row, pos0, seq)
            params[("v41_kl", attn.layer_idx, inst)] = (kl, bt_row)
        else:
            kl, bt_row = params[("v41_kl", attn.kv_source_layer, inst)]
        indices, k_len = None, 0
        if ec > 0:
            if attn.is_index_source:
                keys = attn._gather_pool_idx(kl, bt_row, ec)
                indices, k_len = attn._v41_indexer(y, params, q_res, keys, ec, pos0, seq, inst)
                params[("v41_idx", attn.layer_idx, inst)] = (indices, k_len)
            elif attn.index_source_layer is not None:
                indices, k_len = params[("v41_idx", attn.index_source_layer, inst)]
        if indices is None:
            indices = torch.full((seq, 32), -1, dtype=torch.int32, device=device)
            k_len = 0
        w = attn.sliding_window
        n_prev = min(w - 1, pos0 - rs.window_beg, pos0)
        win_floor = pos0 - n_prev
        ring = rsl.ring[slot]
        ring_beg = rs.window_beg
        hpg = attn.num_q_heads // attn.o_groups
        out = dsa_attn(
            q[0].contiguous(), kl.pool_c_view(), kl.pool_r, bt_row, sinks=attn.sinks,
            ring=ring, kv_chunk=kv[0], win_len=w, win_floor=win_floor, ring_beg=ring_beg,
            indices=indices, k_len=k_len, pool_len=ec, q_pos0=pos0, compress_rate=m,
            scale=attn.sm_scale, derot_inv_freq=attn._rope_type_neg(), groups=attn.o_groups,
            group_major=True, page_size=kl.epp, qc=kl.qc(),
            out=torch.empty((attn.o_groups, seq, hpg * attn.head_dim), dtype=torch.half, device=device),
        )
        offset = pos0 - rs.window_beg
        pos_end = pos0 + seq
        if offset + seq <= rsl.ring_rows:
            ring[offset: offset + seq].copy_(kv[0])
        elif seq < PAGE_SIZE:
            need = offset + seq - rsl.ring_rows
            shift = -(-need // PAGE_SIZE) * PAGE_SIZE
            ring[:rsl.ring_rows - shift].copy_(ring[shift:].clone())
            rs.wshift = shift
            ring[offset - shift: offset - shift + seq].copy_(kv[0])
        return out

    # ------------------------------------------------------------------ capture
    def capture(self, fn):
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        with torch.xpu.graph(g):
            out = fn()
        torch.xpu.synchronize()
        self.ncap += 1
        return g, out

    def run_block(self, bi, blk, s, params, rsg, bt):
        B = s["xs"].shape[0]
        key = (bi, B)
        ent = self.blocks.get(key)
        if ent is None:
            # warm-up eagerly once (allocates scratch buffers outside the graph), then capture
            a = self.body_a(blk, s, params)
            if USE_A:
                gA, aout = self.capture(lambda: self.body_a(blk, s, params))
            else:
                gA, aout = None, None
            ent = dict(gA=gA, a=aout)
            self.blocks[key] = ent
            o_buf = torch.empty((blk.attn.o_groups, B, 1, (blk.attn.num_q_heads // blk.attn.o_groups) * blk.attn.head_dim),
                                dtype=torch.half, device=s["xs"].device)
            ent["o"] = o_buf
            ent["gB"] = None
        if ent["gA"] is not None:
            a = ent["a"]
            ent["gA"].replay()
        else:
            a = self.body_a(blk, s, params)
        for i in range(B):
            o = self.core(blk, a["y"][i:i + 1], a["q_res"][i:i + 1], a["q"][i:i + 1], a["kv"][i:i + 1], params,
                          rsg[i], bt[i:i + 1] if bt is not None else None)
            ent["o"][:, i].copy_(o)
        if not USE_B or (B_LAYERS is not None and bi not in B_LAYERS):
            self.body_b(blk, s, a, ent["o"], params)
            return
        if ent["gB"] is None:
            xs_save, pre_save = s["xs"].clone(), s["pre"].clone()
            self.body_b(blk, s, a, ent["o"], params)                     # warm-up (eager) ...
            s["xs"].copy_(xs_save)
            s["pre"].copy_(pre_save)
            gB, bout = self.capture(lambda: self.body_b(blk, s, a, ent["o"], params))
            s["xs"].copy_(xs_save)
            s["pre"].copy_(pre_save)
            ent["gB"], ent["b"] = gB, bout
        if B_DEBUG:
            torch.xpu.synchronize()
            log(f"replay B block {bi} (step {getattr(self, 'steps', 0)})")
        ent["gB"].replay()
        if B_DEBUG:
            torch.xpu.synchronize()
        pend = ent["b"]["pending"]
        if pend is not None:
            blk.mlp._pending = pend

    # ------------------------------------------------------------------ merged schedule
    def _ent(self, bi, blk, B, dev):
        key = (bi, B)
        ent = self.blocks.get(key)
        if ent is None:
            hd = (blk.attn.num_q_heads // blk.attn.o_groups) * blk.attn.head_dim
            ent = self.blocks[key] = dict(o=torch.empty((blk.attn.o_groups, B, 1, hd), dtype=torch.half, device=dev))
        return ent

    def _capture_eager_first(self, s, eager_fn, cap_fn):
        """Run eager_fn once (allocations, lazy init) on saved state, restore, capture cap_fn, restore."""
        xs_save, pre_save = s["xs"].clone(), s["pre"].clone()
        eager_fn()
        s["xs"].copy_(xs_save)
        s["pre"].copy_(pre_save)
        g, out = self.capture(cap_fn)
        s["xs"].copy_(xs_save)
        s["pre"].copy_(pre_save)
        return g, out

    def forward(self, input_ids, params):
        model = self.model
        B, S = input_ids.shape
        x = model.prepare_inputs(input_ids, params)
        rsg = params["recurrent_states"]
        bt = params.get("block_table")
        s = None
        mods = model.fwd_modules
        nmods = len(mods)
        bi = 0
        a_next = None          # A outputs of the current block, produced by the previous merged graph
        j = 0
        while j < nmods:
            module, instance, idx = mods[j]
            params["layer_instance"] = instance
            if isinstance(module, DSV41TransformerBlock):
                if s is None:
                    dev = module.attn_norm.weight.device
                    s = self.st(B, x.shape[-1], dev)
                    s["xs"].copy_(x if x.device == dev else x.to(dev))
                    s["pre"].zero_()
                    s["pre"][..., 0] = 1.0
                    for i, r in enumerate(rsg):
                        s["pos_pin"][i] = r.position
                    s["pos"].copy_(s["pos_pin"], non_blocking=True)
                    if bt is not None and bt.device != dev:
                        bt = bt.to(dev)
                blk = module
                ent = self._ent(bi, blk, B, s["xs"].device)
                # graph A (alone) unless the previous merged graph already produced this block's A outputs
                if a_next is None:
                    if "gA" not in ent:
                        ent["gA"], ent["a"] = self._capture_eager_first(
                            s, lambda: self.body_a(blk, s, params), lambda: self.body_a(blk, s, params))
                    ent["gA"].replay()
                    a = ent["a"]
                else:
                    a = a_next
                    a_next = None
                for i in range(B):
                    o = self.core(blk, a["y"][i:i + 1], a["q_res"][i:i + 1], a["q"][i:i + 1], a["kv"][i:i + 1],
                                  params, rsg[i], bt[i:i + 1] if bt is not None else None)
                    ent["o"][:, i].copy_(o)
                nxt = mods[j + 1][0] if j + 1 < nmods else None
                if MERGE and isinstance(nxt, DSV41TransformerBlock):
                    if "gBA" not in ent:
                        def both(blk=blk, nxt=nxt, a=a, o=ent["o"]):
                            b = self.body_b(blk, s, a, o, params)
                            an = self.body_a(nxt, s, params)
                            return dict(b=b, a=an)
                        ent["gBA"], ent["ba"] = self._capture_eager_first(s, both, both)
                    if B_DEBUG:
                        torch.xpu.synchronize()
                        log(f"replay BA block {bi} (step {getattr(self, 'steps', 0)})")
                    ent["gBA"].replay()
                    if B_DEBUG:
                        torch.xpu.synchronize()
                    out = ent["ba"]
                    a_next = out["a"]
                    pend = out["b"]["pending"]
                else:
                    if "gB" not in ent:
                        ent["gB"], ent["b"] = self._capture_eager_first(
                            s, lambda: self.body_b(blk, s, a, ent["o"], params),
                            lambda: self.body_b(blk, s, a, ent["o"], params))
                    ent["gB"].replay()
                    pend = ent["b"]["pending"]
                if pend is not None:
                    blk.mlp._pending = pend
                bi += 1
                x = s["xs"]
                j += 1
                continue
            if isinstance(module, EngramLayer) and s is not None:
                y = module.forward(s["xs"], params)
                if y is not s["xs"]:
                    s["xs"].copy_(y)
                x = s["xs"]
                j += 1
                continue
            if s is not None and PRE_MIX_KEY not in params:
                params[PRE_MIX_KEY] = s["pre"]
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            j += 1
        advance_recurrent_states(input_ids, params, model)
        self.steps = getattr(self, "steps", 0) + 1
        if self.steps <= 3 or self.steps % 50 == 0:
            log(f"step {self.steps} (B={B}) done, {self.ncap} graphs captured")
        return x


STEP: Step | None = None


def eligible(model, input_ids, params):
    if not ENABLED or STEP is None or STEP.disabled:
        return False
    if params.get("attn_mode") != "flash_attn" or params.get("recurrent_history") or params.get("export_state_layers"):
        return False
    B, S = input_ids.shape
    return S == 1 and B <= MAX_B and moemod.TIER is not None and moemod.TIER.nosync


def install(model):
    global STEP
    STEP = Step(model)
