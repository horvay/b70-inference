"""Load DeepSeek-V4.1-Flash (SAGE EXL3 1.62 bpw) on the B70: exllamav3 (vcruz305 fork) + the XPU shim + the expert tier.

  from dsv_xpu.runner import load
  model, config, cache, tokenizer, tier = load("/models", "/nvx")
"""
import os

for k, v in (("EXL3_XPU", "1"), ("TRITON_F32_DEFAULT", "ieee"), ("EXL3_BC_DSA", "0"), ("EXL3_DSV4_BATCH_EAGER", "0")):
    os.environ.setdefault(k, v)

from glm_xpu import compat  # noqa: E402

compat.install()

import time  # noqa: E402
import faulthandler, signal  # noqa: E402

faulthandler.register(signal.SIGUSR1, all_threads=True)     # kill -USR1 <pid>: dump every thread's stack

import torch  # noqa: E402


def log(*a):
    print(" -- dsv:", *a, flush=True)


def _fix_head(model, dev):
    """The output head is EXL3 K=8 (no ESIMD kernel for K=8): dequantize it once into an fp16 weight."""
    from dsv_xpu import exl3_tri
    head = model.modules[model.logit_layer_idx]
    inner = head.inner
    if getattr(inner, "quant_type", None) == "fp16" or not hasattr(inner, "trellis"):
        return
    K = int(inner.K)
    tr = inner.trellis
    kdim, n = tr.shape[0] * 16, tr.shape[1] * 16
    Hm = exl3_tri.had128(dev).float()
    Wh = torch.empty((kdim, n), dtype=torch.half, device=dev)
    step = 16384                                                         # columns per slice (tiles: 1024)
    suh = inner.suh.float().view(-1, 1)
    for c0 in range(0, n, step):
        c1 = min(n, c0 + step)
        W = exl3_tri.dequant(tr[:, c0 // 16:c1 // 16].contiguous(), K).float()      # W_hat slice
        w = c1 - c0
        W = (Hm @ W.view(kdim // 128, 128, w)).view(kdim, w)             # left Hadamard per 128-row block
        W = (W.view(kdim, w // 128, 128) @ Hm).view(kdim, w)             # right Hadamard per 128-col block
        Wh[:, c0:c1] = (W * suh * inner.svh[c0:c1].float().view(1, -1)).half()
        del W
    out_features = head.out_features

    def fwd(x, params, out_dtype=None):
        shp = x.shape[:-1] + (out_features,)
        y = torch.matmul(x.view(-1, x.shape[-1]).half(), Wh[:, :out_features])
        y = y.view(shp)
        return y.float() if (out_dtype or torch.float) == torch.float else y.to(out_dtype)

    inner.trellis = None
    head.inner = type("HeadFP16", (), {"forward": staticmethod(fwd), "quant_type": "fp16", "weight": Wh})()
    torch.xpu.empty_cache()
    log(f"head: K={K} trellis dequantized to fp16 ({Wh.numel() * 2 / 2**30:.2f} GiB)")


def load(model_dir="/models", store_dir="/nvx", cache_tokens=65536, max_batch_size=3, vram_expert_gb=None,
         ram_expert_gb=24.0, vram_reserve_gb=3.0, draft=None):
    """draft: load the DSpark drafter (mtp component) for speculative decoding (default: DSV_DRAFT=1); it is then
    model._dsv_draft = (draft_model, draft_cache), for Generator(draft_model=..., draft_cache=...)."""
    if draft is None:
        draft = os.environ.get("DSV_DRAFT", "1") == "1"     # DSpark on: +24% on one topic with block boost 0.3, exact acceptance
    from exllamav3 import Config, Model, Cache, Tokenizer
    from dsv_xpu import moe
    from dsv_xpu.moe import Tier

    from dsv_xpu import fp8
    dev = torch.device("xpu", 0)
    moe.install(None, None)
    fp8.install()
    from dsv_xpu import block
    block.install()
    from dsv_xpu import attn
    attn.install()
    if os.environ.get("DSV_NO_ENGRAM") == "1":
        # debug only: skip the Engram layers (their tables are 189 GB on other drives)
        from exllamav3.modules.engram import EngramLayer
        EngramLayer.load = lambda self, device, **kw: setattr(self, "device", device)
        EngramLayer.forward = lambda self, x, params, out_dtype=None: x
        log("Engram layers DISABLED (DSV_NO_ENGRAM=1)")
    t0 = time.time()
    config = Config.from_directory(model_dir, load_method="python")
    config.infer_params.no_reconstruct = True
    model = Model.from_config(config)
    dmodel = dcache = None
    max_history = 0
    if draft:
        dmodel = Model.from_config(config, component="mtp")
        max_history = dmodel.caps.get("default_draft_size", 5)          # verify rows the target may roll back
    cache = Cache(model, max_num_tokens=cache_tokens, max_batch_size=max_batch_size, max_history=max_history) \
        if cache_tokens else None
    if draft:
        dcache = Cache(dmodel, max_num_tokens=cache_tokens, max_batch_size=max_batch_size)
    model.load(device=dev, progressbar=True)
    _fix_head(model, dev)
    if draft:
        dmodel.load(device=dev, progressbar=False)
        log(f"DSpark drafter loaded ({dmodel.num_mtp_layers} blocks, draft size {max_history})")
    torch.xpu.synchronize()
    free, total = torch.xpu.mem_get_info(0)
    log(f"model loaded in {time.time() - t0:.0f} s; VRAM free {free / 2**30:.2f} of {total / 2**30:.2f} GiB, "
        f"allocated {torch.xpu.memory_allocated(0) / 2**30:.2f} GiB")
    if vram_expert_gb is None:
        if draft:
            vram_reserve_gb += float(os.environ.get("DSV_DRAFT_RESERVE_GB", "1.5"))   # verify / draft scratch
        vram_expert_gb = max(1.0, free / 2**30 - vram_reserve_gb - 0.7)      # 0.7: the expert vectors
    tier = Tier(store_dir, dev, int(vram_expert_gb * 2**30), int(ram_expert_gb * 2**30))
    moe.TIER = tier
    stage_gb = float(os.environ.get("DSV_STAGE_GB", "1.9"))      # per area (two areas)
    if stage_gb > 0:
        moe.STAGER = moe.Stager(tier, int(stage_gb * 2**30))
    mods = moe.moe_modules(model)
    moe.link(mods)
    model._dsv_moes = mods
    if draft:
        dmods = moe.moe_modules(dmodel)
        moe.link(dmods)                    # the drafter's layers predict each other, not the main model's
        model._dsv_draft = (dmodel, dcache)
        tier.draft_layers = [m._li for m in dmods]
    orig_fwd, orig_pre = model.forward, model.prefill

    from dsv_xpu import graphs
    graphs.install(model)

    if os.environ.get("DSV_ASYNC", "1") == "1":
        import sys
        from dsv_xpu import pager
        moe.PAGER = pager.Pager(tier)
        # the pager thread needs the GIL for its bookkeeping while the main thread enqueues: switch often
        sys.setswitchinterval(float(os.environ.get("DSV_SWITCH_S", "0.0002")))

    warm_b = set()
    GRAPHS_ON = os.environ.get("DSV_GRAPH", "1") == "1"
    from exllamav3.modules.dsv4 import DSV4Attention as _A4
    from exllamav3.modules.dsv41_hc import DSV41TransformerBlock as _Blk
    attn.ATTNS[:] = [m.attn for m in model.modules if isinstance(m, _Blk) and isinstance(m.attn, _A4)]

    def forward(input_ids, params=None):
        P = moe.PAGER
        if P is not None:
            P.wait_idle()            # the previous step's layers are done; the tier is the main thread's again
        tier.tick()
        if tier.nosync:
            moe.drain(mods)          # previous step's picks -> bookkeeping, fetches, tables for this step
        else:
            tier.reap()
        if params is not None and graphs.eligible(model, input_ids, params):
            try:
                return graphs.STEP.forward(input_ids, params)
            except Exception as e:      # a failed capture can leave the XPU graph pool broken: never retry
                graphs.STEP.disabled = True
                log(f"XPU graphs disabled after an error: {type(e).__name__}: {e}")
                raise
        # exact-routing decode step: layers sync with the pager thread instead of waiting on the host
        use_async = (P is not None and moe.ZERO_COPY and not tier.nosync and not moe.IN_PREFILL[0] and input_ids.dim() == 2
                     and input_ids.shape[-1] <= 8 and input_ids.shape[0] * input_ids.shape[-1] <= 8)   # (verify blocks too)
        if use_async:
            # the first decode step of a shape (batch size, and whether the indexers run their top-k: past 512
            # compressed entries) JIT-compiles kernels inside launch calls that hold the GIL for seconds, and the
            # pager could not answer the sync points in time: run it synchronously
            rsg = (params or {}).get("recurrent_states") or []
            pos = [getattr(r, "position", 0) for r in rsg]
            S_ = input_ids.shape[-1]
            key = (input_ids.shape[0], any(p + S_ > 512 for p in pos), any((p + S_) // 2 > 512 for p in pos), S_)
            if key not in warm_b:
                out = orig_fwd(input_ids, params)
                warm_b.add(key)
                return out
        if not use_async:
            return orig_fwd(input_ids, params)
        moe.ASYNC[0] = P.begin_step()
        B = input_ids.shape[0]
        use_graph = GRAPHS_ON and not key[1] and not key[2] and key[3] == 1 and not block.GRAPH_FAIL[0]
        if use_graph:
            # graph step: job state into the fixed buffers, every layer handed to the pager up front, then the blocks
            # replay their graphs (recorded on first use)
            attn.prepare_step(params, B, dev)
            boost = tier.boost > 0
            items = []
            for mm in mods:
                items.append((mm._li, B, B * mm.num_experts_per_tok, tuple(m2._li for m2 in moe.ahead_mods(mm)),
                              boost))
            P.push_all(items)
            moe.GRAPH[0] = (boost, moe.AHEAD, moe.AHEAD_DEPTH[0])
        if os.environ.get("DSV_PAGER_DEBUG") == "1":
            faulthandler.dump_traceback_later(float(os.environ.get("DSV_PAGER_DEBUG_S", "1")), repeat=False)  # a stalled step: every stack
        try:
            out = orig_fwd(input_ids, params)
        except BaseException:
            P.abort()
            raise
        finally:
            moe.ASYNC[0] = 0
            moe.GRAPH[0] = None
        if os.environ.get("DSV_PAGER_DEBUG") == "1":
            faulthandler.cancel_dump_traceback_later()
        # wait (GIL released) until the pager has served every layer of this step: the caller's next GPU sync (the
        # sampler's readback) may hold the GIL, and the pager thread needs it to release the GPU's sync points
        P.wait_idle()
        return out

    import triton
    PREFILL_NATIVE = os.environ.get("DSV_PREFILL_NATIVE", "1") == "1" and hasattr(triton.knobs, "intel")

    def prefill(input_ids, params=None):
        if moe.PAGER is not None:
            moe.PAGER.wait_idle()
        if tier.nosync:
            moe.drain(mods)
        moe.IN_PREFILL[0] = True
        if PREFILL_NATIVE:
            # prompt kernels as native GPU binaries in the Triton cache (no per-process driver JIT of the SPIR-V: ~45 s
            # on a new process's first prompt). Not for decode: a kernel variant first met inside a pager step would
            # compile with ocloc for seconds while the GPU waits at a sync point
            triton.knobs.intel.gen_native_code = True
        t0 = time.time()
        try:
            return orig_pre(input_ids, params)
        finally:
            moe.IN_PREFILL[0] = False
            if PREFILL_NATIVE:
                triton.knobs.intel.gen_native_code = False
            if input_ids.shape[-1] >= 512:
                torch.xpu.synchronize()
                st = moe.PREFILL_STATS
                log(f"prefill chunk {tuple(input_ids.shape)} in {time.time() - t0:.1f} s (MoE {st['moe']:.1f} s, "
                    f"of which waiting for staged experts {st['wait']:.1f} s; {int(st.get('scratch', 0))} pieces "
                    f"read synchronously; probe: enqueue {st.get('enqueue', 0):.1f} s, GPU tail {st.get('gpu_tail', 0):.1f} s)")
                for k in st:
                    st[k] = 0.0

    model.forward, model.prefill = forward, prefill
    if tier.profile_save:
        import atexit
        atexit.register(lambda: tier.save_profile())      # the next start warms from this session's use
    tokenizer = Tokenizer.from_config(config)
    return model, config, cache, tokenizer, tier
