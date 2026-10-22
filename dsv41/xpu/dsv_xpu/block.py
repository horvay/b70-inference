"""DSV41TransformerBlock.forward for decode rows with the fused hyper-connection site kernel (dsv_hc.hc41_site):
mix (pre, post, comb) + collapse with the previous site's pre + the block's RMSNorm in two launches per site."""
import os

import torch

from exllamav3.modules.dsv41_hc import DSV41TransformerBlock, PRE_MIX_KEY, pending_pre_mix

MAX_R = int(os.environ.get("DSV_HC_MAX_R", "16"))
CHECK = [int(os.environ.get("DSV_HC_CHECK", "0"))]
_H = None
_orig = DSV41TransformerBlock.forward


def H():
    global _H
    if _H is None:
        torch.ops.load_library(os.path.join(os.path.dirname(__file__), "..", "lib", "dsv_hc.so"))
        _H = torch.ops.dsv_hc
    return _H


def site(hc, norm, x, pre_in):
    """-> (pre [R,4], post [b,s,4], comb [b,s,4,4], y [b,s,D] fp16 normed collapse with pre_in)."""
    b, s, Hm, D = x.shape
    R = b * s
    dev = x.device
    pre = torch.empty((R, 4), dtype=torch.float, device=dev)
    post = torch.empty((R, 4), dtype=torch.float, device=dev)
    comb = torch.empty((R, 4, 4), dtype=torch.float, device=dev)
    y = torch.empty((R, D), dtype=torch.half, device=dev)
    part = torch.empty((R * (4 * D // 512) * 25,), dtype=torch.float, device=dev)
    H().hc41_site(x, hc.fn, hc.base, hc.scale, pre_in.reshape(R, 4).contiguous() if pre_in is not None else None,
                  norm.weight, pre, post, comb, y, part, hc.sinkhorn_iters, hc.rms_eps, hc.hc_eps, norm.rms_norm_eps)
    return pre.view(b, s, 4), post.view(b, s, 4), comb.view(b, s, 4, 4), y.view(b, s, D)


def site_apply(hc, norm, x, ay, apost, acomb, pre_in):
    """x <- apost (x) ay + acomb^T x (the previous site's residual update, in place), then site() on the new x."""
    b, s, Hm, D = x.shape
    R = b * s
    dev = x.device
    pre = torch.empty((R, 4), dtype=torch.float, device=dev)
    post = torch.empty((R, 4), dtype=torch.float, device=dev)
    comb = torch.empty((R, 4, 4), dtype=torch.float, device=dev)
    y = torch.empty((R, D), dtype=torch.half, device=dev)
    part = torch.empty((R * (D // 128) * 25,), dtype=torch.float, device=dev)
    H().hc41_apply_site(x, ay.reshape(R, D), apost.reshape(R, 4), acomb.reshape(R, 4, 4), hc.fn, hc.base, hc.scale,
                        pre_in.reshape(R, 4).contiguous() if pre_in is not None else None, norm.weight, pre, post,
                        comb, y, part, hc.sinkhorn_iters, hc.rms_eps, hc.hc_eps, norm.rms_norm_eps)
    return pre.view(b, s, 4), post.view(b, s, 4), comb.view(b, s, 4, 4), y.view(b, s, D)


FUSE_APPLY = os.environ.get("DSV_HC_FUSE_APPLY", "1") == "1"


def forward(self, x, params, out_dtype=None):
    b, s, Hm, D = x.shape
    if b * s > MAX_R or Hm != 4 or x.dtype != torch.float or not x.is_contiguous() \
            or params.get("prefill") or self.attn_norm.constant_bias != 0.0 or self.mlp_norm.constant_bias != 0.0:
        return _orig(self, x, params, out_dtype)
    export = params.get("export_state_layers")
    if export and self.layer_idx in export and params.get("layer_instance", 0) == 0:
        # DSpark drafter taps: the stream mean at the block's INPUT (as exllamav3's DSV41TransformerBlock)
        states = params.get("export_states")
        if not states:
            states = params["export_states"] = []
        states.append(x.mean(dim=2).clamp(-65504.0, 65504.0).half())
    pre_in = params.get(PRE_MIX_KEY)
    if pre_in is not None and pre_in.shape[:2] != x.shape[:2]:
        pre_in = None
    if CHECK[0] > 0:
        CHECK[0] -= 1
        ref_pre, ref_post, ref_comb = self.attn_hc.mix_full(x)
        from exllamav3.modules.dsv41_hc import collapse
        ref_y = self.attn_norm.forward(collapse(x, pending_pre_mix(params, x, Hm)).half(), params, out_dtype=torch.half)
    attn_pre, attn_post, attn_comb, y = site(self.attn_hc, self.attn_norm, x, pre_in)
    if CHECK[0] >= 0 and "ref_pre" in locals():
        e = lambda a, r: ((a.float() - r.float()).abs().max() / (r.float().abs().max() + 1e-12)).item()
        print(f" -- hc check L{self.layer_idx}: pre {e(attn_pre, ref_pre):.2g} post {e(attn_post, ref_post):.2g} "
              f"comb {e(attn_comb, ref_comb):.2g} y {e(y, ref_y):.2g}", flush=True)
    y = self.attn.forward(y, params)
    if FUSE_APPLY and y.is_contiguous() and y.dtype in (torch.float, torch.half):
        mlp_pre, mlp_post, mlp_comb, y = site_apply(self.mlp_hc, self.mlp_norm, x, y, attn_post, attn_comb, attn_pre)
    else:
        x = self.attn_hc.apply_(x, y, attn_post, attn_comb, params)
        mlp_pre, mlp_post, mlp_comb, y = site(self.mlp_hc, self.mlp_norm, x, attn_pre)
    y = self.mlp.forward(y, params)
    if FUSE_APPLY and y.is_contiguous() and y.dtype in (torch.float, torch.half):
        from .attn import A
        R = b * s
        A().hc_apply(x.view(R, 4, D), y.view(R, D), mlp_post.reshape(R, 4), mlp_comb.reshape(R, 4, 4))
    else:
        x = self.mlp_hc.apply_(x, y, mlp_post, mlp_comb, params)
    params[PRE_MIX_KEY] = mlp_pre
    return x


# ------------------------------------------------------------------------------------------------ XPU graphs
# A decode step's block (attention, MoE with its pager sync point, hyper-connections) recorded once per key and replayed
# with one call. The residual streams and the pending pre-mix live in fixed per-batch-size buffers; everything the
# kernels read per step (positions, ring rows, block table, step id) is in fixed device buffers set before the step.

GRAPHS = {}
STATIC = {}
CAP_STREAM = [None]
GRAPH_FAIL = [False]


def _static(dev, B, D):
    st = STATIC.get((dev, B))
    if st is None:
        st = STATIC[(dev, B)] = dict(x=torch.zeros((B, 1, 4, D), dtype=torch.float, device=dev),
                                     pre=torch.zeros((B, 1, 4), dtype=torch.float, device=dev))
    return st


def _capture(dev, fn):
    """Record fn's GPU work into an XPU graph without a host sync (torch.xpu.graph synchronizes on entry, which would
    wait on the GPU while it is parked at a sync point)."""
    if CAP_STREAM[0] is None:
        CAP_STREAM[0] = torch.xpu.Stream(dev)
    s = CAP_STREAM[0]
    cur = torch.xpu.current_stream(dev)
    s.wait_stream(cur)
    g = torch.xpu.XPUGraph()
    with torch.xpu.stream(s):
        g.capture_begin()
        try:
            fn()
        finally:
            g.capture_end()
    cur.wait_stream(s)
    return g


def graph_forward(self, x, params, out_dtype=None):
    from . import moe as moemod
    b, s_, Hm, D = x.shape
    st = _static(x.device, b, D)
    xs, pre = st["x"], st["pre"]
    if x is not xs:
        xs.copy_(x)
    pin = params.get(PRE_MIX_KEY)
    if pin is None:
        pre.zero_()
        pre[..., 0] = 1.0
    elif pin is not pre:
        pre.copy_(pin.reshape(pre.shape))
    key = (self.layer_idx, b, moemod.GRAPH[0])
    g = GRAPHS.get(key)
    if g is None and not GRAPH_FAIL[0]:
        def body():
            params[PRE_MIX_KEY] = pre
            forward(self, xs, params, out_dtype)
            pre.copy_(params[PRE_MIX_KEY].reshape(pre.shape))
        try:
            g = GRAPHS[key] = _capture(x.device, body)
        except Exception as e:
            GRAPH_FAIL[0] = True
            print(f" -- dsv.graphs: capture of block {self.layer_idx} failed, running eagerly from now on: "
                  f"{type(e).__name__}: {e}", flush=True)
            g = None
    if g is None:
        params[PRE_MIX_KEY] = pre
        forward(self, xs, params, out_dtype)
        pre.copy_(params[PRE_MIX_KEY].reshape(pre.shape))
    else:
        g.replay()
    params[PRE_MIX_KEY] = pre
    return xs


def dispatch(self, x, params, out_dtype=None):
    from . import moe as moemod
    if moemod.GRAPH[0] is not None and x.dim() == 4 and x.shape[0] * x.shape[1] <= MAX_R and x.dtype == torch.float \
            and not params.get("prefill") and not params.get("export_state_layers"):
        return graph_forward(self, x, params, out_dtype)
    return forward(self, x, params, out_dtype)


def install():
    if os.environ.get("DSV_HC_FUSED", "1") == "1":
        DSV41TransformerBlock.forward = dispatch
