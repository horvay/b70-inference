"""Development server: load GLM-5.3-Flash on the B70 once, then run test snippets against the live model.

  server:  in_rootfs.sh --gpu python3 /w/xpu/devserver.py [--ram-gb 20]      (socket /w/run/dev.sock)
  client:  python3 glm/xpu/dev.py snippet.py   or   python3 dev.py -c "print(tier.summary())"

Each request is Python source executed in a persistent namespace holding model, config, cache, tok, tier, gen, torch,
and helpers (reload(), decode(n), log). Before each request the glm_xpu kernel modules whose files changed are
reloaded and their ops re-exported into exllamav3_ext, so kernel edits apply without reloading the model.
Output (stdout/stderr and tracebacks) goes back to the client. One request at a time.
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import socket
import sys
import time
import traceback

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--ram-gb", type=float, default=27.0)
ap.add_argument("--cache", type=int, default=16384)
ap.add_argument("--sock", default="/w/run/dev.sock")
ap.add_argument("--no-warm", action="store_true")
a = ap.parse_args()

meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
PROFILE = "/w/run/route_profile.npy"   # our own decode pick counts per expert (tier.save_profile after each request)
order = []
if not a.no_warm:
    # warm order (hottest first -> VRAM, then RAM): glm-flash-lite's decode routing stats, blended with our own profile
    # once it has data; both are normalised per MoE layer so neither dominates by token count alone
    import numpy as np
    stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))   # glm-flash-lite data/stats_own_dec.json (MIT)
    E = meta["experts_per_layer"]
    up = np.zeros((meta["n_layers"], E))
    for k, v in stats.items():
        if k in li_of:
            up[li_of[k]] = np.asarray(v, np.float64) / max(sum(v), 1e-9)
    own = np.load(PROFILE).reshape(up.shape) if os.path.exists(PROFILE) else np.zeros_like(up)
    n_own = own.sum(1, keepdims=True)
    w = np.minimum(n_own.sum() / (8 * 42 * 2000), 0.8)       # trust our profile more as it grows (2000 tokens: 0.8)
    score = (1 - w) * up + w * own / np.maximum(n_own, 1e-9)
    order = [(int(li), int(e)) for li, e in zip(*np.unravel_index(np.argsort(-score, axis=None), score.shape))
             if score[li, e] > 0]
    log(f"warm order: {len(order)} experts, own routing profile weight {w:.2f}")
MTP = int(os.environ.get("GLM_XPU_MTP", "0"))       # > 0: speculative decoding with the MTP head, this many draft tokens
model, config, cache, tok, tier = load("/models", a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order, mtp=MTP)

from exllamav3 import Generator
from exllamav3.generator.sampler import ArgmaxSampler
import exllamav3_ext

gkw = {}
if MTP:
    gkw = dict(draft_model=model.mtp[0], draft_cache=model.mtp[1], num_draft_tokens=MTP, dynamic_draft_tokens=True)
# prompt chunk: one pass fetches every routed expert once, so a larger chunk means fewer re-fetches. Activations: 8K
# tokens peak ~1.5 GiB on top of the expert pool (the 3 GiB reserve; 1.7 GiB stays free); 12K ran out of device memory.
# Bounded per layer: KDA split every 2048 tokens, MLA in 1024-token sub-chunks, dense MLP row slices, KDA prefill
# without dead fp32 tensors; the prompt's cached blocks are released before the first decode graph replay
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=int(os.environ.get("GLM_XPU_CHUNK", "8192")),
                **gkw)
from glm_xpu import block_graphs
block_graphs.install(tier)
from glm_xpu import live
live.install(gen, "/w/run/live.log")   # live prefill / decode speed: tail -f glm/run/live.log          # decode-step XPU graphs (GLM_XPU_BLOCK_GRAPHS=0 to disable)

KMODS = ["glm_xpu.ops", "glm_xpu.k_hc", "glm_xpu.k_kda"]
_mtime = {}


def _note_mtimes():
    for m in KMODS:
        mod = sys.modules.get(m)
        if mod is not None:
            _mtime[m] = os.path.getmtime(mod.__file__)


def reload(force=False):
    """Reload changed glm_xpu kernel modules and re-export ops into exllamav3_ext."""
    changed = []
    for m in KMODS:
        mod = sys.modules.get(m)
        if mod is None:
            continue
        mt = os.path.getmtime(mod.__file__)
        if not force and m not in _mtime:       # imported lazily after startup: note it, a reload would recompile its kernels
            _mtime[m] = mt
            continue
        if force or mt != _mtime.get(m):
            importlib.reload(mod)
            _mtime[m] = mt
            changed.append(m)
    if changed:
        ops = sys.modules["glm_xpu.ops"]
        for k, v in vars(ops).items():
            if not k.startswith("_"):
                setattr(exllamav3_ext, k, v)
        print(f"[reloaded {', '.join(changed)}]")
    return changed


def decode(n=16, prompt="Write a short story about a lighthouse keeper.", warm=True):
    """Greedy decode n tokens after a short prompt; returns (tok/s, text, tier stats delta)."""
    p = model.default_chat_prompt(prompt)
    if warm:
        gen.generate(prompt=p, max_new_tokens=4, sampler=ArgmaxSampler(), completion_only=True)
    s0 = dict(tier.stats)
    t0 = time.perf_counter()
    out = gen.generate(prompt=p, max_new_tokens=n, sampler=ArgmaxSampler(), completion_only=True)
    dt = time.perf_counter() - t0
    d = {k: round(tier.stats[k] - s0[k], 3) for k in s0}
    return n / dt, out, d


ns = dict(model=model, config=config, cache=cache, tok=tok, tier=tier, gen=gen, torch=torch, log=log,
          reload=reload, decode=decode, ArgmaxSampler=ArgmaxSampler)
_note_mtimes()

if os.path.exists(a.sock):
    os.unlink(a.sock)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(a.sock)
srv.listen(1)
log(f"dev server ready on {a.sock}")
while True:
    conn, _ = srv.accept()
    with conn:
        data = b""
        while not data.endswith(b"\0"):
            chunk = conn.recv(1 << 16)
            if not chunk:
                break
            data += chunk
        src = data.rstrip(b"\0").decode()
        if src.strip() == "__quit__":
            conn.sendall(b"bye\n")
            break
        buf = io.StringIO()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                reload()
                if cache is not None and not gen.active_jobs and not gen.pending_jobs \
                        and len(cache.free_list) < cache.num_slots:
                    print(f"[recurrent-state slot leak: {cache.num_slots - len(cache.free_list)} slot(s) reclaimed]")
                    cache.reset_states()
                with torch.inference_mode():
                    exec(compile(src, "<request>", "exec"), ns)
            except BaseException:
                traceback.print_exc()
        try:
            tier.save_profile(PROFILE)
        except Exception as e:                  # noqa: BLE001
            buf.write(f"[routing profile not saved: {e}]\n")
        buf.write(f"[{time.perf_counter() - t0:.1f} s]\n")
        try:
            conn.sendall(buf.getvalue().encode())
        except BrokenPipeError:
            pass
log("dev server stopped")
# Skip interpreter teardown: the Level Zero / SYCL runtime has been seen spinning for 15+ minutes while tearing down
# ~30 GB of pinned host memory at exit. The OS reclaims everything at once.
try:
    torch.xpu.synchronize()
finally:
    sys.stdout.flush()
    os._exit(0)
