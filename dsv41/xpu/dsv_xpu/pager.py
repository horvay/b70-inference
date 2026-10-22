"""Exact-routing decode without host waits.

Exact routing needs the host between a layer's router and its expert kernels: experts that are not in VRAM or RAM
must be read from NVMe and their addresses written into the pointer table first. Done synchronously (read the picks
back, decide, enqueue the rest), every layer serializes CPU and GPU: the CPU waits for the GPU to reach the router,
then the GPU idles while the CPU does the bookkeeping and enqueues the next kernels.

Here the main thread enqueues a whole decode step without waiting. After each router a sync-point kernel copies the
picks to pinned host memory, publishes ready[l] = step and spins (one work-item, polling pinned memory) until the
pager thread publishes done[l] >= step. The pager thread runs the same host logic as the synchronous path
(moe.host_layer: residency, NVMe reads of misses, LRU, next-layer prefetch) and publishes done. The GPU stalls only
for real misses; the CPU-side enqueue overlaps the GPU.

Ordering: when the pager handles layer l, every kernel before l's sync point has completed and none after it has
started, so it may move pieces and rewrite pointers (the expert kernels read the pinned pointer table when they run).
Only the pager touches the tier while a step is in flight; the main thread waits for it (wait_idle) before prompt
processing or the next step. A spin that times out (err flag) or a pager error stops the run at the next wait_idle.
"""
from __future__ import annotations

import ctypes
import os
import queue
import threading
import time

import numpy as np
import torch

from . import moe

S = 64            # flag stride in int64s (512 bytes): flags written by the GPU and by the host never share a cache line
                  # (with ready and done in one line, a done written right after the GPU's ready store was never seen)
TIMEOUT_S = float(os.environ.get("DSV_PAGER_TIMEOUT", "60"))   # host stalls (swap-in under memory pressure) are not faults
DEBUG = os.environ.get("DSV_PAGER_DEBUG", "0") == "1"
SPIN_MAX = int(os.environ.get("DSV_SPIN_MAX", "2000000"))       # GPU-side polls before giving up (~1-4 s)


def log(*a):
    print(" -- dsv.pager:", *a, flush=True)


class Pager:
    def __init__(self, tier, max_rows=8, k=6):
        self.T = tier
        L = tier.L
        self.L = L
        self.maxn = 5 * max_rows * k          # picks + up to 3 predicted layers + unboosted picks, R <= max_rows
        self.flags = torch.zeros((2 * L + 4) * S, dtype=torch.int64).pin_memory()   # ready[L], done[L], err, abort
        self.f = self.flags.numpy()
        self.ERR, self.ABORT = 2 * L * S, (2 * L + 1) * S
        self.picks = torch.zeros((L, self.maxn), dtype=torch.int64).pin_memory()
        self.ph = self.picks.numpy()
        self.waits = torch.zeros(L * S, dtype=torch.int64).pin_memory()
        self.wh = self.waits.numpy()
        lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "..", "lib", "dsv_spin.so"))
        lib.wait_ge.restype = ctypes.c_int64
        lib.wait_ge.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.fast_release.restype = ctypes.c_int64
        lib.fast_release.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p, ctypes.c_void_p]
        self.flipped = ctypes.c_int64(0)
        self.lib = lib
        assert tier.vslot.dtype == np.int64 and tier.rslot.dtype == np.int64 and tier.vslot.flags.c_contiguous
        # the step id the sync kernels compare against: device memory (graph replays read it), set by a copy from a
        # pinned ring at the start of each step
        self.step_dev = torch.zeros(1, dtype=torch.int64, device=tier.dev)
        self.step_pin = torch.zeros(8, dtype=torch.int64).pin_memory()
        self.capturing = False
        self.base = self.flags.data_ptr()
        self.step = 0
        self.q = queue.Queue()
        self.pending = 0
        self.cv = threading.Condition()
        self.exc = None
        self.stats = dict(layers=0, t_ready=0.0, t_host=0.0, spin=0)
        self.th = threading.Thread(target=self._run, name="dsv-pager", daemon=True)
        self.th.start()
        self._jit()

    def _jit(self):
        """Compile the sync-point kernels now (a first-use JIT inside a step holds the GIL for seconds): one round
        on a scratch flag slot whose done is already published."""
        dev = self.T.dev
        p = torch.zeros(6, dtype=torch.int64, device=dev)
        scratch = torch.zeros(4 * S, dtype=torch.int64).pin_memory()
        sc = scratch.numpy()
        sc[S] = 1
        w = torch.zeros(S, dtype=torch.int64).pin_memory()
        one = torch.ones(1, dtype=torch.int64, device=dev)
        moe.G().sync_point(p, self.picks[0], scratch, 0, S, 2 * S, w[0:1], one, 1000, 2)
        moe.X().spin_done(scratch, S, 2 * S, w[0:1], one, 1000, p)
        torch.xpu.synchronize(dev)
        assert sc[0] == 1 and sc[2 * S] == 0, "sync-point self-test failed"

    # ------------------------------------------------------------------ main thread
    def begin_step(self):
        """Called by the runner before enqueuing a decode step. Returns the step id."""
        self.wait_idle()
        self.f[self.ABORT] = 0
        self.step += 1
        sp = self.step_pin[self.step % 8:self.step % 8 + 1]
        sp.fill_(self.step)
        self.step_dev.copy_(sp, non_blocking=True)
        return self.step

    def push_all(self, items):
        """Graph steps: hand the pager every layer of the step up front (the sync points run inside the graphs).
        items: [(li, R, n, ahead (layer ids predicted), boost)] in execution order."""
        with self.cv:
            self.pending += len(items)
        for it in items:
            self.q.put((self.step,) + tuple(it))

    def sync(self, li, picks, R, ahead, boost, push=True):
        """Enqueue layer li's sync point (after its router) and hand the layer to the pager thread (push=False: graph
        capture records the kernels only; the replaying step hands the layers over with push_all)."""
        if DEBUG:
            log(f"main: enqueue sync step {self.step} layer {li}, t={time.perf_counter() % 1000:.4f}")
        n = picks.shape[-2] * picks.shape[-1]          # this layer's picks (R x k); picks also holds next / want
        assert picks.numel() <= self.maxn
        # publish the picks + ready (SYCL), then spin on done with loads that bypass L1 / L3 (ESIMD): with atomic
        # loads the GPU kept reading a cached copy of the host's flag and nearly every handoff timed out
        moe.G().sync_point(picks.reshape(-1), self.picks[li], self.flags, li * S, (self.L + li) * S, self.ERR,
                           self.waits[li * S], self.step_dev, SPIN_MAX, 2)
        moe.X().spin_done(self.flags, (self.L + li) * S, self.ERR, self.waits[li * S], self.step_dev, SPIN_MAX, picks)
        if not push:
            return
        with self.cv:
            self.pending += 1
        self.q.put((self.step, li, R, n, tuple(ahead), boost))

    def wait_idle(self):
        """Block until the pager has handled every enqueued layer; raise its error (or a GPU spin timeout)."""
        with self.cv:
            while self.pending:
                self.cv.wait()
        if self.exc is not None:
            e, self.exc = self.exc, None
            raise RuntimeError("dsv pager failed") from e
        if self.f[self.ERR]:
            step = int(self.f[self.ERR])
            self.f[self.ERR] = 0
            late = [li for li in range(self.L) if self.wh[li * S] > SPIN_MAX]
            raise RuntimeError(f"dsv pager: a GPU sync point timed out in step {step} (layers {late}; expert output of "
                               f"that step is wrong)")

    def abort(self):
        self.f[self.ABORT] = 1

    # ------------------------------------------------------------------ pager thread
    def _run(self):
        T = self.T
        while True:
            step, li, R, n, ahead, boost = self.q.get()
            try:
                if self.exc is None and not self.f[self.ABORT]:
                    T.reap()
                    t0 = time.perf_counter()
                    # C fast path, GIL released: wait for the GPU's ready, release it at once if every pick is
                    # resident (r = 1); r = 0: some piece must be read first (host_layer below releases)
                    ns = int(T.scount[li]) if T.stage_on and T.sstep[li] == step else 0
                    r = self.lib.fast_release(self.base + 8 * li * S, step, int(TIMEOUT_S * 1e9),
                                              self.base + 8 * self.ABORT, self.picks[li].data_ptr(), n,
                                              T.vslot.ctypes.data, T.rslot.ctypes.data, li * T.E * 3,
                                              self.base + 8 * (self.L + li) * S,
                                              T.sflags.data_ptr() + 8 * li * T.FS if ns else None, ns,
                                              T.skeys[li].ctypes.data, T.saddrs[li].ctypes.data, T.ptr_h.ctypes.data,
                                              ctypes.byref(self.flipped))
                    if ns:
                        st_ = T.stats
                        st_["stage_layers"] = st_.get("stage_layers", 0) + 1
                        st_["stage_used"] = st_.get("stage_used", 0) + (ns if self.flipped.value else 0)
                    t1 = time.perf_counter()
                    if r == -1:
                        raise TimeoutError(f"layer {li} step {step}: the GPU did not reach the sync point in "
                                           f"{TIMEOUT_S} s")
                    if r >= 0:
                        k = n // R
                        m = n * (1 + len(ahead) + int(boost))
                        both = self.ph[li, :m].copy()
                        done_i = (self.L + li) * S

                        def release(done_i=done_i, step=step):
                            self.f[done_i] = step
                        moe.host_layer(T, li, both, R, n, ahead, boost, None if r == 1 else release, step)
                        st = self.stats
                        st["fast"] = st.get("fast", 0) + int(r == 1)
                        st["layers"] += 1
                        st["t_ready"] += t1 - t0
                        st["t_host"] += time.perf_counter() - t1
                        st["spin"] += int(self.wh[li * S])
                        if DEBUG:
                            log(f"step {step} layer {li}: ready after {(t1 - t0) * 1e3:.2f} ms, host "
                                f"{(time.perf_counter() - t1) * 1e3:.2f} ms, t={time.perf_counter() % 1000:.4f}")
            except BaseException as e:             # never leave the GPU spinning: publish done regardless
                if self.exc is None:
                    self.exc = e
                    log(f"error: {type(e).__name__}: {e}")
            finally:
                self.f[(self.L + li) * S] = step
                with self.cv:
                    self.pending -= 1
                    self.cv.notify_all()
