"""DeepSeek-V4.1-Flash routed experts on the B70: a three-tier piece cache (VRAM, pinned RAM, NVMe store) and the MoE
forward that replaces exllamav3's BlockSparseMLP.

A piece is one projection of one expert (gate, up or down) at its own K (SAGE MixedK: 1..4 bits), K units long
(unit = 1,474,560 bytes). Pieces live in per-K slot pools in VRAM and in pinned RAM; the MoE kernels read a piece
through a pointer table [L, E, 3] wherever it is (RAM zero-copy over PCIe). The NVMe store (pack_store.py) holds all.

Decode (<= 4 rows): routing on the GPU; `nosync` routing only picks experts whose three pieces are in VRAM or RAM (a
non-resident pick falls to the next-best resident expert) so the step never waits; the host learns the true picks one
step later and fetches the experts that were wanted but missing. Exact mode syncs per layer and reads misses from
NVMe before the kernels run.
Prefill (more rows): every expert with rows is dequantized (Triton) and run as fp16 GEMMs.
"""
from __future__ import annotations

import concurrent.futures as cf
import ctypes
import json
import mmap
import os
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from . import exl3_tri

H, I = 5120, 2304
UNIT = H * I // 8
OFF_SUH_G, OFF_SUH_U, OFF_SVH_G, OFF_SVH_U, OFF_SUH_D, OFF_SVH_D = 0, H, 2 * H, 2 * H + I, 2 * H + 2 * I, 2 * H + 3 * I
VEC = 2 * H + 3 * I + H
ACT_LIMIT = 10.0


def log(*a):
    print(" -- dsv.moe:", *a, flush=True)


def s64(p: int) -> int:
    return p - (1 << 64) if p >= (1 << 63) else p


_X = None


def X():
    global _X
    if _X is None:
        torch.ops.load_library(os.environ.get("DSV_MOE_LIB") or os.path.join(os.path.dirname(__file__), "..", "lib",
                                                                              "dsv_moe.so"))
        _X = torch.ops.dsv_moe
    return _X


def host_alloc(nbytes: int) -> torch.Tensor:
    """Pinned (USM host) uint8 buffer, zero-copy readable by the GPU. Keep each under 2 GiB (larger fail on XPU)."""
    t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    assert t.is_pinned() and t.data_ptr() != 0
    return t


IO_THREADS = int(os.environ.get("DSV_IO_THREADS", "32"))
IO_MIN_BATCH = int(os.environ.get("DSV_IO_MIN_BATCH", "16"))      # fewer pieces than this: read through the pool


def _io_lib():
    """The C reader (lib/dsv_io.so: parallel O_DIRECT reads with its own threads), or None to read through the pool."""
    if os.environ.get("DSV_C_IO", "1") != "1":      # on: 664-token prompt 31-35 -> 24 s, decode even (batches >= IO_MIN_BATCH)
        return None
    path = os.path.join(os.path.dirname(__file__), "..", "lib", "dsv_io.so")
    if not os.path.exists(path):
        return None
    lib = ctypes.CDLL(path)
    lib.read_pieces.restype = ctypes.c_int64
    lib.read_pieces.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                ctypes.c_int, ctypes.c_int64]
    return lib


class Tier:
    def __init__(self, store_dir: str, device, vram_bytes: int, ram_bytes: int, readers: int = 16):
        self.dev = torch.device(device)
        meta = json.load(open(os.path.join(store_dir, "experts.json")))
        assert meta["format"] == "dsv41-expert-store-v1"
        self.L, self.E = meta["n_layers"], meta["n_experts"]
        self.mtp_layers = meta.get("mtp_layers", [])        # the DSpark drafter's MoE layers (128 experts each)
        rows = np.array(meta["layers"], np.int64)                    # [L, E, 4] Kg Ku Kd off
        self.K3 = rows[..., :3].copy()                                # [L, E, 3]
        off = rows[..., 3]
        po = np.stack([off, off + self.K3[..., 0], off + self.K3[..., 0] + self.K3[..., 1]], -1)
        self.piece_off = (po * UNIT).reshape(-1)                      # bytes, per piece key
        self.piece_K = self.K3.reshape(-1)
        self.NP = self.L * self.E * 3
        self.Kdev = torch.from_numpy(self.K3.astype(np.int32)).to(self.dev)
        vec = np.load(os.path.join(store_dir, "experts_vectors.bin.npy"), mmap_mode="r")
        self.vec = torch.empty((self.L, self.E, VEC), dtype=torch.half, device=self.dev)
        for l in range(self.L):
            self.vec[l].copy_(torch.from_numpy(np.ascontiguousarray(vec[l])))
        self.fd = os.open(os.path.join(store_dir, "experts.bin"), os.O_RDONLY | os.O_DIRECT)

        # budget split over K classes in proportion to the units each class holds
        units_per_K = {K: int((self.piece_K == K).sum()) * K for K in (1, 2, 3, 4)}
        tot = sum(units_per_K.values())

        def split(budget):
            return {K: int(budget * u / tot) // (K * UNIT) for K, u in units_per_K.items()}

        # staging ring (two halves, alternating layers): the next layer's predicted RAM pieces are copied here on a
        # side stream while the current layer computes (see stage())
        self.stage_on = os.environ.get("DSV_STAGE", "0") == "1"
        self.stage_units = int(os.environ.get("DSV_STAGE_UNITS", "160"))       # per half (1 unit = 1.47 MB)
        if self.stage_on:
            self.sring = torch.empty(2 * self.stage_units * UNIT, dtype=torch.uint8, device=self.dev)
            self.sbase = s64(self.sring.data_ptr())
            vram_bytes -= 2 * self.stage_units * UNIT
        self.copy_stream = torch.xpu.Stream(self.dev)
        FS = 64
        self.FS = FS
        self.sflags = torch.zeros(self.L * FS, dtype=torch.int64).pin_memory()     # per layer: staging copies done
        self.sflags_h = self.sflags.numpy()
        self.skeys = np.zeros((self.L, 512), np.int64)
        self.saddrs = np.zeros((self.L, 512), np.int64)
        self.scount = np.zeros(self.L, np.int64)
        self.sstep = np.zeros(self.L, np.int64)
        self.shalf = [-1, -1]
        self.nofree = set()                 # RAM pieces whose staging copy may not have run yet
        self.nofree_pending = []            # (layer, step, keys)
        self.pred = {}                      # layer -> predicted expert ids (prediction accuracy stats)
        # promotion of experts hot right now (RAM -> VRAM on the copy stream; the RAM copy stays: eviction from VRAM
        # is a pointer change, no write-back)
        self.promo_on = os.environ.get("DSV_PROMO", "1") == "1"
        self.promo_min = float(os.environ.get("DSV_PROMO_MIN", "2.5"))        # recent-use score to qualify
        self.promo_per_token = int(os.environ.get("DSV_PROMO_PER_TOKEN", "24"))  # pieces per token
        self.recent_decay = float(os.environ.get("DSV_RECENT_DECAY", "0.75"))
        self.recent = np.zeros((self.L, self.E), np.float32)
        self.promo_budget = self.promo_per_token
        self.promo_ticket = 0
        self.promo_flag = torch.zeros(64, dtype=torch.int64).pin_memory()
        self.promo_flag_h = self.promo_flag.numpy()
        self.promo_pending = []             # (ticket, pk, K, vslot)

        # VRAM pools
        nv = split(vram_bytes)
        self.vpool, self.vaddr = {}, {}
        for K, n in nv.items():
            t = torch.empty((max(n, 1), K * UNIT), dtype=torch.uint8, device=self.dev)
            self.vpool[K] = t
            self.vaddr[K] = [s64(t.data_ptr() + i * K * UNIT) for i in range(n)]
        self.vfree = {K: list(range(n - 1, -1, -1)) for K, n in nv.items()}
        self.vlru = {K: OrderedDict() for K in nv}                   # piece -> slot
        # RAM pools: USM host chunks just under 2 GiB (larger pinned allocations fail on XPU)
        nr = split(ram_bytes)
        self.rpool, self.rnp, self.raddr = {}, {}, {}
        self._chunks = []
        for K, n in nr.items():
            rec = K * UNIT
            per = (2**31 - 1) // rec
            rows_t, rows_np, addrs = [], [], []
            left = n
            while left > 0:
                m = min(per, left)
                ch = host_alloc(m * rec)
                assert ch.data_ptr() != 0
                self._chunks.append(ch)
                cv = ch.view(m, rec)
                cn = ch.numpy().reshape(m, rec)
                for i in range(m):
                    rows_t.append(cv[i])
                    rows_np.append(cn[i] if cn is not None else None)
                    addrs.append(s64(ch.data_ptr() + i * rec))
                left -= m
            self.rpool[K], self.rnp[K], self.raddr[K] = rows_t, rows_np, addrs
        self.rfree = {K: list(range(n - 1, -1, -1)) for K, n in nr.items()}
        self.rlru = {K: OrderedDict() for K in nr}

        self.vslot = np.full(self.NP, -1, np.int64)
        self.rslot = np.full(self.NP, -1, np.int64)
        self.ptr_pin = torch.zeros((self.L, self.E, 3), dtype=torch.int64).pin_memory()
        self.ptr_h = self.ptr_pin.numpy().reshape(-1)
        self.ptr = torch.zeros((self.L, self.E, 3), dtype=torch.int64, device=self.dev)
        self.res_pin = torch.full((self.L, self.E), float("-inf"), dtype=torch.float32).pin_memory()
        self.res_h = self.res_pin.numpy()
        self.res = self.res_pin.to(self.dev)
        # residency boost (exact routing with a preference for cached experts): an expert whose three pieces are in VRAM
        # or RAM competes with its score * (1 + boost); its routing weight stays the plain score. 0 = plain exact.
        self.boost = float(os.environ.get("DSV_BOOST", "0"))
        self.block_boost = float(os.environ.get("DSV_BLOCK_BOOST", "0.3"))    # verify blocks: prefer the block's experts (NLL, repetition unchanged)
        self.mul_pin = torch.ones((self.L, self.E), dtype=torch.float32).pin_memory()
        self.mul_h = self.mul_pin.numpy()
        self.dirty = np.zeros(self.L, bool)
        self.hits = np.zeros((self.L, self.E), np.float64)
        self.uses = np.zeros((self.L, self.E), np.float64)      # this session's picks (merged into the profile)
        self.profile_save = os.environ.get("DSV_SAVE_PROFILE", "1") == "1"
        self.profile_every = int(os.environ.get("DSV_PROFILE_EVERY", "1000"))   # decode steps between saves
        self.pool = cf.ThreadPoolExecutor(readers)
        self._tl = threading.local()
        self._io = _io_lib()
        self.side = torch.xpu.Stream(self.dev)
        self.stats = dict(picks=0, vram=0, ram=0, nvme=0, t_nvme=0.0)
        self.nosync = os.environ.get("DSV_NOSYNC", "0") == "1"
        self.promote_min = 2
        self.promote_budget = int(os.environ.get("DSV_PROMOTE_BUDGET", "0"))   # RAM -> VRAM pieces per layer call
        self.ntick = 0
        self.margin = float(os.environ.get("DSV_RAM_MARGIN", "0.0"))
        self.max_promote = int(os.environ.get("DSV_MAX_PROMOTE", "0"))    # pieces RAM -> VRAM per step (no-sync)
        self.fetching = {}                                                 # piece -> future (no-sync NVMe reads)
        self.rdefer = []                                                   # RAM slots freed by promotions in flight
        log(f"tier: VRAM {sum(n * K for K, n in nv.items()) * UNIT / 2**30:.1f} GiB {nv}, "
            f"RAM {sum(n * K for K, n in nr.items()) * UNIT / 2**30:.1f} GiB {nr}")

    # ------------------------------------------------------------------------------------------- NVMe
    def _read(self, pk: int, dst: np.ndarray):
        """O_DIRECT read of piece pk into dst (via a thread-local anonymous bounce: O_DIRECT into USM host fails)."""
        n = int(self.piece_K[pk]) * UNIT
        tl = self._tl
        if not hasattr(tl, "buf"):
            tl.buf = mmap.mmap(-1, 4 * UNIT, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            tl.np = np.frombuffer(tl.buf, dtype=np.uint8)
        mv = memoryview(tl.buf)
        got = 0
        base = int(self.piece_off[pk])
        while got < n:
            r = os.preadv(self.fd, [mv[got:n]], base + got)
            if r <= 0:
                raise IOError(f"short read of piece {pk}")
            got += r
        np.copyto(dst[:n], tl.np[:n])

    def read_many(self, pks, dsts):
        """Read pieces pks into the host buffers dsts (numpy views) in one call: the C reader's own threads (no GIL, no
        Python per piece; ~25% more NVMe throughput than the pool of _read), or the pool without it."""
        if not pks:
            return
        if self._io is None or len(pks) < IO_MIN_BATCH:
            # small batches (decode prefetches, ~80 a token): the pool's standing threads beat starting C threads per call
            for f in [self.pool.submit(self._read, pk, d) for pk, d in zip(pks, dsts)]:
                f.result()
            return
        pk = np.asarray(pks, np.int64)
        offs = np.ascontiguousarray(self.piece_off[pk], np.int64)
        lens = np.ascontiguousarray(self.piece_K[pk] * UNIT, np.int64)
        addrs = np.array([d.ctypes.data for d in dsts], np.int64)
        bad = self._io.read_pieces(self.fd, offs.ctypes.data, lens.ctypes.data, addrs.ctypes.data, len(pk),
                                   min(IO_THREADS, len(pk)), 4 * UNIT)
        if bad:
            raise IOError(f"{bad} of {len(pk)} expert pieces failed to read")

    def read_async(self, pks, dsts):
        """Background reads of pks into dsts: one future per piece (small batches: the pool, one read each; a large
        batch: one C call on a pool thread, its future shared). Never waits on the pool from inside it."""
        if self._io is None or len(pks) < IO_MIN_BATCH:
            return [self.pool.submit(self._read, pk, d) for pk, d in zip(pks, dsts)]
        f = self.pool.submit(self.read_many, list(pks), list(dsts))
        return [f] * len(pks)

    def _ram_np(self, K, slot):
        return self.rnp[K][slot]

    # ------------------------------------------------------------------------------------------- residency
    def _set_ptr(self, pk, addr):
        self.ptr_h[pk] = addr
        self.dirty[pk // (self.E * 3)] = True

    def _update_res(self, l, e):
        base = (l * self.E + e) * 3
        ok = all(self.vslot[base + p] >= 0 or self.rslot[base + p] >= 0 for p in range(3))
        inv = all(self.vslot[base + p] >= 0 for p in range(3))
        v = (0.0 if inv else -self.margin) if ok else float("-inf")
        if self.res_h[l, e] != v:
            self.res_h[l, e] = v
            self.dirty[l] = True
        self.mul_h[l, e] = (1.0 + self.boost) if ok else 1.0

    def set_boost(self, b: float):
        """Change the residency boost at run time (recomputes every expert's selection multiplier)."""
        self.boost = float(b)
        res = ((self.vslot >= 0) | (self.rslot >= 0)).reshape(self.L, self.E, 3).all(-1)
        self.mul_h[:] = np.where(res, 1.0 + self.boost, 1.0)

    def _take_ram(self, K, protect):
        if self.rfree[K]:
            return self.rfree[K].pop()
        lru = self.rlru[K]
        for pk in lru:
            if pk in protect or pk in self.nofree:
                continue
            s = lru.pop(pk)
            self.rslot[pk] = -1
            if self.vslot[pk] < 0:
                self._set_ptr(pk, 0)
                e = pk // 3
                self._update_res(e // self.E, e % self.E)
            return s
        return -1

    def _take_vram(self, K, protect):
        if self.vfree[K]:
            return self.vfree[K].pop()
        lru = self.vlru[K]
        for pk in lru:
            if pk in protect:
                continue
            s = lru.pop(pk)
            self.vslot[pk] = -1
            # write back to RAM (exclusive tiers) if a RAM slot is free or reclaimable
            rs = self._take_ram(K, protect)
            if rs >= 0:
                self.rpool[K][rs].copy_(self.vpool[K][s], non_blocking=True)   # D2H, stream-ordered before the refill
                self.rslot[pk] = rs
                self.rlru[K][pk] = rs
                self._set_ptr(pk, self.raddr[K][rs])
            else:
                self._set_ptr(pk, 0)
            e = pk // 3
            self._update_res(e // self.E, e % self.E)
            return s
        return -1

    def load_to_ram(self, pks, protect):
        """Read pieces from NVMe into RAM slots (parallel); returns the number read. Pieces already being fetched in
        the background are waited for instead of read twice."""
        waits = [pk for pk in pks if pk in self.fetching]
        if waits:
            t0 = time.time()
            for pk in waits:
                self.fetching[pk][2].result()
            self.stats["t_wait_ahead"] = self.stats.get("t_wait_ahead", 0.0) + time.time() - t0
            self.stats["ahead_waits"] = self.stats.get("ahead_waits", 0) + len(waits)
            self.reap()
        jobs = []
        for pk in pks:
            if self.rslot[pk] >= 0 or self.vslot[pk] >= 0:
                continue
            K = int(self.piece_K[pk])
            s = self._take_ram(K, protect)
            if s < 0:
                continue
            jobs.append((pk, K, s))
        if not jobs:
            return 0
        t0 = time.time()
        self.read_many([pk for pk, _, _ in jobs], [self._ram_np(K, s) for _, K, s in jobs])
        self.stats["t_nvme"] += time.time() - t0
        for pk, K, s in jobs:
            self.rslot[pk] = s
            self.rlru[K][pk] = s
            self._set_ptr(pk, self.raddr[K][s])
            e = pk // 3
            self._update_res(e // self.E, e % self.E)
        self.stats["nvme"] += len(jobs)
        return len(jobs)

    def promote(self, pk, protect):
        """RAM -> VRAM (exclusive: the RAM slot is freed)."""
        K = int(self.piece_K[pk])
        rs = self.rslot[pk]
        if rs < 0 or self.vslot[pk] >= 0:
            return
        if not self.vfree[K]:
            # frequency-gated admission: only displace a VRAM piece whose expert is used less than this one
            e = pk // 3
            ch = self.hits.flat[e]
            for v in self.vlru[K]:
                if v in protect:
                    continue
                if self.hits.flat[v // 3] >= ch:
                    self.stats["promote_denied"] = self.stats.get("promote_denied", 0) + 1
                    return
                break
        vs = self._take_vram(K, protect)
        if vs < 0:
            return
        self.stats["promoted"] = self.stats.get("promoted", 0) + 1
        self.vpool[K][vs].copy_(self.rpool[K][rs], non_blocking=True)
        self.rlru[K].pop(pk, None)
        self.rslot[pk] = -1
        # the async H2D copy still reads the RAM slot: it is only reusable once the copy has run (release_deferred)
        self.rdefer.append((K, rs))
        self.vslot[pk] = vs
        self.vlru[K][pk] = vs
        self._set_ptr(pk, self.vaddr[K][vs])
        e = pk // 3
        self._update_res(e // self.E, e % self.E)

    def touch(self, l, ids: np.ndarray, fetch: bool):
        """Bookkeeping for the picks `ids` of layer l: LRU refresh, hit counts, promotion, NVMe fetch of misses."""
        uniq = np.unique(ids)
        self.hits[l, uniq] += 1
        self.uses[l, uniq] += 1
        self.recent[l, uniq] += 1.0
        protect = set()
        for e in uniq:
            base = (l * self.E + int(e)) * 3
            protect.update((base, base + 1, base + 2))
        miss = []
        for e in uniq:
            base = (l * self.E + int(e)) * 3
            for p in range(3):
                pk = base + p
                K = int(self.piece_K[pk])
                if self.vslot[pk] >= 0:
                    self.vlru[K].move_to_end(pk)
                    self.stats["vram"] += 1
                elif self.rslot[pk] >= 0:
                    self.rlru[K].move_to_end(pk)
                    self.stats["ram"] += 1
                else:
                    miss.append(pk)
        self.stats["picks"] += len(uniq)
        if miss and fetch:
            self.load_to_ram(miss, protect)
        budget = self.promote_budget
        for e in sorted(uniq, key=lambda e: -self.hits[l, e]):
            if budget <= 0:
                break
            if self.hits[l, e] >= self.promote_min:
                base = (l * self.E + int(e)) * 3
                for p in range(3):
                    if self.rslot[base + p] >= 0:
                        self.promote(base + p, protect)
                        budget -= 1
        self.release_deferred()

    def fetch_async(self, pks):
        """No-sync: start NVMe reads of wanted pieces into RAM slots; they become resident in reap()."""
        jobs = []
        for pk in pks:
            if self.rslot[pk] >= 0 or self.vslot[pk] >= 0 or pk in self.fetching:
                continue
            K = int(self.piece_K[pk])
            s = self._take_ram(K, self.fetching)
            if s < 0:
                break
            jobs.append((pk, K, s))
        self._fetch_batch(jobs)

    def _fetch_batch(self, jobs):
        """One background read of [(pk, K, slot)] into RAM slots; each piece is in fetching until the batch is done."""
        if not jobs:
            return
        futs = self.read_async([pk for pk, _, _ in jobs], [self._ram_np(K, s) for _, K, s in jobs])
        for (pk, K, s), f in zip(jobs, futs):
            self.fetching[pk] = (K, s, f)

    def prefetch(self, l, ids: np.ndarray, protect):
        """Exact mode, layer ahead: start background NVMe reads of layer l's predicted experts that are not resident."""
        miss = []
        for e in np.unique(ids):
            base = (l * self.E + int(e)) * 3
            for p in range(3):
                pk = base + p
                if self.rslot[pk] < 0 and self.vslot[pk] < 0 and pk not in self.fetching:
                    miss.append(pk)
        if miss:
            self.stats["ahead_issued"] = self.stats.get("ahead_issued", 0) + len(miss)
            prot = set(protect) | set(self.fetching)
            jobs = []
            for pk in miss:
                K = int(self.piece_K[pk])
                s = self._take_ram(K, prot)
                if s < 0:
                    break
                prot.add(pk)
                jobs.append((pk, K, s))
            self._fetch_batch(jobs)

    def reap(self):
        done = [pk for pk, (_, _, f) in self.fetching.items() if f.done()]
        for pk in done:
            K, s, f = self.fetching.pop(pk)
            f.result()
            self.rslot[pk] = s
            self.rlru[K][pk] = s
            self._set_ptr(pk, self.raddr[K][s])
            e = pk // 3
            self._update_res(e // self.E, e % self.E)
            self.stats["nvme"] += 1
        return len(done)

    def bookkeep(self, l, used: np.ndarray, wanted: np.ndarray, budget: list):
        """No-sync step bookkeeping for layer l: LRU / hit counts for the used picks, NVMe fetch of wanted pieces
        that were not resident, bounded promotion of hot RAM pieces to VRAM."""
        uniq = np.unique(used)
        self.hits[l, uniq] += 1
        self.uses[l, uniq] += 1
        self.stats["picks"] += len(uniq)
        protect = set()
        for e in uniq:
            base = (l * self.E + int(e)) * 3
            for p in range(3):
                pk = base + p
                protect.add(pk)
                K = int(self.piece_K[pk])
                if self.vslot[pk] >= 0:
                    self.vlru[K].move_to_end(pk)
                    self.stats["vram"] += 1
                elif self.rslot[pk] >= 0:
                    self.rlru[K].move_to_end(pk)
                    self.stats["ram"] += 1
        miss = []
        for e in np.unique(wanted):
            self.hits[l, e] += 0.5
            base = (l * self.E + int(e)) * 3
            for p in range(3):
                pk = base + p
                if self.rslot[pk] < 0 and self.vslot[pk] < 0:
                    miss.append(pk)
        if miss:
            self.stats["want_miss"] = self.stats.get("want_miss", 0) + len(miss) // 3
            self.fetch_async(miss)
        for e in uniq:
            if budget[0] <= 0:
                break
            if self.hits[l, e] >= self.promote_min:
                base = (l * self.E + int(e)) * 3
                for p in range(3):
                    if self.rslot[base + p] >= 0 and budget[0] > 0:
                        self.promote(base + p, protect)
                        budget[0] -= 1

    def _ptr_of(self, pk):
        K = int(self.piece_K[pk])
        if self.vslot[pk] >= 0:
            return self.vaddr[K][self.vslot[pk]]
        if self.rslot[pk] >= 0:
            return self.raddr[K][self.rslot[pk]]
        return 0

    def _unstage(self, l):
        """Point layer l's staged pieces back at their tier copies (its kernels have run: the GPU is past a later
        layer's sync point)."""
        n = int(self.scount[l])
        for i in range(n):
            pk, a = int(self.skeys[l, i]), int(self.saddrs[l, i])
            if self.ptr_h[pk] == a:
                self.ptr_h[pk] = self._ptr_of(pk)
        self.scount[l] = 0
        if n:
            self.nofree_pending.append((l, int(self.sstep[l]), [int(k) for k in self.skeys[l, :n]]))

    def stage(self, ln, ids, step):
        """Copy layer ln's predicted pieces that are in RAM but not in VRAM into the staging half ln % 2 on the copy
        stream, then a flag (sflags[ln] = step) behind them; the pager's fast path points the pieces at the copies
        if the flag is up when layer ln reaches its sync point (else they are read from RAM as usual)."""
        if not self.stage_on:
            return
        still = []
        for ent in self.nofree_pending:             # sources whose copies have run may be evicted again
            l, st, keys = ent
            if self.sflags_h[l * self.FS] >= st:
                self.nofree.difference_update(keys)
            else:
                still.append(ent)
        self.nofree_pending = still
        half = ln % 2
        if self.shalf[half] >= 0:
            self._unstage(self.shalf[half])
        self.shalf[half] = ln
        off = half * self.stage_units
        end = off + self.stage_units
        n = 0
        with torch.xpu.stream(self.copy_stream):
            for e in np.unique(ids):
                base = (ln * self.E + int(e)) * 3
                for p in range(3):
                    pk = base + p
                    if self.vslot[pk] >= 0 or self.rslot[pk] < 0:
                        continue
                    K = int(self.piece_K[pk])
                    if off + K > end or n >= self.skeys.shape[1]:
                        break
                    self.sring[off * UNIT:(off + K) * UNIT].copy_(self.rpool[K][self.rslot[pk]], non_blocking=True)
                    self.skeys[ln, n] = pk
                    self.saddrs[ln, n] = self.sbase + off * UNIT
                    self.nofree.add(pk)
                    self.rlru[K].move_to_end(pk)
                    off += K
                    n += 1
            if n:
                X().set_host_flag(self.sflags, ln * self.FS, int(step), self.sring)
        self.scount[ln] = n
        self.sstep[ln] = step
        self.stats["staged"] = self.stats.get("staged", 0) + n

    def promote_hot(self, l, uniq, protect):
        """Copy this layer's RAM pieces whose expert is hot right now into VRAM (copy stream), displacing a strictly
        colder VRAM piece of the same K. The pointer moves to VRAM once the copy has run (promote_finish)."""
        self.promote_finish()
        if not self.promo_on or self.promo_budget <= 0:
            return
        issued = False
        for e in uniq:
            r = self.recent[l, e]
            if r < self.promo_min:
                continue
            base = (l * self.E + int(e)) * 3
            for p in range(3):
                pk = base + p
                if self.vslot[pk] >= 0 or self.rslot[pk] < 0 or pk in self.nofree or self.promo_budget <= 0:
                    continue
                K = int(self.piece_K[pk])
                vs = -1
                if self.vfree[K]:
                    vs = self.vfree[K].pop()
                else:
                    # victim: a colder VRAM piece, preferably one that still has a RAM copy (evicting a VRAM-only
                    # piece sends its next use to NVMe)
                    scanned = 0
                    cand = -1
                    for q in self.vlru[K]:
                        scanned += 1
                        if scanned > 64:
                            break
                        if q in protect:
                            continue
                        qe = q // 3
                        if self.recent.flat[qe] < r - 1.0:
                            if self.rslot[q] >= 0:
                                cand = q
                                break
                            if cand < 0:
                                cand = q
                    if cand >= 0:
                        q = cand
                        qe = q // 3
                        vs = self.vlru[K].pop(q)
                        self.vslot[q] = -1
                        self._set_ptr(q, self.raddr[K][self.rslot[q]] if self.rslot[q] >= 0 else 0)
                        self._update_res(qe // self.E, qe % self.E)
                        self.stats["promo_evict"] = self.stats.get("promo_evict", 0) + 1
                if vs < 0:
                    self.stats["promo_denied"] = self.stats.get("promo_denied", 0) + 1
                    return
                with torch.xpu.stream(self.copy_stream):
                    self.vpool[K][vs].copy_(self.rpool[K][self.rslot[pk]], non_blocking=True)
                self.nofree.add(pk)
                self.promo_pending.append((self.promo_ticket + 1, pk, K, vs))
                self.promo_budget -= 1
                issued = True
        if issued:
            self.promo_ticket += 1
            with torch.xpu.stream(self.copy_stream):
                X().set_host_flag(self.promo_flag, 0, self.promo_ticket, self.sring if self.stage_on else self.vec)

    def promote_finish(self):
        """Promotions whose copies have run: the piece now lives in VRAM (its RAM copy stays)."""
        if not self.promo_pending:
            return
        done = int(self.promo_flag_h[0])
        still = []
        for ent in self.promo_pending:
            t, pk, K, vs = ent
            if t > done:
                still.append(ent)
                continue
            self.nofree.discard(pk)
            if self.rslot[pk] < 0 or self.vslot[pk] >= 0:       # (cannot happen while in nofree; keep the slot)
                self.vfree[K].append(vs)
                continue
            self.vslot[pk] = vs
            self.vlru[K][pk] = vs
            self._set_ptr(pk, self.vaddr[K][vs])
            e = pk // 3
            self._update_res(e // self.E, e % self.E)
            self.stats["promoted"] = self.stats.get("promoted", 0) + 1
        self.promo_pending = still

    def tick(self):
        """Once per decode step: halve the hit counts every 64 steps (admission follows the current workload)."""
        self.recent *= self.recent_decay
        self.promo_budget = self.promo_per_token
        self.ntick += 1
        if self.profile_save and self.profile_every > 0 and self.ntick % self.profile_every == 0:
            try:
                self.save_profile()
            except Exception as e:
                log(f"routing profile save failed: {e}")
        if self.ntick % 64 == 0:
            self.hits *= 0.5

    def release_deferred(self):
        """RAM slots vacated by RAM -> VRAM promotions become free once the stream has run the copies."""
        if self.rdefer:
            torch.xpu.current_stream(self.dev).synchronize()
            for K, rs in self.rdefer:
                self.rfree[K].append(rs)
            self.rdefer = []

    def upload(self, l=None):
        """Push changed pointer-table / residency rows to the device (stream-ordered). With ZERO_COPY the kernels read
        the pinned host tables directly and nothing is pushed."""
        if ZERO_COPY:
            self.dirty[:] = False
            return
        ls = np.nonzero(self.dirty)[0] if l is None else [l]
        for li in ls:
            self.ptr[li].copy_(self.ptr_pin[li], non_blocking=True)
            self.res[li].copy_(self.res_pin[li], non_blocking=True)
            self.dirty[li] = False

    PROFILE = "/nvx/routing_profile.npy"

    PROFILE_DECAY = 0.8

    def save_profile(self, path=None):
        """Merge this session's pick counts into the routing profile (the warm order of the next start): older
        sessions count PROFILE_DECAY less per save, so the profile follows recent use. Atomic replace."""
        path = path or self.PROFILE
        if self.uses.sum() == 0:
            return
        h = self.uses.copy()
        if os.path.exists(path):
            try:
                old = np.load(path)
                h[:old.shape[0]] += old[:h.shape[0]] * self.PROFILE_DECAY
            except Exception as e:
                log(f"routing profile unreadable ({e}): starting a new one")
        tmp = path + ".tmp.npy"
        np.save(tmp, h)
        os.replace(tmp, path)
        self.uses[:] = 0
        log(f"routing profile saved ({int(h.sum())} weighted picks)")

    def load_profile(self, path=None):
        path = path or self.PROFILE
        if not os.path.exists(path):
            return None
        h0 = np.load(path)
        h = np.zeros((self.L, self.E), np.float64)
        h[:h0.shape[0]] = h0[:self.L]
        # seed this run's hit counts from the profile (scaled to a few hits), so the warmed experts are not
        # displaced by the first lukewarm candidates
        if h.max() > 0:
            self.hits[:] = h / h.max() * 4.0
        flat = np.argsort(-h, axis=None, kind="stable")
        return [(int(i // self.E), int(i % self.E)) for i in flat if self.K3.flat[(int(i)) * 3] > 0]

    def warm(self, order=None):
        """Fill VRAM, then RAM, with whole experts in `order` ([(l, e)], hottest first; default: random), using
        free slots only (no eviction)."""
        if order is None:
            rng = np.random.default_rng(0)
            order = [(l, int(e)) for e in range(self.E) for l in range(self.L)]
            rng.shuffle(order)
        order = [(l, e) for l, e in order if self.K3[l, e].min() > 0]       # (padding rows of the drafter layers)
        if getattr(self, "draft_layers", None):
            # the drafter's experts run every step: warm them first (VRAM)
            dl = set(self.draft_layers)
            first = [(l, e) for l in self.draft_layers for e in range(self.E) if self.K3[l, e].min() > 0]
            order = first + [(l, e) for l, e in order if l not in dl]
        elif self.mtp_layers:
            dl = set(self.mtp_layers)
            order = [(l, e) for l, e in order if l not in dl]                  # no drafter: leave its experts out
        t0 = time.time()
        nv = nr = 0
        i = 0
        batch = []
        skipped = []

        def flush(to_vram):
            nonlocal batch
            if not batch:
                return
            pks = [pk for b in batch for pk in b]
            self.load_to_ram(pks, set(pks))
            if to_vram:
                for pk in pks:
                    self.promote(pk, set(pks))
                self.release_deferred()
            batch = []

        # phase 1: VRAM (RAM slots stage the reads and are freed by the promotion)
        while i < len(order):
            l, e = order[i]
            base = (l * self.E + e) * 3
            pks = (base, base + 1, base + 2)
            need = {}
            for pk in pks:
                K = int(self.piece_K[pk])
                need[K] = need.get(K, 0) + 1
            i += 1
            if any(len(self.vfree[K]) < n + sum(1 for b in batch for q in b if int(self.piece_K[q]) == K)
                   for K, n in need.items()):
                skipped.append((l, e))             # its K class is full: may still fit in RAM
                if len(skipped) > 4096:
                    break
                continue
            batch.append(pks)
            nv += 1
            if len(batch) >= 48:
                flush(True)
        flush(True)
        order = skipped + list(order[i:])
        i = 0
        torch.xpu.synchronize(self.dev)
        # phase 2: RAM
        miss = 0
        while i < len(order):
            l, e = order[i]
            base = (l * self.E + e) * 3
            pks = (base, base + 1, base + 2)
            if self.vslot[base] >= 0:
                i += 1
                continue
            need = {}
            for pk in pks:
                K = int(self.piece_K[pk])
                need[K] = need.get(K, 0) + 1
            i += 1
            if any(len(self.rfree[K]) < n + sum(1 for b in batch for q in b if int(self.piece_K[q]) == K)
                   for K, n in need.items()):
                miss += 1
                if miss > 4096:
                    break
                continue
            batch.append(pks)
            nr += 1
            if len(batch) >= 64:
                flush(False)
        flush(False)
        self.upload()
        for k in self.stats:
            self.stats[k] = 0.0 if isinstance(self.stats[k], float) else 0
        log(f"warm: {nv} experts in VRAM, {nr} in RAM ({(nv + nr) / (self.L * self.E) * 100:.0f}% of all) in "
            f"{time.time() - t0:.0f} s")


# ------------------------------------------------------------------------------------------------- MoE forward

_HAD = {}


def had128(x: torch.Tensor) -> torch.Tensor:
    """Blockwise 128-point Hadamard / sqrt(128) over the last dim, fp32."""
    H = _HAD.get(x.device)
    if H is None:
        H = exl3_tri.had128(x.device).float()
        _HAD[x.device] = H
    s = x.shape
    return (x.float().reshape(-1, 128) @ H).view(s)


TIER: Tier | None = None
ZERO_COPY = os.environ.get("DSV_ZERO_COPY", "1") == "1"      # kernels read the pinned pointer / residency tables
DECODE_MAX_R = int(os.environ.get("DSV_DECODE_MAX_R", "16"))
AHEAD = os.environ.get("DSV_AHEAD", "1") == "1"
AHEAD_DEPTH = [int(os.environ.get("DSV_AHEAD_DEPTH", "2"))]     # layers predicted ahead (NVMe prefetch; runtime-settable)            # exact mode: prefetch the next layer's predicted experts     # rows that take the GEMV (decode) path
P_GU = int(os.environ.get("DSV_P_GU", "8"))
P_DN = int(os.environ.get("DSV_P_DN", "4"))


def _splits(p, kdim):
    """Split-K partials moe_gemv writes for `p` requested splits over kdim inputs: it gives each split
    ceil(rows / p) of the kdim / 16 rows, so a p that does not divide them leaves trailing splits empty (and unwritten:
    a buffer sized for p would add garbage)."""
    rows = kdim // 16
    per = -(-rows // p)
    return -(-rows // per)


ROUTE_FUSED = os.environ.get("DSV_ROUTE_FUSED", "1") == "1"
ROUTE_CHECK = [int(os.environ.get("DSV_ROUTE_CHECK", "0"))]


PREFILL_BOOST = [float(os.environ.get("DSV_PREFILL_BOOST", "0.3"))]   # prompt rows: residency boost (512-token prompt 47 -> 20 s, NLL +-0.01)
IN_PREFILL = [False]          # set by the runner while model.prefill runs (prompt tokens use exact routing)


def _ptr(l):
    return TIER.ptr_pin[l] if ZERO_COPY else TIER.ptr[l]


def _res(l):
    return TIER.res_pin[l] if ZERO_COPY else TIER.res[l]


def route_decode(mod, y, nosync, ahead, boost=False, S=1, bb=0.0):
    """Decode rows: this layer's router (+ the next layer's on the same input when `ahead`) in one fused op.
    -> ids [R, k], w [R, k], want [R, k] or None, picks [n, R, k]: one tensor for the host readback holding this layer's
    ids, then the next layer's (ahead), then the unboosted top-k (boost: how often the boost changed a pick).
    boost: selection with the tier's residency multipliers (this layer's, and the next layer's for the prediction)."""
    R = y.shape[0]
    k = mod.num_experts_per_tok
    dev = y.device
    ah = ahead_mods(mod) if ahead else []
    nxt = ah[0] if ah else None
    stats = bool(boost) or bb > 0          # (a want row: the unboosted picks)
    np_ = 1 + len(ah) + int(stats)
    picks = torch.empty((np_, R, k), dtype=torch.int64, device=dev)
    w = torch.empty((R, k), dtype=torch.float, device=dev)
    want = torch.empty((R, k), dtype=torch.int64, device=dev) if nosync else (picks[-1] if stats else None)
    w1 = torch.empty((R, k), dtype=torch.float, device=dev) if nxt is not None else None
    logits = _buf(dev, "rlog", (R * 2 * mod.num_experts,), torch.float)
    T = TIER
    G().route_fused(y, mod._gate_w, nxt._gate_w if nxt is not None else None, mod._bias,
                    nxt._bias if nxt is not None else None, _res(mod._li) if nosync else None, picks[0], w, want,
                    picks[1] if nxt is not None else None, w1, logits, float(mod.routed_scaling_factor),
                    float(nxt.routed_scaling_factor) if nxt is not None else 1.0,
                    T.mul_pin[mod._li] if boost else None, T.mul_pin[nxt._li] if boost and nxt is not None else None,
                    S, float(bb))
    for d, am in enumerate(ah[1:], start=2):           # deeper predictions: that layer's router on this input
        wd = _buf(dev, "rwd", (R, k), torch.float)
        G().route_fused(y, am._gate_w, None, am._bias, None, None, picks[d], wd, None, None, None, logits,
                        float(am.routed_scaling_factor), 1.0, T.mul_pin[am._li] if boost else None, None)
    return picks[0], w, (want if nosync else None), picks


ROUTE_DECODE = os.environ.get("DSV_ROUTE_DECODE", "1") == "1"


def route(mod, y, nosync=None):
    """sqrt-softplus noaux_tc routing. y [R, H] fp16 -> (ids [R, k] int64, w [R, k] fp32, wanted ids or None).
    No-sync: the selection key gets the tier's residency bias (0 VRAM, -margin RAM, -inf not resident), so only
    readable experts are picked; the unbiased top-k (what exact routing would pick) is returned for bookkeeping."""
    logits = torch.matmul(y, mod._gate_t).float()                  # [R, E]
    k = mod.num_experts_per_tok
    R = y.shape[0]
    if nosync is None:
        nosync = TIER.nosync
    if IN_PREFILL[0] and PREFILL_BOOST[0] > 0 and not nosync:
        # prompt rows: experts already in VRAM / RAM compete with score * (1 + boost) (weights stay the plain scores),
        # so a short prompt reads far fewer experts from NVMe
        s = F.softplus(logits).sqrt()
        # from the host's copy of the residency table, one small blocking upload (an async copy straight out of the
        # pinned table, which kernels read zero-copy and the pager rewrites, crashed a prompt)
        resident = np.isfinite(TIER.res_h[mod._li])
        mul = torch.from_numpy(np.where(resident, 1.0 + PREFILL_BOOST[0], 1.0).astype(np.float32)).to(y.device)
        # the pick on the CPU: torch.topk on the XPU segfaulted here (host side) on the first prompt chunk of two runs;
        # prompts run synchronously anyway, and this is one 0.8 MB copy per layer
        key = (s * mul + mod._bias).cpu()
        idx = torch.topk(key, k, dim=-1, sorted=False).indices.to(y.device)
        w = s.gather(-1, idx)
        w = w * (mod.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20))
        return idx, w, None
    if ROUTE_FUSED and y.device.type == "xpu":
        ids = torch.empty((R, k), dtype=torch.int64, device=y.device)
        w = torch.empty((R, k), dtype=torch.float, device=y.device)
        want = torch.empty((R, k), dtype=torch.int64, device=y.device) if nosync else None
        G().moe_route(logits, mod._bias, _res(mod._li) if nosync else None, ids, w, want,
                      float(mod.routed_scaling_factor))
        if ROUTE_CHECK[0] > 0:
            ROUTE_CHECK[0] -= 1
            s_ = F.softplus(logits).sqrt()
            key = s_ + mod._bias + (_res(mod._li).to(y.device) if nosync else 0)
            ri = torch.topk(key, k, dim=-1).indices
            rw = s_.gather(-1, ri)
            rw = rw * (mod.routed_scaling_factor / rw.sum(-1, keepdim=True))
            same = (ids.sort(-1).values == ri.sort(-1).values).all().item()
            werr = (w.gather(-1, ids.argsort(-1)) - rw.gather(-1, ri.argsort(-1))).abs().max().item()
            print(f" -- route check L{mod._li}: same picks {same}, weight err {werr:.2g}", flush=True)
        return ids, w, want
    s = F.softplus(logits).sqrt()
    key = s + mod._bias
    want = None
    if nosync:
        want = torch.topk(key, k, dim=-1, sorted=False).indices
        key = key + _res(mod._li).to(y.device)
    idx = torch.topk(key, k, dim=-1, sorted=False).indices
    w = s.gather(-1, idx)
    w = w * (mod.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20))
    return idx, w, want


def moe_decode(mod, y, ids, w):
    """y [R, H] fp16 (R <= 4), ids/w [R, k]. One item per (row, pick, projection)."""
    T = TIER
    l = mod._li
    R, k = ids.shape
    n = R * k
    e = ids.reshape(-1)
    vec = T.vec[l].index_select(0, e)                                 # [n, VEC]
    ptr = _ptr(l).to(y.device).index_select(0, e)                     # [n, 3]
    kk = T.Kdev[l].index_select(0, e)                                 # [n, 3]
    xr = y.repeat_interleave(k, 0).float()                            # [n, H]
    xg = had128(xr * vec[:, OFF_SUH_G:OFF_SUH_G + H].float())
    xu = had128(xr * vec[:, OFF_SUH_U:OFF_SUH_U + H].float())
    xh = torch.stack((xg, xu), 1).half().view(2 * n, 1, H)            # items (pick, g/u)
    wp = ptr[:, :2].reshape(-1).contiguous()
    kg = kk[:, :2].reshape(-1).contiguous()
    cnt = mod._cnt[2 * n]
    part = torch.empty((_splits(P_GU, H), 2 * n, 1, I), dtype=torch.float, device=y.device)
    X().moe_gemv(xh, wp, kg, cnt, part, 0x1E)
    t = had128(part.sum(0).view(n, 2, I))
    g = t[:, 0] * vec[:, OFF_SVH_G:OFF_SVH_G + I].float()
    u = t[:, 1] * vec[:, OFF_SVH_U:OFF_SVH_U + I].float()
    a = (F.silu(g).clamp_max(ACT_LIMIT) * u.clamp(-ACT_LIMIT, ACT_LIMIT))
    xd = had128(a * vec[:, OFF_SUH_D:OFF_SUH_D + I].float()).half().view(n, 1, I)
    part2 = torch.empty((_splits(P_DN, I), n, 1, H), dtype=torch.float, device=y.device)
    X().moe_gemv(xd, ptr[:, 2].contiguous(), kk[:, 2].contiguous(), mod._cnt[n], part2, 0x1E)
    yd = had128(part2.sum(0).view(n, H)) * vec[:, OFF_SVH_D:OFF_SVH_D + H].float()
    return (yd.view(R, k, H) * w.unsqueeze(-1)).sum(1)               # [R, H] fp32


class Stager:
    """Prompt-path expert streaming: the pieces of a layer that are in neither VRAM nor RAM are read from NVMe into a
    pinned staging area (not the RAM tier, so a long prompt does not flush it), one layer ahead of the compute. Two
    areas alternate between layers; each area is several pinned chunks (< 2 GiB each); the GPU reads staged pieces
    zero-copy."""

    def __init__(self, tier: Tier, area_bytes: int):
        self.T = tier
        per = (2**31 - 1) // UNIT                  # units per chunk
        total = max(1, area_bytes // UNIT)
        self.areas = []
        for _ in range(2):
            chunks, left = [], total
            while left > 0:
                n = min(per, left)
                chunks.append(host_alloc_units(n) + (n,))
                left -= n
            self.areas.append(chunks)
        self.state = {}             # layer -> (area index, {pk: (chunk, unit offset, future)})
        self.busy = [None, None]
        self.events = [None, None]

    def start(self, l, used=None):
        """Queue the NVMe reads of layer l's non-resident pieces (experts `used`, default all)."""
        T = self.T
        if l in self.state or l < 0 or l >= T.L:
            return
        a = 0 if self.busy[0] is None else (1 if self.busy[1] is None else -1)
        if a < 0:
            return
        if self.events[a] is not None:
            self.events[a].synchronize()            # the GPU is done reading the area's previous layer
        self.busy[a] = l
        chunks = self.areas[a]
        used = range(T.E) if used is None else used
        jobs = {}
        c, o = 0, 0
        full = False
        for e in used:
            base = (l * T.E + int(e)) * 3
            for p in range(3):
                pk = base + p
                if T.vslot[pk] >= 0 or T.rslot[pk] >= 0:
                    continue
                K = int(T.piece_K[pk])
                while c < len(chunks) and o + K > chunks[c][2]:
                    c, o = c + 1, 0
                if c >= len(chunks):
                    full = True
                    break
                jobs[pk] = (c, o, chunks[c][1][o * UNIT:(o + K) * UNIT])
                o += K
            if full:
                break
        futs = T.read_async(list(jobs), [d for _, _, d in jobs.values()])
        self.state[l] = (a, {pk: (c, o, f) for (pk, (c, o, _)), f in zip(jobs.items(), futs)})

    def addr(self, l, pk):
        st = self.state.get(l)
        if st is None:
            return 0
        a, jobs = st
        j = jobs.get(pk)
        if j is None:
            return 0
        c, o, f = j
        if not f.done():
            t0 = time.time()
            f.result()
            PREFILL_STATS["wait"] += time.time() - t0
        return s64(self.areas[a][c][0].data_ptr() + o * UNIT)

    def finish(self, l, dev):
        st = self.state.pop(l, None)
        if st is None:
            return
        a, jobs = st
        for c, o, f in jobs.values():
            f.result()
        ev = torch.xpu.Event()
        ev.record(torch.xpu.current_stream(dev))
        self.events[a] = ev
        self.busy[a] = None


def host_alloc_units(n):
    """n units of pinned memory in < 2 GiB chunks... as one logical area: allocate a single tensor per area when it
    fits, else fall back to the largest single chunk (areas are sized to fit)."""
    nbytes = n * UNIT
    assert nbytes < 2**31, "staging area must stay under 2 GiB"
    t = host_alloc(nbytes)
    return t, t.numpy()


STAGER: Stager | None = None
PREFILL_STATS = dict(moe=0.0, wait=0.0)
STAGE_MIN_ROWS = int(os.environ.get("DSV_STAGE_MIN_ROWS", "64"))
PREFILL_TOUCH_MAX = int(os.environ.get("DSV_PREFILL_TOUCH_MAX", "2048"))


_G = None


def G():
    global _G
    if _G is None:
        torch.ops.load_library(os.path.join(os.path.dirname(__file__), "..", "lib", "dsv_glue.so"))
        _G = torch.ops.dsv_glue
    return _G


GLUE = os.environ.get("DSV_GLUE", "1") == "1"
GLUE_CHECK = int(os.environ.get("DSV_GLUE_CHECK", "0"))
_scr = {}


def _buf(dev, tag, shape, dtype):
    key = (tag, shape, dtype)
    t = _scr.get(key)
    if t is None:
        t = _scr[key] = torch.empty(shape, dtype=dtype, device=dev)
    return t


def moe_decode_fused(mod, y, ids, w):
    """moe_decode with the glue kernels: prep -> gemv(gate|up) -> mid -> gemv(down) -> post."""
    T = TIER
    l = mod._li
    R, k = ids.shape
    n = R * k
    dev = y.device
    y = y.contiguous()
    ids = ids.contiguous()
    xh = _buf(dev, "xh", (2 * n, 1, H), torch.half)
    wp = _buf(dev, "wp", (2 * n,), torch.int64)
    kk = _buf(dev, "kk", (2 * n,), torch.int32)
    wpd = _buf(dev, "wpd", (n,), torch.int64)
    kkd = _buf(dev, "kkd", (n,), torch.int32)
    G().moe_prep(y, ids, T.vec[l], _ptr(l), T.Kdev[l], xh, wp, kk, wpd, kkd)
    part = _buf(dev, "p1", (_splits(P_GU, H), 2 * n, 1, I), torch.float)
    X().moe_gemv(xh, wp, kk, mod._cnt[2 * n], part, 0x1E)
    xd = _buf(dev, "xd", (n, 1, I), torch.half)
    G().moe_mid(part, ids, T.vec[l], xd, ACT_LIMIT)
    part2 = _buf(dev, "p2", (_splits(P_DN, I), n, 1, H), torch.float)
    X().moe_gemv(xd, wpd, kkd, mod._cnt[n], part2, 0x1E)
    out = torch.empty((R, H), dtype=torch.float, device=dev)
    G().moe_post(part2, ids, w.float().contiguous(), T.vec[l], out)
    return out


GROUPED = os.environ.get("DSV_MOE_GROUPED", "1") == "1"
GROUPED_CHECK = [int(os.environ.get("DSV_GROUPED_CHECK", "0"))]


def moe_decode_grouped(mod, y, ids, w):
    """Multi-row decode (a verify block, R = 2..4 rows): one GEMV item per (unique expert, projection) over all R rows,
    so rows that picked the same expert read its weights once (dedupe on the device: graph-capturable)."""
    T = TIER
    l = mod._li
    R, k = ids.shape
    U = R * k
    dev = y.device
    y = y.contiguous()
    ids = ids.contiguous()
    uniq = _buf(dev, "g_u", (U,), torch.int64)
    slot = _buf(dev, "g_s", (U,), torch.int32)
    nu = _buf(dev, "g_n", (2,), torch.int32)
    G().moe_group(ids, uniq, slot, nu)
    xh = _buf(dev, "g_xh", (2 * U, R, H), torch.half)
    wp = _buf(dev, "g_wp", (2 * U,), torch.int64)
    kk = _buf(dev, "g_kk", (2 * U,), torch.int32)
    wpd = _buf(dev, "g_wpd", (U,), torch.int64)
    kkd = _buf(dev, "g_kkd", (U,), torch.int32)
    G().moe_prep_g(y, uniq, T.vec[l], _ptr(l), T.Kdev[l], xh, wp, kk, wpd, kkd)
    part = _buf(dev, "g_p1", (_splits(P_GU, H), 2 * U, R, I), torch.float)
    X().moe_gemv(xh, wp, kk, nu[0:1], part, 0x1E)
    xd = _buf(dev, "g_xd", (U, R, I), torch.half)
    G().moe_mid_g(part, uniq, T.vec[l], xd, R, ACT_LIMIT)
    part2 = _buf(dev, "g_p2", (_splits(P_DN, I), U, R, H), torch.float)
    X().moe_gemv(xd, wpd, kkd, nu[1:2], part2, 0x1E)
    out = torch.empty((R, H), dtype=torch.float, device=dev)
    G().moe_post_g(part2, ids, slot, w.float().contiguous(), T.vec[l], out)
    return out


def moe_prefill(mod, y, ids, w):
    t0 = time.time()
    try:
        return _moe_prefill(mod, y, ids, w)
    finally:
        PREFILL_STATS["moe"] += time.time() - t0


def _moe_prefill(mod, y, ids, w):
    """Many rows: per expert with rows, dequantize its three pieces (wherever they are) and run fp16 GEMMs."""
    T = TIER
    l = mod._li
    R, k = ids.shape
    out = torch.zeros((R, H), dtype=torch.float, device=y.device)
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=T.E).cpu().numpy()
    rows = (order // k)
    wf = w.reshape(-1)[order]
    used = np.nonzero(counts)[0]
    stg = STAGER
    short_stage = PREFILL_TAIL > 0 and stg is not None and R >= STAGE_MIN_ROWS
    if R <= PREFILL_TOUCH_MAX and short_stage:
        # short prompts: only the experts of the last rows (what decode will want next) go into the RAM tier; the rest
        # stream through the staging areas, so the prompt does not flush the cache with experts decode never uses
        tail = np.unique(ids[-min(R, PREFILL_TAIL):].reshape(-1).cpu().numpy())
        T.touch(l, tail, fetch=True)
        T.upload(l)
        stg.start(l, used)
        if l + 1 < T.L and R * k >= 2 * T.E:
            stg.start(l + 1)                # nearly every expert of the next layer is used
    elif R <= PREFILL_TOUCH_MAX:
        # short prompts: read their non-resident experts into the RAM tier (LRU), so decode finds them resident
        T.touch(l, used, fetch=True)
        T.upload(l)
    elif stg is not None and R >= STAGE_MIN_ROWS:
        stg.start(l, used)
        nxt = l + 1 if l + 1 < T.L else None
        if nxt is not None:
            stg.start(nxt)                  # next layer: all experts (a long prompt uses nearly all of them)
    ptr_h = T.ptr_h.reshape(T.L, T.E, 3)
    if PREFILL_GROUPED:
        addrs = {}
        miss = []
        for e in used:
            for p in range(3):
                pk = (l * T.E + int(e)) * 3 + p
                K = int(T.K3[l, e, p])
                a = int(ptr_h[l, e, p])
                if a == 0 and stg is not None:
                    a = stg.addr(l, pk)
                if a == 0:
                    miss.append((int(e), p, pk, K))
                addrs[(int(e), p)] = (a, K)
        scratch_bufs = _scratch_read(T, miss, addrs) if miss else None
        out = moe_prefill_grouped(mod, y, ids, w, addrs, counts, rows, wf)
        if PREFILL_CHECK[0] > 0:
            PREFILL_CHECK[0] -= 1
            globals()["PREFILL_GROUPED"] = False
            ref = _moe_prefill(mod, y, ids, w)
            globals()["PREFILL_GROUPED"] = True
            print(f" -- prefill grouped check L{l} R={R}: rel err {((out - ref).abs().max() / ref.abs().max()).item():.3g}",
                  flush=True)
        if stg is not None:
            stg.finish(l, y.device)
        if scratch_bufs:
            torch.xpu.synchronize(y.device)
        return out
    start = 0
    scratch = {}
    if PREFILL_PROBE:
        torch.xpu.synchronize(y.device)
        globals()["_t_loop0"] = time.time()
    for e in range(T.E):
        c = int(counts[e])
        if c == 0:
            continue
        r = rows[start:start + c]
        ww = wf[start:start + c]
        start += c
        v = T.vec[l, e]
        x = y.index_select(0, r).float()
        Ws = []
        keep = []
        for p in range(3):
            K = int(T.K3[l, e, p])
            pk = (l * T.E + e) * 3 + p
            addr = int(ptr_h[l, e, p])
            if addr == 0 and stg is not None:
                addr = stg.addr(l, pk)
            if addr == 0:
                buf = _scratch_take(y.device)
                T._read(pk, buf.numpy())
                addr = s64(buf.data_ptr())
                keep.append(buf)                 # released (event recorded) after this expert's kernels are enqueued
                PREFILL_STATS["scratch"] = PREFILL_STATS.get("scratch", 0.0) + 1
            Ws.append((addr, K))
        Wg = dequant_addr(Ws[0][0], Ws[0][1], H, I, y.device)
        xg = had128(x * v[OFF_SUH_G:OFF_SUH_G + H].float()).half()
        g = had128((xg @ Wg).float()) * v[OFF_SVH_G:OFF_SVH_G + I].float()
        del Wg
        Wu = dequant_addr(Ws[1][0], Ws[1][1], H, I, y.device)
        xu = had128(x * v[OFF_SUH_U:OFF_SUH_U + H].float()).half()
        u = had128((xu @ Wu).float()) * v[OFF_SVH_U:OFF_SVH_U + I].float()
        del Wu
        a = F.silu(g).clamp_max(ACT_LIMIT) * u.clamp(-ACT_LIMIT, ACT_LIMIT)
        Wd = dequant_addr(Ws[2][0], Ws[2][1], I, H, y.device)
        xd = had128(a * v[OFF_SUH_D:OFF_SUH_D + I].float()).half()
        d = had128((xd @ Wd).float()) * v[OFF_SVH_D:OFF_SVH_D + H].float()
        out.index_add_(0, r, d * ww.unsqueeze(-1))
    if PREFILL_PROBE:
        t1 = time.time()
        torch.xpu.synchronize(y.device)
        t2 = time.time()
        PREFILL_STATS["enqueue"] = PREFILL_STATS.get("enqueue", 0.0) + (t1 - _t_loop0)
        PREFILL_STATS["gpu_tail"] = PREFILL_STATS.get("gpu_tail", 0.0) + (t2 - t1)
    if stg is not None:
        stg.finish(l, y.device)
    return out


PREFILL_PROBE = os.environ.get("DSV_PREFILL_PROBE", "0") == "1"
PREFILL_TAIL = int(os.environ.get("DSV_PREFILL_TAIL", "0"))     # short prompts: rows whose experts go to the RAM tier
_SCRB = []          # reusable pinned chunks for prompt pieces that are neither resident nor staged


def _scratch_read(T, miss, addrs):
    """Read the pieces `miss` [(e, p, pk, K)] from NVMe in parallel into reused pinned chunks; fills addrs. The caller
    synchronizes before the chunks are reused."""
    per = (2**31 - 1) // UNIT
    ci, o = 0, 0
    reads = []
    for e, p, pk, K in miss:
        if ci < len(_SCRB) and o + K > _SCRB[ci][2]:
            ci, o = ci + 1, 0
        if ci >= len(_SCRB):
            n = min(per, max(64, sum(m[3] for m in miss)))
            t = host_alloc(n * UNIT)
            _SCRB.append((t, t.numpy(), n))
            o = 0
        t, a_np, n = _SCRB[ci]
        reads.append((pk, a_np[o * UNIT:(o + K) * UNIT]))
        addrs[(e, p)] = (s64(t.data_ptr() + o * UNIT), K)
        o += K
    t0 = time.time()
    T.read_many([pk for pk, _ in reads], [d for _, d in reads])
    PREFILL_STATS["wait"] += time.time() - t0
    PREFILL_STATS["scratch"] = PREFILL_STATS.get("scratch", 0.0) + len(miss)
    return True


_SCR = []          # ring of (pinned buffer, event) for pieces that are neither resident nor staged
_SCR_I = [0]


def _scratch_take(dev, n=24):
    if not _SCR:
        for _ in range(n):
            _SCR.append([host_alloc(4 * UNIT), None])
    ent = _SCR[_SCR_I[0] % len(_SCR)]
    _SCR_I[0] += 1
    if ent[1] is not None:
        ent[1].synchronize()                    # the kernels that read this buffer last time are done
    ev = torch.xpu.Event()
    ent[1] = ev
    return _ScratchBuf(ent, ev, dev)


class _ScratchBuf:
    """A ring buffer whose event is recorded after the caller's use (record() in __del__ is too late: record now,
    after enqueueing, by recording at the next take: the event marks 'kernels enqueued before the next take')."""

    def __init__(self, ent, ev, dev):
        self.t = ent[0]
        self.ev = ev
        self.dev = dev

    def numpy(self):
        return self.t.numpy()

    def data_ptr(self):
        return self.t.data_ptr()

    def __del__(self):
        try:
            self.ev.record(torch.xpu.current_stream(self.dev))
        except Exception:
            pass


PREFILL_GROUPED = os.environ.get("DSV_PREFILL_GROUPED", "1") == "1"
GROUP = int(os.environ.get("DSV_PREFILL_GROUP", "16"))
PREFILL_CHECK = [int(os.environ.get("DSV_PREFILL_CHECK", "0"))]
ROWCAP = int(os.environ.get("DSV_PREFILL_ROWCAP", "6144"))


def _had_b(x, scale_vec):
    """had128(x * scale) for x [..., n] fp32 with scale broadcast over leading dims."""
    return had128(x * scale_vec)


def moe_prefill_grouped(mod, y, ids, w, addrs, counts, rows, wf):
    """Prompt MoE in groups of GROUP experts: one dequant launch per (projection, K) per group into an fp16 weight
    stack, then batched GEMMs over zero-padded row blocks."""
    T = TIER
    l = mod._li
    R = y.shape[0]
    dev = y.device
    out = torch.zeros((R, H), dtype=torch.float, device=dev)
    used = sorted((e for e in range(T.E) if counts[e] > 0), key=lambda e: -counts[e])
    starts = np.concatenate([[0], np.cumsum(counts)])
    # groups of similar row counts (sorted), capped by padded rows so the stacked buffers stay small
    groups, cur = [], []
    for e in used:
        if cur and (len(cur) >= GROUP or (len(cur) + 1) * int(counts[cur[0]]) > ROWCAP):
            groups.append(cur)
            cur = []
        cur.append(e)
    if cur:
        groups.append(cur)
    NPK = ids.numel()                         # positions index the expert-sorted pick list (R * k long)
    tabs = _group_tables(groups, counts, starts, addrs, NPK, dev) if BATCH_UPLOAD else None
    ypad = torch.cat((y, torch.zeros((1, H), dtype=y.dtype, device=dev)), 0)
    for gi, grp in enumerate(groups):
        n = len(grp)
        mr = int(max(counts[e] for e in grp))
        # padded row index table (R = a zero row appended to y)
        if tabs is not None:
            flat = tabs[0][gi]
        else:
            idx = torch.full((n, mr), NPK, dtype=torch.long)
            for i, e in enumerate(grp):
                c = int(counts[e])
                idx[i, :c] = torch.arange(starts[e], starts[e] + c)
            flat = idx.to(dev).view(-1)
        valid = flat < NPK
        src = torch.where(valid, flat, torch.zeros_like(flat))
        rsel = torch.where(valid, rows[src], torch.full_like(flat, R))
        wsel = torch.where(valid, wf[src], torch.zeros_like(wf[src]))
        xg = ypad.index_select(0, rsel).view(n, mr, H).float()
        vec = T.vec[l].index_select(0, tabs[1][gi] if tabs is not None else torch.tensor(grp, device=dev))  # [n, VEC]
        Wb = _wbuf(dev, n, H, I)
        res = []
        for p, o_suh, o_svh in ((0, OFF_SUH_G, OFF_SVH_G), (1, OFF_SUH_U, OFF_SVH_U)):
            if tabs is not None:
                for K, pt, sl in tabs[2][gi][p]:
                    exl3_tri.dequant_multi(pt, sl, K, H, I, Wb)
            else:
                _dequant_group(addrs, grp, l, p, H, I, Wb)
            xh = _had_b(xg, vec[:, None, o_suh:o_suh + H].float()).half()
            t = torch.bmm(xh, Wb[:n])
            res.append(had128(t.float()) * vec[:, None, o_svh:o_svh + I].float())
        gg, uu = res
        a = F.silu(gg).clamp_max(ACT_LIMIT) * uu.clamp(-ACT_LIMIT, ACT_LIMIT)
        del gg, uu, res
        Wd = _wbuf(dev, n, I, H, tag="d")
        if tabs is not None:
            for K, pt, sl in tabs[2][gi][2]:
                exl3_tri.dequant_multi(pt, sl, K, I, H, Wd)
        else:
            _dequant_group(addrs, grp, l, 2, I, H, Wd)
        xd = _had_b(a, vec[:, None, OFF_SUH_D:OFF_SUH_D + I].float()).half()
        d = had128(torch.bmm(xd, Wd[:n]).float()) * vec[:, None, OFF_SVH_D:OFF_SVH_D + H].float()
        d = d.view(n * mr, H) * wsel.unsqueeze(-1)
        keep = rsel < R
        out.index_add_(0, rsel[keep], d[keep])
    return out


BATCH_UPLOAD = os.environ.get("DSV_PREFILL_BATCH_UPLOAD", "1") == "1"


def _group_tables(groups, counts, starts, addrs, NPK, dev):
    """Every group's row index table, expert ids and (address, slot) lists in one host array, uploaded once per
    layer (one small blocking copy instead of hundreds: each torch.tensor(..., device=xpu) waited for the GPU queue).
    Returns per group: flat row positions, expert ids, and per projection [(K, ptrs, slots)] device views."""
    parts, meta = [], []
    off = 0

    def put(a):
        nonlocal off
        a = np.asarray(a, dtype=np.int64)
        parts.append(a)
        o = off
        off += a.size
        return (o, a.size)
    for grp in groups:
        n = len(grp)
        mr = int(max(counts[e] for e in grp))
        idx = np.full((n, mr), NPK, dtype=np.int64)
        for i, e in enumerate(grp):
            c = int(counts[e])
            idx[i, :c] = np.arange(starts[e], starts[e] + c)
        g_idx = put(idx.reshape(-1))
        g_ids = put(grp)
        projs = []
        for p in range(3):
            byK = {}
            for i, e in enumerate(grp):
                a, K = addrs[(int(e), p)]
                byK.setdefault(K, ([], []))
                byK[K][0].append(a)
                byK[K][1].append(i)
            projs.append([(K, put(ps), put(sl)) for K, (ps, sl) in byK.items()])
        meta.append((g_idx, g_ids, projs))
    d64 = torch.from_numpy(np.concatenate(parts)).to(dev)
    d32 = d64.to(torch.int32)          # slot lists (addresses overflow here; only slot slices are read)
    sl64 = lambda t: d64[t[0]:t[0] + t[1]]
    sl32 = lambda t: d32[t[0]:t[0] + t[1]]
    return ([sl64(m[0]) for m in meta], [sl64(m[1]) for m in meta],
            [[[(K, sl64(pt), sl32(st)) for K, pt, st in pr] for pr in m[2]] for m in meta])


_WB = {}


def _wbuf(dev, n, kd, nd, tag="gu"):
    key = (tag, kd, nd)
    b = _WB.get(key)
    if b is None or b.shape[0] < GROUP:
        b = _WB[key] = torch.empty((GROUP, kd, nd), dtype=torch.half, device=dev)
    return b


def _dequant_group(addrs, grp, l, p, kd, nd, Wb):
    """addrs: {(e, p): (addr, K)} -> Wb[i] = W_hat of grp[i]'s projection p."""
    byK = {}
    for i, e in enumerate(grp):
        a, K = addrs[(e, p)]
        byK.setdefault(K, ([], []))
        byK[K][0].append(a)
        byK[K][1].append(i)
    for K, (ps, sl) in byK.items():
        exl3_tri.dequant_multi(torch.tensor(ps, dtype=torch.int64, device=Wb.device),
                               torch.tensor(sl, dtype=torch.int32, device=Wb.device), K, kd, nd, Wb)


def dequant_addr(addr: int, K: int, kdim: int, n: int, device) -> torch.Tensor:
    out = torch.empty((kdim, n), dtype=torch.half, device=device)
    exl3_tri.dequant_ptr(addr, K, kdim, n, out)
    return out


def install(tier: Tier, model):
    """Swap BlockSparseMLP.load/forward for the tiered versions and link every MoE module to its store layer."""
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    global TIER
    TIER = tier
    BlockSparseMLP.load = _load
    BlockSparseMLP.forward = _forward


MTP_BASE = 40          # store layer of the DSpark drafter's mtp.0 MoE (pack_store.append_mtp)


def _model_layer(key: str) -> int:
    if key.startswith("mtp."):
        return MTP_BASE + int(key.split(".")[1])
    return int(key.split("layers.")[1].split(".")[0])


def _load(self, device, **kwargs):
    self.device = device
    experts = {id(m) for m in self.gates + self.ups + self.downs}
    for m in self.modules:
        if id(m) not in experts:
            m.load(device, **kwargs)
    stc = self.config.stc
    b = stc.get_tensor(f"{self.key}.gate.bias", device, optional=True, allow_bf16=True, no_defer=True)
    self._bias = b.float() if b is not None else torch.zeros(self.num_experts, device=device)
    gw = self.routing_gate.inner.weight                                # [H, E] (exllamav3 stores in x out)
    self._gate_w = gw.half().t().contiguous()                          # [E, H]: rows for the router GEMV
    self._gate_t = self._gate_w.t()
    try:
        self.routing_gate.inner.weight = self._gate_t                  # keep one copy
    except Exception:
        pass
    self._li = _model_layer(self.key)
    self._cnt = {n: torch.tensor([n], dtype=torch.int32, device=device) for n in range(1, 2 * DECODE_MAX_R * 8 + 1)}


def _forward(self, x, params, out_dtype=None):
    y = x.view(-1, self.hidden_size)
    if y.dtype != torch.half:
        y = y.half()
    R = y.shape[0]
    T = TIER
    nosync = T.nosync and not IN_PREFILL[0]
    boost = T.boost > 0 and not nosync and not IN_PREFILL[0]
    S_rows = x.shape[-2] if x.dim() == 3 else 1
    # (main-model verify blocks only: the drafter's own blocks keep its routing; its experts are in VRAM anyway)
    bb = T.block_boost if (S_rows > 1 and not nosync and not IN_PREFILL[0] and self._li < MTP_BASE) else 0.0
    if ROUTE_DECODE and R <= 8 and y.device.type == "xpu":
        ids, w, want, picks = route_decode(self, y, nosync, AHEAD and not nosync, boost, S_rows, bb)
    else:
        ids, w, want = route(self, y, nosync)
        picks = None
    if R <= DECODE_MAX_R:
        if not nosync:
            nxt = getattr(self, "_next_moe", None)
            n = ids.numel()
            ah = [m._li for m in ahead_mods(self)] if picks is not None else ([nxt._li] if AHEAD and nxt is not None else [])
            n_next = n if ah else 0
            boost = (boost or bb > 0) and picks is not None
            if picks is not None:
                both = picks.view(-1)
            elif AHEAD and nxt is not None:
                pids, _, _ = route(nxt, y, False)       # next layer's router on this input: a residency hint
                both = torch.cat((ids.view(-1), pids.view(-1)))
            else:
                both = ids.view(-1)
            if ASYNC[0] and picks is not None:
                # no host wait: the GPU publishes the picks and spins at a sync point until the pager thread has made
                # them resident (see pager.py)
                PAGER.sync(self._li, picks, R, tuple(ah), boost, push=GRAPH[0] is None)
            else:
                t0 = time.perf_counter()
                both = both.cpu().numpy()
                T.stats["t_sync"] = T.stats.get("t_sync", 0.0) + time.perf_counter() - t0
                host_layer(T, self._li, both, R, n, tuple(ah), boost)
        else:
            self._pending = (ids, want)
        if GLUE and GROUPED and 2 <= R <= 4:
            out = moe_decode_grouped(self, y, ids, w)
            if GROUPED_CHECK[0] > 0:
                GROUPED_CHECK[0] -= 1
                ref = moe_decode_fused(self, y, ids, w)
                print(f" -- grouped check L{self._li} R={R}: rel err "
                      f"{((out - ref).abs().max() / ref.abs().max()).item():.3g}", flush=True)
        elif GLUE:
            out = moe_decode_fused(self, y, ids, w)
            if GLUE_CHECK > 0:
                globals()["GLUE_CHECK"] = GLUE_CHECK - 1
                ref = moe_decode(self, y, ids, w)
                print(f" -- glue check L{self._li} R={R}: rel err "
                      f"{((out - ref).abs().max() / ref.abs().max()).item():.3g}", flush=True)
        else:
            out = moe_decode(self, y, ids, w)
    else:
        out = moe_prefill(self, y, ids, w)
    if self.shared_experts is not None:
        if R <= 8 and SHARED_FAST and _shared_fast_ok(self.shared_experts):
            out += _shared_fast(self.shared_experts, y)
        else:
            out += self.shared_experts.forward(x, params).view(out.shape).float()
    out = out.view(x.shape[:-1] + (self.hidden_size,))
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out


SHARED_FAST = os.environ.get("DSV_SHARED_FAST", "1") == "1"


def _shared_fast_ok(se):
    ok = getattr(se, "_dsv_fast", None)
    if ok is None:
        from .fp8 import FP8Linear
        ok = (getattr(se, "num_slices", 1) == 1 and getattr(se, "interm_div", 1.0) == 1.0
              and all(isinstance(l.inner, FP8Linear) for l in (se.gates[0], se.ups[0], se.downs[0]))
              and se.gates[0].inner.in_features % 256 == 0 and se.downs[0].inner.in_features % 256 == 0)
        if ok:
            g, u = se.gates[0].inner, se.ups[0].inner
            wgu = torch.cat((g.w, u.w), 0).contiguous()
            sgu = torch.cat((g.s, u.s), 0).contiguous()
            n = g.w.shape[0]
            g.w, g.s, u.w, u.s = wgu[:n], sgu[:n // 32], wgu[n:], sgu[n // 32:]
            se._dsv_gu = (wgu, sgu, n)
        se._dsv_fast = ok
    return ok


def _shared_fast(se, y):
    """Shared expert for decode rows: one FP8 GEMV for gate | up, SYCL SiLU * up (clamped), FP8 GEMV down."""
    from .fp8 import S as F8
    from .attn import A
    wgu, sgu, n = se._dsv_gu
    R = y.shape[0]
    gu = torch.empty((R, 2 * n), dtype=torch.half, device=y.device)
    F8().fp8_gemv(y.view(1, R, -1), wgu, sgu, gu, 2 * n)
    a = torch.empty((R, n), dtype=torch.half, device=y.device)
    A().silu_mul(gu, a, float(se.act_limit or 0.0))
    d = se.downs[0].inner
    o = torch.empty((R, d.out_features), dtype=torch.float, device=y.device)
    F8().fp8_gemv(a.view(1, R, n), d.w, d.s, o, d.out_features)
    return o


PICKLOG = [] if os.environ.get("DSV_PICKLOG") == "1" else None     # (layer, picks) per decode layer (analysis)
ASYNC = [0]              # set by the runner for a decode step whose layers sync through the pager (pager.py)
GRAPH = [None]           # set by the runner for a graph step (key of the block graphs; layers handed over up front)
PAGER = None


def host_layer(T, li, both, R, n, ahead, boost, release=None, step=None):
    """Host side of one exact-routing decode layer: `both` = this layer's picks (n), then the predicted picks of the
    layers in `ahead` (n each, nearest first), then (boost) the unboosted picks (n). Makes the picks resident (NVMe
    reads of misses), keeps the LRU / hit counts, and starts background reads of the predicted experts.
    release(): called as soon as the picks are resident (the GPU may go on); the bookkeeping follows."""
    T.reap()
    if PICKLOG is not None:
        PICKLOG.append((li, both[:n].copy()))
    if release is not None:
        uniq = np.unique(both[:n])
        keys = ((li * T.E + uniq) * 3)[:, None] + np.arange(3)
        keys = keys.reshape(-1)
        miss = keys[(T.vslot[keys] < 0) & (T.rslot[keys] < 0)]
        if len(miss):
            T.load_to_ram([int(k) for k in miss], set(int(k) for k in keys))
        release()
    preds = [(al, both[n * (d + 1):n * (d + 2)]) for d, al in enumerate(ahead) if al >= 0]
    if step is not None and preds:
        T.stage(preds[0][0], preds[0][1], step)        # (staging: off by default)
    act = np.unique(both[:n])
    for d in range(1, 4):
        pr = T.pred.pop((li, d), None)
        if pr is not None:
            T.stats[f"pred_hit{d}"] = T.stats.get(f"pred_hit{d}", 0) + len(np.intersect1d(act, pr))
            T.stats[f"pred_n{d}"] = T.stats.get(f"pred_n{d}", 0) + len(act)
    for d, (al, ids) in enumerate(preds, start=1):
        T.pred[(al, d)] = np.unique(ids)
    if boost:
        # boost bookkeeping: picks that differ from the unboosted top-k (per row, as sets)
        a, b = both[:n].reshape(R, -1), both[-n:].reshape(R, -1)
        T.stats["boost_picks"] = T.stats.get("boost_picks", 0) + n
        T.stats["boost_changed"] = T.stats.get("boost_changed", 0) + sum(
            a.shape[1] - len(np.intersect1d(a[r], b[r])) for r in range(R))
    T.touch(li, both[:n], fetch=True)
    T.upload(li)
    if T.promo_on and step is not None:
        T.promote_hot(li, act, {(li * T.E + int(e)) * 3 + p for e in act for p in range(3)})
    if preds:
        base = li * T.E * 3
        prot = {base + int(e) * 3 + p for e in act for p in range(3)}
        for al, ids in preds:                          # nearest layer first: its reads are the most urgent
            T.prefetch(al, ids, prot)


def link(mods):
    for i, a in enumerate(mods):
        a._next_moe = mods[i + 1] if i + 1 < len(mods) else None
        a._ahead_mods = mods[i + 1:i + 4]            # up to 3 layers ahead (AHEAD_DEPTH picks how many)


def ahead_mods(mod):
    return getattr(mod, "_ahead_mods", [])[:AHEAD_DEPTH[0]] if AHEAD else []


def moe_modules(model):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    mods = []

    def walk(m):
        if isinstance(m, BlockSparseMLP):
            mods.append(m)
        for c in getattr(m, "modules", []):
            walk(c)
    for m in model.modules:
        walk(m)
    return mods


def drain(mods):
    """No-sync: after a step, read its picks back and do the bookkeeping; then publish the tables for the next step.
    Call between steps (the step's GPU work must be complete or the readback waits for it)."""
    T = TIER
    T.reap()
    pend = [(m, m._pending) for m in mods if getattr(m, "_pending", None) is not None]
    if pend:
        allp = torch.cat([torch.cat((ids.reshape(-1), want.reshape(-1))) for _, (ids, want) in pend]).cpu().numpy()
        budget = [T.max_promote]
        o = 0
        k = pend[0][1][0].shape[-1]
        for m, (ids, want) in pend:
            n1, n2 = ids.numel(), want.numel()
            a = allp[o:o + n1].reshape(-1, k)
            b = allp[o + n1:o + n1 + n2].reshape(-1, k)
            for r in range(a.shape[0]):
                T.stats["exact_picks"] = T.stats.get("exact_picks", 0) + k
                T.stats["exact_served"] = T.stats.get("exact_served", 0) + len(np.intersect1d(a[r], b[r]))
            T.bookkeep(m._li, allp[o:o + n1], allp[o + n1:o + n1 + n2], budget)
            o += n1 + n2
            m._pending = None
    T.release_deferred()
    T.upload()
