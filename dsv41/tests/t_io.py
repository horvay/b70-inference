"""NVMe read throughput of expert pieces: the Python reader pool (Tier._read) vs the C reader (dsv_io.read_pieces).
CPU only.  N=2000 python3 tests/t_io.py"""
import os, json, time, mmap, ctypes, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
UNIT = 5120 * 2304 // 8
meta = json.load(open("/nvx/experts.json"))
rows = np.array(meta["layers"], np.int64)
K3 = rows[..., :3]
off = rows[..., 3]
po = np.stack([off, off + K3[..., 0], off + K3[..., 0] + K3[..., 1]], -1)
piece_off = (po * UNIT).reshape(-1)
piece_K = K3.reshape(-1)
fd = os.open("/nvx/experts.bin", os.O_RDONLY | os.O_DIRECT)
N = int(os.environ.get("N", "2000"))
rng = np.random.default_rng(int(time.time()))
valid = np.nonzero(piece_K > 0)[0]
lib = ctypes.CDLL("/w/dsv41/xpu/lib/dsv_io.so")
lib.read_pieces.restype = ctypes.c_int64
lib.read_pieces.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                            ctypes.c_int, ctypes.c_int64]
tl = threading.local()


def pyread(pk, dst):
    n = int(piece_K[pk]) * UNIT
    if not hasattr(tl, "buf"):
        tl.buf = mmap.mmap(-1, 4 * UNIT, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        tl.np = np.frombuffer(tl.buf, dtype=np.uint8)
    mv = memoryview(tl.buf)
    got = 0
    base = int(piece_off[pk])
    while got < n:
        got += os.preadv(fd, [mv[got:n]], base + got)
    np.copyto(dst[:n], tl.np[:n])


DST = None
for rep in range(2):
    for name in ("python16", "c16", "c32", "c64-chunk1M", "c32-readonly"):
        pks = rng.choice(valid, N, replace=False)                 # fresh pieces (not in the drive's cache)
        lens = (piece_K[pks] * UNIT).astype(np.int64)
        if DST is None or DST.size < lens.sum():
            DST = np.ones(int(lens.sum() * 1.2), np.uint8)         # touched once: the server's pools are mapped too
        dst = DST
        offs_d = np.concatenate([[0], np.cumsum(lens)[:-1]])
        if name.endswith("readonly"):                              # every piece into one small buffer: no copy cost
            offs_d = np.zeros_like(offs_d)
        t0 = time.time()
        if name == "python16":
            with ThreadPoolExecutor(16) as ex:
                list(ex.map(lambda a: pyread(a[0], dst[a[1]:a[1] + a[2]]), zip(pks, offs_d, lens)))
        else:
            th = int(name[1:3])
            chunk = 1 << 20 if "chunk1M" in name else 4 * UNIT
            offs = piece_off[pks].astype(np.int64)
            dsts = (dst.ctypes.data + offs_d).astype(np.int64)
            e = lib.read_pieces(fd, offs.ctypes.data, lens.ctypes.data, dsts.ctypes.data, N, th, chunk)
            assert e == 0, e
        dt = time.time() - t0
        print(f"{name:12s}: {N} pieces, {lens.sum() / 2**30:.1f} GiB in {dt:.2f} s = {lens.sum() / dt / 2**30:.2f} GiB/s",
              flush=True)
