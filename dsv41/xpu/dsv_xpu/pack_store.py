"""Pack DeepSeek-V4.1-Flash's routed experts into an NVMe expert store.

experts.bin: per (layer, expert) one record = gate | up | down trellis, each projection K_p units long, where a unit
is the trellis of a K=1 projection (5120 x 2304 weights at 1 bit = 1,474,560 bytes = 360 x 4 KiB, so every piece is
4 KiB aligned for O_DIRECT). Trellis bytes are copied unchanged ([k/16][n/16][16K] int16).
vectors.bin: fp16 [L, E, 22272] = suh_g[5120] suh_u[5120] svh_g[2304] svh_u[2304] suh_d[2304] svh_d[5120] per expert
(kept in VRAM for all experts, 684 MB).
store.json: per layer and expert [Kg, Ku, Kd, unit offset].

  python3 -m dsv_xpu.pack_store /models /nvx
"""
import json
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

H, I = 5120, 2304
UNIT = H * I // 8
VEC = 2 * H + 2 * I + I + H     # 22272
PROJ = ("w1", "w3", "w2")       # gate, up, down


def headers(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    hdr = {}
    for f in sorted(set(idx.values())):
        p = os.path.join(model_dir, f)
        if not os.path.exists(p):
            continue                    # the Engram shards (not needed here) may live elsewhere / not be there yet
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            v["file"] = p
            v["base"] = 8 + n
            hdr[k] = v
    return hdr


def main(model_dir, out_dir, n_layers=40, n_experts=384, prefix="layers", name="experts"):
    hdr = headers(model_dir)
    os.makedirs(out_dir, exist_ok=True)
    fds = {}

    def fd(p):
        if p not in fds:
            fds[p] = os.open(p, os.O_RDONLY)
        return fds[p]

    def read(key):
        v = hdr[key]
        a, b = v["data_offsets"]
        buf = os.pread(fd(v["file"]), b - a, v["base"] + a)
        assert len(buf) == b - a, key
        return buf

    meta = {"format": "dsv41-expert-store-v1", "H": H, "I": I, "unit": UNIT, "vec": VEC, "n_layers": n_layers,
            "n_experts": n_experts, "prefix": prefix, "layers": []}
    off = 0
    plan = []
    for L in range(n_layers):
        rows = []
        for e in range(n_experts):
            Ks = []
            for p in PROJ:
                sh = hdr[f"{prefix}.{L}.ffn.experts.{e}.{p}.trellis"]["shape"]
                Ks.append(sh[2] // 16)
            rows.append([*Ks, off])
            plan.append((L, e, Ks, off))
            off += sum(Ks)
        meta["layers"].append(rows)
    total = off * UNIT
    print(f"store: {len(plan)} experts, {off} units, {total / 2**30:.2f} GiB", flush=True)

    bin_path = os.path.join(out_dir, f"{name}.bin")
    vec_path = os.path.join(out_dir, f"{name}_vectors.bin")
    out = os.open(bin_path, os.O_WRONLY | os.O_CREAT, 0o644)
    os.ftruncate(out, total)
    vec = np.lib.format.open_memmap(vec_path + ".npy", mode="w+", dtype=np.float16, shape=(n_layers, n_experts, VEC))

    def one(item):
        L, e, Ks, o = item
        pieces = [read(f"{prefix}.{L}.ffn.experts.{e}.{p}.trellis") for p in PROJ]
        for buf, K in zip(pieces, Ks):
            assert len(buf) == K * UNIT, (L, e, K, len(buf))
        data = b"".join(pieces)
        n = os.pwrite(out, data, o * UNIT)
        assert n == len(data)
        g, u, d = (f"{prefix}.{L}.ffn.experts.{e}.{p}" for p in PROJ)
        parts = [read(f"{g}.suh"), read(f"{u}.suh"), read(f"{g}.svh"), read(f"{u}.svh"), read(f"{d}.suh"),
                 read(f"{d}.svh")]
        vec[L, e] = np.frombuffer(b"".join(parts), dtype=np.float16)
        return len(data)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(16) as ex:
        for i, n in enumerate(ex.map(one, plan)):
            done += n
            if i % 1536 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(plan)} experts, {done / 2**30:.1f} GiB, {done / max(el, 1e-3) / 2**30:.2f} GiB/s",
                      flush=True)
    os.fsync(out)
    os.close(out)
    vec.flush()
    del vec
    os.replace(vec_path + ".npy", vec_path + ".npy")
    json.dump(meta, open(os.path.join(out_dir, f"{name}.json"), "w"))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])


def append_mtp(model_dir, out_dir, n_mtp=3, n_mtp_experts=128, name="experts"):
    """Append the DSpark drafter's routed experts (mtp.{0..n_mtp-1}) to an existing store as layers L..L+n_mtp-1
    (E stays the main model's; experts past n_mtp_experts are empty rows: K = 0). The packed main layers are not
    touched: the records go after the end of experts.bin; vectors and the index are rewritten.

      python3 -c "from dsv_xpu.pack_store import append_mtp; append_mtp('/models', '/nvx')"
    """
    hdr = headers(model_dir)
    jpath = os.path.join(out_dir, f"{name}.json")
    meta = json.load(open(jpath))
    L0, E = meta["n_layers"], meta["n_experts"]
    assert L0 == 40 and "mtp_layers" not in meta, "store already has the drafter (or an unexpected layer count)"
    end = max(r[3] + r[0] + r[1] + r[2] for layer in meta["layers"] for r in layer)
    fds = {}

    def read(key):
        v = hdr[key]
        a, b = v["data_offsets"]
        if v["file"] not in fds:
            fds[v["file"]] = os.open(v["file"], os.O_RDONLY)
        buf = os.pread(fds[v["file"]], b - a, v["base"] + a)
        assert len(buf) == b - a, key
        return buf

    off = end
    plan = []
    for m in range(n_mtp):
        rows = []
        for e in range(E):
            if e >= n_mtp_experts:
                rows.append([0, 0, 0, 0])
                continue
            Ks = [hdr[f"mtp.{m}.ffn.experts.{e}.{p}.trellis"]["shape"][2] // 16 for p in PROJ]
            rows.append([*Ks, off])
            plan.append((m, e, Ks, off))
            off += sum(Ks)
        meta["layers"].append(rows)
    print(f"drafter: {len(plan)} experts, {off - end} units, {(off - end) * UNIT / 2**30:.2f} GiB after unit {end}",
          flush=True)
    bin_path = os.path.join(out_dir, f"{name}.bin")
    out = os.open(bin_path, os.O_WRONLY)
    os.ftruncate(out, off * UNIT)
    vec_path = os.path.join(out_dir, f"{name}_vectors.bin.npy")
    old = np.load(vec_path, mmap_mode="r")
    new = np.zeros((L0 + n_mtp, E, VEC), np.float16)
    new[:L0] = old
    for m, e, Ks, o in plan:
        pieces = [read(f"mtp.{m}.ffn.experts.{e}.{p}.trellis") for p in PROJ]
        for buf, K in zip(pieces, Ks):
            assert len(buf) == K * UNIT, (m, e, K, len(buf))
        data = b"".join(pieces)
        assert os.pwrite(out, data, o * UNIT) == len(data)
        g, u, d = (f"mtp.{m}.ffn.experts.{e}.{p}" for p in PROJ)
        parts = [read(f"{g}.suh"), read(f"{u}.suh"), read(f"{g}.svh"), read(f"{u}.svh"), read(f"{d}.suh"),
                 read(f"{d}.svh")]
        new[L0 + m, e] = np.frombuffer(b"".join(parts), dtype=np.float16)
    os.fsync(out)
    os.close(out)
    del old
    tmp = vec_path + ".tmp.npy"
    np.save(tmp, new)
    os.replace(tmp, vec_path)
    meta["n_layers"] = L0 + n_mtp
    meta["mtp_layers"] = list(range(L0, L0 + n_mtp))
    json.dump(meta, open(jpath + ".tmp", "w"))
    os.replace(jpath + ".tmp", jpath)
    print("drafter appended", flush=True)
