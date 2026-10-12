"""Turn a Strata expert profile (STRP: every (layer, expert) pair of Qwen3.8-Flash-Next ranked by routing frequency)
into trellis-serve's EXL3_MOE_HOT JSON: the top N pairs as {"layers.<L>.mlp.experts": [expert ids]}.

    python3 profile_to_hot.py <expert-profile.bin> <N> <out.json> [skip layers, e.g. 0]

Skipped layers (comma-separated) are left out and do not count toward N: a layer whose experts have their own all-static
store (layer 0 of the Lygodactylus checkpoint is 4-bit) needs no hot list.

With EXL3_MOE_STATIC=1 those experts become the static tier: pinned device slots, no host copy.
"""
import json
import struct
import sys

path, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
skip = {int(v) for v in sys.argv[4].split(",") if v != ""} if len(sys.argv) > 4 else set()
with open(path, "rb") as f:
    data = f.read()
if data[:4] != b"STRP":
    raise SystemExit(f"{path}: not a Strata expert profile")
_version, n_layers, n_expert, _slots, n_ranked = struct.unpack_from("<5I", data, 4)
pairs = struct.unpack_from(f"<{2 * n_ranked}H", data, 24)
hot: dict = {}
taken = 0
for i in range(n_ranked):
    if taken == n:
        break
    layer, expert = pairs[2 * i], pairs[2 * i + 1]
    if layer >= n_layers or expert >= n_expert:
        raise SystemExit(f"pair {i} out of range: ({layer}, {expert})")
    if layer in skip:
        continue
    hot.setdefault(f"layers.{layer}.mlp.experts", []).append(expert)
    taken += 1
n = taken
with open(out, "w") as f:
    json.dump(hot, f)
sizes = sorted(len(v) for v in hot.values())
print(f"{n} of {n_ranked} pairs over {len(hot)} layers (per layer {sizes[0]}..{sizes[-1]}) -> {out}")
