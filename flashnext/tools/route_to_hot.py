#!/usr/bin/env python3
"""Build an EXL3_MOE_HOT list from routing counts (EXL3_MOE_PROFILE), or score a hot list against counts.

    route_to_hot.py build <route.json> <N> <out.json> [--minus <earlier route.json>] [--strata <expert-profile.bin>]
    route_to_hot.py cover <hot.json> <route.json> [--minus <earlier route.json>]
    route_to_hot.py ram <route.json> <hot.json> <N> <out.json> [--minus <earlier route.json>]

build: the N (layer, expert) pairs with the most decode routings across all layers become the static tier (the
same global ranking profile_to_hot.py makes from a Strata profile). Layer 0 is left out (its own all-static 4-bit
store). Ties, including experts never seen in the sample, are broken by the Strata ranking when given.
cover: the share of decode routings that land on the hot list's experts, i.e. served from a pinned GPU slot
rather than the LRU cache or host RAM. --minus subtracts an earlier cumulative dump (phase B = B dump - A dump).
ram: the RAM tier for EXL3_MOE_RAM: the N (layer, expert) pairs outside the hot list with the most decode routings
(ties broken by prompt routings) keep a host copy; every other non-hot expert goes to the NVMe cold store.
"""
import json
import re
import struct
import sys


def layer_of(key: str) -> int:
    m = re.search(r"layers\.(\d+)\.", key)
    if not m:
        raise SystemExit(f"no layer number in {key!r}")
    return int(m.group(1))


def load_counts(path: str, minus: str | None, kind: str = "decode") -> dict:
    d = json.load(open(path))["layers"]
    counts = {layer_of(k): v[kind] for k, v in d.items()}
    if minus:
        b = {layer_of(k): v[kind] for k, v in json.load(open(minus))["layers"].items()}
        counts = {L: [x - y for x, y in zip(c, b[L])] for L, c in counts.items()}
    return {L: c for L, c in counts.items() if L != 0}


def strata_rank(path: str) -> dict:
    data = open(path, "rb").read()
    n_ranked = struct.unpack_from("<5I", data, 4)[4]
    pairs = struct.unpack_from(f"<{2 * n_ranked}H", data, 24)
    return {(pairs[2 * i], pairs[2 * i + 1]): 1.0 - i / n_ranked for i in range(n_ranked)}


def opt(args: list, name: str):
    return args[args.index(name) + 1] if name in args else None


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "build":
        route, n, out = args[0], int(args[1]), args[2]
        counts = load_counts(route, opt(args, "--minus"))
        sp = opt(args, "--strata")
        tie = strata_rank(sp) if sp else {}
        scored = sorted(((c + 0.5 * tie.get((L, e), 0.0), L, e) for L, cs in counts.items() for e, c in enumerate(cs)),
                        reverse=True)[:n]
        hot: dict = {}
        for _, L, e in scored:
            hot.setdefault(f"layers.{L}.mlp.experts", []).append(e)
        json.dump(hot, open(out, "w"))
        sizes = sorted(len(v) for v in hot.values())
        print(f"{n} pairs over {len(hot)} layers (per layer {sizes[0]}..{sizes[-1]}) -> {out}")
    elif cmd == "ram":
        route, hot_p, n, out = args[0], args[1], int(args[2]), args[3]
        hot = {layer_of(k): set(v) for k, v in json.load(open(hot_p)).items()}
        dec = load_counts(route, opt(args, "--minus"))
        pre = load_counts(route, opt(args, "--minus"), "prefill")
        scored = sorted(((c + 1e-6 * pre[L][e], L, e) for L, cs in dec.items()
                         for e, c in enumerate(cs) if e not in hot.get(L, ())), reverse=True)
        ram: dict = {}
        for _, L, e in scored[:n]:
            ram.setdefault(f"layers.{L}.mlp.experts", []).append(e)
        for L in dec:                              # every layer gets an entry (an empty one: no RAM tier there)
            ram.setdefault(f"layers.{L}.mlp.experts", [])
        json.dump(ram, open(out, "w"))
        n_cold = len(scored) - min(n, len(scored))
        print(f"{min(n, len(scored))} RAM experts, {n_cold} cold (NVMe, {n_cold * 1.8624e-3:.1f} GB) -> {out}")
    elif cmd == "cover":
        hot = {layer_of(k): set(v) for k, v in json.load(open(args[0])).items()}
        counts = load_counts(args[1], opt(args, "--minus"))
        tot = sum(sum(c) for c in counts.values())
        hit = sum(c[e] for L, c in counts.items() for e in hot.get(L, ()))
        print(f"{args[0]}: {hit / tot:.1%} of {tot} decode routings on the static tier "
              f"({sum(len(v) for v in hot.values())} pinned experts)")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
