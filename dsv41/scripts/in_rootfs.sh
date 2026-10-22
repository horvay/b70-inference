#!/usr/bin/env bash
# Run a command in the shared SGLang XPU rootfs (torch 2.13 XPU, Triton XPU, oneAPI 2026.0) through bubblewrap with a
# throw-away overlay, the repo root at /w. DeepSeek-V4.1-Flash variant of glm/scripts/in_rootfs.sh.
#   in_rootfs.sh [--gpu] <command...>
# Binds: repo root at /w, model at /models (DSV41_MODEL), expert store dir at /nvx (DSV41_NVX). The model folder's
# Engram shards are symlinks into other drives; DSV41_EXTRA_RO (colon list) binds those directories at their own paths.
set -euo pipefail
D=$(dirname "$(dirname "$(readlink -f "$0")")")
W=$(dirname "$D")
ROOT=${B70_ROOTFS:-$W/state/image/rootfs}
CONF=$W/common/image-config.json
MODEL=${DSV41_MODEL:-$HOME/models/dsv41-flash}
NVX=${DSV41_NVX:-$HOME/models/dsv41-store}
EXTRA_RO=${DSV41_EXTRA_RO:-/mnt/old/models:/mnt/sandisk/models}
DEV=(--dev /dev)
if [ "${1:-}" = "--gpu" ]; then shift; DEV=(--dev-bind /dev /dev); fi
mkdir -p "$D/run" "$D/cache" "$NVX"
IMGENV=()
while IFS= read -r kv; do IMGENV+=(--setenv "${kv%%=*}" "${kv#*=}"); done < <(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1]))["config"].get("Env", []): print(e)' "$CONF")
PP=$(python3 -c '
import json, sys
env = dict(e.split("=", 1) for e in json.load(open(sys.argv[1]))["config"].get("Env", []))
print(env.get("PYTHONPATH", ""))' "$CONF")
RO=()
IFS=: read -ra ex <<< "$EXTRA_RO"
for p in "${ex[@]}"; do [ -d "$p" ] && RO+=(--ro-bind "$p" "$p"); done
EXTRA=()
if [ -n "${BWRAP_EXTRA:-}" ]; then eval "EXTRA=($BWRAP_EXTRA)"; fi
exec bwrap \
    --overlay-src "$(readlink -f "$ROOT")" --tmp-overlay / \
    "${DEV[@]}" \
    --proc /proc \
    --ro-bind /sys /sys \
    --tmpfs /tmp \
    --tmpfs /root \
    --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --bind "$W" /w \
    --ro-bind-try "$MODEL" /models \
    "${RO[@]}" \
    --bind "$NVX" /nvx \
    --bind "$D/cache" /root/.cache \
    --setenv HOME /root \
    "${IMGENV[@]}" \
    --setenv PYTHONPATH "/w/dsv41/xpu:/w/glm/xpu:/w/dsv41/exllamav3:/w/glm/pylib${PP:+:$PP}" \
    --setenv TRITON_CACHE_DIR /root/.cache/triton \
    --setenv SYCL_CACHE_PERSISTENT 1 \
    --setenv SYCL_CACHE_DIR /root/.cache/sycl \
    "${EXTRA[@]}" \
    --unshare-pid --die-with-parent \
    --chdir /w/dsv41 \
    "$@"
