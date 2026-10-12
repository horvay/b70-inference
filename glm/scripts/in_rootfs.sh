#!/usr/bin/env bash
# Run a command inside the unpacked SGLang XPU root filesystem (torch 2.13 XPU, Triton XPU, oneAPI 2026.0) with this
# repo bound in, through bubblewrap with a throw-away overlay (the rootfs is never written). No Docker, no root.
#   in_rootfs.sh [--gpu] <command...>
# Without --gpu no GPU device is passed in (builds, CPU tests). With --gpu, /dev is bound.
# Binds: repo at /w, model weights read-only at /models (GLM53_MODELS), expert store dir at /nvx (GLM53_NVX).
# Python packages from scripts/setup.sh live in /w/pylib (on PYTHONPATH).
set -euo pipefail
W=$(dirname "$(dirname "$(readlink -f "$0")")")
ROOT=${GLM_B70_ROOTFS:-${B70_ROOTFS:-$(dirname "$W")/state/image/rootfs}}
CONF=${GLM_B70_IMAGE_CONFIG:-$(dirname "$W")/common/image-config.json}
MODELS=${GLM53_MODELS:-/mnt/sandisk/glm53/model}
NVX=${GLM53_NVX:-$HOME/models/glm53-store}
DEV=(--dev /dev)
if [ "${1:-}" = "--gpu" ]; then
    shift
    DEV=(--dev-bind /dev /dev)
fi
mkdir -p "$W/run" "$W/cache"
IMGENV=()
while IFS= read -r kv; do IMGENV+=(--setenv "${kv%%=*}" "${kv#*=}"); done < <(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1]))["config"].get("Env", []): print(e)' "$CONF")
PP=$(python3 -c '
import json, sys
env = dict(e.split("=", 1) for e in json.load(open(sys.argv[1]))["config"].get("Env", []))
print(env.get("PYTHONPATH", ""))' "$CONF")
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
    --setenv TZ "$(tail -n1 /etc/localtime 2>/dev/null)" \
    --bind "$W" /w \
    --ro-bind-try "$MODELS" /models \
    --bind-try "$NVX" /nvx \
    --bind "$W/cache" /root/.cache \
    --setenv HOME /root \
    "${IMGENV[@]}" \
    --setenv PYTHONPATH "/w/xpu:/w/exllamav3:/w/pylib${PP:+:$PP}" \
    --setenv TRITON_CACHE_DIR /root/.cache/triton \
    "${EXTRA[@]}" \
    --unshare-pid --die-with-parent \
    --chdir /w \
    "$@"
