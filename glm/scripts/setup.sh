#!/usr/bin/env bash
# One-time setup (no Docker, no root, no GPU needed): the shared SGLang XPU image, glm/pylib and the SYCL kernels.
# The repo's `b70 setup glm` does it; this wrapper stays for the dev scripts' sake.
exec "$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")/b70" setup glm "$@"
