#!/usr/bin/env bash
# Build dsv_moe.so (mixed-K grouped EXL3 GEMV) inside the rootfs: scripts/in_rootfs.sh bash /w/dsv41/xpu/build.sh
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
cd "$(dirname "$0")"
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
# AOT for the Arc Pro B70 (Battlemage G31): no JIT through IGC at first use in every process (that took minutes);
# DSV_SYCL_TARGET=spir64 builds the portable JIT form instead
TARGET=${DSV_SYCL_TARGET:-bmg}
if [ "$TARGET" = bmg ]; then TFLAGS='-fsycl-targets=spir64_gen -Xs "-device bmg-g31"'; else TFLAGS="-fsycl-targets=spir64"; fi
MOE_SPLIT=${DSV_MOE_SPLIT:--fsycl-device-code-split=per_source}
COMMON="-fsycl $TFLAGS -O3 -ffast-math -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 -I csrc -I /w/glm/xpu/csrc -I$T/include -I$T/include/torch/csrc/api/include"
LIBS="-L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu"
mkdir -p lib
what=${1:-all}
if [ "$what" = all ] || [ "$what" = moe ]; then
  # ESIMD: ocloc rejects its AOT link ("more than one module with an entry point"): JIT form, persistent cache
  eval icpx ${COMMON/$TFLAGS/-fsycl-targets=spir64} -x c++ csrc/dsv_moe.sycl -x none -o lib/dsv_moe.so.tmp $LIBS
  mv -f lib/dsv_moe.so.tmp lib/dsv_moe.so
fi
if [ "$what" = all ] || [ "$what" = fp8 ]; then
  eval icpx $COMMON -x c++ csrc/dsv_fp8.sycl -x none -o lib/dsv_fp8.so.tmp $LIBS
  mv -f lib/dsv_fp8.so.tmp lib/dsv_fp8.so
fi
if [ "$what" = all ] || [ "$what" = rope ]; then
  eval icpx $COMMON -x c++ csrc/dsv_rope.sycl -x none -o lib/dsv_rope.so.tmp $LIBS
  mv -f lib/dsv_rope.so.tmp lib/dsv_rope.so
fi
if [ "$what" = all ] || [ "$what" = glue ]; then
  eval icpx $COMMON -x c++ csrc/dsv_moe_glue.sycl -x none -o lib/dsv_glue.so.tmp $LIBS
  mv -f lib/dsv_glue.so.tmp lib/dsv_glue.so
fi
if [ "$what" = all ] || [ "$what" = hc ]; then
  eval icpx $COMMON -x c++ csrc/dsv_hc.sycl -x none -o lib/dsv_hc.so.tmp $LIBS
  mv -f lib/dsv_hc.so.tmp lib/dsv_hc.so
fi
if [ "$what" = all ] || [ "$what" = attn ]; then
  eval icpx $COMMON -x c++ csrc/dsv_attn.sycl -x none -o lib/dsv_attn.so.tmp $LIBS
  mv -f lib/dsv_attn.so.tmp lib/dsv_attn.so
fi
if [ "$what" = all ] || [ "$what" = spin ]; then
  icx -O2 -fPIC -shared csrc/dsv_spin.c -o lib/dsv_spin.so
fi
if [ "$what" = all ] || [ "$what" = io ]; then
  icx -O2 -fPIC -shared -pthread csrc/dsv_io.c -o lib/dsv_io.so
fi
ls -la lib/
