# glm-b70: GLM-5.3-Flash (glm-flash-lite) with the Arc Pro B70 as the primary GPU

glm-flash-lite (sybil-solutions) serves GLM-5.3-Flash EXL3 3.05 bpw (125.3 GB) from one 24 GB RTX 3090 + RAM + an NVMe
expert store, on stock exllamav3 1.5.1 (CUDA). This workspace ports it so the 32 GB Intel Arc Pro B70 runs the model.
Since 2026-10-10 the model is MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3 (`d6eb7f51`, 105 GB, abliterated):
routed experts are K=2 in 22 layers and K=3 in 20 (+ MTP), so store records, cache slot pools and kernel K are per layer.
No Docker: everything runs in trellis-b70's unpacked SGLang XPU rootfs (torch 2.13 XPU, Triton XPU, oneAPI 2026.0)
through bwrap, with a tmp overlay so the rootfs itself is never written.

## Layout

| path | what |
|---|---|
| `exllamav3/` | turboderp exllamav3 at v1.5.1 (`958ec93`), branch `xpu`: Python model code, CUDA calls mapped to XPU |
| `xpu/exllamav3_ext/` | the XPU shim that stands in for exllamav3's compiled CUDA extension (torch / Triton / ESIMD) |
| `xpu/exl3xpu/`, `xpu/csrc/` | exl3xpu ESIMD kernels vendored from trellis-serve (`xpu/TRELLIS_COMMIT`) + the GLM MoE kernel (`exl3_moe.n128.sycl` from glm-flash-lite b70tier) |
| `xpu/build.sh` | builds `exl3xpu/_C.so` (EXL3 linear) and `exl3xpu/_moe_glm.so` (grouped MoE, swiglu clamp 10) |
| `xpu/pack_store.py` | packs the routed experts into the NVMe store in exl3xpu blob order |
| `glm-flash-lite/` | upstream repo, branch `b70-primary` |
| `moetier/` | upstream campaign notes (B70 microbench, latency analyses), reference only |
| `scripts/in_rootfs.sh` | `in_rootfs.sh [--gpu] cmd...`: workspace at `/w`, weights ro at `/models`, store at `/nvx` |

## Data

- Weights: `/mnt/sandisk/glm53/model` (MikeRoz 2.51 bpw uncensored, 105 GB, SATA SSD; read at load and by the packer
  only). The 3.05 bpw turboderp weights and their store were deleted on 2026-10-10. The MTP layer (model layer 45) is in
  the main shards.
- Expert store: `~/models/glm53-store/glm53_flash_exl3_experts_blob.{bin,json}` (format v2: per-layer K, record size,
  offset; 990 PRO, btrfs dir with `chattr +C` so O_DIRECT is not defeated by zstd compression), ~95 GB.

## Status (2026-10-10)

Correct: reference panel top-1 0.987 / KL 0.0037 vs stock exllamav3 on CUDA. Decode ~5.2 tok/s exact (256-token
greedy, 27-30 GB RAM tier). Prefill being rebuilt (streaming stager, DPAS MoE re-enabled).

| piece | how it runs on the B70 |
|---|---|
| EXL3 linears | exl3xpu ESIMD (`exl3_gemm_small` / `linear`) |
| routed experts | `glm_xpu/tier.py`: VRAM slots (frequency-gated admission) + pinned RAM tier read zero-copy by the MoE kernel + NVMe store; layer-ahead prefetch; exclusive tiers with a VRAM victim ring; prefill streams non-resident experts through a pinned ring |
| MoE kernel | `csrc/exl3_moe.n128.sycl`: grouped GEMV for decode, DPAS for M >= 256 (gate/up work-group split to <= 32 threads: 64 hung the B70 in large-GRF mode) |
| KDA | decode step `glm_xpu/k_kda.py` (Triton); prefill vendored fla chunk_kda (Triton) |
| MLA | sparse (DSA) attention `glm_xpu/sparse_mla.py` (gather + oneDNN bmm, 5-22x the Triton kernel); dense prefill torch SDPA (40-90x) |
| mHC, norms | `glm_xpu/k_hc.py` (Triton) |
| decode step | `glm_xpu/block_graphs.py`: XPU graphs per block around the one host sync per MoE layer |

Dev loop: `scripts/dev_restart.sh` starts `xpu/devserver.py` (model loaded once, kernel modules hot-reload);
`python3 xpu/dev.py xpu/snip/<snippet>.py` runs a snippet on the live model. Kernel unit tests: `xpu/tests/kernels.py`,
`xpu/tests/units.py`. Router substitution ("dynamic REAP", `GLM_XPU_SUBST_KEEP`) exists but is off: ~7x KL for <10%.

Measured 2026-10-10 (decode timed without the prompt): no-sync 12.3-13.2 tok/s, exact 8.7 (no-sync per token: routed
experts 30.7 ms, of which RAM-tier experts over PCIe ~20; MLA attention eager 15.7; phase A 16.3; Bn 10.6; bookkeeping
4.1). MTP spec decode runs (~5 tok/s; the multi-token verify forward is eager and exact). Short prompts go through
ensure() (admit_max 512). Prompts are chunked at 2048 tokens: prompt activations grow ~0.7 GiB per 1K tokens and 8192
ran the B70 out of memory (Level Zero segfault) - cause not found yet.

Measured 2026-10-11 (all run on the B70):
- DPAS prefill for K=2/K=4 (`build4_any`): `units.py --moe` rel err 7.2e-4 at M=300 for K=2, 3, 4 (same as K=3 before).
- MLA decode in the phase-A graph (`mla_graph.py`, phase "Am"): vs `MLAttention.forward` on the same state, rel err
  1.2e-4 at 2.6K context (sparse), 1.4e-3 short (eager uses dense attention there). MLA per token 15.7 -> 8.9 ms.
  Short-context decode 15.4-15.9 tok/s with a warm expert cache (12-13.4 right after a restart: ~20 NVMe reads/token).
- Prompt memory: KDA prefill held dead fp32 tensors (projection, gate temporaries, fla input_guard copies, the
  cumsum copy): per-layer peak 388 -> 208 MiB per 1K tokens, values bit-identical (`snip/kda_eq.py`). Whole prompt:
  2K tokens 1.00 -> 0.59 GiB extra; 4K tokens 1.83 GiB. The crash after a long prompt was the first decode graph
  replay getting OUT_OF_DEVICE_MEMORY (torch kept the prompt's blocks cached): `_end_large_forward` now runs at the
  first block of the next step and empties the cache. Prompt chunk 2048 -> 4096.
- Prompt speed: 2K tokens 16.5 s (122 tok/s), 3.3K 17.2 s, 4K 17.9 s (was 72 s for 3.6K). The first prompt after a
  server start is slow (Triton compile/autotune: 80 s at 4K). Most remaining time is the MoE layers waiting on
  RAM/NVMe expert reads.
- Staging refill after a prompt: "refilled 288 staging slots ... in 0.4-0.7 s".

Prefill, later 2026-10-11:
- One pass per prompt: the recurrent checkpoint at the last full page is stashed inside the forward (GatedDeltaNet
  splits at `recurrent_ckpt_offset`, job.py `_put_inpass_checkpoint`) instead of exllamav3's separate last-page forward,
  which fetched every routed expert a second time. Checkpoint differs from the old one by the same rounding as changing
  the chunk size (`snip/inpass_ref.py`). 2K prompt 16-20 s -> 9.0 s.
- Per-layer activation bounds (KDA split every 2048, MLA sequential 1024-token sub-chunks, dense MLP row slices, MLA
  dense prefill in head groups / query blocks): ~85 MiB per 1K prompt tokens. Chunk 8192: 4K 10.5 s, 6K 13.2 s, 8K
  14.4 s (556 tok/s), peak 1.46 GiB extra; 12K runs out of device memory (and leaves the context broken: kill the server).
- Chat follow-up (2K cached + ~40 new tokens): prompts under 2048 picks read only their picked experts in parallel
  (tier `pf_min_picks`) instead of streaming every non-resident expert ahead; prompts over 8 tokens take the staging
  path (`admit_max` 64). First token 18.5 s -> 1.7 s (38 new tokens), 3.1 s (59).
- First prompt after a server start is still ~60 s (one-time kernel JIT: SYCL kernels are spir64, not AOT;
  SYCL_CACHE_PERSISTENT crashed the server in libc, NEO_CACHE_PERSISTENT had no effect).

## Next

1. Prefill: streaming stager (written, untested), then measure 4K / 16K prompts.
2. MTP speculative decoding (checkpoint ships `mtp.safetensors`; GLM-5.3 MTP acceptance ~70-75% on real text).
3. A real OpenAI-compatible server (glm-flash-lite `glm53/serve.py` on this runner) for chat clients.
4. Later: sglang Intel `chunk_kda`, the 3080 Ti as an extra expert tier.

## Hardware facts

- B70 sits behind its own PCIe switch on a PCIe 4.0 x16 root port (~24-27 GB/s H2D). RTX 3080 Ti: PCIe 4.0 x4.
- Ryzen 9 5900XT, 32 threads, 62 GiB RAM shared with the desktop. 990 PRO (PCIe 4.0 x4) under LUKS + btrfs: ~5.5 GB/s
  O_DIRECT with 16 readers x 8 pieces per record, ~2 ms per 9.47 MB record.
- Level Zero pitfalls hit here: O_DIRECT into USM host memory -> EFAULT; one pinned allocation > ~4 GB silently returns
  a null pointer; torch's pinned allocator rounds to powers of two (use 226-record chunks); interpreter teardown with
  ~30 GB pinned can spin for 15+ minutes (devserver exits with os._exit).
