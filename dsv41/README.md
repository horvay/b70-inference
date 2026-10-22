# DeepSeek-V4.1-Flash (SAGE EXL3 1.62 bpw) on an Intel Arc Pro B70

[vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw)
(262B-parameter MoE, 40 layers x 384 experts, top-6, mHC hyper-connections, compressed sparse attention, Engram
n-gram tables) running with the 32 GB B70 as the only GPU, 62 GB of RAM and NVMe.

Status and measured numbers: see "Results" at the end (filled in as measured).

## What the model needs

| Part | Size | Where it lives |
|---|---|---|
| Routed experts (SAGE MixedK EXL3, K = 1..4 bits *per projection*) | 102 GiB | VRAM / pinned RAM / NVMe store tiers |
| Attention, shared experts (block-32 FP8 + E8M0 scales) | 6.5 GB | VRAM, kept in FP8 |
| Output head (EXL3 K=8) | 0.66 GB | VRAM, dequantized once to fp16 (1.2 GB) |
| Expert vectors (suh/svh of every expert) | 0.68 GB | VRAM |
| Engram tables (2 x 384M rows x 256 fp8) | 189 GiB | NVMe/SATA, a few rows per token |

## The machine behind these numbers

| Part | What | Used for |
|---|---|---|
| GPU | Intel Arc Pro B70, 32 GB (Battlemage G31), PCIe 4.0 x16 (the card is 5.0, the AM4 platform tops out at 4.0) | everything on the GPU; the machine's RTX 3080 Ti drives the displays and is not used |
| CPU | AMD Ryzen 9 5900XT, 16 cores | host side: expert bookkeeping, NVMe reads |
| RAM | 64 GB DDR4 (62 GiB usable) | pinned expert tier, the process |
| Disk 1 | Samsung 990 PRO 2 TB NVMe (LUKS + btrfs; the store folder nodatacow) | model folder, expert store |
| Disk 2 | WD Black SN750 1 TB NVMe | Engram shard 1 |
| Disk 3 | SanDisk Ultra II 480 GB SATA SSD | Engram shard 2 |

What the chat server (`scripts/serve.sh`, the default configuration) takes:

| Resource | Amount |
|---|---|
| VRAM | ~30 of 31.9 GiB: 9.0 GiB of non-expert weights with the DSpark drafter, 17.4 GiB expert cache (~17% of the 102 GiB of experts; another ~23% in the RAM tier), 0.7 GiB expert vectors, the rest prompt and verify scratch |
| RAM | 24 GB pinned expert tier + 2 x 0.5 GB staging areas, plus the Python/torch process; the server starts only with 34 GB free |
| Disk | 116 GiB model folder + 105 GiB expert store (the experts repacked for O_DIRECT reads) on the 990 PRO, 189 GiB of Engram tables on the other two drives: ~410 GiB in all |

## Quality against the model card

The [model card](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw) measures the 1.62 bpw pack
against DeepSeek's official FP8/FP4 release: **94.52% top-1 agreement, mean KL 0.0982** (held-out sequences, their
harness). This port runs the same weights unchanged; what it adds on top, each measured teacher-forced against this
port's exact computation:

| Change | Top-1 agreement | Mean KL | Notes |
|---|---|---|---|
| Fused XPU kernels vs exllamav3's reference math | 97.9% | 0.0055 | fp16 rounding; measured with Engram off |
| Residency boost 0.3 (routing prefers cached experts on near-ties) | 94.8% | 0.011 | on by default in the server |
| Prompt residency boost 0.3 | - | - | continuation NLL within +-0.01 |
| DSpark, exact acceptance | - | <= 0.007 (verify pass vs plain decode) | every emitted token is the model's own pick |
| DSpark block boost 0.3 | - | - | NLL 0.193, same as plain decode, on 4 x 320 tokens |

So the port's own deviations are about a tenth of the pack's distance from the official release (KL ~0.01 against
0.098). The top-1 figures look alike but measure different things: ours count near-tie flips against the pack's exact
output, the card's count the quantization's disagreements with the original. Not measured: agreement with the official
release directly (it does not fit on this machine) or on the card's held-out set.

## Layout

- `exllamav3/`: vcruz305's exllamav3 fork (DeepSeek-V4.1 support), a submodule at its `master`; `exllamav3.patch` holds
  the B70 changes `b70 setup dsv41` applies (extension loader, device checks, Triton tile sizes for Xe2, a row-blocked
  V4.1 indexer, lenient draft acceptance). Regenerate it with `git -C exllamav3 diff > exllamav3.patch`.
- `xpu/serve.py`: the OpenAI-compatible server (`b70 serve` or `scripts/serve.sh`), on GLM's HTTP layer; prompts go
  through `xpu/dsv_encoding.py` (DeepSeek's reference encoder, verbatim) and DSML tool calls come back as `tool_calls`.
  `xpu/chat.html` is the chat page it serves at `/`.
- `xpu/exllamav3_ext/`: the XPU stand-in for the CUDA extension (GLM port's ops + `dsv_xpu.ops`).
- `xpu/dsv_xpu/`:
  - `moe.py`: the expert tiers (per-projection pieces in per-K slot pools; VRAM, pinned RAM read zero-copy, NVMe
    store), no-sync router, decode MoE (fused kernels) and prompt MoE (dequantize + GEMM, layer-ahead NVMe staging).
  - `fp8.py`: FP8 linears (SYCL GEMV for decode rows, dequantize + oneDNN GEMM for prompts, fused grouped wo_a).
  - `block.py`: fused hyper-connection site (mix + collapse + RMSNorm) for decode rows.
  - `graphs.py`: XPU graphs per transformer block around an eager attention core.
  - `exl3_tri.py`: Triton EXL3 dequant for any K (bit-exact with the reference).
  - `pack_store.py`: packs the experts into the NVMe store.
  - `runner.py`: loads everything.
- `xpu/csrc/`: SYCL/ESIMD kernels: `dsv_moe.sycl` (mixed-K grouped EXL3 GEMV over pointer tables, one launch per K),
  `dsv_moe_glue.sycl` (router, MoE prep/mid/post), `dsv_fp8.sycl`, `dsv_rope.sycl`, `dsv_hc.sycl`.
- `tests/`: kernel tests, profilers, benchmarks.

## How a decode step runs

1. **Routing (no-sync).** Each MoE layer's router picks the top 6 experts among those whose three projection pieces are
   readable right now (VRAM or pinned RAM); a non-resident pick falls to the next-best resident expert. The router
   also records the unrestricted top 6 ("wanted"). Between steps the host reads both back, refreshes the LRU and hit
   counts, starts background NVMe reads of wanted-but-missing pieces into RAM, and promotes hot RAM pieces to VRAM.
   Prompt tokens always use exact routing (the host waits for missing experts).
2. **Experts.** One item per (pick, projection); `moe_prep` builds the Hadamard-domain inputs and the pointer/K tables,
   `moe_gemv` (ESIMD, one launch per K = 1..4, each skipping items of other K) runs the EXL3 GEMVs straight from VRAM
   or zero-copy from pinned RAM, `moe_mid` does SiLU * up and the down input transform, a second `moe_gemv` the down
   projections, `moe_post` the output transform and the weighted sum.
3. **Dense parts.** FP8 GEMVs (SYCL, ~550-590 GB/s), the grouped wo_a projection in one call, fused hyper-connection
   sites, SYCL rope, the V4.1 compressed sparse attention (exllamav3's Triton DSA kernel, tiles sized for Xe2).

## Pitfalls hit

- SAGE MixedK: K differs per expert *projection* (gate/up/down can each be 1..4 bits), so the GLM port's per-layer-K
  MoE kernel does not apply; pieces are cached per projection in per-K pools.
- A RAM slot freed by a RAM -> VRAM promotion must not be refilled before the asynchronous copy has run (it corrupted
  promoted experts); freed slots are held until the stream has caught up.
- exllamav3's Triton DSA config picker reads CUDA device properties; XPU gets a fixed 64 KB SLM budget.
- The first prompt of a process spends ~60 s in Intel's JIT (SYCL spir64 and Triton kernels).

## Quality (teacher-forced, 96 steps of documentation text, vs exact routing)

| Configuration | Top-1 agreement | KL | NLL |
|---|---|---|---|
| Exact routing, Engram on (reference) | - | - | 1.883 (ppl 6.6) |
| No-sync routing, Engram on | 82.3% | 0.137 | 2.005 |
| Exact routing, Engram **off** | 43.8% | 1.78 | 3.889 |
| New fused kernels vs torch reference paths (Engram off, exact) | 97.9% | 0.0055 | 3.880 vs 3.889 |

Engram is not optional (without it the model is a much worse model). No-sync routing (only resident experts) costs
~0.12 nats at the ~35% expert coverage this machine allows, so the default is **exact routing with layer-ahead
prefetch**: at each MoE layer the next layer's router runs on the same hidden state and its predicted experts are read
from NVMe in the background; the current layer waits only for the experts that were not predicted (or not finished).
`DSV_NOSYNC=1` switches to no-sync routing (~20-40% faster decode, lower quality).

## Results (2026-10-21, final configuration)

Exact routing + layer-ahead prefetch, Engram on, greedy. Arc Pro B70 32 GB, 62 GB RAM (16 GB pinned expert tier),
990 PRO NVMe expert store; ~2,900 experts in VRAM and ~2,400 in RAM (35% of 15,360). Prompts are documentation text.
Short-context runs keep 3 GB of VRAM spare; 40K runs keep 6 GB (prompt activations), so their expert cache is ~3 GB
smaller.

| Context | Users | Time to first token | Prefill | Decode per user | Decode total |
|---|---|---|---|---|---|
| 32-token prompt | 1 | 4.3 s | - | 7.34 tok/s | 7.34 tok/s |
| 32-token prompt | 2 | 8.6 s | - | 3.20 tok/s | 6.40 tok/s |
| 32-token prompt | 3 | 12.6 s | - | 2.59 tok/s | 7.77 tok/s |
| 40K-token prompt | 1 | 354 s | 113 tok/s | 6.31 tok/s | 6.31 tok/s |
| 40K-token prompt | 2 | 688 s (prompts run one after the other) | 116 tok/s | 2.67 tok/s | 5.34 tok/s |
| 40K-token prompt | 3 | 1,063 s | 113 tok/s | 1.95 tok/s | 5.85 tok/s |

Decode varies ~15% run to run with the expert cache state. The first prompt of a process adds ~60 s of kernel JIT.
No-sync routing (`DSV_NOSYNC=1`, 82% top-1 vs exact) measured 7.3 / 8.2 / 7.6 tok/s total at short context.

What moved the numbers overnight: fused MoE/router/hyper-connection kernels (3.7 -> 5.3 tok/s eager), budgeted and
frequency-gated RAM -> VRAM promotion (copy churn was 30 ms/token), fixed-width DSA selection (no re-JIT every 32
tokens of context), grouped prompt MoE + 32-head single-stage DSA tiles for prompts (40K prefill 632 s -> 354 s).

## 2026-10-21 day: decode host overhead

- `attn.py` + `dsv_attn.sycl`: lean decode attention for every layer (2 fused FP8 GEMVs, SYCL norm/rope, one SYCL
  split + combine over every job's window and selected pool entries, ring append folded in). Attention GPU time 10-17
  -> 1.8 ms/token, host time ~77 -> ~27 ms/token. Matches exllamav3's path to fp16 rounding on every layer. Keeps the
  hand-over state (kv-source pool, index lists, candidate masks) per job: exllamav3 keys it per layer only, so with
  several jobs a consumer layer read the last job's block-table row and indices (multi-user decode was partly wrong).
- Fused hyper-connection update + next site (`hc41_apply_site`), shared expert gate|up GEMV + SYCL SiLU, router of this
  and the next layer in one SYCL op, zero-copy pointer / residency tables (no per-layer uploads), RAM -> VRAM promotion
  off by default (it cost more PCIe traffic than it saved).
- Residency boost (`DSV_BOOST`): exact routing where an expert already in VRAM or RAM competes with score * (1 + boost)
  (weights stay the plain scores). Measured (dbench, 1 user, short prompts): exact 6.3 tok/s; 0.1: 7.4; 0.2: 8.4 (95.8%
  top-1 / KL 0.007 vs exact); 0.3: 9.4 (94.8% / KL 0.011); no-sync 8.1 (82% / KL 0.137).
- `pager.py` (default on, `DSV_ASYNC=0` off): exact-routing decode without per-layer host waits. After each router a
  sync-point kernel copies the picks to pinned memory and publishes ready[l]; an ESIMD kernel spins on done[l] with
  uncached LSC atomic loads (SYCL atomic / volatile / plain loads read a stale cached copy of host memory in ~half the
  handoffs: tests/t_spin.py); the pager thread makes the picks resident, publishes done, then does the bookkeeping.
  Pitfalls: anything on the main thread that syncs with the GPU while holding the GIL deadlocks against the pager
  (indexing a pool with a device scalar did: page numbers come from the host block table now); first-use JIT inside a
  step holds the GIL for seconds (the first step of each shape runs synchronously). Results (dbench, 1 user): exact
  6.1 -> 7.2 tok/s; boost 0.2 8.5 -> 8.4; boost 0.3 9.3 -> 9.0 (no gain with the boost: the remaining stalls are NVMe
  reads of missing experts, ~25-40 ms/token, and ~80 ms/token of GPU work, much of it RAM-tier reads over PCIe).

## 2026-10-21 evening: caching and prediction

- XPU block graphs (`block.py`, DSV_GRAPH) + C pager fast path: correct, small gain (decode is GPU-bound).
- PCIe 4.0 x16 (platform max; card is 5.0): RAM-resident experts are read at ~25 GB/s, the biggest per-token cost.
- Promotions (on by default): experts hot in the current conversation are copied RAM -> VRAM on a side stream
  (frequency-gated against the coldest VRAM piece, pointer switched when the copy has run; RAM copy kept).
- Staging the next layer's predicted RAM experts into VRAM (DSV_STAGE=1): off, a net loss (doubles PCIe traffic; the
  copies rarely land in time).
- Prediction depth (DSV_AHEAD_DEPTH, default 2): NVMe prefetch of the layers 1 and 2 ahead (69-71% / 63-65% right).
  Boost 0.3: 10.8 -> 12.2 tok/s on rotating topics.
- Routing profile saved at exit and every 1000 steps (DSV_SAVE_PROFILE, tests off): the next start warms from it.
  First 100 tokens of a run: 6.9 -> 13.4 tok/s.
- Speed grows within a conversation as RAM / VRAM converge on its experts: prose (tests/t_long.py, boost 0.3,
  24 GB RAM tier) ~11 tok/s at the start, ~14-16 tok/s after 500+ tokens.
- DSpark: consecutive tokens share only ~1/3 of their experts (a 3-token block needs 2.27x one token's), so verifying
  drafts reads nearly as many expert bytes per output token as plain decode: estimated <= ~20% gain; not wired up.
- `~/models/.trackerignore`: the desktop indexer had read 138 GB from the NVMe during benchmarks.

## 2026-10-21 night / 10-22: DSpark, prompts, reads

- DSpark (the model's 3 drafter blocks) runs: drafter experts appended to the store as layers 40-42, multi-row lean
  attention and grouped 2-4 row expert GEMVs for verify passes, block boost (`DSV_BLOCK_BOOST`: a verify block's rows
  prefer experts another row already picked) and lenient acceptance (`EXL3_LENIENT_TAU`). A forced-draft check
  (tests/t_verify.py) matches plain decode: same argmax at 40/40 positions, KL <= 0.007.
  - Short prompts, 160 tokens: plain 10.7 tok/s (NLL 0.203); block boost 0.3 + tau 0.3: 15.9 (0.212); block boost
    0.6 / 1.0 cost 0.06-0.08 NLL for little speed. Block boost is the quality cost; tau barely moves NLL.
  - One long topic (700 tokens, residency boost 0.3): plain 13.7 tok/s average; exact DSpark 14.6; DSpark with block
    boost 0.3 + tau 0.5 19.2 (22-23 after 500 tokens), but its text turned repetitive (212 of 342 distinct 8-grams).
  - Without the residency boost DSpark loses (7.3 vs 9.1 tok/s): the drafter takes ~4 GB of VRAM from main-model
    experts and a verify block reads more distinct experts per token.
- Prompt processing (512-token prompt, 2nd prompt of a process): 27-30 s -> ~20 s.
  - Residency boost for prompt rows (`DSV_PREFILL_BOOST`, default 0.3): NVMe pieces 31K -> 22K, continuation NLL
    within +-0.01. Its top-k runs on the CPU: torch.topk on the XPU segfaulted (host side) on the first chunk.
  - Native Triton binaries during prompts only (runner sets `knobs.intel.gen_native_code`): a process's first prompt
    84 -> 39 s. Not for decode: a variant first compiled inside a pager step took the GPU past the 10 s timeout.
  - One host-to-device upload of the grouped prompt MoE's tables per layer instead of ~430 small blocking ones.
  - The C reader (`lib/dsv_io.so`, `DSV_C_IO`, batches of >= 16 pieces): 664-token prompt 31-35 -> 24 s, decode even.
  - Short-prompt staging (`DSV_PREFILL_TAIL`: only the last rows' experts into the RAM tier): 26-29 -> 24 s, off.
- DSpark is on by default (`DSV_DRAFT=1`) with block boost 0.3 (`DSV_BLOCK_BOOST`) and exact acceptance
  (`EXL3_LENIENT_TAU=1.0`). Measured with interleaved runs, plain before and after (the cache warms ~10% over a run,
  which had made the earlier sweeps favour whatever ran late): 4 prompts x 320 tokens, +24% tok/s, NLL 0.193 vs
  0.193/0.180, distinct 8-token sequences 77% vs 72/81%. tau 0.5-0.7: +36-37%, repetition unclear; tau 0.3: +24-66%
  but loops (46-54% distinct). One long topic, fresh process per run from the same warm profile: plain 14.8 tok/s,
  exact DSpark (no block boost) 14.3, block boost 0.3 + tau 0.3 19.4.
- Measured, no gain: split-K 64/48 or 8/48 vs 8/4 (equal within noise); promotions with a longer memory
  (`DSV_RECENT_DECAY=0.95`): 12.5-13.0 vs 13.7 tok/s.
- Where decode time goes (graphs off, profiler): expert GEMVs 42 ms/token, GPU waiting at sync points (NVMe misses)
  35, FP8 GEMVs 13, the rest ~15. Expert GEMVs on host-RAM pieces run at the PCIe limit (21-26 GB/s at every K); on
  VRAM pieces at 181 (K=1) to 448 GB/s (K=4). RAM-resident experts (~200 pieces a token) are the cost: ~22 ms/token.
- NVMe (expert store on LUKS + btrfs, nodatacow): 4.3 GiB/s through the Python pool, 5.3 through the C reader, for
  random 1.5-6 MB pieces into mapped memory.
- Pager: sync-point timeout 10 -> 60 s (`DSV_PAGER_TIMEOUT`); a host stalled by swap-in is not a GPU fault.

## Chat server

`b70 serve <model> --port N --prepare` (what Neverending Quest runs) or, for development, `scripts/serve.sh
[start|stop|status]` queues `xpu/serve.py` on the B70 (b70q owner `dsv41`, `--idle 15m`): a chat page at
http://127.0.0.1:8054/ and an OpenAI-compatible API at `/v1` (chat and text completions, streaming, thinking via
`chat_template_kwargs.enable_thinking` + `reasoning_effort` low / high / max or 1-100; tools and tool calls in
DeepSeek's DSML format, rendered and parsed by the reference encoder, checked by `tests/t_encoding.py` against
DeepSeek's golden prompts).
It runs the tested defaults (DSpark, residency boost 0.3) with a 32K context and a 24 GB RAM tier; ready ~45 s after its
turn. First measurements (fresh process): 26-token prompt, first token 3-4 s; ~380-token stories at 8.1 tok/s
(temperature 1.0) and 10.8 tok/s (greedy). Sampling lowers DSpark's acceptance, since a draft must equal the sampled
token.

## Open items

- Decode is bound by expert reads: ~240 expert projections per token, ~22% of them from RAM over PCIe (~25 GB/s).
  More VRAM/RAM for experts, or a K=1-specific GEMV, are the levers.
- XPU graphs (`DSV_GRAPHS=1`, no-sync only): graph A works, graph B (MoE) replay segfaults on the host after
  capture; every kernel captures and replays fine alone. Not diagnosed.
- DSpark: a repetition-aware lenient acceptance (lenient only when the drafted token does not extend a repeat) could
  keep tau 0.5's extra ~12%.
- Prompt processing is now about half NVMe staging, half compute; layer-major prefill (stream each layer's experts
  once per prompt instead of once per 8K chunk) would cut the NVMe part ~5x for 40K prompts.
