# Qwen3.8-Flash-Next (EXL3) on the Arc Pro B70

[Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw](https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw)
(3.05 bpw, 68 GB) served by 0xSero's [trellis-serve](https://github.com/0xSero/trellis-serve/tree/ft-xpu-vision/xpu)
(SGLang 0.5.20-xpu + the exl3xpu plugin and its SYCL kernels), with `trellis-serve.patch` on top of `ft-xpu-vision`
@ 72fd2d4. `b70 setup sglang` fetches that commit and applies the patch (shared with Gemma); `b70 serve <model dir> --port N` runs it.

| | |
|---|---|
| Decode, 1 user, short game-master turn | 41.7 tok/s |
| Decode, 1 user, at 18K context | 36.7 tok/s |
| Decode, 2 / 4 users | 60-68 / 89-100 tok/s total (22-25 each with 4) |
| Next turn over a cached 21K prompt, four games taking turns | 0.2-0.8 s to the first token |
| 18K-token prompt (cold) | 15.9 s, ~1,130 tok/s (all experts in RAM: ~13 s) |
| 3K-token prompt | ~1.2 s; 0.5 s with a cached system prompt |
| Host RAM | ~15 GB (all experts in RAM: ~33 GB) |
| Load | ~2.5 min |

Measured on an Arc Pro B70 (32 GB) with 62 GB of RAM and a PCIe 4 NVMe SSD, fp8 KV cache, 200K context.

## Where the experts live

The model has 48 layers of 512 routed experts (10 per token), 1.86 MB each at 3 bits: 45.8 GB. Upstream keeps every
one of them in pinned host memory; this patch splits them in three:

- **VRAM (static tier):** the 7,500 experts game-master turns route to most (`hot/gm-7500.json`) are packed into device
  slots at load and never evicted (`EXL3_MOE_STATIC`, `EXL3_MOE_HOT`); 100 more slots are an LRU cache. Layer 0's
  4-bit experts sit in their own fully static store (the patch adds K=4 MoE kernels; upstream only had K=3).
- **RAM:** 4,500 more (`hot/gm-ram-4500.json`, `EXL3_MOE_RAM`), 8.4 GB of pinned memory.
- **NVMe (cold tier):** the other 12,064 (22.5 GB) in a sparse O_DIRECT store (`EXL3_MOE_COLD_STORE`), written on the
  first load with a RAM list; a later list only writes the records it adds.

Decode never routes to a cold expert, and pays a margin against RAM ones: experts outside a VRAM slot are lowered by
0.5 logits for ranking only (the weights stay the unbiased softmax). Prompts route exactly: chunks of 2,048+ rows read
all of a layer's cold experts two layers ahead in the background (`xpu/csrc/cold_read.c`, parallel preads without
the GIL, ~6.5 GB/s), smaller ones read just the experts they pick. `--all-ram` keeps every non-VRAM expert in RAM
instead (~33 GB).

The margin, teacher-forced over 894 game-master reply tokens against exact routing (decode speeds from before the
kernel work below; VRAM already covers ~94% of decode routings, so the margin buys little speed):

| margin | same top token as exact | KL | decode |
|---|---|---|---|
| 0 | 94.0% | 0.020 | 33.8 tok/s |
| **0.5** (default) | 92.6% | 0.025 | 34.6 |
| 1 | 92.1% | 0.030 | 34.9 |
| 2 | 91.4% | 0.041 | 35.1 |
| 4 | 91.5% | 0.065 | 35.3 |

## What else the patch changes

- **Decode graphs** for every batch size up to the user count (SGLang leaves them off on XPU; eager decode is ~6.5
  tok/s).
- **Fused kernels** (`exl3xpu/fused_xpu.py`, Triton): the router (softmax, top-k, renormalise and the routing bias, ~19
  kernels per layer) and the hyper-connection RMSNorm (~10 kernels per call, 96 calls per token) as one kernel each.
  A token went from 4,886 kernels to 3,113: decode 34.8 -> 41.6 tok/s. Tried and left off, slower than SGLang's own
  path inside a graph: a fused hyper-connection mix and combine (`hc_xpu.py`), Triton QSA decode attention
  (`EXL3_QSA_TRITON_DECODE`), int8/fp8 hyper-connection GEMVs.
- **Staging buffers reserved at load**, so SGLang sizes the KV pool around them (long prompts used to run out of
  memory).
- **N-gram table writes without atomics.** Upstream's `nvme_publish` wrote host USM with system-scope atomics; xe can
  only resolve an atomic page fault by migrating the page, which a host buffer can't do, so under RAM pressure the
  first request faulted the GPU (`Engine memory CAT error`, AccessType 2). ESIMD stores that bypass the caches plus a
  system fence keep the visibility guarantee.
- **MTP** works (the draft layer's input projections served as EXL3, its experts pinned) but is off: once decode was
  fast, MTP 2 gained 5% on short turns and lost on long context (18K prompt 26.6 s). The 3-bit draft head predicts
  prose poorly (accept length ~1.8 of 3).
- SGLang's in-batch prefix deprioritisation is off: each prefill streams every non-resident expert, so concurrent turns
  sharing a system prompt must batch into one forward (4 users: 28-35 s to first token -> 10-17 s).
- **Linear-attention state slots scale with `--parallel`.** A running request takes 5 of SGLang's state slots and each
  cached prompt up to 2; a fixed 12 let only 2 games run at once whatever `--parallel` said (4 games: ~64 tok/s total,
  two at a time). b70 gives 8 slots a game and stores the states in bf16 (`--mamba-ssm-dtype bfloat16`): in fp32, 28
  slots took 3 GB and halved the full-attention pool (217K -> 99K tokens). Greedy replies under bf16 states differ from
  fp32 no more than two fp32 runs differ from each other, and a long-context recall answer was identical. When four
  turns arrive at the same moment, the fourth can still wait for the other three while earlier prompts' cached states
  hold the spare slots.

## Running it

```sh
b70 setup sglang                             # once: image, kernels, plugin (CPU only)
b70 serve /path/to/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw --port 30250 --parallel 4
```

OpenAI API on `http://127.0.0.1:30250/v1` (model name `flashnext-uncensored`, or `--name`), thinking on by default
(`chat_template_kwargs.enable_thinking: false` turns it off), tools through SGLang's `qwen3_coder` parser. The first
load writes the cold store under `state/stores/flashnext` (`--store DIR` elsewhere; it must be a filesystem where
O_DIRECT works, on btrfs a `chattr +C` directory).

Options: `--ctx` (204800, at most 217088), `--parallel` (1-8), `--all-ram`, `--ram-list`/`--hot` (lists in `hot/`;
`general-*` ones are built from general prompts instead of game-master turns), `--margin`. `EXL3_*` and `SGLANG_*`
variables pass through; `EXL3_MOE_ROUTE_CTL=/run/b70/route_ctl.json` re-reads `margin`, `bias_prompts` and
`cold_all_min` from `state/run/sglang/route_ctl.json` while the server runs.

## Tools

- `tools/cold_bench.py speed|margins|quality`: concurrent-user speed, decode speed per margin, teacher-forced quality.
- `tools/profile_run.sh` + `tools/route_profile.py` + `tools/route_to_hot.py`: count which experts real turns route to
  (`EXL3_MOE_PROFILE`), then build hot and RAM lists from the counts and score any list against held-out turns.
- `tools/prof_live.sh` + `tools/prof_cats.py`: a torch-profiler trace of one request, every kernel attributed to its
  module. Graph replays hide kernels from the profiler, so profile with graphs off (`EXL3_PROF_SCOPES=1`), and ask
  for the "XPU" activity (SGLang maps "GPU" to CUDA). `EXL3_MODBENCH=1` times each module inside its own XPU graph at
  startup instead.
