# GLM-5.3-Flash on an Intel Arc Pro B70

GLM-5.3-Flash (a ~105 GB EXL3 mixture-of-experts model: 42 MoE layers × 288 experts, KDA linear attention,
MLA/DSA sparse attention, mHC hyper-connections) running with one **Intel Arc Pro B70** (Battlemage, 32 GB) as the
only GPU, on a desktop with 62 GB of RAM and one PCIe 4.0 NVMe SSD. The model does not fit in VRAM or RAM, so the
routed experts are spread over three tiers: VRAM, pinned RAM, and an NVMe expert store.

It started from [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) (which serves the model from a
24 GB RTX 3090 + RAM + NVMe on CUDA exllamav3) and [exllamav3](https://github.com/turboderp-org/exllamav3), and
runs exllamav3's Python model code on PyTorch XPU with our own SYCL/ESIMD and Triton kernels underneath.

Model used: [MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3)
(routed experts 2-bit in 22 layers, 3-bit in 20).

**Result** (single user, greedy):

| | First working version | Now |
|---|---|---|
| Decode | 5.2 tok/s | **~15-18 tok/s** |
| Prompt, 2K tokens | ~40 s | **9.0 s** (225 tok/s) |
| Prompt, 4K tokens | 72 s for 3.6K | **10.5 s** (382 tok/s) |
| Prompt, 8K tokens | out of memory | **14.4 s** (556 tok/s), one pass |
| Chat follow-up (2K cached + a short message) | re-read the whole history | **1.7-3 s** to the first token |

## Where the code is

- `xpu/glm_xpu/`: the runtime.
  - `tier.py`: the three-tier expert cache (VRAM slots with frequency-gated admission, pinned RAM read zero-copy
    by the kernels over PCIe, NVMe store with parallel O_DIRECT reads; prompt staging).
  - `block_graphs.py`: per-block XPU graphs for decode, the no-sync router and its RAM margin, MTP verify steps.
  - `mla_graph.py`: MLA/DSA attention with device-side lengths so a whole layer fits in a graph.
  - `k_hc.py`, `k_kda.py`, `sparse_mla.py`: Triton / oneDNN kernels (mHC mix, KDA step, sparse MLA).
  - `live.py`: live speed meter (`tail -f run/live.log`).
- `xpu/csrc/`: SYCL/ESIMD kernels. `exl3_moe.n128.sycl` is the grouped EXL3 MoE kernel (GEMV for decode, DPAS for
  prompts) reading expert weights through a pointer table, so VRAM- and RAM-resident experts look the same to it.
  The EXL3 linear kernels started from [trellis-serve](https://github.com/0xSero/trellis-serve) (`xpu/TRELLIS_COMMIT`).
- `xpu/exllamav3_ext/`: stands in for exllamav3's compiled CUDA extension on XPU.
- `xpu/pack_store.py`: packs the routed experts into the NVMe store.
- `xpu/devserver.py`, `xpu/dev.py`, `xpu/snip/`: a persistent dev server (model loaded once) and the measurement
  snippets every number below comes from.
- **exllamav3 fork:** [horvay/exllamav3 `b70-xpu`](https://github.com/horvay/exllamav3/tree/b70-xpu), also as
  `patches/exllamav3.patch` against turboderp-org/exllamav3 `958ec93` (v1.5.1).
- `docs/dev-notes.md`: the full working notes, including hardware facts and Level Zero pitfalls.

## The bottlenecks, in the order they were hit

### Decode

| # | Symptom | Cause | Fix | Effect |
| --- | --- | --- | --- | --- |
| 1 | ~5 tok/s | One host sync per MoE layer: the CPU had to read the router's picks and make those experts resident before the GPU could continue (42 times per token) | **No-sync routing**: the router only picks experts already in VRAM or RAM (an NVMe-only pick becomes the next-best resident one); cache bookkeeping runs once per token, one token late, overlapped with the dense layers | ~12-13 tok/s |
| 2 | MLA attention 15.7 ms/token | ~70 separately launched ops per layer, host-bound; their grids depended on host-side lengths | Device-side lengths and fixed-size top-k selection, whole MLA layer captured in an XPU graph (`mla_graph.py`) | 8.9 ms/token |
| 3 | ~15 ms/token in tiny ops | The mHC hyper-connection mix ran its 1.5 MB reduction in a single work-group at batch 1, twice per block | Split the reduction over 64 work-groups + a finishing kernel | 166 → 17 µs per call |
| 4 | ~20 ms/token reading RAM experts | Every RAM-resident pick is streamed over PCIe (~27 GB/s vs ~450 GB/s VRAM) | **RAM margin** in the router: a RAM expert's score is lowered by 0.2 for ranking only, so near-ties go to VRAM experts | 12.9 → 17.8 tok/s at the same quality (below) |

RAM margin sweep (top-1 agreement and KL against exact routing, teacher-forced over 192 positions; speed on 128-token
generations):

| Margin | Top-1 vs exact | KL | Decode |
| --- | --- | --- | --- |
| 0 (plain no-sync) | 92.7% | 0.072 | 12.9 tok/s |
| 0.05 | 92.7% | 0.061 | 15.2 tok/s |
| 0.1 | 90.6% | 0.083 | 17.0 tok/s |
| **0.2 (default)** | **91.7%** | **0.085** | **17.8 tok/s** |
| 0.3 | 88.5% | 0.135 | 18.7 tok/s |
| 0.5 | 88.5% | 0.171 | 18.9 tok/s |
| VRAM only | 77.1% | 0.524 | 21.2 tok/s |

### Prompts

| # | Symptom | Cause | Fix | Effect |
| --- | --- | --- | --- | --- |
| 5 | Slow prompts in half the layers | The DPAS (matrix engine) prompt kernel was 3-bit only; the 2-bit layers fell back to GEMV | Generalized the DPAS B-operand build to 2- and 4-bit | prompt kernel for every layer |
| 6 | Every prompt fetched all experts twice | exllamav3 runs a separate forward for the last page to save a recurrent (KDA) checkpoint | Stash the checkpoint inside the single forward (the KDA layer splits itself at that position) | 2K prompt 16-20 s → 9 s |
| 7 | Out of memory above ~3.5K tokens; 0.7 GiB per 1K tokens | KDA prefill kept dead fp32 tensors alive; MLA's sparse path held ~200 KB/token of per-row tensors; XPU SDPA materialized the scores (quadratic) | Free early, in-place cumsum, per-layer splits (KDA every 2048, MLA 1024-token sub-chunks, MLP row slices) | ~85 MiB per 1K tokens; 8K in one pass |
| 8 | Crash on the first token after a long prompt | PyTorch kept the prompt's freed blocks reserved; the next graph replay got `OUT_OF_DEVICE_MEMORY` | Release the cache at the end of a large forward | fixed |
| 9 | 18.5 s for a short chat follow-up | The prompt path streamed every non-resident expert of the layers ahead | Short prompts read only the experts they picked, in parallel | 1.7 s |

### MTP (speculative decoding): tried, not a win here

The model ships an MTP head, and it works (68-88% of drafts accepted), with verify steps on the graph/no-sync path.
But consecutive tokens share only ~1 of 8 experts per layer, and expert reads dominate the step, so checking 2 tokens
costs about twice one token. At margin 0.2: MTP off ~15-18 tok/s, MTP 1 ~15-20, MTP 2 ~12-16, MTP 4 ~10. MTP pays off
when the experts are in VRAM; with a 105 GB model on a 32 GB card it is off by default (`GLM_XPU_MTP`).

## Running it

From the repo root (see the [top-level README](../README.md) for what the host needs):

```bash
./b70 setup glm                                   # once: the shared image, glm/pylib, the SYCL kernels (CPU only)
huggingface-cli download MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3 --local-dir /path/to/glm53-model
./b70 prepare /path/to/glm53-model                # once: pack the routed experts into the NVMe store (~95 GB written)
./b70 serve /path/to/glm53-model --port 8053      # OpenAI-compatible server (glm/xpu/serve.py), one request at a time
```

The server takes chat completions (streamed or not; the model's own chat template, tools, thinking as
`reasoning_content`, `chat_template_kwargs.enable_thinking` and `reasoning_effort`) and raw completions. A chat
follow-up that repeats the previous prompt reuses its cached pages and recurrent checkpoint. The first prompt after a
start is slow (~60 s of one-time kernel compiling); expect the first decode right after a start to be slower too: the
RAM tier fills in the background and the VRAM tier adapts to what the model actually uses.

For development, `scripts/dev_restart.sh` starts a persistent dev server and `xpu/dev.py` runs a Python snippet inside
it (`model`, `tok`, `gen` (exllamav3 Generator), `tier` are in scope; see `xpu/snip/`). `scripts/in_rootfs.sh` runs a
command in the image with this folder at `/w` (`GLM53_MODELS`, `GLM53_NVX` set the model and store binds).

Settings (environment):

| Variable | Default | |
| --- | --- | --- |
| `GLM_XPU_RAM_MARGIN` | 0.2 | router margin against RAM experts (0 = plain no-sync) |
| `GLM_XPU_NOSYNC` | 1 | 0 = exact routing (one host sync per MoE layer, ~8.7 tok/s) |
| `GLM_XPU_MTP` | 0 | MTP draft tokens |
| `GLM_XPU_CHUNK` | 8192 | prompt tokens per forward |
| `GLM_XPU_MLA_GRAPH` | 1 | MLA attention inside the decode graphs |

## Known limits

- The first prompt after a server start takes ~60 s (one-time JIT compile of the SYCL kernels).
- Prompts over 8K tokens run in 8K pieces, each fetching the experts again; 12K in one pass runs out of VRAM.
- Batch size 1: one request at a time.
- Faults seen while developing (all fixed, see `docs/dev-notes.md`): a failed XPU graph capture leaves PyTorch's
  graph pool recording and later replays fault the GPU, so a capture failure now disables graphs for the session;
  multi-row `topk`/`sort` touched the pinned host allocator during capture.

## License

MIT (see `LICENSE`). Third-party credits in `NOTICE`.
