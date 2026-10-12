# Qwen3.8-Flash-Next Uncensored (EXL3) on the Arc Pro B70 with trellis-serve

The [Lygodactylus 3.05 bpw EXL3 uncensored build](https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw)
served by 0xSero's [trellis-serve](https://github.com/0xSero/trellis-serve/tree/ft-xpu-vision/xpu) (SGLang 0.5.20-xpu
plus the exl3xpu plugin and its SYCL kernels). On the B70, 0xSero measured turboderp's EXL3 build of the same model at
~1,000 tok/s prompt reading (8K) and 32-36 tok/s decode.

There is no Docker. The pinned SGLang image is unpacked into a plain folder and run with bubblewrap, the same way NQ
runs exl3xpu.

**Status (2026-10-05): ~34 tok/s decode and ~1,100 tok/s prefill on the B70**, level with 0xSero's numbers. It loads
in ~140-170 s with no GPU faults. Three things got it there:
- decode graphs (`--cuda-graph-backend-decode full`; SGLang leaves them off on XPU, eager decode is ~6.5 tok/s);
- a static expert tier profiled on game-master turns (`hot/gm-7500.json`): decode ~29 -> ~34 tok/s on held-out turns;
- `CHUNK=8192`: a 4,608-token prompt in one chunk, 4.8 s (~1,115 tok/s) against 8.7 s at 4,096.
Prefix caching works: a GM turn that repeats a 3K-token system prompt reuses 3,008 cached tokens and starts answering
in ~0.5 s.

**Game-master hot list and prefill (2026-10-05).** `EXL3_MOE_PROFILE=<path>` makes the plugin count, on the device,
which experts every token routes to (graph-safe, so decode replays count too) and time the staged prefill MoE.
`profile_run.sh` drives 10 GM turns (phase A) and 6 held-out ones (phase B) with NQ's Game Master voice prompt and
synthetic player turns; `route_to_hot.py` builds a hot list from the counts and scores any list against them.

| 7,500 static experts | held-out GM decode routings covered | decode, held-out turns | cached-prefix first token |
|---|---|---|---|
| Strata profile | 39.2% | 28.9 tok/s | 0.71 s |
| built from phase A | 94.0% | 34.4 tok/s | 0.53 s |

`hot/gm-7500.json` is built from phases A and B together and is the default. Generic prompts (no GM system prompt)
still decode at ~34 tok/s once warm.

Prefill of a 4,608-token prompt with that list: 8.7 s at `CHUNK=4096` (MoE 3.7 s), 4.8 s at `CHUNK=8192` (MoE 1.75 s).
Each chunk streams every layer's non-resident experts, so one chunk halves the MoE time; the rest (attention, linear
attention, dense layers, n-gram lookups) also fell from ~4.3 to ~2.4 s. The staging copy itself runs at 28 GB/s, the
full PCIe Gen4 x16 link (`tests/bench_staging.py`: one layer's MoE at 4,096 rows is 13.4 ms of compute against 34 ms
of copy). An earlier `CHUNK=8192` try looked slower only because other work was loading the machine.

**More expert slots (in progress).** 11,000 slots (10,000 static) at `MEMFRAC` 0.87 loads with 4.26 GB of VRAM
free. Warm decode rises from ~30 to 31.8-32.6 tok/s. A 4,596-token prompt then ran out of memory (cleanly, no GPU
fault): long-prompt prefill streams non-resident experts through two staging buffers of 512 x 1.86 MB (0.91 GB each).
They were allocated on first use, so the load-time memory sizing never counted them, and layer 0's 4-bit store
allocated one too (1.27 GB) although all its experts are pinned.

Fixed in the patch, and verified on the B70 at the defaults (no faults, no run-time growth): `ExpertStore.reserve_staging()` allocates the staging buffers during
weight loading, so SGLang sizes the KV pool around them. Each holds only a layer's non-resident experts, at most 512
minus its static tier: 398 experts (0.74 GB) at 7,500 static, 369 (0.69 GB) at 10,000; none for layer 0's store. That
is ~1.4 GB of staging instead of up to 3.2 GB. `MAMBA_CACHE=N` caps SGLang's linear-attention state cache (10 states,
1.16 GB; the extra states hold cached prompt prefixes, so it stays off by default). `MEMFRAC` now defaults to 0.75 (0.80 with `SPEC=1`): the staging is counted in it, and 0.70 left no room for the
caches.

Measured at the defaults (8,500 / 7,500 / 0.75): 74K tokens of KV, 8.1 GB of VRAM free after graph capture, decode
~30 tok/s, and a 4,608-token prompt (cold) in 9.6 s including 20 output tokens, about 516 tok/s of prefill. Each
prefill chunk streams every layer's non-resident experts (the ~31 GB in host RAM at this tier), so the 4,096 + 512 chunks paid that
twice; `CHUNK=8192` (now the default) fixed that, see the game-master section below. Still to run: 10,000 or 11,000
slots, with `MEMFRAC` set so the KV pool doesn't take the freed memory.

**MTP speculative decoding (`SPEC=1`) works but doesn't help, so it's off by default.** Upstream had never run it for
this model. SGLang's Qwen4-exp MTP class builds its input projections (`fc_embedding`, `fc_hidden`) as plain
`nn.Linear`. The patch serves them as EXL3 and shares the target's embedding, and `SPEC=1` pins the MTP layer's 512
experts on the GPU.

- 3 draft steps: accept length 2.12 of 4 (38% acceptance), about half the speed of graphs alone.
- 1 draft step: accept length ~1.6 of 2 (60%). Warm, 150 tokens took about 5 s, the same as without speculation.

Each verify pass runs 2-4 tokens through every MoE layer, which multiplies the expert fetches. That traffic is the
bottleneck here, so 0xSero's dense-model gain (32 -> 51 tok/s) doesn't carry over. The uncensored fine-tune's draft
head may also fit the modified weights less well, though that is untested.

## Smaller RAM footprint and faster decode (2026-10-11/12, measured on the B70)

**Cold tier.** The GLM-5.3 port's approach (`~/work/glm-b70`): keep only a small RAM tier, put the rest of the experts
on NVMe. `RAM_JSON=hot/gm-ram-4500.json HOT_JSON=hot/gm-7500.json scripts/start_server.sh` keeps 4,500 experts in RAM
(8.4 GB) next to the 7,500 in VRAM; the other 12,064 (22.5 GB) live in an NVMe store in `QWEN_NVX`
(`~/models/qwen38-store`, `chattr +C`), written on the first load with a list (only new records later). Server RAM:
~33 GB -> ~15 GB. With a list, `start_server.sh` lowers the guard's start threshold to 25 GB.
- **Decode** never routes to a cold expert (logit -1e4 for decode and MTP-verify batches, by SGLang's forward mode).
  `RAM_MARGIN=x` lowers experts outside a VRAM slot by x for ranking only (weights from the unbiased logits). Qwen's
  router is a softmax, so x is in logit units. Teacher-forced over 894 GM reply tokens vs exact routing:

  | margin | top-1 vs exact | KL | decode |
  |---|---|---|---|
  | 0 | 94.0% | 0.020 | 33.8 tok/s |
  | **0.5** (GLM's 0.2 equivalent: ~1 pt top-1) | 92.6% | 0.025 | 34.6 |
  | 1 | 92.1% | 0.030 | 34.9 |
  | 2 | 91.4% | 0.041 | 35.1 |
  | 4 | 91.5% | 0.065 | 35.3 |

  (speeds from before the kernel work below). VRAM already covers ~94% of GM decode routings, so the margin buys little.
- **Prompts** route exactly. Chunks of >= 2048 rows read ALL of a layer's cold experts, two layers ahead, in the
  background (native C reader `xpu/csrc/cold_read.c` via ctypes, ~6.5 GB/s, no GIL), one upload per layer; smaller
  ones read the experts they pick. 18K prompt: 17.4 s before the read-ahead, ~14-16 s now (all-in-RAM: ~13 s).
- Runtime knobs (`EXL3_MOE_ROUTE_CTL=/run/trellis/route_ctl.json`, file `run/route_ctl.json`): `margin`,
  `bias_prompts` (decode routing on prompts, for teacher-forced quality tests), `cold_all_min`.
- `USERS=N` runs N requests at once (decode graphs for 1, 2, 4..N). `serve_b70.sh` disables SGLang's in-batch prefix
  deprioritisation: with it, 4 concurrent GM turns ran as 4 separate expert-streaming prefills (28-35 s to first token
  -> 10-17 s, 1-6 s once the GM prefix is cached).
- Tools: `scripts/cold_bench.py` (speed / margins / quality), `run/ab_long.py`, `run/mtp_matrix.sh`.

**Decode kernels.** Graph replays hide their kernels from the profiler; `GRAPHS=0 EXL3_PROF_SCOPES=1` +
`scripts/prof_live.sh <id> decode` + `scripts/prof_cats.py` (the profile must ask for "XPU", not "GPU") attribute every
kernel to its module, and `EXL3_MODBENCH=1` times each module kind inside its own XPU graph at startup. A decode token
was 4,886 kernels. `exl3xpu/fused_xpu.py` (Triton): the router (softmax + top-k + renormalise + routing bias: ~19
kernels per layer, two of them radix sorts) as one kernel, and the hyper-connection RMSNorm (~10 kernels per call, 96
calls) as one. 3,113 kernels per token; **decode 34.8 -> 41.6 tok/s**, decode at 18K context 31.4 -> 36.7.
Tried and left off (slower in a graph than SGLang's path): fused hyper-connection mix (`hc_xpu.py`), fused combine,
Triton QSA decode attention (`EXL3_QSA_TRITON_DECODE`).

**MTP.** Before the kernel work MTP 2 was best (+10%, accept 1.79); after it MTP 2 gives +5% on short prompts but
slows long-context decode and doubles 18K prompt time (26.6 s), so it stays off. MTP works (counting: 3.75 of 4
accepted); prose acceptance is just low with this 3-bit MTP head.

**Measured now** (cold tier, margin 0.5, no MTP, `USERS=4`): 1 user 41.7 tok/s (short GM turn), 36.7 at 18K context,
18K prompt 15.9 s; 2 users 68 tok/s total; 4 users ~50 tok/s total. A real Neverending Quest turn (throwaway campaign,
`local-server/flashnext-uncensored`) ran correctly in 31 s.

- `models/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw/`: the checkpoint (68 GB; all files match the published sizes)
- `image/rootfs/`: `lmsysorg/sglang@sha256:e4e10d66…` unpacked (Python 3.12, torch 2.13.0+xpu, SGLang 0.5.20, oneAPI
  2026.0 / icpx), the same base digest as 0xSero's `sglang-exl3-xpu-flashnext` image. `image/oci_pull.py` pulls an
  image by digest without Docker; `image/image-config.json` holds its environment.
- `trellis-serve/`: branch `ft-xpu-vision` @ 72fd2d4 (the image's pinned commit) plus our patch, on local branch
  `b70-static-experts` (`patches/trellis-serve-b70-static-experts.patch`). Its three kernel libraries are built
  (`xpu/exl3xpu/_C_sgl.so` with oneDNN, `_moe_sgl.so`, `exl3xpu_ngram_nvme.so`) and installed editable in the rootfs.
- `scripts/`
  - `in_rootfs.sh [--gpu] cmd…`: runs a command in the rootfs. It gets no GPU unless `--gpu` is passed.
  - `setup_rootfs.sh`: the one-time build (already done).
  - `profile_to_hot.py`: converts Strata's expert routing profile into an `EXL3_MOE_HOT` list.
  - `serve_b70.sh`: the launch.
  - `guarded.sh <log> cmd…`: runs a GPU command in its own process group, follows `journalctl -k`, and kills the group
    at the first xe fault, hang or reset. Repeated faulting runs wedged the xe driver and froze the machine once.
    It is also a RAM guard: it won't start with under `RAM_START_GB` (45) available or with one of the user's GPU jobs
    running (`OTHER_JOBS`: sd-cli, ComfyUI, llama-server), and kills the run when available RAM drops under
    `RAM_FLOOR_GB` (6), memory pressure (PSI full avg10) passes `PSI_MAX` (40), or such a job starts. The server
    pins ~35 GB of host RAM; once, alongside a video job, that ran the machine out of memory and systemd-oomd killed
    the whole terminal session. The command runs with `oom_score_adj` 1000, so the OOM killer takes it first.
  - `diag_run.sh`: a guarded server launch, one request, then stop. By default it runs with debug env and 20 tokens;
    `QUICK=1 MAX_TOKENS=100` gives real speed.
  - `stop_server.sh`: stops a running server.

## Checked without the GPU

- SGLang 0.5.20 has `Qwen4ExpForConditionalGeneration`.
- trellis-serve's scanner reads all 75,751 EXL3 modules: K = 3 / 4 / 5, all the `mul1` codebook. Upstream's MoE
  kernels only handle K=3, but layer 0's experts in this checkpoint are K=4. The patch adds K=4 MoE kernels; on the
  B70 they match the reference to the bf16 floor. K=4 experts get their own fully static store (1.27 GB).
- The n-gram table parses: 320,001,536 rows, K=3 (turboderp's is K=5). The decoder takes K at run time.

## The RAM problem and the patch

Upstream keeps **every** routed expert in pinned host memory: 512 × 48 × 1.86 MB = 45.8 GB. Their B70 reference run
peaked at 67 GB of host memory, and this machine has 62 GB.

The patch adds a static tier. With `EXL3_MOE_STATIC=1`, the experts listed in `EXL3_MOE_HOT` are packed straight into
device slots at load. Those slots are pinned with `slot_last = INT_MAX`, which the LRU kernel already treats as
"never a victim", and they get no host copy.
- The host arena holds only the other experts, compacted.
- `host_base` became a per-expert home-pointer table, used by the eviction path, the copy-engine prefetch worker and
  the prefetch copy kernel (three lines in `exl3_moe.sycl`).
- Cache reset, eviction and prefill staging respect the static tier.

The hot list is the top N of Strata's 24,576 ranked (layer, expert) pairs (`~/work/strata-b70/Strata/data/expert-profile.bin`).

Defaults in `serve_b70.sh`: 8,500 K=3 slots, 7,500 of them static (15.8 GB of VRAM for all slots) and 1,000 left as
the LRU cache, at `MEMFRAC` 0.70 (65K-token fp8 KV, ~9.7 GB of VRAM still free). That leaves 30.8 GB of experts in
pinned host memory, MTP layer included. Add the n-gram RAM cache (4 GB) and the
process (~5 GB): about 41 GB, against the ~48 GB this machine has available when idle.

## The n-gram publish fault (fixed)

The first runs died on the first request with `UR_RESULT_ERROR_OUT_OF_RESOURCES`. The kernel log showed xe
`Fault response: Unsuccessful -EINVAL` and `Engine memory CAT error`, with AccessType 2 (atomic) on the n-gram NVMe
tier's `req` host-USM buffer.

Upstream's `nvme_publish` wrote the lookup ids and sequence words into host memory with system-scope atomic exchanges.
xe can only service an atomic page fault by migrating the page to VRAM, and a host USM buffer can't move. Under the
server's RAM pressure the shrinker unbinds host buffers, so the first atomic touch after that faulted. Standalone
probes, with plenty of free RAM, never faulted.

The patch writes them with ESIMD LSC stores that bypass L1 and L3, plus a system fence. These are ordinary writes, so a
fault on them can be resolved. They also keep the host-visibility guarantee upstream used atomics for.
`tests/test_ngram_nvme_k3.py` passes on the B70, bit-identical against the host table, graph replays included.
Copy-prefetch's `host_plan` has the same atomic-store pattern, but serving doesn't use copy-prefetch.

## Running it (needs the B70 and ~42 GB of free RAM)

```sh
~/work/trellis-b70/scripts/serve_b70.sh      # OpenAI API: http://127.0.0.1:30250/v1, model "flashnext-uncensored"
```

What to watch in the log:
- `exl3xpu: layers.N.mlp.experts: 512 experts K=3, … packed to host USM (… GB), … static`: one line per layer; the
  static counts should add up to 8,000.
- `n-gram table … K=3`: the NVMe tier opening.
- KV pool size and `Uvicorn running`.

If the static tier misbehaves, set `EXL3_MOE_STATIC=0` to fall back to upstream's behaviour
(all experts in host RAM, 45.8 GB), which only fits with almost everything else closed.

Knobs (env): `GRAPHS` (1), `SPEC` (0), `SPEC_STEPS` (1), `SLOTS`, `HOT`, `CTX` (65536), `NGRAM_RAM_GB` (4), `MEMFRAC` (0.75), `MAMBA_CACHE` (unset: 10), `CHUNK` (8192), `HOT_JSON` (hot/gm-7500.json; empty = Strata profile with `HOT`), `PORT` (30250).
