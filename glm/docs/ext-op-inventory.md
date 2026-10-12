# exllamav3_ext → XPU porting spec for GLM-5.3-Flash (ext op inventory)

Target: exllamav3 **v1.5.1** (commit `958ec93`, checkout `glm/exllamav3`) running
GLM-5.3-Flash EXL3 (`turboderp/GLM-5.3-Flash-exl3`, rev `332ab457`, 3.05 bpw) on one Intel Arc Pro B70.
The plan replaces the compiled `exllamav3_ext` module with a Python shim built from torch, Triton and the
existing XPU kernels. Text-only, single device, no tensor parallel, driven by
`glm/glm-flash-lite/glm53/serve.py`
(`-cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4`, AsyncGenerator, no `--mtp`/`-dm`/`-ngram`/`-cq`).

Conventions:
- Paths are relative to `glm/exllamav3`. The Python package is `exllamav3/` and the
  CUDA sources are `exllamav3/exllamav3_ext/`. Most per-part tables drop the `exllamav3/` prefix; each part
  states its base.
- Line numbers are for v1.5.1.
- "Shim" means the replacement `exllamav3_ext` Python module.
- Checkpoint facts come from the safetensors headers of all 15 shards, read with HTTP range requests (no
  weights downloaded), and from `/mnt/sandisk/glm53/model/config.json` and `kpool_aux.safetensors`
  (read-only).

Layout of this document:
1. Direct answers: MTP, codebook, batch sizes.
2. GLM5Next forward at a high level, and the state each layer keeps.
3. Recommended XPU configuration: env vars, flags, Python patches, shim rules.
4. Master op table in hot-path order, then the ops the configuration avoids and the ops never reached.
5. Per-op detail, Parts A–F: signatures, exact math, data layouts, call sites, CUDA deps and Triton kernels per area.
   - Part A: EXL3/FP16 linear and the quant kernels (trellis layout, mul1 decode, Hadamard).
   - Part B: dense MLP, MoE, routing, activations.
   - Part C: KDA linear attention and recurrent state, plus the vendored fla Triton kernels.
   - Part D: MLA (NoPE) and the k-pool DSA indexer, the MLA/DSA Triton kernels, and the KV/indexer cache.
   - Part E: hyper-connections, RMSNorm, TransformerBlock, model forward, load, warmup, loader, graphs.
   - Part F: generator, page table, sampling, cache allocation, recurrent checkpoints.
6. Consolidated non-ext CUDA dependencies and Triton kernel summary.
7. Minimal op set for a correct (not fast) forward.

---

## 1. Direct answers

**Is MTP used by default? No.**
- `model_init.init` turns MTP on only with `--mtp` or `-dm <model_dir>` (`exllamav3/model_init.py:197-204`), and
  serve.py's documented command passes neither.
- `--ngram_match_min` defaults to 0, so `num_draft_tokens = 0` and `max_history = 0`
  (`model_init.py:270-275`, `generator/generator.py:168-171`).
- As a result `recurrent_history` is always False, and the rewind ops are not reached
  (`batched_conv_rewind`/`batched_state_rewind`).
- **The checkpoint has no MTP layer at all.** `layers.45.*` is absent even though
  `num_nextn_predict_layers = 1`. `--mtp` would first need a `convert_mtp.py` output placed next to the
  model.
- `architecture/glm5_next_mtp.py` is off the path. Skip it.

**Which codebook and bitrates?**
- **mul1 everywhere.** Every EXL3 linear ships a scalar I32 `<key>.mul1` (value `0x83DCD12D`, a flag only;
  the multiplier is hard-coded in the kernels). There are zero `.mcg` tensors, no bias, and no packed
  `su`/`sv` sign tensors, only `suh`/`svh` fp16.
- Bitrates are whole numbers, so the fractional-trellis (`frac.cu`, half-integer K) code is never reached.
  The checkpoint was produced by exllamav3 1.4.4; fractional support landed after 1.5.0.

  | tensors | K |
  |---|---|
  | routed experts | 3 |
  | dense MLP (layers 0–2) | 4 |
  | shared experts, every attention/KDA projection, indexer `wq_b` | 5 |
  | `lm_head` (`head_bits`) | 6 |

  This matches the K set the existing trellis-serve XPU kernels support. Details in Part A §1.0.

**Decode batch sizes.**
- serve.py's `-ambs 4` is passed by `model_init` as the Cache's `max_batch_size`, which sets
  `Cache.num_slots = 4` recurrent-state slots.
- For recurrent models the Generator then clamps its batch to `min(max_batch_size, cache.num_slots)`
  (`generator/generator.py:235`). So **decode runs at bsz 1–4 (q_len 1)** despite `--max-batch-size 8`.
- The ops must still accept up to 8 rows: warmup and `MAX_BSZN = 8` buffers use 8, and `-ambs 8` would
  make bsz 5–8 real.
- Prefill is always batch 1, in chunks of up to 8192 tokens.

---

## 2. GLM5Next forward at a high level

Config: hidden 4096, 45 layers, vocab 154880, rms_norm_eps 1e-5. Module list from
`architecture/glm5_next.py:149-344`:

| idx | module | notes |
|---|---|---|
| 0 | `Embedding` | `prefer_cpu`: lives on the **CPU**, bf16/fp16 table, output **fp32** (b,s,4096); uploaded through a pinned buffer at decode |
| 1 | `ExpandStreams` | `x.float().unsqueeze(2).expand(..., 4, ...)`: the residual becomes **4 identical fp32 streams** (b,s,4,4096) |
| 2..46 | 45 × `TransformerBlock` | below |
| 47 | `HyperHead(mean=True)` | `x.mean(dim=2)`, no weights (`hc_head` op not used) |
| 48 | `RMSNorm` (final) | fp32 in, fp16 out |
| 49 | `Linear lm_head` | EXL3 K=6, fp16 logits; the generator slices to the last token first (`last_tokens_only`) |

**Per-layer structure.** `modules/transformer.py:137-215` is the same for every layer. Only the attention
and MLP kinds differ.

```
post, comb, y = hc_mix(streams)                         # mHC "pre": y = Σ_h pre_h·stream_h (fp16), plus post (4), comb (4x4)
y  = input_layernorm(y)                                  # RMSNorm fp16 -> fp16
y  = attention(y)                                        # KDA (34 layers) or MLA+indexer (11 layers); out fp32
streams = post ⊗ y + combᵀ · streams                     # hc_apply, in place, fp32
post, comb, y = hc_mix(streams)                          # second HC site (ffn)
y  = post_attention_layernorm(y)
y  = mlp(y)                                              # dense GatedMLP (layers 0-2) or MoE (3-44); out fp32
streams = post ⊗ y + combᵀ · streams
```

For non-final prefill chunks, `Model.prefill` stops after the **attention** of the last block. That fills
the caches; HyperHead, the norm and lm_head do not run (`transformer.py:160`, `model_ls.py:298-314`).

**Hyper-connections** (mHC, `hc_mult = 4`, Sinkhorn 20 iterations, `hc_eps = 1e-6`). Each layer has two
sites, `hc_attn` and `hc_ffn`. Each site has `fn` (24,16384) fp32, `base` (24) and `scale` (3).

```
mix   = streams_flat(16384) · fnᵀ ;  r = rsqrt(mean(streams²) + 1e-5)
pre   = σ(mix[0:4]·r·scale0 + base) + 1e-6
post  = 2σ(mix[4:8]·r·scale1 + base)
comb  = Sinkhorn(softmax_j(mix[8:24]·r·scale2 + base) + eps)
```

Sinkhorn is one column normalisation followed by 19 (row, column) normalisation pairs, each adding eps.
Exact kernel math is in Part E. The HC streams carry **no cache state**.

**Layer kinds.** `layer_types` puts MLA at 3, 7, 11, …, 43 (11 layers) and KDA at the other 34 (3:1).
`mlp_layer_types` makes layers 0, 1 and 2 dense and 3–44 MoE (`first_k_dense_replace = 3`; layer 44 is
sparse). Every `indexer_types` entry is `"full"`.

**KDA (Kimi delta attention) layers**: `GatedDeltaNet` in KDA mode, 64 heads, Dk = Dv = 128.
1. `qkv = qkv_proj(x)` (EXL3, 4096 → 24576).
2. Depthwise causal conv (K=4) with SiLU over the 24576 channels, using the per-slot conv state.
3. q and k are L2-normalised (eps 1e-6).
4. Per-channel log-decay `g = −5·σ(exp(A_log[h])·(f_b(f_a(x)) + dt_bias))`, from low-rank f_a/f_b fp16
   linears (`gate_lower_bound = −5`).
5. `β = σ(b_proj(x))`.
6. Delta rule per token: `S = diag(e^g)·S; u = v − Sᵀk; S += β·k·uᵀ; o = Sᵀq/√128`.
7. Gated RMSNorm per head (eps 1e-5, weight `o_norm` (128), gate `σ(g_b(g_a(x)))`).
8. `o_proj` (EXL3, 8192 → 4096, fp32 out).

Prefill of 64 tokens or more uses the vendored fla `chunk_kda` Triton kernels. Shorter prefills and decode
use the recurrent kernel.

Cache state per layer per slot:

| tensor | dtype | shape |
|---|---|---|
| `recurrent_state` | fp32 | (slots, 1, 64, 128, 128), 4 MiB per slot, `[slot, hist, h, k, v]` with v contiguous |
| `conv_state` | bf16 | (slots, 24576, 4) |

Host-RAM checkpoints of these states are taken every 2048 tokens (32768 during long prompts) into the
4 GiB `RecurrentCache`. This is pure torch copying (Part C, Part F).

**MLA layers**: NoPE (`qk_rope_head_dim = 0`), 64 heads, `kv_lora_rank` 512, `q_lora_rank` 1536,
qk 256, v 256, `sm_scale = 1/16`.
1. `q = q_b(RMSNorm(q_a(x)))` → (R,64,256).
2. The latent `ckv = RMSNorm(kv_a(x))` (512) is **the only per-token KV cached**: fp16, paged at 256 tokens
   per page.
3. `kv_b_proj` (fp16, unquantized) is split at load into `W_UK` and `W_UV`. Decode and short chunks use
   the absorbed form (`q_lat = q·W_UK`, attention over the latent, `o = o_lat·W_UV`). Dense prefill chunks
   over 16 tokens use the MHA form.
4. `o_proj` (16384 → 4096, fp32 out).

**k-pool DSA indexer** (32 heads × 128, `index_topk` 2048, `index_kpool` 4, `always_select_tail`):
- **Every** forward computes indexer keys `k = LayerNorm(wk·x)` (eps 1e-6, no rope) and gate scores
  `gate_w·x` (128; from `kpool_aux.safetensors`) and caches them.
- Each completed group of 4 tokens is pooled: `pool_k[d] = Σ_j softmax_j(gate[j,d] + ape[j,d])·k[j,d]`.
- Once visible context exceeds 2048 tokens (the "sparse" regime, which already includes the first
  8192-token prefill chunk), each query row:
  - scores pools with `(1/√(128·32))·Σ_h w_h·relu(q_idx,h·pool_k)`, where `q_idx = wq_b(q_resid)` and
    `w = weights_proj(x)`;
  - takes the top 512 pools (`dsa_topk`) and expands each to its 4 tokens, plus the raw tokens of its own
    incomplete tail pool. That is ≤ 2051 token indices, padded to 2080;
  - attends only to those latent rows.

Cache state per MLA layer (fp16, 512 pages at `-cs 131072`, 1600 B/token in total):

| tensor | shape | contents |
|---|---|---|
| `k` | (pages,256,1,512) | latent |
| `v` | (pages,256,1,0) | empty (NoPE) |
| `k_idx` | (pages,256,256) | indexer key ‖ gate |
| `k_pool` | (pages,64,128) | pooled keys |

There is no recurrent state (Part D).

**MLP layers**:
- **Dense (layers 0–2):** `GatedMLP` 4096 → 12288 → 4096.
- **MoE (layers 3–44):** 288 routed experts (I = 2048) + 1 shared expert (I = 2048), top-8.
- **Activation** in both: `a = min(silu(g), 10)·clamp(u, −10, 10)` (`swiglu_limit = 10`; the gate is
  clamped *after* SiLU, from above only).
- **Routing** (`router_type "dots"` → `routing_ds3_nogroup`), sigmoid / noaux_tc, `n_group = 1`:
  1. `logits = fp16(x·gate)`.
  2. `s = σ(fp32(logits))`.
  3. Select the top-8 by `s + e_score_correction_bias`. The bias is used for selection only.
  4. Weights `w = s·2.5/(Σ_sel s + 1e-20)` (`norm_topk_prob`, `routed_scaling_factor = 2.5`), stored fp16.
- **Output** = `Σ_k w_k·down_k(act(gate_k x, up_k x))` + `shared(x)`, accumulated in fp32, with no shared
  gate (Part B).

---

## 3. Recommended XPU configuration (correct first, fast later)

**Env vars and flags**

| setting | why |
|---|---|
| `EXL3_BC_GDN=0` | KDA layers use the torch path. Avoids `BC_GatedDeltaNetSplit` (CUDA-graph fused layer), `kda_gate_op`, `gdn_ba_gemv`, `gdn_lowrank_gemv_f` (`modules/gated_delta_net.py:20`) |
| `EXL3_BC_ATTN=0` | MLA uses the eager Triton path. Avoids `BC_MLAttention`, `TritonKernel` and the cubin AOT compile. It covers `EXL3_BC_MLA` **and** the load-time `autosplit_extra_measure → build_bc_mla` call that `EXL3_BC_MLA=0` alone misses (`modules/attention_fn/bc_attn.py:42`, `modules/mla_attn.py:1110-1118`) |
| `config.infer_params.no_reconstruct = True` (set between `Config.from_directory` and `model.load`, e.g. patch `model_init.py` around :194) | Every EXL3 `Linear` goes through `BC_LinearEXL3.run_alloc` at every row count (`modules/quant/exl3.py:140`). The MoE uses the torch per-expert loop at every batch size (`block_sparse_mlp.py:1052-1055`), and no `MultiLinear`/`exl3_mgemm`/`BC_BlockSparseMLP`/fused-MoE buffers are built (`block_sparse_mlp.py:527,533,612`, `mlp.py:635`). This removes `exl3_mgemm` and its raw-pointer tables entirely |
| `TRITON_F32_DEFAULT=ieee` | The fp32 `tl.dot` in `chunk_kda_fwd_kernel_inter_solve_fused` has no explicit precision (Part C (d)) |
| `-nw` during bring-up | Skips warmup, which only pays Triton/autotune/graph first-use costs. Re-enable later to pre-JIT |
| leave `GLM53_*` unset | All glm-flash-lite monkeypatches (`k_hcfuse` inline CUDA, `triton_pin` NVIDIA autotune picks, the tiers) are CUDA-specific |
| no `-cq`, no `--mtp`, `-ccs 0` | Keeps the quantized-cache, draft and CPU page-cache ops unreached |
| `EXL3_FUSED_SAMPLER` default (on) | The whole sampler collapses to one `fused_sampler` call (Part F) |

**Python patches the shim cannot cover.** Each one is a hard failure or a silent wrong path on XPU.

| file:line | problem | fix |
|---|---|---|
| `model_init.py:360` / `model/model.py:551` / `model/model_ls.py:150` | autosplit uses `torch.cuda.device_count()` and int device ordinals (= CUDA). It also runs an 8192-token dummy forward per module at load and calls `torch.cuda` memory APIs (`util/memory.py`) | load single-device: `model.load(device=torch.device("xpu", 0))` (`model_init.init(args, device=...)` forwards kwargs) → `_load_single`, with no dummy forward and no memory accounting |
| `loader/safetensors.py:413` (default `load_method="mt_fread"`), `:793` | the stloader routes jobs on `tensor.is_cuda`, so an XPU tensor goes down the CPU path and the host writes through a device pointer | `load_method="python"` (`:839-857`): no ext calls at all. `Config.from_directory` passes `kwargs.get("load_method")` (`model/config.py:132`); patch `model_init.py:207` or the default |
| `modules/block_sparse_mlp.py:818` | `load_local()`/`load_routing()` run only if `device.type == "cuda"`. On XPU `routing_cfg` stays None and the first MoE forward crashes | accept `"xpu"` |
| `modules/gated_delta_net_fn/conv1d.py:245`, `:279` | the Triton conv raises `if not x.is_cuda`, and launches under `torch.cuda.device` | device-type check; `torch.xpu.device` |
| `modules/gated_delta_net_fn/conv1d.py:386` | `is_cuda` gate for the CUDA conv op | leave it: on XPU it routes every conv to Triton, so `cuda_causal_conv1d_update` is never needed |
| `vendor/fla/utils.py:117` (`input_guard`) | `torch.cuda.device(idx)` on every `chunk_kda` | `torch.xpu.device` |
| `modules/attention_fn/mla_triton.py:803,823,865,977,1029,1045,1127,1210`, `dsa_triton.py:1052,1082,1153` | `with torch.cuda.device(...)` around launches | `torch.xpu.device` (or a no-op context) |
| `modules/attention_fn/mla_triton.py:786` | `torch.cuda.get_device_properties(i).multi_processor_count` for split heuristics | XPU equivalent (Xe-core count, e.g. `gpu_subslice_count`; verify the field name in the installed torch) |
| `generator/generator.py:1128` | `torch.cuda.synchronize(batch_logits.device)` **every decode step** (also the barrier that makes reusing pinned staging buffers safe) | `torch.xpu.synchronize(dev)`. Do not drop it |
| `generator/generator.py:1122` | `next_token.is_cuda` gates the batched pinned read-back | `device.type != "cpu"` (otherwise correct, just slower) |
| `util/misc.py:35-43` (`cuda_sync_active`) | becomes a silent no-op on XPU (first-token timing only) | iterate XPU devices |
| `pin_memory=True` sites (`generator.py:938,1113-1114`, `job.py:1537`, `modules/embedding.py:164`) | XPU-pinned host memory | verify `.is_pinned()` on the installed torch-xpu. Fall back to unpinned (synchronous, still correct) |

**Shim construction rules**
- **Export every name in `bindings.cpp`.** Most only need to raise on call (e.g. a module `__getattr__`
  that returns a raising stub). Some names are touched at import or construct time:
  - `ext.relu_mul` at import (`modules/moe_batch_recon.py:61`; dict built at `:55-61`);
  - `gelu_mul`/`relu2_mul`/`silu_oai_mul`/`xielu` in `match` statements (`mlp.py:593-596`,
    `block_sparse_mlp.py:392-411`);
  - the int constants `FUSED_SAMPLER_MAX_BLOCKS = 128` and `FUSED_SAMPLER_HIST_STRIDE = 24640`.
- **`BC_LinearEXL3` must exist as a real Python class.** `LinearEXL3.__init__` constructs it
  unconditionally (`modules/quant/exl3.py:86`) and every quantized linear calls `self.bc.run_alloc`. It
  must expose the fields `trellis, suh, svh, K, bias, mcg, mul1` (other BC constructors read them).
  Likewise `BC_LinearFP16` (`modules/quant/fp16.py:54`) and `BC_GatedRMSNorm`
  (`modules/gated_rmsnorm.py:55`) are constructed unconditionally at load.
- **`BC_GatedMLP` must also exist (as a small Python class) even with `no_reconstruct`.** `GatedMLP.load_local`
  builds it whenever `downs[0].inner.bc` exists and gate/up are EXL3 with equal width (`mlp.py:655-687`),
  which is always true once `BC_LinearEXL3` exists. `GatedMLP.forward` then calls
  `bc.run_bszN(x, d)` for rows ≤ 8 (`mlp.py:744`). This covers the 3 dense layers and the shared expert of
  every MoE layer. Its body is `g = gate.run_alloc(x); u = up.run_alloc(x); a = silu_mul(g, u, 10); down.run(a, d)`
  (Part B, `BC_GatedMLP`). Part B suggests omitting `BC_LinearEXL3` to avoid this, but
  that is not possible: see `exl3.py:86`.
- **Shim functions must honour in-place outputs.** Many callers pass preallocated outputs and ignore the
  return value, and the caller reads the output tensor (examples: `hc_mix` → post/comb/collapsed,
  `rms_norm` → y, `routing_ds3_nogroup` → topk buffers, `fused_sampler` → out,
  `cuda_recurrent_gated_delta_rule` → state and out).
- **No raw pointers.** Under the recommended config no reached op receives device-address tables
  (`exl3_mgemm` and the `*_ptrs` MoE tables are avoided by `no_reconstruct`; rewind jobs by having no
  drafts). If the stock paths are revived later, the shim needs a `data_ptr → tensor` registry
  (Part A §1.2.9).

**Numerical reference.** To compare XPU output against a CUDA run of stock exllamav3:
- run CUDA with `EXL3_INT8_GEMV=0` (the default int8-activation GEMV adds about 0.9 % RMS noise on mul1
  decode, Part A §1.2.2);
- run CUDA with `EXL3_BC_GDN=0`, because the BC KDA path rounds differently: f_a/g_a stay fp32 and it uses
  a bf16 dt_bias (Part C).

---

## 4. Master op table (hot-path order, recommended XPU configuration)

"Calls/token" is per decode token at bsz 1, under the section 3 configuration. KDA = 34 layers, MLA = 11,
MoE = 42, dense = 3. "Implement" means the shim must provide a real implementation.

| # | op (ext symbol) | area | reached in | calls / decode token | primary Python call site(s) | XPU implementation | detail |
|---|---|---|---|---|---|---|---|
| 1 | `BC_LinearEXL3` (`__init__`, `run`, `run_alloc`) | EXL3 linear | load (ctor), every forward | ≈1270: KDA 2×34, MLA 4–5×11, dense 3×3, MoE 8×3×42 routed + shared (via #7), lm_head 1 | `modules/quant/exl3.py:86` (ctor), `:142` (`run_alloc`) | **Implement.** Trellis decode (mul1 LUT) + Hadamard-128 + matmul (`exl3_linear_ref`). Fast path: trellis-serve `torch.ops.exl3xpu_C.linear` (identical layout, K 3/4/5/6), cast to fp32 when fp32 output is requested. With `no_reconstruct` this also serves 8192-row prefill and full-sequence lm_head, so decode-and-matmul in column slices | A §1.2.1–1.3 |
| 2 | `hc_mix` | mHC | every forward | 90 | `modules/hyperconnections.py:152` | **Implement** (torch reference in Part E; later one Triton kernel). The branch at `:127` is not gated on ext availability | E (b) |
| 3 | `hc_apply` (in place) | mHC | every forward | 90 | `hyperconnections.py:190` | **Implement**: `x = post⊗y + combᵀx` (fp32) | E (b) |
| 4 | `rms_norm` | norm | every forward | 113 (2×45 block norms + final + 2×11 MLA `q_a`/`kv_a` norms) | `modules/rmsnorm.py:143,158`; `mla_attn.py:441,820` | **Implement** (fp32 math, fp16 clamp on read, eps inside sqrt; not `forward_torch`) | E (b) |
| 5 | `routing_ds3_nogroup` | MoE routing | every MoE layer | 42 | `modules/block_sparse_mlp_routing.py:214` (bsz 1), `:245` | **Implement**: `logits = fp16(x·gate)`, then the sigmoid / bias-select / normalise math | B (b) |
| 6 | `silu_mul` (`act_limit = 10`) | activation | every routed expert call; dense/shared above 8 rows | ≈336 (8×42 routed) | `block_sparse_mlp.py:1294`, `mlp.py:750,782` | **Implement**: `min(silu(g),10)·clamp(u,±10)` → fp16 | B (b) |
| 7 | `BC_GatedMLP` (`__init__`, `run_bszN`) | dense / shared MLP | rows ≤ 8 | 45 (3 dense + 42 shared) | `mlp.py:672` (ctor), `:744` | **Implement** as a Python wrapper over #1 and #6 | B (b) |
| 8 | `cuda_recurrent_gated_delta_rule` (channelwise/KDA) | KDA | decode; prefill chunks < 64 tokens | 34 | `modules/gated_delta_net_fn/gated_delta_rule.py:150` | **Implement** (torch, Part C reference; bf16 output rounding RZ on CUDA) | C (b) |
| 9 | `gated_rms_norm` | KDA o_norm | every KDA forward | 34 | `modules/gated_rmsnorm.py:120` | **Implement** (sigmoid gate, eps 1e-5) | C (b) |
| 10 | `hgemm` | FP16 linear (dtype-changing) | KDA `b/f_a/f_b/g_a/g_b` with fp32 out | ≈5×34 | `modules/quant/fp16.py:96` | **Implement**: `matmul` with fp32 accumulate into `c` | A §1.2.7 |
| 11 | `dsa_topk` | MLA indexer | every MLA layer once context > 2048 | 11 | `modules/mla_attn.py:696` | **Implement**: k largest finite fp16, lowest-index tie-break, emitted in ascending index order, −1 padded | D (b) |
| 12 | `fused_sampler` (+ 2 int attrs) | sampling | once per job per step | 1 per job | `generator/sampler/custom.py:248` | **Implement** modes 0–3 (greedy / Gumbel / min-P / top-K-top-P) | F (b) |
| 13 | `BC_LinearFP16` (`__init__`, `run`) | FP16 linear | load; `run` only from C++ BC classes | 0 at runtime | `modules/quant/fp16.py:54` | **Implement** (ctor + trivial `run`) | A §1.2.8 |
| 14 | `BC_GatedRMSNorm` (`__init__`, `run`) | KDA | load | 0 | `gated_rmsnorm.py:55` | **Implement** (holder) | C (b) |
| 15 | `det_quant_weight` | routing | first routing call per layer | 0 (lazy) | `block_sparse_mlp_routing.py:86` | no-op (its output feeds only the CUDA det GEMM) | B (b) |
| 16 | `hc_mix_num_chunks` | mHC | every `hc_mix` | 90 (host int) | `hyperconnections.py:130` | port the 3-line formula or return 1 | E (b) |
| 17 | `partial_strings_match` | stop strings (CPU) | per sampled token, only with stop strings | ≤ 1 per job | `generator/job.py:988` | pure Python | F (b) |
| 18 | `cache_rotate` | page-table defrag | idle only | 0 | `generator/pagetable.py:1025` | torch per-page copy loop, or `Generator(enable_defrag=False)` | F (b) |
| — | Triton (vendored fla `chunk_kda`: 7 kernels) | KDA prefill ≥ 64 tokens | prefill | — | `gated_delta_rule.py:114-137` | run on Intel Triton after the `utils.py:117` patch | C (d) |
| — | Triton conv `_causal_conv1d_update_slotted*` (3) | KDA conv | every KDA forward on XPU | 34 | `gated_delta_net_fn/conv1d.py:409` | run on Intel Triton after the `:245/:279` patches | C (d) |
| — | Triton MLA/DSA (`_mla_kv_update`, `_mla_plane_update`, `_mla_absorb`, `_mla_unfold`, `_mla_decode_split/combine`, `_mla_gather_tile`, `_mla_prefill_mha`, `_dsa_indexer(_fewq)`, `_dsa_attn_split/combine`, `_dsa_attn`) | MLA | every MLA forward | ≈6–8×11 | `mla_attn.py`, `attention_fn/mla_triton.py`, `dsa_triton.py` | run on Intel Triton. Lower `block_m` (MHA prefill) and `block_h` (one-shot DSA) explicitly | D (d) |

**Reached by the stock CUDA configuration but avoided by section 3** (no shim implementation needed):

| op(s) | avoided by | detail |
|---|---|---|
| `BC_GatedDeltaNetSplit` (`needs_configure`, `configure_slot_kda`, `run_bszN`), `kda_gate_op`, `gdn_ba_gemv`, `gdn_lowrank_gemv_f`, C++ `Graph` | `EXL3_BC_GDN=0` | C |
| `BC_MLAttention` (+ `set_indexer`, `configure_slot*`, `run`), `TritonKernel` (cubin + `cuLaunchKernel`) | `EXL3_BC_ATTN=0` | D |
| `BC_BlockSparseMLP` (`run_bszN`, `run_single_expert_dq`), `exl3_moe`, `exl3_moe_gather`, `exl3_moe_max_concurrency`, `exl3_mgemm`, `reconstruct_had_batch`, `hgemm_batched` | `no_reconstruct = True` | B |
| `reconstruct_had_slice`, `reconstruct`, `reconstruct_slice`, `had_r_128`, `hgemm_recon` (prefill above 144 rows) | `no_reconstruct = True`. These become internal helpers of the `BC_LinearEXL3` shim. Without the flag, set `EXL3_NO_FUSED_RECONSTRUCT=1` and implement the unfused four | A §1.2.4–1.2.7 |
| `cuda_causal_conv1d_update` | the unpatched `is_cuda` gate at `conv1d.py:386` (Triton conv instead) | C |
| `stloader_*`, `TensorLoadJob` | `load_method="python"` | E |
| `exl3_gemv_int8_max_k` | already skipped on non-CUDA devices (`model/config.py:74`) | A, E |

**Never reached for GLM-5.3-Flash text inference under this config.** Stub them, raising on call.
- Routing and MoE: `routing_std*`, `routing_sel_norm`, `routing_gemm_det`, `routing_ds3_nogroup_logits`,
  `blocksparse_mlp_routing`, `exl3_moe_coop`, `BC_MLP`, `moe_split_*`, `moe_unswizzle_trellis`,
  `exl3_moe_cpu_*`, `exl3_moe_flag_*`, `add_sigmoid_gate*`.
- Activations and elementwise: `gelu_mul`, `relu2_mul`, `relu_mul`, `silu_oai_mul`, `xielu`, `softcap`,
  `add`, `mul_sigmoid*`, `mul_softplus*`, `deinterleave_qg`.
- Other attention / sequence-mixing modules: `gated_delta_net_fused_op(_2)`, `BC_GatedDeltaNet`,
  `mamba2_*`, `cuda_recurrent_mamba2`, `BC_Mamba2`, `BC_Attention`, `bighead_attn*`,
  `dsa_topk_tile`/`dsa_topk_merge_tiles`, `dsv4_*`, `BC_DSV4*`.
- Hyper-connection variants and norms: `hc_head`, `gr_mix*`, `rms_norm_res_in`.
- Rope and the ngram/PLE/dflash ops: `rope`, `gen_mrope_pos_ids`, `ple_*`, `ngram_*`, `dflash2_*`.
- Cache helpers: `dspark_write_rows`, `paged_kv_cache_update`, `quant_cache_*`/`dequant_cache_*` (only
  with `-cq`), `count_match_tensor` (skipped for recurrent models).
- Rewind and drafting (need drafting): `batched_conv_rewind`, `batched_state_rewind`, `ConvRewindJob`,
  `StateRewindJob`, `BC_SAM`.
- Sampling: `gumbel_noise_*`, `gumbel_sample`, `argmax_sample`, `apply_rep_pens`,
  `apply_pres_freq_pens`, `dry_penalty`, `adaptivep_gumbel_noise_f32`, `apply_logit_bitmask`.
- Host memory and process groups: `cuda_host_*`, `pinned_cuda_view`, `cuda_device_get_attribute`,
  `pg_*` and the CPU-reduce jobs.
- Quantization and diagnostics: all quantize/pack/unpack ops, `decode`, `test_distribution`,
  `had_paley*`, `exl3_gemv`, `hgemm_f16acc*`, `g_get_cc`, `g_get_num_sms`, `exl3_gemm_shape_compat`,
  `exl3_gemm_num_kernel_shapes`, `histogram`, `count_inf_nan`, `det_math_test`.

Some grep hits for `ext.` are false positives, because the variables happen to be named `*ext`:
- `ext.encode`/`ext.endswith`/`ext.lower` in `generator/job.py` are methods of `held_text`;
- `ext.device`/`ext.shape`/`ext.index_*` in `modules/moe_batch_recon.py` are on a `*ext` tensor variable;
- `ext.get` in `modules/module.py` and `model/model_tp_fn.py`.

**Performance shape of this configuration**
- Decode is correct but slow:
  - about 1270 EXL3 linears per token;
  - one host sync per MoE layer (`expert_count.tolist()`, `block_sparse_mlp.py:1104`);
  - about 360 small HC launches.
- The obvious next steps, in order:
  - route `BC_LinearEXL3.run` to `exl3xpu_C.linear`;
  - replace the MoE decode loop with trellis-serve `moe_forward` (repack each expert into its blob; same
    tile bytes; the clamp convention differs by ≤ 4.5e-4, Part B);
  - fuse HC (`k_hcfuse.py` is the template, Part E);
  - optionally capture the decode step with `torch.xpu` graphs.

---


## 5. Per-op detail

Each part keeps its own op table (call sites, conditions, recommended path), per-op signatures and math, a step-by-step forward, and its CUDA-dependency table. The recommended XPU configuration in section 3 takes precedence where a part discusses alternatives.


---

## Part A. EXL3 / FP16 linear layers and the quant kernels

Paths are relative to `glm/exllamav3`. Python lives in `exllamav3/`; CUDA in
`exllamav3/exllamav3_ext/`.

### 1.0 Checkpoint facts (verified against the HF safetensors headers of rev 332ab457, all 15 shards)

These were read straight from the safetensors headers of `turboderp/GLM-5.3-Flash-exl3` (HTTP range
requests; nothing downloaded under `/mnt/sandisk`).

- **Codebook: every EXL3 tensor is `mul1`.** Each quantized linear ships `<key>.mul1` (I32 scalar,
  value `0x83DCD12D`), and the checkpoint has **zero `.mcg` tensors**. The value is only a flag: the
  multiplier is hard-coded in the kernels.
- **Every EXL3 tensor uses an integer bitrate**, and none use the half-integer `frac` format. The model
  was converted with exllamav3 1.4.4 (`quantization_config.version`). Fractional trellis support only
  arrived in commit 07b8a2e (after v1.5.0). So `frac.cu`, `dq8_half`, and the `HALF=true` instances are
  never reached.
- **No bias** on any linear. **No `su`/`sv` packed sign tensors**: only `suh`/`svh` (F16) are stored.
- The bitrate is `K = trellis.shape[-1] / 16`:

| tensor class | count | trellis shape `[in/16, out/16, 16K]` | K |
|---|---|---|---|
| routed experts gate/up `L.mlp.experts.E.{gate,up}_proj` | 12096 each | `[256, 128, 48]` | 3 |
| routed experts down | 12096 | `[128, 256, 48]` | 3 |
| dense MLP (layers 0-2) gate/up | 3 each | `[256, 768, 64]` | 4 |
| dense MLP down | 3 | `[768, 256, 64]` | 4 |
| shared expert gate/up/down | 42 each | (2048 wide) | 5 |
| KDA `qkv_proj` (34 layers) | 34 | `[256, 1536, 80]` (4096 -> 24576) | 5 |
| `o_proj` (all 45 layers) | 45 | KDA `[512,256,80]` (8192->4096); MLA `[1024,256,80]` (16384->4096) | 5 |
| MLA `q_a_proj` 4096->1536, `q_b_proj` 1536->16384, `kv_a_proj_with_mqa` 4096->512, `indexer.wq_b` 1536->4096 | 11 each | | 5 |
| `lm_head` 4096 -> 154880 | 1 | `[256, 9680, 96]` | 6 |

- The following tensors are **FP16/BF16/F32 rather than EXL3**, and are loaded through
  `Linear.load_fp16` -> `LinearFP16` or read raw by their module:
  - KDA: `b_proj` [64,4096], `f_a_proj` [128,4096], `f_b_proj` [8192,128], `g_a_proj` [128,4096],
    `g_b_proj` [8192,128] (all F16); `conv1d.weight` BF16 [24576,1,4]; `A_log` F32 [64];
    `dt_bias` F32 [8192]; `o_norm` BF16 [128].
  - MLA: `kv_b_proj.weight` F16 [32768,512]; `indexer.wk` F16 [128,4096];
    `indexer.weights_proj` F16 [32,4096]; `indexer.k_norm.{weight,bias}` F16 [128].
  - MoE router: `mlp.gate.weight` F16 [288,4096] and `e_score_correction_bias` F32 [288].
  - Hyperconnections: `hc_{attn,ffn}_{base,fn,scale}` F32 ([24], [24,16384], [3]).
  - Norms: BF16. `embed_tokens` BF16 [154880,4096].
  - The k-pool aux tensors (`index_kpool_compress_{ape,gate}`) live in `kpool_aux.safetensors`.
- **The checkpoint has no MTP layer** (`layers.45.*` is absent, even though
  `num_nextn_predict_layers = 1`). MTP would need a `convert_mtp.py` output placed next to the model.
  The model card also ships 1007 `model.visual.*` tensors, which text inference does not use.

### 1.1 Op table (this part)

Hot-path rank: 1 is hottest. "BC path" means `LinearEXL3.forward` -> `BC_LinearEXL3.run_alloc`, or a
C++ BC class of another slice (BC_GatedMLP, BC_BlockSparseMLP, BC_MLAttention, BC_GatedDeltaNet)
calling `BC_LinearEXL3::run_gr` internally.

| op | category | reached in | rank | Python call sites | condition / avoidable? | recommended XPU path |
|---|---|---|---|---|---|---|
| `BC_LinearEXL3` (ctor, `run`, `run_alloc`) | bound class | load (ctor); decode and small prefill (run) | 1 | ctor `modules/quant/exl3.py:86`, `modules/linear.py:474` (pin only); `run_alloc` `modules/quant/exl3.py:142`; held by other BC classes via `.inner.bc` (`modules/mlp.py:223-224`, `modules/gated_delta_net.py:647-658,704-717,778-794`, block_sparse_mlp, mla) | `rows = x.numel()/x.shape[-1] <= 144` (`AUTO_RECONSTRUCT_THRESHOLD`, `exl3.py:10,140`), or `infer_params.no_reconstruct`. **Not avoidable**: the decode path for every quantized linear. | Python class holding trellis/suh/svh/K/mcg/mul1/bias/xh (other parts' BC shims read these fields). `run` = `exl3_linear_ref` below; fast path = trellis-serve `torch.ops.exl3xpu_C.linear`. |
| `exl3_gemm` (C++-internal, not called from Python) | EXL3 GEMM/GEMV | decode, prefill <= 144 rows | 1 | only via `BC_LinearEXL3::run_gr` (`libtorch/linear.cpp:38,44`) and other BC classes | Internal dispatch: int8-act GEMV (mul1, m <= 2, K <= 5 on non-Hopper) -> QTIP GEMV (K 2..4) -> cooperative tensor-core GEMM. All compute the same math. | One function `exl3_linear_ref(x, ...)`. Fast: exl3xpu `linear` / `exl3_gemm_small` (layout identical, see 1.4). |
| trellis decode (`dq*` in `quant/exl3_dq.cuh` + `decode_3inst<2>` in `quant/codebook.cuh`) | device helper | everywhere | 1 | - | - | 64K-entry fp16 LUT plus a vectorized window extract (1.3). |
| `reconstruct_had_slice` | dequant to original-basis W | prefill, rows >= 1024 | 2 | `modules/quant/exl3.py:200,212` | `rows > 144` and `rows >= 1024`, `in%128==0`, `out%128==0`, and `EXL3_NO_FUSED_RECONSTRUCT` unset (`exl3.py:14,181-189`). Avoid with `EXL3_NO_FUSED_RECONSTRUCT=1` (falls to reconstruct + had_r_128). | Force `EXL3_NO_FUSED_RECONSTRUCT=1` at first; or implement as `diag(suh) H W_hat H diag(svh)`. |
| `hgemm_recon` | fp16 GEMM | prefill rows > 144 | 2 | `exl3.py:203,216` | always on the reconstruct path | `torch.matmul` with fp32 accumulation into `c` (strided column slice allowed) |
| `had_r_128` | 128-point Hadamard + scales | prefill 145..1023 rows (unfused), also other parts | 3 | `exl3.py:195` (pre, suh), `exl3.py:219` (post, svh); `model/moe_cpu_host.py` (CPU offload, n/a) | unfused reconstruct path only | torch: `(x.view(-1,128) [* suh]) @ H128 / sqrt(128) [* svh]` (exact rounding in 1.2.4) |
| `reconstruct` | dequant to Hadamard-domain W_hat | prefill 145..1023 rows; `get_weight_tensor` | 3 | `exl3.py:229` | unfused path, `out_features <= 32768` | LUT decode (1.3) |
| `reconstruct_slice` | same, column slice | prefill 145..1023 rows, lm_head (N > 32768) | 3 | `exl3.py:231` | unfused path, `out_features > 32768` (lm_head only) | LUT decode on a tile-column range |
| `BC_LinearFP16` (ctor, `run`) | bound class | load; any FP16 linear | 4 | ctor `modules/quant/fp16.py:54`, `modules/linear.py:470`; `run` called from other BC classes (BlockSparseMLP's `shared_gate`) | fp16 tensors listed in 1.0 | `y = x @ W (+bias)` in torch |
| `hgemm` | fp16 GEMM (cuBLAS) | any FP16 linear whose out dtype != in dtype | 4 | `modules/quant/fp16.py:96` (plus `modules/conv.py`, `moe_cpu_host.py`, not on path) | `LinearFP16.forward` uses `torch.matmul` when `dtype == x.dtype`, else `ext.hgemm` | `torch.matmul(x.float(), w.float())` or `torch.mm(...).to(out)`; fp32 accumulate |
| `exl3_gemv_int8_max_k` | query | load (mgemm fusion heuristic) | 5 | `model/config.py:76` | Only when `device.type == "cuda"` (`config.py:74`). **Never reached on XPU**: `K_thr` stays at env `EXL3_MGEMM_K_THRESHOLD` (default 6). | Do not implement; or return 5. |
| `exl3_mgemm` | multi-matrix EXL3 GEMM | decode (fused gate/up, MoE, qkv) | 1 (owned by Parts B–D) | `modules/mlp.py:761`, `modules/attn.py`, `modules/gated_delta_net.py`, `modules/block_sparse_mlp.py`, `modules/sliding_attn.py`, `modules/dsv4.py` | see Part B; math summarized in 1.2.9 | loop of `exl3_linear_ref` over a pointer-to-tensor registry |
| `softcap` | elementwise | never (GLM lm_head `softcap=0`) | - | `modules/linear.py:634` | `Linear.softcap != 0` | `y = tanh(x/s)*s` if ever needed |
| `count_inf_nan` | stats | never (conversion capture only) | - | `modules/linear.py:569` | `"capture" in params` | skip |
| `pinned_cuda_view` | host alias | never (vision tower pinning only) | - | `modules/linear.py:466` | `EXL3_VISION_PINNED=1` and the vision component | skip |
| `reconstruct_batch`, `reconstruct_had_batch`, `had_r_128_batch`, `hgemm_batched` | batched expert recon | MoE prefill (`modules/moe_batch_recon.py`), owned by Part B | - | `moe_batch_recon.py` | Part B decides | per-expert loop of the 1.2 ops |
| `g_get_cc`, `g_get_num_sms`, `exl3_gemm_shape_compat`, `exl3_gemm_num_kernel_shapes`, `exl3_gemv`, `hgemm_f16acc`, `hgemm_f16acc_status`, `had_paley`, `had_paley2` | queries / test entries | not reached from inference Python | - | none outside tests/conversion; `had_paley*` from `util/hadamard.py:67,102` for non-power-of-2 sizes only | H128 is Sylvester, built from `hadamard_1.txt` | stub (raise) |
| `quantize_tiles*`, `pack_trellis*`, `unpack_trellis*`, `pack_signs`, `decode`, `test_distribution` | conversion | never | - | `modules/quant/exl3_lib/quantize.py` (imported at load, but not called) | conversion only | stub |

#### Path selection inside `LinearEXL3.forward` (`modules/quant/exl3.py:119-144`)

1. `params["ovr"]` is an override hook (conversion); `params["reconstruct"]` is set only by conversion.
2. `rows = x.numel() // x.shape[-1]`.
   - If `rows <= 144` or `no_reconstruct`, it calls `self.bc.run_alloc(x, out_features, dtype == float)`,
     with `dtype = out_dtype or self.default_out_dtype` (`Linear.out_dtype` or fp16).
   - Otherwise it calls `reconstruct_hgemm`:
     - `use_fused = in%128==0 and out%128==0 and not EXL3_NO_FUSED_RECONSTRUCT and rows >= 1024`.
       This is always true for GLM chunks of 1024 to 8192 rows.
     - Fused: `reconstruct_had_slice` (`w` = original-basis W) followed by `hgemm_recon(x, w, y)`.
     - Unfused: `had_r_128(x -> xh, pre=suh)`, then `reconstruct[_slice]` (W_hat), then
       `hgemm_recon(xh, W_hat, y)`, then `had_r_128(y, y, post=svh)` in place.
     - `out_features > 32768` (only `lm_head`) is processed in 32768-column slices through a shared
       scratch `w_`.
3. Bias add (none in GLM).
4. `x` must be contiguous (asserted, `exl3.py:135`).

With `-chunk_size 8192` and batch <= 8, decode always takes the BC path. Prefill chunks take the fused
reconstruct path. A final short tail chunk of 145 to 1023 rows takes the unfused path, and <= 144 takes
the BC path. `model.load` (autosplit dummy forwards) and `model.warmup` (`model/model.py:190`) exercise
both paths at load.

**lm_head** (`architecture/glm5_next.py:334-343`, K=6, `caps={"logits_output": True}`, `out_dtype=None`):
- It returns **fp16 logits**.
- The generator sets `params["last_tokens_only"] = 1` (`generator/job.py:1396`), and
  `model/model_ls.py:326-327` slices x to the last token before the head. So the head always runs at
  rows <= batch, on the BC path.
- Only a full-sequence forward reconstructs it in 5 slices of 32768 columns. Examples are `serve.py`
  `teacher_forced` (`model.forward(ids, {"attn_mode": "flash_attn_nc"})`) and `perplexity`-style calls.

`select_hq_bits` (GLM uses 1 or 2) is **conversion-only**: `conversion/allocation.py` raises the bitrate
floor. It has no runtime effect.

`InferParams.use_mgemm` (`model/config.py:59-77`) decides fused-mgemm against separate GEMMs in other
slices. On XPU the `exl3_gemv_int8_max_k` probe is skipped, so with defaults:
- **mul1 tensors stay fused when `K >= 6` or `out_features < 8192`**;
- non-mul1 tensors are always fused.

### 1.2 Per-op detail

#### 1.2.1 `BC_LinearEXL3`

Binding: `exllamav3_ext/libtorch/linear_bc.h:13-35`. Struct: `libtorch/linear.h:64-99`. Implementation:
`libtorch/linear.cpp:34-71`.

- **ctor** `BC_LinearEXL3(trellis: int16 [k/16, n/16, 16K], suh: fp16 [k], svh: fp16 [n], K: float,
  bias: fp16 [n] | None, mcg: bool, mul1: bool, xh: Tensor)`.
  - `xh` is a scratch buffer for the transformed input at m == 1. Python passes
    `g_tensor_cache.get(device, (1, in_features), out_dtype)` (`exl3.py:84,94`). That buffer has dtype
    `Linear.out_dtype`, possibly `float32` or `None` (the torch default, float32). The kernel treats it
    as raw fp16 scratch, so any buffer with >= 2*k bytes works.
  - The object only stores the tensors. Other C++ BC classes read the `trellis/suh/svh/K/mcg/mul1`
    fields directly to build `exl3_mgemm` calls. **A Python shim must expose those attributes with
    these names.**
- **`run(x, y) -> None`**: `x` fp16 `[..., k]` (2D in practice), `y` preallocated fp16 or fp32
  `[..., n]` (out dtype selected by `y.dtype`).
  - If `x.numel() == x.size(-1)` (one row), it calls `exl3_gemm_gr(x, trellis, y, suh, xh, svh, -1, mcg,
    mul1, 0, graph)` with `A_had = self.xh`.
  - Otherwise it allocates `A_had = empty_like(x)` and calls `exl3_gemm` (a graph with bsz > 1 is a
    TORCH_CHECK error).
  - Then it adds `bias` in place if present (`add_gr`).
- **`run_alloc(x, out_features: int, output_fp32: bool) -> Tensor`**: allocates
  `y = empty(x.shape[:-1] + (out_features,), fp32 if output_fp32 else fp16)`. It returns `y`
  immediately if `out_features == 0`, otherwise calls `run(x.view(-1,k), y.view(-1,n))` and returns
  `y`.
- `run_gr(x, y, Graph*)` is not bound. It is used only by other BC classes' graph capture.

#### 1.2.2 `exl3_gemm` (internal; also bound but unused from Python)

Binding: `bindings.cpp:162` (`exl3_gemm(A, B, C, suh, A_had, svh, force_shape_idx, mcg, mul1,
force_num_sms) -> int`). Host: `quant/exl3_gemm.cu:114-346`. Kernel: `quant/exl3_gemm_kernel.cuh:9-55`.
Inner loop: `quant/exl3_gemm_inner.cuh`.

Arguments:
- `A` fp16 `[..., k]`, contiguous. `size_m = prod(A.shape[:-1])`.
- `B` int16 `[k/16, n/16, 16K]` (trellis), with `size_n = B.size(1)*16`.
- `C` fp16 or fp32 `[..., n]` (written).
- `suh` fp16 `[k]`, `A_had` fp16 scratch with `>= m*k` elements, `svh` fp16 `[n]`.
- Checks: `A.size(-1) == B.size(0)*16`, `C.size(-1) == B.size(1)*16`, A fp16, B int16. K is read back
  from `B.size(2)`: `K = B.size(2)/16`, and `B.size(2) % 16 == 8` means half-integer K.

**Math** (identical across all three internal kernels, up to accumulation order):

```
A_had[r, :] = fp16( H128_blockwise( fp16(A[r,:] * suh) ) / sqrt(128) )     # per 128-block of k
acc[r, :]   = A_had[r,:] @ W_hat            (fp16 x fp16 MMA, fp32 accumulate*)
C[r, :]     = H128_blockwise(acc[r,:]) / sqrt(128) * svh                   # per 128-block of n
```

- `W_hat` is the decoded trellis matrix `[k, n]` (1.3).
- `H128` is the natural-order Sylvester matrix, `H[i,j] = (-1)^popcount(i&j)` (unnormalized, symmetric).
- `suh` and `svh` already include the sign flips and channel scales.
- The input transform runs in-kernel (`exl3_gemm_kernel.cuh:13-28`, `had_hf_r_128_inner<pre=true>`):
  `fp16(x*suh)` is `__hmul2` (fp16 rounding), the butterfly runs in fp32, the result is multiplied by
  `1/sqrt(128)` in fp32 and rounded to fp16.
- The output transform (`exl3_gemm_inner.cuh:595-620`) works on the fp32 tile sum:
  - fp32 out (`had_ff`): butterfly fp32, `*1/sqrt(128)` fp32, `*float(svh)` fp32.
  - fp16 out (`had_fh`): butterfly fp32, `*r` fp32, round to fp16, then `__hmul2` by `svh` (fp16
    rounding).
- \*Accumulation details:
  - sm_86 only (`EXL3_GEMM_H_ACC`, `exl3_gemm_inner.cuh:16`) accumulates MMA in fp16 within a k-slice,
    then folds into fp32.
  - k-split partial sums go through global memory **in C's dtype**. For fp16 outputs the partials are
    rounded to fp16 (`read_sum_gl`/`write_sum_gl`).
  - This is numerical noise; the reference should accumulate in fp32.

**Internal dispatch** (all CUDA-specific, none needs porting):

1. If `mul1 && exl3_gemv_int8_enabled()` (env `EXL3_INT8_GEMV`, **default 2 = on**), it tries
   `exl3_gemv_int8` (`quant/exl3_gemv_int8.cu:236-377`). Requirements: suh/A_had/svh present,
   `n%256==0`, `k%128==0`, `K <= exl3_gemv_int8_max_k(dev)` (5, or 6 on Hopper/Blackwell; env
   `EXL3_INT8_GEMV_MAX_K`), and `m <= 2` (plain mode) or `m == 1`.
   - It **quantizes the transformed activations to int8**: `q_s = max(|a|, 1e-8)/127` per k-slice (or
     per row), `i = clamp(rint(a/q_s), -127, 127)`.
   - Mode 1 adds a residual int8 with `q2 = q/254`.
   - It computes `y = k_inv*(q*acc1 + q2*acc2) + (1024*k_inv + k_bias)*(q*sum1 + q2*sum2)`, then the
     output Hadamard and svh.
   - **This changes numerics vs fp16** (about 0.9% output RMS in mode 2). It is taken at decode for
     GLM's K=3/4/5 mul1 tensors.
   - To compare XPU output against a CUDA reference, run CUDA with `EXL3_INT8_GEMV=0`.
2. `exl3_gemv_try_launch` (`quant/exl3_gemv.cu:92`): QTIP-style GEMV for `K in 2..4` (`cb != 0` unless
   K == 4), `m <= EXL3_GEMV_MAX_M`, k/n % 128. Env `EXL3_GEMV` mode (`exl3_gemv_env_mode`).
3. Otherwise a cooperative tensor-core GEMM. It is autotuned (`CoopKernelAutotuner`, keyed by shape,
   K, cc, SM count, cb), uses `cudaLaunchCooperativeKernel`, a `grid.sync`, and a global `locks`
   buffer from `DevCtx`.

Return value: kernel tag (ignored by callers).

#### 1.2.3 Trellis decode `decode_3inst<cb>` / `dq*`

Sources: `quant/codebook.cuh:56-123` and `quant/exl3_dq.cuh`. See 1.3 for the full layout. Codebooks:

- cb=2 `mul1` (**GLM**):
  - `x = state * 0x83DCD12D (mod 2^32)`.
  - `s = sum of the 4 unsigned bytes of x` (`__dp4a(x, 0x01010101, 0x6400)`).
  - `h = fp16 bits (0x6400 + s) = 1024 + s` (exact).
  - `w = fp16(h * 0.00676727294921875 + (-10.3828125))` with one rounding (`__hfma`; constants are the
    fp16 bit patterns `0x1eee` and `0xc931`).
  - The fp32 computation `(1024+s)*k_inv + k_bias` is exact (22-bit product, |sum| < 4 at lsb 2^-18),
    so `float32 -> fp16` round-to-nearest-even gives a bit-exact LUT.
  - Values lie in [-3.453, 3.348], with mean about 0 and std about 1.0.
- cb=1 `mcg` (not used by GLM): `x = state * 0xCBAC1FED`; `x = (x & 0x8FFF8FFF) ^ 0x3B603B60` (lop3
  0x6a); `w = fp16(lo16) + fp16(hi16)` (`__hadd`).
- cb=0 `3INST` (not used): `x = state*89226354 + 64248484`, then the same lop3 and add.

#### 1.2.4 `had_r_128`

Binding: `bindings.cpp:160` (`had_r_128(input, output, pre_scale, post_scale, scale)`). Host:
`quant/hadamard.cu:121-~210`. Inner: `quant/hadamard_inner.cuh:95-212`.

- Arguments: `input` and `output` must have the same shape, 2D `[rows, cols]`, with `cols % 128 == 0`.
  Both are fp16 (`had_hf`) or both fp32 (`had_ff`). In-place is allowed (`output is input`).
- `pre_scale` and `post_scale` are optional fp16 `[cols]`. **If both are given, only `pre_scale` is
  applied** (the if/else chain). Callers never pass both.
- `scale` is a float; the effective factor is `r = scale/sqrt(128)` (`0.088388347648f`).
- Math per 128-block: `v = x[blk] (* pre)`; `out[blk] = (H128 @ v) * r (* post)`. H128 is symmetric,
  so `x @ H == H @ x`.
- fp16 path rounding:
  - `pre`: `__hmul2` in fp16.
  - Butterfly in fp32 (4-point in registers, then 5 xor-shuffle stages; lane bit set means
    `v = partner - v`).
  - `*r` in fp32, then round to fp16.
  - `post`: `__hmul2` in fp16.
- fp32 path: scales are converted to float, with no intermediate rounding.
- Grid `(rows, cols/128)`, 32 threads, each thread holding 4 consecutive elements.

#### 1.2.5 `reconstruct` / `reconstruct_slice`

Binding: `bindings.cpp:155,159` (`reconstruct(unpacked, packed, K, mcg, mul1)`,
`reconstruct_slice(unpacked, packed, K, mcg, mul1, n_offset)`). Source: `quant/reconstruct.cu:139-200`.

- `unpacked`: fp16 `[k, n_slice]`, written row-major.
- `packed`: trellis int16 `[k/16, n_total/16, 16K]`.
- `K` float. `mcg`/`mul1` bool select the codebook: `cbi = K-1 + (mcg ? 8 : mul1 ? 16 : 0)`, or
  `+24` for half-integer K.
- Checks: `unpacked.size(0) == packed.size(0)*16`, `n_slice % 128 == 0`, `n_offset % 128 == 0`, and
  `n_offset + n_slice <= n_total`.
- Output: `unpacked[r, c] = W_hat[r, n_offset + c]` (Hadamard domain, no suh/svh applied).
- `reconstruct` is `reconstruct_slice` with `n_offset = 0` (the Python wrapper only calls it for the
  full width).
- Grid: `(n_slice/128, k/16)`, 256 threads. Each warp decodes one 16x16 tile, and a block covers
  16 rows x 128 columns.

#### 1.2.6 `reconstruct_had_slice`

Binding: `bindings.cpp:156` (`reconstruct_had_slice(unpacked, packed, suh, svh, K, mcg, mul1, n_offset)`).
Source: `quant/reconstruct.cu:202-~360`. Python passes `svh[n_start:]` **and** `n_offset = n_start`
(`exl3.py:212-213`): svh is pre-sliced, the trellis is offset in-kernel.

- Output: original-basis weights `W[k, n_slice] = diag(suh) · Hblk · W_hat[:, slice] · Hblk ·
  diag(svh_slice)`, where `Hblk` is block-diagonal `H128/sqrt(128)`.
- The kernel works on 128x128 tiles in shared memory. The column butterfly runs **in fp16** (`__hadd2`)
  with `r` pre-multiplied per stage. The row butterfly's first 4-point stage is fp32, then fp16 shuffles.
  It finally multiplies `*suh[row]` and then `*svh[col]` in fp16.
- This path is noticeably less precise than the unfused one, but within quantization noise.
- `y = x @ W` is then exactly the linear's output (no had calls needed). It requires `k%128 == 0` and
  `n%128 == 0`.

#### 1.2.7 `hgemm_recon` / `hgemm`

Binding: `bindings.cpp:191,193` (`hgemm(a, b, c)`, `hgemm_recon(a, b, c)`). Sources: `hgemm.cu:18-107` and
`hgemm_f16acc.cu:527-531`.

- `a`: fp16 `[..., k]`. `b`: fp16 `[k, n]` (2D). `c`: fp16 or fp32 `[..., n]` (written). The last dim of
  `c` must be contiguous; **the row stride of `c` may be larger than n** (Python writes into
  `y_[:, n_start:n_end]` column slices).
- Math: `c = a @ b`, computed in fp32 (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, alpha=1, beta=0,
  overwrite), then rounded to c's dtype.
- `hgemm_recon` first tries `hgemm_f16acc_try` (fp16-accumulator tensor-core kernel; GeForce only; env
  `EXL3_HGEMM_F16ACC`), else calls `hgemm`. The semantics are the same.
- XPU: `torch.matmul(a, b, out=c_view)` works for an fp16 `c` (oneDNN accumulates in fp32). For fp32
  `c`, use `c.copy_(a.float() @ b.float())` or `torch.mm(a, b).float()`; the latter rounds through fp16
  first, a minor deviation.

#### 1.2.8 `BC_LinearFP16`

Binding: `libtorch/linear_bc.h:1-10`. Struct: `libtorch/linear.h:45-62`. Implementation:
`libtorch/linear.cpp:11-25`.

- ctor `BC_LinearFP16(weight: fp16 [in, out] (already transposed at load), bias: fp16 [out] | None)`.
- `run(x, y)`: if `x.dtype == y.dtype` (and no graph), `at::matmul_out(y, x, weight)`; otherwise
  `hgemm(x, weight, y)` (fp32 out). Then `y += bias`.

`LinearFP16.forward` (`modules/quant/fp16.py:69-100`) does the same in Python: `torch.matmul` when the
out dtype equals the x dtype, else `ext.hgemm`.

Load notes:
- `Linear.load_fp16` transposes checkpoint `[out, in]` to `[in, out]` (`transposed_load=True`) and pads
  dims to multiples of 128 (`pad_to=128`).
- KDA's `b_proj` is 64-wide, which pads to 128; `f_b_proj` has in=128.
- MoE `gate.weight` (288) pads to 384, with `trim_padded_out` per the module.
- Float biases are converted to fp16. FP8/E8M0 dequant paths are not used by GLM.

#### 1.2.9 `exl3_mgemm` (core math only; call sites belong to Parts B–D)

Binding: `bindings.cpp:185-190`. Host: `quant/exl3_gemm.cu:351-730`. Kernel:
`quant/exl3_gemm_kernel.cuh:65-343`.

Arguments:
- `A` fp16 `[a_batches, m, k]`. `B`, `suh`, `svh` are **int64 tensors of device addresses**, one entry
  per matrix (trellis ptr, suh ptr, svh ptr).
- `C` fp16 or fp32 `[c_batches, m, n]`. `A_had` is scratch for `bszm*m*k` elements.
- `indices` int64 `[*, num]` (optional), `weights` fp16 (optional), `K` float, `force_shape_idx`.
- `mcg`, `mul1` are uint32 in C++ (non-zero means true).
- `min_index`, `max_index`, `force_num_sms`, `num_tokens`, plus the optional `size_n_list`, `c_ptrs`,
  `n_stride_list`, `had_src_list`, `num_had_src`.

Math: for each active slot j:
- `q = indices[j]` (or j); slots with `q < 0` are skipped.
- `C[j] = exl3_linear(A[j or 0], B[q], suh[q], svh[q]) * (weights[j] if weights)`.
- If `weights` is given, `C[t*m*n..]` receives the sum over the `stride = bszm/num_tokens` slots of each
  token. Accumulation is fp32, or **fp16 `__hadd` for fp16 C**.
- `min_index`/`max_index` filter and rebase the indices.

**Shim implication:** the pointer tables cannot be dereferenced in torch. They are built in
`modules/multilinear.py` and `modules/block_sparse_mlp.py:637-657`. The shim needs a
`data_ptr() -> Tensor` registry, filled in `BC_LinearEXL3.__init__` and when `LinearEXL3` tensors are
created, or those Python sites must be patched to keep tensor lists.

#### 1.2.10 Minor ops

- `softcap(x, y, s)` (`softcap.cu`): `y = tanh(x/s)*s` (fp32 math, fp16 or fp32 storage). Not used by GLM.
- `count_inf_nan(x, counter)`: conversion only.
- `pinned_cuda_view(t, device)` (`cuda_host.cpp:75-92`): `cudaHostGetDevicePointer` alias of a pinned
  CPU tensor. Vision only. On XPU, use USM host memory, or skip it.
- `exl3_gemv_int8_max_k(device) -> int`: `EXL3_INT8_GEMV_MAX_K` env, else 6 on Hopper/Blackwell, else 5.

### 1.3 EXL3 data layout (exact)

**Tensors per linear `<key>`:**
- `trellis` int16 `[in/16, out/16, 16*K]`. Tile `(tk, tn)` covers W rows (input dim)
  `16tk..16tk+15` and columns (output dim) `16tn..16tn+15`.
- `suh` fp16 `[in]`, `svh` fp16 `[out]`.
- `mul1` I32 scalar flag (or `mcg`), optional `bias` fp16 `[out]`.
- Old `su`/`sv` int16 sign bitfields (`LinearEXL3.unpack_bf`) are absent here.

**Dequantized weight** (`modules/quant/exl3.py:239-249`; quantizer `exl3_lib/quantize.py:1641-1645`):

```
W[in, out] = diag(suh) · Hblk · W_hat · Hblk · diag(svh)        Hblk = blockdiag(H128 / sqrt(128))
y = x @ W = ( Hblk( (x ∘ suh) ) @ W_hat ) Hblk ∘ svh            (row-vector x; H symmetric)
```

H128 is Sylvester from `hadamard_1.txt` (`util/hadamard.py:34-42,107-131`). There is no Paley step at
128, and no random permutation. The 1/sqrt(128) normalization applies on each side.

**Bit stream of a tile:**
- A tile is 256 weights at K bits each, giving `256K` bits = `16K` int16 = `8K` uint32.
- Reading the int16 array as little-endian uint32 (`word[i] = u16[2i] | u16[2i+1] << 16`, which is how
  every kernel loads it) gives a **MSB-first** bit stream. Bit `b` is bit `31 - b%32` of `word[b//32]`.
- The packer (`quant/pack.cu:9-57`) writes 16-bit words MSB-first and then `SWAP16`s each uint32 pair.
- Position `t` (0..255) owns the K bits ending at stream bit `(t+1)K`.
- Its **16-bit state** is the window of stream bits `[(t+1)K-16, (t+1)K)`, read MSB-first and
  **circular** within the tile (tail-biting). For t=0 the window starts in the last word.
- Decoded value: `w = codebook(state)` (1.2.3).

**Position to (row, col) inside the 16x16 tile** (tensor-core fragment order; `tensor_core_perm`,
`exl3_lib/quantize.py:22-44`; `reconstruct.cu:42-80`):

```
row (input/k)  = 8*((t>>1)&1) + 2*((t>>3)&3) + (t&1)
col (output/n) = (t>>5)       + 8*((t>>2)&1)
```

**Torch reference** (self-checked in numpy against the packer and `tensor_core_perm`):

```python
MUL1 = 0x83DCD12D
def mul1_lut(dev):                                  # [65536] fp16, bit-exact with decode_3inst<2>
    st = torch.arange(65536, dtype=torch.int64)
    x = (st * MUL1) & 0xFFFFFFFF
    s = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    return ((1024 + s).float() * 0.00676727294921875 - 10.3828125).half().to(dev)  # exact in fp32, 1 rounding

def _pos_tables(K, dev):
    t = torch.arange(256, device=dev)
    e = (t + 1) * K; ih = (e - 1) // 32; sh = 32 * (ih + 1) - e; il = (ih - 1) % (8 * K)
    row = 8 * ((t >> 1) & 1) + 2 * ((t >> 3) & 3) + (t & 1)
    col = (t >> 5) + 8 * ((t >> 2) & 1)
    return il, ih, sh, row, col

def reconstruct_w_hat(trellis, K, lut, n0=0, n1=None):   # == ext.reconstruct_slice
    tk, tn, _ = trellis.shape
    tr = trellis[:, n0 // 16 : (n1 or tn * 16) // 16]
    words = tr.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF          # [tk, tn', 8K]
    il, ih, sh, row, col = _pos_tables(K, trellis.device)
    st = (((words[..., il] << 32) | words[..., ih]) >> sh) & 0xFFFF                # [tk, tn', 256]
    tiles = torch.empty(*st.shape[:2], 16, 16, dtype=torch.half, device=trellis.device)
    tiles[..., row, col] = lut[st]
    return tiles.permute(0, 2, 1, 3).reshape(tk * 16, -1)                          # [in, n']

def had128(x, pre=None, post=None):                 # == ext.had_r_128 (fp16 in/out)
    H = sylvester128(x.device, torch.float32) / math.sqrt(128)
    v = x if pre is None else x * pre                # fp16 multiply (rounded)
    y = (v.float().view(-1, 128) @ H).view(x.shape).half()
    return y if post is None else y * post

def exl3_linear_ref(x, trellis, suh, svh, K, lut, out_dtype):
    xh = had128(x.reshape(-1, x.shape[-1]), pre=suh)
    acc = xh.float() @ reconstruct_w_hat(trellis, K, lut).float()     # or chunk over n
    y = (acc.view(-1, 128) @ (sylvester128(x.device, torch.float32) / math.sqrt(128))).view(acc.shape)
    y = y * svh.float() if out_dtype == torch.float else (y.half() * svh)
    return y.view(*x.shape[:-1], -1).to(out_dtype)
```

Memory note: W_hat for one expert (4096x2048 fp16) is 16 MB. lm_head is 1.27 GB fp16, so
reconstruct it in column slices as the CUDA path does (32768 columns).

### 1.4 Comparison with the existing XPU kernels (`~/work/trellis-b70/trellis-serve/xpu`)

The layout is **identical**, so the kernels can be reused directly:
- `csrc/exl3_esimd.h:4-8` documents the same MSB-first stream and the same
  "window ending at (t+1)K, circular".
- It uses the same `row = 8*((t>>1)&1) + 2*((t>>3)&3) + (t&1)` and `col = (t>>5) + 8*((t>>2)&1)`.
- It reads the int16 trellis tensor as uint32 words.
- `decode_cb_h<2>` (`exl3_esimd.h:40-80`) uses the same `*0x83DCD12D`, `dp4a(0x6400, x, 0x01010101)`,
  and fp16 fma `0.00676727294921875 / -10.3828125`, and claims bit-exactness.
- The input Hadamard also rounds `x*suh` to fp16 first (`exl3_esimd.h:300`).

Differences and adapter work:
- **Op signature.** The op is `torch.ops.exl3xpu_C.linear(x, trellis, suh, svh, shard_of_nb,
  group_bounds, K, cb, small_m_max, slice_n)` (`csrc/exl3_ops.sycl:651-757`).
  - `suh` is `[S, k]`, one row per fused shard: pass `suh.view(1, -1)`.
  - `shard_of_nb` is int32 `[n/128]` of zeros. `group_bounds = [0, n]`. `cb = 2`.
- **Coverage.** `exl3_supported` covers `cb==2, K in {3,4,5,6}` (`exl3_ops.sycl:767-774`), exactly
  GLM's set. `exl3_reconstruct` also has K=2 and some cb 0/1.
- **Output dtype** is always `x.dtype` (fp16 or bf16). exllamav3 often requests **fp32 outputs**
  (`out_dtype=torch.float` on o_proj, down, and other parts). Cast afterwards, or extend the op; there
  is a small precision difference, since CUDA never rounds the fp32 output to fp16.
- **Prefill.** Above `small_m_max`: `had_in_rm` -> `exl3_reconstruct` slices -> `at::matmul_out` (fp16
  y) -> `exl3_had_out_h`. This equals exllamav3's *unfused* path; there is no fused
  `reconstruct_had_slice` equivalent, and none is needed.
- **Optional modes.**
  - `exl3_set_int8(1)` enables a W8A8 prefill (int8 weights and activations, K=4/6 only), off by default.
  - `exl3_set_fused` covers K=4/6 small-m fused decode.
  - Triton fallbacks exist in `exl3xpu/triton_kernels.py` (`reconstruct`, `had_in`, `had_out`,
    `gemm_small`).
- **Not covered by the XPU kernels:** `exl3_mgemm` (multi-matrix, pointer tables, weighted reduce). The
  grouped-MoE `moe_forward` in `csrc/exl3_moe.sycl` uses its own pointer-table/blob format (Part B).

### 1.5 Non-ext CUDA dependencies in this part

| file:line | what | XPU mapping |
|---|---|---|
| `modules/linear.py:458` | `if device.type != "cuda": return` in `pin_linears` | Already a no-op on XPU; vision only. |
| `modules/linear.py:464` | `torch.empty(..., pin_memory=True)` | vision only; skip |
| `modules/linear.py:466` | `ext.pinned_cuda_view` | vision only; skip |
| `modules/quant/fp16.py:92` | `weight.copy_(pinned, non_blocking=True)` (pinned store) | only when pinned; skip |
| `modules/quant/exl3.py:304-375`, `modules/quant/fp16.py:151-197` | `consumer.recv(..., cuda=True)` | TP only, not reached |
| `model/config.py:72-76` | `device.type == "cuda"` gate around `exl3_gemv_int8_max_k` | Naturally skipped on XPU, so `K_thr = 6`. |
| `model/config.py:29-33` | env `EXL3_INT8_GEMV` (default 2) sets the mgemm thresholds | Keep the default (thresholds 6 / 8192), or set `EXL3_INT8_GEMV=0` to make `use_mgemm` always true (all same-input groups fused). |
| `modules/quant/exl3.py:84,94`, `util/tensor.py:222-229` | `g_tensor_cache.get(device, (1,in), out_dtype)` scratch | Device-generic; OK. |
| `modules/quant/exl3_lib/quantize.py:1-12` | imported at load: `util.memory`, `cuda_sync_active` (import only) | Check these imports don't touch `torch.cuda` at import time; the module-level objects (`AutoSplit`, locks) do not. |
| C++: `hgemm.cu` cuBLAS, `exl3_gemm.cu` cooperative launch + `DevCtx` locks/workspace, `cudaFuncSetAttribute`, `CoopKernelAutotuner` | | All replaced by the shim. |

Env vars in this part:
- `EXL3_NO_FUSED_RECONSTRUCT` (Python) — recommend `=1` on XPU first.
- C++ only, and they vanish with the shim: `EXL3_INT8_GEMV`, `EXL3_INT8_GEMV_MAX_K`, `EXL3_GEMV` mode,
  `EXL3_HGEMM_F16ACC`.
- `EXL3_MGEMM_K_THRESHOLD` and `EXL3_MGEMM_N_THRESHOLD` (Python, `config.py`).
- `EXL3_VISION_PINNED` (vision only).

### 1.6 Minimal set for a correct forward (this part)

1. `BC_LinearEXL3` Python class: ctor storing fields, `run(x, y)`, and `run_alloc(x, n, fp32)`, all via
   `exl3_linear_ref`.
2. `reconstruct`, `reconstruct_slice`, `had_r_128`, and `hgemm_recon` (= matmul with fp32 accumulate),
   with `EXL3_NO_FUSED_RECONSTRUCT=1` so `reconstruct_had_slice` is not needed. Or implement it too; it
   is two `had128` calls around `reconstruct_w_hat`.
3. `BC_LinearFP16` (ctor, `run`) and `hgemm`.
4. A `data_ptr -> tensor` registry so the Part B-4 `exl3_mgemm` shim can resolve pointer tables.
5. Stubs that raise for everything else in the table.


---

## Part B. Dense MLP, MoE (routed + shared expert), routing, activations

> **Reconciliation note (see section 3).** This part's phase-1 advice says "don't provide `BC_LinearEXL3`". That is not possible: `LinearEXL3.__init__` constructs `ext.BC_LinearEXL3` unconditionally (`modules/quant/exl3.py:86`), and `LinearEXL3.forward` calls `self.bc.run_alloc`. With a Python `BC_LinearEXL3`, `GatedMLP.load_local` builds `ext.BC_GatedMLP` for the dense MLPs and the shared experts even under `no_reconstruct` (`mlp.py:655-687`; `multi_gu` is None, so `can_separate` holds). The shim therefore provides a small Python `BC_GatedMLP` whose `run_bszN(x, d)` runs gate/up `run_alloc`, `silu_mul(g, u, a, 10)` and `down.run(a, d)`. Everything else in the `no_reconstruct` plan stands: no `BC_BlockSparseMLP`, no `MultiLinear`, no `exl3_mgemm`, and the torch per-expert loop for every MoE batch size. Decode batch is 1–4 under `-ambs 4` (section 1), not 1–8.

Paths are relative to `glm/exllamav3/exllamav3/` unless stated. "ext" means `exllamav3_ext`.

### GLM-5.3-Flash facts that select the paths

| item | value | where it matters |
|---|---|---|
| dense MLP layers | 0, 1, 2 only (`mlp_layer_types`; layer 44 is `sparse`) | `GatedMLP`, hidden 4096 -> 12288 -> 4096, `select_hq_bits=1` |
| MoE layers | 3..44 (42 layers) | `BlockSparseMLP`, 288 experts, top-8, I = 2048 |
| shared expert | 1, `GatedMLP` 4096 -> 2048 -> 4096, `select_hq_bits=2` | merged into the routed sum |
| router | `router_type="dots"` -> `routing_dots` -> `ext.routing_ds3_nogroup` (sigmoid, noaux_tc bias, `norm_topk_prob`, x2.5) | every MoE layer, every token |
| activation | `silu`, `act_limit = swiglu_limit = 10.0`, `interm_dtype=torch.half`, `out_dtype=torch.float` | `ext.silu_mul` / fused kernels |
| expert codebook | mul1, K = 3 for routed experts (glm-flash-lite `docs/reference.md:444` "MUL1/K=3"); shared expert and dense MLP may carry higher K (`select_hq_bits`) | `support_fused` (mul1 uniform) and `mtile_ok` are True |
| `f_threshold` | `min(288 // 8, 4) = 4` (`block_sparse_mlp.py:190,736`) | only matters when `self.bc is None` |
| MLP input / output | input `(bsz, q_len, 4096)` fp16 (output of `post_attention_layernorm`, `transformer.py:177`); output fp32 to `mlp_hc.apply_` (`transformer.py:189`) | |
| CPU expert offload | off: `-mcl`/`-mcs` default 0 and serve.py never passes them (`model_init.py:66-68`, `glm53/serve.py:43`); also gated on `device.type == "cuda"` (`block_sparse_mlp_cpu.py:113,151`) | `block_sparse_mlp_cpu.py`, `model/moe_cpu_host.py` are unreachable |

glm-flash-lite's MoE monkeypatches (`exl3_tiers`, `expert_cache`, `cpu_tier`, `k_overlap`, `k_hcfuse`, `k_lookahead`) are only installed when their `GLM53_*` env vars are set (`glm53/serve.py:51-104`). Leave them all unset on XPU.

### (a) Op table (this part), hot-path order

Rank: 1 = every decode step of every MoE layer. "Stock" = the path taken with default flags; "no_reconstruct" = with `config.infer_params.no_reconstruct = True`.

| op | category | when reached | rank | Python call sites | condition / avoidable? | recommended XPU path |
|---|---|---|---|---|---|---|
| `routing_ds3_nogroup` | routing | every MoE layer, every forward (decode and prefill, incl. warmup) | 1 | `modules/block_sparse_mlp_routing.py:214` (bsz 1), `:245` (bsz > 1) via `routing_dots` (`block_sparse_mlp.py:451,998`) | always (router_type "dots"); only `activate_all_experts` (conversion) avoids it | torch: `logits = (y.float() @ gate.float()).half()`, then the top-k math below. Implement as a shim function |
| `det_quant_weight` | routing (load-ish) | first routing call per layer (lazy `_gate_t`) | 10 | `block_sparse_mlp_routing.py:86` | only if the gate is fp16 (it is). Output is used only by the det GEMM inside `routing_ds3_nogroup` | shim: no-op (or fill with a torch quantization) if the shim's routing ignores `gate_i8`/`gate_sb` |
| `BC_BlockSparseMLP` (ctor) | bound class | load (`load_local`) | load | `block_sparse_mlp.py:666` | built when `(support_quant_paths or support_bc_bszn) and not no_reconstruct` (`:612`) | stock: shim class. Minimal: avoid with `no_reconstruct` |
| `BC_BlockSparseMLP.run_bszN` | fused MoE decode | MoE layer, flattened rows `bsz <= 8` (decode 1-8, warmup rows 1/2/4/8) | 2 | `block_sparse_mlp.py:1319` | needs `self.bc` (`bszn_eligible`, `:989`). Avoid with `no_reconstruct` (then the torch per-expert path runs) | stock: shim implementing the "expert math" below in torch, writing `out_bszn`. Fast: repack to the trellis-serve `moe_forward` blob |
| `BC_GatedMLP` (ctor) | bound class | load of every `GatedMLP` whose `downs[0].inner.bc` exists | load | `mlp.py:672` | needs `BC_LinearEXL3` objects from the shim (`inner.bc`); absent `BC_LinearEXL3` means `self.bc = None` | minimal: don't provide `BC_LinearEXL3`, so this is never built |
| `BC_GatedMLP.run_bszN` | dense / shared MLP decode | dense layers 0-2 at `bsz*q_len <= 8`; shared expert at bsz <= 8 only when `sh_coop` is false (it is true by default, so the shared expert runs through the coop launch inside `BC_BlockSparseMLP.run_bszN`) | 3 | `mlp.py:744`; C++ from `BC_BlockSparseMLP::run_bszN` (`libtorch/blocksparse_mlp.cpp:100`) | `self.bc is not None and bsz*q_len <= MAX_BSZN(8)` | shim: gate/up EXL3 linear, `silu_mul`, down, written into `d` |
| `exl3_moe` | fused MoE prefill | MoE layer with `bsz > 8` (stock, any prefill chunk) | 4 | `block_sparse_mlp.py:1171` (via `run_fused`, `:1209/1218/1220/1222/1224`) | `fused_mode_buffers` exists (`support_fused`: uniform mul1 or mcg codebook); experts with `0 < count <= fused_rows` (256 with mtile) | stock: shim loop over experts (torch); fast: trellis-serve `moe_forward` prefill (DPAS) |
| `exl3_moe_gather` | MoE reduction | prefill whenever the fused or batched tier ran and `EXL3_MOE_FUSED_DET != 0` (default) | 5 | `block_sparse_mlp.py:1235` | `EXL3_MOE_FUSED_DET=0` swaps in atomic adds (`exl3_moe` without scratch) and skips this | shim: torch gather (exact semantics below) |
| `silu_mul` | activation | dense/shared `GatedMLP` at `> 8` rows (or always when `bc` is None); MoE per-expert torch path; batched-recon tier | 6 | `mlp.py:594` (bound), `:750`, `:782`; `block_sparse_mlp.py:394` (bound), `:1294`; `moe_batch_recon.py:56,280,283` | always on prefill | torch (below). Needed in every configuration |
| `exl3_mgemm` | fused gate+up GEMM | shared expert (`multi_gu` present, out 2048 < 8192) at `8 < rows <= 32` | 7 | `mlp.py:761` | `multi_gu` built when `use_mgemm(...)` (`mlp.py:643`, `model/config.py:61`); dense MLP (out 12288 >= 8192, K < 6) has no `multi_gu`. Avoid: `no_reconstruct` (`mlp.py:635`) | minimal: avoid. Else shim: two EXL3 linears on the same input |
| `BC_BlockSparseMLP.run_single_expert_dq` | per-expert prefill | prefill experts with `count > 512` (above batched-recon `max_rows`) | 8 | `block_sparse_mlp.py:1284` | `self.bc` set and `count > TEMP_ROWS_GRAPH(32)` and not fused/batched | shim: reconstruct + matmul (expert math below) |
| `reconstruct_had_batch` | batched recon | prefill experts with `256 < count <= 512` | 9 | `moe_batch_recon.py:199` | `EXL3_MOE_BATCH_RECON=1` (default) and `EXL3_MOE_RECON_FOLDED=1` (default); `EXL3_MOE_BATCH_RECON=0` avoids the whole tier | shim: reconstruct per expert, then fold (Part A's reconstruct) |
| `hgemm_batched` | batched GEMM | same as above | 9 | `moe_batch_recon.py:188` | same | `torch.bmm` with fp32 accumulation |
| `reconstruct_batch`, `had_r_128_batch` | batched recon (unfolded) | only with `EXL3_MOE_RECON_FOLDED=0` | n/r | `moe_batch_recon.py:202,185,191` | off by default | not needed |
| `exl3_moe_max_concurrency` | device query | load, when `support_fused` | load | `block_sparse_mlp.py:729` | only sizes `(C, R, H)` temp buffers | shim: return a small constant (for example 4) |
| `BC_BlockSparseMLP.run_single_expert` | graphed per-expert | not reached for GLM (fused tier takes every count <= 256, so `min_rows = 256`) | n/r | `block_sparse_mlp.py:1260` | only when `fused_mode_buffers is None` | stub that raises |
| `add_sigmoid_gate`, `add_sigmoid_gate_proj` | shared gate | not reached (no `shared_gate` key in GLM) | n/r | `block_sparse_mlp.py:1361,1363` | | not needed |
| `gelu_mul`, `relu2_mul`, `relu_mul`, `silu_oai_mul`, `xielu` | activations | not reached (activation is silu) | n/r | `mlp.py:595-596`, `block_sparse_mlp.py:397-411`, `moe_batch_recon.py:55-61` (dict built at import) | import-time attribute access only in `moe_batch_recon.py:55-61` and `mlp.py`/`block_sparse_mlp.py` match statements | the shim must expose the names (import-time dict); bodies can raise |
| `routing_std`, `routing_sel_norm`, `routing_gemm_det`, `routing_ds3_nogroup_logits`, `routing_std_logits`, `blocksparse_mlp_routing`, `exl3_moe_coop` (Python), `BC_MLP`, `moe_split_*`, `moe_unswizzle_trellis`, `exl3_moe_cpu_*`, `exl3_moe_flag_*`, `exl3_gemv_int8_max_k`, `add` | n/r | not reached for this model on XPU (`exl3_gemv_int8_max_k` is only called when `device.type == "cuda"`, `model/config.py:72-76`) | n/r | | | not needed |

Internal C++ dependencies of the bound classes (not called from Python, but part of what the shim's BC classes replace): `exl3_moe_coop_prepare/run` (`quant/exl3_moe_coop.cu`), `exl3_gemm_gr`, `exl3_mgemm_gr` (`quant/exl3_gemm.cu`), `had_r_128`, `had_r_128_dual`, `reconstruct`, `hgemm_recon` (used by `run_single_expert_dq`, `libtorch/blocksparse_mlp.cpp:491-524`), `silu_mul_gr`, `routing_gemv` and `routing_gemm_det_` (inside `routing_ds3_nogroup`), CUDA `Graph` (`graph.cuh`), `CoopKernelAutotuner` (`quant/coop_autotune.cu`), `DevCtx` (num_sms, locks).

### Recommended XPU strategy for this part

**Phase 1 (correct, not fast): set `config.infer_params.no_reconstruct = True` and don't provide `BC_LinearEXL3` in the shim.** Set it after `Config.from_directory` and before `model.load` (`model_init.py:194`). `InferParams.no_reconstruct` is a plain attribute (`model/config.py:17`). Effects in this part:

- `BlockSparseMLP.load_local` builds no `MultiLinear`s, no `BC_BlockSparseMLP`, no `fused_mode_buffers` (`block_sparse_mlp.py:527,533,612`). `forward` takes the "Torch/C++/fused" branch for every batch size (`:1052-1055`, `no_reconstruct` is one of the OR terms). Inside it, `fused_mode_buffers is None`, `_batch_recon_layer` returns None (`bc is None`), so every expert with rows runs the **torch per-expert path** (`:1288-1304`): `ups[e].forward`, `gates[e].forward`, `ext.silu_mul(g, u, a, 10.0)`, `downs[e].forward`, `mul_(w)`, `index_add_`.
- `GatedMLP.load_local` builds no `multi_gu` (`mlp.py:635`), and with no `inner.bc` the BC is never built (`mlp.py:655`). Every row count takes `mlp.py:746-755`: gate Linear, up Linear, `silu_mul`, down Linear.
- The only ext ops left in this part are **`routing_ds3_nogroup`, `det_quant_weight`, `silu_mul`**, plus the EXL3 `Linear.forward` from Part A (with `no_reconstruct`, `quant/exl3.py:140` always takes the non-reconstruct branch).
- Cost: one host sync per MoE layer (`expert_count.tolist()`, `block_sparse_mlp.py:1104`) and up to 3 x 288 Linear calls per layer at prefill. That is acceptable for correctness.

**Required Python patch regardless of path:** `block_sparse_mlp.py:818` runs `cpu_post_load()`, `load_local()` and `load_routing()` only when `torch.device(device).type == "cuda"`. On XPU none of them run, so `routing_cfg` stays None and the first forward crashes in `routing_dots`. Change it to `in ("cuda", "xpu")`, or monkeypatch.

**Phase 2 (fast decode):** drop `no_reconstruct` for the MoE (or keep it and monkeypatch `BlockSparseMLP.forward` for `bsz <= 8`) and route decode to an XPU grouped kernel:

- The trellis-serve `moe_forward` (`~/work/trellis-b70/trellis-serve/xpu/csrc/exl3_moe.sycl:1183`) is the natural fit. Its requirements: mul1, uniform K in {3, 4} for gate/up/down, `H % 128 == 0`, `I % 128 == 0`, E <= 1024.
- Repack exllamav3's per-expert tensors into one blob per expert (`moe_offload.py:81 pack_expert`):
  - gate|up trellis concatenated along the n-tile axis: `torch.cat([gate.trellis, up.trellis], dim=1)` gives `(H/16, 2I/16, 16K)` int16, viewed as uint32 words in PLANAR4 order;
  - then the down trellis `(I/16, H/16, 16K)`;
  - then fp16 `suh_g[H] suh_u[H] svh_g[I] svh_u[I] suh_d[I] svh_d[H]`.
- The tile bytes are identical (16·K int16 = 8·K uint32 per 16x16 tile), so it is a repack, not a re-quantization.
- Convert `selected_experts` int64 to int32 and `routing_weights` fp16 to fp32. Its output has x's dtype (fp16): cast to fp32 and add the shared expert separately.
- **Activation clamp differs (see below):** the n128 build clamps the gate *before* SiLU, sglang style, while exllamav3 clamps `silu(g)` *after*. The difference is at most `10 - silu(10) ≈ 4.5e-4` on the activated gate. It is not bit-exact. If exactness to exllamav3 matters, change `EXL3_ACT_CLAMP` to clamp after SiLU.
- The b70tier n128 file (`glm-flash-lite/b70tier/kernels/moe-n128/exl3_moe.n128.sycl`) is the same kernel plus that clamp (`-DEXL3_ACT_LIMIT=10.0f`).
- Prefill: `moe_forward` has a DPAS path (`moe_prefill` when `M >= g_prefill_min_m`). Alternatively keep the torch per-expert loop.

### (b) Per-op detail

#### Expert math (shared definition used below)

Let `lin(x, L)` be the EXL3 linear of Part A: `lin(x) = Hn( Hk(x ⊙ suh) @ W ) ⊙ svh`, where:

- `Hk`/`Hn` are block-diagonal normalized Hadamard-128 transforms along the feature axis (scale `1/sqrt(128) = 0.088388347648`);
- `W` is the decoded trellis (k x n, fp16);
- `suh` is (k,) fp16 and `svh` is (n,) fp16.

This matches `run_single_expert_dq` (`libtorch/blocksparse_mlp.cpp:491-524`: `had_r_128(y, yh, suh, -)`, `hgemm_recon`, `had_r_128(out, out, -, svh)`) and the coop kernel epilogues (`quant/exl3_moe_coop_kernel.cuh:772-799,920-923`: had128 then `* svh`; down input `* d_suh` then had128).

Expert e on input row x (fp16, width 4096):

```
g = lin(x, gate_e)        # (2048,) ; fp16 in the separate/torch paths, fp32 inside the coop kernel
u = lin(x, up_e)
a = min(silu(g), L) * clamp(u, -L, L)       # L = 10.0; a rounded to fp16
d = lin(a, down_e)        # (4096,) fp32 (down Linear out_dtype float)
```

Routed sum for token t: `R_t = Σ_{k=0..7} w_{t,k} · d_{e_{t,k}}(x_t)` (fp32). MoE output: `R_t + S(x_t)`, where `S` is the shared-expert GatedMLP (same formula, its own weights, fp32 out). No shared gate, no post norms, no latent projections, no biases, `interm_div = 1`.

#### `silu_mul(x, y, z, act_limit)` (`activation.cu:24-100`, `activation_kernels.cuh:140-260`)

- **Signature:** `silu_mul(Tensor x [gate], Tensor y [up], Tensor z [out, in place OK], float act_limit)`.
  - `x` and `y` are both fp16 (kernel `act_mul_kernel_h`) or both fp32 (`act_mul_kernel_f`). `z` must be fp16.
  - Any shape. Elementwise over `numel`, processed as pairs (`numel/2` threads × 2), so numel must be even. Contiguous.
- **fp16 kernel (GLM, interm_dtype half):** all in **half arithmetic**:
  - `s = x * hrcp(1 + hexp(-x))` (h2exp/h2rcp);
  - if `act_limit != 0`: `y = clamp(y, -L, L)` (`__hmax2`/`__hmin2`) and `s = min(s, L)`;
  - `z = s * y` (`__hmul2`), fp16, no finite clamp.
- **fp32 kernel:** `s = x * (1/(1+__expf(-x)))` in fp32; same clamps in fp32; `z = half_rn(s*y)` then clamped to ±65504.
- **Clamp semantics:** the gate is clamped *after* SiLU, from above only; the up lane is clamped symmetrically. `act_limit == 0.0` disables both.
- **Torch reference:** `z = (torch.clamp_max(F.silu(x.float()), L) * y.float().clamp(-L, L)).half()`. Computing in fp32 is the coop kernel's precision; the fp16 kernel differs by fp16 rounding only.
- Called with `z` aliasing `y` (`mlp.py:749-750` when `interm_dtype == half`; `moe_batch_recon.py:283`).

#### `routing_ds3_nogroup(hidden, gate, scores, bias, topk_indices, topk_weights, scaling_factor, gate_t, act_fn, gate_i8, gate_sb)` (`routing.cu:649-710`)

**Arguments:**

| arg | dtype / shape | role |
|---|---|---|
| `hidden` | fp16 `(bsz, 4096)` | router input `z = y` (no `router_pre_norm`) |
| `gate` | fp16 `(4096, 288)` | the routing Linear's `inner.weight`, loaded with `float2half=True`, transposed (`linear.py:212`) |
| `scores` | fp16 `(bsz, 288)` | out: logits |
| `bias` | optional fp16 `(288,)` | from `_esb_h`; see below |
| `topk_indices` | int64 `(bsz, 8)` | out |
| `topk_weights` | fp16 `(bsz, 8)` | out |
| `scaling_factor` | float | 2.5 |
| `gate_t` | fp16 `(288, 4096)` | contiguous transpose |
| `act_fn` | int | `ROUTING_ACT_SIGMOID = 0` |
| `gate_i8`, `gate_sb` | `(2, 288, 4096)` int8 / `(288,)` fp32 | from `det_quant_weight` |

- **`bias` construction:** `e_score_correction_bias` is loaded with `allow_bf16` and becomes `.float()` unless already fp16 (`block_sparse_mlp.py:785-795`). `_esb_h` (`block_sparse_mlp_routing.py:43-51`) mean-centers it (`esb - esb.mean()`) and casts to fp16. Selection is shift-invariant, so this is equivalent to the unshifted bias.
- **Outputs:**
  - for bsz 1: the persistent `cfg.*_bsz1` buffers;
  - for bsz <= 128: `g_tensor_cache.get_bucketed` workspaces **shared by all MoE layers on the device** (consumed before the next layer routes);
  - otherwise fresh tensors (`block_sparse_mlp_routing.py:19-37`).
- **Step 1, logits** (`routing_gemv`, `routing.cu:243-281`): `scores = half_rn(hidden @ gate)`.
  - bsz 1 with `gate_t`: one warp per expert; fp32 FMA dot over `half2` pairs, lane-strided, `__shfl_down` tree reduce, then rounded to fp16.
  - bsz > 1 with `gate_i8`: `routing_gemm_det` (`routing_gemm.cu`). This is an exact int8 hi/lo split GEMM: activations quantized per (row, 128-wide K chunk) to |q| <= 16319, weights per row; fp32 combine; fp16 out. Its accuracy is that of an fp16 tensor-core GEMM.
  - Fallback: `hgemm`.
  - On XPU, `(hidden.float() @ gate.float()).half()` is equivalent up to rounding. The point of these kernels is cross-arch bit-reproducibility for TP, which we don't need.
- **Step 2, top-k** (`routing_ds3_nogroup_topk_kernel`, `routing.cu:284-339`). One block per row, `ceil(E/32)*32` threads. Per expert i:
  - `act_i = sigmoid(float(scores_i))`, using `sigmoid_stable_hf` (`routing.cu:19-25`): `ez = exp_det(-|x|); base = ez/(1+ez)` (correctly rounded div), returning `1-base` for x >= 0 and `base` otherwise. `exp_det` is a FMA-only polynomial with input clamped to [-87, 88] (`det_gemm.cuh:124-139`).
  - `key_i = act_i + float(bias_i)` and `payload_i = act_i` (with no bias: `key = logit`, `payload = logit`, and sigmoid applied after selection).
  - **Top-8 by key, descending**, via warp 0 (`warp_topk_shared`, `routing.cu:88-134`): K rounds of argmax. Within a lane, the strict `>` keeps the lowest position. Across lanes, the lowest tied lane wins (sm80 `__reduce_max_sync` + `__ffs(__ballot)`; the shfl_down fallback also keeps the lower lane). **Tie-break:** among exactly equal keys, the expert with the smallest `index % 32` wins first, then the smallest `index / 32`. This is not "lowest index", but exact fp32 ties are practically nonexistent. Output slot k = k-th best (descending key order).
  - `sum = Σ_{k<8} payload_k` (warp tree reduce of the first K lanes) `+ 1e-20f`.
  - `topk_weights[k] = half_rn(payload_k * scaling_factor / sum)`.
  - **Weights come from the un-biased sigmoid scores.** The bias is used for selection only. Normalization is over the 8 selected; epsilon 1e-20 is added to the sum; scaling (2.5) is applied after normalization.
- **Torch reference:**

```python
logits = (z.float() @ gate.float()).half()
s = torch.sigmoid(logits.float())
key = s + bias_h.float()
idx = torch.topk(key, 8, dim=-1, sorted=True).indices
w = s.gather(-1, idx)
w = (w * 2.5 / (w.sum(-1, keepdim=True) + 1e-20)).half()
```

  Returns `(idx (int64), w (fp16))` and also writes `scores`, `topk_indices`, `topk_weights` in place, since bsz 1 and decode graphs reuse those buffers. Order of the k slots matters only for summation order.
- HF reference difference: the HF `noaux_tc` computes `scores_for_choice = sigmoid(logits) + bias` with fp32 logits. exllamav3 rounds logits to fp16 first. Keep fp16 rounding for parity with CUDA exllamav3.

#### `det_quant_weight(w, w_i8, w_s)` (`routing_gemm.cu:330-372`)

- `w`: fp16 `(N, K)` contiguous (`gate_t`, N=288, K=4096). `w_i8`: int8 out `(2, N, K)`. `w_s`: fp32 out `(N,)`.
- Per row: `scale = max|w|` (1 if 0), `inv = 16319 / scale`, `q = clamp(round_rn(w*inv), ±16319)`, `hi = (q+64) >> 7`, `lo = q - hi*128`, `w_s = scale/16319`.
- Only consumed by `routing_gemm_det_`. A shim that computes logits with torch can make this a no-op. The tensors are pre-allocated by Python at `block_sparse_mlp_routing.py:84-85`.

#### `BC_BlockSparseMLP` (`libtorch/blocksparse_mlp_bc.h`, `.h`, `.cpp`)

**Constructor** (`block_sparse_mlp.py:666-719`). Positional order exactly as bound (`blocksparse_mlp_bc.h`):

| # | args |
|---|---|
| 1-10 | `yh2` fp16 `(max(64, 64), Hi)`, `yh` fp16 `(64, 1, Hi)`, `interm_gu` fp16 `(max(64,128), I)`, `interm_g` `(64,1,I)`, `interm_u` `(64,1,I)`, `interm_a` fp16 `(64,1,I)`, `interm_a2` fp16 `(64, I)`, `out_d` fp32 `(64,1,Ho)`, `out_d2` fp32 `(64, Ho)`, `out_d_sh` optional fp32 `(1, 8, H)` |
| 11-16 | `coop_ctr` int32 `(64*(I/128) + 8*(Ho/128) + 2*64 + 3,)` zeroed, `had_u` fp16 `(64, Hi)`, `dq_temp_up` fp16 `(Hi, I)`, `dq_temp_down` (view `(I, Ho)` of the same storage), `min_expert`, `max_expert` (both -1 = no shard) |
| 17-34 | gate/up/down each as `ptrs_trellis`, `ptrs_suh`, `ptrs_svh` (int64 device tensors `(288,)` of `data_ptr()`s), `K` (float), `mcg` (bool), `mul1` (bool) |
| 35-37 | `act_silu`, `act_gelu`, `act_silu_oai` |
| 38-40 | `shared_experts` (`BC_GatedMLP` or None), `shared_gate` (`BC_LinearFP16` or None; None for GLM), `act_limit` |
| 41-46 | `gates`, `ups`, `downs` as lists of `BC_LinearEXL3`; `gu_trellis_ptr`, `gu_suh_ptr`, `gu_svh_ptr` (int64 `(288, 2)`, [gate, up] per expert) |
| 47 | `out_bszn` fp32 `(8, H)` |
| keyword | `gate_bias_ptrs=None`, `up_bias_ptrs=None`, `down_bias_ptrs=None`, `act_relu2=False`, `sh_coop=bool` |

Notes on the constructor:
- With H = Hi = Ho = 4096, I = 2048, `numex = 8`, `bszn_rows = 64`, `TEMP_ROWS_GRAPH = 32`.
- `g_tensor_cache` keys are device-global: every MoE layer shares the same `yh`/`interm`/`out_bszn` buffers.
- `sh_coop` is `shared_coop_ok()` = True for GLM (`block_sparse_mlp.py:1394-1417`; env `EXL3_MOE_SHARED_COOP=0` disables it).
- Read-only attribute: `sh_coop`.

**`run_bszN(y, selected_experts, routing_weights)`:**
- Inputs: `y` fp16 `(bsz, 4096)`, bsz 1..8; `selected_experts` int64 `(bsz, 8)`; `routing_weights` fp16 `(bsz, 8)`.
- Returns nothing. Writes `out_bszn[:bsz]` = routed sum + shared expert (fp32). **Python reads `experts_cfg.out_bszn[:bsz]` afterwards** (`block_sparse_mlp.py:1320`) and skips the Python shared-expert add (`bc_sh_exp`), so the shim must write into the constructor's `out_bszn` tensor and must include the shared expert.
- Shared expert: if `sh_coop`, a one-expert coop launch at the shared expert's own K writes `out_d_sh[:, :bsz]`. Otherwise `shared_experts.run_bszN(y.unsqueeze(0), out_d_sh[:, :bsz])`.
- Routed launches (`quant/exl3_moe_coop_kernel.cuh`):
  - **rot** (bsz > 1): `had_g/had_u[slot] = Hk(x_row ⊙ suh)`.
  - **A**: per (expert run, 32-col group, projection) trellis GEMV. On the last arrival per (slot, 128-chunk), sum the split-k partials, apply had128 · svh to g and u, compute `act_gate` in fp32 (`min(silu(g),L)·clamp(u,±L)`, `:113-129`), multiply by d_suh, apply had128, and store fp16 `act_out`.
  - **B**: down GEMV, then per (token, chunk): for k = 0..7 in order (top_k 8 <= 16 warps, so warp k owns slot k), `o += w_k · (had128(d_out) · d_svh)`. Then sum the warp partials in warp order and `o += 1.0 · sh_out[row]` (gate 1, no `sh_gate_w`). Store fp32, trimmed to H.
- A slot with `w == 0` or an out-of-range expert is inactive (contributes 0). Duplicate experts across tokens are grouped into runs (weights read once). A token with no active slots gets zeros plus shared.
- No CUDA graph on this path (`blocksparse_mlp.cpp:74-79`).

**`run_single_expert(y, expert_idx)`:** `y` fp16 `(count<=32, H)`. Graphed gate/up `exl3_gemm`, `silu_mul`, down into `out_d2[:count]`, which Python reads (`block_sparse_mlp.py:1261`). **Not reached for GLM.**

**`run_single_expert_dq(y, expert_idx, yh, interm, interm_a, out)`:**

| arg | dtype / shape | role |
|---|---|---|
| `y` | fp16 `(count, H)` | input rows |
| `yh` | fp16 `(2*count, H)` | scratch |
| `interm` | fp16 `(2*count, I)` | [gate rows; up rows] |
| `interm_a` | fp16 `(count, I)` | activation |
| `out` | fp32 `(count, H)` | written in place |

- Math: `yh1 = Hk(y⊙suh_g)`, `yh2 = Hk(y⊙suh_u)`, `interm1 = yh1 @ recon(gate)`, `interm2 = yh2 @ recon(up)` (hgemm_recon: fp16 in, fp16 out, fp32 accumulate), `interm{1,2} = Hn(·)⊙svh`, `interm_a = silu_mul(interm1, interm2, L)` (fp16 kernel), `interm_a = Hk(interm_a⊙suh_d)` (in place), `out = interm_a @ recon(down)` (fp32 out), `out = Hn(out)⊙svh_d`.
- Python then does `out_state.mul_(w)` and `index_add_` (`block_sparse_mlp.py:1303-1304`).
- Note the Python sizing quirk: `out_state_` is a slice but the call passes the full `out_state`, so the C++ side uses `y.size(0)` rows.

#### `exl3_moe(...)` (`quant/exl3_moe.cu:69-341`, kernel `quant/exl3_moe_kernel.cuh`)

**Positional args:**

| # | arg | dtype / shape | notes |
|---|---|---|---|
| 1 | `hidden_state` | fp16 `(rows, H)` | |
| 2 | `output_state` | fp32 `(rows, H)` | zero-initialized `fhs_ext[:rows]` |
| 3 | `expert_count` | int64 `(E+1,)` | bincount; last = sentinel |
| 4 | `token_sorted` | int64 `(rows*8,)` | |
| 5 | `weight_sorted` | fp16 `(rows*8,)` | |
| 6-9 | `temp_state_g`, `temp_state_u` | fp16 `(C, R, H)` | |
| | `temp_intermediate_g`, `temp_intermediate_u` | fp16 `(C, R, I)` | |
| 10 | `act_function` | int | 0 = silu |
| 11-13 | `K_gate`, `K_up`, `K_down` | float | |
| 14-22 | 9 pointer tables | int64 `(E,)` | |
| 23-28 | 6 codebook bools | | must be uniform; mcg xor mul1 |
| 29 | `act_limit` | | |
| 30 | `num_active` | | |
| 31 | `output_scratch` | optional fp32 `(slots, H)` | |
| 32 | `fused_base` | optional int64 `(E+1,)` | `tables[0]` |
| 33-35 | `count_lo`, `count_hi`, `m_tile` | | |

**Semantics.** For every expert e with `count_lo <= count_e <= count_hi` and `0 < count_e <= R` (R = `temp_state_g.size(1)`, here 256), process its rows `token_sorted[start_e : start_e+count_e]` (`start_e = cumsum(count)[e-1]`) with expert math:
- Gate/up input had is `had_hf_r_128_inner` on fp16.
- g/u GEMM out fp16. Activation in **half arithmetic** with the same clamp order (`hadamard_inner.cuh:355-411`). Down input had fp16.
- Down output: `had(·) · (0.0884·w_row) · svh_d` in fp32.
- With `output_scratch`: stored to `output_scratch[fused_base[e] + row_in_expert]`. Without it: `atomicAdd` into `output_state[token]`.

The `m_tile` (16/32/64) and `num_active` parameters only pick kernel instances and launch width; they don't change math beyond accumulation-order rounding.

Python issues one launch per tier (`block_sparse_mlp.py:1207-1224`):
- if `rows*8 <= 256` (rows <= 32): `run_fused(-1)` with `expert_count_list = None` and no host sync;
- else per-tier launches with `[1,16]`, `[17,32]`, `[33,256]` (m_tile 16/32/64).

#### `exl3_moe_gather(output_state, output_scratch, flat_expert, inv_order, expert_start, slot_base, slot_kind, weight_sorted)` (`quant/exl3_moe.cu:376-504`)

- All index tensors are int64. `output_state`/`output_scratch` are fp32; `weight_sorted` is fp16.
- Shapes: `flat_expert`/`inv_order` `(rows*8,)`; tables `(E,)` (Python passes `tables[1|0|2, :288]`).
- For each token t and column c:
  - `slots = [(slot_base[e] + inv_order[a] - expert_start[e], wt) for k in 0..7, a = t*8+k, e = flat_expert[a], 0 <= e < E, slot_kind[e] != 0]`;
  - `wt = 1` for kind 1 (fused, already weighted), `wt = weight_sorted[inv_order[a]]` for kind 2 (batched recon, unweighted);
  - `output_state[t,c] += Σ_k scratch[slot_k, c] * wt_k`, summed in k order in fp32.

#### `exl3_mgemm` (GatedMLP fused gate+up only) (`quant/exl3_gemm.cu:347-678`)

**Bound signature:** `(A, B, C, suh, A_had, svh, indices, weights, K, force_shape_idx, mcg, mul1, min_index, max_index, force_num_sms, num_tokens=1, size_n_list=None, c_ptrs=None, n_stride_list=None, had_src_list=None, num_had_src=0)`. Returns the int shape index.

**General semantics:** `B`, `suh`, `svh` are int64 device tables of addresses (trellis / suh / svh per matrix). Active slot j uses `q = indices[j]` or j, and computes `C[j] = lin(A[j or 0], q)`:
- `a_batches == 1` broadcasts A.
- With `weights`, `C[0] = Σ_j weights[j] · C[j]`.
- `min_index >= 0` filters to `[min, max)` and rebases.
- Negative indices are skipped.
- Sliced mode (`size_n_list`/`c_ptrs`/`n_stride_list`/`had_src_list`) is for equal-width column slices of several source matrices written in place to per-slice output pointers. Used by attention, not here.

`A_had` scratch needs `bszm * m * k` fp16. `k % 16 == 0`, `n % 128 == 0`. Kernel shape is autotuned with cooperative launch.

**Call in this part** (`mlp.py:761-777`):
- `A = xv` fp16 `(1, rows, 4096)`, `B = ptrs_trellis` `(2,)` [gate, up], `C = gu` fp16 `(2, rows, 2048)`, `suh`/`svh` tables `(2,)`, `A_had = guh` fp16 `(2, rows, 4096)`.
- `indices = weights = None`, `K = multi_gu.K`, `force_shape_idx = -1`, mcg/mul1 as stored, `min = max = -1`, `force_num_sms = 0`, `num_tokens = 1`.
- So it is just `C[0] = lin(x, gate)` and `C[1] = lin(x, up)`. Reached only for the shared expert at 9-32 rows.

#### `BC_GatedMLP` (`libtorch/mlp_bc.h`, `mlp.h`, `mlp.cpp`)

**Constructor** (`mlp.py:672-687`): `(guh fp16 (2,8,4096), gu interm (2,8,out_f), a fp16 (1,8,out_f), down_xh fp16 (1,8,out_f), gu_ptrs_trellis|None, gu_ptrs_suh|None, gu_ptrs_svh|None, gu_K float, gu_mcg, gu_mul1, act_silu, act_gelu, act_relu2, gate BC_LinearEXL3|None, up BC_LinearEXL3|None, down BC_LinearEXL3, act_limit)`.
- Requires either the fused tables or both gate and up handles.
- Dense MLP (K < 6, out 12288 >= 8192): separate gate/up.
- Shared expert: fused tables.

**`run_bszN(x, d)`:**
- `x` fp16 `(1, n, 4096)` with n = 1..8; `d` fp32 `(1, n, 4096)` written in place.
- Math: `g, u = lin(x, gate), lin(x, up)` (fp16 out); `a = silu_mul(g, u, L)` (fp16 kernel); `d = lin(a, down)` (fp32). No biases.
- CUDA graph per n (`graph_bszN[8]`): records on the 2nd call, replays on the 3rd+, patching A/C pointers.

#### `exl3_moe_max_concurrency(device)` (`quant/exl3_moe.cu:15-19`)

Returns `num_sms / 8`. Python uses it only to size `fused_mode_buffers` `(C, R=256, H|I)` fp16 x 4 (`block_sparse_mlp.py:729-735`).

#### `hgemm_batched(a, w, c)` and `reconstruct_had_batch(W, ptrs, suh_ptrs, svh_ptrs, K, mcg, mul1)` (batched-recon tier)

- **`hgemm_batched`** (`hgemm.cu:109-...`): `a` fp16 `(B, m, k)`, `w` fp16 `(B, k, n)`, `c` fp16|fp32 `(B, m, n)`, all contiguous. `c = a @ w` with fp32 accumulation; on GeForce the fp16-accumulator variant with fp32 accumulation across K chunks.
- **`reconstruct_had_batch`**: `W` fp16 `(B, k, n)` out; `ptrs`, `suh_ptrs`, `svh_ptrs` int64 `(B,)` addresses. Writes `W[b]` = the folded dense weight such that `x @ W[b] == lin(x, expert b)` (suh, Hadamards and svh folded in, rounded to fp16). See Part A for `reconstruct` / `reconstruct_had_slice`.
- **`run_group`** (`moe_batch_recon.py:208-311`):
  1. gather rows padded to cmax (pads read a zero row and write a sink row);
  2. `u = x @ Wu` (fp16), `g = x @ Wg` (fp16);
  3. `silu_mul(g, u, u, L)`;
  4. `d = u @ Wd` into the fp32 slot scratch, unweighted (kind 2), summed by `exl3_moe_gather`.
- Tier bounds for GLM: `RECON_TILES=64` gives `max_rows = max(128, 64*128*128 // min(2048, 4096)) = 512`; `cap = min(16, (256 << 20) // (2*(max(4096*2048, 2048*4096) + 4096*2048))) = min(16, 8) = 8` experts per group; `PAD_MAX = 1.1`.

### (c) MoE layer forward, step by step (`BlockSparseMLP.forward`, `block_sparse_mlp.py:964-1377`)

Input `x` fp16 `(b, q, 4096)` gives `y = x.view(-1, 4096)` and `bsz = b*q` rows.

1. **Routing** (`:997-998`): `selected (bsz,8) int64, weights (bsz,8) fp16 = routing_dots(...)` (ext `routing_ds3_nogroup`). During warmup (`params["tp_warmup"]`, set by `model.warmup`) these are overwritten with random experts and weights 1/8 (`:1018-1023`).
2. **Path choice** (stock flags; `bc` built, `support_fused`, mul1):
   - **bsz <= 8 (decode, batch 1-8):** `bc.run_bszN(y, selected, weights)` gives `out_bszn[:bsz]`, which already includes the shared expert (`:1317-1321`). No host sync, no graph.
   - **bsz > 8 (prefill chunks, up to 8192 rows; also bsz 9-32 multi-token verify):**
     1. `fhs = zeros(bsz+1, 4096) fp32`.
     2. `flat_token = arange(bsz).repeat_interleave(8)`; `order = flat_expert.argsort()`; `token_sorted`, `weight_sorted`; `expert_count = bincount(flat_expert, minlength=289)`.
     3. If `bsz*8 <= 256`: all fused, no host sync. Otherwise `expert_count.tolist()` (host sync).
     4. Tiers by row count c:
        - `0 < c <= 256`: `exl3_moe` (fused, m_tile 16/32/64 by c);
        - `256 < c <= 512`: batched recon groups;
        - `c > 512`: per-expert `run_single_expert_dq` (count > 32, so the DQ branch).
     5. With `EXL3_MOE_FUSED_DET=1` (default), the fused and batched tiers write per-assignment fp32 slots and `exl3_moe_gather` sums them per token in k order. Per-expert DQ experts `index_add_` afterwards, in expert-index order.
     6. Shared expert (`:1352-1365`): `self.shared_experts.forward(x)` (GatedMLP):
        - 9-32 rows: `exl3_mgemm` gate+up;
        - > 32 rows: separate Linears;
        - then `silu_mul` and down (fp32), and `fhs += y_shared`.
   - **With `no_reconstruct` (recommended phase 1):** the bsz > 8 structure for every bsz, but every expert takes the torch path: `Linear` forwards plus `silu_mul`, weighted with `mul_(w)` (fp16 weight broadcast onto fp32), then `index_add_` in expert order. The shared expert is added by the Python `+=`.
3. **Output:** fp32 `(b, q, 4096)` = routed + shared. Note that the decode path returns a **view of the device-shared `out_bszn` buffer**. The caller (`mlp_hc.apply_`) consumes it before the next MoE layer overwrites it.

**Accumulation dtype:** fp32 throughout the routed sum and the shared add. Intermediate gate/up/activation are fp16 (fp32 inside the coop decode kernel before the fp16 store of the activation). Routing weights are fp16.

**Dense MLP (layers 0-2)** `GatedMLP.forward` (`mlp.py:721-792`):
- `rows <= 8` with `bc`: `BC_GatedMLP.run_bszN`.
- Otherwise (multi_gu None for dense): gate Linear (fp16 out), up Linear (fp16 out), `silu_mul(g, u, u, 10)`, down Linear (fp32 out).
- One slice (12288 < `MAX_MLP_INTERMEDIATE` 55296).

### (d) Non-ext CUDA dependencies in this part

| file:line | what | reached? | XPU mapping |
|---|---|---|---|
| `modules/block_sparse_mlp.py:818` | `if torch.device(device).type == "cuda": cpu_post_load(); load_local(); load_routing()` | **yes, load** | must accept `"xpu"`, otherwise `routing_cfg` is None and the first forward crashes |
| `modules/block_sparse_mlp.py:888` | `autosplit_extra_measure` returns early unless `device.type == "cuda"` | load (autosplit) | harmless; loses the worst-case prefill VRAM reservation. Accept `"xpu"` if autosplit under-reserves |
| `modules/block_sparse_mlp.py:1104` | `expert_count.tolist()` host sync per MoE layer (prefill with > 32 rows; every bsz under `no_reconstruct`) | yes | works on XPU (sync) |
| `modules/block_sparse_mlp.py:1164`, `modules/moe_batch_recon.py:243` | `torch.from_numpy(...).to(dev, non_blocking=True)` from pageable memory | prefill | works on XPU |
| `modules/block_sparse_mlp.py:1606-1611`, `modules/mlp.py:435,911` | `consumer.recv(cuda=True)`, `torch.cuda.synchronize()` in `tp_import` | no (TP only) | not needed |
| `model/config.py:72-76` | `use_mgemm`: `device.type == "cuda"` then `ext.exl3_gemv_int8_max_k` | load | on XPU the branch is skipped and `K_thr = 6` (`EXL3_MGEMM_K_THRESHOLD` default); fine |
| `modules/block_sparse_mlp_cpu.py:19-20,73,113,151` | `torch.cuda.Event`, `torch.cuda.synchronize`, `type == "cuda"` gates | no (offload off; gated on cuda) | not needed |
| `model/moe_cpu_host.py` (whole file) | `cudaHostRegister`, flags, streams | no | not needed |
| C++ (inside BC classes) | CUDA graphs (`BC_GatedMLP::graph_bszN`, `BC_BlockSparseMLP::graph_single`), cooperative launches (`exl3_mgemm`, `exl3_moe` group barriers via `DevCtx` locks), completion counters/atomics (coop kernels), cuBLAS (`hgemm_batched`, routing fallback), `cudaFuncSetAttribute` | stock path | replaced wholesale by the shim. Optional later: `torch.xpu` graph capture if available in the installed torch; not required |

No `torch.compile`, flash-attn, xformers, pinned memory or explicit streams in `mlp.py`, `block_sparse_mlp.py`, `block_sparse_mlp_routing.py`, `moe_batch_recon.py` or `multilinear.py`. `util/tensor.py:162-163` uses `is_pinned()` for non_blocking copies (generic).

### Part B summary

- **Minimal op set for a correct forward from this part:** with `no_reconstruct=True`, no `BC_LinearEXL3` in the shim, and the `:818` cuda-check patch, only `routing_ds3_nogroup`, `det_quant_weight` (may be a no-op) and `silu_mul` are needed. The shim must also expose the names `gelu_mul`, `relu2_mul`, `relu_mul`, `silu_oai_mul`, `xielu` because they are referenced at import/construct time (`moe_batch_recon.py:55-61`, `mlp.py:593-596`, `block_sparse_mlp.py:392-411`). Plus Part A's EXL3 Linear forward.
- **Stock path (no flags):** `BC_BlockSparseMLP` (ctor, `run_bszN`, `run_single_expert_dq`), `BC_GatedMLP` (ctor, `run_bszN`), `exl3_moe`, `exl3_moe_gather`, `exl3_mgemm`, `exl3_moe_max_concurrency`, `hgemm_batched`, `reconstruct_had_batch`, plus the three above.
- **Surprises:**
  - (1) The `type == "cuda"` gate at `block_sparse_mlp.py:818` silently skips routing setup on XPU.
  - (2) Router logits are rounded to fp16 before the sigmoid.
  - (3) `act_limit` clamps `silu(g)` after the activation, while the trellis-serve n128 SYCL kernel clamps g before SiLU. The difference is small but the paths are not bit-equal.
  - (4) Decode returns a view of a device-global static `out_bszn` buffer shared by all MoE layers. A shim `run_bszN` must write it in place and include the shared expert.
  - (5) Top-k tie-break is lane-major, not index-major.
  - (6) `run_single_expert` (graphed) is unreachable for GLM. Prefill uses fused (<= 256 rows), batched-recon (257-512) and DQ (> 512) tiers.
- **Cross-slice:** `lin()` / `reconstruct` / `had_r_128` / `BC_LinearEXL3` / `exl3_gemm` semantics and the `no_reconstruct` effect on `quant/exl3.py:140` are Part A's. The `Graph` class is Part E's. The MLP input (fp16, from `mlp_norm` after `mlp_hc.mix`) and output (fp32, into `mlp_hc.apply_`) are Part E's HC. Warmup (`model.warmup`, `tp_warmup` random routing; `-nw` skips it) is Part E's.


---

## Part C. KDA linear attention (GatedDeltaNet in KDA mode) and recurrent state

Paths are relative to `glm/exllamav3/exllamav3`; C++/CUDA paths are relative to `exllamav3_ext/`.

### Checkpoint facts (shard-1 safetensors header, rev 332ab457, layer 0)

These come from an HTTP range read of the header of `model-00001-of-00015.safetensors` (rev 332ab457).

| tensor | dtype | shape | notes |
|---|---|---|---|
| `self_attn.qkv_proj.{trellis,suh,svh,mul1}` | I16 / F16 / F16 / I32 scalar | trellis `[256, 1536, 80]` (K/16, N/16, 16·5 bits), suh `[4096]`, svh `[24576]` | EXL3 5 bpw (select_hq_bits=2 bump), mul1 codebook. The q/k/v projections are already fused in the checkpoint (`key_qkv="qkv_proj"`). The `q_proj/k_proj/v_proj` alt keys are not present. |
| `self_attn.o_proj.*` | same | trellis `[512, 256, 80]`, suh `[8192]`, svh `[4096]` | EXL3 5 bpw; `out_dtype=float` |
| `self_attn.b_proj.weight` | F16 | `[64, 4096]` | unquantized (qmap None), no bias |
| `self_attn.f_a_proj.weight` | F16 | `[128, 4096]` | low-rank forget gate, stage 1 |
| `self_attn.f_b_proj.weight` | F16 | `[8192, 128]` | stage 2 (per k-channel, 64 heads × 128) |
| `self_attn.g_a_proj.weight` | F16 | `[128, 4096]` | low-rank output gate, stage 1 |
| `self_attn.g_b_proj.weight` | F16 | `[8192, 128]` | stage 2 |
| `self_attn.A_log` | F32 | `[64]` | per head |
| `self_attn.dt_bias` | F32 | `[8192]` | per k-channel (H·Dk), not per head |
| `self_attn.conv1d.weight` | BF16 | `[24576, 1, 4]` | already fused q\|k\|v depthwise conv. There is no conv bias. |
| `self_attn.o_norm.weight` | BF16 | `[128]` | per-head gated RMSNorm weight, shared by all 64 heads |

Geometry: H = Hk = Hv = 64, Dk = Dv = 128, F = fdim_qkv = 2·64·128 + 64·128 = 24576, conv K = 4, hidden = 4096,
`gate_lower_bound = -5.0`, `beta_scale = 1.0` (default; GLM does not pass it), `rms_norm_eps = 1e-5`.

### (a) Op table

Hot-path rank: 1 = every decode token on every KDA layer; 2 = every prefill chunk; 3 = load or once; 4 = speculative decoding only; — = not reached for GLM.

| op | category | when reached | rank | Python call sites | condition / avoidable? | recommended XPU path |
|---|---|---|---|---|---|---|
| `BC_GatedDeltaNetSplit` (KDA ctor), `.needs_configure`, `.configure_slot_kda`, `.run_bszN` | bound class (CUDA-graph fused layer) | load (ctor); decode and prefill steps with 1≤bsz≤8 and 1≤seqlen≤16 | 1 | ctor `modules/gated_delta_net.py:777`; configure `:1058-1062` → `:890` (`_bc_configure_slot_kda`); run `:1064` | Built only if `EXL3_BC_GDN != "0"` (`:20`) and the device is not CPU and qkv/o are exl3, b/f/g are fp16, conv weight is bf16 and dt_bias exists (`:747-757`; all true for this checkpoint). **Avoidable: `EXL3_BC_GDN=0`**, which leaves `self.bc=None` and `bc_split=False`, so every shape takes the torch path. | **Force `EXL3_BC_GDN=0`.** Later, for speed, optionally a Python class with the same methods that runs the torch-path sequence below (optionally under `torch.xpu.graph`). |
| `cuda_recurrent_gated_delta_rule` (channelwise / KDA variant) | CUDA kernel | torch path when seqlen < 64 or history=True: decode (with BC off), prefill tails of 17..63 tokens, warmup "pre 32" step | 1 | `modules/gated_delta_net_fn/gated_delta_rule.py:150` (KDA branch). `:210` is the GDN branch and is not reached. | Threshold `seqlen >= num_v_heads (64) and not history` → fla chunk; otherwise this kernel. No flag. | Torch implementation (exact math below). This is the decode hot path. |
| `cuda_causal_conv1d_update` | CUDA kernel | torch path with seqlen ≤ 32 and all bf16 | 1 | `modules/gated_delta_net_fn/conv1d.py:395` | Gated by `mixed_qkv.is_cuda` (`conv1d.py:386`), **which is False on XPU**, so the shim function is never called there. The code falls to the Triton conv, which then raises at `conv1d.py:245` (`if not x.is_cuda`). | Patch `conv1d.py:386` and `:245` (`is_cuda` → `device.type in ("cuda","xpu")`) and `:279` (`torch.cuda.device` → device-agnostic), then implement this op in torch, or let all lengths use the Triton conv. |
| `gated_rms_norm` | CUDA kernel | every torch-path KDA forward (decode with BC off, and prefill) | 1 | `modules/gated_rmsnorm.py:120` | Python fallback at `gated_rmsnorm.py:112-118` is taken only if x is not bf16 or not contiguous, or the gate is not contiguous. Here x is bf16 and contiguous and z is fp32 and contiguous, so ext is always called. | Torch implementation (exact math below; the fallback at :114-118 is already the reference). |
| `BC_GatedRMSNorm` (ctor only) | bound class | model load, once per KDA layer | 3 | `modules/gated_rmsnorm.py:55` (and TP-only `:164`, `:189`) | Unconditional in `GatedRMSNorm.load`. Its `.run` is called only from C++ BC classes. | Trivial Python holder class (store weight, eps, constant_bias, w_groups, gate_first, gate_act; `.run(x,y,gate)` → `gated_rms_norm`). |
| `batched_conv_rewind`, `batched_state_rewind`, `ConvRewindJob`, `StateRewindJob` | CUDA kernel + POD structs | only after a forward with `recurrent_history=True` (MTP / n-gram / draft-model speculative decoding) | 4 | jobs `modules/gated_delta_net.py:266`, `:283`; launches `:62`, `:64`; driven by `GDNState.rewind` `:115-123` ← `generator/generator.py:1089`, `:1246`, `generator/job.py:901`, `:1438` | Without drafting, `last_history == 0`, so both job builders return None and no ext call is made (`rewind(0)` is a no-op). The serve.py command line uses no `--mtp`, `-ngram` or `-dm` → **not reached**. | Not needed for the minimal port. If drafting is added, use plain torch slice copies (as `GDNLayerState.rewind`, `:238-251`, already does). |
| `kda_gate_op` | CUDA kernel (also bound standalone) | only inside the BC run | — with BC off | none from Python (the binding exists in `bindings.cpp`, but only `run_bszN_gr` calls `kda_gate_op_gr`) | Avoided by `EXL3_BC_GDN=0` | Only needed for a BC re-implementation; math below. |
| `gdn_lowrank_gemv_f`, `gdn_ba_gemv` | CUDA kernels | only inside the BC run | — with BC off | none from Python | same | `x @ w_t.T` in fp32 |
| `exl3_mgemm` (sliced qkv+z) | EXL3 | GDN split path only (`multi_qkvz`) | — | `modules/gated_delta_net.py:927` | `self.kda` takes the `elif` at `:1094` before the `else` at `:1115`, and `multi_qkvz` is only built in the non-KDA split branch | not needed |
| `BC_GatedDeltaNet` (fused qkvz), `gated_delta_net_fused_op`, `gated_delta_net_fused_op_2` | — | Qwen3-Next / Qwen3.5 GDN only | — | `:645`, `:1083`, `:1130` | require `qkvz_proj` / non-KDA | not needed |
| fla Triton `chunk_kda` (8 kernels, see (d)) | vendored Triton | prefill chunks with seqlen ≥ 64 (normal `-chunk_size 8192` prefill; warmup "chunk 8192") | 2 | `modules/gated_delta_net_fn/gated_delta_rule.py:114-137` | No flag. Can only be avoided by patching the threshold so everything goes to the recurrent op, which is O(T) sequential and very slow in torch. | Run the vendored Triton on Intel XPU Triton after patching `vendor/fla/utils.py:117` (`torch.cuda.device`). As a correctness oracle use `torch_recurrent_kda` (`gated_delta_rule.py:64`). |
| Triton conv `_causal_conv1d_update_slotted_*` (3 kernels) | in-tree Triton | torch-path conv with seqlen > 32, or non-bf16 | 2 | `modules/gated_delta_net_fn/conv1d.py:409` → `:283` / `:306` / `:325` | host wrapper `is_cuda` check (`:245`) and `torch.cuda.device` (`:279`) | Patch the two host checks. The kernels themselves are portable. |

Projection linears used by the torch path (`qkv_proj`/`o_proj` EXL3 → `BC_LinearEXL3`/`exl3_gemm`/`reconstruct`…, and `b/f_a/f_b/g_a/g_b` fp16 `Linear` → `hgemm` or `BC_LinearFP16`) belong to Part A. The torch KDA path calls `Linear.forward` 7 times per layer.

### (b) Per-op detail

#### `cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, recurrent_state, core_attn_out, num_k_heads, num_v_heads, k_head_dim, v_head_dim, slots, history)`
Source: `gdn.cuh:42-55` (decl), `gdn.cu:851-1031` (host), `gdn.cu:660-848` (`cuda_recurrent_gated_delta_rule_kernel_128<save_history, V_SPLIT, CHANNELWISE=true>`); binding `bindings.cpp` `m.def("cuda_recurrent_gated_delta_rule")`, positional with no kwargs.

Arguments (all must be contiguous; the Python KDA branch calls `.contiguous()`):
- `mixed_qkv`: bf16 `[bsz, seqlen, F]`. Row layout `[q(H·128) | k(H·128) | v(Hv·128)]`. For head h, q is at `h*128`, k at `(H+h)*128`, and v at `2H*128 + h*128`.
- `g`: fp32. KDA (channelwise) is selected by `g.dim()==4`, shape `[bsz, seqlen, Hv, Dk]`, holding the natural-log decay per k-channel. KDA requires Dk = Dv = 128.
- `beta`: bf16 `[bsz, seqlen, Hv]`.
- `recurrent_state`: fp32 `[num_slots, max_history+1, Hv, Dk, Dv]`. Element `S[k][v]` is at `k*128 + v`, so v is contiguous. **In place.**
- `core_attn_out`: bf16 `[bsz, seqlen, Hv, Dv]`. **Output, in place.**
- `slots`: optional int32 `[bsz]` mapping each batch row to a state slot (identity if None). In the KDA call the Python passes `recurrent_slots`, an int32 device tensor.
- `history`: bool.

Launch: grid `(bsz, Hv, v_split)`, block `(128 / v_split, 4)` (`SUBK=4`). `v_split = 4` iff bsz==1 (and Dk≤128, Dv==128, Hv≤64). The batch-1 split only partitions V columns, so the math is identical either way. `scale = 1/sqrtf(Dk)`.

Per token s, per head h (state read from `R` and written to `W`; see the history addressing below):
```
q = float(q_bf16); k = float(k_bf16)                       # length 128
q = q * rsqrtf(sum(q*q) + 1e-6);  k = k * rsqrtf(sum(k*k) + 1e-6)   # fp32, eps inside the sqrt
a = __expf(g[s,h,:])                                      # fast exp, per k-channel decay factor
dot1[v] = sum_k k[k] * a[k] * R[k][v]                     # read OLD state, decayed per k (fixed order: 4 k-slices of 32, summed slice 0..3)
vprime[v] = float(v_bf16[v]) - dot1[v]
W[k][v] = R[k][v] * a[k] + k[k] * vprime[v] * float(beta[s,h])   # fp32, written in place
o[v] = sum_k q[k] * W[k][v]                               # read NEW state
out[s,h,v] = __float2bfloat16_rz(o[v] * scale)            # NOTE: round-toward-zero to bf16 (truncation), scale = 1/sqrt(128)
```
In matrix form: `S' = diag(exp g) S`, `u = v − Sᵀ'k`, `S = S' + β·k uᵀ`, `o = (Sᵀ q̂)/√Dk`, where q̂ and k̂ are L2-normalized and β multiplies after the subtraction. The reference `torch_recurrent_kda` (`gated_delta_rule.py:64-85`) computes the same thing but scales q before the dot and rounds the output with RN. Expect last-bit bf16 differences from the RZ rounding only.

History addressing (`save_history=true`, i.e. `history=True`): `slot_state = state[slot]`, and `final = slot_state[0]`.
- Token 0 reads `final`. Token s>0 reads `slot_state[s]`.
- Token s<seqlen-1 writes `slot_state[s+1]`. The last token writes `final`.
- Result: `state[slot,0]` is the state after the last token, and `state[slot,i]` (1≤i≤seqlen−1) is the state after token i−1.
- Non-history: read and write `state[slot,0]` in place.
- The host asserts `state.size(1) >= seqlen` when history is set.

Torch reimplementation (exact semantics except the fast-exp and summation-order ULPs):
```python
def recurrent_kda(mixed_qkv, g, beta, state, out, H, Hv, Dk, Dv, slots, history):
    B, S, _ = mixed_qkv.shape
    q, k, v = mixed_qkv.float().split([H*Dk, H*Dk, Hv*Dv], -1)
    q = q.view(B,S,H,Dk); k = k.view(B,S,H,Dk); v = v.view(B,S,Hv,Dv)
    q = q * torch.rsqrt((q*q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k*k).sum(-1, keepdim=True) + 1e-6)
    a = g.exp(); bt = beta.float()
    idx = slots.long() if slots is not None else torch.arange(B, device=state.device)
    for b in range(B):
        sl = idx[b]
        Sst = state[sl, 0].clone()                      # [Hv, Dk, Dv]
        for t in range(S):
            Sst = Sst * a[b,t].unsqueeze(-1)              # decay rows (k)
            u = v[b,t] - torch.einsum('hk,hkv->hv', k[b,t], Sst)
            Sst = Sst + bt[b,t,:,None,None] * k[b,t].unsqueeze(-1) * u.unsqueeze(-2)
            o = torch.einsum('hk,hkv->hv', q[b,t], Sst) * Dk**-0.5
            out[b,t] = o.to(torch.bfloat16)               # CUDA uses RZ; use RN or emulate RZ via view-int truncation
            if history and t < S-1: state[sl, t+1] = Sst
        state[sl, 0] = Sst
```
Vectorize over b for decode, since all rows have seqlen 1. Note: the CUDA kernel reads the old (undecayed) state and folds `a[k]` into dot1. This is algebraically identical to decaying first.

#### `cuda_causal_conv1d_update(x, conv_state, slots, weight, bias, out, activation, history)`
Source: `gdn.cuh:158-168`, `gdn.cu:1320-1515` (`conv1d_update_kernel<ACT,HISTORY>`). Binding name `cuda_causal_conv1d_update`, positional.

Arguments: `x` bf16 `[bsz, F, seqlen]` (channels-major, made by `qkv.transpose(1,2).to(bf16).contiguous()`); `conv_state` bf16 `[num_slots, F, state_size]`, **in place**; `slots` int32 `[bsz]` or None (identity); `weight` bf16 `[F, K]` (K ≤ 16, here `conv1d_weight_flat` = `conv1d.weight.squeeze(1)`, shape `[24576, 4]`); `bias` bf16 `[F]` or None (None for GLM); `out` bf16 `[bsz, seqlen, F]`, **output in place, transposed to token-major**; `activation` (Python passes True = SiLU); `history`.

Math, per channel d, with `in = concat(conv_state[slot,d,0:K], x[b,d,0:seqlen])`:
`out[b,s,d] = bf16_rn(silu( bias[d] + Σ_{k=0..K-1} w[d,k]·in[s+k+1] ))`. It accumulates in fp32 via fmaf in k order 0..K-1, starting from the bias, and uses `silu(a) = a / (1 + __expf(-a))`. So `w[d,K-1]` multiplies the current token and `conv_state[slot,d,0]` (the oldest of K stored inputs) is never used. This is identical to `F.conv1d(cat(state[:,:,:K], x), w[:,None,:], groups=F)[..., -seqlen:]` (`conv1d.py:341-361`).

State write-back:
- Non-history: `conv_state[slot,d,0:K] = in[seqlen : seqlen+K]`, the last K inputs, oldest first.
- History: write the last `min(state_size, K+seqlen)` inputs to the tail `conv_state[slot,d, state_size-w : state_size]`. The head `[0:K]` is left stale until `rewind` copies the right window into it. That is why the generator always calls `rewind(0)` after a fully accepted history step.

`conv_state` is allocated as `[max_batch_size, F, K + max_history]` (`gated_delta_net.py:184-193`). With no drafting, state_size = K = 4.

#### `gated_rms_norm(x, w, y, g, epsilon, constant_bias, w_groups, gate_first, gate_act)`
Source: `norm.cuh` decl, `norm.cu:499-590` (kernel), `:608-707` (host); called as `ext.gated_rms_norm(x, self.weight, y, gate, eps, constant_bias, groups, gate_first, gate_act)` (`gated_rmsnorm.py:120`).

For KDA, the arguments are: x = core_attn_out bf16 `[B,S,64,128]`; w = o_norm bf16 `[128]`; y = half `[B,S,64,128]` (`out_dtype=torch.half`, `gated_delta_net.py:542-544`); g = z fp32 `[B,S,64,128]` (the g_b output); eps = 1e-5; constant_bias = 0.0; w_groups = 1; gate_first = False; gate_act = 1 (sigmoid).

The host accepts x bf16 only, w fp32 or bf16, y half or fp32, and g bf16 or fp32. Rows = B·S·64 and dim = 128, so the "small" variant runs with 32 threads per row.

Math per row (fp32 throughout): `ss = Σ x²` (fma, warp reduce); `r = rsqrtf(ss/dim + eps)`; `y = half( (x · (w + constant_bias) · r) · sigmoid(g) )`, with `sigmoid(g) = __fdividef(1, 1 + __expf(-g))`. For gate_act = 0 the gate is `silu(g)`. With gate_first, the gate multiplies x before the sum of squares (the Mamba2 variant, not used here).

Torch: the fallback at `gated_rmsnorm.py:114-118` (`h = x.float(); h *= rsqrt(mean(h²)+eps); h = w.float()*h; h *= sigmoid(gate.float()); h.to(half)`) is the exact reference up to ULPs.

#### `BC_GatedRMSNorm(weight, rms_norm_eps, constant_bias, w_groups=1, gate_first=False, gate_act=0)`
Source: `libtorch/gated_rmsnorm.h`, `gated_rmsnorm.cpp`, binding `gated_rmsnorm_bc.h`. It is a data holder whose only method, `.run(x, y, gate)`, calls `gated_rms_norm(x, weight, y, gate, eps, constant_bias, w_groups, gate_first, gate_act)`. Python constructs it positionally with 6 args.

#### `BC_GatedDeltaNetSplit` — KDA constructor (only if BC enabled)
Binding `libtorch/gated_delta_net_bc.h` (2nd `py::init`). Positional args: `qkv_proj: BC_LinearEXL3, o_proj: BC_LinearEXL3, b_weight_t [64,4096] f16, f_a_weight_t [128,4096] f16, f_b_weight_t [8192,128] f16, g_a_weight_t [128,4096] f16, g_b_weight_t [8192,128] f16, dt_bias [8192] bf16, a_log [64] f32|bf16, lower_bound float (-5.0), num_k_heads, num_v_heads, k_head_dim, v_head_dim, conv1d_weight [F,4] bf16, conv1d_bias opt, norm: BC_GatedRMSNorm, beta_scale float`.

The `*_t` tensors are empty at load and filled on the first forward from `Linear.inner.get_weight_tensor().T` (`gated_delta_net.py:1025-1032`). `dt_bias_bc` is a bf16 copy of the fp32 dt_bias.

Methods:
- `needs_configure(bsz, seqlen, history) -> bool`. It asserts 1≤bsz≤8 and 1≤seqlen≤16.
- `configure_slot_kda(bsz, seqlen, history, qkv f32[B,S,F], z f32[B,S,64,128], b_out f32[B,S,64], fa_out f32[B,S,128], fb_out f32[B,S,8192], ga_out f32[B,S,128], beta bf16[B,S,64], g f32[B,S,64,128], mixed_qkv bf16[B,F,S], conv_out bf16[B,S,F], core_attn_out bf16[B,S,64,128], core_attn_out_f f16[B,S,8192], qkv_xh f16[B,S,4096], o_xh f16[B,S,8192])`.
- `run_bszN(x f16[B,S,4096], y f32[B,S,4096] out, conv_state, recurrent_state, slots int32[B], history)`.

Run sequence (`libtorch/gated_delta_net.cpp:216-324`):
1. `exl3_gemm(x→qkv)`.
2. `b_out = x@b_wᵀ`, `fa_out = x@f_aᵀ`, `ga_out = x@g_aᵀ` (`gdn_ba_gemv`: fp16 in, fp32 accumulate and out).
3. `fb_out = fa_out@f_bᵀ` and `z = ga_out@g_bᵀ` (`gdn_lowrank_gemv_f`: **fp32 input**, fp16 weight, fp32 out).
4. `kda_gate_op`.
5. `cuda_causal_conv1d_update(mixed_qkv→conv_out, act=true)`.
6. `cuda_recurrent_gated_delta_rule(conv_out, g, beta → core_attn_out)`.
7. `norm.run(core_attn_out → core_attn_out_f, gate=z)`.
8. `exl3_gemm(core_attn_out_f → y)`.

Graph capture policy (`gated_delta_net.cpp:326-411`): the first call per slot runs eager. The next call captures a CUDA graph (via the `Graph` class in graph.cu, Graph slice). After that it replays with patched pointers: x is patched into 4 nodes, plus conv_state, slots ×2, recurrent_state, and y. If the state-buffer geometry differs from the captured one, it falls back to eager.

Numerical difference vs the torch path (both are "correct"):
- The torch path rounds the f_a and g_a outputs to fp16 before f_b and g_b (`gated_delta_net.py:1101`, `:1109`). BC keeps them fp32.
- BC uses a bf16 copy of dt_bias. Torch uses fp32.
- BC uses `__expf` fast exp.

#### `kda_gate_op(qkv, b, f, dt_bias, a_log, mixed_qkv, beta, g, lower_bound, beta_scale)`
Source: `gdn.cu:1786-1896`. It is a single elementwise kernel over three index ranges:
- `mixed_qkv[b,f,s] = bf16_rn(qkv[b,s,f])`, a transpose and cast.
- `beta = bf16_rn(sigmoid_fast(b) · beta_scale)`.
- `g[b,s,h,c] = lower_bound · sigmoid_fast(exp(a_log[h]) · (f[b,s,h·Dk+c] + dt_bias[h·Dk+c]))` when lower_bound ≠ 0, otherwise `−exp(a_log[h]) · softplus(f+dt_bias)` with `softplus(x) = x>20 ? x : log1p(exp(x))`.

Dtypes: qkv, b, f fp32; dt_bias bf16; a_log fp32 or bf16; outputs bf16, bf16, fp32.

#### `gdn_ba_gemv(x f16[..,k], w_t f16[n,k], bias f16[n]|None, y f32[..,n])` / `gdn_lowrank_gemv_f(x f32[..,k], w_t f16[n,k], y f32[..,n])`
Source: `gdn.cu:1630-1784`. Both compute `y = x @ w_tᵀ (+bias)` with fp32 accumulation. Lanes stride over k and the reduction uses warp `__shfl_down`. k must be even for `ba_gemv`.

#### `batched_conv_rewind(jobs: list[ConvRewindJob], device_index)` / `batched_state_rewind(jobs: list[StateRewindJob], device_index)`
Source: `gdn.cuh:207-250`, `gdn.cu:1912-2010`. Jobs are raw device pointers.
- `ConvRewindJob(src, dst, dim, cdim, stride)`: for each channel d<dim, copy `cdim` bf16 values from `src + d*stride` to `dst + d*stride`. It reads all values before writing, so overlapping windows are safe. It implements `conv_state[slot,:, :K] = conv_state[slot,:, p-K:p]` with `p = state_size - num_tokens`.
- `StateRewindJob(src, dst, num_elements)`: flat fp32 copy (float4) of `num_elements` values. Python passes `rs.stride(1)` = Hv·Dk·Dv as `num_elements`. It implements `state[slot,0] = state[slot, last_history+1-num_tokens]`.

Torch equivalent: `GDNLayerState.rewind` (`gated_delta_net.py:238-251`) already does this with slice copies. In the shim, build the jobs as tuples and do `ctypes`-free copies, or monkeypatch `_dispatch_rewind_jobs` to call `l.rewind()`.

### (c) KDA layer forward, step by step (torch path = recommended XPU path; `gated_delta_net.py:980-1177`)

Input: `x` f16 `[B,S,4096]`, the output of `attn_norm` after the hyperconnection mix (`transformer.py:150-155`). Output: fp32 `[B,S,4096]` (out_dtype float), added back through the HC apply.

1. **qkv** `qkv = qkv_proj(x)`: EXL3, fp32 `[B,S,24576]`. Then `mixed_qkv = qkv.transpose(1,2).to(bf16).contiguous()`, giving `[B,F,S]` (`:1095-1096`).
2. **output gate (z)** `z = g_b_proj( g_a_proj(x).to(f16) )`: fp16 Linears with fp32 out, viewed `[B,S,64,128]` (`:1099-1100`). There is no activation here; the sigmoid is applied inside the norm.
3. **beta** `beta = sigmoid(b_proj(x).float() * beta_scale).to(bf16)`, shape `[B,S,64]` (`:1102-1103`).
4. **decay** `f = f_b_proj( f_a_proj(x).to(f16) )`: fp32 `[B,S,8192]`. Then `gf = (f + dt_bias.float())`, viewed `[B,S,64,128]`, and `g = lower_bound · sigmoid( exp(A_log)[h] · gf )` with lower_bound −5.0, so g ∈ (−5, 0) is the natural-log decay per k-channel (`:1107-1114`). The softplus form `g = −exp(A_log)·softplus(gf)` (threshold 20) is used only if lower_bound is None.
5. **short conv** `causal_conv1d_update` (`:1139`, `conv1d.py:364-419`): a depthwise causal conv with K=4 over the 24576 channels, followed by SiLU, using and updating `conv_state[slot]`. Output bf16 `[B,S,F]`, token-major.
   - CUDA kernel if is_cuda, S≤32, and everything is bf16.
   - Else the Triton kernel `_causal_conv1d_update_slotted_kernel` for S≤256.
   - Else the `_output_kernel` + `_state_kernel` pair.
   - If no recurrent state is passed (no cache, e.g. `teacher_forced` in serve.py with `attn_mode="flash_attn_nc"`), a zero conv_state is used.
6. **delta rule** `gated_delta_rule_fn(..., channelwise_g=True)` (`gated_delta_rule.py:112-163`).
   - If S ≥ 64 and not history: split q, k, v, and loop **per batch row** over `recurrent_slots` (CPU copy): `chunk_kda(q,k,v,g,beta, initial_state=state[slot,0][None], output_final_state=save_state, use_qk_l2norm_in_kernel=True)`, then `state.copy_(new_state)`. Output cast to bf16.
   - Else `cuda_recurrent_gated_delta_rule`.
   - Both normalize q and k internally (eps 1e-6) and use scale = 1/√128.
7. **gated norm** `core = o_norm(core_attn_out, gate=z)`: `gated_rms_norm`, sigmoid gate, eps 1e-5, output f16 `[B,S,64,128]`, then viewed `[B,S,8192]`.
8. **o_proj** EXL3, `[B,S,8192]` → fp32 `[B,S,4096]`.

Branch selection summary:

| shape | EXL3_BC_GDN=1 (default) | EXL3_BC_GDN=0 |
|---|---|---|
| decode B∈1..8, S=1 | BC graph (all-CUDA) | torch path: conv CUDA op, recurrent CUDA op |
| S ≤ 16 (B ≤ 8) | BC | torch: conv CUDA op (S≤32), recurrent op |
| 17 ≤ S ≤ 32 | torch: conv CUDA op, recurrent op | same |
| 33 ≤ S ≤ 63 | torch: Triton conv, recurrent op | same |
| S ≥ 64 (prefill, chunk 8192) | torch: Triton conv (two-kernel variant for S>256), fla `chunk_kda` | same |

On XPU, `is_cuda` is False, so the "conv CUDA op" rows go to the Triton conv once the `is_cuda` gate at `conv1d.py:245` is patched.

**Cache state per KDA layer** (`GDNLayerState`, `gated_delta_net.py:174-315`; one per layer instance, allocated by `Cache` with `max_batch_size = args.autosplit_max_batch_size` slots, `cache/cache.py:160-171`; note `model_init.py` passes `-ambs` (4 in the serve command), and `generator.py:235` then caps the generator batch to `cache.num_slots`):
- `conv_state`: bf16 `[slots, 24576, 4 + max_history]` (192 KiB per slot with max_history = 0).
- `recurrent_state`: fp32 `[slots, max_history + 1, 64, 128, 128]` (4 MiB per slot per history entry), indexed `[slot, hist, h, k, v]`.
- Zeroed at alloc, and `clear(slot)` zeroes it for each new `GDNState` (`:95-100`, `:225-228`).
- Total for 34 layers × 4 slots ≈ 570 MiB with max_history = 0.
- `max_history = max(num_draft_tokens, MTP default draft size, 4 if ngram)`, which is 0 for the serve.py command.

**Checkpointing / second tier**:
- `RecurrentCache` (`cache/recurrent.py:21-124`) is an LRU dict in system RAM. Its size is `recurrent_cache_size` (serve.py: `-rcs`, default 4.0 GiB).
- Keys are page hashes. `GDNState.stash()` (`gated_delta_net.py:132-145`) copies `recurrent_state[slot,:1].cpu()` and `conv_state[slot,:,:4].cpu()` for every recurrent layer. These are plain torch D2H copies, pageable and synchronous.
- `checkpoint_size = 24576·4·2 + 64·128·128·4 = 4,390,912` bytes per layer, ≈149 MB per checkpoint for 34 layers, so 4 GiB holds about 28 checkpoints.
- Restore: `Cache.new_from_stashed` → `unstash` → `.copy_` H2D (`:147-155`, `:290-302`).
- Checkpoints are taken every `default_recurrent_checkpoint_interval = 2048` tokens near the generation boundary, and every 32768 (`recurrent_checkpoint_interval_pp`) during long prompt ingestion (`generator/job.py:1565-1579`).
- Prefill chunks are aligned to those boundaries.
- Rewinding to an arbitrary earlier point (`rollback_capacity()==0`) restores the nearest page-aligned stash and replays.
- **No ext ops are involved**: it is all torch `.cpu()` / `.copy_`, which works unchanged on XPU.

**Rewinds**: a cheap in-place rewind is only possible right after a forward with `recurrent_history=True`, which the generator sets only when `draft_tokens is not None` (`generator/generator.py:1036`), i.e. MTP, a draft model, or n-gram. Not reached in the target configuration.

### (d) Triton kernels reached on the KDA path

Common host-side issues for every fla kernel:
- **`vendor/fla/utils.py:117`** `input_guard` enters `torch.cuda.device(t.device.index)` for any tensor with an index. That fails on an XPU-only torch, so patch it to `torch.xpu.device` or a device-type dispatch.
- `utils.py:35-44`: `device_platform` comes from `triton.runtime.driver.active.get_current_target().backend`, which is "xpu" on Intel Triton. So `IS_NVIDIA`, `IS_NVIDIA_HOPPER`, `IS_NVIDIA_BLACKWELL` and `IS_TF32_SUPPORTED` are all False, and `IS_AMD` is False. This means:
  - `safe_dot` takes the plain `tl.dot` (no inline PTX `mov.f32` asm, `op.py:50-59`; that branch is Blackwell only).
  - `SOLVE_TRIL_DOT_PRECISION='ieee'` (`kda_chunk_intra.py:21-24`).
  - The `TRITON_F32_DEFAULT` env is not set (`utils.py:56-58` is NVIDIA only).
- `utils.py:49`: `torch.cuda.get_device_capability` is only evaluated when IS_NVIDIA.
- `utils.py:62`: `device="cuda"` allocator, Blackwell only. Not reached.
- `utils.py:80-87`, `check_shared_mem`: uses `torch.cuda.device_count()`, which is 0 on XPU, so `min([])` raises, gets caught, and the function returns False. Every `check_shared_mem(...)` is therefore False on XPU and the **smallest** tile lists are chosen: cumsum `BS_LIST=[16,32]`; chunk_delta_h `num_stages∈[2,1]` and `BV∈[32]`. This is harmless but slower. If you patch it to `torch.xpu.device_count()`: BMG SLM is about 128 KiB, so `'ada'` (101376) and the default (102400) would pass and `'ampere'` would not.
- `op.py:12`: `import triton.language.extra.libdevice as tldevice` runs at package import. It must resolve on Intel Triton (upstream ≥3.x has a backend-dispatching `extra/libdevice`). Only `tanh` (unused by KDA) and the `FLA_USE_FAST_OPS=1` variants touch it; keep `FLA_USE_FAST_OPS` unset. `exp2` = `tl.math.exp2`.
- `IS_GATHER_SUPPORTED = hasattr(tl, "gather")` is only used in the safe-gate sub-chunk kernel, which is not reached. `IS_TMA_SUPPORTED=False`, and `make_tensor_descriptor` is only resolved at import (solve_tril, not reached).
- Autotune: all configs are expressed in NVIDIA `num_warps` (1…32) and `num_stages` (1…4). On Intel Triton, `num_warps` is the number of sub-groups (`threads_per_warp` 16 or 32), so `num_warps=32` → up to 1024 work-items. That fits BMG's 1024 max workgroup with 32-wide warps but is invalid if a larger warp is configured. `num_stages` is accepted and is a software-pipelining hint. `cache_results=True` is used when the Triton version supports it.
- **glm-flash-lite `glm53/triton_pin.py`** pre-fills `Autotuner.cache` with NVIDIA picks from a reference run, for bit-reproducible KDA prefill. Those picks are not meaningful on XPU, so don't install it (`GLM53_TRITON_PIN` unset). Make a new pin set on the B70 if reproducibility matters.
- A fp32 `tl.dot` without explicit precision runs in `chunk_kda_fwd_kernel_inter_solve_fused` (`kda_chunk_intra.py:158-215`, fp32 operands). On NVIDIA it uses tf32 by default. On XPU it uses Triton's default `input_precision` (tf32 unless `TRITON_F32_DEFAULT=ieee`). Xe2 DPAS tf32 support is a risk: **set `TRITON_F32_DEFAULT=ieee` for correctness parity**. That is more accurate than NVIDIA's tf32 and slower.
- No inline PTX, no `tl.extra.cuda`, and no fp8 in any reached kernel.

| kernel (file:line) | reached via | dtypes | CUDA-specific flags | XPU risk |
|---|---|---|---|---|
| `_causal_conv1d_update_slotted_kernel` (`modules/gated_delta_net_fn/conv1d.py:16`) | torch-path conv, 32 < S ≤ 256 | bf16 io, fp32 acc | host: `x.is_cuda` check `:245`, `torch.cuda.device` `:279`; `num_warps=4` fixed | low (pure tl ops, `tl.sigmoid`) once the host checks are patched |
| `_causal_conv1d_update_slotted_output_kernel` (`conv1d.py:116`) + `_state_kernel` (`:186`) | S > 256 (every 8192 prefill chunk) | same | same | low |
| `l2norm_fwd_kernel` (`vendor/fla/l2norm.py:54`) | chunk_kda q and k (D=128 ≤ 512 branch) | bf16 in/out, fp32 rstd, eps 1e-6: `x/sqrt(Σx²+eps)` | autotune `num_warps` 1..16 × BT | low |
| `chunk_local_cumsum_vector_kernel` (`fla/cumsum.py:90`) | g `[B,T,H,K]`, scale RCP_LN2=1.4426950216, chunk-local (64) inclusive cumsum, fp32 out | fp32 | autotune `BS∈BS_LIST`, warps 2..8 | low |
| `chunk_kda_fwd_kernel_intra_token_parallel` (`fla/kda_chunk_intra_token_parallel.py:29`) | `safe_gate=False` (default; `chunk_kda` never passes safe_gate) | Aqk bf16 `[B,T,HV,64]`, Akkd fp32 `[B,T,HV,16]` | autotune BH × warps 1..8 | low |
| `chunk_kda_fwd_kernel_inter_solve_fused` (`fla/kda_chunk_intra.py:40`) | always | fp32 dots (default precision!) plus `ieee` solve_tril dots; Akk bf16 | autotune BK∈{32,64} × warps 1..4 | medium: fp32 dot precision (see above) |
| `recompute_w_u_fwd_kda_kernel` (`fla/kda_wy_fast.py:32`) | always | bf16 dots | autotune warps 2..8 × stages 2..4 | low |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (`fla/chunk_delta_h.py:43`) | always (gk path) | bf16 dots, h stored bf16 `[B,NT,HV,K,V]`, final_state fp32 | warps from `IS_NVIDIA_*` lists ([2,4] on XPU); stages and BV from `check_shared_mem` | low or medium (large register tiles: 4×64-row K blocks for K=128) |
| `chunk_gla_fwd_kernel_o` (`fla/gla_chunk.py:36`) | always | bf16 dots | autotune BK{32,64} × BV{64,128} × warps 2..8 × stages 2..4 (literal lists; `BK_LIST`/`BV_LIST` are computed but unused) | low |

Imported but not executed for KDA: `gdn_chunk_fwd`, `chunk_scaled_dot_kkt`, `gdn_wy_fast`, `solve_tril`, `chunk_o`, `chunk_h`, and `l2norm_fwd_kernel1` (D>512). Their module-level code (autotune decorators, `IS_*` flags, `make_tensor_descriptor` lookup) still runs at `import exllamav3.vendor.fla`, which happens on the first prefill ≥ 64 tokens (`gated_delta_rule.py:114`).

Numerics of the chunk path:
- q and k are L2-normalized into **bf16** before the chunk math. The CUDA recurrent kernel keeps them fp32.
- Per-chunk states h are bf16, while the carried final state is fp32.
- g is pre-scaled by 1/ln2 and exp2 is used.
- The output is cast to q's dtype (bf16).
- The existing test `exllamav3/tests/test_fla_vendored_.py` compares the vendored `chunk_kda` against upstream fla. Use `torch_recurrent_kda` (`gated_delta_rule.py:64`) as the oracle on XPU.

### (e) Non-ext CUDA dependencies in this part

| file:line | what | path reached? | torch.xpu mapping |
|---|---|---|---|
| `modules/gated_delta_net_fn/conv1d.py:386` | `mixed_qkv.is_cuda` gate for the CUDA conv op | every torch-path conv | `mixed_qkv.device.type in ("cuda","xpu")`; on XPU it otherwise silently routes to Triton |
| `modules/gated_delta_net_fn/conv1d.py:245` | `if not x.is_cuda: raise` | every Triton conv | same replacement; **hard blocker** |
| `modules/gated_delta_net_fn/conv1d.py:279` | `with torch.cuda.device(x.device)` | every Triton conv | `torch.xpu.device(x.device)` or `torch.accelerator` / `torch.device` context |
| `vendor/fla/utils.py:117` | `torch.cuda.device(idx)` in `input_guard` | every chunk_kda call | `torch.xpu.device(idx)`; **hard blocker** |
| `vendor/fla/utils.py:49`, `:84` | `torch.cuda.get_device_capability`, `torch.cuda.device_count` | import (49 skipped when not NVIDIA; 84 returns [] and falls back to False) | `torch.xpu.device_count()` (optional, tile sizes) |
| `vendor/fla/utils.py:62` | `torch.empty(..., device="cuda")` Triton allocator | Blackwell only | n/a |
| `vendor/fla/op.py:12` | `triton.language.extra.libdevice` import | package import | must exist in Intel Triton |
| `vendor/fla/op.py:50-59` | `tl.inline_asm_elementwise("mov.f32 ...")` PTX | Blackwell only | n/a |
| `modules/gated_delta_net.py:677`, `:748` | `device != torch.device("cpu")` gates BC construction | load | true on XPU; set `EXL3_BC_GDN=0` |
| `modules/gated_delta_net.py:1389`, `modules/gated_rmsnorm.py:167` | `torch.cuda.synchronize()` | TP import only | n/a (single GPU) |
| `modules/gated_delta_net.py:44-46` | `torch.device(l.device).index` for rewind batching | drafting only | works for "xpu:0" |
| `libtorch/gated_delta_net.cpp:326-411` | CUDA graph capture/replay (`Graph`) with pointer patching | BC only | `torch.xpu.graph` (XPUGraph, torch ≥ 2.8), or drop it by disabling BC |
| `GDNState.stash/unstash` `gated_delta_net.py:290-302` | `.cpu()` / `.copy_` | checkpointing | device-agnostic |

### Minimal KDA op set for a correct XPU forward

Set `EXL3_BC_GDN=0` (no BC_GatedDeltaNetSplit, no kda_gate_op, no gemvs, no graphs). Then:
1. Shim `gated_rms_norm` (torch) and `BC_GatedRMSNorm` (holder).
2. Shim `cuda_recurrent_gated_delta_rule` (torch, channelwise and history-aware as specified above).
3. Shim `cuda_causal_conv1d_update` (torch, via F.conv1d plus the state write-back), or skip it by routing all conv through Triton.
4. Patch three Python CUDA-isms: `conv1d.py:245/279/386` and `vendor/fla/utils.py:117`.
5. Export `TRITON_F32_DEFAULT=ieee`.
6. Rewind ops are only needed if MTP or n-gram drafting is enabled.

The fp16 Linears (b/f/g) and EXL3 linears (qkv/o) come from Part A.


---

## Part D. MLA attention (NoPE), k-pool DSA indexer, KV/indexer caches

All paths are relative to `glm/exllamav3/` (Python package in `exllamav3/`).
GLM-5.3-Flash MLA geometry (config): 11 MLA layers (`full_attn_layers` 3,7,...,43), all with
`indexer_types == "full"` (no "shared" layers). H = 64 heads, q_lora_rank 1536, kv_lora_rank
D_c = 512, qk_nope_head_dim = 256, **qk_rope_head_dim D_r = 0 (NoPE)**, v_head_dim D_v = 256,
sm_scale = 256^-0.5 = 0.0625, rms_norm_eps 1e-5. Indexer: H_i = 32, D_i = 128, index_topk 2048,
index_kpool P = 4, kpool_tail True, index_norm_eps = 1e-6 (MLAttention default; glm5_next.py
does not pass it). PAGE_SIZE = 256 (`exllamav3/constants.py:2`).

### Key findings first

1. **No ext op is mandatory for MLA if `EXL3_BC_ATTN=0`.** The eager ("dispatch") MLA path is
   pure torch + Triton, except `ext.dsa_topk` (needed whenever visible context > 2048) and the
   generic ops owned by other parts (EXL3 linears for q_a/q_b/kv_a/o/idx_wq_b, `ext.rms_norm`
   for q_a_layernorm/kv_a_layernorm). `ext.rope` is never reached (rope is None for NoPE, and
   the indexer's rope is tied to qk_rope_head_dim = 0, so `indexer_rope_interleave` in
   config.json is ignored by exllamav3).
2. **Force `EXL3_BC_ATTN=0` on XPU** (not just `EXL3_BC_MLA=0`). `EXL3_BC_MLA=0` only gates
   `bc_mla_step` (`modules/mla_attn.py:1027`); `autosplit_extra_measure`
   (`modules/mla_attn.py:1110-1118`, called at load from `model/model_ls.py:200`) calls
   `build_bc_mla` directly, which then AOT-compiles cubins via `triton.compile(...).asm["cubin"]`
   and constructs `ext.TritonKernel`/`ext.BC_MLAttention` (`modules/attention_fn/bc_attn.py:84-87`,
   `bc_mla.py:86`). `bc_attn_enable` (`bc_attn.py:42`, env `EXL3_BC_ATTN`) is checked inside
   `build_bc_mla` (`bc_mla.py:575`) and turns all of it off. Alternatively
   `EXL3_AUTOSPLIT_WORSTCASE=0` skips the extra measure, but BC must still be off for decode.
3. Even on CUDA the BC graph path is used only for **bsz == 1** decode on kpool layers
   (`bc_mla.py:520-521`: kpool slots return None for bsz > 1), so batch 2-8 decode already runs
   the eager path. The eager path is the reference to port.
4. No flash-attn / xformers is involved for MLA. `attn_mode` is `"flash_attn"` (paged cache;
   generator, warmup, autosplit) or `"flash_attn_nc"` (cache-less; serve.py teacher-forced
   `/generate` logprobs, `glm53/serve.py:266`). Both route to MLA's own Triton kernels
   (`modules/mla_attn.py:421-428`); the generic `attn_dispatch` backend list
   (`attention_fn/dispatch.py`) is imported but never called by MLA. xformers/torch-SDPA
   fallbacks are lazily imported and unused.
5. Default cache is fp16 (`model_init.py` builds `CacheLayer_fp16` unless `-cq`; serve.py
   passes no `-cq`, and reports `kv_cache_dtype = args.cache_quant or "fp16"`). MLA maps it to
   `CacheLayer_MLA_fp16` (`mla_attn.py:308-323`). So `quant_cache_cont`, `dequant_cache_cont`,
   `_mla_kv_quant_scatter_kernel`, all `QC>0` Triton branches and `_rot_h32` are unreached.
6. The Triton kernels on the path contain **no** inline PTX/asm, `tl.extra.cuda`, libdevice,
   fp8, tf32/`input_precision`, atomics, `tl.sort/topk/histogram/gather`, or cache/eviction
   hints (grep over `mla_triton.py`, `dsa_triton.py`, `triton_paged.py` is clean). The CUDA
   specifics are all in the Python wrappers: `with torch.cuda.device(...)` around every launch
   and `torch.cuda.get_device_properties(i).multi_processor_count` for split heuristics.
7. **Zero-width tensors**: with D_r = 0 the rope-key cache `kpe` is `(pages, 256, 1, 0)` and
   `q_pe` is `(R, H, 0)`; both are passed to Triton kernels as pointer args (never dereferenced:
   all D_r uses are `if D_r > 0` constexpr-guarded). A zero-numel torch tensor can have
   `data_ptr() == 0`. Verify the Intel Triton launcher accepts a null pointer arg; if not, pass a
   1-element dummy fp16 tensor for those args on XPU.

### (a) Op table (MLA slice)

Hot-path rank: 1 = every MLA layer every step, 2 = every step once context > 2048 tokens,
3 = load/warmup/rare, 4 = not reached by default.

| op | category | when reached | rank | Python call sites | condition / avoidable? | recommended XPU path |
|---|---|---|---|---|---|---|
| `ext.dsa_topk` | ext fn (CUDA radix select) | sparse regime: visible ctx `max(cache_seqlens)+seqlen > 2048`, prefill and decode, every MLA layer | 2 | `modules/mla_attn.py:696, 708` (kpool path, used); `:553, :567` (non-kpool `_indexer_topk`, unreached for GLM5.3) | not avoidable once ctx > 2048 (autosplit load forward and warmup's long-context family already hit it) | torch: stable sort / topk on fp16 scores, see detail |
| `ext.BC_MLAttention` (+ `set_indexer`, `configure_slot`, `configure_slot_dsa`, `run`, `needs_configure`) | bound class, CUDA-graph decode block | bsz==1, q_len<=16 decode, `EXL3_BC_ATTN!=0`, `EXL3_BC_MLA!=0`; also built at load by autosplit_extra_measure | 1 on CUDA bsz1 | `modules/attention_fn/bc_mla.py:86, 138, 158, 303, 499, 547`; `mla_attn.py:989, 1020-1050, 1110-1118` | **avoidable**: `EXL3_BC_ATTN=0` | do not implement; force eager |
| `ext.TritonKernel` | bound class (cubin via CUDA driver API) | only inside `_compile_kernel` for BC paths | 1 on CUDA bsz1 | `modules/attention_fn/bc_attn.py:87` (binding in `exllamav3_ext/libtorch/attention_bc.h:1`) | avoidable with `EXL3_BC_ATTN=0` | do not implement |
| `ext.BC_Attention` | bound class | standard GQA attention only; GLM5Next text has none | 4 | `attention_fn/bc_attn.py` | unreached for text model | none |
| `ext.rope` | ext fn | MLA rope / indexer rope | 4 | `mla_attn.py:451` (`_indexer_rope_`, returns early: `self.rope is None`), `util/rope.py` via `self.rope.apply` (`mla_attn.py:828`, guarded by `self.rope is not None`) | NoPE: never reached | none |
| `ext.quant_cache_cont` / `ext.dequant_cache_cont` | ext fn | `-cq` quantized MLA cache only | 4 | `attention_fn/mla_triton.py:862`, `dsa_triton.py:859`, `modules/attn.py:35-36` | avoid `-cq` | none |
| `ext.paged_kv_cache_update` | ext fn | `CacheLayer_fp16` (MHA) appends | 4 | `cache/fp16.py:76` | MLA uses Triton `_mla_kv_update_kernel` instead | none |
| `ext.cache_rotate` | ext fn | generator page defrag; rotates every cache tensor incl. MLA `k`, `v`(0 bytes), `k_idx`, `k_pool` | 3 | `generator/pagetable.py:1025` | Part F owns it; listed here because it touches MLA planes | torch page copies (detail below) |
| `ext.dsa_topk_tile`, `ext.dsa_topk_merge_tiles` | ext fn | QSA indexer only (`modules/qsa_indexer.py`) | 4 | n/a | unreached | none |
| `ext.dspark_write_rows` | ext fn | DSpark / DSV4 caches (`cache/dsa.py`) | 4 | n/a | unreached | none |
| `ext.bighead_attn*` | ext fn | `attn_dispatch` fallback for huge head dims | 4 | `attention_fn/bighead_scalar.py` | unreached | none |
| `ext.rms_norm` (Part E op) | ext fn | `q_a_layernorm`, `kv_a_layernorm` | 1 | `mla_attn.py:441, 820` | n/a | per Part E |
| EXL3 linear ops (Part A op) | `BC_LinearEXL3` / exl3 gemm | `q_a_proj` (4096->1536), `q_b_proj` (1536->16384), `kv_a_proj_with_mqa` (4096->512), `o_proj` (16384->4096, out fp32), `indexer.wq_b` (1536->4096) | 1 | `mla_attn.py:438-442, 817, 900/927/961, 498/632` | n/a | per Part A |
| fp16 Linear (torch.matmul) | torch | `indexer.wk` (4096->128), `indexer.weights_proj` (4096->32), unquantized | 1 | `mla_attn.py:473, 502/633` via `modules/quant/fp16.py:94` (`torch.matmul` because x and out are fp16; `ext.hgemm` only on dtype mismatch) | n/a | torch.matmul as is |
| Triton `_mla_kv_update_kernel` | Triton | every MLA forward (cache append) | 1 | `cache/mla.py:115-122` -> `mla_triton.py:790-808`; nc path `mla_attn.py:1071` | always | keep Triton (or torch index_put) |
| Triton `_mla_plane_update_kernel` | Triton | every MLA forward (indexer key plane), and pool-plane append | 1 | `cache/mla.py:125-131, 142-147` -> `mla_triton.py:812-828` | always | keep Triton |
| Triton `_mla_absorb_kernel` | Triton | every step except dense MHA-form prefill | 1 | `mla_attn.py:840` -> `mla_triton.py:1022-1037` | always on decode | keep Triton (or torch bmm) |
| Triton `_mla_unfold_kernel` | Triton | every step except dense MHA-form prefill | 1 | `mla_attn.py:925, 959` -> `mla_triton.py:1040-1053` | always on decode | keep Triton (or torch bmm) |
| Triton `_mla_decode_split_kernel` + `_mla_decode_combine_kernel` | Triton | dense regime (ctx <= 2048), q_len <= 16 | 1 (short ctx) | `mla_attn.py:902-910` -> `mla_triton.py:876-1019` | n/a | keep Triton |
| Triton `_mla_gather_tile_kernel` + `_mla_prefill_mha_kernel` (+ 2x torch.mm) | Triton | dense regime prefill, chunk > 16 tokens (`EXL3_MLA_PREFILL` default "mha") | 3 (prefill of short prompts) | `mla_attn.py:891-899` -> `mla_triton.py:1056-1162` | n/a | keep Triton; fix block_m |
| Triton `_mla_prefill_kernel` | Triton | only `EXL3_MLA_PREFILL=absorbed` or non-causal | 4 | `mla_attn.py:902` | unreached by default | none |
| Triton `_dsa_indexer_kernel` / `_dsa_indexer_fewq_kernel` | Triton | sparse regime scoring; fewq when rows <= 4 | 2 | `mla_attn.py:680` -> `dsa_triton.py:1106-1177` | ctx > 2048 | keep Triton |
| Triton `_dsa_attn_split_kernel` + `_dsa_attn_combine_kernel` | Triton | sparse attention with R = bsz*seqlen <= 8 (decode) | 2 | `mla_attn.py:951-958` -> `dsa_triton.py:1004-1080` | ctx > 2048 | keep Triton |
| Triton `_dsa_attn_kernel` (one-shot) | Triton | sparse attention with R > 8 (prefill chunks) | 2 | same, `dsa_triton.py:1081-1104` | ctx > 2048 | keep Triton; reduce block_h on XPU |
| Triton `_mla_stage_kernel`, `_mla_idx_norm_kernel`, `_dsa_pool_update_kernel`, `_dsa_pool_expand_kernel` | Triton (BC only) | BC graph path | 4 with BC off | `bc_mla.py:188, 362, 381, 439` | `EXL3_BC_ATTN=0` | none (eager torch equivalents run instead) |

### (b) Per-op detail

#### `ext.dsa_topk(scores, indices, k, t_ptr, t_seq)`

- Binding: `bindings.cpp: m.def("dsa_topk", &dsa_topk)`; source `exllamav3_ext/dsa_topk.cu:925-935`
  (wrapper) -> `dsa_topk_gr` `:684-777`; kernels `dsa_topk_kernel<VEC>` `:69-276`,
  `dsa_topk_split_kernel` / `dsa_topk_merge_kernel` (`:278-680`).
- Signature: `scores` (R, T) **fp16**, innermost stride 1, any row stride (here a view of a
  fixed-stride backing, `s_stride = ceil(t_tile/128)*128`); `indices` (R, k_pad) **int32**,
  contiguous, written in place; `k` int <= k_pad; `t_ptr` optional device int tensor overriding T
  (always None from Python); `t_seq` 0.
- Python call (kpool path, `mla_attn.py:696`): `ext.dsa_topk(sc, pool_idx, k_sel, None, 0)`
  with `sc = dsa_indexer_scores(...)[:, :pools_slab]`, `k_sel = min(512, pools_slab)`,
  `pool_idx` (rows, ceil(k_sel/32)*32) int32, then sliced `[:, :k_sel]`.
- Exact semantics:
  - Key = 16-bit order-preserving map of the fp16 bits: `u & 0x8000 ? ~u : u | 0x8000`.
    Entries with key <= key(-inf) = 0x03ff are **excluded** (-inf and negative-sign NaNs;
    positive NaN 0x7e00 maps above +inf and would be selected as largest).
  - Selected set = the k largest finite fp16 scores; ties at the k-th value are broken by
    **lowest column index** (ordered compaction).
  - Output order: **ascending column index** (all entries > threshold in index order, followed
    by the needed ties == threshold in index order, so not globally sorted when ties are
    taken). Not sorted by score.
  - If fewer than k finite entries exist, all finite entries are emitted (index order).
  - Columns `[emitted, k_pad)` are filled with -1.
  - Split variant (R <= 16, T >= 32768, k_pad <= 4096, `EXL3_DSA_TOPK_SPLIT` != 0): identical
    result (per-span top-k superset then merge, same total order).
- Torch reimplementation: `s = scores.float().masked_fill(scores == -inf, -inf)`;
  `v, i = torch.sort(s, dim=1, descending=True, stable=True)` (stable gives lowest index first
  among ties); take `i[:, :k]`, drop where `v[:, :k] == -inf`; `torch.sort` the kept indices
  ascending; pad with -1 to k_pad. For speed `torch.topk(s, k)` + `-inf` mask is equivalent up to
  tie-breaking and order (attention is order-invariant up to fp summation order).
- The tiled merge in Python (`mla_attn.py:699-717`) uses `torch.topk` already; only taken when
  `pools_slab > t_tile = 32768` pools (> 131072 tokens), i.e. never at `-cs 131072`.

#### `ext.BC_MLAttention` (do not port; documented for reference)

- Binding `exllamav3_ext/libtorch/mla_attention_bc.h` (ctor with 29 args: dims, the
  `BC_LinearEXL3` handles of q/q_a/kv_a/o, norm weights, rope params, `w_uk_flat`,
  `w_uv_flat`, cache tensors, `xh`, `h32`; methods `set_indexer`, `needs_configure`,
  `configure_slot`, `configure_slot_dsa`, `run`). Source `libtorch/mla_attention.cpp`
  (`run_gr` `:307-818`, `run` `:819-989`).
- `run` executes the slot eagerly on its first call, captures a CUDA graph (`Graph` class,
  `graph.cu`) on the second, and replays it afterwards, patching pointer/scalar params
  (`GP_gemm_A`, `GP_attn_block_table`, `GP_attn_seqlens`, `GP_dsa_T`, ...). Kernels launched
  through `TritonKernel::launch` (CUDA driver `cuLaunchKernel`), EXL3 GEMMs through
  `exl3_gemm_gr`, fp16 GEMMs through `hgemm_gr` (cuBLASLt), top-k through `dsa_topk_gr`.
- Stage order (useful as a cross-check of the eager math, bsz 1 decode): q_a GEMM -> rms_norm(q_a)
  -> q_b GEMM -> kv_a GEMM -> `_mla_stage_kernel` (fp32 RMS norm of the latent: `x*w*rsqrt(mean(x^2)+eps)`)
  -> absorb -> cache append -> [indexer: copy x, hgemm wk, `_mla_idx_norm_kernel` (biased
  LayerNorm fp32), plane append (k at offset 0 of the 256-wide row), hgemm gate, plane append at
  offset 128, `_dsa_pool_update_kernel`] -> regime 0: decode split+combine; regime 1: wq_b GEMM,
  hgemm weights_proj, `_dsa_indexer_fewq_kernel` over the pool plane (T in pool units),
  `dsa_topk_gr(k = topk/P)`, `_dsa_pool_expand_kernel`, `_dsa_attn_split_kernel` +
  `_dsa_attn_combine_kernel` -> unfold -> o_proj GEMM.
- Differences vs eager: pool update is a Triton kernel that also (re)writes partial pools
  (never selected); eager writes only pools completed by the append. Selection and attention
  results are the same set.

#### `ext.TritonKernel(cubin: bytes, name: str, num_warps: int, shared_bytes: int)`

- `exllamav3_ext/triton_kernel.{h,cpp}`: `cuModuleLoadData` on the cubin, `cuModuleGetFunction`,
  sets `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` if > 48 KB; `launch()` appends two null
  trailing scratch pointers and calls `cuLaunchKernel(fn, grid, 32*num_warps threads, shared)`.
  Uses `cuda_drv.cpp` (dlopen'd driver API). Python side compiles with
  `triton.compile(ASTSource(...)).asm["cubin"]` under `torch.cuda.device` (`bc_attn.py:64-89`).
- No XPU equivalent needed: disabled by `EXL3_BC_ATTN=0`. (An XPU port would need SPIR-V/zebin
  from `asm["spv"]`/`asm["zebin"]` and a Level Zero launcher; not worth it for a correct port.)

#### `ext.cache_rotate(cache, order, temp)` (owned by Part F; MLA-relevant semantics)

- `exllamav3_ext/generator/cache.cu:65-97`. `cache` contiguous (num_pages, ...), any dtype,
  treated as bytes; `order` int32 (2n,) of (dst, src) page pairs; `temp` one page of bytes.
  For i in 0..n-1: `(cache[a] if a >= 0 else temp)[:] = (cache[b] if b >= 0 else temp)[:]`,
  sequentially. Early return if page_size == 0 (the MLA `kpe` tensor has 0 bytes, so `temp`
  must also be 0 bytes; torch handles it).
- MLA tensors passed by `Cache.get_all_tensors()`: `k` (pages,256,1,512) fp16, `v`
  (pages,256,1,0), `k_idx` (pages,256,256) fp16, `k_pool` (pages,64,128) fp16.

### (c) MLA + indexer layer forward, step by step (eager path, `EXL3_BC_ATTN=0`)

Entry: `MLAttention.forward` (`mla_attn.py:409-433`) -> `decode_flash_attn` (`:964-1017`) with
`params["attn_mode"] == "flash_attn"` (generator/warmup/load) or `decode_flash_attn_nc`
(`:1053-1086`) for cache-less forwards. Input `x` (bsz, seqlen, 4096), the post-attn_norm
hidden (fp16). Output (bsz, seqlen, 4096) **fp32** (`out_dtype = torch.float`, glm5_next.py:197).

Notation: R = bsz*seqlen, H = 64, D_c = 512, D_nope = D_v = 256, H_i = 32, D_i = 128, P = 4.

0. `host_seqlens` = host list of `cache_seqlens` (pre-append lengths), cached in
   `params["_mla_host_seqlens"]` (`:41-55`); free when the generator passes CPU tensors.
   `sparse = max(host_seqlens) + seqlen > 2048` (`:796-802`).
1. Query: `q_resid = RMSNorm_{q_a_layernorm}(EXL3(q_a_proj, x))` (fp16, R x 1536);
   `q = EXL3(q_b_proj, q_resid)` -> view (R, 64, 256) = q_nope (q_pe is empty).
2. Latent: `ckv = RMSNorm_{kv_a_layernorm}(EXL3(kv_a_proj_with_mqa, x))` -> (bsz, seqlen, 512)
   fp16. (kv_a output width is exactly 512, no rope part.) k_pe is (bsz, seqlen, 1, 0).
3. If not MHA-form prefill (`use_mha = seqlen > 16 and EXL3_MLA_PREFILL=="mha" and causal and not sparse`, `:833`):
   absorb `q_lat[h, r, :] = q_nope[r, h, :] @ W_UK[h]` where `W_UK[h] = w_uk_flat[:, h*256:(h+1)*256]^T`
   i.e. `q_lat[h,r,c] = sum_d q[r,h,d] * w_uk_flat[c, h*256+d]`, fp32 accumulate, stored fp16,
   layout (H, R, 512) head-major (`_mla_absorb_kernel`, `mla_triton.py:555-591`).
   `w_uk_flat (512, 64*256)` / `w_uv_flat (512, 64*256)` are built at load from
   `kv_b_proj.weight` (64*(256+256), 512) fp16 (unquantized; fp8 checkpoints get the block
   inverse scale applied) as `w.view(64, 512_rows=256+256, 512)`: rows [0,256) per head -> W_UK,
   rows [256,512) -> W_UV, permuted to (D_c, H, dim) (`mla_attn.py:368-382`).
4. Cache append (`append(ckv, k_pe)` -> `CacheLayer_MLA_fp16.update_kv_direct`,
   `cache/mla.py:110-122` -> `_mla_kv_update_kernel`): for row (b, i):
   `abs = cache_seqlens[b] + i; page = block_table[b, abs // 256]; ckv_cache[page, abs % 256, 0, :] = ckv[b, i]`.
5. Indexer keys, every forward, both regimes (`_indexer_keys`, `:467-486`):
   `k = LayerNorm(x @ wk^T; weight, bias, eps=1e-6)` (F.layer_norm in fp16 -> fp16, 128 wide);
   no rope; `gate = F.linear(x.half(), idx_kpool_gate)` (gate weight (128, 4096) bf16->fp16 from
   `kpool_aux.safetensors`) -> 128 wide; plane row = `cat(k, gate)` (256 wide fp16) appended to
   `k_idx` plane at the token's slot (`update_idx_direct`, `_mla_plane_update_kernel`).
6. Pool plane update (`_update_pool_plane`, `:581-613`), per batch row b: pools
   `p in [pos0//4, (pos0+seqlen)//4)` (those completed by this append):
   members j = 0..3 = tokens 4p+j read back from the `k_idx` plane (fp32),
   `probs[j, d] = softmax_j(gate[4p+j, d] + ape[j, d])` (ape (4,128) bf16 -> **fp32**),
   `pool_k[p, d] = sum_j probs[j, d] * k[4p+j, d]` -> fp16, appended to `k_pool`
   (pages, 64, 128) with page = `block_table[b, p // 64]`, slot `p % 64`
   (`update_pool_direct` -> `_mla_plane_update_kernel` with `cache_seqlens := [first_pool]`).
7. Dense regime (visible ctx <= 2048):
   - seqlen <= 16 (decode / short chunk): `o_lat = mla_attn_triton_decode(q_lat, q_pe_hm, ...)`:
     for query row (b, i) at absolute position `q_abs = cache_seqlens[b] + seqlen - seqlen + i`
     (cache already holds the chunk; `pre_appended_len = seqlen`), keys n in `[0, total)`, causal
     `n <= q_abs`: `s = (q_lat[h,r] . ckv[n]) * 0.0625` (fp16 dot, fp32 acc),
     online softmax fp32, `o_lat[h,r] = sum_n p_n ckv[n] / sum_n p_n` (P cast to fp16 before the
     PV dot), split-K flash-decoding (`num_splits = min(2*SM/programs (cap 128), ceil(max_k_len/128))`,
     combine kernel merges with max-rescaling; fully masked rows give 0).
     Then unfold `o[r, h, :] = o_lat[h, r, :] @ w_uv_flat[:, h*256:(h+1)*256]` (fp32 acc, fp16
     out, token-major (R, 64, 256)) and `o_proj`.
   - seqlen > 16 (prefill chunk): MHA form (`mla_attn_triton_prefill_mha`): for each batch row,
     tiles of 4096 cache rows `[a, e)` over `[0, total)`: gather latent tile (fp16), `K_nope = tile @ w_uk_flat`,
     `V = tile @ w_uv_flat` (torch.mm fp16, (tile, 16384)), then per head standard causal
     attention `softmax(q_nope_h . K_nope_h^T * 0.0625) V_h` with running (m, l, acc) state in
     global fp32 buffers across tile launches (same-stream ordering); output token-major ->
     `o_proj`. Mathematically identical to the absorbed form.
8. Sparse regime (visible ctx > 2048), `_indexer_topk_kpool` paged path (`:617-728`):
   - `q_idx = EXL3(idx_wq_b, q_resid)` -> (bsz, seqlen, 32, 128) fp16 (no rope);
     `w = x @ weights_proj^T` (bsz, seqlen, 32) fp16, raw.
   - Per batch row b, per slab of 256 query rows: `pools_slab = (pos_b + r1) // 4` complete
     pools visible to the slab's last row, `k_sel = min(512, pools_slab)`.
     Scores over pools s in [0, pools_slab):
     `score[r, s] = (128^-0.5 * 32^-0.5) * sum_h w[r,h] * relu(q_idx[r,h] . pool_k[s])`
     (fp16 dot, fp32 accumulate over heads, stored **fp16**), with the causal bound
     `s < (pos_b + r0 + r + 1) // 4` (complete pools only; others -inf).
     Kernel: `_dsa_indexer_fewq_kernel` if rows <= 4 else `_dsa_indexer_kernel`
     (block_m 64, block_n 128, 8 warps), reading `k_pool` through the block table
     (`epp = 64` entries per page, row = `bt[s // 64]*64 + s % 64`).
   - `ext.dsa_topk(scores, pool_idx, k_sel)` -> pool ids; expand
     `tok = pool*4 + [0,1,2,3]` (-1 propagates) into columns `[0, 4*k_sel)`.
   - Tail (`append_tail`, `:645-657`): for query position q (absolute), `vis = q+1`,
     `tail_count = vis % 4`, tokens `[vis - tail_count, vis)` are appended raw in columns
     `[base_cols, base_cols+3)`, rest -1. Note: when `vis % 4 == 0` the query's own token
     sits in a complete pool and is attended only if that pool is selected; there is no forced
     "most recent pool" selection beyond the tail.
   - `indices` (R, k_pad) int32, k_pad = ceil((512*4 + 3)/32)*32 = **2080**, -1 padded.
     Indices are absolute token positions within the sequence (pre-page-table).
   - Sparse attention (`_attend_sparse`, `:930-961` -> `dsa_attn(q_lat, ckv_cache, kpe_cache, bt, indices=..., q_pe=(R,H,0), out_latent=True)`):
     for each row r, head h: over selected entries n (idx >= 0): page lookup
     `tok = bt[row, idx // 256]*256 + idx % 256`, `s = (q_lat[h,r] . ckv[tok]) * 0.0625`, online
     softmax fp32, `o_lat[h,r] = sum p ckv[tok] / sum p` (fp32 acc, P cast to fp16), no causal
     mask (selection is causal by construction). Block table: shared single row when bsz==1 or
     seqlen==1 (stride 0 trick), else `repeat_interleave(seqlen)`. R <= 8 -> split (16 splits if
     k_len > 256) + combine kernels, block_h 16; R > 8 -> one-shot kernel block_h 32,
     block_n 32, num_stages 3. Output head-major (H, R, 512) -> unfold -> o_proj.
9. `flash_attn_nc` (cache-less) variant: scratch pages `ckv (bsz*pages, 256, 1, 512)` with
   identity block table, `cache_seqlens = 0`, `host_seqlens = [0]*bsz`; the indexer uses the
   eager cache-less branch (`:730-760`) on the packed chunk rows: pools built with
   `torch.softmax`, scores via einsum (`relu(q.k * D_i^-0.5)` then `w * H_i^-0.5`, fp32), causal
   mask `pool_end = 4p+3 <= q_pos`, `torch.topk(min(512, n_pools))`, expand + tail. No
   `ext.dsa_topk` on this branch.

State kept per MLA layer in the cache (`cache/mla.py:13-89`, all fp16, allocated zeroed at load,
pages = cache_size/256 = 512 for `-cs 131072`):

| tensor | shape | bytes/token | contents |
|---|---|---|---|
| `k` (ckv) | (pages, 256, 1, 512) | 1024 | kv_a_layernorm-normalized latent |
| `v` (kpe) | (pages, 256, 1, 0) | 0 | NoPE: empty |
| `k_idx` | (pages, 256, 256) | 512 | [LayerNorm'd indexer key (128) or gate scores (128)] |
| `k_pool` | (pages, 64, 128) | 64 | pooled indexer keys, one per 4 tokens |

Total 1600 B/token/layer, 11 layers = 17.6 KB/token, 2.31 GB at 131072 tokens. No per-sequence
recurrent state for MLA layers; `params["dsa_topk_indices"]` is set per full layer
(`:869`) but only consumed by "shared" layers (none in GLM5.3).

### (d) Triton kernel inventory (MLA slice)

Files: `exllamav3/modules/attention_fn/mla_triton.py`, `dsa_triton.py`, helpers imported from
`triton_paged.py` (`_qc_load_kt`, `_qc_load_v`, `_rot_h32`, `_get_h32`; only in QC branches).

| kernel | file:line | reached | grid / config | XPU notes |
|---|---|---|---|---|
| `_mla_kv_update_kernel` | mla_triton.py:113 | yes | (R,), 4 warps, 2 stages; `tl.arange(0, 512)` | trivial; D_r=0 branch dead |
| `_mla_plane_update_kernel` | mla_triton.py:149 | yes | (R,), 2 warps; D=256 (idx) or 128 (pool) | trivial |
| `_mla_absorb_kernel` | mla_triton.py:555 | yes | (ceil(R/bm), 4, 64), bm = min(64, pow2(max(R,16))), BLOCK_N 128, NOPE_PAD 256, 4 warps | `tl.dot` (bm x 256)·(256 x 128) fp16 |
| `_mla_unfold_kernel` | mla_triton.py:594 | yes | (ceil(R/bm), 64), BLOCK_K 128, acc (bm, 256) fp32 | fine |
| `_mla_decode_split_kernel` | mla_triton.py:205 | yes (dense) | (bsz*ceil(64/block_h), num_splits); block_m = pow2(q_len), block_h = max(16//block_m,1), BLOCK_ROWS 16, block_n 32, 4 warps, 3 stages; acc (16, 512) fp32 | `tl.trans` of (32, 512) tile; uses `tl.cdiv`, `tl.minimum`; num_stages=3 pipelining is a CUDA cp.async concept, Intel backend ignores/handles stages differently; SM-count split heuristic needs XPU property |
| `_mla_decode_combine_kernel` | mla_triton.py:346 | yes when splits > 1 | (programs,), 4 warps | fine |
| `_mla_gather_tile_kernel` | mla_triton.py:506 | yes (dense prefill) | (ceil(tile/64),) | fine |
| `_mla_prefill_mha_kernel` | mla_triton.py:665 | yes (dense prefill) | (ceil(q_len/block_m), 64); block_m 128 (halved on `triton.runtime.errors.OutOfResources`), block_n 64, 8 warps, 2 stages; q tile (128,256) fp16 + acc (128,256) fp32 | heavy register/SLM use; the OutOfResources retry loop (`mla_triton.py:1148-1160`) relies on a launch-time exception; on XPU prefer setting block_m 64 or 32 explicitly; same-stream RMW of state buffers needs an in-order queue (torch XPU default is in-order) |
| `_mla_prefill_kernel` | mla_triton.py:399 | no (`EXL3_MLA_PREFILL=absorbed` only) | - | - |
| `_mla_stage_kernel`, `_mla_idx_norm_kernel` | mla_triton.py:626, 180 | BC only | - | - |
| `_mla_kv_quant_scatter_kernel` | mla_triton.py:72 | `-cq` only | - | - |
| `_dsa_indexer_kernel` | dsa_triton.py:601 | yes (sparse, > 4 rows) | (ceil(R/64), ceil(T/128)), 8 warps, 2 stages; head loop of 32 `tl.dot` (64x128)·(128x128) | `debug=dsa_debug_bounds` kwarg on `@triton.jit` and `tl.device_assert` only when `EXL3_DSA_DEBUG_BOUNDS=1`; S_stride is a constexpr (fixed 32768) |
| `_dsa_indexer_fewq_kernel` | dsa_triton.py:662 | yes (sparse, <= 4 rows, i.e. decode) | (R, ceil(T/128)); H_pad = 32; dot (32x128)·(128x128) | early `return` inside kernel (supported) |
| `_dsa_attn_split_kernel` | dsa_triton.py:286 | yes (sparse, R <= 8) | (R*4, 16 splits), BLOCK_H 16, BLOCK_N 32, D_c_pad 512, 4 warps, 2 stages | `tl.num_programs`; window branch (`tl.range(..., num_stages=1)`) dead |
| `_dsa_attn_combine_kernel` | dsa_triton.py:505 | yes (with split) | (R*4, 4), BLOCK_D 128 | `tl.split/tl.interleave/tl.cos/tl.sin` only under DEROTATE (dead) |
| `_dsa_attn_kernel` | dsa_triton.py:45 | yes (sparse, R > 8: prefill) | (R*2,), BLOCK_H 32, BLOCK_N 32, 4 warps, 3 stages; acc_c (32, 512) fp32 = 64 KB per program | likely heavy register spill on Xe2; use block_h 16 (or Intel `grf_mode="large"`) on XPU |
| `_dsa_pool_update_kernel`, `_dsa_pool_expand_kernel` | dsa_triton.py:733, 792 | BC only | - | - |

General XPU-Triton notes for these kernels: all dots are fp16 x fp16 -> fp32 with M,N,K >= 16
(supported by Intel XMX/DPAS lowering); no fp8, no tf32, no libdevice, no inline asm, no
`tl.extra.cuda`, no atomics. 32-bit index arithmetic is sufficient at 131072 tokens
(largest: MHA state 64*8192*256 = 134M elements). `num_warps` values (2/4/8) are interpreted
by Intel Triton as sub-group counts; correctness is unaffected, tuning will differ.
Triton JIT specialization keys include `do_not_specialize` lists (supported by Intel Triton).

### (e) Non-ext CUDA dependencies in this part

| file:line | dependency | XPU mapping |
|---|---|---|
| `attention_fn/mla_triton.py:803, 823, 865, 977, 1029, 1045, 1127, 1210` | `with torch.cuda.device(t.device):` around launches | `torch.xpu.device(...)` or a no-op context (single GPU); simplest: alias `torch.cuda.device = torch.xpu.device` in a compat shim |
| `attention_fn/dsa_triton.py:1052, 1082, 1153` | same | same |
| `attention_fn/mla_triton.py:786`, `bc_attn.py:96`, `triton_paged.py:1203, 1886` | `torch.cuda.get_device_properties(i).multi_processor_count` (split heuristics) | `torch.xpu.get_device_properties(i).gpu_subslice_count` (Xe-cores, 32 on B70) |
| `attention_fn/mla_triton.py:63` | `torch.cuda.synchronize(device)` under `EXL3_MLA_DEBUG_SYNC` | `torch.xpu.synchronize` (debug only) |
| `attention_fn/bc_attn.py:84-87` | `torch.cuda.device`, `triton.compile(...).asm["cubin"]`, `ext.TritonKernel` | disabled by `EXL3_BC_ATTN=0` |
| `attention_fn/triton_paged.py:290, 1693-1694, 1852` | `.is_cuda`, `torch.cuda.Event` timing autotune, `get_device_capability` | not reached by MLA (generic attention backends) |
| `modules/mla_attn.py:1297`, `modules/attn.py:1368` | `torch.cuda.synchronize()` in `tp_import` | TP only, unreached |
| `cache/mla.py:160-167` | `copy_(..., non_blocking=True)` page copies (prompt-cache partial page reuse) | works on XPU as is |
| `util/device_copy.py:39-40, 81` | `torch.cuda.synchronize` for cuda->cuda peer copies; `t.device.type == "cuda"` check | single device: the cuda->cuda branch is never taken; host->device `t.to(dev, non_blocking=...)` works on XPU |
| `util/tensor.py:158-163` (`get_for_device`) | `v.is_pinned()` -> non_blocking H2D of pinned host `cache_seqlens`/`block_table` | pinned memory on XPU builds: `pin_memory()` targets the current accelerator in torch >= 2.5; verify the generator's pinned staging tensors are pinned for XPU (Part F) |
| `modules/attention_fn/mla_triton.py:1148-1160` | `triton.runtime.errors.OutOfResources` retry loop | check the exception type raised by the Intel backend; pin block_m instead |

No CUDA graphs, streams, or events are used on the eager MLA path. CUDA graphs only exist inside
`BC_MLAttention` (`Graph` in `exllamav3_ext/graph.cu`), disabled with `EXL3_BC_ATTN=0`.

### Minimal set for a correct MLA forward on XPU

- Env: `EXL3_BC_ATTN=0` (covers `EXL3_BC_MLA`), keep default `EXL3_MLA_PREFILL=mha`, no `-cq`.
- Compat shim: `torch.cuda.device` -> `torch.xpu.device`; SM count -> XPU subslice count.
- ext: `dsa_topk` (torch sort/topk). Plus the Part A EXL3 linear and the Part E `rms_norm`.
- Triton kernels as listed "yes" above, unchanged except launch configs (prefill MHA block_m,
  one-shot DSA block_h). If any Triton kernel fails on XPU, each has a direct torch equivalent
  given in (c): dense attention = gather pages + `softmax((q_lat @ ckv^T) * 0.0625, causal)` @ ckv;
  sparse = same over gathered `indices` rows; absorb/unfold = per-head bmm; appends = index_put.


---

## Part E. Hyper-connections, norms, TransformerBlock, model forward / load / warmup, loader, graphs

All paths are relative to `glm/exllamav3`. Python files are under `exllamav3/`, and
CUDA files are under `exllamav3/exllamav3_ext/`. The line numbers are for v1.5.1 (commit 958ec93).

### (a) Op table for this part

"Calls/fwd" means calls per model forward, the same for prefill chunks and decode steps. GLM has 45 blocks and 2 HC
sites per block, so there are 90 HC sites.

| op | category | when reached | calls/fwd | Python call site(s) | condition / avoidable? | recommended XPU path |
|---|---|---|---|---|---|---|
| `hc_mix` | mHC mixing | every forward (prefill, decode, warmup, autosplit measure) | 90 | `modules/hyperconnections.py:152` (`HyperConnection.mix`) | Taken when `hc==4 and streams.dtype==float and D%4==0 and streams.is_contiguous()` (`:127`). These always hold for GLM. Otherwise the torch fallback at `:155-170` runs. **Avoidable**: if the shim leaves `hc_mix` out, nothing falls back automatically, because the branch is not gated on ext availability. Either patch `:127` to `False` or provide `hc_mix` in the shim. | Implement `hc_mix` in the shim with torch (reference below), writing into the given `post`/`comb`/`collapsed` out tensors. Later, a single Triton kernel. |
| `hc_mix_num_chunks` | mHC helper (host int) | before every `hc_mix` | 90 | `modules/hyperconnections.py:130` | Host-side arithmetic only. | Shim: port the formula (below) or return 1. The torch `hc_mix` ignores the `partials` workspace. |
| `hc_apply` | mHC residual write-back (in place) | every forward | 90 | `modules/hyperconnections.py:190` (`HyperConnection.apply_`) | Taken unless converting (`"quant_preserve"`/`"capture"` in params) and when the dtype/contiguity checks at `:186-188` hold. These always hold for GLM inference. The torch fallback at `:192` allocates a new tensor, which the caller assigns (`x = ...apply_(...)`), so it is correct. | Shim: in-place torch `x.copy_(post[...,None]*y + comb^T @ x)` (below). |
| `rms_norm` | RMSNorm | `input_layernorm` and `post_attention_layernorm` in every block, plus the final `norm`. Also used inside MLA/GDN (see Parts C and D). | 91 here | `modules/rmsnorm.py:143` (the x.dim()>2 path; this is the GLM case because y is (b,s,4096)) and `:158` (2-D or span_heads) | Always taken. `RMSNorm.forward_torch` (`:71`) exists but nothing calls it. | Shim: torch implementation (below). Alternative: monkeypatch `RMSNorm.forward` to use `forward_torch`, but that rounds differently (see detail). |
| `rms_norm_res_in` | fused residual add + norm | **not reached for GLM** | 0 | `modules/rmsnorm.py:121` | Only used when there is no HC and `mlp_norm.can_fuse_residual` (`transformer.py:168,182`). GLM always has `attn_hc`. | Skip, or a trivial torch version. |
| `hc_head` | mHC weighted collapse | **not reached** | 0 | `modules/hyperconnections.py:656` | GLM builds `HyperHead(mean=True)` (`architecture/glm5_next.py:320-327`), and `forward` returns `x.mean(dim=2)` at `:641`. | none |
| `gr_mix`, `gr_mix_tiled`, `gr_mix_tiled_slices`, `det_quant_weight` (HC use) | GatedResidual (Qwen4Exp) | **not reached** | 0 | `modules/hyperconnections.py:340,343,427,436,440` | Only in the `GatedResidual` class, which GLM does not use. `det_quant_weight` is also called from `block_sparse_mlp_routing.py`; that use belongs to Part B. | none (for this part) |
| `add` / `add_gr` | bias add | **not reached** | 0 | no Python call site. C++ BC classes call `add_gr` only for biases (`libtorch/linear.cpp:19,48`, `gated_delta_net.cpp`, `mlp.cpp`). | GLM has no biases (`attention_bias=False`). | none |
| `stloader_open_file`, `stloader_read`, `stloader_close_file`, `TensorLoadJob`, `stloader_deferred_cpu`, `stloader_deferred_cuda` | weight loading | model load only | n/a | `loader/safetensors.py:762,803,820,876,886,938,1001,1008` | Used when `load_method` is `"mt_fread"` (the default, `:413`). **Fully avoidable**: with `load_method="python"` (`:839-857`; handles stay None from `:446`), no `ext` call is made, because nothing is ever deferred and no handles are opened. | Force `load_method="python"`, or replace the loader with `safetensors.safe_open(..., device="xpu:0")`. **Do not** let the default run on XPU (see the CUDA-deps entry for `safetensors.py:793`). |
| `cuda_host_register` / `unregister` / `get_device_pointer`, `cuda_device_get_attribute` | pinned host mapping | **not reached** (TP and `vision_pinned` only) | 0 | `model/model_tp_cuda.py:20-…`, used by `model_tp_backend.py:363` and `model_tp_shared.py:177,192` | TP only. | none |
| `pinned_cuda_view` | zero-copy alias | **not reached** | 0 | `modules/linear.py` (`pin_linears`) | Only with `EXL3_VISION_PINNED` and the vision component (`model/model_ls.py:38,176`). | none |
| `exl3_gemv_int8_max_k` | device query used in a config decision | load (each `use_mgemm` decision) | n/a | `model/config.py:76` | Guarded by `device.type == "cuda"` (`:74`), so on XPU it is skipped and `K_thr` stays at the env default (6). The mgemm/gemv choice belongs to Part A. | No shim needed for XPU devices. Return 6 if the shim is ever asked. |
| `TritonKernel` (bound class) | Triton cubin launcher used by the BC graphs | inside the BC attention graph | n/a | `modules/attention_fn/bc_attn.py` (Part D) | Lives only inside BC graph paths. | Don't implement. Disable the BC paths. |
| Graph (`graph.cuh`, C++ only, **not bound to Python**) | CUDA-graph capture/replay inside `BC_*` objects | decode in the BC classes | n/a | none in Python. It is owned by `BC_LinearEXL3`, `BC_GatedDeltaNet`, `BC_MLAttention`, `BC_BlockSparseMLP`, `BC_GatedMLP`, `BC_GatedRMSNorm`, and so on. | Avoid it by not exposing `BC_*` classes from the shim, and by setting env switches such as `EXL3_BC_GDN=0` (`modules/gated_delta_net.py:19`), `EXL3_BC_MLA` (`modules/mla_attn.py:26`) and `EXL3_BC_ATTN` (`attention_fn/bc_attn.py:37`, `bc_mla.py:20`). Parts A–D give the exact per-module gates. | none |
| `rope`, `gen_mrope_pos_ids` (`util/rope.py`) | RoPE | **not reached from this part**. GLM is NoPE (`rope_settings=None`, `glm5_next.py:57,211`). | 0 | `util/rope.py` | Any indexer RoPE belongs to Part D. | none |

### (b) Per-op detail

#### `hc_mix_num_chunks(int R, int row_len) -> int`  (`hc_mix.cu:894`)
```
chunks_a   = min(128, max(1, 512 / max(R,1)))            // integer division
chunk_cols = ceil((row_len / chunks_a) / 256) * 256        // round up to a multiple of 4*NUM_THREADS_A = 256
return ceil(row_len / chunk_cols)
```
For GLM, row_len = 4·4096 = 16384. R=1 gives 64 chunks. R=8 gives 64. R=8192 gives 1. The value only sizes the
`partials` workspace `(R, chunks, M+1)`. A torch shim may ignore the workspace.

#### `hc_mix(streams, fn, base, scale, rms_eps: float, hc_eps: float, sinkhorn_iters: int, partials, post, comb, collapsed) -> None`
Bindings: `bindings.cpp` `m.def("hc_mix", &hc_mix)`. Declared in `hc_mix.cuh:12`. Implemented in `hc_mix.cu:901`, with the
launcher at `:807`, kernels at `:41` (partials) and `:118` (finalize).

Arguments (all positional, and the checks are enforced):
- `streams`: `(R, H=4, D=4096)` **fp32**, contiguous. The Python caller passes `streams.view(R,H,D)`.
- `fn`: `(M=24, H*D=16384)`. This is **fp32 when R > 32, and fp16 when R ≤ 32**. Python caches `self.fn_h = self.fn.half()`
  (`hyperconnections.py:146-151`), and the kernel dots in fp32 either way. A bit-faithful port must upcast the fp16 `fn` it
  was given, not use the fp32 original.
- `base`: `(24,)` fp32. `scale`: `(3,)` fp32.
- `rms_eps` = `rms_norm_eps` = 1e-5. `hc_eps` = 1e-6. `sinkhorn_iters` = 20.
- Outputs, all written in place: `partials` `(R, chunks, 25)` fp32 (scratch), `post` `(R,4)` fp32, `comb` `(R,4,4)` fp32,
  `collapsed` `(R,4096)` **fp16** (the Python fused path always allocates half, `:143`). The kernel also accepts fp32
  `collapsed`.
- Workspaces: for R ≤ 32 they come from the static `g_tensor_cache.get_bucketed` shared by every site (`:136-139`). For
  larger R they are fresh `torch.empty` allocations.

Checkpoint tensors per site:
- Key `model.language_model.layers.{i}.hc_attn` or `.hc_ffn`.
- Tensors `{key}_fn` `(24, 16384)`, `{key}_base` `(24,)`, `{key}_scale` `(3,)`. They are loaded as `.float()` (`hyperconnections.py:91-93`).
- The unweighted `{key}.norm` RMSNorm submodule has no tensor and is used only by the torch fallback.
- `fn` row layout: rows 0–3 are `pre`, rows 4–7 are `post`, rows 8–23 are `comb`, with comb element (i,j) at row `8 + 4*i + j`.
- `fn` column index is `h*4096 + d`, so the flattened streams are stream-major.

Exact math, per row r. Everything is fp32. The CUDA build uses `--use_fast_math`, `__expf` and `__fdividef`.
```
s      = streams[r].reshape(16384)                       # fp32
ss     = sum(s*s)                                        # fp32, chunked tree reduce, deterministic order
mix[m] = sum(s * fn[m].float())   m=0..23                # dot on the RAW (un-normalised) streams
rmr    = rsqrt(ss / 16384 + rms_eps)                     # == unweighted RMSNorm applied after the (linear) projection
pre[h]  = sigmoid(mix[h]   * rmr * scale[0] + base[h])     + hc_eps          # fmaf(mix*rmr, scale, base)
post[h] = 2 * sigmoid(mix[4+h] * rmr * scale[1] + base[4+h])
L[i,j]  = mix[8+4i+j] * rmr * scale[2] + base[8+4i+j]
C = softmax(L, dim=-1)  (over j, max-subtracted) + hc_eps
C = C / (C.sum(dim=-2, keepdim) + hc_eps)                 # column normalise (sum over i)
repeat sinkhorn_iters-1 (=19) times:
    C = C / (C.sum(dim=-1, keepdim) + hc_eps)             # row normalise
    C = C / (C.sum(dim=-2, keepdim) + hc_eps)             # column normalise
comb[r] = C          # stored row-major comb[r, i, j]
collapsed[r, d] = sum_h pre[h] * streams[r, h, d]          # fmaf in h order 0..3, then round-to-nearest to fp16
```
The torch fallback (`hyperconnections.py:155-170`) is identical math. It normalises first and then projects, and it
keeps `collapsed` in fp32. This equivalence is exact in real arithmetic and differs only in fp32 rounding.

Torch shim:
```python
def hc_mix(streams, fn, base, scale, rms_eps, hc_eps, iters, partials, post, comb, collapsed):
    R, H, D = streams.shape
    s = streams.reshape(R, H * D)
    mix = s @ fn.float().t()                                   # (R, 24)
    rmr = torch.rsqrt(s.pow(2).sum(-1, keepdim=True) / (H * D) + rms_eps)
    mix = mix * rmr
    pre = torch.sigmoid(mix[:, :H] * scale[0] + base[:H]) + hc_eps
    post.copy_(2 * torch.sigmoid(mix[:, H:2*H] * scale[1] + base[H:2*H]))
    c = (mix[:, 2*H:] * scale[2] + base[2*H:]).view(R, H, H)
    c = torch.softmax(c, -1) + hc_eps
    c = c / (c.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdim=True) + hc_eps)
        c = c / (c.sum(-2, keepdim=True) + hc_eps)
    comb.copy_(c)
    collapsed.copy_((pre.unsqueeze(-1) * streams).sum(1))        # copy_ rounds to collapsed.dtype (fp16)
```
Performance note: about 20 small launches times 90 sites per token. It is fine for correctness. A one-kernel Triton
version is the obvious follow-up, and `glm-flash-lite/glm53/k_hcfuse.py` (described below) shows the fusion target.

#### `hc_apply(x, y, post, comb: Optional[Tensor]) -> None`  (`hc_mix.cu:962`, kernel `:723`)
Arguments:
- `x`: `(R,4,4096)` fp32 contiguous, **updated in place**.
- `y`: `(R,4096)` fp32 or fp16 contiguous. For GLM it is fp32, because the attention and MLP modules use `out_dtype=torch.float`.
- `post`: `(R,4)` fp32.
- `comb`: `(R,4,4)` fp32, or `None`. Only GatedResidual passes `None`.

Math, all in fp32:
```
x_new[r, h, d] = post[r,h] * y[r,d] + sum_g comb[r, g, h] * x_old[r, g, d]      # i.e. x_new = post⊗y + combᵀ·x
(comb None: x_new[r,h,d] = post[r,h]*y[r,d] + x_old[r,h,d])
```
Each thread reads all 4 stream values of a column before it writes any of them, which is what makes the in-place
update safe. The FMA order is: start from `post*y`, then add `fmaf(comb[g][h], x[g])` for g = 0..3.

Torch shim:
```python
def hc_apply(x, y, post, comb):
    yf = y.float().unsqueeze(1)
    if comb is None:
        x.add_(post.unsqueeze(-1) * yf)
    else:
        x.copy_(post.unsqueeze(-1) * yf + torch.matmul(comb.transpose(-1, -2), x))
```

#### `hc_head(...)` (`hc_mix.cu:933`)
Not reached. If it ever were reached, it is the finalize kernel with M=4: `collapsed = sum_h (sigmoid(mix[h]*rmr*scale[0]+base[h])+hc_eps) * streams[h]`.

#### `rms_norm(x, w: Optional[Tensor], y, epsilon: float, constant_bias: float, constant_scale: float, span_heads: bool, add_residual: bool, w_groups: int = 1) -> None`
Bindings: `bindings.cpp` with py::args `x,w,y,epsilon,constant_bias,constant_scale,span_heads,add_residual,w_groups=1`.
Implemented in `norm.cu:448`, with `rms_norm_impl` at `:318` and the kernel at `:159`.

GLM calls it as `ext.rms_norm(x_2d, weight, y_2d, 1e-5, 0.0, 1.0, False, False, 1)`:
- `x_2d` is `(R,4096)` **fp16**. This is the `collapsed` output of `hc_mix`, and `y.half()` is a no-op.
- For the final `norm`, `x_2d` is **fp32**, the HyperHead mean.
- `y_2d` is fp16. The final norm also has `out_dtype=torch.half` (`glm5_next.py:332`).

Allowed dtypes:
- x: fp16 or fp32.
- w: fp16 or bf16. It is loaded with `float2half=True, allow_bf16=True` (`rmsnorm.py:54`), so a bf16 checkpoint weight stays bf16 and an fp32 one becomes fp16.
- y: fp16 or fp32. **No bf16 output.**
- The dim must be divisible by 4.

Exact math, per row:
```
xf = x.float();  if x is fp16: xf = clamp(xf, -65504, 65504)          # CLAMP_FP16 on read (turns ±inf into ±65504)
rmf = rsqrt(sum(xf*xf) / dim + epsilon) * constant_scale               # fp32, eps INSIDE the sqrt, mean over dim
out = (xf * (w.float() + constant_bias)) * rmf                          # w None -> xf * rmf ; order: (x*w)*rmf
add_residual (res_mode POST): out += y_old.float()                     # y read then overwritten
y = out rounded RN to y.dtype
w_groups>1: w row = (row % w_groups);   span_heads: x,y flattened over the last two dims first
```
Do not reuse `forward_torch` (`rmsnorm.py:71`). It rounds to the input dtype before multiplying by the weight, so it is
not bit-identical to the kernel. A shim that follows the formula above is the correct reference.

Torch shim:
```python
def rms_norm(x, w, y, eps, cbias, cscale, span_heads, add_residual, w_groups=1):
    xs, ys = (x.flatten(-2), y.flatten(-2)) if span_heads else (x, y)
    xf = xs.float()
    if xs.dtype == torch.half:
        xf = xf.clamp(-65504, 65504)
    rmf = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * cscale
    if w is not None:
        wf = w.float()
        if w_groups > 1:
            wf = wf.view(w_groups, -1).repeat(xf.shape[0] // w_groups, 1)   # row % groups
        xf = xf * (wf + cbias)
    out = xf * rmf
    if add_residual:
        out = out + ys.float()
    ys.copy_(out)
```

#### `rms_norm_res_in(x, w, y, r, epsilon, constant_bias, constant_scale)` (`norm.cu`, RES_IN)
Not reached for GLM. Semantics: `r = round_to_r_dtype(x + r)` (written back), then `y = norm(r) * w`.

#### stloader ops (`stloader.cpp`, `stloader.h`, `stloader_cu.cu`)
Signatures:
- `stloader_open_file(const char*) -> list[uintptr]` (8 per-thread handles).
- `stloader_close_file(list)`.
- `stloader_read(handles, offset, size, target: Tensor)` reads raw bytes at the file offset into `target`. For a CUDA
  target it stages through a pinned pool (`cudaMallocHost`, `stloader.cpp:344`) and copies with `cudaMemcpyAsync`.
- `TensorLoadJob(handles, file_offset, bytesize, dest_ptr, dest_size, bf16_to_fp16, fp32_to_fp16, cuda, device_id)`.
- `stloader_deferred_cpu(jobs)` and `stloader_deferred_cuda(jobs, max_chunk_size=4 MiB)` run batched jobs. The CUDA path
  coalesces reads into 4 MiB pinned slots per thread (8 threads × 4 slots), then does an async H2D copy, plus an
  **in-place bf16→fp16 kernel** (`__float2half_rn`, no clamp) when `bf16_to_fp16` is set, then a per-slot event and a
  final `cudaDeviceSynchronize`. `fp32_to_fp16` is not done in C++: Python loads fp32 into a temp tensor and `copy_`s it
  (`safetensors.py:981-997`).

Semantics to reproduce: raw safetensors bytes go to the device. If a tensor is bf16 and the caller did not pass
`allow_bf16`, it becomes fp16 with RN rounding, so values above 65504 become ±inf. If it is fp32 and the caller passed
`float2half`, it becomes fp16. Transpose and `pad_to` are applied afterwards.

The `"python"` method (`safetensors.py:839-857`) reproduces the same semantics with `torch.frombuffer` and `.to()`.

### (c) GLM5Next forward through this part (hyper-connections, block order, model loop)

Module list (`architecture/glm5_next.py:149-344`):

| index | module |
|---|---|
| 0 | `Embedding` (caps `prefer_cpu`, so it lives on the **CPU**) |
| 1 | `ExpandStreams` |
| 2..46 | 45 × `TransformerBlock` (`first_block_idx=2`, `last_kv_module_idx=46`) |
| 47 | `HyperHead(mean=True)` |
| 48 | `RMSNorm` final, `out_dtype=half` |
| 49 | `Linear lm_head` (EXL3, 6-bit `head_bits`, caps `logits_output`; `logit_layer_idx=49`) |

Model loop. `Model.forward` (`model/model.py:374`) runs these steps:
1. `prepare_inputs`: `prepare_for_attn` and `prepare_for_recurrence`. These are Parts C and D. They fill params with
   cache seqlens, block tables and recurrent states.
2. `forward_ls` (`model/model_ls.py:317-330`) iterates `fwd_modules`:
   - For each module it calls `module.prepare_for_device(x)`, which does `x.to(module.device)`, non-blocking if the
     source is pinned (`modules/module.py:117-125`).
   - It then calls `module.forward(x, params)`.
   - Just before the logits layer, if `params["last_tokens_only"]` is set (the generator sets it to 1 at
     `generator/job.py:1396`), it slices `x = x[..., -n:, :]`.
3. `advance_recurrent_states` runs at the end (Part C).

`Model.prefill` (`model/model.py:349`, `model_ls.py:298-314`) is used for the non-final chunks of a prompt:
- It sets `params["prefill"] = True` only on module 46, the last block.
- That block runs its attention, so the KV/recurrent caches fill, and then **returns before its MLP**
  (`transformer.py:160`).
- The loop then breaks, so HyperHead, the norm and lm_head do not run, and it returns `None`.

Shapes and dtypes along the residual:
- Embedding: `nn.Embedding` on the CPU. The weight is fp16, or bf16 because of `allow_bf16`. The output is cast to
  **fp32** (`out_dtype=torch.float`) with shape (b,s,4096) on the CPU.
  - In the generator's decode loop (`generator/generator.py:1037`, `pinned_staging=True`) it is copied into a reused
    `pin_memory=True` buffer (`modules/embedding.py:158-166`) so that the upload is asynchronous.
- ExpandStreams (`hyperconnections.py:37-38`): `x.float().unsqueeze(2).expand(-1,-1,4,-1).contiguous()` on the
  device. The result is fp32 **(b, s, 4, 4096)**, with 4 identical copies, and this stack is the residual for the rest
  of the network.
- TransformerBlock.forward (`modules/transformer.py:137-215`), for GLM, which has HC and no post-norms or scalars:
  1. `hc_post, hc_comb, y = attn_hc.mix(x)` at `:151`. This is `ext.hc_mix`, and `y` is (b,s,4096) **fp16**.
  2. `y = input_layernorm(y, out_dtype=half)` at `:154`. This is `ext.rms_norm`, fp16 to fp16, eps 1e-5.
  3. `y = self_attn(y)` at `:159`. This is either GatedDeltaNet (KDA) or MLAttention, both with `out_dtype=torch.float`,
     so `y` is fp32 (b,s,4096).
  4. Prefill early-out for the last block (`:160`).
  5. `x = attn_hc.apply_(x, y, hc_post, hc_comb)` at `:165`. This is `ext.hc_apply`, in place on the fp32 stream stack.
  6. `hc_post, hc_comb, y = mlp_hc.mix(x)` at `:175`, then `post_attention_layernorm` at `:178` (`ext.rms_norm`).
  7. `y = mlp(y)` at `:187`. This is either GatedMLP (layers 0–2, dense) or BlockSparseMLP with a shared expert, both
     with `out_dtype=torch.float`.
  8. `x = mlp_hc.apply_(x, y, ...)` at `:191`.

  The block's output is the in-place updated (b,s,4,4096) fp32 stack. `to2(x, None, None)` does nothing.
- HyperHead with `mean=True` (`hyperconnections.py:640-641`): `x.mean(dim=2)` gives fp32 (b,s,4096) and has no
  weights.
- Final `RMSNorm` gives fp16 output (`ext.rms_norm` with an fp32 input).
- `lm_head` produces the logits (Part A).

State: the HC streams are carried only through the forward pass. There is **no HC state in the cache**.

Decode cost: each block has 2 sites, and each site runs `hc_mix` (2 kernels) + `rms_norm` + `hc_apply`. That is about
360 launches per token, which `k_hcfuse.py` measured at about 1.1 ms/token on CUDA.

`glm-flash-lite/glm53/k_hcfuse.py` is the optional decode fusion, enabled with `GLM53_K_HCFUSE=1`. It applies for
R ≤ 8 and D%256==0, and is bit-exact with the stock kernels. It replaces the four launches per site with two:
- `k_apply_partials` is the previous site's deferred `hc_apply` fused with this site's `hc_mix` partials.
- `k_finalize_norm` is the `hc_mix` finalize fused with the site's RMSNorm.

The MLP site's apply is deferred into the next block's attention site and is flushed with the stock `ext.hc_apply`
before any other consumer, such as HyperHead. It is an inline CUDA `load_inline` extension, so it is **not usable on
XPU**. Leave `GLM53_K_HCFUSE` unset. It is a good template for a later Triton fusion.

### (d) Model load and warmup sequence (serve.py → `model_init.init(args)`)

1. `Config.from_directory` (`model_init.py:207`) parses the config and constructs `SafetensorsCollection(directory, load_method=kwargs.get("load_method"))` (`model/config.py:132`).
   - Only headers are read here, in Python.
   - The default `load_method` is `"mt_fread"` (`loader/safetensors.py:413`).
   - **XPU: pass `load_method="python"`.** `model_init` does not expose this, so patch `model_init.py:207` or the default
     at `safetensors.py:413`.
2. `Model.from_config` (`model_init.py:262`) builds the modules and `prepare_layer_map`. No ext calls.
3. `Cache(model, 131072, CacheLayer_fp16, max_history, max_batch_size=args.autosplit_max_batch_size=4)` at
   `model_init.py:303`. It attaches to the model. Cache tensors are allocated when the owning attention or recurrent
   module loads (`cache/cache.py:134,178`). Details are in Parts D and F.
4. `model.load(use_per_device=None, tensor_p=False, max_batch_size=4, max_chunk_size=8192, ...)` at `model_init.py:360`,
   which calls `load_gen` (`model/model.py:413`):
   - `free_mem()` (`gc` plus `torch.cuda.empty_cache`).
   - `device=None` takes the **autosplit** path: `num_devices = torch.cuda.device_count()` (`model.py:551`), a reserve
     of 0.5 GB per device, `active_devices=[0..n-1]` as **ints**, then `_load_autosplit` (`model_ls.py:66-295`). Per module:
     - `torch.device(active_devices[i])` (`:150`). **An int device ordinal means CUDA.**
     - The first time a device is used, `set_memory_fraction_reserve` (`util/memory.py:27`) calls `touch_device`,
       `torch.cuda.mem_get_info`, `memory_reserved` and `set_per_process_memory_fraction`.
     - It allocates the dummy hidden state `zeros((1,8192), long)` on the device. Because Embedding is `prefer_cpu`,
       `x_device` is the CPU for module 0.
     - Inside `begin_deferred_load` / `module.load(device, max_chunk_size=8192)` / `end_deferred_load` it runs
       `stloader_deferred_cuda` or `_cpu`.
     - **It then runs a real forward of the dummy 8192-token chunk through the module** (`:197-198`). The params are
       `attn_mode="flash_attn"`, the cache, `past_len=0`, `batch_shape=(1,8192)`, plus `prepare_for_attn` and
       `prepare_for_recurrence` (`:94-109`). So the **entire 8192-token prefill path runs at load time**.
     - Memory is measured with `reset_peak_memory_stats`, `memory_allocated`, `max_memory_allocated` and `mem_get_info`.
       An OOM moves loading on to the next device.
   - At the end it calls `unset_memory_fraction`, frees the recurrent states and runs `stc.close()`
     (`stloader_close_file`).
   - It then calls `free_mem()` and `g_tensor_cache.drop_all()`, and sets `cache.initialized=True` (`model.py:636-645`).
   - The forward at load can be skipped with `-asnf` (`--autosplit_no_forward`). Single-device loading also skips it:
     `model.load(device="xpu:0")` takes `_load_single` (`model_ls.py:30-55`), which runs no forward and does no memory
     accounting.
     - `model_init.init(args, device=torch.device("xpu",0))` forwards `**kwargs` to `model.load`, so `serve.py` needs a
       one-line change, or `model_init.py:360` can be patched.
     - `_load_single` calls `module.load(device)` **without** `max_chunk_size`. Check that the modules have a default for
       it (Part A and 2 modules size workspaces from it).
5. Warmup (`model_init.py:377`, `Model.warmup` at `model/model.py:190-346`) runs unless `-nw`.
   - `max_batch_size` keeps its default of 8 (model_init does not pass it).
   - The families (PAGE_SIZE=256) are:
     - "rows n" for n in 1, 2, 4, 8, 16 at past_len 0.
     - "chunk 8192".
     - "batch B decode" with B = min(8, cap//256, free recurrent slots). Its steps are a pre-fill of
       min(32, 256-16-2)=32 tokens, a 1-token step, and a q_hist = min(16, cache.max_history+1)-token step with
       `recurrent_history=True`.
     - "batch 1 decode" with the same steps.
     - "long context 8192", which runs 8192 tokens, then 64 tokens, then 1 token, so the DSA top-k sparse regime
       compiles.
   - Each step is a real `self.forward(...)` with `attn_mode="flash_attn"`, the cache and the recurrent states. Failures
     are caught and printed (`:332-336`), so warmup cannot crash a load.
   - **It can be skipped with `-nw`.** It only pays first-use costs: Triton autotune and compile, the GEMM autotuner and
     the BC graph capture.
   - On XPU, skip it while bringing up correctness. Use it later to pre-JIT the Triton kernels.
6. Loading the tokenizer is pure Python.

Load-time numerics to preserve:
- HC tensors: `.float()`.
- RMSNorm weight: fp16, or bf16 if the checkpoint has bf16.
- Embedding: on the CPU, fp16 or bf16.

### (e) CUDA dependencies in this part (non-ext) and the torch.xpu mapping

| file:line | dependency | reached? | XPU mapping |
|---|---|---|---|
| `model/model.py:551` | `torch.cuda.device_count()` | load (autosplit) | `torch.xpu.device_count()`. Better: load single-device with `device="xpu:0"`. |
| `model/model_ls.py:150-152` | `torch.device(active_devices[i])` with an **int**, which means CUDA | load (autosplit) | `torch.device("xpu", i)`. A bare int is not safe on an XPU build. |
| `model/model_ls.py:193` | `measure = load_device.type == "cuda"` | load | On XPU `measure` becomes False, so memory accounting is silently skipped. This is harmless for a single device. Use `in ("cuda","xpu")` with the `torch.xpu` equivalents if accounting is wanted. |
| `model/model_ls.py:195-226` | `torch.cuda.reset_peak_memory_stats / memory_allocated / max_memory_allocated / mem_get_info / memory_reserved`, `torch.cuda.OutOfMemoryError` | load (only if measure) | The `torch.xpu.*` equivalents exist (`reset_peak_memory_stats`, `memory_allocated`, `max_memory_allocated`, `memory_reserved`, `mem_get_info`). Use `torch.OutOfMemoryError`. |
| `util/memory.py:13-17` (`touch_device(int)`) | `torch.empty(..., device=int)` | load | Same int-device issue. Pass `"xpu:N"`. |
| `util/memory.py:32,37,40,55,56,61,63,70` | `torch.cuda.mem_get_info`, `memory_reserved`, `get_device_properties().total_memory`, `set_per_process_memory_fraction` | load (autosplit) | `torch.xpu.mem_get_info`, `torch.xpu.memory_reserved` and `torch.xpu.get_device_properties(i).total_memory`. **`set_per_process_memory_fraction` may be missing or no-op in torch.xpu** depending on the version. Guard it, or bypass autosplit. |
| `util/memory.py:76` (`free_mem`) | `torch.cuda.empty_cache()` | load, and wherever `free_mem` is called | `torch.xpu.empty_cache()` |
| `util/memory.py:364-467` (`vram_accounting`) | various `torch.cuda.*` | diagnostics only, not on the path | none |
| `util/misc.py:35-43` (`cuda_sync_active`) | `torch.cuda.synchronize` over `cuda:{i}` | callers are outside this part (check the generator in Part F) | `torch.xpu.synchronize()` |
| `util/device_copy.py:39-40,81` | `torch.cuda.synchronize` for the P2P probe. Only for cuda→cuda. | no (single device) | none |
| `loader/safetensors.py:658` | arena only if `device.type == "cuda"` | load | On XPU the arena is off and each tensor gets its own allocation. This is fine. Expect allocator fragmentation with about 288×45×3 expert tensors. Consider porting the arena with `xpu` as the type. |
| `loader/safetensors.py:793-794` | `"cuda": tensor.is_cuda` routes deferred jobs | load (mt_fread) | **Hazard**: an XPU tensor has `is_cuda == False`, so its job goes to `stloader_deferred_cpu`, which writes through a raw **device** pointer from the host. Use `load_method="python"`. |
| `stloader.cpp:187-411` | `cudaMallocHost` pinned pool, `cudaMemcpyAsync`, `cudaStreamCreate`, events, `cudaDeviceSynchronize`, bf16 to fp16 kernel | load (mt_fread) | Replaced by the `"python"` loader or by `safetensors` `safe_open(device="xpu:0")` followed by `.to(torch.half)`. |
| `model/config.py:74` | `device.type == "cuda"` guard around `exl3_gemv_int8_max_k` | load | Skipped on XPU. OK. |
| `modules/embedding.py:164` | `torch.empty_like(x, pin_memory=True)` (generator decode only) | decode | Pinned memory on an XPU build pins for the current accelerator in recent torch (2.5+). Verify it. If it fails, set `pinned_staging` False or drop the staging, which only costs a synchronous H2D of 4096 floats per row. |
| `modules/module.py:123`, `util/tensor.py:162` | `x.is_pinned()` lets a copy use `non_blocking` | forward | Works for XPU tensors. A non-blocking H2D copy from pinned memory is stream-ordered on XPU too. |
| `modules/rmsnorm.py:220`, `modules/attn.py:1368` | `torch.cuda.synchronize()` | TP import only | none |
| `modules/hyperconnections.py:330` | `torch.version.hip`, `dev.type == "cuda"` | GatedResidual only | none |
| `exllamav3_ext/graph.cu:33-…` | `cudaStreamBeginCapture`, `cudaGraphInstantiate`, kernel-node param patching, driver-API nodes for Triton cubins | only inside `BC_*` | Don't port. Disable the BC paths so that no graphs are used. `torch.xpu` graphs (`torch.xpu.XPUGraph`, torch ≥ 2.7/2.8) could later capture the eager decode step as a whole. |
| `exllamav3_ext/triton_kernel.cpp:7-49` | `cudaFree(nullptr)`, `cuModuleLoadData` of a **cubin**, `cuLaunchKernel` | only inside the BC attention graph | Don't port. Call Triton kernels from Python instead. |
| `exllamav3_ext/cuda_host.cpp` | `cudaHostRegister` / `cudaHostGetDevicePointer` | TP and `vision_pinned` only | none |
| all `.cu` in this part | `at::cuda::getCurrentCUDAStream`, `OptionalCUDAGuard` | n/a | In the shim, torch ops on XPU tensors use the current XPU stream automatically. |

Minimal set for this part to run a correct forward:
- `hc_mix` and `hc_mix_num_chunks`, as torch versions of the references above.
- `hc_apply`.
- `rms_norm`.
- Loader: `load_method="python"`, so no stloader ops are needed.
- Loading: a single-device `xpu:0` load (`_load_single`), or autosplit with the device-string, int-device and memory-API
  patches.
- Warmup off (`-nw`) during bring-up.

Everything else in this part is unreachable for GLM-5.3-Flash: `hc_head`, `gr_mix*`, `det_quant_weight` (HC use),
`rms_norm_res_in`, `add`, `cuda_host_*`, `pinned_cuda_view` and `rope`.


---

## Part F. Generator, page table, sampling, cache allocation, recurrent-state machinery

Paths are relative to `glm/exllamav3`. "serve config" means
`glm53/serve.py -cs 131072 --max-batch-size 8 -chunk_size 8192 -ambs 4`, with no `--mtp`, `-dm`, `-ngram`, `-cq`
or `-ccs`, and the samplers built by `make_sampler` (`ArgmaxSampler()` when temperature <= 0, otherwise
`ComboSampler(temperature, top_p, top_k, min_p)`). The `GLM53_*` monkeypatches are off by default.

### Key findings first

1. **Decode batch is capped at 4, not 8, under the serve config.** `model_init.init` builds the cache with
   `max_batch_size = args.autosplit_max_batch_size` (`exllamav3/model_init.py:300-309` (fp16 branch)). `-ambs 4` therefore gives
   `Cache.num_slots = 4` (`exllamav3/cache/cache.py:161`), and the Generator clamps
   `self.max_batch_size = min(8, cache.num_slots)` for recurrent models (`exllamav3/generator/generator.py:235`). The
   warmup batched family also uses `min(8, len(cache.free_list)) = 4` (`exllamav3/model/model.py:281-285`). The decode
   shapes to support are bsz 1-4 with q_len 1. Raising `-ambs` to 8 would make bsz 5-8 reachable.
2. **No drafting is reached.** `model_init` sets `use_mtp` only when `--mtp` is passed or `-dm` points at the model dir
   (`exllamav3/model_init.py:197-204`). `--ngram_match_min` defaults to 0 (`model_init.py:121`), so
   `num_draft_tokens = 0` (`generator.py:168-171`) and `max_history = 0` (`model_init.py:270-275`). As a result
   `recurrent_history` is always False, `GDNState.last_history` is always 0, and the GDN history dimension is 1.
   **MTP is not used by default; `glm5_next_mtp.py` is not on the path.**
3. **On the default path the sampler is always the fused kernel.** `EXL3_FUSED_SAMPLER` defaults to on
   (`generator/sampler/custom.py:19`). Both `ArgmaxSampler` and every `ComboSampler` that serve.py builds collapse
   to a single `SS_Fused` step with no head steps, so `fused_only = True`. That means `ext.fused_sampler` is the
   **only** sampling extension op reached. `gumbel_noise_*`, `apply_rep_pens`, `apply_pres_freq_pens`, `dry_penalty`,
   `adaptivep_gumbel_noise_f32`, `apply_logit_bitmask`, `argmax_sample` and `gumbel_sample` are not reached.
4. **The generator calls a CUDA-only API on the decode path.** `torch.cuda.synchronize(batch_logits.device)` at
   `generator/generator.py:1128` runs every decode step and raises on an XPU device or on a CUDA-less torch. The
   pinned-readback branch is gated on `next_token.is_cuda` (`generator.py:1122`), which is False on XPU. Both must be
   patched (see section D).
5. **Partial-page prompt reuse (`count_match_tensor`, `Cache.copy_page`) is skipped for recurrent models**
   (`generator/job.py:1268`), so neither is reached. Prompt-cache reuse for GLM5Next works only at page granularity
   and only up to the last page that has a stashed recurrent checkpoint.

### (a) Op table (this part)

Hot-path rank: D = every decode step; P = prefill/allocation; R = rare (checkpoints, requeue); I = idle-time
housekeeping; N = not reached under the serve config.

| op | category | when reached | rank | Python call sites | condition / avoidable? | recommended XPU path |
|---|---|---|---|---|---|---|
| `fused_sampler` | sampling | once per active job per decode step (bsz=1 each call) | D | `generator/sampler/custom.py:248` (SS_Fused.run) | Avoidable with `EXL3_FUSED_SAMPLER=0`, which falls back to step-by-step torch + `gumbel_noise_*` | Implement in torch (modes 0-3) inside the shim. The eager fallback still needs `gumbel_noise_*`, so the fused op is the smaller port |
| `FUSED_SAMPLER_MAX_BLOCKS` (=128), `FUSED_SAMPLER_HIST_STRIDE` (=24640) | module attrs | read when SS_Fused allocates workspaces | D | `custom.py:229`, `custom.py:240` | required attrs | export the same int constants; the buffers can be ignored |
| `cache_rotate` | page table / defrag | `on_queue_drained()` → `PageTable.defrag()`, only when the queue is empty, ≥ 8×512 page accesses since the last defrag, and >10% of pages misplaced | I | `generator/pagetable.py:1025` | Avoidable: `Generator(enable_defrag=False)`, or a shim no-op is safe for correctness. Defrag only improves page contiguity | torch per-page `copy_` loop (see detail), or patch `defrag` to return early |
| `partial_strings_match` | job / stop strings (CPU) | per sampled token, only if the request has stop **strings** (EOS ids alone do not trigger it) | D (CPU) | `generator/job.py:988` (stop strings); `job.py:966` (banned strings, N) | CPU-only pybind; no GPU | pure Python/numpy reimplementation |
| `count_match_tensor` | job / partial-page reuse (CPU) | — | N | `generator/job.py:1275` | skipped because `recurrent_cache is not None` | pure Python, or omit |
| `batched_conv_rewind`, `batched_state_rewind`, `ConvRewindJob`, `StateRewindJob` | recurrent-state rewind | only via `GDNState.rewind()`: draft rejection (`generator.py:1089`, `1246`), atomic-MM prefill overrun (`job.py:1437-1438`), banned-string rewind (not taken, since `rollback_capacity()==0`) | N | `modules/gated_delta_net.py:62-64`, `266`, `283` | not reached without drafts | patch `_collect_rewind_jobs`/`_dispatch_rewind_jobs` to call the existing torch `GDNLayerState.rewind()` (`gated_delta_net.py:238-250`). The raw-pointer job classes cannot be honoured in torch |
| `gumbel_noise_f16` / `_f32` / `_log` | sampling (eager) | only with `EXL3_FUSED_SAMPLER=0` or a non-collapsible stack | N | `custom.py:120,123,126,129,133` | avoidable (default) | torch if ever needed (see detail) |
| `apply_logit_bitmask` | sampling | packed-bitmask filter mask on a non-fused stack | N | `custom.py:1226` | no filters in serve.py | — |
| `apply_rep_pens`, `apply_pres_freq_pens`, `dry_penalty`, `adaptivep_gumbel_noise_f32` | sampling penalties | serve.py's ComboSampler leaves rep_p=1, pres/freq=0, dry=0, adaptive=1, so `alt()` turns each into `SS_NoOp` | N | `custom.py:770,779,833,843,1019,1079` | avoidable (default) | — |
| `argmax_sample`, `gumbel_sample` | sampling (basic) | never called from library Python (only `tests/test_ext_sampling_edges_.py`) | N | — | — | — |
| `BC_SAM` | n-gram drafting | `ngram_match_min > 0` | N | `generator/job.py:1173` | default 0 | — |
| `quant_cache_*`, `dequant_cache_*` | cache quant | only with `-cq` (and `CacheLayer_MLA_quant` packing in `mla_triton`) | N | `cache/quant.py`, `attention_fn/mla_triton.py` | `-cq` not passed | — |
| `paged_kv_cache_update`, `dspark_write_rows` | other cache layers | MHA fp16 cache / DSpark only; MLA uses Triton `mla_kv_append` | N | `cache/fp16.py:76`, `cache/dsa.py:486` | not this model | — |
| `histogram`, `count_inf_nan` | diagnostics | calibration/eval only (`linear.py:569` under `params["capture"]`) | N | — | — | — |

The `ext.encode` / `ext.endswith` / `ext.lower` names in the earlier grep of `generator/job.py` are false positives.
They are `held_text.lower().encode(...)` and `held_text.endswith(...)`, where `held_text` happens to contain `ext.`.
The same applies to `ext.device`, `ext.shape` and `ext.index_*` in `modules/moe_batch_recon.py`.

### (b) Per-op detail

#### fused_sampler (hot, the only GPU sampling op on the default path)

**Binding:** `m.def("fused_sampler", &fused_sampler)` (`exllamav3_ext/bindings.cpp:238`), with the module attrs
`FUSED_SAMPLER_MAX_BLOCKS = 128` and `FUSED_SAMPLER_HIST_STRIDE = 2*1024*12 + 64 = 24640`
(`exllamav3_ext/generator/sampling_fused.cuh:5-11`).

**Signature** (`exllamav3_ext/generator/sampling_fused.cu:770-791`), positional order as called at `custom.py:248-264`:
```
fused_sampler(
  logits:        Tensor (bsz, dim) half|float, contiguous    # per-job (1, vocab) view of batch_logits
  logit_mask:    Optional[Tensor] (1|bsz, >=size) half        # additive mask; None on serve path
  logit_bitmask: Optional[Tensor] (1|bsz, words) int32        # bit clear = masked; None on serve path
  out:           Tensor (bsz, 1) int64                        # written in place
  workspace:     Tensor float, >= bsz*128*3                   # scratch (partial max + ValIdx partials)
  size: int        # effective vocab bound <= dim: min(dim, tokenizer.actual_vocab_size[, mask width])
  inv_temp: float  # 1/T
  minp_log: float  # additive log-space min-P threshold
  random: uint32   # per-sample seed (job.rng.randint(0, 2**32-1), job.py:578)
  mode: int        # 0 greedy, 1 gumbel, 2 gumbel+minP, 3 gumbel+topK/topP(+minP)
  filters: int     # mode 3 bits: 1=TOPK, 2=TOPP, 4=MINP
  top_k: int; top_p: float
  inv_temp_filter: float   # 1/T if temperature precedes the filters in the stack, else 1.0
  histogram: Optional[Tensor] uint8 >= bsz*24640, 8-byte aligned  # mode 3 only; zeroed internally
)
```
Before the call, `CustomSampler.forward` writes `-inf` in place into `logits[..., tokenizer.actual_vocab_size:]`
(`custom.py:1182-1183`) and passes `size = min(dim, actual_vocab_size)` (`custom.py:1200-1213`).

**How serve.py stacks collapse** (`presets.py:105-151`, `custom.py:279-344`, `custom.py:1131-1159`):

| serve.py request | stack after `alt()` | SS_Fused mode |
|---|---|---|
| temperature <= 0 → `ArgmaxSampler` | `[Argmax]` | 0 (greedy) |
| `ComboSampler(T, top_p=1, top_k=0, min_p=0)` | `[Temperature(T)?, Sample]` (Temperature is a NoOp when T=1) | 1, inv_temp=1/T |
| `ComboSampler(..., min_p>0)` only | `[Temp?, MinP, Sample]` | 2, minp_log = T·ln(min_p) if Temp present (temp_first) else ln(min_p) |
| any top_k>1 and/or top_p<1 | `[Temp?, MinP?, TopK?, TopP?, Sample]` | 3, filters bitset; inv_temp_filter = 1/T if Temp present else 1 |
| top_k == 1, or top_p == 0 (impossible via serve.py, which maps 0 → 1.0) | — | 0 |

`ComboSampler` places temperature first (`temp_last=False`), so when T≠1 the filters act on the tempered
distribution. `make_sampler` reads `top_p or 1.0` and `top_k or 0`, so the typical "temperature 1.0, top_p 0.95"
request gives mode 3 with `filters=TOPP`, `inv_temp=1` and `inv_temp_filter=1`.

**Exact math** (per row; `x_i` = fp32(logit_i) plus mask, for i < size):
- **Mode 0:** `argmax_i x_i`. Ties go to the lowest index (`fs_amax_first`, `sampling_fused.cu:62-69`). If every
  value is -inf or NaN the result is index 0 (`out = idx==INT_MAX ? 0 : idx`, `:764`). NaN never wins because both
  `>` and `==` are false.
- **Gumbel noise** (modes 1-3, `sampling_fused.cu:722-727`):
  `u = curand_uniform(Philox4x32-10; seed = random (u32 widened to u64), subsequence = row*dim + i, offset 0)`.
  - `curand_uniform` returns `x*2^-32 + 2^-33` with `x` = the first 32-bit output, so u ∈ (0, 1].
  - Philox setup: counter = (0, 0, subseq_lo, subseq_hi); key = (seed_lo, 0); standard 10 rounds with
    M0=0xD2511F53, M1=0xCD9E8D57, W0=0x9E3779B9, W1=0xBB67AE85; output word .x.
  - `g = -__logf(max(-__logf(max(min(u, 0.99999994f), 1e-20)), 1e-20))`.
  - Objective: `v_i = x_i * inv_temp + g_i`. The result is `argmax v` with lowest-index tie-break.
  - Noise depends only on (random, flat index), so skipped (filtered) elements do not shift anyone else's noise.
  - Bit-matching the CUDA stream is **not** required for correctness. A torch port may use `torch.rand` with an XPU
    `torch.Generator` seeded from `random`, using the same clamp/log formula.
- **Mode 2:** `M = max_i x_i` (masked). Elements with `x_i < M + minp_log` are dropped, then Gumbel-argmax runs over
  the rest.
- **Mode 3:** kept set K, then Gumbel-argmax over K.
  - The min-P threshold is the same as mode 2 when the MINP bit is set.
  - Top-K keeps the k largest (after min-P). All tokens tied with the k-th value are kept, since ties share a
    sub-bucket.
  - Top-P uses weights `w_i = exp((x_i - M) * inv_temp_filter)` over the min-P-filtered, top-K-truncated set.
    `target = top_p * Σw`. Walking tokens in descending order, the token whose cumulative mass first **strictly
    exceeds** the target is dropped, along with everything below it. The top token is always kept.
  - This matches the eager rule `mask = cumsum(p) <= top_p; p[1:] *= mask[1:]` (`custom.py:465-468`) up to tie
    handling. The eager path renormalizes after MinP/TopK through `SS_Normalize` before TopP; the fused histogram
    excludes filtered tokens from Σw, which is the same thing.
  - Implementation detail that matters only for bit-exactness: CUDA uses a 1024-bucket coarse histogram over 32
    nats below M (bin = `(M - x)*1024/32*inv_temp_filter`), then 1024 refinement sub-buckets (1/32768 nat) and u64
    fixed-point mass at 2^40. Logits deeper than 32 nats land in the last bucket, which a top-K crossing never
    truncates. A torch reimplementation by sort + cumsum gives the same kept set except at exact ties and in that
    deep tail.
- **Torch reference, correct rather than fast:**
  ```python
  x = logits.float()[:, :size]  # mask added
  M = x.max(-1, keepdim=True)
  keep = x >= M + minp_log      # only if the MINP bit is set (or mode 2); with -inf excluded
  # top-K / top-P: sort x[keep] descending, apply the rules above
  v = torch.where(keep, x * inv_temp + gumbel(u), -inf)
  out = v.argmax(-1)            # stable lowest-index tie-break
  ```
  `torch.argmax` returns the first maximal index, which matches the CUDA tie-break.

#### cache_rotate (idle defrag)

**Signature** (`exllamav3_ext/generator/cache.cu:84-108`): `cache_rotate(cache: Tensor (num_pages, ...) any dtype,
contiguous; order: Tensor (2n,) int32 on the cache device; temp: Tensor, nbytes == one page)`. It returns nothing and
writes `cache` and `temp` in place. If page_size or n is 0 it returns immediately, so the zero-width MLA `v` plane
(`qk_rope_head_dim == 0`) is a no-op.

**Semantics:**
```
for i in range(n):
    a, b = order[2i], order[2i+1]
    (page[a] if a >= 0 else temp) <- (page[b] if b >= 0 else temp)
```
The copies run strictly in order, bytewise per page. Each CTA owns a disjoint byte range of every page, so the
sequential order holds per byte.

**Order encoding** (`pagetable.py:977-999`): each permutation cycle is encoded as
`[-1, s, d0, s0, d1, s1, ..., dk, -1]`. The first pair saves page `s` to temp, the middle pairs shift pages along
the cycle, and the last pair restores temp into `dk`.

**Caller** (`pagetable.py:1018-1025`): rotates every tensor in `Cache.get_all_tensors()`. For GLM5Next that is, per
MLA layer, `k` (latent), `v` (0-wide), `k_idx` and `k_pool`. The draft cache is not rotated because none exists.
Recurrent state is not paged and is not rotated.

**Torch port:**
```python
flat = cache.view(cache.shape[0], -1)
t = temp.view(-1)
if flat.shape[1] == 0:
    return
for a, b in pairs(order.cpu().tolist()):
    (flat[a] if a >= 0 else t).copy_(flat[b] if b >= 0 else t)
```
Alternatively, set `enable_defrag=False`. Defrag is a pure layout optimization and does not affect correctness.

#### partial_strings_match (CPU)

**Signature** (`exllamav3_ext/generator/strings.cpp:16-63`): `partial_strings_match(match: buffer u8 = UTF-32LE of
held_text, offsets: buffer u8 = int32 byte offsets (n+1), strings: buffer u8 = concatenated UTF-32LE needles) -> int`.

**Return value:**
- -1: no match.
- -2: some needle runs off the right end of `q` (a partial match at the tail) starting before every full match.
- k ≥ 0: the code-point index of the **earliest** full match over all needles.

**Algorithm:** for each needle and each start `a0 < min(best_full, best_partial)`, extend the match. A full match
sets `best_full = a0` and breaks; running off the end of `q` sets `best_partial = a0` and breaks. The function
returns -2 if `best_partial < best_full`, otherwise the full index or -1.

**Caller:** `job.py:988` decodes the result as a string index into `held_text`. A plain Python port over `str` is
exact.

#### batched_conv_rewind / batched_state_rewind (not reached; listed because they take raw pointers)

- `ConvRewindJob(src, dst, dim, cdim, stride)` copies `conv_state[slot, :, :cdim] <- conv_state[slot, :, p-cdim:p]`
  in bf16 (`gdn.cu:1924-1945`, `gated_delta_net.py:253-272`).
- `StateRewindJob(src, dst, num_elements)` copies `recurrent_state[slot, 0] <- recurrent_state[slot, last_history+1-n]`
  as a flat fp32 copy of `stride(1)` elements (`gdn.cu:1949-1959`, `gated_delta_net.py:275-287`).
- The torch equivalent already exists in `GDNLayerState.rewind()` (`gated_delta_net.py:238-250`). Route
  `_dispatch_rewind_jobs` to it in the shim. Do not try to emulate pointer jobs.

#### gumbel_noise_f16 / f32 / log (eager path only)

Source: `exllamav3_ext/generator/gumbel.cu:12-169`. All variants use the same Philox per element: seed = `random`,
subsequence = flat element index, `curand_uniform`, and the same `gumbel()` clamp formula.

| variant | output |
|---|---|
| f16 | `out = half_rn(float(in) + g)`, processed in element pairs, each element with its own subsequence |
| f32 | `out = in + g` |
| log | `out = __logf(p) + g` |

`out` may alias `in`.

### (c) Generator step for GLM5Next (single GPU, no TP, no drafts)

**Construction** (`generator.py:138-247`):
- `PageTable(cache)`: `max_pages = 131072/256 = 512` pages of 256 tokens (`constants.py:2`).
- `RecurrentCache(model, 4 GiB)`: an OrderedDict LRU of host-RAM stashes. `-rcs` defaults to 4.0 and serve.py passes
  `int(args.recurrent_cache_size*1024**3)`.
- `cache.reset_states()`, and `max_batch_size = min(8, 4) = 4`.
- `recurrent_checkpoint_interval = 2048` (model caps), `recurrent_checkpoint_interval_pp = ceil(32768/8192)*8192 = 32768`.
- `cpu_cache_size = 0`, so there is no `CPUPageCache`.
- `max_q_size` is computed but never stored or used (`generator.py:176`). It is inert.

**Cache allocation for `-cs 131072`** (`model_init.py:300-309`, `cache/cache.py:145-171`, `cache/mla.py:31-82`,
`modules/gated_delta_net.py:174-216`). Tensors are allocated at `model.load()` by each module's `load_local`
(`mla_attn.py:312-313`, `gated_delta_net.py:622-623`) with `torch.zeros/empty(..., device=<layer device>)`.

| per layer | tensor | shape | dtype | size |
|---|---|---|---|---|
| MLA (11 layers: 3, 7, ..., 43; all `indexer_types = "full"`) | `k` (latent ckv) | (512, 256, 1, 512) | fp16 | 128 MiB |
| | `v` (rope key) | (512, 256, 1, **0**) | fp16 | 0 |
| | `k_idx` (indexer keys: `idx_plane_dim = index_head_dim*2 = 256` because `index_kpool` is set, holding [k ‖ gate scores]) | (512, 256, 256) | fp16 | 64 MiB |
| | `k_pool` (pooled keys, `PAGE_SIZE // kpool = 64` per page) | (512, 64, 128) | fp16 | 8 MiB |
| KDA (34 layers) | `conv_state` (`fdim_qkv = 2*64*128 + 64*128 = 24576`, `conv_kernel 4 + max_history 0`) | (4 slots, 24576, 4) | bf16 | 0.75 MiB |
| | `recurrent_state` (slots, history+1, heads, k_dim, v_dim) | (4, 1, 64, 128, 128) | fp32 | 16 MiB |

Totals:
- MLA ≈ 200 MiB/layer × 11 ≈ 2.15 GiB of paged KV.
- KDA state ≈ 16.75 MiB × 34 ≈ 570 MiB.
- One recurrent checkpoint (`GDNState.checkpoint_size`) = 34 × (24576·4·2 + 64·128·128·4) B ≈ 142 MiB of host RAM,
  so the 4 GiB recurrent cache holds about 28 checkpoints.

**`iterate()`** (`generator.py:443-563`). One call per asyncio tick (`async_generator.py:36`). It is fully
synchronous on the event-loop thread.

**1. `iterate_start_jobs`** (`generator.py:1386-1443`). Admits jobs while `sum(seqs) < 4` and enough unreferenced
pages exist.
- `Job.allocate_pages` → `Sequence.allocate_pages` (`pagetable.py:258-297`):
  - Prompt pages are keyed by a chained blake2b hash (CPU).
  - Pages are revived by hash or evicted.
  - The usable cached prefix is cut back to the last page that **also** has a stashed recurrent checkpoint
    (`pagetable.py:626-634`).
- Then `cache.get_new_state()` takes a slot from the free list. `GDNState(clear=True)` zeroes
  `conv_state[slot]` and `recurrent_state[slot]` with torch `zero_`.
- Or `cache.new_from_stashed(stash, position=cached_pages*256)` → `GDNLayerState.unstash` runs
  `copy_(cpu_tensor)` into `recurrent_state[slot,:1]` and `conv_state[slot,:,:4]` (`gated_delta_net.py:298-302`).

**2. Prefill, one chunk per job per iterate** (`job.py:1203-1479`).
- Chunk: `[kv_position, min(floor((kv_position+8192)/256)*256, len-1))`.
- The **last prompt token is never prefilled**; it becomes the first decode input.
- Because the model is recurrent, a chunk that would cross the last full-page boundary of the prompt is split at
  that boundary (`job.py:1305-1313`). The forward ending there is followed by `maybe_stash_recurrent(interval=256)`,
  so the prompt's last full page gets a checkpoint.
- Call: `model.prefill(ids (1, n) long CPU, params)`, with params:
  - `attn_mode = "flash_attn"`
  - `block_table = seq.block_index_tensor` (CPU int32 (1, pages), not pinned)
  - `cache`
  - `cache_seqlens = torch.tensor([start], int32)` (CPU)
  - `recurrent_states = [state]`
  - `indexed_embeddings = []`, `inv_freq = None`, `mm_span_prefix = 0`
- `Model.prefill` (`model/model.py:349-371`) → `prepare_inputs` (`prepare_flash_attn` + `prepare_for_recurrence`,
  which adds `recurrent_slots`, a CPU int32 tensor with `_static_dev_cache`) → `prefill_ls`, which stops after the
  last TransformerBlock (`model_ls.py:298-314`; HyperHead, norm and lm_head are skipped) →
  `advance_recurrent_states` (position += n; `last_history = 0`).
- Prefill is always batch 1. It interleaves with the decode forward of other jobs in the same `iterate()`.

**3. Recurrent checkpoints** (`generator.py:585-587`, `job.py:1562-1620`). Each iterate, for every active job:
- Stash when `(kv_position - cached_pages*256) % interval == 0` and the page is full.
- interval = 2048 once `kv_position >= prompt_len - 2*8192`, otherwise 32768.
- `RecurrentCache.put` (`cache/recurrent.py:55-95`) → `GDNState.stash` → per layer
  `recurrent_state[slot,:1].cpu()` and `conv_state[slot,:,:4].cpu()` (`gated_delta_net.py:290-295`). These are
  **synchronous D2H copies to pageable memory**, about 142 MiB per stash. Eviction is LRU, dropping stranded entries
  first.
- When a requeue happens (`GLM53_MAX_RQ_TOKENS`, aligned to the 2048-token checkpoint boundary at `job.py:1110-1117`),
  the job stashes at a page boundary before releasing its pages (`generator.py:1339-1343`). The requeued job then
  resumes from KV pages + stash.

**4. Decode, `iterate_gen`** (`generator.py:943-1353`). Assembles a compact batch of every prefill-complete job
(bsz ≤ 4):
- `block_index`: pinned int32 staging `(bsz, ceil(max_seq_len/256) rounded up to a multiple of 16)`, zero-filled,
  each row copied from `seq.block_index_tensor`.
- `cache_seqlens`: pinned int32 `(bsz,)` = `seq.kv_position`.
- `positions = None`: no mrope caps; the model is NoPE.
- `batch_ids`: pinned int64 `(bsz, 1)` = each sequence's next token.
- `recurrent_states`: a list of per-job `GDNState`. Slot indices become `recurrent_slots` via `prepare_for_recurrence`.
- Params: `{attn_mode: "flash_attn", block_table, cache, cache_seqlens, recurrent_states, indexed_embeddings: [],
  positions: None, recurrent_history: False, pinned_staging: True}`. `pinned_staging` is not read anywhere.
- `model.forward(batch_ids, params)` returns logits `(bsz, 1, vocab=154880)`.
- Modules upload `block_table`/`cache_seqlens` with `get_for_device` (`util/tensor.py:115-165`). Pinned sources are
  copied `non_blocking=True`. This is safe only because the generator synchronizes before the staging buffers are
  refilled next iteration.
- Sampling (`generator.py:1108-1151`): for each job, `job.receive_logits(batch_logits[a:b])` →
  `CustomSampler.forward(logits (1,1,V), current_device_ids, rand_u32, tokenizer, logit_mask=None)` → one
  `fused_sampler` call per job.
  - Before that, `job.prepare_sampling_past_ids` (`job.py:1531-1547`) stages all sequence ids into pinned memory and
    copies them H2D `non_blocking` every step for ComboSampler. `SS_RepP.reqs_past_ids()` returns True even when
    RepP collapses to a NoOp (`custom.py:796-797`, `1136`). This is wasted but harmless; ArgmaxSampler skips it.
  - On CUDA each 1-element token is copied into a pinned buffer, then one `torch.cuda.synchronize(device)` runs, then
    `receive_sample` does `next_token.cpu().item()` (`job.py:620-621`), decodes the text, and checks stop tokens and
    strings.
- Completed or requeued jobs `deallocate_pages()` and `free()` their state slot. When the queue drains:
  `on_queue_drained()` → `recurrent_cache.prune_stranded()`, `pagetable.defrag()` (`cache_rotate`) and
  `malloc_trim()` (ctypes libc, not CUDA).

**Warmup** (`model/model.py:190-345`, called by `model_init.init` before the generator exists):
- Rectangular `batch_shape` mode with `past_len`.
- Single-row families at 1, 2, 4, 8 and 16 tokens.
- One 8192-token chunk.
- A bsz-4 family `[(32, False), (1, False)]` and a bsz-1 family with the same steps. There is no history step because
  `max_history = 0`.
- A long-context family `[(8192), (64), (1)]` on 8448 cache rows.
- Uses `cache.get_new_state()` / `release_state`. `-nw` skips warmup.

### (d) Non-ext CUDA dependencies in this part

| file:line | dependency | reached? | XPU mapping |
|---|---|---|---|
| `generator/generator.py:1128` | `torch.cuda.synchronize(batch_logits.device)` | **D, every decode step** | **Must patch:** `torch.xpu.synchronize(dev)` (or `torch.accelerator.synchronize`). It is also the sync point that makes refilling the pinned staging buffers safe, so do not turn it into a no-op |
| `generator/generator.py:1122` | `next_token.is_cuda` gates the pinned-readback batching | D | Use `next_token.device.type != "cpu"`. Without this, readback falls back to per-job `.cpu()` (correct, just slower) |
| `generator/generator.py:938, 1113-1114`; `job.py:1537` | `torch.zeros/empty(..., pin_memory=True)` staging (block table, seqlens, ids, sampled tokens, past ids) | D | Recent XPU torch builds back `pin_memory=True` with the accelerator host allocator; verify `.is_pinned()`. If unsupported, use `pin_memory=False` (copies become synchronous and stay correct) |
| `util/tensor.py:162-163`; `job.py:490, 1547` | `non_blocking=True` H2D from pinned memory | D | Works on XPU; stream-ordered on the current XPU queue |
| `generator/generator.py:994` (`645, 745, 843` draft only); `util/misc.py:35-43` | `cuda_sync_active()`: `torch.cuda.device_count()`, `memory_allocated`, `synchronize` over `cuda:{i}` | P (first token timing) | On a CUDA-less build `device_count()` returns 0, so this silently becomes a no-op and only `time_first_token` is affected. Patch to iterate `torch.xpu.device_count()` |
| `generator/job.py:620-621` | `next_token.cpu().item()` (implicit sync) | D | fine |
| `modules/gated_delta_net.py:293-294, 301-302` | stash `.cpu()` (sync D2H, pageable); unstash `copy_` H2D | R | fine on XPU. Optional later: pinned stash buffers |
| `cache/mla.py:79-82`; `gated_delta_net.py:212-215` | cache/state alloc `torch.zeros/empty(device=device)` | load | `device` comes from the loader's device strings (see Part E) |
| `generator/pagetable.py:838, 1052` | `torch.cuda.synchronize()` | debug only (`defrag(debug=True)`) | N |
| `generator/pagetable.py:1003-1005` | `all_rotations_cpu.to(device)` for `cache_rotate` | I | fine; a torch shim can use the CPU list |
| `modules/gated_delta_net.py:46` | `torch.device(l.device).index` + raw `data_ptr()` pointer jobs | N | route to `GDNLayerState.rewind()` |
| `generator/cpu_cache.py:119, 202, 328, 355` | `device.type == "cuda"` assert, pinned slab, non_blocking | N (`-ccs 0`) | — |
| `generator/generator.py:199-207` | pinned draft buffers | N | — |
| `cache/*.py` `copy_page(..., non_blocking=True)` | partial-page reuse | N (recurrent) | — |

There are no CUDA graphs, streams, events, `mem_get_info` or `torch.compile` calls in `generator/`, `cache/` or
`generator/sampler/`. The generator's thread pool (`generator.py:188`) only runs grammar filters, which are not
reached.

### Minimal set for this part (correct, not fast)

1. `ext.fused_sampler` in torch, modes 0-3 (mode 0 for greedy, 1 or 3 for typical requests), plus the attrs
   `FUSED_SAMPLER_MAX_BLOCKS` and `FUSED_SAMPLER_HIST_STRIDE`.
2. `ext.partial_strings_match` in Python (CPU), needed only when requests send stop strings.
3. `ext.cache_rotate` as a torch loop, or `enable_defrag=False`.
4. Python patches:
   - `generator.py:1122/1128` → device-agnostic synchronize.
   - `util/misc.py:cuda_sync_active` → XPU.
   - Check that `pin_memory=True` works on the XPU torch.
   - Optionally route `_dispatch_rewind_jobs` to the torch `GDNLayerState.rewind()`.

Nothing else in the generator/cache/sampler path calls into `exllamav3_ext`.


---

## 6. Consolidated CUDA dependencies and Triton summary

### 6.1 Non-ext CUDA dependencies on the reached path

The full per-area tables are in Part A §1.5, B (d), C (e), D (e), E (e) and F (d). This table lists only the
ones that matter for the section 3 configuration.

| category | where | status on XPU | mapping |
|---|---|---|---|
| device count / int ordinals / memory APIs | `model/model.py:551`, `model/model_ls.py:150-226`, `util/memory.py:13-76` | **breaks** in autosplit | single-device `device="xpu:0"` load. `torch.xpu.{device_count, mem_get_info, memory_reserved, max_memory_allocated, empty_cache}` exist. `set_per_process_memory_fraction` may not, so guard it |
| allocator config at import | `exllamav3/__init__.py` (`_accelerator_setAllocatorSettings` / `torch.cuda.memory._set_allocator_settings`) | wrapped in try | harmless. Applies expandable segments to the current accelerator on new torch |
| stloader (pinned pool, `cudaMemcpyAsync`, events, bf16→fp16 kernel) | `loader/safetensors.py`, `stloader*.cpp/.cu` | **breaks** (hazard at `:793`) | `load_method="python"` |
| `device.type == "cuda"` load gates | `block_sparse_mlp.py:818` (**breaks**), `:888` (autosplit only), `model/config.py:74` (OK), `loader/safetensors.py:658` (arena off, OK), `model_ls.py:193` (measure off, OK) | | patch `:818` |
| `is_cuda` checks | `conv1d.py:245` (**breaks**), `conv1d.py:386` (routes to Triton, OK), `generator.py:1122` (slow path, OK) | | patch `:245` |
| `torch.cuda.device(...)` contexts | `vendor/fla/utils.py:117` (**breaks**), `conv1d.py:279`, `mla_triton.py` ×8, `dsa_triton.py` ×3 | | `torch.xpu.device` |
| `torch.cuda.get_device_properties().multi_processor_count` | `mla_triton.py:786` | breaks | XPU property |
| `torch.cuda.synchronize` | `generator/generator.py:1128` (**every decode step**), `util/misc.py:35-43` | breaks / silent no-op | `torch.xpu.synchronize` |
| pinned host memory + `non_blocking` H2D | generator staging (`generator.py:938,1113`, `job.py:1537`), `modules/embedding.py:164`, `util/tensor.py:162` | depends on torch-xpu | verify `pin_memory=True`; unpinned is correct, just synchronous |
| D2H/H2D stash copies | `modules/gated_delta_net.py:290-302` | OK | device-agnostic |
| CUDA graphs / streams / events | only inside C++ `BC_*` (`graph.cu`), `triton_kernel.cpp` (driver API), `block_sparse_mlp_cpu.py` / `moe_cpu_host.py` (offload, off) | avoided | optional later: `torch.xpu` graph capture of the eager decode step |
| `torch.compile`, flash-attn, xformers, ctypes CUDA | none on the path. MLA uses its own Triton for both `attn_mode="flash_attn"` and `"flash_attn_nc"`; xformers is lazily imported and unused; ctypes appears only in Windows file I/O and libc `malloc_trim` | n/a | — |
| `util/device_copy.py:39-40,81` | cuda→cuda P2P probe | not reached (single device) | — |

### 6.2 Triton kernels on the path

| group | kernels | file | CUDA-specific content | XPU risk / action |
|---|---|---|---|---|
| KDA chunk prefill (fla, vendored) | `l2norm_fwd_kernel`, `chunk_local_cumsum_vector_kernel`, `chunk_kda_fwd_kernel_intra_token_parallel`, `chunk_kda_fwd_kernel_inter_solve_fused`, `recompute_w_u_fwd_kda_kernel`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_gla_fwd_kernel_o` | `exllamav3/vendor/fla/*.py` | Inline PTX (`op.py:50-59`) is Blackwell-only and not taken. `libdevice` is imported at module level (`op.py:12`) and must resolve. `check_shared_mem` returns False on XPU, so the smallest tiles are used. Autotune configs use NVIDIA `num_warps`/`num_stages`. An fp32 `tl.dot` has default precision | `TRITON_F32_DEFAULT=ieee`; patch `utils.py:117`. Do **not** use glm-flash-lite `triton_pin.py`. Oracle: `torch_recurrent_kda` (`gated_delta_rule.py:64`) and `exllamav3/tests/test_fla_vendored_.py` |
| KDA conv | `_causal_conv1d_update_slotted_kernel`, `_output_kernel`, `_state_kernel` | `modules/gated_delta_net_fn/conv1d.py` | host checks only | patch `:245`/`:279` |
| MLA / DSA | `_mla_kv_update`, `_mla_plane_update`, `_mla_absorb`, `_mla_unfold`, `_mla_decode_split`/`_combine`, `_mla_gather_tile`, `_mla_prefill_mha`, `_dsa_indexer`, `_dsa_indexer_fewq`, `_dsa_attn_split`/`_combine`, `_dsa_attn` | `modules/attention_fn/mla_triton.py`, `dsa_triton.py` | No PTX, `tl.extra.cuda`, libdevice, fp8, tf32 or atomics. All dots are fp16×fp16→fp32. The wrappers use `torch.cuda.device` and the SM count. The `OutOfResources` retry loop in MHA prefill relies on the CUDA exception type. Zero-width NoPE tensors (`kpe`, `q_pe`) are passed as possibly-null pointers | Lower `block_m` to 64/32 for `_mla_prefill_mha` and `block_h` to 16 for `_dsa_attn` (or use Intel `grf_mode="large"`). Pass 1-element dummies if the Intel launcher rejects null pointers |
| not reached | `qsa_triton.py`, `triton_paged.py` (generic attention), `util/measures.py`, the BC-only `_mla_stage`/`_mla_idx_norm`/`_dsa_pool_update`/`_dsa_pool_expand`, `-cq`-only `_mla_kv_quant_scatter` | | | — |

Intel Triton maps `num_warps` to sub-groups. That does not affect correctness, but expect to re-tune. No
reached kernel uses fp8.

---

## 7. Minimal op set for a correct (not fast) forward

Configuration (section 3):
- `EXL3_BC_GDN=0`, `EXL3_BC_ATTN=0`, `TRITON_F32_DEFAULT=ieee`;
- `config.infer_params.no_reconstruct = True`, `load_method="python"`, single-device `xpu:0` load, `-nw`;
- no `-cq`, no `--mtp`, `GLM53_*` unset;
- the Python patches at `block_sparse_mlp.py:818`, `conv1d.py:245/279`, `vendor/fla/utils.py:117`, the
  MLA/DSA `torch.cuda.device` contexts and SM count, and `generator.py:1122/1128`.

Shim implementations needed:

| area | ops |
|---|---|
| **EXL3 / FP16 linear** (Part A) | `BC_LinearEXL3` (ctor fields + `run` + `run_alloc` = mul1 trellis decode → `diag(suh)·H·W_hat·H·diag(svh)` applied as `had128(x∘suh) @ W_hat → had128 ∘ svh`, fp32 accumulate, column-sliced for lm_head), `BC_LinearFP16`, `hgemm` |
| **MLP / MoE** (Part B) | `BC_GatedMLP` (Python wrapper), `silu_mul`, `routing_ds3_nogroup`, `det_quant_weight` (no-op) |
| **Hyper-connections / norms** (Part E) | `hc_mix`, `hc_mix_num_chunks`, `hc_apply`, `rms_norm` |
| **KDA** (Part C) | `cuda_recurrent_gated_delta_rule` (channelwise; history not needed), `gated_rms_norm`, `BC_GatedRMSNorm` |
| **MLA** (Part D) | `dsa_topk` |
| **Generator** (Part F) | `fused_sampler` + `FUSED_SAMPLER_MAX_BLOCKS` / `FUSED_SAMPLER_HIST_STRIDE`, `partial_strings_match` (only with stop strings), `cache_rotate` (or `enable_defrag=False`) |
| **Everything else in `bindings.cpp`** | name-only stubs that raise. `relu_mul` is read at import time |

That is **16 ext functions/classes with real bodies** (most of them small elementwise or reduction ops), plus
three groups of existing Triton kernels: 7 fla KDA kernels, 3 conv kernels, and about 12 MLA/DSA kernels.
They should run on Intel Triton with the noted launch-config tweaks. The one heavy piece is the EXL3
linear, and the existing trellis-serve SYCL kernels (`linear`, `exl3_reconstruct`, `had_in_rm`,
`had_out_h`) already match its layout bit for bit.

Suggested order to bring it up:
1. EXL3 linear reference, checked against the Part A torch reference on one real tensor.
2. `rms_norm`, `hc_mix`, `hc_apply`.
3. Routing, `silu_mul`, `BC_GatedMLP`.
4. KDA (decode recurrent op, then the fla chunk kernels on XPU).
5. MLA dense, then sparse with `dsa_topk`.
6. `fused_sampler`.
7. Compare logits with a CUDA run using `EXL3_INT8_GEMV=0 EXL3_BC_GDN=0`.
