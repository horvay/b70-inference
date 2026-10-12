# Gemma 4 31B (dense, EXL3) on the B70

Gemma 4 runs on the same SGLang install as Qwen3.8-Flash-Next (`b70 setup sglang`): SGLang 0.5.20-xpu with 0xSero's
trellis-serve EXL3 kernels and `../trellis-serve.patch`. It fits the card, so there is no expert tiering; Google's
`gemma-4-31B-it-assistant` drafts for it.

```sh
b70 serve /path/to/gemma-4-31b-exl3 --port 8000 --drafter /path/to/gemma-4-31B-it-assistant [--parallel 4]
```

Measured with a 4-bit EXL3 Gemma 4 31B merge (Twilight Embrace), bf16 KV cache, decode graphs, drafter on:

| | |
|---|---|
| Decode, 1 game, short prompt | 36-39 tok/s (accept length ~2 of 3) |
| Decode, 1 game, 12K context | 33.5 tok/s |
| Decode, 4 games at once | 31-34 tok/s each, ~107 total |
| Prompt, 3.6K / 12K tokens | 2.7 s / 6.5 s (~1,300 / ~1,800 tok/s) |
| Follow-up turn over a cached 12K prompt | 0.5 s to the first token |
| Context | ~46K tokens (bf16 cache), ~107K with `--kv-pack` (on automatically above 45K) |
| Decode at 35K / 88K context (packed) | 28 / 16 tok/s; a fact from the start of an 88K prompt recalled |

Without the drafter, decode is ~23 tok/s: a step is 38.8 ms of GPU time, 28.6 of it the 4-bit EXL3 matrix kernels
reading ~15.5 GB of weights (about what the card's bandwidth allows) and 4.7 the bf16 output head.

What it needed:
- **Shared K/V layers.** Gemma 4's full-attention layers store no `v_proj`; SGLang loads `k_proj` into the V shard,
  and the plugin now gives that shard `k_proj`'s EXL3 layout.
- **Vision tower names.** The plugin shortened module names to `layers.N...`, so vision layer N picked up text layer
  N's EXL3 layout and crashed on the first request; `vision_tower` now keeps its own prefix (its weights stay bf16).
- **The drafter's view of the sliding-window cache.** SGLang's frozen-KV MTP drafter attends over the target's KV
  cache, which keeps the 50 sliding-window layers in a separate pool. The XPU attention backend decided once, from
  the drafter's own small pool, that there was no such pool, and read it at full-pool positions: fine for the first
  ~1K tokens, garbage after (accept length ~1.05, decode 17 tok/s at 12K). The plugin makes it read the pool in use.
- **Cache split.** The 50 sliding layers (16 KV heads x 256 x 2, ~820 KB per token) only need the 1024-token window
  plus a prefill chunk per request, so their pool is sized to exactly that (`EXL3_SWA_TOKENS`) and the 10
  full-attention layers get the rest. With 2048-token prompt chunks (as fast as 8192 here) that took the bf16 cache
  from 13K tokens (SGLang's default 0.8 ratio) to ~46K next to the drafter.
- **Packed cache (`--kv-pack`, kvpack.py).** The full-attention layers store keys rotated by a fixed orthogonal
  matrix and keys and values as 4-bit asymmetric groups of 32: 640 bytes per head and token against 2,048 in bf16,
  ~107K tokens. That is TurboQuant's first stage with uniform groups instead of its Lloyd-Max codebook (which did
  no better here); its QJL residual bit made attention clearly worse (KL 0.16 at 3 bits). Teacher-forced over 1,118
  reply tokens against the exact cache: same top token 98.0%, KL 0.003. Attention unpacks each request's pages into
  one reused bf16 scratch per layer (~450 GB/s) and runs Intel's flash attention on it: Triton decode kernels over
  the packed cache ran 17-70x slower than Intel's bf16 kernel, which reads 128-token pages near the card's
  bandwidth. Decode pays the unpack: +0.4 ms per layer at 16K tokens of context, +1.6 ms at 64K.
- **bf16 cache.** Intel's attention kernel has no fp8 path for the 512-wide global heads; SGLang's Triton attention
  takes fp8 but decoded at 16 tok/s and ran out of resources on a long prompt.

Before this, Gemma ran on vLLM with 0xSero's [exl3xpu](https://github.com/0xSero/exl3xpu) and the changes in
[horvay/vllm](https://github.com/horvay/vllm): ~2,000 tok/s prefill, ~38-41 tok/s decode with the drafter, 23.5 tok/s
each for four games. That path is no longer used.
