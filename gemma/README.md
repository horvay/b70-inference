# Gemma 4 31B (dense, EXL3) on the B70

Gemma 4 runs on the same SGLang install as Qwen3.8-Flash-Next (`b70 setup sglang`): SGLang 0.5.20-xpu with 0xSero's
trellis-serve EXL3 kernels and `../trellis-serve.patch`. It fits the card, so there is no expert tiering; Google's
`gemma-4-31B-it-assistant` drafts for it.

```sh
b70 serve /path/to/gemma-4-31b-exl3 --port 8000 --drafter /path/to/gemma-4-31B-it-assistant [--parallel 4]
```

Measured with a 4-bit EXL3 Gemma 4 31B merge (Twilight Embrace), decode graphs, drafter on (300-token replies,
EOS ignored):

| | |
|---|---|
| Decode, 1 / 2 / 4 games, short prompts, bf16 cache | 43.5 / 68.5 / 127 tok/s in total |
| Same, sliding-window layers in fp8 (`--kv-pack window`) | 42 / 91 / 135 tok/s in total |
| Same, full-attention layers also packed (`--kv-pack all`) | 45 / 73 / 104 tok/s in total |
| Decode, 4 games at ~20K context each (packed) | 76 tok/s in total, 19 each |
| Decode, 1 game, 12K context (bf16) | 33.5 tok/s |
| Prompt, 3.6K / 12K / 20K tokens | 2.7 s / 6.5 s / 18 s (~1,100-1,800 tok/s) |
| Next turn over a cached 20K prompt, four games taking turns | 0.8 s to the first token (all four stay cached) |
| Context, 1 game | ~46K tokens (bf16), ~60K (fp8 windows), ~107K (packed) |
| Context, 4 games together | ~32K tokens (fp8 windows), ~87K (packed) |
| Decode at 35K / 88K context (packed, 1 game) | 28 / 16 tok/s; a fact from the start of an 88K prompt recalled |

`--kv-pack auto` (the default) keeps one game on the exact bf16 cache while its context fits, stores the windows in
fp8 for two or more games or a longer context, and packs the full-attention layers too when `--ctx` x `--parallel`
does not fit unpacked.

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
  plus a prefill chunk per request, and one window for each other game's cached prompt, so their pool is sized to
  exactly that (`EXL3_SWA_TOKENS`) and the 10 full-attention layers get the rest. With 2048-token prompt chunks (as
  fast as 8192 here) that took the bf16 cache from 13K tokens (SGLang's default 0.8 ratio) to ~46K next to the
  drafter.
- **Several games' prompts cached at once.** SGLang caches a long prompt chunk by chunk, so every chunk's window
  slots belonged to the prompt cache, and the running request could not free them as they left its window. Being the
  newest, they outlived every other game's cached window: with two games, each turn re-read its whole prompt (18 s
  for 20K tokens instead of 0.8). kvpack.py moves a request's nodes outside the window that a later match needs (the
  one behind its prompt's end; a reply's thinking is not in the next prompt) to the eviction end of the window LRU.
- **fp8 sliding windows (`--kv-pack window`).** Intel's flash attention reads an fp8 cache for heads up to 256 wide
  with a per-tensor descale; the 512-wide full-attention heads have no fp8 path. Teacher-forced against the exact
  cache: same top token 97.8%, KL 0.0065. Its first runs decoded slower than bf16: the drafter's 3-token verify ran
  the windowed layers through the slow prefill path. The plugin now serves verify rows as decode queries for windowed
  layers too (4 games x 3 queries: 435 -> 199 us per layer in bf16, 979 -> 106 us in fp8).
- **Packed cache (`--kv-pack all`, kvpack.py).** The full-attention layers store keys rotated by a fixed orthogonal
  matrix and keys and values as 4-bit asymmetric groups of 32: 640 bytes per head and token against 2,048 in bf16,
  ~107K tokens. That is TurboQuant's first stage with uniform groups instead of its Lloyd-Max codebook (which did
  no better here); its QJL residual bit made attention clearly worse (KL 0.16 at 3 bits). Teacher-forced over 1,118
  reply tokens against the exact cache: same top token 98.0%, KL 0.003. Attention unpacks each request's pages into
  one reused bf16 scratch per layer (~450 GB/s) and runs Intel's flash attention on it: Triton decode kernels over
  the packed cache ran 17-70x slower than Intel's bf16 kernel, which reads 128-token pages near the card's
  bandwidth. Decode pays the unpack: +0.4 ms per layer at 16K tokens of context, +1.6 ms at 64K.
- **No SGLang fp8 cache.** SGLang's own `--kv-cache-dtype fp8` covers both kinds of layer, and Intel's kernel cannot
  read the 512-wide ones in fp8; SGLang's Triton attention takes fp8 but decoded at 16 tok/s and ran out of resources
  on a long prompt. So SGLang's cache type stays bf16 and kvpack.py replaces each half of the cache.

Before this, Gemma ran on vLLM with 0xSero's [exl3xpu](https://github.com/0xSero/exl3xpu) and the changes in
[horvay/vllm](https://github.com/horvay/vllm): ~2,000 tok/s prefill, ~38-41 tok/s decode with the drafter, 23.5 tok/s
each for four games. That path is no longer used.
