# Gemma 4 31B (dense, EXL3) on the B70

Dense models that fit the card's 32 GB do not need expert tiers, so they are not served from this repo: they run on
0xSero's [exl3xpu](https://github.com/0xSero/exl3xpu) (vLLM XPU plugin with EXL3 ESIMD kernels), with vLLM changes kept
in [horvay/vllm](https://github.com/horvay/vllm) (branch `main`, based on the commit the exl3xpu image was built from).

Measured with a 4-bit EXL3 Gemma 4 31B merge: ~2,190 tok/s prompt reading at 3K, ~1,945 at 12K; decode ~38 tok/s with
XPU decode graphs, ~33-41 with Google's `gemma-4-31B-it-assistant` drafter at 2 draft tokens (llama.cpp SYCL Q4_K_M
for comparison: ~1,129 / 930 prefill, ~25 decode).

What it needed on top of the published image:
- the plugin treats V bits as K's for Gemma 4's K=V attention layers, and leaves a tied `lm_head` unquantized
  (`EXL3_TIED_LM_HEAD=1`);
- `--attention-backend FLASH_ATTN` (text only; vLLM's Triton fallback for the multimodal prefix-LM mask is ~30x slower
  at 12K);
- `TreatNonUsmForTransfersAsSharedSystem=0` and `EnableSharedSystemUsmSupport=0`: host copies through staging buffers,
  because the copy engine reading a swapped page in place faulted the card while loading;
- decode graphs (`FULL_DECODE_ONLY`) after reading a lazy `layer_scalar.item()` at load, and a pruned draft vocabulary
  (~11K tokens) for the drafter: both in the vLLM fork;
- several users at once: vLLM's CPU offloader with a lazy Intel copy path (fork), sliding-window layers caching only
  the tail a follow-up turn reuses.
