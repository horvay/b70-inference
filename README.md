# EXL3 models on an Intel Arc Pro B70

Serving EXL3 models on one **Intel Arc Pro B70** (Battlemage, 32 GB) in a desktop with 62 GB of RAM and a PCIe 4 NVMe
SSD, including mixture-of-experts models far larger than the card: the experts a workload uses most stay on the card,
more sit in pinned RAM, and the rest come from an NVMe store; the router is nudged towards experts that are close by,
at a measured cost in quality.

| Model | Size | Decode, 1 user | Prompt | Host RAM | Server |
|---|---|---|---|---|---|
| [GLM-5.3-Flash](glm/README.md) 2.51 bpw EXL3 | ~105 GB | ~15-18 tok/s | 8K tokens in 14.4 s | ~30 GB | exllamav3 port (`glm/`) |
| [Qwen3.8-Flash-Next](flashnext/README.md) 3.05 bpw EXL3 | 68 GB | 41.7 tok/s (68 total for 2 users) | 18K tokens in 15.9 s | ~15 GB | SGLang + trellis-serve + patch |
| [Gemma 4 31B](gemma/README.md) 4.0 bpw EXL3 (dense) | 17 GB | 36-44 tok/s with its drafter (~107 total for 4) | 12K tokens in 6.7 s; context ~46K, ~107K packed | small | SGLang + trellis-serve + patch |

No Docker and no root: both servers run inside an unpacked copy of the `lmsysorg/sglang` XPU image (PyTorch 2.13 XPU,
Triton XPU, oneAPI 2026.0) through bubblewrap.

## Running it

You need Linux with the `xe` kernel driver, `bwrap` (bubblewrap), `git` and `python3` on the host, ~25 GB for the
image, the model, and NVMe space for the expert store (GLM ~95 GB, Flash-Next ~23 GB) on a filesystem where O_DIRECT
works (on btrfs, a `chattr +C` directory).

```sh
git clone --recursive https://github.com/horvay/b70-inference && cd b70-inference
./b70 setup                              # once, CPU only: the image, kernels and plugins for every model
./b70 prepare /path/to/glm-model         # once per GLM model: pack its NVMe expert store (Flash-Next fills its own)
./b70 serve /path/to/model --port 8000   # OpenAI-compatible server; Ctrl-C or `./b70 stop` stops it
```

`b70 serve` picks the server from the model's `config.json` (`gemma4`, `qwen4_exp` or `glm5_next`) and options:
`--ctx` (context tokens), `--parallel` (requests at once; GLM serves one), `--name` (model name in the API),
`--drafter` (Gemma: its assistant model), `--store` (expert store directory), `--chat-template` (a template to use
instead of the model's own), `--prepare` (run `prepare` first if needed). `./b70 info <model>` says whether setup and
prepare are done. The requested port opens only after one real warm-up request, so a client never meets a server that
is up but still compiling.

`serve` also guards the machine: it refuses to start with too little free RAM, stops the server at the first `xe`
fault, hang or reset for the B70 in the kernel log (faulting kernels that keep running have wedged the driver), when
RAM runs low or memory pressure gets high, or when another GPU job (`B70_OTHER_JOBS`, a regex over command lines)
opens the B70. The server runs with `oom_score_adj` 1000, so the OOM killer takes it first.

State (image, overlays, caches, stores, logs) lives in `state/` (`B70_STATE` elsewhere; `B70_ROOTFS` reuses an image
already unpacked).

## Layout

- `b70`: the command above.
- `common/`: the image puller (`oci_pull.py`, by digest, resumable, OCI whiteouts) and the image's environment.
- `glm/`: GLM-5.3-Flash on exllamav3 (submodule [horvay/exllamav3](https://github.com/horvay/exllamav3) `b70-xpu`) with
  the three-tier expert cache, SYCL/ESIMD and Triton kernels, and its server; history from
  [horvay/b70-glm53-flash](https://github.com/horvay/b70-glm53-flash).
- `trellis-serve.patch`: everything changed in trellis-serve (SGLang's EXL3 plugin and kernels) for Gemma and Flash-Next.
- `flashnext/`: the VRAM and RAM expert lists, and tools to rebuild them from your own workload.
- `gemma/`: notes and measurements.
