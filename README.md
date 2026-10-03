# YuE2UI

Small self-hosted web UI for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) (frontier open music generation with editable scores), with first-class **AMD ROCm** support.

Built and battle-tested on an **RX 7800 XT (16 GB, gfx1101)** running Linux + ROCm 7.1.

![screenshot](screenshot.png)

## Features

- **Generate songs** from style + lyrics — full songs with vocals and accompaniment, 48 kHz stereo
- **Preview / full quality** — 16 vs 32 NAR steps (same take, different render); 16 steps is the house default after listening tests
- **Editable ABC scores** — every song ships with the melody+chord plan; paste a score to force the composition
- **Plan-reuse reroll** — re-render the same plan+seed at full quality instantly skipping the AR planning stage, or write new lyrics over an existing melody
- **Live progress** — stage + token counts while the GPU works
- **Library** — every generation is kept with its audio, score, latents and settings
- **One job at a time** — internal queue so the GPU is never double-booked

## The ROCm bits (why this fork-of-a-workflow exists)

Upstream YuE2 targets NVIDIA (24 GB). Three fixes make it work well on AMD:

1. **Attention backend** — PyTorch's fused flash-attention op rejects `seqused_k` on ROCm, and the forced cuDNN backend has no kernel. Fix: plain SDPA (works on ROCm, even inside CUDA graph capture). Patches `YuE/src/yue2/cuda_graph.py` at install time.
2. **Long-song OOM** — the NAR acoustic stage materializes a full `[heads, seq, seq]` score matrix (~8.4 GiB at ~12k frames) because ROCm falls back to the math SDPA kernel. Fix: pass `query_chunk_size` through to the NAR solver (upstream already supports it; the pipeline just never wired it). Patches `YuE/src/yue2/pipeline.py`. Tunable via `YUE2_NAR_QUERY_CHUNK` (default 1024).
3. **MIOpen JIT stalls** — the VAE decode pays a ~40–90 s per-shape kernel search on first use, per process. `MIOPEN_FIND_MODE=FAST` (set by `run.sh`) cuts that to <1 s with identical output.

## Install

Linux. Requires an AMD GPU with [ROCm](https://rocm.docs.amd.com/) installed (7.x recommended) or an NVIDIA GPU with BF16 support. ~16 GB VRAM is the tested budget for long songs.

```bash
git clone https://github.com/Alexthestampede/YuE2UI.git
cd YuE2UI
./install.sh
./run.sh
```

First launch downloads ~8 GB of model weights from Hugging Face and generates a test song. Model files are cached in `~/.cache/huggingface`.

## Timing expectations (RX 7800 XT, idle GPU)

| Song | First render (plan, preview) | Re-render (no plan) | Full quality |
|---|---|---|---|
| 1 min | ~5 min | ~4 min | ~5 min |
| 4 min | ~25 min | ~18 min | ~30 min |

The dominant cost is YuE2's autoregressive semantic stage (~6–14 tok/s on RDNA3); it is quality-competitive with Suno v5/v6, which is exactly why it is not turbo-diffusion fast. Don't game on the same GPU unless you enjoy 3× slower renders.

## Tips

- **Approve takes in preview, render the keeper at full.** Preview and full are the *same performance* — only the final render pass differs.
- **New take, same melody:** copy a song's ABC into the score box, clear the seed box, regenerate. New lyrics over an old melody: same, but edit section/line counts to match the score's structure.
- **Longer lyrics = longer songs**, roughly linearly; very long songs chunk the NAR stage (that's the OOM fix above doing its job).

## Acknowledgements

- [YuE / YuE2](https://github.com/multimodal-art-projection/YuE) by M·A·P — the model and pipeline (Apache 2.0 code, CC BY-NC 4.0 weights — **non-commercial**)
- Workflow patterns borrowed from my [SqualusShiraii](https://github.com/Alexthestampede/SqualusShiraii) ACE-Step UI

## License

Apache 2.0 for this UI. Model weights are CC BY-NC 4.0: generated songs are free to use and monetize personally, but commercial use *by companies* requires contacting the YuE2 authors.