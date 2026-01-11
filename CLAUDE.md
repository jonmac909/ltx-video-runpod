# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RunPod serverless worker for WAN 2.2 text-to-video generation. Generates 720p video clips from text prompts using HuggingFace Diffusers, uploads to Supabase storage.

**Model:** `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (14B parameters, ~27GB)
**Target GPU:** A100 80GB (no memory offloading needed)

## Commands

```bash
# Build Docker image locally
docker build -t ltx-video-runpod .

# Test locally (requires NVIDIA GPU)
docker run --gpus all \
  -e SUPABASE_URL=... \
  -e SUPABASE_SERVICE_ROLE_KEY=... \
  ltx-video-runpod

# Deploy to RunPod (auto-builds on push)
git push origin main
```

## Architecture

Single-file handler (`handler.py`) with three main functions:

1. **`load_model()`** - Loads WAN 2.2 pipeline once, keeps warm for subsequent requests
2. **`generate_video()`** - Generates video frames, exports to MP4
3. **`handler()`** - RunPod entry point, orchestrates generation + upload

Model is stored in `/tmp/hf_cache` (container disk, not system disk) to handle the ~40GB model files.

## Critical Performance Notes

**DO NOT use `leaf_level` group offloading** - It moves individual layers between CPU/GPU and causes 100x slowdown (8+ min/step instead of 5s/step).

On A100 80GB, the model fits entirely in VRAM (~75GB peak). Just use:
```python
pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

## WAN 2.2 Constraints

- Frame count must be `4k + 1` (e.g., 81, 121, 237)
- Dimensions must be divisible by 16
- `flow_shift=5.0` for 720p, `3.0` for 480p
- `guidance_scale` range 5-7 (too high = flicker)
- Expected performance: ~3-5 seconds per inference step

## Environment Variables (RunPod)

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>
```

## Triggering Rebuilds

Push to main branch triggers RunPod auto-build. If build gets stuck, push a trivial change to trigger a fresh webhook.
