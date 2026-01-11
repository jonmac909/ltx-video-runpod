"""
WAN 2.2 Video Generation RunPod Serverless Handler
Generates video clips from text prompts using HuggingFace Diffusers.
Optimized for 720p@24fps on 80GB GPUs (A100/H100).

Optimal settings based on official docs:
- flow_shift: 5.0 for 720p, 3.0 for 480p
- num_inference_steps: 50 (default), 30 for speed
- guidance_scale: 5.0 (range 5-7, too high = flicker)
"""

import os
import time
import tempfile
import traceback
import shutil
from typing import Optional

import runpod
import torch
import numpy as np
from supabase import create_client, Client

# Set HF cache to container disk (not system disk)
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/tmp/hf_cache"
# Increase download timeout for large model files (default 10s is too short)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes per file
# Enable download progress logging
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable hf_transfer for better logging

# Initialize model as global for warm starts
PIPE = None


def log_disk_space(label: str):
    """Log disk space for debugging."""
    try:
        total, used, free = shutil.disk_usage("/")
        print(f"[DISK {label}] Total: {total // (1024**3)}GB, Used: {used // (1024**3)}GB, Free: {free // (1024**3)}GB")

        # Also check /tmp specifically
        tmp_total, tmp_used, tmp_free = shutil.disk_usage("/tmp")
        print(f"[DISK {label}] /tmp - Total: {tmp_total // (1024**3)}GB, Used: {tmp_used // (1024**3)}GB, Free: {tmp_free // (1024**3)}GB")
    except Exception as e:
        print(f"[DISK {label}] Error checking disk: {e}")


def get_supabase_client() -> Client:
    """Create Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


def load_model():
    """Load WAN 2.2 model once and keep it warm."""
    global PIPE
    if PIPE is not None:
        return PIPE

    log_disk_space("BEFORE_LOAD")

    print("[WAN2.2] Loading model components...")
    start = time.time()

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

    # Load VAE with float32 for better decoding quality
    print("[WAN2.2] Loading VAE (float32 for quality)...", flush=True)
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    print(f"[WAN2.2] VAE loaded in {time.time()-t0:.1f}s", flush=True)

    # Load full pipeline - A100 80GB has enough VRAM, no offloading needed!
    print("[WAN2.2] Loading pipeline (A100 80GB - no offloading needed)...", flush=True)
    t0 = time.time()
    PIPE = WanPipeline.from_pretrained(
        model_id,
        vae=vae,
        torch_dtype=torch.bfloat16
    )
    print(f"[WAN2.2] Pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    # Configure scheduler with optimal flow_shift for 720p
    PIPE.scheduler = UniPCMultistepScheduler.from_config(
        PIPE.scheduler.config,
        flow_shift=5.0  # Optimal for 720p resolution
    )

    # Move everything to GPU - A100 80GB can handle it
    print("[WAN2.2] Moving to CUDA...", flush=True)
    PIPE.to("cuda")

    log_disk_space("AFTER_LOAD")
    print(f"[WAN2.2] Model loaded in {time.time() - start:.1f}s")
    return PIPE


def generate_video(
    prompt: str,
    duration_seconds: float = 10.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 24.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, ugly, deformed, disfigured",
    seed: Optional[int] = None,
) -> str:
    """
    Generate a video from a text prompt using WAN 2.2.

    Args:
        prompt: Text description of the video to generate
        duration_seconds: Video duration in seconds (10s default for 80GB GPU)
        width: Video width in pixels (1280 for 720p)
        height: Video height in pixels (720 for 720p)
        fps: Frames per second (24 recommended)
        num_inference_steps: Denoising steps (50 default, 30 for speed)
        guidance_scale: CFG scale (5.0 recommended, 5-7 range)
        negative_prompt: What to avoid in the video
        seed: Random seed for reproducibility

    Returns:
        Path to the generated video file
    """
    pipe = load_model()

    # Calculate number of frames: must be 4k + 1 for WAN
    # For 5s at 24fps = 120 frames, nearest 4k+1 = 121 (k=30)
    raw_frames = int(duration_seconds * fps)
    k = (raw_frames - 1) // 4
    num_frames = 4 * k + 1

    # Ensure dimensions are divisible by 16 (WAN requirement)
    width = (width // 16) * 16
    height = (height // 16) * 16

    print(f"[WAN2.2] Generating {duration_seconds:.1f}s video ({num_frames} frames, {width}x{height})")
    print(f"[WAN2.2] Prompt: {prompt[:100]}...")

    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()

    # Progress callback to log step completion
    def progress_callback(pipe, step, timestep, callback_kwargs):
        pct = int((step + 1) / num_inference_steps * 100)
        print(f"[WAN2.2] Step {step + 1}/{num_inference_steps} ({pct}%)", flush=True)
        return callback_kwargs

    # Generate video (flow_shift=5.0 configured in scheduler for 720p)
    print(f"[WAN2.2] Starting generation with {num_inference_steps} steps, guidance={guidance_scale}...", flush=True)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        callback_on_step_end=progress_callback,
    )

    # Debug: Print output structure
    print(f"[WAN2.2] Output type: {type(output)}")
    print(f"[WAN2.2] Output frames type: {type(output.frames)}")
    print(f"[WAN2.2] Output frames length: {len(output.frames) if hasattr(output.frames, '__len__') else 'N/A'}")

    if len(output.frames) == 0:
        raise ValueError("Pipeline returned empty frames - generation failed")

    # Get video frames
    video = output.frames[0]
    print(f"[WAN2.2] Video shape: {video.shape if hasattr(video, 'shape') else type(video)}")

    gen_time = time.time() - start
    print(f"[WAN2.2] Video generated in {gen_time:.1f}s ({gen_time/duration_seconds:.1f}x realtime)")

    # Save to temp file using diffusers export helper
    output_path = tempfile.mktemp(suffix=".mp4")

    from diffusers.utils import export_to_video
    export_to_video(video, output_path, fps=int(fps))

    print(f"[WAN2.2] Video saved to {output_path}")
    return output_path


def upload_to_supabase(video_path: str, project_id: str, clip_index: int) -> str:
    """
    Upload video to Supabase storage.

    Args:
        video_path: Local path to the video file
        project_id: Project ID for organizing in storage
        clip_index: Index of this clip (0-based)

    Returns:
        Public URL of the uploaded video
    """
    supabase = get_supabase_client()
    bucket = "generated-assets"

    # Create storage path
    storage_path = f"{project_id}/clips/clip_{clip_index:03d}.mp4"

    print(f"[WAN2.2] Uploading to Supabase: {storage_path}")

    with open(video_path, "rb") as f:
        video_data = f.read()

    # Upload with upsert to overwrite if exists
    result = supabase.storage.from_(bucket).upload(
        storage_path,
        video_data,
        file_options={"content-type": "video/mp4", "upsert": "true"}
    )

    # Get public URL
    public_url = supabase.storage.from_(bucket).get_public_url(storage_path)

    print(f"[WAN2.2] Uploaded: {public_url}")
    return public_url


def handler(job: dict) -> dict:
    """
    RunPod handler for video generation.

    Expected input:
    {
        "prompt": "A cinematic scene of...",
        "project_id": "abc123",
        "clip_index": 0,
        "duration": 5,       # optional, default 5s (max ~8s for 720p on 48GB)
        "width": 1280,       # optional, default 1280 (720p)
        "height": 720,       # optional, default 720 (720p)
        "fps": 24,           # optional, default 24
        "seed": 42,          # optional
    }

    Returns:
    {
        "video_url": "https://...",
        "duration": 5.0,
        "width": 1280,
        "height": 720,
        "generation_time": 120.5
    }
    """
    try:
        log_disk_space("HANDLER_START")

        job_input = job.get("input", {})

        # Required fields
        prompt = job_input.get("prompt")
        project_id = job_input.get("project_id")
        clip_index = job_input.get("clip_index", 0)

        if not prompt:
            return {"error": "prompt is required"}
        if not project_id:
            return {"error": "project_id is required"}

        # Optional fields with defaults optimized for 720p
        duration = job_input.get("duration", 5)
        width = job_input.get("width", 1280)
        height = job_input.get("height", 720)
        fps = job_input.get("fps", 24)
        seed = job_input.get("seed")
        num_inference_steps = job_input.get("num_inference_steps", 30)
        guidance_scale = job_input.get("guidance_scale", 5.0)
        negative_prompt = job_input.get(
            "negative_prompt",
            "Bright tones, overexposed, static, blurred details, subtitles, worst quality, low quality, ugly, deformed, disfigured"
        )

        start_time = time.time()

        # Generate video
        video_path = generate_video(
            prompt=prompt,
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
        )

        # Upload to Supabase
        video_url = upload_to_supabase(video_path, project_id, clip_index)

        # Clean up temp file
        try:
            os.remove(video_path)
        except:
            pass

        generation_time = time.time() - start_time

        return {
            "video_url": video_url,
            "duration": duration,
            "width": width,
            "height": height,
            "generation_time": round(generation_time, 1),
        }

    except Exception as e:
        print(f"[WAN2.2] Error: {e}")
        traceback.print_exc()
        log_disk_space("ERROR")
        return {"error": str(e)}


# Start the RunPod serverless handler
if __name__ == "__main__":
    print("[WAN2.2] Starting RunPod serverless handler...")
    log_disk_space("STARTUP")

    # Pre-load model for faster first request
    try:
        load_model()
    except Exception as e:
        print(f"[WAN2.2] Warning: Failed to pre-load model: {e}")
        log_disk_space("PRELOAD_FAILED")

    runpod.serverless.start({"handler": handler})

