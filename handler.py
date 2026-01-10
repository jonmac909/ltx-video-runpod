"""
LTX-2 Video Generation RunPod Serverless Handler
Generates video clips from text prompts using HuggingFace Diffusers.
"""

import os
import time
import tempfile
import traceback
from typing import Optional

import runpod
import torch
import numpy as np
from PIL import Image
from supabase import create_client, Client

# Initialize model as global for warm starts
PIPE = None


def get_supabase_client() -> Client:
    """Create Supabase client from environment variables."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


def load_model():
    """Load LTX-2 model once and keep it warm."""
    global PIPE
    if PIPE is not None:
        return PIPE

    print("[LTX-2] Loading model...")
    start = time.time()

    from diffusers import LTX2Pipeline

    PIPE = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
    )

    # Enable memory optimizations
    PIPE.enable_model_cpu_offload()

    print(f"[LTX-2] Model loaded in {time.time() - start:.1f}s")
    return PIPE


def generate_video(
    prompt: str,
    duration_seconds: float = 10.0,
    width: int = 768,
    height: int = 512,
    fps: float = 24.0,
    num_inference_steps: int = 40,
    guidance_scale: float = 4.0,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    seed: Optional[int] = None,
) -> str:
    """
    Generate a video from a text prompt.

    Args:
        prompt: Text description of the video to generate
        duration_seconds: Video duration in seconds (max ~10s recommended)
        width: Video width in pixels (must be divisible by 32)
        height: Video height in pixels (must be divisible by 32)
        fps: Frames per second
        num_inference_steps: Denoising steps (more = higher quality but slower)
        guidance_scale: CFG scale (higher = closer to prompt but less natural)
        negative_prompt: What to avoid in the video
        seed: Random seed for reproducibility

    Returns:
        Path to the generated video file
    """
    from diffusers.pipelines.ltx2.export_utils import encode_video

    pipe = load_model()

    # Calculate number of frames (must be divisible by 8 + 1)
    num_frames = int(duration_seconds * fps)
    num_frames = ((num_frames - 1) // 8) * 8 + 1  # Ensure divisible by 8 + 1

    # Ensure dimensions are divisible by 32
    width = (width // 32) * 32
    height = (height // 32) * 32

    print(f"[LTX-2] Generating {duration_seconds:.1f}s video ({num_frames} frames, {width}x{height})")
    print(f"[LTX-2] Prompt: {prompt[:100]}...")

    # Set up generator for reproducibility
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()

    # Generate video (returns video and audio tensors)
    video, audio = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
        return_dict=False,
    )

    gen_time = time.time() - start
    print(f"[LTX-2] Video generated in {gen_time:.1f}s ({gen_time/duration_seconds:.1f}x realtime)")

    # Convert to uint8
    video = (video * 255).round().astype("uint8")
    video_tensor = torch.from_numpy(video)

    # Save to temp file (video only, no audio - we use our own TTS)
    output_path = tempfile.mktemp(suffix=".mp4")

    # Use imageio for simple video writing without audio
    import imageio
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,  # 0-10, higher is better
        pixelformat="yuv420p",
    )

    # video_tensor shape: [batch, frames, height, width, channels]
    for frame in video_tensor[0]:
        writer.append_data(frame.numpy())
    writer.close()

    print(f"[LTX-2] Video saved to {output_path}")
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

    print(f"[LTX-2] Uploading to Supabase: {storage_path}")

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

    print(f"[LTX-2] Uploaded: {public_url}")
    return public_url


def handler(job: dict) -> dict:
    """
    RunPod handler for video generation.

    Expected input:
    {
        "prompt": "A cinematic scene of...",
        "project_id": "abc123",
        "clip_index": 0,
        "duration": 10,  # optional, default 10s
        "width": 768,    # optional, default 768
        "height": 512,   # optional, default 512
        "fps": 24,       # optional, default 24
        "seed": 42,      # optional
    }

    Returns:
    {
        "video_url": "https://...",
        "duration": 10.0,
        "width": 768,
        "height": 512,
        "generation_time": 120.5
    }
    """
    try:
        job_input = job.get("input", {})

        # Required fields
        prompt = job_input.get("prompt")
        project_id = job_input.get("project_id")
        clip_index = job_input.get("clip_index", 0)

        if not prompt:
            return {"error": "prompt is required"}
        if not project_id:
            return {"error": "project_id is required"}

        # Optional fields with defaults
        duration = job_input.get("duration", 10)
        width = job_input.get("width", 768)
        height = job_input.get("height", 512)
        fps = job_input.get("fps", 24)
        seed = job_input.get("seed")
        num_inference_steps = job_input.get("num_inference_steps", 40)
        guidance_scale = job_input.get("guidance_scale", 4.0)
        negative_prompt = job_input.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted"
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
        print(f"[LTX-2] Error: {e}")
        traceback.print_exc()
        return {"error": str(e)}


# Start the RunPod serverless handler
if __name__ == "__main__":
    print("[LTX-2] Starting RunPod serverless handler...")

    # Pre-load model for faster first request
    try:
        load_model()
    except Exception as e:
        print(f"[LTX-2] Warning: Failed to pre-load model: {e}")

    runpod.serverless.start({"handler": handler})
