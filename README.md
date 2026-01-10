# LTX-2 Video Generation RunPod Worker

RunPod serverless worker for generating video clips using LTX-2 (Lightricks text-to-video model).

## Features

- Text-to-video generation using HuggingFace Diffusers
- 10-second clips at 768x512 resolution (configurable)
- Automatic upload to Supabase storage
- Pre-loaded model for fast inference

## GPU Requirements

- **Minimum:** NVIDIA GPU with 32GB+ VRAM
- **Recommended:** H100 80GB for 10-second 1080p videos

## Environment Variables

Set these in RunPod endpoint settings:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## Input Schema

```json
{
  "prompt": "A cinematic scene of a medieval castle at sunset",
  "project_id": "abc123",
  "clip_index": 0,
  "duration": 10,
  "width": 768,
  "height": 512,
  "fps": 24,
  "seed": 42
}
```

## Output Schema

```json
{
  "video_url": "https://your-project.supabase.co/storage/v1/object/public/generated-assets/abc123/clips/clip_000.mp4",
  "duration": 10.0,
  "width": 768,
  "height": 512,
  "generation_time": 120.5
}
```

## Local Testing

```bash
docker build -t ltx-video-runpod .
docker run --gpus all -e SUPABASE_URL=... -e SUPABASE_SERVICE_ROLE_KEY=... ltx-video-runpod
```

## Deployment

1. Push to GitHub: `git push origin main`
2. RunPod will automatically build from the linked repo
3. Configure endpoint with H100 80GB GPU
