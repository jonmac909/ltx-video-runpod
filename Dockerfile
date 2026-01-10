# WAN 2.2 Video Generation RunPod Serverless Handler
# Uses HuggingFace Diffusers WanPipeline (720p@24fps optimized)
# v1.1.1 - Fixed model ID: Wan-AI/Wan2.2-T2V-A14B-Diffusers (A14B not 14B)

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Verify all imports work (model downloaded at runtime on first request)
RUN python -c "from diffusers import WanPipeline, AutoModel; from transformers import UMT5EncoderModel; import torch; print('All WAN 2.2 imports successful')"

# Copy handler
COPY handler.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_VISIBLE_DEVICES=0
# Set HF cache to container disk (100GB)
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/tmp/hf_cache

# Run handler
CMD ["python", "-u", "handler.py"]
