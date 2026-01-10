# LTX-2 Video Generation RunPod Serverless Handler
# Uses HuggingFace Diffusers for simplicity

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

# Pre-download the model to bake into image (faster cold starts)
# This downloads ~50GB+ of model files - will take a while but speeds up runtime
RUN python -c "import torch; from diffusers import LTX2Pipeline; LTX2Pipeline.from_pretrained('Lightricks/LTX-2', torch_dtype=torch.bfloat16)"

# Copy handler
COPY handler.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/root/.cache/huggingface

# Run handler
CMD ["python", "-u", "handler.py"]
