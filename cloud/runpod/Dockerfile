# ==============================================================================
# Wan2GP RunPod Serverless Container (Official deepbeepmeep/Wan2GP + Compatibility Patches)
# ==============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# Install system dependencies & multimedia tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    curl \
    ca-certificates \
    build-essential \
    clang \
    libsox-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone official original Wan2GP repository (deepbeepmeep)
RUN git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git /workspace/Wan2GP

WORKDIR /workspace/Wan2GP

# Install complete dependencies (all requirements + media tools + wheels + RunPod SDK)
RUN pip install --no-cache-dir \
    -r requirements.txt \
    runpod>=1.7.0 \
    hf-transfer \
    audio-separator==0.36.1 \
    insightface==0.7.3 \
    facexlib==0.3.0 \
    pyannote.audio==3.3.2 \
    sherpa-onnx==1.13.2 \
    speechbrain==1.0.3 \
    torchcodec==0.10.0 \
    https://github.com/deepbeepmeep/smplfitter/releases/download/v0.2.10/smplfitter-0.2.10-py3-none-any.whl \
    https://github.com/deepbeepmeep/chumpy/releases/download/v0.71/chumpy-0.71-py3-none-any.whl

# Apply source patches to official Wan2GP
COPY patch_runpod.py /workspace/Wan2GP/patch_runpod.py
RUN python /workspace/Wan2GP/patch_runpod.py /workspace/Wan2GP

# Copy RunPod Serverless entrypoint
COPY rp_handler.py /workspace/Wan2GP/rp_handler.py

CMD ["python", "-u", "/workspace/Wan2GP/rp_handler.py"]
