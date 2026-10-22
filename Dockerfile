# syntax=docker/dockerfile:1.4
# Multi-stage Dockerfile for cloud/RunPod deployment.
#
# Stages:
#   sage-compile — apt + uv + PyTorch + SageAttention two-pass NVCC build.
#                  Cache-stable; only rebuilds when these deps change.
#                  Preserves dist/*.whl for use by the deps stage and
#                  for wheel extraction by the sage-wheels.yml workflow.
#   deps         — installs SageAttention wheel + project requirements
#                  on top of sage-compile. Rebuilds on requirements.txt changes.
#   runtime      — source copy + SSH + filebrowser + entrypoint.
#
# Build args (set at build time):
#   CUDA_ARCHITECTURES   Semicolon-separated SM targets. Default covers
#                        Ampere, Ada, Hopper, and Blackwell via PTX.
#   FILEBROWSER_VERSION  filebrowser release tag to install.
#
# Runtime env vars (set on pod, no image rebuild needed):
#   SSH_PUBLIC_KEY       Ed25519/RSA public key injected into /root/.ssh/authorized_keys
#   SSH_PORT             Port for sshd. sshd is skipped if unset.
#   FILEBROWSER_PORT     Port for filebrowser. Skipped if unset.
#   WGP_PROFILE          Override auto-detected WanGP profile (1-5)
#   WGP_ATTENTION        Override auto-detected attention mode (sage2/sage/sdpa)
#   WGP_ARGS             Extra arguments passed to wgp.py

ARG CUDA_ARCHITECTURES="8.0;8.6;8.9;9.0;10.0;12.0+PTX"
ARG FILEBROWSER_VERSION="2.32.0"

# ─────────────────────────────────────────────────────────────────────────────
# Stage: base
# Common base for both tools (compiler) and production images.
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS base

ARG CUDA_ARCHITECTURES
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip \
    git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 ffmpeg \
    openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED

RUN pip install --no-cache-dir uv==0.6.2
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_HTTP_TIMEOUT=300

# Install PyTorch cu128 (RTX 30/40)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    torchaudio==2.10.0+cu128 \
    torchao==0.16.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ─────────────────────────────────────────────────────────────────────────────
# Stage: sage-tools
# Full NVCC compiler stage. Targeted ONLY by the sage-wheels.yml robot.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS sage-tools

ENV FORCE_CUDA="1"
ARG CUDA_ARCHITECTURES
RUN pip install wheel packaging && \
    git clone --branch v2.2.0 --depth 1 \
    https://github.com/thu-ml/SageAttention.git /tmp/SageAttention && \
    cd /tmp/SageAttention && \
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}" && \
    MAX_JOBS=1 python3 setup.py bdist_wheel && \
    mkdir -p /tmp/sa_dist && cp dist/*.whl /tmp/sa_dist/ && \
    pip install --no-deps /tmp/sa_dist/*.whl && \
    rm -rf /tmp/SageAttention

# ─────────────────────────────────────────────────────────────────────────────
# Stage: sage-compile
# Fast production stage. Downloads pre-built wheels from GitHub Releases.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS sage-compile

RUN ssh-keygen -A && \
    sed -i \
    -e 's/#\?\(PermitRootLogin\).*/\1 yes/' \
    -e 's/#\?\(PubkeyAuthentication\).*/\1 yes/' \
    -e 's/#\?\(PasswordAuthentication\).*/\1 no/' \
    /etc/ssh/sshd_config && \
    mkdir -p /run/sshd

# ── SageAttention 2++ ────────────────────────────────────────────────────────
# We download ALL parallel factory wheels. entrypoint.sh installs the best match.
ARG SAGE_VERSION="2.2.0"
# REPO_OWNER defines where optimized wheels and kernel assets are hosted.
# Overridden by CI (${{ github.repository }}) or build_local.sh (git remote).
ARG REPO_OWNER="deepbeepmeep/Wan2GP"
RUN mkdir -p /opt/sage_wheels && \
    cd /opt/sage_wheels && \
    VTAG="v${SAGE_VERSION}" && \
    WURL="https://github.com/${REPO_OWNER}/releases/download/sage-${VTAG}-cu128-cp312" && \
    curl -L -O "${WURL}/sageattention-${SAGE_VERSION}+ampere.ada.rtx30.40-cp312-cp312-linux_x86_64.whl" && \
    curl -L -O "${WURL}/sageattention-${SAGE_VERSION}+hopper.h100.h200-cp312-cp312-linux_x86_64.whl" && \
    curl -L -O "${WURL}/sageattention-${SAGE_VERSION}+blackwell.rtx50-cp312-cp312-linux_x86_64.whl"

# ─────────────────────────────────────────────────────────────────────────────
# Stage: deps
# Adds project requirements on top of the sage-compile base.
# The sage-compile stage is a cache hit; only this stage rebuilds when
# requirements.txt changes.
# ─────────────────────────────────────────────────────────────────────────────
FROM sage-compile AS deps

# SageAttention is already installed in sage-compile's site-packages.
# /tmp/sa_dist/*.whl is preserved for sage-wheels.yml extraction.

# Project requirements.
# --index-strategy unsafe-best-match: uv stops at the first index that has a
# package name; this flag enables cross-index version resolution matching pip's
# default behaviour. Needed because the aiinfra nightly index appears first but
# does not have all versions (e.g. onnxruntime-gpu==1.22.0 is only on PyPI).
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    --index-strategy unsafe-best-match \
    -r /tmp/requirements.txt && \
    uv pip install -U --system --force-reinstall "huggingface-hub==0.36.2" git+https://github.com/huggingface/diffusers@main

# ─────────────────────────────────────────────────────────────────────────────
# Stage: runtime
# Source copy + optional services + entrypoint.
# Rebuilds on every source-code push; deps stage is a cache hit.
# ─────────────────────────────────────────────────────────────────────────────
FROM deps AS runtime

ARG FILEBROWSER_VERSION
ARG CUDA_ARCHITECTURES
ENV CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

# Filebrowser — lightweight browser-based file manager (optional, env-gated)
RUN curl -fsSL \
    "https://github.com/filebrowser/filebrowser/releases/download/v${FILEBROWSER_VERSION}/linux-amd64-filebrowser.tar.gz" \
    -o /tmp/fb.tar.gz && \
    tar -xzf /tmp/fb.tar.gz -C /usr/local/bin/ filebrowser && \
    rm /tmp/fb.tar.gz

# Project source baked into image (no host bind mount for cloud deployments).
# /workspace/ maps to the pod network volume and persists across restarts.
WORKDIR /workspace/wan2gp
COPY . .

RUN mkdir -p \
    /workspace/models \
    /workspace/outputs \
    /workspace/.cache/huggingface \
    /workspace/.cache/triton

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7860 22
ENTRYPOINT ["/entrypoint.sh"]
