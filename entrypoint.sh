#!/usr/bin/env bash
#
# entrypoint.sh
# Container entrypoint for cloud/RunPod deployment.
#
# Optionally starts sshd and filebrowser as background services, then
# detects the GPU at runtime to auto-select the WanGP profile (1-5) and
# attention mode (sage2/sage/sdpa) before launching wgp.py.
#
# Runtime overrides (env vars — no image rebuild needed):
#   SSH_PUBLIC_KEY      Public key appended to /root/.ssh/authorized_keys
#   SSH_PORT            Start sshd on this port. Skipped if unset.
#   FILEBROWSER_PORT    Start filebrowser on this port. Skipped if unset.
#   WGP_PROFILE         Force a specific WanGP profile (1-5)
#   WGP_ATTENTION       Force attention mode (sage2/sage/sdpa)
#   WGP_ARGS            Extra wgp.py arguments (e.g. "--compile --teacache 2.0")

set -euo pipefail

export PYTHONUNBUFFERED=1

# ── Library Path Sanitization ────────────────────────────────────────────────
# FIXED: Unset LD_LIBRARY_PATH to prevent "Library Shadowing" conflicts.
# NVIDIA base images (and some cloud providers) often set LD_LIBRARY_PATH to 
# include /usr/local/cuda/lib64. However, modern PyTorch (2.10.0+cu130) packages 
# its own optimized CUDA/cuBLAS binaries via pip (e.g., nvidia-cublas).
# If LD_LIBRARY_PATH is set, the dynamic linker prioritizes system stubs 
# over PyTorch's bundled libraries, leading to RuntimeError: CUBLAS_STATUS_INVALID_VALUE.
# Clearing this ensures PyTorch's internal search path (rpath) is respected.
unset LD_LIBRARY_PATH

# ── Persistent cache dirs (pod network volume) ───────────────────────────────
export HF_HOME=/workspace/.cache/huggingface
export TRITON_CACHE_DIR=/workspace/.cache/triton

# ── Privacy & Telemetry Disable ──────────────────────────────────────────────
export HF_HUB_DISABLE_TELEMETRY="1"
export TRANSFORMERS_DISABLE_TELEMETRY="1"
export HF_DATASETS_DISABLE_TELEMETRY="1"
export DO_NOT_TRACK="1"
export DISABLE_TELEMETRY="1"
export GRADIO_ANALYTICS_ENABLED="False"

# ── CPU thread tuning ────────────────────────────────────────────────────────
_nproc=$(nproc)
export OMP_NUM_THREADS=$_nproc
export MKL_NUM_THREADS=$_nproc
export OPENBLAS_NUM_THREADS=$_nproc
export NUMEXPR_NUM_THREADS=$_nproc

# ── TF32 acceleration (Ampere+) ──────────────────────────────────────────────
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# ── Audio dummy (suppress ALSA/PulseAudio noise in headless containers) ──────
export SDL_AUDIODRIVER=dummy
export PULSE_RUNTIME_PATH=/tmp/pulse-runtime

# ── Architecture hints ───────────────────────────────────────────────────────
# We derive TORCH_CUDA_ARCH_LIST from the image's CUDA_ARCHITECTURES.
# This ensures PyTorch uses the correct kernels (including Sm 100/120).
if [ -n "${CUDA_ARCHITECTURES:-}" ]; then
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
else
    export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0+PTX"
fi

# ── SSH key injection ─────────────────────────────────────────────────────────
_SSH_KEY="${SSH_PUBLIC_KEY:-${PUBLIC_KEY:-}}"
if [ -n "$_SSH_KEY" ]; then
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "$_SSH_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    echo "[INFO] SSH public key injected."
fi

# ── sshd ─────────────────────────────────────────────────────────────────────
# Only starts if SSH_PORT is explicitly set.
if [ -n "${SSH_PORT:-}" ]; then
    echo "[INFO] Starting sshd on port ${SSH_PORT}..."
    sed -i "s/#\?Port .*/Port ${SSH_PORT}/" /etc/ssh/sshd_config
    mkdir -p /run/sshd
    /usr/sbin/sshd
fi

# ── Filebrowser ───────────────────────────────────────────────────────────────
# Starts only if FILEBROWSER_PORT is set.
# SECURITY: Scoped to /workspace/ to prevent accidental system file deletion.
if [ -n "${FILEBROWSER_PORT:-}" ]; then
    echo "[INFO] Starting filebrowser on port ${FILEBROWSER_PORT}..."
    mkdir -p /root/.filebrowser
    nohup /usr/local/bin/filebrowser \
        --address 0.0.0.0 \
        --port "${FILEBROWSER_PORT}" \
        --database /root/.filebrowser/filebrowser.db \
        --root /workspace/ \
        --noauth \
        &>/root/.filebrowser/filebrowser.log &
fi

# ─────────────────────────────────────────────────────────────────────────────
# GPU detection helpers (ported from run-docker-cuda-deb.sh, but with added fallbacks)
# ─────────────────────────────────────────────────────────────────────────────

_detect_gpu_name() {
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "unknown"
}

_detect_vram_gb() {
    local mb
    mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    echo $(( mb / 1024 ))
}

# Maps GPU name + VRAM → WanGP profile number (1-5).
# Profile definitions (from WanGP UI):
#   1 HighRAM_HighVRAM  48GB+ RAM, 24GB+ VRAM
#   2 HighRAM_LowVRAM   48GB+ RAM, 12GB+ VRAM  (recommended for most)
#   3 LowRAM_HighVRAM   32GB+ RAM, 24GB+ VRAM
#   4 LowRAM_LowVRAM    32GB+ RAM, 12GB+ VRAM  (default)
#   5 VeryLowRAM_LowVRAM 16GB+ RAM, 10GB+ VRAM (fail-safe)
_map_profile() {
    local name="$1" vram_gb="$2"
    case "$name" in
        *"RTX 50"*|*"5090"*|*"5080"*|*"5070"*|\
        *"A100"*|*"A800"*|*"H100"*|*"H800"*)
            [ "$vram_gb" -ge 24 ] && echo 1 || echo 2 ;;
        *"RTX 40"*|*"4090"*|*"RTX 30"*|*"3090"*)
            [ "$vram_gb" -ge 24 ] && echo 3 || echo 2 ;;
        *"4080"*|*"4070"*|*"3080"*|*"3070"*|\
        *"RTX 20"*|*"2080"*|*"2070"*)
            [ "$vram_gb" -ge 12 ] && echo 2 || echo 4 ;;
        *"4060"*|*"3060"*|*"2060"*|*"GTX 16"*|*"1660"*|*"1650"*)
            [ "$vram_gb" -ge 10 ] && echo 4 || echo 5 ;;
        *"GTX 10"*|*"1080"*|*"1070"*|*"1060"*|*"Tesla"*)
            echo 5 ;;
        *)
            echo 4 ;;  # safe default
    esac
}

# Maps GPU name → attention mode
_map_attention() {
    local name="$1"
    case "$name" in
        *"RTX 50"*|*"RTX 40"*|*"RTX 30"*|\
        *"A100"*|*"A800"*|*"H100"*|*"H800"*)
            echo sage2 ;;
        *"RTX 20"*)
            echo sage ;;
        *)
            echo sdpa ;;
    esac
}

# ── Runtime GPU detection ─────────────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[WARN] nvidia-smi not found — GPU passthrough may be missing." >&2
    PROFILE=${WGP_PROFILE:-4}
    ATTN=${WGP_ATTENTION:-sdpa}
    GPU_NAME="unknown"
    VRAM_GB=0
else
    GPU_NAME=$(_detect_gpu_name)
    VRAM_GB=$(_detect_vram_gb)
    PROFILE=${WGP_PROFILE:-$(_map_profile "$GPU_NAME" "$VRAM_GB")}
    ATTN=${WGP_ATTENTION:-$(_map_attention "$GPU_NAME")}
fi

echo "[INFO] GPU: ${GPU_NAME} | VRAM: ${VRAM_GB}GB | Profile: ${PROFILE} | Attention: ${ATTN}"

# ── Dynamic SageAttention Install ───────────────────────────────────────────
# We detect the Compute Capability to install the corresponding native wheel.
if [ -d "/opt/sage_wheels" ] && command -v nvidia-smi >/dev/null 2>&1; then
    _CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 || echo "0.0")
    _MAJOR=$(echo "$_CAP" | cut -d. -f1)
    
    if [ "$_MAJOR" -eq 8 ]; then
        _W_SUFFIX="ampere.ada.rtx30.40"
    elif [ "$_MAJOR" -eq 9 ] || [ "$_MAJOR" -eq 10 ]; then
        _W_SUFFIX="hopper.h100.h200"
    elif [ "$_MAJOR" -eq 12 ]; then
        _W_SUFFIX="blackwell.rtx50"
    else
        _W_SUFFIX="blackwell.rtx50"
    fi

    echo "[INFO] Detected SM ${_CAP} - Selecting native SageAttention wheel part: ${_W_SUFFIX}"
    _WHEEL_PATH=$(ls /opt/sage_wheels/*+${_W_SUFFIX}-*.whl 2>/dev/null | head -1 || true)
    
    if [ -n "$_WHEEL_PATH" ]; then
        echo "[INFO] Installing native wheel: $(basename "$_WHEEL_PATH")"
        pip install --no-deps "$_WHEEL_PATH" --break-system-packages || echo "[ERROR] Failed to install wheel."
    else
        echo "[WARN] No matching wheel found for architecture suffix '${_W_SUFFIX}'."
    fi

    # ── Blackwell Ultra-Fast Path (NVFP4 / LightX2V) ──────────────────────────
    # Only triggers if hardware is Blackwell (sm_12x) AND we are in a cu13 image.
    if [[ "$_CAP" == "12."* ]] && [[ "$CONTAINER_CUDA_VERSION" == "13.0"* ]]; then
        echo "[INFO] Blackwell detected (sm_${_CAP}) - Activating NVFP4 (LightX2V) optimized path..."
        
        # Install the one-time build NVFP4 kernel
        _BW_WHEEL=$(ls /opt/bw_wheels/lightx2v_kernel-*.whl 2>/dev/null | head -1 || true)
        if [ -n "$_BW_WHEEL" ] && [ -f "$_BW_WHEEL" ]; then
            echo "[INFO] Installing native NVFP4 kernel: $(basename "$_BW_WHEEL")"
            pip install --no-deps "$_BW_WHEEL" || echo "[ERROR] NVFP4 kernel install failed."
        else
            echo "[WARN] Blackwell hardware detected but no valid NVFP4 kernel found in /opt/bw_wheels."
        fi
    fi
fi

# ── Launch ────────────────────────────────────────────────────────────────────
cd /workspace/wan2gp
exec python3 wgp.py \
    --listen \
    --profile "${PROFILE}" \
    --attention "${ATTN}" \
    ${WGP_ARGS:-} \
    "$@"
