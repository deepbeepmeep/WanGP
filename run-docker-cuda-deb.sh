#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────── helpers ─────────────────────────

install_nvidia_smi_if_missing() {
    if command -v nvidia-smi &>/dev/null; then
        return
    fi

    echo "⚠️  nvidia-smi not found. Installing nvidia-utils…"
    if [ "$EUID" -ne 0 ]; then
        SUDO='sudo'
    else
        SUDO=''
    fi

    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-utils-535 || $SUDO apt-get install -y nvidia-utils

    if ! command -v nvidia-smi &>/dev/null; then
        echo "❌ Failed to install nvidia-smi. Cannot detect GPU architecture."
        exit 1
    fi
    echo "✅ nvidia-smi installed successfully."
}

detect_gpu_name() {
    install_nvidia_smi_if_missing
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1
}

map_gpu_to_arch() {
    local name="$1"
    case "$name" in
    *"RTX 50"* | *"5090"* | *"5080"* | *"5070"*) echo "12.0" ;;
    *"H100"* | *"H800"*) echo "9.0" ;;
    *"RTX 40"* | *"4090"* | *"4080"* | *"4070"* | *"4060"*) echo "8.9" ;;
    *"RTX 30"* | *"3090"* | *"3080"* | *"3070"* | *"3060"*) echo "8.6" ;;
    *"A100"* | *"A800"* | *"A40"*) echo "8.0" ;;
    *"Tesla V100"*) echo "7.0" ;;
    *"RTX 20"* | *"2080"* | *"2070"* | *"2060"* | *"Titan RTX"*) echo "7.5" ;;
    *"GTX 16"* | *"1660"* | *"1650"*) echo "7.5" ;;
    *"GTX 10"* | *"1080"* | *"1070"* | *"1060"* | *"Tesla P100"*) echo "6.1" ;;
    *"Tesla K80"* | *"Tesla K40"*) echo "3.7" ;;
    *)
        echo "❌ Unknown GPU model: $name"
        echo "Please update the map_gpu_to_arch function for this model."
        exit 1
        ;;
    esac
}

get_gpu_vram() {
    install_nvidia_smi_if_missing
    # Get VRAM in MB, convert to GB
    local vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo $((vram_mb / 1024))
}

map_gpu_to_profile() {
    local name="$1"
    local vram_gb="$2"

    # WanGP Profile descriptions from the actual UI:
    # Profile 1: HighRAM_HighVRAM - 48GB+ RAM, 24GB+ VRAM (fastest for short videos, RTX 3090/4090)
    # Profile 2: HighRAM_LowVRAM - 48GB+ RAM, 12GB+ VRAM (recommended, most versatile)
    # Profile 3: LowRAM_HighVRAM - 32GB+ RAM, 24GB+ VRAM (RTX 3090/4090 with limited RAM)
    # Profile 4: LowRAM_LowVRAM - 32GB+ RAM, 12GB+ VRAM (default, little VRAM or longer videos)
    # Profile 5: VerylowRAM_LowVRAM - 16GB+ RAM, 10GB+ VRAM (fail safe, slow but works)

    case "$name" in
    # High-end data center GPUs with 24GB+ VRAM - Profile 1 (HighRAM_HighVRAM)
    *"RTX 50"* | *"5090"* | *"A100"* | *"A800"* | *"H100"* | *"H800"*)
        if [ "$vram_gb" -ge 24 ]; then
            echo "1" # HighRAM_HighVRAM - fastest for short videos
        else
            echo "2" # HighRAM_LowVRAM - most versatile
        fi
        ;;
    # High-end consumer GPUs (RTX 3090/4090) - Profile 1 or 3
    *"RTX 40"* | *"4090"* | *"RTX 30"* | *"3090"*)
        if [ "$vram_gb" -ge 24 ]; then
            echo "3" # LowRAM_HighVRAM - good for limited RAM systems
        else
            echo "2" # HighRAM_LowVRAM - most versatile
        fi
        ;;
    # Mid-range GPUs (RTX 3070/3080/4070/4080) - Profile 2 recommended
    *"4080"* | *"4070"* | *"3080"* | *"3070"* | *"RTX 20"* | *"2080"* | *"2070"*)
        if [ "$vram_gb" -ge 12 ]; then
            echo "2" # HighRAM_LowVRAM - recommended for these GPUs
        else
            echo "4" # LowRAM_LowVRAM - default for little VRAM
        fi
        ;;
    # Lower-end GPUs with 6-12GB VRAM - Profile 4 or 5
    *"4060"* | *"3060"* | *"2060"* | *"GTX 16"* | *"1660"* | *"1650"*)
        if [ "$vram_gb" -ge 10 ]; then
            echo "4" # LowRAM_LowVRAM - default
        else
            echo "5" # VerylowRAM_LowVRAM - fail safe
        fi
        ;;
    # Older/lower VRAM GPUs - Profile 5 (fail safe)
    *"GTX 10"* | *"1080"* | *"1070"* | *"1060"* | *"Tesla"*)
        echo "5" # VerylowRAM_LowVRAM - fail safe
        ;;
    *)
        echo "4" # LowRAM_LowVRAM - default fallback
        ;;
    esac
}

# ───────────────────────── main ────────────────────────────

echo "🔧 NVIDIA CUDA Setup Check:"

# NVIDIA driver check
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    echo "✅ NVIDIA Driver: $DRIVER_VERSION"
    
    # Quick CUDA 12.4 compatibility check
    if [[ "$DRIVER_VERSION" =~ ^([0-9]+) ]]; then
        MAJOR=${BASH_REMATCH[1]}
        if [ "$MAJOR" -lt 520 ]; then
            echo "⚠️  Driver $DRIVER_VERSION may not support CUDA 12.4 (need 520+)"
        fi
    fi
else
    echo "❌ nvidia-smi not found - no NVIDIA drivers"
    exit 1
fi

GPU_NAME=$(detect_gpu_name)
echo "🔍 Detected GPU: $GPU_NAME"

VRAM_GB=$(get_gpu_vram)
echo "🧠 Detected VRAM: ${VRAM_GB}GB"

CUDA_ARCH=$(map_gpu_to_arch "$GPU_NAME")
echo "🚀 Using CUDA architecture: $CUDA_ARCH"

PROFILE=$(map_gpu_to_profile "$GPU_NAME" "$VRAM_GB")
echo "⚙️  Selected profile: $PROFILE"

docker build --build-arg CUDA_ARCHITECTURES="$CUDA_ARCH" -t deepbeepmeep/wan2gp .

# sudo helper for later commands
if [ "$EUID" -ne 0 ]; then
    SUDO='sudo'
else
    SUDO=''
fi

# Ensure NVIDIA runtime is available
if ! docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
    echo "⚠️  NVIDIA Docker runtime not found. Installing nvidia-docker2…"
    $SUDO apt-get update
    $SUDO apt-get install -y curl ca-certificates gnupg
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | $SUDO apt-key add -
    distribution=$(
        . /etc/os-release
        echo $ID$VERSION_ID
    )
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |
        $SUDO tee /etc/apt/sources.list.d/nvidia-docker.list
    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-docker2
    echo "🔄 Restarting Docker service…"
    $SUDO systemctl restart docker
    echo "✅ NVIDIA Docker runtime installed."
else
    echo "✅ NVIDIA Docker runtime found."
fi

# Quick NVIDIA runtime test
echo "🧪 Testing NVIDIA runtime..."
if timeout 15s docker run --rm --gpus all --runtime=nvidia nvidia/cuda:12.4-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA runtime working"
else
    echo "❌ NVIDIA runtime test failed - check driver/runtime compatibility"
fi

# Prepare cache dirs & volume mounts
cache_dirs=(numba matplotlib huggingface torch)
cache_mounts=()
for d in "${cache_dirs[@]}"; do
    mkdir -p "$HOME/.cache/$d"
    chmod 700 "$HOME/.cache/$d"
    cache_mounts+=(-v "$HOME/.cache/$d:/home/user/.cache/$d")
done

echo "🔧 Optimization settings:"
echo "   Profile: $PROFILE"

# Run the container
docker run --rm -it \
    --name wan2gp \
    --gpus all \
    --runtime=nvidia \
    -p 7860:7860 \
    -v "$(pwd):/workspace" \
    "${cache_mounts[@]}" \
    deepbeepmeep/wan2gp \
    --profile "$PROFILE" \
    --attention sage \
    --compile \
    --perc-reserved-mem-max 1
