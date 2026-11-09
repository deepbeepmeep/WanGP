#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────── Package Management Logic ─────────────────────────

# Detect package manager intelligently based on command availability
detect_package_manager() {
    if command -v apt-get &>/dev/null; then
        echo "apt"
    elif command -v dnf &>/dev/null; then
        echo "dnf"
    elif command -v pacman &>/dev/null; then
        echo "pacman"
    elif command -v zypper &>/dev/null; then
        echo "zypper"
    else
        echo "unknown"
    fi
}

PKG_MANAGER=$(detect_package_manager)

if [ "$PKG_MANAGER" = "unknown" ]; then
    echo "❌ Error: No supported package manager (apt, dnf, pacman, zypper) detected."
    exit 1
fi

# Helper to run sudo if not root
if [ "$EUID" -ne 0 ]; then
    SUDO='sudo'
else
    SUDO=''
fi

pkg_update() {
    echo "🔄 Updating package databases using $PKG_MANAGER..."
    case "$PKG_MANAGER" in
        apt)    $SUDO apt-get update -y ;;
        dnf)    $SUDO dnf check-update || true ;; # dnf check-update returns non-zero if updates exist
        pacman) $SUDO pacman -Sy ;;
        zypper) $SUDO zypper refresh ;;
    esac
}

pkg_install() {
    local pkg="$1"
    echo "📦 Installing $pkg using $PKG_MANAGER..."
    case "$PKG_MANAGER" in
        apt)    $SUDO apt-get install -y "$pkg" ;;
        dnf)    $SUDO dnf install -y "$pkg" ;;
        pacman) $SUDO pacman -S --noconfirm "$pkg" ;;
        zypper) $SUDO zypper install -y "$pkg" ;;
    esac
}

# ───────────────────────── helpers ─────────────────────────

install_nvidia_smi_if_missing() {
    if command -v nvidia-smi &>/dev/null; then
        return
    fi

    echo "⚠️ nvidia-smi not found. Installing nvidia-utils..."

    # Package naming varies by distro
    case "$PKG_MANAGER" in
        apt)    pkg_install "nvidia-utils-535" || pkg_install "nvidia-utils" ;;
        dnf)    pkg_install "nvidia-driver-latest" ;; # Common name for RHEL/Fedora setups
        pacman) pkg_install "nvidia-utils" ;;
        zypper) pkg_install "nvidia-compute-utils" ;;
    esac

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
    *"RTX 20"* | *"2080"* | *"2070"* | *"2060"* | *"Titan RTX"*) echo "7.5" ;;
    *"GTX 16"* | *"1660"* | *"1650"*) echo "7.5" ;;
    *"Tesla V100"*) echo "7.0" ;;
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
    local vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo $((vram_mb / 1024))
}

map_gpu_to_profile() {
    local name="$1"
    local vram_gb="$2"

    case "$name" in
    *"RTX 50"* | *"5090"* | *"A100"* | *"A800"* | *"H100"* | *"H800"*)
        if [ "$vram_gb" -ge 24 ]; then echo "1"; else echo "2"; fi ;;
    *"RTX 40"* | *"4090"* | *"RTX 30"* | *"3090"*)
        if [ "$vram_gb" -ge 24 ]; then echo "3"; else echo "2"; fi ;;
    *"4080"* | *"4070"* | *"3080"* | *"3070"* | *"RTX 20"* | *"2080"* | *"2070"*)
        if [ "$vram_gb" -ge 12 ]; then echo "2"; else echo "4"; fi ;;
    *"4060"* | *"3060"* | *"2060"* | *"GTX 16"* | *"1660"* | *"1650"*)
        if [ "$vram_gb" -ge 10 ]; then echo "4"; else echo "5"; fi ;;
    *"GTX 10"* | *"1080"* | *"1070"* | *"1060"* | *"Tesla"*)
        echo "5" ;;
    *)
        echo "4" ;;
    esac
}

# ───────────────────────── main ────────────────────────────

echo "🔧 NVIDIA CUDA Setup Check (Detected Manager: $PKG_MANAGER):"

if command -v nvidia-smi &>/dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    echo "✅ NVIDIA Driver: $DRIVER_VERSION"

    if [[ "$DRIVER_VERSION" =~ ^([0-9]+) ]]; then
        MAJOR=${BASH_REMATCH[1]}
        if [ "$MAJOR" -lt 520 ]; then
            echo "⚠️ Driver $DRIVER_VERSION may not support CUDA 12.4 (need 520+)"
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

# Check if the image already exists locally
if ! docker image inspect deepbeepmeep/wan2gp > /dev/null 2>&1; then
    echo "🏗️  Image not found. Building 'deepbeepmeep/wan2gp'..."
    docker build --build-arg CUDA_ARCHITECTURES="$CUDA_ARCH" -t deepbeepmeep/wan2gp .
else
    echo "✅ Image 'deepbeepmeep/wan2gp' already exists locally. Skipping build."
fi

# Ensure NVIDIA runtime is available
if ! docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
    echo "⚠️  NVIDIA Docker runtime not found. Installing..."

    case "$PKG_MANAGER" in
        apt)
            pkg_update
            pkg_install "curl ca-certificates gnupg"
            # Note: The original repo logic for Debian is left here as it's specific to apt
            distribution=$(. /etc/os-release && echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | $SUDO tee /etc/apt/sources.list.d/nvidia-docker.list
            pkg_update
            pkg_install "nvidia-docker2"
            ;;
        dnf)
            # RHEL/CentOS/Fedora approach (Assumes NVIDIA repos are configured or uses standard toolkit name)
            pkg_install "nvidia-container-toolkit"
            ;;
        pacman)
            # Arch Linux approach
            pkg_install "nvidia-container-toolkit"
            ;;
        zypper)
            # SUSE approach
            pkg_install "nvidia-container-toolkit"
            ;;
    esac

    if [ "$PKG_MANAGER" != "apt" ]; then
        # Configure Docker to use the NVIDIA runtime
        $SUDO nvidia-ctk runtime configure --runtime=docker
        echo "🔄 Restarting Docker service..."
        $SUDO systemctl restart docker
    fi
    echo "✅ NVIDIA Docker runtime setup attempted."
else
    echo "✅ NVIDIA Docker runtime found."
fi

# Quick NVIDIA runtime test
echo "🧪 Testing NVIDIA runtime..."
if timeout 15s docker run --rm --gpus all --runtime=nvidia nvidia/cuda:12.8.1-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA runtime working"
else
    echo "❌ NVIDIA runtime test failed - check driver/runtime compatibility"
fi

# Prepare cache dirs & volume mounts
cache_dirs=(numba matplotlib huggingface torch)
cache_mounts=()
for d in "${cache_dirs[@]}"; do
    mkdir -p "/tmp/$d"
    chmod 700 "/tmp/$d"
    cache_mounts+=(-v "/tmp/$d:/home/user/.cache/$d")
done

# ───────────────────────── Connectivity Check ─────────────────────────

HF_ENV_FLAG=""
echo "🌐 Checking HuggingFace connectivity..."
if curl -Is https://huggingface.co --connect-timeout 3 > /dev/null; then
    echo "✅ HuggingFace is reachable. Using default endpoint."
else
    echo "⚠️ HuggingFace unreachable. Trying HF-Mirror..."
    if curl -Is https://hf-mirror.com --connect-timeout 3 > /dev/null; then
        echo "✅ HF-Mirror is reachable. Switching to HF-Mirror."
        HF_ENV_FLAG="-e HF_ENDPOINT=https://hf-mirror.com"
    else
        echo "❌ Both HuggingFace and HF-Mirror are unreachable. Maybe offline?"
        # Switch to HF-Mirror anyways just in case
        HF_ENV_FLAG="-e HF_ENDPOINT=https://hf-mirror.com"
    fi
fi

# ───────────────────────── Run Container ──────────────────────────────

echo "🔧 Optimization settings:"
echo "   Profile: $PROFILE"
echo "   CUDA Arch: $CUDA_ARCH"

ATTENTION_FLAG="--attention sdpa"
COMPILE_FLAG=""
if [[ "$CUDA_ARCH" == "7.0" || "$CUDA_ARCH" == "6.1" || "$CUDA_ARCH" == "3.7" ]]; then
    echo "ℹ️ Using sdpa attention instead of sage and skipping --compile flag (not recommended for your CUDA Arch)"
else
    ATTENTION_FLAG="--attention sage"
    COMPILE_FLAG="--compile"
fi

# Run the container
docker run --rm -it \
    --name wan2gp \
    --gpus all \
    --runtime=nvidia \
    -p 7860:7860 \
    -v "$(pwd):/workspace" \
    "${cache_mounts[@]}" \
    "$HF_ENV_FLAG" \
    deepbeepmeep/wan2gp \
    --profile "$PROFILE" \
    $ATTENTION_FLAG \
    $COMPILE_FLAG \
    --perc-reserved-mem-max 1
