#!/usr/bin/env bash
# ==============================================================================
# Wan2GP Cloud Launcher for Linux / macOS
# Zero local GPU requirements - runs in cloud client mode
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Optimize memory on mobile / multi-core CPU
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

if [ -f ".env" ]; then
    set -a
    source ".env"
    set +a
fi

echo "============================================================"
echo "          Starting Wan2GP in Cloud Mode (Modal GPU)         "
echo "============================================================"

# Choose best available Python interpreter
if [ -n "${WAN2GP_CLOUD_PYTHON:-}" ] && [ -f "$WAN2GP_CLOUD_PYTHON" ]; then
    PY_BIN="$WAN2GP_CLOUD_PYTHON"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$HOME/.wan2gp_venv/bin/python" ]; then
    PY_BIN="$HOME/.wan2gp_venv/bin/python"
elif [ -f "$HOME/.wan2gp-cloud-venv/bin/python" ]; then
    PY_BIN="$HOME/.wan2gp-cloud-venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PY_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PY_BIN="$CONDA_PREFIX/bin/python"
elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
else
    PY_BIN="python"
fi

# Check if sync requested or AUTO_SYNC configured
SHOULD_SYNC=false
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--sync" ] || [ "$arg" = "--update" ]; then
        SHOULD_SYNC=true
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

if [ "$SHOULD_SYNC" = "true" ] || [ "${AUTO_SYNC:-false}" = "true" ] || [ "${AUTO_SYNC:-0}" = "1" ]; then
    if [ -n "${MODAL_ENDPOINT:-}" ]; then
        echo "[*] Updating Modal cloud worker volume..."
        "$PY_BIN" -c "import os, urllib.request; req=urllib.request.Request(os.environ['MODAL_ENDPOINT'].rstrip('/')+'/admin/update-wan2gp', data=b'{}', headers={'X-API-Key': os.environ.get('MODAL_API_KEY','')}, method='POST'); print(urllib.request.urlopen(req, timeout=120).read().decode())" || echo "[!] Cloud sync finished with warnings; continuing..."
        echo ""
    fi
fi

# Check if provider is RunPod or Modal
PROV_CHECK="${WAN2GP_CLOUD_PROVIDER:-${WAN2GP_DEFAULT_PROVIDER:-modal}}"
if [[ "$*" =~ "runpod" ]] || [ "$PROV_CHECK" = "runpod" ]; then
    echo "[*] Cloud Provider: RunPod Serverless (Beta Preview)"
    if [ ! -f ".env" ] || ! grep -q "RUNPOD_API_KEY" ".env"; then
        echo "[!] Note: RUNPOD_API_KEY is not configured in .env yet."
        echo "[*] See docs/RUNPOD_SETUP.md or configure .env to connect."
    fi
elif [ ! -f ".env" ] || ! grep -q "MODAL_ENDPOINT" ".env"; then
    echo ""
    echo "[!] Note: MODAL_ENDPOINT is not configured in .env yet."
    echo "[*] Launching Wan2GP in Cloud Client Mode (Modal). You can run 'cloud/modal/setup_modal.py' anytime to deploy."
    echo ""
fi

# Launch Wan2GP in Cloud Client Mode
echo "[*] Launching Wan2GP Web UI..."
BROWSER_FLAG="--open-browser"
if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    BROWSER_FLAG="--listen"
fi

PROVIDER_FLAG=""
if [[ ! "$*" =~ "--provider" ]]; then
    PROVIDER_FLAG="--provider $PROV_CHECK"
fi

"$PY_BIN" wgp.py --cloud $PROVIDER_FLAG $BROWSER_FLAG "${PASSTHROUGH_ARGS[@]}"
