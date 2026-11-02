#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

if [ -f "../.manager/installation.json" ] || [ -f "../.manager/active_operation.json" ]; then
    exec python3 -m installation launch --installation .. -- "$@"
fi

if [ ! -f "envs.json" ]; then
    echo "[!] envs.json not found. Please run install.sh first or add your existing environment with manage.sh if you already have one."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "[*] Fetching active environment..."

ENV_OUTPUT=$(python3 setup.py get_env_info 2>/dev/null | grep "^ENV_INFO|")

if [ -z "$ENV_OUTPUT" ]; then
    echo "[!] No active environment found."
    echo "Please run install.sh first or add your existing environment with manage.sh if you already have one."
    read -p "Press Enter to exit..."
    exit 1
fi

ENV_TYPE=$(echo "$ENV_OUTPUT" | cut -d'|' -f2)
ENV_PATH=$(echo "$ENV_OUTPUT" | cut -d'|' -f3)

if [ "$ENV_TYPE" = "venv" ] || [ "$ENV_TYPE" = "uv" ]; then
    echo "[*] Activating $ENV_TYPE: $ENV_PATH"
    source "$ENV_PATH/bin/activate"

elif [ "$ENV_TYPE" = "conda" ]; then
    echo "[*] Activating conda: $ENV_PATH"

    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        for base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
            if [ -f "$base/etc/profile.d/conda.sh" ]; then
                source "$base/etc/profile.d/conda.sh"
                break
            fi
        done
    fi
    
    if ! command -v conda >/dev/null 2>&1; then
        echo "[!] Could not find conda. Please ensure Conda is installed."
        read -p "Press Enter to exit..."
        exit 1
    fi
    conda activate "$ENV_PATH"

elif [ "$ENV_TYPE" = "none" ]; then
    echo "[*] Using system Python (No virtual environment)"
else
    echo "[!] Unknown environment type: $ENV_TYPE"
    read -p "Press Enter to exit..."
    exit 1
fi

EXTRA_ARGS=""
if [ -f "scripts/args.txt" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ "$line" =~ ^[[:space:]]*[^#[:space:]] ]]; then
            EXTRA_ARGS="$EXTRA_ARGS $line"
        fi
    done < "scripts/args.txt"
fi

if [ "$ENV_TYPE" = "none" ]; then
    PY_CMD="python3"
else
    PY_CMD="python"
fi

while true; do
    echo "[*] Launching WAN2GP..."
    eval "$PY_CMD wgp.py $EXTRA_ARGS"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 42 ]; then
        echo ""
        echo "[*] Restarting..."
    else
        break
    fi
done

read -p "Press Enter to exit..."
