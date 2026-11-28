#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment configuration if available
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

echo "===================================================================="
echo " 🌟 Launching Wan2GP Gradio on Modal GPU with Live Terminal Logs"
echo "===================================================================="
echo " • App Name: ${MODAL_APP_NAME:-wan2gp-cloud}"
echo " • Default GPU: ${MODAL_DEFAULT_GPU:-L40S}"
echo " • Connected Volumes: wan2gp-models, wan2gp-outputs, wan2gp-cache"
echo "===================================================================="
echo ""
echo "Streaming logs directly below (press Ctrl+C to stop container):"
echo ""

modal serve cloud/modal/modal_interactive_ui.py
