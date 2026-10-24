#!/usr/bin/env bash
# WanGP DLSS 5 installer (Linux/portable port of install_dlss5.ps1).
#
# Runs scripts/install_dlss5.py with the WanGP venv Python when present,
# falling back to the active python3. All arguments are passed through,
# e.g. --force, --wan-gp-root. (Linux uses official NVIDIA sources only,
# so no third-party consent flag is needed, unlike the Windows installer.)
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python || true)"
    if [ -z "$PYTHON" ]; then
        echo "[-] No Python interpreter found. Create the WanGP venv (scripts/install.sh) or install Python 3.8+."
        exit 1
    fi
fi

"$PYTHON" scripts/install_dlss5.py "$@"