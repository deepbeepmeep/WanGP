#!/usr/bin/env python3
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Wan2GP One-Click Cloud Modal Setup Wizard.
Designed for non-technical users:
  1. Installs modal if missing
  2. Opens the browser for 1-click Modal login (no copy-pasting API tokens needed)
  3. Deploys your personal serverless GPU worker (scale to zero when idle)
  4. Automatically writes configuration to Wan2GP .env
"""

import os
import sys
import json
import shutil
import subprocess
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

def print_banner():
    print("=" * 64)
    print("   🚀 Wan2GP Modal Cloud GPU Setup (1-Click Automated)   ")
    print("=" * 64)
    print("This wizard connects Wan2GP to Modal Cloud so you can generate")
    print("high-end AI videos using cloud GPUs (L40S / A100 / H100)")
    print("without needing an expensive graphics card on your PC!\n")

def print_step(num: int, title: str):
    print(f"\n[\033[1;36mStep {num}\033[0m] \033[1m{title}\033[0m")

def print_ok(msg: str):
    print(f"\033[1;32m✓\033[0m {msg}")

def print_warn(msg: str):
    print(f"\033[1;33m!\033[0m {msg}")

def print_err(msg: str):
    print(f"\033[1;31m✗\033[0m {msg}")

def find_modal_bin():
    # 1. System PATH
    p = shutil.which("modal")
    if p:
        return p
    # 2. ~/.local/bin
    local_bin = Path.home() / ".local" / "bin" / ("modal.exe" if os.name == "nt" else "modal")
    if local_bin.exists():
        return str(local_bin)
    # 3. Python environment scripts directory
    py_dir = Path(sys.executable).parent
    p2 = py_dir / ("modal.exe" if os.name == "nt" else "modal")
    if p2.exists():
        return str(p2)
    return None

def main():
    print_banner()

    modal_dir = Path(__file__).resolve().parent
    wan2gp_root = modal_dir.parent.parent
    modal_app_py = modal_dir / "modal_app.py"

    if not modal_app_py.exists():
        print_err(f"modal_app.py not found in {modal_dir}!")
        sys.exit(1)

    # 1. Check or install modal CLI
    print_step(1, "Checking Modal Cloud Tools...")
    modal_bin = find_modal_bin()
    if not modal_bin:
        print_warn("Modal CLI is not installed yet. Installing now via pip...")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "modal"]
            if os.name != "nt":
                cmd.append("--break-system-packages")
            subprocess.check_call(cmd)
            modal_bin = find_modal_bin()
        except Exception:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "modal"])
                modal_bin = find_modal_bin()
            except Exception as e:
                print_err(f"Could not auto-install modal: {e}")
                print("Please run manually: pip install modal")
                sys.exit(1)

    print_ok(f"Modal CLI ready: {modal_bin}")

    # 2. Modal account authentication
    print_step(2, "Connecting your Modal account...")
    modal_config = Path.home() / ".modal.toml"
    has_env_token = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))

    if not modal_config.exists() and not has_env_token:
        print("No login found. Opening your browser to sign in...")
        print("👉 Please log in to modal.com and click 'Approve' in your browser.\n")
        try:
            res = subprocess.run([modal_bin, "setup"])
            if res.returncode != 0:
                print_err("Modal login was cancelled. Please rerun this script to try again.")
                sys.exit(1)
        except Exception as e:
            print_err(f"Failed to launch modal setup: {e}")
            sys.exit(1)

    print_ok("Modal account successfully connected!")

    # 3. Configure app name and secret key
    print_step(3, "Configuring your personal cloud worker...")
    import secrets
    default_app = "wan2gp-cloud"
    default_key = f"wgp-{secrets.token_hex(12)}"

    # Check existing Wan2GP .env if already present
    env_path = wan2gp_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MODAL_APP_NAME="):
                default_app = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("MODAL_API_KEY="):
                default_key = line.split("=", 1)[1].strip().strip('"').strip("'")

    print_ok(f"App Name: {default_app}")
    print_ok(f"Generated secure API key: {default_key[:8]}...{default_key[-4:]}")

    # 4. Deploy serverless app to Modal
    print_step(4, "Deploying your cloud GPU worker to Modal...")
    print("Modal will build the environment, prepare fast cloud volumes, and launch your private worker.")
    print("⏳ First-time deployment takes about 2-4 minutes. Please wait...\n")

    deploy_env = os.environ.copy()
    deploy_env["MODAL_API_KEY"] = default_key
    deploy_env["MODAL_APP_NAME"] = default_app

    proc = subprocess.Popen(
        [modal_bin, "deploy", str(modal_app_py)],
        cwd=str(modal_dir),
        env=deploy_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    endpoint_url = None
    url_regex = re.compile(r"https://[a-zA-Z0-9_-]+--" + re.escape(default_app) + r"-fastapi-app\.modal\.run")
    generic_regex = re.compile(r"https://[a-zA-Z0-9_-]+--[a-zA-Z0-9_-]+-fastapi-app\.modal\.run")

    for line in proc.stdout:
        print(line, end="")
        match = url_regex.search(line) or generic_regex.search(line)
        if match:
            endpoint_url = match.group(0)

    proc.wait()

    if proc.returncode != 0:
        print_err("Deployment encountered an error. Please check the logs above.")
        sys.exit(1)

    print_ok("Serverless deployment succeeded!")
    if endpoint_url:
        print_ok(f"Cloud Worker URL: {endpoint_url}")

    # 5. Verify live worker
    if endpoint_url:
        print_step(5, "Verifying cloud worker health...")
        time.sleep(2)
        try:
            req = urllib.request.Request(endpoint_url + "/", headers={"User-Agent": "Wan2GP-Setup/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            print_ok(f"Online! Service: {data.get('service')} | Default GPU: {data.get('default_gpu')}")
            print_ok(f"Supported GPUs: {', '.join(data.get('gpus', []))}")
        except Exception as e:
            print_warn(f"Note: Worker may still be warming up ({e})")

    # 6. Save .env in Wan2GP
    print_step(6, "Saving settings into Wan2GP configuration...")
    if endpoint_url:
        existing_lines = []
        if env_path.exists():
            existing_lines = [
                line for line in env_path.read_text(encoding="utf-8").splitlines()
                if not any(line.startswith(prefix) for prefix in [
                    "WAN2GP_DEFAULT_PROVIDER=", "MODAL_ENDPOINT=",
                    "MODAL_API_KEY=", "MODAL_APP_NAME=", "MODAL_DEFAULT_GPU="
                ])
            ]

        new_settings = [
            'WAN2GP_DEFAULT_PROVIDER="modal"',
            f'MODAL_ENDPOINT="{endpoint_url}"',
            f'MODAL_API_KEY="{default_key}"',
            f'MODAL_APP_NAME="{default_app}"',
            'MODAL_DEFAULT_GPU="L40S"',
        ]

        env_path.write_text("\n".join(existing_lines + new_settings) + "\n", encoding="utf-8")
        print_ok(f"Settings saved to: {env_path}")

    # 7. Initial Cloud Worker Volume Initialization
    print_step(7, "Initializing Modal cloud worker volume...")
    print("Syncing WanGP source and volume venv on Modal...")
    try:
        import urllib.request
        headers = {"Content-Type": "application/json", "X-API-Key": default_key, "Authorization": f"Bearer {default_key}"}
        req = urllib.request.Request(f"{endpoint_url}/admin/update-wan2gp", data=b"{}", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            print_ok(f"Modal cloud worker synced: {resp.read().decode('utf-8', errors='replace')[:200]}")
    except Exception as sync_err:
        print_warn(f"Initial volume sync completed with notice: {sync_err}")
        print_warn(f"You can trigger sync anytime via POST {endpoint_url}/admin/update-wan2gp")

    print("\n" + "=" * 64)
    print(" 🎉 ALL DONE! You are ready to start generating!")
    print("=" * 64)
    print("\nTo start Wan2GP in cloud mode, run:")
    if os.name == "nt":
        print("    run_cloud.bat")
    else:
        print("    ./run_cloud.sh")
    print("or:")
    print("    python wgp.py --cloud\n")

if __name__ == "__main__":
    main()
