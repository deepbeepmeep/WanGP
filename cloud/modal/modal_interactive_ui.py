# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Standalone Cloud GPU Interactive Runner for Wan2GP (Original upstream repo on Modal).
Runs Wan2GP Gradio completely in cloud GPU with live streaming terminal logs.
This is a dedicated, separate utility solely for manual cloud interactive use.
"""
import os
import sys
from pathlib import Path
import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "wan2gp-cloud")
MODELS_VOL_NAME = os.environ.get("MODAL_MODELS_VOLUME_NAME", "wan2gp-models")
OUTPUTS_VOL_NAME = os.environ.get("MODAL_VOLUME_NAME", "wan2gp-outputs")
CACHE_VOL_NAME = "wan2gp-cache"

app = modal.App(f"{APP_NAME}-interactive-ui")

models_volume = modal.Volume.from_name(MODELS_VOL_NAME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUTS_VOL_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOL_NAME, create_if_missing=True)

MODELS_MOUNT = "/models"
OUTPUTS_MOUNT = "/outputs"
CACHE_MOUNT = "/cache"

# Reuse the base CUDA image with all drivers and python libraries
wan2gp_interactive_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "build-essential")
    .pip_install("uv", "gradio==5.29.0", "fastapi>=0.115.0", "uvicorn>=0.30.0")
)

@app.function(
    image=wan2gp_interactive_image,
    gpu=os.environ.get("MODAL_DEFAULT_GPU", "L40S"),
    volumes={
        MODELS_MOUNT: models_volume,
        OUTPUTS_MOUNT: outputs_volume,
        CACHE_MOUNT: cache_volume
    },
    timeout=86400,
    scaledown_window=300
)
@modal.wsgi_app()
def ui():
    wan2gp_dir = Path(MODELS_MOUNT) / "Wan2GP"
    if not (wan2gp_dir / "wgp.py").exists():
        raise RuntimeError("Wan2GP repo not found on volume. Run 'python cloud/modal/setup_modal.py' or trigger /admin/update-wan2gp first.")

    venv_site_packages = list((Path(MODELS_MOUNT) / "venv" / "lib").glob("python*/site-packages"))
    sys.path.insert(0, str(wan2gp_dir))
    for sp in venv_site_packages:
        if str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

    os.chdir(str(wan2gp_dir))
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    import wgp
    print("\n" + "=" * 60)
    print("🚀 Initializing Wan2GP (Original Upstream) on Modal Cloud GPU...")
    print(f" • Repository: {wan2gp_dir}")
    print(f" • Attached Volumes: {MODELS_MOUNT}, {OUTPUTS_MOUNT}")
    print("=" * 60 + "\n")

    demo = wgp.create_ui()
    wgp.clear_startup_lock()
    return demo.app
