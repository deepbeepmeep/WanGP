# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Runtime patches for Gradio server startup."""

from __future__ import annotations

import time
import warnings
from urllib.parse import urlsplit

import httpx


def _url_ok_with_retries(url: str) -> bool:
    parts = urlsplit(url)
    loopback = parts.scheme in ("http", "https") and parts.hostname in ("127.0.0.1", "localhost", "::1")
    # Gradio builds the full first page even for HEAD. Give installed plugins
    # time to render it without queuing more copies of that work.
    options = {"timeout": httpx.Timeout(3, read=60), "trust_env": False} if loopback else {"timeout": 3}
    for attempt in range(5):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                response = httpx.head(url, verify=False, **options)
            if response.status_code in (200, 401, 302, 303, 307):
                return True
            failure = f"HTTP {response.status_code}"
        except (ConnectionError, httpx.ConnectError, httpx.TimeoutException) as exc:
            failure = type(exc).__name__
            if loopback and isinstance(exc, httpx.ReadTimeout):
                failure = "no response headers within 60 seconds"
                break
        if attempt < 4:
            time.sleep(0.5)
    if loopback:
        from tqdm import tqdm
        tqdm.write(f"Gradio startup check failed for local port {parts.port}: {failure}")
    return False


def install() -> bool:
    import gradio.networking as gradio_networking

    if getattr(gradio_networking.url_ok, "_wangp_startup_patch_installed", False):
        return True
    _url_ok_with_retries._wangp_startup_patch_installed = True
    _url_ok_with_retries._wangp_original_url_ok = gradio_networking.url_ok
    gradio_networking.url_ok = _url_ok_with_retries
    return True


__all__ = ["install"]


if __name__ == "__main__":
    # Manager uses this entry point only for releases with the original 3 s
    # probe. The checkout's own wgp.py still controls when Gradio is patched.
    import runpy
    import sys
    from pathlib import Path
    sys.argv = sys.argv[1:]
    sys.path[0] = str(Path(sys.argv[0]).parent)
    sys.modules["shared.gradio.gradio_startup_patch"] = sys.modules[__name__]
    runpy.run_path(sys.argv[0], run_name="__main__")
