# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any


def get_system_handler(name: str | None) -> Any:
    name = str(name or "").strip().lower()
    if name == "flashvsr":
        from postprocessing.flashvsr.process_handler import HANDLER
        return HANDLER
    if name == "seedvr2":
        from postprocessing.seedvr2.process_handler import HANDLER
        return HANDLER
    if name == "pid":
        from postprocessing.pid.process_handler import HANDLER
        return HANDLER
    if name == "coz":
        from postprocessing.chain_of_zoom.process_handler import HANDLER
        return HANDLER
    if name == "ltx2_upsampler":
        from postprocessing.ltx2_upsampler.process_handler import HANDLER
        return HANDLER
    if name == "ltx2_refiner":
        from postprocessing.ltx2_upsampler.process_handler import REFINER_HANDLER
        return REFINER_HANDLER
    if name in ("rife", "dlssg", "dlss5"):
        from .system_upsampler_handler import DLSS5_HANDLER, DLSSG_HANDLER, RIFE_HANDLER
        return {"rife": RIFE_HANDLER, "dlssg": DLSSG_HANDLER, "dlss5": DLSS5_HANDLER}[name]
    if name == "latent_decode":
        from .latent_process_handler import LATENT_DECODE_HANDLER
        return LATENT_DECODE_HANDLER
    return None
