"""Reproduce the BailingMM2 ConvRot file from its prepared BF16 checkpoint.

The initial release was generated through a temporary WanGP load-only hook;
that hook was removed after verification.  This script keeps the conversion
repeatable without adding conversion work to normal model loading.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from shared.convert.convrot import save_convrot_model

from .ming_pipeline import _make_text_encoder
from .upstream.modeling_bailing_moe_v2 import BailingMoeV2Gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    version = parser.add_mutually_exclusive_group()
    version.add_argument("--design-02", action="store_true", help="convert only the release-specific 0.2 conditioner")
    version.add_argument("--layer", action="store_true", help="convert the Design-Layer encoder")
    args = parser.parse_args()
    root = args.checkpoint_root.resolve()
    name = "BailingMM2-Ming-Image-Layer" if args.layer else "BailingMM2-Ming-Image"
    folder = root / name
    project = root / ("ming_image_layer" if args.layer else "ming_image")
    source = folder / f"{name}_bf16.safetensors"
    target = folder / f"{name}_int8_convrot.safetensors"
    connector = json.loads((project / "connector_config.json").read_text(encoding="utf-8"))
    mlp = json.loads((project / "mlp_config.json").read_text(encoding="utf-8"))
    encoder, vision = _make_text_encoder(str(folder / "config.json"), connector, mlp)
    if args.design_02:
        encoder.connector.model.embed_tokens = None
        save_convrot_model(encoder, project / "conditioning_0_2_bf16.safetensors",
                           project / "conditioning_0_2_int8_convrot.safetensors",
                           prefixes=("connector.model.layers.", "proj_in", "proj_out", "proj_directvlm.1"))
        return
    # The legacy source files include the vision tower in the full Bailing checkpoint.
    encoder.vision, encoder.linear_proj = vision.vision, vision.linear_proj
    for module in encoder.modules():
        if isinstance(module, BailingMoeV2Gate):
            # These are raw routing parameters, not nn.Linear modules.
            module._lock_dtype = torch.bfloat16
    save_convrot_model(encoder, str(source), str(target))
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if handle.metadata().get("quantization_format") != "int8_convrot":
            raise ValueError("missing ConvRot metadata")
        if handle.get_slice("model.model.layers.1.mlp.gate.weight").get_dtype() != "BF16":
            raise ValueError("Bailing router gate was quantized")
        count = sum(key.endswith(".weight") and handle.get_slice(key).get_dtype() == "I8" for key in keys)
        if count < 1000:
            raise ValueError(f"only {count} weights were quantized")
    print(f"Verified {target}: {count} ConvRot weights")


if __name__ == "__main__":
    main()
