# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Rebuild the PTX images of the Sol attention kernels (ptx/) from sol_attn.cu and sol_pack.cu with PyTorch's NVRTC.

python build.py
The images are compiled by the driver when they are loaded: sm_89 images run on newer GPUs too, sm_120a images need a
GeForce RTX 50xx."""
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# image: (source, architecture, defines)
IMAGES = {
    "attn_i8_sm89": ("sol_attn.cu", "89", ("KERNEL_NAME=sol_attn_i8", "QK_FP4=0", "BULK=0")),
    "attn_i8_sm120a": ("sol_attn.cu", "120a", ("KERNEL_NAME=sol_attn_i8", "QK_FP4=0", "BULK=1")),
    "attn_fp4_sm120a": ("sol_attn.cu", "120a", ("KERNEL_NAME=sol_attn_fp4", "QK_FP4=1", "BULK=1")),
    "pack_sm89": ("sol_pack.cu", "89", ("FP4=0",)),
    "pack_fp4_sm120a": ("sol_pack.cu", "120a", ("FP4=1",)),
}
ENTRY = {"attn_i8_sm89": "sol_attn_i8", "attn_i8_sm120a": "sol_attn_i8", "attn_fp4_sm120a": "sol_attn_fp4", "pack_sm89": "sol_pack_v",
         "pack_fp4_sm120a": "sol_pack_k4"}


def main():
    for image, (source, arch, defines) in IMAGES.items():
        ptx, _ = torch.cuda._utils._nvrtc_compile((HERE / source).read_text(encoding="utf-8"), ENTRY[image], compute_capability=arch,
                                                 nvcc_options=[f"-D{d}" for d in defines])
        (HERE / "ptx" / f"{image}.ptx").write_bytes(ptx)
        print(image, len(ptx), "bytes")


if __name__ == "__main__":
    main()
