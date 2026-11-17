# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
from pathlib import Path

import torch
from mmgp import offload

CONFIG = json.loads(Path(__file__).with_name("zonos3_mini.json").read_text(encoding="utf-8"))
# The released attention stores its key and value projections as one k|v matrix.
LM_SPLIT_MAP = {"attention.wkv": {"mapped_modules": ["attention.wk", "attention.wv"], "split_sizes": [128, 128]}}


def load_lm(path, features):
    from .lm import Zonos3LM
    with torch.device("meta"):
        model = Zonos3LM(CONFIG["lm"], features)
    offload.load_model_data(model, str(path), writable_tensors=False, default_dtype=None, fused_split_map=LM_SPLIT_MAP)
    return model.eval()


def load_decoder(path):
    from .codec import ZAC1Decoder
    with torch.device("meta"):
        model = ZAC1Decoder(CONFIG["codec"])
    offload.load_model_data(model, str(path), writable_tensors=False, default_dtype=None)
    # Float32 except the BF16 matrices of the transformer stages, locked so that MMGP's profiling neither recasts nor rejects them.
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.dtype == torch.bfloat16:
            module._lock_dtype = torch.bfloat16
    model._convertWeightsFloatTo = None
    model._model_dtype = torch.float32
    return model.eval()


def load_speaker_encoder(path):
    from .speaker import SpeakerEncoder
    with torch.device("meta"):
        model = SpeakerEncoder(CONFIG["speaker"])
    offload.load_model_data(model, str(path), writable_tensors=False, default_dtype=None)
    model._convertWeightsFloatTo = None
    model._model_dtype = torch.float32
    return model.eval()
