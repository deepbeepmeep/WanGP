# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prepare optional INT8 ConvRot weights for the Swin background-remover backbone."""
def save_quantized(source_file, output_file):
    from accelerate import init_empty_weights
    from shared.convert.convrot import save_convrot_model
    from .birefnet import BiRefNet
    with init_empty_weights(include_buffers=True):
        model = BiRefNet()
    save_convrot_model(model, source_file, output_file, prefixes=('bb.layers.',))
