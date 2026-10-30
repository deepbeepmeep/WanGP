"""Tiny preview decoders adapted from GOvEy1nw's Wan2GP PR #2107.

See LICENSES and sources.json for the upstream implementations and weights.
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

REGISTRY = json.loads(Path(__file__).with_name('decoders.json').read_text(encoding='utf-8'))


def decoder_for(architecture, model_def):
    # Handlers declare `tiny_vae_architecture`: the registered architecture whose latent space and preview contract they share.
    return REGISTRY['architectures'].get(model_def.get('tiny_vae_architecture', architecture))


def prepare_decoder(name, gen=None):
    from shared.utils import files_locator as fl
    from shared.utils.download import download_file, check_download_cancelled

    config = REGISTRY['decoders'][name]
    relative = 'preview_decoders/' + config['filename']
    path = fl.locate_file(relative, error_if_none=False)
    if path is None:
        path = fl.get_smart_download_location(config['filename'], 'preview_decoders')
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        download_file(config.get('url', f"https://huggingface.co/{config['repo']}/resolve/main/{relative}"), path, gen=gen)
    check_download_cancelled(gen)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != config['sha256']:
        raise ValueError(f'TinyVAE checkpoint checksum mismatch: {path}')
    return path


def frame_indices(count):
    return list(dict.fromkeys(i * count // min(4, count) for i in range(min(4, count))))


class TinyVAE(nn.Module):
    def __init__(self, name, net):
        super().__init__()
        self.name = name
        self.net = net
        self.config = REGISTRY['decoders'][name]
        self._convertWeightsFloatTo = None

    def forward(self, latents, image=False, abort_check=None, video=False, duration=None):
        # Input is WanGP's existing C,T,H,W preview contract. Image batches
        # occupy T; temporal video decoders must not mix those images.
        value = latents.permute(1, 0, 2, 3).to(next(self.net.parameters()))
        video = video and not image
        count = (len(value) - 1) * self.net.t_upscale + 1 if self.config['kind'] == 'video' and not image else len(value)
        if video:
            samples = min(count, 240, max(1, math.ceil(duration * 12)))
            indices = [round(i * (count - 1) / max(1, samples - 1)) for i in range(samples)]
        else:
            indices = frame_indices(count)

        def to_pixels(frame):
            height, width = frame.shape[-2:]
            if video:
                scale = min(200 / height, 384 / width)
                size = (max(2, round(height * scale / 2) * 2), max(2, round(width * scale / 2) * 2))
            else:
                size = (200, max(1, round(width * 200 / height)))
            frame = F.interpolate(frame, size=size, mode='bilinear', align_corners=False)
            return frame.clamp(0, 1).mul_(255).round_().to(torch.uint8).cpu()

        if self.config['kind'] != 'video':
            frames = []
            for index in indices:
                if abort_check is not None and abort_check():
                    return None
                frames.append(to_pixels(self.net(value[index:index + 1])))
            pixels = torch.cat(frames)
        elif image:
            frames = []
            for index in indices:
                decoded = self.net.decode_video(value[index:index + 1, None], parallel=False, abort_check=abort_check)
                if decoded is None:
                    return None
                frames.append(to_pixels(decoded[:, 0]))
            pixels = torch.cat(frames)
        else:
            decoded = self.net.decode_video(value[None], parallel=False, output_indices=indices, abort_check=abort_check, output_transform=to_pixels)
            if decoded is None:
                return None
            pixels = decoded[0]
        if abort_check is not None and abort_check():
            return None
        if video:
            return [Image.fromarray(frame.permute(1, 2, 0).numpy()) for frame in pixels], len(indices) / duration
        pixels = pixels.permute(2, 0, 3, 1).flatten(1, 2).numpy()
        return Image.fromarray(pixels)


def load_decoder(name, path):
    from mmgp import offload
    config = REGISTRY['decoders'][name]
    with torch.device('meta'):
        if name == 'taeqi2_1':
            from .taesd import F16Decoder
            net = F16Decoder()
        elif config['kind'] == 'image':
            from .taesd import Decoder
            net = Decoder(config['channels'], use_midblock_gn=name == 'taef2')
        elif config['kind'] == 'h3':
            from safetensors import safe_open
            from .taeh3 import build_h3_decoder
            with safe_open(path, framework='pt', device='cpu') as weights:
                shapes = {key: torch.empty(weights.get_slice(key).get_shape(), device='meta') for key in weights.keys()}
            net = build_h3_decoder(shapes)
        else:
            from .taehv import TAEHV
            net = TAEHV(patch_size=config['patch'], latent_channels=config['channels'], decoder_time_upscale=config['temporal'])
        model = TinyVAE(name, net)

    def decoder_weights(state):
        if name == 'taeqi2_1':
            decoder_state = {}
            for key, value in state.items():
                if key.startswith('decoder.layers.'):
                    index, suffix = key.removeprefix('decoder.layers.').split('.', 1)
                    decoder_state[f'{int(index) + 1}.{suffix}'] = value
            state = decoder_state
        elif config['kind'] == 'video':
            state = net.patch_tgrow_layers({key: value for key, value in state.items() if key.startswith('decoder.')})
        return {'net.' + key: value for key, value in state.items()}

    offload.load_model_data(model, path, preprocess_sd=decoder_weights, writable_tensors=False, default_dtype=None)
    return model.eval().requires_grad_(False)
