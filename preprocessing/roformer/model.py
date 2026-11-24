# BS-RoFormer vocal separation, inference only.
# Architecture from lucidrains/BS-RoFormer (MIT, Phil Wang), as shipped in audio-separator 0.36.1
# (separator/uvr_lib_v5/roformer/bs_roformer.py, MIT, karaokenerds); the chunked separation follows that package's
# architectures/mdxc_separator.py. Rotary positions follow rotary-embedding-torch 0.6.5 (MIT, Phil Wang).
# WanGP changes: training loss, linear attention and dropout removed; one checkpoint's configuration fixed below;
# separate q/k/v projections through WanGP's attention kit and shared kernels; no einops / beartype.
# See LICENSE_BS_ROFORMER and LICENSE_AUDIO_SEPARATOR.
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from torch import nn

from shared import attention_kit
from shared.kernels import ops

SAMPLE_RATE = 44100
N_FFT, HOP = 2048, 441
CHUNK = HOP * 800  # audio.hop_length * (inference.dim_t - 1)
STEP = 8 * SAMPLE_RATE  # audio-separator's default MDXC overlap of 8 s
DIM, DEPTH, HEADS, DIM_HEAD = 512, 12, 8, 64
FREQS_PER_BANDS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)
CHANNELS = 2
BAND_WIDTHS = tuple(2 * CHANNELS * f for f in FREQS_PER_BANDS)  # stereo and real/imaginary parts merged into each band
EPS = 1e-30  # the upstream norm divides by the clamped L2 norm: no epsilon of its own


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


def rms_norm(x, weight):
    y = ops.rms_norm_(x.clone(), EPS, weight)
    return y if y is not None else F.rms_norm(x.float(), (x.shape[-1],), weight.float(), EPS).to(x.dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        inner = HEADS * DIM_HEAD
        self.norm = RMSNorm(DIM)
        self.q, self.k, self.v = (nn.Linear(DIM, inner, bias=False) for _ in range(3))
        self.gates = nn.Linear(DIM, HEADS)
        self.out = nn.Linear(inner, DIM, bias=False)

    def forward(self, x, rope):
        """x (batch, tokens, DIM) += attention, in place."""
        cosine, sine = rope
        h = rms_norm(x, self.norm.weight)
        head_gate = self.gates(h).sigmoid_()

        def norm_rope(query, key, group):
            for tensor in (query, key):
                if tensor is not None:
                    attention_kit.rows_(tensor, lambda part, first, last: ops.paired_rope_(part.float(), cosine[:, first:last], sine[:, first:last], interleaved=True).to(part.dtype))

        inputs, h = [h], None
        attention_kit.qkv_attention(inputs, self.q, self.k, self.v, HEADS, DIM_HEAD, norm_rope, out_proj=self.out, residual=x, head_gate=head_gate)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(DIM)
        self.fc1, self.fc2 = nn.Linear(DIM, DIM * 4), nn.Linear(DIM * 4, DIM)

    def forward(self, x):
        ops.mlp([rms_norm(x, self.norm.weight)], self.fc1, self.fc2, "gelu_erf", residual=x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn, self.ff = Attention(), FeedForward()

    def forward(self, x, rope):
        self.attn(x, rope)
        self.ff(x)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.time, self.freq = Block(), Block()


class BandIn(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = RMSNorm(width)
        self.proj = nn.Linear(width, DIM)


class BandOut(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(DIM, DIM * 4), nn.Linear(DIM * 4, width * 2)


def _rope(tokens, device):
    """cos / sin (1, tokens, 1, DIM_HEAD / 2) of rotary-embedding-torch's default table (theta 10000, positions from 0, FP32)."""
    freqs = 1.0 / 10000 ** (torch.arange(0, DIM_HEAD, 2, device=device, dtype=torch.float32) / DIM_HEAD)
    angles = torch.arange(tokens, device=device, dtype=torch.float32)[:, None] * freqs
    return angles.cos()[None, :, None], angles.sin()[None, :, None]


class BSRoformer(nn.Module):
    """One stem (vocals) of a stereo mix: forward((batch, 2, samples) FP32) -> (batch, 2, samples) FP32."""

    def __init__(self):
        super().__init__()
        self.band_split = nn.ModuleList(BandIn(width) for width in BAND_WIDTHS)
        self.layers = nn.ModuleList(Layer() for _ in range(DEPTH))
        self.final_norm = RMSNorm(DIM)
        self.mask = nn.ModuleList(BandOut(width) for width in BAND_WIDTHS)
        self._lock_dtype = torch.float32  # the audio reaches the STFT in FP32

    def forward(self, audio, checkpoint=None):
        batch, channels, samples = audio.shape
        device, dtype = audio.device, self.final_norm.weight.dtype
        window = torch.hann_window(N_FFT, device=device)
        spec = torch.stft(audio.reshape(batch * channels, samples), N_FFT, HOP, N_FFT, window, return_complex=True)  # (b s, f, t)
        freqs, frames = spec.shape[-2:]
        spec = spec.view(batch, channels, freqs, frames)
        x = torch.view_as_real(spec).permute(0, 3, 2, 1, 4).reshape(batch, frames, -1)  # b t (f s c)
        bands = []
        for band, part in zip(self.band_split, x.split(BAND_WIDTHS, dim=-1)):
            part = part / torch.linalg.vector_norm(part, dim=-1, keepdim=True).clamp_min(1e-12) * math.sqrt(part.shape[-1]) * band.norm.weight.float()
            bands.append(band.proj(part.to(dtype)))
        del x, part
        x = torch.stack(bands, dim=-2)  # (b, t, bands, d)
        del bands
        nbands = x.shape[-2]
        time_rope, freq_rope = _rope(frames, device), _rope(nbands, device)
        for layer in self.layers:
            x = x.permute(0, 2, 1, 3).contiguous().view(batch * nbands, frames, DIM)  # residuals are written in place: contiguous copies
            layer.time(x, time_rope)
            x = x.view(batch, nbands, frames, DIM).permute(0, 2, 1, 3).contiguous().view(batch * frames, nbands, DIM)
            layer.freq(x, freq_rope)
            x = x.view(batch, frames, nbands, DIM)
            if checkpoint is not None:
                checkpoint()
        x = rms_norm(x, self.final_norm.weight)
        mask = torch.empty((batch, frames, sum(BAND_WIDTHS)), device=device, dtype=torch.float32)
        start = 0
        for index, (band, width) in enumerate(zip(self.mask, BAND_WIDTHS)):
            mask[..., start:start + width] = F.glu(band.fc2(torch.tanh(band.fc1(x[:, :, index]))).float(), dim=-1)
            start += width
        del x
        mask = torch.view_as_complex(mask.view(batch, frames, freqs, channels, 2).permute(0, 3, 2, 1, 4).contiguous())  # (b, s, f, t)
        spec = (spec * mask).view(batch * channels, freqs, frames)
        del mask
        return torch.istft(spec, N_FFT, HOP, N_FFT, window).view(batch, channels, -1)


def demix(model, mix, device, checkpoint=None, progress=None):
    """Vocals of mix (2, samples) FP32 CPU, as audio-separator's MDXC separation of a Roformer: windows of CHUNK samples every STEP
    samples (the last one ending with the mix), weighted by a Hamming window and normalized by the summed weights."""
    samples = mix.shape[1]
    window = torch.tensor(signal.windows.hamming(CHUNK), dtype=torch.float32, device="cpu")
    result = torch.zeros(mix.shape, dtype=torch.float32, device="cpu")
    counter = torch.zeros(mix.shape, dtype=torch.float32, device="cpu")
    starts = range(0, samples, STEP)
    for i in starts if progress is None else progress(starts):
        if checkpoint is not None:
            checkpoint()
        start = i if i + CHUNK <= samples else samples - CHUNK
        part = mix[:, start:start + CHUNK]
        length = part.shape[-1]
        out = model(part.to(device).unsqueeze(0), checkpoint)[0].cpu()
        result[..., start:start + length] += out[..., :length] * window[:length]
        counter[..., start:start + length] += window[:length]
        del out
    return (result / counter.clamp(min=1e-10)).numpy()


def normalize(wave, max_peak=0.9):
    """audio-separator's peak normalization, in place."""
    peak = np.abs(wave).max()
    if peak > max_peak:
        wave *= max_peak / peak
    return wave
