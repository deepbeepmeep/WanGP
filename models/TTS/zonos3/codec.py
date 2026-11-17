# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""ZAC-1 decoder of ZONOS3-Mini (Zyphra, Apache-2.0 weights; architecture of the MIT release code: the dual codec's RVQ lookups and
semantic head, the pruned keep2_plain stages of its MOSS-v2-style causal transformer decoder, the multi-resolution iSTFT head).

Precision follows the released decoder: its transformer stages run under BF16 autocast (BF16 matrices; LayerNorms, layer scales and
the residual stream float32), the codebook lookups, the semantic head, the iSTFT head, the FIR mixer of its two branches and the
44160 -> 44100 Hz resampler in float32. Causal throughout: a frame's audio depends on the frames before it only."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from shared.attention import pay_attention

BF16 = torch.bfloat16
ATTENTION_ROWS = 512  # query rows per local-attention call


class Quantizer(nn.Module):
    """One codebook: codes -> codebook rows (dimension 8) -> projected latent (a folded weight-normed 1x1 convolution)."""

    def __init__(self, size, dim, out_dim):
        super().__init__()
        self.codebook = nn.Embedding(size, dim)
        self.out_proj = nn.Linear(dim, out_dim)

    def forward(self, codes):
        return self.out_proj(self.codebook(codes))


class ConvNeXt(nn.Module):
    """Causal ConvNeXt block (depthwise convolution of 7 taps on the past, LayerNorm, GELU MLP, residual). compute_dtype: that of its
    convolution and matrices (the residual and the LayerNorm stay float32)."""

    def __init__(self, width, hidden, dtype):
        super().__init__()
        self.dwconv = nn.Conv1d(width, width, 7, groups=width, dtype=dtype)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, hidden, dtype=dtype)
        self.pwconv2 = nn.Linear(hidden, width, dtype=dtype)

    def forward(self, x):
        dtype = self.dwconv.weight.dtype
        h = self.dwconv(F.pad(x.to(dtype), (6, 0))).transpose(1, 2)
        h = self.pwconv2(F.gelu(self.pwconv1(self.norm(h.float()).to(dtype))))
        return x + h.transpose(1, 2)


class TransformerLayer(nn.Module):
    """Pre-norm layer with causal local attention (context frames, RoPE on adjacent pairs) and a GELU MLP, LayerScale residuals."""

    def __init__(self, dim, heads, ffn, context):
        super().__init__()
        self.heads, self.head_dim, self.context = heads, dim // heads, context
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False, dtype=BF16)
        self.attention_out = nn.Linear(dim, dim, bias=False, dtype=BF16)
        self.ffn = nn.ModuleDict({"0": nn.Linear(dim, ffn, bias=False, dtype=BF16), "3": nn.Linear(ffn, dim, bias=False, dtype=BF16)})
        self.attention_scale = nn.Parameter(torch.empty(dim))
        self.ffn_scale = nn.Parameter(torch.empty(dim))

    def attention(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x.to(BF16)).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        frequency = torch.exp(torch.arange(self.head_dim // 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) * 2 / self.head_dim))
        phase = torch.arange(t, device=x.device, dtype=torch.float32)[:, None] * frequency[None, :]
        cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]

        def rope(z):
            r, i = z.float().unflatten(-1, (-1, 2)).unbind(-1)
            return torch.stack((r * cos - i * sin, r * sin + i * cos), -1).flatten(-2).to(z.dtype)

        q, k = rope(q), rope(k)
        out = torch.empty_like(q)
        for start in range(0, t, ATTENTION_ROWS):
            end = min(t, start + ATTENTION_ROWS)
            left = max(0, start - self.context + 1)
            delta = torch.arange(start, end, device=x.device)[:, None] - torch.arange(left, end, device=x.device)[None, :]
            mask = (delta >= 0) & (delta < self.context)
            out[:, start:end] = pay_attention([q[:, start:end], k[:, left:end], v[:, left:end]], attention_mask=mask[None, :, None], force_attention="sdpa")
        return self.attention_out(out.reshape(b, t, d))

    def forward(self, x):
        x = x + self.attention_scale * self.attention(self.norm1(x.float()))
        h = self.ffn["0"](self.norm2(x).to(BF16))
        return x + self.ffn_scale * self.ffn["3"](F.gelu(h))


class TransformerStage(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.input_proj = nn.Linear(spec["input_dim"], spec["dim"], bias=False, dtype=BF16)
        self.layers = nn.ModuleList([TransformerLayer(spec["dim"], spec["heads"], spec["ffn"], spec["context"]) for _ in range(spec["layers"])])
        self.output_proj = None if spec["output_dim"] is None else nn.Linear(spec["dim"], spec["output_dim"], bias=False, dtype=BF16)

    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2).to(BF16))
        for layer in self.layers:
            x = layer(x)
        x = x if self.output_proj is None else self.output_proj(x.to(BF16))
        return x.transpose(1, 2)


class Unpatch(nn.Module):
    """[b, c, t] -> [b, c / size, t * size]: each frame's channels become size consecutive frames."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        b, c, t = x.shape
        return x.reshape(b, c // self.size, self.size, t).permute(0, 1, 3, 2).reshape(b, c // self.size, t * self.size)


class MultiResISTFTHead(nn.Module):
    """Causal ConvNeXt trunk, one log-magnitude / phase projection per window size, a centered iSTFT per branch at the shared hop and
    a linear FIR mixer of the branches (float32)."""

    def __init__(self, dim, spec):
        super().__init__()
        self.hop, self.n_ffts = spec["hop"], spec["n_ffts"]
        self.blocks = nn.Sequential(*[ConvNeXt(dim, spec["hidden"], torch.float32) for _ in range(spec["depth"])])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.outs = nn.ModuleList([nn.Linear(dim, n + 2) for n in self.n_ffts])
        self.mixer = nn.Conv1d(len(self.n_ffts), 1, spec["fir_kernel"], padding=(spec["fir_kernel"] - 1) // 2, bias=False)

    def forward(self, x):
        frames = x.shape[-1]
        h = self.norm(self.blocks(x.float()).transpose(1, 2))
        branches = []
        for n_fft, out in zip(self.n_ffts, self.outs):
            log_magnitude, phase = out(h).chunk(2, dim=-1)
            magnitude = torch.exp(log_magnitude.clamp(max=math.log(100.0))).transpose(1, 2)
            phase = phase.transpose(1, 2)
            spec = torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
            window = torch.hann_window(n_fft, periodic=True, device="cpu").to(x.device)  # built on the CPU as released
            branches.append(torch.istft(spec, n_fft, hop_length=self.hop, win_length=n_fft, window=window, center=True, length=frames * self.hop))
            del spec, magnitude, phase, log_magnitude
        return self.mixer(torch.stack(branches, dim=1))


class ZAC1Decoder(nn.Module):
    """Codes [batch, 10, frames] (codebook 0 semantic, 1-9 acoustic) -> audio [batch, frames * hop] at 44.1 kHz, float32."""

    def __init__(self, c):
        super().__init__()
        self.hop, self.sample_rate = c["hop"], c["sample_rate"]
        semantic = c["semantic_head"]
        self.semantic_vq = nn.ModuleList([Quantizer(4096, 8, semantic["width"])])
        self.acoustic_vq = nn.ModuleList([Quantizer(4096, 8, 1024) for _ in range(9)])
        self.semantic_head = nn.Sequential(*[ConvNeXt(semantic["width"], semantic["hidden"], torch.float32) for _ in range(semantic["depth"])],
                                           nn.Conv1d(semantic["width"], 1024, 1))
        self.stages = nn.ModuleList([TransformerStage(spec) if spec["type"] == "transformer" else Unpatch(spec["size"]) for spec in c["stages"]])
        self.head = MultiResISTFTHead(c["stages"][-1]["dim"], c["head"])
        self.to_external = nn.Module()
        self.to_external.kernel = nn.Parameter(torch.empty(735, 1, 770), requires_grad=False)
        self.internal_rate = c["internal_sample_rate"]

    def latent(self, codes):
        """Dequantized latent [batch, 1024, frames], float32."""
        z = self.semantic_head(self.semantic_vq[0](codes[:, 0]).transpose(1, 2))
        for index, vq in enumerate(self.acoustic_vq):
            z = z + vq(codes[:, 1 + index]).transpose(1, 2)
        return z

    def resample(self, audio):
        """44160 -> 44100 Hz with the codec's windowed-sinc kernel (torchaudio's resampling, lowpass width 16, rolloff 0.99)."""
        divisor = math.gcd(self.internal_rate, self.sample_rate)
        orig, new = self.internal_rate // divisor, self.sample_rate // divisor
        width = (self.to_external.kernel.shape[-1] - orig) // 2
        length = audio.shape[-1]
        resampled = F.conv1d(F.pad(audio, (width, width + orig))[:, None], self.to_external.kernel, stride=orig)
        return resampled.transpose(1, 2).reshape(audio.shape[0], -1)[:, :math.ceil(new * length / orig)]

    def forward(self, codes):
        frames = codes.shape[-1]
        x = self.latent(codes.long())
        for stage in self.stages:
            x = stage(x)
        audio = self.resample(self.head(x)[:, 0])
        return audio[:, :frames * self.hop]
