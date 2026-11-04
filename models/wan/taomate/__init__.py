# Adapted from TaoMate-FVSR (https://github.com/TaoLiveAIGC/TaoMate-FVSR, Apache-2.0, see LICENSE in this folder): the audio projection keeps its
# checkpoint names. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""TaoMate-FVSR on the Wan 2.2 5B transformer: no text, HuBERT windows attended per latent frame in every block, a reference canvas added to the
patch embeddings, sampled by windows of 20 latent frames (refine.py)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.attention import pay_attention

AUDIO_FRAMES = 20
AUDIO_WINDOW = 8
AUDIO_FEATURES = 4096
AUDIO_PROJECTION_DIM = 1536
AUDIO_PROJECTION_LAYERS = 4
AUDIO_PROJECTION_HEADS = 24


class AudioPositions(nn.Module):
    def __init__(self, dim, length=100):
        super().__init__()
        self.register_buffer("pe", torch.empty(1, length, dim))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class AudioAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])

    def forward(self, x):
        qkv_list = [layer(x).unflatten(-1, (self.heads, -1)) for layer in (self.to_q, self.to_k, self.to_v)]
        return self.to_out[0](pay_attention(qkv_list, force_attention="sdpa").flatten(2))


class AudioGEGLU(nn.Module):
    def __init__(self, dim, inner):
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class AudioFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.ModuleList([AudioGEGLU(dim, dim * mult), nn.Dropout(0.0), nn.Linear(dim * mult, dim)])

    def forward(self, x):
        return self.net[2](self.net[0](x))


class AudioProjectionLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AudioAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = AudioFeedForward(dim)
        self.pos_embed = AudioPositions(dim)

    def forward(self, x):
        x = x + self.attn(self.pos_embed(self.norm1(x)))
        return x + self.ff(self.norm2(x))


class AudioProjection(nn.Module):
    """(frames, window, features) raw hidden states to (frames, window, model width), each frame's window attended on its own."""

    def __init__(self, output_dim):
        super().__init__()
        self.proj_in = nn.Linear(AUDIO_FEATURES, AUDIO_PROJECTION_DIM)
        self.layers = nn.ModuleList([AudioProjectionLayer(AUDIO_PROJECTION_DIM, AUDIO_PROJECTION_HEADS) for _ in range(AUDIO_PROJECTION_LAYERS)])
        self.proj_out = nn.Linear(AUDIO_PROJECTION_DIM, output_dim)

    def forward(self, features):
        x = self.proj_in(features)
        for layer in self.layers:
            x = layer(x)
        return self.proj_out(x)


class FrameAudioAttention(nn.Module):
    """Cross attention of each latent frame's tokens with its own window of audio tokens: the frames are the batch of the attention."""

    def __init__(self, dim, num_heads, eps):
        super().__init__()
        from ..modules.model import WanRMSNorm

        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x_list, audio):
        q = self.q(x_list[0])
        x_list.clear()
        self.norm_q(q)
        k = self.k(audio)
        self.norm_k(k)
        qkv_list = [q.unflatten(-1, (self.num_heads, -1)), k.unflatten(-1, (self.num_heads, -1)), self.v(audio).unflatten(-1, (self.num_heads, -1))]
        q = k = None
        return self.o(pay_attention(qkv_list, recycle_q=True).flatten(2))


def add_modules(model):
    """Replaces the text path of a WanModel with the audio path and adds the reference projection, the names of the checkpoint."""
    from ..modules.model import WanLayerNorm

    del model.text_embedding
    model.spark_ref_proj = nn.Conv3d(model.in_dim, model.dim, kernel_size=model.patch_size, stride=model.patch_size, bias=False)
    model.audio_embedding = AudioProjection(model.dim)
    for block in model.blocks:
        block.cross_attn = block.norm3 = None
        block.norm4 = WanLayerNorm(model.dim, model.eps, elementwise_affine=True)
        block.audio_attn = FrameAudioAttention(model.dim, model.num_heads, model.eps)


def audio_attention(block, x, audio, attention_dtype):
    """x (1, frames * tokens, dim) + the attention of each frame's tokens with audio (frames, window, dim), in place."""
    y_list = [block.norm4(x).to(attention_dtype).view(audio.shape[0], -1, x.shape[-1])]
    x += block.audio_attn(y_list, audio).view(x.shape).to(x.dtype)
    return x


__all__ = ["AUDIO_FEATURES", "AUDIO_FRAMES", "AUDIO_WINDOW", "add_modules", "audio_attention"]
