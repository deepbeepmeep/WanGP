"""AuK flow transformer. Adapted from Tencent-Hunyuan/AuK (MIT)."""

import torch
from torch import nn

from .modules import AdaLayerNorm_Final, ConvPositionEmbedding, DiTBlock, MMDiTBlock, TimestepEmbedding, join_tokens
from .rotary import RotaryEmbedding


class AudioPromptEmbedding(nn.Module):
    def __init__(self, latent_dim, dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim)

    def forward(self, audio):
        audio = self.linear(audio)
        return audio.add_(self.conv_pos_embed(audio))


class Flux2Edit(nn.Module):
    def __init__(self, dim=1536, heads=24, ff_mult=2, text_hidden_dim=2048, latent_dim=64, num_layers=10, num_single_layers=20):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)
        self.audio_embed = AudioPromptEmbedding(latent_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim // heads)
        self.transformer_blocks = nn.ModuleList([MMDiTBlock(dim, heads, dim // heads, ff_mult=ff_mult, attn_mask_enabled=True) for _ in range(num_layers)])
        self.single_transformer_blocks = nn.ModuleList([DiTBlock(dim, heads, dim // heads, ff_mult=ff_mult, attn_mask_enabled=True) for _ in range(num_single_layers)])
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, latent_dim)
        self._interrupt_check = lambda: False

    def forward(self, x, text, time, reference, guidance_strength, joint_pass=True):
        # Batch one has no padding: both text and audio are entirely valid.
        timestep = self.time_embed(time.expand(x.shape[0]))
        context = self.txt_norm(self.txt_proj(text))
        reference_length = reference.shape[1]

        def embed_reference(drop):
            target = self.audio_embed(x)
            if reference_length:
                ref = self.audio_embed(torch.zeros_like(reference) if drop else reference)
                target = torch.cat((ref, target), dim=1)
            return target

        audio = [embed_reference(False)]
        contexts = [context]
        if guidance_strength > 0:
            audio.append(embed_reference(True))
            contexts.append(torch.zeros_like(context))
        audio_rope = self.rotary_embed(audio[0].shape[1], x.device)
        text_rope = self.rotary_embed(context.shape[1], x.device)
        combined_rope = self.rotary_embed(audio[0].shape[1] + context.shape[1], x.device)
        del context
        if not joint_pass and len(audio) == 2:
            audio = [torch.cat(audio)]
            contexts = [torch.cat(contexts)]
            timestep = timestep.repeat(2, 1)
        for block in self.transformer_blocks:
            contexts, audio = block(audio, contexts, timestep, rope=audio_rope, c_rope=text_rope)
            if self._interrupt_check():
                raise InterruptedError('AuK generation interrupted')
        audio = [join_tokens([contexts.pop(0), audio.pop(0)]) for _ in range(len(audio))]
        for block in self.single_transformer_blocks:
            audio = block(audio, timestep, rope=combined_rope)
            if self._interrupt_check():
                raise InterruptedError('AuK generation interrupted')
        predictions = [self.proj_out(self.norm_out(branch[:, text.shape[1] + reference_length:], timestep)) for branch in audio]
        if guidance_strength == 0:
            return predictions[0]
        conditional, unconditional = predictions if joint_pass else predictions[0].chunk(2)
        return conditional + guidance_strength * (conditional - unconditional)


class AuKModel(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.transformer = Flux2Edit(**architecture)
        # holds no weights itself: its lock keeps MMGP from rounding the float32 time to BF16 before `1000 * t` and the sinusoid (its layers run under BF16 autocast)
        self.transformer._lock_dtype = torch.float32
        self.layer_weights = nn.Parameter(torch.zeros(36))
        self.layer_scale = nn.Parameter(torch.ones(1))

    @classmethod
    def from_config(cls, config):
        return cls(config['transformer'])

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)
