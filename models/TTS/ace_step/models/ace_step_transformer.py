# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin


from .attention import LinearTransformerBlock, t2i_modulate
from .lyrics_utils.lyric_encoder import ConformerEncoder as LyricEncoder


def cross_norm(hidden_states, controlnet_input):
    # input N x T x c
    mean_hidden_states, std_hidden_states = hidden_states.mean(
        dim=(1, 2), keepdim=True
    ), hidden_states.std(dim=(1, 2), keepdim=True)
    mean_controlnet_input, std_controlnet_input = controlnet_input.mean(
        dim=(1, 2), keepdim=True
    ), controlnet_input.std(dim=(1, 2), keepdim=True)
    controlnet_input = (controlnet_input - mean_controlnet_input) * (
        std_hidden_states / (std_controlnet_input + 1e-12)
    ) + mean_hidden_states
    return controlnet_input


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size=[16, 1], out_channels=256):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels
        self.patch_size = patch_size

    def unpatchfy(
        self,
        hidden_states: torch.Tensor,
        width: int,
    ):
        # 4 unpatchify
        new_height, new_width = 1, hidden_states.size(1)
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                new_height,
                new_width,
                self.patch_size[0],
                self.patch_size[1],
                self.out_channels,
            )
        ).contiguous()
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                new_height * self.patch_size[0],
                new_width * self.patch_size[1],
            )
        ).contiguous()
        if width > new_width:
            output = torch.nn.functional.pad(
                output, (0, width - new_width, 0, 0), "constant", 0
            )
        elif width < new_width:
            output = output[:, :, :, :width]
        return output

    def forward(self, x, t, output_length):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # unpatchify
        output = self.unpatchfy(x, output_length)
        return output


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=16,
        width=4096,
        patch_size=(16, 1),
        in_channels=8,
        embed_dim=1152,
        bias=True,
    ):
        super().__init__()
        patch_size_h, patch_size_w = patch_size
        self.early_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 256,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=bias,
            ),
            torch.nn.GroupNorm(
                num_groups=32, num_channels=in_channels * 256, eps=1e-6, affine=True
            ),
            nn.Conv2d(
                in_channels * 256,
                embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )
        self.patch_size = patch_size
        self.height, self.width = height // patch_size_h, width // patch_size_w
        self.base_size = self.width

    def forward(self, latent):
        # early convolutions, N x C x H x W -> N x 256 * sqrt(patch_size) x H/patch_size x W/patch_size
        latent = self.early_conv_layers(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return latent


@dataclass
class Transformer2DModelOutput(BaseOutput):

    sample: torch.FloatTensor
    proj_losses: Optional[Tuple[Tuple[str, torch.Tensor]]] = None


class ACEStepTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin
):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: Optional[int] = 8,
        num_layers: int = 28,
        inner_dim: int = 1536,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        mlp_ratio: float = 4.0,
        out_channels: int = 8,
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        speaker_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        ssl_encoder_depths: List[int] = [9, 9],
        ssl_names: List[str] = ["mert", "m-hubert"],
        ssl_latent_dims: List[int] = [1024, 768],
        lyric_encoder_vocab_size: int = 6681,
        lyric_hidden_size: int = 1024,
        patch_size: List[int] = [16, 1],
        max_height: int = 16,
        max_width: int = 4096,
        **kwargs,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.out_channels = out_channels
        self.max_position = max_position
        self.patch_size = patch_size

        self.rope_theta = rope_theta

        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=self.attention_head_dim,
            max_position_embeddings=self.max_position,
            base=self.rope_theta,
        )

        # 2. Define input layers
        self.in_channels = in_channels

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                LinearTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    add_cross_attention=True,
                    add_cross_attention_dim=self.inner_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.num_layers = num_layers

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=self.inner_dim
        )
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(self.inner_dim, 6 * self.inner_dim, bias=True)
        )

        # speaker
        self.speaker_embedder = nn.Linear(speaker_embedding_dim, self.inner_dim)

        # genre
        self.genre_embedder = nn.Linear(text_embedding_dim, self.inner_dim)

        # lyric
        self.lyric_embs = nn.Embedding(lyric_encoder_vocab_size, lyric_hidden_size)
        self.lyric_encoder = LyricEncoder(
            input_size=lyric_hidden_size, static_chunk_size=0
        )
        self.lyric_proj = nn.Linear(lyric_hidden_size, self.inner_dim)

        projector_dim = 2 * self.inner_dim

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.inner_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, ssl_dim),
                )
                for ssl_dim in ssl_latent_dims
            ]
        )

        self.ssl_latent_dims = ssl_latent_dims
        self.ssl_encoder_depths = ssl_encoder_depths

        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")
        self.ssl_names = ssl_names

        self.proj_in = PatchEmbed(
            height=max_height,
            width=max_width,
            patch_size=patch_size,
            embed_dim=self.inner_dim,
            bias=True,
        )

        self.final_layer = T2IFinalLayer(
            self.inner_dim, patch_size=patch_size, out_channels=out_channels
        )
        self.gradient_checkpointing = False
        # These hold no weights themselves: their FP32 locks keep MMGP from rounding the float32 timesteps (0..1000) to BF16 before the sinusoid
        self._lock_dtype = self.time_proj._lock_dtype = torch.float32

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward_lyric_encoder(
        self,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
    ):
        # N x T x D
        lyric_embs = self.lyric_embs(lyric_token_idx)
        prompt_prenet_out, _mask = self.lyric_encoder(
            lyric_embs, lyric_mask, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        prompt_prenet_out = self.lyric_proj(prompt_prenet_out)
        return prompt_prenet_out

    def encode(
        self,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
    ):

        bs = encoder_text_hidden_states.shape[0]
        device = encoder_text_hidden_states.device

        # speaker embedding
        encoder_spk_hidden_states = self.speaker_embedder(speaker_embeds).unsqueeze(1)
        speaker_mask = torch.ones(bs, 1, device=device)

        # genre embedding
        encoder_text_hidden_states = self.genre_embedder(encoder_text_hidden_states)

        # lyric
        encoder_lyric_hidden_states = self.forward_lyric_encoder(
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        encoder_hidden_states = torch.cat(
            [
                encoder_spk_hidden_states,
                encoder_text_hidden_states,
                encoder_lyric_hidden_states,
            ],
            dim=1,
        )
        encoder_hidden_mask = torch.cat(
            [speaker_mask, text_attention_mask, lyric_mask], dim=1
        )
        return encoder_hidden_states, encoder_hidden_mask

    def decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_mask: torch.Tensor,
        timestep: Optional[torch.Tensor],
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        output_length: int = 0,
        block_controlnet_hidden_states: Optional[
            Union[List[torch.Tensor], torch.Tensor]
        ] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
    ):

        embedded_timestep = self.timestep_embedder(
            self.time_proj(timestep).to(dtype=hidden_states.dtype)
        )
        temb = self.t_block(embedded_timestep)

        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_mask = encoder_hidden_mask.to(hidden_states.dtype)  # float32 from encode(): the blocks' compute dtype, as MMGP converted it before the root lock

        # controlnet logic
        if block_controlnet_hidden_states is not None:
            control_condi = cross_norm(hidden_states, block_controlnet_hidden_states)
            hidden_states = hidden_states + control_condi * controlnet_scale

        inner_hidden_states = []

        rotary_freqs_cis = self.rotary_emb(
            hidden_states, seq_len=hidden_states.shape[1]
        )
        encoder_rotary_freqs_cis = self.rotary_emb(
            encoder_hidden_states, seq_len=encoder_hidden_states.shape[1]
        )

        for index_block, block in enumerate(self.transformer_blocks):

            if self.training and self.gradient_checkpointing:

                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_hidden_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                    temb=temb,
                    use_reentrant=False,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_hidden_mask,
                    rotary_freqs_cis=rotary_freqs_cis,
                    rotary_freqs_cis_cross=encoder_rotary_freqs_cis,
                    temb=temb,
                )

            for ssl_encoder_depth in self.ssl_encoder_depths:
                if index_block == ssl_encoder_depth:
                    inner_hidden_states.append(hidden_states)

        proj_losses = []
        if (
            len(inner_hidden_states) > 0
            and ssl_hidden_states is not None
            and len(ssl_hidden_states) > 0
        ):

            for inner_hidden_state, projector, ssl_hidden_state, ssl_name in zip(
                inner_hidden_states, self.projectors, ssl_hidden_states, self.ssl_names
            ):
                if ssl_hidden_state is None:
                    continue
                # 1. N x T x D1 -> N x D x D2
                est_ssl_hidden_state = projector(inner_hidden_state)
                # 3. projection loss
                bs = inner_hidden_state.shape[0]
                proj_loss = 0.0
                for i, (z, z_tilde) in enumerate(
                    zip(ssl_hidden_state, est_ssl_hidden_state)
                ):
                    # 2. interpolate
                    z_tilde = (
                        F.interpolate(
                            z_tilde.unsqueeze(0).transpose(1, 2),
                            size=len(z),
                            mode="linear",
                            align_corners=False,
                        )
                        .transpose(1, 2)
                        .squeeze(0)
                    )

                    z_tilde = torch.nn.functional.normalize(z_tilde, dim=-1)
                    z = torch.nn.functional.normalize(z, dim=-1)
                    # T x d -> T x 1 -> 1
                    target = torch.ones(z.shape[0], device=z.device)
                    proj_loss += self.cosine_loss(z, z_tilde, target)
                proj_losses.append((ssl_name, proj_loss / bs))

        output = self.final_layer(hidden_states, embedded_timestep, output_length)
        if not return_dict:
            return (output, proj_losses)

        return Transformer2DModelOutput(sample=output, proj_losses=proj_losses)

    # @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_text_hidden_states: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.LongTensor] = None,
        speaker_embeds: Optional[torch.FloatTensor] = None,
        lyric_token_idx: Optional[torch.LongTensor] = None,
        lyric_mask: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.Tensor] = None,
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        block_controlnet_hidden_states: Optional[
            Union[List[torch.Tensor], torch.Tensor]
        ] = None,
        controlnet_scale: Union[float, torch.Tensor] = 1.0,
        return_dict: bool = True,
    ):
        encoder_hidden_states, encoder_hidden_mask = self.encode(
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_idx,
            lyric_mask=lyric_mask,
        )

        output_length = hidden_states.shape[-1]

        output = self.decode(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_mask=encoder_hidden_mask,
            timestep=timestep,
            ssl_hidden_states=ssl_hidden_states,
            output_length=output_length,
            block_controlnet_hidden_states=block_controlnet_hidden_states,
            controlnet_scale=controlnet_scale,
            return_dict=return_dict,
        )

        return output
