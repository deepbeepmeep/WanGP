from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmgp import cuda_graphs
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys
import torch.amp as amp

import numpy as np 
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import Attention, MultiHeadCrossAttention
from .blocks import TimestepEmbedder, CaptionEmbedder, PatchEmbed3D, FeedForwardSwiGLU, FinalLayer_FP32, LayerNorm_FP32, modulate_fp32, _take_tensor


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn  = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))
        self.ffn_mult = self.ffn.ffn_mult
        self.ffn_chunk_min = 128

    def _apply_ffn_chunked(self, x_list):
        return ops.mlp(x_list, self.ffn.w1, self.ffn.w2, "swiglu", up=self.ffn.w3)

    def forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None, skip_crs_attn=False, rope=None):
        """
            x: [B, N, C]
            y: [1, N_valid_tokens, C]
            t: [B, T, C_t]
            y_seqlen: [B]; type of a list
            latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape # S != T*H*W in case of CP split on H*W.

        # compute modulation params in fp32
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1) # [B, T, 1, C]

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
        x_m_list = [x_m]
        x_m = None

        x_s = self.attn(x_m_list, shape=latent_shape, num_cond_latents=num_cond_latents,
                        return_kv=return_kv, layer_cache=kv_cache, rope=rope)
        x_m = None

        x.view(B, -1, N//T, C).addcmul_(x_s.view(B, -1, N//T, C), gate_msa)
        del x_s, gate_msa, shift_msa, scale_msa
        x = x.to(x_dtype)

        # cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            cross_in = self.pre_crs_attn_norm(x)
            cross_in_list = [cross_in]
            cross_in = None
            cond_tokens, cross_out = self.cross_attn.forward_noise(cross_in_list, y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)
            if cond_tokens:
                x[:, cond_tokens:].add_(cross_out)
            else:
                x.add_(cross_out)
            del cross_out

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_m_list = [x_m]
        x_m = None
        x_s = self._apply_ffn_chunked(x_m_list)
        x.view(B, -1, N//T, C).addcmul_(x_s.view(B, -1, N//T, C), gate_mlp)
        del x_s, gate_mlp, shift_mlp, scale_mlp
        x = x.to(x_dtype)

        return x


class LongCatVideoTransformer3DModel(
    ModelMixin, ConfigMixin
):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
        text_tokens_zero_pad: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self._rope_cache = {}
        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self._interrupt_check = None
    
    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")
    
    def enable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = True
    
    def disable_bsa(self,):
        for block in self.blocks:
            block.attn.enable_bsa = False    

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(state_dict, dict(self.named_modules()),
                                 split_linear_modules_map=getattr(self, "split_linear_modules_map", None))

    def clear_runtime_caches(self):
        self._rope_cache.clear()
        for block in self.blocks:
            block.attn.rope_3d.freqs_dict.clear()

    def forward(
        self, 
        hidden_states, 
        timestep, 
        encoder_hidden_states, 
        encoder_attention_mask=None, 
        num_cond_latents=0,
        return_kv=False, 
        kv_cache_dict=None,
        skip_crs_attn=False, 
    ):
        x_list = hidden_states if isinstance(hidden_states, list) else [hidden_states]
        joint_pass = isinstance(hidden_states, list)

        if not isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [encoder_hidden_states] * len(x_list)
        if not isinstance(encoder_attention_mask, list):
            encoder_attention_mask = [encoder_attention_mask] * len(x_list)
        if not isinstance(timestep, list):
            timestep = [timestep] * len(x_list)
        if not isinstance(num_cond_latents, list):
            num_cond_latents = [num_cond_latents] * len(x_list)
        if kv_cache_dict is None:
            kv_cache_dict = [None] * len(x_list)
        elif not isinstance(kv_cache_dict, list):
            kv_cache_dict = [kv_cache_dict] * len(x_list)

        dtype = self.x_embedder.proj.weight.dtype
        t_list = []
        enc_list = []
        y_seqlens_list = []
        latent_shapes = []
        rope_tables = []
        for idx, (x, step, enc, mask) in enumerate(zip(x_list, timestep, encoder_hidden_states, encoder_attention_mask)):
            B, _, T, H, W = x.shape
            N_t = T // self.patch_size[0]
            N_h = H // self.patch_size[1]
            N_w = W // self.patch_size[2]

            assert self.patch_size[0] == 1, "Currently, 3D x_embedder should not compress the temporal dimension."

            if len(step.shape) == 1:
                step = step.unsqueeze(1).expand(-1, N_t)

            x = x.to(dtype)
            step = step.to(dtype)
            enc = enc.to(dtype)

            x = self.x_embedder(x)

            with amp.autocast(device_type='cuda', dtype=torch.float32):
                t = self.t_embedder(step.float().flatten(), torch.float32).reshape(B, N_t, -1)
            t = t.to(torch.float32)

            enc = self.y_embedder(enc)

            if self.text_tokens_zero_pad and mask is not None:
                enc = enc * mask[:, None, :, None]

            if mask is not None:
                if mask.dim() > 2:
                    mask = mask.squeeze(1).squeeze(1)
                y_seqlens = mask.sum(dim=1).to(torch.int64).tolist()
            else:
                y_seqlens = [enc.shape[2]] * enc.shape[0]

            enc = enc.squeeze(1)

            x_list[idx] = x
            t_list.append(t)
            enc_list.append(enc)
            y_seqlens_list.append(y_seqlens)
            latent_shapes.append((N_t, N_h, N_w))
            total_t = N_t + (int(num_cond_latents[idx] or 0) if kv_cache_dict[idx] is not None and not return_kv else 0)
            rope_key = (total_t, N_h, N_w, None, None, x.device)
            if rope_key not in self._rope_cache:
                freqs = self.blocks[0].attn.rope_3d.precompute_freqs_cis_3d((total_t, N_h, N_w)).to(x.device)
                self._rope_cache[rope_key] = torch.stack((freqs.cos(), freqs.sin()), dim=-1)[None, :, None]
            rope_tables.append(self._rope_cache[rope_key])

        with cuda_graphs.scope(not return_kv):
            for block_idx, block in enumerate(self.blocks):
                if self._interrupt_check is not None and self._interrupt_check():
                    return [None] * len(x_list) if joint_pass else None
                for i in range(len(x_list)):
                    x_list[i] = block(
                        x_list[i], enc_list[i], t_list[i], y_seqlens_list[i], latent_shapes[i],
                        num_cond_latents[i], return_kv,
                        kv_cache_dict[i][block_idx] if kv_cache_dict[i] is not None else None,
                        skip_crs_attn, rope_tables[i])
                    if self._interrupt_check is not None and self._interrupt_check():
                        return [None] * len(x_list) if joint_pass else None
        if return_kv:
            return None  # Only the PrefixCache is kept; no final projection of clean latents.

        outputs = []
        for x, t, latent_shape in zip(x_list, t_list, latent_shapes):
            x = self.final_layer(x, t, latent_shape)
            x = self.unpatchify(x, *latent_shape)
            outputs.append(x.to(torch.float32))

        return outputs if joint_pass else outputs[0]
    

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x
