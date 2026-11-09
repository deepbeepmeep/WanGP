# Adapted from HunyuanVideo (https://github.com/Tencent-Hunyuan/HunyuanVideo), Copyright (C) Tencent, Tencent Hunyuan
# Community License Agreement (https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/LICENSE.txt). Modified for WanGP.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection, VisionProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, modulate_ , apply_gate, apply_gate_and_accumulate_
from .token_refiner import SingleTokenRefiner
import numpy as np
from mmgp import offload
from ..text_encoder.byT5 import ByT5Mapper
from shared.attention import pay_attention
from .audio_adapters import AudioProjNet2, PerceiverAttentionCA

def get_linear_split_map():
    hidden_size = 3072
    split_linear_modules_map =  {
                                "img_attn_qkv" : {"mapped_modules" : ["img_attn_q", "img_attn_k", "img_attn_v"] , "split_sizes": [hidden_size, hidden_size, hidden_size]},
                                "linear1" : {"mapped_modules" : ["linear1_attn_q", "linear1_attn_k", "linear1_attn_v", "linear1_mlp"] , "split_sizes":  [hidden_size, hidden_size, hidden_size, 7*hidden_size- 3*hidden_size]}
                                }
    return split_linear_modules_map


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attention_mode: str = "sdpa",
        pre_split_qkv = False,        
    ):  
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.attention_mode = attention_mode
        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.pre_split_qkv = pre_split_qkv
        if pre_split_qkv:
            self.img_attn_q  = nn.Linear( hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs )
            self.img_attn_k  = nn.Linear( hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs )
            self.img_attn_v  = nn.Linear( hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs )
        else: 
            self.img_attn_qkv = nn.Linear(
                hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
            )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        if pre_split_qkv:
            self.txt_attn_q = nn.Linear( hidden_size, hidden_size , bias=qkv_bias, **factory_kwargs )
            self.txt_attn_k = nn.Linear( hidden_size, hidden_size , bias=qkv_bias, **factory_kwargs )
            self.txt_attn_v = nn.Linear( hidden_size, hidden_size , bias=qkv_bias, **factory_kwargs )
        else:
            self.txt_attn_qkv = nn.Linear(
                hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
            )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        attn_mask = None,  
        seqlens_q: Optional[torch.Tensor] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        freqs_cis: tuple = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,        
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if condition_type == "token_replace":
            img_mod1, token_replace_img_mod1 = self.img_mod(vec, condition_type=condition_type, \
                                                            token_replace_vec=token_replace_vec)
            (img_mod1_shift,
             img_mod1_scale,
             img_mod1_gate,
             img_mod2_shift,
             img_mod2_scale,
             img_mod2_gate) = img_mod1.chunk(6, dim=-1)
            (tr_img_mod1_shift,
             tr_img_mod1_scale,
             tr_img_mod1_gate,
             tr_img_mod2_shift,
             tr_img_mod2_scale,
             tr_img_mod2_gate) = token_replace_img_mod1.chunk(6, dim=-1)
        else:
            (
                img_mod1_shift,
                img_mod1_scale,
                img_mod1_gate,
                img_mod2_shift,
                img_mod2_scale,
                img_mod2_gate,
            ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)


        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = img_modulated.to(torch.bfloat16)

        if condition_type == "token_replace":
            modulate_(img_modulated[:, :frist_frame_token_num], shift=tr_img_mod1_shift, scale=tr_img_mod1_scale)
            modulate_(img_modulated[:, frist_frame_token_num:], shift=img_mod1_shift, scale=img_mod1_scale)
        else:
            modulate_( img_modulated, shift=img_mod1_shift, scale=img_mod1_scale )

        shape = (*img_modulated.shape[:2], self.heads_num, int(img_modulated.shape[-1] / self.heads_num) )
        img_q = self.img_attn_q(img_modulated).view(*shape)
        img_k = self.img_attn_k(img_modulated).view(*shape)        
        img_v = self.img_attn_v(img_modulated).view(*shape)
        del img_modulated

        # Apply QK-Norm if needed
        self.img_attn_q_norm.apply_(img_q).to(img_v)
        img_q_len = img_q.shape[1]
        self.img_attn_k_norm.apply_(img_k).to(img_v)
        img_kv_len= img_k.shape[1]        
        batch_size = img_k.shape[0]
        # Apply RoPE if needed.
        qklist = [img_q, img_k]
        del img_q, img_k
        img_q, img_k = apply_rotary_emb(qklist, freqs_cis, head_first=False)
        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        modulate_(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale )

        if self.pre_split_qkv:
            new_shape = (*txt_modulated.shape[:-1], self.heads_num, -1)
            txt_q = self.txt_attn_q(txt_modulated).reshape(*new_shape)
            txt_k = self.txt_attn_k(txt_modulated).reshape(*new_shape)
            txt_v = self.txt_attn_v(txt_modulated).reshape(*new_shape)
            del txt_modulated
        else:
            txt_qkv = self.txt_attn_qkv(txt_modulated)
            del txt_modulated
            txt_q, txt_k, txt_v = rearrange(
                txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
            )
            del txt_qkv
        # Apply QK-Norm if needed.
        self.txt_attn_q_norm.apply_(txt_q).to(txt_v)
        self.txt_attn_k_norm.apply_(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        del img_q, txt_q
        k = torch.cat((img_k, txt_k), dim=1)        
        del img_k, txt_k
        v = torch.cat((img_v, txt_v), dim=1)
        del img_v, txt_v
        
        # attention computation start
        qkv_list = [q,k,v]
        del q, k, v

        attn = pay_attention(
            qkv_list,
            attention_mask=attn_mask,                
            q_lens=seqlens_q,
            k_lens=seqlens_kv,
        )
        b, s, a, d = attn.shape
        attn = attn.reshape(b, s, -1)        
        del qkv_list

        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        del attn
        # Calculate the img bloks.

        if condition_type == "token_replace":
            img_attn = self.img_attn_proj(img_attn)
            apply_gate_and_accumulate_(img[:, :frist_frame_token_num], img_attn[:, :frist_frame_token_num], gate=tr_img_mod1_gate)
            apply_gate_and_accumulate_(img[:, frist_frame_token_num:], img_attn[:, frist_frame_token_num:], gate=img_mod1_gate)
            del img_attn
            img_modulated = self.img_norm2(img)
            img_modulated = img_modulated.to(torch.bfloat16)
            modulate_( img_modulated[:, :frist_frame_token_num], shift=tr_img_mod2_shift, scale=tr_img_mod2_scale)
            modulate_( img_modulated[:, frist_frame_token_num:], shift=img_mod2_shift, scale=img_mod2_scale)
            self.img_mlp.apply_(img_modulated)        
            apply_gate_and_accumulate_(img[:, :frist_frame_token_num], img_modulated[:, :frist_frame_token_num], gate=tr_img_mod2_gate)
            apply_gate_and_accumulate_(img[:, frist_frame_token_num:], img_modulated[:, frist_frame_token_num:], gate=img_mod2_gate)
            del img_modulated
        else:
            img_attn = self.img_attn_proj(img_attn)
            apply_gate_and_accumulate_(img, img_attn, gate=img_mod1_gate)
            del img_attn
            img_modulated = self.img_norm2(img)
            img_modulated = img_modulated.to(torch.bfloat16)
            modulate_( img_modulated , shift=img_mod2_shift, scale=img_mod2_scale)
            self.img_mlp.apply_(img_modulated)        
            apply_gate_and_accumulate_(img, img_modulated, gate=img_mod2_gate)
            del img_modulated

        # Calculate the txt bloks.
        txt_attn  = self.txt_attn_proj(txt_attn)
        apply_gate_and_accumulate_(txt, txt_attn, gate=txt_mod1_gate)
        del txt_attn
        txt_modulated = self.txt_norm2(txt)
        txt_modulated = txt_modulated.to(torch.bfloat16)
        modulate_(txt_modulated, shift=txt_mod2_shift, scale=txt_mod2_scale)
        txt_mlp = self.txt_mlp(txt_modulated)
        del txt_modulated 
        apply_gate_and_accumulate_(txt, txt_mlp, gate=txt_mod2_gate)
        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attention_mode: str = "sdpa",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attention_mode = attention_mode
        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        # x: torch.Tensor,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        attn_mask= None,
        seqlens_q: Optional[torch.Tensor] = None,
        seqlens_kv: Optional[torch.Tensor] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        condition_type: str = None,
        token_replace_vec: torch.Tensor = None,
        frist_frame_token_num: int = None,        
    ) -> torch.Tensor:


        if condition_type == "token_replace":
            mod, tr_mod = self.modulation(vec,
                                          condition_type=condition_type,
                                          token_replace_vec=token_replace_vec)
            (mod_shift,
             mod_scale,
             mod_gate) = mod.chunk(3, dim=-1)
            (tr_mod_shift,
             tr_mod_scale,
             tr_mod_gate) = tr_mod.chunk(3, dim=-1)
        else:
            mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)

        img_mod = self.pre_norm(img)
        img_mod = img_mod.to(torch.bfloat16)
        if condition_type == "token_replace":
            modulate_(img_mod[:, :frist_frame_token_num], shift=tr_mod_shift, scale=tr_mod_scale)
            modulate_(img_mod[:, frist_frame_token_num:], shift=mod_shift, scale=mod_scale)
        else:
            modulate_(img_mod, shift=mod_shift, scale=mod_scale)
        txt_mod = self.pre_norm(txt)
        txt_mod = txt_mod.to(torch.bfloat16)
        modulate_(txt_mod, shift=mod_shift, scale=mod_scale)

        shape = (*img_mod.shape[:2], self.heads_num, int(img_mod.shape[-1] / self.heads_num) )
        img_q = self.linear1_attn_q(img_mod).view(*shape)
        img_k = self.linear1_attn_k(img_mod).view(*shape)
        img_v = self.linear1_attn_v(img_mod).view(*shape)

        shape = (*txt_mod.shape[:2], self.heads_num, int(txt_mod.shape[-1] / self.heads_num) )
        txt_q = self.linear1_attn_q(txt_mod).view(*shape)
        txt_k = self.linear1_attn_k(txt_mod).view(*shape)
        txt_v = self.linear1_attn_v(txt_mod).view(*shape)

        batch_size = img_mod.shape[0]        

        # Apply QK-Norm if needed.
        # q = self.q_norm(q).to(v)
        self.q_norm.apply_(img_q)
        self.k_norm.apply_(img_k)
        self.q_norm.apply_(txt_q)
        self.k_norm.apply_(txt_k)

        qklist = [img_q, img_k]
        del img_q, img_k
        img_q, img_k = apply_rotary_emb(qklist, freqs_cis, head_first=False)
        img_q_len=img_q.shape[1]
        q = torch.cat((img_q, txt_q), dim=1)
        del img_q, txt_q
        k = torch.cat((img_k, txt_k), dim=1)
        img_kv_len=img_k.shape[1]
        del img_k, txt_k
        
        v = torch.cat((img_v, txt_v), dim=1)
        del img_v, txt_v

        # attention computation start
        qkv_list = [q,k,v]
        del q, k, v
        attn = pay_attention(
            qkv_list,
            attention_mask=attn_mask,                
            q_lens = seqlens_q,
            k_lens = seqlens_kv,
        )
        b, s, a, d = attn.shape
        attn = attn.reshape(b, s, -1)        
        del qkv_list
        # attention computation end
      
        x_mod =  torch.cat((img_mod, txt_mod), 1)
        del img_mod, txt_mod
        x_mod_shape = x_mod.shape
        x_mod = x_mod.view(-1, x_mod.shape[-1])
        chunk_size = int(x_mod.shape[0]/6)
        x_chunks = torch.split(x_mod, chunk_size)
        attn = attn.view(-1, attn.shape[-1])
        attn_chunks =torch.split(attn, chunk_size)
        for x_chunk, attn_chunk in zip(x_chunks, attn_chunks):
            mlp_chunk = self.linear1_mlp(x_chunk)
            mlp_chunk = self.mlp_act(mlp_chunk)
            attn_mlp_chunk = torch.cat((attn_chunk, mlp_chunk), -1)
            del attn_chunk, mlp_chunk 
            x_chunk[...] = self.linear2(attn_mlp_chunk)
            del attn_mlp_chunk
        x_mod = x_mod.view(x_mod_shape)

        if condition_type == "token_replace":
            apply_gate_and_accumulate_(img[:, :frist_frame_token_num, :], x_mod[:, :frist_frame_token_num, :], gate=tr_mod_gate)
            apply_gate_and_accumulate_(img[:, frist_frame_token_num:, :], x_mod[:, frist_frame_token_num:-txt_len, :], gate=mod_gate)
        else:
            apply_gate_and_accumulate_(img, x_mod[:, :-txt_len, :], gate=mod_gate)

        apply_gate_and_accumulate_(txt, x_mod[:, -txt_len:, :], gate=mod_gate)

        return img, txt

class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    def preprocess_loras(self, model_type, sd):
        if model_type != "hunyuan_i2v" :
            return sd
        new_sd = {}
        for k,v in sd.items():
            repl_list = ["double_blocks", "single_blocks", "final_layer", "img_mlp", "img_attn_qkv", "img_attn_proj","img_mod", "txt_mlp", "txt_attn_qkv","txt_attn_proj", "txt_mod", "linear1", 
                        "linear2", "modulation",  "mlp_fc1"]
            src_list = [k +"_" for k in repl_list] +  ["_" + k for k in repl_list]
            tgt_list = [k +"." for k in repl_list] +  ["." + k for k in repl_list]
            if k.startswith("Hunyuan_video_I2V_lora_"):
                # crappy conversion script for non reversible lora naming  
                k = k.replace("Hunyuan_video_I2V_lora_","diffusion_model.")
                k = k.replace("lora_up","lora_B")
                k = k.replace("lora_down","lora_A")
                if "txt_in_individual" in k:
                    pass
                for s,t in zip(src_list, tgt_list):
                    k = k.replace(s,t)
                if  "individual_token_refiner" in k:
                    k = k.replace("txt_in_individual_token_refiner_blocks_", "txt_in.individual_token_refiner.blocks.")
                    k = k.replace("_mlp_fc", ".mlp.fc",)
                    k = k.replace(".mlp_fc", ".mlp.fc",)
            new_sd[k] = v
        return new_sd    
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        i2v_condition_type,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attention_mode: Optional[str] = "sdpa",
        video_condition: bool = False,
        audio_condition: bool = False,
        avatar = False,
        custom = False,
        vision_projection = False,
        glyph_byT5_v2 = False,
        use_meanflow = False,
        use_cond_type_embedding = False,
        text_pool_type = True,
        vision_states_dim: int = 1280,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        pre_split_qkv = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # mm_double_blocks_depth , mm_single_blocks_depth = 5, 5 

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.i2v_condition_type = i2v_condition_type
        self.attention_mode = attention_mode
        self.video_condition = video_condition
        self.audio_condition = audio_condition
        self.avatar = avatar
        self.custom = custom

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2
        self.vision_states_dim = vision_states_dim
        self.text_pool_type = text_pool_type

        self.glyph_byT5_v2 = glyph_byT5_v2
        if self.glyph_byT5_v2:
            self.byt5_in = ByT5Mapper(
                in_dim=1472,
                out_dim=2048,
                hidden_dim=2048,
                out_dim1=hidden_size,
                use_residual=False
            )
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # Vision projection
        if vision_projection == "linear":
            self.vision_in = VisionProjection(
                input_dim=self.vision_states_dim, output_dim=self.hidden_size
            )
        else:
            self.vision_in = None

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = (
            MLPEmbedder(
                self.config.text_states_dim_2, self.hidden_size, **factory_kwargs
            ) if self.text_pool_type is not None else None
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        self.time_r_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
            if use_meanflow
            else None
        )        

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    attention_mode = attention_mode,
                    pre_split_qkv = pre_split_qkv,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    attention_mode = attention_mode,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        if self.video_condition:
            self.bg_in = PatchEmbed(
                self.patch_size, self.in_channels * 2, self.hidden_size, **factory_kwargs
            )
            self.bg_proj = nn.Linear(self.hidden_size, self.hidden_size)

        if audio_condition:
            if avatar:
                self.ref_in = PatchEmbed(
                    self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
                    )

                # -------------------- audio_proj_model --------------------
                self.audio_proj = AudioProjNet2(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=3072, context_tokens=4)
                
                # -------------------- motion-embeder --------------------
                self.motion_exp = TimestepEmbedder(
                        self.hidden_size // 4,
                        get_activation_layer("silu"),
                        **factory_kwargs
                    )
                self.motion_pose = TimestepEmbedder(
                        self.hidden_size // 4,
                        get_activation_layer("silu"),
                        **factory_kwargs
                    )

                self.fps_proj = TimestepEmbedder(
                        self.hidden_size,
                        get_activation_layer("silu"),
                        **factory_kwargs
                    )
                
                self.before_proj = nn.Linear(self.hidden_size, self.hidden_size)

                # -------------------- audio_insert_model --------------------
                self.double_stream_list = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
                audio_block_name = "audio_adapter_blocks"
            elif custom:
                self.audio_proj = AudioProjNet2(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=3072, context_tokens=4)
                self.double_stream_list = [1, 3, 5, 7, 9, 11]
                audio_block_name = "audio_models"

            self.double_stream_map = {str(i): j for j, i in enumerate(self.double_stream_list)}
            self.single_stream_list = []
            self.single_stream_map = {str(i): j+len(self.double_stream_list) for j, i in enumerate(self.single_stream_list)}
            setattr(self, audio_block_name,  nn.ModuleList([
                PerceiverAttentionCA(dim=3072, dim_head=1024, heads=33) for _ in range(len(self.double_stream_list) + len(self.single_stream_list))
            ]))

        if use_cond_type_embedding:
            self.cond_type_embedding = nn.Embedding(3, self.hidden_size) 
            self.cond_type_embedding.weight.data.fill_(0)
            assert self.glyph_byT5_v2, "text type embedding is only used when glyph_byT5_v2 is True"
            assert vision_projection is not None, "text type embedding is only used when vision_projection is not None"
            # 0: text_encoder feature
            # 1: byt5 feature
            # 2: vision_encoder feature
        else:
            self.cond_type_embedding = None



    def lock_layers_dtypes(self, dtype = torch.float32):
        layer_list = [self.final_layer, self.final_layer.linear, self.final_layer.adaLN_modulation[1]]
        for layer in layer_list:
            layer._lock_dtype = dtype
        # These hold no weights themselves: their FP32 locks keep MMGP from rounding the float32 timesteps / guidance to BF16 before their sinusoids
        for module in (self, self.time_in, self.txt_in, self.guidance_in, self.time_r_in):
            if module is not None:
                module._lock_dtype = torch.float32

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def compute_magcache_threshold(self, start_step, num_inference_steps = 0, speed_factor =0):
        skips_step_cache = self.cache

        def nearest_interp(src_array, target_length):
            src_length = len(src_array)
            if target_length == 1:
                return np.array([src_array[-1]])
            scale = (src_length - 1) / (target_length - 1)
            mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
            return src_array[mapped_indices]
        def_mag_ratios = np.array([1.0]+ skips_step_cache.def_mag_ratios)
        if len(def_mag_ratios) != num_inference_steps:
            skips_step_cache.mag_ratios = nearest_interp(def_mag_ratios, num_inference_steps)
        else:
            skips_step_cache.mag_ratios = def_mag_ratios

        best_deltas = None
        best_threshold = 0.01
        best_diff = 1000
        best_signed_diff = 1000
        target_nb_steps= int(num_inference_steps / speed_factor)
        threshold = 0.01
        while threshold <= 0.6:
            nb_steps = 0
            diff = 1000
            accumulated_err, accumulated_steps, accumulated_ratio = 0, 0, 1.0
            for i in range(num_inference_steps):
                if i<=start_step:
                    skip  = False
                else:
                    cur_mag_ratio = skips_step_cache.mag_ratios[i] # conditional and unconditional in one list
                    accumulated_ratio *= cur_mag_ratio # magnitude ratio between current step and the cached step
                    accumulated_steps += 1 # skip steps plus 1
                    cur_skip_err = np.abs(1-accumulated_ratio) # skip error of current steps
                    accumulated_err += cur_skip_err # accumulated error of multiple steps
                    if accumulated_err<threshold and accumulated_steps<=skips_step_cache.magcache_K:
                        skip  = True
                    else:
                        skip  = False
                        accumulated_err, accumulated_steps, accumulated_ratio = 0, 0, 1.0
                if not skip:
                    nb_steps += 1
                    signed_diff = target_nb_steps - nb_steps               
                    diff = abs(signed_diff)  
            if diff < best_diff:
                best_threshold = threshold
                best_diff = diff
                best_signed_diff = signed_diff
            elif diff > best_diff:
                break
            threshold += 0.01
        skips_step_cache.magcache_thresh = best_threshold
        print(f"Mag Cache, best threshold found:{best_threshold:0.2f} with gain x{num_inference_steps/(target_nb_steps - best_signed_diff):0.2f} for a target of x{speed_factor}")
        return best_threshold

    def reorder_txt_token(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
        if is_reorder:
            reorder_txt = []
            reorder_mask = []
            for i in range(text_mask.shape[0]):
                byt5_text_mask_i = byt5_text_mask[i].bool()
                text_mask_i = text_mask[i].bool()

                byt5_txt_i = byt5_txt[i]
                txt_i = txt[i]
                if zero_feat:
                    # When using block mask with approximate computation, set pad to zero to reduce error
                    pad_byt5 = torch.zeros_like(byt5_txt_i[~byt5_text_mask_i])
                    pad_text = torch.zeros_like(txt_i[~text_mask_i])
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], pad_byt5, pad_text], dim=0
                    )
                else:
                    reorder_txt_i = torch.cat(
                        [byt5_txt_i[byt5_text_mask_i], txt_i[text_mask_i], byt5_txt_i[~byt5_text_mask_i], txt_i[~text_mask_i]], dim=0
                    )
                reorder_mask_i = torch.cat(
                    [byt5_text_mask_i[byt5_text_mask_i], text_mask_i[text_mask_i], byt5_text_mask_i[~byt5_text_mask_i], text_mask_i[~text_mask_i]], dim=0
                )

                reorder_txt.append(reorder_txt_i)
                reorder_mask.append(reorder_mask_i)

            reorder_txt = torch.stack(reorder_txt)
            reorder_mask = torch.stack(reorder_mask).to(dtype=torch.int64)
        else:
            reorder_txt = torch.concat([byt5_txt, txt], dim=1)
            reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int64)

        return reorder_txt, reorder_mask
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        ref_latents: torch.Tensor=None,        
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        pipeline=None,
        x_id = 0,
        step_no = 0,
        callback = None,
        audio_prompts = None,
        motion_exp = None,
        motion_pose = None,
        fps = None,
        face_mask = None,
        audio_strength = None,
        bg_latents = None,
        vision_states: torch.Tensor = None,
        byt5_text_states=None,
        byt5_text_mask=None,
        timesteps_r=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    
        img = x
        bsz, _, ot, oh, ow = x.shape
        del x
        txt = text_states   
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)
        if motion_exp != None:
            vec += self.motion_exp(motion_exp.view(-1)).view(bsz, -1)     # (b, 3072)
        if motion_pose != None:
            vec += self.motion_pose(motion_pose.view(-1)).view(bsz, -1)  # (b, 3072)
        if fps != None:
            vec += self.fps_proj(fps)   # (b, 3072)
        if audio_prompts != None:
            audio_feature_all = self.audio_proj(audio_prompts)
            audio_feature_pad = audio_feature_all[:,:1].repeat(1,3,1,1) 
            audio_feature_all_insert = torch.cat([audio_feature_pad, audio_feature_all], dim=1).view(bsz, ot, 16, 3072)
            audio_feature_all = None

        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None
            # token_replace_mask_img = None
            # token_replace_mask_txt = None

        # text modulation
        if text_states_2 is not None:
            vec_2 = self.vector_in(text_states_2)
            del text_states_2
            vec += vec_2
            if self.i2v_condition_type == "token_replace":
                token_replace_vec += vec_2
            del vec_2
        
        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec += self.guidance_in(guidance)

        if timesteps_r is not None:
            vec += self.time_r_in(timesteps_r)

        # Embed image and text.
        img, shape_mask = self.img_in(img)
        if self.avatar:
            ref_latents_first = ref_latents[:, :, :1].clone()
            ref_latents,_ = self.ref_in(ref_latents)
            ref_latents_first,_ = self.img_in(ref_latents_first)
        elif self.custom:
            if ref_latents != None:
                ref_latents, _ = self.img_in(ref_latents)
            if bg_latents is not None and self.video_condition:
                bg_latents, _ = self.bg_in(bg_latents)
                img += self.bg_proj(bg_latents)

        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        if self.cond_type_embedding is not None:
            cond_emb = self.cond_type_embedding( torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long) )
            txt += cond_emb

        if self.glyph_byT5_v2:
            byt5_txt = self.byt5_in(byt5_text_states)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding( torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long) )
                byt5_txt += cond_emb
            txt, text_mask = self.reorder_txt_token( byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=True )

        if self.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = self.vision_in(vision_states).repeat(bsz,1, 1)
            extra_attention_mask = torch.ones( (bsz, extra_encoder_hidden_states.shape[1]), dtype=text_mask.dtype, device=text_mask.device, )
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(
                    2 * torch.ones_like( extra_encoder_hidden_states[:, :, 0], dtype=torch.long, device=extra_encoder_hidden_states.device, )
                )
                extra_encoder_hidden_states += cond_emb
            txt, text_mask = self.reorder_txt_token( extra_encoder_hidden_states, txt, extra_attention_mask, text_mask )

        if self.avatar:
            img += self.before_proj(ref_latents)
            ref_length = ref_latents_first.shape[-2]          # [b s c]
            img = torch.cat([ref_latents_first, img], dim=-2) # t c
            img_len = img.shape[1]
            mask_len = img_len - ref_length
            if face_mask.shape[2] == 1:
                face_mask = face_mask.repeat(1,1,ot,1,1)  # repeat if number of mask frame is 1
            face_mask = torch.nn.functional.interpolate(face_mask, size=[ot, shape_mask[-2], shape_mask[-1]], mode="nearest")
            # face_mask = face_mask.view(-1,mask_len,1).repeat(1,1,img.shape[-1]).type_as(img)
            face_mask = face_mask.view(-1,mask_len,1).type_as(img)
        elif ref_latents == None:
            ref_length  = None
        else:
            ref_length = ref_latents.shape[-2]
            img = torch.cat([ref_latents, img], dim=-2) # t c
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        text_len = text_mask.sum(1)
        total_len = text_len + img_seq_len
        seqlens_q = seqlens_kv = total_len 
        attn_mask = None

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        should_calc = True
        skip_steps_cache  = self.cache
        if skip_steps_cache is not None:
            cache_type = skip_steps_cache.cache_type
            if x_id == 0:
                skip_steps_cache.should_calc = True
                if cache_type == "mag":
                    if step_no > skip_steps_cache.start_step:
                        cur_mag_ratio = skip_steps_cache.mag_ratios[step_no]
                        skip_steps_cache.accumulated_ratio = skip_steps_cache.accumulated_ratio*cur_mag_ratio
                        cur_skip_err = np.abs(1-skip_steps_cache.accumulated_ratio)
                        skip_steps_cache.accumulated_err += cur_skip_err
                        skip_steps_cache.accumulated_steps += 1
                        if skip_steps_cache.accumulated_err<=skip_steps_cache.magcache_thresh and skip_steps_cache.accumulated_steps<=skip_steps_cache.magcache_K:
                            skip_steps_cache.should_calc = False
                            skip_steps_cache.skipped_steps += 1
                        else:
                            skip_steps_cache.accumulated_ratio, skip_steps_cache.accumulated_steps, skip_steps_cache.accumulated_err = 1.0, 0, 0
                else:
                    inp = img[0:1] 
                    vec_ = vec[0:1] 
                    ( img_mod1_shift, img_mod1_scale, _ , _ , _ , _ , ) = self.double_blocks[0].img_mod(vec_).chunk(6, dim=-1)
                    normed_inp = self.double_blocks[0].img_norm1(inp)
                    normed_inp = normed_inp.to(torch.bfloat16)
                    modulated_inp = modulate( normed_inp, shift=img_mod1_shift, scale=img_mod1_scale )
                    del normed_inp, img_mod1_shift, img_mod1_scale
                    if step_no <= skip_steps_cache.start_step or step_no == skip_steps_cache.num_steps-1:
                        skip_steps_cache.accumulated_rel_l1_distance = 0
                    else:                         
                        rescale_func = np.poly1d(skip_steps_cache.coefficients)
                        skip_steps_cache.accumulated_rel_l1_distance += rescale_func(((modulated_inp-skip_steps_cache.previous_modulated_input).abs().mean() / skip_steps_cache.previous_modulated_input.abs().mean()).cpu().item())
                        if skip_steps_cache.accumulated_rel_l1_distance < skip_steps_cache.rel_l1_thresh:
                            skip_steps_cache.should_calc = False
                            skip_steps_cache.skipped_steps += 1
                        else:
                            skip_steps_cache.accumulated_rel_l1_distance = 0
                    skip_steps_cache.previous_modulated_input = modulated_inp  
            should_calc = skip_steps_cache.should_calc

        if not should_calc:
            img += skip_steps_cache.previous_residual[x_id]
        else:
            if skip_steps_cache is not None:            
                skip_steps_cache.previous_residual[x_id] = None
                ori_img = img[0:1].clone()
            # --------------------- Pass through DiT blocks ------------------------
            for layer_num, block in enumerate(self.double_blocks):
                for i in range(len(img)):
                    if callback != None:
                        callback(-1, None, False, True)
                    if pipeline._interrupt:
                        return None
                    double_block_args = [
                        img[i:i+1],
                        txt[i:i+1],
                        vec[i:i+1],
                        attn_mask,                
                        seqlens_q[i:i+1],
                        seqlens_kv[i:i+1],
                        freqs_cis,
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,                    
                    ]

                    img[i], txt[i] = block(*double_block_args)
                    double_block_args = None
                    # insert audio feature to img 
                    if audio_prompts != None: 
                        audio_adapter = getattr(self.double_blocks[layer_num], "audio_adapter", None)
                        if audio_adapter != None:
                            real_img = img[i:i+1,ref_length:].view(1, ot, -1, 3072)  
                            real_img = audio_adapter(audio_feature_all_insert[i:i+1], real_img).view(1, -1, 3072)
                            if face_mask != None:
                                real_img *= face_mask[i:i+1]
                            if audio_strength != None and audio_strength != 1:
                                real_img *= audio_strength
                            img[i:i+1, ref_length:] += real_img
                            real_img = None


            for _, block in enumerate(self.single_blocks):
                for i in range(len(img)):
                    if callback != None:
                        callback(-1, None, False, True)
                    if pipeline._interrupt:
                        return None
                    single_block_args = [
                        # x,
                        img[i:i+1],
                        txt[i:i+1],
                        vec[i:i+1],
                        txt_seq_len,
                        attn_mask,                
                        seqlens_q[i:i+1],
                        seqlens_kv[i:i+1],
                        (freqs_cos, freqs_sin),
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,                    
                    ]

                    img[i], txt[i] = block(*single_block_args)
                    single_block_args = None

            # img = x[:, :img_seq_len, ...]
            if skip_steps_cache is not None:
                if len(img) > 1:
                    skip_steps_cache.previous_residual[0] = torch.empty_like(img)
                    for i, (x, residual) in enumerate(zip(img, skip_steps_cache.previous_residual[0])):
                        if i < len(img) - 1:
                            residual[...] = torch.sub(x, ori_img) 
                        else:
                            residual[...] = ori_img
                            torch.sub(x, ori_img, out=residual)                     
                    x = None
                else:
                    skip_steps_cache.previous_residual[x_id] = ori_img
                    torch.sub(img, ori_img, out=skip_steps_cache.previous_residual[x_id]) 


        if ref_length != None:
            img = img[:, ref_length:]
        # ---------------------------- Final layer ------------------------------
        out_dtype = self.final_layer.linear.weight.dtype
        vec = vec.to(out_dtype)        
        img_list  = []
        for img_chunk, vec_chunk in zip(img,vec):
             img_list.append( self.final_layer(img_chunk.to(out_dtype).unsqueeze(0), vec_chunk.unsqueeze(0))) # (N, T, patch_size ** 2 * out_channels) 
        img = torch.cat(img_list)
        img_list = None

        # img = self.unpatchify(img, tt, th, tw)
        img = self.unpatchify(img, tt, th, tw)

        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts       


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
    "HYVideo-S/2": {
        "mm_double_blocks_depth": 6,
        "mm_single_blocks_depth": 12,
        "rope_dim_list": [12, 42, 42],
        "hidden_size": 480,
        "heads_num": 5,
        "mlp_width_ratio": 4,
    },
    'HYVideo-T/2-custom': {                                                                       #   9.0B   / 12.5B
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        'custom' : True
    },
    'HYVideo-T/2-custom-audio': {                                                                       #   9.0B   / 12.5B
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        'custom' : True,
        'audio_condition' : True,        
    },
    'HYVideo-T/2-custom-edit': {                                                                       #   9.0B   / 12.5B
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        'custom' : True,
        'video_condition' : True,        
    },
    'HYVideo-T/2-avatar': {                                                                       #   9.0B   / 12.5B
        'mm_double_blocks_depth': 20,
        'mm_single_blocks_depth': 40,
        'rope_dim_list': [16, 56, 56],
        'hidden_size': 3072,
        'heads_num': 24,
        'mlp_width_ratio': 4,
        'avatar': True,
        'audio_condition' : True,
    },
    'HYVideo-1_5': {
        "mm_double_blocks_depth": 54,
        "mm_single_blocks_depth": 0,
        "heads_num": 16,
        "hidden_size": 2048,
        'rope_dim_list': [16, 56, 56],
        'mlp_width_ratio': 4,
        "text_pool_type": None,
        "text_states_dim": 3584,
        "text_states_dim_2": None,
        "use_attention_mask": True,
        "use_cond_type_embedding": True,
        "use_meanflow": False,
        "vision_projection": "linear",
        "vision_states_dim": 1152,
        "glyph_byT5_v2": True,
        "patch_size": [1, 1, 1],
        "pre_split_qkv": True,
    },    
    'HYVideo-1_5-upsampler': {
        "mm_double_blocks_depth": 54,
        "mm_single_blocks_depth": 0,
        "heads_num": 16,
        "hidden_size": 2048,
        'rope_dim_list': [16, 56, 56],
        'mlp_width_ratio': 4,
        "text_pool_type": None,
        "text_states_dim": 3584,
        "text_states_dim_2": None,
        "use_attention_mask": True,
        "use_cond_type_embedding": True,
        "use_meanflow": True,
        "vision_projection": "linear",
        "vision_states_dim": 1152,
        "glyph_byT5_v2": True,
        "patch_size": [1, 1, 1],
        "pre_split_qkv": True,
    },    
}
