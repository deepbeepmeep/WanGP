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

from ..attention import MultiHeadCrossAttention
from ..blocks import TimestepEmbedder, CaptionEmbedder, PatchEmbed3D, FeedForwardSwiGLU, FinalLayer_FP32, LayerNorm_FP32, modulate_fp32, _take_tensor

from .attention import Attention, SingleStreamAttention
from .blocks import AudioProjModel


class LongCatAvatarSingleStreamBlock(nn.Module):
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
        cp_split_hw=None,
        # avatar config
        output_dim=768,
        audio_prenorm=True,
        class_range=24,
        class_interval=4
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)
        )
        self.audio_adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True)
        )

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn  = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.pre_audio_crs_attn_norm = LayerNorm_FP32(output_dim, eps=1e-6, elementwise_affine=True) if audio_prenorm else nn.Identity()
        
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
            enable_xformers=enable_xformers
        )

        self.audio_cross_attn = SingleStreamAttention(
                dim=hidden_size,
                encoder_hidden_states_dim=output_dim,
                num_heads=num_heads,
                qk_norm=True,
                qkv_bias=True,
                class_range=class_range,
                class_interval=class_interval,
                cp_split_hw=cp_split_hw,
                enable_flashattn3=enable_flashattn3,
                enable_flashattn2=enable_flashattn2,
                enable_xformers=enable_xformers
            )

        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))
        self.ffn_mult = self.ffn.ffn_mult
        self.ffn_chunk_min = 128

    def _apply_ffn_chunked(self, x_list):
        return ops.mlp(x_list, self.ffn.w1, self.ffn.w2, "swiglu", up=self.ffn.w3)

    def forward(
        self, 
        x, 
        y, 
        t, 
        y_seqlen, 
        latent_shape, 
        num_cond_latents=None, 
        return_kv=False, 
        kv_cache=None, 
        skip_crs_attn=False,
        # avatar related params
        num_ref_latents=None,
        audio_hidden_states=None, 
        ref_img_index=None,
        mask_frame_range=None,
        token_ref_target_masks=None,
        human_num=None,
        rope=None,
    ):
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

        x_s, x_ref_attn_map = self.attn(
            x_m_list, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv,
            layer_cache=kv_cache, rope=rope, num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index, mask_frame_range=mask_frame_range,
            ref_target_masks=token_ref_target_masks)
        x_m = None

        x.view(B, -1, N//T, C).addcmul_(x_s.view(B, -1, N//T, C), gate_msa)
        del x_s, gate_msa, shift_msa, scale_msa
        x = x.to(x_dtype)

        # text cross attn
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
        
        # audio cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = 0

            with amp.autocast(device_type='cuda', dtype=torch.float32):  
                audio_shift_mca, audio_scale_mca, audio_gate_mca = \
                        self.audio_adaLN_modulation(t[:, num_cond_latents:]).unsqueeze(2).chunk(3, dim=-1) # [B, T, 1, C]

            video_cross = self.pre_video_crs_attn_norm(x)
            audio_cross = self.pre_audio_crs_attn_norm(audio_hidden_states)
            video_cross_list = [video_cross]
            audio_cross_list = [audio_cross]
            video_cross = audio_cross = None
            audio_cond_tokens, audio_output_noise = self.audio_cross_attn(video_cross_list, audio_cross_list, shape=latent_shape, num_cond_latents=num_cond_latents, x_ref_attn_map=x_ref_attn_map, human_num=human_num, speaker_token_masks=token_ref_target_masks)

            audio_frames = T - num_cond_latents
            audio_output_list = [audio_output_noise.view(B, audio_frames, -1, C)]
            audio_output_noise = None
            audio_output_noise = modulate_fp32(self.mod_norm_attn, audio_output_list, audio_shift_mca, audio_scale_mca).view(B, -1, C)
            if audio_cond_tokens:
                x[:, audio_cond_tokens:].view(B, audio_frames, -1, C).addcmul_(audio_output_noise.view(B, audio_frames, -1, C), audio_gate_mca)
            else:
                x.view(B, audio_frames, -1, C).addcmul_(audio_output_noise.view(B, audio_frames, -1, C), audio_gate_mca)
            del audio_output_noise, audio_gate_mca, audio_shift_mca, audio_scale_mca
            x = x.to(x_dtype)

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N//T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_m_list = [x_m]
        x_m = None
        x_s = self._apply_ffn_chunked(x_m_list)
        x.view(B, -1, N//T, C).addcmul_(x_s.view(B, -1, N//T, C), gate_mlp)
        del x_s, gate_mlp, shift_mlp, scale_mlp
        x = x.to(x_dtype)

        return x


class LongCatVideoAvatarTransformer3DModel(
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
        # avatar config
        audio_window: int = 5,
        audio_block: int = 12,
        audio_channel: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4, 
        audio_prenorm: bool = False,
        class_range: int = 24,
        class_interval: int = 4
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw
        self.vae_scale = vae_scale
        self.audio_window = audio_window

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatAvatarSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=cp_split_hw,
                    output_dim=output_dim,
                    audio_prenorm=audio_prenorm,
                    class_range=class_range,
                    class_interval=class_interval
                )
                for i in range(depth)
            ]
        )

        self.audio_proj = AudioProjModel(
                    seq_len=audio_window,
                    seq_len_vf=audio_window+vae_scale-1,
                    blocks=audio_block,
                    channels=audio_channel,
                    intermediate_dim=intermediate_dim,
                    output_dim=output_dim,
                    context_tokens=context_tokens
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

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(self._decode_upstream_lora(state_dict), dict(self.named_modules()),
                                 split_linear_modules_map=getattr(self, "split_linear_modules_map", None))

    def _decode_upstream_lora(self, state_dict):
        if not any(key.startswith("lora___lorahyphen___") for key in state_dict):
            return state_dict

        def decode_module_name(prefix):
            return prefix.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")

        def block_index(key):
            return int(key.split(".lora_up.blocks.", 1)[1].split(".", 1)[0])

        converted = {}
        prefixes = sorted(key.rsplit(".lora_down.weight", 1)[0] for key in state_dict if key.endswith(".lora_down.weight"))
        for prefix in prefixes:
            module_name = decode_module_name(prefix)
            down = state_dict.pop(f"{prefix}.lora_down.weight")
            converted[f"{module_name}.lora_down.weight"] = down

            alpha_scale = state_dict.pop(f"{prefix}.alpha_scale", None)
            if alpha_scale is not None:
                alpha_scale = float(alpha_scale.item()) if torch.is_tensor(alpha_scale) else float(alpha_scale)
                converted[f"{module_name}.alpha"] = torch.tensor(alpha_scale * down.shape[0], dtype=torch.float32)

            up_key = f"{prefix}.lora_up.weight"
            if up_key in state_dict:
                converted[f"{module_name}.lora_up.weight"] = state_dict.pop(up_key)
                continue

            up_block_keys = sorted(
                [key for key in state_dict if key.startswith(f"{prefix}.lora_up.blocks.") and key.endswith(".weight")],
                key=block_index,
            )
            if not up_block_keys:
                raise ValueError(f"Missing LoRA up weights for {module_name}.")
            total_rank = sum(state_dict[key].shape[1] for key in up_block_keys)
            if down.shape[0] != total_rank:
                raise ValueError(f"LoRA rank mismatch for {module_name}: down={down.shape[0]} up={total_rank}")
            up = down.new_zeros((sum(state_dict[key].shape[0] for key in up_block_keys), down.shape[0]))
            row_start = 0
            col_start = 0
            for key in up_block_keys:
                block = state_dict.pop(key)
                rows, rank = block.shape
                up[row_start : row_start + rows, col_start : col_start + rank] = block if block.dtype == up.dtype else block.to(up.dtype)
                row_start += rows
                col_start += rank
            converted[f"{module_name}.lora_up.weight"] = up

        for key, value in list(state_dict.items()):
            if key.startswith("lora___lorahyphen___"):
                raise ValueError(f"Unsupported LongCat LoRA key: {key}")
            converted[key] = value
        state_dict.clear()
        return converted
    
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
        # avatar related params
        audio_embs=None,
        num_ref_latents=None,
        ref_img_index=None, 
        mask_frame_range=None,
        ref_target_masks=None
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
        if not isinstance(audio_embs, list):
            audio_embs = [audio_embs] * len(x_list)
        if not isinstance(ref_target_masks, list):
            ref_target_masks = [ref_target_masks] * len(x_list)

        dtype = self.x_embedder.proj.weight.dtype
        t_list = []
        enc_list = []
        y_seqlens_list = []
        latent_shapes = []
        rope_tables = []
        audio_hidden_states_list = []
        token_ref_target_masks_list = []
        human_num_list = []

        for idx, (x, step, enc, mask, audio_cond, ref_masks) in enumerate(
            zip(x_list, timestep, encoder_hidden_states, encoder_attention_mask, audio_embs, ref_target_masks)
        ):
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

            # Eager beside the graphed blocks, on the graph pool's idle memory (with the head below): its temporaries do not keep ordinary
            # memory beside the pool, and its output moves into the static buffer the first block reads. Not graphed: its weights may be
            # streamed, and a replay does not load them
            with cuda_graphs.scope(not return_kv):
                x = cuda_graphs.pooled(self, self.x_embedder, x)

            with amp.autocast(device_type='cuda', dtype=torch.float32):
                t = self.t_embedder(step.float().flatten(), torch.float32).reshape(B, N_t, -1)
            t = t.to(torch.float32)

            enc = self.y_embedder(enc)

            # audio tokens
            audio_hidden_states = None
            human_num = None
            token_ref_target_masks = None
            if audio_cond is not None:
                audio_cond = audio_cond.to(device=x.device, dtype=x.dtype)
                first_frame_audio_emb_s = audio_cond[:, :1, ...]

                latter_frame_audio_emb = audio_cond[:, 1:, ...]
                latter_frame_audio_emb = rearrange(
                    latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=self.vae_scale
                )
                middle_index = self.audio_window // 2
                latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index + 1, ...]
                latter_first_frame_audio_emb = rearrange(
                    latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
                )

                latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
                latter_last_frame_audio_emb = rearrange(
                    latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
                )

                latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index + 1, ...]
                latter_middle_frame_audio_emb = rearrange(
                    latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c"
                )
                latter_frame_audio_emb_s = torch.concat(
                    [latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2
                )
                audio_hidden_states = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)

                if num_ref_latents is not None and num_ref_latents > 0:
                    audio_start_ref = audio_hidden_states[:, [0], :, :]
                    audio_hidden_states = torch.cat([audio_start_ref, audio_hidden_states], dim=1).contiguous()
                audio_hidden_states = audio_hidden_states[:, -N_t:]

                if ref_masks is not None:
                    human_num = len(audio_hidden_states)
                    audio_hidden_states = torch.concat(audio_hidden_states.split(1), dim=2)
                    audio_hidden_states = audio_hidden_states.squeeze(0)
                else:
                    audio_hidden_states = rearrange(audio_hidden_states, "b t n c -> (b t) n c")

            if ref_masks is not None:
                ref_masks = ref_masks.unsqueeze(0).to(torch.float32)
                token_ref_target_masks = nn.functional.interpolate(ref_masks, size=(N_h, N_w), mode='nearest')
                token_ref_target_masks = token_ref_target_masks.squeeze(0)
                token_ref_target_masks = token_ref_target_masks > 0
                token_ref_target_masks = token_ref_target_masks.view(token_ref_target_masks.shape[0], -1).to(dtype)

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
            rope_key = (total_t, N_h, N_w, ref_img_index, num_ref_latents, x.device)
            if rope_key not in self._rope_cache:
                freqs = self.blocks[0].attn.rope_3d.precompute_freqs_cis_3d((total_t, N_h, N_w), ref_img_index, num_ref_latents).to(x.device)
                self._rope_cache[rope_key] = torch.stack((freqs.cos(), freqs.sin()), dim=-1)[None, :, None]
            rope_tables.append(self._rope_cache[rope_key])
            audio_hidden_states_list.append(audio_hidden_states)
            token_ref_target_masks_list.append(token_ref_target_masks)
            human_num_list.append(human_num)

        with cuda_graphs.scope(not return_kv):
            for block_idx, block in enumerate(self.blocks):
                if self._interrupt_check is not None and self._interrupt_check():
                    return [None] * len(x_list) if joint_pass else None
                for i in range(len(x_list)):
                    x_list[i] = block(
                        x_list[i], enc_list[i], t_list[i], y_seqlens_list[i], latent_shapes[i],
                        num_cond_latents[i], return_kv,
                        kv_cache_dict[i][block_idx] if kv_cache_dict[i] is not None else None,
                        skip_crs_attn, num_ref_latents, audio_hidden_states_list[i], ref_img_index,
                        mask_frame_range, token_ref_target_masks_list[i], human_num_list[i], rope_tables[i])
                    if self._interrupt_check is not None and self._interrupt_check():
                        return [None] * len(x_list) if joint_pass else None
        if return_kv:
            return None  # Only the PrefixCache is kept; no final projection of clean latents.

        outputs = []
        for x, t, latent_shape in zip(x_list, t_list, latent_shapes):
            with cuda_graphs.scope():  # its FP32 temporaries (two copies of the hidden state) on the graph pool's idle memory
                x = cuda_graphs.pooled(self, self.final_layer, x, t, latent_shape)
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
