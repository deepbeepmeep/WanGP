from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from shared.kernels import ops

from .configuration_neo_vit import NEOVisionConfig


def _take_tensor(tensor_list):
    tensor = tensor_list[0]
    tensor_list.clear()
    return tensor


def precompute_rope_freqs_sincos(
    dim: int, max_position: int, base: float = 10000.0, device=None
):
    """预计算 RoPE 的 cos 和 sin 值 (1D)。"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_position, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def build_abs_positions_from_grid_hw(grid_hw: torch.Tensor, device=None):
    """
    Compute patch coordinates (x, y)

    Args:
        grid_hw: (B, 2) tensor representing (H, W) per image
    """
    device = grid_hw.device
    B = grid_hw.shape[0]

    # Get the number of patches per image
    H = grid_hw[:, 0]
    W = grid_hw[:, 1]
    N = H * W
    N_total = N.sum()

    # Create the batch index for each patch (B x patch count)
    patch_to_sample = torch.repeat_interleave(torch.arange(B, device=device), N)  # (N_total,)

    # Generate intra-image patch index (row-major order)
    patch_id_within_image = torch.arange(N_total, device=device)
    patch_id_within_image = patch_id_within_image - torch.cumsum(
        torch.cat([torch.tensor([0], device=device), N[:-1]]), dim=0
    )[patch_to_sample]

    # Get H/W for each patch according to its image
    W_per_patch = W[patch_to_sample]
    abs_x = patch_id_within_image % W_per_patch
    abs_y = patch_id_within_image // W_per_patch

    return abs_x, abs_y


def apply_rotary_emb_1d(
    x: torch.Tensor,
    cos_cached: torch.Tensor,
    sin_cached: torch.Tensor,
    positions: torch.Tensor,
    scratch: torch.Tensor,
):
    """对输入张量的一部分应用1D RoPE。"""
    # x: (..., seq_len, dim_part)
    # positions: (..., seq_len)
    # cos_cached: (max_pos, dim_part / 2)
    
    cos = cos_cached[positions] # Shape: (positions.shape, dim_part / 2)
    sin = sin_cached[positions] # Shape: (positions.shape, dim_part / 2)
    
    return ops.paired_rope_(x, cos, sin, scratch, interleaved=True)


def apply_2d_rotary_pos_emb(
    x_list,
    cos_cached_x: torch.Tensor,
    sin_cached_x: torch.Tensor,
    cos_cached_y: torch.Tensor,
    sin_cached_y: torch.Tensor,
    abs_positions_x: torch.Tensor,
    abs_positions_y: torch.Tensor
):
    """应用2D RoPE到输入张量x。"""
    x = _take_tensor(x_list)
    dim = x.shape[-1]
    dim_half = dim // 2

    # 假设我们将embedding的前半部分用于一个方向的RoPE，后半部分用于另一个方向
    # 例如，前一半给X坐标，后一半给Y坐标 (或者反过来，但要保持一致)
    x_part_1 = x[..., :dim_half]
    x_part_2 = x[..., dim_half:]

    # 将与 abs_positions_x 相关的旋转应用于 x_part_1
    scratch = torch.empty_like(x_part_1[..., 0::2])
    apply_rotary_emb_1d(x_part_1, cos_cached_x, sin_cached_x, abs_positions_x, scratch)
    # 将与 abs_positions_y 相关的旋转应用于 x_part_2
    apply_rotary_emb_1d(x_part_2, cos_cached_y, sin_cached_y, abs_positions_y, scratch)
    return x


class NEOVisionEmbeddings(nn.Module):
    """
    Embedding Module for Vision.
    """

    def __init__(self, config: NEOVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.llm_embed_dim = config.llm_hidden_size[0]
        self.downsample_factor = int(1 / config.downsample_ratio[0])
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.dense_embedding = nn.Conv2d(
            in_channels=self.embed_dim, out_channels=self.llm_embed_dim, kernel_size=self.downsample_factor, stride=self.downsample_factor
        )
        self.gelu = nn.GELU()

        self.rope_dim_part = self.embed_dim // 2
        self.max_position_embeddings_vision = config.max_position_embeddings_vision
        self.rope_theta_vision = config.rope_theta_vision

        # These deterministic caches are not checkpoint state.  In
        # Transformers 5, ``from_pretrained`` constructs models on the meta
        # device; tensors computed here would later be materialized as
        # uninitialized memory because persistent=False buffers are absent
        # from the checkpoint.  Build them lazily on the first real device.
        self.register_buffer("cos_cached_x", None, persistent=False)
        self.register_buffer("sin_cached_x", None, persistent=False)
        self.register_buffer("cos_cached_y", None, persistent=False)
        self.register_buffer("sin_cached_y", None, persistent=False)

    cpu_rope_tables = False  # model def `vision_rope_cpu_tables` (U1.6): FP32 tables computed on the CPU like the checkpoint's training code, then moved

    def _ensure_rope_cache(self, device: torch.device) -> None:
        if self.cos_cached_x is not None and self.cos_cached_x.device == device:
            return

        cos, sin = precompute_rope_freqs_sincos(
            self.rope_dim_part,
            self.max_position_embeddings_vision,
            base=self.rope_theta_vision,
            device="cpu" if self.cpu_rope_tables else device,
        )
        cos, sin = cos.to(device), sin.to(device)
        self.cos_cached_x = cos
        self.sin_cached_x = sin
        self.cos_cached_y = cos
        self.sin_cached_y = sin

    def _apply_2d_rotary_pos_emb(self, patch_embeds_list, grid_hw):
        """
        Apply 2D Rotary Position Embedding to the patch embeddings.
        """
        patch_embeds = _take_tensor(patch_embeds_list).to(torch.float32)
        abs_pos_x, abs_pos_y = build_abs_positions_from_grid_hw(grid_hw, device=patch_embeds.device)
        patch_embeds_list = [patch_embeds]
        del patch_embeds
        embeddings = apply_2d_rotary_pos_emb(
            patch_embeds_list,
            self.cos_cached_x, self.sin_cached_x,
            self.cos_cached_y, self.sin_cached_y,
            abs_pos_x,
            abs_pos_y
        ).to(self.patch_embedding.weight.dtype)
        return embeddings
        
    def forward(self, pixel_values_list, grid_hw=None) -> torch.Tensor:
        pixel_values = _take_tensor(pixel_values_list)
        pixel_values = pixel_values.view(  # 
            -1,
            3,
            self.patch_size,
            self.patch_size,
        )   #  [28072, 768] -> [28072, 3, 16, 16]
        patch_embeds = self.gelu(self.patch_embedding(pixel_values)).view(-1, self.embed_dim)
        del pixel_values
        self._ensure_rope_cache(patch_embeds.device)
        patch_embeds_list = [patch_embeds]
        del patch_embeds
        patch_embeds = self._apply_2d_rotary_pos_emb(patch_embeds_list, grid_hw) # [28072, 1024]
        assert (grid_hw[:,0] * grid_hw[:,1]).sum() == patch_embeds.shape[0]

        patches_list = []
        cur_position = 0
        for i in range(grid_hw.shape[0]):
            h, w = grid_hw[i]
            patches_per_img = patch_embeds[cur_position : cur_position + h * w].view(h, w, -1).unsqueeze(0)
            patches_per_img = self.dense_embedding(patches_per_img.permute(0, 3, 1, 2))
            patches_per_img = patches_per_img.permute(0, 2, 3, 1)
            patches_list.append(patches_per_img.view(-1, patches_per_img.shape[-1]))
            cur_position += h * w

        embeddings = torch.cat(patches_list, dim=0)  # (N_total // downsample_factor**2, C)

        assert cur_position == patch_embeds.shape[0]
        assert embeddings.shape[0] == int(patch_embeds.shape[0] / self.downsample_factor**2)

        return embeddings


class NEOVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    config_class = NEOVisionConfig
    # support transformers 4.51.+
    _tp_plan = ''

    def __init__(self, config: NEOVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = NEOVisionEmbeddings(config)

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
            grid_hw: Optional[torch.Tensor] = None,
            pixel_values_list=None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None and pixel_values_list is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if pixel_values_list is None:
                assert pixel_values.dim() == 2, f"pixel_values must be 2D for native resolution, got: {pixel_values.dim()}"
                pixel_values_list = [pixel_values]
                del pixel_values
            hidden_states = self.embeddings(pixel_values_list, grid_hw=grid_hw)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )
