from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

import copy
import math
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple
from transformers.utils.deprecation import deprecate_kwarg
from transformers import Qwen3Config

from mmgp import cuda_graphs
from shared import attention_kit
from shared.attention import pay_attention
from shared.kernels import ops

from .transformers_compat import causal_mask_kwargs, model_input_compat, tied_weights_keys


def _take_tensor(tensor_list):
    tensor = tensor_list[0]
    tensor_list.clear()
    return tensor


def _linear_disposable(linear, tensor_list):
    return linear(_take_tensor(tensor_list))


def _attention_mask_for_pay_attention(attention_mask, q_len, num_heads):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        if attention_mask.shape[1] == num_heads and attention_mask.shape[2] == q_len:
            return attention_mask.transpose(1, 2)
        if attention_mask.shape[2] in (1, num_heads) and attention_mask.shape[1] == q_len:
            return attention_mask
        if attention_mask.shape[1] == 1 and attention_mask.shape[2] == q_len:
            return attention_mask.transpose(1, 2)
    if attention_mask.dim() == 3:
        return attention_mask[:, :, None, :]
    if attention_mask.dim() == 2:
        return attention_mask[:, None, None, :]
    return attention_mask


def _pay_attention_gqa(qkv_list, dropout_p=0.0, softmax_scale=None, causal=False, attention_mask=None):
    q, k, v = qkv_list
    qkv_list.clear()
    batch, q_len, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.to(q.dtype)
    if num_heads == num_kv_heads:
        qkv_list = [q, k, v]
        del q, k, v
        return pay_attention(qkv_list, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, attention_mask=attention_mask, recycle_q=True)

    groups = num_heads // num_kv_heads
    k_len = k.shape[1]
    q = q.view(batch, q_len, num_kv_heads, groups, head_dim).permute(0, 2, 1, 3, 4).reshape(batch * num_kv_heads, q_len, groups, head_dim)
    k = k.permute(0, 2, 1, 3).reshape(batch * num_kv_heads, k_len, 1, head_dim).expand(-1, -1, groups, -1)
    v = v.permute(0, 2, 1, 3).reshape(batch * num_kv_heads, k_len, 1, head_dim).expand(-1, -1, groups, -1)
    if attention_mask is not None:
        if attention_mask.shape[2] == 1:
            attention_mask = attention_mask[:, None].expand(batch, num_kv_heads, q_len, 1, k_len).reshape(batch * num_kv_heads, q_len, 1, k_len)
        else:
            attention_mask = attention_mask.view(batch, q_len, num_kv_heads, groups, k_len).permute(0, 2, 1, 3, 4).reshape(batch * num_kv_heads, q_len, groups, k_len)
    qkv_list = [q, k, v]
    del q, k, v
    output = pay_attention(qkv_list, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, attention_mask=attention_mask, force_attention="sdpa", recycle_q=True)
    return output.view(batch, num_kv_heads, q_len, groups, head_dim).permute(0, 2, 1, 3, 4).reshape(batch, q_len, num_heads, head_dim)


def create_block_causal_mask(index: torch.Tensor):
    """
    index: (L)
    return: (1, 1, L, L) block-wise causal attention mask
    """
    L = index.size(0)
    idx_i = index.unsqueeze(1).expand(L, L)
    idx_j = index.unsqueeze(0).expand(L, L)

    arange = torch.arange(L, device=index.device)
    mask = (idx_j == idx_i) | (arange.unsqueeze(0) <= arange.unsqueeze(1))

    return torch.zeros_like(mask, dtype=torch.float32).masked_fill_(~mask, float("-inf"))[None, None]


def visualize_mask(mask: torch.Tensor, i: int = 0, j: int = 12):
    """
    mask: (1,1, L, L)
    """
    submask = torch.where(mask[0, 0, :, :] == 0, torch.tensor(1.0), torch.tensor(0.0))
    submask = mask[i:j, i:j].int().cpu().numpy()
    for row in submask:
        print(" ".join(map(str, row)))


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_list, hidden_states = [hidden_states], None
        return self.forward_list(hidden_states_list)

    def forward_list(self, hidden_states_list) -> torch.Tensor:
        return ops.rms_norm_rounded(hidden_states_list, self.variance_epsilon).mul_(self.weight)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x_list):
        x = _take_tensor(x_list)
        seq_len = x.shape[-2]
        if seq_len <= 512:
            x_list, x = [x], None
            return ops.gated_mlp(x_list, self.gate_proj, self.up_proj, self.down_proj, self.act_fn)
        chunk_size = max(128, min(seq_len, seq_len * self.hidden_size // self.intermediate_size))
        output = x.new_empty(*x.shape[:-1], self.hidden_size)
        for start in range(0, seq_len, chunk_size):
            chunk = x.narrow(-2, start, min(chunk_size, seq_len - start))
            chunk_list, chunk = [chunk], None
            chunk_output = ops.gated_mlp(chunk_list, self.gate_proj, self.up_proj, self.down_proj, self.act_fn)
            output.narrow(-2, start, chunk_output.shape[-2]).copy_(chunk_output)
            del chunk_output
        return output


def _apply_rotary_inplace(x, cos, sin, scratch, unsqueeze_dim):
    return ops.paired_rope_(x, cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim), scratch)


def _cached_prefix(past_key_values, layer_idx):
    """The cached prompt/reference keys and values of a layer, (batch, tokens, kv heads, head_dim) views, or None. A prompt cache expanded
    to the batch gives its single row: _forward_gen_kit expands it inside the layer, as CUDA graph buffers cannot hold expanded views."""
    layer = None if past_key_values is None else past_key_values.layers[layer_idx]
    if layer is None or layer.keys is None:
        return None
    keys, values = (layer.keys[:1], layer.values[:1]) if layer.keys.stride(0) == 0 else (layer.keys, layer.values)
    return keys.transpose(1, 2), values.transpose(1, 2)


class _StaticCacheLayer:
    def __init__(self, cache, index):
        self.cache, self.index = cache, index

    @property
    def keys(self):  # (batch, kv heads, tokens, head_dim) views of the tokens written, as a transformers cache layer gives
        return self.cache.k[self.index][:, :self.cache.lengths[self.index]].transpose(1, 2)

    @property
    def values(self):
        return self.cache.v[self.index][:, :self.cache.lengths[self.index]].transpose(1, 2)


class StaticKVCache:
    """A KV cache of fixed capacity for interleaved generation (self-correction). Prompts and image appends write through update(), as into
    a transformers DynamicCache; a written token goes through decode steps (Qwen3Model.decode_step) that write at the device slot `position`
    and attend to the slots up to it, with fixed shapes: CUDA graphs replay them (the cache is their registered state). Token-major storage
    (batch, slots, kv heads, head_dim): a layer's prefix is then a contiguous slice."""

    def __init__(self, config, capacity, rotary_emb_hw, device, dtype=torch.bfloat16):
        shape = (1, capacity, config.num_key_value_heads, getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        self.k = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.v = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)]
        self.lengths = [0] * config.num_hidden_layers
        self.layers = [_StaticCacheLayer(self, index) for index in range(config.num_hidden_layers)]
        self.capacity = capacity
        self.position = torch.zeros(1, dtype=torch.long, device=device)  # the slot of the token a decode step writes
        self.slots = torch.arange(capacity, device=device)
        # a written token's height and width positions are 0: their rotary tables, computed once (the module reads its frequencies from RAM)
        self.rope_hw = rotary_emb_hw(self.k[0], torch.zeros(1, 1, dtype=torch.long, device=device))

    def get_seq_length(self, layer_idx=0):
        return self.lengths[layer_idx]

    def reserve(self, count):
        """Room for count more tokens: a larger cache, its tokens copied layer by layer (one layer's copy at a time). New tensors: their CUDA
        graphs go (register the cache again)."""
        capacity = self.lengths[0] + count
        if capacity <= self.capacity:
            return False
        for index, length in enumerate(self.lengths):
            for tensors in (self.k, self.v):
                grown = tensors[index].new_zeros((1, capacity, *tensors[index].shape[2:]))
                grown[:, :length].copy_(tensors[index][:, :length])
                tensors[index] = grown
        self.capacity = capacity
        self.slots = torch.arange(capacity, device=self.slots.device)
        return True

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start, count = self.lengths[layer_idx], key_states.shape[2]
        if start + count > self.capacity:
            raise RuntimeError(f"SenseNova KV cache capacity {self.capacity} exceeded")
        self.k[layer_idx][:, start:start + count].copy_(key_states.transpose(1, 2))
        self.v[layer_idx][:, start:start + count].copy_(value_states.transpose(1, 2))
        self.lengths[layer_idx] = start + count
        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    def decode_bias(self, dtype):
        """Additive mask of a decode step: the slots up to `position`, (batch, tokens, 1, slots) as _pay_attention_gqa takes it."""
        return torch.zeros(1, 1, 1, self.capacity, dtype=dtype, device=self.slots.device).masked_fill_(self.slots > self.position, float("-inf"))

    def begin_decode(self):
        """Before a decode step: its token goes to the slot after the tokens written (prompts and images included)."""
        if self.lengths[0] >= self.capacity:
            raise RuntimeError(f"SenseNova KV cache capacity {self.capacity} exceeded")
        self.position.fill_(self.lengths[0])

    def end_decode(self):
        self.lengths = [length + 1 for length in self.lengths]


def _norm_rope_rows(x, norm_t, norm_hw, cos, sin, partner):
    """forward_gen's norms of the t and hw halves of the head dimension, then RoPE of its t, h and w parts, of rows x (batch, tokens,
    heads, head_dim), in one pass each: the same arithmetic as the Qwen3RMSNorm calls and the three rotate-half passes."""
    normed = ops.rms_norm_rounded([x.unflatten(-1, (2, -1))], norm_t.variance_epsilon)
    normed = normed.mul_(torch.stack((norm_t.weight, norm_hw.weight))).flatten(-2)
    normed_list, normed = [normed], None
    return ops.indexed_rope(normed_list, cos, sin, partner)


def apply_rotary_pos_emb(qk_list, cos, sin, scratch, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    q, k = qk_list
    qk_list.clear()
    return _apply_rotary_inplace(q, cos, sin, scratch, unsqueeze_dim), _apply_rotary_inplace(k, cos, sin, scratch, unsqueeze_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    attention_mask = _attention_mask_for_pay_attention(attention_mask, query.shape[1], query.shape[2])
    qkv_list = [query, key, value]
    del query, key, value
    return _pay_attention_gqa(qkv_list, dropout_p=dropout, softmax_scale=scaling, attention_mask=attention_mask), None


def _compute_default_rope_parameters(config, device=None, **_kwargs):
    """Default RoPE frequencies, inlined to avoid breakage across transformers versions.

    transformers <=4.x exposes this as ``ROPE_INIT_FUNCTIONS["default"]``, but
    5.x dropped the ``"default"`` key from that table. Having a local copy keeps
    ``Qwen3RotaryEmbedding`` working on both.
    """
    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)

    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default" or self.rope_type is None:
            base_rope_init_fn = _compute_default_rope_parameters
        else:
            base_rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        def _rope_init_fn_keep_freq_range(cfg: Qwen3Config, dev=None):
            inv_freq, attention_scaling = base_rope_init_fn(cfg, dev)

            cfg2 = copy.deepcopy(cfg)
            head_dim = getattr(cfg2, "head_dim", None)
            if head_dim is None:
                head_dim = cfg2.hidden_size // cfg2.num_attention_heads
                setattr(cfg2, "head_dim", head_dim)
            cfg2.head_dim = int(head_dim) * 2

            inv_freq_full, _ = base_rope_init_fn(cfg2, dev)
            inv_freq = inv_freq_full[::2]

            return inv_freq, attention_scaling

        self.rope_init_fn = _rope_init_fn_keep_freq_range

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def reset_inv_freq(self):
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, torch.device("cpu"))
        self.inv_freq = inv_freq
        self.original_inv_freq = inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos().mul_(self.attention_scaling)
            sin = freqs.sin().mul_(self.attention_scaling)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_proj_mot_gen = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )

        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_mot_gen = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )

        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_mot_gen = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.o_proj_mot_gen = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        self.k_norm = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)
        self.k_norm_hw = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.k_norm_hw_mot_gen = Qwen3RMSNorm(self.head_dim // 2, eps=config.rms_norm_eps)

        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

        t_config = copy.deepcopy(config)
        t_config.head_dim = config.head_dim // 2
        self.rotary_emb = Qwen3RotaryEmbedding(config=t_config)

        hw_config = copy.deepcopy(config)
        hw_config.head_dim = config.head_dim // 4
        hw_config.rope_theta = config.rope_theta_hw
        if isinstance(getattr(hw_config, "rope_parameters", None), dict):
            hw_config.rope_parameters = {**hw_config.rope_parameters, "rope_theta": config.rope_theta_hw}
        hw_config.max_position_embeddings = config.max_position_embeddings_hw
        self.rotary_emb_hw = Qwen3RotaryEmbedding(config=hw_config)
    
    def forward_und(
        self,
        hidden_states_list,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = _take_tensor(hidden_states_list)
        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        hidden_states_list = [hidden_states]
        del hidden_states
        value_states = _linear_disposable(self.v_proj, hidden_states_list).view(hidden_shape)

        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t.copy_(self.q_norm(query_states_t))
        query_states_hw.copy_(self.q_norm_hw(query_states_hw))
        query_states_t = query_states_t.transpose(1, 2)
        query_states_hw = query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t.copy_(self.k_norm(key_states_t))
        key_states_hw.copy_(self.k_norm_hw(key_states_hw))
        key_states_t = key_states_t.transpose(1, 2)
        key_states_hw = key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        rope_scratch = torch.empty_like(query_states_t[..., :query_states_t.shape[-1] // 2])
        qk_list = [query_states_t, key_states_t]
        del query_states_t, key_states_t
        query_states_t, key_states_t = apply_rotary_pos_emb(qk_list, cos_t, sin_t, rope_scratch)
        qk_list = [query_states_h, key_states_h]
        del query_states_h, key_states_h
        query_states_h, key_states_h = apply_rotary_pos_emb(qk_list, cos_h, sin_h, rope_scratch)
        qk_list = [query_states_w, key_states_w]
        del query_states_w, key_states_w
        query_states_w, key_states_w = apply_rotary_pos_emb(qk_list, cos_w, sin_w, rope_scratch)

        del query_states_t, query_states_hw, query_states_h, query_states_w
        del key_states_t, key_states_hw, key_states_h, key_states_w
        del cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, rope_scratch

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
            else:
                # only use the past key values but do not append the current one
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output_list = [attn_output]
        del attn_output
        attn_output = _linear_disposable(self.o_proj, attn_output_list)
        return attn_output, attn_weights

    def forward_decode(self, hidden_states_list, cache, rope, bias):
        """forward_und for one written token (batch 1), with fixed shapes for CUDA graphs: its keys/values go to the slot cache.position and
        it attends to the slots up to it (bias). rope: the (cos, sin) tables of its time, height and width positions."""
        hidden_states = _take_tensor(hidden_states_list)
        hidden_shape = (1, 1, -1, self.head_dim)
        (cos_t, sin_t), (cos_h, sin_h), (cos_w, sin_w) = rope
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        hidden_states_list = [hidden_states]
        del hidden_states
        value_states = _linear_disposable(self.v_proj, hidden_states_list).view(hidden_shape)
        # the norms and rotations of forward_und, in place on views of query_states / key_states
        for states, norm_t, norm_hw in ((query_states, self.q_norm, self.q_norm_hw), (key_states, self.k_norm, self.k_norm_hw)):
            states_t, states_hw = states.chunk(2, dim=-1)
            states_t.copy_(norm_t(states_t))
            states_hw.copy_(norm_hw(states_hw))
        query_t, query_hw = query_states.chunk(2, dim=-1)
        key_t, key_hw = key_states.chunk(2, dim=-1)
        query_t, query_hw, key_t, key_hw = query_t.transpose(1, 2), query_hw.transpose(1, 2), key_t.transpose(1, 2), key_hw.transpose(1, 2)
        query_h, query_w = query_hw.chunk(2, dim=-1)
        key_h, key_w = key_hw.chunk(2, dim=-1)
        rope_scratch = torch.empty_like(query_t[..., :query_t.shape[-1] // 2])
        for query, key, cos, sin in ((query_t, key_t, cos_t, sin_t), (query_h, key_h, cos_h, sin_h), (query_w, key_w, cos_w, sin_w)):
            apply_rotary_pos_emb([query, key], cos, sin, rope_scratch)
        del query_t, query_hw, query_h, query_w, key_t, key_hw, key_h, key_w, rope_scratch
        k_cache, v_cache = cache.k[self.layer_idx], cache.v[self.layer_idx]
        k_cache.index_copy_(1, cache.position, key_states)
        v_cache.index_copy_(1, cache.position, value_states)
        del key_states, value_states
        qkv_list = [query_states, k_cache, v_cache]
        del query_states
        attn_output = _pay_attention_gqa(qkv_list, softmax_scale=self.scaling, attention_mask=bias).reshape(1, 1, -1)
        return _linear_disposable(self.o_proj, [attn_output])

    # def forward_gen(
    #     self,
    #     hidden_states: torch.Tensor,
    #     indexes: Optional[torch.LongTensor],
    #     attention_mask: Optional[torch.Tensor],
    #     past_key_values: Optional[Cache] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     **kwargs: Unpack[FlashAttentionKwargs],
    # ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     assert self.config._attn_implementation == "eager"
    #     input_shape = hidden_states.shape[:-1]
    #     hidden_shape = (*input_shape, -1, self.head_dim)

    #     query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
    #     query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
    #     query_states_t = self.q_norm_mot_gen(query_states_t).transpose(1, 2)
    #     query_states_hw = self.q_norm_hw_mot_gen(query_states_hw).transpose(1, 2)
    #     query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

    #     key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
    #     key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
    #     key_states_t = self.k_norm_mot_gen(key_states_t).transpose(1, 2)
    #     key_states_hw = self.k_norm_hw_mot_gen(key_states_hw).transpose(1, 2)
    #     key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

    #     value_states = self.v_proj_mot_gen(hidden_states).view(hidden_shape).transpose(1, 2)

    #     cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
    #     query_states_t, key_states_t = apply_rotary_pos_emb(query_states_t, key_states_t, cos_t, sin_t)

    #     cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
    #     query_states_h, key_states_h = apply_rotary_pos_emb(query_states_h, key_states_h, cos_h, sin_h)

    #     cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
    #     query_states_w, key_states_w = apply_rotary_pos_emb(query_states_w, key_states_w, cos_w, sin_w)

    #     query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
    #     key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)


    #     if past_key_values is not None:
    #         # sin and cos are specific to RoPE models; cache_position needed for the static cache
    #         # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    #         # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
    #         update_cache = kwargs.get("update_cache", True)
    #         if update_cache:
    #             key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
    #         else:
    #             # only use the past key values but do not append the current one
    #             layer = past_key_values.layers[self.layer_idx]
    #             past_k, past_v = layer.keys, layer.values

    #             if past_k is not None:
    #                 key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
    #                 value_states = torch.cat([past_v, value_states], dim=2)

    #     attention_interface: Callable = eager_attention_forward
    #     if self.config._attn_implementation != "eager":
    #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    #     attn_output, attn_weights = attention_interface(
    #         self,
    #         query_states,
    #         key_states,
    #         value_states,
    #         attention_mask,
    #         dropout=0.0 if not self.training else self.attention_dropout,
    #         scaling=self.scaling,
    #         sliding_window=self.sliding_window,  # diff with Llama
    #         **kwargs,
    #     )

    #     attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    #     attn_output = self.o_proj_mot_gen(attn_output)
    #     return attn_output, attn_weights

    def gen_rope_tables(self, hidden_states, indexes):
        """forward_gen's t, h and w rotary tables over the whole head dimension, for one rotate-half pass: cos and signed sin
        (1, tokens, 1, head_dim), and the index of each channel's partner. The same for every layer of a step."""
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        cos = torch.cat((cos_t, cos_t, cos_h, cos_h, cos_w, cos_w), dim=-1).unsqueeze(2)
        sin = torch.cat((-sin_t, sin_t, -sin_h, sin_h, -sin_w, sin_w), dim=-1).unsqueeze(2)
        partner, start = [], 0
        for half in (cos_t.shape[-1], cos_h.shape[-1], cos_w.shape[-1]):  # each part: the first half takes the second one, and back
            partner += [torch.arange(start + half, start + 2 * half, device=cos.device), torch.arange(start, start + half, device=cos.device)]
            start += 2 * half
        return cos, sin, torch.cat(partner)

    def _norm_rope_gen_(self, query, key, rope):
        """attention_kit callback: forward_gen's norms and RoPE of q and/or k in place, by chunks of tokens."""
        cos, sin, partner = rope
        for tensor, norm_t, norm_hw in ((query, self.q_norm_mot_gen, self.q_norm_hw_mot_gen), (key, self.k_norm_mot_gen, self.k_norm_hw_mot_gen)):
            if tensor is not None:
                attention_kit.rows_(tensor, lambda part, first, last: _norm_rope_rows(part, norm_t, norm_hw, cos[:, first:last], sin[:, first:last], partner))

    def _o_proj_gen(self, attn_output_list, residual):
        """o_proj_mot_gen of the handed-off attention output, or added into residual (returned) when given."""
        if residual is None:
            return _linear_disposable(self.o_proj_mot_gen, attn_output_list)
        return ops.linear_residual_(self.o_proj_mot_gen, attn_output_list, residual)

    def _forward_gen_kit(self, hidden_states_list, indexes, kv_prefix, rope, residual):
        """forward_gen without mask: the current tokens attend to the cached prefix kv_prefix and themselves through attention_kit;
        rope: the step's gen_rope_tables, computed here when not given."""
        if rope is None:
            rope = self.gen_rope_tables(hidden_states_list[0], indexes)
        batch = hidden_states_list[0].shape[0]
        if kv_prefix is not None and kv_prefix[0].shape[0] != batch:  # one prompt for the whole batch
            kv_prefix = tuple(tensor.expand(batch, *tensor.shape[1:]) for tensor in kv_prefix)
        attn_output = attention_kit.qkv_attention(hidden_states_list, self.q_proj_mot_gen, self.k_proj_mot_gen, self.v_proj_mot_gen,
                                                  self.config.num_attention_heads, self.head_dim, lambda query, key, group: self._norm_rope_gen_(query, key, rope),
                                                  kv_heads=self.config.num_key_value_heads, kv_prefix=kv_prefix)
        attn_output_list = [attn_output.flatten(2)]
        del attn_output, kv_prefix
        return self._o_proj_gen(attn_output_list, residual)

    def forward_gen(
        self,
        hidden_states_list,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        residual: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """residual given: the output projection is added into it (returned in place of the attention output)."""
        update_cache = kwargs.get("update_cache", True)
        if attention_mask is None and (past_key_values is None or not update_cache):
            kv_prefix = kwargs["kv_prefix"] if "kv_prefix" in kwargs else _cached_prefix(past_key_values, self.layer_idx)
            return self._forward_gen_kit(hidden_states_list, indexes, kv_prefix, kwargs.get("rope"), residual), None
        hidden_states = _take_tensor(hidden_states_list)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # -----------------------------
        # Build q / k / v for current tokens
        # Internal layout before flash:
        #   q/k/v: [B, H, S, D]
        # Flash layout:
        #   q/k/v: [B, S, H, D]
        # -----------------------------
        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        query_states = self.q_proj_mot_gen(hidden_states).view(hidden_shape)
        key_states = self.k_proj_mot_gen(hidden_states).view(hidden_shape)
        hidden_states_list = [hidden_states]
        del hidden_states
        value_states = _linear_disposable(self.v_proj_mot_gen, hidden_states_list).view(hidden_shape)

        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)
        query_states_t.copy_(self.q_norm_mot_gen(query_states_t))
        query_states_hw.copy_(self.q_norm_hw_mot_gen(query_states_hw))
        query_states_t = query_states_t.transpose(1, 2)   # [B,H,S,D/2]
        query_states_hw = query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)
        key_states_t.copy_(self.k_norm_mot_gen(key_states_t))
        key_states_hw.copy_(self.k_norm_hw_mot_gen(key_states_hw))
        key_states_t = key_states_t.transpose(1, 2)       # [B,H,S,D/2]
        key_states_hw = key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        rope_scratch = torch.empty_like(query_states_t[..., :query_states_t.shape[-1] // 2])
        qk_list = [query_states_t, key_states_t]
        del query_states_t, key_states_t
        query_states_t, key_states_t = apply_rotary_pos_emb(qk_list, cos_t, sin_t, rope_scratch)
        qk_list = [query_states_h, key_states_h]
        del query_states_h, key_states_h
        query_states_h, key_states_h = apply_rotary_pos_emb(qk_list, cos_h, sin_h, rope_scratch)
        qk_list = [query_states_w, key_states_w]
        del query_states_w, key_states_w
        query_states_w, key_states_w = apply_rotary_pos_emb(qk_list, cos_w, sin_w, rope_scratch)

        del query_states_t, query_states_hw, query_states_h, query_states_w
        del key_states_t, key_states_hw, key_states_h, key_states_w
        del cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, rope_scratch

        # ------------------------------------------------------------------
        # Original eager fallback path
        # ------------------------------------------------------------------
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if past_key_values is not None:
            if update_cache:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs=None
                )
            else:
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values
                if past_k is not None:
                    key_states = torch.cat([past_k, key_states], dim=2)
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        del query_states, key_states, value_states

        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output_list = [attn_output]
        del attn_output
        return self._o_proj_gen(attn_output_list, residual), attn_weights

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(hidden_states, indexes, attention_mask, past_key_values, cache_position, **kwargs)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(hidden_states, indexes, attention_mask, past_key_values, cache_position, **kwargs)

        # Mixed und/gen path: mirrors forward_und / forward_gen per token type.
        # (Fixed per issue #207: the time-dim qk-norm, the `.view(hidden_shape)` before
        # chunking, and the transpose on the time chunk were previously missing.)
        # Note: Remove this raise once fully tested.
        raise NotImplementedError(
            "The mixed und/gen forward path is not yet validated (issue #207): known "
            "issues are fixed, but it has no parity test and no production caller. "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        assert self.config._attn_implementation == "eager"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = hidden_states.new_zeros((*input_shape, self.config.num_attention_heads*self.head_dim))
        if exist_non_image_gen_tokens:
            query_states[~image_gen_indicators] = self.q_proj(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            query_states[image_gen_indicators] = self.q_proj_mot_gen(hidden_states[image_gen_indicators])
        query_states = query_states.view(hidden_shape)  # [B, S, H, D]
        query_states_t, query_states_hw = query_states.chunk(2, dim=-1)

        _query_states_t = query_states_t.new_zeros(query_states_t.shape)
        if exist_non_image_gen_tokens:
            _query_states_t[~image_gen_indicators] = self.q_norm(query_states_t[~image_gen_indicators])
        if exist_image_gen_tokens:
            _query_states_t[image_gen_indicators] = self.q_norm_mot_gen(query_states_t[image_gen_indicators])
        query_states_t = _query_states_t.transpose(1, 2)  # [B, H, S, D/2]

        _query_states_hw = query_states_hw.new_zeros(query_states_hw.shape)
        if exist_non_image_gen_tokens:
            _query_states_hw[~image_gen_indicators] = self.q_norm_hw(query_states_hw[~image_gen_indicators])
        if exist_image_gen_tokens:
            _query_states_hw[image_gen_indicators] = self.q_norm_hw_mot_gen(query_states_hw[image_gen_indicators])
        query_states_hw = _query_states_hw.transpose(1, 2)
        query_states_h, query_states_w = query_states_hw.chunk(2, dim=-1)

        key_states = hidden_states.new_zeros((*input_shape, self.config.num_key_value_heads*self.head_dim))
        if exist_non_image_gen_tokens:
            key_states[~image_gen_indicators] = self.k_proj(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            key_states[image_gen_indicators] = self.k_proj_mot_gen(hidden_states[image_gen_indicators])
        key_states = key_states.view(hidden_shape)  # [B, S, H_kv, D]
        key_states_t, key_states_hw = key_states.chunk(2, dim=-1)

        _key_states_t = key_states_t.new_zeros(key_states_t.shape)
        if exist_non_image_gen_tokens:
            _key_states_t[~image_gen_indicators] = self.k_norm(key_states_t[~image_gen_indicators])
        if exist_image_gen_tokens:
            _key_states_t[image_gen_indicators] = self.k_norm_mot_gen(key_states_t[image_gen_indicators])
        key_states_t = _key_states_t.transpose(1, 2)  # [B, H_kv, S, D/2]

        _key_states_hw = key_states_hw.new_zeros(key_states_hw.shape)
        if exist_non_image_gen_tokens:
            _key_states_hw[~image_gen_indicators] = self.k_norm_hw(key_states_hw[~image_gen_indicators])
        if exist_image_gen_tokens:
            _key_states_hw[image_gen_indicators] = self.k_norm_hw_mot_gen(key_states_hw[image_gen_indicators])
        key_states_hw = _key_states_hw.transpose(1, 2)
        key_states_h, key_states_w = key_states_hw.chunk(2, dim=-1)

        value_states = hidden_states.new_zeros((*input_shape, self.config.num_key_value_heads*self.head_dim))
        if exist_non_image_gen_tokens:
            value_states[~image_gen_indicators] = self.v_proj(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            value_states[image_gen_indicators] = self.v_proj_mot_gen(hidden_states[image_gen_indicators])
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos_t, sin_t = self.rotary_emb(hidden_states, indexes[0].unsqueeze(0))
        cos_h, sin_h = self.rotary_emb_hw(hidden_states, indexes[1].unsqueeze(0))
        cos_w, sin_w = self.rotary_emb_hw(hidden_states, indexes[2].unsqueeze(0))
        rope_scratch = torch.empty_like(query_states_t[..., :query_states_t.shape[-1] // 2])
        qk_list = [query_states_t, key_states_t]
        del query_states_t, key_states_t
        query_states_t, key_states_t = apply_rotary_pos_emb(qk_list, cos_t, sin_t, rope_scratch)
        qk_list = [query_states_h, key_states_h]
        del query_states_h, key_states_h
        query_states_h, key_states_h = apply_rotary_pos_emb(qk_list, cos_h, sin_h, rope_scratch)
        qk_list = [query_states_w, key_states_w]
        del query_states_w, key_states_w
        query_states_w, key_states_w = apply_rotary_pos_emb(qk_list, cos_w, sin_w, rope_scratch)

        query_states = torch.cat([query_states_t, query_states_h, query_states_w], dim=-1)
        key_states = torch.cat([key_states_t, key_states_h, key_states_w], dim=-1)


        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            update_cache = kwargs.get("update_cache", True)
            if update_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
            else:
                # only use the past key values but do not append the current one
                layer = past_key_values.layers[self.layer_idx]
                past_k, past_v = layer.keys, layer.values

                if past_k is not None:
                    key_states   = torch.cat([past_k, key_states], dim=2)   # concat on seq_len
                    value_states = torch.cat([past_v, value_states], dim=2)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )
        del query_states, key_states, value_states

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        _attn_output = attn_output.new_zeros((*input_shape, self.config.hidden_size))
        if exist_non_image_gen_tokens:
            _attn_output[~image_gen_indicators] = self.o_proj(attn_output[~image_gen_indicators])
        if exist_image_gen_tokens:
            _attn_output[image_gen_indicators] = self.o_proj_mot_gen(attn_output[image_gen_indicators])

        attn_output = _attn_output
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states_list,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = _take_tensor(hidden_states_list)
        hidden_states = self.input_layernorm(residual)
        hidden_states_list = [hidden_states]
        del hidden_states
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states_list,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual.add_(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states_list = [hidden_states]
        del hidden_states
        hidden_states = self.mlp(hidden_states_list)
        hidden_states = residual.add_(hidden_states)
        return hidden_states

    def forward_gen(
        self,
        hidden_states_list,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = _take_tensor(hidden_states_list)
        hidden_states = self.input_layernorm_mot_gen(residual)
        hidden_states_list = [hidden_states]
        del hidden_states
        # Self Attention, its output projection added into the residual
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states_list,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            residual=residual,
            **kwargs,
        )

        # Fully Connected, by chunks of tokens added into the residual (hidden_act silu)
        hidden_states_list = [self.post_attention_layernorm_mot_gen(hidden_states)]
        mlp = self.mlp_mot_gen
        return ops.mlp(hidden_states_list, mlp.gate_proj, mlp.down_proj, "swiglu", up=mlp.up_proj, residual=hidden_states)

    def forward_decode(self, hidden_states_list, cache, rope, bias):
        """forward_und for one written token through a StaticKVCache (Qwen3Model.decode_step)."""
        residual = _take_tensor(hidden_states_list)
        hidden_states_list = [self.input_layernorm(residual)]
        hidden_states = residual.add_(self.self_attn.forward_decode(hidden_states_list, cache, rope, bias))
        residual = hidden_states
        hidden_states_list = [self.post_attention_layernorm(hidden_states)]
        return residual.add_(self.mlp(hidden_states_list))

    def forward_gen_branches(self, hidden_states, branches):
        outputs = []
        for branch in branches:
            branch_hidden_states = [hidden_states.pop(0)]
            outputs.append(self.forward_gen(
                branch_hidden_states,
                image_gen_indicators=None,
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
                indexes=branch["indexes"],
                attention_mask=branch["attention_mask"][self.attention_type],
                past_key_values=branch["past_key_values"],
                use_cache=True,
                update_cache=False,
                rope=branch["rope"],
                kv_prefix=branch["kv_prefix"],
            ))
        return outputs

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        branches=None,
        decode_cache=None,
        decode_rope=None,
        decode_bias=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if branches is not None:
            return self.forward_gen_branches(hidden_states, branches)
        if decode_cache is not None:
            return self.forward_decode(hidden_states, decode_cache, decode_rope, decode_bias)
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(hidden_states, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, indexes, attention_mask, position_ids, past_key_values, use_cache, cache_position, **kwargs)
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(hidden_states, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, indexes, attention_mask, position_ids, past_key_values, use_cache, cache_position, **kwargs)

        # Mixed und/gen path — see the NOTE in Qwen3Attention.forward for caveats.
        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )

        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.input_layernorm(hidden_states[~image_gen_indicators])
        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.input_layernorm_mot_gen(hidden_states[image_gen_indicators])
        hidden_states = _hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        _hidden_states = hidden_states.new_zeros(hidden_states.shape)
        if exist_non_image_gen_tokens:
            _hidden_states[~image_gen_indicators] = self.mlp([self.post_attention_layernorm(hidden_states[~image_gen_indicators])])

        if exist_image_gen_tokens:
            _hidden_states[image_gen_indicators] = self.mlp_mot_gen([self.post_attention_layernorm_mot_gen(hidden_states[image_gen_indicators])])

        hidden_states = _hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.current_index = -1

        # Initialize weights and apply final processing
        self.post_init()

    @model_input_compat
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_gen_indicators: Optional[torch.Tensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds_list=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        
        # assert position_ids is not None
        # assert cache_position is not None
        # assert past_key_values is not None 

        if image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            # Normalize once before the decoder loop. Leaving these as CUDA
            # scalar tensors makes every layer's Python branch read them back
            # to the host, serializing the compute stream and defeating weight
            # prefetch overlap.
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())
        
        if sum(value is not None for value in (input_ids, inputs_embeds, inputs_embeds_list)) != 1:
            raise ValueError("You must specify exactly one of input_ids, inputs_embeds, or inputs_embeds_list")

        inputs_embeds_owned = inputs_embeds_list is not None or input_ids is not None
        if inputs_embeds_list is not None:
            inputs_embeds = _take_tensor(inputs_embeds_list)
        elif inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            if input_ids is not None:
                mask_kwargs = causal_mask_kwargs(
                    create_causal_mask,
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                # Create the masks
                causal_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                }
                self.current_index += 1
                indexes = torch.LongTensor([[self.current_index], [0], [0]]).to(input_ids.device)
            else:
                causal_mask_mapping = {
                    "full_attention": create_block_causal_mask(indexes[0]),
                }
                self.current_index = indexes[0].max()
        else:
            self.current_index = indexes[0].max()
            # raise NotImplementedError('not isinstance(causal_mask_mapping := attention_mask, dict)')

            # The sliding window alternating layers are not always activated depending on the config
            # if self.has_sliding_layers:
            #     causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds if inputs_embeds_owned else inputs_embeds.clone()
        del inputs_embeds

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states_list = [hidden_states]
            del hidden_states
            hidden_states = decoder_layer(
                hidden_states_list,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            if getattr(self, "_interrupt", False):
                raise InterruptedError("SenseNova-U1 generation aborted")
        if not exist_image_gen_tokens:
            hidden_states_list = [hidden_states]
            del hidden_states
            hidden_states = self.norm.forward_list(hidden_states_list)
        elif not exist_non_image_gen_tokens:
            hidden_states_list = [hidden_states]
            del hidden_states
            hidden_states = self.norm_mot_gen.forward_list(hidden_states_list)
        else:
            _hidden_states = hidden_states.new_zeros(hidden_states.shape)
            _hidden_states[~image_gen_indicators] = self.norm(hidden_states[~image_gen_indicators])
            _hidden_states[image_gen_indicators] = self.norm_mot_gen(hidden_states[image_gen_indicators])
            hidden_states = _hidden_states
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def decode_step(self, hidden_states, cos_t, sin_t, cache):
        """One written token (embedded, (1, 1, hidden)) through the layers: its keys/values go to the slot cache.position, its final hidden
        state is returned. Fixed shapes and the cache as registered state: one CUDA graph when the layers are in VRAM, else a graph per
        layer (shared.llm_engines.cudagraph_kit). cos_t / sin_t: the rotary tables of its time position."""
        bias = cache.decode_bias(hidden_states.dtype)
        rope = ((cos_t, sin_t), cache.rope_hw, cache.rope_hw)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer([hidden_states], image_gen_indicators=None, exist_non_image_gen_tokens=True, exist_image_gen_tokens=False, decode_cache=cache, decode_rope=rope, decode_bias=bias)
        return self.norm.forward_list([hidden_states])

    def forward_gen_branches(self, input_embeds_list, branches):
        input_embeds = _take_tensor(input_embeds_list)
        # the rotary tables depend on the positions only: computed once per branch for all the layers
        attention = self.layers[0].self_attn
        branches = [dict(branch, rope=attention.gen_rope_tables(input_embeds, branch["indexes"])) for branch in branches]
        hidden_states = [input_embeds]
        hidden_states.extend(input_embeds.clone() for _ in branches[1:])
        del input_embeds
        # without masks, each layer gets its cached prefix as tensors, not the cache: its calls can then replay CUDA graphs
        unmasked = all(mask is None for branch in branches for mask in branch["attention_mask"].values())
        with cuda_graphs.scope(unmasked):
            for index, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
                layer_branches = [dict(branch, past_key_values=None, kv_prefix=_cached_prefix(branch["past_key_values"], index)) if unmasked else dict(branch, kv_prefix=None)
                                  for branch in branches]
                hidden_states = decoder_layer(hidden_states, image_gen_indicators=None, exist_non_image_gen_tokens=False, exist_image_gen_tokens=True, branches=layer_branches)
                if getattr(self, "_interrupt", False):
                    raise InterruptedError("SenseNova-U1 generation aborted")
        outputs = []
        while hidden_states:
            hidden_states_list = [hidden_states.pop(0)]
            outputs.append(self.norm_mot_gen.forward_list(hidden_states_list))
        return outputs


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = tied_weights_keys("lm_head.weight", "model.embed_tokens.weight")
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        indexes: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inputs_embeds_list=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            inputs_embeds_list=inputs_embeds_list,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        past_key_values = outputs.past_key_values
        output_hidden_states = outputs.hidden_states
        output_attentions = outputs.attentions
        hidden_states = outputs.last_hidden_state
        del outputs
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        hidden_states_list = [hidden_states]
        del hidden_states
        logits = _linear_disposable(self.lm_head, hidden_states_list)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=output_hidden_states,
            attentions=output_attentions,
        )


class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]
