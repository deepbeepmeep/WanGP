# Copyright (c) 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for WanGP (2026): imports limited to this package (no model registry, no sys.path change, no unused torchaudio import).
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .moe import HybridMoE

class StreamingCache:
    def __init__(self):
        self.conv2d_cache = {}
        self.pos_cache = {}
        self.kv_cache = {}
        
    def clear(self):
        self.conv2d_cache.clear()
        self.pos_cache.clear()
        self.kv_cache.clear()
        
    def get_conv2d_cache(self, name: str):
        return self.conv2d_cache.get(name, None)
    
    def set_conv2d_cache(self, name: str, cache: torch.Tensor):
        self.conv2d_cache[name] = cache

    def get_pos_cache(self, name: str):
        return self.pos_cache.get(name, None)
    
    def set_pos_cache(self, name: str, cache: torch.Tensor):
        self.pos_cache[name] = cache

    def get_kv_cache(self, layer_idx: int):
        return self.kv_cache.get(layer_idx, None)
    
    def set_kv_cache(self, layer_idx: int, k_cache: torch.Tensor, v_cache: torch.Tensor):
        self.kv_cache[layer_idx] = (k_cache, v_cache)


class LayerNorm(nn.Module): 
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, T, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, T).
    """
    def __init__(self, C, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(C))
        self.bias = nn.Parameter(torch.zeros(C))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.C = (C, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.C, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            
            w = self.weight
            b = self.bias
            for _ in range(x.ndim-2):
                w = w.unsqueeze(-1)
                b = b.unsqueeze(-1)
            x = w * x + b 
            return x
        else:
            raise NotImplementedError
        
    def extra_repr(self) -> str:
        return ", ".join([f"{k}={v}" for k,v in {"C" : self.C, "data_format" : self.data_format, "eps" : self.eps}.items()])

# ============================================ Transformer =================================================

class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MultiHeadAttention(nn.Module):
    """Multi-headed attention from ' Attention Is All You Need' paper"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got ` embed_dim`: {self.embed_dim}"
                f"and ` num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias= bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias= bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias= bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias= bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(self, hidden_states: torch.Tensor, use_cache=False, kv_cache=None, causal=False):
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len,_= hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if use_cache and kv_cache is not None:
            k_cache, v_cache = kv_cache
            key_states = torch.cat([k_cache, key_states], dim=2)   # [bsz, num_heads, cache_len+tgt_len, head_dim]
            value_states = torch.cat([v_cache, value_states], dim=2)
        
        new_kv_cache = None
        if use_cache:
            new_kv_cache = (key_states.detach(), value_states.detach())

        proj_shape = (bsz* self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if causal:
            mask = torch.triu(torch.ones(tgt_len, tgt_len, device=attn_weights.device, dtype=torch.bool), diagonal=1)  # [tgt_len, tgt_len] 上三角=True
            if src_len > tgt_len:
                prefix = torch.zeros(tgt_len, src_len - tgt_len, device=attn_weights.device, dtype=torch.bool)  # 历史KV全可见
                mask = torch.cat([prefix, mask], dim=1)  # [tgt_len, src_len]
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).expand(bsz * self.num_heads, -1, -1), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the ` embed_dim` from the config (stored in the class) rather than` hidden_state` because ` attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        return attn_output, new_kv_cache


class FeedForward(nn.Module):
    def __init__(self,
            hidden_size: int,
            intermediate_size: int,
            activation_dropout: float=0.0,
            hidden_dropout: float=0.0,
            moe: bool = False,
            moe_type: str = "token",
            num_experts_shard: int = 6,
            num_experts_token: int = 6,
            top_k_shard: int = None,
            top_k_token: int = None,
            shard_scale: float = 1.0
        ):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        if moe:
            self.intermediate_dense = HybridMoE(
                hidden_size,
                intermediate_size,
                num_experts_shard=num_experts_shard,
                num_experts_token=num_experts_token,
                top_k_shard=top_k_shard,
                top_k_token=top_k_token,
                moe_type=moe_type,
                shard_scale=shard_scale
            )
        else:
            self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = NewGELUActivation()
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
            n_state: int,
            n_mlp : int,
            n_head: int,
            channel_last: bool = False,
            act_do: float = 0.0,
            att_do: float = 0.0,
            hid_do: float = 0.0,
            ln_eps: float=1e-6,
            moe: bool = False,
            moe_type: str = "token",
            num_experts_shard: int = 6,
            num_experts_token: int = 6,
            top_k_shard: int = None,
            top_k_token: int = None,
            shard_scale: float = 1.0,
            erf_norm: bool = False
        ):

        hidden_size=n_state
        num_attention_heads=n_head
        intermediate_size=n_mlp
        activation_dropout= act_do
        attention_dropout= att_do
        hidden_dropout= hid_do
        layer_norm_eps= ln_eps

        super().__init__()
        self.channel_last = channel_last
        self.attention = MultiHeadAttention(
            embed_dim= hidden_size,
            num_heads= num_attention_heads,
            dropout= attention_dropout
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.feed_forward = FeedForward(
            hidden_size= hidden_size,
            intermediate_size= intermediate_size,
            activation_dropout= activation_dropout,
            hidden_dropout= hidden_dropout,
            moe=moe,
            moe_type=moe_type,
            num_experts_shard=num_experts_shard,
            num_experts_token=num_experts_token,
            top_k_shard=top_k_shard,
            top_k_token=top_k_token,
            shard_scale=shard_scale
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states, use_cache=False, kv_cache=None, causal=False):
        if not self.channel_last:
            hidden_states = hidden_states.permute(0, 2, 1)
        attn_residual = hidden_states
        hidden_states, new_kv_cache = self.attention(hidden_states, use_cache=use_cache, kv_cache=kv_cache, causal=causal)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states
        if not self.channel_last:
            outputs = outputs.permute(0, 2, 1)
        return outputs, new_kv_cache

# ======================================= ResNet block =============================================

class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_divisor=None, stride=1, pw=True, causal=False):
        super(BasicResBlock, self).__init__()
        conv_groups =  1 if group_divisor is None else in_channels//group_divisor

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=0, groups=conv_groups, bias=False)
        self.conv_pw1 = nn.Conv2d(out_channels, out_channels, 1) if (group_divisor is not None and pw) else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, groups=conv_groups, bias=False)
        self.conv_pw2 = nn.Conv2d(out_channels, out_channels, 1) if (group_divisor is not None and pw) else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)
        self.act2 = nn.ReLU(inplace=True)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            padding=0, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, affine=True)
            )
        
        # add for streaming
        self.causal = causal
        self.stride_t = stride[-1] if isinstance(stride, tuple) else stride
        self.cache_frames = 2 + 2 * self.stride_t
        
    def forward(self, x):
        if self.causal:
            x_padded = F.pad(x, (2, 0, 1, 1))
        else:
            x_padded = F.pad(x, (1, 1, 1, 1))
        out = self.conv1(x_padded)
        out = self.conv_pw1(out)
        out = self.bn1(out)
        out = self.act1(out)

        if self.causal:
            out = F.pad(out, (2, 0, 1, 1))
        else:
            out = F.pad(out, (1, 1, 1, 1))
        out = self.conv2(out)
        out = self.conv_pw2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = out + self.shortcut(x)
        return out

    def streaming_forward(self, x, cache:StreamingCache, layer_name: str):
        t = x.size()[-1]

        prev_input_cache = cache.get_conv2d_cache(f"{layer_name}_BasicResBlock")
        if prev_input_cache is not None:
            x_padded = torch.cat([prev_input_cache, x], dim=-1)
        else:
            x_padded = x
        cache.set_conv2d_cache(f"{layer_name}_BasicResBlock", x_padded[..., -self.cache_frames:].detach())

        x_padded = F.pad(x_padded, (2, 0, 1, 1))
        out = self.conv1(x_padded)
        out = self.conv_pw1(out)
        out = self.bn1(out)
        out = self.act1(out)

        out = F.pad(out, (2, 0, 1, 1))
        out = self.conv2(out)
        out = self.conv_pw2(out)
        out = self.bn2(out)
        out = self.act2(out)

        out = out[:,:,:,-t//self.stride_t:]
        out = out + self.shortcut(x)
        return out, cache


class PosConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, causal=False):
        super(PosConv, self).__init__()
        self.causal = causal
        self.kernel_size = kernel_size

        self.pos_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0, groups=out_channels),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Conv1d(out_channels, out_channels, 1, 1)
            )
    
    def forward(self, x):
        if not self.causal:
            x_pad = F.pad(x, (self.kernel_size//2, self.kernel_size//2))
        else:
            x_pad = F.pad(x, (self.kernel_size//2 * 2, 0))
        out = self.pos_conv(x_pad)
        return out


# ======================================= MECT block =============================================

class MECT_block(nn.Module):
    def __init__(self, cnn_num, in_channels, out_channels, in_fdims, hidden_dims, group_divisor, stride=1, moe=True, 
            moe_type="local",
            num_experts_shard=2,
            num_experts_token=2,
            top_k_shard=None,
            top_k_token=None,
            shard_scale=1.0, expansion=1,
            causal=False
        ):
        super().__init__()
        self.scale_f = stride[0] if isinstance(stride, tuple) else stride
        self.scale_t = stride[-1] if isinstance(stride, tuple) else stride
        self.in_channels = in_channels
        self.causal = causal
        
        # resnet block
        self.expansion = expansion
        self.cnn_block = self._make_layer(BasicResBlock, int(out_channels * expansion), cnn_num, group_divisor, stride=stride)

        if expansion > 1:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, in_channels * self.scale_f, kernel_size=1, stride=1),
                nn.BatchNorm2d(in_channels * self.scale_f)
            )
        else:
            self.reduce_conv = nn.Identity()

        # project for transformer
        self.fcdims = in_fdims * out_channels // self.scale_f
        self.reduce_project = nn.Sequential(
            nn.Conv1d(self.fcdims, hidden_dims, 1),
            LayerNorm(hidden_dims, eps=1e-6, data_format="channels_first")
        )

        # positional embedding
        self.pos_conv = PosConv(hidden_dims, hidden_dims, kernel_size=113, causal=causal)

        self.transformer = TransformerEncoderLayer(
            n_state=hidden_dims, 
            n_mlp=hidden_dims, 
            n_head=4, 
            moe=moe, 
            moe_type=moe_type,
            num_experts_shard=num_experts_shard,
            num_experts_token=num_experts_token,
            top_k_shard=top_k_shard,
            top_k_token=top_k_token,
            shard_scale=shard_scale
        )

        self.expansion_project = nn.Sequential(
            nn.Conv1d(hidden_dims, self.fcdims, 1),
        )

        self.up_sample = nn.Upsample(scale_factor=self.scale_t, mode='nearest')

        # calculate the receptive field of the pos_conv layer
        self.pos_conv_receptive_field = 113 - 1

    def _make_layer(self, block, out_channels, num_blocks, group_divisor, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, group_divisor, stride, causal=self.causal))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    
    def forward(self, x):
        t = x.size()[-1]
        out = self.cnn_block(x)
        out = self.reduce_conv(out)
        b, c, f, _ = out.size()

        x2 = out.permute((0, 2, 1, 3)).reshape((b, c*f, -1))
        out = self.reduce_project(x2)

        out = out + self.pos_conv(out)
        out, _ = self.transformer(out, causal=self.causal)
        out = self.expansion_project(out)
        out = out + x2

        out = self.up_sample(out)[...,:t] + x.permute((0, 2, 1, 3)).reshape((b, c*f, -1)) # 上采样后保持和输入shape一致
        out = out.reshape((b,f,c,out.shape[-1])).permute((0,2,1,3))
        return out
    
    def streaming_forward(self, x: torch.Tensor, cache: StreamingCache, layer_name: str):
        t = x.size()[-1]

        # cnn streaming
        x_padded = x
        for block_idx, block in enumerate(self.cnn_block):
            x_padded, cache = block.streaming_forward(x_padded, cache, f"{layer_name}_{block_idx}")

        out = self.reduce_conv(x_padded)
        out_t = t//self.scale_t

        b, c, f, _ = out.size()

        x2 = out.permute((0, 2, 1, 3)).reshape((b, c*f, -1))
        out = self.reduce_project(x2)

        # pos_conv streaming
        pos_cache_key = f"{layer_name}_pos_input"
        prev_pos_cache = cache.get_pos_cache(pos_cache_key)

        if prev_pos_cache is not None:
            out_padded = torch.cat([prev_pos_cache, out], dim=-1)
        else:
            out_padded = out

        cache_frames = min(self.pos_conv_receptive_field, out_padded.size(-1))
        cache.set_pos_cache(pos_cache_key, out_padded[..., -cache_frames:].detach())

        out = out + self.pos_conv(out_padded)[:,:,-out_t:] # cache 后截断当前chunk该有的长度

        # Transformer KV Cache
        prev_kv_cache = cache.get_kv_cache(f"{layer_name}_kv")
        out, new_kv_cache = self.transformer(out, use_cache=True, kv_cache=prev_kv_cache, causal=True)
        cache.set_kv_cache(f"{layer_name}_kv", new_kv_cache[0], new_kv_cache[1])

        out = self.expansion_project(out)
        out = out + x2

        out = self.up_sample(out)[...,:t] + x.permute((0, 2, 1, 3)).reshape((b, c*f, -1)) # 上采样后保持和输入shape一致
        out = out.reshape((b,f,c,out.shape[-1])).permute((0,2,1,3))
        return out, cache


class DenseWeight(nn.Module):
    def __init__(self, input_num, hidden_dim):
        super().__init__()
        self.inputs_weights = nn.Parameter(torch.zeros(1, input_num, hidden_dim, 1), requires_grad=True)

    def forward(self, xs):
        w = F.softmax(self.inputs_weights, dim=1)
        x = (w * xs).sum(dim=1)
        return x

# =============================== MECT =================================
class MECT(nn.Module):
    def __init__(
        self,
        in_channels,
        feat_dim,
        block1=[(16,80,16,128,4,1)] * 3,
        block2=[(16,80,32,128,4,2)] + [(32,40,32,128,4,1)]*2,
        block3=[(32,40,64,128,4,2)] + [(64,20,64,128,4,1)]*2,
        block4=[(64,20,128,128,4,2)] + [(128,10,128,128,4,1)]*2,
        moe_type="local",
        num_experts_shard=2,
        num_experts_token=2,
        top_k_shard=None,
        top_k_token=None,
        shard_scale=1.0,
        causal=False
    ):
        super().__init__()
        if not causal:
            self.head = nn.Sequential(
                nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels, affine=True),
            )
        else:
            self.head = nn.Sequential(
                nn.ConstantPad2d((2, 0, 1, 1), 0),
                nn.Conv2d(1, in_channels, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(in_channels, affine=True),
            )
        
        self.moe_type=moe_type
        self.num_experts_shard=num_experts_shard
        self.num_experts_token=num_experts_token
        self.top_k_shard=top_k_shard
        self.top_k_token=top_k_token
        self.shard_scale=shard_scale
        self.causal = causal

        self.in_channels = in_channels
        self.feat_dim = feat_dim
        self.channels = in_channels*feat_dim
        self.mect_block1 = self._make_layer(MECT_block, block1)
        self.mect_block2 = self._make_layer(MECT_block, block2)
        self.mect_block3 = self._make_layer(MECT_block, block3)
        self.mect_block4 = self._make_layer(MECT_block, block4)

        self.dense3 = DenseWeight(4, self.channels)


    def _make_layer(self, block, block_cfg):
        layers = []
        for cnn_num, out_channels, hidden_dims, group_divisor, stride, moe, expansion in block_cfg:
            layers.append(
                block(cnn_num, self.in_channels, out_channels, self.feat_dim, hidden_dims, group_divisor, stride=stride, moe=moe, 
                    moe_type=self.moe_type,
                    num_experts_shard=self.num_experts_shard,
                    num_experts_token=self.num_experts_token,
                    top_k_shard=self.top_k_shard,
                    top_k_token=self.top_k_token,
                    shard_scale=self.shard_scale, expansion=expansion, causal=self.causal
                )
            )
            scale_f = stride[0] if isinstance(stride, tuple) else stride
            self.in_channels = out_channels
            self.feat_dim = int(self.feat_dim // scale_f)
        return nn.Sequential(*layers)

    def forward(self, x, device_labels=None):
        x = x.unsqueeze(1)
        x = self.head(x)

        x = self.mect_block1(x)
        feat_1 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))

        x = self.mect_block2(x)
        feat_2 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))

        x = self.mect_block3(x)
        feat_3 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))

        x = self.mect_block4(x)
        feat_4 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))

        x = self.dense3(torch.cat([feat_1.unsqueeze(1), feat_2.unsqueeze(1), feat_3.unsqueeze(1), feat_4.unsqueeze(1)], dim=1))
        return x


    def get_out_dim(self):
        return self.channels

    def streaming_forward(self, x: torch.Tensor, cache: StreamingCache, device_labels=None):
        x = x.unsqueeze(1)
        t = x.size()[-1]

        # [streaming head]
        prev_input_cache = cache.get_conv2d_cache("head_cnn_input")
        if prev_input_cache is not None:
            x_padded = torch.cat([prev_input_cache, x], dim=-1)
        else:
            x_padded = x
        cache.set_conv2d_cache("head_cnn_input", x_padded[..., -2:].detach())

        x = self.head(x_padded)[:,:,:,-t:]
        
        # Block 1
        for block_idx, block in enumerate(self.mect_block1):
            x, cache = block.streaming_forward(x, cache, f"block1_{block_idx}")
        feat_1 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))
        
        # Block 2
        for block_idx, block in enumerate(self.mect_block2):
            x, cache = block.streaming_forward(x, cache, f"block2_{block_idx}")
        feat_2 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))
        
        # # Block 3
        for block_idx, block in enumerate(self.mect_block3):
            x, cache = block.streaming_forward(x, cache, f"block3_{block_idx}")
        feat_3 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))
        
        # Block 4
        for block_idx, block in enumerate(self.mect_block4):
            x, cache = block.streaming_forward(x, cache, f"block4_{block_idx}")
        feat_4 = x.permute((0, 2, 1, 3)).reshape(x.size(0), -1, x.size(-1))
        
        x = self.dense3(torch.cat([
            feat_1.unsqueeze(1), 
            feat_2.unsqueeze(1), 
            feat_3.unsqueeze(1), 
            feat_4.unsqueeze(1)
        ], dim=1))
        
        return x, cache

    def streaming_inference_test(self, x, chunk_size):
        total_time = x.size()[-1]
        cache = self.create_streaming_cache()
        stream_outputs = []
        for i in range(0, total_time, chunk_size):
            chunk = x[..., i:i+chunk_size]
            if chunk.size()[-1] < chunk_size:
                continue
            out_chunk, cache = self.streaming_forward(chunk, cache)
            stream_outputs.append(out_chunk)
        streaming_output = torch.cat(stream_outputs, dim=-1)
        return streaming_output

    
    def create_streaming_cache(self) -> StreamingCache:
        return StreamingCache()


class MECT_A1(MECT):
    def __init__(self, config={}):
        in_channels=16
        feat_dim=80
        hidden_dim=64
        conv_group=1
        super().__init__(
            in_channels=in_channels,
            feat_dim=feat_dim,
            block1=[(2, 16, hidden_dim, conv_group, 1, True, 4)] * 2,
            block2=[(2, 32, hidden_dim, conv_group, (2,1), True, 2)] + [(2, 32, hidden_dim, conv_group, (1,2), True, 2)]*3,
            block3=[(2, 64, hidden_dim, conv_group, (2,1), True, 2)] + [(2, 64, hidden_dim, conv_group, (1,2), True, 2)]*4,
            block4=[(2, 128, hidden_dim, conv_group, (2,1), True, 1)] + [(1, 128, hidden_dim, conv_group, (1,2), True, 1)]*2,
            moe_type=config.get("moe_type", "token"),
            num_experts_shard=config.get("num_experts_shard", 2),
            num_experts_token=config.get("num_experts_token", 4),
            top_k_shard=config.get("top_k_shard", None),
            top_k_token=config.get("top_k_token", None),
            shard_scale=config.get("shard_scale", 1.0),
            causal=config.get("causal", False)
        )


class MECT_A2(MECT):
    def __init__(self, config={}):
        in_channels=16
        feat_dim=80
        hidden_dim=64
        conv_group=1
        super().__init__(
            in_channels=in_channels,
            feat_dim=feat_dim,
            block1=[(3, 16, hidden_dim, conv_group, 1, True, 4)] * 2,
            block2=[(3, 32, hidden_dim, conv_group, (2,1), True, 2)] + [(3, 32, hidden_dim, conv_group, (1,2), True, 2)]*3,
            block3=[(3, 64, hidden_dim, conv_group, (2,1), True, 2)] + [(3, 64, hidden_dim, conv_group, (1,2), True, 2)]*4,
            block4=[(3, 128, hidden_dim, conv_group, (2,1), True, 1)] + [(2, 128, hidden_dim, conv_group, (1,2), True, 1)]*2,
            moe_type=config.get("moe_type", "token"),
            num_experts_shard=config.get("num_experts_shard", 2),
            num_experts_token=config.get("num_experts_token", 4),
            top_k_shard=config.get("top_k_shard", None),
            top_k_token=config.get("top_k_token", None),
            shard_scale=config.get("shard_scale", 1.0),
            causal=config.get("causal", False)
        )


class MECT_B1(MECT):
    def __init__(self, config={}):
        in_channels=32
        feat_dim=80
        hidden_dim=64
        conv_group=1
        super().__init__(
            in_channels=in_channels,
            feat_dim=feat_dim,
            block1=[(2, 32, hidden_dim, conv_group, 1, True, 4)] * 2,
            block2=[(2, 64, hidden_dim, conv_group, (2,1), True, 2)] + [(2, 64, hidden_dim, conv_group, (1,2), True, 2)]*3,
            block3=[(2, 128, hidden_dim, conv_group, (2,1), True, 2)] + [(2, 128, hidden_dim, conv_group, (1,2), True, 2)]*4,
            block4=[(2, 256, hidden_dim, conv_group, (2,1), True, 1)] + [(1, 256, hidden_dim, conv_group, (1,2), True, 1)]*2,
            moe_type=config.get("moe_type", "token"),
            num_experts_shard=config.get("num_experts_shard", 2),
            num_experts_token=config.get("num_experts_token", 4),
            top_k_shard=config.get("top_k_shard", None),
            top_k_token=config.get("top_k_token", None),
            shard_scale=config.get("shard_scale", 1.0),
            causal=config.get("causal", False)
        )


class MECT_B2(MECT):
    def __init__(self, config={}):
        in_channels=32
        feat_dim=80
        hidden_dim=64
        conv_group=1
        super().__init__(
            in_channels=in_channels,
            feat_dim=feat_dim,
            block1=[(3, 32, hidden_dim, conv_group, 1, True, 4)] * 2,
            block2=[(3, 64, hidden_dim, conv_group, (2,1), True, 2)] + [(3, 64, hidden_dim, conv_group, (1,2), True, 2)]*3,
            block3=[(3, 128, hidden_dim, conv_group, (2,1), True, 2)] + [(3, 128, hidden_dim, conv_group, (1,2), True, 2)]*4,
            block4=[(3, 256, hidden_dim, conv_group, (2,1), True, 1)] + [(2, 256, hidden_dim, conv_group, (1,2), True, 1)]*2,
            moe_type=config.get("moe_type", "token"),
            num_experts_shard=config.get("num_experts_shard", 2),
            num_experts_token=config.get("num_experts_token", 4),
            top_k_shard=config.get("top_k_shard", None),
            top_k_token=config.get("top_k_token", None),
            shard_scale=config.get("shard_scale", 1.0),
            causal=config.get("causal", False)
        )

class MECT_B2_Causal(MECT):
    def __init__(self, config={}):
        in_channels=32
        feat_dim=80
        hidden_dim=64
        conv_group=1
        super().__init__(
            in_channels=in_channels,
            feat_dim=feat_dim,
            block1=[(3, 32, hidden_dim, conv_group, 1, True, 4)] * 2,
            block2=[(3, 64, hidden_dim, conv_group, (2,1), True, 2)] + [(3, 64, hidden_dim, conv_group, (1,2), True, 2)]*3,
            block3=[(3, 128, hidden_dim, conv_group, (2,1), True, 2)] + [(3, 128, hidden_dim, conv_group, (1,2), True, 2)]*4,
            block4=[(3, 256, hidden_dim, conv_group, (2,1), True, 1)] + [(2, 256, hidden_dim, conv_group, (1,2), True, 1)]*2,
            moe_type=config.get("moe_type", "token"),
            num_experts_shard=config.get("num_experts_shard", 2),
            num_experts_token=config.get("num_experts_token", 4),
            top_k_shard=config.get("top_k_shard", None),
            top_k_token=config.get("top_k_token", None),
            shard_scale=config.get("shard_scale", 1.0),
            causal=config.get("causal", True)
        )
