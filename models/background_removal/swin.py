# Swin backbone adapted from ZhengPeng7/BiRefNet and Microsoft/Swin-Transformer.
# MIT licenses: LICENSE_BIREFNET and LICENSE_SWIN. Inference-only WanGP adaptation.
import torch
from torch import nn
from torch.nn import functional as F
from shared import attention_kit
from shared.kernels import ops
from .common import check_abort


def partition(x, size):
    b, h, w, c = x.shape
    return x.view(b, h // size, size, w // size, size, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, size, size, c)


def reverse(windows, size, h, w):
    return windows.view(-1, h // size, w // size, size, size, windows.shape[-1]).permute(0, 1, 3, 2, 4, 5).reshape(-1, h, w, windows.shape[-1])


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return ops.mlp([x], self.fc1, self.fc2, 'gelu_erf')


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, size=12):
        super().__init__()
        self.heads, self.size = heads, size
        self.relative_position_bias_table = nn.Parameter(torch.empty((2 * size - 1) ** 2, heads))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        # Geometry is generated on CPU; it is not a checkpoint tensor.
        self.register_buffer('relative_position_index', torch.empty(0, dtype=torch.int64), persistent=False)
        self.reset_index()

    def reset_index(self):
        size = self.size
        coords = torch.stack(torch.meshgrid(torch.arange(size, device='cpu'), torch.arange(size, device='cpu'), indexing='ij')).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        index = (relative[..., 0] + size - 1) * (2 * size - 1) + relative[..., 1] + size - 1
        self.relative_position_index = index

    def forward(self, x, mask=None):
        batch, tokens, channels = x.shape
        bias = self.relative_position_bias_table[self.relative_position_index.flatten()].view(tokens, tokens, self.heads).permute(0, 2, 1)
        bias = bias.unsqueeze(0)
        if mask is not None:
            bias = (bias.unsqueeze(0) + mask[:, :, None, :].unsqueeze(0)).expand(batch // mask.shape[0], -1, -1, -1, -1).reshape(batch, tokens, self.heads, tokens)
        inputs, x = [x], None
        return attention_kit.qkv_attention(inputs, self.q, self.k, self.v, self.heads, channels // self.heads,
                                           lambda q, k, group: None, attention_mask=bias, split_heads=False, out_proj=self.proj)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, heads, shift, size=12):
        super().__init__()
        self.size, self.shift = size, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x, h, w, mask):
        check_abort()
        residual = x
        value = self.norm1(x).view(x.shape[0], h, w, -1)
        value = F.pad(value, (0, 0, 0, (-w) % self.size, 0, (-h) % self.size))
        hp, wp = value.shape[1:3]
        if self.shift:
            value = torch.roll(value, (-self.shift, -self.shift), (1, 2))
        windows = partition(value, self.size).view(-1, self.size ** 2, value.shape[-1])
        del value
        windows = self.attn(windows, mask if self.shift else None)
        value = reverse(windows, self.size, hp, wp)
        if self.shift:
            value = torch.roll(value, (self.shift, self.shift), (1, 2))
        x = residual + value[:, :h, :w].reshape_as(residual)
        x = x + self.mlp(self.norm2(x))
        check_abort()
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x, h, w):
        x = x.view(x.shape[0], h, w, -1)
        if h % 2 or w % 2:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), -1)
        return self.reduction(self.norm(x.reshape(x.shape[0], -1, x.shape[-1])))


class BasicLayer(nn.Module):
    def __init__(self, dim, heads, depth, downsample, size=12):
        super().__init__()
        self.size = size
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim, heads, 0 if i % 2 == 0 else size // 2, size) for i in range(depth)])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, h, w):
        size = self.size
        hp, wp = ((h + size - 1) // size) * size, ((w + size - 1) // size) * size
        regions = torch.zeros((1, hp, wp, 1), device=x.device, dtype=x.dtype)
        slices = (slice(0, -size), slice(-size, -(size // 2)), slice(-(size // 2), None))
        for i, hs in enumerate(slices):
            for j, ws in enumerate(slices):
                regions[:, hs, ws] = 3 * i + j
        windows = partition(regions, size).view(-1, size * size)
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100).masked_fill(mask == 0, 0)
        for block in self.blocks:
            x = block(x, h, w, mask)
        return x, self.downsample(x, h, w) if self.downsample is not None else x


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 192, 4, 4)
        self.norm = nn.LayerNorm(192)

    def forward(self, x):
        x = F.pad(x, (0, (-x.shape[-1]) % 4, 0, (-x.shape[-2]) % 4))
        x = self.proj(x)
        h, w = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed()
        self.layers = nn.ModuleList([BasicLayer(192 * 2 ** i, heads, depth, i < 3) for i, (heads, depth) in enumerate(zip((6, 12, 24, 48), (2, 2, 18, 2)))])
        for i in range(4):
            self.add_module(f'norm{i}', nn.LayerNorm(192 * 2 ** i))

    def forward(self, x):
        x, h, w = self.patch_embed(x)
        outputs = []
        for i, layer in enumerate(self.layers):
            result, x = layer(x, h, w)
            outputs.append(getattr(self, f'norm{i}')(result).view(x.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous())
            h, w = (h + 1) // 2, (w + 1) // 2
            check_abort()
        return outputs


def split_map():
    return {'attn.qkv': {'mapped_modules': ['attn.q', 'attn.k', 'attn.v']}}
