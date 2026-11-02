from shared.utils.phase_progress import vae_decoding_progress, set_phase_status
import math
from dataclasses import dataclass, field

import torch
from shared.kernels import vae_ops
from shared.utils.conv_bands import band_convs, conv_bands, downsample_bands
from .autoencoder import norm_swish
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AutoEncoderParamsFlux2:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = vae_ops.attention(q, k, v)  # by chunks of queries where SDPA has no memory-efficient kernel (BF16 on Turing)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_list, scratch=None):
        """Consume [input]. scratch, when supplied, borrows the disjoint hidden half of this level's row-interleaved storage."""
        x = x_list.pop()
        if vae_ops.image_norm_conv_supported(x, self.norm1):
            if x.stride(1) != 1:
                x = x.contiguous(memory_format=torch.channels_last)
            if (self.in_channels == 2 * self.out_channels and x.is_contiguous(memory_format=torch.channels_last)
                    and vae_ops.image_norm_conv_is_banded(x)):
                # Each original NHWC row becomes two shorter NHWC rows: conv1's hidden state and the shortcut. The last row
                # waits for the next strip to read its halo. No full-size normalized input or additional residual is needed.
                n, _, height, width = x.shape
                shape = (n, self.out_channels, height, width)
                strides = (x.stride(0), 1, x.stride(2), self.out_channels)
                hidden = x.as_strided(shape, strides)
                residual = x.as_strided(shape, strides, x.storage_offset() + width * self.out_channels)
                pending = None
                for start, stop, band in vae_ops.image_norm_conv_bands(x, self.norm1, self.conv1):
                    shortcut = self.nin_shortcut(x[:, :, start:stop])
                    if pending is not None:
                        row, h_row, r_row = pending
                        hidden[:, :, row:row+1].copy_(h_row)
                        residual[:, :, row:row+1].copy_(r_row)
                        pending = None
                        del h_row, r_row
                    end = stop if stop == height else stop - 1
                    if end < stop:
                        pending = (end, band[:, :, -1:].clone(), shortcut[:, :, -1:].clone())
                    hidden[:, :, start:end].copy_(band[:, :, :end-start])
                    residual[:, :, start:end].copy_(shortcut[:, :, :end-start])
                    del band, shortcut
                if scratch is not None:
                    scratch[:] = [hidden]
                x = residual
                h = hidden
            else:
                if self.in_channels != self.out_channels and scratch is not None:
                    scratch.clear()
                h = vae_ops.image_norm_conv([x], self.norm1, self.conv1, out=scratch[0] if scratch else None)
                if self.in_channels != self.out_channels:
                    x = self.nin_shortcut(x)
            h_list, h = [h], None
            return vae_ops.image_norm_conv(h_list, self.norm2, self.conv2, out=x, add=True)
        if scratch is not None:
            scratch.clear()
        x = x.contiguous()  # small PyTorch norms use NCHW; avoid their FP32 channels-last conversion
        h = norm_swish(self.norm1, x)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        h = self.conv1(h)
        h = norm_swish(self.norm2, h)
        h = self.conv2(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        return downsample_bands(x, self.conv)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_list):
        x = x_list.pop()
        if torch.is_autocast_enabled(x.device.type) and x.is_cuda:
            # Convert before expansion, and release the spent hidden half of a row-interleaved workspace.
            x = x.contiguous(memory_format=torch.channels_last)
        return conv_bands(x, self.conv, upsample=True)  # by bands of rows: no full-size upsampled input


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling (only the current activation is kept: each block releases its input once read)
        if torch.is_autocast_enabled(x.device.type) and x.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block_input, h = [h], None
                h = self.down[i_level].block[i_block](block_input)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1([h])
        h = self.mid.attn_1(h)
        h = self.mid.block_2([h])
        # end
        if vae_ops.image_norm_conv_supported(h, self.norm_out):
            h_list, h = [h], None
            h = vae_ops.image_norm_conv(h_list, self.norm_out, self.conv_out)
        else:
            h = norm_swish(self.norm_out, h)
            h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1([h])
        h = self.mid.attn_1(h)
        h = self.mid.block_2([h])

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        scratch = []  # the hidden half of a channel-reducing block's storage, reused by the following equal-width blocks
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                block_input, h = [h], None  # handoff: the block releases its input once read
                h = self.up[i_level].block[i_block](block_input, scratch=scratch)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                scratch.clear()
                h_list, h = [h], None
                h = self.up[i_level].upsample(h_list)

        # end
        scratch.clear()
        if vae_ops.image_norm_conv_supported(h, self.norm_out):
            h_list, h = [h], None
            h = vae_ops.image_norm_conv(h_list, self.norm_out, self.conv_out)
        else:
            h = norm_swish(self.norm_out, h)
            h = self.conv_out(h)
        return h


class AutoencoderKLFlux2(nn.Module):
    def __init__(self, params: AutoEncoderParamsFlux2):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        band_convs(self)  # large 3x3 convolutions by bands of rows: no full-image cuDNN workspace

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s.to(z) + m.to(z)

    def encode(self, x: Tensor) -> Tensor:
        set_phase_status("VAE Encoding")
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def pre_decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return z
            
    def decode(self, z: Tensor) -> Tensor:

        with vae_decoding_progress(1, self.decoder):
            dec = self.decoder(z)
            return dec
