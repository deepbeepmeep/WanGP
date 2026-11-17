from shared.utils.phase_progress import vae_decoding_progress, set_phase_status
from dataclasses import dataclass

import torch
from shared.image_vae import blocks, tiles, tiling
from shared.utils.conv_bands import band_convs, downsample_bands
from shared.kernels import vae_ops
from einops import rearrange
from torch import Tensor, nn


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


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
        """Consume [input]; scratch: see shared.image_vae.blocks.resnet_block."""
        shortcut = self.nin_shortcut if self.in_channels != self.out_channels else None
        return blocks.resnet_block(x_list, self.norm1, self.conv1, self.norm2, self.conv2, shortcut, scratch)


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
        return blocks.upsample(x_list, self.conv)


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
        h_list, h = [h], None
        h = blocks.norm_conv(h_list, self.norm_out, self.conv_out)
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
        h_list, h = [h], None
        h = blocks.norm_conv(h_list, self.norm_out, self.conv_out)
        return h


class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams, sample_z: bool = False):
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
        self.reg = DiagonalGaussian(sample=sample_z)
        band_convs(self)  # large 3x3 convolutions by bands of rows: no full-image cuDNN workspace

        self.scale_factor = params.scale_factor
        self.shift_factor = params.shift_factor

    # measured VRAM (shared.image_vae.tiling): BF16 autocast, the larger of the routes with and without the Triton convolutions, + 10 %
    COSTS = {"weights": 0.16, "decode": [(1.05, 0.75), (4.2, 2.8)]}

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        """The decoding tile size in pixels (0: no tiling): the largest within the VAE Tiling setting's VRAM budget, no tiling when
        the whole picture fits (tiling.choose_tile). Reference pictures are encoded whole."""
        # (tiles of a GroupNorm VAE each take their own statistics: 4 levels off the untiled picture on average at 2048 and 1024, more below)
        return tiling.choose_tile(vae_config, device_mem_capacity, AutoEncoder.COSTS, output_height, output_width, (2048, 1024))

    def encode(self, x: Tensor) -> Tensor:
        set_phase_status("VAE Encoding")
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor, tile_size=0) -> Tensor:
        """tile_size (pixels, 0: none): the picture by blended tiles of that size when it is larger (shared.image_vae.tiles)."""
        with vae_decoding_progress(tiles.count(*z.shape[-2:], tile_size // 8), self.decoder):
            z = z / self.scale_factor + self.shift_factor
            return tiles.tiled(self.decoder, z, tile_size // 8, 8)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))
