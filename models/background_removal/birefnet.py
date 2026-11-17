# Inference adaptation of ZhengPeng7/BiRefNet (MIT; see LICENSE_BIREFNET).
# Swin-L, multiscale input and deformable ASPP decoder of the General release.
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d
from mmgp import runtime
from .swin import SwinTransformer, split_map
from .common import check_abort


def resize(x, reference):
    return F.interpolate(x, size=reference.shape[-2:], mode='bilinear', align_corners=True)


def patches(x, reference):
    b, c, h, w = x.shape
    ph, pw = reference.shape[-2:]
    return x.reshape(b, c, h // ph, ph, w // pw, pw).permute(0, 1, 2, 4, 3, 5).reshape(b, -1, ph, pw)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size ** 2, kernel_size, padding=padding)
        self.modulator_conv = nn.Conv2d(in_channels, kernel_size ** 2, kernel_size, padding=padding)
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        for module in (self.offset_conv, self.modulator_conv, self.regular_conv):
            module._lock_dtype = torch.float16

    def forward(self, x):
        dtype = x.dtype
        # torchvision's deformable operator has no BF16 implementation.
        with torch.autocast(x.device.type, enabled=False):
            x = x.to(self.regular_conv.weight.dtype)
            offset = self.offset_conv(x)
            modulator = self.modulator_conv(x).sigmoid() * 2
            # This convolution calls the operator directly, not regular_conv.forward.
            runtime.ensure_loaded(self.regular_conv)
            return deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias,
                                 padding=self.padding, mask=modulator).to(dtype)


class ASPPModule(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.atrous_conv = DeformableConv2d(channels, 256, kernel, kernel // 2)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.aspp1 = ASPPModule(channels, 1)
        self.aspp_deforms = nn.ModuleList([ASPPModule(channels, kernel) for kernel in (1, 3, 7)])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(1280, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        values = [self.aspp1(x)]
        for block in self.aspp_deforms:
            values.append(block(x))
            check_abort()
        values.append(resize(self.global_avg_pool(x), x))
        return self.relu(self.bn1(self.conv1(torch.cat(values, dim=1))))


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable()
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn_out(self.conv_out(self.dec_att(self.relu_in(self.bn_in(self.conv_in(x))))))
        check_abort()
        return x


class BasicLatBlk(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (3072, 1536, 768, 384)
        for level, inputs, outputs in ((5, 3072, 384), (4, 768, 384), (3, 192, 192), (2, 48, 96), (1, 3, 48)):
            setattr(self, f'ipt_blk{level}', SimpleConvs(inputs, outputs))
        self.decoder_block4 = BasicDecBlk(3072 + 384, 1536)
        self.decoder_block3 = BasicDecBlk(1536 + 384, 768)
        self.decoder_block2 = BasicDecBlk(768 + 192, 384)
        self.decoder_block1 = BasicDecBlk(384 + 96, 192)
        self.conv_out1 = nn.Sequential(nn.Conv2d(192 + 48, 1, 1))
        for level, channel in zip((4, 3, 2), channels[1:]):
            setattr(self, f'lateral_block{level}', BasicLatBlk(channel))
            setattr(self, f'gdt_convs_{level}', nn.Sequential(nn.Conv2d(channel, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True)))
            setattr(self, f'gdt_convs_attn_{level}', nn.Sequential(nn.Conv2d(16, 1, 1)))

    def forward(self, image, features):
        value = features[-1]
        for level in (4, 3, 2, 1):
            image_features = getattr(self, f'ipt_blk{level + 1}')(patches(image, value))
            value = getattr(self, f'decoder_block{level}')(torch.cat((value, image_features), dim=1))
            if level > 1:
                attention = getattr(self, f'gdt_convs_attn_{level}')(getattr(self, f'gdt_convs_{level}')(value)).sigmoid()
                value = value * attention
                reference = features[level - 2]
                value = resize(value, reference) + getattr(self, f'lateral_block{level}')(reference)
            check_abort()
        value = resize(value, image)
        return self.conv_out1(torch.cat((value, self.ipt_blk1(image)), dim=1))


class BiRefNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = SwinTransformer()
        self.squeeze_module = nn.Sequential(BasicDecBlk(5760, 3072))
        self.decoder = Decoder()

    def forward(self, image):
        features = self.bb(image)
        small = self.bb(F.interpolate(image, scale_factor=.5, mode='bilinear', align_corners=True))
        features = [torch.cat((full, resize(half, full)), dim=1) for full, half in zip(features, small)]
        del small
        context = torch.cat([resize(value, features[-1]) for value in features[:-1]] + [features[-1]], dim=1)
        features[-1] = self.squeeze_module(context)
        del context
        check_abort()
        return self.decoder(image, features)

    def split_linear_modules(self):
        return split_map()
