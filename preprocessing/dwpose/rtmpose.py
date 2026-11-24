# RTMPose-L whole-body estimator (DWPose dw-ll_ucoco_384), inference only. Derived from OpenMMLab mmpose / mmdet (Apache-2.0, see
# LICENSE in this folder): mmdet/models/backbones/cspnext.py, mmdet/models/layers/csp_layer.py, mmpose/models/heads/coord_cls_heads/
# rtmcc_head.py, mmpose/models/utils/rtmcc_block.py. Checkpoint: IDEA-Research DWPose (Apache-2.0).
# WanGP changes: batch norms folded into the convolutions (as in the released DWPose export), the RTMCC block reduced to the released
# configuration (self-attention, no relative bias, no positional encoding, no dropout), training code removed.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, cin, cout, k, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups)

    def forward(self, x):
        return F.silu(self.conv(x), inplace=True)


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.depthwise_conv, self.pointwise_conv = ConvModule(cin, cin, k, groups=cin), ConvModule(cin, cout, 1)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    def __init__(self, channels, add_identity):
        super().__init__()
        self.conv1, self.conv2, self.add_identity = ConvModule(channels, channels, 3), DepthwiseSeparableConvModule(channels, channels, 5), add_identity

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.add_identity else y


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x * F.hardsigmoid(self.fc(x.mean((2, 3), keepdim=True)))


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, num_blocks, add_identity):
        super().__init__()
        mid = cout // 2
        self.main_conv, self.short_conv, self.final_conv = ConvModule(cin, mid, 1), ConvModule(cin, mid, 1), ConvModule(2 * mid, cout, 1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid, add_identity) for _ in range(num_blocks)])
        self.attention = ChannelAttention(2 * mid)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        return self.final_conv(self.attention(torch.cat((x_main, x_short), 1)))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, kernel_sizes=(5, 9, 13)):
        super().__init__()
        mid = cin // 2
        self.conv1, self.conv2 = ConvModule(cin, mid, 1), ConvModule(mid * (len(kernel_sizes) + 1), cout, 1)
        self.poolings = nn.ModuleList(nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.poolings], 1))


class CSPNeXt(nn.Module):
    # P5 architecture, deepen and widen factors 1: (in, out, blocks, add_identity, use_spp) per stage.
    ARCH = ((64, 128, 3, True, False), (128, 256, 6, True, False), (256, 512, 6, True, False), (512, 1024, 3, False, True))

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvModule(3, 32, 3, 2), ConvModule(32, 32, 3), ConvModule(32, 64, 3))
        for i, (cin, cout, blocks, add_identity, use_spp) in enumerate(self.ARCH, 1):
            layers = [ConvModule(cin, cout, 3, 2)] + ([SPPBottleneck(cout, cout)] if use_spp else []) + [CSPLayer(cout, cout, blocks, add_identity)]
            setattr(self, f"stage{i}", nn.Sequential(*layers))

    def forward(self, x):
        return self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale, self.eps = dim ** -0.5, eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) * self.scale).clamp(min=self.eps) * self.g


class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class RTMCCBlock(nn.Module):
    """Gated attention unit over the keypoint tokens (RTMCC block, released self-attention configuration)."""
    def __init__(self, dim, expansion=2, s=128):
        super().__init__()
        self.e, self.s = dim * expansion, s
        self.ln, self.uv, self.o = ScaleNorm(dim), nn.Linear(dim, 2 * self.e + s, bias=False), nn.Linear(self.e, dim, bias=False)
        self.gamma, self.beta = nn.Parameter(torch.ones(2, s)), nn.Parameter(torch.zeros(2, s))
        self.res_scale = Scale(dim)

    def forward(self, x):
        u, v, base = torch.split(F.silu(self.uv(self.ln(x))), [self.e, self.e, self.s], dim=2)
        q, k = torch.unbind(base.unsqueeze(2) * self.gamma + self.beta, dim=2)
        kernel = torch.square(F.relu(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.s)))
        return self.res_scale(x) + self.o(u * torch.bmm(kernel, v))


class RTMCCHead(nn.Module):
    def __init__(self, in_channels=1024, keypoints=133, input_size=(288, 384), feature_size=(9, 12), hidden=256, split_ratio=2.0):
        super().__init__()
        flatten_dims = feature_size[0] * feature_size[1]
        self.final_layer = nn.Conv2d(in_channels, keypoints, 7, 1, 3)
        self.mlp = nn.Sequential(ScaleNorm(flatten_dims), nn.Linear(flatten_dims, hidden, bias=False))
        self.gau = RTMCCBlock(hidden)
        self.cls_x = nn.Linear(hidden, int(input_size[0] * split_ratio), bias=False)
        self.cls_y = nn.Linear(hidden, int(input_size[1] * split_ratio), bias=False)

    def forward(self, x):
        x = self.gau(self.mlp(self.final_layer(x).flatten(2)))
        return self.cls_x(x), self.cls_y(x)


class RTMPose(nn.Module):
    """Input: (B, 3, 384, 288) RGB-normalized crops (mean 123.675 / 116.28 / 103.53, std 58.395 / 57.12 / 57.375, applied to the
    channel order given). Output: SimCC x (B, 133, 576) and y (B, 133, 768) logits at half-pixel resolution."""
    input_size = (288, 384)

    def __init__(self):
        super().__init__()
        self.backbone, self.head = CSPNeXt(), RTMCCHead()

    def forward(self, x):
        return self.head(self.backbone(x))
