# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""PyTorch forms of the InsightFace buffalo_l networks (https://github.com/deepinsight/insightface, code MIT: LICENSE_INSIGHTFACE;
model terms: NOTICE), laid out as their released graphs: the SCRFD-10G detector, the 2D-106 landmark MobileNet and the IResNet-50
ArcFace (w600k_r50). Inputs are uint8 RGB batches (N, 3, H, W): each network normalizes in FP32, then runs in its weights' dtype."""
import torch
import torch.nn.functional as F
from torch import nn


def _conv(cin, cout, kernel=3, stride=1, groups=1, bias=True):
    return nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=bias)


def _normalize(x, mean, std, dtype):
    return ((x.float() - mean) / std).to(dtype)


class _Block(nn.Module):
    """SCRFD backbone block: two 3x3 convolutions, an average-pooled 1x1 shortcut when it downsamples."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1, self.conv2 = _conv(cin, cout, stride=stride), _conv(cout, cout)
        self.shortcut = _conv(cin, cout, 1) if stride > 1 else None

    def forward(self, x):
        out = self.conv2(F.relu(self.conv1(x)))
        return F.relu(out + (x if self.shortcut is None else self.shortcut(F.avg_pool2d(x, 2, 2, ceil_mode=True, count_include_pad=False))))


class _Head(nn.Module):
    """One SCRFD level: scores (2 anchors per position), box distances scaled by the level's learned factor, 5-point offsets."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([_conv(56, 80), _conv(80, 80), _conv(80, 80)])
        self.cls, self.reg, self.kps = _conv(80, 2), _conv(80, 8), _conv(80, 20)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        flat = lambda tensor, width: tensor.permute(2, 3, 0, 1).reshape(-1, width).float()
        return flat(self.cls(x), 1).sigmoid(), flat(self.reg(x) * self.scale, 4), flat(self.kps(x), 10)


class SCRFD(nn.Module):
    """SCRFD-10G with keypoints: per stride 8 / 16 / 32, the scores, box and keypoint distances of one image (FP32)."""
    strides = (8, 16, 32)

    def __init__(self):
        super().__init__()
        self.stem = nn.ModuleList([_conv(3, 28, stride=2), _conv(28, 28), _conv(28, 56)])
        stages = ((56, 56, 3, 1), (56, 88, 4, 2), (88, 88, 2, 2), (88, 224, 3, 2))
        self.stages = nn.ModuleList(nn.Sequential(*[_Block(cin if index == 0 else cout, cout, stride if index == 0 else 1) for index in range(count)]) for cin, cout, count, stride in stages)
        self.lateral = nn.ModuleList(_conv(channels, 56, 1) for channels in (88, 88, 224))
        self.fpn = nn.ModuleList(_conv(56, 56) for _ in range(3))
        self.downsample = nn.ModuleList(_conv(56, 56, stride=2) for _ in range(2))
        self.pafpn = nn.ModuleList(_conv(56, 56) for _ in range(2))
        self.heads = nn.ModuleList(_Head() for _ in self.strides)

    def forward(self, x):
        x = _normalize(x, 127.5, 128.0, self.stem[0].weight.dtype)
        for conv in self.stem:
            x = F.relu(conv(x))
        c2 = self.stages[1](self.stages[0](F.max_pool2d(x, 2, 2)))
        c3 = self.stages[2](c2)
        c4 = self.stages[3](c3)
        l2, l3, l4 = (conv(feature) for conv, feature in zip(self.lateral, (c2, c3, c4)))
        l3 = l3 + F.interpolate(l4, size=l3.shape[-2:], mode="nearest")
        l2 = l2 + F.interpolate(l3, size=l2.shape[-2:], mode="nearest")
        p2, p3, p4 = self.fpn[0](l2), self.fpn[1](l3), self.fpn[2](l4)
        p3 = p3 + self.downsample[0](p2)
        p4 = p4 + self.downsample[1](p3)
        outputs = [head(feature) for head, feature in zip(self.heads, (p2, self.pafpn[0](p3), self.pafpn[1](p4)))]
        return tuple(list(level) for level in zip(*outputs))


class _ConvBNPReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel, stride=1, groups=1):
        super().__init__(_conv(cin, cout, kernel, stride, groups, bias=False), nn.BatchNorm2d(cout, eps=1e-3), nn.PReLU(cout))


class Landmark106(nn.Module):
    """2D-106 landmarks of 192x192 face crops, in [-1, 1] crop coordinates (FP32)."""

    def __init__(self):
        super().__init__()
        layers = [_ConvBNPReLU(3, 16, 3, 2)]
        for cin, cout, stride in ((16, 32, 1), (32, 64, 2), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), *[(256, 256, 1)] * 5, (256, 512, 2), (512, 512, 1)):
            layers += [_ConvBNPReLU(cin, cin, 3, stride, cin), _ConvBNPReLU(cin, cout, 1)]
        self.features = nn.Sequential(*layers, _ConvBNPReLU(512, 64, 3, 2))
        self.fc = nn.Linear(576, 212)

    def forward(self, x):
        return self.fc(self.features(_normalize(x, 127.5, 128.0, self.fc.weight.dtype)).flatten(1)).float().reshape(len(x), -1, 2)


class _IRBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1, self.conv1, self.prelu, self.conv2 = nn.BatchNorm2d(cin), _conv(cin, cout), nn.PReLU(cout), _conv(cout, cout, stride=stride)
        self.downsample = _conv(cin, cout, 1, stride) if stride > 1 else None

    def forward(self, x):
        out = self.conv2(self.prelu(self.conv1(self.bn1(x))))
        return out + (x if self.downsample is None else self.downsample(x))


class IResNet50(nn.Module):
    """ArcFace IResNet-50 (w600k_r50): the 512-d identity embeddings of 112x112 aligned crops, unnormalized (FP32)."""

    def __init__(self):
        super().__init__()
        self.conv1, self.prelu = _conv(3, 64), nn.PReLU(64)
        self.layers = nn.Sequential(*[_IRBlock(cin if index == 0 else cout, cout, 2 if index == 0 else 1)
                                      for cin, cout, count in ((64, 64, 3), (64, 128, 4), (128, 256, 14), (256, 512, 3)) for index in range(count)])
        self.bn2, self.fc, self.features = nn.BatchNorm2d(512), nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.layers(self.prelu(self.conv1(_normalize(x, 127.5, 127.5, self.conv1.weight.dtype))))
        return self.features(self.fc(self.bn2(x).flatten(1))).float()
