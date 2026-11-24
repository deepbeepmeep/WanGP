# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
#
# Inference-only ERes2Net of 3D-Speaker (speakerlab/models/eres2net/ERes2Net.py, fusion.py, pooling_layers.py)
# for iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k (m_channels 32, 512-dimensional embeddings, TSTP pooling).
# Res2Net is adapted there from wenet-e2e/wespeaker (Apache-2.0). WanGP changes: every BatchNorm folded into the
# convolution before it (the checkpoint stores the folded weights), training code removed.
"""ERes2Net speaker embedding: (batch, frames, 80) Kaldi fbank -> (batch, 512)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _relu(x):
    return F.hardtanh(x, 0.0, 20.0)


class AFF(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 2, channels // r, 1)
        self.conv2 = nn.Conv2d(channels // r, channels, 1)

    def forward(self, x, ds_y):
        att = 1.0 + torch.tanh(self.conv2(F.silu(self.conv1(torch.cat((x, ds_y), dim=1)))))
        return x * att + ds_y * (2.0 - att)


class BasicBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1, fuse=False, base_width=32, scale=2):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        self.width, self.scale = width, scale
        self.conv1 = nn.Conv2d(in_planes, width * scale, 1, stride=stride)
        self.convs = nn.ModuleList(nn.Conv2d(width, width, 3, padding=1) for _ in range(scale))
        self.fuse_models = nn.ModuleList(AFF(width) for _ in range(scale - 1)) if fuse else None
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, 1)
        self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride) if stride != 1 or in_planes != planes * self.expansion else None

    def forward(self, x):
        spx = torch.split(_relu(self.conv1(x)), self.width, 1)
        outs, sp = [], spx[0]
        for i, conv in enumerate(self.convs):
            if i:
                sp = self.fuse_models[i - 1](sp, spx[i]) if self.fuse_models is not None else sp + spx[i]
            sp = _relu(conv(sp))
            outs.append(sp)
        out = self.conv3(torch.cat(outs, 1))
        return _relu(out + (x if self.shortcut is None else self.shortcut(x)))


class ERes2Net(nn.Module):
    def __init__(self, num_blocks=(3, 4, 6, 3), m_channels=32, feat_dim=80, embedding_size=512):
        super().__init__()
        self.in_planes = m_channels
        stats_dim = int(feat_dim / 8) * m_channels * 8 * BasicBlock.expansion
        self.conv1 = nn.Conv2d(1, m_channels, 3, padding=1)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], 1, False)
        self.layer2 = self._make_layer(m_channels * 2, num_blocks[1], 2, False)
        self.layer3 = self._make_layer(m_channels * 4, num_blocks[2], 2, True)
        self.layer4 = self._make_layer(m_channels * 8, num_blocks[3], 2, True)
        self.layer1_downsample = nn.Conv2d(m_channels * 2, m_channels * 4, 3, stride=2, padding=1, bias=False)
        self.layer2_downsample = nn.Conv2d(m_channels * 4, m_channels * 8, 3, stride=2, padding=1, bias=False)
        self.layer3_downsample = nn.Conv2d(m_channels * 8, m_channels * 16, 3, stride=2, padding=1, bias=False)
        self.fuse_mode12 = AFF(m_channels * 4)
        self.fuse_mode123 = AFF(m_channels * 8)
        self.fuse_mode1234 = AFF(m_channels * 16)
        self.seg_1 = nn.Linear(stats_dim * 2, embedding_size)

    def _make_layer(self, planes, num_blocks, stride, fuse):
        layers = []
        for block_stride in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, block_stride, fuse))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x.transpose(1, 2).unsqueeze(1)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        fuse12 = self.fuse_mode12(out2, self.layer1_downsample(out1))
        out3 = self.layer3(out2)
        fuse123 = self.fuse_mode123(out3, self.layer2_downsample(fuse12))
        out4 = self.layer4(out3)
        fuse1234 = self.fuse_mode1234(out4, self.layer3_downsample(fuse123)).float()
        # TSTP: mean and standard deviation over time, as in x-vectors
        stats = torch.cat((fuse1234.mean(dim=-1).flatten(1), torch.sqrt(fuse1234.var(dim=-1) + 1e-8).flatten(1)), 1)
        return self.seg_1(stats.to(self.seg_1.weight.dtype))
