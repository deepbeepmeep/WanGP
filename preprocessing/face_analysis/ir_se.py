# IR-SE50 ArcFace backbone, from facexlib (https://github.com/xinntao/facexlib, facexlib/recognition/arcface_arch.py, MIT: LICENSE_FACEXLIB).
# WanGP changes: the 50-layer IR-SE variant only, inference only (no dropout), FP32 normalization of a uint8 RGB input and
# the channel swap to the BGR order the checkpoint was trained on.
from collections import namedtuple

import torch
from torch.nn import AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d, Identity, Linear, MaxPool2d, Module, PReLU, ReLU, Sequential, Sigmoid


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel), Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth), SEModule(depth, 16))

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


class Backbone(Module):
    """facexlib's arcface model: Backbone(num_layers=50, mode='ir_se'); outputs L2-normalized 512-d embeddings (FP32)."""

    def __init__(self):
        super().__init__()
        blocks = [get_block(64, 64, 3), get_block(64, 128, 4), get_block(128, 256, 14), get_block(256, 512, 3)]
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512), Identity(), Flatten(), Linear(512 * 7 * 7, 512), BatchNorm1d(512))
        self.body = Sequential(*[bottleneck_IR_SE(bottleneck.in_channel, bottleneck.depth, bottleneck.stride) for block in blocks for bottleneck in block])

    def forward(self, x):
        # facexlib's input: RGB in [0, 1], normalized with mean and std 0.5, then reordered to BGR.
        x = ((x.float() / 255.0 - 0.5) / 0.5)[:, [2, 1, 0]].to(self.input_layer[0].weight.dtype)
        x = self.output_layer(self.body(self.input_layer(x))).float()
        return x / torch.norm(x, 2, 1, True)
