# YOLOX-L person detector, inference only. Derived from Megvii YOLOX (Apache-2.0, see LICENSE in this folder):
# https://github.com/Megvii-BaseDetection/YOLOX (yolox/models/darknet.py, network_blocks.py, yolo_pafpn.py, yolo_head.py).
# WanGP changes: batch norms folded into the convolutions (as in the released DWPose export), export-mode head only
# (raw box regressions, sigmoid objectness and class scores), training code removed.
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv(nn.Module):
    def __init__(self, cin, cout, k, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, (k - 1) // 2)

    def forward(self, x):
        return F.silu(self.conv(x), inplace=True)


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut):
        super().__init__()
        self.conv1, self.conv2, self.use_add = BaseConv(channels, channels, 1), BaseConv(channels, channels, 3), shortcut

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class CSPLayer(nn.Module):
    def __init__(self, cin, cout, n, shortcut=True):
        super().__init__()
        hidden = cout // 2
        self.conv1, self.conv2, self.conv3 = BaseConv(cin, hidden, 1), BaseConv(cin, hidden, 1), BaseConv(2 * hidden, cout, 1)
        self.m = nn.Sequential(*[Bottleneck(hidden, shortcut) for _ in range(n)])

    def forward(self, x):
        x_1, x_2 = self.conv1(x), self.conv2(x)
        return self.conv3(torch.cat((self.m(x_1), x_2), 1))


class SPPBottleneck(nn.Module):
    def __init__(self, cin, cout, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden = cin // 2
        self.conv1, self.conv2 = BaseConv(cin, hidden, 1), BaseConv(hidden * (len(kernel_sizes) + 1), cout, 1)
        self.m = nn.ModuleList(nn.MaxPool2d(k, 1, k // 2) for k in kernel_sizes)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.m], 1))


class Focus(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = BaseConv(cin * 4, cout, k)

    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))


class CSPDarknet(nn.Module):
    def __init__(self, base=64, depth=3):
        super().__init__()
        self.stem = Focus(3, base, 3)
        self.dark2 = nn.Sequential(BaseConv(base, base * 2, 3, 2), CSPLayer(base * 2, base * 2, depth))
        self.dark3 = nn.Sequential(BaseConv(base * 2, base * 4, 3, 2), CSPLayer(base * 4, base * 4, depth * 3))
        self.dark4 = nn.Sequential(BaseConv(base * 4, base * 8, 3, 2), CSPLayer(base * 8, base * 8, depth * 3))
        self.dark5 = nn.Sequential(BaseConv(base * 8, base * 16, 3, 2), SPPBottleneck(base * 16, base * 16), CSPLayer(base * 16, base * 16, depth, shortcut=False))

    def forward(self, x):
        x3 = self.dark3(self.dark2(self.stem(x)))
        x4 = self.dark4(x3)
        return x3, x4, self.dark5(x4)


class YOLOPAFPN(nn.Module):
    def __init__(self, channels=(256, 512, 1024), depth=3):
        super().__init__()
        c3, c4, c5 = channels
        self.backbone = CSPDarknet()
        self.lateral_conv0 = BaseConv(c5, c4, 1)
        self.C3_p4 = CSPLayer(2 * c4, c4, depth, False)
        self.reduce_conv1 = BaseConv(c4, c3, 1)
        self.C3_p3 = CSPLayer(2 * c3, c3, depth, False)
        self.bu_conv2 = BaseConv(c3, c3, 3, 2)
        self.C3_n3 = CSPLayer(2 * c3, c4, depth, False)
        self.bu_conv1 = BaseConv(c4, c4, 3, 2)
        self.C3_n4 = CSPLayer(2 * c4, c5, depth, False)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([F.interpolate(fpn_out0, scale_factor=2, mode="nearest"), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([F.interpolate(fpn_out1, scale_factor=2, mode="nearest"), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes=80, channels=(256, 512, 1024), width=256):
        super().__init__()
        self.stems = nn.ModuleList(BaseConv(c, width, 1) for c in channels)
        self.cls_convs = nn.ModuleList(nn.Sequential(BaseConv(width, width, 3), BaseConv(width, width, 3)) for _ in channels)
        self.reg_convs = nn.ModuleList(nn.Sequential(BaseConv(width, width, 3), BaseConv(width, width, 3)) for _ in channels)
        self.cls_preds = nn.ModuleList(nn.Conv2d(width, num_classes, 1) for _ in channels)
        self.reg_preds = nn.ModuleList(nn.Conv2d(width, 4, 1) for _ in channels)
        self.obj_preds = nn.ModuleList(nn.Conv2d(width, 1, 1) for _ in channels)

    def forward(self, features):
        outputs = []
        for k, x in enumerate(features):
            x = self.stems[k](x)
            cls_output = self.cls_preds[k](self.cls_convs[k](x))
            reg_feat = self.reg_convs[k](x)
            outputs.append(torch.cat([self.reg_preds[k](reg_feat), self.obj_preds[k](reg_feat).sigmoid(), cls_output.sigmoid()], 1).flatten(2))
        return torch.cat(outputs, 2).permute(0, 2, 1)


class YOLOX(nn.Module):
    """Input: (B, 3, 640, 640) BGR pixels in 0-255 (letterboxed with 114). Output: (B, 8400, 85) = cx, cy, log w, log h offsets per
    grid cell (strides 8/16/32), objectness, 80 COCO class scores."""
    def __init__(self):
        super().__init__()
        self.backbone, self.head = YOLOPAFPN(), YOLOXHead()

    def forward(self, x):
        return self.head(self.backbone(x))
