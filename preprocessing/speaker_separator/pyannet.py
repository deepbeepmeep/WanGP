# The MIT License (MIT)
#
# Copyright (c) 2019- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHOR
# Hervé Bredin - http://herve.niderb.fr
#
# Inference-only PyanNet of pyannote.audio 3.3.2 (pyannote/audio/models/segmentation/PyanNet.py and
# models/blocks/sincnet.py) for the pyannote/segmentation-3.0 checkpoint. WanGP changes: the SincNet
# filterbank (asteroid-filterbanks ParamSincFB, MIT) is stored as the fixed convolution weight it computes
# at inference, dropout and training code removed, hyper-parameters fixed to the checkpoint's.
"""pyannote segmentation-3.0: 10 s of 16 kHz audio -> powerset log-probabilities of 3 local speakers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincNet(nn.Module):
    def __init__(self, stride=10):
        super().__init__()
        self.stride = stride
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        self.sinc_filters = nn.Parameter(torch.empty(80, 1, 251), requires_grad=False)
        self.conv1d = nn.ModuleList([nn.Conv1d(80, 60, 5), nn.Conv1d(60, 60, 5)])
        self.norm1d = nn.ModuleList([nn.InstanceNorm1d(80, affine=True), nn.InstanceNorm1d(60, affine=True), nn.InstanceNorm1d(60, affine=True)])

    def forward(self, waveforms):
        x = F.conv1d(self.wav_norm1d(waveforms), self.sinc_filters, stride=self.stride).abs()
        x = F.leaky_relu(self.norm1d[0](F.max_pool1d(x, 3, 3)))
        for conv, norm in zip(self.conv1d, self.norm1d[1:]):
            x = F.leaky_relu(norm(F.max_pool1d(conv(x), 3, 3)))
        return x


class PyanNet(nn.Module):
    """Powerset classes: no speaker, speakers 0 / 1 / 2, pairs (0, 1) / (0, 2) / (1, 2)."""

    def __init__(self):
        super().__init__()
        self.sincnet = SincNet()
        self.lstm = nn.LSTM(60, 128, num_layers=4, bidirectional=True, batch_first=True)
        self.linear = nn.ModuleList([nn.Linear(256, 128), nn.Linear(128, 128)])
        self.classifier = nn.Linear(128, 7)

    def forward(self, waveforms):
        """waveforms (batch, 1, samples) -> (batch, frames, 7) log-probabilities."""
        x, _ = self.lstm(self.sincnet(waveforms).transpose(1, 2))
        for linear in self.linear:
            x = F.leaky_relu(linear(x))
        return F.log_softmax(self.classifier(x), dim=-1)
