# Copyright (c) 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for WanGP (2026): VPR (models/model.py) and ASTP (models/poolings.py) of AntSpeaker, without its model registry and
# checkpoint loader.
import torch
import torch.nn as nn

from .mect import MECT_B2


class ASTP(nn.Module):
    """Attentive statistics pooling: Channel - and context-dependent statistics pooling, first used in ECAPA_TDNN.
    """
    def __init__(self,
                input_dim,
                config={}):
        super(ASTP, self).__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = config.get("bottleneck_dim", 128)
        self.global_context_att = config.get("global_context_att", False)
        # Use Convld with stride == 1 rather than Linear, then we don't
        # need to transpose inputs.
        if self.global_context_att:
            self.linear1 = nn.Conv1d(
                input_dim * 3, self.bottleneck_dim,
                kernel_size=1) # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(
                input_dim, self.bottleneck_dim,
                kernel_size=1) # equals W and b in the paper
        self.linear2 = nn.Conv1d(self.bottleneck_dim, input_dim,
                                kernel_size=1) # equals V and k in the paper
        self.out_dim = 2 * self.input_dim

    def forward(self, x):
        """
        x: a 3-dimensional tensor in tdnn-based architecture (B,F,T)
        or a 4-dimensional tensor in resnet architecture (B,C,F,T)
        0-dim: batch-dimension, last-dim: time-dimension (frame-dimension)
        """
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        assert len(x.shape) == 3

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(
                torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! ReLU may be hard to converge.
        alpha = torch.tanh(self.linear1(x_in)) # alpha= F.relu(self.linear1(x_ in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha* (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        return torch.cat([mean, std], dim=1)

    def get_out_dim(self):
        self.out_dim = 2 * self.input_dim
        return self.out_dim


class VPR(nn.Module):
    """
        For recognition tasks that need to extract vectors: speaker recognition
    """
    def __init__(self, config):
        super().__init__()
        self.cmn = config.get('cmn', True)
        self.feature_extractor = MECT_B2(config["backbone"]["config"])
        self.encoding_layers = ASTP(self.feature_extractor.get_out_dim())
        self.pool_out_dim = self.encoding_layers.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.embed_dim = config.get("embed_dim", 192)
        self.linear = nn.Linear(self.pool_out_dim, self.embed_dim, bias=False)

    def forward(self, inputs):
        if self.cmn:
            with torch.no_grad():
                inputs = inputs - torch.mean(inputs, -1, keepdim=True)
        x = self.feature_extractor(inputs)
        x = self.encoding_layers(x)
        embeddings = self.linear(self.bn(x))
        return embeddings
    
