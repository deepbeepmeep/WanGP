# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import Callable, Optional
from torch import nn
import torch
import torch.nn.functional as F
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init

# (dim: int, eps: float, elementwise_affine: bool)
norm_layer_type = Callable[[int, float, bool], nn.Module]


class CustomLayerNorm(nn.Module):
    """
    Custom LayerNorm implementation to replace Apex FusedLayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomLayerNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            self.bias = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)


class CustomRMSNorm(nn.Module):
    """
    Custom RMSNorm implementation to replace Apex FusedRMSNorm
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(CustomRMSNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(torch.ones(*normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, input):
        return F.rms_norm(input, self.normalized_shape, self.weight, self.eps)


def get_norm_layer(norm_type: Optional[str]) -> norm_layer_type:

    def _norm_layer(dim: int, eps: float, elementwise_affine: bool):
        if norm_type is None:
            return nn.Identity()

        if norm_type == "layer":
            return nn.LayerNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "rms":
            return nn.RMSNorm(
                normalized_shape=dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

        if norm_type == "fusedln":
            # Use custom LayerNorm instead of Apex FusedLayerNorm
            return CustomLayerNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        if norm_type == "fusedrms":
            # Use custom RMSNorm instead of Apex FusedRMSNorm
            return CustomRMSNorm(
                normalized_shape=dim,
                elementwise_affine=elementwise_affine,
                eps=eps,
            )

        raise NotImplementedError(f"{norm_type} is not supported")

    return _norm_layer
