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

from typing import Optional, Union
import torch
from diffusers.models.embeddings import get_timestep_embedding
from torch import nn


def emb_add(emb1: torch.Tensor, emb2: Optional[torch.Tensor]):
    return emb1 if emb2 is None else emb1 + emb2


class TimeEmbedding(nn.Module):
    def __init__(
        self,
        sinusoidal_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.proj_in = nn.Linear(sinusoidal_dim, hidden_dim)
        self.proj_hid = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.SiLU()

    def forward(
        self,
        timestep: Union[int, float, torch.IntTensor, torch.FloatTensor],
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.FloatTensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=target_dtype)
        if timestep.ndim == 0:
            timestep = timestep[None]

        emb = get_timestep_embedding(
            timesteps=timestep,
            embedding_dim=self.sinusoidal_dim,
            flip_sin_to_cos=False,
            downscale_freq_shift=0,
        )
        emb = emb.to(target_dtype)
        emb = self.proj_in(emb)
        emb = self.act(emb)
        emb = self.proj_hid(emb)
        emb = self.act(emb)
        emb = self.proj_out(emb)
        return emb
