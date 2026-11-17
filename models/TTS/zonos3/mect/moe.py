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

# Modified for WanGP (2026): imports limited to this package (no model registry, no sys.path change, no unused torchaudio import).
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math


class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))



class TokenSMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.Linear(self.hidden_size, intermediate_size * self.num_experts, bias=False)
        self.act_fn = NewGELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        hidden_states_experts = self.act_fn(self.experts(hidden_states)).view(-1, self.num_experts, hidden_dim)
        experts_out = hidden_states_experts * routing_weights.unsqueeze(2)
        experts_out = experts_out.sum(dim=1).reshape(batch_size, sequence_length, hidden_dim)

        return experts_out

class UtteranceSMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.Linear(hidden_size, intermediate_size * num_experts, bias=False)
        self.act_fn = NewGELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, seq_len, hidden_dim = hidden_states.shape
        
        utterance_feat = hidden_states.mean(dim=1)
        router_logits = self.gate(utterance_feat)
        routing_weights = F.softmax(router_logits, dim=1)  # [B, num_experts]

        expert_outputs = self.experts(hidden_states)
        expert_outputs = expert_outputs.view(B, seq_len, self.num_experts, self.intermediate_size)
        expert_outputs = self.act_fn(expert_outputs)  # [B, seq_len, num_experts,intermediate_size]

        routing_weights = routing_weights.unsqueeze(1).unsqueeze(-1)
        
        weighted = expert_outputs * routing_weights
        
        output = weighted.sum(dim=2)

        return output


class TokenMoE(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=6, top_k=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])

        self.act_fn = NewGELUActivation()

    def forward(self, hidden_states):
        B, seq_len, hidden_dim = hidden_states.shape

        hidden_flat = hidden_states.view(-1, hidden_dim)  # [B*seq_len, hidden]

        gate_logits = self.gate(hidden_flat)  # [B*seq_len, num_experts]
        top_logits, top_idx = torch.topk(gate_logits, self.top_k, dim=1)  # [B*seq, K]
        top_weights = F.softmax(top_logits.float(), dim=1)  # [B*seq, K]

        experts_out = torch.zeros(
            B * seq_len, self.intermediate_size,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for k_idx in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = (top_idx[:, k_idx] == e_idx)
                if not mask.any():
                    continue

                expert_input = hidden_flat[mask]  # [N, hidden]
                expert_output = self.experts[e_idx](expert_input)  # [N, intermediate]
                expert_output = self.act_fn(expert_output)

                experts_out[mask] += expert_output * top_weights[mask, k_idx:k_idx+1]

        experts_out = experts_out.view(B, seq_len, self.intermediate_size)

        return experts_out


class UtteranceMoE(nn.Module):
    """
    整句级别MoE：一条语音的所有帧共享相同的专家选择
    """
    def __init__(self, hidden_size, intermediate_size, num_experts=6, top_k=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        self.experts = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])

        self.act_fn = NewGELUActivation()

    def forward(self, hidden_states):
        B, seq_len, hidden_dim = hidden_states.shape

        utterance_feat = hidden_states.mean(dim=1)  # [B, hidden_size]
        gate_logits = self.gate(utterance_feat)     # [B, num_experts]

        top_logits, top_idx = torch.topk(gate_logits, self.top_k, dim=1)  # [B, K]
        top_weights = F.softmax(top_logits, dim=1).unsqueeze(-1)  # [B, K, 1]

        experts_out = torch.zeros(
            B, seq_len, self.intermediate_size,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        for k_idx in range(self.top_k):
            expert_ids = top_idx[:, k_idx]  # [B]
            weights = top_weights[:, k_idx]  # [B, 1]

            for e_idx in range(self.num_experts):
                mask = (expert_ids == e_idx)  # [B]
                if not mask.any():
                    continue
            
                expert_input = hidden_states[mask]  # [N, seq_len, hidden]
                expert_output = self.experts[e_idx](expert_input)  # [N, seq_len, intermediate]
                expert_output = self.act_fn(expert_output)

                experts_out[mask] += expert_output * weights[mask].unsqueeze(1)

        return experts_out


class HybridMoE(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts_shard=6,
        num_experts_token=6,
        top_k_shard=None,
        top_k_token=None,
        moe_type="shard",
        shard_scale=1, 
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        if moe_type == "shard":
            if top_k_shard is not None:
                self.shard_moe = UtteranceMoE(
                    hidden_size, intermediate_size, num_experts=num_experts_shard, top_k=top_k_shard
                )
            else:
                self.shard_moe = UtteranceSMoE(
                      hidden_size, intermediate_size, num_experts=num_experts_shard
                  )
        elif moe_type == "token":
            if top_k_token is not None:
                self.token_moe = TokenMoE(
                    hidden_size, intermediate_size, num_experts=num_experts_token, top_k=top_k_token
                )
            else:
                self.token_moe = TokenSMoE(
                    hidden_size, intermediate_size, num_experts=num_experts_token
                )
        elif moe_type == "hmoe":
            if top_k_shard is not None:
                self.shard_moe = UtteranceMoE(
                    hidden_size, intermediate_size, num_experts=num_experts_shard, top_k=top_k_shard
                )
            else:
                self.shard_moe = UtteranceSMoE(
                      hidden_size, intermediate_size, num_experts=num_experts_shard
                  )
            if top_k_token is not None:
                self.token_moe = TokenMoE(
                    hidden_size, intermediate_size, num_experts=num_experts_token, top_k=top_k_token
                )
            else:
                self.token_moe = TokenSMoE(
                    hidden_size, intermediate_size, num_experts=num_experts_token
                )

        self.moe_type = moe_type
        self.shard_scale = shard_scale

    def forward(self, hidden_states, return_aux_loss=False):
        if self.moe_type == "hmoe":
            out = self.shard_moe(hidden_states)
            out = self.token_moe(out)
            out = out + hidden_states
        elif self.moe_type == "shard":
            out = self.shard_moe(hidden_states)
        elif self.moe_type == "token":
            out = self.token_moe(hidden_states)
        return out
