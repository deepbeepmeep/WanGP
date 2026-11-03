# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Routed mixture-of-experts building blocks of the LM engine.

- Routing functions on router logits: softmax top-k (Qwen MoE, Fish Drama 3) and sigmoid scores with a selection bias (DeepSeek-V3,
  Nemotron-H). The logits come from the model's own router (an nn.Linear that MMGP may quantize, or SigmoidBiasRouter below).
- routed_experts(): the routed experts of an MMGP expert bank (mmgp.expert_compute.RoutedExperts: SwiGLU or squared ReLU, CPU storage
  with optional GPU copies): a graph-safe queued call for decoding passes on CUDA (the CPU experts run as a CUDA host function), an eager
  call for prompt processing (resident experts decoded once for all their rows, CPU experts grouped by expert).
"""
import torch
from torch import nn
from torch.nn import functional as F

QUEUED_ROWS = 16  # decoding and verification passes; larger passes (prompt processing) run eagerly


def softmax_topk(logits, top_k, normalize):
    probabilities = logits.float().softmax(-1)
    weights, ids = probabilities.topk(top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return ids, weights


def sigmoid_bias_topk(logits, bias, top_k, normalize, scaling):
    """Selection by sigmoid score + bias (aux-loss-free balancing); weights: the uncorrected scores, normalized, scaled."""
    scores = logits.float().sigmoid()
    ids = torch.topk(scores + bias.float(), k=top_k, dim=-1, sorted=False)[1]
    weights = scores.gather(1, ids)
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return ids, weights * scaling


class SigmoidBiasRouter(nn.Module):
    """Router of DeepSeek-V3-style MoE layers: float32 logits of an unquantized weight, a float32 per-expert selection bias. Group-limited
    selection (n_group > 1) is not implemented."""
    def __init__(self, hidden, experts, top_k, normalize, scaling, groups=1):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("Group-limited routing (n_group > 1)")
        self.top_k, self.normalize, self.scaling = top_k, normalize, scaling
        self.weight = nn.Parameter(torch.zeros(experts, hidden))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(experts, dtype=torch.float32))
        # precision lock: the routing stays unquantized (ConvRot exports skip the module), the bias float32 and the weight as stored
        self._lock_dtype = torch.float32
        self.weight._lock_dtype = None

    def forward(self, x):
        return sigmoid_bias_topk(F.linear(x.float(), self.weight.float()), self.e_score_correction_bias, self.top_k, self.normalize, self.scaling)


def routed_experts(experts, x, ids, weights, packed=None):
    """[rows, hidden] sum of the selected experts in x's dtype (float32 accumulation)."""
    if x.is_cuda and x.shape[0] <= QUEUED_ROWS:
        return experts.queue_forward(x, ids, weights, packed)
    return experts(x, ids, weights, packed)
