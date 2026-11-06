# Adapted from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm), Copyright (c) 2025 Xingkai Yu,
# MIT License (see docs/third_party_licenses/MIT.txt). Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional
import os
from shared.kernels.dispatch import triton_dtype_supported

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_SAMPLER_NUMERIC_GUARD = os.environ.get("WAN2GP_LMGP_SAMPLER_NUMERIC_GUARD", "0") == "1"
_REPETITION_INCREMENT_LIMIT = 6
_REPETITION_RUNTIME_SCALARS = ("stored_count", "new_count", "virtual_count", "penalty", *(f"{kind}_{index}" for kind in ("new", "virtual") for index in range(_REPETITION_INCREMENT_LIMIT)))


if triton is not None:
    @triton.jit(do_not_specialize=_REPETITION_RUNTIME_SCALARS, do_not_specialize_on_alignment=_REPETITION_RUNTIME_SCALARS)
    def _sparse_repetition_penalty_kernel(logits, stored_ids, stored_count, new_count, virtual_count, penalty, new_0, new_1, new_2, new_3, new_4, new_5, virtual_0, virtual_1, virtual_2, virtual_3, virtual_4, virtual_5, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        stored_mask = offsets < stored_count
        new_offsets = offsets - stored_count
        new_mask = (new_offsets >= 0) & (new_offsets < new_count)
        virtual_offsets = new_offsets - new_count
        virtual_mask = (virtual_offsets >= 0) & (virtual_offsets < virtual_count)
        stored_token = tl.load(stored_ids + offsets, mask=stored_mask, other=0)
        new_token = tl.where(new_offsets == 0, new_0, tl.where(new_offsets == 1, new_1, tl.where(new_offsets == 2, new_2, tl.where(new_offsets == 3, new_3, tl.where(new_offsets == 4, new_4, new_5)))))
        virtual_token = tl.where(virtual_offsets == 0, virtual_0, tl.where(virtual_offsets == 1, virtual_1, tl.where(virtual_offsets == 2, virtual_2, tl.where(virtual_offsets == 3, virtual_3, tl.where(virtual_offsets == 4, virtual_4, virtual_5)))))
        token = tl.where(stored_mask, stored_token, tl.where(new_mask, new_token, virtual_token))
        active = stored_mask | new_mask | virtual_mask
        tl.store(stored_ids + offsets, token, mask=new_mask)
        score = tl.load(logits + token, mask=active, other=0.0)
        tl.store(logits + token, tl.where(score < 0, score * penalty, score / penalty), mask=active)


    @triton.jit(do_not_specialize=("n_elements", "log_min_p"), do_not_specialize_on_alignment=("n_elements", "log_min_p"))
    def _min_p_mask_kernel(logits, max_logit, n_elements, log_min_p, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        scores = tl.load(logits + offsets, mask=mask, other=-float("inf"))
        threshold = tl.load(max_logit) + log_min_p
        tl.store(logits + offsets, tl.where(scores >= threshold, scores, -float("inf")), mask=mask)


def apply_sparse_repetition_penalty_(logits: torch.Tensor, stored_ids: torch.Tensor, stored_count: int, new_token_ids: list[int], virtual_token_ids: list[int], penalty: float, work_values: torch.Tensor | None = None, use_triton: bool = True) -> None:
    """Apply one action-local repetition penalty using persistent sparse buffers."""

    new_count, virtual_count = len(new_token_ids), len(virtual_token_ids)
    if new_count > _REPETITION_INCREMENT_LIMIT:
        raise ValueError(f"Sparse repetition updates support at most {_REPETITION_INCREMENT_LIMIT} new tokens per decode.")
    total = int(stored_count) + new_count + virtual_count
    if total == 0 or float(penalty) == 1.0:
        return
    if use_triton and logits.is_cuda and triton is not None and triton_dtype_supported(logits.device, logits.dtype):
        new_values = [*new_token_ids, *([-1] * (_REPETITION_INCREMENT_LIMIT - new_count))]
        first_virtual_count = min(virtual_count, _REPETITION_INCREMENT_LIMIT)
        virtual_values = [*virtual_token_ids[:first_virtual_count], *([-1] * (_REPETITION_INCREMENT_LIMIT - first_virtual_count))]
        _sparse_repetition_penalty_kernel[(triton.cdiv(int(stored_count) + new_count + first_virtual_count, 256),)](logits, stored_ids, int(stored_count), new_count, first_virtual_count, float(penalty), *new_values, *virtual_values, BLOCK_SIZE=256)
        for start in range(_REPETITION_INCREMENT_LIMIT, virtual_count, _REPETITION_INCREMENT_LIMIT):
            chunk = virtual_token_ids[start:start + _REPETITION_INCREMENT_LIMIT]
            virtual_values = [*chunk, *([-1] * (_REPETITION_INCREMENT_LIMIT - len(chunk)))]
            _sparse_repetition_penalty_kernel[(1,)](logits, stored_ids, 0, 0, len(chunk), float(penalty), *new_values, *virtual_values, BLOCK_SIZE=256)
        return

    if work_values is None:
        work_values = torch.empty(total, dtype=logits.dtype, device=logits.device)
    for index, token_id in enumerate((*new_token_ids, *virtual_token_ids)):
        stored_ids[stored_count + index] = token_id
    token_ids = stored_ids[:total]
    scores = work_values[:total]
    torch.index_select(logits, 0, token_ids, out=scores)
    scores.div_(penalty)
    F.leaky_relu_(scores, negative_slope=float(penalty) * float(penalty))
    logits.index_copy_(0, token_ids, scores)


def apply_min_p_mask_(logits: torch.Tensor, min_p: float, use_triton: bool = False) -> torch.Tensor:
    """Apply min-p in place without creating a dynamically sized candidate tensor."""

    max_logit = logits.max()
    log_min_p = math.log(float(min_p))
    if use_triton and logits.is_cuda and logits.is_contiguous() and triton is not None and triton_dtype_supported(logits.device, logits.dtype):
        n_elements = logits.numel()
        _min_p_mask_kernel[(triton.cdiv(n_elements, 256),)](logits, max_logit, n_elements, log_min_p, BLOCK_SIZE=256, num_warps=8, num_stages=3)
    else:
        logits.masked_fill_(logits < max_logit + log_min_p, float("-inf"))
    return logits


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
    k_max: Optional[int] = None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits (vLLM style).
    
    The logits tensor is updated in-place.
    """
    squeezed = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeezed = True
    if k is not None:
        k = k.reshape(-1)
    if p is not None:
        p = p.reshape(-1)

    if p is None:
        if k is None:
            return logits.squeeze(0) if squeezed else logits
        # Avoid sorting vocab for top-k only case
        logits = apply_top_k_only(logits, k, k_max)
        return logits.squeeze(0) if squeezed else logits

    # Need to sort for top-p
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k first
        vocab_size = logits_sort.size(1)
        # Clamp k to valid range
        k_clamped = k.clamp(1, vocab_size).long()
        top_k_mask_idx = vocab_size - k_clamped  # shape: [B]
        # Get the threshold value for each batch
        top_k_thresh = logits_sort.gather(1, top_k_mask_idx.unsqueeze(1))
        top_k_mask = logits_sort < top_k_thresh
        logits_sort.masked_fill_(top_k_mask, float('-inf'))

    # Apply top-p
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)  # reuse buffer
    top_p_mask = probs_sum <= (1.0 - p.unsqueeze(1))
    # Ensure at least one token is kept
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, float('-inf'))

    # Re-sort back to original positions
    logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
    return logits.squeeze(0) if squeezed else logits


def apply_min_p(
    logits: torch.Tensor,
    min_p: Optional[torch.Tensor],
    known_active: bool = False,
) -> torch.Tensor:
    if min_p is None:
        return logits
    squeezed = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeezed = True
    min_p = min_p.reshape(-1)
    active_mask = min_p > 0
    if not known_active and not torch.any(active_mask):
        return logits.squeeze(0) if squeezed else logits
    probs = logits.softmax(dim=-1)
    max_probs, max_idx = probs.max(dim=-1, keepdim=True)
    threshold = max_probs * min_p.unsqueeze(1)
    mask = probs < threshold
    mask.scatter_(1, max_idx, False)
    mask &= active_mask.unsqueeze(1)
    logits.masked_fill_(mask, float("-inf"))
    return logits.squeeze(0) if squeezed else logits


def apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
    k_max: Optional[int] = None,
) -> torch.Tensor:
    """Apply top-k mask without sorting the entire vocab (vLLM style).
    
    This is much faster than sorting for top-k only cases.
    The logits tensor is updated in-place.
    """
    squeezed = False
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeezed = True
    k = k.reshape(-1)

    vocab_size = logits.shape[1]
    # Handle cases where k >= vocab_size (no filtering needed)
    no_top_k_mask = (k <= 0) | (k >= vocab_size)
    # Set invalid k to 1 so we can still gather
    k_safe = k.masked_fill(no_top_k_mask, 1).long()
    # NOTE: This int() causes CPU-GPU sync, but torch.topk requires Python int: the operations after it would wait for the host one by one (k_max: known to the caller)
    max_top_k = int(k_safe.max().clamp(max=vocab_size)) if k_max is None else k_max
    
    # Get top-k values for all batches
    # topk.values has shape [batch_size, max_top_k]
    topk_values = logits.amax(dim=1, keepdim=True) if k_max == 1 else logits.topk(max_top_k, dim=1).values
    
    # Convert k to 0-based index: we want the k-th largest value (index k-1)
    # Clamp to valid range for gather
    k_index = (k_safe - 1).clamp(0, max_top_k - 1).unsqueeze(1)  # shape: [B, 1]
    # Gather the threshold value (the k-th largest)
    top_k_thresh = topk_values.gather(1, k_index)
    
    # For rows with no top-k filtering, set threshold to -inf so nothing gets masked
    top_k_thresh.masked_fill_(no_top_k_mask.unsqueeze(1), float('-inf'))
    
    # Mask all values below the threshold
    logits.masked_fill_(logits < top_k_thresh, float('-inf'))
    return logits.squeeze(0) if squeezed else logits


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, 
        logits: torch.Tensor, 
        temperatures: torch.Tensor,
        top_ks: Optional[torch.Tensor] = None,
        top_ps: Optional[torch.Tensor] = None,
        min_ps: Optional[torch.Tensor] = None,
        repetition_penalties: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        k_max: Optional[int] = None,
        min_p_active: bool = False,
    ):
        """
        Sample tokens from logits with optional top-k and top-p filtering.
        
        Condition checking is done OUTSIDE the compiled function to avoid
        graph breaks from .any() calls.
        """
        # Apply temperature
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))

        logits = apply_top_k_top_p(
            logits,
            top_ks,
            top_ps,
            k_max,
        )
        logits = apply_min_p(logits, min_ps, min_p_active)
        if _SAMPLER_NUMERIC_GUARD:
            logits = torch.nan_to_num(logits, nan=float("-inf"))
            invalid_rows = ~torch.isfinite(logits).any(dim=-1)
            if invalid_rows.any():
                logits[invalid_rows, 0] = 0.0
        probs = torch.softmax(logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1, generator=generator).clamp_min_(1e-10)
        sample_tokens = probs.div_(noise).argmax(dim=-1).reshape(-1)
        return sample_tokens


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
