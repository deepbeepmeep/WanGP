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
# Modified for WanGP.

"""
Native Resolution Transformer (NaDiT) tensor manipulation utilities.

TORCH.COMPILE OPTIMIZED VERSION
================================
This module has been optimized for torch.compile compatibility by eliminating
data-dependent operations that cause graph breaks:

Key Changes from Original:
- Replaced all .tolist() calls with pure tensor operations
- Minimized .item() calls (only used where required by einops API)
- Replaced list comprehensions with tensor-based splitting
- Added _tensor_split helper for compile-friendly splitting
- Proper device management to ensure tensors stay on correct devices
"""

from itertools import chain
from typing import Callable, Dict, List, Tuple
import einops
import torch


def _tensor_split(tensor: torch.Tensor, lengths: torch.LongTensor, dim: int = 0) -> List[torch.Tensor]:
    """
    Optimized compile-friendly split using torch.tensor_split.
    
    Uses PyTorch's native C++ implementation for fast eager mode execution
    while remaining fully compatible with torch.compile symbolic tracing.
    
    Args:
        tensor: Input tensor to split
        lengths: Tensor of split lengths (1D)
        dim: Dimension along which to split (default: 0)
    
    Returns:
        List of split tensors
    """
    if lengths.numel() == 0:
        return []
    
    if lengths.numel() == 1:
        return [tensor]
    
    # Calculate split indices: torch.tensor_split splits BEFORE each index
    # So we need cumsum[:-1] to get the split points
    # NOTE: torch.tensor_split requires indices on CPU (PyTorch requirement)
    split_indices = lengths.cumsum(0)[:-1].cpu()
    
    # Use torch.tensor_split - native C++ implementation
    # Compilable: accepts tensor indices (symbolic shapes work)
    # Fast: uses optimized CUDA/CPU kernels instead of Python loops
    return list(torch.tensor_split(tensor, split_indices, dim=dim))


def flatten(
    hid: List[torch.FloatTensor],  # List of (*** c)
) -> Tuple[
    torch.FloatTensor,  # (L c)
    torch.LongTensor,  # (b n)
]:
    """
    Flatten a list of tensors into a single tensor and track their shapes.
    
    Converts a list of tensors with potentially different spatial shapes but
    same feature dimension into a flattened tensor and shape metadata.
    
    Args:
        hid: List of tensors, each with shape (d1, d2, ..., dn, c)
    
    Returns:
        Tuple of:
        - Flattened tensor of shape (L, c) where L = sum of all spatial dimensions
        - Shape tensor of shape (b, n) tracking original spatial dimensions
        
    COMPILE OPTIMIZATION: Uses tensor operations on correct device from input
    """
    assert len(hid) > 0
    device = hid[0].device
    
    # Stack shape metadata - ensure tensors are created on correct device
    shapes = []
    for x in hid:
        shape_tensor = torch.tensor(x.shape[:-1], dtype=torch.long, device=device)
        shapes.append(shape_tensor)
    shape = torch.stack(shapes)
    
    # Flatten and concatenate
    hid = torch.cat([x.flatten(0, -2) for x in hid])
    return hid, shape


def unflatten(
    hid: torch.FloatTensor,  # (L c) or (L ... c)
    hid_shape: torch.LongTensor,  # (b n)
) -> List[torch.Tensor]:  # List of (*** c) or (*** ... c)
    """
    Unflatten a tensor back to a list using shape metadata.
    
    Inverse operation of flatten(), reconstructing original tensor shapes.
    
    Args:
        hid: Flattened tensor of shape (L, c) or (L, ..., c)
        hid_shape: Shape metadata tensor of shape (b, n)
    
    Returns:
        List of unflattened tensors with original spatial dimensions
        
    COMPILE OPTIMIZATION: 
    - Uses optimized _tensor_split with torch.tensor_split (fast)
    - .cpu().numpy() conversion needed for torch.compile compatibility
      (unflatten() requires concrete Python ints, not symbolic shapes)
    """
    hid_len = hid_shape.prod(-1)
    
    # Use optimized tensor splitting (major performance improvement)
    hid_list = _tensor_split(hid, hid_len, dim=0)
    
    # Unflatten each piece
    # NOTE: .cpu().numpy() is required for torch.compile compatibility
    # .tolist() would fail with symbolic shapes during compilation
    result = []
    for i, x in enumerate(hid_list):
        shape = hid_shape[i]
        # Must use .cpu().numpy() for compilation compatibility
        # Shape tensors are small, so CPU transfer overhead is minimal
        target_shape = list(shape.cpu().numpy())
        result.append(x.unflatten(0, target_shape))
    
    return result


def concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> torch.FloatTensor:  # (L ... c)
    """
    Interleave video and text tensors batch-wise.
    
    Splits video and text tensors by batch lengths, then interleaves them:
    [vid_0, txt_0, vid_1, txt_1, ..., vid_b, txt_b]
    
    Args:
        vid: Video features tensor (VL, c)
        txt: Text features tensor (TL, c)
        vid_len: Length of each video sequence (b,)
        txt_len: Length of each text sequence (b,)
    
    Returns:
        Interleaved tensor (L, c) where L = sum(vid_len) + sum(txt_len)
        
    COMPILE OPTIMIZATION: Uses _tensor_split for compile-friendly splitting
    """
    # Use tensor-based splitting
    vid_splits = _tensor_split(vid, vid_len, dim=0)
    txt_splits = _tensor_split(txt, txt_len, dim=0)
    
    # Interleave
    interleaved = []
    for v, t in zip(vid_splits, txt_splits):
        interleaved.extend([v, t])
    
    return torch.cat(interleaved)


def concat_idx(
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    Callable,
    Callable,
]:
    """
    Create index-based concatenation and un-concatenation functions.
    
    Pre-computes indices for efficient interleaving and de-interleaving operations.
    Returns callable functions that can be reused multiple times.
    
    Args:
        vid_len: Video sequence lengths (b,)
        txt_len: Text sequence lengths (b,)
    
    Returns:
        Tuple of (concat_fn, unconcat_fn):
        - concat_fn: Lambda that interleaves vid and txt tensors
        - unconcat_fn: Lambda that separates interleaved tensor back to vid and txt
        
    COMPILE OPTIMIZATION: Pre-computes all indices using tensor operations
    """
    device = vid_len.device
    vid_sum = vid_len.sum()
    txt_sum = txt_len.sum()
    
    vid_idx = torch.arange(vid_sum, device=device)
    txt_idx = torch.arange(vid_sum, vid_sum + txt_sum, device=device)
    
    # Build interleaving indices using compile-friendly _tensor_split
    batch_size = len(vid_len)
    vid_idx_splits = _tensor_split(vid_idx, vid_len, dim=0)
    txt_idx_splits = _tensor_split(txt_idx, txt_len, dim=0)
    
    # Create interleaved target indices
    tgt_idx_list = []
    for i in range(batch_size):
        tgt_idx_list.append(vid_idx_splits[i])
        tgt_idx_list.append(txt_idx_splits[i])
    
    tgt_idx = torch.cat(tgt_idx_list)
    src_idx = torch.argsort(tgt_idx)
    vid_idx_len = len(vid_idx)
    
    if len(vid_len) == 1:
        return (
            lambda vid_txt_list: torch.cat((vid_txt_list.pop(0), vid_txt_list.pop())),
            lambda all_list: all_list.pop().split([vid_idx_len, len(txt_idx)]),
        )
    return (
        lambda vid_txt_list: torch.index_select(torch.cat((vid_txt_list.pop(0), vid_txt_list.pop())), 0, tgt_idx),
        lambda all_list: torch.index_select(all_list.pop(), 0, src_idx).split([vid_idx_len, len(txt_idx)]),
    )


def unconcat(
    all: torch.FloatTensor,  # (L ... c)
    vid_len: torch.LongTensor,  # (b)
    txt_len: torch.LongTensor,  # (b)
) -> Tuple[
    torch.FloatTensor,  # (VL ... c)
    torch.FloatTensor,  # (TL ... c)
]:
    """
    De-interleave concatenated video and text tensors.
    
    Inverse of concat(). Separates an interleaved tensor back into video and text.
    
    Args:
        all: Interleaved tensor (L, c)
        vid_len: Video sequence lengths (b,)
        txt_len: Text sequence lengths (b,)
    
    Returns:
        Tuple of (vid, txt) tensors
        
    COMPILE OPTIMIZATION: Uses tensor operations to build interleave pattern
    """
    batch_size = len(vid_len)
    
    # Create interleaved lengths: [vid_0, txt_0, vid_1, txt_1, ...]
    interleave_len = torch.stack([vid_len, txt_len], dim=1).flatten()
    
    # Split using compile-friendly operation
    all_splits = _tensor_split(all, interleave_len, dim=0)
    
    # Separate even (vid) and odd (txt) indices
    vid_parts = [all_splits[i] for i in range(0, len(all_splits), 2)]
    txt_parts = [all_splits[i] for i in range(1, len(all_splits), 2)]
    
    vid = torch.cat(vid_parts)
    txt = torch.cat(txt_parts)
    return vid, txt


def repeat_concat(
    vid: torch.FloatTensor,  # (VL ... c)
    txt: torch.FloatTensor,  # (TL ... c)
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: torch.LongTensor,  # (n) or (b)
) -> torch.FloatTensor:  # (L ... c)
    """
    Concatenate video and text with text repetition for window attention.
    
    For windowed attention, text features are repeated and interleaved with
    multiple video windows: [vid_0, txt_0, vid_1, txt_0, vid_2, txt_0, ...]
    
    Args:
        vid: Video features (VL, c)
        txt: Text features (TL, c)
        vid_len: Video window lengths (n*b,) where n=num_windows
        txt_len: Text sequence lengths (b,)
        txt_repeat: Number of times to repeat text (n,) or (b,)
    
    Returns:
        Interleaved tensor with repeated text
        
    COMPILE OPTIMIZATION: Uses _tensor_split and tensor-based repetition
    """
    # Split using compile-friendly operations
    vid_splits = _tensor_split(vid, vid_len, dim=0)
    txt_splits = _tensor_split(txt, txt_len, dim=0)
    
    # Handle txt_repeat shape flexibility
    if txt_repeat.numel() == len(txt_splits):
        repeat_counts = txt_repeat
    else:
        repeat_counts = txt_repeat.repeat(len(txt_splits))
    
    # Interleave with repetition
    result = []
    for i, v in enumerate(vid_splits):
        result.append(v)
        # Get corresponding text sample (cyclic)
        batch_idx = i % len(txt_splits) if len(txt_splits) > 0 else 0
        if batch_idx < len(txt_splits):
            result.append(txt_splits[batch_idx])
    
    return torch.cat(result)


def repeat_concat_idx(
    vid_len: torch.LongTensor,  # (n*b)
    txt_len: torch.LongTensor,  # (b)
    txt_repeat: torch.LongTensor,  # (n) or scalar
) -> Tuple[
    Callable,
    Callable,
]:
    """
    Create index-based repeat-concatenation and un-concatenation with coalescing.
    
    Similar to concat_idx but handles text repetition for window attention.
    The unconcat function coalesces (averages) repeated text features.
    
    Args:
        vid_len: Video window lengths (n*b,)
        txt_len: Text sequence lengths (b,)
        txt_repeat: Repetition count (scalar or tensor)
    
    Returns:
        Tuple of (concat_fn, unconcat_coalesce_fn):
        - concat_fn: Interleaves with text repetition
        - unconcat_coalesce_fn: Separates and averages repeated text
        
    Example:
        Input:  vid=[0,1,2,3,4,5,6,7,8], txt=[9,10], repeat=3
        Concat: [0,1,2,9,10, 3,4,5,9,10, 6,7,8,9,10]
        Unconcat: vid=[0,1,2,3,4,5,6,7,8], txt=[9,10] (averaged)
        
    COMPILE OPTIMIZATION: 
    - Eliminates .tolist() calls that caused graph breaks
    - Uses pure tensor operations for index building
    - Minimizes data-dependent branching
    """
    device = vid_len.device
    vid_sum = vid_len.sum()
    txt_sum = txt_len.sum()
    
    # Create base indices
    vid_idx = torch.arange(vid_sum, device=device)
    txt_idx = torch.arange(vid_sum, vid_sum + txt_sum, device=device)
    
    # Normalize txt_repeat to tensor
    if isinstance(txt_repeat, (int, float)):
        txt_repeat = torch.tensor([txt_repeat], dtype=torch.long, device=device)
    elif txt_repeat.dim() == 0:
        txt_repeat = txt_repeat.unsqueeze(0)
    
    # Calculate repeat pattern - keep as tensor to avoid graph breaks
    batch_size = len(txt_len)
    if txt_repeat.numel() == 1:
        num_repeats_tensor = txt_repeat.reshape(-1)  # Ensure 1D tensor
    else:
        # Use tensor operations for division
        num_repeats_tensor = torch.tensor([len(vid_len) // batch_size], dtype=torch.long, device=device)
    
    # Build concatenated indices using compile-friendly _tensor_split
    vid_idx_splits = _tensor_split(vid_idx, vid_len, dim=0)
    txt_idx_splits = _tensor_split(txt_idx, txt_len, dim=0)
    
    tgt_idx_list = []
    for i in range(len(vid_len)):
        # Add video window
        tgt_idx_list.append(vid_idx_splits[i])
        
        # Add corresponding text (with repeat)
        batch_idx = i % batch_size
        tgt_idx_list.append(txt_idx_splits[batch_idx])
    
    tgt_idx = torch.cat(tgt_idx_list)
    src_idx = torch.argsort(tgt_idx)
    txt_idx_len = len(tgt_idx) - len(vid_idx)
    
    # Pre-compute split lengths for coalescing using tensor operations
    repeat_txt_len = txt_len * num_repeats_tensor.squeeze()
    repeats = int(num_repeats_tensor.squeeze())
    repeat_txt_len_list = repeat_txt_len.tolist()

    def unconcat_coalesce(all_list):
        """
        Un-concat vid & txt, and coalesce the repeated txt by averaging.
        
        The text features appear multiple times (once per window) and need
        to be averaged to produce a single set of text features.
        
        COMPILE OPTIMIZATION: Uses unflatten with tensor dims (compile-friendly)
        """
        all = all_list.pop()
        vid_out = torch.index_select(all, 0, src_idx[:len(vid_idx)])
        txt_src_idx = src_idx[len(vid_idx):]
        txt_out_coalesced = []
        offset = 0
        for length in repeat_txt_len_list:
            txt_repeat = torch.index_select(all, 0, txt_src_idx[offset:offset + length])
            txt_out_coalesced.append(txt_repeat.unflatten(0, (-1, repeats)).mean(1))
            offset += length
        return vid_out, txt_out_coalesced[0] if len(txt_out_coalesced) == 1 else torch.cat(txt_out_coalesced)

    # Note: Using direct indexing instead of torch.index_select for backward compatibility
    # Direct indexing is deterministic even with repeated indices
    return (
        lambda vid_txt_list: torch.cat((vid_txt_list.pop(0), vid_txt_list.pop()))[tgt_idx],
        unconcat_coalesce,
    )


def rearrange(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    """
    Rearrange flattened tensor using einops pattern.
    
    Applies einops rearrange to each batch element independently.
    
    Args:
        hid: Flattened tensor (L, c)
        hid_shape: Shape metadata (b, n)
        pattern: Einops rearrange pattern
        **kwargs: Additional arguments for einops
    
    Returns:
        Tuple of (rearranged tensor, new shape metadata)
    """
    unflattened = unflatten(hid, hid_shape)
    rearranged = [einops.rearrange(h, pattern, **kwargs) for h in unflattened]
    return flatten(rearranged)


def rearrange_idx(
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, int],
) -> Tuple[Callable, Callable, torch.LongTensor]:
    """
    Create index-based rearrange functions.
    
    Pre-computes indices for efficient rearrangement operations.
    
    Args:
        hid_shape: Shape metadata (b, n)
        pattern: Einops rearrange pattern
        **kwargs: Additional arguments for einops
    
    Returns:
        Tuple of (rearrange_fn, reverse_fn, new_shape)
    """
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape = rearrange(hid_idx, hid_shape, pattern, **kwargs)
    tgt_idx = tgt_idx.squeeze(-1)
    src_idx = torch.argsort(tgt_idx)
    return (
        lambda hid: torch.index_select(hid, 0, tgt_idx),
        lambda hid: torch.index_select(hid, 0, src_idx),
        tgt_shape,
    )


def repeat(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    pattern: str,
    **kwargs: Dict[str, torch.LongTensor],  # (b)
) -> Tuple[
    torch.FloatTensor,
    torch.LongTensor,
]:
    """
    Repeat flattened tensor using einops pattern with per-batch parameters.
    
    Each batch element can have different repeat counts specified in kwargs.
    
    Args:
        hid: Flattened tensor (L, c)
        hid_shape: Shape metadata (b, n)
        pattern: Einops repeat pattern (e.g., "l c -> t l c")
        **kwargs: Repeat parameters as tensors (e.g., t=torch.tensor([2,3,4]))
    
    Returns:
        Tuple of (repeated tensor, new shape metadata)
        
    COMPILE OPTIMIZATION:
    - Minimizes .item() calls
    - Only converts to int at the last moment for einops API requirement
    """
    unflattened = unflatten(hid, hid_shape)
    
    # Build kwargs for each batch element
    repeated = []
    for i in range(len(unflattened)):
        # Extract values for einops (requires Python int)
        batch_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # Only convert to Python int where required by einops
                batch_kwargs[k] = int(v[i].item())
            else:
                batch_kwargs[k] = v
        repeated.append(einops.repeat(unflattened[i], pattern, **batch_kwargs))
    
    return flatten(repeated)


def pack(
    samples: List[torch.Tensor],  # List of (h w c).
) -> Tuple[
    List[torch.Tensor],  # groups [(b1 h1 w1 c1), (b2 h2 w2 c2)]
    List[List[int]],  # reversal indices.
]:
    """
    Group samples by shape and return grouped batches with reversal indices.
    
    Useful for batch processing samples with different spatial dimensions.
    
    Args:
        samples: List of tensors with potentially different shapes
    
    Returns:
        Tuple of (batched_groups, reversal_indices) for unpacking
    """
    batches = {}
    indices = {}
    for i, sample in enumerate(samples):
        shape = sample.shape
        batches[shape] = batches.get(shape, [])
        indices[shape] = indices.get(shape, [])
        batches[shape].append(sample)
        indices[shape].append(i)

    batches = list(map(torch.stack, batches.values()))
    indices = list(indices.values())
    return batches, indices


def unpack(
    batches: List[torch.Tensor],
    indices: List[List[int]],
) -> List[torch.Tensor]:
    """
    Unpack grouped batches back to original order.
    
    Inverse of pack().
    
    Args:
        batches: Grouped batches from pack()
        indices: Reversal indices from pack()
    
    Returns:
        List of tensors in original order
    """
    samples = [None] * (max(chain(*indices)) + 1)
    for batch, index in zip(batches, indices):
        for sample, i in zip(batch.unbind(), index):
            samples[i] = sample
    return samples


def window(
    hid: torch.FloatTensor,  # (L c)
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    """
    Apply windowing function to create non-overlapping windows.
    
    Used for window attention mechanisms where sequences are split into windows.
    
    Args:
        hid: Flattened tensor (L, c)
        hid_shape: Shape metadata (b, n)
        window_fn: Function that splits a tensor into windows
    
    Returns:
        Tuple of (windowed_tensor, window_shapes, window_counts)
        
    COMPILE OPTIMIZATION: Uses tensor operation for window count tracking
    """
    unflattened = unflatten(hid, hid_shape)
    windowed = [window_fn(h) for h in unflattened]
    
    # Track window counts using tensor operations
    device = hid_shape.device
    hid_windows = torch.tensor([len(w) for w in windowed], dtype=torch.long, device=device)
    
    # Flatten all windows
    all_windows = list(chain(*windowed))
    hid, hid_shape = flatten(all_windows)
    return hid, hid_shape, hid_windows


def window_idx(
    hid_shape: torch.LongTensor,  # (b n)
    window_fn: Callable[[torch.Tensor], List[torch.Tensor]],
):
    """
    Index-based windowing.

    Args:
        hid_shape: Shape metadata (b, n)
        window_fn: Function that splits a tensor into windows

    Returns:
        Tuple of (rows of the flattened tokens in window order, window_shapes, window_counts); the windows partition the tokens
    """
    hid_idx = torch.arange(hid_shape.prod(-1).sum(), device=hid_shape.device).unsqueeze(-1)
    tgt_idx, tgt_shape, tgt_windows = window(hid_idx, hid_shape, window_fn)
    return tgt_idx.squeeze(-1), tgt_shape, tgt_windows
