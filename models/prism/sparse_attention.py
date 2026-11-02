"""Released fixed 4x4x4 BSA: 75% sparsity plus 0.2 nucleus selection.

The PyTorch implementation is the numerical reference. The optional upstream
Triton kernel is selected explicitly by device and WanGP's precision policy.
"""
import functools
import math

import torch
from torch.nn import functional as F

from shared.kernels import kernel_policy
from shared.kernels.triton_tuner import LaunchTuner
from shared.utils.phase_progress import check_abort

_TUNER = LaunchTuner('wangp_prism_bsa', 3, 'Prism BSA')
try:
    import triton
    from .sparse_kernel import _attn_fwd_bsa_varlen_align
except (ImportError, OSError, RuntimeError):
    _attn_fwd_bsa_varlen_align = None


@functools.cache
def supported(device, dtype):
    return _attn_fwd_bsa_varlen_align is not None and dtype == torch.bfloat16 and device.type == 'cuda' and torch.cuda.get_device_capability(device) >= (8, 0)


def block_layout(grid, device):
    t, h, w = grid
    tp, hp, wp = (math.ceil(n / 4) * 4 for n in grid)
    z, y, x = torch.meshgrid(torch.arange(tp, device=device), torch.arange(hp, device=device), torch.arange(wp, device=device), indexing='ij')
    valid = (z < t) & (y < h) & (x < w)
    index = (z * h * w + y * w + x).clamp_max(t * h * w - 1)
    def reorder(v):
        return v.reshape(tp // 4, 4, hp // 4, 4, wp // 4, 4).permute(0, 2, 4, 1, 3, 5).reshape(-1)
    return reorder(index), reorder(valid)


def select_blocks(q, k, valid, padded):
    batch, heads, tokens, dim = q.shape
    blocks = tokens // 64
    def pool(x):
        x = x.view(batch, heads, blocks, 64, dim)
        if not padded:
            return x.mean(3)
        mask = valid.view(1, 1, blocks, 64, 1).float()
        return (x * mask).sum(3) / mask.sum(3).clamp_min(1)
    qm, km = pool(q), pool(k)
    minimum = max(1, int(.25 * blocks))
    bound = min(blocks, max(minimum, int(.2 * blocks) + 1))
    indices = torch.empty((batch, heads, blocks, bound), dtype=torch.int64, device=q.device)
    lengths = torch.empty((batch, heads, blocks), dtype=torch.int32, device=q.device)
    positions = torch.arange(bound, device=q.device)
    # Keep the block-scoring temporary bounded even for long, high-resolution clips.
    for start in range(0, blocks, 64):
        score = qm[:, :, start:start + 64] @ km.transpose(-1, -2)
        weights = (score * (dim ** -.5)).softmax(-1)
        values, ids = weights.topk(bound, dim=-1, sorted=True)
        count = ((values.cumsum(-1) < .2).int().sum(-1, keepdim=True) + 1).clamp(minimum, blocks)
        indices[:, :, start:start + 64] = ids.masked_fill(positions >= count, blocks).sort(-1).values
        lengths[:, :, start:start + 64] = count.squeeze(-1)
        check_abort()
    return indices, lengths


def reference(q, k, v, indices, lengths, valid):
    batch, heads, tokens, dim = q.shape
    output = torch.empty_like(q)
    offsets = torch.arange(64, device=q.device)
    rank = torch.arange(indices.shape[-1], device=q.device)
    b = torch.arange(batch, device=q.device)[:, None, None]
    h = torch.arange(heads, device=q.device)[None, :, None]
    for block in range(tokens // 64):
        ids = indices[:, :, block]
        allowed = rank < lengths[:, :, block, None]
        positions = (ids.clamp_max(tokens // 64 - 1)[..., None] * 64 + offsets).flatten(-2)
        mask = allowed.repeat_interleave(64, -1) & valid[positions]
        keys, values = k[b, h, positions], v[b, h, positions]
        output[:, :, block * 64:(block + 1) * 64] = F.scaled_dot_product_attention(q[:, :, block * 64:(block + 1) * 64], keys, values, attn_mask=mask[:, :, None])
        check_abort()
    return output


def launch(q, k, v, indices, lengths, valid, output, lse, config):
    batch, heads, tokens, dim = q.shape
    stages, warps = config
    _attn_fwd_bsa_varlen_align[(tokens // 64, batch * heads)](q, k, v, dim ** -.5, lse, output, indices, lengths, valid,
        *q.stride(), *k.stride(), *v.stride(), *output.stride(), *indices.stride(), *lengths.stride(), heads, tokens, dim,
        BLOCK_M=64, BLOCK_N_LG=64, SPARSITY=.75, HAS_KV_MASK=True, UNIT_INNER_STRIDE=all(x.stride(-1) == 1 for x in (q, k, v, output)), num_stages=stages, num_warps=warps)


def attend(q, k, v, indices, lengths, valid):
    if not (supported(q.device, q.dtype) and kernel_policy.allow_approximate()):
        return reference(q, k, v, indices, lengths, valid)
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    def scratch():
        count = min(q.shape[2], 1024)
        def sample(x):
            part = x[:1, :1, :count]
            if x.stride(-1) == 1:
                return part.contiguous().clone()
            # Tune the general-stride specialization when that is the live path.
            result = part.new_empty((*part.shape[:-1], part.shape[-1] * 2))[..., ::2]
            return result.copy_(part)
        sq, sk, sv = (sample(x) for x in (q, k, v))
        mask = valid[:count].clone()
        ids, lens = select_blocks(sq, sk, mask, True)
        out = torch.empty_like(sq)
        stats = torch.empty(sq.shape[:-1], device=q.device, dtype=torch.float32)
        expected = reference(sq, sk, sv, ids, lens, mask)
        launch(sq, sk, sv, ids, lens, mask, out, stats, (3, 4))
        torch.testing.assert_close(out, expected, atol=.03, rtol=.03)
        return lambda config: launch(sq, sk, sv, ids, lens, mask, out, stats, config)
    unit_stride = all(x.stride(-1) == 1 for x in (q, k, v, output))
    config = _TUNER.config('fixed64', f'dim={q.shape[-1]}|dtype={q.dtype}|unit_stride={unit_stride}|triton={triton.__version__}', q.device, [(3, 4), (2, 4), (3, 8)], scratch, q.shape[2] >= 256)
    launch(q, k, v, indices, lengths, valid, output, lse, config)
    return output


def sparse_attention(qkv_list, group, layout, grid):
    index, valid = layout
    padded = any(n % 4 for n in grid)
    blocked = []
    while qkv_list:
        source = qkv_list.pop(0)
        part = source.transpose(1, 2).index_select(2, index)
        part.masked_fill_(~valid[None, None, :, None], 0)
        blocked.append(part)
        del source, part
    q, k, v = blocked
    blocked.clear()
    indices, lengths = select_blocks(q, k, valid, padded)
    output = attend(q, k, v, indices, lengths, valid)
    del q, k, v, indices, lengths
    result = output.new_empty((output.shape[0], math.prod(grid), output.shape[1], output.shape[-1]))
    result[:, index[valid]] = output[:, :, valid].transpose(1, 2)
    return result
