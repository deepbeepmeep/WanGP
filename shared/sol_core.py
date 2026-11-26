"""WanGP Sol projection staging. The model owns the schedule and protected token range."""
import math
import os
from functools import lru_cache

from mmgp import cuda_graphs, offload

import torch

from . import sparse_kernels as kernels


WANGP_TAU = 0.8
_graph_layout = None


def routing_tau():
    tau = offload.shared_state.get("_sol3_tau")
    return WANGP_TAU if tau is None else tau


cuda_graphs.register_key(lambda: (_graph_layout, routing_tau() if offload.shared_state.get("_attention") == "sol3" else None))


class Layout:
    """Per-denoiser conditioning span and scheduled threshold, shared with its attention blocks."""
    sink_tokens = 0
    tau = 1.0

    def update(self, prefix, tau, mode):
        global _graph_layout
        self.sink_tokens, self.tau = int(prefix), float(tau)
        _graph_layout = (mode, self.sink_tokens, self.tau, _classic_precision()) if mode == "sol" else (
            (mode, self.sink_tokens) if mode == "sol3" else None)


@lru_cache(None)
def _classic_precision():
    value = os.environ.get("WGP_SOL_CUDA", "i8")
    return value if value in ("i8", "fp4", "i8_mixed", "fp4_mixed") else ""


def backend_for(device):
    if kernels.native(device) is not None:
        return "hip" if torch.version.hip else "cuda"
    return "triton" if kernels.triton_available(device) else None


@lru_cache(None)
def classic_available(device):
    if device.type != "cuda":
        return False
    if _classic_precision() and (kernels.native(device) is not None or kernels.triton_available(device)):
        return True
    if torch.version.hip:
        return False
    try:
        from shared.sol_attn import validate_runtime
        validate_runtime(device, torch.bfloat16)
    except (ImportError, RuntimeError):
        return False
    return True


@torch.compiler.disable
def staged_settings(device, dim, dtype, tokens, *, prefix=0, suffix=0, scale=None, classic=False, tau=WANGP_TAU, precision=None):
    if prefix < 0 or suffix < 0 or prefix and suffix or max(prefix, suffix) > tokens:
        raise ValueError("Specify one protected prefix or suffix within the token sequence")
    if classic and (tokens < 8192 or dtype != torch.bfloat16 or dim != 128):
        return None
    if tokens < 1024 or dim not in (64, 128) or dtype not in (torch.float16, torch.bfloat16) or torch.is_grad_enabled():
        return None
    if not classic and (prefix or suffix) == tokens:
        return None
    precision = (_classic_precision() if classic else "i8") if precision is None else precision
    packed = None
    if dim == 128 and precision and kernels.native(device) is not None and not torch.version.hip:
        from sparsegp import packed_cuda
        if packed_cuda.supported(device, dtype) and (not precision.startswith("fp4") or torch.cuda.get_device_capability(device) == (12, 0)):
            packed = packed_cuda
    full_classic = False
    engine = None
    if packed is None:
        if not classic:
            engine = kernels.mixed_engine(device)
        elif precision:
            engine = kernels.mixed_engine(device) if torch.version.hip else kernels.native(device)
    if classic and packed is None and engine is None:
        from shared.sol_attn import staged_supported, validate_runtime
        try:
            validate_runtime(device, dtype)
        except (ImportError, RuntimeError):
            engine = kernels.mixed_engine(device) if precision else None
            if engine is None:
                return None
        else:
            full_classic = not staged_supported(device, tokens)
    if not classic and not packed and engine is None:
        return None
    return dict(packed=packed, engine=engine, classic=classic, full_classic=full_classic, mixed=not classic or precision.endswith("_mixed"),
                fp4=precision.startswith("fp4"), start=0 if prefix else tokens-suffix, protected=prefix or suffix,
                scale=dim ** -0.5 if scale is None else scale, tau=tau)


def head_bytes(settings, batch, tokens, width, dtype, ratio=1):
    """Per-head storage for the kit's splitting decision, including the mixed routing bitmap."""
    n = (tokens + 63) // 64
    if settings["packed"]:
        # Q, one packed K/V record per block, and one centroid record per 16 blocks.
        packed = settings["packed"]
        record = packed._K4_RECORD if settings["fp4"] else packed._K8_RECORD
        rows = ratio * tokens * width * dtype.itemsize + n * record + ((n + packed.GROUP - 1) // packed.GROUP) * packed._C_RECORD
        if settings["mixed"]:
            # The token selector reads the original keys, flags, selected bits and partial moments.
            rows += tokens * width * dtype.itemsize + ratio * (n*n*9 + n*((tokens+255)//256)*12)
        if settings["protected"] and not settings["classic"]:
            rows += tokens * width * dtype.itemsize
    else:
        # FP16 keys and values, INT8 Q/K, 16-bit Q and output in the tile kernels.
        rows = tokens * width * (ratio * (dtype.itemsize + 1) + dtype.itemsize + 5)
        if settings["engine"] is not None:
            rows += n*n*21
            if settings["protected"]:
                rows += 2*tokens*width*dtype.itemsize
    return batch * rows


def _aligned(x):
    return (x.stride(-1) == 1 and x.data_ptr() % 16 == 0 and x.stride(1) % 8 == 0
            and x.stride(2) % 8 == 0 and max(x.stride()[1:3]) < 2**30)


def _packed_input(x):
    return x if _aligned(x) else x.clone(memory_format=torch.contiguous_format)


def _groups(x):
    for batch in range(x.size(0)):
        for start in range(0, x.size(2), 4):
            yield x[batch:batch+1, :, start:start+4]


def _means(rows, tokens):
    groups, padded, dim = rows.shape
    tiled = rows.view(groups, padded//64, 64, dim)
    var, mean = torch.var_mean(tiled.float(), dim=2, correction=0)
    if tokens % 64:
        var[:, -1], mean[:, -1] = torch.var_mean(tiled[:, -1, :tokens%64].float(), dim=1, correction=0)
    return mean.to(rows.dtype), var


def _representatives(rows, means):
    n = means.size(1)
    tail = torch.nn.functional.pad(means, (0, 0, 0, kernels.round_up(n)-n))
    return torch.cat((rows, tail), dim=1)


@torch.compiler.disable
def staged_quantize_v(v_list, settings):
    v = v_list.pop()
    packed = settings["packed"]
    exact = v if settings["protected"] and not settings["classic"] else None
    if packed:
        states = []
        for item in v.split(1):
            state = packed.allocate(item.size(1), item.size(2), item.device, fp4=settings["fp4"], mixed=settings["mixed"])
            packed.pack_values(state, _packed_input(item))
            states.append(state)
        return states, exact
    if settings["classic"] and settings["engine"] is None:
        return v, None
    groups = []
    tokens = v.size(1)
    lengths = (tokens - torch.arange((tokens+63)//64, device=v.device)*64).clamp(max=64).float()
    for item in _groups(v):
        rows = kernels.padded_rows(item, kernels.round_up(tokens))
        mean, var = _means(rows, tokens)
        disp = (lengths * var.mean(-1).clamp_min(0).sqrt().clamp_min(1.e-6)).log2()
        groups.append((_representatives(rows, mean), disp))
    return groups, exact


@torch.compiler.disable
def staged_quantize_k_with_v(k_list, values, settings):
    """Write K into the V stage's packed records before the kit projects Q."""
    k = k_list.pop()
    packed = settings["packed"]
    states, _ = values
    exact = k if settings["protected"] and not settings["classic"] else None
    if packed:
        prepared = []
        for state, item in zip(states, k.split(1)):
            packed.pack_keys(state, _packed_input(item))
            prepared.append(packed.finish(state))
        states.clear()
        return prepared, exact
    if settings["classic"] and settings["engine"] is None:
        if settings["full_classic"]:
            return k, None
        from shared.sol_attn import prepare_kv
        return prepare_kv(k, states), None
    prepared = []
    for item in _groups(k):
        rows = kernels.padded_rows(item, kernels.round_up(k.size(1)))
        mean, var = _means(rows, k.size(1))
        key, scale = settings["engine"].pack(_representatives(rows, mean))
        prepared.append((rows, mean, var, key, scale))
    return prepared, exact


def staged_quantize_q(q, settings):
    # Both routers read the normalized query. Packed CUDA packs it inside the attention launch.
    return q


def _block_attention(q, out, values, keys, settings):
    tokens = q.size(1)
    n = (tokens+63)//64
    scale = settings["scale"]
    s = scale * math.log2(math.e)
    fac = 0.5 * scale * scale * math.log2(math.e)
    engine = settings["engine"]
    idx = torch.arange(n, device=q.device)
    full_neighbors = (idx[:, None]-idx[None, :]).abs() <= 1
    start, length = settings["start"], settings["protected"]
    if length:
        full_neighbors |= ((idx*64 < start+length) & ((idx+1)*64 > start))[None, :]
    for item, target, (v, disp), (k, km, kvar, ki, ks) in zip(_groups(q), _groups(out), values, keys):
        rows = kernels.padded_rows(item, n*64)
        qm, _ = _means(rows, tokens)
        qf = qm.float()
        between, center = torch.var_mean(km.float(), dim=1, correction=0, keepdim=True)
        mu = (qf * center).sum(-1) * s
        q2 = qf.square()
        sigma = ((q2 * between).sum(-1) * s * s + 1.e-6).sqrt()
        proxy = torch.bmm(qm, km.transpose(1, 2)).float() * s
        full = (proxy > (mu + settings["tau"] * sigma)[:, :, None]) | full_neighbors
        if settings["mixed"]:
            risk = proxy + disp[:, None, :] + fac * torch.bmm(q2, kvar.transpose(1, 2))
            cut = mu + disp.mean(-1, keepdim=True) + fac * (q2 * kvar.mean(1, keepdim=True)).sum(-1)
            flags = torch.where(full, -1, (risk >= cut[:, :, None]).to(torch.int8)).to(torch.int8)
            bits = engine.select(qm, k, flags, tokens, s)
        else:
            bits = (-full.int()).unsqueeze(-1).expand(*full.shape,2).contiguous()
        qi, qs = engine.pack(rows)
        result = engine.attend(qi, ki, v, qs, ks, bits, tokens, 64, scale, True, False)
        target[0].copy_(result[:, :tokens].transpose(0, 1))


@torch.compiler.disable
def staged_attention(out, stages, settings, attn_mask=None):
    query, (keys, exact_k), (values, exact_v) = stages
    stages.clear()
    start, length = settings["start"], settings["protected"]
    dense = None
    if exact_v is not None:
        dense = kernels.dense(query[:, start:start+length].transpose(1, 2), exact_k.transpose(1, 2),
                              exact_v.transpose(1, 2), scale=settings["scale"]).transpose(1, 2)
        del exact_v, exact_k
    options = dict(tau=settings["tau"], scale=settings["scale"], sink_tokens=length, sink_start=start)
    if settings["packed"]:
        for q, target, state in zip(query.split(1), out.split(1), keys):
            destination = target if _aligned(target) else None
            packed = settings["packed"]
            ratio = q.size(2) // state[0].size(0)
            # Older wheels index K/V by the query head. Give each launch one
            # query per KV head, using strided views of Q and the output.
            lanes = 1 if getattr(packed, "GROUPED_QUERY", False) else ratio
            for lane in range(lanes):
                dest = None if destination is None else destination[:, :, lane::lanes]
                result = packed.attention(_packed_input(q[:, :, lane::lanes]), state, out=dest, **options)
                if dest is None:
                    target[:, :, lane::lanes].copy_(result)
    elif settings["full_classic"]:
        from shared.sol_attn import sol_attn
        out.copy_(sol_attn(query, keys, values, int8_qk=True, thresh_type="diag", **options))
    elif settings["classic"] and settings["engine"] is None:
        from shared.sol_attn import sol_attn_prepared
        sol_attn_prepared(query, values, keys, out=out, **options)
    else:
        kv_heads = sum(group[0].size(0) for group in keys) // query.size(0)
        ratio = query.size(2) // kv_heads
        for lane in range(ratio):
            _block_attention(query[:, :, lane::ratio], out[:, :, lane::ratio], values, keys, settings)
    if dense is not None:
        out[:, start:start+length].copy_(dense)
    return out


@torch.compiler.disable
def attention(qkv_list, settings, *, recycle_q=False):
    q, k, v = qkv_list
    qkv_list.clear()
    with torch.cuda.device(q.device):
        values = staged_quantize_v([v], settings)
        del v
        keys = staged_quantize_k_with_v([k], values, settings)
        del k
        out = q if recycle_q else torch.empty_like(q, memory_format=torch.contiguous_format)
        return staged_attention(out, [q, keys, values], settings)
