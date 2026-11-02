# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prism Hadamard-folded GGUF weights; ordinary GGUF modules are unaffected."""
from typing import List, Optional

import torch

from .gguf import (GGUFFirstRowsLinear, GGUFWeightTensor, PrismQuantizationType, QLinearGGUF, QEmbedding,
                   _gguf_get_index, _gguf_cuda_module)


@torch.library.custom_op("wangp_prism::linear", mutates_args=())
def ptq_linear(input: torch.Tensor, raw: torch.Tensor, shape: List[int], bias: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    native = _gguf_cuda_module()
    if native is None or not native.supports_linear_qtype_name("PTQ1_0"):
        raise RuntimeError("PTQ1_0 requires WanGP llamacpp-gguf-cuda 1.0.22 or newer.")
    return native.linear(raw, "PTQ1_0", shape, input, bias, dtype)


@ptq_linear.register_fake
def _ptq_linear_fake(input, raw, shape, bias, dtype):
    return input.new_empty((*input.shape[:-1], shape[0]), dtype=dtype)


@torch.library.custom_op("wangp_prism::embedding", mutates_args=())
def ptq_embedding(indices: torch.Tensor, raw: torch.Tensor, shape: List[int], dtype: torch.dtype) -> torch.Tensor:
    native = _gguf_cuda_module()
    if native is None or not native.supports_embedding_qtype_name("PTQ1_0"):
        raise RuntimeError("PTQ1_0 requires WanGP llamacpp-gguf-cuda 1.0.22 or newer.")
    return native.embedding(raw, "PTQ1_0", shape, indices, dtype)


@ptq_embedding.register_fake
def _ptq_embedding_fake(indices, raw, shape, dtype):
    return indices.new_empty((*indices.shape, shape[1]), dtype=dtype)


@torch.library.custom_op("wangp_prism::hadamard", mutates_args=())
def _hadamard_cuda(x: torch.Tensor, signs: torch.Tensor, inverse: bool, grouped_shape: List[int]) -> torch.Tensor:
    return _gguf_cuda_module().prism_hadamard(x, signs, inverse, grouped_shape)


@_hadamard_cuda.register_fake
def _hadamard_fake(x, signs, inverse, grouped_shape):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("wangp_prism::decode", mutates_args=())
def _decode_cuda(x: torch.Tensor, raw: torch.Tensor, signs: torch.Tensor, bias: Optional[torch.Tensor], rows: int, grouped_shape: List[int], row_tile: int) -> torch.Tensor:
    return _gguf_cuda_module().prism_decode(x, raw, signs, bias, rows, grouped_shape, row_tile)


@_decode_cuda.register_fake
def _decode_fake(x, raw, signs, bias, rows, grouped_shape, row_tile):
    return x.new_empty((*x.shape[:-1], rows))


def install_prism_decode(model):
    # The native fusion is architecture-generic. Selection checks exact results
    # and benchmarks each launch before capture; Ampere+ also supports BF16.
    if torch.version.hip is not None or torch.cuda.get_device_capability(0)[0] < 8:
        return
    native = _gguf_cuda_module()
    if not callable(getattr(native, "has_prism_decode", None)) or not native.has_prism_decode():
        return
    for module in model.modules():
        if (isinstance(module, PrismLinear) or getattr(module, "_router_forward_impl", None) is PrismLinear.forward) and module.weight._tensor_type == PrismQuantizationType.PTQ1_0:
            module._prism_decode_enabled = True
    print("[GGUF][Prism] Fused Decode Available; Launch Selection Runs Before Graph Capture.")


def hadamard_reference(x, signs, inverse=False, grouped_shape=(0, 0, 0)):
    """CPU implementation and independent numerical reference for the CUDA FWHT."""
    shape = x.shape
    nk, rep, hd = grouped_shape
    if nk:
        x = x.reshape(*shape[:-1], rep, nk, hd).transpose(-3, -2).reshape(shape)
    y = x.float()
    if not inverse:
        y = y * signs
    y = y.reshape(-1, 1024)
    for stride in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        pairs = y.reshape(-1, 2, stride)
        a, b = pairs.unbind(1)
        y = torch.stack((a + b, a - b), dim=1).reshape(-1, 1024)
    y = y.reshape(shape) / 32
    if inverse:
        y = y * signs
    return y.to(x.dtype)


class PrismHadamard(torch.nn.Module):
    def __init__(self, signs, inverse=False, grouped_shape=(0, 0, 0)):
        super().__init__()
        self.register_buffer("signs", signs, persistent=False)
        self.inverse = inverse
        self.grouped_shape = list(grouped_shape)

    def forward(self, x):
        if x.device.type == "cuda":
            return _hadamard_cuda(x, self.signs, self.inverse, self.grouped_shape)
        return hadamard_reference(x, self.signs, self.inverse, self.grouped_shape)


class PrismLinear(QLinearGGUF):
    _prism_decode_enabled = False

    def forward(self, input):
        # the fused decode reads weights in VRAM: a block MMGP leaves in RAM (cpuCompute) takes QLinearGGUF's CPU branch
        if self._prism_decode_enabled and input.is_cuda and self.weight._data.is_cuda and input.numel() == input.shape[-1] and input.dtype == self._gguf_default_dtype and (self.bias is None or self.bias.dtype == input.dtype):
            from shared.kernels.prism_decode import select_decode
            transform = self.prism_transform
            raw = self.weight._data
            tile = select_decode(input, raw, transform.signs, self.bias, self.out_features, transform.grouped_shape, _gguf_cuda_module())
            if tile:
                return _decode_cuda(input, raw, transform.signs, self.bias, self.out_features, transform.grouped_shape, tile)
        return QLinearGGUF.forward(self, self.prism_transform(input))


class PrismEmbedding(QEmbedding):
    def forward(self, input):
        raw = self.weight._data
        if raw.device.type == "cuda":
            # MMGP can reuse the resident decoder embedding for vision while the
            # processor's token IDs are still on CPU.  The loaded weight owns the
            # execution device; keeping the decision tied to the indices would
            # run CPU dequantization and mix its output with CUDA Prism signs.
            if input.device != raw.device:
                input = input.to(raw.device, non_blocking=True)
            output = ptq_embedding(input, raw, list(self.weight.shape), self._gguf_default_dtype)
            return self._apply_embed_scale(output)
        return super().forward(input)

    def _apply_embed_scale(self, output):
        return super()._apply_embed_scale(self.prism_transform(output))


class PrismFirstRowsLinear(GGUFFirstRowsLinear):
    """Keep the target's input rotation when restricting the draft vocabulary."""
    def __init__(self, source, row_count):
        super().__init__(source, row_count)
        self.prism_transform = source.prism_transform

    def forward(self, input):
        return super().forward(self.prism_transform(input))


def get_prism_metadata(path):
    index = _gguf_get_index(path)
    metadata = dict(index.prism_metadata)
    if not metadata and any(t.tensor_type == PrismQuantizationType.PTQ1_0 for t in index.tensor_infos):
        raise ValueError("PTQ1_0 checkpoint is missing Prism Hadamard metadata.")
    if metadata:
        forward = metadata.get("prism.hadamard.weight_names", [])
        inverse = metadata.get("prism.hadamard.inverse_weight_names", [])
        if len(set(forward + inverse)) != len(forward + inverse):
            raise ValueError("Duplicate or conflicting Prism Hadamard weight names.")
        folded = set(forward + inverse)
        tensors = {t.name: t for t in index.tensor_infos}
        if folded - tensors.keys():
            raise ValueError("Prism Hadamard metadata references missing weights.")
        missing = {t.name for t in index.tensor_infos if t.tensor_type == PrismQuantizationType.PTQ1_0} - folded
        if missing:
            raise ValueError(f"PTQ1_0 weights have no Hadamard metadata: {sorted(missing)}")
    return metadata


def preserve_checkpoint_dtypes(model):
    model._convertWeightsFloatTo = None
    for module in model.modules():
        module._lock_dtype = None


@torch.no_grad()
def prepare_prism_gdn_layout(model, metadata):
    """Store GDN output rows in execution order, without unpacking weights.

    Prism's grouped SSM output basis cancels the GGUF head permutation. Move
    the remaining row permutations to loading so all recurrent math can stay
    grouped, including its convolution and speculative state snapshots.
    """
    if not metadata.get("prism.hadamard.gdn_v_grouped", False):
        return

    def reorder_parameter(module, name, rows):
        weight = module._parameters[name]
        rows = rows.to(weight.device)
        if isinstance(weight, GGUFWeightTensor):
            raw = weight._data.reshape(weight.shape[0], -1).index_select(0, rows)
            reordered = GGUFWeightTensor.create(
                raw, tuple(weight.shape), tuple(weight.stride()), weight.dtype,
                device=weight.device, requires_grad=False,
                tensor_type=weight._tensor_type, tensor_shape=tuple(weight.shape),
            )
        else:
            reordered = weight.detach().index_select(0, rows)
        module._parameters[name] = torch.nn.Parameter(reordered, requires_grad=False)

    for block in getattr(model, "blk", ()):
        if (getattr(block, "attn_qkv_gate", None) is None
                or not getattr(block, "_gguf_v_head_reordered", False)):
            continue
        if not block._gguf_ssm_param_reordered or block._gguf_interleave_ssm_ab:
            raise ValueError("Unsupported Prism GDN parameter layout.")
        nk, nv, hd = block.num_k_heads, block.num_v_heads, block.head_v_dim
        heads = torch.arange(nv, device="cpu").reshape(nv // nk, nk).t().reshape(-1)
        values = (heads[:, None] * hd + torch.arange(hd, device="cpu")).reshape(-1)
        qk = torch.arange(2 * block.key_dim, device="cpu")
        conv_rows = torch.cat((qk, 2 * block.key_dim + values))
        projection_rows = torch.cat((conv_rows, 2 * block.key_dim + block.value_dim + values))
        for module, rows in ((block.attn_qkv_gate, projection_rows),
                             (block.ssm_ab, torch.cat((heads, nv + heads))),
                             (block.ssm_conv1d, conv_rows)):
            reorder_parameter(module, "weight", rows)
            if module._parameters.get("bias") is not None:
                reorder_parameter(module, "bias", rows)
        for name in ("ssm_a", "ssm_dt"):
            reorder_parameter(block, name, heads)
        block._gguf_v_head_reordered = False
        block._gguf_ssm_param_reordered = False
        block._prism_gdn_grouped = True


def install_prism_transforms(model, metadata):
    """Called after compatible projection fusions and before MMGP profiling."""
    expected = dict(version=1, block_size=1024, transform="normalized-sylvester-walsh-hadamard",
                    axis="input-last-dimension", sign_mode="explicit")
    for key, value in expected.items():
        if metadata.get("prism.hadamard." + key) != value:
            raise ValueError(f"Unsupported Prism Hadamard {key}: {metadata.get('prism.hadamard.' + key)!r}")
    native = _gguf_cuda_module()
    if native is None or not native.supports_linear_qtype_name("PTQ1_0") or not callable(getattr(native, "prism_hadamard", None)):
        raise RuntimeError("Bonsai PTQ1_0 requires WanGP llamacpp-gguf-cuda 1.0.22 or newer, including Prism Hadamard kernels.")
    widths = metadata["prism.hadamard.sign_widths"]
    values = metadata["prism.hadamard.sign_values"]
    if len(set(widths)) != len(widths) or sum(widths) != len(values) or any(s not in (-1, 1) for s in values):
        raise ValueError("Invalid Prism Hadamard sign vectors.")
    signs, offset = {}, 0
    for width in widths:
        if width <= 0 or width % 1024:
            raise ValueError(f"Invalid Prism Hadamard width: {width}")
        signs[width] = torch.tensor(values[offset:offset + width], dtype=torch.int8, device="cpu")
        offset += width
    prepare_prism_gdn_layout(model, metadata)
    installed = set()
    for inverse, key in ((False, "weight_names"), (True, "inverse_weight_names")):
        names = list(metadata.get("prism.hadamard." + key, []))
        if getattr(model, "mtp", None) is not None:
            # Draft matrices are unrotated Q8; only its shared target embedding
            # and head carry the target's folded Hadamard representation.
            names.append("mtp.embed_tokens.weight" if inverse else "mtp.lm_head.weight")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate Prism Hadamard {key}.")
        for weight_name in names:
            module_name = weight_name.removesuffix(".weight")
            if module_name.endswith((".ffn_gate", ".ffn_up")):
                module_name = module_name.rsplit(".", 1)[0] + ".ffn_gate_up"
            if module_name.endswith((".attn_qkv", ".attn_gate")):
                parent = module_name.rsplit(".", 1)[0]
                if getattr(model.get_submodule(parent), "attn_qkv_gate", None) is not None:
                    module_name = parent + ".attn_qkv_gate"
            elif module_name.endswith((".attn_q", ".attn_k", ".attn_v")):
                parent = module_name.rsplit(".", 1)[0]
                if getattr(model.get_submodule(parent), "attn_qkv_full", None) is not None:
                    module_name = parent + ".attn_qkv_full"
            if module_name in installed:
                if inverse:
                    raise ValueError(f"Conflicting Prism transforms for {module_name}")
                continue
            module = model.get_submodule(module_name)
            expected_class = QEmbedding if inverse else QLinearGGUF
            is_router = not inverse and getattr(module, "_router_forward_impl", None) is QLinearGGUF.forward
            if not isinstance(module, expected_class) and not is_router:
                raise ValueError(f"Unsupported Prism transformed module: {module_name}")
            width = int(module.weight.shape[-1])
            if width not in signs:
                raise ValueError(f"Missing Prism sign vector for {module_name} width {width}")
            grouped_shape = (0, 0, 0)
            if metadata.get("prism.hadamard.gdn_v_grouped", False) and module_name.endswith(".ssm_out"):
                block = model.get_submodule(module_name.rsplit(".", 1)[0])
                if not getattr(block, "_prism_gdn_grouped", False):
                    grouped_shape = (block.num_k_heads, block.num_v_heads // block.num_k_heads, block.head_v_dim)
            module.prism_transform = PrismHadamard(signs[width], inverse, grouped_shape)
            module._prism_decode_enabled = False
            if is_router:
                module._router_forward_impl = PrismLinear.forward
            else:
                module.__class__ = PrismEmbedding if inverse else PrismLinear
            installed.add(module_name)
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, GGUFWeightTensor) and weight._tensor_type == PrismQuantizationType.PTQ1_0 and name not in installed:
            raise ValueError(f"PTQ1_0 weight has no activation transform: {name}")
    preserve_checkpoint_dtypes(model)
    print(f"[GGUF][Prism] Installed {len(installed)} Hadamard transforms with native CUDA kernels.")
