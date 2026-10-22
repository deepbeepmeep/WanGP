"""Affine conversion between MiniMax H3 AdaLN compression widths."""

from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file


FULL_TIME_DIM = 2688
_PRUNED_WIDTHS = (4, 8, 64)
_MAP_DIR = Path(__file__).with_name("lora_affine_maps")
_ARCHITECTURES = {
    "minimax_h3_fl2va": "fl2va",
    "minimax_h3_fl2va_pruned": "fl2va",
    "minimax_h3_control": "fl2va",
    "minimax_h3_control_pruned": "fl2va",
    "minimax_h3_ref2va": "ref2va",
    "minimax_h3_ref2va_pruned": "ref2va",
    "viggle_animate": "viggle",
    "minimax_h3_tts_ref2va_pruned": "ref2va",
    "minimax_h3_vdn": "fl2va",
    "minimax_h3_vdn_pruned": "fl2va",
}
_LORA_SUFFIXES = (
    ("lora_A.weight", "lora_B.weight"),
    ("lora_A.default.weight", "lora_B.default.weight"),
    ("lora_down.weight", "lora_up.weight"),
    ("lora_down.default.weight", "lora_up.default.weight"),
    ("lora.A.weight", "lora.B.weight"),
    ("lora.A.default.weight", "lora.B.default.weight"),
    ("lora.down.weight", "lora.up.weight"),
    ("lora.down.default.weight", "lora.up.default.weight"),
)


def _architecture(model_type):
    try:
        return _ARCHITECTURES[model_type]
    except KeyError as error:
        raise ValueError(f"Unsupported MiniMax H3 architecture for AdaLN LoRA conversion: {model_type}") from error


@lru_cache(maxsize=6)
def _load_affine_package(architecture, width=8):
    package_width = 8 if width == 4 else width
    tensors = load_file(str(_MAP_DIR / f"{architecture}_rank{package_width}.sft"), device="cpu")
    table, affine = tensors["adaln_t_table"].float(), tensors["adaln_affine_map"].float()
    if table.ndim != 2 or table.shape[1] != package_width or affine.shape != (package_width + 1, FULL_TIME_DIM):
        raise ValueError(f"Invalid MiniMax H3 {architecture} rank-{package_width} AdaLN affine package")
    if width == 4:
        table = table[:, :width]
        affine = torch.cat((affine[:width], affine[-1:]))
    return table, affine


def _aligned_affine_map(architecture, target_table):
    target_table = target_table.detach().to(device="cpu", dtype=torch.float64)
    if target_table.ndim != 2 or target_table.shape[1] not in _PRUNED_WIDTHS:
        raise ValueError(f"Unsupported MiniMax H3 {architecture} AdaLN target table shape {tuple(target_table.shape)}")
    canonical_table, canonical_affine = _load_affine_package(architecture, target_table.shape[1])
    if target_table.shape[0] != canonical_table.shape[0]:
        position = torch.linspace(0, canonical_table.shape[0] - 1, target_table.shape[0], dtype=torch.float64, device=target_table.device)
        lower = position.floor().long().clamp(max=canonical_table.shape[0] - 2)
        canonical_table = torch.lerp(canonical_table[lower].double(), canonical_table[lower + 1].double(), (position - lower).unsqueeze(1))
    elif torch.equal(target_table.float(), canonical_table):
        return canonical_affine
    ones = target_table.new_ones(target_table.shape[0], 1)
    target_h = torch.cat((target_table, ones), dim=1)
    canonical_h = torch.cat((canonical_table.double(), ones), dim=1)
    fit = torch.linalg.lstsq(target_h, canonical_h, rcond=1e-14)
    if int(fit.rank) != target_h.shape[1]:
        raise ValueError(f"MiniMax H3 {architecture} checkpoint has a rank-deficient AdaLN table")
    relative_error = torch.linalg.vector_norm(target_h @ fit.solution - canonical_h) / torch.linalg.vector_norm(canonical_h)
    if relative_error > 1e-4:
        raise ValueError(f"MiniMax H3 {architecture} checkpoint AdaLN table is incompatible with the canonical LoRA map (relative error {relative_error:.3g})")
    return (fit.solution @ canonical_affine.double()).float()


@lru_cache(maxsize=6)
def _canonical_encoder(architecture, width):
    _, affine = _load_affine_package(architecture, width)
    return torch.linalg.pinv(affine[:width].double(), rtol=1e-14).T.float()


def _add_bias_delta(state_dict, key, delta):
    existing = state_dict.get(key)
    if existing is not None:
        if existing.shape != delta.shape:
            raise ValueError(f"MiniMax H3 LoRA bias delta shape mismatch for {key}: {tuple(existing.shape)} != {tuple(delta.shape)}")
        delta.add_(existing.float())
    state_dict[key] = delta


def convert_adaln_loras(model_type, state_dict, target_table=None, hybrid_ref2va_blocks=None, ref2va_target_table=None):
    """Convert the AdaLN input width without changing the LoRA adapter rank."""
    architecture = _architecture(model_type)
    target_width = FULL_TIME_DIM if target_table is None else int(target_table.shape[1])
    if hybrid_ref2va_blocks is not None and (target_table is None or ref2va_target_table is None or
                                             ref2va_target_table.shape[1] != target_width):
        raise ValueError("MiniMax H3 hybrid AdaLN conversion requires matching FL2VA and Ref2VA target tables")
    candidates = []

    for down_suffix, up_suffix in _LORA_SUFFIXES:
        marker = "." + down_suffix
        for down_key in [key for key in state_dict if key.endswith(marker) and ".adaln_proj.linear." in key]:
            down = state_dict[down_key]
            if down.ndim != 2:
                continue
            module_name = down_key[:-len(marker)]
            up_key = module_name + "." + up_suffix
            if up_key not in state_dict:
                raise ValueError(f"MiniMax H3 LoRA is missing {up_key}")
            up = state_dict[up_key]
            if up.ndim != 2 or up.shape[1] != down.shape[0]:
                raise ValueError(f"MiniMax H3 LoRA factors are incompatible for {module_name}: A={tuple(down.shape)}, B={tuple(up.shape)}")
            candidates.append((module_name, down_key, up_key, int(down.shape[1])))

    for diff_key in [key for key in state_dict if key.endswith(".adaln_proj.linear.diff")]:
        diff = state_dict[diff_key]
        if diff.ndim == 2:
            candidates.append((diff_key[:-len(".diff")], diff_key, None, int(diff.shape[1])))

    if not candidates:
        return 0, architecture, target_width, target_width

    source_widths = {candidate[3] for candidate in candidates}
    if len(source_widths) != 1:
        raise ValueError(f"MiniMax H3 LoRA mixes AdaLN input widths: {sorted(source_widths)}")
    source_width = source_widths.pop()
    supported_widths = (*_PRUNED_WIDTHS, FULL_TIME_DIM)
    if source_width == target_width and hybrid_ref2va_blocks is None:
        return 0, architecture, source_width, target_width
    if source_width not in supported_widths or target_width not in supported_widths:
        raise ValueError(f"Unsupported MiniMax H3 AdaLN LoRA conversion {source_width} -> {target_width}; supported widths are {supported_widths}")

    source_affine = None if source_width == FULL_TIME_DIM else _load_affine_package(architecture, source_width)[1]
    source_encoder = None if source_width == FULL_TIME_DIM else _canonical_encoder(architecture, source_width)
    target_tables = {architecture: target_table}
    if hybrid_ref2va_blocks is not None:
        target_tables = {"fl2va": target_table, "ref2va": ref2va_target_table}
    target_affines = {target_architecture: None if table is None else _aligned_affine_map(target_architecture, table)
                      for target_architecture, table in target_tables.items()}

    converted_count = 0
    for module_name, down_key, up_key, _ in candidates:
        target_architecture = architecture
        if hybrid_ref2va_blocks is not None:
            block = int(module_name.split(".", 2)[1]) if module_name.startswith("blocks.") else -1
            target_architecture = "ref2va" if hybrid_ref2va_blocks[0] <= block <= hybrid_ref2va_blocks[1] else "fl2va"
        if source_width == target_width and target_architecture == architecture:
            continue
        down = state_dict[down_key]
        mapped = down.float() if source_encoder is None else down.float() @ source_encoder
        inner_bias = mapped.new_zeros(mapped.shape[0]) if source_affine is None else -(mapped @ source_affine[-1])
        target_affine = target_affines[target_architecture]
        if target_affine is not None:
            mapped = mapped @ target_affine.T
            inner_bias.add_(mapped[:, target_width])
            mapped = mapped[:, :target_width]
        state_dict[down_key] = mapped
        bias_delta = inner_bias if up_key is None else state_dict[up_key].float() @ inner_bias
        _add_bias_delta(state_dict, module_name + ".diff_b", bias_delta)
        converted_count += 1

    return converted_count, architecture, source_width, target_width
