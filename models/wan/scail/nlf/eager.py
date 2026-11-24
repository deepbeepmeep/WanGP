"""Eager mode loader for NLF multiperson model by DeepBeepMeep - enable Forward Pytorch Compatibility."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models

from . import field as nlf_field
from .nlf_model import NLFModel, NLFModelConfig
from .multiperson_model import MultipersonNLF
from .person_detector import PersonDetector


class MeanStdPreproc(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Buffer(torch.tensor(mean, dtype=torch.float32), persistent=False)
        self.std = nn.Buffer(torch.tensor(std, dtype=torch.float32), persistent=False)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return (inp - self.mean.to(inp.dtype)) / self.std.to(inp.dtype)


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _apply_centered_stride_padding_fix(
    backbone: nn.Module, *, centered_stride: bool, backbone_name: str
) -> None:
    """Match NLF's EfficientNet stride alignment when `centered_stride=True`.

    Upstream NLF uses a modified EfficientNet that applies asymmetric padding on the
    deepest strided MBConv block. Using the vanilla torchvision EfficientNet can yield
    an almost-constant pixel translation in the reconstructed 2D projections.
    """

    if not centered_stride:
        return
    if not backbone_name.startswith("efficientnet"):
        return

    try:
        import torchvision.models.efficientnet as tv_effnet
    except Exception:
        return

    candidates: list[tuple[int, nn.Module]] = []
    for module in backbone.modules():
        if not isinstance(module, tv_effnet.MBConv):
            continue
        if not (hasattr(module, "block") and len(module.block) > 1):
            continue

        cna = module.block[1]
        if not (isinstance(cna, nn.Sequential) and len(cna) > 0 and isinstance(cna[0], nn.Conv2d)):
            continue
        conv = cna[0]
        if tuple(conv.stride) != (2, 2):
            continue
        candidates.append((int(conv.in_channels), module))

    if not candidates:
        return

    _in_ch, target = max(candidates, key=lambda x: x[0])
    cna = target.block[1]
    if not (isinstance(cna, nn.Sequential) and len(cna) > 0 and isinstance(cna[0], nn.Conv2d)):
        return

    conv = cna[0]
    kernel = int(conv.kernel_size[0])
    if kernel <= 1:
        return

    pad_total = kernel - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # Apply `shifts=(1,1)` as in upstream `fixed_padding_layer(..., shifts=(1,1))`.
    left = pad_beg - 1
    right = pad_end + 1
    top = pad_beg - 1
    bottom = pad_end + 1
    if left < 0 or top < 0:
        return

    conv.padding = (0, 0)
    target.block[1] = nn.Sequential(nn.ZeroPad2d((left, right, top, bottom)), cna)


def _build_effnet_backbone(backbone_name: str, *, bn_eps: float = 1e-3) -> nn.Module:
    if backbone_name == 'efficientnetv2-l':
        return torchvision.models.efficientnet_v2_l(weights=None).features
    if backbone_name == 'efficientnetv2-s':
        return torchvision.models.efficientnet_v2_s(weights=None).features
    raise ValueError(f'Unsupported backbone: {backbone_name}')


def build_crop_model_from_state_dict(
    crop_state_dict: Dict[str, torch.Tensor], *, meta: Dict
) -> NLFModel:
    cm = meta['crop_model']
    config = NLFModelConfig(
        proc_side=int(cm['proc_side']),
        stride_test=int(cm['stride_test']),
        centered_stride=bool(cm['centered_stride']),
        backbone_link_dim=int(cm['backbone_link_dim']),
        depth=int(cm['depth']),
        box_size_m=float(cm['box_size_m']),
        uncert_bias=float(cm['uncert_bias']),
        uncert_bias2=float(cm['uncert_bias2']),
        fix_uncert_factor=bool(cm['fix_uncert_factor']),
        mix_3d_inside_fov=float(cm['mix_3d_inside_fov']),
        weak_perspective=bool(cm['weak_perspective']),
    )

    backbone_name = str(cm.get('backbone', 'efficientnetv2-l'))
    backbone_channels = int(crop_state_dict['heatmap_head.layer.0.weight'].shape[1])
    backbone = nn.Sequential(
        MeanStdPreproc(mean=0.5, std=0.5),
        _build_effnet_backbone(backbone_name),
    )

    pred_keys = [
        k
        for k in crop_state_dict.keys()
        if k.startswith('heatmap_head.weight_field.pred_mlp.') and k.endswith('.weight')
    ]
    pred_keys_sorted = sorted(pred_keys, key=lambda x: int(x.split('.')[3]))
    field_posenc_dim = int(crop_state_dict[pred_keys_sorted[0]].shape[1])
    layer_dims = [int(crop_state_dict[k].shape[0]) for k in pred_keys_sorted]

    gps_pos_enc_dim = int(
        crop_state_dict['heatmap_head.weight_field.gps_net.learnable_fourier.linear.weight'].shape[0]
        * 2
    )
    gps_hidden_dim = int(
        crop_state_dict['heatmap_head.weight_field.gps_net.mlp.0.weight'].shape[0]
    )
    gps_output_dim = int(
        crop_state_dict['heatmap_head.weight_field.gps_net.mlp.2.weight'].shape[0]
    )

    gps_net = nlf_field.GPSNet(
        pos_enc_dim=gps_pos_enc_dim,
        hidden_dim=gps_hidden_dim,
        output_dim=gps_output_dim,
        load_from_projdir=False,
    )
    weight_field = nlf_field.GPSField(
        gps_net,
        layer_dims=layer_dims,
        posenc_dim=field_posenc_dim,
    )

    normalizer = partial(nn.BatchNorm2d, eps=1e-3)

    n_left = int(crop_state_dict['canonical_lefts'].shape[0])
    n_center = int(crop_state_dict['canonical_centers'].shape[0])
    n_joints = int(crop_state_dict['inv_permutation'].shape[0])

    crop_model = NLFModel(
        backbone,
        weight_field,
        normalizer,
        backbone_channels=backbone_channels,
        config=config,
        n_joints=n_joints,
        n_left_joints=n_left,
        n_center_joints=n_center,
    )
    crop_model.load_state_dict(crop_state_dict, strict=False)
    with torch.no_grad():
        if 'inv_permutation' in crop_state_dict:
            crop_model.inv_permutation.copy_(crop_state_dict['inv_permutation'])
        if 'canonical_locs_init' in crop_state_dict:
            crop_model.canonical_locs_init.copy_(crop_state_dict['canonical_locs_init'])
        if 'canonical_delta_mask' in crop_state_dict:
            crop_model.canonical_delta_mask.copy_(crop_state_dict['canonical_delta_mask'])
        # Handle scalar vs tensor mismatch (safetensors doesn't preserve 0-dim tensors well)
        if 'backbone.0.mean' in crop_state_dict:
            mean_val = crop_state_dict['backbone.0.mean']
            if mean_val.dim() == 0:
                mean_val = mean_val.unsqueeze(0)
            crop_model.backbone[0].mean = nn.Buffer(mean_val.to(dtype=torch.float32), persistent=False)
        if 'backbone.0.std' in crop_state_dict:
            std_val = crop_state_dict['backbone.0.std']
            if std_val.dim() == 0:
                std_val = std_val.unsqueeze(0)
            crop_model.backbone[0].std = nn.Buffer(std_val.to(dtype=torch.float32), persistent=False)

        wf = crop_model.heatmap_head.weight_field
        if 'heatmap_head.weight_field.r_sqrt_eigva' in crop_state_dict:
            wf.r_sqrt_eigva.copy_(crop_state_dict['heatmap_head.weight_field.r_sqrt_eigva'])
        if 'heatmap_head.weight_field.gps_net.mini' in crop_state_dict:
            wf.gps_net.mini.copy_(crop_state_dict['heatmap_head.weight_field.gps_net.mini'])
        if 'heatmap_head.weight_field.gps_net.maxi' in crop_state_dict:
            wf.gps_net.maxi.copy_(crop_state_dict['heatmap_head.weight_field.gps_net.maxi'])
        if 'heatmap_head.weight_field.gps_net.center' in crop_state_dict:
            wf.gps_net.center.copy_(crop_state_dict['heatmap_head.weight_field.gps_net.center'])

    if isinstance(crop_model.backbone, nn.Sequential) and len(crop_model.backbone) > 1:
        _apply_centered_stride_padding_fix(
            crop_model.backbone[1],
            centered_stride=bool(cm.get('centered_stride', False)),
            backbone_name=backbone_name,
        )

    return crop_model


def _body_model_from_tensors(
    *,
    model_name: str,
    gender: str,
    tensors: Dict[str, torch.Tensor],
) -> nn.Module:
    import smplfitter.pt

    BodyModel = smplfitter.pt.BodyModel
    model = BodyModel.__new__(BodyModel)
    nn.Module.__init__(model)
    model.gender = gender
    model.model_name = model_name

    model.v_template = nn.Buffer(tensors['v_template'].to(dtype=torch.float32), persistent=False)
    model.shapedirs = nn.Buffer(tensors['shapedirs'].to(dtype=torch.float32), persistent=False)
    model.posedirs = nn.Buffer(tensors['posedirs'].to(dtype=torch.float32), persistent=False)
    model.J_regressor_post_lbs = nn.Buffer(
        tensors['J_regressor_post_lbs'].to(dtype=torch.float32), persistent=False
    )
    model.J_template = nn.Buffer(tensors['J_template'].to(dtype=torch.float32), persistent=False)
    model.J_shapedirs = nn.Buffer(tensors['J_shapedirs'].to(dtype=torch.float32), persistent=False)
    model.kid_shapedir = nn.Buffer(
        tensors['kid_shapedir'].to(dtype=torch.float32), persistent=False
    )
    model.kid_J_shapedir = nn.Buffer(
        tensors['kid_J_shapedir'].to(dtype=torch.float32), persistent=False
    )
    model.weights = nn.Buffer(tensors['weights'].to(dtype=torch.float32), persistent=False)
    model.kintree_parents_tensor = nn.Buffer(
        tensors['kintree_parents_tensor'].to(dtype=torch.int64), persistent=False
    )

    model.kintree_parents = [int(x) for x in model.kintree_parents_tensor.cpu().tolist()]
    model.faces = np.zeros((0, 3), dtype=np.int64)
    model.num_joints = int(model.J_template.shape[0])
    model.num_vertices = int(model.v_template.shape[0])
    model.num_betas = int(model.shapedirs.shape[2])
    model.vertex_subset = np.arange(model.num_vertices)
    return model


def _build_body_models_and_fitters(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[nn.ModuleDict, nn.ModuleDict]:
    import smplfitter.pt

    body_models = nn.ModuleDict()
    fitters = nn.ModuleDict()

    for model_name in ('smpl', 'smplx'):
        full_tensors = _strip_prefix(state_dict, f'body_models.{model_name}.')
        partial_tensors = _strip_prefix(state_dict, f'fitters.{model_name}.body_model.')
        if not full_tensors or not partial_tensors:
            raise KeyError(
                f'Missing {model_name} SMPL tensors in checkpoint state_dict (body_models / fitters).'
            )

        body_models[model_name] = _body_model_from_tensors(
            model_name=model_name, gender='neutral', tensors=full_tensors
        )
        partial_body = _body_model_from_tensors(
            model_name=model_name, gender='neutral', tensors=partial_tensors
        )
        fitters[model_name] = smplfitter.pt.BodyFitter(partial_body)

    return body_models, fitters


def _load_safetensors_checkpoint(checkpoint_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Load checkpoint from safetensors format with separate meta.json."""
    from safetensors.torch import load_file

    # Load tensors
    state_dict = load_file(str(checkpoint_path))

    # Load meta from JSON
    meta_path = checkpoint_path.with_suffix('.meta.json')
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, 'r') as f:
        meta_json = json.load(f)

    # Reconstruct cano_all from prefixed tensors
    cano_all = {}
    cano_all_keys = meta_json.get('cano_all_keys', [])
    for k in cano_all_keys:
        tensor_key = f'__meta_cano_all_{k}'
        if tensor_key in state_dict:
            cano_all[k] = state_dict.pop(tensor_key)

    # Reconstruct skeleton_infos from prefixed tensors and meta
    skeleton_infos = {}
    skeleton_meta = meta_json.get('skeleton_infos', {})
    for skel_name, skel_info in skeleton_meta.items():
        skeleton_infos[skel_name] = {'names': skel_info.get('names', [])}
        indices_key = f'__meta_skeleton_{skel_name}_indices'
        edges_key = f'__meta_skeleton_{skel_name}_edges'
        if indices_key in state_dict:
            skeleton_infos[skel_name]['indices'] = state_dict.pop(indices_key)
        if edges_key in state_dict:
            skeleton_infos[skel_name]['edges'] = state_dict.pop(edges_key)

    # Build full meta dict
    meta = {
        'crop_model': meta_json.get('crop_model', {}),
        'pad_white_pixels': meta_json.get('pad_white_pixels', True),
        'cano_all': cano_all,
        'skeleton_infos': skeleton_infos,
    }

    return state_dict, meta


def load_multiperson_nlf_eager(
    *,
    checkpoint_path: str | Path,
    device: str | torch.device = 'cuda',
) -> MultipersonNLF:
    checkpoint_path = Path(checkpoint_path)

    # Support both .pth and .safetensors formats
    if checkpoint_path.suffix == '.safetensors':
        state_dict, meta = _load_safetensors_checkpoint(checkpoint_path)
    elif checkpoint_path.suffix == '.pth':
        ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            meta = ckpt.get('meta', {})
        else:
            raise ValueError(
                'Unsupported checkpoint format. Expected dict with `state_dict` and `meta`.'
            )
    else:
        raise ValueError(f'Unsupported checkpoint format: {checkpoint_path.suffix}')

    crop_state_dict = _strip_prefix(state_dict, 'crop_model.')
    crop_model = build_crop_model_from_state_dict(crop_state_dict, meta=meta)

    crop_model.backbone.half()
    crop_model.heatmap_head.layer.half()

    body_models, fitters = _build_body_models_and_fitters(state_dict)

    detector = PersonDetector()

    skeleton_infos = meta['skeleton_infos']
    pad_white_pixels = bool(meta.get('pad_white_pixels', True))
    cano_all = meta['cano_all']

    model = MultipersonNLF(
        crop_model,
        detector,
        skeleton_infos,
        pad_white_pixels=pad_white_pixels,
        cano_all=cano_all,
        body_models=body_models,
        fitters=fitters,
    )
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    model.cano_all = {k: v.to(device=device, dtype=torch.float32) for k, v in model.cano_all.items()}
    return model
