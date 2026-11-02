# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Control preprocessing and its on-demand assets."""

import torch
from shared.utils import files_locator as fl
from shared.utils.download import process_files_def_if_needed

DEPTH_VARIANT_CHOICES = [
    ("Depth Anything 2 Large (more precise, slower)", "vitl"),
    ("Depth Anything 2 Big (less precise, faster)", "vitb"),
    ("Depth Anything 3 Metric Large (better temporal stability ?)", "da3_metric_large"),
]


def depth_checkpoint_inventory():
    for label, variant in DEPTH_VARIANT_CHOICES:
        yield {"name": "Depth Anything", "variant": label.removeprefix("Depth Anything ").split(" (")[0],
               "download_defs": [query_preprocessor_download_def("depth", {"depth_anything_v2_variant": variant})]}


def query_preprocessor_download_def(process_type, server_config=None):
    if process_type in ("pose", "pose_align"):
        from preprocessing.dwpose.assets import query_download_def
    elif process_type == "depth":
        variant = (server_config or {}).get("depth_anything_v2_variant", "vitl")
        if variant == "da3_metric_large":
            from preprocessing.depth_anything_v3.assets import query_download_def
            return query_download_def(metric=True)
        from preprocessing.depth_anything_v2.assets import query_download_def
        return query_download_def(variant)
    elif process_type in ("canny", "scribble"):
        from preprocessing.scribble import query_download_def
    elif process_type == "flow":
        from preprocessing.flow import query_download_def
    else:
        return None
    return query_download_def()


def get_preprocessor(process_type, inpaint_color, pre_video_guide=None, server_config=None):
    server_config = server_config or {}
    process_files_def_if_needed(query_preprocessor_download_def(process_type, server_config))
    if process_type in ["pose", "pose_align"]:
        from preprocessing.dwpose.pose import PoseBodyFaceVideoAnnotator
        cfg_dict = {
            "DETECTION_MODEL": fl.locate_file("pose/yolox_l.onnx"),
            "POSE_MODEL": fl.locate_file("pose/dw-ll_ucoco_384.onnx"),
            "RESIZE_SIZE": 1024
        }
        if process_type == "pose_align" and torch.is_tensor(pre_video_guide) and pre_video_guide.ndim == 4 and pre_video_guide.shape[1] > 0:
            cfg_dict["REF_IMAGE"] = pre_video_guide[:, -1]
        anno_ins = lambda img: PoseBodyFaceVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="depth":

        depth_variant = server_config.get("depth_anything_v2_variant", "vitl")
        if depth_variant == "da3_metric_large":
            from preprocessing.depth_anything_v3.depth import DepthV3VideoAnnotator
            cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v3_metric_large_bf16.safetensors"),
                "MODEL_NAME": "da3metric-large",
                "PROCESS_RES": 0,
                "CHUNK_SIZE": -1,
                "CHUNK_OVERLAP": 8,
            }
            anno_ins = lambda img: DepthV3VideoAnnotator(cfg_dict).forward(img)
        elif depth_variant == "vitb":
            from preprocessing.depth_anything_v2.depth import DepthV2VideoAnnotator
            cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v2_vitb.pth"),
                'MODEL_VARIANT': 'vitb',
            }
            anno_ins = lambda img: DepthV2VideoAnnotator(cfg_dict).forward(img)
        else:
            from preprocessing.depth_anything_v2.depth import DepthV2VideoAnnotator
            cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v2_vitl.pth"),
                'MODEL_VARIANT': 'vitl'
            }
            anno_ins = lambda img: DepthV2VideoAnnotator(cfg_dict).forward(img)
    elif process_type=="gray":
        from preprocessing.gray import GrayVideoAnnotator
        cfg_dict = {}
        anno_ins = lambda img: GrayVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="canny":
        from preprocessing.canny import CannyVideoAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("scribble/netG_A_latest.pth")
            }
        anno_ins = lambda img: CannyVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="scribble":
        from preprocessing.scribble import ScribbleVideoAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("scribble/netG_A_latest.pth")
            }
        anno_ins = lambda img: ScribbleVideoAnnotator(cfg_dict).forward(img)
    elif process_type=="flow":
        from preprocessing.flow import FlowVisAnnotator
        cfg_dict = {
                "PRETRAINED_MODEL": fl.locate_file("flow/raft-things.pth")
            }
        anno_ins = lambda img: FlowVisAnnotator(cfg_dict).forward(img)
    elif process_type=="inpaint":
        color = tuple(int(v) for v in inpaint_color.view(-1).tolist())
        anno_ins = lambda img :  len(img) * [color]
    elif process_type == None or process_type in ["raw", "identity"]:
        anno_ins = lambda img : img
    else:
        raise Exception(f"process type '{process_type}' non supported")
    return anno_ins


def prepare_generation_assets(model_def, base_model_type, settings, server_config, guide_processes, mask_processes):
    """Preflight only selected features, before model allocation and temporary media."""
    video_mode = settings.get("video_prompt_type", "") or ""
    audio_mode = settings.get("audio_prompt_type", "") or ""
    guide = settings.get("video_guide")
    has_guide = guide is not None or settings.get("image_guide") is not None
    definitions = []
    if settings.get("image_refs") and settings.get("remove_background_images_ref", 0) > 0 and not model_def.get("no_background_removal", False):
        from preprocessing.rembg.assets import ensure_assets
        ensure_assets()
    if has_guide and not model_def.get("skip_video_guide_preprocess", False):
        if not model_def.get("custom_preprocessor"):
            kinds = [kind for letter, kind in guide_processes.items() if letter in video_mode]
            kinds += [kind for letter, kind in mask_processes.items() if letter in video_mode]
            definitions.extend(query_preprocessor_download_def(kind, server_config) for kind in dict.fromkeys(kinds))
        if "B" in video_mode:
            from models.hyvideo.data_kits.assets import query_download_def
            definitions.append(query_download_def())
        if base_model_type in ("scail", "steadydancer") or base_model_type.startswith("scail2"):
            from preprocessing.dwpose.assets import query_download_def
            definitions.append(query_download_def())
    if settings.get("image_refs") and (model_def.get("standin_class") or (model_def.get("lynx_class") and base_model_type not in ("lynx_lite", "vace_lynx_lite_14B"))):
        from models.hyvideo.data_kits.assets import query_download_def
        definitions.append(query_download_def())
    if has_guide and base_model_type == "scail":
        import re
        count = re.search(r"#(\d+)#", video_mode)
        if count is not None and int(count.group(1)) > 1:
            from preprocessing.matanyone.utils.model_assets import ensure_selected_matanyone_assets
            ensure_selected_matanyone_assets({"matanyone_version": "v1"})
    has_audio = settings.get("audio_guide") is not None or ("K" in audio_mode and has_guide)
    if has_audio:
        if "V" in audio_mode:
            from preprocessing.roformer.assets import query_download_def
            definitions.append(query_download_def())
        if "X" in audio_mode and settings.get("audio_guide2") is None:
            from preprocessing.speaker_separator.assets import download_speaker_separator
            download_speaker_separator()
        if base_model_type == "fantasy":
            from models.wan.fantasytalking.assets import query_download_def
            definitions.append(query_download_def())
        if model_def.get("multitalk_class", False):
            from models.wan.multitalk.assets import query_download_def
            definitions.append(query_download_def())
    for definition in definitions:
        process_files_def_if_needed(definition)
