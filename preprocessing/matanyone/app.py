import sys

import os
import json
import re
import gc
import time
from contextlib import nullcontext
# import ffmpeg
import imageio
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from datetime import datetime
from .tools.painter import mask_painter, point_painter
from .tools.interact_tools import SamControler
from .tools.misc import get_device
from .tools.base_segmenter import set_image_encoder_patch
from .utils.model_assets import MATANYONE_SAM3, ensure_selected_matanyone_assets, get_matanyone_title_html, get_selected_matanyone_version, load_selected_matanyone_model
from .matanyone.inference.inference_core import InferenceCore
from .matanyone_wrapper import matanyone
from shared.utils.audio_video import save_video, save_image
from mmgp import offload
from shared.utils import files_locator as fl 
from shared.utils.utils import truncate_for_filesystem, sanitize_file_name, process_images_multithread, calculate_new_dimensions, get_default_workers
from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources, any_GPU_process_running
from preprocessing.sam3.logger import get_logger

from functools import wraps
import inspect
from shared.gradio.progress import WangpProgress
from shared.utils.download_progress import download_operation

logger = get_logger(__name__)


def mask_action(fn):
    signature = inspect.signature(fn)

    @wraps(fn)
    def run(*args, progress=None, **kwargs):
        progress = progress or WangpProgress(track_tqdm=True)
        arguments = signature.bind(*args, **kwargs).arguments
        if any(key in arguments and arguments[key] is None for key in ("image_input", "video_input")):
            return fn(*args, **kwargs)
        progress(0, desc="Preparing Mask Models")
        with download_operation(progress.gen):
            ensure_selected_matanyone_assets(server_config_ref, gen=progress.gen)
            progress.check_cancelled()
            return fn(*args, **kwargs)

    run.__signature__ = signature.replace(parameters=[*signature.parameters.values(), inspect.Parameter("progress", inspect.Parameter.KEYWORD_ONLY, default=WangpProgress(track_tqdm=True))])
    return run


arg_device = str(get_device())
arg_sam_model_type="vit_h"
arg_mask_save = False
model_loaded = False
model = None
matanyone_model = None
model_in_GPU = False
matanyone_in_GPU = False
bfloat16_supported = False
PlugIn = None
server_config_ref = None
loaded_matanyone_version = None
sam3_predictor = None
sam3_click_session = None
SAM3_MATANYONE_FILL_HOLE_AREA = 2
MATANYONE_MASK_TYPE_CHOICES = [
    ("Grey with Alpha (used by WanGP)", "wangp"),
    ("Matted Video", "greenscreen"),
    ("RGB With Alpha Channel (local Zip file)", "alpha"),
]
SAM3_MASK_TYPE_CHOICES = [
    ("B & W (used by WanGP)", "wangp"),
    ("Matted Video", "greenscreen"),
]
MATTE_COLORS = {
    "Green": (0, 255, 0), "Red": (255, 0, 0), "Orange": (255, 165, 0),
    "Yellow": (255, 255, 0), "Cyan": (0, 255, 255), "Blue": (0, 0, 255),
    "Indigo": (75, 0, 130), "Violet": (143, 0, 255), "White": (255, 255, 255), "Black": (0, 0, 0),
}

# SAM generator
import copy
GPU_process_was_running = False
def acquire_GPU(state):
    global GPU_process_was_running
    GPU_process_was_running = any_GPU_process_running(state, "matanyone")
    acquire_GPU_ressources(state, "matanyone", "MatAnyone", gr= gr)
def release_GPU(state):
    release_GPU_ressources(state, "matanyone")
    if GPU_process_was_running:
        global matanyone_in_GPU, model_in_GPU
        if model_in_GPU:  
            model.samcontroler.sam_controler.model.to("cpu")
            model_in_GPU = False
        if matanyone_in_GPU:
            matanyone_model.to("cpu")
            matanyone_in_GPU = False


def perform_spatial_upsampling(frames, new_dim):
    if new_dim =="":
        return frames
    h, w = frames[0].shape[:2]

    from shared.utils.utils import resize_lanczos 
    pos = new_dim.find(" ")
    fit_into_canvas = "Outer" in new_dim
    new_dim = new_dim[:pos]
    if new_dim == "1080p":
        canvas_w, canvas_h = 1920, 1088
    elif new_dim == "720p":
        canvas_w, canvas_h = 1280, 720
    else:
        canvas_w, canvas_h = 832, 480
    h, w = calculate_new_dimensions(canvas_h, canvas_w, h, w, fit_into_canvas=fit_into_canvas,  block_size= 16  )


    def upsample_frames(frame):
        return np.array(Image.fromarray(frame).resize((w,h), resample=Image.Resampling.LANCZOS))

    output_frames = process_images_multithread(upsample_frames, frames, "upsample", wrap_in_list = False, max_workers=get_default_workers(), in_place=True)    
    return output_frames

class MaskGenerator():
    def __init__(self, sam_checkpoint, device):
        global args_device
        args_device  = device
        self.samcontroler = SamControler(sam_checkpoint, arg_sam_model_type, arg_device)
       
    def first_frame_click(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True):
        mask, logit, painted_image = self.samcontroler.first_frame_click(image, points, labels, multimask)
        return mask, logit, painted_image

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


def is_sam3_selected():
    return get_selected_matanyone_version(server_config_ref) == MATANYONE_SAM3


def _matanyone_morphology_visibility():
    return gr.update(visible=not is_sam3_selected())


def _matanyone_mask_type_choices(media_type="Video"):
    choices = SAM3_MASK_TYPE_CHOICES if is_sam3_selected() else MATANYONE_MASK_TYPE_CHOICES
    return [(f"Matted {media_type}" if value == "greenscreen" else label, value) for label, value in choices]


def _matanyone_background_color_update(mask_type):
    return gr.update(visible=mask_type == "greenscreen")


def _matanyone_mask_type_update(mask_type, media_type="Video"):
    choices = _matanyone_mask_type_choices(media_type)
    values = [value for _, value in choices]
    mask_type = mask_type if mask_type in values else "wangp"
    return gr.update(choices=choices, value=mask_type), _matanyone_background_color_update(mask_type)


def _matte_frame(frame, alpha, background_color):
    alpha = alpha.astype(np.uint16)
    color = np.asarray(MATTE_COLORS[background_color], dtype=np.uint16)
    return ((frame.astype(np.uint16) * (255 - alpha) + color * alpha) // 255).astype(np.uint8)


def _matanyone_mask_output_button_label():
    return "B & W Mask Output" if is_sam3_selected() else "Alpha Mask Output"


def _matanyone_mask_output_button_update():
    return gr.update(value=_matanyone_mask_output_button_label())


def _ensure_sam3_predictor():
    global sam3_predictor, loaded_matanyone_version, model_loaded
    if sam3_predictor is None:
        ensure_selected_matanyone_assets(server_config_ref)
        from preprocessing.sam3.preprocessor import load_sam3_mask_predictor

        sam3_predictor = load_sam3_mask_predictor(include_text_encoder=False, postprocess_batch_size=1, use_batched_grounding=True, manual_model_loading=True)
        sam3_predictor.load_model_to_gpu()
        loaded_matanyone_version = MATANYONE_SAM3
        model_loaded = True
    return sam3_predictor


def _sam3_load_model_to_gpu():
    if sam3_predictor is not None and hasattr(sam3_predictor, "load_model_to_gpu"):
        sam3_predictor.load_model_to_gpu()


def _sam3_start_session(video_state, start_frame=0, end_frame=None, cache_frame_outputs=True):
    predictor = _ensure_sam3_predictor()
    _sam3_load_model_to_gpu()
    frames = [Image.fromarray(frame) for frame in video_state["origin_images"][start_frame:end_frame]]
    response = predictor.handle_request({"type": "start_session", "resource_path": frames, "offload_video_to_cpu": False, "cache_frame_outputs": cache_frame_outputs})
    return response["session_id"]


def _sam3_start_frame_session(frame):
    predictor = _ensure_sam3_predictor()
    _sam3_load_model_to_gpu()
    response = predictor.handle_request({"type": "start_session", "resource_path": [Image.fromarray(frame)], "offload_video_to_cpu": False})
    return response["session_id"]


def _sam3_close_session(session_id):
    if sam3_predictor is not None and session_id is not None:
        sam3_predictor.handle_request({"type": "close_session", "session_id": session_id})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sam3_close_click_session():
    global sam3_click_session
    if sam3_click_session is not None:
        _sam3_close_session(sam3_click_session["session_id"])
        sam3_click_session = None


def _sam3_get_click_session(video_state, frame_idx):
    global sam3_click_session
    frame = video_state["origin_images"][frame_idx]
    frame_identity = id(frame)
    if sam3_click_session is None or sam3_click_session["frame_idx"] != frame_idx or sam3_click_session["frame_identity"] != frame_identity:
        _sam3_close_click_session()
        sam3_click_session = {
            "frame_idx": frame_idx,
            "frame_identity": frame_identity,
            "session_id": _sam3_start_frame_session(frame),
        }
    return sam3_click_session["session_id"]


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sam3_autocast_context():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sam3_bf16_prompt_payload(value):
    if torch.is_tensor(value):
        return value.to(dtype=torch.bfloat16) if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: _sam3_bf16_prompt_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sam3_bf16_prompt_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sam3_bf16_prompt_payload(item) for item in value)
    return value


def _sam3_points_payload(points):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return torch.as_tensor(points, dtype=dtype)


def _sam3_labels_payload(labels):
    return torch.as_tensor(labels, dtype=torch.int32)


def _sam3_outputs_to_mask(outputs, height, width, obj_ids=None, fill_hole_area=0):
    if outputs is None or "out_binary_masks" not in outputs:
        return np.zeros((height, width), dtype=np.uint8)
    masks = _to_numpy(outputs["out_binary_masks"])
    if masks.size == 0:
        return np.zeros((height, width), dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim > 3:
        masks = masks.reshape((-1, *masks.shape[-2:]))
    if obj_ids is not None:
        out_obj_ids = _to_numpy(outputs.get("out_obj_ids", np.arange(masks.shape[0])))
        keep = np.isin(out_obj_ids, np.asarray(list(obj_ids)))
        masks = masks[keep]
        if masks.size == 0:
            return np.zeros((height, width), dtype=np.uint8)
    if masks.shape[-2:] != (height, width):
        masks = np.stack([cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) for mask in masks], axis=0)
    mask = masks.astype(bool).any(axis=0)
    if fill_hole_area > 0:
        from preprocessing.sam3.preprocessor import fill_sam3_binary_mask_holes

        mask = fill_sam3_binary_mask_holes(mask, fill_hole_area)
    return mask.astype(np.uint8)


def _paint_sam3_mask(image, mask, points=None, labels=None, mask_color=3):
    painted = mask_painter(image, mask.astype("uint8"), mask_color=mask_color) if np.any(mask) else image.copy()
    if points is not None and labels is not None:
        height, width = image.shape[:2]
        points = np.asarray(points, dtype=np.int32)
        if points.size > 0:
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        labels = np.asarray(labels)
        if np.any(labels > 0):
            painted = point_painter(painted, points[labels > 0], 8, 0.9, 15, 2, 5)
        if np.any(labels < 1):
            painted = point_painter(painted, points[labels < 1], 50, 0.9, 15, 2, 5)
    return Image.fromarray(painted)


def _parse_sam3_keywords(keyword_text):
    return [keyword.strip() for keyword in re.split(r"[\n,;]+", keyword_text or "") if keyword.strip()]


def _sam3_relative_points(points, width, height):
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size == 0:
        return points.reshape(0, 2).tolist()
    points[:, 0] = np.clip(points[:, 0], 0, width - 1) / max(width - 1, 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1) / max(height - 1, 1)
    return points.tolist()


def _sam3_preview_point_mask(video_state, frame_idx, points, labels):
    frame = video_state["origin_images"][frame_idx]
    height, width = frame.shape[:2]
    session_id = _sam3_get_click_session(video_state, frame_idx)
    from preprocessing.sam3.preprocessor import encode_sam3_keyword_prompts

    preencoded = _sam3_bf16_prompt_payload(encode_sam3_keyword_prompts(["<text placeholder>"], keep_text_encoder_loaded=True)["<text placeholder>"])
    with _sam3_autocast_context():
        result = sam3_predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            # This preview session contains only the selected source frame, so its local index is 0.
            "frame_index": 0,
            "points": _sam3_points_payload(_sam3_relative_points(points, width, height)),
            "point_labels": _sam3_labels_payload(labels),
            "obj_id": 1,
            "rel_coordinates": True,
            "clear_old_points": True,
            "preencoded_text_outputs": preencoded,
        })
    return _sam3_outputs_to_mask(result["outputs"], height, width, fill_hole_area=SAM3_MATANYONE_FILL_HOLE_AREA)


def _sam3_preview_keyword_mask(video_state, frame_idx, keyword):
    frame = video_state["origin_images"][frame_idx]
    height, width = frame.shape[:2]
    session_id = _sam3_start_frame_session(frame)
    try:
        from preprocessing.sam3.preprocessor import encode_sam3_keyword_prompts

        preencoded = _sam3_bf16_prompt_payload(encode_sam3_keyword_prompts([keyword], keep_text_encoder_loaded=True)[keyword])
        # This preview session contains only the selected source frame, so its local index is 0.
        with _sam3_autocast_context():
            result = sam3_predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": keyword, "preencoded_text_outputs": preencoded})
        return _sam3_outputs_to_mask(result["outputs"], height, width, fill_hole_area=SAM3_MATANYONE_FILL_HOLE_AREA)
    finally:
        _sam3_close_session(session_id)


def _selected_sam3_prompts(interactive_state, mask_dropdown):
    multi_mask = interactive_state.get("multi_mask", {})
    prompts = multi_mask.get("sam3_prompts", [])
    if len(prompts) == 0:
        current_prompt = interactive_state.get("sam3_current_prompt")
        return [current_prompt] if current_prompt is not None else []
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    selected = []
    for mask_name in sorted(mask_dropdown):
        try:
            mask_number = int(mask_name.split("_")[1]) - 1
        except (IndexError, ValueError):
            continue
        if 0 <= mask_number < len(prompts) and prompts[mask_number] is not None:
            selected.append(prompts[mask_number])
    return selected


def _sam3_propagate_keywords(video_state, keyword_prompts, start_frame, end_frame):
    frames = video_state["origin_images"][start_frame:end_frame]
    if len(keyword_prompts) == 0 or len(frames) == 0:
        return []

    _sam3_close_click_session()
    from preprocessing.sam3.preprocessor import encode_sam3_keyword_prompts

    alpha = [np.zeros((*frames[0].shape[:2], 1), dtype=np.uint8) for _ in frames]
    video_pil = [Image.fromarray(frame) for frame in frames]
    keywords = sorted({prompt["keyword"] for prompt in keyword_prompts})
    logger.info("SAM3 encoding keywords before propagation: %s", ", ".join(f"'{keyword}'" for keyword in keywords))
    preencoded_prompts = encode_sam3_keyword_prompts(keywords, keep_text_encoder_loaded=True)
    for prompt in keyword_prompts:
        local_frame_idx = prompt["frame_idx"] - start_frame
        if local_frame_idx < 0 or local_frame_idx >= len(frames):
            continue
        session_id = None
        try:
            logger.info("SAM3 keyword currently being processed: '%s'", prompt["keyword"])
            preencoded = _sam3_bf16_prompt_payload(preencoded_prompts[prompt["keyword"]])
            _sam3_load_model_to_gpu()
            response = _ensure_sam3_predictor().handle_request({"type": "start_session", "resource_path": video_pil, "offload_video_to_cpu": False, "cache_frame_outputs": False})
            session_id = response["session_id"]
            with _sam3_autocast_context():
                sam3_predictor.handle_request({"type": "add_prompt", "session_id": session_id, "frame_index": local_frame_idx, "text": prompt["keyword"], "preencoded_text_outputs": preencoded})
                for result in sam3_predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": local_frame_idx,
                    "max_frame_num_to_track": len(frames) - local_frame_idx,
                }):
                    frame_idx = result["frame_index"]
                    if local_frame_idx <= frame_idx < len(frames):
                        alpha[frame_idx][:, :, 0] |= _sam3_outputs_to_mask(result["outputs"], *frames[frame_idx].shape[:2], fill_hole_area=SAM3_MATANYONE_FILL_HOLE_AREA) * 255
        finally:
            _sam3_close_session(session_id)
    return alpha


def _sam3_propagate_prompts(video_state, prompts, start_frame, end_frame):
    frames = video_state["origin_images"][start_frame:end_frame]
    height, width = frames[0].shape[:2]
    alpha = [np.zeros((height, width, 1), dtype=np.uint8) for _ in range(end_frame - start_frame)]
    point_prompts = [prompt for prompt in prompts if prompt.get("type") == "points"]
    keyword_prompts = [prompt for prompt in prompts if prompt.get("type") == "keyword"]

    _sam3_close_click_session()
    if point_prompts:
        session_id = _sam3_start_session(video_state, start_frame, end_frame, cache_frame_outputs=False)
        try:
            from preprocessing.sam3.preprocessor import encode_sam3_keyword_prompts

            point_preencoded = _sam3_bf16_prompt_payload(encode_sam3_keyword_prompts(["<text placeholder>"], keep_text_encoder_loaded=True)["<text placeholder>"])
            has_point_prompt = False
            for obj_id, prompt in enumerate(point_prompts, start=1):
                prompt_frame = video_state["origin_images"][prompt["frame_idx"]]
                prompt_height, prompt_width = prompt_frame.shape[:2]
                local_frame_idx = prompt["frame_idx"] - start_frame
                if local_frame_idx < 0 or local_frame_idx >= len(frames):
                    continue
                has_point_prompt = True
                with _sam3_autocast_context():
                    sam3_predictor.handle_request({
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": local_frame_idx,
                        "points": _sam3_points_payload(prompt.get("relative_points", _sam3_relative_points(prompt["points"], prompt_width, prompt_height))),
                        "point_labels": _sam3_labels_payload(prompt["labels"]),
                        "obj_id": obj_id,
                        "rel_coordinates": True,
                        "clear_old_points": True,
                        "preencoded_text_outputs": point_preencoded,
                    })
            if has_point_prompt:
                with _sam3_autocast_context():
                    for result in sam3_predictor.handle_stream_request({
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": "forward",
                        "start_frame_index": 0,
                        "max_frame_num_to_track": end_frame - start_frame,
                    }):
                        frame_idx = result["frame_index"]
                        if 0 <= frame_idx < len(frames):
                            alpha[frame_idx][:, :, 0] |= _sam3_outputs_to_mask(result["outputs"], height, width, fill_hole_area=SAM3_MATANYONE_FILL_HOLE_AREA) * 255
        finally:
            _sam3_close_session(session_id)

    for index, mask in enumerate(_sam3_propagate_keywords(video_state, keyword_prompts, start_frame, end_frame)):
        alpha[index] |= mask
    return alpha


@mask_action
def get_frames_from_image(state, image_input, image_state, new_dim):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """

    if image_input is None:
       gr.Info("Please select an Image file")
       return [gr.update()] * 20


    if len(new_dim)  > 0:
        image_input = perform_spatial_upsampling([image_input], new_dim)[0]

    user_name = time.time()
    frames = [image_input] * 2  # hardcode: mimic a video with 2 frames
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    image_state = {
        "user_name": user_name,
        "image_name": "output.png",
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "last_frame_numer": 0,
        "fps": None,
        "new_dim": new_dim,
        }
        
    image_info = "Image Name: N/A,\nFPS: N/A,\nTotal Frames: {},\nImage Size:{}".format(len(frames), image_size)
    acquire_GPU(state)
    if is_sam3_selected():
        _sam3_close_click_session()
        _ensure_sam3_predictor()
    else:
        set_image_encoder_patch()
        select_SAM(state)
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(image_state["origin_images"][0])
    torch.cuda.empty_cache()
    release_GPU(state)

    return image_state, gr.update(interactive=False), image_info, image_state["origin_images"][0], \
                        gr.update(visible=True, maximum=10, value=10), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(), \
                        gr.update(visible=False), gr.update(value="", visible=False),  gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)


# extract frames from upload video
@mask_action
def get_frames_from_video(state, video_input, video_state, new_dim):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    if video_input is None:
       gr.Info("Please select a Video file")
       return [gr.update()] * 19
     
        
    video_path = video_input
    frames = []
    user_name = time.time()

    # extract Audio
    # try:
    #     audio_path = video_input.replace(".mp4", "_audio.wav")
    #     ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ac=2, ar='44100').run(overwrite_output=True, quiet=True)
    # except Exception as e:
    #     print(f"Audio extraction error: {str(e)}")
    #     audio_path = ""  # Set to "" if extraction fails
    # print(f'audio_path: {audio_path}')
    audio_path = ""     
    # extract frames
    expected_frame_count = 0
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    finally:
        if cap is not None:
            cap.release()
    if expected_frame_count > 0 and len(frames) < expected_frame_count:
        print(f"MatAnyone: only loaded {len(frames)} of {expected_frame_count} frames from {video_path}.")
    image_size = (frames[0].shape[0],frames[0].shape[1]) 

    if len(new_dim) > 0:
        frames = perform_spatial_upsampling(frames, new_dim)
        image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # resize if resolution too big
    if image_size[0] >= 1280 and image_size[1] >= 1280:
        scale = 1080 / min(image_size)
        new_w = int(image_size[1] * scale)
        new_h = int(image_size[0] * scale)
        # update frames
        frames = [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
        # update image_size
        image_size = (frames[0].shape[0],frames[0].shape[1]) 

    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "last_frame_number": 0,
        "fps": fps,
        "audio": audio_path,
        "new_dim": new_dim,
        }
    video_info = "Video Name: {},\nFPS: {},\nTotal Frames: {},\nImage Size:{}".format(video_state["video_name"], round(video_state["fps"], 0), len(frames), image_size)
    acquire_GPU(state)
    if is_sam3_selected():
        _sam3_close_click_session()
        _ensure_sam3_predictor()
    else:
        set_image_encoder_patch()
        select_SAM(state)
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    torch.cuda.empty_cache()    
    release_GPU(state)
    return video_state, gr.update(interactive=False), video_info, video_state["origin_images"][0], \
                        gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), gr.update(visible=False, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=False), \
                        gr.update(visible=False), gr.update(visible=True), \
                        gr.update(visible=True)

# get the select frame from gradio slider
def select_video_template(image_selection_slider,  video_state, interactive_state):

    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    if not is_sam3_selected():
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])
    else:
        _sam3_close_click_session()

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

def select_image_template(image_selection_slider, video_state, interactive_state):

    image_selection_slider = 0 # fixed for image
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam
    if not is_sam3_selected():
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])
    else:
        _sam3_close_click_session()

    return video_state["painted_images"][image_selection_slider], video_state, interactive_state

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider

    return video_state["painted_images"][track_pause_number_slider],interactive_state


# use sam to get the mask
@mask_action
def sam_refine(state, video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData ): #
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1

    prompt = get_prompt(click_state=click_state, click_input=coordinate)
    if is_sam3_selected():
        frame_idx = video_state["select_frame_number"]
        image = video_state["origin_images"][frame_idx]
        acquire_GPU(state)
        try:
            mask = _sam3_preview_point_mask(video_state, frame_idx, prompt["input_point"], prompt["input_label"])
        finally:
            release_GPU(state)
        painted_image = _paint_sam3_mask(image, mask, prompt["input_point"], prompt["input_label"])
        video_state["masks"][frame_idx] = mask
        video_state["logits"][frame_idx] = None
        video_state["painted_images"][frame_idx] = painted_image
        interactive_state["sam3_current_prompt"] = {
            "type": "points",
            "frame_idx": frame_idx,
            "points": copy.deepcopy(prompt["input_point"]),
            "relative_points": _sam3_relative_points(prompt["input_point"], image.shape[1], image.shape[0]),
            "labels": copy.deepcopy(prompt["input_label"]),
        }
        return painted_image, video_state, interactive_state

    acquire_GPU(state)
    select_SAM(state)
    # prompt for sam model
    set_image_encoder_patch()
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    torch.cuda.empty_cache()
    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    torch.cuda.empty_cache()
    release_GPU(state)
    return painted_image, video_state, interactive_state

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    masks = video_state["masks"]
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload a Video")
        return [gr.update()]*4
    if is_sam3_selected() and interactive_state.get("sam3_current_prompt") is None:
        gr.Info("Please click the reference frame or add keywords before adding a SAM3 mask.")
        return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), video_state["origin_images"][video_state["select_frame_number"]], [[],[]]
    mask = masks[video_state["select_frame_number"]]
    interactive_state["multi_mask"]["masks"].append(mask)
    interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    if is_sam3_selected():
        interactive_state["multi_mask"].setdefault("sam3_prompts", []).append(copy.deepcopy(interactive_state["sam3_current_prompt"]))
        interactive_state["sam3_current_prompt"] = None
        _sam3_close_click_session()
    mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    select_frame = show_mask(video_state, interactive_state, mask_dropdown)

    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]]

def clear_click(video_state, click_state):
    masks = video_state["masks"]
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload a Video")
        return [gr.update()]*2

    if is_sam3_selected():
        _sam3_close_click_session()
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    return template_frame, click_state

def remove_multi_mask(interactive_state, mask_dropdown):
    if is_sam3_selected():
        _sam3_close_click_session()
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []
    interactive_state["multi_mask"]["sam3_prompts"] = []
    interactive_state["sam3_current_prompt"] = None

    return interactive_state, gr.update(choices=[],value=[])


@mask_action
def add_sam3_keyword_masks(state, video_state, interactive_state, keyword_text, mask_dropdown):
    if video_state["masks"] is None:
        gr.Info("SAM3 session lost. Please reload the media")
        return [gr.update()] * 3
    if not is_sam3_selected():
        return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), show_mask(video_state, interactive_state, mask_dropdown)

    keywords = _parse_sam3_keywords(keyword_text)
    if len(keywords) == 0:
        gr.Info("Please enter at least one keyword.")
        return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), show_mask(video_state, interactive_state, mask_dropdown)

    frame_idx = video_state["select_frame_number"]
    acquire_GPU(state)
    try:
        for keyword in keywords:
            mask = _sam3_preview_keyword_mask(video_state, frame_idx, keyword)
            interactive_state["multi_mask"]["masks"].append(mask)
            interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
            interactive_state["multi_mask"].setdefault("sam3_prompts", []).append({"type": "keyword", "frame_idx": frame_idx, "keyword": keyword})
            mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
    finally:
        release_GPU(state)

    select_frame = show_mask(video_state, interactive_state, mask_dropdown)
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame


def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    if video_state["origin_images"]:
        select_frame = video_state["origin_images"][video_state["select_frame_number"]]
        for i in range(len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1
            mask = interactive_state["multi_mask"]["masks"][mask_number]
            if np.any(mask):
                select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
        
        return select_frame


# def save_video(frames, output_path, fps):

#     writer = imageio.get_writer( output_path, fps=fps, codec='libx264', quality=8)
#     for frame in frames:
#         writer.append_data(frame)
#     writer.close()

#     return output_path

def mask_to_xyxy_box(mask):
    rows, cols = np.where(mask == 255)
    if len(rows) == 0 or len(cols) == 0: return []
    xmin = min(cols)
    xmax = max(cols) + 1
    ymin = min(rows)
    ymax = max(rows) + 1
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, mask.shape[1])
    ymax = min(ymax, mask.shape[0])
    box = [xmin, ymin, xmax, ymax]
    box = [int(x) for x in box]
    return box

def get_dim_file_suffix(new_dim):
    if not " " in new_dim: return ""
    pos = new_dim.find(" ")
    return new_dim[:pos]

# image matting
@mask_action
def image_matting(state, video_state, interactive_state, mask_type, matting_type, new_new_dim, mask_dropdown, erode_kernel_size, dilate_kernel_size, refine_iter, background_color="Green"):
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload an Image")
        return [gr.update(visible=False)]*12
    if is_sam3_selected() and mask_type == "alpha":
        mask_type = "wangp"

    new_dim = video_state.get("new_dim", "")
    if new_new_dim != new_dim:
        gr.Info(f"You have changed the Input / Output Dimensions after loading the Video into Matanyone. The output dimension will be the ones when loading the image ({'original' if len(new_dim) == 0 else new_dim})")

    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]

    if is_sam3_selected():
        alpha = [((template_mask > 0).astype(np.uint8) * 255)[:, :, None] for _ in following_frames]
    else:
        # operation error
        if len(np.unique(template_mask))==1:
            template_mask[0][0]=1
        acquire_GPU(state)
        select_matanyone(state)
        matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
        foreground, alpha = matanyone(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size, n_warmup=refine_iter)
        torch.cuda.empty_cache()
        release_GPU(state)

    foreground_mat = matting_type == "Foreground"

    foreground_output = None
    foreground_title = "Image with Background"
    alpha_title = "B & W Mask Image Output" if is_sam3_selected() else "Alpha Mask Image Output"

    if mask_type == "wangp":
        white_image = np.full_like(following_frames[-1], 255, dtype=np.uint8)
        alpha_output = alpha[-1] if foreground_mat else 255 - alpha[-1] 
        output_frame = (white_image.astype(np.uint16) * (255 - alpha_output.astype(np.uint16)) + 
                        following_frames[-1].astype(np.uint16) * alpha_output.astype(np.uint16))
        output_frame = output_frame // 255
        output_frame = output_frame.astype(np.uint8)
        foreground_output = output_frame
        control_output = following_frames[-1]
        alpha_output = alpha_output[:,:,0]     

        foreground_title = "Image without Background" if foreground_mat else "Image with Background"
        control_title = "Control Image"
        allow_export = True
        control_output = following_frames[-1]
        tab_label = "Control Image & Mask"
    elif mask_type == "greenscreen":
        alpha_output = alpha[-1] if foreground_mat else 255 - alpha[-1] 
        control_output = _matte_frame(following_frames[-1], alpha_output, background_color)
        alpha_output = alpha_output[:,:,0]                
        control_title = "Matted Image"
        tab_label = "Matted Image"
        allow_export = False
    elif mask_type == "alpha":
        alpha_output = alpha[-1] if foreground_mat else 255 - alpha[-1] 
        from models.wan.alpha.utils import render_video, from_BRGA_numpy_to_RGBA_torch
        from shared.utils.utils import convert_tensor_to_image
        _, BGRA_frames =  render_video(following_frames[-1:], [alpha_output])
        RGBA_image = from_BRGA_numpy_to_RGBA_torch(BGRA_frames).squeeze(1)
        control_output = convert_tensor_to_image(RGBA_image)
        alpha_output = alpha_output[:,:,0]                
        control_title = "RGBA Output"
        tab_label = "RGBA"
        allow_export = False


    bbox_info = mask_to_xyxy_box(alpha_output)
    h = alpha_output.shape[0]
    w = alpha_output.shape[1]
    if len(bbox_info) == 0:
        bbox_info = ""
    else:
        bbox_info = [str(int(bbox_info[0]/ w * 100 )), str(int(bbox_info[1]/ h * 100 )),  str(int(bbox_info[2]/ w * 100 )), str(int(bbox_info[3]/ h * 100 )) ]
        bbox_info = ":".join(bbox_info)
    alpha_output = Image.fromarray(alpha_output)
 
    return gr.update(visible=True, selected =0), gr.update(label=tab_label, visible=True), gr.update(visible = foreground_output is not None), foreground_output, control_output, alpha_output, gr.update(visible=foreground_output is not None, label=foreground_title),gr.update(visible=True, label=control_title), gr.update(visible=True, label=alpha_title), gr.update(value=bbox_info, visible= True), gr.update(visible=allow_export), gr.update(visible=allow_export)


# video matting
@mask_action
def video_matting(state, video_state, mask_type, video_input, end_slider, matting_type, new_new_dim, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, background_color="Green"):
    if video_state["masks"] is None:
        gr.Info("Matanyone Session Lost. Please reload a Video")
        return [gr.update(visible=False)]*6
    if is_sam3_selected() and mask_type == "alpha":
        mask_type = "wangp"

    # if interactive_state["track_end_number"]:
    #     following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    # else:
    start_frame = video_state["select_frame_number"]
    end_slider = max(start_frame + 1, end_slider)
    following_frames = video_state["origin_images"][start_frame: end_slider]

    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]]= template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    fps = video_state["fps"]
    new_dim = video_state.get("new_dim", "")
    if new_new_dim != new_dim:
        gr.Info(f"You have changed the Input / Output Dimensions after loading the Video into Matanyone. The output dimension will be the ones when loading the video ({'original' if len(new_dim) == 0 else new_dim})")

    if is_sam3_selected():
        prompts = _selected_sam3_prompts(interactive_state, mask_dropdown)
        if len(prompts) == 0:
            gr.Info("Please add at least one SAM3 mask before generating video matting.")
            return [gr.update(visible=False)]*6
        acquire_GPU(state)
        try:
            alpha = _sam3_propagate_prompts(video_state, prompts, video_state["select_frame_number"], end_slider)
        finally:
            release_GPU(state)
    else:
        # operation error
        if len(np.unique(template_mask))==1:
            template_mask[0][0]=1
        acquire_GPU(state)
        select_matanyone(state)
        matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
        foreground, alpha = matanyone(matanyone_processor, following_frames, template_mask*255, r_erode=erode_kernel_size, r_dilate=dilate_kernel_size)
        torch.cuda.empty_cache()
        release_GPU(state)
    foreground_mat = matting_type == "Foreground"
    alpha_title = "B & W Mask Video Output" if is_sam3_selected() else "Alpha Mask Video Output"
    alpha_suffix = "_bw_mask" if is_sam3_selected() else "_alpha"
    output_frames = []
    new_alpha = []
    BGRA_frames = None
    if mask_type == "" or mask_type == "wangp":
        if not foreground_mat:
            alpha = [255 - frame_alpha for frame_alpha in alpha ]
        output_frames = following_frames
        foreground_title = "Original Video Input"
        foreground_suffix = ""
        allow_export = True
    elif mask_type == "greenscreen":
        for frame_origin, frame_alpha in zip(following_frames, alpha):
            if not foreground_mat:
                frame_alpha = 255 - frame_alpha 
            output_frames.append(_matte_frame(frame_origin, frame_alpha, background_color))
            new_alpha.append(frame_alpha)
        alpha = new_alpha 
        foreground_title = "Matted Video"
        foreground_suffix = f"_matted_{background_color.lower()}"
        allow_export = False
    elif mask_type == "alpha":
        if not foreground_mat:
            alpha = [255 - frame_alpha for frame_alpha in alpha ]
        from models.wan.alpha.utils import render_video
        output_frames, BGRA_frames =  render_video(following_frames, alpha)
        foreground_title = "Checkboard Output"
        foreground_suffix = "_RGBA"
        allow_export = False

    if not os.path.exists("mask_outputs"):
        os.makedirs("mask_outputs")

    file_name= video_state["video_name"]
    file_name = ".".join(file_name.split(".")[:-1]) 
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
    file_name = f"{file_name}_{time_flag}"
    if len(new_dim) > 0: file_name += "_" + get_dim_file_suffix(new_dim) 
 
    from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks
    output_fg_path =  f"./mask_outputs/{file_name}{foreground_suffix}.mp4"
    output_fg_temp_path =  f"./mask_outputs/{file_name}{foreground_suffix}_tmp.mp4"
    if extract_audio_tracks(video_input, query_only=True) == 0:
        foreground_output = save_video(output_frames, output_fg_path , fps=fps, codec_type= video_output_codec)
    else:
        try:
            save_video(output_frames, output_fg_temp_path, fps=fps, codec_type=video_output_codec)
            # Copy an unchanged beginning; later starts need sample-accurate seeking and one audio encode.
            combine_video_with_audio_tracks(output_fg_temp_path, [video_input], output_fg_path, audio_codec_key="copy" if start_frame == 0 else "aac_128", audio_start_time=start_frame / fps)
        finally:
            if os.path.exists(output_fg_temp_path):
                os.remove(output_fg_temp_path)
        foreground_output = output_fg_path

    alpha_output = save_video(alpha, f"./mask_outputs/{file_name}{alpha_suffix}.mp4", fps=fps, codec_type= video_output_codec)
    if BGRA_frames is not None:
        from models.wan.alpha.utils import write_zip_file
        write_zip_file(f"./mask_outputs/{file_name}{foreground_suffix}.zip", BGRA_frames)
    return foreground_output, alpha_output, gr.update(visible=True, label=foreground_title), gr.update(visible=True, label=alpha_title), gr.update(visible=allow_export), gr.update(visible=allow_export)


def show_outputs():
    return gr.update(visible=True), gr.update(visible=True)

def add_audio_to_video(video_path, audio_path, output_path):
    pass
    # try:
    #     video_input = ffmpeg.input(video_path)
    #     audio_input = ffmpeg.input(audio_path)

    #     _ = (
    #         ffmpeg
    #         .output(video_input, audio_input, output_path, vcodec="copy", acodec="aac")
    #         .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
    #     )
    #     return output_path
    # except ffmpeg.Error as e:
    #     print(f"FFmpeg error:\n{e.stderr.decode()}")
    #     return None


def generate_video_from_frames(frames, output_path, fps=30, gray2rgb=False, audio_path=""):
    """
    Generates a video from a list of frames.

    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    frames = torch.from_numpy(np.asarray(frames))
    _, h, w, _ = frames.shape
    if gray2rgb:
        frames = np.repeat(frames, 3, axis=3)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    video_temp_path = output_path.replace(".mp4", "_temp.mp4")

    # resize back to ensure input resolution
    imageio.mimwrite(video_temp_path, frames, fps=fps, quality=7, 
                     codec='libx264', ffmpeg_params=["-vf", f"scale={w}:{h}"])

    # add audio to video if audio path exists
    if audio_path != "" and os.path.exists(audio_path):
        output_path = add_audio_to_video(video_temp_path, audio_path, output_path)    
        os.remove(video_temp_path)
        return output_path
    else:
        return video_temp_path

# reset all states for a new input

def get_default_states():
    return {
            "user_name": "",
            "video_name": "",
            "origin_images": None,
            "painted_images": None,
            "masks": None,
            "inpaint_masks": None,
            "logits": None,
            "select_frame_number": 0,
            "fps": 30
        }, {
            "inference_times": 0,
            "negative_click_times" : 0,
            "positive_click_times": 0,
            "mask_save": False,
            "multi_mask": {
                "mask_names": [],
                "masks": [],
                "sam3_prompts": [],
            },
            "sam3_current_prompt": None,
            "track_end_number": None,
        }, [[],[]]

def restart():
    return *(get_default_states()), gr.update(interactive=True), gr.update(visible=False), None,  None, None, \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),\
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False, choices=[], value=[]), "", gr.update(visible=False)

# def load_sam():
#     global model_loaded
#     global model
#     model.samcontroler.sam_controler.model.to(arg_device)

#     global matanyone_model 
#     matanyone_model.to(arg_device)


def select_matanyone(state):
    global matanyone_in_GPU, model_in_GPU 
    if is_sam3_selected():
        _ensure_sam3_predictor()
        return
    if matanyone_model is None or loaded_matanyone_version != get_selected_matanyone_version(server_config_ref):
        load_unload_models(state, True, True)
    if matanyone_in_GPU: return
    model.samcontroler.sam_controler.model.to("cpu")
    model_in_GPU = False
    torch.cuda.empty_cache()
    matanyone_model.to(arg_device)
    matanyone_in_GPU = True

def select_SAM(state):
    global matanyone_in_GPU, model_in_GPU 
    if is_sam3_selected():
        _ensure_sam3_predictor()
        return
    if matanyone_model is None or loaded_matanyone_version != get_selected_matanyone_version(server_config_ref):
        load_unload_models(state, True, True)
    if model_in_GPU: return
    matanyone_model.to("cpu")
    matanyone_in_GPU = False
    torch.cuda.empty_cache()
    model.samcontroler.sam_controler.model.to(arg_device)
    model_in_GPU = True

load_in_progress = False

def load_unload_models(state = None, selected = True, force = False):
    global model_loaded, load_in_progress
    global model
    global matanyone_model, matanyone_processor, matanyone_in_GPU , model_in_GPU, bfloat16_supported, loaded_matanyone_version, sam3_predictor, sam3_click_session

    if selected:
        selected_version = get_selected_matanyone_version(server_config_ref)
        if model_loaded and loaded_matanyone_version != selected_version:
            load_unload_models(state, False, True)

        if (not force) and any_GPU_process_running(state, "matanyone"):
            return

        if load_in_progress:
            while load_in_progress:
                time.sleep(1)
            return
        # print("Matanyone Tab Selected")
        if model_loaded or load_in_progress:
            return
        else:
            load_in_progress = True
            try:
                if selected_version == MATANYONE_SAM3:
                    ensure_selected_matanyone_assets(server_config_ref)
                    _ensure_sam3_predictor()
                    model_loaded = True
                    loaded_matanyone_version = selected_version
                    matanyone_in_GPU = model_in_GPU = False
                    return
                sam_checkpoint = None
                ensure_selected_matanyone_assets(server_config_ref)

                transfer_stream = torch.cuda.Stream()
                with torch.cuda.stream(transfer_stream):
                    # initialize sams
                    major, minor = torch.cuda.get_device_capability(arg_device)
                    if  major < 8:
                        bfloat16_supported = False
                    else:
                        bfloat16_supported = True

                    model = MaskGenerator(sam_checkpoint, "cpu")
                    model.samcontroler.sam_controler.model.to("cpu").to(torch.bfloat16).to(arg_device)
                    model_in_GPU = True
                    matanyone_model, loaded_matanyone_version, _ = load_selected_matanyone_model(server_config_ref)
                    # pipe ={"mat" : matanyone_model, "sam" :model.samcontroler.sam_controler.model }
                    # offload.profile(pipe)
                    matanyone_model = matanyone_model.to("cpu").eval()
                    matanyone_in_GPU = False
                    matanyone_processor = InferenceCore(matanyone_model, cfg=matanyone_model.cfg)
                model_loaded  = True
            except Exception:
                load_unload_models(state, False, True)
                raise
            finally:
                load_in_progress = False

    else:
        # print("Matanyone Tab UnSelected")
        import gc
        # model.samcontroler.sam_controler.model.to("cpu")
        # matanyone_model.to("cpu")
        model = matanyone_model = matanyone_processor = None
        _sam3_close_click_session()
        if sam3_predictor is not None:
            sam3_predictor.shutdown()
            sam3_predictor = None
        from preprocessing.sam3.preprocessor import clear_sam3_text_encoder_cache

        clear_sam3_text_encoder_cache()
        sam3_click_session = None
        loaded_matanyone_version = None
        matanyone_in_GPU = model_in_GPU = False
        gc.collect()
        torch.cuda.empty_cache()
        model_loaded = False


def get_mask_generator_event_handler():
    return load_unload_models


def ensure_selected_assets(server_config=None):
    return ensure_selected_matanyone_assets(server_config if server_config is not None else server_config_ref)


def get_title_markdown():
    return get_matanyone_title_html(server_config_ref)


def export_image(state, image_output):
    ui_settings = get_current_model_settings(state)
    image_refs = ui_settings.get("image_refs", None)
    if image_refs == None:
        image_refs =[]
    image_refs.append( image_output)
    ui_settings["image_refs"] = image_refs 
    gr.Info("Masked Image transferred to Current Image Generator")
    return time.time()

def export_image_mask(state, image_input, image_mask):
    ui_settings = get_current_model_settings(state)
    ui_settings["image_guide"] = image_input
    ui_settings["image_mask"] = image_mask

    gr.Info("Input Image & Mask transferred to Current Image Generator")
    return time.time()


def export_to_current_video_engine(state, foreground_video_output, alpha_video_output):
    ui_settings = get_current_model_settings(state)
    ui_settings["video_guide"] = foreground_video_output
    ui_settings["video_mask"] = alpha_video_output

    gr.Info("Original Video and Full Mask have been transferred")
    return time.time()


def teleport_to_media_tab(tab_state, state):
    return PlugIn.goto_media_tab(state)


def display(tabs, tab_state, state, refresh_form_trigger, server_config, get_current_model_settings_fn): #,  vace_video_input, vace_image_input, vace_video_mask, vace_image_mask, vace_image_refs):
    # my_tab.select(fn=load_unload_models, inputs=[], outputs=[])
    global image_output_codec, video_output_codec, get_current_model_settings, server_config_ref
    get_current_model_settings = get_current_model_settings_fn
    server_config_ref = server_config

    image_output_codec = server_config.get("image_output_codec", None)
    video_output_codec = server_config.get("video_output_codec", None)

    media_url = "https://github.com/pq-yang/MatAnyone/releases/download/media/"

    click_brush_js = """
    () => {
        setTimeout(() => {
            const brushButton = document.querySelector('button[aria-label="Brush"]');
            if (brushButton) {
                brushButton.click();
                console.log('Brush button clicked');
            } else {
                console.log('Brush button not found');
            }
        }, 1000);
    }    """

    # download assets

    gr.HTML('''<style>
        .matanyone-media-tab:has(.wangp-progress:not(.finished)),
        .matanyone-media-tab:has(.wangp-progress:not(.finished)) * { cursor: progress !important; }
    </style>''', visible=False)
    matanyone_title_md = gr.Markdown(get_title_markdown())
    refresh_form_trigger.change(fn=get_title_markdown, inputs=[], outputs=[matanyone_title_md], show_progress="hidden")
    gr.Markdown("If you have some trouble creating the perfect mask, be aware of these tips:")
    gr.Markdown("- Using the Matanyone Settings you can also define Negative Point Prompts to remove parts of the current selection.")
    gr.Markdown("- Sometime it is very hard to fit everything you want in a single mask, it may be much easier to combine multiple independent sub Masks before producing the Matting : each sub Mask is created by selecting an  area of an image and by clicking the Add Mask button. Sub masks can then be enabled / disabled in the Matanyone settings.")
    gr.Markdown("The Mask Generation time and the VRAM consumed are proportional to the number of frames and the resolution. So if relevant, you may reduce the number of frames in the Matanyone Settings. You will need for the moment to resize yourself the video if needed.")

    with gr.Column( visible=True):
        with gr.Row():
            # The tutorial players load only while the accordion is open: a loaded player keeps a video decoder in the browser's GPU memory
            with gr.Accordion("Video Tutorial (click to expand)", open=False, elem_classes="custom-bg") as tutorial_accordion:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Case 1: Single Target")
                        tutorial_single_video = gr.Video(elem_classes="video")

                    with gr.Column():
                        gr.Markdown("### Case 2: Multiple Targets")
                        tutorial_multi_video = gr.Video(elem_classes="video")
            tutorial_accordion.expand(fn=lambda: ("preprocessing/matanyone/tutorial_single_target.mp4", "preprocessing/matanyone/tutorial_multi_targets.mp4"), outputs=[tutorial_single_video, tutorial_multi_video], show_progress="hidden")
            tutorial_accordion.collapse(fn=lambda: (None, None), outputs=[tutorial_single_video, tutorial_multi_video], show_progress="hidden")

        with gr.Row():
            new_dim= gr.Dropdown(
                choices=[
                    ("Original Dimensions", ""),
                    ("1080p - Pixels Budgets", "1080p - Pixels Budget"),
                    ("720p - Pixels Budgets", "720p - Pixels Budget"),
                    ("480p - Pixels Budgets", "480p - Pixels Budget"),                     
                    ("1080p - Outer Frame", "1080p - Outer Frame"),
                    ("720p - Outer Frame", "720p - Outer Frame"),
                    ("480p - Outer Frame", "480p - Outer Frame"),                     
                ],   label = "Resize Input / Output", value = ""
            ) 

            mask_type= gr.Dropdown(
                choices=_matanyone_mask_type_choices(), label = "Mask Type", value = "wangp"
            ) 
            media_type = gr.State("Video")
            background_color = gr.Dropdown(choices=list(MATTE_COLORS), value="Green", label="Background Color", visible=False)
            mask_type.input(fn=_matanyone_background_color_update, inputs=[mask_type], outputs=[background_color], queue=False, show_progress="hidden")
            refresh_form_trigger.change(fn=_matanyone_mask_type_update, inputs=[mask_type, media_type], outputs=[mask_type, background_color], queue=False, show_progress="hidden")

            matting_type = gr.Radio(
                choices=["Foreground", "Background"],
                value="Foreground",
                label="Type of Video Matting to Generate",
                scale=1)

            
        with gr.Row(visible=False):
            dummy = gr.Text()        

        with gr.Tabs():
            with gr.TabItem("Video", elem_classes="matanyone-media-tab") as video_tab:
                video_tab.select(fn=lambda mask: ("Video", *_matanyone_mask_type_update(mask, "Video")), inputs=[mask_type], outputs=[media_type, mask_type, background_color], queue=False, show_progress="hidden")
                # The hidden progress still drives the cursor; leaving this tab restores the normal pointer.
                with gr.Column(visible=False):
                    mask_progress = WangpProgress.component(stop=True)

                click_state = gr.State([[],[]])

                interactive_state = gr.State({
                    "inference_times": 0,
                    "negative_click_times" : 0,
                    "positive_click_times": 0,
                    "mask_save": arg_mask_save,
                    "multi_mask": {
                        "mask_names": [],
                        "masks": [],
                        "sam3_prompts": [],
                    },
                    "sam3_current_prompt": None,
                    "track_end_number": None,
                    }
                )

                video_state = gr.State(
                    {
                    "user_name": "",
                    "video_name": "",
                    "origin_images": None,
                    "painted_images": None,
                    "masks": None,
                    "inpaint_masks": None,
                    "logits": None,
                    "select_frame_number": 0,
                    "fps": 16,
                    "audio": "",
                    }
                )

                with gr.Column( visible=True):
                    with gr.Row():
                        with gr.Accordion('MatAnyone Settings (click to expand)', open=False):
                            with gr.Row(visible=not is_sam3_selected()) as video_morphology_row:
                                erode_kernel_size = gr.Slider(label='Erode Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Erosion on the added mask",
                                                        interactive=True)
                                dilate_kernel_size = gr.Slider(label='Dilate Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Dilation on the added mask",
                                                        interactive=True)

                            with gr.Row():
                                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Start Frame", info="Choose the start frame for target assignment and video matting", visible=False)
                                end_selection_slider = gr.Slider(minimum=1, maximum=300, step=1, value=81, label="Last Frame to Process", info="Last Frame to Process", visible=False)

                                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="End frame", visible=False)
                            with gr.Row():
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    info="Click to add positive or negative point for target mask",
                                    interactive=True,
                                    visible=False,
                                    min_width=100,
                                    scale=1)
                                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False, scale=2, allow_custom_value=True)
                    # input video
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2): 
                            gr.Markdown("## Step1: Upload video")
                        with gr.Column(scale=2): 
                            step2_title = gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>", visible=False)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):      
                            video_input = gr.Video(label="Input Video", elem_classes="video")
                            extract_frames_button = gr.Button(value="Load Video", interactive=True, elem_classes="new_button")
                        with gr.Column(scale=2):
                            video_info = gr.Textbox(label="Video Info", visible=False)
                            template_frame = gr.Image(label="Start Frame", type="pil",interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                            with gr.Row():
                                clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False,  min_width=100)
                                add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, min_width=100)
                                remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False,  min_width=100) # no use
                                matting_button = gr.Button(value="Generate Video Matting", interactive=True, visible=False,  min_width=100)
                            with gr.Row(visible=False, equal_height=True, elem_classes="sam3-keyword-row") as sam3_keyword_row:
                                sam3_keyword_text = gr.Textbox(label="Keyword", show_label=False, placeholder="keyword", lines=1, min_width=120, scale=1)
                                sam3_keyword_button = gr.Button(value="Add Mask based on Keyword", interactive=True, min_width=260, scale=1, elem_classes="sam3-keyword-button")
                            with gr.Row():
                                gr.Markdown("")            

                    # output video
                    with gr.Column() as output_row: #equal_height=True
                        with gr.Row():
                            with gr.Column(scale=2):
                                foreground_video_output = gr.Video(label="Original Video Input", visible=False, interactive=False, elem_classes="video")
                                foreground_output_button = gr.Button(value="Black & White Video Output", visible=False, elem_classes="new_button")
                            with gr.Column(scale=2):
                                alpha_video_output = gr.Video(label="Mask Video Output", visible=False, interactive=False, elem_classes="video")
                                export_image_mask_btn = gr.Button(value=_matanyone_mask_output_button_label(), visible=False, elem_classes="new_button")
                        with gr.Row():
                            with gr.Row(visible= False):
                                export_to_vace_video_14B_btn = gr.Button("Export to current Video Input Video For Inpainting", visible= False)
                            with gr.Row(visible= True):
                                export_to_current_video_engine_btn = gr.Button("Export to Control Video Input and Video Mask Input", visible= False)
                                    
                export_to_current_video_engine_btn.click(  fn=export_to_current_video_engine, inputs= [state, foreground_video_output, alpha_video_output], outputs= [refresh_form_trigger]).then( #video_prompt_video_guide_trigger, 
                    fn=teleport_to_media_tab, inputs= [tab_state, state], outputs= [tabs])
                refresh_form_trigger.change(fn=_matanyone_mask_output_button_update, inputs=[], outputs=[export_image_mask_btn], show_progress="hidden")


                # first step: get the video information     
                WangpProgress.bind(extract_frames_button.click,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=get_frames_from_video,
                    inputs=[
                        state, video_input, video_state, new_dim
                    ],
                    outputs=[video_state, extract_frames_button, video_info, template_frame,
                            image_selection_slider, end_selection_slider,  track_pause_number_slider, point_prompt, dummy, clear_button_click, add_mask_button, matting_button, template_frame,
                            foreground_video_output, alpha_video_output, foreground_output_button, export_image_mask_btn, mask_dropdown, step2_title]
                ).then(fn=lambda: gr.update(visible=is_sam3_selected()), inputs=[], outputs=[sam3_keyword_row], show_progress="hidden")

                # second step: select images from slider
                image_selection_slider.release(fn=select_video_template, 
                                            inputs=[image_selection_slider, video_state, interactive_state], 
                                            outputs=[template_frame, video_state, interactive_state], api_name="select_image")
                track_pause_number_slider.release(fn=get_end_number, 
                                            inputs=[track_pause_number_slider, video_state, interactive_state], 
                                            outputs=[template_frame, interactive_state], api_name="end_image")
                
                # click select image to get mask using sam
                WangpProgress.bind(template_frame.select,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=sam_refine,
                    inputs=[state, video_state, point_prompt, click_state, interactive_state],
                    outputs=[template_frame, video_state, interactive_state]
                )

                # add different mask
                add_mask_button.click(
                    fn=add_multi_mask,
                    inputs=[video_state, interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown, template_frame, click_state]
                )

                remove_mask_button.click(
                    fn=remove_multi_mask,
                    inputs=[interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown]
                )

                # video matting
                WangpProgress.bind(matting_button.click(
                    fn=show_outputs,
                    inputs=[],
                    outputs=[foreground_video_output, alpha_video_output]).then,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=video_matting,
                    inputs=[state, video_state, mask_type, video_input, end_selection_slider, matting_type, new_dim, interactive_state, mask_dropdown, erode_kernel_size, dilate_kernel_size, background_color],
                    outputs=[foreground_video_output, alpha_video_output,foreground_video_output, alpha_video_output, export_to_vace_video_14B_btn, export_to_current_video_engine_btn]
                )

                # click to get mask
                mask_dropdown.change(
                    fn=show_mask,
                    inputs=[video_state, interactive_state, mask_dropdown],
                    outputs=[template_frame]
                )

                refresh_form_trigger.change(fn=lambda video_state: gr.update(visible=is_sam3_selected() and video_state.get("origin_images") is not None), inputs=[video_state], outputs=[sam3_keyword_row], show_progress="hidden")

                WangpProgress.bind(sam3_keyword_button.click,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=add_sam3_keyword_masks,
                    inputs=[state, video_state, interactive_state, sam3_keyword_text, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown, template_frame]
                )
                
                # clear input
                video_input.change(
                    fn=restart,
                    inputs=[],
                    outputs=[ 
                        video_state,
                        interactive_state,
                        click_state, 
                        extract_frames_button, dummy,
                        foreground_video_output, dummy, alpha_video_output,
                        template_frame,
                        image_selection_slider, end_selection_slider, track_pause_number_slider,point_prompt, export_to_vace_video_14B_btn, export_to_current_video_engine_btn, dummy, clear_button_click, 
                        add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, export_image_mask_btn, mask_dropdown, video_info, step2_title
                    ],
                    queue=False,
                    show_progress=False).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[sam3_keyword_row], show_progress="hidden")
                
                video_input.clear(
                    fn=restart,
                    inputs=[],
                    outputs=[ 
                        video_state,
                        interactive_state,
                        click_state,
                        extract_frames_button, dummy,
                        foreground_video_output, dummy, alpha_video_output,
                        template_frame,
                        image_selection_slider , end_selection_slider, track_pause_number_slider,point_prompt, export_to_vace_video_14B_btn, export_to_current_video_engine_btn, dummy, clear_button_click, 
                        add_mask_button, matting_button, template_frame, foreground_video_output, alpha_video_output, remove_mask_button, foreground_output_button, export_image_mask_btn, mask_dropdown, video_info, step2_title
                    ],
                    queue=False,
                    show_progress=False).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[sam3_keyword_row], show_progress="hidden")
                
                # points clear
                clear_button_click.click(
                    fn = clear_click,
                    inputs = [video_state, click_state,],
                    outputs = [template_frame,click_state],
                )



            with gr.TabItem("Image", elem_classes="matanyone-media-tab") as image_tab:
                image_tab.select(fn=lambda mask: ("Image", *_matanyone_mask_type_update(mask, "Image")), inputs=[mask_type], outputs=[media_type, mask_type, background_color], queue=False, show_progress="hidden")
                with gr.Column(visible=False):
                    mask_progress = WangpProgress.component(stop=True)
                click_state = gr.State([[],[]])

                interactive_state = gr.State({
                    "inference_times": 0,
                    "negative_click_times" : 0,
                    "positive_click_times": 0,
                    "mask_save": False,
                    "multi_mask": {
                        "mask_names": [],
                        "masks": [],
                        "sam3_prompts": [],
                    },
                    "sam3_current_prompt": None,
                    "track_end_number": None,
                    }
                )

                image_state = gr.State(
                    {
                    "user_name": "",
                    "image_name": "",
                    "origin_images": None,
                    "painted_images": None,
                    "masks": None,
                    "inpaint_masks": None,
                    "logits": None,
                    "select_frame_number": 0,
                    "fps": 30
                    }
                )

                with gr.Group(elem_classes="gr-monochrome-group", visible=True):
                    with gr.Row():
                        with gr.Accordion('MatAnyone Settings (click to expand)', open=False):
                            with gr.Row(visible=not is_sam3_selected()) as image_morphology_row:
                                erode_kernel_size = gr.Slider(label='Erode Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Erosion on the added mask",
                                                        interactive=True)
                                dilate_kernel_size = gr.Slider(label='Dilate Kernel Size',
                                                        minimum=0,
                                                        maximum=30,
                                                        step=1,
                                                        value=10,
                                                        info="Dilation on the added mask",
                                                        interactive=True)
                                
                            with gr.Row():
                                image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Num of Refinement Iterations", info="More iterations → More details & More time", visible=False)
                                track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
                            with gr.Row():
                                point_prompt = gr.Radio(
                                    choices=["Positive", "Negative"],
                                    value="Positive",
                                    label="Point Prompt",
                                    info="Click to add positive or negative point for target mask",
                                    interactive=True,
                                    visible=False,
                                    min_width=100,
                                    scale=1)
                                mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask Selection", info="Choose 1~all mask(s) added in Step 2", visible=False)
                

                with gr.Column():
                    # input image
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2): 
                            gr.Markdown("## Step1: Upload image")
                        with gr.Column(scale=2): 
                            step2_title = gr.Markdown("## Step2: Add masks <small>(Several clicks then **`Add Mask`** <u>one by one</u>)</small>", visible=False)
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=2):      
                            image_input = gr.Image(label="Input Image", elem_classes="image")
                            extract_frames_button = gr.Button(value="Load Image", interactive=True, elem_classes="new_button")
                        with gr.Column(scale=2):
                            image_info = gr.Textbox(label="Image Info", visible=False)
                            template_frame = gr.Image(type="pil", label="Start Frame", interactive=True, elem_id="template_frame", visible=False, elem_classes="image")
                            with gr.Row(equal_height=True, elem_classes="mask_button_group"):
                                clear_button_click = gr.Button(value="Clear Clicks", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                                add_mask_button = gr.Button(value="Add Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                                remove_mask_button = gr.Button(value="Remove Mask", interactive=True, visible=False, elem_classes="new_button", min_width=100)
                                matting_button = gr.Button(value="Image Matting", interactive=True, visible=False, elem_classes="green_button", min_width=100)
                            with gr.Row(visible=False, equal_height=True, elem_classes="sam3-keyword-row") as image_sam3_keyword_row:
                                image_sam3_keyword_text = gr.Textbox(label="Keyword", show_label=False, placeholder="keyword", lines=1, min_width=120, scale=1)
                                image_sam3_keyword_button = gr.Button(value="Add Mask based on Keyword", interactive=True, min_width=260, scale=1, elem_classes="sam3-keyword-button")

                    # output image
                    with gr.Tabs(visible = False) as image_tabs:
                        with gr.TabItem("Control Image & Mask", visible = False) as image_first_tab:
                            with gr.Row(equal_height=True):
                                control_image_output = gr.Image(type="pil", label="Control Image", visible=False, interactive=False, elem_classes="image")
                                alpha_image_output = gr.Image(type="pil", label="Mask", visible=False, interactive=False, elem_classes="image")
                            with gr.Row():
                                export_image_mask_btn = gr.Button(value="Set to Control Image & Mask", visible=False, elem_classes="new_button")
                        with gr.TabItem("Reference Image", visible = False) as image_second_tab:
                            with gr.Row():
                                foreground_image_output = gr.Image(type="pil", label="Foreground Output", visible=False, interactive=False, elem_classes="image")
                            with gr.Row():
                                export_image_btn = gr.Button(value="Add to current Reference Images", visible=False, elem_classes="new_button")

                    with gr.Row(equal_height=True):
                        bbox_info = gr.Text(label ="Mask BBox Info (Left:Top:Right:Bottom)", visible = False, interactive= False)

                export_image_btn.click(  fn=export_image, inputs= [state, foreground_image_output], outputs= [refresh_form_trigger]).then( #video_prompt_video_guide_trigger, 
                    fn=teleport_to_media_tab, inputs= [tab_state, state], outputs= [tabs])
                export_image_mask_btn.click(  fn=export_image_mask, inputs= [state, control_image_output, alpha_image_output], outputs= [refresh_form_trigger]).then( #video_prompt_video_guide_trigger, 
                    fn=teleport_to_media_tab, inputs= [tab_state, state], outputs= [tabs]).then(fn=None, inputs=None, outputs=None, js=click_brush_js)

                # first step: get the image information 
                WangpProgress.bind(extract_frames_button.click,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=get_frames_from_image,
                    inputs=[
                        state, image_input, image_state, new_dim
                    ],
                    outputs=[image_state, extract_frames_button, image_info, template_frame,
                            image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, add_mask_button, matting_button, template_frame,
                            foreground_image_output, alpha_image_output, control_image_output, image_tabs, bbox_info, export_image_btn, export_image_mask_btn, mask_dropdown, step2_title]
                ).then(fn=lambda: gr.update(visible=is_sam3_selected()), inputs=[], outputs=[image_sam3_keyword_row], show_progress="hidden")

                # points clear
                clear_button_click.click(
                    fn = clear_click,
                    inputs = [image_state, click_state,],
                    outputs = [template_frame,click_state],
                )


                # second step: select images from slider
                image_selection_slider.release(fn=select_image_template, 
                                            inputs=[image_selection_slider, image_state, interactive_state], 
                                            outputs=[template_frame, image_state, interactive_state], api_name="select_image")
                track_pause_number_slider.release(fn=get_end_number, 
                                            inputs=[track_pause_number_slider, image_state, interactive_state], 
                                            outputs=[template_frame, interactive_state], api_name="end_image")
                
                # click select image to get mask using sam
                WangpProgress.bind(template_frame.select,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=sam_refine,
                    inputs=[state, image_state, point_prompt, click_state, interactive_state],
                    outputs=[template_frame, image_state, interactive_state]
                )

                # add different mask
                add_mask_button.click(
                    fn=add_multi_mask,
                    inputs=[image_state, interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown, template_frame, click_state]
                )

                remove_mask_button.click(
                    fn=remove_multi_mask,
                    inputs=[interactive_state, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown]
                )

                WangpProgress.bind(image_sam3_keyword_button.click,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=add_sam3_keyword_masks,
                    inputs=[state, image_state, interactive_state, image_sam3_keyword_text, mask_dropdown],
                    outputs=[interactive_state, mask_dropdown, template_frame]
                )

                # image matting
                WangpProgress.bind(matting_button.click,
                    component=mask_progress, concurrency_id="matanyone",
                    fn=image_matting,
                    inputs=[state, image_state, interactive_state, mask_type, matting_type, new_dim, mask_dropdown, erode_kernel_size, dilate_kernel_size, image_selection_slider, background_color],
                    outputs=[image_tabs, image_first_tab, image_second_tab, foreground_image_output, control_image_output, alpha_image_output, foreground_image_output, control_image_output, alpha_image_output, bbox_info, export_image_btn, export_image_mask_btn]
                )

                nada = gr.State({})
                # clear input
                gr.on(
                    triggers=[image_input.clear], #image_input.change,
                    fn=restart,
                    inputs=[],
                    outputs=[ 
                        image_state,
                        interactive_state,
                        click_state,
                        extract_frames_button, image_tabs,
                        foreground_image_output, control_image_output, alpha_image_output,
                        template_frame,
                        image_selection_slider, image_selection_slider, track_pause_number_slider,point_prompt, export_image_btn, export_image_mask_btn, bbox_info, clear_button_click, 
                        add_mask_button, matting_button, template_frame, foreground_image_output, alpha_image_output, remove_mask_button, export_image_btn, export_image_mask_btn, mask_dropdown, nada, step2_title
                    ],
                    queue=False,
                    show_progress=False).then(fn=lambda: gr.update(visible=False), inputs=[], outputs=[image_sam3_keyword_row], show_progress="hidden")

                refresh_form_trigger.change(
                    fn=lambda image_state: gr.update(visible=is_sam3_selected() and image_state.get("origin_images") is not None),
                    inputs=[image_state],
                    outputs=[image_sam3_keyword_row],
                    show_progress="hidden",
                )

                refresh_form_trigger.change(
                    fn=lambda: [_matanyone_morphology_visibility(), _matanyone_morphology_visibility()],
                    inputs=[],
                    outputs=[video_morphology_row, image_morphology_row],
                    show_progress="hidden",
                )
                
