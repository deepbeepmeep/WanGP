############# WanGP Copyright DeepBeepMeep 2025-2026 #############
import os, sys
os.environ["GRADIO_LANG"] = "en"
from shared.default_device import set_default_cuda_device_from_arg; set_default_cuda_device_from_arg("gpu")
# # os.environ.pop("TORCH_LOGS", None)  # make sure no env var is suppressing/overriding
# os.environ["TORCH_LOGS"]= "recompiles"
import torch._logging as tlog
# tlog.set_logs(recompiles=True, guards=True, graph_breaks=True)    
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)
# from shared.utils.crash_diagnostics import install_wgp_crash_diagnostics; install_wgp_crash_diagnostics(__file__)
# Ensure plugin-side `import wgp` resolves to this live module instance.
if sys.modules.get("wgp") is not sys.modules.get(__name__):
    sys.modules["wgp"] = sys.modules[__name__]
import asyncio
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
if sys.platform.startswith("linux") and "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
from shared.asyncio_utils import silence_proactor_connection_reset
silence_proactor_connection_reset()
import time
import threading
import warnings
warnings.filterwarnings('ignore', message='Failed to find.*', module='triton')
from mmgp import offload, safetensors2, profile_type , quant_router
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from datetime import datetime
import gradio as gr
from shared.gradio import gradio_queue_focus_patch
from gradio.themes.utils.sizes import Size
import random
import json
import numpy as np
import importlib
from shared.utils import notification_sound
from shared.utils.loras_mutipliers import preparse_loras_multipliers, parse_loras_multipliers
from shared.utils.utils import convert_tensor_to_image, save_image, get_video_info, get_file_creation_date, convert_image_to_video, calculate_new_dimensions, convert_image_to_tensor, calculate_dimensions_and_resize_image, rescale_and_crop, get_video_frame, resize_and_remove_background, rgb_bw_to_rgba_mask, to_rgb_tensor, get_resampled_video_transparent, get_video_summary_extras
from shared.utils.utils import calculate_new_dimensions, get_outpainting_dims, get_outpainting_frame_location, get_outpainting_full_area_dimensions, resolve_outpainting_dims
from shared.utils.utils import has_video_file_extension, has_image_file_extension, has_audio_file_extension
from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks, combine_and_concatenate_video_with_audio_tracks, cleanup_temp_audio_files, normalize_audio_pair_volumes_to_temp_files, save_video, save_hdr_video, save_image
from shared.utils.audio_video import append_sliding_window_audio, read_image_metadata, extract_audio_track_to_wav, write_wav_file, save_audio_file, get_audio_codec_extension, create_silent_wav_file
from shared.utils.audio_metadata import read_audio_metadata, extract_creation_datetime_from_metadata, resolve_audio_creation_datetime
from shared.utils.media_recording import record_file_metadata as shared_record_file_metadata
from shared.utils.video_decode import decode_video_frames_ffmpeg, probe_video_stream_metadata
from shared.utils.virtual_media import get_virtual_image, get_virtual_media_entry, get_virtual_media_vsource, parse_virtual_media_path, replace_virtual_media_source, strip_virtual_media_suffix
from shared.match_archi import match_nvidia_architecture
from shared.attention import get_attention_modes, get_supported_attention_modes
from shared.utils.utils import truncate_for_filesystem, sanitize_file_name, process_images_multithread, get_default_workers
from shared.utils.process_locks import (
    acquire_GPU_ressources,
    acquire_main_GPU_ressources,
    any_GPU_process_running,
    gen_lock,
    register_GPU_resident,
    release_GPU_ressources,
    set_main_generation_running,
    unregister_GPU_resident,
)
from shared.deepy.config import get_deepy_default_runtime_config, set_deepy_runtime_config
from shared.loras_migration import migrate_loras_layout
from huggingface_hub import hf_hub_download, snapshot_download
from shared.utils import files_locator as fl 
from shared.gradio.audio_gallery import AudioGallery  
from shared.utils.self_refiner import normalize_self_refiner_plan, ensure_refiner_list, add_refiner_rule, remove_refiner_rule
from shared.deepy import controller as deepy_controller
from shared.deepy import cli as deepy_cli
from shared.deepy import gradio_ui as deepy_gradio_ui
from shared import extra_settings
import torch
import gc
import traceback
import math 
import typing
import inspect
from shared.utils import prompt_parser
import base64
import io
from PIL import Image
import zipfile
import tempfile
import atexit
import shutil
import glob
import cv2
import html
try:
    from gradio_rangeslider import RangeSlider
except ImportError:
    RangeSlider = None
import re
from transformers.utils import logging
logging.set_verbosity_error
from tqdm import tqdm
import requests
from shared.gradio.gallery import AdvancedMediaGallery, get_gradio_file_path
from shared.ffmpeg_setup import download_ffmpeg
from shared.api import get_api_output_options, store_api_output_artifact
from shared.utils.plugins import PluginManager, WAN2GPApplication, SYSTEM_PLUGINS
from shared.llm_engines.nanovllm.vllm_support import resolve_lm_decoder_engine
from shared.gradio import assistant_chat
from shared.gradio.magic_mask import MagicMaskUI
from shared import model_dropdowns
from shared.cli_args import parse_wgp_args
from collections import defaultdict

MagicMaskUI.patch_image_editor()

# import torch._dynamo as dynamo
# dynamo.config.recompile_limit = 2000   # default is 256
# dynamo.config.accumulated_recompile_limit = 2000  # or whatever limit you want

STARTUP_LOCK_FILE = "startup.lock"
global_queue_ref = []
AUTOSAVE_FILENAME = "queue.zip"
AUTOSAVE_PATH = AUTOSAVE_FILENAME
AUTOSAVE_ERROR_FILENAME = "error_queue.zip"
AUTOSAVE_TEMPLATE_PATH = AUTOSAVE_FILENAME
CONFIG_FILENAME = "wgp_config.json"
PROMPT_VARS_MAX = 10
target_mmgp_version = "3.7.6"
WanGP_version = "11.52"
settings_version = 2.58
max_source_video_frames = 3000
prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None
image_names_list = ["image_start", "image_end", "image_refs"]
CUSTOM_SETTINGS_MAX = 6
CUSTOM_SETTINGS_PER_ROW = 2
CUSTOM_SETTING_TYPES = {"int", "float", "text"}
lm_decoder_engine = ""
enable_int8_kernels = 0
theme_text_size = Size("8.1px", "9px", "10.8px", "12.6px", "14.4px", "19.8px", "23.4px", name="wangp_text_90")
theme_spacing_size = Size("0.9px", "1.8px", "3.6px", "5.4px", "7.2px", "9px", "14.4px", name="wangp_spacing_90")
theme_radius_size = Size("0.9px", "1.8px", "3.6px", "5.4px", "7.2px", "10.8px", "19.8px", name="wangp_radius_90")
app = None
# All media attachment keys for queue save/load
ATTACHMENT_KEYS = ["image_start", "image_end", "image_refs", "image_guide", "image_mask",
                   "video_guide",  "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_source", "custom_guide"]

from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
lock = threading.Lock()
current_task_id = None
task_id = 0
unique_id = 0
unique_id_lock = threading.Lock()
offloadobj = enhancer_offloadobj = wan_model = None
reload_needed = True
_HANDLER_MODULES = [
    "shared.qtypes.scaled_fp8",
    "shared.qtypes.nvfp4",
    "shared.qtypes.nunchaku_int4",
    "shared.qtypes.nunchaku_fp4",
    "shared.qtypes.gguf",
]
quant_router.unregister_handler(".fp8_quanto_bridge")
for handler in _HANDLER_MODULES:
    quant_router.register_handler(handler)
from shared.qtypes import gguf as gguf_handler
quant_router.register_file_extension("gguf", gguf_handler)
from shared.kernels.quanto_int8_inject import maybe_enable_quanto_int8_kernel, disable_quanto_int8_kernel


def apply_int8_kernel_setting(enabled: int, notify_disabled = False) -> bool:
    global enable_int8_kernels, verbose_level
    try:
        enable_int8_kernels = 1 if int(enabled) == 1 else 0
    except Exception:
        enable_int8_kernels = 0
    os.environ["WAN2GP_QUANTO_INT8_KERNEL"] = "1" if enable_int8_kernels == 1 else "0"
    if enable_int8_kernels == 1:
        return bool(maybe_enable_quanto_int8_kernel(verbose_level=verbose_level))
    disable_quanto_int8_kernel(notify_disabled)
    return False

def set_wgp_global(variable_name: str, new_value: any) -> str:
    if variable_name not in globals():
        error_msg = f"Plugin tried to modify a non-existent global: '{variable_name}'."
        print(f"ERROR: {error_msg}")
        gr.Warning(error_msg)
        return f"Error: Global variable '{variable_name}' does not exist."

    try:
        globals()[variable_name] = new_value
    except Exception as e:
        error_msg = f"Error while setting global '{variable_name}': {e}"
        print(f"ERROR: {error_msg}")
        return error_msg

def clear_gen_cache():
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]



def release_model():
    global wan_model, offloadobj, reload_needed
    wan_model = None
    clear_gen_cache()
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
    offload.flush_torch_caches()
    gc.collect()
    reload_needed = True
def get_unique_id():
    global unique_id  
    with unique_id_lock:
        unique_id += 1
    return str(time.time()+unique_id)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif seconds >= 60:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{seconds:.1f}s"

def format_generation_time(seconds):
    """Format generation time showing raw seconds with human-readable time in parentheses when over 60s"""
    raw_seconds = f"{int(seconds)}s"
    
    if seconds < 60:
        return raw_seconds
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        human_readable = f"{hours}h {minutes}m {secs}s"
    else:
        human_readable = f"{minutes}m {secs}s"
    
    return f"{raw_seconds} ({human_readable})"

def pil_to_base64_uri(pil_image, format="png", quality=75):
    if pil_image is None:
        return None

    if isinstance(pil_image, str):
        virtual_image = get_virtual_image(pil_image)
        if virtual_image is not None:
            pil_image = virtual_image
        # Check file type and load appropriately
        elif has_video_file_extension(pil_image):
            from shared.utils.utils import get_video_frame
            pil_image = get_video_frame(pil_image, 0)
        elif has_image_file_extension(pil_image):
            pil_image = Image.open(pil_image)
        else:
            # Audio or unknown file type - can't convert to image
            return None

    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'jpeg' and pil_image.mode == 'RGBA':
            img_to_save = pil_image.convert('RGB')
        elif format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
             img_to_save = pil_image.convert('RGBA')
        elif pil_image.mode == 'P':
             img_to_save = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
        if format.lower() == 'jpeg':
            img_to_save.save(buffer, format=format, quality=quality)
        else:
            img_to_save.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None


def _open_image_input(image):
    if not isinstance(image, str):
        return image
    virtual_image = get_virtual_image(image)
    return virtual_image if virtual_image is not None else Image.open(image)

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def get_state_model_type(state):
    key= "model_type" if state.get("active_form", "add") == "add" else "edit_model_type"
    return state[key]

def compute_sliding_window_no(current_video_length, sliding_window_size, discard_last_frames, reuse_frames):
    left_after_first_window = current_video_length - sliding_window_size + discard_last_frames
    return 1 + math.ceil(left_after_first_window / (sliding_window_size - discard_last_frames - reuse_frames))

def clean_image_list(gradio_list):
    if not isinstance(gradio_list, list): gradio_list = [gradio_list]
    gradio_list = [ tup[0] if isinstance(tup, tuple) else tup for tup in gradio_list ]        

    if any( not isinstance(image, (Image.Image, str))  for image in gradio_list): return None
    if any( isinstance(image, str) and not has_image_file_extension(image) for image in gradio_list): return None
    gradio_list = [convert_image(_open_image_input(img)) for img in gradio_list]
    return gradio_list


def silent_cancel_edit(state):
    gen = get_gen_info(state)
    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gen["queue_paused_for_edit"] = False
    return gr.Tabs(selected="video_gen"), None, gr.update(visible=False)

def cancel_edit(state):
    gen = get_gen_info(state)
    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gen["queue_paused_for_edit"] = False
        gr.Info("Edit cancelled. Resuming queue processing.")
    else:
        gr.Info("Edit cancelled.")
    return gr.Tabs(selected="video_gen"), gr.update(visible=False)

def validate_edit(state):
    state["validate_edit_success"] = 0
    model_type = get_state_model_type(state)

    inputs = state.get("edit_state", None)
    if inputs is None: 
        return
    override_inputs, prompts, image_start, image_end, _validation_error = validate_settings(state, model_type, True, inputs)
    if override_inputs is None: 
        return
    inputs.update(override_inputs) 
    state["edit_state"] = inputs
    state["validate_edit_success"] = 1

def edit_task_in_queue( state ):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    editing_task_id = state.get("editing_task_id", None)

    new_inputs = state.pop("edit_state", None)

    if editing_task_id is None or new_inputs is None:
        gr.Warning("No task selected for editing.")
        return None, gr.Tabs(selected="video_gen"), gr.update(visible=False), gr.update()

    if state.get("validate_edit_success", 0) == 0:
        return None, gr.update(), gr.update(), gr.update()


    task_to_edit_index = -1
    with lock:
        task_to_edit_index = next((i for i, task in enumerate(queue) if task['id'] == editing_task_id), -1)

    if task_to_edit_index == -1:
        gr.Warning("Task not found in queue. It might have been processed or deleted.")
        state["editing_task_id"] = None
        gen["queue_paused_for_edit"] = False
        return None, gr.Tabs(selected="video_gen"), gr.update(visible=False), gr.update()
    model_type = get_state_model_type(state)
    new_inputs["model_type"] = model_type 
    new_inputs["state"] = state 
    new_inputs["model_filename"] = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)

    task_to_edit = queue[task_to_edit_index]
            
    task_to_edit['params'] = new_inputs
    task_to_edit['prompt'] = new_inputs.get('prompt')
    task_to_edit['length'] = new_inputs.get('video_length')
    task_to_edit['steps'] = new_inputs.get('num_inference_steps')
    update_task_thumbnails(task_to_edit, task_to_edit['params'])
    
    gr.Info(f"Task ID {task_to_edit['id']} has been updated successfully.")

    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gr.Info("Resuming queue processing.")
        gen["queue_paused_for_edit"] = False

    return task_to_edit_index -1, gr.Tabs(selected="video_gen"), gr.update(visible=False), update_queue_data(queue)

def process_prompt_and_add_tasks(state, current_gallery_tab, model_choice):
    def ret():
        return gr.update(), gr.update()

    gen = get_gen_info(state)

    current_gallery_tab
    gen["last_was_audio"] = current_gallery_tab == 1

    if state.get("validate_success",0) != 1:
        ret()
    
    state["validate_success"] = 0
    model_type = get_state_model_type(state)
    inputs = get_model_settings(state, model_type)

    if model_choice != model_type or inputs ==None:
        raise gr.Error("Webform can not be used as the App has been restarted since the form was displayed. Please refresh the page")
    
    inputs["state"] =  state
    inputs["model_type"] = model_type
    inputs.pop("lset_name", None)
    if inputs == None:
        gr.Warning("Internal state error: Could not retrieve inputs for the model.")
        queue = gen.get("queue", [])
        return ret()
    
    mode = inputs["mode"]
    if mode.startswith("edit_"):
        edit_video_source =gen.get("edit_video_source", None)
        edit_overrides =gen.get("edit_overrides", None)
        _ , _ , _, frames_count = get_video_info(edit_video_source)
        if frames_count > max_source_video_frames:
            gr.Info(f"Post processing is not supported on videos longer than {max_source_video_frames} frames. Output Video will be truncated")
            # return
        for prop in ["state", "model_type", "mode"]:
            edit_overrides[prop] = inputs[prop]
        for k,v in inputs.items():
            inputs[k] = None    
        inputs.update(edit_overrides)
        del gen["edit_video_source"], gen["edit_overrides"]
        inputs["video_source"]= edit_video_source 
        prompt = []

        repeat_generation = 1
        if mode == "edit_postprocessing":
            spatial_upsampling = inputs.get("spatial_upsampling","")
            if len(spatial_upsampling) >0: prompt += ["Spatial Upsampling"]
            temporal_upsampling = inputs.get("temporal_upsampling","")
            if len(temporal_upsampling) >0: prompt += ["Temporal Upsampling"]
            if has_image_file_extension(edit_video_source)  and len(temporal_upsampling) > 0:
                gr.Info("Temporal Upsampling can not be used with an Image")
                return ret()
            film_grain_intensity  = inputs.get("film_grain<<_intensity",0)
            film_grain_saturation  = inputs.get("film_grain_saturation",0.5)        
            # if film_grain_intensity >0: prompt += [f"Film Grain: intensity={film_grain_intensity}, saturation={film_grain_saturation}"]
            if film_grain_intensity >0: prompt += ["Film Grain"]
        elif mode =="edit_remux":
            MMAudio_setting = inputs.get("MMAudio_setting",0)
            repeat_generation= inputs.get("repeat_generation",1)
            audio_source = inputs["audio_source"]
            if  MMAudio_setting== 1:
                prompt += ["MMAudio"]
                audio_source = None 
                inputs["audio_source"] = audio_source
            else:
                if audio_source is None:
                    gr.Info("You must provide a custom Audio")
                    return ret()
                prompt += ["Custom Audio"]
                repeat_generation = 1
            seed = inputs.get("seed",None)
        inputs["repeat_generation"] = repeat_generation
        if len(prompt) == 0:
            if mode=="edit_remux":
                gr.Info("You must choose at least one Remux Method")
            else:
                gr.Info("You must choose at least one Post Processing Method")
            return ret()
        inputs["prompt"] = ", ".join(prompt)
        add_video_task(**inputs)
        new_prompts_count = gen["prompts_max"] = 1 + gen.get("prompts_max",0)
        state["validate_success"] = 1
        queue= gen.get("queue", [])
        return update_queue_data(queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()

    inputs, prompts, image_start, image_end, _validation_error = validate_settings(state, model_type, False, inputs)

    if inputs is None:
        return ret()

    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]

    if "W" not in multi_prompts_gen_type:
        image_slots = image_start if image_start != None and len(image_start) > 0 else image_end
        if image_slots != None and len(image_slots) > 0:
            if inputs["multi_images_gen_type"] == 0:
                new_prompts = []
                new_image_start = []
                new_image_end = []
                for i in range(len(prompts) * len(image_slots)):
                    new_prompts.append(  prompts[ i % len(prompts)] )
                    if image_start != None:
                        new_image_start.append(image_start[i // len(prompts)] )
                    if image_end != None:
                        new_image_end.append(image_end[i // len(prompts)] )
                prompts = new_prompts
                image_start = new_image_start if image_start != None else None
                image_end = new_image_end if image_end != None else None
            else:
                if len(prompts) >= len(image_slots):
                    if len(prompts) % len(image_slots) != 0:
                        gr.Info("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                        return ret()
                    rep = len(prompts) // len(image_slots)
                    new_image_start = []
                    new_image_end = []
                    for i, _ in enumerate(prompts):
                        if image_start != None:
                            new_image_start.append(image_start[i//rep] )
                        if image_end != None:
                            new_image_end.append(image_end[i//rep] )
                    image_start = new_image_start if image_start != None else None
                    image_end = new_image_end if image_end != None else None
                else: 
                    if len(image_slots) % len(prompts)  !=0:
                        gr.Info("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                        return ret()
                    rep = len(image_slots) // len(prompts)
                    new_prompts = []
                    for i, _ in enumerate(image_slots):
                        new_prompts.append(  prompts[ i//rep] )
                    prompts = new_prompts
            if image_start == None or len(image_start) == 0:
                image_start = [None] * len(prompts)
            if image_end == None or len(image_end) == 0:
                image_end = [None] * len(prompts)

            for single_prompt, start, end in zip(prompts, image_start, image_end) :
                inputs.update({
                    "prompt" : single_prompt,
                    "image_start": start,
                    "image_end" : end,
                })
                add_video_task(**inputs)
        else:
            for single_prompt in prompts :
                inputs["prompt"] = single_prompt 
                add_video_task(**inputs)
        new_prompts_count = len(prompts)
    else:
        new_prompts_count = 1
        add_video_task(**inputs)
    new_prompts_count += gen.get("prompts_max",0)
    gen["prompts_max"] = new_prompts_count
    state["validate_success"] = 1
    queue= gen.get("queue", [])
    return update_queue_data(queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()

def get_custom_setting_key(index):
    return f"custom_setting_{index + 1}"

def _normalize_custom_setting_type(setting_type):
    parsed_type = str(setting_type or "text").strip().lower()
    return parsed_type if parsed_type in CUSTOM_SETTING_TYPES else "text"

def _normalize_custom_setting_name(name):
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")
    return normalized

def get_custom_setting_id(setting_def, setting_index):
    explicit_id = setting_def.get("id", None)
    if explicit_id is not None and len(str(explicit_id).strip()) > 0:
        normalized_id = _normalize_custom_setting_name(explicit_id)
        if len(normalized_id) > 0:
            return normalized_id
    for field_name in ("name", "param"):
        normalized_name = _normalize_custom_setting_name(setting_def.get(field_name, ""))
        if len(normalized_name) > 0:
            return normalized_name
    return get_custom_setting_key(setting_index)

def get_model_custom_settings(model_def):
    if not isinstance(model_def, dict):
        return []
    custom_settings = model_def.get("custom_settings", [])
    if not isinstance(custom_settings, list):
        return []
    normalized = []
    used_ids = set()
    for idx, setting in enumerate(custom_settings[:CUSTOM_SETTINGS_MAX]):
        if not isinstance(setting, dict):
            continue
        one = setting.copy()
        one["label"] = str(one.get("label", f"Custom Setting {idx + 1}"))
        one["name"] = str(one.get("name", f"Custom Setting {idx + 1}"))
        one["type"] = _normalize_custom_setting_type(one.get("type", "text"))
        setting_id = get_custom_setting_id(one, idx)
        if setting_id in used_ids:
            setting_id = get_custom_setting_key(idx)
        used_ids.add(setting_id)
        one["id"] = setting_id
        normalized.append(one)
    return normalized


def end_frames_always_enabled(model_def):
    return bool((model_def or {}).get("end_frames_always_enabled", False))


def end_frames_option_visible(model_def, image_prompt_type):
    if "E" not in (model_def or {}).get("image_prompt_types_allowed", ""):
        return False
    return end_frames_always_enabled(model_def) or any_letters(image_prompt_type or "", "SVL")


def injected_frames_positions_visible(video_prompt_type):
    return "F" in (video_prompt_type or "")


def input_video_strength_visible(model_def, image_prompt_type, video_prompt_type=""):
    input_video_strength = model_def.get("input_video_strength", {})
    input_video_strength_label = input_video_strength.get("label", "").strip()
    return len(input_video_strength_label) > 0 and (
        any_letters(image_prompt_type or "", "SVLE") or injected_frames_positions_visible(video_prompt_type)
    )


def parse_custom_setting_typed_value(raw_value, setting_type):
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if len(raw_value) == 0:
            return None, None
    setting_type = _normalize_custom_setting_type(setting_type)
    if setting_type == "int":
        if isinstance(raw_value, bool):
            return None, "Expected an integer value."
        if isinstance(raw_value, int):
            return raw_value, None
        if isinstance(raw_value, float):
            if raw_value.is_integer():
                return int(raw_value), None
            return None, "Expected an integer value."
        try:
            return int(str(raw_value).strip()), None
        except Exception:
            try:
                float_value = float(str(raw_value).strip())
                if float_value.is_integer():
                    return int(float_value), None
            except Exception:
                pass
            return None, "Expected an integer value."
    if setting_type == "float":
        if isinstance(raw_value, bool):
            return None, "Expected a float value."
        try:
            return float(raw_value), None
        except Exception:
            return None, "Expected a float value."
    return str(raw_value).strip(), None

def get_custom_setting_value_from_dict(custom_settings_values, setting_def, setting_index):
    setting_id = setting_def.get("id", get_custom_setting_id(setting_def, setting_index))
    if isinstance(custom_settings_values, dict) and setting_id in custom_settings_values:
        return custom_settings_values.get(setting_id, None)
    return setting_def.get("default", "")

def collect_custom_settings_from_inputs(model_def, inputs, strict=False):
    custom_settings_dict = {}
    existing_custom_settings = inputs.get("custom_settings", None)
    if not isinstance(existing_custom_settings, dict):
        existing_custom_settings = {}
    custom_settings = get_model_custom_settings(model_def)
    for idx, setting_def in enumerate(custom_settings):
        slot_key = get_custom_setting_key(idx)
        setting_id = setting_def["id"]
        raw_value = inputs.get(slot_key, None)
        if raw_value is None and setting_id in existing_custom_settings:
            raw_value = existing_custom_settings.get(setting_id, None)
        parsed_value, parse_error = parse_custom_setting_typed_value(raw_value, setting_def.get("type", "text"))
        if parse_error is not None:
            if strict:
                return None, f"{setting_def.get('label', slot_key)} {parse_error}"
            if raw_value is not None:
                raw_text = str(raw_value).strip() if isinstance(raw_value, str) else raw_value
                if not (isinstance(raw_text, str) and len(raw_text) == 0):
                    custom_settings_dict[setting_id] = raw_text
            continue
        if parsed_value is not None:
            custom_settings_dict[setting_id] = parsed_value
    return custom_settings_dict if len(custom_settings_dict) > 0 else None, None

def clear_custom_setting_slots(inputs):
    for idx in range(CUSTOM_SETTINGS_MAX):
        inputs.pop(get_custom_setting_key(idx), None)

def validate_settings(state, model_type, single_prompt, inputs, silent=False):
    def err(error=""):
        error = str(error or "")
        if len(error) > 0 and not silent:
            gr.Info(error)
        return None, None, None, None, error

    model_def = get_model_def(model_type)
    model_handler = get_model_handler(model_type)
    image_outputs = inputs["image_mode"] > 0
    any_steps_skipping = model_def.get("tea_cache", False) or model_def.get("mag_cache", False)
    model_type = get_base_model_type(model_type)

    model_filename = get_model_filename(model_type)  


    if inputs.get("cfg_star_switch", 0) != 0 and inputs.get("apg_switch", 0) != 0:
        return err("Adaptive Progressive Guidance and Classifier Free Guidance Star can not be set at the same time")
    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]
    prompt = inputs["prompt"]
    keep_empty_lines = model_def.get("preserve_empty_prompt_lines", False) or "P" in multi_prompts_gen_type or prompt_parser.PROMPT_UNIT_PREFIX in prompt
    prompt, errors = prompt_parser.process_template(prompt, keep_comments=prompt_parser.PROMPT_UNIT_PREFIX in prompt, keep_empty_lines=keep_empty_lines)
    if len(errors) > 0:
        return err("Error processing prompt template: " + errors)
    prompt = prompt.strip("\n").strip()

    prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type, single_prompt=single_prompt)
    if len(prompts) == 0:
        return err("Prompt cannot be empty.")
    inputs["prompt"] = prompt_parser.serialize_prompt_units(prompt, prompts, multi_prompts_gen_type)

    parsed_custom_settings, custom_settings_error = collect_custom_settings_from_inputs(model_def, inputs, strict=True)
    if custom_settings_error is not None:
        return err(custom_settings_error)
    inputs["custom_settings"] = parsed_custom_settings
    clear_custom_setting_slots(inputs)
    extra_settings_error = extra_settings.validate_inputs(inputs, model_def, get_max_frames=get_max_frames)
    if len(extra_settings_error) > 0:
        return err(extra_settings_error)

    if hasattr(model_handler, "validate_generative_prompt"):
        for one_prompt in prompts:
            error = model_handler.validate_generative_prompt(model_type, model_def, inputs, one_prompt)
            if error is not None and len(error) > 0:
                return err(error)

    resolution = inputs["resolution"]
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    image_start = inputs["image_start"]
    image_end = inputs["image_end"]
    image_refs = inputs["image_refs"]
    image_prompt_type = inputs["image_prompt_type"]
    audio_prompt_type = inputs["audio_prompt_type"]
    if image_prompt_type == None: image_prompt_type = ""
    video_prompt_type = inputs["video_prompt_type"]
    if video_prompt_type == None: video_prompt_type = ""
    force_fps = inputs["force_fps"]
    audio_guide = inputs["audio_guide"]
    audio_guide2 = inputs["audio_guide2"]
    audio_source = inputs["audio_source"]
    video_guide = inputs["video_guide"]
    image_guide = inputs["image_guide"]
    video_mask = inputs["video_mask"]
    image_mask = inputs["image_mask"]
    custom_guide = inputs["custom_guide"]
    speakers_locations = inputs["speakers_locations"]
    video_source = inputs["video_source"]
    frames_positions = inputs["frames_positions"]
    keep_frames_video_guide= inputs["keep_frames_video_guide"] 
    keep_frames_video_source = inputs["keep_frames_video_source"]
    denoising_strength= inputs["denoising_strength"]     
    masking_strength= inputs["masking_strength"]     
    input_video_strength = inputs.get("input_video_strength", 1.0)
    sliding_window_size = inputs["sliding_window_size"]
    sliding_window_overlap = inputs["sliding_window_overlap"]
    sliding_window_discard_last_frames = inputs["sliding_window_discard_last_frames"]
    video_length = inputs["video_length"]
    num_inference_steps= inputs["num_inference_steps"]
    skip_steps_cache_type= inputs["skip_steps_cache_type"]
    MMAudio_setting = inputs["MMAudio_setting"]
    image_mode = inputs["image_mode"]
    switch_threshold = inputs["switch_threshold"]
    loras_multipliers = inputs["loras_multipliers"]
    activated_loras = inputs["activated_loras"]
    guidance_phases= inputs["guidance_phases"]
    model_switch_phase = inputs["model_switch_phase"]    
    switch_threshold = inputs["switch_threshold"]
    switch_threshold2 = inputs["switch_threshold2"]
    video_guide_outpainting = inputs["video_guide_outpainting"]
    video_guide_outpainting_ratio = inputs.get("video_guide_outpainting_ratio", "")
    spatial_upsampling = inputs["spatial_upsampling"]
    motion_amplitude = inputs["motion_amplitude"]
    self_refiner_setting = inputs["self_refiner_setting"]
    self_refiner_plan = inputs["self_refiner_plan"]
    model_mode = inputs["model_mode"]
    medium = "Videos" if image_mode == 0 else "Images"

    if image_start is not None and not isinstance(image_start, list): image_start = [image_start]
    outpainting_modes = model_def.get("video_guide_outpainting", [])
    if image_mode not in outpainting_modes: 
        video_guide_outpainting = ""
        video_guide_outpainting_ratio = ""

    outpainting_dims = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio)

    model_modes_visibility = [0,1,2]
    model_mode_choices = model_def.get("model_modes", None)
    if model_mode_choices is not None: model_modes_visibility= model_mode_choices.get("image_modes", model_modes_visibility)
    if model_mode is not None and image_mode not in model_modes_visibility:
        model_mode = None
    if server_config.get("fit_canvas", 0) == 2 and outpainting_dims is not None and any_letters(video_prompt_type, "VKF"):
        gr.Info("Output Resolution Cropping will be not used for this Generation as it is not compatible with Video Outpainting")
    if self_refiner_setting != 0:
        from shared.utils.self_refiner import normalize_self_refiner_plan, convert_refiner_list_to_string
        if isinstance(self_refiner_plan, list):
            self_refiner_plan = convert_refiner_list_to_string(self_refiner_plan)
        max_p = model_def.get("self_refiner_max_plans", 1)
        _, error = normalize_self_refiner_plan(self_refiner_plan, max_plans=max_p)
        if len(error):
            return err(error)

    if not model_def.get("motion_amplitude", False): motion_amplitude = 1.
    if "vae" in spatial_upsampling:
        if image_mode not in model_def.get("vae_upsampler", []):
            return err(f"VAE Spatial Upsampling is not available for {medium}")

    if len(activated_loras) > 0:
        activated_loras = update_loras_url_cache(get_lora_dir(model_type), activated_loras)
        inputs["activated_loras"] = activated_loras
        error = check_loras_exist(model_type, activated_loras)
        if len(error) > 0:
            return err(error)
    if  model_def.get("lock_guidance_phases", False):
        guidance_phases = model_def.get("guidance_max_phases", 0)
    else:
        guidance_phases = min(guidance_phases, model_def.get("guidance_max_phases", 0))
                          
    if len(loras_multipliers) > 0:
        _, _, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases= guidance_phases)
        if len(errors) > 0: 
            return err(f"Error parsing Loras Multipliers: {errors}")
    if guidance_phases == 3:
        if switch_threshold < switch_threshold2:
            return err(f"Phase 1-2 Switch Noise Level ({switch_threshold}) should be Greater than Phase 2-3 Switch Noise Level ({switch_threshold2}). As a reminder, noise will gradually go down from 1000 to 0.")
    else:
        model_switch_phase = 1
        
    if not any_steps_skipping: skip_steps_cache_type = ""
    if not model_def.get("lock_inference_steps", False) and model_type in ["ltxv_13B"] and num_inference_steps < 20:
        return err("The minimum number of steps should be 20")
    if skip_steps_cache_type == "mag":
        if num_inference_steps > 50:
            return err("Mag Cache maximum number of steps is 50")
        
    if image_mode > 0:
        audio_prompt_type = ""

    if "K" in audio_prompt_type and "V" not in video_prompt_type:
        return err("You must enable a Control Video to use the Control Video Audio Track as an audio prompt")

    if ("B" in audio_prompt_type or "X" in audio_prompt_type) and not model_def.get("one_speaker_only", False):
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
        if len(error) > 0:
            return err(error)

    if MMAudio_setting != 0 and get_mmaudio_settings(server_config)[0] and video_length <16: #should depend on the architecture
        gr.Info("MMAudio can generate an Audio track only if the Video is at least 1s long")
    if "F" in video_prompt_type:
        if len(frames_positions.strip()) > 0:
            positions = frames_positions.replace(","," ").split(" ")
            for pos_str in positions:
                if not pos_str in ["L", "l"] and len(pos_str)>0: 
                    if not is_integer(pos_str):
                        return err(f"Invalid Frame Position '{pos_str}'")
                    pos = int(pos_str)
                    if pos <1 or pos > max_source_video_frames:
                        return err(f"Invalid Frame Position Value'{pos_str}'")
    else:
        frames_positions = None

    if audio_source is not None and MMAudio_setting != 0:
        return err("MMAudio and Custom Audio Soundtrack can't not be used at the same time")
    if len(filter_letters(image_prompt_type, "VLG")) > 0 and len(keep_frames_video_source) > 0:
        if not is_integer(keep_frames_video_source) or int(keep_frames_video_source) == 0:
            return err("The number of frames to keep must be a non null integer")
    else:
        keep_frames_video_source = ""

    if image_outputs:
        image_prompt_type = image_prompt_type.replace("V", "").replace("L", "")
    custom_guide_def = model_def.get("custom_guide", None)
    if custom_guide_def is not None:
        if custom_guide is None and custom_guide_def.get("required", False):
            return err(f"You must provide a {custom_guide_def.get('label', 'Custom Guide')}")
    else:
        custom_guide = None

    if "V" in image_prompt_type:
        if video_source == None:
            return err("You must provide a Source Video file to continue")
    else:
        video_source = None

    if not input_video_strength_visible(model_def, image_prompt_type, video_prompt_type):
        input_video_strength = 1.0

    if "A" in audio_prompt_type:
        if audio_guide == None and not model_def.get("auto_null_audio", False):
            return err("You must provide an Audio Source")
    else:
        audio_guide = None


    if "B" in audio_prompt_type:
        if audio_guide2 == None:
            return err("You must provide a second Audio Source")
    else:
        audio_guide2 = None
    if not all_letters(audio_prompt_type, "AB"):
        audio_prompt_type = del_in_sequence(audio_prompt_type, "N")

    if model_type in ["vace_multitalk_14B"] and ("B" in audio_prompt_type or "X" in audio_prompt_type):
        if not "I" in video_prompt_type and not not "V" in video_prompt_type:
            gr.Info("To get good results with Multitalk and two people speaking, it is recommended to set a Reference Frame or a Control Video (potentially truncated) that contains the two people one on each side")

    if model_def.get("one_image_ref_needed", False):
        if image_refs is None:
            return err("You must provide an Image Reference")
        if len(image_refs) > 1:
            return err("Only one Image Reference (a person) is supported for the moment by this model")
    if model_def.get("at_least_one_image_ref_needed", False):
        if image_refs is None:
            return err("You must provide at least one Image Reference")
        
    if "I" in video_prompt_type:
        if image_refs == None or len(image_refs) == 0:
            return err("You must provide at least one Reference Image")
        image_refs = clean_image_list(image_refs)
        if image_refs == None :
            return err("A Reference Image should be an Image")
    else:
        image_refs = None

    if "V" in video_prompt_type:
        if image_outputs:
            if image_guide is None:
                return err("You must provide a Control Image")
        else:
            if video_guide is None:
                return err("You must provide a Control Video")
        if "A" in video_prompt_type and not "U" in video_prompt_type:             
            if image_outputs:
                if image_mask is None:
                    return err("You must provide a Image Mask")
            else:
                if video_mask is None:
                    return err("You must provide a Video Mask")
        else:
            video_mask = None
            image_mask = None

        if "G" in video_prompt_type:
                if denoising_strength < 1. and not model_def.get("custom_denoising_strength", False):
                    gr.Info(f"With Denoising Strength {denoising_strength:.1f}, Denoising will start at Step no {int(round(num_inference_steps * (1. - denoising_strength),4))} ")
        else: 
            denoising_strength = 1.0
                    
        if "G" in video_prompt_type or model_def.get("mask_strength_always_enabled", False):
                if "A" in video_prompt_type and "U" not in video_prompt_type and masking_strength < 1.:
                    masking_duration = math.ceil(num_inference_steps * masking_strength)
                    if masking_strength:
                        gr.Info(f"With Masking Strength {masking_strength:.1f}, Masking will last {masking_duration}{' Step' if masking_duration==1 else ' Steps'}")
        else: 
            masking_strength = 1.0
        if len(keep_frames_video_guide) > 0 and model_type in ["ltxv_13B"]:
            return err("Keep Frames for Control Video is not supported with LTX Video")
        _, error = parse_keep_frames_video_guide(keep_frames_video_guide, video_length)
        if len(error) > 0:
            return err(f"Invalid Keep Frames property: {error}")
    else:
        video_guide = None
        image_guide = None
        video_mask = None
        image_mask = None
        keep_frames_video_guide = ""
        denoising_strength = 1.0
        masking_strength = 1.0
    
    if image_outputs:
        video_guide = None
        video_mask = None
    else:
        image_guide = None
        image_mask = None
    image_prompt_types_allowed = model_def.get("image_prompt_types_allowed", "")
    if "S" in image_prompt_type:
        if "S" not in image_prompt_types_allowed:
            return err("This model doesn't accept a Start Image")
    
        if model_def.get("black_frame", False) and len(image_start or [])==0:
            if "E" in image_prompt_type and len(image_end or []):
                image_end = clean_image_list(image_end)        
                image_start = [Image.new("RGB", image.size, (0, 0, 0, 255)) for image in image_end] 
            else:
                image_start = [Image.new("RGB", (width, height), (0, 0, 0, 255))] 

        if image_start == None or isinstance(image_start, list) and len(image_start) == 0:
            return err("You must provide a Start Image")
        image_start = clean_image_list(image_start)        
        if image_start == None :
            return err("Start Image should be an Image")
        if "W" in multi_prompts_gen_type and len(image_start) > 1:
            return err("Only one Start Image is supported when a multi-prompt Sliding Window mode is selected")
    else:
        image_start = None

    if not end_frames_always_enabled(model_def) and not any_letters(image_prompt_type, "SVL"):
        image_prompt_type = image_prompt_type.replace("E", "")
    if "E" in image_prompt_type:
        if "E" not in image_prompt_types_allowed:
            return err("This model doesn't accept an End Image")
    
        if image_end == None or isinstance(image_end, list) and len(image_end) == 0:
            return err("You must provide an End Image")
        image_end = clean_image_list(image_end)        
        if image_end == None :
            return err("End Image should be an Image")
        if (video_source is not None or "L" in image_prompt_type):
            if "W" not in multi_prompts_gen_type and len(image_end)> 1:
                return err("If you want to Continue a Video, you can use Multiple End Images only when a multi-prompt Sliding Window mode is selected")        
        elif "W" not in multi_prompts_gen_type:
            if len(image_start or []) > 0 and len(image_start or []) != len(image_end or []):
                return err("The number of Start and End Images should be the same unless a multi-prompt Sliding Window mode is selected")    
    else:        
        image_end = None

    if "V" in video_prompt_type and "O" in video_prompt_type:
        if image_start is None and video_source is None and "L" not in video_prompt_type and not all_letters(video_prompt_type, "IK"):
            return err("Aligned Pose transfer requires a Start Image, a Source Video to continue or Background Ref Frame to be used")    
        if "A" in video_prompt_type and any_letters(video_prompt_type, "YWZ"):
            return err("Aligned Pose transfer supports only Inpainting process outside the masked area")    

    if test_any_sliding_window(model_type) and image_mode == 0:
        if video_length > sliding_window_size:
            if test_class_t2v(model_type) and not "G" in video_prompt_type :
                return err(f"You have requested to Generate Sliding Windows with a Text to Video model. Unless you use the Video to Video feature this is useless as a t2v model doesn't see past frames and it will generate the same video in each new window.")
            full_video_length = video_length if video_source is None else video_length +  sliding_window_overlap -1
            extra = "" if full_video_length == video_length else f" including {sliding_window_overlap} added for Video Continuation"
            no_windows = compute_sliding_window_no(full_video_length, sliding_window_size, sliding_window_discard_last_frames, sliding_window_overlap)
            gr.Info(f"The Number of Frames to generate ({video_length}{extra}) is greater than the Sliding Window Size ({sliding_window_size}), {no_windows} Windows will be generated")
    if "recam" in model_filename:
        if video_guide == None:
            return err("You must provide a Control Video")
        computed_fps = get_computed_fps(force_fps, model_type , video_guide, video_source )
        frames = get_resampled_video(video_guide, 0, 81, computed_fps)
        if len(frames)<81:
            return err(f"Recammaster Control video should be at least 81 frames once the resampling at {computed_fps} fps has been done")

    if "hunyuan_custom_custom_edit" in model_filename:
        if len(keep_frames_video_guide) > 0: 
            return err("Filtering Frames with this model is not supported")

    if "W" in multi_prompts_gen_type or single_prompt:
        if image_start != None and len(image_start) > 1:
            return err("Only one Start Image can be provided in Edit Mode" if single_prompt else "Only one Start Image must be provided if multiple prompts are used for different windows")

        # if image_end != None and len(image_end) > 1:
        #     gr.Info("Only one End Image must be provided if multiple prompts are used for different windows") 
        #     return

    override_inputs = {
        "image_start": image_start[0] if image_start !=None and len(image_start) > 0 else None,
        "image_end": image_end, #[0] if image_end !=None and len(image_end) > 0 else None,
        "image_refs": image_refs,
        "audio_guide": audio_guide,
        "audio_guide2": audio_guide2,
        "audio_source": audio_source,
        "video_guide": video_guide,
        "image_guide": image_guide,
        "video_mask": video_mask,
        "image_mask": image_mask,
        "custom_guide": custom_guide,
        "video_source": video_source,
        "frames_positions": frames_positions,
        "keep_frames_video_source": keep_frames_video_source,
        "input_video_strength": input_video_strength,
        "keep_frames_video_guide": keep_frames_video_guide,
        "denoising_strength": denoising_strength,
        "masking_strength": masking_strength,
        "image_prompt_type": image_prompt_type,
        "video_prompt_type": video_prompt_type,        
        "audio_prompt_type": audio_prompt_type,
        "guidance_phases": guidance_phases,
        "skip_steps_cache_type": skip_steps_cache_type,
        "model_switch_phase": model_switch_phase,
        "motion_amplitude": motion_amplitude,
        "model_mode": model_mode,
        "video_guide_outpainting": video_guide_outpainting,
        "video_guide_outpainting_ratio": inputs.get("video_guide_outpainting_ratio", ""),
        "custom_settings": inputs.get("custom_settings", None),
        "self_refiner_plan": self_refiner_plan,
    } 
    inputs.update(override_inputs)
    if hasattr(model_handler, "validate_generative_settings"):
        error = model_handler.validate_generative_settings(model_type, model_def, inputs)
        if error is not None and len(error) > 0:
            return err(error)
    return inputs, prompts, image_start, image_end, ""


def get_preview_images(inputs):
    inputs_to_query = ["image_start", "video_source", "image_end",  "video_guide", "image_guide", "video_mask", "image_mask", "image_refs" ]
    labels = ["Start Image", "Video Source", "End Image", "Video Guide", "Image Guide", "Video Mask", "Image Mask", "Image Reference"]
    start_image_data = None
    start_image_labels = []
    end_image_data = None
    end_image_labels = []
    for label, name in  zip(labels,inputs_to_query):
        image= inputs.get(name, None)
        if image is not None:
            image= [image] if not isinstance(image, list) else image.copy()
            if start_image_data == None:
                start_image_data = image
                start_image_labels += [label] * len(image)
            else:
                if end_image_data == None:
                    end_image_data = image
                else:
                    end_image_data += image 
                end_image_labels += [label] * len(image)

    if start_image_data != None and len(start_image_data) > 1 and  end_image_data  == None:
        end_image_data = start_image_data [1:]
        end_image_labels = start_image_labels [1:]
        start_image_data = start_image_data [:1] 
        start_image_labels = start_image_labels [:1] 
    return start_image_data, end_image_data, start_image_labels, end_image_labels 

def add_video_task(**inputs):
    global task_id
    state = inputs["state"]
    gen = get_gen_info(state)
    queue = gen["queue"]
    task_id += 1
    current_task_id = task_id

    start_image_data, end_image_data, start_image_labels, end_image_labels = get_preview_images(inputs)
    plugin_data = inputs.pop('plugin_data', {})
    
    queue.append({
        "id": current_task_id,
        "params": inputs.copy(),
        "plugin_data": plugin_data,
        "repeats": inputs.get("repeat_generation",1),
        "length": inputs.get("video_length",0) or 0, 
        "steps": inputs.get("num_inference_steps",0) or 0,
        "prompt": inputs.get("prompt", ""),
        "start_image_labels": start_image_labels,
        "end_image_labels": end_image_labels,
        "start_image_data": start_image_data,
        "end_image_data": end_image_data,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })

def update_task_thumbnails(task,  inputs):
    start_image_data, end_image_data, start_labels, end_labels = get_preview_images(inputs)

    task.update({
        "start_image_labels": start_labels,
        "end_image_labels": end_labels,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })

def move_task(queue, old_index_str, new_index_str):
    try:
        old_idx = int(old_index_str)
        new_idx = int(new_index_str)
    except (ValueError, IndexError):
        return update_queue_data(queue)

    with lock:
        old_idx += 1
        new_idx += 1

        if not (0 < old_idx < len(queue)):
            return update_queue_data(queue)

        item_to_move = queue.pop(old_idx)
        if old_idx < new_idx:
            new_idx -= 1
        clamped_new_idx = max(1, min(new_idx, len(queue)))
        
        queue.insert(clamped_new_idx, item_to_move)

    return update_queue_data(queue)

def remove_task(queue, task_id_to_remove):
    if not task_id_to_remove:
        return update_queue_data(queue)

    with lock:
        idx_to_del = next((i for i, task in enumerate(queue) if task['id'] == task_id_to_remove), -1)
        
        if idx_to_del != -1:
            if idx_to_del == 0:
                wan_model._interrupt = True
            del queue[idx_to_del]
            
    return update_queue_data(queue)

def update_global_queue_ref(queue):
    global global_queue_ref
    with lock:
        global_queue_ref = queue[:]

def _save_queue_to_zip(queue, output):
    """Save queue to ZIP. output can be a filename (str) or BytesIO buffer.
    Returns True on success, False on failure.
    """
    if not queue:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_manifest = []
        file_paths_in_zip = {}

        for task_index, task in enumerate(queue):
            if task is None or not isinstance(task, dict) or task.get('id') is None:
                continue

            params_copy = task.get('params', {}).copy()
            task_id_s = task.get('id', f"task_{task_index}")

            for key in ATTACHMENT_KEYS:
                value = params_copy.get(key)
                if value is None:
                    continue

                is_originally_list = isinstance(value, list)
                items = value if is_originally_list else [value]

                processed_filenames = []
                for item_index, item in enumerate(items):
                    if isinstance(item, Image.Image):
                        item_id = id(item)
                        if item_id in file_paths_in_zip:
                            processed_filenames.append(file_paths_in_zip[item_id])
                            continue
                        filename_in_zip = f"task{task_id_s}_{key}_{item_index}.png"
                        save_path = os.path.join(tmpdir, filename_in_zip)
                        try:
                            item.save(save_path, "PNG")
                            processed_filenames.append(filename_in_zip)
                            file_paths_in_zip[item_id] = filename_in_zip
                        except Exception as e:
                            print(f"Error saving attachment {filename_in_zip}: {e}")
                    elif isinstance(item, str):
                        if item in file_paths_in_zip:
                            processed_filenames.append(file_paths_in_zip[item])
                            continue
                        if not os.path.isfile(item):
                            continue
                        _, extension = os.path.splitext(item)
                        filename_in_zip = f"task{task_id_s}_{key}_{item_index}{extension if extension else ''}"
                        save_path = os.path.join(tmpdir, filename_in_zip)
                        try:
                            shutil.copy2(item, save_path)
                            processed_filenames.append(filename_in_zip)
                            file_paths_in_zip[item] = filename_in_zip
                        except Exception as e:
                            print(f"Error copying attachment {item}: {e}")

                if processed_filenames:
                    params_copy[key] = processed_filenames if is_originally_list else processed_filenames[0]

            # Remove runtime-only keys
            for runtime_key in ['state', 'start_image_labels', 'end_image_labels',
                                'start_image_data_base64', 'end_image_data_base64',
                                'start_image_data', 'end_image_data']:
                params_copy.pop(runtime_key, None)

            params_copy['settings_version'] = settings_version
            params_copy['base_model_type'] = get_base_model_type(params_copy["model_type"])

            manifest_entry = {"id": task.get('id'), "params": params_copy}
            manifest_entry = {k: v for k, v in manifest_entry.items() if v is not None}
            queue_manifest.append(manifest_entry)

        manifest_path = os.path.join(tmpdir, "queue.json")
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(queue_manifest, f, indent=4)
        except Exception as e:
            print(f"Error writing queue.json: {e}")
            return False

        try:
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue.json")
                for saved_file_rel_path in file_paths_in_zip.values():
                    saved_file_abs_path = os.path.join(tmpdir, saved_file_rel_path)
                    if os.path.exists(saved_file_abs_path):
                        zf.write(saved_file_abs_path, arcname=saved_file_rel_path)
            return True
        except Exception as e:
            print(f"Error creating zip: {e}")
            return False

def save_queue_action(state):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    if not queue or len(queue) == 0:
        gr.Info("Queue is empty. Nothing to save.")
        return ""

    zip_buffer = io.BytesIO()
    try:
        if _save_queue_to_zip(queue, zip_buffer):
            zip_buffer.seek(0)
            zip_base64 = base64.b64encode(zip_buffer.getvalue()).decode('utf-8')
            print(f"Queue saved ({len(zip_base64)} chars)")
            return zip_base64
        else:
            gr.Warning("Failed to save queue.")
            return None
    finally:
        zip_buffer.close()

def clean_settings(model_type, params):
    # Use primary_settings as base (not model-specific saved settings)
    # This ensures loaded queues/settings behave predictably
    saved_settings_version = params.get('settings_version', 0)
    merged = primary_settings.copy()
    merged.update(params)
    params.clear()
    params.update(merged)
    fix_settings(model_type, params, saved_settings_version)
    for k, v in primary_settings.items():
        params.setdefault(k, v)
    params.setdefault("client_id", "")
    params.setdefault("mode", "")
    for meta_key in ['type', 'base_model_type']:
        params.pop(meta_key, None)


def _attachment_has_path_values(value):
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, str) and len(item.strip()) > 0 for item in value)
    return False


def _task_has_path_attachments(params):
    return any(_attachment_has_path_values(params.get(key)) for key in ATTACHMENT_KEYS)


def _load_task_attachments(params, media_base_path, cache_dir=None, log_prefix="[load]"):

    for key in ATTACHMENT_KEYS:
        value = params.get(key)
        if value is None:
            continue

        is_originally_list = isinstance(value, list)
        filenames = value if is_originally_list else [value]

        loaded_items = []
        for filename in filenames:
            if not isinstance(filename, str) or not filename.strip():
                print(f"{log_prefix} Warning: Invalid filename for key '{key}'. Skipping.")
                continue
            virtual_spec = parse_virtual_media_path(filename)
            if virtual_spec is not None and get_virtual_media_vsource(virtual_spec) is not None:
                loaded_items.append(filename)
                print(f"{log_prefix} Using virtual source: {filename}")
                continue
            source_name = virtual_spec.source_path if virtual_spec is not None else filename

            if os.path.isabs(source_name):
                source_path = source_name
            else:
                source_path = os.path.join(media_base_path, source_name)

            if not os.path.exists(source_path):
                print(f"{log_prefix} Warning: File not found for '{key}': {source_path}")
                continue

            if cache_dir:
                final_path = os.path.join(cache_dir, os.path.basename(source_name))
                try:
                    shutil.copy2(source_path, final_path)
                except Exception as e:
                    print(f"{log_prefix} Error copying {filename}: {e}")
                    continue
            else:
                final_path = source_path

            # Load images as PIL, keep videos/audio as paths
            if has_image_file_extension(final_path):
                try:
                    with Image.open(final_path) as loaded_image:
                        loaded_items.append(loaded_image.copy())
                    print(f"{log_prefix} Loaded image: {final_path}")
                except Exception as e:
                    print(f"{log_prefix} Error loading image {final_path}: {e}")
            else:
                loaded_items.append(replace_virtual_media_source(filename, final_path) if virtual_spec is not None else final_path)
                print(f"{log_prefix} Using path: {final_path}")

        # Update params, preserving list/single structure
        if loaded_items:
            if key == "image_refs" or is_originally_list:
                params[key] = loaded_items
            else:
                params[key] = loaded_items[0]
        else:
            params.pop(key, None)


def _build_runtime_task(task_id_val, params, plugin_data=None):
    """Build a runtime task dict from params."""
    primary_preview, secondary_preview, primary_labels, secondary_labels = get_preview_images(params)

    start_b64 = [pil_to_base64_uri(primary_preview[0], format="jpeg", quality=70)] if isinstance(primary_preview, list) and primary_preview else None
    end_b64 = [pil_to_base64_uri(secondary_preview[0], format="jpeg", quality=70)] if isinstance(secondary_preview, list) and secondary_preview else None

    return {
        "id": task_id_val,
        "params": params,
        "plugin_data": plugin_data or {},
        "repeats": params.get('repeat_generation', 1),
        "length": params.get('video_length'),
        "steps": params.get('num_inference_steps'),
        "prompt": params.get('prompt'),
        "start_image_labels": primary_labels,
        "end_image_labels": secondary_labels,
        "start_image_data": params.get("image_start") or params.get("image_refs"),
        "end_image_data": params.get("image_end"),
        "start_image_data_base64": start_b64,
        "end_image_data_base64": end_b64,
    }


def _extract_model_type(params, state, log_prefix="[load]"):
    base_model_type = params.get('base_model_type', None)
    model_type = original_model_type = params.get('model_type', base_model_type)

    if model_type is not None and get_model_def(model_type) is None:
        model_type = base_model_type

    if model_type is None:
        return None, "Settings must contain 'model_type'"
    params["model_type"] = model_type
    if get_model_def(model_type) is None:
        return None, f"Unknown model type: {original_model_type}"
    return model_type, None


def _parse_task_manifest(manifest, state, media_base_path, cache_dir=None, log_prefix="[load]", verbose_output = True):
    global task_id
    newly_loaded_queue = []
    first_error = None

    for task_index, task_data in enumerate(manifest):
        if task_data is None or not isinstance(task_data, dict):
            if first_error is None:
                first_error = f"Invalid task data at index {task_index}"
            print(f"{log_prefix} Skipping invalid task data at index {task_index}")
            continue

        params = task_data.get('params', {})
        task_id_loaded = task_data.get('id', task_id + 1)

        model_type, error = _extract_model_type(params, state, log_prefix)
        if error:
            if first_error is None:
                first_error = error
            print(f"{log_prefix} {error} for task #{task_id_loaded}. Skipping.")
            continue

        params['state'] = state

        if media_base_path is not None or _task_has_path_attachments(params):
            _load_task_attachments(params, media_base_path or os.path.dirname(os.path.abspath(__file__)), cache_dir, log_prefix)

        params, error = validate_task(task_data, state)
        if error:
            if first_error is None:
                first_error = error
            print(f"{log_prefix} {error} for task #{task_id_loaded}. Skipping.")
            continue

        # Build runtime task
        runtime_task = _build_runtime_task(task_id_loaded, params, task_data.get('plugin_data', {}))
        newly_loaded_queue.append(runtime_task)
        if verbose_output:
            print(f"{log_prefix} Task {task_index+1}/{len(manifest)} ready, ID: {task_id_loaded}, model: {model_type}")

    # Update global task_id
    if newly_loaded_queue:
        current_max_id = max([t['id'] for t in newly_loaded_queue if 'id' in t] + [0])
        if current_max_id >= task_id:
            task_id = current_max_id + 1

    return newly_loaded_queue, None if len(newly_loaded_queue) > 0 else first_error or "No valid task could be unpacked."


def _parse_queue_zip(filename, state):
    """Parse queue ZIP file. Returns (queue_list, error_msg or None)."""
    save_path_base = server_config.get("save_path", "outputs")
    cache_dir = os.path.join(save_path_base, "_loaded_queue_cache")

    try:
        print(f"[load_queue] Attempting to load queue from: {filename}")
        os.makedirs(cache_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(filename, 'r') as zf:
                if "queue.json" not in zf.namelist():
                    return None, "queue.json not found in zip file"
                print(f"[load_queue] Extracting to temp directory...")
                zf.extractall(tmpdir)

            manifest_path = os.path.join(tmpdir, "queue.json")
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            print(f"[load_queue] Loaded manifest with {len(manifest)} tasks.")

            return _parse_task_manifest(manifest, state, tmpdir, cache_dir, "[load_queue]")

    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def _parse_settings_json(filename, state):
    """Parse a single settings JSON file. Returns (queue_list, error_msg or None).

    Media paths in JSON are filesystem paths (absolute or relative to WanGP folder).
    """
    global task_id

    try:
        print(f"[load_settings] Loading settings from: {filename}")

        with open(filename, 'r', encoding='utf-8') as f:
            params = json.load(f)

        if isinstance(params, list):
            # Accept full queue manifests or a list of settings dicts
            if all(isinstance(item, dict) and "params" in item for item in params):
                manifest = params
            else:
                manifest = []
                for item in params:
                    if not isinstance(item, dict):
                        continue
                    task_id += 1
                    manifest.append({"id": task_id, "params": item, "plugin_data": {}})
        elif isinstance(params, dict):
            # Wrap as single-task manifest
            task_id += 1
            manifest = [{"id": task_id, "params": params, "plugin_data": {}}]
        else:
            return None, "Settings file must contain a JSON object or a list of tasks"

        # Media paths are relative to WanGP folder (no cache needed)
        wgp_folder = os.path.dirname(os.path.abspath(__file__))

        return _parse_task_manifest(manifest, state, wgp_folder, None, "[load_settings]")

    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def record_queue_error(state, queue, error, abort= False):
    gen = get_gen_info(state)
    queue_errors = gen.get("queue_errors", None)
    if queue_errors is None:
        gen["queue_errors"] = queue_errors = {}

    for i, task in enumerate(queue):
        params = task["params"]
        client_id= params.get("client_id", "")
        if len(client_id):
            queue_errors[client_id] = (error, abort, i>0)


def _normalize_inline_queue_priority(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "0", "false", "off", "no"}:
            return False
        if value in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def _pop_runtime_task_priority(task):
    if not isinstance(task, dict):
        return False
    params = task.get("params", None)
    raw_value = params.pop("priority", None) if isinstance(params, dict) else None
    if raw_value is None:
        raw_value = task.pop("priority", None)
    return _normalize_inline_queue_priority(raw_value)


def load_queue_action(filepath, state, evt:gr.EventData):
    """Load queue from ZIP or JSON file (Gradio UI wrapper)."""
    global task_id
    gen = get_gen_info(state)
    original_queue = gen.get("queue", [])

    # Determine filename (autoload vs user upload)
    delete_autoqueue_file = False
    filename = file_path = None
    verbose_output = True
    newly_loaded_queue = gen.pop("inline_queue", None)
    if newly_loaded_queue is not None:
        verbose_output = False
        inline_queue_source = newly_loaded_queue
        if isinstance(newly_loaded_queue, dict): 
            newly_loaded_queue = [ {"id": 0, "params": newly_loaded_queue}]
        else:
            inline_queue_source = newly_loaded_queue
        newly_loaded_queue, error = _parse_task_manifest(newly_loaded_queue, state, None, None, "[unpack queue]", verbose_output = verbose_output )
        if error:
            if isinstance(inline_queue_source, dict):
                inline_queue_source = [{"id": 0, "params": inline_queue_source}]
            record_queue_error(state, inline_queue_source or [], error)
            gr.Warning(f"Failed to unpack innline queue: {error[:200]}")
            return update_queue_data(original_queue)

    elif evt.target == None:
        # Autoload only works with empty queue
        if original_queue:
            return
        autoload_path = None
        if Path(AUTOSAVE_PATH).is_file():
            autoload_path = AUTOSAVE_PATH
            delete_autoqueue_file = True
        elif AUTOSAVE_TEMPLATE_PATH != AUTOSAVE_PATH and Path(AUTOSAVE_TEMPLATE_PATH).is_file():
            autoload_path = AUTOSAVE_TEMPLATE_PATH
        else:
            return
        print(f"Autoloading queue from {autoload_path}...")
        filename = autoload_path
    else:
        if not filepath or not hasattr(filepath, 'name') or not Path(filepath.name).is_file():
            print("[load_queue_action] Warning: No valid file selected or file not found.")
            return update_queue_data(original_queue)
        filename = filepath.name

    try:
        # Detect file type and use appropriate parser
        if filename is not None:
            if filename.lower().endswith('.json'):
                newly_loaded_queue, error = _parse_settings_json(filename, state)
                # Safety: clear attachment paths when loading JSON through UI
                # (JSON files contain filesystem paths which could be security-sensitive)
                if newly_loaded_queue:
                    for task in newly_loaded_queue:
                        params = task.get('params', {})
                        for key in ATTACHMENT_KEYS:
                            if key in params:
                                params[key] = None
            else:
                newly_loaded_queue, error = _parse_queue_zip(filename, state)
            if error:
                gr.Warning(f"Failed to load queue: {error[:200]}")
                return update_queue_data(original_queue)

        # Merge with existing queue: renumber task IDs to avoid conflicts
        # IMPORTANT: Modify list in-place to preserve references held by process_tasks
        if original_queue:
            # Find the highest existing task ID
            max_existing_id = max([t.get('id', 0) for t in original_queue] + [0])
            # Renumber newly loaded tasks
            for i, task in enumerate(newly_loaded_queue):
                task['id'] = max_existing_id + 1 + i
            priority_tasks = []
            regular_tasks = []
            for task in newly_loaded_queue:
                if _pop_runtime_task_priority(task):
                    priority_tasks.append(task)
                else:
                    regular_tasks.append(task)
            # Update global task_id counter
            task_id = max_existing_id + len(newly_loaded_queue) + 1
            with lock:
                if priority_tasks:
                    original_queue[1:1] = priority_tasks
                if regular_tasks:
                    original_queue.extend(regular_tasks)
                gen["queue"] = original_queue
            action_msg = f"Merged {len(newly_loaded_queue)} task(s) with existing {len(original_queue) - len(newly_loaded_queue)} task(s)"
            merged_queue = original_queue
        else:
            for task in newly_loaded_queue:
                _pop_runtime_task_priority(task)
            # No existing queue - assign newly loaded queue directly
            merged_queue = newly_loaded_queue
            action_msg = f"Loaded {len(newly_loaded_queue)} task(s)"
            with lock:
                gen["queue"] = merged_queue

        # Update state (Gradio-specific)
        with lock:
            gen["prompts_max"] = len(merged_queue)
        update_global_queue_ref(merged_queue)
        if verbose_output:
            print(f"[load_queue_action] {action_msg}.")
            gr.Info(action_msg)
        return update_queue_data(merged_queue)

    except Exception as e:
        error_message = f"Error during queue load: {e}"
        print(f"[load_queue_action] Caught error: {error_message}")
        traceback.print_exc()
        gr.Warning(f"Failed to load queue: {error_message[:200]}")
        return update_queue_data(original_queue)

    finally:
        if filename and delete_autoqueue_file:
            if os.path.isfile(filename):
                os.remove(filename)
                print(f"Clear Queue: Deleted autosave file '{filename}'.")

        if filepath and hasattr(filepath, 'name') and filepath.name and os.path.exists(filepath.name):
            if tempfile.gettempdir() in os.path.abspath(filepath.name):
                try:
                    os.remove(filepath.name)
                    print(f"[load_queue_action] Removed temporary upload file: {filepath.name}")
                except OSError as e:
                    print(f"[load_queue_action] Info: Could not remove temp file {filepath.name}: {e}")
            else:
                print(f"[load_queue_action] Info: Did not remove non-temporary file: {filepath.name}")


def clear_queue_action(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    queue = gen.get("queue", [])
    aborted_current = False
    cleared_pending = False

    with lock:
        if "in_progress" in gen and gen["in_progress"]:
            print("Clear Queue: Signalling abort for in-progress task.")
            gen["abort"] = True
            gen["extra_orders"] = 0
            if wan_model is not None:
                wan_model._interrupt = True
            aborted_current = True

        if queue:
             if len(queue) > 1 or (len(queue) == 1 and queue[0] is not None and queue[0].get('id') is not None):
                 print(f"Clear Queue: Clearing {len(queue)} tasks from queue.")
                 queue.clear()
                 cleared_pending = True
             else:
                 pass

        if aborted_current or cleared_pending:
            gen["prompts_max"] = 0

    if cleared_pending:
        try:
            if os.path.isfile(AUTOSAVE_PATH):
                os.remove(AUTOSAVE_PATH)
                print(f"Clear Queue: Deleted autosave file '{AUTOSAVE_PATH}'.")
        except OSError as e:
            print(f"Clear Queue: Error deleting autosave file '{AUTOSAVE_PATH}': {e}")
            gr.Warning(f"Could not delete the autosave file '{AUTOSAVE_PATH}'. You may need to remove it manually.")

    if aborted_current and cleared_pending:
        gr.Info("Queue cleared and current generation aborted.")
    elif aborted_current:
        gr.Info("Current generation aborted.")
    elif cleared_pending:
        gr.Info("Queue cleared.")
    else:
        gr.Info("Queue is already empty or only contains the active task (which wasn't aborted now).")

    return update_queue_data([])
def quit_application():
    print("Save and Quit requested...")
    clear_startup_lock()
    autosave_queue()
    import signal
    os.kill(os.getpid(), signal.SIGINT)

def start_quit_process():
    return 5, gr.update(visible=False), gr.update(visible=True)

def cancel_quit_process():
    return -1, gr.update(visible=True), gr.update(visible=False)

def show_countdown_info_from_state(current_value: int):
    if current_value > 0:
        gr.Info(f"Quitting in {current_value}...")
        return current_value - 1
    return current_value
quitting_app = False
def autosave_queue():
    global quitting_app
    quitting_app = True
    global global_queue_ref
    if not global_queue_ref:
        print("Autosave: Queue is empty, nothing to save.")
        return

    print(f"Autosaving queue ({len(global_queue_ref)} items) to {AUTOSAVE_PATH}...")
    try:
        if _save_queue_to_zip(global_queue_ref, AUTOSAVE_PATH):
            print(f"Queue autosaved successfully to {AUTOSAVE_PATH}")
        else:
            print("Autosave failed.")
    except Exception as e:
        print(f"Error during autosave: {e}")
        traceback.print_exc()

def finalize_generation_with_state(current_state):
     if not isinstance(current_state, dict) or 'gen' not in current_state:
         return (
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(interactive=True),
             gr.update(interactive=True),
             gr.update(visible=True),
             gr.update(visible=False),
             gr.update(visible=False),
             gr.update(visible=False, value=""),
             gr.update(),
             current_state,
         )

     gallery_tabs_update, current_gallery_tab_update, gallery_update, audio_files_paths_update, audio_file_selected_update, audio_gallery_refresh_trigger_update, abort_btn_update, earlystop_btn_update, gen_btn_update, add_queue_btn_update, current_gen_col_update, gen_info_update = finalize_generation(current_state)
     accordion_update = gr.Accordion(open=False) if len(get_gen_info(current_state).get("queue", [])) <= 1 else gr.update()
     return gallery_tabs_update, current_gallery_tab_update, gallery_update, audio_files_paths_update, audio_file_selected_update, audio_gallery_refresh_trigger_update, abort_btn_update, earlystop_btn_update, gen_btn_update, add_queue_btn_update, current_gen_col_update, gen_info_update, accordion_update, current_state

def generate_queue_html(queue):
    if len(queue) <= 1:
        return "<div style='text-align: center; color: grey; padding: 20px;'>Queue is empty.</div>"

    top_button_html = ""
    bottom_button_html = ""
    
    if len(queue) > 11:
        btn_style = "width: 100%; padding: 8px; margin-bottom: 2px; font-weight: bold; display: flex; justify-content: center; align-items: center;"
        
        top_button_html = f"""
        <div style="margin-bottom: 5px;">
            <button onclick="scrollToQueueTop()" 
                    ondragenter="scrollToQueueTop(); event.preventDefault();" 
                    ondragover="event.preventDefault();"
                    class="gr-button gr-button-secondary" 
                    style="{btn_style}">
                <img src="/gradio_api/file=icons/top.svg" alt="Top" style="width: 1.3em; height: 1.3em; margin-right: 6px;">
                Scroll to Top
            </button>
        </div>
        """
        
        bottom_button_html = f"""
        <div style="margin-top: 5px;">
            <button onclick="scrollToQueueBottom()" 
                    ondragenter="scrollToQueueBottom(); event.preventDefault();" 
                    ondragover="event.preventDefault();"
                    class="gr-button gr-button-secondary" 
                    style="{btn_style.replace('margin-bottom', 'margin-top')}">
                <img src="/gradio_api/file=icons/bottom.svg" alt="Bottom" style="width: 1.3em; height: 1.3em; margin-right: 6px;">
                Scroll to Bottom
            </button>
        </div>
        """

    table_header = """
    <table>
        <thead>
            <tr>
                <th style="width:5%;" class="center-align">Qty</th>
                <th style="width:auto;" class="text-left">Prompt</th>
                <th style="width:7%;" class="center-align">Length</th>
                <th style="width:7%;" class="center-align">Steps</th>
                <th style="width:10%;" class="center-align">Start/Ref</th>
                <th style="width:10%;" class="center-align">End</th>
                <th style="width:4%;" class="center-align" title="Edit"></th>
                <th style="width:4%;" class="center-align" title="Remove"></th>
            </tr>
        </thead>
        <tbody>
    """
    
    table_rows = []
    scheme = server_config.get("queue_color_scheme", "pastel")

    for i, item in enumerate(queue):
        if i == 0:
            continue
        
        row_index = i - 1
        task_id = item['id']
        full_prompt = html.escape(item['prompt'])
        truncated_prompt = (html.escape(item['prompt'][:97]) + '...') if len(item['prompt']) > 100 else full_prompt
        prompt_cell = f'<div class="prompt-cell" title="{full_prompt}">{truncated_prompt}</div>'
        
        start_img_data = item.get('start_image_data_base64') or [None]
        start_img_uri = start_img_data[0]
        start_img_labels = item.get('start_image_labels', [''])
        
        end_img_data = item.get('end_image_data_base64') or [None]
        end_img_uri = end_img_data[0]
        end_img_labels = item.get('end_image_labels', [''])
        
        num_steps = item.get('steps')
        length = item.get('length')
        
        start_img_md = ""
        if start_img_uri:
            start_img_md = f'<div class="hover-image" onclick="showImageModal(\'start_{row_index}\')"><img src="{start_img_uri}" alt="{start_img_labels[0]}" /></div>'
            
        end_img_md = ""
        if end_img_uri:
            end_img_md = f'<div class="hover-image" onclick="showImageModal(\'end_{row_index}\')"><img src="{end_img_uri}" alt="{end_img_labels[0]}" /></div>'

        edit_btn = f"""<button onclick="updateAndTrigger('edit_{task_id}')" class="action-button" title="Edit"><img src="/gradio_api/file=icons/edit.svg" style="width: 20px; height: 20px;"></button>"""
        remove_btn = f"""<button onclick="updateAndTrigger('remove_{task_id}')" class="action-button" title="Remove"><img src="/gradio_api/file=icons/remove.svg" style="width: 20px; height: 20px;"></button>"""

        row_class = "draggable-row"
        row_style = ""
        
        if scheme == "pastel":
            hue = (task_id * 137.508 + 22241) % 360
            row_class += " pastel-row"
            row_style = f'--item-hue: {hue:.0f};'
        else:
            row_class += " alternating-grey-row"
            if row_index % 2 == 0:
                row_class += " even-row"
                
        row_html = f"""
        <tr draggable="true" class="{row_class}" data-index="{row_index}" style="{row_style}" title="Drag to reorder">
            <td class="center-align">{item.get('repeats', "1")}</td>
            <td>{prompt_cell}</td>
            <td class="center-align">{length}</td>
            <td class="center-align">{num_steps}</td>
            <td class="center-align">{start_img_md}</td>
            <td class="center-align">{end_img_md}</td>
            <td class="center-align">{edit_btn}</td>
            <td class="center-align">{remove_btn}</td>
        </tr>
        """
        table_rows.append(row_html)
        
    table_footer = "</tbody></table>"
    table_html = table_header + "".join(table_rows) + table_footer
    scrollable_div = f'<div id="queue-scroll-container" style="max-height: 650px; overflow-y: auto;">{table_html}</div>'

    return top_button_html + scrollable_div + bottom_button_html

def update_queue_data(queue):
    update_global_queue_ref(queue)
    html_content = generate_queue_html(queue)
    return gr.HTML(value=html_content)


def create_html_progress_bar(percentage=0.0, text="Idle", is_idle=True):
    bar_class = "progress-bar-custom idle" if is_idle else "progress-bar-custom"
    bar_text_html = f'<div class="progress-bar-text">{text}</div>'

    html = f"""
    <div class="progress-container-custom">
        <div class="{bar_class}" style="width: {percentage:.1f}%;" role="progressbar" aria-valuenow="{percentage:.1f}" aria-valuemin="0" aria-valuemax="100">
           {bar_text_html}
        </div>
    </div>
    """
    return html

def update_generation_status(html_content):
    if(html_content):
        return gr.update(value=html_content)

family_handlers = ["models.wan.wan_handler", "models.wan.ovi_handler", "models.wan.df_handler", "models.hyvideo.hunyuan_handler", "models.ltx_video.ltxv_handler", "models.ltx2.ltx2_handler", "models.longcat.longcat_handler", "models.flux.flux_handler", "models.qwen.qwen_handler", "models.kandinsky5.kandinsky_handler",  "models.z_image.z_image_handler", "models.magi_human.magi_human_handler", "models.TTS.ace_step_handler", "models.TTS.chatterbox_handler", "models.TTS.qwen3_handler", "models.TTS.yue_handler", "models.TTS.heartmula_handler", "models.TTS.kugelaudio_handler", "models.TTS.index_tts2_handler"]
DEFAULT_LORA_ROOT = "loras"

def get_lora_root():
    cli_lora_root = getattr(args, "loras", "")
    if isinstance(cli_lora_root, str):
        cli_lora_root = cli_lora_root.strip()
    config_lora_root = None
    if "server_config" in globals():
        config_lora_root = server_config.get("loras_root", DEFAULT_LORA_ROOT)
    lora_root = cli_lora_root or config_lora_root or DEFAULT_LORA_ROOT
    return lora_root

def get_lora_dir(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception("loras unknown")

    handler = get_model_handler(model_type)
    get_dir = getattr(handler, "get_lora_dir", None)
    if get_dir is None:
        raise Exception("loras unknown")

    lora_root = get_lora_root()

    lora_dir = get_dir(base_model_type, args, lora_root)
    if lora_dir is None:
        raise Exception("loras unknown")
    if os.path.isfile(lora_dir):
        raise Exception(f"loras path '{lora_dir}' exists and is not a directory")
    if not os.path.isdir(lora_dir):
        os.makedirs(lora_dir, exist_ok=True)
    return lora_dir

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
args = parse_wgp_args(family_handlers, CONFIG_FILENAME, DEFAULT_LORA_ROOT)
migrate_loras_layout()

gpu_major, gpu_minor = torch.cuda.get_device_capability(args.gpu if len(args.gpu) > 0 else None)
if  gpu_major < 8:
    print("Switching to FP16 models when possible as GPU architecture doesn't support optimed BF16 Kernels")
    bfloat16_supported = False
else:
    bfloat16_supported = True

args.flow_reverse = True
processing_device = args.gpu
if len(processing_device) == 0:
    processing_device ="cuda"
# torch.backends.cuda.matmul.allow_fp16_accumulation = True
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

force_profile_no = float(args.profile)
verbose_level = int(args.verbose)
check_loras = args.check_loras ==1

with open("models/_settings.json", "r", encoding="utf-8") as f:
    primary_settings = json.load(f)

wgp_root = os.path.abspath(os.getcwd())
config_dir = args.config.strip()
server_config_filename = CONFIG_FILENAME
server_config_fallback = server_config_filename
if config_dir:
    config_dir = os.path.abspath(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    server_config_filename = os.path.join(config_dir, CONFIG_FILENAME)
    server_config_fallback = os.path.join(wgp_root, CONFIG_FILENAME)
    AUTOSAVE_PATH = os.path.join(config_dir, AUTOSAVE_FILENAME)
    AUTOSAVE_TEMPLATE_PATH = os.path.join(wgp_root, AUTOSAVE_FILENAME)
else:
    AUTOSAVE_PATH = AUTOSAVE_FILENAME
    AUTOSAVE_TEMPLATE_PATH = AUTOSAVE_FILENAME

if not os.path.isdir("settings"):
    os.mkdir("settings")
if os.path.isfile("t2v_settings.json"):
    for f in glob.glob(os.path.join(".", "*_settings.json*")):
        target_file = os.path.join("settings",  Path(f).parts[-1])
        shutil.move(f, target_file)

config_load_filename = server_config_filename
if config_dir and not Path(server_config_filename).is_file():
    if Path(server_config_fallback).is_file():
        config_load_filename = server_config_fallback

src_move = [ "ltx-2-19b-dev-fp4_diffusion_model.safetensors" ]
tgt_move = [ "ltx-2-19b-dev-nvfp4_diffusion_model.safetensors" ]
for src_name, tgt_name in zip(src_move, tgt_move):
    src = fl.locate_file(src_name, error_if_none=False)
    if src is not None:
        tgt = os.path.join(os.path.dirname(src), tgt_name)
        try:
            if os.path.isfile(tgt):
                os.remove(src)
            else:
                os.replace(src, tgt)
        except:
            pass
    

if not Path(config_load_filename).is_file():
    server_config = {
        "attention_mode" : "auto",  
        "transformer_types": [], 
        "transformer_quantization": "int8",
        "text_encoder_quantization" : "int8",
        "lm_decoder_engine": "",
        "save_path": "outputs",  
        "image_save_path": "outputs",  
        "compile" : "",
        "metadata_type": "metadata",
        "boost" : 1,
        "enable_int8_kernels": 1,
        "clear_file_list" : 5,
        "keep_intermediate_sliding_windows": 1,
        "enable_4k_resolutions": 0,
        "max_reserved_loras": -1,
        "vae_config": 0,
        "profile" : profile_type.LowRAM_LowVRAM,
        "video_profile": profile_type.LowRAM_LowVRAM,
        "image_profile": profile_type.LowRAM_LowVRAM,
        "audio_profile": 3.5,
        "preload_model_policy": [],
        "UI_theme": "default",
        "checkpoints_paths": fl.default_checkpoints_paths,
        "loras_root": DEFAULT_LORA_ROOT,
        "save_queue_if_crash": 1,
        "queue_color_scheme": "pastel",
        "process_queues_when_browser_unfocused": 1,
        "model_hierarchy_type": 1,
        "mmaudio_mode": 0,
        "mmaudio_persistence": 1,
        "rife_version": "v4",
        **get_deepy_default_runtime_config(),
        "prompt_enhancer_quantization": "quanto_int8",
        "prompt_enhancer_temperature": 0.6,
        "prompt_enhancer_top_p": 0.9,
        "prompt_enhancer_randomize_seed": True,
        "audio_save_path": "outputs",
    }

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config))
else:
    with open(config_load_filename, "r", encoding="utf-8") as reader:
        text = reader.read()
    server_config = json.loads(text)

server_config.setdefault("prompt_enhancer_quantization", "quanto_int8")
server_config.setdefault(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1)
gradio_queue_focus_patch.BACKGROUND_SCHEDULER_DEFAULT_ENABLED = bool(server_config.get(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1))
gradio_queue_focus_patch.install()

checkpoints_paths = server_config.get("checkpoints_paths", None)
if checkpoints_paths is None: checkpoints_paths = server_config["checkpoints_paths"] = fl.default_checkpoints_paths
# --- LOCAL-ONLY MODE: add roots parsed from docs/docs/models.txt ---
if server_config.get("use_models_txt", True):
    _wgp_root = os.path.dirname(os.path.abspath(__file__))
    extra_roots = fl.load_local_paths_from_file(os.path.join(_wgp_root, "docs", "docs", "models.txt"))
    expanded = [r for r in extra_roots if os.path.isdir(r)]
    if len(expanded) > 0:
        checkpoints_paths = expanded + list(checkpoints_paths)
        server_config["checkpoints_paths"] = checkpoints_paths
if server_config.get("local_only", False):
    fl.set_local_only(True)
# -------------------------------------------------------------------
fl.set_checkpoints_paths(checkpoints_paths)
three_levels_hierarchy = server_config.get("model_hierarchy_type", 1) == 1

MMAUDIO_MODE_OFF = 0
MMAUDIO_MODE_V2 = 1
MMAUDIO_MODE_NEW = 2
MMAUDIO_PERSIST_UNLOAD = 1
MMAUDIO_PERSIST_RAM = 2
MMAUDIO_STANDARD = "mmaudio_large_44k_v2.pth"
MMAUDIO_ALTERNATE = "mmaudio_large_44k_gold_8.5k_final_fp16.safetensors"

def _normalize_mmaudio_config(config):
    mode = config.get("mmaudio_mode", None)
    persistence = config.get("mmaudio_persistence", None)
    if mode is None:
        old = config.get("mmaudio_enabled", 0)
        mode = MMAUDIO_MODE_OFF if old == 0 else MMAUDIO_MODE_V2
    if persistence is None:
        old = config.get("mmaudio_enabled", 0)
        persistence = MMAUDIO_PERSIST_RAM if old == MMAUDIO_PERSIST_RAM else MMAUDIO_PERSIST_UNLOAD
    if mode not in (MMAUDIO_MODE_OFF, MMAUDIO_MODE_V2, MMAUDIO_MODE_NEW):
        mode = MMAUDIO_MODE_OFF
    if persistence not in (MMAUDIO_PERSIST_UNLOAD, MMAUDIO_PERSIST_RAM):
        persistence = MMAUDIO_PERSIST_UNLOAD
    config["mmaudio_mode"] = mode
    config["mmaudio_persistence"] = persistence
    return mode, persistence

def get_mmaudio_settings(config):
    mode, persistence = _normalize_mmaudio_config(config)
    enabled = mode != MMAUDIO_MODE_OFF
    if mode == MMAUDIO_MODE_V2:
        model_name = "large_44k_v2"
        model_path = MMAUDIO_STANDARD
    elif mode == MMAUDIO_MODE_NEW:
        model_name = "large_44k"
        model_path = MMAUDIO_ALTERNATE
    else:
        model_name = None
        model_path = None
    return enabled, mode, persistence, model_name, model_path

_normalize_mmaudio_config(server_config)

def _normalize_profile_defaults(config):
    if "profile" not in config:
        config["profile"] = profile_type.LowRAM_LowVRAM
    base_profile = config.get("profile", profile_type.LowRAM_LowVRAM)
    config.setdefault("video_profile", base_profile)
    config.setdefault("image_profile", base_profile)
    config.setdefault("audio_profile", 3.5)
    return config["video_profile"], config["image_profile"], config["audio_profile"]

def _normalize_output_paths(config):
    if "save_path" not in config:
        config["save_path"] = "outputs"
    if "image_save_path" not in config:
        config["image_save_path"] = config["save_path"]
    if "audio_save_path" not in config:
        config["audio_save_path"] = config["save_path"]

_normalize_profile_defaults(server_config)
_normalize_output_paths(server_config)
lm_decoder_engine = server_config.get("lm_decoder_engine", "")

from preprocessing.matanyone.utils.model_assets import migrate_matanyone_install, query_matanyone_download_def
migration_note = migrate_matanyone_install(server_config)
if migration_note:
    print(migration_note)

#   Deprecated models
for path in  ["wan2.1_Vace_1.3B_preview_bf16.safetensors", "sky_reels2_diffusion_forcing_1.3B_bf16.safetensors","sky_reels2_diffusion_forcing_720p_14B_bf16.safetensors",
"sky_reels2_diffusion_forcing_720p_14B_quanto_int8.safetensors", "sky_reels2_diffusion_forcing_720p_14B_quanto_fp16_int8.safetensors", "wan2.1_image2video_480p_14B_bf16.safetensors", "wan2.1_image2video_480p_14B_quanto_int8.safetensors",
"wan2.1_image2video_720p_14B_quanto_int8.safetensors", "wan2.1_image2video_720p_14B_quanto_fp16_int8.safetensors", "wan2.1_image2video_720p_14B_bf16.safetensors",
"wan2.1_text2video_14B_bf16.safetensors", "wan2.1_text2video_14B_quanto_int8.safetensors",
"wan2.1_Vace_14B_mbf16.safetensors", "wan2.1_Vace_14B_quanto_mbf16_int8.safetensors", "wan2.1_FLF2V_720p_14B_quanto_int8.safetensors", "wan2.1_FLF2V_720p_14B_bf16.safetensors",  "wan2.1_FLF2V_720p_14B_fp16.safetensors", "wan2.1_Vace_1.3B_mbf16.safetensors", "wan2.1_text2video_1.3B_bf16.safetensors",
"ltxv_0.9.7_13B_dev_bf16.safetensors", "ltx-2-19b-distilled-fp8.safetensors", "ltx-2-19b-dev-fp8.safetensors", "ltx-2-19b-distilled.safetensors", "ltx-2-19b-dev.safetensors"
]:
    if fl.locate_file(path, error_if_none= False) is not None:
        print(f"Removing old version of model '{path}'. A new version of this model will be downloaded next time you use it.")
        os.remove( fl.locate_file(path))

models_def = {}


# only needed for imported old settings files
model_signatures = {"t2v": "text2video_14B", "t2v_1.3B" : "text2video_1.3B",   "fun_inp_1.3B" : "Fun_InP_1.3B",  "fun_inp" :  "Fun_InP_14B", 
                    "i2v" : "image2video_480p", "i2v_720p" : "image2video_720p" , "vace_1.3B" : "Vace_1.3B", "vace_14B": "Vace_14B", "recam_1.3B": "recammaster_1.3B", 
                    "sky_df_1.3B" : "sky_reels2_diffusion_forcing_1.3B", "sky_df_14B" : "sky_reels2_diffusion_forcing_14B", 
                    "sky_df_720p_14B" : "sky_reels2_diffusion_forcing_720p_14B",
                    "phantom_1.3B" : "phantom_1.3B", "phantom_14B" : "phantom_14B", "ltxv_13B" : "ltxv_0.9.7_13B_dev", "ltxv_13B_distilled" : "ltxv_0.9.7_13B_distilled", 
                    "hunyuan" : "hunyuan_video_720", "hunyuan_i2v" : "hunyuan_video_i2v_720", "hunyuan_custom" : "hunyuan_video_custom_720", "hunyuan_custom_audio" : "hunyuan_video_custom_audio", "hunyuan_custom_edit" : "hunyuan_video_custom_edit",
                    "hunyuan_avatar" : "hunyuan_video_avatar"  }


def map_family_handlers(family_handlers):
    base_types_handlers, families_infos, models_eqv_map, models_comp_map = {}, {"unknown": (100, "Unknown")}, {}, {}
    for path in family_handlers:
        handler = importlib.import_module(path).family_handler
        for model_type in handler.query_supported_types():
            if model_type in base_types_handlers:
                prev = base_types_handlers[model_type].__name__
                raise Exception(f"Model type {model_type} supported by {prev} and {handler.__name__}")
            base_types_handlers[model_type] = handler
        families_infos.update(handler.query_family_infos())
        eq_map, comp_map = handler.query_family_maps()
        models_eqv_map.update(eq_map); models_comp_map.update(comp_map)
    return base_types_handlers, families_infos, models_eqv_map, models_comp_map

# models_eqv_map: bidirectional compatibility between base model types
# models_comp_map : mono directional compatibility between base model types {"A" : {"B", "C"} } means B & C model types can accept A  model types (but not the other way). Said otherwise B & C are derived data types

model_types_handlers, families_infos,  models_eqv_map, models_comp_map = map_family_handlers(family_handlers)

def get_base_model_type(model_type):
    model_def = get_model_def(model_type)
    if model_def == None:
        return model_type if model_type in model_types_handlers else None 
        # return model_type
    else:
        return model_def["architecture"]

def get_parent_model_type(model_type):
    base_model_type =  get_base_model_type(model_type)
    if base_model_type is None: return None
    model_def = get_model_def(base_model_type)
    return model_def.get("parent_model_type", base_model_type)
    
def get_model_handler(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception(f"Unknown model type {model_type}")
    model_handler = model_types_handlers.get(base_model_type, None)
    if model_handler is None:
        raise Exception(f"No model handler found for base model type {base_model_type}")
    return model_handler

def are_model_types_compatible(imported_model_type, current_model_type):
    imported_base_model_type = get_base_model_type(imported_model_type)
    curent_base_model_type = get_base_model_type(current_model_type)

    if imported_base_model_type in models_eqv_map:
        imported_base_model_type = models_eqv_map[imported_base_model_type]

    if curent_base_model_type in models_eqv_map:
        curent_base_model_type = models_eqv_map[curent_base_model_type]

    if imported_base_model_type == curent_base_model_type:
        return True

    comp_list=  models_comp_map.get(imported_base_model_type, None)
    if comp_list == None: return False
    return curent_base_model_type in comp_list 

def get_model_def(model_type):
    return models_def.get(model_type, None )


extra_settings.configure(get_model_def=get_model_def)


def get_model_type(model_filename):
    for model_type, signature in model_signatures.items():
        if signature in model_filename:
            return model_type
    return None
    # raise Exception("Unknown model:" + model_filename)

def get_model_family(model_type, for_ui = False):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        return "unknown"
    
    if for_ui : 
        model_def = get_model_def(model_type)
        model_family = model_def.get("group", None)
        if model_family is not None and model_family in families_infos:
            return model_family
    handler = model_types_handlers.get(base_model_type, None)
    if handler is None: 
        return "unknown"
    return handler.query_model_family()

def test_class_i2v(model_type):    
    model_def = get_model_def(model_type)
    return model_def.get("i2v_class", False)

def test_vace_module(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("vace_class", False)

def test_class_t2v(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("t2v_class", False)

def test_any_sliding_window(model_type):
    model_def = get_model_def(model_type)
    if model_def is None:
        return False
    return model_def.get("sliding_window", False)

def get_model_min_frames_and_step(model_type):
    model_def = get_model_def(model_type)
    frames_minimum = model_def.get("frames_minimum", 5)
    frames_steps = model_def.get("frames_steps", 4)
    latent_size = model_def.get("latent_size", frames_steps)
    return frames_minimum, frames_steps, latent_size 
    
def get_model_fps(model_type):
    model_def = get_model_def(model_type)
    fps= model_def.get("fps", 16)
    return fps

def get_computed_fps(force_fps, base_model_type , video_guide, video_source ):
    if force_fps == "auto":
        if video_source != None:
            fps,  _, _, _ = get_video_info(video_source)
        elif video_guide != None:
            fps,  _, _, _ = get_video_info(video_guide)
        else:
            fps = get_model_fps(base_model_type)
    elif force_fps == "control" and video_guide != None:
        fps,  _, _, _ = get_video_info(video_guide)
    elif force_fps == "source" and video_source != None:
        fps,  _, _, _ = get_video_info(video_source)
    elif len(force_fps) > 0 and is_integer(force_fps) :
        fps = int(force_fps)
    else:
        fps = get_model_fps(base_model_type)
    return fps

def get_model_name(model_type, description_container = [""]):
    model_def = get_model_def(model_type)
    if model_def == None: 
        return f"Unknown model {model_type}"
    model_name = model_def["name"]
    description = model_def["description"]
    description_container[0] = description
    return model_name

def get_model_record(model_name):
    return f"WanGP v{WanGP_version} by DeepBeepMeep - " +  model_name

def get_model_recursive_prop(model_type, prop = "URLs", sub_prop_name = None, return_list = True,  stack= []):
    model_def = models_def.get(model_type, None)
    if model_def != None: 
        prop_value = model_def.get(prop, None)
        if prop_value == None:
            return []
        if sub_prop_name is not None:
            if sub_prop_name == "_list":
                if not isinstance(prop_value,list) or len(prop_value) != 1:
                    raise Exception(f"Sub property value for property {prop} of model type {model_type} should be a list of size 1")
                prop_value = prop_value[0]
            else:
                if not isinstance(prop_value,dict) and not sub_prop_name in prop_value:
                    raise Exception(f"Invalid sub property value {sub_prop_name} for property {prop} of model type {model_type}")
                prop_value = prop_value[sub_prop_name]
        if isinstance(prop_value, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {prop} dependencies: {stack}")
            return get_model_recursive_prop(prop_value, prop = prop, sub_prop_name =sub_prop_name, stack = stack + [prop_value] )
        else:
            return prop_value
    else:
        if model_type in model_types:
            return [] if return_list else model_type 
        else:
            raise Exception(f"Unknown model type '{model_type}'")
        

def get_model_filename(model_type, quantization ="int8", dtype_policy = "", module_type = None, submodel_no = 1, URLs = None, stack=[]):
    if URLs is not None:
        pass
    elif module_type is not None:
        base_model_type = get_base_model_type(model_type) 
        # model_type_handler = model_types_handlers[base_model_type]
        # modules_files = model_type_handler.query_modules_files() if hasattr(model_type_handler, "query_modules_files") else {}
        if isinstance(module_type, list):
            URLs = module_type
        else:
            if "#" not in module_type:
                sub_prop_name = "_list"
            else:
                pos = module_type.rfind("#")
                sub_prop_name =  module_type[pos+1:]
                module_type = module_type[:pos]  
            URLs = get_model_recursive_prop(module_type, "modules", sub_prop_name =sub_prop_name, return_list= False)

        # choices = modules_files.get(module_type, None)
        # if choices == None: raise Exception(f"Invalid Module Id '{module_type}'")
    else:
        key_name = "URLs" if submodel_no  <= 1 else f"URLs{submodel_no}"

        model_def = models_def.get(model_type, None)
        if model_def == None: return ""
        URLs = model_def.get(key_name, [])
        if isinstance(URLs, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {key_name} dependencies: {stack}")
            return get_model_filename(URLs, quantization=quantization, dtype_policy=dtype_policy, submodel_no = submodel_no, stack = stack + [URLs])

    choices = URLs if isinstance(URLs, list) else [URLs]
    if len(choices) == 0:
        return ""
    if len(quantization) == 0:
        quantization = "bf16"

    dtype = get_transformer_dtype(model_type, dtype_policy)
    if len(choices) <= 1:
        raw_filename = choices[0]
    else:
        quant_tokens = []
        quant_order = []
        if quantization != "bf16":
            if quantization == "int8":
                quant_order =["int8", "fp8"]
            elif quantization == "fp8":
                quant_order =["fp8", "int8"]

        for quant_type in quant_order:
            quant_tokens += quant_router.get_quantization_tokens(quant_type) or []
        sub_choices = []
        for token in quant_tokens:
            sub_choices += [name for name in choices if token in os.path.basename(name).lower()]

        if len(sub_choices) > 0:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"
            new_sub_choices = [ name for name in sub_choices if dtype_str in os.path.basename(name) or dtype_str.upper() in os.path.basename(name)]
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 else sub_choices
            raw_filename = sub_choices[0]
        else:
            raw_filename = choices[0]

    return raw_filename

def get_transformer_dtype(model_type, transformer_dtype_policy):
    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(base_model_type)
    dtype = model_def.get("dtype", None)
    if dtype is not None: 
        return torch.float16 if dtype =="fp16" else torch.bfloat16
    model_family =  get_model_family(base_model_type) 
    if not isinstance(transformer_dtype_policy, str):
        return transformer_dtype_policy
    if len(transformer_dtype_policy) == 0:
        if not bfloat16_supported:
            return torch.float16
        else:
            if model_family == "wan"and False:
                return torch.float16
            else: 
                return torch.bfloat16
        return transformer_dtype
    elif transformer_dtype_policy =="fp16":
        return torch.float16
    else:
        return torch.bfloat16

def get_settings_file_name(model_type):
    return  os.path.join(args.settings, model_type + "_settings.json")

def fix_settings(model_type, ui_defaults, min_settings_version = 0):
    if model_type is None: return

    settings_version =  max(min_settings_version, ui_defaults.get("settings_version", 0))
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)

    prompts = ui_defaults.get("prompts", "")
    if len(prompts) > 0:
        ui_defaults["prompt"] = prompts
    image_prompt_type = ui_defaults.get("image_prompt_type", None)
    if image_prompt_type != None :
        if not isinstance(image_prompt_type, str):
            image_prompt_type = "S" if image_prompt_type  == 0 else "SE"
        if settings_version <= 2:
            image_prompt_type = image_prompt_type.replace("G","")
        ui_defaults["image_prompt_type"] = image_prompt_type

    if "alt_prompt" not in ui_defaults:
        ui_defaults["alt_prompt"] = ""

    if "lset_name" in ui_defaults: del ui_defaults["lset_name"]

    if settings_version < 2.54:
        renamed_settings = {
            "slg_switch": "perturbation_switch",
            "slg_layers": "perturbation_layers",
            "slg_start_perc": "perturbation_start_perc",
            "slg_end_perc": "perturbation_end_perc",
        }
        for old_name, new_name in renamed_settings.items():
            if old_name in ui_defaults:
                ui_defaults.setdefault(new_name, ui_defaults[old_name])
                del ui_defaults[old_name]

    if settings_version < 2.55:
        ui_defaults.setdefault("alt_scale", 0.0)

    if settings_version < 2.56:
        legacy_multi_prompts_mode = ui_defaults.get("multi_prompts_gen_type", None)
        if legacy_multi_prompts_mode is None:
            ui_defaults["multi_prompts_gen_type"] = "G"
        else:
            ui_defaults["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(legacy_multi_prompts_mode)
    else:
        ui_defaults["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(ui_defaults.get("multi_prompts_gen_type", "G"))

    audio_prompt_type = ui_defaults.get("audio_prompt_type", None)
    if settings_version < 2.2: 
        if audio_prompt_type == None :
            if any_audio_track(base_model_type):
                audio_prompt_type ="A"
                ui_defaults["audio_prompt_type"] = audio_prompt_type

    if settings_version < 2.35 and any_audio_track(base_model_type): 
        audio_prompt_type = audio_prompt_type or ""
        audio_prompt_type += "V"
        ui_defaults["audio_prompt_type"] = audio_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")

    if base_model_type in ["hunyuan"]:
        video_prompt_type = video_prompt_type.replace("I", "")

    if base_model_type in ["flux"] and settings_version < 2.23:
        video_prompt_type = video_prompt_type.replace("K", "").replace("I", "KI")

    remove_background_images_ref = ui_defaults.get("remove_background_images_ref", None)
    if settings_version < 2.22:
        if "I" in video_prompt_type:
            if remove_background_images_ref == 2:
                video_prompt_type = video_prompt_type.replace("I", "KI")
        if remove_background_images_ref != 0:
            remove_background_images_ref = 1
    if base_model_type in ["hunyuan_avatar"]: 
        remove_background_images_ref = 0
        if settings_version < 2.26:
            if not "K" in video_prompt_type: video_prompt_type = video_prompt_type.replace("I", "KI")
    if remove_background_images_ref is not None:
        ui_defaults["remove_background_images_ref"] = remove_background_images_ref

    ui_defaults["video_prompt_type"] = video_prompt_type

    tea_cache_setting = ui_defaults.get("tea_cache_setting", None)
    tea_cache_start_step_perc = ui_defaults.get("tea_cache_start_step_perc", None)

    if tea_cache_setting != None:
        del ui_defaults["tea_cache_setting"]
        if tea_cache_setting > 0:
            ui_defaults["skip_steps_multiplier"] = tea_cache_setting
            ui_defaults["skip_steps_cache_type"] = "tea"
        else:
            ui_defaults["skip_steps_multiplier"] = 1.75
            ui_defaults["skip_steps_cache_type"] = ""

    if tea_cache_start_step_perc != None:
        del ui_defaults["tea_cache_start_step_perc"]
        ui_defaults["skip_steps_start_step_perc"] = tea_cache_start_step_perc

    image_prompt_type = ui_defaults.get("image_prompt_type", "")
    if len(image_prompt_type) > 0:
        image_prompt_types_allowed = model_def.get("image_prompt_types_allowed","")
        image_prompt_type = filter_letters(image_prompt_type, image_prompt_types_allowed)
    ui_defaults["image_prompt_type"] = image_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")
    image_ref_choices_list = model_def.get("image_ref_choices", {}).get("choices", [])
    if model_def.get("guide_custom_choices", None) is  None:
        if len(image_ref_choices_list)==0:
            video_prompt_type = del_in_sequence(video_prompt_type, "IK")
        else:
            first_choice = image_ref_choices_list[0][1]
            if "I" in first_choice and not "I" in video_prompt_type: video_prompt_type += "I"
            if len(image_ref_choices_list)==1 and "K" in first_choice and not "K" in video_prompt_type: video_prompt_type += "K"
        ui_defaults["video_prompt_type"] = video_prompt_type

    model_handler = get_model_handler(base_model_type)
    if hasattr(model_handler, "fix_settings"):
            model_handler.fix_settings(base_model_type, settings_version, model_def, ui_defaults)

def get_default_settings(model_type):
    def get_default_prompt(i2v):
        if i2v:
            return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
        else:
            return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."
    i2v = test_class_i2v(model_type)
    defaults_filename = get_settings_file_name(model_type)
    if not Path(defaults_filename).is_file():
        model_def = get_model_def(model_type)
        base_model_type = get_base_model_type(model_type)

        ui_defaults = {
            "settings_version" : settings_version,
            "prompt": get_default_prompt(i2v),
            "resolution": "1280x720" if "720" in base_model_type else "832x480",
            "flow_shift": 7.0 if not "720" in base_model_type and i2v else 5.0, 
        }

        model_handler = get_model_handler(model_type)
        model_handler.update_default_settings(base_model_type, model_def, ui_defaults)

        ui_defaults_update = model_def.get("settings", None) 
        if ui_defaults_update is not None: ui_defaults.update(ui_defaults_update)

        if len(ui_defaults.get("prompt","")) == 0:
            ui_defaults["prompt"]= get_default_prompt(i2v)

        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(ui_defaults, f, indent=4)
    else:
        with open(defaults_filename, "r", encoding="utf-8") as f:
            ui_defaults = json.load(f)
        fix_settings(model_type, ui_defaults)            
    
    default_seed = args.seed
    if default_seed > -1:
        ui_defaults["seed"] = default_seed
    default_number_frames = args.frames
    if default_number_frames > 0:
        ui_defaults["video_length"] = default_number_frames
    default_number_steps = args.steps
    if default_number_steps > 0:
        ui_defaults["num_inference_steps"] = default_number_steps
    return ui_defaults


def init_model_def(model_type, model_def):
    base_model_type = get_base_model_type(model_type)
    family_handler = model_types_handlers.get(base_model_type, None)
    if family_handler is None:
        if model_def.get("visible", True):
            print(f"Skipping model type '{model_type}' with unsupported architecture '{base_model_type}'.")
        model_def["visible"] = False
        return model_def
    default_model_def = family_handler.query_model_def(base_model_type, model_def)
    if default_model_def is None: return model_def
    default_model_def.update(model_def)
    return default_model_def


def refresh_model_defs():
    global models_def, model_types, displayed_model_types
    models_def = {}
    displayed_model_types = []
    model_types = []
    models_def_paths =  glob.glob( os.path.join("defaults", "*.json") ) 
    defaults_paths = models_def_paths.copy()
    models_def_paths += glob.glob( os.path.join("finetunes", "*.json") ) 
    models_def_paths.sort()
    for file_path in models_def_paths:
        model_type = os.path.basename(file_path)[:-5]
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_def = json.load(f)
            except Exception as e:
                if file_path in defaults_paths:
                    raise Exception(f"Error while parsing Model Definition File '{file_path}': {str(e)}")
                else:
                    print(f"Finetune Definition File '{file_path}' will be ignored as there was an error in its parsing: {str(e)}")
                    continue
        model_def = json_def["model"]
        model_def["path"] = file_path
        del json_def["model"]      
        settings = json_def   
        existing_model_def = models_def.get(model_type, None) 
        if existing_model_def is not None:
            existing_settings = models_def.get("settings", None)
            if existing_settings != None:
                existing_settings.update(settings)
            existing_model_def.update(model_def)
        else:
            models_def[model_type] = model_def # partial def
            model_def= init_model_def(model_type, model_def)
            models_def[model_type] = model_def # replace with full def
            model_def["settings"] = settings

    model_types = models_def.keys()
    displayed_model_types= []
    for model_type in model_types:
        model_def = get_model_def(model_type)
        if not model_def is None and model_def.get("visible", True): 
            displayed_model_types.append(model_type)

refresh_model_defs()

transformer_types = server_config.get("transformer_types", [])
new_transformer_types = []
for model_type in transformer_types:
    if get_model_def(model_type) == None:
        print(f"Model '{model_type}' is missing. Either install it in the finetune folder or remove this model from ley 'transformer_types' in {CONFIG_FILENAME}")
    else:
        new_transformer_types.append(model_type)
transformer_types = new_transformer_types
transformer_type = server_config.get("last_model_type", None)
advanced = server_config.get("last_advanced_choice", False)
last_resolution = server_config.get("last_resolution_choice", None)
if args.advanced: advanced = True 

if transformer_type != None and not transformer_type in model_types and not transformer_type in models_def: transformer_type = None
if transformer_type == None:
    transformer_type = transformer_types[0] if len(transformer_types) > 0 else "t2v"

transformer_quantization =server_config.get("transformer_quantization", "int8")

transformer_dtype_policy = server_config.get("transformer_dtype_policy", "")
if args.fp16:
    transformer_dtype_policy = "fp16" 
if args.bf16:
    transformer_dtype_policy = "bf16" 
text_encoder_quantization =server_config.get("text_encoder_quantization", "int8")
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

default_profile_video = force_profile_no if force_profile_no >= 0 else server_config["video_profile"]
default_profile_image = force_profile_no if force_profile_no >= 0 else server_config["image_profile"]
default_profile_audio = force_profile_no if force_profile_no >= 0 else server_config["audio_profile"]
default_profile = default_profile_video
loaded_profile = force_profile_no = -1
compile = server_config.get("compile", "")
boost = server_config.get("boost", 1)
enable_int8_kernels = server_config.get("enable_int8_kernels", 1)
apply_int8_kernel_setting(enable_int8_kernels)
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

reload_needed = False
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "outputs"))
image_save_path = server_config.get("image_save_path", os.path.join(os.getcwd(), "outputs"))
audio_save_path = server_config.get("audio_save_path", save_path)
if not "video_output_codec" in server_config: server_config["video_output_codec"]= "libx264_8"
if not "hdr_video_crf" in server_config: server_config["hdr_video_crf"] = 8
if not "video_container" in server_config: server_config["video_container"]= "mp4"
if not "embed_source_images" in server_config: server_config["embed_source_images"]= False
if not "enable_4k_resolutions" in server_config: server_config["enable_4k_resolutions"]= 0
if not "max_reserved_loras" in server_config: server_config["max_reserved_loras"]= -1
if not "image_output_codec" in server_config: server_config["image_output_codec"]= "jpeg_95"
if not "audio_output_codec" in server_config: server_config["audio_output_codec"]= "aac_128"
if not "audio_stand_alone_output_codec" in server_config: server_config["audio_stand_alone_output_codec"]= "wav"
if not "rife_version" in server_config: server_config["rife_version"] = "v4"
if "loras_root" not in server_config: server_config["loras_root"] = DEFAULT_LORA_ROOT
if "save_queue_if_crash" not in server_config: server_config["save_queue_if_crash"] = 1
if "keep_intermediate_sliding_windows" not in server_config: server_config["keep_intermediate_sliding_windows"] = 1
if "prompt_enhancer_temperature" not in server_config: server_config["prompt_enhancer_temperature"] = 0.6
if "prompt_enhancer_top_p" not in server_config: server_config["prompt_enhancer_top_p"] = 0.9
if "prompt_enhancer_randomize_seed" not in server_config: server_config["prompt_enhancer_randomize_seed"] = True
set_deepy_runtime_config(server_config, server_config_filename)
if "enable_int8_kernels" not in server_config: server_config["enable_int8_kernels"] = 0

preload_model_policy = server_config.get("preload_model_policy", []) 


if args.t2v_14B or args.t2v: 
    transformer_type = "t2v"

if args.i2v_14B or args.i2v: 
    transformer_type = "i2v"

if args.t2v_1_3B:
    transformer_type = "t2v_1.3B"

if args.i2v_1_3B:
    transformer_type = "fun_inp_1.3B"

if args.vace_1_3B: 
    transformer_type = "vace_1.3B"

only_allow_edit_in_advanced = False
lora_preselected_preset = args.lora_preset
lora_preset_model = transformer_type

if  args.compile: #args.fastest or
    compile="transformer"
    lock_ui_compile = True


def save_model(model, model_type, dtype,  config_file,  submodel_no = 1,  is_module = False, filter = None, no_fp16_main_model = True, module_source_no = 1):
    model_def = get_model_def(model_type)
    # To save module and quantized modules
    # 1) set Transformer Model Quantization Type to 16 bits
    # 2) insert in def module_source : path and "model_fp16.safetensors in URLs"
    # 3) Generate (only quantized fp16 will be created)
    # 4) replace in def module_source : path and "model_bf16.safetensors in URLs"
    # 5) Generate (both bf16 and quantized bf16 will be created)
    if model_def == None: return
    if is_module:
        url_key = "modules"
        source_key = "module_source" if module_source_no <=1 else "module_source2"
    else:
        url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
        source_key = "source" if submodel_no <=1 else "source2"
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to save model for a finetune that references external files")
        return
    from mmgp import offload    
    dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if no_fp16_main_model: dtypestr = dtypestr.replace("fp16", "bf16")
    model_filename = None
    if is_module:
        if not isinstance(URLs,list) or len(URLs) != 1:
            print("Target Module files are missing")
            return 
        URLs= URLs[0]
    if isinstance(URLs, dict):
        url_dict_key = "URLs" if module_source_no ==1 else "URLs2"
        URLs = URLs[url_dict_key]
    for url in URLs:
        if "quanto" not in url and dtypestr in url:
            model_filename = os.path.basename(url)
            break
    if model_filename is None:
        print(f"No target filename with bf16 or fp16 in its name is mentioned in {url_key}")
        return

    finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
    with open(finetune_file, 'r', encoding='utf-8') as reader:
        saved_finetune_def = json.load(reader)

    update_model_def = False
    model_filename_path = os.path.join(fl.get_download_location(), model_filename)
    quanto_dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if ("m" + dtypestr) in model_filename: 
        dtypestr = "m" + dtypestr 
        quanto_dtypestr = "m" + quanto_dtypestr 
    if fl.locate_file(model_filename, error_if_none= False) is None and (not no_fp16_main_model or dtype == torch.bfloat16):
        offload.save_model(model, model_filename_path, config_file_path=config_file, filter_sd=filter)
        print(f"New model file '{model_filename}' had been created for finetune Id '{model_type}'.")
        del saved_finetune_def["model"][source_key]
        del model_def[source_key]
        print(f"The 'source' entry has been removed in the '{finetune_file}' definition file.")
        update_model_def = True

    if is_module:
        quanto_filename = model_filename.replace(dtypestr, "quanto_" + quanto_dtypestr + "_int8" )
        quanto_filename_path = os.path.join(fl.get_download_location() , quanto_filename)
        if hasattr(model, "_quanto_map"):
            print("unable to generate quantized module, the main model should at full 16 bits before quantization can be done")
        elif fl.locate_file(quanto_filename, error_if_none= False) is None:
            offload.save_model(model, quanto_filename_path, config_file_path=config_file, do_quantize= True, filter_sd=filter)
            print(f"New quantized file '{quanto_filename}' had been created for finetune Id '{model_type}'.")
            if isinstance(model_def[url_key][0],dict): 
                model_def[url_key][0][url_dict_key].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0][url_dict_key].append(quanto_filename)
            else: 
                model_def[url_key][0].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0].append(quanto_filename)
            update_model_def = True
    if update_model_def:
        with open(finetune_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(saved_finetune_def, indent=4))

def save_quantized_model(model, model_type, model_filename, dtype,  config_file, submodel_no = 1):
    if "quanto" in model_filename: return
    model_def = get_model_def(model_type)
    if model_def == None: return
    url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to create a quantized model for a finetune that references external files")
        return
    from mmgp import offload
    if dtype == torch.bfloat16:
         model_filename =  model_filename.replace("fp16", "bf16").replace("FP16", "bf16")
    elif dtype == torch.float16:
         model_filename =  model_filename.replace("bf16", "fp16").replace("BF16", "bf16")

    for rep in ["mfp16", "fp16", "mbf16", "bf16"]:
        if "_" + rep in model_filename:
            model_filename = model_filename.replace("_" + rep, "_quanto_" + rep + "_int8")
            break
    if not "quanto" in model_filename:
        pos = model_filename.rfind(".")
        model_filename =  model_filename[:pos] + "_quanto_int8" + model_filename[pos:] 

    model_filename = os.path.basename(model_filename)
    if fl.locate_file(model_filename, error_if_none= False) is not None:
        print(f"There isn't any model to quantize as quantized model '{model_filename}' aready exists")
    else:
        model_filename_path = os.path.join(fl.get_download_location(), model_filename)
        offload.save_model(model, model_filename_path, do_quantize= True, config_file_path=config_file)
        print(f"New quantized file '{model_filename}' had been created for finetune Id '{model_type}'.")
        if not model_filename in URLs:
            URLs.append(model_filename)
            finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
            with open(finetune_file, 'r', encoding='utf-8') as reader:
                saved_finetune_def = json.load(reader)
            saved_finetune_def["model"][url_key] = URLs
            with open(finetune_file, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(saved_finetune_def, indent=4))
            print(f"The '{finetune_file}' definition file has been automatically updated with the local path to the new quantized model.")


def get_loras_preprocessor(transformer, model_type):
    preprocessor =  getattr(transformer, "preprocess_loras", None)
    if preprocessor == None:
        return None
    
    def preprocessor_wrapper(sd):
        return preprocessor(model_type, sd)

    return preprocessor_wrapper    


def process_files_def(repoId = None, sourceFolderList = None, fileList = None, targetFolderList = None):
    if fl.is_local_only():
        print(f"[LOCAL-ONLY] Downloads deaktiviert - ueberspringe HF-Repo '{repoId}'. Nur lokale Modelle werden verwendet.")
        return
    if targetFolderList is None:
        targetFolderList = [None] * len(sourceFolderList)
    for targetFolder, sourceFolder, files in zip(targetFolderList, sourceFolderList,fileList ):
        if targetFolder is not None and len(targetFolder) == 0:  targetFolder = None
        explicit_target = targetFolder if targetFolder is not None else (sourceFolder if len(sourceFolder) > 0 else None)
        targetRoot = fl.get_smart_download_root(explicit_target)
        local_dir = os.path.join(targetRoot, targetFolder) if targetFolder is not None else targetRoot
        if len(files)==0:
            if fl.locate_folder(sourceFolder if targetFolder is None else os.path.join(targetFolder, sourceFolder), error_if_none= False ) is None:
                snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= local_dir)
        else:
            for onefile in files:     
                if len(sourceFolder) > 0: 
                    if fl.locate_file( (sourceFolder + "/" + onefile)  if targetFolder is None else os.path.join(targetFolder, sourceFolder, onefile), error_if_none= False) is None:   
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = local_dir, subfolder=sourceFolder)
                else:
                    if fl.locate_file(onefile if targetFolder is None else os.path.join(targetFolder, onefile), error_if_none= False) is None:          
                        hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = local_dir)


def download_mmaudio():
    mmaudio_enabled, mmaudio_mode, _, _, _ = get_mmaudio_settings(server_config)
    if mmaudio_enabled:
        mmaudio_files = ["synchformer_state_dict.pth", "v1-44.pth", MMAUDIO_STANDARD if mmaudio_mode == MMAUDIO_MODE_V2 else MMAUDIO_ALTERNATE]
        bigvgan_v2_files = ["config.json", "bigvgan_generator.pt"]
        enhancer_def = {
            "repoId" : "DeepBeepMeep/Wan2.1",
            "sourceFolderList" : [ "mmaudio", "DFN5B-CLIP-ViT-H-14-378", "bigvgan_v2_44khz_128band_512x"  ],
            "fileList" : [ mmaudio_files, ["open_clip_config.json", "open_clip_pytorch_model.bin"], bigvgan_v2_files]
        }
        process_files_def(**enhancer_def)


def download_file(url,filename):
    if fl.is_local_only():
        raise Exception(f"[LOCAL-ONLY] Download deaktiviert fuer '{url}'. Bitte lege die Datei manuell in deinen lokalen Modellordern ab (siehe docs/docs/models.txt).")
    url = url.split("|")[0]
    if url.startswith("https://huggingface.co/") and "/resolve/main/" in url:
        base_dir = os.path.dirname(filename)
        url = url[len("https://huggingface.co/"):]
        url_parts = url.split("/resolve/main/")
        repoId = url_parts[0]
        onefile = os.path.basename(url_parts[-1])
        sourceFolder = os.path.dirname(url_parts[-1])
        if len(sourceFolder) == 0:
            hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = fl.get_download_location() if len(base_dir)==0 else base_dir)
        else:
            temp_dir_path = os.path.join(fl.get_download_location(), "temp")
            target_path = os.path.join(temp_dir_path, sourceFolder)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = temp_dir_path, subfolder=sourceFolder)
            shutil.move(os.path.join( target_path, onefile), fl.get_download_location() if len(base_dir)==0 else base_dir)
            shutil.rmtree(temp_dir_path)
    else:
        from urllib.request import urlretrieve
        from shared.utils.download import create_progress_hook
        urlretrieve(url,filename, create_progress_hook(filename))

RIFE_V4_FILENAME = "rife4.26.pkl"
RIFE_V3_FILENAME = "flownet.pkl"
download_shared_done = False
def download_models(model_filename = None, model_type= None, file_type = 0, submodel_no = 1, force_path = None):
    def computeList(filename):
        if filename == None:
            return []
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        


    if file_type == 0:
        depth_variant = server_config.get("depth_anything_v2_variant", "vitl")
        depth_file = {"vitb": "depth_anything_v2_vitb.pth", "da3_metric_large": "depth_anything_v3_metric_large_bf16.safetensors"}.get(depth_variant, "depth_anything_v2_vitl.pth")
        shared_def = {
            "repoId" : "DeepBeepMeep/Wan2.1",
            "sourceFolderList" : [ "pose", "scribble", "flow", "depth", "wav2vec", "chinese-wav2vec2-base", "roformer", "pyannote", "det_align", "" ],
            "fileList" : [ ["dw-ll_ucoco_384.onnx", "yolox_l.onnx"],["netG_A_latest.pth"],  ["raft-things.pth"], 
                        [depth_file],
                        ["config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
                        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
                        ["model_bs_roformer_ep_317_sdr_12.9755.ckpt", "model_bs_roformer_ep_317_sdr_12.9755.yaml", "download_checks.json"],
                        ["pyannote_model_wespeaker-voxceleb-resnet34-LM.bin", "pytorch_model_segmentation-3.0.bin"], ["detface.pt"], [ RIFE_V3_FILENAME if server_config.get("rife_version", "v3") == "v3" else RIFE_V4_FILENAME  ] ]
        }
        process_files_def(**shared_def)
        process_files_def(**query_matanyone_download_def(server_config))


        enhancer_enabled = int(server_config.get("enhancer_enabled", 0) or 0)
        if enhancer_enabled > 0:
            from shared.prompt_enhancer import ensure_prompt_enhancer_assets

            ensure_prompt_enhancer_assets(
                process_files_def,
                enhancer_enabled=enhancer_enabled,
                qwen_backend=server_config.get("prompt_enhancer_quantization", "quanto_int8"),
            )

        download_mmaudio()
        global download_shared_done
        download_shared_done = True

    if model_filename is None: return

    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    
    any_source = ("source2" if submodel_no ==2 else "source") in model_def
    any_module_source = ("module_source2" if submodel_no ==2 else "module_source") in model_def 
    model_type_handler = model_types_handlers[base_model_type]
 
    if not (any_source and file_type==0 or any_module_source and file_type==1):
        local_model_filename = fl.get_local_model_filename(model_filename, extra_paths= force_path)
        if local_model_filename is None and len(model_filename) > 0:
            local_model_filename = fl.get_smart_download_location(os.path.basename(model_filename), force_path= force_path)
            url = model_filename

            if not url.startswith("http"):
                raise Exception(f"Model '{model_filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, local_model_filename)
            except Exception as e:
                if os.path.isfile(local_model_filename): os.remove(local_model_filename) 
                raise Exception(f"'{url}' is invalid for Model '{model_type}' : {str(e)}'")
            if file_type!=0: return
    lora_dir = get_lora_dir(model_type) 
    for prop, recursive in zip(["preload_URLs", "VAE_URLs"], [True, False]):
        if recursive:
            preload_URLs = get_model_recursive_prop(model_type, prop, return_list= True)
        else:
            preload_URLs = model_def.get(prop, [])
            if isinstance(preload_URLs, str): preload_URLs = [preload_URLs]

        for url in preload_URLs:
            filename = fl.get_local_model_filename(url, lora_dir = lora_dir)
            if filename is None: 
                filename = fl.get_download_location(url, lora_dir = lora_dir)
                if not url.startswith("http"):
                    raise Exception(f"{prop} '{filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
                try:
                    download_file(url, filename)
                except Exception as e:
                    if os.path.isfile(filename): os.remove(filename) 
                    raise Exception(f"{prop} '{url}' is invalid: {str(e)}'")

    model_loras = get_model_recursive_prop(model_type, "loras", return_list= True)
    for url in model_loras:
        filename = os.path.join(lora_dir, url.split("/")[-1])
        if not os.path.isfile(filename ): 
            if not url.startswith("http"):
                raise Exception(f"Lora '{filename}' was not found in the Loras Folder and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, filename)
            except Exception as e:
                if os.path.isfile(filename): os.remove(filename) 
                raise Exception(f"Lora URL '{url}' is invalid: {str(e)}'")
            
    if file_type != 0: return            
    model_files = model_type_handler.query_model_files(computeList, base_model_type, model_def)
    if not isinstance(model_files, list): model_files = [model_files]
    for one_repo in model_files:
        process_files_def(**one_repo)

offload.default_verboseLevel = verbose_level



def check_loras_exist(model_type, loras_choices_files, download = False, send_cmd = None):
    _ensure_loras_url_cache()
    lora_dir = get_lora_dir(model_type)
    missing_local_loras = []
    missing_remote_loras = []
    for lora_file in loras_choices_files:
        local_path = os.path.join(lora_dir, os.path.basename(lora_file))
        if not os.path.isfile(local_path):
            url = loras_url_cache.get(local_path, None)         
            if url is not None:
                if download:
                    if send_cmd is not None:
                        send_cmd("status", f'Downloading Lora {os.path.basename(lora_file)}...')
                    try:
                        download_file(url, local_path)
                    except:
                        missing_remote_loras.append(lora_file)
            else:
                missing_local_loras.append(lora_file)

    error = ""
    if len(missing_local_loras) > 0:
        error += f"The following Loras files are missing or invalid: {missing_local_loras}."
    if len(missing_remote_loras) > 0:
        error += f"The following Loras files could not be downloaded: {missing_remote_loras}."
    
    return error

def extract_preset(model_type, lset_name, loras):
    loras_choices = []
    loras_choices_files = []
    loras_mult_choices = ""
    prompt =""
    full_prompt =""
    lset_name = sanitize_file_name(lset_name)
    lora_dir = get_lora_dir(model_type)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 
    error = ""
    if not os.path.isfile(lset_name_filename):
        error = f"Preset '{lset_name}' not found "
    else:

        with open(lset_name_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        lset = json.loads(text)

        loras_choices = lset["loras"]
        loras_mult_choices = lset["loras_mult"]
        prompt = lset.get("prompt", "")
        full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt, full_prompt, error


def setup_loras(model_type, transformer,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []
    default_lora_preset = ""
    default_lora_preset_prompt = ""

    from pathlib import Path
    base_model_type = get_base_model_type(model_type)
    lora_dir = get_lora_dir(base_model_type)
    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")


    if lora_dir != None:
        dir_loras =  glob.glob( os.path.join(lora_dir , "*.sft") ) + glob.glob( os.path.join(lora_dir , "*.safetensors") ) 
        dir_loras.sort()
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets_settings = glob.glob( os.path.join(lora_dir , "*.json") ) 
        dir_presets_settings.sort()
        dir_presets =   glob.glob( os.path.join(lora_dir , "*.lset") ) 
        dir_presets.sort()
        # loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets_settings + dir_presets]
        loras_presets = [ Path(file_path).parts[-1] for file_path in dir_presets_settings + dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=get_loras_preprocessor(transformer, base_model_type), split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras = [ os.path.basename(lora) for lora in loras  ]

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, _ , error = extract_preset(base_model_type, default_lora_preset, loras)
        if len(error) > 0:
            print(error[:200])
    return loras, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset

def get_transformer_model(model, submodel_no = 1):
    if submodel_no > 1:
        model_key = f"model{submodel_no}"
        if not hasattr(model, model_key): return None

    if hasattr(model, "model"):
        if submodel_no > 1:
            return getattr(model, f"model{submodel_no}")
        else:
            return model.model
    elif hasattr(model, "transformer"):
        return model.transformer
    else:
        raise Exception("no transformer found")

def _normalize_output_type(output_type):
    if output_type is None:
        return "video"
    output_type = str(output_type).lower()
    if output_type not in ("video", "image", "audio"):
        return "video"
    return output_type

def get_default_profile(output_type):
    if force_profile_no >= 0:
        return force_profile_no
    output_type = _normalize_output_type(output_type)
    if output_type == "image":
        return default_profile_image
    if output_type == "audio":
        return default_profile_audio
    return default_profile_video

def compute_profile(override_profile, output_type="video"):
    return override_profile if override_profile != -1 else get_default_profile(output_type)

def get_output_type_for_model(model_type, image_mode=0):
    model_def = get_model_def(model_type)
    if model_def is not None and model_def.get("audio_only", False):
        return "audio"
    if image_mode and image_mode > 0:
        return "image"
    return "video"

def init_pipe(pipe, kwargs, profile):
    preload =int(args.preload)
    if preload == 0:
        preload = server_config.get("preload_in_VRAM", 0)

    kwargs["extraModelsToQuantize"]=  None
    source_budgets = kwargs.get("budgets", None)
    if source_budgets is None:  kwargs["budgets"] = source_budgets = {}
    mmgp_profile = int(profile)
    if mmgp_profile in (2, 4, 5):
        default_transformer_budget = default_transformer2_budget= kwargs.get("budgets", 100) 
        if isinstance(default_transformer_budget, dict):
            default_transformer_budget = default_transformer_budget.get("transformer", 100) 
            default_transformer2_budget = default_transformer2_budget.get("transformer2", 100) 

        budgets = { "transformer" : default_transformer_budget if preload  == 0 else preload, "text_encoder" : 100 if preload  == 0 else preload, "*" : max(1000 if profile==5 else 3000 , preload) }
        if "transformer2" in pipe:
            budgets["transformer2"] = default_transformer2_budget if preload  == 0 else preload
        source_budgets.update(budgets)
    elif mmgp_profile == 3:
        source_budgets.update({ "*" : "70%" })

    if "transformer2" in pipe:
        if profile in [3,4]:
            kwargs["pinnedMemory"] = ["transformer", "transformer2"]
    
    if profile == 4.5:
        kwargs["asyncTransfers"] = False
    elif profile == 3.5:
        kwargs["pinnedMemory"] = False

    return mmgp_profile

reset_prompt_enhancer_requested = False
def unload_prompt_enhancer_runtime():
    deepy_controller._unload_prompt_enhancer_runtime(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model)


def reset_prompt_enhancer():
    global reset_prompt_enhancer_requested
    reset_prompt_enhancer_requested = True

def reset_prompt_enhancer_if_requested():
    global reset_prompt_enhancer_requested, prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer, enhancer_offloadobj
    if not reset_prompt_enhancer_requested:
        return
    reset_prompt_enhancer_requested = False
    unload_prompt_enhancer_runtime()
    prompt_enhancer_image_caption_model = None
    prompt_enhancer_image_caption_processor = None
    prompt_enhancer_llm_model = None
    prompt_enhancer_llm_tokenizer = None
    if enhancer_offloadobj is not None:
        enhancer_offloadobj.release()
        enhancer_offloadobj = None

def setup_prompt_enhancer(pipe, kwargs):
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    model_no = server_config.get("enhancer_enabled", 0) 
    if model_no != 0:
        from shared.prompt_enhancer import load_prompt_enhancer_runtime

        runtime = load_prompt_enhancer_runtime(
            process_files_def,
            enhancer_enabled=model_no,
            lm_decoder_engine=server_config.get("lm_decoder_engine", ""),
            qwen_backend=server_config.get("prompt_enhancer_quantization", "quanto_int8"),
        )
        prompt_enhancer_image_caption_model = runtime.image_caption_model
        prompt_enhancer_image_caption_processor = runtime.image_caption_processor
        prompt_enhancer_llm_model = runtime.llm_model
        prompt_enhancer_llm_tokenizer = runtime.llm_tokenizer
        pipe.update(runtime.pipe_models)
        if runtime.budgets:
            kwargs.setdefault("budgets", {}).update(runtime.budgets)
    else:
        reset_prompt_enhancer()


def ensure_prompt_enhancer_loaded(override_profile=None, progress=None):
    global enhancer_offloadobj

    reset_prompt_enhancer_if_requested()
    if enhancer_offloadobj is None:
        if progress is not None:
            progress(0, "Please Wait While Loading Prompt Enhancer")
        kwargs = {}
        pipe = {}
        download_models()
        setup_prompt_enhancer(pipe, kwargs)
        profile = compute_profile(override_profile, "video")
        mmgp_profile = init_pipe(pipe, kwargs, profile)
        enhancer_offloadobj = offload.profile(pipe, profile_no=mmgp_profile, **kwargs)

    if prompt_enhancer_llm_model is None or prompt_enhancer_llm_tokenizer is None:
        raise gr.Error("Prompt enhancer text runtime is not available.")
    return prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer

def load_models(model_type, override_profile = -1, output_type="video", **model_kwargs):
    global transformer_type, loaded_profile
    def _load_models_info(message):
        if int(verbose_level) > 0:
            print(message)

    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    save_quantized = args.save_quantized and model_def != None
    model_filename = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy) 
    if "URLs2" in model_def:
        model_filename2 = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy, submodel_no=2) # !!!!
    else:
        model_filename2 = None
    modules = get_model_recursive_prop(model_type, "modules",  return_list= True)
    modules = [get_model_recursive_prop(module, "modules", sub_prop_name  ="_list",  return_list= True) if isinstance(module, str) else module for module in modules ]
    if save_quantized and "quanto" in model_filename:
        save_quantized = False
        print("Need to provide a non quantized model to create a quantized model to be saved") 
    if save_quantized and len(modules) > 0:
        print(f"Unable to create a finetune quantized model as some modules are declared in the finetune definition. If your finetune includes already the module weights you can remove the 'modules' entry and try again. If not you will need also to change temporarly the model 'architecture' to an architecture that wont require the modules part ({modules}) to quantize and then add back the original 'modules' and 'architecture' entries.")
        save_quantized = False
    quantizeTransformer = not save_quantized and model_def !=None and transformer_quantization in ("int8", "fp8") and model_def.get("auto_quantize", False) and not "quanto" in model_filename
    if quantizeTransformer and len(modules) > 0:
        print(f"Autoquantize is not yet supported if some modules are declared")
        quantizeTransformer = False
    model_family = get_model_family(model_type)
    transformer_dtype = get_transformer_dtype(model_type, transformer_dtype_policy)
    if quantizeTransformer or "quanto" in model_filename:
        transformer_dtype = torch.bfloat16 if "bf16" in model_filename or "BF16" in model_filename else transformer_dtype
        transformer_dtype = torch.float16 if "fp16" in model_filename or"FP16" in model_filename else transformer_dtype
    perc_reserved_mem_max = args.perc_reserved_mem_max
    vram_safety_coefficient = args.vram_safety_coefficient 
    model_file_list = [model_filename]
    model_type_list = [model_type]
    source_type_list = [0]
    model_submodel_no_list = [1]
    if model_filename2 != None:
        model_file_list += [model_filename2]
        model_type_list += [model_type]
        source_type_list += [0]
        model_submodel_no_list += [2]
    for module_type in modules:
        if isinstance(module_type,dict):
            URLs1 = module_type.get("URLs", None)
            if URLs1 is None: raise Exception(f"No URLs defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs1))
            URLs2 = module_type.get("URLs2", None)
            if URLs2 is None: raise Exception(f"No URL2s defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs2))
            model_type_list += [model_type] * 2
            source_type_list += [1] * 2
            model_submodel_no_list += [1,2]
        else:
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, module_type= module_type))
            model_type_list.append(model_type)
            source_type_list.append(True)
            model_submodel_no_list.append(0) 

    local_model_file_list= []
    for filename, file_model_type, file_source_type, submodel_no in zip(model_file_list, model_type_list, source_type_list, model_submodel_no_list):
        if len(filename) == 0: continue 
        download_models(filename, file_model_type, file_source_type, submodel_no)
        local_file_name = fl.get_local_model_filename(filename )
        local_model_file_list.append( os.path.basename(filename) if local_file_name is None else local_file_name )
    if len(local_model_file_list) == 0:
        download_models("", model_type, 0, -1)

    VAE_dtype = torch.float16 if server_config.get("vae_precision","16") == "16" else torch.float
    mixed_precision_transformer =  server_config.get("mixed_precision","0") == "1"
    transformer_type = None

    for source_type, filename in zip(source_type_list, local_model_file_list):
        if source_type==0:  
            _load_models_info(f"Loading Model '{filename}' ...")
        elif source_type==1:  
            _load_models_info(f"Loading Module '{filename}' ...")


    model_type_handler = model_types_handlers[base_model_type] 
    text_encoder_URLs= get_model_recursive_prop(model_type, "text_encoder_URLs", return_list= True)
    if text_encoder_URLs is not None:
        text_encoder_filename = get_model_filename(model_type=model_type, quantization= text_encoder_quantization, dtype_policy = transformer_dtype_policy, URLs=text_encoder_URLs)
    if text_encoder_filename is not None and len(text_encoder_filename):
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        if text_encoder_filename is not None:
            download_models(text_encoder_filename, file_model_type, 2, -1, force_path =text_encoder_folder)
            text_encoder_filename =  fl.get_local_model_filename(text_encoder_filename, extra_paths=text_encoder_folder)
            _load_models_info(f"Loading Text Encoder '{text_encoder_filename}' ...")


    profile = compute_profile(override_profile, output_type)
    lm_decoder_engine_obtained = resolve_lm_decoder_engine(lm_decoder_engine, model_def.get("lm_engines", []) )
    if lm_decoder_engine_obtained in ("cg", "vllm") and int(profile) not in [ 1, 3]:
        _load_models_info(f"Unable to use LM Engine '{lm_decoder_engine_obtained}' as it requires a Memory Profile such as 1,3 or 3+ that loads entirely the Main Models in VRAM. Switching to Legacy LM Engine...")
        lm_decoder_engine_obtained = "legacy"
    torch.set_default_device('cpu')    
    wan_model, pipe = model_type_handler.load_model(
                local_model_file_list, model_type, base_model_type, model_def, quantizeTransformer = quantizeTransformer, text_encoder_quantization = text_encoder_quantization,
                dtype = transformer_dtype, VAE_dtype = VAE_dtype, mixed_precision_transformer = mixed_precision_transformer, save_quantized = save_quantized, submodel_no_list   = model_submodel_no_list, text_encoder_filename = text_encoder_filename, profile=profile, lm_decoder_engine=lm_decoder_engine_obtained, **model_kwargs )

    kwargs = {}
    if "pipe" in pipe:
        kwargs = pipe
        pipe = kwargs.pop("pipe")
    if "coTenantsMap" not in kwargs: kwargs["coTenantsMap"] = {}
    mmgp_profile = init_pipe(pipe, kwargs, profile)
    if server_config.get("enhancer_mode", 1) == 0:
        setup_prompt_enhancer(pipe, kwargs)
    loras_transformer = kwargs.pop("loras", [])
    if "transformer" in pipe:
        loras_transformer += ["transformer"]        
    if "transformer2" in pipe:
        loras_transformer += ["transformer2"]
    if len(compile) > 0 and hasattr(wan_model, "custom_compile"):
        wan_model.custom_compile(backend= "inductor", mode ="default")
    compile_modules = model_def.get("compile", compile) if len(compile) > 0 else ""
    if compile_modules == False:
        _load_models_info("Pytorch compilation is not supported for this Model")
    # kwargs["pinnedMemory"] = "text_encoder"
    offloadobj = offload.profile(pipe, profile_no= mmgp_profile, compile = compile_modules, quantizeTransformer = False, loras = loras_transformer, perc_reserved_mem_max = perc_reserved_mem_max , vram_safety_coefficient = vram_safety_coefficient , convertWeightsFloatTo = transformer_dtype, **kwargs)  
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)
    transformer_type = model_type
    loaded_profile = profile
    return wan_model, offloadobj 

if not "P" in preload_model_policy:
    wan_model, offloadobj, transformer = None, None, None
    reload_needed = True
else:
    wan_model, offloadobj = load_models(
        transformer_type,
        output_type=get_output_type_for_model(transformer_type, 0),
    )
    if check_loras:
        transformer = get_transformer_model(wan_model)
        if hasattr(wan_model, "get_trans_lora"):
            transformer, _ = wan_model.get_trans_lora()
        setup_loras(transformer_type, transformer,  get_lora_dir(transformer_type), "", None)
        exit()

gen_in_progress = False

def is_generation_in_progress():
    global gen_in_progress
    return gen_in_progress

def get_auto_attention():
    for attn in ["sage2","sage","sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def generate_header(model_type, compile, attention_mode):

    description_container = [""]
    get_model_name(model_type, description_container)
    full_filename = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
    model_filename = os.path.basename(full_filename)
    description  = description_container[0]
    description = f"<DIV style=height:{60 if server_config.get('display_stats', 0) == 1 else 40}px>{description}</DIV>"
    overridden_attention = get_overridden_attention(model_type)
    attn_mode = attention_mode if overridden_attention == None else overridden_attention 
    header = "<DIV style='align:right;width:100%'><FONT SIZE=2>Attention mode <B>" + (attn_mode if attn_mode!="auto" else "auto/" + get_auto_attention() )
    if attention_mode not in attention_modes_installed:
        header += " -NOT INSTALLED-"
    elif attention_mode not in attention_modes_supported:
        header += " -NOT SUPPORTED-"
    elif overridden_attention is not None and attention_mode != overridden_attention:
        header += " -MODEL SPECIFIC-"
    header += "</B>"

    if compile:
        header += ", Pytorch compilation <B>ON</B>"
    if "fp16" in model_filename:
        header += ", Data Type <B>FP16</B>"
    else:
        header += ", Data Type <B>BF16</B>"

    quant_label = quant_router.detect_quantization_label_from_filename(fl.get_local_model_filename(full_filename))
    if quant_label:
        header += f", Quantization <B>{quant_label}</B>"
    header += "</DIV>"

    return description,header

def release_RAM():
    if gen_in_progress:
        gr.Info("Unable to release RAM when a Generation is in Progress")
    else:
        release_model()
        gr.Info("Models stored in RAM have been released")

def get_gen_info(state):
    cache = state.get("gen", None)
    if cache == None:
        cache = dict()
        state["gen"] = cache
    return cache

def build_callback(state, pipe, send_cmd, status, num_inference_steps, preview_meta=None):
    gen = get_gen_info(state)
    gen["num_inference_steps"] = num_inference_steps
    start_time = time.time()    
    def callback(step_idx = -1, latent = None, force_refresh = True, read_state = False, override_num_inference_steps = -1, pass_no = -1, preview_meta=preview_meta, denoising_extra ="", progress_unit = None):
        in_pause = False
        with gen_lock:
            process_status = gen.get("process_status", None)
            pause_msg = None
            if isinstance(process_status, str) and process_status.startswith("request:"):
                gen["process_status"] = "process:" + process_status[len("request:"):]
                offloadobj.unload_all()
                pause_msg = gen.get("pause_msg", "Unknown Pause")
                in_pause = True

        if in_pause:
            send_cmd("progress", [0, pause_msg])
            while True:
                time.sleep(0.1)            
                with gen_lock:
                    process_status = gen.get("process_status", None)
                    if isinstance(process_status, str) and process_status.startswith("request:"):
                        gen["process_status"] = "process:" + process_status[len("request:"):]
                        continue
                    if process_status == "process:main": break
            force_refresh = True
        if gen.get("early_stop", False) and not gen.get("early_stop_forwarded", False):
            gen["early_stop_forwarded"] = True
            if hasattr(pipe, "request_early_stop"):
                pipe.request_early_stop()
            elif wan_model is not None and hasattr(wan_model, "request_early_stop"):
                wan_model.request_early_stop()
            elif hasattr(pipe, "_early_stop"):
                pipe._early_stop = True
            elif wan_model is not None and hasattr(wan_model, "_early_stop"):
                wan_model._early_stop = True
        refresh_id =  gen.get("refresh", -1)
        if force_refresh or step_idx >= 0:
            pass
        else:
            refresh_id =  gen.get("refresh", -1)
            if refresh_id < 0:
                return
            UI_refresh = state.get("refresh", 0)
            if UI_refresh >= refresh_id:
                return  
        if override_num_inference_steps > 0:
            gen["num_inference_steps"] = override_num_inference_steps
             
        num_inference_steps = gen.get("num_inference_steps", 0)
        status = gen["progress_status"]
        state["refresh"] = refresh_id
        if read_state:
            phase, state_step_idx = gen["progress_phase"]
            step_idx = state_step_idx if step_idx < 0 else step_idx + 1
        else:
            step_idx += 1         
            if gen.get("abort", False):
                # pipe._interrupt = True
                phase = "Aborting"    
            elif gen.get("early_stop", False):
                phase = "Early Stop in progress"
            elif step_idx  == num_inference_steps:
                phase = "VAE Decoding"    
            else:
                if pass_no <=0:
                    phase = "Denoising"
                elif pass_no == 1:
                    phase = "Denoising First Pass"
                elif pass_no == 2:
                    phase = "Denoising Second Pass"
                elif pass_no == 3:
                    phase = "Denoising Third Pass"
                else:
                    phase = f"Denoising {pass_no}th Pass"

                if len(denoising_extra) > 0: phase += " | " + denoising_extra

            gen["progress_phase"] = (phase, step_idx)
        status_msg = merge_status_context(status, phase)      

        elapsed_time = time.time() - start_time
        status_msg = merge_status_context(status, f"{phase} | {format_time(elapsed_time)}")              
        if step_idx >= 0:
            progress_args = [(step_idx , num_inference_steps) , status_msg  ,  num_inference_steps]
            if progress_unit:
                progress_args.append(progress_unit)
        else:
            progress_args = [0, status_msg]
        
        # progress(*progress_args)
        send_cmd("progress", progress_args)
        if latent is not None:
            payload = pipe.prepare_preview_payload(latent, preview_meta) if hasattr(pipe, "prepare_preview_payload") else latent
            if isinstance(payload, dict):
                data = payload.copy()
                lat = data.get("latents")
                if torch.is_tensor(lat):
                    data["latents"] = lat.to("cpu", non_blocking=True)
                payload = data
            elif torch.is_tensor(payload):
                payload = payload.to("cpu", non_blocking=True)
            if payload is not None:
                send_cmd("preview", payload)
            
        # gen["progress_args"] = progress_args
            
    return callback

def pause_generation(state):
    gen = get_gen_info(state)
    process_id = "pause"
    GPU_process_running = any_GPU_process_running(state, process_id, ignore_main= True )
    if GPU_process_running:
        gr.Info("Unable to pause, a PlugIn is using the GPU")
        yield gr.update(), gr.update()
        return
    gen["resume"] = False
    yield gr.Button(interactive= False), gr.update()
    pause_msg = "Generation on Pause, click Resume to Restart Generation"
    acquire_GPU_ressources(state, process_id , "Pause", gr= gr, custom_pause_msg= pause_msg, custom_wait_msg= "Please wait while the Pause Request is being Processed...")      
    gr.Info(pause_msg)
    yield gr.Button(visible= False, interactive= True), gr.Button(visible= True)
    while not gen.get("resume", False):
        time.sleep(0.5)

    release_GPU_ressources(state, process_id )
    gen["resume"] = False
    yield gr.Button(visible= True, interactive= True), gr.Button(visible= False)

def resume_generation(state):
    gen = get_gen_info(state)
    gen["resume"] = True

def abort_generation(state, client_id="", notify = True):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    with lock:
        if len(queue):
            if len(client_id):
                for i, task in enumerate(queue):
                    queue_client_id = task["params"].get("client_id","")
                    if queue_client_id == client_id:
                        if i == 0:
                            if "in_progress" not in gen:
                                del queue[0]
                                if "prompt_no" in gen: gen["prompt_no"] += 1
                                return gr.update(), gr.HTML(value=generate_queue_html(queue))
                            break
                        del queue[i]
                        if "prompt_no" in gen: gen["prompt_no"] += 1
                        return gr.update(), gr.HTML(value=generate_queue_html(queue))
            elif "in_progress" not in gen:
                del queue[0]
                gen["prompt_no"] += 1
                return gr.update(), gr.HTML(value=generate_queue_html(queue))

    gen["resume"] = True
    if "in_progress" in gen: # and wan_model != None:
        if wan_model != None:
            wan_model._interrupt= True
        gen["abort"] = True            
        msg = "Processing Request to abort Current Generation"
        gen["status"] = msg
        if notify:
            gr.Info(msg)
        return gr.Button(interactive=  False), gr.update()
    else:
        return gr.Button(interactive=  True), gr.update()

def early_stop_generation(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    if "in_progress" in gen:
        queue = gen.get("queue", [])
        model_type = queue[0].get("params", {}).get("model_type") if queue else None
        model_def = get_model_def(model_type) if model_type else None
        if not model_def or not model_def.get("supports_early_stop", False):
            gr.Info("Early Stop is not supported for this model.")
            return gr.Button(interactive=True)
        if gen.get("early_stop", False):
            return gr.Button(interactive=False)
        gen["early_stop"] = True
        gen["early_stop_forwarded"] = False
        msg = "Early Stop in progress"
        gen["status"] = msg
        gr.Info(msg)
        return gr.Button(interactive=False)
    return gr.Button(interactive=True)

def pack_audio_gallery_state(audio_file_list, selected_index, refresh = True):
    return [json.dumps(audio_file_list), selected_index, time.time()]

def unpack_audio_list(packed_audio_file_list):
    return json.loads(packed_audio_file_list)

def refresh_gallery(state): #, msg
    gen = get_gen_info(state)

    # gen["last_msg"] = msg
    clear_deleted_files(state, False)
    clear_deleted_files(state, True)
    file_list = gen.get("file_list", None)      
    choice = gen.get("selected",0)
    audio_file_list = gen.get("audio_file_list", None)      
    audio_choice = gen.get("audio_selected",-1)

    header_text = gen.get("header_text", "")
    in_progress = "in_progress" in gen
    if gen.get("last_selected", True) and file_list is not None:
        choice = max(len(file_list) - 1,0)  
    if gen.get("audio_last_selected", True) and audio_file_list is not None:
        audio_choice = max(len(audio_file_list) - 1,-1)  
    last_was_audio = gen.get("last_was_audio", False)
    queue = gen.get("queue", [])
    abort_interactive = not gen.get("abort", False)
    early_stop_interactive = not gen.get("early_stop", False)
    early_stop_visible = False

    if gen.pop("refresh_tab", False):
        if last_was_audio: 
            output_tabs = [gr.Tabs(selected= "audio"), 1]
        else:
            output_tabs = [gr.Tabs(selected= "video_images"), 0]
    else:
        output_tabs = [gr.update(), gr.update()]

    if not in_progress or len(queue) == 0:
        return *output_tabs, gr.Gallery(value = file_list) if last_was_audio else gr.Gallery(selected_index=choice, value = file_list), gr.update() if last_was_audio else choice, *pack_audio_gallery_state(audio_file_list, audio_choice), gr.HTML("", visible= False),  gr.Button(visible=True), gr.Button(visible=False), gr.Row(visible=False), gr.Row(visible=False), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(interactive=  early_stop_interactive, visible= early_stop_visible), gr.Button(visible= False)
    else:
        task = queue[0]
        prompt =  task["prompt"]
        params = task["params"]
        model_type = params["model_type"] 
        multi_prompts_gen_type = params["multi_prompts_gen_type"]
        base_model_type = get_base_model_type(model_type)
        model_def = get_model_def(model_type) 
        onemorewindow_visible = test_any_sliding_window(base_model_type) and params.get("image_mode",0) == 0 and (not params.get("mode","").startswith("edit_")) and not model_def.get("preprocess_all", False)
        early_stop_visible = bool(model_def.get("supports_early_stop", False))
        enhanced = False
        if prompt.startswith(prompt_parser.ENHANCED_PROMPT_PREFIX):
            enhanced = True
            prompt = prompt[len(prompt_parser.ENHANCED_PROMPT_PREFIX):]
        prompt_units = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
        if multi_prompts_gen_type == "FG" or len(prompt_units) <= 1:
            prompt = html.escape(prompt_units[0] if len(prompt_units) > 0 else prompt).replace("\n", "<BR>")
        else:
            window_no = gen.get("window_no", 1)
            if window_no > len(prompt_units):
                window_no = len(prompt_units)
            window_no -= 1
            escaped_prompts = []
            for idx, prompt_unit in enumerate(prompt_units):
                escaped_prompt = html.escape(prompt_unit).replace("\n", "<BR>")
                if "W" in multi_prompts_gen_type and idx == window_no:
                    escaped_prompt = "<B>" + escaped_prompt + "</B>"
                escaped_prompts.append(escaped_prompt)
            prompt = "<BR><DIV style='height:8px'></DIV>".join(escaped_prompts)
        if enhanced:
            prompt = "<U><B>Enhanced:</B></U><BR>" + prompt

        if len(header_text) > 0:
            prompt =  "<I>" + header_text + "</I><BR><BR>" + prompt
        thumbnail_size = "100px"
        thumbnails = ""
        
        start_img_data = task.get('start_image_data_base64')
        start_img_labels = task.get('start_image_labels')
        if start_img_data and start_img_labels:
            for i, (img_uri, img_label) in enumerate(zip(start_img_data, start_img_labels)):
                thumbnails += f'<td><div class="hover-image" onclick="showImageModal(\'current_start_{i}\')"><img src="{img_uri}" alt="{img_label}" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" /><span class="tooltip">{img_label}</span></div></td>'
        
        end_img_data = task.get('end_image_data_base64')
        end_img_labels = task.get('end_image_labels')
        if end_img_data and end_img_labels:
            for i, (img_uri, img_label) in enumerate(zip(end_img_data, end_img_labels)):
                thumbnails += f'<td><div class="hover-image" onclick="showImageModal(\'current_end_{i}\')"><img src="{img_uri}" alt="{img_label}" style="max-width:{thumbnail_size}; max-height:{thumbnail_size}; display: block; margin: auto; object-fit: contain;" /><span class="tooltip">{img_label}</span></div></td>'
        
        # Get current theme from server config  
        current_theme = server_config.get("UI_theme", "default")
        
        # Use minimal, adaptive styling that blends with any background
        # This creates a subtle container that doesn't interfere with the page's theme
        table_style = """
            border: 1px solid rgba(128, 128, 128, 0.3); 
            background-color: transparent; 
            color: inherit; 
            padding: 8px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        """
        if params.get("mode", None) in ['edit'] : onemorewindow_visible = False
        gen_buttons_visible = True
        html_content =  f"<TABLE WIDTH=100% ID=PINFO style='{table_style}'><TR style='height:140px'><TD width=100% style='{table_style}'>" + prompt + "</TD>" + thumbnails + "</TR></TABLE>" 
        html_output = gr.HTML(html_content, visible= True)
        if last_was_audio:
            audio_choice = max(-1, audio_choice)
        else:
            choice = max(0, choice)
                    
        return *output_tabs, gr.Gallery(value = file_list) if last_was_audio else gr.Gallery(selected_index=choice, value = file_list), gr.update() if last_was_audio else choice, *pack_audio_gallery_state(audio_file_list, audio_choice), html_output, gr.Button(visible=False), gr.Button(visible=True), gr.Row(visible=True), gr.Row(visible= gen_buttons_visible), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(interactive=  early_stop_interactive, visible= early_stop_visible), gr.Button(visible= onemorewindow_visible)



def finalize_generation(state):
    gen = get_gen_info(state)
    choice = gen.get("selected",0)
    if "in_progress" in gen:
        del gen["in_progress"]
    if gen.get("last_selected", True):
        file_list = gen.get("file_list", [])
        choice = len(file_list) - 1

    audio_file_list = gen.get("audio_file_list", [])
    audio_choice  = gen.get("audio_selected", -1)
    if gen.get("audio_last_selected", True):
        audio_choice = len(audio_file_list) - 1

    gen["extra_orders"] = 0
    last_was_audio = gen.get("last_was_audio", False)
    gallery_tabs = gr.Tabs(selected= "audio" if last_was_audio else "video_images")
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    gen["early_stop"] = False
    gen["early_stop_forwarded"] = False
    return gallery_tabs, 1 if last_was_audio else 0, gr.update() if last_was_audio else gr.Gallery(selected_index=choice),  *pack_audio_gallery_state(audio_file_list, audio_choice), gr.Button(interactive=  True), gr.Button(interactive=  True, visible= False), gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.HTML(visible= False, value="")

def get_default_video_info():
    return "Please Select an Video / Image"    


def get_file_list(state, input_file_list, audio_files = False):
    gen = get_gen_info(state)
    with lock:
        if audio_files:
            file_list_name = "audio_file_list"
            file_settings_name = "audio_file_settings_list"
        else:
            file_list_name = "file_list"
            file_settings_name = "file_settings_list"

        if file_list_name in gen:
            file_list = gen[file_list_name]
            file_settings_list = gen[file_settings_name]
        else:
            file_list = []
            file_settings_list = []
            if input_file_list != None:
                if not isinstance(input_file_list, list): input_file_list = [input_file_list]
                for file_path in input_file_list:
                    file_path = get_gradio_file_path(file_path)
                    if not file_path or not os.path.isfile(file_path):
                        continue
                    file_settings, _, _ = get_settings_from_file(state, file_path, False, False, False)
                    file_list.append(file_path)
                    file_settings_list.append(file_settings)
 
            gen[file_list_name] = file_list 
            gen[file_settings_name] = file_settings_list 
    return file_list, file_settings_list

def set_file_choice(gen, file_list, choice, audio_files = False):
    if len(file_list) > 0: choice = max(choice,0)
    gen["audio_last_selected" if audio_files else "last_selected"] = (choice + 1) >= len(file_list)
    gen["audio_selected" if audio_files else "selected"] = choice
    gen["current_gallery_source"] = "audio" if audio_files else "video"
    gen["selected_video_time"] = None if audio_files or choice < 0 or choice >= len(file_list) or not has_video_file_extension(file_list[choice]) else 0.0

def select_audio(state, audio_files_paths, audio_file_selected):
    gen = get_gen_info(state)
    audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths))

    if audio_file_selected >= 0:
        choice = audio_file_selected
    else:
        choice = min(len(audio_file_list)-1, gen.get("audio_selected",-1)) if len(audio_file_list) > 0 else -1
    set_file_choice(gen,  audio_file_list, choice, audio_files=True )


video_guide_processes = "OPEDSLCMU"
all_guide_processes = video_guide_processes + "VGBH"

process_map_outside_mask = { "Y" : "depth", "W": "scribble", "X": "inpaint", "Z": "flow"}
process_map_video_guide = { "O": "pose_align", "P": "pose", "D" : "depth", "S": "scribble", "E": "canny", "L": "flow", "C": "gray", "M": "inpaint", "U": "identity"}
all_process_map_video_guide =  { "B": "face", "H" : "bbox"}
all_process_map_video_guide.update(process_map_video_guide)
processes_names = { "pose": "Open Pose", "pose_align": "Aligned Open Pose", "depth": "Depth Mask", "scribble" : "Shapes", "flow" : "Flow Map", "gray" : "Gray Levels", "inpaint" : "Inpaint Mask", "identity": "Identity Mask", "raw" : "Raw Format", "canny" : "Canny Edges", "face": "Face Movements", "bbox": "BBox"}


def resolve_media_creation_date(file_name, configs=None):
    creation_dt = extract_creation_datetime_from_metadata(configs) if isinstance(configs, dict) else None
    if creation_dt is None and has_audio_file_extension(file_name):
        try:
            creation_dt = resolve_audio_creation_datetime(file_name, wangp_metadata=configs if isinstance(configs, dict) else None)
        except Exception:
            creation_dt = None
    if creation_dt is None:
        creation_dt = get_file_creation_date(file_name)
    creation_date = str(creation_dt)
    if "." in creation_date:
        creation_date = creation_date[:creation_date.rfind(".")]
    return creation_date


def is_deepy_display_metadata(configs):
    return isinstance(configs, dict) and str(configs.get("model_type", "") or "").strip() == "Deepy"


def update_video_prompt_type(state, any_video_guide = False, any_video_mask = False, any_background_image_ref = False, process_type = None, default_update = ""):
    letters = default_update
    settings = get_current_model_settings(state)
    video_prompt_type = settings["video_prompt_type"]
    if process_type  is not None:
        video_prompt_type = del_in_sequence(video_prompt_type, video_guide_processes)
        for one_process_type in process_type: 
            for k,v in process_map_video_guide.items():
                if v== one_process_type:
                    letters += k
                    break
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    guide_preprocessing = model_def.get("guide_preprocessing", None) 
    mask_preprocessing = model_def.get("mask_preprocessing", None) 
    guide_custom_choices = model_def.get("guide_custom_choices", None) 
    if any_video_guide: letters += "V"
    if any_video_mask: letters += "A"
    if any_background_image_ref: 
        video_prompt_type = del_in_sequence(video_prompt_type, "F")
        letters += "KI"
    validated_letters = ""
    for letter in letters:
        if not guide_preprocessing is None:
            if any(letter in choice for choice in guide_preprocessing["selection"] ):
                validated_letters += letter
                continue
        if not mask_preprocessing is None:
            if any(letter in choice for choice in mask_preprocessing["selection"] ):
                validated_letters += letter
                continue
        if not guide_custom_choices is None:
            if any(letter in choice for label, choice in guide_custom_choices["choices"] ):
                validated_letters += letter
                continue
    video_prompt_type = add_to_sequence(video_prompt_type, letters)
    settings["video_prompt_type"] = video_prompt_type 


def select_video(state, current_gallery_tab, input_file_list, file_selected, audio_files_paths, audio_file_selected, source, event_data: gr.EventData):
    gen = get_gen_info(state)
    if source=="video":
        if current_gallery_tab != 0:
            return [gr.update()] * 9
        file_list, file_settings_list = get_file_list(state, input_file_list)
        data=  event_data._data
        if data!=None and isinstance(data, dict):
            choice = data.get("index",0)        
        elif file_selected >= 0:
            choice = file_selected
        else:
            choice = gen.get("selected",0)
        choice = min(len(file_list)-1, choice) 
        set_file_choice(gen, file_list, choice)
        files, settings_list = file_list, file_settings_list
    else:
        if current_gallery_tab != 1:
            return [gr.update()] * 9
        audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths), audio_files= True)
        if audio_file_selected >= 0:
            choice = audio_file_selected
        else:
            choice = gen.get("audio_selected",-1)
        choice = min(len(audio_file_list)-1, choice)
        set_file_choice(gen,  audio_file_list, choice, audio_files=True )
        files, settings_list = audio_file_list, audio_file_settings_list

    is_audio = False
    is_image = False
    is_video = False
    is_deleted = False
    if len(files) > 0:
        if len(settings_list) <= choice:
            pass 
        configs = settings_list[choice]
        file_name = files[choice]
        values = [os.path.basename(strip_virtual_media_suffix(file_name))]
        labels = [ "File Name"]
        misc_values= []
        misc_labels = []
        pp_values= []
        pp_labels = []
        nb_audio_tracks =  0 

        if not os.path.isfile(file_name):
            is_deleted = True
            configs = None
        elif has_audio_file_extension(file_name):
            is_audio = True        
            width, height = 0, 0
            frames_count = fps = 1
        elif not has_video_file_extension(file_name):
            img = _open_image_input(file_name)
            width, height = img.size
            is_image = True
            frames_count = fps = 1
        else:
            fps, width, height, frames_count = get_video_info(file_name)
            is_video = True
            nb_audio_tracks = extract_audio_tracks(file_name,query_only = True)

        if configs != None:
            video_model_name =  configs.get("type", "Unknown model")
            if "-" in video_model_name: video_model_name =  video_model_name[video_model_name.find("-")+2:] 
            misc_values += [video_model_name]
            misc_labels += ["Model"]
            video_temporal_upsampling = configs.get("temporal_upsampling", "")
            video_spatial_upsampling = configs.get("spatial_upsampling", "")
            video_film_grain_intensity = configs.get("film_grain_intensity", 0)
            video_film_grain_saturation = configs.get("film_grain_saturation", 0.5)
            video_MMAudio_setting = configs.get("MMAudio_setting", 0)
            video_MMAudio_prompt = configs.get("MMAudio_prompt", "")
            video_MMAudio_neg_prompt = configs.get("MMAudio_neg_prompt", "")
            video_seed = configs.get("seed", -1)
            video_MMAudio_seed = configs.get("MMAudio_seed", video_seed)        
            if len(video_spatial_upsampling) > 0:
                video_temporal_upsampling += " " + video_spatial_upsampling
            if len(video_temporal_upsampling) > 0:
                pp_values += [ video_temporal_upsampling ]
                pp_labels += [ "Upsampling" ]
            if video_film_grain_intensity > 0:
                pp_values += [ f"Intensity={video_film_grain_intensity}, Saturation={video_film_grain_saturation}" ]
                pp_labels += [ "Film Grain" ]
            if video_MMAudio_setting != 0:
                pp_values += [ f'Prompt="{video_MMAudio_prompt}", Neg Prompt="{video_MMAudio_neg_prompt}", Seed={video_MMAudio_seed}'  ]
                pp_labels += [ "MMAudio" ]


        if is_deepy_display_metadata(configs):
            values += ["Deepy"]
            labels += ["Made By"]
            video_prompt = html.escape(str(configs.get("prompt", "") or "")[:1024]).replace("\n", "<BR>")
            if len(video_prompt) > 0:
                values += [video_prompt]
                labels += ["Prompt"]
            video_creation_date = "Deleted" if is_deleted else resolve_media_creation_date(file_name, configs)
            if is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            elif is_video:
                values += [f"{width}x{height}", f"{frames_count} frames (duration={frames_count/fps:.1f} s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
            else:
                duration_seconds = configs.get("duration_seconds", None)
                if duration_seconds is not None:
                    values += [f"{duration_seconds}s"]
                duration_def = model_def.get("duration_slider", None)
                duration_label = "Max Duration"
                if duration_def is not None:
                    duration_label = duration_def.get("label", duration_label)
                    labels += [duration_label]
            if nb_audio_tracks > 0:
                values += [nb_audio_tracks]
                labels += ["Nb Audio Tracks"]
            values += [video_creation_date]
            labels += ["Creation Date"]
        elif configs == None or not "seed" in configs:
            values += misc_values
            labels += misc_labels
            
            video_creation_date = "Deleted" if is_deleted else resolve_media_creation_date(file_name, configs)
            if is_audio:
                pass
            elif is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            elif is_video:
                values += [f"{width}x{height}",  f"{frames_count} frames (duration={frames_count/fps:.1f} s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
                extra_values, extra_labels = get_video_summary_extras(file_name)
                values += extra_values
                labels += extra_labels
            if nb_audio_tracks  > 0:
                values +=[nb_audio_tracks]
                labels +=["Nb Audio Tracks"]

            values += pp_values
            labels += pp_labels

            values +=[video_creation_date]
            labels +=["Creation Date"]
        else: 
            video_prompt =  html.escape(configs.get("prompt", "")[:1024]).replace("\n", "<BR>")
            enhanced_video_prompt = html.escape(configs.get("enhanced_prompt", "")[:1024]).replace("\n", "<BR>")
            video_video_prompt_type = configs.get("video_prompt_type", "")
            video_image_prompt_type = configs.get("image_prompt_type", "")
            video_audio_prompt_type = configs.get("audio_prompt_type", "")
            def check(src, cond):
                pos, neg = cond if isinstance(cond, tuple) else (cond, None)
                if not all_letters(src, pos): return False
                if neg is not None and any_letters(src, neg): return False
                return True
            image_outputs = configs.get("image_mode",0) > 0
            video_model_type =  configs.get("model_type", "t2v")
            model_family = get_model_family(video_model_type)
            model_def = get_model_def(video_model_type)
            map_video_prompt  = {"V" : "Control Image" if image_outputs else "Control Video", ("VA", "U") : "Mask Image" if image_outputs else "Mask Video", "I" : "Reference Images", "&": "HDR Output"}
            map_image_prompt  = {"V" : "Source Video", "L" : "Last Video", "S" : "Start Image", "E" : "End Image"}
            map_audio_prompt  = {"A" : "Audio Source", "O": "Force Output Audio", "B" : "Audio Source #2", "K": "Control Video Audio Track", "N": "Normalized Audio Volumes"}
            audio_prompt_type_sources_def = model_def.get("audio_prompt_type_sources", None)
            if isinstance(audio_prompt_type_sources_def, dict):
                custom_flags = audio_prompt_type_sources_def.get("custom_flags", {})
                if isinstance(custom_flags, dict):
                    for flag, label in custom_flags.items():
                        flag = str(flag or "")
                        if len(flag) == 1 and flag in "0123456789" and isinstance(label, str) and len(label) > 0:
                            map_audio_prompt[flag] = label
            video_other_prompts =  [ v for s,v in map_image_prompt.items() if all_letters(video_image_prompt_type,s)] \
                                 + [ v for s,v in map_video_prompt.items() if check(video_video_prompt_type,s)] \
                                 + [ v for s,v in map_audio_prompt.items() if all_letters(video_audio_prompt_type,s)] 
            any_mask = "A" in video_video_prompt_type and not "U" in video_video_prompt_type            
            multiple_submodels = model_def.get("multiple_submodels", False)
            video_other_prompts = ", ".join(video_other_prompts)
            if is_audio:
                video_resolution = None
                video_length_summary = None
                video_length_label = ""
                original_fps = 0
                video_num_inference_steps = configs.get("num_inference_steps", None)
            else:
                video_length = configs.get("video_length", 0)
                original_fps= int(video_length/frames_count*fps)
                video_length_summary = f"{video_length} frames"
                video_window_no = configs.get("window_no", 0)
                if video_window_no > 0: video_length_summary +=f", Window no {video_window_no }" 
                if is_image:
                    video_length_summary = configs.get("batch_size", 1)
                    video_length_label = "Number of Images"
                else:
                    video_length_summary += " ("
                    video_length_label = "Video Length"
                    if video_length != frames_count: video_length_summary += f"real: {frames_count} frames, "
                    video_length_summary += f"{frames_count/fps:.1f}s, {round(fps)} fps)"
                video_resolution = configs.get("resolution", "") + f" (real: {width}x{height})"
                video_num_inference_steps = configs.get("num_inference_steps", 0)

            video_guidance_scale = configs.get("guidance_scale", None)
            video_guidance2_scale = configs.get("guidance2_scale", None)
            video_guidance3_scale = configs.get("guidance3_scale", None)
            video_audio_guidance_scale = configs.get("audio_guidance_scale", None)
            video_alt_guidance_scale = configs.get("alt_guidance_scale", None)
            video_alt_scale = configs.get("alt_scale", None)
            video_temperature = configs.get("temperature", None)
            video_top_p = configs.get("top_p", None)
            video_top_k = configs.get("top_k", None)
            video_switch_threshold = configs.get("switch_threshold", 0)
            video_switch_threshold2 = configs.get("switch_threshold2", 0)
            video_model_switch_phase = configs.get("model_switch_phase", 1)
            video_guidance_phases = configs.get("guidance_phases", 0)
            video_embedded_guidance_scale = configs.get("embedded_guidance_scale", None)
            video_guidance_label = "Guidance"
            visible_phases = model_def.get("visible_phases", video_guidance_phases)
            if model_def.get("embedded_guidance", False):
                video_guidance_scale = video_embedded_guidance_scale
                video_guidance_label = "Embedded Guidance Scale"
            elif video_guidance_phases == 0 or visible_phases ==0:
                video_guidance_scale = None 
            elif video_guidance_phases > 0:
                if video_guidance_phases == 1 and visible_phases >=1:
                    video_guidance_scale = f"{video_guidance_scale}"
                elif video_guidance_phases == 2 and visible_phases >=2:
                    if multiple_submodels:
                        video_guidance_scale = f"{video_guidance_scale} (High Noise), {video_guidance2_scale} (Low Noise) with Switch at Noise Level {video_switch_threshold}"
                    else:
                        video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale}" + ("" if video_switch_threshold ==0 else " with Guidance Switch at Noise Level {video_switch_threshold}")
                elif visible_phases >=3:
                    video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale} & {video_guidance3_scale} with Switch at Noise Levels {video_switch_threshold} & {video_switch_threshold2}"
                    if multiple_submodels:
                        video_guidance_scale += f" + Model Switch at {video_switch_threshold if video_model_switch_phase ==1 else video_switch_threshold2}"
            video_phases_label = "Phases"
            video_phases_value = None
            if video_guidance_phases != visible_phases :
                video_phases_value = str(video_guidance_phases)

            if  model_def.get("flow_shift", False): 
                video_flow_shift = configs.get("flow_shift", None)
            else:
                video_flow_shift = None 

            video_video_guide_outpainting = configs.get("video_guide_outpainting", "")
            video_video_guide_outpainting_ratio = configs.get("video_guide_outpainting_ratio", "")
            video_outpainting = ""
            if len(video_video_guide_outpainting) > 0  and not video_video_guide_outpainting.startswith("#") \
                    and (any_letters(video_video_prompt_type, "VFK") ) :
                video_video_guide_outpainting = video_video_guide_outpainting.split(" ")
                video_outpainting = f"Top={video_video_guide_outpainting[0]}%, Bottom={video_video_guide_outpainting[1]}%, Left={video_video_guide_outpainting[2]}%, Right={video_video_guide_outpainting[3]}%" 
            elif len(video_video_guide_outpainting_ratio) > 0 and not video_video_guide_outpainting.startswith("#") and any_letters(video_video_prompt_type, "VFK"):
                video_outpainting = "Top=0%, Bottom=0%, Left=0%, Right=0%"
            if len(video_outpainting) > 0 and len(video_video_guide_outpainting_ratio) > 0:
                video_outpainting += f", Fit {video_video_guide_outpainting_ratio}"
            video_creation_date = resolve_media_creation_date(file_name, configs)
            video_generation_time = format_generation_time(float(configs.get("generation_time", "0")))
            video_activated_loras = configs.get("activated_loras", [])
            video_loras_multipliers = configs.get("loras_multipliers", "")
            video_loras_multipliers =  preparse_loras_multipliers(video_loras_multipliers)
            video_loras_multipliers += [""] * len(video_activated_loras)

            video_activated_loras = [ f"<span class='copy-swap' tabindex=0><SPAN class='copy-swap__trunc' >{os.path.basename(lora)}</span><span class='copy-swap__full'>{lora}</span></span>" for lora in video_activated_loras] 
            video_activated_loras = [ f"<TR><TD style='padding-top:0px;padding-left:0px'>{lora}</TD><TD>x{multiplier if len(multiplier)>0 else '1'}</TD></TR>" for lora, multiplier in zip(video_activated_loras, video_loras_multipliers) ]
            video_activated_loras_str = "<TABLE style='border:0px;padding:0px'>" + "".join(video_activated_loras) + "</TABLE>" if len(video_activated_loras) > 0 else ""
            video_duration_seconds = configs.get("duration_seconds", 0)
            if model_def.get("duration_slider", None) is not None and video_duration_seconds > 0:
                misc_values += [ f"{video_duration_seconds}s"]
                misc_labels += ["Duration"]                             
            prompt_class = model_def.get("prompt_class","Text Prompt")
            values +=  misc_values + [video_prompt]
            labels +=  misc_labels + [ prompt_class]
            video_comments = html.escape(str(configs.get("comments", "") or "")[:4096]).replace("\n", "<BR>")
            if len(video_comments) > 0:
                values += [video_comments]
                labels += ["Comments"]
            alt_prompt_def = model_def.get("alt_prompt", None)
            if alt_prompt_def is not None:
                alt_prompt_label = alt_prompt_def.get("name", alt_prompt_def.get("label")) 
                alt_prompt = html.escape(configs.get("alt_prompt", "")[:1024]).replace("\n", "<BR>")
                if len(alt_prompt):
                    values += [alt_prompt]
                    labels += [alt_prompt_label]
            extra_info = configs.get("extra_info", None)
            if isinstance(extra_info, dict):
                for extra_label, extra_text in extra_info.items():
                    if extra_text is None:
                        continue
                    extra_text = str(extra_text).strip()
                    if len(extra_text) == 0:
                        continue
                    values += [html.escape(extra_text[:4096]).replace("\n", "<BR>")]
                    labels += [html.escape(str(extra_label))]
            if len(enhanced_video_prompt):
                values += [enhanced_video_prompt]
                labels += ["Enhanced {prompt_class}"]
            if len(video_other_prompts) >0 :
                values += [video_other_prompts]
                labels += ["Other Prompts"]
            def gen_process_list(map):
                video_preprocesses = ""
                for k,v in map.items():
                    if k in video_video_prompt_type:
                        process_name = processes_names[v]
                        video_preprocesses += process_name if len(video_preprocesses) == 0 else ", " + process_name 
                return video_preprocesses 

            video_preprocesses_in = gen_process_list(all_process_map_video_guide) if "V" else ""
            video_preprocesses_out = gen_process_list(process_map_outside_mask) if "V" else ""
            if "N" in video_video_prompt_type:
                alt = video_preprocesses_in
                video_preprocesses_in = video_preprocesses_out
                video_preprocesses_out = alt
            if len(video_preprocesses_in) >0 and "V" in video_video_prompt_type:
                values += [video_preprocesses_in]
                labels += [ "Process Inside Mask" if any_mask else "Preprocessing"]

            if len(video_preprocesses_out) >0 and "V" in video_video_prompt_type:
                values += [video_preprocesses_out]
                labels += [ "Process Outside Mask"]
            video_frames_positions = configs.get("frames_positions", "")
            if "F" in video_video_prompt_type and len(video_frames_positions):
                values += [video_frames_positions]
                labels += [ "Injected Frames"]
            if len(video_outpainting) >0:
                values += [video_outpainting]
                labels += ["Outpainting"]
            if input_video_strength_visible(model_def, video_image_prompt_type, video_video_prompt_type):
                values += [configs.get("input_video_strength",1)]
                labels += [extra_settings.get_summary_label("input_video_strength", model_def, fallback="Input Image Strength")]

            if "G" in video_video_prompt_type and "V" in video_video_prompt_type:
                values += [configs.get("denoising_strength",1)]
                labels += [extra_settings.get_summary_label("denoising_strength", model_def, fallback="Denoising Strength")]
            if ("G" in video_video_prompt_type or model_def.get("mask_strength_always_enabled", False)) and "A" in video_video_prompt_type and "U" not in video_video_prompt_type:
                values += [configs.get("masking_strength",1)]
                labels += [extra_settings.get_summary_label("masking_strength", model_def, fallback="Masking Strength")]

            video_sample_solver = configs.get("sample_solver", "")
            if model_def.get("sample_solvers", None) is not None and len(video_sample_solver) > 0 :
                values += [video_sample_solver]
                labels += ["Sampler Solver"]                                        
            values += [video_resolution, video_length_summary, video_seed, video_phases_value, video_guidance_scale, video_audio_guidance_scale]
            labels += ["Resolution", video_length_label, "Seed", video_phases_label,  video_guidance_label, "Audio Guidance Scale"]
            if is_video:
                extra_values, extra_labels = get_video_summary_extras(file_name)
                values += extra_values
                labels += extra_labels
            video_custom_settings = configs.get("custom_settings", None)
            if isinstance(video_custom_settings, dict):
                custom_settings = get_model_custom_settings(model_def)
                for idx, setting_def in enumerate(custom_settings):
                    setting_id = setting_def.get("id", get_custom_setting_id(setting_def, idx))
                    setting_value = video_custom_settings.get(setting_id, None)
                    if setting_value is None:
                        continue
                    if isinstance(setting_value, str) and len(setting_value.strip()) == 0:
                        continue
                    values += [setting_value]
                    labels += [setting_def.get("name", f"Custom Setting {idx + 1}")]
            if model_def.get("temperature", True) and video_temperature is not None:
                values += [video_temperature]
                labels += ["Temperature"]
            if model_def.get("top_p_slider", False) and video_top_p is not None:
                values += [video_top_p]
                labels += ["Top-p"]
            if model_def.get("top_k_slider", False) and video_top_k is not None:
                values += [video_top_k]
                labels += ["Top-k"]
            if is_audio and model_def.get("pause_between_sentences", False):
                values += [configs.get("pause_seconds", 0.0)]
                labels += ["Pause (s)"]
            alt_guidance_type = model_def.get("alt_guidance", None)
            if alt_guidance_type is not None and video_alt_guidance_scale is not None:
                values += [video_alt_guidance_scale]
                labels += [alt_guidance_type]
            alt_scale_type = model_def.get("alt_scale", None)
            if alt_scale_type is not None and video_alt_scale is not None:
                values += [video_alt_scale]
                labels += [alt_scale_type]
            if model_def.get("flow_shift", False):
                values += [video_flow_shift]
                labels += ["Shift Scale"]
            if model_def.get("inference_steps", True) and video_num_inference_steps is not None:
                values += [video_num_inference_steps]
                labels += ["Num Inference steps"]
            video_negative_prompt = configs.get("negative_prompt", "")
            if len(video_negative_prompt) > 0:
                values += [video_negative_prompt]
                labels += ["Negative Prompt"]        
            video_NAG_scale = configs.get("NAG_scale", None)
            if video_NAG_scale is not None and video_NAG_scale > 1: 
                video_NAG_tau = configs.get("NAG_tau", None)
                video_NAG_alpha = configs.get("NAG_alpha", None)
                values += [f"scale={video_NAG_scale}, tau={video_NAG_tau}, alpha={video_NAG_alpha}"]
                labels += ["NAG"]      
            video_self_refiner_setting = configs.get("self_refiner_setting", 0)
            if video_self_refiner_setting > 0:  
                video_self_refiner_plan = configs.get('self_refiner_plan','')
                if len(video_self_refiner_plan)==0: video_self_refiner_plan ='default'
                values += [f"Norm P{video_self_refiner_setting}, Plan='{video_self_refiner_plan}', Uncertainty={configs.get('self_refiner_f_uncertainty',0.0)}, Certain Percentage='{configs.get('self_refiner_certain_percentage', 0.999)} "]
                # values += [f"Norm P{video_self_refiner_setting}, Plan='{video_self_refiner_plan}'"]
                labels += ["Self Refiner"]      
            video_apg_switch = configs.get("apg_switch", None)
            if video_apg_switch is not None and video_apg_switch != 0: 
                values += ["on"]
                labels += ["APG"]      
            video_motion_amplitude = configs.get("motion_amplitude", 1.)
            if  video_motion_amplitude != 1: 
                values += [video_motion_amplitude]
                labels += ["Motion Amplitude"]
            control_net_weight_name = model_def.get("control_net_weight_name", "")
            control_net_weight = ""
            if len(control_net_weight_name):
                video_control_net_weight = configs.get("control_net_weight", 1)
                if len(filter_letters(video_video_prompt_type, video_guide_processes))> 1:
                    video_control_net_weight2 = configs.get("control_net_weight2", 1)
                    control_net_weight = f"{control_net_weight_name} #1={video_control_net_weight}, {control_net_weight_name} #2={video_control_net_weight2}"
                else:
                    control_net_weight = f"{control_net_weight_name}={video_control_net_weight}"
            control_net_weight_alt_name = model_def.get("control_net_weight_alt_name", "")
            if len(control_net_weight_alt_name) >0:
                if len(control_net_weight): control_net_weight += ", "
                control_net_weight += control_net_weight_alt_name + "=" + str(configs.get("control_net_weight_alt", 1))
            if len(control_net_weight) > 0: 
                values += [control_net_weight]
                labels += ["Control Net Weights"]      

            audio_scale_name = model_def.get("audio_scale_name", "")
            if len(audio_scale_name) > 0 and any_letters(video_audio_prompt_type,"AB"):
                values += [configs.get("audio_scale", 1)]
                labels += [audio_scale_name]

            video_skip_steps_cache_type = configs.get("skip_steps_cache_type", "")
            video_skip_steps_multiplier = configs.get("skip_steps_multiplier", 0)
            video_skip_steps_cache_start_step_perc = configs.get("skip_steps_start_step_perc", 0)
            if len(video_skip_steps_cache_type) > 0:
                video_skip_steps_cache = "TeaCache" if video_skip_steps_cache_type == "tea" else "MagCache"
                video_skip_steps_cache += f" x{video_skip_steps_multiplier }"
                if video_skip_steps_cache_start_step_perc >0:  video_skip_steps_cache += f", Start from {video_skip_steps_cache_start_step_perc}%"
                values += [ video_skip_steps_cache ]
                labels += [ "Skip Steps" ]

            values += pp_values
            labels += pp_labels

            if len(video_activated_loras_str) > 0:
                values += [video_activated_loras_str]
                labels += ["Loras"] 
            if nb_audio_tracks  > 0:
                values +=[nb_audio_tracks]
                labels +=["Nb Audio Tracks"]
            values += [ video_creation_date, video_generation_time ]
            labels += [ "Creation Date", "Generation Time" ]
        labels = [label for value, label in zip(values, labels) if value is not None]
        values = [value for value in values if value is not None]

        table_style = """<STYLE>
            #video_info, #video_info TR, #video_info TD {
            background-color: transparent; 
            color: inherit; 
            padding: 3px 4px;
            border:0px !important;
            font-size:11px;
            }
            </STYLE>
        """
        rows = [f"<TR><TD style='text-align: right;' WIDTH=1% NOWRAP VALIGN=TOP>{label}</TD><TD><B>{value}</B></TD></TR>" for label, value in zip(labels, values)]
        html_content = f"{table_style}<TABLE ID=video_info WIDTH=100%>" + "".join(rows) + "</TABLE>"
    else:
        html_content =  get_default_video_info()
    visible= len(files) > 0 
    return choice if source=="video" else gr.update(), html_content, gr.update(visible=visible and is_video) , gr.update(visible=visible and is_image), gr.update(visible=visible and is_audio), gr.update(visible=visible and is_deleted and source=="video"), gr.update(visible=visible and is_deleted and source=="audio"), gr.update(visible=visible and is_video) , gr.update(visible=visible and is_video) 

def convert_image(image):

    from PIL import ImageOps
    from typing import cast
    if isinstance(image, str):
        image = _open_image_input(image)
    image = image.convert('RGB')
    return cast(Image, ImageOps.exif_transpose(image))

def get_resampled_video(video_in, start_frame, max_frames, target_fps, bridge='torch', hdr_linear=False):
    if hdr_linear:
        return decode_video_frames_ffmpeg(video_in, start_frame, max_frames, target_fps=target_fps, bridge=bridge, hdr_linear=True)
    return get_resampled_video_transparent(video_in, start_frame, max_frames, target_fps, bridge)

def _virtual_media_has_hdr_flag(value):
    spec = parse_virtual_media_path(value) if isinstance(value, str) else None
    extras = {str(k).strip().lower(): str(v).strip().lower() for k, v in (spec.extras if spec is not None else ())}
    if extras.get("hdr") in {"1", "true", "yes"}:
        return True
    entry = get_virtual_media_entry(value) if isinstance(value, str) else None
    return bool(isinstance(entry, dict) and entry.get("hdr"))

def _video_input_is_hdr(value):
    if _virtual_media_has_hdr_flag(value):
        return True
    if not isinstance(value, str):
        return False
    metadata = probe_video_stream_metadata(value)
    return bool(metadata and (metadata.get("hdr") or metadata.get("needs_tonemap")))

# def get_resampled_video(video_in, start_frame, max_frames, target_fps):
#     from torchvision.io import VideoReader
#     import torch
#     from shared.utils.utils import resample

#     vr = VideoReader(video_in, "video")
#     meta = vr.get_metadata()["video"]

#     fps = round(float(meta["fps"][0]))
#     duration_s = float(meta["duration"][0])
#     num_src_frames = int(round(duration_s * fps))  # robust length estimate

#     if max_frames < 0:
#         max_frames = max(int(num_src_frames / fps * target_fps + max_frames), 0)

#     frame_nos = resample(
#         fps, num_src_frames,
#         max_target_frames_count=max_frames,
#         target_fps=target_fps,
#         start_target_frame=start_frame
#     )
#     if len(frame_nos) == 0:
#         return torch.empty((0,))  # nothing to return

#     target_ts = [i / fps for i in frame_nos]

#     # Read forward once, grabbing frames when we pass each target timestamp
#     frames = []
#     vr.seek(target_ts[0])
#     idx = 0
#     tol = 0.5 / fps  # half-frame tolerance
#     for frame in vr:
#         t = float(frame["pts"])       # seconds
#         if idx < len(target_ts) and t + tol >= target_ts[idx]:
#             frames.append(frame["data"].permute(1,2,0))  # Tensor [H, W, C]
#             idx += 1
#             if idx >= len(target_ts):
#                 break

#     return frames


def get_preprocessor(process_type, inpaint_color, pre_video_guide=None):
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




def extract_faces_from_video_with_mask(input_video_path, input_mask_path, max_frames, start_frame, target_fps, size = 512):
    if not input_video_path or max_frames <= 0:
        return None, None
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    any_mask = input_mask_path != None
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if len(video) == 0: return None
    frame_height, frame_width, _ = video[0].shape
    num_frames = len(video)
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)
        num_frames = min(num_frames, len(mask_video))
    if num_frames == 0: return None
    video = video[:num_frames]
    if any_mask:
        mask_video = mask_video[:num_frames]

    from preprocessing.face_preprocessor  import FaceProcessor 
    face_processor = FaceProcessor()

    face_list = []
    for frame_idx in range(num_frames):
        frame = video[frame_idx].cpu().numpy() 
        # video[frame_idx] = None
        if any_mask:
            mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) 
            # mask_video[frame_idx] = None
            if (frame_width, frame_height) != mask.size:
                mask = mask.resize((frame_width, frame_height), resample=Image.Resampling.LANCZOS)
            mask = np.array(mask)
            alpha_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            alpha_mask[mask > 127] = 1
            frame = frame * alpha_mask
        frame = Image.fromarray(frame)
        face = face_processor.process(frame, resize_to=size)
        face_list.append(face)

    face_processor = None
    gc.collect()
    torch.cuda.empty_cache()

    face_tensor= torch.tensor(np.stack(face_list, dtype= np.float32) / 127.5 - 1).permute(-1, 0, 1, 2 ) # t h w c -> c t h w
    if pad_frames > 0:
        face_tensor= torch.cat([face_tensor[:, -1:].expand(-1, pad_frames, -1, -1), face_tensor ], dim=2)
        
    if args.save_masks:
        from preprocessing.dwpose.pose import save_one_video
        saved_faces_frames = [np.array(face) for face in face_list ]
        save_one_video(f"faces.mp4", saved_faces_frames, fps=target_fps, quality=8, macro_block_size=None)
    return face_tensor


def preprocess_video_with_mask(pre_video_guide, input_video_path, input_mask_path, height, width,  max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size= 16, expand_scale = 2, process_type = "inpaint", process_type2 = None, to_bbox = False, RGB_Mask = False, negate_mask = False, process_outside_mask = None, inpaint_color = 127, outpainting_dims = None, outpainting_ratio = "", proc_no = 1):

    def mask_to_xyxy_box(mask):
        rows, cols = np.where(mask == 255)
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
    inpaint_color = parse_guide_inpaint_color(inpaint_color)
    inpaint_color = to_rgb_tensor(inpaint_color, device="cpu", dtype=torch.uint8)
    inpaint_color_np = tuple(int(v) for v in inpaint_color.view(-1).tolist())
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    if not input_video_path or max_frames <= 0:
        return None, None
    any_mask = input_mask_path != None
    pose_special = "pose" in process_type
    any_identity_mask = False
    if process_type == "identity":
        any_identity_mask = True
        negate_mask = False
        process_outside_mask = None
    if process_type == "pose_align" and any_mask:
        process_outside_mask = None
    preproc = get_preprocessor(process_type, inpaint_color, pre_video_guide=pre_video_guide)
    preproc2 = None
    if process_type2 != None:
        preproc2 = get_preprocessor(process_type2, inpaint_color, pre_video_guide=pre_video_guide) if process_type != process_type2 else preproc
    if process_outside_mask == process_type :
        preproc_outside = preproc
    elif preproc2 != None and process_outside_mask == process_type2 :
        preproc_outside = preproc2
    else:
        preproc_outside = get_preprocessor(process_outside_mask, inpaint_color)
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)

    if len(video) == 0 or any_mask and len(mask_video) == 0:
        return None, None
    if fit_crop and outpainting_dims != None:
        fit_crop = False
        fit_canvas = 0 if fit_canvas is not None else None

    frame_height, frame_width, _ = video[0].shape

    source_frame_height, source_frame_width = frame_height, frame_width
    if outpainting_dims != None:
        if fit_canvas != None:
            frame_height, frame_width = get_outpainting_full_area_dimensions(frame_height, frame_width, outpainting_dims, outpainting_ratio)
        else:
            frame_height, frame_width = height, width

    if fit_canvas != None:
        height, width = calculate_new_dimensions(height, width, frame_height, frame_width, fit_into_canvas = fit_canvas, block_size = block_size)

    if outpainting_dims != None:
        final_height, final_width = height, width
        height, width, margin_top, margin_left = get_outpainting_frame_location(final_height, final_width, outpainting_dims, 1, outpainting_ratio, source_frame_height, source_frame_width)

    if any_mask:
        num_frames = min(len(video), len(mask_video))
    else:
        num_frames = len(video)

    if any_identity_mask:
        any_mask = True

    proc_list =[]
    proc_list_outside =[]
    proc_mask = []

    # for frame_idx in range(num_frames):
    def prep_prephase(frame_idx):
        frame = Image.fromarray(video[frame_idx].cpu().numpy()) #.asnumpy()
        if fit_crop:
            frame = rescale_and_crop(frame, width, height)
        else:
            frame = frame.resize((width, height), resample=Image.Resampling.LANCZOS) 
        frame = np.array(frame) 
        if any_mask:
            if any_identity_mask:
                mask = np.full( (height, width, 3), 0, dtype= np.uint8)
            else:
                mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) #.asnumpy()
                if fit_crop:
                    mask = rescale_and_crop(mask, width, height)
                else:
                    mask = mask.resize((width, height), resample=Image.Resampling.LANCZOS) 
                mask = np.array(mask)

            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
            original_mask = mask.copy()
            if expand_scale != 0:
                kernel_size = abs(expand_scale)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                op_expand = cv2.dilate if expand_scale > 0 else cv2.erode
                mask = op_expand(mask, kernel, iterations=3)

            if to_bbox and np.sum(mask == 255) > 0 : #or True 
                x0, y0, x1, y1 = mask_to_xyxy_box(mask)
                mask = mask * 0
                mask[y0:y1, x0:x1] = 255
            if negate_mask:
                mask = 255 - mask
                if pose_special:
                    original_mask = 255 - original_mask

        if pose_special and any_mask:            
            target_frame = np.where(original_mask[..., None], frame, 0) 
        else:
            target_frame = frame 

        if any_mask:
            return (target_frame, frame, mask) 
        else:
            return (target_frame, None, None)
    max_workers = get_default_workers()
    proc_lists = process_images_multithread(prep_prephase, [frame_idx for frame_idx in range(num_frames)], "prephase", wrap_in_list= False, max_workers=max_workers, in_place= True)
    proc_list, proc_list_outside, proc_mask = [None] * len(proc_lists), [None] * len(proc_lists), [None] * len(proc_lists)
    for frame_idx, frame_group in enumerate(proc_lists): 
        proc_list[frame_idx], proc_list_outside[frame_idx], proc_mask[frame_idx] = frame_group
    prep_prephase = None
    video = None
    mask_video = None

    if preproc2 != None:
        proc_list2 = process_images_multithread(preproc2, proc_list, process_type2, max_workers=max_workers)
        #### to be finished ...or not
    proc_list = process_images_multithread(preproc, proc_list, process_type, max_workers=max_workers)
    if any_mask:
        proc_list_outside = process_images_multithread(preproc_outside, proc_list_outside, process_outside_mask, max_workers=max_workers)
    else:
        proc_list_outside = proc_mask = len(proc_list) * [None]

    masked_frames = []
    masks = []
    for frame_no, (processed_img, processed_img_outside, mask) in enumerate(zip(proc_list, proc_list_outside, proc_mask)):
        if isinstance(processed_img, (list, tuple)):
            processed_img = np.full((height, width, 3), processed_img, dtype=np.uint8)
        if isinstance(processed_img_outside, (list, tuple)):
            processed_img_outside = np.full((height, width, 3), processed_img_outside, dtype=np.uint8)
        if any_mask :
            if process_type == "pose_align":
                masked_frame = processed_img
                mask = np.full_like(mask, 0)
            else:
                masked_frame = np.where(mask[..., None], processed_img, processed_img_outside)
            if process_outside_mask != None:
                mask = np.full_like(mask, 255)
            mask = torch.from_numpy(mask)
            if RGB_Mask:
                mask =  mask.unsqueeze(-1).repeat(1,1,3)
            if outpainting_dims != None:
                full_frame= torch.full( (final_height, final_width, mask.shape[-1]), 255, dtype= torch.uint8, device= mask.device)
                full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = mask
                mask = full_frame 
            masks.append(mask[:, :, 0:1].clone())
        else:
            masked_frame = processed_img

        if isinstance(masked_frame, (int, float, np.integer)) or (isinstance(masked_frame, (list, tuple)) and len(masked_frame) == 3):
            masked_frame= np.full( (height, width, 3), inpaint_color_np, dtype= np.uint8)

        masked_frame = torch.from_numpy(masked_frame)
        if masked_frame.shape[-1] == 1:
            masked_frame =  masked_frame.repeat(1,1,3).to(torch.uint8)

        if outpainting_dims != None:
            color = inpaint_color.to(masked_frame.device).view(1, 1, 3)
            full_frame = color.expand(final_height, final_width, masked_frame.shape[-1]).clone()
            full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = masked_frame
            masked_frame = full_frame 

        masked_frames.append(masked_frame)
        proc_list[frame_no] = proc_list_outside[frame_no] = proc_mask[frame_no] = None


    # if args.save_masks:
    #     from preprocessing.dwpose.pose import save_one_video
    #     saved_masked_frames = [mask.cpu().numpy() for mask in masked_frames ]
    #     save_one_video(f"masked_frames{'' if proc_no==1 else str(proc_no)}.mp4", saved_masked_frames, fps=target_fps, quality=8, macro_block_size=None)
    #     if any_mask:
    #         saved_masks = [mask.cpu().numpy() for mask in masks ]
    #         save_one_video("masks.mp4", saved_masks, fps=target_fps, quality=8, macro_block_size=None)
    preproc = None
    preproc_outside = None
    gc.collect()
    torch.cuda.empty_cache()
    if pad_frames > 0:
        masked_frames = masked_frames[0] * pad_frames + masked_frames
        if any_mask: masked_frames = masks[0] * pad_frames + masks
    masked_frames = torch.stack(masked_frames).permute(-1,0,1,2).float().div_(127.5).sub_(1.)
    masks = torch.stack(masks).permute(-1,0,1,2).float().div_(255) if any_mask else None

    return masked_frames, masks

def preprocess_video(height, width, video_in, max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size = 16, preserve_hdr = False):

    hdr_input = bool(preserve_hdr and _video_input_is_hdr(video_in))
    frames_list = get_resampled_video(video_in, start_frame, max_frames, target_fps, hdr_linear=hdr_input)

    if len(frames_list) == 0:
        return None
    frames_list = list(frames_list)

    if fit_canvas == None or fit_crop:
        new_height = height
        new_width = width
    else:
        frame_height, frame_width, _ = frames_list[0].shape
        if fit_canvas :
            scale1  = min(height / frame_height, width /  frame_width)
            scale2  = min(height / frame_width, width /  frame_height)
            scale = max(scale1, scale2)
        else:
            scale =   ((height * width ) /  (frame_height * frame_width))**(1/2)

        new_height = round(frame_height * scale / block_size) * block_size
        new_width = round(frame_width * scale / block_size) * block_size

    def resize_hdr_frame(frame):
        frame_t = frame.detach().cpu().to(dtype=torch.float32) if torch.is_tensor(frame) else torch.from_numpy(np.asarray(frame, dtype=np.float32))
        if frame_t.ndim == 3 and frame_t.shape[0] in (1, 3, 4) and frame_t.shape[-1] not in (1, 3, 4):
            frame_t = frame_t.permute(1, 2, 0)
        frame_t = frame_t[..., :3].permute(2, 0, 1).unsqueeze(0)
        if fit_crop:
            src_h, src_w = int(frame_t.shape[2]), int(frame_t.shape[3])
            scale = max(new_height / max(1, src_h), new_width / max(1, src_w))
            resize_h = max(1, int(round(src_h * scale)))
            resize_w = max(1, int(round(src_w * scale)))
            frame_t = torch.nn.functional.interpolate(frame_t, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
            top = max(0, (resize_h - new_height) // 2)
            left = max(0, (resize_w - new_width) // 2)
            frame_t = frame_t[:, :, top:top + new_height, left:left + new_width]
        else:
            frame_t = torch.nn.functional.interpolate(frame_t, size=(new_height, new_width), mode="bilinear", align_corners=False)
        return frame_t[0].permute(1, 2, 0).contiguous()

    def resize_frame(frame):
        if hdr_input:
            return resize_hdr_frame(frame)
        if torch.is_tensor(frame):
            arr = frame.cpu().numpy()
        else:
            arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        if fit_crop:
            img = rescale_and_crop(img, new_width, new_height)
        else:
            img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(img))

    frames_list = process_images_multithread(
        resize_frame,
        frames_list,
        "upsample",
        wrap_in_list=False,
        max_workers=get_default_workers(),
        in_place=True,
    )

    # from preprocessing.dwpose.pose import save_one_video
    # save_one_video("test.mp4", frames_list, fps=8, quality=8, macro_block_size=None)

    return torch.stack(frames_list) 

 
def parse_keep_frames_video_guide(keep_frames, video_length):
        
    def absolute(n):
        if n==0:
            return 0
        elif n < 0:
            return max(0, video_length + n)
        else:
            return min(n-1, video_length-1)
    keep_frames = keep_frames.strip()
    if len(keep_frames) == 0:
        return [True] *video_length, "" 
    frames =[False] *video_length
    error = ""
    sections = keep_frames.split(" ")
    for section in sections:
        section = section.strip()
        if ":" in section:
            parts = section.split(":")
            if not is_integer(parts[0]):
                error =f"Invalid integer {parts[0]}"
                break
            start_range = absolute(int(parts[0]))
            if not is_integer(parts[1]):
                error =f"Invalid integer {parts[1]}"
                break
            end_range = absolute(int(parts[1]))
            for i in range(start_range, end_range + 1):
                frames[i] = True
        else:
            if not is_integer(section) or int(section) == 0:
                error =f"Invalid integer {section}"
                break
            index = absolute(int(section))
            frames[index] = True

    if len(error ) > 0:
        return [], error
    for i in range(len(frames)-1, 0, -1):
        if frames[i]:
            break
    frames= frames[0: i+1]
    return  frames, error


def perform_temporal_upsampling(sample, previous_last_frame, temporal_upsampling, fps):
    exp = 0
    if temporal_upsampling == "rife2":
        exp = 1
    elif temporal_upsampling == "rife4":
        exp = 2
    output_fps = fps
    if exp == 0:
        return sample, previous_last_frame, output_fps
    rife_version = server_config.get("rife_version", "v4")
    rife_model_path = None
    if rife_version == "v4":
        rife_model_path = fl.locate_file(RIFE_V4_FILENAME)
    else:
        rife_model_path = fl.locate_file(RIFE_V3_FILENAME)
    if previous_last_frame is not None and previous_last_frame.dtype != sample.dtype:
        if sample.dtype == torch.uint8:
            previous_last_frame = _video_tensor_to_uint8_chunk_inplace(previous_last_frame)
        else:
            previous_last_frame = previous_last_frame.float().div_(127.5).sub_(1.0)
    if exp > 0: 
        from postprocessing.rife.inference import temporal_interpolation
        if previous_last_frame != None:
            sample = torch.cat([previous_last_frame, sample], dim=1)
            previous_last_frame = sample[:, -1:].clone()
            sample = temporal_interpolation(rife_model_path, sample, exp, device=processing_device, rife_version=rife_version)
            sample = sample[:, 1:]
        else:
            sample = temporal_interpolation(rife_model_path, sample, exp, device=processing_device, rife_version=rife_version)
            previous_last_frame = sample[:, -1:].clone()

        output_fps = output_fps * 2**exp
    return sample, previous_last_frame, output_fps 


def perform_spatial_upsampling(sample, spatial_upsampling):
    from shared.utils.utils import resize_lanczos 
    if spatial_upsampling == "vae2":
        return sample
    method = None
    if spatial_upsampling == "vae1":
        scale = 0.5
        method = Image.Resampling.BICUBIC
    elif spatial_upsampling == "lanczos1.5":
        scale = 1.5
    else:
        scale = 2
    h, w = sample.shape[-2:]
    h *= scale
    h = round(h/16) * 16
    w *= scale
    w = round(w/16) * 16
    h = int(h)
    w = int(w)
    frames_to_upsample = [sample[:, i] for i in range( sample.shape[1]) ] 
    if sample.dtype == torch.uint8:
        resample = Image.Resampling.LANCZOS if method is None else method
        def upsample_frames(frame):
            np_frame = frame.permute(1, 2, 0).cpu().numpy()
            if np_frame.shape[2] == 1:
                np_frame = np_frame[:, :, 0]
            img = Image.fromarray(np_frame)
            img = img.resize((w, h), resample=resample)
            out = np.array(img)
            if out.ndim == 2:
                out = out[:, :, None]
            out = torch.from_numpy(out).permute(2, 0, 1).to(torch.uint8)
            return out.unsqueeze(1)
    else:
        def upsample_frames(frame):
            return resize_lanczos(frame, h, w, method).unsqueeze(1)
    sample = torch.cat(process_images_multithread(upsample_frames, frames_to_upsample, "upsample", wrap_in_list = False, max_workers=get_default_workers(), in_place=True), dim=1)
    frames_to_upsample = None
    return sample 


def any_audio_track(model_type):
    model_def = get_model_def(model_type)
    if not model_def:
        return False
    return ( model_def.get("returns_audio", False) or model_def.get("any_audio_prompt", False) )

def get_available_filename(target_path, video_source, suffix = "", force_extension = None):
    name, extension =  os.path.splitext(os.path.basename(strip_virtual_media_suffix(video_source)))
    if force_extension != None:
        extension = force_extension
    name+= suffix
    full_path= os.path.join(target_path, f"{name}{extension}")
    if not os.path.exists(full_path):
        return full_path
    counter = 2
    while True:
        full_path= os.path.join(target_path, f"{name}({counter}){extension}")
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def set_seed(seed):
    import random
    seed = random.randint(0, 999999999) if seed == None or seed < 0 else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def edit_video(
                send_cmd,
                state,
                mode,
                video_source,
                seed,   
                temporal_upsampling,
                spatial_upsampling,
                film_grain_intensity,
                film_grain_saturation,
                MMAudio_setting,
                MMAudio_prompt,
                MMAudio_neg_prompt,
                repeat_generation,
                audio_source,
                **kwargs
                ):



    gen = get_gen_info(state)

    if gen.get("abort", False): return 
    abort = False
		
		
    MMAudio_setting = MMAudio_setting or 0
    configs, _ , _ = get_settings_from_file(state, video_source, False, False, False)
    if configs == None: configs = { "type" : get_model_record("Post Processing") }

    has_already_audio = False
    audio_tracks = []
    if MMAudio_setting == 0:
        audio_tracks, audio_metadata  = extract_audio_tracks(video_source)
        has_already_audio = len(audio_tracks) > 0
    
    if audio_source is not None:
        audio_tracks = [audio_source]

    with lock:
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]



    seed = set_seed(seed)

    from shared.utils.utils import get_video_info
    fps, width, height, frames_count = get_video_info(video_source)        
    frames_count = min(frames_count, max_source_video_frames)
    sample = None

    if mode == "edit_postprocessing":
        if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 or film_grain_intensity > 0:                
            send_cmd("progress", [0, get_latest_status(state,"Upsampling" if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 else "Adding Film Grain"  )])
            sample = get_resampled_video(video_source, 0, max_source_video_frames, fps)
            sample = sample.permute(-1,0,1,2)
            frames_count = sample.shape[1] 

        output_fps  = round(fps)
        if len(temporal_upsampling) > 0:
            sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, None, temporal_upsampling, fps)
            configs["temporal_upsampling"] = temporal_upsampling
            frames_count = sample.shape[1] 


        if len(spatial_upsampling) > 0:
            sample = perform_spatial_upsampling(sample, spatial_upsampling )
            configs["spatial_upsampling"] = spatial_upsampling

        if film_grain_intensity > 0:
            from postprocessing.film_grain import add_film_grain
            sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
            configs["film_grain_intensity"] = film_grain_intensity
            configs["film_grain_saturation"] = film_grain_saturation
    else:
        output_fps  = round(fps)

    mmaudio_enabled, mmaudio_mode, mmaudio_persistence, mmaudio_model_name, mmaudio_model_path = get_mmaudio_settings(server_config)
    any_mmaudio = MMAudio_setting != 0 and mmaudio_enabled and frames_count >=output_fps
    if any_mmaudio: download_mmaudio()

    tmp_path = None
    any_change = False
    if sample != None:
        video_path =get_available_filename(save_path, video_source, "_tmp") if any_mmaudio or has_already_audio else get_available_filename(save_path, video_source, "_post")  
        save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type= server_config.get("video_output_codec", None), container=server_config.get("video_container", "mp4"))

        if any_mmaudio or has_already_audio: tmp_path = video_path
        any_change = True
    else:
        video_path = video_source

    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    any_change_initial = any_change
    while not gen.get("abort", False): 
        any_change = any_change_initial
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation         
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        suffix =  "" if "_post" in video_source else "_post"

        if audio_source is not None:
            audio_prompt_type = configs.get("audio_prompt_type", "")
            if not "T" in audio_prompt_type:audio_prompt_type += "T"
            configs["audio_prompt_type"] = audio_prompt_type
            any_change = True

        if any_mmaudio:
            send_cmd("progress", [0, get_latest_status(state,"MMAudio Soundtrack Generation")])
            from postprocessing.mmaudio.mmaudio import video_to_audio
            new_video_path = get_available_filename(save_path, video_source, suffix)
            video_to_audio(video_path, prompt = MMAudio_prompt, negative_prompt = MMAudio_neg_prompt, seed = seed, num_steps = 25, cfg_strength = 4.5, duration= frames_count /output_fps, save_path = new_video_path , persistent_models = mmaudio_persistence == MMAUDIO_PERSIST_RAM, verboseLevel = verbose_level, model_name = mmaudio_model_name, model_path = mmaudio_model_path)
            configs["MMAudio_setting"] = MMAudio_setting
            configs["MMAudio_prompt"] = MMAudio_prompt
            configs["MMAudio_neg_prompt"] = MMAudio_neg_prompt
            configs["MMAudio_seed"] = seed
            any_change = True
        elif len(audio_tracks) > 0:
            # combine audio files and new video file
            new_video_path = get_available_filename(save_path, video_source, suffix)
            combine_video_with_audio_tracks(video_path, audio_tracks, new_video_path, audio_metadata=audio_metadata)
        else:
            new_video_path = video_path
        if tmp_path != None:
            os.remove(tmp_path)

        if any_change:
            if mode == "edit_remux":
                print(f"Remuxed Video saved to Path: "+ new_video_path)
            else:
                print(f"Postprocessed video saved to Path: "+ new_video_path)
            with lock:
                file_list.append(new_video_path)
                file_settings_list.append(configs)

            if configs != None:
                from shared.utils.video_metadata import extract_source_images, save_video_metadata
                temp_images_path = get_available_filename(save_path, video_source, force_extension= ".temp")
                embedded_images = extract_source_images(video_source, temp_images_path)
                save_video_metadata(new_video_path, configs, embedded_images, verbose_level=verbose_level)
                if os.path.isdir(temp_images_path):
                    shutil.rmtree(temp_images_path, ignore_errors= True)
            gen["last_was_audio"] = False
            send_cmd("output")
            seed = set_seed(-1)
    if has_already_audio:
        cleanup_temp_audio_files(audio_tracks)
    clear_status(state)

def get_overridden_attention(model_type):
    model_def = get_model_def(model_type)
    override_attention = model_def.get("attention", None)
    if override_attention is None: return None
    gpu_version = gpu_major * 10 + gpu_minor
    attention_list = match_nvidia_architecture(override_attention, gpu_version) 
    if len(attention_list ) == 0: return None
    override_attention = attention_list[0]
    if override_attention is not None and override_attention not in attention_modes_supported: return None
    return override_attention

def get_transformer_loras(model_type):
    model_def = get_model_def(model_type)
    transformer_loras_filenames = get_model_recursive_prop(model_type, "loras", return_list=True)
    lora_dir = get_lora_dir(model_type)
    transformer_loras_filenames = [ os.path.join(lora_dir, os.path.basename(filename)) for filename in transformer_loras_filenames]
    transformer_loras_multipliers = get_model_recursive_prop(model_type, "loras_multipliers", return_list=True) + [1.] * len(transformer_loras_filenames)
    transformer_loras_multipliers = transformer_loras_multipliers[:len(transformer_loras_filenames)]
    return transformer_loras_filenames, transformer_loras_multipliers

class DynamicClass:
    def __init__(self, **kwargs):
        self._data = {}
        # Preassign default properties from kwargs
        for key, value in kwargs.items():
            self._data[key] = value
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
    
    def assign(self, **kwargs):
        """Assign multiple properties at once"""
        for key, value in kwargs.items():
            self._data[key] = value
        return self  # For method chaining
    
    def update(self, dict):
        """Alias for assign() - more dict-like"""
        return self.assign(**dict)

def process_prompt_enhancer(model_type, model_def, prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, audio_only, seed, prompt_enhancer_instructions = None, text_encoder_max_tokens = 512, enhancer_kwargs = None ):
    global enhancer_offloadobj
    prompt_enhancer_mode = str(prompt_enhancer or "")
    prompt_enhancer_instructions, text_encoder_max_tokens = resolve_prompt_enhancer_settings(
        model_type,
        model_def,
        prompt_enhancer_mode,
        is_image,
        prompt_enhancer_instructions=prompt_enhancer_instructions,
        text_encoder_max_tokens=text_encoder_max_tokens,
        enhancer_kwargs = enhancer_kwargs,
    )

    from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt
    prompt_images = []
    if "I" in prompt_enhancer_mode:
        if image_start != None:
            if not isinstance(image_start, list): image_start= [image_start] 
            prompt_images += image_start[:1]
        if original_image_refs != None:
            prompt_images += original_image_refs[:1]
    prompt_images = [_open_image_input(img) if isinstance(img, str) else img for img in prompt_images]
    if len(original_prompts) == 0 and "T" not in prompt_enhancer_mode:
        return None
    else:
        import secrets
        enhancer_temperature = server_config.get("prompt_enhancer_temperature", 0.6)
        enhancer_top_p = server_config.get("prompt_enhancer_top_p", 0.9)
        randomize_seed = server_config.get("prompt_enhancer_randomize_seed", True)
        if randomize_seed:
            enhancer_seed = secrets.randbits(32)
        else:
            enhancer_seed = seed if seed is not None and seed >= 0 else 0
        post_image_caption_hook = None
        if len(prompt_images) > 0 and enhancer_offloadobj is not None:
            if hasattr(prompt_enhancer_image_caption_model, "vision_tower_model") and hasattr(prompt_enhancer_llm_model, "generate_messages"):
                post_image_caption_hook = enhancer_offloadobj.unload_all
        prompts = generate_cinematic_prompt(
            prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer,
            original_prompts if "T" in prompt_enhancer_mode else ["an image"],
            prompt_images if len(prompt_images) > 0 else None,
            video_prompt = not is_image,
            text_prompt = audio_only,
            max_new_tokens=text_encoder_max_tokens,
            prompt_enhancer_instructions = prompt_enhancer_instructions,
            do_sample = True,
            temperature = enhancer_temperature,
            top_p = enhancer_top_p,
            seed = enhancer_seed,
            post_image_caption_hook = post_image_caption_hook,
            thinking_enabled = "K" in prompt_enhancer_mode,
        )
        return prompts


def resolve_prompt_enhancer_settings(model_type, model_def, prompt_enhancer_mode, is_image, prompt_enhancer_instructions = None, text_encoder_max_tokens = 512, enhancer_kwargs = None):
    prompt_enhancer_mode = str(prompt_enhancer_mode or "")
    if model_def is None or len(model_type) == 0:
        return prompt_enhancer_instructions, int(text_encoder_max_tokens)

    model_handler = get_model_handler(model_type)
    if hasattr(model_handler, "get_custom_prompt_enhancer_instructions"):
        ret_prompt_enhancer_instructions, ret_text_encoder_max_tokens =  model_handler.get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs)
        if ret_prompt_enhancer_instructions is not None: prompt_enhancer_instructions = ret_prompt_enhancer_instructions 
        if ret_text_encoder_max_tokens is not None: text_encoder_max_tokens = ret_text_encoder_max_tokens

    prompt_enhancer_instructions = model_def.get("image_prompt_enhancer_instructions" if is_image else "video_prompt_enhancer_instructions", prompt_enhancer_instructions)
    text_encoder_max_tokens = model_def.get("image_prompt_enhancer_max_tokens" if is_image else "video_prompt_enhancer_max_tokens", text_encoder_max_tokens)

    if "I" not in prompt_enhancer_mode:
        prompt_profile_id = "0"
        prompt_profile_match = re.search(r"\d", prompt_enhancer_mode)
        if prompt_profile_match is not None:
            prompt_profile_id = prompt_profile_match.group(0)
        prompt_instructions_key = "text_prompt_enhancer_instructions" if prompt_profile_id == "0" else f"text_prompt_enhancer_instructions{prompt_profile_id}"
        prompt_max_tokens_key = "text_prompt_enhancer_max_tokens" if prompt_profile_id == "0" else f"text_prompt_enhancer_max_tokens{prompt_profile_id}"
        prompt_enhancer_instructions = model_def.get(prompt_instructions_key, model_def.get("text_prompt_enhancer_instructions", prompt_enhancer_instructions))
        text_encoder_max_tokens = model_def.get(prompt_max_tokens_key, model_def.get("text_prompt_enhancer_max_tokens", text_encoder_max_tokens))
    return prompt_enhancer_instructions, int(text_encoder_max_tokens)

def exec_prompt_enhancer_engine(state, model_type, model_def, prompt_enhancer_modes, original_prompts, image_start, original_image_refs, is_image, audio_only, seed, progress, override_profile, send_cmd = None, tools = None, enhancer_kwargs = None):
    global enhancer_offloadobj

    assistant_mode = "A" in prompt_enhancer_modes
    if assistant_mode:
        return _deepy.run_assistant_prompt_turn(state, model_def, prompt_enhancer_modes, original_prompts, seed, override_profile=override_profile, send_cmd=send_cmd, tools=tools)

    acquire_GPU_ressources(state, "prompt_enhancer", "Prompt Enhancer")
    try:
        ensure_prompt_enhancer_loaded(override_profile=override_profile, progress=progress)
    except Exception:
        release_GPU_ressources(state, "prompt_enhancer")
        raise

    seed = set_seed(seed)
    num_prompts = len(original_prompts) 

    enhanced_prompts = []
    for i, (one_prompt, one_image) in enumerate(zip(original_prompts, image_start)):
        start_images = [one_image] if one_image is not None else None
        status = f'Please Wait While Enhancing Prompt' if num_prompts==1 else f'Please Wait While Enhancing Prompt #{i+1}'
        progress((i , num_prompts), desc=status, total= num_prompts)

        try:
            enhanced_prompt = process_prompt_enhancer(model_type, model_def, prompt_enhancer_modes, [one_prompt],  start_images, original_image_refs, is_image, audio_only, seed, enhancer_kwargs = enhancer_kwargs)
        except Exception as e:
            unload_prompt_enhancer_runtime()
            enhancer_offloadobj.unload_all()
            release_GPU_ressources(state, "prompt_enhancer")
            print(traceback.format_exc())
            raise gr.Error(e)
        enhanced_prompts.append(enhanced_prompt)

    unload_prompt_enhancer_runtime()
    enhancer_offloadobj.unload_all()

    release_GPU_ressources(state, "prompt_enhancer")
    return enhanced_prompts

def enhance_prompt(state, prompt, prompt_enhancer, multi_images_gen_type, multi_prompts_gen_type, override_profile, video_prompt_type, image_prompt_type, audio_prompt_type, progress=gr.Progress()):
    model_type = get_state_model_type(state)
    inputs = get_model_settings(state, model_type)
    original_prompts = inputs["prompt"]

    original_prompts, errors = prompt_parser.process_template(
        original_prompts,
        keep_comments=True,
        keep_empty_lines="P" in multi_prompts_gen_type or prompt_parser.PROMPT_UNIT_PREFIX in original_prompts,
    )
    if len(errors) > 0:
        gr.Info("Error processing prompt template: " + errors)
        return gr.update(), gr.update()
    original_prompts = prompt_parser.split_prompt_units(original_prompts, multi_prompts_gen_type, originals=True)
    num_prompts = len(original_prompts) 
    image_prompt_type = inputs["image_prompt_type"]
    video_prompt_type = inputs["video_prompt_type"]
    image_start = inputs["image_start"] if "S" in image_prompt_type else None
    if image_start is None:
        image_start = inputs["image_end"] if "E" in image_prompt_type else None
    if image_start is None or not "I" in prompt_enhancer:
        image_start = [None] * num_prompts
    else:
        image_start = [convert_image(img[0]) for img in image_start]
        if len(image_start) == 1:
            image_start = image_start * num_prompts
        else:
            if multi_images_gen_type !=1:
                gr.Info("On Demand Prompt Enhancer with multiple Start Images requires that option 'Match images and text prompts' is set")
                return gr.update(), gr.update()

            if len(image_start) != num_prompts:
                gr.Info("On Demand Prompt Enhancer supports only mutiple Start Images if their number matches the number of Text Prompts")
                return gr.update(), gr.update()

    original_image_refs = inputs["image_refs"] if "I" in video_prompt_type else None
    if original_image_refs is not None:
        original_image_refs = [ convert_image(tup[0]) for tup in original_image_refs ]        
    is_image = inputs["image_mode"] > 0
    seed = inputs["seed"]

    model_def = get_model_def(get_state_model_type(state))
    audio_only = model_def.get("audio_only", False)
    enhancer_kwargs = {"image_prompt_type":  image_prompt_type, "video_prompt_type":  video_prompt_type, "audio_prompt_type":  audio_prompt_type}
    enhanced_prompts = exec_prompt_enhancer_engine(state, model_type, model_def, prompt_enhancer, original_prompts, image_start, original_image_refs, is_image, audio_only, seed, progress, override_profile, enhancer_kwargs = enhancer_kwargs)

    output_prompts = []
    for enhanced_prompt, one_prompt in zip(enhanced_prompts, original_prompts):
        if enhanced_prompt is not None:
            if any_letters(multi_prompts_gen_type, "PG"):
                enhanced_prompt = enhanced_prompt[0]
            else:
                enhanced_prompt = enhanced_prompt[0].replace("\n", " ").replace("\r", "")
            output_prompts.append(enhanced_prompt)

    prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_prompts, original_prompts)
    if num_prompts > 1:
        gr.Info(f'{num_prompts} Prompts have been Enhanced')
    else:
        gr.Info(f'Prompt "{original_prompts[0][:100]}" has been enhanced')
    return prompt, prompt

def parse_guide_inpaint_color(value):
    if isinstance(value, str):
        cleaned = value.strip()
        hex_value = cleaned[1:] if cleaned.startswith("#") else cleaned
        if len(hex_value) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_value):
            return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
        if cleaned.lower().startswith("rgb"):
            cleaned = cleaned[3:]
        for ch in "()[]{}":
            cleaned = cleaned.replace(ch, "")
        cleaned = cleaned.replace(",", " ")
        parts = [p for p in cleaned.split() if p]
        if len(parts) == 3:
            try:
                return tuple(max(0, min(255, int(round(float(p))))) for p in parts)
            except ValueError:
                return 127.5
        return 127.5
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(max(0, min(255, int(round(float(p))))) for p in value)
    return value

def truncate_audio(generated_audio, trim_video_frames_beginning, trim_video_frames_end, video_fps, audio_sampling_rate):
    samples_per_frame = audio_sampling_rate / video_fps
    start = int(trim_video_frames_beginning * samples_per_frame)
    end = generated_audio.shape[0] - int(trim_video_frames_end * samples_per_frame)
    return generated_audio[start:end if end > 0 else None]

def slice_audio_window(audio_path, start_frame, num_frames, fps, output_dir, suffix="", pad_head=True, pad_tail=True):
    import soundfile as sf
    import numpy as np

    start_sec = float(start_frame) / float(fps)
    duration_sec = float(num_frames) / float(fps)

    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = audio_file.samplerate
        channels = audio_file.channels
        total_frames = len(audio_file)
        start_sample = int(round(start_sec * sample_rate))
        pad_start = 0
        if start_sample < 0 and pad_head:
            pad_start = -start_sample
        if start_sample < 0:
            start_sample = 0
        frames_to_read = int(round(duration_sec * sample_rate))
        if start_sample > total_frames:
            data = np.zeros((0, channels), dtype=np.float32)
        else:
            audio_file.seek(min(start_sample, total_frames))
            data = audio_file.read(frames_to_read, dtype="float32", always_2d=True)

    if pad_head and pad_start > 0:
        data = np.concatenate([np.zeros((pad_start, channels), dtype=np.float32), data], axis=0)
    if pad_tail:
        target_frames = (pad_start if pad_head else 0) + frames_to_read
        if data.shape[0] < target_frames:
            pad_end = target_frames - data.shape[0]
            data = np.concatenate([data, np.zeros((pad_end, channels), dtype=np.float32)], axis=0)
    return data, sample_rate


def get_audio_file_sample_rate(audio_path):
    import ffmpeg

    probe = ffmpeg.probe(os.fspath(audio_path))
    audio_stream = next((stream for stream in probe["streams"] if stream.get("codec_type") == "audio"), None)
    if audio_stream is None or not audio_stream.get("sample_rate"):
        raise ValueError(f"Unable to read audio sample rate from {audio_path}")
    return int(audio_stream["sample_rate"])


def resolve_mux_audio_sampling_rate(default_rate, source_audio_metadata=None, audio_paths=None):
    sample_rates = [int(default_rate)]
    for meta in source_audio_metadata or []:
        sample_rate = int(meta.get("sample_rate", 0) or 0)
        if sample_rate > 0:
            sample_rates.append(sample_rate)
    for audio_path in audio_paths or []:
        if audio_path:
            sample_rates.append(get_audio_file_sample_rate(audio_path))
    return max(sample_rates)


def custom_preprocess_video_with_mask(model_handler, base_model_type, pre_video_guide, video_guide, video_mask, height, width, max_frames, start_frame, fit_canvas, fit_crop, target_fps,  block_size, expand_scale, video_prompt_type):
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    max_workers = get_default_workers()

    if not video_guide or max_frames <= 0:
        return None, None, None, None
    video_guide = get_resampled_video(video_guide, start_frame, max_frames, target_fps).permute(-1, 0, 1, 2)
    video_guide = video_guide / 127.5 - 1.
    any_mask = video_mask is not None
    if video_mask is not None:
        video_mask = get_resampled_video(video_mask, start_frame, max_frames, target_fps).permute(-1, 0, 1, 2)
        video_mask = video_mask[:1] / 255.

    # Mask filtering: resize, binarize, expand mask and keep only masked areas of video guide
    if any_mask:
        invert_mask = "N" in video_prompt_type
        import concurrent.futures
        tgt_h, tgt_w = video_guide.shape[2], video_guide.shape[3]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(expand_scale), abs(expand_scale))) if expand_scale != 0 else None
        op = (cv2.dilate if expand_scale > 0 else cv2.erode) if expand_scale != 0 else None
        def process_mask(idx):
            m = (video_mask[0, idx].numpy() * 255).astype(np.uint8)
            if m.shape[0] != tgt_h or m.shape[1] != tgt_w:
                m = cv2.resize(m, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)  # binarize grey values
            if op: m = op(m, kernel, iterations=3)
            if invert_mask:
                return torch.from_numpy((m <= 127).astype(np.float32))
            else:
                return torch.from_numpy((m > 127).astype(np.float32))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            video_mask = torch.stack([f.result() for f in [ex.submit(process_mask, i) for i in range(video_mask.shape[1])]]).unsqueeze(0)
        video_guide = video_guide * video_mask + (-1) * (1-video_mask)

    if video_guide.shape[1] == 0 or any_mask and video_mask.shape[1] == 0:
        return None, None, None, None
    
    video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2  = model_handler.custom_preprocess(base_model_type = base_model_type, pre_video_guide = pre_video_guide, video_guide = video_guide, video_mask = video_mask, height = height, width = width, fit_canvas = fit_canvas , fit_crop = fit_crop, target_fps = target_fps,  block_size = block_size, max_workers = max_workers, expand_scale = expand_scale, video_prompt_type=video_prompt_type)

    # if pad_frames > 0:
    #     masked_frames = masked_frames[0] * pad_frames + masked_frames
    #     if any_mask: masked_frames = masks[0] * pad_frames + masks

    return video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 

def _video_tensor_to_uint8_chunk_inplace(sample, value_range=(-1, 1)):
    if sample.dtype == torch.uint8:
        return sample
    min_val, max_val = value_range
    sample = sample.clamp_(min_val, max_val)
    sample = sample.sub_(min_val).mul_(255.0 / (max_val - min_val)).to(torch.uint8)
    return sample

def get_output_filepath(file_path, is_image, audio_only):
    if is_image:
        base_path = image_save_path
    elif audio_only:
        base_path = audio_save_path
    else:
        base_path = save_path
    return get_available_filename(base_path, file_path)


def record_file_metadata(video_path, configs, is_image, audio_only, gen, embedded_images=None, replace_last_file=False):
    return shared_record_file_metadata(video_path, configs, is_image, audio_only, gen, get_processed_queue=get_processed_queue, metadata_choice=server_config.get("metadata_type", "metadata"), embedded_images=embedded_images, replace_last_file=replace_last_file, lock=lock, verbose_level=verbose_level)


def generate_video(
    task,
    send_cmd,
    client_id,
    image_mode,
    prompt,
    alt_prompt,
    negative_prompt,    
    resolution,
    video_length,
    duration_seconds,
    pause_seconds,
    batch_size,
    seed,
    force_fps,
    num_inference_steps,
    guidance_scale,
    guidance2_scale,
    guidance3_scale,
    switch_threshold,
    switch_threshold2,
    guidance_phases,
    model_switch_phase,
    alt_guidance_scale,
    alt_scale,
    audio_guidance_scale,
    audio_scale,
    flow_shift,
    sample_solver,
    embedded_guidance_scale,
    repeat_generation,
    multi_prompts_gen_type,
    multi_images_gen_type,
    skip_steps_cache_type,
    skip_steps_multiplier,
    skip_steps_start_step_perc,    
    activated_loras,
    loras_multipliers,
    image_prompt_type,
    image_start,
    image_end,
    model_mode,
    video_source,
    keep_frames_video_source,
    input_video_strength,
    video_prompt_type,
    image_refs,
    frames_positions,
    video_guide,
    image_guide,
    keep_frames_video_guide,
    denoising_strength,
    masking_strength,     
    video_guide_outpainting,
    video_guide_outpainting_ratio,
    video_mask,
    image_mask,
    control_net_weight,
    control_net_weight2,
    control_net_weight_alt,
    motion_amplitude,
    mask_expand,
    audio_guide,
    audio_guide2,
    custom_guide,
    audio_source,
    audio_prompt_type,
    speakers_locations,
    sliding_window_size,
    sliding_window_overlap,
    sliding_window_color_correction_strength,
    sliding_window_overlap_noise,
    sliding_window_discard_last_frames,
    image_refs_relative_size,
    remove_background_images_ref,
    temporal_upsampling,
    spatial_upsampling,
    film_grain_intensity,
    film_grain_saturation,
    MMAudio_setting,
    MMAudio_prompt,
    MMAudio_neg_prompt,    
    RIFLEx_setting,
    NAG_scale,
    NAG_tau,
    NAG_alpha,
    perturbation_switch,
    perturbation_layers,
    perturbation_start_perc,
    perturbation_end_perc,
    apg_switch,
    cfg_star_switch,
    cfg_zero_step,
    prompt_enhancer,
    min_frames_if_references,
    override_profile,
    override_attention,
    temperature,
    custom_settings,
    top_p,
    top_k,
    self_refiner_setting,
    self_refiner_plan,
    self_refiner_f_uncertainty,
    self_refiner_certain_percentage,
    output_filename,
    state,
    model_type,
    mode,
    plugin_data=None,
):

    def remove_temp_filenames(temp_filenames_list):
        for temp_filename in temp_filenames_list: 
            if temp_filename!= None and os.path.isfile(temp_filename):
                os.remove(temp_filename)

    def set_progress_status(status):
        phase_text = str(status or "").strip()
        if len(phase_text) == 0:
            return
        gen["progress_phase"] = (phase_text, -1)
        send_cmd("progress", [0, get_latest_status(state, phase_text)])

    global wan_model, offloadobj, reload_needed
    gen = get_gen_info(state)
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    gen["early_stop"] = False
    gen["early_stop_forwarded"] = False
    gen["last_progress_args"] = None
    torch.set_grad_enabled(False) 
    if mode.startswith("edit_"):
        edit_video(send_cmd, state, mode, video_source, seed, temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, MMAudio_setting, MMAudio_prompt, MMAudio_neg_prompt, repeat_generation, audio_source)
        return True

    model_def = get_model_def(model_type) 
    is_image = image_mode > 0
    audio_only = model_def.get("audio_only", False)
    duration_def = model_def.get("duration_slider", None)

    set_video_prompt_type = model_def.get("set_video_prompt_type", None)
    if set_video_prompt_type is not None:
        video_prompt_type = add_to_sequence(video_prompt_type, set_video_prompt_type)
    if is_image:
        if not model_def.get("custom_video_length", False):
            if min_frames_if_references >= 1000:
                video_length = min_frames_if_references - 1000
            else:
                video_length = min_frames_if_references if "I" in video_prompt_type or "V" in video_prompt_type else 1 
    else:
        batch_size = 1
    temp_filenames_list = []

    if image_guide is not None and isinstance(image_guide, Image.Image):
        video_guide = image_guide
        image_guide = None

    if image_mask is not None and isinstance(image_mask, Image.Image):
        video_mask = image_mask
        image_mask = None

    if model_def.get("no_background_removal", False): remove_background_images_ref = 0
    
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)
    block_size = model_def.get("vae_block_size", 16)

    if "P" in preload_model_policy and not "U" in preload_model_policy:
        while wan_model == None:
            time.sleep(1)
    vae_upsampling = model_def.get("vae_upsampler", None)
    model_kwargs = {}
    if vae_upsampling is not None:
        new_vae_upsampling = None if image_mode not in vae_upsampling or "vae" not in spatial_upsampling else spatial_upsampling
        old_vae_upsampling =  None if reload_needed or wan_model is None or not hasattr(wan_model, "vae") or not hasattr(wan_model.vae, "upsampling_set") else wan_model.vae.upsampling_set
        reload_needed = reload_needed or old_vae_upsampling != new_vae_upsampling
        if new_vae_upsampling: model_kwargs = {"VAE_upsampling": new_vae_upsampling}
    output_type = get_output_type_for_model(model_type, image_mode)
    profile = compute_profile(override_profile, output_type)
    enhancer_mode = server_config.get("enhancer_mode", 1)
    if model_type != transformer_type or reload_needed or profile != loaded_profile:
        release_model()
        send_cmd("status", f"Loading model {get_model_name(model_type)}...")
        wan_model, offloadobj = load_models(
            model_type,
            override_profile,
            output_type=output_type,
            **model_kwargs,
        )
        send_cmd("status", "Model loaded")
        send_cmd("refresh_models", get_unique_id())
        reload_needed=  False
    if args.test:
        send_cmd("info", "Test mode: model loaded, skipping generation.")
        return True
    overridden_attention = override_attention if len(override_attention) else get_overridden_attention(model_type)
    # if overridden_attention is not None and overridden_attention !=  attention_mode: print(f"Attention mode has been overriden to {overridden_attention} for model type '{model_type}'")
    attn = overridden_attention if overridden_attention is not None else attention_mode
    if attn == "auto":
        attn = get_auto_attention()
    elif not attn in attention_modes_supported:
        send_cmd("info", f"You have selected attention mode '{attn}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        send_cmd("exit")
        return True
    
    width, height = resolution.split("x")
    width, height = int(width) // block_size *  block_size, int(height) // block_size *  block_size
    default_image_size = (height, width)

    if perturbation_switch == 0:
        perturbation_layers = None

    offload.shared_state["_attention"] =  attn
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if  hasattr(wan_model, "vae") and hasattr(wan_model.vae, "get_VAE_tile_size"):
        get_tile_size = wan_model.vae.get_VAE_tile_size
        try:
            sig = inspect.signature(get_tile_size)
        except (TypeError, ValueError):
            sig = None
        if sig is not None and "output_height" in sig.parameters:
            VAE_tile_size = get_tile_size(
                vae_config,
                device_mem_capacity,
                server_config.get("vae_precision", "16") == "32",
                output_height=height,
                output_width=width,
            )
        else:
            VAE_tile_size = get_tile_size(
                vae_config,
                device_mem_capacity,
                server_config.get("vae_precision", "16") == "32",
            )
    else:
        VAE_tile_size = None

    trans = get_transformer_model(wan_model)
    trans2 = get_transformer_model(wan_model, 2)
    audio_sampling_rate = 16000

    prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
    parsed_keep_frames_video_source= max_source_video_frames if len(keep_frames_video_source) ==0 else int(keep_frames_video_source) 
    transformer_loras_filenames, transformer_loras_multipliers  = get_transformer_loras(model_type)
    lora_dir = get_lora_dir(model_type)
    if guidance_phases < 1: guidance_phases = 1
    if transformer_loras_filenames != None:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(transformer_loras_multipliers, len(transformer_loras_filenames), num_inference_steps, nb_phases = guidance_phases, model_switch_phase= model_switch_phase )
        if len(errors) > 0: raise Exception(f"Error parsing Transformer Loras: {errors}")
        loras_selected = transformer_loras_filenames 

    if hasattr(wan_model, "get_loras_transformer"):
        extra_loras_transformers, extra_loras_multipliers = wan_model.get_loras_transformer(get_model_recursive_prop, **locals())
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(extra_loras_multipliers, len(extra_loras_transformers), num_inference_steps, nb_phases = guidance_phases, merge_slist= loras_slists, model_switch_phase= model_switch_phase )
        if len(errors) > 0: raise Exception(f"Error parsing Extra Transformer Loras: {errors}")
        loras_selected += extra_loras_transformers 

    if len(activated_loras) > 0:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases = guidance_phases, merge_slist= loras_slists, model_switch_phase= model_switch_phase )
        if len(errors) > 0: raise Exception(f"Error parsing Loras: {errors}")
        errors = check_loras_exist(model_type, activated_loras, True, send_cmd)
        if len(errors) > 0 : raise gr.Error(errors)
        loras_selected += [ os.path.join(lora_dir, os.path.basename(lora)) for lora in activated_loras]

    if hasattr(wan_model, "get_trans_lora"):
        trans_lora, trans2_lora = wan_model.get_trans_lora()
    else:     
        trans_lora, trans2_lora = trans, trans2

    if len(loras_selected) > 0:
        pinnedLora = loaded_profile !=5  # and transformer_loras_filenames == None False # # # 
        preprocess_target = trans_lora if trans_lora is not None else trans
        split_linear_modules_map = getattr(preprocess_target, "split_linear_modules_map", None)
        offload.load_loras_into_model(
            trans_lora,
            loras_selected,
            loras_list_mult_choices_nums,
            activate_all_loras=True,
            preprocess_sd=get_loras_preprocessor(preprocess_target, base_model_type),
            pinnedLora=pinnedLora,
            maxReservedLoras=server_config.get("max_reserved_loras", -1),
            split_linear_modules_map=split_linear_modules_map,
        )
        errors = trans_lora._loras_errors
        if len(errors) > 0:
            error_files = [msg for _ ,  msg  in errors]
            raise gr.Error("Error while loading Loras: " + ", ".join(error_files))
        if trans2_lora is not None: 
            offload.sync_models_loras(trans_lora, trans2_lora)
        
    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference
    model_filename = get_model_filename(base_model_type)  

    _, _, latent_size = get_model_min_frames_and_step(model_type)  
    video_length = (video_length -1) // latent_size * latent_size + 1
    if sliding_window_size !=0:
        sliding_window_size = (sliding_window_size -1) // latent_size * latent_size + 1
    if sliding_window_overlap !=0:
        sliding_window_defaults = model_def.get("sliding_window_defaults", {})
        if sliding_window_defaults.get("overlap_default", 0) != sliding_window_overlap:
            sliding_window_overlap = (sliding_window_overlap -1) // latent_size * latent_size + 1
    if sliding_window_discard_last_frames !=0:
        sliding_window_discard_last_frames = sliding_window_discard_last_frames // latent_size * latent_size 

    current_video_length = video_length
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576
    guide_inpaint_color = model_def.get("guide_inpaint_color", 127.5)
    if image_mode==2:
        guide_inpaint_color = model_def.get("inpaint_color", guide_inpaint_color)
    guide_inpaint_color = parse_guide_inpaint_color(guide_inpaint_color)
    extract_guide_from_window_start = model_def.get("extract_guide_from_window_start", False) 
    hunyuan_custom = "hunyuan_video_custom" in model_filename
    hunyuan_custom_edit =  hunyuan_custom and "edit" in model_filename
    fantasy = base_model_type in ["fantasy"]
    multitalk = model_def.get("multitalk_class", False)

    if "B" in audio_prompt_type or "X" in audio_prompt_type:
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
    else:
        speakers_bboxes = None        
    if "L" in image_prompt_type:
        file_list, _, _, _ = get_processed_queue(gen)
        if len(file_list)>0:
            video_source = file_list[-1]
        else:
            mp4_files = glob.glob(os.path.join(save_path, "*.mp4"))
            video_source = max(mp4_files, key=os.path.getmtime) if mp4_files else None                            
    fps = 1 if is_image else get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    control_audio_tracks = source_audio_tracks = source_audio_metadata = []
    if any_letters(audio_prompt_type, "R") and video_guide is not None and MMAudio_setting == 0 and audio_source is None:
        control_audio_tracks, _  = extract_audio_tracks(video_guide, temp_format="wav")
    if "K" in audio_prompt_type and video_guide is not None:
        try:
            if extract_audio_tracks(video_guide, query_only=True) == 0:
                print(f"No audio track found in Control Video: {video_guide}")
                audio_guide = None
            else:
                audio_guide = extract_audio_track_to_wav(video_guide, get_available_filename(save_path, video_guide, suffix="_control_audio", force_extension=".wav"))
                temp_filenames_list.append(audio_guide)
        except Exception as e:
            print(f"Unable to extract Audio track from Control Video:{e}")
            audio_guide = None
        audio_guide2 = None
    if video_source is not None:
        source_audio_tracks, source_audio_metadata = extract_audio_tracks(video_source, temp_format="wav")
        video_fps, _, _, video_frames_count = get_video_info(video_source)
        video_source_duration = video_frames_count / video_fps
    else:
        video_source_duration = 0

    if "A" in audio_prompt_type and audio_guide is None:
        audio_guide = create_silent_wav_file(save_path, video_length / fps, audio_sampling_rate)
        temp_filenames_list.append(audio_guide)


    reset_control_aligment = "T" in video_prompt_type

    original_image_refs = image_refs
    image_refs = None if image_refs is None else ([] + image_refs) # work on a copy as it is going to be modified
    # image_refs = None
    # nb_frames_positions= 0
    # Output Video Ratio Priorities:
    # Source Video or Start Image > Control Video > Image Ref (background or positioned frames only) >  UI Width, Height
    # Image Ref (non background and non positioned frames) are boxed in a white canvas in order to keep their own width/height ratio
    frames_to_inject = []
    any_background_ref  = 0
    custom_frames_injection = model_def.get("custom_frames_injection", False) and image_refs is not None and len(image_refs) > 0
    if "K" in video_prompt_type: 
        any_background_ref = 2 if model_def.get("all_image_refs_are_background_ref", False) or custom_frames_injection else 1
    outpainting_dims = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio)
    fit_canvas = server_config.get("fit_canvas", 0)
    fit_crop = fit_canvas == 2
    if fit_crop and outpainting_dims is not None:
        fit_crop = False
        fit_canvas = 0

    joint_pass = boost ==1 #and profile != 1 and profile != 3  
    
    skip_steps_cache = None if len(skip_steps_cache_type) == 0 else DynamicClass(cache_type = skip_steps_cache_type) 

    if skip_steps_cache != None:
        skip_steps_cache.update({     
        "multiplier" : skip_steps_multiplier,
        "start_step":  int(skip_steps_start_step_perc*num_inference_steps/100)
        })
        model_handler.set_cache_parameters(skip_steps_cache_type, base_model_type, model_def, locals(), skip_steps_cache)
        if skip_steps_cache_type == "mag":
            def_mag_ratios = model_def.get("magcache_ratios", None) if model_def != None else None
            if def_mag_ratios is not None: skip_steps_cache.def_mag_ratios = def_mag_ratios
        elif skip_steps_cache_type == "tea":
            def_tea_coefficients = model_def.get("teacache_coefficients", None) if model_def != None else None
            if def_tea_coefficients is not None: skip_steps_cache.coefficients = def_tea_coefficients
        else:
            raise Exception(f"unknown cache type {skip_steps_cache_type}")
    trans.cache = skip_steps_cache
    if trans2 is not None: trans2.cache = skip_steps_cache
    face_arc_embeds = None
    src_ref_images = src_ref_masks = None
    output_new_audio_data = None
    output_new_audio_filepath = None
    original_audio_guide = audio_guide
    original_audio_guide2 = audio_guide2
    audio_proj_split = None
    audio_proj_full = None
    audio_scale = audio_scale if model_def.get("audio_scale_name") else None
    audio_context_lens = None
    full_audio_guide_waveform, full_audio_guide_sample_rate = None, 0
    if test_any_sliding_window(model_type) and video_source is not None: current_video_length +=  sliding_window_overlap - 1
    if audio_guide != None:
        from preprocessing.extract_vocals import get_vocals
        import librosa
        duration = librosa.get_duration(path=audio_guide)
        combination_type = "add"
        clean_audio_files = "V" in audio_prompt_type
        if audio_guide2 is not None:
            if "N" in audio_prompt_type:
                audio_guide, audio_guide2, _ = normalize_audio_pair_volumes_to_temp_files(audio_guide, audio_guide2, output_dir=save_path, prefix="audio_norm_")
                temp_filenames_list += [audio_guide, audio_guide2]
            duration2 = librosa.get_duration(path=audio_guide2)
            if "C" in audio_prompt_type: duration += duration2
            else: duration = min(duration, duration2)
            combination_type = "para" if "P" in audio_prompt_type else "add" 
            if clean_audio_files:
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                audio_guide2 = get_vocals(original_audio_guide2, get_available_filename(save_path, audio_guide2, "_clean2", ".wav"))
                temp_filenames_list += [audio_guide, audio_guide2]
        else:
            if "X" in audio_prompt_type: 
                # dual speaker, voice separation
                from preprocessing.speakers_separator import extract_dual_audio
                combination_type = "para"
                if args.save_speakers:
                    audio_guide, audio_guide2  = "speaker1.wav", "speaker2.wav"
                else:
                    audio_guide, audio_guide2  = get_available_filename(save_path, audio_guide, "_tmp1", ".wav"),  get_available_filename(save_path, audio_guide, "_tmp2", ".wav")
                    temp_filenames_list +=   [audio_guide, audio_guide2]                  
                if clean_audio_files:
                    clean_audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, original_audio_guide, "_clean", ".wav"))
                    temp_filenames_list += [clean_audio_guide]
                extract_dual_audio(clean_audio_guide if clean_audio_files else original_audio_guide, audio_guide, audio_guide2)

            elif clean_audio_files:
                # Single Speaker
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                temp_filenames_list += [audio_guide]

            output_new_audio_filepath = original_audio_guide

        video_length_not_limited_by_audio = model_def.get("video_length_not_limited_by_audio", False) and "L" in audio_prompt_type and "F" not in audio_prompt_type
        if "F" in audio_prompt_type:
            full_audio_guide_waveform, full_audio_guide_sample_rate = slice_audio_window(audio_guide, 0, max_source_video_frames, fps, save_path, suffix=f"_full", pad_head=False, pad_tail=False)
        elif not video_length_not_limited_by_audio:
            current_video_length = min(int(fps * duration //latent_size) * latent_size + latent_size + 1, current_video_length)
        if fantasy:
            from models.wan.fantasytalking.infer import parse_audio
            # audio_proj_split_full, audio_context_lens_full = parse_audio(audio_guide, num_frames= max_source_video_frames, fps= fps,  padded_frames_for_embeddings= (reuse_frames if reset_control_aligment else 0), device= processing_device  )
            if audio_scale is None:
                audio_scale = 1.0
        elif multitalk:
            from models.wan.multitalk.multitalk import get_full_audio_embeddings
            # pad audio_proj_full if aligned to beginning of window to simulate source window overlap
            min_audio_duration =  current_video_length/fps if reset_control_aligment else video_source_duration + current_video_length/fps
            audio_proj_full, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = audio_guide, audio_guide2= audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration) 
            if output_new_audio_data is not None: # not none if modified
                if clean_audio_files: # need to rebuild the sum of audios with original audio
                    _, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = original_audio_guide, audio_guide2= original_audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration, return_sum_only= True) 
                output_new_audio_filepath=  None # need to build original speaker track if it changed size (due to padding at the end) or if it has been combined

    if hunyuan_custom_edit and video_guide != None:
        import cv2
        cap = cv2.VideoCapture(video_guide)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_video_length = min(current_video_length, length)

    if test_any_sliding_window(model_type) :
        sliding_window = current_video_length > sliding_window_size
        reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap) 
    else:
        sliding_window = False
        sliding_window_size = current_video_length
        reuse_frames = 0
    seed = set_seed(seed)

    torch.set_grad_enabled(False) 
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(audio_save_path, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    wan_model._interrupt = False
    abort = False
    if gen.get("abort", False):
        return True
    # gen["abort"] = False
    gen["prompt"] = prompt    
    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    discard_last_frames = sliding_window_discard_last_frames
    default_requested_frames_to_generate = current_video_length
    nb_frames_positions = 0
    if sliding_window:
        initial_total_windows= compute_sliding_window_no(default_requested_frames_to_generate, sliding_window_size, discard_last_frames, reuse_frames) 
        current_video_length = sliding_window_size
    else:
        initial_total_windows = 1

    first_window_video_length = current_video_length
    original_prompts = prompts.copy()
    gen["sliding_window"] = sliding_window 
    while not abort: 
        extra_generation += gen.get("extra_orders",0)
        gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation     
        gen["header_text"] = ""    
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        src_video = src_video2 = src_mask = src_mask2 = src_faces = sparse_video_image = full_generated_audio =None
        prefix_video = pre_video_frame = None
        source_video_overlap_frames_count = 0 # number of frames overalapped in source video for first window
        source_video_frames_count = 0  # number of frames to use in source video (processing starts source_video_overlap_frames_count frames before )
        frames_already_processed = []
        frames_already_processed_count = 0
        overlapped_latents = None
        pre_video_guide_is_hdr = False
        context_scale = None
        window_no = 0
        extra_windows = 0
        abort_scheduled = False
        guide_start_frame = 0 # pos of of first control video frame of current window  (reuse_frames later than the first processed frame)
        keep_frames_parsed = [] # aligned to the first control frame of current window (therefore ignore previous reuse_frames)
        pre_video_guide = None # reuse_frames of previous window
        pre_audio_guide, pre_audio_guide_sample_rate = None, 0
        image_size = default_image_size #  default frame dimensions for budget until it is change due to a resize
        sample_fit_canvas = fit_canvas
        current_video_length = first_window_video_length
        gen["extra_windows"] = 0
        gen["total_windows"] = 1
        gen["window_no"] = 1
        input_waveform, input_waveform_sample_rate = None, 0
        num_frames_generated = 0 # num of new frames created (lower than the number of frames really processed due to overlaps and discards)
        requested_frames_to_generate = default_requested_frames_to_generate # num  of num frames to create (if any source window this num includes also the overlapped source window frames)
        cached_video_guide_processed = cached_video_mask_processed = cached_video_guide_processed2 = cached_video_mask_processed2 = None
        cached_video_video_start_frame = cached_video_video_end_frame = -1
        start_time = time.time()
        if prompt_enhancer_image_caption_model != None and prompt_enhancer !=None and len(prompt_enhancer)>0 and enhancer_mode == 0:
            send_cmd("progress", [0, get_latest_status(state, "Enhancing Prompt")])
            enhancer_kwargs = {"image_prompt_type":  image_prompt_type, "video_prompt_type":  video_prompt_type, "audio_prompt_type":  audio_prompt_type}
            enhanced_prompts = process_prompt_enhancer(model_type, model_def, prompt_enhancer, original_prompts,  image_start if image_start is not None else image_end , original_image_refs, is_image, audio_only, seed, enhancer_kwargs = enhancer_kwargs )
            unload_prompt_enhancer_runtime()
            if enhanced_prompts is not None:
                print(f"Enhanced prompts: {enhanced_prompts}" )
                if len(enhanced_prompts) > 1 and any("\n" in one_prompt for one_prompt in enhanced_prompts):
                    task["prompt"] = prompt_parser.ENHANCED_PROMPT_PREFIX + prompt_parser.serialize_prompt_blocks_with_prefix(enhanced_prompts)
                else:
                    task["prompt"] = prompt_parser.ENHANCED_PROMPT_PREFIX + prompt_parser.serialize_prompt_units("", enhanced_prompts, multi_prompts_gen_type)
                gen["last_was_audio"] = audio_only
                send_cmd("output")
                prompts = enhanced_prompts            
                abort = gen.get("abort", False)

 
        while not abort:
            enable_RIFLEx = RIFLEx_setting == 0 and current_video_length > (6* get_model_fps(base_model_type)+1) or RIFLEx_setting == 1
            prompt =  prompts[window_no] if window_no < len(prompts) else prompts[-1]
            new_extra_windows = gen.get("extra_windows",0)
            gen["extra_windows"] = 0
            extra_windows += new_extra_windows
            requested_frames_to_generate +=  new_extra_windows * (sliding_window_size - discard_last_frames - reuse_frames)
            sliding_window = sliding_window  or extra_windows > 0
            if sliding_window and window_no > 0:
                # num_frames_generated -= reuse_frames
                if (requested_frames_to_generate - num_frames_generated) <  latent_size:
                    break
                current_video_length = min(sliding_window_size, ((requested_frames_to_generate - num_frames_generated + reuse_frames + discard_last_frames) // latent_size) * latent_size + 1 )

            total_windows = initial_total_windows + extra_windows
            gen["total_windows"] = total_windows
            if window_no >= total_windows:
                break
            window_no += 1
            gen["window_no"] = window_no
            return_latent_slice = None 
            frames_relative_positions_list = []
            if reuse_frames > 0:                
                return_latent_slice = slice(- max(1, (reuse_frames + discard_last_frames ) // latent_size) , None if discard_last_frames == 0 else -(discard_last_frames // latent_size) )
            refresh_preview  = {"image_guide" : image_guide, "image_mask" : image_mask} if image_mode >= 1 else {}

            if hasattr(model_handler, "custom_prompt_preprocess"):
                prompt = model_handler.custom_prompt_preprocess(**locals())
            image_start_tensor = image_end_tensor = None
            if window_no == 1 and (video_source is not None or image_start is not None):
                if image_start is not None:
                    image_start_tensor, new_height, new_width = calculate_dimensions_and_resize_image(image_start, height, width, sample_fit_canvas, fit_crop, block_size = block_size)
                    if fit_crop: refresh_preview["image_start"] = image_start_tensor 
                    image_start_tensor = convert_image_to_tensor(image_start_tensor)
                    pre_video_guide =  prefix_video = image_start_tensor.unsqueeze(1)
                else:
                    prefix_video_is_hdr = "&" in video_prompt_type and _video_input_is_hdr(video_source)
                    prefix_video  = preprocess_video(width=width, height=height,video_in=video_source, max_frames= parsed_keep_frames_video_source , start_frame = 0, fit_canvas= sample_fit_canvas, fit_crop = fit_crop, target_fps = fps, block_size = block_size, preserve_hdr=prefix_video_is_hdr )
                    prefix_video  = prefix_video.permute(3, 0, 1, 2)

                    if fit_crop or "L" in image_prompt_type: refresh_preview["video_source"] = convert_tensor_to_image(prefix_video, 0) 

                    new_height, new_width = prefix_video.shape[-2:]                    
                    pre_video_guide = prefix_video[:, -reuse_frames:].float()
                    if prefix_video_is_hdr:
                        pre_video_guide_is_hdr = True
                    else:
                        pre_video_guide = pre_video_guide.div_(127.5).sub_(1.) # c, f, h, w
                pre_video_frame = convert_tensor_to_image(prefix_video[:, -1])
                source_video_overlap_frames_count = pre_video_guide.shape[1]
                source_video_frames_count = prefix_video.shape[1]
                if sample_fit_canvas != None: 
                    image_size  = pre_video_guide.shape[-2:]
                    sample_fit_canvas = None
                guide_start_frame =  prefix_video.shape[1]
            if image_end is not None:
                image_end_list=  image_end if isinstance(image_end, list) else [image_end]
                if len(image_end_list) >= window_no:
                    new_height, new_width = image_size                    
                    image_end_tensor, _, _ = calculate_dimensions_and_resize_image(image_end_list[window_no-1], new_height, new_width, sample_fit_canvas, fit_crop, block_size = block_size)
                    # image_end_tensor =image_end_list[window_no-1].resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
                    refresh_preview["image_end"] = image_end_tensor 
                    image_end_tensor = convert_image_to_tensor(image_end_tensor)
                    if sample_fit_canvas != None: 
                        image_size  = image_end_tensor.shape[-2:]
                        sample_fit_canvas = None
                image_end_list= None
            window_start_frame = guide_start_frame - (reuse_frames if window_no > 1 else source_video_overlap_frames_count)
            guide_end_frame = guide_start_frame + current_video_length - (source_video_overlap_frames_count if window_no == 1 else reuse_frames)
            alignment_shift = source_video_frames_count if reset_control_aligment else 0
            aligned_guide_start_frame = guide_start_frame - alignment_shift
            aligned_guide_end_frame = guide_end_frame - alignment_shift
            aligned_window_start_frame = window_start_frame - alignment_shift  
            input_waveform, input_waveform_sample_rate = None, 0
            if full_audio_guide_waveform is not None:
                input_waveform, input_waveform_sample_rate = full_audio_guide_waveform, full_audio_guide_sample_rate
            elif audio_guide is not None and model_def.get("audio_guide_window_slicing", False):
                audio_start_frame = aligned_window_start_frame
                if reset_control_aligment:
                    audio_start_frame += source_video_overlap_frames_count
                input_waveform, input_waveform_sample_rate = slice_audio_window(audio_guide, audio_start_frame, current_video_length, fps, save_path, suffix=f"_win{window_no}", pad_tail=not video_length_not_limited_by_audio) 
                if input_waveform.shape[0] == 0: input_waveform, input_waveform_sample_rate = pre_audio_guide, pre_audio_guide_sample_rate
            elif model_def.get("audio_guide_window_slicing", False):
                if pre_audio_guide is not None and pre_audio_guide.shape[0] > 0:
                    input_waveform, input_waveform_sample_rate = pre_audio_guide, pre_audio_guide_sample_rate
                elif window_no == 1 and source_video_overlap_frames_count > 0 and len(source_audio_tracks) > 0:
                    source_audio_start_frame = max(0, source_video_frames_count - source_video_overlap_frames_count)
                    input_waveform, input_waveform_sample_rate = slice_audio_window(source_audio_tracks[0], source_audio_start_frame, source_video_overlap_frames_count, fps, save_path, suffix=f"_source_overlap_win{window_no}", pad_head=False, pad_tail=False)
                    if input_waveform.shape[0] == 0: input_waveform, input_waveform_sample_rate = None, 0
            if fantasy and audio_guide is not None:
                audio_proj_split , audio_context_lens = parse_audio(audio_guide, start_frame = aligned_window_start_frame, num_frames= current_video_length, fps= fps,  device= processing_device  )
            if multitalk:
                from models.wan.multitalk.multitalk import get_window_audio_embeddings
                # special treatment for start frame pos when alignement to first frame requested as otherwise the start frame number will be negative due to overlapped frames (has been previously compensated later with padding)
                audio_proj_split = get_window_audio_embeddings(audio_proj_full, audio_start_idx= aligned_window_start_frame + (source_video_overlap_frames_count if reset_control_aligment else 0 ), clip_length = current_video_length)

            if repeat_no == 1 and window_no == 1 and image_refs is not None and len(image_refs) > 0:
                frames_positions_list = []
                if frames_positions is not None and len(frames_positions)> 0:
                    positions = frames_positions.replace(","," ").split(" ")
                    cur_end_pos =  -1 + (source_video_frames_count - source_video_overlap_frames_count)
                    last_frame_no = requested_frames_to_generate + source_video_frames_count - source_video_overlap_frames_count
                    joker_used = False
                    project_window_no = 1
                    for pos in positions :
                        if len(pos) > 0:
                            if pos in ["L", "l"]:
                                cur_end_pos += sliding_window_size if project_window_no > 1 else current_video_length 
                                if cur_end_pos >= last_frame_no-1 and not joker_used:
                                    joker_used = True
                                    cur_end_pos = last_frame_no -1
                                project_window_no += 1
                                frames_positions_list.append(cur_end_pos)
                                cur_end_pos -= sliding_window_discard_last_frames + reuse_frames
                            else:
                                frames_positions_list.append(int(pos)-1 + alignment_shift)
                    frames_positions_list = frames_positions_list[:len(image_refs)]
                nb_frames_positions = len(frames_positions_list) 
                if nb_frames_positions > 0:
                    frames_to_inject = [None] * (max(frames_positions_list) + 1)
                    for i, pos in enumerate(frames_positions_list):
                        frames_to_inject[pos] = image_refs[i] 

            video_guide_processed = video_mask_processed = video_guide_processed2 = video_mask_processed2 = sparse_video_image = None
            if video_guide is not None:
                keep_frames_parsed_full, error = parse_keep_frames_video_guide(keep_frames_video_guide, source_video_frames_count -source_video_overlap_frames_count + requested_frames_to_generate)
                if len(error) > 0:
                    raise gr.Error(f"invalid keep frames {keep_frames_video_guide}")
                guide_frames_extract_start = aligned_window_start_frame if extract_guide_from_window_start else aligned_guide_start_frame
                extra_control_frames = model_def.get("extra_control_frames", 0)
                if extra_control_frames > 0 and aligned_guide_start_frame >= extra_control_frames: guide_frames_extract_start -= extra_control_frames
                        
                keep_frames_parsed = [True] * -guide_frames_extract_start if guide_frames_extract_start  <0 else []
                keep_frames_parsed += keep_frames_parsed_full[max(0, guide_frames_extract_start): aligned_guide_end_frame ] 
                guide_frames_extract_count = len(keep_frames_parsed)

                process_all = model_def.get("preprocess_all", False)
                if process_all:
                    guide_slice_to_extract  = guide_frames_extract_count
                    guide_frames_extract_count = (-guide_frames_extract_start if guide_frames_extract_start  <0 else 0) +  len( keep_frames_parsed_full[max(0, guide_frames_extract_start):] )

                # Extract Faces to video
                if "B" in video_prompt_type:
                    send_cmd("progress", [0, get_latest_status(state, "Extracting Face Movements")])
                    src_faces = extract_faces_from_video_with_mask(video_guide, video_mask, max_frames= guide_frames_extract_count, start_frame= guide_frames_extract_start, size= 512, target_fps = fps)
                    if src_faces is not None and src_faces.shape[1] < current_video_length:
                        src_faces = torch.cat([src_faces, torch.full( (3, current_video_length - src_faces.shape[1], 512, 512 ), -1, dtype = src_faces.dtype, device= src_faces.device) ], dim=1)

                # Sparse Video to Video
                sparse_video_image = None
                if "R" in video_prompt_type:
                    sparse_video_image = get_video_frame(video_guide, aligned_guide_start_frame, return_last_if_missing = True, target_fps = fps, return_PIL = True)

                if not process_all or cached_video_video_start_frame < 0:
                    # Generic Video Preprocessing
                    process_outside_mask = process_map_outside_mask.get(filter_letters(video_prompt_type, "YWX"), None)
                    preprocess_type, preprocess_type2 =  "raw", None 
                    for process_num, process_letter in enumerate( filter_letters(video_prompt_type, video_guide_processes)):
                        if process_num == 0:
                            preprocess_type = process_map_video_guide.get(process_letter, "raw")
                        else:
                            preprocess_type2 = process_map_video_guide.get(process_letter, None)
                    custom_preprocessor = model_def.get("custom_preprocessor", None) 
                    if custom_preprocessor is not None:
                        status_info = custom_preprocessor
                        send_cmd("progress", [0, get_latest_status(state, status_info)])
                        video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 =  custom_preprocess_video_with_mask(model_handler, base_model_type, pre_video_guide, video_guide if sparse_video_image is None else sparse_video_image, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  block_size = block_size, expand_scale = mask_expand, video_prompt_type= video_prompt_type)
                    else:
                        status_info = "Extracting " + processes_names[preprocess_type]
                        extra_process_list = ([] if preprocess_type2==None else [preprocess_type2]) + ([] if process_outside_mask==None or process_outside_mask == preprocess_type else [process_outside_mask])
                        if len(extra_process_list) == 1:
                            status_info += " and " + processes_names[extra_process_list[0]]
                        elif len(extra_process_list) == 2:
                            status_info +=  ", " + processes_names[extra_process_list[0]] + " and " + processes_names[extra_process_list[1]]
                        context_scale = [control_net_weight /2, control_net_weight2 /2] if preprocess_type2 is not None else [control_net_weight]

                        if not (preprocess_type == "identity" and preprocess_type2 is None and video_mask is None):send_cmd("progress", [0, get_latest_status(state, status_info)])
                        inpaint_color = 0 if "pose" in preprocess_type and process_outside_mask == "inpaint" else guide_inpaint_color
                        if "O" in video_prompt_type and pre_video_guide is None and all_letters(video_prompt_type, "IK"):
                            from shared.utils.utils import get_outpainting_full_area_dimensions
                            w, h = image_refs[0].size
                            if outpainting_dims != None:
                                h, w = get_outpainting_full_area_dimensions(h, w, outpainting_dims, video_guide_outpainting_ratio)
                            image_size = calculate_new_dimensions(height, width, h, w, fit_canvas)                            
                            sample_fit_canvas = None 
                            ref_pose_tensor  = resize_and_remove_background(image_refs[nb_frames_positions:nb_frames_positions+1], image_size[1], image_size[0],
                                                                                            False, True, 
                                                                                            fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                            block_size=block_size,
                                                                                            outpainting_dims =outpainting_dims,
                                                                                            outpainting_ratio = video_guide_outpainting_ratio,
                                                                                            background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                            return_tensor= True)[0][0]
                        else:
                            ref_pose_tensor = pre_video_guide
 
                        video_guide_processed, video_mask_processed = preprocess_video_with_mask(ref_pose_tensor, video_guide if sparse_video_image is None else sparse_video_image, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, outpainting_ratio = video_guide_outpainting_ratio, proc_no =1, inpaint_color =inpaint_color, block_size = block_size, to_bbox = "H" in video_prompt_type )
                        if preprocess_type2 != None:
                            video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(ref_pose_tensor, video_guide, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type2, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, outpainting_ratio = video_guide_outpainting_ratio, proc_no =2, block_size = block_size, to_bbox = "H" in video_prompt_type )

                    if video_guide_processed is not None  and sample_fit_canvas is not None:
                        image_size = video_guide_processed.shape[-2:]
                        sample_fit_canvas = None

                    if process_all:
                        cached_video_guide_processed, cached_video_mask_processed, cached_video_guide_processed2, cached_video_mask_processed2 = video_guide_processed, video_mask_processed, video_guide_processed2, video_mask_processed2
                        cached_video_video_start_frame = guide_frames_extract_start

                if process_all:
                    process_slice = slice(guide_frames_extract_start - cached_video_video_start_frame, guide_frames_extract_start - cached_video_video_start_frame + guide_slice_to_extract  )
                    video_guide_processed = None if cached_video_guide_processed is None else cached_video_guide_processed[:, process_slice] 
                    video_mask_processed =  None if cached_video_mask_processed is None else cached_video_mask_processed[:, process_slice] 
                    video_guide_processed2 =  None if cached_video_guide_processed2 is None else cached_video_guide_processed2[:, process_slice] 
                    video_mask_processed2 = None if cached_video_mask_processed2 is None else cached_video_mask_processed2[:, process_slice] 
                    
            if window_no == 1 and image_refs is not None and len(image_refs) > 0:
                if sample_fit_canvas is not None and (nb_frames_positions > 0 or "K" in video_prompt_type) :
                    from shared.utils.utils import get_outpainting_full_area_dimensions
                    w, h = image_refs[0].size
                    if outpainting_dims != None:
                        h, w = get_outpainting_full_area_dimensions(h, w, outpainting_dims, video_guide_outpainting_ratio)
                    image_size = calculate_new_dimensions(height, width, h, w, fit_canvas)
                sample_fit_canvas = None
                if repeat_no == 1:
                    if fit_crop:
                        if any_background_ref == 2:
                            end_ref_position = len(image_refs)
                        elif any_background_ref == 1:
                            end_ref_position = nb_frames_positions + 1
                        else:
                            end_ref_position = nb_frames_positions 
                        for i, img in enumerate(image_refs[:end_ref_position]):
                            image_refs[i] = rescale_and_crop(img, default_image_size[1], default_image_size[0])
                        refresh_preview["image_refs"] = image_refs

                    if len(image_refs) > nb_frames_positions:
                        src_ref_images = image_refs[nb_frames_positions:]
                        if "Q" in video_prompt_type:
                            from preprocessing.arc.face_encoder import FaceEncoderArcFace, get_landmarks_from_image
                            image_pil = src_ref_images[-1]
                            face_encoder = FaceEncoderArcFace()
                            face_encoder.init_encoder_model(processing_device)
                            face_arc_embeds = face_encoder(image_pil, need_proc=True, landmarks=get_landmarks_from_image(image_pil))
                            face_arc_embeds = face_arc_embeds.squeeze(0).cpu()
                            face_encoder = image_pil = None
                            gc.collect()
                            torch.cuda.empty_cache()

                        if remove_background_images_ref > 0:
                            send_cmd("progress", [0, get_latest_status(state, "Removing Images References Background")])

                        src_ref_images, src_ref_masks  = resize_and_remove_background(src_ref_images , image_size[1], image_size[0],
                                                                                        remove_background_images_ref > 0, any_background_ref, 
                                                                                        fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                        block_size=block_size,
                                                                                        outpainting_dims =outpainting_dims,
                                                                                        outpainting_ratio = video_guide_outpainting_ratio,
                                                                                        background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                        return_tensor= model_def.get("return_image_refs_tensor", False),
                                                                                        ignore_last_refs =model_def.get("no_processing_on_last_images_refs",0),
                                                                                        background_removal_color = model_def.get("background_removal_color", [255, 255, 255] ))
            frames_to_inject_parsed = frames_to_inject[ window_start_frame if extract_guide_from_window_start else guide_start_frame: guide_end_frame]
            if video_guide is not None or len(frames_to_inject_parsed) > 0 and not custom_frames_injection or model_def.get("forced_guide_mask_inputs", False): 
                any_mask = video_mask is not None or model_def.get("forced_guide_mask_inputs", False)
                any_guide_padding = model_def.get("pad_guide_video", False)
                dont_cat_preguide = extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None 
                from shared.utils.utils import prepare_video_guide_and_mask
                src_videos, src_masks = prepare_video_guide_and_mask(   [video_guide_processed] + ([] if video_guide_processed2 is None else [video_guide_processed2]), 
                                                                        [video_mask_processed] + ([] if video_guide_processed2 is None else [video_mask_processed2]),
                                                                        None if dont_cat_preguide else pre_video_guide, 
                                                                        image_size, current_video_length, latent_size,
                                                                        any_mask, any_guide_padding, guide_inpaint_color, 
                                                                        keep_frames_parsed, [] if custom_frames_injection else frames_to_inject_parsed , outpainting_dims, video_guide_outpainting_ratio)
                video_guide_processed = video_guide_processed2 = video_mask_processed = video_mask_processed2 = None
                if len(src_videos) == 1:
                    src_video, src_video2, src_mask, src_mask2 = src_videos[0], None, src_masks[0], None 
                else:
                    src_video, src_video2 = src_videos 
                    src_mask, src_mask2 = src_masks 
                src_videos = src_masks = None
                if src_video is None or window_no >1 and src_video.shape[1] <= sliding_window_overlap and not dont_cat_preguide:
                    abort = True 
                    break
                if model_def.get("control_video_trim", False) :
                    if src_video is None:
                        abort = True 
                        break
                    elif src_video.shape[1] < current_video_length:
                        current_video_length = src_video.shape[1]
                        abort_scheduled = True 
                if src_faces is not None:
                    if src_faces.shape[1] < src_video.shape[1]:
                        src_faces = torch.concat( [src_faces,  src_faces[:, -1:].repeat(1, src_video.shape[1] - src_faces.shape[1], 1,1)], dim =1)
                    else:
                        src_faces = src_faces[:, :src_video.shape[1]]
                if video_guide is not None or len(frames_to_inject_parsed) > 0:
                    if args.save_masks:
                        if src_video is not None: 
                            save_video( src_video, "masked_frames.mp4", fps)
                            if any_mask: save_video( src_mask, "masks.mp4", fps, value_range=(0, 1))
                        if src_video2 is not None: 
                            save_video( src_video2, "masked_frames2.mp4", fps)
                            if any_mask: save_video( src_mask2, "masks2.mp4", fps, value_range=(0, 1))
                if video_guide is not None:                        
                    preview_frame_no = 0 if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else (guide_start_frame - window_start_frame) 
                    preview_frame_no = min(src_video.shape[1] -1, preview_frame_no)
                    refresh_preview["video_guide"] = convert_tensor_to_image(src_video, preview_frame_no)
                    if src_video2 is not None and not model_def.get("no_guide2_refresh", False):
                        refresh_preview["video_guide"] = [refresh_preview["video_guide"], convert_tensor_to_image(src_video2, preview_frame_no)] 
                    if src_mask is not None and video_mask is not None and not model_def.get("no_mask_refresh", False):                        
                        refresh_preview["video_mask"] = convert_tensor_to_image(src_mask, preview_frame_no, mask_levels = True)

            if src_ref_images is not None or nb_frames_positions:
                if len(frames_to_inject_parsed):
                    if custom_frames_injection:
                        frames_relative_positions_list= [ frame_no + (0 if extract_guide_from_window_start else (aligned_guide_start_frame - aligned_window_start_frame)) for frame_no, frame in enumerate(frames_to_inject_parsed) if frame is not None]
                        frames_to_inject_parsed = new_image_refs = [frame for frame in frames_to_inject_parsed if frame is not None]
                    else:
                        new_image_refs = [convert_tensor_to_image(src_video, frame_no + (0 if extract_guide_from_window_start else (aligned_guide_start_frame - aligned_window_start_frame)) ) for frame_no, inject in enumerate(frames_to_inject_parsed) if inject]
                else:
                    new_image_refs = []
                if src_ref_images is not None:
                    new_image_refs +=  [convert_tensor_to_image(img) if torch.is_tensor(img) else img for img in src_ref_images  ]
                refresh_preview["image_refs"] = new_image_refs
                new_image_refs = None

            if len(refresh_preview) > 0:
                new_inputs= locals()
                new_inputs.update(refresh_preview)
                update_task_thumbnails(task, new_inputs)
                send_cmd("output")

            if window_no ==  1:                
                conditioning_latents_size = ( (source_video_overlap_frames_count-1) // latent_size) + 1 if source_video_overlap_frames_count > 0 else 0
            else:
                conditioning_latents_size = ( (reuse_frames-1) // latent_size) + 1

            status = get_latest_status(state)
            gen["progress_status"] = status
            progress_phase = "Generating Audio" if audio_only else "Encoding Prompt"
            gen["progress_phase"] = (progress_phase , -1 )
            callback = build_callback(state, trans, send_cmd, status, num_inference_steps)
            progress_args = [0, merge_status_context(status, progress_phase )]
            send_cmd("progress", progress_args)

            if skip_steps_cache !=  None:
                skip_steps_cache.update({
                "num_steps" : num_inference_steps,                
                "skipped_steps" : 0,
                "previous_residual": None,
                "previous_modulated_input":  None,
                })
            # samples = torch.empty( (1,2)) #for testing
            # if False:
            def set_header_text(txt):
                gen["header_text"] = txt
                send_cmd("output")

            try:
                input_video_for_model = pre_video_guide
                input_video_is_hdr = pre_video_guide_is_hdr
                prefix_frames_count = source_video_overlap_frames_count if window_no <= 1 else reuse_frames
                prefix_video_for_model = prefix_video
                if prefix_video is not None and prefix_video.dtype == torch.uint8:
                    prefix_video_for_model = prefix_video.float().div_(127.5).sub_(1.0)
                if window_no <= 1 and video_source is not None and "&" in video_prompt_type and _video_input_is_hdr(video_source):
                    input_video_is_hdr = True
                custom_settings_for_model = custom_settings if isinstance(custom_settings, dict) else {}
                model_outpainting_dims = outpainting_dims
                if outpainting_dims is not None and len((video_guide_outpainting_ratio or "").strip()) > 0:
                    if isinstance(video_guide, Image.Image):
                        control_source_width, control_source_height = video_guide.size
                    elif video_guide is not None:
                        _, control_source_width, control_source_height, _ = get_video_info(video_guide)
                    else:
                        control_source_width = control_source_height = None
                    if control_source_height is not None and control_source_width is not None:
                        model_outpainting_dims = resolve_outpainting_dims(control_source_height, control_source_width, outpainting_dims, video_guide_outpainting_ratio)
                overridden_inputs = None
                samples = wan_model.generate(
                    input_prompt = prompt,
                    alt_prompt = alt_prompt,
                    image_start = image_start_tensor,  
                    image_end = image_end_tensor,
                    input_frames = src_video,
                    input_frames2 = src_video2,
                    input_ref_images=  src_ref_images,
                    input_ref_masks = src_ref_masks,
                    input_masks = src_mask,
                    input_masks2 = src_mask2,
                    input_video= input_video_for_model,
                    input_faces = src_faces,
                    input_custom = custom_guide,
                    denoising_strength=denoising_strength,
                    masking_strength=masking_strength,
                    prefix_frames_count = prefix_frames_count,
                    frame_num= (current_video_length // latent_size)* latent_size + 1,
                    batch_size = batch_size,
                    height = image_size[0],
                    width = image_size[1],
                    fit_into_canvas = fit_canvas,
                    shift=flow_shift,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    guide2_scale = guidance2_scale,
                    guide3_scale = guidance3_scale,
                    switch_threshold = switch_threshold, 
                    switch2_threshold = switch_threshold2,
                    guide_phases= guidance_phases,
                    model_switch_phase = model_switch_phase,
                    embedded_guidance_scale=embedded_guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    perturbation_switch = perturbation_switch,
                    perturbation_layers = perturbation_layers,
                    perturbation_start = perturbation_start_perc/100,
                    perturbation_end = perturbation_end_perc/100,
                    apg_switch = apg_switch,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                    alt_guide_scale= alt_guidance_scale,
                    audio_cfg_scale= audio_guidance_scale,
                    input_waveform=input_waveform, 
                    input_waveform_sample_rate=input_waveform_sample_rate,
                    audio_guide=audio_guide,
                    audio_guide2=audio_guide2,
                    audio_prompt_type=audio_prompt_type,
                    audio_proj= audio_proj_split,
                    audio_scale= audio_scale,
                    audio_context_lens= audio_context_lens,
                    context_scale = context_scale,
                    control_scale_alt = control_net_weight_alt,
                    alt_scale = alt_scale,
                    motion_amplitude = motion_amplitude,
                    model_mode = model_mode,
                    causal_block_size = 5,
                    causal_attention = True,
                    fps = fps,
                    overlapped_latents = overlapped_latents,
                    return_latent_slice= return_latent_slice,
                    overlap_noise = sliding_window_overlap_noise,
                    overlap_size = sliding_window_overlap,
                    color_correction_strength = sliding_window_color_correction_strength,
                    conditioning_latents_size = conditioning_latents_size,
                    input_video_is_hdr=input_video_is_hdr,
                    lora_dir=lora_dir,
                    keep_frames_parsed = keep_frames_parsed,
                    model_filename = model_filename,
                    model_type = base_model_type,
                    loras_slists = loras_slists,
                    NAG_scale = NAG_scale,
                    NAG_tau = NAG_tau,
                    NAG_alpha = NAG_alpha,
                    speakers_bboxes =speakers_bboxes,
                    image_mode =  image_mode,
                    video_prompt_type= video_prompt_type,
                    window_no = window_no, 
                    offloadobj = offloadobj,
                    set_header_text= set_header_text,
                    pre_video_frame = pre_video_frame,
                    prefix_video = prefix_video_for_model,
                    original_input_ref_images = original_image_refs[nb_frames_positions:] if original_image_refs is not None else [],
                    image_refs_relative_size = image_refs_relative_size,
                    outpainting_dims = model_outpainting_dims,
                    face_arc_embeds = face_arc_embeds,
                    custom_settings=custom_settings_for_model,
                    temperature=temperature,
                    window_start_frame_no = window_start_frame,
                    input_video_strength = input_video_strength,
                    self_refiner_setting = self_refiner_setting,
                    self_refiner_plan=self_refiner_plan,
                    self_refiner_f_uncertainty = self_refiner_f_uncertainty,
                    self_refiner_certain_percentage = self_refiner_certain_percentage,
                    duration_seconds=duration_seconds,
                    pause_seconds=pause_seconds,
                    top_p=top_p,
                    top_k=top_k,
                    set_progress_status=set_progress_status,
                    loras_selected=loras_selected,
                    frames_relative_positions_list = frames_relative_positions_list,
                    frames_to_inject = frames_to_inject_parsed,
                )
            except Exception as e:
                if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
                    cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)
                remove_temp_filenames(temp_filenames_list)
                clear_gen_cache()
                offloadobj.unload_all()
                trans.cache = None 
                if trans2 is not None: 
                    trans2.cache = None 
                offload.unload_loras_from_model(trans_lora)
                if trans2_lora is not None: 
                    offload.unload_loras_from_model(trans2_lora)
                skip_steps_cache = None
                # if compile:
                #     cache_size = torch._dynamo.config.cache_size_limit                                      
                #     torch.compiler.reset()
                #     torch._dynamo.config.cache_size_limit = cache_size

                gc.collect()
                torch.cuda.empty_cache()
                s = str(e)
                keyword_list = {"CUDA out of memory" : "VRAM", "Tried to allocate":"VRAM", "CUDA error: out of memory": "RAM", "CUDA error: too many resources requested": "RAM"}
                crash_type = ""
                for keyword, tp  in keyword_list.items():
                    if keyword in s:
                        crash_type = tp 
                        break
                state["prompt"] = ""
                if crash_type == "VRAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames."
                elif crash_type == "RAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient RAM and / or Reserved RAM allocation should be reduced using 'perc_reserved_mem_max' or using a different Profile."
                else:
                    new_error =  gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))
                send_cmd("error", new_error)
                clear_status(state)
                return False
            src_video = src_video2 = src_mask = src_mask2 = None
            if skip_steps_cache != None :
                skip_steps_cache.previous_residual = None
                skip_steps_cache.previous_modulated_input = None
                print(f"Skipped Steps:{skip_steps_cache.skipped_steps}/{skip_steps_cache.num_steps}" )
            generated_audio = None
            drop_generated_audio = False
            BGRA_frames = None
            post_decode_pre_trim = 0
            output_audio_sampling_rate= audio_sampling_rate
            sample_is_hdr = False
            if samples != None:
                if isinstance(samples, dict):
                    sample_is_hdr = samples.get("hdr", False)
                    overlapped_latents = samples.get("latent_slice", None)
                    BGRA_frames = samples.get("BGRA_frames", None)
                    generated_audio = samples.get("audio", generated_audio)
                    overridden_inputs = samples.get("overridden_inputs", None)
                    output_audio_sampling_rate = samples.get("audio_sampling_rate", audio_sampling_rate)
                    input_fills_window = input_waveform is not None and input_waveform.shape[0] >= int(round(current_video_length * input_waveform_sample_rate / fps))
                    if generated_audio is not None:
                        if model_def.get("output_audio_is_input_audio", False) and output_new_audio_filepath is not None and "O" not in audio_prompt_type and input_fills_window:
                            drop_generated_audio = True
                        elif input_fills_window:
                            output_new_audio_filepath = None
                    post_decode_pre_trim = samples.get("post_decode_pre_trim", 0) 
                    samples = samples.get("x", None)

                if samples is not None:
                    samples = samples.to("cpu")
  
            clear_gen_cache()
            offloadobj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            if samples == None:
                abort = True
                state["prompt"] = ""
                send_cmd("output")  
            else:
                sample = samples.cpu()
                abort = abort_scheduled or not (is_image or audio_only) and sample.shape[1] < current_video_length    
                # if True: # for testing
                #     torch.save(sample, "output.pt")
                # else:
                #     sample =torch.load("output.pt")
                if post_decode_pre_trim > 0 :
                    sample = sample[:, post_decode_pre_trim:]
                if gen.get("extra_windows",0) > 0:
                    sliding_window = True 
                if sliding_window :
                    guide_start_frame += current_video_length
                    if discard_last_frames > 0:
                        sample = sample[: , :-discard_last_frames]
                        guide_start_frame -= discard_last_frames
                        if generated_audio is not None:
                            generated_audio = truncate_audio( generated_audio, 0, discard_last_frames, fps, output_audio_sampling_rate,)
                    if generated_audio is not None and reuse_frames > 0 and not drop_generated_audio:
                        pre_audio_guide = generated_audio[-int(round(reuse_frames * output_audio_sampling_rate / fps)):]
                        pre_audio_guide_sample_rate = output_audio_sampling_rate
                    else:
                        pre_audio_guide, pre_audio_guide_sample_rate = None, 0

                    if reuse_frames == 0:
                        pre_video_guide =  sample[:,max_source_video_frames :].clone()
                    else:
                        pre_video_guide =  sample[:, -reuse_frames:].clone()
                    pre_video_guide_is_hdr = sample_is_hdr
                    if pre_video_guide.dtype == torch.uint8:
                        pre_video_guide =  pre_video_guide.float().div_(127.5).sub_(1.0)
                if not (audio_only or is_image):                    
                    if not sample_is_hdr:
                        sample = _video_tensor_to_uint8_chunk_inplace(sample)

                if prefix_video != None and window_no == 1 :
                    if sample_is_hdr:
                        prefix_was_uint8 = prefix_video.dtype == torch.uint8
                        prefix_video = prefix_video.float()
                        if prefix_was_uint8:
                            prefix_video = prefix_video.div_(255.0)
                        elif not input_video_is_hdr and torch.is_floating_point(prefix_video):
                            prefix_video = prefix_video.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
                        prefix_video = prefix_video.to(dtype=sample.dtype)
                    elif prefix_video.dtype != sample.dtype:
                        if sample.dtype == torch.uint8:
                            prefix_video = _video_tensor_to_uint8_chunk_inplace(prefix_video)
                        elif prefix_video.dtype == torch.uint8:
                            prefix_video = prefix_video.float().div_(127.5).sub_(1.0)
                    if prefix_video.shape[1] > 1:
                        # remove sliding window overlapped frames at the beginning of the generation
                        sample = torch.cat([ prefix_video, sample[: , source_video_overlap_frames_count:]], dim = 1)
                    else:
                        # remove source video overlapped frames at the beginning of the generation if there is only a start frame
                        sample = torch.cat([ prefix_video[:, :-source_video_overlap_frames_count], sample], dim = 1)
                    prefix_video = None
                    guide_start_frame -= source_video_overlap_frames_count 
                    if generated_audio is not None:
                        generated_audio = truncate_audio( generated_audio, 0 if video_source is None else source_video_overlap_frames_count, 0, fps, output_audio_sampling_rate,)
                elif sliding_window and window_no > 1 and reuse_frames > 0:
                    # remove sliding window overlapped frames at the beginning of the generation
                    sample = sample[: , reuse_frames:]
                    guide_start_frame -= reuse_frames 
                    if generated_audio is not None:
                        generated_audio = truncate_audio( generated_audio, reuse_frames, 0, fps, output_audio_sampling_rate,)

                num_frames_generated = guide_start_frame - (source_video_frames_count - source_video_overlap_frames_count) 
                if drop_generated_audio: generated_audio = None
                if generated_audio is not None:
                    committed_audio_samples = int(round((num_frames_generated - sample.shape[1]) * output_audio_sampling_rate / fps))
                    if full_generated_audio is None:
                        full_generated_audio = append_sliding_window_audio(output_new_audio_data, output_new_audio_filepath, generated_audio, output_audio_sampling_rate, committed_audio_samples) if output_new_audio_data is not None or output_new_audio_filepath is not None else generated_audio
                    else:
                        full_generated_audio = np.concatenate([full_generated_audio, generated_audio], axis=0)
                    output_new_audio_data = full_generated_audio


                if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 and not "vae2" in spatial_upsampling:                
                    send_cmd("progress", [0, get_latest_status(state,"Upsampling")])
                
                output_fps  = fps
                if len(temporal_upsampling) > 0:
                    sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, previous_last_frame if sliding_window and window_no > 1 else None, temporal_upsampling, fps)

                if len(spatial_upsampling) > 0:
                    sample = perform_spatial_upsampling(sample, spatial_upsampling )
                if film_grain_intensity> 0:
                    from postprocessing.film_grain import add_film_grain
                    sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
                mmaudio_enabled, mmaudio_mode, mmaudio_persistence, mmaudio_model_name, mmaudio_model_path = get_mmaudio_settings(server_config)
                if audio_only or is_image:
                    output_video_frames = None
                    output_frame_count = None
                    any_mmaudio = False
                else:
                    frames_already_processed.append(sample)
                    frames_already_processed_count += sample.shape[1]
                    output_video_frames = frames_already_processed
                    output_frame_count = frames_already_processed_count
                    sample = None
                    any_mmaudio = MMAudio_setting != 0 and mmaudio_enabled and output_frame_count >= fps
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                save_prompt = original_prompts[0]
                if audio_only:
                    audio_codec = server_config.get("audio_stand_alone_output_codec", "wav")
                    extension = get_audio_codec_extension(audio_codec)
                    output_dir = audio_save_path
                elif is_image:
                    extension = "jpg"
                    output_dir = image_save_path
                else:
                    container = server_config.get("video_container", "mp4")
                    extension = container
                    output_dir = save_path
                inputs = get_function_arguments(generate_video, locals())
                if overridden_inputs is not None: inputs.update(overridden_inputs)
                if len(output_filename):
                    from shared.utils.filename_formatter import FilenameFormatter
                    file_name = FilenameFormatter.format_filename(output_filename, inputs)                    
                    file_name = f"{sanitize_file_name(truncate_for_filesystem(os.path.splitext(os.path.basename(file_name))[0])).strip()}.{extension}"
                    file_name = os.path.basename(get_available_filename(output_dir, file_name))
                else:
                    file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(truncate_for_filesystem(save_prompt)).strip()}.{extension}"
                video_path = os.path.join(output_dir, file_name)

                if BGRA_frames is not None:
                    from models.wan.alpha.utils import write_zip_file
                    write_zip_file(os.path.splitext(video_path)[0] + ".zip", BGRA_frames)
                    BGRA_frames = None 
                if audio_only:
                    audio_path = os.path.join(output_dir, file_name)
                    audio_path = save_audio_file(audio_path, sample.squeeze(0), output_audio_sampling_rate, audio_codec)
                    video_path = audio_path
                elif is_image:
                    image_path = os.path.join(output_dir, file_name)
                    sample =  sample.transpose(1,0)  #c f h w -> f c h w 
                    new_image_path = []
                    for no, img in enumerate(sample):  
                        img_path = os.path.splitext(image_path)[0] + ("" if no==0 else f"_{no}") + ".jpg" 
                        new_image_path.append(save_image(img, save_file = img_path, quality = server_config.get("image_output_codec", None)))

                    video_path= new_image_path
                elif len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0 or output_new_audio_filepath is not None or any_mmaudio or output_new_audio_data is not None or audio_source is not None:
                    video_path = os.path.join(save_path, file_name)
                    save_path_tmp = video_path.rsplit('.', 1)[0] + f"_tmp.{container}"
                    if sample_is_hdr:
                        save_hdr_video(tensor=output_video_frames, save_file=save_path_tmp, fps=output_fps, codec_type=server_config.get("hdr_video_crf", 8), container=container)
                    else:
                        save_video( tensor=output_video_frames, save_file=save_path_tmp, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type = server_config.get("video_output_codec", None), container=container)
                    output_new_audio_temp_filepath = None
                    new_audio_added_from_audio_start = reset_control_aligment or (full_generated_audio is not None and len(control_audio_tracks) == 0) # if not beginning of audio will be skipped
                    source_audio_duration = 0 if video_source is None else source_video_frames_count / fps
                    if any_mmaudio:
                        send_cmd("progress", [0, get_latest_status(state,"MMAudio Soundtrack Generation")])
                        from postprocessing.mmaudio.mmaudio import video_to_audio
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        video_to_audio(save_path_tmp, prompt = MMAudio_prompt, negative_prompt = MMAudio_neg_prompt, seed = seed, num_steps = 25, cfg_strength = 4.5, duration= output_frame_count / fps, save_path = output_new_audio_filepath, persistent_models = mmaudio_persistence == MMAUDIO_PERSIST_RAM, audio_file_only = True, verboseLevel = verbose_level, model_name = mmaudio_model_name, model_path = mmaudio_model_path)
                        new_audio_added_from_audio_start =  False
                    elif audio_source is not None:
                        output_new_audio_filepath = audio_source
                        new_audio_added_from_audio_start =  True
                    elif output_new_audio_data is not None and len(control_audio_tracks) == 0:
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        write_wav_file(output_new_audio_filepath, output_new_audio_data, output_audio_sampling_rate)
                    if output_new_audio_filepath is not None:
                        new_audio_tracks = [output_new_audio_filepath]
                    else:
                        new_audio_tracks = control_audio_tracks
                    if generated_audio is not None: output_new_audio_filepath = None
                    mux_audio_sampling_rate = resolve_mux_audio_sampling_rate(output_audio_sampling_rate, source_audio_metadata, new_audio_tracks)

                    combine_and_concatenate_video_with_audio_tracks(
                        video_path,
                        save_path_tmp,
                        source_audio_tracks,
                        new_audio_tracks,
                        source_audio_duration,
                        mux_audio_sampling_rate,
                        new_audio_from_start=new_audio_added_from_audio_start,
                        source_audio_metadata=source_audio_metadata,
                        audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                        verbose=verbose_level >= 2,
                    )
                    os.remove(save_path_tmp)
                    if output_new_audio_temp_filepath is not None: os.remove(output_new_audio_temp_filepath)

                else:
                    if sample_is_hdr:
                        video_path = save_hdr_video(tensor=output_video_frames, save_file=video_path, fps=output_fps, codec_type=server_config.get("hdr_video_crf", 8), container=container)
                    else:
                        save_video( tensor=output_video_frames, save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1),  codec_type= server_config.get("video_output_codec", None), container= container)

                end_time = time.time()

                inputs.pop("send_cmd")
                inputs.pop("task")
                inputs.pop("mode")
                inputs["model_type"] = model_type
                inputs["model_filename"] = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
                if is_image:
                    inputs["image_quality"] = server_config.get("image_output_codec", None)
                else:
                    inputs["video_quality"] = server_config.get("video_output_codec", None)
                if sample_is_hdr:
                    inputs["hdr"] = True
                    inputs["hdr_video_crf"] = server_config.get("hdr_video_crf", 8)
                    inputs["video_quality"] = f"x265_crf_{inputs['hdr_video_crf']}"

                modules = get_model_recursive_prop(model_type, "modules", return_list= True)
                if len(modules) > 0 : inputs["modules"] = modules
                if len(transformer_loras_filenames) > 0:
                    inputs.update({
                    "transformer_loras_filenames" : transformer_loras_filenames,
                    "transformer_loras_multipliers" : transformer_loras_multipliers
                    })
                embedded_images = {img_name: inputs[img_name] for img_name in image_names_list } if server_config.get("embed_source_images", False) else None
                configs = prepare_inputs_dict("metadata", inputs, model_type)
                if sliding_window: configs["window_no"] = window_no
                configs["prompt"] = "\n".join(original_prompts)
                if prompt_enhancer_image_caption_model != None and prompt_enhancer !=None and len(prompt_enhancer)>0 and enhancer_mode != 1:
                    configs["enhanced_prompt"] = "\n".join(prompts)
                configs["generation_time"] = round(end_time-start_time)
                configs["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
                configs["creation_timestamp"] = int(end_time)
                # if sample_is_image: configs["is_image"] = True
                record_file_metadata(video_path, configs, is_image, audio_only, gen, embedded_images=embedded_images, replace_last_file=sliding_window and window_no > 1 and not server_config.get("keep_intermediate_sliding_windows", 1))
                if api_return_video_uint8 or api_return_audio:
                    media_type = "audio" if audio_only else ("image" if is_image else "video")
                    artifact_audio = output_new_audio_data if api_return_audio else None
                    if artifact_audio is None and api_return_audio and audio_only and sample is not None:
                        artifact_audio = sample.squeeze(0).detach().cpu().float().numpy()
                    artifact_video = output_video_frames if api_return_video_uint8 else None
                    store_api_output_artifact(gen, client_id, video_path, media_type, artifact_video, artifact_audio, output_audio_sampling_rate, output_fps if not audio_only else None, hdr=bool(sample_is_hdr))

                embedded_images = None
                # Play notification sound for single video
                try:
                    if server_config.get("notification_sound_enabled", 0):
                        volume = server_config.get("notification_sound_volume", 50)
                        notification_sound.notify_video_completion(
                            video_path=video_path, 
                            volume=volume
                        )
                except Exception as e:
                    print(f"Error playing notification sound for individual video: {e}")

                send_cmd("output")

        seed = set_seed(-1)
    clear_status(state)
    trans.cache = None
    offload.unload_loras_from_model(trans_lora)
    if not trans2_lora is None:
        offload.unload_loras_from_model(trans2_lora)

    if not trans2 is None:
       trans2.cache = None
 
    if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
        cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)

    remove_temp_filenames(temp_filenames_list)
    return True

def prepare_generate_video(state):    

    if state.get("validate_success",0) != 1:
        return gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.update(visible=False)
    else:
        return gr.Button(visible= False), gr.Button(visible= True), gr.Column(visible= True), gr.update(visible= False)


def generate_preview(model_type, payload):
    import einops
    if payload is None:
        return None
    if isinstance(payload, dict):
        meta = {k: v for k, v in payload.items() if k != "latents"}
        latents = payload.get("latents")
    else:
        meta = {}
        latents = payload
    if latents is None:
        return None
    # latents shape should be C, T, H, W (no batch)    
    if not torch.is_tensor(latents):
        return None
    model_handler = get_model_handler(model_type)
    base_model_type = get_base_model_type(model_type)
    custom_preview = getattr(model_handler, "preview_latents", None)
    if callable(custom_preview):
        preview = custom_preview(base_model_type, latents, meta)
        if preview is not None:
            return preview
    if hasattr(model_handler, "get_rgb_factors"):
        latent_rgb_factors, latent_rgb_factors_bias = model_handler.get_rgb_factors(base_model_type )
    else:
        return None
    if latent_rgb_factors is None: return None
    latents = latents.unsqueeze(0) 
    nb_latents = latents.shape[2]
    latents_to_preview = 4
    latents_to_preview = min(nb_latents, latents_to_preview)
    skip_latent =  nb_latents / latents_to_preview
    latent_no = 0
    selected_latents = []
    while latent_no < nb_latents:
        selected_latents.append( latents[:, : , int(latent_no): int(latent_no)+1])
        latent_no += skip_latent 

    latents = torch.cat(selected_latents, dim = 2)
    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.add_(1.0).mul_(127.5)
    images = images.detach().cpu()
    if images.dtype == torch.bfloat16:
        images = images.to(torch.float16)
    images = images.numpy().clip(0, 255).astype(np.uint8)
    images = einops.rearrange(images, 'b c t h w -> (b h) (t w) c')
    h, w, _ = images.shape
    scale = 200 / h
    images= Image.fromarray(images)
    images = images.resize(( int(w*scale),int(h*scale)), resample=Image.Resampling.BILINEAR) 
    return images


def process_tasks(state):
    from shared.utils.thread_utils import AsyncStream, async_run_in

    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    progress = None

    if len(queue) == 0:
        gen["status_display"] =  False
        return
    with lock:
        gen = get_gen_info(state)
        clear_file_list = server_config.get("clear_file_list", 0)

        def truncate_list(file_list, file_settings_list, choice):
            if clear_file_list > 0:
                file_list_current_size = len(file_list)
                keep_file_from = max(file_list_current_size - clear_file_list, 0)
                files_removed = keep_file_from
                choice = max(choice- files_removed, 0)
                file_list = file_list[ keep_file_from: ]
                file_settings_list = file_settings_list[ keep_file_from: ]
            else:
                file_list = []
                choice = 0
            return file_list, file_settings_list, choice
        
        file_list = gen.get("file_list", [])
        file_settings_list = gen.get("file_settings_list", [])
        choice = gen.get("selected",0)
        gen["file_list"], gen["file_settings_list"], gen["selected"] = truncate_list(file_list, file_settings_list, choice)         

        audio_file_list = gen.get("audio_file_list", [])
        audio_file_settings_list = gen.get("audio_file_settings_list", [])
        audio_choice = gen.get("audio_selected",-1)
        gen["audio_file_list"], gen["audio_file_settings_list"], gen["audio_selected"] = truncate_list(audio_file_list, audio_file_settings_list, audio_choice)         

    set_main_generation_running(state, True)
    acquire_main_GPU_ressources(state)

    def release_gen():
        set_main_generation_running(state, False)
        with gen_lock:
            process_status = gen.get("process_status", None)
            if isinstance(process_status, str) and process_status.startswith("request:"):
                gen["process_status"] = "process:" + process_status[len("request:"):]
            else:
                gen["process_status"] = None

    start_time = time.time()

    global gen_in_progress
    gen_in_progress = True
    gen["in_progress"] = True
    gen["preview"] = None
    gen["status"] = "Generating..."
    gen["header_text"] = ""    

    yield time.time(), time.time(), gr.update()

    com_stream = AsyncStream()
    send_cmd = com_stream.output_queue.push

    def queue_worker_func():
        gen["prompt_no"] = 0
        try:
            while len(queue) > 0:
                paused_for_edit = False
                while gen.get("queue_paused_for_edit", False):
                    if not paused_for_edit:
                        send_cmd("info", "Queue Paused until Current Task Edition is Done")
                        send_cmd("status", "Queue paused for editing...")
                        send_cmd("output", None) 
                        paused_for_edit = True
                    time.sleep(0.5)
                
                if paused_for_edit:
                    send_cmd("status", "Resuming queue processing...")
                    send_cmd("output", None)

                gen["prompt_no"] += 1

                task = None
                with lock:
                    if len(queue) > 0:
                        task = queue[0]

                if task is None:
                    break

                task_id = task["id"] 
                params = task['params']
                for key in ["model_filename", "lset_name"]:
                    params.pop(key, None)
                
                try:
                    import inspect
                    model_type = params.get('model_type')
                    if model_type:
                        default_settings = get_default_settings(model_type)
                        expected_args = set(inspect.signature(generate_video).parameters.keys())
                        for arg_name in expected_args:
                            if arg_name not in params and arg_name in default_settings:
                                params[arg_name] = default_settings[arg_name]
                    else:
                        expected_args = set(inspect.signature(generate_video).parameters.keys())                    
                    filtered_params = {k: v for k, v in params.items() if k in expected_args}
                    plugin_data = task.pop('plugin_data', {})
                    success = generate_video(task, send_cmd, plugin_data=plugin_data,  **filtered_params)
                    
                except Exception as e:
                    tb = traceback.format_exc().split('\n')[:-1] 
                    print('\n'.join(tb))
                    send_cmd("error", str(e))
                    return

                abort = gen.get("abort", False)
                if abort:
                    record_queue_error(state, queue[:1], "abort", abort=True)
                    gen["abort"] = False
                    send_cmd("status", "Video Generation Aborted")
                    send_cmd("output", None)

                gen["early_stop"] = False
                gen["early_stop_forwarded"] = False
                if not success: break
                with lock:
                    queue[:] = [item for item in queue if item['id'] != task_id]
                update_global_queue_ref(queue)
                
        except Exception as e:
            traceback.print_exc()
            send_cmd("error", f"Queue worker crashed: {e}")
        finally:
            send_cmd("worker_exit", None)

    async_run_in("generation", queue_worker_func)

    while True:
        cmd, data = com_stream.output_queue.next()               
        if cmd == "exit":
            pass
        elif cmd == "worker_exit":
            break
        elif cmd == "info":
            gr.Info(data)
        elif cmd == "error": 
            record_queue_error(state, queue, data)
            queue.clear()
            try:
                save_queue_if_crash = server_config.get("save_queue_if_crash", 1)
                if save_queue_if_crash:
                    error_filename = AUTOSAVE_ERROR_FILENAME if save_queue_if_crash == 1 else get_available_filename("", AUTOSAVE_ERROR_FILENAME, f"_{datetime.now():%Y%m%d_%H%M%S}")
                    if _save_queue_to_zip(global_queue_ref, error_filename):
                        print(f"Error Queue autosaved successfully to {error_filename}")
                        gr.Info(f"Error Queue autosaved successfully to {error_filename}")
                    else:
                        print("Autosave Error Queue failed.")
            except Exception as e:
                print(f"Error during autosave: {e}")

            update_global_queue_ref(queue)
            gen["prompts_max"] = 0
            gen["prompt"] = ""
            gen["status_display"] =  False
            release_gen()
            raise gr.Error(data, print_exception= False, duration = 0)
        elif cmd == "status":
            gen["status"] = data
            status_text = str(data or "").strip()
            gen["last_progress_args"] = [0, status_text] if len(status_text) > 0 else None
        elif cmd == "output":
            gen["preview"] = None
            gen["refresh_tab"] = True
            yield time.time(), time.time(), gr.update()
        elif cmd == "progress":
            gen["last_progress_args"] = gen["progress_args"] = data
        elif cmd == "preview":
            current_model_type = "unknown"
            with lock:
                if len(queue) > 0:
                    current_model_type = queue[0]["params"].get("model_type")
            
            try:
                torch.cuda.current_stream().synchronize()
                preview = None if data is None else generate_preview(current_model_type, data) 
                gen["preview"] = preview
                yield time.time(), gr.Text(), gr.update()
            except Exception:
                pass
        elif cmd == "refresh_models":
            yield gr.update(), gr.update(), (data if data is not None else get_unique_id())
        else:
            pass

    gen["prompts_max"] = 0
    gen["prompt"] = ""
    end_time = time.time()
    if gen.get("abort", False):
        record_queue_error(state, queue[:1], "abort", abort=True)
        status = f"Video generation was aborted. Total Generation Time: {format_time(end_time-start_time)}" 
    else:
        status = f"Total Generation Time: {format_time(end_time-start_time)}"
        try:
            if server_config.get("notification_sound_enabled", 1):
                volume = server_config.get("notification_sound_volume", 50)
                notification_sound.notify_video_completion(volume=volume)
        except Exception as e:
            print(f"Error playing notification sound: {e}")
    gen["status"] = status
    gen["status_display"] =  False
    release_gen()


def validate_task(task, state):
    """Validate a task's settings. Returns (updated params dict or None, validation error)."""
    params = task.get('params', {})
    model_type = params.get('model_type')
    if not model_type:
        print("  [SKIP] No model_type specified")
        return None, "No model_type specified"

    inputs = params.copy()
    clean_settings(model_type, inputs)

    inputs.setdefault('mode', "")
    override_inputs, _, _, _, validation_error = validate_settings(state, model_type, single_prompt=True, inputs=inputs, silent=True)
    if override_inputs is None:
        return None, validation_error or "Task failed validation."
    inputs.update(override_inputs)
    return inputs, ""


def process_tasks_cli(queue, state):
    """Process queue tasks with console output for CLI mode. Returns True on success."""
    from shared.utils.thread_utils import AsyncStream, async_run_in
    import inspect

    gen = get_gen_info(state)
    total_tasks = len(queue)
    completed = 0
    skipped = 0
    start_time = time.time()

    for task_idx, task in enumerate(queue):
        task_no = task_idx + 1
        prompt_preview = (task.get('prompt', '') or '')[:60]
        print(f"\n[Task {task_no}/{total_tasks}] {prompt_preview}...")

        # Validate task settings before processing
        validated_params, validation_error = validate_task(task, state)
        if validated_params is None:
            print(f"  [SKIP] Task {task_no} failed validation: {validation_error or 'Task failed validation.'}")
            skipped += 1
            continue

        # Update gen state for this task
        gen["prompt_no"] = task_no
        gen["prompts_max"] = total_tasks

        params = validated_params.copy()
        params['state'] = state

        com_stream = AsyncStream()
        send_cmd = com_stream.output_queue.push

        def make_error_handler(task, params, send_cmd):
            def error_handler():
                try:
                    # Filter to only valid generate_video params
                    expected_args = set(inspect.signature(generate_video).parameters.keys())
                    filtered_params = {k: v for k, v in params.items() if k in expected_args}
                    filtered_params.setdefault("client_id", "")
                    plugin_data = task.get('plugin_data', {})
                    generate_video(task, send_cmd, plugin_data=plugin_data, **filtered_params)
                except Exception as e:
                    print(f"\n  [ERROR] {e}")
                    traceback.print_exc()
                    send_cmd("error", str(e))
                finally:
                    send_cmd("exit", None)
            return error_handler

        async_run_in("generation", make_error_handler(task, params, send_cmd))

        # Process output stream
        task_error = False
        last_msg_len = 0
        in_status_line = False  # Track if we're in an overwritable line
        while True:
            cmd, data = com_stream.output_queue.next()
            if cmd == "exit":
                if in_status_line:
                    print()  # End the status line
                break
            elif cmd == "error":
                print(f"\n  [ERROR] {data}")
                in_status_line = False
                task_error = True
            elif cmd == "progress":
                if isinstance(data, list) and len(data) >= 2:
                    if isinstance(data[0], tuple):
                        step, total = data[0]
                        msg = data[1] if len(data) > 1 else ""
                    else:
                        step, msg = 0, data[1] if len(data) > 1 else str(data[0])
                        total = 1
                    status_line = f"\r  [{step}/{total}] {msg}"
                    # Pad to clear previous longer messages
                    print(status_line.ljust(max(last_msg_len, len(status_line))), end="", flush=True)
                    last_msg_len = len(status_line)
                    in_status_line = True
            elif cmd == "status":
                # "Loading..." messages are followed by external library output, so end with newline
                if "Loading" in str(data):
                    print(data)
                    in_status_line = False
                    last_msg_len = 0
                else:
                    status_line = f"\r  {data}"
                    print(status_line.ljust(max(last_msg_len, len(status_line))), end="", flush=True)
                    last_msg_len = len(status_line)
                    in_status_line = True
            elif cmd == "output":
                # "output" is used for UI refresh, not just video saves - don't print anything
                pass
            elif cmd == "info":
                print(f"\n  [INFO] {data}")
                in_status_line = False

        if not task_error:
            completed += 1
            print(f"\n  Task {task_no} completed")

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    summary = f"Queue completed: {completed}/{total_tasks} tasks in {format_time(elapsed)}"
    if skipped > 0:
        summary += f" ({skipped} skipped)"
    print(summary)
    return completed == (total_tasks - skipped)


def get_generation_status(prompt_no, prompts_max, repeat_no, repeat_max, window_no, total_windows):
    if prompts_max == 1:        
        if repeat_max <= 1:
            status = ""
        else:
            status = f"Sample {repeat_no}/{repeat_max}"
    else:
        if repeat_max <= 1:
            status = f"Prompt {prompt_no}/{prompts_max}"
        else:
            status = f"Prompt {prompt_no}/{prompts_max}, Sample {repeat_no}/{repeat_max}"
    if total_windows > 1:
        if len(status) > 0:
            status += ", "
        status += f"Sliding Window {window_no}/{total_windows}"

    return status

refresh_id = 0

def get_new_refresh_id():
    global refresh_id
    refresh_id += 1
    return refresh_id

def merge_status_context(status="", context=""):
    if len(status) == 0:
        return context
    elif len(context) == 0:
        return status
    else:
        # Check if context already contains the time
        if "|" in context:
            parts = context.split("|")
            return f"{status} - {parts[0].strip()} | {parts[1].strip()}"
        else:
            return f"{status} - {context}"
        
def clear_status(state):
    gen = get_gen_info(state)
    gen["extra_windows"] = 0
    gen["total_windows"] = 1
    gen["window_no"] = 1
    gen["extra_orders"] = 0
    gen["repeat_no"] = 0
    gen["total_generation"] = 0

def get_latest_status(state, context=""):
    gen = get_gen_info(state)
    prompt_no = gen["prompt_no"] 
    prompts_max = gen.get("prompts_max",0)
    total_generation = gen.get("total_generation", 1)
    repeat_no = gen.get("repeat_no",0)
    total_generation += gen.get("extra_orders", 0)
    total_windows = gen.get("total_windows", 0)
    total_windows += gen.get("extra_windows", 0)
    window_no = gen.get("window_no", 0)
    status = get_generation_status(prompt_no, prompts_max, repeat_no, total_generation, window_no, total_windows)
    return merge_status_context(status, context)

def update_status(state): 
    gen = get_gen_info(state)
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()


def one_more_sample(state):
    gen = get_gen_info(state)
    extra_orders = gen.get("extra_orders", 0)
    extra_orders += 1
    gen["extra_orders"]  = extra_orders
    in_progress = gen.get("in_progress", False)
    if not in_progress :
        return state
    total_generation = gen.get("total_generation", 0) + extra_orders
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()
    gr.Info(f"An extra sample generation is planned for a total of {total_generation} samples for this prompt")

    return state 

def one_more_window(state):
    gen = get_gen_info(state)
    extra_windows = gen.get("extra_windows", 0)
    extra_windows += 1
    gen["extra_windows"]= extra_windows
    in_progress = gen.get("in_progress", False)
    if not in_progress :
        return state
    total_windows = gen.get("total_windows", 0) + extra_windows
    gen["progress_status"] = get_latest_status(state)
    gen["refresh"] = get_new_refresh_id()
    gr.Info(f"An extra window generation is planned for a total of {total_windows} videos for this sample")

    return state 

def get_new_preset_msg(advanced = True):
    if advanced:
        return "Enter here a Name for a Lora Preset or a Settings or Choose one"
    else:
        return "Choose a Lora Preset or a Settings file in this List"


def _normalize_builtin_lset_dirs(settings_dirs):
    if settings_dirs is None:
        return []
    if isinstance(settings_dirs, str):
        return [settings_dirs] if len(settings_dirs) > 0 else []
    return [str(dir) for dir in settings_dirs if isinstance(dir, str) and len(dir) > 0]


def _get_builtin_lset_groups(model_type):
    if model_type is None:
        return []
    top_dir = "profiles"
    group_defs = [
        ("accelerator_profiles", "Accelerators Profiles", "profiles_dir"),
        ("preset_settings", "Presets", "preset_profiles_dir"),
    ]
    builtin_groups = []
    for group_id, title, prop_name in group_defs:
        paths = []
        for dir_name in _normalize_builtin_lset_dirs(get_model_recursive_prop(model_type, prop_name, return_list=False)):
            cur_path = os.path.join(top_dir, dir_name)
            if not os.path.isdir(cur_path):
                continue
            cur_dir_presets = glob.glob(os.path.join(cur_path, "*.json"))
            paths += [os.path.join(dir_name, os.path.basename(path)) for path in cur_dir_presets]
        paths = sorted(dict.fromkeys(paths), key=lambda n: os.path.basename(n).lower())
        if len(paths) > 0:
            builtin_groups.append((group_id, title, paths))
    return builtin_groups


def _get_builtin_lset_type(model_type, lset_name):
    for group_id, _, paths in _get_builtin_lset_groups(model_type):
        if lset_name in paths:
            return group_id
    return None


def compute_lset_choices(model_type, loras_presets):
    lset_list = []
    settings_list = []
    for item in loras_presets:
        if item.endswith(".lset"):
            lset_list.append(item)
        else:
            settings_list.append(item)

    sep = '\u2500' 
    indent = chr(160) * 4
    lset_choices = []
    for group_id, title, paths in _get_builtin_lset_groups(model_type):
        header_value = f">{group_id}"
        left_sep = 12 if group_id == "accelerator_profiles" else 17
        right_sep = 13 if group_id == "accelerator_profiles" else 17
        lset_choices += [((sep * left_sep) + title + (sep * right_sep), header_value)]
        lset_choices += [(indent + os.path.splitext(os.path.basename(preset))[0], preset) for preset in paths]
    if len(settings_list) > 0:
        settings_list.sort()
        lset_choices += [( (sep*16) +"Settings" + (sep*17), ">settings")]
        lset_choices += [ ( indent   + os.path.splitext(preset)[0], preset) for preset in settings_list ]
    if len(lset_list) > 0:
        lset_list.sort()
        lset_choices += [( (sep*18) + "Lsets" + (sep*18), ">lset")]
        lset_choices += [ ( indent   + os.path.splitext(preset)[0], preset) for preset in lset_list ]
    return lset_choices

def get_lset_name(state, lset_name):
    presets = state["loras_presets"]
    if len(lset_name) == 0 or lset_name.startswith(">") or lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False): return ""
    if lset_name in presets: return lset_name
    model_type = get_state_model_type(state)
    choices = compute_lset_choices(model_type, presets)
    for label, value in choices:
        if label == lset_name: return value
    return lset_name

def validate_delete_lset(state, lset_name):
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0 :
        gr.Info(f"Choose a Preset to delete")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False) 
    elif "/" in lset_name or "\\" in lset_name:
        gr.Info(f"You can't delete a built-in profile or preset")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Checkbox(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True) 
    
def validate_save_lset(state, lset_name):
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please enter a name for the preset")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False),gr.Checkbox(visible= False) 
    elif "/" in lset_name or "\\" in lset_name:
        gr.Info(f"You can't edit a built-in profile or preset")
        return  gr.Button(visible= True), gr.Checkbox(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False),gr.Checkbox(visible= False) 
    else:
        return  gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= True), gr.Button(visible= True),gr.Checkbox(visible= True)

def cancel_lset():
    return gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)


def save_lset(state, lset_name, loras_choices, loras_mult_choices, prompt, save_lset_prompt_cbox):    
    if lset_name.endswith(".json") or lset_name.endswith(".lset"):
        lset_name = os.path.splitext(lset_name)[0]

    loras_presets = state["loras_presets"] 
    loras = state["loras"]
    if state.get("validate_success",0) == 0:
        pass
    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please enter a name for the preset / settings file")
        lset_choices =[("Please enter a name for a Lora Preset / Settings file","")]
    else:
        lset_name = sanitize_file_name(lset_name)
        lset_name = lset_name.replace('\u2500',"").strip()


        if save_lset_prompt_cbox ==2:
            lset = collect_current_model_settings(state)
            extension = ".json" 
        else:
            from shared.utils.loras_mutipliers import extract_loras_side
            loras_choices, loras_mult_choices = extract_loras_side(loras_choices, loras_mult_choices, "after")
            lset  = {"loras" : loras_choices, "loras_mult" : loras_mult_choices}
            if save_lset_prompt_cbox!=1:
                prompts = prompt.replace("\r", "").split("\n")
                prompts = [prompt for prompt in prompts if len(prompt)> 0 and prompt.startswith("#")]
                prompt = "\n".join(prompts)
            if len(prompt) > 0:
                lset["prompt"] = prompt
            lset["full_prompt"] = save_lset_prompt_cbox ==1
            extension = ".lset" 
        
        if lset_name.endswith(".json") or lset_name.endswith(".lset"): lset_name = os.path.splitext(lset_name)[0]
        old_lset_name = lset_name + ".json"
        if not old_lset_name in loras_presets:
            old_lset_name = lset_name + ".lset"
            if not old_lset_name in loras_presets: old_lset_name = ""
        lset_name = lset_name + extension

        model_type = get_state_model_type(state)
        lora_dir = get_lora_dir(model_type)
        full_lset_name_filename = os.path.join(lora_dir, lset_name ) 

        with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(lset, indent=4))

        if len(old_lset_name) > 0 :
            if save_lset_prompt_cbox ==2:
                gr.Info(f"Settings File '{lset_name}' has been updated")
            else:
                gr.Info(f"Lora Preset '{lset_name}' has been updated")
            if old_lset_name != lset_name:
                pos = loras_presets.index(old_lset_name)
                loras_presets[pos] = lset_name 
                shutil.move( os.path.join(lora_dir, old_lset_name),  get_available_filename(lora_dir, old_lset_name + ".bkp" ) ) 
        else:
            if save_lset_prompt_cbox ==2:
                gr.Info(f"Settings File '{lset_name}' has been created")
            else:
                gr.Info(f"Lora Preset '{lset_name}' has been created")
            loras_presets.append(lset_name)
        state["loras_presets"] = loras_presets

        lset_choices = compute_lset_choices(model_type, loras_presets)
        lset_choices.append( (get_new_preset_msg(), ""))
    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)

def delete_lset(state, lset_name):
    loras_presets = state["loras_presets"]
    lset_name = get_lset_name(state, lset_name)
    model_type = get_state_model_type(state)
    if len(lset_name) > 0:
        lset_name_filename = os.path.join( get_lora_dir(model_type), sanitize_file_name(lset_name))
        if not os.path.isfile(lset_name_filename):
            gr.Info(f"Preset '{lset_name}' not found ")
            return [gr.update()]*7 
        os.remove(lset_name_filename)
        lset_choices = compute_lset_choices(None, loras_presets)
        pos = next( (i for i, item in enumerate(lset_choices) if item[1]==lset_name ), -1)
        gr.Info(f"Lora Preset '{lset_name}' has been deleted")
        loras_presets.remove(lset_name)
    else:
        pos = -1
        gr.Info(f"Choose a Preset / Settings File to delete")

    state["loras_presets"] = loras_presets

    lset_choices = compute_lset_choices(model_type, loras_presets)
    lset_choices.append((get_new_preset_msg(), ""))
    selected_lset_name = "" if pos < 0 else lset_choices[min(pos, len(lset_choices)-1)][1] 
    return  gr.Dropdown(choices=lset_choices, value= selected_lset_name), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Checkbox(visible= False)

def get_updated_loras_dropdown(loras, loras_choices):
    loras_choices = [os.path.basename(choice) for choice in loras_choices]    
    loras_choices_dict = { choice : True for choice in loras_choices}
    for lora in loras:
        loras_choices_dict.pop(lora, False)
    new_loras = loras[:]
    for choice, _ in loras_choices_dict.items():
        new_loras.append(choice)    

    new_loras_dropdown= [ ( os.path.splitext(choice)[0], choice) for choice in new_loras ]
    return new_loras, new_loras_dropdown

def refresh_lora_list(state, lset_name, loras_choices):
    model_type= get_state_model_type(state)
    loras, loras_presets, _, _, _, _  = setup_loras(model_type, None,  get_lora_dir(model_type), lora_preselected_preset, None)
    state["loras_presets"] = loras_presets
    gc.collect()

    loras, new_loras_dropdown = get_updated_loras_dropdown(loras, loras_choices)
    state["loras"] = loras
    model_type = get_state_model_type(state)    
    lset_choices = compute_lset_choices(model_type, loras_presets)
    lset_choices.append((get_new_preset_msg( state["advanced"]), "")) 
    if not lset_name in loras_presets:
        lset_name = ""
    
    if wan_model != None:
        trans_err = get_transformer_model(wan_model)
        if hasattr(wan_model, "get_trans_lora"):
            trans_err, _ = wan_model.get_trans_lora()
        errors = getattr(trans_err, "_loras_errors", "")
        if errors !=None and len(errors) > 0:
            error_files = [path for path, _ in errors]
            gr.Info("Error while refreshing Lora List, invalid Lora files: " + ", ".join(error_files))
        else:
            gr.Info("Lora List has been refreshed")


    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.Dropdown(choices=new_loras_dropdown, value= loras_choices) 

def update_lset_type(state, lset_name):
    return 1 if lset_name.endswith(".lset") else 2

from shared.utils.loras_mutipliers import merge_loras_settings

def apply_lset(state, wizard_prompt_activated, lset_name, loras_choices, loras_mult_choices, prompt):

    state["apply_success"] = 0

    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please choose a Lora Preset or Setting File in the list or create one")
        return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    else:
        current_model_type = get_state_model_type(state)
        ui_settings = get_current_model_settings(state)
        old_activated_loras, old_loras_multipliers  = ui_settings.get("activated_loras", []), ui_settings.get("loras_multipliers", ""),
        if lset_name.endswith(".lset"):
            loras = state["loras"]
            loras_choices, loras_mult_choices, preset_prompt, full_prompt, error = extract_preset(current_model_type,  lset_name, loras)
            if full_prompt:
                prompt = preset_prompt
            elif len(preset_prompt) > 0:
                prompts = prompt.replace("\r", "").split("\n")
                prompts = [prompt for prompt in prompts if len(prompt)>0 and not prompt.startswith("#")]
                prompt = "\n".join(prompts) 
                prompt = preset_prompt + '\n' + prompt
            loras_choices, loras_mult_choices = merge_loras_settings(old_activated_loras, old_loras_multipliers, loras_choices, loras_mult_choices, "merge after")
            loras_choices = update_loras_url_cache(get_lora_dir(current_model_type), loras_choices)
            loras_choices = [os.path.basename(lora) for lora in loras_choices]
            gr.Info(f"Lora Preset '{lset_name}' has been applied")
            state["apply_success"] = 1
            wizard_prompt_activated = "on"

            return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt, get_unique_id(), gr.update(), gr.update(), gr.update(), gr.update()
        else:
            builtin_lset_type = _get_builtin_lset_type(current_model_type, lset_name)
            accelerator_profile = builtin_lset_type == "accelerator_profiles"
            builtin_preset_settings = builtin_lset_type == "preset_settings"
            lset_path = os.path.join("profiles", lset_name) if builtin_lset_type is not None else os.path.join(get_lora_dir(current_model_type), lset_name)
            merge_loras = "merge before" if accelerator_profile else "merge after"
            configs, _, _ = get_settings_from_file(state,lset_path , True, True, True, min_settings_version=2.38, merge_loras = merge_loras )

            if configs == None:
                gr.Info("File not supported")
                return [gr.update()] * 9
            model_type = configs["model_type"]
            configs["lset_name"] = lset_name
            if accelerator_profile:
                gr.Info(f"Accelerator Profile '{os.path.splitext(os.path.basename(lset_name))[0]}' has been applied")
            elif builtin_preset_settings:
                gr.Info(f"Preset Settings '{os.path.splitext(os.path.basename(lset_name))[0]}' have been applied")
            else:
                gr.Info(f"Settings File '{os.path.basename(lset_name)}' has been applied")
            help = configs.get("help", None)
            if help is not None: gr.Info(help)
            if model_type == current_model_type:
                set_model_settings(state, current_model_type, configs)        
                return *[gr.update()] * 4, gr.update(), gr.update(), gr.update(), gr.update(), get_unique_id()
            else:
                set_model_settings(state, model_type, configs)        
                return *[gr.update()] * 4, gr.update(), *generate_dropdown_model_list(model_type), gr.update()

def extract_prompt_from_wizard(state, variables_names, prompt, wizard_prompt, allow_null_values, *args):

    prompts = wizard_prompt.replace("\r" ,"").split("\n")

    new_prompts = [] 
    macro_already_written = False
    for prompt in prompts:
        if not macro_already_written and not prompt.startswith("#") and "{"  in prompt and "}"  in prompt:
            variables =  variables_names.split("\n")   
            values = args[:len(variables)]
            macro = "! "
            for i, (variable, value) in enumerate(zip(variables, values)):
                if len(value) == 0 and not allow_null_values:
                    return prompt, "You need to provide a value for '" + variable + "'" 
                sub_values= [ "\"" + sub_value + "\"" for sub_value in value.split("\n") ]
                value = ",".join(sub_values)
                if i>0:
                    macro += " : "    
                macro += "{" + variable + "}"+ f"={value}"
            if len(variables) > 0:
                macro_already_written = True
                new_prompts.append(macro)
            new_prompts.append(prompt)
        else:
            new_prompts.append(prompt)

    prompt = "\n".join(new_prompts)
    return prompt, ""

def validate_wizard_prompt(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):
    state["validate_success"] = 0

    if wizard_prompt_activated != "on":
        state["validate_success"] = 1
        return prompt

    prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt, wizard_prompt, False, *args)
    if len(errors) > 0:
        gr.Info(errors)
        return prompt

    state["validate_success"] = 1

    return prompt

def fill_prompt_from_wizard(state, wizard_prompt_activated, wizard_variables_names, prompt, wizard_prompt, *args):

    if wizard_prompt_activated == "on":
        prompt, errors = extract_prompt_from_wizard(state, wizard_variables_names, prompt,  wizard_prompt, True, *args)
        if len(errors) > 0:
            gr.Info(errors)

        wizard_prompt_activated = "off"

    return wizard_prompt_activated, "", gr.Textbox(visible= True, value =prompt) , gr.Textbox(visible= False), gr.Column(visible = True), *[gr.Column(visible = False)] * 2,  *[gr.Textbox(visible= False)] * PROMPT_VARS_MAX

def extract_wizard_prompt(prompt):
    variables = []
    values = {}
    prompts = prompt.replace("\r" ,"").split("\n")
    if sum(prompt.startswith("!") for prompt in prompts) > 1:
        return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

    new_prompts = [] 
    errors = ""
    for prompt in prompts:
        if prompt.startswith("!"):
            variables, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"
            values, errors = prompt_parser.extract_variable_values(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
        else:
            variables_extra, errors = prompt_parser.extract_variable_names(prompt)
            if len(errors) > 0:
                return "", variables, values, "Error parsing Prompt templace: " + errors
            variables += variables_extra
            variables = [var for pos, var in enumerate(variables) if var not in variables[:pos]]
            if len(variables) > PROMPT_VARS_MAX:
                return "", variables, values, "Prompt is too complex for basic Prompt editor, switching to Advanced Prompt"

            new_prompts.append(prompt)
    wizard_prompt = "\n".join(new_prompts)
    return  wizard_prompt, variables, values, errors

def fill_wizard_prompt(state, wizard_prompt_activated, prompt, wizard_prompt):
    def get_hidden_textboxes(num = PROMPT_VARS_MAX ):
        return [gr.Textbox(value="", visible=False)] * num

    hidden_column =  gr.Column(visible = False)
    visible_column =  gr.Column(visible = True)

    wizard_prompt_activated  = "off"  
    if state["advanced"] or state.get("apply_success") != 1:
        return wizard_prompt_activated, gr.Text(), prompt, wizard_prompt, gr.Column(), gr.Column(), hidden_column,  *get_hidden_textboxes() 
    prompt_parts= []

    wizard_prompt, variables, values, errors =  extract_wizard_prompt(prompt)
    if len(errors) > 0:
        gr.Info( errors )
        return wizard_prompt_activated, "", gr.Textbox(prompt, visible=True), gr.Textbox(wizard_prompt, visible=False), visible_column, *[hidden_column] * 2, *get_hidden_textboxes()

    for variable in variables:
        value = values.get(variable, "")
        prompt_parts.append(gr.Textbox( placeholder=variable, info= variable, visible= True, value= "\n".join(value) ))
    any_macro = len(variables) > 0

    prompt_parts += get_hidden_textboxes(PROMPT_VARS_MAX-len(prompt_parts))

    variables_names= "\n".join(variables)
    wizard_prompt_activated  = "on"

    return wizard_prompt_activated, variables_names,  gr.Textbox(prompt, visible = False),  gr.Textbox(wizard_prompt, visible = True),   hidden_column, visible_column, visible_column if any_macro else hidden_column, *prompt_parts

def switch_prompt_type(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars):
    if state["advanced"]:
        return fill_prompt_from_wizard(state, wizard_prompt_activated_var, wizard_variables_names, prompt, wizard_prompt, *prompt_vars)
    else:
        state["apply_success"] = 1
        return fill_wizard_prompt(state, wizard_prompt_activated_var, prompt, wizard_prompt)

visible= False
def switch_advanced(state, new_advanced, lset_name):
    state["advanced"] = new_advanced
    loras_presets = state["loras_presets"]
    model_type = get_state_model_type(state)    
    lset_choices = compute_lset_choices(model_type, loras_presets)
    lset_choices.append((get_new_preset_msg(new_advanced), ""))
    server_config["last_advanced_choice"] = new_advanced
    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

    if lset_name== get_new_preset_msg(True) or lset_name== get_new_preset_msg(False) or lset_name=="":
        lset_name =  get_new_preset_msg(new_advanced)

    if only_allow_edit_in_advanced:
        return  gr.Row(visible=new_advanced), gr.Row(visible=new_advanced), gr.Button(visible=new_advanced), gr.Row(visible= not new_advanced), gr.Dropdown(choices=lset_choices, value= lset_name)
    else:
        return  gr.Row(visible=new_advanced), gr.Row(visible=True), gr.Button(visible=True), gr.Row(visible= False), gr.Dropdown(choices=lset_choices, value= lset_name)


def prepare_inputs_dict(target, inputs, model_type = None, model_filename = None ):
    state = inputs.pop("state")

    plugin_data = inputs.pop("plugin_data", {})
    if "lset_name" in inputs:
        inputs["lset_name"] = get_lset_name(state, inputs["lset_name"])    
    if "loras_choices" in inputs:
        loras_choices = inputs.pop("loras_choices")
        inputs.pop("model_filename", None)
    else:
        loras_choices = inputs["activated_loras"]
    if model_type == None: model_type = get_state_model_type(state)
    
    inputs["activated_loras"] = update_loras_url_cache(get_lora_dir(model_type), loras_choices)
    model_def = get_model_def(model_type)
    custom_settings = get_model_custom_settings(model_def)
    parsed_custom_settings, _ = collect_custom_settings_from_inputs(model_def, inputs, strict=False)
    inputs["custom_settings"] = parsed_custom_settings if len(custom_settings) > 0 else None
    clear_custom_setting_slots(inputs)
    inputs.pop("pace", None)
    inputs.pop("exaggeration", None)
    
    if target in ["state", "edit_state"]:
        return inputs
    
    if "lset_name" in inputs:
        inputs.pop("lset_name")
        
    unsaved_params = ATTACHMENT_KEYS
    for k in unsaved_params:
        inputs.pop(k)
    inputs["type"] = get_model_record(get_model_name(model_type))  
    inputs["settings_version"] = settings_version
    base_model_type = get_base_model_type(model_type)
    model_family = get_model_family(base_model_type)
    if model_type != base_model_type:
        inputs["base_model_type"] = base_model_type
    diffusion_forcing = base_model_type in ["sky_df_1.3B", "sky_df_14B"]
    vace =  test_vace_module(base_model_type) 
    t2v=   test_class_t2v(base_model_type) 
    ltxv = base_model_type in ["ltxv_13B"]
    if target == "settings":
        return inputs

    image_outputs = inputs.get("image_mode",0) > 0

    pop=[]    
    if len(custom_settings) == 0:
        pop += ["custom_settings"]
    if not model_def.get("audio_only", False):
        pop += ["temperature"]
    if model_def.get("duration_slider", None) is None:
        pop += ["duration_seconds"]
    if not model_def.get("pause_between_sentences", False):
        pop += ["pause_seconds"]
    if not model_def.get("top_p_slider", False):
        pop += ["top_p"]
    if not model_def.get("top_k_slider", False):
        pop += ["top_k"]
    if not model_def.get("temperature", True):
        pop += ["temperature"]
    if not model_def.get("inference_steps", True):
        pop += ["num_inference_steps"]

    if "force_fps" in inputs and len(inputs["force_fps"])== 0:
        pop += ["force_fps"]

    if model_def.get("sample_solvers", None) is None:
        pop += ["sample_solver"]
    
    if any_audio_track(base_model_type) or not get_mmaudio_settings(server_config)[0]:
        pop += ["MMAudio_setting", "MMAudio_prompt", "MMAudio_neg_prompt"]

    image_prompt_type = inputs.get("image_prompt_type", "") or ""
    video_prompt_type = inputs["video_prompt_type"]
    if "G" not in video_prompt_type:
        pop += ["denoising_strength"]

    if  "G" not in video_prompt_type and not model_def.get("mask_strength_always_enabled", False):
        pop += ["masking_strength"]
    
    if not input_video_strength_visible(model_def, image_prompt_type, video_prompt_type):
        pop += ["input_video_strength"]


    if not (server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0):
        pop += ["prompt_enhancer"]

    if model_def.get("model_modes", None) is None:
        pop += ["model_mode"]

    if model_def.get("guide_custom_choices", None ) is None and model_def.get("guide_preprocessing", None ) is None:
        pop += ["keep_frames_video_guide", "mask_expand"]

    if not "I" in video_prompt_type:
        pop += ["remove_background_images_ref"]
        if not model_def.get("any_image_refs_relative_size", False):
            pop += ["image_refs_relative_size"]

    if not "F" in video_prompt_type:
        pop += ["frames_positions"]
    
    if model_def.get("control_net_weight_name", None) is None:
        pop += ["control_net_weight", "control_net_weight2"] 

    if not len(model_def.get("control_net_weight_alt_name", "")) >0:
        pop += ["control_net_weight_alt"]

    if not model_def.get("self_refiner", False):
        pop += ["self_refiner_setting", "self_refiner_f_uncertainty", "self_refiner_plan", "self_refiner_certain_percentage"]
        # pop += ["self_refiner_setting", "self_refiner_plan"]

    if model_def.get("audio_scale_name", None) is None:
        pop += ["audio_scale"]

    if not model_def.get("motion_amplitude", False):
        pop += ["motion_amplitude"]

    if model_def.get("video_guide_outpainting", None) is None:
        pop += ["video_guide_outpainting", "video_guide_outpainting_ratio"] 

    if not (vace or t2v):
        pop += ["min_frames_if_references"]

    if not model_def.get("multiple_images_as_text_prompts", False):
        pop += ["multi_images_gen_type"]

    if not (diffusion_forcing or ltxv or vace):
        pop += ["keep_frames_video_source"]

    if not test_any_sliding_window( base_model_type):
        pop += ["sliding_window_size", "sliding_window_overlap", "sliding_window_overlap_noise", "sliding_window_discard_last_frames", "sliding_window_color_correction_strength"]

    if not model_def.get("audio_guidance", False):
        pop += ["audio_guidance_scale", "speakers_locations"]

    if not model_def.get("embedded_guidance", False):
        pop += ["embedded_guidance_scale"]

    if model_def.get("alt_guidance", None) is None:
        pop += ["alt_guidance_scale"]

    if model_def.get("alt_scale", None) is None:
        pop += ["alt_scale"]


    if not (model_def.get("tea_cache", False) or model_def.get("mag_cache", False)) :
        pop += ["skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc"]

    guidance_max_phases = model_def.get("guidance_max_phases", 0)
    guidance_phases = inputs.get("guidance_phases", 1)
    visible_phases = model_def.get("visible_phases", guidance_phases) 

    if guidance_max_phases < 1 and visible_phases < 1:
        pop += ["guidance_scale", "guidance_phases"]

    if guidance_max_phases < 2 or guidance_phases < 2 or visible_phases < 2:
        pop += ["guidance2_scale", "switch_threshold"]

    if guidance_max_phases < 3 or guidance_phases < 3 or visible_phases < 3:
        pop += ["guidance3_scale", "switch_threshold2", "model_switch_phase"]

    if not model_def.get("flow_shift", False):
        pop += ["flow_shift"]

    if model_def.get("no_negative_prompt", False) :
        pop += ["negative_prompt" ] 

    if not model_def.get("perturbation", False):
        pop += ["perturbation_switch", "perturbation_layers", "perturbation_start_perc", "perturbation_end_perc"]

    if not model_def.get("cfg_zero", False):
        pop += [ "cfg_zero_step"  ] 

    if not model_def.get("cfg_star", False):
        pop += ["cfg_star_switch" ] 

    if not model_def.get("adaptive_projected_guidance", False):
        pop += ["apg_switch"] 

    if not model_def.get("NAG", False):
        pop +=["NAG_scale", "NAG_tau", "NAG_alpha" ]

    for k in pop:
        if k in inputs: inputs.pop(k)

    if target == "metadata":
        inputs = {k: v for k,v in inputs.items() if v != None  }
        if hasattr(app, 'plugin_manager'):
            inputs = app.plugin_manager.run_data_hooks(
                'before_metadata_save',
                configs=inputs,
                plugin_data=plugin_data,
                model_type=model_type
            )

    return inputs

def get_function_arguments(func, locals):
    args_names = list(inspect.signature(func).parameters)
    kwargs = typing.OrderedDict()
    for k in args_names:
        kwargs[k] = locals[k]
    return kwargs


def init_generate(state, input_file_list, last_choice, audio_files_paths, audio_file_selected):
    gen = get_gen_info(state)
    current_gallery_source = gen.get("current_gallery_source", "video")
    selected_video_time = gen.get("selected_video_time", None)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) > 0: last_choice = max(last_choice, 0)
    gen["last_selected"] = (last_choice + 1) >= len(file_list)
    gen["selected"] = last_choice
    audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths), audio_files=True)
    if len(audio_file_list) > 0: audio_file_selected = max(audio_file_selected, 0)
    gen["audio_last_selected"] = (audio_file_selected + 1) >= len(audio_file_list)
    gen["audio_selected"] = audio_file_selected
    gen["current_gallery_source"] = current_gallery_source
    gen["selected_video_time"] = selected_video_time if current_gallery_source == "video" and 0 <= last_choice < len(file_list) and has_video_file_extension(file_list[last_choice]) else None
    return get_unique_id(), ""


def video_to_control_video(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info("Selected Video was copied to Control Video input")
    return file_list[choice]

def video_to_source_video(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info("Selected Video was copied to Source Video input")    
    return file_list[choice]

def image_to_ref_image_add(state, input_file_list, choice, target, target_name):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)    
    if model_def.get("one_image_ref_needed", False):
        gr.Info(f"Selected Image was set to {target_name}")
        target =[file_list[choice]]
    else:
        gr.Info(f"Selected Image was added to {target_name}")
        if target == None:
            target =[]
        target.append( file_list[choice])
    return target

def image_to_ref_image_set(state, input_file_list, choice, target, target_name):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info(f"Selected Image was copied to {target_name}")
    return file_list[choice]

def image_to_ref_image_guide(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update(), gr.update()
    ui_settings = get_current_model_settings(state)
    gr.Info(f"Selected Image was copied to Control Image")
    new_image = file_list[choice]
    if ui_settings["image_mode"]==2 or True:
        return new_image, new_image
    else:
        return new_image, None

def audio_to_source_set(state, input_file_list, choice, target_name):
    file_list, file_settings_list = get_file_list(state, unpack_audio_list(input_file_list), audio_files=True)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info(f"Selected Audio File was copied to {target_name}")
    return file_list[choice]


def apply_post_processing(state, input_file_list, choice, PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
        return gr.update(), gr.update(), gr.update()
    
    if not (file_list[choice].endswith(".mp4") or file_list[choice].endswith(".mkv")):
        gr.Info("Post processing is only available with Videos")
        return gr.update(), gr.update(), gr.update()
    overrides = {
        "temporal_upsampling":PP_temporal_upsampling,
        "spatial_upsampling":PP_spatial_upsampling,
        "film_grain_intensity": PP_film_grain_intensity, 
        "film_grain_saturation": PP_film_grain_saturation,
    }

    gen["edit_video_source"] = file_list[choice]
    gen["edit_overrides"] = overrides

    in_progress = gen.get("in_progress", False)
    return "edit_postprocessing", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()


def remux_audio(state, input_file_list, choice, PP_MMAudio_setting, PP_MMAudio_prompt, PP_MMAudio_neg_prompt, PP_MMAudio_seed, PP_repeat_generation, PP_custom_audio):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
        return gr.update(), gr.update(), gr.update()
    
    if not (file_list[choice].endswith(".mp4") or file_list[choice].endswith(".mkv")):
        gr.Info("Post processing is only available with Videos")
        return gr.update(), gr.update(), gr.update()
    overrides = {
        "MMAudio_setting" : PP_MMAudio_setting, 
        "MMAudio_prompt" : PP_MMAudio_prompt,
        "MMAudio_neg_prompt": PP_MMAudio_neg_prompt,
        "seed": PP_MMAudio_seed,
        "repeat_generation": PP_repeat_generation,
        "audio_source": PP_custom_audio,
    }

    gen["edit_video_source"] = file_list[choice]
    gen["edit_overrides"] = overrides

    in_progress = gen.get("in_progress", False)
    return "edit_remux", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()

def clear_deleted_files(state, audio_files):
    gen = get_gen_info(state)
    if audio_files:
        file_list_name = "audio_file_list"
        file_settings_name = "audio_file_settings_list"
    else:
        file_list_name = "file_list"
        file_settings_name = "file_settings_list"
    with lock:
        file_list = gen.get(file_list_name, [])
        file_settings_list = gen.get(file_settings_name, [])
        new_file_list = []
        new_file_settings_list = []
        for file_path, file_settings in zip(file_list, file_settings_list):
            if os.path.isfile(file_path):
                new_file_list.append(file_path)
                new_file_settings_list.append(file_settings)
        file_list[:]=new_file_list 
        file_settings_list[:]=new_file_settings_list 

def eject_video_from_gallery(state, input_file_list, choice):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    with lock:
        if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
            return gr.update(), gr.update(), gr.update()
        
        extend_list = file_list[choice + 1:] # inplace List change
        file_list[:] = file_list[:choice]
        file_list.extend(extend_list)

        extend_list = file_settings_list[choice + 1:]
        file_settings_list[:] = file_settings_list[:choice]
        file_settings_list.extend(extend_list)
        choice = min(choice, len(file_list))
    return gr.Gallery(value = file_list, selected_index= choice), gr.update() if len(file_list) >0 else get_default_video_info(), gr.Row(visible= len(file_list) > 0)

def eject_audio_from_gallery(state, input_file_list, choice):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, unpack_audio_list(input_file_list), audio_files=True)
    with lock:
        if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
            return [gr.update()] * 6
        
        extend_list = file_list[choice + 1:] # inplace List change
        file_list[:] = file_list[:choice]
        file_list.extend(extend_list)

        extend_list = file_settings_list[choice + 1:]
        file_settings_list[:] = file_settings_list[:choice]
        file_settings_list.extend(extend_list)
        choice = min(choice, len(file_list)-1)
    return *pack_audio_gallery_state(file_list, choice), gr.update() if len(file_list) >0 else get_default_video_info(), gr.Row(visible= len(file_list) > 0)


def add_videos_to_gallery(state, input_file_list, choice, audio_files_paths, audio_file_selected, files_to_load):
    gen = get_gen_info(state)
    if files_to_load == None:
        gr.Info("Please Select a File To Import")
        return [gr.update()]*9
    new_audio= False
    new_video= False

    file_list, file_settings_list = get_file_list(state, input_file_list)
    audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths), audio_files= True)
    audio_file = False
    with lock:
        valid_files_count = 0
        invalid_files_count = 0
        for file_path in files_to_load:
            file_settings, _, audio_file = get_settings_from_file(state, file_path, False, False, False)
            if file_settings == None:
                audio_file = False
                fps = 0
                try:
                    if has_audio_file_extension(file_path):
                        audio_file = True
                    elif has_video_file_extension(file_path):
                        fps, width, height, frames_count = get_video_info(file_path)
                    elif has_image_file_extension(file_path):
                        width, height = _open_image_input(file_path).size
                        fps = 1 
                except:
                    pass
                if fps == 0 and not audio_file:
                    invalid_files_count += 1 
                    continue
            if audio_file:
                new_audio= True
                audio_file_list.append(file_path)
                audio_file_settings_list.append(file_settings)
            else:
                new_video= True
                file_list.append(file_path)
                file_settings_list.append(file_settings)
            valid_files_count +=1

    if valid_files_count== 0 and invalid_files_count ==0:
        gr.Info("No Valid Media to Import")
        return [gr.update()] * 9
    else:
        txt = ""
        if valid_files_count > 0:
            txt = f"{valid_files_count} files were added. " if valid_files_count > 1 else  f"One file was added."
        if invalid_files_count > 0:
            txt += f"Unable to add {invalid_files_count} files which were invalid. " if invalid_files_count > 1 else  f"Unable to add one file which was invalid."
        gr.Info(txt)
    if new_video:
        choice = len(file_list) - 1
    else:
        choice = min(len(file_list) - 1, choice)
    gen["selected"] = choice
    if new_audio:
        audio_file_selected = len(audio_file_list) - 1
    else:
        audio_file_selected = min(len(file_list) - 1, audio_file_selected)
    gen["audio_selected"] = audio_file_selected

    gallery_tabs = gr.Tabs(selected= "audio" if audio_file else "video_images")

    # return gallery_tabs, gr.Gallery(value = file_list) if audio_file else gr.Gallery(value = file_list, selected_index=choice, preview= True) , *pack_audio_gallery_state(audio_file_list, audio_file_selected), gr.Files(value=[]),  gr.Tabs(selected="video_info"), "audio" if audio_file else "video"
    return gallery_tabs, 1 if audio_file else 0, gr.Gallery(value = file_list, selected_index=choice, preview= True) , *pack_audio_gallery_state(audio_file_list, audio_file_selected), gr.Files(value=[]),  gr.Tabs(selected="video_info"), "audio" if audio_file else "video"

def get_model_settings(state, model_type):
    all_settings = state.get("all_settings", None)    
    return None if all_settings == None else all_settings.get(model_type, None)

def set_model_settings(state, model_type, settings):
    all_settings = state.get("all_settings", None)    
    if all_settings == None:
        all_settings = {}
        state["all_settings"] = all_settings
    all_settings[model_type] = settings
    
def collect_current_model_settings(state):
    model_type = get_state_model_type(state)
    settings = get_model_settings(state, model_type)
    settings["state"] = state
    settings = prepare_inputs_dict("metadata", settings)
    settings["model_filename"] = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
    settings["model_type"] = model_type 
    return settings 

def export_settings(state):
    model_type = get_state_model_type(state)
    text = json.dumps(collect_current_model_settings(state), indent=4)
    text_base64 = base64.b64encode(text.encode('utf8')).decode('utf-8')
    return text_base64, sanitize_file_name(model_type + "_" + datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss") + ".json")


def extract_and_apply_source_images(file_path, current_settings):
    from shared.utils.video_metadata import extract_source_images
    if not os.path.isfile(file_path): return 0
    extracted_files = extract_source_images(file_path)            
    if not extracted_files: return 0
    applied_count = 0
    for name in image_names_list:
        if name in extracted_files:
            img = extracted_files[name]
            img = img if isinstance(img,list) else [img]
            applied_count += len(img)
            current_settings[name] = img        
    return applied_count


def use_video_settings(state, input_file_list, choice, source):
    gen = get_gen_info(state)
    any_audio = source == "audio"
    if any_audio:
        input_file_list = unpack_audio_list(input_file_list)
    file_list, file_settings_list = get_file_list(state, input_file_list, audio_files=any_audio)
    if choice != None and len(file_list)>0:
        choice= max(0, choice)
        configs = file_settings_list[choice]
        file_name= file_list[choice]
        if configs == None:
            gr.Info("No Settings to Extract")
        elif is_deepy_display_metadata(configs):
            gr.Info("Deepy helper metadata is display-only and cannot be loaded as WanGP settings")
        else:
            current_model_type = get_state_model_type(state)
            model_type = configs["model_type"] 
            models_compatible = are_model_types_compatible(model_type,current_model_type) 
            if models_compatible:
                model_type = current_model_type
            defaults = get_model_settings(state, model_type) 
            defaults = get_default_settings(model_type) if defaults == None else defaults
            defaults.update(configs)
            defaults["model_type"] = model_type
            prompt = configs.get("prompt", "")
                        
            if has_audio_file_extension(file_name):
                set_model_settings(state, model_type, defaults)
                gr.Info(f"Settings Loaded from Audio File with prompt '{prompt[:100]}'")
            elif has_image_file_extension(file_name):
                set_model_settings(state, model_type, defaults)
                gr.Info(f"Settings Loaded from Image with prompt '{prompt[:100]}'")
            elif has_video_file_extension(file_name):
                extracted_images = extract_and_apply_source_images(file_name, defaults)
                set_model_settings(state, model_type, defaults)
                info_msg = f"Settings Loaded from Video with prompt '{prompt[:100]}'"
                if extracted_images:
                    info_msg += f" + {extracted_images} source {'image' if extracted_images == 1 else 'images'} extracted"
                gr.Info(info_msg)
            
            if models_compatible:
                return gr.update(), gr.update(), gr.update(), str(time.time())
            else:
                return *generate_dropdown_model_list(model_type), gr.update()
    else:
        gr.Info(f"Please Select a File")

    return gr.update(), gr.update(), gr.update(), gr.update()
loras_url_cache = None
def update_loras_url_cache(lora_dir, loras_selected):
    if loras_selected is None:
        return None
    global loras_url_cache
    loras_cache_file = "loras_url_cache.json"
    if loras_url_cache is None:
        if os.path.isfile(loras_cache_file):
            try:
                with open(loras_cache_file, 'r', encoding='utf-8') as f:
                    loras_url_cache = json.load(f)
            except:
                loras_url_cache = {}
        else:
            loras_url_cache = {}
    new_loras_selected = []
    update = False
    for lora in loras_selected:
        base_name = os.path.basename(lora)
        local_name = os.path.join(lora_dir, base_name)
        url = loras_url_cache.get(local_name, base_name)
        if (lora.startswith("http:") or lora.startswith("https:")) and url != lora:
            loras_url_cache[local_name]=lora
            update = True
        new_loras_selected.append(url)

    if update:
        with open(loras_cache_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(loras_url_cache, indent=4))

    return new_loras_selected

def _ensure_loras_url_cache():
    update_loras_url_cache("", [])

def get_settings_from_file(state, file_path, allow_json, merge_with_defaults, switch_type_if_compatible, min_settings_version = 0, merge_loras = None):    
    configs = None
    any_image_or_video = False
    any_audio = False
    if file_path.endswith(".json") and allow_json:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        except:
            pass
    elif file_path.endswith(".mp4") or file_path.endswith(".mkv"):
        from shared.utils.video_metadata import read_metadata_from_video
        try:
            configs = read_metadata_from_video(file_path)
            if configs:
                any_image_or_video = True
        except:
            pass
    elif has_image_file_extension(file_path):
        try:
            configs = read_image_metadata(file_path)
            any_image_or_video = True
        except:
            pass
    elif has_audio_file_extension(file_path):
        try:
            configs = read_audio_metadata(file_path)
            any_audio = True
        except:
            pass
    if configs is None: return None, False, False
    try:
        if isinstance(configs, dict):
            if (not merge_with_defaults) and not "WanGP" in configs.get("type", ""): configs = None 
        else:
            configs = None
    except:
        configs = None
    if configs is None: return None, False, False
    if is_deepy_display_metadata(configs):
        return configs, any_image_or_video, any_audio
        

    current_model_type = get_state_model_type(state)
    
    model_type = configs.get("model_type", None)
    if get_base_model_type(model_type) == None:
        model_type = configs.get("base_model_type", None)
  
    if model_type == None:
        model_filename = configs.get("model_filename", "")
        model_type = get_model_type(model_filename)
        if model_type == None:
            model_type = current_model_type
    elif not model_type in model_types:
        model_type = current_model_type
    if switch_type_if_compatible and are_model_types_compatible(model_type,current_model_type):
        model_type = current_model_type
    old_loras_selected = old_loras_multipliers = None
    if merge_with_defaults:
        defaults = get_model_settings(state, model_type) 
        defaults = get_default_settings(model_type) if defaults == None else defaults
        if merge_loras is not None and model_type == current_model_type:
            old_loras_selected, old_loras_multipliers  = defaults.get("activated_loras", []), defaults.get("loras_multipliers", ""),
        defaults.update(configs)
        configs = defaults

    loras_selected =configs.get("activated_loras", [])
    loras_multipliers = configs.get("loras_multipliers", "")
    if loras_selected is not None and len(loras_selected) > 0:
        loras_selected = update_loras_url_cache(get_lora_dir(model_type), loras_selected)
    if old_loras_selected is not None:
        if len(old_loras_selected) == 0 and "|" in loras_multipliers:
            pass
        else:
            old_loras_selected = update_loras_url_cache(get_lora_dir(model_type), old_loras_selected)
            loras_selected, loras_multipliers = merge_loras_settings(old_loras_selected, old_loras_multipliers, loras_selected, loras_multipliers, merge_loras )

    configs["activated_loras"]= loras_selected or []
    configs["loras_multipliers"] = loras_multipliers       
    fix_settings(model_type, configs, min_settings_version)
    configs["model_type"] = model_type

    return configs, any_image_or_video, any_audio

def record_image_mode_tab(state, evt:gr.SelectData):
    state["image_mode_tab"] = evt.index

def switch_image_mode(state):
    image_mode = state.get("image_mode_tab", 0)
    model_type =get_state_model_type(state)
    ui_defaults = get_model_settings(state, model_type)        

    ui_defaults["image_mode"] = image_mode
    video_prompt_type = ui_defaults.get("video_prompt_type", "") 
    model_def = get_model_def( model_type)
    inpaint_support = model_def.get("inpaint_support", False)

    if inpaint_support:
        model_type = get_state_model_type(state)
        inpaint_cache= state.get("inpaint_cache", None)
        if inpaint_cache is None:
            state["inpaint_cache"] = inpaint_cache = {}
        model_cache = inpaint_cache.get(model_type, None)
        if model_cache is None:
            inpaint_cache[model_type] = model_cache ={}
        video_prompt_inpaint_mode = model_def.get("inpaint_video_prompt_type", "VAG")
        video_prompt_image_mode = model_def.get("image_video_prompt_type", "KI")
        old_video_prompt_type = video_prompt_type
        if image_mode == 1:
            model_cache[2] = video_prompt_type
            video_prompt_type = model_cache.get(1, None)
            if video_prompt_type is None:
                video_prompt_type = del_in_sequence(old_video_prompt_type, video_prompt_inpaint_mode + all_guide_processes)  
                video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_image_mode)
        elif image_mode == 2:
            model_cache[1] = video_prompt_type
            video_prompt_type = model_cache.get(2, None)
            if video_prompt_type is None:
                video_prompt_type = del_in_sequence(old_video_prompt_type, video_prompt_image_mode + all_guide_processes)
                video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_inpaint_mode)  
        ui_defaults["video_prompt_type"] = video_prompt_type 
        
    return  str(time.time())

def load_settings_from_file(state, file_path):
    gen = get_gen_info(state)

    if file_path==None:
        return gr.update(), gr.update(), gr.update(), gr.update(), None

    configs, any_video_or_image_file, any_audio = get_settings_from_file(state, file_path, True, True, True)
    if configs == None:
        gr.Info("File not supported")
        return gr.update(), gr.update(), gr.update(), gr.update(), None
    if is_deepy_display_metadata(configs):
        gr.Info("Deepy helper metadata is display-only and cannot be loaded as WanGP settings")
        return gr.update(), gr.update(), gr.update(), gr.update(), None

    current_model_type = get_state_model_type(state)
    model_type = configs["model_type"]
    prompt = configs.get("prompt", "")
    is_image = configs.get("is_image", False)

    # Extract and apply embedded source images from video files
    extracted_images = 0
    if file_path.endswith('.mkv') or file_path.endswith('.mp4'):
        extracted_images = extract_and_apply_source_images(file_path, configs)
    if any_audio:
        gr.Info(f"Settings Loaded from Audio file with prompt '{prompt[:100]}'")
    elif any_video_or_image_file:    
        info_msg = f"Settings Loaded from {'Image' if is_image else 'Video'} generated with prompt '{prompt[:100]}'"
        if extracted_images > 0:
            info_msg += f" + {extracted_images} source image(s) extracted and applied"
        gr.Info(info_msg)
    else:
        gr.Info(f"Settings Loaded from Settings file with prompt '{prompt[:100]}'")

    if model_type == current_model_type:
        set_model_settings(state, current_model_type, configs)        
        return gr.update(), gr.update(), gr.update(), str(time.time()), None
    else:
        set_model_settings(state, model_type, configs)        
        return *generate_dropdown_model_list(model_type), gr.update(), None

def goto_model_type(state, model_type):
    gen = get_gen_info(state)
    return *generate_dropdown_model_list(model_type), gr.update()

def refresh_model_dropdowns(state):
    return *generate_dropdown_model_list(get_state_model_type(state)), gr.update()

def reset_settings(state):
    model_type = get_state_model_type(state)
    ui_defaults = get_default_settings(model_type)
    set_model_settings(state, model_type, ui_defaults)
    gr.Info(f"Default Settings have been Restored")
    return str(time.time())

def save_inputs(
            target,
            image_mask_guide,
            lset_name,
            client_id,
            image_mode,
            prompt,
            alt_prompt,
            negative_prompt,
            resolution,
            video_length,
            duration_seconds,
            pause_seconds,
            batch_size,
            seed,
            force_fps,
            num_inference_steps,
            guidance_scale,
            guidance2_scale,
            guidance3_scale,
            switch_threshold,
            switch_threshold2,
            guidance_phases,
            model_switch_phase,
            alt_guidance_scale,
            alt_scale,
            audio_guidance_scale,
            audio_scale,
            flow_shift,
            sample_solver,
            embedded_guidance_scale,
            repeat_generation,
            multi_prompts_gen_type,
            multi_images_gen_type,
            skip_steps_cache_type,
            skip_steps_multiplier,
            skip_steps_start_step_perc,    
            loras_choices,
            loras_multipliers,
            image_prompt_type,
            image_start,
            image_end,
            model_mode,
            video_source,
            keep_frames_video_source,
              input_video_strength,
              video_guide_outpainting,
              video_guide_outpainting_ratio,
              video_prompt_type,
            image_refs,
            frames_positions,
            video_guide,
            image_guide,
            keep_frames_video_guide,
            denoising_strength,
            masking_strength,
            video_mask,
            image_mask,
            control_net_weight,
            control_net_weight2,
            control_net_weight_alt,
            motion_amplitude,
            mask_expand,
            audio_guide,
            audio_guide2,
            custom_guide,
            audio_source,            
            audio_prompt_type,
            speakers_locations,
            sliding_window_size,
            sliding_window_overlap,
            sliding_window_color_correction_strength,
            sliding_window_overlap_noise,
            sliding_window_discard_last_frames,
            image_refs_relative_size,
            remove_background_images_ref,
            temporal_upsampling,
            spatial_upsampling,
            film_grain_intensity,
            film_grain_saturation,
            MMAudio_setting,
            MMAudio_prompt,
            MMAudio_neg_prompt,            
            RIFLEx_setting,
            NAG_scale,
            NAG_tau,
            NAG_alpha,
            perturbation_switch,
            perturbation_layers,
            perturbation_start_perc,
            perturbation_end_perc,
            apg_switch,
            cfg_star_switch,
            cfg_zero_step,
            prompt_enhancer,
            min_frames_if_references,
            override_profile,
            override_attention,            
            temperature,
            custom_setting_1,
            custom_setting_2,
            custom_setting_3,
            custom_setting_4,
            custom_setting_5,
            top_p,
            top_k,
            self_refiner_setting,
            self_refiner_plan,            
            self_refiner_f_uncertainty,
            self_refiner_certain_percentage,
            output_filename,
            mode,
            state,
            plugin_data,
):

    if state.pop("ignore_save_form", False):
        return

    model_type = get_state_model_type(state)
    if image_mask_guide is not None and image_mode >= 1 and video_prompt_type is not None and "A" in video_prompt_type and not "U" in video_prompt_type:
    # if image_mask_guide is not None and image_mode == 2:
        if "background" in image_mask_guide: 
            image_guide = image_mask_guide["background"]
        if "layers" in image_mask_guide and len(image_mask_guide["layers"])>0: 
            image_mask = image_mask_guide["layers"][0] 
        image_mask_guide = None
    inputs = get_function_arguments(save_inputs, locals())
    inputs.pop("target")
    inputs.pop("image_mask_guide")
    cleaned_inputs = prepare_inputs_dict(target, inputs)
    if target == "settings":
        defaults_filename = get_settings_file_name(model_type)

        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(cleaned_inputs, f, indent=4)

        gr.Info("New Default Settings saved")
    elif target == "state":
        set_model_settings(state, model_type, cleaned_inputs)        
    elif target == "edit_state":
        state["edit_state"] = cleaned_inputs        


def handle_queue_action(state, action_string):
    if not action_string:
        return gr.HTML(), gr.Tabs(), gr.update()
        
    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    
    try:
        parts = action_string.split('_')
        action = parts[0]
        params = parts[1:]
    except (IndexError, ValueError):
        return update_queue_data(queue), gr.Tabs(), gr.update()

    if action == "edit" or action == "silent_edit":
        task_id = int(params[0])
        
        with lock:
            task_index = next((i for i, task in enumerate(queue) if task['id'] == task_id), -1)
        
        if task_index != -1:
            state["editing_task_id"] = task_id
            task_data = queue[task_index]

            if task_index == 1:
                gen["queue_paused_for_edit"] = True
                gr.Info("Queue processing will pause after the current generation, as you are editing the next item to generate.")

            if action == "edit":
                gr.Info(f"Loading task ID {task_id} ('{task_data['prompt'][:50]}...') for editing.")
            return update_queue_data(queue), gr.Tabs(selected="edit"), gr.update(visible=True), get_unique_id()
        else:
            gr.Warning("Task ID not found. It may have already been processed.")
            return update_queue_data(queue), gr.Tabs(), gr.update(), gr.update()
            
    elif action == "move" and len(params) == 3 and params[1] == "to":
        old_index_str, new_index_str = params[0], params[2]
        return move_task(queue, old_index_str, new_index_str), gr.Tabs(), gr.update(), gr.update()
        
    elif action == "remove":
        task_id_to_remove = int(params[0])
        new_queue_data = remove_task(queue, task_id_to_remove)
        gen["prompts_max"] = gen.get("prompts_max", 0) - 1
        update_status(state)
        return new_queue_data, gr.Tabs(), gr.update(), gr.update()

    return update_queue_data(queue), gr.Tabs(), gr.update(), gr.update()

def change_model(state, model_choice):
    if model_choice == None:
        return
    model_filename = get_model_filename(model_choice, transformer_quantization, transformer_dtype_policy)
    last_model_per_family = state["last_model_per_family"] 
    last_model_per_family[get_model_family(model_choice, for_ui= True)] = model_choice
    server_config["last_model_per_family"] = last_model_per_family

    last_model_per_type = state["last_model_per_type"] 
    last_model_per_type[get_base_model_type(model_choice)] = model_choice
    server_config["last_model_per_type"] = last_model_per_type

    server_config["last_model_type"] = model_choice

    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

    state["model_type"] = model_choice
    if hasattr(app, "plugin_manager"):
        app.plugin_manager.notify_model_change(state, model_choice)
    description, header = generate_header(model_choice, compile=compile, attention_mode=attention_mode)
    
    return description, header

def get_current_model_settings(state):
    model_type = get_state_model_type(state)
    ui_defaults = get_model_settings(state, model_type)
    if ui_defaults == None:
        ui_defaults = get_default_settings(model_type)
        set_model_settings(state, model_type, ui_defaults)
    return ui_defaults 

def fill_inputs(state):
    ui_defaults = get_current_model_settings(state)
 
    return generate_video_tab(update_form = True, state_dict = state, ui_defaults = ui_defaults)

def preload_model_when_switching(state):
    global reload_needed, wan_model, offloadobj
    if "S" in preload_model_policy:
        model_type = get_state_model_type(state) 
        if  model_type !=  transformer_type:
            release_model()            
            model_filename = get_model_name(model_type)
            yield f"Loading model {model_filename}..."
            wan_model, offloadobj = load_models(
                model_type,
                output_type=get_output_type_for_model(model_type, 0),
            )
            yield f"Model loaded"
            reload_needed=  False 
        return   
    return gr.Text()

def unload_model_if_needed(state):
    global wan_model
    if "U" in preload_model_policy:
        if wan_model != None:
            release_model()

def all_letters(source_str, letters):
    for letter in letters:
        if not letter in source_str:
            return False
    return True    

def any_letters(source_str, letters):
    for letter in letters:
        if letter in source_str:
            return True
    return False

def filter_letters(source_str, letters, default= ""):
    ret = ""
    for letter in letters:
        if letter in source_str:
            ret += letter
    if len(ret) == 0:
        return default
    return ret    

def add_to_sequence(source_str, letters):
    ret = source_str
    for letter in letters:
        if not letter in source_str:
            ret += letter
    return ret    

def del_in_sequence(source_str, letters):
    ret = source_str
    for letter in letters:
        if letter in source_str:
            ret = ret.replace(letter, "")
    return ret    

def get_prompt_enhancer_letters_filter(prompt_enhancer_def, prompt_enhancer_choices):
    if isinstance(prompt_enhancer_def, dict):
        letters_filter = prompt_enhancer_def.get("letters_filter", "")
        if len(letters_filter): return letters_filter
    letters_filter = ""
    for _label, value in prompt_enhancer_choices:
        letters_filter = add_to_sequence(letters_filter, str(value or ""))
    return letters_filter

def build_prompt_enhancer_value(prompt_enhancer_mode, prompt_enhancer_think):
    if prompt_enhancer_think and len(prompt_enhancer_mode):
        prompt_enhancer_mode = add_to_sequence(prompt_enhancer_mode, "K")
    return prompt_enhancer_mode

def refresh_audio_prompt_type_remux(state, audio_prompt_type, remux):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "R")
    audio_prompt_type = add_to_sequence(audio_prompt_type, remux)
    return audio_prompt_type

def refresh_remove_background_sound(state, audio_prompt_type, remove_background_sound):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "V")
    if remove_background_sound:
        audio_prompt_type = add_to_sequence(audio_prompt_type, "V")
    return audio_prompt_type

def refresh_continue_beyond_audio_end(state, audio_prompt_type, continue_beyond_audio_end):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "L")
    if continue_beyond_audio_end:
        audio_prompt_type = add_to_sequence(audio_prompt_type, "L")
    return audio_prompt_type

def refresh_normalize_audio_volumes(state, audio_prompt_type, normalize_audio_volumes):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "N")
    if normalize_audio_volumes:
        audio_prompt_type = add_to_sequence(audio_prompt_type, "N")
    return audio_prompt_type

def refresh_audio_prompt_type_sources(state, audio_prompt_type, audio_prompt_type_sources):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    letters_filter=  "XCPABOKF"
    audio_prompt_type_sources_def = model_def.get("audio_prompt_type_sources", None)
    if audio_prompt_type_sources_def is not None:
        letters_filter = audio_prompt_type_sources_def.get("letters_filter", letters_filter)
    audio_prompt_type = del_in_sequence(audio_prompt_type, letters_filter)
    audio_prompt_type = add_to_sequence(audio_prompt_type, audio_prompt_type_sources)
    audio_only = model_def.get("audio_only", False) if model_def is not None else False
    speakers_visible = ("B" in audio_prompt_type or "X" in audio_prompt_type) and not audio_only
    remove_background_visible = any_letters(audio_prompt_type, "ABXK")
    normalize_audio_visible = all_letters(audio_prompt_type, "AB")
    audio_options_visible = remove_background_visible or normalize_audio_visible
    if not normalize_audio_visible:
        audio_prompt_type = del_in_sequence(audio_prompt_type, "N")
    return (
        audio_prompt_type,
        gr.update(visible="A" in audio_prompt_type),
        gr.update(visible="B" in audio_prompt_type),
        gr.update(visible=speakers_visible),
        gr.update(visible=remove_background_visible),
        gr.update(visible=normalize_audio_visible),
        gr.update(visible=audio_options_visible),
        gr.update(visible=any_letters(audio_prompt_type, "AB")),
    )

def refresh_image_prompt_type_radio(state, image_prompt_type, image_prompt_type_radio, video_prompt_type):
    image_prompt_type = del_in_sequence(image_prompt_type, "VLTS")
    image_prompt_type = add_to_sequence(image_prompt_type, image_prompt_type_radio)
    any_video_source = len(filter_letters(image_prompt_type, "VL"))>0
    model_def = get_model_def(get_state_model_type(state))
    end_visible = end_frames_option_visible(model_def, image_prompt_type)
    input_strength_visible = input_video_strength_visible(model_def, image_prompt_type, video_prompt_type)
    return image_prompt_type, gr.update(visible = "S" in image_prompt_type ), gr.update(visible = end_visible and ("E" in image_prompt_type) ), gr.update(visible = "V" in image_prompt_type) , gr.update(visible = input_strength_visible), gr.update(visible = any_video_source), gr.update(visible = end_visible)

def refresh_image_prompt_type_endcheckbox(state, image_prompt_type, image_prompt_type_radio, end_checkbox, video_prompt_type):
    image_prompt_type = del_in_sequence(image_prompt_type, "E")
    if end_checkbox: image_prompt_type += "E"
    image_prompt_type = add_to_sequence(image_prompt_type, image_prompt_type_radio)
    model_def = get_model_def(get_state_model_type(state))
    return image_prompt_type, gr.update(visible = end_frames_option_visible(model_def, image_prompt_type) and "E" in image_prompt_type), gr.update(visible = input_video_strength_visible(model_def, image_prompt_type, video_prompt_type))

def refresh_video_prompt_type_image_refs(state, video_prompt_type, video_prompt_type_image_refs, image_mode, image_prompt_type):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    image_ref_choices = model_def.get("image_ref_choices", None)
    if image_ref_choices is not None:
        video_prompt_type = del_in_sequence(video_prompt_type, image_ref_choices["letters_filter"])
    else:
        video_prompt_type = del_in_sequence(video_prompt_type, "KFI")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_image_refs)
    visible = "I" in video_prompt_type
    any_outpainting= image_mode in model_def.get("video_guide_outpainting", [])
    rm_bg_visible= visible and not model_def.get("no_background_removal", False) 
    img_rel_size_visible = visible and model_def.get("any_image_refs_relative_size", False)
    return video_prompt_type, gr.update(visible = visible),gr.update(visible = rm_bg_visible), gr.update(visible = img_rel_size_visible), gr.update(visible = visible and injected_frames_positions_visible(video_prompt_type_image_refs)), gr.update(visible= ("F" in video_prompt_type_image_refs or "K" in video_prompt_type_image_refs or "V" in video_prompt_type) and any_outpainting ), gr.update(visible = input_video_strength_visible(model_def, image_prompt_type, video_prompt_type))

def update_image_mask_guide(state, image_mask_guide):
    img = image_mask_guide["background"]
    if img.mode != 'RGBA':
        return image_mask_guide
    
    arr = np.array(img)
    rgb = Image.fromarray(arr[..., :3], 'RGB')
    alpha_gray = np.repeat(arr[..., 3:4], 3, axis=2)
    alpha_gray = 255 - alpha_gray
    alpha_rgb = Image.fromarray(alpha_gray, 'RGB')

    image_mask_guide = {"background" : rgb, "composite" : None, "layers": [rgb_bw_to_rgba_mask(alpha_rgb)]}

    return image_mask_guide

def switch_image_guide_editor(image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value ):
    if image_mode == 0: return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    mask_in_old = "A" in old_video_prompt_type and not "U" in old_video_prompt_type
    mask_in_new = "A" in video_prompt_type and not "U" in video_prompt_type
    image_mask_guide_value, image_mask_value, image_guide_value = {}, {}, {}
    visible = "V" in video_prompt_type
    if mask_in_old != mask_in_new:
        if mask_in_new:
            if old_image_mask_value is None:
                image_mask_guide_value["value"] = old_image_guide_value
            else:
                image_mask_guide_value["value"] = {"background" : old_image_guide_value, "composite" : None, "layers": [rgb_bw_to_rgba_mask(old_image_mask_value)]}
            image_guide_value["value"] = image_mask_value["value"] = None 
        else:
            if old_image_mask_guide_value is not None and "background" in old_image_mask_guide_value:
                image_guide_value["value"] = old_image_mask_guide_value["background"]
                if "layers" in old_image_mask_guide_value:
                    image_mask_value["value"] = old_image_mask_guide_value["layers"][0] if len(old_image_mask_guide_value["layers"]) >=1 else None
            image_mask_guide_value["value"] = {"background" : None, "composite" : None, "layers": []}
            
    image_mask_guide = gr.update(visible= visible and mask_in_new, **image_mask_guide_value)
    image_guide = gr.update(visible = visible and not mask_in_new, **image_guide_value)
    image_mask = gr.update(visible = False, **image_mask_value)
    return image_mask_guide, image_guide, image_mask

def refresh_video_prompt_type_video_mask(state, video_prompt_type, video_prompt_type_video_mask, image_mode, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value ):
    old_video_prompt_type = video_prompt_type
    video_prompt_type = del_in_sequence(video_prompt_type, "XYZWNA")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_mask)
    visible= "A" in video_prompt_type     
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    image_outputs =  image_mode > 0
    mask_strength_always_enabled = model_def.get("mask_strength_always_enabled", False)  
    image_mask_guide, image_guide, image_mask = switch_image_guide_editor(image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value )
    magic_image_btn, magic_video_btn = MagicMaskUI.button_updates(image_mode, video_prompt_type)
    return video_prompt_type, gr.update(visible= visible and not image_outputs), image_mask_guide, image_guide, image_mask, gr.update(visible= visible ) , gr.update(visible= visible and (mask_strength_always_enabled or "G" in video_prompt_type )  ), magic_image_btn, magic_video_btn

def refresh_video_prompt_type_alignment(state, video_prompt_type, video_prompt_type_video_guide):
    video_prompt_type = del_in_sequence(video_prompt_type, "T")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide)
    return video_prompt_type


def refresh_video_prompt_type_video_guide(state, filter_type, video_prompt_type, video_prompt_type_video_guide,  image_mode, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value, image_prompt_type ):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    old_video_prompt_type = video_prompt_type
    if filter_type == "alt":
        guide_custom_choices = model_def.get("guide_custom_choices",{})
        letter_filter = guide_custom_choices.get("letters_filter","")
    else:
        letter_filter = all_guide_processes
    video_prompt_type = del_in_sequence(video_prompt_type, letter_filter)
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide)
    visible = "V" in video_prompt_type
    any_outpainting= image_mode in model_def.get("video_guide_outpainting", [])
    mask_visible = visible and "A" in video_prompt_type and not "U" in video_prompt_type
    image_outputs =  image_mode > 0
    keep_frames_video_guide_visible = not image_outputs and visible and not model_def.get("keep_frames_video_guide_not_supported", False)
    image_mask_guide, image_guide, image_mask = switch_image_guide_editor(image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value )
    # mask_video_input_visible =  image_mode == 0 and mask_visible
    mask_preprocessing = model_def.get("mask_preprocessing", None)
    if mask_preprocessing  is not None:
        mask_selector_visible = mask_preprocessing.get("visible", True)
    else:
        mask_selector_visible = True
    ref_images_visible = "I" in video_prompt_type
    custom_options = custom_checkbox = False 
    custom_video_selection = model_def.get("custom_video_selection", None)
    if custom_video_selection is not None:
        custom_trigger =  custom_video_selection.get("trigger","")
        if len(custom_trigger) == 0 or custom_trigger in video_prompt_type:
            custom_options = True
            custom_checkbox = custom_video_selection.get("type","") == "checkbox"
    mask_strength_always_enabled = model_def.get("mask_strength_always_enabled", False)  
    magic_image_btn, magic_video_btn = MagicMaskUI.button_updates(image_mode, video_prompt_type)
    return video_prompt_type,  gr.update(visible = visible and not image_outputs), image_guide, gr.update(visible = keep_frames_video_guide_visible), gr.update(visible = visible and "G" in video_prompt_type),  gr.update(visible = mask_visible and( mask_strength_always_enabled or "G" in video_prompt_type)), gr.update(visible= (visible or injected_frames_positions_visible(video_prompt_type) or "K" in video_prompt_type) and any_outpainting), gr.update(visible= visible and mask_selector_visible and  not "U" in video_prompt_type ) ,  gr.update(visible= mask_visible and not image_outputs), image_mask, image_mask_guide, gr.update(visible= mask_visible),  gr.update(visible = ref_images_visible ), gr.update(visible = injected_frames_positions_visible(video_prompt_type)), gr.update(visible= custom_options and not custom_checkbox ), gr.update(visible= custom_options and custom_checkbox ), gr.update(visible = input_video_strength_visible(model_def, image_prompt_type, video_prompt_type)), magic_image_btn, magic_video_btn

def refresh_video_prompt_type_video_custom_dropbox(state, video_prompt_type, video_prompt_type_video_custom_dropbox):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    custom_video_selection = model_def.get("custom_video_selection", None)
    if custom_video_selection is None: return gr.update()
    letters_filter = custom_video_selection.get("letters_filter", "")
    video_prompt_type = del_in_sequence(video_prompt_type, letters_filter)
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_custom_dropbox)
    return video_prompt_type

def refresh_video_prompt_type_video_custom_checkbox(state, video_prompt_type, video_prompt_type_video_custom_checkbox):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    custom_video_selection = model_def.get("custom_video_selection", None)
    if custom_video_selection is None: return gr.update()
    letters_filter = custom_video_selection.get("letters_filter", "")
    video_prompt_type = del_in_sequence(video_prompt_type, letters_filter)
    if video_prompt_type_video_custom_checkbox:
        video_prompt_type = add_to_sequence(video_prompt_type, custom_video_selection["choices"][1][1])
    return video_prompt_type


def refresh_preview(state):
    gen = get_gen_info(state)
    preview_image = gen.get("preview", None)
    if preview_image is None:
        return ""
    
    preview_base64 = pil_to_base64_uri(preview_image, format="jpeg", quality=85)
    if preview_base64 is None:
        return ""

    html_content = f"""
    <div style="display: flex; justify-content: center; align-items: center; height: 200px; cursor: pointer;" onclick="showImageModal('preview_0')">
        <img src="{preview_base64}"
             style="max-height: 100%; max-width: 100%; object-fit: contain;" 
             alt="Preview">
    </div>
    """
    return html_content

def init_process_queue_if_any(state):                
    gen = get_gen_info(state)
    if bool(gen.get("queue",[])):
        state["validate_success"] = 1
        return gr.Button(visible=False), gr.Button(visible=True), gr.Column(visible=True)                   
    else:
        return gr.Button(visible=True), gr.Button(visible=False), gr.Column(visible=False)

def get_modal_image(image_base64, label):
    return f"""
    <div class="modal-flex-container" onclick="closeImageModal()">
        <div class="modal-content-wrapper" onclick="event.stopPropagation()">
            <div class="modal-close-btn" onclick="closeImageModal()">×</div>
            <div class="modal-label">{label}</div>
            <img src="{image_base64}" class="modal-image" alt="{label}">
        </div>
    </div>
    """

def show_modal_image(state, action_string):
    if not action_string:
        return gr.HTML(), gr.Column(visible=False)

    try:
        parts = action_string.split('_')
        gen = get_gen_info(state)
        queue = gen.get("queue", [])

        if parts[0] == 'preview':
            preview_image = gen.get("preview", None)
            if preview_image:
                preview_base64 = pil_to_base64_uri(preview_image)
                if preview_base64:
                    html_content = get_modal_image(preview_base64, "Preview")
                    return gr.HTML(value=html_content), gr.Column(visible=True)
            return gr.HTML(), gr.Column(visible=False)
        elif parts[0] == 'current':
            img_type = parts[1]
            img_index = int(parts[2])
            task_index = 0
        else:
            img_type = parts[0]
            row_index = int(parts[1])
            img_index = 0
            task_index = row_index + 1
            
    except (ValueError, IndexError):
        return gr.HTML(), gr.Column(visible=False)

    if task_index >= len(queue):
        return gr.HTML(), gr.Column(visible=False)

    task_item = queue[task_index]
    image_data = None
    label_data = None

    if img_type == 'start':
        image_data = task_item.get('start_image_data_base64')
        label_data = task_item.get('start_image_labels')
    elif img_type == 'end':
        image_data = task_item.get('end_image_data_base64')
        label_data = task_item.get('end_image_labels')

    if not image_data or not label_data or img_index >= len(image_data):
        return gr.HTML(), gr.Column(visible=False)

    html_content = get_modal_image(image_data[img_index], label_data[img_index])
    return gr.HTML(value=html_content), gr.Column(visible=True)

def get_prompt_labels(multi_prompts_gen_type, model_def, image_outputs = False, audio_only = False):
    prompt_description= model_def.get("prompt_description", None)
    if prompt_description is not None: return prompt_description, prompt_description
    medium = "Image" if image_outputs else ("Audio File" if audio_only else "Video")
    if multi_prompts_gen_type == "FG":
        new_line_text = "all the Lines are Parts of the Same Prompt"
    elif "W" in multi_prompts_gen_type:
        new_line_text = "each Paragraph of Prompt separated by an Empty Line will be used for a Sliding Window" if "P" in multi_prompts_gen_type else "each Line of Prompt will be used for a Sliding Window"
    elif "P" in multi_prompts_gen_type:
        new_line_text = f"each Paragraph of Prompt separated by an Empty Line will generate a new {medium}"
    else:
        new_line_text = f"each Line of Prompt will generate a new {medium}"
    prompt_class= model_def.get("prompt_class", "Prompts")

    return f"{prompt_class} ({new_line_text}, # lines = comments, ! lines = macros)", f"{prompt_class} ({new_line_text}, # lines = comments)"

def get_image_end_label(multi_prompts_gen_type):
    return "Images as ending points for each new Window of the same Video Generation" if "W" in multi_prompts_gen_type else "Images as ending points for new Videos in the Generation Queue"

def refresh_prompt_labels(state, multi_prompts_gen_type, image_mode):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    prompt_label, wizard_prompt_label =  get_prompt_labels(multi_prompts_gen_type, model_def, image_mode > 0, model_def.get("audio_only", False))
    return gr.update(label=prompt_label), gr.update(label = wizard_prompt_label), gr.update(label=get_image_end_label(multi_prompts_gen_type))

def update_video_guide_outpainting(video_guide_outpainting_value, value, pos):
    if len(video_guide_outpainting_value) <= 1:
        video_guide_outpainting_list = ["0"] * 4
    else:
        video_guide_outpainting_list = video_guide_outpainting_value.split(" ")
    video_guide_outpainting_list[pos] = str(value)
    if all(v=="0" for v in video_guide_outpainting_list):
        return ""
    return " ".join(video_guide_outpainting_list)

def refresh_video_guide_outpainting_row(video_guide_outpainting_checkbox, video_guide_outpainting):
    video_guide_outpainting = video_guide_outpainting[1:] if video_guide_outpainting_checkbox else "#" + video_guide_outpainting 
        
    return gr.update(visible=video_guide_outpainting_checkbox), gr.update(visible=video_guide_outpainting_checkbox), video_guide_outpainting

def refresh_video_guide_outpainting_labels(video_guide_outpainting_ratio):
    suffix = "%" if len((video_guide_outpainting_ratio or "").strip()) == 0 else "x"
    return gr.update(label=f"Top {suffix}"), gr.update(label=f"Bottom {suffix}"), gr.update(label=f"Left {suffix}"), gr.update(label=f"Right {suffix}")

custom_resolutions = None
def get_resolution_choices(current_resolution_choice, model_resolutions= None):
    global custom_resolutions


    resolution_file = "resolutions.json"
    if model_resolutions is not None:
        resolution_choices = model_resolutions
    elif custom_resolutions == None and os.path.isfile(resolution_file) :
        with open(resolution_file, 'r', encoding='utf-8') as f:
            try:
                resolution_choices = json.load(f)
            except Exception as e:
                print(f'Invalid "{resolution_file}" : {e}')
                resolution_choices = None
        if resolution_choices ==  None:
            pass 
        elif not isinstance(resolution_choices, list):
            print(f'"{resolution_file}" should be a list of 2 elements lists ["Label","WxH"]')
            resolution_choices == None
        else:
            for tup in resolution_choices:
                if not isinstance(tup, list) or len(tup) != 2 or not isinstance(tup[0], str) or not isinstance(tup[1], str):
                    print(f'"{resolution_file}" contains an invalid list of two elements: {tup}')
                    resolution_choices == None
                    break
                res_list = tup[1].split("x")
                if len(res_list) != 2 or not is_integer(res_list[0])  or not is_integer(res_list[1]):
                    print(f'"{resolution_file}" contains a resolution value that is not in the format "WxH": {tup[1]}')
                    resolution_choices == None
                    break
        custom_resolutions = resolution_choices
    else:
        resolution_choices = custom_resolutions
    if resolution_choices == None:
        resolution_choices=[]
        if server_config.get("enable_4k_resolutions", 0) == 1:
            resolution_choices=[
                # 4K
                ("3840x2176 (16:9)", "3840x2176"),
                ("2176x3840 (9:16)", "2176x3840"),
                ("3840x1664 (21:9)", "3840x1664"),
                ("1664x3840 (9:21)", "1664x3840"),
                # 1440p
                ("2560x1440 (16:9)", "2560x1440"),
                ("1440x2560 (9:16)", "1440x2560"),
                ("1920x1440 (4:3)", "1920x1440"),
                ("1440x1920 (3:4)", "1440x1920"),
                ("2160x1440 (3:2)", "2160x1440"),
                ("1440x2160 (2:3)", "1440x2160"),
                ("1440x1440 (1:1)", "1440x1440"),
                ("2688x1152 (21:9)", "2688x1152"),
                ("1152x2688 (9:21)", "1152x2688"),]
        resolution_choices += [# 1080p
            ("1920x1088 (16:9)", "1920x1088"),
            ("1088x1920 (9:16)", "1088x1920"),
            ("1920x832 (21:9)", "1920x832"),
            ("832x1920 (9:21)", "832x1920"),
            # 720p
            ("1024x1024 (1:1)", "1024x1024"),
            ("1280x720 (16:9)", "1280x720"),
            ("720x1280 (9:16)", "720x1280"), 
            ("1280x544 (21:9)", "1280x544"),
            ("544x1280 (9:21)", "544x1280"),
            ("1104x832 (4:3)", "1104x832"),
            ("832x1104 (3:4)", "832x1104"),
            ("960x960 (1:1)", "960x960"),
            # 540p
            ("960x544 (16:9)", "960x544"),
            ("544x960 (9:16)", "544x960"),
            # 480p
            ("832x624 (4:3)", "832x624"), 
            ("624x832 (3:4)", "624x832"),
            ("720x720 (1:1)", "720x720"),
            ("832x480 (16:9)", "832x480"),
            ("480x832 (9:16)", "480x832"),
            # 384p
            ("672x384 (16:9)", "672x384"),
            ("384x672 (9:16)", "384x672"),
            ("512x512 (1:1)", "512x512"),
            # 320p
            ("576x320 (16:9)", "576x320"),
            ("320x576 (9:16)", "320x576"),
            ("448x448 (1:1)", "448x448"),
            # 256p
            ("448x256 (7:4)", "448x256"),
            ("256x448 (4:7)", "256x448"),
            ("320x320 (1:1)", "320x320"),
        ]


    if current_resolution_choice is not None:
        found = False
        for label, res in resolution_choices:
            if current_resolution_choice == res:
                found = True
                break
        if not found:
            if model_resolutions is None:
                resolution_choices.append( (current_resolution_choice, current_resolution_choice ))
            else:
                if len(resolution_choices) > 0:
                    current_resolution_choice = resolution_choices[0][1]

    return resolution_choices, current_resolution_choice

group_thresholds = {
    "256p": 448 * 256,
    "320p": 448 * 448,
    "384p": 512 * 512,
    "480p": 832 * 624,     
    "540p": 960 * 544,   
    "720p": 1024 * 1024,  
    "1080p": 1920 * 1088,         
    "1440p": 2560 * 1440,
    "2160p": 3840 * 2176,
}
    
def categorize_resolution(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    pixel_count = width * height
    
    for group in group_thresholds.keys():
        if pixel_count <= group_thresholds[group]:
            return group
    return next(reversed(group_thresholds))

def group_resolutions(model_def, resolutions, selected_resolution):

    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None:
        selected_group ="Locked"
        available_groups = [selected_group ]
        selected_group_resolutions = model_resolutions
    else:
        grouped_resolutions = {}
        for resolution in resolutions:
            group = categorize_resolution(resolution[1])
            if group not in grouped_resolutions:
                grouped_resolutions[group] = []
            grouped_resolutions[group].append(resolution)
        
        available_groups = [group for group in group_thresholds if group in grouped_resolutions]
    
        selected_group = categorize_resolution(selected_resolution)
        selected_group_resolutions = grouped_resolutions.get(selected_group, [])
        available_groups.reverse()
    return available_groups, selected_group_resolutions, selected_group

def change_resolution_group(state, selected_group):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    model_resolutions = model_def.get("resolutions", None)
    resolution_choices, _ = get_resolution_choices(None, model_resolutions)   
    if model_resolutions is None:
        group_resolution_choices = [ resolution for resolution in resolution_choices if categorize_resolution(resolution[1]) == selected_group ]
    else:
        last_resolution = group_resolution_choices[0][1]
        return gr.update(choices= group_resolution_choices, value= last_resolution) 

    last_resolution_per_group = state["last_resolution_per_group"]
    last_resolution = last_resolution_per_group.get(selected_group, "")
    if len(last_resolution) == 0 or not any( [last_resolution == resolution[1] for resolution in group_resolution_choices]):
        last_resolution = group_resolution_choices[0][1]
    return gr.update(choices= group_resolution_choices, value= last_resolution ) 
    


def record_last_resolution(state, resolution):

    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    model_resolutions = model_def.get("resolutions", None)
    if model_resolutions is not None: return
    server_config["last_resolution_choice"] = resolution
    selected_group = categorize_resolution(resolution)
    last_resolution_per_group = state["last_resolution_per_group"]
    last_resolution_per_group[selected_group ] = resolution
    server_config["last_resolution_per_group"] = last_resolution_per_group
    with open(server_config_filename, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(server_config, indent=4))

def get_max_frames(nb):
    multiplier = max(1, int(server_config.get("max_frames_multiplier", 1)))
    return (nb - 1) * multiplier + 1


def get_max_duration(seconds):
    multiplier = max(1, int(server_config.get("max_frames_multiplier", 1)))
    return seconds * multiplier


def change_guidance_phases(state, guidance_phases):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    visible_phases = model_def.get("visible_phases", guidance_phases) 
    multiple_submodels = model_def.get("multiple_submodels", False)
    label ="Phase 1-2" if guidance_phases ==3 else ( "Model / Guidance Switch Threshold" if multiple_submodels  else "Guidance Switch Threshold" )
    return gr.update(visible= guidance_phases >=3 and visible_phases >=3 and multiple_submodels) , gr.update(visible= guidance_phases >=2 and visible_phases >=2), gr.update(visible= guidance_phases >=2 and visible_phases >=2, label = label), gr.update(visible= guidance_phases >=3 and visible_phases >=3), gr.update(visible= guidance_phases >=2 and visible_phases >=2), gr.update(visible= guidance_phases >=3 and visible_phases >=3)


memory_profile_choices= [   ("Profile 1, HighRAM_HighVRAM: at least 64 GB of RAM and 24 GB of VRAM, the fastest for short videos with a RTX 3090 / RTX 4090", 1),
                            ("Profile 2, HighRAM_LowVRAM: at least 64 GB of RAM and 12 GB of VRAM, the most versatile profile with high RAM, better suited for RTX 3070/3080/4070/4080 or for RTX 3090 / RTX 4090 with large pictures batches or long videos", 2),
                            ("Profile 3, LowRAM_HighVRAM: at least 32 GB of RAM and 24 GB of VRAM, adapted for RTX 3090 / RTX 4090 with limited RAM for good speed short video",3),
                            ("Profile 3+, VeryLowRAM_HighVRAM: at least 32 GB of RAM and 24 GB of VRAM, variant of Profile 3 that won't used Reserved Memory to reduce RAM usage",3.5),
                            ("Profile 4, LowRAM_LowVRAM (Recommended): at least 32 GB of RAM and 12 GB of VRAM, if you have little VRAM or want to generate longer videos",4),
                            ("Profile 4+, LowRAM_LowVRAM+: at least 32 GB of RAM and 12 GB of VRAM, variant of Profile 4, slightly slower but needs less VRAM",4.5),
                            ("Profile 5, VerylowRAM_LowVRAM (Fail safe): at least 24 GB of RAM and 10 GB of VRAM, if you don't have much it won't be fast but maybe it will work",5)]

def check_attn(mode):
    if mode not in attention_modes_installed: return " (NOT INSTALLED)"
    if mode not in attention_modes_supported: return " (NOT SUPPORTED)"
    return ""

attention_modes_choices= [
    ("Auto: Best available (sage2 > sage > sdpa)", "auto"),
    ("sdpa: Default, always available", "sdpa"),
    (f'flash{check_attn("flash")}: High quality, requires manual install', "flash"),
    (f'xformers{check_attn("xformers")}: Good quality, less VRAM, requires manual install', "xformers"),
    (f'sage{check_attn("sage")}: ~30% faster, requires manual install', "sage"),
    (f'sage2/sage2++{check_attn("sage2")}: ~40% faster, requires manual install', "sage2"),
] + ([(f'radial{check_attn("radial")}: Experimental, may be faster, requires manual install', "radial")] if args.betatest else []) + [
    (f'sage3{check_attn("sage3")}: >50% faster, may have quality trade-offs, requires manual install', "sage3"),
]

def detect_auto_save_form(state, evt:gr.SelectData):
    last_tab_id = state.get("last_tab_id", 0)
    state["last_tab_id"] = new_tab_id = evt.index
    if new_tab_id > 0 and last_tab_id == 0:
        return get_unique_id()
    else:
        return gr.update()

def compute_video_length_label(fps, current_video_length, video_length_locked = None):
    if fps is None:
        ret = f"Number of frames"
    else:
        ret = f"Number of frames ({fps} frames = 1s), current duration: {(current_video_length / fps):.1f}s"
    if video_length_locked is not None:
        ret += ", locked"
    return ret
def refresh_video_length_label(state, current_video_length, force_fps, video_guide, video_source):
    base_model_type = get_base_model_type(get_state_model_type(state))
    computed_fps = get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    return gr.update(label= compute_video_length_label(computed_fps, current_video_length))

def update_value(prompt_type_value, sub_value, letter_filter):                        
    return del_in_sequence(prompt_type_value, letter_filter) + filter_letters(sub_value, letter_filter)

def get_default_value(choices, current_value, default_value = None):
    for label, value in choices:
        if value == current_value:
            return current_value
    return default_value

def download_lora(state, lora_url, progress=gr.Progress(track_tqdm=True),):
    if lora_url is None or not lora_url.startswith("http"):
        gr.Info("Please provide a URL for a Lora to Download")
        return gr.update()
    model_type = get_state_model_type(state)
    lora_short_name = os.path.basename(lora_url)
    lora_dir = get_lora_dir(model_type)
    local_path = os.path.join(lora_dir, lora_short_name)
    try:
        download_file(lora_url, local_path)
    except Exception as e:
        gr.Info(f"Error downloading Lora {lora_short_name}: {e}")
        return gr.update()
    update_loras_url_cache(lora_dir, [lora_url])
    gr.Info(f"Lora {lora_short_name} has been succesfully downloaded")
    return ""
    

def set_gallery_tab(state, evt:gr.SelectData):                
    gen = get_gen_info(state)
    gen["current_gallery_source"] = "video" if evt.index == 0 else "audio"
    if evt.index != 0:
        gen["selected_video_time"] = None
    return evt.index, gen["current_gallery_source"]


def get_processed_queue(gen):
    with lock:
        file_list = gen.get("file_list", [])
        file_settings_list = gen.get("file_settings_list", [])
        audio_file_list = gen.get("audio_file_list", [])
        audio_file_settings_list = gen.get("audio_file_settings_list", [])
    return file_list, file_settings_list, audio_file_list, audio_file_settings_list


_deepy = deepy_controller.create_controller(
    get_server_config=lambda: server_config,
    get_server_config_filename=lambda: server_config_filename,
    get_verbose_level=lambda: verbose_level,
    resolve_prompt_enhancer_settings=resolve_prompt_enhancer_settings,
    get_state_model_type=get_state_model_type,
    get_model_def=get_model_def,
    ensure_prompt_enhancer_loaded=ensure_prompt_enhancer_loaded,
    unload_prompt_enhancer_runtime=unload_prompt_enhancer_runtime,
    get_image_caption_model=lambda: prompt_enhancer_image_caption_model,
    get_image_caption_processor=lambda: prompt_enhancer_image_caption_processor,
    get_enhancer_offloadobj=lambda: enhancer_offloadobj,
    acquire_gpu=lambda state: acquire_GPU_ressources(state, "deepy", "Deepy"),
    release_gpu=lambda state, **kwargs: release_GPU_ressources(state, "deepy", process_name="Deepy", **kwargs),
    register_gpu_resident=lambda state, **kwargs: register_GPU_resident(state, "deepy", "Deepy", **kwargs),
    clear_gpu_resident=lambda state: unregister_GPU_resident(state, "deepy"),
    get_new_refresh_id=get_new_refresh_id,
    get_gen_info=get_gen_info,
    get_processed_queue=get_processed_queue,
    get_output_filepath=get_output_filepath,
    record_file_metadata=record_file_metadata,
    exec_prompt_enhancer_engine=exec_prompt_enhancer_engine,
    clear_queue_action=clear_queue_action,
)
release_deepy_vram = _deepy.release_vram


def generate_video_tab(update_form = False, state_dict = None, ui_defaults = None, model_family = None, model_base_type_choice = None, model_choice = None, model_description = None, header = None, main = None, main_tabs= None, tab_id='generate', edit_tab=None, default_state=None):
    global inputs_names #, advanced
    plugin_data = gr.State({})
    edit_mode = tab_id=='edit'

    if update_form:
        model_type = ui_defaults.get("model_type", state_dict["model_type"])
        advanced_ui = state_dict["advanced"]  
    else:
        model_type = transformer_type
        advanced_ui = advanced
        ui_defaults=  get_default_settings(model_type)
        state_dict = {}
        state_dict["model_type"] = model_type
        state_dict["advanced"] = advanced_ui
        state_dict["last_model_per_family"] = server_config.get("last_model_per_family", {})
        state_dict["last_model_per_type"] = server_config.get("last_model_per_type", {})
        state_dict["last_resolution_per_group"] = server_config.get("last_resolution_per_group", {})
        gen = dict()
        gen["queue"] = []
        state_dict["gen"] = gen

    def ui_get(key, default = None):
        if default is None:
            return ui_defaults.get(key, primary_settings.get(key,""))
        else:
            return ui_defaults.get(key, default)
    
    model_def = get_model_def(model_type)
    if model_def == None: model_def = {} 
    base_model_type = get_base_model_type(model_type)
    audio_only = model_def.get("audio_only", False)
    model_filename = get_model_filename( base_model_type )
    preset_to_load = lora_preselected_preset if lora_preset_model == model_type else "" 

    def get_setting_def(key, **context):
        return extra_settings.get_def(key, model_def, model_type=model_type, get_max_frames=get_max_frames, **context)

    def get_container_def(key, **context):
        return extra_settings.get_container_def(key, model_def, model_type=model_type, get_max_frames=get_max_frames, **context)

    def setting_slider(key, *, value=None, visible=None, label=None, minimum=None, maximum=None, step=None, setting_context=None, respect_setting_visibility=True, **kwargs):
        setting_def = get_setting_def(key, **({} if setting_context is None else setting_context))
        return gr.Slider(
            minimum=setting_def.min if minimum is None else minimum,
            maximum=setting_def.max if maximum is None else maximum,
            value=ui_get(key) if value is None else value,
            step=setting_def.step if step is None else step,
            label=setting_def.label if label is None else label,
            visible=setting_def.visible if visible is None else (setting_def.visible and visible if respect_setting_visibility else visible),
            show_reset_button=False,
            **kwargs,
        )

    loras, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset = setup_loras(model_type,  None,  get_lora_dir(model_type), preset_to_load, None)

    state_dict["loras"] = loras
    state_dict["loras_presets"] = loras_presets
    custom_setting_components_map = {}

    launch_prompt = ""
    launch_preset = ""
    launch_loras = []
    launch_multis_str = ""

    if update_form:
        pass
    if len(default_lora_preset) > 0 and lora_preset_model == model_type:
        launch_preset = default_lora_preset
        launch_prompt = default_lora_preset_prompt 
        launch_loras = default_loras_choices
        launch_multis_str = default_loras_multis_str

    if len(launch_preset) == 0:
        launch_preset = ui_defaults.get("lset_name","")
    launch_preset = get_lset_name(state_dict, launch_preset)
    if len(launch_prompt) == 0:
        launch_prompt = ui_defaults.get("prompt","")
    if len(launch_loras) == 0:
        launch_multis_str = ui_defaults.get("loras_multipliers","")
        launch_loras = [os.path.basename(path) for path in ui_defaults.get("activated_loras",[])]
    with gr.Row():
        column_kwargs = {'elem_id': 'edit-tab-content'} if tab_id == 'edit' else {}
        with gr.Column(**column_kwargs):
            with gr.Column(visible=False, elem_id="image-modal-container") as modal_container:
                modal_html_display = gr.HTML()
                modal_action_input = gr.Text(elem_id="modal_action_input", visible=False)
                modal_action_trigger = gr.Button(elem_id="modal_action_trigger", visible=False)
                close_modal_trigger_btn = gr.Button(elem_id="modal_close_trigger_btn", visible=False)
            with gr.Row(visible= True): #len(loras)>0) as presets_column:
                lset_choices = compute_lset_choices(model_type, loras_presets) + [(get_new_preset_msg(advanced_ui), "")]
                with gr.Column(scale=6):
                    lset_name = gr.Dropdown(show_label=False, allow_custom_value= True, scale=6, filterable=True, choices= lset_choices, value=launch_preset)
                with gr.Column(scale=1):
                    with gr.Row(height=17): 
                        apply_lset_btn = gr.Button("Apply", size="sm", min_width= 1)
                        refresh_lora_btn = gr.Button("Refresh", size="sm", min_width= 1, visible=advanced_ui or not only_allow_edit_in_advanced)
                        if len(launch_preset) == 0 : 
                            lset_type = 2   
                        else:
                            lset_type = 1 if launch_preset.endswith(".lset") else 2
                        save_lset_prompt_drop= gr.Dropdown(
                            choices=[
                                # ("Save Loras & Only Prompt Comments", 0),
                                ("Save Only Loras & Full Prompt", 1),
                                ("Save All the Settings", 2)
                            ],  show_label= False, container=False, value = lset_type, visible= False
                        ) 
                    with gr.Row(height=17, visible=False) as refresh2_row:
                        refresh_lora_btn2 = gr.Button("Refresh", size="sm", min_width= 1)

                    with gr.Row(height=17, visible=advanced_ui or not only_allow_edit_in_advanced) as preset_buttons_rows:
                        confirm_save_lset_btn = gr.Button("Go Ahead Save it !", size="sm", min_width= 1, visible=False) 
                        confirm_delete_lset_btn = gr.Button("Go Ahead Delete it !", size="sm", min_width= 1, visible=False) 
                        save_lset_btn = gr.Button("Save", size="sm", min_width= 1, visible = True)
                        delete_lset_btn = gr.Button("Delete", size="sm", min_width= 1, visible = True)
                        cancel_lset_btn = gr.Button("Don't do it !", size="sm", min_width= 1 , visible=False)  
                        #confirm_save_lset_btn, confirm_delete_lset_btn, save_lset_btn, delete_lset_btn, cancel_lset_btn
            t2v =  test_class_t2v(base_model_type) 
            base_model_family = get_model_family(base_model_type)
            diffusion_forcing = "diffusion_forcing" in model_filename 
            ltxv = "ltxv" in model_filename 
            multiple_images_as_text_prompts = model_def.get("multiple_images_as_text_prompts", False)
            inference_steps_enabled = model_def.get("inference_steps", True)
            lock_inference_steps = model_def.get("lock_inference_steps", False) or (audio_only and not inference_steps_enabled)
            any_tea_cache = model_def.get("tea_cache", False)
            any_mag_cache = model_def.get("mag_cache", False)
            recammaster = base_model_type in ["recam_1.3B"]
            vace = test_vace_module(base_model_type)
            multitalk = model_def.get("multitalk_class", False)
            infinitetalk =  base_model_type in ["infinitetalk"]
            hunyuan_t2v = "hunyuan_video_720" in model_filename
            hunyuan_i2v = "hunyuan_video_i2v" in model_filename
            hunyuan_video_custom_edit = base_model_type in ["hunyuan_custom_edit"]
            hunyuan_video_avatar = "hunyuan_video_avatar" in model_filename
            image_outputs = model_def.get("image_outputs", False)
            sliding_window_enabled = test_any_sliding_window(model_type)
            multi_prompts_gen_type_value = ui_get("multi_prompts_gen_type")
            prompt_label, wizard_prompt_label = get_prompt_labels(multi_prompts_gen_type_value, model_def, image_outputs, audio_only)            
            any_video_source = False
            fps = get_model_fps(base_model_type)
            image_prompt_type_value = ""
            video_prompt_type_value = ""
            any_start_image = any_end_image = any_reference_image = any_image_mask = False
            v2i_switch_supported = model_def.get("v2i_switch_supported", False) and not image_outputs
            ti2v_2_2 = base_model_type in ["ti2v_2_2"]
            gallery_height = 350
            def get_image_gallery(label ="", value = None, single_image_mode = False, visible = False ):
                with gr.Row(visible = visible) as gallery_row:
                    gallery_amg = AdvancedMediaGallery(media_mode="image", height=gallery_height, columns=4, label=label, initial = value , single_image_mode = single_image_mode )
                    gallery_amg.mount(update_form=update_form)
                return gallery_row, gallery_amg.gallery, [gallery_row] + gallery_amg.get_toggable_elements()

            image_mode_value = ui_get("image_mode", 1 if image_outputs else 0 )
            if not v2i_switch_supported and not image_outputs:
                image_mode_value = 0
            else:
                image_outputs = image_mode_value > 0 
            inpaint_support = model_def.get("inpaint_support", False)
            image_mode = gr.Number(value =image_mode_value, visible = False)
            image_mode_tab_selected= "t2i" if image_mode_value == 1 else ("inpaint" if image_mode_value == 2 else "t2v") 
            with gr.Tabs(visible = v2i_switch_supported or inpaint_support, selected= image_mode_tab_selected ) as image_mode_tabs:
                with gr.Tab("Text to Video", id = "t2v", elem_classes="compact_tab", visible = v2i_switch_supported) as tab_t2v:
                    pass
                with gr.Tab("Text to Image", id = "t2i", elem_classes="compact_tab"):
                    pass
                with gr.Tab("Image Inpainting", id = "inpaint", elem_classes="compact_tab", visible=inpaint_support) as tab_inpaint:
                    pass

            if audio_only:
                medium = "Audio"
            elif image_outputs:
                medium = "Image"
            else:
                medium = "Video"
            image_prompt_types_allowed = model_def.get("image_prompt_types_allowed", "")
            model_mode_choices = model_def.get("model_modes", None)
            model_modes_visibility = [0,1,2]
            if model_mode_choices is not None: model_modes_visibility= model_mode_choices.get("image_modes", model_modes_visibility)
            if image_mode_value != 0:
                image_prompt_types_allowed = del_in_sequence(image_prompt_types_allowed, "EVL")
            with gr.Column(visible= image_prompt_types_allowed not in ("", "T") or model_mode_choices is not None and image_mode_value in model_modes_visibility ) as image_prompt_column: 
                # Video Continue /  Start Frame / End Frame
                image_prompt_type_value= ui_get("image_prompt_type")
                image_prompt_type = gr.Text(value= image_prompt_type_value, visible= False)
                end_option_visible = end_frames_option_visible(model_def, image_prompt_type_value) and not image_outputs
                image_prompt_type_choices = []
                if "T" in image_prompt_types_allowed: 
                    image_prompt_type_choices += [("Text Prompt Only" if "S" in image_prompt_types_allowed else "New Video", "")]
                if "S" in image_prompt_types_allowed: 
                    image_prompt_type_choices += [("Start Video with Image", "S")]
                    any_start_image = True
                if "V" in image_prompt_types_allowed:
                    any_video_source = True
                    image_prompt_type_choices += [("Continue Video", "V")]
                if "L" in image_prompt_types_allowed:
                    any_video_source = True
                    image_prompt_type_choices += [("Continue Last Video", "L")]
                with gr.Group(visible= len(image_prompt_types_allowed)>1 and image_mode_value == 0) as image_prompt_type_group:
                    with gr.Row():
                        image_prompt_type_radio_allowed_values= filter_letters(image_prompt_types_allowed, "SVL")
                        image_prompt_type_radio_value = filter_letters(image_prompt_type_value, image_prompt_type_radio_allowed_values,  image_prompt_type_choices[0][1] if len(image_prompt_type_choices) > 0 else "")
                        if len(image_prompt_type_choices) > 0:
                            image_prompt_type_radio = gr.Radio( image_prompt_type_choices, value = image_prompt_type_radio_value, label="Location", show_label= False, visible= len(image_prompt_types_allowed)>1, scale= 3)
                        else:
                            image_prompt_type_radio = gr.Radio(choices=[("", "")], value="", visible= False)
                        if "E" in image_prompt_types_allowed:
                            image_prompt_type_endcheckbox = gr.Checkbox( value ="E" in image_prompt_type_value, label="End Image(s)", show_label= False, visible=end_option_visible, scale= 1, elem_classes="cbx_centered")
                            any_end_image = True
                        else:
                            image_prompt_type_endcheckbox = gr.Checkbox( value =False, show_label= False, visible= False , scale= 1)
                image_start_row, image_start, image_start_extra = get_image_gallery(label= "Images as starting points for new Videos in the Generation Queue" + (" (None for Black Frames)" if model_def.get("black_frame", False) else ''), value = ui_defaults.get("image_start", None), visible= "S" in image_prompt_type_value )
                video_source = gr.Video(label= "Video to Continue", height = gallery_height, visible= "V" in image_prompt_type_value, value= ui_defaults.get("video_source", None), elem_id="video_input")
                image_end_row, image_end, image_end_extra = get_image_gallery(label= get_image_end_label(ui_get("multi_prompts_gen_type")), value = ui_defaults.get("image_end", None), visible=end_option_visible and "E" in image_prompt_type_value)
                if model_mode_choices is None:
                    model_mode = gr.Dropdown(value=None, label="model mode", visible=False)
                else:
                    model_mode_value = ui_defaults["model_mode"] = get_default_value(model_mode_choices["choices"], ui_get("model_mode", None), model_mode_choices["default"] )
                    model_mode = gr.Dropdown(choices=model_mode_choices["choices"], value=model_mode_value, label=model_mode_choices["label"],  visible=image_mode_value in model_modes_visibility)                        
                keep_frames_video_source = gr.Text(value=ui_get("keep_frames_video_source") , visible= len(filter_letters(image_prompt_type_value, "VL"))>0 , scale = 2, label= "Truncate Video beyond this number of resampled Frames (empty=Keep All, negative truncates from End)" ) 

            any_control_video = any_control_image = False
            if image_mode_value ==2:
                guide_preprocessing = { "selection": ["V", "VG"]}
                mask_preprocessing = { "selection": ["A"]}
            else:
                guide_preprocessing = model_def.get("guide_preprocessing", None)
                mask_preprocessing = model_def.get("mask_preprocessing", None)
            guide_custom_choices = model_def.get("guide_custom_choices", None)
            image_ref_choices = model_def.get("image_ref_choices", None)

            with gr.Column(visible= guide_preprocessing is not None or mask_preprocessing is not None or guide_custom_choices is not None or image_ref_choices is not None) as video_prompt_column: 
                video_prompt_type_value= ui_get("video_prompt_type")
                video_prompt_type = gr.Text(value= video_prompt_type_value, visible= False)
                dropdown_selectable = True
                image_ref_inpaint = False
                if image_mode_value==2:
                    dropdown_selectable = False
                    image_ref_inpaint = "I" in model_def.get("inpaint_video_prompt_type", "")

                guide_selection_context_visible = dropdown_selectable or image_ref_inpaint
                guide_selector_visible = guide_preprocessing is not None and guide_preprocessing.get("visible", True)
                guide_alt_selector_visible = guide_custom_choices is not None and guide_custom_choices.get("visible", True)
                mask_selector_visible = mask_preprocessing is not None and "V" in video_prompt_type_value and "U" not in video_prompt_type_value and mask_preprocessing.get("visible", True)
                image_ref_selector_visible = image_ref_inpaint or image_ref_choices is not None and image_ref_choices.get("visible", True)
                custom_video_selection = model_def.get("custom_video_selection", None)
                custom_video_trigger = "" if custom_video_selection is None else custom_video_selection.get("trigger", "")
                custom_selector_visible = custom_video_selection is not None and (len(custom_video_trigger) == 0 or custom_video_trigger in video_prompt_type_value)
                guide_selection_visible = guide_selection_context_visible and (guide_selector_visible or guide_alt_selector_visible or custom_selector_visible or mask_selector_visible or image_ref_selector_visible)

                with gr.Row(visible=guide_selection_visible) as guide_selection_row:
                    # Control Video Preprocessing
                    if guide_preprocessing is None:
                        video_prompt_type_video_guide = gr.Dropdown(choices=[("","")], value="", label="Control Video", scale = 2, visible= False, show_label= True, )
                    else:
                        pose_label = "Pose" if image_outputs else "Motion" 
                        guide_preprocessing_labels_all = {
                            "": "No Control Video",
                            "UV": "Keep Control Video Unchanged",
                            "PV": f"Transfer Human {pose_label}",
                            "OV": f"Transfer Aligned Human {pose_label}",
                            "DV": "Transfer Depth",
                            "EV": "Transfer Canny Edges",
                            "SV": "Transfer Shapes",
                            "LV": "Transfer Flow",
                            "CV": "Recolorize",
                            "MV": "Perform Inpainting",
                            "V": "Use Vace raw format",
                            "PDV": f"Transfer Human {pose_label} & Depth",
                            "PSV": f"Transfer Human {pose_label} & Shapes",
                            "PLV": f"Transfer Human {pose_label} & Flow" ,
                            "DSV": "Transfer Depth & Shapes",
                            "DLV": "Transfer Depth & Flow",
                            "SLV": "Transfer Shapes & Flow",
                        }
                        guide_preprocessing_choices = []
                        guide_preprocessing_labels = guide_preprocessing.get("labels", {}) 
                        for process_type in guide_preprocessing["selection"]:
                            process_label = guide_preprocessing_labels.get(process_type, None)
                            process_label = guide_preprocessing_labels_all.get(process_type,process_type) if process_label is None else process_label
                            if image_outputs: process_label = process_label.replace("Video", "Image")
                            guide_preprocessing_choices.append( (process_label, process_type) )

                        video_prompt_type_video_guide_label = guide_preprocessing.get("label", "Control Video Process")
                        if image_outputs: video_prompt_type_video_guide_label = video_prompt_type_video_guide_label.replace("Video", "Image")
                        video_prompt_type_video_guide = gr.Dropdown(
                            guide_preprocessing_choices,
                            value=filter_letters(video_prompt_type_value,  all_guide_processes, guide_preprocessing.get("default", "") ),
                            label= video_prompt_type_video_guide_label , scale = 1, visible= dropdown_selectable and guide_selector_visible , show_label= True,
                        )
                        any_control_video = True
                        any_control_image = image_outputs 

                    # Alternate Control Video Preprocessing / Options
                    if guide_custom_choices is None:
                        video_prompt_type_video_guide_alt = gr.Dropdown(choices=[("","")], value="", label="Control Video", visible= False, scale = 1 )
                    else:
                        video_prompt_type_video_guide_alt_label = guide_custom_choices.get("label", "Control Video Process")
                        if image_outputs: video_prompt_type_video_guide_alt_label = video_prompt_type_video_guide_alt_label.replace("Video", "Image")
                        video_prompt_type_video_guide_alt_choices = [(label.replace("Video", "Image") if image_outputs else label, value) for label,value in guide_custom_choices["choices"] ]
                        guide_guide_custom_choices_letter_filter = guide_custom_choices["letters_filter"]
                        guide_custom_choices_value = get_default_value(video_prompt_type_video_guide_alt_choices, filter_letters(video_prompt_type_value, guide_guide_custom_choices_letter_filter), guide_custom_choices.get("default", "") )
                        video_prompt_type_value = update_value(video_prompt_type_value, guide_custom_choices_value, guide_guide_custom_choices_letter_filter)                        
                        video_prompt_type_video_guide_alt = gr.Dropdown(
                            choices= video_prompt_type_video_guide_alt_choices,
                            value=guide_custom_choices_value,
                            visible = dropdown_selectable and guide_alt_selector_visible,
                            label= video_prompt_type_video_guide_alt_label, show_label= guide_custom_choices.get("show_label", True), scale = guide_custom_choices.get("scale", 1),
                        )
                        any_control_video = True
                        any_control_image = image_outputs 
                        any_reference_image = any("I" in choice for label, choice in guide_custom_choices["choices"])

                    # Custom dropdown box & checkbox
                    custom_checkbox= False 
                    if custom_video_selection is None:
                        video_prompt_type_video_custom_dropbox = gr.Dropdown(choices=[("","")], value="", label="Custom Dropdown", scale = 1, visible= False, show_label= True, )
                        video_prompt_type_video_custom_checkbox = gr.Checkbox(value=False, label="Custom Checkbbox", scale = 1, visible= False, show_label= True, )

                    else:
                        custom_video_choices = custom_video_selection["choices"]
                        custom_checkbox = custom_video_selection.get("type","") == "checkbox"

                        video_prompt_type_video_custom_label = custom_video_selection.get("label", "Custom Choices")
                        video_prompt_type_video_custom_dropbox = gr.Dropdown(
                            custom_video_choices,
                            value=filter_letters(video_prompt_type_value, custom_video_selection.get("letters_filter", ""), custom_video_selection.get("default", "")),
                            scale = custom_video_selection.get("scale", 1),
                            label= video_prompt_type_video_custom_label , visible= dropdown_selectable and not custom_checkbox and custom_selector_visible,
                            show_label= custom_video_selection.get("show_label", True),
                        )
                        video_prompt_type_video_custom_checkbox = gr.Checkbox(value= custom_video_choices[1][1] in video_prompt_type_value , label=custom_video_choices[1][0] , scale = custom_video_selection.get("scale", 1), visible=dropdown_selectable and custom_checkbox and custom_selector_visible, show_label= True, elem_classes="cbx_centered" )

                    # Control Mask Preprocessing
                    if mask_preprocessing is None:
                        video_prompt_type_video_mask = gr.Dropdown(choices=[("","")], value="", label="Video Mask", scale = 1, visible= False, show_label= True, )
                        any_image_mask = image_outputs
                    else:
                        mask_preprocessing_labels_all = {
                            "": "Whole Frame",
                            "A": "Masked Area",
                            "NA": "Non Masked Area",
                            "XA": "Masked Area, rest Inpainted",
                            "XNA": "Non Masked Area, rest Inpainted", 
                            "YA": "Masked Area, rest Depth",
                            "YNA": "Non Masked Area, rest Depth",
                            "WA": "Masked Area, rest Shapes",
                            "WNA": "Non Masked Area, rest Shapes",
                            "ZA": "Masked Area, rest Flow",
                            "ZNA": "Non Masked Area, rest Flow"
                        }

                        mask_preprocessing_choices = []
                        mask_preprocessing_labels = mask_preprocessing.get("labels", {}) 
                        for process_type in mask_preprocessing["selection"]:
                            process_label = mask_preprocessing_labels.get(process_type, None)
                            process_label = mask_preprocessing_labels_all.get(process_type, process_type) if process_label is None else process_label
                            mask_preprocessing_choices.append( (process_label, process_type) )

                        video_prompt_type_video_mask_label = mask_preprocessing.get("label", "Area Processed")
                        video_prompt_type_video_mask = gr.Dropdown(
                            mask_preprocessing_choices,
                            value=filter_letters(video_prompt_type_value, "XYZWNA", mask_preprocessing.get("default", "")),
                            label= video_prompt_type_video_mask_label , scale = 1, visible= dropdown_selectable and mask_selector_visible,
                            show_label= True,
                        )
                        any_control_video = True
                        any_control_image = image_outputs 

                    # Image Refs Selection
                    if image_ref_inpaint:
                       image_ref_choices = { "choices": [("None", ""), ("People / Objects", "I"), ], "letters_filter":  "I", }

                    if image_ref_choices is None:
                        video_prompt_type_image_refs = gr.Dropdown(
                            # choices=[ ("None", ""),("Start", "KI"),("Ref Image", "I")],
                            choices=[ ("None", ""),],
                            value=filter_letters(video_prompt_type_value, ""),
                            visible = False,
                            label="Start / Reference Images", scale = 1
                        )
                    else:
                        any_reference_image = True
                        video_prompt_type_image_refs = gr.Dropdown(
                            choices= image_ref_choices["choices"],
                            value=filter_letters(video_prompt_type_value, image_ref_choices["letters_filter"]),
                            visible = guide_selection_context_visible and image_ref_selector_visible,
                            label=image_ref_choices.get("label", "Inject Reference Images"), show_label= image_ref_choices.get("show_label", True), scale = 1
                        )

                image_guide = gr.Image(label= "Control Image", height = 800, type ="pil", visible= image_mode_value==1 and "V" in video_prompt_type_value and ("U" in video_prompt_type_value or "A" not in video_prompt_type_value ) , value= ui_defaults.get("image_guide", None))
                video_guide = gr.Video(label= "Control Video", height = gallery_height, visible= (not image_outputs) and "V" in video_prompt_type_value, value= ui_defaults.get("video_guide", None), elem_id="video_input")
                magic_mask_visible = "V" in video_prompt_type_value and "A" in video_prompt_type_value and "U" not in video_prompt_type_value
                magic_mask_uis = []
                if image_mode_value >= 1:  
                    image_guide_value = ui_defaults.get("image_guide", None)
                    image_mask_value = ui_defaults.get("image_mask", None)
                    if image_guide_value is None:
                        image_mask_guide_value = None
                    else:
                        image_mask_guide_value = { "background" : image_guide_value, "composite" : None}
                        image_mask_guide_value["layers"] = [] if image_mask_value is None else [rgb_bw_to_rgba_mask(image_mask_value)]

                    with gr.Column(elem_classes=["wangp-magic-mask-anchor", "wangp-magic-mask-anchor--image-editor"]):
                        image_mask_guide = gr.ImageEditor(
                            label="Control Image to be Inpainted" if image_mode_value == 2 else "Control Image and Mask",
                            value = image_mask_guide_value,
                            type='pil',
                            sources=["upload", "webcam"],
                            image_mode='RGB',
                            layers=False,
                            brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                            # fixed_canvas= True,
                            # width=800,
                            height=800,
                            # transforms=None,
                            # interactive=True,
                            elem_id="img_editor",
                            visible=magic_mask_visible
                        )
                        magic_mask_ui = MagicMaskUI().render(visible=magic_mask_visible, trigger_mode="editor")
                        magic_mask_uis.append(magic_mask_ui)
                        magic_mask_image_btn = magic_mask_ui.trigger
                    any_control_image = True
                else:
                    with gr.Column(elem_classes=["wangp-magic-mask-anchor", "wangp-magic-mask-anchor--image-editor"]):
                        image_mask_guide = gr.ImageEditor(value = None, visible = False, elem_id="img_editor")
                        magic_mask_ui = MagicMaskUI().render(visible=False, trigger_mode="editor")
                        magic_mask_uis.append(magic_mask_ui)
                        magic_mask_image_btn = magic_mask_ui.trigger


                denoising_strength = setting_slider("denoising_strength", visible="G" in video_prompt_type_value)
                keep_frames_video_guide_visible = not image_outputs and  "V" in video_prompt_type_value and not model_def.get("keep_frames_video_guide_not_supported", False)
                keep_frames_video_guide = gr.Text(value=ui_get("keep_frames_video_guide") , visible= keep_frames_video_guide_visible  , scale = 2, label= "Frames to keep in Control Video (empty=All, 1=first, a:b for a range, space to separate values)" ) #, -1=last
                video_guide_outpainting_modes = model_def.get("video_guide_outpainting", [])
                with gr.Column(visible= ("V" in video_prompt_type_value  or "K" in video_prompt_type_value  or "F" in video_prompt_type_value) and image_mode_value in video_guide_outpainting_modes) as video_guide_outpainting_col:
                    video_guide_outpainting_value = ui_get("video_guide_outpainting")
                    video_guide_outpainting_ratio_value = ui_get("video_guide_outpainting_ratio", "")
                    video_guide_outpainting = gr.Text(value=video_guide_outpainting_value , visible= False)
                    with gr.Group():
                        video_guide_outpainting_enabled = not video_guide_outpainting_value.startswith("#") and (len(video_guide_outpainting_value) > 0 or len(video_guide_outpainting_ratio_value) > 0)
                        outpainting_label_suffix = "%" if len(video_guide_outpainting_ratio_value) == 0 else "x"
                        with gr.Row():
                            video_guide_outpainting_checkbox = gr.Checkbox(label=model_def.get("video_guide_outpainting_label", "Enable Spatial Outpainting on Control Video, Landscape or Positioned Reference Frames") if image_mode_value == 0 else "Enable Spatial Outpainting on Control Image", value=video_guide_outpainting_enabled, scale=3)
                            video_guide_outpainting_ratio = gr.Dropdown([("Manual Expansion", ""), ("Fit into a 1:1 Box", "1:1"), ("Fit into a 4:3 Box", "4:3"), ("Fit into a 3:4 Box", "3:4"), ("Fit into a 16:9 Box", "16:9"), ("Fit into a 9:16 Box", "9:16"), ("Fit into a 21:9 Box", "21:9"), ("Fit into a 9:21 Box", "9:21")], value=video_guide_outpainting_ratio_value, visible=video_guide_outpainting_enabled, show_label=False, allow_custom_value=False, scale=1)
                        with gr.Row(visible = not video_guide_outpainting_value.startswith("#")) as video_guide_outpainting_row:
                            video_guide_outpainting_value = video_guide_outpainting_value[1:] if video_guide_outpainting_value.startswith("#") else video_guide_outpainting_value
                            video_guide_outpainting_list = [0] * 4 if len(video_guide_outpainting_value) == 0 else [int(v) for v in video_guide_outpainting_value.split(" ")]
                            video_guide_outpainting_top= gr.Slider(0, 100, value= video_guide_outpainting_list[0], step=5, label=f"Top {outpainting_label_suffix}", show_reset_button= False)
                            video_guide_outpainting_bottom = gr.Slider(0, 100, value= video_guide_outpainting_list[1], step=5, label=f"Bottom {outpainting_label_suffix}", show_reset_button= False)
                            video_guide_outpainting_left = gr.Slider(0, 100, value= video_guide_outpainting_list[2], step=5, label=f"Left {outpainting_label_suffix}", show_reset_button= False)
                            video_guide_outpainting_right = gr.Slider(0, 100, value= video_guide_outpainting_list[3], step=5, label=f"Right {outpainting_label_suffix}", show_reset_button= False)
                # image_mask = gr.Image(label= "Image Mask Area (for Inpainting, white = Control Area, black = Unchanged)", type ="pil", visible= image_mode_value==1 and "V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value , height = gallery_height, value= ui_defaults.get("image_mask", None)) 
                image_mask = gr.Image(label= "Image Mask Area (for Inpainting, white = Control Area, black = Unchanged)", type ="pil", visible= False, height = gallery_height, value= ui_defaults.get("image_mask", None)) 
                with gr.Column(elem_classes=["wangp-magic-mask-anchor"]):
                    video_mask = gr.Video(label= "Video Mask Area (for Inpainting, white = Control Area, black = Unchanged)", visible=(not image_outputs) and magic_mask_visible, height=gallery_height, value=ui_defaults.get("video_mask", None))
                    if not image_outputs:
                        magic_mask_ui = MagicMaskUI().render(visible=magic_mask_visible)
                        magic_mask_uis.append(magic_mask_ui)
                        magic_mask_video_btn = magic_mask_ui.trigger
                    else:
                        magic_mask_ui = MagicMaskUI().render(visible=False)
                        magic_mask_uis.append(magic_mask_ui)
                        magic_mask_video_btn = magic_mask_ui.trigger
                mask_strength_always_enabled = model_def.get("mask_strength_always_enabled", False)  
                masking_strength = setting_slider("masking_strength", visible=(mask_strength_always_enabled or "G" in video_prompt_type_value) and "V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value)
                mask_expand = setting_slider("mask_expand", visible="V" in video_prompt_type_value and "A" in video_prompt_type_value and not "U" in video_prompt_type_value)

                image_refs_single_image_mode = model_def.get("one_image_ref_needed", False)
                image_refs_label = "Start Image" if hunyuan_video_avatar else ("Reference Image" if image_refs_single_image_mode else "Reference Images")  + (" (each Image will be associated to a Sliding Window)" if infinitetalk else "")
                image_refs_row, image_refs, image_refs_extra = get_image_gallery(label= image_refs_label, value = ui_defaults.get("image_refs", None), visible= "I" in video_prompt_type_value, single_image_mode=image_refs_single_image_mode)

                frames_positions = gr.Text(value=ui_get("frames_positions") , visible= "F" in video_prompt_type_value, scale = 2, label= "Positions of Injected Frames (1=first, L=last of a window) no position for other Image Refs)" ) 
                image_refs_relative_size = setting_slider("image_refs_relative_size", visible=image_outputs)

                no_background_removal = model_def.get("no_background_removal", False) #or image_ref_choices is None
                background_removal_label = model_def.get("background_removal_label", "Remove Background behind People / Objects") 
 
                remove_background_images_ref = gr.Dropdown(
                    choices=[
                        ("Keep Backgrounds behind all Reference Images", 0),
                        (background_removal_label, 1),
                    ],
                    value=0 if no_background_removal else ui_get("remove_background_images_ref"),
                    label="Automatic Removal of Background behind People or Objects in Reference Images", scale = 3, visible= "I" in video_prompt_type_value and not no_background_removal
                )

            input_video_strength = setting_slider("input_video_strength", value=ui_get("input_video_strength", 1.0), visible=input_video_strength_visible(model_def, image_prompt_type_value, ui_get("video_prompt_type")))

            any_audio_prompt = model_def.get("any_audio_prompt", False)
            audio_prompt_type_sources_def = model_def.get("audio_prompt_type_sources", None)
            audio_prompt_type_value = ui_get("audio_prompt_type", "A" if any_audio_prompt and audio_prompt_type_sources_def is None else "")
            any_multi_speakers = False
            any_audio_guide2 = False 
            if any_audio_prompt:
                any_single_speaker = not model_def.get("multi_speakers_only", False)
                any_multi_speakers = not model_def.get("one_speaker_only", False) and not audio_only
                audio_prompt_type_sources_labels_all = {
                    "": "None",
                    "A": "One Person Speaking Only",
                    "XA": "Two speakers, Auto Separation of Speakers (will work only if Voices are distinct)",
                    "CAB": "Two speakers, Speakers Audio sources are assumed to be played in a Row",
                    "PAB": "Two speakers, Speakers Audio sources are assumed to be played in Parallel",
                    "K": "Control Video Audio Track",
                }
                if not isinstance(audio_prompt_type_sources_def, dict):
                    selection = [""]
                    if any_single_speaker:
                        selection.append("A")
                    if any_multi_speakers:
                        selection.extend(["XA", "CAB", "PAB"])
                    audio_prompt_type_sources_def = { "selection": selection, "label": "Voices", "scale": 3, "show_label": True, "visible": True, }

                selection = audio_prompt_type_sources_def.get("selection", list(audio_prompt_type_sources_labels_all.keys()))
                audio_prompt_type_sources_labels = audio_prompt_type_sources_def.get("labels", {})
                audio_prompt_type_sources_choices = []
                for choice in selection:
                    if "B" in choice: any_audio_guide2 = True
                    label = audio_prompt_type_sources_labels.get(choice, audio_prompt_type_sources_labels_all.get(choice, choice))
                    audio_prompt_type_sources_choices.append((label, choice))
                if len(audio_prompt_type_sources_choices) == 0:
                    audio_prompt_type_sources_choices = [(audio_prompt_type_sources_labels_all[""], "")]
                letters_filter = audio_prompt_type_sources_def.get("letters_filter", "XCPABOKF")
                default_choice = audio_prompt_type_sources_def.get("default", "")
                audio_prompt_type_sources_value = filter_letters(audio_prompt_type_value, letters_filter, default_choice)
                audio_prompt_type_sources_value = get_default_value(audio_prompt_type_sources_choices, audio_prompt_type_sources_value, default_choice)
                audio_prompt_type_value = update_value(audio_prompt_type_value, audio_prompt_type_sources_value, letters_filter)
                sources_visible = model_def.get("audio_prompt_choices") is not None and not image_outputs and audio_prompt_type_sources_def.get("visible", True)
                audio_prompt_type_sources = gr.Dropdown(
                    audio_prompt_type_sources_choices,
                    value=audio_prompt_type_sources_value,
                    label=audio_prompt_type_sources_def.get("label", "Voices"),
                    scale=audio_prompt_type_sources_def.get("scale", 3),
                    visible=sources_visible,
                    show_label=audio_prompt_type_sources_def.get("show_label", True),
                )
            else:
                audio_prompt_type_sources_value = ""
                audio_prompt_type_sources = gr.Dropdown(choices=[""], value="", visible=False)

            audio_prompt_type = gr.Text(value=audio_prompt_type_value, visible=False)

            with gr.Row(visible = any_audio_prompt and any_letters(audio_prompt_type_sources_value,"AB") and not image_outputs) as audio_guide_row:
                any_audio_guide = any_audio_prompt and not image_outputs
                audio_guide = gr.Audio(value= ui_defaults.get("audio_guide", None), type="filepath", label= model_def.get("audio_guide_label","Voice to follow"), show_download_button= True, visible= any_audio_prompt and any_letters(audio_prompt_type_value, "A") )
                audio_guide2 = gr.Audio(value= ui_defaults.get("audio_guide2", None), type="filepath", label=model_def.get("audio_guide2_label","Voice to follow #2"), show_download_button= True, visible= any_audio_prompt and "B" in audio_prompt_type_value )
            custom_guide_def = model_def.get("custom_guide", None)
            any_custom_guide= custom_guide_def is not None
            with gr.Row(visible = any_custom_guide) as custom_guide_row:
                if custom_guide_def is None:
                    custom_guide = gr.File(value= None, type="filepath", label= "Custom Guide", height=41, visible= False )
                else:
                    custom_guide = gr.File(value= ui_defaults.get("custom_guide", None), type="filepath", label= custom_guide_def.get("label","Custom Guide"), height=41, visible= True, file_types = custom_guide_def.get("file_types", ["*.*"]) )
            remove_background_visible = any_audio_prompt and any_letters(audio_prompt_type_value, "ABXK") and not image_outputs
            normalize_audio_visible = any_audio_prompt and all_letters(audio_prompt_type_value, "AB") and not image_outputs
            with gr.Row(visible=remove_background_visible or normalize_audio_visible) as audio_options_row:
                remove_background_sound = gr.Checkbox(label="Remove Background Music" if audio_only else "Ignore Background Music (for better LipSync)", value="V" in audio_prompt_type_value, visible=remove_background_visible)
                continue_beyond_audio_end = gr.Checkbox(label="Video Length not Limited by Audio", value="L" in audio_prompt_type_value, visible=model_def.get("video_length_not_limited_by_audio", False))
                normalize_audio_volumes = gr.Checkbox(label="Normalize Audio Volumes", value="N" in audio_prompt_type_value, visible=normalize_audio_visible)
            with gr.Row(visible = any_audio_prompt and any_multi_speakers and ("B" in audio_prompt_type_value or "X" in audio_prompt_type_value) and not image_outputs ) as speakers_locations_row:
                speakers_locations = gr.Text( ui_get("speakers_locations"), label="Speakers Locations separated by a Space. Each Location = Left:Right or a BBox Left:Top:Right:Bottom", visible= True)

            advanced_prompt = advanced_ui
            prompt_vars=[]

            client_id = gr.Textbox( visible= False, value=ui_get("client_id", ""))
            if advanced_prompt:
                default_wizard_prompt, variables, values= None, None, None
            else:                 
                default_wizard_prompt, variables, values, errors =  extract_wizard_prompt(launch_prompt)
                advanced_prompt  = len(errors) > 0
            with gr.Column(visible= advanced_prompt) as prompt_column_advanced:
                prompt = gr.Textbox( visible= advanced_prompt, label=prompt_label, value=launch_prompt, lines=3)

            with gr.Column(visible=not advanced_prompt and len(variables) > 0) as prompt_column_wizard_vars:
                gr.Markdown("<B>Please fill the following input fields to adapt automatically the Prompt:</B>")
                wizard_prompt_activated = "off"
                wizard_variables = ""
                with gr.Row():
                    if not advanced_prompt:
                        for variable in variables:
                            value = values.get(variable, "")
                            prompt_vars.append(gr.Textbox( placeholder=variable, min_width=80, show_label= False, info= variable, visible= True, value= "\n".join(value) ))
                        wizard_prompt_activated = "on"
                        if len(variables) > 0:
                            wizard_variables = "\n".join(variables)
                    for _ in range( PROMPT_VARS_MAX - len(prompt_vars)):
                        prompt_vars.append(gr.Textbox(visible= False, min_width=80, show_label= False))
            with gr.Column(visible=not advanced_prompt) as prompt_column_wizard:
                wizard_prompt = gr.Textbox(visible = not advanced_prompt, label=wizard_prompt_label, value=default_wizard_prompt, lines=3)
                wizard_prompt_activated_var = gr.Text(wizard_prompt_activated, visible= False)
                wizard_variables_var = gr.Text(wizard_variables, visible = False)
            with gr.Row(visible= server_config.get("enhancer_enabled", 0) > 0  ) as prompt_enhancer_row:
                on_demand_prompt_enhancer = server_config.get("enhancer_mode", 0) == 1
                prompt_enhancer_value = str(ui_get("prompt_enhancer") or "")
                prompt_enhancer_btn_label = str(model_def.get("prompt_enhancer_button_label", "Enhance Prompt"))
                prompt_enhancer_btn = gr.Button( value =prompt_enhancer_btn_label, visible= on_demand_prompt_enhancer, size="lg", scale=1, elem_classes="btn_centered")
                prompt_enhancer_choices = [] if on_demand_prompt_enhancer else [("Disabled", "")]
                prompt_enhancer_default = ""
                prompt_enhancer_default_labels = {
                    "T": "Based on Text Prompt Content",
                    "I": "Based on Images Prompts Content (such as Start Image and Reference Images)",
                    "TI": "Based on both Text Prompt and Images Prompts Content",
                }
                prompt_enhancer_def = model_def.get("prompt_enhancer_def")
                if isinstance(prompt_enhancer_def, dict):
                    prompt_enhancer_selection = prompt_enhancer_def.get("selection", [])
                    if isinstance(prompt_enhancer_selection, str):
                        prompt_enhancer_selection = [prompt_enhancer_selection]
                    if not isinstance(prompt_enhancer_selection, list):
                        prompt_enhancer_selection = []
                    prompt_enhancer_labels_override = prompt_enhancer_def.get("labels", {})
                    if not isinstance(prompt_enhancer_labels_override, dict):
                        prompt_enhancer_labels_override = {}
                    for selection_value in prompt_enhancer_selection:
                        selection_value = str(selection_value).strip()
                        if len(selection_value) == 0:
                            continue
                        display_label = prompt_enhancer_labels_override.get(selection_value, prompt_enhancer_default_labels.get(selection_value, selection_value))
                        prompt_enhancer_choices.append((str(display_label), selection_value))
                    prompt_enhancer_default = str(prompt_enhancer_def.get("default", "")).strip()
                else:
                    prompt_enhancer_choices_allowed = model_def.get("prompt_enhancer_choices_allowed", ["T"] if audio_only else ["T", "I", "TI"])
                    if isinstance(prompt_enhancer_choices_allowed, str):
                        prompt_enhancer_choices_allowed = [prompt_enhancer_choices_allowed]
                    if not isinstance(prompt_enhancer_choices_allowed, list):
                        prompt_enhancer_choices_allowed = []
                    for selection_value in prompt_enhancer_choices_allowed:
                        selection_value = str(selection_value).strip()
                        if len(selection_value) == 0:
                            continue
                        display_label = prompt_enhancer_default_labels.get(selection_value, selection_value)
                        prompt_enhancer_choices.append((display_label, selection_value))

                prompt_enhancer_letters_filter = get_prompt_enhancer_letters_filter(prompt_enhancer_def, prompt_enhancer_choices)
                prompt_enhancer_values = [value for _, value in prompt_enhancer_choices]
                prompt_enhancer_mode_value = filter_letters(prompt_enhancer_value, prompt_enhancer_letters_filter, prompt_enhancer_default)
                prompt_enhancer_mode_value = get_default_value(prompt_enhancer_choices, prompt_enhancer_mode_value, prompt_enhancer_default)
                prompt_enhancer_value = update_value(prompt_enhancer_value, prompt_enhancer_mode_value, prompt_enhancer_letters_filter)
                if prompt_enhancer_mode_value not in prompt_enhancer_values:
                    if prompt_enhancer_default in prompt_enhancer_values:
                        prompt_enhancer_mode_value = prompt_enhancer_default
                    elif len(prompt_enhancer_values) > 0:
                        prompt_enhancer_mode_value = prompt_enhancer_values[0]
                    else:
                        prompt_enhancer_mode_value = ""
                elif len(prompt_enhancer_mode_value) == 0 and on_demand_prompt_enhancer and len(prompt_enhancer_values) > 0:
                    if prompt_enhancer_default in prompt_enhancer_values:
                        prompt_enhancer_mode_value = prompt_enhancer_default
                    else:
                        prompt_enhancer_mode_value = prompt_enhancer_values[0]

                prompt_enhancer_think_visible = server_config.get("enhancer_enabled", 0) in (3, 4)
                prompt_enhancer_think_value = prompt_enhancer_think_visible and "K" in prompt_enhancer_value and len(prompt_enhancer_mode_value) > 0
                prompt_enhancer_value = build_prompt_enhancer_value(prompt_enhancer_mode_value, prompt_enhancer_think_value)
                prompt_enhancer = gr.Text(value=prompt_enhancer_value, visible=False)
                prompt_enhancer_mode_dropdown = gr.Dropdown(
                    choices=prompt_enhancer_choices,
                    value=prompt_enhancer_mode_value,
                    label=model_def.get("prompt_enhancer_button_label", "Enhance Prompt using a LLM") , scale = 5,
                    visible= True, show_label= not on_demand_prompt_enhancer,
                )
                prompt_enhancer_think = gr.Checkbox(label="Think", value=prompt_enhancer_think_value, visible=prompt_enhancer_think_visible, scale=1, elem_classes="cbx_centered")
            alt_prompt_def = model_def.get("alt_prompt", None)
            alt_prompt_label = None
            alt_prompt_placeholder = ""
            alt_prompt_lines = 2
            if isinstance(alt_prompt_def, dict):
                alt_prompt_label = alt_prompt_def.get("label")
                alt_prompt_placeholder = alt_prompt_def.get("placeholder", "")
                alt_prompt_lines = alt_prompt_def.get("lines", 2)
            if alt_prompt_label:
                with gr.Row(visible=True) as alt_prompt_row:
                    alt_prompt = gr.Textbox(
                        label=alt_prompt_label,
                        value=ui_get("alt_prompt", ""),
                        lines=alt_prompt_lines,
                        placeholder=alt_prompt_placeholder,
                        visible=True,
                    )
            else:
                with gr.Row(visible=False) as alt_prompt_row:
                    alt_prompt = gr.Textbox(value=ui_get("alt_prompt", ""), visible=False)

            custom_settings = get_model_custom_settings(model_def)
            custom_settings_values = ui_get("custom_settings", None)
            if not isinstance(custom_settings_values, dict):
                custom_settings_values = {}
            custom_settings_rows = []
            custom_setting_rows_count = math.ceil(CUSTOM_SETTINGS_MAX / CUSTOM_SETTINGS_PER_ROW)
            for row_idx in range(custom_setting_rows_count):
                row_start = row_idx * CUSTOM_SETTINGS_PER_ROW
                row_end = min(row_start + CUSTOM_SETTINGS_PER_ROW, CUSTOM_SETTINGS_MAX)
                row_visible = row_start < len(custom_settings)
                with gr.Row(visible=row_visible) as custom_settings_row:
                    for setting_index in range(row_start, row_end):
                        setting_key = get_custom_setting_key(setting_index)
                        setting_def = custom_settings[setting_index] if setting_index < len(custom_settings) else None
                        setting_visible = setting_def is not None
                        setting_default = get_custom_setting_value_from_dict(custom_settings_values, setting_def, setting_index) if setting_def is not None else ""
                        if setting_default is None:
                            setting_default = ""
                        setting_label = setting_def.get("label", f"Custom Setting {setting_index + 1}") if setting_def is not None else f"Custom Setting {setting_index + 1}"
                        custom_setting_component = gr.Textbox(
                            value=str(setting_default),
                            label=setting_label,
                            visible=setting_visible,
                            lines=1,
                        )
                        custom_setting_components_map[setting_key] = custom_setting_component
                custom_settings_rows.append(custom_settings_row)


            duration_def = model_def.get("duration_slider", None)
            duration_visible = audio_only and duration_def is not None
            if duration_def is None:
                duration_min = 0
                duration_max = 1
                duration_step = 1
                duration_default = 0
                duration_label = "Max Duration"
            else:
                duration_min = duration_def.get("min", 30)
                duration_max = get_max_duration(duration_def.get("max", 240))
                duration_step = duration_def.get("increment", 1)
                duration_default = duration_def.get("default", 120)
                duration_label = duration_def.get("label", "Max Duration")
            duration_value = ui_get("duration_seconds", duration_default)
            try:
                duration_value = float(duration_value)
            except Exception:
                duration_value = duration_default
            if duration_value < duration_min:
                duration_value = duration_min
            elif duration_value > duration_max:
                duration_value = duration_max
            duration_seconds = gr.Slider(
                duration_min,
                duration_max,
                value=duration_value,
                step=duration_step,
                label=duration_label,
                visible=duration_visible,
                show_reset_button=False,
            )

            with gr.Row(visible=not audio_only) as resolution_row:
                fit_canvas = server_config.get("fit_canvas", 0)
                if fit_canvas == 1:
                    label = "Outer Box Resolution (one dimension may be less to preserve video W/H ratio)"
                elif fit_canvas == 2:
                    label = "Output Resolution (Input Images wil be Cropped if the W/H ratio is different)"
                else:
                    label = "Resolution Budget (Pixels will be reallocated to preserve Inputs W/H ratio)" 
                current_resolution_choice = ui_get("resolution") if update_form or last_resolution is None else last_resolution
                model_resolutions = model_def.get("resolutions", None)
                resolution_choices, current_resolution_choice = get_resolution_choices(current_resolution_choice, model_resolutions)
                available_groups, selected_group_resolutions, selected_group = group_resolutions(model_def,resolution_choices, current_resolution_choice)
                resolution_group = gr.Dropdown(
                choices = available_groups,
                    value= selected_group,
                    label= "Category" 
                )
                resolution = gr.Dropdown(
                choices = selected_group_resolutions,
                    value= current_resolution_choice,
                    label= label,
                    scale = 5
                )
            with gr.Row(visible= not audio_only) as number_frames_row:
                batch_label = model_def.get("batch_size_label", "Number of Images to Generate")
                batch_size = gr.Slider(1, 16, value=ui_get("batch_size"), step=1, label=batch_label, visible = image_outputs, show_reset_button= False)
                if image_outputs:
                    video_length = gr.Slider(1, 9999, value=ui_get("video_length"), step=1, label="Number of frames", visible = False, show_reset_button= False)
                else:
                    video_length_locked = model_def.get("video_length_locked", None)
                    min_frames, frames_step, _ = get_model_min_frames_and_step(base_model_type)
                    
                    current_video_length = video_length_locked if video_length_locked is not None else ui_get("video_length", 81 if get_model_family(base_model_type)=="wan" else 97)

                    computed_fps = get_computed_fps(ui_get("force_fps"), base_model_type , ui_defaults.get("video_guide", None), ui_defaults.get("video_source", None))
                    video_length = gr.Slider(0 if audio_only else min_frames, get_max_frames(737 if test_any_sliding_window(base_model_type) else 337), value=current_video_length, 
                         step=frames_step, label=compute_video_length_label(computed_fps, current_video_length, video_length_locked) , visible = True, interactive= video_length_locked is None, show_reset_button= False)

            with gr.Row(visible = not lock_inference_steps) as inference_steps_row:                                       
                num_inference_steps = gr.Slider(0 if audio_only else 1, 100, value=ui_get("num_inference_steps"), step=1, label="Number of Inference Steps", visible = True, show_reset_button= False)


            show_advanced = gr.Checkbox(label="Advanced Mode", value=advanced_ui)
            with gr.Tabs(visible=advanced_ui) as advanced_row:
                guidance_max_phases = model_def.get("guidance_max_phases", 0)
                no_negative_prompt = model_def.get("no_negative_prompt", False)
                with gr.Tab("General"):
                    with gr.Column():
                        with gr.Row():                        
                            seed = gr.Slider(-1, 999999999, value=ui_get("seed"), step=1, label="Seed (-1 for random)", scale=2, show_reset_button= False) 
                            if model_def.get("lock_guidance_phases", False):
                                guidance_phases_value = model_def.get("guidance_max_phases", 0)
                            else:
                                guidance_phases_value = ui_get("guidance_phases") 
                            visible_phases = model_def.get("visible_phases", 3) 
                            guidance_phases = gr.Dropdown(
                                choices=[
                                    ("One Phase", 1),
                                    ("Two Phases", 2)] + ([("Three Phases", 3)] if guidance_max_phases >=3 else []),
                                value= guidance_phases_value,
                                label="Guidance Phases" if visible_phases>=2 else "Phases",
                                visible= guidance_max_phases >=2 , 
                                interactive = not model_def.get("lock_guidance_phases", False)
                            )
                        with gr.Row(visible = get_container_def("guidance_phases_row").visible and guidance_phases_value >= 2) as guidance_phases_row:
                            multiple_submodels = model_def.get("multiple_submodels", False)
                            model_switch_phase = gr.Dropdown(
                                choices=[
                                    ("Phase 1-2 transition", 1),
                                    ("Phase 2-3 transition", 2)],
                                value=ui_get("model_switch_phase"),
                                label="Model Switch",
                                visible= model_def.get("multiple_submodels", False) and guidance_phases_value >= 3 and multiple_submodels
                            )
                            switch_threshold = setting_slider("switch_threshold", setting_context={"guidance_phases": guidance_phases_value})
                            switch_threshold2 = setting_slider("switch_threshold2", visible=guidance_phases_value >= 3)
                        with gr.Row(visible = get_container_def("guidance_row").visible ) as guidance_row:
                            guidance_scale = setting_slider("guidance_scale", setting_context={"guidance_phases": guidance_phases_value})
                            guidance2_scale = setting_slider("guidance2_scale", setting_context={"guidance_phases": guidance_phases_value}, visible=guidance_phases_value >= 2)
                            guidance3_scale = setting_slider("guidance3_scale", setting_context={"guidance_phases": guidance_phases_value}, visible=guidance_phases_value >= 3)

                        any_audio_guidance = model_def.get("audio_guidance", False) 
                        any_embedded_guidance = model_def.get("embedded_guidance", False)
                        alt_guidance_type = model_def.get("alt_guidance", None)
                        any_alt_guidance = alt_guidance_type is not None
                        alt_scale_type = model_def.get("alt_scale", None)
                        any_alt_scale = alt_scale_type is not None
                        with gr.Row(visible = get_container_def("embedded_guidance_row").visible) as embedded_guidance_row:
                            audio_guidance_scale = setting_slider("audio_guidance_scale")
                            embedded_guidance_scale = setting_slider("embedded_guidance_scale")
                            alt_guidance_scale = setting_slider("alt_guidance_scale")
                            alt_scale = setting_slider("alt_scale")

                        with gr.Row(visible=model_def.get("pause_between_sentences", False)) as pause_row:
                            pause_seconds = gr.Slider(minimum=0.0, maximum=2.0,  value=ui_get("pause_seconds"), step=0.05, label="Pause between Multi Speakers sentences (seconds)", show_reset_button=False,)

                        with gr.Row(visible=get_container_def("temperature_row").visible) as temperature_row:
                            temperature = setting_slider("temperature")
                        with gr.Row(visible = get_container_def("top_pk_row").visible ) as top_pk_row:
                            top_p = setting_slider("top_p", value=ui_get("top_p", 0.9))
                            top_k = setting_slider("top_k", value=ui_get("top_k", 50))

                        sample_solver_choices = model_def.get("sample_solvers", None)
                        with gr.Row(visible = get_container_def("sample_solver_row").visible ) as sample_solver_row:
                            if sample_solver_choices is None:
                                sample_solver = gr.Dropdown( value="",  choices=[ ("", ""), ], visible= False, label= "Sampler Solver / Scheduler" )
                            else:
                                sample_solver = gr.Dropdown( value=ui_get("sample_solver", sample_solver_choices[0][1]), 
                                    choices= sample_solver_choices, visible= True, label= "Sampler Solver / Scheduler"
                                )
                            flow_shift = setting_slider("flow_shift") 
                        with gr.Row(visible=get_container_def("control_net_weights_row").visible) as control_net_weights_row:
                            control_net_weight = setting_slider("control_net_weight")
                            control_net_weight2 = setting_slider("control_net_weight2")
                            control_net_weight_alt = setting_slider("control_net_weight_alt")
                            audio_scale = setting_slider("audio_scale", value=ui_get("audio_scale", 1))
                        with gr.Row(visible = not (hunyuan_t2v or hunyuan_i2v or no_negative_prompt)) as negative_prompt_row:

                            negative_prompt = gr.Textbox(label="Negative Prompt " + ("(ignored if NAG is disabled and no Guidance that is if CFG = 1)"  if get_container_def("NAG_col").visible else "(ignored if no Guidance that is if CFG = 1)") , value=ui_get("negative_prompt")  )
                        with gr.Column(visible = get_container_def("NAG_col").visible) as NAG_col:
                            gr.Markdown("<B>NAG enforces Negative Prompt even if no Guidance is set (CFG = 1), set NAG Scale to > 1 to enable it</B>")
                            with gr.Row():
                                NAG_scale = setting_slider("NAG_scale", visible=True)
                                NAG_tau = setting_slider("NAG_tau", visible=True)
                                NAG_alpha = setting_slider("NAG_alpha", visible=True)
                        with gr.Row():
                            repeat_generation = gr.Slider(1, 25.0, value=ui_get("repeat_generation"), step=1, label=f"Num. of Generated {'Audio Files' if audio_only else 'Videos'} per Prompt", visible = not image_outputs, show_reset_button= False) 
                            multi_images_gen_type = gr.Dropdown( value=ui_get("multi_images_gen_type"), 
                                choices=[
                                    ("Generate every combination of images and texts", 0),
                                    ("Match images and text prompts", 1),
                                ], visible=multiple_images_as_text_prompts and not edit_mode, label= "Multiple Images as Texts Prompts"
                            )
                        with gr.Row():
                            multi_prompts_gen_choices = [(f"Each New Line Will Add a new {medium} Request to the Generation Queue", "G")]
                            multi_prompts_gen_choices += [(f"Each new Paragraph separated by an Empty Line Will Add a new {medium} Request to the Generation Queue", "PG")]
                            if sliding_window_enabled:
                                multi_prompts_gen_choices += [("Each Line Will be used for a new Sliding Window of the same Video Generation", "W")]
                                multi_prompts_gen_choices += [("Each Paragraph Separated by an Empty line will be used for a new Sliding Window of the same Video Generation", "PW")]
                            multi_prompts_gen_choices += [("All the Lines are Part of the Same Prompt", "FG")]

                            multi_prompts_gen_type = gr.Dropdown(
                                choices=multi_prompts_gen_choices,
                                value=ui_get("multi_prompts_gen_type"),
                                visible=not edit_mode,
                                scale = 1,
                                label= "How to Process each Line of the Text Prompt"
                            )

                with gr.Tab("Loras", visible= not audio_only or model_def.get("enabled_audio_lora", False)) as loras_tab:
                    with gr.Column(visible = True): #as loras_column:
                        gr.Markdown("<B>Loras can be used to create special effects on the video by mentioning a trigger word in the Prompt. You can save Loras combinations in presets.</B>")
                        loras, loras_choices = get_updated_loras_dropdown(loras, launch_loras)
                        state_dict["loras"] = loras
                        loras_choices = gr.Dropdown(
                            choices=loras_choices,
                            value= launch_loras,
                            multiselect= True,
                            label="Activated Loras",
                            allow_custom_value= True,
                        )
                        loras_multipliers = gr.Textbox(label="Loras Multipliers (1.0 by default) separated by Space chars or CR, lines that start with # are ignored", value=launch_multis_str)
                with gr.Tab("Steps Skipping", visible = any_tea_cache or any_mag_cache) as speed_tab:
                    with gr.Column():
                        gr.Markdown("<B>Tea Cache and Mag Cache accelerate the Video Generation by skipping intelligently some steps, the more steps are skipped the lower the quality of the video.</B>")
                        gr.Markdown("<B>Steps Skipping  consumes also VRAM. It is recommended not to skip at least the first 10% steps.</B>")
                        steps_skipping_choices = [("None", "")]
                        if any_tea_cache: steps_skipping_choices += [("Tea Cache", "tea")]
                        if any_mag_cache: steps_skipping_choices += [("Mag Cache", "mag")]
                        skip_steps_cache_type = gr.Dropdown(
                            choices= steps_skipping_choices,
                            value="" if not (any_tea_cache or any_mag_cache) else ui_get("skip_steps_cache_type"),
                            visible=True,
                            label="Skip Steps Cache Type"
                        )
 
                        skip_steps_multiplier = gr.Dropdown(
                            choices=[
                                ("around x1.5 speed up", 1.5), 
                                ("around x1.75 speed up", 1.75), 
                                ("around x2 speed up", 2.0), 
                                ("around x2.25 speed up", 2.25), 
                                ("around x2.5 speed up", 2.5), 
                            ],
                            value=float(ui_get("skip_steps_multiplier")),
                            visible=True,
                            label="Skip Steps Cache Global Acceleration"
                        )
                        skip_steps_start_step_perc = gr.Slider(0, 100, value=ui_get("skip_steps_start_step_perc"), step=1, label="Skip Steps starting moment in % of generation", show_reset_button= False) 

                with gr.Tab("Post Processing", visible = not audio_only) as post_processing_tab:
                    

                    with gr.Column():
                        gr.Markdown("<B>Upsampling - postprocessing that may improve fluidity and the size of the video</B>")
                        def gen_upsampling_dropdowns(temporal_upsampling, spatial_upsampling , film_grain_intensity, film_grain_saturation, element_class= None, max_height= None, image_outputs = False, any_vae_upsampling = False):
                            temporal_upsampling = gr.Dropdown(
                                choices=[
                                    ("Disabled", ""),
                                    ("Rife x2 frames/s", "rife2"), 
                                    ("Rife x4 frames/s", "rife4"), 
                                ],
                                value=temporal_upsampling,
                                visible=not image_outputs,
                                scale = 1,
                                label="Temporal Upsampling",
                                elem_classes= element_class
                                # max_height = max_height
                            )
                            
                            spatial_upsampling = gr.Dropdown(
                                choices=[
                                    ("Disabled", ""),
                                    ("Lanczos x1.5", "lanczos1.5"), 
                                    ("Lanczos x2.0", "lanczos2"), 
                                ] + ([("VAE x1.0 (refined)", "vae1"),("VAE x2.0", "vae2")] if any_vae_upsampling else []) ,
                                value=spatial_upsampling,
                                visible=True,
                                scale = 1,
                                label="Spatial Upsampling",
                                elem_classes= element_class
                                # max_height = max_height
                            )

                            with gr.Row():
                                film_grain_intensity = gr.Slider(0, 1, value=film_grain_intensity, step=0.01, label="Film Grain Intensity (0 = disabled)", show_reset_button= False) 
                                film_grain_saturation = gr.Slider(0.0, 1, value=film_grain_saturation, step=0.01, label="Film Grain Saturation", show_reset_button= False) 

                            return temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation
                        temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation = gen_upsampling_dropdowns(ui_get("temporal_upsampling"), ui_get("spatial_upsampling"), ui_get("film_grain_intensity"), ui_get("film_grain_saturation"), image_outputs= image_outputs, any_vae_upsampling= "vae_upsampler"in model_def)

                with gr.Tab("Audio", visible = not (image_outputs or audio_only)) as audio_tab:
                    any_audio_source = not (image_outputs or audio_only)
                    with gr.Column(visible = get_mmaudio_settings(server_config)[0]) as mmaudio_col:
                        gr.Markdown("<B>Add a soundtrack based on the content of the Generated Video</B>")
                        with gr.Row():
                            MMAudio_setting = gr.Dropdown(
                                choices=[("Disabled", 0),  ("Enabled", 1), ],
                                value=ui_get("MMAudio_setting"), visible=True, scale = 1, label="MMAudio",
                            )
                            # if MMAudio_seed != None:
                            #     MMAudio_seed = gr.Slider(-1, 999999999, value=MMAudio_seed, step=1, scale=3, label="Seed (-1 for random)") 
                        with gr.Row():
                            MMAudio_prompt = gr.Text(ui_get("MMAudio_prompt"), label="Prompt (1 or 2 keywords)")
                            MMAudio_neg_prompt = gr.Text(ui_get("MMAudio_neg_prompt"), label="Negative Prompt (1 or 2 keywords)")
                            

                    with gr.Column(visible = any_control_video) as audio_prompt_type_remux_row:
                        gr.Markdown("<B>You may transfer the existing audio tracks of a Control Video</B>")
                        audio_prompt_type_remux = gr.Dropdown(
                            choices=[
                                ("No Remux", ""),
                                ("Remux Audio Files from Control Video if any and if no MMAudio / Custom Soundtrack", "R"),
                            ],
                            value=filter_letters(audio_prompt_type_value, "R"),
                            label="Remux Audio Files",
                            visible = True
                        )

                    with gr.Column():
                        gr.Markdown("<B>Add Custom Soundtrack to Video</B>")
                        audio_source = gr.Audio(value= ui_defaults.get("audio_source", None), type="filepath", label="Soundtrack", show_download_button= True)
                        
                any_perturbation = model_def.get("perturbation", False)
                any_cfg_zero = model_def.get("cfg_zero", False)
                any_cfg_star = model_def.get("cfg_star", False)
                any_apg = model_def.get("adaptive_projected_guidance", False)
                any_motion_amplitude = model_def.get("motion_amplitude", False) and not image_outputs
                any_pnp = True # Enable PnP for all supported models (or restriction logic here)

                with gr.Tab("Quality", visible = (vace and image_outputs or any_perturbation or any_cfg_zero or any_cfg_star or any_apg or any_motion_amplitude or any_pnp) and not audio_only ) as quality_tab:
                        with gr.Column(visible = any_perturbation ) as perturbation_row:
                            gr.Markdown("<B>Perturbation (improves video quality, requires guidance > 1)</B>")
                            perturbation_choices = model_def.get("perturbation_choices", [("OFF", 0), ("Skip Layer Guidance", 1)])
                            perturbation_value = ui_defaults["perturbation_switch"] = get_default_value(perturbation_choices, ui_get("perturbation_switch"), 0)
                            perturbation_layers_max = model_def.get("perturbation_layers_max", 40)
                            with gr.Row():
                                perturbation_switch = gr.Dropdown(
                                    choices=perturbation_choices,
                                    value=perturbation_value,
                                    visible=True,
                                    scale = 1,
                                    label="Perturbation"
                                )
                                perturbation_layers = gr.Dropdown(
                                    choices=[
                                        (str(i), i ) for i in range(perturbation_layers_max)
                                    ],
                                    value=ui_get("perturbation_layers"),
                                    multiselect= True,
                                    label="Perturbation Layers",
                                    scale= 3
                                )
                            with gr.Row():
                                perturbation_start_perc = gr.Slider(0, 100, value=ui_get("perturbation_start_perc"), step=1, label="Denoising Steps % start", show_reset_button= False)
                                perturbation_end_perc = gr.Slider(0, 100, value=ui_get("perturbation_end_perc"), step=1, label="Denoising Steps % end", show_reset_button= False)

                        with gr.Column(visible= any_apg ) as apg_col:
                            gr.Markdown("<B>Correct Progressive Color Saturation during long Video Generations")
                            apg_switch = gr.Dropdown(
                                choices=[
                                    ("OFF", 0),
                                    ("ON", 1), 
                                ],
                                value=ui_get("apg_switch"),
                                visible=True,
                                scale = 1,
                                label="Adaptive Projected Guidance (requires Guidance > 1 or Audio Guidance > 1) " if multitalk else "Adaptive Projected Guidance (requires Guidance > 1)",
                            )

                        with gr.Column(visible = any_cfg_star) as cfg_free_guidance_col:
                            gr.Markdown("<B>Classifier-Free Guidance Zero Star, better adherence to Text Prompt")
                            cfg_star_switch = gr.Dropdown(
                                choices=[
                                    ("OFF", 0),
                                    ("ON", 1), 
                                ],
                                value=ui_get("cfg_star_switch"),
                                visible=True,
                                scale = 1,
                                label="Classifier-Free Guidance Star (requires Guidance > 1)"
                            )
                            with gr.Row():
                                cfg_zero_step = gr.Slider(-1, 39, value=ui_get("cfg_zero_step"), step=1, label="CFG Zero below this Layer (Extra Process)", visible = any_cfg_zero, show_reset_button= False) 

                        with gr.Column(visible = v2i_switch_supported and image_outputs) as min_frames_if_references_col:
                            gr.Markdown("<B>Generating a single Frame alone may not be sufficient to preserve Reference Image Identity / Control Image Information or simply to get a good Image Quality. A workaround is to generate a short Video and keep the First Frame.")
                            min_frames_if_references = gr.Dropdown(
                                choices=[
                                    ("Disabled, generate only one Frame", 1),
                                    ("Generate a 5 Frames long Video only if any Reference Image / Control Image (x1.5 slower)",5),
                                    ("Generate a 9 Frames long Video only if any Reference Image / Control Image (x2.0 slower)",9),
                                    ("Generate a 13 Frames long Video only if any Reference Image / Control Image (x2.5 slower)",13),
                                    ("Generate a 17 Frames long Video only if any Reference Image / Control Image (x3.0 slower)",17),
                                    ("Generate always a 5 Frames long Video (x1.5 slower)",1005),
                                    ("Generate always a 9 Frames long Video (x2.0 slower)",1009),
                                    ("Generate always a 13 Frames long Video (x2.5 slower)",1013),
                                    ("Generate always a 17 Frames long Video (x3.0 slower)",1017),
                                ],
                                value=ui_get("min_frames_if_references",9 if vace else 1),
                                visible=True,
                                scale = 1,
                                label="Generate more frames to preserve Reference Image Identity / Control Image Information or improve"
                            )

                        with gr.Column(visible = get_container_def("motion_amplitude_col").visible and not image_outputs) as motion_amplitude_col:
                            gr.Markdown("<B>Experimental: Accelerate Motion (1: disabled, 1.15 recommended)")
                            motion_amplitude  = setting_slider("motion_amplitude")

                        with gr.Column(visible = model_def.get("self_refiner", False)) as self_refiner_col:
                            gr.Markdown("<B>Self-Refining Video Sampling (PnP) - should improve quality of Motion</B>")
                            self_refiner_setting = gr.Dropdown(choices=[("Disabled", 0),("Enabled with P1-Norm", 1), ("Enabled with P2-Norm", 2)], value=ui_get("self_refiner_setting", 0), scale=1, label="Self Refiner")
                            
                            refiner_val = ensure_refiner_list(ui_get("self_refiner_plan", []))
                            self_refiner_plan = refiner_val if update_form else gr.State(value=refiner_val)
                            
                            with gr.Column(visible=(update_form and ui_get("self_refiner_setting", 0) > 0)) as self_refiner_rules_ui:
                                gr.Markdown("#### Refiner Rules")
                                
                                with gr.Row(elem_id="refiner-input-row"):
                                    if RangeSlider is not None:
                                        refiner_range = RangeSlider(minimum=1, maximum=100, value=(1, 10), step=1, label="Step Range", info="Start - End", scale=3)
                                    else:
                                        refiner_range = gr.Textbox(value="1-10", label="Step Range", info="Start-End", scale=3)
                                    refiner_mult = gr.Slider(label="Iterations", value=3, minimum=1, maximum=5, step=1, scale=2)
                                    refiner_add_btn = gr.Button("➕ Add", variant="primary", scale=0, min_width=100)
                                
                                if not update_form:
                                    refiner_add_btn.click(fn=add_refiner_rule, inputs=[self_refiner_plan, refiner_range, refiner_mult], outputs=[self_refiner_plan])
                                    self_refiner_setting.change(fn=lambda s: gr.update(visible=s > 0), inputs=[self_refiner_setting], outputs=[self_refiner_rules_ui])

                                    @gr.render(inputs=self_refiner_plan)
                                    def render_refiner_rules(rules):
                                        if not rules:
                                            gr.Markdown("<I style='padding: 8px;'>No rules defined. Using defaults: Steps 2-5 (3x), Steps 6-13 (1x).</I>")
                                            return
                                        for i, rule in enumerate(rules):
                                            with gr.Row(elem_classes="rule-row"):
                                                text_display = f"Steps **{rule['start']} - {rule['end']}** : **{rule['steps']}x** iterations"
                                                gr.Markdown(text_display, elem_classes="rule-card")
                                                gr.Button("✖", variant="stop", scale=0, elem_classes="delete-btn").click(
                                                    fn=remove_refiner_rule, 
                                                    inputs=[self_refiner_plan, gr.State(i)], 
                                                    outputs=[self_refiner_plan]
                                                )
                                                
                            with gr.Row():
                                self_refiner_f_uncertainty = gr.Slider(0.0, 1.0, value=ui_get("self_refiner_f_uncertainty", 0.0), step=0.01, label="Uncertainty Threshold", show_reset_button= False)
                                self_refiner_certain_percentage = gr.Slider(0.0, 1.0, value=ui_get("self_refiner_certain_percentage", 0.999), step=0.001, label="Certainty Percentage Skip", show_reset_button= False)
                            

                with gr.Tab("Sliding Window", visible= sliding_window_enabled and not image_outputs and not audio_only) as sliding_window_tab:

                    with gr.Column():  
                        gr.Markdown("<B>A Sliding Window allows you to generate video with a duration not limited by the Model</B>")
                        gr.Markdown("<B>It is automatically turned on if the number of frames to generate is higher than the Window Size</B>")
                        if diffusion_forcing:
                            sliding_window_size = setting_slider("sliding_window_size", minimum=37, maximum=get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=20, label="  (recommended to keep it at 97)", visible=True, respect_setting_visibility=False)
                            sliding_window_overlap = setting_slider("sliding_window_overlap", minimum=17, maximum=97, value=ui_defaults.get("sliding_window_overlap", 17), step=20, visible=True, respect_setting_visibility=False)
                            sliding_window_color_correction_strength = setting_slider("sliding_window_color_correction_strength", value=0, visible=False)
                            sliding_window_overlap_noise = setting_slider("sliding_window_overlap_noise", value=ui_defaults.get("sliding_window_overlap_noise", 20), maximum=100, visible=True)
                            sliding_window_discard_last_frames = setting_slider("sliding_window_discard_last_frames", value=ui_defaults.get("sliding_window_discard_last_frames", 0), visible=False)
                        elif ltxv:
                            sliding_window_size = setting_slider("sliding_window_size", minimum=41, maximum=get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=8, visible=True, respect_setting_visibility=False)
                            sliding_window_overlap = setting_slider("sliding_window_overlap", minimum=1, maximum=97, value=ui_defaults.get("sliding_window_overlap", 9), step=8, visible=True, respect_setting_visibility=False)
                            sliding_window_color_correction_strength = setting_slider("sliding_window_color_correction_strength", value=0, visible=False)
                            sliding_window_overlap_noise = setting_slider("sliding_window_overlap_noise", value=ui_defaults.get("sliding_window_overlap_noise", 20), maximum=100, visible=False)
                            sliding_window_discard_last_frames = setting_slider("sliding_window_discard_last_frames", value=ui_defaults.get("sliding_window_discard_last_frames", 0), step=8, visible=True)
                        elif hunyuan_video_custom_edit:
                            sliding_window_size = setting_slider("sliding_window_size", minimum=5, maximum=get_max_frames(257), value=ui_defaults.get("sliding_window_size", 129), step=4, visible=True, respect_setting_visibility=False)
                            sliding_window_overlap = setting_slider("sliding_window_overlap", minimum=1, maximum=97, value=ui_defaults.get("sliding_window_overlap", 5), step=4, visible=True, respect_setting_visibility=False)
                            sliding_window_color_correction_strength = setting_slider("sliding_window_color_correction_strength", value=0, visible=False)
                            sliding_window_overlap_noise = setting_slider("sliding_window_overlap_noise", value=ui_defaults.get("sliding_window_overlap_noise", 20), visible=False)
                            sliding_window_discard_last_frames = setting_slider("sliding_window_discard_last_frames", value=ui_defaults.get("sliding_window_discard_last_frames", 0), visible=True)
                        else: # Vace, Multitalk
                            sliding_window_defaults = model_def.get("sliding_window_defaults", {})                            
                            sliding_window_size = setting_slider("sliding_window_size", value=ui_get("sliding_window_size", sliding_window_defaults.get("window_default", 81)), interactive=not model_def.get("sliding_window_size_locked"))
                            sliding_window_overlap = setting_slider("sliding_window_overlap", value=ui_get("sliding_window_overlap",sliding_window_defaults.get("overlap_default", 5)))
                            sliding_window_color_correction_strength = setting_slider("sliding_window_color_correction_strength")
                            sliding_window_overlap_noise = setting_slider("sliding_window_overlap_noise", value=ui_get("sliding_window_overlap_noise",20 if vace else 0))
                            sliding_window_discard_last_frames = setting_slider("sliding_window_discard_last_frames")

                        video_prompt_type_alignment = gr.Dropdown(
                            choices=[
                                ("Aligned to the beginning of the Source Video", ""),
                                ("Aligned to the beginning of the First Window of the new Video Sample", "T"),
                            ],
                            value=filter_letters(video_prompt_type_value, "T"),
                            label="Control Video / Control Audio / Positioned Frames Temporal Alignment when any Video to continue",
                            visible = any_control_image or any_control_video or any_audio_guide or any_audio_guide2 or any_custom_guide 
                        )

                        
                with gr.Tab("Misc.") as misc_tab:
                    with gr.Column(visible = not (recammaster or ltxv or diffusion_forcing or audio_only or image_outputs)) as RIFLEx_setting_col:
                        gr.Markdown("<B>With Riflex you can generate videos longer than 5s which is the default duration of videos used to train the model</B>")
                        RIFLEx_setting = gr.Dropdown(
                            choices=[
                                ("Auto (ON if Video longer than 5s)", 0),
                                ("Always ON", 1), 
                                ("Always OFF", 2), 
                            ],
                            value=ui_get("RIFLEx_setting"),
                            label="RIFLEx positional embedding to generate long video",
                            visible = True
                        )
                    with gr.Column(visible = not (audio_only or image_outputs)) as force_fps_col:
                        gr.Markdown("<B>You can change the Default number of Frames Per Second of the output Video, in the absence of Control Video this may create unwanted slow down / acceleration</B>")
                        force_fps_choices =  [(f"Model Default ({fps} fps)", "")]
                        if any_control_video and (any_video_source or recammaster):
                            force_fps_choices +=  [("Auto fps: Source Video if any, or Control Video if any, or Model Default", "auto")]
                        elif any_control_video :
                            force_fps_choices +=  [("Auto fps: Control Video if any, or Model Default", "auto")]
                        elif any_control_video and (any_video_source or recammaster):
                            force_fps_choices +=  [("Auto fps: Source Video if any, or Model Default", "auto")]
                        if any_control_video:
                            force_fps_choices +=  [("Control Video fps", "control")]
                        if any_video_source or recammaster:
                            force_fps_choices +=  [("Source Video fps", "source")]
                        force_fps_choices += [
                                ("15", "15"), 
                                ("16", "16"), 
                                ("23", "23"), 
                                ("24", "24"), 
                                ("25", "25"), 
                                ("30", "30"), 
                                ("50", "50"), 
                            ]
                    
                        force_fps = gr.Dropdown(
                            choices=force_fps_choices,
                            value=ui_get("force_fps"),
                            label=f"Override Frames Per Second (model default={fps} fps)"
                        )


                    gr.Markdown("<B>You can set a more agressive Memory Profile if you generate only Short Videos or Images<B>")
                    override_profile = gr.Dropdown(
                        choices=[("Default Memory Profile", -1)] + memory_profile_choices,
                        value=ui_get("override_profile"),
                        label=f"Override Memory Profile"
                    )

                    gr.Markdown("<B>You can set a different Attention Mode to improve the quality / compatibility<B>")
                    override_attention = gr.Dropdown(
                        choices=[("Default Attention Mode", "")] + attention_modes_choices,
                        value=ui_get("override_attention"),
                        label=f"Override Attention Mode"
                    )

                    with gr.Column():
                        gr.Markdown('<B>Customize the Output Filename using Settings Values (<I>date, seed, resolution, num_inference_steps, prompt, flow_shift, video_length, guidance_scale</I>). For Instance:<BR>"<I>{date(YYYY-MM-DD_HH-mm-ss)}_{seed}_{prompt(50)}, {num_inference_steps}</I>"</B>')
                        output_filename = gr.Text( label= " Output Filename ( Leave Blank for Auto Naming)", value= ui_get("output_filename"))

            if not update_form:
                with gr.Row(visible=(tab_id == 'edit')):
                    edit_btn = gr.Button("Apply Edits", elem_id="edit_tab_apply_button")
                    cancel_btn = gr.Button("Cancel", elem_id="edit_tab_cancel_button")
                    silent_cancel_btn = gr.Button("Silent Cancel", elem_id="silent_edit_tab_cancel_button", visible=False)
            with gr.Column(visible= not edit_mode):
                with gr.Row():
                    save_settings_btn = gr.Button("Set Settings as Default", visible = not args.lock_config)
                    export_settings_from_file_btn = gr.Button("Export Settings to File")
                    reset_settings_btn = gr.Button("Reset Settings")
                with gr.Row():
                    settings_file = gr.File(height=41,label="Load Settings From Video / Image / JSON")
                    settings_base64_output = gr.Text(interactive= False, visible=False, value = "")
                    settings_filename =  gr.Text(interactive= False, visible=False, value = "")
                with gr.Group():
                    with gr.Row():
                        lora_url = gr.Text(label ="Lora URL", placeholder= "Enter Lora URL", scale=4, show_label=False, elem_classes="compact_text" )
                        download_lora_btn = gr.Button("Download Lora", scale=1, min_width=10)

                assistant_ui = None
                assistant_launcher_host = None
                assistant_panel = None
                if tab_id == 'generate':
                    assistant_ui = deepy_gradio_ui.build_deepy_chat_ui(deepy_visible=_deepy.is_available())
                    assistant_launcher_host = assistant_ui.launcher_host
                    assistant_panel = assistant_ui.panel

            mode = gr.Text(value="", visible = False)

        with gr.Column(visible=(tab_id == 'generate')):
            if not update_form:
                state = default_state if default_state is not None else gr.State(state_dict)
                gen_status = gr.Text(interactive=False, label="Status", lines=1, max_lines=1, autoscroll=False)
                main_bridge_elem_ids = tab_id == 'generate'
                status_trigger = gr.Text(interactive= False, visible=False, elem_id="wangp_main_status_trigger" if main_bridge_elem_ids else None)
                load_queue_trigger = gr.Text(interactive= False, visible=False, elem_id="wangp_main_load_queue_trigger" if main_bridge_elem_ids else None)
                abort_client_id= gr.Text(interactive= False, visible=False, elem_id="wangp_main_abort_client_id" if main_bridge_elem_ids else None)
                default_files = []
                current_gallery_tab = gr.Number(0, visible=False)
                with gr.Tabs() as gallery_tabs:
                    with gr.Tab("Video / Images Gallery", id="video_images"):
                        output = gr.Gallery(value =default_files, label="Generated videos", preview= True, show_label=False, elem_id="gallery" , columns=[3], rows=[1], object_fit="contain", height=450, selected_index=0, interactive= False)
                    with gr.Tab("Audio Files Gallery", id="audio"):
                        output_audio = AudioGallery(audio_paths=[], max_thumbnails=999, height=40, update_only=update_form)
                        audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger = output_audio.get_state()
                output_trigger = gr.Text(interactive= False, visible=False, elem_id="wangp_main_output_trigger" if main_bridge_elem_ids else None)
                selected_video_time_input = gr.Text(interactive= False, visible=False, elem_id="selected_video_time_input" if main_bridge_elem_ids else None)
                refresh_models_trigger = gr.Text(interactive= False, visible=False)
                refresh_form_trigger = gr.Text(interactive= False, visible=False)
                fill_wizard_prompt_trigger = gr.Text(interactive= False, visible=False)
                save_form_trigger = gr.Text(interactive= False, visible=False)
                gallery_source = gr.Text(interactive= False, visible=False)
                model_choice_target = gr.Text(interactive= False, visible=False)


            with gr.Accordion("Media Info / Late Post Processing / Import Media", open=False) as video_info_accordion:
                with gr.Tabs() as video_info_tabs:
                    default_visibility_false = {} if update_form else {"visible" : False}                        
                    default_visibility_true = {} if update_form else {"visible" : True}                        

                    with gr.Tab("Information", id="video_info"):
                        video_info = gr.HTML(visible=True, min_height=100, value=get_default_video_info()) 
                        with gr.Row(**default_visibility_false) as audio_buttons_row:
                            video_info_extract_audio_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_audio_guide_btn = gr.Button("To Audio Source", min_width= 1, size ="sm", visible = any_audio_guide)
                            video_info_to_audio_guide2_btn = gr.Button("To Audio Source 2", min_width= 1, size ="sm", visible = any_audio_guide2 )
                            video_info_to_audio_source_btn = gr.Button("To Custom Audio", min_width= 1, size ="sm", visible = any_audio_source )
                            video_info_eject_audio_btn = gr.Button("Eject Audio File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false) as deleted_audio_buttons_row:
                            video_info_eject_deleted_audio_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false) as video_buttons_row:
                            video_info_extract_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_video_source_btn = gr.Button("To Video Source", min_width= 1, size ="sm", visible = any_video_source)
                            video_info_to_control_video_btn = gr.Button("To Control Video", min_width= 1, size ="sm", visible = any_control_video )
                            video_info_eject_video_btn = gr.Button("Eject Video", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false) as deleted_video_buttons_row:
                            video_info_eject_deleted_video_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false) as image_buttons_row:
                            video_info_extract_image_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_start_image_btn = gr.Button("To Start Image", size ="sm", min_width= 1, visible = any_start_image )
                            video_info_to_end_image_btn = gr.Button("To End Image", size ="sm", min_width= 1, visible = any_end_image)
                            video_info_to_image_mask_btn = gr.Button("To Mask Image", min_width= 1, size ="sm", visible = any_image_mask and False)
                            video_info_to_reference_image_btn = gr.Button("To Reference Image", min_width= 1, size ="sm", visible = any_reference_image)
                            video_info_to_image_guide_btn = gr.Button("To Control Image", min_width= 1, size ="sm", visible = any_control_image )
                            video_info_eject_image_btn = gr.Button("Eject Image", min_width= 1, size ="sm")
                    with gr.Tab("Post Processing", id= "post_processing", visible = True) as video_postprocessing_tab:
                        with gr.Group(elem_classes= "postprocess"):
                            with gr.Column():
                                PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation = gen_upsampling_dropdowns("",  "", 0, 0.5, element_class ="postprocess", image_outputs = False)
                        with gr.Row():
                            video_info_postprocessing_btn = gr.Button("Apply Postprocessing", size ="sm", visible=True)
                            video_info_eject_video2_btn = gr.Button("Eject Video", size ="sm", visible=True)
                    with gr.Tab("Audio Remuxing", id= "audio_remuxing", visible = True) as audio_remuxing_tab:

                        with gr.Group(elem_classes= "postprocess"):
                            with gr.Column(visible = get_mmaudio_settings(server_config)[0]) as PP_MMAudio_col:
                                with gr.Row():
                                    PP_MMAudio_setting = gr.Dropdown(
                                        choices=[("Add Custom Audio Sountrack", 0),  ("Use MMAudio to generate a Soundtrack based on the Video", 1), ],
                                        visible=True, scale = 1, label="MMAudio", show_label= False, elem_classes= "postprocess", **({} if update_form else {"value" : 0 })
                                    )
                                with gr.Column(**default_visibility_false) as PP_MMAudio_row:
                                    with gr.Row():
                                        PP_MMAudio_prompt = gr.Text("", label="Prompt (1 or 2 keywords)", elem_classes= "postprocess")
                                        PP_MMAudio_neg_prompt = gr.Text("", label="Negative Prompt (1 or 2 keywords)", elem_classes= "postprocess")
                                    PP_MMAudio_seed = gr.Slider(-1, 999999999, value=-1, step=1, label="Seed (-1 for random)", show_reset_button= False) 
                                    PP_repeat_generation = gr.Slider(1, 25.0, value=1, step=1, label="Number of Sample Videos to Generate", show_reset_button= False) 
                            with gr.Row(**default_visibility_true) as PP_custom_audio_row:
                                    PP_custom_audio = gr.Audio(label = "Soundtrack", type="filepath", show_download_button= True,)
                        with gr.Row():
                            video_info_remux_audio_btn = gr.Button("Remux Audio", size ="sm", visible=True)
                            video_info_eject_video3_btn = gr.Button("Eject Video", size ="sm", visible=True)
                    with gr.Tab("Import Media to Galleries", id= "video_add"):
                        files_to_load = gr.Files(label= "Media to Import in Galleries", height=120)
                        with gr.Row():
                            video_info_add_videos_btn = gr.Button("Import Videos / Images / Audio Files", size ="sm")
 
            if not update_form:
                generate_btn = gr.Button("Generate")
                add_to_queue_btn = gr.Button("Add New Prompt To Queue", visible=False)
                generate_trigger = gr.Text(visible = False) 
                add_to_queue_trigger = gr.Text(visible = False)
                js_trigger_index = gr.Text(visible=False, elem_id="js_trigger_for_edit_refresh")

                with gr.Column(visible= False) as current_gen_column:
                    with gr.Accordion("Preview", open=False):
                        preview = gr.HTML(label="Preview", show_label= False)
                        preview_trigger = gr.Text(visible= False)
                    gen_info = gr.HTML(visible=False, min_height=1) 
                    with gr.Row() as current_gen_buttons_row:
                        onemoresample_btn = gr.Button("One More Sample", visible = True, size='md', min_width=1)
                        onemorewindow_btn = gr.Button("Extend this Sample", visible = False, size='md', min_width=1)
                        pause_btn = gr.Button("Pause", visible = True, size='md', min_width=1)
                        resume_btn = gr.Button("Resume", visible = False, size='md', min_width=1)
                        abort_btn = gr.Button("Abort", visible = True, size='md', min_width=1)
                        earlystop_btn = gr.Button("Early Stop", visible = True, size='md', min_width=1)
                with gr.Accordion("Queue Management", open=False) as queue_accordion:
                    with gr.Row():
                        queue_html = gr.HTML(
                            value=generate_queue_html(state_dict["gen"]["queue"]),
                            elem_id="queue_html_container"
                        )
                    queue_action_input = gr.Text(elem_id="queue_action_input", visible=False)
                    queue_action_trigger = gr.Button(elem_id="queue_action_trigger", visible=False)
                    with gr.Row(visible= True):
                        queue_zip_base64_output = gr.Text(visible=False)
                        save_queue_btn = gr.DownloadButton("Save Queue", size="sm")
                        load_queue_btn = gr.UploadButton("Load Queue", file_types=[".zip", ".json"], size="sm")
                        clear_queue_btn = gr.Button("Clear Queue", size="sm", variant="stop")
                        quit_button = gr.Button("Save and Quit", size="sm", variant="secondary")
                        with gr.Row(visible=False) as quit_confirmation_row:
                            confirm_quit_button = gr.Button("Confirm", elem_id="comfirm_quit_btn_hidden", size="sm", variant="stop")
                            cancel_quit_button = gr.Button("Cancel", size="sm", variant="secondary")
                        hidden_force_quit_trigger = gr.Button("force_quit", visible=False, elem_id="force_quit_btn_hidden")
                        hidden_countdown_state = gr.Number(value=-1, visible=False, elem_id="hidden_countdown_state_num")
                        single_hidden_trigger_btn = gr.Button("trigger_countdown", visible=False, elem_id="trigger_info_single_btn")

        extra_inputs = prompt_vars + [wizard_prompt, wizard_variables_var, wizard_prompt_activated_var, video_prompt_column, image_prompt_column, image_prompt_type_group, image_prompt_type_radio, image_prompt_type_endcheckbox,
                                      prompt_column_advanced, prompt_column_wizard_vars, prompt_column_wizard, alt_prompt_row, lset_name, save_lset_prompt_drop, advanced_row, speed_tab, audio_tab, mmaudio_col, quality_tab,
                                      sliding_window_tab, misc_tab, prompt_enhancer_row, inference_steps_row, perturbation_row, audio_guide_row, custom_guide_row, RIFLEx_setting_col,
                                      video_prompt_type_video_guide, video_prompt_type_video_guide_alt, video_prompt_type_video_mask, video_prompt_type_image_refs, video_prompt_type_video_custom_dropbox, video_prompt_type_video_custom_checkbox,
                                      apg_col, audio_prompt_type_sources,  audio_prompt_type_remux, audio_prompt_type_remux_row, force_fps_col,
                                      video_guide_outpainting_col,video_guide_outpainting_top, video_guide_outpainting_bottom, video_guide_outpainting_left, video_guide_outpainting_right,
                                      video_guide_outpainting_checkbox, video_guide_outpainting_ratio, video_guide_outpainting_row, show_advanced, magic_mask_image_btn, magic_mask_video_btn, video_info_to_control_video_btn, video_info_to_video_source_btn, sample_solver_row,
                                      video_buttons_row, deleted_video_buttons_row, image_buttons_row, video_postprocessing_tab, audio_remuxing_tab, PP_MMAudio_col, PP_MMAudio_setting, PP_MMAudio_row, PP_custom_audio_row, 
                                      audio_buttons_row, deleted_audio_buttons_row, video_info_extract_audio_settings_btn, video_info_to_audio_guide_btn, video_info_to_audio_guide2_btn, video_info_to_audio_source_btn, video_info_eject_audio_btn,
                                      video_info_to_start_image_btn, video_info_to_end_image_btn, video_info_to_reference_image_btn, video_info_to_image_guide_btn, video_info_to_image_mask_btn,
                                      NAG_col, audio_options_row, remove_background_sound, continue_beyond_audio_end, normalize_audio_volumes, speakers_locations_row, embedded_guidance_row, guidance_phases_row, guidance_row, resolution_group, cfg_free_guidance_col, control_net_weights_row, guide_selection_row, image_mode_tabs, prompt_enhancer_mode_dropdown, prompt_enhancer_think,
                                      min_frames_if_references_col, motion_amplitude_col, video_prompt_type_alignment, prompt_enhancer_btn, tab_inpaint, tab_t2v, resolution_row, loras_tab, post_processing_tab, temperature_row, *custom_settings_rows, top_pk_row, 
                                      number_frames_row, negative_prompt_row,
                                      self_refiner_col, pause_row]+\
                                      image_start_extra + image_end_extra + image_refs_extra #  presets_column,
        if update_form:
            locals_dict = locals()
            locals_dict.update(custom_setting_components_map)
            gen_inputs = [state_dict if k=="state" else locals_dict[k]  for k in inputs_names] + [state_dict, plugin_data] + extra_inputs
            return gen_inputs
        else:
            target_state = gr.Text(value = "state", interactive= False, visible= False)
            target_settings = gr.Text(value = "settings", interactive= False, visible= False)
            last_choice = gr.Number(value =-1, interactive= False, visible= False)

            resolution_group.input(fn=change_resolution_group, inputs=[state, resolution_group], outputs=[resolution], show_progress="hidden")
            resolution.change(fn=record_last_resolution, inputs=[state, resolution])

            video_info_add_videos_btn.click(fn=add_videos_to_gallery, inputs =[state, output, last_choice, audio_files_paths, audio_file_selected, files_to_load], outputs = [gallery_tabs, current_gallery_tab, output, audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger, files_to_load, video_info_tabs, gallery_source] ).then(
                fn=select_video, inputs=[state, current_gallery_tab, output, last_choice, audio_files_paths, audio_file_selected, gallery_source], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, deleted_video_buttons_row, deleted_audio_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")
            gallery_tabs.select(fn=set_gallery_tab, inputs=[state], outputs=[current_gallery_tab, gallery_source]).then(
                fn=select_video, inputs=[state, current_gallery_tab, output, last_choice, audio_files_paths, audio_file_selected, gallery_source], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, deleted_video_buttons_row, deleted_audio_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")
            gr.on(triggers=[video_length.release, force_fps.change, video_guide.change, video_source.change], fn=refresh_video_length_label, inputs=[state, video_length, force_fps, video_guide, video_source] , outputs = video_length, trigger_mode="always_last", show_progress="hidden"  )
            guidance_phases.change(fn=change_guidance_phases, inputs= [state, guidance_phases], outputs =[model_switch_phase, guidance_phases_row, switch_threshold, switch_threshold2, guidance2_scale, guidance3_scale ])
            audio_prompt_type_remux.change(fn=refresh_audio_prompt_type_remux, inputs=[state, audio_prompt_type, audio_prompt_type_remux], outputs=[audio_prompt_type])
            remove_background_sound.change(fn=refresh_remove_background_sound, inputs=[state, audio_prompt_type, remove_background_sound], outputs=[audio_prompt_type])
            continue_beyond_audio_end.change(fn=refresh_continue_beyond_audio_end, inputs=[state, audio_prompt_type, continue_beyond_audio_end], outputs=[audio_prompt_type])
            normalize_audio_volumes.change(fn=refresh_normalize_audio_volumes, inputs=[state, audio_prompt_type, normalize_audio_volumes], outputs=[audio_prompt_type])
            audio_prompt_type_sources.change(fn=refresh_audio_prompt_type_sources, inputs=[state, audio_prompt_type, audio_prompt_type_sources], outputs=[audio_prompt_type, audio_guide, audio_guide2, speakers_locations_row, remove_background_sound, normalize_audio_volumes, audio_options_row, audio_guide_row])
            prompt_enhancer_mode_dropdown.input(fn=build_prompt_enhancer_value, inputs=[prompt_enhancer_mode_dropdown, prompt_enhancer_think], outputs=[prompt_enhancer], show_progress="hidden")
            prompt_enhancer_think.input(fn=build_prompt_enhancer_value, inputs=[prompt_enhancer_mode_dropdown, prompt_enhancer_think], outputs=[prompt_enhancer], show_progress="hidden")
            image_prompt_type_radio.change(fn=refresh_image_prompt_type_radio, inputs=[state, image_prompt_type, image_prompt_type_radio, video_prompt_type], outputs=[image_prompt_type, image_start_row, image_end_row, video_source, input_video_strength, keep_frames_video_source, image_prompt_type_endcheckbox], show_progress="hidden" ) 
            image_prompt_type_endcheckbox.change(fn=refresh_image_prompt_type_endcheckbox, inputs=[state, image_prompt_type, image_prompt_type_radio, image_prompt_type_endcheckbox, video_prompt_type], outputs=[image_prompt_type, image_end_row, input_video_strength] ) 
            video_prompt_type_image_refs.input(fn=refresh_video_prompt_type_image_refs, inputs = [state, video_prompt_type, video_prompt_type_image_refs,image_mode, image_prompt_type], outputs = [video_prompt_type, image_refs_row, remove_background_images_ref,  image_refs_relative_size, frames_positions,video_guide_outpainting_col, input_video_strength], show_progress="hidden")
            video_prompt_type_video_guide.input(fn=refresh_video_prompt_type_video_guide,     inputs = [state, gr.State(""),   video_prompt_type, video_prompt_type_video_guide,     image_mode, image_mask_guide, image_guide, image_mask, image_prompt_type], outputs = [video_prompt_type, video_guide, image_guide, keep_frames_video_guide, denoising_strength, masking_strength,  video_guide_outpainting_col, video_prompt_type_video_mask, video_mask, image_mask, image_mask_guide, mask_expand, image_refs_row, frames_positions, video_prompt_type_video_custom_dropbox, video_prompt_type_video_custom_checkbox, input_video_strength, magic_mask_image_btn, magic_mask_video_btn], show_progress="hidden")
            video_prompt_type_video_guide_alt.input(fn=refresh_video_prompt_type_video_guide, inputs = [state, gr.State("alt"),video_prompt_type, video_prompt_type_video_guide_alt, image_mode, image_mask_guide, image_guide, image_mask, image_prompt_type], outputs = [video_prompt_type, video_guide, image_guide, keep_frames_video_guide, denoising_strength, masking_strength, video_guide_outpainting_col, video_prompt_type_video_mask, video_mask, image_mask, image_mask_guide, mask_expand, image_refs_row, frames_positions, video_prompt_type_video_custom_dropbox, video_prompt_type_video_custom_checkbox, input_video_strength, magic_mask_image_btn, magic_mask_video_btn], show_progress="hidden")
            # video_prompt_type_video_guide_alt.input(fn=refresh_video_prompt_type_video_guide_alt, inputs = [state, video_prompt_type, video_prompt_type_video_guide_alt, image_mode, image_mask_guide, image_guide, image_mask], outputs = [video_prompt_type, video_guide, image_guide, image_refs_row, denoising_strength, masking_strength, video_mask, mask_expand, image_mask_guide, image_guide, image_mask, keep_frames_video_guide ], show_progress="hidden")
            video_prompt_type_video_custom_dropbox.input(fn= refresh_video_prompt_type_video_custom_dropbox, inputs=[state, video_prompt_type, video_prompt_type_video_custom_dropbox], outputs = video_prompt_type)
            video_prompt_type_video_custom_checkbox.input(fn= refresh_video_prompt_type_video_custom_checkbox, inputs=[state, video_prompt_type, video_prompt_type_video_custom_checkbox], outputs = video_prompt_type)
            # image_mask_guide.upload(fn=update_image_mask_guide, inputs=[state, image_mask_guide], outputs=[image_mask_guide], show_progress="hidden")
            video_prompt_type_video_mask.input(fn=refresh_video_prompt_type_video_mask, inputs = [state, video_prompt_type, video_prompt_type_video_mask, image_mode, image_mask_guide, image_guide, image_mask], outputs = [video_prompt_type, video_mask, image_mask_guide, image_guide, image_mask, mask_expand, masking_strength, magic_mask_image_btn, magic_mask_video_btn], show_progress="hidden")
            video_prompt_type_alignment.input(fn=refresh_video_prompt_type_alignment, inputs = [state, video_prompt_type, video_prompt_type_alignment], outputs = [video_prompt_type])
            multi_prompts_gen_type.select(fn=refresh_prompt_labels, inputs=[state, multi_prompts_gen_type, image_mode], outputs=[prompt, wizard_prompt, image_end], show_progress="hidden")
            video_guide_outpainting_top.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_top, gr.State(0)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_bottom.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_bottom,gr.State(1)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_left.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_left,gr.State(2)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_right.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_right,gr.State(3)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_ratio.input(fn=refresh_video_guide_outpainting_labels, inputs=[video_guide_outpainting_ratio], outputs=[video_guide_outpainting_top, video_guide_outpainting_bottom, video_guide_outpainting_left, video_guide_outpainting_right], show_progress="hidden")
            video_guide_outpainting_checkbox.input(fn=refresh_video_guide_outpainting_row, inputs=[video_guide_outpainting_checkbox, video_guide_outpainting], outputs= [video_guide_outpainting_row, video_guide_outpainting_ratio, video_guide_outpainting])
            show_advanced.change(fn=switch_advanced, inputs=[state, show_advanced, lset_name], outputs=[advanced_row, preset_buttons_rows, refresh_lora_btn, refresh2_row ,lset_name]).then(
                fn=switch_prompt_type, inputs = [state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars], outputs = [wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars])
            gr.on( triggers=[output.change, output.select],fn=select_video, inputs=[state, current_gallery_tab, output, last_choice, audio_files_paths, audio_file_selected, gr.State("video")], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, deleted_video_buttons_row, deleted_audio_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")
            # gr.on( triggers=[output.change, output.select], fn=select_video, inputs=[state, output, last_choice, audio_files_paths, audio_file_selected, gr.State("video")], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")
            audio_file_selected.change(fn=select_video, inputs=[state, current_gallery_tab, output, last_choice, audio_files_paths, audio_file_selected, gr.State("audio")], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, deleted_video_buttons_row, deleted_audio_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")

            preview_trigger.change(refresh_preview, inputs= [state], outputs= [preview], show_progress="hidden")
            PP_MMAudio_setting.change(fn = lambda value : [gr.update(visible = value == 1), gr.update(visible = value == 0)] , inputs = [PP_MMAudio_setting], outputs = [PP_MMAudio_row, PP_custom_audio_row] )
            download_lora_btn.click(fn=download_lora, inputs = [state, lora_url], outputs = [lora_url]).then(fn=refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
            def refresh_status_async(state, progress=gr.Progress()):
                gen = get_gen_info(state)
                gen["progress"] = progress

                while True: 
                    progress_args= gen.get("progress_args", None)
                    if progress_args != None:
                        progress(*progress_args)
                        gen["progress_args"] = None
                    status= gen.get("status","")
                    if status is not None and len(status) > 0:
                        yield status
                        gen["status"]= ""
                    if not gen.get("status_display", False):
                        return
                    time.sleep(0.5)

            def activate_status(state):
                if state.get("validate_success",0) != 1:
                    return
                gen = get_gen_info(state)
                gen["status_display"] = True
                return time.time()

            start_quit_timer_js, cancel_quit_timer_js, trigger_zip_download_js, trigger_settings_download_js, click_brush_js = get_js()

            status_trigger.change(refresh_status_async, inputs= [state] , outputs= [gen_status], show_progress_on= [gen_status])

            if tab_id == 'generate':
                output_trigger.change(refresh_gallery,
                    inputs = [state], 
                    outputs = [gallery_tabs, current_gallery_tab, output, last_choice, audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger, gen_info, generate_btn, add_to_queue_btn, current_gen_column, current_gen_buttons_row, queue_html, abort_btn, earlystop_btn, onemorewindow_btn],
                    show_progress="hidden"
                    )


            modal_action_trigger.click(
                fn=show_modal_image,
                inputs=[state, modal_action_input],
                outputs=[modal_html_display, modal_container],
                show_progress="hidden"
            )
            close_modal_trigger_btn.click(
                fn=lambda: gr.Column(visible=False),
                outputs=[modal_container],
                show_progress="hidden"
            )
            for magic_mask_ui in magic_mask_uis:
                magic_mask_ui.mount(
                    state=state,
                    image_mode=image_mode,
                    video_guide=video_guide,
                    image_mask_guide=image_mask_guide,
                    image_guide=image_guide,
                    image_mask=image_mask,
                    video_mask=video_mask,
                    download_assets=lambda download_def: process_files_def(**download_def),
                    acquire_gpu=lambda state_value, process_id, process_name: acquire_GPU_ressources(state_value, process_id, process_name, gr=gr),
                    release_gpu=release_GPU_ressources,
                    get_model_settings=get_current_model_settings,
                )
            pause_btn.click(pause_generation, [state], [ pause_btn, resume_btn] )
            resume_btn.click(resume_generation, [state] )
            abort_btn.click(abort_generation, [state, gr.State("")], [ abort_btn, queue_html], show_progress="hidden" ) #.then(refresh_gallery, inputs = [state, gen_info], outputs = [output, gen_info, queue_html] )
            abort_client_id.change(abort_generation, [state, abort_client_id], [ abort_btn, queue_html], show_progress="hidden" ) #.then(refresh_gallery, inputs = [state, gen_info], outputs = [output, gen_info, queue_html] )
            earlystop_btn.click(early_stop_generation, [state], [ earlystop_btn] )
            onemoresample_btn.click(fn=one_more_sample,inputs=[state], outputs= [state])
            onemorewindow_btn.click(fn=one_more_window,inputs=[state], outputs= [state])

            inputs_names= list(inspect.signature(save_inputs).parameters)[1:-2]
            locals_dict = locals()
            locals_dict.update(custom_setting_components_map)
            gen_inputs = [locals_dict[k] for k in inputs_names] + [state, plugin_data]
            save_settings_btn.click( fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
                save_inputs, inputs =[target_settings] + gen_inputs, outputs = [])

            gr.on( triggers=[video_info_extract_settings_btn.click, video_info_extract_image_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, output, last_choice, gr.State("video")] , outputs= [model_family, model_base_type_choice, model_choice, refresh_form_trigger])

            gr.on( triggers=[video_info_extract_audio_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                show_progress="hidden",
                outputs= [prompt],
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, audio_files_paths, audio_file_selected, gr.State("audio")] , outputs= [model_family, model_base_type_choice, model_choice, refresh_form_trigger])


            prompt_enhancer_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=enhance_prompt, inputs =[state, prompt, prompt_enhancer, multi_images_gen_type, multi_prompts_gen_type, override_profile, video_prompt_type, image_prompt_type, audio_prompt_type ] , outputs= [prompt, wizard_prompt])

            # save_form_trigger.change(fn=validate_wizard_prompt,
            def set_save_form_event(trigger):
                return trigger(fn=validate_wizard_prompt,
                    inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                    outputs= [prompt],
                    show_progress="hidden",
                ).then(fn=save_inputs,
                    inputs =[target_state] + gen_inputs,
                    outputs= None
                )
            
            set_save_form_event(save_form_trigger.change)
            if assistant_ui is not None:
                deepy_gradio_ui.bind_deepy_chat_ui(
                    assistant_ui,
                    state=state,
                    output=output,
                    last_choice=last_choice,
                    audio_files_paths=audio_files_paths,
                    audio_file_selected=audio_file_selected,
                    selected_video_time_input=selected_video_time_input,
                    load_queue_trigger=load_queue_trigger,
                    output_trigger=output_trigger,
                    abort_client_id=abort_client_id,
                    handlers=deepy_gradio_ui.DeepyChatHandlers(
                        prepare_request_context=init_generate,
                        update_tool_ui_settings=_deepy.update_tool_ui_settings,
                        store_selected_video_time=_deepy.store_selected_video_time,
                        ask_ai=_deepy.ask_ai,
                        enqueue_ai=_deepy.enqueue_ai_while_busy,
                        stop_ai=_deepy.stop_ai,
                        reset_ai=_deepy.reset_ai,
                    ),
                )
                main.load(
                    fn=_deepy.browser_session_started,
                    inputs=[state],
                    outputs=[assistant_ui.chat_event, load_queue_trigger, assistant_ui.request, abort_client_id],
                    queue=False,
                    show_progress="hidden",
                )
            gr.on(triggers=[video_info_eject_video_btn.click, video_info_eject_video2_btn.click, video_info_eject_video3_btn.click, video_info_eject_deleted_video_btn.click,  video_info_eject_image_btn.click], fn=eject_video_from_gallery, inputs =[state, output, last_choice], outputs = [output, video_info, video_buttons_row] )
            video_info_to_control_video_btn.click(fn=video_to_control_video, inputs =[state, output, last_choice], outputs = [video_guide] )            
            video_info_to_video_source_btn.click(fn=video_to_source_video, inputs =[state, output, last_choice], outputs = [video_source] )
            video_info_to_start_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_start, gr.State("Start Image")], outputs = [image_start] )
            video_info_to_end_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_end, gr.State("End Image")], outputs = [image_end] )
            video_info_to_image_guide_btn.click(fn=image_to_ref_image_guide, inputs =[state, output, last_choice], outputs = [image_guide, image_mask_guide]).then(fn=None, inputs=[], outputs=[], js=click_brush_js )
            video_info_to_image_mask_btn.click(fn=image_to_ref_image_set, inputs =[state, output, last_choice, image_mask, gr.State("Image Mask")], outputs = [image_mask] )
            video_info_to_reference_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_refs, gr.State("Ref Image")],  outputs = [image_refs] )

            gr.on(triggers=[video_info_eject_audio_btn.click, video_info_eject_deleted_audio_btn.click], fn=eject_audio_from_gallery, inputs =[state, audio_files_paths, audio_file_selected], outputs = [audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger, video_info, audio_buttons_row] )
            video_info_to_audio_guide_btn.click(fn=audio_to_source_set, inputs =[state, audio_files_paths, audio_file_selected, gr.State("Audio Source")], outputs = [audio_guide] )
            video_info_to_audio_guide2_btn.click(fn=audio_to_source_set, inputs =[state, audio_files_paths, audio_file_selected, gr.State("Audio Source 2")], outputs = [audio_guide] )
            video_info_to_audio_source_btn.click(fn=audio_to_source_set, inputs =[state, audio_files_paths, audio_file_selected, gr.State("Custom Audio")], outputs = [audio_source] )

            video_info_postprocessing_btn.click(fn=apply_post_processing, inputs =[state, output, last_choice, PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
            video_info_remux_audio_btn.click(fn=remux_audio, inputs =[state, output, last_choice, PP_MMAudio_setting, PP_MMAudio_prompt, PP_MMAudio_neg_prompt, PP_MMAudio_seed, PP_repeat_generation, PP_custom_audio], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
            save_lset_btn.click(validate_save_lset, inputs=[state, lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
            delete_lset_btn.click(validate_delete_lset, inputs=[state, lset_name], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
            confirm_save_lset_btn.click(fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars] , outputs= [prompt], show_progress="hidden",).then(
                fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None).then(
                fn=save_lset, inputs=[state, lset_name, loras_choices, loras_multipliers, prompt, save_lset_prompt_drop], outputs=[lset_name, apply_lset_btn,refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_save_lset_btn, cancel_lset_btn, save_lset_prompt_drop])
            confirm_delete_lset_btn.click(delete_lset, inputs=[state, lset_name], outputs=[lset_name, apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn,confirm_delete_lset_btn, cancel_lset_btn ])
            cancel_lset_btn.click(cancel_lset, inputs=[], outputs=[apply_lset_btn, refresh_lora_btn, delete_lset_btn, save_lset_btn, confirm_delete_lset_btn,confirm_save_lset_btn, cancel_lset_btn,save_lset_prompt_drop ])
            apply_lset_btn.click(fn=save_inputs, inputs =[target_state] + gen_inputs, outputs= None).then(fn=apply_lset, 
                inputs=[state, wizard_prompt_activated_var, lset_name,loras_choices, loras_multipliers, prompt], outputs=[wizard_prompt_activated_var, loras_choices, loras_multipliers, prompt, fill_wizard_prompt_trigger, model_family, model_base_type_choice, model_choice, refresh_form_trigger])
            refresh_lora_btn.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])
            refresh_lora_btn2.click(refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices])

            lset_name.select(fn=update_lset_type, inputs=[state, lset_name], outputs=save_lset_prompt_drop)
            export_settings_from_file_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=export_settings, 
                inputs =[state], 
                outputs= [settings_base64_output, settings_filename]
            ).then(
                fn=None,
                inputs=[settings_base64_output, settings_filename],
                outputs=None,
                js=trigger_settings_download_js
            )
            
            image_mode_tabs.select(fn=record_image_mode_tab, inputs=[state], outputs= None
            ).then(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=switch_image_mode, inputs =[state] , outputs= [refresh_form_trigger], trigger_mode="multiple")

            settings_file.upload(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=load_settings_from_file, inputs =[state, settings_file] , outputs= [model_family, model_base_type_choice, model_choice, refresh_form_trigger, settings_file])


            model_choice_target.change(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=goto_model_type, inputs =[state, model_choice_target] , outputs= [model_family, model_base_type_choice, model_choice, refresh_form_trigger])

            reset_settings_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=reset_settings, inputs =[state] , outputs= [refresh_form_trigger])

 

            fill_wizard_prompt_trigger.change(
                fn = fill_wizard_prompt, inputs = [state, wizard_prompt_activated_var, prompt, wizard_prompt], outputs = [ wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars]
            )

            refresh_form_trigger.change(fn= fill_inputs, 
                inputs=[state],
                outputs=gen_inputs + extra_inputs,
                show_progress= "full" if args.debug_gen_form else "hidden",
            ).then(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars],
                outputs= [prompt],
                show_progress="hidden",
            )                

            if tab_id == 'generate':
                # main_tabs.select(fn=detect_auto_save_form, inputs= [state], outputs= save_form_trigger, trigger_mode="multiple")
                model_family.input(fn=change_model_family, inputs=[state, model_family], outputs= [model_base_type_choice, model_choice], show_progress="hidden")
                model_base_type_choice.input(fn=change_model_base_types, inputs=[state, model_family, model_base_type_choice], outputs= [model_base_type_choice, model_choice], show_progress="hidden")
                refresh_models_trigger.change(fn=refresh_model_dropdowns, inputs=[state], outputs=[model_family, model_base_type_choice, model_choice, refresh_form_trigger], show_progress="hidden")

                model_choice.change(fn=validate_wizard_prompt,
                    inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                    outputs= [prompt],
                    show_progress="hidden",
                ).then(fn=save_inputs,
                    inputs =[target_state] + gen_inputs,
                    outputs= None
                ).then(fn= change_model,
                    inputs=[state, model_choice],
                    outputs= [model_description, header]
                ).then(fn= fill_inputs, 
                    inputs=[state],
                    outputs=gen_inputs + extra_inputs,
                    show_progress="full" if args.debug_gen_form else "hidden",
                ).then(fn= preload_model_when_switching, 
                    inputs=[state],
                    outputs=[gen_status])
            
                generate_btn.click(fn = init_generate, inputs = [state, output, last_choice, audio_files_paths, audio_file_selected], outputs=[generate_trigger, mode])
                add_to_queue_btn.click(fn = lambda : (get_unique_id(), ""), inputs = None, outputs=[add_to_queue_trigger, mode])
                # gr.on(triggers=[add_to_queue_btn.click, add_to_queue_trigger.change],fn=validate_wizard_prompt, 
                add_to_queue_trigger.change(fn=validate_wizard_prompt, 
                    inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                    outputs= [prompt],
                    show_progress="hidden",
                ).then(fn=save_inputs,
                    inputs =[target_state] + gen_inputs,
                    outputs= None
                ).then(fn=process_prompt_and_add_tasks,
                    inputs = [state, current_gallery_tab, model_choice],
                    outputs=[queue_html, queue_accordion],
                    show_progress="hidden",
                ).then(
                    fn=update_status,
                    inputs = [state],
                )

                generate_trigger.change(fn=validate_wizard_prompt,
                    inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                    outputs= [prompt],
                    show_progress="hidden",
                ).then(fn=save_inputs,
                    inputs =[target_state] + gen_inputs,
                    outputs= None
                ).then(fn=process_prompt_and_add_tasks,
                    inputs = [state, current_gallery_tab, model_choice],
                    outputs= [queue_html, queue_accordion],
                    show_progress="hidden",
                ).then(fn=prepare_generate_video,
                    inputs= [state],
                    outputs= [generate_btn, add_to_queue_btn, current_gen_column, current_gen_buttons_row]
                ).then(fn=activate_status,
                    inputs= [state],
                    outputs= [status_trigger],             
                ).then(fn=process_tasks,
                    inputs= [state],
                    outputs= [preview_trigger, output_trigger, refresh_models_trigger], 
                    show_progress="hidden",
                ).then(finalize_generation,
                    inputs= [state], 
                    outputs= [gallery_tabs, current_gallery_tab, output, audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger, abort_btn, earlystop_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info]
                ).then(
                    fn=lambda s: gr.Accordion(open=False) if len(get_gen_info(s).get("queue", [])) <= 1 else gr.update(),
                    inputs=[state],
                    outputs=[queue_accordion]
                ).then(unload_model_if_needed,
                    inputs= [state], 
                    outputs= []
                )

                gr.on(triggers=[load_queue_btn.upload, main.load, load_queue_trigger.change],
                    fn=load_queue_action,
                    inputs=[load_queue_btn, state],
                    outputs=[queue_html]
                ).then(
                     fn=lambda s: (gr.update(visible=bool(get_gen_info(s).get("queue",[]))), gr.Accordion(open=True)) if bool(get_gen_info(s).get("queue",[])) else (gr.update(visible=False), gr.update()),
                     inputs=[state],
                     outputs=[current_gen_column, queue_accordion]
                ).then(
                    fn=init_process_queue_if_any,
                    inputs=[state],
                    outputs=[generate_btn, add_to_queue_btn, current_gen_column, ]
                ).then(fn=activate_status,
                    inputs= [state],
                    outputs= [status_trigger],             
                ).then(
                    fn=process_tasks,
                    inputs=[state],
                    outputs=[preview_trigger, output_trigger, refresh_models_trigger],
                    trigger_mode="once"
                ).then(
                    fn=finalize_generation_with_state,
                    inputs=[state],
                    outputs=[gallery_tabs, current_gallery_tab, output, audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger, abort_btn, earlystop_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info, queue_accordion, state],
                    trigger_mode="always_last"
                ).then(
                    unload_model_if_needed,
                     inputs= [state],
                     outputs= []
                )



            single_hidden_trigger_btn.click(
                fn=show_countdown_info_from_state,
                inputs=[hidden_countdown_state],
                outputs=[hidden_countdown_state]
            )
            quit_button.click(
                fn=start_quit_process,
                inputs=[],
                outputs=[hidden_countdown_state, quit_button, quit_confirmation_row]
            ).then(
                fn=None, inputs=None, outputs=None, js=start_quit_timer_js
            )

            confirm_quit_button.click(
                fn=quit_application,
                inputs=[],
                outputs=[]
            ).then(
                fn=None, inputs=None, outputs=None, js=cancel_quit_timer_js
            )

            cancel_quit_button.click(
                fn=cancel_quit_process,
                inputs=[],
                outputs=[hidden_countdown_state, quit_button, quit_confirmation_row]
            ).then(
                fn=None, inputs=None, outputs=None, js=cancel_quit_timer_js
            )

            hidden_force_quit_trigger.click(
                fn=quit_application,
                inputs=[],
                outputs=[]
            )

            save_queue_btn.click(
                fn=save_queue_action,
                inputs=[state],
                outputs=[queue_zip_base64_output]
            ).then(
                fn=None,
                inputs=[queue_zip_base64_output],
                outputs=None,
                js=trigger_zip_download_js
            )

            clear_queue_btn.click(
                fn=clear_queue_action,
                inputs=[state],
                outputs=[queue_html]
            ).then(
                 fn=lambda: (gr.update(visible=False), gr.Accordion(open=False)),
                 inputs=None,
                 outputs=[current_gen_column, queue_accordion]
            )
    locals_dict = locals()
    locals_dict.update(custom_setting_components_map)
    if update_form:
        gen_inputs = [state_dict if k=="state" else locals_dict[k]  for k in inputs_names] + [state_dict, plugin_data] + extra_inputs
        return gen_inputs
    else:
        app.run_component_insertion(locals_dict)
        return locals_dict


def compact_name(family_name, model_name):
    return model_dropdowns.compact_name(family_name, model_name)

def _get_dropdown_deps():
    return model_dropdowns.DropdownDeps(
        transformer_types=transformer_types,
        displayed_model_types=displayed_model_types,
        transformer_type=transformer_type,
        three_levels_hierarchy=three_levels_hierarchy,
        families_infos=families_infos,
        server_config=server_config,
        transformer_quantization=transformer_quantization,
        transformer_dtype_policy=transformer_dtype_policy,
        text_encoder_quantization=text_encoder_quantization,
        get_model_def=get_model_def,
        get_model_recursive_prop=get_model_recursive_prop,
        get_model_filename=get_model_filename,
        get_local_model_filename=fl.get_local_model_filename,
        get_lora_dir=get_lora_dir,
        get_parent_model_type=get_parent_model_type,
        get_base_model_type=get_base_model_type,
        get_model_family=get_model_family,
        get_model_name=get_model_name,
        get_transformer_dtype=get_transformer_dtype,
    )

def create_models_hierarchy(rows):
    return model_dropdowns.create_models_hierarchy(rows)

def get_sorted_dropdown(dropdown_types, current_model_family, current_model_type, three_levels = True):
    return model_dropdowns.get_sorted_dropdown(_get_dropdown_deps(), dropdown_types, current_model_family, current_model_type, three_levels)

def generate_dropdown_model_list(current_model_type):
    return model_dropdowns.generate_dropdown_model_list(_get_dropdown_deps(), current_model_type)

def change_model_family(state, current_model_family):
    return model_dropdowns.change_model_family(_get_dropdown_deps(), state, current_model_family)

def change_model_base_types(state,  current_model_family, model_base_type_choice):
    return model_dropdowns.change_model_base_types(_get_dropdown_deps(), state, current_model_family, model_base_type_choice)

def get_js():
    start_quit_timer_js = """
    () => {
        function findAndClickGradioButton(elemId) {
            const gradioApp = document.querySelector('gradio-app') || document;
            const button = gradioApp.querySelector(`#${elemId}`);
            if (button) { button.click(); }
        }

        if (window.quitCountdownTimeoutId) clearTimeout(window.quitCountdownTimeoutId);

        let js_click_count = 0;
        const max_clicks = 5;

        function countdownStep() {
            if (js_click_count < max_clicks) {
                findAndClickGradioButton('trigger_info_single_btn');
                js_click_count++;
                window.quitCountdownTimeoutId = setTimeout(countdownStep, 1000);
            } else {
                findAndClickGradioButton('force_quit_btn_hidden');
            }
        }

        countdownStep();
    }
    """

    cancel_quit_timer_js = """
    () => {
        if (window.quitCountdownTimeoutId) {
            clearTimeout(window.quitCountdownTimeoutId);
            window.quitCountdownTimeoutId = null;
            console.log("Quit countdown cancelled (single trigger).");
        }
    }
    """

    trigger_zip_download_js = """
    (base64String) => {
        if (!base64String) {
        console.log("No base64 zip data received, skipping download.");
        return;
        }
        try {
        const byteCharacters = atob(base64String);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/zip' });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = 'queue.zip';
        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        console.log("Zip download triggered.");
        } catch (e) {
        console.error("Error processing base64 data or triggering download:", e);
        }
    }
    """

    trigger_settings_download_js = """
    (base64String, filename) => {
        if (!base64String) {
        console.log("No base64 settings data received, skipping download.");
        return;
        }
        try {
        const byteCharacters = atob(base64String);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'application/text' });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        console.log("settings download triggered.");
        } catch (e) {
        console.error("Error processing base64 data or triggering download:", e);
        }
    }
    """

    click_brush_js = """
    () => {
        setTimeout(() => {
            if (window.__wangpMagicMaskNS?.focusVisibleImageEditor?.(true)) {
                console.log('Image editor focused and Brush button clicked');
                return;
            }
            const brushButton = document.querySelector('button[aria-label="Brush"]');
            if (brushButton) {
                brushButton.click();
                console.log('Brush button clicked');
            } else {
                console.log('Brush button not found');
            }
        }, 1000);
    }    """

    return start_quit_timer_js, cancel_quit_timer_js, trigger_zip_download_js, trigger_settings_download_js, click_brush_js

def create_ui():
    # Load CSS from external file
    css_path = os.path.join(os.path.dirname(__file__), "shared", "gradio", "ui_styles.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    css += "\n" + MagicMaskUI.get_css()
    css += "\n" + assistant_chat.get_css()
    UI_theme = server_config.get("UI_theme", "default")
    UI_theme  = args.theme if len(args.theme) > 0 else UI_theme
    if UI_theme == "gradio":
        theme = None
    else:
        theme = gr.themes.Soft(font=["Verdana"], primary_hue="sky", neutral_hue="slate", spacing_size=theme_spacing_size, radius_size=theme_radius_size, text_size=theme_text_size)

    # Load main JS from external file
    js_path = os.path.join(os.path.dirname(__file__), "shared", "gradio", "ui_scripts.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    js += AudioGallery.get_javascript()
    js += MagicMaskUI.get_javascript()
    js += assistant_chat.get_javascript()
    AudioGallery.install_gradio_upload_mtime_patch()
    app.initialize_plugins(globals())
    plugin_js = ""
    if hasattr(app, "plugin_manager"):
        plugin_js = app.plugin_manager.get_custom_js()
        if isinstance(plugin_js, str) and plugin_js.strip():
            js += f"\n{plugin_js}\n"
    js += """
    }
    """
    if server_config.get("display_stats", 0) == 1:
        from shared.utils.stats import SystemStatsApp
        stats_app = SystemStatsApp() 
    else:
        stats_app = None

    with gr.Blocks(css=css, js=js, theme=theme, title="WanGP", fill_width=True) as main:
        gr.Markdown(f"<div align=center><H1>Wan<SUP>GP</SUP> v{WanGP_version} <FONT SIZE=4>by <I>DeepBeepMeep</I></FONT> <FONT SIZE=3>") # (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A>)</FONT SIZE=3></H1></div>")
        global model_list

        tab_state = gr.State({ "tab_no":0 }) 
        target_edit_state = gr.Text(value = "edit_state", interactive= False, visible= False)
        edit_queue_trigger = gr.Text(value='', interactive= False, visible=False)
        with gr.Tabs(selected="video_gen", ) as main_tabs:
            # JS keepalive patch targets the stable Gradio tab id "video_gen"; the label can change, but if this id changes the patch must be updated too.
            with gr.Tab("Video Generator", id="video_gen") as video_generator_tab:
                with gr.Row():
                    if args.lock_model:    
                        gr.Markdown("<div class='title-with-lines'><div class=line></div><h2>" + get_model_name(transformer_type) + "</h2><div class=line></div>")
                        model_family = gr.Dropdown(visible=False, value= "")
                        model_choice = gr.Dropdown(visible=False, value= transformer_type, choices= [transformer_type])
                    else:
                        gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")                        
                        model_family, model_base_type_choice, model_choice = generate_dropdown_model_list(transformer_type)
                        gr.Markdown("<div class='title-with-lines'><div class=line width=100%></div></div>")
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes="header-markdown-group"):
                            description_html, header_html = generate_header(transformer_type, compile, attention_mode)
                            model_description = gr.Markdown(description_html, visible= True)
                            header = gr.Markdown(header_html, visible= True)
                    if stats_app is not None:
                        stats_element = stats_app.get_gradio_element()

                with gr.Row():
                    generator_tab_components = generate_video_tab(
                        model_family=model_family,
                        model_base_type_choice=model_base_type_choice,
                        model_choice=model_choice,
                        model_description=model_description,
                        header=header,
                        main=main,
                        main_tabs=main_tabs,
                        tab_id='generate'
                    )
                    (state, loras_choices, lset_name, resolution, refresh_form_trigger, save_form_trigger) = generator_tab_components['state'], generator_tab_components['loras_choices'], generator_tab_components['lset_name'], generator_tab_components['resolution'], generator_tab_components['refresh_form_trigger'], generator_tab_components['save_form_trigger']
            with gr.Tab("Edit", id="edit", visible=False) as edit_tab:
                edit_title_md = gr.Markdown()
                edit_tab_components = generate_video_tab(
                    update_form=False,
                    state_dict=state.value,
                    ui_defaults=get_default_settings(transformer_type),
                    model_family=model_family,
                    model_base_type_choice=model_base_type_choice,
                    model_choice=model_choice,
                    header=header,
                    main=main,
                    main_tabs=main_tabs,
                    tab_id='edit',
                    edit_tab=edit_tab,
                    default_state=state
                )

                edit_inputs_names = inputs_names 
                final_edit_inputs = [edit_tab_components[k] for k in edit_inputs_names if k != 'state'] + [edit_tab_components['state'], edit_tab_components['plugin_data']] 
                edit_tab_inputs = final_edit_inputs  + edit_tab_components['extra_inputs']

                def fill_inputs_for_edit(state):
                    editing_task_id = state.get("editing_task_id", None)
                    all_outputs_count = 1 + len(edit_tab_inputs)
                    default_return = [gr.update()] * all_outputs_count

                    if editing_task_id is None:
                        return default_return

                    gen = get_gen_info(state)
                    queue = gen.get("queue", [])
                    task = next((t for t in queue if t.get('id') == editing_task_id), None)

                    if task is None:
                        gr.Warning("Task to edit not found in queue. It might have been processed or deleted.")
                        state["editing_task_id"] = None
                        return default_return

                    prompt_text = task.get('prompt', 'Unknown Prompt')[:80]
                    edit_title_text = f"<div align='center'><h2>Editing task ID {editing_task_id}: '{prompt_text}...'</h2></div>"
                    ui_defaults=task['params'].copy()
                    state["edit_model_type"] = ui_defaults["model_type"] 
                    all_new_component_values = generate_video_tab(update_form=True, state_dict=state, ui_defaults=ui_defaults, tab_id='edit', )
                    return [edit_title_text] + all_new_component_values

                edit_btn = edit_tab_components['edit_btn']
                cancel_btn = edit_tab_components['cancel_btn']
                silent_cancel_btn = edit_tab_components['silent_cancel_btn']
                js_trigger_index = generator_tab_components['js_trigger_index']

                wizard_inputs = [state, edit_tab_components['wizard_prompt_activated_var'], edit_tab_components['wizard_variables_var'], edit_tab_components['prompt'], edit_tab_components['wizard_prompt']] + edit_tab_components['prompt_vars']

                edit_queue_trigger.change(
                    fn=fill_inputs_for_edit,
                    inputs=[state],
                    outputs=[edit_title_md] + edit_tab_inputs
                )

                edit_btn.click(
                    fn=validate_wizard_prompt,
                    inputs=wizard_inputs,
                    outputs=[edit_tab_components['prompt']]
                ).then(fn=save_inputs,
                    inputs =[target_edit_state] + final_edit_inputs,
                    outputs= None
                ).then(fn=validate_edit,
                    inputs =state,
                    outputs= None
                ).then(
                    fn=edit_task_in_queue,
                    inputs=state,
                    outputs=[js_trigger_index, main_tabs, edit_tab, generator_tab_components['queue_html']]
                )
                
                js_trigger_index.change(
                    fn=None, inputs=[js_trigger_index], outputs=None,
                    js="(index) => { if (index !== null && index >= 0) { window.updateAndTrigger('silent_edit_' + index); setTimeout(() => { document.querySelector('#silent_edit_tab_cancel_button')?.click(); }, 50); } }"
                )

                cancel_btn.click(fn=cancel_edit, inputs=[state], outputs=[main_tabs, edit_tab])
                silent_cancel_btn.click(fn=silent_cancel_edit, inputs=[state], outputs=[main_tabs, js_trigger_index, edit_tab])

            generator_tab_components['queue_action_trigger'].click(
                fn=handle_queue_action,
                inputs=[generator_tab_components['state'], generator_tab_components['queue_action_input']],
                outputs=[generator_tab_components['queue_html'], main_tabs, edit_tab, edit_queue_trigger],
                show_progress="hidden"
            )

            video_generator_tab.select(lambda state: state.update({"active_form": "add"}), inputs=state).then(
                fn=refresh_model_dropdowns,
                inputs=[state],
                outputs=[model_family, model_base_type_choice, model_choice, refresh_form_trigger],
                show_progress="hidden",
            )
            edit_tab.select(lambda state: state.update({"active_form": "edit"}), inputs=state)
            app.setup_ui_tabs(main_tabs, state, generator_tab_components["set_save_form_event"])
        if stats_app is not None:
            stats_app.setup_events(main, state)
        return main

def clear_startup_lock():
    if os.path.exists(STARTUP_LOCK_FILE):
        try:
            os.remove(STARTUP_LOCK_FILE)
        except:
            pass

if __name__ == "__main__":
    if args.merge_catalog:
        manager = PluginManager()
        print(manager.merge_local_catalog())
        sys.exit(0)

    if args.refresh_catalog or args.refresh_full_catalog:
        manager = PluginManager()
        installed_only = not args.refresh_full_catalog
        result = manager.refresh_catalog(installed_only=installed_only, use_remote=False)
        checked = result.get("checked", 0)
        updates_available = result.get("updates_available", 0)
        scope = "installed plugins" if installed_only else "catalog plugins"
        if updates_available <= 0:
            message = "No Plugin Update is available"
        elif updates_available == 1:
            message = "One Plugin Update is available"
        else:
            message = f"{updates_available} Plugin Updates are available"
        print(f"[Plugins] Checked {checked} {scope}. {message}.")
        sys.exit(0)

    app = WAN2GPApplication()

    if args.ask_deepy:
        download_ffmpeg()
        if len(args.output_dir) > 0:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            server_config["save_path"] = args.output_dir
            server_config["image_save_path"] = args.output_dir
            server_config["audio_save_path"] = args.output_dir
            save_path = args.output_dir
            image_save_path = args.output_dir
            audio_save_path = args.output_dir
            print(f"Output directory: {args.output_dir}")
        server_config["notification_sound_enabled"] = 0
        sys.exit(
            deepy_cli.run_deepy_cli_session(
                deepy_cli.DeepyCliDeps(
                    controller=_deepy,
                    get_server_config=lambda: server_config,
                    get_gen_info=get_gen_info,
                    get_settings_from_file=get_settings_from_file,
                    load_queue_action=load_queue_action,
                    validate_task=validate_task,
                    generate_video=generate_video,
                    default_model_type=transformer_type,
                    callbacks=deepy_cli.DeepyCliCallbacks(
                        handlers={
                            "abort_generation": (lambda state, client_id: abort_generation(state, client_id, notify=False),),
                        }
                    ),
                )
            )
        )

    # CLI Queue Processing Mode
    if len(args.process) > 0:
        download_ffmpeg()  # Still needed for video encoding

        if not os.path.isfile(args.process):
            print(f"[ERROR] File not found: {args.process}")
            sys.exit(1)

        # Detect file type
        is_json = args.process.lower().endswith('.json')
        file_type = "settings" if is_json else "queue"
        print(f"WanGP CLI Mode - Processing {file_type}: {args.process}")

        # Override output directory if specified
        if len(args.output_dir) > 0:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            server_config["save_path"] = args.output_dir
            server_config["image_save_path"] = args.output_dir
            server_config["audio_save_path"] = args.output_dir
            # Keep module-level paths in sync (used by save_video / get_available_filename).
            save_path = args.output_dir
            image_save_path = args.output_dir
            audio_save_path = args.output_dir
            print(f"Output directory: {args.output_dir}")

        # Headless CLI runs: disable notification sounds to avoid pygame/sounddevice issues.
        server_config["notification_sound_enabled"] = 0

        # Create minimal state with all required fields
        state = {
            "gen": {
                "queue": [],
                "in_progress": False,
                "file_list": [],
                "file_settings_list": [],
                "audio_file_list": [],
                "audio_file_settings_list": [],
                "selected": 0,
                "audio_selected": 0,
                "prompt_no": 0,
                "prompts_max": 0,
                "repeat_no": 0,
                "total_generation": 1,
                "window_no": 0,
                "total_windows": 0,
                "progress_status": "",
                "process_status": "process:main",
            },
            "loras": [],
        }

        # Parse file based on type
        if is_json:
            queue, error = _parse_settings_json(args.process, state)
        else:
            queue, error = _parse_queue_zip(args.process, state)
        if error:
            print(f"[ERROR] {error}")
            sys.exit(1)

        if len(queue) == 0:
            print("Queue is empty, nothing to process")
            sys.exit(0)

        print(f"Loaded {len(queue)} task(s)")

        # Dry-run mode: validate and exit
        if args.dry_run:
            print("\n[DRY-RUN] Queue validation:")
            valid_count = 0
            for i, task in enumerate(queue, 1):
                prompt = (task.get('prompt', '') or '')[:50]
                model = task.get('params', {}).get('model_type', 'unknown')
                steps = task.get('params', {}).get('num_inference_steps', '?')
                length = task.get('params', {}).get('video_length', '?')
                print(f"  Task {i}: model={model}, steps={steps}, frames={length}")
                print(f"          prompt: {prompt}...")
                validated, validation_error = validate_task(task, state)
                if validated is None:
                    print(f"          [INVALID] {validation_error or 'Task failed validation.'}")
                else:
                    print(f"          [OK]")
                    valid_count += 1
            print(f"\n[DRY-RUN] Validation complete. {valid_count}/{len(queue)} task(s) valid.")
            sys.exit(0 if valid_count == len(queue) else 1)

        state["gen"]["queue"] = queue

        try:
            success = process_tasks_cli(queue, state)
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n\nAborted by user")
            sys.exit(130)

    # Normal Gradio mode continues below...
    atexit.register(autosave_queue)

    globals()["SAFE_MODE"] = False

    if os.path.exists(STARTUP_LOCK_FILE):
        print("\n" + "!"*60)
        print("DETECTED FAILED PREVIOUS STARTUP.")
        print("Waiting 2 seconds...")
        print("Press 'c' to CANCEL Safe Mode and force normal startup.")
        if os.name != 'nt':
            print("(On Linux/Mac, type 'c' and press Enter)")
        print("!"*60 + "\n")
        cancel_safe_mode = False
        start_wait = time.time()
        try:
            if os.name == 'nt':
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getwch()
                
                while time.time() - start_wait < 2:
                    if msvcrt.kbhit():
                        if msvcrt.getwch().lower() == 'c':
                            cancel_safe_mode = True
                            break
                    time.sleep(0.05)
            else:
                import select
                while time.time() - start_wait < 2:
                    r, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if r:
                        line = sys.stdin.readline()
                        if 'c' in line.lower():
                            cancel_safe_mode = True
                            break
        except Exception as e:
            print(f"Warning: Input detection failed: {e}")

        if cancel_safe_mode:
            print("\nSAFE MODE CANCELLED. Proceeding with normal startup.")
            globals()["SAFE_MODE"] = False
        else:
            print("\nENTERING SAFE MODE. User plugins disabled.")
            globals()["SAFE_MODE"] = True

    try:
        with open(STARTUP_LOCK_FILE, "w"):
            pass
    except Exception as e:
        print(f"Warning: Could not create startup lock file: {e}")

    download_ffmpeg()
    # threading.Thread(target=runner, daemon=True).start()
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    server_port = int(args.server_port)
    if server_port == 0:
        server_port = int(os.getenv("SERVER_PORT", "7860"))
    server_name = args.server_name
    if args.listen:
        server_name = "0.0.0.0"
    if len(server_name) == 0:
        server_name = os.getenv("SERVER_NAME", "localhost")
    demo = create_ui()
    clear_startup_lock()
    if args.open_browser:
        import webbrowser
        if server_name.startswith("http"):
            url = server_name
        else:
            url = "http://" + server_name
        webbrowser.open(url + ":" + str(server_port), new = 0, autoraise = True)
    demo.launch(
        favicon_path="favicon.png",
        server_name=server_name,
        server_port=server_port,
        share=args.share,
        allowed_paths=list({save_path, image_save_path, audio_save_path, "icons"}),
    )
