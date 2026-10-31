############# WanGP Copyright DeepBeepMeep 2025-2026 #############
import os, sys
from tqdm import tqdm

WanGP_version = "18.888"
_welcome_message = f" WanGP v{WanGP_version} - Copyright (c) 2024-2026 DeepBeepMeep "
tqdm.write(f"\033[1m\033[93m{_welcome_message:*^100}\033[0m")

os.environ["GRADIO_LANG"] = "en"
# Keep Triton's existing fusion default explicit; per-kernel flags and user overrides still apply.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
# ROCm: without it, SDPA silently falls back to the math kernel on GPUs where AOTriton is still marked experimental
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
# OpenBLAS (NumPy, SciPy) commits a 32 MiB buffer per thread at import: 1.5 GB with 24 threads, unused by WanGP's PyTorch computations
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(min(8, os.cpu_count() or 8)))
if "--offline" in sys.argv:  # before any Hugging Face library reads it at import: no download, a missing file is an error (shared/utils/download.py)
    os.environ["HF_HUB_OFFLINE"] = "1"
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)
from shared.native_runtime import preload_preferred_libstdcxx
preload_preferred_libstdcxx()
from shared.default_device import set_default_cuda_device_from_arg; set_default_cuda_device_from_arg("gpu")
# # os.environ.pop("TORCH_LOGS", None)  # make sure no env var is suppressing/overriding
# os.environ["TORCH_LOGS"]= "recompiles"
import torch._logging as tlog
# tlog.set_logs(recompiles=True, guards=True, graph_breaks=True)
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

# ── Apple Silicon MPS patch: MUST come before mmgp import ──
import torch
is_mps = sys.platform == 'darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if is_mps:
    from shared.mps.device_patch import apply_mps_patch
    apply_mps_patch()
from shared.cuda_memory import apply_startup_settings, write_vram_debug_report, write_ram_debug_report, ram_debug_mark, release_ram_cache; apply_startup_settings(sys.argv, "wgp_config.json") # VRAM allocator and CUDA stack reserve, before anything initializes CUDA

import time
import threading
from functools import partial, wraps
from shared.utils.download_progress import DownloadCancelled, check_download_cancelled
from shared.utils.download import DownloadError, generation_downloads
from shared.utils.media_control import MediaProcessingAborted, controlled_model_loading, media_abort_requested
from shared.utils.default_device import call_with_default_device
from shared.utils.power_throttling import prevent_power_throttling
from shared.gradio import metadata_events
from shared.utils.config_store import config_lock, read_config, write_config, write_config_deferred, update_config
import warnings
warnings.filterwarnings('ignore', message='Failed to find.*', module='triton')
warnings.filterwarnings("ignore", message=r"Failed to launch Triton kernels, likely due to missing CUDA toolkit; falling back to a slower .* implementation\.\.\.", category=UserWarning, module=r"whisper\.timing")
from mmgp import offload, safetensors2, profile_type , quant_router, cuda_graphs
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from shared.tinyvae.media import VideoPreview
from datetime import datetime
import gradio as gr
from shared.gradio import downloads as gradio_downloads
from shared.gradio.progress import WangpProgress
from shared.gradio import gradio_event_context_patch, gradio_model_switch_patch, gradio_queue_focus_patch, gradio_startup_patch, video_preview
from gradio.themes.utils.sizes import Size
import random
import json
import copy
import numpy as np
import importlib
from models import model_metadata
from shared import notifications
from shared.utils import notification_sound
from shared.utils.setting_names import settings_error
from shared.utils import generation_overrides
from shared.utils.loras_mutipliers import preparse_loras_multipliers, parse_loras_multipliers
from shared.utils.utils import convert_tensor_to_image, convert_video_tensor_to_uint8_chunked, save_image, get_video_info, get_file_creation_date, convert_image_to_video, calculate_new_dimensions, convert_image_to_tensor, calculate_dimensions_and_resize_image, rescale_and_crop, get_video_frame, resize_and_remove_background, rgb_bw_to_rgba_mask, image_editor_layer_to_rgb_mask, to_rgb_tensor, get_resampled_video_transparent, get_video_summary_extras
from shared.utils.utils import calculate_new_dimensions, get_outpainting_dims, get_outpainting_frame_location, get_outpainting_full_area_dimensions, resolve_outpainting_dims
from shared.utils.utils import has_video_file_extension, has_image_file_extension, has_audio_file_extension, has_mesh_file_extension, has_world_file_extension, has_app_file_extension, MESH_IMPORT_EXTENSIONS, guide_to_float, guide_mask_to_float
from shared.utils.audio_video import extract_audio_tracks, subtitle_stream_count, combine_video_with_audio_tracks, combine_and_concatenate_video_with_audio_tracks, cleanup_temp_audio_files, normalize_audio_volumes_to_temp_files, save_video, save_hdr_video, save_image, get_image_format, get_media_duration_and_audio_layouts
from shared.utils.audio_video import append_sliding_window_audio, audio_overlap_repair_seconds, source_audio_repair_ms, read_image_metadata, extract_audio_track_to_wav, write_wav_file, save_audio_file, get_audio_codec_extension, create_silent_wav_file
from shared.utils.audio_video import truncate_audio, shift_audio_trim_ranges, trim_audio_ranges, trim_audio_file_ranges, slice_audio_window, resolve_mux_audio_sampling_rate
from shared.utils.audio_metadata import read_audio_metadata, extract_creation_datetime_from_metadata, resolve_audio_creation_datetime
from shared.utils.media_recording import record_file_metadata as shared_record_file_metadata
from shared.utils.settings_bundle import is_wangp_settings_filename
from shared.utils.video_decode import decode_video_frames_ffmpeg, probe_video_stream_metadata
from shared.utils.virtual_media import get_virtual_image, get_virtual_media_entry, get_virtual_media_vsource, media_source_exists, parse_virtual_media_path, replace_virtual_media_source, strip_virtual_media_suffix
from shared.utils.frame_scheduler import SLASH_BLOCK_RE, build_default_window_plan, build_extension_window, build_frame_scheduler, fit_video_length, floor_frame_count, has_slash_commands, normalize_frame_count, set_video_length_rounding, prepare_loras_mult_windows
from shared.match_archi import match_nvidia_architecture
from shared.attention import ATTENTION_MODE_AVAILABILITY, get_attention_modes, get_supported_attention_modes, get_default_attention_mode, get_override_attention_modes, get_supported_override_attention_modes
from shared.utils.utils import truncate_for_filesystem, sanitize_file_name, process_images_multithread, get_default_workers, resize_lanczos_frames, expand_or_shrink_mask, prepare_binary_mask_frame
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
from shared.utils.model_unload import model_unload_guard, wait_for_model_unload
from shared.deepy.config import DEEPY_KV_CACHE_QUANTIZATION_DEFAULT, DEEPY_KV_CACHE_QUANTIZATION_KEY, get_deepy_default_runtime_config, set_deepy_runtime_config, python_apps_level
from shared.deepy.onboarding import apply_first_launch_deepy_prime_defaults
from shared.remote_llm.config import LLM_CONFIG_KEY, is_remote_engine, normalize_llm_config, resolve_role_engine
from shared.loras_migration import migrate_loras_layout
from shared.lora_paths import resolve_lora_dir
from shared.utils.wgp_config_migration import migrate_extension_defaults
from shared.utils import files_locator as fl 
from shared.gradio.strip_gallery import StripGallery, STRIP_GALLERY_CLASSES
from shared.utils.gallery_sources import GALLERY_SOURCES, STRIP_SOURCES, VIDEO, MESH, WORLD, APP, gallery_source, source_index, source_for_path, gallery_state_defaults
from shared.utils.self_refiner import normalize_self_refiner_plan, ensure_refiner_list, add_refiner_rule, remove_refiner_rule
from shared.deepy import controller as deepy_controller
from shared.deepy.hybrid import HybridService, SharedState, service_for
from shared.utils.generation_state import set_generation_state
from shared.utils.gallery_view import gallery_window
from shared.gradio.gradio_media_response_patch import allow_listed_media
from shared.gradio.import_files import ImportFiles
from shared.deepy import filesystem as deepy_filesystem
from shared.deepy import cli as deepy_cli
from shared.deepy import voice as deepy_voice
from shared.deepy import gradio_ui as deepy_gradio_ui
from shared.deepy import session_store as deepy_session_store
from shared import extra_settings
from shared import config_groups as model_config_groups
from shared import resolutions as resolution_utils
import torch
import gc
import traceback
import math 
import typing
import inspect
from shared.utils import prompt_parser
from shared.prompt_enhancer import chaining as prompt_enhancer_chaining
from shared.prompt_enhancer import images as prompt_enhancer_images
from shared.prompt_enhancer import templates as prompt_enhancer_templates
from shared.prompt_enhancer import labels as prompt_enhancer_labels
from shared.prompt_enhancer.progress import EnhancementProgress
from shared.prompt_enhancer import in_process as prompt_enhancer_in_process
from shared.prompt_enhancer.config import PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT, PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, normalize_prompt_enhancer_speculative_decoding
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
import requests
from shared.gradio.gallery import AdvancedMediaGallery, get_gradio_file_path
from shared.gradio.hierarchy_selector import HierarchySelector, build_choices_hierarchy
from shared.ffmpeg_setup import download_ffmpeg
from shared.api import apply_video_length_duration, get_api_output_options, store_api_output_artifact
from shared.utils.side_files import process_side_files
from shared.utils import workspace_storage
from shared.utils.plugins import PluginManager, WAN2GPApplication, SYSTEM_PLUGINS
from shared.llm_engines.lmgp.vllm_support import resolve_lm_features, resolve_prompt_enhancer_lm_features
resolve_prompt_enhancer_lm_features()
from shared.gradio import assistant_chat, field_help, finetune_editor, gallery_files, local_file_picker, model_infos, model_output_filter, model_selector_toolbar, ui_studio
from shared.gradio.magic_mask import MagicMaskUI, video_mask_area_visible, video_mask_controls_visible, video_mask_dropdown_visible
from shared import model_dropdowns
from shared import settings_metadata
from postprocessing import audio_processors as audio_processor_api
from postprocessing import temporal_upsamplers as temporal_upsampler_api
from postprocessing import spatial_upsamplers as upsampler_api
from postprocessing import parameter_ui
from shared.utils import mesh_postprocessing
from postprocessing.lanczos import resize_lanczos_spatial
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
VIDEO_LENGTH_ROUNDING = "ceil"  # how a requested video length is fitted to the model's frame steps: "floor", "ceil" (never shorter than requested) or "nearest"
set_video_length_rounding(VIDEO_LENGTH_ROUNDING)
settings_version = 2.82
max_source_video_frames = 3000
prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None
image_names_list = ["image_start", "image_end", "image_refs"]
CUSTOM_SETTINGS_MAX = 5
CUSTOM_SETTINGS_PER_ROW = 2
CUSTOM_SETTING_DROPDOWN_MAX = 5
CUSTOM_SETTING_TYPES = {"int", "float", "text", "dropdown"}
PHASE_2_TILING_VIDEO_PROMPT_FLAG = "~"
PHASE_2_TILING_GUIDANCE_VALUE = "2~"
int8_kernels = "auto"
fp8_kernels = "auto"
theme_text_size = Size("8.1px", "9px", "10.8px", "12.6px", "14.4px", "19.8px", "23.4px", name="wangp_text_90")
theme_spacing_size = Size("0.9px", "1.8px", "3.6px", "5.4px", "7.2px", "9px", "14.4px", name="wangp_spacing_90")
theme_radius_size = Size("0.9px", "1.8px", "3.6px", "5.4px", "7.2px", "10.8px", "19.8px", name="wangp_radius_90")
app = None
# All media attachment keys for queue save/load
ATTACHMENT_KEYS = ["image_start", "image_end", "image_refs", "image_guide", "image_mask",
                   "video_guide", "video_guide2", "video_guide3", "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_guide3", "audio_source", "replace_voice_sample", "replace_voice_sample2", "custom_guide"]
PRESERVE_MEDIA_ON_SETTINGS_IMPORT = True

lock = threading.Lock()
current_task_id = None
task_id = 0
unique_id = 0
unique_id_lock = threading.Lock()
offloadobj = enhancer_offloadobj = wan_model = None
loaded_output_type = "video" # kind of output of the current model: its VRAM Preload mode also applies to the extensions set to Auto
loaded_config = ""
loaded_compile = "disabled"
loaded_quantization = None
reload_needed = True
_HANDLER_MODULES = [
    "shared.qtypes.scaled_fp8",
    "shared.qtypes.nvfp4",
    "shared.qtypes.bnb_nf4",
    "shared.qtypes.nunchaku_int4",
    "shared.qtypes.nunchaku_fp4",
    "shared.qtypes.asym_w4a8_int8",
    "shared.qtypes.int8_convrot",
    "shared.qtypes.gptq_int4",
    "shared.qtypes.gguf",
]
for handler in _HANDLER_MODULES:
    quant_router.register_handler(handler)
from shared.qtypes import gguf as gguf_handler
quant_router.register_file_extension("gguf", gguf_handler)
from shared.kernels import fp8_backend, int8_backend, kernel_policy, rowmap_linear, scaled_fp16  # rowmap_linear: MMGP row views keep interleaved fused weights uncopied
from shared import attention_kit


def apply_int8_kernel_setting(selection: str, notify_disabled=False, resolved=None) -> bool:
    global int8_kernels
    enabled = int8_backend.configure(selection, verbose_level, resolved=resolved)
    int8_kernels = selection
    return enabled


def apply_fp8_kernel_setting(selection: str, resolved=None) -> str:
    global fp8_kernels
    backend = fp8_backend.configure(selection, resolved=resolved)
    fp8_kernels = selection
    return backend


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


_deepy_hybrid = None

def release_model():
    global wan_model, offloadobj, reload_needed
    released = transformer_type if wan_model is not None or offloadobj is not None else None
    wan_model = None
    clear_gen_cache()
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
    offload.flush_torch_caches()
    gc.collect()
    torch.cuda.empty_cache()
    reload_needed = True
    if _deepy_hybrid is not None:
        _deepy_hybrid.host_changed()
    if released is not None:  # the debug reports of a release that freed a model: what it left in RAM and VRAM
        write_vram_debug_report(save_path, f"model_release_{released}")
        write_ram_debug_report(save_path, f"model_release_{released}", census_label=f"after the release of {released}")
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

    if format.lower() == "jpeg" and (pil_image.mode in ("RGBA", "LA", "PA") or "transparency" in pil_image.info):
        format = "png"

    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
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
    return virtual_image if virtual_image is not None else Image.open(strip_virtual_media_suffix(image))

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

def estimate_first_window_overlap_frames(image_start, video_source, keep_frames_video_source, target_fps):
    if image_start is not None:
        return 1
    if video_source is None:
        return 0
    try:
        source_fps, _, _, source_frames = get_video_info(video_source)
        source_frames = int(source_frames / source_fps * target_fps) if source_fps else int(source_frames)
    except Exception:
        return max_source_video_frames
    try:
        keep_frames = max_source_video_frames if len(str(keep_frames_video_source or "")) == 0 else int(keep_frames_video_source)
    except Exception:
        return source_frames
    source_frames = max(0, source_frames + keep_frames) if keep_frames < 0 else min(source_frames, keep_frames)
    return source_frames

def clean_image_list(gradio_list, keep_alpha=False):
    if not isinstance(gradio_list, list): gradio_list = [gradio_list]
    gradio_list = [ tup[0] if isinstance(tup, tuple) else tup for tup in gradio_list ]        

    if any( not isinstance(image, (Image.Image, str))  for image in gradio_list): return None
    if any( isinstance(image, str) and not has_image_file_extension(image) for image in gradio_list): return None
    gradio_list = [convert_image(_open_image_input(img), keep_alpha) for img in gradio_list]
    return gradio_list


def silent_cancel_edit(state):
    gen = get_gen_info(state)
    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gen["queue_paused_for_edit"] = False
    return gr.Tabs(selected="media_gen"), None, gr.update(visible=False)

def cancel_edit(state):
    gen = get_gen_info(state)
    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gen["queue_paused_for_edit"] = False
        gr.Info("Edit cancelled. Resuming queue processing.")
    else:
        gr.Info("Edit cancelled.")
    return gr.Tabs(selected="media_gen"), gr.update(visible=False)

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
        return None, gr.Tabs(selected="media_gen"), gr.update(visible=False), gr.update()

    if state.get("validate_edit_success", 0) == 0:
        return None, gr.update(), gr.update(), gr.update()


    task_to_edit_index = -1
    with lock:
        task_to_edit_index = next((i for i, task in enumerate(queue) if task['id'] == editing_task_id), -1)

    if task_to_edit_index == -1:
        gr.Warning("Task not found in queue. It might have been processed or deleted.")
        state["editing_task_id"] = None
        gen["queue_paused_for_edit"] = False
        return None, gr.Tabs(selected="media_gen"), gr.update(visible=False), gr.update()
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

    return task_to_edit_index -1, gr.Tabs(selected="media_gen"), gr.update(visible=False), update_queue_data(queue)

def process_prompt_and_add_tasks(state, current_gallery_tab, model_choice):
    def ret():
        return gr.update(), gr.update()

    gen = get_gen_info(state)

    gen["last_gallery_source"] = GALLERY_SOURCES[int(current_gallery_tab)].id

    if state.get("validate_success",0) != 1:
        ret()
    
    state["validate_success"] = 0
    model_type = get_state_model_type(state)
    inputs = get_model_settings(state, model_type)

    if model_choice != model_type or inputs ==None:
        raise gr.Error("Webform can not be used as the App has been restarted since the form was displayed. Please refresh the page")
    
    inputs = inputs.copy()
    inputs["state"] =  state
    inputs["model_type"] = model_type
    inputs.pop("lset_name", None)
    if inputs == None:
        gr.Warning("Internal state error: Could not retrieve inputs for the model.")
        queue = gen.get("queue", [])
        return ret()
    if "mode" not in inputs:
        pass
    mode = inputs["mode"]
    if mode == "edit_audio":
        edit_audio_source = gen.get("edit_audio_source", None)
        edit_overrides = gen.get("edit_overrides", None)
        if edit_audio_source is None or edit_overrides is None:
            gr.Info("You must select an Audio file")
            return ret()
        for prop in ["state", "model_type", "mode"]:
            edit_overrides[prop] = inputs[prop]
        for k,v in inputs.items():
            inputs[k] = None
        inputs.update(edit_overrides)
        del gen["edit_audio_source"], gen["edit_overrides"]
        inputs["audio_source"] = edit_audio_source
        postprocess_audio = inputs.get("postprocess_audio", "") or ""
        postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
        validation_error = ""
        if not media_source_exists(edit_audio_source):
            validation_error = "Selected audio file is missing"
        elif not has_audio_file_extension(edit_audio_source):
            validation_error = "Post processing is only available with Audio files"
        elif postprocess_audio:
            validation_error = audio_processor_api.validate_method(
                postprocess_audio,
                audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,
                voice_sample=inputs.get("replace_voice_sample"),
                voice_sample2=inputs.get("replace_voice_sample2"),
            )
        else:
            validation_error = "You must choose at least one Audio Post Processing Method"
        if validation_error:
            gr.Info(validation_error)
            return ret()
        prompt = [audio_processor_api.format_method_label(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING)]
        inputs["repeat_generation"] = 1
        inputs["prompt"] = ", ".join(prompt)
        add_video_task(**inputs)
        new_prompts_count = gen["prompts_max"] = 1 + gen.get("prompts_max",0)
        state["validate_success"] = 1
        queue= gen.get("queue", [])
        update_global_queue_ref(queue)
        return update_queue_data(queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()
    if mode.startswith("edit_"):
        edit_media_source =gen.get("edit_media_source", None)
        edit_overrides =gen.get("edit_overrides", None)
        if edit_media_source is None or edit_overrides is None:
            gr.Info("You must select a Video or Image file")
            return ret()
        frames_count = 1 if has_image_file_extension(edit_media_source) or mesh_postprocessing.is_mesh_file(edit_media_source) else get_video_info(edit_media_source)[3]
        if frames_count > max_source_video_frames:
            gr.Info(f"Post processing is not supported on videos longer than {max_source_video_frames} frames. Output Video will be truncated")
            # return
        for prop in ["state", "model_type", "mode"]:
            edit_overrides[prop] = inputs[prop]
        for k,v in inputs.items():
            inputs[k] = None    
        inputs.update(edit_overrides)
        del gen["edit_media_source"], gen["edit_overrides"]
        inputs["video_source"]= edit_media_source
        prompt = []

        repeat_generation = 1
        if mode == "edit_postprocessing":
            video_source = inputs["video_source"]
            source_is_image = has_image_file_extension(video_source)
            source_is_video = has_video_file_extension(video_source)
            source_is_mesh = mesh_postprocessing.is_mesh_file(video_source)
            temporal_upsampling = temporal_upsampler_api.normalize_temporal_upsampling_value(inputs.get("temporal_upsampling", ""))
            spatial_upsampling = upsampler_api.normalize_upsampling_value(inputs.get("spatial_upsampling", ""))
            inputs["temporal_upsampling"] = temporal_upsampling
            inputs["spatial_upsampling"] = spatial_upsampling
            validation_error = ""
            if not media_source_exists(video_source):
                validation_error = "Selected media file is missing"
            elif source_is_mesh:
                validation_error = mesh_postprocessing.validate_late(spatial_upsampling, temporal_upsampling, parameter_ui.settings_runtime_values(inputs, parameter_ui.SPATIAL), parameter_ui.settings_runtime_values(inputs, parameter_ui.TEMPORAL), video_source)
            elif not (source_is_video or source_is_image):
                validation_error = "Post processing is only available with Videos, Images or 3D Meshes"
            else:
                validation_error = temporal_upsampler_api.validate_temporal_upsampling(temporal_upsampling, source_is_image=source_is_image)
            if not validation_error and not source_is_mesh:
                validation_error = upsampler_api.validate_postprocessing_spatial_upsampling(spatial_upsampling, 1 if source_is_image else 0)
            if not validation_error and not source_is_mesh:
                from postprocessing.film_grain import is_film_grain_enabled
                if len(temporal_upsampling) == 0 and len(spatial_upsampling) == 0 and not is_film_grain_enabled(inputs.get("film_grain_intensity", 0)):
                    validation_error = "You must choose at least one Post Processing Method"
            if validation_error:
                gr.Info(validation_error)
                return ret()
            if len(spatial_upsampling) > 0: prompt += [upsampler_api.format_upsampling_label(spatial_upsampling)]
            if len(temporal_upsampling) >0: prompt += [temporal_upsampler_api.format_temporal_upsampling_label(temporal_upsampling) if source_is_mesh else "Temporal Upsampling"]
            film_grain_intensity  = inputs.get("film_grain_intensity",0)
            film_grain_saturation  = inputs.get("film_grain_saturation",0.5)        
            # if film_grain_intensity >0: prompt += [f"Film Grain: intensity={film_grain_intensity}, saturation={film_grain_saturation}"]
            if film_grain_intensity >0: prompt += ["Film Grain"]
        elif mode =="edit_remux":
            postprocess_audio = inputs.get("postprocess_audio", "") or ""
            postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
            video_source = inputs["video_source"]
            validation_error = ""
            if not media_source_exists(video_source):
                validation_error = "Selected video or image file is missing"
            elif not has_video_file_extension(video_source):
                validation_error = "Audio remuxing is only available with Videos"
            elif postprocess_audio and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"]:
                validation_error = audio_processor_api.validate_method(
                    postprocess_audio,
                    audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK,
                    video_source=video_source,
                    audio_source=inputs.get("audio_source"),
                    media_source_exists=media_source_exists,
                    has_audio_file_extension=has_audio_file_extension,
                )
            elif postprocess_audio and audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT in postprocess_audio_meta["types"]:
                validation_error = audio_processor_api.validate_method(
                    postprocess_audio,
                    audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT,
                    voice_sample=inputs.get("replace_voice_sample"),
                    voice_sample2=inputs.get("replace_voice_sample2"),
                )
            else:
                validation_error = "You must choose at least one Remux Method"
            if validation_error:
                gr.Info(validation_error)
                return ret()
            repeat_generation= inputs.get("repeat_generation",1)
            audio_source = inputs["audio_source"]
            if audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"]:
                prompt += [audio_processor_api.format_method_label(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING)]
                if not postprocess_audio_meta["needs_audio_source"]:
                    audio_source = None
                    inputs["audio_source"] = audio_source
                if not postprocess_audio_meta["supports_repeat"]:
                    repeat_generation = 1
            elif audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT in postprocess_audio_meta["types"]:
                prompt += [audio_processor_api.format_method_label(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING)]
                audio_source = None
                inputs["audio_source"] = audio_source
                repeat_generation = 1
            else:
                gr.Info(f"Unsupported remux method: {postprocess_audio}")
                return ret()
            seed = inputs.get("seed",None)
        inputs["repeat_generation"] = repeat_generation
        if len(prompt) == 0:
            gr.Info("No edit action selected")
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
    prompt_history = prompt_parser.parse_prompt_history(inputs["prompt"], "", multi_prompts_gen_type)
    queued_prompts = prompts.copy()
    if prompt_history is not None:
        original_prompts = prompt_enhancer_chaining.originals_for_outputs(prompt_history[0], len(prompts))
        queued_prompts = [prompt_parser.serialize_prompt_blocks_with_prefix([one_prompt], [original_prompts[idx]]) for idx, one_prompt in enumerate(prompts)]
    model_def = get_model_def(model_type)
    alt_prompt_text = str(inputs.get("alt_prompt") or "")
    alt_prompt_has_history = alt_prompt_text.startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    auto_dependent_alt_prompt = server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0 and prompt_enhancer_chaining.requires_alt_prompt_inheritance(inputs.get("prompt_enhancer", ""))
    queued_alt_prompts = [alt_prompt_text] * len(prompts)
    if model_def.get("alt_prompt_inherits_prompt_paragraphs", False) and (alt_prompt_has_history or auto_dependent_alt_prompt):
        alt_prompt_history = prompt_parser.parse_prompt_history(alt_prompt_text, "", multi_prompts_gen_type)
        alt_prompts = alt_prompt_history[1] if alt_prompt_history is not None else (prompt_parser.split_prompt_units(alt_prompt_text, multi_prompts_gen_type) or [""])
        alt_prompts = prompt_enhancer_chaining.repeat_evenly(alt_prompts, len(prompts))
        if alt_prompt_history is None:
            queued_alt_prompts = alt_prompts
        else:
            original_alt_prompts = prompt_enhancer_chaining.repeat_evenly(alt_prompt_history[0], len(prompts))
            queued_alt_prompts = [prompt_parser.serialize_prompt_blocks_with_prefix([one_prompt], [original_alt_prompts[idx]]) for idx, one_prompt in enumerate(alt_prompts)]

    if "W" not in multi_prompts_gen_type:
        image_slots = image_start if image_start != None and len(image_start) > 0 else image_end
        if image_slots != None and len(image_slots) > 0:
            if inputs["multi_images_gen_type"] == 0:
                new_prompts = []
                new_queued_prompts = []
                new_queued_alt_prompts = []
                new_image_start = []
                new_image_end = []
                for i in range(len(prompts) * len(image_slots)):
                    new_prompts.append(  prompts[ i % len(prompts)] )
                    new_queued_prompts.append(queued_prompts[i % len(queued_prompts)])
                    new_queued_alt_prompts.append(queued_alt_prompts[i % len(queued_alt_prompts)])
                    if image_start != None:
                        new_image_start.append(image_start[i // len(prompts)] )
                    if image_end != None:
                        new_image_end.append(image_end[i // len(prompts)] )
                prompts = new_prompts
                queued_prompts = new_queued_prompts
                queued_alt_prompts = new_queued_alt_prompts
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
                    new_queued_prompts = []
                    new_queued_alt_prompts = []
                    for i, _ in enumerate(image_slots):
                        new_prompts.append(  prompts[ i//rep] )
                        new_queued_prompts.append(queued_prompts[i//rep])
                        new_queued_alt_prompts.append(queued_alt_prompts[i//rep])
                    prompts = new_prompts
                    queued_prompts = new_queued_prompts
                    queued_alt_prompts = new_queued_alt_prompts
            if image_start == None or len(image_start) == 0:
                image_start = [None] * len(prompts)
            if image_end == None or len(image_end) == 0:
                image_end = [None] * len(prompts)

            for queued_prompt, queued_alt_prompt, start, end in zip(queued_prompts, queued_alt_prompts, image_start, image_end) :
                inputs.update({
                    "prompt" : queued_prompt,
                    "alt_prompt": queued_alt_prompt,
                    "image_start": start,
                    "image_end" : end,
                })
                add_video_task(**inputs)
        else:
            for queued_prompt, queued_alt_prompt in zip(queued_prompts, queued_alt_prompts):
                inputs["prompt"] = queued_prompt
                inputs["alt_prompt"] = queued_alt_prompt
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

def get_custom_setting_slider_key(index):
    return f"custom_setting_slider_{index + 1}"

def get_custom_setting_dropdown_key(index):
    return f"custom_setting_dropdown_{index + 1}"

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

def get_custom_setting_dropdown_choices(setting_def):
    if not isinstance(setting_def, dict) or setting_def.get("type") != "dropdown":
        return None
    choices = setting_def.get("choices", [])
    if not isinstance(choices, list):
        return None
    normalized = []
    for choice in choices:
        if isinstance(choice, (list, tuple)) and len(choice) >= 2:
            normalized.append((str(choice[0]), choice[1]))
        else:
            normalized.append((str(choice), choice))
    return normalized if len(normalized) > 0 else None

def get_custom_setting_dropdown_value(raw_value, choices):
    if not choices:
        return raw_value
    values = [value for _, value in choices]
    for value in values:
        if raw_value == value or str(raw_value) == str(value):
            return value
    return values[0]

def get_custom_setting_display_value(setting_def, raw_value):
    choices = get_custom_setting_dropdown_choices(setting_def)
    if choices is not None:
        for label, value in choices:
            if raw_value == value or str(raw_value) == str(value):
                return label
    return raw_value

def get_model_custom_settings(model_def):
    if not isinstance(model_def, dict):
        return []
    custom_settings = model_def.get("custom_settings", [])
    if not isinstance(custom_settings, list):
        return []
    normalized = []
    used_ids = set()
    dropdown_count = 0
    for idx, setting in enumerate(custom_settings[:CUSTOM_SETTINGS_MAX]):
        if not isinstance(setting, dict):
            continue
        one = setting.copy()
        one["label"] = str(one.get("label", f"Custom Setting {idx + 1}"))
        one["name"] = str(one.get("name", f"Custom Setting {idx + 1}"))
        one["type"] = _normalize_custom_setting_type(one.get("type", "text"))
        if one["type"] == "dropdown":
            dropdown_count += 1
            if dropdown_count > CUSTOM_SETTING_DROPDOWN_MAX or get_custom_setting_dropdown_choices(one) is None:
                one["type"] = "text"
        setting_id = get_custom_setting_id(one, idx)
        if setting_id in used_ids:
            setting_id = get_custom_setting_key(idx)
        used_ids.add(setting_id)
        one["id"] = setting_id
        normalized.append(one)
    return normalized

def apply_custom_settings_defaults(model_def, ui_defaults):
    custom_settings = ui_defaults.get("custom_settings")
    for setting_def in get_model_custom_settings(model_def):
        if "default" not in setting_def:
            continue
        if not isinstance(custom_settings, dict):
            custom_settings = {}
            ui_defaults["custom_settings"] = custom_settings
        custom_settings.setdefault(setting_def["id"], copy.deepcopy(setting_def["default"]))

def get_custom_setting_slider_bounds(setting_def):
    if not isinstance(setting_def, dict) or setting_def.get("type") not in {"int", "float"} or not all(key in setting_def for key in ("min", "max", "inc")):
        return None
    try:
        min_value, max_value, step_value = float(setting_def["min"]), float(setting_def["max"]), float(setting_def["inc"])
    except Exception:
        return None
    if max_value < min_value or step_value <= 0:
        return None
    if setting_def.get("type") == "int":
        if not min_value.is_integer() or not max_value.is_integer() or not step_value.is_integer():
            return None
        return int(min_value), int(max_value), int(step_value)
    return min_value, max_value, step_value

def get_custom_setting_slider_value(raw_value, slider_bounds):
    min_value, max_value, _ = slider_bounds
    try:
        value = float(raw_value)
    except Exception:
        value = min_value
    return min(max(value, min_value), max_value)


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


def custom_setting_mode_visible(setting_def, model_mode):
    if not isinstance(setting_def, dict): return False
    for key, excluded in (("model_mode", False), ("model_mode_not", True)):
        if key in setting_def:
            modes = setting_def[key]
            if not isinstance(modes, (list, tuple)): modes = [modes]
            if (model_mode in modes) == excluded: return False
    return True


def custom_setting_visible(setting_def, video_prompt_type="", audio_prompt_type="", model_mode=None):
    """Optional exact mode matches further restrict the existing prompt-flag conditions."""
    if not custom_setting_mode_visible(setting_def, model_mode): return False
    vflags, aflags = str(setting_def.get("video_prompt_type", "") or ""), str(setting_def.get("audio_prompt_type", "") or "")
    vflags_not, aflags_not = str(setting_def.get("video_prompt_type_not", "") or ""), str(setting_def.get("audio_prompt_type_not", "") or "")
    if vflags_not and any_letters(video_prompt_type or "", vflags_not): return False
    if aflags_not and any_letters(audio_prompt_type or "", aflags_not): return False
    if not (vflags or aflags): return True
    return bool(vflags and any_letters(video_prompt_type or "", vflags)) or bool(aflags and any_letters(audio_prompt_type or "", aflags))


def compatible_custom_setting_visible(visible):
    # Check the actual plugin wrapper once; functools.wraps can advertise the core signature.
    signature = inspect.signature(visible, follow_wrapped=False)
    try:
        signature.bind(None, "", "", None)
    except TypeError:
        signature.bind(None, "", "")
    else:
        return visible

    @wraps(visible)
    def compatible(setting_def, video_prompt_type="", audio_prompt_type="", model_mode=None):
        if not custom_setting_mode_visible(setting_def, model_mode): return False
        # Apply mode conditions here, before a legacy wrapper forwards to core without model_mode.
        if "model_mode" in setting_def or "model_mode_not" in setting_def:
            setting_def = {key: value for key, value in setting_def.items() if key not in ("model_mode", "model_mode_not")}
        return visible(setting_def, video_prompt_type, audio_prompt_type)

    return compatible


def refresh_custom_settings_visibility(state, video_prompt_type, audio_prompt_type, model_mode):
    model_def = get_model_def(get_state_model_type(state))
    custom_settings = get_model_custom_settings(model_def)
    defs = [custom_settings[i] if i < len(custom_settings) else None for i in range(CUSTOM_SETTINGS_MAX)]
    visible = [custom_setting_visible(setting, video_prompt_type, audio_prompt_type, model_mode) for setting in defs]
    slider_settings = [get_custom_setting_slider_bounds(setting) is not None for setting in defs]
    dropdown_settings = [get_custom_setting_dropdown_choices(setting) is not None for setting in defs]
    text_settings = [setting is not None and not slider_settings[idx] and not dropdown_settings[idx] for idx, setting in enumerate(defs)]
    row_updates = [gr.update(visible=any(visible[row_idx * CUSTOM_SETTINGS_PER_ROW:min((row_idx + 1) * CUSTOM_SETTINGS_PER_ROW, CUSTOM_SETTINGS_MAX)])) for row_idx in range(math.ceil(CUSTOM_SETTINGS_MAX / CUSTOM_SETTINGS_PER_ROW))]
    guide_update = gr.update(visible=custom_setting_visible(model_def.get("custom_guide"), video_prompt_type, audio_prompt_type, model_mode))
    return row_updates + [gr.update(visible=visible[idx] and text_settings[idx]) for idx in range(CUSTOM_SETTINGS_MAX)] + [gr.update(visible=visible[idx] and slider_settings[idx]) for idx in range(CUSTOM_SETTINGS_MAX)] + [gr.update(visible=visible[idx] and dropdown_settings[idx]) for idx in range(CUSTOM_SETTINGS_MAX)] + [guide_update]


def parse_custom_setting_typed_value(raw_value, setting_type, setting_def=None):
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if len(raw_value) == 0:
            return None, None
    setting_type = _normalize_custom_setting_type(setting_type)
    if setting_type == "dropdown":
        choices = get_custom_setting_dropdown_choices(setting_def)
        if choices is None:
            return str(raw_value).strip(), None
        for _, value in choices:
            if raw_value == value or str(raw_value) == str(value):
                return value, None
        return None, "Expected one of the dropdown choices."
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
        if strict and not custom_setting_visible(setting_def, inputs["video_prompt_type"], inputs["audio_prompt_type"], inputs["model_mode"]):
            continue
        slot_key = get_custom_setting_key(idx)
        slider_key = get_custom_setting_slider_key(idx)
        dropdown_key = get_custom_setting_dropdown_key(idx)
        setting_id = setting_def["id"]
        if get_custom_setting_dropdown_choices(setting_def) is not None:
            raw_value = inputs.get(dropdown_key, None)
        elif get_custom_setting_slider_bounds(setting_def) is not None:
            raw_value = inputs.get(slider_key, None)
        else:
            raw_value = inputs.get(slot_key, None)
        if raw_value is None and setting_id in existing_custom_settings:
            raw_value = existing_custom_settings.get(setting_id, None)
        parsed_value, parse_error = parse_custom_setting_typed_value(raw_value, setting_def.get("type", "text"), setting_def)
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
        inputs.pop(get_custom_setting_slider_key(idx), None)
        inputs.pop(get_custom_setting_dropdown_key(idx), None)

def parse_custom_sigmas(value, phases):
    """UI/CLI boundary: one comma-separated schedule per phase; empty means implicit."""
    if not isinstance(value, str):
        raise ValueError("Custom Sigmas must be text: comma-separated numbers, with ';' between phases.")
    if not value.strip():
        return [None] * phases
    slots = value.split(";")
    if len(slots) != phases:
        raise ValueError(f"Custom Sigmas requires {phases} phase slots separated by {phases - 1} ';' characters. Leave a slot blank for its default schedule.")
    schedules = []
    for phase, slot in enumerate(slots, 1):
        if not slot.strip():
            schedules.append(None)
            continue
        try:
            sigmas = [float(token.strip()) for token in slot.split(",")]
        except ValueError:
            raise ValueError(f"Custom Sigmas phase {phase}: enter numbers separated by commas.") from None
        if any(not math.isfinite(sigma) or not 0 <= sigma < 1 for sigma in sigmas):
            raise ValueError(f"Custom Sigmas phase {phase}: values must be finite numbers from 0 (inclusive) to 1 (exclusive).")
        if any(a <= b for a, b in zip(sigmas, sigmas[1:])):
            raise ValueError(f"Custom Sigmas phase {phase}: values must be in strictly descending order.")
        if sigmas[-1] == 0:
            sigmas.pop()  # the sampler adds its terminal zero; it is not a denoising step
        if not sigmas:
            raise ValueError(f"Custom Sigmas phase {phase}: provide at least one positive sigma, or leave the slot blank.")
        schedules.append(sigmas)
    return schedules


def validate_settings(state, model_type, single_prompt, inputs, silent=False):
    def err(error=""):
        error = str(error or "")
        if len(error) > 0 and not silent:
            gr.Info(error)
        return None, None, None, None, error

    unknown_error = settings_error(inputs)
    if unknown_error:
        return err(unknown_error)
    model_def = get_model_def(model_type)
    overrides_error = get_overrides_error(model_def, inputs["overrides"])
    if overrides_error:
        return err(overrides_error)
    model_handler = get_model_handler(model_type)
    if not (model_def.get("audio_only") or model_def.get("mesh_outputs")):
        try:
            resolution_utils.validate_resolution(model_def, inputs.get("resolution"))
        except ValueError as error:
            return err(str(error))
    image_outputs = inputs["image_mode"] > 0
    if image_outputs:
        image_batch_size_max = max(1, int(model_def.get("image_batch_size_max", 16)))
        if int(inputs.get("batch_size", 1) or 1) > image_batch_size_max:
            return err(f"This model supports a maximum of {image_batch_size_max} image{'s' if image_batch_size_max > 1 else ''} per generation.")
    is_edit_mode = str(inputs.get("mode", "") or "").startswith("edit_")
    any_steps_skipping = model_def.get("tea_cache", False) or model_def.get("mag_cache", False) or model_def.get("spectrum_cache", False) or model_def.get("first_block_cache", False)
    model_type = get_base_model_type(model_type)

    model_filename = get_model_filename(model_type)  


    if inputs.get("cfg_star_switch", 0) != 0 and inputs.get("apg_switch", 0) != 0:
        return err("Adaptive Progressive Guidance and Classifier Free Guidance Star can not be set at the same time")
    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]
    prompt = inputs["prompt"]
    keep_empty_lines = model_def.get("preserve_empty_prompt_lines", False) or "P" in multi_prompts_gen_type or prompt_parser.PROMPT_UNIT_PREFIX in prompt
    if not model_def.get("skip_prompt_template", False):
        prompt, errors = prompt_parser.process_template(prompt, keep_comments=prompt_parser.PROMPT_UNIT_PREFIX in prompt, keep_empty_lines=keep_empty_lines)
        if len(errors) > 0:
            return err("Error processing prompt template: " + errors)
    prompt = prompt.strip("\n").strip()

    prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
    if len(prompts) == 0:
        if not model_def.get("prompt_optional", False):
            return err("Prompt cannot be empty.")
        prompts = [""]  # the prompt only names the output files
    if single_prompt and multi_prompts_gen_type in {"G", "PG"} and len(prompts) > 1:
        return err(f"multi_prompts_gen_type='{multi_prompts_gen_type}' parses this prompt into {len(prompts)} separate generation requests, but this submission accepts one generation task. Submit separate tasks or use 'FG' if the line breaks belong to one prompt.")
    window_boundary_error = prompt_parser.validate_sliding_window_prompt_boundaries(prompts, multi_prompts_gen_type, inputs.get("image_end"))
    if window_boundary_error:
        return err(window_boundary_error)
    validation_prompts = prompts
    frames_minimum, frames_steps, latent_size = get_model_min_frames_and_step(model_type)
    discard_step = model_def.get("sliding_window_defaults", {}).get("discard_step", latent_size)  # 1 for models that recycle no latents
    inputs.pop("frame_scheduler", None)
    frame_scheduler = None
    scheduler_supported = frame_scheduler_supported(model_type, model_def, inputs.get("image_mode", 0), is_edit_mode)
    if not scheduler_supported and has_slash_commands(prompts):
        return err("Prompt slash window commands require a video model with Sliding Window support.")
    if scheduler_supported:
        schedule_fps = get_computed_fps(inputs.get("force_fps", ""), model_type, inputs.get("video_guide"), inputs.get("video_source"))
        frame_scheduler, frame_scheduler_error = build_frame_scheduler(
            prompts,
            total_frames=int(inputs.get("video_length", frames_minimum) or frames_minimum),
            fps=float(schedule_fps),
            window_size=int(inputs.get("sliding_window_size", inputs.get("video_length", frames_minimum)) or frames_minimum),
            default_overlap=int(inputs.get("sliding_window_overlap", 0) or 0),
            minimum=frames_minimum,
            step=frames_steps,
            frame_offset=model_def.get("frames_offset", 1),
            overlap_offset=model_def.get("sliding_window_defaults", {}).get("overlap_offset", 1),
            max_overlap=model_def.get("sliding_window_defaults", {}).get("overlap_max"),
            preserve_exact_output_frames=model_def.get("image_end_frame_position", False),
            output_frame_policy=model_def.get("frame_scheduler_output_policy"),
            supported_model_commands=model_def.get("prompt_slash_commands", []),
            allow_new_shot=image_prompt_types_allow_t2v(model_def, inputs.get("image_mode", 0)),
            first_window_overlap_frames=estimate_first_window_overlap_frames(inputs.get("image_start"), inputs.get("video_source"), inputs.get("keep_frames_video_source", ""), schedule_fps),
            initial_shared_frames=1 if inputs.get("image_start") is not None or inputs.get("video_source") is not None else 0,
            discard_last_frames=int(inputs.get("sliding_window_discard_last_frames", 0) or 0) // discard_step * discard_step,
        )
        if frame_scheduler_error is not None:
            return err(frame_scheduler_error)
        validation_prompts = frame_scheduler["prompts"]
        inputs["frame_scheduler"] = frame_scheduler
    inputs["prompt"] = prompt if prompt.startswith(prompt_parser.PROMPT_UNIT_PREFIX) else prompt_parser.serialize_prompt_units(prompt, prompts, multi_prompts_gen_type)

    parsed_custom_settings, custom_settings_error = collect_custom_settings_from_inputs(model_def, inputs, strict=True)
    if custom_settings_error is not None:
        return err(custom_settings_error)
    inputs["custom_settings"] = parsed_custom_settings
    clear_custom_setting_slots(inputs)
    inputs["guidance_phases"], inputs["video_prompt_type"] = normalize_phase_2_tiling_selection(model_def, inputs["guidance_phases"], inputs["video_prompt_type"])
    sigma_overrides = set()
    if model_def.get("custom_sigmas", False):
        try:
            sigma_schedules = parse_custom_sigmas(inputs.get("custom_sigmas", ""), max(1, int(inputs["guidance_phases"])))
        except ValueError as error:
            return err(str(error))
        compatible_schedulers = model_def.get("custom_sigmas_schedulers")
        if any(schedule is not None for schedule in sigma_schedules) and compatible_schedulers is not None and inputs["sample_solver"] not in compatible_schedulers:
            return err("The selected scheduler is not compatible with Custom Sigmas. Clear Custom Sigmas or select a compatible scheduler: " + ", ".join(scheduler or "Default" for scheduler in compatible_schedulers) + ".")
        continuous_sigmas = model_def.get("custom_sigmas_continuous", False)
        sigma_switch_scale = model_def.get("custom_sigmas_switch_scale", 1.0)
        for index, sigmas in enumerate(sigma_schedules):
            if sigmas is None:
                continue
            if index == 0 and not continuous_sigmas:
                inputs["num_inference_steps"] = len(sigmas)
                sigma_overrides.add("num_inference_steps")
            elif index == 1 and model_def.get("inference_steps2", False):
                inputs["num_inference_steps2"] = len(sigmas)
                sigma_overrides.add("num_inference_steps2")
            if index > 0 and sigma_switch_scale is not None:
                threshold = "switch_threshold" if index == 1 else f"switch_threshold{index}"
                effective_switch = sigmas[0] * sigma_switch_scale
                if not math.isclose(float(inputs[threshold]), effective_switch, rel_tol=1e-6, abs_tol=1e-6 * sigma_switch_scale) and not silent:
                    gr.Info(f"Phase {index + 1} uses starting sigma {sigmas[0]:g} from Custom Sigmas; the phase switch setting is ignored.")
                inputs[threshold] = effective_switch
                sigma_overrides.add(threshold)
        if continuous_sigmas and any(schedule is not None for schedule in sigma_schedules):
            boundaries = [1.0] + [inputs["switch_threshold" if index == 1 else f"switch_threshold{index}"] / sigma_switch_scale for index in range(1, len(sigma_schedules))] + [0.0]
            if any(a <= b for a, b in zip(boundaries, boundaries[1:])):
                return err("Custom Sigmas requires phase switches in descending order between 0 and 1.")
            for index, sigmas in enumerate(sigma_schedules):
                if sigmas is not None and (sigmas[0] > boundaries[index] or sigmas[-1] <= boundaries[index + 1]):
                    return err(f"Custom Sigmas phase {index + 1} crosses a phase switch. This model uses one descending schedule across all phases.")
    # Explicit schedules have already validated their derived counts and switches.
    # Slider ranges constrain manual values, not custom sigma values (e.g. H3 < 0.7).
    extra_settings_error = extra_settings.validate_inputs({key: value for key, value in inputs.items() if key not in sigma_overrides}, model_def, get_max_frames=get_max_frames)
    if len(extra_settings_error) > 0:
        return err(extra_settings_error)

    if hasattr(model_handler, "validate_generative_prompt"):
        for one_prompt in validation_prompts:
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
    audio_guide3 = inputs["audio_guide3"]
    audio_source = inputs["audio_source"]
    replace_voice_method = audio_processor_api.normalize_method(inputs.get("replace_voice_method", "") or "")
    replace_voice_sample = inputs.get("replace_voice_sample", None)
    replace_voice_sample2 = inputs.get("replace_voice_sample2", None)
    video_guide = inputs["video_guide"]
    video_guide2 = inputs["video_guide2"]
    video_guide3 = inputs["video_guide3"]
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
    skip_steps_multiplier = inputs["skip_steps_multiplier"]
    postprocess_audio = inputs.get("postprocess_audio", "") or ""
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
    spatial_upsampling = upsampler_api.normalize_upsampling_value(inputs["spatial_upsampling"])
    temporal_upsampling = temporal_upsampler_api.normalize_temporal_upsampling_value(inputs.get("temporal_upsampling", ""))
    inputs["spatial_upsampling"] = spatial_upsampling
    inputs["temporal_upsampling"] = temporal_upsampling
    motion_amplitude = inputs["motion_amplitude"]
    self_refiner_setting = inputs["self_refiner_setting"]
    self_refiner_plan = inputs["self_refiner_plan"]
    model_mode = inputs["model_mode"]
    if image_mode == 0 and model_def.get("image_outputs", False): image_mode = 1
    medium = "Videos" if image_mode == 0 else "Images"
    prompt_enhancer_choices, prompt_enhancer_default, prompt_enhancer_def = get_prompt_enhancer_choices(model_def, model_def.get("audio_only", False), image_mode, include_disabled=True, video_prompt_type=inputs["video_prompt_type"], inputs=inputs)
    prompt_enhancer_values = [value for _, value in prompt_enhancer_choices]
    prompt_enhancer_mode = prompt_enhancer_chaining.normalize_choice(str(inputs.get("prompt_enhancer") or ""), prompt_enhancer_values, prompt_enhancer_default)
    inputs["prompt_enhancer"] = build_prompt_enhancer_value(prompt_enhancer_mode, "K" in str(inputs.get("prompt_enhancer") or "") and len(prompt_enhancer_mode) > 0)
    inheritance_error = prompt_enhancer_chaining.validate_alt_prompt_inheritance(prompt_enhancer_mode, len(prompts), model_def.get("alt_prompt_inherits_prompt_paragraphs", False)) if server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0 else ""
    if inheritance_error:
        return err(inheritance_error)
    alt_prompt_has_history = str(inputs.get("alt_prompt") or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    auto_dependent_alt_prompt = server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0 and prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer_mode)
    if model_def.get("alt_prompt_inherits_prompt_paragraphs", False) and (alt_prompt_has_history or auto_dependent_alt_prompt):
        alt_prompts = prompt_parser.split_prompt_units(str(inputs.get("alt_prompt") or ""), multi_prompts_gen_type) or [""]
        alt_prompt_count_error = prompt_enhancer_chaining.validate_alt_prompt_count(len(prompts), len(alt_prompts), model_def["prompt_class"], model_def["alt_prompt"]["label"])
        if alt_prompt_count_error:
            return err(alt_prompt_count_error)

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
    guide_custom_choices = get_guide_custom_choices(model_def, image_mode)
    if image_mode > 0 and guide_custom_choices is not None:
        guide_custom_value = filter_letters(video_prompt_type, guide_custom_choices["letters_filter"])
        if len(guide_custom_value) > 0 and guide_custom_value not in {value for _, value in guide_custom_choices["choices"]}:
            video_prompt_type = del_in_sequence(video_prompt_type, guide_custom_choices["letters_filter"])
            inputs["video_prompt_type"] = video_prompt_type
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
    if model_def.get("mesh_outputs", False):  # the 3D refiner (spatial slot) and the texturer (temporal slot) of the generated meshes
        mesh_postprocessing_error = mesh_postprocessing.validate(model_def, inputs, spatial_upsampling, temporal_upsampling)
        if mesh_postprocessing_error:
            return err(mesh_postprocessing_error)
    else:
        if spatial_upsampling and upsampler_api.find_upsampler(spatial_upsampling) is None:
            return err(f"Spatial upsampling method '{spatial_upsampling}' is not supported by the current WanGP install")
        temporal_upsampling_error = temporal_upsampler_api.validate_temporal_upsampling(temporal_upsampling, source_is_image=image_mode > 0)
        if temporal_upsampling_error:
            return err(f"Temporal upsampling method '{temporal_upsampling}' is not supported by the current WanGP install" if temporal_upsampler_api.find_temporal_upsampler(temporal_upsampling) is None else temporal_upsampling_error)
        vae_upsampling_error = upsampler_api.validate_model_vae_upsampling(spatial_upsampling, image_mode, model_type, model_def, medium)
        if vae_upsampling_error:
            return err(vae_upsampling_error)
        edit_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
        if edit_upsampler is not None:
            edit_upsampling_error = edit_upsampler.validate_upsampling(spatial_upsampling, image_mode)
            if edit_upsampling_error:
                return err(edit_upsampling_error)

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
                          
    lora_multiplier_phases = int(model_def.get("lora_multiplier_phases", guidance_phases) or guidance_phases or 1)
    lora_multiplier_branches = model_def.get("lora_multiplier_branches", None)
    if len(loras_multipliers) > 0:
        _, _, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases= lora_multiplier_phases, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: 
            return err(f"Error parsing Loras Multipliers: {errors}")
    loras_mult_error = prepare_loras_mult_windows(frame_scheduler, activated_loras, num_inference_steps, lora_multiplier_phases, lora_multiplier_branches=lora_multiplier_branches)
    if loras_mult_error is not None:
        return err(loras_mult_error)
    if guidance_phases == 3:
        if switch_threshold < switch_threshold2:
            return err(f"Phase 1-2 Switch Noise Level ({switch_threshold}) should be Greater than Phase 2-3 Switch Noise Level ({switch_threshold2}). As a reminder, noise will gradually go down from 1000 to 0.")
    else:
        model_switch_phase = 1
        
    if not any_steps_skipping: skip_steps_cache_type = ""
    supported_cache_types = {""}
    if model_def.get("tea_cache", False): supported_cache_types.add("tea")
    if model_def.get("mag_cache", False): supported_cache_types.add("mag")
    if model_def.get("spectrum_cache", False): supported_cache_types.add("spectrum")
    if model_def.get("first_block_cache", False): supported_cache_types.add("first_block")
    if skip_steps_cache_type not in supported_cache_types:
        return err(f"This model does not support step-skipping type '{skip_steps_cache_type}'.")
    if skip_steps_cache_type == "first_block" and float(skip_steps_multiplier) not in model_def["first_block_cache_thresholds"]:
        return err(f"Unsupported First Block Cache threshold '{skip_steps_multiplier}'.")
    if not model_def.get("lock_inference_steps", False) and model_type in ["ltxv_13B"] and num_inference_steps < 20:
        return err("The minimum number of steps should be 20")
    if skip_steps_cache_type == "mag":
        if num_inference_steps > 50:
            return err("Mag Cache maximum number of steps is 50")
        
    if image_mode > 0:
        audio_prompt_type = ""
        postprocess_audio = ""
        replace_voice_method = ""
        replace_voice_sample = None
        replace_voice_sample2 = None
    postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
    replace_voice_meta = audio_processor_api.method_metadata(replace_voice_method)

    if postprocess_audio and postprocess_audio != "control" and audio_processor_api.find_processor(postprocess_audio) is None:
        return err(f"Audio processing method '{postprocess_audio}' is not supported by the current WanGP install")
    if replace_voice_method and audio_processor_api.find_processor(replace_voice_method) is None:
        return err(f"Audio processing method '{replace_voice_method}' is not supported by the current WanGP install")

    if "K" in audio_prompt_type and "V" not in video_prompt_type:
        return err("You must enable a Control Video to use the Control Video Audio Track as an audio prompt")

    if (model_def.get("multitalk_class", False) or model_def.get("speaker_locations", False)) and ("B" in audio_prompt_type or "X" in audio_prompt_type) and not model_def.get("one_speaker_only", False):
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
        if len(error) > 0:
            return err(error)

    if postprocess_audio and postprocess_audio != "control" and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK not in postprocess_audio_meta["types"]:
        return err(f"{postprocess_audio_meta['label']} is not available as a soundtrack processor")
    if "F" in video_prompt_type:
        if len(frames_positions.strip()) > 0:
            positions = frames_positions.replace(","," ").split(" ")
            for pos_str in positions:
                if not pos_str.upper() in ["L", "X"] and len(pos_str)>0:
                    if not is_integer(pos_str):
                        return err(f"Invalid Frame Position '{pos_str}'")
                    pos = int(pos_str)
                    if pos <1 or pos > max_source_video_frames:
                        return err(f"Invalid Frame Position Value'{pos_str}'")
    else:
        frames_positions = None

    if postprocess_audio and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"]:
        soundtrack_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, audio_source=audio_source, media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension)
        if soundtrack_error:
            return err(soundtrack_error)
        if not postprocess_audio_meta["needs_audio_source"]:
            audio_source = None
    else:
        audio_source = None
    if replace_voice_method:
        replace_voice_error = audio_processor_api.validate_method(replace_voice_method, audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, voice_sample=replace_voice_sample, voice_sample2=replace_voice_sample2)
        if replace_voice_error:
            return err(replace_voice_error)
    else:
        replace_voice_sample = None
        replace_voice_sample2 = None
    if not replace_voice_meta["needs_voice_sample2"]:
        replace_voice_sample2 = None
    if len(filter_letters(image_prompt_type, "VLG")) > 0 and len(keep_frames_video_source) > 0:
        if not is_integer(keep_frames_video_source) or int(keep_frames_video_source) == 0:
            return err("The number of frames to keep must be a non null integer")
    else:
        keep_frames_video_source = ""

    if image_outputs:
        image_prompt_type = image_prompt_type.replace("V", "").replace("L", "")
    custom_guide_def = model_def.get("custom_guide", None)
    if custom_setting_visible(custom_guide_def, video_prompt_type, audio_prompt_type, model_mode):
        if custom_guide is None and custom_guide_def.get("required", False):
            return err(f"You must provide a {custom_guide_def.get('label', 'Custom Guide')}")
    else:
        custom_guide = None

    if not is_edit_mode:
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
    if "D" in audio_prompt_type:
        if audio_guide3 == None:
            return err("You must provide a third Audio Source")
    else:
        audio_guide3 = None
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
        image_refs = clean_image_list(image_refs, keep_alpha=model_def.get("image_refs_keep_alpha", False))
        if image_refs == None :
            return err("A Reference Image should be an Image")
        if model_def.get("one_image_ref_only", False) and (model_def.get("one_image_ref_only_with_background", False) or not any_letters(video_prompt_type, "KF")) and len(image_refs) > 1:
            return err("Only one Reference Image is supported by this model mode")
    else:
        image_refs = None

    video_guides = video_guides_shown(model_def, inputs["image_mode"])
    if "V" in video_prompt_type:
        if not video_guides:
            if image_guide is None:
                return err("You must provide a Control Image")
        else:
            if video_guide is None:
                return err("You must provide a Control Video")
        if "+" in video_prompt_type and video_guide2 is None:
            return err("You must provide a second Control Video")
        if "*" in video_prompt_type and video_guide3 is None:
            return err("You must provide a third Control Video")
        if "A" in video_prompt_type and not "U" in video_prompt_type:             
            if not video_guides:
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
        video_guide2 = None
        video_guide3 = None
        image_guide = None
        video_mask = None
        image_mask = None
        keep_frames_video_guide = ""
        denoising_strength = 1.0
        masking_strength = 1.0
    
    if not video_guides:
        video_guide = None
        video_guide2 = None
        video_guide3 = None
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
        if frame_scheduler is not None and frame_scheduler["active"]:
            extra = f", which is more than the original number of frames {video_length}" if frame_scheduler["predicted_total_frames"] > video_length else ""
            gr.Info(f"{len(frame_scheduler['windows'])} Sliding Windows will be generated, for an estimated total of {frame_scheduler['predicted_total_frames']} output frames{extra}.")
        elif video_length > sliding_window_size:
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
        "audio_guide3": audio_guide3,
        "audio_source": audio_source,
        "replace_voice_method": replace_voice_method,
        "replace_voice_sample": replace_voice_sample,
        "replace_voice_sample2": replace_voice_sample2,
        "postprocess_audio": postprocess_audio,
        "video_guide": video_guide,
        "video_guide2": video_guide2,
        "video_guide3": video_guide3,
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
        "image_mode": image_mode,
    } 
    inputs.update(override_inputs)
    if hasattr(model_handler, "validate_generative_settings"):
        error = model_handler.validate_generative_settings(model_type, model_def, inputs)
        if error is not None and len(error) > 0:
            return err(error)
    if model_def.get("custom_sigmas", False):
        # A mode may force one phase, or restore its usual distilled step count.
        # Check the effective phase count and retain explicit schedule precedence.
        try:
            sigma_schedules = parse_custom_sigmas(inputs.get("custom_sigmas", ""), max(1, int(inputs["guidance_phases"])))
        except ValueError as error:
            return err(str(error))
        if sigma_schedules[0] is not None and not model_def.get("custom_sigmas_continuous", False):
            inputs["num_inference_steps"] = len(sigma_schedules[0])
    inputs.pop("frame_scheduler", None)
    if not silent:
        _, warning = resolution_utils.resolution_advice(model_def, inputs["resolution"], guidance_phases, video_prompt_type)
        if warning:
            gr.Info(warning)
    return inputs, prompts, image_start, image_end, ""


def get_preview_images(inputs):
    inputs_to_query = ["image_start", "video_source", "image_end", "video_guide", "video_guide2", "video_guide3", "image_guide", "video_mask", "image_mask", "image_refs"]
    labels = ["Start Image", "Video Source", "End Image", "Video Guide", "Video Guide 2", "Video Guide 3", "Image Guide", "Video Mask", "Image Mask", "Image Reference"]
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
        **({"workspace_id": workspace_storage.current_binding(state)[1]} if workspace_storage.current_binding(state) else {}),
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

def _unwrap_attachment_item(item):
    if isinstance(item, (tuple, list)) and len(item) > 0:
        item = item[0]
    if isinstance(item, dict):
        item = item.get("path") or item.get("name") or item.get("orig_name") or item.get("url") or item
    elif not isinstance(item, (Image.Image, str)):
        item = getattr(item, "path", None) or getattr(item, "name", None) or item
    return item

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
                    item = _unwrap_attachment_item(item)
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
            if _is_edit_task_params(params_copy):
                params_copy.pop("model_type", None)
                params_copy.pop("base_model_type", None)
            else:
                params_copy['base_model_type'] = get_base_model_type(params_copy["model_type"])

            manifest_entry = {"id": task.get('id'), "params": params_copy}
            if task.get('workspace_id'):
                manifest_entry['workspace_id'] = task['workspace_id']
            if task.get('mcp_agent_id'):
                manifest_entry['mcp_agent_id'] = task['mcp_agent_id']
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

    queue = queue[:]
    return gradio_downloads.register_download("queue.zip", "application/zip", lambda: gradio_downloads.stream_writer(lambda writer: _save_queue_to_zip(queue, writer)))

def clean_settings(model_type, params):
    saved_settings_version = params.get('settings_version', 0)
    merged = get_factory_settings(model_type)
    merged.update({k: v for k, v in params.items() if v is not None or k not in merged})
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

    preserve_edit_media_source_path = str(params.get("mode", "") or "").startswith("edit_")
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
                try:
                    final_path = gr.processing_utils.save_file_to_cache(source_path, cache_dir)
                except Exception as e:
                    print(f"{log_prefix} Error copying {filename}: {e}")
                    continue
            else:
                final_path = source_path

            # Load images as PIL, keep videos/audio as paths
            if key == "video_source" and preserve_edit_media_source_path:
                loaded_items.append(replace_virtual_media_source(filename, final_path) if virtual_spec is not None else final_path)
                print(f"{log_prefix} Using path: {final_path}")
            elif has_image_file_extension(final_path):
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


def _is_edit_task_params(params):
    return isinstance(params, dict) and str(params.get("mode", "") or "").startswith("edit_")


STATIC_AUDIO_POSTPROCESS_STATUS = {"control": "Control Audio Remuxing"}
EDIT_TASK_STATUS = {"edit_audio": ("Applying Audio Post Processing", True), "edit_remux": ("Applying Audio Remuxing", True), "edit_postprocessing": ("Applying Media Post Processing", False)}


def get_task_status_text(task):
    params = task.get("params", {}) if isinstance(task, dict) else {}
    prefix, has_audio_action = EDIT_TASK_STATUS.get(params.get("mode", ""), ("Generating...", False))
    method = audio_processor_api.normalize_method(params.get("postprocess_audio") or "")
    status = STATIC_AUDIO_POSTPROCESS_STATUS.get(method) or audio_processor_api.method_metadata(method)["status"] or "Audio Post Processing"
    return f"{prefix} - {status}" if has_audio_action else prefix


def get_task_progress_label(task):
    params = task["params"]
    if not _is_edit_task_params(params):
        return get_model_name(params["model_type"])
    if params["mode"] != "edit_postprocessing":
        return get_task_status_text(task).removeprefix("Applying ")
    labels = [temporal_upsampler_api.format_temporal_upsampling_label(params.get("temporal_upsampling")), upsampler_api.format_upsampling_label(params.get("spatial_upsampling"))]
    if params.get("film_grain_intensity", 0) > 0:
        labels.append("Film Grain")
    return ", ".join(label for label in labels if label)


def _extract_model_type(params, state, log_prefix="[load]"):
    if _is_edit_task_params(params):
        params.pop("model_type", None)
        params.pop("base_model_type", None)
        return "", None

    convert_retired_model_type(params)
    base_model_type = params.get('base_model_type', None)
    model_type = original_model_type = params.get('model_type', base_model_type)

    if model_type is not None and get_model_def(model_type) is None:
        model_type = base_model_type

    if model_type is None:
        return None, f"Unknown model type: {original_model_type}" if original_model_type is not None else "Settings must contain 'model_type'"
    params["model_type"] = model_type
    if get_model_def(model_type) is None:
        return None, f"Unknown model type: {original_model_type}"
    return model_type, None


def _parse_task_manifest(manifest, state, media_base_path, cache_dir=None, log_prefix="[load]", verbose_output = True, skip_validate_settings=False):
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

        params, error = validate_task(task_data, state, skip_validate_settings=skip_validate_settings)
        if error:
            if first_error is None:
                first_error = error
            print(f"{log_prefix} {error} for task #{task_id_loaded}. Skipping.")
            continue

        # Build runtime task
        runtime_task = _build_runtime_task(task_id_loaded, params, task_data.get('plugin_data', {}))
        if task_data.get('workspace_id'):
            runtime_task['workspace_id'] = task_data['workspace_id']
        if task_data.get('mcp_agent_id'):
            runtime_task['mcp_agent_id'] = task_data['mcp_agent_id']
        workspace_storage.capture_task(runtime_task, state)
        newly_loaded_queue.append(runtime_task)
        if verbose_output:
            task_label = params.get("mode", "") if _is_edit_task_params(params) else model_type
            print(f"{log_prefix} Task {task_index+1}/{len(manifest)} ready, ID: {task_id_loaded}, model: {task_label}")

    # Update global task_id
    if newly_loaded_queue:
        current_max_id = max([t['id'] for t in newly_loaded_queue if 'id' in t] + [0])
        if current_max_id >= task_id:
            task_id = current_max_id + 1

    return newly_loaded_queue, None if len(newly_loaded_queue) > 0 else first_error or "No valid task could be unpacked."


def _parse_queue_zip_tasks(filename, state, task_limit=None, log_prefix="[load_queue]", skip_validate_settings=False):
    """Parse queue ZIP file. Returns (queue_list, error_msg or None, source_task_count)."""
    cache_dir = gr.utils.get_upload_folder()

    try:
        print(f"{log_prefix} Attempting to load queue from: {filename}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(filename, 'r') as zf:
                if "queue.json" not in zf.namelist():
                    return None, "queue.json not found in zip file", 0
                print(f"{log_prefix} Extracting to temp directory...")
                zf.extractall(tmpdir)

            manifest_path = os.path.join(tmpdir, "queue.json")
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            source_task_count = len(manifest)
            print(f"{log_prefix} Loaded manifest with {source_task_count} tasks.")
            if task_limit is not None:
                manifest = manifest[:task_limit]

            queue, error = _parse_task_manifest(manifest, state, tmpdir, cache_dir, log_prefix, skip_validate_settings=skip_validate_settings)
            return queue, error, source_task_count

    except Exception as e:
        traceback.print_exc()
        return None, str(e), 0


def _parse_queue_zip(filename, state):
    """Parse queue ZIP file. Returns (queue_list, error_msg or None)."""
    queue, error, _ = _parse_queue_zip_tasks(filename, state)
    return queue, error


def _parse_settings_zip(filename, state, skip_validate_settings=False):
    """Parse a settings ZIP file and extract only the first task."""
    return _parse_queue_zip_tasks(filename, state, task_limit=1, log_prefix="[load_settings]", skip_validate_settings=skip_validate_settings)


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
        client_id= params.get("client_id", "") or ""
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
    service = service_for(state)
    if service is not None and evt.target is None:
        if service.autoloaded_queue:
            return gr.update()
        service.autoloaded_queue = True
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
        try:
            newly_loaded_queue, error = _parse_task_manifest(newly_loaded_queue, state, None, None, "[unpack queue]", verbose_output = verbose_output )
        except Exception as exception:
            traceback.print_exc()
            newly_loaded_queue, error = [], f"Inline queue validation failed: {exception}"
        if error:
            if isinstance(inline_queue_source, dict):
                inline_queue_source = [{"id": 0, "params": inline_queue_source}]
            record_queue_error(state, inline_queue_source or [], error)
            gr.Warning(f"Failed to unpack inline queue: {error[:200]}")
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
            # A running queue counts every task since it started (prompt_no), so admitted tasks extend its total.
            gen["prompts_max"] = gen.get("prompts_max", 0) + len(newly_loaded_queue) if gen.get("in_progress", False) else len(merged_queue)
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
    aborted_current = False
    cleared_pending = False

    with lock:
        queue = gen.get("queue", [])
        active_task = gen.get("api_active_queue_task")
        pending = [task for task in queue if task is not active_task]
        if pending:
            print(f"Clear Queue: Clearing {len(pending)} pending tasks.")
            record_queue_error(state, pending, "Queue cleared by the user", abort=True)
            queue[:] = [task for task in queue if task is active_task]
            gen["prompts_max"] = gen["prompt_no"] if active_task is not None else 0
            cleared_pending = True
        elif active_task is not None:
            print("Clear Queue: Signalling abort for in-progress task.")
            gen["resume"] = True
            gen["abort"] = True
            gen["extra_orders"] = 0
            set_generation_state(gen, status="Processing Request to abort Current Generation")
            if wan_model is not None:
                wan_model._interrupt = True
            aborted_current = True

    if aborted_current:
        service = service_for(state)
        if service is not None:
            service._generation_aborting = True
            service._generation_event("status", "Aborting generation…")

    if cleared_pending or aborted_current:
        try:
            if os.path.isfile(AUTOSAVE_PATH):
                os.remove(AUTOSAVE_PATH)
                print(f"Clear Queue: Deleted autosave file '{AUTOSAVE_PATH}'.")
        except OSError as e:
            print(f"Clear Queue: Error deleting autosave file '{AUTOSAVE_PATH}': {e}")
            gr.Warning(f"Could not delete the autosave file '{AUTOSAVE_PATH}'. You may need to remove it manually.")

    if aborted_current:
        gr.Info("Processing request to abort current generation.")
    elif cleared_pending:
        gr.Info("Pending tasks cleared." if active_task is not None else "Queue cleared.")
    else:
        gr.Info("Queue is already empty.")

    return update_queue_data(queue)
def quit_application():
    print("Save and Quit requested...")
    clear_startup_lock()
    autosave_queue()
    import signal
    os.kill(os.getpid(), signal.SIGINT)

def restart_application():
    print("Restart requested...")
    clear_startup_lock()
    autosave_queue()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(42)

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

def close_application():
    """After Ctrl+C or Quit: save the queue while it still holds the current task, then abort that task and wait for it to stop before Python shuts down its thread pools."""
    if not quitting_app:
        autosave_queue()
    atexit.unregister(autosave_queue)
    worker = _deepy_hybrid._queue_worker
    if worker is None:
        return
    print("Aborting the current generation before closing (press Ctrl+C again to quit at once)...")
    abort_generation(_deepy_hybrid._state, notify=False)
    deadline = time.time() + 30
    while worker.is_alive() and time.time() < deadline:
        worker.join(0.5)  # short waits let a second Ctrl+C through on Windows
    if worker.is_alive():
        print("The generation did not stop within 30 s, closing anyway.")

def finalize_generation_with_state(current_state):
     service = service_for(current_state)
     if service is not None:
         return (*service.finalized_updates(), gr.update(), current_state)
     if not isinstance(current_state, dict) or 'gen' not in current_state:
         # gallery tabs, current tab, image/video gallery and the strip galleries, then the generation buttons
         return (*[gr.update()] * (3 + 3 * len(STRIP_SOURCES)), gr.update(interactive=True), gr.update(interactive=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=""), gr.update(), current_state)
     accordion_update = gr.Accordion(open=False) if len(get_gen_info(current_state).get("queue", [])) <= 1 else gr.update()
     return (*finalize_generation(current_state), accordion_update, current_state)

def generate_queue_html(queue):
    if len(queue) <= 1:
        message = "Generation in progress. No additional items queued." if len(queue) == 1 and is_generation_in_progress() else "Queue is empty."
        return f"<div style='text-align: center; color: grey; padding: 20px;'>{message}</div>"

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

        edit_btn = "" if _is_edit_task_params(item.get("params", {})) else f"""<button onclick="updateAndTrigger('edit_{task_id}')" class="action-button" title="Edit"><img src="/gradio_api/file=icons/edit.svg" style="width: 20px; height: 20px;"></button>"""
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

family_handlers = ["models.wan.wan_handler", "models.wan.ovi_handler", "models.wan.df_handler", "models.hyvideo.hunyuan_handler", "models.ltx_video.ltxv_handler", "models.ltx2.ltx2_handler", "models.ltx2.ltx_audio_tts_handler", "models.longcat.longcat_handler", "models.minimax_h3.minimax_h3_handler", "models.flux.flux_handler", "models.qwen.qwen_handler", "models.ming_image.ming_handler", "models.kandinsky5.kandinsky_handler", "models.kandinsky6.kandinsky6_handler",  "models.z_image.z_image_handler", "models.hidream.hidream_handler", "models.ideogram4.ideogram4_handler", "models.krea2.krea2_handler", "models.magi_human.magi_human_handler", "models.sensenova_u1.sensenova_u1_handler", "models.hunyuan3d.hunyuan3d_handler", "models.trellis2.trellis2_handler", "models.triposg.triposg_handler", "models.animate_any_mesh.animate_any_mesh_handler", "models.triposplat.triposplat_handler", "models.worldmirror.worldmirror_handler", "models.recon3d.recon3d_handler",  "models.TTS.ace_step_handler", "models.TTS.chatterbox_handler", "models.TTS.qwen3_handler", "models.TTS.heartmula_handler", "models.TTS.kugelaudio_handler", "models.TTS.index_tts2_handler", "models.TTS.stable_audio3_handler", "models.TTS.omnivoice_handler", "models.TTS.minimax_music3.minimax_music3_handler", "models.TTS.auk.auk_handler", "models.TTS.yue2.yue2_handler"]
family_handlers.append("models.prism.prism_handler")
family_handlers.append("models.flux3.flux3_handler")
family_handlers.append("models.moss_image.moss_image_handler")
family_handlers.append("models.fibo2.fibo2_handler")
family_handlers.append("models.TTS.fish_audio.fish_audio_handler")
family_handlers.append("models.TTS.moss_tts.moss_tts_handler")
family_handlers.append("models.hy_nomas.hy_nomas_handler")
family_handlers.append("models.evoke.evoke_handler")
family_handlers.append("models.hi3d.hi3d_handler")
family_handlers.append("models.supratex.supratex_handler")
DEFAULT_LORA_ROOT = "loras" #"models.cosmos3.cosmos3_handler",

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

    try:
        inspect.signature(get_dir).bind(base_model_type)
    except TypeError:
        # Legacy handlers return a resolved path, not a LoRA config key.
        lora_dir = get_dir(base_model_type, args, get_lora_root())
        if lora_dir is None: raise Exception("loras unknown")
        os.makedirs(lora_dir, exist_ok=True)
        return lora_dir
    lora_key = get_dir(base_model_type)
    if lora_key is None: raise Exception("loras unknown")
    return resolve_lora_dir(lora_key, get_lora_root(), args.lora_config)

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
override_attention_modes_installed = get_override_attention_modes()
override_attention_modes_supported = get_supported_override_attention_modes()
args = parse_wgp_args(CONFIG_FILENAME)
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
    processing_device = "mps" if is_mps else "cuda"
# torch.backends.cuda.matmul.allow_fp16_accumulation = True
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

force_profile_no = float(args.profile)
verbose_level = int(args.verbose)
check_loras = args.check_loras ==1
ui_perf_debug = bool(args.debug_gen_form or os.getenv("WANGP_DEBUG_UI", "").strip().lower() in {"1", "true", "yes", "on"})
model_dropdowns.MODEL_SELECTOR_DEBUG = ui_perf_debug

with open("models/_settings.json", "r", encoding="utf-8") as f:
    primary_settings = json.load(f)

wgp_root = os.path.abspath(os.getcwd())
deepy_session_store.configure_sessions_root(args.deepy_sessions_dir or os.path.join(wgp_root, deepy_session_store.DEFAULT_SESSIONS_FOLDER))
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
        "save_path": "outputs",  
        "image_save_path": "outputs",  
        "compile" : "disabled",
        "metadata_type": "metadata",
        "boost" : 1,
        "int8_kernels": "auto",
        "fp8_kernels": "auto",
        "kernel_precision": "fast",
        "attention_head_split": 0,
        "clear_file_list" : 5,
        "keep_intermediate_sliding_windows": 1,
        "keep_resolution_on_model_switch": True,
        "enable_4k_resolutions": 0,
        "max_reserved_loras": -1,
        "video_preload_mode": "default",
        "image_preload_mode": "default",
        "audio_preload_mode": "default",
        "vae_config": 0,
        "profile" : profile_type.LowRAM_LowVRAM,
        "video_profile": profile_type.LowRAM_LowVRAM,
        "image_profile": profile_type.LowRAM_LowVRAM,
        "audio_profile": 3,
        "preload_model_policy": [],
        "UI_theme": "default",
        "floating_generate_button": True,
        "checkpoints_paths": fl.default_checkpoints_paths,
        "loras_root": DEFAULT_LORA_ROOT,
        "save_queue_if_crash": 1,
        "queue_color_scheme": "pastel",
        "process_queues_when_browser_unfocused": 1,
        "multi_prompts_gen_type": prompt_parser.DEFAULT_MULTI_PROMPTS_MODE,
        "model_hierarchy_type": 1,
        upsampler_api.UPSAMPLER_CONFIG_KEY: upsampler_api.default_config_sections(),
        audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY: audio_processor_api.default_config_sections(),
        temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY: temporal_upsampler_api.default_config_sections(),
        **get_deepy_default_runtime_config(),
        LLM_CONFIG_KEY: normalize_llm_config({}),
        "prompt_enhancer_quantization": "quanto_int8",
        PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY: PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT,
        "prompt_enhancer_temperature": 0.6,
        "prompt_enhancer_top_p": 0.9,
        "prompt_enhancer_randomize_seed": True,
        "audio_save_path": "outputs",
        **notifications.default_config(),
    }
    apply_first_launch_deepy_prime_defaults(server_config)

    write_config(server_config, server_config_filename)
else:
    server_config = read_config(config_load_filename)

server_config.setdefault("prevent_power_throttling", True)
if args.prevent_power_throttling is None: args.prevent_power_throttling = server_config["prevent_power_throttling"]
if args.prevent_power_throttling: prevent_power_throttling()

server_config.setdefault("clear_file_list", 5)
server_config.setdefault("prompt_enhancer_quantization", "quanto_int8")
notifications.apply_defaults(server_config)
server_config.setdefault(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)
server_config[LLM_CONFIG_KEY] = normalize_llm_config(server_config)
server_config["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(
    server_config.get("multi_prompts_gen_type", prompt_parser.DEFAULT_MULTI_PROMPTS_MODE),
    default=prompt_parser.DEFAULT_MULTI_PROMPTS_MODE,
)
primary_settings["multi_prompts_gen_type"] = server_config["multi_prompts_gen_type"]
resolution_preferences = resolution_utils.migrate_resolution_preferences(server_config)
if resolution_preferences:
    update_config(server_config, server_config_filename, resolution_preferences)
server_config.setdefault(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1)
gradio_queue_focus_patch.BACKGROUND_SCHEDULER_DEFAULT_ENABLED = bool(server_config.get(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1))
gradio_queue_focus_patch.install()
gradio_model_switch_patch.install(verbose=ui_perf_debug)
gradio_startup_patch.install()
gradio_event_context_patch.install()
deepy_voice.install_gradio_routes(get_service=lambda: _deepy_hybrid, language=args.deepy_voice_language)

checkpoints_paths = server_config.get("checkpoints_paths", None)
if checkpoints_paths is None: checkpoints_paths = server_config["checkpoints_paths"] = fl.default_checkpoints_paths
fl.set_checkpoints_paths(checkpoints_paths)
three_levels_hierarchy = server_config.get("model_hierarchy_type", 1) == 1

if app is None:
    app = WAN2GPApplication()
app.plugin_manager.set_server_config(server_config, server_config_filename)
plugin_model_extensions = app.plugin_manager.discover_plugin_model_extensions(server_config.get("enabled_plugins", []))
model_handler_sources = {path: {"profile_roots": ["profiles"], "plugin_id": ""} for path in family_handlers}
model_definition_sources = [{"defaults_root": "defaults", "plugin_id": ""}]
for extension in plugin_model_extensions:
    model_definition_sources.append({"defaults_root": extension.defaults_root, "plugin_id": extension.plugin_id})
    for handler_path in extension.model_handlers:
        if handler_path not in family_handlers:
            family_handlers.append(handler_path)
        model_handler_sources[handler_path] = {"profile_roots": [extension.profiles_root, "profiles"], "plugin_id": extension.plugin_id}

from shared.utils import offload_registry

upsampler_api.register_spatial_upsamplers(server_config, fl)
audio_processor_api.register_audio_processors(server_config, fl)
temporal_upsampler_api.register_temporal_upsamplers(server_config, fl)
migrate_extension_defaults(server_config, server_config_filename, workspace_root=args.workspaces_dir or server_config.get('workspaces_dir') or os.path.join(wgp_root, 'workspaces'))

def _normalize_profile_defaults(config):
    if "profile" not in config:
        config["profile"] = profile_type.LowRAM_LowVRAM
    base_profile = config.get("profile", profile_type.LowRAM_LowVRAM)
    config.setdefault("video_profile", base_profile)
    config.setdefault("image_profile", base_profile)
    config.setdefault("audio_profile", 3)
    return config["video_profile"], config["image_profile"], config["audio_profile"]

def _normalize_output_paths(config):
    if "save_path" not in config:
        config["save_path"] = "outputs"
    if "image_save_path" not in config:
        config["image_save_path"] = config["save_path"]
    if "audio_save_path" not in config:
        config["audio_save_path"] = config["save_path"]
    if "mesh_save_path" not in config:
        config["mesh_save_path"] = config["save_path"]

_normalize_profile_defaults(server_config)
_normalize_output_paths(server_config)
server_config.pop("lm_decoder_engine", None)  # no longer a choice: LM decoders use every optimization supported (resolve_lm_features)

from preprocessing.matanyone.utils.model_assets import migrate_matanyone_install
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
    base_types_handlers, families_infos, models_eqv_map, models_comp_map, retired_types = {}, {"unknown": (100, "Unknown")}, {}, {}, {}
    for path in family_handlers:
        handler = importlib.import_module(path).family_handler
        handler_source = model_handler_sources.get(path, {})
        profile_roots = handler_source.get("profile_roots", ["profiles"])
        plugin_id = handler_source.get("plugin_id", "")
        for model_type in handler.query_supported_types():
            if model_type in base_types_handlers:
                prev = base_types_handlers[model_type].__name__
                raise Exception(f"Model type {model_type} supported by {prev} and {handler.__name__}")
            base_types_handlers[model_type] = handler
            model_profile_roots_by_architecture[model_type] = list(profile_roots)
            if plugin_id:
                model_plugin_ids_by_architecture[model_type] = plugin_id
        families_infos.update(handler.query_family_infos())
        eq_map, comp_map = handler.query_family_maps()
        models_eqv_map.update(eq_map); models_comp_map.update(comp_map)
        if hasattr(handler, "query_retired_model_types"):  # model types merged into others: {old type: its main new type}
            retired_types.update({old_type: (handler, new_type) for old_type, new_type in handler.query_retired_model_types().items()})
    return base_types_handlers, families_infos, models_eqv_map, models_comp_map, retired_types

# models_eqv_map: bidirectional compatibility between base model types
# models_comp_map : mono directional compatibility between base model types {"A" : {"B", "C"} } means B & C model types can accept A  model types (but not the other way). Said otherwise B & C are derived data types

model_profile_roots_by_architecture = {}
model_plugin_ids_by_architecture = {}

model_types_handlers, families_infos,  models_eqv_map, models_comp_map, retired_model_types = map_family_handlers(family_handlers)

def collect_prompt_helper_assets(asset_name):
    chunks = []
    for handler in dict.fromkeys(model_types_handlers.values()):
        getter = getattr(handler, asset_name, None)
        if getter is not None:
            chunk = getter()
            if chunk:
                chunks.append(str(chunk))
    return "\n".join(chunks)

def _store_model_metadata(model_type, model_def):
    return model_metadata.store_metadata(model_type, model_def, model_types_handlers, families_infos)

def list_model_defs(family=None, base_model_type=None, finetune=None, model_type=None, main_output=None, inputs=None):
    return model_metadata.list_model_defs(models_def, family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs)

def get_model_defs(**filters):
    return list_model_defs(**filters)

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
    return get_model_def(model_type).get("parent_model_type", model_def.get("parent_model_type", base_model_type))
    
def get_model_handler(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception(f"Unknown model type {model_type}")
    model_handler = model_types_handlers.get(base_model_type, None)
    if model_handler is None:
        raise Exception(f"No model handler found for base model type {base_model_type}")
    return model_handler

def convert_retired_model_type(settings):
    """Settings of a retired model type (merged into other model types: a 3D model replaced by a stage model and post-processors)
    converted in place by its family handler. Returns the settings' model type, the new one for converted settings."""
    model_type = settings.get("model_type", settings.get("base_model_type"))
    if model_type not in retired_model_types:
        return model_type
    handler, _ = retired_model_types[model_type]
    new_type = handler.convert_retired_settings(model_type, settings)
    settings["model_type"] = new_type
    if "base_model_type" in settings:
        settings["base_model_type"] = get_base_model_type(new_type)
    return new_type


def retired_model_selection(entry):
    """A model selection entry (model_dropdowns) naming a retired model type, renamed to the type replacing it."""
    level, key = model_dropdowns.parse_model_selection(entry)
    if level == 1 or key not in retired_model_types:
        return entry
    new_type = retired_model_types[key][1]
    return new_type if level == 3 else f"{level}:{new_type}"


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
    model_def = get_model_def(model_type) or {}
    return model_metadata.get_model_family(base_model_type, model_def, model_types_handlers, families_infos, for_ui=for_ui)

def test_class_i2v(model_type):    
    model_def = get_model_def(model_type)
    return model_def.get("i2v_class", False)

def test_vace_module(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("vace_class", False)

def test_class_t2v(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("t2v_class", False)

def image_prompt_types_allow_t2v(model_def, image_mode=0):
    return int(image_mode or 0) == 0 and "T" in model_def.get("image_prompt_types_allowed", "")

def get_guide_custom_choices(model_def, image_mode=0):
    if int(image_mode or 0) > 0 and model_def.get("guide_custom_choices_image", None) is not None:
        return model_def["guide_custom_choices_image"]
    return model_def.get("guide_custom_choices", None)

def frame_scheduler_supported(model_type, model_def, image_mode=0, is_edit_mode=False):
    return not is_edit_mode and int(image_mode or 0) == 0 and not model_def.get("audio_only", False) and test_any_sliding_window(model_type)

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

def get_model_recursive_prop(model_type, prop = "URLs", sub_prop_name = None, return_list = True, stack = [], model_def = None):
    if model_def is None: model_def = models_def.get(model_type, None)
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


def get_model_config_groups(model_type, model_def=None):
    return [get_model_recursive_prop(model_type, key, return_list=False, model_def=model_def) or {} for key in model_config_groups.CONFIG_KEYS]


def get_model_filename(model_type, quantization ="int8", dtype_policy = "", module_type = None, submodel_no = 1, URLs = None, stack=[], model_def = None):
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

        if model_def is None: model_def = models_def.get(model_type, None)
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

def fix_postprocess_audio_settings(ui_defaults, settings_version):
    return audio_processor_api.fix_settings(ui_defaults, settings_version, attachment_has_path_values=_attachment_has_path_values)

def fix_settings(model_type, ui_defaults, min_settings_version = 0):
    resolution_utils.fix_resolution_settings(ui_defaults)
    if model_type is None: return

    settings_version =  max(min_settings_version, ui_defaults.get("settings_version", 0))
    # before 2.82 the post-processors' parameters were flat spatial_upsampler_* settings and two dictionaries: also converted when model
    # defaults or API calls still carry them
    if any(key in ui_defaults for key in parameter_ui.LEGACY_SETTINGS):
        parameter_ui.migrate_settings(ui_defaults)
    # before 2.81 each override was its own setting: also converted when model defaults or API calls still carry them
    legacy_overrides = {"override_profile": "memory_profile", "override_compile": "compilation", "override_attention": "attention_mode"}
    if any(name in ui_defaults for name in legacy_overrides):
        override_values = generation_overrides.parse_overrides(ui_defaults.get("overrides", ""))
        for legacy_name, name in legacy_overrides.items():
            if legacy_name in ui_defaults:
                override_values[name] = ui_defaults.pop(legacy_name) or generation_overrides.OVERRIDE_DEFAULTS[name]
        ui_defaults["overrides"] = generation_overrides.format_overrides(override_values)
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    duration_conversion = apply_video_length_duration(ui_defaults, model_def)
    if duration_conversion is not None:
        print(f"WanGP settings converted video_length={duration_conversion['input']!r} to {duration_conversion['frames']} frames at {duration_conversion['fps']:g} fps for {model_type}")

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
            ui_defaults["multi_prompts_gen_type"] = server_config["multi_prompts_gen_type"]
        else:
            ui_defaults["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(legacy_multi_prompts_mode, default=server_config["multi_prompts_gen_type"])
    else:
        ui_defaults["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(ui_defaults.get("multi_prompts_gen_type", server_config["multi_prompts_gen_type"]), default=server_config["multi_prompts_gen_type"])
    if not test_any_sliding_window(model_type):
        ui_defaults["multi_prompts_gen_type"] = ui_defaults["multi_prompts_gen_type"].replace("W", "G")

    if settings_version < 2.60:
        ui_defaults.setdefault("replace_voice_sample", None)

    if settings_version < 2.61:
        prompt_enhancer = str(ui_defaults.get("prompt_enhancer") or "")
        if prompt_enhancer == "I":
            ui_defaults["prompt_enhancer"] = "TI"
        elif prompt_enhancer == "IK":
            ui_defaults["prompt_enhancer"] = "TIK"

    audio_prompt_type = fix_postprocess_audio_settings(ui_defaults, settings_version)
    if settings_version < 2.78:
        if "spatial_upsampling" in ui_defaults:
            ui_defaults["spatial_upsampling"] = upsampler_api.normalize_upsampling_value(ui_defaults["spatial_upsampling"])
        if "temporal_upsampling" in ui_defaults:
            ui_defaults["temporal_upsampling"] = temporal_upsampler_api.normalize_temporal_upsampling_value(ui_defaults["temporal_upsampling"])
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
    if get_guide_custom_choices(model_def, ui_defaults.get("image_mode", 0)) is None:
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

    ui_defaults["settings_version"] = settings_version

def get_default_prompt(i2v):
    if i2v:
        return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
    return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."


def get_factory_settings(model_type):
    i2v = test_class_i2v(model_type)
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    ui_defaults = copy.deepcopy(primary_settings)
    apply_custom_settings_defaults(model_def, ui_defaults)
    ui_defaults.update({
        "settings_version": settings_version,
        "prompt": get_default_prompt(i2v),
        "resolution": "1280x720" if "720" in base_model_type else "832x480",
        "flow_shift": 7.0 if "720" not in base_model_type and i2v else 5.0,
    })
    get_model_handler(model_type).update_default_settings(base_model_type, model_def, ui_defaults)
    model_settings = model_def.get("settings")
    if model_settings is not None:
        ui_defaults.update(copy.deepcopy(model_settings))
    if len(ui_defaults.get("prompt", "")) == 0:
        ui_defaults["prompt"] = get_default_prompt(i2v)
    # needs to implement settings md version for defaults/finetunes
    # fix_settings(model_type, ui_defaults, settings_version)
    return ui_defaults


def get_default_settings(model_type):
    defaults_filename = get_settings_file_name(model_type)
    if not Path(defaults_filename).is_file():
        ui_defaults = get_factory_settings(model_type)
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
    base_model_type = model_def.get("architecture", None) or get_base_model_type(model_type)
    family_handler = model_types_handlers.get(base_model_type, None)
    if family_handler is None:
        if model_def.get("visible", True):
            print(f"Skipping model type '{model_type}' with unsupported architecture '{base_model_type}'.")
        model_def["visible"] = False
        return model_def
    default_model_def = family_handler.query_model_def(base_model_type, model_def)
    if default_model_def is None: return model_def
    default_model_def = resolution_utils.migrate_resolution_definition(default_model_def)
    default_model_def.update(model_def)
    default_model_def = resolution_utils.migrate_resolution_definition(default_model_def)
    resolution_utils.validate_resolution_definition(default_model_def)
    default_model_def["_profile_roots"] = model_profile_roots_by_architecture.get(base_model_type, ["profiles"])
    plugin_id = model_plugin_ids_by_architecture.get(base_model_type, "")
    if plugin_id:
        default_model_def["_plugin_id"] = plugin_id
    return _store_model_metadata(model_type, default_model_def)


def refresh_model_defs():
    global models_def, model_types, displayed_model_types
    new_models_def, parse_errors, previous_models_def, old_model_types = {}, [], models_def.copy(), set()
    defaults_paths = set()
    for source in model_definition_sources:
        defaults_root = source["defaults_root"]
        defaults_paths.update(glob.glob(os.path.join(defaults_root, "*.json")))
    models_def_paths = sorted([*defaults_paths, *glob.glob(os.path.join("finetunes", "*.json"))])
    def warn(msg):
        print(msg)
        parse_errors.append(msg)
    def use_previous_model_def(model_type, file_path, error):
        previous_model_def = previous_models_def.get(model_type, None)
        if previous_model_def is None:
            return False
        warn(f"Model Definition File '{file_path}' could not be refreshed; using previous definition for '{model_type}': {str(error)}")
        old_model_types.add(model_type)
        new_models_def[model_type] = previous_model_def
        return True
    for idx, file_path in enumerate(models_def_paths, 1):
        file_start = time.perf_counter()
        model_type = os.path.basename(file_path)[:-5]
        if model_type in old_model_types:
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_def = json.load(f)
            except Exception as e:
                if use_previous_model_def(model_type, file_path, e):
                    continue
                elif file_path in defaults_paths:
                    raise Exception(f"Error while parsing Model Definition File '{file_path}': {str(e)}")
                else:
                    warn(f"Finetune Definition File '{file_path}' will be ignored as there was an error in its parsing: {str(e)}")
                    continue
        try:
            resolution_utils.fix_resolution_settings(json_def)
            model_def = json_def.pop("model")
            model_def["path"] = file_path
            existing_model_def = new_models_def.get(model_type, None)
            if existing_model_def is not None:
                existing_model_def.setdefault("settings", {}).update(json_def)
                existing_model_def.update(model_def)
                existing_model_def = resolution_utils.migrate_resolution_definition(existing_model_def)
                resolution_utils.validate_resolution_definition(existing_model_def)
                new_models_def[model_type] = existing_model_def
                _store_model_metadata(model_type, existing_model_def)
            else:
                new_models_def[model_type] = model_def
                model_def = init_model_def(model_type, model_def)
                new_models_def[model_type] = model_def
                model_def["settings"] = json_def
                _store_model_metadata(model_type, model_def)
        except Exception as e:
            if use_previous_model_def(model_type, file_path, e):
                continue
            elif file_path in defaults_paths:
                raise Exception(f"Error while refreshing Model Definition File '{file_path}': {str(e)}")
            else:
                warn(f"Finetune Definition File '{file_path}' will be ignored as there was an error in its refresh: {str(e)}")

    models_def = new_models_def
    model_types = models_def.keys()
    displayed_model_types = [model_type for model_type, model_def in models_def.items() if model_def.get("visible", True)]
    return parse_errors

refresh_model_defs()

transformer_types = []
for model_entry in dict.fromkeys(map(retired_model_selection, server_config.get("transformer_types", []))):
    model_level, model_key = model_dropdowns.parse_model_selection(model_entry)
    if get_model_def(model_key) is None if model_level == 3 else len(model_dropdowns.expand_model_selection([model_entry], displayed_model_types, get_parent_model_type, get_model_family)) == 0:
        print(f"Model '{model_entry}' is missing. Either install it in the finetune folder or remove this model from key 'transformer_types' in {CONFIG_FILENAME}")
    else:
        transformer_types.append(model_entry)
transformer_type = server_config.get("last_model_type", None)
if transformer_type in retired_model_types:
    transformer_type = retired_model_types[transformer_type][1]
advanced = server_config.get("last_advanced_choice", False)
last_resolution = server_config.get("last_resolution_choice", None)
if args.advanced: advanced = True 

if transformer_type != None and not transformer_type in model_types and not transformer_type in models_def: transformer_type = None
if transformer_type == None:
    selected_model_types = model_dropdowns.expand_model_selection(transformer_types, displayed_model_types, get_parent_model_type, get_model_family)
    transformer_type = selected_model_types[0] if len(selected_model_types) > 0 else "t2v"

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
        server_config["attention_mode"] = attention_mode
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

default_profile_video = force_profile_no if force_profile_no >= 0 else server_config["video_profile"]
default_profile_image = force_profile_no if force_profile_no >= 0 else server_config["image_profile"]
default_profile_audio = force_profile_no if force_profile_no >= 0 else server_config["audio_profile"]
default_profile = default_profile_video
loaded_profile = force_profile_no = -1
compile = server_config.get("compile", "disabled")
transformer_quantization_choices = [("Scaled Int8 (recommended)", "int8"), ("Scaled Fp8", "fp8"), ("16-bit (no quantization)", "bf16")]
text_encoder_quantization_choices = [("16-bit (more RAM, better quality)", "bf16"), ("8-bit (less RAM, slightly lower quality)", "int8")]
compile_choices = [("Disabled", "disabled"),("CUDA Graph (faster short steps, the first steps of a new resolution are slightly slower)", "cudagraph"), ("PyTorch Compilation (up to 10% faster, very slow compilation, requires Triton, to be deprecated)", "transformer")]
if args.compile:
    compile="transformer"
    lock_ui_compile = True
if is_mps: compile = "disabled"
boost = server_config.get("boost", 1)
int8_kernels = server_config.get("int8_kernels", "auto")
apply_int8_kernel_setting(int8_kernels)
fp8_kernels = server_config.get("fp8_kernels", "auto")
apply_fp8_kernel_setting(fp8_kernels)
kernel_policy.configure(server_config.get("kernel_precision", "fast"))
attention_kit.configure(server_config.get("attention_head_split", 0))
# the global states that change what the blocks of a CUDA graph compute are part of the keys of their graphs
cuda_graphs.register_key(lambda: (offload.shared_state.get("_attention"), attention_kit.head_split, kernel_policy.precision, int8_backend.revision, fp8_backend.revision))
# CUDA Graph mode: the attention modes and the classes of quantized weights / layers whose kernels were validated in captured graphs
CUDA_GRAPH_ATTENTION_MODES = {"sdpa", "flash", "sage", "sage2", "radial", "sol"}
CUDA_GRAPH_WEIGHT_TYPES = {"WeightQBytesTensor", "ScaledFP8WeightTensor", "Int8ConvRotWeightTensor", "GGUFWeightTensor", "GGUFSourceTensor", "BNBNF4WeightTensor", "AsymW4A8Int8WeightTensor",
                           "NunchakuSVDQWeightTensor", "NunchakuAWQWeightTensor", "NVFP4WeightTensor", "BlockScaledWeight",
                           "QEmbedding"}  # Quanto embedding: a lookup in its WeightQBytesTensor (graph replay equal to eager, 2026-10-20)
cuda_graphs.register_gate(lambda: None if offload.shared_state.get("_attention") in CUDA_GRAPH_ATTENTION_MODES else
                          f"the attention mode '{offload.shared_state.get('_attention')}' is not supported by them (supported: {', '.join(sorted(CUDA_GRAPH_ATTENTION_MODES))})")


def check_cuda_graph_weights(offloadobj):
    """The graphed models whose quantized weights or layers have not been validated with CUDA graphs run their blocks eagerly."""
    for model_id, graphs in offloadobj.cuda_graphs.items():
        model = offloadobj.models[model_id]
        kinds = {type(t).__name__ for t in list(model.parameters()) + list(model.buffers()) if type(t) not in (torch.Tensor, torch.nn.Parameter)}
        kinds |= {type(module).__name__ for module in model.modules() if type(module).__module__.startswith(("shared.qtypes", "optimum.quanto"))}
        unknown = sorted(kinds - CUDA_GRAPH_WEIGHT_TYPES)
        if unknown:
            graphs.disabled = f"its weights of type {', '.join(unknown)} have not been validated with them"
            print(f"[CUDA Graphs] Off for '{model_id}': {graphs.disabled}.")
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

reload_needed = False
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "outputs"))
image_save_path = server_config.get("image_save_path", os.path.join(os.getcwd(), "outputs"))
audio_save_path = server_config.get("audio_save_path", save_path)
mesh_save_path = server_config.get("mesh_save_path", save_path)
if not "video_output_codec" in server_config: server_config["video_output_codec"]= "libx264_8"
if not "hdr_video_crf" in server_config: server_config["hdr_video_crf"] = 8
if not "video_container" in server_config: server_config["video_container"]= "mp4"
if not "embed_source_images" in server_config: server_config["embed_source_images"]= False
if not "keep_resolution_on_model_switch" in server_config: server_config["keep_resolution_on_model_switch"]= True
if not "enable_4k_resolutions" in server_config: server_config["enable_4k_resolutions"]= 0
if not "max_reserved_loras" in server_config: server_config["max_reserved_loras"]= -1
if not "image_output_codec" in server_config: server_config["image_output_codec"]= "jpeg_95"
if not "audio_output_codec" in server_config: server_config["audio_output_codec"]= "aac_128"
if not "audio_stand_alone_output_codec" in server_config: server_config["audio_stand_alone_output_codec"]= "wav"
upsampler_api.require_upsampler_by_method("flashvsr").normalize_config()
if "loras_root" not in server_config: server_config["loras_root"] = DEFAULT_LORA_ROOT
if "save_queue_if_crash" not in server_config: server_config["save_queue_if_crash"] = 1
if "keep_intermediate_sliding_windows" not in server_config: server_config["keep_intermediate_sliding_windows"] = 1
if "prompt_enhancer_temperature" not in server_config: server_config["prompt_enhancer_temperature"] = 0.6
if "prompt_enhancer_top_p" not in server_config: server_config["prompt_enhancer_top_p"] = 0.9
if "prompt_enhancer_randomize_seed" not in server_config: server_config["prompt_enhancer_randomize_seed"] = True
set_deepy_runtime_config(server_config, server_config_filename)
if "int8_kernels" not in server_config: server_config["int8_kernels"] = "auto"
if "fp8_kernels" not in server_config: server_config["fp8_kernels"] = "auto"

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

def save_quantized_model(model, model_type, model_filename, dtype, config_file, submodel_no=1, convrot_layout=None, convrot_prefixes=None):
    if "quanto" in model_filename or "convrot" in model_filename: return
    source_filename = model_filename
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
            replacement = "_int8_convrot" if args.convrot else f"_quanto_{rep}_int8"
            model_filename = model_filename.replace("_" + rep, replacement)
            break
    if not any(token in model_filename for token in ("quanto", "convrot")):
        pos = model_filename.rfind(".")
        suffix = "_int8_convrot" if args.convrot else "_quanto_int8"
        model_filename = model_filename[:pos] + suffix + model_filename[pos:]

    model_filename = os.path.basename(model_filename)
    if fl.locate_file(model_filename, error_if_none= False) is not None:
        print(f"There isn't any model to quantize as quantized model '{model_filename}' aready exists")
    else:
        model_filename_path = os.path.join(fl.get_download_location(), model_filename)
        if args.convrot:
            from shared.convert.convrot import save_convrot_model
            source_path = source_filename if os.path.isfile(source_filename) else fl.locate_file(source_filename)
            save_convrot_model(model, source_path, model_filename_path, layout=convrot_layout, verboseLevel=verbose_level, prefixes=convrot_prefixes)
        else:
            offload.save_model(model, model_filename_path, do_quantize=True, config_file_path=config_file)
        print(f"New quantized file '{model_filename}' had been created for finetune Id '{model_type}'.")
        if not any(os.path.basename(url) == model_filename for url in URLs):
            if os.path.basename(os.path.dirname(model_def["path"])) == "defaults":  # official definitions only list hosted URLs
                print(f"Upload '{model_filename}' and add its URL to '{model_def['path']}'.")
                return
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


def process_files_def(repoId = None, sourceFolderList = None, fileList = None, targetFolderList = None, gen=None, show_filename=True):
    from shared.utils.download import process_files_def as shared_process_files_def

    return shared_process_files_def(repoId=repoId, sourceFolderList=sourceFolderList, fileList=fileList, targetFolderList=targetFolderList, gen=gen, show_filename=show_filename)

def release_flashvsr_vram():
    upsampler_api.require_upsampler_by_method("flashvsr").release_vram()

def release_pid_vram():
    handler = upsampler_api.find_upsampler_by_method("flux_pid") or upsampler_api.find_upsampler_by_method("flux2_pid")
    if handler is not None:
        handler.release_vram()

def release_coz_vram():
    upsampler_api.require_upsampler_by_method("coz").release_vram()

def release_extension_offloadobjs():
    return offload_registry.release_all()

def download_requested_postprocessing_assets(send_cmd, *, postprocess_audio="", temporal_upsampling="", spatial_upsampling="", replace_voice_method="", gen=None):
    process_files = partial(process_files_def, gen=gen)
    for method in dict.fromkeys([audio_processor_api.normalize_method(postprocess_audio), audio_processor_api.normalize_method(replace_voice_method)]):
        if method and method != "control":
            audio_processor_api.download_for_method(method, process_files, send_cmd=send_cmd)
    temporal_upsampler_api.download_for_value(temporal_upsampling, process_files, send_cmd=send_cmd)
    upsampler_api.download_for_value(spatial_upsampling, process_files, send_cmd=send_cmd)


def download_file(url,filename, gen=None, show_filename=True):
    from shared.utils.download import download_url as shared_download_file

    return shared_download_file(url, filename, gen=gen, show_filename=show_filename)

def query_global_shared_model_files():
    from shared.utils.optional_assets import query_download_defs

    return query_download_defs(server_config)

def download_models(model_filename = None, model_type= None, file_type = 0, submodel_no = 1, force_path = None, model_def = None, gen=None, show_filename=True):
    def computeList(filename):
        if filename == None:
            return []
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        


    if model_filename is None: return

    base_model_type = get_base_model_type(model_type)
    model_def = model_def or get_model_def(model_type)
    
    any_source = ("source2" if submodel_no ==2 else "source") in model_def
    any_module_source = ("module_source2" if submodel_no ==2 else "module_source") in model_def 
    model_type_handler = model_types_handlers[base_model_type]

    if not (any_source and file_type==0 or any_module_source and file_type==1):
        local_model_filename = fl.get_local_model_filename(model_filename, extra_paths= force_path)
        if local_model_filename is None and len(model_filename) > 0:
            local_model_filename = fl.get_smart_download_location(model_filename, force_path= force_path)
            url = model_filename

            if not url.startswith("http"):
                raise Exception(f"Model '{model_filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, local_model_filename, gen=gen, show_filename=show_filename)
            except (DownloadCancelled, DownloadError):
                raise
            except Exception as e:
                if os.path.isfile(local_model_filename): os.remove(local_model_filename) 
                raise Exception(f"'{url}' is invalid for Model '{model_type}' : {str(e)}'")
            if file_type!=0: return
    lora_dir = get_lora_dir(model_type) 
    for prop, recursive in zip(["preload_URLs", "VAE_URLs"], [True, False]):
        if recursive:
            preload_URLs = get_model_recursive_prop(model_type, prop, return_list= True, model_def=model_def)
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
                    download_file(url, filename, gen=gen, show_filename=show_filename)
                except (DownloadCancelled, DownloadError):
                    raise
                except Exception as e:
                    if os.path.isfile(filename): os.remove(filename) 
                    raise Exception(f"{prop} '{url}' is invalid: {str(e)}'")

    model_loras = get_model_recursive_prop(model_type, "loras", return_list= True, model_def=model_def)
    for url in model_loras:
        filename = get_lora_local_path(lora_dir, url)
        if not os.path.isfile(filename ): 
            if not url.startswith("http"):
                raise Exception(f"Lora '{filename}' was not found in the Loras Folder and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, filename, gen=gen, show_filename=show_filename)
            except (DownloadCancelled, DownloadError):
                raise
            except Exception as e:
                if os.path.isfile(filename): os.remove(filename) 
                raise Exception(f"Lora URL '{url}' is invalid: {str(e)}'")
            
    if file_type != 0: return            
    model_files = model_type_handler.query_model_files(computeList, base_model_type, model_def)
    if not isinstance(model_files, list): model_files = [model_files]
    for one_repo in model_files:
        process_files_def(**one_repo, gen=gen, show_filename=show_filename)

offload.default_verboseLevel = verbose_level

loras_url_cache = None
loras_cache_file = "loras_url_cache_v2.json"
def _ensure_loras_url_cache():
    global loras_url_cache
    if loras_url_cache is None:
        if os.path.isfile(loras_cache_file):
            try:
                with open(loras_cache_file, 'r', encoding='utf-8') as f:
                    loras_url_cache = json.load(f)
            except:
                loras_url_cache = {}
        else:
            loras_url_cache = {}


def get_lora_local_path(lora_dir, lora):
    if os.path.isabs(lora): return lora
    if (lora.startswith("http:") or lora.startswith("https:")):
        parts = lora.split("|")
        lora_path = os.path.join(fl.clean_relative_path(parts[1]), os.path.basename(parts[0])) if len(parts) > 1 else os.path.basename(lora)
    else:
        lora_path = lora
    return lora_path if lora_dir is None else os.path.join(lora_dir, lora_path) 

def get_lora_URL(lora_dir, lora):
    if os.path.isabs(lora): return lora
    _ensure_loras_url_cache()
    rel_path = get_lora_local_path(None, lora)
    if lora_dir is None: return rel_path
    url = loras_url_cache.get(lora_dir + "|" +  rel_path, None)         
    if url is None:
        return rel_path
    base = os.path.dirname(rel_path)
    return url if len(base)==0 else url + "|" + base

def check_loras_exist(model_type, loras_choices_files, download = False, send_cmd = None, gen=None):
    _ensure_loras_url_cache()
    lora_dir = get_lora_dir(model_type)
    missing_local_loras = []
    missing_remote_loras = []
    for lora_file in loras_choices_files:
        local_path = get_lora_local_path(lora_dir, lora_file)
        if not os.path.isfile(local_path):
            rel_path = get_lora_local_path(None, lora_file)
            url = loras_url_cache.get(lora_dir + "|" +  rel_path, None)         
            if url is not None:
                if download:
                    if send_cmd is not None:
                        send_cmd("status", f'Downloading Lora {os.path.basename(lora_file)}...')
                    try:
                        download_file(url, local_path, gen=gen)
                    except DownloadCancelled:
                        raise
                    except Exception as e:
                        print(f"Error downloading {url}:{e}")
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
        dir_loras = glob.glob(os.path.join(lora_dir, "**", "*.sft"), recursive=True) + glob.glob(os.path.join(lora_dir, "**", "*.safetensors"), recursive=True)
        dir_loras.sort(key=lambda path: os.path.relpath(path, lora_dir).casefold())
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets_settings = glob.glob( os.path.join(lora_dir , "*.json") ) + glob.glob( os.path.join(lora_dir , "*.zip") )
        dir_presets_settings.sort()
        dir_presets =   glob.glob( os.path.join(lora_dir , "*.lset") )
        dir_presets.sort()
        # loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets_settings + dir_presets]
        loras_presets = [ Path(file_path).parts[-1] for file_path in dir_presets_settings + dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=get_loras_preprocessor(transformer, base_model_type), split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras = [get_lora_local_path(None, os.path.relpath(lora, lora_dir).replace("\\", "/")) if lora_dir is not None else get_lora_local_path(None, lora) for lora in loras]

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


def compute_compile(override_compile, default_compile):
    return "disabled" if is_mps else override_compile or default_compile

def compute_quantization(overrides):
    return overrides.get("transformer_quantization") or transformer_quantization, overrides.get("text_encoder_quantization") or text_encoder_quantization

def get_profile_type_for_model(model_type, image_mode=0):
    model_def = get_model_def(model_type)
    if model_def is None: return "video"
    profile_type = model_def.get("profile_type", None)
    if profile_type is not None: return profile_type
    if model_def.get("audio_only", False):
        return "audio"
    if image_mode and image_mode > 0:
        return "image"
    return "video"

def preload_mode(output_type):
    # the VRAM preload of a kind of output with memory profiles 0, 2, 4 and 5: "default" (the profile's), "manual" (its VRAM Preload value) or
    # "dynamic" (the preload follows the VRAM left by each workload, the profile default until measured; mmgp says when it cannot apply it:
    # without an MMGP VRAM allocator, or with Profile 4+, which sends one part at a time)
    return "manual" if int(args.preload) > 0 else server_config[f"{_normalize_output_type(output_type)}_preload_mode"]

def world_memory_profile(model_type):
    # the memory profile of a world model's explorations, and whether it keeps blocks in VRAM (profiles 0 to 3, or a VRAM Preload): the needs
    # its Worlds explorer's Memory menu shows
    output_type = get_profile_type_for_model(model_type)
    profile, mode = int(get_default_profile(output_type)), preload_mode(output_type)
    amount = int(args.preload) or server_config.get(f"{_normalize_output_type(output_type)}_preload_in_VRAM", 0)
    return {"profile": profile, "preload": profile in (0, 1, 2, 3) or mode == "dynamic" or mode == "manual" and amount > 0}

def dynamic_preload(profile, output_type):
    # profiles 1 and 3 always preload dynamically, profile 0 unless its VRAM Preload mode is manual; 2, 4 and 5 when that mode asks for it
    mode = preload_mode(output_type)
    return int(profile) in (1, 3) or mode == "dynamic" or int(profile) == 0 and mode == "default"

auto_preload_measures = {} # model type -> (its model definition, the measures of the automatic preload): measured again with a new definition

def auto_preload_store(model_type):
    model_def = get_model_def(model_type)
    stored = auto_preload_measures.get(model_type, None)
    if stored is None or stored[0] is not model_def: # definitions are rebuilt when they are refreshed, e.g. after a finetune was edited
        stored = auto_preload_measures[model_type] = (model_def, {})
    return stored[1]

def init_pipe(pipe, kwargs, profile, output_type="video", extension=None):
    if extension is not None:
        from postprocessing.memory import configure
        profile = configure(kwargs, server_config)
        preload = kwargs["minimumBudget"]
    else:
        preload = int(args.preload)
        if preload == 0 and preload_mode(output_type) == "manual":
            preload = server_config.get(f"{_normalize_output_type(output_type)}_preload_in_VRAM", 0)
    kwargs["preloadExperts"] = preload > 0
    kwargs["perc_reserved_mem_max"] = args.perc_reserved_mem_max or server_config.get("perc_reserved_mem_max", 0) / 100 # 0: mmgp's environment variable or default

    kwargs["extraModelsToQuantize"]=  None
    budgets = kwargs.setdefault("budgets", {})
    # per-model VRAM budgets (in MB) from the config: they beat the profile defaults below but never a budget a handler set
    user_budgets = server_config.get("budgets")
    if isinstance(user_budgets, dict):
        for name, value in user_budgets.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                budgets.setdefault(name, value)
    mmgp_profile = int(profile)
    if mmgp_profile in (0, 1, 2, 3, 4, 5): # the budgets a handler gives replace the profile's
        # text encoders run once per prompt: streamed, they use less VRAM and overlap their transfers with their layers (a model whose
        # text_encoder_2 runs many passes gives it a budget of its own)
        defaults = {"transformer": 100, "text_encoder": 100, "text_encoder_2": 100, "*": 1000 if profile == 5 else 3000}
        if "transformer2" in pipe: defaults["transformer2"] = 100
        for name, default in defaults.items(): budgets.setdefault(name, default)
    # the VRAM preload is the least budget of every model, also of those whose handler or extension sets one (here or later) or a _budget
    # attribute: the larger of both, the whole model (budget 0) staying whole; models kept in VRAM together (co-tenants) share it beyond their
    # own budgets, the main one first (mmgp minimumBudget, preloadPriority)
    kwargs["minimumBudget"] = preload

    if "transformer2" in pipe:
        if profile in [3,4]:
            kwargs["pinnedMemory"] = [name for name in pipe if name.startswith("transformer")]  # every transformer stage (TRELLIS.2 has five)
    
    if profile == 4.5:
        kwargs["asyncTransfers"] = False
    elif profile == 3.5:
        kwargs["pinnedMemory"] = False
    if is_mps:
        kwargs["pinnedMemory"] = False
        kwargs["asyncTransfers"] = False

    scaled_fp16.install_globally(processing_device)  # once per process: every BF16 convolution on the FP16 tensor cores where the route applies (Turing)
    return mmgp_profile

reset_prompt_enhancer_requested = False
def unload_prompt_enhancer_runtime():
    deepy_controller._unload_prompt_enhancer_runtime(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model)


def release_prompt_enhancer():
    # after a prompt enhancement: a context the language model keeps (Deepy kept loaded, the enhancement ran beside it) stays, its models
    # parked: the next GPU user takes from them only the VRAM it needs (their blocks, then the KV cache, saved first); otherwise they leave
    engine = getattr(prompt_enhancer_llm_model, "_prompt_enhancer_vllm_engine", None)
    if engine is not None and engine.has_live_context() and enhancer_offloadobj is not None and enhancer_offloadobj.park(release="blocks"):
        return
    unload_prompt_enhancer_runtime()
    if enhancer_offloadobj is not None:
        enhancer_offloadobj.unload_all()


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

def setup_prompt_enhancer(pipe, kwargs, gen=None):
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    model_no = server_config.get("enhancer_enabled", 0) 
    if model_no != 0:
        from shared.prompt_enhancer import load_prompt_enhancer_runtime

        runtime = load_prompt_enhancer_runtime(
            partial(process_files_def, gen=gen),
            enhancer_enabled=model_no,
            qwen_backend=server_config.get("prompt_enhancer_quantization", "quanto_int8"),
            speculative_decoding=normalize_prompt_enhancer_speculative_decoding(server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)),
            deepy_kv_cache_quantization=server_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT),
        )
        prompt_enhancer_image_caption_model = runtime.image_caption_model
        prompt_enhancer_image_caption_processor = runtime.image_caption_processor
        prompt_enhancer_llm_model = runtime.llm_model
        prompt_enhancer_llm_tokenizer = runtime.llm_tokenizer
        pipe.update(runtime.pipe_models)
        if runtime.budgets:
            kwargs.setdefault("budgets", {}).update(runtime.budgets)
        if runtime.co_tenants:
            kwargs.setdefault("coTenantsMap", {}).update(runtime.co_tenants)
    else:
        reset_prompt_enhancer()


def ensure_prompt_enhancer_loaded(override_profile=-1, progress=None, send_cmd=None, gen=None):
    global enhancer_offloadobj

    reset_prompt_enhancer_if_requested()
    check_download_cancelled(gen)
    if enhancer_offloadobj is None:
        from shared.prompt_enhancer import download_prompt_enhancer_assets

        download_prompt_enhancer_assets(
            enhancer_enabled=server_config.get("enhancer_enabled", 0),
            qwen_backend=server_config.get("prompt_enhancer_quantization", "quanto_int8"),
            speculative_decoding=normalize_prompt_enhancer_speculative_decoding(server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)),
            send_cmd=send_cmd,
            progress=progress,
            gen=gen,
        )
        if progress is not None:
            progress(0, "Please Wait While Loading Prompt Enhancer")
        def loading_progress(phase, completed, total, model_id):
            progress((completed, total), desc=f"Loading - {phase} Prompt Enhancer", unit="items")

        loading_callback = offload.LoadingCallback(lambda: gen is not None and gen.get("abort", False), loading_progress) if progress is not None else None
        with model_unload_guard():
            if enhancer_offloadobj is None:
                try:
                    with offload.loading_context(loading_callback):
                        kwargs = {}
                        pipe = {}
                        setup_prompt_enhancer(pipe, kwargs, gen=gen)
                        llm_model = pipe.get("llm_model")
                        mmgp_engine = getattr(llm_model, "_lm_engine_mmgp", False)
                        # an LLM on MMGP's engine runs with profile 0 whatever the model's profile: as many of its blocks in VRAM as fit beside
                        # its KV cache and activations (dynamic preload), the blocks it streams pinned while it runs, in reserved RAM shared with
                        # the other models (the profile's); with CPU compute, the CPU processes the blocks out of VRAM while decoding when faster
                        mmgp_profile = init_pipe(pipe, kwargs, 0 if mmgp_engine else compute_profile(override_profile, "video"))
                        kwargs["pinnedMemory"] = False
                        if mmgp_engine:
                            cpu_compute = server_config.get("llm_cpu_compute", "auto")
                            kwargs.update(cudaGraphs=["llm_model"] if llm_model._lm_engine_mmgp_graphs else False, autoPreload=True, smartPinning=0,
                                          cpuCompute=["llm_model"] if cpu_compute != "off" else False, cpuComputeForce=cpu_compute == "on")
                        enhancer_offloadobj = offload.profile(pipe, profile_no=mmgp_profile, loading_callback=loading_callback, **kwargs)
                except (DownloadCancelled, offload.LoadingCancelled, InterruptedError):
                    reset_prompt_enhancer()
                    reset_prompt_enhancer_if_requested()
                    raise

    if prompt_enhancer_llm_model is None or prompt_enhancer_llm_tokenizer is None:
        raise gr.Error("Prompt enhancer text runtime is not available.")
    return prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer

def enhance_prompts_in_process(texts, instructions, max_tokens, seed):
    # the prompt enhancer from inside a running process, which owns the GPU (shared.prompt_enhancer.in_process): for a local engine, loaded
    # for the call, the models of the process leave the VRAM first (MMGP brings them back at their next forward)
    local = not is_remote_engine(resolve_role_engine(server_config, "prompt_enhancer"))
    try:
        if local:
            if offloadobj is not None:
                offloadobj.unload_all()
            offload_registry.unload_vram()
            ensure_prompt_enhancer_loaded()
        return process_prompt_enhancer("", {}, "T", texts, None, None, False, False, seed, prompt_enhancer_instructions=instructions, text_encoder_max_tokens=max_tokens)
    finally:
        if local:
            release_prompt_enhancer()

prompt_enhancer_in_process.register(enhance_prompts_in_process, lambda: server_config.get("enhancer_enabled", 0) > 0)

@controlled_model_loading
def load_models(model_type, override_profile = -1, output_type="video", config_id = None, runtime_model_type=None, track_as_main=True, gen=None, loading_callback=None, overrides={}, extension=None, **model_kwargs):
    global transformer_type, loaded_profile, loaded_config, loaded_compile, loaded_quantization, loaded_output_type
    compilation = compute_compile(overrides.get("compilation", ""), compile)
    transformer_qtype, text_encoder_qtype = compute_quantization(overrides)
    ram_debug_mark(f"loading {model_type}")
    def _load_models_info(message):
        if int(verbose_level) > 0:
            print(message)

    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    if config_id is not None and len(config_id):
        config_groups = get_model_config_groups(model_type, model_def)
        model_def = model_def.copy()
        for _, _, current_config in model_config_groups.selected_model_configs(config_groups, config_id):
            model_def.update(current_config)
    resolver = getattr(model_types_handlers[base_model_type], "resolve_runtime_model_def", None)
    if resolver is not None:
        model_def = resolver(model_def, dict(server_config, transformer_quantization=transformer_qtype,
            text_encoder_quantization=text_encoder_qtype, mixed_precision=server_config.get("mixed_precision", "0"),
            vae_precision="16"))  # VAEs always run in 16-bit
    save_quantized = args.save_quantized and model_def != None
    model_filename = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_qtype, dtype_policy = transformer_dtype_policy, model_def=model_def)
    if "URLs2" in model_def:
        model_filename2 = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_qtype, dtype_policy = transformer_dtype_policy, submodel_no=2, model_def=model_def) # !!!!
    else:
        model_filename2 = None
    modules = get_model_recursive_prop(model_type, "modules", return_list=True, model_def=model_def)
    modules = [get_model_recursive_prop(module, "modules", sub_prop_name  ="_list",  return_list= True) if isinstance(module, str) else module for module in modules ]
    if save_quantized and "quanto" in model_filename:
        save_quantized = False
        print("Need to provide a non quantized model to create a quantized model to be saved") 
    if save_quantized and len(modules) > 0:
        print(f"Unable to create a finetune quantized model as some modules are declared in the finetune definition. If your finetune includes already the module weights you can remove the 'modules' entry and try again. If not you will need also to change temporarly the model 'architecture' to an architecture that wont require the modules part ({modules}) to quantize and then add back the original 'modules' and 'architecture' entries.")
        save_quantized = False
    quantizeTransformer = not save_quantized and model_def !=None and transformer_qtype in ("int8", "fp8") and model_def.get("auto_quantize", False) and not "quanto" in model_filename
    if quantizeTransformer and len(modules) > 0:
        print(f"Autoquantize is not yet supported if some modules are declared")
        quantizeTransformer = False
    model_family = get_model_family(model_type)
    transformer_dtype = get_transformer_dtype(model_type, transformer_dtype_policy)
    if quantizeTransformer or "quanto" in model_filename:
        transformer_dtype = torch.bfloat16 if "bf16" in model_filename or "BF16" in model_filename else transformer_dtype
        transformer_dtype = torch.float16 if "fp16" in model_filename or"FP16" in model_filename else transformer_dtype
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
            model_file_list.append(get_model_filename(model_type, transformer_qtype, transformer_dtype, URLs = URLs1))
            URLs2 = module_type.get("URLs2", None)
            if URLs2 is None: raise Exception(f"No URL2s defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_qtype, transformer_dtype, URLs = URLs2))
            model_type_list += [model_type] * 2
            source_type_list += [1] * 2
            model_submodel_no_list += [1,2]
        else:
            model_file_list.append(get_model_filename(model_type, transformer_qtype, transformer_dtype, module_type= module_type))
            model_type_list.append(model_type)
            source_type_list.append(True)
            model_submodel_no_list.append(0) 

    local_model_file_list= []
    for filename, file_model_type, file_source_type, submodel_no in zip(model_file_list, model_type_list, source_type_list, model_submodel_no_list):
        if len(filename) == 0: continue 
        download_models(filename, file_model_type, file_source_type, submodel_no, model_def = model_def, gen=gen)
        local_file_name = fl.get_local_model_filename(filename )
        local_model_file_list.append( os.path.basename(filename) if local_file_name is None else local_file_name )
    if len(local_model_file_list) == 0:
        download_models("", model_type, 0, -1, model_def = model_def, gen=gen)

    VAE_dtype = torch.float16  # VAEs always run in 16-bit (models whose VAE overflows in FP16 use BF16)
    mixed_precision_transformer =  server_config.get("mixed_precision","0") == "1"
    if track_as_main:
        transformer_type = None

    for source_type, filename in zip(source_type_list, local_model_file_list):
        if source_type==0:  
            _load_models_info(f"Loading Model '{filename}' ...")
        elif source_type==1:  
            _load_models_info(f"Loading Module '{filename}' ...")


    model_type_handler = model_types_handlers[base_model_type] 
    text_encoder_URLs= get_model_recursive_prop(model_type, "text_encoder_URLs", return_list=True, model_def=model_def)
    if text_encoder_URLs is not None:
        text_encoder_filename = get_model_filename(model_type=model_type, quantization= text_encoder_qtype, dtype_policy = transformer_dtype_policy, URLs=text_encoder_URLs)
    if text_encoder_filename is not None and len(text_encoder_filename):
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        if text_encoder_filename is not None:
            download_models(text_encoder_filename, file_model_type, 2, -1, force_path =text_encoder_folder, model_def = model_def, gen=gen)
            text_encoder_filename =  fl.get_local_model_filename(text_encoder_filename, extra_paths=text_encoder_folder)
            _load_models_info(f"Loading Text Encoder '{text_encoder_filename}' ...")


    check_download_cancelled(gen)
    profile = 0 if extension is not None else compute_profile(override_profile, output_type)
    # an LM decoder uses every optimization the installation supports (CUDA graphs with any memory profile, Triton, FlashAttention 2)
    lm_features = resolve_lm_features() if model_def.get("lm_decoder") else None
    if lm_features is not None:
        print(f"LM decoder: {lm_features.describe()}")
    # profile 0 with a Manual VRAM preload: an LM decoding token by token still keeps the layers that fit in VRAM (dynamic preload)
    lm_engine_preload = lm_features is not None and lm_features.graphs and int(profile) == 0 and not dynamic_preload(profile, output_type)
    preview_mode = server_config.get("generation_preview", "rgb")
    preview_name = preview_path = preview_decoder = None
    if preview_mode not in ("rgb", "tiny_vae_frames", "tiny_vae_video"):
        raise ValueError("generation_preview must be rgb, tiny_vae_frames or tiny_vae_video")
    if preview_mode != "rgb":
        from shared.tinyvae.decoder import decoder_for, prepare_decoder, load_decoder
        preview_name = decoder_for(base_model_type, model_def)
        if preview_name is not None:
            if not getattr(offload.offload, "supports_cotenant_wildcards", False):
                raise RuntimeError("GPU TinyVAE previews require MMGP with wildcard cotenant support. Please update MMGP.")
            preview_path = prepare_decoder(preview_name, gen=gen)
    loading_model_ids = {filename: "transformer" if i == 0 else f"transformer {i + 1}" for i, filename in enumerate(local_model_file_list)}
    if text_encoder_filename:
        loading_model_ids[text_encoder_filename] = "text_encoder"
    if preview_path is not None:
        loading_model_ids[preview_path] = "tiny_vae"
    with model_unload_guard(), offload.loading_context(loading_callback, loading_model_ids):
        torch.set_default_device(None) # cpu, without the DeviceContext mode that routes every torch call through Python
        if preview_path is not None:
            preview_decoder = load_decoder(preview_name, preview_path)
        wan_model, pipe = model_type_handler.load_model(
                    local_model_file_list, runtime_model_type or model_type, base_model_type, model_def, quantizeTransformer = quantizeTransformer, text_encoder_quantization = text_encoder_qtype,
                    dtype = transformer_dtype, VAE_dtype = VAE_dtype, mixed_precision_transformer = mixed_precision_transformer, save_quantized = save_quantized, submodel_no_list   = model_submodel_no_list, text_encoder_filename = text_encoder_filename, profile=profile, lm_features=lm_features, **model_kwargs )

        kwargs = {}
        if "pipe" in pipe:
            kwargs = pipe
            pipe = kwargs.pop("pipe")
        if "coTenantsMap" not in kwargs: kwargs["coTenantsMap"] = {}
        if preview_decoder is not None:
            pipe["tiny_vae"] = preview_decoder
            kwargs["coTenantsMap"]["tiny_vae"] = "*"
        mmgp_profile = init_pipe(pipe, kwargs, profile, output_type, extension=extension)
        if extension is None and (lm_engine_preload or dynamic_preload(mmgp_profile, output_type)):
            kwargs["autoPreload"] = auto_preload_store(model_type)
        loras_transformer = kwargs.pop("loras", [])
        if "transformer" in pipe:
            loras_transformer += ["transformer"]
        if "transformer2" in pipe:
            loras_transformer += ["transformer2"]
        compile_modules = model_def.get("compile", compilation) if compilation == "transformer" else False
        custom_compile = compilation == "transformer" and hasattr(wan_model, "custom_compile")
        int8_backend.prepare_compile_cache(compile_modules or custom_compile)
        if custom_compile:
            wan_model.custom_compile(backend= "inductor", mode ="default")
        if compilation == "cudagraph":
            model_graphs, handler_graphs = model_def.get("cuda_graphs", False), kwargs.get("cudaGraphs", False)  # the handler's: an LM's decoding steps
            if not model_graphs:
                _load_models_info("CUDA Graphs are not supported for this Model")
            kwargs["cudaGraphs"] = True if True in (model_graphs, handler_graphs) else list(dict.fromkeys([*(model_graphs or []), *(handler_graphs or [])])) or False
        if compile_modules == False and compilation == "transformer":
            _load_models_info("Pytorch compilation is not supported for this Model")
        # kwargs["pinnedMemory"] = "text_encoder"
        if "prefetch_window" in server_config: kwargs["prefetchWindow"] = server_config["prefetch_window"]
        if server_config.get("smart_memory_pinning", True): kwargs["smartPinning"] = 0 # the models processed block by block that the profile does not pin cross a staging ring
        offloadobj = offload.profile(pipe, profile_no= mmgp_profile, compile = compile_modules, quantizeTransformer = False, loras = loras_transformer, vram_safety_coefficient = vram_safety_coefficient , convertWeightsFloatTo = transformer_dtype, loading_callback=loading_callback, **kwargs)
        if kwargs.get("cudaGraphs"):
            check_cuda_graph_weights(offloadobj)
    offloadobj.tiny_vae = preview_decoder
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)
    if track_as_main:
        transformer_type = model_type
        loaded_profile = profile
        loaded_output_type = output_type
        loaded_config = config_id or ""
        loaded_compile = compilation
        loaded_quantization = (transformer_qtype, text_encoder_qtype)
    return wan_model, offloadobj

if not "P" in preload_model_policy:
    wan_model, offloadobj, transformer = None, None, None
    reload_needed = True
else:
    wan_model, offloadobj = load_models(
        transformer_type,
        output_type=get_profile_type_for_model(transformer_type, 0),
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
    return get_default_attention_mode()

def generate_model_description(model_type):
    description_container = [""]
    model_name = get_model_name(model_type, description_container)
    model_def = get_model_def(model_type) or {}
    description  = description_container[0]
    description = model_infos.render_model_description(description, model_def.get("infos", None), model_type=model_type, model_name=model_name, height=60 if server_config.get('display_stats', 0) == 1 else 40)
    return description


def generate_header(model_type, compile, attention_mode, overrides="", guidance_phases=1, video_prompt_type=""):
    overrides = generation_overrides.parse_overrides(overrides)
    override_attention = overrides.get("attention_mode", "")
    compile = compute_compile(overrides.get("compilation", ""), compile)
    description = generate_model_description(model_type)
    model_def = get_model_def(model_type) or {}
    full_filename = get_model_filename(model_type, compute_quantization(overrides)[0], transformer_dtype_policy)
    model_filename = os.path.basename(full_filename)
    model_attention = get_overridden_attention(model_type)
    overridden_attention = override_attention or model_attention
    attn_mode = attention_mode if overridden_attention is None else overridden_attention
    header = "<DIV style='align:right;width:100%'><FONT SIZE=2>Attention mode <B>" + (attn_mode if attn_mode!="auto" else "auto/" + get_auto_attention() )
    custom_attention = model_def.get("custom_attention_modes", {}).get(attn_mode)
    if custom_attention is not None:
        custom_status = ATTENTION_MODE_AVAILABILITY.get(attn_mode, {})
        custom_installed = custom_attention.get("installed", custom_status.get("installed", True))
        custom_supported = custom_attention.get("supported", custom_status.get("supported", custom_installed))
    else:
        custom_installed = custom_supported = False
    if attn_mode not in override_attention_modes_installed and not custom_installed:
        header += " -NOT INSTALLED-"
    elif attn_mode not in override_attention_modes_supported and not custom_supported:
        header += " -NOT SUPPORTED-"
    elif not override_attention and model_attention is not None and attention_mode != model_attention:
        header += " -MODEL SPECIFIC-"
    header += "</B>"

    if compile == "transformer":
        header += ", Pytorch compilation <B>ON</B>"
    elif compile == "cudagraph":
        header += ", CUDA Graphs <B>ON</B>"
    if "fp16" in model_filename:
        header += ", Data Type <B>FP16</B>"
    else:
        header += ", Data Type <B>BF16</B>"

    quant_label = quant_router.detect_quantization_label_from_filename(fl.get_local_model_filename(full_filename))
    if quant_label:
        header += f", Quantization <B>{quant_label}</B>"
    if server_config.get("enable_4k_resolutions", 0) == 1 and not (model_def.get("audio_only") or model_def.get("mesh_outputs") or model_def.get("lock_resolutions")):
        mode = resolution_utils.resolution_mode(model_def, guidance_phases, video_prompt_type)
        maximum = model_def.get("resolution_budget") or resolution_utils.DEFAULT_RESOLUTION_BUDGET
        if mode == "two_phases_tiled":
            maximum = model_def.get("resolution_budget_tiled") or maximum
        if resolution_utils.resolution_pixels(maximum) == resolution_utils.resolution_pixels(resolution_utils.DEFAULT_RESOLUTION_BUDGET):
            maximum = "1920x1080"
        elif isinstance(maximum, int):
            side = math.isqrt(maximum)
            maximum = f"{'≈' if side * side != maximum else ''}{side}x{side}"
        header += f", Max Recommended Res.: <B>{maximum}</B>"
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

def build_callback(state, pipe, send_cmd, status, num_inference_steps, preview_meta=None, tiny_preview=None):
    gen = get_gen_info(state)
    gen["num_inference_steps"] = num_inference_steps
    generation_start_time = last_refresh_time = time.time()
    denoising_start_time = None
    estimated_total_time = None
    def callback(step_idx = -1, latent = None, force_refresh = True, read_state = False, override_num_inference_steps = -1, pass_no = -1, preview_meta=preview_meta, denoising_extra ="", progress_unit = None, status_prefix = "", progress_title = None):
        nonlocal denoising_start_time, estimated_total_time, last_refresh_time
        step_completed = step_idx >= 0
        in_pause = False
        with gen_lock:
            process_status = gen.get("process_status", None)
            pause_msg = None
            if isinstance(process_status, str) and process_status.startswith("request:"):
                set_generation_state(gen, process_status="process:" + process_status[len("request:"):])
                # a requester that parks the generation (Deepy) gets the VRAM it needs from the models, which stay loaded otherwise
                parked = gen.pop("park_request", False) and offloadobj.park()
                if not parked:
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
                        set_generation_state(gen, process_status="process:" + process_status[len("request:"):])
                        if not gen.pop("park_request", False) and parked:  # e.g. a pause: the VRAM is freed for good
                            offloadobj.unload_all()
                            parked = False
                        continue
                    if process_status == "process:main": break
            if parked:
                offloadobj.unpark()
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
        current_time = time.time()
        if force_refresh and (step_idx < 0 or progress_title is not None):
            denoising_start_time = current_time
            estimated_total_time = None
        if force_refresh or step_idx >= 0:
            pass
        elif read_state:
            if current_time - last_refresh_time < 1:
                return
        else:
            refresh_id =  gen.get("refresh", -1)
            if refresh_id < 0:
                return
            UI_refresh = state.get("refresh", 0)
            if UI_refresh >= refresh_id:
                return  
        if override_num_inference_steps > 0 and progress_title is None:
            gen["num_inference_steps"] = override_num_inference_steps
             
        num_inference_steps = gen.get("num_inference_steps", 0)
        if read_state:
            num_inference_steps, progress_unit = gen.get("phase_progress_units", (num_inference_steps, progress_unit))
        else:
            if progress_title is not None:
                num_inference_steps = override_num_inference_steps
            gen["phase_progress_units"] = (num_inference_steps, progress_unit)
        status = status_prefix or gen["progress_status"]
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
            elif progress_title is not None:
                phase = progress_title
            elif step_idx  == num_inference_steps:
                phase = "VAE Decoding"    
            else:
                if pass_no <=0:
                    phase = "Denoising"
                elif pass_no == 1:
                    phase = "Denoising First Phase"
                elif pass_no == 2:
                    phase = "Denoising Second Phase"
                elif pass_no == 3:
                    phase = "Denoising Third Phase"
                else:
                    phase = f"Denoising {pass_no}th Phase"

                if len(denoising_extra) > 0: phase += " | " + denoising_extra

            gen["progress_phase"] = (phase, step_idx)
        status_msg = merge_status_context(status, phase)      

        elapsed_time = current_time - generation_start_time
        progress_time = format_time(elapsed_time)
        if step_completed and step_idx > 0 and num_inference_steps > 0 and denoising_start_time is not None:
            denoising_elapsed_time = current_time - denoising_start_time
            estimated_total_time = denoising_start_time - generation_start_time + max(denoising_elapsed_time, denoising_elapsed_time * num_inference_steps / step_idx)
        if estimated_total_time is not None:
            progress_time += f" / {format_time(max(elapsed_time, estimated_total_time))}"
        status_msg = merge_status_context(status, f"{phase} | {progress_time}")
        if step_idx >= 0:
            progress_args = [(step_idx , num_inference_steps) , status_msg  ,  num_inference_steps]
            if progress_unit:
                progress_args.append(progress_unit)
        else:
            progress_args = [0, status_msg]
        
        # progress(*progress_args)
        send_cmd("progress", progress_args)
        last_refresh_time = current_time
        if latent is not None:
            payload = pipe.prepare_preview_payload(latent, preview_meta) if hasattr(pipe, "prepare_preview_payload") else latent
            if tiny_preview is not None:
                lat = payload["latents"] if isinstance(payload, dict) else payload
                if lat is not None:
                    if preview_meta is not None and preview_meta.get("first_latent_only", False):
                        lat = lat[:, :1]
                    tiny_preview.capture(lat, step_idx - 1, num_inference_steps, pass_no)
                return
            if isinstance(payload, dict):
                data = payload.copy()
                lat = data.get("latents")
                if torch.is_tensor(lat):
                    if preview_meta is not None and preview_meta.get("first_latent_only", False) and lat.ndim == 4:
                        lat = lat[:, :1]
                    data["latents"] = lat.to("cpu")
                payload = data
            elif torch.is_tensor(payload):
                if preview_meta is not None and preview_meta.get("first_latent_only", False) and payload.ndim == 4:
                    payload = payload[:, :1]
                payload = payload.to("cpu")
            if payload is not None:
                send_cmd("preview", payload)
            
        # gen["progress_args"] = progress_args
            
    callback.checkpoint = lambda: callback(-1, None, False, True)
    callback.tiny_vae = tiny_preview is not None
    return callback

_pause_generation_lock = threading.Lock()


def pause_generation(state):
    # Concurrent pages must not queue a second pause behind the first resume.
    if not _pause_generation_lock.acquire(blocking=False):
        return
    try:
        _pause_generation(state)
    finally:
        _pause_generation_lock.release()


def _pause_generation(state):
    gen = get_gen_info(state)
    process_id = "pause"
    GPU_process_running = any_GPU_process_running(state, process_id, ignore_main= True )
    if GPU_process_running:
        gr.Info("Unable to pause, a PlugIn is using the GPU")
        return
    gen["resume"] = False
    pause_msg = "Generation on Pause, click Resume to Restart Generation"
    acquire_GPU_ressources(state, process_id , "Pause", gr= gr, custom_pause_msg= pause_msg, custom_wait_msg= "Please wait while the Pause Request is being Processed...")      
    gr.Info(pause_msg)
    while not gen.get("resume", False):
        time.sleep(0.5)

    release_GPU_ressources(state, process_id )
    gen["resume"] = False

def resume_generation(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    if str(gen.get("process_status", "")).startswith(("request:deepy_pause_", "process:deepy_pause_", "request:pause", "process:pause")):
        from shared.deepy.engine import resume_assistant
        session = state.get("assistant_session")
        if session is not None and session.media_tool_active:
            resume_assistant(session)

def abort_generation(state, client_id="", notify = True):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    service = service_for(state)
    in_progress = "in_progress" in gen or (service is not None and service.generation_running)
    with lock:
        if len(queue):
            if len(client_id):
                for i, task in enumerate(queue):
                    queue_client_id = task["params"].get("client_id","")
                    if queue_client_id == client_id:
                        if i == 0:
                            if not in_progress:
                                del queue[0]
                                if "prompt_no" in gen: gen["prompt_no"] += 1
                                return gr.update(), gr.HTML(value=generate_queue_html(queue))
                            break
                        del queue[i]
                        if "prompt_no" in gen: gen["prompt_no"] += 1
                        return gr.update(), gr.HTML(value=generate_queue_html(queue))
            elif not in_progress:
                del queue[0]
                gen["prompt_no"] += 1
                return gr.update(), gr.HTML(value=generate_queue_html(queue))

    gen["resume"] = True
    if in_progress:
        if wan_model != None:
            wan_model._interrupt= True
        gen["abort"] = True            
        msg = "Processing Request to abort Current Generation"
        set_generation_state(gen, status=msg)
        if service is not None:
            service._generation_aborting = True
            service._generation_event("status", "Aborting generation…")
        if notify:
            gr.Info(msg)
        return gr.Button(interactive=  False), gr.update()
    else:
        return gr.Button(interactive=  True), gr.update()

def early_stop_generation(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    queue = gen.get("queue", [])
    if not queue:
        return gr.Button(interactive=True)
    model_type = queue[0].get("params", {}).get("model_type")
    model_def = get_model_def(model_type) if model_type else None
    if not model_def or not model_def.get("supports_early_stop", False):
        gr.Info("Early Stop is not supported for this model.")
        return gr.Button(interactive=True)
    with gen_lock:
        if gen.get("early_stop", False):
            return gr.Button(interactive=False)
        gen["early_stop"] = True
        gen["early_stop_forwarded"] = False
    msg = "Early Stop Requested"
    set_generation_state(gen, status=msg)
    gr.Info(msg)
    return gr.Button(interactive=False)

def gallery_update(file_list, selected_index, **kwargs):
    visible, local_index, _ = gallery_window(file_list, selected_index, server_config['clear_file_list'])
    allow_listed_media(visible)
    return gr.Gallery(value=visible, selected_index=local_index, **kwargs)


def pack_strip_gallery_state(file_list, selected_index):
    """State values (paths, selected index, refresh trigger) of a strip gallery (audio, 3D)."""
    visible, _, offset = gallery_window(file_list, selected_index, server_config['clear_file_list'])
    allow_listed_media(visible)
    return [json.dumps({'paths': visible, 'offset': offset}), selected_index, time.time()]

def unpack_strip_list(packed_file_list):
    value = json.loads(packed_file_list)
    return value['paths'] if isinstance(value, dict) else value

def gallery_choice(gen, source):
    """Selected index of a gallery, following its last file while it was selected."""
    files = gen.get(source.files_key, [])
    if gen.get(source.last_selected_key, True):
        return max(len(files) - 1, 0 if source is VIDEO else -1)
    return gen.get(source.selected_key, 0 if source is VIDEO else -1)

def pack_strip_galleries(gen, choices=None):
    """State values of every strip gallery, in STRIP_SOURCES order (3 per gallery)."""
    choices = choices or {}
    return [value for source in STRIP_SOURCES for value in pack_strip_gallery_state(gen.get(source.files_key, []), choices.get(source.id, gallery_choice(gen, source)))]

def gallery_tab_updates(source_id):
    source = gallery_source(source_id)
    return [gr.Tabs(selected=source.tab_id), source_index(source_id)]

def can_extend_sample(params):
    if _is_edit_task_params(params) or params.get("mode") == "edit" or params.get("image_mode", 0) != 0:
        return False
    model_type = params["model_type"]
    base_model_type, model_def = get_base_model_type(model_type), get_model_def(model_type)
    preprocess_all = resolve_model_preprocess_all(model_def, base_model_type=base_model_type, video_prompt_type=params.get("video_prompt_type", ""), image_prompt_type=params.get("image_prompt_type", ""), audio_prompt_type=params.get("audio_prompt_type", ""), custom_settings=params.get("custom_settings", {}), params=params)
    return test_any_sliding_window(base_model_type) and not preprocess_all


def refresh_gallery(state): #, msg
    service = service_for(state)
    if service is not None:
        service.host_changed()
    gen = get_gen_info(state)

    # gen["last_msg"] = msg
    for source in GALLERY_SOURCES:
        clear_deleted_files(state, source.id)
    file_list = gen.get("file_list", [])
    choice = gallery_choice(gen, VIDEO)

    header_text = gen.get("header_text", "")
    in_progress = "in_progress" in gen
    last_source = gen.get("last_gallery_source", VIDEO.id)
    queue = gen.get("queue", [])
    abort_interactive = not gen.get("abort", False)
    early_stop_interactive = not gen.get("early_stop", False)
    early_stop_visible = False

    if gen.pop("refresh_tab", False):
        gen["current_gallery_source"] = last_source
        output_tabs = gallery_tab_updates(last_source)
    else:
        output_tabs = [gr.update(), gr.update()]

    if service is not None:
        output_tabs = gallery_tab_updates(gen["current_gallery_source"])
        last_source = VIDEO.id  # Synchronize the visual selection even while displaying another gallery.

    if not in_progress or len(queue) == 0:
        return *output_tabs, gallery_update(file_list, choice), choice if last_source == VIDEO.id else gr.update(), *pack_strip_galleries(gen), gr.HTML("", visible= False),  gr.Button(visible=True), gr.Button(visible=False), gr.Row(visible=False), gr.update(), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(interactive=  early_stop_interactive, visible= early_stop_visible), gr.Button(visible= False), gr.update(visible=False)
    else:
        task = queue[0]
        prompt =  task["prompt"]
        params = task["params"]
        model_type = params.get("model_type", "")
        is_edit_task = _is_edit_task_params(params)
        multi_prompts_gen_type = "FG" if is_edit_task else params["multi_prompts_gen_type"]
        if is_edit_task:
            base_model_type, model_def = None, None
        else:
            base_model_type, model_def = get_base_model_type(model_type), get_model_def(model_type)
        onemorewindow_visible = can_extend_sample(params)
        early_stop_visible = bool(model_def and model_def.get("supports_early_stop", False))
        enhanced = False
        if prompt.startswith(prompt_parser.ENHANCED_PROMPT_PREFIX):
            enhanced = True
            prompt = prompt[len(prompt_parser.ENHANCED_PROMPT_PREFIX):]
        prompt_units = [prompt] if is_edit_task else prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
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
        if is_edit_task:
            summary, prompt = task["prompt"], ""
        else:
            is_image = params["image_mode"] > 0
            audio_only = model_def.get("audio_only", False)
            count = params["batch_size"] if is_image else params["repeat_generation"]
            world_label = {"world": "World", "world_video": "World and Video", "video": "Video"}[params.get("world_output") or "world"] if model_def.get("world_outputs", False) else None
            summary = "New " + ("Audio" if audio_only else "3D Asset" if model_def.get("mesh_outputs", False) else world_label or ("Image" if is_image else "Video"))
            if count > 1:
                summary += f" x{count}"
            if audio_only:
                duration_def = model_def.get("duration_slider")
                duration = params["duration_seconds"]
                if duration_def is not None and duration > 0 and duration_def.get("name", duration_def["label"]).lower().startswith("max"):
                    summary += f", Max Duration is {duration:g} s"
            else:
                if not is_image:
                    frames = params["video_length"]
                    fps = get_computed_fps(params["force_fps"], base_model_type, params.get("video_guide"), params.get("video_source"))
                    summary += f", {frames} frames ({round(frames / fps, 1):g}s)"
                summary += f", {params['resolution']}, {params['num_inference_steps']} inference steps"

        details = f'<div class="generation-note">{html.escape(header_text)}</div>' if header_text else ""
        if prompt:
            prompt_label = "Enhanced Prompt" if enhanced else "Prompt"
            details += f'<div class="generation-prompt"><div class="generation-label">{prompt_label}</div><div>{prompt}</div></div>'
        thumbnails = ""
        for side in ("start", "end"):
            images, labels = task.get(f'{side}_image_data_base64'), task.get(f'{side}_image_labels')
            if images and labels:
                for i, (img_uri, img_label) in enumerate(zip(images, labels)):
                    label = html.escape(img_label)
                    thumbnails += f'<button type="button" class="generation-reference" onclick="showImageModal(\'current_{side}_{i}\')" aria-label="View {label}" title="View {label}"><img src="{img_uri}" alt="{label}" /><span>{label}</span></button>'
        if thumbnails:
            thumbnails = f'<td><div class="generation-references" role="group" aria-label="Generation References">{thumbnails}</div></td>'
        table = f'<div class="generation-table-wrap"><table id="PINFO"><tbody><tr><td class="generation-prompt-cell">{details}</td>{thumbnails}</tr></tbody></table></div>' if details or thumbnails else ""
        html_content = f'<div class="wangp-generation-info">{table}</div>'
        html_output = gr.HTML(html_content, visible= True)
        return *output_tabs, gallery_update(file_list, choice), choice if last_source == VIDEO.id else gr.update(), *pack_strip_galleries(gen), html_output, gr.Button(visible=False), gr.Button(visible=True), gr.Row(visible=True), gr.update(), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(interactive=  early_stop_interactive, visible= early_stop_visible), gr.Button(visible= onemorewindow_visible), gr.update(label=summary, visible=True)



def finalize_generation(state):
    service = service_for(state)
    return service.finalized_updates() if service is not None else _finalize_generation(state)


def _finalize_generation(state):
    gen = get_gen_info(state)
    if "in_progress" in gen:
        del gen["in_progress"]
    gen["extra_orders"] = 0
    last_source = gen.get("last_gallery_source", VIDEO.id)
    gen["current_gallery_source"] = last_source
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    release_ram_cache()  # the queue is done: the RAM a next generation of the same model would have reused goes back to the system
    gen["early_stop"] = False
    gen["early_stop_forwarded"] = False
    return *gallery_tab_updates(last_source), gallery_update(gen.get("file_list", []), gallery_choice(gen, VIDEO)) if last_source == VIDEO.id else gr.update(), *pack_strip_galleries(gen), gr.Button(interactive=  True), gr.Button(interactive=  True, visible= False), gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.HTML(visible= False, value="")

def get_default_video_info():
    return "Please Select a Video / Image"    


def get_file_list(state, input_file_list, source_id = VIDEO.id):
    gen = get_gen_info(state)
    source = gallery_source(source_id)
    with lock:
        file_list_name, file_settings_name = source.files_key, source.settings_key
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

def set_file_choice(gen, file_list, choice, source_id = VIDEO.id):
    source = gallery_source(source_id)
    if len(file_list) > 0: choice = max(choice,0)
    gen[source.last_selected_key] = (choice + 1) >= len(file_list)
    gen[source.selected_key] = choice
    gen["current_gallery_source"] = source.id
    gen["selected_video_time"] = None if source is not VIDEO or choice < 0 or choice >= len(file_list) or not has_video_file_extension(file_list[choice]) else 0.0
    if _deepy_hybrid is not None and gen is _deepy_hybrid._state["gen"]:
        _deepy_hybrid.host_changed()

def get_selected_late_processing_tabs_visibility(state):
    gen = get_gen_info(state)
    source = gallery_source(gen.get("current_gallery_source", VIDEO.id))
    files = gen.get(source.files_key, [])
    choice = gen.get(source.selected_key, 0 if source is VIDEO else -1)
    if len(files) > 0:
        choice = min(len(files) - 1, max(choice, 0))
    if choice < 0 or choice >= len(files) or not os.path.isfile(files[choice]):
        return False, False, False
    is_audio = has_audio_file_extension(files[choice])
    is_video = has_video_file_extension(files[choice])
    is_image = has_image_file_extension(files[choice])
    return is_audio, is_video or is_image or mesh_postprocessing.is_mesh_file(files[choice]), is_video


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
    guide_custom_choices = get_guide_custom_choices(model_def, settings.get("image_mode", 0))
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


def format_media_info(file_name, configs):
    """Shared selected-media details for Gradio and the standalone Deepy app."""
    model_def = None
    is_audio = False
    is_image = False
    is_video = False
    is_mesh = False
    is_world = False
    is_app = False
    is_deleted = False
    if file_name is not None:
        values = [html.escape(os.path.basename(strip_virtual_media_suffix(file_name)))]
        labels = [ "File Name"]
        prompt_labels = {"Prompt", "Negative Prompt"}
        misc_values= []
        misc_labels = []
        pp_values= []
        pp_labels = []
        configs_summary = ""
        nb_audio_tracks =  0

        if not os.path.isfile(file_name):
            is_deleted = True
            configs = None
        elif has_audio_file_extension(file_name):
            is_audio = True
            width, height = 0, 0
            frames_count = fps = 1
        elif has_mesh_file_extension(file_name):
            from shared.utils.mesh_outputs import asset_stats
            is_mesh = True
            width, height = 0, 0
            frames_count = fps = 1
            stats = asset_stats(file_name)
            if stats["kind"] == "splats":
                values += [html.escape(f"{stats['gaussians']:,} Gaussians, SH degree {stats['sh_degree']}")]
                labels += ["3D Splats"]
            else:
                textures = ", ".join(f"{w}x{h}" for w, h in stats["textures"])
                values += [html.escape(f"{stats['faces']:,} faces, {stats['vertices']:,} vertices" + (f", textures {textures}" if textures else ", untextured"))]
                labels += ["3D Asset"]
        elif has_world_file_extension(file_name):
            from shared.utils.world_outputs import read_manifest, video_path as world_video_path
            is_world = True
            manifest = read_manifest(file_name)  # the world's own video (its history), whether it was saved or not
            width, height, fps = manifest["width"], manifest["height"], manifest["fps"]
            frames_count = sum(segment["video_frames"] for segment in manifest["segments"])
            values += [html.escape(f"{len(manifest['segments'])} segments" + (", its video in the Video gallery" if os.path.isfile(world_video_path(file_name)) else ""))]
            labels += ["World"]
        elif has_app_file_extension(file_name):
            from shared.utils.app_outputs import read_app_metadata
            is_app = True
            width, height = 0, 0
            frames_count = fps = 1
            app = read_app_metadata(file_name)
            values += [html.escape(f"{app['app_name']} (version {app['app_version']})" + (f": {app['app_description']}" if app["app_description"] else ""))]
            labels += ["Web App"]
            if app["app_controls"]:
                values += [html.escape(", ".join(f"{key}: {action}" for key, action in app["app_controls"].items()))]
                labels += ["Controls"]
        elif not has_video_file_extension(file_name):
            img = _open_image_input(file_name)
            width, height = img.size
            is_image = True
            frames_count = fps = 1
        else:
            fps, width, height, frames_count = get_video_info(file_name)
            is_video = True

        if is_audio or is_video:
            duration, layouts = get_media_duration_and_audio_layouts(file_name)
            actual_duration = f"{duration:.2f}s" if duration is not None else "Unknown"
            if len(layouts) > 1:
                layouts = [f"Track {index + 1}: {layout}" for index, layout in enumerate(layouts)]
            audio_summary = "; ".join(layouts) or "No audio"
            if is_audio:
                values += [html.escape(f"{actual_duration}, {audio_summary}")]
                labels += ["Duration"]
            nb_audio_tracks = len(layouts) if is_video else 0
            audio_tracks_summary = html.escape(f"{nb_audio_tracks}, {audio_summary}")

        if configs != None:
            # Deepy reference IDs alone do not describe a generation model.
            if configs.get("type") or configs.get("model_type"):
                video_model_name = configs.get("type") or get_model_name(configs["model_type"])
                if "-" in video_model_name: video_model_name = video_model_name[video_model_name.find("-")+2:]
                misc_values += [video_model_name]
                misc_labels += ["Model"]
            metadata_model_def = get_model_def(configs.get("model_type", None))
            config_groups = get_model_config_groups(configs.get("model_type", None), metadata_model_def) if metadata_model_def is not None else [{} for _ in model_config_groups.CONFIG_KEYS]
            configs_summary = model_config_groups.format_config_selection(config_groups, configs.get("config", ""))
            model_modes_def = metadata_model_def.get("model_modes", None) if metadata_model_def is not None else None
            if model_modes_def is not None and "model_mode" in configs and configs.get("image_mode", 0) in model_modes_def.get("image_modes", [0, 1, 2]):
                misc_values += [next((label for label, value in model_modes_def["choices"] if value == configs["model_mode"]), configs["model_mode"])]
                misc_labels += [model_modes_def["label"]]
            video_temporal_upsampling = temporal_upsampler_api.format_temporal_upsampling_label(temporal_upsampler_api.normalize_temporal_upsampling_value(configs.get("temporal_upsampling", "")))
            video_spatial_upsampling = upsampler_api.format_upsampling_label(upsampler_api.normalize_upsampling_value(configs.get("spatial_upsampling", "")))
            video_film_grain_intensity = configs.get("film_grain_intensity", 0)
            video_film_grain_saturation = configs.get("film_grain_saturation", 0.5)
            video_postprocess_audio = audio_processor_api.normalize_method(configs.get("postprocess_audio", "") or "")
            video_postprocess_audio_meta = audio_processor_api.method_metadata(video_postprocess_audio)
            video_postprocess_audio_prompt = configs.get("postprocess_audio_prompt", "")
            video_postprocess_audio_neg_prompt = configs.get("postprocess_audio_neg_prompt", "")
            video_seed = configs.get("seed", -1)
            video_postprocess_audio_seed = configs.get("postprocess_audio_seed", video_seed)
            video_replace_voice_method = audio_processor_api.normalize_method(configs.get("replace_voice_method", "") or "")
            if is_mesh:  # 3D assets: the 3D Refiner (spatial slot), then the Texturer (temporal slot)
                for value, label in ((video_spatial_upsampling, upsampler_api.slot_label(upsampler_api.UPSAMPLER_PROFILE_3D)), (video_temporal_upsampling, temporal_upsampler_api.slot_label(temporal_upsampler_api.MEDIA_3D))):
                    if value:
                        pp_values += [value]
                        pp_labels += [label]
            else:
                if len(video_spatial_upsampling) > 0:
                    video_temporal_upsampling += " " + video_spatial_upsampling
                if len(video_temporal_upsampling) > 0:
                    pp_values += [ video_temporal_upsampling ]
                    pp_labels += [ "Upsampling" ]
            if video_film_grain_intensity > 0:
                pp_values += [ f"Intensity={video_film_grain_intensity}, Saturation={video_film_grain_saturation}" ]
                pp_labels += [ "Film Grain" ]
            if video_postprocess_audio == "control":
                pp_values += [ "Control Video Audio Track" ]
                pp_labels += [ "Audio Postprocess" ]
            elif video_postprocess_audio:
                audio_summary = audio_processor_api.format_method_label(video_postprocess_audio)
                audio_details = []
                if video_postprocess_audio_meta["needs_prompt"] or video_postprocess_audio_meta["needs_negative_prompt"]:
                    audio_details.append(f'Prompt="{video_postprocess_audio_prompt}", Neg Prompt="{video_postprocess_audio_neg_prompt}", Seed={video_postprocess_audio_seed}')
                pp_values += [", ".join([audio_summary, *audio_details]) if audio_details else audio_summary]
                pp_labels += [ "Audio Postprocess" ]
            if video_replace_voice_method:
                pp_values += [ audio_processor_api.format_method_label(video_replace_voice_method) ]
                pp_labels += [ "Voice Replacement" ]


        if is_deepy_display_metadata(configs):
            values += ["Deepy"]
            labels += ["Made By"]
            video_prompt = html.escape(str(configs.get("prompt", "") or "")[:4096]).replace("\n", "<BR>")
            if len(video_prompt) > 0:
                values += [video_prompt]
                labels += ["Prompt"]
            video_creation_date = "Deleted" if is_deleted else resolve_media_creation_date(file_name, configs)
            if is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            elif is_video:
                values += [f"{width}x{height}", f"{frames_count} frames (duration={frames_count/fps:.1f}s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
            if nb_audio_tracks > 0:
                values += [audio_tracks_summary]
                labels += ["Nb Audio Tracks"]
            values += [video_creation_date]
            labels += ["Creation Date"]
        elif configs == None or not "seed" in configs:
            values += misc_values
            labels += misc_labels
            if configs is not None and configs.get("comments"):  # e.g. a 3D viewer snapshot, which has a comment only
                values += [html.escape(str(configs["comments"])[:4096]).replace("\n", "<BR>")]
                labels += ["Comments"]

            video_creation_date = "Deleted" if is_deleted else resolve_media_creation_date(file_name, configs)
            if is_audio:
                pass
            elif is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            elif is_video:
                values += [f"{width}x{height}",  f"{frames_count} frames (duration={frames_count/fps:.1f}s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
                extra_values, extra_labels = get_video_summary_extras(file_name)
                values += extra_values
                labels += extra_labels
            if nb_audio_tracks  > 0:
                values +=[audio_tracks_summary]
                labels +=["Nb Audio Tracks"]

            values += pp_values
            labels += pp_labels

            values +=[video_creation_date]
            labels +=["Creation Date"]
        else: 
            video_prompt_text = str(configs.get("prompt", "") or "")
            enhanced_video_prompt_text = str(configs.get("enhanced_prompt", "") or "")
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
            multi_prompts_gen_type = prompt_parser.normalize_multi_prompts_mode(configs.get("multi_prompts_gen_type"), "FG")
            prompt_history = prompt_parser.parse_prompt_history(video_prompt_text, enhanced_video_prompt_text, multi_prompts_gen_type)
            if prompt_history is not None:
                original_prompts, enhanced_prompts = prompt_history
                video_prompt_text = prompt_parser.serialize_prompt_units("", original_prompts, multi_prompts_gen_type)
                enhanced_video_prompt_text = prompt_parser.serialize_prompt_units("", enhanced_prompts, multi_prompts_gen_type)
            video_prompt = html.escape(video_prompt_text[:16384]).replace("\n", "<BR>")
            enhanced_video_prompt = html.escape(enhanced_video_prompt_text[:16384]).replace("\n", "<BR>")
            map_video_prompt  = {"V" : "Control Image" if image_outputs else "Control Video", ("VA", "U") : "Mask Image" if image_outputs else "Mask Video", "I" : "Reference Images", "&": "HDR Output"}
            map_image_prompt  = {"V" : "Source Video", "L" : "Last Video", "S" : "Start Image", "E" : "End Image"}
            map_audio_prompt  = {"A" : "Audio Source", "O": "Force Output Audio", "B" : "Audio Source #2", "D" : "Audio Source #3", "K": "Control Video Audio Track", "N": "Normalized Audio Volumes"}
            custom_audio_option_label, custom_audio_option_flag = get_audio_prompt_type_custom_option_def(model_def)
            if len(custom_audio_option_flag) > 0:
                map_audio_prompt[custom_audio_option_flag] = custom_audio_option_label
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
            custom_guide_def = model_def.get("custom_guide")
            if configs.get("custom_guide_used") and custom_setting_visible(custom_guide_def, video_video_prompt_type, video_audio_prompt_type, configs.get("model_mode")):
                video_other_prompts.append(html.escape(custom_guide_def.get("name", custom_guide_def.get("label", "Custom Guide"))))
            any_mask = "A" in video_video_prompt_type and not "U" in video_video_prompt_type            
            multiple_submodels = model_def.get("multiple_submodels", False)
            video_other_prompts = ", ".join(video_other_prompts)
            if is_audio or is_mesh or is_app:
                video_resolution = None
                video_length_summary = None
                video_length_label = ""
                original_fps = 0
                video_num_inference_steps = configs.get("num_inference_steps", None)
            elif is_world:  # the world's own video, from its manifest (instructions make it longer than video_length)
                video_resolution = f"{width}x{height}"
                video_length_label = "World History Duration"
                video_length_summary = f"{frames_count / fps:.1f}s, {frames_count} frames at {fps} fps"
                original_fps = fps
                video_num_inference_steps = configs.get("num_inference_steps", 0)
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
                video_resolution = configs.get("resolution", "")
                real_video_resolution = f"{width}x{height}"
                if video_resolution !=  real_video_resolution: video_resolution +=  f" (real: {real_video_resolution})"
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
                        video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale}" + ("" if video_switch_threshold ==0 else f" with Guidance Switch at Noise Level {video_switch_threshold}")
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
            lora_dir = None if video_model_type is None else get_lora_dir(video_model_type)
            video_activated_loras = [ f"<span class='copy-swap' tabindex=0><SPAN class='copy-swap__trunc' >{get_lora_local_path(None, lora)}</span><span class='copy-swap__full'>{get_lora_URL(lora_dir, lora) .split('|')[0]}</span></span>" for lora in video_activated_loras] 
            video_activated_loras = [ f"<TR><TD style='padding-top:0px;padding-left:0px;width:100%;max-width:0'>{lora}</TD><TD style='width:1%;white-space:nowrap;vertical-align:top'>x{str(multiplier).strip() or '1'}</TD></TR>" for lora, multiplier in zip(video_activated_loras, video_loras_multipliers) ]
            video_activated_loras_str = "<TABLE style='border:0px;padding:0px;width:100%;table-layout:fixed'>" + "".join(video_activated_loras) + "</TABLE>" if len(video_activated_loras) > 0 else ""
            video_duration_seconds = configs.get("duration_seconds", 0)
            duration_def = model_def.get("duration_slider", None)
            if duration_def is not None and video_duration_seconds > 0:
                misc_values += [ f"{video_duration_seconds}s"]
                misc_labels += [html.escape(duration_def.get("name", duration_def.get("label", "Max Duration")))]
            prompt_class = model_def.get("prompt_class","Text Prompt")
            prompt_labels.update((prompt_class, f"Original {prompt_class}"))
            values +=  misc_values + [video_prompt]
            labels += misc_labels + [f"Original {prompt_class}" if enhanced_video_prompt else prompt_class]
            video_comments = html.escape(str(configs.get("comments", "") or "")[:4096]).replace("\n", "<BR>")
            if len(video_comments) > 0:
                values += [video_comments]
                labels += ["Comments"]
            alt_prompt_def = model_def.get("alt_prompt", None)
            if alt_prompt_def is not None:
                alt_prompt_label = alt_prompt_def.get("name", alt_prompt_def.get("label")) 
                prompt_labels.update((alt_prompt_label, f"Original {alt_prompt_label}"))
                alt_prompt_text = str(configs.get("alt_prompt", "") or "")
                enhanced_alt_prompt_text = str(configs.get("enhanced_alt_prompt", "") or "")
                alt_prompt_mode = multi_prompts_gen_type if model_def.get("alt_prompt_inherits_prompt_paragraphs", False) else "FG"
                alt_prompt_history = prompt_parser.parse_prompt_history(alt_prompt_text, enhanced_alt_prompt_text, alt_prompt_mode)
                if alt_prompt_history is not None:
                    original_alt_prompts, enhanced_alt_prompts = alt_prompt_history
                    alt_prompt_text = prompt_parser.serialize_prompt_units("", original_alt_prompts, alt_prompt_mode)
                    enhanced_alt_prompt_text = prompt_parser.serialize_prompt_units("", enhanced_alt_prompts, alt_prompt_mode)
                alt_prompt = html.escape(alt_prompt_text[:16384]).replace("\n", "<BR>")
                if len(alt_prompt):
                    values += [alt_prompt]
                    labels += [f"Original {alt_prompt_label}" if enhanced_alt_prompt_text else alt_prompt_label]
                enhanced_alt_prompt = html.escape(enhanced_alt_prompt_text[:16384]).replace("\n", "<BR>")
                if len(enhanced_alt_prompt):
                    values += [enhanced_alt_prompt]
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
                labels += [prompt_class]
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
                    if not custom_setting_visible(setting_def, video_video_prompt_type, video_audio_prompt_type, configs.get("model_mode")):
                        continue
                    setting_id = setting_def.get("id", get_custom_setting_id(setting_def, idx))
                    setting_value = video_custom_settings.get(setting_id, None)
                    if setting_value is None:
                        continue
                    if isinstance(setting_value, str) and len(setting_value.strip()) == 0:
                        continue
                    values += [get_custom_setting_display_value(setting_def, setting_value)]
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
                video_skip_steps_cache = {"tea": "TeaCache", "mag": "MagCache", "spectrum": "Spectrum", "first_block": "First Block Cache"}.get(video_skip_steps_cache_type, video_skip_steps_cache_type)
                if video_skip_steps_cache_type in ("tea", "mag"):
                    video_skip_steps_cache += f" x{video_skip_steps_multiplier}"
                elif video_skip_steps_cache_type == "first_block":
                    video_skip_steps_cache += f" (threshold {float(video_skip_steps_multiplier):g})"
                if video_skip_steps_cache_start_step_perc >0:  video_skip_steps_cache += f", Start from {video_skip_steps_cache_start_step_perc}%"
                values += [ video_skip_steps_cache ]
                labels += [ "Skip Steps" ]

            values += pp_values
            labels += pp_labels

            if len(video_activated_loras_str) > 0:
                values += [video_activated_loras_str]
                labels += ["LoRAs"] 
            if nb_audio_tracks  > 0:
                values +=[audio_tracks_summary]
                labels +=["Nb Audio Tracks"]
            values += [ video_creation_date, video_generation_time ]
            labels += [ "Creation Date", "Generation Time" ]
        if configs_summary:
            values += [configs_summary]
            labels += ["Configs"]
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
        rows = [f"<TR><TD style='text-align: right;' WIDTH=1% NOWRAP VALIGN=TOP>{label}</TD><TD><B class='{'selected-media-prompt' if label in prompt_labels else ''}'>{value}</B></TD></TR>" for label, value in zip(labels, values)]
        html_content = f"{table_style}<TABLE ID=video_info WIDTH=100%>" + "".join(rows) + "</TABLE>"
    else:
        html_content =  get_default_video_info()
    kind = "image" if is_image else "video" if is_video else "audio" if is_audio else "3d" if is_mesh else "world" if is_world else "app" if is_app else None
    return html_content, kind, is_deleted


# Keep event_data required: on Python 3.10, `event_data: gr.EventData = None` becomes Optional[EventData],
# and Gradio 5.29 stops injecting the gallery selection index, breaking the selected media choice.
# Media info rows refreshed by select_media: one row of actions per media kind, then one "Eject Deleted File" row per gallery.
MEDIA_ACTION_KINDS = ("video", "image", "audio", "3d", "world", "app")

def select_media(state, current_gallery_tab, input_file_list, file_selected, strip_selected, source, current_spatial_upsampling, current_spatial_parameters, current_temporal_parameters, spatial_help_target_id, current_temporal_upsampling, video_info_tab, event_data: gr.EventData):
    """`strip_selected`: index clicked in the strip gallery of `source` (-1: keep the stored selection)."""
    gen = get_gen_info(state)
    shared_gallery = service_for(state) is not None
    # the temporal and spatial helps and contexts, the temporal section, each slot's inputs, their row and values, the shared images and their row, the film grain row
    late_parameter_count = 2 * (sum(parameter_ui.INPUT_COUNTS.values()) + 2) + 8
    if current_gallery_tab != source_index(source):
        return [gr.update()] * (2 + len(MEDIA_ACTION_KINDS) + len(GALLERY_SOURCES) + 9 + late_parameter_count + 2)
    if source == VIDEO.id:
        files, settings_list = get_file_list(state, input_file_list)
        data = event_data._data if event_data is not None else None
        if not shared_gallery and data!=None and isinstance(data, dict):
            choice = data.get("index",0)
        else:
            choice = gen.get("selected", file_selected)
    else:
        files, settings_list = get_file_list(state, None, source)
        choice = strip_selected if not shared_gallery and strip_selected >= 0 else gen.get(gallery_source(source).selected_key, -1)
    choice = min(len(files)-1, choice)
    if choice < 0 and len(files) > 0: choice = 0
    if not shared_gallery:
        set_file_choice(gen, files, choice, source)

    html_content, kind, is_deleted = format_media_info(files[choice] if files else None, settings_list[choice] if files else None)
    is_image, is_video, is_audio = kind == "image", kind == "video", kind == "audio"
    is_mesh = kind == "3d" and mesh_postprocessing.is_mesh_file(files[choice])  # its 3D Refiner and Texturer
    media_kind = mesh_postprocessing.MEDIA_3D if is_mesh else None
    visible= len(files) > 0
    visual_media = is_image or is_video
    if is_video or is_mesh:
        temporal_state = temporal_upsampler_api.late_postprocessing_ui_state(current_temporal_upsampling, media_kind=media_kind or temporal_upsampler_api.MEDIA_VIDEO)
        post_temporal_update = gr.update(visible=False, value=temporal_state["value"])
        post_temporal_method_update = gr.update(choices=temporal_state["method_choices"], value=temporal_state["method"], label=temporal_upsampler_api.slot_label(media_kind or temporal_upsampler_api.MEDIA_VIDEO), visible=True)
        post_temporal_multiplier_update = gr.update(choices=temporal_state["multiplier_choices"], value=temporal_state["scale"], visible=bool(temporal_state["multiplier_choices"]))
    else:
        temporal_state = {"method": "", "context": gr.update()}
        post_temporal_update = gr.update(visible=False, **({"value": ""} if is_image else {}))
        post_temporal_method_update = gr.update(visible=False, value="")
        post_temporal_multiplier_update = gr.update(visible=False)
    if visual_media or is_mesh:
        spatial_state = upsampler_api.late_postprocessing_ui_state(current_spatial_upsampling, image_outputs=is_image, media_kind=media_kind)
        post_spatial_update = gr.update(visible=False, value=spatial_state["value"])
        post_spatial_method_update = gr.update(choices=spatial_state["method_choices"], value=spatial_state["method"], label=upsampler_api.slot_label(upsampler_api.media_kind_of(is_image, media_kind)), visible=True)
        post_spatial_ratio_update = gr.update(choices=spatial_state["ratio_choices"], value=spatial_state["scale"] if spatial_state["ratio_choices"] else None, visible=bool(spatial_state["method"] and spatial_state["ratio_choices"]))
    else:
        spatial_state = {"method": "", "method_choices": [("None", "")], "context": gr.update()}
        post_spatial_update = gr.update(visible=False, value="")
        post_spatial_method_update = gr.update(visible=False, value="")
        post_spatial_ratio_update = gr.update(visible=False)
    help_media_profile = upsampler_api.media_kind_of(is_image, media_kind)
    help_title, help_markdown = upsampler_api.spatial_help_popup(spatial_state["method_choices"], media_profile=help_media_profile, field_help=field_help)
    post_spatial_help_update = gr.update(value=field_help.render_marker(spatial_help_target_id, upsampler_api.spatial_help_id(spatial_state["method_choices"], media_profile=help_media_profile), title=help_title, markdown=help_markdown))
    late_context = upsampler_api.PARAMETER_UI_LATE_POSTPROCESSING
    spatial_parameters, temporal_parameters = parameter_ui.slot_parameters(parameter_ui.SPATIAL, spatial_state["method"], late_context), parameter_ui.slot_parameters(parameter_ui.TEMPORAL, temporal_state["method"], late_context)
    spatial_parameter_state, temporal_parameter_state = parameter_ui.ui_state(spatial_parameters, current_spatial_parameters), parameter_ui.ui_state(temporal_parameters, current_temporal_parameters)
    post_temporal_help_update = gr.update()
    if "method_choices" in temporal_state:  # Temporal Upsampler for videos, Texturer for 3D assets
        help_title, help_markdown = temporal_upsampler_api.temporal_help_popup(temporal_state["method_choices"], field_help=field_help, media_kind=media_kind or temporal_upsampler_api.MEDIA_VIDEO)
        post_temporal_help_update = gr.update(value=field_help.render_marker(spatial_help_target_id.replace("spatial-upsampling", "temporal-upsampling"), temporal_upsampler_api.temporal_help_id(temporal_state["method_choices"]), title=help_title, markdown=help_markdown))
    # the slots' contexts: their method choices follow the selected media
    late_parameter_updates = [post_temporal_help_update, temporal_state["context"], post_spatial_help_update, spatial_state["context"], gr.update(visible=is_video or is_mesh),
                              *parameter_ui.updates(gr, temporal_parameter_state, late_postprocessing=True), temporal_parameter_state["values"],
                              *parameter_ui.updates(gr, spatial_parameter_state, late_postprocessing=True), spatial_parameter_state["values"],
                              *parameter_ui.images_updates(gr, parameter_ui.images_parameter(spatial_parameters, temporal_parameters)), gr.update(visible=visual_media)]
    action_rows = [gr.update(visible=visible and kind == action_kind) for action_kind in MEDIA_ACTION_KINDS]
    deleted_rows = [gr.update(visible=visible and is_deleted and source == gallery.id) for gallery in GALLERY_SOURCES]
    late_tabs = {"audio_postprocessing": visible and is_audio, "post_processing": visible and (is_video or is_image or is_mesh), "audio_remuxing": visible and is_video}
    # a late tab the selected media does not offer (a splat after a mesh, an image after a video) gives way to the media information
    tab_updates = [gr.Tabs(selected="video_info"), "video_info"] if not late_tabs.get(video_info_tab, True) else [gr.update(), video_info_tab]
    return choice if source == VIDEO.id else gr.update(), html_content, *action_rows, *deleted_rows, gr.update(visible=visible and is_audio), gr.update(visible=visible and (is_video or is_image or is_mesh)) , gr.update(visible=visible and is_video), post_temporal_update,post_temporal_method_update, post_temporal_multiplier_update, post_spatial_update, post_spatial_method_update, post_spatial_ratio_update, *late_parameter_updates, *tab_updates

def convert_image(image, keep_alpha=False):

    from PIL import ImageOps
    from typing import cast
    if isinstance(image, str):
        image = _open_image_input(image)
    image = image.convert('RGBA' if keep_alpha and (image.mode in ('RGBA', 'LA', 'PA') or 'transparency' in image.info) else 'RGB')
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
    from preprocessing.processors import get_preprocessor as create_preprocessor

    return create_preprocessor(process_type, inpaint_color, pre_video_guide, server_config)


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


def preprocess_video_with_mask(pre_video_guide, input_video_path, input_mask_path, height, width,  max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size= 16, expand_scale = 2, process_type = "inpaint", process_type2 = None, to_bbox = False, negate_mask = False, process_outside_mask = None, inpaint_color = 127, outpainting_dims = None, outpainting_ratio = "", proc_no = 1, outpainting_quantize_margins = 0, uint8_output = False):
    # Returns the guide (C, T, H, W) and its mask (1, T, H, W) or None: float in [-1, 1] / [0, 1], or uint8 0-255 / 0, 255 with uint8_output
    # (model_def "uint8_guides").

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
        height, width, margin_top, margin_left = get_outpainting_frame_location(final_height, final_width, outpainting_dims, 1, outpainting_ratio, source_frame_height, source_frame_width, quantize_margins=outpainting_quantize_margins)

    if any_mask:
        num_frames = min(len(video), len(mask_video))
    else:
        num_frames = len(video)

    if any_identity_mask:
        any_mask = True

    # the identity mask keeps every pixel: one array shared by all the frames (no pixel selected, so expansion / bbox leave it empty)
    identity_mask = np.zeros((height, width), dtype=np.uint8) if any_identity_mask else None

    # for frame_idx in range(num_frames):
    def prep_prephase(frame_idx):
        frame = Image.fromarray(video[frame_idx].cpu().numpy()) #.asnumpy()
        if fit_crop:
            frame = rescale_and_crop(frame, width, height)
        else:
            frame = frame.resize((width, height), resample=Image.Resampling.LANCZOS)
        frame = np.array(frame)
        if any_identity_mask:
            return (frame, frame, identity_mask)
        if any_mask:
            mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) #.asnumpy()
            if fit_crop:
                mask = rescale_and_crop(mask, width, height)
            else:
                mask = mask.resize((width, height), resample=Image.Resampling.LANCZOS)
            mask = np.array(mask)

            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
            original_mask = mask.copy() if pose_special else None
            mask = expand_or_shrink_mask(mask, expand_scale)

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
    # Each frame is written to the output as soon as it is ready. Processes that work frame by frame run by chunks of frames, so that
    # the resized frames of the whole video are never all in RAM; the others (pose, depth...) see the whole video at once.
    frame_independent = all(kind in (None, "raw", "identity", "inpaint") for kind in (process_type, process_outside_mask))
    chunk_frames = max(16, int(max_workers) * 4) if frame_independent else num_frames
    masked_frames = masks = None
    for chunk_start in range(0, num_frames, chunk_frames):
        frame_groups = process_images_multithread(prep_prephase, list(range(chunk_start, min(num_frames, chunk_start + chunk_frames))), "prephase", wrap_in_list= False, max_workers=max_workers, in_place= True)
        proc_list, proc_list_outside, proc_mask = (list(column) for column in zip(*frame_groups))
        frame_groups = None
        if chunk_start + chunk_frames >= num_frames:
            video = mask_video = None

        proc_list = process_images_multithread(preproc, proc_list, process_type, max_workers=max_workers)
        if any_mask:
            proc_list_outside = process_images_multithread(preproc_outside, proc_list_outside, process_outside_mask, max_workers=max_workers)
        else:
            proc_list_outside = proc_mask = len(proc_list) * [None]

        for frame_no, (processed_img, processed_img_outside, mask) in enumerate(zip(proc_list, proc_list_outside, proc_mask)):
            proc_list[frame_no] = proc_list_outside[frame_no] = proc_mask[frame_no] = None
            if isinstance(processed_img, (list, tuple)):
                processed_img = np.full((height, width, 3), processed_img, dtype=np.uint8)
            if isinstance(processed_img_outside, (list, tuple)):
                processed_img_outside = np.full((height, width, 3), processed_img_outside, dtype=np.uint8)
            if any_mask :
                if process_type == "pose_align":
                    masked_frame = processed_img
                    mask = np.full_like(mask, 0)
                else:
                    masked_frame = processed_img_outside if mask is identity_mask else np.where(mask[..., None], processed_img, processed_img_outside)
                if process_outside_mask != None:
                    mask = np.full_like(mask, 255)
                mask = torch.from_numpy(mask)
                if outpainting_dims != None:
                    full_frame= torch.full( (final_height, final_width), 255, dtype= torch.uint8, device=mask.device)
                    full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = mask
                    mask = full_frame
            else:
                masked_frame = processed_img

            if isinstance(masked_frame, (int, float, np.integer)) or (isinstance(masked_frame, (list, tuple)) and len(masked_frame) == 3):
                masked_frame= np.full( (height, width, 3), inpaint_color_np, dtype= np.uint8)

            masked_frame = torch.from_numpy(masked_frame)
            if masked_frame.shape[-1] == 1:
                masked_frame =  masked_frame.repeat(1,1,3).to(torch.uint8)

            if outpainting_dims != None:
                color = inpaint_color.to(masked_frame.device).view(1, 1, 3)
                if masked_frame.shape[-1] == 4:
                    color = torch.cat([color, color.new_full((1, 1, 1), 255)], dim=-1)
                full_frame = color.expand(final_height, final_width, masked_frame.shape[-1]).clone()
                full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = masked_frame
                masked_frame = full_frame

            if masked_frames is None: # frames stored one after the other (T, H, W, C) and returned as (C, T, H, W), the layout of the former stack
                masked_frames = torch.empty((pad_frames + num_frames, *masked_frame.shape), dtype=torch.uint8 if uint8_output else torch.float32, device="cpu")
                masks = torch.empty(masked_frames.shape[:3], dtype=masked_frames.dtype, device="cpu") if any_mask else None
            masked_frames[pad_frames + chunk_start + frame_no].copy_(masked_frame)
            if any_mask:
                masks[pad_frames + chunk_start + frame_no].copy_(mask)
        proc_list = proc_list_outside = proc_mask = processed_img = processed_img_outside = masked_frame = mask = full_frame = None

    preproc = None
    preproc_outside = None
    gc.collect()
    torch.cuda.empty_cache()
    if pad_frames > 0: # the frames before the start of the video repeat its first frame
        masked_frames[:pad_frames] = masked_frames[pad_frames:pad_frames + 1]
        if any_mask: masks[:pad_frames] = masks[pad_frames:pad_frames + 1]
    if not uint8_output:
        masked_frames.div_(127.5).sub_(1.)
        if any_mask: masks.div_(255)

    return masked_frames.permute(3, 0, 1, 2), masks.unsqueeze(0) if any_mask else None

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

    if hdr_input:
        def resize_frame(frame):
            return resize_hdr_frame(frame)

        frames_list = process_images_multithread(
            resize_frame,
            frames_list,
            "upsample",
            wrap_in_list=False,
            max_workers=get_default_workers(),
            in_place=True,
        )
    else:
        frames_list = resize_lanczos_frames(frames_list, new_height, new_width, crop=fit_crop, max_workers=get_default_workers())

    # from preprocessing.dwpose.pose import save_one_video
    # save_one_video("test.mp4", frames_list, fps=8, quality=8, macro_block_size=None)

    return torch.stack(frames_list) 


def get_reference_video_dimensions(video_in, height, width, max_size, block_size):
    _, source_width, source_height, _ = get_video_info(video_in)
    max_short_edge, max_long_edge = max_size
    scale = min((height * width / (source_height * source_width)) ** 0.5,
                max_short_edge / min(source_height, source_width), max_long_edge / max(source_height, source_width))
    height_cap, width_cap = (max_short_edge, max_long_edge) if source_height <= source_width else (max_long_edge, max_short_edge)
    target_height = min(height_cap // block_size * block_size, max(block_size, round(source_height * scale / block_size) * block_size))
    target_width = min(width_cap // block_size * block_size, max(block_size, round(source_width * scale / block_size) * block_size))
    return target_height, target_width

 
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


def get_loaded_model_context():
    if wan_model is None or offloadobj is None or transformer_type is None or reload_needed:
        return None
    from postprocessing.model_context import LoadedModelContext
    return LoadedModelContext(model=wan_model, offloadobj=offloadobj, model_type=transformer_type, base_model_type=get_base_model_type(transformer_type), model_family=get_model_family(transformer_type), model_def=get_model_def(transformer_type), profile=loaded_profile, config_id=loaded_config)


def perform_temporal_upsampling(sample, previous_last_frame, temporal_upsampling, fps, abort_callback=None, progress_callback=None):
    wait_for_model_unload()
    return temporal_upsampler_api.temporal_upsample(temporal_upsampling, sample, previous_last_frame, fps, main_offloadobj=offloadobj, loaded_model_context=get_loaded_model_context(), processing_device=processing_device, to_uint8_callback=convert_video_tensor_to_uint8_chunked, process_files=process_files_def, init_pipe=partial(init_pipe, extension=temporal_upsampler_api), profile=0, abort_callback=abort_callback, progress_callback=progress_callback)


def perform_spatial_upsampling(sample, spatial_upsampling, seed=0, flashvsr_continue_cache=None, return_flashvsr_continue_cache=False, vae_tile_size=None, still_image=False, abort_callback=None, progress_callback=None, fps=24.0, frame_offset=0, prompt="", negative_prompt="", audio_waveform=None, audio_sample_rate=0, source_audio_path=None, reference_images=None, image_refs_relative_size=100.0, parameter_values=None):
    wait_for_model_unload()
    edit_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    if edit_upsampler is not None:
        profile_type = upsampler_api.profile_type_for_handler(edit_upsampler)
        if profile_type == upsampler_api.UPSAMPLER_PROFILE_IMAGE:
            profile = get_default_profile("image")
        elif profile_type == upsampler_api.UPSAMPLER_PROFILE_AUDIO:
            profile = get_default_profile("audio")
        else:
            profile = loaded_profile if loaded_profile >= 0 else get_default_profile("video")
        sample, upsampler_cache = upsampler_api.upscale_postprocessing(edit_upsampler, sample, spatial_upsampling, main_offloadobj=offloadobj, loaded_model_context=get_loaded_model_context(), seed=seed, continue_cache=flashvsr_continue_cache, return_continue_cache=return_flashvsr_continue_cache, vae_tile_size=vae_tile_size, process_files=process_files_def, vae_config=vae_config, init_pipe=partial(init_pipe, output_type=profile_type, extension=upsampler_api), profile=profile, still_image=still_image, fps=fps, frame_offset=frame_offset, prompt=prompt, negative_prompt=negative_prompt, audio_waveform=audio_waveform, audio_sample_rate=audio_sample_rate, source_audio_path=source_audio_path, reference_images=reference_images, image_refs_relative_size=image_refs_relative_size, abort_callback=abort_callback, progress_callback=progress_callback, parameter_values=parameter_values)
        return (sample, upsampler_cache) if return_flashvsr_continue_cache else sample
    raise ValueError(f"No spatial upsampler registered for '{spatial_upsampling}'")


def perform_mesh_postprocessing(data, spatial_upsampling, temporal_upsampling, *, spatial_values=None, temporal_values=None, reference_image=None, seed=0, abort_callback=None, progress_callback=None):
    """A GLB (bytes) through the 3D refiner then the texturer: the processed GLB and its [C, H, W] preview."""
    wait_for_model_unload()
    return mesh_postprocessing.process_glb(data, spatial_upsampling, temporal_upsampling, spatial_values=spatial_values, temporal_values=temporal_values,
                                           spatial_kwargs={"init_pipe": partial(init_pipe, output_type=upsampler_api.UPSAMPLER_PROFILE_IMAGE, extension=upsampler_api)},
                                           temporal_kwargs={"init_pipe": partial(init_pipe, extension=temporal_upsampler_api)}, reference_image=reference_image, seed=seed,
                                           abort_callback=abort_callback, progress_callback=progress_callback, main_offloadobj=offloadobj, loaded_model_context=get_loaded_model_context(),
                                           process_files=process_files_def)


def perform_image_spatial_upsampling(sample, spatial_upsampling, seed=0, vae_tile_size=None, abort_callback=None, progress_callback=None, fps=24.0, prompt="", negative_prompt="", parameter_values=None):
    edit_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    if edit_upsampler is None or sample.shape[1] <= 1 or getattr(edit_upsampler, "batch_image_inputs", False):
        return perform_spatial_upsampling(sample, spatial_upsampling, seed=seed, vae_tile_size=vae_tile_size, still_image=True, fps=fps, prompt=prompt, negative_prompt=negative_prompt, parameter_values=parameter_values, abort_callback=abort_callback, progress_callback=progress_callback)
    frames = []
    for frame_no in range(sample.shape[1]):
        if abort_callback is not None and abort_callback():
            return None
        frames.append(perform_spatial_upsampling(sample[:, frame_no:frame_no + 1], spatial_upsampling, seed=seed, vae_tile_size=vae_tile_size, still_image=True, fps=fps, prompt=prompt, negative_prompt=negative_prompt, parameter_values=parameter_values, abort_callback=abort_callback, progress_callback=progress_callback))
    return torch.cat(frames, dim=1)


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

def edit_mesh(send_cmd, state, mesh_source, seed, temporal_upsampling, spatial_upsampling, parameter_settings, operation_start_time):
    """parameter_settings: the slots' parameter dictionaries and the shared reference images, by setting name."""
    """Late post-processing of a gallery mesh: its 3D Refiner then its Texturer, the reference image coming from their parameters."""
    gen = get_gen_info(state)
    if gen.get("abort", False): return
    mesh_save_path = workspace_storage.output_config(server_config, state=state)[workspace_storage.OUTPUT_KEYS[3]]
    configs, _, _ = get_settings_from_file(state, mesh_source, False, False, False)
    configs = dict(configs) if configs is not None else {"type": get_model_record("Post Processing")}
    status = "Post Processing"
    if not parameter_settings.get(parameter_ui.IMAGES_SETTING):  # no Object Image given: the one embedded in the GLB (Embed Source Images)
        parameter_settings = {**parameter_settings, parameter_ui.IMAGES_SETTING: mesh_postprocessing.embedded_object_images(mesh_source, extract=True)}

    def progress(phase, current_step=None, total_steps=None, unit=None):
        status_msg = merge_status_context(status, str(phase))
        if current_step is not None and total_steps is not None and int(total_steps) > 0:
            send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)] + ([unit] if unit else []))
        else:
            send_cmd("progress", [0, status_msg])

    progress(upsampler_api.method_progress_label(spatial_upsampling) if str(spatial_upsampling or "").strip() else "Texturing - Starting")
    wait_for_model_unload()
    glb, preview = mesh_postprocessing.process_file(mesh_source, spatial_upsampling, temporal_upsampling, spatial_values=parameter_ui.settings_runtime_values(parameter_settings, parameter_ui.SPATIAL),
                                                    temporal_values=parameter_ui.settings_runtime_values(parameter_settings, parameter_ui.TEMPORAL),
                                                    spatial_kwargs={"init_pipe": partial(init_pipe, output_type=upsampler_api.UPSAMPLER_PROFILE_IMAGE, extension=upsampler_api)},
                                                    temporal_kwargs={"init_pipe": partial(init_pipe, extension=temporal_upsampler_api)}, seed=seed, abort_callback=lambda: media_abort_requested(gen),
                                                    progress_callback=progress, main_offloadobj=offloadobj, loaded_model_context=get_loaded_model_context(), process_files=process_files_def)
    if gen.get("abort", False): return
    from shared.utils.mesh_outputs import save_preview
    os.makedirs(mesh_save_path, exist_ok=True)
    mesh_path = get_available_filename(mesh_save_path, mesh_source, "_post", force_extension=".glb")
    set_saving_file_status(state, gen, send_cmd, mesh_path)
    with open(mesh_path, "wb") as writer:
        writer.write(glb)
    save_preview(preview, mesh_path)
    configs.update(temporal_upsampling=temporal_upsampling, spatial_upsampling=spatial_upsampling, **parameter_settings,
                   generation_time=round(time.time() - operation_start_time))
    print(f"Postprocessed 3D asset saved to Path: {mesh_path}")
    record_file_metadata(mesh_path, configs, True, False, gen)
    send_cmd("output")
    clear_status(state)


def edit_media(
                send_cmd,
                state,
                mode,
                video_source,
                seed,   
                temporal_upsampling,
                spatial_upsampling,
                film_grain_intensity,
                film_grain_saturation,
                postprocess_audio,
                postprocess_audio_prompt,
                postprocess_audio_neg_prompt,
                repeat_generation,
                audio_source,
                replace_voice_method,
                replace_voice_sample,
                replace_voice_sample2,
                prompt="",
                image_refs=None,
                image_refs_relative_size=100.0,
                spatial_postprocessing_parameters=None,
                temporal_postprocessing_parameters=None,
                postprocessing_reference_images=None,
                client_id="",
                plugin_data=None,
                **kwargs
                ):

    operation_start_time = time.time()
    if mesh_postprocessing.is_mesh_file(video_source):
        parameter_settings = {parameter_ui.SETTINGS[parameter_ui.SPATIAL]: dict(spatial_postprocessing_parameters or {}), parameter_ui.SETTINGS[parameter_ui.TEMPORAL]: dict(temporal_postprocessing_parameters or {}),
                              parameter_ui.IMAGES_SETTING: list(postprocessing_reference_images or [])}
        return edit_mesh(send_cmd, state, video_source, seed, temporal_upsampling, spatial_upsampling, parameter_settings, operation_start_time)
    gen = get_gen_info(state)
    output_paths = workspace_storage.output_config(server_config, state=state)
    save_path, image_save_path, audio_save_path = (output_paths[key] for key in workspace_storage.OUTPUT_KEYS[:3])
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    api_options = plugin_data.get("api", {}) if isinstance(plugin_data, dict) and isinstance(plugin_data.get("api", {}), dict) else {}
    api_suppress_source_audio = bool(api_options.get("suppress_source_audio"))
    upsampler_frame_offset = int(api_options.get("upsampler_frame_offset", 0))
    flashvsr_continue_cache = api_options.get("flashvsr_continue_cache")
    return_flashvsr_continue_cache = bool(api_options.get("return_flashvsr_continue_cache"))

    if gen.get("abort", False): return 
    abort = False
		
		
    postprocess_audio = audio_processor_api.normalize_method(postprocess_audio or "")
    replace_voice_method = audio_processor_api.normalize_method(replace_voice_method or "")
    source_is_image = has_image_file_extension(video_source)
    spatial_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    spatial_method = spatial_upsampler.split_value(spatial_upsampling)[0] if spatial_upsampler is not None else ""
    source_audio_conditioning = spatial_upsampler is not None and spatial_upsampler.query_upsampler_def().get("source_audio_conditioning", False)
    source_image_conditioning = upsampler_api.method_uses_setting(spatial_method, "reference_images")
    if source_is_image:
        postprocess_audio = ""
        replace_voice_method = ""
    postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
    soundtrack_method = postprocess_audio if audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"] else ""
    voice_method = postprocess_audio if audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT in postprocess_audio_meta["types"] else replace_voice_method
    soundtrack_meta = audio_processor_api.method_metadata(soundtrack_method)
    configs, _ , _ = get_settings_from_file(state, video_source, False, False, False)
    configs = configs.copy() if configs is not None else { "type" : get_model_record("Post Processing") }
    configs.pop("window_no", None)  # the source's sliding-window position does not describe the processed copy
    reference_images = None
    if not source_is_image and source_image_conditioning:
        reference_images = clean_image_list(postprocessing_reference_images or image_refs or [])
        if not reference_images:
            extract_and_apply_source_images(video_source, configs)
            reference_images = clean_image_list(configs.get("image_refs") or [])
    if len(str(client_id or "").strip()) > 0:
        configs["client_id"] = client_id

    has_already_audio = False
    has_subtitles = not source_is_image and subtitle_stream_count(video_source) > 0
    audio_tracks = []
    audio_metadata = None
    download_requested_postprocessing_assets(
        send_cmd,
        postprocess_audio=postprocess_audio,
        temporal_upsampling=temporal_upsampling if mode == "edit_postprocessing" else "",
        spatial_upsampling=spatial_upsampling if mode == "edit_postprocessing" else "",
        replace_voice_method=voice_method,
        gen=gen,
    )

    temp_audio_tracks = []
    conditioning_audio_path = None
    if not source_is_image and not soundtrack_method and not api_suppress_source_audio:
        audio_tracks, audio_metadata = extract_audio_tracks(video_source, temp_format="wav" if voice_method else None, codec_key=server_config.get("audio_output_codec", "aac_128"))
        temp_audio_tracks = audio_tracks.copy()
        has_already_audio = len(temp_audio_tracks) > 0 
    if not source_is_image and source_audio_conditioning and not api_suppress_source_audio:
        if audio_source:
            conditioning_audio_path = audio_source
        elif voice_method and audio_tracks:
            conditioning_audio_path = audio_tracks[0]
        else:
            conditioning_audio_tracks, _ = extract_audio_tracks(video_source, temp_format="wav")
            temp_audio_tracks += conditioning_audio_tracks
            conditioning_audio_path = conditioning_audio_tracks[0] if conditioning_audio_tracks else None

    source_prompt = str((spatial_postprocessing_parameters or {}).get("prompt") or "").strip() or (configs.get("prompt") if upsampler_api.method_uses_setting(spatial_method, "prompt") else str(prompt or "").strip() or configs.get("prompt"))
    spatial_upsampling_prompt = upsampler_api.resolve_late_postprocessing_prompt(spatial_upsampling, source_prompt) if mode == "edit_postprocessing" else ""
    if mode == "edit_postprocessing" and spatial_upsampler is not None and hasattr(spatial_upsampler, "validate_runtime_inputs"):
        spatial_upsampler.validate_runtime_inputs(prompt=spatial_upsampling_prompt, reference_images=reference_images, source_audio_path=conditioning_audio_path)
    
    if soundtrack_method and soundtrack_meta["needs_audio_source"] and audio_source is not None:
        audio_tracks = [audio_source]

    file_list, file_settings_list = get_processed_queue(gen)[VIDEO.id]



    seed = set_seed(seed)

    if source_is_image:
        image = convert_image(_open_image_input(video_source))
        width, height = image.size
        fps, frames_count = 1, 1
    else:
        from shared.utils.utils import get_video_info_details
        media_info = get_video_info_details(video_source)
        fps = float(media_info["fps_float"] or media_info["fps"])
        width, height, frames_count = media_info["display_width"], media_info["display_height"], media_info["frame_count"]
    frames_count = min(frames_count, max_source_video_frames)
    sample = None

    if mode == "edit_postprocessing":
        if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 or film_grain_intensity > 0:
            spatial_status = upsampler_api.method_progress_label(spatial_upsampling)
            send_cmd("progress", [0, get_latest_status(state, spatial_status if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 else "Adding Film Grain")])
            if source_is_image:
                sample = torch.from_numpy(np.array(image).astype(np.uint8)).unsqueeze(0).permute(-1,0,1,2)
            else:
                sample = get_resampled_video(video_source, 0, max_source_video_frames, None)
                sample = sample.permute(-1,0,1,2)
            frames_count = sample.shape[1] 

        output_fps = fps
        def upsampler_progress(phase, current_step=None, total_steps=None, unit=None):
            phase_text = str(phase)
            gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
            status_msg = get_latest_status(state, phase_text)
            if current_step is not None and total_steps is not None and int(total_steps) > 0:
                send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)] + ([unit] if unit else []))
            else:
                send_cmd("progress", [0, status_msg])
        if len(temporal_upsampling) > 0:
            sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, None, temporal_upsampling, fps, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
            if gen.get("abort", False) or sample is None:
                return
            configs.update(temporal_upsampling=temporal_upsampling, temporal_postprocessing_parameters=dict(temporal_postprocessing_parameters or {}))
            frames_count = sample.shape[1] 


        if len(spatial_upsampling) > 0:
            if source_is_image:
                sample = perform_image_spatial_upsampling(sample, spatial_upsampling, seed=seed, fps=output_fps, prompt=spatial_upsampling_prompt, negative_prompt=str(configs.get("negative_prompt", "")), parameter_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images), abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                flashvsr_continue_cache = None
            else:
                sample = perform_spatial_upsampling(sample, spatial_upsampling, seed=seed, flashvsr_continue_cache=flashvsr_continue_cache, return_flashvsr_continue_cache=return_flashvsr_continue_cache, fps=output_fps, frame_offset=upsampler_frame_offset, prompt=spatial_upsampling_prompt, negative_prompt=str(configs.get("negative_prompt", "")), source_audio_path=conditioning_audio_path, reference_images=reference_images, image_refs_relative_size=image_refs_relative_size, parameter_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images), abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
            if return_flashvsr_continue_cache and not source_is_image:
                sample, flashvsr_continue_cache = sample
            if gen.get("abort", False) or sample is None:
                return
            configs["spatial_upsampling"] = spatial_upsampling
            configs.update(spatial_postprocessing_parameters=dict(spatial_postprocessing_parameters or {}), postprocessing_reference_images=list(postprocessing_reference_images or []))

        if film_grain_intensity > 0:
            from postprocessing.film_grain import add_film_grain
            sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
            configs["film_grain_intensity"] = film_grain_intensity
            configs["film_grain_saturation"] = film_grain_saturation
    else:
        output_fps = fps

    soundtrack_error = audio_processor_api.validate_method(soundtrack_method, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, video_source=None, frames_count=frames_count, fps=output_fps, audio_source=audio_source, media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension) if soundtrack_method else ""
    any_soundtrack_processor = bool(soundtrack_method and not soundtrack_error)
    if soundtrack_error and mode == "edit_remux":
        raise gr.Error(soundtrack_error)
    any_voice_replacement = bool(voice_method and replace_voice_sample is not None and len(audio_tracks) > 0)
    video_container = server_config.get("video_container", "mp4")
    video_extension = f".{video_container}"

    tmp_path = None
    saved_video_duration = None
    any_change = False
    api_video_tensor = None
    if sample != None:
        if source_is_image:
            image_extension = get_image_format(server_config.get("image_output_codec", None), rgba=sample.shape[0] == 4)["ext"]
            image_path = get_available_filename(image_save_path, video_source, "_post", force_extension=image_extension)
            image_paths = []
            set_saving_file_status(state, gen, send_cmd, image_path)
            for no, img in enumerate(sample.transpose(1,0)):
                img_path = get_available_filename(image_save_path, image_path, "" if no == 0 else f"_{no}")
                image_paths.append(save_image(img, save_file=img_path, quality=server_config.get("image_output_codec", None)))
            video_path = image_paths if len(image_paths) > 1 else image_paths[0]
            print(f"Postprocessed image saved to Path: {video_path}")
            configs["generation_time"] = round(time.time() - operation_start_time)
            record_file_metadata(video_path, configs, True, False, gen)
            if api_return_video_uint8 or api_return_audio or return_flashvsr_continue_cache:
                store_api_output_artifact(gen, client_id, video_path, "image", sample if api_return_video_uint8 else None, None, None, None)
            send_cmd("output")
            clear_status(state)
            return
        video_path = get_available_filename(save_path, video_source, "_tmp", force_extension=video_extension) if any_soundtrack_processor or has_already_audio or has_subtitles else get_available_filename(save_path, video_source, "_post", force_extension=video_extension)
        set_saving_file_status(state, gen, send_cmd, video_path)
        video_path = save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type= server_config.get("video_output_codec", None), container=server_config.get("video_container", "mp4"))
        saved_video_duration = sample.shape[1] / output_fps
        api_video_tensor = sample if api_return_video_uint8 else None
        if api_video_tensor is None:
            sample = None
            gc.collect()

        if any_soundtrack_processor or has_already_audio or has_subtitles: tmp_path = video_path
        any_change = True
    else:
        video_path = video_source

    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    any_change_initial = any_change
    while not gen.get("abort", False): 
        any_change = any_change_initial
        with gen_lock:
            extra_generation += gen.get("extra_orders",0)
            gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation         
        if repeat_no >= total_generation: break
        repeat_no +=1
        repeat_start_time = operation_start_time if repeat_no == 1 else time.time()
        gen["repeat_no"] = repeat_no
        suffix =  "" if "_post" in video_source else "_post"

        if soundtrack_method and soundtrack_meta["needs_audio_source"] and audio_source is not None:
            audio_prompt_type = configs.get("audio_prompt_type", "")
            if not "T" in audio_prompt_type:audio_prompt_type += "T"
            configs["audio_prompt_type"] = audio_prompt_type
            configs["postprocess_audio"] = postprocess_audio
            any_change = True
        elif any_voice_replacement:
            configs["postprocess_audio"] = postprocess_audio
            configs["replace_voice_method"] = voice_method
            any_change = True

        if any_soundtrack_processor:
            def audio_progress(phase, current_step=None, total_steps=None):
                phase_text = f"Audio - {phase}"
                gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
                status_msg = get_latest_status(state, phase_text)
                if current_step is not None and total_steps is not None and int(total_steps) > 0:
                    send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)])
                else:
                    send_cmd("progress", [0, status_msg])
            new_video_path = get_available_filename(save_path, video_source, suffix, force_extension=video_extension)
            generated_audio_path = get_available_filename(save_path, f"tmp_audio_seed{seed}_{repeat_no}.wav")
            generated_audio_path = audio_processor_api.generate_soundtrack(
                soundtrack_method,
                video_path=video_path,
                audio_source=audio_source,
                prompt=postprocess_audio_prompt,
                negative_prompt=postprocess_audio_neg_prompt,
                seed=seed,
                duration=frames_count / output_fps,
                output_path=generated_audio_path,
                send_cmd=send_cmd,
                status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
                verbose_level=verbose_level,
                audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                process_files=process_files_def,
                init_pipe=partial(init_pipe, output_type="audio", extension=audio_processor_api),
                profile=server_config.get("audio_profile", 4),
                abort_callback=lambda: media_abort_requested(gen),
                progress_callback=audio_progress,
            )
            if gen.get("abort", False) or generated_audio_path is None:
                return
            combine_video_with_audio_tracks(video_path, [generated_audio_path], new_video_path, audio_codec_key=server_config.get("audio_output_codec", "aac_128"), video_duration=saved_video_duration, subtitle_source=video_source if has_subtitles else None)
            if generated_audio_path is not None and generated_audio_path != audio_source and os.path.isfile(generated_audio_path):
                os.remove(generated_audio_path)
            configs["postprocess_audio"] = postprocess_audio
            configs["postprocess_audio_prompt"] = postprocess_audio_prompt
            configs["postprocess_audio_neg_prompt"] = postprocess_audio_neg_prompt
            configs["postprocess_audio_seed"] = seed
            any_change = True
        elif len(audio_tracks) > 0 or has_subtitles:
            new_video_path = get_available_filename(save_path, video_source, suffix, force_extension=video_extension)
            if any_voice_replacement:
                replaced_audio_tracks, replace_voice_temp_tracks = audio_processor_api.replace_voice_tracks(
                    voice_method,
                    audio_tracks,
                    voice_sample=replace_voice_sample,
                    output_dir=save_path,
                    prefix=f"tmp_seed{seed}_{repeat_no}",
                    process_files=process_files_def,
                    profile_no=server_config.get("audio_profile", 4),
                    verbose_level=verbose_level,
                    init_pipe=partial(init_pipe, output_type="audio", extension=audio_processor_api),
                    voice_sample2=replace_voice_sample2,
                    status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
                )
                replaced_sample_rate = resolve_mux_audio_sampling_rate(22050, audio_paths=replaced_audio_tracks)
                combine_and_concatenate_video_with_audio_tracks(
                    new_video_path,
                    video_path,
                    [],
                    replaced_audio_tracks,
                    0,
                    replaced_sample_rate,
                    audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                    verbose=verbose_level >= 2,
                )
                cleanup_temp_audio_files(replace_voice_temp_tracks)
            else:
                combine_video_with_audio_tracks(video_path, audio_tracks, new_video_path, audio_metadata=audio_metadata, audio_codec_key=server_config.get("audio_output_codec", "aac_128"), video_duration=saved_video_duration, subtitle_source=video_source if has_subtitles else None)
        else:
            new_video_path = video_path

        if any_change:
            configs["generation_time"] = round(time.time() - repeat_start_time)
            if mode == "edit_remux":
                print(f"Remuxed Video saved to Path: "+ new_video_path)
            else:
                print(f"Postprocessed video saved to Path: "+ new_video_path)
            with lock:
                file_list.append(new_video_path)
                file_settings_list.append(configs)

            if configs != None:
                from shared.utils.video_metadata import extract_source_images, save_video_metadata
                embedded_images = None
                temp_images_path = None
                if not bool(api_options.get("suppress_metadata_images")):
                    temp_images_path = get_available_filename(save_path, video_source, force_extension= ".temp")
                    embedded_images = extract_source_images(video_source, temp_images_path)
                save_video_metadata(new_video_path, configs, embedded_images, allow_inplace_update=True, verbose_level=verbose_level)
                if temp_images_path is not None and os.path.isdir(temp_images_path):
                    shutil.rmtree(temp_images_path, ignore_errors= True)
            notifications.record_generation(server_config, gen, new_video_path, configs)
            if api_return_video_uint8 or api_return_audio or return_flashvsr_continue_cache:
                store_api_output_artifact(
                    gen,
                    client_id,
                    new_video_path,
                    "video",
                    api_video_tensor,
                    None,
                    None,
                    output_fps,
                    flashvsr_continue_cache=flashvsr_continue_cache if return_flashvsr_continue_cache else None,
                )
            workspace_storage.recording_gallery(gen)["last_gallery_source"] = VIDEO.id
            send_cmd("output")
            seed = set_seed(-1)
    if tmp_path is not None and os.path.isfile(tmp_path):
        os.remove(tmp_path)
    cleanup_temp_audio_files(temp_audio_tracks)
    clear_status(state)


def edit_audio(send_cmd, state, audio_source, postprocess_audio, replace_voice_sample, replace_voice_sample2, client_id="", plugin_data=None):
    operation_start_time = time.time()
    audio_save_path = workspace_storage.output_config(server_config, state=state)['audio_save_path']
    gen = get_gen_info(state)
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    if gen.get("abort", False):
        return
    postprocess_audio = audio_processor_api.normalize_method(postprocess_audio or "")
    validation_error = ""
    if not media_source_exists(audio_source):
        validation_error = "Selected audio file is missing"
    elif not has_audio_file_extension(audio_source):
        validation_error = "Post processing is only available with Audio files"
    elif postprocess_audio:
        validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT, voice_sample=replace_voice_sample, voice_sample2=replace_voice_sample2)
    else:
        validation_error = "You must choose at least one Audio Post Processing Method"
    if validation_error:
        raise gr.Error(validation_error)

    configs, _, _ = get_settings_from_file(state, audio_source, False, False, False)
    configs = configs.copy() if configs is not None else {"type": get_model_record("Audio Post Processing")}
    if len(str(client_id or "").strip()) > 0:
        configs["client_id"] = client_id
    os.makedirs(audio_save_path, exist_ok=True)

    download_requested_postprocessing_assets(send_cmd, postprocess_audio=postprocess_audio, gen=gen)
    new_audio_path = audio_processor_api.process_audio_file(
        postprocess_audio,
        audio_source=audio_source,
        voice_sample=replace_voice_sample,
        voice_sample2=replace_voice_sample2,
        output_path=get_available_filename(audio_save_path, audio_source, "_post", ".wav"),
        process_files=process_files_def,
        profile_no=server_config.get("audio_profile", 4),
        verbose_level=verbose_level,
        init_pipe=partial(init_pipe, output_type="audio", extension=audio_processor_api),
        status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
    )
    configs["postprocess_audio"] = postprocess_audio
    configs["audio_postprocess"] = audio_processor_api.format_method_label(postprocess_audio)
    configs["generation_time"] = round(time.time() - operation_start_time)

    print("Postprocessed audio saved to Path: " + new_audio_path)
    record_file_metadata(new_audio_path, configs, False, True, gen)
    if api_return_video_uint8 or api_return_audio:
        artifact_audio = None
        artifact_sampling_rate = None
        if api_return_audio:
            import soundfile as sf
            artifact_audio, artifact_sampling_rate = sf.read(new_audio_path, dtype="float32", always_2d=False)
        store_api_output_artifact(gen, client_id, new_audio_path, "audio", None, artifact_audio, artifact_sampling_rate, None)
    send_cmd("output")
    clear_status(state)


def get_overridden_attention(model_type):
    model_def = get_model_def(model_type)
    override_attention = model_def.get("attention", None)
    if override_attention is None: return None
    gpu_version = gpu_major * 10 + gpu_minor
    attention_list = match_nvidia_architecture(override_attention, gpu_version) 
    if len(attention_list ) == 0: return None
    override_attention = attention_list[0]
    custom = model_def.get("custom_attention_modes", {}).get(override_attention)
    custom_status = ATTENTION_MODE_AVAILABILITY.get(override_attention, {})
    custom_supported = custom and custom.get("supported", custom_status.get("supported", custom.get("installed", custom_status.get("installed", True))))
    if override_attention is not None and override_attention not in attention_modes_supported and not custom_supported: return None
    return override_attention

def get_transformer_loras(model_type):
    model_def = get_model_def(model_type)
    transformer_loras_filenames = get_model_recursive_prop(model_type, "loras", return_list=True)
    lora_dir = get_lora_dir(model_type)
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

def process_prompt_enhancer(model_type, model_def, prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, audio_only, seed, prompt_enhancer_instructions = None, text_encoder_max_tokens = 512, enhancer_kwargs = None, batch_images = False):
    global enhancer_offloadobj
    prompt_enhancer_mode = str(prompt_enhancer or "")
    original_prompts, window_commands = prompt_enhancer_images.strip_window_commands(original_prompts)
    prompt_enhancer_instructions, text_encoder_max_tokens = resolve_prompt_enhancer_settings(
        model_type,
        model_def,
        prompt_enhancer_mode,
        is_image,
        prompt_enhancer_instructions=prompt_enhancer_instructions,
        text_encoder_max_tokens=text_encoder_max_tokens,
        enhancer_kwargs = enhancer_kwargs,
    )
    prompt_enhancer_instructions = prompt_enhancer_chaining.append_combined_input_contract(prompt_enhancer_instructions, prompt_enhancer_mode)
    if "W" in prompt_enhancer_mode:
        from shared.prompt_enhancer.transcription import TRANSCRIPT_INSTRUCTIONS, transcript_inputs

        original_prompts = transcript_inputs(original_prompts, enhancer_kwargs["source_audio_transcript"])
        prompt_enhancer_instructions = f"{prompt_enhancer_instructions}\n\n{TRANSCRIPT_INSTRUCTIONS}"

    from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt
    image_contexts = (enhancer_kwargs or {}).get("image_contexts") if prompt_enhancer_images.enabled(server_config, prompt_enhancer_mode) else None
    if image_contexts is not None:
        image_contexts = prompt_enhancer_images.contexts_for_prompts(image_contexts, len(original_prompts))
    duration_contexts = (enhancer_kwargs or {}).get("image_contexts")
    prompt_durations = [context.duration_seconds for context in prompt_enhancer_images.contexts_for_prompts(duration_contexts, len(original_prompts))] if duration_contexts is not None else None
    if image_start is None:
        image_start = (enhancer_kwargs or {}).get("control_image")
    prompt_images, image_contexts = prompt_enhancer_images.select_images(prompt_enhancer_mode, image_start, original_image_refs, batch_images, image_contexts)
    prompt_images = [_open_image_input(img) if isinstance(img, str) else img for img in prompt_images]
    template_context = prompt_enhancer_templates.build_context(enhancer_kwargs, is_image, audio_only)
    video_duration = (enhancer_kwargs or {}).get("video_duration_seconds")
    inject_duration = model_def.get("prompt_enhancer_video_duration") is not False
    duration_instructions = ""
    if prompt_durations is None and model_def.get("prompt_enhancer_video_duration", False) and not is_image and not audio_only and video_duration is not None:
        if prompt_enhancer_instructions is None:
            from shared.prompt_enhancer.prompt_enhance_utils import T2V_CINEMATIC_PROMPT, IT2V_CINEMATIC_PROMPT
            prompt_enhancer_instructions = IT2V_CINEMATIC_PROMPT if prompt_images else T2V_CINEMATIC_PROMPT
        duration_text = f"{video_duration:.2f}".rstrip("0").rstrip(".")
        duration_instructions = f"\n\nThe video duration is {duration_text} seconds. Keep the described action within this duration."
    has_text_input = any(letter in prompt_enhancer_mode for letter in "TLB")
    if len(original_prompts) == 0 and not has_text_input:
        return None
    else:
        remote_engine = resolve_role_engine(server_config, "prompt_enhancer")
        if is_remote_engine(remote_engine):
            from shared.remote_llm.prompt_enhancer import enhance_prompt as enhance_prompt_remote

            prompts = enhance_prompt_remote(
                remote_engine,
                server_config,
                original_prompts if has_text_input else ["an image"] * len(original_prompts),
                prompt_images,
                instructions=prompt_enhancer_instructions or "",
                max_output_tokens=text_encoder_max_tokens,
                prompt_durations=prompt_durations,
                template_context=template_context,
                inject_duration=inject_duration,
                duration_instructions=duration_instructions,
                enhancement_progress=(enhancer_kwargs or {}).get("generation_callbacks", {}).get("enhancement_progress"),
            )
            return prompt_enhancer_images.restore_window_commands(prompts, window_commands)
        import secrets
        enhancer_temperature = server_config.get("prompt_enhancer_temperature", 0.6)
        enhancer_top_p = server_config.get("prompt_enhancer_top_p", 0.9)
        randomize_seed = server_config.get("prompt_enhancer_randomize_seed", True)
        prompt_enhancer_images.warn_florence_images(server_config, prompt_enhancer_mode, image_start, original_image_refs, batch_images, len(original_prompts), (enhancer_kwargs or {}).get("caption_input_info"))
        if randomize_seed:
            enhancer_seed = secrets.randbits(32)
        else:
            enhancer_seed = seed if seed is not None and seed >= 0 else 0
        prompts = generate_cinematic_prompt(
            prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer,
            original_prompts if has_text_input else ["an image"] * len(original_prompts),
            prompt_images if len(prompt_images) > 0 else None,
            video_prompt = not is_image,
            text_prompt = audio_only,
            max_new_tokens=text_encoder_max_tokens,
            prompt_enhancer_instructions = prompt_enhancer_instructions,
            do_sample = True,
            temperature = enhancer_temperature,
            top_p = enhancer_top_p,
            seed = enhancer_seed,
            image_contexts=image_contexts,
            offload_manager=enhancer_offloadobj,
            prompt_durations=prompt_durations,
            template_context=template_context,
            inject_duration=inject_duration,
            duration_instructions=duration_instructions,
            thinking_enabled = "K" in prompt_enhancer_mode,
            generation_callbacks=(enhancer_kwargs or {}).get("generation_callbacks"),
        )
        return prompt_enhancer_images.restore_window_commands(prompts, window_commands)


def resolve_prompt_enhancer_settings(model_type, model_def, prompt_enhancer_mode, is_image, prompt_enhancer_instructions = None, text_encoder_max_tokens = 512, enhancer_kwargs = None):
    prompt_enhancer_mode = str(prompt_enhancer_mode or "")
    if model_def is None or len(model_type) == 0:
        return prompt_enhancer_instructions, int(text_encoder_max_tokens)
    prompt_profile_id = "0"
    prompt_profile_match = re.search(r"\d", prompt_enhancer_mode)
    if prompt_profile_match is not None:
        prompt_profile_id = prompt_profile_match.group(0)
    prompt_profile_suffix = "" if prompt_profile_id == "0" else prompt_profile_id

    model_handler = get_model_handler(model_type)
    if hasattr(model_handler, "get_custom_prompt_enhancer_instructions"):
        ret_prompt_enhancer_instructions, ret_text_encoder_max_tokens =  model_handler.get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs)
        if ret_prompt_enhancer_instructions is not None: prompt_enhancer_instructions = ret_prompt_enhancer_instructions 
        if ret_text_encoder_max_tokens is not None: text_encoder_max_tokens = ret_text_encoder_max_tokens

    visual_prompt_prefix = "image" if is_image else "video"
    prompt_instructions_key = f"{visual_prompt_prefix}_prompt_enhancer_instructions{prompt_profile_suffix}"
    prompt_max_tokens_key = f"{visual_prompt_prefix}_prompt_enhancer_max_tokens{prompt_profile_suffix}"
    prompt_enhancer_instructions = model_def.get(prompt_instructions_key, model_def.get(f"{visual_prompt_prefix}_prompt_enhancer_instructions", prompt_enhancer_instructions))
    text_encoder_max_tokens = model_def.get(prompt_max_tokens_key, model_def.get(f"{visual_prompt_prefix}_prompt_enhancer_max_tokens", text_encoder_max_tokens))

    if "I" not in prompt_enhancer_mode:
        prompt_instructions_key = f"text_prompt_enhancer_instructions{prompt_profile_suffix}"
        prompt_max_tokens_key = f"text_prompt_enhancer_max_tokens{prompt_profile_suffix}"
        prompt_enhancer_instructions = model_def.get(prompt_instructions_key, model_def.get("text_prompt_enhancer_instructions", prompt_enhancer_instructions))
        text_encoder_max_tokens = model_def.get(prompt_max_tokens_key, model_def.get("text_prompt_enhancer_max_tokens", text_encoder_max_tokens))
    return prompt_enhancer_instructions, int(text_encoder_max_tokens)

def exec_prompt_enhancer_engine(state, model_type, model_def, prompt_enhancer_modes, original_prompts, image_start, original_image_refs, is_image, audio_only, seed, progress, override_profile, send_cmd = None, tools = None, enhancer_kwargs = None, alt_prompts = None, alt_prompt_inherits_prompt_paragraphs = False):
    global enhancer_offloadobj
    assistant_mode = "A" in prompt_enhancer_modes
    if assistant_mode:
        return _deepy.run_assistant_prompt_turn(state, model_def, prompt_enhancer_modes, original_prompts, seed, override_profile=override_profile, send_cmd=send_cmd, tools=tools)

    if "W" in prompt_enhancer_modes:
        from shared.prompt_enhancer.transcription import transcribe_source_audio

        wait_for_model_unload()
        acquire_GPU_ressources(state, "prompt_enhancer_transcription", "Source Audio Transcription")
        try:
            transcription_gen = get_gen_info(state) if send_cmd is not None else progress.gen
            enhancer_kwargs["source_audio_transcript"] = transcribe_source_audio(enhancer_kwargs["audio_guide"], gen=transcription_gen, report_status=lambda text: progress((0, 1), desc=text, total=1), duration_seconds=enhancer_kwargs["duration_seconds"] if model_def.get("prompt_enhancer_transcription_use_duration", False) else None)
            if enhancer_kwargs["source_audio_transcript"] is None:
                prompt_enhancer_modes = prompt_enhancer_modes.replace("W", "")
        finally:
            release_GPU_ressources(state, "prompt_enhancer_transcription")

    local_runtime = not is_remote_engine(resolve_role_engine(server_config, "prompt_enhancer"))
    if local_runtime:
        wait_for_model_unload()
        acquire_GPU_ressources(state, "prompt_enhancer", "Prompt Enhancer", parks=True)
        try:
            download_gen = get_gen_info(state) if send_cmd is not None else progress.gen if isinstance(progress, WangpProgress) else None
            ensure_prompt_enhancer_loaded(override_profile=override_profile, progress=progress, send_cmd=send_cmd, gen=download_gen)
        except Exception:
            release_GPU_ressources(state, "prompt_enhancer")
            raise

    seed = set_seed(seed)
    num_prompts = len(original_prompts) 

    if alt_prompts is not None and prompt_enhancer_chaining.uses_routing(prompt_enhancer_modes):
        chain_steps = prompt_enhancer_chaining.parse_mode(prompt_enhancer_modes)
        step_no = 0

        def enhance_step(step, step_inputs):
            nonlocal step_no
            progress((step_no, len(chain_steps)), desc=f"Please Wait While Enhancing {step.output_field.replace('_', ' ').title()}", total=len(chain_steps))
            step_no += 1
            step_images = image_start if image_start is not None and "I" in step.mode else None
            return process_prompt_enhancer(model_type, model_def, step.mode, step_inputs, step_images, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=enhancer_kwargs, batch_images=True)

        try:
            enhanced_result = prompt_enhancer_chaining.run_chain(prompt_enhancer_modes, original_prompts, alt_prompts, alt_prompt_inherits_prompt_paragraphs, enhance_step)
        except Exception as e:
            if local_runtime:
                release_prompt_enhancer()
                release_GPU_ressources(state, "prompt_enhancer")
            if isinstance(e, (InterruptedError, DownloadCancelled, offload.LoadingCancelled)):
                raise
            print(traceback.format_exc())
            raise gr.Error(e)
        if local_runtime:
            release_prompt_enhancer()
            release_GPU_ressources(state, "prompt_enhancer")
        return enhanced_result

    enhanced_prompts = []
    batch_windows = (enhancer_kwargs or {}).get("image_contexts") is not None and prompt_enhancer_images.enabled(server_config, prompt_enhancer_modes)
    batches = [(original_prompts, image_start)] if batch_windows else [([one_prompt], [one_image] if one_image is not None else None) for one_prompt, one_image in zip(original_prompts, image_start)]
    for i, (batch_prompts, start_images) in enumerate(batches):
        status = 'Enhancing Prompt' + (f' {i+1}/{num_prompts}' if num_prompts > 1 else '')
        progress((i , num_prompts), desc=status, total= num_prompts)

        try:
            enhanced_prompt = process_prompt_enhancer(model_type, model_def, prompt_enhancer_modes, batch_prompts, start_images, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=prompt_enhancer_images.batch_kwargs(enhancer_kwargs, i, batch_windows, num_prompts))
        except Exception as e:
            if local_runtime:
                release_prompt_enhancer()
                release_GPU_ressources(state, "prompt_enhancer")
            if isinstance(e, (InterruptedError, DownloadCancelled, offload.LoadingCancelled)):
                raise
            print(traceback.format_exc())
            raise gr.Error(e)
        enhanced_prompts.extend([[value] for value in enhanced_prompt] if batch_windows else [enhanced_prompt])

    if local_runtime:
        release_prompt_enhancer()
        release_GPU_ressources(state, "prompt_enhancer")
    return enhanced_prompts

def keep_generated_prompt_newlines(multi_prompts_gen_type):
    multi_prompts_gen_type = str(multi_prompts_gen_type or "")
    return "P" in multi_prompts_gen_type or multi_prompts_gen_type == "FG"

def prompt_enhancer_outputs_multiple_prompts(prompt_enhancer_mode):
    return "M" in str(prompt_enhancer_mode or "")

def normalize_generated_prompt_lines(prompt, multi_prompts_gen_type, multi_prompt_output=False):
    texts, commands = prompt_enhancer_images.strip_window_commands([str(prompt or "")])
    output = _normalize_generated_prompt_lines(texts[0], multi_prompts_gen_type, multi_prompt_output)
    return prompt_enhancer_images.restore_window_commands([output], commands, multi_prompts_gen_type)[0]

def _normalize_generated_prompt_lines(prompt, multi_prompts_gen_type, multi_prompt_output=False):
    prompt = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
    if multi_prompt_output:
        prompt_lines = [line.strip() for line in prompt.split("\n") if line.strip()]
        if len(prompt_lines) == 0:
            return ""
        if str(multi_prompts_gen_type or "") == "FG":
            return "\n".join(prompt_lines)
        return ("\n\n" if "P" in str(multi_prompts_gen_type or "") else "\n").join(prompt_lines)
    if "P" in str(multi_prompts_gen_type or ""):
        return "\n".join(line for line in prompt.split("\n") if line.strip()).strip()
    if keep_generated_prompt_newlines(multi_prompts_gen_type):
        return prompt
    return re.sub(r"[\r\n]+", " ", prompt).strip()

def enhance_prompt(state, prompt, alt_prompt, prompt_enhancer, multi_images_gen_type, multi_prompts_gen_type, overrides, video_prompt_type, image_prompt_type, audio_prompt_type, progress=WangpProgress()):
    override_profile = generation_overrides.get_override(overrides, "memory_profile")
    model_type = get_state_model_type(state)
    inputs = get_model_settings(state, model_type)
    model_def = get_model_def(model_type)
    original_prompts = inputs["prompt"]

    if not model_def.get("skip_prompt_template", False):
        original_prompts, errors = prompt_parser.process_template(
            original_prompts,
            keep_comments=True,
            keep_empty_lines="P" in multi_prompts_gen_type or prompt_parser.PROMPT_UNIT_PREFIX in original_prompts,
        )
        if len(errors) > 0:
            gr.Info("Error processing prompt template: " + errors)
            return gr.update(), gr.update(), gr.update()
    original_prompts = prompt_parser.split_prompt_units(original_prompts, multi_prompts_gen_type, originals=True)
    routed = prompt_enhancer_chaining.uses_routing(prompt_enhancer)
    alt_prompt_inherits = bool(model_def.get("alt_prompt_inherits_prompt_paragraphs", False))
    alt_prompt_has_history = str(inputs["alt_prompt"] or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    alt_prompt_mode = multi_prompts_gen_type if alt_prompt_inherits and (alt_prompt_has_history or prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer)) else "FG"
    original_alt_prompts = prompt_parser.split_prompt_units(str(inputs["alt_prompt"] or ""), alt_prompt_mode, originals=True) or [""]
    multi_prompt_output = prompt_enhancer_outputs_multiple_prompts(prompt_enhancer)
    if multi_prompt_output:
        original_prompts = original_prompts[:1]
    num_prompts = len(original_prompts) 
    alt_prompt_count_error = prompt_enhancer_chaining.validate_alt_prompt_count(num_prompts, len(original_alt_prompts), model_def["prompt_class"], model_def["alt_prompt"]["label"]) if alt_prompt_inherits and prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer) else ""
    if alt_prompt_count_error:
        gr.Info(alt_prompt_count_error)
        return gr.update(), gr.update(), gr.update()
    image_prompt_type = inputs["image_prompt_type"]
    video_prompt_type = inputs["video_prompt_type"]
    multi_image_enhancer = prompt_enhancer_images.enabled(server_config, prompt_enhancer)
    image_contexts = None
    image_start = inputs["image_start"] if "S" in image_prompt_type else None
    continuation_image = None
    if inputs["image_mode"] == 0 and any_letters(image_prompt_type, "VL"):
        inputs = dict(inputs, video_source=prompt_enhancer_images.resolve_continue_video(inputs["video_source"], image_prompt_type, get_processed_queue(get_gen_info(state))[VIDEO.id][0], workspace_storage.output_config(server_config, state=state)['save_path']))
        if "I" in prompt_enhancer and not image_start and inputs["video_source"] is not None:
            progress(0, desc="Reading Continuation Start Image")
            fps = get_computed_fps(inputs["force_fps"], get_base_model_type(model_type), inputs["video_guide"], inputs["video_source"])
            continuation_image = prompt_enhancer_images.continuation_start_image(inputs["video_source"], inputs["keep_frames_video_source"], fps, max_source_video_frames)
    if multi_image_enhancer or has_slash_commands(original_prompts) or continuation_image is not None:
        fps = get_computed_fps(inputs["force_fps"], get_base_model_type(model_type), inputs["video_guide"], inputs["video_source"])
        source_frames = estimate_first_window_overlap_frames(image_start, inputs["video_source"] if any_letters(image_prompt_type, "VL") else None, inputs["keep_frames_video_source"], fps)
        original_prompts, image_contexts = prompt_enhancer_images.prepare_manual(original_prompts, inputs, model_def, fps=fps, source_frames=source_frames, open_image=convert_image, multi_prompt_output=multi_prompt_output, continuation_image=continuation_image)
        num_prompts = len(original_prompts)
        if multi_image_enhancer:
            image_start = None
    if image_start is None and not multi_image_enhancer:
        image_start = inputs["image_end"] if "E" in image_prompt_type else None
    if image_start is None or not "I" in prompt_enhancer:
        image_start = [None] * num_prompts
    else:
        image_start = [convert_image(img[0]) for img in image_start]
        if multi_prompt_output:
            image_start = image_start[:1] or [None] * num_prompts
        elif len(image_start) == 1:
            image_start = image_start * num_prompts
        else:
            if multi_images_gen_type !=1:
                gr.Info("On Demand Prompt Enhancer with multiple Start Images requires that option 'Match images and text prompts' is set")
                return gr.update(), gr.update(), gr.update()

            if len(image_start) != num_prompts:
                gr.Info("On Demand Prompt Enhancer supports only mutiple Start Images if their number matches the number of Text Prompts")
                return gr.update(), gr.update(), gr.update()

    if continuation_image is not None and not multi_image_enhancer:
        image_start = [next((image for image, label in zip(context.images, context.labels) if label in ("start image", "end image")), None) for context in image_contexts]
    original_image_refs = inputs["image_refs"] if "I" in video_prompt_type else None
    if original_image_refs is not None:
        original_image_refs = [ convert_image(tup[0]) for tup in original_image_refs ]        
    is_image = inputs["image_mode"] > 0
    seed = inputs["seed"]

    model_def = get_model_def(get_state_model_type(state))
    audio_only = model_def.get("audio_only", False)
    enhancer_kwargs = {"image_prompt_type": image_prompt_type, "video_prompt_type": video_prompt_type, "audio_prompt_type": audio_prompt_type, "audio_guide": inputs["audio_guide"] if "A" in audio_prompt_type else None, "duration_seconds": inputs["duration_seconds"]}
    enhancer_kwargs["resolution"] = inputs["resolution"]
    enhancer_kwargs["control_image"] = prompt_enhancer_images.control_image_input(inputs)
    enhancer_kwargs["caption_input_info"] = prompt_enhancer_images.caption_input_info((inputs["image_start"] if "S" in image_prompt_type else None) or continuation_image, inputs["image_end"] if "E" in image_prompt_type else None, original_image_refs, enhancer_kwargs["control_image"], "K" in video_prompt_type)
    if image_contexts is not None:
        enhancer_kwargs["image_contexts"] = image_contexts
    if not is_image and not audio_only:
        frames = min(inputs["video_length"], inputs["sliding_window_size"]) if model_def.get("sliding_window", False) else inputs["video_length"]
        fps = get_computed_fps(inputs["force_fps"], get_base_model_type(model_type), inputs["video_guide"], inputs["video_source"])
        enhancer_kwargs.update(video_duration_seconds=frames / fps, fps=fps, frame_count=frames)
    enhancer_kwargs["generation_callbacks"] = {"stop_requested": lambda: progress.gen["abort"], "enhancement_progress": EnhancementProgress(progress, progress.stream_tokens)}
    enhanced = exec_prompt_enhancer_engine(state, model_type, model_def, prompt_enhancer, original_prompts, image_start, original_image_refs, is_image, audio_only, seed, progress, override_profile, enhancer_kwargs=enhancer_kwargs, alt_prompts=original_alt_prompts if routed else None, alt_prompt_inherits_prompt_paragraphs=alt_prompt_inherits)

    progress.check_cancelled()

    if routed:
        if enhanced.prompt_changed:
            output_prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type, multi_prompt_output=multi_prompt_output) for value in enhanced.prompts]
            prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_prompts, original_prompts)
        if enhanced.alt_prompt_changed:
            output_alt_prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type) for value in enhanced.alt_prompts]
            alt_prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_alt_prompts, prompt_enhancer_chaining.originals_for_outputs(original_alt_prompts, len(output_alt_prompts)))
    else:
        output_prompts = [normalize_generated_prompt_lines(value[0], multi_prompts_gen_type, multi_prompt_output=multi_prompt_output) for value in enhanced if value is not None]
        prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_prompts, original_prompts)
    if num_prompts > 1:
        gr.Info(f'{num_prompts} Prompts have been Enhanced')
    else:
        gr.Info("Prompt enhancement completed")
    return prompt, alt_prompt, prompt

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

def resolve_model_preprocess_all(model_def, **kwargs):
    preprocess_all = model_def.get("preprocess_all", False)
    return preprocess_all(**kwargs) if callable(preprocess_all) else preprocess_all

def get_outpainting_quantize_margins(model_def):
    value = model_def.get("outpainting_quantize_margins", 0)
    return int(model_def.get("vae_block_size", 16) if value is True else value or 0)

def custom_preprocess_video_with_mask(model_handler, base_model_type, pre_video_guide, video_guide, video_mask, height, width, max_frames, start_frame, fit_canvas, fit_crop, target_fps, block_size, expand_scale, video_prompt_type, model_def=None, custom_settings=None):
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    max_workers = get_default_workers()

    if not video_guide or max_frames <= 0:
        return None, None, None, None
    any_mask = video_mask is not None
    model_def = model_def or {}
    raw_custom_preprocessor_inputs = model_def.get("custom_preprocessor_raw_inputs", False)
    video_guide = get_resampled_video(video_guide, start_frame, max_frames, target_fps)
    if not raw_custom_preprocessor_inputs: # in place: one float copy of the video, the values of the former / 127.5 - 1.
        video_guide = video_guide.permute(-1, 0, 1, 2).float().div_(127.5).sub_(1.)
    pose_mask = None
    if video_mask is not None:
        video_mask = get_resampled_video(video_mask, start_frame, max_frames, target_fps)
        if not raw_custom_preprocessor_inputs:
            video_mask = video_mask.permute(-1, 0, 1, 2).float()[:1] / 255.
            pose_mask = video_mask

    # Mask filtering: resize, binarize, expand mask and keep only masked areas of video guide
    if any_mask and not raw_custom_preprocessor_inputs:
        invert_mask = "N" in video_prompt_type
        tgt_h, tgt_w = video_guide.shape[2], video_guide.shape[3]
        def process_mask(idx):
            return torch.from_numpy(prepare_binary_mask_frame(pose_mask[0, idx] * 255, tgt_h, tgt_w, expand_scale=expand_scale, invert=invert_mask))
        pose_mask = torch.stack(process_images_multithread(process_mask, list(range(pose_mask.shape[1])), "prephase", wrap_in_list=False, max_workers=max_workers, in_place=False)).unsqueeze(0)
        video_guide = video_guide.mul_(pose_mask).sub_(1 - pose_mask) # in place, the values of video_guide * pose_mask + (-1) * (1-pose_mask)
        video_mask = pose_mask

    guide_frame_count = video_guide.shape[0] if raw_custom_preprocessor_inputs else video_guide.shape[1]
    mask_frame_count = video_mask.shape[0] if raw_custom_preprocessor_inputs and any_mask else video_mask.shape[1] if any_mask else 0
    if guide_frame_count == 0 or any_mask and mask_frame_count == 0:
        return None, None, None, None
    
    video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2  = model_handler.custom_preprocess(base_model_type = base_model_type, pre_video_guide = pre_video_guide, video_guide = video_guide, video_mask = video_mask, pose_mask = pose_mask, height = height, width = width, fit_canvas = fit_canvas , fit_crop = fit_crop, target_fps = target_fps,  block_size = block_size, max_workers = max_workers, expand_scale = expand_scale, video_prompt_type=video_prompt_type, model_def=model_def, custom_settings=custom_settings)

    # if pad_frames > 0:
    #     masked_frames = masked_frames[0] * pad_frames + masked_frames
    #     if any_mask: masked_frames = masks[0] * pad_frames + masks

    return video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 

def get_output_filepath(file_path, is_image, audio_only):
    config = workspace_storage.output_config(server_config)
    base_path = config['image_save_path' if is_image else 'audio_save_path' if audio_only else 'save_path']
    return get_available_filename(base_path, file_path)


def record_file_metadata(video_path, configs, is_image, audio_only, gen, embedded_images=None, replace_last_file=False, notify_generation=True, write_metadata=True, record_notification=True, output_role="final"):
    shared_record_file_metadata(video_path, configs, is_image, audio_only, gen, get_processed_queue=get_processed_queue, metadata_choice=server_config.get("metadata_type", "metadata"), embedded_images=embedded_images, replace_last_file=replace_last_file, lock=lock, verbose_level=verbose_level, write_metadata=write_metadata, output_role=output_role)
    if record_notification:
        notifications.record_generation(server_config, gen, video_path, configs, replace_last=replace_last_file, notify=notify_generation)


@workspace_storage.workspace_generation(headless=lambda state: workspace_storage.HeadlessWorkspaceHost(state, args.workspaces_dir or server_config.get('workspaces_dir') or os.path.join(wgp_root, 'workspaces'), lambda path: get_settings_from_file(state, path, False, False, False)[0], output_override=bool(args.output_dir)))
@generation_downloads(get_gen_info, lambda: offloadobj)
def generate_media(
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
    world_output,
    batch_size,
    seed,
    force_fps,
    num_inference_steps,
    num_inference_steps2,
    custom_sigmas,
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
    video_guide2,
    video_guide3,
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
    audio_guide3,
    custom_guide,
    audio_source,
    replace_voice_method,
    replace_voice_sample,
    replace_voice_sample2,
    audio_prompt_type,
    speakers_locations,
    sliding_window_size,
    sliding_window_overlap,
    sub_parallel_window_size,
    sub_parallel_window_overlap,
    sliding_window_color_correction_strength,
    sliding_window_overlap_noise,
    sliding_window_discard_last_frames,
    sliding_window_trim_first_frames,
    image_refs_relative_size,
    remove_background_images_ref,
    temporal_upsampling,
    spatial_upsampling,
    film_grain_intensity,
    film_grain_saturation,
    postprocess_audio,
    postprocess_audio_prompt,
    postprocess_audio_neg_prompt,
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
    overrides,
    attention_sparsity,
    temperature,
    custom_settings,
    top_p,
    top_k,
    self_refiner_setting,
    self_refiner_plan,
    self_refiner_f_uncertainty,
    self_refiner_certain_percentage,
    output_filename,
    config,
    state,
    model_type,
    mode,
    plugin_data=None,
    spatial_postprocessing_parameters=None,
    temporal_postprocessing_parameters=None,
    postprocessing_reference_images=None,
):
    filename_formatter = None
    if output_filename:
        from shared.utils.filename_formatter import FilenameFormatter
        filename_formatter = FilenameFormatter(output_filename)
    wait_for_model_unload()

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

    global wan_model, offloadobj, reload_needed, loaded_config
    gen = get_gen_info(state)
    output_paths = workspace_storage.output_config(server_config, state=state)
    save_path, image_save_path, audio_save_path, mesh_save_path = (output_paths[key] for key in workspace_storage.OUTPUT_KEYS)
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    api_options = plugin_data.get("api", {}) if isinstance(plugin_data, dict) and isinstance(plugin_data.get("api", {}), dict) else {}
    flashvsr_continue_cache = api_options.get("flashvsr_continue_cache")
    return_flashvsr_continue_cache = bool(api_options.get("return_flashvsr_continue_cache"))
    api_return_side_files = bool(api_options.get("return_side_files", api_return_video_uint8 or api_return_audio))
    gen["last_progress_args"] = None
    torch.set_grad_enabled(False) 
    if mode == "edit_audio":
        edit_audio(send_cmd, state, audio_source, postprocess_audio, replace_voice_sample, replace_voice_sample2, client_id=client_id, plugin_data=plugin_data)
        return True
    if mode.startswith("edit_"):
        edit_media(send_cmd, state, mode, video_source, seed, temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, postprocess_audio, postprocess_audio_prompt, postprocess_audio_neg_prompt, repeat_generation, audio_source, replace_voice_method, replace_voice_sample, replace_voice_sample2, prompt=prompt, image_refs=image_refs, image_refs_relative_size=image_refs_relative_size, spatial_postprocessing_parameters=spatial_postprocessing_parameters, temporal_postprocessing_parameters=temporal_postprocessing_parameters, postprocessing_reference_images=postprocessing_reference_images, client_id=client_id, plugin_data=plugin_data)
        return True
    enhancer_mode = server_config.get("enhancer_mode", 1)
    auto_prompt_enhancer_requested = server_config.get("enhancer_enabled", 0) > 0 and enhancer_mode == 0 and prompt_enhancer is not None and len(prompt_enhancer) > 0
    postprocess_audio = audio_processor_api.normalize_method(postprocess_audio or "")
    replace_voice_method = audio_processor_api.normalize_method(replace_voice_method or "")
    postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
    replace_voice_meta = audio_processor_api.method_metadata(replace_voice_method)
    if not postprocess_audio_meta["needs_audio_source"]:
        audio_source = None
    if not replace_voice_method:
        replace_voice_sample = None
        replace_voice_sample2 = None
    elif not replace_voice_meta["needs_voice_sample2"]:
        replace_voice_sample2 = None

    model_def = get_model_def(model_type)
    config_groups = get_model_config_groups(model_type, model_def)
    config = model_config_groups.normalize_config_selection(config_groups, config)
    override_values = generation_overrides.parse_overrides(overrides)
    override_profile, override_attention = override_values.get("memory_profile", -1), override_values.get("attention_mode", "")
    is_image = image_mode > 0
    audio_only = model_def.get("audio_only", False)
    mesh_outputs = model_def.get("mesh_outputs", False)
    world_only = model_def.get("world_outputs", False) and (world_output or "world") == "world"  # the world file alone, no video
    if mesh_outputs or world_only:  # the image of a 3D result is its preview: image post-processing does not apply
        film_grain_intensity = 0
        if world_only or model_def.get("mesh_postprocessing", True) is False:
            spatial_upsampling, temporal_upsampling = "", ""
    duration_def = model_def.get("duration_slider", None)

    set_video_prompt_type = model_def.get("set_video_prompt_type", None)
    if set_video_prompt_type is not None:
        video_prompt_type = add_to_sequence(video_prompt_type, set_video_prompt_type)
    reference_videos = "-" in video_prompt_type
    preprocess_video_guide2 = model_def.get("preprocess_video_guide2", False)
    if is_image:
        image_batch_size_max = max(1, int(model_def.get("image_batch_size_max", 16)))
        if batch_size > image_batch_size_max:
            raise ValueError(f"This model supports a maximum of {image_batch_size_max} image{'s' if image_batch_size_max > 1 else ''} per generation.")
        if not model_def.get("custom_video_length", False):
            if min_frames_if_references >= 1000:
                video_length = min_frames_if_references - 1000
            else:
                video_length = min_frames_if_references if "I" in video_prompt_type or "V" in video_prompt_type else 1 
    else:
        batch_size = 1
    temp_filenames_list = []

    if image_guide is not None:
        if isinstance(image_guide, str): 
            image_guide = _open_image_input(image_guide)
        if isinstance(image_guide, Image.Image):
            video_guide = image_guide
            image_guide = None

    if image_mask is not None:
        if isinstance(image_mask, str):
            image_mask = _open_image_input(image_mask)
        if isinstance(image_mask, Image.Image):
            video_mask = image_mask
            image_mask = None

    if model_def.get("no_background_removal", False): remove_background_images_ref = 0
    
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)
    block_size = model_def.get("vae_block_size", 16)
    width, height = resolution.split("x")
    width, height = int(width) // block_size * block_size, int(height) // block_size * block_size

    from preprocessing.processors import prepare_generation_assets
    prepare_generation_assets(model_def, base_model_type, locals(), server_config, process_map_video_guide, process_map_outside_mask)
    download_requested_postprocessing_assets(
        send_cmd,
        postprocess_audio=postprocess_audio if not (is_image or audio_only) else "",
        temporal_upsampling=temporal_upsampling if not (is_image or audio_only) else "",
        spatial_upsampling=spatial_upsampling if not audio_only else "",
        replace_voice_method=replace_voice_method if not (is_image or audio_only) else "",
        gen=gen,
    )

    if "P" in preload_model_policy and not "U" in preload_model_policy:
        while wan_model == None:
            time.sleep(1)
    model_kwargs = {}
    new_model_vae_upsampling = upsampler_api.model_load_vae_upsampling_value(spatial_upsampling, base_model_type, model_def, image_mode)
    old_model_vae_upsampling = None if reload_needed or wan_model is None else upsampler_api.loaded_model_vae_upsampling_value(wan_model)
    reload_needed = reload_needed or old_model_vae_upsampling != new_model_vae_upsampling
    if new_model_vae_upsampling:
        model_kwargs.update(upsampler_api.model_load_kwargs_for_vae_upsampling(spatial_upsampling, base_model_type, model_def, image_mode))
    output_type = get_profile_type_for_model(model_type, image_mode)
    profile = compute_profile(override_profile, output_type)
    compilation = compute_compile(override_values.get("compilation", ""), compile)
    if model_type != transformer_type or reload_needed or profile != loaded_profile or config != loaded_config or compilation != loaded_compile or compute_quantization(override_values) != loaded_quantization:
        release_model()
        send_cmd("status", "Loading...")
        def loading_progress(phase, completed, total, model_id):
            component = " ".join(word.upper() if word.lower() in ("vae", "clip", "llm", "t5") else word.capitalize() for word in model_id.replace("_", " ").split())
            subtask = f"Pinning Model{' ' + component if component else ''} to Reserved RAM" if phase == "Pinning" else phase + (f" {component} Model" if component else "")
            title = f"Loading - {subtask}"
            send_cmd("progress", [(completed, total), title, total, "blocks" if phase == "Pinning" else "phases"])

        try:
            wan_model, offloadobj = load_models(model_type, override_profile, output_type=output_type, config_id=config, gen=gen, loading_callback=offload.LoadingCallback(lambda: gen.get("abort", False), loading_progress), overrides=override_values, **model_kwargs)
        except offload.LoadingCancelled as error:
            traceback.clear_frames(error.__traceback__)
            gc.collect()
            return True
        send_cmd("status", "Model Loaded")
        send_cmd("refresh_models", get_unique_id())
        reload_needed=  False
    vae_upsampler_handler = upsampler_api.find_vae_upsampler(spatial_upsampling)
    vae_upsampler_session = None
    if vae_upsampler_handler is not None and hasattr(vae_upsampler_handler, "prepare_vae_upsampler"):
        upsampler_name = vae_upsampler_handler.query_upsampler_def()["name"]
        send_cmd("status", f"Preparing {upsampler_name} upsampler...")
        vae_upsampler_session = upsampler_api.prepare_vae_upsampler(vae_upsampler_handler, spatial_upsampling, send_cmd=send_cmd, process_files=process_files_def, init_pipe=partial(init_pipe, output_type=upsampler_api.profile_type_for_handler(vae_upsampler_handler), extension=upsampler_api), profile=compute_profile(override_profile, upsampler_api.profile_type_for_handler(vae_upsampler_handler)), attention_mode=attention_mode, parameter_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images))
        send_cmd("status", f"{upsampler_name} upsampler prepared")
    if args.test and auto_prompt_enhancer_requested:
        try:
            ensure_prompt_enhancer_loaded(override_profile=override_profile, send_cmd=send_cmd, gen=gen)
        finally:
            unload_prompt_enhancer_runtime()
            if enhancer_offloadobj is not None:
                enhancer_offloadobj.unload_all()
    if args.test:
        upsampler_api.release_vae_upsampler(vae_upsampler_handler, vae_upsampler_session)
        send_cmd("info", "Test mode: model loaded, skipping generation.")
        return True
    overridden_attention = override_attention if len(override_attention) else get_overridden_attention(model_type)
    # if overridden_attention is not None and overridden_attention !=  attention_mode: print(f"Attention mode has been overriden to {overridden_attention} for model type '{model_type}'")
    attn = overridden_attention if overridden_attention is not None else attention_mode
    if attn == "auto":
        attn = get_auto_attention()
    elif attn not in override_attention_modes_supported and not (
        (custom := model_def.get("custom_attention_modes", {}).get(attn)) and
        custom.get("supported", ATTENTION_MODE_AVAILABILITY.get(attn, {}).get("supported", custom.get("installed", ATTENTION_MODE_AVAILABILITY.get(attn, {}).get("installed", True))))
    ):
        send_cmd("info", f"You have selected attention mode '{attn}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        send_cmd("exit")
        return True
    elif attn in ATTENTION_MODE_AVAILABILITY and attn not in model_def.get("custom_attention_modes", {}):
        send_cmd("info", f"Attention mode '{attn}' is not implemented for model type '{model_type}'.")
        send_cmd("exit")
        return True
    
    default_image_size = (height, width)

    if perturbation_switch == 0:
        perturbation_layers = None

    offload.shared_state["_attention_fallback"] = get_auto_attention() if attention_mode == "auto" else attention_mode
    offload.shared_state["_attention"] =  attn
    offload.shared_state["_sage_v_scaling"] = model_def.get("sage_v_scaling", False)
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
                False,
                output_height=height,
                output_width=width,
            )
        else:
            VAE_tile_size = get_tile_size(
                vae_config,
                device_mem_capacity,
                False,
            )
    else:
        VAE_tile_size = None

    trans = get_transformer_model(wan_model)
    trans2 = get_transformer_model(wan_model, 2)
    audio_sampling_rate = 16000

    prompt_has_history = str(prompt or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type) or [""]  # an empty optional prompt (validated by validate_settings)
    display_prompts = prompts.copy()
    display_original_prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type, originals=True) if prompt_has_history else display_prompts.copy()
    alt_prompt_inherits = bool(model_def.get("alt_prompt_inherits_prompt_paragraphs", False))
    alt_prompt_has_history = str(alt_prompt or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    auto_dependent_alt_prompt = auto_prompt_enhancer_requested and prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer)
    alt_prompt_mode = multi_prompts_gen_type if alt_prompt_inherits and (alt_prompt_has_history or auto_dependent_alt_prompt) else "FG"
    alt_prompts = prompt_parser.split_prompt_units(str(alt_prompt or ""), alt_prompt_mode) or [""]
    display_alt_prompts = alt_prompts.copy()
    display_original_alt_prompts = prompt_parser.split_prompt_units(str(alt_prompt or ""), alt_prompt_mode, originals=True) if alt_prompt_has_history else display_alt_prompts.copy()
    frames_minimum, frames_steps, latent_size = get_model_min_frames_and_step(model_type)
    parsed_keep_frames_video_source= max_source_video_frames if len(keep_frames_video_source) ==0 else int(keep_frames_video_source) 
    transformer_loras_filenames, transformer_loras_multipliers  = get_transformer_loras(model_type)
    lora_dir = get_lora_dir(model_type)
    if guidance_phases < 1: guidance_phases = 1
    lora_multiplier_phases = int(model_def.get("lora_multiplier_phases", guidance_phases) or guidance_phases or 1)
    lora_multiplier_branches = model_def.get("lora_multiplier_branches", None)
    if transformer_loras_filenames != None:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(transformer_loras_multipliers, len(transformer_loras_filenames), num_inference_steps, nb_phases = lora_multiplier_phases, model_switch_phase= model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: raise Exception(f"Error parsing Transformer Loras: {errors}")
        loras_selected = transformer_loras_filenames[:] 

    if hasattr(wan_model, "get_loras_transformer"):
        extra_loras_transformers, extra_loras_multipliers = wan_model.get_loras_transformer(get_model_recursive_prop, **locals())
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(extra_loras_multipliers, len(extra_loras_transformers), num_inference_steps, nb_phases = lora_multiplier_phases, merge_slist= loras_slists, model_switch_phase= model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: raise Exception(f"Error parsing Extra Transformer Loras: {errors}")
        loras_selected += extra_loras_transformers 

    base_loras_slists = loras_slists
    if len(activated_loras) > 0:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases = lora_multiplier_phases, merge_slist= loras_slists, model_switch_phase= model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: raise Exception(f"Error parsing Loras: {errors}")
        loras_selected += activated_loras

    if hasattr(wan_model, "get_trans_lora"):
        trans_lora, trans2_lora = wan_model.get_trans_lora()
    else:
        trans_lora, trans2_lora = trans, trans2

    if len(loras_selected) > 0:
        loras_selected = update_loras_url_cache(lora_dir, loras_selected)
        errors = check_loras_exist(model_type, loras_selected, True, send_cmd, gen=gen)
        if len(errors) > 0 : raise gr.Error(errors)
        loras_selected = [ get_lora_local_path(lora_dir, lora) for lora in loras_selected]
        pinnedLora = not is_mps and loaded_profile !=5  # and transformer_loras_filenames == None False # # #
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

    frames_offset = model_def.get("frames_offset", 1)
    video_length = fit_video_length(video_length, frames_minimum, latent_size, frames_offset)
    if sliding_window_size !=0:
        sliding_window_size = floor_frame_count(sliding_window_size, frames_minimum, latent_size, frames_offset)
    sliding_window_defaults = model_def.get("sliding_window_defaults", {})
    if sliding_window_overlap !=0:
        if sliding_window_defaults.get("overlap_default", 0) != sliding_window_overlap:
            overlap_offset = sliding_window_defaults.get("overlap_offset", 1)
            sliding_window_overlap = sliding_window_overlap // latent_size * latent_size if overlap_offset == 0 else (sliding_window_overlap - overlap_offset) // latent_size * latent_size + overlap_offset
    sub_parallel_window_size = max(0, int(sub_parallel_window_size or 0))
    if sub_parallel_window_size !=0:
        sub_parallel_window_size += 1
    sub_parallel_window_overlap = max(0, int(sub_parallel_window_overlap or 0))
    if sliding_window_discard_last_frames !=0:
        discard_step = sliding_window_defaults.get("discard_step", latent_size)  # 1 for models that recycle no latents
        sliding_window_discard_last_frames = sliding_window_discard_last_frames // discard_step * discard_step

    current_video_length = video_length
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576
    outpainting_dims = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio)
    any_outpainting = outpainting_dims is not None
    outpainting_quantize_margins = get_outpainting_quantize_margins(model_def) if any_outpainting else 0
    guide_inpaint_color = model_def.get("guide_inpaint_color", 127.5)
    if image_mode==2:
        guide_inpaint_color = model_def.get("inpaint_color", guide_inpaint_color)
    if callable(guide_inpaint_color):
        guide_inpaint_color = guide_inpaint_color(video_prompt_type, any_outpainting, custom_settings if isinstance(custom_settings, dict) else {})
    guide_inpaint_color = parse_guide_inpaint_color(guide_inpaint_color)
    extract_guide_from_window_start = model_def.get("extract_guide_from_window_start", False) 
    hunyuan_custom = "hunyuan_video_custom" in model_filename
    hunyuan_custom_edit =  hunyuan_custom and "edit" in model_filename
    fantasy = base_model_type in ["fantasy"]
    multitalk = model_def.get("multitalk_class", False)
    fake_start_image = model_def.get("fake_start_image", False) and image_start is not None
    uint8_guides = model_def.get("uint8_guides", False) # control videos and masks reach the model as uint8 (0-255), a quarter of the RAM of float ones
    if callable(uint8_guides): # float ones for the modes where the model works on the whole float control video
        uint8_guides = uint8_guides(video_prompt_type, audio_prompt_type, any_outpainting)

    if (multitalk or model_def.get("speaker_locations", False)) and ("B" in audio_prompt_type or "X" in audio_prompt_type):
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
    else:
        speakers_bboxes = None        
    if "L" in image_prompt_type:
        video_source = prompt_enhancer_images.resolve_continue_video(video_source, image_prompt_type, get_processed_queue(gen)[VIDEO.id][0], save_path)
    fps = 1 if is_image else get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    gen_state = {}
    frame_scheduler = None
    first_window_available_overlap = estimate_first_window_overlap_frames(None if fake_start_image else image_start, video_source, keep_frames_video_source, fps)
    scheduler_supported = frame_scheduler_supported(model_type, model_def, image_mode)
    if not scheduler_supported and has_slash_commands(prompts):
        raise gr.Error("Prompt slash window commands require a video model with Sliding Window support.")
    if scheduler_supported:
        frame_scheduler, frame_scheduler_error = build_frame_scheduler(
            prompts,
            total_frames=int(video_length or frames_minimum),
            fps=float(fps),
            window_size=int(sliding_window_size or video_length or frames_minimum),
            default_overlap=int(sliding_window_overlap or 0),
            minimum=frames_minimum,
            step=frames_steps,
            frame_offset=frames_offset,
            overlap_offset=model_def.get("sliding_window_defaults", {}).get("overlap_offset", 1),
            max_overlap=model_def.get("sliding_window_defaults", {}).get("overlap_max"),
            preserve_exact_output_frames=model_def.get("image_end_frame_position", False),
            output_frame_policy=model_def.get("frame_scheduler_output_policy"),
            supported_model_commands=model_def.get("prompt_slash_commands", []),
            allow_new_shot=image_prompt_types_allow_t2v(model_def, image_mode),
            first_window_overlap_frames=first_window_available_overlap,
            initial_shared_frames=1 if (not fake_start_image and image_start is not None) or video_source is not None else 0,
            discard_last_frames=sliding_window_discard_last_frames,
        )
        if frame_scheduler_error is not None:
            raise gr.Error(frame_scheduler_error)
    if frame_scheduler is not None and frame_scheduler["active"]:
        loras_mult_error = prepare_loras_mult_windows(frame_scheduler, activated_loras, num_inference_steps, lora_multiplier_phases, base_loras_slists=base_loras_slists, model_switch_phase=model_switch_phase, store_slists=True, lora_multiplier_branches=lora_multiplier_branches)
        if loras_mult_error is not None: raise gr.Error(loras_mult_error)
        prompts = frame_scheduler["prompts"]
        video_length = frame_scheduler["predicted_total_frames"]
        current_video_length = video_length
    control_audio_tracks = source_audio_tracks = source_audio_metadata = []
    if postprocess_audio == "control" and video_guide is not None:
        control_audio_tracks, _  = extract_audio_tracks(video_guide, temp_format="wav")
    if "K" in audio_prompt_type and video_guide is not None:
        extracted_guides = [None, None, None]
        for index, guide in enumerate((video_guide, video_guide2, video_guide3)):
            if guide is None:
                continue
            try:
                if extract_audio_tracks(guide, query_only=True) == 0:
                    print(f"No audio track found in Control Video #{index + 1}: {guide}")
                else:
                    extracted_guides[index] = extract_audio_track_to_wav(guide, get_available_filename(save_path, guide, suffix=f"_control_audio{index + 1}", force_extension=".wav"))
                    temp_filenames_list.append(extracted_guides[index])
            except Exception as e:
                print(f"Unable to extract Audio track from Control Video #{index + 1}: {e}")
        audio_guide, audio_guide2, audio_guide3 = extracted_guides
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
    custom_frames_injection = model_def.get("custom_frames_injection", False) and "F" in video_prompt_type and image_refs is not None and len(image_refs) > 0
    if "K" in video_prompt_type: 
        any_background_ref = 2 if model_def.get("all_image_refs_are_background_ref", False) or custom_frames_injection else 1
    fit_canvas = 2 if model_def.get("lock_resolutions", False) else server_config.get("fit_canvas", 0)  # a locked resolution is the output size: inputs are cropped to it
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
        elif skip_steps_cache_type in ("spectrum", "first_block"):
            pass
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
    original_audio_guide3 = audio_guide3
    audio_proj_split = None
    audio_proj_full = None
    audio_scale = audio_scale if model_def.get("audio_scale_name") else None
    audio_context_lens = None
    full_audio_guide_waveform, full_audio_guide_sample_rate = None, 0
    control_video_trim = not model_def.get("control_video_trim_disabled", False) and (model_def.get("control_video_trim", False) or "|" in video_prompt_type)

    if audio_guide != None:
        from preprocessing.extract_vocals import get_vocals
        import librosa
        duration = librosa.get_duration(path=audio_guide)
        combination_type = "add"
        clean_audio_files = "V" in audio_prompt_type
        if audio_guide2 is not None:
            duration2 = librosa.get_duration(path=audio_guide2)
            if clean_audio_files:  # isolate vocals first so normalization measures the voices, not the music
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                audio_guide2 = get_vocals(original_audio_guide2, get_available_filename(save_path, audio_guide2, "_clean2", ".wav"))
                temp_filenames_list += [audio_guide, audio_guide2]
                if audio_guide3 is not None:
                    audio_guide3 = get_vocals(original_audio_guide3, get_available_filename(save_path, audio_guide3, "_clean3", ".wav"))
                    temp_filenames_list.append(audio_guide3)
            if "N" in audio_prompt_type:
                max_total_duration = model_def.get("audio_reference_max_total_duration", None)
                guides = [guide for guide in (audio_guide, audio_guide2, audio_guide3) if guide is not None]
                total_duration = duration + duration2 + (librosa.get_duration(path=audio_guide3) if audio_guide3 is not None else 0)
                normalization_max_duration = float(max_total_duration) / len(guides) if max_total_duration is not None and total_duration > float(max_total_duration) else None
                normalized_guides = normalize_audio_volumes_to_temp_files(guides, output_dir=save_path, prefix="audio_norm_", max_duration_seconds=normalization_max_duration)
                temp_filenames_list += normalized_guides
                audio_guide, audio_guide2, audio_guide3 = (normalized_guides + [None])[:3]
                if normalization_max_duration is not None:
                    duration, duration2 = min(duration, normalization_max_duration), min(duration2, normalization_max_duration)
            if "C" in audio_prompt_type: duration += duration2
            else: duration = min(duration, duration2)
            combination_type = "para" if "P" in audio_prompt_type else "add"
        else:
            if "X" in audio_prompt_type: 
                # dual speaker, voice separation
                from preprocessing.speaker_separator import extract_dual_audio
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
        video_length_limited_by_audio = not model_def.get("video_length_not_limited_by_audio", False) or control_video_trim 
        if "F" in audio_prompt_type:
            full_audio_guide_waveform, full_audio_guide_sample_rate = slice_audio_window(audio_guide, 0, max_source_video_frames, fps, save_path, suffix=f"_full", pad_head=False, pad_tail=False)
        elif video_length_limited_by_audio:
            current_video_length = min(normalize_frame_count(int(fps * duration) + latent_size, frames_minimum, latent_size, frames_offset), current_video_length)
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

    scheduler_active = frame_scheduler is not None and frame_scheduler["active"]
    scheduled_windows_template = [dict(window) for window in frame_scheduler["windows"]] if scheduler_active else []
    scheduled_windows = [dict(window) for window in scheduled_windows_template]
    any_sliding_window = test_any_sliding_window(model_type)
    default_reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap) if any_sliding_window else 0
    if scheduler_active:
        sliding_window = True
        reuse_frames = default_reuse_frames
        current_video_length = scheduled_windows[0]["frame_num"]
    elif any_sliding_window:
        sliding_window = current_video_length > sliding_window_size
        reuse_frames = default_reuse_frames
    else:
        sliding_window = False
        sliding_window_size = current_video_length
        reuse_frames = 0
    if scheduler_active:
        default_windows_template = []
    elif any_sliding_window:
        first_window_overlap = min(max(1, default_reuse_frames), first_window_available_overlap) if video_source is not None else min(1, first_window_available_overlap)
        default_windows_template = build_default_window_plan(total_frames=current_video_length, window_size=sliding_window_size, default_overlap=default_reuse_frames, discard_last_frames=sliding_window_discard_last_frames, minimum=frames_minimum, step=frames_steps, frame_offset=frames_offset, overlap_offset=sliding_window_defaults.get("overlap_offset", 1), max_overlap=sliding_window_defaults.get("overlap_max"), first_window_overlap=first_window_overlap, first_window_available_overlap=first_window_available_overlap, initial_shared_frames=1 if first_window_overlap > 0 else 0, preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy"))
    else:
        default_windows_template = [{"output_frames": current_video_length, "overlap_frames": 0, "discard_last_frames": 0, "trim_last_frames": 0, "frame_num": current_video_length}]
    default_windows = [dict(window) for window in default_windows_template]
    if not scheduler_active:
        sliding_window = len(default_windows) > 1 or default_windows[0]["overlap_frames"] > 0
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
    discard_last_frames = sliding_window_discard_last_frames
    default_discard_last_frames = discard_last_frames
    default_requested_frames_to_generate = current_video_length if scheduler_active else sum(window["output_frames"] for window in default_windows)
    nb_frames_positions = 0
    if scheduler_active:
        default_requested_frames_to_generate = frame_scheduler["predicted_total_frames"]
        current_video_length = scheduled_windows[0]["frame_num"]
    else:
        current_video_length = default_windows[0]["frame_num"]

    first_window_video_length = current_video_length
    original_prompts = display_original_prompts.copy()
    original_alt_prompts = display_original_alt_prompts.copy()
    gen["sliding_window"] = sliding_window 
    while not abort: 
        stop_current_sample = False
        with gen_lock:
            extra_generation += gen.get("extra_orders",0)
            gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation     
        gen["header_text"] = ""    
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        gen_state = {}
        if scheduler_active:
            scheduled_windows = [dict(window) for window in scheduled_windows_template]
        else:
            default_windows = [dict(window) for window in default_windows_template]
        src_video = src_video2 = src_video3 = src_mask = src_mask2 = src_faces = sparse_video_image = full_generated_audio =None
        prefix_video = pre_video_frame = None
        source_video_overlap_frames_count = 0 # number of frames overalapped in source video for first window
        source_video_frames_count = 0  # number of frames to use in source video (processing starts source_video_overlap_frames_count frames before )
        frames_already_processed = []
        frames_already_processed_count = 0
        retained_video_frames = 0  # output timeline at generation FPS, before temporal upsampling
        source_audio_repair_seconds = 0.0
        audio_window_alignment = model_def.get("audio_window_alignment", False) and not (audio_only or is_image or audio_source is not None or control_audio_tracks or audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"])
        external_audio_trim_ranges = []
        overlapped_latents = None
        pre_video_guide_is_hdr = False
        context_scale = None
        window_no = 0
        end_frame_no = 0
        extra_windows = 0
        gen_cache = {}
        stop_sample_scheduled = False
        guide_start_frame = 0 # pos of of first control video frame of current window  (reuse_frames later than the first processed frame)
        keep_frames_parsed = [] # aligned to the first control frame of current window (therefore ignore previous reuse_frames)
        pre_video_guide = None # reuse_frames of previous window
        pre_audio_guide, pre_audio_guide_sample_rate = None, 0
        image_size = default_image_size #  default frame dimensions for budget until it is change due to a resize
        sample_fit_canvas = fit_canvas
        current_video_length = first_window_video_length
        gen["total_windows"] = 1
        gen["window_no"] = 1
        input_waveform, input_waveform_sample_rate = None, 0
        num_frames_generated = 0 # num of new frames created (lower than the number of frames really processed due to overlaps and discards)
        requested_frames_to_generate = default_requested_frames_to_generate # num  of num frames to create (if any source window this num includes also the overlapped source window frames)
        cached_video_guide_processed = cached_video_mask_processed = cached_video_guide_processed2 = cached_video_mask_processed2 = cached_video_guide_processed3 = None
        cached_video_video_start_frame = cached_video_video_end_frame = -1
        start_time = time.time()
        prompt_was_enhanced = prompt_has_history
        alt_prompt_was_enhanced = alt_prompt_has_history
        if auto_prompt_enhancer_requested:
            send_cmd("progress", [0, get_latest_status(state, "Enhancing Prompt")])
            enhancer_kwargs = {"image_prompt_type":  image_prompt_type, "video_prompt_type":  video_prompt_type, "audio_prompt_type":  audio_prompt_type}
            enhancer_kwargs["resolution"] = resolution
            enhancer_kwargs["duration_seconds"] = duration_seconds
            enhancer_kwargs["control_image"] = video_guide if is_image and "V" in video_prompt_type and not video_guides_shown(model_def, image_mode) else None
            enhancer_kwargs["generation_callbacks"] = {"stop_requested": lambda: gen.get("abort", False), "enhancement_progress": EnhancementProgress(lambda value, desc, total, unit: send_cmd("progress", [value, get_latest_status(state, desc), total, unit]))}
            if not is_image and not audio_only:
                frames = min(video_length, sliding_window_size) if model_def.get("sliding_window", False) else video_length
                enhancer_kwargs.update(video_duration_seconds=frames / fps, fps=fps, frame_count=frames)
            multi_prompt_output = prompt_enhancer_outputs_multiple_prompts(prompt_enhancer)
            prompts_to_enhance = original_prompts[:1] if multi_prompt_output else original_prompts
            routed = prompt_enhancer_chaining.uses_routing(prompt_enhancer)
            try:
                continuation_image = None
                enhancer_windows = scheduled_windows if scheduler_active else default_windows
                if "I" in prompt_enhancer and not is_image and not audio_only and image_start is None and video_source is not None and not enhancer_windows[0].get("new_shot", False):
                    send_cmd("progress", [0, get_latest_status(state, "Reading Continuation Start Image")])
                    continuation_image = prompt_enhancer_images.continuation_start_image(video_source, keep_frames_video_source, fps, max_source_video_frames)
                enhancer_start = image_start if image_start is not None else continuation_image
                enhancer_kwargs["caption_input_info"] = prompt_enhancer_images.caption_input_info(enhancer_start, image_end, original_image_refs, enhancer_kwargs["control_image"], "K" in video_prompt_type)
                if "W" in prompt_enhancer:
                    from shared.prompt_enhancer.transcription import transcribe_source_audio

                    offloadobj.unload_all()
                    enhancer_kwargs["source_audio_transcript"] = transcribe_source_audio(audio_guide if "A" in audio_prompt_type else None, gen=gen, report_status=lambda text: send_cmd("progress", [0, get_latest_status(state, text)]), duration_seconds=duration_seconds if model_def.get("prompt_enhancer_transcription_use_duration", False) else None)
                    if enhancer_kwargs["source_audio_transcript"] is None:
                        prompt_enhancer = prompt_enhancer.replace("W", "")
                ensure_prompt_enhancer_loaded(override_profile=override_profile, send_cmd=send_cmd, gen=gen)
                if prompt_enhancer_images.enabled(server_config, prompt_enhancer) or has_slash_commands(original_prompts):
                    original_prompts, enhancer_kwargs["image_contexts"] = prompt_enhancer_images.prepare_auto(original_prompts, None if fake_start_image else image_start, image_end, original_image_refs, windows=scheduled_windows if scheduler_active else default_windows, fps=fps, window_size=sliding_window_size, positions=frames_positions, source_frames=first_window_available_overlap, reset_alignment=reset_control_aligment, model_def=model_def, discard_frames=sliding_window_discard_last_frames, reuse_frames=default_reuse_frames, multi_prompt_output=multi_prompt_output, video=not is_image and not audio_only, video_prompt_type=video_prompt_type, control_image=enhancer_kwargs["control_image"], continuation_image=continuation_image)
                    prompts_to_enhance = original_prompts[:1] if multi_prompt_output else original_prompts
                if routed:
                    enhanced = prompt_enhancer_chaining.run_chain(prompt_enhancer, original_prompts, original_alt_prompts, alt_prompt_inherits, lambda step, values: process_prompt_enhancer(model_type, model_def, step.mode, values, enhancer_start if enhancer_start is not None else image_end, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=enhancer_kwargs, batch_images=True))
                else:
                    enhanced = process_prompt_enhancer(model_type, model_def, prompt_enhancer, prompts_to_enhance, enhancer_start if enhancer_start is not None else image_end, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=enhancer_kwargs)
            finally:
                release_prompt_enhancer()
            if enhanced is not None:
                if routed:
                    if enhanced.prompt_changed:
                        prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type, multi_prompt_output=multi_prompt_output) for value in enhanced.prompts]
                        task["prompt"] = prompt_parser.ENHANCED_PROMPT_PREFIX + prompt_parser.serialize_prompt_units("", prompts, multi_prompts_gen_type)
                        prompt_was_enhanced = True
                    if enhanced.alt_prompt_changed:
                        alt_prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type) for value in enhanced.alt_prompts]
                        task["alt_prompt"] = prompt_parser.serialize_prompt_blocks_with_prefix(alt_prompts, prompt_enhancer_chaining.originals_for_outputs(original_alt_prompts, len(alt_prompts)))
                        alt_prompt_was_enhanced = True
                    print(f"Enhanced prompts: {prompts}; enhanced alt prompts: {alt_prompts}")
                else:
                    enhanced_prompts = enhanced
                    print(f"Enhanced prompts: {enhanced_prompts}" )
                    if multi_prompt_output:
                        enhanced_prompt = normalize_generated_prompt_lines(enhanced_prompts[0], multi_prompts_gen_type, multi_prompt_output=True)
                        enhanced_prompts = prompt_parser.split_prompt_units(enhanced_prompt, multi_prompts_gen_type)
                    else:
                        enhanced_prompts = [normalize_generated_prompt_lines(one_prompt, multi_prompts_gen_type) for one_prompt in enhanced_prompts]
                    # On-the-fly enhancement keeps task prompts clean; originals are saved in metadata.
                    task["prompt"] = prompt_parser.ENHANCED_PROMPT_PREFIX + prompt_parser.serialize_prompt_units("", enhanced_prompts, multi_prompts_gen_type)
                    prompts = enhanced_prompts
                    prompt_was_enhanced = True
                workspace_storage.recording_gallery(gen)["last_gallery_source"] = "audio" if audio_only else MESH.id if model_def.get("mesh_outputs", False) else WORLD.id if model_def.get("world_outputs", False) else VIDEO.id
                send_cmd("output")
                if scheduler_active:
                    for idx, window in enumerate(scheduled_windows):
                        window["prompt"] = prompt_enhancer_images.strip_window_commands([prompts[idx] if idx < len(prompts) else prompts[-1]])[0][0]
                    frame_scheduler["prompts"] = [window["prompt"] for window in scheduled_windows]
                abort = gen.get("abort", False)

 
        while not abort and not stop_current_sample:
            with gen_lock:
                new_extra_windows = gen.get("extra_windows",0)
                gen["extra_windows"] = 0
            extra_windows += new_extra_windows
            if scheduler_active:
                for _ in range(new_extra_windows):
                    scheduled_windows.append(build_extension_window(scheduled_windows[-1]["prompt"], window_size=sliding_window_size, overlap_frames=default_reuse_frames, discard_last_frames=default_discard_last_frames, minimum=frames_minimum, step=frames_steps, frame_offset=frames_offset, preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy")))
                    requested_frames_to_generate += scheduled_windows[-1]["output_frames"]
                if window_no >= len(scheduled_windows):
                    break
                frame_window_options = scheduled_windows[window_no]
                prompt, reuse_frames, current_video_length, new_shot, discard_last_frames = frame_window_options["prompt"], frame_window_options["overlap_frames"], frame_window_options["frame_num"], frame_window_options["new_shot"], frame_window_options["discard_last_frames"]
                automatic_trim_last_frames = frame_window_options["trim_last_frames"]
                if "requested_output_frames" in frame_window_options and abs(frame_window_options["requested_output_frames"] - frame_window_options["output_frames"]) > 1:
                    print(f"Requested frame contribution adjusted from {frame_window_options['requested_output_frames']} to {frame_window_options['output_frames']} for model-compatible scheduling (Sliding Window {window_no + 1}).")
                current_loras_slists = frame_window_options.get("loras_slists", loras_slists)
                sliding_window = True
            else:
                frame_window_options, current_loras_slists, new_shot, discard_last_frames = None, loras_slists, False, default_discard_last_frames
                for _ in range(new_extra_windows):
                    default_windows.append(build_extension_window("", window_size=sliding_window_size, overlap_frames=default_reuse_frames, discard_last_frames=default_discard_last_frames, minimum=frames_minimum, step=frames_steps, frame_offset=frames_offset, preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy")))
                    requested_frames_to_generate += default_windows[-1]["output_frames"]
                if window_no >= len(default_windows):
                    break
                default_window = default_windows[window_no]
                reuse_frames, current_video_length, discard_last_frames = default_window["overlap_frames"], default_window["frame_num"], default_window["discard_last_frames"]
                automatic_trim_last_frames = default_window["trim_last_frames"]
                if "requested_output_frames" in default_window and abs(default_window["requested_output_frames"] - default_window["output_frames"]) > 1:
                    print(f"Requested frame contribution adjusted from {default_window['requested_output_frames']} to {default_window['output_frames']} for model-compatible scheduling (Sliding Window {window_no + 1}).")
                prompt =  prompts[window_no] if window_no < len(prompts) else prompts[-1]
                sliding_window = len(default_windows) > 1 or reuse_frames > 0
            gen["sliding_window"] = sliding_window
            current_alt_prompt = alt_prompts[window_no] if window_no < len(alt_prompts) else alt_prompts[-1]
            if scheduler_active:
                next_overlap_frames = scheduled_windows[window_no + 1]["overlap_frames"] if window_no + 1 < len(scheduled_windows) else default_reuse_frames
            else:
                next_overlap_frames = default_windows[window_no + 1]["overlap_frames"] if window_no + 1 < len(default_windows) else default_reuse_frames

            total_windows = len(scheduled_windows) if scheduler_active else len(default_windows)
            gen["total_windows"] = total_windows
            if window_no >= total_windows:
                break
            window_no += 1
            gen["window_no"] = window_no
            enable_RIFLEx = model_def.get("riflex", False) and image_mode == 0 and (RIFLEx_setting == 0 and current_video_length > (6 * fps + 1) or RIFLEx_setting == 1)
            return_latent_slice = None 
            frames_relative_positions_list = []
            tail_trim_frames = discard_last_frames + automatic_trim_last_frames
            if next_overlap_frames > 0:
                tail_trim_latents = tail_trim_frames // latent_size
                return_latent_slice = slice(-max(tail_trim_latents + 1, (next_overlap_frames + tail_trim_frames) // latent_size), None if tail_trim_latents == 0 else -tail_trim_latents)
            refresh_preview  = {"image_guide" : image_guide, "image_mask" : image_mask} if image_mode >= 1 else {}
            if new_shot:
                pre_video_guide, pre_audio_guide, pre_audio_guide_sample_rate = None, None, 0

            if hasattr(model_handler, "custom_prompt_preprocess"):
                prompt = model_handler.custom_prompt_preprocess(**locals())
            image_start_tensor = image_end_tensor = None
            if window_no == 1 and (video_source is not None or (image_start is not None and not new_shot)):
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
                    source_overlap = 0 if new_shot else min(prefix_video.shape[1], reuse_frames if reuse_frames > 0 else 1)
                    if source_overlap > 0:
                        pre_video_guide = prefix_video[:, -source_overlap:].float()
                        if prefix_video_is_hdr:
                            pre_video_guide_is_hdr = True
                        else:
                            pre_video_guide = pre_video_guide.div_(127.5).sub_(1.) # c, f, h, w
                pre_video_frame = convert_tensor_to_image(prefix_video[:, -1])
                source_video_overlap_frames_count = source_overlap if video_source is not None else pre_video_guide.shape[1]
                source_video_frames_count = prefix_video.shape[1]

                if sample_fit_canvas != None: 
                    image_size = (pre_video_guide if pre_video_guide is not None else prefix_video).shape[-2:]
                    sample_fit_canvas = None
                guide_start_frame =  prefix_video.shape[1]
                if fake_start_image:
                    source_video_overlap_frames_count = source_video_frames_count = guide_start_frame = 0
            if image_end is not None and not (frame_window_options is not None and frame_window_options.get("no_end_image", False)):
                image_end_list=  image_end if isinstance(image_end, list) else [image_end]
                if end_frame_no < len(image_end_list):
                    new_height, new_width = image_size                    
                    image_end_tensor, _, _ = calculate_dimensions_and_resize_image(image_end_list[end_frame_no], new_height, new_width, sample_fit_canvas, fit_crop, block_size = block_size)
                    end_frame_no += 1
                    refresh_preview["image_end"] = image_end_tensor 
                    image_end_tensor = convert_image_to_tensor(image_end_tensor)
                    if sample_fit_canvas != None: 
                        image_size  = image_end_tensor.shape[-2:]
                        sample_fit_canvas = None
                image_end_list= None
            # the end image of a non-final window can be generated before its discarded frames: it steers the window without being shown
            end_image_before_discard = model_def.get("end_image_before_discarded_frames", False) and window_no < total_windows
            image_end_frame_position = current_video_length - (automatic_trim_last_frames if end_image_before_discard else tail_trim_frames) - 1 if model_def.get("image_end_frame_position", False) else None  # also locates an injected frame used as the window's end image
            window_start_frame = guide_start_frame - (reuse_frames if window_no > 1 else source_video_overlap_frames_count)
            guide_end_frame = guide_start_frame + current_video_length - (source_video_overlap_frames_count if window_no == 1 else reuse_frames)
            alignment_shift = source_video_frames_count if reset_control_aligment else 0
            aligned_guide_start_frame = guide_start_frame - alignment_shift
            aligned_guide_end_frame = guide_end_frame - alignment_shift
            aligned_window_start_frame = window_start_frame - alignment_shift  
            input_waveform, input_waveform_sample_rate = None, 0
            input_audio_is_overlap = False
            if full_audio_guide_waveform is not None:
                input_waveform, input_waveform_sample_rate = full_audio_guide_waveform, full_audio_guide_sample_rate
            elif audio_guide is not None and model_def.get("audio_guide_window_slicing", False):
                audio_start_frame = aligned_window_start_frame
                # if reset_control_aligment:
                #     audio_start_frame += source_video_overlap_frames_count
                input_waveform, input_waveform_sample_rate = slice_audio_window(audio_guide, audio_start_frame, current_video_length, fps, save_path, suffix=f"_win{window_no}", pad_tail= video_length_limited_by_audio) 
                if input_waveform.shape[0] == 0:
                    input_waveform, input_waveform_sample_rate = pre_audio_guide, pre_audio_guide_sample_rate
                    input_audio_is_overlap = True
            elif model_def.get("audio_guide_window_slicing", False):
                if pre_audio_guide is not None and pre_audio_guide.shape[0] > 0:
                    input_waveform, input_waveform_sample_rate = pre_audio_guide, pre_audio_guide_sample_rate
                    input_audio_is_overlap = True
                elif window_no == 1 and source_video_overlap_frames_count > 0 and len(source_audio_tracks) > 0:
                    source_audio_start_frame = max(0, source_video_frames_count - source_video_overlap_frames_count)
                    input_waveform, input_waveform_sample_rate = slice_audio_window(source_audio_tracks[0], source_audio_start_frame, source_video_overlap_frames_count, fps, save_path, suffix=f"_source_overlap_win{window_no}", pad_head=False, pad_tail=False)
                    input_audio_is_overlap = True
                    if input_waveform.shape[0] == 0: input_waveform, input_waveform_sample_rate = None, 0
            keep_input_audio = (model_def.get("output_audio_is_input_audio", False) or "S" in audio_prompt_type) and output_new_audio_filepath is not None and "O" not in audio_prompt_type
            audio_overlap_frames = reuse_frames if window_no > 1 else source_video_overlap_frames_count if video_source is not None else 0
            window_audio_repair_seconds = 0.0
            if audio_window_alignment and audio_overlap_frames > 0 and not keep_input_audio and (window_no > 1 or source_audio_tracks):
                overlap_audio = pre_audio_guide if window_no > 1 else input_waveform if input_audio_is_overlap else None
                overlap_rate = pre_audio_guide_sample_rate if window_no > 1 else input_waveform_sample_rate
                if window_no > 1:  # the previous window's audio is in memory, complete when the model returns an audio tail
                    repair_ms = 0 if model_def.get("audio_tail_ms") else model_def.get("audio_overlap_repair_ms", 0)
                else:
                    repair_ms = source_audio_repair_ms(model_def.get("audio_overlap_repair_ms", 0), source_audio_tracks[0], source_audio_metadata[0].get("codec"), source_video_frames_count / fps)
                window_audio_repair_seconds = audio_overlap_repair_seconds(audio_overlap_frames, fps, repair_ms, overlap_audio, overlap_rate)
            audio_conditioning_end = audio_overlap_frames / fps - window_audio_repair_seconds if input_audio_is_overlap and window_audio_repair_seconds > 0 else None
            if fantasy and audio_guide is not None:
                audio_proj_split , audio_context_lens = parse_audio(audio_guide, start_frame = aligned_window_start_frame, num_frames= current_video_length, fps= fps,  device= processing_device  )
            if multitalk:
                from models.wan.multitalk.multitalk import get_window_audio_embeddings
                # special treatment for start frame pos when alignement to first frame requested as otherwise the start frame number will be negative due to overlapped frames (has been previously compensated later with padding)
                audio_proj_split = get_window_audio_embeddings(audio_proj_full, audio_start_idx= aligned_window_start_frame + (source_video_overlap_frames_count if reset_control_aligment else 0 ), clip_length = current_video_length)

            if repeat_no == 1 and window_no == 1 and image_refs is not None and len(image_refs) > 0:
                frames_positions_list = prompt_enhancer_images.resolve_injected_positions(frames_positions, len(image_refs), windows=scheduled_windows if scheduler_active else default_windows, source_frames=source_video_frames_count, source_overlap=source_video_overlap_frames_count, window_size=sliding_window_size, discard_frames=sliding_window_discard_last_frames, reuse_frames=default_reuse_frames, alignment_shift=alignment_shift)
                nb_frames_positions = len(frames_positions_list) 
                if nb_frames_positions > 0:
                    frames_to_inject = [None] * (max(frames_positions_list) + 1)
                    for i, pos in enumerate(frames_positions_list):
                        frames_to_inject[pos] = image_refs[i] 

            video_guide_processed = video_mask_processed = video_guide_processed2 = video_mask_processed2 = video_guide_processed3 = sparse_video_image = None
            skip_video_guide_preprocess = model_def.get("skip_video_guide_preprocess", False)  # True, or video_prompt_type letters that make the model read the guide itself
            skip_video_guide_preprocess = any_letters(video_prompt_type, skip_video_guide_preprocess) if isinstance(skip_video_guide_preprocess, str) else bool(skip_video_guide_preprocess)
            if video_guide is not None and not skip_video_guide_preprocess:
                guide_frames_limit = round(model_def["reference_video_max_frames"] * fps / model_def["fps"]) if reference_videos else max(aligned_guide_end_frame, source_video_frames_count - source_video_overlap_frames_count + requested_frames_to_generate)
                keep_frames_parsed_full, error = parse_keep_frames_video_guide(keep_frames_video_guide, guide_frames_limit)
                if len(error) > 0:
                    raise gr.Error(f"invalid keep frames {keep_frames_video_guide}")
                guide_frames_extract_start = 0 if reference_videos else aligned_window_start_frame if extract_guide_from_window_start else aligned_guide_start_frame
                extra_control_frames = model_def.get("extra_control_frames", 0)
                if extra_control_frames > 0 and aligned_guide_start_frame >= extra_control_frames: guide_frames_extract_start -= extra_control_frames
                        
                keep_frames_parsed = [True] * -guide_frames_extract_start if guide_frames_extract_start  <0 else []
                keep_frames_parsed += keep_frames_parsed_full[max(0, guide_frames_extract_start): guide_frames_limit if reference_videos else aligned_guide_end_frame]
                guide_frames_extract_count = len(keep_frames_parsed)

                process_all = resolve_model_preprocess_all(model_def, base_model_type=base_model_type, video_prompt_type=video_prompt_type, image_prompt_type=image_prompt_type, audio_prompt_type=audio_prompt_type, custom_settings=custom_settings)
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
                    if reference_videos and preprocess_type == "identity":
                        preprocess_type = "raw"
                    custom_preprocessor = model_def.get("custom_preprocessor", None) 
                    if custom_preprocessor is not None:
                        status_info = custom_preprocessor
                        send_cmd("progress", [0, get_latest_status(state, status_info)])
                        video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 =  custom_preprocess_video_with_mask(model_handler, base_model_type, pre_video_guide, video_guide if sparse_video_image is None else sparse_video_image, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  block_size = block_size, expand_scale = mask_expand, video_prompt_type= video_prompt_type, model_def=model_def, custom_settings=custom_settings)
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
                            image_size = calculate_new_dimensions(height, width, h, w, fit_canvas, block_size=block_size, max_width=model_def.get("max_width"), max_height=model_def.get("max_height"))
                            sample_fit_canvas = None 
                            ref_pose_tensor  = resize_and_remove_background(image_refs[nb_frames_positions:nb_frames_positions+1], image_size[1], image_size[0],
                                                                                            False, True, 
                                                                                            fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                            max_width=model_def.get("max_width"), max_height=model_def.get("max_height"),
                                                                                            block_size=block_size,
                                                                                            outpainting_dims =outpainting_dims,
                                                                                            outpainting_ratio = video_guide_outpainting_ratio,
                                                                                            background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                            outpainting_quantize_margins = outpainting_quantize_margins,
                                                                                            return_tensor= True)[0][0]
                        else:
                            ref_pose_tensor = pre_video_guide
 
                        guide_source = video_guide if sparse_video_image is None else sparse_video_image
                        guide_height, guide_width = image_size
                        guide_fit_canvas = sample_fit_canvas
                        if reference_videos:
                            guide_height, guide_width = get_reference_video_dimensions(guide_source, *image_size, model_def["reference_video_max_size"], block_size)
                            guide_fit_canvas = None
                        video_guide_processed, video_mask_processed = preprocess_video_with_mask(ref_pose_tensor, guide_source, video_mask, height=guide_height, width=guide_width, max_frames=guide_frames_extract_count, start_frame=guide_frames_extract_start, fit_canvas=guide_fit_canvas, fit_crop=fit_crop, target_fps=fps, process_type=preprocess_type, expand_scale=mask_expand, negate_mask="N" in video_prompt_type, process_outside_mask=process_outside_mask, outpainting_dims=outpainting_dims, outpainting_ratio=video_guide_outpainting_ratio, proc_no=1, inpaint_color=inpaint_color, block_size=block_size, to_bbox="H" in video_prompt_type, outpainting_quantize_margins=outpainting_quantize_margins, uint8_output=uint8_guides)
                        if preprocess_video_guide2 and "+" in video_prompt_type and video_guide2 is not None:
                            guide2_height, guide2_width = image_size
                            guide2_fit_canvas = sample_fit_canvas
                            if reference_videos:
                                guide2_height, guide2_width = get_reference_video_dimensions(video_guide2, *image_size, model_def["reference_video_max_size"], block_size)
                                guide2_fit_canvas = None
                            video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(None, video_guide2, None, height=guide2_height, width=guide2_width, max_frames=guide_frames_extract_count, start_frame=guide_frames_extract_start, fit_canvas=guide2_fit_canvas, fit_crop=fit_crop, target_fps=fps, process_type=preprocess_type, proc_no=2, block_size=block_size, uint8_output=uint8_guides)
                        elif preprocess_type2 != None:
                            video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(ref_pose_tensor, video_guide, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type2, expand_scale = mask_expand, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, outpainting_ratio = video_guide_outpainting_ratio, proc_no =2, block_size = block_size, to_bbox = "H" in video_prompt_type, outpainting_quantize_margins = outpainting_quantize_margins, uint8_output=uint8_guides)
                        if preprocess_video_guide2 and "*" in video_prompt_type and video_guide3 is not None:
                            guide3_height, guide3_width = get_reference_video_dimensions(video_guide3, *image_size, model_def["reference_video_max_size"], block_size) if reference_videos else image_size
                            video_guide_processed3, _ = preprocess_video_with_mask(None, video_guide3, None, height=guide3_height, width=guide3_width, max_frames=guide_frames_extract_count, start_frame=guide_frames_extract_start, fit_canvas=None if reference_videos else sample_fit_canvas, fit_crop=fit_crop, target_fps=fps, process_type=preprocess_type, proc_no=3, block_size=block_size, uint8_output=uint8_guides)

                    if video_guide_processed is not None and sample_fit_canvas is not None and not reference_videos:
                        image_size = video_guide_processed.shape[-2:]
                        sample_fit_canvas = None

                    if process_all:
                        cached_video_guide_processed, cached_video_mask_processed, cached_video_guide_processed2, cached_video_mask_processed2, cached_video_guide_processed3 = video_guide_processed, video_mask_processed, video_guide_processed2, video_mask_processed2, video_guide_processed3
                        cached_video_video_start_frame = guide_frames_extract_start

                if process_all:
                    process_slice = slice(guide_frames_extract_start - cached_video_video_start_frame, guide_frames_extract_start - cached_video_video_start_frame + guide_slice_to_extract  )
                    video_guide_processed = None if cached_video_guide_processed is None else cached_video_guide_processed[:, process_slice] 
                    video_mask_processed =  None if cached_video_mask_processed is None else cached_video_mask_processed[:, process_slice] 
                    video_guide_processed2 =  None if cached_video_guide_processed2 is None else cached_video_guide_processed2[:, process_slice] 
                    video_mask_processed2 = None if cached_video_mask_processed2 is None else cached_video_mask_processed2[:, process_slice] 
                    video_guide_processed3 = None if cached_video_guide_processed3 is None else cached_video_guide_processed3[:, process_slice]
                    
            if window_no == 1 and image_refs is not None and len(image_refs) > 0:
                ignored_image_refs = model_def.get("no_processing_on_last_images_refs", 0)
                if sample_fit_canvas is not None and (nb_frames_positions > 0 or "K" in video_prompt_type) :
                    from shared.utils.utils import get_outpainting_full_area_dimensions
                    w, h = image_refs[0].size
                    if outpainting_dims != None:
                        h, w = get_outpainting_full_area_dimensions(h, w, outpainting_dims, video_guide_outpainting_ratio)
                    image_size = calculate_new_dimensions(height, width, h, w, fit_canvas, block_size=block_size, max_width=model_def.get("max_width"), max_height=model_def.get("max_height"))
                sample_fit_canvas = None
                if repeat_no == 1:
                    if fit_crop:
                        if any_background_ref == 2:
                            end_ref_position = len(image_refs)
                        elif any_background_ref == 1:
                            end_ref_position = nb_frames_positions + 1
                        else:
                            end_ref_position = nb_frames_positions 
                        end_ref_position = min(end_ref_position, max(nb_frames_positions, len(image_refs) - ignored_image_refs))
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

                        background_removal_color = model_def.get("background_removal_color", [255, 255, 255])
                        if callable(background_removal_color):
                            background_removal_color = background_removal_color(video_prompt_type, custom_settings if isinstance(custom_settings, dict) else {})
                        background_removal_color = parse_guide_inpaint_color(background_removal_color)
                        background_removal_color = list(background_removal_color) if isinstance(background_removal_color, tuple) else background_removal_color
                        src_ref_images, src_ref_masks  = resize_and_remove_background(src_ref_images , image_size[1], image_size[0],
                                                                                        remove_background_images_ref > 0, any_background_ref, 
                                                                                        fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                        max_width=model_def.get("max_width"), max_height=model_def.get("max_height"),
                                                                                        block_size=block_size,
                                                                                        outpainting_dims =outpainting_dims,
                                                                                        outpainting_ratio = video_guide_outpainting_ratio,
                                                                                        background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                        outpainting_quantize_margins = outpainting_quantize_margins,
                                                                                        return_tensor= model_def.get("return_image_refs_tensor", False),
                                                                                        ignore_last_refs=ignored_image_refs,
                                                                                        background_removal_color = background_removal_color,
                                                                                        keep_alpha = model_def.get("image_refs_keep_alpha", False),
                                                                                        rembg_model = model_def.get("background_removal_model", "u2net"))
            custom_postprocessor = model_def.get("custom_image_ref_postprocessor", None)
            if window_no == 1 and repeat_no == 1 and custom_postprocessor is not None:
                src_ref_images, src_ref_masks = custom_postprocessor(
                    src_ref_images, src_ref_masks, image_size[1], image_size[0], image_start, image_prompt_type, image_end, video_prompt_type,
                    send_cmd, model_def, custom_settings, image_start_tensor=image_start_tensor, pre_video_frame=pre_video_frame
                )

            frames_to_inject_parsed = frames_to_inject[ window_start_frame if extract_guide_from_window_start else guide_start_frame: guide_end_frame]
            if (video_guide is not None and not skip_video_guide_preprocess) or len(frames_to_inject_parsed) > 0 and not custom_frames_injection or model_def.get("forced_guide_mask_inputs", False):
                any_mask = video_mask is not None or model_def.get("forced_guide_mask_inputs", False)
                any_guide_padding = model_def.get("pad_guide_video", False)
                dont_cat_preguide = extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None 
                from shared.utils.utils import prepare_video_guide_and_mask
                src_videos, src_masks = prepare_video_guide_and_mask(   [video_guide_processed] + ([] if video_guide_processed2 is None else [video_guide_processed2]) + ([] if video_guide_processed3 is None else [video_guide_processed3]),
                                                                        [video_mask_processed] + ([] if video_guide_processed2 is None else [video_mask_processed2]) + ([] if video_guide_processed3 is None else [None]),
                                                                        None if dont_cat_preguide or fake_start_image and window_no==1 else pre_video_guide, 
                                                                        image_size, current_video_length, latent_size,
                                                                        any_mask, any_guide_padding, guide_inpaint_color, 
                                                                        keep_frames_parsed, [] if custom_frames_injection else frames_to_inject_parsed , outpainting_dims, video_guide_outpainting_ratio, outpainting_quantize_margins=outpainting_quantize_margins, frame_offset=frames_offset)
                video_guide_processed = video_guide_processed2 = video_guide_processed3 = video_mask_processed = video_mask_processed2 = None
                src_video, src_video2, src_video3 = (src_videos + [None, None])[:3]
                src_mask, src_mask2 = (src_masks + [None])[:2] 
                src_videos = src_masks = None
                if control_video_trim:
                    if src_video is None:
                        abort = True 
                        break
                    elif window_no >1 and src_video.shape[1] <= sliding_window_overlap and not dont_cat_preguide:
                        abort = True 
                        break
                    elif src_video.shape[1] < current_video_length:
                        current_video_length = src_video.shape[1]
                        stop_sample_scheduled = True 
                if src_faces is not None:
                    if src_faces.shape[1] < src_video.shape[1]:
                        src_faces = torch.concat( [src_faces,  src_faces[:, -1:].repeat(1, src_video.shape[1] - src_faces.shape[1], 1,1)], dim =1)
                    else:
                        src_faces = src_faces[:, :src_video.shape[1]]
                if (video_guide is not None and not skip_video_guide_preprocess) or len(frames_to_inject_parsed) > 0:
                    if args.save_masks:
                        if src_video is not None: 
                            save_video( guide_to_float(src_video), "masked_frames.mp4", fps)
                            if any_mask: save_video( guide_mask_to_float(src_mask), "masks.mp4", fps, value_range=(0, 1))
                        if src_video2 is not None:
                            save_video( guide_to_float(src_video2), "masked_frames2.mp4", fps)
                            if any_mask: save_video( guide_mask_to_float(src_mask2), "masks2.mp4", fps, value_range=(0, 1))
                if video_guide is not None and not skip_video_guide_preprocess:
                    preview_frame_no = 0 if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else (guide_start_frame - window_start_frame) 
                    if src_video is not None:
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
                # locals() is this frame's own dict (Python 3.11-3.12): kept in a local, the next locals() call would store the dict in itself
                # and the reference cycle would keep every local, the generated video included, until a garbage collection
                new_inputs = None
                send_cmd("output")

            if window_no ==  1:                
                conditioning_latents_size = ( (source_video_overlap_frames_count-1) // latent_size) + 1 if source_video_overlap_frames_count > 0 else 0
            else:
                conditioning_latents_size = ( (reuse_frames-1) // latent_size) + 1

            status = get_latest_status(state)
            gen["progress_status"] = status
            progress_phase = "Generating Audio" if audio_only else "Encoding Prompt"
            gen["progress_phase"] = (progress_phase , -1 )
            tiny_preview = None
            if offloadobj.tiny_vae is not None:
                from shared.tinyvae.session import PreviewSession
                tiny_preview = PreviewSession(offloadobj.tiny_vae, send_cmd, gen, is_image, video=server_config.get("generation_preview", "rgb") == "tiny_vae_video", duration=floor_frame_count(current_video_length, frames_minimum, latent_size, frames_offset) / fps)
            callback = build_callback(state, trans, send_cmd, status, num_inference_steps, preview_meta={"first_latent_only": not model_def.get("preview_all_images", False)} if is_image else None, tiny_preview=tiny_preview)
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
                input_video_for_model = None if new_shot else pre_video_guide
                input_video_is_hdr = pre_video_guide_is_hdr
                prefix_frames_count = 0 if new_shot else source_video_overlap_frames_count if window_no <= 1 else reuse_frames
                prefix_video_for_model = None if model_def.get("joyai_echo", False) or str(base_model_type).startswith("ltx2") else prefix_video
                if new_shot:
                    prefix_video_for_model = None
                if prefix_video_for_model is not None and prefix_video_for_model.dtype == torch.uint8:
                    prefix_video_for_model = prefix_video_for_model.float().div_(127.5).sub_(1.0)
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
                if vae_upsampler_handler is not None and vae_upsampler_session is None:
                    vae_upsampler_session = upsampler_api.prepare_vae_upsampler(vae_upsampler_handler, spatial_upsampling, send_cmd=send_cmd, process_files=process_files_def, init_pipe=partial(init_pipe, output_type=upsampler_api.profile_type_for_handler(vae_upsampler_handler), extension=upsampler_api), profile=compute_profile(override_profile, upsampler_api.profile_type_for_handler(vae_upsampler_handler)), attention_mode=attention_mode, parameter_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images))
                samples = call_with_default_device(model_def, model_type, wan_model.generate,
                    input_prompt = prompt,
                    alt_prompt = current_alt_prompt,
                    image_start = image_start_tensor,  
                    image_end = image_end_tensor,
                    image_end_frame_position=image_end_frame_position,
                    input_frames = src_video,
                    input_frames2 = src_video2,
                    input_frames3 = src_video3,
                    input_ref_images=  src_ref_images,
                    input_ref_masks = src_ref_masks,
                    input_masks = src_mask,
                    input_masks2 = src_mask2,
                    input_video= input_video_for_model,
                    input_faces = src_faces,
                    input_custom = custom_guide,
                    video_guide= video_guide,
                    video_guide2=video_guide2,
                    video_guide3=video_guide3,
                    denoising_strength=denoising_strength,
                    masking_strength=masking_strength,
                    prefix_frames_count = prefix_frames_count,
                    frame_num=floor_frame_count(current_video_length, frames_minimum, latent_size, frames_offset),
                    batch_size = batch_size,
                    height = image_size[0],
                    width = image_size[1],
                    fit_into_canvas = fit_canvas,
                    shift=flow_shift,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    sampling_steps2=num_inference_steps2,
                    custom_sigmas=parse_custom_sigmas(custom_sigmas, max(1, int(guidance_phases))) if model_def.get("custom_sigmas", False) else None,
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
                    audio_conditioning_end=audio_conditioning_end,
                    audio_guide=audio_guide,
                    audio_guide2=audio_guide2,
                    audio_guide3=audio_guide3,
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
                    overlap_size = reuse_frames,
                    sub_parallel_window_size=sub_parallel_window_size,
                    sub_parallel_window_overlap=sub_parallel_window_overlap,
                    color_correction_strength = sliding_window_color_correction_strength,
                    conditioning_latents_size = conditioning_latents_size,
                    input_video_is_hdr=input_video_is_hdr,
                    lora_dir=lora_dir,
                    keep_frames_parsed = keep_frames_parsed,
                    model_filename = model_filename,
                    model_type = base_model_type,
                    loras_slists = current_loras_slists,
                    NAG_scale = NAG_scale,
                    NAG_tau = NAG_tau,
                    NAG_alpha = NAG_alpha,
                    attention_sparsity = attention_sparsity,
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
                    frame_window_options=frame_window_options,
                    gen_state=gen_state,
                    temperature=temperature,
                    window_start_frame_no = window_start_frame,
                    input_video_strength = input_video_strength,
                    self_refiner_setting = self_refiner_setting,
                    self_refiner_plan=self_refiner_plan,
                    self_refiner_f_uncertainty = self_refiner_f_uncertainty,
                    self_refiner_certain_percentage = self_refiner_certain_percentage,
                    duration_seconds=duration_seconds,
                    pause_seconds=pause_seconds,
                    world_output=world_output,
                    top_p=top_p,
                    top_k=top_k,
                    set_progress_status=set_progress_status,
                    loras_selected=loras_selected,
                    frames_relative_positions_list = frames_relative_positions_list,
                    frames_to_inject = frames_to_inject_parsed,
                    verbose_level=verbose_level,
                    gen_cache=gen_cache,
                    vae_upsampler=vae_upsampler_session,
                    save_masks=args.save_masks,
                )
                if tiny_preview is not None:
                    tiny_preview.close()
                upsampler_api.release_vae_upsampler(vae_upsampler_handler, vae_upsampler_session)
                vae_upsampler_session = None
            except Exception as e:
                if tiny_preview is not None:
                    tiny_preview.close(cancel=True)
                upsampler_api.release_vae_upsampler(vae_upsampler_handler, vae_upsampler_session)
                vae_upsampler_session = None
                if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
                    cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)
                remove_temp_filenames(temp_filenames_list)
                gen_state = plugin_data = None
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

                reraise = isinstance(e, (DownloadCancelled, DownloadError, MediaProcessingAborted))
                tb = traceback.format_exc().split('\n')[:-1]
                last_frame = traceback.extract_tb(e.__traceback__)[-1]
                exc = None if reraise else e
                while exc is not None:  # the failed generation's frames, and the generators and closures they ran, still hold its tensors
                    exc.__traceback__, exc = None, exc.__context__
                gc.collect()
                torch.cuda.empty_cache()
                if reraise:
                    raise
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
                    new_error =  gr.Error(f"The generation has encountered an error ({type(e).__name__} in {os.path.basename(last_frame.filename)}:{last_frame.lineno} {last_frame.name}): '{s}'. The full traceback is in the WanGP console.")
                print('\n'.join(tb))
                send_cmd("error", new_error)
                clear_status(state)
                return False
            src_video = src_video2 = src_video3 = src_mask = src_mask2 = None
            if skip_steps_cache != None :
                skip_steps_cache.previous_residual = None
                skip_steps_cache.previous_modulated_input = None
                print(f"Skipped Steps:{skip_steps_cache.skipped_steps}/{skip_steps_cache.num_steps}" )
            generated_audio = audio_tail = None
            drop_generated_audio = False
            audio_repair_samples = 0
            side_files = {}
            meshes = splats = None
            post_decode_pre_trim = 0
            output_audio_sampling_rate= audio_sampling_rate
            sample_is_hdr = False
            if samples != None:
                if isinstance(samples, dict):
                    sample_is_hdr = samples.get("hdr", False)
                    overlapped_latents = samples.get("latent_slice", None)
                    side_files = samples.get("side_files", {})
                    generated_audio = samples.get("audio", generated_audio)
                    overridden_inputs = samples.get("overridden_inputs", None)
                    output_audio_sampling_rate = samples.get("audio_sampling_rate", audio_sampling_rate)
                    input_fills_window = input_waveform is not None and input_waveform.shape[0] >= int(round(current_video_length * input_waveform_sample_rate / fps))
                    if generated_audio is not None:
                        if keep_input_audio and input_fills_window:
                            drop_generated_audio = True
                        elif input_fills_window:
                            output_new_audio_filepath = None
                    post_decode_pre_trim = samples.get("post_decode_pre_trim", 0) 
                    meshes = samples.get("meshes", None)
                    splats = samples.get("splats", None)
                    samples = samples.get("x", None)

                if samples is not None and (audio_only or is_image or sample_is_hdr):
                    samples = samples.to("cpu") # a video stays where it was decoded until it is converted to uint8 below, the RAM only gets the uint8 frames
  
            clear_gen_cache()
            offloadobj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()
            ram_debug_mark("output processing")

            if samples == None:
                abort = True
                state["prompt"] = ""
                send_cmd("output")  
            else:
                sample = samples
                samples = None
                stop_current_sample = stop_sample_scheduled or (not (is_image or audio_only) and sample.shape[1] < current_video_length)
                # if True: # for testing
                #     torch.save(sample, "output.pt")
                # else:
                #     sample =torch.load("output.pt")
                if post_decode_pre_trim > 0 :
                    sample = sample[:, post_decode_pre_trim:]
                if gen.get("extra_windows",0) > 0:
                    sliding_window = True 
                if generated_audio is not None and audio_window_alignment and not drop_generated_audio and model_def.get("audio_tail_ms"):
                    # audio past the last kept frame (the model's tail or discarded frames) is saved with the video: continuing it needs no repair
                    kept_audio_samples = round((sample.shape[1] - (discard_last_frames + automatic_trim_last_frames if sliding_window else 0)) * output_audio_sampling_rate / fps)
                    audio_tail = generated_audio[kept_audio_samples:kept_audio_samples + round(model_def["audio_tail_ms"] * output_audio_sampling_rate / 1000)]
                if sliding_window :
                    guide_start_frame += current_video_length
                    if discard_last_frames > 0:
                        sample = sample[: , :-discard_last_frames]
                        guide_start_frame -= discard_last_frames
                        if generated_audio is not None and not audio_window_alignment:
                            generated_audio = truncate_audio(generated_audio, 0, discard_last_frames, fps, output_audio_sampling_rate)
                    if automatic_trim_last_frames > 0:
                        frame_label = "frame" if automatic_trim_last_frames == 1 else "frames"
                        print(f"{automatic_trim_last_frames} {frame_label} trimmed to match expected duration (Sliding Window {window_no}).")
                        sample = sample[:, :-automatic_trim_last_frames]
                        guide_start_frame -= automatic_trim_last_frames
                        if generated_audio is not None and not audio_window_alignment:
                            generated_audio = truncate_audio(generated_audio, 0, automatic_trim_last_frames, fps, output_audio_sampling_rate)
                    if generated_audio is not None and audio_window_alignment:
                        generated_audio = generated_audio[:round(sample.shape[1] * output_audio_sampling_rate / fps)]
                    if generated_audio is not None and next_overlap_frames > 0 and not drop_generated_audio:
                        overlap_start_sample = round(max(0, sample.shape[1] - next_overlap_frames) * output_audio_sampling_rate / fps)
                        pre_audio_guide = generated_audio[overlap_start_sample:] if audio_window_alignment else generated_audio[-int(round(next_overlap_frames * output_audio_sampling_rate / fps)):]
                        pre_audio_guide_sample_rate = output_audio_sampling_rate
                    else:
                        pre_audio_guide, pre_audio_guide_sample_rate = None, 0

                    pre_video_guide = sample[:, -next_overlap_frames:].to("cpu", copy=True) if next_overlap_frames > 0 else None if scheduler_active else sample[:, max_source_video_frames:].to("cpu", copy=True)
                    pre_video_guide_is_hdr = sample_is_hdr
                    if pre_video_guide is not None and pre_video_guide.dtype == torch.uint8:
                        pre_video_guide =  pre_video_guide.float().div_(127.5).sub_(1.0)
                    window_overlap_frames = source_video_overlap_frames_count if window_no == 1 else reuse_frames
                    trim_first_frames = min(sliding_window_trim_first_frames, max(0, sample.shape[1] - 1)) if window_overlap_frames == 0 else 0
                    if trim_first_frames > 0:
                        audio_trim_start_frame = retained_video_frames + (prefix_video.shape[1] if prefix_video is not None and window_no == 1 else 0)
                        external_audio_trim_ranges.append((audio_trim_start_frame, trim_first_frames))
                        sample = sample[:, trim_first_frames:]
                        if generated_audio is not None:
                            generated_audio = truncate_audio(generated_audio, trim_first_frames, 0, fps, output_audio_sampling_rate)
                if not (audio_only or is_image):
                    if not sample_is_hdr:
                        sample = convert_video_tensor_to_uint8_chunked(sample, max_buffer_mb=64, output_device="cpu")
                if upsampler_api.has_post_model_process_vae_upsampling(spatial_upsampling): # VAE upsamplers reach their final size before source frames are joined and frames are interpolated
                    send_cmd("progress", [0, merge_status_context(status, upsampler_api.method_progress_label(spatial_upsampling))])
                    sample = upsampler_api.post_model_process_vae_upsampling(sample, spatial_upsampling)

                if generated_audio is not None and audio_window_alignment and not drop_generated_audio:
                    generated_audio = generated_audio[:round(sample.shape[1] * output_audio_sampling_rate / fps)]
                    audio_repair_samples = round(window_audio_repair_seconds * output_audio_sampling_rate)
                    if video_source is not None and window_no == 1:
                        source_audio_repair_seconds = audio_repair_samples / output_audio_sampling_rate
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
                            prefix_video = convert_video_tensor_to_uint8_chunked(prefix_video)
                        elif prefix_video.dtype == torch.uint8:
                            prefix_video = prefix_video.float().div_(127.5).sub_(1.0)
                    if prefix_video.shape[1] > 1:
                        if prefix_video.shape[-2:] != sample.shape[-2:]: # a VAE upsampler enlarged the generated frames
                            prefix_video = resize_lanczos_spatial(prefix_video, None, size=sample.shape[-2:])
                        # remove sliding window overlapped frames at the beginning of the generation
                        sample = torch.cat([ prefix_video, sample[: , source_video_overlap_frames_count:]], dim = 1)
                    prefix_video = None
                    guide_start_frame -= source_video_overlap_frames_count 
                    if generated_audio is not None:
                        if audio_window_alignment:
                            trim_samples = round(audio_overlap_frames * output_audio_sampling_rate / fps) - audio_repair_samples
                            generated_audio = generated_audio[trim_samples:]
                        else:
                            generated_audio = truncate_audio( generated_audio, 0 if video_source is None else source_video_overlap_frames_count, 0, fps, output_audio_sampling_rate,)
                elif sliding_window and window_no > 1 and reuse_frames > 0:
                    # remove sliding window overlapped frames at the beginning of the generation
                    sample = sample[: , reuse_frames:]
                    guide_start_frame -= reuse_frames 
                    if generated_audio is not None:
                        if audio_window_alignment:
                            trim_samples = round(reuse_frames * output_audio_sampling_rate / fps) - audio_repair_samples
                            generated_audio = generated_audio[trim_samples:]
                        else:
                            generated_audio = truncate_audio( generated_audio, reuse_frames, 0, fps, output_audio_sampling_rate,)

                num_frames_generated = guide_start_frame - (source_video_frames_count - source_video_overlap_frames_count) 
                if drop_generated_audio: generated_audio = None
                if generated_audio is not None:
                    # The generated track starts at the source boundary, including any repaired source tail.
                    source_audio_frames = source_video_frames_count if video_source is not None else 0
                    committed_audio_samples = int(round(max(0, retained_video_frames - source_audio_frames) * output_audio_sampling_rate / fps))
                    committed_audio_samples += round(source_audio_repair_seconds * output_audio_sampling_rate)
                    if full_generated_audio is None:
                        # a kept soundtrack file is aligned with the start of the Source Video unless control alignment is reset: skip the source part
                        prefix_start_samples = 0 if reset_control_aligment or output_new_audio_data is not None else int(round(source_audio_frames * output_audio_sampling_rate / fps))
                        full_generated_audio = append_sliding_window_audio(output_new_audio_data, output_new_audio_filepath, generated_audio, output_audio_sampling_rate, committed_audio_samples,
                                                                           prefix_start_samples=prefix_start_samples, keep_existing=keep_input_audio and output_new_audio_data is None,
                                                                           trim_ranges=shift_audio_trim_ranges(external_audio_trim_ranges, source_audio_frames), video_fps=fps,
                                                                           align_to_video=audio_window_alignment, repair_samples=audio_repair_samples) if output_new_audio_data is not None or output_new_audio_filepath is not None else generated_audio
                    elif audio_window_alignment:
                        full_generated_audio = append_sliding_window_audio(full_generated_audio, None, generated_audio, output_audio_sampling_rate, committed_audio_samples, align_to_video=True, repair_samples=audio_repair_samples)
                    else:
                        full_generated_audio = np.concatenate([full_generated_audio, generated_audio], axis=0)
                    output_new_audio_data = full_generated_audio
                if not (audio_only or is_image):
                    retained_video_frames += sample.shape[1]


                if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 and not upsampler_api.is_vae_upsampling(spatial_upsampling):
                    spatial_status = upsampler_api.method_progress_label(spatial_upsampling)
                    send_cmd("progress", [0, merge_status_context(status, spatial_status)])
                
                output_fps  = fps
                def upsampler_progress(phase, current_step=None, total_steps=None, unit=None):
                    phase_text = str(phase)
                    gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
                    status_msg = merge_status_context(status, phase_text)
                    if current_step is not None and total_steps is not None and int(total_steps) > 0:
                        send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)] + ([unit] if unit else []))
                    else:
                        send_cmd("progress", [0, status_msg])
                if len(temporal_upsampling) > 0 and not mesh_outputs:
                    sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, previous_last_frame if sliding_window and window_no > 1 else None, temporal_upsampling, fps, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                    if gen.get("abort", False) or sample is None:
                        abort = True
                        break

                if mesh_outputs and splats is None and mesh_postprocessing.selected(spatial_upsampling, temporal_upsampling):
                    processed = [perform_mesh_postprocessing(asset, spatial_upsampling, temporal_upsampling, spatial_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images), temporal_values=parameter_ui.runtime_values(temporal_postprocessing_parameters, postprocessing_reference_images),
                                                             reference_image=original_image_refs[0] if original_image_refs else None,
                                                             seed=seed, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress) for asset in meshes]
                    if gen.get("abort", False):
                        abort = True
                        break
                    meshes, sample = [glb for glb, _ in processed], torch.stack([preview for _, preview in processed], dim=1)
                    del processed
                elif len(spatial_upsampling) > 0 and not upsampler_api.is_vae_upsampling(spatial_upsampling):
                    if is_image:
                        sample = perform_image_spatial_upsampling(sample, spatial_upsampling, seed=seed, vae_tile_size=VAE_tile_size, fps=output_fps, prompt=prompt, negative_prompt=negative_prompt, parameter_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images), abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                        flashvsr_continue_cache = None
                    else:
                        late_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
                        upsampler_uses_audio = late_upsampler is not None and late_upsampler.query_upsampler_def().get("source_audio_conditioning", False)
                        upsampler_audio = full_generated_audio if upsampler_uses_audio and full_generated_audio is not None else input_waveform if upsampler_uses_audio else None
                        upsampler_audio_sample_rate = (output_audio_sampling_rate if full_generated_audio is not None else input_waveform_sample_rate) if upsampler_uses_audio else 0
                        if upsampler_audio is not None and upsampler_audio_sample_rate > 0:
                            required_audio_samples = int(round(sample.shape[1] * upsampler_audio_sample_rate / output_fps))
                            if upsampler_audio.shape[0] < required_audio_samples:
                                upsampler_audio = np.pad(upsampler_audio, ((required_audio_samples - upsampler_audio.shape[0], 0), (0, 0))) if upsampler_audio.ndim == 2 else np.pad(upsampler_audio, (required_audio_samples - upsampler_audio.shape[0], 0))
                            else:
                                upsampler_audio = upsampler_audio[-required_audio_samples:]
                        upsampler_reference_images = original_image_refs[nb_frames_positions:] if original_image_refs is not None else None
                        sample = perform_spatial_upsampling(sample, spatial_upsampling, seed=seed, flashvsr_continue_cache=flashvsr_continue_cache, return_flashvsr_continue_cache=return_flashvsr_continue_cache, vae_tile_size=VAE_tile_size, fps=output_fps, prompt=prompt, negative_prompt=negative_prompt, audio_waveform=upsampler_audio, audio_sample_rate=upsampler_audio_sample_rate, reference_images=upsampler_reference_images, image_refs_relative_size=image_refs_relative_size, parameter_values=parameter_ui.runtime_values(spatial_postprocessing_parameters, postprocessing_reference_images), abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                    if return_flashvsr_continue_cache and not is_image:
                        sample, flashvsr_continue_cache = sample
                    if gen.get("abort", False) or sample is None:
                        abort = True
                        break
                if film_grain_intensity> 0:
                    from postprocessing.film_grain import add_film_grain
                    sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
                if audio_only:
                    output_video_frames = None
                    output_frame_count = None
                    any_soundtrack_processor = False
                elif is_image:
                    output_video_frames = sample.detach().cpu().clone().contiguous() if api_return_video_uint8 else None
                    output_frame_count = None
                    any_soundtrack_processor = False
                else:
                    frames_already_processed.append(sample)
                    frames_already_processed_count += sample.shape[1]
                    output_video_frames = frames_already_processed
                    output_frame_count = frames_already_processed_count
                    sample = None
                    soundtrack_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, frames_count=output_frame_count, fps=fps, audio_source=audio_source, media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension) if audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"] else ""
                    any_soundtrack_processor = bool(postprocess_audio and not soundtrack_error and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"])
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                save_prompt = SLASH_BLOCK_RE.sub("", original_prompts[0]).strip()  # file names show the prompt, not window commands
                if audio_only:
                    audio_codec = server_config.get("audio_stand_alone_output_codec", "wav")
                    extension = get_audio_codec_extension(audio_codec)
                    output_dir = audio_save_path
                elif mesh_outputs:
                    extension = "glb" if splats is None else "ply"
                    output_dir = mesh_save_path
                elif world_only:
                    extension = "wgpworld"
                    output_dir = save_path
                elif is_image:
                    extension = get_image_format(server_config.get("image_output_codec", None), rgba=sample.shape[0] == 4)["ext"][1:]
                    output_dir = image_save_path
                else:
                    container = server_config.get("video_container", "mp4")
                    extension = container
                    output_dir = save_path
                inputs = get_function_arguments(generate_media, locals())
                if overridden_inputs is not None: inputs.update(overridden_inputs)
                if scheduler_active and output_frame_count is not None:
                    inputs["video_length"] = output_frame_count
                if filename_formatter is not None:
                    file_name = filename_formatter.format(inputs)
                    file_name = f"{sanitize_file_name(truncate_for_filesystem(os.path.splitext(os.path.basename(file_name))[0])).strip()}.{extension}"
                    file_name = os.path.basename(get_available_filename(output_dir, file_name))
                else:
                    prompt_name = sanitize_file_name(truncate_for_filesystem(save_prompt)).strip()
                    file_name = f"{time_flag}_seed{seed}{'_' + prompt_name if prompt_name else ''}.{extension}"
                video_path = os.path.join(output_dir, file_name)
                saving_status = f"Saving File {file_name if len(file_name) <= 50 else file_name[:12] + '...' + file_name[-35:]}"

                if not is_image:
                    set_progress_status(saving_status)
                if audio_only:
                    audio_path = os.path.join(output_dir, file_name)
                    audio_path = save_audio_file(audio_path, sample.squeeze(0), output_audio_sampling_rate, audio_codec)
                    video_path = audio_path
                elif mesh_outputs:
                    from shared.utils.mesh_outputs import save_preview
                    os.makedirs(output_dir, exist_ok=True)
                    mesh_path = os.path.join(output_dir, file_name)
                    video_path = []
                    for no, (asset, preview) in enumerate(zip(meshes if splats is None else splats, sample.transpose(1, 0))):  # one preview frame per asset
                        asset_path = get_available_filename(output_dir, mesh_path, "" if no == 0 else f"_{no}")
                        asset_name = os.path.basename(asset_path)
                        set_progress_status(f"Saving File {asset_name if len(asset_name) <= 50 else asset_name[:12] + '...' + asset_name[-35:]}")
                        with open(asset_path, "wb") as writer:
                            writer.write(asset)
                        save_preview(preview, asset_path)
                        video_path.append(asset_path)
                elif world_only:  # the world file is the output (only the frames of its still were decoded)
                    os.makedirs(output_dir, exist_ok=True)
                    with open(video_path, "wb") as writer:
                        writer.write(side_files.pop(".wgpworld"))
                elif is_image:
                    image_path = os.path.join(output_dir, file_name)
                    sample =  sample.transpose(1,0)  #c f h w -> f c h w 
                    new_image_path = []
                    for no, img in enumerate(sample):  
                        img_path = get_available_filename(output_dir, image_path, "" if no == 0 else f"_{no}")
                        img_name = os.path.basename(img_path)
                        set_progress_status(f"Saving File {img_name if len(img_name) <= 50 else img_name[:12] + '...' + img_name[-35:]}")
                        new_image_path.append(save_image(img, save_file = img_path, quality = server_config.get("image_output_codec", None)))

                    video_path= new_image_path
                elif len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0 or output_new_audio_filepath is not None or any_soundtrack_processor or output_new_audio_data is not None or audio_source is not None:
                    video_path = os.path.join(save_path, file_name)
                    save_path_tmp = video_path.rsplit('.', 1)[0] + f"_tmp.{container}"
                    if sample_is_hdr:
                        save_hdr_video(tensor=output_video_frames, save_file=save_path_tmp, fps=output_fps, codec_type=server_config.get("hdr_video_crf", 8), container=container)
                    else:
                        save_video( tensor=output_video_frames, save_file=save_path_tmp, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type = server_config.get("video_output_codec", None), container=container)
                    output_new_audio_temp_filepath = None
                    new_audio_tracks_are_pretrimmed = False
                    saved_audio_tail_seconds = 0.0
                    new_audio_added_from_audio_start = reset_control_aligment or (full_generated_audio is not None and len(control_audio_tracks) == 0) # if not beginning of audio will be skipped
                    source_audio_duration = 0 if video_source is None else source_video_frames_count / fps - source_audio_repair_seconds
                    new_audio_trim_ranges = shift_audio_trim_ranges(external_audio_trim_ranges, source_video_frames_count if new_audio_added_from_audio_start else 0)
                    if any_soundtrack_processor:
                        def audio_progress(phase, current_step=None, total_steps=None):
                            phase_text = f"Audio - {phase}"
                            gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
                            status_msg = get_latest_status(state, phase_text)
                            if current_step is not None and total_steps is not None and int(total_steps) > 0:
                                send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)])
                            else:
                                send_cmd("progress", [0, status_msg])
                        requested_audio_path = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        output_new_audio_filepath = audio_processor_api.generate_soundtrack(
                            postprocess_audio,
                            video_path=save_path_tmp,
                            audio_source=audio_source,
                            prompt=postprocess_audio_prompt,
                            negative_prompt=postprocess_audio_neg_prompt,
                            seed=seed,
                            duration=output_frame_count / fps,
                            output_path=requested_audio_path,
                            send_cmd=send_cmd,
                            status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
                            verbose_level=verbose_level,
                            audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                            process_files=process_files_def,
                            init_pipe=partial(init_pipe, output_type="audio", extension=audio_processor_api),
                            profile=server_config.get("audio_profile", 4),
                            abort_callback=lambda: media_abort_requested(gen),
                            progress_callback=audio_progress,
                        )
                        if gen.get("abort", False) or output_new_audio_filepath is None:
                            abort = True
                            break
                        if os.path.abspath(str(output_new_audio_filepath)) == os.path.abspath(str(requested_audio_path)): output_new_audio_temp_filepath = requested_audio_path
                        new_audio_added_from_audio_start = postprocess_audio_meta["needs_audio_source"]
                    elif audio_source is not None:
                        output_new_audio_filepath = audio_source
                        new_audio_added_from_audio_start =  True
                    elif output_new_audio_data is not None and len(control_audio_tracks) == 0:
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        audio_data_to_write = output_new_audio_data if full_generated_audio is not None or len(new_audio_trim_ranges) == 0 else trim_audio_ranges(output_new_audio_data, new_audio_trim_ranges, fps, output_audio_sampling_rate)
                        new_audio_tracks_are_pretrimmed = full_generated_audio is not None or audio_data_to_write is not output_new_audio_data
                        if full_generated_audio is not None and audio_tail is not None and len(audio_tail):
                            audio_data_to_write, saved_audio_tail_seconds = np.concatenate([audio_data_to_write, audio_tail]), len(audio_tail) / output_audio_sampling_rate
                        write_wav_file(output_new_audio_filepath, audio_data_to_write, output_audio_sampling_rate)
                    if output_new_audio_filepath is not None:
                        new_audio_tracks = [output_new_audio_filepath]
                    else:
                        new_audio_tracks = control_audio_tracks
                    trimmed_mux_audio_tracks = []
                    if len(new_audio_trim_ranges) > 0 and len(new_audio_tracks) > 0 and not (any_soundtrack_processor or audio_source is not None or new_audio_tracks_are_pretrimmed):
                        new_audio_tracks = trimmed_mux_audio_tracks = [ trim_audio_file_ranges(path, new_audio_trim_ranges, fps, get_available_filename(save_path, path, suffix=f"_trim{i}", force_extension=".wav")) for i, path in enumerate(new_audio_tracks) ]
                    replace_voice_temp_audio_tracks = []
                    if replace_voice_method and replace_voice_sample is not None and len(new_audio_tracks) > 0:
                        new_audio_tracks, replace_voice_temp_audio_tracks = audio_processor_api.replace_voice_tracks(replace_voice_method, new_audio_tracks, voice_sample=replace_voice_sample, output_dir=save_path, prefix=f"tmp{time_flag}", process_files=process_files_def, profile_no=server_config.get("audio_profile", 4), verbose_level=verbose_level, init_pipe=partial(init_pipe, output_type="audio", extension=audio_processor_api), voice_sample2=replace_voice_sample2, status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]))
                    if generated_audio is not None: output_new_audio_filepath = None
                    mux_audio_sampling_rate = resolve_mux_audio_sampling_rate(output_audio_sampling_rate, source_audio_metadata, new_audio_tracks)

                    set_progress_status(saving_status)
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
                        pad_audio_to_video=audio_window_alignment and full_generated_audio is not None,
                        verbose=verbose_level >= 2,
                        audio_tail_seconds=saved_audio_tail_seconds,
                    )
                    os.remove(save_path_tmp)
                    if output_new_audio_temp_filepath is not None: os.remove(output_new_audio_temp_filepath)
                    cleanup_temp_audio_files(trimmed_mux_audio_tracks)
                    cleanup_temp_audio_files(replace_voice_temp_audio_tracks)

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
                inputs["model_filename"] = get_model_filename(model_type, compute_quantization(override_values)[0], transformer_dtype_policy)
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
                if replace_voice_method and replace_voice_sample is not None:
                    configs["replace_voice_method"] = replace_voice_method
                if sliding_window: configs["window_no"] = window_no
                configs.pop("prompt_enhancer", None)
                configs.pop("enhanced_prompt", None)
                configs.pop("enhanced_alt_prompt", None)
                configs["prompt"] = prompt_parser.serialize_prompt_blocks_with_prefix(prompts, prompt_enhancer_chaining.originals_for_outputs(original_prompts, len(prompts))) if prompt_was_enhanced else prompt_parser.serialize_prompt_units("", original_prompts, multi_prompts_gen_type)
                configs["alt_prompt"] = prompt_parser.serialize_prompt_blocks_with_prefix(alt_prompts, prompt_enhancer_chaining.originals_for_outputs(original_alt_prompts, len(alt_prompts))) if alt_prompt_was_enhanced else prompt_parser.serialize_prompt_units("", original_alt_prompts, alt_prompt_mode)
                configs["generation_time"] = round(end_time-start_time)
                configs["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
                configs["creation_timestamp"] = int(end_time)
                # if sample_is_image: configs["is_image"] = True
                keep_intermediate_windows = server_config.get("keep_intermediate_sliding_windows", 1)
                record_file_metadata(video_path, configs, is_image, audio_only, gen, embedded_images=embedded_images, replace_last_file=sliding_window and window_no > 1 and not keep_intermediate_windows, notify_generation=not sliding_window or keep_intermediate_windows or window_no == total_windows, output_role="intermediate" if sliding_window and window_no < total_windows else "final")
                output_side_files = process_side_files(video_path[0] if isinstance(video_path, list) else video_path, side_files, return_files=api_return_side_files)
                for side_name in output_side_files:  # a saved world file joins the Worlds gallery
                    side_path = os.path.join(os.path.dirname(video_path[0] if isinstance(video_path, list) else video_path), side_name)
                    if has_world_file_extension(side_name) and os.path.isfile(side_path):
                        record_file_metadata(side_path, configs, False, False, gen, notify_generation=False, record_notification=False)
                if api_return_video_uint8 or api_return_audio or return_flashvsr_continue_cache or (api_return_side_files and output_side_files):
                    media_type = "audio" if audio_only else "3d" if mesh_outputs else "world" if world_only else ("image" if is_image else "video")
                    artifact_audio = output_new_audio_data if api_return_audio else None
                    if artifact_audio is None and api_return_audio and audio_only and sample is not None:
                        artifact_audio = sample.squeeze(0).detach().cpu().float().numpy()
                    artifact_video = output_video_frames if api_return_video_uint8 else None
                    store_api_output_artifact(gen, client_id, video_path, media_type, artifact_video, artifact_audio, output_audio_sampling_rate, output_fps if not audio_only else None, hdr=bool(sample_is_hdr), flashvsr_continue_cache=flashvsr_continue_cache if return_flashvsr_continue_cache else None, side_files=output_side_files if api_return_side_files else None)
                side_files = output_side_files = None
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
    gen_state = plugin_data = None
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

def prepare_generate_media(state):

    if state.get("validate_success",0) != 1:
        return gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.Button(visible=False), gr.Button(visible=False)
    else:
        if not get_gen_info(state).get("in_progress", False):
            clear_status(state)
        params = get_gen_info(state)["queue"][0]["params"]
        extend_visible = can_extend_sample(params)
        early_stop_visible = not _is_edit_task_params(params) and bool(get_model_def(params["model_type"]).get("supports_early_stop", False))
        return gr.Button(visible= False), gr.Button(visible= True), gr.Column(visible= True), gr.Button(visible=extend_visible), gr.Button(visible=early_stop_visible, interactive=True)


def generate_preview(model_type, payload):
    if isinstance(payload, (Image.Image, VideoPreview)):
        return payload
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
    service = service_for(state)
    yield from service.gradio_generation() if service is not None else _process_tasks(state)


def _process_tasks(state):
    from shared.utils.thread_utils import AsyncStream, async_run_in

    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    progress = None

    if len(queue) == 0:
        set_generation_state(gen, status_display=False)
        return
    # Gallery limits apply only to rendered views. Workspaces retain every output.
    set_main_generation_running(state, True)
    gpu_acquired = False  # by the queue worker, so the queue shows as running (and can be aborted) while another process holds the GPU

    def release_gen():
        set_main_generation_running(state, False)
        if not gpu_acquired:
            return
        with gen_lock:
            process_status = gen.get("process_status", None)
            if isinstance(process_status, str) and process_status.startswith("request:"):
                set_generation_state(gen, process_status="process:" + process_status[len("request:"):])
            else:
                set_generation_state(gen, process_status=None)

    start_time = time.time()
    notification_run = notifications.start_queue(gen, len(queue))

    global gen_in_progress
    gen_in_progress = True
    gen["in_progress"] = True
    gen["preview"] = None
    set_generation_state(gen, status=get_task_status_text(queue[0]))
    gen["header_text"] = ""    

    yield time.time(), time.time(), gr.update()

    com_stream = AsyncStream()
    send_cmd = com_stream.output_queue.push

    def queue_worker_func():
        nonlocal gpu_acquired
        gen["prompt_no"] = 0
        try:
            while len(queue) > 0 and not quitting_app:
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
                        gen["api_active_queue_task"] = task

                if task is None:
                    break

                task_id = task["id"] 
                params = task['params']
                send_cmd("status", get_task_status_text(task))
                for key in ["model_filename", "lset_name"]:
                    params.pop(key, None)
                
                try:
                    import inspect
                    model_type = params.get('model_type')
                    if model_type and not _is_edit_task_params(params):
                        default_settings = get_factory_settings(model_type)
                        expected_args = set(inspect.signature(generate_media).parameters.keys())
                        for arg_name in expected_args:
                            if arg_name not in params and arg_name in default_settings:
                                params[arg_name] = default_settings[arg_name]
                    else:
                        expected_args = set(inspect.signature(generate_media).parameters.keys())
                    filtered_params = {k: v for k, v in params.items() if k in expected_args}
                    if _is_edit_task_params(params):
                        filtered_params.setdefault("model_type", "")
                    plugin_data = task.pop('plugin_data', {})
                    gpu_acquired = gpu_acquired or acquire_main_GPU_ressources(state)
                    if gpu_acquired:
                        success = generate_media(task, send_cmd, plugin_data=plugin_data,  **filtered_params)
                        write_vram_debug_report(save_path, params.get("model_type"))
                        write_ram_debug_report(save_path, params.get("model_type"))
                    else:
                        success = True  # aborted while waiting for the GPU: dropped below like any aborted task

                except DownloadCancelled:
                    success = True
                except Exception as e:
                    tb = traceback.format_exc().split('\n')[:-1] 
                    print('\n'.join(tb))
                    send_cmd("error", str(e))
                    return

                with lock:
                    abort = gen.get("abort", False)
                    if abort:
                        record_queue_error(state, [task], "abort", abort=True)
                        gen["abort"] = False
                    gen.pop("api_active_queue_task", None)
                    if success:
                        queue[:] = [item for item in queue if item['id'] != task_id]
                if abort:
                    notification_run.interrupt("Generation aborted by the user.", aborted=True)
                    send_cmd("status", "Video Generation Aborted")
                    send_cmd("output", None)

                clear_status(state)
                if not success:
                    notification_run.interrupt("Generation stopped before the queue completed.")
                    break
                notification_run.task_completed()
                update_global_queue_ref(queue)
                
        except Exception as e:
            traceback.print_exc()
            notification_run.fail(str(e))
            send_cmd("error", f"Queue worker crashed: {e}")
        finally:
            with lock:
                gen.pop("api_active_queue_task", None)
            send_cmd("worker_exit", None)

    async_run_in("generation", queue_worker_func)

    while True:
        cmd, data = com_stream.output_queue.next()
        service = service_for(state)
        if service is not None and cmd not in {"status", "progress"}:
            service.generation_event(cmd, data)
        if cmd == "exit":
            pass
        elif cmd == "worker_exit":
            break
        elif cmd == "info":
            gr.Info(data)
        elif cmd == "error": 
            notification_run.fail(str(data))
            record_queue_error(state, queue, data)
            queue.clear()
            try:
                save_queue_if_crash = server_config.get("save_queue_if_crash", 1)
                if save_queue_if_crash and not quitting_app:  # the quit autosave already holds the queue
                    error_filename = AUTOSAVE_ERROR_FILENAME if save_queue_if_crash == 1 else get_available_filename("", AUTOSAVE_ERROR_FILENAME, f"_{datetime.now():%Y%m%d_%H%M%S}")
                    if _save_queue_to_zip(global_queue_ref, error_filename):
                        print(f"Error Queue autosaved successfully to {error_filename}")
                        from gradio.context import LocalContext
                        if LocalContext.blocks.get() is not None and LocalContext.event_id.get() is not None:
                            gr.Info(f"Error Queue autosaved successfully to {error_filename}")
                    else:
                        print("Autosave Error Queue failed.")
            except Exception as e:
                print(f"Error during autosave: {e}")

            update_global_queue_ref(queue)
            gen["prompts_max"] = 0
            gen["prompt"] = ""
            set_generation_state(gen, status_display=False)
            notifications.finish_queue(server_config, gen, notification_run, time.time() - start_time)
            release_gen()
            raise gr.Error(data, print_exception= False, duration = 0)
        elif cmd == "status":
            set_generation_state(gen, status=data)
            status_text = str(data or "").strip()
            gen["last_progress_args"] = [0, status_text] if len(status_text) > 0 else None
            if service is not None:
                service.generation_event(cmd, data)
        elif cmd == "output":
            gen["preview"] = None
            gen["refresh_tab"] = not gen.get("api_active_queue_task", {}).get("workspace_background")  # agent outputs leave the shown gallery tab alone
            yield time.time(), time.time(), gr.update()
        elif cmd == "progress":
            gen["last_progress_args"] = gen["progress_args"] = data
            if service is not None:
                service.generation_event(cmd, data)
        elif cmd == "preview":
            current_model_type = "unknown"
            with lock:
                if len(queue) > 0:
                    current_model_type = queue[0]["params"].get("model_type")
            
            try:
                if not isinstance(data, (Image.Image, VideoPreview)):
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
        notification_run.interrupt("Generation aborted by the user.", aborted=True)
    if notification_run.interrupted:
        if notification_run.aborted:
            record_queue_error(state, queue[:1], "abort", abort=True)
        status = f"Queue processing was interrupted. Total Generation Time: {format_time(end_time-start_time)}"
    else:
        status = f"Total Generation Time: {format_time(end_time-start_time)}"
        try:
            if server_config.get("notification_sound_enabled", 1):
                volume = server_config.get("notification_sound_volume", 50)
                notification_sound.notify_video_completion(volume=volume)
        except Exception as e:
            print(f"Error playing notification sound: {e}")
    notifications.finish_queue(server_config, gen, notification_run, end_time - start_time)
    set_generation_state(gen, status=status, status_display=False)
    release_gen()


def validate_task(task, state, skip_validate_settings=False):
    """Validate a task's settings. Returns (updated params dict or None, validation error)."""
    params = task.get('params', {})
    if _is_edit_task_params(params):
        inputs = primary_settings.copy()
        inputs.update(params)
        inputs.pop("model_type", None)
        inputs.pop("base_model_type", None)
        fix_postprocess_audio_settings(inputs, params.get("settings_version", 0))
        inputs.setdefault("prompt", "Edit")
        inputs.setdefault("image_mode", 0)
        inputs.setdefault("client_id", "")
        mode = inputs.get("mode", "") or ""
        if mode == "edit_postprocessing":
            video_source = inputs.get("video_source")
            source_is_image = has_image_file_extension(video_source)
            source_is_video = has_video_file_extension(video_source)
            source_is_mesh = mesh_postprocessing.is_mesh_file(video_source)
            temporal_upsampling = inputs.get("temporal_upsampling", "") or ""
            spatial_upsampling = inputs.get("spatial_upsampling", "") or ""
            validation_error = ""
            if not media_source_exists(video_source):
                validation_error = "Selected media file is missing"
            elif source_is_mesh:
                validation_error = mesh_postprocessing.validate_late(spatial_upsampling, temporal_upsampling, parameter_ui.settings_runtime_values(inputs, parameter_ui.SPATIAL), parameter_ui.settings_runtime_values(inputs, parameter_ui.TEMPORAL), video_source)
            elif not (source_is_video or source_is_image):
                validation_error = "Post processing is only available with Videos, Images or 3D Meshes"
            else:
                validation_error = temporal_upsampler_api.validate_temporal_upsampling(temporal_upsampling, source_is_image=source_is_image)
            if not validation_error and not source_is_mesh:
                validation_error = upsampler_api.validate_postprocessing_spatial_upsampling(spatial_upsampling, 1 if source_is_image else 0)
            if not validation_error and not source_is_mesh:
                from postprocessing.film_grain import is_film_grain_enabled
                if len(temporal_upsampling) == 0 and len(spatial_upsampling) == 0 and not is_film_grain_enabled(inputs.get("film_grain_intensity", 0)):
                    validation_error = "You must choose at least one Post Processing Method"
        elif mode == "edit_remux":
            video_source = inputs.get("video_source")
            postprocess_audio = audio_processor_api.normalize_method(inputs.get("postprocess_audio", "") or "")
            validation_error = ""
            if not media_source_exists(video_source):
                validation_error = "Selected video or image file is missing"
            elif not has_video_file_extension(video_source):
                validation_error = "Audio remuxing is only available with Videos"
            elif postprocess_audio and audio_processor_api.method_has_type(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK):
                validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, video_source=video_source, audio_source=inputs.get("audio_source"), media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension)
            elif postprocess_audio and audio_processor_api.method_has_type(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT):
                validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, voice_sample=inputs.get("replace_voice_sample"), voice_sample2=inputs.get("replace_voice_sample2"))
            else:
                validation_error = "You must choose at least one Remux Method"
        elif mode == "edit_audio":
            audio_source = inputs.get("audio_source")
            postprocess_audio = audio_processor_api.normalize_method(inputs.get("postprocess_audio", "") or "")
            validation_error = ""
            if not media_source_exists(audio_source):
                validation_error = "Selected audio file is missing"
            elif not has_audio_file_extension(audio_source):
                validation_error = "Post processing is only available with Audio files"
            elif postprocess_audio:
                validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT, voice_sample=inputs.get("replace_voice_sample"), voice_sample2=inputs.get("replace_voice_sample2"))
            else:
                validation_error = "You must choose at least one Audio Post Processing Method"
        else:
            validation_error = f"Unknown edit mode: {mode}"
        if validation_error:
            return None, validation_error
        return inputs, ""
    model_type = params.get('model_type')
    if not model_type:
        print("  [SKIP] No model_type specified")
        return None, "No model_type specified"

    inputs = params.copy()
    clean_settings(model_type, inputs)

    inputs.setdefault('mode', "")
    if skip_validate_settings:
        return inputs, ""
    if "api" in task.get("plugin_data", {}) or str(inputs.get("client_id", "")).startswith("ai_"):
        try:
            resolution_utils.validate_agent_resolution(get_model_def(model_type), inputs["resolution"], inputs["guidance_phases"], inputs["video_prompt_type"])
        except ValueError as error:
            return None, str(error)
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
    failed = 0
    start_time = time.time()
    notification_run = notifications.start_queue(gen, total_tasks)

    for task_idx, task in enumerate(queue):
        task_no = task_idx + 1
        prompt_preview = (task.get('prompt', '') or '')[:60]
        print(f"\n[Task {task_no}/{total_tasks}] {prompt_preview}...")

        # Validate task settings before processing
        validated_params, validation_error = validate_task(task, state)
        if validated_params is None:
            print(f"  [SKIP] Task {task_no} failed validation: {validation_error or 'Task failed validation.'}")
            skipped += 1
            notification_run.task_skipped()
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
                    # Filter to only valid generate_media params
                    expected_args = set(inspect.signature(generate_media).parameters.keys())
                    filtered_params = {k: v for k, v in params.items() if k in expected_args}
                    if _is_edit_task_params(params):
                        filtered_params.setdefault("model_type", "")
                    filtered_params.setdefault("client_id", "")
                    plugin_data = task.get('plugin_data', {})
                    generate_media(task, send_cmd, plugin_data=plugin_data, **filtered_params)
                    write_vram_debug_report(save_path, params.get("model_type"))
                    write_ram_debug_report(save_path, params.get("model_type"))
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
            notification_run.task_completed()
            print(f"\n  Task {task_no} completed")
        else:
            failed += 1
            notification_run.task_failed()

    release_ram_cache()  # the queue is done: the RAM a next generation of the same model would have reused goes back to the system
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    summary = f"Queue completed: {completed}/{total_tasks} tasks in {format_time(elapsed)}"
    if skipped > 0:
        summary += f" ({skipped} skipped)"
    print(summary)
    notifications.finish_queue(server_config, gen, notification_run, elapsed)
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
    with gen_lock:
        gen["extra_windows"] = 0
        gen["extra_orders"] = 0
        gen["early_stop"] = False
        gen["early_stop_forwarded"] = False
    gen["total_windows"] = 1
    gen["window_no"] = 1
    gen["repeat_no"] = 0
    gen["total_generation"] = 0

def set_saving_file_status(state, gen, send_cmd, file_path):
    """The generic 'Saving File ...' status of a post-processed output, shown while it is encoded and written (the last upsampler or
    processor phase would stay on screen otherwise)."""
    file_name = os.path.basename(file_path)
    phase_text = f"Saving File {file_name if len(file_name) <= 50 else file_name[:12] + '...' + file_name[-35:]}"
    gen["progress_phase"] = (phase_text, -1)
    send_cmd("progress", [0, get_latest_status(state, phase_text)])

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
    with gen_lock:
        extra_orders = gen.get("extra_orders", 0) + 1
        gen["extra_orders"] = extra_orders
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
    with gen_lock:
        extra_windows = gen.get("extra_windows", 0) + 1
        gen["extra_windows"] = extra_windows
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


def _normalize_model_profile_roots(model_type):
    roots = get_model_recursive_prop(model_type, "_profile_roots", return_list=False)
    if isinstance(roots, str):
        roots = [roots]
    elif not isinstance(roots, list):
        roots = []
    roots = [str(root) for root in roots if isinstance(root, str) and len(root) > 0]
    return roots or ["profiles"]


def _builtin_profile_choice_path(root, dir_name, file_path):
    if os.path.isabs(root) or os.path.isabs(dir_name):
        return file_path
    return os.path.join(dir_name, os.path.basename(file_path))


def _builtin_lset_file_path(lset_name):
    return lset_name if os.path.isabs(lset_name) else os.path.join("profiles", lset_name)


def _get_builtin_lset_groups(model_type):
    if model_type is None:
        return []
    group_defs = [
        ("accelerator_profiles", "Accelerators Profiles", "profiles_dir"),
        ("preset_settings", "Presets", "preset_profiles_dir"),
    ]
    builtin_groups = []
    for group_id, title, prop_name in group_defs:
        paths = []
        for dir_name in _normalize_builtin_lset_dirs(get_model_recursive_prop(model_type, prop_name, return_list=False)):
            for root in _normalize_model_profile_roots(model_type):
                cur_path = os.path.join(root, dir_name)
                if not os.path.isdir(cur_path):
                    continue
                cur_dir_presets = glob.glob(os.path.join(cur_path, "*.json"))
                paths += [_builtin_profile_choice_path(root, dir_name, path) for path in cur_dir_presets]
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
    if is_wangp_settings_filename(lset_name) or lset_name.endswith(".lset"):
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
        elif save_lset_prompt_cbox == 3:
            lset = collect_current_model_settings_with_media(state)
            extension = ".zip"
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
        
        if is_wangp_settings_filename(lset_name) or lset_name.endswith(".lset"): lset_name = os.path.splitext(lset_name)[0]
        old_lset_name = ""
        for old_extension in [".json", ".lset", ".zip"]:
            candidate_lset_name = lset_name + old_extension
            if candidate_lset_name in loras_presets:
                old_lset_name = candidate_lset_name
                break
        lset_name = lset_name + extension

        model_type = get_state_model_type(state)
        lora_dir = get_lora_dir(model_type)
        full_lset_name_filename = os.path.join(lora_dir, lset_name ) 

        if extension == ".zip":
            if not _save_queue_to_zip([{"id": 1, "params": lset}], full_lset_name_filename):
                gr.Warning(f"Failed to save Settings File '{lset_name}'")
                lset_choices = compute_lset_choices(model_type, loras_presets)
                lset_choices.append((get_new_preset_msg(), ""))
                return gr.Dropdown(choices=lset_choices, value=""), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= True), gr.Button(visible= False), gr.Button(visible= False), gr.Checkbox(visible= False)
        else:
            with open(full_lset_name_filename, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(lset, indent=4))

        if len(old_lset_name) > 0 :
            if save_lset_prompt_cbox in (2, 3):
                gr.Info(f"Settings File '{lset_name}' has been updated")
            else:
                gr.Info(f"Lora Preset '{lset_name}' has been updated")
            if old_lset_name != lset_name:
                pos = loras_presets.index(old_lset_name)
                loras_presets[pos] = lset_name 
                shutil.move( os.path.join(lora_dir, old_lset_name),  get_available_filename(lora_dir, old_lset_name + ".bkp" ) ) 
        else:
            if save_lset_prompt_cbox in (2, 3):
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
    if loras_choices is None:
        loras_choices = []
    loras_choices_dict = { choice : True for choice in loras_choices}
    for lora in loras:
        loras_choices_dict.pop(lora, False)
    new_loras = loras[:]
    for choice, _ in loras_choices_dict.items():
        new_loras.append(choice)    

    return new_loras, build_choices_hierarchy(new_loras)

def refresh_lora_list(state, lset_name, loras_choices):
    model_type= get_state_model_type(state)
    lora_dir = get_lora_dir(model_type)
    loras, loras_presets, _, _, _, _  = setup_loras(model_type, None, lora_dir, lora_preselected_preset, None)
    state["loras_presets"] = loras_presets
    gc.collect()

    loras_choices = [] if loras_choices is None else loras_choices
    loras, new_loras_hierarchy = get_updated_loras_dropdown(loras, loras_choices)
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
            gr.Info("LoRAs List and Settings List have been refreshed")


    return gr.Dropdown(choices=lset_choices, value= lset_name), gr.update(hierarchy=new_loras_hierarchy, value=loras_choices) 

def update_lset_type(state, lset_name):
    if lset_name.endswith(".lset"):
        return 1
    if lset_name.endswith(".zip"):
        return 3
    return 2

from shared.utils.loras_mutipliers import merge_loras_settings

def apply_lset(state, wizard_prompt_activated, lset_name, loras_choices, loras_mult_choices, prompt):

    state["apply_success"] = 0

    lset_name = get_lset_name(state, lset_name)
    if len(lset_name) == 0:
        gr.Info("Please choose a Lora Preset or Setting File in the list or create one")
        return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt, gr.update(), gr.update(), gr.update()
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
            lora_dir = get_lora_dir(current_model_type)
            loras_choices, loras_mult_choices = merge_loras_settings(old_activated_loras, old_loras_multipliers, loras_choices, loras_mult_choices, "merge after")
            loras_choices = update_loras_url_cache(lora_dir, loras_choices)
            gr.Info(f"Lora Preset '{lset_name}' has been applied")
            state["apply_success"] = 1
            wizard_prompt_activated = "on"

            return wizard_prompt_activated, loras_choices, loras_mult_choices, prompt, get_unique_id(), gr.update(), gr.update()
        else:
            builtin_lset_type = _get_builtin_lset_type(current_model_type, lset_name)
            accelerator_profile = builtin_lset_type == "accelerator_profiles"
            builtin_preset_settings = builtin_lset_type == "preset_settings"
            lset_path = _builtin_lset_file_path(lset_name) if builtin_lset_type is not None else os.path.join(get_lora_dir(current_model_type), lset_name)
            merge_loras = "merge before" if accelerator_profile else "merge after"
            configs, _, _ = get_settings_from_file(state,lset_path , True, True, True, min_settings_version=2.38, merge_loras = merge_loras, skip_validate_settings=True)

            if configs == None:
                gr.Info("File not supported" + (f": {state.get('_last_settings_file_error')}" if state.get("_last_settings_file_error") else ""))
                return [gr.update()] * 7
            settings_bundle_task_count = configs.pop("_settings_bundle_task_count", 0)
            model_type = configs["model_type"]
            configs["lset_name"] = lset_name
            if settings_bundle_task_count > 1:
                gr.Info(f"Settings bundle contains {settings_bundle_task_count} tasks; only the first task has been extracted.")
            help = configs.get("help", None)
            if help is not None: 
                gr.Info(help)
            elif accelerator_profile:
                gr.Info(f"Accelerator Profile '{os.path.splitext(os.path.basename(lset_name))[0]}' has been applied")
            elif builtin_preset_settings:
                gr.Info(f"Preset Settings '{os.path.splitext(os.path.basename(lset_name))[0]}' have been applied")
            else:
                gr.Info(f"Settings File '{os.path.basename(lset_name)}' has been applied")
            if model_type == current_model_type:
                set_model_settings(state, current_model_type, configs)        
                return *[gr.update()] * 4, gr.update(), get_unique_id(), gr.update()
            else:
                set_model_settings(state, model_type, configs)        
                state["ignore_save_form"] = True
                state["skip_resolution_transfer_on_model_switch"] = model_type
                return *[gr.update()] * 5, gr.update(), _model_choice_target_value(model_type)

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
    update_config(server_config, server_config_filename, {"last_advanced_choice": new_advanced})

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
    
    lora_dir = get_lora_dir(model_type)    
    inputs["activated_loras"] = [get_lora_URL(lora_dir, lora) for lora in loras_choices]
    model_def = get_model_def(model_type)
    custom_settings = get_model_custom_settings(model_def)
    parsed_custom_settings, _ = collect_custom_settings_from_inputs(model_def, inputs, strict=False)
    inputs["custom_settings"] = parsed_custom_settings if len(custom_settings) > 0 else None
    clear_custom_setting_slots(inputs)
    inputs["guidance_phases"], inputs["video_prompt_type"] = normalize_phase_2_tiling_selection(model_def, inputs["guidance_phases"], inputs["video_prompt_type"])
    inputs.pop("pace", None)
    inputs.pop("exaggeration", None)
    
    if target in ["state", "edit_state"]:
        inputs["settings_version"] = settings_version
        return inputs
    
    if "lset_name" in inputs:
        inputs.pop("lset_name")
        
    unsaved_params = ATTACHMENT_KEYS
    if target == "metadata" and inputs["custom_guide"] is not None:
        inputs["custom_guide_used"] = True
    for k in unsaved_params:
        inputs.pop(k)
    inputs["type"] = get_model_record(get_model_name(model_type))  
    inputs["settings_version"] = settings_version
    base_model_type = get_base_model_type(model_type)
    if model_type != base_model_type:
        inputs["base_model_type"] = base_model_type
    if target == "settings":
        if model_def.get("input_video_strength") is None:
            inputs.pop("input_video_strength", None)
        return inputs

    if target == "metadata":
        effective_attention_mode = generation_overrides.get_override(inputs.get("overrides", ""), "attention_mode") or get_overridden_attention(model_type) or attention_mode
        prompt_enhancer_visible = server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0
        settings_metadata.clean_metadata_settings(inputs, model_def, attention_mode=effective_attention_mode, prompt_enhancer_visible=prompt_enhancer_visible)
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


def init_generate(state, input_file_list, last_choice, *strip_selected):
    """Records the gallery selections of the page (image/video gallery, then the strip galleries in STRIP_SOURCES order)."""
    if service_for(state) is not None:
        return get_unique_id(), ""
    gen = get_gen_info(state)
    current_gallery_source = gen.get("current_gallery_source", VIDEO.id)
    selected_video_time = gen.get("selected_video_time", None)
    for source, choice in zip(GALLERY_SOURCES, (last_choice, *strip_selected)):
        files, _ = get_file_list(state, input_file_list if source is VIDEO else None, source.id)
        if len(files) > 0: choice = max(choice, 0)
        gen[source.last_selected_key] = (choice + 1) >= len(files)
        gen[source.selected_key] = choice
    file_list, last_choice = gen["file_list"], gen["selected"]
    gen["current_gallery_source"] = current_gallery_source
    gen["selected_video_time"] = selected_video_time if current_gallery_source == VIDEO.id and 0 <= last_choice < len(file_list) and has_video_file_extension(file_list[last_choice]) else None
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
    ui_settings = get_current_model_settings(state)
    video_prompt_type = ui_settings.get("video_prompt_type", "") or ""
    one_image_ref_only = model_def.get("one_image_ref_only", False) and "I" in video_prompt_type and (model_def.get("one_image_ref_only_with_background", False) or not any_letters(video_prompt_type, "KF"))
    if model_def.get("one_image_ref_needed", False) or one_image_ref_only:
        gr.Info(f"Selected Image was set to {target_name}")
        target =[file_list[choice]]
    else:
        gr.Info(f"Selected Image was added to {target_name}")
        if target == None:
            target =[]
        target.append( file_list[choice])
    return gr.update(value=target, selected_index=len(target) - 1)

def image_to_ref_image_set(state, input_file_list, choice, target, target_name):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info(f"Selected Image was copied to {target_name}")
    return file_list[choice]

def image_to_ref_image_guide(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update(), gr.update(), gr.update()
    ui_settings = get_current_model_settings(state)
    gr.Info(f"Selected Image was copied to Control Image")
    new_image = file_list[choice]
    if ui_settings.get("image_mode",0)==2 or True:
        return new_image, new_image, None
    else:
        return new_image, None, None

def mesh_to_custom_guide(state, input_file_list, choice, video_prompt_type, audio_prompt_type, model_mode):
    """The selected mesh as the Input Mesh, switching to the first mode that shows it (e.g. Texture a Mesh) when the current one hides it."""
    file_list, file_settings_list = get_file_list(state, unpack_strip_list(input_file_list), MESH.id)
    visibility = refresh_custom_settings_visibility(state, video_prompt_type, audio_prompt_type, model_mode)
    if len(file_list) == 0 or choice == None or choice < 0 or choice >= len(file_list): return [gr.update()] * (2 + len(visibility))
    if not mesh_postprocessing.is_mesh_file(file_list[choice]):
        gr.Info("A Gaussian splat can not be used as an Input Mesh")
        return [gr.update()] * (2 + len(visibility))
    model_def = get_model_def(get_state_model_type(state))
    guide_def, message = model_def["custom_guide"], "Selected 3D Asset was copied to the Input Mesh"
    if not custom_setting_visible(guide_def, video_prompt_type, audio_prompt_type, model_mode):
        label, model_mode = next((label, value) for label, value in model_def["model_modes"]["choices"] if custom_setting_visible(guide_def, video_prompt_type, audio_prompt_type, value))
        visibility, message = refresh_custom_settings_visibility(state, video_prompt_type, audio_prompt_type, model_mode), f"{message}, Mode set to {label}"
    gr.Info(message)
    return [file_list[choice], model_mode, *visibility]

def explore_world_outside_queue(state, model_type, world_path):
    """An exploration started while a generation runs: it takes the GPU (the generation pauses at its next step, its models unloaded from
    VRAM) instead of waiting in the queue. The paused generation's own model serves it when it is the world's model."""
    from shared.attention import attention_shared_state
    acquire_GPU_ressources(state, "world_exploration", "World Exploration", gr=gr)
    explore_offloadobj = None
    try:
        if transformer_type == model_type and wan_model is not None:
            model = wan_model
        else:
            model, explore_offloadobj = load_models(model_type, track_as_main=False)
            offload_registry.register_offloadobj("world_exploration", explore_offloadobj)
        with attention_shared_state():  # the attention mode a generation runs with
            model.explore(world_path)
    finally:
        if explore_offloadobj is not None:
            offload_registry.unregister_offloadobj("world_exploration", explore_offloadobj)
            explore_offloadobj.release()
        release_GPU_ressources(state, "world_exploration")

def world_to_custom_guide(state, input_file_list, choice):
    file_list, file_settings_list = get_file_list(state, unpack_strip_list(input_file_list), WORLD.id)
    if len(file_list) == 0 or choice == None or choice < 0 or choice >= len(file_list): return gr.update()
    if ".wgpworld" not in (get_model_def(get_state_model_type(state)).get("custom_guide") or {}).get("file_types", []):
        gr.Info("Select a world model first: its generation continues the world")
        return gr.update()
    gr.Info("Selected World will be continued by the next generation (the prompt's segments follow its end)")
    return file_list[choice]

def audio_to_source_set(state, input_file_list, choice, target_name):
    file_list, file_settings_list = get_file_list(state, unpack_strip_list(input_file_list), "audio")
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list): return gr.update()
    gr.Info(f"Selected Audio File was copied to {target_name}")
    return file_list[choice]


def apply_post_processing(state, input_file_list, choice, PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation,
                          PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_postprocessing_reference_images, seed):
    gen = get_gen_info(state)
    if gen.get("current_gallery_source", VIDEO.id) == MESH.id:  # a mesh of the 3D gallery
        file_list, file_settings_list = get_file_list(state, None, MESH.id)
        choice = gen.get(MESH.selected_key, -1)
    else:
        file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice >= len(file_list)  :
        return gr.update(), gr.update(), gr.update()
    
    media_file = file_list[choice]
    overrides = {
        "temporal_upsampling":PP_temporal_upsampling,
        "spatial_upsampling":PP_spatial_upsampling,
        "spatial_postprocessing_parameters": dict(PP_spatial_postprocessing_parameters or {}),
        "temporal_postprocessing_parameters": dict(PP_temporal_postprocessing_parameters or {}),
        "postprocessing_reference_images": list(PP_postprocessing_reference_images or []),
        "image_refs_relative_size": 100.0,
        "seed": seed,
        "film_grain_intensity": PP_film_grain_intensity, 
        "film_grain_saturation": PP_film_grain_saturation,
    }
    is_image = has_image_file_extension(media_file)
    is_video = has_video_file_extension(media_file)
    is_mesh = mesh_postprocessing.is_mesh_file(media_file)
    validation_error = ""
    if not media_source_exists(media_file):
        validation_error = "Selected media file is missing"
    elif is_mesh:
        validation_error = mesh_postprocessing.validate_late(PP_spatial_upsampling, PP_temporal_upsampling, parameter_ui.settings_runtime_values(overrides, parameter_ui.SPATIAL),
                                                             parameter_ui.settings_runtime_values(overrides, parameter_ui.TEMPORAL), media_file)
    elif not (is_video or is_image):
        validation_error = "Post processing is only available with Videos, Images or 3D Meshes"
    else:
        validation_error = temporal_upsampler_api.validate_temporal_upsampling(PP_temporal_upsampling, source_is_image=is_image)
    if not validation_error and not is_mesh:
        validation_error = upsampler_api.validate_postprocessing_spatial_upsampling(PP_spatial_upsampling, 1 if is_image else 0)
    if not validation_error and not is_mesh:
        from postprocessing.film_grain import is_film_grain_enabled
        if len(PP_temporal_upsampling or "") == 0 and len(PP_spatial_upsampling or "") == 0 and not is_film_grain_enabled(PP_film_grain_intensity):
            validation_error = "You must choose at least one Post Processing Method"
    if validation_error:
        gr.Info(validation_error)
        return gr.update(), gr.update(), gr.update()

    gen["edit_media_source"] = media_file
    gen["edit_overrides"] = overrides

    in_progress = gen.get("in_progress", False)
    return "edit_postprocessing", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()


def remux_audio(state, input_file_list, choice, PP_postprocess_audio, PP_postprocess_audio_prompt, PP_postprocess_audio_neg_prompt, PP_postprocess_audio_seed, PP_repeat_generation, PP_custom_audio, PP_replace_voice_sample, PP_replace_voice_sample2):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, input_file_list)
    if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
        return gr.update(), gr.update(), gr.update()
    
    overrides = {
        "postprocess_audio": audio_processor_api.normalize_method(PP_postprocess_audio or ""),
        "postprocess_audio_prompt" : PP_postprocess_audio_prompt,
        "postprocess_audio_neg_prompt": PP_postprocess_audio_neg_prompt,
        "seed": PP_postprocess_audio_seed,
        "repeat_generation": PP_repeat_generation,
        "audio_source": PP_custom_audio,
        "replace_voice_sample": PP_replace_voice_sample,
        "replace_voice_sample2": PP_replace_voice_sample2,
    }
    video_source = file_list[choice]
    postprocess_audio = overrides["postprocess_audio"]
    validation_error = ""
    if not media_source_exists(video_source):
        validation_error = "Selected video or image file is missing"
    elif not has_video_file_extension(video_source):
        validation_error = "Audio remuxing is only available with Videos"
    elif postprocess_audio and audio_processor_api.method_has_type(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK):
        validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, video_source=video_source, audio_source=overrides["audio_source"], media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension)
    elif postprocess_audio and audio_processor_api.method_has_type(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT):
        validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, voice_sample=PP_replace_voice_sample, voice_sample2=PP_replace_voice_sample2)
    else:
        validation_error = "You must choose at least one Remux Method"
    if validation_error:
        gr.Info(validation_error)
        return gr.update(), gr.update(), gr.update()

    gen["edit_media_source"] = video_source
    gen["edit_overrides"] = overrides

    in_progress = gen.get("in_progress", False)
    return "edit_remux", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()

def postprocess_audio_file(state, audio_files_paths, choice, PP_late_audio_postprocess, PP_late_audio_replace_voice_sample, PP_late_audio_replace_voice_sample2):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, unpack_strip_list(audio_files_paths), "audio")
    if len(file_list) == 0 or choice is None or choice < 0 or choice >= len(file_list):
        return gr.update(), gr.update(), gr.update()
    choice = int(choice)
    audio_path = file_list[choice]
    postprocess_audio = audio_processor_api.normalize_method(PP_late_audio_postprocess or "")
    validation_error = ""
    if not media_source_exists(audio_path):
        validation_error = "Selected audio file is missing"
    elif not has_audio_file_extension(audio_path):
        validation_error = "Post processing is only available with Audio files"
    elif postprocess_audio:
        validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT, voice_sample=PP_late_audio_replace_voice_sample, voice_sample2=PP_late_audio_replace_voice_sample2)
    else:
        validation_error = "You must choose at least one Audio Post Processing Method"
    if validation_error:
        gr.Info(validation_error)
        return gr.update(), gr.update(), gr.update()
    gen["edit_audio_source"] = audio_path
    gen["edit_overrides"] = {
        "postprocess_audio": postprocess_audio,
        "replace_voice_sample": PP_late_audio_replace_voice_sample,
        "replace_voice_sample2": PP_late_audio_replace_voice_sample2,
        "repeat_generation": 1,
    }
    in_progress = gen.get("in_progress", False)
    return "edit_audio", get_unique_id() if not in_progress else gr.update(), get_unique_id() if in_progress else gr.update()

def clear_deleted_files(state, source_id):
    gen = get_gen_info(state)
    source = gallery_source(source_id)
    with lock:
        file_list = gen.get(source.files_key, [])
        file_settings_list = gen.get(source.settings_key, [])
        selected_key = source.selected_key
        choice = gen.get(selected_key, -1)
        selected_path = file_list[choice] if 0 <= choice < len(file_list) else None
        new_file_list = []
        new_file_settings_list = []
        for file_path, file_settings in zip(file_list, file_settings_list):
            if os.path.isfile(file_path):
                new_file_list.append(file_path)
                new_file_settings_list.append(file_settings)
        changed = len(file_list) != len(new_file_list)
        file_list[:]=new_file_list 
        file_settings_list[:]=new_file_settings_list 
        if changed:
            gen[selected_key] = new_file_list.index(selected_path) if selected_path in new_file_list else min(choice, len(new_file_list) - 1)
            gen[source.last_selected_key] = gen[selected_key] + 1 >= len(new_file_list)
    service = service_for(state)
    if changed and service is not None:
        service.host_changed()

def eject_video_from_gallery(state, input_file_list, choice):
    # print(f"eject:{time.time()}")
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
        choice = min(choice, len(file_list) - 1)
    set_file_choice(gen, file_list, choice)
    return gallery_update(file_list, choice), gr.update() if len(file_list) >0 else get_default_video_info(), gr.Row(visible= len(file_list) > 0)

def eject_from_strip_gallery(state, source_id, input_file_list, choice):
    gen = get_gen_info(state)
    file_list, file_settings_list = get_file_list(state, unpack_strip_list(input_file_list), source_id)
    with lock:
        if len(file_list) == 0 or choice == None or choice < 0 or choice > len(file_list)  :
            return [gr.update()] * 5
        
        extend_list = file_list[choice + 1:] # inplace List change
        file_list[:] = file_list[:choice]
        file_list.extend(extend_list)

        extend_list = file_settings_list[choice + 1:]
        file_settings_list[:] = file_settings_list[:choice]
        file_settings_list.extend(extend_list)
        choice = min(choice, len(file_list)-1)
    set_file_choice(gen, file_list, choice, source_id)
    return *pack_strip_gallery_state(file_list, choice), gr.update() if len(file_list) >0 else get_default_video_info(), gr.Row(visible= len(file_list) > 0)


def _import_source(file_path):
    """Gallery source of an imported file without WanGP settings, None when the file cannot be read."""
    try:
        if has_audio_file_extension(file_path):
            return gallery_source("audio")
        if has_video_file_extension(file_path):
            return VIDEO if get_video_info(file_path)[0] > 0 else None
        if has_image_file_extension(file_path):
            _open_image_input(file_path).size
            return VIDEO
        if os.path.splitext(file_path)[1].lower() in MESH_IMPORT_EXTENSIONS:
            return MESH
        if has_world_file_extension(file_path):
            return WORLD
        if has_app_file_extension(file_path) and python_apps_level() != "off":
            from shared.apps.bundles import package_manifest
            package_manifest(file_path)  # a readable package with its app.json
            return APP
    except Exception:
        pass
    return None

def add_videos_to_gallery(state, input_file_list, choice, files_to_load):
    from shared.utils.media_imports import persist_gallery_import
    from shared.utils.mesh_outputs import import_mesh_file
    gen = get_gen_info(state)
    output_count = 3 + 3 * len(STRIP_SOURCES) + 3
    if files_to_load == None:
        gr.Info("Please Select a File To Import")
        return [gr.update()] * output_count
    get_file_list(state, input_file_list)
    new_choices, last_source = {}, VIDEO
    valid_files_count = 0
    invalid_files_count = 0
    for file_path in files_to_load:
        file_settings, _, audio_file = get_settings_from_file(state, file_path, False, False, False)
        source = gallery_source("audio") if audio_file else source_for_path(file_path) if file_settings is not None else _import_source(file_path)
        if source is None:
            invalid_files_count += 1
            continue
        if source is MESH:
            file_path, reused = import_mesh_file(file_path, workspace_storage.output_config(server_config, state=state)['mesh_save_path'], filename=getattr(file_path, 'orig_name', None))
        else:
            file_path, reused = persist_gallery_import(file_path, workspace_storage.output_config(server_config, state=state)['save_path'], filename=getattr(file_path, 'orig_name', None))
        if source is APP:
            from shared.utils.app_outputs import extract_thumbnail
            extract_thumbnail(file_path)
        if reused:
            gr.Info(f"Existing file reused: {os.path.basename(file_path)}")
        files, settings_list = get_file_list(state, None, source.id)
        with lock:
            if file_path in files:
                index = files.index(file_path)
                files.pop(index)
                file_settings = settings_list.pop(index)
            files.append(file_path)
            settings_list.append(file_settings)
            new_choices[source.id], last_source = len(files) - 1, source
            valid_files_count +=1

    if valid_files_count== 0 and invalid_files_count ==0:
        gr.Info("No Valid Media to Import")
        return [gr.update()] * output_count
    else:
        txt = ""
        if valid_files_count > 0:
            txt = f"{valid_files_count} files were added. " if valid_files_count > 1 else  f"One file was added."
        if invalid_files_count > 0:
            txt += f"Unable to add {invalid_files_count} files which were invalid. " if invalid_files_count > 1 else  f"Unable to add one file which was invalid."
        gr.Info(txt)
    choices = {}
    for source in GALLERY_SOURCES:
        files = gen.get(source.files_key, [])
        if source.id in new_choices:
            source_choice = new_choices[source.id]
            gen[source.last_selected_key] = source_choice == len(files) - 1
        else:
            source_choice = min(len(files) - 1, gen.get(source.selected_key, choice if source is VIDEO else -1))
        gen[source.selected_key] = choices[source.id] = source_choice
    gen['current_gallery_source'] = last_source.id
    service = service_for(state)
    if service is not None:
        service.host_changed()
    return *gallery_tab_updates(last_source.id), gallery_update(gen.get("file_list", []), choices[VIDEO.id], preview=True), *pack_strip_galleries(gen, choices), gr.Files(value=[]),  gr.Tabs(selected="video_info"), last_source.id

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


def collect_current_model_settings_with_media(state):
    model_type = get_state_model_type(state)
    settings = (get_model_settings(state, model_type) or {}).copy()
    settings["model_filename"] = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
    settings["model_type"] = model_type
    return settings

def export_settings(state, include_media=False):
    model_type = get_state_model_type(state)
    filename = sanitize_file_name(model_type + "_" + datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss") + (".zip" if include_media else ".json"))
    if include_media:
        queue = [{"id": 1, "params": collect_current_model_settings_with_media(state)}]
        return gradio_downloads.register_download(filename, "application/zip", lambda: gradio_downloads.stream_writer(lambda writer: _save_queue_to_zip(queue, writer)))
    text = json.dumps(collect_current_model_settings(state), indent=4)
    data = text.encode("utf-8")
    return gradio_downloads.register_download(filename, "application/json", lambda: gradio_downloads.stream_bytes(data))


def extract_and_apply_source_images(file_path, current_settings):
    from shared.utils.mesh_outputs import extract_mesh_source_images
    from shared.utils.video_metadata import extract_source_images
    if not os.path.isfile(file_path): return 0
    extracted_files = extract_mesh_source_images(file_path) if has_mesh_file_extension(file_path) else extract_source_images(file_path)
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
    if source != VIDEO.id:
        input_file_list = unpack_strip_list(input_file_list)
    file_list, file_settings_list = get_file_list(state, input_file_list, source)
    if choice != None and len(file_list)>0:
        choice= max(0, choice)
        configs = file_settings_list[choice]
        file_name= file_list[choice]
        if configs == None or "model_type" not in configs:  # e.g. a 3D viewer snapshot: a comment only
            gr.Info("No Settings to Extract")
        elif is_deepy_display_metadata(configs):
            gr.Info("Deepy helper metadata is display-only and cannot be loaded as WanGP settings")
        else:
            configs = configs.copy()
            current_model_type = get_state_model_type(state)
            model_type = configs["model_type"] 
            metadata_model_def = get_model_def(model_type)
            multi_prompts_gen_type = prompt_parser.normalize_multi_prompts_mode(configs.get("multi_prompts_gen_type"), "FG")

            def combine_prompt_history(prompt_key, enhanced_prompt_key, mode):
                history = prompt_parser.parse_prompt_history(configs.get(prompt_key, ""), configs.get(enhanced_prompt_key, ""), mode)
                if history is not None:
                    original_prompts, enhanced_prompts = history
                    configs[prompt_key] = prompt_parser.serialize_prompt_blocks_with_prefix(enhanced_prompts, prompt_enhancer_chaining.originals_for_outputs(original_prompts, len(enhanced_prompts)))
                configs.pop(enhanced_prompt_key, None)

            combine_prompt_history("prompt", "enhanced_prompt", multi_prompts_gen_type)
            alt_prompt_mode = multi_prompts_gen_type if metadata_model_def.get("alt_prompt_inherits_prompt_paragraphs", False) else "FG"
            combine_prompt_history("alt_prompt", "enhanced_alt_prompt", alt_prompt_mode)
            models_compatible = are_model_types_compatible(model_type,current_model_type) 
            if models_compatible:
                model_type = current_model_type
            defaults = get_factory_settings(model_type)
            if PRESERVE_MEDIA_ON_SETTINGS_IMPORT and (current_settings := get_model_settings(state, model_type)) is not None:
                defaults.update({key: current_settings[key] for key in ATTACHMENT_KEYS if key not in configs and key in current_settings})
            defaults.update(configs)
            defaults["model_type"] = model_type
            prompt = configs.get("prompt", "")
                        
            if has_audio_file_extension(file_name):
                set_model_settings(state, model_type, defaults)
                gr.Info(f"Settings Loaded from Audio File with prompt '{prompt[:100]}'")
            elif has_image_file_extension(file_name):
                set_model_settings(state, model_type, defaults)
                gr.Info(f"Settings Loaded from Image with prompt '{prompt[:100]}'")
            elif has_video_file_extension(file_name) or has_mesh_file_extension(file_name):
                extracted_images = extract_and_apply_source_images(file_name, defaults)
                set_model_settings(state, model_type, defaults)
                info_msg = f"Settings Loaded from {'Video' if has_video_file_extension(file_name) else '3D Asset'} with prompt '{prompt[:100]}'"
                if extracted_images:
                    info_msg += f" + {extracted_images} source {'image' if extracted_images == 1 else 'images'} extracted"
                gr.Info(info_msg)
            
            if models_compatible:
                return str(time.time()), gr.update()
            else:
                state["ignore_save_form"] = True
                state["skip_resolution_transfer_on_model_switch"] = model_type
                return gr.update(), _model_choice_target_value(model_type)
    else:
        gr.Info(f"Please Select a File")

    return gr.update(), gr.update()
def update_loras_url_cache(lora_dir, loras_selected, return_URLs = False):
    if loras_selected is None:
        return None
    _ensure_loras_url_cache()
    new_loras_selected = []
    update = False
    for lora in loras_selected:
        if os.path.isabs(lora):
            new_loras_selected.append(lora)
        else:
            rel_path= get_lora_local_path(None, lora)
            if (lora.startswith("http:") or lora.startswith("https:")):
                url = loras_url_cache.get( lora_dir + "|" + rel_path, None)
                if url is None:
                    loras_url_cache[lora_dir + "|" + rel_path]= lora.split("|")[0]
                    update = True
            new_loras_selected.append(rel_path)

    if update:
        with open(loras_cache_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(loras_url_cache, indent=4))

    return new_loras_selected


def get_settings_from_file(state, file_path, allow_json, merge_with_defaults, switch_type_if_compatible, min_settings_version = 0, merge_loras = None, skip_validate_settings=False, merge_factory_defaults=False):
    configs = None
    any_image_or_video = False
    any_audio = False
    state["_last_settings_file_error"] = None
    file_path = getattr(file_path, "name", file_path)
    file_path = str(file_path or "")
    file_path_lower = file_path.lower()
    if file_path_lower.endswith(".json") and allow_json:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                configs = json.load(f)
        except:
            pass
    elif file_path_lower.endswith(".zip") and allow_json:
        loaded_queue, error, source_task_count = _parse_settings_zip(file_path, state, skip_validate_settings=skip_validate_settings)
        state["_last_settings_file_error"] = error
        if error is None and loaded_queue:
            configs = (loaded_queue[0].get("params", {}) or {}).copy()
            if source_task_count > 1:
                configs["_settings_bundle_task_count"] = source_task_count
    elif file_path_lower.endswith((".mp4", ".mkv", ".mov")):
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
    elif has_mesh_file_extension(file_path):
        from shared.utils.mesh_outputs import read_mesh_metadata
        try:
            configs = read_mesh_metadata(file_path)
        except:
            pass
    elif has_world_file_extension(file_path):
        from shared.utils.world_outputs import read_world_metadata
        try:
            configs = read_world_metadata(file_path)
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

    profile_merge_loras = configs.pop("merge_loras", None)
    if profile_merge_loras is not None:
        if profile_merge_loras not in ("merge before", "merge after"):
            state["_last_settings_file_error"] = "merge_loras must be 'merge before' or 'merge after'."
            return None, False, False
        if merge_loras is not None:
            merge_loras = profile_merge_loras

    current_model_type = get_state_model_type(state)
    convert_retired_model_type(configs)
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
        current_settings = get_model_settings(state, model_type)
        defaults = get_factory_settings(model_type) if merge_factory_defaults else (get_default_settings(model_type) if current_settings is None else current_settings.copy())
        if PRESERVE_MEDIA_ON_SETTINGS_IMPORT and current_settings is not None:
            defaults.update({key: current_settings[key] for key in ATTACHMENT_KEYS if key not in configs and key in current_settings})
        has_loras_without_multipliers = "activated_loras" in configs and "loras_multipliers" not in configs
        if merge_loras is not None and model_type == current_model_type:
            lora_settings = current_settings or defaults
            old_loras_selected, old_loras_multipliers = lora_settings.get("activated_loras", []), lora_settings.get("loras_multipliers", "")
        defaults.update(configs)
        if has_loras_without_multipliers:
            defaults["loras_multipliers"] = ""
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
        return gr.update(), gr.update(), None

    configs, any_video_or_image_file, any_audio = get_settings_from_file(state, file_path, True, True, True, skip_validate_settings=True, merge_factory_defaults=True)
    if configs == None:
        gr.Info("File not supported" + (f": {state.get('_last_settings_file_error')}" if state.get("_last_settings_file_error") else ""))
        return gr.update(), gr.update(), None
    if is_deepy_display_metadata(configs):
        gr.Info("Deepy helper metadata is display-only and cannot be loaded as WanGP settings")
        return gr.update(), gr.update(), None

    current_model_type = get_state_model_type(state)
    model_type = configs["model_type"]
    prompt = configs.get("prompt", "")
    is_image = configs.get("is_image", False)
    settings_bundle_task_count = configs.pop("_settings_bundle_task_count", 0)

    # Extract and apply embedded source images from video files
    extracted_images = 0
    file_path_text = str(getattr(file_path, "name", file_path) or "")
    if file_path_text.lower().endswith((".mkv", ".mov", ".mp4", ".glb")):
        extracted_images = extract_and_apply_source_images(file_path_text, configs)
    if settings_bundle_task_count > 1:
        gr.Info(f"Settings bundle contains {settings_bundle_task_count} tasks; only the first task has been extracted.")
    if any_audio:
        gr.Info(f"Settings Loaded from Audio file with prompt '{prompt[:100]}'")
    elif any_video_or_image_file:    
        info_msg = f"Settings Loaded from {'Image' if is_image else 'Video'} generated with prompt '{prompt[:100]}'"
        if extracted_images > 0:
            info_msg += f" + {extracted_images} source image(s) extracted and applied"
        gr.Info(info_msg)
    elif has_mesh_file_extension(file_path_text):
        gr.Info(f"Settings Loaded from 3D Asset with prompt '{prompt[:100]}'" + (f" + {extracted_images} source image(s) extracted and applied" if extracted_images > 0 else ""))
    else:
        gr.Info(f"Settings Loaded from Settings file with prompt '{prompt[:100]}'")

    if model_type == current_model_type:
        set_model_settings(state, current_model_type, configs)        
        return str(time.time()), gr.update(), None
    else:
        set_model_settings(state, model_type, configs)        
        state["ignore_save_form"] = True
        state["skip_resolution_transfer_on_model_switch"] = model_type
        return gr.update(), _model_choice_target_value(model_type), None

def _model_choice_target_model_type(model_type):
    return str(model_type or "").split("|", 1)[0].strip()

def _model_choice_target_value(model_type):
    model_type = _model_choice_target_model_type(model_type)
    caller = inspect.currentframe().f_back.f_code.co_name
    model_dropdowns.debug_model_selector_event("target.write", source=caller, target=model_type)
    return gr.update() if len(model_type) == 0 else f"{model_type}|{time.time()}"

def switch_to_model(model_type, open_media_tab=False):
    return _model_choice_target_value(model_type), gr.Tabs(selected="media_gen") if open_media_tab else gr.update()

def goto_model_type(state, model_type):
    model_type = _model_choice_target_model_type(model_type)
    if len(model_type) == 0:
        return gr.update(), gr.update(), gr.update(), gr.update()
    dropdowns = generate_dropdown_model_list(model_type, state)
    model_dropdowns.debug_model_selector_event("target.render", target=model_type, filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state), dropdown_value=dropdowns[2].constructor_args.get("value"))
    return *dropdowns, gr.update()

def goto_model_type_with_filter(state, model_type):
    model_type = _model_choice_target_model_type(model_type)
    if len(model_type) == 0:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    filter_update = model_output_filter.refresh_for_model_target(_get_dropdown_deps(), state, model_type)
    dropdowns = generate_dropdown_model_list(model_type, state)
    model_dropdowns.debug_model_selector_event("target.render_with_filter", target=model_type, filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state), dropdown_value=dropdowns[2].constructor_args.get("value"), filter_update=filter_update)
    return *dropdowns, gr.update(), filter_update

def change_model_from_target(state, model_type):
    model_type = _model_choice_target_model_type(model_type)
    model_dropdowns.debug_model_selector_event("target.apply", target=model_type, state_filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state))
    return change_model(state, model_type)[0]

def refresh_model_dropdowns(state):
    model_type = get_state_model_type(state)
    dropdowns = generate_dropdown_model_list(model_type, state)
    model_dropdowns.debug_model_selector_event("dropdowns.refresh", state_model=model_type, filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state), dropdown_value=dropdowns[2].constructor_args.get("value"))
    return *dropdowns, gr.update()

def reset_settings(state):
    model_type = get_state_model_type(state)
    ui_defaults = get_default_settings(model_type)
    set_model_settings(state, model_type, ui_defaults)
    gr.Info(f"Default Settings have been Restored")
    return str(time.time())


def transfer_current_resolution_to_model(state, source_model_type, target_model_type):
    if not resolution_utils.keep_resolution_on_model_switch_enabled(server_config.get("keep_resolution_on_model_switch", True)):
        return
    source_settings = get_model_settings(state, source_model_type)
    source_resolution = source_settings.get("resolution") if source_settings is not None else server_config.get("last_resolution_choice", None)
    target_model_def = get_model_def(target_model_type)
    target_resolution = resolution_utils.resolve_model_switch_resolution(source_resolution, target_model_def, server_config.get("enable_4k_resolutions", 0) == 1, target_model_def.get("vae_block_size", 16))
    if target_resolution is None:
        return
    if target_resolution != source_resolution:
        gr.Info(f"Canvas aligned from {source_resolution} to {target_resolution} for the selected model.")
    target_settings = get_model_settings(state, target_model_type)
    target_settings = get_default_settings(target_model_type) if target_settings is None else target_settings.copy()

    target_settings["resolution"] = target_resolution
    set_model_settings(state, target_model_type, target_settings)
    server_config["last_resolution_choice"] = target_resolution
    server_config["last_resolution_per_group"] = resolution_utils.remember_last_resolution(state["last_resolution_per_group"], target_resolution)

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
            world_output,
            batch_size,
            seed,
            force_fps,
            num_inference_steps,
            num_inference_steps2,
            custom_sigmas,
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
            video_guide2,
            video_guide3,
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
            audio_guide3,
            custom_guide,
            audio_source,            
            replace_voice_method,
            replace_voice_sample,
            replace_voice_sample2,
            audio_prompt_type,
            speakers_locations,
            sliding_window_size,
            sliding_window_overlap,
            sub_parallel_window_size,
            sub_parallel_window_overlap,
            sliding_window_color_correction_strength,
            sliding_window_overlap_noise,
            sliding_window_discard_last_frames,
            sliding_window_trim_first_frames,
            image_refs_relative_size,
            remove_background_images_ref,
            temporal_upsampling,
            spatial_upsampling,
            spatial_postprocessing_parameters,
            temporal_postprocessing_parameters,
            postprocessing_reference_images,
            film_grain_intensity,
            film_grain_saturation,
            postprocess_audio,
            postprocess_audio_prompt,
            postprocess_audio_neg_prompt,
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
            overrides,
            attention_sparsity,
            temperature,
            custom_setting_1,
            custom_setting_2,
            custom_setting_3,
            custom_setting_4,
            custom_setting_5,
            custom_setting_slider_1,
            custom_setting_slider_2,
            custom_setting_slider_3,
            custom_setting_slider_4,
            custom_setting_slider_5,
            custom_setting_dropdown_1,
            custom_setting_dropdown_2,
            custom_setting_dropdown_3,
            custom_setting_dropdown_4,
            custom_setting_dropdown_5,
            top_p,
            top_k,
            self_refiner_setting,
            self_refiner_plan,            
            self_refiner_f_uncertainty,
            self_refiner_certain_percentage,
            output_filename,
            config,
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
            image_mask = image_editor_layer_to_rgb_mask(image_mask_guide["layers"][0])
        elif image_mask is None and image_guide is not None:
            image_mask = Image.new("RGB", image_guide.size, (0, 0, 0))
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
        service = service_for(state)
        if service is not None:
            service.record_model_form(model_type, cleaned_inputs)
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
            task_data = queue[task_index]
            if _is_edit_task_params(task_data.get("params", {})):
                if action == "edit": gr.Info("Post-processing tasks cannot be edited.")
                return update_queue_data(queue), gr.Tabs(), gr.update(), gr.update()

            state["editing_task_id"] = task_id

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

def change_model(state, model_choice, *, refresh_header=True):
    if model_choice == None:
        return
    previous_model_type = get_state_model_type(state)
    service = service_for(state)
    if service is not None and previous_model_type != model_choice and state.get("skip_resolution_transfer_on_model_switch") != model_choice:
        recorded = service.load_model_form(model_choice)
        if recorded is not None:
            set_model_settings(state, model_choice, recorded)
    model_filename = get_model_filename(model_choice, transformer_quantization, transformer_dtype_policy)
    with config_lock:
        last_model_per_family = state["last_model_per_family"]
        last_model_per_family[get_model_family(model_choice, for_ui= True)] = model_choice
        server_config["last_model_per_family"] = last_model_per_family

        last_model_per_type = state["last_model_per_type"]
        last_model_per_type[get_base_model_type(model_choice)] = model_choice
        server_config["last_model_per_type"] = last_model_per_type

        server_config["last_model_type"] = model_choice
        server_config["resolution_picker_mode"] = state.get("resolution_picker_mode", server_config.get("resolution_picker_mode", "presets"))
        model_dropdowns.store_model_output_filter(server_config, state, model_choice)
        model_dropdowns.debug_model_selector_event("model.saved", model=model_choice, filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state), family=get_model_family(model_choice, for_ui=True), base=get_parent_model_type(model_choice))
        skip_resolution_transfer = state.pop("skip_resolution_transfer_on_model_switch", None) == model_choice
        if previous_model_type != model_choice and not skip_resolution_transfer:
            transfer_current_resolution_to_model(state, previous_model_type, model_choice)

        write_config_deferred(server_config, server_config_filename)

    state["model_type"] = model_choice
    if hasattr(app, "plugin_manager"):
        app.plugin_manager.notify_model_change(state, model_choice)
    if not refresh_header:
        return generate_model_description(model_choice), None
    model_settings = get_model_settings(state, model_choice) or get_default_settings(model_choice)
    description, header = generate_header(model_choice, compile=compile, attention_mode=attention_mode, overrides=model_settings.get("overrides", ""), guidance_phases=model_settings.get("guidance_phases", 1), video_prompt_type=model_settings.get("video_prompt_type", ""))
    
    return description, header

def get_current_model_settings(state):
    model_type = get_state_model_type(state)
    ui_defaults = get_model_settings(state, model_type)
    if ui_defaults == None:
        ui_defaults = get_default_settings(model_type)
        set_model_settings(state, model_type, ui_defaults)
    return ui_defaults 

def custom_setting_input_name(name):
    return str(name).startswith("custom_setting_")

def build_save_inputs(input_names, locals_dict, state_value, plugin_data_value, custom_setting_extra_inputs):
    inputs = []
    custom_settings_inserted = False
    for key in input_names:
        if custom_setting_input_name(key):
            if not custom_settings_inserted:
                inputs += custom_setting_extra_inputs
                custom_settings_inserted = True
            continue
        inputs.append(state_value if key == "state" else locals_dict[key])
    return inputs + [state_value, plugin_data_value]

def build_form_refresh_outputs(input_names, locals_dict, state_value, plugin_data_value, extra_inputs):
    extra_input_ids = {id(component) for component in extra_inputs}
    outputs = []
    for key in input_names:
        if custom_setting_input_name(key):
            continue
        component = state_value if key == "state" else locals_dict[key]
        if id(component) not in extra_input_ids:
            outputs.append(component)
    return outputs + [state_value, plugin_data_value] + extra_inputs

def fill_inputs(state):
    ui_defaults = get_current_model_settings(state)
 
    return generate_media_tab(update_form = True, state_dict = state, ui_defaults = ui_defaults)

def get_loaded_model_label():
    return f"Model {get_model_name(transformer_type)} is Loaded" if wan_model is not None else "No Model is Loaded"


def render_idle_generation_status(state):
    progress = WangpProgress()
    progress.status(get_gen_info(state).get("status") or "Ready")
    return progress.render(active=False, model_name=get_loaded_model_label())


def preload_model_when_switching(state):
    global reload_needed, wan_model, offloadobj
    if "S" in preload_model_policy:
        model_type = get_state_model_type(state) 
        if  model_type !=  transformer_type:
            gen = get_gen_info(state)
            set_generation_state(gen, model_preloading=True)
            try:
                release_model()
                progress = WangpProgress()
                progress.status("Loading...")
                yield progress.render(model_name=get_model_name(model_type))
                override_values = generation_overrides.parse_overrides(get_current_model_settings(state).get("overrides", ""))
                wan_model, offloadobj = load_models(
                    model_type,
                    override_values.get("memory_profile", -1),
                    output_type=get_profile_type_for_model(model_type, 0),
                    overrides=override_values,
                )
                reload_needed = False
            finally:
                set_generation_state(gen, model_preloading=False)
            yield render_idle_generation_status(state)
        return   
    return gr.Text()

def unload_model_if_needed(state):
    if service_for(state) is None:
        _unload_model_if_needed(state)


def _unload_model_if_needed(state):
    global wan_model
    if "U" in preload_model_policy:
        if wan_model != None:
            release_model()

def request_reload_if_loaded(model_type):
    global reload_needed
    if model_type == transformer_type:
        reload_needed = True

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

def force_control_video_trim_visible(model_def, image_mode, video_prompt_type, audio_prompt_type):
    return image_mode == 0 and not model_def.get("control_video_trim", False) and not model_def.get("control_video_trim_disabled", False) and ("V" in video_prompt_type or any_letters(audio_prompt_type, "ABX"))

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

def normalize_phase_2_tiling_selection(model_def, guidance_phases, video_prompt_type):
    return resolution_utils.normalize_phase_2_tiling_selection(model_def, guidance_phases, video_prompt_type)

def get_prompt_enhancer_letters_filter(prompt_enhancer_def, prompt_enhancer_choices):
    if prompt_enhancer_def is not None and len(prompt_enhancer_def.get("letters_filter", "")): return prompt_enhancer_def["letters_filter"]
    letters_filter = ""
    for _label, value in prompt_enhancer_choices:
        letters_filter = add_to_sequence(letters_filter, str(value or ""))
    return letters_filter

def get_prompt_enhancer_choices(model_def, audio_only=False, image_mode=0, include_disabled=True, video_prompt_type=None, inputs=None):
    choices = [("Disabled", "")] if include_disabled else []
    prompt_enhancer_default = ""
    default_labels = prompt_enhancer_labels.default_labels(model_def, image_mode, video_prompt_type, inputs)
    prompt_enhancer_def = prompt_enhancer_labels.resolve_definition(model_def, image_mode, video_prompt_type, inputs)
    if prompt_enhancer_def is not None:
        labels = prompt_enhancer_def["labels"]
        prompt_enhancer_default = prompt_enhancer_def["default"]
        mode_letter = "P" if image_mode > 0 else "V"
        choices += [(label, del_in_sequence(key, "VP")) for key, label in labels.items() if not (modes := filter_letters(key, "VP")) or mode_letter in modes]
    else:
        selection = model_def.get("prompt_enhancer_choices_allowed", ["T"] if audio_only else ["T", "TI"])
        choices += [(default_labels.get(selection_value, selection_value), selection_value) for selection_value in selection]
    return choices, prompt_enhancer_default, prompt_enhancer_def


def refresh_prompt_enhancer_labels(state, metadata, *, tab_id="generate"):
    # The first value scopes browser deduplication to this model selection.
    inputs = dict(zip(prompt_enhancer_labels.INPUT_KEYS, metadata[1:]))
    inputs["image_refs"] = [True] * min(inputs["image_refs"], 2)
    model_def = get_model_def(get_state_model_type(state))
    choices, _, _ = get_prompt_enhancer_choices(model_def, model_def.get("audio_only", False), inputs["image_mode"], include_disabled=server_config.get("enhancer_mode", 0) != 1, video_prompt_type=inputs["video_prompt_type"], inputs=inputs)
    labels = gr.skip() if state.get("enhancer_label_choices_" + tab_id) == choices else gr.update(choices=choices)
    state["enhancer_label_choices_" + tab_id] = choices
    return labels


def build_prompt_enhancer_value(prompt_enhancer_mode, prompt_enhancer_think):
    return prompt_enhancer_chaining.with_thinking(prompt_enhancer_mode, prompt_enhancer_think)

def refresh_remove_background_sound(state, audio_prompt_type, remove_background_sound):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "V")
    if remove_background_sound:
        audio_prompt_type = add_to_sequence(audio_prompt_type, "V")
    return audio_prompt_type

def refresh_normalize_audio_volumes(state, audio_prompt_type, normalize_audio_volumes):
    audio_prompt_type = del_in_sequence(audio_prompt_type, "N")
    if normalize_audio_volumes:
        audio_prompt_type = add_to_sequence(audio_prompt_type, "N")
    return audio_prompt_type

def get_audio_prompt_type_custom_option_def(model_def):
    option_def = (model_def or {}).get("audio_prompt_type_custom_option", None)
    if isinstance(option_def, dict):
        flag = str(option_def.get("flag", "") or "").strip().upper()
        label = str(option_def.get("label", "") or "").strip()
    elif isinstance(option_def, str) and len(option_def.strip()) > 0:
        flag = ""
        label = option_def.strip()
    else:
        flag = ""
        label = ""
    return label or "Audio Option", flag[:1]

def refresh_audio_prompt_type_custom_option(state, audio_prompt_type, custom_audio_option):
    model_def = get_model_def(get_state_model_type(state))
    _, custom_audio_option_flag = get_audio_prompt_type_custom_option_def(model_def)
    if len(custom_audio_option_flag) == 0:
        return audio_prompt_type
    audio_prompt_type = del_in_sequence(audio_prompt_type, custom_audio_option_flag)
    if custom_audio_option:
        audio_prompt_type = add_to_sequence(audio_prompt_type, custom_audio_option_flag)
    return audio_prompt_type

def refresh_audio_prompt_type_sources(state, audio_prompt_type, audio_prompt_type_sources, video_prompt_type, image_mode):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    letters_filter=  "XCPABOKF"
    audio_prompt_type_sources_def = model_def.get("audio_prompt_type_sources", None)
    if audio_prompt_type_sources_def is not None:
        letters_filter = audio_prompt_type_sources_def.get("letters_filter", letters_filter)
    audio_prompt_type = del_in_sequence(audio_prompt_type, letters_filter)
    audio_prompt_type = add_to_sequence(audio_prompt_type, audio_prompt_type_sources)
    audio_only = model_def.get("audio_only", False) if model_def is not None else False
    speakers_visible = ("B" in audio_prompt_type or "X" in audio_prompt_type) and not audio_only and (model_def.get("multitalk_class", False) or model_def.get("speaker_locations", False))
    remove_background_visible = any_letters(audio_prompt_type, "ABXK")
    normalize_audio_visible = all_letters(audio_prompt_type, "AB")
    _, custom_audio_option_flag = get_audio_prompt_type_custom_option_def(model_def)
    custom_audio_option_visible = len(custom_audio_option_flag) > 0
    audio_options_visible = remove_background_visible or normalize_audio_visible or custom_audio_option_visible
    if not normalize_audio_visible:
        audio_prompt_type = del_in_sequence(audio_prompt_type, "N")
    if not custom_audio_option_visible and len(custom_audio_option_flag) > 0:
        audio_prompt_type = del_in_sequence(audio_prompt_type, custom_audio_option_flag)
    return (
        audio_prompt_type,
        gr.update(visible="A" in audio_prompt_type),
        gr.update(visible="B" in audio_prompt_type),
        gr.update(visible="D" in audio_prompt_type),
        gr.update(visible=speakers_visible),
        gr.update(visible=remove_background_visible),
        gr.update(visible=normalize_audio_visible),
        gr.update(visible=custom_audio_option_visible, value=custom_audio_option_flag in audio_prompt_type),
        gr.update(visible=audio_options_visible),
        gr.update(visible=any_letters(audio_prompt_type, "AB")),
        gr.update(visible=force_control_video_trim_visible(model_def, image_mode, video_prompt_type, audio_prompt_type)),
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

def video_guides_shown(model_def, image_mode):
    """Image modes show a control image in place of the control videos, except image / 3D models that read reference videos."""
    return image_mode == 0 or model_def.get("image_outputs", False) and model_def.get("reference_video_enabled", False)

def switch_image_guide_editor(model_def, image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value ):
    if video_guides_shown(model_def, image_mode): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    mask_in_old = video_mask_area_visible(old_video_prompt_type)
    mask_in_new = video_mask_area_visible(video_prompt_type)
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
    visible = video_mask_area_visible(video_prompt_type)
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    image_outputs =  image_mode > 0
    mask_strength_always_enabled = model_def.get("mask_strength_always_enabled", False)  
    image_mask_guide, image_guide, image_mask = switch_image_guide_editor(model_def, image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value )
    magic_image_btn, magic_video_btn = MagicMaskUI.button_updates(image_mode, video_prompt_type)
    return video_prompt_type, gr.update(visible= visible and not image_outputs), image_mask_guide, image_guide, image_mask, gr.update(visible= visible ) , gr.update(visible= visible and (mask_strength_always_enabled or "G" in video_prompt_type )  ), magic_image_btn, magic_video_btn

def refresh_video_prompt_type_alignment(state, video_prompt_type, video_prompt_type_video_guide):
    video_prompt_type = del_in_sequence(video_prompt_type, "T")
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide)
    return video_prompt_type


def refresh_video_prompt_type_video_guide(state, filter_type, video_prompt_type, video_prompt_type_video_guide,  image_mode, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value, image_prompt_type, audio_prompt_type):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    old_video_prompt_type = video_prompt_type
    if filter_type == "alt":
        guide_custom_choices = get_guide_custom_choices(model_def, image_mode) or {}
        letter_filter = guide_custom_choices.get("letters_filter","")
    else:
        letter_filter = all_guide_processes
    video_prompt_type = del_in_sequence(video_prompt_type, letter_filter)
    video_prompt_type = add_to_sequence(video_prompt_type, video_prompt_type_video_guide)
    mask_controls_visible = video_mask_controls_visible(video_prompt_type)
    if not mask_controls_visible:
        video_prompt_type = del_in_sequence(video_prompt_type, "XYZWNA")
    visible = "V" in video_prompt_type
    any_outpainting= image_mode in model_def.get("video_guide_outpainting", [])
    mask_visible = video_mask_area_visible(video_prompt_type)
    image_outputs =  image_mode > 0
    keep_frames_video_guide_visible = not image_outputs and visible and not model_def.get("keep_frames_video_guide_not_supported", False)
    image_mask_guide, image_guide, image_mask = switch_image_guide_editor(model_def, image_mode, old_video_prompt_type , video_prompt_type, old_image_mask_guide_value, old_image_guide_value, old_image_mask_value )
    mask_preprocessing = model_def.get("mask_preprocessing", None)
    mask_selector_visible = video_mask_dropdown_visible(mask_preprocessing, video_prompt_type)
    mask_selector_update = gr.update(visible=mask_selector_visible) if mask_controls_visible else gr.update(value="", visible=False)
    ref_images_visible = "I" in video_prompt_type
    remove_background_images_ref_visible = ref_images_visible and not model_def.get("no_background_removal", False)
    custom_options = custom_checkbox = False 
    custom_video_selection = model_def.get("custom_video_selection", None)
    if custom_video_selection is not None:
        custom_trigger =  custom_video_selection.get("trigger","")
        if len(custom_trigger) == 0 or custom_trigger in video_prompt_type:
            custom_options = True
            custom_checkbox = custom_video_selection.get("type","") == "checkbox"
    mask_strength_always_enabled = model_def.get("mask_strength_always_enabled", False)  
    magic_image_btn, magic_video_btn = MagicMaskUI.button_updates(image_mode, video_prompt_type)
    video_guides = video_guides_shown(model_def, image_mode)
    return video_prompt_type, gr.update(visible=visible and video_guides), gr.update(visible="+" in video_prompt_type and video_guides), gr.update(visible="*" in video_prompt_type and video_guides), image_guide, gr.update(visible=keep_frames_video_guide_visible), gr.update(visible=visible and "G" in video_prompt_type), gr.update(visible=mask_visible and (mask_strength_always_enabled or "G" in video_prompt_type)), gr.update(visible=(visible or injected_frames_positions_visible(video_prompt_type) or "K" in video_prompt_type) and any_outpainting), mask_selector_update, gr.update(visible=mask_visible and not image_outputs), image_mask, image_mask_guide, gr.update(visible=mask_visible), gr.update(visible=ref_images_visible), gr.update(visible=remove_background_images_ref_visible), gr.update(visible=injected_frames_positions_visible(video_prompt_type)), gr.update(visible=custom_options and not custom_checkbox), gr.update(visible=custom_options and custom_checkbox), gr.update(visible=input_video_strength_visible(model_def, image_prompt_type, video_prompt_type)), magic_image_btn, magic_video_btn, gr.update(visible=force_control_video_trim_visible(model_def, image_mode, video_prompt_type, audio_prompt_type))

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
    placeholder = '<div style="text-align: center; color: var(--body-text-color-subdued); padding: 16px 0;">Preview not yet Available</div>'
    preview_image = gen.get("preview", None)
    if isinstance(preview_image, VideoPreview):
        return preview_image.to_html()
    if preview_image is None:
        return placeholder
    
    preview_base64 = pil_to_base64_uri(preview_image, format="jpeg", quality=85)
    if preview_base64 is None:
        return placeholder

    html_content = f"""
    <div style="display: flex; justify-content: center; align-items: center; cursor: pointer;" onclick="showImageModal('preview_0')">
        <img src="{preview_base64}"
             style="display: block; height: auto; max-height: 200px; max-width: 100%; object-fit: contain;"
             alt="Preview">
    </div>
    """
    return html_content

def init_process_queue_if_any(state):                
    gen = get_gen_info(state)
    if bool(gen.get("queue",[])):
        state["validate_success"] = 1
        return prepare_generate_media(state)
    else:
        return gr.Button(visible=True), gr.Button(visible=False), gr.Column(visible=False), gr.Button(visible=False), gr.Button(visible=False)

def get_modal_image(image_base64, label):
    return f"""
    <div class="modal-flex-container" onclick="closeImageModal()">
        <div class="modal-content-wrapper" onclick="event.stopPropagation()">
            <div class="modal-close-btn" onclick="closeImageModal()">×</div>
            <div class="modal-label">{label}</div>
            <div class="modal-image-viewport"><img src="{image_base64}" class="modal-image" alt="{label}" onload="window.WanGPImageZoom.mount(this)"></div>
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
            if isinstance(preview_image, VideoPreview):
                html_content = '<div class="modal-flex-container" onclick="closeImageModal()"><div class="modal-content-wrapper" onclick="event.stopPropagation()"><div class="modal-close-btn" onclick="closeImageModal()">×</div><div class="modal-label">Preview</div>' + preview_image.to_html(modal=True) + '</div></div>'
                return gr.HTML(value=html_content), gr.Column(visible=True)
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

def get_prompt_infos(model_def):
    return field_help.get_model_prompt_help(model_def)

PROMPT_ADVANCED_ELEM_ID = "wangp-prompt-advanced"
PROMPT_WIZARD_ELEM_ID = "wangp-prompt-wizard"
RESOLUTION_ELEM_ID = "wangp-resolution"
PROMPT_TOOLS_ATTACH_JS = "() => { requestAnimationFrame(() => requestAnimationFrame(() => { window.wangpPromptTools?.attach?.(document); window.wangpRefreshInputModes(); window.wangpRefreshSettingsTools(); window.wangpRefreshActionBars(); window.__wangpMagicMaskNS.scheduleMount(); })); }"

def get_prompt_helper_popup_id(model_type, prompt_id):
    safe_model = re.sub(r"[^A-Za-z0-9_-]", "-", str(model_type or "model")).strip("-").lower()
    safe_prompt = re.sub(r"[^A-Za-z0-9_-]", "-", str(prompt_id or "prompt")).strip("-").lower()
    return f"wangp-prompt-helper-{safe_model}-{safe_prompt}"

def get_prompt_helper_html(model_type, model_def, prompt_id, elem_id):
    try:
        model_handler = get_model_handler(model_type)
    except Exception:
        return None, ""
    render_prompt_helper = getattr(model_handler, "render_prompt_helper", None)
    if render_prompt_helper is None:
        return None, ""
    popup_id = get_prompt_helper_popup_id(model_type, prompt_id)
    return popup_id, render_prompt_helper(model_type, model_def, prompt_id, popup_id, elem_id, RESOLUTION_ELEM_ID)

def render_prompt_info_label(label, model_type, model_def, prompt_id):
    elem_id = PROMPT_WIZARD_ELEM_ID if prompt_id == "wizard" else PROMPT_ADVANCED_ELEM_ID
    helper_popup_id, helper_html = get_prompt_helper_html(model_type, model_def, prompt_id, elem_id)
    return field_help.render_model_prompt_tools(label, elem_id, model_type, model_def, prompt_id, helper_popup_id=helper_popup_id if helper_html else None, helper_title="Prompt Helper") + helper_html

def get_image_end_label(multi_prompts_gen_type):
    return "Images as ending points for each new Window of the same Video Generation" if "W" in multi_prompts_gen_type else "Images as ending points for new Videos in the Generation Queue"

def refresh_prompt_labels(state, multi_prompts_gen_type, image_mode):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    prompt_label, wizard_prompt_label =  get_prompt_labels(multi_prompts_gen_type, model_def, image_mode > 0, model_def.get("audio_only", False))
    prompt_info_html = render_prompt_info_label(prompt_label, model_type, model_def, "advanced")
    wizard_prompt_info_html = render_prompt_info_label(wizard_prompt_label, model_type, model_def, "wizard")
    return (
        gr.update(label=prompt_label),
        gr.update(label=wizard_prompt_label),
        gr.update(label=get_image_end_label(multi_prompts_gen_type)),
        gr.update(value=prompt_info_html, visible=True),
        gr.update(value=wizard_prompt_info_html, visible=True),
    )

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

def get_resolution_choices(current_resolution_choice, model_def=None):
    if model_def is None:
        model_def = {}
    elif isinstance(model_def, list):
        model_def = {"resolutions": model_def}
    return resolution_utils.resolve_resolution_choices(current_resolution_choice, model_def, server_config.get("enable_4k_resolutions", 0) == 1, model_def.get("vae_block_size", 16))

def change_resolution_group(state, selected_group, current_resolution):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    last_resolution_per_group = state["last_resolution_per_group"]
    resolution_choices, _ = get_resolution_choices(last_resolution_per_group.get(selected_group, current_resolution), model_def)
    group_resolution_choices = resolution_utils.group_choices(resolution_choices, selected_group)
    if len(group_resolution_choices) == 0:
        return gr.update(choices=[], value=None)

    last_resolution = last_resolution_per_group.get(selected_group, "")
    if len(last_resolution) == 0 or not any( [last_resolution == resolution[1] for resolution in group_resolution_choices]):
        last_resolution = group_resolution_choices[0][1]
    return gr.update(choices= group_resolution_choices, value= last_resolution ) 
    


def record_last_resolution(state, resolution):

    if not resolution_utils.is_resolution_value(resolution or ""):
        return
    with config_lock:
        update_config(server_config, server_config_filename, {"last_resolution_choice": resolution, "last_resolution_per_group": resolution_utils.remember_last_resolution(state["last_resolution_per_group"], resolution), "resolution_picker_mode": state.get("resolution_picker_mode", server_config.get("resolution_picker_mode", "presets"))})


def resolution_mp_label(resolution):
    width, height = resolution_utils.parse_resolution(resolution)
    return f"Resolution Budget (MP) · {width}×{height}"


def sync_resolution_mp(state, resolution):
    if resolution is None:
        return gr.update(), gr.update()
    model_def = get_model_def(get_state_model_type(state))
    choices, _ = get_resolution_choices(resolution, model_def)
    values = resolution_utils.megapixel_controls(resolution, choices, model_def, server_config.get("enable_4k_resolutions", 0) == 1)
    return gr.update(choices=values["ratios"], value=values["ratio"]), gr.update(value=values["value"], minimum=values["minimum"], maximum=values["maximum"], label=resolution_mp_label(resolution))


def change_resolution_picker(state):
    if get_model_def(get_state_model_type(state)).get("lock_resolutions"):
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    mode = "presets" if state.get("resolution_picker_mode", "presets") == "mp" else "mp"
    state["resolution_picker_mode"] = mode
    return gr.update(value="Switch to Presets" if mode == "mp" else "Switch to Megapixels"), gr.update(visible=mode == "presets"), gr.update(visible=mode == "mp")


def preview_resolution_mp(state, megapixels, ratio):
    model_def = get_model_def(get_state_model_type(state))
    resolution = resolution_utils.megapixels_to_resolution(megapixels, ratio, model_def.get("vae_block_size", 16), model_def)
    return gr.update(label=resolution_mp_label(resolution))


def commit_resolution_mp(state, megapixels, ratio):
    model_def = get_model_def(get_state_model_type(state))
    if model_def.get("lock_resolutions"):
        return gr.update(), gr.update(), gr.update()
    resolution = resolution_utils.megapixels_to_resolution(megapixels, ratio, model_def.get("vae_block_size", 16), model_def)
    choices, resolution = get_resolution_choices(resolution, model_def)
    groups, selected, group = resolution_utils.group_resolution_choices(choices, resolution)
    values = resolution_utils.megapixel_controls(resolution, choices, model_def, server_config.get("enable_4k_resolutions", 0) == 1)
    return gr.update(choices=selected, value=resolution), gr.update(choices=[(resolution_utils.GROUP_LABELS.get(item, item), item) for item in groups], value=group), gr.update(value=values["value"], minimum=values["minimum"], maximum=values["maximum"], label=resolution_mp_label(resolution))


def get_max_frames(nb):
    multiplier = max(1, int(server_config.get("max_frames_multiplier", 1)))
    return (nb - 1) * multiplier + 1


def get_max_duration(seconds):
    multiplier = max(1, int(server_config.get("max_frames_multiplier", 1)))
    return seconds * multiplier


def change_guidance_phases(state, guidance_phases, video_prompt_type):
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    tiled_phase_2 = model_def.get("phase_2_spatial_tiling", False) and guidance_phases == PHASE_2_TILING_GUIDANCE_VALUE
    guidance_phases = 2 if tiled_phase_2 else guidance_phases
    video_prompt_type = add_to_sequence(video_prompt_type, PHASE_2_TILING_VIDEO_PROMPT_FLAG) if tiled_phase_2 else del_in_sequence(video_prompt_type, PHASE_2_TILING_VIDEO_PROMPT_FLAG)
    visible_phases = model_def.get("visible_phases", guidance_phases) 
    multiple_submodels = model_def.get("multiple_submodels", False)
    switch_threshold_def = extra_settings.get_def("switch_threshold", model_def, guidance_phases=guidance_phases)
    phase_2_steps_def = extra_settings.get_def("num_inference_steps2", model_def, guidance_phases=guidance_phases)
    phase_controls_visible = guidance_phases >= 2 and extra_settings.get_container_def("guidance_phases_row", model_def).visible
    return gr.update(visible= guidance_phases >=3 and visible_phases >=3 and multiple_submodels) , gr.update(visible=phase_controls_visible), gr.update(visible=phase_controls_visible and switch_threshold_def.visible, label=switch_threshold_def.label), gr.update(visible= guidance_phases >=3 and visible_phases >=3), gr.update(visible= guidance_phases >=2 and visible_phases >=2), gr.update(visible= guidance_phases >=3 and visible_phases >=3), video_prompt_type, gr.update(visible=phase_2_steps_def.visible)


memory_profile_choices= [   ("Profile 0, AutoRAM_AutoVRAM: Profile 4 that reserves RAM only for the model running: the parts sent to the GPU at each step reach it at full speed from Reserved RAM, which passes to the next model when you switch. Fast without the RAM to keep every model reserved. By default its VRAM Preload keeps as much of each model in VRAM as fits", 0),
                            ("Profile 1, HighRAM_HighVRAM: Profile 2 with Dynamic Preload. All models kept in Reserved RAM; as many layers as fit stay in VRAM, the rest are streamed", 1),
                            ("Profile 2, HighRAM_LowVRAM: all models kept in Reserved RAM, sent to the GPU part by part. Runs models larger than your VRAM, leaves VRAM for long videos or large images and switches models fast, needs a lot of RAM", 2),
                            ("Profile 3, LowRAM_HighVRAM (Recommended for Audio): Profile 4 with Dynamic Preload. Only the main models kept in Reserved RAM; as many layers as fit stay in VRAM, the rest are streamed", 3),
                            ("Profile 3+, VeryLowRAM_HighVRAM: Profile 3 without any Reserved RAM. Dynamic Preload keeps as much of each model in VRAM as fits; models load more slowly", 3.5),
                            ("Profile 4, LowRAM_LowVRAM (Recommended): only the main models kept in Reserved RAM, sent to the GPU part by part. The most versatile: runs models larger than your VRAM and leaves VRAM for long videos or large images", 4),
                            ("Profile 4+, LowRAM_LowVRAM+: Profile 4 sending one part at a time. Saves up to about 1 GB of VRAM, slightly slower", 4.5),
                            ("Profile 5, VerylowRAM_LowVRAM (Fail safe): almost no Reserved RAM, all models sent to the GPU part by part. For PCs short of RAM and VRAM, slower with short steps such as images", 5)]

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
    (f'sage2/sage2++{check_attn("sage2")}: ~40% faster, (recommended, extra optims when used) requires manual install', "sage2"),
] + ([(f'radial{check_attn("radial")}: Experimental, may be faster, requires manual install', "radial")] if args.betatest else []) + [
    (f'sage3{check_attn("sage3")}: >50% faster, may have quality trade-offs, requires manual install', "sage3"),
]

def refresh_attention_header(state, overrides, guidance_phases=1, video_prompt_type=""):
    return generate_header(get_state_model_type(state), compile, attention_mode, overrides, guidance_phases, video_prompt_type)[1]

def serialize_overrides(*values):
    return generation_overrides.format_overrides(dict(zip(generation_overrides.OVERRIDE_DEFAULTS, values)))

def get_attention_override_choices(model_def, selected_attention):
    choices = [("Default Attention Mode", "")] + attention_modes_choices if model_def.get("default_attention_modes_supported", True) else []
    for mode, definition in model_def.get("custom_attention_modes", {}).items():
        status = ATTENTION_MODE_AVAILABILITY.get(mode, {})
        installed = definition.get("installed", status.get("installed", True))
        supported = definition.get("supported", status.get("supported", installed))
        status_label = "AVAILABLE" if supported else ("NOT SUPPORTED" if installed else "NOT INSTALLED")
        choices.append((f"{mode} ({status_label}): {definition['label']}", mode))
    return selected_attention if any(value == selected_attention for _, value in choices) else choices[0][1], choices

def get_overrides_error(model_def, overrides):
    try:
        values = generation_overrides.parse_overrides(overrides)
    except ValueError as error:
        return str(error)
    choices = {"memory_profile": memory_profile_choices, "compilation": compile_choices, "attention_mode": get_attention_override_choices(model_def, "")[1],
               "transformer_quantization": transformer_quantization_choices, "text_encoder_quantization": text_encoder_quantization_choices}
    for name, value in values.items():
        allowed = [choice for _, choice in choices[name]]
        if value != generation_overrides.OVERRIDE_DEFAULTS[name] and value not in allowed:
            return f"Invalid {name} override {value!r}: expected one of {', '.join(map(repr, allowed))}"
    return ""

def get_quantization_overridable(model_type, model_def):
    """Whether choosing another transformer / text encoder quantization selects other weights for this model."""
    transformer_files = {get_model_filename(model_type, quantization, transformer_dtype_policy, model_def=model_def) for _, quantization in transformer_quantization_choices}
    text_encoder_URLs = get_model_recursive_prop(model_type, "text_encoder_URLs", return_list=True, model_def=model_def)
    text_encoder_files = {get_model_filename(model_type, quantization, transformer_dtype_policy, URLs=text_encoder_URLs) for _, quantization in text_encoder_quantization_choices} if text_encoder_URLs else ()
    return len(transformer_files) > 1 or model_def.get("auto_quantize", False), len(text_encoder_files) > 1

def refresh_attention_sparsity(state, override_attention):
    modes = get_model_def(get_state_model_type(state)).get("custom_attention_modes", {})
    return gr.update(visible=modes.get(override_attention, {}).get("supports_sparsity", False))

def detect_auto_save_form(state, evt:gr.SelectData):
    last_tab_id = state.get("last_tab_id", 0)
    state["last_tab_id"] = new_tab_id = evt.index
    if new_tab_id > 0 and last_tab_id == 0:
        return get_unique_id()
    else:
        return gr.update()

def compute_video_length_label(fps, current_video_length, video_length_locked = None, world = False):
    # world models: the frames are the world's history, which explorations and continuations start from
    if fps is None:
        ret = "World History Duration" if world else "Number of frames"
    elif world:
        ret = f"World History Duration ({fps} frames = 1s): {(current_video_length / fps):.1f}s"
    else:
        ret = f"Number of frames ({fps} frames = 1s), current duration: {(current_video_length / fps):.1f}s"
    if video_length_locked is not None:
        ret += ", locked"
    return ret
def refresh_video_length_label(state, current_video_length, force_fps, video_guide, video_source):
    base_model_type = get_base_model_type(get_state_model_type(state))
    computed_fps = get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    return gr.update(label= compute_video_length_label(computed_fps, current_video_length, world=get_model_def(get_state_model_type(state)).get("world_outputs", False)))

def video_input_label_with_info(base_label, video_path):
    if video_path is None:
        return base_label
    try:
        fps, width, height, frames_count = get_video_info(video_path)
        fps = float(fps or 0)
        width = int(width or 0)
        height = int(height or 0)
        frames_count = int(frames_count or 0)
    except Exception:
        return base_label
    if width <= 0 or height <= 0:
        return base_label
    info = [f"{width}x{height}"]
    if fps > 0:
        info += [f"{fps:g} fps"]
    if frames_count > 0:
        info += [f"{frames_count} {'frame' if frames_count == 1 else 'frames'}"]
    return f"{base_label} ({', '.join(info)})"

def refresh_video_input_label(video_path, base_label):
    return gr.update(label=video_input_label_with_info(base_label, video_path))

def change_force_control_video_trim(state, video_prompt_type, force_control_video_trim):
    video_prompt_type = del_in_sequence( video_prompt_type, "|")
    if force_control_video_trim == 1: video_prompt_type = add_to_sequence(video_prompt_type, "|")
    return video_prompt_type

def update_value(prompt_type_value, sub_value, letter_filter):                        
    return del_in_sequence(prompt_type_value, letter_filter) + filter_letters(sub_value, letter_filter)

def get_default_value(choices, current_value, default_value = None):
    for label, value in choices:
        if value == current_value:
            return current_value        
    return choices[0][1] if default_value is None else default_value

def download_lora(state, lora_url, progress=WangpProgress(track_tqdm=True),):
    if lora_url is None or not lora_url.startswith("http"):
        gr.Info("Please provide a URL for a Lora to Download")
        return gr.update()
    model_type = get_state_model_type(state)
    lora_short_name = os.path.basename(lora_url)
    lora_dir = get_lora_dir(model_type)
    local_path = os.path.join(lora_dir, lora_short_name)
    try:
        progress.status(f"Downloading LoRA {lora_short_name}…")
        download_file(lora_url, local_path, gen={"abort": False, "download_progress_callback": progress.set_download})
    except Exception as e:
        gr.Info(f"Error downloading Lora {lora_short_name}: {e}")
        return gr.update()
    update_loras_url_cache(lora_dir, [lora_url])
    gr.Info(f"Lora {lora_short_name} has been succesfully downloaded")
    return ""
    

def set_video_info_tab(evt:gr.SelectData):
    tab_ids = ("video_info", "audio_postprocessing", "post_processing", "audio_remuxing", "video_add")
    return tab_ids[evt.index] if isinstance(evt.index, int) and 0 <= evt.index < len(tab_ids) else str(evt.index or "video_info")

# Late post processing tab of each gallery that has one
GALLERY_POSTPROCESSING_TABS = {VIDEO.id: "post_processing", "audio": "audio_postprocessing", MESH.id: "post_processing"}

def set_gallery_tab(state, video_info_tab, evt:gr.SelectData):
    # The clicked tab gives the gallery: with a shared gallery (hybrid mode), the browser's selection message updates the shared
    # state concurrently and may arrive after this event.
    gen = get_gen_info(state)
    index = evt.index
    source_id = GALLERY_SOURCES[index].id
    if service_for(state) is None:
        gen["current_gallery_source"] = source_id
        if index != 0:
            gen["selected_video_time"] = None
    # Viewing another gallery's post processing switches to this gallery's one (or to the media information)
    target_tab = GALLERY_POSTPROCESSING_TABS.get(source_id, "video_info")
    switch_tab = video_info_tab in GALLERY_POSTPROCESSING_TABS.values() and video_info_tab != target_tab
    return index, source_id, gr.Tabs(selected=target_tab) if switch_tab else gr.update(), target_tab if switch_tab else video_info_tab


def get_processed_queue(gen):
    """Files and settings of every gallery: {gallery source id: (files, settings)}."""
    gen = workspace_storage.recording_gallery(gen)
    with lock:
        return {source.id: (gen.setdefault(source.files_key, []), gen.setdefault(source.settings_key, [])) for source in GALLERY_SOURCES}

def get_current_video_gallery(state):
    gen = get_gen_info(state)
    with lock:
        return gen.setdefault("file_list", []), gen.setdefault("file_settings_list", [])

def get_current_audio_gallery(state):
    gen = get_gen_info(state)
    with lock:
        return gen.setdefault("audio_file_list", []), gen.setdefault("audio_file_settings_list", [])


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
    acquire_gpu=lambda state: acquire_GPU_ressources(state, "deepy", "Deepy", parks=True),
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


def _deepy_runtime_deps():
    return deepy_cli.DeepyCliDeps(controller=_deepy, get_server_config=lambda: server_config, get_gen_info=get_gen_info, get_settings_from_file=get_settings_from_file, load_queue_action=load_queue_action, validate_task=validate_task, generate_media=generate_media, default_model_type=transformer_type, callbacks=deepy_cli.DeepyCliCallbacks(handlers={"abort_generation": (lambda state, client_id: abort_generation(state, client_id, notify=False),), "format_media_info": (lambda path, settings: format_media_info(path, settings)[0],), "save_workspace_choice": (lambda workspace_id: update_config(server_config, server_config_filename, {"last_workspace_id": workspace_id}),)}))



@upsampler_api.ui_query_scope()
def generate_media_tab(update_form = False, state_dict = None, ui_defaults = None, model_family = None, model_base_type_choice = None, model_choice = None, model_description = None, header = None, main = None, main_tabs= None, tab_id='generate', edit_tab=None, default_state=None, model_toolbar=None, model_filter=None):
    global inputs_names #, advanced
    plugin_data = gr.State({})
    edit_mode = tab_id=='edit'
    floating_generate_button = server_config.get("floating_generate_button", True)

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
        state_dict[model_dropdowns.LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY] = server_config.get(model_dropdowns.LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY, {})
        state_dict[model_dropdowns.MODEL_OUTPUT_FILTER_CONFIG_KEY] = model_dropdowns.get_model_output_filter(_get_dropdown_deps())
        state_dict["last_resolution_per_group"] = server_config.get("last_resolution_per_group", {})
        state_dict["resolution_picker_mode"] = server_config.get("resolution_picker_mode", "presets")
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
        launch_loras = ui_defaults.get("activated_loras",[])
    launch_loras = update_loras_url_cache(get_lora_dir(model_type), launch_loras)
    with gr.Row():
        column_kwargs = {'elem_id': 'edit-tab-content'} if tab_id == 'edit' else {}
        with gr.Column(**column_kwargs, elem_classes=["wangp-actions-column"] if floating_generate_button else None):
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
                            lset_type = 1 if launch_preset.endswith(".lset") else (3 if launch_preset.endswith(".zip") else 2)
                        save_lset_prompt_drop= gr.Dropdown(
                            choices=[
                                # ("Save Loras & Only Prompt Comments", 0),
                                ("Save Only Loras & Full Prompt", 1),
                                ("Save All the Settings except Media", 2),
                                ("Save All the Settings including Media", 3),
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
            any_spectrum_cache = model_def.get("spectrum_cache", False)
            any_first_block_cache = model_def.get("first_block_cache", False)
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
            multi_prompts_gen_type_value = prompt_parser.normalize_multi_prompts_mode(ui_get("multi_prompts_gen_type"), default=server_config["multi_prompts_gen_type"])
            if not sliding_window_enabled:
                multi_prompts_gen_type_value = multi_prompts_gen_type_value.replace("W", "G")
            prompt_label, wizard_prompt_label = get_prompt_labels(multi_prompts_gen_type_value, model_def, image_outputs, audio_only)            
            any_video_source = False
            fps = get_model_fps(base_model_type)
            image_prompt_type_value = ""
            video_prompt_type_value = ""
            any_start_image = any_end_image = any_reference_image = any_image_mask = False
            v2i_switch_supported = model_def.get("v2i_switch_supported", False) and not image_outputs
            ti2v_2_2 = base_model_type in ["ti2v_2_2"]
            gallery_height = 350
            def get_image_gallery(label ="", value = None, single_image_mode = False, visible = False, remove_selected_from_preview = False):
                with gr.Row(visible = visible) as gallery_row:
                    gallery_classes = ("adv-media-gallery", "amg-remove-selected") if remove_selected_from_preview else ("adv-media-gallery",)
                    gallery_amg = AdvancedMediaGallery(media_mode="image", height=gallery_height, columns=4, label=label, initial=value,
                                                       single_image_mode=single_image_mode, elem_classes=gallery_classes)
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
                image_prompt_types_allowed = del_in_sequence(image_prompt_types_allowed, "SEVL")
            with gr.Column(visible= image_prompt_types_allowed not in ("", "T") or model_mode_choices is not None and image_mode_value in model_modes_visibility ) as image_prompt_column: 
                # Video Continue /  Start Frame / End Frame
                image_prompt_type_value= ui_get("image_prompt_type")
                image_prompt_type = gr.Text(value= image_prompt_type_value, visible= False)
                end_option_visible = end_frames_option_visible(model_def, image_prompt_type_value) and not image_outputs
                image_prompt_type_choices = []
                if "T" in image_prompt_types_allowed: 
                    image_prompt_type_choices += [("Text Only", "")]
                if "S" in image_prompt_types_allowed: 
                    image_prompt_type_choices += [("Start with Image", "S")]
                    any_start_image = True
                if "V" in image_prompt_types_allowed:
                    any_video_source = True
                    image_prompt_type_choices += [("Continue Video", "V")]
                if "L" in image_prompt_types_allowed:
                    any_video_source = True
                    image_prompt_type_choices += [("Continue Last Video", "L")]
                with gr.Group(visible= len(image_prompt_types_allowed)>1 and image_mode_value == 0) as image_prompt_type_group:
                    with gr.Row(elem_classes=["wangp-input-mode-row"]):
                        image_prompt_type_radio_allowed_values= filter_letters(image_prompt_types_allowed, "SVL")
                        image_prompt_type_radio_value = filter_letters(image_prompt_type_value, image_prompt_type_radio_allowed_values,  image_prompt_type_choices[0][1] if len(image_prompt_type_choices) > 0 else "")
                        if len(image_prompt_type_choices) > 0:
                            image_prompt_type_radio = gr.Radio(image_prompt_type_choices, value=image_prompt_type_radio_value, label="Video Input Mode", show_label=False, visible=len(image_prompt_types_allowed)>1, min_width=0, elem_classes=["wangp-input-modes"])
                        else:
                            image_prompt_type_radio = gr.Radio(choices=[("", "")], value="", visible= False)
                        if "E" in image_prompt_types_allowed:
                            image_prompt_type_endcheckbox = gr.Checkbox(value="E" in image_prompt_type_value, label="End image", show_label=False, visible=end_option_visible, scale=0, min_width=0, elem_classes=["wangp-input-end"])
                            any_end_image = True
                        else:
                            image_prompt_type_endcheckbox = gr.Checkbox( value =False, show_label= False, visible= False , scale= 1)
                image_start_row, image_start, image_start_extra = get_image_gallery(label= "Images as starting points for new Videos in the Generation Queue" + (" (None for Black Frames)" if model_def.get("black_frame", False) else ''), value = ui_defaults.get("image_start", None), visible= "S" in image_prompt_type_value )
                video_source_label = "Video to Continue"
                video_source_value = ui_defaults.get("video_source", None)
                video_source = gr.Video(label= video_input_label_with_info(video_source_label, video_source_value), height = gallery_height, visible= "V" in image_prompt_type_value, value= video_source_value, elem_id="video_input")
                image_end_row, image_end, image_end_extra = get_image_gallery(label=get_image_end_label(multi_prompts_gen_type_value), value = ui_defaults.get("image_end", None), visible=end_option_visible and "E" in image_prompt_type_value)
                if model_mode_choices is None:
                    model_mode_value = None
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
            guide_custom_choices = get_guide_custom_choices(model_def, image_mode_value)
            image_ref_choices = model_def.get("image_ref_choices", None)
            video_prompt_type_value = ui_get("video_prompt_type")
            dropdown_selectable = image_mode_value != 2
            image_ref_inpaint = image_mode_value == 2 and model_def.get("inpaint_with_image_ref", False)
            guide_selection_context_visible = dropdown_selectable or image_ref_inpaint
            guide_selector_visible = guide_preprocessing is not None and guide_preprocessing.get("visible", True)
            guide_alt_selector_visible = guide_custom_choices is not None and guide_custom_choices.get("visible", True)
            mask_selector_visible = video_mask_dropdown_visible(mask_preprocessing, video_prompt_type_value)
            image_ref_selector_visible = image_ref_inpaint or image_ref_choices is not None and image_ref_choices.get("visible", True)
            custom_video_selection = model_def.get("custom_video_selection", None)
            custom_video_trigger = "" if custom_video_selection is None else custom_video_selection.get("trigger", "")
            custom_selector_visible = custom_video_selection is not None and (len(custom_video_trigger) == 0 or custom_video_trigger in video_prompt_type_value)
            guide_selection_visible = guide_selection_context_visible and (guide_selector_visible or guide_alt_selector_visible or custom_selector_visible or mask_selector_visible or image_ref_selector_visible)
            video_prompt_defaults = (guide_custom_choices.get("default", "") if guide_custom_choices is not None else "") + (image_ref_choices["choices"][0][1] if image_ref_choices is not None else "")
            video_prompt_inputs_visible = any_letters(video_prompt_type_value + video_prompt_defaults, "VIKFG")

            with gr.Column(visible=guide_selection_visible or video_prompt_inputs_visible) as video_prompt_column:
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
                            if not video_guides_shown(model_def, image_mode_value): process_label = process_label.replace("Video", "Image")
                            guide_preprocessing_choices.append( (process_label, process_type) )

                        video_prompt_type_video_guide_label = guide_preprocessing.get("label", "Control Video Process")
                        if not video_guides_shown(model_def, image_mode_value): video_prompt_type_video_guide_label = video_prompt_type_video_guide_label.replace("Video", "Image")
                        video_prompt_type_video_guide = gr.Dropdown(
                            guide_preprocessing_choices,
                            value=filter_letters(video_prompt_type_value,  all_guide_processes, guide_preprocessing.get("default", "") ),
                            label= video_prompt_type_video_guide_label , scale = 1, visible= dropdown_selectable and guide_selector_visible , show_label= True,
                        )
                        any_control_video = True
                        any_control_image = not video_guides_shown(model_def, image_mode_value) and any("V" in choice for choice in guide_preprocessing["selection"])

                    # Alternate Control Video Preprocessing / Options
                    if guide_custom_choices is None:
                        video_prompt_type_video_guide_alt = gr.Dropdown(choices=[("","")], value="", label="Control Video", visible= False, scale = 1 )
                    else:
                        video_prompt_type_video_guide_alt_label = guide_custom_choices.get("label", "Control Video Process")
                        if not video_guides_shown(model_def, image_mode_value): video_prompt_type_video_guide_alt_label = video_prompt_type_video_guide_alt_label.replace("Video", "Image")
                        video_prompt_type_video_guide_alt_choices = [(label if video_guides_shown(model_def, image_mode_value) else label.replace("Video", "Image"), value) for label,value in guide_custom_choices["choices"] ]
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
                        any_control_image |= not video_guides_shown(model_def, image_mode_value) and any("V" in choice for _, choice in guide_custom_choices["choices"])
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
                        mask_letter_filter = "XYZWNA"
                        mask_choices_value = get_default_value(mask_preprocessing_choices, filter_letters(video_prompt_type_value, mask_letter_filter, mask_preprocessing.get("default", "")) )
                        video_prompt_type_value = update_value(video_prompt_type_value, mask_choices_value, mask_letter_filter)
                        video_prompt_type_video_mask = gr.Dropdown(
                            mask_preprocessing_choices,
                            value= mask_choices_value,
                            label= video_prompt_type_video_mask_label , scale = 1, visible= dropdown_selectable and mask_selector_visible,
                            show_label= True,
                        )
                        any_control_video = True

                    # Image Refs Selection
                    if image_ref_inpaint:
                       image_ref_choices = { "choices": [("None", ""), ("People / Objects", "I"), ], "letters_filter":  "I", }

                    if image_ref_choices is None:
                        if not any_reference_image:
                            video_prompt_type_value = update_value(video_prompt_type_value, "", "I")
                        video_prompt_type_image_refs = gr.Dropdown(
                            # choices=[ ("None", ""),("Start", "KI"),("Ref Image", "I")],
                            choices=[ ("None", ""),],
                            value=filter_letters(video_prompt_type_value, ""),
                            visible = False,
                            label="Start / Reference Images", scale = 1
                        )
                    else:
                        any_reference_image = True
                        images_ref_letter_filter = image_ref_choices["letters_filter"]
                        images_ref_value = get_default_value(image_ref_choices["choices"], filter_letters(video_prompt_type_value, images_ref_letter_filter) )
                        video_prompt_type_value = update_value(video_prompt_type_value, images_ref_value , images_ref_letter_filter)                        

                        video_prompt_type_image_refs = gr.Dropdown(
                            choices= image_ref_choices["choices"],
                            value= images_ref_value,
                            visible = guide_selection_context_visible and image_ref_selector_visible,
                            label=image_ref_choices.get("label", "Inject Reference Images"), show_label= image_ref_choices.get("show_label", True), scale = 1
                        )

                video_prompt_type = gr.Text(value= video_prompt_type_value, visible= False)

                image_guide = gr.Image(label= "Control Image", height = 800, type ="pil", visible= image_mode_value==1 and not video_guides_shown(model_def, image_mode_value) and "V" in video_prompt_type_value and ("U" in video_prompt_type_value or "A" not in video_prompt_type_value ) , value= ui_defaults.get("image_guide", None))
                video_guide_label = model_def.get("video_guide_label", "Control Video")
                video_guide_value = ui_defaults.get("video_guide", None)
                video_guide = gr.Video(label= video_input_label_with_info(video_guide_label, video_guide_value), height = gallery_height, visible= video_guides_shown(model_def, image_mode_value) and "V" in video_prompt_type_value, value= video_guide_value, elem_id="video_input")
                video_guide2_label = model_def.get("video_guide2_label", "Control Video 2")
                video_guide2_value = ui_defaults.get("video_guide2", None)
                video_guide2 = gr.Video(label=video_input_label_with_info(video_guide2_label, video_guide2_value), height=gallery_height, visible=video_guides_shown(model_def, image_mode_value) and "+" in video_prompt_type_value, value=video_guide2_value, elem_id="video_input2")
                video_guide3_label = model_def.get("video_guide3_label", "Control Video 3")
                video_guide3_value = ui_defaults.get("video_guide3", None)
                video_guide3 = gr.Video(label=video_input_label_with_info(video_guide3_label, video_guide3_value), height=gallery_height, visible=video_guides_shown(model_def, image_mode_value) and "*" in video_prompt_type_value, value=video_guide3_value, elem_id="video_input3")
                magic_mask_visible = video_mask_area_visible(video_prompt_type_value)
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
                    video_mask_label = model_def.get("video_mask_label", "Video Mask Area (for Inpainting, white = Control Area, black = Unchanged)")
                    video_mask_value = ui_defaults.get("video_mask", None)
                    video_mask = gr.Video(label=video_input_label_with_info(video_mask_label, video_mask_value), visible=(not image_outputs) and magic_mask_visible, height=gallery_height, value=video_mask_value)
                    if not image_outputs:
                        magic_mask_ui = MagicMaskUI().render(visible=magic_mask_visible)
                        magic_mask_uis.append(magic_mask_ui)
                        magic_mask_video_btn = magic_mask_ui.trigger
                    else:
                        magic_mask_ui = MagicMaskUI().render(visible=False)
                        magic_mask_uis.append(magic_mask_ui)
                        magic_mask_video_btn = magic_mask_ui.trigger
                mask_strength_always_enabled = model_def.get("mask_strength_always_enabled", False)  
                masking_strength = setting_slider("masking_strength", visible=(mask_strength_always_enabled or "G" in video_prompt_type_value) and video_mask_area_visible(video_prompt_type_value))
                mask_expand = setting_slider("mask_expand", visible=video_mask_area_visible(video_prompt_type_value))

                image_refs_single_image_mode = model_def.get("one_image_ref_needed", False) or ("I" in video_prompt_type_value and (model_def.get("one_image_ref_only_with_background", False) or not any_letters(video_prompt_type_value, "KF")) and model_def.get("one_image_ref_only", False))
                image_refs_label = "Start Image" if hunyuan_video_avatar else ("Reference Image" if image_refs_single_image_mode else "Reference Images")  + (" (each Image will be associated to a Sliding Window)" if infinitetalk else "")
                image_refs_row, image_refs, image_refs_extra = get_image_gallery(label= image_refs_label, value = ui_defaults.get("image_refs", None), visible= "I" in video_prompt_type_value, single_image_mode=image_refs_single_image_mode, remove_selected_from_preview=True)

                frames_positions = gr.Text(value=ui_get("frames_positions") , visible= "F" in video_prompt_type_value, scale = 2, label= "Positions of Injected Frames (1=first, L=window end, X=skip window; no position for other Image Refs)" )
                image_refs_relative_size = setting_slider("image_refs_relative_size", visible="I" in video_prompt_type_value)

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

            input_video_strength = setting_slider("input_video_strength", value=ui_get("input_video_strength", 1.0), visible=input_video_strength_visible(model_def, image_prompt_type_value, video_prompt_type_value))

            # audio selection
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
                audio_guide3 = gr.Audio(value= ui_defaults.get("audio_guide3", None), type="filepath", label=model_def.get("audio_guide3_label","Voice to follow #3"), show_download_button= True, visible= any_audio_prompt and "D" in audio_prompt_type_value )
            custom_guide_def = model_def.get("custom_guide", None)
            any_custom_guide= custom_guide_def is not None
            any_mesh_input = "3d" in model_metadata.custom_guide_kinds(model_def)  # the input kinds model discovery (Deepy, MCP) reports
            with gr.Row(visible=custom_setting_visible(custom_guide_def, video_prompt_type_value, audio_prompt_type_value, model_mode_value)) as custom_guide_row:
                if custom_guide_def is None:
                    custom_guide = gr.File(value= None, type="filepath", label= "Custom Guide", height=41, visible= False )
                else:
                    custom_guide = gr.File(value=ui_defaults.get("custom_guide", custom_guide_def.get("default")), type="filepath", label=custom_guide_def.get("label", "Custom Guide"), height=41, visible=True, file_types=custom_guide_def.get("file_types", ["*.*"]), elem_classes="wangp-settings-upload")
            remove_background_visible = any_audio_prompt and any_letters(audio_prompt_type_value, "ABXK") and not image_outputs
            normalize_audio_visible = any_audio_prompt and all_letters(audio_prompt_type_value, "AB") and not image_outputs
            custom_audio_option_label, custom_audio_option_flag = get_audio_prompt_type_custom_option_def(model_def)
            custom_audio_option_visible = any_audio_prompt and len(custom_audio_option_flag) > 0 and not image_outputs
            with gr.Row(visible=remove_background_visible or normalize_audio_visible or custom_audio_option_visible) as audio_options_row:
                remove_background_sound = gr.Checkbox(label="Remove Background Music / Noise" if audio_only else "Ignore Background Music (for better LipSync)", value="V" in audio_prompt_type_value, visible=remove_background_visible)
                normalize_audio_volumes = gr.Checkbox(label="Normalize Audio Volumes", value="N" in audio_prompt_type_value, visible=normalize_audio_visible)
                audio_prompt_type_custom_option = gr.Checkbox(label=custom_audio_option_label, value=custom_audio_option_flag in audio_prompt_type_value, visible=custom_audio_option_visible)
            with gr.Row(visible = any_audio_prompt and any_multi_speakers and (model_def.get("multitalk_class", False) or model_def.get("speaker_locations", False)) and ("B" in audio_prompt_type_value or "X" in audio_prompt_type_value) and not image_outputs ) as speakers_locations_row:
                speakers_locations = gr.Text( ui_get("speakers_locations"), label="Speakers Locations separated by a Space. Each Location = Left:Right or a BBox Left:Top:Right:Bottom", visible= True)

            advanced_prompt = advanced_ui
            prompt_vars=[]

            client_id = gr.Textbox( visible= False, value=ui_get("client_id", ""))
            if advanced_prompt:
                default_wizard_prompt, variables, values= None, None, None
            else:                 
                default_wizard_prompt, variables, values, errors =  extract_wizard_prompt(launch_prompt)
                advanced_prompt  = len(errors) > 0
            prompt_info_html = render_prompt_info_label(prompt_label, model_type, model_def, "advanced")
            wizard_prompt_info_html = render_prompt_info_label(wizard_prompt_label, model_type, model_def, "wizard")
            with gr.Column(visible=advanced_prompt, elem_classes=["wangp-prompt-tools-stack"]) as prompt_column_advanced:
                prompt_info_label = gr.HTML(value=prompt_info_html, visible=True, elem_classes=["wangp-prompt-tools-anchor"])
                prompt = gr.Textbox(visible=advanced_prompt, label=prompt_label, value=launch_prompt, lines=3, elem_id=PROMPT_ADVANCED_ELEM_ID, elem_classes=["wangp-voice-prompt"])

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
            with gr.Column(visible=not advanced_prompt, elem_classes=["wangp-prompt-tools-stack"]) as prompt_column_wizard:
                wizard_prompt_info_label = gr.HTML(value=wizard_prompt_info_html, visible=True, elem_classes=["wangp-prompt-tools-anchor"])
                wizard_prompt = gr.Textbox(visible=not advanced_prompt, label=wizard_prompt_label, value=default_wizard_prompt, lines=3, elem_id=PROMPT_WIZARD_ELEM_ID, elem_classes=["wangp-voice-prompt"])
                wizard_prompt_activated_var = gr.Text(wizard_prompt_activated, visible= False)
                wizard_variables_var = gr.Text(wizard_variables, visible = False)
            on_demand_prompt_enhancer = server_config.get("enhancer_mode", 0) == 1
            prompt_enhancer_choices, prompt_enhancer_default, prompt_enhancer_def = get_prompt_enhancer_choices(model_def, audio_only, image_mode_value, include_disabled=not on_demand_prompt_enhancer, video_prompt_type=video_prompt_type_value, inputs={**ui_defaults, "image_prompt_type": image_prompt_type_value})
            state_dict["enhancer_label_choices_" + tab_id] = prompt_enhancer_choices
            with gr.Row(visible=server_config.get("enhancer_enabled", 0) > 0 and any(value for _, value in prompt_enhancer_choices), elem_classes=["wangp-prompt-enhancer-row"]) as prompt_enhancer_row:
                prompt_enhancer_value = str(ui_get("prompt_enhancer") or "")
                prompt_enhancer_btn_label = str(model_def.get("prompt_enhancer_button_label", "Enhance Prompt"))
                # Always visible (not only in on-demand mode): Automatic-mode
                # users otherwise lose the manual trigger entirely, e.g. to
                # re-enhance after editing the prompt. The dropdown default
                # is unchanged; the button just runs enhancement on demand.
                prompt_enhancer_btn = gr.Button( value =prompt_enhancer_btn_label, visible= True, size="lg", scale=1, elem_classes="btn_centered")
                prompt_enhancer_values = [value for _, value in prompt_enhancer_choices]
                prompt_enhancer_mode_value = prompt_enhancer_chaining.normalize_choice(prompt_enhancer_value, prompt_enhancer_values, prompt_enhancer_default, require_choice=on_demand_prompt_enhancer)

                prompt_enhancer_think_visible = server_config.get("enhancer_enabled", 0) in (3, 4, 5, 6)
                prompt_enhancer_think_value = prompt_enhancer_think_visible and "K" in prompt_enhancer_value and len(prompt_enhancer_mode_value) > 0
                prompt_enhancer_value = build_prompt_enhancer_value(prompt_enhancer_mode_value, prompt_enhancer_think_value)
                prompt_enhancer_think_classes = "cbx_centered" if on_demand_prompt_enhancer else "cbx_bottom"
                prompt_enhancer = gr.Text(value=prompt_enhancer_value, visible=False)
                prompt_enhancer_mode_dropdown = gr.Dropdown(
                    choices=prompt_enhancer_choices,
                    value=prompt_enhancer_mode_value if prompt_enhancer_choices else None,
                    label=model_def.get("prompt_enhancer_button_label", "Enhance Prompt using a LLM") , scale = 5,
                    visible= True, show_label= not on_demand_prompt_enhancer, min_width=0,
                )
                prompt_enhancer_think = gr.Checkbox(label="Think", value=prompt_enhancer_think_value, visible=prompt_enhancer_think_visible, scale=0, min_width=0, elem_classes=prompt_enhancer_think_classes)
            prompt_enhancer_progress = None if update_form else WangpProgress.component(stop=True)
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
                        elem_classes=["wangp-voice-prompt"],
                        visible=True,
                    )
            else:
                with gr.Row(visible=False) as alt_prompt_row:
                    alt_prompt = gr.Textbox(value=ui_get("alt_prompt", ""), visible=False, elem_classes=["wangp-voice-prompt"])

            custom_settings = get_model_custom_settings(model_def)
            custom_settings_values = ui_get("custom_settings", None)
            if not isinstance(custom_settings_values, dict):
                custom_settings_values = {}
            custom_settings_rows = []
            custom_setting_text_inputs, custom_setting_slider_inputs, custom_setting_dropdown_inputs = [], [], []
            custom_setting_rows_count = math.ceil(CUSTOM_SETTINGS_MAX / CUSTOM_SETTINGS_PER_ROW)
            for row_idx in range(custom_setting_rows_count):
                row_start = row_idx * CUSTOM_SETTINGS_PER_ROW
                row_end = min(row_start + CUSTOM_SETTINGS_PER_ROW, CUSTOM_SETTINGS_MAX)
                row_visible = any(custom_setting_visible(custom_settings[i] if i < len(custom_settings) else None, video_prompt_type_value, audio_prompt_type_value, model_mode_value) for i in range(row_start, row_end))
                with gr.Row(visible=row_visible) as custom_settings_row:
                    for setting_index in range(row_start, row_end):
                        setting_def = custom_settings[setting_index] if setting_index < len(custom_settings) else None
                        setting_visible = custom_setting_visible(setting_def, video_prompt_type_value, audio_prompt_type_value, model_mode_value)
                        setting_default = get_custom_setting_value_from_dict(custom_settings_values, setting_def, setting_index) if setting_def is not None else ""
                        if setting_default is None:
                            setting_default = ""
                        setting_label = setting_def.get("label", f"Custom Setting {setting_index + 1}") if setting_def is not None else f"Custom Setting {setting_index + 1}"
                        slider_bounds = get_custom_setting_slider_bounds(setting_def)
                        dropdown_choices = get_custom_setting_dropdown_choices(setting_def)
                        slider_setting = slider_bounds is not None
                        dropdown_setting = dropdown_choices is not None
                        text_setting = setting_def is not None and not slider_setting and not dropdown_setting
                        custom_setting_text_inputs.append(gr.Textbox(value=str(setting_default) if text_setting else "", label=setting_label if text_setting else "", visible=setting_visible and text_setting, lines=1))
                        custom_setting_slider_inputs.append(gr.Slider(minimum=slider_bounds[0] if slider_setting else 0, maximum=slider_bounds[1] if slider_setting else 1, value=get_custom_setting_slider_value(setting_default, slider_bounds) if slider_setting else 0, step=slider_bounds[2] if slider_setting else 1, label=setting_label if slider_setting else "", visible=setting_visible and slider_setting, show_reset_button=False))
                        custom_setting_dropdown_inputs.append(gr.Dropdown(choices=dropdown_choices or [], value=get_custom_setting_dropdown_value(setting_default, dropdown_choices) if dropdown_setting else None, label=setting_label if dropdown_setting else "", visible=setting_visible and dropdown_setting))
                custom_settings_rows.append(custom_settings_row)
            custom_setting_extra_inputs = custom_setting_text_inputs + custom_setting_slider_inputs + custom_setting_dropdown_inputs


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

            with gr.Column(visible=not audio_only and not model_def.get("mesh_outputs", False)) as resolution_row:  # 3D models size their own inputs
                fit_canvas = server_config.get("fit_canvas", 0)
                if fit_canvas == 1:
                    label = "Outer Box Resolution"
                elif fit_canvas == 2:
                    label = "Output Resolution"
                else:
                    label = "Resolution Budget"
                current_resolution_choice = ui_get("resolution") if update_form or last_resolution is None else last_resolution
                if audio_only:
                    selected_group = resolution_utils.categorize_resolution(current_resolution_choice)
                    available_groups = [selected_group]
                    selected_group_resolutions = [(current_resolution_choice, current_resolution_choice)]
                else:
                    resolution_choices, current_resolution_choice = get_resolution_choices(current_resolution_choice, model_def)
                    available_groups, selected_group_resolutions, selected_group = resolution_utils.group_resolution_choices(resolution_choices, current_resolution_choice)
                picker_mode = "presets" if model_def.get("lock_resolutions") else state_dict.get("resolution_picker_mode", server_config.get("resolution_picker_mode", "presets"))
                mp_resolution = current_resolution_choice or selected_group_resolutions[0][1]
                mp_values = resolution_utils.megapixel_controls(mp_resolution, selected_group_resolutions if audio_only else resolution_choices, model_def, server_config.get("enable_4k_resolutions", 0) == 1)
                with gr.Row(elem_classes=["resolution-picker-row"]):
                    resolution_picker_toggle = gr.Button("Switch to Presets" if picker_mode == "mp" else "Switch to Megapixels", visible=not model_def.get("lock_resolutions", False), elem_classes=["resolution-picker-toggle"], size="sm", scale=0, min_width=24)
                    with gr.Column(scale=1, min_width=0):
                        with gr.Row(visible=picker_mode == "presets") as resolution_presets_row:
                            resolution_group = gr.Dropdown(choices=[(resolution_utils.GROUP_LABELS.get(group, group), group) for group in available_groups], value=selected_group, label="Category", min_width=120)
                            resolution = gr.Dropdown(choices=selected_group_resolutions, value=current_resolution_choice, label=label, scale=5, elem_id=RESOLUTION_ELEM_ID)
                        with gr.Row(visible=picker_mode == "mp") as resolution_mp_row:
                            resolution_ratio = gr.Dropdown(choices=mp_values["ratios"], value=mp_values["ratio"], label="Aspect Ratio", min_width=130)
                            resolution_mp = gr.Slider(minimum=mp_values["minimum"], maximum=mp_values["maximum"], value=mp_values["value"], step=0.05, label=resolution_mp_label(mp_resolution), scale=5, min_width=180, show_reset_button=False)
            with gr.Row(visible= not audio_only) as number_frames_row:
                batch_label = model_def.get("batch_size_label", "Number of Images to Generate")
                batch_size_max = max(1, int(model_def.get("image_batch_size_max", 16)))
                batch_size_value = min(batch_size_max, max(1, int(ui_get("batch_size") or 1)))
                batch_size = gr.Slider(1, batch_size_max, value=batch_size_value, step=1, label=batch_label, visible = image_outputs, show_reset_button= False)
                if image_outputs:
                    video_length = gr.Slider(1, 9999, value=ui_get("video_length"), step=1, label="Number of frames", visible = False, show_reset_button= False)
                else:
                    video_length_locked = model_def.get("video_length_locked", None)
                    min_frames, frames_step, _ = get_model_min_frames_and_step(base_model_type)
                    
                    current_video_length = video_length_locked if video_length_locked is not None else ui_get("video_length", 81 if get_model_family(base_model_type)=="wan" else 97)

                    computed_fps = get_computed_fps(ui_get("force_fps"), base_model_type , ui_defaults.get("video_guide", None), ui_defaults.get("video_source", None))
                    maximum_frames = get_max_frames(model_def.get("frames_selection_maximum", 737 if test_any_sliding_window(base_model_type) else 337))
                    if "frames_selection_maximum" in model_def:
                        maximum_frames = floor_frame_count(maximum_frames, min_frames, frames_step, model_def.get("frames_offset", 1))
                    video_length = gr.Slider(0 if audio_only else min_frames, maximum_frames, value=current_video_length,
                         step=frames_step, label=compute_video_length_label(computed_fps, current_video_length, video_length_locked, model_def.get("world_outputs", False)) , scale=5, visible = True, interactive= video_length_locked is None, show_reset_button= False)

                force_control_video_trim= gr.Dropdown(
                    choices=[
                        ("Nothing", 0),
                        ("Control Length", 1)],
                    value=1 if "|" in video_prompt_type_value else 0,
                    label="Capped By",
                    visible=force_control_video_trim_visible(model_def, image_mode_value, video_prompt_type_value, audio_prompt_type_value)
                )

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
                            guidance_phases_count = 2 if guidance_phases_value == PHASE_2_TILING_GUIDANCE_VALUE else guidance_phases_value
                            phase_2_spatial_tiling = model_def.get("phase_2_spatial_tiling", False)
                            if phase_2_spatial_tiling and guidance_phases_count == 2 and PHASE_2_TILING_VIDEO_PROMPT_FLAG in video_prompt_type_value:
                                guidance_phases_value = PHASE_2_TILING_GUIDANCE_VALUE
                            visible_phases = model_def.get("visible_phases", 3) 
                            guidance_phases = gr.Dropdown(
                                choices= (["none", 0] if guidance_phases_count == 0 else []) + [("One Phase", 1),("Two Phases", 2)] + ([("Two Phases with Tiling", PHASE_2_TILING_GUIDANCE_VALUE)] if phase_2_spatial_tiling else []) + ([("Three Phases", 3)] if guidance_max_phases >=3 else []),
                                value= guidance_phases_value,
                                label="Guidance Phases" if visible_phases>=2 else "Phases",
                                visible= guidance_max_phases >=2,
                                interactive = not model_def.get("lock_guidance_phases", False)
                            )
                        with gr.Row(visible = get_container_def("guidance_phases_row").visible and guidance_phases_count >= 2) as guidance_phases_row:
                            multiple_submodels = model_def.get("multiple_submodels", False)
                            model_switch_phase = gr.Dropdown(
                                choices=[
                                    ("Phase 1-2 transition", 1),
                                    ("Phase 2-3 transition", 2)],
                                value=ui_get("model_switch_phase"),
                                label="Model Switch",
                                visible= model_def.get("multiple_submodels", False) and guidance_phases_count >= 3 and multiple_submodels
                            )
                            switch_threshold = setting_slider("switch_threshold", setting_context={"guidance_phases": guidance_phases_count})
                            num_inference_steps2 = setting_slider("num_inference_steps2", setting_context={"guidance_phases": guidance_phases_count})
                            switch_threshold2 = setting_slider("switch_threshold2", visible=guidance_phases_count >= 3)
                        with gr.Row(visible = get_container_def("guidance_row").visible ) as guidance_row:
                            guidance_scale = setting_slider("guidance_scale", setting_context={"guidance_phases": guidance_phases_count})
                            guidance2_scale = setting_slider("guidance2_scale", setting_context={"guidance_phases": guidance_phases_count}, visible=guidance_phases_count >= 2)
                            guidance3_scale = setting_slider("guidance3_scale", setting_context={"guidance_phases": guidance_phases_count}, visible=guidance_phases_count >= 3)

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

                        with gr.Row(visible=model_def.get("world_outputs", False)) as world_output_row:
                            world_output = gr.Dropdown(choices=[("World Only (explore it in the Worlds gallery)", "world"), ("World and its Video", "world_video"), ("Video Only (no explorable world)", "video")],
                                                       value=ui_get("world_output") or "world", label="Generate")

                        with gr.Row(visible=get_container_def("temperature_row").visible) as temperature_row:
                            temperature = setting_slider("temperature")
                        with gr.Row(visible = get_container_def("top_pk_row").visible ) as top_pk_row:
                            top_p = setting_slider("top_p", value=ui_get("top_p", 0.9))
                            top_k = setting_slider("top_k", value=ui_get("top_k", 50))

                        sample_solver_choices = model_def.get("sample_solvers", None)
                        with gr.Group(visible=get_container_def("sample_solver_row").visible) as sample_solver_row:
                            with gr.Row():
                                if sample_solver_choices is None:
                                    sample_solver = gr.Dropdown( value="",  choices=[ ("", ""), ], visible= False, label= "Sampler / Scheduler" )
                                else:
                                    sample_solver_value = ui_defaults["sample_solver"] = get_default_value(sample_solver_choices, ui_get("sample_solver"), sample_solver_choices[0][1])
                                    sample_solver = gr.Dropdown( value=sample_solver_value,
                                        choices= sample_solver_choices, visible= True, label= "Sampler / Scheduler"
                                    )
                                flow_shift = setting_slider("flow_shift")
                            custom_sigmas = gr.Textbox(label="Custom Sigmas", value=ui_get("custom_sigmas", ""), visible=get_setting_def("custom_sigmas").visible,
                                                      placeholder="Default if blank", elem_id=f"wangp-{tab_id}-custom-sigmas", min_width=160)
                            custom_sigmas_help = gr.update(value=field_help.render_marker(f"wangp-{tab_id}-custom-sigmas", "custom_sigmas")) if update_form else field_help.bind(custom_sigmas, "custom_sigmas")
                        with gr.Row(visible=get_container_def("control_net_weights_row").visible) as control_net_weights_row:
                            control_net_weight = setting_slider("control_net_weight")
                            control_net_weight2 = setting_slider("control_net_weight2")
                            control_net_weight_alt = setting_slider("control_net_weight_alt")
                            audio_scale = setting_slider("audio_scale", value=ui_get("audio_scale", 1), visible = not image_outputs)
                        with gr.Row(visible = not (hunyuan_t2v or hunyuan_i2v or no_negative_prompt)) as negative_prompt_row:

                            negative_prompt = gr.Textbox(label="Negative Prompt " + ("(ignored if NAG is disabled and no Guidance that is if CFG = 1)"  if get_container_def("NAG_col").visible else "(ignored if no Guidance that is if CFG = 1)") , value=ui_get("negative_prompt"), elem_classes=["wangp-voice-prompt"] )
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
                            multi_prompts_gen_choices = prompt_parser.get_multi_prompts_gen_choices(medium, include_sliding_window=sliding_window_enabled)

                            multi_prompts_gen_type = gr.Dropdown(
                                choices=multi_prompts_gen_choices,
                                value=multi_prompts_gen_type_value,
                                visible=not edit_mode,
                                scale = 1,
                                label= "How to Process each Line of the Text Prompt"
                            )

                with gr.Tab("LoRAs", visible= not audio_only or model_def.get("enabled_audio_lora", False)) as loras_tab:
                    with gr.Column(visible = True): #as loras_column:
                        gr.Markdown("<B>LoRAs can be used to create special effects on the video by mentioning a trigger word in the Prompt. You can save Loras combinations in presets.</B>")
                        loras, loras_hierarchy = get_updated_loras_dropdown(loras, launch_loras)
                        state_dict["loras"] = loras
                        loras_choices = HierarchySelector(
                            hierarchy=loras_hierarchy,
                            value= launch_loras,
                            height=0,
                            label="Activated LoRAs",
                            search_empty_label="No matching LoRAs",
                        )
                        loras_multipliers = gr.Textbox(label="LoRA Multipliers", value=launch_multis_str, elem_id=f"wangp-{tab_id}-lora-multipliers")
                        loras_multipliers_help = gr.update(value=field_help.render_marker(f"wangp-{tab_id}-lora-multipliers", "loras_multipliers")) if update_form else field_help.bind(loras_multipliers, "loras_multipliers")
                with gr.Tab("Steps Skipping", visible=any_tea_cache or any_mag_cache or any_spectrum_cache or any_first_block_cache) as speed_tab:
                    with gr.Column():
                        gr.Markdown("<B>Step-skipping accelerators avoid selected full transformer evaluations. More skipped evaluations may reduce output quality.</B>")
                        gr.Markdown("<B>Keep an initial warmup so the accelerator can establish a stable history.</B>")
                        steps_skipping_choices = [("None", "")]
                        if any_tea_cache: steps_skipping_choices += [("Tea Cache", "tea")]
                        if any_mag_cache: steps_skipping_choices += [("Mag Cache", "mag")]
                        if any_spectrum_cache: steps_skipping_choices += [("Spectrum Feature Forecasting", "spectrum")]
                        if any_first_block_cache: steps_skipping_choices += [("First Block Cache", "first_block")]
                        skip_steps_cache_type_value = ui_get("skip_steps_cache_type")
                        skip_steps_cache_type_value = get_custom_setting_dropdown_value(skip_steps_cache_type_value, steps_skipping_choices)
                        skip_steps_cache_type = gr.Dropdown(
                            choices= steps_skipping_choices,
                            value="" if not (any_tea_cache or any_mag_cache or any_spectrum_cache or any_first_block_cache) else skip_steps_cache_type_value,
                            visible=True,
                            label="Skip Steps Cache Type"
                        )
 
                        skip_steps_multiplier_choices = model_def.get("skip_steps_multiplier_choices", [
                            ("around x1.5 speed up", 1.5),
                            ("around x1.75 speed up", 1.75),
                            ("around x2 speed up", 2.0),
                            ("around x2.25 speed up", 2.25),
                            ("around x2.5 speed up", 2.5),
                        ])
                        skip_steps_multiplier_value = get_custom_setting_dropdown_value(float(ui_get("skip_steps_multiplier")), skip_steps_multiplier_choices)
                        skip_steps_multiplier = gr.Dropdown(
                            choices=skip_steps_multiplier_choices,
                            value=skip_steps_multiplier_value,
                            visible=skip_steps_cache_type_value in ("tea", "mag", "first_block"),
                            label=model_def.get("skip_steps_multiplier_label", "Skip Steps Cache Global Acceleration")
                        )
                        if not update_form:
                            skip_steps_cache_type.input(fn=lambda cache_type: gr.update(visible=cache_type in ("tea", "mag", "first_block")), inputs=[skip_steps_cache_type], outputs=[skip_steps_multiplier])
                        skip_steps_start_step_perc = gr.Slider(0, 100, value=ui_get("skip_steps_start_step_perc"), step=1, label="Skip Steps starting moment in % of generation", show_reset_button= False) 

                with gr.Tab("Post Processing", visible = not audio_only and not (model_def.get("mesh_outputs", False) and model_def.get("mesh_postprocessing", True) is False)) as post_processing_tab:  # 3D assets: 3D Refiner and Texturer
                    

                    with gr.Column():
                        postprocessing_header = gr.Markdown("<B>3D Refinement and Texturing</B>" if model_def.get("mesh_outputs", False) else "<B>Upsampling and visual refinement</B>")
                        def gen_upsampling_dropdowns(temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, element_class=None, max_height=None, image_outputs=False, late_postprocessing=False, parameter_values=None, media_kind=None):
                            """parameter_values: the slots' parameter dictionaries and the shared reference images, by setting name."""
                            mesh_kind = media_kind == mesh_postprocessing.MEDIA_3D  # 3D assets: the spatial slot holds 3D Refiners, the temporal slot Texturers
                            if image_outputs and not mesh_kind:
                                temporal_upsampling = ""
                                if len(str(spatial_upsampling or "").strip()) == 0:
                                    spatial_upsampling = ""
                            parameter_values = parameter_values or {}
                            ui_context = upsampler_api.PARAMETER_UI_LATE_POSTPROCESSING if late_postprocessing else upsampler_api.PARAMETER_UI_POSTPROCESSING
                            # each slot in its section: its selector, then its processor's parameters
                            with gr.Group(visible=mesh_kind or not image_outputs, elem_classes=["wangp-postprocessing-slot"]) as temporal_section:
                                temporal_ui = temporal_upsampler_api.create_generation_temporal_ui(gr, temporal_upsampling, visible=mesh_kind or not image_outputs, late_postprocessing=late_postprocessing, elem_classes=element_class, update_form=update_form, field_help=field_help,
                                                                                                   media_kind=media_kind or temporal_upsampler_api.MEDIA_VIDEO, help_target_id=f"wangp-{tab_id}-{'late-' if late_postprocessing else ''}temporal-upsampling")
                                temporal_parts = parameter_ui.create_slot(gr, parameter_ui.TEMPORAL, temporal_ui["method"], temporal_ui["value"], temporal_ui["method_value"], parameter_values.get(parameter_ui.SETTINGS[parameter_ui.TEMPORAL]),
                                                                          ui_context=ui_context, late_postprocessing=late_postprocessing, update_form=update_form)
                            temporal_upsampling = temporal_ui["value"]
                            temporal_upsampling_method = temporal_ui["method"]
                            temporal_upsampling_multiplier = temporal_ui["multiplier"]

                            image_mode_for_choices = 1 if image_outputs else 0
                            vae_choices = [] if late_postprocessing or mesh_kind else upsampler_api.query_model_vae_method_choices(base_model_type, model_def, image_mode_for_choices)
                            excluded_methods = set() if late_postprocessing else set(model_def.get("excluded_spatial_upsamplers", ()))
                            with gr.Group(elem_classes=["wangp-postprocessing-slot"]):
                                spatial_ui = upsampler_api.create_generation_spatial_ui(gr, spatial_upsampling, image_outputs=image_outputs, late_postprocessing=late_postprocessing, vae_choices=vae_choices, excluded_methods=excluded_methods, elem_classes=element_class, update_form=update_form, field_help=field_help, help_target_id=f"wangp-{tab_id}-{'late-' if late_postprocessing else ''}spatial-upsampling", media_kind=media_kind)
                                spatial_parts = parameter_ui.create_slot(gr, parameter_ui.SPATIAL, spatial_ui["method"], spatial_ui["value"], spatial_ui["method_value"], parameter_values.get(parameter_ui.SETTINGS[parameter_ui.SPATIAL]),
                                                                         ui_context=ui_context, late_postprocessing=late_postprocessing, update_form=update_form)
                            spatial_upsampling = spatial_ui["value"]
                            spatial_upsampling_method = spatial_ui["method"]
                            spatial_upsampling_ratio = spatial_ui["ratio"]
                            spatial_upsampler_extra = [*temporal_ui["extra_components"], *spatial_ui["extra_components"]]  # both slots' helps
                            spatial_upsampler_media_outputs = [*temporal_ui["media_outputs"], *spatial_ui["media_outputs"]]
                            spatial_upsampler_help_target_id = spatial_ui["help_target_id"]
                            # the reference images input shared by both slots
                            images_parts = parameter_ui.create_images(gr, [spatial_upsampling_method, temporal_upsampling_method], [spatial_ui["method_value"], temporal_ui["method_value"]],
                                                                      parameter_values.get(parameter_ui.IMAGES_SETTING), ui_context=ui_context, update_form=update_form)
                            spatial_postprocessing_parameters, temporal_postprocessing_parameters = spatial_parts["parameters"], temporal_parts["parameters"]
                            postprocessing_reference_images = images_parts["images"]
                            parameter_parts = [temporal_parts, spatial_parts, images_parts]

                            with gr.Row(visible=not mesh_kind) as film_grain_row:
                                film_grain_intensity = gr.Slider(0, 1, value=film_grain_intensity, step=0.01, label="Film Grain Intensity (0 = disabled)", show_reset_button= False)
                                film_grain_saturation = gr.Slider(0.0, 1, value=film_grain_saturation, step=0.01, label="Film Grain Saturation", show_reset_button= False)

                            return (temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, temporal_upsampling_method, temporal_upsampling_multiplier, spatial_upsampling_method, spatial_upsampling_ratio,
                                    spatial_postprocessing_parameters, temporal_postprocessing_parameters, postprocessing_reference_images, spatial_upsampler_extra, spatial_upsampler_media_outputs, spatial_upsampler_help_target_id,
                                    [temporal_section, *(component for parts in parameter_parts for component in parts["extra_components"])], [temporal_section, *(component for parts in parameter_parts for component in parts["media_outputs"])], film_grain_row)
                        (temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, temporal_upsampling_method, temporal_upsampling_multiplier, spatial_upsampling_method, spatial_upsampling_ratio,
                         spatial_postprocessing_parameters, temporal_postprocessing_parameters, postprocessing_reference_images, spatial_upsampler_extra, spatial_upsampler_media_outputs, spatial_upsampler_help_target_id,
                         postprocessing_parameter_extra, postprocessing_parameter_media_outputs, film_grain_row) = gen_upsampling_dropdowns(ui_get("temporal_upsampling"), ui_get("spatial_upsampling"), ui_get("film_grain_intensity"), ui_get("film_grain_saturation"), image_outputs=image_outputs,
                            parameter_values={setting: ui_get(setting) for setting in (*parameter_ui.SETTINGS.values(), parameter_ui.IMAGES_SETTING)}, media_kind=mesh_postprocessing.MEDIA_3D if model_def.get("mesh_outputs", False) else None)

                with gr.Tab("Audio", visible = not (image_outputs or audio_only)) as audio_tab:
                    audio_processor_ui = audio_processor_api.create_generation_audio_ui(gr, ui_get, ui_defaults, any_control_video=any_control_video, update_form=update_form)
                    postprocess_audio = audio_processor_ui["postprocess_audio"]
                    postprocess_audio_prompt_col = audio_processor_ui["postprocess_audio_prompt_col"]
                    postprocess_audio_prompt = audio_processor_ui["postprocess_audio_prompt"]
                    postprocess_audio_neg_prompt = audio_processor_ui["postprocess_audio_neg_prompt"]
                    postprocess_audio_control_col = audio_processor_ui["postprocess_audio_control_col"]
                    postprocess_audio_source_col = audio_processor_ui["postprocess_audio_source_col"]
                    audio_source = audio_processor_ui["audio_source"]
                    replace_voice_col = audio_processor_ui["replace_voice_col"]
                    replace_voice_method = audio_processor_ui["replace_voice_method"]
                    replace_voice_sample_row = audio_processor_ui["replace_voice_sample_row"]
                    replace_voice_sample = audio_processor_ui["replace_voice_sample"]
                    replace_voice_sample2_row = audio_processor_ui["replace_voice_sample2_row"]
                    replace_voice_sample2 = audio_processor_ui["replace_voice_sample2"]
                        
                any_perturbation = model_def.get("perturbation", False)
                any_cfg_zero = model_def.get("cfg_zero", False)
                any_cfg_star = model_def.get("cfg_star", False)
                any_apg = model_def.get("adaptive_projected_guidance", False)
                any_motion_amplitude = get_container_def("motion_amplitude_col").visible and not image_outputs
                any_extra_frames = v2i_switch_supported and image_outputs
                any_pnp = model_def.get("self_refiner", False)

                with gr.Tab("Quality", visible=any((any_perturbation, any_cfg_star, any_apg, any_motion_amplitude, any_extra_frames, any_pnp)) and not audio_only) as quality_tab:
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

                        with gr.Column(visible=any_extra_frames) as min_frames_if_references_col:
                            gr.Markdown("<B>WanGP normally runs the shortest model-compatible generation and keeps its first frame. Generating additional frames may improve still-image quality or preserve Reference / Control Image details, at extra generation cost.</B>")
                            _, _, temporal_latent = get_model_min_frames_and_step(base_model_type)
                            temporal_latent = max(1, int(temporal_latent or 1))
                            frames_offset = max(0, int(model_def.get("frames_offset", 1)))
                            first_frame_count = frames_offset if frames_offset > 1 else temporal_latent + frames_offset
                            frame_counts = [first_frame_count + temporal_latent * index for index in range(4)]
                            image_mode_video_frame_choices = [("Shortest generation, keep only the First Frame", 1)]
                            image_mode_video_frame_choices += [(f"Generate {frame_count} Frames and keep the First only when using a Reference / Control Image", frame_count) for frame_count in frame_counts]
                            image_mode_video_frame_choices += [(f"Always generate {frame_count} Frames and keep the First", 1000 + frame_count) for frame_count in frame_counts]
                            min_frames_if_references_value = ui_get("min_frames_if_references", frame_counts[1] if vace else 1)
                            if min_frames_if_references_value not in {choice[1] for choice in image_mode_video_frame_choices}:
                                min_frames_if_references_value = 1
                            min_frames_if_references = gr.Dropdown(
                                choices=image_mode_video_frame_choices,
                                value=min_frames_if_references_value,
                                visible=True,
                                scale = 1,
                                label="Generate additional frames before keeping the first image"
                            )

                        with gr.Column(visible=any_motion_amplitude) as motion_amplitude_col:
                            gr.Markdown("<B>Experimental: Accelerate Motion (1: disabled, 1.15 recommended)")
                            motion_amplitude  = setting_slider("motion_amplitude")

                        with gr.Column(visible=any_pnp) as self_refiner_col:
                            gr.Markdown("<B>Self-Refining Video Sampling (PnP) - should improve quality of Motion</B>")
                            self_refiner_setting = gr.Dropdown(choices=[("Disabled", 0),("Enabled with P1-Norm", 1), ("Enabled with P2-Norm", 2)], value=ui_get("self_refiner_setting", 0), scale=1, label="Self Refiner")
                            
                            refiner_val = ensure_refiner_list(ui_get("self_refiner_plan", []))
                            self_refiner_plan = refiner_val if update_form else gr.State(value=refiner_val)
                            
                            with gr.Column(visible=ui_get("self_refiner_setting", 0) > 0) as self_refiner_rules_ui:
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
                                    self_refiner_setting.input(fn=lambda s: gr.update(visible=s > 0), inputs=[self_refiner_setting], outputs=[self_refiner_rules_ui])

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
                        with gr.Row(visible=model_def.get("sub_parallel_windows", False)) as sub_parallel_window_row:
                            sub_parallel_window_size = setting_slider("sub_parallel_window_size", value=ui_get("sub_parallel_window_size", 0))
                            sub_parallel_window_overlap = setting_slider("sub_parallel_window_overlap", value=ui_get("sub_parallel_window_overlap", 17))
                        sliding_window_trim_first_frames = setting_slider("sliding_window_trim_first_frames", value=ui_get("sliding_window_trim_first_frames", 0))

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
                    with gr.Column(visible=model_def.get("riflex", False)) as RIFLEx_setting_col:
                        gr.Markdown("<B>With Riflex you can generate videos longer than 5s which is the default duration of videos used to train the model</B>")
                        RIFLEx_setting = gr.Dropdown(
                            choices=[
                                ("Auto (ON if Video longer than 5s)", 0),
                                ("Always ON", 1), 
                                ("Always OFF", 2), 
                            ],
                            value=ui_get("RIFLEx_setting"),
                            label="RIFLEx positional embedding to generate long video",
                            visible=model_def.get("riflex", False)
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
                                ("48", "48"), 
                                ("50", "50"), 
                            ]
                    
                        force_fps = gr.Dropdown(
                            choices=force_fps_choices,
                            value=ui_get("force_fps"),
                            label=f"Override Frames Per Second (model default={fps} fps)"
                        )

                    with gr.Column():
                        gr.Markdown('<B>Customize the Output Filename using Settings Values (<I>date, seed, resolution, num_inference_steps, prompt, flow_shift, video_length, guidance_scale</I>). For Instance:<BR>"<I>{date(YYYY-MM-DD_HH-mm-ss)}_{seed}_{prompt(50)}, {num_inference_steps}</I>"</B>')
                        output_filename = gr.Text( label= " Output Filename (Leave Blank for Auto Naming)", value= ui_get("output_filename"))

                with gr.Tab("Configs"):
                    override_values = generation_overrides.parse_overrides(ui_get("overrides"))
                    profile_type = get_profile_type_for_model(base_model_type, image_mode_value)
                    profile_type = profile_type[0].upper() + profile_type[1:]
                    gr.Markdown("<B>You can set a more agressive Memory Profile if you generate only Short Videos or Images<B>")
                    memory_profile_override = gr.Dropdown(
                        choices=[(f"Default {profile_type} Memory Profile", -1)] + memory_profile_choices,
                        value=override_values.get("memory_profile", -1),
                        label=f"Override Memory Profile"
                    )

                    gr.Markdown("<B>You can override Transformer Compilation for this generation. Default uses the current Config choice.</B>")
                    compilation_override = gr.Dropdown(
                        choices=[("Default", "")] + compile_choices,
                        value=override_values.get("compilation", ""),
                        label="Override Compilation"
                    )

                    gr.Markdown("<B>You can set a different Attention Mode to improve the quality / compatibility<B>")
                    selected_attention, override_attention_choices = get_attention_override_choices(model_def, override_values.get("attention_mode", ""))
                    attention_mode_override = gr.Dropdown(
                        choices=override_attention_choices,
                        value=selected_attention,
                        interactive=len(override_attention_choices) > 1,
                        label=f"Override Attention Mode"
                    )
                    attention_sparsity = setting_slider("attention_sparsity", visible=model_def.get("custom_attention_modes", {}).get(selected_attention, {}).get("supports_sparsity", False))

                    transformer_quantization_overridable, text_encoder_quantization_overridable = get_quantization_overridable(model_type, model_def)
                    with gr.Column(visible=transformer_quantization_overridable or text_encoder_quantization_overridable) as quantization_overrides_col:
                        gr.Markdown("<B>You can use other Quantized Weights for this generation (if they are available, otherwise the closest available ones)</B>")
                        with gr.Row():
                            transformer_quantization_override = gr.Dropdown(
                                choices=[("Default Transformer Quantization", "")] + transformer_quantization_choices,
                                value=override_values.get("transformer_quantization", ""),
                                visible=transformer_quantization_overridable,
                                label="Override Transformer Quantization"
                            )
                            text_encoder_quantization_override = gr.Dropdown(
                                choices=[("Default Text Encoder Quantization", "")] + text_encoder_quantization_choices,
                                value=override_values.get("text_encoder_quantization", ""),
                                visible=text_encoder_quantization_overridable,
                                label="Override Text Encoder Quantization"
                            )
                    overrides_inputs = [memory_profile_override, compilation_override, attention_mode_override, transformer_quantization_override, text_encoder_quantization_override]
                    overrides_value = generation_overrides.format_overrides(dict(override_values, attention_mode=selected_attention))
                    overrides = gr.Text(value=overrides_value, interactive=False, visible=False)

                    config_groups = get_model_config_groups(model_type, model_def)
                    grouped_model_configs = [model_config_groups.get_config_items(configs) for configs in config_groups]
                    config_value = model_config_groups.normalize_config_selection(config_groups, ui_get("config"))
                    config_group_values = model_config_groups.split_config_selection(config_value)
                    config_group_labels = [model_config_groups.get_config_name(configs) for configs in config_groups]
                    config_group_default_labels = [model_config_groups.get_default_label(configs) for configs in config_groups]
                    with gr.Column(visible=any(grouped_model_configs)) as config_column:
                        gr.Markdown('<B>You may pick a Variation of the Default Config (it may impact RAM/VRAM consumption or performance)</B>')
                        config_group_dropdowns = []
                        config_group_help = []
                        with gr.Row():
                            for index, group_configs in enumerate(grouped_model_configs):
                                config_choices = [(config_group_default_labels[index], "")] + [(config_def.get("name", config_id), config_id) for config_id, config_def in group_configs]
                                config_target_id = f"wangp-{tab_id}-config-{index}"
                                config_group_dropdowns.append(gr.Dropdown(choices=config_choices, value=config_group_values[index], visible=bool(group_configs), label=config_group_labels[index], elem_id=config_target_id))
                                config_help_html = field_help.render_marker(config_target_id, "config", title=config_group_labels[index], markdown=model_config_groups.get_config_infos(config_groups[index]) if group_configs else "", keep_empty=True)
                                config_group_help.append(gr.HTML(config_help_html, elem_classes=["wangp-field-help-inline-host"]))
                        config = gr.Text(value=config_value, interactive=False, visible=False)

            if not update_form:
                with gr.Row(visible=(tab_id == 'edit'), elem_classes=["wangp-settings-actions"] if floating_generate_button else None):
                    edit_btn = gr.Button("Apply Edits", elem_id="edit_tab_apply_button")
                    cancel_btn = gr.Button("Cancel", elem_id="edit_tab_cancel_button")
                    silent_cancel_btn = gr.Button("Silent Cancel", elem_id="silent_edit_tab_cancel_button", visible=False)
            with gr.Column(visible= not edit_mode, elem_classes=["wangp-settings-tools"]):
                with gr.Row(elem_classes=["wangp-settings-file-actions"]):
                    save_settings_btn = gr.Button("Set Settings as Default", visible = not args.lock_config, min_width=0, elem_classes=["wangp-settings-default"])
                    export_settings_from_file_btn = gr.Button("Export Settings to File", min_width=0, elem_classes=["wangp-settings-export"])
                    export_settings_include_media = gr.Checkbox(label="Include Media", value=False, container=False, scale=0, min_width=0, elem_classes=["wangp-settings-include-media"])
                    reset_settings_btn = gr.Button("Reset Settings", elem_classes=["wangp-reset-settings"], size="sm", scale=0, min_width=36)
                with gr.Row():
                    settings_file = gr.File(height=41, label="Load Settings From Media File / Json / Zip", elem_classes="wangp-settings-upload")
                    settings_download_payload = gr.Text(interactive=False, visible=False, value="")
                with gr.Group():
                    with gr.Row(elem_classes=["wangp-lora-download-row"]):
                        lora_url = gr.Text(label ="Lora URL", placeholder= "Enter Lora URL", scale=4, min_width=0, show_label=False, elem_classes="compact_text" )
                        download_lora_btn = gr.Button("Download", scale=0, min_width=0)
                    lora_progress = WangpProgress.component()

                assistant_ui = None
                assistant_launcher_host = None
                assistant_panel = None
                assistant_chat_event = None
                if tab_id == 'generate' and not update_form:
                    assistant_ui = deepy_gradio_ui.build_deepy_chat_ui(deepy_visible=_deepy.is_available())
                    assistant_launcher_host = assistant_ui.launcher_host
                    assistant_panel = assistant_ui.panel
                    assistant_chat_event = assistant_ui.chat_event

            mode = gr.Text(value="", visible = False)

        with gr.Column(visible=(tab_id == 'generate'), elem_classes=["wangp-actions-column"] if floating_generate_button else None):
            if not update_form:
                if tab_id == 'generate' and default_state is None:
                    global _deepy_hybrid
                    state_dict = SharedState(state_dict)
                    _deepy_hybrid = HybridService(_deepy_runtime_deps(), state_dict, process_queue=_process_tasks, finalize_queue=_finalize_generation, unload=_unload_model_if_needed, gallery_lock=lock, model_label=get_loaded_model_label, task_label=get_task_progress_label)
                    _deepy_hybrid.configure_workspaces(args.workspaces_dir or server_config.get('workspaces_dir') or os.path.join(wgp_root, 'workspaces'))
                state = default_state if default_state is not None else gr.State(state_dict)
                overrides_event = gr.on(triggers=[dropdown.input for dropdown in overrides_inputs], fn=serialize_overrides, inputs=overrides_inputs, outputs=overrides, show_progress="hidden")
                if tab_id == "generate" and header is not None:
                    header_refresh = dict(fn=refresh_attention_header, inputs=[state, overrides, guidance_phases, video_prompt_type], outputs=[header], show_progress="hidden")
                    overrides_event.then(**header_refresh)
                    gr.on(triggers=[guidance_phases.change, video_prompt_type.change], **header_refresh)
                attention_mode_override.input(fn=refresh_attention_sparsity, inputs=[state, attention_mode_override], outputs=[attention_sparsity], show_progress="hidden")
                gen_status = WangpProgress.component(visible=True, value=render_idle_generation_status(state_dict))
                main_bridge_elem_ids = tab_id == 'generate'
                load_queue_trigger = gr.Text(interactive= False, visible=False, elem_id="wangp_main_load_queue_trigger" if main_bridge_elem_ids else None)
                abort_client_id= gr.Text(interactive= False, visible=False, elem_id="wangp_main_abort_client_id" if main_bridge_elem_ids else None)
                default_files = []
                current_gallery_tab = gr.Number(0, visible=False)
                with gr.Tabs(elem_id='wangp-gallery-tabs' if main_bridge_elem_ids else None) as gallery_tabs:
                    with gr.Tab(VIDEO.tab_label, id=VIDEO.tab_id):
                        output = gr.Gallery(value =default_files, label="Generated videos", preview= True, show_label=False, elem_id="gallery" , columns=[3], rows=[1], object_fit="contain", height=450, selected_index=0, interactive= False)
                    strip_galleries, strip_tabs = {}, {}
                    for strip_source in STRIP_SOURCES:
                        with gr.Tab(strip_source.tab_label, id=strip_source.tab_id, visible=strip_source is not APP or python_apps_level() != "off") as strip_tabs[strip_source.id]:
                            strip_galleries[strip_source.id] = STRIP_GALLERY_CLASSES[strip_source.id](max_thumbnails=None, update_only=update_form)
                    apps_gallery_tab = strip_tabs[APP.id]  # App Development (Advanced): the Configuration tab shows or hides it when saved
                # (paths, selected, refresh trigger) of each strip gallery, in STRIP_SOURCES order
                strip_states = [component for gallery in strip_galleries.values() for component in gallery.get_state()]
                strip_selected = [gallery.state_selected for gallery in strip_galleries.values()]
                audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger = strip_galleries["audio"].get_state()
                mesh_files_paths, mesh_file_selected, mesh_gallery_refresh_trigger = strip_galleries[MESH.id].get_state()
                world_files_paths, world_file_selected, world_gallery_refresh_trigger = strip_galleries[WORLD.id].get_state()
                app_files_paths, app_file_selected, app_gallery_refresh_trigger = strip_galleries[APP.id].get_state()
                output_trigger = gr.Text(interactive= False, visible=False, elem_id="wangp_main_output_trigger" if main_bridge_elem_ids else None)
                selected_video_time_input = gr.Text(interactive= False, visible=False, elem_id="selected_video_time_input" if main_bridge_elem_ids else None)
                refresh_models_trigger = gr.Text(interactive= False, visible=False)
                refresh_form_trigger = gr.Text(interactive= False, visible=False)
                fill_wizard_prompt_trigger = gr.Text(interactive= False, visible=False)
                save_form_trigger = gr.Text(interactive= False, visible=False)
                gallery_source = gr.Text(interactive= False, visible=False)
                model_choice_target = gr.Text(interactive= False, visible=False, elem_id="wangp_model_choice_target" if tab_id == "generate" else None)


            with gr.Accordion("Media Info / Late Post Processing / Import Media", open=False) as video_info_accordion:
                late_audio_postprocessing_visible, late_video_postprocessing_visible, late_audio_remuxing_visible = get_selected_late_processing_tabs_visibility(state_dict)
                video_info_tab = gr.Text(value="video_info", interactive=False, visible=False)
                with gr.Tabs() as video_info_tabs:
                    default_visibility_false = {} if update_form else {"visible" : False}                        
                    default_visibility_true = {} if update_form else {"visible" : True}                        

                    with gr.Tab("Information", id="video_info"):
                        video_info = gr.HTML(visible=True, min_height=100, value=get_default_video_info()) 
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as audio_buttons_row:
                            video_info_extract_audio_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_audio_guide_btn = gr.Button("To Audio Source", min_width= 1, size ="sm", visible = any_audio_guide)
                            video_info_to_audio_guide2_btn = gr.Button("To Audio Source 2", min_width= 1, size ="sm", visible = any_audio_guide2 )
                            video_info_to_audio_source_btn = gr.Button("To Soundtrack", min_width= 1, size ="sm", visible=audio_processor_api.has_audio_source_soundtrack())
                            video_info_eject_audio_btn = gr.Button("Eject Audio File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as deleted_audio_buttons_row:
                            video_info_eject_deleted_audio_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as video_buttons_row:
                            video_info_extract_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_video_source_btn = gr.Button("To Video Source", min_width= 1, size ="sm", visible = any_video_source)
                            video_info_to_control_video_btn = gr.Button("To Control Video", min_width= 1, size ="sm", visible = any_control_video )
                            video_info_eject_video_btn = gr.Button("Eject Video", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as deleted_video_buttons_row:
                            video_info_eject_deleted_video_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as mesh_buttons_row:
                            video_info_extract_mesh_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_mesh_input_btn = gr.Button("To Input Mesh", min_width= 1, size ="sm", visible = any_mesh_input)
                            video_info_eject_mesh_btn = gr.Button("Eject 3D Asset", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as deleted_mesh_buttons_row:
                            video_info_eject_deleted_mesh_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as world_buttons_row:
                            video_info_extract_world_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_continue_world_btn = gr.Button("Continue World", min_width= 1, size ="sm")
                            video_info_eject_world_btn = gr.Button("Eject World", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as deleted_world_buttons_row:
                            video_info_eject_deleted_world_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as app_buttons_row:
                            video_info_eject_app_btn = gr.Button("Eject App", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as deleted_app_buttons_row:
                            video_info_eject_deleted_app_btn = gr.Button("Eject Deleted File", min_width= 1, size ="sm")
                        with gr.Row(**default_visibility_false, elem_classes=["wangp-media-actions"] if floating_generate_button else []) as image_buttons_row:
                            video_info_extract_image_settings_btn = gr.Button("Extract Settings", min_width= 1, size ="sm")
                            video_info_to_start_image_btn = gr.Button("To Start Image", size ="sm", min_width= 1, visible = any_start_image )
                            video_info_to_end_image_btn = gr.Button("To End Image", size ="sm", min_width= 1, visible = any_end_image)
                            video_info_to_image_mask_btn = gr.Button("To Mask Image", min_width= 1, size ="sm", visible = any_image_mask and False)
                            video_info_to_reference_image_btn = gr.Button("To Reference Image", min_width= 1, size ="sm", visible = any_reference_image)
                            video_info_to_image_guide_btn = gr.Button("To Control Image", min_width= 1, size ="sm", visible = any_control_image )
                            video_info_eject_image_btn = gr.Button("Eject Image", min_width= 1, size ="sm")
                    with gr.Tab("Post Processing", id="audio_postprocessing", visible=late_audio_postprocessing_visible) as audio_postprocessing_tab:
                        with gr.Group(elem_classes="postprocess"):
                            late_audio_ui = audio_processor_api.create_late_audio_edit_ui(gr, default_visibility_false=default_visibility_false, update_form=update_form)
                            PP_late_audio_postprocess = late_audio_ui["postprocess_audio"]
                            PP_late_audio_replace_voice_sample_row = late_audio_ui["replace_voice_sample_row"]
                            PP_late_audio_replace_voice_sample = late_audio_ui["replace_voice_sample"]
                            PP_late_audio_replace_voice_sample2_row = late_audio_ui["replace_voice_sample2_row"]
                            PP_late_audio_replace_voice_sample2 = late_audio_ui["replace_voice_sample2"]
                        with gr.Row(elem_classes=["wangp-media-actions"] if floating_generate_button else []):
                            video_info_audio_postprocessing_btn = gr.Button("Apply Audio Postprocessing", size="sm", visible=True)
                            video_info_eject_audio2_btn = gr.Button("Eject Audio", size="sm", visible=True)
                    with gr.Tab("Post Processing", id= "post_processing", visible = late_video_postprocessing_visible) as video_postprocessing_tab:
                        with gr.Group(elem_classes= "postprocess"):
                            with gr.Column():
                                PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation, PP_temporal_upsampling_method, PP_temporal_upsampling_multiplier, PP_spatial_upsampling_method, PP_spatial_upsampling_ratio, PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_postprocessing_reference_images, PP_spatial_upsampler_extra, PP_spatial_upsampler_media_outputs, PP_spatial_upsampler_help_target_id, PP_postprocessing_parameter_extra, PP_postprocessing_parameter_media_outputs, PP_film_grain_row = gen_upsampling_dropdowns("", "", 0, 0.5, element_class="postprocess", image_outputs=False, late_postprocessing=True)
                        with gr.Row(elem_classes=["wangp-media-actions"] if floating_generate_button else []):
                            video_info_postprocessing_btn = gr.Button("Apply Postprocessing", size ="sm", visible=True)
                            video_info_eject_video2_btn = gr.Button("Eject Media", size ="sm", visible=True)
                    with gr.Tab("Audio Remuxing", id= "audio_remuxing", visible = late_audio_remuxing_visible) as audio_remuxing_tab:

                        with gr.Group(elem_classes= "postprocess"):
                            late_remux_ui = audio_processor_api.create_late_remux_ui(gr, update_form=update_form, default_visibility_false=default_visibility_false)
                            PP_postprocess_audio_col = late_remux_ui["postprocess_audio_col"]
                            PP_postprocess_audio = late_remux_ui["postprocess_audio"]
                            PP_postprocess_audio_prompt_row = late_remux_ui["postprocess_audio_prompt_row"]
                            PP_postprocess_audio_prompt = late_remux_ui["postprocess_audio_prompt"]
                            PP_postprocess_audio_neg_prompt = late_remux_ui["postprocess_audio_neg_prompt"]
                            PP_postprocess_audio_seed = late_remux_ui["postprocess_audio_seed"]
                            PP_repeat_generation = late_remux_ui["repeat_generation"]
                            PP_audio_source_row = late_remux_ui["audio_source_row"]
                            PP_custom_audio = late_remux_ui["audio_source"]
                            PP_replace_voice_sample_row = late_remux_ui["replace_voice_sample_row"]
                            PP_replace_voice_sample = late_remux_ui["replace_voice_sample"]
                            PP_replace_voice_sample2_row = late_remux_ui["replace_voice_sample2_row"]
                            PP_replace_voice_sample2 = late_remux_ui["replace_voice_sample2"]
                        with gr.Row(elem_classes=["wangp-media-actions"] if floating_generate_button else []):
                            video_info_remux_audio_btn = gr.Button("Remux Audio", size ="sm", visible=True)
                            video_info_eject_video3_btn = gr.Button("Eject Video", size ="sm", visible=True)
                    with gr.Tab("Import Media to Galleries", id= "video_add"):
                        files_to_load = ImportFiles(file_count="multiple", label= "Media to Import in Galleries", height=120)
                        with gr.Row():
                            video_info_add_videos_btn = gr.Button("Import Videos / Images / Audio Files", size ="sm")
 
            if not update_form:
                generate_trigger = gr.Text(visible = False) 
                add_to_queue_trigger = gr.Text(visible = False)
                js_trigger_index = gr.Text(visible=False, elem_id="js_trigger_for_edit_refresh")

                with gr.Column(elem_classes=["wangp-settings-actions"] if floating_generate_button else None):
                    generate_btn = gr.Button("Generate")
                    with gr.Column(visible=False, elem_id=f"wangp-{tab_id}-current-actions" if floating_generate_button else None) as current_gen_column:
                        with gr.Row() as current_gen_buttons_row:
                            onemoresample_btn = gr.Button("One More", visible = True, size='md', min_width=1)
                            onemorewindow_btn = gr.Button("Extend", visible = False, size='md', min_width=1)
                            pause_btn = gr.Button("Pause", visible = True, size='md', min_width=1)
                            resume_btn = gr.Button("Resume", visible = False, size='md', min_width=1)
                            abort_btn = gr.Button("Abort", visible = True, size='md', min_width=1)
                            earlystop_btn = gr.Button("Early Stop", visible = False, size='md', min_width=1)
                    add_to_queue_btn = gr.Button("Add New Prompt To Queue", visible=False)
                with gr.Column(elem_classes=["wangp-generation-details"]) if floating_generate_button else current_gen_column:
                    with gr.Accordion("Preview", open=False):
                        preview = gr.HTML(value=refresh_preview(state_dict), label="Preview", show_label= False)
                        preview_trigger = gr.Text(visible= False)
                    with gr.Accordion("Current Prompt and Media", open=True, visible=False) as gen_info_accordion:
                        gen_info = gr.HTML(visible=False, min_height=1)
                with gr.Accordion("Queue Management", open=False) as queue_accordion:
                    with gr.Row():
                        queue_html = gr.HTML(
                            value=generate_queue_html(state_dict["gen"]["queue"]),
                            elem_id="queue_html_container"
                        )
                    queue_action_input = gr.Text(elem_id="queue_action_input", visible=False)
                    queue_action_trigger = gr.Button(elem_id="queue_action_trigger", visible=False)
                    with gr.Row(visible= True):
                        queue_zip_download_payload = gr.Text(visible=False)
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

        extra_inputs = prompt_vars + [wizard_prompt, wizard_variables_var, wizard_prompt_activated_var, prompt_info_label, wizard_prompt_info_label, video_prompt_column, image_prompt_column, image_prompt_type_group, image_prompt_type_radio, image_prompt_type_endcheckbox,
                                      prompt_column_advanced, prompt_column_wizard_vars, prompt_column_wizard, alt_prompt_row, lset_name, save_lset_prompt_drop, advanced_row, speed_tab, audio_tab, postprocess_audio_prompt_col, quality_tab,
                                      sliding_window_tab, sub_parallel_window_row, misc_tab, prompt_enhancer_row, inference_steps_row, perturbation_row, audio_guide_row, custom_guide_row, RIFLEx_setting_col,
                                      video_prompt_type_video_guide, video_prompt_type_video_guide_alt, video_prompt_type_video_mask, video_prompt_type_image_refs, video_prompt_type_video_custom_dropbox, video_prompt_type_video_custom_checkbox,
                                      apg_col, audio_prompt_type_sources, postprocess_audio_control_col, postprocess_audio_source_col, replace_voice_col, replace_voice_method, replace_voice_sample_row, replace_voice_sample2_row, force_fps_col,
                                      video_guide_outpainting_col,video_guide_outpainting_top, video_guide_outpainting_bottom, video_guide_outpainting_left, video_guide_outpainting_right,
                                      video_guide_outpainting_checkbox, video_guide_outpainting_ratio, video_guide_outpainting_row, show_advanced, magic_mask_image_btn, magic_mask_video_btn, video_info_to_control_video_btn, video_info_to_video_source_btn, sample_solver_row,
                                      video_buttons_row, deleted_video_buttons_row, image_buttons_row, audio_postprocessing_tab, video_postprocessing_tab, audio_remuxing_tab, PP_late_audio_postprocess, PP_late_audio_replace_voice_sample_row, PP_late_audio_replace_voice_sample, PP_late_audio_replace_voice_sample2_row, PP_late_audio_replace_voice_sample2, PP_postprocess_audio_col, PP_postprocess_audio, PP_postprocess_audio_prompt_row, PP_audio_source_row, PP_replace_voice_sample_row, PP_replace_voice_sample2_row,
                                      mesh_buttons_row, deleted_mesh_buttons_row, video_info_to_mesh_input_btn,
                                      audio_buttons_row, deleted_audio_buttons_row, video_info_extract_audio_settings_btn, video_info_to_audio_guide_btn, video_info_to_audio_guide2_btn, video_info_to_audio_source_btn, video_info_audio_postprocessing_btn, video_info_eject_audio_btn, video_info_eject_audio2_btn,
                                      video_info_to_start_image_btn, video_info_to_end_image_btn, video_info_to_reference_image_btn, video_info_to_image_guide_btn, video_info_to_image_mask_btn,
                                      NAG_col, audio_options_row, remove_background_sound, normalize_audio_volumes, audio_prompt_type_custom_option, speakers_locations_row, embedded_guidance_row, guidance_phases_row, guidance_row, resolution_group, cfg_free_guidance_col, control_net_weights_row, guide_selection_row, image_mode_tabs, prompt_enhancer_mode_dropdown, prompt_enhancer_think, force_control_video_trim,
                                      min_frames_if_references_col, motion_amplitude_col, video_prompt_type_alignment, prompt_enhancer_btn, tab_inpaint, tab_t2v, resolution_row, loras_tab, post_processing_tab, postprocessing_header, temporal_upsampling_method, temporal_upsampling_multiplier, spatial_upsampling_method, spatial_upsampling_ratio, temperature_row, *spatial_upsampler_extra, *postprocessing_parameter_extra, film_grain_row,*custom_settings_rows, *custom_setting_extra_inputs, top_pk_row,
                                      number_frames_row, negative_prompt_row, config_column, *config_group_dropdowns, *config_group_help, *overrides_inputs, quantization_overrides_col,
                                      self_refiner_col, self_refiner_rules_ui, pause_row, world_output_row, resolution_picker_toggle, resolution_presets_row, resolution_mp_row, resolution_ratio, resolution_mp, custom_sigmas_help, loras_multipliers_help]+\
                                      image_start_extra + image_end_extra + image_refs_extra #  presets_column,
        if tab_id == 'generate':
            # Restore the header with the effective performance settings in the same update.
            extra_inputs.append(gr.update(value=generate_header(model_type, compile, attention_mode, overrides_value, ui_get("guidance_phases"), video_prompt_type_value)[1]) if update_form else header)
        if update_form:
            locals_dict = locals()
            gen_inputs = build_form_refresh_outputs(inputs_names, locals_dict, state_dict, plugin_data, extra_inputs)
            return gen_inputs
        else:
            target_state = gr.Text(value = "state", interactive= False, visible= False)
            target_settings = gr.Text(value = "settings", interactive= False, visible= False)
            last_choice = gr.Number(value =-1, interactive= False, visible= False)
            PP_spatial_upsampler_help_target = gr.State(PP_spatial_upsampler_help_target_id)

            resolution_group.input(fn=change_resolution_group, inputs=[state, resolution_group, resolution], outputs=[resolution], show_progress="hidden").then(fn=sync_resolution_mp, inputs=[state, resolution], outputs=[resolution_ratio, resolution_mp], show_progress="hidden")
            resolution.input(fn=sync_resolution_mp, inputs=[state, resolution], outputs=[resolution_ratio, resolution_mp], show_progress="hidden")
            resolution.change(fn=record_last_resolution, inputs=[state, resolution], show_progress="hidden")
            resolution_picker_toggle.click(fn=change_resolution_picker, inputs=[state], outputs=[resolution_picker_toggle, resolution_presets_row, resolution_mp_row], show_progress="hidden")
            resolution_mp.input(fn=preview_resolution_mp, inputs=[state, resolution_mp, resolution_ratio], outputs=[resolution_mp], show_progress="hidden", trigger_mode="always_last")
            gr.on(triggers=[resolution_mp.release, resolution_ratio.input], fn=commit_resolution_mp, inputs=[state, resolution_mp, resolution_ratio], outputs=[resolution, resolution_group, resolution_mp], show_progress="hidden")
            gr.on(triggers=[dropdown.input for dropdown in config_group_dropdowns], fn=model_config_groups.serialize_config_selection, inputs=config_group_dropdowns, outputs=config, show_progress="hidden")
            force_control_video_trim.input(fn=change_force_control_video_trim, inputs= [state, video_prompt_type, force_control_video_trim], outputs = video_prompt_type)
            # Media information refreshed by select_media: action rows per media kind (MEDIA_ACTION_KINDS), deleted file rows per gallery (GALLERY_SOURCES)
            select_media_outputs = [last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, mesh_buttons_row, world_buttons_row, app_buttons_row, deleted_video_buttons_row, deleted_audio_buttons_row, deleted_mesh_buttons_row, deleted_world_buttons_row, deleted_app_buttons_row, audio_postprocessing_tab, video_postprocessing_tab, audio_remuxing_tab, PP_temporal_upsampling, PP_temporal_upsampling_method, PP_temporal_upsampling_multiplier, PP_spatial_upsampling, PP_spatial_upsampling_method, PP_spatial_upsampling_ratio, *PP_spatial_upsampler_media_outputs, *PP_postprocessing_parameter_media_outputs, PP_film_grain_row, video_info_tabs, video_info_tab]
            video_info_add_videos_btn.click(fn=add_videos_to_gallery, inputs =[state, output, last_choice, files_to_load], outputs = [gallery_tabs, current_gallery_tab, output, *strip_states, files_to_load, video_info_tabs, gallery_source] ).then(
                fn=select_media, inputs=[state, current_gallery_tab, output, last_choice, gr.State(-1), gallery_source, PP_spatial_upsampling, PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_spatial_upsampler_help_target, PP_temporal_upsampling, video_info_tab], outputs=select_media_outputs, show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
            video_info_tabs.select(fn=set_video_info_tab, outputs=[video_info_tab], show_progress="hidden")
            gallery_tabs.select(fn=set_gallery_tab, inputs=[state, video_info_tab], outputs=[current_gallery_tab, gallery_source, video_info_tabs, video_info_tab]).then(
                fn=select_media, inputs=[state, current_gallery_tab, output, last_choice, gr.State(-1), gallery_source, PP_spatial_upsampling, PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_spatial_upsampler_help_target, PP_temporal_upsampling, video_info_tab], outputs=select_media_outputs, show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
            gr.on(triggers=[video_length.release, force_fps.change, video_guide.change, video_source.change], fn=refresh_video_length_label, inputs=[state, video_length, force_fps, video_guide, video_source] , outputs = video_length, trigger_mode="always_last", show_progress="hidden"  )
            video_source.change(fn=refresh_video_input_label, inputs=[video_source, gr.State(video_source_label)], outputs=video_source, show_progress="hidden")
            video_guide.change(fn=refresh_video_input_label, inputs=[video_guide, gr.State(video_guide_label)], outputs=video_guide, show_progress="hidden")
            video_guide2.change(fn=refresh_video_input_label, inputs=[video_guide2, gr.State(video_guide2_label)], outputs=video_guide2, show_progress="hidden")
            video_guide3.change(fn=refresh_video_input_label, inputs=[video_guide3, gr.State(video_guide3_label)], outputs=video_guide3, show_progress="hidden")
            video_mask.change(fn=refresh_video_input_label, inputs=[video_mask, gr.State(video_mask_label)], outputs=video_mask, show_progress="hidden")
            guidance_phases.change(fn=change_guidance_phases, inputs= [state, guidance_phases, video_prompt_type], outputs =[model_switch_phase, guidance_phases_row, switch_threshold, switch_threshold2, guidance2_scale, guidance3_scale, video_prompt_type, num_inference_steps2])
            remove_background_sound.input(fn=refresh_remove_background_sound, inputs=[state, audio_prompt_type, remove_background_sound], outputs=[audio_prompt_type])
            normalize_audio_volumes.input(fn=refresh_normalize_audio_volumes, inputs=[state, audio_prompt_type, normalize_audio_volumes], outputs=[audio_prompt_type])
            audio_prompt_type_custom_option.input(fn=refresh_audio_prompt_type_custom_option, inputs=[state, audio_prompt_type, audio_prompt_type_custom_option], outputs=[audio_prompt_type])
            audio_prompt_type_sources.change(fn=refresh_audio_prompt_type_sources, inputs=[state, audio_prompt_type, audio_prompt_type_sources, video_prompt_type, image_mode], outputs=[audio_prompt_type, audio_guide, audio_guide2, audio_guide3, speakers_locations_row, remove_background_sound, normalize_audio_volumes, audio_prompt_type_custom_option, audio_options_row, audio_guide_row, force_control_video_trim])
            prompt_enhancer_mode_dropdown.input(fn=build_prompt_enhancer_value, inputs=[prompt_enhancer_mode_dropdown, prompt_enhancer_think], outputs=[prompt_enhancer], show_progress="hidden")
            prompt_enhancer_think.input(fn=build_prompt_enhancer_value, inputs=[prompt_enhancer_mode_dropdown, prompt_enhancer_think], outputs=[prompt_enhancer], show_progress="hidden")
            image_prompt_type_radio.change(fn=refresh_image_prompt_type_radio, inputs=[state, image_prompt_type, image_prompt_type_radio, video_prompt_type], outputs=[image_prompt_type, image_start_row, image_end_row, video_source, input_video_strength, keep_frames_video_source, image_prompt_type_endcheckbox], show_progress="hidden").then(fn=None, inputs=None, outputs=None, js="() => window.wangpRefreshInputModes()")
            image_prompt_type_endcheckbox.change(fn=refresh_image_prompt_type_endcheckbox, inputs=[state, image_prompt_type, image_prompt_type_radio, image_prompt_type_endcheckbox, video_prompt_type], outputs=[image_prompt_type, image_end_row, input_video_strength] ) 
            video_prompt_type_image_refs.input(fn=refresh_video_prompt_type_image_refs, inputs = [state, video_prompt_type, video_prompt_type_image_refs,image_mode, image_prompt_type], outputs = [video_prompt_type, image_refs_row, remove_background_images_ref,  image_refs_relative_size, frames_positions,video_guide_outpainting_col, input_video_strength], show_progress="hidden")
            enhancer_label_inputs = [image_mode, video_prompt_type, image_prompt_type, image_start, image_end, image_refs, image_guide, image_mask_guide, video_source]
            enhancer_metadata_sources = [(model_choice_target, "value"), *zip(enhancer_label_inputs, ["value", "value", "value", "presence", "presence", "count", "presence", "editor", "presence"])]
            metadata_events.bind([component.change for component in enhancer_label_inputs], enhancer_metadata_sources, lambda s, values: refresh_prompt_enhancer_labels(s, values, tab_id=tab_id), state, prompt_enhancer_mode_dropdown)
            video_prompt_type_video_guide.input(fn=refresh_video_prompt_type_video_guide,     inputs = [state, gr.State(""),   video_prompt_type, video_prompt_type_video_guide,     image_mode, image_mask_guide, image_guide, image_mask, image_prompt_type, audio_prompt_type], outputs = [video_prompt_type, video_guide, video_guide2, video_guide3, image_guide, keep_frames_video_guide, denoising_strength, masking_strength, video_guide_outpainting_col, video_prompt_type_video_mask, video_mask, image_mask, image_mask_guide, mask_expand, image_refs_row, remove_background_images_ref, frames_positions, video_prompt_type_video_custom_dropbox, video_prompt_type_video_custom_checkbox, input_video_strength, magic_mask_image_btn, magic_mask_video_btn, force_control_video_trim], show_progress="hidden", queue=False)
            video_prompt_type_video_guide_alt.input(fn=refresh_video_prompt_type_video_guide, inputs = [state, gr.State("alt"),video_prompt_type, video_prompt_type_video_guide_alt, image_mode, image_mask_guide, image_guide, image_mask, image_prompt_type, audio_prompt_type], outputs = [video_prompt_type, video_guide, video_guide2, video_guide3, image_guide, keep_frames_video_guide, denoising_strength, masking_strength, video_guide_outpainting_col, video_prompt_type_video_mask, video_mask, image_mask, image_mask_guide, mask_expand, image_refs_row, remove_background_images_ref, frames_positions, video_prompt_type_video_custom_dropbox, video_prompt_type_video_custom_checkbox, input_video_strength, magic_mask_image_btn, magic_mask_video_btn, force_control_video_trim], show_progress="hidden", queue=False)
            custom_visibility_inputs = [video_prompt_type, audio_prompt_type, model_mode]
            custom_visibility_outputs = custom_settings_rows + custom_setting_extra_inputs + [custom_guide_row]
            metadata_events.bind([video_prompt_type.change, audio_prompt_type.change], [(model_choice_target, "value"), *[(component, "value") for component in custom_visibility_inputs]], lambda s, values: refresh_custom_settings_visibility(s, *values[1:]), state, custom_visibility_outputs)
            model_mode.input(fn=refresh_custom_settings_visibility, inputs=[state, *custom_visibility_inputs], outputs=custom_visibility_outputs, queue=False, trigger_mode="always_last", show_progress="hidden", api_name=False)
            # video_prompt_type_video_guide_alt.input(fn=refresh_video_prompt_type_video_guide_alt, inputs = [state, video_prompt_type, video_prompt_type_video_guide_alt, image_mode, image_mask_guide, image_guide, image_mask], outputs = [video_prompt_type, video_guide, image_guide, image_refs_row, denoising_strength, masking_strength, video_mask, mask_expand, image_mask_guide, image_guide, image_mask, keep_frames_video_guide ], show_progress="hidden")
            video_prompt_type_video_custom_dropbox.input(fn= refresh_video_prompt_type_video_custom_dropbox, inputs=[state, video_prompt_type, video_prompt_type_video_custom_dropbox], outputs = video_prompt_type)
            video_prompt_type_video_custom_checkbox.input(fn= refresh_video_prompt_type_video_custom_checkbox, inputs=[state, video_prompt_type, video_prompt_type_video_custom_checkbox], outputs = video_prompt_type)
            # image_mask_guide.upload(fn=update_image_mask_guide, inputs=[state, image_mask_guide], outputs=[image_mask_guide], show_progress="hidden")
            video_prompt_type_video_mask.input(fn=refresh_video_prompt_type_video_mask, inputs = [state, video_prompt_type, video_prompt_type_video_mask, image_mode, image_mask_guide, image_guide, image_mask], outputs = [video_prompt_type, video_mask, image_mask_guide, image_guide, image_mask, mask_expand, masking_strength, magic_mask_image_btn, magic_mask_video_btn], show_progress="hidden")
            video_prompt_type_alignment.input(fn=refresh_video_prompt_type_alignment, inputs = [state, video_prompt_type, video_prompt_type_alignment], outputs = [video_prompt_type])
            main.load(fn=refresh_prompt_labels, inputs=[state, multi_prompts_gen_type, image_mode], outputs=[prompt, wizard_prompt, image_end, prompt_info_label, wizard_prompt_info_label], show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
            multi_prompts_gen_type.select(fn=refresh_prompt_labels, inputs=[state, multi_prompts_gen_type, image_mode], outputs=[prompt, wizard_prompt, image_end, prompt_info_label, wizard_prompt_info_label], show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
            video_guide_outpainting_top.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_top, gr.State(0)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_bottom.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_bottom,gr.State(1)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_left.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_left,gr.State(2)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_right.input(fn=update_video_guide_outpainting, inputs=[video_guide_outpainting, video_guide_outpainting_right,gr.State(3)], outputs = [video_guide_outpainting], trigger_mode="multiple" )
            video_guide_outpainting_ratio.input(fn=refresh_video_guide_outpainting_labels, inputs=[video_guide_outpainting_ratio], outputs=[video_guide_outpainting_top, video_guide_outpainting_bottom, video_guide_outpainting_left, video_guide_outpainting_right], show_progress="hidden")
            video_guide_outpainting_checkbox.input(fn=refresh_video_guide_outpainting_row, inputs=[video_guide_outpainting_checkbox, video_guide_outpainting], outputs= [video_guide_outpainting_row, video_guide_outpainting_ratio, video_guide_outpainting])
            show_advanced.change(fn=switch_advanced, inputs=[state, show_advanced, lset_name], outputs=[advanced_row, preset_buttons_rows, refresh_lora_btn, refresh2_row ,lset_name]).then(
                fn=switch_prompt_type, inputs = [state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars], outputs = [wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, prompt_column_advanced, prompt_column_wizard, prompt_column_wizard_vars, *prompt_vars]).then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
            gr.on( triggers=[output.change, output.select],fn=select_media, inputs=[state, current_gallery_tab, output, last_choice, gr.State(-1), gr.State("video"), PP_spatial_upsampling, PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_spatial_upsampler_help_target, PP_temporal_upsampling, video_info_tab], outputs=select_media_outputs, show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
            # gr.on( triggers=[output.change, output.select], fn=select_media, inputs=[state, output, last_choice, audio_files_paths, audio_file_selected, gr.State("video")], outputs=[last_choice, video_info, video_buttons_row, image_buttons_row, audio_buttons_row, video_postprocessing_tab, audio_remuxing_tab], show_progress="hidden")
            for strip_source, strip_gallery in zip(STRIP_SOURCES, strip_galleries.values()):
                strip_gallery.state_selected.change(fn=select_media, inputs=[state, current_gallery_tab, output, last_choice, strip_gallery.state_selected, gr.State(strip_source.id), PP_spatial_upsampling, PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_spatial_upsampler_help_target, PP_temporal_upsampling, video_info_tab], outputs=select_media_outputs, show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)

            preview_trigger.change(refresh_preview, inputs= [state], outputs= [preview], show_progress="hidden")
            WangpProgress.bind(download_lora_btn.click, download_lora, inputs=[state, lora_url], outputs=[lora_url], component=lora_progress).then(fn=refresh_lora_list, inputs=[state, lset_name,loras_choices], outputs=[lset_name, loras_choices], show_progress="hidden")
            def activate_status(state):
                if state.get("validate_success",0) != 1:
                    return
                gen = get_gen_info(state)
                set_generation_state(gen, status_display=True)

            start_quit_timer_js, cancel_quit_timer_js, trigger_zip_download_js, trigger_settings_download_js, click_brush_js = get_js()

            if tab_id == 'generate':
                from shared.deepy.hybrid_ui import bind_gallery_sync
                bind_gallery_sync(_deepy_hybrid, state, refresh_gallery, [gallery_tabs, current_gallery_tab, output, last_choice, *strip_states, gen_info, generate_btn, add_to_queue_btn, current_gen_column, current_gen_buttons_row, queue_html, abort_btn, earlystop_btn, onemorewindow_btn, gen_info_accordion], gallery=output, strip_states=strip_states, main=main)
                from shared.gradio import world_explorer
                world_explorer.bind(state, lambda state_value: workspace_storage.output_config(server_config, state=state_value)['save_path'], get_model_def, get_model_handler, explore_world_outside_queue,
                                    server_config, lambda choices: update_config(server_config, server_config_filename, {"world_explorer": choices}), world_memory_profile)
                output_trigger.change(refresh_gallery,
                    inputs = [state], 
                    outputs = [gallery_tabs, current_gallery_tab, output, last_choice, *strip_states, gen_info, generate_btn, add_to_queue_btn, current_gen_column, current_gen_buttons_row, queue_html, abort_btn, earlystop_btn, onemorewindow_btn, gen_info_accordion],
                    show_progress="hidden"
                    )


            modal_action_trigger.click(
                fn=show_modal_image,
                inputs=[state, modal_action_input],
                outputs=[modal_html_display, modal_container],
                show_progress="hidden"
            )
            close_modal_trigger_btn.click(
                fn=lambda: (gr.HTML(value=""), gr.Column(visible=False)),
                outputs=[modal_html_display, modal_container],
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
                    get_model_def=lambda state_value: get_model_def(get_state_model_type(state_value)),
                    model_def=model_def,
                )
            pause_btn.click(pause_generation, [state], show_progress="hidden", concurrency_limit=None, trigger_mode="multiple")
            resume_btn.click(resume_generation, [state], queue=False)
            from shared.deepy.hybrid_ui import bind_generation_state
            bind_generation_state(tab_id, gen_status, pause_btn, resume_btn)
            abort_btn.click(abort_generation, [state, gr.State("")], [ abort_btn, queue_html], show_progress="hidden" ) #.then(refresh_gallery, inputs = [state, gen_info], outputs = [output, gen_info, queue_html] )
            abort_client_id.change(abort_generation, [state, abort_client_id], [ abort_btn, queue_html], show_progress="hidden" ) #.then(refresh_gallery, inputs = [state, gen_info], outputs = [output, gen_info, queue_html] )
            earlystop_btn.click(early_stop_generation, [state], [ earlystop_btn] )
            onemoresample_btn.click(fn=one_more_sample,inputs=[state], outputs= [state])
            onemorewindow_btn.click(fn=one_more_window,inputs=[state], outputs= [state])

            inputs_names= list(inspect.signature(save_inputs).parameters)[1:-2]
            locals_dict = locals()
            gen_inputs = build_save_inputs(inputs_names, locals_dict, state, plugin_data, custom_setting_extra_inputs)
            refresh_outputs = build_form_refresh_outputs(inputs_names, locals_dict, state, plugin_data, extra_inputs)
            save_settings_btn.click( fn=validate_wizard_prompt, inputs =[state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] , outputs= [prompt]).then(
                save_inputs, inputs =[target_settings] + gen_inputs, outputs = [])

            gr.on( triggers=[video_info_extract_settings_btn.click, video_info_extract_image_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, output, last_choice, gr.State("video")] , outputs= [refresh_form_trigger, model_choice_target])

            gr.on( triggers=[video_info_extract_audio_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                show_progress="hidden",
                outputs= [prompt],
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, audio_files_paths, audio_file_selected, gr.State("audio")] , outputs= [refresh_form_trigger, model_choice_target])

            gr.on( triggers=[video_info_extract_mesh_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                show_progress="hidden",
                outputs= [prompt],
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, mesh_files_paths, mesh_file_selected, gr.State(MESH.id)] , outputs= [refresh_form_trigger, model_choice_target])

            gr.on( triggers=[video_info_extract_world_settings_btn.click], fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                show_progress="hidden",
                outputs= [prompt],
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then( fn=use_video_settings, inputs =[state, world_files_paths, world_file_selected, gr.State(WORLD.id)] , outputs= [refresh_form_trigger, model_choice_target])

            if tab_id == 'generate':
                from shared.deepy.hybrid_ui import bind_workspace_extract
                bind_workspace_extract(_deepy_hybrid, state, validate_wizard_prompt, [state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars],
                    save_inputs, [target_state] + gen_inputs, use_video_settings, [refresh_form_trigger, model_choice_target])

            enhance_prompt_inputs = [state, prompt, alt_prompt, prompt_enhancer, multi_images_gen_type, multi_prompts_gen_type, overrides, video_prompt_type, image_prompt_type, audio_prompt_type]
            enhance_prompt_sinks = [gr.State(), gr.State()]
            enhance_prompt_outputs = [[prompt, enhance_prompt_sinks[0], wizard_prompt], [enhance_prompt_sinks[0], alt_prompt, enhance_prompt_sinks[1]], [prompt, alt_prompt, wizard_prompt]]
            enhance_prompt_triggers = [gr.Text(visible=False) for _ in range(3)]

            def prompt_enhancer_ui(mode, busy):
                output_fields = {step.output_field for step in prompt_enhancer_chaining.parse_mode(mode)}
                fields = [gr.update(interactive=not busy) if field in output_fields else gr.update() for field in ("prompt", "alt_prompt", "prompt")]
                return fields

            enhance_prompt_ui_outputs = [prompt, alt_prompt, wizard_prompt]
            for trigger, outputs in zip(enhance_prompt_triggers, enhance_prompt_outputs):
                WangpProgress.bind(trigger.change, enhance_prompt, inputs=enhance_prompt_inputs, outputs=outputs, component=prompt_enhancer_progress, hide=[prompt_enhancer_row]).then(
                    fn=lambda mode: prompt_enhancer_ui(mode, False), inputs=prompt_enhancer, outputs=enhance_prompt_ui_outputs, show_progress="hidden",
                )

            def route_prompt_enhancer(mode):
                output_fields = {step.output_field for step in prompt_enhancer_chaining.parse_mode(mode)}
                target = 2 if len(output_fields) > 1 else int("alt_prompt" in output_fields)
                triggers = [str(time.time_ns()) if index == target else gr.update() for index in range(3)]
                return [*triggers, *prompt_enhancer_ui(mode, True)]

            prompt_enhancer_btn.click(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=route_prompt_enhancer, inputs=prompt_enhancer, outputs=[*enhance_prompt_triggers, *enhance_prompt_ui_outputs], show_progress="hidden")

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
                        get_session_ui_settings=_deepy.get_session_ui_settings,
                        update_session_ui_settings=_deepy.update_session_ui_settings,
                        list_saved_sessions=_deepy.list_saved_sessions,
                        resume_saved_session=_deepy.resume_saved_session,
                        prefill_restored_session_context=_deepy.prefill_restored_session_context,
                        rename_saved_session=_deepy.rename_saved_session,
                        duplicate_saved_session=_deepy.duplicate_saved_session,
                        export_saved_session=_deepy.export_saved_session,
                        import_saved_session=_deepy.import_saved_session,
                        delete_saved_session=_deepy.delete_saved_session,
                    ),
                )
                main.load(
                    fn=lambda state_value: (assistant_chat.build_sync_event(service_for(state_value)._session), gr.update(), gr.update(), gr.update()) if service_for(state_value) is not None else _deepy.browser_session_started(state_value),
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
            video_info_to_image_guide_btn.click(fn=image_to_ref_image_guide, inputs =[state, output, last_choice], outputs = [image_guide, image_mask_guide, image_mask]).then(fn=None, inputs=[], outputs=[], js=click_brush_js )
            video_info_to_image_mask_btn.click(fn=image_to_ref_image_set, inputs =[state, output, last_choice, image_mask, gr.State("Image Mask")], outputs = [image_mask] )
            video_info_to_reference_image_btn.click(fn=image_to_ref_image_add, inputs =[state, output, last_choice, image_refs, gr.State("Ref Image")],  outputs = [image_refs] )

            gr.on(triggers=[video_info_eject_audio_btn.click, video_info_eject_audio2_btn.click, video_info_eject_deleted_audio_btn.click], fn=eject_from_strip_gallery, inputs =[state, gr.State("audio"), audio_files_paths, audio_file_selected], outputs = [audio_files_paths, audio_file_selected, audio_gallery_refresh_trigger, video_info, audio_buttons_row] )
            gr.on(triggers=[video_info_eject_mesh_btn.click, video_info_eject_deleted_mesh_btn.click], fn=eject_from_strip_gallery, inputs =[state, gr.State(MESH.id), mesh_files_paths, mesh_file_selected], outputs = [mesh_files_paths, mesh_file_selected, mesh_gallery_refresh_trigger, video_info, mesh_buttons_row] )
            video_info_to_mesh_input_btn.click(fn=mesh_to_custom_guide, inputs =[state, mesh_files_paths, mesh_file_selected, video_prompt_type, audio_prompt_type, model_mode], outputs = [custom_guide, model_mode, *custom_visibility_outputs] )
            gr.on(triggers=[video_info_eject_world_btn.click, video_info_eject_deleted_world_btn.click], fn=eject_from_strip_gallery, inputs =[state, gr.State(WORLD.id), world_files_paths, world_file_selected], outputs = [world_files_paths, world_file_selected, world_gallery_refresh_trigger, video_info, world_buttons_row] )
            video_info_continue_world_btn.click(fn=world_to_custom_guide, inputs =[state, world_files_paths, world_file_selected], outputs = [custom_guide] )
            gr.on(triggers=[video_info_eject_app_btn.click, video_info_eject_deleted_app_btn.click], fn=eject_from_strip_gallery, inputs =[state, gr.State(APP.id), app_files_paths, app_file_selected], outputs = [app_files_paths, app_file_selected, app_gallery_refresh_trigger, video_info, app_buttons_row] )
            video_info_to_audio_guide_btn.click(fn=audio_to_source_set, inputs =[state, audio_files_paths, audio_file_selected, gr.State("Audio Source")], outputs = [audio_guide] )
            video_info_to_audio_guide2_btn.click(fn=audio_to_source_set, inputs =[state, audio_files_paths, audio_file_selected, gr.State("Audio Source 2")], outputs = [audio_guide2] )
            video_info_to_audio_source_btn.click(fn=audio_to_source_set, inputs =[state, audio_files_paths, audio_file_selected, gr.State("Custom Audio")], outputs = [audio_source] )

            video_info_postprocessing_btn.click(fn=apply_post_processing, inputs =[state, output, last_choice, PP_temporal_upsampling, PP_spatial_upsampling, PP_film_grain_intensity, PP_film_grain_saturation, PP_spatial_postprocessing_parameters, PP_temporal_postprocessing_parameters, PP_postprocessing_reference_images, seed], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
            video_info_audio_postprocessing_btn.click(fn=postprocess_audio_file, inputs =[state, audio_files_paths, audio_file_selected, PP_late_audio_postprocess, PP_late_audio_replace_voice_sample, PP_late_audio_replace_voice_sample2], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
            video_info_remux_audio_btn.click(fn=remux_audio, inputs =[state, output, last_choice, PP_postprocess_audio, PP_postprocess_audio_prompt, PP_postprocess_audio_neg_prompt, PP_postprocess_audio_seed, PP_repeat_generation, PP_custom_audio, PP_replace_voice_sample, PP_replace_voice_sample2], outputs = [mode, generate_trigger, add_to_queue_trigger ] )
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
                inputs=[state, wizard_prompt_activated_var, lset_name,loras_choices, loras_multipliers, prompt], outputs=[wizard_prompt_activated_var, loras_choices, loras_multipliers, prompt, fill_wizard_prompt_trigger, refresh_form_trigger, model_choice_target])
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
                inputs =[state, export_settings_include_media],
                outputs= [settings_download_payload]
            ).then(
                fn=None,
                inputs=[settings_download_payload],
                outputs=[settings_download_payload],
                js=trigger_settings_download_js
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=[settings_download_payload],
                show_progress="hidden"
            )
            
            tab_prompt_save_index = gen_inputs.index(prompt)

            def switch_image_mode_form(evt: gr.SelectData, state_value, wizard_active, wizard_variables, current_prompt, wizard_text, *values):
                record_image_mode_tab(state_value, evt)
                validated_prompt = validate_wizard_prompt(state_value, wizard_active, wizard_variables, current_prompt, wizard_text, *values[:len(prompt_vars)])
                save_values = list(values[len(prompt_vars):])
                save_values[1 + tab_prompt_save_index] = validated_prompt
                save_inputs(*save_values)
                return validated_prompt, switch_image_mode(state_value)

            image_mode_tabs.select(fn=switch_image_mode_form,
                inputs=[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars, target_state, *gen_inputs],
                outputs=[prompt, refresh_form_trigger], show_progress="hidden")

            settings_file.upload(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars] ,
                outputs= [prompt],
                show_progress="hidden",
            ).then(fn=save_inputs,
                inputs =[target_state] + gen_inputs,
                outputs= None
            ).then(fn=load_settings_from_file, inputs =[state, settings_file] , outputs= [refresh_form_trigger, model_choice_target, settings_file])


            if tab_id == 'generate':
                goto_model_type_fn = goto_model_type_with_filter if model_filter is not None else goto_model_type
                goto_model_type_outputs = [model_family, model_base_type_choice, model_choice, refresh_form_trigger, model_filter.refresh_trigger] if model_filter is not None else [model_family, model_base_type_choice, model_choice, refresh_form_trigger]
                def restore_recorded_form(state_value):
                    model_type = get_state_model_type(state_value)
                    recorded = service_for(state_value).load_model_form(model_type)
                    if recorded is None:
                        return (gr.update(),) * len(refresh_outputs)
                    set_model_settings(state_value, model_type, recorded)
                    return fill_inputs(state_value)

                main.load(restore_recorded_form, inputs=[state], outputs=refresh_outputs, queue=False, show_progress="hidden").then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)
                prompt_save_index = gen_inputs.index(prompt)
                finetune_outputs = [model_toolbar.finetune_button] if model_toolbar is not None and model_toolbar.finetune_button is not None else []

                def switch_model_form(state_value, model_target, wizard_active, wizard_variables, current_prompt, wizard_text, *values):
                    validated_prompt = validate_wizard_prompt(state_value, wizard_active, wizard_variables, current_prompt, wizard_text, *values[:len(prompt_vars)])
                    save_values = list(values[len(prompt_vars):])
                    save_values[1 + prompt_save_index] = validated_prompt
                    save_inputs(*save_values)
                    dropdown_updates = goto_model_type_fn(state_value, model_target)
                    description, _ = change_model(state_value, _model_choice_target_model_type(model_target), refresh_header=False)
                    finetune_updates = [finetune_editor.toolbar_button_updates(_get_finetune_editor_deps(), get_state_model_type(state_value))] if finetune_outputs else []
                    return [*dropdown_updates, description, *fill_inputs(state_value), *finetune_updates]

                switch_model_form._wangp_model_switch_tail = True
                model_choice_target.change(fn=switch_model_form,
                    inputs=[state, model_choice_target, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars, target_state, *gen_inputs],
                    outputs=[*goto_model_type_outputs, model_description, *refresh_outputs, *finetune_outputs],
                    trigger_mode="always_last",
                    show_progress="full" if args.debug_gen_form else "hidden",
                ).then(
                    fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS
                ).then(fn= preload_model_when_switching, 
                    inputs=[state],
                    outputs=[gen_status],
                    show_progress="hidden")

            if tab_id == 'generate' and model_toolbar is not None:
                model_selector_toolbar.bind_toolbar(
                    model_toolbar,
                    deps_factory=_get_dropdown_deps,
                    state=state,
                    model_family=model_family,
                    model_base_type_choice=model_base_type_choice,
                    model_choice=model_choice,
                    model_choice_target=model_choice_target,
                    refresh_form_trigger=refresh_form_trigger,
                    lset_name=lset_name,
                    loras_choices=loras_choices,
                    refresh_model_defs=refresh_model_defs,
                    refresh_model_dropdowns=refresh_model_dropdowns,
                    refresh_lora_list=refresh_lora_list,
                    unload_handler=lambda state_value: model_selector_toolbar.unload_models_from_ram(
                        state_value,
                        server_config=server_config,
                        any_GPU_process_running=any_GPU_process_running,
                        release_deepy_vram=release_deepy_vram,
                        reset_prompt_enhancer=reset_prompt_enhancer,
                        reset_prompt_enhancer_if_requested=reset_prompt_enhancer_if_requested,
                        release_extensions=release_extension_offloadobjs,
                        release_model=release_model,
                    ),
                )
                if model_toolbar.finetune_button is not None:
                    finetune_editor_ui = finetune_editor.create_editor()
                    finetune_editor.bind_editor(
                        finetune_editor_ui,
                        deps_factory=_get_finetune_editor_deps,
                        state=state,
                        toolbar_button=model_toolbar.finetune_button,
                        bind_model_change=False,
                        model_family=model_family,
                        model_base_type_choice=model_base_type_choice,
                        model_choice=model_choice,
                        model_choice_target=model_choice_target,
                        refresh_form_trigger=refresh_form_trigger,
                        model_description=model_description,
                        header=header,
                        validate_wizard_prompt=validate_wizard_prompt,
                        wizard_inputs=[state, wizard_prompt_activated_var, wizard_variables_var, prompt, wizard_prompt, *prompt_vars],
                        prompt_output=prompt,
                        save_inputs_handler=save_inputs,
                        target_state=target_state,
                        generation_inputs=gen_inputs,
                    )

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
            ).then(fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS)

            refresh_form_trigger.change(fn= fill_inputs, 
                inputs=[state],
                outputs=refresh_outputs,
                show_progress= "full" if args.debug_gen_form else "hidden",
            ).then(
                fn=None, inputs=None, outputs=None, js=PROMPT_TOOLS_ATTACH_JS
            ).then(fn=validate_wizard_prompt,
                inputs= [state, wizard_prompt_activated_var, wizard_variables_var,  prompt, wizard_prompt, *prompt_vars],
                outputs= [prompt],
                show_progress="hidden",
            )                

            if tab_id == 'generate':
                # main_tabs.select(fn=detect_auto_save_form, inputs= [state], outputs= save_form_trigger, trigger_mode="multiple")
                model_family.select(fn=change_model_family_target, inputs=[state], outputs=[model_choice_target], show_progress="hidden", queue=False)
                model_base_type_choice.select(fn=change_model_base_types_target, inputs=[state], outputs=[model_choice_target], show_progress="hidden", queue=False)
                if model_filter is not None:
                    model_output_filter.bind_filter(model_filter, deps_factory=_get_dropdown_deps, state=state, model_choice_target=model_choice_target, current_model_type_getter=get_state_model_type)
                refresh_models_trigger.change(fn=refresh_model_dropdowns, inputs=[state], outputs=[model_family, model_base_type_choice, model_choice, refresh_form_trigger], show_progress="hidden")

                model_choice.select(fn=change_model_choice_target, outputs=[model_choice_target], show_progress="hidden", queue=False)
            
                generate_btn.click(fn = init_generate, inputs = [state, output, last_choice, *strip_selected], outputs=[generate_trigger, mode])
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
                ).then(fn=prepare_generate_media,
                    inputs= [state],
                    outputs= [generate_btn, add_to_queue_btn, current_gen_column, onemorewindow_btn, earlystop_btn]
                ).then(fn=activate_status,
                    inputs= [state],
                    outputs=None,
                ).then(fn=process_tasks,
                    inputs= [state],
                    outputs= [preview_trigger, output_trigger, refresh_models_trigger], 
                    show_progress="hidden",
                ).then(finalize_generation,
                    inputs= [state], 
                    outputs= [gallery_tabs, current_gallery_tab, output, *strip_states, abort_btn, earlystop_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info]
                ).then(
                    fn=lambda s: gr.Accordion(open=False) if len(get_gen_info(s).get("queue", [])) <= 1 else gr.update(),
                    inputs=[state],
                    outputs=[queue_accordion]
                ).then(unload_model_if_needed,
                    inputs= [state], 
                    outputs= []
                )

                # Hybrid startup restores the queue once in the server lifespan.
                # A new browser must not restart an aborted or paused queue.
                queue_load_triggers = [load_queue_btn.upload, load_queue_trigger.change]
                if service_for(state.value) is None:
                    queue_load_triggers.append(main.load)
                gr.on(triggers=queue_load_triggers,
                    fn=load_queue_action,
                    inputs=[load_queue_btn, state],
                    outputs=[queue_html]
                ).then(
                     fn=lambda s: gr.Accordion(open=True) if get_gen_info(s).get("queue", []) else gr.update(),
                     inputs=[state],
                     outputs=[queue_accordion]
                ).then(
                    fn=init_process_queue_if_any,
                    inputs=[state],
                    outputs=[generate_btn, add_to_queue_btn, current_gen_column, onemorewindow_btn, earlystop_btn]
                ).then(fn=activate_status,
                    inputs= [state],
                    outputs=None,
                ).then(
                    fn=process_tasks,
                    inputs=[state],
                    outputs=[preview_trigger, output_trigger, refresh_models_trigger],
                    trigger_mode="once"
                ).then(
                    fn=finalize_generation_with_state,
                    inputs=[state],
                    outputs=[gallery_tabs, current_gallery_tab, output, *strip_states, abort_btn, earlystop_btn, generate_btn, add_to_queue_btn, current_gen_column, gen_info, queue_accordion, state],
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
                outputs=[queue_zip_download_payload]
            ).then(
                fn=None,
                inputs=[queue_zip_download_payload],
                outputs=[queue_zip_download_payload],
                js=trigger_zip_download_js
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=[queue_zip_download_payload],
                show_progress="hidden"
            )

            clear_queue_btn.click(
                fn=clear_queue_action,
                inputs=[state],
                outputs=[queue_html]
            ).then(
                 fn=lambda: gr.Accordion(open=False),
                 inputs=None,
                 outputs=[queue_accordion]
            )
    locals_dict = locals()
    if update_form:
        gen_inputs = build_form_refresh_outputs(inputs_names, locals_dict, state_dict, plugin_data, extra_inputs)
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
        get_lora_local_path=get_lora_local_path,
        get_parent_model_type=get_parent_model_type,
        get_base_model_type=get_base_model_type,
        get_model_family=get_model_family,
        get_model_name=get_model_name,
        get_transformer_dtype=get_transformer_dtype,
        list_model_defs=list_model_defs,
    )

def _get_finetune_editor_deps():
    return finetune_editor.FinetuneEditorDeps(
        settings_version=settings_version,
        families_infos=families_infos,
        transformer_types=transformer_types,
        displayed_model_types=displayed_model_types,
        three_levels_hierarchy=three_levels_hierarchy,
        get_model_def=get_model_def,
        get_model_name=get_model_name,
        get_base_model_type=get_base_model_type,
        get_parent_model_type=get_parent_model_type,
        get_model_family=get_model_family,
        get_state_model_type=get_state_model_type,
        get_model_settings=get_model_settings,
        set_model_settings=set_model_settings,
        get_default_settings=get_default_settings,
        get_settings_file_name=get_settings_file_name,
        get_lora_dir=get_lora_dir,
        get_lora_URL=get_lora_URL,
        update_loras_url_cache=update_loras_url_cache,
        refresh_model_defs=refresh_model_defs,
        refresh_model_dropdowns=refresh_model_dropdowns,
        change_model=change_model,
        request_reload_if_loaded=request_reload_if_loaded,
    )

def create_models_hierarchy(rows):
    return model_dropdowns.create_models_hierarchy(rows)

def create_models_selector_hierarchy(dropdown_types=None):
    return model_dropdowns.create_models_selector_hierarchy(_get_dropdown_deps(), dropdown_types)

def get_sorted_dropdown(dropdown_types, current_model_family, current_model_type, three_levels = True):
    return model_dropdowns.get_sorted_dropdown(_get_dropdown_deps(), dropdown_types, current_model_family, current_model_type, three_levels)

def generate_dropdown_model_list(current_model_type, state=None):
    return model_dropdowns.generate_dropdown_model_list(_get_dropdown_deps(), current_model_type, state)

def change_model_family_target(state, evt: gr.SelectData):
    current_model_family = evt.value
    _, model_choice_update = model_dropdowns.change_model_family(_get_dropdown_deps(), state, current_model_family)
    target = model_choice_update.constructor_args["value"]
    model_dropdowns.debug_model_selector_event("selector.family.select", family=current_model_family, target=target, filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state))
    return _model_choice_target_value(target)

def change_model_base_types_target(state, evt: gr.SelectData):
    model_base_type_choice = evt.value
    current_model_family = _get_dropdown_deps().get_model_family(model_base_type_choice, for_ui=True)
    _, model_choice_update = model_dropdowns.change_model_base_types(_get_dropdown_deps(), state, current_model_family, model_base_type_choice)
    target = model_choice_update.constructor_args["value"]
    model_dropdowns.debug_model_selector_event("selector.base.select", family=current_model_family, base=model_base_type_choice, target=target, filter=model_dropdowns.get_model_output_filter(_get_dropdown_deps(), state))
    return _model_choice_target_value(target)

def change_model_choice_target(evt: gr.SelectData):
    model_dropdowns.debug_model_selector_event("selector.model.select", target=evt.value)
    return _model_choice_target_value(evt.value)

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
    (payload) => window.WanGPDownloads.trigger(payload)
    """

    trigger_settings_download_js = """
    (payload) => window.WanGPDownloads.trigger(payload)
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
    global transformer_type, custom_setting_visible
    if not args.lock_model:
        transformer_type = model_dropdowns.select_model_for_output_filter(_get_dropdown_deps(), server_config, transformer_type)
    gradio_downloads.install_routes()
    gallery_files.install(deepy_filesystem.build_file_access_policy(server_config))
    # Load CSS from external file
    css_path = os.path.join(os.path.dirname(__file__), "shared", "gradio", "ui_styles.css")
    with open(css_path, "r", encoding="utf-8") as f:
        css = f.read()
    css += "\n" + WangpProgress.css()
    css += "\n" + Path(__file__).with_name("shared").joinpath("gradio", "image_zoom.css").read_text(encoding="utf-8")
    css += "\n" + MagicMaskUI.get_css()
    css += "\n" + assistant_chat.get_css()
    css += "\n" + model_infos.get_css()
    css += "\n" + field_help.get_css()
    css += "\n" + collect_prompt_helper_assets("get_prompt_helper_css")
    css += "\n" + model_output_filter.get_css()
    css += "\n" + model_selector_toolbar.get_css()
    css += "\n" + finetune_editor.get_css()
    local_file_picker.configure_last_directory_store(server_config)
    UI_theme = server_config.get("UI_theme", "default")
    UI_theme  = args.theme if len(args.theme) > 0 else UI_theme
    theme = ui_studio.create_theme(UI_theme, spacing_size=theme_spacing_size, radius_size=theme_radius_size, text_size=theme_text_size)

    # Load main JS from external file
    js_path = os.path.join(os.path.dirname(__file__), "shared", "gradio", "ui_scripts.js")
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    js += StripGallery.get_javascript()
    js += MagicMaskUI.get_javascript()
    js += assistant_chat.get_javascript()
    js += model_infos.get_javascript()
    js += field_help.get_javascript()
    js += collect_prompt_helper_assets("get_prompt_helper_javascript")
    js += model_output_filter.get_javascript()
    js += model_selector_toolbar.get_javascript()
    js += finetune_editor.get_javascript()
    StripGallery.install_gradio_upload_mtime_patch()
    app.initialize_plugins(globals())
    custom_setting_visible = compatible_custom_setting_visible(custom_setting_visible)
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

    with ui_studio.Blocks(css=css, js=js, theme=theme, title="WanGP", fill_width=True) as main:
        gr.Markdown(f'<div align=center><H1>Wan<SUP style="color: #2563eb;">GP</SUP> v{WanGP_version} <FONT SIZE=4>by <I>DeepBeepMeep</I></FONT> <FONT SIZE=3>') # (<A HREF='https://github.com/deepbeepmeep/Wan2GP'>Updates</A>)</FONT SIZE=3></H1></div>")
        global model_list

        tab_state = gr.State({ "tab_no":0 }) 
        target_edit_state = gr.Text(value = "edit_state", interactive= False, visible= False)
        edit_queue_trigger = gr.Text(value='', interactive= False, visible=False)
        with gr.Tabs(selected="media_gen", ) as main_tabs:
            # JS keepalive patch targets the Gradio tab id "media_gen".
            with gr.Tab("Media Generator", id="media_gen") as media_generator_tab:
                model_toolbar = None
                model_filter = None
                with gr.Row():
                    if args.lock_model:    
                        gr.Markdown("<div class='title-with-lines'><div class=line></div><h2>" + get_model_name(transformer_type) + "</h2><div class=line></div>")
                        model_family = gr.Dropdown(visible=False, value= "")
                        model_choice = gr.Dropdown(visible=False, value= transformer_type, choices= [transformer_type])
                    else:
                        model_filter = model_output_filter.create_filter(model_dropdowns.get_model_output_filter(_get_dropdown_deps()), model_dropdowns.available_output_filters(_get_dropdown_deps()))
                        model_family, model_base_type_choice, model_choice = generate_dropdown_model_list(transformer_type)
                        model_toolbar = model_selector_toolbar.create_toolbar(is_finetune_editor=finetune_editor.is_finetune_model(_get_finetune_editor_deps(), transformer_type))
                if model_toolbar is not None:
                    model_selector_toolbar.create_search_panel(model_toolbar)
                with gr.Row():
                    with gr.Column():
                        with gr.Group(elem_classes="header-markdown-group"):
                            header_settings = get_default_settings(transformer_type)
                            description_html, header_html = generate_header(transformer_type, compile, attention_mode, header_settings.get("overrides", ""), header_settings.get("guidance_phases", 1), header_settings.get("video_prompt_type", ""))
                            model_description = gr.HTML(description_html, visible= True)
                            header = gr.Markdown(header_html, visible= True)
                    if stats_app is not None:
                        stats_element = stats_app.get_gradio_element()

                with gr.Row():
                    generator_tab_components = generate_media_tab(
                        model_family=model_family,
                        model_base_type_choice=model_base_type_choice,
                        model_choice=model_choice,
                        model_description=model_description,
                        header=header,
                        main=main,
                        main_tabs=main_tabs,
                        tab_id='generate',
                        model_toolbar=model_toolbar,
                        model_filter=model_filter,
                    )
                    (state, loras_choices, lset_name, resolution, refresh_form_trigger, save_form_trigger) = generator_tab_components['state'], generator_tab_components['loras_choices'], generator_tab_components['lset_name'], generator_tab_components['resolution'], generator_tab_components['refresh_form_trigger'], generator_tab_components['save_form_trigger']
            with gr.Tab("Edit", id="edit", visible=False) as edit_tab:
                edit_title_md = gr.Markdown()
                edit_tab_components = generate_media_tab(
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
                final_edit_inputs = build_save_inputs(edit_inputs_names, edit_tab_components, edit_tab_components['state'], edit_tab_components['plugin_data'], edit_tab_components['custom_setting_extra_inputs'])
                edit_tab_inputs = build_form_refresh_outputs(edit_inputs_names, edit_tab_components, edit_tab_components['state'], edit_tab_components['plugin_data'], edit_tab_components['extra_inputs'])

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
                    if _is_edit_task_params(task.get("params", {})):
                        gr.Info("Post-processing tasks cannot be edited.")
                        state["editing_task_id"] = None
                        return default_return

                    prompt_text = task.get('prompt', 'Unknown Prompt')[:80]
                    edit_title_text = f"<div align='center'><h2>Editing task ID {editing_task_id}: '{prompt_text}...'</h2></div>"
                    ui_defaults=task['params'].copy()
                    state["edit_model_type"] = ui_defaults["model_type"] 
                    all_new_component_values = generate_media_tab(update_form=True, state_dict=state, ui_defaults=ui_defaults, tab_id='edit', )
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

            media_generator_tab.select(lambda state: state.update({"active_form": "add"}), inputs=state)
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

def _mcp_forwarded_wgp_args():
    mcp_value_args = {"--mcp-transport", "--mcp-host", "--mcp-port", "--mcp-api-version"}
    forwarded = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--mcp", "--mcp-console-output", "--mcp-allow-read-file-system", "--mcp-async"}:
            continue
        if arg in mcp_value_args:
            skip_next = True
            continue
        if any(arg.startswith(f"{name}=") for name in mcp_value_args):
            continue
        forwarded.append(arg)
    return forwarded

def run_mcp_server():
    from types import SimpleNamespace
    from shared.mcp_server import build_server, run_server

    output_dir = args.output_dir.strip() if isinstance(args.output_dir, str) else args.output_dir
    mcp_args = SimpleNamespace(
        root=wgp_root,
        config=args.config.strip() or None,
        output_dir=output_dir or None,
        cli_arg=_mcp_forwarded_wgp_args(),
        console_output=args.mcp_console_output,
        transport=args.mcp_transport,
        mcp_api_version=args.mcp_api_version,
        mcp_async=args.mcp_async,
        host=args.mcp_host.strip() or None,
        port=args.mcp_port,
        allow_read_file_system=args.mcp_allow_read_file_system,
    )
    try:
        server = build_server(mcp_args)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return run_server(server, mcp_args)

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

    if args.mcp:
        sys.exit(run_mcp_server())

    if app is None:
        app = WAN2GPApplication()

    if args.ask_deepy or args.deepy_server:
        download_ffmpeg()
        if len(args.output_dir) > 0:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir, exist_ok=True)
            server_config["save_path"] = args.output_dir
            server_config["image_save_path"] = args.output_dir
            server_config["audio_save_path"] = args.output_dir
            server_config["mesh_save_path"] = args.output_dir
            save_path = args.output_dir
            image_save_path = args.output_dir
            audio_save_path = args.output_dir
            mesh_save_path = args.output_dir
            print(f"Output directory: {args.output_dir}")
        server_config["notification_sound_enabled"] = 0
        runner = deepy_cli.run_deepy_cli_session
        if args.deepy_server:
            from shared.deepy.server import run_server
            runner = lambda deps: run_server(deps, args)
        sys.exit(runner(_deepy_runtime_deps()))

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
            server_config["mesh_save_path"] = args.output_dir
            # Keep module-level paths in sync (used by save_video / get_available_filename).
            save_path = args.output_dir
            image_save_path = args.output_dir
            audio_save_path = args.output_dir
            mesh_save_path = args.output_dir
            print(f"Output directory: {args.output_dir}")

        # Headless CLI runs: disable notification sounds to avoid pygame/sounddevice issues.
        server_config["notification_sound_enabled"] = 0

        # Create minimal state with all required fields
        state = {
            "model_type": "",
            "edit_model_type": "",
            "gen": {
                "queue": [],
                "in_progress": False,
                **gallery_state_defaults(0),
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
    from shared.deepy.hybrid import launch_gradio
    launch_gradio(demo, _deepy_hybrid, args,
        favicon_path="favicon.png",
        server_name=server_name,
        server_port=server_port,
        share=args.share,
        allowed_paths=list({save_path, image_save_path, audio_save_path, mesh_save_path, args.workspaces_dir or server_config.get('workspaces_dir') or os.path.join(wgp_root, 'workspaces'), "icons", os.path.join("shared", "gradio", "vendor")}),
    )
    close_application()
