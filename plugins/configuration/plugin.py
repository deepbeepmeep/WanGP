from postprocessing import memory as processor_memory
from shared.kernels import fp8_backend, int8_backend, kernel_policy
from shared import attention_kit, model_dropdowns
from shared.cuda_memory import COMPUTE_CACHE_KEY, COMPUTE_CACHE_PATH_DEFAULT, COMPUTE_CACHE_PATH_KEY, RAM_ALLOCATOR_DEFAULT, RAM_ALLOCATOR_KEY, compute_cache_folder, compute_cache_useful, mmgp_vram_allocator
import gradio as gr
from shared.utils.plugins import WAN2GPPlugin
import copy
import html
import json
import os
from shared.utils.config_store import config_snapshot, update_config
from shared.mcp.config import DEFAULT_PORT, server_port
from shared.gradio.form_sync import GradioForm
from shared.utils.form_sync import Saved
from shared.deepy.config import DEEPY_VOICE_LANGUAGE_KEY, deepy_voice_language_choices, normalize_deepy_voice_language
from shared.deepy.assets import WHISPER_MODEL_CHOICES, WHISPER_MODEL_KEY
from shared.deepy.voice import VOICE_MODE_KEY, VOICE_MODE_CHOICES, voice_mode
from shared.deepy.engine import get_or_create_assistant_session
from shared.gradio import assistant_chat, gradio_queue_focus_patch, model_output_filter, ui_studio
from shared.gradio.hierarchy_selector import HierarchySelector
from shared import notifications
from shared.utils import prompt_parser
from shared.gradio.model_selector_toolbar import unload_models_from_ram
from shared.utils.video_codecs import RGBA_VIDEO_OUTPUT_CHOICES, SDR_VIDEO_CODEC_CHOICES, VIDEO_CONTAINER_CHOICES, validate_video_output_settings
from shared.deepy.config import (
    DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT,
    DEEPY_ALLOW_READ_FILE_SYSTEM_KEY,
    DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
    DEEPY_FILE_SYSTEM_ACCESS_READ,
    DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE,
    DEEPY_FILE_SYSTEM_PATHS_DEFAULT,
    DEEPY_FILE_SYSTEM_PATHS_KEY,
    DEEPY_READ_EVERYWHERE_DEFAULT,
    DEEPY_READ_EVERYWHERE_KEY,
    DEEPY_COMPACTION_TYPE_DEFAULT,
    DEEPY_COMPACTION_TYPE_DISCARD,
    DEEPY_COMPACTION_TYPE_KEY,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    DEEPY_COMPACTION_CHOICE_THINKING,
    DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
    DEEPY_COMPACTION_THINKING_MIN_TOKENS,
    DEEPY_COMPACTION_THINKING_DEFAULT,
    DEEPY_COMPACTION_THINKING_KEY,
    DEEPY_REPETITION_PENALTY_DEFAULT,
    DEEPY_REPETITION_PENALTY_KEY,
    DEEPY_CONTEXT_TOKENS_MIN,
    DEEPY_CONTEXT_TOKENS_DEFAULT,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT,
    DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY,
    DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_PRIME_GUIDANCE_DEFAULT,
    DEEPY_PRIME_MCP_SERVERS_KEY,
    DEEPY_TYPE_DEFAULT,
    DEEPY_TYPE_DISABLED,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    DEEPY_TYPE_ZERO,
    DEEPY_KV_CACHE_QUANTIZATION_DEFAULT,
    DEEPY_KV_CACHE_QUANTIZATION_AUTO,
    DEEPY_KV_CACHE_QUANTIZATION_KEY,
    DEEPY_VRAM_MODE_KEY,
    DEEPY_VRAM_MODE_ALWAYS_LOADED,
    DEEPY_VRAM_MODE_UNLOAD,
    DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST,
    deepy_available,
    deepy_mode_from_config,
    format_deepy_context_tokens_label,
    deepy_requirement_message,
    normalize_deepy_context_tokens,
    normalize_deepy_file_system_access,
    normalize_deepy_file_system_paths,
    parse_deepy_file_system_paths,
    normalize_deepy_read_everywhere,
    normalize_deepy_python_sandbox,
    DEEPY_PYTHON_SANDBOX_DEFAULT,
    DEEPY_PYTHON_SANDBOX_KEY,
    PYTHON_APPS_KEY,
    normalize_python_apps,
    python_apps_level,
    normalize_deepy_web_search,
    DEEPY_WEB_SEARCH_DEFAULT,
    DEEPY_WEB_SEARCH_KEY,
    CIVITAI_API_KEY_KEY,
    CIVITAI_API_KEY_SECURE_KEY,
    HUGGINGFACE_TOKEN_KEY,
    HUGGINGFACE_TOKEN_SECURE_KEY,
    configured_secret,
    prepare_secret_update,
    normalize_deepy_compaction_type,
    normalize_deepy_compaction_thinking,
    normalize_deepy_repetition_penalty,
    normalize_deepy_custom_system_prompt,
    normalize_deepy_enabled,
    normalize_deepy_kv_cache_quantization,
    split_deepy_kv_cache_quantization,
    join_deepy_kv_cache_quantization,
    deepy_kv_cache_quantization_is_mixed,
    normalize_deepy_mcp_auto_discover_paths,
    normalize_deepy_prime_guidance,
    normalize_deepy_prime_mcp_servers,
    normalize_deepy_vram_mode,
    normalize_deepy_type,
    set_deepy_runtime_config,
    split_deepy_mode,
    deepy_compaction_min_tokens,
    validate_deepy_version_config,
)
from shared.prompt_enhancer.config import (
    PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT,
    PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY,
    speculative_decoding_ui_state,
    speculative_decoding_config,
    validate_prompt_enhancer_speculative_decoding,
)
from shared.deepy.onboarding import deepy_prime_upgrade_message
from shared.remote_llm.config import (
    DEEPY_ENGINE_CHOICES,
    ENGINE_CLAUDE,
    ENGINE_CODEX,
    ENGINE_OPENCODE,
    ENGINE_QWEN35_4B,
    ENGINE_QWEN38_9B,
    LLM_CONFIG_KEY,
    is_remote_engine,
    local_enhancer_id,
    normalize_llm_config,
    privacy_warning,
    resolve_role_engine,
    validate_llm_config,
)
from shared.remote_llm.claude_config import bind_claude_config_ui, claude_profile_from_values, create_claude_config_ui
from shared.remote_llm.codex_config import bind_codex_config_ui, codex_profile_from_values, create_codex_config_ui
from shared.remote_llm.opencode_config import bind_opencode_config_ui, create_opencode_config_ui, opencode_profile_from_values
from postprocessing import audio_processors as audio_processor_api
from postprocessing import temporal_upsamplers as temporal_upsampler_api
from postprocessing import spatial_upsamplers as upsampler_api
from shared.utils.wgp_config_migration import (
    PROMPT_ENHANCER_CHOICES,
    enabled_choice_value,
    get_prompt_enhancer_default_mode,
    migrate_extension_defaults,
)


QWEN35_PROMPT_ENHANCER_IDS = (3, 4)
QWEN38_PROMPT_ENHANCER_ID = 5
QWEN38_9B_PROMPT_ENHANCER_ID = 6
QWEN_PROMPT_ENHANCER_IDS = (3, 4, 5, 6)
QWEN35_QUANTIZATION_CHOICES = [("Quanto Int8 (recommended, better quality)", "quanto_int8"), ("GGUF Q4 (less VRAM/RAM & faster if kernels are installed, but worse quality)", "gguf")]
QWEN38_9B_QUANTIZATION_CHOICES = [("GGUF Uncensored Q4_K_M (default, fits 8 GB of VRAM)", "gguf"), ("GGUF Uncensored Q8_0 (closest to full precision, needs about 12 GB of VRAM)", "gguf_q8")]
QWEN38_QUANTIZATION_CHOICES = [("GGUF Uncensored Q4 (default, highest quality and VRAM/RAM use)", "gguf"), ("GGUF Uncensored IQ3_S (recommended Q3, middle quality and VRAM/RAM use)", "gguf_q3"), ("GGUF Uncensored Q2 (lowest quality and VRAM/RAM use)", "gguf_q2"), ("Bonsai 2 Abliterated PTQ1_0 (ternary, requires kernels 1.0.22+)", "gguf_ptq1")]


def prompt_enhancer_quantization_ui_state(enhancer_enabled, quantization):
    enhancer_enabled = int(enhancer_enabled)
    if enhancer_enabled == QWEN38_PROMPT_ENHANCER_ID:
        value = quantization if quantization in ("gguf", "gguf_q3", "gguf_q2", "gguf_ptq1") else "gguf"
        return QWEN38_QUANTIZATION_CHOICES, value, True
    if enhancer_enabled == QWEN38_9B_PROMPT_ENHANCER_ID:
        return QWEN38_9B_QUANTIZATION_CHOICES, quantization if quantization in ("gguf", "gguf_q8") else "gguf", True
    value = "gguf" if quantization in ("gguf", "gguf_q3", "gguf_q2", "gguf_ptq1") else "quanto_int8"
    return QWEN35_QUANTIZATION_CHOICES, value, enhancer_enabled in QWEN35_PROMPT_ENHANCER_IDS


def deepy_filesystem_ui_state(deepy_type):
    prime = normalize_deepy_type(deepy_type) == DEEPY_TYPE_PRIME
    labels = ("Outputs only (read)", "Read outputs + selected folders", "Read + create in outputs; read/write selected folders") if prime else ("Disabled", "Read Outputs + Selected Folders", "Read / Write Outputs + Selected Folders")
    access = {
        "choices": list(zip(labels, (DEEPY_FILE_SYSTEM_ACCESS_DISABLED, DEEPY_FILE_SYSTEM_ACCESS_READ, DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE))),
        "label": "Filesystem access outside workspace" if prime else "Deepy Filesystem Access",
        "info": "Deepy freely manages its workspace for drafts and experiments. Write access allows new files in outputs and subfolders; existing output files cannot be modified, overwritten, deleted, renamed or moved." if prime else "Output folders are always the default scope. Add one extra folder per line below.",
    }
    everywhere = {
        "label": "Read Everywhere" if prime else "Read Everywhere (Warning!)",
        "info": "Allows reading any server file. Writing stays limited to the workspace, new output files with write access, and selected RW folders." if prime else "Allows Deepy to read any server file. Writing always remains limited to output and selected folders.",
    }
    return access, everywhere


IMPLICIT_DYNAMIC_PROFILES = (1, 3, 3.5) # these profiles always enable dynamic preload
PRELOAD_MODES = [("Default", "default"), ("Dynamic", "dynamic"), ("Manual", "manual")]
MEMORY_ADVICE = """**Profile 4 is recommended for most PCs.** It sends the main model to the GPU part by part while the GPU works: models larger than your VRAM still run, most of the VRAM stays free for long videos or large images, and the speed cost is small. *Reserved RAM* is memory set aside to make these transfers fast. Other programs cannot use it, so only part of your RAM is reserved.

**For image models, use Profile 4 or 5 with a *Dynamic* or *Manual* VRAM Preload for faster generations.** Their steps are short, so the transfers set the speed. The *VRAM Preload* next to each memory profile (Profiles 0, 2, 4 and 5) keeps part of each model in VRAM: *Default* keeps the profile's choice (*Dynamic* for Profile 0, none for Profiles 2, 4 and 5); *Dynamic* keeps as much of the model as each generation leaves free and gives part of it back if VRAM runs short (it needs the MMGP Optimized VRAM Allocator); *Manual* keeps the amount set below it (8000 MB or more brings Flux close to Profile 1); models that work side by side (a world model and its decoder, two stages of a model) share that amount, the main one first. Video steps are usually long enough to hide the transfers.

**Profile 0 adapts to your PC:** like profile 4, it sends each model to the GPU part by part, but no model keeps Reserved RAM for good: the model running gets it for its parts, which then reach the GPU at full speed, and it passes to the next model when you switch. Fast with much less RAM than the profiles that keep models in Reserved RAM, also for models larger than your VRAM. By default its *VRAM Preload* also keeps as much of each model in VRAM as fits, like profile 3 (*Dynamic*); choose *Manual* for a fixed amount. The language models of Deepy and the prompt enhancer always use it, with *Dynamic* preload.

**Profiles 1 and 3 enable Dynamic Preload automatically:** profile 1 uses profile 2's RAM policy, and profile 3 uses profile 4's. They keep as much of each model in VRAM as fits alongside the generation's working memory, streaming the remaining layers. **Profile 3 is the default for new audio configurations.** Profile 3+ has the same placement policy without reserved RAM. LM engines that require fixed weight addresses use uncaptured steps while layers can move; models with MMGP block graphs can retain those graphs during offloading.

Change the settings below only if needed:
- **Your PC runs short of RAM:** lower *Reserved RAM for Pinning* and keep *Smart Memory Pinning* On.
- **Out of VRAM errors:** lower the preloads, or choose a higher *VAE Tiling* preset or Profile 4+.
- **Plenty of RAM and VRAM for your models:** Profile 1 is a bit faster."""

class ConfigTabPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "Configuration Tab"
        self.version = "1.1.6"
        self.description = "Lets you adjust all your performance and UI options for WAN2GP"

    def setup_ui(self):
        self.request_global("args")
        self.request_global("server_config")
        self.request_global("server_config_filename")
        self.request_global("attention_mode")
        self.request_global("compile")
        self.request_global("compile_choices")
        self.request_global("transformer_quantization_choices")
        self.request_global("text_encoder_quantization_choices")
        self.request_global("default_profile_video")
        self.request_global("default_profile_image")
        self.request_global("default_profile_audio")
        self.request_global("vae_config")
        self.request_global("boost")
        self.request_global("int8_kernels")
        self.request_global("fp8_kernels")
        self.request_global("preload_model_policy")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("transformer_types")
        self.request_global("text_encoder_quantization")
        self.request_global("attention_modes_installed")
        self.request_global("attention_modes_supported")
        self.request_global("displayed_model_types")
        self.request_global("memory_profile_choices")
        self.request_global("attention_modes_choices")
        self.request_global("save_path")
        self.request_global("image_save_path")
        self.request_global("audio_save_path")
        self.request_global("mesh_save_path")
        self.request_global("quit_application")
        self.request_global("release_model")
        self.request_global("release_flashvsr_vram")
        self.request_global("release_pid_vram")
        self.request_global("release_extension_offloadobjs")
        self.request_global("app")
        self.request_global("fl")
        self.request_global("is_generation_in_progress")
        self.request_global("generate_header")
        self.request_global("generate_dropdown_model_list")
        self.request_global("_model_choice_target_value")
        self.request_global("_get_dropdown_deps")
        self.request_global("create_models_selector_hierarchy")
        self.request_global("get_unique_id")
        self.request_global("reset_prompt_enhancer")
        self.request_global("reset_prompt_enhancer_if_requested")
        self.request_global("release_deepy_vram")
        self.request_global("any_GPU_process_running")
        self.request_global("apply_int8_kernel_setting")
        self.request_global("apply_fp8_kernel_setting")

        self.request_component("model_description")
        self.request_component("header")
        self.request_component("overrides")
        self.request_component("model_family")
        self.request_component("model_base_type_choice")
        self.request_component("model_choice")
        self.request_component("model_choice_target")
        self.request_component("model_filter")
        self.request_component("refresh_form_trigger")      
        self.request_component("state")
        self.request_component("resolution")
        self.request_component("assistant_launcher_host")
        self.request_component("assistant_panel")
        self.request_component("assistant_chat_event")
        self.request_component("apps_gallery_tab")

        self.add_tab(
            tab_id="configuration",
            label="Configuration",
            component_constructor=self.create_config_ui,
        )

    def create_config_ui(self):
        first_event = len(gr.context.get_blocks_context().fns)
        migrate_extension_defaults(self.server_config, self.server_config_filename)
        set_deepy_runtime_config(self.server_config, self.server_config_filename)
        prompt_enhancer_default_mode = get_prompt_enhancer_default_mode()
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("General"):
                    with gr.Group():
                        self.transformer_types_choices = HierarchySelector(
                            hierarchy=self.create_models_selector_hierarchy(self.displayed_model_types),
                            value=self.transformer_types,
                            height=0,
                            label="Selectable Generative Models: whole families, models with all their finetunes, or single finetunes (leave empty for all)",
                            display_mode="breadcrumb",
                            sort_hierarchy=False,
                            search_empty_label="No matching models",
                            elem_classes=["block"],
                            interactive=not self.args.lock_config,
                        )
                        self.invert_model_selection_choice = gr.Checkbox(
                            value=self.server_config.get(model_dropdowns.INVERT_MODEL_SELECTION_CONFIG_KEY, False),
                            label="Invert selection (hide selected models)",
                            info="Show unselected and new entries. Selected families and models are hidden together with all their finetunes, including future finetunes.",
                            interactive=not self.args.lock_config,
                        )
                        self.show_all_model_filter_choice = gr.Checkbox(
                            value=self.server_config.get(model_dropdowns.SHOW_ALL_MODEL_FILTER_CONFIG_KEY, False),
                            label='Show "All" Model Filter',
                            interactive=not self.args.lock_config,
                        )
                    self.model_hierarchy_type_choice = gr.Dropdown(
                        choices=[
                            ("Two Levels: Model Family > Models & Finetunes", 0),
                            ("Three Levels: Model Family > Models > Finetunes", 1),
                        ],
                        value=self.server_config.get("model_hierarchy_type", 1),
                        label="Models Hierarchy In User Interface",
                        interactive=not self.args.lock_config
                    )
                    self.fit_canvas_choice = gr.Dropdown(
                        choices=[
                            ("Dimensions are Pixel Budget (preserves aspect ratio, may exceed dimensions)", 0),
                            ("Dimensions are Max Width/Height (preserves aspect ratio, fits within box)", 1),
                            ("Dimensions are Exact Output (crops input to fit exact dimensions)", 2),
                        ],
                        value=self.server_config.get("fit_canvas", 0),
                        label="Input Image/Video Sizing Behavior",
                        interactive=not self.args.lock_config
                    )

                    self.generation_preview_choice = gr.Dropdown(
                        choices=[("Frames Selection using RGB Factors (fast)", "rgb"), ("Frames Selection using Tiny VAE (when available, slower)", "tiny_vae_frames"), ("Video using Tiny VAE (when available, even slower)", "tiny_vae_video")],
                        value=self.server_config.get("generation_preview", "rgb"), label="Generation Preview",
                        info="Tiny VAE modes use additional GPU memory and download a small decoder on first use. Unsupported models keep RGB previews; image generations show still images.",
                        interactive=not self.args.lock_config,
                    )
                    self.preload_model_policy_choice = gr.CheckboxGroup(
                        [("Preload Model on App Launch","P"), ("Preload Model on Switch", "S"), ("Unload Model when Queue is Done", "U")],
                        value=self.preload_model_policy, label="Model Loading/Unloading Policy"
                    )
                    self.clear_file_list_choice = gr.Dropdown(
                        choices=[("All", 0), ("Last media", 1), ("Last 5", 5), ("Last 10", 10), ("Last 20", 20), ("Last 30", 30)],
                        value=self.server_config.get("clear_file_list", 5), label="Media Visible per Gallery", info="Older media remain in the workspace and can be managed with its magnifier."
                    )
                    self.multi_prompts_gen_type_choice = gr.Dropdown(
                        choices=prompt_parser.get_multi_prompts_gen_choices("Video"),
                        value=prompt_parser.normalize_multi_prompts_mode(self.server_config.get("multi_prompts_gen_type", prompt_parser.DEFAULT_MULTI_PROMPTS_MODE), default=prompt_parser.DEFAULT_MULTI_PROMPTS_MODE),
                        label="How to Process each Line of the Text Prompt (First Time Model Default)",
                    )
                    self.display_stats_choice = gr.Dropdown(
                        choices=[("Disabled", 0), ("Enabled", 1)],
                        value=self.server_config.get("display_stats", 0), label="Display real-time RAM/VRAM stats (requires restart)"
                    )
                    self.max_frames_multiplier_choice = gr.Dropdown(
                        choices=[("Default", 1), ("x2", 2), ("x3", 3), ("x4", 4), ("x5", 5), ("x6", 6), ("x7", 7)],
                        value=self.server_config.get("max_frames_multiplier", 1), label="Max Frames / Duration Multiplier (requires restart)"
                    )
                    with gr.Row():
                        self.keep_resolution_on_model_switch_choice = gr.Dropdown(
                            choices=[("Yes", True), ("No", False)],
                            value=self.server_config.get("keep_resolution_on_model_switch", True),
                            label="Try to Keep Resolution when Switching Model",
                        )
                        self.enable_4k_resolutions_choice = gr.Dropdown(
                            choices=[("Off", 0), ("On", 1)],
                            value=self.server_config.get("enable_4k_resolutions", 0),
                            label="3K/4K+ Resolutions are available for all models"
                        )
                    default_paths = self.fl.default_checkpoints_paths
                    checkpoints_paths_text = "\n".join(self.server_config.get("checkpoints_paths", default_paths))
                    self.checkpoints_paths_choice = gr.Textbox(
                        label="Model Checkpoint Folders (One Path per Line. First is Default Download Path)",
                        value=checkpoints_paths_text,
                        lines=3,
                        interactive=not self.args.lock_config
                    )
                    self.loras_root_choice = gr.Textbox(
                        label="Loras Root Folder",
                        value=self.server_config.get("loras_root", "loras"),
                        interactive=not self.args.lock_config
                    )
                    self.save_queue_if_crash_choice = gr.Dropdown(
                        choices=[("Disabled", 0), ("Overwrite Last Error Queue", 1), ("Create a New Error Queue File", 2)],
                        value=self.server_config.get("save_queue_if_crash", 1),
                        label="Save Queue if Crash during Generation",
                        interactive=not self.args.lock_config
                    )
                    self.UI_theme_choice = gr.Dropdown(
                        choices=ui_studio.THEME_CHOICES,
                        value=self.server_config.get("UI_theme", "default"), label="UI Theme (requires restart)"
                    )
                    self.floating_generate_button_choice = gr.Checkbox(
                        value=self.server_config.get("floating_generate_button", True),
                        label="Enable Floating Action Buttons (requires restart)",
                        info="Keep generation, selected-media and Edit Mode controls visible until they reach their place on the page. Media actions use a separate floating panel.",
                        interactive=not self.args.lock_config
                    )
                    self.queue_color_scheme_choice = gr.Dropdown(
                        choices=[
                            ("Pastel (Unique color for each item)", "pastel"),
                            ("Alternating Grey Shades", "alternating_grey"),
                        ],
                        value=self.server_config.get("queue_color_scheme", "pastel"),
                        label="Queue Color Scheme"
                    )
                    self.voice_mode_choice = gr.Dropdown(choices=VOICE_MODE_CHOICES, value=lambda: voice_mode(self.server_config), label="Microphone transcription", elem_id="voice_transcription_mode", interactive=not self.args.lock_config)
                    self.deepy_voice_language_choice = gr.Dropdown(
                        choices=deepy_voice_language_choices(),
                        value=lambda: normalize_deepy_voice_language(self.server_config.get(DEEPY_VOICE_LANGUAGE_KEY, "auto")),
                        label="Voice transcription language",
                        info="Auto detects the spoken language. Selecting a known language can improve recognition of short recordings. --deepy-voice-language overrides this choice.",
                        elem_id="deepy_voice_language_choice", interactive=not self.args.lock_config,
                    )

                with gr.Tab("Performance"):
                    self.attention_choice = gr.Dropdown(
                        choices=self.attention_modes_choices,
                        value=self.attention_mode, label="Attention Type", interactive=not self.args.lock_config
                    )
                    self.quantization_choice = gr.Dropdown(choices=self.transformer_quantization_choices, value=self.transformer_quantization, label="Transformer Model Quantization (if available otherwise get the closest available)")
                    self.transformer_dtype_policy_choice = gr.Dropdown(choices=[("Auto (Best for Hardware)", ""), ("FP16", "fp16"), ("BF16", "bf16")], value=self.transformer_dtype_policy, label="Transformer Data Type (if available)")
                    self.mixed_precision_choice = gr.Dropdown(choices=[("16-bit only (less VRAM)", "0"), ("Mixed 16/32-bit (better quality)", "1")], value=self.server_config.get("mixed_precision", "0"), label="Transformer Engine Precision")
                    self.text_encoder_quantization_choice = gr.Dropdown(choices=self.text_encoder_quantization_choices, value=self.text_encoder_quantization, label="Text Encoder Quantization")
                    self.compile_choice = gr.Dropdown(choices=self.compile_choices, value=self.compile, label="Transformer Compilation (faster, but the first generation is slower and some GPUs/Models may not support it)", interactive=not self.args.lock_config)
                    self.boost_choice = gr.Dropdown(choices=[("ON", 1), ("OFF", 2)], value=self.boost, label="Boost Layers Offloading when CFG > 1 (~10% speedup for ~1GB VRAM)")
                    self.int8_kernels_choice = gr.Dropdown(choices=int8_backend.CHOICES, value=self.server_config.get("int8_kernels", "auto"), label="INT8 Math Kernels", info="Auto selects WGPKernels, then Comfy Kitchen, then Triton, then PyTorch; an unavailable choice behaves as Auto. Disabled uses PyTorch. Other fused kernels (VAE, attention normalization) also try the selected family first. Changes apply to the next generation without reloading weights.")
                    self.fp8_kernels_choice = gr.Dropdown(choices=fp8_backend.CHOICES, value=self.server_config.get("fp8_kernels", "auto"), label="FP8 Math Kernels", info="For FP8 checkpoints. Auto uses CUDA FP8 when available, then Triton, then PyTorch. Changes apply to the next generation without reloading weights.")
                    self.kernel_precision_choice = gr.Dropdown(choices=kernel_policy.CHOICES, value=self.server_config.get("kernel_precision", "fast"), label="CUDA Kernels Optimized Ops Precision (When Available)", info="Preserve Precision keeps only kernels at least as precise as standard PyTorch math. Allow Faster Approximate Kernels (default) adds faster kernels with small rounding differences. INT8 math is controlled separately.")
                    self.attention_head_split_choice = gr.Dropdown(choices=attention_kit.HEAD_SPLIT_CHOICES, value=self.server_config.get("attention_head_split", 0), label="Attention Head Split (much lower VRAM for long videos, up to 10% slower)", info="Models that support it compute attention one group of heads at a time for long sequences, which lowers the VRAM peak of denoising. Higher levels use more groups, saving more VRAM at a small speed cost; how many groups each level gives depends on the model. Applies to the next generation without reloading.")
                    self.process_queues_when_browser_unfocused_choice = gr.Checkbox(
                        value=bool(self.server_config.get(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1)),
                        label="Continue Browser Actions in Background (may use more energy)",
                        info="Let pending interface actions and updates continue while this browser tab is in the background. Already-started generation queues run on the server and continue even when all browser tabs are closed. Does not change server process priority.",
                        interactive=not self.args.lock_config
                    )
                    self.prevent_power_throttling_choice = gr.Checkbox(
                        value=self.server_config.get("prevent_power_throttling", True),
                        label="Prevent Windows Background Power Throttling (requires restart)",
                        info="Keep the server at full CPU performance when minimized or in the background. Applies to all browser sessions. Turn off to let Windows manage background power use. The --prevent-power-throttling and --no-prevent-power-throttling launch options override this choice.",
                        visible=os.name == "nt", interactive=not self.args.lock_config,
                    )
                    compute_cache, compute_cache_shown = self.server_config.get(COMPUTE_CACHE_KEY, False), compute_cache_useful()
                    with gr.Group(visible=compute_cache_shown):
                        self.compute_cache_choice = gr.Checkbox(
                            value=compute_cache, label="WanGP Compute Cache (requires restart)",
                            info="On RTX 50xx and other Blackwell GPUs, the NVIDIA driver optimizes the GPU code of WanGP for the exact GPU at each launch. This keeps the result in a folder of its own, where other programs no longer evict it: the code is optimized once instead of after every restart, so the first generation after a restart starts sooner and the SSD is no longer rewritten at each launch. May take up to 2 GB of disk space.",
                            interactive=not self.args.lock_config,
                        )
                        self.compute_cache_path_choice = gr.Textbox(
                            value=self.server_config.get(COMPUTE_CACHE_PATH_KEY, COMPUTE_CACHE_PATH_DEFAULT), label="Compute Cache Folder",
                            info="Relative to the WanGP folder, or a full path.", visible=compute_cache, interactive=not self.args.lock_config,
                        )
                    self.compute_cache_choice.change(fn=lambda enabled: gr.update(visible=enabled), inputs=[self.compute_cache_choice], outputs=[self.compute_cache_path_choice], show_progress="hidden")

                with gr.Tab("RAM/VRAM Management"):
                    self.ram_allocator_choice = gr.Dropdown(
                        choices=[("MMGP RAM Allocator (the RAM of freed CPU tensors goes back to the system when the queue is done, a model is released or the RAM runs short)", "mmgp"),
                                 ("PyTorch Default RAM Allocator (keeps the RAM of freed CPU tensors for reuse)", "default")],
                        value=self.server_config.get(RAM_ALLOCATOR_KEY, RAM_ALLOCATOR_DEFAULT),
                        label="RAM Allocator (requires restart)",
                        info="The MMGP RAM Allocator frees several GB that PyTorch would keep after a generation or a model release, at the same speed. Same as --ram-allocator, which takes precedence.",
                    )
                    self.vram_allocator_choice = gr.Dropdown(
                        choices=[("MMGP Optimized VRAM Allocator with RAM Spilling (a generation slightly too large for the VRAM can still finish, more slowly)", "vmm_spill"),
                                 ("MMGP Optimized VRAM Allocator (out of memory error when no VRAM is left)", "vmm"),
                                 ("PyTorch Default VRAM Allocator", "default")],
                        value=self.server_config.get("vram_allocator", "vmm_spill"),
                        label="VRAM Allocator (requires restart)",
                        info="The MMGP Optimized VRAM Allocator recycles more efficiently the VRAM that is no longer used: lower peak VRAM with long videos, large images and Deepy. Same as --vram-allocator, which takes precedence.",
                    )
                    gr.Markdown(MEMORY_ADVICE)
                    preload_info = "This part of each model stays in VRAM (shared by models that work side by side, the main one first): less to transfer at each step, faster for short steps such as images, at the cost of VRAM."
                    profiles, modes, preloads = {}, {}, {}
                    for output_type, title, profile in (("video", "Video", self.default_profile_video), ("image", "Image", self.default_profile_image), ("audio", "Audio", self.default_profile_audio)):
                        streamed, mode = profile not in IMPLICIT_DYNAMIC_PROFILES, self.server_config[f"{output_type}_preload_mode"]
                        with gr.Group():
                            with gr.Row():
                                profiles[output_type] = gr.Dropdown(choices=self.memory_profile_choices, value=profile, label=f"{title} - Default Memory Profile", scale=6)
                                modes[output_type] = gr.Dropdown(choices=PRELOAD_MODES, value=mode, label="VRAM Preload", scale=1, min_width=140, visible=streamed)
                            preloads[output_type] = gr.Slider(0, 40000, value=self.server_config.get(f"{output_type}_preload_in_VRAM", 0), step=100, label=f"{title} VRAM Preload (MB)", info=preload_info, visible=streamed and mode == "manual")
                    self.video_profile_choice, self.image_profile_choice, self.audio_profile_choice = profiles.values()
                    self.video_preload_mode_choice, self.image_preload_mode_choice, self.audio_preload_mode_choice = modes.values()
                    self.video_preload_choice, self.image_preload_choice, self.audio_preload_choice = preloads.values()
                    with gr.Row():
                        self.smart_memory_pinning_choice = gr.Dropdown(
                            choices=[("On: faster, uses 1-2 GB more Reserved RAM", True), ("Off: slower, saves 1-2 GB of Reserved RAM", False)],
                            value=self.server_config.get("smart_memory_pinning", True),
                            label="Smart Memory Pinning",
                            info="Helps most with Profile 5 and image models. Allocate a 1-2GB Reserved RAM buffer to accelerate transfers.",
                        )
                        self.perc_reserved_mem_max_choice = gr.Slider(
                            0, 80, value=self.server_config.get("perc_reserved_mem_max", 0), step=5,
                            label="Reserved RAM for Pinning (% of RAM, 0 = Auto)",
                            info="Auto: 40% on Windows, 60% on Linux. Same as the command line --perc-reserved-mem-max (a fraction, e.g. 0.4), which takes precedence, and as the environment variable perc_reserved_mem_max, which this setting overrides.",
                        )
                    self.llm_cpu_compute_choice = gr.Dropdown(
                        choices=[("Auto: the CPU helps when it is faster on this PC (recommended)", "auto"),
                                 ("On: the CPU always helps, even when it is slower (to compare)", "on"),
                                 ("Off: the layers that do not fit in VRAM are transferred to the GPU", "off")],
                        value=self.server_config.get("llm_cpu_compute", "auto") if self.server_config.get("llm_cpu_compute", "auto") in ("on", "off") else "auto",
                        label="Language Models CPU Compute (Prompt Enhancer and Deepy)",
                        info="When the Language Model does not fit in VRAM, writing is much faster if the CPU processes part of the layers left in RAM while the GPU receives the others. Auto measures what is faster on this PC for each model format and leaves the layers to the GPU when the CPU is slower; On uses the CPU anyway, to compare. No effect when the model fits in VRAM. The Reserved RAM these models use while they write follows Memory Profile 0, which they always use.",
                    )
                    def update_preload_visibility(profile, mode):
                        streamed = profile not in IMPLICIT_DYNAMIC_PROFILES
                        return gr.update(visible=streamed), gr.update(visible=streamed and mode == "manual")
                    for output_type in profiles:
                        for trigger in (profiles[output_type], modes[output_type]):
                            trigger.change(fn=update_preload_visibility, inputs=[profiles[output_type], modes[output_type]], outputs=[modes[output_type], preloads[output_type]], show_progress="hidden")
                    self.max_reserved_loras_choice = gr.Slider(
                        -1,
                        10000,
                        value=self.server_config.get("max_reserved_loras", -1),
                        step=1,
                        label="Max Amount of Loras (in MB) to be Pinned To Reserved Memory (set it to 0-500MB if Out of Memory when starting Gen, -1= No limit)"
                    )
                    vae_config_value = self.vae_config if self.vae_config in [0, 1, 2, 3] else 0
                    if mmgp_vram_allocator():  # no fragmentation to guard against: a fast default and a Low VRAM level
                        self.vae_config_choice = gr.Dropdown(
                            choices=[("Default (faster, fewer tiles)", 0), ("Low VRAM (slower, smaller tiles)", 3)],
                            value=3 if vae_config_value == 3 else 0, label="VAE Tiling",
                            info="Default already switches to smaller tiles on GPUs with less than 8 GB of VRAM. Low VRAM forces them, for GPUs where other programs also need VRAM.",
                        )
                    else:
                        self.vae_config_choice = gr.Dropdown(
                            choices=[
                                ("Auto", 0),
                                ("Spatial / Temporal Tiling Preset for 16GB+", 1),
                                ("Spatial / Temporal Tiling Preset for 8GB+", 2),
                                ("Spatial / Temporal Tiling Preset for 6GB+", 3),
                            ],
                            value=vae_config_value,
                            label="VAE Tiling (higher presets use less VRAM and may increase artifacts like banding)",
                        )
                    self.release_RAM_btn = gr.Button("Force Unload Models from RAM")

                with gr.Tab("Extensions"):
                    self.postprocessor_preload_mode_choice, self.postprocessor_preload_choice = processor_memory.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    gr.Markdown("**Audio Postprocessors**")
                    self.audio_processor_config_bindings = audio_processor_api.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    self.audio_processor_config_components = audio_processor_api.config_components(self.audio_processor_config_bindings)
                    with gr.Column() as temporal_config_group:
                        gr.Markdown("**Temporal Postprocessors**")
                        self.temporal_upsampler_config_bindings = temporal_upsampler_api.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    temporal_config_group.visible = bool(self.temporal_upsampler_config_bindings)
                    self.temporal_upsampler_config_components = temporal_upsampler_api.config_components(self.temporal_upsampler_config_bindings)
                    gr.Markdown("**Spatial Upsamplers / Visual Refiners**")
                    self.upsampler_config_bindings = upsampler_api.create_config_ui(gr, self.server_config, lock_config=self.args.lock_config)
                    self.upsampler_config_components = upsampler_api.config_components(self.upsampler_config_bindings)
                    gr.Markdown("**Video Preprocessors**")
                    with gr.Tabs():
                        with gr.Tab("Mask Generator"):
                            self.matanyone_version_choice = gr.Dropdown(
                                choices=[("MatAnyone v1 (original, default)", "v1"), ("MatAnyone v2", "v2"), ("SAM3 (no Alpha / Grey level support but better Temporal Stability & Auto Mask Selection by Keyword)", "sam3")],
                                value=self.server_config.get("matanyone_version", "v1"),
                                label="Mask Generator Engine",
                                interactive=not self.args.lock_config
                            )
                        with gr.Tab("Depth Anything"):
                            self.depth_anything_v2_variant_choice = gr.Dropdown(choices=[("Depth Anything 2 Large (more precise, slower)", "vitl"), ("Depth Anything 2 Big (less precise, faster)", "vitb"), ("Depth Anything 3 Metric Large (better temporal stability ?)", "da3_metric_large")], value=self.server_config.get("depth_anything_v2_variant", "vitl"), label="Depth Anything Preprocessor")
                    gr.Markdown("**Speech Recognition**")
                    with gr.Group():
                        self.whisper_model_choice = gr.Dropdown(choices=WHISPER_MODEL_CHOICES, value=self.server_config.get(WHISPER_MODEL_KEY, "auto"), interactive=not self.args.lock_config,
                            label="Whisper Model (Deepy Voice Input and Transcriptions, Prompt Enhancer, Reference Transcripts and Speech Alignment of Audio Models)")


                with gr.Tab("Prompt Enhancer / Deepy"):
                    deepy_prime_recommendation = deepy_prime_upgrade_message(self.server_config)
                    self.deepy_prime_recommendation = gr.Markdown(value=deepy_prime_recommendation, visible=bool(deepy_prime_recommendation), elem_classes=["deepy-prime-recommendation"])
                    llm_config = normalize_llm_config(self.server_config)
                    if llm_config["deepy"] not in {value for _label, value in DEEPY_ENGINE_CHOICES}:
                        llm_config["deepy"] = ENGINE_QWEN35_4B
                    llm_config_view = {**self.server_config, LLM_CONFIG_KEY: llm_config}
                    active_llm_engines = {resolve_role_engine(llm_config_view, "deepy")}
                    deepy_remote_default = is_remote_engine(resolve_role_engine(llm_config_view, "deepy"))
                    deepy_type_default = deepy_mode_from_config(self.server_config.get(DEEPY_ENABLED_KEY, 0), self.server_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT))
                    with gr.Group(elem_classes=["wangp-transparent-group"]):
                        self.deepy_llm_engine_choice = gr.Dropdown(choices=[choice for choice in DEEPY_ENGINE_CHOICES if choice[1] != ENGINE_QWEN38_9B or llm_config["deepy"] == ENGINE_QWEN38_9B], value=llm_config["deepy"], label="Prompt Enhancer / Deepy LLM Engine")  # Qwen3.8 9B hidden until it works reliably
                        self.remote_llm_warning_md = gr.Markdown(value=privacy_warning(self.server_config))
                        self.remote_llm_auth_md = gr.Markdown("Authentication is managed by each external engine. WanGP does not request or store passwords, API keys, access tokens, or refresh tokens.", visible=deepy_remote_default)
                        self.codex_config_ui = create_codex_config_ui(gr, llm_config["profiles"][ENGINE_CODEX], visible=ENGINE_CODEX in active_llm_engines, lock_config=self.args.lock_config)
                        self.claude_config_ui = create_claude_config_ui(gr, llm_config["profiles"][ENGINE_CLAUDE], visible=ENGINE_CLAUDE in active_llm_engines, lock_config=self.args.lock_config)
                        self.opencode_config_ui = create_opencode_config_ui(gr, llm_config["profiles"][ENGINE_OPENCODE], visible=ENGINE_OPENCODE in active_llm_engines, lock_config=self.args.lock_config)
                    enhancer_enabled_value = local_enhancer_id(llm_config["deepy"], enabled_choice_value(self.server_config.get("enhancer_enabled", prompt_enhancer_default_mode), PROMPT_ENHANCER_CHOICES, prompt_enhancer_default_mode))
                    self.enhancer_enabled_choice = gr.Dropdown(
                        choices=PROMPT_ENHANCER_CHOICES,
                        value=enhancer_enabled_value,
                        label="Local model used to power Prompt Enhancer / Deepy",
                        visible=False,
                    )
                    enhancer_quantization_choices, enhancer_quantization_value, enhancer_quantization_visible = prompt_enhancer_quantization_ui_state(enhancer_enabled_value, self.server_config.get("prompt_enhancer_quantization", "quanto_int8"))
                    self.enhancer_quantization_choice = gr.Dropdown(
                        choices=enhancer_quantization_choices,
                        value=enhancer_quantization_value,
                        label="Qwen LLM Quantization",
                        info="Lower precision saves VRAM/RAM at some cost to quality. Use the highest precision that fits comfortably.",
                        visible=enhancer_quantization_visible and not deepy_remote_default,
                    )
                    deepy_context_tokens_default = normalize_deepy_context_tokens(self.server_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))
                    if deepy_type_default == DEEPY_TYPE_PRIME:
                        deepy_context_tokens_default = max(deepy_compaction_min_tokens(self.server_config.get(DEEPY_COMPACTION_TYPE_KEY), self.server_config.get(DEEPY_COMPACTION_THINKING_KEY, DEEPY_COMPACTION_THINKING_DEFAULT)), deepy_context_tokens_default)
                    deepy_kv_cache_quantization_default = normalize_deepy_kv_cache_quantization(self.server_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT))
                    deepy_kv_cache_body_default, deepy_kv_cache_exact_default = split_deepy_kv_cache_quantization(deepy_kv_cache_quantization_default)
                    def current_compaction_choice():
                        compaction_type = normalize_deepy_compaction_type(self.server_config.get(DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_TYPE_DEFAULT))
                        if deepy_mode_from_config(self.server_config.get(DEEPY_ENABLED_KEY, 0), self.server_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT)) == DEEPY_TYPE_PRIME:
                            compaction_type = DEEPY_COMPACTION_TYPE_SUMMARIZE
                        if compaction_type == DEEPY_COMPACTION_TYPE_SUMMARIZE and normalize_deepy_compaction_thinking(self.server_config.get(DEEPY_COMPACTION_THINKING_KEY, DEEPY_COMPACTION_THINKING_DEFAULT)):
                            return DEEPY_COMPACTION_CHOICE_THINKING
                        return compaction_type
                    with gr.Row():
                        self.enhancer_mode_choice = gr.Dropdown(choices=[("Manual Button Only", 1),("Manual Button + Automatic on Generation", 0)], value=self.server_config.get("enhancer_mode", 1), label="Prompt Enhancer Usage")
                    with gr.Row(visible=not deepy_remote_default) as self.enhancer_sampling_row:
                        self.prompt_enhancer_temperature_choice = gr.Slider(
                            0.1,
                            1.5,
                            value=self.server_config.get("prompt_enhancer_temperature", 0.6),
                            step=0.01,
                            show_reset_button=False,
                            label="Sampling Temperature (High = More Creativity)",
                            interactive=not self.args.lock_config,
                        )
                        self.prompt_enhancer_top_p_choice = gr.Slider(
                            0.1,
                            1.0,
                            value=self.server_config.get("prompt_enhancer_top_p", 0.9),
                            step=0.01,
                            show_reset_button=False,
                            label="Sampling Top-p (High = More Variety)",
                            interactive=not self.args.lock_config,
                        )
                        self.prompt_enhancer_randomize_seed_choice = gr.Checkbox(
                            value=self.server_config.get("prompt_enhancer_randomize_seed", True),
                            label="Randomize Prompt Enhancer Seed",
                            elem_classes="cbx_bottom",
                            interactive=not self.args.lock_config,
                        )
                    speculative_choices, speculative_method, token_choices, speculative_tokens = speculative_decoding_ui_state(
                        enhancer_enabled_value, enhancer_quantization_value, self.server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT))
                    with gr.Row(visible=not deepy_remote_default and enhancer_enabled_value in QWEN_PROMPT_ENHANCER_IDS) as self.enhancer_speculative_row:
                        self.enhancer_speculative_decoding_choice = gr.Dropdown(
                            choices=speculative_choices, value=speculative_method, label="Speculative Decoding",
                            info="Speculative decoding uses extra VRAM. For Bonsai PTQ1, Auto disables MTP at 10 GiB VRAM or less and uses 2 draft tokens above 10 GiB.",
                            interactive=not self.args.lock_config,
                        )
                        self.enhancer_speculative_tokens_choice = gr.Dropdown(
                            choices=token_choices, value=speculative_tokens, label="Number of Tokens",
                            info="Maximum draft tokens per prediction. More tokens can increase memory use and may be slower",
                            interactive=bool(token_choices) and not self.args.lock_config,
                        )
                    with gr.Row(visible=not deepy_remote_default and enhancer_enabled_value in QWEN_PROMPT_ENHANCER_IDS) as self.enhancer_decoding_row:
                        self.deepy_kv_cache_quantization_choice = gr.Dropdown(
                            choices=[
                                ("Auto", DEEPY_KV_CACHE_QUANTIZATION_AUTO), ("Disabled (BF16)", ""), ("INT8 (about half the KV-cache VRAM)", "int8"),
                                ("INT8 keys, Q4 values (-24% vs INT8)", "int8_q4"), ("Q6 keys and values (-24% vs INT8)", "q6"), ("Q6 keys, Q4 values (-35% vs INT8)", "q6_q4"),
                                ("Q4.1 keys and values (-41% vs INT8)", "q4_1"), ("Q4 keys and values (-47% vs INT8)", "q4"),
                            ],
                            value=deepy_kv_cache_body_default,
                            label="KV Cache Quantization",
                            info="INT8 halves KV-cache VRAM use. Below INT8, the oldest and most recent tokens stay in INT8 and the rest of the cache is smaller still; "
                                 "Q6 and INT8/Q4 are nearly indistinguishable from INT8, Q4.1 and Q4 cost a little precision.",
                        )
                        self.deepy_kv_cache_exact_choice = gr.Dropdown(
                            choices=[
                                ("First page + last 1024 tokens in INT8 (recommended)", "s1r1024"), ("First page + last 256 tokens in INT8", "s1r256"),
                                ("Last 1024 tokens in INT8, no first page", "s0r1024"), ("None (uniform quantization)", "s0r0"),
                            ],
                            value=deepy_kv_cache_exact_default,
                            label="Exact Sink and Recent Tokens",
                            info="With a KV cache below INT8, which tokens keep the INT8 precision: the first page holds the system prompt, the recent tokens matter most.",
                            visible=deepy_kv_cache_quantization_is_mixed(deepy_kv_cache_body_default),
                        )
                        self.deepy_repetition_penalty_choice = gr.Dropdown(
                            choices=[("Disabled", False), ("Enabled", True)],
                            value=normalize_deepy_repetition_penalty(self.server_config.get(DEEPY_REPETITION_PENALTY_KEY, DEEPY_REPETITION_PENALTY_DEFAULT)),
                            label="Repetition Penalty",
                            info="Reduces repeated phrases in Prompt Enhancer and Deepy; about 10% slower.",
                            visible=not deepy_remote_default and enhancer_enabled_value in QWEN_PROMPT_ENHANCER_IDS,
                        )
                    self.deepy_type_choice = gr.Dropdown(
                        choices=[("Disabled", DEEPY_TYPE_DISABLED), ("Deepy Zero", DEEPY_TYPE_ZERO), ("Deepy Prime", DEEPY_TYPE_PRIME)],
                        value=deepy_type_default,
                        label="Deepy",
                        info="Deepy Zero uses local Qwen models for focused tasks such as generation, editing and transcription. Deepy Prime plans multi-step projects, manages workspace files and can use external tools; it requires a local Qwen3.8 VL model (9B or 27B) or an external LLM.",
                        elem_id="deepy_type_choice",
                    )
                    with gr.Row(visible=deepy_type_default != DEEPY_TYPE_DISABLED and not deepy_remote_default) as self.deepy_vram_row:
                        self.deepy_vram_mode_choice = gr.Dropdown(
                            choices=[
                                ("Unload from VRAM as soon as possible", DEEPY_VRAM_MODE_UNLOAD),
                                ("Unload from VRAM if VRAM requested by another WanGP component", DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST),
                                ("Always loaded in VRAM", DEEPY_VRAM_MODE_ALWAYS_LOADED),
                            ],
                            value=normalize_deepy_vram_mode(self.server_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD)),
                            label="Deepy VRAM Loading Mode (the longer Deepy stays in VRAM, the faster Deepy is)",
                        )
                    with gr.Row(visible=deepy_type_default != DEEPY_TYPE_DISABLED and not deepy_remote_default and enhancer_enabled_value in QWEN_PROMPT_ENHANCER_IDS) as self.deepy_context_row:
                        with gr.Column(scale=2):
                            self.deepy_context_tokens_choice = gr.Slider(
                                minimum=DEEPY_CONTEXT_TOKENS_MIN,
                                maximum=256000,
                                value=deepy_context_tokens_default,
                                step=512,
                                show_reset_button=False,
                                label=format_deepy_context_tokens_label(self.server_config.get("enhancer_enabled", 0), deepy_context_tokens_default, deepy_kv_cache_quantization_default),
                                info="More tokens retain more conversation and tool history, but use more VRAM.",
                            )
                        with gr.Column(scale=1, visible=deepy_type_default != DEEPY_TYPE_DISABLED and not deepy_remote_default and enhancer_enabled_value in QWEN_PROMPT_ENHANCER_IDS) as self.deepy_compaction_column:
                            self.deepy_compaction_type_choice = gr.Dropdown(
                                choices=[("Discard Oldest Entries", DEEPY_COMPACTION_TYPE_DISCARD), (f"Summarize (recommended, {DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS:,}+ tokens)", DEEPY_COMPACTION_TYPE_SUMMARIZE), (f"Summarize with Thinking ({DEEPY_COMPACTION_THINKING_MIN_TOKENS:,}+ tokens)", DEEPY_COMPACTION_CHOICE_THINKING)],
                                value=current_compaction_choice,
                                label="Compaction Type When Context is Full",
                                info="Choose how to make room when the conversation fills the context window.",
                            )
                    self.deepy_type_value = gr.HTML(value=deepy_type_default, elem_id="deepy_type_value")
                    with gr.Column(visible=deepy_type_default != DEEPY_TYPE_DISABLED) as self.deepy_options_group:
                        prime_guidance_value = normalize_deepy_prime_guidance(self.server_config.get(DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_GUIDANCE_DEFAULT))
                        deepy_file_system_access = normalize_deepy_file_system_access(self.server_config.get(DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT))
                        access_ui, everywhere_ui = deepy_filesystem_ui_state(deepy_type_default)
                        self.deepy_zero_custom_system_prompt_choice = gr.Textbox(
                            value=normalize_deepy_custom_system_prompt(self.server_config.get(DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY, "")),
                            visible=False,
                            lines=7,
                            label="Deepy Zero Custom System Prompt",
                            info="Added after the built-in Deepy Zero system prompt on the next user interaction.",
                        )
                        with gr.Tabs(selected="filesystem") as self.deepy_settings_tabs:
                            with gr.Tab("Filesystem", id="filesystem"):
                                self.deepy_allow_read_file_system_choice = gr.Dropdown(
                                    value=deepy_file_system_access,
                                    **access_ui,
                                )
                                self.deepy_file_system_paths_choice = gr.Textbox(
                                    value="\n".join(normalize_deepy_file_system_paths(self.server_config.get(DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_FILE_SYSTEM_PATHS_DEFAULT))),
                                    lines=4,
                                    label="Additional Filesystem Folders",
                                    info='One folder per line, optionally followed by an optional alias. Quote paths containing spaces, for example: "D:\\My Media" projects.',
                                    visible=deepy_file_system_access != DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
                                )
                                self.deepy_read_everywhere_choice = gr.Checkbox(
                                    value=normalize_deepy_read_everywhere(self.server_config.get(DEEPY_READ_EVERYWHERE_KEY, DEEPY_READ_EVERYWHERE_DEFAULT)),
                                    visible=deepy_file_system_access != DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
                                    **everywhere_ui,
                                )
                                self.deepy_allow_read_file_system_choice.change(fn=lambda value: (gr.update(visible=value != DEEPY_FILE_SYSTEM_ACCESS_DISABLED), gr.update(visible=value != DEEPY_FILE_SYSTEM_ACCESS_DISABLED)), inputs=[self.deepy_allow_read_file_system_choice], outputs=[self.deepy_file_system_paths_choice, self.deepy_read_everywhere_choice], show_progress="hidden")
                                self.deepy_type_choice.change(fn=lambda value: tuple(gr.update(**settings) for settings in deepy_filesystem_ui_state(value)), inputs=[self.deepy_type_choice], outputs=[self.deepy_allow_read_file_system_choice, self.deepy_read_everywhere_choice], show_progress="hidden")
                            with gr.Tab("Guidance", id="guidance", visible=deepy_type_default == DEEPY_TYPE_PRIME) as self.deepy_guidance_tab:
                                self.deepy_prime_custom_system_prompt_choice = gr.Textbox(
                                    value=prime_guidance_value,
                                    lines=7,
                                    label="Deepy Prime User Guidance",
                                    info="Treated as the user's standing preferences and appended to Deepy Prime's trusted system instructions.",
                                )
                            with gr.Tab("MCP Servers", id="mcp_servers", visible=deepy_type_default == DEEPY_TYPE_PRIME) as self.deepy_mcp_servers_tab:
                                self.deepy_prime_mcp_servers_choice = gr.Textbox(
                                    value=json.dumps(self.server_config.get(DEEPY_PRIME_MCP_SERVERS_KEY, {}), indent=2),
                                    lines=8,
                                    label="External MCP Servers (JSON)",
                                    info='Optional servers keyed by name. Use {"transport":"stdio","command":"...","args":[]} or {"transport":"streamable-http","url":"..."}.',
                                )
                                self.deepy_mcp_auto_discover_paths_choice = gr.Checkbox(
                                    value=normalize_deepy_mcp_auto_discover_paths(self.server_config.get(DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT)),
                                    label="Allow Searching for Changed MCP Executable Paths",
                                    info="Disabled by default. If a versioned stdio executable disappears, search only sibling version folders under the same runtime root for the newest exact filename.",
                                )
                            with gr.Tab("Python", id="python", visible=deepy_type_default == DEEPY_TYPE_PRIME) as self.deepy_python_tab:
                                self.deepy_python_sandbox_choice = gr.Checkbox(
                                    value=normalize_deepy_python_sandbox(self.server_config.get(DEEPY_PYTHON_SANDBOX_KEY, DEEPY_PYTHON_SANDBOX_DEFAULT)) == "enabled",
                                    label="Enable Python Sandbox",
                                    info="On by default. Lets Deepy Prime and MCP agents run Python (numpy, PIL, SVG, matplotlib, data and text) in a separate, restartable interpreter to compute layouts, compose images and build clips. It can only write inside the session's sandbox folder, reads Gallery media and documents, and has no network, processes or GPU; reads of other files are refused by the interpreter rather than by the operating system.",
                                )
                                apps_level = normalize_python_apps(self.server_config.get(PYTHON_APPS_KEY))["development"]
                                self.python_apps_choice = gr.Dropdown(
                                    choices=[("Off", "off"), ("Web Apps", "apps")],  # "Web Apps and Plugins" joins when the plugin tier ships
                                    value="apps" if apps_level == "plugins" else apps_level,
                                    label="App Development (Advanced)",
                                    info="Off by default; needs the Python Sandbox. Web Apps lets Deepy Prime and MCP agents build interactive animations, games and three.js scenes, with pictures, 3D models and music made by WanGP, that you play in the chat and keep in the Apps gallery. Apps run in your browser in an isolated frame: they cannot reach WanGP's controls, your files or the internet, and keep only their own saved data.",
                                )
                            with gr.Tab("Internet Access", id="internet_access", visible=deepy_type_default == DEEPY_TYPE_PRIME) as self.deepy_internet_access_tab:
                                self.deepy_web_search_choice = gr.Checkbox(
                                    value=normalize_deepy_web_search(self.server_config.get(DEEPY_WEB_SEARCH_KEY, DEEPY_WEB_SEARCH_DEFAULT)) == "enabled",
                                    label="Enable Web Search",
                                    info="Off by default. Lets Deepy Prime and MCP agents look things up on the web (search results, news, images, videos and the text of public pages), download files into the session documents, and find and install LoRAs from Civitai and Hugging Face, through free public services without an account. Queries leave this machine; what comes back is treated as untrusted content, and local or private network addresses cannot be opened.",
                                )
                                def secret_value(name):
                                    try:
                                        return configured_secret(self.server_config, name)
                                    except notifications.SecureStorageError:
                                        return ""
                                self.civitai_api_key_choice = gr.Textbox(
                                    value=secret_value("civitai"),
                                    type="password",
                                    label="Civitai API Key (optional)",
                                    info="Needed only for Deepy to download LoRAs from Civitai (searching works without it): create one at civitai.com/user/account under API Keys. It is sent to civitai.com only.",
                                )
                                self.civitai_api_key_secure_choice = gr.Checkbox(
                                    value=bool(self.server_config.get(CIVITAI_API_KEY_SECURE_KEY, not str(self.server_config.get(CIVITAI_API_KEY_KEY, "") or "").strip())),
                                    label="Store Civitai API Key in OS Credential Manager",
                                    info="Recommended. wgp_config.json then keeps only a reference; uncheck to save the key in plaintext in wgp_config.json.",
                                )
                                self.huggingface_token_choice = gr.Textbox(
                                    value=secret_value("huggingface"),
                                    type="password",
                                    label="Hugging Face Token (optional)",
                                    info="Only for LoRAs in gated or private Hugging Face repositories (public ones need no token): a read token from huggingface.co/settings/tokens. Sent to huggingface.co only; a wrong token blocks public downloads too until it is fixed. Leave empty to use the Hugging Face CLI login when there is one.",
                                )
                                self.huggingface_token_secure_choice = gr.Checkbox(
                                    value=bool(self.server_config.get(HUGGINGFACE_TOKEN_SECURE_KEY, not str(self.server_config.get(HUGGINGFACE_TOKEN_KEY, "") or "").strip())),
                                    label="Store Hugging Face Token in OS Credential Manager",
                                    info="Recommended. wgp_config.json then keeps only a reference; uncheck to save the token in plaintext in wgp_config.json.",
                                )
                        deepy_prime_tabs = [self.deepy_guidance_tab, self.deepy_mcp_servers_tab, self.deepy_python_tab, self.deepy_internet_access_tab]
                        self.deepy_type_choice.change(
                            fn=lambda value: (gr.update(selected="filesystem") if value != DEEPY_TYPE_PRIME else gr.update(), *(gr.update(visible=value == DEEPY_TYPE_PRIME) for _ in deepy_prime_tabs)),
                            inputs=[self.deepy_type_choice], outputs=[self.deepy_settings_tabs, *deepy_prime_tabs], show_progress="hidden",
                        )
                        self.deepy_requirement_md = gr.Markdown(value=deepy_requirement_message(self.server_config) if deepy_type_default != DEEPY_TYPE_DISABLED else "")

                with gr.Tab('Agents'):
                    self.mcp_webui_choice = gr.Checkbox(value=self.server_config.get('mcp_webui', False), label='Enable MCP with Web UI (requires restart)', info='Allow external agents to connect to WanGP. Off by default.')
                    self.mcp_port_choice = gr.Number(value=self.server_config.get('mcp_port', DEFAULT_PORT), precision=0, minimum=1, maximum=65535, label='MCP Server Port (requires restart)', info='--mcp-port overrides this port for the current launch.')
                    mcp_workspace_mode = self.server_config.get('mcp_workspace_mode', 'auto')
                    if mcp_workspace_mode == 'auto':
                        mcp_workspace_mode = 'per_agent' if self.server_config.get('workspace_storage_mode', 'shared') == 'dedicated' else 'shared'
                    self.mcp_workspace_mode_choice = gr.Dropdown(
                        choices=[('Use Selected Workspace (agents keep the workspace selected when they first connect)', 'shared'), ('Create a Workspace per Agent (each agent automatically gets its own workspace)', 'per_agent')],
                        value=mcp_workspace_mode, label='MCP Agent Workspace (requires restart)',
                    )
                    self.mcp_workspace_visibility_choice = gr.Dropdown(
                        choices=[('Their Own Workspaces', 'own'), ('Their Own and Your Workspaces', 'own_and_user'), ('All Workspaces, Including Deepy Conversations and Other Agents', 'all')],
                        value=self.server_config.get('mcp_workspace_visibility', 'own_and_user'), label='Workspaces Agents Can See',
                        info='Agents can list and work in these workspaces. Your workspaces are the ones you created; Deepy conversation workspaces are named after your requests.',
                    )
                    self.mcp_visual_inspection_choice = gr.Dropdown(
                        choices=[('Agents Look Themselves (they receive the images and video frames they choose)', 'agent'), ("WanGP's Visual Inspector Answers Their Questions (only text is sent back)", 'wangp'), ('Both', 'both')],
                        value=self.server_config.get('mcp_visual_inspection', 'agent'), label='Visual Inspection by Agents',
                        info="Agents with their own vision judge images best themselves. WanGP's Visual Inspector keeps images on this machine and saves network traffic: it uses the prompt enhancer's Qwen VL model (a running generation pauses briefly while it answers), or Deepy's remote engine when Deepy runs on one.",
                    )

                with gr.Tab("Outputs"):
                    self.video_container_choice = gr.Dropdown(choices=VIDEO_CONTAINER_CHOICES, value=self.server_config.get("video_container", "mp4"), label="Video Container")
                    self.video_output_codec_choice = gr.Dropdown(choices=SDR_VIDEO_CODEC_CHOICES, value=self.server_config.get("video_output_codec", "libx264_8"), label="SDR Video Codec")
                    self.rgba_video_output_choice = gr.Dropdown(choices=RGBA_VIDEO_OUTPUT_CHOICES, value=self.server_config.get("rgba_video_output", "png_zip"), label="RGBA Video Output", info="Alpha models save transparency beside the gallery video as lossless PNG frames in a ZIP, or as a ProRes 4444 MOV for video editors. This choice is independent of the gallery codec and container.")
                    self.hdr_video_crf_choice = gr.Dropdown(
                        choices=[
                            ("Low (x265 CRF 14)", 14),
                            ("Medium (x265 CRF 8)", 8),
                            ("High (x265 CRF 4)", 4),
                        ],
                        value=self.server_config.get("hdr_video_crf", 8),
                        label="HDR Video Codec",
                    )
                    self.image_output_codec_choice = gr.Dropdown(choices=[("JPEG Q85", 'jpeg_85'), ("WEBP Q85", 'webp_85'), ("JPEG Q95", 'jpeg_95'), ("WEBP Q95", 'webp_95'), ("WEBP Lossless", 'webp_lossless'), ("PNG Lossless", 'png')], value=self.server_config.get("image_output_codec", "jpeg_95"), label="Image Codec")
                    self.audio_output_codec_choice = gr.Dropdown(
                        choices=[
                            ("AAC 128 kbps", "aac_128"),
                            ("AAC 192 kbps", "aac_192"),
                            ("AAC 256 kbps (High Quality, Recommended)", "aac_256"),
                            ("AAC 320 kbps (Very High Quality)", "aac_320"),
                            ("ALAC Lossless (preview/playback compatibility may be limited)", "alac"),
                            ("FLAC 16-bit (MP4 / MKV)", "flac"),
                        ],
                        value=self.server_config.get("audio_output_codec", "aac_128"),
                        visible=True,
                        label="Audio Codec to use for MP4/MOV/MKV container",
                    )
                    audio_standalone_default = self.server_config.get("audio_stand_alone_output_codec", "wav")
                    if audio_standalone_default == "mp3":
                        audio_standalone_default = "mp3_192"
                    self.audio_stand_alone_output_codec_choice = gr.Dropdown(
                        choices=[
                            ("WAV (Lossless)", "wav"),
                            ("FLAC 16-bit (Lossless)", "flac"),
                            ("MP3 128 kbps", "mp3_128"),
                            ("MP3 192 kbps", "mp3_192"),
                            ("MP3 320 kbps", "mp3_320"),
                        ],
                        value=audio_standalone_default,
                        visible=True,
                        label="Audio Codec to use for standalone audio files",
                    )
                    with gr.Group():
                        self.metadata_choice = gr.Dropdown(
                            choices=[("Export JSON files", "json"), ("Embed metadata in file (Exif/tag)", "metadata"), ("Export JSON files + Embed metadata in file (Exif/tag)", "json_metadata"), ("None", "none")],
                            value=self.server_config.get("metadata_type", "metadata"), label="Metadata Handling"
                        )
                        self.embed_source_images_choice = gr.Checkbox(
                            value=self.server_config.get("embed_source_images", False),
                            label="Embed Source Images",
                            info="Saves the source images inside MP4/MOV/MKV videos and GLB 3D assets (their object image, reused by late 3D post-processing)"
                        )
                    self.keep_intermediate_sliding_windows_choice = gr.Dropdown(
                        choices=[("Yes", 1), ("No", 0)],
                        value=self.server_config.get("keep_intermediate_sliding_windows", 1),
                        label="Keep Intermediate Sliding Windows"
                    )
                    self.workspace_storage_choice = gr.Radio(choices=[('Shared Output Folders', 'shared'), ('Dedicated Folder per Workspace', 'dedicated')], value=self.server_config.get('workspace_storage_mode', 'shared'), label='Default Storage for New Workspaces', info='Existing workspaces keep their storage mode. Convert them in the workspace manager. Dedicated workspaces use the Workspaces Root instead of the shared output folders below.')
                    self.workspaces_root_choice = gr.Textbox(label='Workspaces Root (requires restart)', value=self.server_config.get('workspaces_dir', 'workspaces'), info='Contains workspace JSON files and dedicated media folders. Move the complete existing root before changing this path. --workspaces-dir overrides this value.')
                    gr.Markdown('### Shared Workspace Output Folders\nExisting shared workspaces keep using these folders, even when new workspaces use dedicated folders. To move an existing workspace, convert it in Workspace Manager; changing the default above does not move its media.')
                    self.video_save_path_choice = gr.Textbox(label="Shared Video Output Folder (requires restart)", value=self.save_path)
                    self.image_save_path_choice = gr.Textbox(label="Shared Image Output Folder (requires restart)", value=self.image_save_path)
                    self.audio_save_path_choice = gr.Textbox(label="Shared Audio Output Folder (requires restart)", value=self.audio_save_path)
                    self.mesh_save_path_choice = gr.Textbox(label="Shared 3D Output Folder (requires restart)", value=self.mesh_save_path)

                with gr.Tab("Notifications"):
                    self.notification_sound_enabled_choice = gr.Dropdown(choices=[("On", 1), ("Off", 0)], value=self.server_config.get("notification_sound_enabled", 0), label="Notification Sound")
                    self.notification_sound_volume_choice = gr.Slider(0, 100, value=self.server_config.get("notification_sound_volume", 50), step=5, label="Notification Volume")
                    self.notification_config_ui = notifications.create_config_ui(gr, self.server_config)
                    self.notification_apprise_urls_choice, self.notification_secure_storage_choice, self.notification_on_generation_choice, self.notification_on_queue_complete_choice, self.notification_on_queue_interrupted_choice = self.notification_config_ui.save_components

            self.msg = gr.Markdown()
            with gr.Row():
                self.apply_btn = gr.Button("Save Settings")

        def update_deepy_requirement(enhancer_enabled_choice, deepy_type_choice, deepy_context_tokens_choice, deepy_compaction_type_choice, deepy_llm_engine_choice):
            if deepy_type_choice == DEEPY_TYPE_DISABLED:
                return ""
            runtime_config = dict(self.server_config)
            deepy_enabled_choice, deepy_type_choice = split_deepy_mode(deepy_type_choice)
            runtime_config["enhancer_enabled"] = enhancer_enabled_choice
            runtime_config[DEEPY_ENABLED_KEY] = deepy_enabled_choice
            runtime_config[DEEPY_TYPE_KEY] = deepy_type_choice
            runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = deepy_context_tokens_choice
            runtime_config[DEEPY_COMPACTION_TYPE_KEY] = deepy_compaction_type_choice
            runtime_config[DEEPY_COMPACTION_THINKING_KEY] = deepy_compaction_type_choice == DEEPY_COMPACTION_CHOICE_THINKING
            runtime_config[LLM_CONFIG_KEY] = {**normalize_llm_config(runtime_config), "deepy": deepy_llm_engine_choice}
            return deepy_requirement_message(runtime_config)

        deepy_requirement_inputs = [self.enhancer_enabled_choice, self.deepy_type_choice, self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.deepy_llm_engine_choice]
        for requirement_input in (self.enhancer_enabled_choice, self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.deepy_llm_engine_choice):
            requirement_input.input(fn=update_deepy_requirement, inputs=deepy_requirement_inputs, outputs=[self.deepy_requirement_md], show_progress="hidden")

        def enforce_deepy_prime_requirements(deepy_type_choice, deepy_context_tokens_choice, deepy_compaction_type_choice, enhancer_enabled_choice, deepy_kv_cache_quantization_choice, deepy_llm_engine_choice):
            if deepy_type_choice == DEEPY_TYPE_DISABLED:
                return gr.update(), gr.update(), "", DEEPY_TYPE_DISABLED, gr.update()
            qwen_local = not is_remote_engine(deepy_llm_engine_choice) and enhancer_enabled_choice in QWEN_PROMPT_ENHANCER_IDS
            runtime_config = dict(self.server_config)
            deepy_enabled_choice, deepy_type_choice = split_deepy_mode(deepy_type_choice)
            runtime_config["enhancer_enabled"] = enhancer_enabled_choice
            runtime_config[DEEPY_ENABLED_KEY] = deepy_enabled_choice
            runtime_config[DEEPY_TYPE_KEY] = deepy_type_choice
            if qwen_local and normalize_deepy_type(deepy_type_choice) == DEEPY_TYPE_PRIME:
                deepy_context_tokens_choice = max(deepy_compaction_min_tokens(deepy_compaction_type_choice), normalize_deepy_context_tokens(deepy_context_tokens_choice))
                if normalize_deepy_compaction_type(deepy_compaction_type_choice) != DEEPY_COMPACTION_TYPE_SUMMARIZE:
                    deepy_compaction_type_choice = DEEPY_COMPACTION_TYPE_SUMMARIZE
            runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = deepy_context_tokens_choice
            runtime_config[DEEPY_COMPACTION_TYPE_KEY] = deepy_compaction_type_choice
            runtime_config[DEEPY_COMPACTION_THINKING_KEY] = deepy_compaction_type_choice == DEEPY_COMPACTION_CHOICE_THINKING
            runtime_config[LLM_CONFIG_KEY] = {**normalize_llm_config(runtime_config), "deepy": deepy_llm_engine_choice}
            context_update = gr.update(value=deepy_context_tokens_choice, label=format_deepy_context_tokens_label(enhancer_enabled_choice, deepy_context_tokens_choice, deepy_kv_cache_quantization_choice)) if qwen_local else gr.update()
            compaction_update = gr.update(value=deepy_compaction_type_choice) if qwen_local else gr.update()
            recommendation = deepy_prime_upgrade_message(runtime_config)
            return context_update, compaction_update, deepy_requirement_message(runtime_config), deepy_mode_from_config(deepy_enabled_choice, deepy_type_choice), gr.update(value=recommendation, visible=bool(recommendation))

        self.deepy_type_choice.input(fn=enforce_deepy_prime_requirements, inputs=[self.deepy_type_choice, self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.enhancer_enabled_choice, self.deepy_kv_cache_quantization_choice, self.deepy_llm_engine_choice], outputs=[self.deepy_context_tokens_choice, self.deepy_compaction_type_choice, self.deepy_requirement_md, self.deepy_type_value, self.deepy_prime_recommendation], show_progress="hidden")

        def update_deepy_context_label(enhancer_enabled_choice, deepy_context_tokens_choice, deepy_kv_cache_quantization_choice, deepy_type_choice, deepy_llm_engine_choice, deepy_kv_cache_exact_choice):
            if deepy_type_choice == DEEPY_TYPE_DISABLED or is_remote_engine(deepy_llm_engine_choice) or enhancer_enabled_choice not in QWEN_PROMPT_ENHANCER_IDS:
                return gr.update()
            layout = join_deepy_kv_cache_quantization(deepy_kv_cache_quantization_choice, deepy_kv_cache_exact_choice)
            return gr.update(label=format_deepy_context_tokens_label(enhancer_enabled_choice, deepy_context_tokens_choice, layout))

        deepy_context_label_inputs = [self.enhancer_enabled_choice, self.deepy_context_tokens_choice, self.deepy_kv_cache_quantization_choice, self.deepy_type_choice, self.deepy_llm_engine_choice, self.deepy_kv_cache_exact_choice]
        self.deepy_kv_cache_exact_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")
        self.deepy_kv_cache_quantization_choice.input(fn=lambda body: gr.update(visible=deepy_kv_cache_quantization_is_mixed(body)), inputs=[self.deepy_kv_cache_quantization_choice],
                                                      outputs=[self.deepy_kv_cache_exact_choice], show_progress="hidden")
        self.enhancer_enabled_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")
        self.deepy_context_tokens_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")
        self.deepy_kv_cache_quantization_choice.input(fn=update_deepy_context_label, inputs=deepy_context_label_inputs, outputs=[self.deepy_context_tokens_choice], show_progress="hidden")

        def update_speculative_decoding_choice(enhancer_enabled_choice, enhancer_quantization_choice, method, tokens):
            _, quantization, _ = prompt_enhancer_quantization_ui_state(enhancer_enabled_choice, enhancer_quantization_choice)
            choices, method, counts, tokens = speculative_decoding_ui_state(enhancer_enabled_choice, quantization, method, tokens)
            return gr.update(choices=choices, value=method, interactive=not self.args.lock_config), gr.update(choices=counts, value=tokens, interactive=bool(counts) and not self.args.lock_config)

        def update_enhancer_quantization_choice(enhancer_enabled_choice, enhancer_quantization_choice):
            choices, value, visible = prompt_enhancer_quantization_ui_state(enhancer_enabled_choice, enhancer_quantization_choice)
            return gr.update(choices=choices, value=value, visible=visible)

        self.enhancer_enabled_choice.input(fn=update_enhancer_quantization_choice, inputs=[self.enhancer_enabled_choice, self.enhancer_quantization_choice], outputs=[self.enhancer_quantization_choice], show_progress="hidden")
        speculative_inputs = [self.enhancer_enabled_choice, self.enhancer_quantization_choice, self.enhancer_speculative_decoding_choice, self.enhancer_speculative_tokens_choice]
        speculative_outputs = [self.enhancer_speculative_decoding_choice, self.enhancer_speculative_tokens_choice]
        self.enhancer_enabled_choice.change(fn=update_speculative_decoding_choice, inputs=speculative_inputs, outputs=speculative_outputs, show_progress="hidden")
        self.enhancer_quantization_choice.change(fn=update_speculative_decoding_choice, inputs=speculative_inputs, outputs=speculative_outputs, show_progress="hidden")
        self.enhancer_speculative_decoding_choice.input(fn=update_speculative_decoding_choice, inputs=speculative_inputs, outputs=speculative_outputs, show_progress="hidden")

        def update_remote_engine_ui(deepy_engine, enhancer_quantization, deepy_type):
            view = normalize_llm_config(self.server_config)
            view["deepy"] = deepy_engine
            runtime_config = {**self.server_config, LLM_CONFIG_KEY: view}
            deepy_remote = is_remote_engine(resolve_role_engine(runtime_config, "deepy"))
            active = {resolve_role_engine(runtime_config, "deepy")}
            quantization_choices, quantization_value, quantization_visible = prompt_enhancer_quantization_ui_state(local_enhancer_id(deepy_engine), enhancer_quantization)
            qwen_local = not deepy_remote and local_enhancer_id(deepy_engine) in QWEN_PROMPT_ENHANCER_IDS
            deepy_enabled = deepy_type != DEEPY_TYPE_DISABLED
            return (
                privacy_warning(runtime_config),
                gr.update(visible=deepy_remote),
                gr.update(visible=ENGINE_CODEX in active), gr.update(visible=ENGINE_CLAUDE in active), gr.update(visible=ENGINE_OPENCODE in active),
                gr.update(value=local_enhancer_id(deepy_engine), visible=False), gr.update(choices=quantization_choices, value=quantization_value, visible=quantization_visible and not deepy_remote),
                gr.update(visible=qwen_local), gr.update(visible=qwen_local), gr.update(visible=not deepy_remote), gr.update(visible=deepy_enabled and not deepy_remote),
                gr.update(visible=deepy_enabled and qwen_local), gr.update(visible=deepy_enabled and qwen_local), gr.update(visible=deepy_enabled), gr.update(visible=qwen_local),
            )

        remote_engine_state_inputs = [self.deepy_llm_engine_choice, self.enhancer_quantization_choice, self.deepy_type_choice]
        remote_engine_outputs = [
            self.remote_llm_warning_md, self.remote_llm_auth_md, self.codex_config_ui.group, self.claude_config_ui.group, self.opencode_config_ui.group,
            self.enhancer_enabled_choice, self.enhancer_quantization_choice,
            self.enhancer_speculative_row, self.enhancer_decoding_row, self.enhancer_sampling_row, self.deepy_vram_row,
            self.deepy_context_row, self.deepy_compaction_column, self.deepy_options_group, self.deepy_repetition_penalty_choice,
        ]
        self.deepy_llm_engine_choice.input(fn=update_remote_engine_ui, inputs=remote_engine_state_inputs, outputs=remote_engine_outputs, show_progress="hidden")
        self.deepy_type_choice.change(fn=update_remote_engine_ui, inputs=remote_engine_state_inputs, outputs=remote_engine_outputs, show_progress="hidden")

        bind_codex_config_ui(gr, self.codex_config_ui, self.server_config, self.server_config_filename)
        bind_claude_config_ui(gr, self.claude_config_ui, self.server_config, self.server_config_filename)
        bind_opencode_config_ui(gr, self.opencode_config_ui, self.server_config, self.server_config_filename)
        self.process_queues_when_browser_unfocused_choice.change(
            fn=None,
            inputs=[self.process_queues_when_browser_unfocused_choice],
            js="""
                (enabled) => {
                    if (window.__gradioFocusQueuePatch) {
                        window.__gradioFocusQueuePatch.enableBackgroundScheduler = Number(enabled) !== 0;
                    }
                }
            """,
            queue=False,
            show_progress="hidden",
        )

        inputs = [
            self.state,
            self.transformer_types_choices, self.invert_model_selection_choice, self.show_all_model_filter_choice, self.model_hierarchy_type_choice, self.fit_canvas_choice,
            self.attention_choice, self.generation_preview_choice, self.preload_model_policy_choice, self.clear_file_list_choice, self.multi_prompts_gen_type_choice, self.keep_intermediate_sliding_windows_choice,
            self.display_stats_choice, self.max_frames_multiplier_choice, self.keep_resolution_on_model_switch_choice, self.enable_4k_resolutions_choice, self.checkpoints_paths_choice, self.loras_root_choice, self.save_queue_if_crash_choice,
            self.UI_theme_choice, self.floating_generate_button_choice, self.queue_color_scheme_choice, self.process_queues_when_browser_unfocused_choice, self.prevent_power_throttling_choice, self.compute_cache_choice, self.compute_cache_path_choice,
            self.quantization_choice, self.transformer_dtype_policy_choice, self.mixed_precision_choice,
            self.text_encoder_quantization_choice, self.compile_choice,
            self.depth_anything_v2_variant_choice,
            self.vae_config_choice, self.boost_choice, self.int8_kernels_choice, self.fp8_kernels_choice, self.kernel_precision_choice, self.attention_head_split_choice,
            self.video_profile_choice, self.image_profile_choice, self.audio_profile_choice, self.video_preload_choice, self.image_preload_choice, self.audio_preload_choice,
            self.video_preload_mode_choice, self.image_preload_mode_choice, self.audio_preload_mode_choice,
            self.postprocessor_preload_mode_choice, self.postprocessor_preload_choice,
            self.smart_memory_pinning_choice, self.perc_reserved_mem_max_choice, self.vram_allocator_choice, self.ram_allocator_choice, self.max_reserved_loras_choice,
            self.llm_cpu_compute_choice, self.deepy_llm_engine_choice,
            *self.codex_config_ui.save_components,
            *self.claude_config_ui.save_components,
            *self.opencode_config_ui.save_components,
            self.enhancer_enabled_choice, self.enhancer_quantization_choice, self.enhancer_speculative_decoding_choice, self.enhancer_speculative_tokens_choice, self.enhancer_mode_choice,
            self.prompt_enhancer_temperature_choice, self.prompt_enhancer_top_p_choice, self.prompt_enhancer_randomize_seed_choice,
            self.matanyone_version_choice, self.whisper_model_choice,
            self.deepy_type_choice, self.deepy_vram_mode_choice, self.deepy_voice_language_choice, self.voice_mode_choice, self.deepy_allow_read_file_system_choice, self.deepy_file_system_paths_choice, self.deepy_read_everywhere_choice,
            self.deepy_context_tokens_choice, self.deepy_kv_cache_quantization_choice, self.deepy_kv_cache_exact_choice, self.deepy_compaction_type_choice, self.deepy_repetition_penalty_choice, self.deepy_zero_custom_system_prompt_choice, self.deepy_prime_custom_system_prompt_choice, self.deepy_prime_mcp_servers_choice, self.deepy_mcp_auto_discover_paths_choice, self.deepy_python_sandbox_choice, self.python_apps_choice, self.deepy_web_search_choice, self.civitai_api_key_choice, self.civitai_api_key_secure_choice, self.huggingface_token_choice, self.huggingface_token_secure_choice,
            self.video_container_choice, self.video_output_codec_choice, self.rgba_video_output_choice, self.hdr_video_crf_choice, self.image_output_codec_choice, self.audio_output_codec_choice, self.audio_stand_alone_output_codec_choice,
            self.metadata_choice, self.embed_source_images_choice,
            self.video_save_path_choice, self.image_save_path_choice, self.audio_save_path_choice, self.mesh_save_path_choice,
            self.workspace_storage_choice, self.workspaces_root_choice, self.mcp_workspace_mode_choice, self.mcp_port_choice, self.mcp_webui_choice, self.mcp_workspace_visibility_choice, self.mcp_visual_inspection_choice,
            self.notification_sound_enabled_choice, self.notification_sound_volume_choice, self.notification_apprise_urls_choice, self.notification_secure_storage_choice, self.notification_on_generation_choice, self.notification_on_queue_complete_choice, self.notification_on_queue_interrupted_choice,
            *self.audio_processor_config_components,
            *self.temporal_upsampler_config_components,
            *self.upsampler_config_components,
            self.resolution
        ]

        # These configuration input callbacks only derive presentation/choices.
        view_events = [event for index, event in gr.context.get_blocks_context().fns.items() if index >= first_event and event.fn and event.outputs and all(kind in ('input', 'change') for _, kind in event.targets)]
        self.synced_form = GradioForm(self.state.value.service.forms, 'configuration', inputs[1:-1], view_events=view_events)
        saved = self.synced_form.bind_save(
            self.apply_btn.click, lambda *values: self._save_changes(*values[:-1], overrides=values[-1]),
            inputs=[*inputs, self.overrides],
            outputs=[
                self.msg,
                self.model_description,
                self.header,
                self.model_family,
                self.model_base_type_choice,
                self.model_choice,
                self.refresh_form_trigger,
                self.assistant_launcher_host,
                self.assistant_panel,
                self.assistant_chat_event,
            ]
        )
        # Server-side follow-ups chain on the saved step: Gradio fires no .then after the script-only step below.
        # App Development (Advanced) shows or hides the Apps gallery.
        saved.then(lambda: gr.update(visible=python_apps_level() != "off"), outputs=[self.apps_gallery_tab], show_progress="hidden", queue=False)
        saved.then(
            fn=None, inputs=None, outputs=None,
            js="""() => {
                const button = document.querySelector('#assistant_chat_session_refresh_button button, #assistant_chat_session_refresh_button');
                if (button) button.click();
            }""",
        )

        if self.model_filter is not None:
            def refresh_model_filters(state, select_visible_model=False):
                deps = self._get_dropdown_deps()
                selected_filter, _ = model_dropdowns.reconcile_model_output_filter_for_model_type(deps, state, state["model_type"])
                return gr.update(value=model_output_filter.render_filter(selected_filter, model_dropdowns.available_output_filters(deps))), *self.generate_dropdown_model_list(state["model_type"], state, select_visible_model=select_visible_model)

            filter_outputs = [self.model_filter.html, self.model_family, self.model_base_type_choice, self.model_choice]
            save_event = saved.then(lambda state: refresh_model_filters(state, select_visible_model=True), inputs=[self.state], outputs=filter_outputs, show_progress="hidden")
            self.show_all_model_filter_choice.change(refresh_model_filters, inputs=[self.state], outputs=filter_outputs, show_progress="hidden")
        else:
            save_event = saved

        def switch_to_visible_model(state, model_type):
            if model_type is None or model_type == state["model_type"]:
                return gr.update()
            return self._model_choice_target_value(model_type)

        save_event.then(switch_to_visible_model, inputs=[self.state, self.model_choice], outputs=[self.model_choice_target], show_progress="hidden")

        def release_ram_and_notify(state):
            unload_models_from_ram(
                state,
                server_config=self.server_config,
                any_GPU_process_running=self.any_GPU_process_running,
                release_deepy_vram=self.release_deepy_vram,
                reset_prompt_enhancer=self.reset_prompt_enhancer,
                reset_prompt_enhancer_if_requested=self.reset_prompt_enhancer_if_requested,
                release_extensions=self.release_extension_offloadobjs,
                release_model=self.release_model,
            )

        self.release_RAM_btn.click(fn=release_ram_and_notify, inputs=[self.state])
        return [self.release_RAM_btn]

    def _save_changes(self, state, *args, overrides=""):
        gen_in_progress = self.is_generation_in_progress()
        # return "<div style='color:red; text-align:center;'>Unable to change config when a generation is in progress.</div>", *[gr.update()]*5

        if self.args.lock_config:
            return "<div style='color:red; text-align:center;'>Configuration is locked by command-line arguments.</div>", *[gr.update()]*9

        old_server_config = config_snapshot(self.server_config)
        audio_processor_component_count = len(getattr(self, "audio_processor_config_components", []))
        temporal_upsampler_component_count = len(getattr(self, "temporal_upsampler_config_components", []))
        upsampler_component_count = len(getattr(self, "upsampler_config_components", []))
        extension_component_count = audio_processor_component_count + temporal_upsampler_component_count + upsampler_component_count
        if extension_component_count:
            extension_config_values = args[-extension_component_count - 1:-1]
            audio_processor_config_values = extension_config_values[:audio_processor_component_count]
            temporal_upsampler_config_values = extension_config_values[audio_processor_component_count:audio_processor_component_count + temporal_upsampler_component_count]
            upsampler_config_values = extension_config_values[audio_processor_component_count + temporal_upsampler_component_count:]
            fixed_args = args[:-extension_component_count - 1] + (args[-1],)
        else:
            audio_processor_config_values = []
            temporal_upsampler_config_values = []
            upsampler_config_values = []
            fixed_args = args

        (
            transformer_types_choices, invert_model_selection_choice, show_all_model_filter_choice, model_hierarchy_type_choice, fit_canvas_choice,
            attention_choice, generation_preview_choice, preload_model_policy_choice, clear_file_list_choice, multi_prompts_gen_type_choice, keep_intermediate_sliding_windows_choice,
            display_stats_choice, max_frames_multiplier_choice, keep_resolution_on_model_switch_choice, enable_4k_resolutions_choice, checkpoints_paths_choice, loras_root_choice, save_queue_if_crash_choice,
            UI_theme_choice, floating_generate_button_choice, queue_color_scheme_choice, process_queues_when_browser_unfocused_choice, prevent_power_throttling_choice, compute_cache_choice, compute_cache_path_choice,
            quantization_choice, transformer_dtype_policy_choice, mixed_precision_choice,
            text_encoder_quantization_choice, compile_choice,
            depth_anything_v2_variant_choice,
            vae_config_choice, boost_choice, int8_kernels_choice, fp8_kernels_choice, kernel_precision_choice, attention_head_split_choice,
            video_profile_choice, image_profile_choice, audio_profile_choice, video_preload_choice, image_preload_choice, audio_preload_choice,
            video_preload_mode_choice, image_preload_mode_choice, audio_preload_mode_choice,
            postprocessor_preload_mode_choice, postprocessor_preload_choice,
            smart_memory_pinning_choice, perc_reserved_mem_max_choice, vram_allocator_choice, ram_allocator_choice, max_reserved_loras_choice,
            llm_cpu_compute_choice, deepy_llm_engine_choice,
            codex_executable_choice, codex_model_choice, codex_reasoning_effort_choice,
            claude_executable_choice, claude_model_choice, claude_reasoning_effort_choice,
            opencode_executable_choice, opencode_base_url_choice, opencode_provider_choice, opencode_model_choice, opencode_reasoning_effort_choice, opencode_config_choice,
            enhancer_enabled_choice, enhancer_quantization_choice, enhancer_speculative_decoding_choice, enhancer_speculative_tokens_choice, enhancer_mode_choice,
            prompt_enhancer_temperature_choice, prompt_enhancer_top_p_choice, prompt_enhancer_randomize_seed_choice,
            matanyone_version_choice, whisper_model_choice,
            deepy_type_choice, deepy_vram_mode_choice, deepy_voice_language_choice, voice_mode_choice, deepy_allow_read_file_system_choice, deepy_file_system_paths_choice, deepy_read_everywhere_choice,
            deepy_context_tokens_choice, deepy_kv_cache_quantization_choice, deepy_kv_cache_exact_choice, deepy_compaction_type_choice, deepy_repetition_penalty_choice, deepy_zero_custom_system_prompt_choice, deepy_prime_custom_system_prompt_choice, deepy_prime_mcp_servers_choice, deepy_mcp_auto_discover_paths_choice, deepy_python_sandbox_choice, python_apps_choice, deepy_web_search_choice, civitai_api_key_choice, civitai_api_key_secure_choice, huggingface_token_choice, huggingface_token_secure_choice,
            video_container_choice, video_output_codec_choice, rgba_video_output_choice, hdr_video_crf_choice, image_output_codec_choice, audio_output_codec_choice, audio_stand_alone_output_codec_choice,
            metadata_choice, embed_source_images_choice,
            save_path_choice, image_save_path_choice, audio_save_path_choice, mesh_save_path_choice,
            workspace_storage_choice, workspaces_root_choice, mcp_workspace_mode_choice, mcp_port_choice, mcp_webui_choice, mcp_workspace_visibility_choice, mcp_visual_inspection_choice,
            notification_sound_enabled_choice, notification_sound_volume_choice, notification_apprise_urls_choice, notification_secure_storage_choice, notification_on_generation_choice, notification_on_queue_complete_choice, notification_on_queue_interrupted_choice,
            last_resolution_choice
        ) = fixed_args

        llm_config_choice = normalize_llm_config(old_server_config)
        llm_config_choice["deepy"] = deepy_llm_engine_choice
        profiles = llm_config_choice["profiles"]
        if deepy_llm_engine_choice == ENGINE_CODEX:
            profiles[ENGINE_CODEX] = codex_profile_from_values(codex_executable_choice, codex_model_choice, codex_reasoning_effort_choice, profiles[ENGINE_CODEX]["model_catalog"])
        elif deepy_llm_engine_choice == ENGINE_CLAUDE:
            profiles[ENGINE_CLAUDE] = claude_profile_from_values(claude_executable_choice, claude_model_choice, claude_reasoning_effort_choice, profiles[ENGINE_CLAUDE]["model_catalog"])
        elif deepy_llm_engine_choice == ENGINE_OPENCODE:
            profiles[ENGINE_OPENCODE] = opencode_profile_from_values(opencode_executable_choice, opencode_base_url_choice, opencode_provider_choice, opencode_model_choice, opencode_reasoning_effort_choice, opencode_config_choice, profiles[ENGINE_OPENCODE]["model_catalog"])

        deepy_enabled_choice, deepy_type_choice = split_deepy_mode(deepy_type_choice)
        validation_config = {**old_server_config, LLM_CONFIG_KEY: llm_config_choice}
        try:
            llm_config_choice = validate_llm_config(validation_config, deepy_enabled=deepy_enabled_choice, deepy_type=deepy_type_choice)
        except ValueError as exc:
            gr.Info(f"Configuration was not saved: {exc}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*9

        deepy_remote = is_remote_engine(deepy_llm_engine_choice)
        if not deepy_remote:
            enhancer_enabled_choice = local_enhancer_id(deepy_llm_engine_choice, enhancer_enabled_choice)
        qwen_local = not deepy_remote and enhancer_enabled_choice in QWEN_PROMPT_ENHANCER_IDS

        if not deepy_remote and int(enhancer_enabled_choice) == QWEN38_PROMPT_ENHANCER_ID and enhancer_quantization_choice not in ("gguf", "gguf_q3", "gguf_q2", "gguf_ptq1"):
            error = "Qwen3.8-27B is available only as GGUF. Select GGUF Q2, Q3, Q4, or Bonsai PTQ1_0 as the Qwen LLM quantization."
            gr.Info(f"Configuration was not saved: {error}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {error}</div>", *[gr.update()]*9
        if not deepy_remote and int(enhancer_enabled_choice) == QWEN38_9B_PROMPT_ENHANCER_ID and enhancer_quantization_choice not in ("gguf", "gguf_q8"):
            enhancer_quantization_choice = "gguf"
        if not deepy_remote and int(enhancer_enabled_choice) in QWEN35_PROMPT_ENHANCER_IDS and enhancer_quantization_choice not in ("quanto_int8", "gguf"):
            error = "Qwen3.5 is available as Quanto Int8 or GGUF Q4."
            gr.Info(f"Configuration was not saved: {error}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {error}</div>", *[gr.update()]*9

        if qwen_local:
            try:
                _, method, _, tokens = speculative_decoding_ui_state(enhancer_enabled_choice, enhancer_quantization_choice, enhancer_speculative_decoding_choice, enhancer_speculative_tokens_choice)
                enhancer_speculative_decoding_choice = validate_prompt_enhancer_speculative_decoding(enhancer_enabled_choice, speculative_decoding_config(method, tokens))
            except ValueError as exc:
                gr.Info(f"Configuration was not saved: {exc}")
                return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*9

        try:
            mcp_port_choice = server_port(mcp_port_choice)
            if mcp_workspace_mode_choice not in ('shared', 'per_agent'):
                raise ValueError('Choose an MCP Agent Workspace option.')
            if deepy_enabled_choice:
                if not deepy_remote and not qwen_local:
                    raise ValueError("Florence 2 is not compatible with Deepy. Select a Qwen model or disable Deepy before saving.")
                if not deepy_remote and deepy_type_choice == DEEPY_TYPE_PRIME and enhancer_enabled_choice not in (QWEN38_PROMPT_ENHANCER_ID, QWEN38_9B_PROMPT_ENHANCER_ID):
                    raise ValueError("Deepy Prime requires a Qwen3.8 VL model.")
                if qwen_local:
                    deepy_compaction_thinking_choice = deepy_compaction_type_choice == DEEPY_COMPACTION_CHOICE_THINKING
                    deepy_compaction_type_choice = normalize_deepy_compaction_type(deepy_compaction_type_choice)
                    deepy_type_choice, deepy_compaction_type_choice, deepy_context_tokens_choice = validate_deepy_version_config(deepy_type_choice, deepy_compaction_type_choice, deepy_context_tokens_choice, enhancer_enabled_choice, compaction_thinking=deepy_compaction_thinking_choice)
                if deepy_type_choice == DEEPY_TYPE_PRIME:
                    deepy_prime_mcp_servers_choice = normalize_deepy_prime_mcp_servers(deepy_prime_mcp_servers_choice)
                if deepy_allow_read_file_system_choice != DEEPY_FILE_SYSTEM_ACCESS_DISABLED:
                    deepy_file_system_paths_choice = normalize_deepy_file_system_paths(deepy_file_system_paths_choice)
                    parse_deepy_file_system_paths(deepy_file_system_paths_choice)
        except ValueError as exc:
            gr.Info(f"Configuration was not saved: {exc}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*9

        if len(checkpoints_paths_choice.strip()) == 0:
            checkpoints_paths = self.fl.default_checkpoints_paths
        else:
            checkpoints_paths = [path.strip() for path in checkpoints_paths_choice.replace("\r", "").split("\n") if len(path.strip()) > 0]

        video_output_error = validate_video_output_settings(video_output_codec_choice, video_container_choice, audio_output_codec_choice)
        if rgba_video_output_choice not in {value for _, value in RGBA_VIDEO_OUTPUT_CHOICES}:
            video_output_error = "Choose RGBA PNG Frames (ZIP) or ProRes 4444 (MOV) for RGBA Video Output."
        if video_output_error is not None:
            gr.Info(f"Configuration was not saved: {video_output_error}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {video_output_error}</div>", *[gr.update()]*9
        compute_cache_path_choice = compute_cache_path_choice.strip() or COMPUTE_CACHE_PATH_DEFAULT
        if compute_cache_choice:
            try:
                os.makedirs(compute_cache_folder(compute_cache_path_choice), exist_ok=True)
            except OSError as exc:
                gr.Info(f"Configuration was not saved: the Compute Cache folder cannot be created ({exc})")
                return f"<div style='color:red; text-align:center;'>Configuration was not saved: the Compute Cache folder cannot be created ({html.escape(str(exc))})</div>", *[gr.update()]*9

        self.fl.set_checkpoints_paths(checkpoints_paths)

        audio_processor_config_update = audio_processor_api.collect_config_update(self.audio_processor_config_bindings, audio_processor_config_values)
        for message in audio_processor_api.validate_config_update_messages(self.audio_processor_config_bindings, audio_processor_config_update):
            gr.Info(message)
        temporal_upsampler_config_update = temporal_upsampler_api.collect_config_update(self.temporal_upsampler_config_bindings, temporal_upsampler_config_values)
        for message in temporal_upsampler_api.validate_config_update_messages(self.temporal_upsampler_config_bindings, temporal_upsampler_config_update):
            gr.Info(message)
        upsampler_config_update = upsampler_api.collect_config_update(self.upsampler_config_bindings, upsampler_config_values)
        for message in upsampler_api.validate_config_update_messages(self.upsampler_config_bindings, upsampler_config_update):
            gr.Info(message)

        try:
            notification_config_update = notifications.prepare_config_update(old_server_config, notification_apprise_urls_choice, notification_secure_storage_choice, notification_on_generation_choice, notification_on_queue_complete_choice, notification_on_queue_interrupted_choice)
            civitai_config_update = {**prepare_secret_update(old_server_config, "civitai", civitai_api_key_choice, civitai_api_key_secure_choice), **prepare_secret_update(old_server_config, "huggingface", huggingface_token_choice, huggingface_token_secure_choice)} if deepy_enabled_choice and deepy_type_choice == DEEPY_TYPE_PRIME else {}
        except notifications.SecureStorageError as exc:
            gr.Info(f"Configuration was not saved: {exc}")
            return f"<div style='color:red; text-align:center;'>Configuration was not saved: {exc}</div>", *[gr.update()]*9

        new_server_config = copy.deepcopy(old_server_config)
        new_server_config.update({
            "attention_mode": attention_choice, "transformer_types": transformer_types_choices,
            model_dropdowns.INVERT_MODEL_SELECTION_CONFIG_KEY: invert_model_selection_choice,
            model_dropdowns.SHOW_ALL_MODEL_FILTER_CONFIG_KEY: show_all_model_filter_choice,
            "generation_preview": generation_preview_choice,
            "text_encoder_quantization": text_encoder_quantization_choice, "save_path": save_path_choice,
            "image_save_path": image_save_path_choice, "audio_save_path": audio_save_path_choice, "mesh_save_path": mesh_save_path_choice,
            'workspace_storage_mode': workspace_storage_choice, 'workspaces_dir': workspaces_root_choice.strip() or 'workspaces', 'mcp_workspace_mode': mcp_workspace_mode_choice, 'mcp_port': mcp_port_choice, 'mcp_webui': mcp_webui_choice, 'mcp_workspace_visibility': mcp_workspace_visibility_choice, 'mcp_visual_inspection': mcp_visual_inspection_choice,
            "compile": compile_choice, "profile": video_profile_choice,
            "video_profile": video_profile_choice, "image_profile": image_profile_choice, "audio_profile": audio_profile_choice,
            "video_preload_in_VRAM": video_preload_choice, "image_preload_in_VRAM": image_preload_choice, "audio_preload_in_VRAM": audio_preload_choice,
            "video_preload_mode": video_preload_mode_choice, "image_preload_mode": image_preload_mode_choice, "audio_preload_mode": audio_preload_mode_choice,
            processor_memory.MODE_KEY: postprocessor_preload_mode_choice, processor_memory.PRELOAD_KEY: int(postprocessor_preload_choice),
            "smart_memory_pinning": smart_memory_pinning_choice, "perc_reserved_mem_max": perc_reserved_mem_max_choice, "vram_allocator": vram_allocator_choice, RAM_ALLOCATOR_KEY: ram_allocator_choice,
            "llm_cpu_compute": llm_cpu_compute_choice,
            "vae_config": vae_config_choice,
            "mixed_precision": mixed_precision_choice, "metadata_type": metadata_choice,
            "transformer_quantization": quantization_choice, "transformer_dtype_policy": transformer_dtype_policy_choice,
            "boost": boost_choice, "int8_kernels": int8_kernels_choice, "fp8_kernels": fp8_kernels_choice, "kernel_precision": kernel_precision_choice, "attention_head_split": attention_head_split_choice, "clear_file_list": clear_file_list_choice,
            "multi_prompts_gen_type": prompt_parser.normalize_multi_prompts_mode(multi_prompts_gen_type_choice, default=prompt_parser.DEFAULT_MULTI_PROMPTS_MODE),
            "keep_intermediate_sliding_windows": keep_intermediate_sliding_windows_choice,
            "preload_model_policy": preload_model_policy_choice, "UI_theme": UI_theme_choice,
            "floating_generate_button": floating_generate_button_choice,
            "fit_canvas": fit_canvas_choice,
            LLM_CONFIG_KEY: llm_config_choice,
            "enhancer_mode": enhancer_mode_choice,
            "matanyone_version": matanyone_version_choice,
            WHISPER_MODEL_KEY: whisper_model_choice,
            DEEPY_ENABLED_KEY: normalize_deepy_enabled(deepy_enabled_choice),
            DEEPY_TYPE_KEY: normalize_deepy_type(deepy_type_choice),
            DEEPY_VOICE_LANGUAGE_KEY: normalize_deepy_voice_language(deepy_voice_language_choice),
            VOICE_MODE_KEY: voice_mode({VOICE_MODE_KEY: voice_mode_choice}),
            "depth_anything_v2_variant": depth_anything_v2_variant_choice,
            "notification_sound_enabled": notification_sound_enabled_choice,
            "notification_sound_volume": notification_sound_volume_choice,
            **notification_config_update,
            "max_frames_multiplier": max_frames_multiplier_choice, "display_stats": display_stats_choice,
            "keep_resolution_on_model_switch": keep_resolution_on_model_switch_choice,
            "enable_4k_resolutions": enable_4k_resolutions_choice,
            "max_reserved_loras": max_reserved_loras_choice,
            "video_output_codec": video_output_codec_choice, "hdr_video_crf": hdr_video_crf_choice,
            "rgba_video_output": rgba_video_output_choice,
            "image_output_codec": image_output_codec_choice,
            "audio_output_codec": audio_output_codec_choice,
            "audio_stand_alone_output_codec": audio_stand_alone_output_codec_choice,
            "model_hierarchy_type": model_hierarchy_type_choice,
            "checkpoints_paths": checkpoints_paths,
            "loras_root": loras_root_choice,
            "save_queue_if_crash": save_queue_if_crash_choice,
            "queue_color_scheme": queue_color_scheme_choice,
            gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY: process_queues_when_browser_unfocused_choice,
            "prevent_power_throttling": prevent_power_throttling_choice,
            COMPUTE_CACHE_KEY: compute_cache_choice, COMPUTE_CACHE_PATH_KEY: compute_cache_path_choice,
            "embed_source_images": embed_source_images_choice,
            "video_container": video_container_choice,
        })
        # Preserve saved values for controls hidden by the selected engine or Deepy mode.
        if not deepy_remote:
            new_server_config.update({
                "enhancer_enabled": enhancer_enabled_choice,
                "prompt_enhancer_temperature": prompt_enhancer_temperature_choice,
                "prompt_enhancer_top_p": prompt_enhancer_top_p_choice,
                "prompt_enhancer_randomize_seed": prompt_enhancer_randomize_seed_choice,
            })
        if qwen_local:
            new_server_config.update({
                "prompt_enhancer_quantization": enhancer_quantization_choice,
                PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY: enhancer_speculative_decoding_choice,
                DEEPY_KV_CACHE_QUANTIZATION_KEY: join_deepy_kv_cache_quantization(deepy_kv_cache_quantization_choice, deepy_kv_cache_exact_choice),
                DEEPY_REPETITION_PENALTY_KEY: normalize_deepy_repetition_penalty(deepy_repetition_penalty_choice),
            })
        if deepy_enabled_choice:
            new_server_config[DEEPY_ALLOW_READ_FILE_SYSTEM_KEY] = normalize_deepy_file_system_access(deepy_allow_read_file_system_choice)
            if deepy_type_choice == DEEPY_TYPE_ZERO:
                new_server_config[DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY] = normalize_deepy_custom_system_prompt(deepy_zero_custom_system_prompt_choice)
            if deepy_type_choice == DEEPY_TYPE_PRIME:
                new_server_config.update({
                    DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY: normalize_deepy_prime_guidance(deepy_prime_custom_system_prompt_choice),
                    DEEPY_PRIME_MCP_SERVERS_KEY: deepy_prime_mcp_servers_choice,
                    DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY: normalize_deepy_mcp_auto_discover_paths(deepy_mcp_auto_discover_paths_choice),
                    DEEPY_PYTHON_SANDBOX_KEY: normalize_deepy_python_sandbox(deepy_python_sandbox_choice),
                    PYTHON_APPS_KEY: normalize_python_apps({**normalize_python_apps(old_server_config.get(PYTHON_APPS_KEY)), "development": python_apps_choice}),
                    DEEPY_WEB_SEARCH_KEY: normalize_deepy_web_search(deepy_web_search_choice),
                    **civitai_config_update,
                })
            if not deepy_remote:
                new_server_config[DEEPY_VRAM_MODE_KEY] = normalize_deepy_vram_mode(deepy_vram_mode_choice)
            if qwen_local:
                new_server_config.update({
                    DEEPY_CONTEXT_TOKENS_KEY: normalize_deepy_context_tokens(deepy_context_tokens_choice),
                    DEEPY_COMPACTION_TYPE_KEY: deepy_compaction_type_choice,
                    DEEPY_COMPACTION_THINKING_KEY: deepy_compaction_thinking_choice,
                })
            if deepy_allow_read_file_system_choice != DEEPY_FILE_SYSTEM_ACCESS_DISABLED:
                new_server_config.update({
                    DEEPY_FILE_SYSTEM_PATHS_KEY: deepy_file_system_paths_choice,
                    DEEPY_READ_EVERYWHERE_KEY: normalize_deepy_read_everywhere(deepy_read_everywhere_choice),
                })
        audio_processor_api.apply_config_update(new_server_config, self.audio_processor_config_bindings, audio_processor_config_update)
        temporal_upsampler_api.apply_config_update(new_server_config, self.temporal_upsampler_config_bindings, temporal_upsampler_config_update)
        upsampler_api.apply_config_update(new_server_config, self.upsampler_config_bindings, upsampler_config_update)
        processor_memory.migrate_config(new_server_config)

        if self.args.lock_config:
            if "attention_mode" in old_server_config: new_server_config["attention_mode"] = old_server_config["attention_mode"]
            if "compile" in old_server_config: new_server_config["compile"] = old_server_config["compile"]

        for key in ("depth_anything_v3_process_res", "depth_anything_v3_chunk_size", "depth_anything_v3_chunk_overlap"):
            new_server_config.pop(key, None)

        if not show_all_model_filter_choice and new_server_config.get(model_dropdowns.MODEL_OUTPUT_FILTER_CONFIG_KEY) == model_dropdowns.MODEL_OUTPUT_FILTER_ALL:
            new_server_config[model_dropdowns.MODEL_OUTPUT_FILTER_CONFIG_KEY] = model_dropdowns.get_model_output_filter_for_model_type(self._get_dropdown_deps(), state["model_type"])

        updates = {key: value for key, value in new_server_config.items() if key not in old_server_config or value != old_server_config[key]}
        if gen_in_progress and processor_memory.config_values(old_server_config) != processor_memory.config_values(new_server_config):
            raise gr.Error("Wait for the current generation to finish before changing Postprocessor VRAM Usage.")
        int8_resolution = fp8_resolution = None
        if "int8_kernels" in updates or "fp8_kernels" in updates or "kernel_precision" in updates:
            if gen_in_progress:
                raise gr.Error("Wait for the current generation to finish before changing kernel options.")
            try:
                if new_server_config["kernel_precision"] not in ("strict", "fast"):
                    raise ValueError("Unknown non-INT8 kernel precision")
                if "int8_kernels" in updates:
                    int8_resolution = int8_backend.resolve_backend(new_server_config["int8_kernels"])
                if "fp8_kernels" in updates:
                    fp8_resolution = fp8_backend.resolve_backend(new_server_config["fp8_kernels"])
            except (ValueError, RuntimeError) as exc:
                raise gr.Error(str(exc)) from exc
        removed = old_server_config.keys() - new_server_config.keys()
        update_config(self.server_config, self.server_config_filename, updates, remove=removed)
        if VOICE_MODE_KEY in updates:
            self.state.value.service.publish("voice_config", {"mode": voice_mode(self.server_config)})
        if 'clear_file_list' in updates:
            self.state.value.service.host_changed()
        try:
            notifications.cleanup_config_update(old_server_config, new_server_config)
        except notifications.SecureStorageError as exc:
            gr.Warning(f"Configuration was saved, but the previous credential-store entry could not be removed: {exc}")
        
        changes = list(updates)

        no_reload_keys = [
            model_dropdowns.SHOW_ALL_MODEL_FILTER_CONFIG_KEY, model_dropdowns.MODEL_OUTPUT_FILTER_CONFIG_KEY,
            processor_memory.MODE_KEY, processor_memory.PRELOAD_KEY,
            'workspace_storage_mode', 'workspaces_dir', 'mcp_workspace_mode', 'mcp_port', 'mcp_webui', 'mcp_workspace_visibility', 'mcp_visual_inspection',
            VOICE_MODE_KEY,
            DEEPY_VOICE_LANGUAGE_KEY,
            "attention_mode", "vae_config", "boost", "int8_kernels", "fp8_kernels", "kernel_precision", "attention_head_split", "save_path", "image_save_path", "audio_save_path", "mesh_save_path",
            "metadata_type", "clear_file_list", "multi_prompts_gen_type", "keep_intermediate_sliding_windows", "fit_canvas", "depth_anything_v2_variant",
            "notification_sound_enabled", "notification_sound_volume", *notifications.CONFIG_KEYS, "audio_processors", "temporal_upsamplers", "spatial_upsamplers", "matanyone_version",
            "prompt_enhancer_temperature", "prompt_enhancer_top_p", "prompt_enhancer_randomize_seed", "prompt_enhancer_quantization", PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, "enhancer_mode",
            DEEPY_ENABLED_KEY, DEEPY_TYPE_KEY, DEEPY_VRAM_MODE_KEY, DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_READ_EVERYWHERE_KEY, DEEPY_CONTEXT_TOKENS_KEY, DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_THINKING_KEY, DEEPY_REPETITION_PENALTY_KEY, DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_MCP_SERVERS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, PYTHON_APPS_KEY,
            LLM_CONFIG_KEY,
            "max_frames_multiplier", "display_stats", "keep_resolution_on_model_switch", "enable_4k_resolutions", "max_reserved_loras", "video_output_codec", "hdr_video_crf", "video_container",
            "embed_source_images", "image_output_codec", "audio_output_codec", "audio_stand_alone_output_codec", "checkpoints_paths", "loras_root", "save_queue_if_crash",
            "model_hierarchy_type", "UI_theme", "floating_generate_button", "queue_color_scheme", gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, "prevent_power_throttling",
            COMPUTE_CACHE_KEY, COMPUTE_CACHE_PATH_KEY,
        ]

        needs_reload = not all(change in no_reload_keys for change in changes)

        self.set_global("three_levels_hierarchy", new_server_config["model_hierarchy_type"] == 1)
        self.set_global("attention_mode", new_server_config["attention_mode"])
        self.set_global("default_profile", new_server_config["profile"])
        self.set_global("default_profile_video", new_server_config["video_profile"])
        self.set_global("default_profile_image", new_server_config["image_profile"])
        self.set_global("default_profile_audio", new_server_config["audio_profile"])
        self.set_global("compile", new_server_config["compile"])
        self.set_global("text_encoder_quantization", new_server_config["text_encoder_quantization"])
        self.set_global("vae_config", new_server_config["vae_config"])
        self.set_global("boost", new_server_config["boost"])
        self.set_global("int8_kernels", new_server_config["int8_kernels"])
        self.set_global("fp8_kernels", new_server_config["fp8_kernels"])
        self.set_global("save_path", new_server_config["save_path"])
        self.set_global("image_save_path", new_server_config["image_save_path"])
        self.set_global("audio_save_path", new_server_config["audio_save_path"])
        self.set_global("mesh_save_path", new_server_config["mesh_save_path"])
        self.set_global("preload_model_policy", new_server_config["preload_model_policy"])
        self.set_global("transformer_quantization", new_server_config["transformer_quantization"])
        self.set_global("transformer_dtype_policy", new_server_config["transformer_dtype_policy"])
        self.set_global("transformer_types", new_server_config["transformer_types"])
        set_deepy_runtime_config(self.server_config, self.server_config_filename)
        if needs_reload: self.set_global("reload_needed", True)

        enhancer_runtime_changed = LLM_CONFIG_KEY in changes or "enhancer_enabled" in changes or "prompt_enhancer_quantization" in changes or DEEPY_KV_CACHE_QUANTIZATION_KEY in changes or DEEPY_ENABLED_KEY in changes or DEEPY_VRAM_MODE_KEY in changes
        deepy_type_changed = DEEPY_TYPE_KEY in changes
        deepy_prime_mcp_servers_changed = DEEPY_PRIME_MCP_SERVERS_KEY in changes or DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY in changes or PYTHON_APPS_KEY in changes  # Prime's tools change: the app actions of wangp_python
        deepy_file_access_changed = any(key in changes for key in (DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_READ_EVERYWHERE_KEY, "save_path", "image_save_path", "audio_save_path", "mesh_save_path"))
        speculative_decoding_changed = PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY in changes
        enhancer_profile_changed = "profile" in changes or "video_profile" in changes or "llm_cpu_compute" in changes
        chat_event_update = gr.update()
        if enhancer_runtime_changed:
            session = get_or_create_assistant_session(state)
            self.release_deepy_vram(state, clear_session_state=True, discard_runtime_snapshot=True)
            session.force_loading_status_once = True
            self.reset_prompt_enhancer()
            self.reset_prompt_enhancer_if_requested()
            chat_event_update = assistant_chat.build_reset_event(session)
        elif speculative_decoding_changed:
            get_or_create_assistant_session(state).force_loading_status_once = True
            self.release_deepy_vram(state, clear_session_state=False, discard_runtime_snapshot=True)
            self.reset_prompt_enhancer()
            self.reset_prompt_enhancer_if_requested()
        elif enhancer_profile_changed:
            self.release_deepy_vram(state, clear_session_state=False, discard_runtime_snapshot=True)
            self.reset_prompt_enhancer()
            self.reset_prompt_enhancer_if_requested()
        elif deepy_type_changed or deepy_prime_mcp_servers_changed or deepy_file_access_changed:
            self.release_deepy_vram(state, clear_session_state=True, discard_runtime_snapshot=True)
            chat_event_update = assistant_chat.build_reset_event(get_or_create_assistant_session(state))
        audio_processor_api.release_changed_config_processors(old_server_config, new_server_config, changes)
        temporal_upsampler_api.release_changed_config_temporal_upsamplers(old_server_config, new_server_config, changes)
        upsampler_api.release_changed_config_upsamplers(old_server_config, new_server_config, changes)
        if "kernel_precision" in changes:
            kernel_policy.configure(new_server_config["kernel_precision"])
        if "attention_head_split" in changes:
            attention_kit.configure(new_server_config["attention_head_split"])
        if "int8_kernels" in changes:
            self.apply_int8_kernel_setting(new_server_config["int8_kernels"], True, resolved=int8_resolution)
        if "fp8_kernels" in changes:
            self.apply_fp8_kernel_setting(new_server_config["fp8_kernels"], resolved=fp8_resolution)

        model_type = state["model_type"]
        model_dropdowns.reconcile_model_output_filter_for_model_type(self._get_dropdown_deps(), state, model_type)
        model_family_update, model_base_type_update, model_choice_update = self.generate_dropdown_model_list(model_type, state, select_visible_model=True)
        description_update, header_update = self.generate_header(model_type, compile=new_server_config["compile"], attention_mode=new_server_config["attention_mode"], overrides=overrides)

        if gen_in_progress:
            msg = "<div style='color:green; text-align:center;'>The new configuration has been succesfully applied. Some of the Settings will be only effective when you will start another Generation</div>"
        else:
            msg = "<div style='color:green; text-align:center;'>The new configuration has been succesfully applied.</div>"

        deepy_visible = deepy_available(new_server_config)
        launcher_update = gr.update(value=assistant_chat.render_launcher_html() if deepy_visible else "", visible=deepy_visible)
        panel_update = gr.update(visible=deepy_visible)

        return Saved((
            msg,
            description_update,
            header_update,
            model_family_update,
            model_base_type_update,
            model_choice_update,
            self.get_unique_id(),
            launcher_update,
            panel_update,
            chat_event_update,
        ))
