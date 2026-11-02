# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
from pathlib import Path
import torch
from shared.utils import files_locator as fl
from .help import INFOS, PROMPT_INFOS, DEEPY_INFOS, DEEPY_PROMPT_INFOS, SPECIALITIES, DEFAULT_PROMPT

ARCHITECTURE = "nemotron_transcribe"
FAMILY = "transcription"
REPO_ID = "DeepBeepMeep/Nemotron"
VIDEO_SOURCE = "K"  # audio_prompt_type letter of the Control Video's soundtrack: WanGP extracts its first audio track into audio_guide
TASKS = ("transcribe", "translate")  # model_mode values 0 and 1
CUSTOM_SETTINGS = [
    {"id": "speakers", "name": "Speakers", "label": "Speaker Attribution", "type": "dropdown", "choices": [("Label Speakers (Who Said What)", 1), ("Off", 0)], "default": 1},
    {"id": "speaker_names", "name": "Speaker Names", "label": "Names for Speaker 1, 2... (Optional)", "type": "text", "default": ""},
    {"id": "word_timestamps", "name": "Timestamps", "label": "Timestamps and Subtitles", "type": "dropdown", "choices": [("Per Turn", 1), ("Per Word", 2), ("Off", 0)], "default": 1},
    {"id": "style", "name": "Style", "label": "Transcript Style", "type": "dropdown", "choices": [("Verbatim (Keep Fillers and Repetitions)", "verbatim"), ("Readable (Light Clean-Up)", "readable")], "default": "verbatim",
     "model_mode": [0]},  # a translation is not styled
    {"id": "target_language", "name": "Translate To", "label": "Translate To (Language Name)", "type": "text", "default": "English", "model_mode": [1]},
    {"id": "output", "name": "Output", "label": "Output", "type": "dropdown", "choices": [("Audio and Text Files", "audio"), ("Video with Subtitles", "video")], "default": "audio",
     "audio_prompt_type": VIDEO_SOURCE},  # only for a video's soundtrack
]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return FAMILY

    @staticmethod
    def query_family_infos():
        return {FAMILY: (2205, "Transcription")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return ARCHITECTURE

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": FAMILY, "audio_only": True, "image_outputs": False, "sliding_window": False,
            "device_explicit": True, "dtype": "bf16", "auto_quantize": False, "lm_decoder": True, "no_model_swap": True, "profiles_dir": [ARCHITECTURE],
            "no_negative_prompt": True, "inference_steps": False, "guidance_max_phases": 0, "temperature": False, "compile": False, "no_lora": True,
            "image_prompt_types_allowed": "", "any_audio_prompt": True,
            "guide_custom_choices": {"choices": [("Audio File", ""), ("Video File (Its Soundtrack)", "V")], "letters_filter": "V", "default": "", "label": "Recording",
                                     "audio_letters": {"": "A", "V": VIDEO_SOURCE}, "audio_letters_filter": "A" + VIDEO_SOURCE},  # the video is the Control Video
            "audio_guide_label": "Recording to Transcribe", "video_guide_label": "Video to Transcribe (Its First Audio Track)",
            "skip_video_guide_preprocess": True, "keep_frames_video_guide_not_supported": True,  # only the soundtrack is read: no frames decoded
            "model_modes": {"label": "Task", "choices": [("Transcribe (Language Spoken)", 0), ("Translate", 1)], "default": 0},
            "custom_settings": [dict(setting) for setting in CUSTOM_SETTINGS],
            "prompt_optional": True, "prompt_class": "Context Terms",
            "prompt_description": "Context Terms (optional): names, places and jargon spoken in the recording, separated by commas or lines",
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "extra_outputs": ["text", "video"],  # the transcript and subtitles beside the audio; the source video with subtitles (output "video")
            "capability_overrides": {"text_to_audio": False, "audio_to_audio": False, "audio_to_text": True, "video_to_text": True},
            "specialities": [dict(entry) for entry in SPECIALITIES],
            "infos": INFOS, "prompt_infos": PROMPT_INFOS, "deepy_infos": DEEPY_INFOS, "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        from .pipeline import ASSET_FOLDER, ASSET_FILES, DRAFT_FILE
        return [{"repoId": REPO_ID, "sourceFolderList": [ASSET_FOLDER], "fileList": [ASSET_FILES + [DRAFT_FILE]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, lm_features=None, profile=0, **kwargs):
        from .pipeline import ASSET_FOLDER, ASSET_FILES, DRAFT_FILE, NemotronTranscribePipeline
        from .wangp_model import TranscriptionModel
        lm_file, = model_filename
        assets = {name: fl.locate_file(f"{ASSET_FOLDER}/{name}") for name in ASSET_FILES + [DRAFT_FILE]}
        experts_file = fl.locate_file(os.path.basename(model_def["preload_URLs"][0]))  # routed experts, in their own format (INT8 ConvRot or INT4)
        perception_file = assets["NemotronTranscribe_perception_bf16.safetensors"]
        pipeline = NemotronTranscribePipeline(lm_file, experts_file, perception_file, os.path.dirname(assets["tokenizer.json"]), lm_features, draft_file=assets[DRAFT_FILE])
        if save_quantized:
            from wgp import save_quantized_model
            # the decoder tower only: the head, embeddings, routers and MTP tensors keep their prepared precision
            save_quantized_model(pipeline.transformer, model_type, lm_file, torch.bfloat16, assets["config.json"], submodel_no=1, convrot_prefixes=("model.layers.",))
        # co-tenants: the word alignment runs the perception model again after the decoding, which would otherwise unload the language model
        offload_def = {"pipe": {"transformer": pipeline.transformer, "perception": pipeline.perception}, "cudaGraphs": ["transformer"] if lm_features.graphs else False,
                       "coTenantsMap": {"transformer": ["perception"], "perception": ["transformer"]}}
        if int(profile) in (0, 2, 4, 5):
            # The language model runs at every output token: streaming its dense layers would cost a transfer of their weights per token. They
            # stay whole in VRAM; the routed experts take what a VRAM Preload leaves beyond them, or the free VRAM with a dynamic preload.
            offload_def["budgets"] = {"transformer": 0}
        return TranscriptionModel(pipeline), offload_def

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(prompt=DEFAULT_PROMPT, audio_prompt_type="A", video_prompt_type="", model_mode=0, custom_settings={setting["id"]: setting["default"] for setting in CUSTOM_SETTINGS},
            negative_prompt="", prompt_enhancer="", guidance_scale=1.0, num_inference_steps=1, video_length=0, duration_seconds=0, repeat_generation=1, multi_prompts_gen_type="FG")

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pass  # New architecture; no older saved settings to migrate.

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        video = inputs["video_guide"] if VIDEO_SOURCE in inputs["audio_prompt_type"] else None
        if video is not None:  # a missing video is reported by WanGP's input checks
            from shared.utils.audio_video import extract_audio_tracks
            if extract_audio_tracks(video, query_only=True) == 0:
                return f"{Path(video).name} has no audio track to transcribe."
            settings = inputs["custom_settings"]
            if settings.get("output") == "video" and not settings.get("word_timestamps", 1):
                return "Video with Subtitles needs Timestamps and Subtitles: subtitles are timed with them."
        if inputs["model_mode"] not in (0, 1):
            return "Choose Transcribe or Translate."
        return None
