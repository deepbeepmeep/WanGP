# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""WanGP Showrunner: productions directed by Deepy, defined and reviewed in a dedicated page (a tab, or full screen at /showrunner/)."""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Showrunner"
PlugIn_Id = "Showrunner"
WEB_PATH = "/showrunner"


class _Progress:
    def __init__(self, on_progress):
        self._on_progress = on_progress

    def on_progress(self, update):
        if self._on_progress is not None:
            self._on_progress(update)


ADVISED_CONTEXT = 64000  # tokens: a production's conversation is long, and below this Deepy keeps condensing it


def advice_for_deepy(context_tokens: int, kv_cache: str, lower_kv_cache: bool, room_for_context: bool = True) -> str:
    """One notice for what a local Deepy's settings cost a production: a short context, a KV cache that could leave room for a longer one. Empty when both are fine. room_for_context: this GPU holds
    the model with the advised context; on a smaller one, following the advice would push the model out of VRAM."""
    short, large = room_for_context and context_tokens < ADVISED_CONTEXT, lower_kv_cache and kv_cache in ("", "int8")
    context, cache = f"Context Window Tokens to at least 64K (it is {context_tokens // 1000}K)", f"KV Cache Quantization to Q4 or lower (it is {'INT8' if kv_cache else 'BF16'})"
    where = "In Configuration > Prompt Enhancer / Deepy."
    if short and large:
        return f"For long productions, set Deepy's {context} and its {cache}, which fits that context in less VRAM. {where}"
    if short:
        return f"For long productions, set Deepy's {context}: with less, it keeps condensing the conversation and loses details. {where}"
    return f"To fit more context in the same VRAM, set Deepy's {cache}. {where}" if large else ""


class WanGPHost:
    """What Showrunner needs from WanGP: Deepy jobs, the production workspaces, the web UI queue and the output codec settings."""

    def __init__(self, plugin):
        self.plugin = plugin

    def _client(self):
        return self.plugin.deepy

    def director_unavailable(self) -> str:
        try:
            client = self._client()
        except RuntimeError as error:
            return str(error)
        if client.unavailable_reason:
            return client.unavailable_reason
        if client.deepy_type != "prime":
            return "Showrunner needs Deepy Prime (local or with a remote agent): Deepy Zero cannot plan long productions. Choose Prime in Configuration > Prompt Enhancer / Deepy, or make the shots yourself: + Shot, then Render now."
        return ""

    def director_info(self) -> dict:
        reason = self.director_unavailable()
        try:
            kind = self._client().deepy_type
        except RuntimeError:
            kind = None
        return {"available": not reason, "reason": reason, "type": kind, "advice": "" if reason else self.deepy_advice()}

    def deepy_advice(self) -> str:
        from shared.deepy.config import DEEPY_CONTEXT_TOKENS_DEFAULT, DEEPY_CONTEXT_TOKENS_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT, DEEPY_KV_CACHE_QUANTIZATION_KEY, deepy_lower_kv_cache_supported, get_deepy_runtime_config, normalize_deepy_context_tokens, resolve_deepy_kv_cache_quantization
        from shared.deepy.onboarding import DEEPY_FULL_CONTEXT_MIN_VRAM_GB, deepy_performance_warning, reported_vram
        from shared.remote_llm.config import is_remote_engine, resolve_role_engine
        config = get_deepy_runtime_config()
        if is_remote_engine(resolve_role_engine(config, "deepy")):  # a remote agent has its own context
            return ""
        warning, vram = deepy_performance_warning(config), reported_vram()
        advice = advice_for_deepy(normalize_deepy_context_tokens(config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT)), resolve_deepy_kv_cache_quantization(config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT))[0], deepy_lower_kv_cache_supported(), not warning and (vram == 0 or vram >= DEEPY_FULL_CONTEXT_MIN_VRAM_GB))
        return " ".join(part for part in (warning, advice) if part)

    def production_workspace(self, title: str, folder: Path) -> str:
        return self._client().ensure_workspace(f"Showrunner - {title}"[:120], folder)

    def gallery_media(self, workspace: str, media_id: str) -> str | None:
        """The file of a Gallery media id in a production's workspace, None when its Gallery has no such media."""
        return next((item["path"] for item in self._client().gallery(workspace) if item["id"] == media_id), None)

    def gpu_turn(self, cancelled):
        """The turn of renders made without the showrunner: after the showrunners at work in other productions, in order with them, never beside Deepy's model."""
        return self._client().turn(cancelled)

    def run_director(self, prompt: str, *, session_dir: Path, workspace: str, title: str, instructions: str = ""):
        return self._client().run(prompt, session_dir=session_dir, workspace=workspace, title=title[:120], instructions=instructions)

    def release_gpu(self) -> bool:
        """Free the VRAM after the director paused: WanGP's loaded models and Deepy's model leave the GPU (Deepy's conversation stays), as the
        Unload Models tool does. Nothing is released while a generation runs or Deepy works; returns whether it was."""
        import torch
        from shared.utils.model_unload import model_unload_guard
        plugin = self.plugin
        free = lambda: torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        before = free()
        with model_unload_guard():
            if plugin.is_generation_in_progress() or not self._client().release_vram():
                print("[Showrunner] The director paused; the GPU stays in use (a generation runs or Deepy works).", flush=True)
                return False
            plugin.release_extension_offloadobjs()
            plugin.release_model()
        print(f"[Showrunner] The director paused: {(free() - before) / 2 ** 30:.1f} GB of VRAM freed.", flush=True)
        return True

    def lip_reader(self):
        """How a shot's lips follow the voice: a function for the shots of one review, whose model goes with it."""
        from shared.deepy import lip_sync
        models = lip_sync.load_models()
        return lambda video, start_s, frames, fps, voice, voice_at: lip_sync.measure(video, start_s, frames, fps, voice, voice_at, models)

    def speaker_turns(self, ctx, path) -> list[tuple[str, str]]:
        """Who is heard in a version, as (speaker number, words) turns, when Nemotron is enabled in the Deepy configuration; [] otherwise."""
        from shared.deepy.config import advanced_transcription_model
        import re
        if not advanced_transcription_model():
            return []
        result = ctx.call("wangp_toolbox", {"action": "advanced_transcription", "arguments": {"media_id": ctx.virtualize(str(path)), "speakers": True, "timestamps": "turn"}})
        return re.findall(r"Speaker (\d+): (.*)", str((result.get("result") or result).get("text") or ""))

    def voice_pitch(self, path) -> float | None:
        """The median pitch of a voice alone (a voice sample, a spoken line, a song's vocal stem)."""
        from shared.deepy.audio_inspection import voice_pitch
        return voice_pitch(str(path))

    def song_words(self, path, language: str | None = None) -> list[tuple[str, float, float]]:
        """The words heard in a song with their times. Each stretch is read on its own: read after the words before it, a transcript of a song falls into repeating its chorus."""
        from shared.deepy import transcription
        payload = transcription.transcribe_media(str(path), timestamp_type="word", language=language, condition_on_previous_text=False)
        return [(word["word"], word["start"], word["end"]) for segment in payload["segments"] for word in segment.get("words", []) if word["start"] is not None and word["end"] is not None]

    def submit_generation(self, settings: dict, workspace: str, on_progress=None):
        """settings may name a Deepy template ("task/name"), as the shots' do: they start from its settings, as with wangp_generate (whose
        expansion lives inside the MCP server): the template's example prompts dropped, the shot's keys over it, custom_settings merged."""
        session = self._client().web_session(workspace)
        if settings.get("template"):
            from shared.mcp.server import _deepy_template_settings
            task, _, name = settings["template"].partition("/")
            base = {key: value for key, value in _deepy_template_settings(session, task, name.strip() or "default")["settings"].items() if key not in ("prompt", "alt_prompt")}
            merged = {**base, **{key: value for key, value in settings.items() if key != "template"}}
            if isinstance(base.get("custom_settings"), dict) and isinstance(settings.get("custom_settings"), dict):
                merged["custom_settings"] = {**base["custom_settings"], **settings["custom_settings"]}
            settings = merged
        return session.submit_task(settings, callbacks=_Progress(on_progress))

    def template_catalog(self, keep=()) -> list[dict]:
        """Deepy's templates of the categories a production uses, as Deepy's settings list them (the built-in ones and the custom one selected
        there), plus the custom templates in `keep` ("category/name") that Showrunner still uses after Deepy's settings moved on; with what the
        video ones take: reference images ("I" in video_prompt_type) and driving audio ("A" in audio_prompt_type)."""
        from shared import resolutions
        from shared.deepy import tool_settings
        from shared.deepy.ui_settings import get_persisted_assistant_tool_ui_settings
        from .templates import EDIT_TASKS, ROLES, VIDEO_TASKS
        settings = get_persisted_assistant_tool_ui_settings(self.plugin.server_config)
        catalog = []
        for task in dict.fromkeys(role["task"] for role in ROLES):  # two roles draw on the image editor's templates
            default = settings[tool_settings.TOOL_UI_KEYS[task]]
            kept = [name for category, _, name in (str(item or "").partition("/") for item in keep) if category == task]
            choices = {variant: label for current in (default, *kept) for label, variant in tool_settings.list_tool_variant_choices(task, current_variant=current)}
            for variant, label in choices.items():
                preset = tool_settings.clone_tool_preset(task, variant)  # the model type of every template: speech models differ in how they take cues
                video = preset if task in (*VIDEO_TASKS, *EDIT_TASKS) else {}
                model_def = (self.plugin.get_model_def(preset["model_type"]) or {}) if video or task == "edit_image" else {}  # an image editor may declare its trained swap sentences
                if task in EDIT_TASKS and not model_def.get("video_edit"):  # a remake's shots take painted frames and pins: only a model made for that (Viggle Remake) is offered
                    continue
                sources = model_def.get("audio_prompt_type_sources") or {}
                refs = str((preset if task in (*VIDEO_TASKS, *EDIT_TASKS, "edit_image") else {}).get("video_prompt_type") or "")  # image edits take the cast's pictures too
                catalog.append({"task": task, "category": tool_settings.TOOL_DISPLAY_NAMES[task], "name": variant, "label": label, "default": variant == default,
                                "model_type": preset.get("model_type"), "refs": "I" in refs, "audio": "A" in str(video.get("audio_prompt_type") or ""),
                                "video_edit": model_def.get("video_edit"), "needs_mask": task in EDIT_TASKS and "A" in refs,  # what a remake must prepare for it
                                "pixels": resolutions.resolution_budget(model_def)["default"] if task in EDIT_TASKS else None,
                                "soundtrack": "AS" in (sources.get("selection") or []),  # its model can follow an audio_guide kept as the soundtrack
                                "spoken_line": model_def.get("spoken_line_prompt"),  # how its model takes a line to voice, written in its prompt
                                "swap_prompt": model_def.get("character_swap_prompt"),  # the sentences its model was trained on to replace people, filled by the frame painter
                                "sized_by_ref": "K" in refs,  # its output takes the first reference's size, not the production's format
                                "shortest_s": math.floor(model_def.get("frames_minimum", 0) / model_def.get("fps", 16) * 1000) / 1000,  # the shortest take its model makes
                                "take_frames": [model_def.get("frames_minimum", 0), model_def.get("frames_steps", 1), model_def.get("fps", 16)],  # its first length and the steps after it
                                "longest_frames": video.get("sliding_window_size") or (model_def.get("sliding_window_defaults") or {}).get("window_default"),  # the longest take of one window: past it, WanGP generates a second one
                                "sliding_window": bool(model_def.get("sliding_window")),  # its model continues a take past one window
                                "window_max": (model_def.get("sliding_window_defaults") or {}).get("window_max")})  # the longest one window can be
        for task in EDIT_TASKS:  # Deepy's settings selected another editing model: a remake uses the first one made for it
            entries = [entry for entry in catalog if entry["task"] == task]
            if entries and not any(entry["default"] for entry in entries):
                entries[0]["default"] = True
        return catalog

    def default_templates(self) -> dict:
        """The templates new productions start with, by kind of shot or media (the brief's picks)."""
        return self.plugin.config().get("templates") or {}

    def save_default_templates(self, picks: dict) -> None:
        self.plugin.save_config(templates=picks)

    def prompt_rules(self) -> dict:
        """The user's rules for writing prompts with a template ("category/name": text), for every production."""
        return self.plugin.config().get("prompt_rules") or {}

    def save_prompt_rules(self, rules: dict) -> None:
        self.plugin.save_config(prompt_rules=rules)

    def frame_size(self, entry: dict, resolution: str) -> str:
        """The size a template's model makes of a resolution: fitted to its grid, as generation does."""
        from shared import resolutions
        return resolutions.align_resolution_value(resolution, (self.plugin.get_model_def(entry["model_type"]) or {}).get("vae_block_size", 16))

    def settings(self) -> dict:
        """Showrunner's own settings (its Settings screen), for every production."""
        config = self.plugin.config()
        return {"notify_turns": bool(config.get("notify_turns")), "cast_loras": bool(config.get("cast_loras")), "show_all_errors": bool(config.get("show_all_errors")), "show_deepy_tab": bool(config.get("show_deepy_tab")), "any_shot_length": bool(config.get("any_shot_length"))}

    def save_settings(self, **fields) -> None:
        self.plugin.save_config(**fields)

    def notify(self, title: str, message: str) -> None:
        """WanGP's notification sound on this computer, whatever it does for generations, at its volume, and its notification destinations when any is set."""
        from shared.notifications import send_notification_async
        from shared.utils.notification_sound import play_notification_async
        config = self.plugin.server_config
        play_notification_async(config.get("notification_sound_volume", 50))
        send_notification_async(config, title, message)

    def lora_folder(self, model_type: str) -> str | None:
        """The LoRA folder of a model (None: it takes no LoRAs): a LoRA fits the models that share its folder."""
        return None if (self.plugin.get_model_def(model_type) or {}).get("no_lora") else self._client().web_session().get_lora_dir(model_type)

    def list_loras(self, model_type: str) -> list[str]:
        return list(self._client().web_session().list_loras(model_type)["loras"])

    def template_loras(self, template: str) -> tuple[list[str], str]:
        """The LoRAs a Deepy template generates with, and their multipliers."""
        from shared.mcp.server import _deepy_template_settings
        task, _, name = template.partition("/")
        settings = _deepy_template_settings(self._client().web_session(), task, name.strip() or "default")["settings"]
        return list(settings.get("activated_loras") or []), str(settings.get("loras_multipliers") or "")

    def encode_settings(self):
        from shared.utils.audio_video import get_mp4_audio_codec_settings, get_video_audio_extra_args
        from shared.utils.video_codecs import get_video_container_extension, get_video_encode_args
        config = self.plugin.server_config
        container = str(config.get("video_container", "mp4") or "mp4")
        audio_key = str(config.get("audio_output_codec", "aac_128") or "aac_128")
        audio = get_mp4_audio_codec_settings(audio_key)
        audio_args = ["-c:a", audio["codec"], *get_video_audio_extra_args(audio_key), *(["-b:a", audio["bitrate"]] if audio.get("bitrate") else [])]
        return get_video_encode_args(str(config.get("video_output_codec", "libx264_8") or "libx264_8"), container), audio_args, get_video_container_extension(container)


class ShowrunnerPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "0.2.1"
        self.description = "Productions directed by Deepy: music videos, films, ads and stories, defined and reviewed shot by shot."
        self._showrunner = None
        self._lock = threading.Lock()

    def setup_ui(self):
        self.request_global("server_config")
        self.request_global("args")
        self.request_global("get_model_def")  # which templates can follow a soundtrack
        for name in ("is_generation_in_progress", "release_model", "release_extension_offloadobjs"):  # pausing the director frees the GPU
            self.request_global(name)
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_ui)
        self.register_web_app(WEB_PATH, self.create_web_app)
        from .toolbox import ShowrunnerToolbox
        ShowrunnerToolbox(_LazyShowrunner(self)).register(self)

    def settings_folder(self) -> Path:
        """The plugin's configuration lives in WanGP's settings folder."""
        from .store import settings_folder
        return settings_folder(getattr(self.args, "settings", None))

    def config(self) -> dict:
        path = self.settings_folder() / "config.json"
        config = {"productions_root": str(self.settings_folder() / "productions")}
        if path.is_file():
            config.update(json.loads(path.read_text(encoding="utf-8")))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, indent=1), encoding="utf-8")
        return config

    def save_config(self, **fields) -> None:
        with self._lock:
            (self.settings_folder() / "config.json").write_text(json.dumps({**self.config(), **fields}, indent=1), encoding="utf-8")

    @property
    def showrunner(self):
        with self._lock:
            if self._showrunner is None:
                from .service import Showrunner
                from .store import ProductionStore
                root = Path(os.path.expanduser(self.config()["productions_root"]))
                self._showrunner = Showrunner(ProductionStore(root), WanGPHost(self))
            return self._showrunner

    def create_web_app(self):
        from .api import create_app
        return create_app(self.showrunner)

    def create_ui(self):
        # data-src: WanGP's navigation loads the app when its tab is first shown and tells it when the tab is hidden.
        gr.HTML(f"""<div class="wangp-addon-web-view">
  <iframe data-src=".{WEB_PATH}/?embedded=1" title="Showrunner" allow="fullscreen; clipboard-write" style="width:100%;height:calc(100vh - 150px);min-height:640px;border:0;background:#0b0d12"></iframe>
</div>
""", padding=False)


class _LazyShowrunner:
    """The toolbox is registered at setup, before WanGP's globals exist; it reaches the Showrunner on its first call."""

    def __init__(self, plugin):
        self._plugin = plugin

    def __getattr__(self, name):
        return getattr(self._plugin.showrunner, name)
