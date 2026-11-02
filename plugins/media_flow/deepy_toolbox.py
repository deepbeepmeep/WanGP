# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Media Flow for agents (wangp_plugins, MCP and Deepy Python): its templates by task, runs on one file, a list, a folder or wildcards, and the
status of the process running from the tab or an agent. A run drives the tab's own runner, so it follows the same rules and resumes the same way."""
from __future__ import annotations

import difflib
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal

from postprocessing import spatial_upsamplers as spatial_upsampler_api
from postprocessing import temporal_upsamplers as temporal_upsampler_api
from shared.gradio.local_file_picker import IMAGE_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS

from . import constants as ui_constants
from . import frame_planning as frames
from . import process_catalog as catalog
from .form_controller import create_form_controller
from .process_library import ProcessLibrary
from .process_runner import ProcessRunner, plugin_active_job, remember_status, request_stop, reset_live_chunk_status
from .status_ui import render_chunk_status_html

TASKS = {
    "outpaint": "Extend the frame to a new aspect ratio, inventing the missing borders (target: the ratio, such as 16:9).",
    "refocus": "Remove focus and motion blur.",
    "uncompress": "Remove compression artifacts (blocking, banding, smearing) of heavily encoded media.",
    "ungrade": "Remove a stylized color grade and restore neutral colors.",
    "hdr": "Convert SDR media to HDR.",
    "detail": "Add fine detail and texture (sharper surfaces and edges).",
    "upscale": "Increase the resolution (target: the multiplier, such as x2).",
    "interpolate": "Add frames between frames for smoother motion and a higher frame rate (target: the multiplier, such as x2).",
    "custom": "User settings linked in the Media Flow tab (Add User Settings): the process their settings define.",
}
SUMMARY = ("Restore (deblur, decompress, ungrade, SDR to HDR, add detail), outpaint, upscale or interpolate existing videos and images with ready-made templates, "
           "one file or whole folders: long videos are processed in chunks with low RAM and the original audio is kept; stopped runs resume.")
GUIDE = ("Media Flow applies a template (a restoration, outpaint, upscale or interpolation process) to existing media. Choose it with templates(media), which "
         "groups the available templates by task with the recommended one first; templates(media, name=...) gives the settings that apply to one template. "
         "run(process, source) then processes one file, a list, a folder or wildcards (several files form a resumable batch). process is a template name or a "
         "task, which picks that task's recommended template. Videos of any length are processed in chunks, keeping the original audio; a stopped or "
         "interrupted run continues where it stopped when the same run is called again. Outputs go next to each source when that folder is writable, else to "
         "WanGP's outputs folder (media_flow); output names a file (one source) or a folder. One Media Flow process runs at a time, shared with its tab.")
ROUTING = ("Deblurring, removing compression artifacts or a color grade, SDR to HDR, adding detail, outpainting, upscaling or interpolating existing videos or images "
           "that are long (over about a minute), several files, a folder or wildcards, or a job to resume")
_WILDCARD = re.compile(r"[*?\[]")
_EXTENSIONS = {"video": VIDEO_FILE_EXTENSIONS, "image": IMAGE_FILE_EXTENSIONS}


def _media_of(path: str) -> str | None:
    suffix = Path(str(path)).suffix.lower()
    return next((media for media, extensions in _EXTENSIONS.items() if suffix in extensions), None)


def _timestamp(value) -> str:
    if isinstance(value, (int, float)):
        minutes, seconds = divmod(float(value), 60.0)
        return f"{int(minutes):02d}:{seconds:05.2f}" if seconds % 1 else f"{int(minutes):02d}:{int(seconds):02d}"
    return str(value).strip()


def _prompt_text(prompt) -> str:
    """A timed prompt list [{"at": "01:30", "text": "..."}] as the tab's timed blocks."""
    if not isinstance(prompt, list):
        return str(prompt)
    return "\n\n".join(f"{_timestamp(item['at'])}\n{str(item['text']).strip()}" for item in prompt)


def _scale(label: str) -> float | None:
    try:
        return float(str(label).strip().lower().lstrip("x"))
    except ValueError:
        return None


class MediaFlowAgent:
    def __init__(self, plugin) -> None:
        self.plugin = plugin

    def register(self) -> None:
        plugin = self.plugin
        plugin.register_deepy_toolbox(SUMMARY, guide=GUIDE)
        plugin.register_deepy_action(self.templates, uses_gpu=False, summary="List the templates for video or image media by task, recommended first; name gives one template's settings.",
                                     parameters={"media": {"description": "The kind of media to process."},
                                                 "task": {"type": "string", "enum": list(TASKS), "description": "Only the templates of this task."},
                                                 "name": {"type": "string", "description": "A template name: returns the settings that apply to it and an example run."},
                                                 "include_legacy": {"description": "Also list older template versions."}},
                                     example={"plugin": "media_flow", "action": "templates", "arguments": {"media": "video", "task": "refocus"}})
        plugin.register_deepy_action(self.run, summary="Process one file, a list, a folder or wildcards with a template; resumable.", paths={"source": "read", "output": "write"}, routing=ROUTING, instead_of="wangp_postprocess",
                                     parameters={
                                         "process": {"description": "A template name from templates, or a task (outpaint, refocus, uncompress, ungrade, hdr, detail, upscale, interpolate) for its recommended template."},
                                         "source": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "maxItems": 200}],
                                                    "description": "A Gallery media id or an authorized path; a folder, a wildcard pattern (@documents/clips/*.mp4) or a list makes a batch."},
                                         "output": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "maxItems": 200}],
                                                    "description": "Optional: an output file (one source), a folder ending with /, or one line per source line; wildcard outputs mirror wildcard sources. Omitted: next to each source when writable, else @outputs/media_flow/."},
                                         "target": {"anyOf": [{"type": "string"}, {"type": "number"}], "description": "Outpaint: the aspect ratio (16:9, 9:16, 4:3, 3:4, 1:1, 21:9, 9:21). Upscale / interpolate: the multiplier (2 or x2). See the template's settings."},
                                         "strength": {"type": "number", "minimum": 0, "maximum": 3, "description": "Process strength (the LoRA multiplier) for templates that offer it."},
                                         "prompt": {"anyOf": [{"type": "string"}, {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"at": {"anyOf": [{"type": "string", "pattern": "^\\d{1,2}:\\d{2}(?::\\d{2})?(?:\\.\\d+)?$"}, {"type": "number", "minimum": 0}]}, "text": {"type": "string", "minLength": 1}}, "required": ["at", "text"], "additionalProperties": False}}],
                                                    "description": "Optional prompt for templates that use one (their default otherwise); for a video, a list of {at: MM:SS or seconds, text} changes it over time."},
                                         "output_resolution": {"type": "string", "enum": [value for _label, value in ui_constants.HIGH_OUTPUT_RESOLUTION_CHOICES + ui_constants.STANDARD_OUTPUT_RESOLUTION_CHOICES], "description": "Resolution budget of generative templates (default 720p)."},
                                         "chunk_seconds": {"type": "number", "minimum": 0.1, "description": "Video chunk length in seconds (the template's default otherwise)."},
                                         "overlap": {"type": "integer", "minimum": 1, "description": "Frames shared by consecutive video chunks."},
                                         "start": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "string"}], "description": "Video start: seconds or MM:SS(.xx) / HH:MM:SS(.xx)."},
                                         "end": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "string"}], "description": "Video end: seconds or MM:SS(.xx) / HH:MM:SS(.xx)."},
                                         "audio_track": {"type": "integer", "minimum": 1, "maximum": 9, "description": "Source audio track kept in the output (the first otherwise)."},
                                         "parameters": {"type": "object", "description": "Upscaler parameters listed in the template's settings, by name."},
                                         "batch_name": {"type": "string", "minLength": 1, "description": "Name of a batch (several sources); the same name and arguments resume it."},
                                         "resume": {"description": "Continue a stopped run or an existing partial output instead of starting over."},
                                         "media": {"description": "The media kind, when process is a task and source a folder."}},
                                     example={"plugin": "media_flow", "action": "run", "arguments": {"process": "refocus", "source": "@documents/clips/interview.mp4"}})
        plugin.register_deepy_action(self.status, uses_gpu=False, summary="State of the Media Flow process running from the tab or an agent, or of the last one.")
        plugin.register_deepy_action(self.stop, uses_gpu=False, scripts=False, summary="Stop the running Media Flow process after its current chunk; the same run resumes it later.")

    def _library(self) -> ProcessLibrary:
        return ProcessLibrary(get_model_def=self.plugin.get_model_def, get_lora_dir=self.plugin.get_lora_dir, get_base_model_type=self.plugin.get_base_model_type)

    @staticmethod
    def _available(library: ProcessLibrary, name: str) -> bool:
        handler = library.system_handler_for_process(name)
        if handler is None:
            return True
        if getattr(handler, "temporal", False):
            upsampler = temporal_upsampler_api.find_temporal_upsampler_by_method(handler.method)
            return upsampler is not None and temporal_upsampler_api.handler_enabled(upsampler)
        method = library.spatial_upsampler_method(name)
        upsampler = spatial_upsampler_api.find_upsampler_by_method(method) if method else None
        return not method or upsampler is not None and spatial_upsampler_api.handler_enabled(upsampler)

    def _entries(self, library: ProcessLibrary, media: str) -> list[dict[str, Any]]:
        entries = []
        for name, definition in catalog.PROCESS_DEFINITIONS.items():
            if definition["media_kind"] == media and self._available(library, name):
                info = definition["info"]
                entries.append({"name": name, "value": name, "task": info.get("task", "custom"), "summary": info.get("summary", ""), "recommended": bool(info.get("recommended")), "legacy": bool(info.get("legacy")), "source": "system", "refs": []})
        saved, _changed = catalog.migrate_mediaflow_settings(catalog.load_saved_mediaflow_settings(), library.media_kind_for_user_ref)
        refs = catalog.get_saved_user_settings_refs(saved, media)
        for value, definition in library.user_process_definitions(refs, media).items():
            model = library.model_type_label(library.process_definition_model_type(definition))
            entries.append({"name": definition["name"], "value": value, "task": "custom", "summary": f"User settings for {model}.", "recommended": False, "legacy": False, "source": "user", "refs": refs})
        return entries

    def _find(self, library: ProcessLibrary, process: str, media: str | None) -> dict[str, Any]:
        """A template by name, whatever its media (or by task: that task's recommended template for media)."""
        entries = [{**entry, "media": kind} for kind in catalog.VALID_MEDIA_KINDS for entry in self._entries(library, kind)]
        wanted = str(process).strip().casefold()
        named = sorted((entry for entry in entries if wanted in (entry["name"].casefold(), entry["value"].casefold())), key=lambda entry: (entry["media"] != media, entry["source"] != "system"))
        if named:
            return named[0]
        if wanted in TASKS:
            if media is None:
                raise ValueError(f"process={process!r} is a task: say whether the media is a video or an image with media, or name a template. Nothing was started.")
            candidates = [entry for entry in entries if entry["media"] == media and entry["task"] == wanted and not entry["legacy"]]
            if not candidates:
                raise ValueError(f"No available {media} template for {wanted}. Nothing was started.")
            return next((entry for entry in candidates if entry["recommended"]), candidates[0])
        close = difflib.get_close_matches(str(process), [entry["name"] for entry in entries] + list(TASKS), n=3, cutoff=0.4)
        raise ValueError(f"Unknown Media Flow template: {process}. " + (f"Did you mean {' or '.join(close)}? " if close else "") + "List them with action templates. Nothing was started.")

    def _form(self, library: ProcessLibrary, state, entry):
        return create_form_controller(library, self.plugin.get_model_def, self.plugin.server_config).build_form_state(entry["value"], {}, state, entry["refs"], entry["media"])

    def _controls(self, library: ProcessLibrary, state, entry) -> dict[str, Any]:
        """The run arguments that apply to a template, with their choices and defaults (the tab's visible fields)."""
        value, refs, video = entry["value"], entry["refs"], entry["media"] == "video"
        form = self._form(library, state, entry)
        controls: dict[str, Any] = {}
        choices = library.target_control_choices(value, state, refs)
        if choices:
            default = next((label for label, choice in choices if choice == form.target_ratio), choices[0][0])
            controls["target"] = {"label": library.target_control_label(value, state, refs), "choices": [label for label, _choice in choices], "default": default}
        elif library.has_process_outpaint(value, state, refs):
            controls["target"] = {"label": "Target Ratio", "choices": [choice for _label, choice in ui_constants.RATIO_CHOICES], "default": form.target_ratio}
        if library.is_process_strength_visible(value, state, refs):
            controls["strength"] = {"default": form.process_strength, "minimum": 0.0, "maximum": 3.0}
        if not library.hides_prompt(value, state, refs):
            controls["prompt"] = {"default": form.prompt, **({"timed": "a list of {at, text} changes the prompt over time"} if video else {})}
        if not library.hides_output_resolution(value, state, refs):
            controls["output_resolution"] = {"choices": [choice for _label, choice in ui_constants.output_resolution_choices(self.plugin.server_config)], "default": form.output_resolution}
        if video and not library.hides_chunk_size(value, state, refs):
            controls["chunk_seconds"] = {"default": form.chunk_size_seconds}
        if video and not library.hides_sliding_window_overlap(value, state, refs):
            model_type = library.process_definition_model_type(library.process_definition(value, state, refs))
            controls["overlap"] = {"default": form.sliding_window_overlap, "minimum": 1, "maximum": frames.get_overlap_slider_max(model_type, self.plugin.get_model_def), "step": frames.get_vae_temporal_latent_size(model_type, self.plugin.get_model_def)}
        parameters = [{key: item for key, item in parameter.items() if key in ("name", "label", "description", "type", "default", "minimum", "maximum", "step", "choices", "enum")} for parameter in library.spatial_upsampler_parameters(value, state, refs)]
        if parameters:
            controls["parameters"] = parameters
        if video:
            controls.update(start={"default": "the beginning"}, end={"default": "the end"}, audio_track={"default": "the first track"})
        return controls

    def templates(self, ctx, media: Literal["video", "image"], task: str | None = None, name: str | None = None, include_legacy: bool = False) -> dict[str, Any]:
        """Media Flow's templates for one kind of media, grouped by task with the recommended one first, or one template's settings."""
        library = self._library()
        if name is not None:
            entry = self._find(library, name, media)
            definition = library.process_definition(entry["value"], ctx.state, entry["refs"])
            method = library.spatial_upsampler_method(entry["value"], ctx.state, entry["refs"])
            template = {"name": entry["name"], "media": entry["media"], "task": entry["task"], "does": TASKS[entry["task"]], "summary": entry["summary"], "source": entry["source"],
                        "model": library.model_type_label(library.process_definition_model_type(definition))}
            if method and (description := spatial_upsampler_api.method_description(method)):
                template["description"] = description
            if entry["legacy"]:
                template["legacy"] = True
            example = {"process": entry["name"], "source": "@documents/clip.mp4" if entry["media"] == "video" else "@documents/photo.jpg"}
            return {"status": "done", "template": template, "settings": self._controls(library, ctx.state, entry), "example": {"plugin": "media_flow", "action": "run", "arguments": example}}
        if task is not None and task not in TASKS:
            raise ValueError(f"Unknown task: {task}. Tasks: {', '.join(TASKS)}.")
        groups: dict[str, dict[str, Any]] = {}
        for entry in self._entries(library, media):
            if task is not None and entry["task"] != task or entry["legacy"] and not include_legacy:
                continue
            group = groups.setdefault(entry["task"], {"does": TASKS[entry["task"]], "templates": {}})
            label = f"{entry['summary']} (legacy)" if entry["legacy"] else entry["summary"]
            if entry["recommended"] and "recommended" not in group:
                group["recommended"] = entry["name"]
                group["templates"] = {entry["name"]: label, **group["templates"]}
            else:
                group["templates"][entry["name"]] = label
        for task_key, group in groups.items():
            if task_key != "custom":  # user settings are listed, never recommended
                group.setdefault("recommended", next(iter(group["templates"])))
            group["templates"] = group.pop("templates")  # after recommended in the listing
        return {"status": "done", "media": media, "tasks": groups,
                "next": "templates(media, name=<template>) gives one template's settings; run(process=<template or task>, source=...) processes media."}

    def _target(self, library: ProcessLibrary, state, entry, target) -> str:
        choices = library.target_control_choices(entry["value"], state, entry["refs"])
        if choices:
            wanted = _scale(target) if isinstance(target, (int, float)) or _scale(target) is not None else None
            for label, value in choices:
                if str(target).strip().casefold() in (label.casefold(), str(value).casefold()) or wanted is not None and _scale(label) is not None and math.isclose(_scale(label), wanted):
                    return value
            raise ValueError(f"target {target!r} is not offered by {entry['name']}: choose one of {', '.join(label for label, _value in choices)}. Nothing was started.")
        ratios = [choice for _label, choice in ui_constants.RATIO_CHOICES]
        if str(target).strip() not in ratios:
            raise ValueError(f"target {target!r} is not an outpaint ratio: choose one of {', '.join(ratios)}. Nothing was started.")
        return str(target).strip()

    def _output(self, ctx, sources: list[str], output, single: bool) -> str:
        if output is not None:
            if isinstance(output, list):
                if single and len(output) > 1:
                    raise ValueError("One source takes one output. Nothing was started.")
                return "\n".join(output)
            return output
        folders = [Path(source).parent if not Path(source).is_dir() else Path(source) for source in sources]
        if all(ctx.can_write(folder) for folder in folders):
            return ""  # next to each source, as the tab does
        target = ctx.outputs_dir / "media_flow"
        target.mkdir(parents=True, exist_ok=True)
        return str(target) + "/"

    def run(self, ctx, process: str, source: str | list[str], output: str | list[str] | None = None, target: str | float | None = None, strength: float | None = None,
            prompt: str | list[dict[str, Any]] | None = None, output_resolution: str | None = None, chunk_seconds: float | None = None, overlap: int | None = None,
            start: float | str | None = None, end: float | str | None = None, audio_track: int | None = None, parameters: dict[str, Any] | None = None,
            batch_name: str | None = None, resume: bool = True, media: Literal["video", "image"] | None = None) -> dict[str, Any]:
        """Process existing media with a Media Flow template: one file, or a batch of a list, folders or wildcard patterns. A run that was stopped or
        interrupted continues where it stopped when called again with the same arguments (resume=true). Only the settings of the template's settings list apply.
        Runs keep only the last Media Generator artifact; the tab's retention setting is restored afterward. The result lists the outputs in generated_files."""
        job = plugin_active_job(self.plugin)
        if job.get("running") or job.get("batch_running"):
            raise ValueError("Media Flow is already running a process (from its tab or another agent): follow it with action status or stop it with action stop. Nothing was started.")
        sources = source if isinstance(source, list) else [source]
        single = len(sources) == 1 and not _WILDCARD.search(sources[0]) and Path(sources[0]).is_file()
        library = self._library()
        entry = self._find(library, process, media or next((kind for kind in map(_media_of, sources) if kind), None))
        if any(kind not in (None, entry["media"]) for kind in map(_media_of, sources)):
            raise ValueError(f"{entry['name']} processes {entry['media']}s: every source must be a {entry['media']}, a folder or a {entry['media']} pattern. Nothing was started.")
        controls = self._controls(library, ctx.state, entry)
        supplied = {"target": target, "strength": strength, "prompt": prompt, "output_resolution": output_resolution, "chunk_seconds": chunk_seconds, "overlap": overlap, "start": start, "end": end, "audio_track": audio_track, "parameters": parameters}
        extra = [key for key, value in supplied.items() if value is not None and key not in controls]
        if extra:
            raise ValueError(f"{', '.join(extra)} {'does' if len(extra) == 1 else 'do'} not apply to {entry['name']}; its settings: {', '.join(controls) or 'none'}. Nothing was started.")
        if isinstance(prompt, list) and entry["media"] != "video":
            raise ValueError("A timed prompt list applies to videos only: give one prompt string. Nothing was started.")
        known = {parameter["name"] for parameter in controls.get("parameters", [])}
        upsampler_parameters = dict(parameters or {})
        if set(upsampler_parameters) - known:
            raise ValueError(f"Unknown parameters for {entry['name']}: {', '.join(sorted(set(upsampler_parameters) - known))}; offered: {', '.join(sorted(known))}. Nothing was started.")
        form = self._form(library, ctx.state, entry)
        output_path = self._output(ctx, sources, output, single)
        if not single and not batch_name:
            first = Path(sources[0])
            batch_name = f"{entry['name']} {first.stem if first.suffix else first.name}".strip()
        arguments = dict(state=ctx.state, process_name=entry["value"], user_refs=entry["refs"], source_path=sources[0] if single else "", process_strength=strength, output_path=output_path,
                         prompt_text=form.prompt if prompt is None else _prompt_text(prompt), continue_enabled=bool(resume), preview_enabled=catalog.normalize_preview_enabled(job.get(catalog.PREVIEW_OUTPUT_STORAGE_KEY)),
                         source_audio_track="" if audio_track is None else str(audio_track), output_resolution=output_resolution or form.output_resolution,
                         target_ratio=form.target_ratio if target is None else self._target(library, ctx.state, entry, target), chunk_size_seconds=chunk_seconds or form.chunk_size_seconds,
                         sliding_window_overlap=overlap or form.sliding_window_overlap, start_seconds="" if start is None else str(start), end_seconds="" if end is None else str(end),
                         batch_mode="single" if single else "batch", batch_name=batch_name or "", batch_source_path="" if single else "\n".join(sources), spatial_upsampler_parameters=upsampler_parameters)
        for key in ("last_process_result", "last_batch", "status_fields", "batch_fields"):
            job[key] = None
        notice = {"text": ""}
        skip = object()

        def ui_update(status=skip, output=skip, preview_refresh=skip, *, batch_status=skip, **_ignored):
            remember_status(job, status, batch_status)

        def info_exit(message: str, **_ignored):
            notice["text"] = str(message or "").strip()
            ui_update(render_chunk_status_html(1, 0, 1, "Info", notice["text"]))

        runner = ProcessRunner(plugin=self.plugin, api_session=ctx.session, library=library, get_model_def=self.plugin.get_model_def, active_job=job, preview_state={"image": None},
                               ui_skip=skip, ui_update=ui_update, info_exit=info_exit, reset_live_chunk_status=reset_live_chunk_status, remember_ui_settings=False)
        finished = threading.Event()

        def watch():  # the caller's Stop (chat button, cancelled job) stops the process after its current chunk; progress at most three times a second
            stopping = False
            while not finished.wait(0.34):
                if not stopping and ctx.cancelled():
                    stopping = request_stop(job) or True
                self._report(ctx, job, single)

        started = time.monotonic()
        watcher = threading.Thread(target=watch, name="mediaflow-agent-run", daemon=True)
        watcher.start()
        preserve_artifact_count = job[catalog.PRESERVE_ARTIFACTS_STORAGE_KEY]
        job[catalog.PRESERVE_ARTIFACTS_STORAGE_KEY] = 1
        try:
            for _update in runner.start_process(**arguments):
                pass
        finally:
            job[catalog.PRESERVE_ARTIFACTS_STORAGE_KEY] = preserve_artifact_count
            finished.set()
            watcher.join()
        return self._result(job, entry, single, notice["text"], round(time.monotonic() - started, 1))

    @staticmethod
    def _report(ctx, job: dict, single: bool) -> None:
        status, batch = job.get("status_fields") or {}, job.get("batch_fields")
        phase = str(status.get("phase") or status.get("status") or "").strip()
        if status.get("step") is not None and status.get("steps"):
            phase = f"{phase} {status['step']}/{status['steps']}".strip()
        if batch and not single:
            ctx.progress(batch["completed_files"], batch["total_files"], " · ".join(part for part in (Path(batch["current_file"]).name, phase) if part))
        elif status.get("total_chunks", 0) > 1:
            ctx.progress(status["completed_chunks"], status["total_chunks"], f"chunk {status['current_chunk']}" + (f" · {phase}" if phase else ""))
        else:
            ctx.progress(None, None, phase)

    @staticmethod
    def _result(job: dict, entry: dict, single: bool, notice: str, elapsed: float) -> dict[str, Any]:
        result = job.get("last_process_result")
        if single or job.get("last_batch") is None:
            if result is None:
                return {"status": "error", "process": entry["name"], "error": notice or "Media Flow ended without a result."}
            payload = {"status": "done" if result.success else "stopped" if result.stopped else "error", "process": entry["name"], "message": result.message, "elapsed_s": elapsed}
            if result.output_path and Path(result.output_path).is_file():
                payload.update(output_file=result.output_path, generated_files=[result.output_path])  # generated_files as in a batch result
            if entry["media"] == "video" and result.chunks_total:
                payload["chunks"] = {"done": result.chunks_completed, "total": result.chunks_total}
            if result.stopped:
                payload["resume"] = "Run the same call again (resume=true) to continue where it stopped."
            elif not result.success:
                payload["error"] = result.message
            return payload
        batch = job["last_batch"]
        items = batch["state"]["items"]
        done = [item for item in items if item["status"] == "success"]
        failed = [{"source": item["source_path"], "error": item["error"]} for item in items if item["status"] == "failed"]
        stopped = bool(result is not None and result.stopped)
        payload = {"status": "stopped" if stopped else "done" if done or not failed else "error", "process": entry["name"], "batch_name": batch["name"], "message": result.message if result is not None else notice,
                   "items": {"total": len(items), "done": len(done), "failed": len(failed), "remaining": len(items) - len(done) - len(failed)}, "elapsed_s": elapsed,
                   "generated_files": [item["actual_output_path"] for item in done if item["actual_output_path"]][:100], "state_file": batch["state_file"], "log_file": batch["log_file"]}
        if failed:
            payload["failed_items"] = failed[:20]
        if stopped:
            payload["resume"] = "Run the same call again (same batch_name, resume=true) to continue with the remaining items."
        return payload

    def status(self, ctx) -> dict[str, Any]:
        """Whether a Media Flow process runs (started from the tab or by an agent), with its progress, or the outcome of the last one."""
        job = plugin_active_job(self.plugin)
        running = bool(job.get("running") or job.get("batch_running"))
        payload: dict[str, Any] = {"status": "running" if running else "idle"}
        if job.get("batch_fields") and job.get("last_batch"):
            fields = job["batch_fields"]
            payload["batch"] = {"files_done": fields["completed_files"], "files_total": fields["total_files"], "failed": fields["failed_files"], "current_file": fields["current_file"], "status": fields["status"]}
        if fields := job.get("status_fields"):
            current = {key: fields[key] for key in ("completed_chunks", "total_chunks", "current_chunk", "phase", "status", "step", "steps") if fields.get(key) not in (None, "")}
            payload["current" if running else "last_status"] = current
        result = job.get("last_process_result")
        if result is not None and not running:
            payload["last_result"] = {"success": result.success, "stopped": result.stopped, "message": result.message, **({"output_file": result.output_path} if result.output_path else {})}
        if job.get("cancel_requested") and running:
            payload["stopping"] = True
        return payload

    def stop(self, ctx) -> dict[str, Any]:
        """Stop the running Media Flow process after its current chunk; calling the same run again later resumes it."""
        if not request_stop(plugin_active_job(self.plugin)):
            return {"status": "idle", "message": "No Media Flow process is running."}
        return {"status": "stopping", "message": "Media Flow stops after its current chunk; the same run resumes it."}
