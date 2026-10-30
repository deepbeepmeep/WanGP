# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
from PIL import Image

from shared.utils.hdr import VIDEO_PROMPT_HDR_OUTPUT_FLAG
from shared.utils.utils import has_image_file_extension
from postprocessing import parameter_ui

from .chunk_executor import ChunkExecutor, ChunkProgress, ProcessContext, build_task_settings
from . import batch_queue
from . import common
from . import constants as ui_constants
from . import continuation_recovery
from . import frame_planning as frames
from . import media_io as media
from . import output_paths
from . import process_catalog as catalog
from . import process_metadata
from . import prompt_schedule as prompts
from . import status_ui
from . import video_buffers as video
from .mux_session import MuxSession
from .run_preparation import ProcessInfoExit, prepare_run


@dataclass(frozen=True)
class RunRequest:
    state: dict | None = None
    process_name: str = ""
    user_refs: list[str] = field(default_factory=list)
    source_path: str = ""
    process_strength: object = None
    output_path: str = ""
    prompt_text: str = ""
    continue_enabled: bool = True
    source_audio_track: str = ""
    output_resolution: str = "720p"
    target_ratio: str = ""
    chunk_size_seconds: object = 10.0
    sliding_window_overlap: object = 1
    start_seconds: str = ""
    end_seconds: str = ""
    spatial_upsampler_parameters: dict = field(default_factory=dict)

    @classmethod
    def from_gradio(
        cls,
        state=None,
        process_name="",
        user_refs=None,
        source_path="",
        process_strength=None,
        output_path="",
        prompt_text="",
        continue_enabled=True,
        source_audio_track="",
        output_resolution="720p",
        target_ratio="",
        chunk_size_seconds=10.0,
        sliding_window_overlap=1,
        start_seconds="",
        end_seconds="",
        spatial_upsampler_parameters=None,
    ) -> "RunRequest":
        return cls(
            state=state,
            process_name=str(process_name or "").strip(),
            user_refs=list(user_refs or []),
            source_path=str(source_path or "").strip(),
            process_strength=process_strength,
            output_path=str(output_path or "").strip(),
            prompt_text=str(prompt_text or ""),
            continue_enabled=bool(continue_enabled),
            source_audio_track=str(source_audio_track or "").strip(),
            output_resolution=str(output_resolution or "720p").strip(),
            target_ratio=str(target_ratio or "").strip(),
            chunk_size_seconds=chunk_size_seconds,
            sliding_window_overlap=sliding_window_overlap,
            start_seconds="" if start_seconds in (None, "") else str(start_seconds),
            end_seconds="" if end_seconds in (None, "") else str(end_seconds),
            spatial_upsampler_parameters={parameter_ui.LEGACY_NAMES.get(str(key), str(key)): value for key, value in dict(spatial_upsampler_parameters or {}).items()},
        )


@dataclass(frozen=True)
class ProcessRunResult:
    source_path: str = ""
    output_path: str = ""
    success: bool = False
    stopped: bool = False
    message: str = ""
    chunks_completed: int = 0
    chunks_total: int = 0


def plugin_active_job(plugin) -> dict:
    """The one job state of the plugin, shared by its tab and its agent actions so that a single process runs at a time; made on first use."""
    if getattr(plugin, "active_job", None) is None:
        plugin.active_job = {"job": None, "running": False, "batch_running": False, "cancel_requested": False, "write_state": None, catalog.PRESERVE_ARTIFACTS_STORAGE_KEY: catalog.PRESERVE_ARTIFACTS_DEFAULT,
                             catalog.PREVIEW_OUTPUT_STORAGE_KEY: catalog.PREVIEW_OUTPUT_DEFAULT, "generated_artifact_paths": []}
    return plugin.active_job


def remember_status(active_job: dict, status, batch_status) -> None:
    """Keep the values of the last rendered statuses, for agents asking how the running process (from the tab or an agent) is going."""
    if hasattr(status, "fields"):
        active_job["status_fields"] = status.fields
    if hasattr(batch_status, "fields"):
        active_job["batch_fields"] = batch_status.fields


def reset_live_chunk_status(state: dict) -> None:
    gen = state.get("gen") if isinstance(state, dict) else None
    if not isinstance(gen, dict):
        return
    gen["status"] = ""
    gen["status_display"] = False
    gen["progress_args"] = None
    gen["progress_phase"] = None
    gen["progress_status"] = ""
    gen["preview"] = None


def request_stop(active_job: dict) -> bool:
    """Stop the running process after its current chunk (a later run with Continue resumes it); True when a process was running."""
    active_job["cancel_requested"] = True
    write_state = active_job.get("write_state")
    if write_state is not None:
        write_state.stopped = True
    job = active_job.get("job")
    if job is not None and not job.done:
        try:
            job.cancel()
        except RuntimeError as exc:
            print(f"[MediaFlow] Stop requested; WanGP abort bridge was not available: {exc}")
        return True
    return bool(active_job.get("running") or active_job.get("batch_running"))


class ProcessRunner:
    def __init__(self, *, plugin, api_session, library, get_model_def, active_job: dict, preview_state: dict, ui_skip, ui_update, info_exit, reset_live_chunk_status, remember_ui_settings: bool = True) -> None:
        self.plugin = plugin
        self.api_session = api_session
        self.library = library
        self.get_model_def = get_model_def
        self.active_job = active_job
        self.preview_state = preview_state
        self.ui_skip = ui_skip
        self.ui_update = ui_update
        self.info_exit = info_exit
        self.reset_live_chunk_status = reset_live_chunk_status
        self.remember_ui_settings = remember_ui_settings  # agent runs leave the tab's saved form alone

    def _save_ui_settings(self, settings: dict, media_kind: str, batch_mode: str) -> None:
        if self.remember_ui_settings:
            catalog.save_mediaflow_ui_settings(settings, media_kind, batch_mode)

    def _record_process_result(self, result: ProcessRunResult) -> ProcessRunResult:
        self.active_job["last_process_result"] = result
        return result

    def _preserve_artifact_count(self) -> int:
        return catalog.normalize_preserve_artifact_count(self.active_job.get(catalog.PRESERVE_ARTIFACTS_STORAGE_KEY))

    def _preview_enabled(self) -> bool:
        return catalog.normalize_preview_enabled(self.active_job.get(catalog.PREVIEW_OUTPUT_STORAGE_KEY))

    def _reset_generated_artifacts(self) -> None:
        self.active_job["generated_artifact_paths"] = []

    def _prune_generated_artifacts(self, state: dict, *, preserve_paths: list[str] | None = None):
        kept_paths, gallery_changed = media.prune_active_job_artifacts(self.plugin, self.active_job, state, preserve_paths=preserve_paths)
        return kept_paths, str(time.time_ns()) if gallery_changed else self.ui_skip

    @staticmethod
    def _preserve_failed_mux_output(write_state: MuxSession, system_handler) -> str:
        write_state.cleanup_partial_outputs()
        working_path = write_state.mux_output_path
        if not working_path or not os.path.isfile(working_path) or os.path.getsize(working_path) <= 0:
            return ""
        try:
            cache_source_path, preserved_path = write_state.promote_output(notify=common.plugin_info)
        except Exception as exc:
            common.plugin_info(f"Unable to promote the partial output; preserving it at {working_path}. {exc}")
            return working_path
        if cache_source_path != preserved_path and system_handler is not None and callable(getattr(system_handler, "move_continue_cache", None)):
            try:
                system_handler.move_continue_cache(cache_source_path, preserved_path)
            except Exception as exc:
                common.plugin_info(f"Unable to move the continuation cache beside {preserved_path}; it remains beside {cache_source_path}. {exc}")
        common.plugin_info(f"Preserved partial output at {preserved_path}")
        return preserved_path

    @staticmethod
    def _batch_completed_count(state: dict) -> int:
        return sum(1 for item in list(state.get("items") or []) if str(item.get("status") or "") in {"success", "failed"})

    @staticmethod
    def _batch_failed_count(state: dict) -> int:
        return sum(1 for item in list(state.get("items") or []) if str(item.get("status") or "") == "failed")

    @staticmethod
    def _batch_resume_index(state: dict, total_items: int) -> int:
        for index, item in enumerate(list(state.get("items") or [])[:total_items]):
            if str(item.get("status") or "") not in {"success", "failed"}:
                return index
        return total_items

    @staticmethod
    def _process_is_image(process_definition: dict | None) -> bool:
        settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
        if not isinstance(settings, dict):
            return False
        try:
            return int(settings.get("image_mode") or 0) == 1
        except (TypeError, ValueError):
            return False

    def _image_output_extension(self) -> str:
        return output_paths.image_output_extension(self.plugin.server_config.get("image_output_codec", "jpeg_95"))

    @staticmethod
    def _image_result_path(result) -> str:
        for path in list(getattr(result, "generated_files", []) or []):
            if path and os.path.isfile(path):
                return str(path)
        for artifact in tuple(getattr(result, "artifacts", ()) or ()):
            if str(getattr(artifact, "media_type", "") or "").strip() == "image" and getattr(artifact, "path", None) and os.path.isfile(artifact.path):
                return str(artifact.path)
        return ""

    @staticmethod
    def _image_preview_from_result(result):
        for artifact in tuple(getattr(result, "artifacts", ()) or ()):
            video_tensor_uint8 = getattr(artifact, "video_tensor_uint8", None)
            if video_tensor_uint8 is not None and int(video_tensor_uint8.shape[1]) > 0:
                return video.frame_to_image(video_tensor_uint8[:, -1].detach().cpu().contiguous())
        return None

    @staticmethod
    def _map_source_image_settings(settings: dict, source_path: str) -> None:
        video_prompt_type = str(settings.get("video_prompt_type") or "")
        image_prompt_type = str(settings.get("image_prompt_type") or "")
        if "V" in video_prompt_type:
            settings["image_guide"] = source_path
            settings.pop("video_guide", None)
        elif "I" in video_prompt_type or settings.get("image_refs") is not None:
            settings["image_refs"] = [source_path]
        elif "S" in image_prompt_type or settings.get("image_start") is not None:
            settings["image_start"] = source_path
        else:
            settings["video_prompt_type"] = "V"
            settings["image_guide"] = source_path
            settings.pop("video_guide", None)

    @staticmethod
    def _is_spatial_upsampling_task(settings: dict) -> bool:
        return len(str(settings.get("spatial_upsampling") or settings.get("spatial_upsampling_method") or "").strip()) > 0

    def _build_image_task_settings(self, process_definition: dict, system_handler, *, source_path: str, prompt_text: str, process_strength: float, use_lora_strength_override: bool, system_target_control: str, target_ratio: str, output_resolution: str, spatial_upsampler_parameters: dict) -> dict:
        process_settings = process_definition["settings"]
        if system_handler is not None and callable(getattr(system_handler, "build_image_queue_settings", None)):
            seed = int(time.time_ns() % 2_147_483_647)
            settings = system_handler.build_image_queue_settings(process_settings, source_path=source_path, target_control=system_target_control, seed=seed)
        else:
            settings = build_task_settings(process_settings, is_user_process=process_definition.get("source") == "user")
            settings["model_type"] = str(process_settings.get("model_type") or "")
            settings["image_mode"] = 1
            settings["repeat_generation"] = 1
            settings["batch_size"] = 1
            settings["prompt"] = prompt_text.strip() or str(process_settings.get("prompt") or "")
            if use_lora_strength_override:
                settings["loras_multipliers"] = str(process_strength)
            self._map_source_image_settings(settings, source_path)
            if "video_guide_outpainting" in settings:
                settings["video_guide_outpainting_ratio"] = str(target_ratio or "").strip()
        if not self._is_spatial_upsampling_task(settings):
            settings["resolution"] = output_paths.choose_resolution(output_resolution)
        spatial = parameter_ui.SETTINGS[parameter_ui.SPATIAL]  # the process's own values completed with the user's
        settings[spatial] = {**dict(settings.get(spatial) or {}), **dict(spatial_upsampler_parameters)}
        api_settings = settings.get("_api")
        settings["_api"] = dict(api_settings) if isinstance(api_settings, dict) else {}
        for key in ("return_media", "return_video_uint8", "return_audio", "return_flashvsr_continue_cache", "flashvsr_continue_cache"):
            settings["_api"].pop(key, None)
        if self._preview_enabled():
            settings["_api"]["return_video_uint8"] = True
        return settings

    def _move_image_output(self, generated_path: str, target_path: str) -> str:
        generated = Path(generated_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if generated.resolve() == target.resolve():
            return str(target)
        shutil.copy2(str(generated), str(target))
        return str(target)

    def _start_image_process(self, request: RunRequest, process_definition: dict, system_handler, *, process_display_name: str, active_process_strength: float, use_lora_strength_override: bool, active_target_ratio: str, system_target_control: str, batch_internal: bool, batch_name: str, batch_source_path: str):
        source_path = request.source_path
        output_path = request.output_path
        if not os.path.isfile(source_path):
            message = f"Source image not found: {source_path}"
            yield self.info_exit(message)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
            return
        if not has_image_file_extension(source_path):
            message = "Source Image Path must use a supported image extension."
            yield self.info_exit(message)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
            return
        target_output_path = output_paths.resolve_image_output_path(source_path, output_path, process_display_name, default_extension=self._image_output_extension(), notify=common.plugin_info)
        if not batch_internal:
            try:
                self._save_ui_settings({
                    "process_model_type": str(process_definition["settings"].get("model_type") or ""),
                    "process_name": request.process_name,
                    "source_path": source_path,
                    "batch_source_path": str(batch_source_path or "").strip(),
                    "batch_mode": "single",
                    "media_kind": "image",
                    "batch_name": str(batch_name or "").strip(),
                    "process_strength": active_process_strength,
                    "output_path": output_path,
                    "prompt": request.prompt_text,
                    "continue_enabled": request.continue_enabled,
                    "source_audio_track": request.source_audio_track,
                    "output_resolution": request.output_resolution,
                    "target_ratio": system_target_control if system_handler is not None else active_target_ratio,
                    "chunk_size_seconds": request.chunk_size_seconds,
                    "sliding_window_overlap": request.sliding_window_overlap,
                    "start_seconds": request.start_seconds,
                    "end_seconds": request.end_seconds,
                    "spatial_upsampler_parameters": request.spatial_upsampler_parameters,
                    catalog.PRESERVE_ARTIFACTS_STORAGE_KEY: self._preserve_artifact_count(),
                    catalog.PREVIEW_OUTPUT_STORAGE_KEY: self._preview_enabled(),
                }, "image", "single")
            except OSError as exc:
                message = f"Unable to save plugin settings to {catalog.MEDIAFLOW_SETTINGS_FILE}: {exc}"
                yield self.info_exit(message)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
                return
        callbacks = status_ui.ChunkCallbacks()
        if not batch_internal:
            self.preview_state["image"] = None
        self.reset_live_chunk_status(request.state)
        self.active_job["cancel_requested"] = False
        self.active_job["running"] = True
        run_started_at = time.time()
        settings = self._build_image_task_settings(
            process_definition,
            system_handler,
            source_path=source_path,
            prompt_text=request.prompt_text,
            process_strength=active_process_strength,
            use_lora_strength_override=use_lora_strength_override,
            system_target_control=system_target_control,
            target_ratio=active_target_ratio,
            output_resolution=request.output_resolution,
            spatial_upsampler_parameters=request.spatial_upsampler_parameters,
        )
        try:
            yield self.ui_update(status_ui.render_process_status_html("Initializing", "Preparing image processing job..."), self.ui_skip, str(time.time_ns()) if not batch_internal else self.ui_skip, start_enabled=False, abort_enabled=False)
            job = self.api_session.submit_task(settings, callbacks=callbacks)
            self.active_job["job"] = job
            yield self.ui_update(start_enabled=False, abort_enabled=True)
            next_status_refresh_at = 0.0
            stop_requested = False
            while not job.done:
                if self.active_job.get("cancel_requested") and not stop_requested:
                    try:
                        job.cancel()
                    except RuntimeError as exc:
                        print(f"[MediaFlow] Stop requested; WanGP abort bridge was not available: {exc}")
                    stop_requested = True
                now = time.monotonic()
                if now >= next_status_refresh_at:
                    elapsed_seconds = time.time() - run_started_at
                    yield self.ui_update(
                        status_ui.render_process_status_html(
                            callbacks.phase_label,
                            callbacks.status_text,
                            phase_current_step=callbacks.current_step,
                            phase_total_steps=callbacks.total_steps,
                            elapsed_seconds=elapsed_seconds,
                            prefer_status_phase=True,
                        ),
                        self.ui_skip,
                        str(time.time_ns()),
                        start_enabled=False,
                        abort_enabled=True,
                    )
                    next_status_refresh_at = now + ui_constants.STATUS_REFRESH_INTERVAL_SECONDS
                time.sleep(0.05)
            result = job.result()
            self.active_job["job"] = None
            if result.cancelled or stop_requested:
                message = "Stopped before image output was completed."
                yield self.ui_update(status_ui.render_process_status_html("Stopped", message, elapsed_seconds=time.time() - run_started_at), self.ui_skip, self.ui_skip, start_enabled=True, abort_enabled=False)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=target_output_path, success=False, stopped=True, message=message, chunks_completed=0, chunks_total=1))
                return
            if not result.success:
                message = "\n".join(str(error) for error in result.errors) or "Image processing failed."
                raise gr.Error(message)
            generated_path = self._image_result_path(result)
            if not generated_path:
                raise gr.Error("Image processing completed without creating an output image.")
            final_output_path = self._move_image_output(generated_path, target_output_path)
            media.remember_generated_artifacts(self.active_job, media.result_generated_artifact_paths(result))
            _kept_artifact_paths, gallery_refresh = self._prune_generated_artifacts(request.state, preserve_paths=[final_output_path])
            if self._preview_enabled():
                image_preview = self._image_preview_from_result(result)
                try:
                    if image_preview is None:
                        with Image.open(final_output_path) as preview:
                            image_preview = preview.copy()
                    if image_preview is not None:
                        try:
                            delattr(image_preview, "filename")
                        except (AttributeError, TypeError):
                            pass
                        self.preview_state["image"] = image_preview
                except Exception:
                    self.preview_state["image"] = None
            message = "Completed image process."
            yield self.ui_update(status_ui.render_process_status_html("Completed", message, elapsed_seconds=time.time() - run_started_at), final_output_path, str(time.time_ns()), start_enabled=True, abort_enabled=False, gallery_refresh=gallery_refresh)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=final_output_path, success=True, message=message, chunks_completed=1, chunks_total=1))
        except gr.Error as exc:
            self.active_job["job"] = None
            message = common.get_error_message(exc) or "Image processing failed."
            yield self.ui_update(status_ui.render_process_status_html("Error", message, elapsed_seconds=time.time() - run_started_at), target_output_path if os.path.isfile(target_output_path) else self.ui_skip, self.ui_skip, start_enabled=True, abort_enabled=False)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=target_output_path if os.path.isfile(target_output_path) else "", success=False, message=message, chunks_completed=0, chunks_total=1))
            return
        finally:
            self.active_job["running"] = False

    def start_batch_process(self, state=None, process_name="", user_refs=None, source_path="", process_strength=None, output_path="", prompt_text="", continue_enabled=True, preview_enabled=True, source_audio_track="", output_resolution="720p", target_ratio="", chunk_size_seconds=10.0, sliding_window_overlap=1, start_seconds="", end_seconds="", batch_name="", batch_source_path="", spatial_upsampler_parameters=None):
        if self.active_job.get("running") or self.active_job.get("batch_running"):
            yield self.info_exit("A process is already running.")
            return
        self.active_job[catalog.PREVIEW_OUTPUT_STORAGE_KEY] = catalog.normalize_preview_enabled(preview_enabled)
        if not self._preview_enabled():
            self.preview_state["image"] = None
        batch_name = str(batch_name or "").strip()
        if len(batch_name) == 0:
            yield self.info_exit("Batch Name is required in batch mode.")
            return
        requested_batch_name = batch_name
        request = RunRequest.from_gradio(state, process_name, user_refs, source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, spatial_upsampler_parameters)
        process_definition = self.library.process_definition(request.process_name, request.state, request.user_refs)
        if process_definition is None:
            yield self.info_exit(f"Unsupported process: {request.process_name}")
            return
        system_handler = self.library.system_handler_for_definition(process_definition)
        model_type = str((process_definition.get("settings") or {}).get("model_type") or "")
        if len(model_type) == 0 and system_handler is None:
            yield self.info_exit(f"Unsupported process: {request.process_name}")
            return
        media_kind = "image" if self._process_is_image(process_definition) else "video"
        try:
            items = batch_queue.expand_batch_items(batch_source_path, output_path, media_kind=media_kind)
        except gr.Error as exc:
            yield self.info_exit(common.get_error_message(exc) or "Invalid batch paths.")
            return
        if len(items) == 0:
            yield self.info_exit(f"Batch has no source {media_kind}s to process.")
            return
        self._reset_generated_artifacts()
        signature = batch_queue.build_signature(request, items, batch_name=batch_name, process_model_type=model_type, media_kind=media_kind)
        signature_hash = batch_queue.signature_hash(signature)
        files = batch_queue.resolve_batch_files(batch_name, items[0], signature_hash, bool(request.continue_enabled))
        if files.renamed:
            batch_name = files.batch_name
            signature = batch_queue.build_signature(request, items, batch_name=batch_name, process_model_type=model_type, media_kind=media_kind)
        elif files.batch_name:
            batch_name = files.batch_name
        batch_state = files.state if files.state is not None else batch_queue.make_state(batch_name, signature, items)
        self.active_job["last_batch"] = {"name": batch_name, "state_file": str(files.state_path), "log_file": str(files.log_path), "state": batch_state}  # updated in place as items finish
        if files.renamed:
            common.plugin_info(f'Batch name "{requested_batch_name}" already has a state file. Using "{batch_name}".')
        try:
            self._save_ui_settings({
                "process_model_type": model_type,
                "process_name": request.process_name,
                "source_path": source_path,
                "batch_source_path": str(batch_source_path or "").strip(),
                "batch_mode": "batch",
                "batch_name": batch_name,
                "media_kind": media_kind,
                "process_strength": request.process_strength,
                "output_path": output_path,
                "prompt": prompt_text,
                "continue_enabled": request.continue_enabled,
                "source_audio_track": source_audio_track,
                "output_resolution": output_resolution,
                "target_ratio": target_ratio,
                "chunk_size_seconds": chunk_size_seconds,
                "sliding_window_overlap": sliding_window_overlap,
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "spatial_upsampler_parameters": request.spatial_upsampler_parameters,
                catalog.PRESERVE_ARTIFACTS_STORAGE_KEY: self._preserve_artifact_count(),
                catalog.PREVIEW_OUTPUT_STORAGE_KEY: self._preview_enabled(),
            }, media_kind, "batch")
        except OSError as exc:
            yield self.info_exit(f"Unable to save plugin settings to {catalog.MEDIAFLOW_SETTINGS_FILE}: {exc}")
            return

        if files.completed:
            completed_files = self._batch_completed_count(batch_state)
            failed_files = self._batch_failed_count(batch_state)
            message = "Batch already completed. Nothing to do."
            gr.Info(message)
            yield self.ui_update(
                status_ui.render_process_status_html("Batch Completed", message) if media_kind == "image" else status_ui.render_chunk_status_html(0, 0, 0, "Batch Completed", message),
                self.ui_skip,
                self.ui_skip,
                batch_status=status_ui.render_batch_status_html(len(items), completed_files, len(items), "", message, failed_files),
                start_enabled=True,
                abort_enabled=False,
            )
            self._record_process_result(ProcessRunResult(success=True, message=message, chunks_completed=completed_files, chunks_total=len(items)))
            return

        batch_queue.write_state(files.state_path, batch_state)
        batch_queue.append_log(files.log_path, "resumed" if files.resumed else "started", batch_name=batch_name, requested_batch_name=requested_batch_name, state_file=str(files.state_path), total_files=len(items), renamed=files.renamed)
        self.active_job["batch_running"] = True
        self.active_job["cancel_requested"] = False
        start_index = self._batch_resume_index(batch_state, len(items))
        completed_files = self._batch_completed_count(batch_state)
        failed_files = self._batch_failed_count(batch_state)
        yield self.ui_update(
            status_ui.render_process_status_html("Batch", "Preparing batch queue...") if media_kind == "image" else status_ui.render_chunk_status_html(0, 0, 0, "Batch", "Preparing batch queue..."),
            self.ui_skip,
            str(time.time_ns()),
            batch_status=status_ui.render_batch_status_html(len(items), completed_files, start_index + 1 if start_index < len(items) else len(items), Path(items[start_index].source_path).name if start_index < len(items) else "", "Resuming batch..." if files.resumed else "Starting batch...", failed_files),
            start_enabled=False,
            abort_enabled=False,
            batch_name_value=batch_name if files.renamed else self.ui_skip,
        )
        try:
            for index in range(start_index, len(items)):
                if self.active_job.get("cancel_requested"):
                    batch_state["status"] = "stopped"
                    batch_state["next_index"] = index
                    batch_queue.write_state(files.state_path, batch_state)
                    batch_queue.append_log(files.log_path, "stopped", batch_name=batch_name, next_index=index)
                    yield self.ui_update(batch_status=status_ui.render_batch_status_html(len(items), completed_files, index + 1, Path(items[index].source_path).name, "Stopped.", failed_files), start_enabled=True, abort_enabled=False)
                    self._record_process_result(ProcessRunResult(success=False, stopped=True, message="Batch stopped.", chunks_completed=completed_files, chunks_total=len(items)))
                    return
                item = items[index]
                item_output_path = batch_queue.output_path_for_process(item)
                item_state = batch_state["items"][index]
                item_state["status"] = "running"
                item_state["started_at"] = batch_queue.now_iso()
                item_state["ended_at"] = ""
                item_state["error"] = ""
                item_state["actual_output_path"] = ""
                batch_state["status"] = "running"
                batch_state["next_index"] = index
                batch_queue.write_state(files.state_path, batch_state)
                batch_queue.append_log(files.log_path, "file_started", batch_name=batch_name, index=index, source_path=item.source_path, output_path=item_output_path)
                yield self.ui_update(batch_status=status_ui.render_batch_status_html(len(items), completed_files, index + 1, Path(item.source_path).name, "Processing file...", failed_files), start_enabled=False, abort_enabled=False)
                file_started_at = time.time()
                try:
                    yield from self.start_process(
                        state=request.state,
                        process_name=request.process_name,
                        user_refs=request.user_refs,
                        source_path=item.source_path,
                        process_strength=request.process_strength,
                        output_path=item_output_path,
                        prompt_text=request.prompt_text,
                        continue_enabled=request.continue_enabled,
                        preview_enabled=self._preview_enabled(),
                        source_audio_track=request.source_audio_track,
                        output_resolution=request.output_resolution,
                        target_ratio=request.target_ratio,
                        chunk_size_seconds=request.chunk_size_seconds,
                        sliding_window_overlap=request.sliding_window_overlap,
                        start_seconds=request.start_seconds,
                        end_seconds=request.end_seconds,
                        batch_mode="single",
                        batch_name=batch_name,
                        batch_source_path=batch_source_path,
                        spatial_upsampler_parameters=request.spatial_upsampler_parameters,
                        batch_internal=True,
                    )
                    result = self.active_job.get("last_process_result")
                    if not isinstance(result, ProcessRunResult):
                        result = ProcessRunResult(source_path=item.source_path, output_path=item.output_path, success=False, message="Processing ended without a result.")
                except Exception as exc:
                    result = ProcessRunResult(source_path=item.source_path, output_path=item.output_path, success=False, message=common.get_error_message(exc) or str(exc))
                elapsed_seconds = max(0.0, time.time() - file_started_at)
                item_state["ended_at"] = batch_queue.now_iso()
                item_state["elapsed_seconds"] = elapsed_seconds
                item_state["actual_output_path"] = result.output_path
                item_state["chunks_completed"] = int(result.chunks_completed)
                item_state["chunks_total"] = int(result.chunks_total)
                if result.stopped:
                    item_state["status"] = "stopped"
                    item_state["error"] = result.message
                    batch_state["status"] = "stopped"
                    batch_state["next_index"] = index
                    batch_queue.write_state(files.state_path, batch_state)
                    batch_queue.append_log(files.log_path, "file_stopped", batch_name=batch_name, index=index, source_path=item.source_path, output_path=result.output_path, started_at=item_state["started_at"], ended_at=item_state["ended_at"], elapsed_seconds=elapsed_seconds, chunks_completed=result.chunks_completed, chunks_total=result.chunks_total, message=result.message)
                    batch_queue.append_log(files.log_path, "stopped", batch_name=batch_name, next_index=index)
                    yield self.ui_update(batch_status=status_ui.render_batch_status_html(len(items), completed_files, index + 1, Path(item.source_path).name, "Stopped.", failed_files), start_enabled=True, abort_enabled=False)
                    self._record_process_result(ProcessRunResult(success=False, stopped=True, message="Batch stopped.", chunks_completed=completed_files, chunks_total=len(items)))
                    return
                item_state["status"] = "success" if result.success else "failed"
                item_state["error"] = "" if result.success else result.message
                batch_state["next_index"] = index + 1
                completed_files = self._batch_completed_count(batch_state)
                failed_files = self._batch_failed_count(batch_state)
                batch_queue.write_state(files.state_path, batch_state)
                batch_queue.append_log(files.log_path, "file_finished", batch_name=batch_name, index=index, source_path=item.source_path, output_path=result.output_path, started_at=item_state["started_at"], ended_at=item_state["ended_at"], elapsed_seconds=elapsed_seconds, success=bool(result.success), chunks_completed=result.chunks_completed, chunks_total=result.chunks_total, error="" if result.success else result.message)
                yield self.ui_update(batch_status=status_ui.render_batch_status_html(len(items), completed_files, index + 1, Path(item.source_path).name, "File completed." if result.success else "File failed and was skipped.", failed_files), start_enabled=False, abort_enabled=False)
            batch_state["status"] = "completed"
            batch_state["next_index"] = len(items)
            batch_queue.write_state(files.state_path, batch_state)
            batch_queue.append_log(files.log_path, "completed", batch_name=batch_name, total_files=len(items), completed_files=completed_files, failed_files=failed_files)
            yield self.ui_update(
                status_ui.render_process_status_html("Batch Completed", f"Batch completed. {completed_files} file(s) processed.") if media_kind == "image" else status_ui.render_chunk_status_html(0, 0, 0, "Batch Completed", f"Batch completed. {completed_files} file(s) processed."),
                self.ui_skip,
                self.ui_skip,
                batch_status=status_ui.render_batch_status_html(len(items), completed_files, len(items), "", "Completed.", failed_files),
                start_enabled=True,
                abort_enabled=False,
            )
            self._record_process_result(ProcessRunResult(success=True, message="Batch completed.", chunks_completed=completed_files, chunks_total=len(items)))
            return
        finally:
            self.active_job["batch_running"] = False

    def start_process(self, state=None, process_name="", user_refs=None, source_path="", source_image_path=None, process_strength=None, output_path="", prompt_text="", continue_enabled=True, preview_enabled=True, source_audio_track="", output_resolution="720p", target_ratio="", chunk_size_seconds=10.0, sliding_window_overlap=1, start_seconds="", end_seconds="", batch_mode="single", batch_name="", batch_source_path="", batch_image_source_path=None, spatial_upsampler_parameters=None, batch_internal=False):
        self.active_job[catalog.PREVIEW_OUTPUT_STORAGE_KEY] = catalog.normalize_preview_enabled(preview_enabled)
        if not self._preview_enabled():
            self.preview_state["image"] = None
        process_definition = self.library.process_definition(str(process_name or "").strip(), state, list(user_refs or []))
        selected_source_path = source_path
        selected_batch_source_path = batch_source_path
        if process_definition is not None and self._process_is_image(process_definition):
            selected_source_path = source_image_path if source_image_path is not None else source_path
            selected_batch_source_path = batch_image_source_path if batch_image_source_path is not None else batch_source_path
        if str(batch_mode or "").strip() == "batch" and not batch_internal:
            yield from self.start_batch_process(state, process_name, user_refs, selected_source_path, process_strength, output_path, prompt_text, continue_enabled, self._preview_enabled(), source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, batch_name, selected_batch_source_path, spatial_upsampler_parameters)
            return
        if self.active_job.get("running") or (self.active_job.get("batch_running") and not batch_internal):
            yield self.info_exit("A process is already running.")
            self._record_process_result(ProcessRunResult(source_path=str(selected_source_path or ""), output_path=str(output_path or ""), success=False, message="A process is already running."))
            return
        if not batch_internal:
            self._reset_generated_artifacts()
            self.active_job.update(last_batch=None, batch_fields=None)
        self.active_job["last_process_result"] = None
        request = RunRequest.from_gradio(state, process_name, user_refs, selected_source_path, process_strength, output_path, prompt_text, continue_enabled, source_audio_track, output_resolution, target_ratio, chunk_size_seconds, sliding_window_overlap, start_seconds, end_seconds, spatial_upsampler_parameters)
        if process_definition is None:
            yield self.info_exit(f"Unsupported process: {request.process_name}")
            self._record_process_result(ProcessRunResult(source_path=selected_source_path, output_path=output_path, success=False, message=f"Unsupported process: {request.process_name}"))
            return
        system_handler = self.library.system_handler_for_definition(process_definition)
        if process_definition.get("source") == "user":
            problems = self.library.validate_user_process_definition(process_definition)
            if len(problems) > 0:
                message = self.library.format_user_process_validation_error(process_definition, problems)
                yield self.info_exit(message)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
                return
        process_settings = process_definition["settings"]
        model_type = str(process_settings.get("model_type") or "")
        if len(model_type) == 0 and system_handler is None:
            yield self.info_exit(f"Unsupported process: {request.process_name}")
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=f"Unsupported process: {request.process_name}"))
            return
        process_display_name = str(process_definition.get("name") or request.process_name or "").strip()
        process_is_hdr = False if system_handler is not None else VIDEO_PROMPT_HDR_OUTPUT_FLAG in str(process_settings.get("video_prompt_type") or "")
        is_user_process = process_definition.get("source") == "user"
        has_outpaint_setting = "video_guide_outpainting" in process_settings
        uses_builtin_outpaint_ui = self.library.uses_builtin_outpaint_ui(process_definition)
        user_lora_strength_override_default = self.library.user_lora_strength_override_default(process_definition)
        use_lora_strength_override = system_handler is None and not uses_builtin_outpaint_ui and (not is_user_process or user_lora_strength_override_default is not None)
        process_strength_default = user_lora_strength_override_default if user_lora_strength_override_default is not None else common.get_default_process_strength(process_settings)
        active_process_strength = 1.0 if uses_builtin_outpaint_ui else (common.coerce_float(request.process_strength, process_strength_default) if use_lora_strength_override else process_strength_default)
        source_path = request.source_path
        output_path = request.output_path
        source_audio_track = request.source_audio_track
        output_resolution = request.output_resolution
        target_ratio = request.target_ratio
        prompt_text = request.prompt_text
        start_seconds = request.start_seconds
        end_seconds = request.end_seconds
        if system_handler is not None and callable(getattr(system_handler, "normalize_target_control_for_process", None)):
            system_target_control = system_handler.normalize_target_control_for_process(request.target_ratio, process_settings)
        else:
            system_target_control = system_handler.normalize_target_control(request.target_ratio) if system_handler is not None and hasattr(system_handler, "normalize_target_control") else ""
        if uses_builtin_outpaint_ui:
            target_ratio = self.library.normalize_outpaint_target_ratio(process_settings, target_ratio)
        system_supports_continue_cache = system_handler is not None and (not callable(getattr(system_handler, "supports_continue_cache_for_target", None)) or system_handler.supports_continue_cache_for_target(system_target_control))
        system_crossfades_overlap_outputs = system_handler is not None and bool(getattr(system_handler, "crossfade_overlap_outputs", False))
        effective_continue_enabled = request.continue_enabled and not bool(getattr(system_handler, "disable_continuation", False))
        if self._process_is_image(process_definition):
            yield from self._start_image_process(
                request,
                process_definition,
                system_handler,
                process_display_name=process_display_name,
                active_process_strength=active_process_strength,
                use_lora_strength_override=use_lora_strength_override,
                active_target_ratio=target_ratio,
                system_target_control=system_target_control,
                batch_internal=batch_internal,
                batch_name=batch_name,
                batch_source_path=batch_source_path,
            )
            return
        try:
            chunk_size_seconds = common.require_float(request.chunk_size_seconds, "Chunk Size", minimum=0.1)
            sliding_window_overlap = int(getattr(system_handler, "overlap_frames", 0)) if system_handler is not None else common.require_int(request.sliding_window_overlap, "Sliding Window Overlap", minimum=1)
        except gr.Error as exc:
            message = common.get_error_message(exc) or "Invalid processing settings."
            yield self.info_exit(message)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
            return
        if not batch_internal:
            try:
                self._save_ui_settings({
                    "process_model_type": model_type,
                    "process_name": request.process_name,
                    "source_path": source_path,
                    "batch_source_path": str(batch_source_path or "").strip(),
                    "batch_mode": "single",
                    "media_kind": "video",
                    "batch_name": str(batch_name or "").strip(),
                    "process_strength": active_process_strength,
                    "output_path": output_path,
                    "prompt": prompt_text,
                    "continue_enabled": request.continue_enabled,
                    "source_audio_track": source_audio_track,
                    "output_resolution": output_resolution,
                    "target_ratio": system_target_control if system_handler is not None else target_ratio,
                    "chunk_size_seconds": chunk_size_seconds,
                    "sliding_window_overlap": sliding_window_overlap,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "spatial_upsampler_parameters": request.spatial_upsampler_parameters,
                    catalog.PRESERVE_ARTIFACTS_STORAGE_KEY: self._preserve_artifact_count(),
                    catalog.PREVIEW_OUTPUT_STORAGE_KEY: self._preview_enabled(),
                }, "video", "single")
            except OSError as exc:
                message = f"Unable to save plugin settings to {catalog.MEDIAFLOW_SETTINGS_FILE}: {exc}"
                yield self.info_exit(message)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
                return
        active_target_ratio = target_ratio if uses_builtin_outpaint_ui else ""
        default_prompt_text = str(process_settings.get("prompt") or "")
        if len(prompt_text.strip()) == 0:
            prompt_text = default_prompt_text
        if not os.path.isfile(source_path):
            yield self.info_exit(f"Source video not found: {source_path}")
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=f"Source video not found: {source_path}"))
            return
        try:
            start_seconds = prompts.parse_time_input(start_seconds, label="Start", allow_empty=False)
            end_seconds = prompts.parse_time_input(end_seconds, label="End", allow_empty=True)
        except gr.Error as exc:
            message = common.get_error_message(exc) or "Invalid start/end selection."
            yield self.info_exit(message)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
            return
        try:
            prompt_schedule = prompts.parse_prompt_schedule(prompt_text)
        except gr.Error as exc:
            message = common.get_error_message(exc) or f"Invalid prompt syntax.\n\nExample:\n{prompts.TIMED_PROMPT_EXAMPLE}"
            yield self.info_exit(message)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
            return
        started_ui = False
        preflight_stage = True
        write_state = MuxSession()
        total_chunks_display = 1
        completed_chunks = 0
        resumed_unique_frames = 0
        continuation_output_path = ""
        ffprobe_path = ""
        fps_float = 0.0
        merged_continuation_signatures = []
        verbose_level = 0
        self.active_job["cancel_requested"] = False
        self.active_job["write_state"] = write_state
        self.active_job["running"] = True
        try:
            video.clear_process_full_video_source()
            yield self.ui_update(status_ui.render_chunk_status_html(1, 0, 1, "Initializing", "Preparing processing job..."), self.ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
            started_ui = True
            try:
                prepared_run = prepare_run(
                    plugin=self.plugin,
                    get_model_def=self.get_model_def,
                    process_name=request.process_name,
                    process_display_name=process_display_name,
                    source_path=source_path,
                    output_path=output_path,
                    output_resolution=output_resolution,
                    active_target_ratio=active_target_ratio,
                    continue_enabled=effective_continue_enabled,
                    source_audio_track=source_audio_track,
                    chunk_size_seconds=chunk_size_seconds,
                    sliding_window_overlap=sliding_window_overlap,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    model_type=model_type,
                    process_is_hdr=process_is_hdr,
                    uses_builtin_outpaint_ui=uses_builtin_outpaint_ui,
                    system_handler=system_handler,
                    system_target_control=system_target_control,
                )
                verbose_level = prepared_run.verbose_level
                start_frame = prepared_run.start_frame
                end_frame_exclusive = prepared_run.end_frame_exclusive
                fps_float = prepared_run.fps_float
                total_source_frames = prepared_run.total_source_frames
                processing_fps = prepared_run.processing_fps
                selected_audio_track = prepared_run.selected_audio_track
                frame_plan_rules = prepared_run.frame_plan_rules
                budget_resolution = prepared_run.budget_resolution
                chunk_frames = prepared_run.chunk_frames
                overlap_frames = prepared_run.overlap_frames
                full_plans = prepared_run.full_plans
                requested_unique_frames = prepared_run.requested_unique_frames
                requested_source_segment = prepared_run.requested_source_segment
                output_path = prepared_run.output_path
                resume_existing_output = prepared_run.resume_existing_output
                ffmpeg_path = prepared_run.ffmpeg_path
                ffprobe_path = prepared_run.ffprobe_path
                output_container = prepared_run.output_container
                merged_continuation_signatures = prepared_run.merged_continuation_signatures
            except ProcessInfoExit as exc:
                yield self.info_exit(exc.message, output=exc.output_path or self.ui_skip)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=exc.output_path or output_path, success=False, message=exc.message))
                return
            except gr.Error as exc:
                message = common.get_error_message(exc) or "Invalid process settings."
                yield self.info_exit(message, output=output_path if isinstance(output_path, str) and os.path.isfile(output_path) else self.ui_skip)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, message=message))
                return
            preflight_stage = False
            if resume_existing_output:
                recovery_result = yield from continuation_recovery.recover_residual_continuations(
                    plugin=self.plugin,
                    ffmpeg_path=ffmpeg_path,
                    ffprobe_path=ffprobe_path,
                    output_path=output_path,
                    full_plans=full_plans,
                    output_container=output_container,
                    fps_float=fps_float,
                    selected_audio_track=selected_audio_track,
                    source_path=source_path,
                    start_frame=start_frame,
                    merged_signatures=merged_continuation_signatures,
                    verbose_level=verbose_level,
                    ui_update=self.ui_update,
                    ui_skip=self.ui_skip,
                    system_handler=system_handler,
                    system_supports_continue_cache=system_supports_continue_cache,
                )
                merged_continuation_signatures = recovery_result.merged_signatures
                if recovery_result.blocked:
                    return
                if self.active_job.get("cancel_requested"):
                    write_state.stopped = True
                    yield self.ui_update(status_ui.render_chunk_status_html(len(full_plans) if len(full_plans) > 0 else 1, 0, 1, "Stopped", "Stopped before processing a new chunk."), output_path if os.path.isfile(output_path) else self.ui_skip, self.ui_skip, start_enabled=True, abort_enabled=False)
                    self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, stopped=True, message="Stopped before processing a new chunk.", chunks_completed=0, chunks_total=len(full_plans) if len(full_plans) > 0 else 1))
                    return
            last_frame_image = None
            last_segment_path = None
            chunk_output_paths: list[str] = []
            written_unique_frames = 0
            resumed_unique_frames = 0
            resume_overlap_frames = 0
            completed_chunks = 0
            resolved_resolution = ""
            resolved_width = 0
            resolved_height = 0
            merged_continuation = False
            resume_audio_trim_seconds = 0.0
            continue_cache = None
            self.preview_state["image"] = None
            write_state.set_output_path(output_path)
            exact_start_seconds = start_frame / fps_float
            if resume_existing_output:
                yield self.ui_update(status_ui.render_chunk_status_html(len(full_plans), 0, 1, "Inspecting Existing Output", f"Inspecting existing output to continue: {output_path}"), output_path, str(time.time_ns()))
                process_metadata.log_existing_output_metadata(output_path, verbose_level)
                resumed_unique_frames, resume_reason = media.probe_resume_frame_count(ffprobe_path, output_path, fps_float)
                recorded_written_unique_frames = process_metadata.read_recorded_written_unique_frames(output_path)
                if 0 < recorded_written_unique_frames < resumed_unique_frames:
                    common.plugin_info(f"Output contains {resumed_unique_frames} readable frame(s), but metadata recorded {recorded_written_unique_frames}. Using the real frame count for continuation.")
                    process_metadata.store_process_progress(output_path, written_unique_frames=resumed_unique_frames, merged_signatures=merged_continuation_signatures, verbose_level=verbose_level)
                elif recorded_written_unique_frames > resumed_unique_frames:
                    common.plugin_info(f"Ignoring recorded output progress of {recorded_written_unique_frames} frame(s) because the output only probes as {resumed_unique_frames} frame(s).")
                if resumed_unique_frames < 0:
                    common.plugin_info(f"Ignoring negative probed output progress from {output_path}.")
                    resumed_unique_frames = 0
                elif resumed_unique_frames > requested_unique_frames:
                    common.plugin_info(f"Existing output already covers the requested {requested_unique_frames} frame(s).")
                    resumed_unique_frames = requested_unique_frames
                if resumed_unique_frames <= 0:
                    common.plugin_info(f"Unable to continue from existing output: {output_path}. {resume_reason or 'Starting a new file instead.'}")
                    output_path = output_paths.make_output_variant(Path(output_path), notify=common.plugin_info)
                    write_state.set_output_path(output_path)
                    resumed_unique_frames = 0
                    resume_overlap_frames = 0
                    completed_chunks = 0
                    exact_start_seconds = start_frame / fps_float
                    resume_existing_output = False
                else:
                    common.plugin_info(f"Continuing existing output: {output_path}")
                    resolved_resolution, resolved_width, resolved_height = video.probe_existing_output_resolution(output_path)
                    print(f"[MediaFlow] Continuing with locked output resolution {resolved_resolution}")
                    completed_chunks, _ = (frames.count_completed_written_chunks if system_handler is not None else frames.count_completed_chunks)(full_plans, resumed_unique_frames)
                    exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                    if resumed_unique_frames < requested_unique_frames:
                        resume_phase = "Planning Source Overlap" if system_handler is not None else "Loading Overlap Frames"
                        yield self.ui_update(status_ui.render_chunk_status_html(len(full_plans), 0, 1, resume_phase, f"Continuing existing output with {resumed_unique_frames} frame(s) already written."), output_path, str(time.time_ns()))
                        checked_unique_frames, last_frame_image, tail_reason = video.resolve_resume_last_frame(output_path, resumed_unique_frames)
                        if system_handler is not None:
                            if checked_unique_frames > 0:
                                resumed_unique_frames = checked_unique_frames
                            if tail_reason:
                                common.plugin_info(tail_reason)
                            if self._preview_enabled() and last_frame_image is not None:
                                self.preview_state["image"] = last_frame_image
                            if system_supports_continue_cache and hasattr(system_handler, "load_continue_cache"):
                                sidecar_exists = callable(getattr(system_handler, "cache_sidecar_path", None)) and Path(system_handler.cache_sidecar_path(output_path)).is_file()
                                if sidecar_exists or not callable(getattr(system_handler, "continue_cache_from_tail_frames", None)):
                                    continue_cache = system_handler.load_continue_cache(output_path)
                                else:
                                    fallback_frames = min(int(getattr(system_handler, "overlap_frames", 0)), int(resumed_unique_frames))
                                    fallback_tail = video.load_process_full_video_overlap_buffer(output_path, fallback_frames, resumed_unique_frames)
                                    continue_cache = system_handler.continue_cache_from_tail_frames(fallback_tail, system_target_control)
                                    if continue_cache is None:
                                        continue_cache = system_handler.load_continue_cache(output_path)
                                    common.plugin_info("FlashVSR continuation cache is missing; continuing from decoded output tail frames. The first resumed chunk may have lower temporal continuity.")
                            completed_chunks, _ = frames.count_completed_written_chunks(full_plans, resumed_unique_frames)
                            exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                            resume_overlap_frames = min(overlap_frames, resumed_unique_frames) if system_crossfades_overlap_outputs else 0
                            if resumed_unique_frames < requested_unique_frames:
                                print(f"[MediaFlow] Continuing system process from source frame {start_frame + resumed_unique_frames}" + (" using continue cache" if system_supports_continue_cache else ""))
                        elif checked_unique_frames <= 0 or last_frame_image is None:
                            common.plugin_info(f"Unable to continue from existing output: {output_path}. {tail_reason or 'Starting a new file instead.'}")
                            output_path = output_paths.make_output_variant(Path(output_path), notify=common.plugin_info)
                            write_state.set_output_path(output_path)
                            resumed_unique_frames = 0
                            resume_overlap_frames = 0
                            completed_chunks = 0
                            self.preview_state["image"] = None
                            exact_start_seconds = start_frame / fps_float
                            resume_existing_output = False
                        else:
                            resumed_unique_frames = checked_unique_frames
                            if tail_reason:
                                common.plugin_info(tail_reason)
                            if self._preview_enabled():
                                self.preview_state["image"] = last_frame_image
                            completed_chunks, _ = frames.count_completed_chunks(full_plans, resumed_unique_frames)
                            exact_start_seconds = (start_frame + resumed_unique_frames) / fps_float
                            resume_overlap_frames = overlap_frames
                            if resumed_unique_frames < resume_overlap_frames:
                                resume_overlap_frames = resumed_unique_frames
                            overlap_tensor = video.load_process_full_video_hdr_overlap_buffer(output_path, resume_overlap_frames, resumed_unique_frames) if process_is_hdr else video.load_process_full_video_overlap_buffer(output_path, resume_overlap_frames, resumed_unique_frames)
                            if overlap_tensor is None:
                                common.plugin_info(f"Unable to continue from existing output: {output_path}. Failed to load the overlap frames from the recorded output.")
                                output_path = output_paths.make_output_variant(Path(output_path), notify=common.plugin_info)
                                write_state.set_output_path(output_path)
                                resumed_unique_frames = 0
                                resume_overlap_frames = 0
                                completed_chunks = 0
                                self.preview_state["image"] = None
                                exact_start_seconds = start_frame / fps_float
                                resume_existing_output = False
                            else:
                                resume_overlap_frames = int(overlap_tensor.shape[1])
                                video.set_process_full_video_overlap_buffer(overlap_tensor, processing_fps, hdr=process_is_hdr)
                                print(f"[MediaFlow] Loaded overlap buffer from existing output: {frames.describe_frame_range(start_frame + resumed_unique_frames - resume_overlap_frames, resume_overlap_frames)}")
                    if resume_existing_output and resumed_unique_frames < requested_unique_frames:
                        if not continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE:
                            resume_audio_trim_seconds = media.probe_selected_audio_overhang(ffprobe_path, output_path, selected_audio_track, resumed_unique_frames / fps_float)
                        if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None:
                            print(f"[MediaFlow] Final merge: rebuilding audio from source starting at {start_frame / fps_float:.6f}s")
                        elif resume_audio_trim_seconds > 0.0:
                            print(f"[MediaFlow] Final merge: trimming {resume_audio_trim_seconds:.6f}s from continuation audio and clamping segment audio to visible video duration")
                        remaining_resume_unique_frames = frames.align_total_unique_frames(
                            end_frame_exclusive - (start_frame + resumed_unique_frames),
                            frame_step=frame_plan_rules.frame_step,
                            minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                            initial_overlap_frames=resume_overlap_frames,
                            frame_offset=frame_plan_rules.frame_offset,
                        )
                        if remaining_resume_unique_frames <= 0:
                            trailing_frames = requested_unique_frames - resumed_unique_frames
                            common.plugin_info(f"Existing output has {resumed_unique_frames} frame(s). The remaining {trailing_frames} frame(s) are too short to build another continuation chunk for the current model, so the existing output is treated as complete.")
                            resumed_unique_frames = requested_unique_frames
                            completed_chunks = len(full_plans)
                            plans = []
                        else:
                            continuation_output_path = output_paths.make_continuation_output_path(output_path)
                            write_state.set_output_path(continuation_output_path)
                            try:
                                plans = frames.build_chunk_plan(
                                    start_frame + resumed_unique_frames,
                                    end_frame_exclusive,
                                    total_source_frames,
                                    chunk_frames,
                                    frame_step=frame_plan_rules.frame_step,
                                    minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
                                    frame_offset=frame_plan_rules.frame_offset,
                                    overlap_frames=overlap_frames,
                                    initial_overlap_frames=resume_overlap_frames,
                                )
                            except frames.FramePlanningError as exc:
                                raise gr.Error(str(exc)) from exc
                    elif resume_existing_output:
                        plans = []
            if not resume_existing_output:
                plans = full_plans
            continued_mode = resumed_unique_frames > 0
            use_live_av_mux = selected_audio_track is not None
            total_chunks_display = completed_chunks + len(plans)
            current_chunk_display = completed_chunks + 1
            run_started_at = time.time()
            initial_completed_chunks = completed_chunks

            def _timing_kwargs(current_completed_chunks=None, phase_current_step=None, phase_total_steps=None):
                elapsed_seconds = time.time() - run_started_at
                if elapsed_seconds < 0.0:
                    elapsed_seconds = 0.0
                if len(plans) == 0:
                    return {"elapsed_seconds": elapsed_seconds, "eta_seconds": None}
                completed_for_eta = completed_chunks if current_completed_chunks is None else current_completed_chunks
                run_completed_chunks = completed_for_eta - initial_completed_chunks
                phase_ratio = 0.0
                if phase_current_step is not None and phase_total_steps is not None and phase_total_steps > 0:
                    phase_ratio = float(phase_current_step) / float(phase_total_steps)
                overall_ratio = (run_completed_chunks + phase_ratio) / float(len(plans))
                eta_seconds = None if overall_ratio <= 0.0 or overall_ratio >= 1.0 else elapsed_seconds * (1.0 - overall_ratio) / overall_ratio
                return {"elapsed_seconds": elapsed_seconds, "eta_seconds": eta_seconds}

            if len(plans) == 0:
                if system_handler is not None and callable(getattr(system_handler, "delete_continue_cache", None)):
                    system_handler.delete_continue_cache(output_path)
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, completed_chunks, "Completed", "Existing output already covers the requested range.", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=True, message="Existing output already covers the requested range.", chunks_completed=completed_chunks, chunks_total=total_chunks_display))
                return
            planning_text = f"Resuming from {resumed_unique_frames} frame(s) already written." if resumed_unique_frames > 0 else f"Preparing {len(plans)} chunk(s)..."
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Planning", planning_text, continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()))

            write_state.set_initial_metadata({
                "video_length": int(resumed_unique_frames),
                "frame_count": int(resumed_unique_frames),
                process_metadata.PROCESS_FULL_VIDEO_METADATA_KEY: {
                    "process": process_display_name,
                    "written_unique_frames": int(resumed_unique_frames),
                    "chunks": int(total_chunks_display),
                    "sliding_window_overlap": int(overlap_frames),
                    "start_seconds": float(start_seconds),
                    "end_seconds": float(start_seconds + (resumed_unique_frames / float(fps_float))),
                    "source_video": source_path,
                    "source_segment": requested_source_segment,
                    "merged_continuations": process_metadata.normalize_merged_continuation_signatures(merged_continuation_signatures),
                },
            })

            chunk_progress = ChunkProgress(
                completed_chunks=completed_chunks,
                current_chunk_display=current_chunk_display,
                chunk_output_paths=chunk_output_paths,
                last_segment_path=last_segment_path,
                write_state=write_state,
                resolved_resolution=resolved_resolution,
                resolved_width=resolved_width,
                resolved_height=resolved_height,
                continue_cache=continue_cache,
            )
            continue_cache = None
            chunk_result = yield from ChunkExecutor(
                plugin=self.plugin,
                api_session=self.api_session,
                active_job=self.active_job,
                preview_state=self.preview_state,
                ui_update=self.ui_update,
                ui_skip=self.ui_skip,
                reset_live_chunk_status=self.reset_live_chunk_status,
            ).run(ProcessContext(
                state=request.state,
                process_settings=process_definition["settings"],
                model_type=model_type,
                process_is_hdr=process_is_hdr,
                is_user_process=is_user_process,
                has_outpaint_setting=has_outpaint_setting,
                uses_builtin_outpaint_ui=uses_builtin_outpaint_ui,
                use_lora_strength_override=use_lora_strength_override,
                active_process_strength=active_process_strength,
                active_target_ratio=active_target_ratio,
                source_path=source_path,
                output_path=output_path,
                selected_audio_track=selected_audio_track,
                prompt_schedule=prompt_schedule,
                default_prompt_text=default_prompt_text,
                budget_resolution=budget_resolution,
                start_frame=start_frame,
                resumed_unique_frames=resumed_unique_frames,
                requested_unique_frames=requested_unique_frames,
                overlap_frames=overlap_frames,
                processing_fps=processing_fps,
                fps_float=fps_float,
                continued_mode=continued_mode,
                plans=plans,
                total_chunks_display=total_chunks_display,
                ffmpeg_path=ffmpeg_path,
                use_live_av_mux=use_live_av_mux,
                output_container=output_container,
                exact_start_seconds=exact_start_seconds,
                timing_kwargs=_timing_kwargs,
                system_handler=system_handler,
                system_target_control=system_target_control,
                spatial_upsampler_parameters=request.spatial_upsampler_parameters,
            ), chunk_progress)
            written_unique_frames = chunk_result.written_unique_frames
            completed_chunks = chunk_result.completed_chunks
            current_chunk_display = chunk_result.current_chunk_display
            resolved_resolution = chunk_result.resolved_resolution
            resolved_width = chunk_result.resolved_width
            resolved_height = chunk_result.resolved_height
            last_segment_path = chunk_result.last_segment_path
            chunk_output_paths = chunk_result.chunk_output_paths
            continue_cache = chunk_result.continue_cache
            if write_state.mux_process is None:
                if write_state.stopped and resumed_unique_frames > 0:
                    common.plugin_info(f"Processing was stopped before writing a new chunk. Kept existing output at {output_path}")
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Stopped", "Stopped before a new chunk was written. Existing output kept.", continued=continued_mode, **_timing_kwargs()), output_path, self.ui_skip, start_enabled=True, abort_enabled=False)
                    self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, stopped=True, message="Stopped before a new chunk was written. Existing output kept.", chunks_completed=completed_chunks, chunks_total=total_chunks_display))
                    return
                if write_state.stopped:
                    common.plugin_info("Processing was stopped before any output chunk was written.")
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Stopped", "Stopped before any output chunk was written.", continued=continued_mode, **_timing_kwargs()), self.ui_skip, self.ui_skip, start_enabled=True, abort_enabled=False)
                    self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=False, stopped=True, message="Stopped before any output chunk was written.", chunks_completed=completed_chunks, chunks_total=total_chunks_display))
                    return
                raise gr.Error("Processing completed without creating an output file.")
            if self.active_job.get("cancel_requested"):
                write_state.stopped = True
            finalizing_message = "Finalizing written output before merge..." if continuation_output_path and os.path.isfile(write_state.mux_output_path) else "Finalizing written output..."
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Finalizing Output", finalizing_message, continued=continued_mode, **_timing_kwargs()), output_path if os.path.isfile(output_path) else self.ui_skip, str(time.time_ns()), start_enabled=False, abort_enabled=False)
            finalize_label = "continuation segment" if continuation_output_path and os.path.isfile(write_state.mux_output_path) else "output"
            return_code, stderr, forced_termination = write_state.finalize(label=finalize_label)
            if self.active_job.get("cancel_requested"):
                write_state.stopped = True
            if forced_termination:
                raise gr.Error("ffmpeg did not finalize the partial output in time.")
            if return_code != 0 and not (write_state.stopped and os.path.isfile(write_state.mux_output_path)):
                raise gr.Error(stderr or "ffmpeg failed while assembling the processed video.")
            if os.path.isfile(write_state.mux_output_path) and os.path.getsize(write_state.mux_output_path) <= 0:
                media.delete_file_if_exists(write_state.mux_output_path, label="empty working output")
                raise gr.Error("ffmpeg created an empty continuation file.")
            if os.path.isfile(write_state.mux_output_path):
                requested_write_path = write_state.output_path_for_write
                cache_source_path, promoted_path = write_state.promote_output(notify=common.plugin_info)
                if cache_source_path != promoted_path and system_handler is not None and callable(getattr(system_handler, "move_continue_cache", None)):
                    system_handler.move_continue_cache(cache_source_path, promoted_path)
                if continuation_output_path:
                    continuation_output_path = promoted_path
                elif requested_write_path == output_path:
                    output_path = promoted_path
            undeleted_merged_continuation_paths: list[str] = []
            if continuation_output_path and os.path.isfile(write_state.output_path_for_write):
                continuation_signature = process_metadata.make_continuation_signature(write_state.output_path_for_write)
                try:
                    existing_output_generation_time = process_metadata.read_metadata_generation_time(output_path)
                    merged_duration_seconds = float(resumed_unique_frames + written_unique_frames) / fps_float
                    committed_signatures = process_metadata.append_merged_continuation_signature(merged_continuation_signatures, continuation_signature)
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Merging Continuation", "Merging the continued segment into the main output...", continued=continued_mode, **_timing_kwargs()), output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
                    media.concat_video_segments(
                        ffmpeg_path,
                        [output_path, write_state.output_path_for_write],
                        output_path,
                        self.plugin.server_config.get("video_output_codec", "libx264_8"),
                        output_container,
                        self.plugin.server_config.get("audio_output_codec", "aac_128"),
                        segment_audio_trim_seconds=[0.0, resume_audio_trim_seconds],
                        segment_audio_duration_seconds=[(float(resumed_unique_frames) / fps_float) if resumed_unique_frames > 0 else None, (float(written_unique_frames) / fps_float) if written_unique_frames > 0 else None],
                        fps_float=fps_float,
                        selected_audio_track_no=selected_audio_track,
                        reserved_metadata_path=write_state.reserved_metadata_path,
                        source_audio_path=source_path if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        source_audio_start_seconds=(start_frame / fps_float) if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        source_audio_duration_seconds=merged_duration_seconds if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                        source_audio_track_no=selected_audio_track if continuation_recovery.USE_SOURCE_AUDIO_FOR_CONTINUATION_MERGE and selected_audio_track is not None else None,
                    )
                    merged_continuation = True
                    merged_continuation_signatures = committed_signatures
                    process_metadata.store_process_progress(output_path, written_unique_frames=resumed_unique_frames + written_unique_frames, merged_signatures=merged_continuation_signatures, verbose_level=verbose_level)
                except media.ContinuationMergeOutputLockedError:
                    locked_message = f"{Path(output_path).name} is open, so the continuation merge could not replace it. Existing output was kept and {Path(write_state.output_path_for_write).name} was preserved. Release the base file and start a process again."
                    gr.Info(locked_message)
                    media.delete_file_if_exists(write_state.reserved_metadata_path, label="reserved metadata file")
                    yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Merge Pending", locked_message, continued=continued_mode, **_timing_kwargs()), write_state.output_path_for_write, str(time.time_ns()), start_enabled=True, abort_enabled=False)
                    self._record_process_result(ProcessRunResult(source_path=source_path, output_path=write_state.output_path_for_write, success=False, message=locked_message, chunks_completed=completed_chunks, chunks_total=total_chunks_display))
                    return
                except Exception as exc:
                    raise gr.Error(f"Failed to finalize continued output. Existing output kept, and continuation was preserved at {continuation_output_path}. {exc}") from exc
                if os.path.isfile(write_state.output_path_for_write):
                    try:
                        os.remove(write_state.output_path_for_write)
                        if system_handler is not None and callable(getattr(system_handler, "delete_continue_cache", None)):
                            system_handler.delete_continue_cache(write_state.output_path_for_write)
                    except OSError:
                        undeleted_merged_continuation_paths.append(write_state.output_path_for_write)
                        common.plugin_info(f"Merged continuation progress into {Path(output_path).name}, but {Path(write_state.output_path_for_write).name} could not be deleted because it is still open. Delete it manually when released.")
            else:
                existing_output_generation_time = 0.0
            media.delete_file_if_exists(write_state.reserved_metadata_path, label="reserved metadata file")
            total_written_unique_frames = resumed_unique_frames + written_unique_frames
            if not write_state.stopped and total_written_unique_frames < requested_unique_frames:
                raise gr.Error(f"Processing wrote {total_written_unique_frames} frame(s), but {requested_unique_frames} frame(s) were required.")
            metadata_target_path = output_path if merged_continuation or not continuation_output_path else write_state.output_path_for_write
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Writing Metadata", "Writing final output metadata...", continued=continued_mode, **_timing_kwargs()), metadata_target_path if os.path.isfile(metadata_target_path) else output_path, str(time.time_ns()), start_enabled=False, abort_enabled=False)
            metadata_source_path = last_segment_path or media.get_last_generated_video_path(chunk_output_paths) or metadata_target_path
            actual_output_frames = media.probe_resume_frame_count(ffprobe_path, metadata_target_path, fps_float)[0] if os.path.isfile(metadata_target_path) else 0
            if actual_output_frames <= 0:
                if metadata_target_path != output_path or not resume_existing_output:
                    media.delete_file_if_exists(metadata_target_path, label="invalid output")
                raise gr.Error("Final output does not contain a readable video frame.")
            if actual_output_frames < total_written_unique_frames:
                if write_state.stopped:
                    common.plugin_info(f"Stopped output contains {actual_output_frames} readable frame(s), lower than the {total_written_unique_frames} frame(s) attempted. Recording the probed frame count.")
                    total_written_unique_frames = actual_output_frames
                else:
                    raise gr.Error(f"Final output contains {actual_output_frames} readable frame(s), but {total_written_unique_frames} frame(s) were written.")
            total_generation_time = existing_output_generation_time + process_metadata.read_metadata_generation_time(metadata_source_path) if merged_continuation else process_metadata.read_metadata_generation_time(metadata_source_path)
            output_process_metadata = {
                "process": process_display_name,
                "written_unique_frames": int(total_written_unique_frames),
                "chunks": int(total_chunks_display),
                "sliding_window_overlap": int(overlap_frames),
                "start_seconds": float(start_seconds),
                "end_seconds": float(start_seconds + (total_written_unique_frames / float(fps_float))),
                "source_video": source_path,
                "source_segment": requested_source_segment,
                "merged_continuations": process_metadata.normalize_merged_continuation_signatures(merged_continuation_signatures),
            }
            if process_is_hdr:
                output_process_metadata["hdr"] = True
            metadata_written = process_metadata.store_output_metadata(metadata_target_path, metadata_source_path, source_path=source_path, process_name=process_display_name, source_start_seconds=start_seconds, start_frame=start_frame, fps_float=fps_float, selected_audio_track=selected_audio_track, total_generation_time=total_generation_time, actual_frame_count=actual_output_frames, source_frame_count=total_written_unique_frames, process_metadata=output_process_metadata, verbose_level=verbose_level)
            completed_output = not write_state.stopped and (total_written_unique_frames >= requested_unique_frames or completed_chunks >= total_chunks_display)
            if system_handler is not None and completed_output and callable(getattr(system_handler, "delete_continue_cache", None)):
                for cache_output_path in dict.fromkeys([metadata_target_path, output_path, write_state.output_path_for_write]):
                    if cache_output_path:
                        system_handler.delete_continue_cache(cache_output_path)
                continue_cache = None
            elif system_handler is not None and continue_cache is not None and hasattr(system_handler, "save_continue_cache"):
                system_handler.save_continue_cache(continue_cache, metadata_target_path, metadata=output_process_metadata)
            if not metadata_written:
                raise gr.Error(f"Failed to write WanGP metadata to {metadata_target_path}. The partial output was kept, but resuming the process may require the continuation cache file.")
            kept_artifact_paths, gallery_refresh = self._prune_generated_artifacts(request.state)
            kept_artifact_set = set(kept_artifact_paths)
            chunk_output_paths = [path for path in chunk_output_paths if str(Path(path).resolve()) in kept_artifact_set]
            if write_state.stopped:
                stopped_output_path = output_path
                if merged_continuation:
                    common.plugin_info(f"Processing was stopped. Merged continued progress into {output_path}")
                    stop_message = f"Stopped after {total_written_unique_frames} frame(s). Continued progress was merged into the output."
                elif continuation_output_path and os.path.isfile(write_state.output_path_for_write):
                    stopped_output_path = write_state.output_path_for_write
                    common.plugin_info(f"Processing was stopped. Kept existing output at {output_path} and preserved continuation clip at {write_state.output_path_for_write}")
                    stop_message = f"Stopped after {total_written_unique_frames} frame(s). Existing output kept and continuation clip preserved."
                else:
                    common.plugin_info(f"Processing was stopped. Kept partial output at {output_path}")
                    stop_message = f"Stopped after {total_written_unique_frames} frame(s). Partial output kept."
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, completed_chunks, current_chunk_display, "Stopped", stop_message, continued=continued_mode, **_timing_kwargs()), stopped_output_path, self.ui_skip, start_enabled=True, abort_enabled=False, gallery_refresh=gallery_refresh)
                self._record_process_result(ProcessRunResult(source_path=source_path, output_path=stopped_output_path, success=False, stopped=True, message=stop_message, chunks_completed=completed_chunks, chunks_total=total_chunks_display))
                return
            yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_display, total_chunks_display, total_chunks_display, "Completed", f"Completed {total_chunks_display} chunk(s).", continued=continued_mode, **_timing_kwargs()), output_path, self.ui_skip, start_enabled=True, abort_enabled=False, gallery_refresh=gallery_refresh)
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=output_path, success=True, message=f"Completed {total_chunks_display} chunk(s).", chunks_completed=total_chunks_display, chunks_total=total_chunks_display))
            self.active_job["job"] = None
            write_state.cleanup_partial_outputs()
        except gr.Error as exc:
            self.active_job["job"] = None
            preserved_output_path = self._preserve_failed_mux_output(write_state, system_handler)
            if preserved_output_path and not continuation_output_path:
                recovered_frame_count, _ = media.probe_resume_frame_count(ffprobe_path, preserved_output_path, fps_float)
                if recovered_frame_count > 0:
                    process_metadata.store_process_progress(preserved_output_path, written_unique_frames=recovered_frame_count, merged_signatures=merged_continuation_signatures, verbose_level=verbose_level)
            status_message = common.get_error_message(exc) or "Processing failed."
            if preserved_output_path:
                status_message = f"{status_message} Partial output preserved at {preserved_output_path}"
            if not started_ui:
                gr.Info(status_message)
                return
            if started_ui:
                total_chunks_value = total_chunks_display
                completed_value = completed_chunks
                current_value = completed_chunks + 1 if completed_chunks < total_chunks_display else total_chunks_display
                continued_value = resumed_unique_frames > 0
                output_value = preserved_output_path or (output_path if isinstance(output_path, str) and os.path.isfile(output_path) else self.ui_skip)
                if preflight_stage:
                    gr.Info(status_message)
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_value, completed_value, current_value, "Info" if preflight_stage else "Error", status_message, continued=continued_value), output_value, self.ui_skip, start_enabled=True, abort_enabled=False)
            failed_output_path = preserved_output_path or (output_path if isinstance(output_path, str) else "")
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=failed_output_path, success=False, message=status_message, chunks_completed=completed_chunks, chunks_total=total_chunks_display))
            return
        except BaseException as exc:
            self.active_job["job"] = None
            preserved_output_path = self._preserve_failed_mux_output(write_state, system_handler)
            if preserved_output_path and not continuation_output_path:
                recovered_frame_count, _ = media.probe_resume_frame_count(ffprobe_path, preserved_output_path, fps_float)
                if recovered_frame_count > 0:
                    process_metadata.store_process_progress(preserved_output_path, written_unique_frames=recovered_frame_count, merged_signatures=merged_continuation_signatures, verbose_level=verbose_level)
            status_message = common.get_error_message(exc) or exc.__class__.__name__
            if preserved_output_path:
                status_message = f"{status_message} Partial output preserved at {preserved_output_path}"
            if started_ui:
                total_chunks_value = total_chunks_display
                completed_value = completed_chunks
                current_value = completed_chunks + 1 if completed_chunks < total_chunks_display else total_chunks_display
                continued_value = resumed_unique_frames > 0
                output_value = preserved_output_path or (output_path if isinstance(output_path, str) and os.path.isfile(output_path) else self.ui_skip)
                yield self.ui_update(status_ui.render_chunk_status_html(total_chunks_value, completed_value, current_value, "Error", status_message, continued=continued_value), output_value, self.ui_skip, start_enabled=True, abort_enabled=False)
            failed_output_path = preserved_output_path or (output_path if isinstance(output_path, str) else "")
            self._record_process_result(ProcessRunResult(source_path=source_path, output_path=failed_output_path, success=False, message=status_message, chunks_completed=completed_chunks, chunks_total=total_chunks_display))
            raise
        finally:
            self.active_job["running"] = False
            self.active_job["write_state"] = None
            video.clear_process_full_video_source()

