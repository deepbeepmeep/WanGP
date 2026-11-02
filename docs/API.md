# WanGP Python API

`shared/api.py` provides a lightweight in-process wrapper over WanGP's existing generation path.

The main goal is to let third-party code call WanGP directly, keep the last loaded model alive across requests, receive structured progress updates, and still capture the same stdout/stderr output that would normally go to the console.

This same API can be used directly from a WanGP Plugin and uses WanGP Web Queue to process the requested Jobs of called from a Third Party App (for instance https://github.com/deepbeepmeep/LTX-Desktop-WanGP).

**Please note that use of the WanGP API is subject to the WanGP Terms and Conditions. Any product that integrates WanGP should clearly disclose that it uses WanGP in both its user interface and its documentation.**

## Quick Start

The WanGP API consumes mostly *WanGP Settings*. In order to get the format of the settings, just launch the Web interface of WanGP, pick the model you want to use and fill the UI with settings you want to use, then click the *Export Settings* button at the bottom.

```python
from pathlib import Path

from shared.api import init

session = init(
    root=Path(r"C:\WanGP"),
    cli_args=["--attention", "sdpa", "--profile", "4"],
)

settings = {
    "model_type": "ltx2_22B_distilled",
    "prompt": "Cinematic shot of a neon train entering a rainy station",
    "resolution": "1280x704",
    "num_inference_steps": 8,
    "video_length": 97,
    "duration_seconds": 4,
    "force_fps": 24,
}

job = session.submit_task(settings)

for event in job.events.iter(timeout=0.2):
    if event.kind == "progress":
        progress = event.data
        print(progress.phase, progress.progress, progress.current_step, progress.total_steps)
    elif event.kind == "preview":
        preview = event.data
        if preview.image is not None:
            preview.image.save("preview.png")
    elif event.kind == "stream":
        line = event.data
        print(f"[{line.stream}] {line.text}")

result = job.result()
if result.success:
    print(result.generated_files)
else:
    for error in result.errors:
        print(error.message)
```

To use existing LoRA collections, add `"--lora-config", "C:/WanGP/lora_paths.json"` to `cli_args` when initializing the session. This uses the same folder keys and overrides as the web app; see the [LoRA path configuration guide](LORAS.md#custom-lora-directories).

## Main Entry Points

- `init(...) -> WanGPSession`
  - Creates a reusable session and eagerly loads the runtime.
  - This is the normal entrypoint for standalone / CLI-style third-party integrations.
- `WanGPSession.submit(source, callbacks=None) -> SessionJob`
  - Starts a job from a settings dict, a manifest list, or a saved `.json` / `.zip` file.
- `WanGPSession.submit_task(settings, callbacks=None) -> SessionJob`
  - Preferred single-task entrypoint.
- `WanGPSession.submit_manifest(settings_list, callbacks=None) -> SessionJob`
  - Batch entrypoint for multiple tasks.
- `WanGPSession.submit_media_postprocessing(media_source, ...) -> SessionJob`
  - Late postprocess an existing image/video with spatial upsampling, temporal upsampling, or film grain.
- `WanGPSession.submit_audio_remux(video_source, postprocess_audio=..., ...) -> SessionJob`
  - Late remux an existing video with a registered soundtrack or voice-replacement audio processor.
- `WanGPSession.submit_audio_postprocessing(audio_source, postprocess_audio=..., ...) -> SessionJob`
  - Late postprocess an existing audio file.
- `WanGPSession.list_model_defs(...) -> list[dict]`
  - Returns model definitions with inferred metadata and optional filters.
- `WanGPSession.list_model_metadata(...) -> list[dict]`
  - Returns compact inferred metadata records with the same filters.
  - Pass `include_availability=True` to add local file availability; this performs the same filesystem scan as the UI status squares.
- `WanGPSession.get_default_settings(model_type) -> dict`
  - Returns pristine defaults generated from WanGP, the model handler, and the model definition, with `model_type` included. User-saved UI defaults are not included. Deepy/MCP responses omit fixed `settings_version`, display-only `type` metadata and the UI's `prompt_enhancer` choice.
- `WanGPSession.get_model_schema(model_type) -> dict | None`
  - Returns compact capability, media-role, frame-limit, prompt-guidance, and sliding-window metadata.
- `WanGPSession.get_model_availability(model_type) -> dict`
  - Returns local file availability for one model using the same status as the UI selector.
- `WanGPSession.list_model_availability(...) -> list[dict]`
  - Returns availability records with the same filters as `list_model_metadata(...)`.
- `SessionJob.result() -> GenerationResult`
  - Waits for completion and returns a structured result object.
- `SessionJob.cancel()`
  - Requests cancellation of the active generation.

### When `init(...)` Is Needed

- Standalone / CLI-style third-party app or Python script:
  - yes, call `init(...)`
- WanGP Plugin tab receiving an injected `api_session` in `create_config_ui(...)`:
  - no, do not call `init(...)`
  - use the injected `api_session` directly
- Low-level in-process integration inside WanGP with direct access to the live Gradio `state`:
  - possible with `init(webui_state=state, ...)`
  - this is an advanced/internal path, not the normal Plugin entrypoint

## `init(...)` Parameters

```python
session = init(
    root=Path(r"C:\WanGP"),
    config_path=Path(r"C:\WanGP\wgp_config.json"),  # optional
    output_dir=Path(r"C:\WanGP\outputs_override"),  # optional
    callbacks=MyCallbacks(),                        # optional
    cli_args=["--attention", "sdpa"],              # optional
    console_output=True,                           # optional, default=True
    console_isatty=True,                           # optional, default=True
    webui_state=None,                              # optional
)
```

- `root`
  - Path to the WanGP installation folder.
  - Example: `C:\WanGP`

- `config_path`
  - Optional path to `wgp_config.json`.
  - If omitted, WanGP uses `C:\WanGP\wgp_config.json`.
  - This must point to a file named `wgp_config.json`.

- `output_dir`
  - Optional override for generated outputs.
  - If omitted, WanGP uses the output paths defined in the config file.

- `callbacks`
  - Optional callback object. See the callback section below.

- `cli_args`
  - Optional WanGP startup flags.
  - Example: `["--attention", "sdpa", "--profile", "4"]`

- `console_output`
  - Enables or disables writing WanGP stdout/stderr to the real console.
  - Default: `True`
  - The stream object always receives a copy of stdout/stderr, regardless of this setting.

- `console_isatty`
  - Controls the TTY capability reported by the API's console capture wrapper.
  - Default: `True`
  - Keep this enabled if you want tqdm or other terminal-style progress output to behave like a live console stream even when WanGP is called from another Python process.

- `webui_state`
  - Optional live WanGP WebUI state dictionary.
  - When provided, submitted tasks target the existing WanGP Gradio queue instead of calling the headless generation path directly.
  - This is mainly a low-level internal / advanced integration hook.
  - Normal WanGP Plugins should use the injected `api_session` instead of calling `init(...)` themselves.
  - In this mode, `output_dir` is ignored and `on_stream(...)` is not expected because the WebUI queue path is probe-based rather than stdout-capture based.

### WebUI Queue Mode

This is the WebUI-backed queue mode used by WanGP Plugin integrations.

If you are implementing a low-level in-process integration and explicitly need to bind the API to WanGP's live Gradio queue yourself, pass the current `state` dict to `init(...)`:

```python
from shared.api import init

session = init(webui_state=state, console_output=False)
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "A cinematic rainy street at night",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
    }
)
```

This mode reuses WanGP's existing WebUI queue loader server-side, then probes the live queue by `client_id` until the task is admitted, completed, fails, or is cancelled.

Differences versus standalone / CLI-style mode:

- third-party apps normally use plain `init(...)` without `webui_state`
- this WebUI-backed mode is the WanGP-side queue path
- low-level direct use still calls `init(...)`, but with `webui_state=state`
- jobs run through WanGP's existing Gradio queue instead of the headless direct generation path
- `output_dir` is ignored
- `on_stream(...)` / `job.events` console lines are not the primary signal because this path is probe-based
- the API banner is intentionally not printed in this mode

### Plugin Tab API Object

For WanGP Plugins, a WebUI-backed session can be injected directly into the tab constructor.

If a Plugin tab constructor accepts one extra positional argument, WanGP now passes a session-shaped API object there:

```python
def create_config_ui(self, api_session):
    ...
```

That object behaves like a normal `WanGPSession`. In a WanGP Plugin tab, the WebUI queue integration stays inside the injected session object, so Plugin code can keep using the same `submit_task(...)`, `job.result()`, and `job.cancel()` pattern as CLI code.

Plugin-specific difference:

- in a Plugin tab, you normally do **not** call `init(...)`
- WanGP creates and injects the WebUI-backed session object for you
- Plugin code should treat `api_session` as the API entrypoint
- this is the normal WebUI/API path for WanGP Plugins

Minimal Plugin pattern:

```python
def create_config_ui(self, api_session):
    active_job = {"job": None}

    def start_demo(progress=gr.Progress(track_tqdm=False)):
        class DemoCallbacks:
            ratio = 0.0

            def on_status(self, status):
                status = str(status or "").strip()
                if status:
                    progress(self.ratio, desc=status)

            def on_progress(self, update):
                self.ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
                progress(self.ratio, desc=str(getattr(update, "status", "") or "Generating..."))

        job = api_session.submit_task(settings, callbacks=DemoCallbacks())
        active_job["job"] = job
        result = job.result()
        return result.generated_files[0] if result.success and result.generated_files else gr.update()

    def cancel_demo():
        job = active_job.get("job")
        if job is not None and not job.done:
            job.cancel()
```

In a Plugin tab, callback methods like `on_status(...)` and `on_progress(...)` can safely update a local `gr.Progress(...)` while the job itself still runs through WanGP's main WebUI queue.

## Model Metadata

Use `session.list_model_defs(...)` to inspect available model definitions without starting a generation:

```python
image_models = session.list_model_defs(main_output="image")
video_i2v_models = session.list_model_defs(main_output="video", inputs="image")
ltx2_finetunes = session.list_model_defs(family="ltx2", finetune=True)
one_model = session.list_model_defs(model_type="ltx2_22B_distilled")
krea_models = session.list_model_metadata(query="Krea 2", main_output="image", limit=10)
resolution_matches = session.list_model_metadata(main_output="image", resolution="2048x2048")
turbo_models = session.list_model_metadata(model_type="*_turbo*", limit=20)
```

Agents that only need discovery data should use the compact metadata calls:

```python
models = session.list_model_metadata(main_output=["image", "video"], inputs="image")
metadata = session.get_model_metadata("ltx2_22B_distilled")
defaults = session.get_default_settings("ltx2_22B_distilled")
schema = session.get_model_schema("ltx2_22B_distilled")
```

Supported filters can be combined:

- `family`
- `base_model_type`
- `finetune`
- `model_type`
- `main_output`
- `inputs`
- `name`
- `query`
- `limit`
- `offset`

All string filters accept case-insensitive `*` and `?` glob patterns. `query` is a case-insensitive substring search across the user-facing name, model type, architecture, family id/label, and description. `main_output` and `inputs` accept a single value or a list. Multiple `main_output` values are OR filters; multiple `inputs` values match any requested input modality. `main_output` filters the primary generation mode, while `metadata.outputs` lists every intrinsic media output the model can produce. `limit` and `offset` provide bounded pagination; omit `limit` in the Python API when the complete filtered result is intentionally required.

Each returned model definition is a copy of WanGP's in-memory definition with `model_type` added at the top level and a `metadata` object:

```python
{
    "model_type": "ltx2_22B_distilled",
    "name": "LTX-2 2.3 Distilled 1.0 22B",
    "architecture": "ltx2_22B",
    "metadata": {
        "model_type": "ltx2_22B_distilled",
        "family": "ltx2",
        "family_label": "LTX-2",
        "base_model_type": "ltx2_22B",
        "finetune": False,
        "main_output": ["video"],
        "outputs": ["video", "audio"],
        "inputs": ["text", "image", "video", "audio"],
        "media_inputs": {
            "image": {
                "start": True,
                "end": True,
                "reference": True,
                "background": False,
                "injected_frames": False,
            },
            "video": {"continue": True, "control": True},
            "audio": {"prompt": True, "output": True},
        },
        "capabilities": {
            "text_to_video": True,
            "image_to_video": True,
            "audio_output": True,
        },
        "setting_values": {
            "image_prompt_type": {"allowed": "TVSE"},
            "video_prompt_type": {"image_ref_choices": {"choices": []}},
        },
    },
}
```

`session.get_model_schema(model_type)` returns one compact `metadata` object with generic capabilities and media roles plus `name`, `model_type`, `description`, FPS/frame limits, `infos`, `prompt_infos`, and sliding-window support. Call `session.get_default_settings(model_type)` separately when generation settings are needed.

At task normalization, WanGP fills missing media-mode flags when the supplied model definition allows them: start images add `S`, end images add `E` without removing `S`, source-video continuation adds `V`, reference images use the first declared reference mode containing `I` (for example `I` or `KI`), and audio guides use declared `A`/`B` source modes. Explicit compatible flags remain unchanged.

API and MCP tasks may provide `video_length` as a seconds string such as `"10s"`. WanGP uses a numeric `force_fps` when supplied, otherwise the model FPS, and converts the duration to a frame count allowed by the model's frame minimum, step, and offset. Frame counts, given in frames or seconds, are rounded up to the next allowed count, so a video is never shorter than requested: a speech or song driving it keeps its last words.

Useful `SessionJob` handles for Plugin authors:

- `job.result(timeout=None) -> GenerationResult`
  - Waits for completion and returns generated files, errors, and any requested returned media artifacts.
- `job.cancel()`
  - Requests cancellation of the current job.
- `job.done`
  - `True` once the job has fully finished.
- `job.events`
  - Event stream for progress, preview, status, output, and completion events.
- `job.cancel_requested`
  - `True` after cancellation was requested.

Useful `GenerationResult` fields:

- `result.generated_files`
  - Output paths collected from WanGP's normal gallery / save path handling.
- `result.errors`
  - Structured generation errors. Runtime failures do not raise from `submit_task(...)`; they appear here.
- `result.artifacts`
  - Optional returned media payloads requested through `_api`.
- `result.cancelled`
  - Optional returned if job was cancelled / aborted.

Useful `GeneratedArtifact` fields:

- `artifact.path`
  - Saved output path for that artifact when WanGP produced one.
- `artifact.media_type`
  - `"video"`, `"image"`, or `"audio"`.
- `artifact.video_tensor_uint8`
  - Optional returned video tensor in WanGP's native post-decode layout: `[C, F, H, W]`, `uint8`.
- `artifact.audio_tensor`
  - Optional returned audio tensor / array when requested.
- `artifact.audio_sampling_rate`
  - Sampling rate associated with `artifact.audio_tensor` when present.
- `artifact.fps`
  - Output FPS associated with `artifact.video_tensor_uint8` when present.
- `artifact.side_files`
  - Companion filenames mapped to their file contents as `bytes`, such as `{"song.abc": b"...", "song.mid": b"..."}`. Python session calls return these in memory by default, while the main media is still saved for the gallery. Set `_api={"return_side_files": False}` to save the companions beside the media instead. They are not separate gallery entries.

For YuE2, set `custom_settings={"save_score": 1}` to export both the ABC and MIDI composition. This works with automatic planning, a supplied score or source-audio scoring; direct generation has no score. These describe the composition used to generate the song, not an exact transcription of the finished audio.

Wan Alpha and LTX Alpha Gen return their transparent output through the same `side_files` field. **Configuration / Outputs / RGBA Video Output** selects lossless RGBA PNG frames in a `.zip` (default) or a ProRes 4444 `_rgba.mov` for video editors. The shared `wgp_config.json` key is `rgba_video_output`, with values `png_zip` and `prores_4444`. Wan Alpha's gallery video is a checkerboard preview; LTX Alpha Gen's is the grayscale matte. This choice is independent of the gallery codec and container. Transparent still images are saved as PNG. Deepy model search accepts the `alpha output` speciality (alias `RGBA`); each matching model describes how to enable its alpha output.

To supply a YuE2 score, set `custom_guide` to the path of a UTF-8 `.abc` file and use `audio_prompt_type=""` with composition mode 0 or 1. Source-audio scoring (`audio_prompt_type="A"`) hides and ignores this optional file. The former pasted `custom_settings.abc` value is ignored; upload a file instead.

```python
for artifact in result.artifacts:
    for filename, content in artifact.side_files.items():
        (Path("my_exports") / filename).write_bytes(content)
```

Create the destination directory before writing. Companion names follow the final media filename, including any suffix WanGP adds to avoid overwriting an existing output.

## MCP Server

### MCP authentication and HTTPS

See [MCP Authentication and HTTPS](AUTHENTICATION.md#mcp-authentication-and-https) for OAuth setup, passphrases, client approval, token lifetimes, certificates, and reverse proxy hosting. MCP authorization is separate from the Gradio/Deepy browser login.

### MCP server inside the web UI

MCP is off by default with the web UI, including after upgrading. Enable **Configuration > Agents > Enable MCP with Web UI** and restart, or launch with `--mcp-webui`, to let agents work in the WanGP you are using. `--mcp` starts a separate, headless WanGP: it loads its own copy of the models and keeps its own queue and gallery, so nothing it does appears in a web UI you already have open.

```bash
python wgp.py --mcp-webui
python wgp.py --mcp-webui --mcp-port 7866 --mcp-allow-read-file-system
```

WanGP prints the MCP address at startup, `http://127.0.0.1:8000/mcp` by default. Set **MCP Server Port** under **Configuration > Agents** and restart to use a different port. `--mcp-port` overrides that saved value for the current launch, including standalone HTTP MCP servers. The **Enable MCP with Web UI** checkbox in Agents controls autostart (off when no preference has been saved). `--mcp-webui` and `--no-mcp-webui` override that choice for one launch without changing the saved configuration. It serves the current API (v2.1) over Streamable HTTP; `--mcp-transport sse` selects SSE. stdio is unavailable in this mode because the agent does not start the WanGP process. `--mcp-host`, `--mcp-api-version`, `--mcp-allow-read-file-system` and the [MCP OAuth options](AUTHENTICATION.md#mcp-authentication-and-https) work as they do with `--mcp`.

Agents and the people using the web UI then share one WanGP:

- An agent's generation joins the main queue like **Add New Prompt To Queue**: it runs after the tasks already waiting, in submission order. Every open page shows it in the queue, progress bar and live preview. Its output goes to the agent's workspace without switching the gallery you are viewing, and appears when you open that workspace. Deepy's own generations still run next, ahead of waiting tasks.
- Agents can queue several jobs. Asynchronous calls are always available in this mode: `wangp_generate(settings={...}, wait=false)` returns a `job_id` at once. Follow a job with `wangp_session(action="get_job", arguments={"job_id": ...})` and stop it with `cancel_job`. `list_queue` shows the whole queue, including tasks started from the browser, and `cancel_queue_task` removes a single task.
- An agent waiting for its result does not delay other agents' calls.
- **Abort** in the web UI stops an agent's running job, and the agent receives it as cancelled.
- `wangp_list_gallery` reads the agent's assigned workspace, or the server's gallery when no workspace is assigned. Agents can reuse its media as inputs without changing the browser's current view. Media creation also records into the agent's workspace without changing that view; an agent works in another workspace by attaching to it, for instance when the user names one.
- Models are loaded once for both, instead of a second copy in VRAM and RAM.

Every connected agent can read and cancel any queued task, including yours, and all agents share one job list. Keep the MCP address on localhost or a trusted network, or enable MCP OAuth. Web `--auth` protects only the browser pages.

### MCP API v2.1 additions

**API v2.1 is the default**, including for Deepy Prime. Select `--mcp-api-version 2` or `1` for an earlier contract. The version selects WanGP's tools, not the MCP transport protocol. V2.1 uses the existing v2 generation, queue and post-processing calls described below.

External v2.1 servers send WanGP's agent instructions when a client connects: the tool-calling conventions, model choice concepts and rules, settings layering, prompt separators and media input roles that Deepy Prime also follows. The `wangp_agent` prompt returns the same text, also readable as `wangp://guides/agent`.

External agents now use `wangp_toolbox(action="remux_media", arguments={...})` in place of `replace_audio`. This is a deliberate change in v2.1: use `audio_ids` (an array), optionally `video_id`, `mode`, `gains_db` and `subtitle_tracks`; a mix also accepts per-file `trim_start_s` and `trim_duration_s` excerpts, `offsets_s`, `fade_in_s`, `fade_out_s`, `gain_points` volume automation, `ducking` under voices, `video_audio_gain_db` for the video's own soundtrack and `normalize_lufs` for a loudness target. One audio file plus a video replaces its soundtrack; multiple audio files mix by default. `mode="multitrack"` keeps separate selectable streams. Subtitle tracks accept authorized SRT/VTT/ASS/SSA paths. Read the action contract for supported formats and options.

**Configuration > Agents > Visual Inspection by Agents** decides who looks at media for external agents. With **Agents Look Themselves** (the default), `wangp_toolbox(action="view_media", arguments={...})` returns **native MCP images** for the agent's own vision: Gallery images, and video frames at chosen `times_s` or `frame_nos`, or `count` frames spread evenly over `range_s` (the whole video by default), optionally cropped with `bbox` and sized with `max_edge`. `contact_sheet=true` tiles them into one image labelled with their times, convenient for checking motion, continuity or variants at a glance. A call returns at most 32 images and 8 megapixels in all: `max_edge` is lowered when needed, and the result reports the size used. Render 3D assets with `render_3d_views` first. With **WanGP's Visual Inspector Answers Their Questions**, `inspect_media` and `inspect_video` instead give the question and the selected images or sampled frames to WanGP's Visual Inspector and return only its text answer, which keeps the media on this machine and saves network traffic. Locally, the inspector is the prompt enhancer's Qwen3.5VL or Qwen3.8VL model, which looks at the images themselves (Deepy need not be enabled): a running generation pauses at its next step while it answers, and Deepy's VRAM mode decides whether the model stays loaded for the next question. When Deepy runs on a remote engine, the images go to that engine instead. **Both** offers all three actions. Deepy Prime always keeps its configured visual inspector.

Agents get two virtual roots. `@documents` is a writable folder for working files: prompt files, captions and scores, notes, ZIP packages. It is unrelated to gallery workspaces, and was named `@workspace` before (that name is still accepted). `@outputs` holds the media of the agent's gallery workspace: that workspace's own folder in dedicated storage, or the configured outputs folder in shared storage; media tools write there, and `wangp_io` only reads it. `wangp_workspace()` reports the disk folder behind each root in `folders`. In shared assignment mode, each external server has its `@documents` under its configured output directory. Its files remain on disk after the server stops; a new server starts a new documents folder. Clients using that server's shared context share it. Assigned agent contexts instead retain their own documents folders across reconnections; see [External MCP 2.1 Workspaces](#external-mcp-21-workspaces). Writes and deletion are confined to `@documents`. `--mcp-allow-read-file-system` permits additional filesystem reads; it does not grant arbitrary writes. Plugin filesystem access still requires the explicit read option.

General file delivery is available through `wangp_session`:

- `create_file_upload`, with `filename`, returns a resumable token and, on HTTP servers, a `PUT` URL. Upload the file bytes to that URL; the response gives its authorized workspace `path`. This accepts documents and scores as well as media and does not automatically add them to Gallery. Use `create_gallery_upload` when the upload should appear in Gallery instead. General uploads support every transport; HTTP servers also provide a PUT URL.
- `create_file_download`, with `source` set to an authorized path or Gallery ID, returns file metadata and, on HTTP transports, a `download_url`. `wangp_io(action="download", arguments={"source": ...})` performs the same operation externally. Resolve relative HTTP URLs against the MCP server's origin. V2.1 links expire after 24 hours, support repeated/ranged reads and survive server restarts. Their tokens are short random identifiers: the file path and permissions stay on the server.
- Every transport, including stdio, also receives a binary `resource_uri`. Read it through MCP `resources/read`; large files can be fetched as successive `wangp://files/<token>/<offset>/<length>` resources, with at most 8 MiB per read. Offset and length count decoded bytes. The response includes `size_bytes` and continuation instructions; reassemble byte ranges in order. Resource reads can be repeated until expiry. For text, use IO `write_text` or `append_text`; binary uploads also work over stdio through the resumable upload actions below.

V2.1 generation results include companion files under each artifact's `side_files`: filename, authorized path, MIME type and byte count. Download those paths with the same file-delivery actions. The main generated media stays in Gallery.

`mcp_resource` is now available to external agents as well as Prime. Call with `query` alone to search the documentation index; with `uri` and `query` to search passages; with `query="*"` to list headings; or with `uri` and `section` to read matching sections. Omit query/section to read the whole resource. Follow `next_call` until complete. Standard MCP resources remain available.

```python
mcp_resource(uri="wangp://docs/settings", query="video_length")
wangp_io(action="write_text", arguments={"path": "@documents/captions.srt", "text": captions})
wangp_toolbox(action="remux_media", arguments={"video_id": video_id, "subtitle_tracks": [{"path": "@documents/captions.srt", "language": "eng"}]})
wangp_session(action="create_file_download", arguments={"source": "@documents/captions.srt"})
wangp_list_gallery(media_type="3d", selected_only=True)
```

`wangp_list_gallery(selected_only=true)` returns the media you selected in the gallery WanGP shows, with the video playhead time, even while the agent records in its own workspace; `selection_workspace` names that workspace. When agents may not see it (**Workspaces Agents Can See**), the list is empty and `note` says why. Agent outputs never select themselves. External agents' gallery rows include the virtual `path` (such as `@outputs/clip.mp4`) of files inside their folders, for use with `wangp_io`.

Gallery discovery includes 3D assets in `all`, supports `media_type="3d"`, and preserves 3D selections in MCP v2/v2.1 and Prime. Deepy Zero's Get Selected Media and Resolve Media also support `3d`; selecting `all` includes the 3D gallery. Directory listings can likewise filter by `3d`.

Worlds of interactive world models are listed the same way for MCP clients and Prime: `all` includes the Worlds gallery, `media_type="world"` lists only worlds, with their resolution, duration, perspective and style, and `wangp_io(action="info")` on a world's `media_id` returns each segment's time range, description, actions and camera keys. Each world's video is in the image/video gallery (`video_filename`), where the media tools apply. `wangp_templates(task="gen_world")` returns the world templates and `wangp_models` finds world models with `filters.main_output="world"`, and models that continue a world list `world` among their inputs (`filters.inputs="world"`). A generation with a world model takes `world_output`: `"world"` (the default: the world alone, without decoding a video), `"world_video"` or `"video"`; its prompt may follow the world description with exploration instructions, one move per line (`[Forward]`, `[Look_Left]`, `[Event] a deer crosses the trail`), and `custom_guide` set to a world's `media_id` continues that world, in which case instructions alone are a valid prompt. Interactive exploration is done by the user, in the Worlds gallery.

`wangp_templates(task="gen_3d")` returns the main recipes that make a 3D asset from one picture of an object in `image_refs` (TRELLIS.2 Shape with SUPRA-Tex by default, Hunyuan3D 2.1 with its Paint Texturer, Pixal3D with SUPRA-Tex, with or without the Hi3D Refiner), so a request in words first generates that picture with a `gen_image` template. `wangp_templates(task="texture_3d")` textures a mesh of the 3D gallery given as `custom_guide`, with its object picture in `image_refs` (SUPRA-Tex by default, Hunyuan3D 2.1 Paint, or Hi3D refinement then SUPRA-Tex). Other 3D models (untextured shapes, parts, Gaussian splats, segmentation, animation, reconstruction) are found with `wangp_models` and run with their defaults. The `gen_world`, `gen_3d` and `texture_3d` tasks are for Deepy Prime and MCP clients.

File-producing media toolbox actions accept an optional `name` for their output file, without extension or folder; WanGP numbers it when the name exists, and actions writing several files prefix each one with it. Automatic generation file names omit window commands such as `[/duration=5s]`; set the `output_filename` setting for a chosen name. Gallery records label outputs created by external MCP clients with `source="MCP"`.

#### Preflight and safe retries

Call `wangp_validate(settings={...})` or pass a list for a batch before generation. It uses the submission resolver for prompts, media, standing defaults and model declarations. The default summary includes `requested_dimensions` and `requested_video` from normalized settings, `adjustments` made to supplied settings, active input modes, input file metadata, declared limits, inactive settings, checkpoint/LoRA availability, current runtime configuration and queue counts. These requested values are not output predictions: reference sizing, window scheduling, continuation and trimming can change the saved dimensions or duration. Read completed media metadata for actual values. Mesh generation omits these image/video fields. Use `detail="full"` for all effective settings, their origins and the full queue. `valid=false` includes diagnostics; nothing is submitted. For processing, use `wangp_validate(media=media_id, process=process_id, parameters={...})` with the processor's discovered parameters.

For visual generation, `sizing` reports the model's dimension `multiple`, the configured sizing `mode` (`pixel_budget`, `outer_box` or `crop`), and the grid-adjusted `canvas_dimensions`. It also reports `initial_dimensions` for a plain canvas or a single starting image, using the same image sizing rules as generation. For example, a requested 832×480 canvas becomes 832×448 on a model with a 64-pixel grid. When `video_prompt_type` contains `K`, the first reference image is the main subject or landscape and sets the canvas: `initial_dimensions` follows its aspect ratio within the requested pixel budget and `dimensions_from` says so; use `I` to keep the requested resolution. More complex reference/control combinations, prompt scheduling and outpainting are marked unresolved. These dimensions precede model-specific processing and optional upscaling; later windows may use different inputs.

For a sliding-window video, `window_plan` lists each window as WanGP plans it: generated frames and their length in seconds (`generated_seconds`, the window's own timeline that window-timed prompts refer to), new frames and their duration, overlap, the pictures it receives (start or continued frame, end image, injected frames, references) and the head of its prompt, with `total_frames` and `total_seconds`. Window rounding can make the total differ from `requested_video`; `note` then says so. Diagnostics warn when window prompts, windows and end images do not line up, when the plan has more windows than the length needs, and when the resolution is generated at another size because of the model's alignment. `wangp_generate` also reports that size change as `canvas` (`requested` and `generated`) when it accepts the job, so overlays and anchors can match the generated size. The plan of a continued video depends on its retained source frames and is marked unresolved.

Preflight performs no downloads, model loads or inference. It is a snapshot, and submission checks again. Checkpoint availability does not establish that all optional/ancillary assets are installed; the response lists checks it cannot complete. Queue state covers the connected instance only. It does not reserve the GPU or predict VRAM requirements. Settings copied from profiles/templates count as explicit inputs because their original source is not carried in a plain settings object.

`wangp_generate` and `wangp_postprocess` accept an optional top-level `request_id`. Reusing the ID with the same submission payload returns the original job, with `reused=true`. A different payload with that ID fails. Waiting options are independent of the payload. Keep the same ID after a lost response, and use a new ID only for a deliberately new submission.

Small durable receipts retain job/client IDs and outcomes. After restart, completed receipts remain readable through `get_job`. Incomplete receipts are matched against surviving queue tasks and Gallery metadata; `unknown_after_restart` means completion could not be established. Such requests are never automatically resubmitted. Matching surviving tasks can be cancelled through `cancel_job`; recovered outputs do not by themselves prove an entire batch completed.

Newly saved outputs expose `role="final"` or `"intermediate"` in Gallery records, with `task_id` identifying the generating task's `client_id`. When present in saved settings, `window_no` identifies the 1-based window that produced that cumulative output. Sliding-window checkpoints are intermediate; the finished video is final. A batch or one task can have several final outputs. Older files without this metadata omit the corresponding fields. Job `gallery_items` are captured at completion; query the Gallery for current selection or membership. Media utility results use the same compact records and canonical source IDs; retrieve full settings through `wangp_toolbox(action="media_settings", arguments={"media_id": ...})`.

Running job snapshots include the latest `progress` even with `event_limit=0`, the v2.1 default of `get_job` and `cancel_job`; request recent events explicitly. Its task index/count belong to that job, and window index/count describe live execution. Progress is an indication, not an output identity or completion contract: use the job result for completion and saved `window_no` for output identity. Use the phase and its `current_step`/`total_steps` for progress; an unknown count is null. The legacy percentage is a local phase estimate, not a batch-wide percentage or ETA.

Completed v2.1 snapshots carry a top-level `status` (`done`, `error` or `interrupted`) beside the result. They omit the last progress and timestamps, and omit `generated_files` when the result's `gallery_items` name every output: use those media IDs. `wangp_session(action="get_job", arguments={"job_id": ..., "wait_s": 120})` waits up to that many seconds (at most 600) for the job to finish; `wait_jobs` with `job_ids` and `mode="any"` or `"all"` waits for several jobs and returns every job's state, with `status="timeout"` when the condition was not met.

The WebUI MCP HTTP endpoint returns `X-WanGP-Request-ID` for correlating requests and `Server-Timing` durations in milliseconds: `queue` is the worker dispatch delay, `handler` is tool execution (including assigned-workspace routing), and `mcp` is elapsed time until response headers. Workspace calls and polling run in workers so routing does not block other HTTP requests. The `wangp.mcp.timing` Python logger records the same request IDs and timing boundaries at DEBUG level, with calls taking at least one second logged at WARNING. Compare these with client send/receive timestamps when investigating delays; a deliberate wait for generation also contributes to handler time. Headers remain available when a model temporarily suppresses Python logging.

```python
wangp_validate(settings=settings)
wangp_generate(settings=settings, request_id="scene-04-take-1")
# Retry the identical submission after a lost response:
wangp_generate(settings=settings, request_id="scene-04-take-1")
```

#### Shared skills

External MCP v2.1 and Deepy Prime share the same skill library. Browse compact descriptions with `mcp_resource(query="skills")` or search by task, then read a returned URI for the full Markdown procedure. Shipped skills live under `wangp://skills/<name>`; user skills live under `wangp://skills/user/<name>`. Skills provide on-demand guidance, not automatically executed code.

Use `wangp_io(action="save_skill")` for the contract. To create a skill, supply `name` and `content`, where content is a complete UTF-8 `SKILL.md` with YAML front matter and a Markdown body:

```python
content = """---
name: review-video-joins
description: Check motion and camera continuity at video-window boundaries.
---

# Review video joins

Read the saved outputs' window_no and frame counts. Inspect neighboring frames
at each cumulative boundary, comparing subject pose, position and camera motion.
Use the chosen model's current guidance when planning a continuation.
"""
saved = wangp_io(action="save_skill", arguments={"name": "review-video-joins", "content": content})
current = mcp_resource(uri=saved["uri"])
# After editing content, update against the revision you read:
wangp_io(action="save_skill", arguments={"name": "review-video-joins", "content": content, "expected_revision": current["revision"]})
```

Names are lowercase hyphenated slugs of at most 64 characters. Front matter must have the matching `name` and a nonempty `description` of at most 1024 characters; the complete file is limited to 128 KiB. Omit `expected_revision` only for creation. An existing file or stale revision produces a conflict instead of overwriting another agent's work. Read again, reconcile the changes, and resubmit with that revision.

User skills are stored in `user_skills/<name>/SKILL.md` under the WanGP installation and persist across sessions and restarts. They are shared across its workspaces and external/Prime connections, and appear on the next resource browse without reconnecting. Existing pagination cursors continue their original snapshot; start a fresh browse to see changes. Shipped skills cannot be overwritten through this action. Refer to current model metadata and templates instead of copying a model catalogue, full settings, temporary media IDs or session-specific paths into a skill.

#### Audio inspection

External MCP and Deepy Prime expose `wangp_toolbox(action="inspect_audio", arguments={...})`. Pass `media_id`, optionally a 1-based `audio_track_no`, `start_s` and `duration_s`. Measurements use CPU FFmpeg and streaming samples: sample rate/channels, per-channel peak and RMS, full-scale sample counts, EBU R128 integrated loudness/true peak and silence intervals. Coverage and thresholds are returned; at most 200 silence intervals are listed while aggregate counts/duration cover the whole analyzed range. `window_s` (0.1 to 600 seconds, at most 600 windows) adds an `envelope`: lists of RMS, peak and loudest momentary loudness per window (window i starts at `start_s + i * window_s`), to locate climaxes, gaps or a balance problem in one call.

`source_duration_seconds` describes the container; `audio_start_time_seconds` and `audio_duration_seconds` describe the selected track. The requested range is intersected with that track's playable timeline when its duration is known, excluding decoder tail padding such as AAC's final padded samples. An unavailable track duration is null; `coverage` always reports the samples actually measured. Times are relative to the source file. A silence interval means RMS below `silence_db`, not necessarily zero samples or missing audio; a quiet musical ending may fall below the threshold.

Use `preview=true` and optional `preview_s` (default 10, maximum 30) to create a playable Gallery excerpt starting at `start_s`. The preview is stereo 24 kHz WAV; measurements retain the source channels/rate. External clients can download the excerpt using its returned Gallery ID. Prime can present it to the user. Metrics and playback do not provide semantic listening for a text-only model; transcription remains a separate tool.

`merge_videos` joins 2 to 32 videos in order (`media_ids`), with a cut, `crossfade`, `fade_black` or `fade_white` at each join (`transitions`, one per join or one for all) lasting `transition_s` seconds. Clips are fitted to the first video's size and frame rate; their soundtracks follow the picture, with silence for a clip without sound. `draw_svg` with `duration_s` makes a silent video clip from SVG markup, animated by SVG animation elements (`animate`, `animateTransform`, `set`, with `begin`, `dur`, `values`, `keyTimes`, `calcMode`, `fill="freeze"`): title and end cards, credits, lower thirds or a slow zoom over a Gallery image. `like_video_id` gives the clip a video's size and frame rate so that it merges directly. With `video_id`, `draw_svg` draws over that video instead, from `start_s` to `end_s`: the drawing is burned into the picture with its transparent parts showing the footage, its animations start at `start_s`, and the new video keeps the original's sound and subtitle tracks. Use it for designed on-screen text and graphics (titles, lower thirds, labels, interface screens, timers), not for subtitles, which cannot be removed once burned in and belong in a subtitle track. `create_color_frame` with `duration_s` makes a solid colour clip. `transcribe_media` tightens segment and word times to the detected speech, so a pause is not counted in the next word, and returns `speech_intervals`; transcribe the clean voice, or isolate it first with the `remove_background` audio treatment, for speech over music. `inspect_video` samples up to 2 frames per second, and up to 8 per second over a range shorter than 3 seconds, to check joins and cuts. External v2.1 servers with asynchronous calls accept `wangp_toolbox(..., wait=false)` for slow actions such as transcription, 3D views or masks: it returns a `job_id` followed with `get_job` / `wait_jobs`; such a job runs to completion and cannot be cancelled. Media results name files in the agent's virtual roots (`@outputs/...`), like generation results.

When combining tracks with `remux_media(mode="mix")`, each file plays at its own level plus its gain: nothing is averaged, so a file's level does not change when another one starts or ends, and a true-peak limiter (computed at four times the sample rate, at -2 dBTP for WAV, FLAC and ALAC and -3 dBTP for AAC and MP3, whose encoding adds about 1 dB) keeps the delivered file below -1 dBTP. Every input is converted to the highest sample rate and the widest channel layout among them. `gain_points` automates a file's volume with `[time_s, gain_db]` points on the mix timeline, for swells or fades within a file. `ducking` lowers music or ambience while voice files speak: `{"tracks": [0], "under": [1], "depth_db": 10}` dips the first file by 10 dB whenever the second one speaks, with ramps set by `attack_s` and `release_s`; speech is found in the voice files themselves. `trim_start_s` and `trim_duration_s` select the excerpt used from each file without separate extraction, `offsets_s` places each excerpt on the mix timeline (back-to-back offsets concatenate clips), `fade_in_s` and `fade_out_s` apply on the excerpt's own timeline, and `video_audio_gain_db` sets the gain of the video's soundtrack. `normalize_lufs` measures the finished mix and applies one gain to reach that integrated loudness before the limiter; very loud targets may then measure slightly lower. Inspect the result when balancing a soundtrack.

#### Resumable file delivery

V2.1 download descriptors last 24 hours and survive server restarts while the source file and transfer permissions remain valid. HTTP downloads accept `Range: bytes=<offset>-<last-byte>`; stdio uses the returned binary resource URI with decoded-byte offsets. No second copy of the media or per-client progress journal is kept. A changed or vanished source invalidates the descriptor. `create_file_download` accepts `include_sha256=true` when the client needs a full-file integrity check; computing it reads the file once on CPU.

Create a binary upload on either transport:

```python
wangp_session(action="create_file_upload", arguments={"filename": "reference.wav", "size_bytes": byte_count, "sha256": digest})
wangp_session(action="upload_chunk", arguments={"token": token, "offset": 0, "data_base64": encoded_chunk})
wangp_session(action="get_upload", arguments={"token": token})
wangp_session(action="finish_upload", arguments={"token": token})
```

`size_bytes` and `sha256` are optional, but supply them to verify the expected file before publication. Chunks are limited to 8 MiB of decoded bytes, and the total upload limit is 8 GiB. Upload sequentially at the returned offset. Identical chunk retries are safe; conflicting bytes or offsets fail. Acknowledged bytes are flushed to disk. After a lost connection or restart, call `get_upload` and continue from its offset. A disconnect may have written additional bytes beyond the last acknowledged offset.

HTTP clients may PUT a whole file without `Upload-Offset`, or send sequential chunks with that header. A complete declared-size HTTP upload finalizes automatically; chunked uploads without a declared size need `finish_upload`. Repeating finalization returns the completed path. One managed `.part` file becomes the final file through rename after size/checksum verification; no duplicate full media copy is retained. The original destination survives a new server workspace. Upload descriptors also expire after 24 hours; expired/incomplete partial files remain on disk for operator cleanup and cannot be resumed using an expired token.

The server keeps small request receipts and transfer grants under the user's `WanGP/mcp` application-data directory (Windows: `%LOCALAPPDATA%`; otherwise `~/.cache`), scoped to the project. These are required for restart recovery. Tokens are random bearer capabilities whose grants stay on the server; treat them like download links and retain them with the client's transfer offset.

### MCP API v2 and migration

WanGP serves **API v2.1 by default**. The v2 call conventions below also apply to v2.1, except where the additions above specify a change. The version is fixed at server startup; it versions WanGP tool contracts, not the MCP protocol. Pin existing integrations to v1 or v2 when needed:

```bash
python wgp.py --mcp --mcp-api-version 1
python wgp.py --mcp --mcp-api-version 2
python wgp.py --mcp --mcp-api-version 2.1
python -m shared.mcp --mcp-api-version 1
```

V1 preserves historical names, schemas, defaults, results, errors and execution behavior. V2 uses compact toolbox definitions and deferred action contracts. Underlying operations are shared; aliases are normalized internally rather than advertising both syntaxes. Deepy Prime pins v2.1 independently of this external-server setting; Deepy Zero keeps its historical internal API.

V2 toolboxes use progressive discovery for advanced operations: omit `action` and `arguments` to list actions; supply `action` with absent/null `arguments` to read its contract; supply an `arguments` object to execute. `{}` executes an action with defaults. Arguments without an action are rejected. Declared shortcuts and known contracts execute directly without repeated discovery. `wangp_model` actions only read the chosen model, so they run when `arguments` is omitted; its discovery lists their optional arguments. `wangp_model` additionally requires `model_type`; `wangp_postprocess` requires `media` (image/video/audio/3d or a concrete Gallery ID/authorized path; meshes take the 3D Refiner and Texturer, Gaussian splats have no treatment). Post-processing execution requires concrete media. A contract read for a concrete image or video gives each upscaler's `approximate_output_dimensions` per multiplier; spatial and temporal treatments of a video copy its soundtracks and text subtitle tracks. Deepy Prime and external v2.1 expose `remux_media` in place of `replace_audio`: pass `audio_ids` and optionally `video_id`. One audio input plus a video replaces its soundtrack; multiple inputs mix by default. `mode="multitrack"` retains separate selectable streams. To add selectable subtitles, pass `video_id` and `subtitle_tracks`, each with an authorized SRT/VTT/ASS/SSA file `path` and optional ISO 639 `language`, `title`, and `default`. With no `audio_ids`, existing audio streams are copied when compatible with the output container; incompatible streams use the configured video audio codec. Existing video subtitles are retained by default; `include_video_subtitles=false` removes them. MP4/MOV use `mov_text`; MKV copies compatible subtitle streams or converts unsupported ones. Audio-only mixes use the configured standalone WAV or MP3 format by default. When requested, `output_extension` can select WAV, MP3, FLAC or M4A for a mix, or MP4, MOV or MKV for a video output. Audio-only multitrack files use M4A because WAV and MP3 cannot hold separate streams. M4A copies AAC/ALAC streams and converts other streams to lossless ALAC; multitrack video uses MKV and copies audio streams. Video soundtrack replacement copies compatible audio streams and otherwise uses the configured video audio codec. Mixing re-encodes the combined result. External MCP V1 and V2 retain `replace_audio`.

For a named model, use `wangp_models(query="SenseNova")` directly. The case-insensitive substring search covers name, ID, family and description; it needs no wildcard. Omit all arguments for discovery. For advanced filters, use `action="search", arguments={"query":"SenseNova","filters":{"main_output":"image"}}`; do not combine the shortcut with action/arguments. String filters match whole values, with optional `*` and `?` globs. An incomplete shortcut result includes a `next_call` recipe for continuation. Results name each model's specialities; `action="specialities"` lists their aliases. `templates` lists the ready-made recipes built on a model as `task: template`, marked `(default)` for a task's default; read one with `wangp_templates(task=..., template=...)`.

To discover models for an exact canvas, use `wangp_models(action="search", arguments={"filters":{"main_output":"image","resolution":"2048x2048"}})`. The `resolution` filter accepts a positive `WIDTHxHEIGHT` string, not globs or category names. It compares pixel area against recommended budgets before ranking and pagination; results include a `resolution_assessment` with the requested and block-aligned canvas, recommended mode, and any existing phase setting to apply. Missing or ambiguous upstream limits use the shared 1080p budget (1920x1088, including VAE-aligned presets). Agent generation requests outside the selected mode's accepted range are rejected, even when a model is named directly.

Visual model definitions accept `resolution_budget`, `resolution_budget_tiled` and `resolution_min` as positive integer pixel areas or `"WIDTHxHEIGHT"` strings. The dimensions are an area shorthand, not side or ratio constraints. Discovery normalizes them to integer pixels per image/video frame. `resolution_budget_tiled` applies to the existing Two Phases with Tiling workflow, and inherits `resolution_budget` when omitted. A missing/null maximum uses the shared 1080p fallback. Bounds are not memory guarantees. Python API, MCP and Deepy generation reject requests above the active maximum or below the declared minimum during submission validation, before generation. If another available phase fits, the error suggests its setting; it is never enabled silently. The global UI resolution override does not bypass these checks.

`resolutions` adds custom presets to the predefined menu and does not imply a maximum. `lock_resolutions: true` restricts selection and generation to that model's list; discovery exposes its block-aligned `allowed_resolutions`. Exact resolution searches check the minimum, available budgets and locked-list membership before pagination. Locked sizes must also fit the explicit or fallback budget. An unsupported locked size is rejected with the allowed values, never silently replaced. The minimum normally hides smaller presets. The global 3K/4K+ UI option exposes the full shared/custom catalog and removes the ordinary slider minimum for unlocked models; out-of-range UI selections produce a nonblocking warning when submitting, never when changing the picker. Imported dimensions are retained.

Optional `max_width` and `max_height` declare independent hard limits in pixels, as integers at least as large as `vae_block_size`. Omitted or null values leave that axis unrestricted. These limits filter all preset sources and restored selections, including with the global 3K/4K+ option enabled, and constrain the megapixel picker for its chosen aspect ratio. Direct generation requests exceeding either limit are rejected. When a main reference sets the output shape (`KI`), its fitted canvas is scaled down as needed to respect both limits, preserving its aspect ratio within the model's pixel grid; Deepy preflight reports that capped size. Discovery and Deepy templates expose the limits as model information, and resolution searches exclude models that exceed either limit even when their pixel budget fits.

For example, `"resolution_budget": "4096x4096"` is exposed as `16777216`. With `"max_width": 4096` and `"max_height": 4096`, `4096x4096` is allowed but `5440x3072` is not. A model with `"resolution_budget": "2048x2048"` and `"resolution_budget_tiled": "3840x2176"` can recommend `guidance_phases: "2~"` for a larger target when that workflow is supported.

Pass the same exact `resolution` string to generation. No MP, ratio or preset/category argument is required. Block alignment, input fitting and postprocessing remain independent; inspect a returned alignment difference, or use preflight sizing when final dimensions matter. The UI's Presets / MP preference does not change API semantics. Python `list_model_defs` and `list_model_metadata` accept the same resolution filter. The legacy MCP v1 tool contract remains unchanged.

For contract discovery, `null` replaces the entire `arguments` object: `{"action":"capabilities","arguments":null}`. Partial objects and null values inside them are execution attempts. Validation errors reject them and explain how to retrieve the contract. Action lists are compact name-to-description maps. A known call recipe can be executed without another contract request. A model's root `definition` lists loading, packaging and UI-enhancer declarations by name under `other_declarations`; read any of them by `property`.

V2 rejects unknown top-level tool parameters instead of discarding them. Single-path IO actions use `path`, including `read_text`, `write_text`, `append_text` and `edit`; `file_path` remains accepted as a compatibility alias. `info` also accepts the historical `source` alias. Ripgrep uses `wangp_io(action="rg", arguments={"command":"-n pattern -- @documents/file.txt"})`; its former inner `arguments` name remains accepted. Conflicting alias values are rejected. Action contracts contain complete call examples. Markdown appends return a compact `structure` receipt with heading count, headings added, last heading and its occurrence count. This can replace routine heading searches; it does not establish semantic consistency. V1 and Deepy Zero retain their existing call formats and append receipts.

Discovery also gives direct execution syntax for actions with no parameters. Generation needs no discovery: `wangp_generate(settings={...})` takes one flat settings object with `model_type` and a non-empty `prompt` (or a list of them for a batch). Keys in `custom_settings` must be custom settings declared by the model. The earlier `wangp_generate(action="generate", arguments={"source": ...})` form remains accepted. The model `capabilities` result includes `input_guidance` for available input instructions (`infos`) and `prompt_guidance` for available prompt instructions (`prompt_infos`); reuse the same `model_type` with each recipe. These properties can also be read directly through `definition` without a capabilities query. Shared setting meanings belong to `wangp://docs/settings`. Workflow resources include `wangp://guides/workflows` and `wangp://guides/long-video`; writable-workspace servers also expose the long-story and long-generation-prompt skills on demand.

V2 accepts two exact model-query aliases: `definition` with `property="defaults"` or `property="capabilities"` reads the corresponding action when no actual declaration has that name. Other unknown properties return a complete call for reading available properties. For `wangp_toolbox` actions that take `media_id`, such as `inspect_media`, `inspect_video`, `inspect_audio` or `render_3d_views`, `media` is accepted as an alias for `media_id` when a string, or `media_ids` when a list and the action takes one. Conflicting values or multiple input forms are rejected before inspection. All inspection inputs belong inside `arguments`; a single video reference still inspects frame 0 by default, while `inspect_video` samples a time range. V1 and Deepy Zero retain their existing contracts.

Deepy Prime and external v2.1 clients read agent-oriented model help: optional `deepy_infos` and `deepy_prompt_infos`, written with setting names and flags, replace the corresponding `infos` and `prompt_infos` values. Each override is independent; absent overrides retain the standard help. Responses expose only the canonical names, and `prompt_guidance` reads the selected `prompt_infos`. `definition` with `detail="full"` returns the standard help written for the UI instead. Root definitions retain their 256-character previews; requesting a property returns its complete text. This does not add startup context. The UI, Python API, MCP v1 and v2 keep the standard help.

External MCP v2.1 and Deepy Prime expose the same recipes as `wangp_templates`, using the existing Deepy template files, configured defaults and model metadata:

- `wangp_templates()` lists the available tasks, their purpose, configured default template and template count.
- `wangp_templates(task="gen_image")` browses compact template summaries, with the configured default first. Each row includes its model, supported `media_inputs`, declared speciality names and selected `input_modes`. Full settings are omitted.
- Generation and validation settings can name a template instead of copying it: `{"template": "gen_video", "prompt": "..."}` starts from the task's configured default, `"gen_video/<template name>"` from an alternative. WanGP fills in the template's resolved settings except its example `prompt` and `alt_prompt`; the object's own keys override them and `custom_settings` merge. Batches stay short, and long values such as LoRA URLs are not retyped.
- `wangp_templates(task="gen_image", template="default")` reads the configured default's full resolved settings directly, with the model's information beside them in `model` (supported `media_inputs`, resolution budget and grid such as `vae_block_size`); that information is not a setting. Supply an exact returned template name to read an alternative.

Use templates as starting recipes; explicit user choices and overrides take precedence. `media_inputs` describes the model's supported roles, not mandatory inputs or populated media. Read the model's `infos` and `prompt_infos` for input combinations and prompt syntax, add the prompt and media, then check the resulting settings with `wangp_validate`. Full template settings already include active General Properties; apply explicit overrides last. Browsing does not submit generation, download assets or load models. Template lists accept `limit` and `cursor`; follow `next_call` when present. Full settings are returned only on selection and do not include the alternative-template list.

Explicit MCP v2 retains `wangp_deepy_templates(tool_id="gen_image")`, which returns the default settings and alternative names together; v1 retains its separate list/read tools. When moving a v2.1 client to the neutral name, replace `tool_id` with `task` and pass `template="default"` explicitly to read settings. The Gallery continues to use `wangp_list_gallery(media_type="video", selected_only=true)` directly, with its earlier `action`/`arguments` form also accepted.

| Historical capability | Current owner / action |
|---|---|
| Model search | `wangp_models` / `search` |
| Model schema, definition, defaults | `wangp_model` / `capabilities`, `definition`, `defaults` |
| Accelerator profiles, presets, user settings, LoRAs | `wangp_model` / `profiles` (with the recommended profile's settings), `presets`, `user_settings`, `loras` |
| Shared templates and resolved settings | `wangp_templates()` / `wangp_templates(task=...)` / `wangp_templates(task=..., template="default")` (v2.1) |
| Gallery inventory / selections | `wangp_list_gallery(media_type=..., selected_only=...)` |
| Generation | `wangp_generate(settings=...)`: one settings object, or a list for a batch |
| Post-processing | `wangp_postprocess` / discovered processor ID |
| Media utilities / previous generation settings | `wangp_toolbox` / discovered action or `media_settings` |
| File navigation, search, text edits, archives | `wangp_io` / `list`, `rg`, `read_text`, `info`, `edit`, `append_text`, `write_text`, etc. |
| Jobs, notifications, HTTP Gallery transfers | `wangp_session` / `get_job`, `cancel_job`, `list_queue`, `cancel_queue_task`, `notify`, `create_gallery_upload`, `create_gallery_download` (`list_queue` and `cancel_queue_task` are external-only) |
| Plugin workflows (such as Media Flow batches) | `wangp_plugins` (v2.1, when enabled Plugins declare a toolbox) / `plugin`, then a discovered action; `wait=false` runs it as a job |
| Documentation | Standard MCP resources; Prime and external v2.1 expose `mcp_resource` |

Example v2.1 calls:

```python
wangp_templates()  # available tasks
wangp_templates(task="gen_image")  # compact browsing
wangp_templates(task="gen_image", template="default")  # full settings on demand
wangp_list_gallery(selected_only=True)
wangp_generate(settings=prepared_settings)
wangp_generate(settings=[first_settings, second_settings])
wangp_io(action="info", arguments={"path": "@outputs/example.png"})
wangp_io(action="info", arguments={"paths": ["@outputs/clip.mp4", "@outputs/voice.wav"]})
```

V2 IO `info` accepts exactly one of `path` for a single authorized file/directory path or Gallery media ID, or `paths` for 1–20 such references. Single results keep their existing shape; batches return `files` in input order, each with the same metadata and per-file probe status. A failed probe sets the batch status to `error` while retaining the other results. Invalid or unauthorized paths reject the call. Use exact returned paths/IDs, not display labels or shortened filenames. Audio and video results list their `streams`: codec, sample rate, channels and layout, plus language, title and default flag for subtitle tracks. The legacy `source` argument remains accepted as a single-path alias; supplying both with different values is rejected. V1 retains its `source` contract.

Compact Gallery resolution, frame count, FPS and duration describe the actual media file, independently of requested generation settings. Audio duration is physically probed; animated-image duration comes from frame timing, while still images have no duration. Video statistics refresh when the file's modification time or size changes. Use `info` for further physical details and `media_settings` for the separate generation settings.

On the external MCP server, `wangp_generate` and `wangp_postprocess` take `wait`, `timeout_s` and `event_limit` as top-level parameters, beside `settings` or `media`, `action` and `arguments`. Post-processing still accepts them inside `arguments`, as earlier contracts listed them; the same control with different values in both places is rejected. `wangp_toolbox` takes `wait` only, and `wangp_validate` runs no job. V2 defaults to `wait=true` and rejects `wait=false` before submission unless launched with `--mcp-async`. This switch initially defaults off. A timeout returns the job ID and timeout state without cancelling; it does not authorize resubmission. Notifications remain separate, explicitly invoked operations.

Invalid top-level arguments are reported in plain words and nothing is executed: an unknown argument lists the accepted ones, with the closest name when it looks misspelled; a wrong type names the expected one; and an object or array sent as a JSON string reports where its JSON fails to parse. Action arguments get the same JSON check.

Deepy Prime manages synchronous waiting itself, including with remote LLMs. Its generation and post-processing contracts omit `wait` and `timeout_s`; Prime accepts but ignores these two controls at the top level or inside action arguments and reports this in the result. Other unknown arguments remain errors. `event_limit` still controls the returned event count. External MCP v1/v2 and Deepy Zero keep their existing behavior.

Batch generation is preserved: `settings` accepts a settings list, a task list, or a manifest containing `tasks`. The entire batch is passed in one submission to the existing WanGP queue, with its order and per-task settings intact. V2 does not split it into separate jobs or alter queue scheduling.

V2 checks the required model field before submitting any part of a generation batch. Each settings object, supplied directly or wrapped in `params` / `settings`, needs a non-empty `model_type`; legacy `base_model_type` is accepted when `model_type` is absent. Existing `edit_*` post-processing tasks need no model. A missing/invalid model field or malformed task container rejects the whole call with its exact location (for example `source.tasks[1].params.model_type`) and creates no job. V1 and Deepy Zero retain their existing validation paths.

V2 also checks non-empty supplied media inputs against the model declarations and effective input modes, including existing automatic media-flag inference and model defaults. Unsupported inputs, or inputs that an inactive mode would discard, reject the whole batch before submission with the offending field and reason. Empty media fields in full settings objects remain valid. These checks cover start/end images, reference images or injected frames, source/control videos, control images and masks, audio guides, custom guides, and media supplied to selected audio post-processors. Model-specific constraints such as frame-position counts remain part of normal generation validation. Direct Python API, MCP v1 and Deepy Zero validation paths are unchanged.

Frame injection and reference-image conditioning are separate capabilities. For example, H3 FL2VA reports `injected_frames: true` and `reference_images: false`: it accepts `image_refs` in `KFI` frame-injection mode with `frames_positions`, but not as general subject/appearance references. H3 REF2VA declares reference-image conditioning separately.

This is separate from multiple tool calls in one assistant response. Deepy's existing local execution loop keeps those calls grouped in the assistant message, executes them in order, and records one result per call before the next LLM pass. V2 preserves that loop, individual call IDs/results and the existing transcript display; it does not force callers to combine separate requests into one batch.

V2 collections use `limit` (default 20, maximum 100) and opaque `cursor`; results report `count`, `has_more`, `next_cursor`. A page also has a 6,000-character item budget. Repeat the original filters with `next_cursor` as `cursor`. Snapshots are stored outside the model context, remain stable across source changes, expire after ten minutes, and are evicted after 16 newer retained searches. Snapshot storage is limited to 128 MiB each. Expired cursors or changed filters produce an explicit error requiring a new search. `summary_only=true` stores the collection and returns its count and starting cursor without loading its contents.

IO `rg` adds this pagination around ripgrep; it is not a native rg option. Names-only results omit metadata; content searches return paths, line numbers and bounded excerpts. `complete=false` reports a timeout or incomplete search; `excerpt_truncated=true` requires a bounded `read_text` call if full text is needed. `rg -m` limits matches per file, not page size. IO `list` only navigates one directory; recursive/text search belongs to `rg`. IO `append_text` creates missing files or appends literal text without adding newlines. `write_text` supports exclusive creation or explicit overwrite. V2 prompt-file references require read permission only.

When the folder is named, list it directly, for example `wangp_io(action="list", arguments={"path":"@outputs/selection"})`. Avoid paging through its parent first. For a recursive filename search inside such folders, use `command="--files -g '**/selection/**' -- @outputs"`; `-g '*selection*'` matches basenames, not every descendant of the folder. `-i` changes text matching, not glob casing.

`wangp://guides/workflows` is an index of about 3 KB: one line per workflow section (templates, model limits, media inputs, post-processing and utilities, long video, prompt files and editing) with the URI that reads that section alone, such as `wangp://guides/workflows/long-video-and-sliding-windows`. `mcp_resource` searches and section reads on `wangp://guides/workflows` still cover the whole guide. External v2.1 agents also get a one-line call per action when they list a toolbox's actions, so most calls need no contract read; contracts keep the rules, defaults and examples. Detailed procedural guidance is loaded on demand. Explicit external v1/v2 selection retains its previous filesystem scope. External v2.1 adds the workspace and readable output directories described above; Plugin filesystem access remains tied to explicit read permission.

In Prime, prefer the session workspace for experiments and editable drafts. Filesystem tools manage that workspace freely. With write access they can create new files in configured output folders and their subfolders, but cannot change, overwrite, delete, rename or move existing output files. Other authorized folders retain their R/RW permissions; deleting or moving an external source removes its vanished-path Gallery entry. This policy does not change generation, post-processing, MCP v1 or Zero contracts.

All product manuals under `docs/` are available as complete MCP resources, including installation, configuration, troubleshooting, model selection, Plugins and API integration. Each document ends with a compact `Applies to` note describing its scope for readers. The current server uses that footer as its resource description; Prime and external v2.1 also include it as `applicability` metadata with document reads, section reads and search results, including continuation pages. The scope is therefore available from the first response without reading to the end. Section indexing excludes the footer so its keywords are not attributed to the final section; full-document reads retain it. Interface-specific examples are labelled within mixed documents. This documentation policy does not change filesystem permissions or Deepy Zero's routing.

The current workflow resource uses `wangp://guides/workflows`, without a version label in normal discovery or agent instructions. Its previous version-labelled URI is retained as an unlisted read alias. New documents are published automatically and must include an `Applies to` footer describing their subject, audience and relevant platform or interface limits; review tool examples, links and setting semantics before adding them. Product-help documents explain the UI and configuration; current callable contracts govern Deepy's actual tool execution.

### Server launch and historical v1 tool reference

The following tool signatures describe **external MCP v1 clients**. Select `--mcp-api-version 1` when using them. These historical signatures do not describe Prime's current toolboxes. Transport and runtime configuration options also apply to the current interface.

WanGP also exposes the same reusable session through an MCP server. The preferred launch path is:

```bash
python wgp.py --mcp --mcp-api-version 1 --config <config dir> --output-dir <output dir>
```

This preserves normal WanGP CLI/config arguments. MCP-specific launch options are prefixed:

```bash
python wgp.py --mcp --mcp-api-version 1 --mcp-transport stdio
python wgp.py --mcp --mcp-api-version 1 --mcp-transport streamable-http --mcp-host 127.0.0.1 --mcp-port 7866
python wgp.py --mcp --mcp-api-version 1 --mcp-transport streamable-http --mcp-allow-read-file-system
```

For Streamable HTTP, connect MCP clients to `http://<host>:<port>/mcp`. Use `--mcp-host 0.0.0.0` only on a trusted network or with [OAuth and HTTPS](#mcp-authentication-and-https). Direct server filesystem paths are rejected by default; media IDs returned by `wangp_list_gallery` remain usable. `--mcp-allow-read-file-system` explicitly permits agents to supply arbitrary existing server paths.

The lower-level adapter can still be launched directly:

```bash
python -m shared.mcp --mcp-api-version 1 --root <WanGP repo> --output-dir <output dir> --job-event-limit 20
python -m shared.mcp --mcp-api-version 1 --root <WanGP repo> --transport streamable-http --allow-read-file-system
```

The previous `python -m shared.mcp_server` entry point remains supported for existing client configurations.

Set `--job-event-limit 0` when the MCP client only needs terminal job state/results. Deepy opens its in-process WanGP MCP session in this mode because WanGP's UI already displays live generation progress.

The server keeps one warm `WanGPSession`, so agents can perform discovery and multiple generations in a row without starting a new WanGP process each time.

HTTP transports expose short-lived, one-use media-transfer routes. Call `wangp_create_gallery_upload(filename)` and PUT the raw file bytes to the returned URL; WanGP validates the media, adds images/videos to the Visual Gallery or audio to the Audio Gallery, and selects the new item. Call `wangp_create_gallery_download(media_id)` and GET the returned URL to stream a Gallery item. URLs expire after ten minutes and upload size is capped at 8 GiB. Resolve relative transfer URLs against the MCP server origin. These tools are omitted for stdio transport.

Prompt:

- `wangp_agent`
  - The bundled WanGP operating guide for model discovery, schema inspection, bounded searches, generation, cancellation, and media handling.

Resources:

- `wangp://docs/<document>`
  - Native MCP resources for WanGP Markdown documentation under `docs/`.
- `wangp://docs/settings/prompt-flags`
  - Focused definitions for `image_prompt_type`, `video_prompt_type`, and `audio_prompt_type` flags.
- `wangp://skills/long-story-writing`
  - Long-story workflow using a workspace, bounded ripgrep searches, exact edits, and literal appends. Compatibility servers expose it when long-text support and writable workspace access are enabled; Prime exposes it with its session workspace.
- `wangp://skills/long-generation-prompts`
  - Workflow for prompts longer than 4096 characters or tokens, including blank-line-separated sliding windows. Exposed under the same conditions as `long-story-writing`.

Skill exposure follows workspace support; the tool signatures below remain specific to external MCP v1.

Tools:

- `rg(arguments)`, `edit(file_path, old_string, new_string, replace_all=False)`, `append_text(file_path, text)`
  - Experimental long-document primitives exposed only with the long-text experiment and Deepy read/write access. `rg` runs bounded searches within authorized writable roots; `edit` performs literal exact-match replacement and requires a unique match by default; `append_text` appends literal UTF-8 text and creates a missing file. No patch-line prefixes or implicit newlines are added.

- `wangp_models(query="", filters=None, limit=10, offset=0)`
  - Searches models and returns compact records with supported capabilities and media roles as arrays. `filters` accepts `family`, `base_model_type`, `finetune`, `model_type`, `main_output`, `inputs`, or `name`; string filters support case-insensitive `*` and `?` globs.
- `wangp_model(model_type, view="schema", property=None)`
  - Returns one model's compact `schema`, `definition`, or pristine generation `defaults`. Deepy's compact server previews top-level definition strings longer than 256 characters and appends the exact property name and full character count; pass that root `property` with `view="definition"` to retrieve its complete native value. Standalone MCP definitions remain unabridged. Use the default schema first and request the definition only when exact parameters remain unclear. Availability is intentionally separate from this tool.
- `wangp_model_settings(model_type, setting_id=None)`
  - With only `model_type`, lists saved user settings, accelerator profiles, and presets. Each entry has a prefixed `id` and `type`; pass one returned id as `setting_id` to fetch its full content.
- `wangp_list_loras(model_type, name=None)`
  - Recursively returns locally available `.safetensors` and `.sft` LoRAs for the model family. `name` optionally filters subfolder-relative identifiers using case-insensitive `*` and `?` globs. Returned values are suitable for `activated_loras`; supply corresponding weights through `loras_multipliers`.
- `wangp_list_gallery(media_type="all", limit=50, selected_only=False)`
  - Lists compact summaries of the current session's image, video, and audio Gallery items with one `media_id` per item and selection state. Paths and generation settings are omitted. Set `selected_only=True` to return only the live visual and/or audio selections, including the selected video's current playback time when available.
- `wangp_get_media_settings(media_id=None, path=None)`
  - Returns the generation settings stored in or extracted from exactly one media file. Use `media_id` with a value returned by `wangp_list_gallery`; use the mutually exclusive `path` input only when filesystem reads are enabled.
- `wangp_io(action=None, arguments=None)`
  - With no action, returns the compact allowed action list and accessible roots; pass one action without arguments for its schema, then pass arguments to execute it. Actions cover listing, metadata, ranged UTF-8 reading, text search, literal text writing, directory creation, copying, moving, permanent deletion, safe ZIP creation/extraction, and downloads. `read_text` accepts only its declared line-range arguments and returns `eof` plus `next_line`; unsupported byte-offset parameters are rejected. ZIP extraction rejects traversal and symlink entries and does not replace existing files unless `overwrite=true`. Directory listings accept an implicit-extension `media_type` filter (`all`, `image`, `video`, `audio`, `txt`, or `other`) in addition to `pattern`; typed filters return files only, while `other` excludes supported media and `.txt` files. Listings return `has_more` and `next_offset`; repeat the same list call with `offset=next_offset` to continue. Deepy may shorten a requested page to keep its tool result within the active context-safe output budget. Move and delete accept writable paths only, and deleting a non-empty directory requires `recursive=true`. Deepy's scoped mode uses `@alias/path` and resolves plain paths from `@outputs`; standalone MCP `--mcp-allow-read-file-system` permits absolute reads.
- `wangp_create_gallery_upload(filename)` *(HTTP transports only)*
  - Creates a short-lived HTTP PUT URL. A successful upload returns the new Gallery record and selects it.
- `wangp_create_gallery_download(media_id)` *(HTTP transports only)*
  - Creates a short-lived HTTP GET URL for streaming one registered Gallery item.
- `wangp_list_deepy_templates(tool_id=None)`
  - Lists every settings template available in Deepy's Template Settings section and marks the current default for each tool.
- `wangp_get_deepy_template_settings(tool_id, template)`
  - Returns merged, migrated, and model-filtered non-conflicting `settings` for any named template. Pass `template="default"` to resolve the tool's current default directly. With template properties enabled, inactive General Property defaults are omitted and dimensions, frame count or audio duration, and seed remain template-owned. With template properties disabled, those keys are removed from `settings` and the active defaults are returned separately in `general_properties`.
- `wangp_postprocess(media_id=None, path=None, process=None, parameters=None)`
  - Provide exactly one source: `media_id` for Gallery media, or `path` for a server file when filesystem reads are enabled. With no process, discovers compatible post-processing operations. With a returned process id, queues the operation through WanGP and returns a job id.
- `wangp_toolbox(action=None, arguments=None)`
  - With no action, returns a compact utility list; pass one action without arguments for its exact schema, then pass arguments to execute it. Supports adding authorized image/video/audio files to its Gallery, joint visual inspection of up to five images or video frames, extraction, transcription, resize/crop, audio replacement, video merging, media-detail, and documentation operations using media IDs. `add_to_gallery` accepts one `path` or up to 50 `paths`, including `output_file` values returned by other toolbox actions; it applies visible-history retention once per Gallery before the batch, keeps the entire batch, selects existing items without duplication, does not rewrite source metadata, and does not create generated-media chat cards. Direct paths require filesystem-read permission.
- `wangp_generate(source, wait=False, timeout_s=None, event_limit=20)`
  - Starts a job from a settings dict, task dict, manifest dict, or task list. Media fields accept media IDs returned by `wangp_list_gallery`; direct paths require filesystem-read permission. Terminal results include matching compact `gallery_items` records.
- `wangp_get_job(job_id, event_limit=20)`
  - Polls progress/events/result.
- `wangp_cancel_job(job_id)`
  - Requests cancellation.

Deepy's in-process MCP exposes only the three compact model-data tools above. Standalone MCP clients retain the legacy granular endpoints (`wangp_list_models`, `wangp_search_models`, `wangp_list_model_defs`, `wangp_get_model`, `wangp_get_model_metadata`, `wangp_get_model_availability`, `wangp_list_model_availability`, `wangp_get_default_settings`, and `wangp_get_model_schema`) for compatibility.

## Getting Outputs In Memory

By default, the API gives you output file paths in `result.generated_files`.

If you also want the generated media directly in memory, request it explicitly:

- pass `_api={"return_media": True}` in the submitted task settings
- then read the returned payloads from `result.artifacts`

Important notes:

- `result.generated_files` and `result.artifacts` are not mutually exclusive
  - WanGP still saves normal output files
  - the same completed task can also return in-memory media through `result.artifacts`
- `result.artifacts` is ordered like the submitted tasks for the tasks that actually returned media
- for video, the returned tensor layout is `[C, F, H, W]` with `dtype=uint8`
- for audio, use `artifact.audio_tensor` together with `artifact.audio_sampling_rate`

Minimal example:

```python
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "generate a video",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
        "_api": {"return_media": True},
    }
)

result = job.result()

artifact = result.artifacts[0]
video_tensor = artifact.video_tensor_uint8   # [C, F, H, W], uint8
audio_tensor = artifact.audio_tensor         # optional
audio_sr = artifact.audio_sampling_rate      # optional
saved_path = artifact.path                   # optional saved file path
```

If you only care about the in-memory result, you can ignore `result.generated_files` and work directly from `result.artifacts`.

## Accepted Input Shapes

Relative attachment paths are normalized to absolute paths when the job is submitted.

- For direct settings dictionaries and `.json` settings files, the base is the API caller's current working directory at submit time.
- For `.zip` queue files, WanGP keeps the queue bundle behavior and resolves bundled media from the extracted queue contents.
- A few WanGP string-like fields are normalized for convenience. For example, `force_fps` may be passed as `24` or `"24"`.

### Targeting A Frame Range Inside A Video File

For media inputs that accept a file path, WanGP also supports a virtual-media suffix so you can target only part of a source file without extracting an intermediate clip first.

Syntax:

- `path/to/file.ext|start_frame=123,end_frame=456`

Notes:

- `start_frame` is zero-based
- `end_frame` is inclusive
- the underlying source file stays the same; WanGP just decodes the requested frame range
- this is especially useful for `video_guide`, `video_mask`, and similar video-input fields

Optional audio-track targeting is also supported:

- `path/to/file.ext|start_frame=123,end_frame=456,audio_track_no=2`

Example:

```python
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "generate a video",
        "video_prompt_type": "VG",
        "video_guide": r"F:\ALIENS_t01.mkv|start_frame=57542,end_frame=57782",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
    }
)
```

If you want to build that string programmatically, use `shared.utils.virtual_media.build_virtual_media_path(...)`.

### Optional API Meta Settings

`submit_task(settings, ...)` also accepts a reserved `_api` dictionary inside `settings`. This is API metadata, not a normal WanGP generation setting.

Current keys:

- `_api={"return_media": True}`
  - Requests returned media artifacts in `result.artifacts`.
  - Video outputs return `artifact.video_tensor_uint8` when WanGP can expose a contiguous `uint8` tensor.
  - Audio outputs return `artifact.audio_tensor` and `artifact.audio_sampling_rate` when available.

Example:

```python
job = session.submit_task(
    {
        "model_type": "ltx2_22B_distilled",
        "prompt": "generate a video",
        "resolution": "1280x720",
        "num_inference_steps": 8,
        "video_length": 241,
        "_api": {"return_media": True},
    }
)

result = job.result()
artifact = result.artifacts[0]
video_tensor = artifact.video_tensor_uint8
```

### Late Postprocessing And Media Editing

The API can submit WanGP late postprocessing jobs through the same queue, manifest, and MCP paths as generation jobs. These tasks set `mode` explicitly and do not need a normal generation `model_type`; `generate_media(...)` routes them to `edit_media(...)` or `edit_audio(...)`.

Convenience helpers are available for the common edit task shapes:

- `session.submit_media_postprocessing(media_source, ...)` for image/video postprocessing.
- `session.submit_audio_remux(video_source, ...)` for replacing or generating a video soundtrack.
- `session.submit_audio_postprocessing(audio_source, ...)` for standalone audio edits.

#### Image And Video Postprocessing

`submit_media_postprocessing(...)` accepts one media source plus any combination of temporal upsampling, spatial upsampling, and film grain. Meshes take a 3D Refiner (`spatial_upsampling`) and a Texturer (`temporal_upsampling`):

```python
job = session.submit_media_postprocessing(
    r"C:\media\input.mp4",
    spatial_upsampling="h3facerefine",
    spatial_postprocessing_parameters={"face_count": 2},
    return_media=True,
)

result = job.result()
print(result.generated_files)
```

Postprocessing values use the registered postprocessor value strings:

- `temporal_upsampling`: registered temporal upsamplers such as `rife*2` or `dlssg*4` for videos, or texturers such as `supratex` for meshes.
- `spatial_upsampling`: registered decoded-media upsamplers such as `lanczos*2`, `flashvsr*2`, `coz*4`, or the no-scale visual refiner `h3facerefine`. VAE upsamplers are model-pipeline features and are not accepted for late postprocessing.
- `spatial_postprocessing_parameters` / `temporal_postprocessing_parameters`: the options of the spatial and of the temporal processor, each a dictionary keyed by the parameter ids returned by postprocessing discovery. For example, H3 accepts `prompt`, `face_count` and `face_strength`; the texturers accept `texture_size` and `texture_max_faces`.
- `postprocessing_reference_images`: the reference images shared by both processors: H3's face references, or the object image of Hi3D and the texturers (their `reference_images` parameter).
- `film_grain_intensity` / `film_grain_saturation`: late film grain settings. Film grain is active when intensity is greater than `0`.

At least one postprocessing operation must be selected.

#### Video Audio Remuxing

`submit_audio_remux(...)` edits a video audio track. `postprocess_audio` is a registered audio processor method:

```python
job = session.submit_audio_remux(
    r"C:\media\input.mp4",
    postprocess_audio="custom",
    audio_source=r"C:\media\soundtrack.wav",
    return_media=True,
)

result = job.result()
print(result.generated_files)
```

Common built-in remux methods:

| Method | Use | Main inputs |
| --- | --- | --- |
| `custom` | Replace soundtrack with an existing audio file | `audio_source` |
| `mmaudio` | Generate soundtrack from video content | `postprocess_audio_prompt`, `postprocess_audio_neg_prompt`, `seed`, `repeat_generation` |
| `prismaudio` | Generate soundtrack from video content | `postprocess_audio_prompt`, `postprocess_audio_neg_prompt`, `seed`, `repeat_generation` |
| `seedvc_one_speaker` | Replace one voice track | `replace_voice_sample` |
| `seedvc_two_speakers` | Replace two voice tracks | `replace_voice_sample`, `replace_voice_sample2` |

Model-backed processors must be enabled/configured before use. Voice replacement methods require source audio in the video. Legacy `seedvc` and `seedvc2` values are normalized for compatibility, but new callers should use `seedvc_one_speaker` and `seedvc_two_speakers`.

#### Standalone Audio Postprocessing

`submit_audio_postprocessing(...)` edits an audio file without a video container:

```python
job = session.submit_audio_postprocessing(
    r"C:\media\dialog.wav",
    postprocess_audio="remove_background",
    return_media=True,
)

result = job.result()
artifact = result.artifacts[0]
print(artifact.path, artifact.audio_sampling_rate)
```

Common built-in standalone audio methods:

| Method | Use | Main inputs |
| --- | --- | --- |
| `remove_background` | Remove music/background noise | `audio_source` from the helper's first argument |
| `seedvc_one_speaker` | Replace one speaker in an audio file | `replace_voice_sample` |
| `seedvc_two_speakers` | Replace two speakers in an audio file | `replace_voice_sample`, `replace_voice_sample2` |

#### Raw Edit Tasks

All helper calls build normal task settings, so Plugins, Deepy, saved queues, manifests, and MCP clients can submit edit tasks directly through `submit_task(...)`, `submit_manifest(...)`, or `wangp_generate(...)`.

```python
settings = {
    "mode": "edit_postprocessing",
    "video_source": r"C:\media\input.mp4",
    "temporal_upsampling": "rife*4",
    "spatial_upsampling": "lanczos*2",
    "_api": {"return_media": True},
}

job = session.submit_task(settings)
```

Raw task modes:

| Mode | Source field | Processor fields |
| --- | --- | --- |
| `edit_postprocessing` | `video_source` | `temporal_upsampling`, `spatial_upsampling`, `spatial_postprocessing_parameters` / `temporal_postprocessing_parameters` (each processor's options by id), `postprocessing_reference_images`, `film_grain_intensity`, `film_grain_saturation` |
| `edit_remux` | `video_source` | `postprocess_audio`, `audio_source`, `postprocess_audio_prompt`, `postprocess_audio_neg_prompt`, `replace_voice_sample`, `replace_voice_sample2` |
| `edit_audio` | `audio_source` | `postprocess_audio`, `replace_voice_sample`, `replace_voice_sample2` |

### Single Task

For single-task use, the intended input is the task settings dictionary itself:

```python
settings = {
    "model_type": "qwen_image_20B",
    "prompt": "A red bicycle parked in front of a bakery",
    "resolution": "1024x1024",
    "num_inference_steps": 4,
    "image_mode": 1,
}

job = session.submit_task(settings)
```

### Manifest

`submit_manifest(...)` accepts a list of settings dictionaries:

```python
settings_list = [
    {
        "model_type": "qwen_image_20B",
        "prompt": "A quiet library at sunrise",
        "resolution": "1024x1024",
        "num_inference_steps": 4,
        "image_mode": 1,
    },
    {
        "model_type": "qwen_image_20B",
        "prompt": "A rainy alley with neon signs",
        "resolution": "1024x1024",
        "num_inference_steps": 4,
        "image_mode": 1,
    },
]

job = session.submit_manifest(settings_list)
```

### Saved Queue / Settings File

`submit(...)` also accepts:

- a `.json` settings file path
- a `.zip` saved queue path

Example:

```python
job = session.submit(Path(r"C:\WanGP\my_queue.zip"))
```

## Inspecting and Cancelling Queue Tasks

The Python API can list the connected session's entire generation queue, including tasks added through the WebUI when the session is bound to its live `webui_state`:

```python
queue = session.list_queue()
print(queue["queued_count"])   # Waiting tasks, excluding the running task.
print(queue["running_count"])  # Includes a running task that is being cancelled.
for task in queue["tasks"]:
    print(task["queue_id"], task["status"], task["model_type"], task["prompt"])

# Use the queue_id of the specific task you want to cancel.
session.cancel_queue_task(queue_id)
```

The result contains `total_count`, `queued_count`, `running_count`, and `tasks` in queue order. Each task includes an opaque `queue_id`, one-based `position`, original `task_id`, optional `client_id` (empty for ordinary UI tasks), `model_type`, a prompt preview of up to 320 characters, and `status`: `queued`, `running`, or `cancelling`. Completed tasks are omitted. Counts are per queue task, not per output image, repeat, or batch submission.

Cancelling a waiting task removes it immediately and returns `status="cancelled"`. Cancelling a running task requests its normal abort and returns `status="cancelling"`; poll a fresh list until it disappears. Other tasks in the batch remain queued. Cancelling an unknown or finished ID raises an error without aborting another task. Use the returned `queue_id`, not the displayed numeric `task_id`, which can be reused when a new queue is loaded.

External **MCP v2** exposes the same operations through `wangp_session`:

```python
wangp_session(action="list_queue", arguments={})
wangp_session(action="list_queue", arguments={"summary_only": True})
wangp_session(action="cancel_queue_task", arguments={"queue_id": "<queue_id from list_queue>"})
```

MCP lists use the standard `limit` / `cursor` pagination. `count` is the number of tasks on that page; the three queue counts cover the entire snapshot. Continuations preserve that snapshot. Omit `cursor` to get current counts after a cancellation or completion. `summary_only=true` returns counts without task details.

These two MCP actions are **not exposed to Deepy** or MCP v1. Existing `get_job` / `cancel_job` continue to operate on a submitted MCP job, which can contain multiple queue tasks; `cancel_queue_task` targets just one of those tasks.

Queue access is scoped to the connected session, not every WanGP process on the machine. A standalone MCP server owns its headless session and does not inspect a separately running WebUI; the web UI's built-in MCP server reads and edits that web UI's queue. A WebUI-bound Python API reads that UI's admitted generation queue; submissions still waiting for UI admission are not yet queue entries. HTTP MCP clients call these tools through `/mcp`; there is no separate REST queue route.

## Streaming Events

Each job exposes `job.events`, a `SessionStream`.

The stream yields `SessionEvent` objects:

```python
SessionEvent(
    kind="progress",
    data=ProgressUpdate(...),
    timestamp=1710000000.0,
)
```

Known `kind` values:

- `started`
  - Job accepted and session processing started.
- `progress`
  - Structured progress update.
- `preview`
  - Generation preview update, using the configured RGB or Tiny VAE mode.
- `stream`
  - One stdout/stderr line.
- `status`
  - WanGP status message.
- `info`
  - WanGP informational message.
- `output`
  - Raw output refresh event from WanGP.
- `refresh_models`
  - Raw model-refresh event from WanGP.
- `completed`
  - Final `GenerationResult`.
- `error`
  - One `GenerationError` record.

## Returned Objects

### `GenerationResult`

Returned by `job.result()`:

```python
GenerationResult(
    success=False,
    generated_files=[
        r"C:\WanGP\outputs\clip_001.mp4",
    ],
    errors=[
        GenerationError(
            message="Task 2 failed validation",
            task_index=2,
            task_id=2,
            stage="validation",
        ),
    ],
    total_tasks=3,
    successful_tasks=2,
    failed_tasks=1,
)
```

Fields:

- `success: bool`
  - `True` only when every submitted task completed without error.
- `generated_files: list[str]`
  - Absolute paths to every file generated by the job, including partial-success runs.
- `errors: list[GenerationError]`
  - Structured error records collected during the run.
- `total_tasks: int`
  - Number of tasks submitted in the job.
- `successful_tasks: int`
  - Number of tasks that completed successfully.
- `failed_tasks: int`
  - Number of tasks that failed or were cancelled.

`job.result()` does not raise generation-task failures. Instead, inspect `result.success` and `result.errors`.

### `GenerationError`

Delivered through `error` events, `on_error(...)`, and `GenerationResult.errors`:

```python
GenerationError(
    message="Task 2 did not complete successfully",
    task_index=2,
    task_id=2,
    stage="generation",
)
```

Fields:

- `message: str`
  - Human-readable error message.
- `task_index: int | None`
  - One-based task index when the error is associated with a specific task.
- `task_id: Any`
  - Task identifier from the manifest when available.
- `stage: str | None`
  - Error stage such as `validation`, `generation`, `cancelled`, or `runtime`.

### `ProgressUpdate`

Delivered through `progress` events and `on_progress(...)`:

```python
ProgressUpdate(
    phase="inference",
    status="Prompt 1/1 | Denoising | 7.2s",
    progress=54,
    current_step=4,
    total_steps=8,
    raw_phase="Denoising",
    unit=None,
)
```

Fields:

- `phase: str`
  - Normalized phase. Typical values:
  - `loading_model`
  - `encoding_text`
  - `inference`
  - `decoding`
  - `downloading_output`
  - `cancelled`
- `status: str`
  - Human-readable status string produced by WanGP.
- `progress: int`
  - Estimated percentage from `0` to `100`.
- `current_step: int | None`
  - Completed units in the current phase, starting at zero; not a zero-based sampler index.
- `total_steps: int | None`
  - Total units in the current phase when available (steps, layers or tiles).
- `raw_phase: str | None`
  - Original WanGP phase label before normalization.
- `unit: str | None`
  - Optional progress unit if WanGP provides one.
- `task_id: str | None`, `task_index: int | None`, `task_count: int | None`
  - Stable task client ID, one-based task index and total tasks within this job.
- `window_index: int | None`, `window_count: int | None`
  - One-based generation window and planned window count; null when unavailable.

### `PreviewUpdate`

Delivered through `preview` events and `on_preview(...)`:

```python
PreviewUpdate(
    image=<PIL.Image.Image image mode=RGB size=800x200>,
    phase="inference",
    status="Prompt 1/1 | Denoising",
    progress=54,
    current_step=4,
    total_steps=8,
)
```

Fields:

- `image: PIL.Image.Image | None`
  - The frame strip or image preview. For video previews, this is the first frame, so existing image consumers keep working.
- `video: bytes | None`
  - A complete, silent MP4 preview when Generation Preview uses Tiny VAE video mode; otherwise `None`. Save these bytes with an `.mp4` extension or play them as `video/mp4`.
- `phase`, `status`, `progress`, `current_step`, `total_steps`
  - Same interpretation as `ProgressUpdate`.

### `StreamMessage`

Delivered through `stream` events and `on_stream(...)`:

```python
StreamMessage(
    stream="stdout",
    text="New video saved to Path: C:\\WanGP\\outputs\\clip_001.mp4",
)
```

Fields:

- `stream: str`
  - Usually `stdout` or `stderr`.
- `text: str`
  - One redirected line of console output.

### `SessionEvent`

Generic event wrapper:

```python
SessionEvent(
    kind="stream",
    data=StreamMessage(stream="stdout", text="Model loaded"),
    timestamp=1710000000.0,
)
```

Fields:

- `kind: str`
  - Event type.
- `data: Any`
  - Payload object for that event.
- `timestamp: float`
  - Event creation time.

## Callback Object

You can pass a callback object to `init(...)` / `WanGPSession(...)` as the session default, or pass one directly to `submit(...)`, `submit_task(...)`, or `submit_manifest(...)` for a specific job.

Supported callback methods:

- `on_progress(progress_update)`
  - Called when WanGP emits a structured progress update.
  - Use this for progress bars, step counters, and status text.

- `on_preview(preview_update)`
  - Called when a generation preview is available.
  - Use `image` for still previews or `video` for animated Tiny VAE previews during inference.

- `on_stream(stream_message)`
  - Called for every redirected stdout/stderr line.
  - This is the programmatic equivalent of watching the terminal output.

- `on_status(text)`
  - Called for WanGP status messages.
  - Use this if you want coarse status without parsing full progress objects.

- `on_info(text)`
  - Called for informational messages.

- `on_output(data)`
  - Called for raw WanGP output refresh events.
  - This is a low-level hook and is usually not needed by third-party integrations.

- `on_complete(result)`
  - Called when the job finishes.
  - Receives a `GenerationResult`.

- `on_error(error)`
  - Called each time WanGP reports a task or runtime error.
  - Receives a `GenerationError`.

- `on_event(session_event)`
  - Generic catch-all event hook.
  - Called alongside the specific callback above, not instead of it.

Example:

```python
class Callbacks:
    def on_progress(self, progress):
        print("progress:", progress.progress, progress.phase)

    def on_preview(self, preview):
        if preview.image is not None:
            preview.image.save("latest_preview.png")

    def on_stream(self, line):
        print(f"[{line.stream}] {line.text}")

    def on_complete(self, result):
        print("success:", result.success)
        print("generated:", result.generated_files)

    def on_error(self, error):
        print("error:", error.message)
```

Full signature example:

```python
from shared.api import GenerationError, GenerationResult, PreviewUpdate, ProgressUpdate, SessionEvent, StreamMessage


class VerboseCallbacks:
    def on_progress(self, progress: ProgressUpdate) -> None:
        print("progress", progress.progress, progress.current_step, progress.total_steps)

    def on_preview(self, preview: PreviewUpdate) -> None:
        print("preview", preview.phase, preview.image.size if preview.image is not None else None)

    def on_stream(self, line: StreamMessage) -> None:
        print(line.stream, line.text)

    def on_status(self, text: str) -> None:
        print("status", text)

    def on_info(self, text: str) -> None:
        print("info", text)

    def on_output(self, data: object) -> None:
        print("output", data)

    def on_complete(self, result: GenerationResult) -> None:
        print("success", result.success)
        print("files", result.generated_files)

    def on_error(self, error: GenerationError) -> None:
        print("error", error.stage, error.task_index, error.message)

    def on_event(self, event: SessionEvent) -> None:
        print("event", event.kind)
```

## Cancellation

```python
job = session.submit_task(settings)
job.cancel()
```

Cancellation is cooperative and forwards WanGP's normal abort signal to the active model. A cancelled run completes with `result.success == False` and a cancellation entry in `result.errors`.

## External MCP 2.1 Workspaces

`wangp_workspace()` returns the connection's current assignment. With **MCP Agent Workspace** set to **Create a Workspace per Agent** in Configuration / Agents, each connection gets its own persistent gallery workspace, using the storage mode selected in Outputs. `action="list"` lists the workspaces the agent may use; optional `query`, `limit` and `offset` narrow the results without changing the visible gallery. **Workspaces Agents Can See** in Configuration / Agents sets which ones: their own (the context's workspace and the workspaces it created), their own and the user's ordinary workspaces (the default), or all of them, including Deepy conversation workspaces and other agents' workspaces. The same choice governs `attach`. `wangp_validate` also takes `agent_context`, so it checks media from the agent's workspace. In **Use Selected Workspace** mode, the reply of `wangp_workspace()` for the shared context says so in `assignment`. `action="create", name="Project"` creates an independent agent context. `action="attach", workspace_id="..."` creates a context for an existing collection.

Keep the returned `agent_context`. Pass it on `wangp_generate`, `wangp_postprocess`, `wangp_toolbox`, `wangp_plugins`, `wangp_list_gallery`, `wangp_io` and `wangp_session` calls to resume after reconnecting or to run several logical agents over one connection. Pass it to `wangp_workspace` to inspect that assignment. For example:

```python
wangp_workspace(action="list", query="Film")
wangp_workspace(action="create", name="Film Draft")
wangp_io(agent_context="<returned context>", action="write_text", arguments={"path": "@documents/plan.txt", "text": "Opening shot..."})
wangp_generate(agent_context="<returned context>", settings={"model_type": "<model>", "prompt": "A wide shot of..."})
```

With **Use Selected Workspace**, agents without an `agent_context` share one context bound to the workspace selected when the first agent call arrives; later browsing does not move it. Media creation and editing record into the assigned workspace without switching the WebUI's workspace or gallery tab, and toolbox actions can run during a generation. To work in a workspace the user names, attach to it with `wangp_workspace(action="attach", workspace_id=...)` and pass the returned `agent_context`. Gallery listing, stored settings, inspection, discovery, filesystem IO and polling keep the current view too. Standalone MCP uses the same storage without a browser view; there, actions that add media to a context wait for that context's running job and report it as retryable. `wangp_workspace` returns both structured content and its JSON text representation, like other object-returning tools. Explicit `--output-dir` and a bound agent workspace cannot be combined.

`@outputs` points to the assigned workspace's media folder; `@documents` is a separate writable folder for working files. Workspace storage conversion can change output roots, invalidating transfers issued for the earlier scope. Keep the original context when polling jobs or resuming transfers. Agent contexts organize work; the current shared-user policy still permits querying/attaching to other workspaces and viewing/cancelling common queued jobs. See [Gallery Workspaces](WORKSPACES.md) for storage conversion and backups.

---

> Applies to: Python integrations, Deepy Prime, and external MCP clients. Examples identify the applicable interface and version; the connected MCP server supplies its current tool contracts.
