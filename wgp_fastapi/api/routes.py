import asyncio
import glob
import logging
import os
import platform
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional, List
import threading
import time

# Low-quality override: when set, all i2v / flux_image / krea_image calls use low_quality GGUF models
W2GP_LOW_QUALITY = bool(os.environ.get("_W2GP_LOW_QUALITY_OVERRIDE"))

# Low-res override: when set, i2v/v2v capped to 256px max side (preserving aspect ratio)
W2GP_LOW_RES = bool(os.environ.get("_W2GP_LOW_RES_OVERRIDE"))

# Performance profile override (-1 = default, 0-3 = profile number)
W2GP_PROFILE = os.environ.get("_W2GP_PROFILE", "-1")

# Import WanGPSession from shared.api
from wgp_fastapi.api.memory_utils import TaskMemoryTracker, log_task_stats
import sys

from numpy import asarray
from starlette.responses import Response

from wgp_fastapi import upscaler

# On Windows, use SelectorEventLoop instead of ProactorEventLoop to avoid
# OSError [WinError 64] "The specified network name is no longer available"
# when clients connect and immediately disconnect during accept.
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

output_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "outputs")
)

from wgp_fastapi.models import (
    ImageToVideoRequest,
    FluxImageRequest,
    FluxImageResponse,
    TaskStatus,
    FluxImageModel,
    MagicMaskResponse,
    NsfwImageResponse,
    NsfwTextRequest,
    NsfwTextResponse,
    PointMaskResponse,
    PromptEnhancerModel,
    PromptEnhanceRequest,
    PromptEnhanceResponse,
    DeepyMessageRequest,
    DeepyMessageResponse,
    KreaImageRequest,
    KreaImageResponse,
    KreaImageModel,
    InpaintPreset,
)
from wgp_fastapi.models.v2v import (
    V2VModel,
    V2VLoRA,
    VideoToVideoRequest,
)
from wgp_fastapi.models.audio2video import (
    Audio2VideoModel,
    Audio2VideoRequest,
)

# --- Suppress noisy access logs for high-frequency endpoints ---
_SUPPRESSED_PATHS = {"/status"}

class _EndpointFilter(logging.Filter):
    """Drop uvicorn access log lines for noisy paths (e.g. health checks, polling)."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args and len(record.args) >= 3:  # type: ignore
            path = record.args[2]  # type: ignore
            return path not in _SUPPRESSED_PATHS and not path.startswith("/api/v1/tasks/")
        return True

logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())


_no_swagger = bool(os.environ.get("_W2GP_NO_SWAGGER"))

app = FastAPI(
    title="FluxMotion FastAPI",
    description="FastAPI wrapper for Wan2GP - Facilitates communication with the FluxMotion mobile app",
    version="1.0.0",
    docs_url=None if _no_swagger else "/docs",
    redoc_url=None if _no_swagger else "/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Disable caching for all responses. Without an explicit Cache-Control, browsers
# heuristically cache responses (e.g. /files images, /status) and can keep serving
# stale copies that lack the CORS headers, causing false CORS failures after config
# changes. This middleware is added after CORSMiddleware so it also covers preflights.
@app.middleware("http")
async def no_store_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response

# Global session instance and task tracking
_wgp_session: Optional["WanGPSession"] = None
_tasks: dict = {}  # task_id -> {"status": str, "result": GenerationResult, "job": SessionJob, "settings": dict}

# Async queue system
_async_queue: list = []  # Queue of pending task_id in order
_current_task_id: Optional[str] = None  # Currently executing task
_worker_running: bool = False  # Worker thread flag
_worker_lock = threading.Lock()

# Task states
TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_SUCCESS = "success"
TASK_FAILED = "failed"

# Status tracking for /status endpoint
_download_in_progress: bool = False  # Whether a model is currently being downloaded
_model_loading: bool = False         # Whether a model is currently being loaded
_model_load_message: str = ""        # Human-readable status message for model loading


def _get_wgp_session():
    """Get or create the WanGPSession instance."""
    global _wgp_session, _model_loading, _model_load_message

    # Import here to avoid issues with module loading order
    from shared.api import WanGPSession

    if _wgp_session is None:
        _model_loading = True
        _model_load_message = "Initializing WanGP runtime..."
        cli_args = []
        if W2GP_PROFILE != "-1":
            cli_args.extend(["--profile", W2GP_PROFILE])
        print(f"[WAN2GP] Initializing session with cli_args={cli_args}")
        _wgp_session = WanGPSession(
            root=str(project_root),
            console_output=True,
            console_isatty=False,
            cli_args=cli_args,
        ).ensure_ready()
        # Verify profile was applied inside the runtime
        try:
            runtime = _wgp_session._ensure_runtime()
            mod = runtime.module
            fpn = getattr(mod, "force_profile_no", "N/A")
            dpv = getattr(mod, "default_profile_video", "N/A")
            api = getattr(mod, "args", None)
            profile_arg = getattr(api, "profile", "N/A") if api else "N/A"
            print(f"[WAN2GP] Profile check — args.profile={profile_arg}, force_profile_no={fpn}, default_profile_video={dpv}")
        except Exception as e:
            print(f"[WAN2GP] Profile verification failed: {e}")
        _model_loading = False
        _model_load_message = ""

    return _wgp_session


def _remux_v2v_audio(settings: dict, result) -> "GenerationResult":
    """Extract audio from the source video and combine it with the generated output."""
    import uuid

    from shared.api import GenerationResult

    try:
        from shared.utils.audio_video import (
            extract_audio_tracks,
            combine_video_with_audio_tracks,
            cleanup_temp_audio_files,
        )

        source_video = settings["video_guide"]
        audio_tracks, audio_metadata = extract_audio_tracks(source_video, temp_format="wav")

        if not audio_tracks:
            return result

        remuxed_files = []
        for i, output_video in enumerate(result.generated_files):
            save_path = os.path.dirname(output_video)
            suffix = f"_v2v_{i}_{uuid.uuid4().hex[:8]}" if len(result.generated_files) > 1 else f"_v2v_{uuid.uuid4().hex[:8]}"
            name, ext = os.path.splitext(os.path.basename(output_video))
            remuxed_path = os.path.join(save_path, f"{name}{suffix}{ext}")
            combine_video_with_audio_tracks(
                output_video,
                audio_tracks,
                remuxed_path,
                audio_metadata=audio_metadata,
            )
            remuxed_files.append(remuxed_path)

        cleanup_temp_audio_files(audio_tracks)

        print(f"[QUEUE WORKER] Audio remuxed {len(remuxed_files)} file(s)")
        return GenerationResult(
            success=True,
            generated_files=remuxed_files,
            errors=[],
            total_tasks=result.total_tasks,
            successful_tasks=result.successful_tasks,
            failed_tasks=result.failed_tasks,
        )
    except Exception as e:
        print(f"[QUEUE WORKER] Audio remux failed (returning original): {e}")
        return result


def _is_oom_error(exc) -> bool:
    """Check if an error (exception or GenerationError) is an out-of-memory error."""
    return "it is likely that you have unsufficient" in str(exc).lower()


def _process_queue_worker():
    """Background worker that processes the async queue one task at a time."""
    global _current_task_id, _worker_running, _model_loading, _model_load_message

    with _worker_lock:
        if _worker_running:
            return
        _worker_running = True

    try:
        while True:
            # Get next task from queue
            with _worker_lock:
                if not _async_queue:
                    _current_task_id = None
                    break
                task_id = _async_queue.pop(0)
                _current_task_id = task_id

            task = _tasks.get(task_id)
            if not task:
                continue

            # Mark as running
            task["status"] = TASK_RUNNING

            # Set model loading state at task start; will be cleared once
            # we receive first progress/preview event (loading → generating)
            task_loading = True
            _model_loading = True
            _model_load_message = "Loading model..."

            try:
                session = _get_wgp_session()

                # Set up memory tracking for this task
                tracker = TaskMemoryTracker()
                task_start = time.monotonic()

                # Replace -1 seed with an actual random seed so task status
                # returns the real seed used, not -1
                settings = task["settings"]
                if settings.get("seed", -1) == -1:
                    import random

                    settings["seed"] = random.randint(0, 999999999)

                print(
                    f"[QUEUE WORKER] Submitting task {task_id} with settings: {settings}"
                )
                # Submit and store job reference for status polling.
                # Strip FastAPI-internal bookkeeping keys: the shared API validates
                # setting names and rejects non-catalog keys.
                submit_settings = {k: v for k, v in settings.items() if k not in ("id", "_endpoint", "_allow_oom_retry")}
                job = session.submit_task(submit_settings)
                print(f"[QUEUE WORKER] Got job: {job}, job.done: {job.done}")
                task["job"] = job

                # Poll for preview events while waiting for completion
                last_mem_sample = time.monotonic()
                while not job.done:
                    try:
                        event = job.events.get(timeout=0.1)
                        if event:
                            if event.kind == "preview":
                                task["latest_preview"] = event.data
                                if task_loading:
                                    task_loading = False
                                    _model_loading = False
                                    _model_load_message = ""
                            elif event.kind == "progress":
                                task["latest_progress"] = event.data
                                if task_loading:
                                    task_loading = False
                                    _model_loading = False
                                    _model_load_message = ""
                            elif event.kind == "status":
                                # Update model load message from generation status
                                if task_loading and event.data:
                                    _model_load_message = str(event.data)
                    except:
                        pass
                    # Sample memory periodically (every 10s as specified)
                    if time.monotonic() - last_mem_sample >= 10:
                        tracker.sample()
                        last_mem_sample = time.monotonic()

                print(f"[QUEUE WORKER] Job done, getting result...")
                # Final memory sample after job completes
                tracker.sample()
                result = job.result()
                print(
                    f"[QUEUE WORKER] Got result: success={result.success}, files={result.generated_files}, errors={[e.message for e in result.errors]}"
                )

                # For v2v tasks: remux source video audio onto generated output
                if result.success and result.generated_files and settings.get("video_guide"):
                    result = _remux_v2v_audio(settings, result)

                # OOM retry: check result errors for OOM before marking failed
                if not result.success and settings.get("_allow_oom_retry"):
                    oom_msgs = [e.message for e in result.errors if _is_oom_error(e)]
                    if oom_msgs:
                        retry_count = settings.get("_oom_retry_count", 0)
                        if retry_count < 1:
                            print(f"[QUEUE WORKER] OOM detected in result for task {task_id}, retrying (attempt {retry_count + 1})")
                            settings["_oom_retry_count"] = retry_count + 1
                            task["status"] = TASK_PENDING
                            task["job"] = None
                            task.pop("result", None)
                            with _worker_lock:
                                _async_queue.append(task_id)
                            continue

                task["result"] = result
                task["status"] = TASK_SUCCESS if result.success else TASK_FAILED

                # Log task completion statistics
                duration = time.monotonic() - task_start
                endpoint_name = (task["settings"].get("_endpoint") or f"unknown:{task_id[:8]}")
                model_name = settings.get("model_type", "unknown")
                log_task_stats(
                    task_id=task_id,
                    endpoint=endpoint_name,
                    model=model_name,
                    peak_ram_mb=tracker.peak_ram_mb,
                    peak_vram_mb=tracker.peak_vram_mb,
                    duration=duration,
                )
            except Exception as e:
                import traceback

                print(f"Queue worker error for task {task_id}: {e}")
                print(traceback.format_exc())

                from shared.api import GenerationResult

                task["result"] = GenerationResult(
                    success=False,
                    generated_files=[],
                    errors=[],
                    total_tasks=0,
                    successful_tasks=0,
                    failed_tasks=1,
                )
                task["status"] = TASK_FAILED

                # Log stats even for failed tasks
                duration = time.monotonic() - task_start
                endpoint_name = (settings.get("_endpoint") or f"unknown:{task_id[:8]}")
                log_task_stats(
                    task_id=task_id,
                    endpoint=endpoint_name,
                    model=settings.get("model_type", "unknown"),
                    peak_ram_mb=tracker.peak_ram_mb,
                    peak_vram_mb=tracker.peak_vram_mb,
                    duration=duration,
                )
            finally:
                # If the task errored before producing progress/preview, clear loading state
                if task_loading:
                    _model_loading = False
                    _model_load_message = ""
    finally:
        with _worker_lock:
            _worker_running = False
            _current_task_id = None
            _model_loading = False
            _model_load_message = ""


def _queue_task(settings: dict, endpoint: str = "") -> str:
    """Add a task to the queue and start worker if needed. Returns task_id."""
    import uuid

    # Generate task_id and add to settings
    task_id = str(uuid.uuid4())
    settings = dict(settings)  # Don't mutate original
    settings["id"] = task_id
    settings["_endpoint"] = endpoint

    # Add task to tracking - accept settings dict directly
    _tasks[task_id] = {
        "status": TASK_PENDING,
        "result": None,
        "job": None,
        "settings": settings,
    }

    # Add to queue
    _async_queue.append(task_id)

    # Start worker in background if not running
    with _worker_lock:
        if not _worker_running:
            thread = threading.Thread(
                target=_process_queue_worker,
                daemon=True,
                name="flux-queue-worker",
            )
            thread.start()

    return task_id


# Backwards compatibility wrapper for flux-specific usage
def _queue_flux_task(
    flux_request: "FluxImageRequest", image_start_path: str | None = None
) -> str:
    """Queue a flux image task (backwards compatibility wrapper)."""
    settings = flux_request.to_wgp_settings(image_start_path=image_start_path)
    settings["_allow_oom_retry"] = True
    return _queue_task(settings, endpoint="flux-image")


def _get_save_path() -> str:
    """Get the save path from the session."""
    session = _get_wgp_session()
    # Try to get the actual save_path from the runtime module
    try:
        runtime = session._ensure_runtime()
        module = runtime.module
        save_p = getattr(module, "save_path", None)
        if save_p:
            return str(save_p)
    except Exception:
        pass
    # Fall back to config or default
    config = getattr(session, "_output_dir", None)
    if config:
        return str(config)
    # Default to project root/output
    return str(project_root / "output")


def _build_file_url(request: Request, file_path: str) -> str:
    """Build a full URL for a generated file."""
    base_url = str(request.base_url).rstrip("/")
    save_path = _get_save_path()

    # Use pathlib for cross-platform path handling
    from pathlib import Path

    file_p = Path(file_path)

    # Get relative path - works cross-platform
    if file_p.is_absolute():
        try:
            rel_path = str(file_p.relative_to(Path(save_path)))
        except ValueError:
            # File is in different directory - just use basename
            rel_path = file_p.name
    else:
        rel_path = file_path

    # Normalize path separators for URL
    return f"{base_url}/files/{rel_path.replace(os.sep, '/')}"


def _save_upload_file(upload_file: UploadFile, suffix: str = ".png") -> str:
    """Save an uploaded file to a temp directory and return the path.

    Automatically converts non-JPEG images (HEIC, HEIF, etc.) to JPEG using Pillow.
    For non-image files (e.g. video), saves directly without conversion.
    """
    content = upload_file.file.read()

    # Try image conversion; fall through for non-image files (video, etc.)
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()

        from PIL import Image
        import io

        image = Image.open(io.BytesIO(content))
        if image.format not in ("JPEG", "JPG"):
            if image.mode in ("RGBA", "P", "LA"):
                image = image.convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name, "JPEG")
                return tmp.name
    except Exception:
        pass  # Not an image (e.g. video) — save raw

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return tmp.name


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global _wgp_session
    if _wgp_session is not None:
        _wgp_session.close()
        _wgp_session = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get(
    "/api/v1/os",
    summary="Get OS type",
    description="Returns the current operating system type.",
)
async def get_os():
    """Return the operating system type."""
    return {"os": platform.system()}


@app.get("/api/v1/version")
async def get_version():
    """Get the FluxMotion API version from flux_motion.txt."""
    version_path = project_root / "flux_motion.txt"
    if not version_path.exists():
        raise HTTPException(status_code=404, detail="Version file not found")
    version = version_path.read_text().strip()
    return {"version": version, "file": "flux_motion.txt"}


@app.get("/status")
async def server_status():
    """Get the current server status, including whether the server is
    downloading models, loading models, or idle/generating."""
    from shared.utils.download_progress import _current_download
    global _download_in_progress, _model_loading, _model_load_message

    # Check if models are being downloaded (either via magic_mask or via download_models/download_file)
    any_download = _download_in_progress or _current_download.get() is not None

    # Determine the overall server status
    if any_download:
        status = "downloading"
    elif _model_loading:
        status = "loading"
    elif _current_task_id is not None or _worker_running:
        status = "generating"
    else:
        status = "idle"

    return {
        "status": status,
        "download_in_progress": any_download,
        "loading_in_progress": _model_loading,
        "generation_in_progress": _current_task_id is not None or _worker_running,
        "current_task_id": _current_task_id,
        "message": _model_load_message or "",
    }


@app.post(
    "/api/v1/upscale",
    summary="Text to Image Generation",
    description="Generate images from text prompts. Supports Flux 1 Kontext, Flux 1 Dev, and Flux 2 Klein.",
)
async def upscale(scale: int, image: UploadFile = File(None)):
    from PIL import Image

    # save the image to a temporary path
    image_path = _save_upload_file(image, suffix=".png")

    return Response(upscaler.upscale(image_path, scale))


@app.post(
    "/api/v1/i2v",
    summary="Image to Video Generation",
    description="Generate videos from input images and text prompts. Only LTX 2 Video is supported. Returns task_id immediately - poll /api/v1/tasks/{task_id} for status.",
)
async def image_to_video(
    prompt: str,
    seed: int,
    num_inference_steps: int,
    width: int,
    height: int,
    batch_size: int,
    model: str,
    video_length: int,
    guidance_scale: float,
    fps: int,
    image: UploadFile = File(None),
    image_end: UploadFile = File(None),
    low_quality: bool = Form(False),
    accelerator_profile: bool = Form(False),
    first_block_cache: bool = Form(False),
    prompt_enhancer: bool = Form(False),
):
    image_path = None
    image_end_path = None

    try:
        # Handle image uploads
        if image is not None:
            image_path = _save_upload_file(image, suffix=".png")
        if image_end is not None:
            image_end_path = _save_upload_file(image_end, suffix=".png")

        # Create request object
        from wgp_fastapi.models.i2v import I2VVideoModel

        model_enum = I2VVideoModel(model)

        # ponytail: low_quality override forces GGUF models for all i2v calls
        if W2GP_LOW_QUALITY:
            low_quality = True
        # ponytail: low_res override caps video to 256px max side, preserving aspect ratio
        if W2GP_LOW_RES:
            max_side = 256
            scale = max_side / max(width, height)
            width = round(width * scale / 8) * 8
            height = round(height * scale / 8) * 8
            print(f"[LOW_RES] i2v resolution capped to {width}x{height}")
        request_obj = ImageToVideoRequest(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            batch_size=batch_size,
            model=model_enum,
            video_length=video_length,
            guidance_scale=guidance_scale,
            fps=fps,
            low_quality=low_quality,
            accelerator_profile=accelerator_profile,
            first_block_cache=first_block_cache,
            prompt_enhancer=prompt_enhancer,
        )

        # Convert request to WanGP settings
        settings = request_obj.to_wgp_settings(
            image_start_path=image_path,
            image_end_path=image_end_path,
        )

        # Is this not being setup correctly?
        print(f"Settings: {settings}")

        # Enable OOM retry for this endpoint
        settings["_allow_oom_retry"] = True

        # Queue the task and return immediately
        task_id = _queue_task(settings, endpoint="i2v")

        # Return just the task_id immediately for async
        return JSONResponse(content={"task_id": task_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/v2v",
    summary="Video to Video Generation",
    description="Generate/edit videos using Wan2.2 Bernini models. Supports optional reference images and LoRA accelerators. Returns task_id immediately - poll /api/v1/tasks/{task_id} for status.",
)
async def video_to_video(
    prompt: str,
    seed: int,
    num_inference_steps: int,
    width: int,
    height: int,
    batch_size: int,
    model: str,
    guidance_scale: float,
    video: UploadFile = File(...),
    image: UploadFile = File(None),
    lora: Optional[str] = Form(None),
    video_length: Optional[int] = Form(None),
    fps: Optional[int] = Form(None),
):
    video_path = None
    image_ref_path = None

    try:
        # Handle video upload (required)
        video_path = _save_upload_file(video, suffix=".mp4")

        # Derive fps and video_length from source video if not provided
        if fps is None or video_length is None:
            from shared.utils.utils import get_video_info

            src_fps, _, _, src_frames = get_video_info(video_path)
            if fps is None:
                fps = int(src_fps)
            if video_length is None:
                video_length = src_frames
            print(f"[V2V] Derived from source: fps={fps}, video_length={video_length} frames ({video_length / float(fps):.1f}s)")

        # Handle optional reference image
        if image is not None:
            image_ref_path = _save_upload_file(image, suffix=".png")

        # Create request object
        model_enum = V2VModel(model)
        lora_enum = V2VLoRA(lora) if lora else None

        # ponytail: low_res override caps video to 256px max side, preserving aspect ratio
        if W2GP_LOW_RES:
            max_side = 256
            scale = max_side / max(width, height)
            width = round(width * scale / 8) * 8
            height = round(height * scale / 8) * 8
            print(f"[LOW_RES] v2v resolution capped to {width}x{height}")

        request_obj = VideoToVideoRequest(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            batch_size=batch_size,
            model=model_enum,
            video_length=video_length,
            guidance_scale=guidance_scale,
            fps=fps,
            lora=lora_enum,
        )

        # Convert request to WanGP settings
        settings = request_obj.to_wgp_settings(
            video_path=video_path,
            image_ref_path=image_ref_path,
        )

        print(f"V2V Settings: {settings}")

        # Enable OOM retry for this endpoint
        settings["_allow_oom_retry"] = True

        # Queue the task and return immediately
        task_id = _queue_task(settings, endpoint="v2v")

        return JSONResponse(content={"task_id": task_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/audio2video",
    summary="Generate Audio from Video using LTX-2.3",
    description="Add audio/soundtrack to an existing video using LTX-2.3 models. Takes a text prompt and a video file, generates audio based on the video content and prompt. Returns task_id immediately - poll /api/v1/tasks/{task_id} for status.",
)
async def audio_to_video(
    prompt: str,
    seed: int,
    model: str,
    video: UploadFile = File(...),
):
    video_path = None

    try:
        # Handle video upload (required)
        video_path = _save_upload_file(video, suffix=".mp4")

        # Derive fps, resolution, and frame count from source video
        from shared.utils.utils import get_video_info

        src_fps, src_width, src_height, src_frames = get_video_info(video_path)
        fps = int(src_fps)
        video_length = src_frames
        width = src_width
        height = src_height
        print(
            f"[AUDIO2VIDEO] Derived from source: fps={fps}, "
            f"resolution={width}x{height}, video_length={video_length} frames "
            f"({video_length / float(fps):.1f}s)"
        )

        # Create request object
        model_enum = Audio2VideoModel(model)

        request_obj = Audio2VideoRequest(
            prompt=prompt,
            seed=seed,
            model=model_enum,
        )

        # Convert request to WanGP settings
        settings = request_obj.to_wgp_settings(
            video_path=video_path,
            fps=fps,
            video_length=video_length,
            width=width,
            height=height,
        )

        print(f"AUDIO2VIDEO Settings: {settings}")

        # Queue the task and return immediately
        task_id = _queue_task(settings, endpoint="audio2video")

        return JSONResponse(content={"task_id": task_id})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve generated files."""
    from pathlib import Path

    save_path = _get_save_path()
    save_path_p = Path(save_path)

    # Try multiple possible locations
    possible_paths = [
        save_path_p / file_path,
        Path(project_root) / "outputs" / file_path,
        Path("outputs") / file_path,
    ]

    for requested_path in possible_paths:
        requested_path = requested_path.resolve()
        if requested_path.exists():
            return FileResponse(requested_path)

    # Debug: list what directories exist
    debug_info = {
        "save_path": str(save_path),
        "save_path_exists": save_path_p.exists(),
        "project_root": str(project_root),
        "outputs_exists": (Path(project_root) / "outputs").exists(),
        "trying_file": file_path,
    }
    raise HTTPException(status_code=404, detail=f"File not found. Debug: {debug_info}")


@app.post(
    "/api/v1/flux-image",
    response_model=FluxImageResponse,
    summary="Flux Image Generation",
    description="Generate images using Flux 2 Klein 9B model with all available parameters. Supports text-to-image and image-to-image generation. Optionally accepts a mask for inpainting and an inpaint-reference image.",
)
async def flux_image(
    prompt: str,
    seed: int,
    num_inference_steps: int,
    width: int,
    height: int,
    batch_size: int,
    model: FluxImageModel = FluxImageModel.FLUX_2_KLEIN,
    image_prompt_type: str = "I",
    image: UploadFile = File(None),
    activated_loras: Optional[str] = Form(None),
    mask: UploadFile = File(None),
    inpaint_reference: UploadFile = File(None),
    low_quality: bool = Form(False),
) -> FluxImageResponse:
    # ponytail: low_quality override forces GGUF models for all flux_image calls
    if W2GP_LOW_QUALITY:
        low_quality = True
    # ponytail: low_res override forces klein 4b for all flux calls
    if W2GP_LOW_RES:
        model = FluxImageModel.FLUX_2_KLEIN_4B
    flux_request = FluxImageRequest(
        prompt=prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        # batch size internally used for queueing more than 1 request
        batch_size=1,
        model=model,
        image_prompt_type=image_prompt_type,
        activated_loras=activated_loras,
        low_quality=low_quality,
    )

    image_path: str | None = None
    mask_path: str | None = None
    inpaint_reference_path: str | None = None

    try:
        try:
            # Handle image upload
            if image is not None:
                image_path = _save_upload_file(image, suffix=".png")
        except Exception as e:
            print(f"Unable to process the image: {e}")

        try:
            # Handle mask upload
            if mask is not None:
                mask_path = _save_upload_file(mask, suffix=".png")
        except Exception as e:
            print(f"Unable to process the mask: {e}")

        try:
            # Handle inpaint-reference upload
            if inpaint_reference is not None:
                inpaint_reference_path = _save_upload_file(inpaint_reference, suffix=".png")
        except Exception as e:
            print(f"Unable to process the inpaint-reference: {e}")

        # Set mask and inpaint-reference paths on the request
        flux_request.mask_path = mask_path
        flux_request.inpaint_reference_path = inpaint_reference_path

        # first to be processed
        task_id = _queue_flux_task(flux_request, image_path)

        # queue up the rest
        for i in range(1, batch_size - 1):
            _queue_flux_task(flux_request, image_path)

        # Return just the task_id immediately for async - bypass Pydantic serialization
        return JSONResponse(content={"task_id": task_id})
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # Print to console so it shows in run.bat terminal
        print(f"[ERROR] flux-image endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "image_uploaded": image is not None,
                "mask_uploaded": mask is not None,
                "inpaint_reference_uploaded": inpaint_reference is not None,
            }
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post(
    "/api/v1/krea-image",
    response_model=KreaImageResponse,
    summary="KREA Text-to-Image Generation",
    description="Generate images using KREA2 Turbo or RAW models. "
    "Optionally accepts a mask, reference image, and inpaint preset for LanPaint inpainting. "
    "Returns task_id immediately — poll /api/v1/tasks/{task_id} for status.",
)
async def krea_image(
    prompt: str = Form(...),
    seed: int = Form(...),
    num_inference_steps: int = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    batch_size: int = Form(...),
    model: KreaImageModel = Form(default=KreaImageModel.KREA2_TURBO),
    guidance_scale: Optional[float] = Form(default=None),
    mask: UploadFile = File(None),
    inpaint_reference: UploadFile = File(None),
    inpaint_preset: Optional[InpaintPreset] = Form(default=None),
    low_quality: bool = Form(False),
) -> KreaImageResponse:
    """Queue a KREA text-to-image generation task."""
    # When low-res mode is active, redirect to flux-image with klein 4b (lighter model)
    if W2GP_LOW_RES:
        return await flux_image(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            batch_size=batch_size,
            model=FluxImageModel.FLUX_2_KLEIN_4B,
            image_prompt_type="I",
            image=None,
            activated_loras=None,
            mask=mask,
            inpaint_reference=inpaint_reference,
            low_quality=low_quality,
        )

    # ponytail: low_quality override forces GGUF models for all krea_image calls
    if W2GP_LOW_QUALITY:
        low_quality = True
    mask_path = None
    inpaint_ref_path = None

    try:
        if mask is not None:
            mask_path = _save_upload_file(mask, suffix=".png")
        if inpaint_reference is not None:
            inpaint_ref_path = _save_upload_file(inpaint_reference, suffix=".png")

        # Build request object
        krea_request = KreaImageRequest(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            batch_size=1,  # batch_size internally used for queueing >1 request
            model=model,
            guidance_scale=guidance_scale,
            mask_path=mask_path,
            inpaint_reference_path=inpaint_ref_path,
            inpaint_preset=inpaint_preset,
            low_quality=low_quality,
        )

        # Convert to WanGP settings
        settings = krea_request.to_wgp_settings()

        # Enable OOM retry for this endpoint
        settings["_allow_oom_retry"] = True

        # Queue first task
        task_id = _queue_task(settings, endpoint="krea-image")

        # Queue additional copies if batch_size > 1
        for _ in range(1, batch_size - 1):
            _queue_task(KreaImageRequest(
                prompt=prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
                width=width,
                height=height,
                batch_size=1,
                model=model,
                guidance_scale=guidance_scale,
                mask_path=mask_path,
                inpaint_reference_path=inpaint_ref_path,
                inpaint_preset=inpaint_preset,
                low_quality=low_quality,
            ).to_wgp_settings(), endpoint="krea-image")

        # Return task_id immediately
        return JSONResponse(content={"task_id": task_id})

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] krea-image endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "prompt": prompt,
                "model": model,
                "seed": seed,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "batch_size": batch_size,
            }
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post(
    "/api/v1/magic-mask",
    response_model=MagicMaskResponse,
    summary="Magic Mask Generation",
    description="Generate a segmentation mask from an image using text prompts. Returns the masked image URL directly.",
)
async def magic_mask(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(None),
    no_hole: bool = Form(True),
    negative_mask: bool = Form(False),
) -> MagicMaskResponse:
    """Generate a mask from an image using keyword prompts and return the mask image URL."""
    from PIL import Image

    if image is None:
        raise HTTPException(status_code=400, detail="An image file is required.")

    try:
        # Save uploaded file and open as PIL Image
        image_path = _save_upload_file(image, suffix=".png")
        pil_image = Image.open(image_path)

        # Lazy import magic_mask to avoid loading order issues
        from shared import magic_mask as mm

        # Auto-download SAM3 model assets if missing
        global _download_in_progress

        from shared.utils.download import process_files_def

        _download_in_progress = True
        try:
            process_files_def(**mm.query_download_def())
        finally:
            _download_in_progress = False

        # Generate the mask
        background, mask_image, keywords = mm.generate_image_mask(
            pil_image,
            prompt,
            no_hole=no_hole,
            negative_mask=negative_mask,
        )

        # Check if the mask is effectively blank — SAM3 couldn't find anything
        if mask_image.getextrema()[1] <= 5:
            keywords_label = ", ".join(keywords)
            raise HTTPException(
                status_code=422,
                detail=f"SAM3 was unable to generate a mask for '{keywords_label}' on this image. "
                "Try different keywords or a different image.",
            )

        # Save the mask to outputs dir with mask- prefix
        import uuid

        mask_filename = f"mask-{uuid.uuid4()}.png"
        mask_path = os.path.join(output_dir, mask_filename)
        os.makedirs(output_dir, exist_ok=True)
        mask_image.save(mask_path, "PNG")

        # Build the overlay: original image + white mask composited on top
        base_rgba = background.convert("RGBA")
        overlay = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
        overlay.putalpha(mask_image)
        composited = Image.alpha_composite(base_rgba, overlay)

        overlay_filename = f"mask-overlay-{uuid.uuid4()}.png"
        overlay_path = os.path.join(output_dir, overlay_filename)
        composited.save(overlay_path, "PNG")

        # Build the full URLs
        image_url = _build_file_url(request, mask_path)
        overlay_url = _build_file_url(request, overlay_path)

        return MagicMaskResponse(
            image_url=image_url,
            maskOverlay=overlay_url,
            keywords=keywords,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] magic-mask endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "prompt": prompt,
                "image_uploaded": image is not None,
            },
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post(
    "/api/v1/point-mask",
    response_model=PointMaskResponse,
    summary="Point-Click Mask Generation",
    description="Generate a segmentation mask from an image using point clicks. "
    "Each point is [x, y] in pixel coordinates. All points are positive (include) clicks. "
    "Optionally include a text prompt to help the model understand the target object.",
)
async def point_mask(
    request: Request,
    points: str = Form(
        ...,
        description='Point coordinates, e.g. "339, 196" or "339, 196; 400, 250" for multiple',
    ),
    image: UploadFile = File(None),
    prompt: Optional[str] = Form(
        None,
        description="Optional text prompt describing the target object (e.g. 'apple', 'person in red shirt')",
    ),
    no_hole: bool = Form(True),
) -> PointMaskResponse:
    """Generate a mask from an image using click points and return the mask image URL."""
    from PIL import Image
    import numpy as np

    if image is None:
        raise HTTPException(status_code=400, detail="An image file is required.")

    predictor = None
    try:
        # Parse points: "339, 196" or "339, 196; 400, 250"
        try:
            pairs = points.split(";")
            pts = []
            for pair in pairs:
                nums = [n.strip() for n in pair.split(",") if n.strip()]
                if len(nums) != 2:
                    raise ValueError(f"Expected 2 numbers, got {len(nums)}: {pair!r}")
                pts.append([int(nums[0]), int(nums[1])])
        except (ValueError, IndexError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid points format: {e}. Use 'x, y' or 'x1, y1; x2, y2'.")

        if len(pts) == 0:
            raise HTTPException(status_code=400, detail="Points must be a non-empty string of coordinates.")

        pixel_points = pts
        labels = [1] * len(pts)  # all positive

        # Save uploaded file and open as PIL Image
        image_path = _save_upload_file(image, suffix=".png")
        pil_image = Image.open(image_path)
        image_np = np.array(pil_image)

        # Lazy import magic_mask to reuse SAM3 download helper
        from shared import magic_mask as mm

        # Auto-download SAM3 model assets if missing
        global _download_in_progress

        from shared.utils.download import process_files_def

        _download_in_progress = True
        try:
            process_files_def(**mm.query_download_def())
        finally:
            _download_in_progress = False

        # Load SAM3 predictor
        from preprocessing.sam3.preprocessor import (
            load_sam3_mask_predictor,
            encode_sam3_keyword_prompts,
        )

        predictor = load_sam3_mask_predictor(
            include_text_encoder=False,
            postprocess_batch_size=1,
            use_batched_grounding=True,
            manual_model_loading=True,
        )
        predictor.load_model_to_gpu()

        height, width = image_np.shape[:2]

        # Start session with the single image
        import torch

        session_resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": [pil_image],
            "offload_video_to_cpu": False,
        })
        session_id = session_resp["session_id"]

        try:
            # Normalize points to [0, 1] relative coordinates (same as matanyone)
            norm_points = [[
                max(0.0, min(1.0, p[0] / max(width - 1, 1))),
                max(0.0, min(1.0, p[1] / max(height - 1, 1))),
            ] for p in pixel_points]

            # Encode text placeholder or real prompt
            # Text is stored in feature cache; SAM2 point path doesn't use it directly,
            # but encoding the real prompt is correct if the model evolves to combine them.
            text_to_encode = prompt if prompt and prompt.strip() else "<text placeholder>"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            def _to_bf16(v):
                if torch.is_tensor(v):
                    return v.to(dtype=dtype) if v.is_floating_point() else v
                if isinstance(v, dict):
                    return {k: _to_bf16(item) for k, item in v.items()}
                return v

            preencoded = _to_bf16(
                encode_sam3_keyword_prompts(
                    [text_to_encode], keep_text_encoder_loaded=False
                )[text_to_encode]
            )

            # Add point prompt
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else __import__("contextlib").nullcontext()
            with autocast_ctx:
                result = predictor.handle_request({
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "points": torch.as_tensor(norm_points, dtype=dtype),
                    "point_labels": torch.as_tensor(labels, dtype=torch.int32),
                    "obj_id": 1,
                    "rel_coordinates": True,
                    "clear_old_points": True,
                    "preencoded_text_outputs": preencoded,
                })

            # Extract binary mask from outputs (same logic as matanyone _sam3_outputs_to_mask)
            outputs = result.get("outputs", {})
            out_masks = outputs.get("out_binary_masks")
            if out_masks is None:
                raise HTTPException(
                    status_code=422,
                    detail="SAM3 returned no mask for these points. Try different click locations.",
                )

            mask = out_masks.detach().cpu().numpy() if torch.is_tensor(out_masks) else np.asarray(out_masks)
            if mask.ndim == 2:
                mask = mask[None]
            elif mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask[:, 0]
            elif mask.ndim > 3:
                mask = mask.reshape((-1, *mask.shape[-2:]))
            if mask.shape[-2:] != (height, width):
                import cv2
                mask = np.stack([
                    cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    for m in mask
                ], axis=0)
            mask = mask.astype(bool).any(axis=0)

            # Fill holes
            if no_hole:
                from preprocessing.sam3.preprocessor import fill_sam3_binary_mask_holes
                mask = fill_sam3_binary_mask_holes(mask, 2)

            mask_uint8 = mask.astype(np.uint8)

        finally:
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass
            try:
                predictor.shutdown()
            except Exception:
                pass

        # Save mask + overlay (same pattern as magic-mask)
        import uuid

        mask_image = Image.fromarray(mask_uint8 * 255, mode="L")

        mask_filename = f"mask-{uuid.uuid4()}.png"
        mask_path_out = os.path.join(output_dir, mask_filename)
        os.makedirs(output_dir, exist_ok=True)
        mask_image.save(mask_path_out, "PNG")

        base_rgba = pil_image.convert("RGBA")
        overlay = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
        overlay.putalpha(mask_image)
        composited = Image.alpha_composite(base_rgba, overlay)

        overlay_filename = f"mask-overlay-{uuid.uuid4()}.png"
        overlay_path = os.path.join(output_dir, overlay_filename)
        composited.save(overlay_path, "PNG")

        return PointMaskResponse(
            image_url=_build_file_url(request, mask_path_out),
            maskOverlay=_build_file_url(request, overlay_path),
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] point-mask endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "points": points,
                "image_uploaded": image is not None,
            },
        }
        if predictor is not None:
            try:
                predictor.shutdown()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=error_detail)


@app.post(
    "/api/v1/prompt-enhance",
    response_model=PromptEnhanceResponse,
    summary="Prompt Enhancement",
    description="Enhance a text prompt using the prompt enhancer model. "
    "Optionally accepts an image for image-guided enhancement.",
)
async def prompt_enhance(
    prompt: str = Form(
        ...,
        description="The text prompt to enhance",
    ),
    image: UploadFile = File(
        None,
        description="Optional image for image-guided prompt enhancement",
    ),
    model: PromptEnhancerModel = Form(
        default=PromptEnhancerModel.FLORENCE2_LLAMA32,
        description="Prompt enhancer model to use",
    ),
    max_new_tokens: int = Form(
        default=512,
        ge=64,
        le=2048,
        description="Maximum tokens for generation",
    ),
    temperature: float = Form(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    ),
    top_p: float = Form(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    ),
    seed: int = Form(
        default=-1,
        description="Random seed (-1 for random)",
    ),
) -> PromptEnhanceResponse:
    """Enhance a text prompt, optionally guided by an image."""
    try:
        image_bytes = None
        if image is not None:
            image_bytes = image.file.read()

        # Call the service
        from wgp_fastapi.services.prompt_enhance import enhance_prompt

        enhanced = enhance_prompt(
            prompt=prompt,
            model=model,
            image_bytes=image_bytes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed if seed >= 0 else None,
        )

        return PromptEnhanceResponse(
            enhanced_prompt=enhanced,
            model_used=model.display_name,
            seed_used=seed if seed >= 0 else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] prompt-enhance endpoint failed: {e}")
        print(tb)
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
            "context": {
                "prompt": prompt,
                "model": model,
                "image_uploaded": image is not None,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            },
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post(
    "/api/v1/deepy-message",
    response_model=DeepyMessageResponse,
    summary="Deepy LLM Message",
    description="Send a message to the built-in Deepy LLM (Qwen3.5 VL). "
    "Accepts a text prompt and an optional base64-encoded image. "
    "Async: waits until Deepy loads the model and generates a response, then returns it.",
)
async def deepy_message(
    prompt: str,
    image: Optional[str] = Form(
        None,
        description="Base64-encoded image as a data URI (e.g. data:image/png;base64,...)",
    ),
    model: str = Form(
        "qwen35_9b_abliterated",
        description="Model variant: qwen35_9b_abliterated or qwen35_4b_abliterated",
    ),
    max_new_tokens: int = Form(
        512,
        ge=64,
        le=4096,
        description="Maximum tokens to generate",
    ),
    temperature: float = Form(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    ),
    top_p: float = Form(
        0.9,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    ),
    seed: int = Form(
        -1,
        description="Random seed (-1 for random)",
    ),
) -> DeepyMessageResponse:
    """Send a message to Deepy LLM and get a response."""
    try:
        # Parse enhancer_enabled from model string
        enhancer_map = {
            "qwen35_9b_abliterated": 4,
            "qwen35_4b_abliterated": 3,
        }
        enhancer_enabled = enhancer_map.get(model, 4)

        # Decode optional image from data URI
        image_bytes = None
        if image:
            # Strip data URI prefix to get raw base64
            if "," in image:
                image = image.split(",", 1)[1]
            import base64
            image_bytes = base64.b64decode(image)

        # Call the service (it handles loading, generation, and VRAM cleanup)
        from wgp_fastapi.services.deepy_message import run_deepy_message

        response_text, used_seed = run_deepy_message(
            prompt=prompt,
            image=image_bytes,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed if seed >= 0 else None,
            enhancer_enabled=enhancer_enabled,
        )

        return DeepyMessageResponse(
            response=response_text,
            model_used=model,
            seed_used=used_seed,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] deepy-message endpoint failed: {e}")
        print(tb)
        raise HTTPException(status_code=500, detail={"error": str(e), "type": type(e).__name__, "traceback": tb})


@app.post(
    "/api/v1/nsfw-check/text",
    response_model=NsfwTextResponse,
    summary="NSFW Text Check",
    description="Check whether an incoming text prompt is sexual in nature. "
    "Returns true/false. The model auto-downloads to the ckpts folder on first use "
    "and unloads after each call.",
)
async def nsfw_check_text(request: NsfwTextRequest) -> NsfwTextResponse:
    """Check if text is sexual in nature."""
    try:
        from wgp_fastapi.services.nsfw_check import check_text_nsfw

        nsfw = check_text_nsfw(request.text)
        return NsfwTextResponse(nsfw=nsfw)
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] nsfw-check/text endpoint failed: {e}")
        print(tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__, "traceback": tb},
        )


@app.post(
    "/api/v1/nsfw-check/image",
    response_model=NsfwImageResponse,
    summary="NSFW Image Check",
    description="Check whether an uploaded image contains NSFW content. "
    "Returns true/false and, when NSFW, a blurred copy of the image as a base64 "
    "data URL. The model auto-downloads to the ckpts folder on first use "
    "and unloads after each call.",
)
async def nsfw_check_image(image: UploadFile = File(...)) -> NsfwImageResponse:
    """Check if an uploaded image is NSFW; return blurred copy when NSFW."""
    try:
        image_bytes = image.file.read()
        from wgp_fastapi.services.nsfw_check import check_image_nsfw

        try:
            nsfw, blurred = check_image_nsfw(image_bytes)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return NsfwImageResponse(nsfw=nsfw, blurred_image=blurred)
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] nsfw-check/image endpoint failed: {e}")
        print(tb)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "type": type(e).__name__, "traceback": tb},
        )


@app.get(
    "/api/v1/tasks/{task_id}",
    response_model=TaskStatus,
    summary="Get task status",
    description="Retrieve the status of a generation task.",
)
async def get_task_status(request: Request, task_id: str) -> TaskStatus:
    """Get the status of a generation task."""
    from shared.api import GenerationResult

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    status = task["status"]
    settings = task.get("settings", {})
    seed = settings.get("seed")

    # Task still pending
    if status == TASK_PENDING:
        return TaskStatus(
            progress=0,
            preview_image=None,
            finished_image=None,
            seed=seed,
        )

    if status == TASK_RUNNING:
        # Check for progress from stored values (updated by worker) or job events
        progress = 5  # Start at 5% when running but no progress yet
        preview_image = None

        # First check stored preview/progress from worker
        latest_preview = task.get("latest_preview")
        latest_progress = task.get("latest_progress")

        if not latest_preview or not latest_progress:
            # Fall back to polling job events
            job = task.get("job")
            if job:
                events = getattr(job, "events", None)
                if events:
                    try:
                        while True:
                            event = events.get(timeout=0.001)
                            if event is None:
                                break
                            if event.kind == "progress":
                                latest_progress = event.data
                            elif event.kind == "preview":
                                latest_preview = event.data
                    except:
                        pass

        if latest_progress and hasattr(latest_progress, "progress"):
            progress = latest_progress.progress

        if latest_preview and hasattr(latest_preview, "image") and latest_preview.image:
            # Convert PIL image to base64 for preview
            import io
            import base64
            from PIL import Image

            img_buffer = io.BytesIO()
            latest_preview.image.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()
            preview_image = (
                f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
            )

        return TaskStatus(
            progress=progress,
            preview_image=preview_image,
            finished_image=None,
            seed=seed,
        )

    # Task completed - check result
    result: GenerationResult = task.get("result")
    settings = task.get("settings", {})
    seed = settings.get("seed")

    if result and result.success and result.generated_files:
        # Use first generated file as finished_image URL
        finished_file_url = _build_file_url(request, result.generated_files[0])
        return TaskStatus(
            progress=100,
            preview_image=None,
            finished_image=finished_file_url,
            seed=seed,
        )
    else:
        return TaskStatus(
            progress=100,
            preview_image=None,
            finished_image=None,
            seed=seed,
        )


@app.delete(
    "/api/v1/tasks/{task_id}",
    summary="Cancel task",
    description="Cancel a running or queued generation task.",
)
async def cancel_task(task_id: str):
    """Cancel a running or queued generation task."""
    from shared.api import GenerationResult

    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    status = task["status"]

    # Already completed — nothing to cancel
    if status in (TASK_SUCCESS, TASK_FAILED):
        return JSONResponse(
            content={"status": "already_completed", "task_id": task_id},
            status_code=200,
        )

    # Pending task — remove from async queue
    if status == TASK_PENDING:
        with _worker_lock:
            if task_id in _async_queue:
                _async_queue.remove(task_id)
        task["status"] = TASK_FAILED
        task["result"] = GenerationResult(
            success=False,
            generated_files=[],
            errors=[],
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=1,
        )
        return JSONResponse(
            content={"status": "cancelled", "task_id": task_id},
            status_code=200,
        )

    # Running task — cancel via session
    if status == TASK_RUNNING:
        session = _get_wgp_session()
        session.cancel()
        return JSONResponse(
            content={"status": "cancelling", "task_id": task_id},
            status_code=200,
        )

    # Fallback
    raise HTTPException(status_code=500, detail="Unexpected task state")


@app.delete(
    "/api/v1/files/{file_name:path}",
    summary="Delete a file",
    description="Delete a file from the outputs directory by filename.",
)
async def delete_file(file_name: str):
    # find the file path given just the file name alone
    for file in glob.iglob(os.path.join(f"{output_dir}", "**", "*"), recursive=True):
        if file_name in file:
            print(f"Deleting file: {file}")

            if os.name == "nt" or sys.platform == "darwin":
                os.remove(file)
            else:
                os.remove(file.replace("/", "\\\\"))

            return Response(status_code=200)

    # file was not found
    return Response(status_code=404)


@app.delete(
    "/api/v1/files",
    summary="Delete files",
    description="Delete one or more files from the outputs directory by filename. "
    "Accepts a JSON array of filenames in the request body. "
    "Supports all file extension types.",
)
async def delete_files(
    file_names: List[str] = Body(
        ...,
        description="List of filenames to delete (e.g., [\"image.png\", \"video.mp4\"])",
        examples=[["image.png", "video.mp4"]],
    ),
):
    """
    Delete one or more files. Accepts a JSON body with a list of filenames.
    Example body: ["image1.png", "video.mp4", "subfolder/image.jpg"]
    Supports all file extension types.
    """
    if not file_names:
        return JSONResponse(
            content={"error": "No filenames provided. Send a JSON array of filenames."},
            status_code=400,
        )

    results: dict[str, list] = {"deleted": [], "not_found": [], "errors": []}

    # Collect all files once to avoid re-walking the directory
    all_files = list(
        glob.iglob(os.path.join(f"{output_dir}", "**", "*"), recursive=True)
    )

    for file_name in file_names:
        matched = False
        for file_path in all_files:
            if file_name in file_path:
                try:
                    os.remove(file_path)
                    results["deleted"].append(file_name)
                    print(f"Deleted file: {file_path}")
                except Exception as e:
                    results["errors"].append(
                        {"file": file_name, "detail": str(e)}
                    )
                    print(f"Error deleting {file_path}: {e}")
                matched = True
                break  # only delete the first match per filename

        if not matched:
            results["not_found"].append(file_name)

    if not results["deleted"] and not results["errors"]:
        return JSONResponse(content=results, status_code=404)

    return JSONResponse(content=results, status_code=200)


@app.post(
    "/api/v1/files/{file_name:path}/duplicate",
    summary="Duplicate a file",
    description="Copy a file in the outputs directory and return the new URL.",
)
async def duplicate_file(request: Request, file_name: str):
    """Find a file by name in outputs, copy it, return the new URL."""
    for file in glob.iglob(os.path.join(f"{output_dir}", "**", "*"), recursive=True):
        if file_name in file and os.path.isfile(file):
            ext = os.path.splitext(file)[1]
            copy_path = os.path.join(os.path.dirname(file), f"{uuid.uuid4()}{ext}")
            shutil.copy2(file, copy_path)
            print(f"Duplicated file: {file} -> {copy_path}")
            return JSONResponse(
                content={"url": _build_file_url(request, copy_path)}
            )

    raise HTTPException(status_code=404, detail="File not found")


if __name__ == "__main__":
    import uvicorn

    kwargs = {"host": "0.0.0.0", "port": 8000}
    if os.name == "nt":
        # uvicorn 0.36+ builds the loop from its own loop factory (Proactor on
        # Windows) and ignores set_event_loop_policy(). Passing loop="none"
        # makes it fall back to the event loop policy — the
        # WindowsSelectorEventLoopPolicy set at the top of this file — which
        # avoids OSError [WinError 64] on accept.
        kwargs["loop"] = "none"
    uvicorn.run(app, **kwargs)
