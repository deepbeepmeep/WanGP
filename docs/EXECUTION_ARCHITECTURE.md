# Wan2GP Execution Architecture & Cloud Adapter Contract

## Overview

Wan2GP separates **generation configuration** (what is generated) from **generation execution** (where it runs).

This abstraction preserves Wan2GP's existing Gradio UI, parameter normalizations, and memory-managed local generation pipeline (`mmgp`), while enabling execution to run either on the **Local GPU** or via a remote **Cloud Runtime**.

```
                   Wan2GP Gradio UI / CLI / API
                                |
                                v
                        GenerationRequest
                                |
                      get_execution_adapter()
                               / \
                              /   \
                             v     v
          LocalExecutionAdapter   CloudExecutionAdapter
                     |                     |
              Local GPU (Wan2GP)           v
                                    Cloud Runtime API
                                           |
                                           v
                                    Remote GPU Worker
```

---

## Core Principles

1. **Upstream Wan2GP Preservation**:
   - Local GPU generation is the default and uses the exact existing `generate_media(...)` pipeline.
   - Zero changes to model weights, attention kernels, offloading logic, or existing Gradio workflows.

2. **Provider Agnostic Cloud Boundary**:
   - Wan2GP contains **no provider-specific code** (no RunPod, Modal, AWS, or Lambda code).
   - Infrastructure provisioning, worker scaling, volume mounting, and billing are handled by an external **Cloud Runtime** service.
   - Wan2GP communicates with the Cloud Runtime exclusively via a standard HTTP/REST contract.

3. **Reusable Settings (WanStudio Compatibility)**:
   - The normalized `GenerationRequest` format is independent of Gradio UI components.
   - Future applications (such as WanStudio) can produce the exact same `GenerationRequest` JSON and send it directly to the Cloud Runtime or local worker.

---

## Execution Modes

The Gradio UI exposes an **Execution Target** selector:

- **Local GPU** (`local`): Dispatches to `LocalExecutionAdapter`, executing on the machine's local GPU using Wan2GP's `mmgp` engine.
- **Cloud** (`cloud`): Dispatches to `CloudExecutionAdapter`.

If Cloud is selected and the Cloud Runtime is not configured, Wan2GP presents an immediate, descriptive configuration message rather than silently falling back to local execution.

---

## The `GenerationRequest` Contract

`GenerationRequest` is a normalized, JSON-serializable dataclass located in `shared/execution/types.py`.

### Schema (JSON)

```json
{
  "engine": "wan2gp",
  "version": "1.0",
  "task_id": 42,
  "execution_mode": "local",
  "model": "t2v",
  "model_type": "t2v",
  "base_model_type": "t2v",
  "settings": {
    "prompt": "A cinematic shot of a drone flying through a neon canyon",
    "negative_prompt": "blurry, low quality, artifacts",
    "resolution": "1280x720 (16:9)",
    "width": 1280,
    "height": 720,
    "video_length": 81,
    "force_fps": 16,
    "num_inference_steps": 30,
    "guidance_scale": 5.0,
    "seed": 8839210,
    "sample_solver": "unipc",
    "flow_shift": 5.0,
    "activated_loras": ["cinematic.safetensors"],
    "loras_multipliers": [1.0],
    "temporal_upsampling": "rife",
    "spatial_upsampling": "flashvsr"
  },
  "inputs": {
    "image_start": "/path/or/url/to/start.png",
    "image_end": null,
    "image_refs": [],
    "video_source": null,
    "audio_source": null
  },
  "plugin_data": {},
  "metadata": {
    "source": "wan2gp",
    "client_id": "client_abc123",
    "created_at": "2026-10-03T00:30:00"
  }
}
```

### Property Mapping

| Field | Source in Wan2GP | Description |
| :--- | :--- | :--- |
| `prompt` | `settings["prompt"]` | Positive text prompt |
| `negative_prompt` | `settings["negative_prompt"]` | Negative guidance prompt |
| `model_type` | `params["model_type"]` | Model architecture ID (e.g. `t2v`, `i2v_2_2`, `vace_14B`) |
| `width`, `height` | `settings["width"]`, `settings["height"]` | Parsed resolution integers |
| `frames` | `settings["video_length"]` | Number of video frames to sample |
| `steps` | `settings["num_inference_steps"]` | Denoising steps |
| `guidance` | `settings["guidance_scale"]` | Classifier-free guidance multiplier |
| `seed` | `settings["seed"]` | Random seed |
| `inputs` | `ATTACHMENT_KEYS` | File paths or URLs for media conditioning |

---

## The `ExecutionResult` Contract

Adapters return an `ExecutionResult` on completion:

```json
{
  "success": true,
  "status": "completed",
  "task_id": 42,
  "output_files": [
    "/workspace/outputs/video_20261003_003015.mp4"
  ],
  "media_type": "video",
  "metadata": {
    "adapter": "Local GPU Adapter",
    "execution_mode": "local",
    "model_type": "t2v"
  },
  "generation_info": {
    "execution_time_seconds": 18.42
  },
  "error": null,
  "execution_time": 18.42
}
```

Status codes:
- `completed`: Generation finished and outputs are ready.
- `failed`: An error occurred during inference or communication.
- `aborted`: User requested interruption.

---

## Cloud Runtime API Contract

Future Cloud Runtimes (Modal, RunPod, or dedicated clusters) implement these REST endpoints:

### 1. Submit Generation Job
- **Method**: `POST /v1/jobs`
- **Request Body**: `GenerationRequest` JSON
- **Response**:
  ```json
  {
    "job_id": "job_94821a",
    "status": "queued"
  }
  ```

### 2. Poll Job Status
- **Method**: `GET /v1/jobs/{job_id}`
- **Response**:
  ```json
  {
    "job_id": "job_94821a",
    "status": "running",
    "progress": [12, 30],
    "status_message": "Denoising step 12/30",
    "output_files": [],
    "error": null
  }
  ```
  When completed:
  ```json
  {
    "job_id": "job_94821a",
    "status": "completed",
    "output_files": ["https://storage.example.com/output.mp4"]
  }
  ```

### 3. Cancel Job
- **Method**: `POST /v1/jobs/{job_id}/cancel`

---

## Configuration & Environment Variables

| Variable | Description |
| :--- | :--- |
| `WAN2GP_CLOUD_ENDPOINT` | Base URL of the Cloud Runtime (e.g. `http://localhost:8000` or `https://cloud.example.com`). |
| `WAN2GP_CLOUD_API_KEY` | Bearer token / API key for Cloud Runtime authentication. |
| `WAN2GP_CLOUD_TIMEOUT` | Max timeout in seconds for job completion (default: `300`). |
| `WAN2GP_CLOUD_MOCK` | Set to `1` or `true` to enable simulated cloud execution for offline tests. |

---

## Programmatic Usage

### Executing via Python

```python
from shared.execution import (
    GenerationRequest,
    get_execution_adapter,
    ExecutionMode,
)

# 1. Create a request
request = GenerationRequest(
    model_type="t2v",
    execution_mode="local",  # or "cloud"
    settings={
        "prompt": "A futuristic city in mist",
        "resolution": "1280x720",
        "num_inference_steps": 25,
        "seed": 42,
    }
)

# 2. Get the appropriate adapter
adapter = get_execution_adapter(request.execution_mode)

# 3. Execute with progress streaming
def event_callback(cmd, data):
    print(f"[{cmd}] {data}")

result = adapter.execute(request, send_cmd=event_callback)
if result.success:
    print("Generated files:", result.output_files)
else:
    print("Failed:", result.error)
```
