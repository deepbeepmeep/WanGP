# RunPod Serverless Setup Guide for Wan2GP [Beta Preview]

> [!NOTE]
> **RunPod Serverless support is currently in Beta Preview.**
> Modal is the default stable cloud provider for Wan2GP. The RunPod integration enables testing RTX 4090 and L40S serverless instances using **70 GB Ephemeral Container Disk** to keep ongoing storage costs at **$0.00 when idle**.
> Feedback and contributions from the community are warmly welcome!

## 💡 The 70 GB Container Disk Sweet Spot

Unlike traditional virtual machines that charge continuous hourly fees, RunPod Serverless allows you to allocate up to **70 GB of ephemeral container disk** per worker:

* **Why 70 GB?**:
  - Base container + PyTorch 2.5 + CUDA 12.4: ~14 GB
  - Wan 2.1 14B model weights + text encoders: ~28–32 GB
  - Working memory, scratch frames, audio & video VAE output: ~10–15 GB
  - Total requirement: ~55–60 GB. **70 GB provides safety against out-of-disk errors (ENOSPC).**
* **Why Ephemeral Container Disk instead of Network Volumes?**:
  - Network volumes on RunPod charge continuous 24/7 storage fees (~$0.07–$0.10/GB/month).
  - With **70 GB Container Disk**, when the worker finishes generating and scales to 0, **the storage cost is $0.00**. You only pay for the exact seconds the GPU spends generating.

---

## 🛠️ Step-by-Step Setup

### 1. Test Setup Without Credits (Zero Cost)
You do not need any credits to test the API or explore available GPUs.
Run the included discovery tool:
```bash
python cloud/runpod/test_runpod.py
```
It will guide you to set `RUNPOD_API_KEY` in `.env` and verify your setup.

### 2. Create the Serverless Endpoint on RunPod
1. Go to the [RunPod Console → Serverless](https://www.runpod.io/console/serverless).
2. Click **New Endpoint**.
3. Configure the endpoint:
   - **Container Image**: `your-dockerhub-username/wan2gp-runpod:latest` (or use the Dockerfile in `cloud/runpod/Dockerfile`).
   - **Container Disk**: **`70 GB`** (Safe spot for Wan 2.1 / 14B models).
   - **GPU Selection**: Select **RTX 4090** (~$0.44/hr) or **L40S** (~$0.74/hr).
   - **Active Workers (Min)**: Set to `0` (so it scales to zero when idle = $0 cost).
   - **Max Workers**: `2` or `3`.
   - **Idle Timeout**: Set to `300` (5 minutes) so follow-up video generations stay warm.
   - **Network Volume**: Leave **unattached** for $0 idle cost mode, or attach if you prefer persistent caching.

### 3. Connect Wan2GP to RunPod
Add your endpoint ID and API key to `.env`:
```env
WAN2GP_DEFAULT_PROVIDER="runpod"
RUNPOD_API_KEY="your_runpod_api_key"
RUNPOD_ENDPOINT_ID="your_endpoint_id"
```

### 4. Launch Wan2GP
- **Linux / macOS**:
  ```bash
  ./run_cloud.sh --provider runpod
  ```
- **Windows**:
  ```cmd
  run_cloud.bat --provider runpod
  ```

Outputs are automatically generated on the cloud GPU, streamed back to your machine, saved into `outputs/`, and displayed directly in the Wan2GP gallery.
