# DLSS 5 optional runtime installation

> [!CAUTION]
> **Copyright, license, and security warning:** this optional integration executes native worker binaries with access to your GPU and files; several required runtime components are closed source. WanGP does not audit, authenticate, endorse, or redistribute NVIDIA, ReShade, RenoDX, or other third-party runtime binaries. The WanGP, Merserk, and DLSS5-Feeder MIT licenses do **not** grant rights to copy or redistribute those third-party components. Download only from sources you trust, verify signatures and hashes where available, scan every archive before use, and run it entirely at your own risk. You are responsible for complying with every component's license.

DLSS 5 support is optional and is not installed by the normal WanGP installer. It uses:

- a WanGP depth-aware worker derived from the MIT-licensed [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder);
- a buildable WanGP Frame Generation worker;
- the optional legacy no-depth worker from the MIT-licensed Merserk [dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer);
- separately obtained NVIDIA DLSS runtimes, ReShade with full add-on support, and the RenoDX DLSS 5 add-on.

> [!NOTE]
> On Linux the Windows-only pieces (ReShade, RenoDX, DLSSNR, the D3D12 worker executables) have no equivalents, so Neural Rendering runs through the WSL Windows worker described below, and Frame Generation through the native Linux worker in [Native Linux Frame Generation (experimental)](#native-linux-frame-generation-experimental). `scripts/install_dlss5.sh` and `scripts/install_dlss5.py` install the official NVIDIA DLSS SDK Linux runtimes used by that worker. The remaining sections describe the Windows installation.

## WSL Neural Rendering through the Windows worker

WanGP running in WSL can use **DLSS 5 Neural Rendering** through the existing Windows worker. Python preprocessing runs in WSL, while the Windows worker performs Neural Rendering on the NVIDIA GPU using the Windows driver.

> [!IMPORTANT]
> **WSL users do not need the Linux DLSSG worker or its Linux NVIDIA runtime for Neural Rendering.** Use the Windows installer below. There is no Linux worker to build or install for this WSL workflow, and no NVIDIA Vulkan driver is needed inside WSL. The Linux package in the next section is for the separate experimental native Linux Frame Generation feature.

Keep the WanGP checkout and its `dlss5` runtime folder on a Windows drive accessible from WSL, for example under `/mnt/e/`. Install the Windows runtime with `scripts\install_dlss5.bat` from Windows, as described below, then restart WanGP in WSL. Windows executable interoperability must be enabled in WSL, and the Windows NVIDIA driver and GPU must meet the normal Neural Rendering requirements. The same x1 refinement, upscaling modes, intensity, depth, and motion controls are available.

If the mode is disabled, check the reported missing runtime file, WSL interoperability, and `/usr/lib/wsl/lib/nvidia-smi`. Diagnostics from the Windows worker are still written to `dlss5/host/ReShade.log`. Native Linux without WSL cannot use this Windows-worker path.

On systems with integrated graphics plus an NVIDIA GPU, open **Windows Settings > System > Display > Graphics**, add `dlss5\host\nr-depth-worker.exe`, and select **High performance** with the NVIDIA GPU. Otherwise the worker may select integrated graphics and fail NGX initialization even though CUDA sees the NVIDIA GPU. The same Windows preference applies to `dlss5\dlssg\dlssg-worker.exe` when using that Windows executable.

## Native Linux Frame Generation (experimental)

Linux has an experimental **DLSS Frame Generation** path for NVIDIA RTX GPUs with a compatible NVIDIA driver and Vulkan support. It increases the video frame rate using the same `Original` or `RAFT` motion estimation choices as Windows. Upstream support on Linux is **x2 only**; the native worker built by this checkout (see below) is listed as a second menu entry and unlocks the higher multipliers the capability probe reports for your GPU. Native Linux DLSS 5 Neural Rendering is not available; the WSL Windows-worker option above is separate.

**This Linux package is not needed for WSL Neural Rendering; use the Windows installation above for that workflow.** This path has passed compilation, protocol failure checks, and software Vulkan transfer checks. **Actual DLSS generation and visual quality still require validation on native Linux with a supported NVIDIA GPU.** CUDA working in WSL is not sufficient: the worker must see the NVIDIA GPU through Vulkan, not a software renderer such as llvmpipe.

### Install the compiled Linux worker

Download [WanGP-DLSSG-linux-x86_64-v0.1.0.tar.gz](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/download/wangp-linux-v0.1.0/WanGP-DLSSG-linux-x86_64-v0.1.0.tar.gz) and `SHA256SUMS.txt` from the [experimental Linux release](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/tag/wangp-linux-v0.1.0). The compiled worker requires Linux x86_64, glibc 2.34 or newer, `libstdc++.so.6` with `GLIBCXX_3.4.32` or newer, and the Vulkan loader (`libvulkan.so.1`). Ubuntu 24.04 is an example of a compatible userspace; the NVIDIA driver and GPU must also support Vulkan Frame Generation.

With both downloads in the WanGP root, verify the archive and extract it into `dlss5`:

```bash
sha256sum -c SHA256SUMS.txt
mkdir -p dlss5
tar -xzf WanGP-DLSSG-linux-x86_64-v0.1.0.tar.gz -C dlss5
```

The archive supplies `dlssg/dlssg-worker`, license notices, and installation information. **It does not contain NVIDIA's runtime shared library.** Obtain that file from the pinned official NVIDIA SDK and review its `LICENSE.txt`:

```bash
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/DLSS.git NVIDIA-DLSS
git -C NVIDIA-DLSS sparse-checkout set include lib/Linux_x86_64
git -C NVIDIA-DLSS checkout 9981c6350dc58c0ce21f024fd65e31e851c31689
cp NVIDIA-DLSS/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.310.9.16 dlss5/dlssg/
cp NVIDIA-DLSS/LICENSE.txt dlss5/dlssg/LICENSE-NVIDIA-DLSS-Linux.txt
./dlss5/dlssg/dlssg-worker --probe
```

Restart WanGP after installation, then select **DLSS Frame Generation (x2)** in temporal postprocessing. A failed capability probe keeps the option disabled and reports the missing requirement. This package does not install Windows Neural Rendering dependencies.

### Build the Linux worker yourself

Use a C++17 compiler and Vulkan development headers with `VK_NV_optical_flow` support (1.3.239 or newer). On Ubuntu 24.04, install build dependencies with `sudo apt install g++ libvulkan-dev git`. After obtaining the NVIDIA SDK above, clone the worker release and build directly into your WanGP runtime directory:

```bash
git clone --branch wangp-linux-v0.1.0 --depth 1 https://github.com/DeepBeepMeep/dlss5-visual-enhancer.git
bash dlss5-visual-enhancer/native/WanGP-Adapter/build_linux.sh /path/to/NVIDIA-DLSS /path/to/WanGP/dlss5/dlssg
./dlss5/dlssg/dlssg-worker --probe
```

The build installs the worker, NVIDIA runtime, and license notices together. See the [worker build instructions](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/blob/wangp-linux-v0.1.0/native/WanGP-Adapter/README.md) for more details.

For a transfer-only diagnostic, run `./dlss5/dlssg/dlssg-worker --self-test`. This checks Vulkan uploads and readbacks; a pass does **not** establish DLSS support or quality. On supported NVIDIA hardware, check a short clip with motion and scene cuts before processing longer videos. Frame Generation uses estimated motion and a constant depth guide, so fast motion, occlusions and fine details can produce artifacts.

### The native Linux worker with the full multiplier range

Upstream support on Linux is **x2 only**, pinned to `libnvidia-ngx-dlssg.so.310.9.16`. This checkout also carries the source and build scripts for a native Linux worker built from the same MIT-licensed source: `scripts/build_dlssg_worker_linux.sh` with `scripts/dlssg_worker_linux/`, and `scripts/install_dlss5.sh` / `scripts/install_dlss5.py` for the NVIDIA Linux runtimes. Build it with:

```bash
bash scripts/build_dlssg_worker_linux.sh
./dlss5/dlssg-native/dlssg-worker --probe
```

The worker is installed in **`dlss5/dlssg-native/`**, next to a symlink of the newest installed NVIDIA runtime — the worker loads the runtime that sits in its own directory, so each worker keeps its own directory and `dlss5/dlssg/` stays exactly as upstream installs it. No environment variable and no restart is needed: WanGP probes every installed worker and lists them side by side in **Temporal Upsampling**, each label showing the multipliers its worker, runtime and GPU report.

| Menu entry | Worker | Multipliers |
|---|---|---|
| `DLSS Frame Generation` | upstream `dlss5/dlssg/dlssg-worker` + pinned runtime | x2 on native Linux (x2-x4, capability-gated x5/x6 on RTX 50 series on Windows) |
| `DLSS Frame Generation (x3-x6 supported)` | `dlss5/dlssg-native/dlssg-worker` (or a v2 worker in `dlss5/dlssg/`) + newest runtime | probe-driven: x2-x4, x5/x6 on RTX 50 series, capped by the reported `multi_frame_count_max` |

The native entry is named by the multipliers it brings here. **`supported`** is the range the capability probe reported on this GPU; **`max`** is this GPU's ceiling, shown only while the worker cannot be probed at all (missing file, failed probe). **x2 is left out of the native range when a usable upstream entry already offers it** — the native Multiplier list keeps offering x2, so nothing is lost, the label just avoids claiming the same multiplier twice. An x2-only native worker is hidden from the *Temporal Upsampling* menus while the upstream x2 entry works, keeping its multipliers and its validation, so a saved selection asking for it is restored by the menus; it reappears on its own when it is the only path, reading `(x2 supported)`. A blocker follows after `; disabled: `, for instance `(x2-x6 max; disabled: missing libnvidia-ngx-dlssg.so.310.7.0)`, `(x2-x6 max; disabled: Linux probe failed, run dlssg-native/dlssg-worker --probe)` or `(x2-x6 supported; disabled: Vulkan/NGX initialization failed)`; the Multiplier list then only offers probe-verified values. Task status lines and video metadata use the compact form `DLSS Frame Generation x4`.

**Multi-GPU hosts.** The range and the x5/x6 gate follow the GPU WanGP runs on, not the fastest card in the host. WanGP tiers every GPU it may use (`CUDA_VISIBLE_DEVICES`, which `--default-device-id` sets) and keeps the weakest, so a 5090 + 3090 host pinned to the 3090 with `CUDA_VISIBLE_DEVICES=1` reads `(x2-x4 supported)` and refuses x5/x6, while the same host with `CUDA_VISIBLE_DEVICES=0` reads `(x2-x6 supported)`. A host that pins nothing is capped by its oldest card, because WanGP is free to run the job on that one. The native worker is pinned to WanGP's GPU as well (`--gpu <index>`, resolved by name from the worker's own device listing), because the Vulkan ICD ignores `CUDA_VISIBLE_DEVICES` and would otherwise generate the frames on the fastest card; the Frame Generation log line and the probe report both name the GPU in use, the latter as `"gpu":"NVIDIA GeForce RTX 3090"`. Rebuild the worker with `scripts/build_dlssg_worker_linux.sh` to get the pinning and that `gpu` field; an older worker still gets the right range, it just keeps the GPU it picked for itself.

Pick the entry per generation; the two installs coexist and neither one needs a restart to become visible. When a worker is missing, its entry stays disabled and names the missing file. This path is not an upstream PR and is not validated on every driver, so treat it as an experimental feature.

## Automatic installation (Windows)

Close WanGP, then double-click `scripts\install_dlss5.bat`. From a command prompt, the equivalent command is:

```bat
scripts\install_dlss5.bat
```

Read the warning and type `I ACCEPT` to continue. The script downloads all required workers and runtime components, verifies their pinned SHA-256 checksums, extracts them into the root `dlss5` folder, and verifies the NVIDIA signatures on the standard DLSS and Frame Generation DLLs. It does not execute any downloaded installer or runtime during installation. The current worker also suppresses ReShade's unnecessary GitHub update check inside the worker process, so media processing does not need network access.

Existing matching files are kept. If different files already exist, the script changes nothing and asks you to stop WanGP and rerun `scripts\install_dlss5.bat -Force`; that mode backs up every replaced file first. Add `-AcceptThirdPartyRisk` only after reviewing the warning and this document if you need to skip the consent prompt.

The remaining sections document the same installation manually and provide integrity and troubleshooting information.

## Install the worker bundle

Download [WanGP-DLSS5-workers-v1.1.3.zip](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/download/wangp-v1.1.3/WanGP-DLSS5-workers-v1.1.3.zip) from the [DeepBeepMeep dlss5-visual-enhancer fork release](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/releases/tag/wangp-v1.1.3). The ZIP contains the two buildable WanGP workers, the optional legacy Merserk no-depth worker, attribution/license notices, and the required directory structure. It intentionally does **not** contain NVIDIA, ReShade, or RenoDX binaries.

Version 1.1.3 fixes x3 Neural Rendering for portrait outputs that exceed 4320 pixels in height while remaining inside the supported rotated 8K boundary.

```text
Size:   132627 bytes
SHA256: EC470D8EB990CC04FE142C037B2F9E84C1D59A70B111DF51F110767897F5B0C2
```

Create `WanGP/dlss5`, then extract the **contents** of the ZIP into that folder. Do not extract it into `postprocessing/dlss5`, which contains Python source only. The archive entries begin with `host/`, `dlss/`, and `dlssg/`, so the result must look like this after the separately sourced dependencies are added:

```text
WanGP/
|-- dlss5/
|   |-- host/
|   |   |-- nr-depth-worker.exe       # WanGP depth-aware worker
|   |   |-- nvngx.dll                 # legacy Merserk no-depth worker; optional
|   |   |-- dxgi.dll                  # ReShade full add-on build
|   |   |-- renodx-dlss5.addon64      # RenoDX DLSS 5 add-on
|   |   `-- nvngx_dlssnr.dll          # NVIDIA Neural Rendering runtime
|   |-- dlss/
|   |   `-- nvngx_dlss.dll            # NVIDIA DLSS Super Resolution runtime
|   `-- dlssg/
|       |-- dlssg-worker.exe           # WanGP open D3D12 Frame Generation worker
|       `-- nvngx_dlssg.dll            # NVIDIA Frame Generation runtime
```

Keep every runtime component in the indicated subfolder under the single root `dlss5` folder.

## Obtain the third-party dependencies separately

The worker ZIP is not a complete third-party runtime pack. Obtain and place these files yourself:

> [!WARNING]
> The two version-pinned RHI links below are community mirrors provided for convenience. They are not NVIDIA or RenoDX official releases. In particular, `nvngx_dlssnr.dll` 310.8.SF-v2 is a modified, unsigned NVIDIA-derived runtime; it is **not** included in the public NVIDIA DLSS SDK. Download and use it only if you accept the security, copyright, and licensing risks and its use is permitted in your jurisdiction. Otherwise, copy a genuine NVIDIA-signed `nvngx_dlssnr.dll` from licensed software you own that includes it. Never use random DLL download sites.

| Destination | Component | Download or source |
| --- | --- | --- |
| `dlss5/host/dxgi.dll` | ReShade 64-bit **with full add-on support** | [reshade.me](https://reshade.me/) |
| `dlss5/host/renodx-dlss5.addon64` | RenoDX DLSS 5 add-on 4.70 | [Direct ZIP: `renodx-dlss5_4.70.zip`](https://github.com/RankFTW/rhi-repo/releases/download/renodx-dlss5-4.70/renodx-dlss5_4.70.zip) (community RHI mirror) |
| `dlss5/host/nvngx_dlssnr.dll` | DLSS Neural Rendering 310.8.SF-v2 | [Direct ZIP: `nvngx_dlssnr_310.8.SF-v2.zip`](https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.8.SF-v2/nvngx_dlssnr_310.8.SF-v2.zip) (community-modified, unsigned) |
| `dlss5/dlss/nvngx_dlss.dll` | NVIDIA DLSS Super Resolution | [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) or another authorized NVIDIA distribution |
| `dlss5/dlssg/nvngx_dlssg.dll` | NVIDIA DLSS Frame Generation | [NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS) or another authorized NVIDIA distribution |

For the tested Neural Rendering setup:

1. Extract `renodx-dlss5.addon64` from `renodx-dlss5_4.70.zip` into `WanGP/dlss5/host`.
2. Extract `nvngx_dlssnr.dll` from `nvngx_dlssnr_310.8.SF-v2.zip` into the same folder.

The archive SHA-256 values are:

```text
renodx-dlss5_4.70.zip:          D6E356D01B429AF6288F488A4926C44F1D779A7D4586EE8C79D04D3A09A536E6
nvngx_dlssnr_310.8.SF-v2.zip:  1DA35941894994EB087E017577829E492454E9BAE3A6A9397027069CEB74955C
```

The file names on the public [RenoDX GitHub releases](https://github.com/clshortfuse/renodx/releases) do not currently include this generic DLSS 5 add-on. Newer development builds are announced in the [official RenoDX Discord](https://discord.com/invite/renodx); the links above stay pinned to the versions WanGP tested.

NVIDIA components are governed by the [NVIDIA RTX SDK License](https://github.com/NVIDIA/DLSS/blob/main/LICENSE.txt). Review ReShade, RenoDX, and RHI licensing at their source before copying or redistributing their binaries.

## Build the WanGP workers yourself

Install Visual Studio 2022 C++ build tools and a Windows SDK, then clone NVIDIA's official DLSS SDK repository. From the WanGP root, run:

```powershell
git clone https://github.com/NVIDIA/DLSS C:\temp\NVIDIA-DLSS
powershell -ExecutionPolicy Bypass -File native\dlss5\build.ps1 -NgxSdk C:\temp\NVIDIA-DLSS
```

The script writes `dlss5/host/nr-depth-worker.exe` and `dlss5/dlssg/dlssg-worker.exe`. Pass `-Target nr` or `-Target dlssg` to build only one. See `native/dlss5/LICENSE-DLSS5-Feeder` for attribution.

## Tested component integrity

These hashes identify the exact files tested with the v1.1.3 worker bundle. They do not establish safety or redistribution rights.

| File | SHA-256 | Windows signature |
| --- | --- | --- |
| `dlss/nvngx_dlss.dll` | `C85F971CE023C9F3492FC7455F0B01A24BA18EA39636407A846902C4360B0B7E` | Valid, NVIDIA Corporation |
| `dlssg/nvngx_dlssg.dll` | `135EAF0733C1E37381A8C28ABCF7A862404A54132B81787C04E35D09EFC5E36F` | Valid, NVIDIA Corporation |
| `host/nr-depth-worker.exe` | `F8E2967912E5D596E8E36049370487B83620B0CB5845937B681CF835BAFC6D0B` | Unsigned, buildable from the fork source |
| `dlssg/dlssg-worker.exe` | `D93084633E0AAB4A08C43A5EE240176716EF73D87F06F35C2293509FBFC8BD00` | Unsigned, buildable from the fork source |
| `host/dxgi.dll` | `0CEE63F9C9F13F3AC909C5B4903F4DBB4B719A7AB3B4F13B0DEAF83C814B94F7` | Unsigned |
| `host/nvngx.dll` | `58191F4D38288C6BFBDA47EF56911D32052A9789E65714F4583F426E01464638` | Unsigned |
| `host/nvngx_dlssnr.dll` | `6EB209E764F39872625DEBD6ABAF45E2BB6322F6F270F781F70C059AE30B3927` | Unsigned, community-modified SF-v2 |
| `host/renodx-dlss5.addon64` | `D5ADF82EB44B065F4C590AC91FE824BAB07AFEA0EB9F994BDE936710C8593952` | Unsigned, RenoDX 4.70 community mirror |

`nr-depth-worker.exe` is compiled from the included source and has a release-specific hash in the worker ZIP's `SHA256SUMS.txt`. Prefer rebuilding it yourself when you need source-to-binary assurance.

Before installation, scan the downloaded archives and the extracted directory with current security software. Microsoft Defender reported no detections for the development runtime on 3 September 2026; that result is informational, not a safety guarantee.

## Linux

The Windows-only components of DLSS 5 — ReShade, the RenoDX add-on, the DLSSNR (Neural Rendering) runtime, and the D3D12 worker executables — have no Linux builds, so on Linux the installer installs the **official NVIDIA DLSS SDK Linux runtimes** instead. These come straight from the public [NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) repository, pinned to tag `v310.7.0` and SHA-256 verified per file; no community mirrors are used and nothing is executed during installation. The files are governed by the NVIDIA RTX SDKs License (installed as `LICENSE-NVIDIA-DLSS.txt`): NVIDIA-GPU-only use, no standalone redistribution.

```bash
scripts/install_dlss5.sh              # uses the venv Python when present
scripts/install_dlss5.sh --force      # also replace files whose hash differs from the pin
```

Installed layout under `WanGP/dlss5` (pinned SHA-256 values are embedded in `scripts/install_dlss5.py`):

```text
WanGP/
|-- dlss5/
|   |-- LICENSE-NVIDIA-DLSS.txt
|   |-- README-LINUX.txt
|   |-- dlss/
|   |   |-- libnvidia-ngx-dlss.so.310.7.0        # DLSS Super Resolution runtime (CUDA/Vulkan API)
|   |   |-- LICENSE-NVIDIA-DLSS.txt
|   |-- dlssg/
|   |   |-- libnvidia-ngx-dlssg.so.310.7.0       # DLSS Frame Generation runtime (CUDA/Vulkan API)
|   |   |-- LICENSE-NVIDIA-DLSS.txt
|   |   |-- dlssg-worker                          # Linux Vulkan Frame Generation worker (built by scripts/build_dlssg_worker_linux.sh)
|   |-- sdk/
|       |-- include/...                           # NVSDK NGX API headers (D3D/CUDA/Vulkan)
|       |-- lib/libnvsdk_ngx.a                    # static import library
|       |-- lib/libnvidia-ngx-dlssd.so.310.7.0    # legacy DLSS Dynamic feature library
|       |-- LICENSE-NVIDIA-DLSS.txt
```

**What this enables on Linux**

- **DLSS 5 Neural Rendering: unavailable.** The Neural Rendering runtime and the ReShade/RenoDX hook exist only for Windows, so the mode stays disabled on Linux and its dropdown label reads `Windows only: ReShade/RenoDX`.
- **DLSS Frame Generation: worker available (build required).** The official `libnvidia-ngx-dlssg.so` runtime is installed, and WanGP now ships a Linux (Vulkan) build of the MIT-licensed [`native/WanGP-Adapter/dlssg_worker.cpp`](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/blob/wangp-v1.1.3/native/WanGP-Adapter/dlssg_worker.cpp) as [`scripts/dlssg_worker_linux/dlssg_worker_linux.cpp`](../scripts/dlssg_worker_linux/dlssg_worker_linux.cpp). It calls the documented NVSDK NGX Frame Generation Vulkan API from `dlss5/sdk`, keeps the Windows worker's stdin/stdout protocol byte-for-byte, and links the bundled runtime next to itself (`RUNPATH=$ORIGIN`). Build it with:

  ```bash
  scripts/build_dlssg_worker_linux.sh    # -> dlss5/dlssg/dlssg-worker
  ```

  The build needs an NVIDIA Vulkan ICD (`/etc/vulkan/icd.d/`), the Vulkan loader (`libvulkan.so.1`), and g++; it uses the distro Vulkan headers (`libvulkan-dev`) or fetches the official [KhronosGroup/Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) of the loader's 1.3 series. Once the `dlss5/dlssg/dlssg-worker` binary exists (executable), WanGP detects it automatically, probes its capabilities, and enables x2–x4 (x5/x6 on RTX 50 when the runtime reports them). Until it is built, the mode is labelled `missing dlssg-worker`.

## Hardware and diagnostics

Neural Rendering requires Windows 11 and GeForce RTX 30 or newer; RTX 30 is experimental, while RTX 40/50 are the primary targets. Frame Generation requires GeForce RTX 40 or newer, a compatible driver, and Hardware-accelerated GPU scheduling (HAGS). WanGP offers 2x through 4x on compatible RTX 40/50 GPUs and only offers 5x and 6x on RTX 50 GPUs when supported by the installed runtime.

Restart WanGP after installing or replacing the runtime. Unavailable modes are labelled with the missing requirement in their dropdown. WanGP respects an explicitly disabled HAGS setting; if Windows does not expose that setting reliably, the native DLSS capability probe decides availability instead of reporting a false `HAGS disabled` (HAGS is a Windows concept; on Linux the probe decides availability from the runtime). For additional Frame Generation diagnostics, run `dlss5/dlssg/dlssg-worker.exe --probe` (Windows) or `dlss5/dlssg/dlssg-worker --probe` (Linux) from the `dlss5/dlssg` directory. Neural Rendering writes diagnostic information to `dlss5/host/ReShade.log` (Windows only); the Linux Frame Generation worker writes its diagnostics to stderr.

`nr-depth-worker.exe` v1.1.2 or newer does not require network access while processing media and may safely be denied outbound access. Older workers appear to contact GitHub because ReShade performs an automatic version check inside the worker process; v1.1.2 disables that check.

## Recorded-video depth and motion guides

Recorded videos do not contain the depth and motion information normally supplied by a game engine. WanGP estimates these guides automatically for DLSS 5 processing.

Open **Config > Extensions > Spatial Upsamplers / Visual Refiners** to configure both DLSS 5 paths:

- **DLSS 5 Depth Resolution Precision**: `Full Res`, `Half Res` (default), or `Quarter Res`. Lower resolutions reduce depth-estimation time and memory use, at a possible cost to fine depth detail.
- **DLSS 5 Motion Vector**: `Original` (default, faster) or `RAFT` (slower, generally better quality). This choice applies to both Neural Rendering and Frame Generation.

The Postprocessing, Late Postprocessing, and Media Flow controls expose **DLSS 5 NR Intensity** from `0.0` through `2.0`, with a default of `1.0`.

Because these guides are estimated from the video, results can differ from DLSS integrated directly into a game engine.

---

> Applies to: Installation and diagnostics for optional DLSS 5 on Windows, Neural Rendering from WSL using the Windows worker, and experimental DLSS Frame Generation on Linux. Available processors and generation options depend on the installed runtime and compatible NVIDIA hardware.
