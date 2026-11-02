# DLSS 5 optional runtime installation

> [!CAUTION]
> **Copyright, license, and security warning:** this optional integration executes native worker binaries with access to your GPU and files; several required runtime components are closed source. WanGP does not audit, authenticate, endorse, or redistribute NVIDIA, ReShade, RenoDX, or other third-party runtime binaries. The WanGP, Merserk, and DLSS5-Feeder MIT licenses do **not** grant rights to copy or redistribute those third-party components. Download only from sources you trust, verify signatures and hashes where available, scan every archive before use, and run it entirely at your own risk. You are responsible for complying with every component's license.

DLSS 5 support is optional and is not installed by the normal WanGP installer. It uses:

- a WanGP depth-aware worker derived from the MIT-licensed [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder);
- a buildable WanGP Frame Generation worker;
- the optional legacy no-depth worker from the MIT-licensed Merserk [dlss5-visual-enhancer](https://github.com/Merserk/dlss5-visual-enhancer);
- separately obtained NVIDIA DLSS runtimes, ReShade with full plugin support, and the RenoDX DLSS 5 plugin.

## WSL Neural Rendering through the Windows worker

WanGP running in WSL can use **DLSS 5 Neural Rendering** through the existing Windows worker. Python preprocessing runs in WSL, while the Windows worker performs Neural Rendering on the NVIDIA GPU using the Windows driver.

> [!IMPORTANT]
> **WSL users do not need the Linux DLSSG worker or its Linux NVIDIA runtime for Neural Rendering.** Use the Windows installer below. There is no Linux worker to build or install for this WSL workflow, and no NVIDIA Vulkan driver is needed inside WSL. The Linux package in the next section is for the separate experimental native Linux Frame Generation x2 feature.

Keep the WanGP checkout and its `dlss5` runtime folder on a Windows drive accessible from WSL, for example under `/mnt/e/`. Install the Windows runtime with `scripts\install_dlss5.bat` from Windows, as described below, then restart WanGP in WSL. Windows executable interoperability must be enabled in WSL, and the Windows NVIDIA driver and GPU must meet the normal Neural Rendering requirements. The same x1 refinement, upscaling modes, intensity, depth, and motion controls are available.

If the mode is disabled, check the reported missing runtime file, WSL interoperability, and `/usr/lib/wsl/lib/nvidia-smi`. Diagnostics from the Windows worker are still written to `dlss5/host/ReShade.log`. Native Linux without WSL cannot use this Windows-worker path.

On systems with integrated graphics plus an NVIDIA GPU, open **Windows Settings > System > Display > Graphics**, add `dlss5\host\nr-depth-worker.exe`, and select **High performance** with the NVIDIA GPU. Otherwise the worker may select integrated graphics and fail NGX initialization even though CUDA sees the NVIDIA GPU. The same Windows preference applies to `dlss5\dlssg\dlssg-worker.exe` when using that Windows executable.

## Native Linux Frame Generation x2 (experimental)

Linux has an experimental **DLSS Frame Generation x2** path for NVIDIA RTX GPUs with a compatible NVIDIA driver and Vulkan support. It doubles the video frame rate using the same `Original` or `RAFT` motion estimation choices as Windows. Higher multipliers and native Linux DLSS 5 Neural Rendering are not available; the WSL Windows-worker option above is separate.

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

Restart WanGP after installation, then select **DLSS Frame Generation x2** in temporal postprocessing. A failed capability probe keeps the option disabled and reports the missing requirement. This package does not install Windows Neural Rendering dependencies.

### Build the Linux worker yourself

Use a C++17 compiler and Vulkan development headers with `VK_NV_optical_flow` support (1.3.239 or newer). On Ubuntu 24.04, install build dependencies with `sudo apt install g++ libvulkan-dev git`. After obtaining the NVIDIA SDK above, clone the worker release and build directly into your WanGP runtime directory:

```bash
git clone --branch wangp-linux-v0.1.0 --depth 1 https://github.com/DeepBeepMeep/dlss5-visual-enhancer.git
bash dlss5-visual-enhancer/native/WanGP-Adapter/build_linux.sh /path/to/NVIDIA-DLSS /path/to/WanGP/dlss5/dlssg
./dlss5/dlssg/dlssg-worker --probe
```

The build installs the worker, NVIDIA runtime, and license notices together. See the [worker build instructions](https://github.com/DeepBeepMeep/dlss5-visual-enhancer/blob/wangp-linux-v0.1.0/native/WanGP-Adapter/README.md) for more details.

For a transfer-only diagnostic, run `./dlss5/dlssg/dlssg-worker --self-test`. This checks Vulkan uploads and readbacks; a pass does **not** establish DLSS support or quality. On supported NVIDIA hardware, check a short clip with motion and scene cuts before processing longer videos. Frame Generation uses estimated motion and a constant depth guide, so fast motion, occlusions and fine details can produce artifacts.

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
|   |   |-- dxgi.dll                  # ReShade full plugin build
|   |   |-- renodx-dlss5.addon64      # RenoDX DLSS 5 plugin
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
| `dlss5/host/dxgi.dll` | ReShade 64-bit **with full plugin support** | [reshade.me](https://reshade.me/) |
| `dlss5/host/renodx-dlss5.addon64` | RenoDX DLSS 5 plugin 4.70 | [Direct ZIP: `renodx-dlss5_4.70.zip`](https://github.com/RankFTW/rhi-repo/releases/download/renodx-dlss5-4.70/renodx-dlss5_4.70.zip) (community RHI mirror) |
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

The file names on the public [RenoDX GitHub releases](https://github.com/clshortfuse/renodx/releases) do not currently include this generic DLSS 5 plugin. Newer development builds are announced in the [official RenoDX Discord](https://discord.com/invite/renodx); the links above stay pinned to the versions WanGP tested.

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

## Hardware and diagnostics

Neural Rendering requires Windows 11 and GeForce RTX 30 or newer; RTX 30 is experimental, while RTX 40/50 are the primary targets. Frame Generation requires GeForce RTX 40 or newer, a compatible driver, and Hardware-accelerated GPU scheduling (HAGS). WanGP offers 2x through 4x on compatible RTX 40/50 GPUs and only offers 5x and 6x on RTX 50 GPUs when supported by the installed runtime.

Restart WanGP after installing or replacing the runtime. Unavailable modes are labelled with the missing requirement in their dropdown. WanGP respects an explicitly disabled HAGS setting; if Windows does not expose that setting reliably, the native DLSS capability probe decides availability instead of reporting a false `HAGS disabled`. For additional Frame Generation diagnostics, run `dlss5/dlssg/dlssg-worker.exe --probe` from the `dlss5/dlssg` directory. Neural Rendering writes diagnostic information to `dlss5/host/ReShade.log`.

`nr-depth-worker.exe` v1.1.2 or newer does not require network access while processing media and may safely be denied outbound access. Older workers appear to contact GitHub because ReShade performs an automatic version check inside the worker process; v1.1.2 disables that check.

## Recorded-video depth and motion guides

Recorded videos do not contain the depth and motion information normally supplied by a game engine. WanGP estimates these guides automatically for DLSS 5 processing.

Open **Config > Extensions > Spatial Upsamplers / Visual Refiners** to configure both DLSS 5 paths:

- **DLSS 5 Depth Resolution Precision**: `Full Res`, `Half Res` (default), or `Quarter Res`. Lower resolutions reduce depth-estimation time and memory use, at a possible cost to fine depth detail.
- **DLSS 5 Motion Vector**: `Original` (default, faster) or `RAFT` (slower, generally better quality). This choice applies to both Neural Rendering and Frame Generation.

The Postprocessing, Late Postprocessing, and Media Flow controls expose **DLSS 5 NR Intensity** from `0.0` through `2.0`, with a default of `1.0`.

Because these guides are estimated from the video, results can differ from DLSS integrated directly into a game engine.

---

> Applies to: Installation and diagnostics for optional DLSS 5 on Windows, Neural Rendering from WSL using the Windows worker, and experimental DLSS Frame Generation x2 on Linux. Available processors and generation options depend on the installed runtime and compatible NVIDIA hardware.
