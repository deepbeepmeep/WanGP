# Manual Installation Guide For Windows & Linux

This guide covers manual installation for different GPU generations and operating systems. Alternatively you may use the 1 click install / update scripts (please check the repo readme for instructions).

It is recommended to use Python 3.10.9, PyTorch 2.7.1 with Cuda 12.8 for GTX 10XX and Python 3.11.14, PyTorch 2.10 with Cuda 13.0/13.1 for RTX 20XX - RTX 50XX. RTX 20XX requires Triton 3.2.

It is not recommended to use either PytTorch 2.8.0 as some System RAM memory leaks have been observed when switching models or 2.9.0 which has some Convolution 3D perf issues (VAE VRAM requirements explode).

If you want to use the NV FP4 optimized kernels for RTX 50xx, you will need to upgrade to Python 3.11, PyTorch 2.10 with Cuda 13.0 if you are still using the old install setup based on cuda 12.8.

## Setup Conda

You need to install anaconda or miniconda first (https://www.anaconda.com/download/success?reg=skipped) 

## Minimal WanGP installation
### RTX 20xx - RTX 50xx Installation
you must install Cuda 13.1: https://developer.nvidia.com/cuda-13-1-0-download-archive

Then open a Terminal Window get in the parent folder where you would to install WanGP and then type in:
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### GTX 10xx Installation

you must install Cuda 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive

Then open a Terminal Window get in the parent folder where you would to install WanGP and then type in:
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

## INT8 Math Kernels

In the configuration menu, **INT8 Math Kernels** selects the backend for INT8 quantized linear layers, including INT8 ConvRot checkpoints:

- **Auto** (default): tries WGPKernels, then Comfy Kitchen CUDA on NVIDIA or HIP on AMD ROCm, then Triton, then PyTorch when no other backend is available.
- **WGPKernels**: selects WanGP's own INT8 kernels, with higher precision. See [WGPKernels Installation](#wgpkernels-installation) below.
- **Triton**: selects WanGP's Triton INT8 kernels. See [Triton Installation](#triton-installation) below.
- **Comfy Kitchen Kernels**: selects Kitchen's native CUDA or HIP kernels for the current PyTorch runtime.
- **Disabled: PyTorch**: uses the standard PyTorch/Quanto path.

An unavailable selection behaves as Auto. Existing `enable_int8_kernels` configurations migrate automatically: enabled becomes `int8_kernels: "auto"`, disabled becomes `int8_kernels: "disabled"`.

Comfy Kitchen is installed with `requirements.txt`; its prebuilt CUDA kernels require an NVIDIA R580+ driver. AMD INT8/ConvRot uses ROCm PyTorch and Kitchen's HIP extension on supported RDNA3/3.5/4 devices. RDNA2 and unavailable or failing HIP backends retain the Triton/PyTorch fallback. WanGP probes small plain and ConvRot linear operations before selecting Kitchen, and reports CUDA or HIP in the startup message. This option selects INT8 linear math; it does not change the attention backend or enable FP16 accumulation.

### WGPKernels Installation

The Windows and Linux install scripts include WGPKernels for RTX 20XX, 30XX, 40XX and 50XX, choosing the wheel for the selected CUDA stack. AMD RDNA4 targets (`gfx1200`, `gfx1201`) receive the experimental ROCm 10 wheel. For an existing installation, choose **Upgrade** in the update script; **Update** only refreshes code and requirements.

WGPKernels are WanGP's own INT8 kernels, with higher precision than the other fast INT8 backends. Install the wheel matching the CUDA version of your PyTorch (NVIDIA RTX 20XX or newer; WanGP checks the kernels on your GPU before using them):

For **PyTorch 2.10 / CUDA 13** (NVIDIA R580+ driver):

```bash
# Windows
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.2/wgpkernels-1.4.2%2Bcu130-py3-none-win_amd64.whl
# Linux (glibc 2.34+)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.2/wgpkernels-1.4.2%2Bcu130-py3-none-linux_x86_64.whl
```

For **PyTorch 2.7.1 / CUDA 12.8** (NVIDIA R570+ driver):

```bash
# Windows
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.2/wgpkernels-1.4.2%2Bcu128-py3-none-win_amd64.whl
# Linux (glibc 2.34+)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.2/wgpkernels-1.4.2%2Bcu128-py3-none-linux_x86_64.whl
```

For **AMD RDNA4 (`gfx1200`, `gfx1201`) with PyTorch 2.13.0 / ROCm 10.0** (experimental):

```bash
# Windows
python -m pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.2/wgpkernels-1.4.2%2Brocm100-py3-none-win_amd64.whl
# Linux
python -m pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.4.2/wgpkernels-1.4.2%2Brocm100-py3-none-linux_x86_64.whl
```

The AMD wheel provides INT8 linear kernels. Native AdaLN, PixelNorm + SiLU and Conv3D remain CUDA-only; AMD retains the independent supported backends for those operations. AMD packages pass installation and library-loading checks, but kernel execution on AMD hardware remains unvalidated.

### CUDA Kernels Optimized Ops Precision (When Available)

This dropdown controls optional CUDA kernel optimizations:

- **Preserve Precision**: uses kernels at least as accurate as standard PyTorch math; tiny output differences are still possible.
- **Allow Faster Approximate Kernels** (default, `kernel_precision: "fast"`): allows small precision trade-offs for potential speed or VRAM savings.

Changes apply between generations without reloading weights. Saved choices are preserved; missing values use the default. **INT8 Math Kernels** sets the preferred kernel family, with automatic fallback when unavailable. Additional H3 optimizations require validated RTX 50-series (SM120) GPUs. Checkpoint dtypes, FP16 accumulation and attention settings are unchanged.



### H3 Video VAE Selection

H3's **Video VAE** configuration defaults to **Auto**, which follows the current **Transformer Model Quantization** choice: 16-bit selects BF16, FP8 selects FP8 Mixed Precision, and INT8 selects INT8 ConvRot Decoder. Explicit **BF16**, **FP8 Mixed Precision**, and **INT8 ConvRot Decoder** choices override Auto. The selected VAE is downloaded automatically before loading; INT8 math still follows the separate **INT8 Math Kernels** selection.

## Optional DLSS 5 upsamplers

WanGP can expose NVIDIA DLSS 5 Neural Rendering as a native-resolution refiner or spatial upsampler, and DLSS Frame Generation as a temporal upsampler. These optional Windows components are not installed by the normal WanGP installer and include closed-source third-party binaries with separate licenses and security implications. Close WanGP and run `scripts\install_dlss5.bat` for the checksum-verified automatic installation.

Read the full **[DLSS 5 runtime installation, directory layout, copyright, and safety instructions](DLSS5.md)** before downloading or running them. The WanGP worker release ZIP is extracted into the root `dlss5` folder; the guide provides version-pinned downloads for the tested community components and identifies which files are modified, unsigned, or unavailable from NVIDIA's public SDK.


## Triton Installation
The Triton library is required for Pytorch compilation and Sage Attention and by various kernels to accelerate tensors processing.

### Windows RTX 30XX - RTX 50XX

For the recommended **Python 3.11 / PyTorch 2.10** environment, new installations use Triton 3.7:

```bash
python -m pip install -U "triton-windows>=3.7.0.post26,<=3.7.1.post27"
```

**Existing installations do not need to upgrade.** Triton 3.6 continues to work as before and remains available in the installer's manual selection. Run the command above in your WanGP environment only if you want to upgrade, for example when antivirus software blocks Triton's generated CUDA helper.

WanGP includes a small prebuilt CUDA helper for **3.7.0.post26 and 3.7.1.post27**, enabled automatically on 64-bit Windows with Python 3.11. No separate helper installation, Visual Studio or antivirus exclusion is needed. Triton acceleration remains enabled, including GPU kernel compilation on first use. Other Python/Triton versions keep their existing behavior.

For an existing **PyTorch 2.7 / 2.7.1** environment:

```bash
python -m pip install -U "triton-windows>=3.3,<3.4"
```

Triton 3.3 or newer is required for optimized ConvRot support.

### Windows RTX 20XX

RTX 20XX must use Triton 3.2:

```bash
python -m pip install -U "triton-windows>=3.2,<3.3"
```

### Windows AMD / ROCm 10

For **Python 3.12 / PyTorch 2.13.0 / ROCm 10.0.0**, WanGP includes a separate prebuilt HIP helper, automatically enabled with **3.7.0.post26 or 3.7.1.post27**. It uses the installed ROCm runtime and avoids compiling Triton's host helper locally; GPU kernels still compile normally. There is no extra helper installation or Visual Studio requirement.

Upgrading an existing environment is optional. To use one of these releases, run this command in your AMD WanGP environment:

```bash
python -m pip install -U "triton-windows>=3.7.0.post26,<=3.7.1.post27"
```

Older Triton versions retain their existing behavior. The prebuilt HIP helper is experimental pending AMD GPU generation and antivirus validation; it does not extend ROCm's supported GPU list. Linux is unchanged.

### Linux

PyTorch normally installs the matching Triton version automatically: 3.3 for PyTorch 2.7, or 3.6 for PyTorch 2.10. RTX 20XX requires Triton 3.2.

## Sage Attention
Sage Attention accelerates Video / Image Generation up to x2 with very little quality loss. Sage does not support GTX 10XX in WanGP.

Use the version matching your GPU generation:

- **RTX 20XX (Turing):** SageAttention 1.0.6. SageAttention 2 is not supported.
- **RTX 30XX or newer (Ampere, Ada, and Blackwell):** SageAttention 2.2.0.

#### Windows: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Windows: Install SageAttention 2 for RTX 30XX-50XX
For the recommended PyTorch 2.10 environment:
```
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0+cu130torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl
```
This version (or a later one) is needed for the **CUDA Graph** compilation mode to keep SageAttention's fastest kernels on RTX 40XX/50XX: with older versions, CUDA Graph mode switches the attention to slower SageAttention kernels.

#### Linux: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Linux: Install SageAttention 2 for RTX 30XX-50XX
Prebuilt wheels come from the [WanGP SageAttention fork](https://github.com/deepbeepmeep/SageAttention), no compilation needed. They also keep SageAttention's fastest kernels in **CUDA Graph** mode.

For the recommended Python 3.11 / PyTorch 2.10 / CUDA 13 environment:
```
pip install https://github.com/deepbeepmeep/SageAttention/releases/download/v2.2.0-linux.cg1/sageattention-2.2.0+cu130torch2.10.0.cg1-cp311-cp311-linux_x86_64.whl
```
The `cp311` part of the file name (it appears twice) is the Python version (3.11) and must match your environment's Python: with Python 3.10 or 3.12, replace it with `cp310` or `cp312` (see the [release page](https://github.com/deepbeepmeep/SageAttention/releases/tag/v2.2.0-linux.cg1)). These wheels work only with PyTorch 2.10 and CUDA 13, and need a Linux distribution from 2018 or later (glibc 2.28). The 1 click installer picks the right wheel automatically.

For another PyTorch version, compile the fork instead (this takes a while):
```
python -m pip install "setuptools<=75.8.2" ninja wheel --force-reinstall
git clone https://github.com/deepbeepmeep/SageAttention
cd SageAttention
pip install --no-build-isolation -e .
```

## Sparge Attention
Sparge Attention (`spas_sage_attn`) provides the optimized sparse attention kernels used by FlashVSR. Install it after Pytorch and Triton.

#### Windows Install Sparge Attention for Pytorch 2.10 / Python 3.11 / Cuda 13
```
pip install https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post4/spas_sage_attn-0.1.0%2Bcu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

#### Windows Install Sparge Attention for Pytorch 2.7.1 / Python 3.10 / Cuda 12.8
```
pip install https://github.com/woct0rdho/SpargeAttn/releases/download/v0.1.0-windows.post3/spas_sage_attn-0.1.0%2Bcu128torch2.7.1.post3-cp39-abi3-win_amd64.whl
```

#### Linux Install Sparge Attention
Activate your WanGP Python environment before running these commands; SpargeAttn's build compiles against that environment's PyTorch. The installer prepends the environment's `bin` directory to `PATH`, so the build's kernel generators run with the same Python that builds the package.

CUDA 13 builds can fail with `error: identifier "__assert_fail" is undefined` in `cuda_fp8.hpp`, `cuda_fp6.hpp`, or `cuda_fp4.hpp`. The 1 click installer sets `NVCC_PREPEND_FLAGS='-DNDEBUG'` for the Linux CUDA 13 Sparge entry. This turns off C assertions during the build, so the CUDA header assertion does not stop it. `-DNDEBUG` is intentional: it disables assertion checks for this build and does not change the sparse attention kernels.

Manual equivalent:
```
python -m pip install ninja wheel packaging
NVCC_PREPEND_FLAGS='-DNDEBUG' python -m pip install --no-build-isolation git+https://github.com/woct0rdho/SpargeAttn.git
```

If you only need the kernels for one GPU family, limit the build:
```
TORCH_CUDA_ARCH_LIST="12.0" NVCC_PREPEND_FLAGS='-DNDEBUG' \
  python -m pip install --no-build-isolation git+https://github.com/woct0rdho/SpargeAttn.git   # RTX 50xx only
TORCH_CUDA_ARCH_LIST="8.9" NVCC_PREPEND_FLAGS='-DNDEBUG' \
  python -m pip install --no-build-isolation git+https://github.com/woct0rdho/SpargeAttn.git    # RTX 40xx only
TORCH_CUDA_ARCH_LIST="8.6" NVCC_PREPEND_FLAGS='-DNDEBUG' \
  python -m pip install --no-build-isolation git+https://github.com/woct0rdho/SpargeAttn.git    # RTX 30xx only
```

The 1 click installer selects Sparge Attention for the Linux CUDA 13 RTX 30XX-50XX profiles. Windows uses the prebuilt wheels above; macOS/MPS and AMD/ROCm do not install Sparge.


## Flash Attention
Flash attention is not as fast as Sage for Generating Videos or Images but it preserves quality. However when used with a Language Model (prompt enhancer, Text to Speech, Deepy) it can offer a significant speedup.

The FlashAttention 2 packages below require RTX 30XX or newer. RTX 20XX users should use SDPA or SageAttention 1 where supported; the language models of autoregressive models (speech, music, prompt enhancer) then use WanGP's PyTorch kernels with CUDA Graphs automatically.

 
### Flash Attention Windows
#### Windows Pytorch 2.10 / Python 3.11
```
pip install https://github.com/deepbeepmeep/kernels/releases/download/Flash2/flash_attn-2.8.3-cp311-cp311-win_amd64.whl
```

#### Windows Pytorch 2.7.1 / Python 3.10
```
pip install https://github.com/Redtash1/Flash_Attention_2_Windows/releases/download/v2.7.0-v2.7.4/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
```

#### Linux
Run this in your WanGP environment after installing PyTorch. If a matching wheel is unavailable, building from source requires a C++ compiler and the CUDA toolkit matching your PyTorch stack (CUDA 12.8 or 13.0, including `nvcc`).

```
pip install flash-attn==2.8.3.post1 --no-build-isolation
```


## GGUF llama.cpp CUDA Kernels

Version 1.0.28 accelerates GGUF models with packed MMVQ/MMQ, direct FP16/BF16 activation quantization, CUDA-graph-safe workspaces and quantized KV-cache attention.

Large linears, such as those of long video token sequences or long prompts, now run in a single kernel that quantizes the activations while other parts of the GPU multiply earlier rows, and adds the bias as it writes the result. On Q4_K/Q6_K layers this was 2-10% faster without bias and 5-19% faster with bias, used slightly less memory, and turned each linear into a single CUDA graph node instead of dozens, which lowers the memory kept by CUDA graphs. The first time each layer shape runs on a GPU, WanGP times the new kernel against the previous row slices and keeps the faster one; the choice is cached. With an older GGUF wheel, WanGP keeps using the previous row slices.

On RTX 20, Bonsai and other PTQ1_0 language models decode their tokens and speculative drafts faster: each block of weights is decoded once for all the tokens of a step.

Packed rows crossing quantization blocks are supported on RTX 20/30/40/50 for Q2_K through Q6_K, Q4_0/Q4_1, Q5_0/Q5_1 and Q8_0. This fixes MiniMax H3 image-reference projections without dense weight materialization or matrix padding. RTX 20 uses NVRTC; RTX 30/40/50 use the installed Triton compiler. First use compiles and tunes the kernels.

Install the wheel matching your Python, PyTorch and CUDA stack. `--no-deps` preserves the installed PyTorch environment.

### Python 3.11 / PyTorch 2.10 / CUDA 13

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/llamacpp_gguf_cuda-1.0.29%2Btorch210cu130py311-cp311-cp311-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/llamacpp_gguf_cuda-1.0.29%2Btorch210cu130py311-cp311-cp311-linux_x86_64.whl
```

### Python 3.10 / PyTorch 2.7.1 / CUDA 12.8

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/llamacpp_gguf_cuda-1.0.29%2Btorch271cu128py310-cp310-cp310-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/llamacpp_gguf_cuda-1.0.29%2Btorch271cu128py310-cp310-cp310-linux_x86_64.whl
```

The CUDA 13 builds contain native GPU code for SM75 through the architectures supported by CUDA 13.1. CUDA 12.8 builds additionally contain pre-SM75 code, subject to PyTorch's own support. Linux wheels need glibc 2.28 or newer. The release includes the exact architecture lists, source and build instructions.

### Experimental AMD HIP wheel: Python 3.12 / PyTorch 2.13 / ROCm 10

For RDNA4 GPUs (`gfx1200`, `gfx1201`, RX 9060/9070 series), the WanGP installer adds this wheel to its AMD environment (Python 3.12, PyTorch 2.13.0+rocm10.0.0). To install it manually in that environment:

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/llamacpp_gguf_cuda-1.0.29%2Btorch213rocm100py312-cp312-cp312-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.29/llamacpp_gguf_cuda-1.0.29%2Btorch213rocm100py312-cp312-cp312-linux_x86_64.whl
```

The HIP and CUDA wheels use the same package name; select the matching backend. The HIP wheel requires exactly PyTorch 2.13.0+rocm10.0.0 and rejects a different PyTorch build at import. It keeps the previous GGUF kernels: the new single-kernel row loop is CUDA only. Compilation and import checks do not establish AMD inference correctness: hardware validation remains pending.


## INT4 / FP4 quantized support

These kernels will offer optimized INT4 / FP4 dequantization.

**Please Note FP4 support is hardware dependent and will work only with RTX 50xx / sm120+ GPUs**


### Lightx2v NVP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13 (RTX 50xx / sm120+ only !)
- Windows
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-win_amd64.whl
   ```

- Linux
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.2+torch2.10.0-cp311-abi3-linux_x86_64.whl
   ```


### Nunchaku INT4/FP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13

- Windows 
   ```
  pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-win_amd64.whl
   ```

- Linux 
   ```
  pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu13.0torch2.10-cp311-cp311-linux_x86_64.whl
   ```


### Nunchaku INT4/FP4 Kernels Wheels for Python 3.10 / Pytorch 2.7.1 / Cuda 12.8  
- Windows
   ```
   pip install https://github.com/deepbeepmeep/kernels/releases/download/v1.2.0_Nunchaku/nunchaku-1.2.0+torch2.7-cp310-cp310-win_amd64.whl
   ```

- Linux (Pytorch 2.7.1 / Cuda 12.8) 
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/v1.2.0_Nunchaku/nunchaku-1.2.0+torch2.7-cp310-cp310-linux_x86_64.whl
   ```


### Bitsandbytes NF4 Kernels for Python 3.11 / Pytorch 2.10 / Cuda 13

These kernels accelerate bitsandbytes 4-bit / NF4 checkpoints. Install them after Pytorch; pip will pick the matching Windows or Linux wheel automatically.
```
pip install bitsandbytes==0.49.2
```

---

> Applies to: Manual installation and dependencies on Windows and Linux. Requirements vary by operating system and GPU; the commands in this guide concern installation and launch.
