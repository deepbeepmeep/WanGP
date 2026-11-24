# Manual Installation Guide For Windows & Linux

This guide covers manual installation for different GPU generations and operating systems. The [README's four guided installation methods](../README.md#-quick-start) also cover the Bat/SH scripts, Pinokio, Wan2GP Desktop and **WanGP Manager (Experimental, Windows x64)**.

WanGP Manager guides installation, installs Python and dependencies, launches WanGP in an app window or browser, and manages updates and rollback. **Download:** [WanGPManager.exe](https://github.com/deepbeepmeep/WanGP/releases/latest/download/WanGPManager.exe). This permanent link downloads the latest Manager release. See the [Manager instructions](../README.md#wangp-manager-experimental-windows).

For an existing installation, choose **Install / Add → Use existing installation** at the top of the page. Browse to the WanGP folder: Manager detects its saved environment and local `env_uv`, `venv`, `.venv` or `env` folders. A single compatible environment is selected automatically. Under **Choose another Conda environment or Python**, you can pick a discovered Conda environment or browse to Python. Discovery and registration leave packages unchanged.

Choose **Update WanGP → Prepare latest update**, review the result and confirm **Update WanGP**. Manager checks the current Python environment against the selected version, optional kernels and installed plugins. Automatic updates keep compatible packages, including on the first update of an existing installation. When packages need to change, Manager validates a separate Conda clone or a new virtual environment before switching. **Choose another version or change dependencies** lets you select an older build.

Under **Saved versions and rollback**, use **Preview restore** to return to a saved source and environment pair. Use **Remove** to discard a rollback version. Manager reclaims an environment it created after its last reference is removed. It preserves imported environments and environments used by another saved version, the active installation or another registered installation.

Existing-installation updates require Git and a WanGP Git checkout with no pending tracked source edits; Conda installations also require Conda. Stop WanGP and save or stash those edits first. Allow disk space for another environment; retained environments are kept for rollback. Launch through Manager or WanGP's installation scripts to use the selected environment: activating the original Conda environment by its old name still selects the unchanged original. Manager keeps the chosen source revision separate from your Git branches. Installations without a Git checkout or virtual environment can be launched, but cannot be updated through this workflow.

To update **Manager itself**, use **Preferences → Check for Manager updates**, download the latest installer, close Manager and run it. The installer defaults to **Update in place** when upgrading and preserves your installation list and preferences; uninstalling first is unnecessary. The permanent download URL stays the same across releases.

It is recommended to use Python 3.10.9, PyTorch 2.7.1 with Cuda 12.6 for GTX 10XX and Python 3.11.14, PyTorch 2.10 with Cuda 13.0/13.1 for RTX 20XX - RTX 50XX. RTX 20XX requires Triton 3.2.

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

you must install Cuda 12.6: https://developer.nvidia.com/cuda-12-6-0-download-archive

Then open a Terminal Window get in the parent folder where you would to install WanGP and then type in:
```bash
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Use the CUDA 12.6 build on Windows and Linux: the Linux CUDA 12.8 build of PyTorch 2.7.1 only runs on RTX 20XX and newer, so WanGP stops at startup with "no kernel image is available for execution on the device" on a GTX 10XX.

## INT8 Math Kernels

In the configuration menu, **INT8 Math Kernels** selects the backend for INT8 quantized linear layers, including INT8 ConvRot checkpoints:

- **Auto** (default): tries WGPKernels, then Comfy Kitchen CUDA on NVIDIA or HIP on AMD ROCm, then Triton, then PyTorch when no other backend is available. With **Preserve Precision** (see below), Auto skips the less precise Comfy Kitchen kernels even when they are installed; selecting **Comfy Kitchen Kernels** explicitly still uses them.
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
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.5.0/wgpkernels-1.5.0%2Bcu130-py3-none-win_amd64.whl
# Linux (glibc 2.34+)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.5.0/wgpkernels-1.5.0%2Bcu130-py3-none-linux_x86_64.whl
```

For **PyTorch 2.7.1 / CUDA 12.8** (NVIDIA R570+ driver):

```bash
# Windows
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.5.0/wgpkernels-1.5.0%2Bcu128-py3-none-win_amd64.whl
# Linux (glibc 2.34+)
pip install https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.5.0/wgpkernels-1.5.0%2Bcu128-py3-none-linux_x86_64.whl
```

For **AMD RDNA4 (`gfx1200`, `gfx1201`) with PyTorch 2.13.0 / ROCm 10.0** (experimental):

```bash
# Windows
python -m pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.5.0/wgpkernels-1.5.0%2Brocm100-py3-none-win_amd64.whl
# Linux
python -m pip install --no-deps https://github.com/deepbeepmeep/WGPKernels/releases/download/v1.5.0/wgpkernels-1.5.0%2Brocm100-py3-none-linux_x86_64.whl
```

The AMD wheel provides INT8 linear kernels. Native AdaLN, PixelNorm + SiLU and Conv3D remain CUDA-only; AMD retains the independent supported backends for those operations. AMD packages pass installation and library-loading checks, but kernel execution on AMD hardware remains unvalidated.

### CUDA Kernels Optimized Ops Precision (When Available)

This dropdown controls optional CUDA kernel optimizations:

- **Preserve Precision**: uses kernels at least as accurate as standard PyTorch math; tiny output differences are still possible. With **INT8 Math Kernels** on Auto, it also skips Comfy Kitchen.
- **Allow Faster Approximate Kernels** (default, `kernel_precision: "fast"`): allows small precision trade-offs for potential speed or VRAM savings.

Changes apply between generations without reloading weights. Saved choices are preserved; missing values use the default. **INT8 Math Kernels** sets the preferred kernel family, with automatic fallback when unavailable. Additional H3 optimizations require validated RTX 50-series (SM120) GPUs. Checkpoint dtypes, FP16 accumulation and attention settings are unchanged.



### SageGP Attention Modes (RTX 50XX)

With [SageGP](#sage-attention) installed on an RTX 50-series GPU, **Attention Type** in **Configuration → Performance** offers three SageGP modes. SageGP takes priority over SageAttention 2 and 3 when they are installed too. Save the setting to apply it to the next generation.

- **sage2 gp** (recommended) keeps SageAttention 2's precision with WanGP's INT8 attention path, including in-place query packing where supported. Long video attention (more than 8,192 tokens) uses SageGP's INT8 kernel, about 8% faster than SageAttention 2's at about the same error.
- **sage 2/3 adaptative gp** chooses, for each head, how much of the score and value calculation can use cheaper formats, by comparing candidate outputs on samples of the current activations. Its speed therefore depends on the model and the content: large on blocks whose attention is easy to compress, small on the others. It is less precise than sage2 gp.
- **sage3 gp** computes every score in FP4 on the RTX 50 series' FP4 tensor cores and keeps the values in FP8. It runs at about the speed of SageAttention 3 on every block, with roughly half of its error. It changes fine detail more than the adaptative mode; with few-step (distilled) models the video can follow a somewhat different path at the same seed.

**Run the First Denoising Step in Precise** (on by default, shown only with SageGP on an RTX 50-series GPU) applies to sage 2/3 adaptative gp and sage3 gp: the first step, which sets the composition, runs with sage2 gp's precision, for a small part of the speed gain.

The adaptative and sage3 gp modes need Triton 3.6 or newer. They handle unmasked, noncausal BF16 attention with batch size 1, 128 channels per head (256 for wide heads, such as Ideogram 4), equal query and KV head counts, and matching sequence lengths within the kernels' indexing limits. Up to 8,192 tokens, both run sage2 gp. Other shapes and masks use the existing backend.

The adaptative mode also applies when **Smart Inputs Chunker** splits eligible video self-attention into head groups. Each group then uses samples from its own heads to select the kernel precision. All queries in a group still attend to the full sequence of keys. Split projections that normalize Q and K across all heads, including LTX-2.5 video self-attention, need extra preparation: the adaptative mode switches from sage2 gp to its own kernels above 32,768 tokens, and sage3 gp hands these split calls to the adaptative kernels, the fastest that support them.

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
- **RTX 30XX or newer (Ampere, Ada, and Blackwell):** [SageGP](https://github.com/deepbeepmeep/SageGP) with PyTorch 2.10 (SageAttention 2.2.0 with PyTorch 2.7.1). SageGP is SageAttention 2.2 for every GPU plus WanGP's RTX 50XX kernels: the faster **sage2 gp** kernel and the **sage 2/3 adaptative gp** and **sage3 gp** modes (see [SageGP Attention Modes](#sagegp-attention-modes-rtx-50xx)).

On RTX 20XX (Sage1) and RTX 30XX (Sage2), WanGP can pack an owned FP16 or BF16 query into its own rows using Triton. Each eligible projection saves one temporary INT8 buffer, one byte per query element. The attention kernel can then write its output into those rows. Borrowed queries and incompatible strides use separate storage. These cards keep their existing per-block INT8 attention and 16-bit values.

On GeForce RTX 50XX with SageGP, **sage 2/3 adaptative gp** and **sage3 gp** automatically select the wide-head kernel for eligible dense BF16 self-attention with 256 channels per head, including Ideogram 4. The selector reads the tensor dimensions. The kernel checks the channel ranges while preparing each head and uses all 256 score channels wherever the compact calculation exceeds its bounds. **sage2 gp** uses the optimized INT8 path for these wide heads. Dense 128-channel attention, including LTX2 and MiniMax H3, uses its own preset-dependent routing.

#### Windows: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Windows: Install SageGP for RTX 30XX-50XX
For the recommended PyTorch 2.10 environment (any Python from 3.10). SageGP replaces SageAttention, so remove it first:
```
pip uninstall -y sageattention
pip install https://github.com/deepbeepmeep/SageGP/releases/download/v1.0.2/sagegp-1.0.2+cu130torch2.10.0andhigher-cp310-abi3-win_amd64.whl
```
SageGP keeps SageAttention's fastest kernels in the **CUDA Graph** compilation mode on RTX 40XX/50XX.

#### Linux: Install SageAttention 1 for RTX 20XX
```
pip install sageattention==1.0.6
```

#### Linux: Install SageGP for RTX 30XX-50XX
Prebuilt wheels, no compilation needed. They keep SageAttention's fastest kernels in **CUDA Graph** mode.

For the recommended PyTorch 2.10 / CUDA 13 environment (any Python from 3.10). SageGP replaces SageAttention, so remove it first:
```
pip uninstall -y sageattention
pip install https://github.com/deepbeepmeep/SageGP/releases/download/v1.0.2/sagegp-1.0.2+cu130torch2.10.0andhigher-cp310-abi3-linux_x86_64.whl
```
The wheel needs a Linux distribution from 2018 or later (glibc 2.28). The 1 click installer picks the right wheel automatically.

For PyTorch 2.7.1, use the SageAttention 2.2.0 wheels of the [older release](https://github.com/deepbeepmeep/SageGP/releases/tag/v2.2.0-linux.cg1) (`cu128torch2.7.1`; the SageGP modes need PyTorch 2.10). For another PyTorch version, compile SageGP instead (this takes a while):
```
python -m pip install "setuptools<=75.8.2" ninja wheel --force-reinstall
git clone https://github.com/deepbeepmeep/SageGP
cd SageGP
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
Activate your WanGP Python environment before running these commands; SpargeAttn's build also invokes `python` from `PATH` to generate its kernels.
```
python -m pip install ninja wheel packaging
python -m pip install --no-build-isolation git+https://github.com/woct0rdho/SpargeAttn.git
```


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

Version 1.0.30 accelerates GGUF models with packed MMVQ/MMQ, direct FP16/BF16 activation quantization, CUDA-graph-safe workspaces and quantized KV-cache attention.

Deepy and the Prompt Enhancer write faster with this version when speculative decoding is on: about 5% for the Qwen3.8 27B Q4 checkpoint and 10-15% for its smaller Q3 and Q2 checkpoints, measured on an RTX 50 GPU (other NVIDIA GPUs have not been measured; for the Q3 and Q2 checkpoints WanGP times the new kernels on first use and keeps them only where they are faster). This version also fixes a rare wrong value or crash with a few GGUF layer shapes. The new kernels are prepared on your GPU the first time they are used, which takes a moment once; no compiler needs to be installed.

Large linears, such as those of long video token sequences or long prompts, now run in a single kernel that quantizes the activations while other parts of the GPU multiply earlier rows, and adds the bias as it writes the result. On Q4_K/Q6_K layers this was 2-10% faster without bias and 5-19% faster with bias, used slightly less memory, and turned each linear into a single CUDA graph node instead of dozens, which lowers the memory kept by CUDA graphs. The first time each layer shape runs on a GPU, WanGP times the new kernel against the previous row slices and keeps the faster one; the choice is cached. With an older GGUF wheel, WanGP keeps using the previous row slices.

On RTX 20, Bonsai and other PTQ1_0 language models decode their tokens and speculative drafts faster: each block of weights is decoded once for all the tokens of a step.

Packed rows crossing quantization blocks are supported on RTX 20/30/40/50 for Q2_K through Q6_K, Q4_0/Q4_1, Q5_0/Q5_1 and Q8_0. This fixes MiniMax H3 image-reference projections without dense weight materialization or matrix padding. RTX 20 uses NVRTC; RTX 30/40/50 use the installed Triton compiler. First use compiles and tunes the kernels.

Install the wheel matching your Python, PyTorch and CUDA stack. `--no-deps` preserves the installed PyTorch environment.

### Python 3.11 / PyTorch 2.10 / CUDA 13

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch210cu130py311-cp311-cp311-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch210cu130py311-cp311-cp311-linux_x86_64.whl
```

### Python 3.10 / PyTorch 2.7.1 / CUDA 12.8

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch271cu128py310-cp310-cp310-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch271cu128py310-cp310-cp310-linux_x86_64.whl
```

The CUDA 13 builds contain native GPU code for SM75 through the architectures supported by CUDA 13.1. CUDA 12.8 builds additionally contain pre-SM75 code, subject to PyTorch's own support. Linux wheels need glibc 2.28 or newer. The release includes the exact architecture lists, source and build instructions.

### Experimental AMD HIP wheel: Python 3.12 / PyTorch 2.13 / ROCm 10

For RDNA3 GPUs (`gfx1100`, `gfx1101`, `gfx1102`, `gfx1103`: RX 7600 to 7900 series, Radeon 780M / 760M) and RDNA4 GPUs (`gfx1200`, `gfx1201`, RX 9060/9070 series), the WanGP installer adds this wheel to its AMD environment (Python 3.12, PyTorch 2.13.0+rocm10.0.0). To install it manually in that environment:

Windows:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch213rocm100py312-cp312-cp312-win_amd64.whl
```

Linux:
```bash
pip install --no-deps https://github.com/deepbeepmeep/kernels/releases/download/gguf-v1.0.30/llamacpp_gguf_cuda-1.0.30%2Btorch213rocm100py312-cp312-cp312-linux_x86_64.whl
```

The HIP and CUDA wheels use the same package name; select the matching backend. The HIP wheel requires exactly PyTorch 2.13.0+rocm10.0.0 and rejects a different PyTorch build at import. It keeps the previous GGUF kernels: the new single-kernel row loop is CUDA only. Compilation and import checks do not establish AMD inference correctness: hardware validation remains pending.


## INT4 / FP4 quantized support

These kernels will offer optimized INT4 / FP4 dequantization.

**Please Note FP4 support is hardware dependent and will work only with RTX 50xx / sm120+ GPUs**


### Lightx2v NVP4 Kernels Wheels for Python 3.11 / Pytorch 2.10 / Cuda 13 (RTX 50xx / sm120+ only !)
- Windows
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.3+torch2.10.0-cp311-abi3-win_amd64.whl
   ```

- Linux
   ```
  pip install https://github.com/deepbeepmeep/kernels/releases/download/Light2xv/lightx2v_kernel-0.0.3+torch2.10.0-cp311-abi3-linux_x86_64.whl
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
