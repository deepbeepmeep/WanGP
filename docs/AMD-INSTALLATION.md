# AMD Installation Guide for Windows (TheRock)

This guide covers running WanGP on AMD Radeon GPUs under Windows with the official ROCm PyTorch wheels built by [TheRock](https://github.com/ROCm/TheRock).

## Supported GPUs

TheRock publishes Windows PyTorch wheels for the following architectures (see the [support matrix](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)). Each GPU is identified by its `gfx` target, which is used during installation.

| Family   | Target  | Example GPUs                                   |
| -------- | ------- | ---------------------------------------------- |
| RDNA 4   | gfx1201 | RX 9070 XT, RX 9070, Radeon AI PRO R9700       |
| RDNA 4   | gfx1200 | RX 9060 XT, RX 9060                            |
| RDNA 3.5 | gfx1151 | Ryzen AI Max (Strix Halo)                      |
| RDNA 3.5 | gfx1150 | Radeon 890M / 880M (Ryzen AI 9 HX 370 / 375)   |
| RDNA 3.5 | gfx1152 | Ryzen AI 7 350                                 |
| RDNA 3.5 | gfx1153 | Radeon 820M                                    |
| RDNA 3   | gfx1100 | RX 7900 XTX, RX 7900 XT, RX 7900 GRE           |
| RDNA 3   | gfx1101 | RX 7800 XT, RX 7700 XT                         |
| RDNA 3   | gfx1102 | RX 7600 XT, RX 7600                            |
| RDNA 3   | gfx1103 | Radeon 780M / 760M (Ryzen 7040 / 8040)         |
| RDNA 2   | gfx1030 | RX 6950 XT, RX 6900 XT, RX 6800 XT, RX 6800    |
| RDNA 2   | gfx1031 | RX 6750 XT, RX 6700 XT                         |
| RDNA 2   | gfx1032 | RX 6650 XT, RX 6600 XT, RX 6600                |
| RDNA 2   | gfx1034 | RX 6500 XT                                     |
| RDNA 2   | gfx1035 | Radeon 680M                                    |

RDNA 1 (gfx1010 - gfx1012) and the remaining RDNA 2 iGPUs (gfx1033, gfx1036) are published as well. On RDNA 2 and older, PyTorch has no flash attention kernel, so attention runs noticeably slower than on RDNA 3 and newer.

To check the target of your GPU, run the following in a terminal (`clinfo` is installed with the AMD driver):

```cmd
clinfo | findstr gfx
```

## Requirements

- Windows 10 or 11 with the latest [AMD Adrenalin driver](https://www.amd.com/en/support/download/drivers.html)
- Python 3.12 (the ROCm package index only carries a NumPy build for Python 3.12 and newer, and some WanGP dependencies have no Windows wheels beyond 3.12)
- [Git](https://git-scm.com/install/windows)

## Automatic Installation

`scripts\install.bat` detects AMD GPUs through `clinfo` and installs a Python 3.12 environment with ROCm PyTorch for the detected target(s), Triton and SageAttention 1. See the Installation section of the main [README](../README.md) for how to use the scripts.

For RDNA4 (`gfx1200`, `gfx1201`), it also installs the experimental WGPKernels INT8 wheel for PyTorch 2.13.0 / ROCm 10.0. See [WGPKernels Installation](INSTALLATION.md#wgpkernels-installation) for manual commands and validation limitations. Other AMD targets keep their existing supported INT8 backends.

For RDNA3 (`gfx1100` to `gfx1103`) and RDNA4, it installs the experimental GGUF kernels for the same stack. Without them, GGUF models such as the Deepy and Prompt Enhancer Qwen checkpoints run every layer through a slow, memory-hungry fallback: a 27B model may not fit a 24 GB card. See [GGUF llama.cpp CUDA Kernels](INSTALLATION.md#gguf-llamacpp-cuda-kernels) for manual commands.

Environments created by older versions of the installer or of this guide (the `rocm65` option, or the per-family `rocm.nightlies.amd.com/v2/gfx...` indexes) should be replaced: create a new environment with `install.bat` and make it the active one with `manage.bat`.

## Manual Installation

The commands below are for the Windows Command Prompt (CMD).

### Step 1: Clone WanGP and create a virtual environment

```cmd
cd \your-path-to-wan2gp
git clone https://github.com/deepbeepmeep/Wan2GP.git
cd Wan2GP
py -3.12 -m venv wan2gp-env
wan2gp-env\Scripts\activate
```

### Step 2: Install ROCm PyTorch

Replace `gfx1201` with the target of your GPU from the table above:

```cmd
pip install "torch[device-gfx1201]==2.13.0+rocm10.0.0" "torchvision[device-gfx1201]==0.28.0+rocm10.0.0" torchaudio==2.11.0.2+rocm10.0.0 --index-url https://stable.repo.amd.com/rocm/whl-next/
```

The ROCm runtime is installed automatically as a dependency. On a system with several AMD GPUs (for example an iGPU and a dedicated card), list all targets: `torch[device-gfx1036,device-gfx1201]`, and the same for `torchvision`.

Use `--index-url` exactly as shown. With `--extra-index-url`, pip prefers the newer CPU-only PyTorch from PyPI and silently installs that instead.

### Step 3: Install WanGP dependencies

```cmd
pip install -r requirements.txt
```

### Step 4: Verify the installation

```cmd
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Expected output example:

```
PyTorch: 2.13.0+rocm10.0.0
GPU available: True
Device: AMD Radeon RX 9070 XT
```

If the PyTorch version does not end in `+rocm...`, a CPU-only build was installed: repeat Step 2.

### Nightly builds (optional)

TheRock also publishes nightly builds with newer PyTorch and ROCm versions. They may be unstable:

```cmd
pip install --pre "torch[device-gfx1201]" "torchvision[device-gfx1201]" torchaudio --index-url https://nightly.repo.amd.com/rocm/whl-next/
```

When using a nightly, install the `triton-windows` minor version that matches that PyTorch release instead of the one shown below.

## Attention Modes

- SDPA (default): PyTorch's built-in attention uses AOTriton flash attention kernels on RDNA 3 and newer. On GPUs where PyTorch still marks them experimental they are disabled unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is set; `wgp.py` sets it automatically.
- SageAttention 1: requires Triton. Install both, then select Sage in the configuration menu:

```cmd
pip install -U "triton-windows>=3.7,<3.8"
pip install sageattention==1.0.6
```

`triton-windows` 3.7 matches PyTorch 2.13. It finds the ROCm SDK from the active environment, so no Visual Studio, `rocm-sdk init` or extra environment variables are needed.

- FlashAttention-2: uses the Triton kernels from [aiter](https://github.com/ROCm/aiter), which the build installs from a bundled submodule. It requires `triton-windows` (see above). Nothing is compiled, so no Visual Studio is needed. Build it from source, then select Flash in the configuration menu:

```cmd
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
set FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
pip install --no-build-isolation .
```

The variable is only needed for the build. At runtime flash-attn prints a one-time warning that it is falling back to the Triton implementation, which is expected.

## Running WanGP

Activate the environment and start WanGP:

```cmd
cd \path-to\Wan2GP
wan2gp-env\Scripts\activate
python wgp.py
```

If VAE encoding or decoding is very slow the first time a resolution is used, MIOpen (AMD's equivalent of cuDNN) may be benchmarking convolution kernels. Its fast find mode uses heuristics instead:

```cmd
set MIOPEN_FIND_MODE=FAST
```

## Troubleshooting

### GPU Not Detected

If `torch.cuda.is_available()` returns `False`:

1. Check that your GPU is in the [Supported GPUs](#supported-gpus) list and that the `device-gfx...` target in Step 2 matches the output of `clinfo | findstr gfx`.
2. Update the AMD Adrenalin driver.

### Installation Errors

- "No solution found" or "Could not find a version that satisfies the requirement": check that the virtual environment uses Python 3.12 (`python --version`) and that the target name is spelled correctly (for example `device-gfx1201`, not `device-gfx120X`).
- "No matching distribution found": the GPU architecture is not published for Windows, or the virtual environment is not activated.

### Performance Issues

- Close applications that use GPU acceleration (browsers, Discord, etc.) to free VRAM.
- Lower the resolution or pick a lower-VRAM memory profile in the configuration menu.

Known issues with the ROCm Python wheels are tracked at https://github.com/ROCm/TheRock/issues/808.

## Additional Resources

- [TheRock GitHub Repository](https://github.com/ROCm/TheRock/)
- [Release and installation documentation](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
- [Supported GPU architectures](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)

For additional troubleshooting guidance for WanGP, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

> Applies to: Installation on Windows with AMD GPUs and TheRock. Dependency and driver instructions are specific to this platform.
