# Background-removal checkpoints

These are native PyTorch safetensors for WanGP. MMGP manages their GPU memory.
U2NET and the two Swin/BiRefNet models retain the released inference architecture;
training-only heads and generated relative-position indices are omitted.

| Checkpoint | Source | License |
| --- | --- | --- |
| `u2net_bf16.safetensors` | [U-2-Net](https://github.com/xuebinqin/U-2-Net), official `u2net.pth` | Apache-2.0 |
| `birefnet-general_bf16.safetensors` | [BiRefNet General](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4) | MIT |
| `bria-rmbg2_bf16.safetensors` | [Bria RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0/tree/6a88d6a65119856a743be87ec1e6bee6f08e1f93) | CC BY-NC 4.0 |

Changes: weights prepared in BF16; deformable-convolution parameters in BiRefNet
and RMBG-2.0 use FP16 because their torchvision operator does not support BF16.
Their remaining parameters and running statistics use BF16; integer counters
are preserved. Each tensor was verified against its intended source cast.
The native outputs were compared against the original FP32 ONNX outputs on
eight portraits and one animal image. No significant visual regression was
observed. WanGP uses the floating-point checkpoints by default.

Matching `birefnet-general_int8_convrot.safetensors` and
`bria-rmbg2_int8_convrot.safetensors` quantize the Swin transformer matrices.
They retain BF16 nonquantized weights, FP16 deformable-convolution parameters
and FP32 quantization scales. These optional files were also validated on the
same nine images; the configuration dropdown continues to use BF16.

The MIT license of BiRefNet's architecture does not apply to Bria's weights.
Bria weights are for non-commercial use under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Commercial use requires a separate agreement with Bria.

RMBG-3.0 is a separate Fibo variant in `DeepBeepMeep/Fibo2`, not one of these
background-remover dropdown choices.
