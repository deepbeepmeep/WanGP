# Fibo RMBG-3.0

Native WanGP safetensors prepared from [Bria RMBG-3.0](https://huggingface.co/briaai/RMBG-3.0/tree/12b88d7d646046887b5289adf20a844efd8a2b0b).
The four source transformer shards were merged without changing their BF16
tensor values. `Fibo_RMBG_3.0_bf16.safetensors` is the merged checkpoint;
`Fibo_RMBG_3.0_int8_convrot.safetensors` quantizes eligible transformer matrices
with INT8 ConvRot, retaining BF16 weights/biases elsewhere and FP32 scales.

This is a Fibo Edit Turbo variant for four-step grayscale alpha-matte generation.
It shares the Fibo encoder and VAE assets. The default WanGP prompt is the
official alpha-matting instruction. White is opaque foreground, black is
background, and gray carries partial opacity. It is separate from the U2NET,
BiRefNet General and RMBG-2.0 background-remover dropdown.

Weights remain subject to Bria's [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
Non-commercial use only; commercial use requires a separate agreement with Bria.
The WanGP packaging and quantization do not change those terms.
