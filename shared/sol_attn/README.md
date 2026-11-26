# Sol-Attn Triton backend

This reusable package vendors the portable Triton implementation from
[`NVlabs/Sana` at commit `46031940`](https://github.com/NVlabs/Sana/tree/46031940ba8af5d18054217e571149579424c0b1/techniques/sparse_backends/sol_attn),
under its Apache-2.0 license.

The `saganaki/` path bundles the optimized INT8-QK implementation from
[`ComfyUI-sol-attn` v0.6.2](https://github.com/Saganaki22/ComfyUI-sol-attn/tree/930a4d6), also
under Apache-2.0. WanGP uses its residual INT8-QK path for MiniMax H3; the original BF16
implementation remains available through the same public interface.

`sol_attn(q, k, v, ...)` expects BF16 tensors in `[batch, tokens, heads, 128]` layout. The head
dimension must be contiguous; the batch, token, and head strides may come directly from a fused
QKV projection.
The optimized Triton backend supports SM86, SM89, SM90, SM100, SM120, and SM121. It requires
CUDA 12.8 and Triton 3.6 or newer.

The attention kit and shared Sol core apply the precision setting and handle protected token ranges.
Models supply their conditioning layout and any model-specific threshold schedule.

`qk_rms_norm_rope_(...)` is the reusable zero-copy preparation path. It applies Q/K RMSNorm and
RoPE in place to either separate projection outputs or strided views of a fused QKV output.

`query_start` skips complete query blocks before the requested token. Their output is intentionally
left uninitialized, so callers using it must replace that prefix with an exact attention result.

### Native kernels and WanGP integration

WanGP's shared Sol core stages projections through the attention kit for original Sol and Sol3.
The [SparseGP release](https://github.com/deepbeepmeep/SparseGP) supplies the native CUDA and HIP images.
WanGP contains the Triton fallbacks. Without SparseGP, Sol3 uses the local mixed-block INT8 Triton kernels
on supported CUDA and ROCm configurations. The original Sol mode retains its own threshold schedule.
H3 supplies its conditioning-token count and current threshold to the same attention entry point as other models.

On compatible GPUs the original Sol setting `WGP_SOL_CUDA` selects the SparseGP packed INT8 or NVFP4 operators
(`i8` or `fp4`). The `i8_mixed` and `fp4_mixed` variants also route individual keys in mixed blocks.
