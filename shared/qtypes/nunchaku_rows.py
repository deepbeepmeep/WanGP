# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Row ranges for Nunchaku's shared 128-row packing, retaining each handler's INT4/FP4 arithmetic."""
import torch
from torch._subclasses.fake_tensor import is_fake

_ROW_SLICE_BYTES = 64 << 20  # scratch of the activation rows quantized at a time: 4-bit values and their group scales
_ROW_SLICE_MIN = 32768  # rows per W4A4 GEMM: its FP4 GEMM loses up to 10 % on fewer rows at some shapes


def w4a4_linear(ops, x, weight, bias, fp4, alpha, wcscales, act_unsigned):
    """Nunchaku's W4A4 linear of x (M, K): activations (and the low-rank branch input) quantized and multiplied by slices of 256-row
    tiles when x spans two or more, each written to its rows of the output, so that their quantized copy never exceeds one slice. The
    quantization is per row, so the slices compute what the whole tensor does."""
    rows, k = x.shape
    out = torch.empty((-(-rows // 256) * 256, weight.shape[0]), dtype=weight._proj_up.dtype, device=x.device)
    if bias is not None and bias.dtype != out.dtype:
        bias = bias.to(out.dtype)
    step = max(_ROW_SLICE_MIN, _ROW_SLICE_BYTES // (k // 2 + k // 16) // 256 * 256)
    step = -(-rows // (max(1, rows // step) * 256)) * 256  # slices of even size, none below the minimum
    for start in range(0, rows, step):
        qx, ascales, lora_act = ops.svdq_quantize_w4a4_act_fuse_lora_cuda(x[start:start + step], lora_down=weight._proj_down,
                                                                          smooth=weight._smooth_factor, fp4=fp4, pad_size=256)
        ops.svdq_gemm_w4a4_cuda(act=qx, wgt=weight._qweight, out=out[start:start + qx.shape[0]], ascales=ascales, wscales=weight._wscales,
                                lora_act_in=lora_act, lora_up=weight._proj_up, bias=bias, fp4=fp4, alpha=alpha, wcscales=wcscales,
                                act_unsigned=act_unsigned)
        del qx, ascales, lora_act
    return out[:rows]


class NunchakuRowProjection:
    def __init__(self, handler):
        self.handler = handler
        self.fp4 = handler.HANDLER_NAME == "nunchaku_fp4"

    def supports(self, module, x):
        weight = getattr(module, "weight", None)
        return (isinstance(weight, self.handler.NunchakuBaseWeightTensor) and not is_fake(x)
                and weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0
                and getattr(weight, "_nunchaku_transpose_ref", None) is None
                and not any(getattr(obj, flag, False) for obj in (module, weight)
                            for flag in ("_nunchaku_mod_reorder", "_nunchaku_mod_scale_shift")))

    def key(self, module):
        weight = module.weight
        if isinstance(weight, self.handler.NunchakuAWQWeightTensor):
            return None
        return self.handler.HANDLER_NAME, weight._smooth_factor, weight._proj_down

    def prepare(self, module, x):
        weight, handler = module.weight, self.handler
        ops = handler._load_nunchaku_ops() if x.is_cuda else None
        if ops:
            q, scales, lowrank = ops.svdq_quantize_w4a4_act_fuse_lora_cuda(x, lora_down=weight._proj_down,
                                                                       smooth=weight._smooth_factor, fp4=self.fp4, pad_size=256)
            return ops, q, scales, lowrank, x.shape[0], 0
        smooth = handler._unpack_nunchaku_scale_vector(weight._smooth_factor, weight.shape[1])
        rotated = x / smooth.to(x.dtype) if torch.is_tensor(smooth) else x
        down = handler._unpack_lowrank_weight(weight._proj_down, down=True).to(x.dtype)
        return None, rotated, None, x @ down.t(), x.shape[0], 0

    @staticmethod
    def slice_input(prepared, start, stop):
        """Rows of the input (offload.linear_input_rows). Nunchaku's kernels: quantized rows, their scales stored as flat tiles of 256
        rows ((groups, rows) per tile) and the low-rank input, from the 256-row tile holding `start`, and the offset of `start` in it
        (rows() trims its output to them). The CPU form: plain rows. An AWQ input is the 16-bit input itself."""
        if torch.is_tensor(prepared):
            return prepared[start:stop]
        ops, q, scales, lowrank, tokens, offset = prepared
        start, stop = start + offset, stop + offset
        if ops is None:
            return None, q[start:stop], None, lowrank[start:stop], stop - start, 0
        first, last = start // 256 * 256, -(-stop // 256) * 256
        groups = scales.shape[0]
        tiles = scales.reshape(-1)[first * groups:last * groups].view(groups, last - first)
        return ops, q[first:last], tiles, lowrank[first:last], stop - start, start - first

    def rows(self, module, prepared, start, stop):
        source, h = module.weight, self.handler
        first, last = start // 128 * 128, (stop + 127) // 128 * 128
        count, k = last - first, source.shape[1]
        bias = None if module.bias is None else module.bias[first:last]
        if isinstance(source, h.NunchakuAWQWeightTensor):
            weight = h.NunchakuAWQWeightTensor.create(source._qweight[first // 4:last // 4], source._wscales[:, first:last].contiguous(),
                source._wzeros[:, first:last].contiguous(), (count, k), (k, 1), source.dtype)
            return weight.linear(prepared, bias)[:, start - first:stop - first]
        qweight = source._qweight[first:last]
        scales = source._wscales.reshape(source.shape[0] // 128, -1)[first // 128:last // 128].reshape(k // source._group_size, count)
        up = source._proj_up[first:last]
        ops, q, act_scales, lowrank, tokens, offset = prepared
        if ops:
            if self.fp4 and "_alpha" not in source.__dict__:
                source._alpha = float(source._wtscale.item()) if torch.is_tensor(source._wtscale) else 1.0 if source._wtscale is None else float(source._wtscale)
            out = torch.empty((q.shape[0], count), dtype=up.dtype, device=q.device)
            ops.svdq_gemm_w4a4_cuda(act=q, wgt=qweight, out=out, ascales=act_scales, wscales=scales,
                lora_act_in=lowrank, lora_up=up, bias=None if bias is None else bias.to(out.dtype), fp4=self.fp4,
                alpha=source._alpha if self.fp4 else 1.0, wcscales=source._wcscales[first:last] if self.fp4 and source._wcscales is not None else None,
                act_unsigned=False if self.fp4 else source._act_unsigned)
            out = out[offset:offset + tokens]
        else:
            if self.fp4:
                values = h._unpack_nunchaku_fp4_weight(qweight, count, k).to(torch.int64)
                values = h._get_fp4_lut(q.device, q.dtype)[values]
            else:
                values = h._unpack_nunchaku_w4a4_weight(qweight, count, k).to(q.dtype)
            scales = h._unpack_nunchaku_wscales(scales, count, k, source._group_size)
            dense = values * h._expand_group_scales(scales, source._group_size).to(q.dtype)
            out = q @ dense.t()
            if self.fp4:
                if source._wtscale is not None:
                    out.mul_(source._wtscale.to(q.dtype) if torch.is_tensor(source._wtscale) else float(source._wtscale))
                if source._wcscales is not None:
                    out.mul_(h._unpack_nunchaku_scale_vector(source._wcscales[first:last], count).to(q.dtype))
            out.add_(lowrank @ h._unpack_lowrank_weight(up, down=False).to(q.dtype).t())
            if bias is not None:
                out.add_(bias)
        return out[:, start - first:stop - first]
