import time

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .ssim import ssim_matlab
from .RIFE_V4 import Model as ModelV4


def get_frame(frames, frame_no):
    frame = frames[:, frame_no] if frame_no < frames.shape[1] else None
    if frame is None:
        return None
    if frame.dtype == torch.uint8:
        return frame.float().div_(255.0)
    return ((frame + 1) / 2).clamp_(0, 1)


def add_frame(output, index, frame, h, w):
    # Quantize before the blocking device-to-host copy. Never retain FP32 output.
    frame = frame.reshape(-1, *frame.shape[-2:])[:, :h, :w]
    output[:, index].copy_(frame.mul(2).sub_(1).clamp_(-1, 1).add_(1).mul_(127.5).clamp_(0, 255).to(dtype=torch.uint8))


@torch.inference_mode()
def process_frames(model, device, frames, multiplier, *, previous_last_frame=None, abort_callback=None, progress_callback=None):
    prefix = previous_last_frame is not None
    total = frames.shape[1] + int(prefix)
    pos = 0
    last_report = -float("inf")

    def read(index):
        if prefix and index == 0:
            return get_frame(previous_last_frame, 0)
        return get_frame(frames, index - int(prefix))

    def cancelled():
        return abort_callback is not None and abort_callback()

    if cancelled():
        return None
    lastframe = read(0)
    if lastframe is None:
        raise ValueError("RIFE received an empty video")
    _, h, w = lastframe.shape
    padding = (0, (-w) % model.pad_mod, 0, (-h) % model.pad_mod)

    def prepare(frame):
        return F.pad(frame.to(device).unsqueeze(0), padding)

    def similarity(first, second):
        small0 = F.interpolate(first, (32, 32), mode="bilinear", align_corners=False)
        small1 = F.interpolate(second, (32, 32), mode="bilinear", align_corners=False)
        return ssim_matlab(small0[:, :3], small1[:, :3])

    # A pair at a time is smaller than a 100-frame batch; lookahead stays live
    # across the whole sequence, so no artificial chunk boundary affects RIFE.
    output = torch.empty((total - 1) * multiplier + 1, frames.shape[0], h, w, dtype=torch.uint8).transpose(0, 1)
    output_index = 0
    with tqdm(total=max(0, total - 1), desc="Temporal Upsampling - RIFE", unit="frame") as progress:
        I1 = prepare(lastframe)
        temp = None
        while True:
            if cancelled():
                return None
            now = time.monotonic()
            if progress_callback is not None and now - last_report >= 1 / 3:
                progress_callback("Temporal Upsampling - RIFE", progress.n, progress.total)
                last_report = now
            if temp is not None:
                frame, temp = temp, None
            else:
                pos += 1
                frame = read(pos)
            if frame is None:
                break
            I0 = I1
            I1 = prepare(frame)
            ssim = similarity(I0, I1)
            if ssim > 0.996:
                pos += 1
                temp = read(pos)
                if temp is not None:  # a still pair: its second frame is drawn halfway to the one after it. The last pair keeps its own frame: nothing follows it here, and whatever continues the video starts from that frame
                    I1 = model.inference(I0, prepare(temp), 0.5, 1)
                    if cancelled():
                        return None
                    ssim = similarity(I0, I1)
                    frame = I1[0, :, :h, :w]
            add_frame(output, output_index, lastframe, h, w)
            output_index += 1
            for index in range(1, multiplier):
                if cancelled():
                    return None
                mid = I0 if ssim < 0.2 else model.inference(I0, I1, index / multiplier, 1)
                add_frame(output, output_index, mid, h, w)
                output_index += 1
                del mid
            lastframe = frame
            progress.update(1)
        if cancelled():
            return None
        add_frame(output, output_index, lastframe, h, w)
        if progress_callback is not None:
            progress_callback("Temporal Upsampling - RIFE", progress.total, progress.total)
    return output


def temporal_interpolation(model_path, frames, multiplier, device="cuda", *, previous_last_frame=None, abort_callback=None, progress_callback=None):
    if abort_callback is not None and abort_callback():
        return None
    if progress_callback is not None:
        progress_callback("Loading - RIFE Model")
    model = ModelV4()
    model.load_model(model_path, -1, device=device)
    model.eval()
    try:
        return process_frames(model, device, frames, multiplier, previous_last_frame=previous_last_frame, abort_callback=abort_callback, progress_callback=progress_callback)
    finally:
        model._grid_cache.clear()
