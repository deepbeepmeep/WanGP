# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Native CUDA background removal. MMGP owns every model's residency."""
import threading

import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm

from mmgp import offload, runtime
from shared.utils import offload_registry
from shared.utils.download_progress import check_download_cancelled, resolve_download_gen
from shared.utils.media_control import inference_checkpoint, controlled_model_loading, MediaProcessingAborted
from models.background_removal.common import abort_scope
from .assets import selected_model, ensure_assets

_lock = threading.RLock()
_loaded = None
_owner = 'background_removal'


def release():
    global _loaded
    with _lock:
        entry, _loaded = _loaded, None
        if entry is not None:
            entry['manager'].release()
            offload_registry.unregister_offloadobj(_owner, entry['manager'])


@controlled_model_loading
def _load(name, path, loading_callback=None):
    global _loaded
    if _loaded is not None and _loaded['name'] == name:
        return _loaded
    release()
    if not torch.cuda.is_available():
        raise RuntimeError('Native Background Removal Requires a CUDA GPU.')
    tqdm.write(f'Loading - Background Removal Model ({name})')
    from accelerate import init_empty_weights
    from models.background_removal.u2net import U2NET
    from models.background_removal.birefnet import BiRefNet
    from models.background_removal.swin import WindowAttention, split_map
    with init_empty_weights(include_buffers=True):
        model = U2NET() if name == 'u2net' else BiRefNet()
    for module in model.modules():
        if isinstance(module, WindowAttention):
            module.reset_index()
    offload.load_model_data(model, path, writable_tensors=False, default_dtype=torch.bfloat16,
                           fused_split_map=split_map() if name != 'u2net' else None)
    model._convertWeightsFloatTo = None
    model.eval().requires_grad_(False)
    modules, budgets = {'background_remover': model}, {'background_remover': 0}
    previous_device = torch.get_default_device()
    try:
        manager = offload.profile(modules, 0, budgets=budgets, quantizeTransformer=False, loading_callback=loading_callback)
    finally:
        torch.set_default_device(None if previous_device.type == 'cpu' else previous_device)
    _loaded = {'name': name, 'model': model, 'manager': manager}
    offload_registry.register_offloadobj(_owner, manager, release)
    return _loaded


def _checkpoint(gen, interrupted):
    inference_checkpoint()
    check_download_cancelled(gen)
    if interrupted is not None and interrupted():
        raise MediaProcessingAborted()


def _input(image, name):
    side = 320 if name == 'u2net' else 1024
    pixels = np.asarray(image.convert('RGB').resize((side, side), Image.Resampling.LANCZOS))
    # Match the former rembg input normalization, including pictures below full white.
    pixels = pixels / max(1, int(pixels.max()))
    pixels = (pixels - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    return torch.from_numpy(pixels.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)


class NativeSession:
    def __init__(self, model_name='u2net'):
        self.model_name = selected_model(model_name)

    @torch.inference_mode()
    def predict(self, image, *, gen=None, interrupted=None):
        from shared.attention import attention_shared_state
        gen = resolve_download_gen(gen)
        checkpoint = lambda: _checkpoint(gen, interrupted)
        checkpoint()
        path = ensure_assets(gen=gen, model_name=self.model_name)
        with _lock, abort_scope(checkpoint), runtime.execution_scope(checkpoint), attention_shared_state():
            entry = _load(self.model_name, path)
            model, manager = entry['model'], entry['manager']
            try:
                checkpoint()
                dtype = next(model.parameters()).dtype
                inputs = _input(image, self.model_name).to(device=runtime.execution_device(model), dtype=dtype)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(inputs).float()
                probabilities = logits.sigmoid()[0, 0]
                minimum, maximum = probabilities.amin(), probabilities.amax()
                mask = ((probabilities - minimum) / (maximum - minimum).clamp_min(1e-8) * 255).clamp(0, 255).byte().cpu().numpy()
                mask = Image.fromarray(mask, 'L')
                del inputs, logits, probabilities
                checkpoint()
                return [mask.resize(image.size, Image.Resampling.LANCZOS)]
            finally:
                # CPU checkpoint pages remain reusable; GPU weights/caches yield to Deepy's next segment.
                manager.unload_all()


def _matting(image, mask, foreground, background, erode):
    # rembg's MIT alpha-matting recipe; no rembg/ONNX import. LICENSE_REMBG accompanies the implementation.
    from scipy.ndimage import binary_erosion
    from pymatting import estimate_alpha_cf, estimate_foreground_ml
    values = np.asarray(mask)
    structure = np.ones((erode, erode), np.uint8) if erode > 0 else None
    fg = binary_erosion(values > foreground, structure=structure)
    bg = binary_erosion(values < background, structure=structure, border_value=1)
    trimap = np.full(values.shape, 128, np.uint8)
    trimap[fg], trimap[bg] = 255, 0
    pixels = np.asarray(image.convert('RGB')) / 255.0
    alpha = estimate_alpha_cf(pixels, trimap / 255.0)
    colors = estimate_foreground_ml(pixels, alpha)
    return Image.fromarray(np.clip(np.dstack((colors, alpha)) * 255, 0, 255).astype(np.uint8))


def remove(image, *, session=None, alpha_matting=False, alpha_matting_erode_size=10,
           alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10,
           only_mask=False, bgcolor=None, gen=None, interrupted=None):
    image = ImageOps.exif_transpose(image)
    mask = (session or NativeSession()).predict(image, gen=gen, interrupted=interrupted)[0]
    if only_mask:
        return mask
    if alpha_matting:
        try:
            result = _matting(image, mask, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size)
        except ValueError:
            result = Image.composite(image, Image.new('RGBA', image.size, 0), mask)
    else:
        result = Image.composite(image, Image.new('RGBA', image.size, 0), mask)
    return Image.alpha_composite(Image.new('RGBA', image.size, tuple(bgcolor)), result) if bgcolor is not None else result
