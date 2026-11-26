"""FlashVSR's UI preference and status. WanGP's shared Sparge core resolves and runs it."""
import torch
from tqdm import tqdm
from mmgp import cuda_graphs

from shared import sparge_core
from .sparse_backend_config import normalize_sparse_backend

_preference = "auto"
_printed = None
cuda_graphs.register_key(lambda: ("flashvsr_sparge", _preference))


def set_sparse_backend(backend):
    global _preference
    _preference = normalize_sparse_backend(backend)
    return _preference


def attention_mode(backend=None):
    return sparge_core.MODES[_preference if backend is None else normalize_sparse_backend(backend)]


def _device():
    return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")


def sparse_attention_requirement_message(backend=None):
    try:
        sparge_core.require(_device(), mode=attention_mode(backend))
    except RuntimeError as exc:
        return str(exc)
    return None


def sparge_attention_available():
    return sparse_attention_requirement_message("sparge") is None


def require_sparge_attention():
    sparge_core.require(_device(), mode=attention_mode())


def get_sparse_backend_name():
    return sparge_core.require(_device(), mode=attention_mode())["name"]


def staged_sparge_available(device=None, head_dim=128):
    return sparge_core.staged_settings(device or _device(), head_dim, attention_mode()) is not None


def log_sparse_backend():
    global _printed
    name = get_sparse_backend_name()
    if _printed != name:
        tqdm.write(f"[FlashVSR] Sparse attention backend: {name}")
        _printed = name


def sparse_attention(qkv_list, mask_id, recycle_q=False):
    log_sparse_backend()
    return sparge_core.attention(qkv_list, mask_id, mode=attention_mode(), recycle_q=recycle_q)
