# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Production Modal Application for Wan2GP (High-Performance GPU-Accelerated)
Architecture: Immutable Base Container + Persistent Volume Venv (--system-site-packages)

Fully compliant with:
  - WanVideo-2nd-stage services/gpu-provider.ts
  - WanVideo-2nd-stage server.ts orchestration
  - High-performance GPU tiers (L4, L40S, A100-40GB, A100-80GB, H100)

KEY HIGHLIGHTS:
1. IMMUTABLE BASE IMAGE:
   - Ubuntu 22.04 + CUDA 12.4.1 devel + Python 3.11
   - PyTorch 2.5.1 + TorchVision + Torchaudio (matching CUDA)
   - Pre-compiled kernels: FlashAttention, Triton, SageAttention, mmgp
   - Fast installer: uv
2. PERSISTENT VOLUME VENV (/models/venv):
   - Created with --system-site-packages on wan2gp-models volume
   - Inherits 4GB+ PyTorch & CUDA binaries with ZERO disk duplication
   - Holds fast-changing dependencies: transformers, diffusers, huggingface_hub, etc.
   - Synchronized on CPU via uv pip install in 3-5 seconds with $0 GPU cost
3. ZERO-TOUCH MODEL WEIGHTS:
   - Checkpoints remain untouched in /models/Wan2GP/ckpts across all updates
"""

import os
import sys
import json
import uuid
import shutil
import hashlib
import zipfile
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import modal
from fastapi import FastAPI, Request, HTTPException, Query, Header
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, StreamingResponse

# ---------------------------------------------------------------------------
# Configuration & Volumes
# ---------------------------------------------------------------------------
APP_NAME = os.environ.get("MODAL_APP_NAME", "wan2gp-cloud")
EXPECTED_API_KEY = os.environ.get("MODAL_API_KEY", "")

# Source repository used by the Modal worker. Override with WAN2GP_REPO_URL /
# WAN2GP_GIT_REF for a custom fork or branch.
WAN2GP_REPO_URL = os.environ.get("WAN2GP_REPO_URL", "https://github.com/deepbeepmeep/WanGP.git")
WAN2GP_GIT_REF = os.environ.get("WAN2GP_GIT_REF", "main")

# Maximum concurrent GPU containers (0 = unlimited, Modal account quota applies)
MAX_CONCURRENT_CLOUD_JOBS = int(os.environ.get("MAX_CONCURRENT_CLOUD_JOBS", "0"))

OUTPUTS_VOL_NAME = os.environ.get("MODAL_VOLUME_NAME", "wan2gp-outputs")
MODELS_VOL_NAME = os.environ.get("MODAL_MODELS_VOLUME_NAME", "wan2gp-models")
CACHE_VOL_NAME = "wan2gp-cache"

outputs_volume = modal.Volume.from_name(OUTPUTS_VOL_NAME, create_if_missing=True)
models_volume = modal.Volume.from_name(MODELS_VOL_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOL_NAME, create_if_missing=True)

OUTPUTS_MOUNT = "/outputs"
MODELS_MOUNT = "/models"
CACHE_MOUNT = "/cache"

VENV_PATH = Path(MODELS_MOUNT) / "venv"
VENV_PYTHON = VENV_PATH / "bin" / "python"

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Source Patching Helper (Build-time & Runtime)
# ---------------------------------------------------------------------------
def apply_source_patches(wgp_dir: Path, log_fn=print):
    """
    Applies source-level patches to a Wan2GP installation:
    1. Triton compatibility (removes do_not_specialize_on_alignment in nanovllm).
    2. higgs_audio_v2_tokenizer (adds exist_ok=True).
    3. download_progress.py (adds **kwargs to _http_get & _xet_get).
    Can be called during container build time OR dynamically at runtime on volume code.
    """
    try:
        from shared.patch_manager import apply_source_patches as _shared_apply
        _shared_apply(wgp_dir, log_fn=log_fn)
        return
    except Exception:
        pass
    import re
    if not wgp_dir or not wgp_dir.exists():
        return

    # 1. Triton compatibility (nanovllm)
    nanovllm_dir = wgp_dir / "shared" / "llm_engines" / "nanovllm"
    if nanovllm_dir.exists():
        for py_file in nanovllm_dir.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                original = content
                if "do_not_specialize_on_alignment" in content:
                    content = re.sub(r',\s*do_not_specialize_on_alignment=\([^)]*\)', '', content)
                    content = re.sub(r'do_not_specialize_on_alignment=\([^)]*\),\s*', '', content)
                content = re.sub(
                    r'do_not_specialize=\(([a-zA-Z0-9_"\']+)\)',
                    r'do_not_specialize=(\1,)',
                    content
                )
                if content != original:
                    py_file.write_text(content, encoding="utf-8")
                    log_fn(f"[PATCH] Patched {py_file.name} for Triton compat")
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {py_file}: {e}")

    # 2. higgs_audio_v2_tokenizer exist_ok=True
    tts_tok = wgp_dir / "models" / "TTS" / "omnivoice" / "higgs_audio_v2_tokenizer" / "__init__.py"
    if tts_tok.exists():
        try:
            content = tts_tok.read_text(encoding="utf-8")
            original = content
            content = content.replace(
                'AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig)',
                'AutoConfig.register("higgs_audio_v2_tokenizer", HiggsAudioV2TokenizerConfig, exist_ok=True)'
            )
            content = content.replace(
                'AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel)',
                'AutoModel.register(HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, exist_ok=True)'
            )
            if content != original:
                tts_tok.write_text(content, encoding="utf-8")
                log_fn("[PATCH] Patched higgs_audio_v2_tokenizer for exist_ok=True")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch tts tokenizer: {e}")

    # 3. download_progress.py: accept **kwargs & huggingface_hub _request_wrapper compat
    dl_progress = wgp_dir / "shared" / "utils" / "download_progress.py"
    if dl_progress.exists():
        try:
            content = dl_progress.read_text(encoding="utf-8")
            original = content
            if "_tqdm_bar=None):" in content:
                content = content.replace("_tqdm_bar=None):", "_tqdm_bar=None, **kwargs):")

            hf_target = "from huggingface_hub import file_download as hf"
            hf_shim = (
                "from huggingface_hub import file_download as hf\n"
                "    if not hasattr(hf, '_request_wrapper'):\n"
                "        import requests\n"
                "        hf._request_wrapper = lambda *args, **kwargs: requests.request(*args, **kwargs)\n"
                "    if not hasattr(hf, '_get_file_length_from_http_response'):\n"
                "        hf._get_file_length_from_http_response = lambda r: int(r.headers.get('content-length', 0)) or None\n"
                "    if not hasattr(hf, '_adjust_range_header'):\n"
                "        hf._adjust_range_header = lambda r, c: f'bytes={c}-'\n"
            )
            if "requests.request(*args, **kwargs)" not in content:
                if "http_backoff" in content:
                    content = content.replace("lambda method, url, **p: http_backoff(requests.request, method=method, url=url, **p)", "lambda *args, **kwargs: requests.request(*args, **kwargs)")
                elif hf_target in content:
                    content = content.replace(hf_target, hf_shim, 1)
            if content != original:
                dl_progress.write_text(content, encoding="utf-8")
                log_fn("[PATCH] Patched download_progress.py for huggingface_hub compat")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch download_progress.py: {e}")

    # 0. Restore any altered files in models/ back to pristine upstream git state
    if (wgp_dir / ".git").exists():
        try:
            subprocess.run(["git", "checkout", "--", "models/"], cwd=str(wgp_dir), capture_output=True)
            log_fn("[PATCH] Restored models/ in Wan2GP git directory to pristine state")
        except Exception:
            pass

    # 4. wgp.py: Add torch.accelerator & universal RoPE compatibility shim
    wgp_py = wgp_dir / "wgp.py"
    if wgp_py.exists():
        try:
            wgp_code = wgp_py.read_text(encoding="utf-8")
            shim = (
                "# --- Torch Accelerator & RoPE Compat Shim ---\n"
                "import torch\n"
                "if not hasattr(torch, 'accelerator'):\n"
                "    class _TorchAccShim:\n"
                "        @staticmethod\n"
                "        def is_available(): return False\n"
                "    torch.accelerator = _TorchAccShim\n"
                "try:\n"
                "    import flash_attn_2_cuda\n"
                "except Exception:\n"
                "    import sys\n"
                "    for _k in list(sys.modules.keys()):\n"
                "        if _k == 'flash_attn' or _k.startswith('flash_attn.'):\n"
                "            sys.modules.pop(_k, None)\n"
                "    sys.modules['flash_attn'] = None\n"
                "    sys.modules['flash_attn_2_cuda'] = None\n"
                "    try:\n"
                "        import transformers.utils.import_utils as _iu\n"
                "        _iu.is_flash_attn_2_available = lambda: False\n"
                "        _iu.is_flash_attn_greater_or_equal_2_10 = lambda: False\n"
                "    except Exception:\n"
                "        pass\n"
                "try:\n"
                "    import transformers.integrations.heterogeneity.configuration_utils as _hetero_cfg\n"
                "    _hetero_cfg.HeterogeneousConfigMixin.__getattribute__ = object.__getattribute__\n"
                "    _hetero_cfg.HeterogeneousConfigMixin.allow_global_per_layer_attribute_access = True\n"
                "except Exception:\n"
                "    pass\n"
                "try:\n"
                "    from transformers.configuration_utils import PretrainedConfig\n"
                "    PretrainedConfig.allow_global_per_layer_attribute_access = True\n"
                "except Exception:\n"
                "    pass\n"
                "try:\n"
                "    import transformers.modeling_rope_utils as _rope_utils\n"
                "    if hasattr(_rope_utils, 'ROPE_INIT_FUNCTIONS') and 'default' not in _rope_utils.ROPE_INIT_FUNCTIONS:\n"
                "        def _default_rope_fn(cfg, dev=None, **_kw):\n"
                "            base = getattr(cfg, 'rope_theta', 10000.0)\n"
                "            partial_factor = getattr(cfg, 'partial_rotary_factor', 1.0)\n"
                "            h_dim = getattr(cfg, 'head_dim', getattr(cfg, 'hidden_size', 4096) // getattr(cfg, 'num_attention_heads', 32))\n"
                "            dim = int(h_dim * partial_factor)\n"
                "            inv_f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(dev) / dim))\n"
                "            return inv_f, 1.0\n"
                "        _rope_utils.ROPE_INIT_FUNCTIONS['default'] = getattr(_rope_utils, '_compute_default_rope_parameters', _default_rope_fn)\n"
                "except Exception:\n"
                "    pass\n"
                "# ----------------------------------------------\n"
            )
            # Remove legacy shim if present to avoid duplicate headers
            legacy_shim = (
                "# --- Torch Accelerator Compat Shim ---\n"
                "import torch\n"
                "if not hasattr(torch, 'accelerator'):\n"
                "    class _TorchAccShim:\n"
                "        @staticmethod\n"
                "        def is_available(): return False\n"
                "    torch.accelerator = _TorchAccShim\n"
                "# --------------------------------------\n"
            )
            if legacy_shim in wgp_code:
                wgp_code = wgp_code.replace(legacy_shim, "")
            
            if "_default_rope_fn" not in wgp_code:
                wgp_py.write_text(shim + wgp_code, encoding="utf-8")
                log_fn("[PATCH] Injected torch.accelerator & RoPE compatibility shim into wgp.py")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch wgp.py: {e}")

    # 4a. shared/attention.py: exception safety for flash_attn import
    attn_py = wgp_dir / "shared" / "attention.py"
    if attn_py.exists():
        try:
            content = attn_py.read_text(encoding="utf-8")
            if "except ModuleNotFoundError:" in content and "FLASH_ATTN_2_AVAILABLE = False" in content:
                content = content.replace(
                    "except ModuleNotFoundError:\n    FLASH_ATTN_2_AVAILABLE = False",
                    "except (ModuleNotFoundError, ImportError, Exception):\n    FLASH_ATTN_2_AVAILABLE = False"
                )
                attn_py.write_text(content, encoding="utf-8")
                log_fn("[PATCH] Patched shared/attention.py for safe flash_attn import")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch shared/attention.py: {e}")


    # 4b. Qwen3VL pad_token_id compatibility for Hugging Face Transformers
    for q_rel in [
        "models/ideogram4/qwen3_vl_transformers.py",
        "models/hidream/qwen3_vl_transformers.py",
    ]:
        q_path = wgp_dir / q_rel
        if q_path.exists():
            try:
                txt = q_path.read_text(encoding="utf-8")
                if "self.padding_idx = config.pad_token_id" in txt:
                    txt = txt.replace(
                        "self.padding_idx = config.pad_token_id",
                        'self.padding_idx = getattr(config, "pad_token_id", None)'
                    )
                    q_path.write_text(txt, encoding="utf-8")
                    log_fn(f"[PATCH] Patched {q_rel} for pad_token_id attribute safety")
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")

    for q_rel in [
        "models/ideogram4/qwen3_vl_configuration.py",
        "models/hidream/qwen3_vl_configuration.py",
    ]:
        q_path = wgp_dir / q_rel
        if q_path.exists():
            try:
                txt = q_path.read_text(encoding="utf-8")
                if "pad_token_id=None" not in txt and "class Qwen3VLTextConfig(PretrainedConfig):" in txt:
                    txt = txt.replace(
                        "attention_dropout=0.0,\n        **kwargs,",
                        "attention_dropout=0.0,\n        pad_token_id=None,\n        **kwargs,"
                    )
                    txt = txt.replace(
                        "self.vocab_size = vocab_size",
                        "self.pad_token_id = pad_token_id\n        self.vocab_size = vocab_size"
                    )
                    q_path.write_text(txt, encoding="utf-8")
                    log_fn(f"[PATCH] Patched {q_rel} for pad_token_id config default")
            except Exception as e:
                log_fn(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")

    # 5. diffusers torch_utils: patch if available
    for p in [
        Path("/usr/local/lib/python3.11/site-packages/diffusers/utils/torch_utils.py"),
        Path(MODELS_MOUNT) / "venv/lib/python3.11/site-packages/diffusers/utils/torch_utils.py"
    ]:
        if p.exists():
            try:
                txt = p.read_text(encoding="utf-8")
                if "if torch.accelerator.is_available():" in txt:
                    p.write_text(txt.replace("if torch.accelerator.is_available():", "if hasattr(torch, 'accelerator') and torch.accelerator.is_available():"), encoding="utf-8")
                    log_fn(f"[PATCH] Patched {p.name} for torch.accelerator")
            except Exception as e:
                pass

    # 6. sitecustomize.py shim (Universal compatibility for PyTorch & Transformers across all models)
    sitecustomize_code = (
        "import sys\n"
        "# 1. Torch Accelerator compat shim\n"
        "try:\n"
        "    import torch\n"
        "    if not hasattr(torch, 'accelerator'):\n"
        "        class _TorchAccShim:\n"
        "            @staticmethod\n"
        "            def is_available(): return False\n"
        "        torch.accelerator = _TorchAccShim\n"
        "except Exception:\n"
        "    pass\n\n"
        "# 2. Universal RoPE compatibility shim for transformers (fixes KeyError: 'default')\n"
        "try:\n"
        "    import transformers.modeling_rope_utils as _rope_utils\n"
        "    if hasattr(_rope_utils, 'ROPE_INIT_FUNCTIONS') and 'default' not in _rope_utils.ROPE_INIT_FUNCTIONS:\n"
        "        if hasattr(_rope_utils, '_compute_default_rope_parameters'):\n"
        "            _rope_utils.ROPE_INIT_FUNCTIONS['default'] = _rope_utils._compute_default_rope_parameters\n"
        "        else:\n"
        "            import torch\n"
        "            def _default_rope_init(config, device=None, **kwargs):\n"
        "                base = getattr(config, 'rope_theta', 10000.0)\n"
        "                partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)\n"
        "                head_dim = getattr(config, 'head_dim', getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32))\n"
        "                dim = int(head_dim * partial_rotary_factor)\n"
        "                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))\n"
        "                return inv_freq, 1.0\n"
        "            _rope_utils.ROPE_INIT_FUNCTIONS['default'] = _default_rope_init\n"
        "except Exception:\n"
        "    pass\n\n"
        "# 3. Universal Transformers Heterogeneity compat shim\n"
        "try:\n"
        "    import transformers.integrations.heterogeneity.configuration_utils as _hetero_cfg\n"
        "    _hetero_cfg.HeterogeneousConfigMixin.__getattribute__ = object.__getattribute__\n"
        "    _hetero_cfg.HeterogeneousConfigMixin.allow_global_per_layer_attribute_access = True\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    from transformers.configuration_utils import PretrainedConfig\n"
        "    PretrainedConfig.allow_global_per_layer_attribute_access = True\n"
        "except Exception:\n"
        "    pass\n\n"
        "# 4. Flash Attention integrity check & safe SDPA fallback\n"
        "try:\n"
        "    import torch\n"
        "    import flash_attn_2_cuda\n"
        "except Exception:\n"
        "    import sys\n"
        "    for _k in list(sys.modules.keys()):\n"
        "        if _k == 'flash_attn' or _k.startswith('flash_attn.'):\n"
        "            sys.modules.pop(_k, None)\n"
        "    sys.modules['flash_attn'] = None\n"
        "    sys.modules['flash_attn_2_cuda'] = None\n"
        "    try:\n"
        "        import transformers.utils.import_utils as _iu\n"
        "        _iu.is_flash_attn_2_available = lambda: False\n"
        "        _iu.is_flash_attn_greater_or_equal_2_10 = lambda: False\n"
        "    except Exception:\n"
        "        pass\n"
    )
    target_site_dirs = set()
    try:
        import site
        if hasattr(site, "getsitepackages"):
            target_site_dirs.update(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    target_site_dirs.add(Path("/usr/local/lib/python3.11/site-packages"))
    venv_lib = Path(MODELS_MOUNT) / "venv" / "lib"
    if venv_lib.exists():
        for sp in venv_lib.glob("python*/site-packages"):
            target_site_dirs.add(sp)
    for site_dir in target_site_dirs:
        if site_dir.exists():
            try:
                sc = site_dir / "sitecustomize.py"
                sc.write_text(sitecustomize_code, encoding="utf-8")
                log_fn(f"[PATCH] Injected universal sitecustomize.py into {site_dir}")
            except Exception as e:
                pass

    # 7. LTX-2 Gemma4UnifiedTextConfig heterogeneity compatibility
    gemma4_path = wgp_dir / "models" / "ltx2" / "ltx_core" / "text_encoders" / "gemma" / "gemma4_unified.py"
    if gemma4_path.exists():
        try:
            content = gemma4_path.read_text(encoding="utf-8")
            original = content
            if "allow_global_per_layer_attribute_access = True" not in content:
                content = content.replace(
                    'class Gemma4UnifiedTextConfig(PretrainedConfig):\n    model_type = "gemma4_unified_text"',
                    'class Gemma4UnifiedTextConfig(PretrainedConfig):\n    model_type = "gemma4_unified_text"\n    allow_global_per_layer_attribute_access = True'
                )
            if 'config.__dict__.pop("_heterogeneity_spec", None)' not in content:
                content = content.replace(
                    'class Gemma4UnifiedTextRotaryEmbedding(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__()',
                    'class Gemma4UnifiedTextRotaryEmbedding(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__()\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True'
                )
                content = content.replace(
                    'class Gemma4UnifiedTextAttention(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):\n        super().__init__()',
                    'class Gemma4UnifiedTextAttention(nn.Module):\n    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):\n        super().__init__()\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True'
                )
                content = content.replace(
                    'class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__(config)',
                    'class Gemma4UnifiedTextModel(Gemma4UnifiedPreTrainedModel):\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True\n        super().__init__(config)'
                )
                content = content.replace(
                    'class Gemma4UnifiedForCausalLM(Gemma4UnifiedPreTrainedModel):\n    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}\n\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        super().__init__(config)',
                    'class Gemma4UnifiedForCausalLM(Gemma4UnifiedPreTrainedModel):\n    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}\n\n    def __init__(self, config: Gemma4UnifiedTextConfig):\n        config.__dict__.pop("_heterogeneity_spec", None)\n        config.__dict__["allow_global_per_layer_attribute_access"] = True\n        super().__init__(config)'
                )
                content = content.replace(
                    "num_key_value_heads = config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads",
                    "num_key_value_heads = (config.__dict__.get('num_global_key_value_heads') or getattr(config, 'num_global_key_value_heads', 1)) if self.use_alternative_attention else (config.__dict__.get('num_key_value_heads') or getattr(config, 'num_key_value_heads', 4))"
                )
                content = content.replace(
                    "num_key_value_heads = config.num_global_key_value_heads if self.use_alternative_attention else (config.__dict__.get('num_key_value_heads') or getattr(config, 'num_key_value_heads', 4))",
                    "num_key_value_heads = (config.__dict__.get('num_global_key_value_heads') or getattr(config, 'num_global_key_value_heads', 1)) if self.use_alternative_attention else (config.__dict__.get('num_key_value_heads') or getattr(config, 'num_key_value_heads', 4))"
                )
                content = content.replace(
                    "self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim",
                    "self.head_dim = (config.__dict__.get('head_dim') or getattr(config, 'head_dim', 256)) if self.is_sliding else getattr(config, 'global_head_dim', 512)"
                )
                content = content.replace(
                    'head_dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim',
                    'head_dim = (config.__dict__.get("global_head_dim") or getattr(config, "global_head_dim", 512)) if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))'
                )
                content = content.replace(
                    'head_dim = config.global_head_dim if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))',
                    'head_dim = (config.__dict__.get("global_head_dim") or getattr(config, "global_head_dim", 512)) if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))'
                )
            if content != original:
                gemma4_path.write_text(content, encoding="utf-8")
                log_fn("[PATCH] Patched gemma4_unified.py for complete heterogeneity compat")
        except Exception as e:
            log_fn(f"[PATCH-WARN] Failed to patch gemma4_unified.py: {e}")


def _bake_patches():
    """
    BUILD-TIME ONLY: Called via modal.Image.run_function() during image build.
    Applies source-level patches to /root/Wan2GP and installs helper wheels.
    """
    import subprocess
    import sys
    from pathlib import Path

    apply_source_patches(Path("/root/Wan2GP"), log_fn=print)

    # 4. smplfitter + chumpy wheels
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://github.com/deepbeepmeep/smplfitter/releases/download/v0.2.10/smplfitter-0.2.10-py3-none-any.whl",
        "https://github.com/deepbeepmeep/chumpy/releases/download/v0.71/chumpy-0.71-py3-none-any.whl"
    ], check=False)
# ---------------------------------------------------------------------------
# Base Container Image Definition
# ---------------------------------------------------------------------------
wan2gp_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libportaudio2",
        "curl",
        "wget",
        "xz-utils",
        "build-essential",
        "ninja-build",
        "cmake",
        "clang",
        "libsox-dev"
    )
    .env({
        "TORCH_CUDA_ARCH_LIST": "8.0;8.9;9.0+PTX",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_ALLOW_TF32_CUBLAS": "1",
        "TORCH_ALLOW_TF32_CUDNN": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTHONUNBUFFERED": "1",
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
        "TORCH_HOME": "/cache/torch",
        "TRITON_CACHE_DIR": "/cache/triton"
    })
    .pip_install(
        "uv>=0.4.0",
        "ninja",
        "wheel",
        "packaging",
        "setuptools<=75.8.2",
        "triton>=3.0.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        "mmgp==3.8.2",
        "rotary-embedding-torch>=0.5.3",
        "onnxruntime",
        "rembg",
        "scipy",
        "av",
        "ffmpeg-python",
        "pygame",
        "pyloudnorm",
        "mutagen",
        "einshape",
        "timm",
        "decord",
        "peft",
        "gguf",
        "torchdiffeq",
        "tensordict",
        "omegaconf",
        "soundfile",
        "ftfy",
        "moviepy==1.0.3",
        "gradio-rangeslider",
        "fastapi[standard]>=0.115.0",
        "uvicorn>=0.30.0",
        "pydantic>=2.0.0",
        "diffusers>=0.31.0",
        "transformers>=4.46.0",
        "accelerate>=1.0.0",
        "gradio==5.29.0",
        "safetensors>=0.4.5",
        "einops>=0.8.0",
        "sentencepiece>=0.2.0",
        "imageio>=2.36.0",
        "imageio-ffmpeg>=0.5.1",
        "opencv-python-headless>=4.10.0",
        "numpy==2.1.2",
        "optimum-quanto>=0.2.6",
        "hf-transfer>=0.1.8",
        "huggingface-hub>=0.28.1",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
        "openai-whisper",
        "loguru",
        "orjson",
        "tiktoken",
        "open_clip_torch",
        "librosa",
        "hydra-core",
        "easydict",
        "conformer",
        "vector-quantize-pytorch",
        "Cython",
        "markdown",
        "apprise",
        "keyring",
        "piexif",
        "mcp",
        "misaki",
        "gitpython",
        "stringzilla",
        "xxhash",
        "munch",
        "matplotlib",
        "pandas",
        "taichi",
        "sageattention",
        # Complete media, reference, and audio separation stack
        "audio-separator==0.36.1",
        "pyannote.audio==3.3.2",
        "sherpa-onnx==1.13.2",
        "speechbrain==1.0.3",
        "torchcodec==0.10.0",
        "insightface==0.7.3",
        "facexlib==0.3.0",
        "mir_eval==0.8.2",
        "pretty_midi==0.2.11.post0",
        "mido==1.3.3",
        "symusic==0.6.0",
        "fugashi>=1.2.0",
        "unidic-lite>=1.0.0",
        "num2words==0.5.14",
        "s3tokenizer",
        "spacy_pkuseg",
        "spacy",
        "pycocotools",
        "segment-anything",
        "iopath>=0.1.10",
        "phonemizer-fork",
        "espeakng-loader",
        "wetext==0.1.2",
        "hf_xet>=1.5.2",
        "runpod>=1.7.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu124 --progress-bar off"
    )
    .run_commands(
        f"git clone --depth 1 --branch {WAN2GP_GIT_REF} {WAN2GP_REPO_URL} /root/Wan2GP",
        "pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' --no-build-isolation || echo 'Flash-attn wheel install failed, non-fatal'"
    )
    .run_function(_bake_patches)
)

# ---------------------------------------------------------------------------
# Volume Venv Management: Self-Healing & Synchronization
# ---------------------------------------------------------------------------
def compute_requirements_hash(req_path: Path) -> str:
    """Computes SHA256 of requirements.txt for change detection."""
    if not req_path.exists():
        return ""
    return hashlib.sha256(req_path.read_bytes()).hexdigest()[:16]

def ensure_volume_venv(log_file: Optional[Path] = None, allow_install: bool = False) -> Path:
    """
    Ensures a persistent virtual environment exists at /models/venv
    configured with --system-site-packages.
    """
    def log(msg: str):
        print(msg)
        if log_file and log_file.parent.exists():
            with open(log_file, "a") as f:
                f.write(f"{msg}\n")

    venv_dir = Path(MODELS_MOUNT) / "venv"
    venv_py = venv_dir / "bin" / "python"
    req_file = Path(MODELS_MOUNT) / "Wan2GP" / "requirements.txt"
    if not req_file.exists():
        fallback_req = Path("/root/Wan2GP/requirements.txt")
        if fallback_req.exists():
            req_file = fallback_req
    hash_file = venv_dir / ".req_hash"

    # Step 1: Create venv if missing
    if not venv_py.exists():
        log("[VENV] Creating persistent volume venv with --system-site-packages at /models/venv...")
        venv_dir.mkdir(parents=True, exist_ok=True)
        
        # Try uv first, fallback to standard venv
        uv_cmd = shutil.which("uv")
        if uv_cmd:
            cmd = [uv_cmd, "venv", str(venv_dir), "--system-site-packages", "--python", sys.executable]
        else:
            cmd = [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            log(f"[VENV-ERROR] Failed to create venv: {res.stderr}")
            return Path(sys.executable)
        
        log("[VENV] Created /models/venv successfully.")

    def _is_missing_core_pkgs(py_bin: Path) -> bool:
        test_script = "import audio_separator, insightface, facexlib"
        chk = subprocess.run([str(py_bin), "-c", test_script], capture_output=True)
        return chk.returncode != 0

    # Step 2: Synchronize requirements if changed, missing, or explicitly requested
    missing_core = _is_missing_core_pkgs(venv_py)
    current_hash = compute_requirements_hash(req_file) if req_file.exists() else ""
    saved_hash = hash_file.read_text().strip() if hash_file.exists() else ""
    needs_sync = allow_install or missing_core or (current_hash and current_hash != saved_hash)

    if req_file.exists() and needs_sync:
        log(f"[VENV] Syncing all dependencies from {req_file} (missing_core={missing_core}, allow_install={allow_install})...")
        uv_cmd = shutil.which("uv")
        uv_ok = False
        if uv_cmd:
            install_cmd = [
                uv_cmd, "pip", "install",
                "--python", str(venv_py),
                "-r", str(req_file)
            ]
            res = subprocess.run(install_cmd, capture_output=True, text=True)
            uv_ok = (res.returncode == 0)
            if not uv_ok:
                log(f"[VENV-WARN] uv pip install returned code {res.returncode}: {res.stderr[:200]}")

        # Fallback to standard pip if uv failed or had dependency conflicts
        if not uv_ok:
            log("[VENV] Running standard pip install for full requirements...")
            fallback_cmd = [
                str(venv_py), "-m", "pip", "install", "--no-cache-dir",
                "-r", str(req_file)
            ]
            subprocess.run(fallback_cmd, capture_output=True, text=True)

        # Explicitly ensure critical media separation and face analysis packages + wheels
        critical_pkgs = [
            "audio-separator==0.36.1",
            "insightface==0.7.3",
            "facexlib==0.3.0",
            "pyannote.audio==3.3.2",
            "sherpa-onnx==1.13.2",
            "speechbrain==1.0.3",
            "torchcodec==0.10.0",
            "runpod>=1.7.0",
            "hf-transfer",
            "https://github.com/deepbeepmeep/smplfitter/releases/download/v0.2.10/smplfitter-0.2.10-py3-none-any.whl",
            "https://github.com/deepbeepmeep/chumpy/releases/download/v0.71/chumpy-0.71-py3-none-any.whl"
        ]
        subprocess.run([str(venv_py), "-m", "pip", "install", "--no-cache-dir", *critical_pkgs], capture_output=True)

        if current_hash:
            hash_file.write_text(current_hash)
        log("[VENV] Successfully verified and updated all dependencies in /models/venv.")
    elif req_file.exists():
        log("[VENV] Using volume venv (fast startup, verified dependencies).")

    return venv_py if venv_py.exists() else Path(sys.executable)

# ---------------------------------------------------------------------------
# GPU Configuration & Tuning Profiles
# ---------------------------------------------------------------------------
GPU_PROFILE_MAP = {
    "H100": {"profile": 1, "attention": "sage2"},
    "A100_80GB": {"profile": 1, "attention": "sage2"},
    "A100_40GB": {"profile": 2, "attention": "sage2"},
    "L40S": {"profile": 1, "attention": "sage2"},
    "L4": {"profile": 3, "attention": "sage2"},
    "RTX_PRO_6000": {"profile": 1, "attention": "sage2"},
    "T4": {"profile": 5, "attention": "sdpa"}
}

def resolve_gpu_tuning(requested_gpu: str) -> Tuple[int, str, str]:
    norm_gpu = requested_gpu.upper().replace("-", "_").replace(" ", "_")
    detected_device = "Unknown"

    try:
        import torch
        if torch.cuda.is_available():
            detected_device = torch.cuda.get_device_name(0).upper()
    except Exception:
        pass

    matched_key = "L40S"
    test_str = f"{detected_device} {norm_gpu}"

    if "H100" in test_str:
        matched_key = "H100"
    elif "80GB" in test_str or ("A100" in test_str and "80" in test_str):
        matched_key = "A100_80GB"
    elif "A100" in test_str:
        matched_key = "A100_40GB"
    elif "40S" in test_str or "L40" in test_str:
        matched_key = "L40S"
    elif "L4" in test_str:
        matched_key = "L4"
    elif "6000" in test_str:
        matched_key = "RTX_PRO_6000"
    elif "T4" in test_str:
        matched_key = "T4"

    tuning = GPU_PROFILE_MAP.get(matched_key, GPU_PROFILE_MAP["L40S"])
    target_attention = tuning.get("attention", "auto")

    # Verify if target attention mode can actually be imported; fallback to 'auto' or 'sdpa'
    attention = "auto"
    if target_attention in ("sage", "sage2"):
        try:
            import sageattention  # noqa: F401
            attention = target_attention
        except Exception:
            try:
                import flash_attn  # noqa: F401
                attention = "flash"
            except Exception:
                attention = "sdpa"
    elif target_attention == "flash":
        try:
            import flash_attn  # noqa: F401
            attention = "flash"
        except Exception:
            attention = "sdpa"
    elif target_attention == "sdpa":
        attention = "sdpa"

    return tuning["profile"], attention, detected_device

def setup_wan2gp_environment(wgp_dir: Path, profile_num: int, attention_mode: str, log_file: Path):
    """
    Runtime setup (~50ms total):
    1. Ensures source patches (Triton, tokenizer, etc.) are applied to the active Wan2GP code.
    2. Symlink model weights from /models volume into Wan2GP/models (skips Wan2GP code)
    3. Writes optimal wgp_config.json for active GPU
    """
    def log_line(msg: str):
        with open(log_file, "a") as log:
            log.write(msg + chr(10))

    # 1. Apply runtime patches to active Wan2GP directory (critical for persistent volume /models/Wan2GP)
    apply_source_patches(wgp_dir, log_fn=log_line)

    wgp_models_dir = wgp_dir / "models"
    wgp_models_dir.mkdir(parents=True, exist_ok=True)

    models_volume_path = Path(MODELS_MOUNT)
    linked_count = 0

    if models_volume_path.exists():
        for item in models_volume_path.iterdir():
            if item.name in ("Wan2GP", "venv"):
                continue
            target_link = wgp_models_dir / item.name
            if not target_link.exists():
                try:
                    target_link.symlink_to(item)
                    linked_count += 1
                except Exception as e:
                    log_line(f"[WARN] Failed to symlink {item.name}: {e}")

    config_data = {
        "attention_mode": "auto",
        "compile": "",
        "video_profile": profile_num,
        "image_profile": profile_num,
        "audio_profile": profile_num
    }
    config_file = wgp_dir / "wgp_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    log_line(f"[SETUP] Linked {linked_count} model assets from {MODELS_MOUNT}")
    log_line(f"[SETUP] Configured wgp_config.json: profile={profile_num}, attention=auto")

    # Commit any patch modifications if wgp_dir resides on the persistent volume
    if str(wgp_dir).startswith(MODELS_MOUNT):
        try:
            models_volume.commit()
        except Exception:
            pass
# ---------------------------------------------------------------------------
# Job Execution Engine
# ---------------------------------------------------------------------------
def execute_wan2gp_job(job_id: str, batch_uuid: Optional[str] = None, requested_gpu: str = "L40S"):
    import time
    job_dir = Path(OUTPUTS_MOUNT) / job_id

    for _ in range(5):
        try:
            outputs_volume.reload()
        except Exception:
            pass
        if (job_dir / "input.zip").exists():
            break
        time.sleep(1)

    job_dir.mkdir(parents=True, exist_ok=True)
    status_file = job_dir / "status.json"
    log_file = job_dir / "run.log"
    zip_path = job_dir / "input.zip"

    def update_status(status_str: str, error_msg: Optional[str] = None, files: Optional[list] = None):
        st = {
            "status": status_str,
            "job_id": job_id,
            "batch_uuid": batch_uuid,
            "requested_gpu": requested_gpu,
            "error": error_msg,
            "files": files or []
        }
        with open(status_file, "w") as f:
            json.dump(st, f, indent=2)
        outputs_volume.commit()

    update_status("running")
    profile_num, attention_mode, detected_device = resolve_gpu_tuning(requested_gpu)

    with open(log_file, "a") as log:
        log.write(f"=== Starting Wan2GP job {job_id} ===\n")
        log.write(f"Batch UUID: {batch_uuid}\n")
        log.write(f"Requested GPU: {requested_gpu} | Detected Hardware: {detected_device}\n")
        log.write(f"Tuning Strategy: Profile={profile_num}, Attention={attention_mode}\n")

    try:
        extract_dir = Path("/tmp") / job_id
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not zip_path.exists():
            raise FileNotFoundError(f"input.zip not found at {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        queue_file = extract_dir / "queue.json"
        if not queue_file.exists():
            raise FileNotFoundError("queue.json not found in input.zip")

        with open(queue_file, "r") as f:
            queue_data = json.load(f)

        # 1. Resolve Wan2GP root (persistent volume override takes priority)
        custom_volume_wgp = Path(MODELS_MOUNT) / "Wan2GP"
        if custom_volume_wgp.is_dir() and (custom_volume_wgp / "wgp.py").is_file():
            wan2gp_dir = custom_volume_wgp
            with open(log_file, "a") as log:
                log.write(f"[INFO] Using custom Wan2GP from volume: {custom_volume_wgp}\n")
        else:
            wan2gp_dir = Path("/root/Wan2GP")

        # 2. Resolve Python interpreter (Volume Venv with system site-packages)
        python_exec = ensure_volume_venv(log_file=log_file)
        with open(log_file, "a") as log:
            log.write(f"[INFO] Active Python environment: {python_exec}\n")

        # 3. Mount models & write runtime wgp_config.json
        setup_wan2gp_environment(wan2gp_dir, profile_num, attention_mode, log_file)

        # 4. Invoke Wan2GP CLI
        # Note: Do not force --attention here so wgp.py uses its native automatic selection
        # (or the user-configured setting in queue/config) without aborting when a specific kernel is missing.
        cmd = [
            str(python_exec),
            str(wan2gp_dir / "wgp.py"),
            "--profile", str(profile_num),
            "--process", str(zip_path),
            "--output-dir", str(job_dir)
        ]

        exec_env = os.environ.copy()
        exec_env["HF_XET_HIGH_PERFORMANCE"] = "0"
        exec_env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        exec_env["TORCH_ALLOW_TF32_CUBLAS"] = "1"
        exec_env["TORCH_ALLOW_TF32_CUDNN"] = "1"
        exec_env["CUDA_MODULE_LOADING"] = "LAZY"
        exec_env["PYTHONUNBUFFERED"] = "1"

        with open(log_file, "a") as log:
            log.write(f"Executing: {' '.join(cmd)}\n\n")

        proc = subprocess.Popen(
            cmd,
            cwd=str(wan2gp_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=exec_env,
            text=True,
            bufsize=1
        )

        stopped_by_user = False
        last_commit_time = time.time()
        last_reported_files = set()
        with open(log_file, "a") as log:
            while True:
                line = proc.stdout.readline()
                if line:
                    log.write(line)
                    log.flush()
                elif proc.poll() is not None:
                    break

                now = time.time()
                if now - last_commit_time >= 2.0:
                    try:
                        outputs_volume.reload()
                        if (job_dir / "STOP").exists():
                            stopped_by_user = True
                            log.write("\n[STOP] Cancellation requested by client. Terminating Wan2GP worker process...\n")
                            log.flush()
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            break

                        # Progressive intermediate file reporting for sequential batches
                        current_files = sorted([
                            f.name for f in job_dir.glob("*")
                            if f.is_file()
                            and f.name not in ("input.zip", "status.json", "run.log", "STOP", "results.json")
                            and not f.name.endswith(".tmp")
                            and "_tmp." not in f.name
                        ])
                        if set(current_files) != last_reported_files:
                            last_reported_files = set(current_files)
                            st = {
                                "status": "running",
                                "job_id": job_id,
                                "batch_uuid": batch_uuid,
                                "requested_gpu": requested_gpu,
                                "error": None,
                                "files": current_files
                            }
                            with open(status_file, "w") as f:
                                json.dump(st, f, indent=2)

                        outputs_volume.commit()
                    except Exception:
                        pass
                    last_commit_time = now

        try:
            outputs_volume.commit()
        except Exception:
            pass

        if stopped_by_user:
            update_status("stopped", error_msg="Job stopped by user")
            with open(log_file, "a") as log:
                log.write(f"\n[STOP] Job {job_id} terminated.\n")
            try:
                outputs_volume.commit()
            except Exception:
                pass
            return

        if proc.returncode != 0:
            raise RuntimeError(f"Wan2GP exited with error code {proc.returncode}")

        # 5. Collect outputs & write results.json
        output_files = [
            f.name for f in job_dir.glob("*")
            if f.is_file() and f.name not in ("input.zip", "status.json", "run.log")
        ]

        jobs_results = []
        for task in queue_data:
            t_id = str(task.get("id", 1))
            t_meta_id = task.get("meta_id", t_id)
            params = task.get("params", {})
            out_fn = params.get("output_filename", "")
            if not out_fn and output_files:
                mp4s = [f for f in output_files if f.endswith(".mp4")]
                out_fn = mp4s[0] if mp4s else output_files[0]

            jobs_results.append({
                "job_id": t_meta_id,
                "status": "completed",
                "output_filename": out_fn
            })

        results_json = {
            "batch_uuid": batch_uuid,
            "status": "completed",
            "gpu_hardware": detected_device,
            "profile_used": profile_num,
            "attention_used": attention_mode,
            "jobs": jobs_results
        }

        with open(job_dir / "results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        output_files.append("results.json")
        try:
            models_volume.commit()
            cache_volume.commit()
        except Exception:
            pass
        update_status("completed", files=output_files)

        with open(log_file, "a") as log:
            log.write(f"\n[OK] Job {job_id} successfully completed. Outputs: {output_files}\n")

    except Exception as e:
        err_msg = str(e)
        with open(log_file, "a") as log:
            log.write(f"\n[ERROR] Job failed: {err_msg}\n")
        update_status("failed", error_msg=err_msg)

# ---------------------------------------------------------------------------
# GPU Workers Fleet
# ---------------------------------------------------------------------------
@app.function(
    image=wan2gp_image,
    gpu="H100",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600,
    single_use_containers=True,
    scaledown_window=2,
    retries=0,
    **(dict(concurrency_limit=MAX_CONCURRENT_CLOUD_JOBS) if MAX_CONCURRENT_CLOUD_JOBS > 0 else {})
)
def run_worker_h100(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="H100")

@app.function(
    image=wan2gp_image,
    gpu="A100-80GB",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600,
    single_use_containers=True,
    scaledown_window=2,
    retries=0,
    **(dict(concurrency_limit=MAX_CONCURRENT_CLOUD_JOBS) if MAX_CONCURRENT_CLOUD_JOBS > 0 else {})
)
def run_worker_a100_80gb(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="A100-80GB")

@app.function(
    image=wan2gp_image,
    gpu="A100-40GB",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600,
    single_use_containers=True,
    scaledown_window=2,
    retries=0,
    **(dict(concurrency_limit=MAX_CONCURRENT_CLOUD_JOBS) if MAX_CONCURRENT_CLOUD_JOBS > 0 else {})
)
def run_worker_a100_40gb(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="A100-40GB")

@app.function(
    image=wan2gp_image,
    gpu="L40S",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600,
    single_use_containers=True,  # every .spawn() call gets its own dedicated container
    scaledown_window=120,
    retries=0,
    **(dict(concurrency_limit=MAX_CONCURRENT_CLOUD_JOBS) if MAX_CONCURRENT_CLOUD_JOBS > 0 else {})
)
def run_worker_l40s(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="L40S")

@app.function(
    image=wan2gp_image,
    gpu="L4",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600,
    single_use_containers=True,  # every .spawn() call gets its own dedicated container
    scaledown_window=120,
    retries=0,
    **(dict(concurrency_limit=MAX_CONCURRENT_CLOUD_JOBS) if MAX_CONCURRENT_CLOUD_JOBS > 0 else {})
)
def run_worker_l4(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="L4")

@app.function(
    image=wan2gp_image,
    gpu="T4",
    volumes={OUTPUTS_MOUNT: outputs_volume, MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=3600,
    single_use_containers=True,
    scaledown_window=2,
    retries=0,
    **(dict(concurrency_limit=MAX_CONCURRENT_CLOUD_JOBS) if MAX_CONCURRENT_CLOUD_JOBS > 0 else {})
)
def run_worker_t4(job_id: str, batch_uuid: Optional[str] = None):
    execute_wan2gp_job(job_id, batch_uuid, requested_gpu="T4")

# ---------------------------------------------------------------------------
# Diagnostics & Admin Update Hook (CPU-Only, $0 GPU cost)
# ---------------------------------------------------------------------------
@app.function(
    image=wan2gp_image,
    gpu="L40S",
    volumes={MODELS_MOUNT: models_volume, CACHE_MOUNT: cache_volume},
    timeout=600
)
def inspect_gpu_environment() -> str:
    import torch
    report = {
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "device_name": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "None",
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "cuda_version": str(getattr(torch.version, "cuda", "unknown")),
        "accelerators": {}
    }
    for mod in ("sageattention", "triton", "flash_attn"):
        try:
            m = __import__(mod)
            report["accelerators"][mod] = str(getattr(m, "__version__", "installed"))
        except Exception as e:
            report["accelerators"][mod] = f"not available: {e}"
    return json.dumps(report)

@app.function(
    image=wan2gp_image,
    volumes={MODELS_MOUNT: models_volume},
    timeout=120
)
def patch_volume_repo() -> Dict[str, Any]:
    """CPU-only worker to apply source patches directly to /models/Wan2GP on volume."""
    target_dir = Path(MODELS_MOUNT) / "Wan2GP"
    msgs = []
    if target_dir.exists():
        apply_source_patches(target_dir, log_fn=lambda m: msgs.append(m))
        try:
            models_volume.commit()
        except Exception:
            pass
        return {"status": "ok", "patched_dir": str(target_dir), "messages": msgs}
    return {"status": "skipped", "message": f"{target_dir} does not exist"}


@app.function(
    image=wan2gp_image,
    volumes={MODELS_MOUNT: models_volume},
    timeout=900
)
def update_wan2gp_volume_repo() -> Dict[str, str]:
    """
    CPU-Only update worker ($0 GPU cost):
    1. Git pull the configured Wan2GP fork into /models/Wan2GP
    2. Synchronizes /models/venv via pip install (all packages)
    3. Re-applies source patches to Wan2GP
    """
    target_dir = Path(MODELS_MOUNT) / "Wan2GP"
    if not target_dir.exists():
        res = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", WAN2GP_GIT_REF, WAN2GP_REPO_URL, str(target_dir)],
            capture_output=True, text=True
        )
        git_msg = f"Cloned fresh Wan2GP fork: {res.stdout or res.stderr}"
    else:
        # Reset altered files so fetch/reset succeeds cleanly without conflict
        subprocess.run(["git", "checkout", "--", "."], cwd=str(target_dir), capture_output=True)
        subprocess.run(["git", "remote", "set-url", "origin", WAN2GP_REPO_URL], cwd=str(target_dir), capture_output=True)
        subprocess.run(["git", "fetch", "origin", WAN2GP_GIT_REF], cwd=str(target_dir), capture_output=True)
        res = subprocess.run(
            ["git", "reset", "--hard", f"origin/{WAN2GP_GIT_REF}"],
            cwd=str(target_dir),
            capture_output=True, text=True
        )
        git_msg = f"Pulled latest Wan2GP fork updates: {res.stdout or res.stderr}"

    # Sync volume venv with ALL requirements (installing full requirements.txt)
    venv_py = ensure_volume_venv(allow_install=True)

    # Re-apply source patches directly to volume repo
    apply_source_patches(target_dir, log_fn=print)

    # Obtain exact commit hash for lockstep verification
    commit_hash = "unknown"
    try:
        c_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(target_dir), capture_output=True, text=True)
        if c_res.returncode == 0:
            commit_hash = c_res.stdout.strip()
    except Exception:
        pass

    models_volume.commit()
    return {
        "status": "ok",
        "git_message": git_msg,
        "venv_python": str(venv_py),
        "commit": commit_hash
    }

# ---------------------------------------------------------------------------
# FastAPI Web Gateway (Lightweight CPU ASGI Server)
# ---------------------------------------------------------------------------
web_app = FastAPI(title="Wan2GP High-Performance Modal Service")

def verify_auth(x_api_key: Optional[str] = None, authorization: Optional[str] = None):
    token = x_api_key
    if not token and authorization:
        if authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()
    if EXPECTED_API_KEY and token != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key")

@web_app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "wan2gp-cloud-api",
        "architecture": "volume-venv",
        "volumes": [OUTPUTS_VOL_NAME, MODELS_VOL_NAME, CACHE_VOL_NAME],
        "gpus": ["L40S", "A100-40GB", "A100-80GB", "H100", "L4", "T4"],
        "default_gpu": "L40S",
        "min_recommended_gpu": "L4 (24GB VRAM)"
    }

@web_app.get("/gpu-info")
def gpu_info_endpoint(x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    res_str = inspect_gpu_environment.remote()
    return json.loads(res_str)

@web_app.post("/admin/update-wan2gp")
def update_wan2gp_endpoint(x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    return update_wan2gp_volume_repo.remote()

@web_app.post("/admin/patch-volume")
def patch_volume_endpoint(x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    return patch_volume_repo.remote()

@web_app.post("/generate")
@web_app.post("/v1/jobs")
async def generate_endpoint(
    request: Request,
    gpu_type: str = Query("L40S"),
    batch_uuid: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    verify_auth(x_api_key, authorization)

    data = await request.body()
    if not data or len(data) < 2:
        raise HTTPException(status_code=400, detail="Empty or corrupted payload")

    job_id = str(uuid.uuid4())
    job_dir = Path(OUTPUTS_MOUNT) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    zip_path = job_dir / "input.zip"

    # Support Mode A (binary ZIP bundle) and Mode B (JSON GenerationRequest)
    if data.startswith(b"PK"):
        with open(zip_path, "wb") as f:
            f.write(data)
    else:
        try:
            json_obj = json.loads(data.decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid payload: neither ZIP nor JSON ({e})")

        if isinstance(json_obj, list):
            tasks = json_obj
        elif isinstance(json_obj, dict):
            if "queue" in json_obj and isinstance(json_obj["queue"], list):
                tasks = json_obj["queue"]
            elif "tasks" in json_obj and isinstance(json_obj["tasks"], list):
                tasks = json_obj["tasks"]
            else:
                task_params = dict(json_obj.get("params") or json_obj.get("settings") or {})
                if not task_params:
                    task_params = dict(json_obj)
                
                # Extract infrastructure parameter
                if "gpu_type" in json_obj and json_obj["gpu_type"]:
                    gpu_type = str(json_obj["gpu_type"])
                elif "gpu_type" in task_params:
                    gpu_type = str(task_params.pop("gpu_type"))

                m_type = json_obj.get("model_type") or json_obj.get("model") or task_params.get("model_type", "t2v")
                task_params["model_type"] = m_type

                for k in ("prompt", "negative_prompt", "resolution", "video_length", "num_inference_steps", "guidance_scale", "seed", "flow_shift"):
                    if k in json_obj and k not in task_params:
                        task_params[k] = json_obj[k]

                if "inputs" in json_obj and isinstance(json_obj["inputs"], dict):
                    for in_k, in_v in json_obj["inputs"].items():
                        if in_k not in task_params and in_v is not None:
                            task_params[in_k] = in_v

                tasks = [{
                    "id": 1,
                    "meta_id": str(json_obj.get("task_id", uuid.uuid4())),
                    "params": task_params
                }]

        else:
            raise HTTPException(status_code=400, detail="Invalid JSON task structure")

        # Clean all tasks to satisfy strict manifest validation
        for t in tasks:
            if isinstance(t, dict):
                p = t.get("params")
                if isinstance(p, dict):
                    for disallowed in ("gpu_type", "width", "height", "task_id", "execution_mode", "engine", "version", "provider", "inputs", "plugin_data", "metadata"):
                        p.pop(disallowed, None)

        import io
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("queue.json", json.dumps(tasks, indent=2))
        with open(zip_path, "wb") as f:
            f.write(zip_buf.getvalue())

    initial_status = {
        "status": "queued",
        "job_id": job_id,
        "batch_uuid": batch_uuid,
        "gpu_type": gpu_type,
        "files": []
    }
    with open(job_dir / "status.json", "w") as f:
        json.dump(initial_status, f, indent=2)

    await outputs_volume.commit.aio()

    # Route to appropriate GPU worker
    norm_gpu = gpu_type.upper().replace("-", "_").replace(" ", "_")
    if "H100" in norm_gpu:
        await run_worker_h100.spawn.aio(job_id, batch_uuid)
    elif "80GB" in norm_gpu or ("A100" in norm_gpu and "80" in norm_gpu):
        await run_worker_a100_80gb.spawn.aio(job_id, batch_uuid)
    elif "A100" in norm_gpu:
        await run_worker_a100_40gb.spawn.aio(job_id, batch_uuid)
    elif norm_gpu == "L4" or norm_gpu.startswith("L4_"):
        await run_worker_l4.spawn.aio(job_id, batch_uuid)
    elif "T4" in norm_gpu:
        await run_worker_t4.spawn.aio(job_id, batch_uuid)
    else:
        await run_worker_l40s.spawn.aio(job_id, batch_uuid)

    return {"job_id": job_id, "status": "queued", "gpu_selected": gpu_type}

# ---------------------------------------------------------------------------
# Batch Job Submission (Fire N independent jobs in parallel)
# ---------------------------------------------------------------------------
@web_app.post("/v1/jobs/batch")
async def batch_generate_endpoint(
    request: Request,
    gpu_type: str = Query("L40S"),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    """Submit multiple independent jobs that run in parallel on separate GPU containers.

    Expects JSON body:
    {
        "jobs": [
            {"prompt": "dragon walking", "model_type": "t2v", ...},
            {"prompt": "cat dancing", "model_type": "t2v", ...}
        ],
        "gpu_type": "L40S",          // optional, per-batch default
        "max_parallel": 3            // optional, cap concurrent containers
    }

    Each job in "jobs" gets its own container and job_id.
    Returns list of job_ids for polling.
    """
    verify_auth(x_api_key, authorization)

    try:
        data = await request.body()
        batch_request = json.loads(data.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

    jobs_list = batch_request.get("jobs") or batch_request.get("tasks", [])
    if not isinstance(jobs_list, list) or len(jobs_list) == 0:
        raise HTTPException(status_code=400, detail="'jobs' must be a non-empty list of generation tasks")

    batch_gpu = batch_request.get("gpu_type") or gpu_type
    max_parallel = int(batch_request.get("max_parallel", 0))
    if MAX_CONCURRENT_CLOUD_JOBS > 0:
        max_parallel = min(max_parallel, MAX_CONCURRENT_CLOUD_JOBS) if max_parallel > 0 else MAX_CONCURRENT_CLOUD_JOBS

    batch_uuid = str(uuid.uuid4())
    spawned_jobs = []

    for idx, job_spec in enumerate(jobs_list):
        if max_parallel > 0 and idx >= max_parallel:
            # Remaining jobs will be queued on the Modal side (container auto-scaling)
            pass

        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_MOUNT) / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Determine per-job GPU override
        job_gpu = str(job_spec.pop("gpu_type", "")).strip() or batch_gpu

        # Build single-task queue.json
        task_params = dict(job_spec)
        for disallowed in ("gpu_type", "width", "height", "task_id", "execution_mode", "engine", "version", "provider", "inputs", "plugin_data", "metadata"):
            task_params.pop(disallowed, None)
        task = {"id": idx + 1, "meta_id": str(uuid.uuid4()), "params": task_params}

        import io
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("queue.json", json.dumps([task], indent=2))
        with open(job_dir / "input.zip", "wb") as f:
            f.write(zip_buf.getvalue())

        initial_status = {
            "status": "queued",
            "job_id": job_id,
            "batch_uuid": batch_uuid,
            "batch_index": idx,
            "gpu_type": job_gpu,
            "files": []
        }
        with open(job_dir / "status.json", "w") as f:
            json.dump(initial_status, f, indent=2)

        spawned_jobs.append({"job_id": job_id, "batch_index": idx, "gpu_type": job_gpu, "status": "queued"})

    await outputs_volume.commit.aio()

    # Spawn all containers (Modal handles queuing if concurrency_limit is hit)
    for job_info in spawned_jobs:
        jid = job_info["job_id"]
        norm_gpu = job_info["gpu_type"].upper().replace("-", "_").replace(" ", "_")
        if "H100" in norm_gpu:
            await run_worker_h100.spawn.aio(jid, batch_uuid)
        elif "80GB" in norm_gpu or ("A100" in norm_gpu and "80" in norm_gpu):
            await run_worker_a100_80gb.spawn.aio(jid, batch_uuid)
        elif "A100" in norm_gpu:
            await run_worker_a100_40gb.spawn.aio(jid, batch_uuid)
        elif norm_gpu == "L4" or norm_gpu.startswith("L4_"):
            await run_worker_l4.spawn.aio(jid, batch_uuid)
        elif "T4" in norm_gpu:
            await run_worker_t4.spawn.aio(jid, batch_uuid)
        else:
            await run_worker_l40s.spawn.aio(jid, batch_uuid)

    return {
        "batch_uuid": batch_uuid,
        "total_jobs": len(spawned_jobs),
        "max_concurrent": max_parallel or "unlimited",
        "jobs": spawned_jobs
    }

# ---------------------------------------------------------------------------
# Active Jobs Listing
# ---------------------------------------------------------------------------
@web_app.get("/v1/jobs/active")
def active_jobs_endpoint(
    batch_uuid: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None)
):
    """List all active (queued/running) jobs, optionally filtered by batch_uuid."""
    verify_auth(x_api_key, authorization)
    outputs_volume.reload()

    outputs_root = Path(OUTPUTS_MOUNT)
    active_jobs = []
    for job_dir in outputs_root.iterdir():
        if not job_dir.is_dir():
            continue
        status_file = job_dir / "status.json"
        if not status_file.exists():
            continue
        try:
            with open(status_file, "r") as f:
                st = json.load(f)
            if st.get("status") in ("queued", "running"):
                if batch_uuid and st.get("batch_uuid") != batch_uuid:
                    continue
                active_jobs.append({
                    "job_id": st.get("job_id", job_dir.name),
                    "batch_uuid": st.get("batch_uuid"),
                    "status": st.get("status"),
                    "gpu_type": st.get("gpu_type") or st.get("requested_gpu"),
                })
        except Exception:
            continue

    return {
        "active_count": len(active_jobs),
        "max_concurrent": MAX_CONCURRENT_CLOUD_JOBS or "unlimited",
        "jobs": active_jobs
    }

@web_app.get("/status/{job_id}")
@web_app.get("/v1/jobs/{job_id}")
def status_endpoint(job_id: str, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    verify_auth(x_api_key, authorization)
    outputs_volume.reload()

    status_file = Path(OUTPUTS_MOUNT) / job_id / "status.json"
    if not status_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    with open(status_file, "r") as f:
        return json.load(f)

@web_app.get("/download/{job_id}")
def download_list_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    outputs_volume.reload()

    job_dir = Path(OUTPUTS_MOUNT) / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    files = [f.name for f in job_dir.glob("*") if f.is_file() and f.name != "input.zip"]
    return {"job_id": job_id, "files": files}

@web_app.get("/download/{job_id}/{filename}")
def download_file_endpoint(job_id: str, filename: str, x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    outputs_volume.reload()

    target_file = Path(OUTPUTS_MOUNT) / job_id / filename
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"File {filename} not found for job {job_id}")

    media_type = "application/json" if filename.endswith(".json") else (
        "video/mp4" if filename.endswith(".mp4") else "application/octet-stream"
    )
    file_size = target_file.stat().st_size

    def _iter_chunks(path: Path, chunk_size: int = 2 * 1024 * 1024):
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _iter_chunks(target_file),
        media_type=media_type,
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )

@web_app.get("/logs/{job_id}")
def logs_endpoint(job_id: str, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    verify_auth(x_api_key, authorization)
    outputs_volume.reload()

    log_file = Path(OUTPUTS_MOUNT) / job_id / "run.log"
    if not log_file.exists():
        return PlainTextResponse("No logs found for this job.")

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        return PlainTextResponse(f.read())

@web_app.post("/stop/{job_id}")
def stop_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    verify_auth(x_api_key)
    outputs_volume.reload()

    job_dir = Path(OUTPUTS_MOUNT) / job_id
    if not job_dir.exists():
        job_dir.mkdir(parents=True, exist_ok=True)

    status_file = job_dir / "status.json"
    st = {"status": "stopped", "job_id": job_id, "error": "Stopped by client"}
    with open(status_file, "w") as f:
        json.dump(st, f, indent=2)

    stop_file = job_dir / "STOP"
    stop_file.touch()

    try:
        outputs_volume.commit()
    except Exception:
        pass

    return {"job_id": job_id, "status": "stopped"}

# ---------------------------------------------------------------------------
# Modal Entrypoint
# ---------------------------------------------------------------------------
web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]>=0.115.0",
    "uvicorn>=0.30.0",
    "pydantic>=2.0.0"
)

@app.function(
    image=web_image,
    volumes={OUTPUTS_MOUNT: outputs_volume},
    scaledown_window=10
)
@modal.asgi_app()
def fastapi_app():
    return web_app

if __name__ == "__main__":
    app.run()

