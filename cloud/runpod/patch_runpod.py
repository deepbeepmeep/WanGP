#!/usr/bin/env python3
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Patch script for official Wan2GP in RunPod Serverless container.
Applies all compatibility shims to the original deepbeepmeep/Wan2GP installation:
1. Triton compatibility (nanovllm)
2. higgs_audio_v2_tokenizer (exist_ok=True)
3. download_progress.py (accepts **kwargs & huggingface_hub _request_wrapper compat)
4. wgp.py (Torch accelerator & RoPE compatibility shim)
5. Qwen3VL (pad_token_id attribute safety & config default)
6. sitecustomize.py (Universal PyTorch/Transformers compatibility)
7. Gemma4UnifiedTextConfig (Transformers heterogeneity compatibility)
"""

import os
import re
import sys
import subprocess
from pathlib import Path

def apply_patches(wgp_dir: Path):
    print(f"[PATCH] Applying compatibility patches to {wgp_dir}...")
    if not wgp_dir or not wgp_dir.exists():
        print(f"[PATCH-ERROR] Directory {wgp_dir} does not exist!")
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
                    print(f"[PATCH] Patched {py_file.name} for Triton compat")
            except Exception as e:
                print(f"[PATCH-WARN] Failed to patch {py_file}: {e}")

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
                print("[PATCH] Patched higgs_audio_v2_tokenizer for exist_ok=True")
        except Exception as e:
            print(f"[PATCH-WARN] Failed to patch tts tokenizer: {e}")

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
                print("[PATCH] Patched download_progress.py for huggingface_hub compat")
        except Exception as e:
            print(f"[PATCH-WARN] Failed to patch download_progress.py: {e}")

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
                print("[PATCH] Injected torch.accelerator & RoPE compatibility shim into wgp.py")
        except Exception as e:
            print(f"[PATCH-WARN] Failed to patch wgp.py: {e}")

    # 5. Qwen3VL pad_token_id compatibility
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
                    print(f"[PATCH] Patched {q_rel} for pad_token_id attribute safety")
            except Exception as e:
                print(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")

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
                    print(f"[PATCH] Patched {q_rel} for pad_token_id config default")
            except Exception as e:
                print(f"[PATCH-WARN] Failed to patch {q_rel}: {e}")

    # 6. sitecustomize.py shim
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
        "# 2. Universal RoPE compatibility shim for transformers\n"
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
        "    pass\n"
    )
    for sp in sys.path:
        if "site-packages" in sp:
            try:
                sc = Path(sp) / "sitecustomize.py"
                sc.write_text(sitecustomize_code, encoding="utf-8")
                print(f"[PATCH] Injected sitecustomize.py into {sp}")
                break
            except Exception:
                pass

    # 7. LTX-2 Gemma4UnifiedTextConfig heterogeneity compatibility
    gemma4_path = wgp_dir / "models" / "ltx2" / "ltx_core" / "text_encoders" / "gemma" / "gemma4_unified.py"
    if gemma4_path.exists():
        try:
            content = gemma4_path.read_text(encoding="utf-8")
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
                    "self.head_dim = config.head_dim if self.is_sliding else config.global_head_dim",
                    "self.head_dim = (config.__dict__.get('head_dim') or getattr(config, 'head_dim', 256)) if self.is_sliding else getattr(config, 'global_head_dim', 512)"
                )
                content = content.replace(
                    'head_dim = config.global_head_dim if layer_type == "full_attention" else config.head_dim',
                    'head_dim = (config.__dict__.get("global_head_dim") or getattr(config, "global_head_dim", 512)) if layer_type == "full_attention" else (config.__dict__.get("head_dim") or getattr(config, "head_dim", 256))'
                )
            gemma4_path.write_text(content, encoding="utf-8")
            print("[PATCH] Patched gemma4_unified.py for complete heterogeneity compat")
        except Exception as e:
            print(f"[PATCH-WARN] Failed to patch gemma4_unified.py: {e}")

if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/Wan2GP")
    apply_patches(target)
