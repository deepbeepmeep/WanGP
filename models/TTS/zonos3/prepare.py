# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Repack the released ZONOS3-Mini weights (Zyphra/ZONOS3-Mini, Apache-2.0) for WanGP.

python models/TTS/zonos3/prepare.py --source HF_RELEASE_FOLDER --code GITHUB_RELEASE_FOLDER --output CHECKPOINT_ROOT

- ZONOS3_Mini_bf16.safetensors (root): the LM as released (BF16, the routers' balancing biases FP32) with the phoneme / start /
  pause text rows appended to its text table (stored BF16 like the table), the attention's k|v projection and the first layer's
  feed-forward input projection stored as 2D matrices, the latter's rows reordered gate | value.
- zonos3/ZAC1_keep2_decoder.safetensors: the codec's quantizers and semantic head (weight norms folded) from generator.pt and the
  pruned keep2_plain decoder stages and iSTFT head. The transformer stages' matrices, which the released decoder runs under BF16
  autocast, are stored BF16; the parts it computes in FP32 (quantizer lookups, semantic head, norms, layer scales, iSTFT head, FIR
  mixer, resampler kernel) stay FP32.
- zonos3/MECT_ownmect_s2.safetensors: the speaker encoder (VPR / MECT-B2 / ASTP), without its training classifier.
- zonos3/voices.safetensors: the default voices' reference codes (int32) and speaker embeddings (FP32).
- zonos3/superbpe8192.json: the text tokenizer.
- zonos3/steering.safetensors: the direction vectors of the release steering bundle (controls metadata in steering.json).
- reading_heads.json (next to this script, --only heads): the alignment heads of each language (GitHub release, assets/laci_heads),
  whose attention follows the text being read: inline cues start when the reading reaches their words.
Every output tensor is compared with its source before the file is written.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file, load_file

TEXT_ROWS_FORMAT = "zonos-text-extra/1"


def _load(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=True)


def _folded(state, prefix):
    """The weight of a weight-normed convolution (g * v / |v|, the norm over every dimension but the first)."""
    g, v = state[prefix + "weight_g"].double(), state[prefix + "weight_v"].double()
    return (g * v / v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))).float()


def _save(tensors, path, metadata):
    tensors = {key: value.contiguous() for key, value in tensors.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    check = load_file(str(path))
    for key, value in tensors.items():
        if not torch.equal(check[key], value):
            raise RuntimeError(f"{path.name}: {key} differs after writing")
    print(f"{path}: {len(tensors)} tensors, {sum(t.numel() * t.element_size() for t in tensors.values()) / 2**20:.1f} MiB")


def prepare_lm(source, output):
    state = _load(source / "lm/zonos3_mini/model.pth")
    extra = _load(source / "lm/zonos3_mini/text_rows/pause_tokens_s5000.pt")
    if extra.get("format") != TEXT_ROWS_FORMAT or int(extra["text_vocab"]) != 8196 or tuple(extra["rows"].shape) != (143, 1024):
        raise ValueError("unexpected text rows export")
    out = {}
    for key, value in state.items():
        if key.endswith("attention.wkv.weight"):
            value = value.reshape(-1, value.shape[-1])  # k rows then v rows
        elif key == "layers.0.feed_forward.w_in.weight":
            value = torch.cat((value[1], value[0]))  # released value | gate -> gate | value (shared SwiGLU layout)
        elif key == "multi_embedder.embedders.10.weight":
            if value.shape[0] != 8197:
                raise ValueError("unexpected text table")
            value = torch.cat((value, extra["rows"].to(value.dtype)))
        out[key] = value
    fp32 = sorted(key for key, value in out.items() if value.dtype == torch.float32)
    assert all(key.endswith("balancing_biases") for key in fp32), fp32
    _save(out, output / "ZONOS3_Mini_bf16.safetensors", {"source": "Zyphra/ZONOS3-Mini lm/zonos3_mini + text_rows/pause_tokens_s5000"})


def prepare_decoder(source, output):
    generator = _load(source / "autoencoder/generator.pt")["generator"]
    trainable = _load(source / "autoencoder/heads/keep2_plain/step_0050000.pt")["trainable"]
    out = {}
    for name, count in (("semantic_quantizer", 1), ("quantizer", 9)):
        for index in range(count):
            prefix = f"quantizer.{name}.quantizers.{index}."
            target = f"semantic_vq.{index}." if name == "semantic_quantizer" else f"acoustic_vq.{index}."
            out[target + "codebook.weight"] = generator[prefix + "codebook.weight"]
            out[target + "out_proj.weight"] = _folded(generator, prefix + "out_proj.")[..., 0]
            out[target + "out_proj.bias"] = generator[prefix + "out_proj.bias"]
    for key, value in generator.items():
        if key.startswith("semantic_head.") and not key.startswith("semantic_head.4."):
            out[key] = value
    out["semantic_head.4.weight"] = _folded(generator, "semantic_head.4.")
    out["semantic_head.4.bias"] = generator["semantic_head.4.bias"]
    out["to_external.kernel"] = generator["to_external.kernel"]
    # The released decoder runs its transformer stages under BF16 autocast (their matrices in BF16) and its iSTFT head without autocast.
    bf16 = (".input_proj.weight", ".output_proj.weight", ".qkv.weight", ".attention_out.weight", ".ffn.0.weight", ".ffn.3.weight")
    for key, value in trainable.items():
        out[key] = value.to(torch.bfloat16) if key.startswith("stages.") and key.endswith(bf16) else value
    _save(out, output / "zonos3" / "ZAC1_keep2_decoder.safetensors", {"source": "Zyphra/ZONOS3-Mini autoencoder/generator.pt + autoencoder/heads/keep2_plain"})


def prepare_speaker(source, output):
    state = _load(source / "speaker_encoders/ownmect_s2/ckpt_0037.pth")["modules"]["spk_model"]
    out = {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")}
    if len(out) != len(state):
        raise ValueError("unexpected speaker encoder keys")
    _save(out, output / "zonos3" / "MECT_ownmect_s2.safetensors", {"source": "Zyphra/ZONOS3-Mini speaker_encoders/ownmect_s2/ckpt_0037.pth"})


def prepare_voices(source, output):
    out = {}
    for codes in sorted((source / "gguf/voices").glob("*/*.codes.npy")):
        stem = codes.name[:-len(".codes.npy")]
        out[stem + ".codes"] = torch.from_numpy(np.load(codes, allow_pickle=False).astype(np.int32))
        out[stem + ".embedding"] = torch.from_numpy(np.load(codes.with_name(stem + ".embedding.npy"), allow_pickle=False).astype(np.float32)).reshape(-1)
        assert out[stem + ".codes"].ndim == 2 and out[stem + ".codes"].shape[1] == 10 and out[stem + ".embedding"].numel() == 192
    _save(out, output / "zonos3" / "voices.safetensors", {"source": "Zyphra/ZONOS3-Mini gguf/voices"})


def prepare_steering(code, output):
    """The direction vectors of the studio steering bundle (GitHub release, assets/steering): one float32 vector per
    "<set>.<direction>" (192-d speaker-embedding directions, 1024-d hidden-state directions); steering.json holds the rest."""
    import json
    root = code / "assets/steering"
    bundle = json.loads((root / "studio_bundle_mect60k_x10.json").read_text(encoding="utf-8"))
    out = {}
    for name, folder in bundle["sets"].items():
        manifest = json.loads((root / folder / "manifest.json").read_text(encoding="utf-8"))
        for direction, entry in manifest["directions"].items():
            vector = np.asarray(np.load(root / folder / entry["file"], allow_pickle=False), dtype=np.float32).reshape(-1)
            assert vector.shape[0] == int(manifest["dim"])
            out[f"{name}.{direction}"] = torch.from_numpy(vector.copy())
    _save(out, output / "zonos3" / "steering.safetensors", {"source": "Zyphra/ZONOS3-Mini assets/steering/studio_bundle_mect60k_x10.json"})


def prepare_heads(code):
    """[layer, head] pairs of the alignment heads per WanGP language code (zonos3_mini_heads.json for English, else
    zonos3_mini_heads_<speech code>.json)."""
    import json
    here = Path(__file__).resolve().parent
    speech = {"en_us": "", "en_gb": "", "fr_fr": "_fr", "pt_br": "_pt", "cmn": "_zh"}
    out = {}
    for language in json.loads((here / "zonos3_mini.json").read_text(encoding="utf-8"))["languages"]:
        suffix = speech.get(language["code"], "_" + language["code"])
        heads = json.loads((code / "assets/laci_heads" / f"zonos3_mini_heads{suffix}.json").read_text(encoding="utf-8"))
        assert heads["model"] == "zonos3_mini" and not heads["fallback"]
        out[language["code"]] = [[head["layer"], head["head"]] for head in heads["heads"]]
    (here / "reading_heads.json").write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="the Hugging Face release (weights)")
    parser.add_argument("--code", type=Path, help="the GitHub release (steering assets, alignment heads)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--only", nargs="*", default=["lm", "decoder", "speaker", "voices", "tokenizer", "steering"])
    args = parser.parse_args()
    steps = {"lm": prepare_lm, "decoder": prepare_decoder, "speaker": prepare_speaker, "voices": prepare_voices,
             "tokenizer": lambda source, output: shutil.copyfile(source / "lm/zonos3_mini/tokenizer/superbpe8192.json", output / "zonos3" / "superbpe8192.json"),
             "steering": lambda source, output: prepare_steering(args.code, output), "heads": lambda source, output: prepare_heads(args.code)}
    for name in args.only:
        steps[name](args.source, args.output)


if __name__ == "__main__":
    main()
