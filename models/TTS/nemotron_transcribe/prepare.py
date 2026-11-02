# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prepare the WanGP files of NVIDIA Nemotron 3.5 Transcribe 30B-A4B from its released NeMo checkpoint (OpenMDW-1.1).

python -m models.TTS.nemotron_transcribe.prepare --source DIR --output DIR [--parts lm,experts_int8,experts_int4,perception,assets,draft]

- lm: the dense language model (every LM tensor but the routed experts), payloads copied unchanged (BF16; the SSM parameters
  A_log / D / dt_bias and the routing correction bias stay float32, as released); "_fp32_params." dropped from their names.
- experts_int8 / experts_int4: the routed experts of the 23 MoE layers and of the MTP layer, as MMGP expert banks of squared-ReLU
  experts (w13 = up rows, w2 = down rows, [experts, out, in]): a ConvRot rotation of each group of 64 input columns, then INT8 with a
  float32 scale per row, or symmetric INT4 (two values per byte, offset 8) with an FP16 scale per group of 32 columns.
- perception: the audio encoders, projection and word-timestamp head (perception.* and ctc_timestamp.*), payloads unchanged.
- assets: tokenizer, chat template, configs and license notices; draft: the DFlash2 drafter, unchanged.
Conversions are bounded in memory (one bank at a time, streamed to disk); copied payloads are verified by digest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

import torch

LM_FILE = "NemotronTranscribe_30B_A4B_lm_bf16.safetensors"
EXPERT_FILES = {"experts_int8": "NemotronTranscribe_30B_A4B_experts_int8_convrot.safetensors", "experts_int4": "NemotronTranscribe_30B_A4B_experts_int4.safetensors"}
ASSET_FOLDER = "nemotron_transcribe"
PERCEPTION_FILE = "NemotronTranscribe_perception_bf16.safetensors"
DRAFT_FILE = "NemotronTranscribe_dflash2_bf16.safetensors"
ASSETS = {"tokenizer.json": "tokenizer.json", "tokenizer_config.json": "tokenizer_config.json", "chat_template.jinja": "chat_template.jinja",
          "config.json": "config.json", "processor_config.json": "processor_config.json", "generation_config.json": "generation_config.json",
          "LICENSE": "LICENSE_NEMOTRON_TRANSCRIBE.txt", "THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md"}
ROTATION, INT4_GROUP = 64, 32
DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16, "I8": torch.int8, "U8": torch.uint8, "I64": torch.int64}
NAMES = {value: key for key, value in DTYPES.items()}


def read_header(path):
    with open(path, "rb") as reader:
        length = struct.unpack("<Q", reader.read(8))[0]
        header = json.loads(reader.read(length))
    header.pop("__metadata__", None)
    return header, 8 + length


class Source:
    def __init__(self, path):
        self.path = Path(path)
        self.header, self.base = read_header(self.path)
        self.file = open(self.path, "rb")

    def raw(self, key):
        begin, end = self.header[key]["data_offsets"]
        self.file.seek(self.base + begin)
        return self.file.read(end - begin)

    def tensor(self, key):
        entry = self.header[key]
        return torch.frombuffer(bytearray(self.raw(key)), dtype=DTYPES[entry["dtype"]]).view(entry["shape"])


def write(path, entries, metadata=None):
    """entries: name -> (dtype name, shape, producer of the payload bytes), written in order with a header computed first."""
    header, offset = {}, 0
    for name, (dtype, shape, _) in entries.items():
        size = DTYPES[dtype].itemsize
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    temporary = Path(str(path) + ".partial")
    with open(temporary, "wb") as writer:
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        for name, (dtype, shape, producer) in entries.items():
            payload = producer()
            begin, end = header[name]["data_offsets"]
            if len(payload) != end - begin:
                raise ValueError(f"{name}: {len(payload)} bytes instead of {end - begin}")
            writer.write(payload)
    temporary.replace(path)
    print(f"wrote {path} ({offset / 2**30:.2f} GiB, {len(entries)} tensors)", flush=True)


def tensor_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def copy_entries(source, mapping):
    entries = {}
    for key, name in mapping.items():
        spec = source.header[key]
        entries[name] = (spec["dtype"], spec["shape"], lambda key=key: source.raw(key))
    return entries


def verify_copies(path, source, mapping):
    target = Source(path)
    for key, name in mapping.items():
        if hashlib.sha256(source.raw(key)).digest() != hashlib.sha256(target.raw(name)).digest():
            raise ValueError(f"Copied payload differs: {name}")
    print(f"verified {len(mapping)} copied payloads in {path}", flush=True)


def hadamard(n):
    from shared.qtypes.int8_convrot import _regular_hadamard
    return _regular_hadamard(n, torch.device("cpu"), torch.float32)


def rotated(weight, h):
    # ConvRot: W' = W @ H by groups of input columns (H symmetric orthogonal); activations are rotated by H at run time
    rows, cols = weight.shape[-2:]
    return torch.matmul(weight.float().reshape(*weight.shape[:-1], cols // h.shape[0], h.shape[0]), h).reshape(weight.shape)


def quantize_int8(weight):
    scale = weight.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127
    return torch.round(weight / scale).clamp(-127, 127).to(torch.int8), scale


def quantize_int4(weight, group):
    # symmetric groups as GGUF Q4_0: the scale maps the largest magnitude to -8 (its sign kept), values offset by 8
    grouped = weight.reshape(*weight.shape[:-1], -1, group)
    extreme = grouped.gather(-1, grouped.abs().argmax(-1, keepdim=True))
    scale = (extreme / -8).to(torch.float16)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale).float()
    values = (torch.round(grouped / safe) + 8).clamp(0, 15).to(torch.uint8).reshape(weight.shape)
    return values[..., 0::2] | (values[..., 1::2] << 4), scale[..., 0]


def dequantize_check(name, weight, rotated_weight, quantized, h, kind):
    if kind == "int8":
        values, scale = quantized
        restored = values.float() * scale
    else:
        packed, scale = quantized
        values = torch.stack((packed & 15, packed >> 4), -1).reshape(*packed.shape[:-1], -1).float() - 8
        restored = (values.reshape(*values.shape[:-1], -1, INT4_GROUP) * scale.float()[..., None]).reshape(values.shape)
    restored = rotated(restored, h)  # H @ H = I: back to the unrotated weights
    error = ((restored - weight.float()).norm() / weight.float().norm()).item()
    print(f"  {name}: relative error {error:.4f}", flush=True)
    return error


def expert_banks(source):
    banks = []
    for key in source.header:
        if key.endswith(".experts.gate_and_up_projs"):
            prefix = key[len("llm."):-len("gate_and_up_projs")]  # model.layers.N.mixer.experts. / mtp.layers.N.mixer.experts.
            banks.append((prefix, key, key.replace("gate_and_up_projs", "down_projs")))
    return banks


def prepare_experts(source, output, kind, chunk=16):
    h = hadamard(ROTATION)
    entries, errors = {}, []
    for prefix, up_key, down_key in expert_banks(source):
        experts, hidden, intermediate = source.header[up_key]["shape"]
        cache = {}

        def produce(part, prefix=prefix, up_key=up_key, down_key=down_key, cache=cache):
            # w13 then w2 weights, then their scales: each source matrix is quantized once (one bank in memory at a time)
            if part not in cache:
                cache.clear()
                for which, key in (("w13", up_key), ("w2", down_key)):
                    source_weight = source.tensor(key)  # [experts, in, out]
                    weights, scales = [], []
                    for start in range(0, experts, chunk):
                        weight = source_weight[start:start + chunk].transpose(1, 2).float()  # [chunk, out, in]
                        quantized = quantize_int8(rotated(weight, h)) if kind == "int8" else quantize_int4(rotated(weight, h), INT4_GROUP)
                        if start == 0:
                            errors.append(dequantize_check(f"{prefix}{which}", weight[:2], None, tuple(t[:2] for t in quantized), h, kind))
                        weights.append(quantized[0])
                        scales.append(quantized[1])
                    cache[f"{which}_weight"] = tensor_bytes(torch.cat(weights))
                    cache[f"{which}_scales"] = tensor_bytes(torch.cat(scales))
                    source_weight = weights = scales = None
            return cache.pop(part)

        if kind == "int8":
            shapes = {"w13_weight": ("I8", (experts, intermediate, hidden)), "w2_weight": ("I8", (experts, hidden, intermediate)),
                      "w13_scales": ("F32", (experts, intermediate, 1)), "w2_scales": ("F32", (experts, hidden, 1))}
        else:
            shapes = {"w13_weight": ("U8", (experts, intermediate, hidden // 2)), "w2_weight": ("U8", (experts, hidden, intermediate // 2)),
                      "w13_scales": ("F16", (experts, intermediate, hidden // INT4_GROUP)), "w2_scales": ("F16", (experts, hidden, intermediate // INT4_GROUP))}
        for part, (dtype, shape) in shapes.items():
            entries[prefix + part] = (dtype, shape, lambda part=part, produce=produce: produce(part))
        entries[prefix + "convrot_group_size"] = ("I64", (), lambda: tensor_bytes(torch.tensor(ROTATION, dtype=torch.int64)))
    write(output, entries, {"format": f"nemotron_transcribe_experts_{kind}", "convrot_group_size": str(ROTATION)})
    print(f"{kind} experts: mean relative error {sum(errors) / len(errors):.4f}, max {max(errors):.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="folder of the released checkpoint (model.safetensors, configs, tokenizer)")
    parser.add_argument("--draft-source", default=None, help="folder of the DFlash2 drafter release")
    parser.add_argument("--output", required=True, help="checkpoint root (the assets go to its nemotron_transcribe subfolder)")
    parser.add_argument("--parts", default="lm,experts_int8,experts_int4,perception,assets,draft")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    source_dir, output = Path(args.source), Path(args.output)
    (output / ASSET_FOLDER).mkdir(parents=True, exist_ok=True)
    source = Source(source_dir / "model.safetensors")
    parts = args.parts.split(",")
    if "lm" in parts:
        mapping = {key: key[len("llm."):].replace("._fp32_params.", ".") for key in source.header
                   if key.startswith("llm.") and ".experts." not in key and not key.endswith("._extra_state")}
        write(output / LM_FILE, copy_entries(source, mapping), {"format": "nemotron_transcribe_lm"})
        verify_copies(output / LM_FILE, source, mapping)
    for part in ("experts_int8", "experts_int4"):
        if part in parts:
            prepare_experts(source, output / EXPERT_FILES[part], part.split("_")[1])
    if "perception" in parts:
        mapping = {key: key for key in source.header if key.startswith(("perception.", "ctc_timestamp."))}
        metadata = {"format": "nemotron_transcribe_perception"}
        with open(source_dir / "model.safetensors", "rb") as reader:
            length = struct.unpack("<Q", reader.read(8))[0]
            released = json.loads(reader.read(length)).get("__metadata__", {})
        metadata.update({key: value for key, value in released.items() if key.startswith("ctc_timestamp")})
        write(output / ASSET_FOLDER / PERCEPTION_FILE, copy_entries(source, mapping), metadata)
        verify_copies(output / ASSET_FOLDER / PERCEPTION_FILE, source, mapping)
    if "assets" in parts:
        for name, target in ASSETS.items():
            shutil.copyfile(source_dir / name, output / ASSET_FOLDER / target)
        for license_file in sorted((source_dir / "licenses").glob("*")):
            shutil.copyfile(license_file, output / ASSET_FOLDER / license_file.name)
        print(f"copied the assets to {output / ASSET_FOLDER}", flush=True)
    if "draft" in parts and args.draft_source:
        draft = Path(args.draft_source)
        shutil.copyfile(draft / "model.safetensors", output / ASSET_FOLDER / DRAFT_FILE)
        shutil.copyfile(draft / "config.json", output / ASSET_FOLDER / "dflash2_config.json")
        print(f"copied the DFlash2 drafter to {output / ASSET_FOLDER}", flush=True)


if __name__ == "__main__":
    main()
