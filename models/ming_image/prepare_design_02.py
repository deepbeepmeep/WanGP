"""Prepare the two release-specific BF16 files from the pinned 0.2 Design snapshot.

python -m models.ming_image.prepare_design_02 SOURCE CHECKPOINT_ROOT

The shared encoder, vision tower, tokenizer and VAE come from the existing Ming
distribution. The transformer and conditioner retain the family's MMGP key layout.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from safetensors import safe_open

from .split_encoder_checkpoints import read_header


REVISION = "3538f4c311cf8f5256d631ddb7f47793f741aa2e"
CHUNK_BYTES = 16 * 1024 * 1024


SOURCE_SHA256 = {
    "connector/model-00001-of-00002.safetensors": "61de36bbaeb01ea90e951e83c5e51d796212a853cbb3bc46bdda145e14580b15",
    "connector/model-00002-of-00002.safetensors": "00791834d06a42457fb6f2d54ce20157edee41091da4bdc830b63095be4cefcc",
    "mlp/model.safetensors": "7bb7e09c8f56cc568a22afb9e4dfebfdf1efca06586c7dfcb995588cc984ca44",
    "transformer/diffusion_pytorch_model-00001-of-00005.safetensors": "ed3d06ee23f3262cce711ab28474f1d89849c14092c71de26570fcab0f89dd61",
    "transformer/diffusion_pytorch_model-00002-of-00005.safetensors": "efb606d8e090a294a30b8ff6543619f47fef38e0651c415fc1d3ec9805e78739",
    "transformer/diffusion_pytorch_model-00003-of-00005.safetensors": "209c3e966b4277a8a2e10bb2d5da705b5a2d38acb0edefe94ccbcc3673f026fa",
    "transformer/diffusion_pytorch_model-00004-of-00005.safetensors": "f2119547bc314265cd1822ed85bcc1922de0187929c21c200f151f20313fc28c",
    "transformer/diffusion_pytorch_model-00005-of-00005.safetensors": "73a34df0a36e328912c3c9450f06f94e96885bb6e5d335bfc1c52ba99ad65b7f"
}


def prepare(source, groups, target):
    sources = []
    header = {"__metadata__": {"source": "inclusionAI/Ming-Image-0.2-Design", "source_revision": REVISION}}
    offset = 0
    for component, prefix in groups:
        for name, expected_sha in SOURCE_SHA256.items():
            path = source / name
            if path.parent.name != component:
                continue
            digest = hashlib.sha256()
            with path.open("rb") as reader:
                for chunk in iter(lambda: reader.read(CHUNK_BYTES), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise ValueError(f"Source file does not match {REVISION}: {path}")
            entries, _ = read_header(path)
            entries.pop("__metadata__", None)
            if component == "connector":
                # The Bailing core owns input tokenization. The connector consumes embeddings directly.
                entries.pop("model.embed_tokens.weight", None)
            entries = sorted(entries.items(), key=lambda item: item[1]["data_offsets"][0])
            for name, entry in entries:
                key = prefix + name
                if key in header or entry["dtype"] != ("BF16" if component == "transformer" else "F32"):
                    raise ValueError(f"Unexpected key or dtype in {path}: {name}")
                shape = entry["shape"]
                end = offset + math.prod(shape) * 2
                header[key] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, end]}
                offset = end
            sources.append((path, prefix, entries))

    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)

    def stream(output, verify):
        for path, prefix, entries in sources:
            with safe_open(path, framework="pt", device="cpu") as src:
                for name, entry in entries:
                    tensor = src.get_slice(name)
                    rows = max(1, CHUNK_BYTES // (4 * math.prod(entry["shape"][1:])))
                    for start in range(0, entry["shape"][0], rows):
                        data = tensor[start:start + rows].to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
                        if verify:
                            if output.read(len(data)) != data:
                                raise ValueError(f"Tensor data differs: {prefix}{name} at row {start}")
                        else:
                            output.write(data)

    candidate = target
    try:
        if not target.exists():
            with NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".partial", delete=False) as output:
                candidate = Path(output.name)
                output.write(len(encoded).to_bytes(8, "little"))
                output.write(encoded)
                stream(output, verify=False)
        actual, base = read_header(candidate)
        if header != actual or candidate.stat().st_size != base + offset:
            raise ValueError(f"Checkpoint header or length differs: {candidate}")
        with candidate.open("rb") as output:
            output.seek(base)
            stream(output, verify=True)
        if candidate != target:
            candidate.replace(target)
    finally:
        if candidate != target:
            candidate.unlink(missing_ok=True)
    print(f"Verified {target} ({len(header) - 1} tensors, {offset:,} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prepare(args.source, [("transformer", "transformer.")], args.root / "Ming-Image-0.2-Design_bf16.safetensors")
    prepare(args.source, [("connector", "connector."), ("mlp", "")], args.root / "ming_image" / "conditioning_0_2_bf16.safetensors")
