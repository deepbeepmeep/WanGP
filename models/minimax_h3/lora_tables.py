"""Small, fingerprinted AdaLN conversion packages shared by H3 checkpoints and adapters."""

import base64
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load, save


MAP_DIR = Path(__file__).with_name("lora_affine_maps")
METADATA_KEY = "wangp_h3_lora_affine"
SOURCE_TENSOR = "__wangp_h3_lora_source__"
TIME_KEYS = ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")


def tensor_fingerprint(tensor):
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def time_fingerprint(tensors):
    return hashlib.sha256("".join(tensor_fingerprint(tensors[key]) for key in TIME_KEYS).encode()).hexdigest()


@dataclass
class AffinePackage:
    name: str
    table: torch.Tensor
    affine: torch.Tensor
    full_fingerprint: str = ""

    @property
    def width(self):
        return self.table.shape[1]

    @property
    def table_fingerprint(self):
        return tensor_fingerprint(self.table)


def package_bytes(package):
    return save({"adaln_t_table": package.table.float().cpu().contiguous(), "adaln_affine_map": package.affine.float().cpu().contiguous()},
                metadata={"h3_map_id": package.name, "full_time_fingerprint": package.full_fingerprint})


def read_package(data, name="embedded"):
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("MiniMax H3 affine package exceeds 4 MiB")
    tensors = load(data)
    table, affine = tensors["adaln_t_table"], tensors["adaln_affine_map"]
    if table.dtype != torch.float32 or affine.dtype != torch.float32 or table.ndim != 2 or not 2 <= table.shape[0] <= 4097 or table.shape[1] not in (4, 8, 64) or affine.shape != (table.shape[1] + 1, 2688):
        raise ValueError("Invalid MiniMax H3 affine package dimensions or dtype")
    if not torch.isfinite(table).all() or not torch.isfinite(affine).all():
        raise ValueError("MiniMax H3 affine package contains non-finite values")
    header_size = int.from_bytes(data[:8], "little")
    metadata = json.loads(data[8:8 + header_size]).get("__metadata__", {})
    return AffinePackage(metadata.get("h3_map_id", name), table, affine, metadata.get("full_time_fingerprint", ""))


def encode_package(package):
    data = package_bytes(package)
    return json.dumps({"version": 1, "sha256": hashlib.sha256(data).hexdigest(), "package": base64.b64encode(data).decode("ascii")}, separators=(",", ":"))


def decode_package(value):
    if not isinstance(value, str) or len(value) > 6 * 1024 * 1024:
        raise ValueError("Invalid MiniMax H3 affine metadata")
    payload = json.loads(value)
    if payload["version"] != 1:
        raise ValueError("Unsupported MiniMax H3 affine metadata version")
    data = base64.b64decode(payload["package"], validate=True)
    if hashlib.sha256(data).hexdigest() != payload["sha256"]:
        raise ValueError("MiniMax H3 affine metadata checksum mismatch")
    return read_package(data)


@lru_cache(maxsize=64)
def _read_local(path, mtime, size):
    return read_package(Path(path).read_bytes(), Path(path).stem)


def local_packages():
    return [_read_local(str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in sorted(MAP_DIR.glob("*.sft"))]


def _unique(matches):
    if not matches:
        return None
    first = matches[0]
    if any(not torch.equal(first.affine, other.affine) or not torch.equal(first.table, other.table) for other in matches[1:]):
        raise ValueError("Ambiguous MiniMax H3 affine fingerprint: multiple conversion maps match")
    return first


def detect_table(table):
    fingerprint = tensor_fingerprint(table)
    matches = []
    for package in local_packages():
        if table.shape[1] == 4 and package.width == 8:
            package = AffinePackage(package.name, package.table[:, :4], torch.cat((package.affine[:4], package.affine[-1:])), package.full_fingerprint)
        if package.table_fingerprint == fingerprint:
            matches.append(package)
    return _unique(matches)


def _conditioning_tensors(source):
    keys = source.keys()
    table_keys = [key for key in keys if key.endswith("adaln_t_table") and not key.endswith("ref2va_adaln_t_table")]
    if len(table_keys) == 1:
        return table_keys[0], source.get_tensor(table_keys[0]), None
    time_keys = [key for key in keys if key.endswith("time_embedder.proj_in.weight")]
    if len(time_keys) != 1:
        return None, None, None
    prefix = time_keys[0][:-len("proj_in.weight")]
    return None, None, {key: source.get_tensor(prefix + key) for key in TIME_KEYS}


def checkpoint_tensors(filename):
    """Read only the conditioning table or four timestep tensors, never transformer weights."""
    with safe_open(str(filename), framework="pt", device="cpu") as source:
        _, table, times = _conditioning_tensors(source)
        return source.metadata() or {}, table, times


def validate_checkpoint_package(package, table, times):
    if table is not None and package.table_fingerprint != tensor_fingerprint(table):
        raise ValueError("MiniMax H3 embedded affine map does not match the checkpoint table")
    if times is not None and (not package.full_fingerprint or package.full_fingerprint != time_fingerprint(times)):
        raise ValueError("MiniMax H3 embedded affine map does not match the checkpoint timestep embedder")
    if table is None and times is None:
        raise ValueError("MiniMax H3 affine metadata has no checkpoint conditioning tensors to validate")


def _native_package(source, table_key, table, name):
    if table.ndim != 2 or not 2 <= table.shape[0] <= 4097 or table.shape[1] not in (4, 8, 64) or not table.is_floating_point():
        return None
    keys = source.keys()
    prefix = table_key.removesuffix("adaln_t_table")
    basis_key, mean_key = prefix + "adaln_basis", prefix + "adaln_mean"
    if basis_key not in keys or mean_key not in keys:
        return None
    if source.get_slice(basis_key).get_shape() != [table.shape[1], 2688] or source.get_slice(mean_key).get_shape() != [2688]:
        return None
    basis, mean = source.get_tensor(basis_key), source.get_tensor(mean_key)
    if not basis.is_floating_point() or not mean.is_floating_point():
        return None
    # The full timestep embedding is table @ basis + mean.
    table, affine = table.float(), torch.cat((basis.float(), mean.float().unsqueeze(0)))
    if not torch.isfinite(table).all() or not torch.isfinite(affine).all():
        return None
    if torch.linalg.matrix_rank(affine[:-1].double()) != table.shape[1]:
        return None
    return AffinePackage(name, table, affine)


@lru_cache(maxsize=16)
def _detect_checkpoint(path, mtime, size, allow_native):
    with safe_open(path, framework="pt", device="cpu") as source:
        table_key, table, times = _conditioning_tensors(source)
        metadata = source.metadata() or {}
        if METADATA_KEY in metadata:
            package = decode_package(metadata[METADATA_KEY])
            validate_checkpoint_package(package, table, times)
            return package
        if table is not None:
            package = detect_table(table)
            if package is None and allow_native:
                package = _native_package(source, table_key, table, Path(path).stem)
            return package
    if times is not None:
        fingerprint = time_fingerprint(times)
        return _unique([package for package in local_packages() if package.full_fingerprint == fingerprint])
    return None


def detect_checkpoint(filename, *, allow_native=True):
    path = Path(filename)
    if path.suffix.lower() not in (".safetensors", ".sft"):
        return None
    stat = path.stat()
    return _detect_checkpoint(str(path.resolve()), stat.st_mtime_ns, stat.st_size, allow_native)


def source_tag(package):
    """A tensor tag survives adapter loaders that discard file-level metadata."""
    return torch.frombuffer(bytearray(encode_package(package).encode("utf-8")), dtype=torch.uint8)


def pop_source_tag(state_dict):
    value = state_dict.pop(SOURCE_TENSOR, None)
    if value is None:
        return None
    if value.dtype != torch.uint8 or value.ndim != 1 or value.numel() > 6 * 1024 * 1024:
        raise ValueError("Invalid MiniMax H3 LoRA source-table tag")
    return decode_package(value.cpu().numpy().tobytes().decode("utf-8"))
