# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Metadata values of a GGUF file, read without its tensors."""
import struct

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _string(reader):
    return reader.read(struct.unpack("<Q", reader.read(8))[0]).decode("utf-8", "replace")


def _value(reader, kind, keep):
    if kind == 8:
        return _string(reader)
    if kind == 9:
        element, count = struct.unpack("<IQ", reader.read(12))
        if element == 8:
            items = [_string(reader) for _ in range(count)]
            return items if keep else None
        fmt = _SCALARS[element]
        data = reader.read(struct.calcsize(fmt) * count)
        return list(struct.unpack(f"<{count}{fmt[1]}", data)) if keep else None
    fmt = _SCALARS[kind]
    return struct.unpack(fmt, reader.read(struct.calcsize(fmt)))[0]


def read_metadata(path, keys):
    """The values of the metadata keys of a GGUF file (arrays as lists)."""
    keys, found = set(keys), {}
    with open(path, "rb") as reader:
        if reader.read(4) != b"GGUF":
            raise ValueError(f"Not a GGUF file: {path}")
        _version, _tensors, count = struct.unpack("<IQQ", reader.read(20))
        for _ in range(count):
            name = _string(reader)
            kind = struct.unpack("<I", reader.read(4))[0]
            value = _value(reader, kind, name in keys)
            if name in keys:
                found[name] = value
                if len(found) == len(keys):
                    break
    return found
