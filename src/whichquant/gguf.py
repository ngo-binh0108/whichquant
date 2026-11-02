"""Read the header of a GGUF file: metadata and tensor shapes, never the weights.

The reader only needs a `read(n)` callable, so the same code parses a local
file or a remote one fetched with HTTP range requests.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Callable

# ggml type id -> (name, elements per block, bytes per block), from ggml/src/ggml.c
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
    43: ("F8_E4M3", 1, 1),
}

# GGUF metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
# String arrays longer than this (the tokenizer vocabulary) are skipped, not decoded.
_MAX_STRINGS = 4096


class GGUFError(ValueError):
    pass


class SkippedStrings:
    """A long string array (the tokenizer vocabulary) that was skipped; only its length is kept."""

    def __init__(self, count: int):
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"<{self.count} strings>"


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"type{self.type_id}", 1, 0))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        if self.type_id not in GGML_TYPES:
            raise GGUFError(f"tensor {self.name}: unknown ggml type {self.type_id}")
        _, block, size = GGML_TYPES[self.type_id]
        return self.n_elements // block * size


@dataclass
class GGUFHeader:
    version: int
    metadata: dict[str, Any]
    tensors: list[Tensor] = field(default_factory=list)
    header_size: int = 0
    tensor_info_offset: int = 0  # where the tensor table starts, right after the metadata

    @property
    def arch(self) -> str:
        return self.metadata.get("general.architecture", "unknown")

    def get(self, key: str, default: Any = None) -> Any:
        """Look up an architecture key, e.g. get("block_count") -> "<arch>.block_count"."""
        return self.metadata.get(f"{self.arch}.{key}", default)


class _Stream:
    def __init__(self, read: Callable[[int], bytes]):
        self._read = read
        self.pos = 0

    def read(self, n: int) -> bytes:
        data = self._read(n)
        if len(data) != n:
            raise GGUFError("file ended inside the GGUF header")
        self.pos += n
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.read(self.unpack("<Q")).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        self.read(self.unpack("<Q"))

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.unpack(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            itype, count = self.unpack("<I"), self.unpack("<Q")
            if itype in _SCALARS:
                fmt = _SCALARS[itype]
                size = struct.calcsize(fmt)
                raw = self.read(size * count)
                return list(struct.unpack(f"<{count}{fmt[1:]}", raw))
            if itype == _STRING and count > _MAX_STRINGS:
                for _ in range(count):
                    self.skip_string()
                return SkippedStrings(count)
            return [self.value(itype) for _ in range(count)]
        raise GGUFError(f"unknown GGUF value type {vtype}")


def read_header(read: Callable[[int], bytes]) -> GGUFHeader:
    """Parse a GGUF header from a `read(n) -> bytes` callable positioned at byte 0."""
    s = _Stream(read)
    if s.read(4) != b"GGUF":
        raise GGUFError("not a GGUF file")
    version = s.unpack("<I")
    if version < 2:
        raise GGUFError(f"GGUF version {version} is too old")
    n_tensors, n_kv = s.unpack("<Q"), s.unpack("<Q")
    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = s.string()
        metadata[key] = s.value(s.unpack("<I"))
    info_offset = s.pos
    tensors = _tensor_table(s, n_tensors)
    return GGUFHeader(version, metadata, tensors, s.pos, info_offset)


def _tensor_table(s: _Stream, n_tensors: int) -> list[Tensor]:
    tensors = []
    for _ in range(n_tensors):
        name = s.string()
        n_dims = s.unpack("<I")
        if not 1 <= n_dims <= 4:
            raise GGUFError(f"tensor {name!r} has {n_dims} dimensions")
        shape = tuple(s.unpack("<Q") for _ in range(n_dims))
        type_id = s.unpack("<I")
        s.unpack("<Q")  # data offset
        tensors.append(Tensor(name, shape, type_id))
    return tensors


def read_counts(data: bytes) -> tuple[int, int]:
    """(tensor count, metadata count) from the first 24 bytes of a GGUF file."""
    if len(data) < 24 or data[:4] != b"GGUF":
        raise GGUFError("not a GGUF file")
    return struct.unpack("<QQ", data[8:24])


def read_tensor_table(data: bytes, start: int, n_tensors: int) -> list[Tensor]:
    """Parse a tensor table that starts at `start` in `data`."""
    view = memoryview(data)
    pos = start

    def read(n: int) -> bytes:
        nonlocal pos
        chunk = bytes(view[pos : pos + n])
        pos += n
        return chunk

    tensors = _tensor_table(_Stream(read), n_tensors)
    for t in tensors:
        if t.type_id not in GGML_TYPES:
            raise GGUFError(f"tensor {t.name}: unknown ggml type {t.type_id}")
    return tensors


def read_local(path) -> GGUFHeader:
    with open(path, "rb") as f:
        return read_header(f.read)
