from __future__ import annotations

import ast
import io
import struct
import zipfile
from pathlib import Path

import torch


_DTYPE_MAP = {
    "<f4": torch.float32,
    "|f4": torch.float32,
    "<f8": torch.float64,
    "|u1": torch.uint8,
    "<i8": torch.int64,
    "<i4": torch.int32,
}


def _parse_npy_header(buffer: bytes) -> tuple[dict, int]:
    if buffer[:6] != b"\x93NUMPY":
        raise ValueError("invalid npy header")
    major = buffer[6]
    if major == 1:
        header_len = struct.unpack("<H", buffer[8:10])[0]
        offset = 10
    else:
        header_len = struct.unpack("<I", buffer[8:12])[0]
        offset = 12
    header = ast.literal_eval(buffer[offset : offset + header_len].decode("latin1").strip())
    return header, offset + header_len


def _torch_dtype_from_descr(descr: str) -> torch.dtype:
    if descr not in _DTYPE_MAP:
        raise ValueError(f"unsupported npy dtype descriptor: {descr}")
    return _DTYPE_MAP[descr]


def load_npy_bytes_as_tensor(buffer: bytes) -> torch.Tensor:
    header, payload_offset = _parse_npy_header(buffer)
    dtype = _torch_dtype_from_descr(header["descr"])
    shape = tuple(header["shape"])
    tensor = torch.frombuffer(memoryview(buffer)[payload_offset:], dtype=dtype)
    tensor = tensor.reshape(shape)
    return tensor.clone()


def load_npz_tensor_dict(path: str | Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            with zf.open(name, "r") as f:
                tensors[name.replace(".npy", "")] = load_npy_bytes_as_tensor(f.read())
    return tensors
