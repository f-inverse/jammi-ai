"""Programmatic ``q8_0`` GGUF fixture generation (issue #351).

The book has no Rust/candle binding, so it cannot call
``candle_core::quantized::gguf_file::write`` directly the way the engine's own
hermetic fixture builder does
(``crates/jammi-ai/tests/it/gguf_qlora.rs::write_gguf_checkpoint``). This module
is a pure-Python re-derivation of that exact byte format, built once at
chapter-setup time from an existing safetensors fixture directory — never a
checked-in binary ``.gguf``, matching this book's "no invented binary artifacts"
convention.

Two things are reproduced byte-for-byte against the engine's own quantized
encoder, both verified against the pinned ``candle-core`` 0.11.0 source under
``~/.cargo/registry`` while this module was written:

* **The GGUF v2 container** — magic ``0x47475546`` ("GGUF"), version 2, 8-byte
  (u64) length prefixes throughout, 32-byte tensor-data alignment, dims stored
  **reversed** relative to a row-major numpy shape (candle's own
  ``gguf_file::write``/``Content::read`` round-trip convention). No metadata KV
  pairs are written — ``load_gguf_backbone`` reads an architecture's geometry
  from the sidecar ``config.json``, never from GGUF metadata (``crates/jammi-ai/
  src/model/backend/gguf.rs``), so an empty metadata table is a faithful
  from-scratch encode, not a shortcut.
* **The ``q8_0`` block-quantization rule** — ``candle-core``
  ``src/quantized/k_quants.rs::BlockQ8_0::from_float``: each contiguous
  (row-major) block of 32 elements is quantized independently as one ``f16``
  scale ``d = max(|x|) / 127`` plus 32 ``int8`` values ``round(x / d)`` (rounded
  half-away-from-zero, matching Rust's ``f32::round``), 34 bytes/block. Every
  matmul-site weight tensor this workspace's GGUF loader treats as a genuinely
  block-quantized dtype (``crates/jammi-ai/src/model/backend/gguf.rs``'s
  ``weight_quantization_from_ggml``) is quantized this way; every other tensor
  (embeddings, LayerNorms, biases) is written densely at ``F32`` (GGUF dtype id
  0), mirroring ``write_gguf_checkpoint``'s own "F32-'quantized' is a legitimate
  lossless wrap" convention for non-matmul-site tensors.

``q8_0``'s block size is 32, and GGUF requires a quantized tensor's **last**
dimension be divisible by the block size (``candle-core``
``src/quantized/mod.rs::check_shape``) — the reason this only works on a
fixture whose ``hidden_size`` (and every matmul-site tensor's last dim) is a
multiple of 32, e.g. this book's ``tiny_bert`` fixture (``hidden_size=32``).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_GGUF_MAGIC = 0x47475546
_GGUF_VERSION = 2
_ALIGNMENT = 32
_GGML_DTYPE_F32 = 0
_GGML_DTYPE_Q8_0 = 8
_Q8_0_BLOCK = 32
_Q8_0_TYPE_SIZE = 34  # 2-byte f16 scale + 32 int8 values


def bert_matmul_site_weight_names(num_layers: int) -> set[str]:
    """The ``.weight`` tensor names GGUF quantizes for a BERT-family checkpoint.

    The six per-layer matmul-site prefixes ``jammi_ai::model::backend::gguf::
    matmul_site_names``'s ``Bert`` arm names, reproduced here the same way
    ``crates/jammi-ai/tests/it/gguf_qlora.rs::bert_matmul_site_prefixes`` does
    (that module is ``pub(crate)``, unreachable from outside the engine crate) —
    an independent re-derivation from the same raw (unwrapped) BERT tensor
    names ``tiny_bert/model.safetensors`` itself carries.
    """
    names: set[str] = set()
    for n in range(num_layers):
        p = f"encoder.layer.{n}"
        for site in (
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ):
            names.add(f"{p}.{site}.weight")
    return names


def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read a ``model.safetensors`` file's F32 tensors, by name, as numpy arrays.

    A minimal from-scratch reader of the safetensors container (an 8-byte LE
    header length, a JSON header, then raw little-endian tensor bytes) — the
    book has no ``safetensors`` Python dependency, and the format is simple
    enough that adding one would buy nothing over reading it directly.
    """
    raw = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    header = json.loads(raw[8 : 8 + header_len])
    data_start = 8 + header_len
    tensors: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "F32":
            raise ValueError(
                f"gguf_fixture only reads F32 safetensors tensors; {name!r} is "
                f"{meta['dtype']!r} in {path}"
            )
        start, end = meta["data_offsets"]
        shape = tuple(meta["shape"])
        arr = np.frombuffer(raw, dtype="<f4", count=(end - start) // 4, offset=data_start + start)
        tensors[name] = arr.reshape(shape).astype(np.float32, copy=True)
    return tensors


def _quantize_q8_0_block(flat: np.ndarray) -> bytes:
    """One ``q8_0`` block: 2-byte f16 scale + 32 int8 values, 34 bytes total.

    Reproduces ``BlockQ8_0::from_float`` exactly: ``d = max(|x|) / 127``,
    ``id = 1/d`` (or ``0`` when ``d == 0``), each element quantized as
    ``round(x * id)`` with round-half-away-from-zero (Rust's ``f32::round``,
    not numpy's default round-half-to-even).
    """
    assert flat.shape == (_Q8_0_BLOCK,)
    # Every step below stays in float32 (never upcasting to float64), matching
    # Rust's `f32`-only arithmetic in `BlockQ8_0::from_float` exactly — an
    # upcast changes the last-bit rounding of `x * id` at exact `.5` block
    # boundaries and would silently diverge from the engine's own q8_0 bytes.
    amax = np.max(np.abs(flat)).astype(np.float32)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inv_d = (np.float32(1.0) / d) if d != np.float32(0.0) else np.float32(0.0)
    scaled = (flat * inv_d).astype(np.float32)
    # Round-half-away-from-zero, matching Rust's `f32::round`.
    rounded = np.trunc(scaled + np.copysign(np.float32(0.5), scaled))
    qs = np.clip(rounded, -128, 127).astype(np.int8)
    return struct.pack("<e", float(d)) + qs.tobytes()


def _quantize_q8_0(tensor: np.ndarray) -> bytes:
    """Quantize a row-major tensor to ``q8_0``: sequential 32-element blocks
    over its flattened (C-order) element stream, matching
    ``QTensor::quantize``'s own ``flatten_all()`` + block-chunk convention."""
    flat = tensor.reshape(-1).astype(np.float32)
    n = flat.shape[0]
    if n % _Q8_0_BLOCK != 0:
        raise ValueError(f"q8_0 requires elem_count % 32 == 0, got {n}")
    out = bytearray()
    for i in range(0, n, _Q8_0_BLOCK):
        out += _quantize_q8_0_block(flat[i : i + _Q8_0_BLOCK])
    return bytes(out)


def _write_string(buf: bytearray, s: str) -> None:
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _gguf_bytes(tensors: dict[str, np.ndarray], matmul_site_weight_names: set[str]) -> bytes:
    """Encode ``tensors`` as a GGUF v2 file, byte-for-byte against
    ``candle_core::quantized::gguf_file::write``'s own layout (module doc)."""
    names = sorted(tensors)  # deterministic write order

    header = bytearray()
    header += struct.pack("<I", _GGUF_MAGIC)
    header += struct.pack("<I", _GGUF_VERSION)
    header += struct.pack("<Q", len(names))  # tensor_count
    header += struct.pack("<Q", 0)  # metadata_kv_count — none (module doc)

    payloads: list[bytes] = []
    offset = 0
    for name in names:
        t = tensors[name]
        is_quantized = name in matmul_site_weight_names
        dtype_id = _GGML_DTYPE_Q8_0 if is_quantized else _GGML_DTYPE_F32
        data = _quantize_q8_0(t) if is_quantized else t.astype("<f4").tobytes()

        _write_string(header, name)
        dims = list(t.shape)
        header += struct.pack("<I", len(dims))
        for dim in reversed(dims):  # GGUF stores dims fastest-varying-first
            header += struct.pack("<Q", dim)
        header += struct.pack("<I", dtype_id)
        header += struct.pack("<Q", offset)

        size = len(data)
        pad = (-size) % _ALIGNMENT
        payloads.append(data + b"\x00" * pad)
        offset += size + pad

    pad = (-len(header)) % _ALIGNMENT
    header += b"\x00" * pad

    return bytes(header) + b"".join(payloads)


def write_q8_0_gguf(safetensors_dir: Path, num_layers: int, dest_dir: Path) -> Path:
    """Write ``dest_dir/model.gguf``: ``safetensors_dir/model.safetensors``'s
    tensors, matmul-site weights quantized at ``q8_0``, everything else dense
    ``F32`` (module doc). Returns the written path.

    ``dest_dir`` is expected to already carry the sidecar ``config.json`` /
    ``tokenizer.json`` this book's resolve→load→embed path reads alongside
    ``model.gguf`` — this function writes only the weights file.
    """
    tensors = _read_safetensors(safetensors_dir / "model.safetensors")
    matmul_sites = bert_matmul_site_weight_names(num_layers)
    missing = matmul_sites - set(tensors)
    if missing:
        raise ValueError(f"matmul-site tensor(s) not found in {safetensors_dir}: {sorted(missing)}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "model.gguf"
    out_path.write_bytes(_gguf_bytes(tensors, matmul_sites))
    return out_path
