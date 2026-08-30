#!/usr/bin/env python3
"""Rename a stock Google-checkpoint-named safetensors file's LayerNorm
parameters to the names jammi's loader expects (unit #356 census
execution fix, defect 2).

The public `bert-base-uncased` safetensors checkpoint carries its
LayerNorm affine parameters under the ORIGINAL Google BERT naming --
`...LayerNorm.gamma` / `...LayerNorm.beta` -- not the `weight`/`bias`
names every other tensor (and every OTHER framework's loader) uses. HF
`transformers` silently aliases `gamma`->`weight` and `beta`->`bias` on
load; jammi's own loader does not (that engine-side gap is filed
separately -- this script is the tracked, deterministic workaround so a
stock checkpoint can be loaded at all, never a fix to the loader itself).

Renames every tensor name whose LAST path component is exactly `gamma`
(-> `weight`) or `beta` (-> `bias`) -- suffix-anchored: a name that merely
CONTAINS "gamma"/"beta" earlier in the string, or as part of a longer
final component, is left untouched. Refuses loudly (no output written) if:

  - the rename would produce a NAME COLLISION -- two tensors (a renamed
    one and either another renamed one or an already-`weight`/`bias`-named
    one already present) mapping to the same output name would silently
    drop a tensor in the round-trip; this script never resolves a
    collision on the caller's behalf.
  - ZERO renames would occur -- a safetensors file with no
    `.gamma`/`.beta` suffix at all is not a legacy Google-named BERT
    checkpoint (the caller likely pointed this at the wrong file, or a
    checkpoint that is already jammi-compatible), and running it through
    unchanged would silently produce a byte-identical no-op a caller could
    mistake for "conversion succeeded".
  - `out_path` resolves to the SAME file as `in_path` -- this script never
    rewrites in place; the caller keeps `config.json`/`tokenizer.json`
    alongside the ORIGINAL checkpoint and merely retargets those at the
    new file this script writes.

Uses the `safetensors` package (`safetensors.deserialize`/`serialize`/
`TensorSpec`, no `numpy`/`torch` needed) to actually parse and
re-serialize the checkpoint -- both a real-format validation (a malformed
file fails inside `deserialize`, not inside hand-rolled header parsing
here) and the guarantee that every renamed tensor's VALUE is copied
byte-for-byte from input to output (only the header's name-and-metadata
JSON changes; each tensor's raw byte buffer is passed through unmodified).
Loud degrade if `safetensors` is not importable (the `fixture_width_report.py`
precedent): refuses with a named, non-network, non-torch error rather than
an unhandled `ImportError` traceback. Copies nothing else -- only the one
named safetensors file; `config.json`/`tokenizer.json` are the caller's own
responsibility to keep alongside whichever path this script wrote.

Deterministic: same input bytes -> same renamed set -> the same output
bytes modulo the safetensors writer's own (stable) header serialization.
No network, no GPU, no build.

Usage: convert_legacy_bert_checkpoint.py IN.safetensors OUT.safetensors

Exit codes: 0 = converted, rename count printed; 1 = `safetensors` package
not importable; 2 = usage error (missing input file, or out_path ==
in_path); 3 = a rename collision or a zero-rename result (refused, no
output written).
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

# The on-disk safetensors dtype CODE (`safetensors.deserialize()`'s own
# vocabulary, e.g. "F32") -> the friendly constructor name
# `safetensors.TensorSpec` validates against (e.g. "float32"). Every stock
# `bert-base-uncased` tensor is F32; the rest of this table covers every
# other concrete, UNPACKED dtype a real checkpoint might carry. Packed
# dtypes (e.g. `float4_e2m1fn_x2`, two values per byte) are deliberately
# NOT in this table -- `convert()` refuses loudly on an unrecognized code
# rather than risk silently mis-encoding a packed tensor's element count.
_DTYPE_CODE_TO_CTOR = {
    "BOOL": "bool",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "U16": "uint16",
    "I32": "int32",
    "U32": "uint32",
    "I64": "int64",
    "U64": "uint64",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "BF16": "bfloat16",
    "C64": "complex64",
}

GAMMA_SUFFIX = ".gamma"
BETA_SUFFIX = ".beta"


class CheckpointConversionError(RuntimeError):
    """Named exception for both loud-refusal conditions this module owns:
    a rename collision, a zero-rename (wrong-input) result, an unsupported
    (packed/exotic) dtype, or `out_path == in_path`. See module doc."""


def renamed_name(name: str) -> str:
    """Suffix-anchored `.gamma`->`.weight` / `.beta`->`.bias` rename.
    Returns `name` unchanged if neither suffix matches -- a name that
    contains "gamma"/"beta" but not as its OWN trailing dotted component
    (e.g. a hypothetical `gamma_scale.other`) is never touched."""
    if name.endswith(GAMMA_SUFFIX):
        return name[: -len(GAMMA_SUFFIX)] + ".weight"
    if name.endswith(BETA_SUFFIX):
        return name[: -len(BETA_SUFFIX)] + ".bias"
    return name


def convert(in_path: str, out_path: str) -> int:
    """Reads the safetensors file at `in_path`, renames every
    `.gamma`/`.beta` suffix, writes the result to `out_path`. Returns the
    number of tensors renamed (always > 0 on success). Raises
    `CheckpointConversionError` on any refusal condition (see module doc)
    -- `out_path` is never opened for writing until every check above has
    passed, so a refusal never leaves a partial/truncated file behind.

    Imports `safetensors` lazily (inside this function, not at module
    import time) so callers that only want `renamed_name`'s pure string
    logic -- e.g. this module's own test suite -- never pay the import
    cost or the ImportError risk; `main()` performs its own top-level
    importability check first so the CLI path still refuses loudly, in
    one place, before any file I/O.
    """
    in_abspath = Path(in_path).resolve()
    out_abspath = Path(out_path).resolve()
    if in_abspath == out_abspath:
        raise CheckpointConversionError(
            f"in_path and out_path both resolve to {in_abspath} -- refusing an in-place rewrite; "
            "write to a different path"
        )

    from safetensors import TensorSpec, deserialize, serialize

    raw = Path(in_path).read_bytes()
    items = deserialize(raw)  # [(name, {"shape", "dtype", "data"}), ...]

    new_names: dict[str, str] = {}
    seen: set[str] = set()
    renamed_count = 0
    for name, _info in items:
        new_name = renamed_name(name)
        if new_name != name:
            renamed_count += 1
        if new_name in seen:
            raise CheckpointConversionError(
                f"name collision: renaming produces {new_name!r} more than once (at least one of "
                f"the tensors mapping to it is {name!r}) -- refusing rather than silently dropping "
                "a tensor"
            )
        seen.add(new_name)
        new_names[name] = new_name

    if renamed_count == 0:
        raise CheckpointConversionError(
            f"{in_path}: zero '{GAMMA_SUFFIX}'/'{BETA_SUFFIX}' suffixed tensor names found -- "
            "refusing (this does not look like a legacy Google-named BERT checkpoint; wrong input?)"
        )

    tensor_dict = {}
    buffers: list[tuple] = []  # keeps every backing buffer alive until serialize() returns
    for name, info in items:
        data = info["data"]
        buf = bytearray(data)
        arr = (ctypes.c_char * len(buf)).from_buffer(buf)
        buffers.append((buf, arr))
        code = info["dtype"]
        try:
            ctor_dtype = _DTYPE_CODE_TO_CTOR[code]
        except KeyError as e:
            raise CheckpointConversionError(
                f"tensor {name!r} has dtype {code!r}, which this converter does not know how to "
                "re-serialize (packed/exotic dtype) -- refusing rather than risk a silent "
                "mis-encode"
            ) from e
        tensor_dict[new_names[name]] = TensorSpec(
            dtype=ctor_dtype,
            shape=info["shape"],
            data_ptr=ctypes.addressof(arr),
            data_len=len(buf),
        )

    out_blob = serialize(tensor_dict)
    Path(out_path).write_bytes(out_blob)
    return renamed_count


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        prog="convert_legacy_bert_checkpoint.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s IN.safetensors OUT.safetensors",
    )
    ap.add_argument("in_path", help="a legacy Google-named BERT safetensors checkpoint")
    ap.add_argument("out_path", help="where to write the renamed checkpoint (never in_path)")
    args = ap.parse_args(argv)

    try:
        import safetensors  # noqa: F401
    except ImportError as e:
        print(
            f"::error::convert_legacy_bert_checkpoint: the 'safetensors' package is not "
            f"importable ({e}) -- refusing (no conversion can be produced without it)",
            file=sys.stderr,
        )
        return 1

    if not Path(args.in_path).exists():
        print(
            f"::error::convert_legacy_bert_checkpoint: {args.in_path} does not exist",
            file=sys.stderr,
        )
        return 2

    if Path(args.in_path).resolve() == Path(args.out_path).resolve():
        print(
            f"::error::convert_legacy_bert_checkpoint: in_path and out_path both resolve to "
            f"{Path(args.in_path).resolve()} -- refusing an in-place rewrite; write to a "
            "different path",
            file=sys.stderr,
        )
        return 2

    try:
        n = convert(args.in_path, args.out_path)
    except CheckpointConversionError as e:
        print(f"::error::convert_legacy_bert_checkpoint: {e}", file=sys.stderr)
        return 3

    print(f"convert_legacy_bert_checkpoint: renamed {n} tensor name(s) -> {args.out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
