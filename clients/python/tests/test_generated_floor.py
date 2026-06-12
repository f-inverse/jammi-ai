"""Structural guard: the declared runtime floors cannot lie about the stubs.

The wheel ships stubs generated at build time, and those stubs carry
import-time version guards: every ``*_pb2_grpc.py`` raises unless the grpcio
runtime is at least its ``GRPC_GENERATED_VERSION``, and every ``*_pb2.py``
calls ``ValidateProtobufRuntimeVersion`` which raises unless the protobuf
runtime is at least the generator's bundled gencode version. If pyproject's
``grpcio``/``protobuf`` floors fall below what the pinned ``grpcio-tools``
emits, a consumer installing the wheel at its declared minima gets a wheel
that crashes on import — pip resolves it fine, import doesn't.

These tests parse the requirements out of the freshly generated stubs (CI
runs ``make generate`` before pytest) and assert each declared floor
satisfies them. Hermetic: reads only files inside this package.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

CLIENT_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = CLIENT_ROOT / "jammi_client" / "_generated" / "jammi" / "v1"
PYPROJECT = CLIENT_ROOT / "pyproject.toml"

# The grpc guard in generated code, e.g. GRPC_GENERATED_VERSION = '1.80.0'.
_GRPC_GUARD = re.compile(r"GRPC_GENERATED_VERSION = '([0-9.]+)'")

# The protobuf gencode guard, e.g.
# _runtime_version.ValidateProtobufRuntimeVersion(
#     _runtime_version.Domain.PUBLIC, 6, 31, 1, '', 'jammi/v1/catalog.proto')
_PROTOBUF_GUARD = re.compile(
    r"ValidateProtobufRuntimeVersion\(\s*"
    r"_runtime_version\.Domain\.PUBLIC,\s*"
    r"(\d+),\s*(\d+),\s*(\d+),"
)


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.strip(".").split("."))


def _satisfies(floor: tuple[int, ...], required: tuple[int, ...]) -> bool:
    """True when ``floor`` >= ``required``, padding short tuples with zeros."""
    width = max(len(floor), len(required))
    def pad(version: tuple[int, ...]) -> tuple[int, ...]:
        return version + (0,) * (width - len(version))
    return pad(floor) >= pad(required)


def _dependencies() -> list[str]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["dependencies"]


def _dev_extra() -> list[str]:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]["dev"]


def _declared_floor(package: str) -> tuple[int, ...]:
    for spec in _dependencies():
        name = re.split(r"[\s\[><=!~;]", spec.strip(), maxsplit=1)[0]
        if name == package:
            floor = re.search(r">=\s*([0-9][0-9.]*)", spec)
            assert floor, f"`{package}` must declare a `>=` floor, got {spec!r}"
            return _version_tuple(floor.group(1))
    raise AssertionError(f"`{package}` missing from [project.dependencies]")


def _generated_files(pattern: str) -> list[Path]:
    files = sorted(GENERATED_DIR.glob(pattern))
    assert files, (
        f"no {pattern} under {GENERATED_DIR} — run `make generate` first; "
        "this guard checks the floors against freshly generated stubs"
    )
    return files


def test_grpcio_floor_satisfies_every_generated_guard():
    floor = _declared_floor("grpcio")
    for stub in _generated_files("*_pb2_grpc.py"):
        guard = _GRPC_GUARD.search(stub.read_text())
        assert guard, f"no GRPC_GENERATED_VERSION guard in {stub.name}"
        required = _version_tuple(guard.group(1))
        assert _satisfies(floor, required), (
            f"{stub.name} raises at import below grpcio {guard.group(1)}, but "
            f"pyproject declares grpcio>={'.'.join(map(str, floor))} — bump the "
            "floor to match the pinned grpcio-tools generator"
        )


def test_protobuf_floor_satisfies_every_generated_gencode():
    floor = _declared_floor("protobuf")
    for stub in _generated_files("*_pb2.py"):
        guard = _PROTOBUF_GUARD.search(stub.read_text())
        assert guard, f"no ValidateProtobufRuntimeVersion guard in {stub.name}"
        required = tuple(int(g) for g in guard.groups())
        assert _satisfies(floor, required), (
            f"{stub.name} raises at import below protobuf "
            f"{'.'.join(guard.groups())}, but pyproject declares "
            f"protobuf>={'.'.join(map(str, floor))} — bump the floor to match "
            "the gencode the pinned grpcio-tools generator emits"
        )


def test_generator_is_exactly_pinned():
    """The floors are derived from what one exact generator emits; a range
    would let a routine CI run regenerate with a newer toolchain and ship a
    wheel whose floors silently lie again."""
    specs = [s for s in _dev_extra() if s.strip().startswith("grpcio-tools")]
    assert specs, "`grpcio-tools` missing from the dev extra"
    (spec,) = specs
    assert re.fullmatch(r"grpcio-tools==[0-9.]+", spec.strip()), (
        f"grpcio-tools must be pinned with `==`, got {spec!r}"
    )
