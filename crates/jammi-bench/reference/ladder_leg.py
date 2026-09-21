"""What every PyTorch-reference ladder leg captures the same way.

The PyTorch twin of `crates/jammi-bench/src/leg.rs`: a leg is one run of one
implementation stack of a workload — identity (what two legs must agree on to
be comparable), provenance (recorded, never compared) and measurements. A
producer emits legs and decides nothing, so this module holds capture only:

* `IterationSeries` — the per-iteration wall-clock series after warm-up;
* `peak_rss_bytes` — the kernel's high-water mark of this process's resident
  set, the same instrument the engine rungs report;
* `artifact_of` / `write_jsonl` — outcome files, addressed by their sha256;
* `read_keyed_vectors` / `write_keyed_vectors` / `keyed_vector_digest` — the
  one on-disk shape and the one checksum for "one f32 vector per key";
* `leg_per_point` — one fresh process per sweep point, because the resident
  high-water mark never falls;
* `emit` — the document a leg-producing script prints.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

VECTORS_TENSOR = "vectors"


class IterationSeries:
    """Drops the first `warmup` recorded iterations, keeps the rest in order."""

    def __init__(self, warmup: int, iterations: int) -> None:
        self.warmup = warmup
        self.iterations = iterations
        self._recorded = 0
        self.seconds: list[float] = []

    @property
    def total(self) -> int:
        return self.warmup + self.iterations

    def record(self, elapsed_s: float) -> None:
        if self._recorded >= self.warmup:
            self.seconds.append(elapsed_s)
        self._recorded += 1


def peak_rss_bytes() -> dict[str, Any]:
    """The process's peak resident set as the kernel accounts it: `VmHWM` where
    `/proc` exists, otherwise `getrusage`'s `ru_maxrss` (bytes on macOS,
    kibibytes elsewhere). Both are the same high-water mark."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return {"value": float(line.split()[1]) * 1024.0, "unit": "bytes"}
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return {"value": float(maxrss * scale), "unit": "bytes"}


NOT_MEASURED_BYTES = {"value": None, "unit": "bytes"}


def artifact_of(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write_jsonl(directory: Path, name: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return artifact_of(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_keyed_vectors(tensor_file: Path):
    """`(keys, [n, d] float32 tensor)` of a keyed-vector file and its sibling
    `<stem>.keys.txt`."""
    from safetensors.torch import load_file

    keys = tensor_file.with_suffix("").with_suffix(".keys.txt").read_text(encoding="utf-8").splitlines()
    vectors = load_file(str(tensor_file))[VECTORS_TENSOR]
    if vectors.shape[0] != len(keys):
        raise ValueError(f"{tensor_file}: {vectors.shape[0]} rows but {len(keys)} keys")
    return keys, vectors


def write_keyed_vectors(directory: Path, stem: str, keys: Sequence[str], vectors) -> dict[str, Any]:
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.safetensors"
    save_file({VECTORS_TENSOR: vectors.detach().to("cpu").float().contiguous()}, str(path))
    (directory / f"{stem}.keys.txt").write_text("".join(f"{k}\n" for k in keys), encoding="utf-8")
    return artifact_of(path)


def keyed_vector_digest(keys: Sequence[str], vectors) -> str:
    """FNV-1a over the rows in the order given: each key's bytes, a separator,
    each lane's little-endian f32 bits — `leg.rs`'s `keyed_vector_digest`."""
    import numpy as np

    lanes = np.ascontiguousarray(vectors.detach().to("cpu").float().numpy()).astype("<f4")
    mask = (1 << 64) - 1
    h = 0xCBF29CE484222325
    for key, row in zip(keys, lanes):
        for byte in key.encode("utf-8") + b"\xff" + row.tobytes():
            h = ((h ^ byte) * 0x100000001B3) & mask
    return f"{h:016x}"


def package_versions(*names: str) -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {name: version(name) for name in names}


def provenance(**extra: Any) -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "packages": package_versions("torch", "torch_geometric", "torch_cluster", "pyg-lib", "safetensors", "numpy"),
        "peak_rss_source": "VmHWM" if Path("/proc/self/status").is_file() else "getrusage.ru_maxrss",
        **extra,
    }


def leg(workload: str, identity: dict[str, Any], provenance: dict[str, Any], measured: dict[str, Any], rung: str = "torch") -> dict[str, Any]:
    return {"workload": workload, "rung": rung, "identity": identity, "provenance": provenance, "measured": measured}


def emit(script: str, legs: list[dict[str, Any]]) -> None:
    print(json.dumps({"script": script, "legs": legs}, indent=2))


def leg_per_point(points: Sequence[Any], in_process: Callable[[Any], dict[str, Any]], argv_for: Callable[[Any], list[str]]) -> list[dict[str, Any]]:
    """One leg per point, each owning its process's peak resident set: a single
    point is measured here; several are a sweep and each runs in a fresh
    interpreter, `argv_for(point)` being this script's arguments for that one
    point."""
    if len(points) == 1:
        return [in_process(points[0])]
    legs = []
    for point in points:
        done = subprocess.run([sys.executable, sys.argv[0], *argv_for(point)], capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"child {argv_for(point)} exited {done.returncode}:\n{done.stderr}")
        child = json.loads(done.stdout)["legs"]
        if len(child) != 1:
            raise RuntimeError(f"child {argv_for(point)} did not print exactly one leg")
        legs.append(child[0])
    return legs
