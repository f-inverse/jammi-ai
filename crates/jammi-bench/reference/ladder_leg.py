"""What every PyTorch-reference leg of the graph-learning workloads captures
the same way — the Python side of `crates/jammi-bench/src/capture.rs`.

A leg is one run of one rung of a workload, filed as
`<legs-dir>/<rung>__<unit>__r<take>.json` with the leg's block at the top
level under the workload's key (`graph_sample`, `propagate`,
`predictor_train_run`): the workload's identity fields, its measurements
(`iter_wall_s`, `work`, `peak_rss_bytes`, `peak_vram_bytes`, `outcome_digest`,
and `vectors_file` + `vector_dim`, `law_observed`, `held_out_example_mean` +
`trajectory` as the workload pairs its outcome), and the facts a rung's
premises read. A producer emits legs and decides nothing — its `iter_wall_s` is every
iteration in run order, and where the run settled is the ladder's to find; the comparison is
`jammi-bench ladder <workload> <legs-dir>`.

* `peak_rss_bytes` — the kernel's high-water mark of this process's resident
  set, the same instrument the engine rungs report;
* `artifact_of` / `write_jsonl` / `read_jsonl` — files by their sha256;
* `read_vector_rows` / `write_vector_rows` / `vector_rows_digest` — the
  ladder's vector rows: little-endian `f32`, row-major, keys beside;
* `file_leg` — the leg file;
* `add_take_argument` — `--take`, the takes of each point, as the engine
  producers' `Takes` reads it;
* `legs_per_point` — one fresh process per point, because the resident
  high-water mark never falls.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def read_vector_rows(vectors_file: Path, dim: int):
    """`(keys, [n, dim] float32 tensor)` of `<stem>.vectors.f32` and its
    sibling `<stem>.keys.txt`."""
    import numpy as np
    import torch

    stem = str(vectors_file)
    if not stem.endswith(".vectors.f32"):
        raise ValueError(f"{vectors_file} is not a `<stem>.vectors.f32` file")
    keys = Path(stem[: -len(".vectors.f32")] + ".keys.txt").read_text(encoding="utf-8").splitlines()
    flat = np.fromfile(str(vectors_file), dtype="<f4")
    if dim == 0 or flat.size % dim != 0 or flat.size // dim != len(keys):
        raise ValueError(f"{vectors_file}: {flat.size} values is not {len(keys)} rows of {dim}")
    return keys, torch.from_numpy(flat.reshape(len(keys), dim).copy())


def write_vector_rows(directory: Path, stem: str, keys: Sequence[str], vectors) -> tuple[dict[str, Any], int]:
    """The ladder's vector rows under `directory/<stem>`: little-endian `f32`,
    row-major, in the order given, the keys one per line beside them.
    Returns the rows file's artifact and the row width."""
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    rows = np.ascontiguousarray(vectors.detach().to("cpu").float().numpy()).astype("<f4")
    if rows.ndim != 2 or rows.shape[0] != len(keys):
        raise ValueError(f"{stem}: {rows.shape} rows for {len(keys)} keys")
    path = directory / f"{stem}.vectors.f32"
    rows.tofile(str(path))
    (directory / f"{stem}.keys.txt").write_text("".join(f"{k}\n" for k in keys), encoding="utf-8")
    return artifact_of(path), int(rows.shape[1])


def vector_rows_digest(keys: Sequence[str], vectors) -> str:
    """FNV-1a over the rows in the order given: each key's bytes, a separator,
    each lane's little-endian f32 bits — `capture.rs`'s `vector_rows_digest`."""
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
    """What this leg was, recorded on the block and never compared."""
    import torch

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "given_cpus": given_cpus(),
        "packages": package_versions("torch", "torch_geometric", "torch_cluster", "pyg-lib", "safetensors", "numpy"),
        "peak_rss_source": "VmHWM" if Path("/proc/self/status").is_file() else "getrusage.ru_maxrss",
        **extra,
    }


def leg_stem(rung: str, unit: str, take: int) -> str:
    """The ladder's leg name: `<rung>__<unit>__r<take>`."""
    return f"{rung}__{unit}__r{take}"


def file_leg(legs_dir: Path, key: str, stem: str, block: dict[str, Any], tool: str) -> str:
    """File `block` as the leg `<stem>.json` under `legs_dir`, at the top
    level under the workload's `key`, and return the file name."""
    legs_dir.mkdir(parents=True, exist_ok=True)
    name = f"{stem}.json"
    (legs_dir / name).write_text(json.dumps({"tool": tool, key: block}, indent=2), encoding="utf-8")
    return name


def given_cpus() -> int:
    """The CPUs this process may run on: its affinity mask where the platform
    has one (a leg pinned with `taskset` is given exactly that set), otherwise
    every logical CPU."""
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1


MIN_REPEATS = 2
"""The fewest takes the ladder measures a rung against itself with —
`SpeedInstrument::MIN_REPEATS`, held to it by `test_ladder_twin_defaults.py`."""


def takes(text: str) -> list[int]:
    """`--take`'s value: comma-separated take numbers, each at least 1."""
    parsed = [int(part) for part in text.split(",")]
    if not parsed or min(parsed) < 1:
        raise ValueError(f"takes are numbered from 1: {text!r}")
    return parsed


def add_take_argument(parser) -> None:
    """`--take 1,2`: the takes each point is measured as, each in a process of
    its own; the default is the fewest the ladder measures a rung against
    itself with. A run naming one take of one point — the invocation a sweep
    hands each point — is that single point, filed as that take."""
    parser.add_argument(
        "--take",
        type=takes,
        default=list(range(1, MIN_REPEATS + 1)),
        help="the takes each point is measured as, comma-separated, each in its own process; the ladder measures a rung against itself with two",
    )


def legs_per_point(points: Sequence[Any], in_process: Callable[[Any], list[str]], argv_for: Callable[[Any], list[str]]) -> list[str]:
    """One leg per point, each owning its process's peak resident set: a single
    point is filed here; several are a sweep and each runs in a fresh
    interpreter, `argv_for(point)` being this script's arguments for that one
    point. Returns every leg's file name; a sweep of no points is refused,
    never an empty filing.

    A leg runs torch's intra-op pool on exactly the CPUs the process was
    given, as the engine's pools do: torch otherwise sizes it to the machine's
    physical cores whatever the affinity mask, and a pinned leg would
    oversubscribe the set its engine counterpart is held to."""
    if not points:
        raise ValueError("the sweep has no points: every swept flag needs a value")
    if len(points) == 1:
        import torch

        torch.set_num_threads(given_cpus())
        return in_process(points[0])
    files: list[str] = []
    for point in points:
        done = subprocess.run([sys.executable, sys.argv[0], *argv_for(point)], capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"child {argv_for(point)} exited {done.returncode}:\n{done.stderr}")
        files.extend(json.loads(done.stdout))
    return files
