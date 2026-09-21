#!/usr/bin/env python3
"""The torch venv the PyTorch reference producers run in.

`TORCH_VENV` names it; the default is `<repo>/.venv-torch-ref`, the one
`finetune_ab.sh` provisions. Run as a script this is the probe of the
`torch-venv` need in `ci/guards.toml`: exit 0 when the venv imports every
package the reference producers use, otherwise exit 1 naming what is missing.
`--graph` probes the `torch-graph-venv` need instead: the packages the
graph-learning reference producers use.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_VENV = Path(os.environ.get("TORCH_VENV", str(REPO_ROOT / ".venv-torch-ref")))
TORCH_PY = TORCH_VENV / "bin" / "python3"
PACKAGES = ("torch", "transformers", "peft", "safetensors")
GRAPH_PACKAGES = ("torch", "torch_geometric", "torch_cluster", "safetensors", "numpy")


def missing(packages: tuple[str, ...] = PACKAGES) -> str | None:
    """`None` when the venv imports `packages`; otherwise what it lacks."""
    if not TORCH_PY.is_file():
        return f"no torch venv at {TORCH_VENV} (set TORCH_VENV): {TORCH_PY} does not exist"
    probe = subprocess.run(
        [str(TORCH_PY), "-c", f"import {', '.join(packages)}"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode != 0:
        return (
            f"the torch venv at {TORCH_VENV} does not import {', '.join(packages)}:\n"
            f"{probe.stderr.strip()}"
        )
    return None


def run(script: Path, *args: str, timeout: int) -> str:
    """The standard output of `script args...` under the venv's interpreter,
    run from the repository root; a failure carries the child's output."""
    done = subprocess.run(
        [str(TORCH_PY), str(script), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if done.returncode != 0:
        raise AssertionError(
            f"{script.name} {' '.join(args)} exited {done.returncode}:\n{done.stdout}{done.stderr}"
        )
    return done.stdout


if __name__ == "__main__":
    if sys.argv[1:] == ["--path"]:
        print(TORCH_VENV)
        sys.exit(0)
    if why := missing(GRAPH_PACKAGES if sys.argv[1:] == ["--graph"] else PACKAGES):
        print(why, file=sys.stderr)
        sys.exit(1)
