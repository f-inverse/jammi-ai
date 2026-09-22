#!/usr/bin/env python3
"""The torch venv the PyTorch reference producers run in.

`TORCH_VENV` names it; the default is `<repo>/.venv-torch-ref`. This module
is the one place the venv is resolved, probed and provisioned:

    torch_venv.py              probe: exit 0 when the venv imports every package
                               the reference producers use, otherwise exit 1
                               naming what is missing (the `torch-venv` need in
                               `ci/guards.toml`)
    torch_venv.py --graph      the same probe over the graph-learning
                               producers' packages (`torch-graph-venv`)
    torch_venv.py --path       print the venv's path
    torch_venv.py --provision  make the venv usable, or exit 1 naming why not
    torch_venv.py --preflight  one real forward and backward of a tiny model
                               on CUDA device 0 through the reference
                               producer, or exit 1 naming why it cannot run

The venv is built from the interpreter running this script -- the host's own,
the one the operator chose -- never from one a tool downloads on its own
initiative: a managed interpreter newer than the packages' published wheels
cannot install them, and nothing would have named it. A leg run in the venv
records that interpreter's version in its provenance (`python_version`).

On a box with an NVIDIA driver the torch wheel is chosen by the DRIVER: the
newest PyTorch wheel index whose CUDA version the driver supports, read from
`nvidia-smi`. A wheel built for a newer CUDA than the driver offers imports
cleanly, reports `torch.cuda.is_available() == False`, and would run every
reference leg on the CPU; so on such a box a venv is usable only if its torch
can see the device, and provisioning ends with the preflight — a tiny model's
forward and backward on the device, which also exercises whatever the
framework JIT-compiles on first use (Triton's C shim needs the interpreter's
development headers).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_VENV = Path(os.environ.get("TORCH_VENV", str(REPO_ROOT / ".venv-torch-ref")))
TORCH_PY = TORCH_VENV / "bin" / "python3"
PACKAGES = ("torch", "transformers", "peft", "safetensors", "pyarrow", "usearch")
TORCH_REQUIREMENT = "torch"
REQUIREMENTS = ("transformers>=4.48", "peft", "safetensors", "pyarrow", "usearch")
GRAPH_PACKAGES = ("torch", "torch_geometric", "torch_cluster", "safetensors", "numpy")
REFERENCE_STEP = REPO_ROOT / "crates" / "jammi-bench" / "reference" / "torch_finetune_step.py"

# PyTorch's CUDA wheel indexes, newest first: `(CUDA version, index name)`.
WHEEL_INDEXES = (((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"), ((11, 8), "cu118"))
WHEEL_INDEX_URL = "https://download.pytorch.org/whl/{index}"


def _nvidia_smi() -> str | None:
    try:
        done = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def driver_cuda_version(nvidia_smi=_nvidia_smi) -> tuple[int, int] | None:
    """The newest CUDA version this box's driver supports, from `nvidia-smi`'s
    banner; `None` on a box with no NVIDIA driver."""
    banner = nvidia_smi()
    if banner is None:
        return None
    found = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", banner)
    return (int(found.group(1)), int(found.group(2))) if found else None


def wheel_index(driver: tuple[int, int]) -> str | None:
    """The newest wheel index the driver supports; `None` when it supports
    none of them."""
    return next((name for version, name in WHEEL_INDEXES if version <= driver), None)


def missing(driver=driver_cuda_version) -> str | None:
    """`None` when the venv is usable; otherwise what it lacks. On a box with
    an NVIDIA driver, a torch that cannot see the device is not usable."""
    return _unimportable() or _cannot_see_device(driver)


def missing_for_graphs(driver=driver_cuda_version) -> str | None:
    """[`missing`] for the graph-learning reference producers, whose packages
    (`torch-graph-venv` in `ci/guards.toml`) are `GRAPH_PACKAGES` and no
    other: `torch_cluster` and `pyg-lib` resolve against the installed torch,
    so they follow it into the venv."""
    return _unimportable(GRAPH_PACKAGES) or _cannot_see_device(driver)


def _cannot_see_device(driver) -> str | None:
    """Why the venv's torch cannot use this box's GPU, on a box that has one."""
    version = driver()
    if version is None:
        return None
    probe = subprocess.run(
        [str(TORCH_PY), "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    seen = probe.stdout.split()
    if probe.returncode != 0 or len(seen) != 3 or seen[2] != "True":
        built_for = seen[1] if len(seen) == 3 else "unknown"
        return (
            f"the torch venv at {TORCH_VENV} cannot use this box's GPU: its torch is built for "
            f"CUDA {built_for}, the driver supports CUDA {version[0]}.{version[1]} "
            f"(wheel index {wheel_index(version)}), and torch.cuda.is_available() is False — "
            "every reference leg would run on the CPU"
        )
    return None


def _unimportable(packages: tuple[str, ...] | None = None) -> str | None:
    """What the venv cannot import of `packages` — `PACKAGES` as the module
    holds it at call time when none are given."""
    packages = packages or PACKAGES
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


def interpreter() -> str:
    """The interpreter a venv is built from, named the way a refusal needs."""
    return f"{sys.executable} ({sys.implementation.name} {sys.version.split()[0]})"


def _pip_install(python: Path, requirements: tuple[str, ...], index: str | None = None) -> subprocess.CompletedProcess:
    source = ["--index-url", WHEEL_INDEX_URL.format(index=index), "--force-reinstall"] if index else []
    return subprocess.run(
        [str(python), "-m", "pip", "install", *source, *requirements],
        capture_output=True,
        text=True,
        check=False,
    )


def preflight(run_step=None) -> str | None:
    """One real forward and backward of a tiny model on CUDA device 0, through
    the reference producer itself. `None` when it ran; otherwise why it could
    not, by name."""
    run_step = run_step or (
        lambda: subprocess.run(
            [str(TORCH_PY), str(REFERENCE_STEP), "--dry-run", "--cuda", "0",
             "--batch", "2", "--seq", "8", "--steps", "1", "--warmup", "0"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800, check=False,
        )
    )
    done = run_step()
    if done.returncode == 0:
        return None
    tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-8:])
    hint = (
        "\nthe framework's JIT compiles a C shim on first use and needs the interpreter's "
        "development headers (Python.h): install the python3-devel package matching "
        f"{interpreter()}"
        if "Python.h" in (done.stderr or "")
        else ""
    )
    return f"the reference step cannot run a forward and backward on cuda:0:\n{tail}{hint}"


def provision(install=_pip_install, driver=driver_cuda_version, check=preflight) -> str | None:
    """Make the venv usable: reuse it when it already is, otherwise build it
    from this interpreter, install torch from the wheel index the driver
    supports (the default index on a box with no driver) and the rest of
    `REQUIREMENTS`, and on a GPU box run the preflight. `None` on success;
    otherwise the refusal, by name."""
    version = driver()
    if missing(driver) is not None:
        index = None
        if version is not None:
            index = wheel_index(version)
            if index is None:
                return (
                    f"this box's driver supports CUDA {version[0]}.{version[1]}, older than every "
                    f"PyTorch wheel index ({', '.join(name for _, name in WHEEL_INDEXES)}): update the driver"
                )
        venv.EnvBuilder(with_pip=True, symlinks=True).create(TORCH_VENV)
        for requirements, source in (((TORCH_REQUIREMENT,), index), (REQUIREMENTS, None)):
            done = install(TORCH_PY, requirements, source)
            if done.returncode != 0:
                tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-5:])
                return (
                    f"cannot install {', '.join(requirements)} for {interpreter()}:\n{tail}\n"
                    "run this under an interpreter those packages publish wheels for"
                )
        if why := missing(driver):
            return why
    return check() if version is not None else None


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
    if sys.argv[1:] == ["--preflight"]:
        if why := preflight():
            print(why, file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if sys.argv[1:] == ["--provision"]:
        if why := provision():
            print(why, file=sys.stderr)
            sys.exit(1)
        print(f"torch venv at {TORCH_VENV}, built from {interpreter()}")
        sys.exit(0)
    if sys.argv[1:] == ["--graph"]:
        if why := missing_for_graphs():
            print(why, file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
    if why := missing():
        print(why, file=sys.stderr)
        sys.exit(1)
