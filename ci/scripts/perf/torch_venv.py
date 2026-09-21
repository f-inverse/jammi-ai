#!/usr/bin/env python3
"""The torch venv the PyTorch reference producers run in.

`TORCH_VENV` names it; the default is `<repo>/.venv-torch-ref`. This module
is the one place the venv is resolved, probed and provisioned:

    torch_venv.py              probe: exit 0 when the venv imports every package
                               the reference producers use, otherwise exit 1
                               naming what is missing (the `torch-venv` need in
                               `ci/guards.toml`)
    torch_venv.py --path       print the venv's path
    torch_venv.py --provision  make the venv usable, or exit 1 naming why not

The venv is built from the interpreter running this script -- the host's own,
the one the operator chose -- never from one a tool downloads on its own
initiative: a managed interpreter newer than the packages' published wheels
cannot install them, and nothing would have named it. A leg run in the venv
records that interpreter's version in its provenance (`python_version`).
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_VENV = Path(os.environ.get("TORCH_VENV", str(REPO_ROOT / ".venv-torch-ref")))
TORCH_PY = TORCH_VENV / "bin" / "python3"
PACKAGES = ("torch", "transformers", "peft", "safetensors")
REQUIREMENTS = ("torch", "transformers>=4.48", "peft", "safetensors")


def missing() -> str | None:
    """`None` when the venv is usable; otherwise what it lacks."""
    if not TORCH_PY.is_file():
        return f"no torch venv at {TORCH_VENV} (set TORCH_VENV): {TORCH_PY} does not exist"
    probe = subprocess.run(
        [str(TORCH_PY), "-c", f"import {', '.join(PACKAGES)}"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if probe.returncode != 0:
        return (
            f"the torch venv at {TORCH_VENV} does not import {', '.join(PACKAGES)}:\n"
            f"{probe.stderr.strip()}"
        )
    return None


def interpreter() -> str:
    """The interpreter a venv is built from, named the way a refusal needs."""
    return f"{sys.executable} ({sys.implementation.name} {sys.version.split()[0]})"


def _pip_install(python: Path, requirements: tuple[str, ...]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(python), "-m", "pip", "install", *requirements],
        capture_output=True,
        text=True,
        check=False,
    )


def provision(install=_pip_install) -> str | None:
    """Make the venv usable: reuse it when it already is, otherwise build it
    from this interpreter and install `REQUIREMENTS`. `None` on success;
    otherwise the refusal, naming the interpreter the packages could not be
    installed for."""
    if missing() is None:
        return None
    venv.EnvBuilder(with_pip=True, symlinks=True).create(TORCH_VENV)
    done = install(TORCH_PY, REQUIREMENTS)
    if done.returncode != 0:
        tail = "\n".join((done.stderr or done.stdout).strip().splitlines()[-5:])
        return (
            f"cannot install {', '.join(REQUIREMENTS)} for {interpreter()}:\n{tail}\n"
            "run this under an interpreter those packages publish wheels for"
        )
    return missing()


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
    if sys.argv[1:] == ["--provision"]:
        if why := provision():
            print(why, file=sys.stderr)
            sys.exit(1)
        print(f"torch venv at {TORCH_VENV}, built from {interpreter()}")
        sys.exit(0)
    if why := missing():
        print(why, file=sys.stderr)
        sys.exit(1)
