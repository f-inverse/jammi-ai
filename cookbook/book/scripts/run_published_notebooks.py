#!/usr/bin/env python3
"""Run the published cookbook notebooks the way a reader runs them.

A reader opens a notebook's Colab link at a release tag and runs it top to
bottom in a fresh runtime: the setup cell installs that release from PyPI and
the cookbook from the tag, and every later cell runs against what it installed.
The book's own gate runs the chapter sources against wheels built from HEAD, so
it cannot see a release whose wheel never reached PyPI, an install line that no
longer resolves, or a dependency release that breaks a notebook after the fact.
This runs the notebooks exactly as they were tagged, each in what a fresh
runtime gives it: an empty working directory and a new virtual environment
holding only the notebook kernel, with that environment's `bin` on `PATH` (where
a pip-installed `jammi-server` lands). A notebook passes when every cell runs;
its cells check the claims the chapter makes on that run.

Run against a checkout of the release tag:

    python3 run_published_notebooks.py --notebooks <tag-checkout>/cookbook/notebooks \
        [--shard 2/8] [--work DIR]

The interpreter that runs this script is the one each notebook's environment is
created from. Exits non-zero when any notebook fails, naming each one.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# What a notebook host provides before the first cell runs. Colab ships both.
KERNEL_INSTALL = (
    "-m",
    "pip",
    "install",
    "-q",
    "--disable-pip-version-check",
    "nbclient>=0.9",
    "ipykernel>=6.29",
)


@dataclass(frozen=True)
class Shard:
    """The `index`-th of `count` round-robin slices of the sorted notebooks (1-based)."""

    index: int
    count: int

    @classmethod
    def parse(cls, text: str) -> Shard:
        index, _, count = text.partition("/")
        shard = cls(int(index), int(count))
        if not 1 <= shard.index <= shard.count:
            raise argparse.ArgumentTypeError(f"shard {text!r} is not i/n with 1 <= i <= n")
        return shard

    def __str__(self) -> str:
        return f"{self.index}/{self.count}"

    def select(self, notebooks: list[Path]) -> list[Path]:
        return notebooks[self.index - 1 :: self.count]


@dataclass(frozen=True)
class Outcome:
    notebook: Path
    seconds: float
    error: str | None


def notebooks_under(root: Path) -> list[Path]:
    """Every notebook under `root`, relative to it, in a stable order."""
    return sorted(p.relative_to(root) for p in root.rglob("*.ipynb"))


def execute(notebook: Path, cwd: Path, cell_timeout: int) -> None:
    """Run every cell of `notebook` with `cwd` as the kernel's working directory.

    Runs inside the notebook's own environment, where nbclient is installed; the
    kernel it starts is that environment's interpreter.
    """
    import nbclient
    import nbformat

    nb = nbformat.read(notebook, as_version=4)
    nbclient.NotebookClient(
        nb, timeout=cell_timeout, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}
    ).execute()


def run_one(root: Path, notebook: Path, work: Path, cell_timeout: int) -> Outcome:
    """Run `notebook` in a fresh environment, working directory and temporary
    directory under `work`, and remove all three once it ends, as a runtime is
    discarded: a CUDA environment alone holds gigabytes, so keeping each one
    fills the host before the last notebook runs."""
    home = work / notebook.with_suffix("").as_posix().replace("/", "__")
    env_dir, cwd, tmp = home / "venv", home / "content", home / "tmp"
    cwd.mkdir(parents=True)
    tmp.mkdir()
    started = time.monotonic()
    try:
        subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
        python = env_dir / "bin" / "python"
        subprocess.run([str(python), *KERNEL_INSTALL], check=True)
        env = {
            **os.environ,
            "PATH": f"{env_dir / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "VIRTUAL_ENV": str(env_dir),
            "TMPDIR": str(tmp),
        }
        run = subprocess.run(
            [
                str(python),
                __file__,
                "--execute",
                str(root / notebook),
                "--cwd",
                str(cwd),
                "--cell-timeout",
                str(cell_timeout),
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        error = None if run.returncode == 0 else (run.stderr or run.stdout).strip()
    except subprocess.CalledProcessError as exc:
        error = f"environment setup failed: {exc}"
    finally:
        shutil.rmtree(home)
    return Outcome(notebook, time.monotonic() - started, error)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebooks", type=Path, help="the tag's cookbook/notebooks directory")
    parser.add_argument("--shard", type=Shard.parse, default=Shard(1, 1))
    parser.add_argument("--work", type=Path, help="where each notebook's environment is created")
    parser.add_argument("--cell-timeout", type=int, default=1800)
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cwd", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.execute:
        execute(args.execute, args.cwd, args.cell_timeout)
        return 0
    if args.notebooks is None:
        parser.error("--notebooks is required")

    root = args.notebooks.resolve()
    selected = args.shard.select(notebooks_under(root))
    if not selected:
        print(f"no notebooks under {root} for shard {args.shard}", file=sys.stderr)
        return 1
    work = (args.work or Path(tempfile.mkdtemp(prefix="published-notebooks-"))).resolve()
    print(f"running {len(selected)} notebooks from {root} (shard {args.shard})", flush=True)

    outcomes = []
    for notebook in selected:
        print(f"::group::{notebook}", flush=True)
        outcome = run_one(root, notebook, work, args.cell_timeout)
        if outcome.error:
            print(outcome.error, flush=True)
        print("::endgroup::", flush=True)
        print(
            f"{'ok  ' if outcome.error is None else 'FAIL'} {outcome.seconds:7.1f}s  {notebook}",
            flush=True,
        )
        outcomes.append(outcome)

    failed = [o for o in outcomes if o.error is not None]
    for outcome in failed:
        print(f"::error::{outcome.notebook} failed as published", flush=True)
    print(f"{len(outcomes) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
