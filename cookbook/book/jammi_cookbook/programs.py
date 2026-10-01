"""The programs chapters run on a surface other than Python, and the toolchains
they run under.

Each program is a directory under ``_programs/``: ``one_program`` — register a
source, embed it, search it — as a Rust binary (``main.rs``) and a TypeScript
script (``search.ts``), the surfaces chapter's; ``monitor_inference`` — an
inference observer watching an embedding run — as a Rust binary, the operations
chapter's. Each helper here sets a surface up the way its user would, and says
which it did:

* in a working checkout (the book renders from one), a surface is built from
  the checkout's own sources — the Rust program against the workspace crates,
  the TypeScript program against ``clients/typescript``, the CLI from
  ``jammi-cli`` — so the chapter measures the code beside it;
* anywhere else (a fresh Colab runtime), from what that release published —
  the crates on crates.io, the client on npm, the ``jammi`` binary on the
  GitHub release — installing a missing toolchain (Rust, Node) the way a user
  would.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from . import fixtures

GITHUB = "f-inverse/jammi-ai"
RUST_TOOLCHAIN = "1.94.0"
NODE_VERSION = "v24.21.0"
PROGRAMS = Path(__file__).resolve().parent / "_programs"
CACHE = Path(os.environ.get("JAMMI_COOKBOOK_CACHE", Path.home() / ".cache" / "jammi-cookbook"))


def program(name: str) -> Path:
    """The directory holding the program `name`'s sources."""
    return PROGRAMS / name


def checkout() -> Path | None:
    """The repository root when the fixtures resolve into a working checkout."""
    root = fixtures.path("tiny_corpus.parquet").resolve().parents[2]
    return root if (root / "Cargo.toml").exists() and (root / "crates").is_dir() else None


def _run(args: list[str], cwd: Path | None = None, env: dict | None = None) -> str:
    done = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed ({done.returncode}):\n{done.stderr[-4000:]}")
    return done.stdout


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def cli(version: str) -> str:
    """The ``jammi`` binary: built from a checkout's ``jammi-cli``, else the
    release tarball for this platform."""
    root = checkout()
    if root is not None:
        _run([cargo(), "build", "--release", "-q", "-p", "jammi-cli", "--bin", "jammi"], cwd=root)
        metadata = json.loads(
            _run([cargo(), "metadata", "--format-version", "1", "--no-deps"], cwd=root)
        )
        return str(Path(metadata["target_directory"]) / "release" / "jammi")
    triple = {
        ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
        ("Darwin", "arm64"): "aarch64-apple-darwin",
    }[(platform.system(), platform.machine())]
    name = f"jammi-{version}-{triple}.tar.gz"
    archive = _fetch(f"https://github.com/{GITHUB}/releases/download/v{version}/{name}",
                     CACHE / "bin" / name)
    target = CACHE / "bin" / version
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    return str(target / "jammi")


def cargo() -> str:
    """``cargo``: the one on ``PATH``, else a rustup-installed toolchain."""
    found = shutil.which("cargo")
    if found:
        return found
    home = Path.home() / ".cargo" / "bin"
    if not (home / "cargo").exists():
        script = _fetch("https://sh.rustup.rs", CACHE / "rustup-init.sh")
        _run(["sh", str(script), "-y", "--profile", "minimal",
              "--default-toolchain", RUST_TOOLCHAIN])
    return str(home / "cargo")


def rust_program(workdir: Path, version: str, name: str) -> Path:
    """A Cargo project for the Rust program `name` under ``workdir``: path
    dependencies on a checkout's crates (with its lockfile), else the published
    crates."""
    project = workdir / name
    (project / "src").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(program(name) / "main.rs", project / "src" / "main.rs")
    root = checkout()
    if root is not None:
        engine = {c: f'{{ path = "{root / "crates" / c}" }}' for c in ("jammi-ai", "jammi-db")}
        shutil.copyfile(root / "Cargo.lock", project / "Cargo.lock")
    else:
        engine = {c: f'"={version}"' for c in ("jammi-ai", "jammi-db")}
    (project / "Cargo.toml").write_text(
        f"""[package]
name = "{name}"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
jammi-ai = {engine["jammi-ai"]}
jammi-db = {engine["jammi-db"]}
arrow = "58"
tokio = {{ version = "1", features = ["macros", "rt-multi-thread"] }}

# gemm-f16's FP16 kernels do not assemble at opt-level 0 on aarch64 Linux.
[profile.dev.package.gemm-f16]
opt-level = 1

[workspace]
"""
    )
    return project


def node() -> Path:
    """The directory holding ``node`` and ``npm``: the ones on ``PATH``, else a
    Node release unpacked into the cache."""
    found = shutil.which("node")
    if found:
        return Path(found).parent
    arch = {"x86_64": "x64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine()]
    system = {"Linux": "linux", "Darwin": "darwin"}[platform.system()]
    stem = f"node-{NODE_VERSION}-{system}-{arch}"
    archive = _fetch(f"https://nodejs.org/dist/{NODE_VERSION}/{stem}.tar.gz",
                     CACHE / "node" / f"{stem}.tar.gz")
    with tarfile.open(archive) as tar:
        tar.extractall(CACHE / "node", filter="data")
    return CACHE / "node" / stem / "bin"


def ts_program(workdir: Path, version: str) -> tuple[Path, dict]:
    """An npm project for the TypeScript program under ``workdir``, installed,
    and the environment that runs it: the client built from a checkout's
    ``clients/typescript``, else the published package."""
    bin_dir = node()
    env = dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    npm = str(bin_dir / "npm")
    root = checkout()
    if root is not None:
        client_dir = root / "clients" / "typescript"
        _run([npm, "ci", "--no-audit", "--no-fund"], cwd=client_dir, env=env)
        _run([npm, "run", "build"], cwd=client_dir, env=env)
        client = f"file:{client_dir}"
    else:
        client = version
    project = workdir / "one_program_ts"
    project.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(program("one_program") / "search.ts", project / "search.ts")
    (project / "package.json").write_text(
        f"""{{
  "name": "one-program",
  "private": true,
  "type": "module",
  "dependencies": {{
    "@f-inverse/jammi-client": "{client}",
    "apache-arrow": "^21.2.0",
    "tsx": "^4.23.0"
  }}
}}
"""
    )
    _run([npm, "install", "--no-audit", "--no-fund"], cwd=project, env=env)
    return project, env
