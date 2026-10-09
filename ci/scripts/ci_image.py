#!/usr/bin/env python3
"""The tree's CI environment: the images its lanes build and run in, and the
service images they run beside, each named by what defines it.

    python3 ci/scripts/ci_image.py refs              # KEY=VALUE lines, for $GITHUB_OUTPUT
    python3 ci/scripts/ci_image.py describe --dockerfile F            # one image: ref, tag, platforms
    python3 ci/scripts/ci_image.py build-args --dockerfile F --arch A # its build arguments on arch A
    python3 ci/scripts/ci_image.py built --dockerfile F               # build=true|false: the registry lacks its tag
    python3 ci/scripts/ci_image.py check             # every Dockerfile reads only its key's inputs
    python3 ci/scripts/ci_image.py await --repo R --head-sha S IMAGE...

How an image is built is defined here once: `describe` and `build-args` are
what ci.yml's image lanes build from (`ci/lanes.toml`), and what
`ci/dev.sh --build-image` builds from locally. The lanes carry each image's
reference as a literal: `ci/scripts/lanes.py` renders it from `refs`, so a
change to an image's inputs re-renders the workflows that run in it.

An image's tag is `ctx-<hash>` over exactly what defines it: its Dockerfile,
every file it copies, the base it builds FROM (pinned by digest in
`.docker/base-images.env`), the toolchain pin it receives as RUST_VERSION, and
the recipe that builds it, this file, whose `build_args` decide what reaches
the build. The CUDA image builds FROM the
CPU image, so its key folds in the CPU key. A tag therefore names one
environment: the same tree always runs in the same image, and a pull request
that changes an image is tested in the image it defines — never in the one
`main` had.

`ci.yml` builds a tree's images when the registry lacks them (`built` says
whether a tag exists). Every other workflow starts beside it on the
same push and `await`s them: it polls the registry until each image exists,
and stops early, naming the failed job, when `ci.yml`'s build for the same head
commit fails.

`check` holds the key's closure — a Dockerfile that copies a file, or declares
a build argument, outside its image's inputs would make two different images
share one tag — and the service images: every deploy file names a
service by the name and tag of the digest `ci/service-images.env` pins, so a
smoke that pulls the pin (`ci/scripts/pull_service_images.sh`) runs it under
the deploy file's own name.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ci" / "scripts"))
import github_api  # noqa: E402
from github_api import API_BASE, ApiError, FetchFn  # noqa: E402

REGISTRY = "ghcr.io/f-inverse/jammi-ai"
BASE_IMAGES = ".docker/base-images.env"
SERVICE_IMAGES = "ci/service-images.env"
DOCKER_CONTEXT = ".docker"
# The build arguments a CI Dockerfile may declare without a default: each is
# supplied from a key input (BASE_IMAGE from `.docker/base-images.env` or the
# CPU image, RUST_VERSION from `rust-toolchain.toml`) or is buildx's own
# per-platform TARGETARCH.
KEYED_BUILD_ARGS = frozenset({"BASE_IMAGE", "RUST_VERSION", "TARGETARCH"})
# The prefix of `ci.yml`'s image-building jobs' names (`ci/lanes.toml` holds
# every builder to it); `await` reads their conclusions to stop early when a
# build fails.
BUILDER_JOB_PREFIX = "CI image ("
BUILDER_WORKFLOW = "ci.yml"
# The deploy files that run a service image: each `image:` they name must be
# the name and tag of a pinned reference.
DEPLOY_SERVICE_FILES = ("deploy/docker-compose.yml", "deploy/kubernetes/overlays/ci/postgres.yaml")


@dataclass(frozen=True)
class Image:
    name: str
    suffix: str
    dockerfile: str
    platforms: tuple[str, ...]
    inputs: tuple[str, ...]
    base: str | None  # the image this one builds FROM, when it is one of ours


# The recipe every image is built by: an edit to it changes every tag.
RECIPE = ("ci/scripts/ci_image.py",)


IMAGES = {
    "cpu": Image(
        name="cpu",
        suffix="ci",
        dockerfile=".docker/ci.Dockerfile",
        platforms=("linux/amd64", "linux/arm64"),
        inputs=(*RECIPE, ".docker/ci.Dockerfile", ".docker/pinned-tools.sh", BASE_IMAGES, "rust-toolchain.toml"),
        base=None,
    ),
    "cuda": Image(
        name="cuda",
        suffix="ci-cuda",
        dockerfile=".docker/ci-cuda.Dockerfile",
        platforms=("linux/amd64",),
        inputs=(".docker/ci-cuda.Dockerfile",),
        base="cpu",
    ),
}


def parse_env(path: Path) -> dict[str, str]:
    """`KEY=VALUE` lines; `#` comments and blank lines skipped."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"{path}: not a KEY=VALUE line: {line!r}")
        out[key] = value
    return out


def content_key(image: Image, root: Path = REPO_ROOT) -> str:
    """sha256 over each input's path and bytes, after the base image's key."""
    digest = hashlib.sha256()
    if image.base is not None:
        digest.update(content_key(IMAGES[image.base], root).encode() + b"\0")
    for rel in image.inputs:
        digest.update(rel.encode() + b"\0" + (root / rel).read_bytes() + b"\0")
    return digest.hexdigest()


def tag(image: Image, root: Path = REPO_ROOT) -> str:
    return f"ctx-{content_key(image, root)[:16]}"


def ref(image: Image, root: Path = REPO_ROOT) -> str:
    return f"{REGISTRY}-{image.suffix}:{tag(image, root)}"


def image_for(dockerfile: str) -> Image:
    """The image `dockerfile` defines."""
    for image in IMAGES.values():
        if image.dockerfile == dockerfile:
            return image
    raise ValueError(f"{dockerfile} defines no CI image (ci_image.IMAGES)")


def describe(image: Image, root: Path = REPO_ROOT) -> dict[str, str]:
    return {"ref": ref(image, root), "tag": tag(image, root), "platforms": ",".join(image.platforms)}


def build_args(image: Image, arch: str, root: Path = REPO_ROOT) -> dict[str, str]:
    """The build arguments `image` takes on `arch` (`amd64`/`arm64`): every
    one a key input supplies (KEYED_BUILD_ARGS, less buildx's own
    TARGETARCH)."""
    if f"linux/{arch}" not in image.platforms:
        raise ValueError(f"the {image.name} image is not built for {arch} (platforms: {', '.join(image.platforms)})")
    if image.base is None:
        base = parse_env(root / BASE_IMAGES)[f"MANYLINUX_{arch.upper()}"]
    else:
        base = ref(IMAGES[image.base], root)
    channel = subprocess.run(
        ["bash", "ci/scripts/rust_pin.sh"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    return {"BASE_IMAGE": base, "RUST_VERSION": channel}


def refs(root: Path = REPO_ROOT) -> dict[str, str]:
    """Every reference the tree's lanes run in or beside: the two CI images,
    and every pinned service image by its lower-cased key."""
    services = {key.lower(): value for key, value in parse_env(root / SERVICE_IMAGES).items()}
    return {"cpu": ref(IMAGES["cpu"], root), "cuda": ref(IMAGES["cuda"], root), **services}


_COPY_RE = re.compile(r"^\s*(COPY|ADD)\s+(?P<rest>.+)$", re.IGNORECASE)
_ARG_RE = re.compile(r"^\s*ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<default>=.*)?$", re.IGNORECASE)


_IMAGE_LINE_RE = re.compile(r"^\s*(?:-\s*)?image:\s*(?P<ref>\S+)\s*$")


def service_names(root: Path = REPO_ROOT) -> dict[str, str]:
    """`{name:tag: pinned ref}` for every pinned service image."""
    pins = parse_env(root / SERVICE_IMAGES).values()
    return {pin.split("@", 1)[0]: pin for pin in pins}


def deploy_service_findings(root: Path = REPO_ROOT) -> list[str]:
    """Every `image:` a deploy file names that is not the name:tag of a pinned
    service image (the application's own image, named by a variable or a
    registry path under the repository owner, is not a service)."""
    names = service_names(root)
    service_repos = {name.split(":", 1)[0] for name in names}
    findings = []
    for rel in DEPLOY_SERVICE_FILES:
        for n, line in enumerate((root / rel).read_text().splitlines(), 1):
            m = _IMAGE_LINE_RE.match(line)
            if not m:
                continue
            image = m.group("ref")
            repo = image.split("@", 1)[0].rsplit(":", 1)[0]
            if repo not in service_repos:
                continue
            if image not in names:
                findings.append(
                    f"{rel}:{n}: names {image}, not the name and tag of the pinned service image "
                    f"({', '.join(sorted(names))} in {SERVICE_IMAGES})"
                )
    return findings


def check(root: Path = REPO_ROOT) -> list[str]:
    """Findings: a file a Dockerfile copies, or a build argument it declares
    without a default, that its image's key does not cover."""
    findings: list[str] = deploy_service_findings(root)
    for image in IMAGES.values():
        for n, line in enumerate((root / image.dockerfile).read_text().splitlines(), 1):
            where = f"{image.dockerfile}:{n}"
            if m := _COPY_RE.match(line):
                words = m.group("rest").split()
                if any(w.startswith("--from") for w in words):
                    continue  # a stage of the same build, not the context
                sources = [w for w in words if not w.startswith("--")][:-1]
                for src in sources:
                    rel = f"{DOCKER_CONTEXT}/{src}"
                    if rel not in image.inputs:
                        findings.append(
                            f"{where}: copies {src}, which the {image.name} image's key "
                            f"(ci_image.IMAGES[{image.name!r}].inputs) does not cover"
                        )
            elif (m := _ARG_RE.match(line)) and m.group("default") is None:
                if m.group("name") not in KEYED_BUILD_ARGS:
                    findings.append(
                        f"{where}: build argument {m.group('name')} has no default and is not one a key "
                        f"input supplies ({', '.join(sorted(KEYED_BUILD_ARGS))})"
                    )
    return findings


ExistsFn = Callable[[str], tuple[bool, str]]


def built(image_ref: str, exists: "ExistsFn", out=sys.stdout, err=sys.stderr) -> int:
    """`build=false` when the registry holds `image_ref`, `build=true` when it
    reports the tag missing; any other answer fails, naming it."""
    found, error = exists(image_ref)
    if found:
        print("build=false", file=out)
        print(f"{image_ref} is built", file=err)
        return 0
    if "not found" in error:
        print("build=true", file=out)
        print(f"{image_ref} is not built yet", file=err)
        return 0
    print(f"::error::reading {image_ref} failed: {error}", file=err)
    return 1


def registry_has(image_ref: str) -> tuple[bool, str]:
    """Whether the registry holds `image_ref`, and the inspector's error when
    it does not."""
    proc = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image_ref],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0, proc.stderr.strip()


def failed_builder(fetch: FetchFn, token: str, repo: str, head_sha: str) -> str | None:
    """The first of `ci.yml`'s image-building jobs for `head_sha` that ended
    without success, as `<name> (<url>)`; `None` while every one is running,
    green, or skipped."""
    url = f"{API_BASE}/repos/{repo}/actions/workflows/{BUILDER_WORKFLOW}/runs?head_sha={head_sha}"
    for run in github_api.paginated(fetch, token, url, "workflow_runs"):
        if run.get("head_sha") != head_sha:
            continue
        for job in github_api.list_jobs(fetch, token, repo, run.get("id")):
            if not str(job.get("name", "")).startswith(BUILDER_JOB_PREFIX):
                continue
            if job.get("conclusion") in ("failure", "cancelled", "timed_out"):
                return f"{job.get('name')} ({job.get('html_url', '')})"
    return None


def await_images(
    image_refs: list[str],
    *,
    exists: ExistsFn,
    builder: Callable[[], str | None],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
    deadline_s: float = 45 * 60,
    poll_s: float = 20,
    err=sys.stderr,
) -> int:
    """Poll until every ref exists. Exit 1 at once when the build fails, and
    at the deadline naming what is still missing and why."""
    start = clock()
    missing = list(image_refs)
    last_error = ""
    while True:
        still = []
        for image_ref in missing:
            found, error = exists(image_ref)
            if not found:
                still.append(image_ref)
                last_error = error or last_error
        missing = still
        if not missing:
            return 0
        try:
            failed = builder()
        except ApiError as e:
            print(f"::warning::ci-image: cannot read ci.yml's image build ({e}); waiting on the registry", file=err)
            failed = None
        if failed is not None:
            print(f"::error::ci-image: ci.yml's image build failed: {failed} -- {', '.join(missing)} will not appear", file=err)
            return 1
        if clock() - start >= deadline_s:
            print(
                f"::error::ci-image: {', '.join(missing)} did not appear in {deadline_s / 60:.0f} min; "
                f"ci.yml builds a tree's images -- is its run for this commit running? last registry answer: "
                f"{last_error or 'not found'}",
                file=err,
            )
            return 1
        sleep(poll_s)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("refs")
    sub.add_parser("check")
    de = sub.add_parser("describe")
    de.add_argument("--dockerfile", required=True)
    ba = sub.add_parser("build-args")
    ba.add_argument("--dockerfile", required=True)
    ba.add_argument("--arch", required=True, choices=["amd64", "arm64"])
    bu = sub.add_parser("built")
    bu.add_argument("--dockerfile", required=True)
    aw = sub.add_parser("await")
    aw.add_argument("--repo", required=True)
    aw.add_argument("--head-sha", required=True, help="The head commit ci.yml's run for this push tests.")
    aw.add_argument("images", nargs="+", choices=sorted(IMAGES))
    args = ap.parse_args(argv)

    if args.command == "refs":
        for key, value in refs().items():
            print(f"{key}={value}")
        return 0
    if args.command == "describe":
        for key, value in describe(image_for(args.dockerfile)).items():
            print(f"{key}={value}")
        return 0
    if args.command == "build-args":
        for key, value in build_args(image_for(args.dockerfile), args.arch).items():
            print(f"{key}={value}")
        return 0
    if args.command == "built":
        return built(ref(image_for(args.dockerfile)), registry_has)
    if args.command == "check":
        findings = check()
        for f in findings:
            print(f"::error::ci-image: {f}", file=sys.stderr)
        return 1 if findings else 0
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("::error::ci-image: GITHUB_TOKEN is not set", file=sys.stderr)
        return 1
    return await_images(
        [ref(IMAGES[name]) for name in args.images],
        exists=registry_has,
        builder=lambda: failed_builder(github_api.default_fetch, token, args.repo, args.head_sha),
        sleep=time.sleep,
        clock=time.monotonic,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
