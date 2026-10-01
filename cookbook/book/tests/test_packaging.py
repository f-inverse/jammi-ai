"""The wheel a notebook's install builds from the release tag carries every
fixture the recipes and chapters read."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

from jammi_cookbook import encoders, fixtures
from jammi_cookbook.scale import Scale

BOOK = Path(__file__).resolve().parents[1]
COOKBOOK = BOOK.parent
_LITERAL = re.compile(r"fixtures\.(?:path|url|model)\(\s*f?[\"']([^\"'/{]+)")
# A path climbed out of a module's own directory: in an install that is the
# environment's site-packages, not the repository the module was built from.
_CLIMB = re.compile(r"__file__\)(?:\.resolve\(\))?\.(?:parents\[|parent\.parent)")


def _sources() -> list[Path]:
    """Every recipe, chapter and library module a reader's install runs."""
    return [*COOKBOOK.glob("recipes/*/example.py"), *COOKBOOK.glob("quickstart/*.py"),
            *BOOK.glob("chapters/**/*.qmd"), *BOOK.glob("jammi_cookbook/*.py")]


def _read() -> set[str]:
    """The top-level fixture names every recipe, chapter and library module
    reads — by literal name, and through the `small` encoders."""
    names = {m for src in _sources() for m in _LITERAL.findall(src.read_text())}
    root = fixtures.path(".").resolve()
    for encoder in (encoders.text, encoders.image, encoders.audio):
        local = Path(encoder(Scale.SMALL).removeprefix("local:")).resolve()
        names.add(local.relative_to(root).parts[0])
    return names


def _packaged(patterns: list[str]) -> set[Path]:
    """The files ``patterns`` select, globbed from the package directory as
    setuptools globs its package data — a ``*`` never crosses a ``/``."""
    package = BOOK / "jammi_cookbook"
    return {f.resolve() for p in patterns for f in package.glob(p) if f.is_file()}


def _unshipped(name: str, shipped: set[Path], excluded: set[Path]) -> list[str]:
    """The files of fixture ``name`` — the file itself, or every file under the
    directory — that the wheel neither ships nor deliberately excludes."""
    target = fixtures.path(name)
    files = [target] if target.is_file() else [f for f in target.rglob("*") if f.is_file()]
    root = fixtures.path(".").resolve()
    return [f.resolve().relative_to(root).as_posix() for f in files
            if f.resolve() not in shipped | excluded]


def test_every_fixture_the_cookbook_reads_ships_in_the_wheel():
    setuptools = tomllib.loads((BOOK / "pyproject.toml").read_text())["tool"]["setuptools"]
    excluded = _packaged(setuptools["exclude-package-data"]["jammi_cookbook"])
    shipped = _packaged(setuptools["package-data"]["jammi_cookbook"]) - excluded
    read = _read()
    assert {"tiny_bert", "tiny_open_clip", "htsat_clap_tiny", "arxiv_small"} <= read
    missing = sorted(f for n in read for f in _unshipped(n, shipped, excluded))
    assert not missing, f"fixture files the cookbook reads but the wheel omits: {missing}"


def test_nothing_the_cookbook_runs_reads_outside_its_installed_package():
    """A file is reached through the package (``fixtures.path``, a module's own
    directory), never by climbing from a module to the repository around it: a
    reader's runtime has the installed package and no checkout."""
    climbing = sorted(
        f"{src.relative_to(COOKBOOK)}:{n}"
        for src in _sources()
        for n, line in enumerate(src.read_text().splitlines(), 1)
        if _CLIMB.search(line)
    )
    assert not climbing, f"paths resolved outside the installed package: {climbing}"
