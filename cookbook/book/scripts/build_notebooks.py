#!/usr/bin/env python3
"""Build the cookbook's Colab notebooks from the chapters and the recipes.

Every book chapter (``chapters/**/*.qmd``) and every recipe
(``cookbook/quickstart``, ``cookbook/recipes/*``) becomes a notebook under
``cookbook/notebooks/``, opened in Colab straight from GitHub. A notebook's
first cell installs the release it was built for — ``jammi-ai`` with the CUDA
engine (``jammi-ai-native-cu12``) on an sm_80+ GPU runtime and the CPU engine
otherwise, ``jammi-server`` when the chapter starts one, and the cookbook's
library and fixtures from the release's own tag on GitHub (the book is not a
published package). It runs the chapter at ``small`` scale — seconds to minutes,
or for a chapter that fine-tunes an encoder, minutes on a GPU and up to hours on a
CPU; ``full`` — the published data and the larger text encoder, where a keystone
fine-tune alone takes hours on an L4 — is one line to opt into.

A notebook and its Colab link name the workspace version: the notebooks at tag
``py-v<version>`` install exactly that release, so a link never runs a chapter
against an engine it was not written for. Bumping the version therefore
rebuilds every notebook, which ``--check`` enforces.

The conversion is plain Python — no Quarto — so the check is hermetic:

* a ```` ```{python} ```` cell becomes a code cell with its ``#|`` options
  dropped; an ``eval: false`` cell becomes a fenced block in markdown, since
  it is shown, never run;
* prose stays markdown; a link to another chapter points at the published
  book, and a citation ``[@key]`` reads "(Author Year)", with the chapter's
  references listed in a closing cell;
* a recipe's README is its overview, and its script's ``# %%`` cells are its
  steps: a ``# %% [markdown]`` cell becomes markdown (its comment markers
  dropped), a ``# %%`` cell a code cell, in order. The script runs top to
  bottom as a program; the reader runs it a step at a time. The same reading
  of the script is also written as a book chapter under ``chapters/recipes/``,
  so the book holds every recipe and renders it run.

Run ``python scripts/build_notebooks.py`` to rebuild, ``--check`` to fail when
a committed notebook differs from what the sources build.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

BOOK = Path(__file__).resolve().parents[1]
REPO = BOOK.parents[1]
OUT = REPO / "cookbook" / "notebooks"
# Every recipe is also a book chapter, generated here from its script.
RECIPE_CHAPTERS = BOOK / "chapters" / "recipes"
GITHUB = "f-inverse/jammi-ai"
BOOK_URL = "https://f-inverse.github.io/jammi-ai/cookbook"

_CELL = re.compile(r"^```\{python\}\s*\n(.*?)^```\s*$", re.S | re.M)
_OPTION = re.compile(r"^#\s?\|\s*([a-z-]+):\s*(.*?)\s*$")
_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_TITLE = re.compile(r'^title:\s*"?(.*?)"?\s*$', re.M)
_CITATION = re.compile(r"\[(@[\w:.-]+(?:\s*;\s*@[\w:.-]+)*)\]")
_QMD_LINK = re.compile(r"\]\(\s*([^)\s#]+)\.qmd(#[^)\s]*)?\s*\)")
# Lines only an engine started through the client's harness, or a `grpc://`
# target, needs a server for; the render selector classifies the same way.
_NEEDS_SERVER = re.compile(r"\bLiveServer\(|connect\(\s*f?[\"']grpc://")
# The cookbook extras a notebook installs, by the lines only each one serves:
# `cloud`, a local S3-compatible server; `postgres`, a pip-installed Postgres a
# fleet of servers shares as its catalog; `otlp`, the trace protocol an
# in-process collector receives spans with.
_EXTRAS = {
    "cloud": re.compile(r"\bThreadedMotoServer\b"),
    "postgres": re.compile(r"\bpgserver\b"),
    "otlp": re.compile(r"\bopentelemetry\.proto\b"),
}


def extras_of(text: str) -> list[str]:
    """The cookbook extras `text` needs, in `_EXTRAS` order."""
    return [name for name, needs in _EXTRAS.items() if needs.search(text)]


def version() -> str:
    """The lockstep workspace version every dist ships at."""
    section = (REPO / "Cargo.toml").read_text().split("[workspace.package]", 1)[1]
    return re.search(r'^version\s*=\s*"([^"]+)"', section, re.M).group(1)


def colab_url(notebook: Path, release: str) -> str:
    rel = notebook.relative_to(REPO).as_posix()
    return f"https://colab.research.google.com/github/{GITHUB}/blob/py-v{release}/{rel}"


# --------------------------------------------------------------------------- #
# Notebook cells
# --------------------------------------------------------------------------- #


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str, *, hidden: bool = False) -> dict:
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": _lines(text),
    }


def notebook(cells: list[dict]) -> dict:
    """The notebook of ``cells``, each given a stable id by its position."""
    return {
        "cells": [{"id": f"cell-{i}", **cell} for i, cell in enumerate(cells)],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def setup_cell(release: str, *, server: bool, extras: list[str]) -> dict:
    """Install the release this notebook was built for, on the engine the
    runtime can run, and choose the scale."""
    extra = f"[{','.join(extras)}]" if extras else ""
    cookbook = (
        f"jammi-cookbook{extra} @ git+https://github.com/{GITHUB}@py-v{release}"
        "#subdirectory=cookbook/book"
    )
    packages = f'"jammi-ai=={release}", engine + "=={release}", "{cookbook}"'
    if server:
        packages += f', server + "=={release}"'
    return code(
        f"""# Setup: jammi {release} — the CUDA engine on an sm_80+ GPU (L4, A100, …), the
# CPU engine otherwise — and the cookbook's library and fixtures, from the release's
# tag on GitHub. The chapter runs
# at `small` scale, over the committed samples with a compact text encoder: seconds
# to minutes, or for a chapter that fine-tunes an encoder, minutes on a GPU and up
# to hours on a CPU. The models it runs are downloaded from the Hugging Face Hub on
# first use. SCALE = "full" runs it over the published data and a larger text
# encoder instead, on the GPU: the chapters that fine-tune take hours there.
import os
import subprocess
import sys


def compute_capability() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return 0.0
    return float(out[0]) if out else 0.0


gpu = compute_capability() >= 8.0
engine = "jammi-ai-native-cu12" if gpu else "jammi-ai-native"
server = "jammi-server-cu12" if gpu else "jammi-server"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", {packages}], check=True)
SCALE = "small"
os.environ["JAMMI_COOKBOOK_SCALE"] = SCALE
print(f"engine: {{engine}}   scale: {{SCALE}}")"""
    )


def header(title: str, source: str, url: str) -> dict:
    return markdown(
        f"""# {title}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})

Built from [`{source}`](https://github.com/{GITHUB}/blob/main/{source}). Run the setup
cell first; every other cell runs top to bottom."""
    )


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #

_TEX = {
    r"{\v{s}}": "š", r"{\'c}": "ć", r"{\`e}": "è", r'{\"u}': "ü", r"{\'e}": "é",
    r"\&": "&", "--": "–",
}


def _detex(value: str) -> str:
    for tex, char in _TEX.items():
        value = value.replace(tex, char)
    value = re.sub(r"\\url\{([^}]*)\}", r"\1", value)
    return re.sub(r"[{}]", "", " ".join(value.split()))


@dataclass(frozen=True)
class Reference:
    key: str
    fields: dict[str, str]

    @property
    def authors(self) -> list[str]:
        return [a.strip() for a in self.fields.get("author", "").split(" and ") if a.strip()]

    @property
    def short(self) -> str:
        names = [a.split(",")[0] for a in self.authors] or [self.fields.get("title", self.key)]
        who = names[0] if len(names) == 1 else (
            f"{names[0]} & {names[1]}" if len(names) == 2 else f"{names[0]} et al.")
        return f"{who} {self.fields.get('year', '')}".strip()

    @property
    def full(self) -> str:
        venue = next((self.fields[f] for f in ("journal", "booktitle", "publisher",
                                                "howpublished") if f in self.fields), "")
        parts = [", ".join(self.authors), f"({self.fields.get('year', 'n.d.')})",
                 f"*{self.fields.get('title', '')}*", venue, self.fields.get("note", "")]
        return " ".join(p for p in parts if p).rstrip(".") + "."


def references(bib: Path = BOOK / "jammi_cookbook" / "references.bib") -> dict[str, Reference]:
    """Every entry of ``bib``, with each field's braces balanced and TeX undone."""
    text = re.sub(r"^%.*$", "", bib.read_text(), flags=re.M)
    refs: dict[str, Reference] = {}
    for m in re.finditer(r"@\w+\{([^,\s]+),", text):
        fields, i = {}, m.end()
        while True:
            f = re.compile(r"\s*(\w+)\s*=\s*\{").match(text, i)
            if not f:
                break
            depth, j = 1, f.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(text[j], 0)
                j += 1
            fields[f.group(1).lower()] = _detex(text[f.end():j - 1])
            i = j
            comma = re.compile(r"\s*,").match(text, i)
            i = comma.end() if comma else i
        refs[m.group(1)] = Reference(m.group(1), fields)
    return refs


def cite(prose: str, refs: dict[str, Reference], cited: list[str]) -> str:
    """``prose`` with each ``[@a; @b]`` read as "(A 2020; B 2021)", recording
    the keys in first-cited order."""

    def one(m: re.Match) -> str:
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        cited.extend(k for k in keys if k not in cited)
        return "(" + "; ".join(refs[k].short for k in keys) + ")"

    return _CITATION.sub(one, prose)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def chapter(qmd: Path, release: str, refs: dict[str, Reference]) -> tuple[Path, dict]:
    text = qmd.read_text()
    front = _FRONT.match(text)
    title = _TITLE.search(front.group(1)).group(1) if front else qmd.stem
    body = text[front.end():] if front else text
    target = OUT / "book" / qmd.relative_to(BOOK / "chapters").with_suffix(".ipynb")
    page = qmd.relative_to(BOOK).with_suffix("")

    def link(m: re.Match) -> str:
        resolved = (BOOK / page).parent / m.group(1)
        rel = resolved.resolve().relative_to(BOOK.resolve()).as_posix()
        return f"]({BOOK_URL}/{rel}.html{m.group(2) or ''})"

    cells, cited, pos = [], [], 0
    executed = []
    for m in _CELL.finditer(body):
        prose = _QMD_LINK.sub(link, body[pos:m.start()]).strip()
        if prose:
            cells.append(markdown(cite(prose, refs, cited)))
        options = {}
        source = []
        for line in m.group(1).rstrip("\n").split("\n"):
            opt = _OPTION.match(line)
            if opt:
                options[opt.group(1)] = opt.group(2)
            else:
                source.append(line)
        src = "\n".join(source).strip("\n")
        if options.get("eval") == "false":
            cells.append(markdown(f"```python\n{src}\n```"))
        elif src:
            executed.append(src)
            cells.append(code(src, hidden=options.get("echo") == "false"))
        pos = m.end()
    tail = _QMD_LINK.sub(link, body[pos:]).strip()
    if tail:
        cells.append(markdown(cite(tail, refs, cited)))
    if cited:
        cells.append(markdown("## References\n\n" + "\n".join(
            f"- {refs[k].full}" for k in cited)))

    run = "\n".join(executed)
    server = bool(_NEEDS_SERVER.search(run))
    url = colab_url(target, release)
    source = qmd.relative_to(REPO).as_posix()
    return target, notebook([header(title, source, url),
                             setup_cell(release, server=server, extras=extras_of(run)), *cells])


@dataclass(frozen=True)
class Recipe:
    """A recipe script read once: what both its notebook and its book chapter
    are built from."""

    name: str
    title: str
    # The README's overview.
    prose: str
    # The script's steps in order, each ("markdown" | "code", text).
    cells: list[tuple[str, str]]
    source: str
    server: bool
    extras: list[str]


_MD_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")
_RUN_IT = re.compile(r"^## Run it\n.*?(?=^## |\Z)", re.S | re.M)
_H1 = re.compile(r"\A# (.+)\n")


def _absolute_links(markdown_text: str, directory: Path) -> str:
    """Relative links in a recipe's pages, resolved to the files on GitHub."""

    def to_github(m: re.Match) -> str:
        target = (directory / m.group(1)).resolve().relative_to(REPO.resolve()).as_posix()
        return f"](https://github.com/{GITHUB}/blob/main/{target})"

    return _MD_LINK.sub(to_github, markdown_text)


def recipe_pages(script: Path) -> tuple[str, str]:
    """A recipe's title and its overview: the README beside the script, less
    its command-line "Run it" section, since a notebook and a chapter run the
    program themselves."""
    readme = (script.parent / "README.md").read_text()
    head = _H1.match(readme)
    if head is None:
        raise ValueError(f"{script.parent / 'README.md'}: a recipe's README opens with its title")
    overview = _RUN_IT.sub("", readme[head.end():]).strip()
    return head.group(1).strip(), _absolute_links(overview, script.parent)


# A cell marker of the percent format: `# %%` opens a code cell, `# %% [markdown]`
# a markdown cell whose every line is a comment.
_CELL_MARK = re.compile(r"^# %%(?P<markdown> \[markdown\])?[ \t]*$", re.M)


def recipe_cells(script: Path) -> list[tuple[str, str]]:
    """A recipe script's steps: after its docstring, every line belongs to a
    ``# %%`` cell, and there is no ``main`` — a program that runs inside one
    function reads as a single step."""
    text = script.read_text()
    tree = ast.parse(text)
    docstring = tree.body[0] if tree.body else None
    if not (isinstance(docstring, ast.Expr) and isinstance(docstring.value, ast.Constant)
            and isinstance(docstring.value.value, str)):
        raise ValueError(f"{script}: a recipe opens with its docstring")
    if any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body) or any(
            isinstance(n, ast.If) and "__main__" in ast.unparse(n.test) for n in tree.body):
        raise ValueError(f"{script}: a recipe's steps are its `# %%` cells, run top to bottom, "
                         "not a `main` it calls")
    rest = "\n".join(text.split("\n")[docstring.end_lineno:])
    marks = list(_CELL_MARK.finditer(rest))
    if not marks or rest[:marks[0].start()].strip():
        raise ValueError(f"{script}: every line after the docstring belongs to a `# %%` cell")
    cells = []
    for mark, following in zip(marks, [*marks[1:], None], strict=True):
        body = rest[mark.end():following.start() if following else len(rest)].strip("\n")
        if mark.group("markdown"):
            lines = body.split("\n")
            if any(line and not line.startswith("#") for line in lines):
                raise ValueError(f"{script}: a `# %% [markdown]` cell is comment lines only")
            prose = "\n".join(line[2:] if line.startswith("# ") else line[1:] for line in lines)
            cells.append(("markdown", _absolute_links(prose.strip(), script.parent)))
        else:
            cells.append(("code", body.rstrip()))
    if any(not text for _, text in cells):
        raise ValueError(f"{script}: a cell is empty")
    return cells


def read_recipe(script: Path) -> Recipe:
    text = script.read_text()
    title, prose = recipe_pages(script)
    return Recipe(
        name=script.parent.name,
        title=title,
        prose=prose,
        cells=recipe_cells(script),
        source=script.relative_to(REPO).as_posix(),
        server=bool(_NEEDS_SERVER.search(text)),
        extras=extras_of(text),
    )


def recipe_notebook(r: Recipe, release: str) -> tuple[Path, dict]:
    target = OUT / "recipes" / f"{r.name}.ipynb"
    cells = [header(r.title, r.source, colab_url(target, release)),
             setup_cell(release, server=r.server, extras=r.extras)]
    if r.prose:
        cells.append(markdown(r.prose))
    cells += [markdown(text) if kind == "markdown" else code(text) for kind, text in r.cells]
    return target, notebook(cells)


def recipe_chapter(r: Recipe) -> tuple[Path, str]:
    """The book chapter a recipe renders as: its overview, then its steps run
    in order."""
    prose = f"{r.prose}\n\n" if r.prose else ""
    steps = "\n\n".join(text if kind == "markdown" else f"```{{python}}\n{text}\n```"
                         for kind, text in r.cells)
    title = r.title.replace('"', '\\"')
    return RECIPE_CHAPTERS / f"{r.name}.qmd", (
        f'---\ntitle: "{title}"\n---\n\n'
        "<!-- Generated by scripts/build_notebooks.py from "
        f"{r.source} — edit the recipe, not this page. -->\n\n"
        f"{prose}"
        f"The program is [`{r.source}`](https://github.com/{GITHUB}/blob/main/{r.source});"
        " its steps run here in order.\n\n"
        f"{steps}\n"
    )


def sources() -> tuple[list[Path], list[Path]]:
    chapters = sorted(p for p in (BOOK / "chapters").rglob("*.qmd")
                      if RECIPE_CHAPTERS not in p.parents)
    recipes = [REPO / "cookbook" / "quickstart" / "quickstart.py",
               *sorted((REPO / "cookbook" / "recipes").glob("*/example.py"))]
    return chapters, recipes


def index(built: dict[Path, dict], release: str) -> str:
    """The notebooks' README: every notebook, its title, its Colab link."""

    def row(path: Path) -> str:
        title = built[path]["cells"][0]["source"][0].removeprefix("# ").strip()
        rel = path.relative_to(OUT).as_posix()
        return f"| [{title}]({rel}) | [Open]({colab_url(path, release)}) |"

    def section(name: str, sub: str) -> list[str]:
        rows = [row(p) for p in sorted(built) if p.relative_to(OUT).parts[0] == sub]
        return [f"## {name}", "", "| Notebook | Colab |", "|---|---|", *rows, ""]

    return "\n".join([
        "# Cookbook notebooks",
        "",
        f"Generated by `cookbook/book/scripts/build_notebooks.py` for jammi {release} —",
        "do not edit by hand. Each notebook installs that release; the Colab links",
        f"open the notebooks as tagged at `py-v{release}`.",
        "",
        *section("Recipes", "recipes"),
        *section("Book chapters", "book"),
    ])


def badge(release: str) -> str:
    """The script the rendered book includes on every page: a chapter page
    (``chapters/<path>.html``) gets its notebook's Open-in-Colab badge under
    its title — a recipe chapter (``chapters/recipes/<name>.html``) its
    recipe's notebook."""
    base = colab_url(OUT, release)
    img = '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">'
    return f"""<script>
(function () {{
  var page = location.pathname.match(/\\/chapters\\/(.+)\\.html$/);
  var title = document.querySelector("#title-block-header");
  if (!page || !title) return;
  var link = document.createElement("a");
  var notebook = page[1].indexOf("recipes/") === 0 ? page[1] : "book/" + page[1];
  link.href = "{base}/" + notebook + ".ipynb";
  link.target = "_blank";
  link.rel = "noopener";
  link.innerHTML = '{img}';
  title.appendChild(link);
}})();
</script>
"""


def build() -> dict[Path, str]:
    """Every generated file and its content."""
    release, refs = version(), references()
    chapters, recipes = sources()
    read = [read_recipe(s) for s in recipes]
    built = dict(chapter(q, release, refs) for q in chapters)
    built |= dict(recipe_notebook(r, release) for r in read)
    files = {p: json.dumps(nb, indent=1, ensure_ascii=False) + "\n" for p, nb in built.items()}
    files |= dict(recipe_chapter(r) for r in read)
    files[OUT / "README.md"] = index(built, release)
    files[BOOK / "_colab.html"] = badge(release)
    return files


def built_notebooks(files: dict[Path, str]) -> list[Path]:
    return [p for p in files if p.suffix == ".ipynb"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="fail when a committed notebook differs from its sources")
    args = parser.parse_args(argv)
    files = build()
    existing = {p for p in OUT.rglob("*") if p.is_file() and p.suffix in {".ipynb", ".md"}}
    existing |= set(RECIPE_CHAPTERS.glob("*.qmd"))
    stale = sorted(p for p, text in files.items() if not p.exists() or p.read_text() != text)
    orphans = sorted(existing - set(files))
    if args.check:
        if stale or orphans:
            for p in stale + orphans:
                print(f"notebooks: {p.relative_to(REPO)} is out of date", file=sys.stderr)
            print("notebooks: FAIL — run `python cookbook/book/scripts/build_notebooks.py`",
                  file=sys.stderr)
            return 1
        print(f"notebooks: {len(built_notebooks(files))} notebooks match their sources")
        return 0
    for p in orphans:
        p.unlink()
    for p in stale:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[p])
    total = len(built_notebooks(files))
    print(f"notebooks: wrote {len(stale)}, removed {len(orphans)}, {total} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
