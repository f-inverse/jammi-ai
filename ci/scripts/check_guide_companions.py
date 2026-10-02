#!/usr/bin/env python3
"""Every guide capability page names the cookbook chapter or recipe that runs it.

The guide says how to call a capability; the cookbook runs it. A capability that
is an argument, a config section or a deployment shape adds no Python verb, so
`check_cookbook_coverage.py` cannot see it: this guard closes that gap from the
guide's side. Every page listed under the `# How-To Guides` and `# Operations`
parts of `docs/guide/src/SUMMARY.md` (nested entries included) opens with a
companion line naming where the capability runs, and every place it names must
exist and run.

## The companion line

The first block after the page's `# ` title is a blockquote whose first line
starts `> **Measured companion:**` (a book chapter) or `> **Runnable companion:**`
(a recipe). Every link in that blockquote is one of:

  - a book chapter, `https://f-inverse.github.io/jammi-ai/cookbook/chapters/<path>.html`:
    `cookbook/book/chapters/<path>.qmd` exists and `cookbook/book/_quarto.yml`
    renders it (the book's render is what executes a chapter);
  - a recipe, `https://github.com/f-inverse/jammi-ai/tree/main/cookbook/recipes/<name>`,
    or the quickstart, `…/tree/main/cookbook/quickstart`: `tests/cookbook_smoke.py`
    registers it (the smoke run is what executes a recipe).

## Fail-closed

  - a capability page with no companion line;
  - a companion line with no link;
  - a link of neither form (a companion names where the capability runs, nothing else);
  - a chapter the repository does not have, or the book does not render;
  - a recipe the smoke run does not register;
  - a SUMMARY entry naming a page that does not exist;
  - a SUMMARY with no How-To or Operations pages (the parse found nothing).

Run: `python3 ci/scripts/check_guide_companions.py`
Self-test: `python3 ci/scripts/check_guide_companions.py --self-test`
Hermetic: reads files in the working tree only.
"""

from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The SUMMARY parts whose pages each show how to use a capability.
CAPABILITY_PARTS = ("How-To Guides", "Operations")
COMPANION_OPENERS = ("> **Measured companion:**", "> **Runnable companion:**")
CHAPTER_URL = re.compile(r"^https://f-inverse\.github\.io/jammi-ai/cookbook/chapters/(.+)\.html$")
RECIPE_URL = re.compile(
    r"^https://github\.com/f-inverse/jammi-ai/tree/main/cookbook/(recipes/[a-z0-9_]+|quickstart)/?$"
)
LINK = re.compile(r"\]\(([^)\s]+)\)")
SUMMARY_ENTRY = re.compile(r"^\s*-\s*\[[^\]]*\]\(\./([^)]+)\)")
RENDERED_CHAPTER = re.compile(r"^\s*-\s*chapters/(\S+)\.qmd\s*$")
SMOKE_RECIPE = re.compile(r"\bexample\(\s*\"([a-z0-9_]+)\"")
SMOKE_QUICKSTART = re.compile(r"COOKBOOK\s*/\s*\"quickstart\"")


@dataclass(frozen=True)
class Tree:
    """The files the guard reads, rooted at `root`."""

    root: Path

    @property
    def guide(self) -> Path:
        return self.root / "docs" / "guide" / "src"

    def text(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")


def capability_pages(summary: str) -> list[str]:
    """The pages listed under the capability parts of `summary`, in order."""
    pages: list[str] = []
    part = None
    for line in summary.splitlines():
        if line.startswith("# "):
            part = line[2:].strip()
            continue
        entry = SUMMARY_ENTRY.match(line)
        if entry and part in CAPABILITY_PARTS:
            pages.append(entry.group(1))
    return pages


def companion_block(page: str) -> list[str]:
    """The blockquote opening `page` after its title: its lines, or `[]`."""
    lines = page.splitlines()
    body = iter(lines)
    for line in body:
        if line.startswith("# "):
            break
    else:
        return []
    block: list[str] = []
    for line in body:
        if not block and not line.strip():
            continue
        if not line.startswith(">"):
            break
        block.append(line)
    return block


def rendered_chapters(quarto: str) -> set[str]:
    """The chapter paths (without `.qmd`) the book renders."""
    return {m.group(1) for line in quarto.splitlines() if (m := RENDERED_CHAPTER.match(line))}


def smoke_registered(smoke: str) -> set[str]:
    """The recipe paths (`recipes/<name>`, `quickstart`) the smoke run executes."""
    registered = {f"recipes/{name}" for name in SMOKE_RECIPE.findall(smoke)}
    if SMOKE_QUICKSTART.search(smoke):
        registered.add("quickstart")
    return registered


def check(tree: Tree) -> list[str]:
    """Every failure, as `<page>: <what is wrong>`."""
    pages = capability_pages((tree.guide / "SUMMARY.md").read_text(encoding="utf-8"))
    if not pages:
        return ["SUMMARY.md: no How-To or Operations pages parsed"]
    chapters = rendered_chapters(tree.text("cookbook/book/_quarto.yml"))
    recipes = smoke_registered(tree.text("tests/cookbook_smoke.py"))
    failures: list[str] = []
    for name in pages:
        path = tree.guide / name
        if not path.is_file():
            failures.append(f"{name}: listed in SUMMARY.md but missing")
            continue
        block = companion_block(path.read_text(encoding="utf-8"))
        if not block or not block[0].startswith(COMPANION_OPENERS):
            failures.append(
                f"{name}: no companion line — open with `> **Measured companion:**` "
                "(a book chapter) or `> **Runnable companion:**` (a recipe)"
            )
            continue
        links = LINK.findall("\n".join(block))
        if not links:
            failures.append(f"{name}: the companion line names no chapter or recipe")
        for url in links:
            if chapter := CHAPTER_URL.match(url):
                rel = chapter.group(1)
                if not (tree.root / "cookbook/book/chapters" / f"{rel}.qmd").is_file():
                    failures.append(f"{name}: no chapter cookbook/book/chapters/{rel}.qmd")
                elif rel not in chapters:
                    failures.append(f"{name}: chapters/{rel}.qmd is not rendered by _quarto.yml")
            elif recipe := RECIPE_URL.match(url):
                rel = recipe.group(1)
                if rel not in recipes:
                    failures.append(f"{name}: cookbook/{rel} is not run by tests/cookbook_smoke.py")
            else:
                failures.append(f"{name}: companion link {url} is neither a chapter nor a recipe")
    return failures


def self_test() -> int:
    """The guard passes a covered page and fails each way a page can fall short."""
    chapter = "https://f-inverse.github.io/jammi-ai/cookbook/chapters/01-a/a.html"
    recipe = "https://github.com/f-inverse/jammi-ai/tree/main/cookbook/recipes/r"
    cases = {
        "a measured companion": (f"> **Measured companion:** see [A]({chapter}).", 0),
        "a runnable companion": (f"> **Runnable companion:** [`r`]({recipe}) runs it.", 0),
        "no companion line": ("Plain prose first.", 1),
        "a companion with no link": ("> **Measured companion:** the book.", 1),
        "a link of neither form": ("> **Measured companion:** [x](https://example.com).", 1),
        "a chapter the tree lacks": (
            "> **Measured companion:** [B](https://f-inverse.github.io/jammi-ai/cookbook/chapters/02-b/b.html).",
            1,
        ),
        "a recipe the smoke run skips": (
            "> **Runnable companion:** [s](https://github.com/f-inverse/jammi-ai/tree/main/cookbook/recipes/s).",
            1,
        ),
    }
    failed = []
    for label, (opening, expected) in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide = root / "docs/guide/src"
            guide.mkdir(parents=True)
            (guide / "SUMMARY.md").write_text(
                "# Summary\n\n# How-To Guides\n\n- [Page](./page.md)\n\n# Reference\n\n- [Ref](./ref.md)\n"
            )
            (guide / "page.md").write_text(f"# Page\n\n{opening}\n\nBody.\n")
            chapters = root / "cookbook/book/chapters/01-a"
            chapters.mkdir(parents=True)
            (chapters / "a.qmd").write_text("---\ntitle: A\n---\n")
            (root / "cookbook/book/_quarto.yml").write_text("book:\n  chapters:\n    - chapters/01-a/a.qmd\n")
            (root / "tests").mkdir()
            (root / "tests/cookbook_smoke.py").write_text('RECIPES = (example("r"),)\n')
            got = len(check(Tree(root)))
            if (got == 0) != (expected == 0):
                failed.append(f"{label}: expected {'a failure' if expected else 'a pass'}, got {got} failure(s)")
    if failed:
        for f in failed:
            print(f"self-test FAILED: {f}", file=sys.stderr)
        return 1
    print("guide-companions self-test: OK — each way a page can fall short is caught")
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    failures = check(Tree(REPO_ROOT))
    if failures:
        print("guide-companions: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        print(
            "\nguide-companions: every How-To and Operations page opens with the cookbook "
            "chapter or recipe that runs its capability. See this script's docstring.",
            file=sys.stderr,
        )
        return 1
    print("guide-companions: OK — every capability page names where it runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
