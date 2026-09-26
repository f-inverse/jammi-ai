"""The notebook builder's conversion rules, over a synthetic chapter and recipe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_notebooks.py"
_spec = importlib.util.spec_from_file_location("build_notebooks", _SCRIPT)
build = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = build
_spec.loader.exec_module(build)

CHAPTER = '''---
title: "Widgets, measured"
---

```{python}
# | echo: false
import jammi_cookbook
```

Widgets follow the classic result [@doe2020; @roe2021], and the
[sibling chapter](../02-other/other.qmd#setup) builds on it.

```{python}
#| eval: false
db.never_run()
```

```{python}
db = jammi.connect("grpc://127.0.0.1:8081")
contracts.assert_close("widget.n", 1)
```

Closing prose.
'''

BIB = r"""
@article{doe2020, title = {On {W}idgets}, author = {Doe, Jane}, journal = {J. Widgets},
  year = {2020}}
@inproceedings{roe2021, title = {Widgets at Scale}, author = {Roe, Ann and Poe, Bo and Loe, Cy},
  booktitle = {Proc. Widgets}, year = {2021}}
"""

RECIPE = '''"""Widgets: make one.

It is made in a moment.

Run with `python cookbook/recipes/widgets/example.py`.
"""

from __future__ import annotations

import jammi


def main() -> int:
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    book, out = tmp_path / "cookbook" / "book", tmp_path / "cookbook" / "notebooks"
    (book / "chapters" / "01-widgets").mkdir(parents=True)
    (book / "chapters" / "01-widgets" / "widgets.qmd").write_text(CHAPTER)
    (book / "references.bib").write_text(BIB)
    (tmp_path / "cookbook" / "recipes" / "widgets").mkdir(parents=True)
    (tmp_path / "cookbook" / "recipes" / "widgets" / "example.py").write_text(RECIPE)
    for name, value in (("BOOK", book), ("REPO", tmp_path), ("OUT", out)):
        monkeypatch.setattr(build, name, value)
    return tmp_path


def _text(cell: dict) -> str:
    return "".join(cell["source"])


def test_a_chapter_keeps_its_cells_in_order_and_drops_their_options(tree):
    refs = build.references(tree / "cookbook" / "book" / "references.bib")
    qmd = tree / "cookbook" / "book" / "chapters" / "01-widgets" / "widgets.qmd"
    target, nb = build.chapter(qmd, "9.9.9", refs)
    assert target == tree / "cookbook" / "notebooks" / "book" / "01-widgets" / "widgets.ipynb"
    kinds = [(c["cell_type"], _text(c).split("\n")[0]) for c in nb["cells"]]
    assert kinds == [
        ("markdown", "# Widgets, measured"),
        ("code", "# Setup: jammi 9.9.9 — the CUDA engine on an sm_80+ GPU (L4, A100, …), the"),
        ("code", "import jammi_cookbook"),
        ("markdown", "Widgets follow the classic result (Doe 2020; Roe et al. 2021), and the"),
        ("markdown", "```python"),
        ("code", 'db = jammi.connect("grpc://127.0.0.1:8081")'),
        ("markdown", "Closing prose."),
        ("markdown", "## References"),
    ]
    hidden = nb["cells"][2]["metadata"]
    assert hidden == {"jupyter": {"source_hidden": True}}
    assert "#" not in _text(nb["cells"][2])
    assert f"{build.BOOK_URL}/chapters/02-other/other.html#setup" in _text(nb["cells"][3])
    assert _text(nb["cells"][-1]).count("\n- ") == 2
    assert [c["id"] for c in nb["cells"]] == [f"cell-{i}" for i in range(len(nb["cells"]))]


def test_the_setup_pins_the_release_and_installs_a_server_only_when_needed(tree):
    served = _text(build.setup_cell("9.9.9", server=True))
    plain = _text(build.setup_cell("9.9.9", server=False))
    assert '"jammi-ai==9.9.9", engine + "==9.9.9", "jammi-cookbook==9.9.9"' in plain
    assert 'server + "==9.9.9"' in served and 'server + "' not in plain


def test_a_recipe_is_its_docstring_its_body_and_a_run_of_main(tree):
    script = tree / "cookbook" / "recipes" / "widgets" / "example.py"
    target, nb = build.recipe(script, "9.9.9")
    assert target.name == "widgets.ipynb"
    texts = [_text(c) for c in nb["cells"]]
    assert texts[0].startswith("# Widgets: make one\n")
    assert texts[2] == "It is made in a moment."
    assert texts[3].startswith("from __future__ import annotations") and "__main__" not in texts[3]
    assert texts[4] == "assert main() == 0"


def test_the_colab_link_opens_the_release_tag(tree):
    path = tree / "cookbook" / "notebooks" / "recipes" / "widgets.ipynb"
    assert build.colab_url(path, "9.9.9") == (
        "https://colab.research.google.com/github/f-inverse/jammi-ai/blob/py-v9.9.9/"
        "cookbook/notebooks/recipes/widgets.ipynb"
    )
