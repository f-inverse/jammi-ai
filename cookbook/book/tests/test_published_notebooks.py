"""How the published-notebook runner divides the committed notebooks among its shards."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_published_notebooks.py"
_spec = importlib.util.spec_from_file_location("run_published_notebooks", _SCRIPT)
runner = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = runner
_spec.loader.exec_module(runner)

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"


def test_the_shards_run_every_committed_notebook_exactly_once():
    notebooks = runner.notebooks_under(NOTEBOOKS)
    assert Path("recipes/quickstart.ipynb") in notebooks
    assert Path("book/construct/construct.ipynb") in notebooks
    shards = [runner.Shard(i, 8).select(notebooks) for i in range(1, 9)]
    assert sorted(nb for shard in shards for nb in shard) == notebooks
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("text", ["0/8", "9/8", "3", "a/b"])
def test_a_shard_outside_its_count_is_refused(text):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        runner.Shard.parse(text)
