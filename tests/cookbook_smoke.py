"""Cookbook smoke runner — every recipe is a release gate.

Runs the quickstart and every recipe, with the real Hub models they teach with;
the cookbook lane (`.github/workflows/cookbook-gpu.yml`) runs it on a GPU. Fails
if any recipe exits non-zero, if quickstart wall-clock exceeds 60 seconds, or if
the smoke runner itself errors. The recipes that talk to a server
(`remote_session`, `flight_sql`) start one with `jammi.testing.LiveServer`,
which runs the `jammi-server` on PATH; the lane that runs this puts one there.

Every step runs under `python -m jammi.session_journal`, so a recipe that
leaves a session open — in any process it starts — fails by the session's
label, and a recipe's exit status is never the only evidence it closed what it
opened.

Run with `python tests/cookbook_smoke.py`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COOKBOOK = REPO_ROOT / "cookbook"

QUICKSTART_BUDGET_S = 60.0


@dataclass(frozen=True)
class Recipe:
    name: str
    script: Path


def example(name: str) -> Recipe:
    return Recipe(name, COOKBOOK / "recipes" / name / "example.py")


RECIPES: tuple[Recipe, ...] = (
    Recipe("quickstart", COOKBOOK / "quickstart" / "quickstart.py"),
    example("mutable_tables"),
    example("cloud_storage"),
    example("trigger_streams"),
    example("eval_embeddings"),
    example("image_search"),
    example("cross_modal_search"),
    example("audio_search"),
    example("eval_inference"),
    example("eval_inference_ner"),
    example("search_audit"),
    example("session_lifecycle"),
    example("remote_model"),
    example("fine_tune"),
    example("graph_and_lineage"),
    example("model_catalog"),
    example("jobs"),
    example("context_predictor"),
    example("remote_session"),
    example("flight_sql"),
    example("compound_query"),
    example("cli"),
)


@dataclass
class Result:
    name: str
    elapsed_s: float
    returncode: int
    stderr: str


def run_recipe(recipe: Recipe) -> Result:
    start = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-m", "jammi.session_journal", "--", sys.executable, str(recipe.script)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return Result(
        name=recipe.name,
        elapsed_s=time.monotonic() - start,
        returncode=completed.returncode,
        stderr=completed.stderr,
    )


def main() -> int:
    print(f"Cookbook smoke — {len(RECIPES)} recipes")
    print("-" * 60)

    failures: list[Result] = []
    budget_breach: Result | None = None
    for recipe in RECIPES:
        result = run_recipe(recipe)
        marker = "PASS" if result.returncode == 0 else "FAIL"
        print(f"  {marker}  {result.name:<26}  {result.elapsed_s:>6.2f}s")
        if result.returncode != 0:
            failures.append(result)
        if recipe.name == "quickstart" and result.elapsed_s > QUICKSTART_BUDGET_S:
            budget_breach = result

    print("-" * 60)

    exit_code = 0
    if failures:
        exit_code = 1
        for failure in failures:
            print(f"\n--- {failure.name} stderr ---\n{failure.stderr}", file=sys.stderr)
    if budget_breach is not None:
        exit_code = 1
        print(
            f"\nQUICKSTART OVER BUDGET: {budget_breach.elapsed_s:.2f}s > "
            f"{QUICKSTART_BUDGET_S:.0f}s",
            file=sys.stderr,
        )

    if exit_code == 0:
        print("All cookbook recipes PASSED")
    else:
        print("Cookbook smoke FAILED")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
