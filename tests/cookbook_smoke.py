"""Cookbook smoke runner — every recipe is a CI gate.

Default: runs the quickstart and the fast recipes. Fails the build if any
recipe exits non-zero, if quickstart wall-clock exceeds 60 seconds, or if
the smoke runner itself errors.

Every step runs under `python -m jammi.session_journal`, so a recipe that
leaves a session open — in any process it starts — fails by the session's
label, and a recipe's exit status is never the only evidence it closed what it
opened.

Set `JAMMI_COOKBOOK_SLOW=1` to additionally run `fine_tune` (slow on CPU)
and `flight_sql` (requires `cargo build --release -p jammi-server` to have
produced `target/release/jammi-server`). The nightly CI cron sets this flag.

Run with `python tests/cookbook_smoke.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COOKBOOK = REPO_ROOT / "cookbook"

QUICKSTART_BUDGET_S = 60.0
SLOW_FLAG = "JAMMI_COOKBOOK_SLOW"


@dataclass(frozen=True)
class Recipe:
    """One or more scripts run in order. A stepwise recipe's steps share one
    fresh working directory, handed to them through `workdir_env`."""

    name: str
    steps: tuple[Path, ...]
    slow: bool = False
    workdir_env: str | None = None


def example(name: str, *, slow: bool = False) -> Recipe:
    return Recipe(name, (COOKBOOK / "recipes" / name / "example.py",), slow=slow)


def stepwise(name: str, workdir_env: str) -> Recipe:
    steps = tuple(sorted((COOKBOOK / "recipes" / name).glob("[0-9][0-9]-*.py")))
    assert steps, f"no numbered steps under cookbook/recipes/{name}"
    return Recipe(f"{name} (stepwise)", steps, workdir_env=workdir_env)


RECIPES: tuple[Recipe, ...] = (
    Recipe("quickstart", (COOKBOOK / "quickstart" / "quickstart.py",)),
    example("mutable_tables"),
    example("trigger_streams"),
    example("eval_embeddings"),
    example("image_search"),
    stepwise("image_search", "JAMMI_IMAGE_WORKDIR"),
    example("audio_search"),
    stepwise("audio_search", "JAMMI_AUDIO_WORKDIR"),
    example("eval_inference"),
    example("eval_inference_ner"),
    example("search_audit"),
    example("session_lifecycle"),
    example("fine_tune", slow=True),
    example("flight_sql", slow=True),
)


@dataclass
class Result:
    name: str
    elapsed_s: float
    returncode: int
    stderr: str


def run_recipe(recipe: Recipe) -> Result:
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="jammi-cookbook-smoke-") as workdir:
        env = dict(os.environ)
        if recipe.workdir_env is not None:
            env[recipe.workdir_env] = workdir
        returncode, stderr = 0, ""
        for step in recipe.steps:
            completed = subprocess.run(
                [sys.executable, "-m", "jammi.session_journal", "--", sys.executable, str(step)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env=env,
            )
            returncode, stderr = completed.returncode, completed.stderr
            if returncode != 0:
                stderr = f"[{step.name}]\n{stderr}"
                break
    return Result(
        name=recipe.name,
        elapsed_s=time.monotonic() - start,
        returncode=returncode,
        stderr=stderr,
    )


def main() -> int:
    include_slow = os.environ.get(SLOW_FLAG) == "1"
    selected = [r for r in RECIPES if include_slow or not r.slow]
    skipped = [r for r in RECIPES if not include_slow and r.slow]

    print(f"Cookbook smoke — {len(selected)} recipes" + (
        f"  (skipping {len(skipped)} slow: {', '.join(r.name for r in skipped)})"
        if skipped
        else ""
    ))
    print("-" * 60)

    failures: list[Result] = []
    budget_breach: Result | None = None
    for recipe in selected:
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
