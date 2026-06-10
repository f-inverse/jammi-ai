"""Cookbook smoke runner — every recipe is a CI gate.

Default: runs the quickstart and the fast recipes. Fails the build if any
recipe exits non-zero, if quickstart wall-clock exceeds 60 seconds, or if
the smoke runner itself errors.

Set `JAMMI_COOKBOOK_SLOW=1` to additionally run `fine_tune` (slow on CPU)
and `flight_sql` (requires `cargo build --release -p jammi-server` to have
produced `target/release/jammi-server`). The nightly CI cron sets this flag.

Run with `python tests/cookbook_smoke.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COOKBOOK = REPO_ROOT / "cookbook"

QUICKSTART_BUDGET_S = 60.0
SLOW_FLAG = "JAMMI_COOKBOOK_SLOW"


@dataclass(frozen=True)
class Recipe:
    name: str
    path: Path
    slow: bool = False


RECIPES: tuple[Recipe, ...] = (
    Recipe("quickstart", COOKBOOK / "quickstart" / "quickstart.py"),
    Recipe("mutable_tables", COOKBOOK / "recipes" / "mutable_tables" / "example.py"),
    Recipe("trigger_streams", COOKBOOK / "recipes" / "trigger_streams" / "example.py"),
    Recipe("eval_embeddings", COOKBOOK / "recipes" / "eval_embeddings" / "example.py"),
    Recipe("image_search", COOKBOOK / "recipes" / "image_search" / "example.py"),
    Recipe("audio_search", COOKBOOK / "recipes" / "audio_search" / "example.py"),
    Recipe("eval_inference", COOKBOOK / "recipes" / "eval_inference" / "example.py"),
    Recipe(
        "eval_inference_ner",
        COOKBOOK / "recipes" / "eval_inference_ner" / "example.py",
    ),
    Recipe("search_audit", COOKBOOK / "recipes" / "search_audit" / "example.py"),
    Recipe(
        "session_lifecycle",
        COOKBOOK / "recipes" / "session_lifecycle" / "example.py",
    ),
    Recipe("fine_tune", COOKBOOK / "recipes" / "fine_tune" / "example.py", slow=True),
    Recipe("flight_sql", COOKBOOK / "recipes" / "flight_sql" / "example.py", slow=True),
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
        [sys.executable, str(recipe.path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    elapsed = time.monotonic() - start
    return Result(
        name=recipe.name,
        elapsed_s=elapsed,
        returncode=completed.returncode,
        stderr=completed.stderr,
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
        print(f"  {marker}  {result.name:<18}  {result.elapsed_s:>6.2f}s")
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
