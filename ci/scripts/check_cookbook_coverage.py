#!/usr/bin/env python3
"""Every public Python surface of jammi is run by something a reader runs.

The cookbook is where a new user learns jammi by running it, so the public
client surface and the cookbook move together: a verb the engine ships is
executed by a recipe (`cookbook/recipes/**`, `cookbook/quickstart/*`, each
smoke-run by `tests/cookbook_smoke.py`) or by an executed cell of a book
chapter (`cookbook/book/chapters/**/*.qmd`). This guard fails closed when a
shipped surface has no such exercise, so a verb cannot ship without its
runnable example.

## What is shipped — parsed from source, never listed by hand

  - every member of the `Session` and `JobHandle` protocols
    (`clients/python/jammi/_backend.py`) — the surface both transports carry;
    dunder members (`__enter__` / `__exit__`) are the `with` plumbing, not
    verbs, and are left out;
  - every `Capability` value (`clients/python/jammi/_capability.py`) — each
    names a one-sided member (`audit`, `ephemeral_session`, `preload_model`,
    `session_id`) that one transport carries;
  - every function `jammi/__init__.py` exports in `__all__`.

## How a surface is exercised — three lanes, each re-verified on every run

  1. **DirectCell** — the anchor (e.g. `db.search(`) appears in an EXECUTED
     cell of a chapter: a ```` ```{python} ```` fence with no `eval: false`
     option. A mention in prose, a bare ```` ```python ```` block, or an
     at-scale cell that does not run in the book's render never counts.
  2. **WrapperLane** — a `jammi_cookbook` helper calls the verb, and a
     chapter's executed cell calls the helper. Both anchors must resolve.
  3. **Recipe** — the anchor appears in the code of a recipe script (comments
     and docstrings stripped), and that recipe is registered in
     `tests/cookbook_smoke.py`. A property counts by an attribute read
     (`job.kind`), a method by a call.

A property of each row's evidence, not of the row: the anchor must still be
there, in executed code, today. Deleting a cell, a call or a recipe
registration reds the guard.

## Fail-closed

  - a shipped surface with no row;
  - a row naming a surface that is no longer shipped;
  - a row whose file is gone or whose anchor no longer resolves;
  - a surface with two rows;
  - a source file that no longer parses into a non-empty shipped set.

Run: `python3 ci/scripts/check_cookbook_coverage.py`
Self-test: `python3 ci/scripts/check_cookbook_coverage.py --self-test`
Hermetic: reads files in the working tree only; no wheel, no network.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT = REPO_ROOT / "clients" / "python" / "jammi"
BOOK = REPO_ROOT / "cookbook" / "book"
CHAPTERS = BOOK / "chapters"
LIB = BOOK / "jammi_cookbook"
COOKBOOK = REPO_ROOT / "cookbook"
SMOKE = REPO_ROOT / "tests" / "cookbook_smoke.py"

Reader = Callable[[Path], "str | None"]


class CoverageError(Exception):
    """A source that should define the shipped set does not — fails closed."""


# --------------------------------------------------------------------------- #
# SHIPPED
# --------------------------------------------------------------------------- #
def _parse(path: Path, read: Reader) -> ast.Module:
    text = read(path)
    if text is None:
        raise CoverageError(f"{path.relative_to(REPO_ROOT)} not found")
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise CoverageError(f"{path.relative_to(REPO_ROOT)} failed to parse: {exc}") from exc


def _class(module: ast.Module, name: str, path: Path) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise CoverageError(f"class {name} not found in {path.relative_to(REPO_ROOT)}")


def protocol_members(read: Reader) -> set[str]:
    path = CLIENT / "_backend.py"
    module = _parse(path, read)
    members: set[str] = set()
    for protocol in ("Session", "JobHandle"):
        for node in _class(module, protocol, path).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not (
                node.name.startswith("__") and node.name.endswith("__")
            ):
                members.add(node.name)
    return members


def capability_members(read: Reader) -> set[str]:
    path = CLIENT / "_capability.py"
    cls = _class(_parse(path, read), "Capability", path)
    return {
        node.value.value
        for node in cls.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def module_functions(read: Reader) -> set[str]:
    """The functions `jammi/__init__.py` exports: `__all__` names defined in
    the package root, or imported into it from a module that defines them as
    functions."""
    init = CLIENT / "__init__.py"
    module = _parse(init, read)
    exported: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exported = set(ast.literal_eval(node.value))
    if not exported:
        raise CoverageError("jammi/__init__.py has no __all__")

    def functions(tree: ast.Module) -> set[str]:
        return {
            n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        }

    found = functions(module) & exported
    for node in module.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            source = CLIENT / f"{node.module}.py"
            names = {alias.asname or alias.name for alias in node.names} & exported
            if names:
                found |= names & functions(_parse(source, read))
    return found


def load_shipped(read: Reader) -> set[str]:
    shipped = protocol_members(read) | capability_members(read) | module_functions(read)
    if not shipped:
        raise CoverageError("parsed an empty shipped set")
    return shipped


# --------------------------------------------------------------------------- #
# Lanes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectCell:
    """`anchor` in an executed cell of `chapter` (relative to chapters/)."""

    chapter: str
    anchor: str


@dataclass(frozen=True)
class WrapperLane:
    """`helper` (relative to jammi_cookbook/) calls the verb; an executed cell
    of `chapter` calls the helper."""

    helper: str
    helper_anchor: str
    chapter: str
    chapter_anchor: str


@dataclass(frozen=True)
class Recipe:
    """`anchor` in the code of `script` (relative to cookbook/), a recipe
    `tests/cookbook_smoke.py` runs."""

    script: str
    anchor: str


Lane = DirectCell | WrapperLane | Recipe


# --------------------------------------------------------------------------- #
# ACCOUNTING — one reviewed row per shipped surface. A list, so a duplicate is
# representable and caught.
# --------------------------------------------------------------------------- #
ACCOUNTING: list[tuple[str, Lane]] = [
    ("acceleration_report", Recipe("recipes/jobs/example.py", "job.acceleration_report(")),
    ("add_channel_columns", DirectCell("14-eval-channels/eval-channels.qmd", "db.add_channel_columns(")),
    ("add_source", Recipe("quickstart/quickstart.py", "db.add_source(")),
    ("asof_join", DirectCell("19-point-in-time/point-in-time.qmd", "db.asof_join(")),
    ("assemble_context", DirectCell("10-retrieval/retrieval.qmd", "db.assemble_context(")),
    ("audit", Recipe("recipes/search_audit/example.py", "db.audit.log(")),
    ("build_lexical_index", DirectCell("10-retrieval/retrieval.qmd", "db.build_lexical_index(")),
    ("build_neighbor_graph", Recipe("recipes/graph_and_lineage/example.py", "db.build_neighbor_graph(")),
    ("cancel", Recipe("recipes/jobs/example.py", "by_handle.cancel(")),
    ("cancel_job", Recipe("recipes/jobs/example.py", "db.cancel_job(")),
    ("close", DirectCell("25-incremental-refresh/incremental-refresh.qmd", "db.close(")),
    ("compact_embeddings", DirectCell("25-incremental-refresh/incremental-refresh.qmd", "db.compact_embeddings(")),
    ("conformalize", DirectCell("08-conformal/conformal.qmd", "db.conformalize(")),
    ("conformalize_cqr", DirectCell("08-conformal/conformal.qmd", "db.conformalize_cqr(")),
    ("conformalize_interval", DirectCell("08-conformal/conformal.qmd", "db.conformalize_interval(")),
    ("connect", Recipe("quickstart/quickstart.py", "jammi.connect(")),
    ("create_mutable_table", Recipe("recipes/mutable_tables/example.py", "db.create_mutable_table(")),
    ("delete_model", Recipe("recipes/model_catalog/example.py", "db.delete_model(")),
    ("derives_from", Recipe("recipes/graph_and_lineage/example.py", "db.derives_from(")),
    ("describe_model", Recipe("recipes/model_catalog/example.py", "db.describe_model(")),
    ("describe_sessions", Recipe("recipes/remote_session/example.py", "jammi.describe_sessions(")),
    ("describe_source", Recipe("recipes/model_catalog/example.py", "db.describe_source(")),
    ("describe_table", Recipe("recipes/graph_and_lineage/example.py", "db.describe_table(")),
    ("drop_mutable_table", Recipe("recipes/mutable_tables/example.py", "db.drop_mutable_table(")),
    ("drop_topic", Recipe("recipes/trigger_streams/example.py", "db.drop_topic(")),
    ("encode_query", Recipe("recipes/remote_session/example.py", "db.encode_query(")),
    ("ephemeral_session", Recipe("recipes/session_lifecycle/example.py", "db.ephemeral_session(")),
    ("eval_calibration", DirectCell("09-calibration/calibration.qmd", "db.eval_calibration(")),
    ("eval_compare", Recipe("recipes/eval_embeddings/example.py", "db.eval_compare(")),
    ("eval_embeddings", Recipe("recipes/eval_embeddings/example.py", "db.eval_embeddings(")),
    ("eval_inference", Recipe("recipes/eval_inference/example.py", "db.eval_inference(")),
    ("eval_per_query", Recipe("recipes/eval_embeddings/example.py", "db.eval_per_query(")),
    ("expire_versions", DirectCell("25-incremental-refresh/incremental-refresh.qmd", "db.expire_versions(")),
    ("fine_tune", Recipe("recipes/fine_tune/example.py", "db.fine_tune(")),
    ("fine_tune_graph", Recipe("recipes/fine_tune/example.py", "db.fine_tune_graph(")),
    ("generate_embeddings", Recipe("quickstart/quickstart.py", "db.generate_embeddings(")),
    ("generate_structure_embeddings", Recipe("recipes/graph_and_lineage/example.py", "db.generate_structure_embeddings(")),
    ("get_server_info", Recipe("recipes/model_catalog/example.py", "db.get_server_info(")),
    ("import_embeddings", Recipe("recipes/context_predictor/example.py", "db.import_embeddings(")),
    ("infer", Recipe("recipes/eval_inference/example.py", "db.infer(")),
    ("job", Recipe("recipes/jobs/example.py", "db.job(")),
    ("job_id", Recipe("recipes/jobs/example.py", "by_handle.job_id")),
    ("kind", Recipe("recipes/jobs/example.py", "by_handle.kind")),
    ("lexical_search", DirectCell("10-retrieval/retrieval.qmd", "db.lexical_search(")),
    ("list_channels", DirectCell("14-eval-channels/eval-channels.qmd", "db.list_channels(")),
    ("list_index_segments", DirectCell("25-incremental-refresh/incremental-refresh.qmd", "db.list_index_segments(")),
    ("list_jobs", Recipe("recipes/jobs/example.py", "db.list_jobs(")),
    ("list_models", Recipe("recipes/model_catalog/example.py", "db.list_models(")),
    ("list_mutable_tables", DirectCell("12-feature-store/feature-store.qmd", "db.list_mutable_tables(")),
    ("list_sources", DirectCell("01-construct/construct.qmd", "db.list_sources(")),
    ("list_topics", Recipe("recipes/trigger_streams/example.py", "db.list_topics(")),
    ("list_workers", Recipe("recipes/jobs/example.py", "db.list_workers(")),
    ("metrics", Recipe("recipes/jobs/example.py", "job.metrics(")),
    ("observe", Recipe("recipes/remote_session/example.py", "jammi.observe(")),
    ("open_session_labels", Recipe("recipes/remote_session/example.py", "jammi.open_session_labels(")),
    ("open_sessions", Recipe("recipes/remote_session/example.py", "jammi.open_sessions(")),
    ("output_model_id", Recipe("recipes/jobs/example.py", "job.output_model_id")),
    ("parse_target", Recipe("recipes/remote_session/example.py", "jammi.parse_target(")),
    ("predict_with_context_predictor", Recipe("recipes/context_predictor/example.py", "db.predict_with_context_predictor(")),
    ("preload_model", Recipe("recipes/model_catalog/example.py", "db.preload_model(")),
    ("progress", Recipe("recipes/jobs/example.py", "by_handle.progress(")),
    ("propagate_embeddings", Recipe("recipes/graph_and_lineage/example.py", "db.propagate_embeddings(")),
    ("prune_jobs", Recipe("recipes/jobs/example.py", "db.prune_jobs(")),
    ("publish_topic", Recipe("recipes/trigger_streams/example.py", "db.publish_topic(")),
    ("recompute", Recipe("recipes/graph_and_lineage/example.py", "db.recompute(")),
    ("reconcile", Recipe("recipes/graph_and_lineage/example.py", "db.reconcile(")),
    ("refresh_embeddings", DirectCell("25-incremental-refresh/incremental-refresh.qmd", "db.refresh_embeddings(")),
    ("register_channel", DirectCell("14-eval-channels/eval-channels.qmd", "db.register_channel(")),
    ("register_topic", Recipe("recipes/trigger_streams/example.py", "db.register_topic(")),
    ("rrf_fuse", DirectCell("10-retrieval/retrieval.qmd", "db.rrf_fuse(")),
    ("search", Recipe("quickstart/quickstart.py", "db.search(")),
    ("session_id", Recipe("recipes/remote_session/example.py", "remote.session_id")),
    ("set_tenant", Recipe("recipes/search_audit/example.py", "db.set_tenant(")),
    ("sql", Recipe("recipes/graph_and_lineage/example.py", "db.sql(")),
    ("staleness", Recipe("recipes/graph_and_lineage/example.py", "db.staleness(")),
    ("status", Recipe("recipes/jobs/example.py", "by_handle.status(")),
    ("subscribe_collect", Recipe("recipes/trigger_streams/example.py", "db.subscribe_collect(")),
    ("supports", Recipe("recipes/remote_session/example.py", "remote.supports(")),
    ("tenant", WrapperLane("rails.py", "db.tenant()", "01-construct/construct.qmd", "rails.tenant(")),
    ("tenant_scope", DirectCell("14-eval-channels/eval-channels.qmd", "db.tenant_scope(")),
    ("train_context_predictor", Recipe("recipes/context_predictor/example.py", "db.train_context_predictor(")),
    ("verify_materialization", Recipe("recipes/graph_and_lineage/example.py", "db.verify_materialization(")),
    ("wait", Recipe("recipes/jobs/example.py", "job.wait(")),
]


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
_CELL = re.compile(r"^```\{python\}[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_EVAL_FALSE = re.compile(r"^#\s*\|\s*eval:\s*false\s*$", re.MULTILINE)


def executed_cells(text: str) -> str:
    """Every executed ```{python}``` cell of a chapter, joined."""
    return "\n".join(c for c in _CELL.findall(text) if not _EVAL_FALSE.search(c))


def code_only(text: str) -> str:
    """Python source with comments and docstrings removed."""
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
    return re.sub(r"#.*", "", text)


def smoke_registered(script: str, smoke_text: str) -> bool:
    """Whether `tests/cookbook_smoke.py` runs `script` (relative to
    cookbook/): a recipe directory named by `example("…")` / `stepwise("…")`,
    or the quickstart script's own path segment."""
    parts = Path(script).parts
    if parts[0] == "recipes":
        name = parts[1]
        return bool(re.search(rf'\b(?:example|stepwise)\(\s*"{re.escape(name)}"', smoke_text))
    return f'"{parts[0]}" / "{parts[-1]}"' in smoke_text


def resolve(name: str, lane: Lane, read: Reader) -> list[str]:
    def chapter_has(chapter: str, anchor: str, what: str) -> list[str]:
        text = read(CHAPTERS / chapter)
        if text is None:
            return [f"{name}: {what} chapter not found: {chapter}"]
        if anchor not in executed_cells(text):
            return [f"{name}: {what} anchor {anchor!r} not in an executed cell of {chapter}"]
        return []

    if isinstance(lane, DirectCell):
        return chapter_has(lane.chapter, lane.anchor, "DirectCell")
    if isinstance(lane, WrapperLane):
        failures = chapter_has(lane.chapter, lane.chapter_anchor, "WrapperLane")
        helper = read(LIB / lane.helper)
        if helper is None:
            failures.append(f"{name}: WrapperLane helper not found: {lane.helper}")
        elif lane.helper_anchor not in code_only(helper):
            failures.append(
                f"{name}: WrapperLane helper anchor {lane.helper_anchor!r} not in the code "
                f"of {lane.helper}"
            )
        return failures
    if isinstance(lane, Recipe):
        script = read(COOKBOOK / lane.script)
        if script is None:
            return [f"{name}: Recipe script not found: {lane.script}"]
        failures = []
        if lane.anchor not in code_only(script):
            failures.append(
                f"{name}: Recipe anchor {lane.anchor!r} not in the code of {lane.script}"
            )
        smoke = read(SMOKE)
        if smoke is None or not smoke_registered(lane.script, smoke):
            failures.append(f"{name}: {lane.script} is not run by tests/cookbook_smoke.py")
        return failures
    raise CoverageError(f"{name}: unknown lane {type(lane)!r}")


DOCTRINE = (
    "a public surface ships with a recipe or an executed chapter cell that runs it — "
    "the cookbook is where a new user learns jammi by running it."
)


def reconcile(shipped: set[str], accounting: list[tuple[str, Lane]], read: Reader) -> list[str]:
    failures: list[str] = []
    seen: dict[str, Lane] = {}
    for name, lane in accounting:
        if name in seen:
            failures.append(f"`{name}` has two rows. {DOCTRINE}")
            continue
        seen[name] = lane
    for name in sorted(shipped - seen.keys()):
        failures.append(f"`{name}` is shipped but nothing a reader runs exercises it. {DOCTRINE}")
    for name in sorted(seen.keys() - shipped):
        failures.append(f"`{name}` has a row but is no longer shipped. {DOCTRINE}")
    for name in sorted(seen.keys() & shipped):
        failures.extend(resolve(name, seen[name], read))
    return failures


def _disk(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


# --------------------------------------------------------------------------- #
# Self-test: the reconciliation bites, on synthetic files.
# --------------------------------------------------------------------------- #
def self_test() -> int:
    backend = (
        "class JobHandle:\n"
        "    @property\n"
        "    def kind(self): ...\n"
        "class Session:\n"
        "    def __enter__(self): ...\n"
        "    def alpha(self): ...\n"
        "    def beta(self): ...\n"
        "    def gamma(self): ...\n"
    )
    capability = 'class Capability:\n    AUDIT = "audit"\n'
    init = (
        'from ._util import helper, NotAFunction\n'
        '__all__ = ["connect", "helper", "NotAFunction"]\n'
        "def connect(): ...\n"
    )
    util = "def helper(): ...\nclass NotAFunction: ...\n"
    chapter = (
        "```{python}\ndb.alpha()\nrails.wrap(db)\n```\n"
        "```{python}\n#| eval: false\ndb.gamma()\n```\n"
        "```python\ndb.gamma()\n```\n"
    )
    files = {
        CLIENT / "_backend.py": backend,
        CLIENT / "_capability.py": capability,
        CLIENT / "__init__.py": init,
        CLIENT / "_util.py": util,
        CHAPTERS / "synthetic.qmd": chapter,
        LIB / "rails.py": "def wrap(db):\n    db.beta()\n",
        COOKBOOK / "recipes" / "demo" / "example.py": (
            '"""Calls db.connect() only in prose."""\n'
            "jammi.connect(x)\nhandle.kind\ndb.audit.log([])\njammi.helper()\n"
        ),
        SMOKE: 'RECIPES = (example("demo"),)\n',
    }
    read = files.get
    failures: list[str] = []

    shipped = load_shipped(read)
    expected = {"kind", "alpha", "beta", "gamma", "audit", "connect", "helper"}
    if shipped != expected:
        failures.append(f"shipped set parsed as {sorted(shipped)}, expected {sorted(expected)}")

    good: list[tuple[str, Lane]] = [
        ("alpha", DirectCell("synthetic.qmd", "db.alpha(")),
        ("beta", WrapperLane("rails.py", "db.beta(", "synthetic.qmd", "rails.wrap(")),
        ("gamma", Recipe("recipes/demo/example.py", "jammi.connect(")),
        ("kind", Recipe("recipes/demo/example.py", "handle.kind")),
        ("audit", Recipe("recipes/demo/example.py", "db.audit.log(")),
        ("connect", Recipe("recipes/demo/example.py", "jammi.connect(")),
        ("helper", Recipe("recipes/demo/example.py", "jammi.helper(")),
    ]
    if clean := reconcile(shipped, good, read):
        failures.append(f"a fully accounted set reported findings: {clean}")

    def bites(label: str, accounting: list[tuple[str, Lane]], needle: str, files_=None) -> None:
        found = reconcile(shipped, accounting, (files_ or files).get)
        if not any(needle in f for f in found):
            failures.append(f"{label} was not caught: {found}")

    bites("an unaccounted surface", good[1:], "`alpha` is shipped")
    bites("a stale row", good + [("omega", DirectCell("synthetic.qmd", "x"))], "no longer shipped")
    bites("a duplicate row", good + [good[0]], "two rows")
    bites(
        "an at-scale (eval: false) cell credited",
        [("gamma", DirectCell("synthetic.qmd", "db.gamma("))] + good[:2] + good[3:],
        "not in an executed cell",
    )
    bites(
        "a docstring mention credited",
        [("connect", Recipe("recipes/demo/example.py", "db.connect("))] + good[:5] + good[6:],
        "not in the code",
    )
    unregistered = dict(files) | {SMOKE: "RECIPES = ()\n"}
    bites("an unregistered recipe", good, "not run by tests/cookbook_smoke.py", unregistered)

    if failures:
        for f in failures:
            print(f"self-test FAILED: {f}", file=sys.stderr)
        return 1
    print(
        "cookbook-coverage self-test: OK — parsing, the three lanes, and every "
        "fail-closed rule bite."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()
    try:
        shipped = load_shipped(_disk)
    except CoverageError as exc:
        print(f"cookbook-coverage: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1
    failures = reconcile(shipped, ACCOUNTING, _disk)
    lanes = dict(ACCOUNTING)
    for name in sorted(shipped):
        lane = lanes.get(name)
        print(f"  {name:<32} {type(lane).__name__ if lane else '!!! NOT EXERCISED'}")
    if failures:
        print("\ncookbook-coverage: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\ncookbook-coverage: PASS — all {len(shipped)} shipped surfaces are run by the cookbook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
