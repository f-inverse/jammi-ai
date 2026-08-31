#!/usr/bin/env python3
"""Chapter-coverage-COMPLETENESS gate — hermetic, static, no build, no network,
no engine wheel required.

## The escape this closes (#351 review)

The engine and cookbook co-evolve: `check_doc_parity.py` / `check_api_reference.py`
already guard the FORWARD half of that loop — a public surface's signature
cannot silently drift out from under a chapter that calls it. But nothing
guarded the RETURN half: when the #351 quantized-serving surface shipped on
the engine side, whether it got a cookbook chapter exercising it depended on
someone *remembering* to author one — the forward guard existed, the return
path was memory. Operator doctrine (2026-08-31): "memories are unreliable —
mechanize the doctrine." This gate is that mechanization: every SHIPPED
Python surface is accounted for, by name, in this file — so a new public
capability shipping with no chapter cell exercising it REDs this gate
instead of silently shipping proof-less.

This is the book's own analog of `ci/scripts/check_gpu_parity_matrix.py`'s
completeness-matrix shape (SHIPPED, parsed from source, reconciled against a
reviewed, in-repo ACCOUNTING that must name every shipped cell) — same
property, applied to the (Python surface) axis instead of the
(architecture × verb) axis.

## SHIPPED surface set — reused, not reinvented

`check_api_reference.py` already enumerates the book's shipped Python
surface: its `REQUIRED: dict[str, list[str]]` (one entry per `Database`
method the chapters rely on, keyed by method name) and its
`MODULE_FUNCTIONS: list[str]` (top-level `jammi.*` functions — today just
`connect`). That is 48 surfaces today. This gate parses those two
module-level bindings directly out of `check_api_reference.py`'s source via
`ast` (never imports `jammi` — hermetic, no wheel needed) rather than
inventing a second enumeration mechanism that could drift from the first.

**Granularity call, made explicit**: `REQUIRED`'s own granularity is one row
per Python-callable *name* (a method or module function), not per kwarg and
not per capability *value* (e.g. `supports(capability=...)`'s many
`Capability` members, or `fine_tune(method="lora")`'s adapter-family values,
are not each a separate row). That is the RIGHT granularity for coverage
too: a "new public capability" manifests as a new key landing in `REQUIRED`
or a new entry in `MODULE_FUNCTIONS` (exactly what trips
`check_api_reference.py` itself when a signature drifts), so tracking
coverage at that same granularity means a newly-shipped verb is caught by
construction — the same edit that adds it to `REQUIRED` is the edit this
gate's ACCOUNTING must also grow to cover. A finer (per-kwarg-value) axis
would be truer to "capability" in the abstract but has no single reviewed
enumeration to parse from source today, and inventing one would violate the
"reuse the enumeration mechanism" instruction below — so this gate is
honest about tracking *named verbs*, not every value they accept.

## Three exercise LANES, each mechanically verified

A verb is "exercised" by a chapter through one of three real, in-repo call
shapes — the cookbook's own architecture surfaces all three:

  1. **DirectCell** — the verb is called directly (`db.<verb>(` or an
     equally explicit call) inside a live `` ```{python} `` cell of a
     `cookbook/book/chapters/**/*.qmd` chapter. The anchor is checked only
     inside FENCED, EXECUTED cells (Quarto only executes `` ```{python} ``
     fences; a bare `` ```python `` block is unexecuted prose, exactly the
     "mentioned in prose, not called in a live cell" case this gate must NOT
     credit) — so a mention in narrative markdown never counts.
  2. **CacheLane** — the family F/N pattern documented at the top of
     `jammi_cookbook/contracts.py`: a chapter reads a *committed* cache via
     `contracts.load_artifact("<dataset>....")` and asserts against the
     frozen golden; it deliberately does not re-derive the artifact live.
     The real call to the verb lives in the paired `scripts/build_<dataset>
     _cache.py` that produced that cache. Both anchors must resolve: the
     build script literally calls the verb, AND the chapter literally reads
     that dataset — so deleting either the call or the chapter's read
     breaks the row.
  3. **WrapperLane** — the verb is called through a `jammi_cookbook` shared
     helper (`rails.py` / `datasets.py`) that a live chapter cell invokes
     directly (e.g. a cell calls `rails.tenant(db, ...)`, whose body calls
     `db.tenant()` / `db.set_tenant(...)`). Both anchors must resolve: the
     helper literally calls the verb (comments/docstrings stripped first, so
     a docstring merely *mentioning* the call — e.g. the worked example in
     `build_finetune_regression_cache.py`'s own module docstring — cannot
     stand in for the real call site), AND the chapter literally invokes
     that helper in a live cell.

A verb not exercised by any of the three lanes today gets exactly one
**Deferred(reason, owner, date)** row — a reviewed, dated, honest gap, never
a silent one. `encode_query` is the one Deferred row as of this gate's
authoring: it is called only from `cookbook/recipes/` and
`cookbook/quickstart/` (a different consumer, outside `cookbook/book/`), not
from any chapter or its paired build script — verified by grep across the
whole `cookbook/` tree, not assumed.

The #351 quantized-serving surface was checked explicitly at authoring time
(2026-08-31): `crates/jammi-wire/src/fine_tune.rs`'s `FineTuneMethod` enum
has exactly one variant, `Lora` — QLoRA/GGUF (`crates/jammi-ai/src/model/
backend/gguf.rs`, `crates/jammi-ai/tests/it/gguf_qlora.rs`) is engine-internal
today and is NOT YET reachable through any Python binding
(`crates/jammi-python/`, `clients/python/jammi/`) or the wire protos
(`crates/jammi-wire/proto/`) — confirmed by grep, not assumed. It is
therefore not a member of `check_api_reference.py`'s `REQUIRED` /
`MODULE_FUNCTIONS` yet and gets NO row here (there is nothing shipped on the
Python surface to account for). When it lands on that surface — the
in-flight `cookbook/351-quantized-serving` branch's eventual target — the
same PR that adds it to `REQUIRED` will make this gate RED with an
UNACCOUNTED finding until a chapter cell (or a reviewed Deferred row naming
that work) is added; this gate does not pre-empt that branch's content, only
enforces the accounting once the surface actually ships.

## Fail-closed contract

  - A SHIPPED surface (a `REQUIRED` key or `MODULE_FUNCTIONS` entry) with no
    ACCOUNTING row is a non-zero exit naming it (the new-capability trigger).
  - An ACCOUNTING row naming a surface no longer in `REQUIRED` /
    `MODULE_FUNCTIONS` is a non-zero exit (stale row — check_api_reference.py
    changed and this file did not).
  - An ACCOUNTING row whose anchor(s) no longer resolve — the named chapter
    file is gone, the named build script / helper is gone, or the literal
    call-site / `load_artifact(...)` / helper-invocation substring is no
    longer present in a live cell / real code — is a non-zero exit (coverage
    cannot rot silently: deleting a chapter or a cell REDs this gate).
  - A surface accounted for twice is a non-zero exit (contradictory
    bookkeeping).
  - A parse failure against `check_api_reference.py` (missing `REQUIRED` /
    `MODULE_FUNCTIONS`, zero surfaces parsed) is a non-zero exit naming what
    could not be resolved.

Run: `python scripts/check_chapter_coverage.py` (from `cookbook/book/`, same
convention as `check_api_reference.py` / `check_citations.py`).
Self-test (proves the reconciliation bites — an unaccounted surface, a stale
row, a no-longer-resolving anchor, and a duplicated row are all caught, on
synthetic in-memory data): `python scripts/check_chapter_coverage.py --self-test`
Hermetic: reads only files in the working tree (or an in-memory fake reader
under `--self-test`); no network, no build, no GPU, no installed `jammi` wheel.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BOOK_ROOT = Path(__file__).resolve().parents[1]
CHAPTERS_DIR = BOOK_ROOT / "chapters"
SCRIPTS_DIR = BOOK_ROOT / "scripts"
LIB_DIR = BOOK_ROOT / "jammi_cookbook"
CHECK_API_REFERENCE = SCRIPTS_DIR / "check_api_reference.py"

Reader = Callable[[Path], str | None]


class CoverageError(Exception):
    """Uncomputable input (parse failure) — fails closed."""


# --------------------------------------------------------------------------- #
# SHIPPED — parsed (not hardcoded) from check_api_reference.py's own REQUIRED
# dict + MODULE_FUNCTIONS list, statically, via `ast`. This is the SAME
# enumeration `check_api_reference.py` uses to guard signature drift; reusing
# it here means the two gates can never disagree about what "shipped" means.
# --------------------------------------------------------------------------- #
def load_shipped_surfaces() -> set[str]:
    if not CHECK_API_REFERENCE.is_file():
        raise CoverageError(f"check_api_reference.py not found: {CHECK_API_REFERENCE}")

    src = CHECK_API_REFERENCE.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename=str(CHECK_API_REFERENCE))
    except SyntaxError as exc:
        raise CoverageError(f"check_api_reference.py failed to parse: {exc}") from exc

    required_node = None
    module_functions_node = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REQUIRED"
        ):
            required_node = node.value
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODULE_FUNCTIONS" for t in node.targets
        ):
            module_functions_node = node.value

    if required_node is None:
        raise CoverageError(
            "REQUIRED dict not found in check_api_reference.py — was it renamed or removed?"
        )
    if module_functions_node is None:
        raise CoverageError(
            "MODULE_FUNCTIONS list not found in check_api_reference.py — was it renamed or removed?"
        )

    try:
        required_keys = [ast.literal_eval(k) for k in required_node.keys]
        module_functions = ast.literal_eval(module_functions_node)
    except (ValueError, TypeError) as exc:
        raise CoverageError(
            f"failed to statically evaluate REQUIRED / MODULE_FUNCTIONS: {exc}"
        ) from exc

    surfaces = set(required_keys) | set(module_functions)
    if not surfaces:
        raise CoverageError("parsed zero surfaces out of check_api_reference.py")
    return surfaces


# --------------------------------------------------------------------------- #
# Exercise lanes — reviewed dataclasses, each carrying the file(s) + literal
# anchor(s) a human verified at authoring time and this gate re-verifies
# mechanically on every run.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DirectCell:
    """The verb is called directly inside a live ```{python}``` cell of one
    chapter. `chapter` is relative to `cookbook/book/chapters/`."""

    chapter: str
    anchor: str


@dataclass(frozen=True)
class CacheLane:
    """Family F/N: `chapter` reads a committed cache (`contracts.load_artifact
    ("<dataset>....")`) built by `script` (relative to `cookbook/book/
    scripts/`), which contains the real, live call to the verb. Both anchors
    must resolve."""

    script: str
    script_anchor: str
    chapter: str
    chapter_anchor: str


@dataclass(frozen=True)
class WrapperLane:
    """The verb is called through a `jammi_cookbook` shared helper (`helper`,
    relative to `cookbook/book/jammi_cookbook/`) that a live chapter cell
    invokes. Both anchors must resolve."""

    helper: str
    helper_anchor: str
    chapter: str
    chapter_anchor: str


@dataclass(frozen=True)
class Deferred:
    """A shipped surface with no chapter exercise yet — a conscious, visible,
    dated gap, never a silent one."""

    reason: str
    owner: str
    date: str


ExerciseEntry = DirectCell | CacheLane | WrapperLane | Deferred


# --------------------------------------------------------------------------- #
# ACCOUNTING — a reviewed LIST (not a dict), so a surface named twice is
# representable and this gate's own reconciliation catches it rather than
# Python dict-literal semantics silently keeping only the last entry (the
# same discipline `check_gpu_parity_matrix.py`'s SILICON_ACCOUNTING uses).
#
# 47 of the 48 REQUIRED / MODULE_FUNCTIONS surfaces are exercised today; the
# lone gap (`encode_query`) is a reviewed, dated Deferred row — see the
# module docstring's "Three exercise LANES" section for how each row below
# was verified (grep, not assumption) at authoring time (2026-08-31).
# --------------------------------------------------------------------------- #
ACCOUNTING: list[tuple[str, ExerciseEntry]] = [
    # -- module-level ---------------------------------------------------- #
    ("connect", DirectCell("01-construct/construct.qmd", "jammi.connect(")),
    # -- sources / setup --------------------------------------------------- #
    ("add_source", DirectCell("01-construct/construct.qmd", "db.add_source(")),
    ("list_sources", DirectCell("01-construct/construct.qmd", "db.list_sources(")),
    ("sql", DirectCell("01-construct/construct.qmd", "db.sql(")),
    # -- tenancy ------------------------------------------------------------ #
    # `tenant` has no direct `db.tenant()` call in any chapter cell; every
    # chapter reads/restores the current tenant scope through the shared
    # `rails.tenant(db, ...)` context-manager helper, whose body calls both
    # `db.tenant()` (to save the prior scope) and `db.set_tenant(...)` /
    # `db.tenant_scope(...)` (jammi_cookbook/rails.py:55-87).
    ("tenant", WrapperLane(
        "rails.py", "db.tenant()",
        "01-construct/construct.qmd", "rails.tenant(",
    )),
    ("set_tenant", DirectCell("11-tenancy/tenancy.qmd", "db.set_tenant(")),
    ("tenant_scope", DirectCell("14-eval-channels/eval-channels.qmd", "db.tenant_scope(")),
    # -- retrieval ------------------------------------------------------------ #
    ("rrf_fuse", DirectCell("10-retrieval/retrieval.qmd", "db.rrf_fuse(")),
    ("assemble_context", DirectCell("10-retrieval/retrieval.qmd", "db.assemble_context(")),
    # `search` is called directly in the 22-precision chapters (binary
    # search-recall sweeps); NOT in 07-bridge (whose only `.search(` hit is
    # the stdlib `re.search(...)`, not `Database.search` — verified by anchor
    # requiring the `db.` prefix so that false positive cannot creep back in).
    ("search", DirectCell("22-precision/binary-precision.qmd", "db.search(")),
    # -- graph primitives — family F/N: chapter 20-recompute's build script
    # (build_recompute_cache.py) is the single richest live-call site for
    # this whole cluster (graph build, propagation, recompute, staleness,
    # derivation, materialization verification all in one script); the
    # chapter reads the committed `artifacts/recompute/*` cache it produces.
    ("build_neighbor_graph", CacheLane(
        "build_recompute_cache.py", "db.build_neighbor_graph(",
        "20-recompute/recompute.qmd", 'load_artifact("recompute.',
    )),
    ("propagate_embeddings", CacheLane(
        "build_recompute_cache.py", "db.propagate_embeddings(",
        "20-recompute/recompute.qmd", 'load_artifact("recompute.',
    )),
    ("staleness", CacheLane(
        "build_recompute_cache.py", "db.staleness(",
        "20-recompute/recompute.qmd", 'load_artifact("recompute.',
    )),
    ("recompute", CacheLane(
        "build_recompute_cache.py", "db.recompute(",
        "20-recompute/recompute.qmd", 'load_artifact("recompute.',
    )),
    ("derives_from", CacheLane(
        "build_recompute_cache.py", "db.derives_from(",
        "20-recompute/recompute.qmd", 'load_artifact("recompute.',
    )),
    ("verify_materialization", CacheLane(
        "build_recompute_cache.py", "db.verify_materialization(",
        "20-recompute/recompute.qmd", 'load_artifact("recompute.',
    )),
    # -- point-in-time -------------------------------------------------------- #
    ("asof_join", DirectCell("19-point-in-time/point-in-time.qmd", "db.asof_join(")),
    # -- fine-tuning ----------------------------------------------------------- #
    ("fine_tune", CacheLane(
        "build_finetune_cache.py", "db.fine_tune(",
        "08-finetune-methods/finetune-methods.qmd", 'load_artifact("finetune.',
    )),
    ("fine_tune_graph", CacheLane(
        "build_finetune_cache.py", "db.fine_tune_graph(",
        "08-finetune-methods/finetune-methods.qmd", 'load_artifact("finetune.',
    )),
    ("infer", CacheLane(
        "build_finetune_regression_cache.py", "db.infer(",
        "15-finetune-regression/finetune-regression.qmd", 'load_artifact("finetune_regression.',
    )),
    # -- evaluation ------------------------------------------------------------ #
    ("eval_embeddings", CacheLane(
        "build_eval_cache.py", "db.eval_embeddings(",
        "14-eval-channels/eval-channels.qmd", 'load_artifact("eval.',
    )),
    ("eval_per_query", CacheLane(
        "build_eval_cache.py", "db.eval_per_query(",
        "14-eval-channels/eval-channels.qmd", 'load_artifact("eval.',
    )),
    ("eval_compare", CacheLane(
        "build_eval_cache.py", "db.eval_compare(",
        "14-eval-channels/eval-channels.qmd", 'load_artifact("eval.',
    )),
    ("eval_inference", CacheLane(
        "build_eval_cache.py", "db.eval_inference(",
        "14-eval-channels/eval-channels.qmd", 'load_artifact("eval.',
    )),
    ("eval_calibration", DirectCell("09-calibration/calibration.qmd", "db.eval_calibration(")),
    # -- conformal --------------------------------------------------------------- #
    ("conformalize", DirectCell("08-conformal/conformal.qmd", "db.conformalize(")),
    ("conformalize_interval", DirectCell(
        "08-conformal/conformal.qmd", "db.conformalize_interval(",
    )),
    ("conformalize_cqr", DirectCell("08-conformal/conformal.qmd", "db.conformalize_cqr(")),
    # -- context predictor (tier 04) ---------------------------------------------- #
    ("train_context_predictor", CacheLane(
        "build_arxiv_cache.py", "db.train_context_predictor(",
        "04-predict/predict.qmd", 'load_artifact("arxiv.',
    )),
    ("predict_with_context_predictor", CacheLane(
        "build_arxiv_cache.py", "db.predict_with_context_predictor(",
        "04-predict/predict.qmd", 'load_artifact("arxiv.',
    )),
    # -- model catalog (control plane) ---------------------------------------------- #
    ("list_models", CacheLane(
        "build_lifecycle_cache.py", "db.list_models(",
        "16-lifecycle/lifecycle.qmd", 'load_artifact("lifecycle.',
    )),
    ("describe_model", CacheLane(
        "build_lifecycle_cache.py", "db.describe_model(",
        "16-lifecycle/lifecycle.qmd", 'load_artifact("lifecycle.',
    )),
    ("delete_model", CacheLane(
        "build_lifecycle_cache.py", "db.delete_model(",
        "16-lifecycle/lifecycle.qmd", 'load_artifact("lifecycle.',
    )),
    # -- server info / embedding generation (scale chapter's build script calls
    # both) -------------------------------------------------------------------- #
    ("get_server_info", CacheLane(
        "build_scale_cache.py", "db.get_server_info(",
        "14-scale/scale.qmd", 'load_artifact("scale.',
    )),
    ("generate_embeddings", CacheLane(
        "build_scale_cache.py", "db.generate_embeddings(",
        "14-scale/scale.qmd", 'load_artifact("scale.',
    )),
    # -- unified client (U1) ------------------------------------------------------ #
    # `Session.supports(...)` is exercised via each backend's OWN bound name
    # (`embedded` / `remote`), never a bare `db.supports(` — the chapter
    # constructs both a Session over the embedded engine and one over a
    # remote handle to compare their capability surfaces side-by-side.
    ("supports", DirectCell("21-unified-client/unified-client.qmd", "embedded.supports(")),
    # -- eval channels / provenance ------------------------------------------------ #
    ("register_channel", DirectCell(
        "14-eval-channels/eval-channels.qmd", "db.register_channel(",
    )),
    ("add_channel_columns", DirectCell(
        "14-eval-channels/eval-channels.qmd", "db.add_channel_columns(",
    )),
    ("list_channels", DirectCell("14-eval-channels/eval-channels.qmd", "db.list_channels(")),
    # -- mutable companion tables --------------------------------------------------- #
    ("create_mutable_table", DirectCell(
        "12-feature-store/feature-store.qmd", "db.create_mutable_table(",
    )),
    ("list_mutable_tables", DirectCell(
        "12-feature-store/feature-store.qmd", "db.list_mutable_tables(",
    )),
    ("drop_mutable_table", CacheLane(
        "build_tenancy_h3_cache.py", "db.drop_mutable_table(",
        "18-tenancy-h3/tenancy-h3.qmd", 'load_artifact("tenancy_h3.',
    )),
    # -- trigger stream / topics ------------------------------------------------------ #
    ("register_topic", DirectCell("13-cdc/cdc.qmd", "db.register_topic(")),
    ("list_topics", DirectCell("13-cdc/cdc.qmd", "db.list_topics(")),
    ("publish_topic", DirectCell("13-cdc/cdc.qmd", "db.publish_topic(")),
    ("subscribe_collect", DirectCell("13-cdc/cdc.qmd", "db.subscribe_collect(")),
    ("drop_topic", CacheLane(
        "build_tenancy_h3_cache.py", "db.drop_topic(",
        "18-tenancy-h3/tenancy-h3.qmd", 'load_artifact("tenancy_h3.',
    )),
    # -- the one honest gap ------------------------------------------------------------- #
    # `encode_query` is called only from `cookbook/recipes/` and
    # `cookbook/quickstart/` (a separate, standalone-example consumer, not
    # part of `cookbook/book/`) — confirmed by grepping the whole `cookbook/`
    # tree for `encode_query` at authoring time: zero hits under
    # `cookbook/book/chapters/` or `cookbook/book/scripts/`. No chapter reads
    # a query-vector artifact whose build script called it either.
    ("encode_query", Deferred(
        reason=(
            "exercised only by cookbook/recipes/ and cookbook/quickstart/ "
            "(outside cookbook/book/, a separate standalone-example consumer); "
            "no cookbook/book/chapters/*.qmd cell or scripts/build_*_cache.py "
            "calls Database.encode_query — needs a chapter cell (e.g. a "
            "retrieval-chapter query-encode-then-search cell) or a build "
            "script + load_artifact pairing to close."
        ),
        owner="maintainers",
        date="2026-08-31",
    )),
]


# --------------------------------------------------------------------------- #
# anchor resolution — mechanical (substring match), against either a live
# ```{python}``` cell (chapters) or comment/docstring-stripped real code
# (scripts / helpers).
# --------------------------------------------------------------------------- #
_LIVE_CELL_RE = re.compile(r"```\{python\}[^\n]*\n(.*?)\n```", re.DOTALL)


def _live_python_cells(text: str) -> str:
    """Text of every EXECUTED ```{python}``` fence, joined. A bare ```python```
    fence (no braces) is unexecuted prose under Quarto and is deliberately
    excluded — exactly the "mentioned in prose, not called live" case this
    gate must not credit."""
    return "\n".join(_LIVE_CELL_RE.findall(text))


def _strip_python_prose(text: str) -> str:
    """Strip `#` comments and triple-quoted docstrings so a call site
    mentioned only in documentation (e.g. a module docstring's worked
    example) cannot stand in for a real call — mirrors the Rust
    comment-stripping discipline `check_gpu_parity_matrix.py` uses for its
    own enum/const parsing, applied to Python source."""
    text = re.sub(r"#.*", "", text)
    text = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
    return text


def _disk_reader(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _resolve_direct(name: str, entry: DirectCell, read: Reader) -> list[str]:
    path = CHAPTERS_DIR / entry.chapter
    text = read(path)
    if text is None:
        return [f"{name}: DirectCell chapter file not found: {entry.chapter}"]
    if entry.anchor not in _live_python_cells(text):
        return [
            f"{name}: DirectCell anchor {entry.anchor!r} not found in a live "
            f"```{{python}}``` cell of {entry.chapter}"
        ]
    return []


def _resolve_wrapper(name: str, entry: WrapperLane, read: Reader) -> list[str]:
    failures: list[str] = []
    helper_path = LIB_DIR / entry.helper
    helper_text = read(helper_path)
    if helper_text is None:
        failures.append(f"{name}: WrapperLane helper file not found: {entry.helper}")
    elif entry.helper_anchor not in _strip_python_prose(helper_text):
        failures.append(
            f"{name}: WrapperLane helper anchor {entry.helper_anchor!r} not found "
            f"in real code of {entry.helper} (comments/docstrings excluded)"
        )

    chapter_path = CHAPTERS_DIR / entry.chapter
    chapter_text = read(chapter_path)
    if chapter_text is None:
        failures.append(f"{name}: WrapperLane chapter file not found: {entry.chapter}")
    elif entry.chapter_anchor not in _live_python_cells(chapter_text):
        failures.append(
            f"{name}: WrapperLane chapter anchor {entry.chapter_anchor!r} not found "
            f"in a live ```{{python}}``` cell of {entry.chapter}"
        )
    return failures


def _resolve_cache_lane(name: str, entry: CacheLane, read: Reader) -> list[str]:
    failures: list[str] = []
    script_path = SCRIPTS_DIR / entry.script
    script_text = read(script_path)
    if script_text is None:
        failures.append(f"{name}: CacheLane script file not found: {entry.script}")
    elif entry.script_anchor not in _strip_python_prose(script_text):
        failures.append(
            f"{name}: CacheLane script anchor {entry.script_anchor!r} not found "
            f"in real code of {entry.script} (comments/docstrings excluded)"
        )

    chapter_path = CHAPTERS_DIR / entry.chapter
    chapter_text = read(chapter_path)
    if chapter_text is None:
        failures.append(f"{name}: CacheLane chapter file not found: {entry.chapter}")
    elif entry.chapter_anchor not in _live_python_cells(chapter_text):
        failures.append(
            f"{name}: CacheLane chapter anchor {entry.chapter_anchor!r} not found "
            f"in a live ```{{python}}``` cell of {entry.chapter}"
        )
    return failures


def _resolve_deferred(name: str, entry: Deferred) -> list[str]:
    failures: list[str] = []
    if not entry.reason.strip():
        failures.append(f"{name}: Deferred row has an empty reason")
    if not entry.owner.strip():
        failures.append(f"{name}: Deferred row has an empty owner")
    if not entry.date.strip():
        failures.append(f"{name}: Deferred row has an empty date")
    return failures


def resolve(name: str, entry: ExerciseEntry, read: Reader) -> list[str]:
    if isinstance(entry, DirectCell):
        return _resolve_direct(name, entry, read)
    if isinstance(entry, WrapperLane):
        return _resolve_wrapper(name, entry, read)
    if isinstance(entry, CacheLane):
        return _resolve_cache_lane(name, entry, read)
    if isinstance(entry, Deferred):
        return _resolve_deferred(name, entry)
    raise CoverageError(f"{name}: unknown ACCOUNTING entry type {type(entry)!r}")


DOCTRINE = (
    "the engine and cookbook co-evolve — a new public surface needs a chapter "
    "cell exercising it or a reviewed deferral; memory is not an accounting."
)


# --------------------------------------------------------------------------- #
# reconciliation — pure function over parsed/declared data, so `--self-test`
# can drive it with synthetic inputs and prove it actually bites.
# --------------------------------------------------------------------------- #
def reconcile(
    shipped: set[str],
    accounting: list[tuple[str, ExerciseEntry]],
    read: Reader,
) -> list[str]:
    failures: list[str] = []

    seen: dict[str, ExerciseEntry] = {}
    for name, entry in accounting:
        if name in seen:
            failures.append(
                f"surface `{name}` appears twice in ACCOUNTING — pick one row per "
                f"surface. {DOCTRINE}"
            )
            continue
        seen[name] = entry

    accounted = set(seen)
    for name in sorted(shipped - accounted):
        failures.append(
            f"surface `{name}` is SHIPPED (in check_api_reference.py's REQUIRED / "
            f"MODULE_FUNCTIONS) but has no ACCOUNTING row. {DOCTRINE}"
        )
    for name in sorted(accounted - shipped):
        failures.append(
            f"ACCOUNTING entry `{name}` names a surface no longer shipped (stale) "
            f"— remove it, or check_api_reference.py changed underneath it. {DOCTRINE}"
        )

    for name in sorted(accounted & shipped):
        failures.extend(resolve(name, seen[name], read))

    return failures


def print_matrix(shipped: set[str], accounting: list[tuple[str, ExerciseEntry]]) -> None:
    by_name = dict(accounting)
    exercised = 0
    deferred = 0
    print("Chapter-coverage matrix (surface -> accounting):")
    for name in sorted(shipped):
        entry = by_name.get(name)
        if entry is None:
            print(f"    !!!! UNACCOUNTED !!!! {name}")
            continue
        if isinstance(entry, Deferred):
            deferred += 1
            print(f"    DEFERRED    {name:<32} owner={entry.owner} date={entry.date}")
        else:
            exercised += 1
            print(f"    EXERCISED   {name:<32} [{type(entry).__name__}]")
    print(
        f"\nSummary: {exercised} EXERCISED, {deferred} DEFERRED out of "
        f"{len(shipped)} SHIPPED surface(s)."
    )


# --------------------------------------------------------------------------- #
# self-test — proves the reconciliation actually bites, on synthetic
# in-memory data (an injected fake reader; the real disk is untouched).
# --------------------------------------------------------------------------- #
def _fake_reader(mapping: dict[Path, str]) -> Reader:
    def read(path: Path) -> str | None:
        return mapping.get(path)

    return read


def self_test() -> int:
    failures: list[str] = []

    shipped = {"alpha", "beta", "gamma"}
    good_accounting: list[tuple[str, ExerciseEntry]] = [
        ("alpha", DirectCell("synthetic.qmd", "db.alpha(")),
        ("beta", CacheLane(
            "build_synth_cache.py", "db.beta(",
            "synthetic.qmd", 'load_artifact("synth.',
        )),
        ("gamma", Deferred("synthetic deferral", "maintainers", "2026-08-31")),
    ]
    mapping = {
        CHAPTERS_DIR / "synthetic.qmd": (
            "```{python}\n"
            "db.alpha()\n"
            'contracts.load_artifact("synth.thing")\n'
            "```\n"
        ),
        SCRIPTS_DIR / "build_synth_cache.py": "db.beta(x)\n",
    }
    read = _fake_reader(mapping)

    clean = reconcile(shipped, good_accounting, read)
    if clean:
        failures.append(
            f"self-test FAILED: a fully-accounted synthetic set reported failures: {clean}"
        )

    # (1) unaccounted surface — the new-capability trigger.
    shipped_extra = shipped | {"delta"}
    unaccounted = reconcile(shipped_extra, good_accounting, read)
    if not any(
        "`delta`" in f and "no ACCOUNTING row" in f for f in unaccounted
    ):
        failures.append(
            f"self-test FAILED: a shipped-but-unaccounted surface was not surfaced: {unaccounted}"
        )

    # (2) stale row — a surface gone from REQUIRED/MODULE_FUNCTIONS.
    accounting_stale = good_accounting + [
        ("epsilon", Deferred("stale", "maintainers", "2026-08-31"))
    ]
    stale = reconcile(shipped, accounting_stale, read)
    if not any("`epsilon`" in f and "no longer shipped" in f for f in stale):
        failures.append(f"self-test FAILED: a stale ACCOUNTING row was not surfaced: {stale}")

    # (3) an exercised_by anchor that no longer resolves — the chapter cell
    # calling the old name was deleted/renamed, but the row was not updated.
    accounting_broken = list(good_accounting)
    accounting_broken[0] = ("alpha", DirectCell("synthetic.qmd", "db.renamed_alpha("))
    broken = reconcile(shipped, accounting_broken, read)
    if not any("alpha" in f and "not found in a live" in f for f in broken):
        failures.append(f"self-test FAILED: a non-resolving anchor was not surfaced: {broken}")

    # (4) a surface accounted for twice — contradictory bookkeeping.
    accounting_dup = good_accounting + [
        ("alpha", Deferred("dup", "maintainers", "2026-08-31"))
    ]
    dup = reconcile(shipped, accounting_dup, read)
    if not any("appears twice" in f for f in dup):
        failures.append(f"self-test FAILED: a duplicated ACCOUNTING row was not surfaced: {dup}")

    if failures:
        for f in failures:
            print(f, file=sys.stderr)
        print("chapter-coverage self-test: FAIL", file=sys.stderr)
        return 1
    print(
        "chapter-coverage self-test: OK — an unaccounted surface, a stale row, a "
        "non-resolving anchor, and a duplicated row are all caught."
    )
    return 0


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return self_test()

    try:
        shipped = load_shipped_surfaces()
    except CoverageError as exc:
        print(f"chapter-coverage: FAIL (uncomputable) — {exc}", file=sys.stderr)
        return 1

    failures = reconcile(shipped, ACCOUNTING, _disk_reader)

    print_matrix(shipped, ACCOUNTING)

    if failures:
        print("\nchapter-coverage: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(f"\nchapter-coverage: {len(failures)} finding(s). {DOCTRINE}", file=sys.stderr)
        return 1

    print(
        "\nchapter-coverage: PASS — every SHIPPED surface is exercised by a live "
        "chapter cell (direct, cache-lane, or wrapper-lane) or consciously "
        "deferred by name; no new capability can ship with silent, unaccounted "
        "cookbook coverage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
