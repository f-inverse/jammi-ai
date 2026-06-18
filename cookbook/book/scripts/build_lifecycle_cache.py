#!/usr/bin/env python3
"""Emit the model-catalog cache — CPU, dual-transport, hermetic.

The engine↔cookbook validator for the `§3.6` model-catalog surface: the model
verbs (`list_models` / `describe_model` / `delete_model`) that the engine carries
on BOTH the embedded `jammi_ai.Database` and the remote
`jammi_client.RemoteDatabase`. The engine's catalog lets you **see** the models it
resolves and trains, and **clean them up**; this script measures that surface as a
real, asserted property — the **referential-integrity matrix** for `delete_model`,
validated `remote == embedded` for every observable.

It runs the whole catalog interaction on BOTH transports and freezes the matrix:

* **register** — the only public path that puts a model row in the catalog is
  TRAINING. A tiny CPU `fine_tune(method="lora", task="text_embedding")` over a
  12-row in-memory `(anchor, positive)` pairs corpus with the engine's public
  `tiny_modernbert` fixture registers two rows: the base model (path-keyed) at
  submission and the fine-tuned model (`jammi:fine-tuned:<uuid>`) on completion.
  CPU/hermetic — the candle backend logs `CUDA requested … running on CPU`; no
  GPU, no keystone corpus.
* **see** — `describe_model` / `list_models` reflect the registered models as the
  minimal client-facing projection.
* **the referential-integrity matrix** for `delete_model`:
  - **delete-while-referenced** → the typed `ModelReferenced` error. Every model
    in the catalog is trained-and-referenced (a fine-tuned model by
    `training_jobs.output_model_id`; its base by `training_jobs.base_model_id`),
    so this is the headline guarantee. Embedded raises `RuntimeError("Model
    referenced: … still referenced by …")`; the wire raises
    `StatusCode.FAILED_PRECONDITION`. Normalized class: `referenced`.
  - **delete-absent without `if_exists`** → **NotFound** (the typed `ModelNotFound`
    — the wire status is `NOT_FOUND`, NOT `INVALID_ARGUMENT`). Embedded raises
    `RuntimeError("Model not found: …")`. Normalized class: `not_found`.
  - **delete-absent with `if_exists=True`** → a no-op (returns None) on both.
  - the delete-unreferenced-succeeds path is simply **unreachable** here: every
    model in the catalog is trained-and-referenced, so there is no bare
    unreferenced model to delete. Measured as a real property of the catalog:
    `every_catalog_model_is_referenced = True`.

`remote == embedded` is asserted live for every observable: the model projections
(key-for-key, with the per-run UUID `model_id` excluded — it differs across
instances by construction) and the NORMALIZED error class for each delete case
(the two transports raise different Python exception TYPES — embedded
`RuntimeError`, wire `grpc.RpcError` — so the honest cross-transport contract is
the error CLASS, derived from each native error). The verdict is recorded in
`artifacts/lifecycle/lifecycle.json`; the committed matrix freezes to
`golden_metrics.json` + `checksums.json`.

Usage::

    JAMMI_SERVER_BIN=/mnt/sagemaker-nvme/jammi-target/debug/jammi-server \\
        python scripts/build_lifecycle_cache.py --fixtures-root /path/to/jammi-ai

The `--fixtures-root` is the engine checkout carrying `tests/fixtures/tiny_modernbert`
(defaults to the `JAMMI_FIXTURES_ROOT` env var). This is an emit-only script: it
imports a build of the engine wheel + client; PR CI never runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import jammi_ai
import jammi_client
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "lifecycle"

# The per-run identifier whose VALUE differs across engine instances by
# construction: the fine-tuned `model_id` is `jammi:fine-tuned:<fresh-uuid>`. Its
# KEY stays pinned by the projection comparison; only its value is excluded from
# the cross-transport / committed comparison.
_INSTANCE_KEYS = {"model_id"}

# The minimal client-facing model projection — the keys a `list_models` /
# `describe_model` entry carries on BOTH transports (pinned by the engine's own
# `test_conformance.py::_MODEL_DICT_KEYS`; reproduced here as the parity contract).
_MODEL_KEYS = {"model_id", "backend", "task", "status"}

# A tiny supervised pairs corpus — `(anchor, positive)` triggers the engine's
# `pairs` training format (MNRL). 12 rows trains in seconds on CPU. Deterministic
# content: a re-emit registers the same shape (the model_id UUID aside).
_PAIRS = {
    "anchor": ["a graph", "a node", "an edge", "a model"] * 3,
    "positive": ["a network", "a vertex", "a link", "a net"] * 3,
}


# --------------------------------------------------------------------------- #
# Error normalisation — the honest cross-transport contract for a typed error
# --------------------------------------------------------------------------- #


def _classify_error(exc: BaseException) -> str:
    """Normalise a transport-native delete error to its CLASS.

    The two transports raise different Python exception TYPES by construction:
    the embedded engine raises `RuntimeError` with a message; the gRPC client
    raises `grpc.RpcError` carrying a `StatusCode`. The honest `remote ==
    embedded` observable for a typed error is the error CLASS, derived from each
    native error — `referenced` (the `ModelReferenced` guard) or `not_found` (the
    `ModelNotFound` condition). An unrecognised error is returned verbatim so a
    real divergence (e.g. the not-found case regressing to invalid-argument)
    surfaces loudly rather than being silently bucketed."""
    code = getattr(exc, "code", None)
    if callable(code):  # grpc.RpcError → StatusCode
        status = code()
        name = getattr(status, "name", str(status))
        if name == "FAILED_PRECONDITION":
            return "referenced"
        if name == "NOT_FOUND":
            return "not_found"
        return f"wire:{name}"
    msg = str(exc).lower()
    if "still referenced" in msg or "model referenced" in msg:
        return "referenced"
    if "not found" in msg:
        return "not_found"
    return f"embed:{type(exc).__name__}:{msg[:60]}"


def _delete_outcome(db, model_id: str, *, if_exists: bool = False) -> dict:
    """Run `delete_model` and capture the OUTCOME as a comparable observable:
    either a clean no-op (`{"raised": False}`) or the normalised error class
    (`{"raised": True, "error_class": …}`). Never lets the exception escape — the
    outcome IS the measurement (the matrix cell), compared across transports."""
    try:
        db.delete_model(model_id, if_exists=if_exists)
        return {"raised": False, "error_class": None}
    except Exception as exc:  # noqa: BLE001 — the error class is the observable
        return {"raised": True, "error_class": _classify_error(exc)}


# --------------------------------------------------------------------------- #
# The catalog run (one transport)
# --------------------------------------------------------------------------- #


def _project(model: dict | None) -> dict | None:
    """A model descriptor reduced to the minimal projection with the instance-
    minted `model_id` stripped — the byte-stable, cross-transport-comparable form
    (the model_id is a per-run UUID; its KEY presence is pinned separately)."""
    if model is None:
        return None
    assert set(model) == _MODEL_KEYS, f"projection keys {set(model)} != {_MODEL_KEYS}"
    return {k: v for k, v in model.items() if k not in _INSTANCE_KEYS}


def _train_one(db, corpus_path: str, base_model: str, tag: str) -> str:
    """Register a model the only public way — a tiny CPU fine_tune — returning the
    fine-tuned model_id. Trains on CPU in seconds (candle falls back off CUDA)."""
    src = f"pairs_{tag}"
    db.add_source(src, url=f"file://{corpus_path}", format="parquet")
    job = db.fine_tune(
        source=src,
        base_model=base_model,
        columns=["anchor", "positive"],
        method="lora",
        task="text_embedding",
        epochs=1,
        batch_size=4,
        seed=0,
    )
    job.wait()
    return job.model_id


def run_catalog(db, corpus_path: str, base_model: str, tag: str) -> dict:
    """Run the whole catalog interaction on one transport and return the
    observable matrix. The SAME call shapes run on both transports — the verb
    surface is transport-agnostic; only the native error TYPE differs (normalised
    away).

    `tag` keeps the registered sources disjoint on the module-shared remote server
    (the embedded peer is isolated by its own temp catalog)."""
    # --- register (via training — the only public path) ---------------------- #
    model_id = _train_one(db, corpus_path, base_model, tag)
    base_id = base_model.removeprefix("local:")
    # --- see (describe / list reflect the registered models) ----------------- #
    registered = _project(db.describe_model(model_id))
    list_after_register = sorted(
        (_project(m) for m in db.list_models()), key=lambda m: m["status"] + m["task"]
    )
    # the headline referential property: every model in the catalog is
    # trained-and-referenced (the fine-tuned by training_jobs.output_model_id, its
    # base by training_jobs.base_model_id) — so neither is hard-deletable.
    ft_delete_referenced = _delete_outcome(db, model_id)
    base_delete_referenced = _delete_outcome(db, base_id)
    every_catalog_model_is_referenced = (
        ft_delete_referenced["error_class"] == "referenced"
        and base_delete_referenced["error_class"] == "referenced"
    )

    # --- the referential-integrity matrix for delete ------------------------- #
    # delete-absent WITHOUT if_exists → NotFound (the typed ModelNotFound: the wire
    # status is NOT_FOUND, never INVALID_ARGUMENT).
    absent_id = "jammi:fine-tuned:00000000-0000-4000-8000-000000000000"
    delete_absent_strict = _delete_outcome(db, absent_id)
    # delete-absent WITH if_exists=True → a no-op (returns None, no raise).
    delete_absent_if_exists = _delete_outcome(db, absent_id, if_exists=True)
    # delete-while-referenced (re-checked on the fine-tuned model) → ModelReferenced.
    delete_referenced = _delete_outcome(db, model_id)

    return {
        "registered": registered,
        "list_after_register": list_after_register,
        "ft_delete_referenced": ft_delete_referenced,
        "base_delete_referenced": base_delete_referenced,
        "every_catalog_model_is_referenced": every_catalog_model_is_referenced,
        "delete_absent_strict": delete_absent_strict,
        "delete_absent_if_exists": delete_absent_if_exists,
        "delete_referenced": delete_referenced,
    }


# --------------------------------------------------------------------------- #
# Parity (remote == embedded for every observable)
# --------------------------------------------------------------------------- #


def _parity(name: str, embedded, remote) -> dict:
    """The live cross-transport parity verdict for one observable. Raises (loudly)
    on a real divergence — that is an ENGINE bug the validator is meant to surface,
    never paper over."""
    equal = embedded == remote
    verdict = {"observable": name, "equal": equal}
    if not equal:
        verdict["embedded"] = embedded
        verdict["remote"] = remote
        raise AssertionError(f"remote != embedded for {name}: {embedded!r} != {remote!r}")
    return verdict


# --------------------------------------------------------------------------- #
# Remote server lifecycle (mirrors the engine conftest live_server)
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """A real CPU `jammi-server` on a free port, readiness-polled via a client
    handshake, torn down on exit — the same shape the engine's conftest uses."""

    def __init__(self, server_bin: str):
        self.server_bin = server_bin
        self.proc = None
        self.endpoint = None
        self._artifact_dir = None

    def __enter__(self) -> str:
        self._artifact_dir = tempfile.mkdtemp(prefix="jammi_srv_lifecycle_")
        flight_port = _free_port()
        health_port = _free_port()
        env = dict(os.environ)
        env["JAMMI_ARTIFACT_DIR"] = self._artifact_dir
        env["JAMMI_SERVER__FLIGHT_LISTEN"] = f"127.0.0.1:{flight_port}"
        env["JAMMI_SERVER__HEALTH_LISTEN"] = f"127.0.0.1:{health_port}"
        env["JAMMI_SERVER__SERVICES"] = "all"
        self.proc = subprocess.Popen(
            [self.server_bin], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self.endpoint = f"grpc://127.0.0.1:{flight_port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read().decode(errors="replace") if self.proc.stdout else ""
                raise RuntimeError(f"jammi-server exited early:\n{out}")
            try:
                handshake = jammi_client.connect(self.endpoint)
                handshake.get_server_info()
                handshake.close()
                return self.endpoint
            except Exception:
                time.sleep(0.25)
        self.proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #


def _fixtures_root(arg: str | None) -> Path:
    root = arg or os.environ.get("JAMMI_FIXTURES_ROOT")
    if not root:
        raise SystemExit(
            "pass --fixtures-root (or set JAMMI_FIXTURES_ROOT) to the engine "
            "checkout carrying tests/fixtures/tiny_modernbert"
        )
    p = Path(root).resolve()
    if not (p / "tests" / "fixtures" / "tiny_modernbert" / "config.json").exists():
        raise SystemExit(f"--fixtures-root {p} has no tests/fixtures/tiny_modernbert")
    return p


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums() -> None:
    sums = {
        p.name: _checksum(p)
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


# The observables compared across transports (the model_id UUID is excluded inside
# the projections, so these are byte-stable and instance-agnostic).
_OBSERVABLES = (
    "registered",
    "list_after_register",
    "ft_delete_referenced",
    "base_delete_referenced",
    "every_catalog_model_is_referenced",
    "delete_absent_strict",
    "delete_absent_if_exists",
    "delete_referenced",
)


def emit(fixtures_root: Path, server_bin: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    base_model = f"local:{fixtures_root / 'tests' / 'fixtures' / 'tiny_modernbert'}"

    with tempfile.TemporaryDirectory() as work:
        corpus_path = str(Path(work) / "pairs.parquet")
        pq.write_table(pa.table(_PAIRS), corpus_path)

        # --- embedded transport (the canonical matrix) ---------------------- #
        with tempfile.TemporaryDirectory() as catalog:
            embedded = jammi_ai.connect(f"file://{catalog}")
            print("== embedded engine: model catalog ==", flush=True)
            embedded_matrix = run_catalog(embedded, corpus_path, base_model, tag="emb")

        # --- remote transport (live grpc:// parity) ------------------------- #
        with LiveServer(server_bin) as endpoint:
            print(f"== remote engine up at {endpoint} ==", flush=True)
            remote = jammi_client.connect(endpoint)
            try:
                print("== remote engine: model catalog ==", flush=True)
                remote_matrix = run_catalog(remote, corpus_path, base_model, tag="rem")
            finally:
                remote.close()

    # --- the live remote == embedded parity verdict ------------------------- #
    parity = [
        _parity(name, embedded_matrix[name], remote_matrix[name]) for name in _OBSERVABLES
    ]
    print(f"== remote == embedded parity: PASS for all {len(parity)} observables ==", flush=True)

    # --- the frozen golden matrix (the measured verdicts) ------------------- #
    m = embedded_matrix
    golden = {
        # registration: a fresh model is registered
        "register.status_registered": {
            "value": 1.0 if m["registered"]["status"] == "registered" else 0.0, "tol": 0.0
        },
        # the referential-integrity matrix (the headline)
        "delete.referenced_raises": {
            "value": 1.0 if m["delete_referenced"]["error_class"] == "referenced" else 0.0,
            "tol": 0.0,
        },
        "delete.every_catalog_model_referenced": {
            "value": 1.0 if m["every_catalog_model_is_referenced"] else 0.0, "tol": 0.0
        },
        "delete.absent_strict_not_found": {
            "value": 1.0 if m["delete_absent_strict"]["error_class"] == "not_found" else 0.0,
            "tol": 0.0,
        },
        "delete.absent_if_exists_noop": {
            "value": 1.0 if m["delete_absent_if_exists"]["raised"] is False else 0.0, "tol": 0.0
        },
        # the cross-transport parity verdict, as a metric
        "parity.all_observables_equal": {
            "value": 1.0 if all(p["equal"] for p in parity) else 0.0, "tol": 0.0
        },
    }

    # --- commit the embedded-canonical matrix ------------------------------- #
    (ARTIFACTS / "matrix.json").write_text(json.dumps(embedded_matrix, indent=2, sort_keys=True))
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))

    catalog_record = {
        "purpose": (
            "The engine↔cookbook validator for the §3.6 model-catalog surface: the "
            "model verbs (list/describe/delete) run on BOTH the embedded engine and a "
            "live remote grpc:// jammi-server, asserted remote == embedded live (the "
            "real cross-transport parity check). The committed matrix is the "
            "embedded-canonical one; PR CI reads it and asserts the verdicts-to-golden, "
            "never a static re-diff of two artifacts."
        ),
        "registration_path": (
            "TRAINING — the only public path that puts a model row in the catalog. A "
            "tiny CPU fine_tune(method='lora', task='text_embedding') over a 12-row "
            "in-memory (anchor, positive) pairs corpus with the engine's public "
            "tiny_modernbert fixture registers the model. CPU/hermetic — no GPU, no "
            "keystone corpus."
        ),
        "transports": [
            "embedded (file://, in-process)",
            "remote (grpc://, live jammi-server)",
        ],
        "parity_observable": (
            "model projections (model_id UUID excluded) and the NORMALIZED delete-error "
            "CLASS (embedded RuntimeError message / wire StatusCode → referenced | "
            "not_found) — the honest cross-transport contract for a typed error."
        ),
        "parity_verdict": "remote == embedded for every catalog observable",
        "parity": parity,
        "catalog_property": (
            "Every model in the catalog is trained-and-referenced (the fine-tuned by "
            "training_jobs.output_model_id, its base by training_jobs.base_model_id), so "
            "delete_model on either is correctly refused with the typed ModelReferenced "
            "guard. The delete-unreferenced-succeeds path is therefore unreachable here: "
            "there is no bare unreferenced model in the catalog to delete. A measured "
            "property of the engine's catalog, not a gap."
        ),
        "measured": {
            "registered_status": m["registered"]["status"],
            "registered_backend": m["registered"]["backend"],
            "registered_task": m["registered"]["task"],
            "delete_referenced_class": m["delete_referenced"]["error_class"],
            "delete_absent_strict_class": m["delete_absent_strict"]["error_class"],
            "delete_absent_if_exists_raised": m["delete_absent_if_exists"]["raised"],
            "every_catalog_model_is_referenced": m["every_catalog_model_is_referenced"],
        },
    }
    (ARTIFACTS / "lifecycle.json").write_text(
        json.dumps(catalog_record, indent=2, sort_keys=True)
    )

    _write_checksums()

    print("\n=== model catalog, measured (embedded canonical) ===", flush=True)
    print(f"  register: status={m['registered']['status']} "
          f"backend={m['registered']['backend']} task={m['registered']['task']}", flush=True)
    print(f"  delete referenced → error_class={m['delete_referenced']['error_class']} "
          f"(every catalog model referenced={m['every_catalog_model_is_referenced']})", flush=True)
    print(f"  delete absent (strict) → error_class={m['delete_absent_strict']['error_class']} "
          f"(the typed ModelNotFound — NOT invalid-argument)", flush=True)
    print(f"  delete absent (if_exists=True) → raised={m['delete_absent_if_exists']['raised']} "
          f"(no-op)", flush=True)
    print(f"  parity: remote == embedded for all {len(parity)} observables", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", default=None,
                    help="engine checkout with tests/fixtures/tiny_modernbert "
                         "(or set JAMMI_FIXTURES_ROOT)")
    ap.add_argument("--server-bin", default=os.environ.get("JAMMI_SERVER_BIN"),
                    help="built CPU jammi-server binary (or set JAMMI_SERVER_BIN)")
    args = ap.parse_args()
    if not args.server_bin or not os.path.exists(args.server_bin):
        raise SystemExit("pass --server-bin (or set JAMMI_SERVER_BIN) to a built jammi-server")
    emit(_fixtures_root(args.fixtures_root), args.server_bin)


if __name__ == "__main__":
    main()
