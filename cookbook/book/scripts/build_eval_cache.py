#!/usr/bin/env python3
"""Emit the eval + provenance-channel cache (chapter 14) — CPU, dual-transport.

The engine↔cookbook validator for the H1 residual: it exercises the newly-landed
remote eval + channel surface — the verbs that, until now, lived only on the
embedded ``Database`` (the ch05 R1 gap) — across **both** transports and proves
the wire path agrees with the in-process engine. It runs once, on a build of the
engine, and commits the embedded-canonical reports + goldens the chapter reads.

What it does, per the engine's own deterministic public fixtures (NOT the arxiv
keystone corpus — those numbers stay the ch05 numpy fold; this measures the wire
SURFACE on the fixtures the engine's live tests drive):

* **eval_embeddings** — the 20-row ``patents`` abstracts embedded with the
  deterministic ``tiny_modernbert`` encoder, scored against ``golden_relevance``;
* **eval_per_query** — the persisted per-query rows read back for that run;
* **eval_inference (classification)** — the deterministic ``tiny_modernbert_classifier``
  over the same corpus against ``tiny_labels``;
* **eval_inference (ner)** — the deterministic ``tiny_modernbert_ner`` over the
  ``tiny_ner_corpus`` against the ``tiny_ner_gold`` char-offset spans (the Python +
  remote NER eval coverage the engine's own Python live tests do not carry);
* **eval_compare** — TWO cases: a self-comparison (the determinism anchor — every
  delta exactly 0.0, significance present) AND a genuine two-table comparison
  (``tiny_modernbert`` vs ``tiny_bert`` over the same corpus — a non-degenerate,
  non-zero delta with a real significance block);
* **the channel sequence** — register two generic provenance channels, append
  columns, list, and the #170 tenant-isolation / non-collision property.

Every verb and the channel sequence run on BOTH the embedded (in-process) engine
and a live remote ``grpc://`` ``jammi-server``, and this script asserts
**remote == embedded** live — ``_shape`` equality plus value-closeness (tol 1e-9,
with the NER ``confidence`` field at its true f32-relative precision since the
engine computes that span score in f32), sorting ``per_query`` / ``per_record`` by
id and excluding the instance-minted keys (``eval_run_id`` / ``table_name``). That
is the real cross-transport parity check
(the ported engine-test assertion), recorded with its verdict in
``artifacts/eval/eval.json``. The chapter and PR CI never re-diff two static
artifacts — they read the embedded-canonical reports and assert aggregates-to-golden.

Determinism: the committed reports have their ``per_query`` / ``per_record`` lists
sorted by id and the instance-minted keys stripped, so they are byte-stable. The
golden scalars are frozen in ``golden_metrics.json``.

Usage::

    # the remote arm needs a built CPU jammi-server on the wire:
    JAMMI_SERVER_BIN=/path/to/jammi-server \\
        python scripts/build_eval_cache.py --fixtures-root /path/to/jammi-ai

The ``--fixtures-root`` is the engine checkout whose ``tests/fixtures`` and
``cookbook/fixtures`` carry the deterministic public fixtures (defaults to the
``JAMMI_FIXTURES_ROOT`` env var). This is an emit-only script: it imports a build
of the engine wheel + client; PR CI never runs it.
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

import jammi_cookbook  # noqa: F401  # applies the determinism env on import

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "eval"

# Per-run identifiers whose VALUES differ across engine instances by
# construction: `eval_run_id` is a fresh UUID per run; `table_name` embeds the
# embedding table's creation timestamp. Their KEYS stay pinned by `_shape`; only
# their values are excluded from the cross-transport / committed comparison.
_INSTANCE_KEYS = {"eval_run_id", "table_name"}

# Cross-instance closeness bound. The fixture encoder is deterministic, so the
# metrics agree far tighter than this; the bound only absorbs accumulation-order
# noise, never a ranking difference (the same bound the engine's live test uses).
_TOL = 1e-9

# Per-field float tolerance overrides, keyed by dict key. A field carried on the
# wire at a narrower precision than f64 cannot agree across transports to the f64
# `_TOL`: the engine computes a NER span's `confidence` in f32, so both transports
# carry the same f32 value but widen it to a Python float by two different
# conventions (the embed wheel serializes the f32 through serde's shortest-decimal;
# the gRPC client reads a 32-bit proto `float` back as its exact f64 widening). The
# two f64s agree only to f32 precision (~1e-7), so `confidence` is compared at an
# f32-relative bound, never the f64 `_TOL`. This is the field's true cross-transport
# contract, not a slackened metric — it mirrors the engine's own
# `_FLOAT_TOL_BY_KEY` in clients/python/tests/test_remote_eval_live.py.
_F32_REL = 1e-6
_FLOAT_TOL_BY_KEY = {"confidence": _F32_REL}

K = 10


# --------------------------------------------------------------------------- #
# Report normalisation + parity (ported from the engine's live eval test)
# --------------------------------------------------------------------------- #


def _shape(obj):
    """The key structure of a report: dicts map to their keyed sub-shapes, lists
    to element shapes, scalars to their type name — so two reports with the same
    shape carry the same keys at every level."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shape(v) for v in obj]
    return type(obj).__name__


def _assert_values_close(a, b, path="", key=None):
    """Recursive value equality with `_TOL`-close floats, skipping the
    instance-minted identifier values (their presence is pinned by `_shape`).

    A float under a key in `_FLOAT_TOL_BY_KEY` is compared at that key's
    f32-relative bound instead of the f64 `_TOL` — see the table for why
    `confidence` is genuinely f32-precision across transports."""
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} != {set(b)}"
        for k in a:
            if k in _INSTANCE_KEYS:
                continue
            _assert_values_close(a[k], b[k], f"{path}.{k}", key=k)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_values_close(x, y, f"{path}[{i}]", key=key)
    elif isinstance(a, float):
        rel = _FLOAT_TOL_BY_KEY.get(key)
        if rel is not None:
            bound = rel * max(abs(a), abs(b), 1.0)
            assert abs(a - b) <= bound, f"{path}: {a} != {b} (Δ {abs(a - b)} > {bound})"
        else:
            assert abs(a - b) <= _TOL, f"{path}: {a} != {b} (Δ {abs(a - b)})"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _sorted_by(records: list[dict], key: str) -> list[dict]:
    """A copy of a record list ordered by `key` — the runner emits per-query /
    per-record rows in retrieval order (no ORDER BY; engine output is never
    reshaped to make a test pass), so two engine instances may interleave the
    same records differently. Normalising on the ASSERTION side pairs the same
    record across transports without touching the engine."""
    return sorted(records, key=lambda r: r[key])


def _normalize_embedding_report(report: dict) -> dict:
    """An embedding report with `per_query` ordered by `query_id`."""
    out = dict(report)
    out["per_query"] = _sorted_by(report["per_query"], "query_id")
    return out


def _normalize_inference_report(report: dict) -> dict:
    """An inference report with `per_record` ordered by `record_id`.

    Both the classification and NER per-record variants carry `record_id`, so it
    is the stable sort key — the runner emits the records in prediction order, not
    a stable sort, so two engine instances may interleave them differently."""
    out = dict(report)
    out["per_record"] = _sorted_by(report["per_record"], "record_id")
    return out


def _normalize_compare_report(report: dict) -> dict:
    """A compare report with each per-table entry's nested `per_query` ordered."""
    out = dict(report)
    out["per_table"] = []
    for entry in report["per_table"]:
        e = dict(entry)
        e["embedding_eval"] = _normalize_embedding_report(entry["embedding_eval"])
        out["per_table"].append(e)
    return out


def _strip_instance_keys(obj):
    """A deep copy of a report with the instance-minted keys removed — the
    byte-stable committed form. `_shape` pins those keys' presence separately;
    the chapter asserts the committed report's shape without them."""
    if isinstance(obj, dict):
        return {k: _strip_instance_keys(v) for k, v in obj.items() if k not in _INSTANCE_KEYS}
    if isinstance(obj, list):
        return [_strip_instance_keys(v) for v in obj]
    return obj


def _parity(name: str, embedded: dict | list, remote: dict | list) -> dict:
    """The live cross-transport parity verdict for one report: `_shape` equality
    plus `_TOL`-close values, excluding instance keys. Returns a record for
    eval.json; raises (loudly) on a real divergence — that is an ENGINE bug the
    validator is meant to surface, never paper over."""
    shape_ok = _shape(embedded) == _shape(remote)
    values_ok = True
    diff = None
    try:
        _assert_values_close(embedded, remote)
    except AssertionError as e:
        values_ok = False
        diff = str(e)
    verdict = {"verb": name, "shape_equal": shape_ok, "values_close": values_ok, "tol": _TOL}
    if diff is not None:
        verdict["diff"] = diff
    if not (shape_ok and values_ok):
        raise AssertionError(
            f"remote != embedded for {name}: shape_equal={shape_ok} "
            f"values_close={values_ok} diff={diff}"
        )
    return verdict


# --------------------------------------------------------------------------- #
# Fixture registration (fresh, source-bound — never the bare arxiv matrix)
# --------------------------------------------------------------------------- #


def _fixtures_root(arg: str | None) -> Path:
    root = arg or os.environ.get("JAMMI_FIXTURES_ROOT")
    if not root:
        raise SystemExit(
            "pass --fixtures-root (or set JAMMI_FIXTURES_ROOT) to the engine "
            "checkout carrying tests/fixtures + cookbook/fixtures"
        )
    p = Path(root).resolve()
    if not (p / "tests" / "fixtures" / "patents.parquet").exists():
        raise SystemExit(f"--fixtures-root {p} has no tests/fixtures/patents.parquet")
    return p


class Fixtures:
    """The deterministic public fixture URLs the emit registers fresh on each
    transport — source-bound tables built inside the engine, never the keystone's
    bare arxiv vector matrix (the exact trap that forced the ch05 numpy fold)."""

    def __init__(self, root: Path):
        eng = root / "tests" / "fixtures"
        cb = root / "cookbook" / "fixtures"
        self.patents_url = f"file://{eng / 'patents.parquet'}"
        self.golden_rel_url = f"file://{eng / 'golden_relevance.csv'}"
        self.tiny_modernbert = f"local:{eng / 'tiny_modernbert'}"
        self.tiny_bert = f"local:{cb / 'tiny_bert'}"
        self.classifier = f"local:{cb / 'tiny_modernbert_classifier'}"
        self.labels_url = f"file://{cb / 'tiny_labels.csv'}"
        self.ner_model = f"local:{cb / 'tiny_modernbert_ner'}"
        self.ner_corpus_url = f"file://{cb / 'tiny_ner_corpus.parquet'}"
        self.ner_gold_url = f"file://{cb / 'tiny_ner_gold.csv'}"
        for attr in ("tiny_bert", "classifier", "ner_model"):
            path = getattr(self, attr).removeprefix("local:")
            if not Path(path).exists():
                raise SystemExit(f"fixture missing: {path}")


def run_eval_suite(db, fx: Fixtures, tag: str) -> dict:
    """Run every eval verb + read-back on one transport, returning the reports.

    `tag` keeps the registered sources disjoint on the module-shared remote
    server (the embedded peer is isolated by its own temp catalog). The same
    call shapes run on both transports — the verb surface is transport-agnostic."""
    patents = f"patents_{tag}"
    golden_rel = f"golden_rel_{tag}"
    golden_lbl = f"golden_lbl_{tag}"
    ner_corpus = f"ner_corpus_{tag}"
    ner_gold = f"ner_gold_{tag}"

    # --- embeddings: register + embed with two different models -------------- #
    db.add_source(patents, url=fx.patents_url, format="parquet")
    table_modernbert = db.generate_embeddings(
        source=patents, model=fx.tiny_modernbert, columns=["abstract"], key="id", modality="text"
    )
    table_bert = db.generate_embeddings(
        source=patents, model=fx.tiny_bert, columns=["abstract"], key="id", modality="text"
    )
    db.add_source(golden_rel, url=fx.golden_rel_url, format="csv")
    golden_rel_path = f"{golden_rel}.public.golden_relevance"

    cohorts = {"q1": {"split": "val"}}
    embeddings = db.eval_embeddings(
        source=patents,
        golden_source=golden_rel_path,
        embedding_table=table_modernbert,
        k=K,
        cohorts=cohorts,
    )
    per_query = db.eval_per_query(embeddings["eval_run_id"])

    # --- eval_compare: self (determinism anchor) + two-table (non-zero) ------ #
    compare_self = db.eval_compare(
        embedding_tables=[table_modernbert, table_modernbert],
        source=patents,
        golden_source=golden_rel_path,
        k=K,
    )
    compare_two = db.eval_compare(
        embedding_tables=[table_modernbert, table_bert],
        source=patents,
        golden_source=golden_rel_path,
        k=K,
    )

    # --- classification inference eval --------------------------------------- #
    db.add_source(golden_lbl, url=fx.labels_url, format="csv")
    inference_cls = db.eval_inference(
        model=fx.classifier,
        source=patents,
        columns=["abstract"],
        task="classification",
        golden_source=f"{golden_lbl}.public.tiny_labels",
        label_column="label",
    )

    # --- NER inference eval (RC1: its own corpus + char-offset span golden) -- #
    db.add_source(ner_corpus, url=fx.ner_corpus_url, format="parquet")
    db.add_source(ner_gold, url=fx.ner_gold_url, format="csv")
    inference_ner = db.eval_inference(
        model=fx.ner_model,
        source=ner_corpus,
        columns=["text"],
        task="ner",
        golden_source=f"{ner_gold}.public.tiny_ner_gold",
        label_column="label",
    )

    return {
        "embeddings": _normalize_embedding_report(embeddings),
        "per_query": _sorted_by(per_query, "query_id"),
        "compare_self": _normalize_compare_report(compare_self),
        "compare_two": _normalize_compare_report(compare_two),
        "inference_cls": _normalize_inference_report(inference_cls),
        "inference_ner": _normalize_inference_report(inference_ner),
    }


# --------------------------------------------------------------------------- #
# Channel sequence (RC5: run on BOTH transports, parity-checked live)
# --------------------------------------------------------------------------- #


def run_channel_sequence(db, tenant_a: str, tenant_b: str) -> dict:
    """Register two generic provenance channels, append columns, list, and the
    #170 tenant-isolation / non-collision property — all under explicit tenant
    scopes so the two transports' namespaces line up for the parity comparison.

    Returns the listings the chapter / parity check assert on. NO consumer
    vocabulary: generic channel ids, opaque tenant UUIDs."""
    # Register + append under tenant A.
    with db.tenant_scope(tenant_a):
        db.register_channel(
            "scored_by", priority=50, columns=[("score", "Float64"), ("model", "Utf8")]
        )
        db.register_channel("annotated_by", priority=10, columns=[("label", "Utf8")])
        db.add_channel_columns(
            "annotated_by", columns=[("confidence", "Float32"), ("rank", "Int32")]
        )
        list_a = db.list_channels()

    # Tenant B does not see A's channels (separate per-tenant namespaces).
    with db.tenant_scope(tenant_b):
        list_b_before = db.list_channels()
        # B registers its OWN "scored_by" with DIFFERENT columns — no collision.
        db.register_channel("scored_by", priority=3, columns=[("note", "Utf8")])
        list_b_after = db.list_channels()

    # A's channels are unchanged by B's registration (own-namespace isolation).
    with db.tenant_scope(tenant_a):
        list_a_again = db.list_channels()

    # An unbound connection sees only the global (NULL-tenant) seed channels.
    unbound = db.list_channels()

    return {
        "list_a": list_a,
        "list_b_before": list_b_before,
        "list_b_after": list_b_after,
        "list_a_again": list_a_again,
        "unbound": unbound,
    }


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
        self._artifact_dir = tempfile.mkdtemp(prefix="jammi_srv_ch14_")
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


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums() -> None:
    sums = {
        p.name: _checksum(p)
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def emit(fx: Fixtures, server_bin: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Stable tenant UUIDs for the channel sequence (committed-report determinism:
    # the listings the chapter compares against carry no tenant value, but the
    # registrations must be reproducible on a re-emit).
    tenant_a = "11111111-1111-4111-8111-111111111111"
    tenant_b = "22222222-2222-4222-8222-222222222222"

    # --- embedded transport (the canonical reports) -------------------------- #
    with tempfile.TemporaryDirectory() as catalog:
        embedded = jammi_ai.connect(f"file://{catalog}")
        print("== embedded engine: eval suite ==", flush=True)
        embedded_reports = run_eval_suite(embedded, fx, tag="emb")
        print("== embedded engine: channel sequence ==", flush=True)
        embedded_channels = run_channel_sequence(embedded, tenant_a, tenant_b)

    # --- remote transport (live grpc:// parity) ------------------------------ #
    with LiveServer(server_bin) as endpoint:
        print(f"== remote engine up at {endpoint} ==", flush=True)
        remote = jammi_client.connect(endpoint)
        try:
            print("== remote engine: eval suite ==", flush=True)
            remote_reports = run_eval_suite(remote, fx, tag="rem")
            print("== remote engine: channel sequence ==", flush=True)
            remote_channels = run_channel_sequence(remote, tenant_a, tenant_b)
        finally:
            remote.close()

    # --- the live remote == embedded parity verdict (RC2 + RC5) -------------- #
    parity = []
    eval_verbs = (
        "embeddings", "per_query", "compare_self", "compare_two", "inference_cls", "inference_ner"
    )
    for name in eval_verbs:
        parity.append(_parity(f"eval.{name}", embedded_reports[name], remote_reports[name]))
    for name in ("list_a", "list_b_before", "list_b_after", "list_a_again", "unbound"):
        parity.append(_parity(f"channel.{name}", embedded_channels[name], remote_channels[name]))
    print("== remote == embedded parity: PASS for all verbs + channel listings ==", flush=True)

    # --- derive the frozen golden scalars from the embedded reports ---------- #
    emb_agg = embedded_reports["embeddings"]["aggregate"]
    cls_agg = embedded_reports["inference_cls"]["aggregate"]
    ner_agg = embedded_reports["inference_ner"]["aggregate"]

    # eval_compare self-anchor: every metric's delta exactly 0.0.
    self_treatment = embedded_reports["compare_self"]["per_table"][1]
    self_deltas = {
        m: self_treatment["delta"][m]["absolute"]
        for m in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
    }
    self_max_abs_delta = max(abs(v) for v in self_deltas.values())
    self_sig_present = float(self_treatment["delta"]["significance"] is not None)

    # eval_compare two-table: the genuine non-zero recall delta + significance.
    two_treatment = embedded_reports["compare_two"]["per_table"][1]
    two_recall_delta = two_treatment["delta"]["recall_at_k"]["absolute"]
    two_sig_present = float(two_treatment["delta"]["significance"] is not None)
    two_recall_p = two_treatment["delta"]["significance"]["recall_at_k"]["p_value"]

    # channel goldens: tenant isolation (#170) — A's channel count, B sees none of
    # A's named channels, no collision on the shared id.
    a_names = {c["channel_id"] for c in embedded_channels["list_a"]}
    b_before_names = {c["channel_id"] for c in embedded_channels["list_b_before"]}
    leak = len(({"scored_by", "annotated_by"}) & b_before_names)
    a_scored = next(c for c in embedded_channels["list_a_again"] if c["channel_id"] == "scored_by")
    b_scored = next(c for c in embedded_channels["list_b_after"] if c["channel_id"] == "scored_by")
    collision = 1.0 if a_scored["columns"] == b_scored["columns"] else 0.0
    annotated = next(c for c in embedded_channels["list_a"] if c["channel_id"] == "annotated_by")

    golden = {
        # embeddings aggregate
        "embeddings.recall_at_k": {"value": float(emb_agg["recall_at_k"]), "tol": 1e-6},
        "embeddings.precision_at_k": {"value": float(emb_agg["precision_at_k"]), "tol": 1e-6},
        "embeddings.mrr": {"value": float(emb_agg["mrr"]), "tol": 1e-6},
        "embeddings.ndcg": {"value": float(emb_agg["ndcg"]), "tol": 1e-6},
        # classification aggregate
        "inference_cls.accuracy": {"value": float(cls_agg["accuracy"]), "tol": 1e-6},
        "inference_cls.f1": {"value": float(cls_agg["f1"]), "tol": 1e-6},
        # NER aggregate
        "inference_ner.precision": {"value": float(ner_agg["precision"]), "tol": 1e-6},
        "inference_ner.recall": {"value": float(ner_agg["recall"]), "tol": 1e-6},
        "inference_ner.f1": {"value": float(ner_agg["f1"]), "tol": 1e-6},
        # eval_compare self-anchor (determinism): max |delta| == 0, significance present
        "compare.self_max_abs_delta": {"value": float(self_max_abs_delta), "tol": 1e-9},
        "compare.self_significance_present": {"value": self_sig_present, "tol": 0.0},
        # eval_compare two-table: the genuine non-zero recall delta + significance present
        "compare.two_recall_delta_abs": {"value": float(two_recall_delta), "tol": 1e-6},
        "compare.two_significance_present": {"value": two_sig_present, "tol": 0.0},
        # channel tenant isolation (#170)
        "channel.a_channel_count": {"value": float(len(a_names)), "tol": 0.0},
        "channel.tenant_leak": {"value": float(leak), "tol": 0.0},
        "channel.collision": {"value": collision, "tol": 0.0},
        "channel.annotated_column_count": {"value": float(len(annotated["columns"])), "tol": 0.0},
    }

    # --- commit the embedded-canonical reports (stripped + byte-stable) ------- #
    committed = {
        "embeddings.json": _strip_instance_keys(embedded_reports["embeddings"]),
        "per_query.json": _strip_instance_keys(embedded_reports["per_query"]),
        "compare_self.json": _strip_instance_keys(embedded_reports["compare_self"]),
        "compare_two.json": _strip_instance_keys(embedded_reports["compare_two"]),
        "inference_cls.json": _strip_instance_keys(embedded_reports["inference_cls"]),
        "inference_ner.json": _strip_instance_keys(embedded_reports["inference_ner"]),
        "channels.json": embedded_channels,
    }
    for fname, payload in committed.items():
        (ARTIFACTS / fname).write_text(json.dumps(payload, indent=2, sort_keys=True))

    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))

    eval_record = {
        "purpose": (
            "The engine↔cookbook validator for the H1 residual: the eval + channel "
            "verbs run on BOTH the embedded engine and a live remote grpc:// jammi-server, "
            "asserted remote == embedded live (the real cross-transport parity check). The "
            "committed reports are the embedded-canonical reports; PR CI reads them and "
            "asserts aggregates-to-golden + shape, never a static re-diff of two artifacts."
        ),
        "fixtures": (
            "engine public deterministic fixtures (patents.parquet + golden_relevance.csv + "
            "tiny_modernbert/tiny_bert; tiny_modernbert_classifier + tiny_labels.csv; "
            "tiny_modernbert_ner + tiny_ner_corpus.parquet + tiny_ner_gold.csv) — NOT the "
            "arxiv keystone corpus; this measures the wire SURFACE, not corpus coverage."
        ),
        "transports": [
            "embedded (file://, in-process)",
            "remote (grpc://, live jammi-server)",
        ],
        "parity_tol": _TOL,
        "parity_tol_by_key": _FLOAT_TOL_BY_KEY,
        "parity_verdict": "remote == embedded for every eval verb and every channel listing",
        "parity": parity,
        "measured": {
            "embeddings_aggregate": {
                k: float(emb_agg[k]) for k in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
            },
            "inference_cls_aggregate": {
                "accuracy": float(cls_agg["accuracy"]),
                "f1": float(cls_agg["f1"]),
            },
            "inference_ner_aggregate": {
                k: float(ner_agg[k]) for k in ("precision", "recall", "f1")
            },
            "compare_self_max_abs_delta": float(self_max_abs_delta),
            "compare_two_recall_delta_abs": float(two_recall_delta),
            "compare_two_recall_p_value": float(two_recall_p),
            "channel_a_count": len(a_names),
            "channel_tenant_leak": leak,
            "channel_collision": collision,
        },
    }
    (ARTIFACTS / "eval.json").write_text(json.dumps(eval_record, indent=2, sort_keys=True))

    _write_checksums()

    print("\n=== eval + channels, measured (embedded canonical) ===", flush=True)
    print(f"  embeddings: recall@k={emb_agg['recall_at_k']:.6f} mrr={emb_agg['mrr']:.6f} "
          f"ndcg={emb_agg['ndcg']:.6f}", flush=True)
    print(f"  classification: accuracy={cls_agg['accuracy']:.6f} f1={cls_agg['f1']:.6f}",
          flush=True)
    print(f"  ner: precision={ner_agg['precision']:.6f} recall={ner_agg['recall']:.6f} "
          f"f1={ner_agg['f1']:.6f}", flush=True)
    print(f"  compare self max|delta|={self_max_abs_delta:.2e} "
          f"(significance present={bool(self_sig_present)})", flush=True)
    print(f"  compare two-table recall Δ={two_recall_delta:+.6f} (p={two_recall_p:.4f}, "
          f"significance present={bool(two_sig_present)})", flush=True)
    print(f"  channels: A count={len(a_names)} tenant_leak={leak} collision={collision}",
          flush=True)
    print(f"  parity: remote == embedded for all {len(parity)} reports "
          f"(tol {_TOL}; NER confidence at f32-relative {_F32_REL})", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", default=None,
                    help="engine checkout with tests/fixtures + cookbook/fixtures "
                         "(or set JAMMI_FIXTURES_ROOT)")
    ap.add_argument("--server-bin", default=os.environ.get("JAMMI_SERVER_BIN"),
                    help="built CPU jammi-server binary (or set JAMMI_SERVER_BIN)")
    args = ap.parse_args()
    if not args.server_bin or not os.path.exists(args.server_bin):
        raise SystemExit("pass --server-bin (or set JAMMI_SERVER_BIN) to a built jammi-server")
    fx = Fixtures(_fixtures_root(args.fixtures_root))
    emit(fx, args.server_bin)


if __name__ == "__main__":
    main()
