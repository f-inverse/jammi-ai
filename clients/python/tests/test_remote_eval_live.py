"""Live remote round-trip for the eval verbs.

Stands up a real CPU `jammi-server` (the shared `live_server` fixture in
`conftest.py`) and drives the eval family through the pure-Python
`RemoteDatabase`, asserting the remote reports agree with an embedded
`jammi_ai.Database`'s over the same fixtures: the `patents.parquet` corpus
embedded with the deterministic `tiny_modernbert` encoder, evaluated against
the `golden_relevance.csv` golden set; and the deterministic
`tiny_modernbert_classifier` run over the same corpus against the
`tiny_labels.csv` golden labels.

The `live_server` fixture is module-scoped, so its catalog persists across the
tests in this module. Each test registers its sources under a per-test name
(`patents_<tag>`, `golden_<tag>`) so the tests' footprints stay disjoint on the
one shared server — the embedded peer is already isolated by its own
`tmp_path`. The golden set is then addressed by that source's full catalog path
(`<source>.public.<table>`).

The parity assertion is key-structure equality everywhere plus per-metric
closeness — never deep float equality across two engine instances (forbidden
by design: the two engines are separate processes, so bit-identity is not a
contract). Instance-minted identifiers are excluded from value comparison:
`eval_run_id` (a fresh UUID per run) and `table_name` (embedding result tables
embed a creation timestamp), though both keys' presence is still pinned by the
structure assertion.

Gated, not hermetic: skipped unless `JAMMI_SERVER_BIN` points at a built
`jammi-server` executable, same as the other live modules.
"""

from __future__ import annotations

import os
from pathlib import Path

import grpc
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

jammi_ai = pytest.importorskip("jammi_ai")
import jammi_client  # noqa: E402

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")

pytestmark = pytest.mark.skipif(
    not SERVER_BIN or not os.path.exists(SERVER_BIN),
    reason="JAMMI_SERVER_BIN not set to a built jammi-server binary",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
# The classifier model and its golden labels are deterministic CPU cookbook
# fixtures (real safetensors weights + per-id labels), the same ones the
# engine's hermetic inference-eval tests drive.
COOKBOOK_FIXTURES = REPO_ROOT / "cookbook" / "fixtures"

PATENTS_URL = f"file://{FIXTURES / 'patents.parquet'}"
GOLDEN_URL = f"file://{FIXTURES / 'golden_relevance.csv'}"
TINY_MODERNBERT = f"local:{FIXTURES / 'tiny_modernbert'}"

TINY_CLASSIFIER = f"local:{COOKBOOK_FIXTURES / 'tiny_modernbert_classifier'}"
LABELS_URL = f"file://{COOKBOOK_FIXTURES / 'tiny_labels.csv'}"

# A second deterministic encoder distinct from `tiny_modernbert`, so a compare
# over the two embedding tables produces a genuinely non-zero delta (not the
# all-zero self-comparison that hides an order-sensitive significance CI).
TINY_BERT = f"local:{COOKBOOK_FIXTURES / 'tiny_bert'}"

# The deterministic NER model + its corpus and per-span gold, the same fixtures
# the engine's hermetic NER eval drives. NER per-record entities carry a
# `confidence`, the field whose cross-transport parity this module pins.
TINY_NER = f"local:{COOKBOOK_FIXTURES / 'tiny_modernbert_ner'}"
NER_CORPUS_URL = f"file://{COOKBOOK_FIXTURES / 'tiny_ner_corpus.parquet'}"
NER_GOLD_URL = f"file://{COOKBOOK_FIXTURES / 'tiny_ner_gold.csv'}"

# Per-run identifiers whose VALUES differ across engine instances by
# construction: `eval_run_id` is a fresh UUID per run; `table_name` embeds the
# embedding table's creation timestamp. Their keys stay pinned by `_shape`.
_INSTANCE_KEYS = {"eval_run_id", "table_name"}

# Closeness bound for cross-instance metric comparison. The fixture encoder is
# deterministic, so the metrics agree far tighter than this; the bound only
# absorbs accumulation-order noise, never a ranking difference.
_TOL = 1e-9


def _shape(obj):
    """The key structure of a report: dicts map to their keyed sub-shapes,
    lists to element shapes, scalars to their type name — so two reports with
    the same shape carry the same keys at every level."""
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shape(v) for v in obj]
    return type(obj).__name__


# Per-field float tolerance overrides, keyed by dict key. A field carried on
# the wire at a narrower precision than f64 cannot agree across transports to
# the f64 `_TOL`: the engine computes a NER span's `confidence` in f32, so both
# transports carry the same f32 value but widen it to a Python float by two
# different conventions (the embed wheel serializes the f32 through serde's
# shortest-decimal; the gRPC client reads a 32-bit proto `float` back as its
# exact f64 widening). The two f64s agree only to f32 precision (~1e-7), so
# `confidence` is compared at an f32-relative bound, never the f64 `_TOL`. This
# is the field's true cross-transport contract, not a slackened metric.
_F32_REL = 1e-6
_FLOAT_TOL_BY_KEY = {"confidence": dict(rel=_F32_REL, abs=_F32_REL)}


def _assert_values_close(a, b, path="", key=None):
    """Recursive value equality with `_TOL`-close floats, skipping the
    instance-minted identifier values (their presence is pinned by `_shape`).
    A float under a key in `_FLOAT_TOL_BY_KEY` is compared at that key's bound
    instead of `_TOL` — see the table for why `confidence` is f32-precision."""
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys {set(a)} != {set(b)}"
        for k in a:
            if k in _INSTANCE_KEYS:
                continue
            _assert_values_close(a[k], b[k], f"{path}.{k}", key=k)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_values_close(x, y, f"{path}[{i}]", key=key)
    elif isinstance(a, float):
        tol = _FLOAT_TOL_BY_KEY.get(key, dict(abs=_TOL))
        assert a == pytest.approx(b, **tol), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _sorted_per_query(report: dict) -> dict:
    """Return a copy of an embedding report with its `per_query` list ordered
    by `query_id`.

    The runner emits per-query records in retrieval order, not a stable sort
    (there is no ORDER BY — engine output is never reshaped to make a test
    pass), so two engine instances may interleave the same records differently.
    Normalising the order on the ASSERTION side lets the cross-transport
    comparison pair the same query without touching the engine."""
    ordered = dict(report)
    ordered["per_query"] = sorted(report["per_query"], key=lambda r: r["query_id"])
    return ordered


def _register_corpus(db, tag: str) -> str:
    """Register the `patents` corpus under a per-test name and embed `abstract`.

    Returns the generated embedding table name. The per-test `tag` keeps the
    source disjoint on the module-shared remote server; the same call runs on
    both transports — the verb surface is transport-agnostic."""
    source = f"patents_{tag}"
    db.add_source(source, url=PATENTS_URL, format="parquet")
    table = db.generate_embeddings(
        source=source,
        model=TINY_MODERNBERT,
        columns=["abstract"],
        key="id",
        modality="text",
    )
    return table


def _register_relevance_golden(db, tag: str) -> str:
    """Register the retrieval golden set under a per-test name; return its full
    catalog path (`<source>.public.golden_relevance`)."""
    source = f"golden_rel_{tag}"
    db.add_source(source, url=GOLDEN_URL, format="csv")
    return f"{source}.public.golden_relevance"


def test_eval_embeddings_and_per_query_round_trip_matches_embedded(
    live_server, tmp_path
):
    """`eval_embeddings` returns the same report shape and `_TOL`-close metrics
    on both transports (cohort tags included), and `eval_per_query` reads back
    one structurally identical persisted row per golden query."""
    tag = "emb"
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        cohorts = {"q1": {"split": "val"}}
        reports = {}
        per_query_rows = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            _register_corpus(db, tag)
            golden = _register_relevance_golden(db, tag)
            report = db.eval_embeddings(
                source=f"patents_{tag}",
                golden_source=golden,
                k=10,
                cohorts=cohorts,
            )
            reports[name] = report
            # Each transport reads back its OWN run's persisted rows; sorted by
            # query_id so the cross-transport comparison pairs the same query.
            per_query_rows[name] = sorted(
                db.eval_per_query(report["eval_run_id"]),
                key=lambda r: r["query_id"],
            )

        # `per_query` carries no stable order across instances — normalise it
        # by query_id before the structural + value comparison.
        remote_sorted = _sorted_per_query(reports["remote"])
        embedded_sorted = _sorted_per_query(reports["embedded"])
        assert _shape(remote_sorted) == _shape(embedded_sorted)
        _assert_values_close(remote_sorted, embedded_sorted)

        # The run id is a fresh UUID per run — present and non-empty on both,
        # never value-compared.
        for report in reports.values():
            assert report["eval_run_id"]

        # The cohort tag rode the wire verbatim into the per-query record.
        remote_q1 = next(
            r for r in reports["remote"]["per_query"] if r["query_id"] == "q1"
        )
        assert remote_q1["cohorts"] == {"split": "val"}

        # eval_per_query: one persisted row per per-query record, structurally
        # identical and metric-close across transports.
        for name, report in reports.items():
            assert len(per_query_rows[name]) == len(report["per_query"]), name
        assert _shape(per_query_rows["remote"]) == _shape(per_query_rows["embedded"])
        _assert_values_close(per_query_rows["remote"], per_query_rows["embedded"])
    finally:
        # The embedded engine releases its resources on drop (RAII); only the
        # remote client holds a gRPC channel that needs an explicit close.
        remote.close()


def test_eval_compare_round_trip_matches_embedded(live_server, tmp_path):
    """`eval_compare` over a self-comparison (the same table twice) returns the
    same report shape on both transports: a baseline with `delta: None` and a
    treatment whose deltas are zero with `significance` present."""
    tag = "cmp"
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        reports = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            table = _register_corpus(db, tag)
            golden = _register_relevance_golden(db, tag)
            report = db.eval_compare(
                embedding_tables=[table, table],
                source=f"patents_{tag}",
                golden_source=golden,
                k=10,
            )
            # Each table's nested embedding report carries an unordered
            # `per_query` — normalise it by query_id before comparing.
            for entry in report["per_table"]:
                entry["embedding_eval"] = _sorted_per_query(entry["embedding_eval"])
            reports[name] = report

        assert _shape(reports["remote"]) == _shape(reports["embedded"])
        _assert_values_close(reports["remote"], reports["embedded"])

        for name, report in reports.items():
            baseline, treatment = report["per_table"]
            # The baseline carries an explicit None delta — the key is present.
            assert baseline["delta"] is None, name
            # A self-comparison's deltas are exactly zero, and the runs share
            # every query, so the paired significance is present (not None)
            # with a CI collapsed onto zero.
            delta = treatment["delta"]
            for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
                assert delta[metric]["absolute"] == 0.0, (name, metric)
                assert delta[metric]["relative"] == 0.0, (name, metric)
                sig = delta["significance"][metric]
                assert sig["ci_lower"] == 0.0 == sig["ci_upper"], (name, metric)
                assert sig["p_value"] > 0.99, (name, metric)
    finally:
        # The embedded engine releases its resources on drop (RAII); only the
        # remote client holds a gRPC channel that needs an explicit close.
        remote.close()


def _embed_with(db, tag: str, model: str, slot: str) -> str:
    """Embed the `patents` corpus under a per-(test, model) source with a chosen
    encoder, returning the embedding table name. The `slot` disambiguates two
    encoders sharing one test's `tag` on the module-shared remote server."""
    source = f"patents_{tag}_{slot}"
    db.add_source(source, url=PATENTS_URL, format="parquet")
    return db.generate_embeddings(
        source=source,
        model=model,
        columns=["abstract"],
        key="id",
        modality="text",
    )


def test_eval_compare_nondegenerate_significance_matches_embedded(
    live_server, tmp_path
):
    """`eval_compare` over two DIFFERENT encoders (a genuine, non-zero delta)
    returns a significance CI that matches across transports AND is reproducible.

    The self-comparison case (every paired difference is exactly zero) collapses
    the bootstrap CI onto [0, 0], hiding any order-sensitivity in the resampler.
    Two distinct encoders give non-zero per-query differences whose multiset the
    two engines may emit in different orders; the CI must depend only on that
    multiset, so it must agree across transports and reproduce within a transport.
    """
    tag = "cmpnd"
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        reports = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            baseline = _embed_with(db, tag, TINY_MODERNBERT, "mb")
            treatment = _embed_with(db, tag, TINY_BERT, "bert")
            golden = _register_relevance_golden(db, tag)
            # Run twice on the SAME transport to pin within-transport
            # reproducibility independently of the cross-transport parity.
            first = db.eval_compare(
                embedding_tables=[baseline, treatment],
                source=f"patents_{tag}_mb",
                golden_source=golden,
                k=10,
            )
            second = db.eval_compare(
                embedding_tables=[baseline, treatment],
                source=f"patents_{tag}_mb",
                golden_source=golden,
                k=10,
            )
            for report in (first, second):
                for entry in report["per_table"]:
                    entry["embedding_eval"] = _sorted_per_query(
                        entry["embedding_eval"]
                    )
            # Reproducible within a transport: the seeded bootstrap over a fixed
            # multiset is byte-identical run-to-run, CI included.
            for entry_a, entry_b in zip(first["per_table"], second["per_table"]):
                _assert_values_close(entry_a["delta"], entry_b["delta"], path=name)
            reports[name] = first

        # The treatment delta is genuinely non-zero — the fixture is not a
        # disguised self-comparison, so the CI is exercised off [0, 0].
        sig = {}
        for name, report in reports.items():
            treatment = report["per_table"][1]
            assert treatment["delta"] is not None, name
            assert treatment["delta"]["significance"] is not None, name
            sig[name] = treatment["delta"]["significance"]
            nonzero = any(
                treatment["delta"][m]["absolute"] != 0.0
                for m in ("recall_at_k", "precision_at_k", "mrr", "ndcg")
            )
            assert nonzero, f"{name}: two distinct encoders must differ on some metric"

        # The significance CI (the field the order-sensitivity bug moved) agrees
        # across transports for every metric.
        for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
            for bound in ("ci_lower", "ci_upper", "p_value"):
                assert sig["remote"][metric][bound] == pytest.approx(
                    sig["embedded"][metric][bound], abs=_TOL
                ), (metric, bound, sig["remote"][metric][bound], sig["embedded"][metric][bound])
    finally:
        remote.close()


def test_eval_inference_ner_round_trip_matches_embedded(live_server, tmp_path):
    """`eval_inference` (NER) over the deterministic NER fixture returns the same
    report shape and `_TOL`-close metrics on both transports, including each
    per-record predicted entity's `confidence`.

    NER is the only eval path whose per-record payload carries a `confidence`
    float, so it is the path that exercises that field's cross-transport parity.
    The engine computes the span confidence in f32; both transports carry that
    same f32 but widen it to a Python float by different conventions (see
    `_FLOAT_TOL_BY_KEY`), so `confidence` agrees only to f32 precision — that
    f32-relative bound is its contract, while every other report float still
    holds to `_TOL`.
    """
    tag = "infner"
    source = f"patents_ner_{tag}"
    golden_source = f"golden_ner_{tag}"
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        reports = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            db.add_source(source, url=NER_CORPUS_URL, format="parquet")
            db.add_source(golden_source, url=NER_GOLD_URL, format="csv")
            reports[name] = db.eval_inference(
                model=TINY_NER,
                source=source,
                columns=["text"],
                task="ner",
                golden_source=f"{golden_source}.public.tiny_ner_gold",
                label_column="label",
            )

        assert _shape(reports["remote"]) == _shape(reports["embedded"])
        _assert_values_close(reports["remote"], reports["embedded"])

        # The aggregate is the NER variant on both transports, and the per-record
        # payload carries predicted entity spans with a `confidence` float.
        saw_confidence = False
        for name, report in reports.items():
            assert report["aggregate"]["task"] == "ner", name
            assert report["per_record"], name
            for rec in report["per_record"]:
                assert rec["task"] == "ner", name
                for entity in rec["predicted"]:
                    assert isinstance(entity["confidence"], float), name
                    saw_confidence = True
        # A predicted span (carrying a populated confidence) must appear, or the
        # parity assertion above never touched the field the test exists to pin.
        assert saw_confidence, "NER fixture must yield at least one predicted span"
    finally:
        remote.close()


def test_eval_inference_classification_round_trip_matches_embedded(
    live_server, tmp_path
):
    """`eval_inference` (classification) over the deterministic classifier
    fixture returns the same report shape and `_TOL`-close metrics on both
    transports: a `{"task": "classification"}`-tagged aggregate and one tagged
    per-record entry per golden-labelled row."""
    tag = "infcls"
    source = f"patents_{tag}"
    golden_source = f"golden_lbl_{tag}"
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        reports = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            db.add_source(source, url=PATENTS_URL, format="parquet")
            db.add_source(golden_source, url=LABELS_URL, format="csv")
            reports[name] = db.eval_inference(
                model=TINY_CLASSIFIER,
                source=source,
                columns=["abstract"],
                task="classification",
                golden_source=f"{golden_source}.public.tiny_labels",
                label_column="label",
            )

        assert _shape(reports["remote"]) == _shape(reports["embedded"])
        _assert_values_close(reports["remote"], reports["embedded"])

        # The aggregate flattens the task tag into the record on both transports.
        for name, report in reports.items():
            assert report["aggregate"]["task"] == "classification", name
            assert report["per_record"], name
            for rec in report["per_record"]:
                assert rec["task"] == "classification", name
    finally:
        remote.close()


def test_eval_embeddings_empty_golden_set_zero_aggregate_matches_embedded(
    live_server, tmp_path
):
    """An empty golden set yields an all-zero aggregate on BOTH transports.

    The retrieval metrics are guarded against a zero-query golden set (the
    aggregate defaults to 0.0 with no queries to average), so the report is a
    well-formed zero report — never a NaN, never a divergence. The remote and
    embedded engines return that identical zero aggregate over a golden set
    that is schema-valid (typed `query_id` / `query_text` / `relevant_id`
    columns) but carries zero rows. A zero-row parquet preserves its column
    types where a header-only CSV would infer them as Null, so the golden
    type-checks and resolves to zero queries on both transports."""
    tag = "empty"
    source = f"patents_{tag}"
    golden_source = f"golden_empty_{tag}"
    empty_golden = tmp_path / "empty_golden.parquet"
    pq.write_table(
        pa.table(
            {
                "query_id": pa.array([], type=pa.string()),
                "query_text": pa.array([], type=pa.string()),
                "relevant_id": pa.array([], type=pa.string()),
            }
        ),
        empty_golden,
    )
    empty_url = f"file://{empty_golden}"

    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path / 'embed'}")
    try:
        reports = {}
        for name, db in (("remote", remote), ("embedded", embedded)):
            db.add_source(source, url=PATENTS_URL, format="parquet")
            db.add_source(golden_source, url=empty_url, format="parquet")
            db.generate_embeddings(
                source=source,
                model=TINY_MODERNBERT,
                columns=["abstract"],
                key="id",
                modality="text",
            )
            reports[name] = db.eval_embeddings(
                source=source,
                golden_source=f"{golden_source}.public.empty_golden",
                k=10,
            )

        # Identical zero report on both transports: same shape, and every
        # aggregate metric is exactly 0.0 (the empty-set guard, not a NaN).
        assert _shape(reports["remote"]) == _shape(reports["embedded"])
        for name, report in reports.items():
            assert report["per_query"] == [], name
            for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg"):
                assert report["aggregate"][metric] == 0.0, (name, metric)
        # The aggregate blocks are value-identical across transports (all zeros).
        assert reports["remote"]["aggregate"] == reports["embedded"]["aggregate"]
    finally:
        remote.close()


def test_eval_error_paths_match_embedded(live_server, tmp_path):
    """A bad call is rejected by BOTH transports, mapping the same engine error.

    An unknown `golden_source` is an engine-side `Eval`/`Source` error: the
    embedded path raises (the engine error surfaces as a Python exception), and
    the remote path raises `grpc.RpcError` (the server maps the same
    `JammiError` through `map_engine_error`). An unknown `task` string is
    rejected by BOTH transports before any inference runs — the embedded engine
    rejects it when it parses the task vocabulary, and the remote client guards
    the same vocabulary statically (it knows `classification`/`ner`), so a typo
    never reaches the wire. Both reject; the report-shaping path is never
    entered with a bad task.
    """
    tag = "err"
    source = f"patents_{tag}"
    remote = jammi_client.connect(live_server)
    embedded = jammi_ai.connect(f"file://{tmp_path}")
    try:
        for db in (remote, embedded):
            db.add_source(source, url=PATENTS_URL, format="parquet")
            db.generate_embeddings(
                source=source,
                model=TINY_MODERNBERT,
                columns=["abstract"],
                key="id",
                modality="text",
            )

        # Unknown golden_source: the embedded engine raises a Python exception;
        # the remote maps the same engine error onto a gRPC status.
        with pytest.raises(Exception):  # noqa: B017 — embed raises RuntimeError
            embedded.eval_embeddings(
                source=source,
                golden_source="no_such.public.relevance",
                k=10,
            )
        with pytest.raises(grpc.RpcError):
            remote.eval_embeddings(
                source=source,
                golden_source="no_such.public.relevance",
                k=10,
            )

        # Unknown task string: rejected by BOTH transports before any inference
        # runs. The remote client guards the task vocabulary statically (a
        # `ValueError` before the wire); the embedded engine rejects the same
        # unknown task when it parses it. Both raise — the bad task never
        # reaches the report-shaping path on either transport.
        with pytest.raises(ValueError):
            remote.eval_inference(
                model=TINY_CLASSIFIER,
                source=source,
                columns=["abstract"],
                task="not_a_task",
                golden_source="ignored.public.labels",
                label_column="label",
            )
        with pytest.raises(Exception):  # noqa: B017 — embed raises RuntimeError
            embedded.eval_inference(
                model=TINY_CLASSIFIER,
                source=source,
                columns=["abstract"],
                task="not_a_task",
                golden_source="ignored.public.labels",
                label_column="label",
            )
    finally:
        remote.close()
