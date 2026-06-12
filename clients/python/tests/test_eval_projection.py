"""Hermetic projection oracle for the eval report helpers.

The client's `_database` projection helpers must reproduce the embed wheel's
dict shapes EXACTLY — the embedded eval verbs return the engine reports through
`serde_json` (`serializable_to_pydict`), so the contract is the serde
serialization of `crates/jammi-wire/src/eval/report.rs`:

  * the internally tagged `task` enums flatten into the record
    (`{"task": "classification", "accuracy": …}`), never nest under a variant
    key;
  * `recall_at_ks` is a list of two-element `[k, recall]` pairs (the engine's
    `(usize, f64)` tuples);
  * absent options are explicit `None`s under always-present keys (`delta` for
    the baseline table, `significance` for an unpairable run) — report.rs
    carries no `skip_serializing_if`.

The oracle is the committed golden fixture
`tests/fixtures/eval_report_projection.json`, generated from the Rust side and
pinned there by `report.rs::projection_fixture_tests` — so the Rust test locks
the fixture to the serde types, and this test locks the Python helpers to the
fixture. The proto messages below are hand-constructed with literal values
(not built from the fixture) so a projection bug cannot cancel against an
inverse construction bug.

Hermetic: no server, no embedded engine — only constructed protos.
"""

from __future__ import annotations

import json
from pathlib import Path

from jammi_client._database import (
    _compare_report_to_dict,
    _embedding_report_to_dict,
    _inference_report_to_dict,
)
from jammi_client._generated.jammi.v1 import eval_pb2

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "eval_report_projection.json"
)


def _fixture(key: str) -> dict:
    return json.loads(FIXTURE.read_text())[key]


def _metrics(recall: float, precision: float, mrr: float, ndcg: float):
    return eval_pb2.QueryMetrics(recall=recall, precision=precision, mrr=mrr, ndcg=ndcg)


def _pairs(*pairs):
    return [eval_pb2.RecallAtK(k=k, recall=r) for k, r in pairs]


def _embedding_report() -> eval_pb2.EmbeddingEvalReport:
    return eval_pb2.EmbeddingEvalReport(
        eval_run_id="run-embedding-fixture",
        aggregate=eval_pb2.AggregateMetrics(
            recall_at_k=0.75, precision_at_k=0.5, mrr=0.625, ndcg=0.8125
        ),
        per_query=[
            eval_pb2.PerQueryRecord(
                query_id="q1",
                metrics=_metrics(1.0, 0.5, 1.0, 1.0),
                recall_at_ks=_pairs((1, 0.5), (3, 1.0), (5, 1.0), (10, 1.0)),
                distance=0.125,
                cohorts={"family": "A", "split": "val"},
            ),
            eval_pb2.PerQueryRecord(
                query_id="q2",
                metrics=_metrics(0.5, 0.5, 0.25, 0.625),
                recall_at_ks=_pairs((1, 0.0), (3, 0.5), (5, 0.5), (10, 0.5)),
                distance=0.0,
            ),
        ],
    )


def test_embedding_report_projection_matches_the_embedded_serde_shape():
    """`recall_at_ks` flattens to `[k, recall]` pairs and an untagged query
    carries `cohorts: {}` — exactly the serde dict the embed wheel returns."""
    assert _embedding_report_to_dict(_embedding_report()) == _fixture("embedding")


def test_classification_report_projection_flattens_the_task_tag():
    """The classification oneof flattens to `{"task": "classification", …}` —
    the internally tagged serde shape, not a nested variant key."""
    report = eval_pb2.InferenceEvalReport(
        aggregate=eval_pb2.InferenceAggregate(
            classification=eval_pb2.ClassificationResult(
                accuracy=0.75,
                f1=0.5,
                per_class={
                    "spam": eval_pb2.ClassMetrics(precision=1.0, recall=0.5, f1=0.625),
                    "ham": eval_pb2.ClassMetrics(precision=0.5, recall=1.0, f1=0.625),
                },
            )
        ),
        per_record=[
            eval_pb2.PerRecordPrediction(
                classification=eval_pb2.PerRecordPrediction.Classification(
                    record_id="r1", predicted="spam", gold="spam"
                )
            ),
            eval_pb2.PerRecordPrediction(
                classification=eval_pb2.PerRecordPrediction.Classification(
                    record_id="r2", predicted="ham", gold="spam"
                )
            ),
        ],
    )
    assert _inference_report_to_dict(report) == _fixture("inference_classification")


def test_ner_report_projection_flattens_the_task_tag():
    """The NER oneof flattens to `{"task": "ner", …}` with full entity spans —
    including the gold record's empty `text` / zero `confidence`."""
    report = eval_pb2.InferenceEvalReport(
        aggregate=eval_pb2.InferenceAggregate(
            ner=eval_pb2.NerMetrics(
                precision=0.5,
                recall=0.25,
                f1=0.375,
                per_type={
                    "PER": eval_pb2.TypeMetrics(
                        precision=0.5, recall=0.25, f1=0.375, support=4
                    )
                },
            )
        ),
        per_record=[
            eval_pb2.PerRecordPrediction(
                ner=eval_pb2.PerRecordPrediction.Ner(
                    record_id="n1",
                    predicted=[
                        eval_pb2.Entity(
                            label="PER", start=0, end=5, text="Alice", confidence=0.5
                        )
                    ],
                    gold=[eval_pb2.Entity(label="PER", start=0, end=5)],
                )
            )
        ],
    )
    assert _inference_report_to_dict(report) == _fixture("inference_ner")


def _compare_member(run_id: str, recall: float) -> eval_pb2.EmbeddingEvalReport:
    return eval_pb2.EmbeddingEvalReport(
        eval_run_id=run_id,
        aggregate=eval_pb2.AggregateMetrics(
            recall_at_k=recall, precision_at_k=0.5, mrr=0.5, ndcg=0.5
        ),
        per_query=[
            eval_pb2.PerQueryRecord(
                query_id="q1",
                metrics=_metrics(recall, 0.5, 0.5, 0.5),
                recall_at_ks=_pairs((1, recall)),
                distance=0.25,
            )
        ],
    )


def _compare_delta(significance=None) -> eval_pb2.AggregateDelta:
    delta = eval_pb2.AggregateDelta(
        recall_at_k=eval_pb2.MetricDelta(absolute=0.25, relative=0.5),
        precision_at_k=eval_pb2.MetricDelta(absolute=0.0, relative=0.0),
        mrr=eval_pb2.MetricDelta(absolute=-0.125, relative=-0.25),
        ndcg=eval_pb2.MetricDelta(absolute=0.125, relative=0.25),
    )
    if significance is not None:
        delta.significance.CopyFrom(significance)
    return delta


def test_compare_report_projection_carries_explicit_nulls():
    """The baseline's `delta` and an unpairable run's `significance` are
    explicit `None`s under always-present keys — never missing keys, never
    fabricated zeros."""

    def sig(p_value, ci_lower, ci_upper):
        return eval_pb2.MetricSignificance(
            p_value=p_value, ci_lower=ci_lower, ci_upper=ci_upper
        )

    report = eval_pb2.CompareEvalReport(
        per_table=[
            eval_pb2.TableEvalReport(
                table_name="emb_baseline",
                embedding_eval=_compare_member("run-baseline", 0.5),
            ),
            eval_pb2.TableEvalReport(
                table_name="emb_paired",
                embedding_eval=_compare_member("run-paired", 0.75),
                delta=_compare_delta(
                    eval_pb2.DeltaSignificance(
                        recall_at_k=sig(0.0625, 0.125, 0.375),
                        precision_at_k=sig(1.0, 0.0, 0.0),
                        mrr=sig(0.5, -0.25, 0.0),
                        ndcg=sig(0.25, 0.0, 0.25),
                    )
                ),
            ),
            eval_pb2.TableEvalReport(
                table_name="emb_unpaired",
                embedding_eval=_compare_member("run-unpaired", 0.25),
                delta=_compare_delta(),
            ),
        ]
    )
    assert _compare_report_to_dict(report) == _fixture("compare")
