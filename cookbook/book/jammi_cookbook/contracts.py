"""The artifact-contract registry (K0 §1) — the cookbook's load-bearing spine.

The heavy work (embedding, neighbor-graph build, fine-tune, context-predictor
train, calibration) runs **once**, in the KV-arxiv keystone slice; everything
else is authored against the recorded results. This module is the two-layer
contract that makes that safe:

* **Layer 1 — Schema** (:data:`ARTIFACTS`): a stable name, columns + dtypes,
  producing tier, and — for graph artifacts — the exact declared-edge parameters.
  Known at design time and encoded here; lets chapters reference an artifact by
  contract before it is materialized.
* **Layer 2 — Golden samples**: the committed small-subset artifact *files* plus
  ``golden_metrics.json``. :func:`load_artifact` reads the files; :func:`golden`
  reads the metrics with tolerances; :func:`assert_close` is the measured-verdict
  oracle every chapter ends in.

**The mandate (the chief risk control): chapters READ the cache; they do not
recompute upstream.** Re-execution drifts numbers and fails CI tolerances with no
provenance — the #1 hand-off failure mode. The keystone produces the cache once;
every later chapter loads ``<dataset>.*`` artifacts and asserts against
``golden_metrics.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_ROOT = _REPO_ROOT / "artifacts"


# --------------------------------------------------------------------------- #
# Layer 1 — schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeclaredEdges:
    """The declared-edge parameters for a graph-conditioned artifact (K0 §1).

    These are the BYOG (bring-your-own-graph) knobs passed verbatim to
    ``propagate_embeddings`` / ``predict_with_context_predictor`` /
    ``assemble_context``. Recording them in the contract pins *which* graph an
    artifact was produced over — derived (similarity) vs declared (citation).
    """

    edge_source: str
    edge_src_column: str = "src"
    edge_dst_column: str = "dst"
    edge_type_column: str | None = None
    edge_weight_column: str | None = None
    edge_hops: int | None = None
    edge_fanout: int | None = None
    edge_direction: str | None = None
    edge_types: tuple[str, ...] | None = None
    min_weight: float | None = None
    hybrid: bool = False


@dataclass(frozen=True)
class Artifact:
    """A committed-cache entry's contract: where it lives and what shape it has."""

    name: str
    kind: str  # "parquet" | "edge_table" | "model_id" | "split"
    filename: str  # relative to artifacts/<dataset>/
    columns: dict[str, str] = field(default_factory=dict)
    produced_by: str = ""  # the tier that emits it
    declared_edges: DeclaredEdges | None = None
    note: str = ""

    @property
    def dataset(self) -> str:
        return self.name.split(".", 1)[0]


def _vec(dim: int) -> str:
    return f"FixedSizeList<f32,{dim}>"


# The committed-cache contract. Edge tables and model ids are produced by the
# keystone; the declared citation / route / hierarchy graphs are external inputs
# registered by the loaders (K2). Embedding dimension follows ModernBERT-base.
_EMB_DIM = 768

ARTIFACTS: dict[str, Artifact] = {
    # --- ogbn-arxiv (KV-arxiv keystone) --------------------------------------
    "arxiv.papers": Artifact(
        name="arxiv.papers",
        kind="parquet",
        filename="papers.parquet",
        columns={"paper_id": "Utf8", "title": "Utf8", "subject": "Utf8", "year": "Int"},
        produced_by="tier01",
        note="Subset paper labels (paper_id, title, subject, year) — for display-only "
        "derived numbers; the embedded text columns are not committed.",
    ),
    "arxiv.embeddings": Artifact(
        name="arxiv.embeddings",
        kind="parquet",
        filename="embeddings.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="tier01",
        note="ModernBERT embeddings of title+abstract.",
    ),
    "arxiv.neighbor_graph": Artifact(
        name="arxiv.neighbor_graph",
        kind="edge_table",
        filename="neighbor_graph.parquet",
        columns={"src": "Utf8", "dst": "Utf8", "rank": "Int", "similarity": "f32"},
        produced_by="tier01",
        note="Derived self-kNN similarity graph (build_neighbor_graph, exact=True).",
    ),
    "arxiv.cite_edges": Artifact(
        name="arxiv.cite_edges",
        kind="edge_table",
        filename="cite_edges.parquet",
        columns={"src": "Utf8", "dst": "Utf8"},
        produced_by="external",
        declared_edges=DeclaredEdges(
            edge_source="arxiv.cite_edges",
            edge_src_column="src",
            edge_dst_column="dst",
            edge_direction="out",
        ),
        note="Declared citation graph — the BYOG signal embeddings cannot reconstruct.",
    ),
    "arxiv.propagated": Artifact(
        name="arxiv.propagated",
        kind="parquet",
        filename="propagated.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="tier02",
        note="cite-edge propagation (propagate_embeddings, degree_normalized + alpha = APPNP).",
    ),
    "arxiv.ft_model": Artifact(
        name="arxiv.ft_model",
        kind="model_id",
        filename="ft_model.json",
        produced_by="tier03",
        note="fine_tune_graph(edge_provenance='declared') checkpoint id + hash.",
    ),
    "arxiv.ctx_predictor": Artifact(
        name="arxiv.ctx_predictor",
        kind="model_id",
        filename="ctx_predictor.json",
        produced_by="tier04",
        note="train_context_predictor(attncnp, gaussian, crps) checkpoint id for the "
        "year-regression-conformal win + the source/context_ref provenance rail.",
    ),
    "arxiv.cal_split": Artifact(
        name="arxiv.cal_split",
        kind="split",
        filename="cal_split.json",
        columns={"calibration": "[Utf8]", "test": "[Utf8]"},
        produced_by="tier04",
        note="Calibration/test row-id partition; disjoint from train and from each other.",
    ),
    "arxiv.tier04_predictions": Artifact(
        name="arxiv.tier04_predictions",
        kind="parquet",
        filename="tier04_predictions.parquet",
        columns={"paper_id": "Utf8", "split": "Utf8", "true_label": "Int",
                 "scores": "[f32]", "emb": "[f32]"},
        produced_by="tier04",
        note="Per-row graph-conditioned class scores + true subject label + the "
        "propagated embedding, for the calibration (2018-era) and test (2019–2020-era) "
        "splits; lets the chapter rerun marginal APS conformal + the weighting no-op "
        "demonstration on CPU.",
    ),
    "arxiv.tier04_regression": Artifact(
        name="arxiv.tier04_regression",
        kind="parquet",
        filename="tier04_regression.parquet",
        columns={"paper_id": "Utf8", "split": "Utf8", "pred_mean": "f64",
                 "pred_std": "f64", "true_year": "Int"},
        produced_by="tier04",
        note="Per-row gaussian year predictions (mean, std) + true year, for the "
        "calibration and test splits; lets the chapter rerun the engine "
        "conformalize_interval (the bidirectional-win regression-conformal) on CPU.",
    ),
    "arxiv.tier04_weighting": Artifact(
        name="arxiv.tier04_weighting",
        kind="model_id",
        filename="tier04_weighting.json",
        produced_by="tier04",
        note="The weighted-conformal no-op record: per-scheme coverage/Δ for the "
        "three test-likeness schemes + the corr(nonconformity, test-likeness) "
        "diagnostic that explains why reweighting cannot move the quantile here.",
    ),
    "arxiv.tier04_regression_weighting": Artifact(
        name="arxiv.tier04_regression_weighting",
        kind="model_id",
        filename="tier04_regression_weighting.json",
        produced_by="tier04",
        note="The regression-conformal no-op record: marginal vs test-likeness-weighted "
        "|y−ŷ| coverage + the cal/test residual-magnitude comparison and the "
        "corr(|residual|, test-likeness) diagnostic — the location-shift evidence that "
        "the under-coverage is point-prediction bias, not a residual-distribution shift.",
    ),
    "arxiv.conformal_synthetic_shift": Artifact(
        name="arxiv.conformal_synthetic_shift",
        kind="model_id",
        filename="conformal_synthetic_shift.json",
        produced_by="conformal",
        note="The score-aligned weighted-restore record — a TRANSPARENTLY-SYNTHETIC "
        "teaching device, not a property of the real time-split. A calibration "
        "subsample biased along a real embedding covariate that correlates with the "
        "APS nonconformity (corr +0.73) MOVES the score distribution; marginal "
        "split-conformal under-covers (0.833) and weighted split-conformal "
        "(Tibshirani 2019) restores coverage to ≥ nominal (0.939). The complement of "
        "the keystone's orthogonal-shift no-op: weighted conformal repairs a shift IFF "
        "it moves the nonconformity-score distribution.",
    ),
    "arxiv.subject_golden": Artifact(
        name="arxiv.subject_golden",
        kind="parquet",
        filename="subject_golden.parquet",
        columns={"query_id": "Utf8", "query_text": "Utf8", "relevant_id": "Utf8"},
        produced_by="tier01",
        note="The same-subject retrieval golden (query paper → same-subject papers) the "
        "keystone's recall target folds against; carries the query's title+abstract text "
        "(query_text) so a lexical / hybrid retriever can be measured on the same target.",
    ),
    "arxiv.calibration_report": Artifact(
        name="arxiv.calibration_report",
        kind="model_id",
        filename="calibration_report.json",
        produced_by="calibration",
        note="The eval_calibration(shape='gaussian') report on the committed tier-04 "
        "gaussian year predictions: the proper scores (CRPS, NLL), sharpness, the "
        "adaptive-ECE and PIT-KS calibration diagnostics, and the central coverage. "
        "The PIT is far from uniform (KS ≈ 0.42) — sharp but miscalibrated, the same "
        "non-exchangeability the conformal tier reports.",
    ),
    # --- fine-tune methods vertical (A1) -------------------------------------
    "finetune.method_recall": Artifact(
        name="finetune.method_recall",
        kind="parquet",
        filename="method_recall.parquet",
        columns={"method": "Utf8", "recall_at_10": "f64",
                 "recall_gain_vs_base": "f64", "dim": "Int"},
        produced_by="finetune",
        note="Per-method same-subject recall@10 (and gain vs the frozen base) for every "
        "fine-tuning method compared side-by-side on the committed ogbn-arxiv subset: "
        "cosent / mnrl (two temperatures) / triplet / hard-negatives / matryoshka / "
        "fine_tune_graph(declared). The recall is the keystone's embedding-independent "
        "same-subject target, folded by the exact cosine-kNN numpy routine per matrix.",
    ),
    "finetune.matryoshka_curve": Artifact(
        name="finetune.matryoshka_curve",
        kind="parquet",
        filename="matryoshka_curve.parquet",
        columns={"dim": "Int", "recall_at_10": "f64"},
        produced_by="finetune",
        note="recall@10 of the Matryoshka-trained matrix read at each nested dimension "
        "(full 768 and the truncated 256 / 64 prefixes) — the Matryoshka recall-vs-"
        "truncated-dim curve, recomputed on CPU off the committed embedding matrix.",
    ),
    "finetune.subject_golden": Artifact(
        name="finetune.subject_golden",
        kind="parquet",
        filename="subject_golden.parquet",
        columns={"query_id": "Utf8", "relevant_id": "Utf8"},
        produced_by="finetune",
        note="The same-subject retrieval golden (query paper → same-subject papers); the "
        "embedding-independent relevance target the per-method recall@10 is folded against.",
    ),
    "finetune.emb_matryoshka": Artifact(
        name="finetune.emb_matryoshka",
        kind="parquet",
        filename="emb_matryoshka.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="finetune",
        note="The Matryoshka-trained embedding matrix over the subset; lets the chapter "
        "recompute the truncated-dim recall curve on CPU (truncate → renormalize → fold).",
    ),
    "finetune.methods": Artifact(
        name="finetune.methods",
        kind="model_id",
        filename="methods.json",
        produced_by="finetune",
        note="The method-comparison record: base model + base recall, supervised-pair "
        "count, epochs, every method's checkpoint id + recall + gain, the Matryoshka "
        "curve, and the best/worst method + recall spread across the method spectrum.",
    ),
    # --- fine-tune regression vertical (the high-offset `year` target) --------
    "finetune_regression.methods": Artifact(
        name="finetune_regression.methods",
        kind="model_id",
        filename="methods.json",
        produced_by="finetune_regression",
        note="The regression-objective comparison record: the year target's range + "
        "std, train/test split sizes, seed, and per-loss held-out RMSE-in-years / MAE / "
        "nominal coverage / std spread for beta_nll / gaussian_nll / crps / pinball, plus "
        "the best loss and the fits-without-collapse verdict (min Gaussian std_mean far "
        "above the documented pre-0.26.2 ~0.001 variance collapse).",
    ),
    "finetune_regression.pred_beta_nll": Artifact(
        name="finetune_regression.pred_beta_nll",
        kind="parquet",
        filename="pred_beta_nll.parquet",
        columns={"paper_id": "Utf8", "true_year": "f64",
                 "predicted_mean": "f64", "predicted_std": "f64"},
        produced_by="finetune_regression",
        note="Held-out test predictions (de-standardized to raw year units) for the "
        "beta_nll Gaussian head; lets the chapter/test re-fold RMSE/coverage on CPU.",
    ),
    "finetune_regression.pred_gaussian_nll": Artifact(
        name="finetune_regression.pred_gaussian_nll",
        kind="parquet",
        filename="pred_gaussian_nll.parquet",
        columns={"paper_id": "Utf8", "true_year": "f64",
                 "predicted_mean": "f64", "predicted_std": "f64"},
        produced_by="finetune_regression",
        note="Held-out test predictions for the gaussian_nll head; CPU re-fold source.",
    ),
    "finetune_regression.pred_crps": Artifact(
        name="finetune_regression.pred_crps",
        kind="parquet",
        filename="pred_crps.parquet",
        columns={"paper_id": "Utf8", "true_year": "f64",
                 "predicted_mean": "f64", "predicted_std": "f64"},
        produced_by="finetune_regression",
        note="Held-out test predictions for the crps head; CPU re-fold source.",
    ),
    "finetune_regression.pred_pinball": Artifact(
        name="finetune_regression.pred_pinball",
        kind="parquet",
        filename="pred_pinball.parquet",
        columns={"paper_id": "Utf8", "true_year": "f64", "quantile_0.05": "f64",
                 "quantile_0.5": "f64", "quantile_0.95": "f64"},
        produced_by="finetune_regression",
        note="Held-out test quantile predictions (q05/q50/q95) for the pinball head; "
        "lets the chapter/test re-fold the median-point RMSE and the [q05,q95] coverage.",
    ),
    # --- Air Routes (KV-air on-ramp) -----------------------------------------
    "air.airports": Artifact(
        name="air.airports",
        kind="parquet",
        filename="airports.parquet",
        columns={"code": "Utf8", "city": "Utf8", "country": "Utf8",
                 "continent": "Utf8", "region": "Utf8"},
        produced_by="tier01",
        note="Subset airport labels (code, city, country, continent, region) — for "
        "display-only derived numbers (homophily, cohort sizes); the embedded "
        "desc/city/country/region text is not committed.",
    ),
    "air.embeddings": Artifact(
        name="air.embeddings",
        kind="parquet",
        filename="embeddings.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="tier01",
        note="Embeddings of airport rows (numeric/categorical + thin desc/city text).",
    ),
    "air.neighbor_graph": Artifact(
        name="air.neighbor_graph",
        kind="edge_table",
        filename="neighbor_graph.parquet",
        columns={"src": "Utf8", "dst": "Utf8", "rank": "Int", "similarity": "f32"},
        produced_by="tier01",
        note="Derived self-kNN similarity graph over airports.",
    ),
    "air.route_edges": Artifact(
        name="air.route_edges",
        kind="edge_table",
        filename="route_edges.parquet",
        columns={"src": "Utf8", "dst": "Utf8", "dist": "f32"},
        produced_by="external",
        declared_edges=DeclaredEdges(
            edge_source="air.route_edges",
            edge_src_column="src",
            edge_dst_column="dst",
            edge_weight_column="dist",
            edge_direction="undirected",
        ),
        note="Declared airport↔airport route graph (Neptune's own).",
    ),
    "air.contains_edges": Artifact(
        name="air.contains_edges",
        kind="edge_table",
        filename="contains_edges.parquet",
        columns={"src": "Utf8", "dst": "Utf8"},
        produced_by="external",
        declared_edges=DeclaredEdges(
            edge_source="air.contains_edges",
            edge_src_column="src",
            edge_dst_column="dst",
            edge_direction="out",
        ),
        note="Declared continent→country→airport containment hierarchy.",
    ),
    "air.propagated": Artifact(
        name="air.propagated",
        kind="parquet",
        filename="propagated.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="tier02",
        note="route-graph propagation (propagate_embeddings, degree_normalized).",
    ),
    "air.tenancy": Artifact(
        name="air.tenancy",
        kind="model_id",
        filename="tenancy.json",
        produced_by="tier01",
        note="The tenancy record: the engine's two genuine isolation layers + the "
        "honest caveat. Catalog-listing isolation (tenant A's list_sources excludes "
        "B's source — hard zero) and row-level discriminator-column isolation (a "
        "tenant_id-tagged source returns disjoint rows under A vs B — hard zero), "
        "plus the global-source caveat (a discriminator-less source is globally "
        "readable: A sees all of B's rows when it names the source).",
    ),
    # --- retrieval / search vertical (B1) ------------------------------------
    "retrieval.method_metrics": Artifact(
        name="retrieval.method_metrics",
        kind="parquet",
        filename="method_metrics.parquet",
        columns={"method": "Utf8", "family": "Utf8",
                 "recall_at_10": "f64", "ndcg_at_10": "f64"},
        produced_by="retrieval",
        note="Per-method retrieval recall@10 + nDCG@10 over the committed ogbn-arxiv "
        "matrices, against the same-subject golden: dense (raw / propagated) cosine-kNN, "
        "lexical BM25 over titles, and the rrf_fuse fusions (dense+dense, dense+lexical). "
        "All folded on CPU from the committed cache; the dense numbers reproduce the "
        "keystone's per-table recall.",
    ),
    "retrieval.finding": Artifact(
        name="retrieval.finding",
        kind="model_id",
        filename="retrieval.json",
        produced_by="retrieval",
        note="The honest fusion finding + the search-multi-table engine finding. RRF "
        "fusion does NOT beat the best single arm on this same-subject target "
        "(best fusion 0.550 < best single arm 0.556); fusing in a weaker arm cannot "
        "exceed the stronger ranker it contains. search_finding records that "
        "search(source, ...) carries no table= argument and is ambiguous once a source "
        "has several embedding tables — a candidate for an explicit table= argument.",
    ),
    # --- mutable companion table / feature store (C1) ------------------------
    "feature_store.paper_features": Artifact(
        name="feature_store.paper_features",
        kind="parquet",
        filename="paper_features.parquet",
        columns={"paper_id": "Utf8", "in_degree": "Int"},
        produced_by="feature_store",
        note="The committed per-paper feature: each paper's citation in-degree over the "
        "committed declared citation graph (arxiv.cite_edges) — a real graph-derived "
        "'how-cited-is-this-paper' feature, the kind a feature store holds. The chapter "
        "loads these rows, INSERTs them into a mutable companion table, and JOINs them "
        "into a query over the papers source. Every paper carries a value (0 if uncited).",
    ),
    "feature_store.record": Artifact(
        name="feature_store.record",
        kind="model_id",
        filename="feature_store.json",
        produced_by="feature_store",
        note="The mutable-companion-table record: the feature provenance, the populated "
        "row count, the subject-level SUM(in_degree) JOIN aggregate (top subject + grand "
        "total), and the append-only probe (UPDATE / DELETE / duplicate-key INSERT are "
        "each rejected on this surface — in-place upsert is a forthcoming engine "
        "capability not yet exposed here). The table is addressed in SQL as "
        "mutable.public.paper_features; the registered papers parquet as papers.public.papers.",
    ),
    # --- change-data-capture / trigger topics (C2) ---------------------------
    "cdc.record": Artifact(
        name="cdc.record",
        kind="model_id",
        filename="cdc.json",
        produced_by="cdc",
        note="The change-data-capture record: a deterministic stream of N record-change "
        "events (op add/update/remove on a key, keyed by a committed arxiv paper_id) "
        "published as single-row batches onto a trigger topic at offsets 0..N-1, then "
        "drained by bounded offset-addressed replay-collect. Carries the op counts, the "
        "predicate, the checkpoint offset, and the measured collect counts (full replay, "
        "predicate-filtered add, checkpoint-resumed tail, resume+filter). The load-bearing "
        "property recorded here: subscribe_collect's max_batches is the TERMINATOR — the "
        "call returns synchronously iff max_batches equals the exact number of yielded "
        "batches from from_offset (num_published - from_offset with no predicate; the count "
        "of offset>=from_offset batches with >=1 matching row with a predicate), because "
        "the in-memory broker tail otherwise blocks forever waiting for the next publish.",
    ),
    # --- eval on the wire + provenance channels (chapter 14) -----------------
    "eval.embeddings": Artifact(
        name="eval.embeddings",
        kind="model_id",
        filename="embeddings.json",
        produced_by="eval",
        note="The embedded-canonical eval_embeddings report over the engine's public "
        "patents fixture embedded with tiny_modernbert against golden_relevance: the "
        "aggregate (recall_at_k / precision_at_k / mrr / ndcg over the golden queries) "
        "and the per_query records (sorted by query_id, instance keys stripped). The "
        "wire SURFACE on a public deterministic fixture — NOT the arxiv keystone "
        "corpus (that stays the ch05 numpy fold).",
    ),
    "eval.per_query": Artifact(
        name="eval.per_query",
        kind="model_id",
        filename="per_query.json",
        produced_by="eval",
        note="The persisted per-query eval rows read back via eval_per_query for the "
        "embeddings run — one row per golden query carrying its cohorts tag and metrics "
        "dict (sorted by query_id, eval_run_id stripped from the committed form).",
    ),
    "eval.inference_cls": Artifact(
        name="eval.inference_cls",
        kind="model_id",
        filename="inference_cls.json",
        produced_by="eval",
        note="The embedded-canonical eval_inference (classification) report: the "
        "tiny_modernbert_classifier over the patents abstracts against tiny_labels — a "
        "task-tagged aggregate (accuracy / f1 / per_class) and per_record predictions "
        "(sorted by record_id).",
    ),
    "eval.inference_ner": Artifact(
        name="eval.inference_ner",
        kind="model_id",
        filename="inference_ner.json",
        produced_by="eval",
        note="The embedded-canonical eval_inference (NER) report: the tiny_modernbert_ner "
        "over the tiny_ner_corpus against the tiny_ner_gold char-offset spans — a "
        "task-tagged aggregate (entity-span precision / recall / f1 / per_type) and "
        "per_record predictions. The Python + remote NER eval coverage the engine's own "
        "Python live tests (classification-only) do not carry.",
    ),
    "eval.compare_self": Artifact(
        name="eval.compare_self",
        kind="model_id",
        filename="compare_self.json",
        produced_by="eval",
        note="The eval_compare SELF-comparison (the determinism anchor): the same "
        "embedding table compared against itself — a baseline with delta None and a "
        "treatment whose every metric delta is exactly 0.0 with significance present "
        "(a CI collapsed onto zero).",
    ),
    "eval.compare_two": Artifact(
        name="eval.compare_two",
        kind="model_id",
        filename="compare_two.json",
        produced_by="eval",
        note="The eval_compare TWO-table comparison (the non-degenerate case): the same "
        "patents corpus embedded with two different models (tiny_modernbert vs tiny_bert) "
        "— a baseline with delta None and a treatment carrying a genuine non-zero recall "
        "delta with a real significance block.",
    ),
    "eval.channels": Artifact(
        name="eval.channels",
        kind="model_id",
        filename="channels.json",
        produced_by="eval",
        note="The embedded-canonical channel listings: tenant A's register/append/list "
        "(scored_by + annotated_by, append-order), tenant B's view before/after "
        "registering its own scored_by (the #170 isolation + non-collision property), and "
        "the unbound (global-seed-only) listing. Generic provenance channel ids, opaque "
        "tenant UUIDs — names no consumer.",
    ),
    "eval.eval": Artifact(
        name="eval.eval",
        kind="model_id",
        filename="eval.json",
        produced_by="eval",
        note="The provenance + parity record: the eval + channel verbs ran on BOTH the "
        "embedded engine and a live remote grpc:// jammi-server, asserted remote == "
        "embedded live (shape + value-close, tol 1e-9, instance keys excluded) — the real "
        "one-time cross-transport parity check, continuously re-guarded by the engine's "
        "gated test_remote_eval_live.py / test_remote_channel_live.py. Records the parity "
        "verdict per verb + the measured aggregates. PR CI reads the embedded-canonical "
        "reports and asserts aggregates-to-golden, never a static re-diff.",
    ),
    # --- scale tier: full ogbn-arxiv ANN-vs-exact recall cross-check (H2) -----
    "scale.corpus_vectors": Artifact(
        name="scale.corpus_vectors",
        kind="parquet",
        filename="arxiv_vectors.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="scale-emit",
        note="The INDEXED corpus: ModernBERT (title+abstract) embeddings of every "
        "ogbn-arxiv node with embeddable text MINUS the held-out query split — "
        "(_row_id, vector), 768-dim f32, sorted by _row_id. The numpy exact "
        "cosine-kNN oracle scans this matrix; the frozen ANN index is built over it. "
        "Committed once on the GPU emit box (Git LFS), never re-embedded in CI.",
    ),
    "scale.query_vectors": Artifact(
        name="scale.query_vectors",
        kind="parquet",
        filename="scale_query_vectors.parquet",
        columns={"_row_id": "Utf8", "vector": _vec(_EMB_DIM)},
        produced_by="scale-emit",
        note="The HELD-OUT query vectors: the last 1000 paper_ids by ascending sorted "
        "paper_id, embedded but NOT indexed. Disjoint from scale.corpus_vectors by "
        "construction, so each query's true nearest neighbour is a DIFFERENT paper and "
        "recall@1 is a real floor (≈0.97, not the structural ≈1.0 of a self-query split).",
    ),
    "scale.ann_index": Artifact(
        name="scale.ann_index",
        kind="usearch_bundle",
        filename="arxiv_ann.usearch",
        produced_by="scale-emit",
        note="The FROZEN ANN sidecar: a usearch HNSW index (cosine, 768-dim f32) over "
        "the corpus, built ONCE by the engine's own SidecarIndex builder on the emit "
        "box and committed (Git LFS). CI LOADS it and never rebuilds (default "
        "IndexOptions are non-deterministic). Its companion arxiv_ann.rowmap maps the "
        "index's sequential keys (0..N−1) back to corpus paper_ids.",
    ),
    "scale.ann_rowmap": Artifact(
        name="scale.ann_rowmap",
        kind="rowmap",
        filename="arxiv_ann.rowmap",
        produced_by="scale-emit",
        note="The sidecar's key→paper_id map: a u32 version header followed by "
        "length-prefixed (u32 little-endian) ASCII paper_id records in index-key order. "
        "Reading it turns a usearch search result's integer keys into corpus paper_ids.",
    ),
    "scale.queries": Artifact(
        name="scale.queries",
        kind="id_list",
        filename="scale_queries.txt",
        produced_by="scale-emit",
        note="The held-out query paper_ids, one per line — the committed identity of the "
        "recall probe set (subset identity is committed, not seeded).",
    ),
    "scale.manifest": Artifact(
        name="scale.manifest",
        kind="model_id",
        filename="scale_manifest.json",
        produced_by="scale-emit",
        note="The emit provenance: corpus/query/dim counts, the held-out split rule, the "
        "engine git SHA + usearch version the frozen sidecar was built with, the metric, "
        "and the load-frozen-only / floors-are-measured-minus-margin contract.",
    ),
    "scale.checksums": Artifact(
        name="scale.checksums",
        kind="model_id",
        filename="checksums.json",
        produced_by="scale-emit",
        note="sha256[:16] of every committed scale artifact — the cross-check verifies "
        "the cache against these before recomputing recall, so a corrupted LFS pull "
        "fails loudly rather than producing a quietly-wrong number.",
    ),
    "scale.recall_sweep_build": Artifact(
        name="scale.recall_sweep_build",
        kind="model_id",
        filename="recall_sweep_build.json",
        produced_by="scale-emit",
        note="The build-knob recall-vs-cost reference: the engine's `jammi-bench "
        "recall-sweep` `recall_sweep` tier emitted ONCE over the full 168k corpus — per "
        "(connectivity, build_expansion) point the build time, on-disk index size, and a "
        "recall reference. The swept graphs themselves are NOT committed (hundreds of MiB "
        "each, LFS-prohibitive), so unlike the re-dialed ef_search curve this axis is an "
        "un-gated reference: the build-cost columns are the deliverable, recall is provenance.",
    ),
    # --- model catalog, measured cross-transport (§3.6) ----------------------
    "lifecycle.matrix": Artifact(
        name="lifecycle.matrix",
        kind="model_id",
        filename="matrix.json",
        produced_by="lifecycle",
        note="The embedded-canonical model-catalog matrix: the model projections "
        "(register / describe / list, model_id UUID stripped) and the "
        "referential-integrity delete cells captured as comparable OUTCOMES — "
        "delete-while-referenced (normalized error class 'referenced', the "
        "ModelReferenced guard), delete-absent-strict ('not_found', the typed "
        "ModelNotFound — NOT invalid-argument), delete-absent-if_exists (a no-op), and "
        "the every_catalog_model_is_referenced property (every model in the catalog is "
        "trained-and-referenced, so there is no unreferenced model to delete). "
        "Registered via a tiny CPU fine_tune — the only public registration path; the "
        "only GPU-free, names-no-consumer way the catalog gets a model row.",
    ),
    "lifecycle.lifecycle": Artifact(
        name="lifecycle.lifecycle",
        kind="model_id",
        filename="lifecycle.json",
        produced_by="lifecycle",
        note="The provenance + parity record: the catalog verbs ran on BOTH the "
        "embedded engine and a live remote grpc:// jammi-server, asserted remote == "
        "embedded live for every observable — the model projections (UUID excluded) "
        "and the NORMALIZED delete-error class (the two transports raise different "
        "Python exception types; the honest contract is the error CLASS). Records the "
        "parity verdict, the measured matrix, the training-only registration path, and "
        "the every-catalog-model-is-referenced property. PR CI reads the "
        "embedded-canonical matrix and asserts verdicts-to-golden, never a static "
        "re-diff.",
    ),
    # --- channel error taxonomy, measured cross-transport (§3.8) --------------
    "channels.matrix": Artifact(
        name="channels.matrix",
        kind="model_id",
        filename="matrix.json",
        produced_by="channels",
        note="The channel error-taxonomy matrix (engine #193): each evidence-channel "
        "failure mode driven on BOTH transports, captured as a comparable OUTCOME — the "
        "remote arm carries the typed gRPC StatusCode (duplicate→ALREADY_EXISTS, "
        "unknown→NOT_FOUND, column_conflict→FAILED_PRECONDITION, "
        "bad_argument→INVALID_ARGUMENT), the embedded arm the same NORMALIZED error class "
        "(duplicate/unknown/conflict/bad_argument) with no wire code. Carries the "
        "embedded + remote cells per mode, the expected wire/class maps, and the "
        "client-side dtype guard (a ValueError that never reaches the wire). Generic "
        "channel ids, opaque tenant UUIDs — names no consumer.",
    ),
    "channels.channels_taxonomy": Artifact(
        name="channels.channels_taxonomy",
        kind="model_id",
        filename="channels_taxonomy.json",
        produced_by="channels",
        note="The provenance + parity record for §3.8: the channel failure modes ran on "
        "BOTH the embedded engine and a live remote grpc:// jammi-server; the wire "
        "StatusCode was measured on the grpc:// transport (where the codes exist), the "
        "embedded normalized error CLASS the cross-transport companion, asserted remote "
        "== embedded class for every mode. Records the measured (mode → wire code) "
        "taxonomy (every mode maps as #193 intended — Internal-for-everything replaced by "
        "typed codes), the documented INTERNAL residual (not fabricated), the "
        "client-side dtype guard property, and the parity verdict. PR CI reads the "
        "committed matrix and asserts the taxonomy-to-golden, never a live re-drive.",
    ),
    # --- per-verb tenant isolation + BYO-auth seam (H3 §3.5) -----------------
    "tenancy_h3.matrix": Artifact(
        name="tenancy_h3.matrix",
        kind="model_id",
        filename="matrix.json",
        produced_by="tenancy_h3",
        note="The embedded-canonical per-verb isolation matrix (the §3.5 standing "
        "oracle): for every tenant-scoped verb, tenant B's listing/read/reach of tenant "
        "A's resource, as a hard-zero leak count (list_sources / describe_source / "
        "list_mutable_tables / list_topics / list_channels / list_models / the "
        "discriminator-column sql row read — every leak 0) or a documented "
        "stated-positive (the built-in global channels B sees; the discriminator-less "
        "source A reads whole). Plus the collision cells (a duplicate mutable-table name "
        "ERRORS on the global PK; duplicate topic/channel ids isolate per-tenant — never "
        "a clobber) and the destructive-verb survival cells (A's mutable table / topic "
        "SURVIVES a foreign-tenant drop — no cross-tenant destruction). Generic verbs, "
        "opaque tenant UUIDs — names no consumer.",
    ),
    "tenancy_h3.record": Artifact(
        name="tenancy_h3.record",
        kind="model_id",
        filename="tenancy_h3.json",
        produced_by="tenancy_h3",
        note="The provenance + parity + BYO-auth record for §3.5: the per-verb matrix ran "
        "on BOTH the embedded engine and a live remote grpc:// jammi-server, asserted "
        "remote == embedded for every cross-transport observable (the catalog reads + the "
        "discriminator sql row read). Carries the parity verdict, the measured matrix, the "
        "explicit no-leak finding (the drop_mutable_table cross-tenant-destruction defect "
        "flagged in scouting is NOT present on the pinned 0.30.0 engine — B's drop resolves "
        "in B's own namespace, A's table survives), and the BYO-auth seam verdict (a "
        "generic HMAC-bearer-token gateway in front of the engine's tenant binding: two "
        "authenticated tenants isolated, a missing/invalid credential rejected not run "
        "unscoped). PR CI reads the committed matrix and asserts verdicts-to-golden.",
    ),
    # --- tenant isolation as a measured property (B2) ------------------------
    "tenancy_b.record": Artifact(
        name="tenancy_b.record",
        kind="model_id",
        filename="tenancy_b.json",
        produced_by="tenancy_b",
        note="The tenant-isolation properties measured LIVE on CPU against the committed "
        "ogbn-arxiv cache: catalog-listing isolation (hard zero), discriminator-column "
        "row isolation (hard zero), the discriminator-less caveat (a positive visible "
        "count), and tenant-conditioned metric parity (the same recall recipe under two "
        "tenants yields each its own scoped result over a disjoint row partition).",
    ),
}


def artifact(name: str) -> Artifact:
    """Return the registered :class:`Artifact` contract for ``name``."""
    try:
        return ARTIFACTS[name]
    except KeyError:
        raise KeyError(f"unknown artifact '{name}'. Registered: {sorted(ARTIFACTS)}") from None


# --------------------------------------------------------------------------- #
# Layer 2 — golden samples
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Golden:
    """A frozen metric: the committed value and the tolerance CI asserts within."""

    value: float
    tol: float

    def contains(self, observed: float) -> bool:
        return abs(observed - self.value) <= self.tol


def _dataset_dir(dataset: str) -> Path:
    return _ARTIFACT_ROOT / dataset


def golden(metric: str) -> Golden:
    """Read one frozen metric, addressed ``"<dataset>.<key>"``.

    e.g. ``golden("arxiv.tier04.marginal_coverage")`` reads
    ``artifacts/arxiv/golden_metrics.json`` key ``"tier04.marginal_coverage"``.
    Raises if the metrics file or key is absent — a chapter that asserts a metric
    the keystone never emitted is a contract gap, surfaced loudly, not a pass.
    """
    dataset, _, key = metric.partition(".")
    if not key:
        raise ValueError(f"metric must be '<dataset>.<key>', got {metric!r}")
    path = _dataset_dir(dataset) / "golden_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"golden metrics not found: {path}. The keystone slice (KV-arxiv) emits "
            f"this once; chapters read it and never recompute upstream."
        )
    data = json.loads(path.read_text())
    if key not in data:
        raise KeyError(f"metric '{key}' not in {path} (have: {sorted(data)})")
    entry = data[key]
    return Golden(value=float(entry["value"]), tol=float(entry["tol"]))


def assert_close(metric: str, observed: float) -> float:
    """Assert ``observed`` matches the frozen ``metric`` within tolerance.

    The measured-verdict oracle every recipe ends in. Returns ``observed`` so a
    chapter cell can both assert and display in one expression. The error names
    the gap and the provenance, so a tolerance failure is diagnosable, not opaque.
    """
    g = golden(metric)
    if not g.contains(observed):
        raise AssertionError(
            f"{metric}: observed {observed:.4f} outside golden {g.value:.4f} "
            f"± {g.tol:.4f} (Δ={abs(observed - g.value):.4f}). Either upstream was "
            f"recomputed (forbidden — read the cache) or the artifact drifted."
        )
    return observed


def load_artifact(name: str):
    """Load a committed artifact by registered name (never recomputes).

    parquet/edge_table → ``pyarrow.Table``; model_id/split → the parsed JSON
    record. Raises if the file is absent: the cache is the source of truth, and a
    missing entry means the keystone has not emitted it yet — surfaced, not faked.
    """
    art = artifact(name)
    path = _dataset_dir(art.dataset) / art.filename
    if not path.exists():
        raise FileNotFoundError(
            f"artifact '{name}' not found at {path}. It is produced by {art.produced_by} "
            f"in the keystone slice; chapters load it and never recompute it."
        )
    if art.kind in ("parquet", "edge_table"):
        import pyarrow.parquet as pq

        return pq.read_table(path)
    if art.kind in ("usearch_bundle", "rowmap"):
        # Binary sidecar artifacts: hand back the resolved path. The cross-check
        # loads them with the usearch library / a small rowmap decoder — the
        # registry only resolves *where* they live.
        return path
    if art.kind == "id_list":
        return path.read_text().split()
    return json.loads(path.read_text())


@dataclass(frozen=True)
class RecallFloor:
    """A one-sided recall floor: the committed ``floor`` (= measured − margin).

    The scale-tier ANN-vs-exact recall is asserted as ``observed >= floor`` (an
    order-insensitive set-intersection floor), NOT as a two-sided tolerance. ANN
    recall is allowed to be *better* than the committed floor — a frozen HNSW index
    searched on a different box may return a slightly different (still-valid)
    neighbour set, and a higher recall is never a regression. ``measured`` is the
    emit-box reading recorded for provenance.
    """

    floor: float
    margin: float
    measured: float

    def clears(self, observed: float) -> bool:
        return observed >= self.floor


def recall_floor(metric: str) -> RecallFloor:
    """Read one frozen recall floor, addressed ``"scale.recall_at_<k>"``.

    Reads ``artifacts/scale/golden_metrics.json`` → ``recall.recall_at_<k>``, whose
    value is a ``{floor, margin, measured}`` record (the scale tier's recall is a
    one-sided floor, not a two-sided golden — see :class:`RecallFloor`). Raises if
    the metrics file or key is absent: a floor the emit never recorded is a contract
    gap, surfaced, not a silent pass.
    """
    dataset, _, key = metric.partition(".")
    if not key:
        raise ValueError(f"metric must be 'scale.recall_at_<k>', got {metric!r}")
    path = _dataset_dir(dataset) / "golden_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"scale golden metrics not found: {path}. The scale-tier emit writes this "
            f"once on the GPU box; the cross-check reads it and never re-embeds."
        )
    recall = json.loads(path.read_text()).get("recall", {})
    if key not in recall:
        raise KeyError(f"recall floor '{key}' not in {path} (have: {sorted(recall)})")
    entry = recall[key]
    return RecallFloor(
        floor=float(entry["floor"]),
        margin=float(entry["margin"]),
        measured=float(entry["measured"]),
    )


def assert_recall_floor(metric: str, observed: float) -> float:
    """Assert a recomputed recall ``observed`` clears the frozen ``metric`` floor.

    The scale-tier measured-verdict oracle: ``observed >= floor`` (one-sided). Returns
    ``observed`` so a chapter cell asserts and displays at once. The error names the
    gap and the provenance, so a sub-floor reading is diagnosable — and a sub-floor
    reading means the cache or the floor is wrong, never that the floor should be
    loosened.
    """
    rf = recall_floor(metric)
    if not rf.clears(observed):
        raise AssertionError(
            f"{metric}: recomputed recall {observed:.4f} is BELOW the committed floor "
            f"{rf.floor:.4f} (emit-box measured {rf.measured:.4f}, margin {rf.margin:.4f}). "
            f"The frozen ANN index or the committed cache is wrong — do NOT loosen the "
            f"floor; the cross-check has caught a real regression."
        )
    return observed
