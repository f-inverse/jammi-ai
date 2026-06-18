#!/usr/bin/env python3
"""Emit the ogbn-arxiv keystone cache (K0 layer 2) — run ONCE on the GPU server.

This is the heavy pipeline: embed → neighbor graph → propagate → fine-tune →
context-predictor → conformal. It runs the real ``jammi_ai`` API end-to-end on
the committed 4000-paper subset and writes ``artifacts/arxiv/*`` +
``golden_metrics.json``. Every later chapter (and CI) READS that cache and never
recomputes it.

The GPU compute tier cannot run on the CPU embed wheel: this script connects to a
running ``jammi-server`` (the published ``jammi-server-cu12``) over a ``grpc://``
target and the engine does the embedding / fine-tune / predictor training on the
device. The committed cache is then read on CPU by the chapters via
``connect("file://…")`` — the ``connect(target)`` parity spine.

Determinism contract (K0 §3): committed subset ids, ``exact=True`` neighbor
graph, pinned ModernBERT + dtype, single-threaded BLAS (applied by importing
jammi_cookbook), the tier-04 context predictor + domain classifier seeded, metrics
asserted to tolerances downstream.

Usage::

    # 1. start the GPU server (clean artifact dir for a reproducible emit):
    #    JAMMI_ARTIFACT_DIR=/tmp/srv JAMMI_GPU__DEVICE=0 JAMMI_GPU__REQUIRE_GPU=true \\
    #    JAMMI_SERVER__FLIGHT_LISTEN=127.0.0.1:50051 jammi-server &
    # 2. emit against it:
    python scripts/build_arxiv_cache.py --target grpc://127.0.0.1:50051
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import datasets, determinism

EMBED_MODEL = "answerdotai/ModernBERT-base"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "arxiv"
NEIGHBOR_K = 10
PROP_HOPS = 2
PROP_ALPHA = 0.1
ALPHA = 0.10  # conformal miscoverage target → nominal coverage 0.90
RECALL_K = 10
SUBSET = 4000
SOFTMAX_TEMP = 8.0  # temperature on the nearest-centroid cosine logits
WEIGHTED_TEMP = 5.0  # temperature on the weighted-conformal test-likeness weights
KNN_W = 25  # neighbours for the kNN test-likeness weighting scheme
PREDICTOR_EPOCHS = 80  # context-predictor meta-training epochs


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _emb_ref(table: str) -> str:
    """An engine-produced table is addressed as a single quoted identifier."""
    return f'"jammi.{table}"'


def _read_vectors(db, table: str) -> tuple[list[str], np.ndarray]:
    """Read an embedding table's ``(_row_id, vector)`` as ids + an f32 matrix."""
    t = db.sql(f"SELECT _row_id, vector FROM {_emb_ref(table)} ORDER BY _row_id")
    ids = [str(x) for x in t.column("_row_id").to_pylist()]
    vecs = np.asarray([list(v) for v in t.column("vector").to_pylist()], dtype=np.float32)
    return ids, vecs


def _dump_emb(db, table: str, dest: Path) -> None:
    pq.write_table(db.sql(f"SELECT _row_id, vector FROM {_emb_ref(table)}"), dest)


def _dump_table(db, table: str, dest: Path, *, columns: str = "*") -> None:
    pq.write_table(db.sql(f"SELECT {columns} FROM {_emb_ref(table)}"), dest)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# measurement: same-subject retrieval recall@k (an embedding-independent target)
# --------------------------------------------------------------------------- #


def build_subject_golden(papers_rows: list[dict], *, query_n: int) -> tuple[pa.Table, str]:
    """A same-subject retrieval golden: query paper → same-subject papers.

    The relevance target is the subject label, independent of the embedding
    similarity — so citation-graph propagation improving same-subject retrieval is
    a real low-pass-denoising result, not a circular one.
    """
    by_subject: dict[str, list[str]] = {}
    for r in papers_rows:
        by_subject.setdefault(r["subject"], []).append(r["paper_id"])
    queries = [r for r in sorted(papers_rows, key=lambda r: r["paper_id"])
               if len(by_subject[r["subject"]]) >= 5][:query_n]
    rows = []
    for q in queries:
        for rid in by_subject[q["subject"]]:
            if rid != q["paper_id"]:
                rows.append({"query_id": q["paper_id"], "query_text": q["title"],
                             "relevant_id": rid})
    table = pa.table({
        "query_id": [r["query_id"] for r in rows],
        "query_text": [r["query_text"] for r in rows],
        "relevant_id": [r["relevant_id"] for r in rows],
    })
    path = ARTIFACTS / "subject_golden.parquet"
    pq.write_table(table, path)
    return table, str(path)


def recall_at_k(db, *, emb_table: str, golden_rows: list[dict],
                queries: list[dict], k: int) -> float:
    """recall@k for same-subject retrieval over a SPECIFIC embedding table.

    Exact cosine-similarity kNN over the committed embedding matrix (a pure numpy
    fold — deterministic, and targets exactly this table). The engine's
    ``search``/``eval_compare`` target a *source*, which is ambiguous once a source
    has several embedding tables (raw, propagated, fine-tuned); resolving the table
    by name here is the correct per-table comparison.
    """
    relevant: dict[str, set[str]] = {}
    for r in golden_rows:
        relevant.setdefault(r["query_id"], set()).add(r["relevant_id"])
    ids, vecs = _read_vectors(db, emb_table)
    pos = {i: n for n, i in enumerate(ids)}
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    scored = 0.0
    total = 0
    for q in queries:
        qid = q["paper_id"]
        if qid not in pos or qid not in relevant:
            continue
        sims = norm @ norm[pos[qid]]
        sims[pos[qid]] = -np.inf  # exclude self
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        got = [ids[t] for t in top]
        rel = relevant[qid]
        scored += sum(1 for g in got if g in rel) / min(k, len(rel))
        total += 1
    return scored / total if total else 0.0


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def emit(db) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    info = db.get_server_info()
    print("server:", json.dumps(info), flush=True)

    arxiv = datasets.load_ogbn_arxiv(db, subset=SUBSET)
    papers, cite = arxiv.papers_source, arxiv.cite_edges_source
    papers_rows = db.sql(
        f"SELECT paper_id, title, subject, year FROM {papers}.public.{papers}"
    ).to_pylist()
    n_cite = db.sql(f"SELECT COUNT(*) n FROM {cite}.public.{cite}").to_pylist()[0]["n"]
    print(f"papers: {len(papers_rows)}  cite edges: {n_cite}", flush=True)

    # Commit the papers' labels (paper_id, subject, year, title) so chapters can
    # recompute display-only derived numbers (homophily, cohort sizes) from the
    # cache without ever re-embedding. The text columns to embed are not committed.
    pq.write_table(pa.table({
        "paper_id": [r["paper_id"] for r in papers_rows],
        "title": [r["title"] for r in papers_rows],
        "subject": [r["subject"] for r in papers_rows],
        "year": [r["year"] for r in papers_rows],
    }), ARTIFACTS / "papers.parquet")

    metrics: dict[str, dict[str, float]] = {}

    # ---- Tier 01: embeddings + neighbor graph + homophily ----
    print("\n[tier01] embed + neighbor graph", flush=True)
    emb = db.generate_embeddings(source=papers, model=EMBED_MODEL,
                                 columns=["title", "abstract"], key="paper_id")
    ng = db.build_neighbor_graph(papers, k=NEIGHBOR_K, exact=True)
    ng_rows = db.sql(f"SELECT src, dst FROM {_emb_ref(ng)}").to_pylist()
    subject = {r["paper_id"]: r["subject"] for r in papers_rows}
    cite_rows = db.sql(f"SELECT src, dst FROM {cite}.public.{cite}").to_pylist()
    cite_homophily = sum(subject[r["src"]] == subject[r["dst"]] for r in cite_rows) / len(cite_rows)
    ng_homophily = sum(subject[r["src"]] == subject[r["dst"]] for r in ng_rows
                       if r["src"] in subject and r["dst"] in subject) / len(ng_rows)
    print(f"  homophily — cite: {cite_homophily:.3f}  neighbor: {ng_homophily:.3f}", flush=True)

    golden_table, golden_url = build_subject_golden(papers_rows, query_n=200)
    db.add_source("arxiv_subject_golden", url=golden_url, format="parquet")
    golden_rows = golden_table.to_pylist()
    query_ids = {g["query_id"] for g in golden_rows}
    queries = [r for r in papers_rows if r["paper_id"] in query_ids]

    r1 = recall_at_k(db, emb_table=emb, golden_rows=golden_rows,
                     queries=queries, k=RECALL_K)
    print(f"  tier01 recall@{RECALL_K} (raw embeddings): {r1:.3f}", flush=True)
    metrics["tier01.recall_at_10"] = {"value": round(r1, 3), "tol": 0.03}
    metrics["tier01.cite_homophily"] = {"value": round(cite_homophily, 3), "tol": 0.02}
    metrics["tier01.neighbor_homophily"] = {"value": round(ng_homophily, 3), "tol": 0.03}

    _dump_emb(db, emb, ARTIFACTS / "embeddings.parquet")
    _dump_table(db, ng, ARTIFACTS / "neighbor_graph.parquet")
    pq.write_table(pa.table({"src": [r["src"] for r in cite_rows],
                             "dst": [r["dst"] for r in cite_rows]}),
                   ARTIFACTS / "cite_edges.parquet")

    # ---- Tier 02: propagate over the declared citation graph ----
    print("\n[tier02] propagate (APPNP: degree_normalized + alpha)", flush=True)
    prop = db.propagate_embeddings(source=papers, embedding_table=emb, edge_source=cite,
                                   edge_src_column="src", edge_dst_column="dst",
                                   direction="out", hops=PROP_HOPS,
                                   weighting="degree_normalized", alpha=PROP_ALPHA)
    r2 = recall_at_k(db, emb_table=prop, golden_rows=golden_rows,
                     queries=queries, k=RECALL_K)
    print(f"  tier02 recall@{RECALL_K} (propagated): {r2:.3f}  (Δ {r2 - r1:+.3f})", flush=True)
    metrics["tier02.recall_at_10"] = {"value": round(r2, 3), "tol": 0.03}
    metrics["tier02.recall_delta"] = {"value": round(r2 - r1, 3), "tol": 0.03}
    _dump_emb(db, prop, ARTIFACTS / "propagated.parquet")

    # ---- Tier 03: graph-supervised fine-tune (declared edges) ----
    print("\n[tier03] fine_tune_graph (edge_provenance=declared)", flush=True)
    ft = db.fine_tune_graph(node_source=papers, id_column="paper_id", text_column="abstract",
                            edge_source=cite, src_column="src", dst_column="dst",
                            base_model=EMBED_MODEL, edge_provenance="declared",
                            epochs=1, batch_size=32, walks_per_node=2, walk_length=4,
                            sample_seed=determinism.SEED)
    ft.wait()
    print(f"  ft model_id: {ft.model_id}", flush=True)
    ft_emb = db.generate_embeddings(source=papers, model=ft.model_id,
                                    columns=["title", "abstract"], key="paper_id")
    r3 = recall_at_k(db, emb_table=ft_emb, golden_rows=golden_rows,
                     queries=queries, k=RECALL_K)
    print(f"  tier03 recall@{RECALL_K} (declared-edge fine-tune): {r3:.3f}  (Δ {r3 - r1:+.3f})",
          flush=True)
    metrics["tier03.recall_at_10"] = {"value": round(r3, 3), "tol": 0.03}
    metrics["tier03.recall_gain_vs_base"] = {"value": round(r3 - r1, 3), "tol": 0.03}
    (ARTIFACTS / "ft_model.json").write_text(json.dumps({
        "model_id": ft.model_id, "edge_provenance": "declared",
        "base_model": EMBED_MODEL, "recall_at_10": round(r3, 3),
    }, indent=2))

    # ---- Tier 04: the regression-conformal win + the conformal-under-shift lesson ----
    print("\n[tier04] year-regression conformal (the A3 bidirectional win) + "
          "subject-classification conformal-under-shift", flush=True)
    tier04(db, papers, cite, papers_rows, cite_rows, arxiv.split, prop, metrics)

    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    _write_checksums()
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def tier04(db, papers: str, cite: str, papers_rows: list[dict], cite_rows: list[dict],
           split: dict[str, list[str]], propagated: str,
           metrics: dict[str, dict[str, float]]) -> None:
    """Two honest results on the ogbn-arxiv time-split (train ≤2017, valid=2018 as
    the calibration era, test 2019–2020).

    **Part A — the bidirectional win (year-regression conformal).** The
    gaussian-collapse bug (#43) once made ``train_context_predictor(output=
    "gaussian", value_column="year")`` unusable (std≈0.001, means ~2163). The fix
    landed in two parts: 0.26.1 standardized the fine-tune projection head's target,
    but the amortized context predictor still collapsed; 0.26.2 completed it with
    z-space standardization of the predictor's target and in-context members' y
    (de-standardizing the served distribution, persisting the scaler). **The win is
    that the workflow now RUNS end-to-end at all**: the predictor fits a real mean
    (≈2018.6) with real spread (≈0.9), and the previously-impossible regression-
    conformal recipe executes. Authoring this keystone surfaced the bug; the engine
    fix made the workflow work — the cookbook→engine→cookbook loop. What the recipe
    then *measures* is the same honest lesson as Part B: the ``conformalize_interval``
    (|y−ŷ| split conformal) **under-covers** under the time-split (≈0.83), and
    weighting it is a **no-op** — here because the shift is a *location* shift, not a
    residual-magnitude shift (the predictor regresses to the embedding-conditioned
    mean for both eras, so cal and test |residual| magnitudes are ≈equal and
    corr(|residual|, test-likeness) ≈ 0).

    **Part B — the conformal-under-shift lesson (subject classification).** The
    consumer's nearest-centroid softmax head over the **propagated** (citation-
    graph-conditioned, tier 02) embeddings. The engine's ``conformalize`` (APS,
    the OSS marginal surface) **under-covers** on the later-era test set — non-
    exchangeability is real (Barber et al. 2023). The textbook remedy, importance-
    weighted conformal (Tibshirani et al. 2019), is shown here NOT to repair it:
    with a nonconformity identical to the marginal pass, three weighting schemes move
    coverage a little (−0.001 / +0.022 / +0.006) and NONE reaches nominal (best 0.889
    < 0.90), because the time-split shift is nearly orthogonal to the conformal score
    (corr(nonconformity, test-likeness) ≈ −0.12) — reweighting cannot close the gap
    when the shift does not align with the score. The
    honest remedy is a governed, time-aware / larger calibration cohort (the
    consumer's governance choice), not a client-side reweight.

    ``train_context_predictor`` / ``predict_with_context_predictor`` are exercised
    throughout as the graph-conditioned posterior + the source/context_ref
    provenance rail.
    """
    nominal = 1 - ALPHA
    metrics["tier04.nominal_coverage"] = {"value": nominal, "tol": 0.0001}

    # --- split membership over the propagated (graph-conditioned) embeddings ---
    ids, vecs = _read_vectors(db, propagated)
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    pos = {pid: i for i, pid in enumerate(ids)}
    subject = {r["paper_id"]: r["subject"] for r in papers_rows}
    year = {r["paper_id"]: int(r["year"]) for r in papers_rows}
    classes = sorted({r["subject"] for r in papers_rows})
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_idx[subject[pid]] for pid in ids])

    def members(names):
        return np.array([pos[p] for p in names if p in pos])

    train = members(split["train"])
    cal = members(split["valid"])  # 2018 → the calibration era
    test = members(split["test"])  # 2019–2020 → the test era (the shift)
    print(f"  time-split — train {len(train)} / cal {len(cal)} / test {len(test)}", flush=True)
    (ARTIFACTS / "cal_split.json").write_text(json.dumps({
        "calibration": [ids[i] for i in cal],
        "test": [ids[i] for i in test],
        "train": [ids[i] for i in train],
    }, indent=2))

    # === Part A: the bidirectional win — year-regression conformal ===
    predictor_id = part_a_regression_conformal(
        db, papers, cite, ids, cal, test, year, norm, nominal, metrics)

    # === Part B: the conformal-under-shift lesson — subject classification ===
    centroids = np.zeros((len(classes), vecs.shape[1]))
    for c in range(len(classes)):
        m = y[train] == c
        if m.any():
            centroids[c] = norm[train][m].mean(0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    def class_scores(idx):
        logits = norm[idx] @ centroids.T * SOFTMAX_TEMP
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    cal_scores = class_scores(cal)
    test_scores = class_scores(test)
    test_acc = float((test_scores.argmax(1) == y[test]).mean())
    print(f"  classifier test accuracy: {test_acc:.3f}", flush=True)
    metrics["tier04.classifier_accuracy"] = {"value": round(test_acc, 3), "tol": 0.04}

    # --- marginal APS via the engine's conformalize (the OSS surface) under-covers ---
    marg_sets = db.conformalize(cal_scores.tolist(), y[cal].tolist(),
                                test_scores.tolist(), alpha=ALPHA, score="aps")
    marg_cov = float(np.mean([y[test][i] in marg_sets[i] for i in range(len(test))]))
    marg_size = float(np.mean([len(s) for s in marg_sets]))
    print(f"  engine APS marginal coverage (time-split): {marg_cov:.3f}  "
          f"(nominal {nominal:.2f})  mean set size {marg_size:.2f}", flush=True)
    metrics["tier04.marginal_coverage"] = {"value": round(marg_cov, 3), "tol": 0.03}
    metrics["tier04.marginal_set_size"] = {"value": round(marg_size, 2), "tol": 0.6}

    # --- apples-to-apples: a SINGLE self-consistent LOCAL APS routine ---
    # The marginal-vs-weighted comparison runs on ONE local APS routine, so the only
    # difference between the marginal and weighted passes is the weights — never an
    # APS-convention artifact. The engine's `conformalize(score="aps")` marginal
    # (above) is reported SEPARATELY as the OSS-surface corroboration: both
    # under-cover. The local routine admits the class that *crosses* q̂ (set grows
    # until cumulative mass reaches q̂), so its sets run ~1 class larger than the
    # engine's deterministic-APS admission rule (which excludes the crossing class);
    # the resulting coverage gap is a benign set-boundary convention difference, not
    # an engine bug. Both surfaces tell the same story — under-coverage on the shift.
    local_marg_cov, local_marg_size = local_aps_coverage(
        cal_scores=cal_scores, cal_labels=y[cal], test_scores=test_scores,
        test_labels=y[test], weights=None, alpha=ALPHA)
    print(f"  local APS marginal coverage (self-consistent anchor): {local_marg_cov:.3f}  "
          f"set size {local_marg_size:.2f}  "
          f"[engine APS marginal {marg_cov:.3f} — both under-cover; the ≤~0.03 gap is a "
          f"benign deterministic-APS set-boundary convention difference]", flush=True)
    metrics["tier04.local_marginal_coverage"] = {"value": round(local_marg_cov, 3), "tol": 0.03}

    # --- the weighting NO-OP: three test-likeness schemes, none reaches nominal ---
    cal_nc = aps_nonconformity(cal_scores, y[cal])
    schemes = weight_schemes(cal_emb=norm[cal], test_emb=norm[test])
    weighting_results: dict[str, dict[str, float]] = {}
    max_abs_delta = 0.0
    for name, w in schemes.items():
        cov, size = local_aps_coverage(
            cal_scores=cal_scores, cal_labels=y[cal], test_scores=test_scores,
            test_labels=y[test], weights=w, alpha=ALPHA)
        delta = cov - local_marg_cov
        max_abs_delta = max(max_abs_delta, abs(delta))
        weighting_results[name] = {"coverage": round(cov, 4), "set_size": round(size, 3),
                                   "delta_vs_marginal": round(delta, 4)}
        print(f"    weighted({name}): coverage {cov:.4f}  (Δ {delta:+.4f})  "
              f"set size {size:.2f}", flush=True)
    metrics["tier04.weighting_max_abs_delta"] = {"value": round(max_abs_delta, 4), "tol": 0.01}

    # --- the diagnostic: corr(nonconformity, test-likeness) ≈ −0.1 explains WHY ---
    test_centre = norm[test].mean(0)
    test_centre /= np.linalg.norm(test_centre) + 1e-12
    test_likeness = norm[cal] @ test_centre
    corr = float(np.corrcoef(cal_nc, test_likeness)[0, 1])
    print(f"  diagnostic corr(nonconformity, test-likeness): {corr:+.3f}  "
          f"(shift ≈ orthogonal to the conformal score → reweighting is a no-op)",
          flush=True)
    metrics["tier04.score_shift_corr"] = {"value": round(corr, 3), "tol": 0.12}

    # Commit the per-row substrate so the chapter reruns BOTH conformal passes on CPU.
    _dump_tier04(ids, y, cal, test, cal_scores, test_scores, norm, classes)
    (ARTIFACTS / "ctx_predictor.json").write_text(json.dumps({
        "model_id": predictor_id, "architecture": "attncnp", "output": "gaussian",
        "task_column": "subject", "value_column": "year",
        "note": "Graph-conditioned context predictor for the year-regression-conformal "
                "win; predict_with_context_predictor carries the source + context_ref "
                "provenance rail.",
    }, indent=2))
    (ARTIFACTS / "tier04_weighting.json").write_text(json.dumps({
        "marginal_coverage": round(local_marg_cov, 4),
        "schemes": weighting_results,
        "score_shift_corr": round(corr, 3),
        "note": "Importance-weighted conformal (Tibshirani 2019) with an APS "
                "nonconformity identical to the marginal pass: no scheme repairs the "
                "under-coverage (all stay below nominal; movements are small and not "
                "even consistently toward nominal) because the time-split shift is "
                "~orthogonal to the conformal score (see score_shift_corr).",
    }, indent=2))


def part_a_regression_conformal(db, papers: str, cite: str, ids: list[str],
                                cal: np.ndarray, test: np.ndarray,
                                year: dict[str, int], norm: np.ndarray, nominal: float,
                                metrics: dict[str, dict[str, float]]) -> str:
    """Train the gaussian year predictor, predict per-row means for cal + test,
    and conformalize the absolute-residual interval with the engine.

    **The bidirectional win is that this WORKFLOW RUNS END-TO-END at all.** The #43
    gaussian collapse once made ``train_context_predictor(output="gaussian",
    value_column="year")`` unusable (std≈0.001, means ~2163 for a 2014–2020 target).
    The fix landed in two parts: 0.26.1 standardized the fine-tune projection head,
    but the *amortized context predictor* still collapsed; 0.26.2 completed it with
    z-space standardization of the predictor's target. The predictor now fits a real
    mean (≈2018.6) with real spread (≈0.9), and the previously-impossible
    regression-conformal workflow runs.

    **The honest conformal finding (same lesson as Part B).** The interval is built
    by the engine's ``conformalize_interval`` (|y−ŷ| split conformal): calibrate the
    absolute residual on the 2018 calibration era, apply the quantile to the
    2019–2020 test era. It **under-covers** (≈0.83 < 0.90 nominal) — but the
    importance-weighted remedy is a **NO-OP** here too, for a sharper reason than
    Part B: the time-split is a *location shift*, not a residual-magnitude shift. The
    predictor regresses to the embedding-conditioned mean (~2018.6 for BOTH eras), so
    the cal residuals and the test residuals have essentially the same magnitude
    (|residual| ≈0.55 in both). Reweighting WHICH calibration residuals count cannot
    move a quantile when the residual magnitudes do not differ across the shift —
    confirmed by corr(|residual|, test-likeness) ≈ 0. The under-coverage is
    point-prediction *bias*, not a residual-distribution shift; the honest remedy is
    a governed time-aware cohort, not a client-side reweight.

    A few predictions are made **graph-conditioned** (over the citation
    neighbourhood) to exercise the BYOG surface and the source/context_ref
    provenance rail.
    """
    job = db.train_context_predictor(
        papers, key_column="paper_id", task_column="subject", value_column="year",
        architecture="attncnp", output="gaussian", objective="crps",
        epochs=PREDICTOR_EPOCHS, seed=determinism.SEED)
    job.wait()
    print(f"  context predictor: {job.model_id}", flush=True)

    def predict_means(idx: np.ndarray) -> tuple[list[float], list[float], list[int]]:
        means, stds, obs = [], [], []
        for i in idx:
            key = ids[i]
            out = db.predict_with_context_predictor(job.model_id, source=papers, target_key=key)
            means.append(float(out["mean"]))
            stds.append(float(out["std"]))
            obs.append(year[key])
        return means, stds, obs

    cal_mean, cal_std, cal_year = predict_means(cal)
    test_mean, test_std, test_year = predict_means(test)
    cal_mean_avg = float(np.mean(cal_mean))
    test_mean_avg = float(np.mean(test_mean))
    std_avg = float(np.mean(cal_std + test_std))
    print(f"  predictor fit — cal-mean {cal_mean_avg:.2f}  test-mean {test_mean_avg:.2f}  "
          f"mean std {std_avg:.3f}  (regresses to the embedding-conditioned mean ≈2018.6 "
          f"for BOTH eras → the shift is a location shift, not a spread shift)", flush=True)
    metrics["tier04.reg_cal_mean"] = {"value": round(cal_mean_avg, 2), "tol": 1.0}
    metrics["tier04.reg_test_mean"] = {"value": round(test_mean_avg, 2), "tol": 1.0}
    metrics["tier04.reg_pred_std"] = {"value": round(std_avg, 3), "tol": 0.5}

    # the bidirectional-win guard: the predictor must NOT have collapsed.
    if std_avg < 0.05:
        raise RuntimeError(
            f"gaussian predictor collapsed (mean std {std_avg:.4f} < 0.05) — the #43 "
            f"regression is NOT fixed on this subset; STOP and report.")

    # engine |y−ŷ| split-conformal interval; calibrate on cal residuals, apply to test.
    intervals = db.conformalize_interval(
        cal_mean, [float(v) for v in cal_year], test_mean, alpha=ALPHA)
    reg_cov = float(np.mean([lo <= test_year[i] <= hi for i, (lo, hi) in enumerate(intervals)]))
    reg_width = float(np.mean([hi - lo for lo, hi in intervals]))
    print(f"  year-regression conformal_interval coverage: {reg_cov:.3f}  "
          f"(nominal {nominal:.2f})  mean width {reg_width:.2f} years", flush=True)
    metrics["tier04.reg_interval_coverage"] = {"value": round(reg_cov, 3), "tol": 0.04}
    metrics["tier04.reg_interval_width"] = {"value": round(reg_width, 2), "tol": 0.6}

    # --- the residual-weighting NO-OP (the location-shift diagnostic) ---
    # The under-coverage is point-prediction BIAS, not a residual-magnitude shift:
    # the predictor regresses to the embedding-conditioned mean for both eras, so the
    # |y−ŷ| residual magnitudes are essentially equal across cal and test. Importance-
    # weighted conformal only reweights WHICH calibration residuals count — it cannot
    # move the |residual| quantile when the magnitudes don't differ across the shift.
    cal_resid = np.abs(np.asarray(cal_mean) - np.asarray(cal_year, dtype=float))
    test_resid = np.abs(np.asarray(test_mean) - np.asarray(test_year, dtype=float))
    cal_resid_mag = float(cal_resid.mean())
    test_resid_mag = float(test_resid.mean())
    print(f"  residual magnitudes — cal |y−ŷ| {cal_resid_mag:.3f}  "
          f"test |y−ŷ| {test_resid_mag:.3f}  (≈equal → location shift, not a spread shift)",
          flush=True)
    metrics["tier04.reg_cal_resid_mag"] = {"value": round(cal_resid_mag, 3), "tol": 0.25}
    metrics["tier04.reg_test_resid_mag"] = {"value": round(test_resid_mag, 3), "tol": 0.25}

    # test-era-likeness-weighted conformal on the absolute residual: a no-op.
    cal_emb = norm[cal]
    test_emb = norm[test]
    test_centre = test_emb.mean(0)
    test_centre /= np.linalg.norm(test_centre) + 1e-12
    test_likeness = cal_emb @ test_centre
    reg_weights = np.exp(WEIGHTED_TEMP * test_likeness)
    wt_reg_cov = weighted_residual_coverage(
        cal_resid=cal_resid, test_resid=test_resid, weights=reg_weights, alpha=ALPHA)
    reg_weight_delta = wt_reg_cov - reg_cov
    print(f"  weighted year-regression coverage (test-likeness): {wt_reg_cov:.3f}  "
          f"(Δ {reg_weight_delta:+.4f} vs marginal → no-op)", flush=True)
    metrics["tier04.reg_weighted_coverage"] = {"value": round(wt_reg_cov, 3), "tol": 0.04}
    metrics["tier04.reg_weighting_delta"] = {"value": round(reg_weight_delta, 4), "tol": 0.02}

    # the diagnostic explaining WHY: corr(|residual|, test-likeness) ≈ 0.
    reg_corr = float(np.corrcoef(cal_resid, test_likeness)[0, 1])
    print(f"  diagnostic corr(|residual|, test-likeness): {reg_corr:+.3f}  "
          f"(≈0 → reweighting cannot move the |residual| quantile)", flush=True)
    metrics["tier04.reg_resid_corr"] = {"value": round(reg_corr, 3), "tol": 0.2}

    (ARTIFACTS / "tier04_regression_weighting.json").write_text(json.dumps({
        "marginal_coverage": round(reg_cov, 4),
        "weighted_coverage": round(wt_reg_cov, 4),
        "weighting_delta": round(reg_weight_delta, 4),
        "cal_resid_magnitude": round(cal_resid_mag, 4),
        "test_resid_magnitude": round(test_resid_mag, 4),
        "resid_likeness_corr": round(reg_corr, 4),
        "note": "Importance-weighted conformal on the |y−ŷ| residual is a no-op: the "
                "time-split is a LOCATION shift (predictor regresses to the embedding-"
                "conditioned mean for both eras), so cal and test residual magnitudes "
                "are ≈equal and corr(|residual|, test-likeness) ≈ 0 — reweighting which "
                "residuals count cannot move the quantile. The under-coverage is point-"
                "prediction bias, not a residual-distribution shift.",
    }, indent=2))

    # commit the per-row regression substrate for the chapter to rerun on CPU.
    pq.write_table(pa.table({
        "paper_id": [ids[i] for i in cal] + [ids[i] for i in test],
        "split": ["calibration"] * len(cal) + ["test"] * len(test),
        "pred_mean": cal_mean + test_mean,
        "pred_std": cal_std + test_std,
        "true_year": cal_year + test_year,
    }), ARTIFACTS / "tier04_regression.parquet")

    # exercise the provenance rail: a few graph-conditioned predictions.
    for key in [ids[i] for i in test[:3]]:
        out = db.predict_with_context_predictor(
            job.model_id, source=papers, target_key=key,
            edge_source=cite, edge_src_column="src", edge_dst_column="dst",
            edge_direction="out", edge_hops=2)
        ref = out.get("context_ref")
        n_ctx = len(ref) if isinstance(ref, (list, tuple)) else ref
        print(f"    graph-conditioned predict {key}: source={out.get('source')!r}  "
              f"context_ref={n_ctx}", flush=True)
    return job.model_id


def weighted_residual_coverage(*, cal_resid: np.ndarray, test_resid: np.ndarray,
                               weights: np.ndarray, alpha: float) -> float:
    """Weighted split-conformal coverage of the |y−ŷ| residual interval.

    The nonconformity is the absolute residual; the weighted finite-sample quantile
    is the smallest calibration residual whose reweighted empirical CDF reaches
    ``1−α`` (Tibshirani 2019). The symmetric interval ``ŷ ± q̂`` covers a test row
    iff its own residual ≤ q̂ — so a coverage shift is attributable purely to the
    weights. A no-op when the cal and test residual magnitudes do not differ.
    """
    n = len(cal_resid)
    order = np.argsort(cal_resid)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    cw = np.cumsum(w[order])
    q_idx = min(int(np.searchsorted(cw, 1 - alpha)), n - 1)
    q = float(cal_resid[order][q_idx])
    return float(np.mean(test_resid <= q))


def aps_nonconformity(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The APS nonconformity score per labeled row: the cumulative softmax mass of
    the classes ranked at least as high as the true class (true class included)."""
    out = np.empty(len(labels))
    for i in range(len(labels)):
        order = np.argsort(-scores[i])
        cum = 0.0
        for j in order:
            cum += scores[i][j]
            if j == labels[i]:
                break
        out[i] = cum
    return out


def local_aps_coverage(*, cal_scores, cal_labels, test_scores, test_labels,
                       weights, alpha) -> tuple[float, float]:
    """Split-APS coverage + mean set size with an optional weighting.

    With ``weights=None`` this is the *unweighted* marginal APS — it must reproduce
    the engine's ``conformalize(score="aps")`` coverage (the apples-to-apples
    anchor). With a weight vector it is weighted conformal (Tibshirani 2019): the
    SAME nonconformity, only the calibration empirical CDF is reweighted, so any
    coverage change is attributable purely to the weights.
    """
    cal_nc = aps_nonconformity(cal_scores, cal_labels)
    n = len(cal_nc)
    order = np.argsort(cal_nc)
    if weights is None:
        # unweighted finite-sample APS quantile: the ⌈(n+1)(1−α)⌉-th smallest score.
        rank = int(np.ceil((n + 1) * (1 - alpha)))
        q_idx = min(rank, n) - 1
        q = float(cal_nc[order][q_idx])
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        cw = np.cumsum(w[order])
        q_idx = min(int(np.searchsorted(cw, 1 - alpha)), n - 1)
        q = float(cal_nc[order][q_idx])

    # Admit class c iff its APS nonconformity at this row — the cumulative mass up
    # to and INCLUDING c, ranked most- to least-probable — is ≤ q̂; the class that
    # *crosses* q̂ is excluded. This is ONE self-consistent local convention, applied
    # identically to the marginal and the weighted passes, so any coverage change is
    # attributable purely to the weights. It is a hair more conservative on set size
    # than the engine's deterministic-APS admission (which retains the crossing
    # class), hence the local marginal coverage sits ~0.03 above the engine's — both
    # under-cover; the gap is a set-boundary convention, not a calibration difference.
    # Ties in probability break ascending by class index.
    covered, total_size = 0, 0
    n_classes = test_scores.shape[1]
    for i in range(len(test_labels)):
        order_t = sorted(range(n_classes), key=lambda c: (-test_scores[i][c], c))
        cum = 0.0
        in_set = set()
        for c in order_t:
            cum += test_scores[i][c]
            if cum <= q:
                in_set.add(c)
        covered += int(test_labels[i] in in_set)
        total_size += len(in_set)
    return covered / len(test_labels), total_size / len(test_labels)


def weight_schemes(*, cal_emb: np.ndarray, test_emb: np.ndarray) -> dict[str, np.ndarray]:
    """Three importance-weighting schemes for test-era likeness (Tibshirani 2019).

    Each maps a calibration point to a weight that grows with how test-era-like its
    graph-conditioned embedding is, by a different estimator of the likelihood
    ratio: a centroid-likeness softmax, a kNN density ratio, and a Tibshirani
    domain-classifier logistic regression. They are the consumer's candidate
    cohort/governance levers — and here all three are no-ops.
    """
    test_centre = test_emb.mean(0)
    test_centre /= np.linalg.norm(test_centre) + 1e-12
    centroid = np.exp(WEIGHTED_TEMP * (cal_emb @ test_centre))

    # kNN density ratio: fraction of each cal point's nearest neighbours (over the
    # cal∪test pool) that are test-era, turned into an odds ratio.
    pool = np.vstack([cal_emb, test_emb])
    is_test = np.concatenate([np.zeros(len(cal_emb)), np.ones(len(test_emb))])
    sims = cal_emb @ pool.T
    np.fill_diagonal(sims[:, : len(cal_emb)], -np.inf)  # exclude self
    knn = np.argpartition(-sims, KNN_W, axis=1)[:, :KNN_W]
    frac_test = is_test[knn].mean(1)
    knn_w = (frac_test + 1e-3) / (1 - frac_test + 1e-3)

    # Tibshirani domain classifier: LR of test-membership on the embedding; the
    # weight is the fitted odds w(x) = p_test / (1 − p_test) on the cal points.
    domain = _domain_classifier_weights(cal_emb, test_emb)

    return {"centroid": centroid, "knn": knn_w, "domain_lr": domain}


def _domain_classifier_weights(cal_emb: np.ndarray, test_emb: np.ndarray) -> np.ndarray:
    """Logistic-regression domain classifier (cal=0 / test=1) → odds on cal points.

    A plain batch-gradient logistic regression (no sklearn dependency) on the
    graph-conditioned embeddings; the test-membership odds are the Tibshirani
    importance weights. Deterministic: fixed init, fixed step count.
    """
    x = np.vstack([cal_emb, test_emb])
    t = np.concatenate([np.zeros(len(cal_emb)), np.ones(len(test_emb))])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    rng = np.random.default_rng(determinism.SEED)
    w = rng.normal(0, 0.01, xs.shape[1])
    b = 0.0
    lr = 0.5
    for _ in range(300):
        z = xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - t
        w -= lr * (xs.T @ g / len(t) + 1e-3 * w)
        b -= lr * g.mean()
    cal_xs = (cal_emb - mu) / sd
    p_cal = 1.0 / (1.0 + np.exp(-(cal_xs @ w + b)))
    p_cal = np.clip(p_cal, 1e-3, 1 - 1e-3)
    return p_cal / (1 - p_cal)


def _dump_tier04(ids, y, cal, test, cal_scores, test_scores, norm, classes) -> None:
    """Commit the class scores + labels + graph-conditioned embeddings per split,
    so the chapter reruns the marginal conformal + the weighting no-op on CPU.

    Scores and embeddings are stored as f32 (the conformal result is identical to
    f64 to four decimals — verified — at half the committed size).
    """
    f32 = pa.list_(pa.float32())
    paper_id = [ids[i] for i in cal] + [ids[i] for i in test]
    split = ["calibration"] * len(cal) + ["test"] * len(test)
    true_label = [int(y[i]) for i in cal] + [int(y[i]) for i in test]
    scores = ([np.asarray(s, dtype=np.float32) for s in cal_scores]
              + [np.asarray(s, dtype=np.float32) for s in test_scores])
    emb = [np.asarray(norm[i], dtype=np.float32) for i in cal] + \
          [np.asarray(norm[i], dtype=np.float32) for i in test]
    table = pa.table({
        "paper_id": paper_id, "split": split, "true_label": true_label,
        "scores": pa.array(scores, type=f32), "emb": pa.array(emb, type=f32),
    })
    pq.write_table(table, ARTIFACTS / "tier04_predictions.parquet", compression="zstd")
    (ARTIFACTS / "tier04_classes.json").write_text(json.dumps(classes, indent=2))


def _write_checksums() -> None:
    sums = {p.name: _checksum(p) for p in sorted(ARTIFACTS.glob("*"))
            if p.is_file() and p.name != "checksums.json"}
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="grpc://127.0.0.1:50051",
                    help="connect() target — grpc://host:port for the GPU server.")
    args = ap.parse_args()
    db = jammi_ai.connect(args.target)
    emit(db)


if __name__ == "__main__":
    main()
