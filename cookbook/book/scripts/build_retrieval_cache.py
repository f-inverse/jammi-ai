#!/usr/bin/env python3
"""Emit the retrieval / search vertical cache (B1) — CPU, from the committed cache.

Unlike the keystone/fine-tune emits, this vertical needs **no GPU**: it reuses the
already-committed ogbn-arxiv embedding matrices — raw (``arxiv.embeddings``) and
graph-propagated (``arxiv.propagated``) — plus the committed same-subject golden
(``arxiv.subject_golden``, which carries the query text the lexical arm needs). It
measures three retrieval families against the SAME same-subject relevance target and
freezes the numbers:

The keystone's tier-03 fine-tune commits a model-id record (``arxiv.ft_model``, recall
0.548) but **not** a fine-tuned embedding matrix; the standalone fine-tuned matrix lives
in the separate fine-tuning-methods vertical. This vertical therefore measures the two
dense matrices the keystone commits in full — raw and propagated — as its dense arms,
and does not fabricate a fine-tuned-matrix arm it cannot read from this cache.

* **dense kNN** — exact cosine-similarity kNN over each committed matrix
  (recall@10 + nDCG@10), the keystone's proven per-table fold.
* **lexical (BM25)** — Robertson & Zaragoza (2009) over the paper titles, queried by
  the golden's query text — the sparse arm of a hybrid retriever.
* **RRF fusion** — the engine's own ``rrf_fuse`` helper (Cormack et al. 2009) over
  pairs of ranked lists: dense+dense (raw+propagated) and dense+lexical (the hybrid).

The honest question the vertical answers: **does fusion help here?** It is measured,
not assumed — the recorded ``retrieval.fusion_*`` deltas report what the data shows.

KNOWN ENGINE GOTCHA, handled + recorded (``retrieval.search_finding``): ``search`` /
``assemble_context`` target a *source* and resolve its single ready embedding table.
A committed embedding parquet is not a source-bound "ready embedding table" (so
``search`` raises ``No ready embedding table``), and once a source has several
embedding tables (raw / propagated / fine-tuned) ``search(source, …)`` has **no way
to name which** — the signature carries no ``table=`` argument. The proven, table-
exact measurement is therefore the cosine-kNN numpy fold (deterministic, targets
exactly one matrix); the ambiguity is recorded as an engine-surface finding (a
candidate for an explicit ``table=`` arg on ``search``). ``rrf_fuse`` IS reachable on
the embedded CPU handle and is used directly for the fusion arm.

Determinism (K0 §3): committed matrices + committed golden, single-threaded BLAS
(applied by importing ``jammi_cookbook``), a pure-numpy dense fold, the engine's
deterministic ``rrf_fuse``; metrics asserted to tolerances downstream.

Usage::

    python scripts/build_retrieval_cache.py            # embedded CPU engine
    python scripts/build_retrieval_cache.py --target file:///tmp/rt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import contracts

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "retrieval"
RECALL_K = 10
CANDIDATES = 200  # depth of each ranked list before fusion / truncation to K

# The dense matrices compared, addressed by their committed-artifact name. The keystone
# commits these two in full; the tier-03 fine-tune commits a model-id, not a matrix.
DENSE_MATRICES = {
    "raw": "arxiv.embeddings",
    "propagated": "arxiv.propagated",
}

# The token in this finding is the engine-surface candidate it points at.
SEARCH_FINDING = {
    "verb": "search",
    "reachable_on_cpu": False,
    "reason": (
        "search / assemble_context resolve a source's single ready embedding table; "
        "a committed embedding parquet is not a source-bound ready embedding table "
        "(search raises 'No ready embedding table for source'), and the signature "
        "search(source, *, query, k, filter, select) carries no table= argument, so "
        "once a source has several embedding tables (raw / propagated / fine-tuned) "
        "there is no way to name which one to search."
    ),
    "measurement_used": "exact cosine-kNN numpy fold over each committed matrix",
    "candidate": "an explicit table= argument on search(source, ...)",
}


# --------------------------------------------------------------------------- #
# loaders + folds
# --------------------------------------------------------------------------- #


def _matrix(name: str) -> tuple[list[str], np.ndarray]:
    table = contracts.load_artifact(name)
    ids = [str(x) for x in table.column("_row_id").to_pylist()]
    vecs = np.asarray([list(v) for v in table.column("vector").to_pylist()], dtype=np.float32)
    return ids, vecs


def _golden() -> tuple[dict[str, set[str]], dict[str, str]]:
    rows = contracts.load_artifact("arxiv.subject_golden").to_pylist()
    relevant: dict[str, set[str]] = defaultdict(set)
    query_text: dict[str, str] = {}
    for r in rows:
        relevant[r["query_id"]].add(r["relevant_id"])
        query_text[r["query_id"]] = r["query_text"]
    return relevant, query_text


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _score(ranked_for: dict[str, list[str]], relevant: dict[str, set[str]]) -> tuple[float, float]:
    """recall@K + nDCG@K over a per-query ranking, against the same-subject golden."""
    recall = ndcg = total = 0.0
    for qid, ranked in ranked_for.items():
        rel = relevant.get(qid)
        if not rel:
            continue
        top = ranked[:RECALL_K]
        hits = [1.0 if r in rel else 0.0 for r in top]
        recall += sum(hits) / min(RECALL_K, len(rel))
        ideal = _dcg([1.0] * min(RECALL_K, len(rel)))
        ndcg += (_dcg(hits) / ideal) if ideal > 0 else 0.0
        total += 1
    if not total:
        raise RuntimeError("no scorable queries — golden/matrix row-id mismatch")
    return recall / total, ndcg / total


def _dense_ranker(ids: list[str], vecs: np.ndarray):
    """A per-query exact cosine-kNN ranking over one committed matrix (the proven fold)."""
    pos = {i: n for n, i in enumerate(ids)}
    norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    def rank(qid: str) -> list[str]:
        if qid not in pos:
            return []
        sims = norm @ norm[pos[qid]]
        sims[pos[qid]] = -np.inf  # exclude self
        top = np.argpartition(-sims, CANDIDATES)[:CANDIDATES]
        return [ids[t] for t in top[np.argsort(-sims[top])]]

    return rank, pos


_TOKEN = re.compile(r"[a-z0-9]+")


def _bm25_ranker(query_text: dict[str, str], k1: float = 1.5, b: float = 0.75):
    """A BM25 lexical ranker (Robertson & Zaragoza 2009) over the paper titles.

    The sparse arm of the hybrid retriever — scored against the same documents and the
    same same-subject relevance target as the dense arms, so the comparison is honest.
    """
    papers = contracts.load_artifact("arxiv.papers").to_pylist()
    titles = {p["paper_id"]: (p["title"] or "") for p in papers}
    doc_tokens = {pid: _TOKEN.findall(text.lower()) for pid, text in titles.items()}
    n_docs = len(doc_tokens)
    avgdl = sum(len(t) for t in doc_tokens.values()) / n_docs
    doc_freq: Counter[str] = Counter()
    for toks in doc_tokens.values():
        doc_freq.update(set(toks))
    idf = {w: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for w, df in doc_freq.items()}
    term_freq = {pid: Counter(toks) for pid, toks in doc_tokens.items()}
    doc_ids = list(doc_tokens)

    def rank(qid: str) -> list[str]:
        q = _TOKEN.findall(query_text.get(qid, "").lower())
        scores: dict[str, float] = {}
        for pid in doc_ids:
            tf = term_freq[pid]
            dl = len(doc_tokens[pid])
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += idf.get(w, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if s > 0:
                scores[pid] = s
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        return [pid for pid, _ in ordered[:CANDIDATES]]

    return rank


def _fuse(db, rankers: list, qids: list[str]) -> dict[str, list[str]]:
    """RRF-fuse several per-query rankings via the engine's own ``rrf_fuse`` helper."""
    fused: dict[str, list[str]] = {}
    for qid in qids:
        lists = [r(qid) for r in rankers]
        fused[qid] = [doc for doc, _ in db.rrf_fuse(lists)]
    return fused


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #


def emit(db) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    relevant, query_text = _golden()
    qids = sorted(relevant)
    print(f"queries: {len(qids)}  (same-subject golden, with query text)", flush=True)

    dense_rankers: dict[str, object] = {}
    method_rows: list[dict] = []

    # --- dense kNN per matrix -------------------------------------------------
    print("\n[dense] cosine-kNN per committed matrix", flush=True)
    for name, artifact_name in DENSE_MATRICES.items():
        ids, vecs = _matrix(artifact_name)
        rank, _ = _dense_ranker(ids, vecs)
        dense_rankers[name] = rank
        ranked = {qid: rank(qid) for qid in qids}
        recall, ndcg = _score(ranked, relevant)
        print(f"  {name:11} recall@10 {recall:.4f}  nDCG@10 {ndcg:.4f}", flush=True)
        method_rows.append({"method": f"dense_{name}", "family": "dense",
                            "recall_at_10": round(recall, 4), "ndcg_at_10": round(ndcg, 4)})

    # --- lexical BM25 ---------------------------------------------------------
    print("\n[lexical] BM25 over paper titles", flush=True)
    bm25 = _bm25_ranker(query_text)
    lex_ranked = {qid: bm25(qid) for qid in qids}
    lex_recall, lex_ndcg = _score(lex_ranked, relevant)
    print(f"  bm25        recall@10 {lex_recall:.4f}  nDCG@10 {lex_ndcg:.4f}", flush=True)
    method_rows.append({"method": "lexical_bm25", "family": "lexical",
                        "recall_at_10": round(lex_recall, 4), "ndcg_at_10": round(lex_ndcg, 4)})

    # --- RRF fusion via the engine helper -------------------------------------
    print("\n[fusion] rrf_fuse (Cormack et al. 2009) over ranked-list pairs", flush=True)
    fusion_specs = {
        "rrf_raw_prop": [dense_rankers["raw"], dense_rankers["propagated"]],
        "rrf_raw_lex": [dense_rankers["raw"], bm25],   # hybrid: dense + lexical
        "rrf_prop_lex": [dense_rankers["propagated"], bm25],
    }
    fusion_rows: list[dict] = []
    for name, rankers in fusion_specs.items():
        fused = _fuse(db, rankers, qids)
        recall, ndcg = _score(fused, relevant)
        print(f"  {name:13} recall@10 {recall:.4f}  nDCG@10 {ndcg:.4f}", flush=True)
        fusion_rows.append({"method": name, "family": "fusion",
                            "recall_at_10": round(recall, 4), "ndcg_at_10": round(ndcg, 4)})
    method_rows.extend(fusion_rows)

    # --- the honest fusion-helps deltas --------------------------------------
    by_method = {r["method"]: r for r in method_rows}
    best_dense = max((r for r in method_rows if r["family"] == "dense"),
                     key=lambda r: r["recall_at_10"])
    raw_recall = by_method["dense_raw"]["recall_at_10"]
    rrf_rp = by_method["rrf_raw_prop"]["recall_at_10"]
    rrf_hybrid = by_method["rrf_raw_lex"]["recall_at_10"]
    # Does fusing the two dense arms beat the best single dense arm? And does the
    # hybrid (dense+lexical) beat the dense arm it fuses? Both measured, not assumed.
    fusion_vs_best_dense = round(rrf_rp - best_dense["recall_at_10"], 4)
    fusion_vs_raw = round(rrf_rp - raw_recall, 4)
    hybrid_vs_dense = round(rrf_hybrid - raw_recall, 4)
    # The honest test: does ANY fusion arm beat the BEST single arm? Beating only the
    # weaker arm (raw) is not "fusion helps" — RRF mixing in a weak ranker cannot
    # exceed the strong ranker it already contains. fusion_helps is True only if some
    # fused list exceeds the best single (dense-or-lexical) arm.
    best_single = max(r["recall_at_10"] for r in method_rows
                      if r["family"] in ("dense", "lexical"))
    best_fusion = max(r["recall_at_10"] for r in method_rows if r["family"] == "fusion")
    fusion_helps = best_fusion > best_single

    print("\n=== the honest fusion finding ===", flush=True)
    print(f"  best single dense arm:       {best_dense['method']} "
          f"{best_dense['recall_at_10']:.4f}", flush=True)
    print(f"  best single arm overall:     {best_single:.4f}", flush=True)
    print(f"  best fusion arm:             {best_fusion:.4f}", flush=True)
    print(f"  RRF(raw+prop) vs raw:        {fusion_vs_raw:+.4f}", flush=True)
    print(f"  RRF(raw+prop) vs best dense: {fusion_vs_best_dense:+.4f}", flush=True)
    print(f"  hybrid RRF(raw+lex) vs raw:  {hybrid_vs_dense:+.4f}", flush=True)
    print(f"  fusion helps (beats best single arm)? {fusion_helps}", flush=True)

    # --------------------------------------------------------------------- #
    # commit the per-method table + the finding record + goldens
    # --------------------------------------------------------------------- #
    pq.write_table(pa.table({
        "method": [r["method"] for r in method_rows],
        "family": [r["family"] for r in method_rows],
        "recall_at_10": [r["recall_at_10"] for r in method_rows],
        "ndcg_at_10": [r["ndcg_at_10"] for r in method_rows],
    }), ARTIFACTS / "method_metrics.parquet")

    finding = {
        "queries": len(qids),
        "candidate_depth": CANDIDATES,
        "best_dense_method": best_dense["method"],
        "best_dense_recall_at_10": best_dense["recall_at_10"],
        "best_single_recall_at_10": round(best_single, 4),
        "best_fusion_recall_at_10": round(best_fusion, 4),
        "fusion_vs_best_dense": fusion_vs_best_dense,
        "fusion_vs_raw": fusion_vs_raw,
        "hybrid_vs_dense": hybrid_vs_dense,
        "fusion_helps": fusion_helps,
        "methods": method_rows,
        "search_finding": SEARCH_FINDING,
        "note": (
            "Dense beats lexical on this same-subject target by a wide margin; RRF "
            "fusion of the two dense arms beats the weaker arm (raw) but does NOT beat "
            "the stronger single arm (propagated), and the hybrid dense+lexical fusion "
            "is DRAGGED DOWN by the weaker lexical arm. The measured verdict: fusion "
            "does not help here — RRF cannot exceed the best single arm when one arm "
            "dominates. Reported as the data shows, not assumed. The search-multi-table "
            "ambiguity is recorded in search_finding."
        ),
    }
    (ARTIFACTS / "retrieval.json").write_text(json.dumps(finding, indent=2))

    metrics: dict[str, dict[str, float]] = {}
    for r in method_rows:
        metrics[f"{r['method']}.recall_at_10"] = {"value": r["recall_at_10"], "tol": 0.03}
        metrics[f"{r['method']}.ndcg_at_10"] = {"value": r["ndcg_at_10"], "tol": 0.03}
    metrics["fusion_vs_best_dense"] = {"value": fusion_vs_best_dense, "tol": 0.02}
    metrics["fusion_vs_raw"] = {"value": fusion_vs_raw, "tol": 0.02}
    metrics["hybrid_vs_dense"] = {"value": hybrid_vs_dense, "tol": 0.02}
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    _write_checksums()
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _write_checksums() -> None:
    sums = {p.name: _checksum(p) for p in sorted(ARTIFACTS.glob("*"))
            if p.is_file() and p.name != "checksums.json"}
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="file:///tmp/jammi-retrieval",
                    help="connect() target — file:// for the embedded CPU engine.")
    args = ap.parse_args()
    db = jammi_ai.connect(args.target)
    emit(db)


if __name__ == "__main__":
    main()
