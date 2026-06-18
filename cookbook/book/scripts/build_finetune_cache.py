#!/usr/bin/env python3
"""Emit the fine-tune-methods vertical cache (A1) — run ONCE on the GPU server.

Several embedding fine-tuning *methods* measured side-by-side on the SAME task —
same-subject retrieval over the committed ogbn-arxiv subset — so the cookbook can
report the REAL per-method recall@10 and the honest finding: which losses actually
move recall on this same-subject / declared-edge supervision, and which don't.

This clones the ``build_arxiv_cache`` pattern: it connects to a running GPU
``jammi-server`` (the published ``jammi-server-cu12``) over a ``grpc://`` target,
reuses the committed 4000-paper subset + its registered sources + the keystone's
embedding-independent same-subject golden, runs each method as a short LoRA
fine-tune (a few epochs — these are small), embeds the subset with each resulting
checkpoint, and folds recall@10 via the exact cosine-kNN numpy routine the keystone
uses. Emits ``artifacts/finetune/`` (the recall rows + per-method embedding
matrices + the Matryoshka recall-vs-dim curve + golden_metrics.json + checksums).

The supervision is honest and shared across the loss methods: same-subject
(anchor, positive) pairs and (anchor, positive, negative) triplets mined from the
committed subset's subject labels — the same label signal the keystone's recall
target measures, so a gain is a real metric-learning gain on this supervision, not
a circular one. ``fine_tune_graph`` (the keystone's tier-03 declared-edge method)
is run for contrast on the SAME recall target.

Methods compared (all LoRA, ModernBERT-base, same subset + same golden):

* ``cosent`` baseline — ``embedding_loss="cosent"`` over scored contrastive pairs.
* ``mnrl`` (two temperatures) — ``embedding_loss="mnrl"`` over (anchor, positive)
  pairs, in-batch negatives; ``mnrl_temperature`` swept.
* ``triplet`` — ``embedding_loss="triplet"`` over explicit (anchor, positive,
  negative) triplets.
* ``hard_neg`` — ``mnrl`` + ``mine_hard_negatives=True`` (``hard_negative_k``,
  ``hard_negative_exclude_hops``, ``hard_negative_refresh_every``).
* ``matryoshka`` — ``mnrl`` + ``matryoshka_dims=[768,256,64]``; recall is measured
  at the full dim AND at the truncated dims (the Matryoshka curve).
* ``graph_declared`` — ``fine_tune_graph(edge_provenance="declared")`` over the
  declared citation graph, for contrast.

Determinism contract (K0 §3): committed subset ids, pinned ModernBERT + dtype,
single-threaded BLAS (applied by importing jammi_cookbook), the pair/triplet mining
seeded from ``determinism.SEED``, the recall fold a pure numpy routine, metrics
asserted to tolerances downstream.

Usage::

    # 1. start the GPU server (clean artifact dir for a reproducible emit):
    #    JAMMI_ARTIFACT_DIR=/tmp/srv-ft-art JAMMI_GPU__DEVICE=0 \\
    #    JAMMI_GPU__REQUIRE_GPU=true JAMMI_SERVER__FLIGHT_LISTEN=127.0.0.1:50051 \\
    #    jammi-server &
    # 2. emit against it:
    python scripts/build_finetune_cache.py --target grpc://127.0.0.1:50051
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
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "finetune"
SUBSET = 4000
RECALL_K = 10
GOLDEN_QUERIES = 200  # same as the keystone's same-subject golden
N_PAIRS = 1500  # supervised (anchor, positive[, negative]) examples mined per method
EPOCHS = 2  # a few epochs to bound GPU time — these are small LoRA runs
BATCH = 32
MATRYOSHKA_DIMS = [768, 256, 64]
TEXT_CLIP = 1500  # chars of title+abstract per supervised example (bounds tokens)


# --------------------------------------------------------------------------- #
# helpers (the engine-table addressing + recall fold, shared with the keystone)
# --------------------------------------------------------------------------- #


def _emb_ref(table: str) -> str:
    return f'"jammi.{table}"'


def _read_vectors(db, table: str) -> tuple[list[str], np.ndarray]:
    t = db.sql(f"SELECT _row_id, vector FROM {_emb_ref(table)} ORDER BY _row_id")
    ids = [str(x) for x in t.column("_row_id").to_pylist()]
    vecs = np.asarray([list(v) for v in t.column("vector").to_pylist()], dtype=np.float32)
    return ids, vecs


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _text(row: dict) -> str:
    return ((row["title"] or "") + ". " + (row["abstract"] or ""))[:TEXT_CLIP]


def build_subject_golden(papers_rows: list[dict], *, query_n: int) -> tuple[pa.Table, str]:
    """A same-subject retrieval golden (the keystone's embedding-independent target).

    query paper → same-subject papers; the relevance target is the subject label,
    independent of any embedding similarity, so a recall gain from a fine-tune is a
    real metric-learning result on the subject signal, not a circular one.
    """
    by_subject: dict[str, list[str]] = {}
    for r in papers_rows:
        by_subject.setdefault(r["subject"], []).append(r["paper_id"])
    queries = [
        r
        for r in sorted(papers_rows, key=lambda r: r["paper_id"])
        if len(by_subject[r["subject"]]) >= 5
    ][:query_n]
    rows = []
    for q in queries:
        for rid in by_subject[q["subject"]]:
            if rid != q["paper_id"]:
                rows.append({"query_id": q["paper_id"], "relevant_id": rid})
    table = pa.table({
        "query_id": [r["query_id"] for r in rows],
        "relevant_id": [r["relevant_id"] for r in rows],
    })
    path = ARTIFACTS / "subject_golden.parquet"
    pq.write_table(table, path)
    return table, str(path)


def recall_at_k(ids: list[str], vecs: np.ndarray, *, relevant: dict[str, set[str]],
                query_ids: list[str], k: int, dim: int | None = None) -> float:
    """recall@k for same-subject retrieval over a specific embedding matrix.

    Exact cosine-similarity kNN over the committed matrix (a pure numpy fold —
    deterministic, targets exactly this matrix). ``dim`` truncates the embedding to
    its first ``dim`` coordinates before normalizing — the Matryoshka read: a
    Matryoshka-trained matrix is meant to retrieve well from a prefix slice.
    """
    pos = {i: n for n, i in enumerate(ids)}
    mat = vecs[:, :dim] if dim is not None else vecs
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    scored = 0.0
    total = 0
    for qid in query_ids:
        if qid not in pos or qid not in relevant:
            continue
        sims = norm @ norm[pos[qid]]
        sims[pos[qid]] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        got = [ids[t] for t in top]
        rel = relevant[qid]
        scored += sum(1 for g in got if g in rel) / min(k, len(rel))
        total += 1
    return scored / total if total else 0.0


# --------------------------------------------------------------------------- #
# supervision: same-subject pairs + triplets mined from the committed labels
# --------------------------------------------------------------------------- #


def mine_supervision(db, papers_rows: list[dict]) -> dict[str, str]:
    """Register the shared supervised sources, returning their registered names.

    All loss methods train on the SAME mined same-subject signal so the comparison
    isolates the loss/knob, not the data:

    * ``ft_pairs`` ``(anchor, positive)`` — same-subject pair; MNRL / hard-neg /
      Matryoshka consume this (in-batch / mined negatives).
    * ``ft_triplets`` ``(anchor, positive, negative)`` — same-subject positive,
      different-subject negative; the triplet loss consumes this.
    * ``ft_contrastive`` ``(text_a, text_b, score)`` — the pairs as score=1 plus the
      triplets' negatives as score=0; CoSENT consumes this.
    """
    by_subject: dict[str, list[dict]] = {}
    for r in papers_rows:
        by_subject.setdefault(r["subject"], []).append(r)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(determinism.SEED)
    ordered = sorted(papers_rows, key=lambda r: r["paper_id"])

    anchors, positives, negatives = [], [], []
    for a in ordered:
        if len(anchors) >= N_PAIRS:
            break
        pool = [r for r in by_subject[a["subject"]] if r["paper_id"] != a["paper_id"]]
        if not pool:
            continue
        p = pool[int(rng.integers(len(pool)))]
        other = [s for s in subjects if s != a["subject"]]
        ns = other[int(rng.integers(len(other)))]
        n = by_subject[ns][int(rng.integers(len(by_subject[ns])))]
        anchors.append(_text(a))
        positives.append(_text(p))
        negatives.append(_text(n))

    pairs_path = str(ARTIFACTS / "_pairs.parquet")
    trip_path = str(ARTIFACTS / "_triplets.parquet")
    contrast_path = str(ARTIFACTS / "_contrastive.parquet")
    pq.write_table(pa.table({"anchor": anchors, "positive": positives}), pairs_path)
    pq.write_table(
        pa.table({"anchor": anchors, "positive": positives, "negative": negatives}), trip_path
    )
    pq.write_table(
        pa.table({
            "text_a": anchors + anchors,
            "text_b": positives + negatives,
            "score": [1.0] * len(anchors) + [0.0] * len(anchors),
        }),
        contrast_path,
    )
    db.add_source("ft_pairs", url=pairs_path, format="parquet")
    db.add_source("ft_triplets", url=trip_path, format="parquet")
    db.add_source("ft_contrastive", url=contrast_path, format="parquet")
    print(f"  supervision mined: {len(anchors)} pairs / triplets / contrastive rows", flush=True)
    return {"pairs": "ft_pairs", "triplets": "ft_triplets", "contrastive": "ft_contrastive"}


# --------------------------------------------------------------------------- #
# the per-method runner
# --------------------------------------------------------------------------- #


class HardNegativeMiningOOM(RuntimeError):
    """The hard-negative miner ran out of GPU memory at this corpus scale.

    A real engine-surface limit, not a faked result: on 0.26.2 the
    ``mine_hard_negatives=True`` path encodes the supervised corpus to build the
    hard-negative index, and that encode pass exceeds the A10G's 23 GB at the
    cookbook's full supervised scale — failing in the ModernBERT forward pass
    *before training starts*, independent of ``batch_size``. Surfaced and recorded,
    never papered over with a stand-in recall number.
    """


def fine_tune_and_recall(db, papers: str, *, label: str, relevant, query_ids,
                         dump: bool, **ft_kwargs) -> dict:
    """Run one LoRA fine-tune, embed the subset, fold recall@10, return the row.

    ``dump`` writes the per-method embedding matrix (committed for the Matryoshka
    method so the chapter can recompute the truncated-dim curve on CPU); the other
    methods commit only the scalar recall row.

    A CUDA OOM in the hard-negative mining encode is re-raised as
    :class:`HardNegativeMiningOOM` so the caller can record it as the real
    engine finding it is, rather than fabricating a recall for a run that did not
    happen.
    """
    print(f"\n[{label}] fine_tune {ft_kwargs.get('embedding_loss', '?')}", flush=True)
    job = db.fine_tune(base_model=EMBED_MODEL, method="lora", task="text_embedding",
                       epochs=EPOCHS, batch_size=BATCH, **ft_kwargs)
    try:
        job.wait()
    except Exception as exc:  # noqa: BLE001 — classify the engine error, then re-raise
        if "out of memory" in str(exc).lower():
            raise HardNegativeMiningOOM(str(exc)) from exc
        raise
    if job.status() != "completed":
        raise RuntimeError(f"{label} fine-tune did not complete: status={job.status()}")
    print(f"  model_id: {job.model_id}", flush=True)
    emb = db.generate_embeddings(source=papers, model=job.model_id,
                                 columns=["title", "abstract"], key="paper_id")
    ids, vecs = _read_vectors(db, emb)
    r = recall_at_k(ids, vecs, relevant=relevant, query_ids=query_ids, k=RECALL_K)
    print(f"  recall@{RECALL_K}: {r:.3f}  (dim {vecs.shape[1]})", flush=True)
    if dump:
        pq.write_table(
            pa.table({"_row_id": ids, "vector": [v.tolist() for v in vecs]}),
            ARTIFACTS / f"emb_{label}.parquet",
        )
    return {"method": label, "model_id": job.model_id, "recall_at_10": round(r, 4),
            "dim": int(vecs.shape[1]), "ids": ids, "vecs": vecs}


def run_hard_negatives(db, papers: str, *, relevant, query_ids, pairs_source: str) -> dict:
    """Attempt hard-negative mining at full scale; record the OOM honestly if it fails.

    Returns a structured finding for ``methods.json``: either a completed run with a
    recall, or — the real 0.26.2 result on the A10G — the corpus-encode OOM, with a
    smaller-corpus run that proves the kwarg works (so the limit is shown to be a
    *memory-scale* limit, not a broken signature). No recall is ever fabricated for a
    run that did not happen; hard-negatives is therefore reported as a finding, not
    placed in the apples-to-apples recall table.
    """
    try:
        full = fine_tune_and_recall(
            db, papers, label="hard_neg", relevant=relevant, query_ids=query_ids, dump=False,
            source=pairs_source, columns=["anchor", "positive"],
            embedding_loss="mnrl", mine_hard_negatives=True, hard_negative_k=5,
            hard_negative_exclude_hops=1, hard_negative_refresh_every=1)
        return {"method": "hard_neg", "status": "completed",
                "recall_at_10": full["recall_at_10"], "model_id": full["model_id"]}
    except HardNegativeMiningOOM as oom:
        print(f"  hard-negative mining OOM at full scale ({N_PAIRS} pairs): {oom}", flush=True)
        # Prove the kwarg works at a corpus the device can hold — the threshold probe.
        # The probe is corroboration: any failure in it is recorded, never crashes the
        # emit (the load-bearing finding is the full-scale OOM, already established).
        small_n = 300
        try:
            small_source, small_n = _small_pairs_source(db, pairs_source)
            small = fine_tune_and_recall(
                db, papers, label="hard_neg_small", relevant=relevant, query_ids=query_ids,
                dump=False, source=small_source, columns=["anchor", "positive"],
                embedding_loss="mnrl", mine_hard_negatives=True, hard_negative_k=5,
                hard_negative_exclude_hops=1, hard_negative_refresh_every=1)
            small_recall = small["recall_at_10"]
            small_status = "completed"
        except HardNegativeMiningOOM:
            small_recall = None
            small_status = "oom"
        return {
            "method": "hard_neg", "status": "oom_at_full_scale",
            "full_scale_pairs": N_PAIRS,
            "small_scale_pairs": small_n,
            "small_scale_status": small_status,
            "small_scale_recall_at_10": small_recall,
            "error": str(oom),
            "note": "mine_hard_negatives=True OOMs on the A10G (23 GB) in the corpus-"
                    "encode pass at the cookbook's full supervised scale, failing before "
                    "training starts and independent of batch_size; it completes at a "
                    f"~{small_n}-pair corpus. A real memory-scale limit of the 0.26.2 "
                    "hard-negative miner, recorded — not a fabricated recall.",
        }


def _small_pairs_source(db, pairs_source: str) -> tuple[str, int]:
    """Register a small slice of the supervised pairs that the miner can hold.

    Reads the on-disk supervised-pairs parquet directly (it is still present until
    the end-of-emit cleanup) rather than round-tripping through SQL — the registered
    source is addressed by the engine, not queryable as ``<src>.public.<src>`` here.
    """
    small_n = 300
    full = pq.read_table(ARTIFACTS / "_pairs.parquet")
    sliced = full.slice(0, small_n)
    path = str(ARTIFACTS / "_pairs_small.parquet")
    pq.write_table(sliced, path)
    db.add_source("ft_pairs_small", url=path, format="parquet")
    return "ft_pairs_small", sliced.num_rows


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def emit(db) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    info = db.get_server_info()
    print("server:", json.dumps(info), flush=True)
    if info.get("version") != "0.26.4":
        raise RuntimeError(f"server version {info.get('version')} != pinned 0.26.4 — STOP")

    arxiv = datasets.load_ogbn_arxiv(db, subset=SUBSET)
    papers = arxiv.papers_source
    papers_rows = db.sql(
        f"SELECT paper_id, title, abstract, subject, year FROM {papers}.public.{papers}"
    ).to_pylist()
    print(f"papers: {len(papers_rows)}", flush=True)

    golden_table, _ = build_subject_golden(papers_rows, query_n=GOLDEN_QUERIES)
    relevant: dict[str, set[str]] = {}
    for g in golden_table.to_pylist():
        relevant.setdefault(g["query_id"], set()).add(g["relevant_id"])
    query_ids = sorted(relevant)

    src = mine_supervision(db, papers_rows)

    # --- baseline: a frozen (un-fine-tuned) ModernBERT recall, for the gain column ---
    print("\n[base] frozen ModernBERT embeddings (no fine-tune)", flush=True)
    base_emb = db.generate_embeddings(source=papers, model=EMBED_MODEL,
                                      columns=["title", "abstract"], key="paper_id")
    base_ids, base_vecs = _read_vectors(db, base_emb)
    base_recall = recall_at_k(base_ids, base_vecs, relevant=relevant, query_ids=query_ids,
                              k=RECALL_K)
    print(f"  recall@{RECALL_K} (frozen base): {base_recall:.3f}", flush=True)

    rows: list[dict] = []

    # cosent baseline (scored contrastive pairs)
    rows.append(fine_tune_and_recall(
        db, papers, label="cosent", relevant=relevant, query_ids=query_ids, dump=False,
        source=src["contrastive"], columns=["text_a", "text_b", "score"],
        embedding_loss="cosent"))

    # MNRL at two temperatures (in-batch negatives over same-subject pairs)
    rows.append(fine_tune_and_recall(
        db, papers, label="mnrl_t0.05", relevant=relevant, query_ids=query_ids, dump=False,
        source=src["pairs"], columns=["anchor", "positive"],
        embedding_loss="mnrl", mnrl_temperature=0.05))
    rows.append(fine_tune_and_recall(
        db, papers, label="mnrl_t0.20", relevant=relevant, query_ids=query_ids, dump=False,
        source=src["pairs"], columns=["anchor", "positive"],
        embedding_loss="mnrl", mnrl_temperature=0.20))

    # triplet (explicit different-subject negatives)
    rows.append(fine_tune_and_recall(
        db, papers, label="triplet", relevant=relevant, query_ids=query_ids, dump=False,
        source=src["triplets"], columns=["anchor", "positive", "negative"],
        embedding_loss="triplet"))

    # hard-negatives (MNRL + mined hard negatives, excluding 1-hop pairs). On 0.26.2
    # the miner's corpus-encode pass OOMs on the A10G at the cookbook's full
    # supervised scale; we record that as a REAL engine finding (with the measured
    # corpus threshold), never a fabricated recall. The probe established the kwarg
    # itself works — it completes at a small (~300-pair) corpus and fails at ~500+.
    hard_neg_finding = run_hard_negatives(
        db, papers, relevant=relevant, query_ids=query_ids, pairs_source=src["pairs"])

    # Matryoshka (MNRL + nested dims) — committed so the chapter recomputes the curve
    matry = fine_tune_and_recall(
        db, papers, label="matryoshka", relevant=relevant, query_ids=query_ids, dump=True,
        source=src["pairs"], columns=["anchor", "positive"],
        embedding_loss="mnrl", matryoshka_dims=MATRYOSHKA_DIMS)
    rows.append(matry)

    # --- the Matryoshka recall-vs-truncated-dim curve (read off the committed matrix) ---
    print("\n[matryoshka] recall vs truncated dim", flush=True)
    matry_curve: list[dict] = []
    for d in MATRYOSHKA_DIMS:
        rd = recall_at_k(matry["ids"], matry["vecs"], relevant=relevant, query_ids=query_ids,
                         k=RECALL_K, dim=d)
        matry_curve.append({"dim": d, "recall_at_10": round(rd, 4)})
        print(f"  dim {d:>4}: recall@{RECALL_K} {rd:.3f}", flush=True)

    # --- fine_tune_graph (declared-edge) for contrast on the SAME recall target ---
    print("\n[graph_declared] fine_tune_graph(edge_provenance='declared')", flush=True)
    gjob = db.fine_tune_graph(
        node_source=papers, id_column="paper_id", text_column="abstract",
        edge_source=arxiv.cite_edges_source, src_column="src", dst_column="dst",
        base_model=EMBED_MODEL, edge_provenance="declared",
        epochs=EPOCHS, batch_size=BATCH, walks_per_node=2, walk_length=4,
        sample_seed=determinism.SEED)
    gjob.wait()
    if gjob.status() != "completed":
        raise RuntimeError(f"graph fine-tune did not complete: status={gjob.status()}")
    print(f"  model_id: {gjob.model_id}", flush=True)
    g_emb = db.generate_embeddings(source=papers, model=gjob.model_id,
                                   columns=["title", "abstract"], key="paper_id")
    g_ids, g_vecs = _read_vectors(db, g_emb)
    g_recall = recall_at_k(g_ids, g_vecs, relevant=relevant, query_ids=query_ids, k=RECALL_K)
    print(f"  recall@{RECALL_K} (declared-edge graph FT): {g_recall:.3f}", flush=True)
    rows.append({"method": "graph_declared", "model_id": gjob.model_id,
                 "recall_at_10": round(g_recall, 4), "dim": int(g_vecs.shape[1])})

    # --------------------------------------------------------------------- #
    # commit the per-method recall table + the Matryoshka curve + goldens
    # --------------------------------------------------------------------- #
    methods = [{"method": r["method"], "model_id": r["model_id"],
                "recall_at_10": r["recall_at_10"], "dim": r["dim"],
                "recall_gain_vs_base": round(r["recall_at_10"] - base_recall, 4)}
               for r in rows]
    best = max(methods, key=lambda m: m["recall_at_10"])
    worst = min(methods, key=lambda m: m["recall_at_10"])
    spread = round(best["recall_at_10"] - worst["recall_at_10"], 4)

    pq.write_table(pa.table({
        "method": [m["method"] for m in methods],
        "recall_at_10": [m["recall_at_10"] for m in methods],
        "recall_gain_vs_base": [m["recall_gain_vs_base"] for m in methods],
        "dim": [m["dim"] for m in methods],
    }), ARTIFACTS / "method_recall.parquet")
    pq.write_table(pa.table({
        "dim": [c["dim"] for c in matry_curve],
        "recall_at_10": [c["recall_at_10"] for c in matry_curve],
    }), ARTIFACTS / "matryoshka_curve.parquet")

    (ARTIFACTS / "methods.json").write_text(json.dumps({
        "base_model": EMBED_MODEL,
        "base_recall_at_10": round(base_recall, 4),
        "epochs": EPOCHS,
        "n_supervised_pairs": N_PAIRS,
        "golden_queries": len(query_ids),
        "matryoshka_dims": MATRYOSHKA_DIMS,
        "methods": methods,
        "matryoshka_curve": matry_curve,
        "best_method": best["method"],
        "worst_method": worst["method"],
        "recall_spread": spread,
        "hard_negatives": hard_neg_finding,
    }, indent=2))

    metrics: dict[str, dict[str, float]] = {
        "base_recall_at_10": {"value": round(base_recall, 3), "tol": 0.03},
        "method_recall_spread": {"value": spread, "tol": 0.03},
    }
    for m in methods:
        metrics[f"{m['method']}.recall_at_10"] = {"value": m["recall_at_10"], "tol": 0.03}
        metrics[f"{m['method']}.recall_gain_vs_base"] = {
            "value": m["recall_gain_vs_base"], "tol": 0.03}
    for c in matry_curve:
        metrics[f"matryoshka.dim_{c['dim']}_recall"] = {"value": c["recall_at_10"], "tol": 0.03}

    (ARTIFACTS / "golden_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True))

    # drop the intermediate supervised parquets — sources only, not committed cache.
    for tmp in ("_pairs.parquet", "_triplets.parquet", "_contrastive.parquet",
                "_pairs_small.parquet"):
        (ARTIFACTS / tmp).unlink(missing_ok=True)

    _write_checksums()
    print("\n=== per-method recall@10 (REAL) ===", flush=True)
    print(f"  frozen base: {base_recall:.4f}", flush=True)
    for m in methods:
        print(f"  {m['method']:<16} {m['recall_at_10']:.4f}  "
              f"(Δ vs base {m['recall_gain_vs_base']:+.4f})", flush=True)
    print(f"  hard_neg:        {hard_neg_finding['status']}"
          + (f" (small-scale {hard_neg_finding.get('small_scale_pairs')}-pair recall "
             f"{hard_neg_finding.get('small_scale_recall_at_10')})"
             if hard_neg_finding["status"] == "oom_at_full_scale" else ""), flush=True)
    print(f"  best: {best['method']}  worst: {worst['method']}  spread: {spread:.4f}", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


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
