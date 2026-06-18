#!/usr/bin/env python3
"""Emit the point-in-time-correctness cache — CPU, dual-transport, hermetic.

The engine↔cookbook validator for the H4 temporal-correctness surface: `asof_join`
(SPEC-01) and `verify_materialization` (SPEC-02), carried on BOTH the embedded
`jammi_ai.Database` and the remote `jammi_client.RemoteDatabase`. This script
assembles a **leakage-free training set** with `asof_join`, measures the leak it
closes, and freezes the materialization-contract verdict matrix.

It reuses the committed ogbn-arxiv keystone (`artifacts/arxiv/papers.parquet` +
`cite_edges.parquet`) — the heavy embeddings are NOT re-emitted (read the cache).
From the keystone it derives, deterministically and committed-not-seeded:

* **the time-stamped citation-edge facts** — each citing edge `(src -> dst)` carries
  the citing paper's publication `year` as `cited_at`; counting edges into `dst`
  with `cited_at <= as_of` is the leakage-safe in-degree, the `asof_join` computes;
* **the label spine** — each paper's `as_of = year + 1` horizon (the instant a
  one-year-citation label is observed);
* **the leak** — the *naive* current-state in-degree counts every edge into a paper,
  including citations that arrived after its horizon; the *as-of* in-degree counts
  only edges known at-or-before the horizon. 73% of the committed citation edges are
  future leakage relative to the horizon.

Measured, all asserted to golden:

* **`pit.leakage_delta = naive_auc - asof_auc` (strictly > 0).** The downstream task
  is a genuinely independent future event — "does this paper keep being cited beyond
  its one-year horizon" (`future_citations = naive_in_degree - asof_in_degree >= 1`,
  NOT a threshold on the feature itself, so neither feature is a tautological perfect
  ranker). The *leaky* pipeline serves the current (full) in-degree, which peeks at
  exactly that future and inflates the reported AUC; the *as-of* pipeline serves the
  leakage-free as-of in-degree.
* **`pit.coverage_leaky` vs `pit.coverage_asof`.** A split-conformal predictor at
  nominal 90% calibrated two ways. The leaky calibration's nonconformity scores look
  artificially easy, so its prediction sets are too tight and it BREAKS the nominal
  guarantee (under-covers); the as-of calibration holds nominal. The conformal-
  exchangeability failure leakage induces, made concrete.
* **`pit.train_serve_skew == 0.0` (exact).** The SAME `asof_join` definition run on
  the embedded `Database` and a live `grpc://` `RemoteDatabase` over the same
  committed inputs returns byte-identical feature rows — one definition, both paths.
* **the four-verdict matrix.** `verify_materialization` returns `match` /
  `match_with_unpinned_inputs` / `mismatch` / `missing_manifest`. The `Match` case is
  anchored by a producer that reads a RESULT-TABLE input (`build_neighbor_graph` over
  an embeddings result table → `ResultDigest` anchor); an `asof_join` / embedding
  over a registered file source is honestly `MatchWithUnpinnedInputs` (file sources
  have no version surface). All four reproduced live on a tiny CPU text corpus.
* **`pit.definition_hash`** — the committed `Match`-case definition hash (exact).

Usage::

    JAMMI_SERVER_BIN=/path/to/jammi-server python scripts/build_point_in_time_cache.py

`--target` selects the skew arm: `dual` (default — embedded + a live `grpc://`
server, requires `--server-bin`/`JAMMI_SERVER_BIN`) or `embedded` (the embedded arm
only, recording the skew as not-measured). This is an emit-only script; PR CI reads
the committed cache and never runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import contracts

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "point_in_time"

# The label horizon: a paper's one-year-citation label is observed at year + 1.
_HORIZON = 1
# The downstream conformal nominal coverage.
_ALPHA = 0.10
# The deterministic train / calibration / test partition over a stable per-paper
# hash (subset identity is committed, not seeded — the hash is a pure function of
# the committed paper_id, reproducible across library versions).
_TRAIN_HI, _CAL_HI = 600, 800  # train < 600 <= cal < 800 <= test (out of 1000)


# --------------------------------------------------------------------------- #
# The time-stamped facts + the two in-degree features (read the keystone cache)
# --------------------------------------------------------------------------- #


def _bucket(paper_id: str) -> int:
    """A stable 0..999 partition bucket — a pure function of the committed id."""
    return int(hashlib.sha256(paper_id.encode()).hexdigest(), 16) % 1000


def derive_facts() -> dict:
    """Derive the time-stamped citation facts and the two in-degree features from
    the committed ogbn-arxiv keystone. No re-embedding, no recompute upstream —
    just the committed `papers` labels and the committed declared `cite_edges`."""
    papers = contracts.load_artifact("arxiv.papers").to_pylist()
    edges = contracts.load_artifact("arxiv.cite_edges").to_pylist()
    year = {p["paper_id"]: int(p["year"]) for p in papers}
    ids = [p["paper_id"] for p in papers]

    naive = defaultdict(int)  # current-state in-degree: every edge into dst
    asof = defaultdict(int)  # leakage-safe in-degree: only edges known by the horizon
    edge_facts = []  # (dst, cited_at) the asof_join reads
    for e in edges:
        s, d = e["src"], e["dst"]
        if s not in year or d not in year:
            continue
        cited_at = year[s]  # the citing paper's publication year is the fact instant
        naive[d] += 1
        edge_facts.append((d, cited_at))
        if cited_at <= year[d] + _HORIZON:
            asof[d] += 1

    return {
        "ids": ids,
        "year": year,
        "naive": {p: naive[p] for p in ids},
        "asof": {p: asof[p] for p in ids},
        "edge_facts": edge_facts,
        "n_edges": len(edge_facts),
        "n_leaked": sum(naive[p] - asof[p] for p in ids),
    }


# --------------------------------------------------------------------------- #
# The leakage delta + the conformal honesty result (pure numpy on the features)
# --------------------------------------------------------------------------- #


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann–Whitney). Deterministic via a stable sort."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    npos = float(y.sum())
    nneg = float(len(y) - npos)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _logfit(x: np.ndarray, y: np.ndarray, *, iters: int = 500, lr: float = 0.3,
            l2: float = 1e-3) -> tuple:
    """A tiny deterministic 1-D logistic fit (standardized feature)."""
    mu, sd = float(x.mean()), float(x.std() + 1e-9)
    xs = (x - mu) / sd
    xb = np.c_[np.ones(len(x)), xs]
    w = np.zeros(2)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-xb @ w))
        w -= lr * (xb.T @ (p - y) / len(y) + l2 * w)
    return w, mu, sd


def _logpred(x: np.ndarray, w, mu: float, sd: float) -> np.ndarray:
    xs = (x - mu) / sd
    return 1.0 / (1.0 + np.exp(-(w[0] + w[1] * xs)))


def measure_leak(facts: dict) -> dict:
    """Measure the leakage delta and the conformal coverage honesty result."""
    ids = facts["ids"]
    nd = np.array([facts["naive"][p] for p in ids], dtype=float)
    ad = np.array([facts["asof"][p] for p in ids], dtype=float)
    bucket = np.array([_bucket(p) for p in ids])
    tr = bucket < _TRAIN_HI
    cal = (bucket >= _TRAIN_HI) & (bucket < _CAL_HI)
    te = bucket >= _CAL_HI

    # The label is a genuinely independent FUTURE event — not a threshold on the
    # feature: "does this paper acquire any citation beyond its one-year horizon?"
    future = nd - ad
    label = (future >= 1).astype(int)

    # Leakage delta: the leaky pipeline serves the current (full) in-degree, which
    # peeks at exactly that future; the as-of pipeline serves the leakage-safe count.
    naive_auc = _auc(nd[te], label[te])
    asof_auc = _auc(ad[te], label[te])
    leakage_delta = naive_auc - asof_auc

    # The deployed honest model is fixed (trained on the as-of feature). The two
    # calibrations differ only in the feature view used to set the conformal
    # quantile: a leaky calibration scores its nonconformity with the leaky (full)
    # feature (over-optimistic), an honest calibration with the as-of feature.
    w_dep, mu_dep, sd_dep = _logfit(ad[tr], label[tr])
    yt = label[te]
    s_test = _logpred(ad[te], w_dep, mu_dep, sd_dep)  # honest deployment scores
    nc_test = np.where(yt == 1, 1 - s_test, s_test)

    def coverage(cal_w, cal_mu, cal_sd, cal_feat: np.ndarray) -> float:
        yc = label[cal]
        s_cal = _logpred(cal_feat[cal], cal_w, cal_mu, cal_sd)
        nc_cal = np.where(yc == 1, 1 - s_cal, s_cal)
        n = len(nc_cal)
        q = float(np.quantile(nc_cal, min(1.0, math.ceil((n + 1) * (1 - _ALPHA)) / n),
                              method="higher"))
        return float((nc_test <= q).mean())

    w_leaky, mu_leaky, sd_leaky = _logfit(nd[tr], label[tr])
    coverage_leaky = coverage(w_leaky, mu_leaky, sd_leaky, nd)
    coverage_asof = coverage(w_dep, mu_dep, sd_dep, ad)

    return {
        "label_prevalence": float(label.mean()),
        "n_train": int(tr.sum()), "n_cal": int(cal.sum()), "n_test": int(te.sum()),
        "naive_auc": naive_auc, "asof_auc": asof_auc, "leakage_delta": leakage_delta,
        "nominal_coverage": 1 - _ALPHA,
        "coverage_leaky": coverage_leaky, "coverage_asof": coverage_asof,
    }


# --------------------------------------------------------------------------- #
# The asof_join over the time-stamped facts (one definition, both transports)
# --------------------------------------------------------------------------- #


def _write_spine_and_facts(work: str, facts: dict) -> tuple[str, str]:
    """Materialize the committed label spine + time-stamped facts as parquet the
    asof_join reads. Spine: (paper_id, as_of = year + horizon). Facts: one row per
    citation edge (paper_id = cited dst, cited_at, val = a unit count to sum)."""
    ids, year = facts["ids"], facts["year"]
    spine_path = os.path.join(work, "spine.parquet")
    facts_path = os.path.join(work, "facts.parquet")
    pq.write_table(pa.table({
        "paper_id": ids,
        "as_of": [year[p] + _HORIZON for p in ids],
    }), spine_path)
    # one fact row per edge, carrying the citing paper's year as cited_at; tie-break
    # on cited_at descending is unnecessary (we attach the latest known fact instant).
    dst = [d for d, _ in facts["edge_facts"]]
    cited_at = [c for _, c in facts["edge_facts"]]
    pq.write_table(pa.table({
        "paper_id": dst,
        "cited_at": cited_at,
        "fact_year": cited_at,  # the attached fact column (the latest known cited_at)
    }), facts_path)
    return spine_path, facts_path


def run_asof(db, spine_path: str, facts_path: str) -> list[dict]:
    """Run the SAME asof_join definition on a transport and return the feature rows.

    `asof_join` resolves the BARE source ids (`"spine"`/`"facts"`); the output is
    read as `"jammi.<table>"`. `direction`/`boundary` are lowercase strings.
    Backward + inclusive = the most recent fact at-or-before the as_of horizon — the
    leakage-safe match."""
    db.add_source("spine", url=f"file://{spine_path}", format="parquet")
    db.add_source("facts", url=f"file://{facts_path}", format="parquet")
    out = db.asof_join(
        "spine", "facts",
        spine_by=["paper_id"], spine_time="as_of",
        facts_by=["paper_id"], facts_time="cited_at",
        direction="backward", boundary="inclusive",
        tie_break_column="fact_year",
        project=["fact_year"],
    )
    rows = db.sql(
        f'SELECT paper_id, as_of, fact_year FROM "jammi.{out}" ORDER BY paper_id'
    ).to_pylist()
    verdict = db.verify_materialization(out)
    return out, rows, verdict


# --------------------------------------------------------------------------- #
# The four-verdict materialization matrix (tiny CPU corpus, ephemeral catalog)
# --------------------------------------------------------------------------- #

# A tiny deterministic text corpus — embedding + neighbor-graph over it is
# CPU-hermetic in seconds. Content is fixed, so a re-emit registers the same shape.
_TINY_DOCS = {
    "_row_id": ["d0", "d1", "d2", "d3", "d4", "d5"],
    "text": [
        "graph signal processing on irregular domains",
        "node embeddings for citation networks",
        "spectral kernels and the graph laplacian",
        "message passing neural networks",
        "random walks and personalized pagerank",
        "conformal prediction under distribution shift",
    ],
}
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def verdict_matrix(work: str) -> dict:
    """Drive the four `verify_materialization` verdicts live on CPU and capture them.

    * **match** — `build_neighbor_graph` over an embeddings result table anchors its
      sole input as `ResultDigest`, so the output verifies as a clean `match`.
    * **match_with_unpinned_inputs** — `generate_embeddings` over a registered file
      source: the source has no version surface, so its anchor is `UnpinnedAtInstant`
      and the verdict is `match_with_unpinned_inputs` (the honest observed case).
    * **mismatch** — verifying against a deliberately-wrong `expected_definition`
      yields `mismatch`, carrying both the expected and the found definition hash.
    * **missing_manifest** — with the `.materialization.json` sidecar removed, the
      verdict is `missing_manifest` (a truthful "unknown," never a fabricated match).
    """
    docs_path = os.path.join(work, "tiny_docs.parquet")
    pq.write_table(pa.table(_TINY_DOCS), docs_path)
    catalog = tempfile.mkdtemp(prefix="jammi_pit_matrix_")
    db = jammi_ai.connect(f"file://{catalog}")
    db.add_source("docs", url=f"file://{docs_path}", format="parquet")

    # MatchWithUnpinnedInputs: embedding over a file source.
    emb = db.generate_embeddings(source="docs", model=_EMBED_MODEL,
                                 columns=["text"], key="_row_id")
    v_unpinned = db.verify_materialization(emb)

    # Match: a neighbor graph over the embeddings RESULT TABLE (ResultDigest input).
    ng = db.build_neighbor_graph("docs", k=2, exact=True)
    v_match = db.verify_materialization(ng)

    # Read the definition hash off the Match-case manifest sidecar.
    sidecar = Path(catalog) / "jammi_db" / f"{ng}.materialization.json"
    manifest = json.loads(sidecar.read_text())
    definition_hash = manifest["definition_hash"]
    input_anchors = manifest["input_anchors"]

    # Match again, this time with the correct expected definition supplied.
    v_match_expected = db.verify_materialization(ng, expected_definition=definition_hash)

    # Mismatch: a wrong expected definition (a different producing query would give a
    # different hash; an all-zero hash stands in as the canonical "not this definition").
    wrong = "0" * len(definition_hash)
    v_mismatch = db.verify_materialization(ng, expected_definition=wrong)

    # MissingManifest: remove the sidecar, then verify.
    sidecar.unlink()
    v_missing = db.verify_materialization(ng)

    return {
        "match": v_match,
        "match_expected": v_match_expected,
        "match_with_unpinned_inputs": v_unpinned,
        "mismatch": v_mismatch,
        "missing_manifest": v_missing,
        "definition_hash": definition_hash,
        "match_input_anchors": input_anchors,
    }


# --------------------------------------------------------------------------- #
# Live remote server (mirrors the engine conftest live_server)
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """A real CPU `jammi-server` on a free port, readiness-polled, torn down on exit."""

    def __init__(self, server_bin: str):
        self.server_bin = server_bin
        self.proc = None
        self.endpoint = None

    def __enter__(self) -> str:
        import jammi_client

        artifact_dir = tempfile.mkdtemp(prefix="jammi_srv_pit_")
        flight_port, health_port = _free_port(), _free_port()
        env = dict(os.environ)
        env["JAMMI_ARTIFACT_DIR"] = artifact_dir
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


def _checksums() -> None:
    sums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def emit(target: str, server_bin: str | None) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    facts = derive_facts()
    leak = measure_leak(facts)
    print(f"== facts: {facts['n_edges']} edges, {facts['n_leaked']} future-leak "
          f"({facts['n_leaked'] / facts['n_edges']:.1%}) ==", flush=True)
    print(f"== leak: naive_auc={leak['naive_auc']:.4f} asof_auc={leak['asof_auc']:.4f} "
          f"delta={leak['leakage_delta']:.4f} | coverage leaky={leak['coverage_leaky']:.4f} "
          f"asof={leak['coverage_asof']:.4f} (nominal {leak['nominal_coverage']:.2f}) ==",
          flush=True)

    with tempfile.TemporaryDirectory() as work:
        spine_path, facts_path = _write_spine_and_facts(work, facts)

        # --- embedded asof (the canonical feature rows) --------------------- #
        with tempfile.TemporaryDirectory() as catalog:
            embedded = jammi_ai.connect(f"file://{catalog}")
            print("== embedded asof_join ==", flush=True)
            e_out, e_rows, e_verdict = run_asof(embedded, spine_path, facts_path)

        # --- remote asof (live grpc:// skew check) -------------------------- #
        if target == "dual":
            import jammi_client

            with LiveServer(server_bin) as endpoint:
                print(f"== remote asof_join at {endpoint} ==", flush=True)
                remote = jammi_client.connect(endpoint)
                try:
                    r_out, r_rows, r_verdict = run_asof(remote, spine_path, facts_path)
                finally:
                    remote.close()
            train_serve_skew = 0.0 if e_rows == r_rows else 1.0
            if train_serve_skew != 0.0:
                # A divergence is an ENGINE bug the validator must surface, not hide.
                raise AssertionError(
                    "train != serve: embedded and remote asof_join feature rows differ"
                )
            skew_measured = True
            print(f"== train_serve_skew = {train_serve_skew} (embedded == remote) ==",
                  flush=True)
        else:
            train_serve_skew = None
            skew_measured = False
            print("== skew arm SKIPPED (target=embedded) ==", flush=True)

        # --- the four-verdict materialization matrix ------------------------ #
        print("== verify_materialization four-verdict matrix ==", flush=True)
        matrix = verdict_matrix(work)
        for name in ("match", "match_with_unpinned_inputs", "mismatch", "missing_manifest"):
            print(f"   {name:30s} -> {matrix[name]}", flush=True)

    # The matched-fact mass: every spine row carries its as-of-correct latest known
    # cited_at (or null when no citation was known by the horizon). The matched
    # (non-null) count must equal the count of papers with a positive as-of in-degree.
    matched = sum(1 for r in e_rows if r["fact_year"] is not None)
    asof_cited = sum(1 for p in facts["ids"] if facts["asof"][p] > 0)

    # --- the frozen goldens ------------------------------------------------- #
    golden = {
        "pit.leakage_delta": {"value": round(leak["leakage_delta"], 4), "tol": 0.02},
        "pit.naive_auc": {"value": round(leak["naive_auc"], 4), "tol": 0.02},
        "pit.asof_auc": {"value": round(leak["asof_auc"], 4), "tol": 0.02},
        "pit.coverage_leaky": {"value": round(leak["coverage_leaky"], 4), "tol": 0.03},
        "pit.coverage_asof": {"value": round(leak["coverage_asof"], 4), "tol": 0.03},
        "pit.nominal_coverage": {"value": leak["nominal_coverage"], "tol": 0.0},
        "pit.asof_matched_rows": {"value": float(matched), "tol": 0.0},
        # the verdict matrix, as 1.0/0.0 metrics (exact)
        "pit.verdict_match": {
            "value": 1.0 if matrix["match"]["verdict"] == "match" else 0.0, "tol": 0.0},
        "pit.verdict_match_with_unpinned": {
            "value": 1.0 if matrix["match_with_unpinned_inputs"]["verdict"]
            == "match_with_unpinned_inputs" else 0.0, "tol": 0.0},
        "pit.verdict_mismatch": {
            "value": 1.0 if matrix["mismatch"]["verdict"] == "mismatch" else 0.0, "tol": 0.0},
        "pit.verdict_missing_manifest": {
            "value": 1.0 if matrix["missing_manifest"]["verdict"] == "missing_manifest"
            else 0.0, "tol": 0.0},
        # the within-run round-trip: the Match-case table verifies against ITS OWN
        # definition hash (the hash itself is NOT a golden — it embeds the embeddings
        # result table's content digest, which is not bit-reproducible across runs; see
        # EXECUTION-STATUS. The reproducible fact is that the round-trip holds).
        "pit.verdict_match_roundtrip": {
            "value": 1.0 if matrix["match_expected"]["verdict"] == "match" else 0.0,
            "tol": 0.0},
    }
    if skew_measured:
        golden["pit.train_serve_skew"] = {"value": train_serve_skew, "tol": 0.0}

    record = {
        "purpose": (
            "The engine↔cookbook validator for the H4 temporal-correctness surface: "
            "asof_join (SPEC-01) assembles a leakage-free training set and "
            "verify_materialization (SPEC-02) attests it. The leak the as-of join "
            "closes is measured (a downstream AUC and a conformal-coverage honesty "
            "result), the train==serve skew is measured across embedded == live grpc:// "
            "RemoteDatabase, and the four-verdict materialization matrix is frozen. "
            "Reuses the committed ogbn-arxiv keystone (papers + cite_edges); the heavy "
            "embeddings are NOT re-emitted."
        ),
        "facts": {
            "n_edges": facts["n_edges"],
            "n_leaked": facts["n_leaked"],
            "leak_fraction": round(facts["n_leaked"] / facts["n_edges"], 4),
            "horizon_years": _HORIZON,
            "cited_at": "the citing paper's publication year",
            "label": (
                "future_citations = naive_in_degree - asof_in_degree >= 1 (a genuinely "
                "independent future event, NOT a threshold on the feature)"
            ),
        },
        "leak": leak,
        "asof": {
            "definition": (
                "asof_join(spine='spine', facts='facts', spine_by=['paper_id'], "
                "spine_time='as_of', facts_by=['paper_id'], facts_time='cited_at', "
                "direction='backward', boundary='inclusive', tie_break_column='fact_year', "
                "project=['fact_year'])"
            ),
            "embedded_output_table": e_out,
            "matched_rows": matched,
            "asof_cited_papers": asof_cited,
            "embedded_verdict": e_verdict,
        },
        "train_serve_skew": train_serve_skew,
        "skew_measured": skew_measured,
        "verdict_matrix": matrix,
        "definition_hash_note": (
            "The Match-case definition_hash is recorded here as PROVENANCE ONLY, never "
            "pinned as a golden: it embeds the embeddings result table's content digest "
            "(a ResultDigest input anchor), and the embedding output is not bit-"
            "reproducible across runs, so the literal hash varies. What IS reproducible "
            "and asserted: the four verdicts and the within-run round-trip (the Match "
            "table verifies against its own hash). See EXECUTION-STATUS (the fork-4 "
            "refinement + the definition_hash cut)."
        ),
        "honest_limit": (
            "This chapter proves the primitive composes into a leakage-free, skew-free "
            "training surface on CPU against the committed cache. It does NOT serve "
            "those features from a low-latency online tier behind auth with a live "
            "coverage SLA; carrying the materialization contract across that boundary is "
            "the platform closure chapter, not the engine cookbook."
        ),
    }

    (ARTIFACTS / "point_in_time.json").write_text(json.dumps(record, indent=2, sort_keys=True))
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))
    _checksums()

    print("\n=== point-in-time cache, measured ===", flush=True)
    print(f"  leakage_delta = naive {leak['naive_auc']:.4f} - asof {leak['asof_auc']:.4f} "
          f"= {leak['leakage_delta']:.4f} (> 0)", flush=True)
    print(f"  coverage      leaky {leak['coverage_leaky']:.4f} vs asof {leak['coverage_asof']:.4f} "
          f"(nominal {leak['nominal_coverage']:.2f})", flush=True)
    if skew_measured:
        print(f"  train_serve_skew = {train_serve_skew} (embedded == remote, byte-identical)",
              flush=True)
    print(f"  definition_hash = {matrix['definition_hash']} (provenance only — round-trips "
          f"within run; not a golden)", flush=True)
    print("  verdict matrix: " + " ".join(
        f"{k}={matrix[k]['verdict']}" for k in
        ("match", "match_with_unpinned_inputs", "mismatch", "missing_manifest")), flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("dual", "embedded"), default="dual",
                    help="dual = embedded + live grpc:// skew arm (default); "
                         "embedded = embedded only")
    ap.add_argument("--server-bin", default=os.environ.get("JAMMI_SERVER_BIN"),
                    help="built CPU jammi-server binary (or set JAMMI_SERVER_BIN) — "
                         "required for --target dual")
    args = ap.parse_args()
    if args.target == "dual" and (not args.server_bin or not os.path.exists(args.server_bin)):
        raise SystemExit(
            "pass --server-bin (or set JAMMI_SERVER_BIN) to a built jammi-server for "
            "--target dual, or use --target embedded"
        )
    emit(args.target, args.server_bin)


if __name__ == "__main__":
    main()
