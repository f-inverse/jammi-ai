"""The keystone's steps, each defined once.

A tier chapter introduces its step and shows it (:func:`show`); a later
chapter that builds on a step runs the same function. So a chapter is always
runnable from a fresh engine, and there is one definition of every step.
"""

from __future__ import annotations

import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from . import datasets, encoders
from .datasets import Arxiv
from .scale import Scale


def show(step) -> "object":
    """``step``'s source as a rendered code block — the exact code a cell runs."""
    from IPython.display import Markdown

    return Markdown(f"```python\n{inspect.getsource(step)}```")


def embed(db, arxiv: Arxiv, scale: Scale) -> str:
    """Tier 01: embed every paper's title and abstract."""
    return db.generate_embeddings(
        source=arxiv.papers,
        model=encoders.text(scale),
        columns=["title", "abstract"],
        key="paper_id",
    )


def propagate(db, arxiv: Arxiv, embeddings: str) -> str:
    """Tier 02: APPNP over the citation graph — two hops of symmetric
    normalized averaging with a 10% teleport back to each paper's own vector."""
    return db.propagate_embeddings(
        arxiv.papers,
        embedding_table=embeddings,
        edge_source=arxiv.cites,
        edge_src_column="src",
        edge_dst_column="dst",
        direction="out",
        hops=2,
        weighting="degree_normalized",
        alpha=0.1,
    )


# Tier 03's epochs: a graph-supervised contrastive fine-tune converges over tens
# of epochs (SPECTER trains for tens), and at full scale the declared-edge gain
# is still rising at 15; the two control graphs train at a matched, cheaper
# budget. The small scale's encoder has nothing to converge to, so a short run
# exercises the same path.
FINE_TUNE_EPOCHS = {Scale.SMALL: 2, Scale.FULL: 15}
CONTROL_EPOCHS = {Scale.SMALL: 2, Scale.FULL: 5}


def fine_tune_on_graph(
    db, arxiv: Arxiv, scale: Scale, *, edges: str, provenance: str, epochs: int
) -> str:
    """Tier 03: fine-tune the text encoder contrastively over random walks on
    ``edges`` (a registered source of ``src``/``dst`` paper ids), then embed
    every paper with the fine-tuned model. Returns the new embedding table."""
    job = db.fine_tune_graph(
        node_source=arxiv.papers,
        id_column="paper_id",
        text_column="abstract",
        edge_source=edges,
        src_column="src",
        dst_column="dst",
        base_model=encoders.text(scale),
        edge_provenance=provenance,
        epochs=epochs,
        batch_size=32,
        walks_per_node=2,
        walk_length=4,
        sample_seed=0,
    )
    job.wait()
    return db.generate_embeddings(
        source=arxiv.papers,
        model=job.output_model_id,
        columns=["title", "abstract"],
        key="paper_id",
    )


def register_edges(db, table: str, name: str) -> str:
    """Register an engine-produced edge table's ``src``/``dst`` rows as a
    source named ``name`` — a training job's graph must be a registered source."""
    path = Path(tempfile.mkdtemp()) / f"{name}.parquet"
    pq.write_table(db.sql(f'SELECT src, dst FROM "jammi.{table}"'), path)
    db.add_source(name, url=str(path), format="parquet")
    return name


# Tier 04's context-predictor meta-training epochs.
PREDICTOR_EPOCHS = {Scale.SMALL: 20, Scale.FULL: 80}


def train_year_predictor(db, arxiv: Arxiv, scale: Scale, embeddings: str) -> str:
    """Tier 04: meta-train a Gaussian context predictor of a paper's ``year``,
    one task per ``subject``, reading each target's context from its nearest
    papers in ``embeddings`` — the table the predictor then always serves
    from. Returns the predictor's model id."""
    job = db.train_context_predictor(
        arxiv.papers,
        embedding_table=embeddings,
        key_column="paper_id",
        task_column="subject",
        value_column="year",
        architecture="attncnp",
        output="gaussian",
        objective="crps",
        epochs=PREDICTOR_EPOCHS[scale],
        seed=0,
    )
    job.wait()
    return job.output_model_id


def subject_golden(db, arxiv: Arxiv) -> str:
    """The same-subject retrieval golden: 200 papers asked by their titles, each
    relevant to every other paper of its subject — a target independent of any
    embedding. Returns the golden's relation for the ``eval_*`` verbs."""
    papers = db.sql(
        f"SELECT paper_id, title, subject FROM {arxiv.papers}.public.{arxiv.papers}"
    ).to_pylist()
    return datasets.same_label_golden(
        db, papers, key="paper_id", label="subject", text="title", queries=200,
        name="arxiv_subject_golden",
    )


def predict_years(db, arxiv: Arxiv, predictor: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Tier 04: the predictor's served year distribution — mean and standard
    deviation — for each paper in ``keys``."""
    served = [
        db.predict_with_context_predictor(predictor, source=arxiv.papers, target_key=k)
        for k in keys
    ]
    return np.array([s["mean"] for s in served]), np.array([s["std"] for s in served])


# Neighbours a paper's subject is voted by, and a vote's temperature on the
# neighbour's similarity.
NEIGHBOURS = 25
VOTE_TEMP = 20.0
# A class no neighbour voted for keeps this share, so every set can grow to it.
VOTE_FLOOR = 1e-3


def neighbours(db, arxiv: Arxiv, embeddings: str, key: str, among: str, k: int) -> list[dict]:
    """``key``'s ``k`` nearest other papers in ``embeddings`` that satisfy
    ``among`` (a SQL predicate), nearest first — a query-by-example
    ``search``: the paper's stored vector never leaves the engine."""
    return db.search(
        arxiv.papers, row_key=key, k=k, embedding_table=embeddings,
        filter=f"({among}) AND paper_id <> '{key}'",
        select=["paper_id", "subject", "year", "similarity"],
    ).to_pylist()


@dataclass(frozen=True)
class SubjectScores:
    """Tier 04's subject classifier over one embedding table: class scores and
    true labels for the calibration (2018) and test (2019–) eras, and each
    era's paper keys."""

    classes: list[str]
    cal_scores: np.ndarray
    cal_labels: np.ndarray
    test_scores: np.ndarray
    test_labels: np.ndarray
    cal_keys: list[str]
    test_keys: list[str]


def subject_scores(db, arxiv: Arxiv, embeddings: str) -> SubjectScores:
    """Tier 04: a paper's class scores are a vote of its nearest training-era
    papers in ``embeddings``, each weighted by its similarity."""
    subject = {
        r["paper_id"]: r["subject"]
        for r in db.sql(
            f"SELECT paper_id, subject FROM {arxiv.papers}.public.{arxiv.papers}"
        ).to_pylist()
    }
    classes = sorted(set(subject.values()))

    def scores(key: str) -> np.ndarray:
        hood = neighbours(db, arxiv, embeddings, key, f"year <= {datasets.TRAIN_UNTIL}", NEIGHBOURS)
        top = hood[0]["similarity"]
        votes = np.full(len(classes), VOTE_FLOOR)
        for n in hood:
            votes[classes.index(n["subject"])] += np.exp(VOTE_TEMP * (n["similarity"] - top))
        return votes / votes.sum()

    cal, test = arxiv.split["valid"], arxiv.split["test"]
    return SubjectScores(
        classes=classes,
        cal_scores=np.array([scores(k) for k in cal]),
        cal_labels=np.array([classes.index(subject[k]) for k in cal]),
        test_scores=np.array([scores(k) for k in test]),
        test_labels=np.array([classes.index(subject[k]) for k in test]),
        cal_keys=cal,
        test_keys=test,
    )


def test_era_shares(db, arxiv: Arxiv, embeddings: str, keys: list[str], sizes: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Each paper's share of test-era (2019–) papers among its nearest
    calibration- and test-era neighbours, at each neighbourhood size — how
    test-era-like its neighbourhood is."""
    hoods = [
        neighbours(db, arxiv, embeddings, key, f"year >= {datasets.VALID_YEAR}", max(sizes))
        for key in keys
    ]
    return {
        size: np.array([np.mean([n["year"] > datasets.VALID_YEAR for n in h[:size]]) for h in hoods])
        for size in sizes
    }
