"""The keystone's steps, each defined once.

A tier chapter introduces its step and shows it (:func:`show`); a later
chapter that builds on a step runs the same function. So a chapter is always
runnable from a fresh engine, and there is one definition of every step.
"""

from __future__ import annotations

import inspect
import tempfile
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


def vectors(db, table: str) -> tuple[list[str], np.ndarray]:
    """An embedding table's row keys and its vectors as a matrix, in key order."""
    rows = db.sql(f'SELECT _row_id, vector FROM "jammi.{table}" ORDER BY _row_id')
    ids = [str(k) for k in rows.column("_row_id").to_pylist()]
    return ids, np.asarray(rows.column("vector").to_pylist(), dtype=np.float32)


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
