"""Functional end-to-end test for the embedded training collapse.

The embedded `Database.fine_tune` drives the SAME pure-Python request assembly
(`jammi_client._assembly`) the remote client uses, serializes the proto, and
hands the bytes to the `_start_training_proto` PyO3 primitive, which decodes
through the shared `jammi_ai::wire` seam into a `TrainingSpec` and runs it on the
in-process session (`InferenceSession::run_training_spec`). Signature parity is
covered by `test_conformance`; this asserts the path actually *trains* — the link
no signature check can prove.

Hermetic: the local `tiny_bert` model fixture + a tiny `training_pairs.csv`, on
CPU, into a temp artifact dir. No network, no GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import jammi_ai

# crates/jammi-python/tests/this_file -> repo root is three parents up.
_ROOT = Path(__file__).resolve().parents[3]
_TINY_BERT = _ROOT / "cookbook" / "fixtures" / "tiny_bert"
_TRAINING_PAIRS = _ROOT / "tests" / "fixtures" / "training_pairs.csv"

pytestmark = pytest.mark.skipif(
    not _TINY_BERT.is_dir() or not _TRAINING_PAIRS.is_file(),
    reason="local tiny_bert / training_pairs fixtures not present",
)


def _connect(tmp_path: Path):
    db = jammi_ai.connect(f"file://{tmp_path}")
    # connect("file://") returns the thin Python wrapper, not the raw handle.
    assert type(db).__name__ == "Database"
    assert type(db).__module__ == "jammi_ai._database"
    return db


def test_embedded_fine_tune_runs_through_the_shared_assembly(tmp_path: Path) -> None:
    db = _connect(tmp_path)
    db.add_source("training", url=str(_TRAINING_PAIRS), format="csv")

    job = db.fine_tune(
        source="training",
        base_model=f"local:{_TINY_BERT}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=2,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )

    # The job submitted through Database.fine_tune -> build_fine_tune_request ->
    # _start_training_proto -> training_spec_from_bytes -> run_training_spec ->
    # session.fine_tune, and the embedded worker carries it to completion.
    assert job.model_id.startswith("jammi:fine-tuned:")
    job.wait()
    assert job.status() == "completed"


def test_embedded_fine_tune_rejects_unknown_method_in_the_assembly(tmp_path: Path) -> None:
    db = _connect(tmp_path)
    db.add_source("training", url=str(_TRAINING_PAIRS), format="csv")

    # The method vocabulary is validated in the shared Python assembly, before any
    # transport — the same `ValueError` the remote client raises, never an opaque
    # decode error from across the PyO3 boundary.
    with pytest.raises(ValueError, match="method must be one of"):
        db.fine_tune(
            source="training",
            base_model=f"local:{_TINY_BERT}",
            columns=["text_a"],
            method="bogus",
            task="text_embedding",
        )
