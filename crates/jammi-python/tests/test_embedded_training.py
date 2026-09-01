"""Functional end-to-end test for the embedded training collapse.

The embedded `Database.fine_tune` drives the SAME pure-Python request assembly
(`jammi._assembly`) the remote client uses, serializes the proto, and
hands the bytes to the `_start_training_proto` PyO3 primitive, which decodes
through the shared `jammi_ai::wire` seam into a `TrainingSpec` and runs it on the
in-process session (`InferenceSession::run_training_spec`). Signature parity is
covered by `test_conformance`; this asserts the path actually *trains* — the link
no signature check can prove.

Hermetic: the local `tiny_bert` model fixture + a tiny `training_pairs.csv`, on
CPU, into a temp artifact dir. No network, no GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import jammi

# crates/jammi-python/tests/this_file -> repo root is three parents up.
_ROOT = Path(__file__).resolve().parents[3]
_TINY_BERT = _ROOT / "cookbook" / "fixtures" / "tiny_bert"
_TRAINING_PAIRS = _ROOT / "tests" / "fixtures" / "training_pairs.csv"

pytestmark = pytest.mark.skipif(
    not _TINY_BERT.is_dir() or not _TRAINING_PAIRS.is_file(),
    reason="local tiny_bert / training_pairs fixtures not present",
)


def _connect(tmp_path: Path):
    db = jammi.connect(f"file://{tmp_path}")
    # connect("file://") returns the embedded `Session` (the migrated
    # `EmbeddedBackend`, defined once in `jammi._embedded` and re-exposed as
    # `jammi.EmbeddedBackend`), not the raw native handle.
    assert type(db).__name__ == "EmbeddedBackend"
    assert type(db).__module__ == "jammi._embedded"
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


def test_embedded_fine_tune_metrics_surfaces_val_loss_run_summary(tmp_path: Path) -> None:
    """`TrainingJob.metrics()` (#441): the completed job's run summary, sourced
    straight from the catalog's `training_jobs.metrics` column via the same
    embedded session the job ran on — proving the built binding, not a log.

    Engine-not-platform check: `metrics()` returns generic training telemetry
    (loss/step/timing) with no consumer-specific field — the same shape any
    fine-tuning caller reaches for (B1-clean).
    """
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
        validation_fraction=0.2,
        early_stopping_metric="val_loss",
    )
    job.wait()
    assert job.status() == "completed"

    metrics = job.metrics()
    assert isinstance(metrics, dict)
    # Divergence-prone case: the val_loss arm must actually be the metric
    # recorded, not silently the train_loss default.
    assert metrics["early_stopping_metric"] == "val_loss"
    assert math.isfinite(metrics["final_loss"])
    assert metrics["total_steps"] > 0
    assert metrics["started_at"]
    assert metrics["completed_at"]

    # Per-epoch loss curves (issue #441) — asserted against the shape
    # `TrainingLoop::run` actually emits (`crates/jammi-ai/src/fine_tune/
    # trainer.rs`): a list of `{"epoch": int, "loss": float}` rows, one per
    # epoch actually run, never a bare `[epoch, loss]` pair (the wrong shape
    # would pass a looser check but silently diverge from the trainer's own
    # `curve_json`).
    train_curve = metrics["train_loss_curve"]
    assert isinstance(train_curve, list)
    assert len(train_curve) == 2  # epochs=2 was requested above
    for row in train_curve:
        assert set(row.keys()) == {"epoch", "loss"}
        assert math.isfinite(row["loss"])

    # This run measures ValLoss every epoch (early_stopping_metric="val_loss"),
    # so val_loss_curve must be present — never silently omitted the way a
    # TrainLoss-monitored run's would be.
    val_curve = metrics["val_loss_curve"]
    assert isinstance(val_curve, list)
    assert len(val_curve) == 2
    for row in val_curve:
        assert set(row.keys()) == {"epoch", "loss"}
        assert math.isfinite(row["loss"])


def test_embedded_fine_tune_acceleration_report_three_states(tmp_path: Path) -> None:
    """`TrainingJob.acceleration_report()` (campaign #443): the catalog's
    `training_jobs.acceleration_report` column, decoded the same way
    `metrics()` decodes its column (issue #441) but preserving THIS column's
    own two-state contract (`TrainingJobRecord::acceleration_report`'s doc,
    migration 026) rather than `metrics()`'s "absent means `{}`" default —
    proven against a REAL embedded engine + catalog on all three states:

      * the submission-time `{"state": "pending"}` marker
        `Catalog::create_training_job` stamps before any claimant has
        recorded a determination.
      * the claiming worker's `{"state": "determined", ...}` payload, from a
        REAL run that actually claimed the job and probed its device/dtype/
        kernel-admission (esc-075) — not a hand-built stand-in, so this
        exercises the real producer path, not just the read side.
      * SQL `NULL` (a row this code never touched) -> Python `None`, never
        silently coerced to `{}` or read as any particular acceleration
        state.

    Reuses ONE real fine-tune run for all three reads with the SAME
    close-before-inject discipline `test_conformance.py`'s
    `test_remote_and_embedded_training_job_metrics_agree_on_all_three_states`
    documents (esc-073): a raw `sqlite3` seed write never overlaps a live
    engine connection on the same WAL file, which otherwise reproduces a hard
    interpreter crash. `jammi.EmbeddedBackend` has no `training_job`/`close`
    convenience (asymmetric with the remote client — see that test's
    docstring), so this drives the compiled `jammi_native` primitives
    directly, like it does.
    """
    import sqlite3

    import jammi_native
    from jammi._assembly import build_fine_tune_request

    pending_marker = '{"state":"pending"}'

    catalog_db = tmp_path / "catalog.db"

    def _set_acceleration_report(job_id: str, value) -> None:
        conn = sqlite3.connect(str(catalog_db))
        try:
            conn.execute(
                "UPDATE training_jobs SET acceleration_report = ? WHERE job_id = ?",
                (value, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    # One real job, driven to completion directly against the compiled
    # `_NativeDatabase` — the same request assembly `EmbeddedBackend.fine_tune`
    # drives, minus the wrapper.
    submit_db = jammi_native.open_local(artifact_dir=str(tmp_path))
    submit_db.add_source("training", url=str(_TRAINING_PAIRS), format="csv")
    request = build_fine_tune_request(
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
    submit_job = submit_db._start_training_proto(request.SerializeToString())
    submit_job.wait()
    assert submit_job.status() == "completed"
    job_id = submit_job.job_id

    # State 1: determined — the job's OWN natural post-completion state. The
    # esc-075 report-computation runs synchronously right after device
    # resolution, before the training loop's first step, so a completed run's
    # row is `"determined"` for its whole lifetime — never read here as the
    # submission-time `"pending"` marker.
    determined = submit_job.acceleration_report()
    assert isinstance(determined, dict)
    assert determined["state"] == "determined"
    assert isinstance(determined["attempt"], int)
    assert determined["device"]
    assert determined["dtype"]
    assert isinstance(determined["cuda_compiled"], bool)
    assert isinstance(determined["flash_compiled"], bool)

    # Deterministic teardown BEFORE any raw-sqlite3 injection (see the
    # docstring): no engine connection this test opened is attached to the
    # catalog file past this point.
    submit_db.close()
    del submit_job, submit_db

    def _seed_and_attach(seed):
        """Inject `seed` through a raw connection that is fully closed before
        this returns, then attach a FRESH `Database` to the shared job by id
        — that pool's first read."""
        _set_acceleration_report(job_id, seed)
        db = jammi_native.open_local(artifact_dir=str(tmp_path))
        return db, db.training_job(job_id)

    # State 2: pending — injected explicitly (the completed job's own natural
    # state has already moved past it; see the docstring on why the "absent"/
    # pre-determination state cannot be reused from the natural run here).
    # The SAME literal `create_training_job` stamps at submission time.
    pending_db, pending_job = _seed_and_attach(pending_marker)
    assert pending_job.acceleration_report() == {"state": "pending"}
    pending_db.close()
    del pending_job, pending_db

    # State 3: `NULL` -> `None` — a row this code never touched (pre-migration
    # or a kind that never records one). Never coerced into `{}` or any state
    # claim.
    null_db, null_job = _seed_and_attach(None)
    assert null_job.acceleration_report() is None
    null_db.close()
    del null_job, null_db


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
