"""A training-job handle outlives the connection that submitted it.

`fine_tune` hands back a handle, but until now that handle was the ONLY way to
reach the job from Python: it died with its connection (on the remote arm, with
the channel), and no public verb re-attached to a job by id. The acceleration
chapter hit this directly and had to build a `jammi.RemoteTrainingJob` out of
`RemoteDatabase._training` / `._metadata` — private access it named as a gap.
`TrainingService.ListTrainingJobs` was likewise served by the engine and exposed
on neither arm, while the native `_NativeDatabase.training_job(job_id)` attach
existed but was reachable through no client verb — the K4 asymmetry its own doc
claims to close, still open one layer up.

This module pins both verbs on the arm that needs no server:

* `training_job(job_id)` — attach by id on a connection that never submitted the
  job, with every read verb (`status`, `metrics`, `acceleration_report`, `wait`)
  working on it;
* `list_training_jobs()` — the wire's `TrainingJobSummary` field set, tenant
  scoped;
* the typed not-found, which must be one CLASS across both arms.

The cross-transport value parity of the same three lives in
`clients/python/tests/test_remote_training_job_live.py` (it needs a real
`jammi-server`).

The submit-then-attach split is driven through `training.run_worker`: the
submitting connection runs no claim loop, so the job is still `queued` when the
successor connection attaches — the attach is then observably an attach, not a
race against a worker that already finished the job in the submitting process.

Hermetic: the local `tiny_bert` model fixture + a tiny `training_pairs.csv`, on
CPU, into a temp artifact dir. No network, no GPU.
"""

from __future__ import annotations

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

_RUN_WORKER_ENV = "JAMMI_TRAINING__RUN_WORKER"

# The `TrainingJobSummary` field set, verbatim from `jammi/v1/training.proto`.
# The embedded listing must carry exactly these keys, so a caller reads one
# vocabulary regardless of transport.
_SUMMARY_KEYS = {
    "job_id",
    "kind",
    "status",
    "base_model_id",
    "output_model_id",
    "created_at",
    "error",
}


def _submit(db, *, source: str = "training"):
    return db.fine_tune(
        source=source,
        base_model=f"local:{_TINY_BERT}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=1,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )


def _connect_with_source(tmp_path: Path, *, source: str = "training"):
    db = jammi.connect(f"file://{tmp_path}")
    db.add_source(source, url=str(_TRAINING_PAIRS), format="csv")
    return db


def test_training_job_attaches_on_a_connection_that_never_submitted_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Submit on one connection, close it, attach on a fresh one — and every
    read verb works on the attached handle, including a `wait()` that carries
    the job to completion.

    The submitting connection runs no claim loop (`run_worker=false`), so the
    job is genuinely still `queued` at the moment of the attach: what is proven
    is a handle built from the catalog by id, not a handle that happened to
    survive in memory. The successor connection claims and finishes the job, so
    the same attached handle is also what a caller polls to completion —
    the whole point of a handle that outlives its submitter.
    """
    monkeypatch.setenv(_RUN_WORKER_ENV, "false")
    submitter = _connect_with_source(tmp_path)
    submitted = _submit(submitter)
    job_id = submitted.job_id
    submitted_model_id = submitted.model_id
    assert submitted.status() == "queued"
    submitter.close()
    del submitted, submitter

    monkeypatch.delenv(_RUN_WORKER_ENV, raising=False)
    successor = jammi.connect(f"file://{tmp_path}")
    try:
        attached = successor.training_job(job_id)

        # Identity: the same job, and the same deterministic output model id the
        # submit call handed back — re-derived from the catalog row, never
        # invented here.
        assert attached.job_id == job_id
        assert attached.model_id == submitted_model_id

        # Every read verb works on a job this connection did not submit.
        assert attached.status() in {"queued", "running", "completed"}
        assert attached.acceleration_report()["state"] in {"pending", "determined"}
        assert isinstance(attached.metrics(), dict)

        attached.wait()
        assert attached.status() == "completed"
        assert attached.acceleration_report()["state"] == "determined"
        assert attached.metrics()["total_steps"] > 0
    finally:
        successor.close()


def test_training_job_not_found_raises_the_typed_error(tmp_path: Path) -> None:
    """An id with no matching row raises the typed `BackendError` — the SAME
    class the remote arm raises for the same miss (the live parity module pins
    the two against each other), never `None` and never a handle that fails
    later on its first read."""
    from jammi.errors import BackendError

    db = jammi.connect(f"file://{tmp_path}")
    try:
        with pytest.raises(BackendError):
            db.training_job("no-such-job-id")
    finally:
        db.close()


def test_list_training_jobs_carries_the_wire_field_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`list_training_jobs()` lists this tenant's jobs with exactly the wire's
    `TrainingJobSummary` field set, and the same empty-string conventions the
    server relays (`output_model_id` empty until completion, `error` empty
    unless failed).

    Two jobs are submitted so the listing is a real listing; they are compared
    as a SET of ids, because `created_at` is a text timestamp two submissions in
    the same second share — an order assertion there would pin clock resolution,
    not the verb.
    """
    monkeypatch.setenv(_RUN_WORKER_ENV, "false")
    db = _connect_with_source(tmp_path)
    try:
        first = _submit(db).job_id
        second = _submit(db).job_id

        jobs = db.list_training_jobs()
        assert isinstance(jobs, list)
        assert {j["job_id"] for j in jobs} == {first, second}
        for entry in jobs:
            assert set(entry.keys()) == _SUMMARY_KEYS, entry
            assert entry["kind"] == "fine_tune"
            assert entry["status"] == "queued"
            assert entry["base_model_id"]
            assert entry["created_at"]
            # Queued: no output model yet, and no failure — both are the empty
            # string, never `None`, matching the wire's own convention.
            assert entry["output_model_id"] == ""
            assert entry["error"] == ""
    finally:
        db.close()


def test_list_training_jobs_is_tenant_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer tenant's job is invisible in the listing — the same row predicate
    the Rust verb applies (`WHERE tenant_id = $1 OR tenant_id IS NULL`), not a
    listing that happens to be empty."""
    tenant_a = "11111111-1111-4111-8111-111111111111"
    tenant_b = "22222222-2222-4222-8222-222222222222"

    monkeypatch.setenv(_RUN_WORKER_ENV, "false")
    db = jammi.connect(f"file://{tmp_path}")
    try:
        db.set_tenant(tenant_a)
        db.add_source("training", url=str(_TRAINING_PAIRS), format="csv")
        job_id = _submit(db).job_id
        assert [j["job_id"] for j in db.list_training_jobs()] == [job_id]

        db.set_tenant(tenant_b)
        assert db.list_training_jobs() == []
        # And B cannot attach to it by id either — the same predicate, on the
        # same row, through the other verb.
        from jammi.errors import BackendError

        with pytest.raises(BackendError):
            db.training_job(job_id)

        db.set_tenant(tenant_a)
        assert [j["job_id"] for j in db.list_training_jobs()] == [job_id]
    finally:
        db.close()
