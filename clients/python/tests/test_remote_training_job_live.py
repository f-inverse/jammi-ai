"""`training_job(id)` and `list_training_jobs()` read IDENTICALLY on both transports.

Stands up a real CPU `jammi-server` over an artifact directory an EMBEDDED
engine seeded and released first, so both arms answer about ONE catalog row.
Two separately-submitted jobs would only prove each arm self-consistent; the
claim here is that a caller who swaps transports reads the same values.

The job is held `queued` on BOTH sides by configuration — the submitting
embedded session and the server are each started with
`JAMMI_TRAINING__RUN_WORKER=false`, so nothing claims it and the row a test
reads twice does not move underneath the comparison. That is also the whole
attach story in miniature: a job submitted by a process that is gone, read by a
process that never submitted it.

Skipped unless `JAMMI_SERVER_BIN` points at a built `jammi-server` AND the
`[embedded]` extra is installed (the seeder is the in-process engine) — the same
gate shape every other live module in this directory declares.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

import pytest

import jammi
from jammi.errors import BackendError

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")

pytestmark = pytest.mark.skipif(
    not (SERVER_BIN and Path(SERVER_BIN).is_file()) or find_spec("jammi_native") is None,
    reason="needs JAMMI_SERVER_BIN and the [embedded] extra (the seeder is the in-process engine)",
)

# A base model directory that does not exist. `create_training_job` runs at
# SUBMIT time and reads nothing but the catalog, so this commits a real
# training-job row in milliseconds — no model load, no fixture, no training. The
# job is never claimed by anything here, so it never fails either: it stays
# exactly `queued`, which is the state under comparison.
_ABSENT_MODEL = "/nonexistent/jammi/attach-parity/model"

_NO_CLAIM = {"JAMMI_TRAINING__RUN_WORKER": "false"}

_SUMMARY_KEYS = {
    "job_id",
    "kind",
    "status",
    "base_model_id",
    "output_model_id",
    "created_at",
    "error",
}


def _seed(artifact_dir: Path, monkeypatch) -> tuple[str, dict, dict, str]:
    """Submit one job through the embedded engine, read it back through the
    ATTACH path, and release the catalog.

    Returns `(job_id, embedded_summary, embedded_report, embedded_status)`.
    """
    monkeypatch.setenv("JAMMI_TRAINING__RUN_WORKER", "false")
    csv = artifact_dir.parent / "pairs.csv"
    csv.write_text("text_a,text_b,score\na,b,1.0\nc,d,0.0\n")

    db = jammi.connect(f"file://{artifact_dir}")
    try:
        db.add_source("attach_parity", url=str(csv), format="csv")
        job_id = db.fine_tune(
            source="attach_parity",
            base_model=f"local:{_ABSENT_MODEL}",
            columns=["text_a", "text_b", "score"],
            method="lora",
            task="text_embedding",
            epochs=1,
            batch_size=8,
            lora_rank=4,
            warmup_steps=0,
        ).job_id

        # The attach path itself, on the arm that owns the catalog.
        attached = db.training_job(job_id)
        status = attached.status()
        report = attached.acceleration_report()
        summaries = db.list_training_jobs()
    finally:
        db.close()
    monkeypatch.delenv("JAMMI_TRAINING__RUN_WORKER", raising=False)

    assert status == "queued"
    assert len(summaries) == 1, summaries
    return job_id, summaries[0], report, status


def test_remote_and_embedded_attach_and_list_agree(tmp_path, monkeypatch, live_server_on):
    """The remote arm's `training_job(id)` reads and `list_training_jobs()`
    entries equal the embedded arm's, on the same row — plus the not-found error
    is one CLASS on both.

    Compared whole, not spot-checked: the summary dicts are `==`, the status
    strings are `==`, and the acceleration reports are `==` (the byte-exact
    `{"state": "pending"}` marker of a job no claimant has touched).
    """
    artifact_dir = tmp_path / "engine"
    artifact_dir.mkdir()
    job_id, embedded_summary, embedded_report, embedded_status = _seed(
        artifact_dir, monkeypatch
    )

    with live_server_on(artifact_dir, env_overrides=_NO_CLAIM) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            handle = remote.training_job(job_id)
            assert handle.job_id == job_id
            assert handle.status() == embedded_status
            assert handle.acceleration_report() == embedded_report
            assert handle.metrics() == {}

            assert remote.list_training_jobs() == [embedded_summary]

            # The not-found error is the same CLASS on both arms. The embedded
            # half of this pair is asserted in
            # `crates/jammi-python/tests/test_training_job_attach.py`; here the
            # remote half is pinned against the same class object.
            with pytest.raises(BackendError):
                remote.training_job("no-such-job-id")
        finally:
            remote.close()

    # The comparison was on real content, not on two empty answers.
    assert embedded_summary["job_id"] == job_id
    assert set(embedded_summary.keys()) == _SUMMARY_KEYS
    assert embedded_summary["status"] == "queued"
    assert embedded_summary["output_model_id"] == ""
    assert embedded_summary["error"] == ""
    assert embedded_report == {"state": "pending"}


def test_remote_attach_model_id_is_empty_until_completion(
    tmp_path, monkeypatch, live_server_on
):
    """The ONE stated difference between the two attach paths, pinned so it
    cannot drift silently.

    `TrainingStatusResponse.model_id` is empty until the job completes — the
    wire's own documented contract — so a remote attach to a QUEUED job has no
    output model id to carry. The embedded attach re-derives the deterministic
    id from the persisted spec, which is an ENGINE rule; the client does not
    reimplement it, because a second implementation of a naming scheme the
    server owns is exactly the drift this conformance suite exists to prevent.

    Closing the difference is a server/wire change (have `TrainingStatus` carry
    the derived id pre-completion, as the embedded attach does), not a client
    one — so it is recorded here as a difference with a reason, not hidden.
    """
    artifact_dir = tmp_path / "engine"
    artifact_dir.mkdir()
    job_id, _, _, _ = _seed(artifact_dir, monkeypatch)

    monkeypatch.setenv("JAMMI_TRAINING__RUN_WORKER", "false")
    embedded = jammi.connect(f"file://{artifact_dir}")
    try:
        embedded_model_id = embedded.training_job(job_id).model_id
    finally:
        embedded.close()
    monkeypatch.delenv("JAMMI_TRAINING__RUN_WORKER", raising=False)

    with live_server_on(artifact_dir, env_overrides=_NO_CLAIM) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            assert remote.training_job(job_id).model_id == ""
        finally:
            remote.close()

    assert embedded_model_id == f"jammi:fine-tuned:{job_id}"
