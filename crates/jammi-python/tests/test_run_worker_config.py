"""`training.run_worker` reaches the EMBEDDED arm through the same configuration
surface the server binary reads (campaign #446, GAP-A-3 embedded leg).

`run_worker` decides whether THIS process runs the training claim loop
(`jammi_db::config::TrainingConfig::run_worker`). It is a runtime/driver
setting, not a build feature and not a server-only knob: the embedded engine a
caller reaches through `jammi.connect("file://…")` both accepts submissions and
runs them, so it is exactly the arm that has to be able to stop running them.
A knob only the server honoured would be a server-only feature — precisely the
divergence between the in-process and over-the-wire deployments the
constitution forbids.

**How the config reaches this arm.** `jammi_native.open_local` builds its
`JammiConfig` with `JammiConfig::load(...)` — the SAME call
`crates/jammi-server/src/main.rs` makes — so an embedded process resolves its
config file (explicit `config=` path, `JAMMI_CONFIG`, `./jammi.toml`, the
platform config dir) and then layers the `JAMMI_*` environment overrides onto
it, `JAMMI_TRAINING__RUN_WORKER` among them. The explicit `open_local` kwargs
(`artifact_dir=`, `gpu_device=`, `inference_batch_size=`) are applied after the
load and still win, so `jammi.connect("file://…")`'s directory is unaffected.

What `run_worker = false` buys, and its SQLite consequence, is the whole point
of the two-process test below: this process still mounts the training surface
and still accepts submissions, but never claims, so on a single-process catalog
(SQLite, `unix-excl`) a submitted job stays `queued` until this process
`close()`s the directory and a claiming process opens it.

Hermetic: the local `tiny_bert` model fixture + a tiny `training_pairs.csv`, on
CPU, into a temp artifact dir. No network, no GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
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

# The submission-time acceleration-report marker `Catalog::create_training_job`
# stamps: "no claimant has computed a determination YET" (esc-075). A job no
# worker ever claimed must read exactly this, byte for byte, for its whole life.
_PENDING = {"state": "pending"}

# The default idle poll (`TrainingConfig::idle_poll_secs`) is 1 second: a worker
# with nothing to do sleeps this long between claim attempts. Every
# "still queued" assertion below has to span comfortably MORE than one such
# window, or it would pass against a worker that simply had not ticked yet.
_IDLE_POLL_SECONDS = 1.0
_POLL_ROUNDS = 6
_POLL_INTERVAL_SECONDS = 0.5


def _submit(db) -> object:
    """Submit the one fine-tune job every test here shares. A REAL job with a
    real, loadable base model: with a claiming worker present it completes in
    seconds (the control below proves exactly that), so "still queued" is
    evidence about the claim loop and never about an unrunnable job."""
    return db.fine_tune(
        source="training",
        base_model=f"local:{_TINY_BERT}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=1,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )


def _connect_with_source(tmp_path: Path):
    db = jammi.connect(f"file://{tmp_path}")
    db.add_source("training", url=str(_TRAINING_PAIRS), format="csv")
    return db


def test_embedded_run_worker_false_accepts_the_submission_and_never_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`JAMMI_TRAINING__RUN_WORKER=false` + `jammi.connect("file://…")`: the
    submission is accepted and the job stays `queued` forever.

    The env var is read by `JammiConfig::load` at open time — the same loader,
    with the same precedence, that `jammi-server` runs. Two things are asserted
    on every poll, and they are the two an operator actually relies on:

      * `status()` is `"queued"` — no claim happened, so no lease, no run;
      * `acceleration_report()` is byte-exactly ``{"state": "pending"}`` — the
        esc-075 tri-state marker still means "no claimant has computed a
        determination yet", never silently retired to `undetermined` (which is
        what a claim-then-fail path would leave) and never `None`.

    The polls span >> the 1 s default idle poll, so a worker that merely had
    not ticked yet could not produce this trace.
    """
    monkeypatch.setenv(_RUN_WORKER_ENV, "false")

    db = _connect_with_source(tmp_path)
    try:
        job = _submit(db)
        # Accepted: the submission surface is untouched by `run_worker` — the
        # job exists, with its deterministic output model id.
        assert job.job_id
        assert job.model_id.startswith("jammi:fine-tuned:")

        started = time.monotonic()
        for poll in range(_POLL_ROUNDS):
            assert job.status() == "queued", (
                f"poll {poll}: a `run_worker = false` process claimed a job it "
                f"must never claim (status {job.status()!r})"
            )
            assert job.acceleration_report() == _PENDING, (
                f"poll {poll}: the submission-time marker was overwritten — "
                f"only a claimant writes this column: {job.acceleration_report()!r}"
            )
            time.sleep(_POLL_INTERVAL_SECONDS)
        elapsed = time.monotonic() - started
        assert elapsed > _IDLE_POLL_SECONDS * 2, (
            f"the trace spans only {elapsed:.2f}s — too short to outlive a "
            f"{_IDLE_POLL_SECONDS}s idle poll, so it proves nothing"
        )
    finally:
        db.close()


def test_embedded_default_config_claims_and_completes_the_same_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: with `run_worker` unset (the default, `true`) the very same
    submission LEAVES `queued` and runs to completion.

    Without this the "still queued" test above would be satisfied by any
    breakage — a job that cannot run, a fixture that cannot load, a submission
    that never committed. Here the identical job, on the identical fixtures,
    reaches `completed`, so the only difference between the two traces is the
    configuration key under test.
    """
    monkeypatch.delenv(_RUN_WORKER_ENV, raising=False)

    db = _connect_with_source(tmp_path)
    try:
        job = _submit(db)
        job.wait()
        assert job.status() == "completed"
        # And the claimant DID record its determination, so the `pending`
        # marker above is the absence of exactly this write.
        report = job.acceleration_report()
        assert report["state"] == "determined", report
    finally:
        db.close()


# A successor process that opens the catalog directory with the DEFAULT config
# (no `JAMMI_TRAINING__RUN_WORKER` in its environment — the parent strips it),
# attaches to the queued job by id and waits for it. It opens the catalog with
# `jammi_native.open_local` — the legitimate low-level entry, and the very call
# whose config resolution this file is about — rather than through the
# `jammi.EmbeddedBackend` wrapper, which does offer the same `training_job`
# attach convenience but would only sit between the test and the arm under test.
_CLAIMING_SUCCESSOR = textwrap.dedent(
    """
    import os
    import sys

    import jammi_native

    assert "JAMMI_TRAINING__RUN_WORKER" not in os.environ, "parent leaked the knob"

    db = jammi_native.open_local(artifact_dir=sys.argv[1])
    try:
        job = db.training_job(sys.argv[2])
        job.wait()
        print("STATUS " + job.status())
        print("REPORT_STATE " + str(job.acceleration_report()["state"]))
    finally:
        db.close()
    """
)


def test_close_hands_the_queued_job_to_a_claiming_successor_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SQLite handoff story, end to end: a `run_worker = false` process
    submits, holds the queued job, `close()`s — and a successor process running
    the default (claiming) config picks the job up and completes it.

    This is the deployment shape the `run_worker` doc describes for a
    single-process catalog, proven rather than asserted: submission and
    claiming split across two processes, serialised by the `unix-excl`
    single-process contract that `close()` is the release point of. The
    successor is a real second OS process — the only witness that cannot be
    faked by an in-process reopen — and it inherits this test's environment
    MINUS the knob, so what changes between the two is the configuration alone.
    """
    monkeypatch.setenv(_RUN_WORKER_ENV, "false")

    db = _connect_with_source(tmp_path)
    job = _submit(db)
    job_id = job.job_id

    time.sleep(_IDLE_POLL_SECONDS * 2)
    assert job.status() == "queued"
    assert job.acceleration_report() == _PENDING

    # The release point: no worker was ever started here, so `close()` has none
    # to join — and it must still hand the catalog file over.
    db.close()

    successor_env = {k: v for k, v in os.environ.items() if k != _RUN_WORKER_ENV}
    proc = subprocess.run(
        [sys.executable, "-c", _CLAIMING_SUCCESSOR, str(tmp_path), job_id],
        capture_output=True,
        text=True,
        timeout=600,
        env=successor_env,
    )
    assert proc.returncode == 0, (
        "the claiming successor must open the released directory and run the "
        f"queued job: {proc.stdout}{proc.stderr}"
    )
    assert "STATUS completed" in proc.stdout, proc.stdout
    # The successor is the claimant, so it is the one that wrote the
    # determination the submitter's process never could.
    assert "REPORT_STATE determined" in proc.stdout, proc.stdout


def test_queued_job_reads_identically_on_the_remote_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Value parity: the `queued` / ``{"state": "pending"}`` pair the embedded
    arm returns for an unclaimed job is byte-identical to what the remote arm
    returns for the same catalog row.

    Driven hermetically through a stub double (the idiom
    `test_conformance.py::test_remote_rpc_status_errors_map_onto_the_taxonomy`
    uses): the server fills `TrainingStatusResponse.status` and
    `acceleration_report_json` VERBATIM from the same two catalog columns the
    embedded read decodes (`training.proto` states the verbatim contract for
    the JSON field), so pinning the remote client's decode of those exact
    column values against the embedded arm's live read is the parity claim
    minus one hop.

    **Where the remaining hop is covered.** The one thing this hermetic test
    cannot show is a live `jammi-server` started with
    `JAMMI_TRAINING__RUN_WORKER=false` actually holding a job `queued`. That is
    the server leg of this campaign, and it is proven where a server can be
    run: `clients/python/tests/test_remote_training_job_live.py` starts one with
    exactly that environment and compares its reads of the seeded row against
    the embedded arm's, and the engine's own `grpc_training` it-suite polls the
    `queued` / `{"state":"pending"}` pair across the idle window. This test
    covers what the Python surface owns without a server — that both clients
    turn the same two column values into the same two Python values.
    """
    from jammi._database import RemoteTrainingJob
    from jammi._generated.jammi.v1 import training_pb2

    monkeypatch.setenv(_RUN_WORKER_ENV, "false")

    db = _connect_with_source(tmp_path)
    try:
        job = _submit(db)
        time.sleep(_IDLE_POLL_SECONDS * 2)
        embedded_status = job.status()
        embedded_report = job.acceleration_report()
    finally:
        db.close()

    class _QueuedStub:
        """A `TrainingServiceStub` double serving the row the engine actually
        committed: the columns are carried across verbatim, never re-encoded."""

        def TrainingStatus(self, _request, metadata=None):  # noqa: N802 - gRPC name
            return training_pb2.TrainingStatusResponse(
                status=embedded_status,
                model_id=job.model_id,
                acceleration_report_json='{"state":"pending"}',
            )

    remote = RemoteTrainingJob(
        _QueuedStub(), None, job_id=job.job_id, model_id=job.model_id
    )

    assert embedded_status == "queued"
    assert embedded_report == _PENDING
    assert remote.status() == embedded_status
    assert remote.acceleration_report() == embedded_report
