"""`close(release=True)` is the embedded engine's RELEASE mode (#482).

`close()` (the default, DRAIN) lets the in-flight training job finish before
the catalog is released — `test_close_releases_catalog.py` and
`test_run_worker_config.py` pin that arm. `close(release=True)` hands every job
lease this process holds back to the catalog at once: the in-flight job's row
stays `running` with a NULL lease and `releases = 1`, so a successor process
claims it within one idle poll (never one lease window) and the job costs no
attempt (`attempts - releases` is what the reclaim cap counts). The call
returns promptly — bounded by two lease heartbeats, never by the run.

The row is read AFTER `close()` through CPython's own `sqlite3`, the sanctioned
foreign-reader pattern of `test_close_releases_catalog.py` (the engine has
let go of the file by then, so both libraries agree on the image). The
successor is a separate process: it opens the directory with the default
config, lets its own worker claim the released row, then releases again — the
row then reads `attempts 2, releases 2` under a new `claimed_by`.

Hermetic: the local `tiny_bert` fixture + `training_pairs.csv`, on CPU.
"""

from __future__ import annotations

import inspect
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import jammi

_ROOT = Path(__file__).resolve().parents[3]
_TINY_BERT = _ROOT / "cookbook" / "fixtures" / "tiny_bert"
_TRAINING_PAIRS = _ROOT / "tests" / "fixtures" / "training_pairs.csv"

pytestmark = pytest.mark.skipif(
    not _TINY_BERT.is_dir() or not _TRAINING_PAIRS.is_file(),
    reason="local tiny_bert / training_pairs fixtures not present",
)

# The successor: opens the released directory with the DEFAULT config (a
# claiming worker, 1 s idle poll), waits for ITS OWN worker to actually claim
# the released row, then RELEASES it again and exits. It reports nothing —
# the parent reads the row it leaves behind.
#
# `job.status()` cannot tell "claimed by the original instance" from "claimed
# by this successor" -- a release never changes `status` away from
# `"running"` (D9/D11: no status is invented for a release), and the row can
# also reach `"completed"` on its own within a few seconds once tiny-bert's
# early stopping converges the run, well inside any fixed sleep bound. The
# reliable, mechanism-grounded signal is `acceleration_report`
# (`crates/jammi-python/src/job.rs`'s `Job.acceleration_report()`): a
# released row is reclaimed by `reclaim_expired_jobs`' arm 1a, which resets
# `acceleration_report` to the `"pending"` marker for the NEW attempt before
# `claim_next` claims it (`crates/jammi-db/src/catalog/jobs_repo.rs`, the
# comment beside `ACCELERATION_REPORT_PENDING` in arm 1a). R7: the reliable
# key is the successor's OWN attempt number, read absolutely rather than
# differentially against a snapshot — `record_acceleration_report`'s guard
# pins `attempts = $6` (`jobs_repo.rs`), so once THIS successor's reclaim
# bumps `attempts` a differing (predecessor's, or the pending marker's own)
# write matches zero rows, and the successor's own payload carries its own
# `"attempt"` (`build_acceleration_report_json`, `worker.rs`). A snapshot
# taken before the predecessor's own probe landed -- the prior key -- could
# be satisfied by that late write instead of this successor's, since the
# pending marker (`ACCELERATION_REPORT_PENDING`) carries no `attempt` field
# to compare against at all.
_RELEASING_SUCCESSOR = textwrap.dedent(
    """
    import sys
    import time

    import jammi_native

    artifact_dir, job_id, expected_attempt = sys.argv[1], sys.argv[2], int(sys.argv[3])

    db = jammi_native.open_local(artifact_dir=artifact_dir)
    try:
        job = db.job(job_id)
        deadline = time.monotonic() + 60
        report = None
        while time.monotonic() < deadline:
            report = job.acceleration_report()
            if (report or {}).get("attempt") == expected_attempt:
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                f"this worker never claimed and re-probed the released job at its "
                f"own attempt {expected_attempt} (acceleration_report stayed {report!r})"
            )
    finally:
        db.close(release=True)
    """
)


def _row(catalog_db: Path, job_id: str) -> tuple:
    conn = sqlite3.connect(str(catalog_db))
    try:
        return conn.execute(
            "SELECT status, claimed_by, lease_expires_at IS NULL, releases, attempts "
            "FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    finally:
        conn.close()


def _wait_running(job, timeout_secs: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        status = job.status()
        if status == "running":
            return
        assert status == "queued", f"the long job must not be terminal yet: {status}"
        time.sleep(0.1)
    raise AssertionError("the job was never claimed")


def test_close_release_true_leaves_the_job_claimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JAMMI_WORKER__ENABLED", raising=False)
    db = jammi.connect(f"file://{tmp_path}")
    db.add_source("training", url=str(_TRAINING_PAIRS), format="csv")
    job = db.fine_tune(
        source="training",
        base_model=f"local:{_TINY_BERT}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        # Large enough that the run is certainly still training at the
        # release; a tiny real epoch is single-digit milliseconds.
        epochs=20_000,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )
    job_id = job.job_id
    _wait_running(job)

    started = time.monotonic()
    db.close(release=True)
    took = time.monotonic() - started
    assert took < 30, f"release must return within two heartbeats, took {took:.1f}s"

    status, claimed_by, lease_null, releases, attempts = _row(
        tmp_path / "catalog.db", job_id
    )
    assert status == "running", "no status is invented for a release"
    assert claimed_by, "the releasing instance stays on the row"
    assert lease_null == 1, "the lease is NULL: reclaimable at once"
    assert (releases, attempts) == (1, 1)

    # Idempotent, and every verb afterwards raises the typed error.
    db.close(release=True)
    db.close()

    # A successor process claims it within one idle poll and releases again;
    # `attempts + 1` (R7) is the expected attempt number ITS OWN claim (and
    # the acceleration probe it runs under that claim) will carry.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _RELEASING_SUCCESSOR,
            str(tmp_path),
            job_id,
            str(attempts + 1),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    status2, claimed_by2, lease_null2, releases2, attempts2 = _row(
        tmp_path / "catalog.db", job_id
    )
    assert status2 == "running", "still never failed"
    assert claimed_by2 != claimed_by, "a NEW instance claimed the released row"
    assert lease_null2 == 1
    assert (releases2, attempts2) == (2, 2), "attempts - releases stays 0: no cap consumed"


def test_close_default_is_unchanged_and_the_remote_arm_accepts_the_flag() -> None:
    """`close()`'s default is still DRAIN (the flag defaults to False on every
    surface), and the remote arm carries the same parameter (accepted and
    ignored — the leases live in the server process)."""
    for cls in (jammi.EmbeddedBackend, jammi.RemoteDatabase):
        param = inspect.signature(cls.close).parameters["release"]
        assert param.default is False, cls
