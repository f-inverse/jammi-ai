"""`close()` is the embedded engine's catalog RELEASE point — natively and publicly.

The SQLite catalog opens through SQLite's `unix-excl` VFS (esc-073,
`jammi_db::catalog::backend_sqlite`): a *process-scoped* exclusive lock is held
for as long as ANY connection in this process has `catalog.db` open, and the
WAL index lives in heap memory with no `-shm` file on disk. Two consequences
this module pins from Python, because both are load-bearing for every consumer
that hands the catalog directory to somebody else:

  1. While the engine holds the file, a SECOND PROCESS opening the same
     directory is refused with a typed `jammi.errors.BackendError` naming the
     single-process contract — never a hang, never a signal.
  2. While the engine holds the file, a FOREIGN SQLite LIBRARY INSTANCE inside
     THIS process (CPython's `sqlite3` alongside the extension's statically
     bundled amalgamation) sees a divergent image of the database: the engine's
     wal-index is in heap memory that no other library instance can read. That
     topology is out of contract; it becomes *harmless* only once the engine
     has actually let go.

"Actually let go" is an AWAITED event, not a drop: `sqlx` returns a connection
to the pool from a background task, so dropping the handle releases nothing at
any bounded moment. `close()` is the one bounded release point — it stops this
connection's embedded training worker and then closes the catalog pool, waiting
for SQLite's own evidence of release (`catalog.db-wal` gone).

Every test here deliberately keeps the Python handle ALIVE across the
assertions, so what is proven is `close()` itself and never an incidental drop.

The module is in two halves. The first drives the low-level
`jammi_native._NativeDatabase` handle: it owns the mechanism (the 20-cycle race
argument, the `-wal` evidence, the FFI-boundary use-after-close guard). The
second drives the SHIPPED PUBLIC surface — `jammi.connect("file://…")`, which is
what a user of the client actually has — because a documented release contract
that the public wrapper does not forward is not a contract anybody can use. Both
halves are needed: a wrapper that dropped the call on the floor is invisible to
every low-level oracle.

The two pre-existing close-before-inject tests
(`test_conformance.py::test_remote_and_embedded_training_job_metrics_agree_on_all_three_states`
and `test_embedded_training.py::test_embedded_training_job_acceleration_report_covers_all_four_states`)
are the only sanctioned raw-`sqlite3` touch points on an engine catalog, and
they depend on exactly this contract.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import jammi
import jammi_native
from jammi._assembly import build_fine_tune_request
from jammi.errors import BackendError

# A base model directory that does not exist. `create_training_job` runs at
# SUBMIT time and reads nothing but the catalog, so this commits a real
# training-job row in milliseconds — no model load, no fixture, no training.
# The worker then fails the job; the columns asserted below (`job_id`,
# `training_source`) are the ones no later write touches.
_ABSENT_MODEL = "/nonexistent/jammi/close-release-oracle/model"

_CYCLES = 20

# A child process that tries to open the same catalog directory and reports,
# on one line, either the typed refusal (with its fully-qualified class) or a
# successful open. `sys.exit(3)` distinguishes "refused" from an interpreter
# failure to even import the extension.
_SUCCESSOR = textwrap.dedent(
    """
    import sys

    import jammi_native

    try:
        db = jammi_native.open_local(artifact_dir=sys.argv[1])
    except BaseException as exc:                     # noqa: BLE001 - reporting
        kind = type(exc).__module__ + "." + type(exc).__name__
        print("REFUSED " + kind + " " + " ".join(str(exc).split()))
        sys.exit(3)
    print("OPENED")
    sys.exit(0)
    """
)


def _submit_job_row(db, training_source: str) -> str:
    """Commit one real `training_jobs` row through the engine and return its id."""
    request = build_fine_tune_request(
        source=training_source,
        base_model=f"local:{_ABSENT_MODEL}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=1,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )
    job = db._start_training_proto(request.SerializeToString())
    job_id = job.job_id
    del job
    return job_id


def _raw_training_jobs(catalog_db: Path) -> list[tuple[str, str]]:
    """`(job_id, training_source)` for every row, read through CPython's OWN
    SQLite library instance — the foreign reader whose view must agree with the
    engine's once the engine has released the file."""
    conn = sqlite3.connect(str(catalog_db))
    try:
        return [
            (row[0], row[1])
            for row in conn.execute(
                "SELECT job_id, training_source FROM training_jobs ORDER BY rowid"
            )
        ]
    finally:
        conn.close()


def _run_successor(artifact_dir: Path) -> tuple[subprocess.CompletedProcess, float]:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _SUCCESSOR, str(artifact_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc, time.monotonic() - started


def test_close_releases_the_catalog_to_a_raw_sqlite3_reader(tmp_path):
    """After `close()`, a raw `sqlite3` connection sees the engine's last
    committed training-job row — every cycle, 20 cycles running.

    Each cycle opens a fresh engine on the SAME directory, commits one more
    training-job row, `close()`s, and then reads the table through CPython's
    stdlib `sqlite3`. Two things are asserted per cycle:

      * `catalog.db-wal` is gone the instant `close()` returns. That sidecar is
        deleted by the LAST `sqlite3_close` in this process, so its absence is
        SQLite's own evidence that the engine let go — the release point the
        `unix-excl` seam makes load-bearing. The handle is still alive here, so
        this proves `close()`, not the drop.
      * the foreign reader's view of `training_jobs` is byte-equal to the set of
        rows the engine committed (each id as the engine minted it, in
        insertion order) — not merely "returns something". While an engine
        connection is open, the two library instances are deterministically
        divergent, so a `close()` that left the pool open is visible here as a
        stale or short row set.

    Twenty cycles because the divergence is a race in the wrong direction: a
    single cycle can pass by luck, and the repeated open/close handoff on one
    accumulating catalog is the shape a consumer actually runs.
    """
    catalog_db = tmp_path / "catalog.db"
    wal = tmp_path / "catalog.db-wal"
    expected: list[tuple[str, str]] = []

    for cycle in range(_CYCLES):
        db = jammi_native.open_local(artifact_dir=str(tmp_path))
        source = f"close_release_cycle_{cycle}"
        expected.append((_submit_job_row(db, source), source))

        db.close()

        assert not wal.exists(), (
            f"cycle {cycle}: `close()` returned with {wal.name} still present — "
            "the engine has not released the catalog file, so a foreign SQLite "
            "library instance reads a divergent image of it"
        )

        observed = _raw_training_jobs(catalog_db)
        assert observed == expected, (
            f"cycle {cycle}: the raw sqlite3 reader disagrees with the engine's "
            f"committed rows after close() — expected {expected}, saw {observed}"
        )

        del db


def test_close_hands_the_catalog_directory_to_a_successor_process(tmp_path):
    """A second PROCESS is refused with the typed error while this engine holds
    the catalog, and opens immediately once `close()` has returned.

    That refusal IS the single-process contract, mechanically enforced: the
    `unix-excl` VFS holds a process-scoped exclusive lock, so the successor
    waits out the 5 s busy timeout and is then refused with a
    `jammi.errors.BackendError` naming the contract — never a hang, never a
    signal, never a silently shared file.

    The handle is deliberately still alive when the second successor runs, so
    what hands the directory over is `close()` and nothing else. "Immediately"
    is measured against a contention-free baseline open on a fresh directory:
    an open that had to wait out the busy timeout would exceed it by the whole
    5 s window.
    """
    db = jammi_native.open_local(artifact_dir=str(tmp_path))
    job_id = _submit_job_row(db, "successor_handoff")

    baseline, baseline_secs = _run_successor(tmp_path / "uncontended")
    assert baseline.returncode == 0, (
        "the successor must open an UNCONTENDED directory cleanly, or this "
        f"test measures a broken child, not the seam: {baseline.stdout}{baseline.stderr}"
    )
    assert "OPENED" in baseline.stdout

    refused, refused_secs = _run_successor(tmp_path)
    assert refused.returncode == 3, (
        "a second process must be REFUSED while this engine holds the catalog, "
        f"got returncode {refused.returncode}: {refused.stdout}{refused.stderr}"
    )
    assert "REFUSED jammi.errors.BackendError" in refused.stdout, (
        "the refusal must be the typed taxonomy error, not a bare exception: "
        f"{refused.stdout}"
    )
    assert "single-process only" in refused.stdout, (
        f"the refusal must name the contract it enforces: {refused.stdout}"
    )

    # The release point. The handle stays alive; only `close()` runs.
    db.close()

    successor, successor_secs = _run_successor(tmp_path)
    assert successor.returncode == 0, (
        "after close() the catalog directory must be free for a successor "
        f"process: {successor.stdout}{successor.stderr}"
    )
    assert "OPENED" in successor.stdout
    assert successor_secs < baseline_secs + 5.0, (
        f"the successor open took {successor_secs:.2f}s against a "
        f"{baseline_secs:.2f}s uncontended baseline — it waited out the busy "
        f"timeout instead of finding the file free (refused arm: {refused_secs:.2f}s)"
    )

    # The row the first engine committed survives the handoff: releasing the
    # file is a clean shutdown, not an abandonment.
    assert [j for j, _ in _raw_training_jobs(tmp_path / "catalog.db")] == [job_id]


def test_use_after_close_raises_the_typed_error_and_close_is_idempotent(tmp_path):
    """Every verb on a closed handle raises `jammi.errors.BackendError`, and
    `close()` is idempotent.

    The FFI-boundary guard: once the worker is stopped and the pool closed,
    "keep using the handle" must be a prompt typed failure at the call site,
    never a silent no-op, a panic across the boundary, or a silently reopened
    pool (which would re-take the exclusive lock the caller was just told they
    could have).
    """
    db = jammi_native.open_local(artifact_dir=str(tmp_path))
    assert db.list_sources() == []

    db.close()
    db.close()  # idempotent: a second close is a no-op, not an error
    db.close()

    with pytest.raises(BackendError, match="closed"):
        db.list_sources()
    with pytest.raises(BackendError, match="closed"):
        db.training_job("no-such-job")
    with pytest.raises(BackendError, match="closed"):
        db.get_server_info()
    with pytest.raises(BackendError, match="closed"):
        db.tenant()

    # And the file really is free: a fresh handle on the same directory opens
    # in this process, so the closed handle is inert rather than wedged.
    fresh = jammi_native.open_local(artifact_dir=str(tmp_path))
    assert fresh.list_sources() == []
    fresh.close()


# ---------------------------------------------------------------------------
# The PUBLIC surface. Everything above drives the low-level `jammi_native`
# handle; a user of the shipped client never touches that. They write
# `jammi.connect("file://…")` and get a `jammi.EmbeddedBackend`. The release
# contract the engine documents ("Releasing the file is an awaited event, not a
# drop" — `jammi_db::catalog::backend_sqlite`, whose single-process seam
# `docs/guide/src/catalog-and-broker.md` states for operators) is only
# real if it is REACHABLE from there, so it is pinned there too — separately,
# because a wrapper that dropped the call (or raised
# `NotSupportedOnBackend`, as it did before this change) is invisible to every
# oracle above.
# ---------------------------------------------------------------------------

# The public-surface peer of `_SUCCESSOR`: the child opens through
# `jammi.connect`, the front door, not `jammi_native.open_local`.
_PUBLIC_SUCCESSOR = textwrap.dedent(
    """
    import sys

    import jammi

    try:
        db = jammi.connect("file://" + sys.argv[1])
    except BaseException as exc:                     # noqa: BLE001 - reporting
        kind = type(exc).__module__ + "." + type(exc).__name__
        print("REFUSED " + kind + " " + " ".join(str(exc).split()))
        sys.exit(3)
    print("OPENED")
    sys.exit(0)
    """
)


def _run_public_successor(artifact_dir: Path) -> tuple[subprocess.CompletedProcess, float]:
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", _PUBLIC_SUCCESSOR, str(artifact_dir)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc, time.monotonic() - started


def _public_submit_job_row(db, training_source: str) -> str:
    """The public-surface peer of `_submit_job_row`: submit through
    `jammi.EmbeddedBackend.fine_tune`, the verb a caller actually writes."""
    job = db.fine_tune(
        source=training_source,
        base_model=f"local:{_ABSENT_MODEL}",
        columns=["text_a", "text_b", "score"],
        method="lora",
        task="text_embedding",
        epochs=1,
        batch_size=8,
        lora_rank=4,
        warmup_steps=0,
    )
    job_id = job.job_id
    del job
    return job_id


def test_public_close_hands_the_catalog_directory_to_a_successor_process(tmp_path):
    """`jammi.connect("file://…").close()` — the PUBLIC front door — hands the
    catalog directory to a successor process.

    The same handoff `test_close_hands_the_catalog_directory_to_a_successor_process`
    proves on the low-level handle, driven end-to-end through the shipped
    client: a second process is refused with the typed
    `jammi.errors.BackendError` while this session holds the file, and opens
    immediately once `close()` has returned. Both the holder and the successor
    speak `jammi.connect`, so what is pinned is the surface a user has, not an
    internal one they never see.

    Before this change the assertion could not even be attempted: the wrapper's
    `close()` raised `NotSupportedOnBackend`, so the engine's documented release
    contract was unreachable from the public API and an embedded session held
    the catalog for the life of the process.

    The session object is deliberately still alive at every assertion, so the
    handover is `close()` and never an incidental drop.
    """
    db = jammi.connect(f"file://{tmp_path}")
    assert isinstance(db, jammi.EmbeddedBackend)
    job_id = _public_submit_job_row(db, "public_successor_handoff")

    baseline, baseline_secs = _run_public_successor(tmp_path / "uncontended")
    assert baseline.returncode == 0, (
        "the successor must open an UNCONTENDED directory cleanly, or this "
        f"test measures a broken child, not the seam: {baseline.stdout}{baseline.stderr}"
    )
    assert "OPENED" in baseline.stdout

    refused, _ = _run_public_successor(tmp_path)
    assert refused.returncode == 3, (
        "a second process must be REFUSED while this session holds the catalog, "
        f"got returncode {refused.returncode}: {refused.stdout}{refused.stderr}"
    )
    assert "REFUSED jammi.errors.BackendError" in refused.stdout, refused.stdout
    assert "single-process only" in refused.stdout, refused.stdout

    # The release point, reached through the public verb and nothing else.
    db.close()

    successor, successor_secs = _run_public_successor(tmp_path)
    assert successor.returncode == 0, (
        "after the public close() the catalog directory must be free for a "
        f"successor process: {successor.stdout}{successor.stderr}"
    )
    assert "OPENED" in successor.stdout
    assert successor_secs < baseline_secs + 5.0, (
        f"the successor open took {successor_secs:.2f}s against a "
        f"{baseline_secs:.2f}s uncontended baseline — it waited out the busy "
        "timeout instead of finding the file free"
    )

    # The row this session committed survives the handoff.
    assert [j for j, _ in _raw_training_jobs(tmp_path / "catalog.db")] == [job_id]


def test_public_close_releases_the_catalog_to_a_raw_sqlite3_reader(tmp_path):
    """After the PUBLIC `close()`, `catalog.db-wal` is gone and a raw `sqlite3`
    reader agrees with the engine — the same per-cycle evidence the low-level
    oracle collects, through `jammi.connect`.

    Fewer cycles than the low-level oracle (which owns the 20-cycle race
    argument): what is at stake here is only that the wrapper actually reaches
    the native release, so a handful of open/commit/close rounds on one
    accumulating catalog is the right cost.
    """
    catalog_db = tmp_path / "catalog.db"
    wal = tmp_path / "catalog.db-wal"
    expected: list[tuple[str, str]] = []

    for cycle in range(5):
        db = jammi.connect(f"file://{tmp_path}")
        source = f"public_close_cycle_{cycle}"
        expected.append((_public_submit_job_row(db, source), source))

        db.close()

        assert not wal.exists(), (
            f"cycle {cycle}: the public close() returned with {wal.name} still "
            "present — the wrapper did not reach the engine's release"
        )
        assert _raw_training_jobs(catalog_db) == expected, (
            f"cycle {cycle}: the raw sqlite3 reader disagrees with the rows the "
            "engine committed after the public close()"
        )

        del db


def test_public_context_manager_exit_releases_the_catalog(tmp_path):
    """`with jammi.connect("file://…") as db:` releases the catalog on block
    exit — the embedded arm's `__exit__` closes, exactly as the remote arm's
    does.

    A context manager whose exit released nothing is the same defect in a
    friendlier shape: the block reads like a scoped resource and is not one.
    Pinned with the successor process, the only witness that cannot be faked by
    an in-process reopen.
    """
    with jammi.connect(f"file://{tmp_path}") as db:
        _public_submit_job_row(db, "public_ctx_exit")
        held, _ = _run_public_successor(tmp_path)
        assert held.returncode == 3, (
            "inside the block the catalog must still be held: "
            f"{held.stdout}{held.stderr}"
        )

    released, _ = _run_public_successor(tmp_path)
    assert released.returncode == 0, (
        f"block exit must release the catalog: {released.stdout}{released.stderr}"
    )
    assert "OPENED" in released.stdout
