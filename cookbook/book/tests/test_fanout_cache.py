"""Cache-backed checks for the fan-out chapter: one embedding plan, run at
partitions 1, 2 and 4, wrote the same artifact under the same definition.

These run on CPU against the committed matrix (no engine re-execution). The
cache is committed, so an absent artifact is a failure naming it.
"""

from __future__ import annotations

from jammi_cookbook import contracts


def _matrix() -> dict:
    return contracts.load_artifact("fanout.matrix")


def test_every_verdict_matches_golden():
    """Every fan-out wrote the reference run's artifact, and every run shares
    one definition: the partition count is not a determinant."""
    for metric in ("p2_artifact_equal", "p4_artifact_equal", "definition_equal_all"):
        contracts.assert_close(f"fanout.{metric}", 1.0)


def test_the_runs_are_the_ones_the_verdicts_read():
    """The matrix carries each run's own artifact and definition, so the
    verdicts are checked against the recorded values, not only asserted."""
    runs = _matrix()["runs"]
    assert [run["partitions"] for run in runs] == [1, 2, 4]
    assert len({str(run["artifact"]) for run in runs}) == 1
    assert len({run["definition_hash"] for run in runs}) == 1


def test_every_fan_out_forwarded_several_chunks():
    """With the recorded batch size, the corpus is cut into more chunks than
    the widest fan-out has partitions, so every partition forwarded rows."""
    matrix = _matrix()
    chunks = -(-matrix["rows"] // matrix["batch_size"])
    assert chunks > max(run["partitions"] for run in matrix["runs"])
