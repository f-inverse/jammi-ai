"""Hermetic tests for `_job_result_to_dict` (`jammi/_database.py`) — the
projection `RemoteJob.wait()` returns.

`ModelResult.cache_outcome` (`crates/jammi-wire/proto/
jammi/v1/job.proto`) is a training kind's peer of `TableResult.cache_outcome`,
which this projection already carried. The embedded `Job.wait()`
(`crates/jammi-python/src/job.rs`) reaches the SAME key for free — it is a
generic `serde_json` projection of the engine's own `jammi_ai::jobs::JobResult`,
so a field added to that Rust enum reaches the embedded dict with no code
change there. The REMOTE projection tested here is hand-written
(`_job_result_to_dict` decodes the wire message's named fields one at a time),
so it does NOT gain a new key for free: this test is the one that would have
caught the field silently missing from the `model` arm, which `RemoteJob.wait()`
would otherwise have returned with three keys where the embedded transport now
returns four (a K4 parity break invisible to any test that never inspects the
dict's key set).

No channel is dialed: `_job_result_to_dict` is a free function over a
hand-built `job_pb2.JobStatusResponse`.
"""

from __future__ import annotations

from jammi._database import _job_result_to_dict
from jammi._generated.jammi.v1 import job_pb2


def test_model_arm_carries_cache_outcome_computed() -> None:
    resp = job_pb2.JobStatusResponse(
        status="completed",
        kind="fine_tune",
        model=job_pb2.ModelResult(
            model_id="jammi:fine-tuned:abc",
            artifact_path="file:///artifacts/abc",
            cache_outcome="computed",
        ),
    )
    result = _job_result_to_dict(resp)
    assert result == {
        "kind": "model",
        "model_id": "jammi:fine-tuned:abc",
        "artifact_path": "file:///artifacts/abc",
        "metrics": None,
        "cache_outcome": "computed",
    }


def test_model_arm_carries_cache_outcome_reused() -> None:
    """A `FineTune` model-level cache hit's own wire result names the reused
    model — the SAME vocabulary `table`'s `cache_outcome` already carries."""
    resp = job_pb2.JobStatusResponse(
        status="completed",
        kind="fine_tune",
        model=job_pb2.ModelResult(
            model_id="jammi:fine-tuned:second",
            artifact_path="file:///artifacts/first",
            cache_outcome="reused:jammi:fine-tuned:first",
        ),
    )
    result = _job_result_to_dict(resp)
    assert result["cache_outcome"] == "reused:jammi:fine-tuned:first"


def test_table_arm_cache_outcome_is_unaffected() -> None:
    """The pre-existing `table` arm's projection is untouched by this unit."""
    resp = job_pb2.JobStatusResponse(
        status="completed",
        kind="embedding",
        table=job_pb2.TableResult(table="jammi.embeddings_1", cache_outcome="computed"),
    )
    result = _job_result_to_dict(resp)
    assert result == {
        "kind": "table",
        "table": "jammi.embeddings_1",
        "cache_outcome": "computed",
    }
