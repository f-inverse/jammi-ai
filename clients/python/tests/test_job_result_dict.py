"""Hermetic tests for `_job_result_to_dict` (`jammi/_database.py`) — the
projection `RemoteJob.wait()` returns — and the one cache-outcome shape both
transports agree on.

`ModelResult.cache_outcome` and `TableResult.cache_outcome`
(`crates/jammi-wire/proto/jammi/v1/job.proto`) are the same
`jammi.v1.inference.CacheOutcome` message every producer response carries.
The embedded `Job.wait()` (`crates/jammi-python/src/job.rs`) returns the
generic `serde_json` projection of the engine's own `jammi_ai::jobs::JobResult`,
whose `cache_outcome` serialises to `{"outcome": "computed"}` /
`{"outcome": "reused", "reused": {"table" | "model": …}}`; the REMOTE
projection tested here is hand-written (`_job_result_to_dict` decodes the
wire message's fields one at a time through `cache_outcome_to_dict`), so
this is the test that keeps the two transports' dicts identical.

No channel is dialed: `_job_result_to_dict` is a free function over a
hand-built `job_pb2.JobStatusResponse`.
"""

from __future__ import annotations

import pytest

from jammi._assembly import cache_outcome_to_dict
from jammi._database import _job_result_to_dict
from jammi._generated.jammi.v1 import inference_pb2, job_pb2
from jammi.errors import BackendError


def _computed() -> inference_pb2.CacheOutcome:
    return inference_pb2.CacheOutcome(computed=inference_pb2.CacheOutcome.Computed())


def test_model_arm_carries_cache_outcome_computed() -> None:
    resp = job_pb2.JobStatusResponse(
        status="completed",
        kind="fine_tune",
        model=job_pb2.ModelResult(
            model_id="jammi:fine-tuned:abc",
            artifact_path="file:///artifacts/abc",
            cache_outcome=_computed(),
        ),
    )
    result = _job_result_to_dict(resp)
    assert result == {
        "kind": "model",
        "model_id": "jammi:fine-tuned:abc",
        "artifact_path": "file:///artifacts/abc",
        "metrics": None,
        "cache_outcome": {"outcome": "computed"},
    }


def test_model_arm_carries_the_reused_model_artifact() -> None:
    """A model-level reuse names the artifact the job's row shares — the
    same one `artifact_path` names."""
    resp = job_pb2.JobStatusResponse(
        status="completed",
        kind="fine_tune",
        model=job_pb2.ModelResult(
            model_id="jammi:fine-tuned:second",
            artifact_path="file:///artifacts/first",
            cache_outcome=inference_pb2.CacheOutcome(
                reused_model_artifact="file:///artifacts/first"
            ),
        ),
    )
    result = _job_result_to_dict(resp)
    assert result["cache_outcome"] == {
        "outcome": "reused",
        "reused": {"model": "file:///artifacts/first"},
    }


def test_table_arm_carries_the_reused_table() -> None:
    computed = job_pb2.JobStatusResponse(
        status="completed",
        kind="embedding",
        table=job_pb2.TableResult(table="jammi.embeddings_1", cache_outcome=_computed()),
    )
    assert _job_result_to_dict(computed) == {
        "kind": "table",
        "table": "jammi.embeddings_1",
        "cache_outcome": {"outcome": "computed"},
    }
    reused = job_pb2.JobStatusResponse(
        status="completed",
        kind="embedding",
        table=job_pb2.TableResult(
            table="jammi.embeddings_1",
            cache_outcome=inference_pb2.CacheOutcome(reused_table="jammi.embeddings_1"),
        ),
    )
    assert _job_result_to_dict(reused)["cache_outcome"] == {
        "outcome": "reused",
        "reused": {"table": "jammi.embeddings_1"},
    }


def test_an_outcome_with_no_arm_is_a_wire_fault_never_computed() -> None:
    with pytest.raises(BackendError):
        cache_outcome_to_dict(inference_pb2.CacheOutcome())
