"""The job queue's vocabulary: a job's status, kind and progress, and a worker's state.

One vocabulary for both transports. The embedded engine hands each value's
spelling across its FFI boundary; the remote client reads the `jammi.v1.job`
wire enums; both decode into these types, so a caller compares
``job.status() is JobStatus.COMPLETED`` the same way on either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Type, TypeVar

from .errors import BackendError


class _Spelled(Enum):
    """An engine vocabulary whose values are the engine's own spellings."""

    def __str__(self) -> str:
        return self.value


class JobStatus(_Spelled):
    """A job's lifecycle status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether the job has ended: completed, failed or cancelled."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class JobKind(_Spelled):
    """What a job runs. A training kind registers a model; a compute kind writes
    a result table."""

    FINE_TUNE = "fine_tune"
    GRAPH_FINE_TUNE = "graph_fine_tune"
    CONTEXT_PREDICTOR = "context_predictor"
    NEIGHBOR_GRAPH = "neighbor_graph"
    PROPAGATE = "propagate"
    GRAPH_STRUCTURE = "graph_structure"
    ASOF_JOIN = "asof_join"
    LEXICAL_INDEX = "lexical_index"
    EMBEDDING = "embedding"
    INFER = "infer"

    @property
    def trains_a_model(self) -> bool:
        """Whether this kind trains a model rather than computing a table."""
        return self in (JobKind.FINE_TUNE, JobKind.GRAPH_FINE_TUNE, JobKind.CONTEXT_PREDICTOR)


class JobPhase(_Spelled):
    """What a running job is doing, and so what its progress counts."""

    ASSEMBLING = "assembling"
    TRAINING = "training"
    COMPUTING = "computing"
    PUBLISHING = "publishing"


class WorkerState(_Spelled):
    """A worker's claim loop state."""

    WARMING = "warming"
    CLAIMING = "claiming"
    DRAINING = "draining"


@dataclass(frozen=True)
class JobProgress:
    """A running job's latest checkpoint: the phase it is in and how far
    through it is — ``done`` of ``total``, in the phase's unit: rows while it
    assembles or computes, optimizer steps while it trains. ``total`` is
    ``None`` when the phase cannot know it up front."""

    phase: JobPhase
    done: Optional[int]
    total: Optional[int]


_E = TypeVar("_E", bound=_Spelled)


def from_wire(cls: Type[_E], wire_enum, value: int) -> _E:
    """Decode a `jammi.v1.job` wire enum value into its vocabulary member.

    The wire names each value ``<ENUM>_<MEMBER>`` (``JOB_STATUS_COMPLETED``),
    and every member here is named after that suffix. ``*_UNSPECIFIED`` is a
    value the server never sends, so receiving it is a backend fault.
    """
    prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", wire_enum.DESCRIPTOR.name).upper() + "_"
    member = wire_enum.Name(value)[len(prefix):]
    if member == "UNSPECIFIED":
        raise BackendError(f"the server sent no {wire_enum.DESCRIPTOR.name}")
    return cls[member]


def progress_from_spelling(
    recorded: Optional[tuple],
) -> Optional[JobProgress]:
    """The embedded engine's `(phase, done, total)`, or `None` before any
    progress is recorded."""
    if recorded is None:
        return None
    phase, done, total = recorded
    return JobProgress(JobPhase(phase), done, total)
