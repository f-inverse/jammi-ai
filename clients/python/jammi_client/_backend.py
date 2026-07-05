"""The `Session` / `Backend` / `TrainingJobHandle` protocols.

`connect(target)` returns a :class:`Session` — the transport-agnostic surface a
caller writes against. Two concrete classes satisfy it structurally: the remote
:class:`~jammi_client.RemoteDatabase` (over a gRPC channel) and the embedded
`jammi_ai.Database` (over the compiled in-process engine). Because both expose
the same verb vocabulary, a caller swaps transports without touching a call —
that is the one-front-door thesis made a type.

The protocols are :func:`~typing.runtime_checkable` so a test can assert
``isinstance(db, Session)`` structurally, and so the two implementations are
pinned to one surface rather than drifting as parallel classes. They enumerate
only the SHARED surface both transports carry; the one-sided, capability-gated
members (`audit`, `ephemeral_session`, `preload_model`, `session_id`) live off
the protocol and are guarded by :meth:`Session.supports` +
:class:`~jammi_client.errors.NotSupportedOnBackend` instead.
"""

from __future__ import annotations

from typing import Any, ContextManager, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from ._capability import Capability


@runtime_checkable
class TrainingJobHandle(Protocol):
    """A handle to a submitted training job — poll it, or block on it.

    Both the embedded engine's `TrainingJob` and the remote
    :class:`~jammi_client.RemoteTrainingJob` satisfy this, so a caller treats the
    two interchangeably: read the ids, poll :meth:`status`, or :meth:`wait` for a
    terminal state (which raises :class:`~jammi_client.errors.TrainingError` on a
    failed job with the worker's message).
    """

    @property
    def job_id(self) -> str:
        """The unique id the submit assigned this job."""

    @property
    def model_id(self) -> str:
        """The deterministic output model id the trained artifact registers under."""

    def status(self) -> str:
        """The job's current status string."""

    def wait(self) -> None:
        """Block until a terminal state; raise on a failed job."""


@runtime_checkable
class Backend(Protocol):
    """The transport a :class:`Session` runs its verbs over.

    A `Session` holds a `Backend`; `connect()` builds the backend for a target
    (a gRPC channel today; an in-process engine handle once the embedded backend
    lands behind this seam) and wraps it. The seam is the transport-level
    lifecycle a session delegates: releasing the underlying resource.
    """

    def close(self) -> None:
        """Release the underlying transport resource. Idempotent."""


@runtime_checkable
class Session(Protocol):
    """The transport-agnostic connection surface `connect(target)` returns.

    The shared verb vocabulary — sources, embeddings, search, inference,
    training, the engine-state pipeline, evaluation, channels, mutable tables,
    topics, the model lifecycle, tenant scope, and SQL — plus the
    capability predicate :meth:`supports` and the lifecycle members. Every member
    here is present on BOTH transports; a caller writes against this surface once
    and it runs local or remote unchanged.
    """

    # --- Capability + lifecycle -------------------------------------------------
    def supports(self, capability: Capability) -> bool:
        """Whether this backend carries a one-sided :class:`Capability`."""

    def close(self) -> None:
        """Release the session. Real teardown remote; a no-op (RAII) embedded."""

    def __enter__(self) -> "Session": ...
    def __exit__(self, *exc: object) -> Optional[bool]: ...

    # --- Tenant scope + handshake ----------------------------------------------
    def set_tenant(self, tenant_id: str) -> None: ...
    def tenant(self) -> Optional[str]: ...
    def tenant_scope(self, tenant_id: str) -> ContextManager["Session"]: ...
    def get_server_info(self) -> Dict[str, Any]: ...
    def sql(self, query: str) -> Any: ...

    # --- Sources + model lifecycle ---------------------------------------------
    def add_source(self, name: str, *, url: str, format: str) -> None: ...
    def list_sources(self) -> List[Dict[str, Any]]: ...
    def describe_source(self, source_id: str) -> Optional[Dict[str, Any]]: ...
    def list_models(self) -> List[Dict[str, Any]]: ...
    def describe_model(self, model_id: str) -> Optional[Dict[str, Any]]: ...
    def delete_model(
        self, model_id: str, *, version: Optional[int] = None, if_exists: bool = False
    ) -> None: ...

    # --- Embeddings + search ----------------------------------------------------
    def encode_query(
        self, *, model: str, query: Union[str, bytes], modality: Optional[str] = None
    ) -> List[float]: ...
    def generate_embeddings(
        self,
        *,
        source: str,
        model: str,
        columns: List[str],
        key: str,
        modality: Optional[str] = None,
        cache: Optional[str] = None,
    ) -> str: ...
    def search(
        self,
        source: str,
        *,
        query: List[float],
        k: int,
        filter: Optional[str] = None,
        select: Optional[List[str]] = None,
        embedding_table: Optional[str] = None,
    ) -> Any: ...
    def infer(
        self,
        *,
        source: str,
        model: str,
        columns: List[str],
        task: str,
        key: str,
        cache: Optional[str] = None,
    ) -> Any: ...

    # --- Training + predict -----------------------------------------------------
    def fine_tune(self, **kwargs: Any) -> TrainingJobHandle: ...
    def fine_tune_graph(self, **kwargs: Any) -> TrainingJobHandle: ...
    def train_context_predictor(self, source: str, **kwargs: Any) -> TrainingJobHandle: ...
    def predict_with_context_predictor(
        self, model_id: str, **kwargs: Any
    ) -> Dict[str, Any]: ...

    # --- Engine-state pipeline --------------------------------------------------
    def build_neighbor_graph(self, source: str, **kwargs: Any) -> str: ...
    def propagate_embeddings(self, source: str, **kwargs: Any) -> str: ...
    def asof_join(self, spine: str, facts: str, **kwargs: Any) -> str: ...
    def assemble_context(self, source: str, **kwargs: Any) -> Dict[str, Any]: ...
    def recompute(self, table: str, **kwargs: Any) -> Dict[str, Any]: ...
    def verify_materialization(
        self, table: str, expected_definition: Optional[str] = None
    ) -> Dict[str, Any]: ...
    def staleness(self, table: str, current_definition: str) -> Dict[str, Any]: ...
    def derives_from(self, table: str) -> List[Dict[str, Any]]: ...

    # --- Evaluation -------------------------------------------------------------
    def eval_embeddings(self, **kwargs: Any) -> Dict[str, Any]: ...
    def eval_per_query(self, eval_run_id: str) -> List[Dict[str, Any]]: ...
    def eval_inference(self, **kwargs: Any) -> Dict[str, Any]: ...
    def eval_compare(self, **kwargs: Any) -> Dict[str, Any]: ...
    def eval_calibration(self, **kwargs: Any) -> Dict[str, Any]: ...

    # --- Channels ---------------------------------------------------------------
    def register_channel(
        self, channel_id: str, *, priority: int, columns: Sequence[Tuple[str, str]]
    ) -> None: ...
    def add_channel_columns(
        self, channel_id: str, *, columns: Sequence[Tuple[str, str]]
    ) -> None: ...
    def list_channels(self) -> List[Dict[str, Any]]: ...

    # --- Mutable companion tables + topics -------------------------------------
    def create_mutable_table(self, name: str, **kwargs: Any) -> str: ...
    def drop_mutable_table(self, name: str, *, if_exists: bool = False) -> None: ...
    def list_mutable_tables(self) -> List[Dict[str, Any]]: ...
    def register_topic(self, name: str, **kwargs: Any) -> str: ...
    def drop_topic(self, name: str, *, if_exists: bool = False) -> None: ...
    def list_topics(self) -> List[str]: ...
    def publish_topic(self, topic: str, *, batch: Any) -> int: ...
    def subscribe_collect(self, topic: str, **kwargs: Any) -> Any: ...

    # --- Stateless numerics -----------------------------------------------------
    def conformalize(self, *args: Any, **kwargs: Any) -> List[List[int]]: ...
    def conformalize_interval(self, *args: Any, **kwargs: Any) -> List[Tuple[float, float]]: ...
    def conformalize_cqr(self, *args: Any, **kwargs: Any) -> List[Tuple[float, float]]: ...
    def rrf_fuse(self, ranked_lists: Sequence[Sequence[str]], **kwargs: Any) -> List[Tuple[str, float]]: ...
