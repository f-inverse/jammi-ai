"""Hermetic tests for the `world_size` keyword on the two LoRA fine-tune verbs.

`world_size` is the data-parallel rank count: how many ranks train one job
cooperatively. It rides `SubmitJobRequest.world_size`, whose `0` means UNSET and
resolves to the engine's default of one rank — so the builder writes the field
ONLY when the caller asked for more than one rank, and a call that never names
the keyword produces byte-for-byte the request it produced before the keyword
existed. The golden hex below pins exactly that: it was captured from the
builders at `main` @ 9db8d395 (pre-`world_size`) with

    python -c "from jammi._assembly import build_fine_tune_request as b; \
               print(b(**FT).SerializeToString(deterministic=True).hex())"

over the `_FINE_TUNE` / `_GRAPH` / `_CONTEXT_PREDICTOR` fixtures in this module.
Regenerating a golden is only ever correct when the request shape changed for a
reason unrelated to this keyword — read the diff before touching one.

The context-predictor verb deliberately carries NO `world_size`: the field is
the wire form of `TrainingCommon.world_size`, and `ContextPredictorSpec` folds
no common block (`crates/jammi-wire/proto/jammi/v1/job.proto`, the `world_size`
comment: "Unused by the context-predictor kind"). `test_context_predictor_*`
below pins that absence on both the builder and the two public verbs so the
keyword cannot be added there by drift.

No channel is dialed: the builders are free functions in the assembly layer.
"""

from __future__ import annotations

import inspect

import pytest

from jammi import EmbeddedBackend, RemoteDatabase
from jammi._assembly import (
    build_context_predictor_request,
    build_fine_tune_graph_request,
    build_fine_tune_request,
)
from jammi._generated.jammi.v1 import job_pb2
from jammi.errors import InvalidArgument

_FINE_TUNE = dict(
    source="docs",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    columns=["text"],
    method="lora",
    task="text_embedding",
    epochs=3,
    idempotency_key="k-1",
)

_GRAPH = dict(
    node_source="nodes",
    id_column="id",
    text_column="text",
    edge_source="edges",
    src_column="src",
    dst_column="dst",
    base_model="sentence-transformers/all-MiniLM-L6-v2",
    edge_provenance="declared",
    epochs=2,
    idempotency_key="k-2",
)

_CONTEXT_PREDICTOR = dict(
    key_column="k",
    task_column="t",
    value_column="v",
    idempotency_key="k-3",
)

# Serialized bytes of each builder's output at `main` @ 9db8d395, BEFORE the
# `world_size` keyword existed. See the module docstring.
_GOLDEN_FINE_TUNE = (
    "0a100a04646f637312047465787418012001222673656e74656e63652d7472616e73"
    "666f726d6572732f616c6c2d4d696e694c4d2d4c362d76322a02280332036b2d31"
)
_GOLDEN_GRAPH = (
    "12440a240a056e6f646573120269641a0474657874220565646765732a0373726332"
    "036473743801121c0804100219000000000000f03f21000000000000f03f28013001"
    "3801222673656e74656e63652d7472616e73666f726d6572732f616c6c2d4d696e69"
    "4c4d2d4c362d76322a0f2802420b1a0909000000000000344032036b2d32"
)
_GOLDEN_CONTEXT_PREDICTOR = (
    "1a580a036f627312510a156f62732d636f6e746578742d707265646963746f721002"
    "1a016b2201742a0176302038404004480252060a040a0212005864617b14ae47e17a"
    "743f69000000000000f03f719a9999999999c93f780432036b2d33"
)

# Every public client verb that submits through one of the two LoRA fine-tune
# builders. Derived from
#   grep -rn 'build_fine_tune_request\|build_fine_tune_graph_request' \
#     clients/python/jammi --include='*.py'
# which names exactly `_database.py` (RemoteDatabase, the gRPC arm) and
# `_embedded.py` (EmbeddedBackend, the in-process arm) as call sites.
_LORA_VERBS = [
    (RemoteDatabase, "fine_tune"),
    (RemoteDatabase, "fine_tune_graph"),
    (EmbeddedBackend, "fine_tune"),
    (EmbeddedBackend, "fine_tune_graph"),
]

# The verbs that must NOT grow the keyword (no common block on the wire).
_NON_LORA_VERBS = [
    (RemoteDatabase, "train_context_predictor"),
    (EmbeddedBackend, "train_context_predictor"),
]


def _hex(message) -> str:
    return message.SerializeToString(deterministic=True).hex()


# --- default call: byte-identical to the pre-`world_size` request -------------


def test_fine_tune_default_bytes_are_the_pre_world_size_bytes() -> None:
    """A `fine_tune` request built without naming `world_size` serializes to
    exactly the bytes the builder produced before the keyword existed."""
    assert _hex(build_fine_tune_request(**_FINE_TUNE)) == _GOLDEN_FINE_TUNE


def test_graph_default_bytes_are_the_pre_world_size_bytes() -> None:
    assert _hex(build_fine_tune_graph_request(**_GRAPH)) == _GOLDEN_GRAPH


def test_context_predictor_bytes_are_the_pre_world_size_bytes() -> None:
    """The context-predictor request is untouched by this change end to end."""
    assert (
        _hex(build_context_predictor_request("obs", **_CONTEXT_PREDICTOR))
        == _GOLDEN_CONTEXT_PREDICTOR
    )


def test_world_size_one_is_the_same_bytes_as_omitting_it() -> None:
    """An explicit `world_size=1` is the default, not a distinct request: `1` is
    what the engine resolves an unset field to, so writing the field would put a
    redundant byte on the wire and split one job identity into two encodings."""
    for build, kwargs in (
        (build_fine_tune_request, _FINE_TUNE),
        (build_fine_tune_graph_request, _GRAPH),
    ):
        assert _hex(build(**kwargs, world_size=1)) == _hex(build(**kwargs))


def test_default_leaves_the_field_off_the_encoding_entirely() -> None:
    """Not merely "equal to zero": the field is absent from the encoded
    message, so a decoder sees the same field set it saw before."""
    for build, kwargs in (
        (build_fine_tune_request, _FINE_TUNE),
        (build_fine_tune_graph_request, _GRAPH),
        (
            lambda **kw: build_context_predictor_request("obs", **kw),
            _CONTEXT_PREDICTOR,
        ),
    ):
        request = build(**kwargs)
        present = {field.name for field, _ in request.ListFields()}
        assert "world_size" not in present
        assert request.world_size == 0


# --- world_size > 1 sets the field -------------------------------------------


@pytest.mark.parametrize("ranks", [2, 8])
def test_fine_tune_world_size_sets_the_field(ranks: int) -> None:
    request = build_fine_tune_request(**_FINE_TUNE, world_size=ranks)
    assert request.world_size == ranks
    assert "world_size" in {field.name for field, _ in request.ListFields()}
    # Survives a wire round-trip on the field the server reads.
    assert job_pb2.SubmitJobRequest.FromString(
        request.SerializeToString()
    ).world_size == ranks


@pytest.mark.parametrize("ranks", [2, 8])
def test_graph_world_size_sets_the_field(ranks: int) -> None:
    request = build_fine_tune_graph_request(**_GRAPH, world_size=ranks)
    assert request.world_size == ranks
    assert "world_size" in {field.name for field, _ in request.ListFields()}
    assert job_pb2.SubmitJobRequest.FromString(
        request.SerializeToString()
    ).world_size == ranks


def test_world_size_is_the_only_difference_from_the_default_request() -> None:
    """Setting the keyword changes the `world_size` field and nothing else: the
    two encodings agree once `world_size` is cleared back off."""
    for build, kwargs in (
        (build_fine_tune_request, _FINE_TUNE),
        (build_fine_tune_graph_request, _GRAPH),
    ):
        request = build(**kwargs, world_size=4)
        request.ClearField("world_size")
        assert _hex(request) == _hex(build(**kwargs))


# --- refusals: values outside the domain never reach the wire ----------------


@pytest.mark.parametrize("ranks", [0, -1, -7])
def test_world_size_below_one_is_refused_by_the_fine_tune_builder(ranks: int) -> None:
    """`0` is the wire's UNSET marker, so a caller who writes `world_size=0`
    means something the wire cannot say — refuse it here rather than encode a
    single-rank job the caller did not ask for."""
    with pytest.raises(InvalidArgument) as excinfo:
        build_fine_tune_request(**_FINE_TUNE, world_size=ranks)
    assert "world_size" in str(excinfo.value)


@pytest.mark.parametrize("ranks", [0, -1, -7])
def test_world_size_below_one_is_refused_by_the_graph_builder(ranks: int) -> None:
    with pytest.raises(InvalidArgument) as excinfo:
        build_fine_tune_graph_request(**_GRAPH, world_size=ranks)
    assert "world_size" in str(excinfo.value)


def test_refusal_is_a_jammi_error_and_a_value_error() -> None:
    """`InvalidArgument` subclasses both `JammiError` and `ValueError`, so an
    existing `except ValueError` around a submit still catches it."""
    import jammi.errors

    with pytest.raises(jammi.errors.JammiError):
        build_fine_tune_request(**_FINE_TUNE, world_size=0)
    with pytest.raises(ValueError):
        build_fine_tune_request(**_FINE_TUNE, world_size=0)


@pytest.mark.parametrize("ranks", ["2", 2.5, None, True])
def test_non_integer_world_size_is_refused(ranks: object) -> None:
    """The keyword's domain is `int >= 1`. A `float`/`str`/`None` would either
    raise a raw protobuf `TypeError` deep in the encoder or (for `bool`, which
    IS an `int` in Python) silently encode `1`/`0` — both are worse than a typed
    refusal at the edge."""
    with pytest.raises(InvalidArgument):
        build_fine_tune_request(**_FINE_TUNE, world_size=ranks)


# --- the public verbs carry the keyword --------------------------------------


@pytest.mark.parametrize(("owner", "verb"), _LORA_VERBS)
def test_public_lora_verb_carries_world_size(owner: type, verb: str) -> None:
    """Every public verb that submits through a LoRA fine-tune builder exposes
    the keyword, keyword-only, defaulted to one rank."""
    parameter = inspect.signature(getattr(owner, verb)).parameters.get("world_size")
    assert parameter is not None, f"{owner.__name__}.{verb} has no world_size"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 1


@pytest.mark.parametrize(("owner", "verb"), _NON_LORA_VERBS)
def test_context_predictor_verb_has_no_world_size(owner: type, verb: str) -> None:
    assert "world_size" not in inspect.signature(getattr(owner, verb)).parameters


def test_context_predictor_builder_has_no_world_size() -> None:
    parameters = inspect.signature(build_context_predictor_request).parameters
    assert "world_size" not in parameters


@pytest.mark.parametrize(("owner", "verb"), _LORA_VERBS)
def test_public_lora_verb_documents_world_size(owner: type, verb: str) -> None:
    """The keyword is documented where a caller reads it, not only in the
    builder: `world_size` names ranks, and the docstring says what `1` means."""
    doc = inspect.getdoc(getattr(owner, verb)) or ""
    assert "world_size" in doc
    assert "cooperatively" in doc
