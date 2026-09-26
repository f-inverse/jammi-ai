"""Hermetic tests for the `cache` keyword on the two LoRA fine-tune verbs.

`cache` carries a fine-tune job's model-level cache policy; the worker decides
a reuse, never the client, so these tests pin the WIRE ENCODING of the
keyword, never a reuse outcome. It rides `SubmitJobRequest.cache`, the same
shared `jammi.v1.inference.CachePolicy` enum every other producer verb's
`cache` field carries, resolved by the same `_cache_policy_value`: the engine
default (`None` or ``"bypass"``) is `UNSPECIFIED` and stays off the wire — the
one encoding every writer of it shares — and only ``"use"`` is written.

The context-predictor verb deliberately carries NO `cache`: the field lives on
the shared `TrainingCommon` both LoRA kinds fold (the same block
`test_world_size.py` documents for `world_size`), and `ContextPredictorSpec`
folds no common block at all — it has no materialization to probe.

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
from jammi._generated.jammi.v1 import inference_pb2, job_pb2

# The same request shapes `test_world_size.py` builds.
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


_LORA_VERBS = [
    (RemoteDatabase, "fine_tune"),
    (RemoteDatabase, "fine_tune_graph"),
    (EmbeddedBackend, "fine_tune"),
    (EmbeddedBackend, "fine_tune_graph"),
]

_NON_LORA_VERBS = [
    (RemoteDatabase, "train_context_predictor"),
    (EmbeddedBackend, "train_context_predictor"),
]


def _hex(message) -> str:
    return message.SerializeToString(deterministic=True).hex()


# --- default call: the field stays off the wire -----------------------------


def test_explicit_bypass_is_the_same_bytes_as_omitting_it() -> None:
    """An explicit `cache="bypass"` is the default, not a distinct request:
    `"bypass"` is what the engine resolves an unset field to, so it takes the
    unset encoding — one encoding per intent, as `world_size=1`'s."""
    for build, kwargs in (
        (build_fine_tune_request, _FINE_TUNE),
        (build_fine_tune_graph_request, _GRAPH),
    ):
        assert _hex(build(**kwargs, cache="bypass")) == _hex(build(**kwargs))


def test_default_leaves_the_field_off_the_encoding_entirely() -> None:
    """Not merely "equal to UNSPECIFIED": the field is absent from the
    encoded message."""
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
        assert "cache" not in present
        assert request.cache == inference_pb2.CachePolicy.CACHE_POLICY_UNSPECIFIED


# --- cache="use" sets the field -----------------------------------------------


def test_fine_tune_cache_use_sets_the_field() -> None:
    request = build_fine_tune_request(**_FINE_TUNE, cache="use")
    assert request.cache == inference_pb2.CachePolicy.CACHE_POLICY_USE
    assert "cache" in {field.name for field, _ in request.ListFields()}
    # Survives a wire round-trip on the field the server reads.
    assert (
        job_pb2.SubmitJobRequest.FromString(request.SerializeToString()).cache
        == inference_pb2.CachePolicy.CACHE_POLICY_USE
    )


def test_graph_cache_use_sets_the_field() -> None:
    request = build_fine_tune_graph_request(**_GRAPH, cache="use")
    assert request.cache == inference_pb2.CachePolicy.CACHE_POLICY_USE
    assert "cache" in {field.name for field, _ in request.ListFields()}
    assert (
        job_pb2.SubmitJobRequest.FromString(request.SerializeToString()).cache
        == inference_pb2.CachePolicy.CACHE_POLICY_USE
    )


def test_cache_is_the_only_difference_from_the_default_request() -> None:
    """Setting the keyword changes the `cache` field and nothing else: the two
    encodings agree once `cache` is cleared back off."""
    for build, kwargs in (
        (build_fine_tune_request, _FINE_TUNE),
        (build_fine_tune_graph_request, _GRAPH),
    ):
        request = build(**kwargs, cache="use")
        request.ClearField("cache")
        assert _hex(request) == _hex(build(**kwargs))


def test_world_size_and_cache_are_independent_determinants() -> None:
    """The two job-level fields of `SubmitJobRequest` do not clobber each
    other: setting both together sets both fields, and each alone leaves the
    other at its unset default."""
    request = build_fine_tune_request(**_FINE_TUNE, world_size=4, cache="use")
    assert request.world_size == 4
    assert request.cache == inference_pb2.CachePolicy.CACHE_POLICY_USE

    world_size_only = build_fine_tune_request(**_FINE_TUNE, world_size=4)
    assert world_size_only.cache == inference_pb2.CachePolicy.CACHE_POLICY_UNSPECIFIED

    cache_only = build_fine_tune_request(**_FINE_TUNE, cache="use")
    assert cache_only.world_size == 0


# --- refusals: values outside the domain never reach the wire ----------------


def test_unknown_cache_value_is_refused_by_the_fine_tune_builder() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_fine_tune_request(**_FINE_TUNE, cache="always")
    assert "cache" in str(excinfo.value)


def test_unknown_cache_value_is_refused_by_the_graph_builder() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_fine_tune_graph_request(**_GRAPH, cache="always")
    assert "cache" in str(excinfo.value)


# --- the public verbs carry the keyword --------------------------------------


@pytest.mark.parametrize(("owner", "verb"), _LORA_VERBS)
def test_public_lora_verb_carries_cache(owner: type, verb: str) -> None:
    """Every public verb that submits through a LoRA fine-tune builder exposes
    the keyword, keyword-only, defaulted to unset (the engine's Bypass)."""
    parameter = inspect.signature(getattr(owner, verb)).parameters.get("cache")
    assert parameter is not None, f"{owner.__name__}.{verb} has no cache"
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize(("owner", "verb"), _NON_LORA_VERBS)
def test_context_predictor_verb_has_no_cache(owner: type, verb: str) -> None:
    assert "cache" not in inspect.signature(getattr(owner, verb)).parameters


def test_context_predictor_builder_has_no_cache() -> None:
    parameters = inspect.signature(build_context_predictor_request).parameters
    assert "cache" not in parameters


@pytest.mark.parametrize(("owner", "verb"), _LORA_VERBS)
def test_public_lora_verb_documents_cache(owner: type, verb: str) -> None:
    """The keyword is documented where a caller reads it, not only in the
    builder."""
    doc = inspect.getdoc(getattr(owner, verb)) or ""
    assert "cache" in doc
    assert "bypass" in doc.lower()
