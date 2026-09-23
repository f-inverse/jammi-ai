"""Hermetic tests for the graph-structure request assembly and the
weighted-sum propagation output.

`generate_structure_embeddings` and `propagate_embeddings` share one graph
arm (`edge_graph_table` xor `edge_source`) and one request assembly the
embedded `Database` and the `RemoteDatabase` both submit; these tests pin
the WIRE ENCODING of the keywords — explicit presence for every optional
scalar, an omitted `weights` left empty for the engine's default, and the
loud refusals a caller gets before anything is dialed. No channel is dialed:
the builders are free functions in the assembly layer.
"""

from __future__ import annotations

import inspect

import pytest

from jammi import EmbeddedBackend, RemoteDatabase
from jammi._assembly import (
    build_generate_structure_embeddings_request,
    build_propagate_embeddings_request,
)
from jammi._generated.jammi.v1 import inference_pb2, pipeline_pb2


def test_structure_request_leaves_every_omitted_knob_to_the_engine():
    request = build_generate_structure_embeddings_request(
        "ledger", edge_source="ledger", edge_src_column="payer", edge_dst_column="payee"
    )
    assert request.source_id == "ledger"
    assert request.WhichOneof("graph") == "edge_source"
    assert request.edge_source.src_column == "payer"
    assert request.edge_source.dst_column == "payee"
    assert not request.edge_source.HasField("weight_column")
    assert not request.HasField("key_column")
    assert not request.HasField("dimensions")
    assert not request.HasField("beta")
    assert not request.HasField("sparsity")
    assert not request.HasField("seed")
    assert list(request.weights) == []
    assert request.direction == inference_pb2.EdgeDirection.EDGE_DIRECTION_UNSPECIFIED
    assert (
        request.weighting
        == pipeline_pb2.PropagationWeighting.PROPAGATION_WEIGHTING_UNSPECIFIED
    )


def test_structure_request_carries_every_named_knob():
    request = build_generate_structure_embeddings_request(
        "ledger",
        key_column="account",
        edge_graph_table="ledger_knn",
        direction="undirected",
        weighting="edge_similarity",
        dimensions=128,
        weights=[0.0, 1.0, 0.5],
        beta=-0.5,
        sparsity=8.0,
        seed=7,
        cache="use",
    )
    assert request.WhichOneof("graph") == "edge_graph_table"
    assert request.edge_graph_table == "ledger_knn"
    assert request.key_column == "account"
    assert request.direction == inference_pb2.EdgeDirection.UNDIRECTED
    assert (
        request.weighting
        == pipeline_pb2.PropagationWeighting.PROPAGATION_WEIGHTING_EDGE_SIMILARITY
    )
    assert request.dimensions == 128
    assert list(request.weights) == [0.0, 1.0, 0.5]
    assert request.beta == -0.5
    assert request.sparsity == 8.0
    assert request.seed == 7
    assert request.cache == inference_pb2.CachePolicy.CACHE_POLICY_USE
    # The bytes round-trip: what the embedded binding hands the engine is what
    # the remote client sends.
    again = pipeline_pb2.GenerateStructureEmbeddingsRequest()
    again.ParseFromString(request.SerializeToString())
    assert again == request


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"edge_graph_table": "g", "edge_source": "edges"},
        {"edge_source": "edges", "direction": "sideways"},
        {"edge_source": "edges", "weighting": "attention"},
    ],
)
def test_structure_request_refuses_a_malformed_graph_or_enum_loudly(kwargs):
    with pytest.raises(ValueError):
        build_generate_structure_embeddings_request("ledger", **kwargs)


def test_propagate_weighted_sum_carries_its_hop_weights_and_nothing_else_does():
    request = build_propagate_embeddings_request(
        "docs", edge_graph_table="g", output="weighted_sum", hop_weights=[0.0, 1.0, 1.0]
    )
    assert request.output == pipeline_pb2.PropagationOutput.PROPAGATION_OUTPUT_WEIGHTED_SUM
    assert list(request.hop_weights) == [0.0, 1.0, 1.0]
    with pytest.raises(ValueError):
        build_propagate_embeddings_request(
            "docs", edge_graph_table="g", output="final", hop_weights=[1.0]
        )
    with pytest.raises(ValueError):
        build_propagate_embeddings_request("docs", edge_graph_table="g", hop_weights=[1.0])


def test_both_transports_expose_the_same_structure_signature():
    remote = inspect.signature(RemoteDatabase.generate_structure_embeddings)
    embedded = inspect.signature(EmbeddedBackend.generate_structure_embeddings)
    assert list(remote.parameters) == list(embedded.parameters)
