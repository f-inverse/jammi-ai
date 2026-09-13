"""The remote `kind` string equals the embedded one for EVERY served kind.

`_result_table_to_dict` spells a result table's `kind` by looking the wire
`ResultTableKind` value up in `_assembly._RESULT_TABLE_KIND_NAME`, falling back
to ``"Unspecified"`` for a value it does not carry. The embedded engine spells
the same column off its own `jammi_db::catalog::result_repo::ResultTableKind` —
a plain ``#[derive(serde::Serialize)]`` unit enum with no ``rename_all``, so a
variant serialises as its Rust variant name (``TrainingSet``). A kind that is
served on the wire but missing from the map therefore reads ``"Unspecified"``
remotely and its real name embedded: the two transports disagree on exactly the
row a caller uses the kind to identify.

So the map is pinned TOTAL over the enum's served values (every value except
``RESULT_TABLE_KIND_UNSPECIFIED``), derived from the generated descriptor rather
than from a hand-kept list — a kind appended to the proto fails this suite until
it is mapped, instead of silently degrading a remote read. The expected spelling
of each is pinned here in one table; the ``Unspecified`` fallback stays, and is
pinned too, because a proto-version skew (a server serving a value this client's
stubs predate) is the one case it exists for.

Hermetic: no server, no embedded engine — only constructed protos.
"""

from __future__ import annotations

from jammi._assembly import _RESULT_TABLE_KIND_NAME
from jammi._database import _result_table_to_dict
from jammi._generated.jammi.v1 import embedding_pb2

# Proto value name → the string the embedded engine's `ResultTableKind`
# serialises to. Every served value of the enum appears here; the totality
# assertion below derives the served set from the descriptor and fails if this
# table (and hence the map under test) misses one.
_EXPECTED_KIND_NAME = {
    "MODEL": "Model",
    "NEIGHBOR_GRAPH": "NeighborGraph",
    "ASOF_JOIN": "AsofJoin",
    "TRAINING_SET": "TrainingSet",
}

_UNSPECIFIED = "RESULT_TABLE_KIND_UNSPECIFIED"


def _served_value_names() -> set:
    """Every `ResultTableKind` value a server may put on the wire.

    Read off the generated descriptor — the same source the stubs are built
    from — so the set this suite ranges over is the enum as it is TODAY, not a
    copy of it that can go stale.
    """
    return {
        value.name
        for value in embedding_pb2.ResultTableKind.DESCRIPTOR.values
        if value.name != _UNSPECIFIED
    }


def test_kind_name_map_is_total_over_the_served_enum():
    """Every served `ResultTableKind` value has a name — no kind falls through
    to `"Unspecified"`, and no stale key survives a value's removal."""
    served = _served_value_names()

    assert set(_EXPECTED_KIND_NAME) == served, (
        "this suite's expected-spelling table must range over exactly the "
        "enum's served values; a kind was appended to (or dropped from) "
        "jammi/v1/embedding.proto"
    )

    mapped = {
        embedding_pb2.ResultTableKind.Value(name): _EXPECTED_KIND_NAME[name]
        for name in served
    }
    assert _RESULT_TABLE_KIND_NAME == mapped, (
        "_RESULT_TABLE_KIND_NAME must carry every served kind, spelled the way "
        "the embedded engine's ResultTableKind serialises it"
    )


def test_every_served_kind_renders_its_own_name_never_the_fallback():
    """Projecting a `ResultTable` of each served kind yields that kind's name.

    Drives the real projection (not the map alone), so a fallback reached at
    the call site — `.get(rt.kind, "Unspecified")` — is caught for every kind,
    the training-set row included.
    """
    for name in sorted(_served_value_names()):
        table = embedding_pb2.ResultTable(
            table_name=f"t_{name.lower()}",
            source_id="s1",
            model_id="m1",
            dimensions=8,
            row_count=3,
            status="ready",
            kind=embedding_pb2.ResultTableKind.Value(name),
        )
        assert _result_table_to_dict(table)["kind"] == _EXPECTED_KIND_NAME[name], (
            f"{name} must project as {_EXPECTED_KIND_NAME[name]!r}, the string "
            "the embedded engine serialises for the same row"
        )


def test_unspecified_still_falls_back_to_unspecified():
    """The skew case the fallback exists for: a value with no concrete kind
    projects as `"Unspecified"` rather than raising or dropping the field."""
    table = embedding_pb2.ResultTable(
        table_name="t_skew",
        source_id="s1",
        model_id="m1",
        dimensions=8,
        row_count=0,
        status="ready",
        kind=embedding_pb2.ResultTableKind.RESULT_TABLE_KIND_UNSPECIFIED,
    )
    assert _result_table_to_dict(table)["kind"] == "Unspecified"
