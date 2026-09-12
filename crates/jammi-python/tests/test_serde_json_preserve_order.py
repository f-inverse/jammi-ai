"""The workspace `serde_json` declaration (`Cargo.toml` `[workspace.dependencies]`,
`features = [.., "preserve_order"]`) reached the wheel's unit graph.

`PerQueryAudit.query_lineage` round-trips a Python object through
`serde_json::Value` (Python `json.dumps` -> `serde_json::from_str` on construct,
`serde_json::to_string` -> `json.loads` on read). With `preserve_order` a
`serde_json::Map` keeps insertion order; without it keys sort, so a dict whose
keys are inserted out of alphabetical order reads back re-sorted. A flip here
means the wheel was built with a different `serde_json` feature set than the
tested binaries, and every persisted `Map`/`json!` producer must be re-swept.
"""

import uuid

import jammi_native


def test_query_lineage_round_trip_keeps_insertion_order():
    record = jammi_native.PerQueryAudit(
        query_id=str(uuid.uuid4()),
        model_id="m",
        model_version="1",
        query_lineage={"b": 1, "a": 2},
        top_k_result_ids=["r0"],
        retrieval_scores=[0.5],
    )
    assert list(record.query_lineage.keys()) == ["b", "a"], (
        "the workspace `preserve_order` declaration did not reach the wheel's "
        f"unit graph: keys came back {list(record.query_lineage.keys())}"
    )
