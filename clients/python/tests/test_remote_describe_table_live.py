"""`describe_table` reads IDENTICALLY on both transports, against one catalog.

An EMBEDDED engine materializes an embedding table through a real local model
and releases the catalog; a real CPU `jammi-server` then opens the same
artifact directory, and the remote description is compared, whole, against the
embedded one. Both read the one `.materialization.json` the producer wrote, so
the claim is that the wire carries that record without reshaping it.

The refusal rides the same comparison: once the sidecar is removed, both
transports refuse the table typed (`MissingManifest` embedded, which refines the
`BackendError` the remote transport raises for its `NOT_FOUND`).

Selected by the `live_server` and `embedded` markers: it needs a built
`jammi-server` (`JAMMI_SERVER_BIN`) and the in-process engine as the parity peer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import urlparse

import pytest

import jammi
from jammi.errors import BackendError, MissingManifest

pytestmark = [pytest.mark.live_server, pytest.mark.embedded]

REPO = Path(__file__).resolve().parents[3]
PATENTS_URL = f"file://{REPO / 'tests' / 'fixtures' / 'patents.parquet'}"
TINY_BERT = f"local:{REPO / 'cookbook' / 'fixtures' / 'tiny_bert'}"


def _sidecar(artifact_dir: Path, table: str) -> Path:
    """The `.materialization.json` beside the table's Parquet artifact."""
    conn = sqlite3.connect(str(artifact_dir / "catalog.db"))
    try:
        (parquet_path,) = conn.execute(
            "SELECT parquet_path FROM result_tables WHERE table_name = ?", (table,)
        ).fetchone()
    finally:
        conn.close()
    parquet = Path(urlparse(parquet_path).path)
    return parquet.with_name(f"{parquet.stem}.materialization.json")


def test_remote_and_embedded_describe_table_agree(tmp_path, live_server_on):
    artifact_dir = tmp_path / "engine"
    artifact_dir.mkdir()

    db = jammi.connect(f"file://{artifact_dir}")
    try:
        db.add_source("patents", url=PATENTS_URL, format="parquet")
        table = db.generate_embeddings(
            source="patents", model=TINY_BERT, columns=["abstract"], key="id"
        )
        embedded = db.describe_table(table)
    finally:
        db.close()

    # The record names what produced the table: the embedding verb, the one
    # local model it ran (with the digest of its files), on the CPU.
    assert embedded["manifest_version"] >= 1
    assert embedded["env"]["device"] == "cpu"
    (model,) = embedded["env"]["models"]
    assert model["run"]["runner"] == "local"
    assert len(model["run"]["content_digest"]) == 64
    assert embedded["input_anchors"][0]["source"] == "patents"

    with live_server_on(artifact_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            assert remote.describe_table(table) == embedded
            # The recorded definition is the one `staleness` compares against:
            # passed back, it decides no `definition_changed` reason. `patents`
            # is a plain file, anchored at a read instant, so the verdict is
            # undecidable rather than fresh.
            verdict = remote.staleness(table, embedded["definition_hash"])
            assert verdict == {
                "staleness": "undecidable",
                "unpinned": ["patents"],
                "decided_reasons": [],
            }
        finally:
            remote.close()

    _sidecar(artifact_dir, table).unlink()

    db = jammi.connect(f"file://{artifact_dir}")
    try:
        with pytest.raises(MissingManifest):
            db.describe_table(table)
    finally:
        db.close()

    with live_server_on(artifact_dir) as endpoint:
        remote = jammi.connect(endpoint)
        try:
            with pytest.raises(BackendError):
                remote.describe_table(table)
        finally:
            remote.close()
