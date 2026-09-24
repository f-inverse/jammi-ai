"""The build scripts' shared steps: locating the engine's fixtures and
recording artifact checksums."""

from __future__ import annotations

import hashlib
import json

import pytest

from jammi_cookbook import cache


def test_write_checksums_records_every_artifact_but_itself(tmp_path):
    (tmp_path / "matrix.json").write_text("{}")
    (tmp_path / "checksums.json").write_text("stale")
    cache.write_checksums(tmp_path)
    sums = json.loads((tmp_path / "checksums.json").read_text())
    assert sums == {"matrix.json": hashlib.sha256(b"{}").hexdigest()}


def test_engine_fixtures_root_names_the_first_missing_fixture(tmp_path):
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "a.parquet").write_text("")
    assert cache.engine_fixtures_root(str(tmp_path), "tests/fixtures/a.parquet") == tmp_path
    with pytest.raises(SystemExit, match="tests/fixtures/b.parquet"):
        cache.engine_fixtures_root(str(tmp_path), "tests/fixtures/a.parquet", "tests/fixtures/b.parquet")


def test_engine_fixtures_root_without_a_root_names_the_flag(monkeypatch):
    monkeypatch.delenv("JAMMI_FIXTURES_ROOT", raising=False)
    with pytest.raises(SystemExit, match="--fixtures-root"):
        cache.engine_fixtures_root(None, "tests/fixtures/a.parquet")
