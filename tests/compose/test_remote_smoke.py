#!/usr/bin/env python3
"""Hermetic unit suite for the shared remote-smoke oracle
(`remote_smoke.py`) and its two drivers (`shape_b_remote.py`, the Compose
driver; `shape_c_kube_remote.py`, the Kubernetes driver).

Stdlib `unittest` only -- no `jammi` import, no Docker, no `kubectl`, no
network. Everything the oracle needs from the client is faked.

Wired into `ci.yml`'s `Guard` matrix as `python3 tests/compose/test_remote_smoke.py`
(`check_ci_guard_wiring.py` tracks every tracked `tests/**/test_*.py`, so this
suite must be wired in the same commit it lands).

What each case proves:
  - `test_compose_restart_and_wait_argv`: the Compose driver's
    `restart_and_wait` issues exactly the `docker compose ... restart
    jammi-server` argv, with `check=True`.
  - `test_compose_driver_passes_durable_after_restart`: the Compose driver
    calls `remote_smoke.run(...)` with `after_restart is
    remote_smoke.durable_after_restart` (callback IDENTITY, not just "a
    callable").
  - `test_kube_driver_passes_shared_catalog_after_restart`: same identity
    check for the Kubernetes driver's `after_restart`.
  - `test_kube_restart_issues_rollout_restart_then_status`: the kube
    driver's restart strategy issues `kubectl -n jammi-ci rollout restart
    deploy/jammi-server` THEN `rollout status ... --timeout=180s`, in that
    order.
  - `test_durable_after_restart_raises_on_key_mismatch`: NEGATIVE control --
    `durable_after_restart` on a fake `Ctx` whose second search differs by
    one key raises `AssertionError` naming both key lists (non-vacuous: the
    oracle actually catches the defect it exists to catch).
  - `test_shared_catalog_after_restart_raises_when_source_missing`:
    NEGATIVE control -- `shared_catalog_after_restart` raises when
    `describe_source` returns `None` (the source did not survive the
    restart onto the new pod).
  - `test_shared_catalog_after_restart_raises_on_wrong_broker`: NEGATIVE
    control -- raises when the re-asserted broker is not `jet_stream`.
  - `test_shared_catalog_after_restart_raises_on_wrong_row_count`:
    NEGATIVE control -- raises when the result table's row count changed.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import remote_smoke
import shape_b_remote
import shape_c_kube_remote


class _FakeColumn:
    def __init__(self, values):
        self._values = list(values)

    def to_pylist(self):
        return list(self._values)


class _FakeHits:
    def __init__(self, keys, scores):
        self._keys = keys
        self._scores = scores

    def column(self, name):
        if name == "key":
            return _FakeColumn(self._keys)
        if name == "score":
            return _FakeColumn(self._scores)
        raise KeyError(name)


class _FakeScalarTable:
    """A fake `db.sql(...)` result: `.column(0)[0].as_py()` -> one int."""

    def __init__(self, value):
        self._value = value

    def column(self, _index):
        return [_Scalar(self._value)]


class _Scalar:
    def __init__(self, value):
        self._value = value

    def as_py(self):
        return self._value


class _FakeDb:
    def __init__(self, *, describe_source_result, sql_count, broker):
        self._describe_source_result = describe_source_result
        self._sql_count = sql_count
        self._broker = broker
        self.sql_calls = []

    def describe_source(self, source_id):
        return self._describe_source_result

    def sql(self, query):
        self.sql_calls.append(query)
        return _FakeScalarTable(self._sql_count)

    def get_server_info(self):
        return {"broker": self._broker}


class DurableAfterRestartTests(unittest.TestCase):
    def test_key_mismatch_raises_with_both_key_lists(self):
        ctx = remote_smoke.Ctx(
            db=_FakeDb(describe_source_result=None, sql_count=0, broker="jet_stream"),
            table="t1",
            do_search=lambda: _FakeHits(["1", "2", "999"], [1.0, 0.9, 0.8]),
            hit_keys=["1", "2", "3"],
            hit_scores=[1.0, 0.9, 0.8],
            segments_before=[{"segment_id": 1}],
            n=3,
        )
        with self.assertRaises(AssertionError) as cm:
            remote_smoke.durable_after_restart(ctx)
        msg = str(cm.exception)
        self.assertIn("['1', '2', '3']", msg)
        self.assertIn("['1', '2', '999']", msg)

    def test_matching_keys_and_segments_pass(self):
        db = _FakeDb(describe_source_result=None, sql_count=0, broker="jet_stream")

        def list_index_segments(_table):
            return [{"segment_id": 1}]

        db.list_index_segments = list_index_segments
        ctx = remote_smoke.Ctx(
            db=db,
            table="t1",
            do_search=lambda: _FakeHits(["1", "2", "3"], [1.0, 0.9, 0.8]),
            hit_keys=["1", "2", "3"],
            hit_scores=[1.0, 0.9, 0.8],
            segments_before=[{"segment_id": 1}],
            n=3,
        )
        remote_smoke.durable_after_restart(ctx)  # must not raise


class SharedCatalogAfterRestartTests(unittest.TestCase):
    def _ctx(self, *, describe_source_result, sql_count, broker):
        db = _FakeDb(
            describe_source_result=describe_source_result,
            sql_count=sql_count,
            broker=broker,
        )
        return remote_smoke.Ctx(
            db=db,
            table="t1",
            do_search=lambda: _FakeHits(["1"], [1.0]),
            hit_keys=["1"],
            hit_scores=[1.0],
            segments_before=[],
            n=3,
        )

    def test_raises_when_source_missing(self):
        ctx = self._ctx(describe_source_result=None, sql_count=3, broker="jet_stream")
        with self.assertRaises(AssertionError) as cm:
            remote_smoke.shared_catalog_after_restart(ctx)
        self.assertIn("describe_source", str(cm.exception))

    def test_raises_on_wrong_row_count(self):
        ctx = self._ctx(
            describe_source_result={"id": "patents"}, sql_count=2, broker="jet_stream"
        )
        with self.assertRaises(AssertionError) as cm:
            remote_smoke.shared_catalog_after_restart(ctx)
        self.assertIn("row count", str(cm.exception))

    def test_raises_on_wrong_broker(self):
        ctx = self._ctx(
            describe_source_result={"id": "patents"}, sql_count=3, broker="in_memory"
        )
        with self.assertRaises(AssertionError) as cm:
            remote_smoke.shared_catalog_after_restart(ctx)
        self.assertIn("jet_stream", str(cm.exception))

    def test_passes_when_everything_matches(self):
        ctx = self._ctx(
            describe_source_result={"id": "patents"}, sql_count=3, broker="jet_stream"
        )
        remote_smoke.shared_catalog_after_restart(ctx)  # must not raise


class ComposeDriverTests(unittest.TestCase):
    def test_restart_and_wait_argv(self):
        with mock.patch.object(shape_b_remote.subprocess, "run") as run_mock, \
                mock.patch.object(remote_smoke, "wait_for_ready") as wait_mock:
            shape_b_remote.restart_and_wait("http://127.0.0.1:8080", timeout_secs=60)

        run_mock.assert_called_once_with(
            [
                "docker",
                "compose",
                "-f",
                "deploy/docker-compose.yml",
                "-f",
                "deploy/docker-compose.ci.yml",
                "restart",
                "jammi-server",
            ],
            check=True,
        )
        wait_mock.assert_called_once_with("http://127.0.0.1:8080", 60)

    def test_passes_durable_after_restart_by_identity(self):
        captured = {}

        def fake_run(target, health_url, *, restart, after_restart):
            captured["target"] = target
            captured["health_url"] = health_url
            captured["restart"] = restart
            captured["after_restart"] = after_restart
            return 0

        with mock.patch.object(remote_smoke, "run", fake_run):
            rc = shape_b_remote.run("grpc://127.0.0.1:8081", "http://127.0.0.1:8080")

        self.assertEqual(rc, 0)
        self.assertIs(captured["after_restart"], remote_smoke.durable_after_restart)


class KubeDriverTests(unittest.TestCase):
    def test_passes_shared_catalog_after_restart_by_identity(self):
        captured = {}

        def fake_run(target, health_url, *, restart, after_restart):
            captured["restart"] = restart
            captured["after_restart"] = after_restart
            return 0

        with mock.patch.object(remote_smoke, "run", fake_run), \
                mock.patch.object(shape_c_kube_remote.subprocess, "Popen") as popen_mock, \
                mock.patch.object(remote_smoke, "wait_for_ready"):
            popen_mock.return_value = mock.Mock()
            rc = shape_c_kube_remote.run(
                namespace="jammi-ci",
                deployment="jammi-server",
                service="jammi-server",
                target="grpc://127.0.0.1:8081",
                health_url="http://127.0.0.1:8080",
            )

        self.assertEqual(rc, 0)
        self.assertIs(captured["after_restart"], remote_smoke.shared_catalog_after_restart)

    def test_restart_issues_rollout_restart_then_status_in_order(self):
        captured = {}

        def fake_run(target, health_url, *, restart, after_restart):
            captured["restart"] = restart
            return 0

        with mock.patch.object(remote_smoke, "run", fake_run), \
                mock.patch.object(shape_c_kube_remote.subprocess, "Popen") as popen_mock, \
                mock.patch.object(remote_smoke, "wait_for_ready"):
            popen_mock.return_value = mock.Mock()
            shape_c_kube_remote.run(
                namespace="jammi-ci",
                deployment="jammi-server",
                service="jammi-server",
                target="grpc://127.0.0.1:8081",
                health_url="http://127.0.0.1:8080",
            )

            with mock.patch.object(shape_c_kube_remote.subprocess, "run") as run_mock:
                captured["restart"]()

        self.assertEqual(run_mock.call_count, 2)
        first_argv, first_kwargs = run_mock.call_args_list[0]
        second_argv, second_kwargs = run_mock.call_args_list[1]
        self.assertEqual(
            first_argv[0],
            ["kubectl", "-n", "jammi-ci", "rollout", "restart", "deploy/jammi-server"],
        )
        self.assertTrue(first_kwargs.get("check"))
        self.assertEqual(
            second_argv[0],
            [
                "kubectl",
                "-n",
                "jammi-ci",
                "rollout",
                "status",
                "deploy/jammi-server",
                "--timeout=180s",
            ],
        )
        self.assertTrue(second_kwargs.get("check"))


if __name__ == "__main__":
    unittest.main()
