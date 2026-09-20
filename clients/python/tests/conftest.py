"""Shared fixtures for the client test suite.

Tests that need a host resource are SELECTED by marker, never skipped at run
time: a lane that has the resource runs them, and inside a selected test a
missing resource fails naming it.

* ``embedded`` — needs the in-process engine (the ``[embedded]`` extra).
* ``live_server`` — needs a built ``jammi-server``; ``JAMMI_SERVER_BIN`` names
  it. The live tests drive verbs through the pure-Python ``RemoteDatabase``
  against an embedded parity peer, so they carry both markers.

``make test`` deselects both for the native-free install; ``make test-embedded``
and ``make test-live`` select what their lanes provide.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
from pathlib import Path

import pytest

from jammi.testing import LiveServer


def _server_bin() -> str:
    """The server binary a selected live test runs, or a failure naming what
    is missing."""
    server_bin = os.environ.get("JAMMI_SERVER_BIN")
    if not server_bin or not Path(server_bin).is_file():
        pytest.fail(
            "a `live_server` test was selected but JAMMI_SERVER_BIN "
            f"({server_bin!r}) does not name a built jammi-server binary",
            pytrace=False,
        )
    return server_bin


@contextlib.contextmanager
def _server_on(artifact_dir, *, env_overrides=None):
    """A real `jammi-server` (CPU, all tiers) over `artifact_dir`, torn down on
    exit; yields its `grpc://` endpoint.

    `artifact_dir` may already CONTAIN a catalog a previous (embedded) process
    seeded and released; the server opens it like any other. That is what lets a
    parity test compare a remote read against an embedded read of the very same
    rows, rather than of two separately-built approximations of them.

    `env_overrides` are the deployment knobs a test needs the server to answer
    differently (e.g. `JAMMI_WORKER__ENABLED=false`, to hold a seeded job
    `queued` so a read is compared against a stable row rather than a moving
    one).
    """
    with LiveServer(artifact_dir, server_bin=_server_bin(), env=env_overrides) as server:
        yield server.endpoint


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real `jammi-server` over a fresh artifact dir; torn down at module
    exit. Yields the `grpc://127.0.0.1:<port>` endpoint."""
    with _server_on(tmp_path_factory.mktemp("jammi-srv")) as endpoint:
        yield endpoint


@pytest.fixture
def live_server_on():
    """Factory for a live server over a CALLER-CHOSEN artifact directory:
    ``with live_server_on(path) as endpoint:``.

    The module-scoped :func:`live_server` owns a fresh directory it created; a
    parity test that must seed the catalog through the embedded engine first
    (the single-process SQLite contract means the seeder has to have released
    the file before the server opens it) needs to hand the server that same
    directory instead."""
    return _server_on


@pytest.fixture
def no_embedded_engine(monkeypatch):
    """The client-only install, simulated: `find_spec("jammi_native")` misses,
    whether or not the engine is installed in this environment."""
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "jammi_native":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


@pytest.fixture(autouse=True)
def _embedded_tests_have_the_engine(request):
    """A selected `embedded` test without the engine fails naming it."""
    if request.node.get_closest_marker("embedded") and (
        importlib.util.find_spec("jammi_native") is None
    ):
        pytest.fail(
            "an `embedded` test was selected but `jammi_native` is not installed "
            "(`pip install jammi-ai[embedded]`)",
            pytrace=False,
        )
