"""Shared fixtures for the live remote round-trip tests.

The live tests stand up a real CPU `jammi-server` and drive verbs through the
pure-Python `RemoteDatabase` against an embedded `jammi.EmbeddedBackend` parity
peer. The server fixture lives here so every live module shares one
implementation; each module still declares its own `pytest.mark.skipif` gate
on `JAMMI_SERVER_BIN` so a bare `pytest` reports a loud per-module skip.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import time

import pytest

import jammi

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _server_on(artifact_dir, *, env_overrides=None):
    """A real `jammi-server` (CPU, all tiers) on a free port over
    `artifact_dir`, torn down on exit. The one implementation both the
    module-scoped :func:`live_server` and the :func:`live_server_on` factory
    use, so "how a live server is started" is stated once.

    `artifact_dir` may already CONTAIN a catalog a previous (embedded) process
    seeded and released; the server opens it like any other. That is what lets a
    parity test compare a remote read against an embedded read of the very same
    rows, rather than of two separately-built approximations of them.

    `env_overrides` are applied last, over this fixture's own `JAMMI_*` keys —
    the deployment knobs a test needs the server to answer differently, read by
    the server's `JammiConfig::load` exactly as an operator's would be (e.g.
    `JAMMI_TRAINING__RUN_WORKER=false`, to hold a seeded job `queued` so a read
    is compared against a stable row rather than a moving one).
    """
    flight_port = _free_port()
    health_port = _free_port()
    env = dict(os.environ)
    env["JAMMI_ARTIFACT_DIR"] = str(artifact_dir)
    env["JAMMI_SERVER__FLIGHT_LISTEN"] = f"127.0.0.1:{flight_port}"
    env["JAMMI_SERVER__HEALTH_LISTEN"] = f"127.0.0.1:{health_port}"
    env["JAMMI_SERVER__SERVICES"] = "all"
    env.update(env_overrides or {})

    proc = subprocess.Popen(
        [SERVER_BIN],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Poll readiness via a trivial RemoteDatabase handshake.
    endpoint = f"grpc://127.0.0.1:{flight_port}"
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(f"jammi-server exited early:\n{out}")
        try:
            db = jammi.connect(endpoint)
            db.get_server_info()
            db.close()
            ready = True
            break
        except Exception:
            time.sleep(0.25)
    if not ready:
        proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    try:
        yield endpoint
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real `jammi-server` (CPU, all tiers) on a free port over a fresh
    artifact dir; torn down at module exit. Yields the
    `grpc://127.0.0.1:<port>` endpoint."""
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
