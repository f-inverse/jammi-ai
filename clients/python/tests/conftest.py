"""Shared fixtures for the live remote round-trip tests.

The live tests stand up a real CPU `jammi-server` and drive verbs through the
pure-Python `RemoteDatabase` against an embedded `jammi_ai.Database` parity
peer. The server fixture lives here so every live module shares one
implementation; each module still declares its own `pytest.mark.skipif` gate
on `JAMMI_SERVER_BIN` so a bare `pytest` reports a loud per-module skip.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time

import pytest

import jammi_client

SERVER_BIN = os.environ.get("JAMMI_SERVER_BIN")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """A real `jammi-server` (CPU, all tiers) on a free port; torn down at module
    exit. Yields the `grpc://127.0.0.1:<port>` endpoint."""
    artifact_dir = tmp_path_factory.mktemp("jammi-srv")
    flight_port = _free_port()
    health_port = _free_port()
    env = dict(os.environ)
    env["JAMMI_ARTIFACT_DIR"] = str(artifact_dir)
    env["JAMMI_SERVER__FLIGHT_LISTEN"] = f"127.0.0.1:{flight_port}"
    env["JAMMI_SERVER__HEALTH_LISTEN"] = f"127.0.0.1:{health_port}"
    env["JAMMI_SERVER__SERVICES"] = "all"

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
            db = jammi_client.connect(endpoint)
            db.get_server_info()
            db.close()
            ready = True
            break
        except Exception:
            time.sleep(0.25)
    if not ready:
        proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    yield endpoint

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
