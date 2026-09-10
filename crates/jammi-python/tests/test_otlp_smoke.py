"""Python smoke test for #486 OTLP wiring in `jammi.connect`'s embedded arm.

The deep proof (the tracing-layer composition `open_local` builds, feature
gate refusal, and the real stub-collector round trip) lives on the Rust side
— `crates/jammi-ai/src/telemetry.rs`'s own tests, `crates/jammi-ai/tests/it/
telemetry_otlp.rs`, and `crates/jammi-python/tests/it.rs`'s
`build_tracing_layers_*` tests — because `open_local`'s process-global
`try_init()` can only ever succeed once per process, which a `pytest` run (one
process, many tests) cannot exercise repeatably per-scenario. This file only
proves the embed wheel's Python-visible contract: `jammi.connect` with an
`[observability]` section configured (via the `JAMMI_OBSERVABILITY__*` env
namespace, same as the server) still connects and serves ordinary requests —
no crash, no hang — and a malformed value is a typed connection failure a
Python caller can catch, not a silent no-op.
"""

import socket

import pytest

import jammi


def _free_loopback_endpoint() -> str:
    """Bind an ephemeral loopback port and return it as an `http://` OTLP
    endpoint URL, then release it. Nothing needs to actually be listening:
    `otlp_layer` builds a LAZY tonic channel (no connection attempt at
    connect time either way), and the point here is only that `connect()`
    accepts a well-formed endpoint and does not hang or crash constructing
    the exporter around it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_connect_with_an_otlp_endpoint_configured_still_serves(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMMI_OBSERVABILITY__OTLP_ENDPOINT", _free_loopback_endpoint())
    monkeypatch.setenv("JAMMI_OBSERVABILITY__SERVICE_NAME", "jammi-python-smoke")

    db = jammi.connect(f"file://{tmp_path}")
    # An ordinary embedded-session operation still works: the OTLP layer is
    # additive, never a gate on the engine's own request path.
    assert db.get_server_info() is not None


def test_connect_with_an_out_of_range_sample_ratio_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JAMMI_OBSERVABILITY__SAMPLE_RATIO", "1.5")

    with pytest.raises(Exception, match="sample_ratio"):
        jammi.connect(f"file://{tmp_path}")
