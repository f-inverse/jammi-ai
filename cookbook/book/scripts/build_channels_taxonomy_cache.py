#!/usr/bin/env python3
"""Emit the channel error-taxonomy cache (§3.8) — CPU, dual-transport, hermetic.

The engine↔cookbook validator for the `§3.8` channel error taxonomy (engine
`#193`): each evidence-channel failure maps to its **correct typed gRPC status
code** on the wire, instead of the `Internal`-for-everything that a thin
catch-all would produce. The channel registry verbs (`register_channel` /
`add_channel_columns` / `list_channels`) are on BOTH the embedded `jammi_ai`
engine and the remote `jammi_client.RemoteDatabase` (chapter 14 exercises the
happy path); this script MEASURES the FAILURE surface — the `(failure_mode →
status_code)` matrix — on the `grpc://` transport, where the status codes only
exist on the wire, with the embedded error class as the cross-transport
companion.

It drives each failure mode on BOTH transports and freezes the matrix:

* **duplicate** — registering a channel id that already exists → `ALREADY_EXISTS`.
* **unknown** — `add_channel_columns` on a channel that was never registered →
  `NOT_FOUND`.
* **column conflict** — re-adding an existing column with a DIFFERENT dtype
  (the append-only redeclare violation) → `FAILED_PRECONDITION`.
* **bad argument** — an empty channel id → `INVALID_ARGUMENT`.

A genuine **internal/DB fault** (`INTERNAL`) is the documented RESIDUAL of the
taxonomy: there is no honest, hermetic way to induce a real storage fault from
the public surface, so it is recorded as the documented residual and NOT
fabricated — the whole point of `#193` is that a failure with a *known* cause no
longer collapses to `Internal`, so the cookbook does not manufacture one.

Each failure mode's measured cell is the `(wire_code, embedded_error_class)`
pair. The two transports raise different Python exception TYPES by construction:
the embedded engine raises `RuntimeError` (a catalog error) or `ValueError` (a
client-side argument guard); the gRPC client raises `grpc.RpcError` carrying a
`StatusCode`. So the honest cross-transport observable for a typed error is the
NORMALIZED error CLASS, derived from each native error — `duplicate`,
`unknown`, `conflict`, `bad_argument`. The matrix records, per mode, the wire
`StatusCode` name AND the normalized class for both transports, and asserts they
agree on the class live. The `(mode → wire_code)` taxonomy freezes to
`golden_metrics.json`; the committed matrix + record + `checksums.json` to
`artifacts/channels/`.

NOTE on the `INVALID_ARGUMENT` cell: an invalid *dtype string* (e.g. a column
typed `"NotAType"`) is rejected CLIENT-SIDE on both transports — a `ValueError`
that never reaches the wire — so it carries no `StatusCode`. The honest wire
`INVALID_ARGUMENT` cell is therefore an empty channel id, which the SERVER
rejects and maps to `INVALID_ARGUMENT`. The client-side dtype guard is recorded
separately as a measured property (validation that never hits the wire).

Usage::

    JAMMI_SERVER_BIN=/mnt/sagemaker-nvme/jammi-target/debug/jammi-server \\
        python scripts/build_channels_taxonomy_cache.py

This is an emit-only script: it imports a build of the engine wheel + client and
a live CPU `jammi-server`; PR CI never runs it (it reads the committed cache).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import jammi_ai
import jammi_client

import jammi_cookbook  # noqa: F401  # applies the determinism env on import

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "channels"

# The four headline failure modes of the channel registry and the gRPC status
# code each maps to under `#193` (the typed taxonomy that replaced
# Internal-for-everything). The emit asserts the measured wire code equals the
# expected one live; a deviation is recorded, never silently rewritten.
_EXPECTED_WIRE = {
    "duplicate": "ALREADY_EXISTS",
    "unknown": "NOT_FOUND",
    "column_conflict": "FAILED_PRECONDITION",
    "bad_argument": "INVALID_ARGUMENT",
}

# The normalized cross-transport error CLASS each mode maps to — derived from each
# native error (the wire StatusCode and the embedded message), so the two
# transports' different Python exception TYPES collapse onto one comparable class.
_EXPECTED_CLASS = {
    "duplicate": "duplicate",
    "unknown": "unknown",
    "column_conflict": "conflict",
    "bad_argument": "bad_argument",
}


# --------------------------------------------------------------------------- #
# Error normalisation — the honest cross-transport contract for a typed error
# --------------------------------------------------------------------------- #


def _wire_code(exc: BaseException) -> str | None:
    """The gRPC `StatusCode` NAME a remote error carries, or None for an error
    that never reached the wire (a client-side guard raises a plain Python
    exception with no `code()`)."""
    code = getattr(exc, "code", None)
    if callable(code):  # grpc.RpcError → StatusCode
        status = code()
        return getattr(status, "name", str(status))
    return None


def _classify(exc: BaseException) -> str:
    """Normalise a transport-native channel error to its CLASS.

    Derived from each native error so the two transports' different exception
    TYPES (embedded `RuntimeError`/`ValueError`, wire `grpc.RpcError`) collapse
    onto one comparable class. Prefers the wire `StatusCode` when present, else
    reads the message. An unrecognised error is returned verbatim so a real
    divergence surfaces loudly rather than being silently bucketed."""
    wire = _wire_code(exc)
    if wire is not None:
        return {
            "ALREADY_EXISTS": "duplicate",
            "NOT_FOUND": "unknown",
            "FAILED_PRECONDITION": "conflict",
            "INVALID_ARGUMENT": "bad_argument",
        }.get(wire, f"wire:{wire}")
    msg = str(exc).lower()
    if "already exists" in msg:
        return "duplicate"
    if "not registered" in msg or "not found" in msg:
        return "unknown"
    if "cannot redeclare" in msg:
        return "conflict"
    if "is required" in msg or "must be one of" in msg:
        return "bad_argument"
    return f"embed:{type(exc).__name__}:{msg[:60]}"


def _capture(fn) -> dict:
    """Run a failing channel call and capture the OUTCOME as a comparable
    observable: the normalized error class, the wire StatusCode name (or None
    for a client-side guard), and the native exception type. Never lets the
    exception escape — the outcome IS the measurement (the matrix cell)."""
    try:
        fn()
        return {"raised": False, "error_class": None, "wire_code": None, "native_type": None}
    except Exception as exc:  # noqa: BLE001 — the error class is the observable
        return {
            "raised": True,
            "error_class": _classify(exc),
            "wire_code": _wire_code(exc),
            "native_type": type(exc).__name__,
        }


# --------------------------------------------------------------------------- #
# The failure drive (one transport)
# --------------------------------------------------------------------------- #


def run_taxonomy(db, tenant: str, tag: str) -> dict:
    """Drive each channel failure mode on one transport under an explicit tenant
    scope, returning the observable matrix. The SAME call shapes run on both
    transports — the verb surface is transport-agnostic; only the native error
    TYPE differs (normalised away).

    `tag` keeps the registered channel ids disjoint on the module-shared remote
    server (the embedded peer is isolated by its own temp catalog)."""
    base = f"base_{tag}"
    ghost = f"ghost_{tag}"
    matrix: dict[str, dict] = {}
    with db.tenant_scope(tenant):
        # A real channel to collide / conflict against.
        db.register_channel(base, priority=10, columns=[("a", "Utf8")])

        # duplicate — register an id that already exists → ALREADY_EXISTS.
        matrix["duplicate"] = _capture(
            lambda: db.register_channel(base, priority=20, columns=[("b", "Utf8")])
        )
        # unknown — add columns to a channel never registered → NOT_FOUND.
        matrix["unknown"] = _capture(
            lambda: db.add_channel_columns(ghost, columns=[("x", "Utf8")])
        )
        # column conflict — re-add an existing column with a DIFFERENT dtype
        # (the append-only redeclare violation) → FAILED_PRECONDITION.
        matrix["column_conflict"] = _capture(
            lambda: db.add_channel_columns(base, columns=[("a", "Float64")])
        )
        # bad argument — an empty channel id → INVALID_ARGUMENT (server-side).
        matrix["bad_argument"] = _capture(
            lambda: db.register_channel("", priority=1, columns=[("a", "Utf8")])
        )

        # Client-side dtype guard: an invalid dtype STRING is rejected before the
        # wire on both transports (a ValueError, no StatusCode) — recorded as a
        # measured property, not a wire cell.
        client_side_dtype_guard = _capture(
            lambda: db.register_channel(
                f"badtype_{tag}", priority=1, columns=[("a", "NotAType")]
            )
        )
    return {"matrix": matrix, "client_side_dtype_guard": client_side_dtype_guard}


# --------------------------------------------------------------------------- #
# Parity (remote == embedded on the normalized CLASS)
# --------------------------------------------------------------------------- #


def _parity(name: str, embedded: dict, remote: dict) -> dict:
    """The live cross-transport parity verdict for one failure mode: the
    normalized error CLASS must agree (the two transports raise different native
    types; the honest observable is the class). Raises loudly on a real
    divergence — an engine bug the validator surfaces, never papers over."""
    e_class = embedded["error_class"]
    r_class = remote["error_class"]
    equal = e_class == r_class and embedded["raised"] == remote["raised"]
    verdict = {
        "mode": name,
        "class_equal": equal,
        "embedded_class": e_class,
        "remote_class": r_class,
        "embedded_native": embedded["native_type"],
        "remote_native": remote["native_type"],
        "wire_code": remote["wire_code"],
    }
    if not equal:
        raise AssertionError(
            f"remote != embedded class for {name}: embedded={e_class!r} remote={r_class!r}"
        )
    return verdict


# --------------------------------------------------------------------------- #
# Remote server lifecycle (mirrors the engine conftest live_server)
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LiveServer:
    """A real CPU `jammi-server` on a free port, readiness-polled via a client
    handshake, torn down on exit — the same shape the engine's conftest uses."""

    def __init__(self, server_bin: str):
        self.server_bin = server_bin
        self.proc = None
        self.endpoint = None
        self._artifact_dir = None

    def __enter__(self) -> str:
        self._artifact_dir = tempfile.mkdtemp(prefix="jammi_srv_channels_tax_")
        flight_port = _free_port()
        health_port = _free_port()
        env = dict(os.environ)
        env["JAMMI_ARTIFACT_DIR"] = self._artifact_dir
        env["JAMMI_SERVER__FLIGHT_LISTEN"] = f"127.0.0.1:{flight_port}"
        env["JAMMI_SERVER__HEALTH_LISTEN"] = f"127.0.0.1:{health_port}"
        env["JAMMI_SERVER__SERVICES"] = "all"
        self.proc = subprocess.Popen(
            [self.server_bin], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        self.endpoint = f"grpc://127.0.0.1:{flight_port}"
        deadline = time.time() + 30
        while time.time() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read().decode(errors="replace") if self.proc.stdout else ""
                raise RuntimeError(f"jammi-server exited early:\n{out}")
            try:
                handshake = jammi_client.connect(self.endpoint)
                handshake.get_server_info()
                handshake.close()
                return self.endpoint
            except Exception:
                time.sleep(0.25)
        self.proc.terminate()
        raise RuntimeError("jammi-server did not become ready within 30s")

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checksums() -> None:
    sums = {
        p.name: _checksum(p)
        for p in sorted(ARTIFACTS.glob("*"))
        if p.is_file() and p.name != "checksums.json"
    }
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


_MODES = ("duplicate", "unknown", "column_conflict", "bad_argument")


def emit(server_bin: str) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Stable tenant UUIDs (the matrix carries no tenant value; the scopes only need
    # to be reproducible on a re-emit).
    tenant_a = "11111111-1111-4111-8111-111111111111"
    tenant_b = "22222222-2222-4222-8222-222222222222"

    # --- embedded transport (the error-class companion) --------------------- #
    with tempfile.TemporaryDirectory() as catalog:
        embedded = jammi_ai.connect(f"file://{catalog}")
        print("== embedded engine: channel error taxonomy ==", flush=True)
        embedded_run = run_taxonomy(embedded, tenant_a, tag="emb")

    # --- remote transport (live grpc:// — the typed wire codes) ------------- #
    with LiveServer(server_bin) as endpoint:
        print(f"== remote engine up at {endpoint} ==", flush=True)
        remote = jammi_client.connect(endpoint)
        try:
            print("== remote engine: channel error taxonomy ==", flush=True)
            remote_run = run_taxonomy(remote, tenant_b, tag="rem")
        finally:
            remote.close()

    embedded_matrix = embedded_run["matrix"]
    remote_matrix = remote_run["matrix"]

    # --- assert each mode maps to its expected wire code + class LIVE -------- #
    deviations = []
    for mode in _MODES:
        rem_cell = remote_matrix[mode]
        assert rem_cell["raised"] is True, f"{mode}: expected a raise on the wire"
        measured_wire = rem_cell["wire_code"]
        expected_wire = _EXPECTED_WIRE[mode]
        if measured_wire != expected_wire:
            deviations.append(
                f"{mode}: measured wire {measured_wire} != #193-intended {expected_wire}"
            )
        # the embedded companion carries the same normalized class (no wire code).
        assert embedded_matrix[mode]["error_class"] == _EXPECTED_CLASS[mode], (
            f"{mode}: embedded class {embedded_matrix[mode]['error_class']} "
            f"!= {_EXPECTED_CLASS[mode]}"
        )
    if deviations:
        # Surface a candidate engine finding loudly — never rewrite the code to pass.
        raise AssertionError(
            "channel taxonomy DEVIATION from #193 intent (record as an engine finding, "
            "do NOT fake the code):\n  " + "\n  ".join(deviations)
        )
    print("== each failure mode maps to its #193-intended wire code ==", flush=True)

    # --- the live remote == embedded class parity --------------------------- #
    parity = [_parity(mode, embedded_matrix[mode], remote_matrix[mode]) for mode in _MODES]
    print(f"== remote == embedded class parity: PASS for all {len(parity)} modes ==", flush=True)

    # the client-side dtype guard is identical on both transports (a ValueError,
    # never the wire) — a measured property, asserted here.
    e_guard = embedded_run["client_side_dtype_guard"]
    r_guard = remote_run["client_side_dtype_guard"]
    client_dtype_guard_both_client_side = (
        e_guard["raised"] and r_guard["raised"]
        and e_guard["wire_code"] is None and r_guard["wire_code"] is None
        and e_guard["error_class"] == r_guard["error_class"] == "bad_argument"
    )

    # --- the frozen golden taxonomy (the measured (mode → code) verdicts) ---- #
    golden = {}
    for mode in _MODES:
        measured_wire = remote_matrix[mode]["wire_code"]
        golden[f"taxonomy.{mode}_wire_code_matches"] = {
            "value": 1.0 if measured_wire == _EXPECTED_WIRE[mode] else 0.0, "tol": 0.0
        }
        golden[f"taxonomy.{mode}_class_parity"] = {
            "value": 1.0 if embedded_matrix[mode]["error_class"]
            == remote_matrix[mode]["error_class"] else 0.0, "tol": 0.0
        }
    golden["taxonomy.all_modes_typed_not_internal"] = {
        "value": 1.0 if all(
            remote_matrix[m]["wire_code"] not in (None, "INTERNAL", "UNKNOWN") for m in _MODES
        ) else 0.0, "tol": 0.0
    }
    golden["taxonomy.client_dtype_guard_client_side"] = {
        "value": 1.0 if client_dtype_guard_both_client_side else 0.0, "tol": 0.0
    }
    golden["parity.all_modes_class_equal"] = {
        "value": 1.0 if all(p["class_equal"] for p in parity) else 0.0, "tol": 0.0
    }

    # --- commit the embedded-canonical + remote matrices -------------------- #
    committed_matrix = {
        "modes": _MODES,
        "expected_wire": _EXPECTED_WIRE,
        "expected_class": _EXPECTED_CLASS,
        "embedded": embedded_matrix,
        "remote": remote_matrix,
        "client_side_dtype_guard": {"embedded": e_guard, "remote": r_guard},
    }
    (ARTIFACTS / "matrix.json").write_text(
        json.dumps(committed_matrix, indent=2, sort_keys=True)
    )
    (ARTIFACTS / "golden_metrics.json").write_text(json.dumps(golden, indent=2, sort_keys=True))

    record = {
        "purpose": (
            "The engine↔cookbook validator for the §3.8 channel error taxonomy (engine "
            "#193): each evidence-channel failure maps to its CORRECT typed gRPC status "
            "code on the wire, not Internal-for-everything. The channel registry verbs "
            "(register_channel / add_channel_columns / list_channels) run the failure "
            "modes on BOTH the embedded engine and a live remote grpc:// jammi-server; "
            "the wire StatusCode is measured on the grpc:// transport (where the codes "
            "exist), with the embedded normalized error CLASS as the cross-transport "
            "companion. PR CI reads the committed matrix and asserts the (mode → code) "
            "taxonomy-to-golden, never a live re-drive."
        ),
        "taxonomy": {mode: remote_matrix[mode]["wire_code"] for mode in _MODES},
        "expected_taxonomy": _EXPECTED_WIRE,
        "deviations": deviations,  # empty == every mode maps as #193 intended
        "transports": [
            "embedded (file://, in-process — the error-class companion)",
            "remote (grpc://, live jammi-server — the typed wire codes)",
        ],
        "parity_observable": (
            "the NORMALIZED channel-error CLASS (embedded RuntimeError/ValueError message "
            "/ wire StatusCode → duplicate | unknown | conflict | bad_argument) — the "
            "honest cross-transport contract for a typed error."
        ),
        "parity_verdict": "remote == embedded class for every channel failure mode",
        "parity": parity,
        "internal_residual": (
            "INTERNAL is the documented residual of the taxonomy: there is no honest, "
            "hermetic way to induce a genuine storage/DB fault from the public channel "
            "surface, so it is NOT fabricated. The point of #193 is that a failure with a "
            "KNOWN cause no longer collapses to Internal — the four typed codes above are "
            "the measured proof; INTERNAL remains the bucket for a true unexpected fault."
        ),
        "client_side_dtype_guard": (
            "An invalid column dtype STRING is rejected CLIENT-SIDE on both transports (a "
            "ValueError that never reaches the wire — no StatusCode), so the honest wire "
            "INVALID_ARGUMENT cell is an empty channel id (server-rejected). The dtype "
            "guard is recorded as a measured property: validation that never hits the wire."
        ),
        "measured": {
            mode: {
                "wire_code": remote_matrix[mode]["wire_code"],
                "embedded_class": embedded_matrix[mode]["error_class"],
                "embedded_native": embedded_matrix[mode]["native_type"],
                "remote_native": remote_matrix[mode]["native_type"],
            }
            for mode in _MODES
        },
        "client_dtype_guard_client_side": client_dtype_guard_both_client_side,
    }
    (ARTIFACTS / "channels_taxonomy.json").write_text(json.dumps(record, indent=2, sort_keys=True))

    _write_checksums()

    print("\n=== channel error taxonomy, measured (mode → wire code) ===", flush=True)
    for mode in _MODES:
        print(
            f"  {mode:16s} → {remote_matrix[mode]['wire_code']:20s} "
            f"(embedded class {embedded_matrix[mode]['error_class']!r}, "
            f"#193-intended {_EXPECTED_WIRE[mode]})",
            flush=True,
        )
    print(f"  client-side dtype guard (ValueError, never the wire): "
          f"both={client_dtype_guard_both_client_side}", flush=True)
    print(f"  parity: remote == embedded class for all {len(parity)} modes", flush=True)
    print(f"  deviations from #193: {deviations or 'none — every mode maps as intended'}",
          flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", default=os.environ.get("JAMMI_SERVER_BIN"),
                    help="built CPU jammi-server binary (or set JAMMI_SERVER_BIN)")
    args = ap.parse_args()
    if not args.server_bin or not os.path.exists(args.server_bin):
        raise SystemExit("pass --server-bin (or set JAMMI_SERVER_BIN) to a built jammi-server")
    emit(args.server_bin)


if __name__ == "__main__":
    main()
