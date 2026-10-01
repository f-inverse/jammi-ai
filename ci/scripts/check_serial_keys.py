#!/usr/bin/env python3
"""Every test lock is classified for a runner that gives each test a process.

`cargo test` runs a binary's tests as threads of one process, so a
`serial_test` key or a hand-written `static ... Mutex<()>` keeps two of them
apart. nextest runs every test in a process of its own (`.config/nextest.toml`):
a lock guarding state of the process — a counter, an armed hook, an
environment variable — is moot there, and a lock guarding a resource of the
host — CPU under a wall-clock deadline, an on-disk cache, a device — no longer
excludes anything. So every key is either a nextest test group of the same
name or named process-local here, and every hand-written lock is named here
with what it guards. An unclassified one, or an entry naming nothing that
exists, is a finding.

    python3 ci/scripts/check_serial_keys.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NEXTEST_CONFIG = ".config/nextest.toml"
SOURCE_ROOTS = ("crates", "ci/tools")

# `serial_test` keys guarding state of the process.
PROCESS_LOCAL_KEYS = {
    "acceleration_report": "the process-wide kernel admission dispatch registries",
    "grad_clip_sync_read_count": "a process-wide counter of host reads",
    "tokenize_dispatch_calls": "a process-wide counter of tokenizer dispatches",
    "trainer_host_read_count": "a process-wide counter of host reads",
    "materialization_park": "a test hook parked process-wide",
    "preload_park": "a test hook parked process-wide",
}

# Hand-written test locks, by `<path>::<static>`, with what each guards. Only
# the GPU device is a resource of the host, and the GPU lanes run each suite
# under `cargo test` on a pod, never under nextest.
LOCKS = {
    "crates/jammi-ai/tests/gpu_capability/harness.rs::GPU_SERIAL": "host: the pod's GPU, under cargo test",
    "crates/jammi-ai/tests/gpu_capability/harness.rs::ADMISSION_COUNTER_SERIAL": "process: an admission counter",
    "crates/jammi-bench/src/finetune_step.rs::CLIP_COUNTER_SERIAL": "process: the clip-invocation counter",
    "crates/jammi-db/src/audit/key_store.rs::ENV_LOCK": "process: JAMMI_AUDIT_MASTER_KEY",
    "crates/jammi-db/tests/it/audit.rs::ENV_LOCK": "process: JAMMI_AUDIT_MASTER_KEY",
    "crates/jammi-encoders/src/modernbert.rs::FLASH_D2H_TEST_LOCK": "process: the flash device-to-host sync counter",
    "crates/jammi-encoders/src/test_support.rs::SEAM_COUNTER_TEST_LOCK": "process: the attention dispatch counters",
    "crates/jammi-encoders/tests/eager_training_memory.rs::GPU_SERIAL": "host: the pod's GPU, under cargo test",
    "crates/jammi-encoders/tests/it/modernbert.rs::DISPATCH_COUNTER_TEST_LOCK": "process: the dispatch counters",
    "crates/jammi-lora/src/lora_linear.rs::LOCK": "process: the LoRA dispatch counters",
    "crates/jammi-lora/tests/fused_epilogue.rs::DISPATCH_COUNTER_PAIR_LOCK": "process: the epilogue dispatch counters",
    "crates/jammi-server/src/runtime.rs::LOCK": "process: JAMMI_AUDIT_MASTER_KEY",
    "crates/jammi-server/tests/it/grpc_mutable_topic_audit.rs::ENV_LOCK": "process: JAMMI_AUDIT_MASTER_KEY",
    "crates/jammi-server/tests/it/grpc_remote_session.rs::LOCK": "process: JAMMI_AUDIT_MASTER_KEY",
    "crates/jammi-test-utils/src/child.rs::SPAWN_LOCK": "process: the window a spawned child inherits descriptors in",
}

_KEY_ATTR = re.compile(r"#\[\s*(?:serial_test::)?(?:serial|file_serial|parallel)\s*(?:\((?P<keys>[^)]*)\))?\s*\]")
_LOCK = re.compile(r"static\s+(?P<name>[A-Z_][A-Z0-9_]*)\s*:\s*[^=;]*Mutex<\(\)>")


def sources(root: Path):
    for top in SOURCE_ROOTS:
        yield from sorted((root / top).rglob("*.rs"))


def scan(root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """`({key: [where, ...]}, {<path>::<static>: where})` across the sources."""
    keys: dict[str, list[str]] = {}
    locks: dict[str, str] = {}
    for path in sources(root):
        rel = path.relative_to(root).as_posix()
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("//"):
                continue
            for m in _KEY_ATTR.finditer(line):
                names = [k.strip() for k in (m.group("keys") or "").split(",") if k.strip()] or [""]
                for key in names:
                    keys.setdefault(key, []).append(f"{rel}:{n}")
            for m in _LOCK.finditer(line):
                locks[f"{rel}::{m.group('name')}"] = f"{rel}:{n}"
    return keys, locks


def test_groups(root: Path) -> set[str]:
    with open(root / NEXTEST_CONFIG, "rb") as f:
        return set(tomllib.load(f).get("test-groups", {}))


def check(root: Path = REPO_ROOT) -> list[str]:
    keys, locks = scan(root)
    groups = test_groups(root)
    findings: list[str] = []
    for key, sites in sorted(keys.items()):
        if key == "":
            findings.append(f"{sites[0]}: a keyless serial attribute: name what it guards, so it can be classified")
        elif key not in groups and key not in PROCESS_LOCAL_KEYS:
            findings.append(
                f"{sites[0]}: serial key `{key}` is neither a test group in {NEXTEST_CONFIG} (a resource "
                f"of the host) nor in PROCESS_LOCAL_KEYS (state of the process)"
            )
    for key in sorted(set(PROCESS_LOCAL_KEYS) - set(keys)):
        findings.append(f"PROCESS_LOCAL_KEYS names `{key}`, which no test carries")
    for key in sorted(set(PROCESS_LOCAL_KEYS) & groups):
        findings.append(f"`{key}` is both a test group and process-local")
    for lock, where in sorted(locks.items()):
        if lock not in LOCKS:
            findings.append(f"{where}: test lock `{lock}` is not classified in LOCKS")
    for lock in sorted(set(LOCKS) - set(locks)):
        findings.append(f"LOCKS names `{lock}`, which does not exist")
    return findings


def main() -> int:
    findings = check()
    for f in findings:
        print(f"::error::serial keys: {f}", file=sys.stderr)
    if not findings:
        print("serial keys: every serial_test key and test lock is classified")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
