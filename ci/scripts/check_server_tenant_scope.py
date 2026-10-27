#!/usr/bin/env python3
"""No server code writes a session's sticky tenant.

A served engine session is shared by every request, and so is its tenant
binding's sticky value: a request that wrote it would decide the tenant of
every other request that reads it. The Flight SQL lane once did, and two
tenants querying at once read each other's rows (#738). A request's tenant is
the scope the tenant-binding layer installs around that request alone. This
fails on any call in `jammi-server`'s sources that writes the sticky value —
`TenantBinding::set_shared` or a session verb over it.

    python3 ci/scripts/check_server_tenant_scope.py [--self-test]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = "crates/jammi-server/src"
STICKY_WRITE = re.compile(r"\.(set_shared|bind_tenant|unbind_tenant|with_tenant)\s*\(")


def writes(source: str) -> list[tuple[int, str]]:
    """Each `(line, call)` in `source`'s code (comments aside) that writes a sticky tenant."""
    return [
        (number, match.group(1))
        for number, line in enumerate(source.splitlines(), 1)
        for match in STICKY_WRITE.finditer(line.split("//", 1)[0])
    ]


def problems(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root)}:{number}: `{call}` writes the served session's sticky tenant; "
        "a request's tenant is the scope the tenant-binding layer installs around it"
        for path in sorted((root / SOURCES).rglob("*.rs"))
        for number, call in writes(path.read_text())
    ]


def self_test() -> int:
    assert writes("        self.binding.set_shared(ctx);") == [(1, "set_shared")]
    assert writes("s.bind_tenant(t);\ns.unbind_tenant();") == [(1, "bind_tenant"), (2, "unbind_tenant")]
    assert writes("let s = s.with_tenant(t);") == [(1, "with_tenant")]
    assert writes("s.with_tenant_scoped(t, |scope| f(scope))") == []
    assert writes("// self.binding.set_shared(ctx);") == []
    assert writes("/// never calls `TenantBinding::set_shared`") == []
    assert writes("binding.scope(ctx, f); // not .set_shared(ctx)") == []
    print("check_server_tenant_scope: self-test passed")
    return 0


def main() -> int:
    if sys.argv[1:] == ["--self-test"]:
        return self_test()
    found = problems(REPO_ROOT)
    for problem in found:
        print(f"FAIL {problem}")
    if not found:
        print(f"check_server_tenant_scope: no code under {SOURCES} writes a sticky tenant")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
