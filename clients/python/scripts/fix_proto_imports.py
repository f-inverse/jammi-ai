"""Rewrite grpc_tools.protoc's package-style imports into relative imports.

`python -m grpc_tools.protoc --python_out=jammi_client/_generated` emits
modules with `from jammi.v1 import inference_pb2` (and the `_pb2_grpc`
companions emit `from jammi.v1 import embedding_pb2 as ...`). When those
files are imported as part of `jammi_client._generated.jammi.v1.…`, Python
looks for a top-level `jammi` package — which does not exist — and the
import fails.

Every jammi.v1 stub lives at the same `jammi/v1/<name>_pb2.py` level and only
ever cross-references siblings at that same level (that is the only
absolute-import shape protoc emits for this flat proto tree), so the rewrite is
always to a single-dot, same-package relative import: `from . import
inference_pb2`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Matches `from jammi.v1 import <module>` and the `import … as <alias>` form the
# gRPC plugin emits. Capture the module being pulled in and the optional alias
# so the rewrite preserves the alias verbatim (the alias name is what the rest
# of the generated file actually references).
_IMPORT_RE = re.compile(
    r"^from\s+jammi\.v1\s+import\s+(?P<mod>\w+)(?P<alias>\s+as\s+\w+)?\s*$",
    flags=re.MULTILINE,
)


def _assert_flat_layout(path: Path, root: Path) -> None:
    """Guard the single-level assumption the rewrite depends on.

    Generated stubs sit at `jammi/v1/<name>_pb2.py`; a sibling import is always
    a same-package single-dot relative import. If someone reorganises the tree
    so a stub lives at a different depth, this catches it instead of emitting a
    wrong number of dots.
    """
    rel = path.parent.relative_to(root).parts
    if rel != ("jammi", "v1"):
        raise RuntimeError(
            f"unexpected generated layout for {path}: parent is {rel!r}, "
            "expected jammi/v1/"
        )


def _rewrite_one(path: Path, root: Path) -> bool:
    _assert_flat_layout(path, root)
    text = path.read_text()

    def _sub(m: re.Match[str]) -> str:
        alias = m.group("alias") or ""
        return f"from . import {m.group('mod')}{alias}"

    new = _IMPORT_RE.sub(_sub, text)
    if new == text:
        return False
    path.write_text(new)
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: fix_proto_imports.py <generated-root>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    changed = 0
    for ext in (".py", ".pyi"):
        for path in root.rglob(f"*_pb2{ext}"):
            if _rewrite_one(path, root):
                changed += 1
        for path in root.rglob(f"*_pb2_grpc{ext}"):
            if _rewrite_one(path, root):
                changed += 1
    print(f"fix_proto_imports: rewrote {changed} file(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
