"""The compiled native engine, importable as a top-level module.

`jammi_native` is the in-process Jammi engine — the maturin-built PyO3
extension. It is a bare top-level import so a consumer can discover the local
backend on its own, without going through the `jammi_ai` convenience package.

The compiled extension is shipped inside this package (maturin's mixed
Rust/Python layout places it beside this file, so the pure-Python `jammi_ai`
package can ride the same wheel via `python-source`). This module re-exports the
extension's public surface so `import jammi_native` yields the engine directly.
The re-export is explicit — a bare ``import *`` would drop the underscore-named
`_NativeDatabase` low-level handle.
"""

from .jammi_native import (
    AuditHandle,
    EphemeralSession,
    ModelTask,
    PerQueryAudit,
    TrainingJob,
    _NativeDatabase,
    open_local,
)

__all__ = [
    "AuditHandle",
    "EphemeralSession",
    "ModelTask",
    "PerQueryAudit",
    "TrainingJob",
    "_NativeDatabase",
    "open_local",
]
