"""The compiled native engine, importable as a top-level module.

`jammi_native` is the in-process Jammi engine — the maturin-built PyO3
extension. It is a bare top-level import so the base client (`jammi`) can
discover the local backend on its own, on demand.

The compiled extension is shipped inside this package (maturin's mixed
Rust/Python layout places it beside this file). This module re-exports every
name the extension defines, the underscore-named `_NativeDatabase` handle
included, so the package's surface is exactly the extension's: a function or
class the extension registers is importable here with no second list to keep
in step.
"""

from . import jammi_native as _compiled

__all__ = [name for name in dir(_compiled) if not name.startswith("__")]
globals().update({name: getattr(_compiled, name) for name in __all__})
