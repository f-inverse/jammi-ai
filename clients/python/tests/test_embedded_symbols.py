"""`jammi`'s lazily-resolved embedded-only types.

They resolve through the package-level `__getattr__`, so none is imported at
`import jammi` — the native-free direction the conformance lane pins. These stay
hermetic: the client-only arm monkeypatches `find_spec` to simulate the engine's
absence.
"""

from __future__ import annotations


import pytest

import jammi
from jammi import NoEmbeddedEngineError


def test_unknown_attribute_raises_attribute_error():
    """A genuinely unknown name falls through `__getattr__` to `AttributeError`,
    so normal attribute semantics (and `hasattr`) hold."""
    with pytest.raises(AttributeError):
        _ = jammi.does_not_exist
    assert not hasattr(jammi, "does_not_exist")


def test_dir_includes_lazy_names():
    """`__dir__` surfaces the lazily-resolved names for introspection /
    tab-completion alongside the eagerly-bound surface."""
    names = dir(jammi)
    for name in ("AuditHandle", "EphemeralSession", "Job", "ModelTask",
                 "PerQueryAudit"):
        assert name in names


def test_embedded_symbol_without_engine_raises(no_embedded_engine):
    """In the client-only build an embedded-only symbol is a truthful
    `NoEmbeddedEngineError` naming both the attribute and the `[embedded]`
    extra — never a bare `ImportError`."""
    with pytest.raises(NoEmbeddedEngineError) as info:
        _ = jammi.PerQueryAudit
    msg = str(info.value)
    assert "PerQueryAudit" in msg
    assert "pip install jammi-ai[embedded]" in msg


@pytest.mark.embedded
def test_embedded_symbol_with_engine_returns_native_type():
    """With the engine importable, an embedded-only accessor returns the very
    type `jammi_native` exports — the lazy re-export, not a copy."""
    import jammi_native

    assert jammi.PerQueryAudit is jammi_native.PerQueryAudit
    assert jammi.Job is jammi_native.Job
