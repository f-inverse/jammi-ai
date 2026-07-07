"""`jammi`'s lazily-resolved optional surfaces: embedded-only types + `platform`.

Both resolve through the package-level `__getattr__`, so neither is imported at
`import jammi` — the native-free / extension-free direction the conformance lane
pins. These stay hermetic: the platform arm monkeypatches the discovery
indirection (`jammi._load_extension`) so no entry point need be installed, and
the embedded arm monkeypatches `find_spec` to simulate the engine's absence.
"""

from __future__ import annotations

import importlib.util

import pytest

import jammi
from jammi import NoEmbeddedEngineError, PlatformNotInstalledError


def test_platform_returns_registered_extension(monkeypatch):
    """With an extension registered under the `platform` role, `jammi.platform`
    returns exactly what the discovery loaded — the plug-in object, verbatim."""
    sentinel = object()
    monkeypatch.setattr(
        jammi,
        "_load_extension",
        lambda role: sentinel if role == "platform" else None,
    )
    assert jammi.platform is sentinel


def test_platform_absent_raises_pointing_at_install(monkeypatch):
    """With nothing registered, `jammi.platform` is a truthful
    `PlatformNotInstalledError` naming the platform SDK install, not an
    `AttributeError`."""
    monkeypatch.setattr(jammi, "_load_extension", lambda role: None)
    with pytest.raises(PlatformNotInstalledError) as info:
        _ = jammi.platform
    assert "pip install jammi-ai-platform" in str(info.value)


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
    assert "platform" in names
    for name in ("AuditHandle", "EphemeralSession", "ModelTask", "PerQueryAudit",
                 "TrainingJob"):
        assert name in names


def test_embedded_symbol_without_engine_raises(monkeypatch):
    """Simulate the client-only build: `find_spec('jammi_native')` misses, so an
    embedded-only symbol is a truthful `NoEmbeddedEngineError` naming both the
    attribute and the `[embedded]` extra — never a bare `ImportError`."""
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "jammi_native":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    with pytest.raises(NoEmbeddedEngineError) as info:
        _ = jammi.PerQueryAudit
    msg = str(info.value)
    assert "PerQueryAudit" in msg
    assert "pip install jammi-ai[embedded]" in msg


@pytest.mark.skipif(
    importlib.util.find_spec("jammi_native") is None,
    reason="embedded extra (`jammi_native`) not installed in this lane",
)
def test_embedded_symbol_with_engine_returns_native_type():
    """With the engine importable, an embedded-only accessor returns the very
    type `jammi_native` exports — the lazy re-export, not a copy."""
    import jammi_native

    assert jammi.PerQueryAudit is jammi_native.PerQueryAudit
    assert jammi.TrainingJob is jammi_native.TrainingJob
