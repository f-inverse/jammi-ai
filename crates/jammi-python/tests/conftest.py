"""Pytest config — ensures the built `jammi_ai._native` module is importable.

`maturin develop --release` (run by the CI test-python job before pytest)
drops `_native.abi3.so` into `python/jammi_ai/` next to the package's
`__init__.py`. The `python-source = "python"` setting in `pyproject.toml`
tells maturin where to put it; this `conftest.py` doesn't need to fiddle
with `sys.path` because pytest auto-discovers the `python/` parent of the
package once it's on `PYTHONPATH` (set by the workflow's
`maturin develop` invocation).
"""
