"""Pytest config — ensures the built `jammi_native` module is importable.

`maturin develop --release` (run by the CI test-python job before pytest)
installs the top-level `jammi_native` package into the venv's site-packages,
with the compiled extension beside its `__init__.py` as
`jammi_native/jammi_native.abi3.so` (the package re-exports the engine surface,
so `import jammi_native` yields it directly — a bare top-level import). The
`python-source = "python"` setting in `packaging/native/pyproject.toml` puts
`jammi_native/` on the path; this `conftest.py` doesn't need to fiddle with
`sys.path` because `maturin develop` installs it into the active environment.
"""
