# 1. Install

```bash
pip install "jammi-ai[embedded]"
```

`jammi-ai` is the client (`import jammi`). The `[embedded]` extra adds the
engine itself, `jammi-ai-native`, so the client can run it in-process:
`jammi.connect("file://…")` in [step 2](./02_connect.md) needs it. Without the
extra the client only reaches a remote `jammi-server` (`connect("grpc://…")`)
and a `file://` target raises `NoEmbeddedEngineError`. Both pull in `pyarrow`
(Arrow tables flow zero-copy between the engine and Python) and `numpy`.

On an NVIDIA GPU of compute capability 8.0 or newer (A100, L4, RTX 30-series
and later), install the CUDA engine in place of the CPU one:

```bash
pip install jammi-ai jammi-ai-native-cu12
```

It carries its CUDA libraries as pip dependencies; the host needs only the
NVIDIA driver.

To build the engine from a source checkout instead (you are changing the Rust
core):

```bash
pip install maturin
maturin develop --release -m crates/jammi-python/Cargo.toml
```

## Requirements

- Python 3.9 or newer
- Linux (x86_64 or aarch64, glibc 2.28+) or macOS (Apple Silicon or Intel);
  the CUDA engine is Linux x86_64

Windows is not supported: the storage layer uses POSIX memory mapping.

## Verify

```python
import jammi
import jammi_native  # the engine the [embedded] extra installed
print(jammi.__name__)
# jammi
```

If both imports work, you're ready for [step 2](./02_connect.md).
