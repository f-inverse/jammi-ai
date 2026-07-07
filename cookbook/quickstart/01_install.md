# 1. Install

```bash
pip install jammi-ai
```

On Linux and macOS this resolves to a pre-built wheel and pulls in
`pyarrow` (Arrow tables flow zero-copy between Rust and Python) and
`numpy` automatically.

The `jammi-ai` wheel is CPU-only. GPU acceleration is served by the Jammi
*server*, not the embed wheel — install the CUDA server wheel and connect to it
over gRPC (see [step 2](./02_connect.md)):

```bash
pip install jammi-server-cu12   # ships the CUDA jammi-server binary
jammi-server                    # then connect with jammi.connect("grpc://…")
```

To build from source (e.g. you cloned the repo and are iterating on the
Rust core):

```bash
pip install maturin
maturin develop --release
```

## Requirements

- Python ≥ 3.9
- Linux (x86_64, glibc 2.28+) or macOS (Apple Silicon or Intel)

Windows is not yet supported (POSIX memory-mapping APIs are used in the
storage layer).

## Verify

```python
import jammi
print(jammi.__name__)
# jammi
```

If `import jammi` works, you're ready for [step 2](./02_connect.md).
