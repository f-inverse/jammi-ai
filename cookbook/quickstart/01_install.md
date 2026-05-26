# 1. Install

```bash
pip install jammi-ai
```

On Linux and macOS this resolves to a pre-built wheel and pulls in
`pyarrow` (Arrow tables flow zero-copy between Rust and Python) and
`numpy` automatically.

For CUDA GPU support, install the CUDA-flavored wheel instead:

```bash
pip install jammi-ai-cu12
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
import jammi_ai
print(jammi_ai.__name__)
# jammi_ai
```

If `import jammi_ai` works, you're ready for [step 2](./02_connect.md).
