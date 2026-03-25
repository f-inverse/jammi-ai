# Installation

## Rust

Add Jammi to your `Cargo.toml`:

```toml
[dependencies]
jammi-engine = "0.1"
jammi-ai = "0.1"
tokio = { version = "1", features = ["full"] }
```

Or install the CLI:

```bash
cargo install jammi-cli
```

### Build dependencies (Linux)

If building from source, you need:

```bash
apt-get install protobuf-compiler libonig-dev liblzma-dev pkg-config mold
```

These are pre-installed in the devcontainer and CI images.

## Python

```bash
pip install jammi
```

Requires Python 3.8+. Pre-built wheels are available for Linux, macOS, and Windows.

## From source

```bash
git clone https://github.com/f-inverse/jammi-ai.git
cd jammi-ai
cargo build --release
```

The CLI binary is at `target/release/jammi`.

For the Python package from source:

```bash
pip install maturin
maturin develop --release
```

## Runtime requirements

Jammi has **no mandatory runtime dependencies** beyond the binary itself.

Optional:
- **CUDA toolkit + cuDNN** for GPU inference (CPU works out of the box)
- **HuggingFace Hub access** for downloading models (first run downloads ~90MB for MiniLM, cached thereafter)
- **PostgreSQL / MySQL client libraries** if using federated database sources

Set `HF_TOKEN` for gated models, or `HF_HOME` to control the cache location.
