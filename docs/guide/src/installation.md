# Installation

## Rust

Add Jammi to your `Cargo.toml`:

```toml
[dependencies]
jammi-db = "0.25"
jammi-ai = "0.25"
tokio = { version = "1", features = ["full"] }
```

## CLI

The `jammi` CLI registers sources, runs SQL, and starts the server. There are
three ways to get it.

### `cargo install` (CPU)

Builds from source on your machine. Needs the build dependencies below.

```bash
cargo install jammi-cli
```

The installed binary is `jammi`.

### Prebuilt binary (CPU)

Download a stripped, ready-to-run binary from the
[GitHub releases](https://github.com/f-inverse/jammi-ai/releases). No build
toolchain required. Assets are published per release:

- `jammi-<version>-x86_64-unknown-linux-gnu.tar.gz` — Linux x86-64 (built on a
  glibc 2.28 floor, so it runs on any newer Linux)
- `jammi-<version>-aarch64-apple-darwin.tar.gz` — macOS on Apple silicon

```bash
tar -xzf jammi-0.25.0-x86_64-unknown-linux-gnu.tar.gz
./jammi --help
```

### GPU (CUDA 12)

GPU inference ships as a container image, not a bare binary. The
`jammi-ai-server-cu12` image runs `jammi-server` as its entrypoint and also
carries the `jammi` admin CLI; it is turnkey:

```bash
docker run --gpus all \
  -p 8080:8080 -p 8081:8081 \
  ghcr.io/f-inverse/jammi-ai-server-cu12:latest
```

That runs `jammi-server` with zero config. See
[Deploy as a Server](./deploy-server.md#gpu-serving) for GPU configuration and
persistence.

### Build dependencies (Linux)

If building from source, you need a C compiler and `protoc`:

```bash
# Debian/Ubuntu
apt-get install protobuf-compiler gcc g++ pkg-config

# RHEL/AlmaLinux
yum install protobuf-compiler gcc gcc-c++ pkg-config
```

All other native libraries (lzma, zstd, zlib, sqlite) are vendored and compiled from source automatically. These tools are pre-installed in the devcontainer and CI images.

## Python

```bash
pip install jammi-ai
```

Requires Python 3.8+. Pre-built wheels are available for Linux, macOS, and Windows.

## From source

```bash
git clone https://github.com/f-inverse/jammi-ai.git
cd jammi-ai
cargo build --release
```

The CLI binary is at `target/release/jammi` (a strict gRPC client) and the
server binary at `target/release/jammi-server`.

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
