# Installation

## Rust

Add Jammi to your `Cargo.toml`:

```toml
[dependencies]
jammi-db = "0.55.0"
jammi-ai = "0.55.0"
tokio = { version = "1", features = ["full"] }
```

On `aarch64` Linux, also compile `gemm-f16` optimized in debug builds. Its FP16 kernels
fail to assemble at `opt-level = 0` ("instruction requires: fullfp16"); optimized, they
build for any ARMv8 target and are chosen at run time on hardware that has FP16:

```toml
[profile.dev.package.gemm-f16]
opt-level = 1
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
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
  ghcr.io/f-inverse/jammi-ai-server-cu12:latest
```

Both ports bind to `127.0.0.1`: the server performs no authentication of
its own (see [The identity seam](./deploy-server.md#the-identity-seam)),
so a loopback bind keeps the unauthenticated admin surface off the host's
public network until a terminator or reverse proxy is put in front of it.

`:latest` on every image is the newest `v*` release and nothing else moves
it. Pin an exact `:X.Y.Z` tag for a reproducible pull.

That runs `jammi-server` with zero config. See
[Deploy as a Server](./deploy-server.md#gpu-serving) for GPU configuration and
persistence.

Alternatively, install the CUDA server as a pip wheel — it ships the same
`jammi-server` binary and pulls the CUDA runtime from `nvidia-*-cu12` wheels, so
no system CUDA install is required (only an NVIDIA driver on the host):

```bash
pip install jammi-server-cu12
jammi-server
```

The `jammi-ai` embed wheel is CPU-only; GPU inference runs in the server, reached
from Python via `jammi.connect("grpc://…")`.

### Build dependencies (Linux)

If building from source, you need a C compiler and `protoc`:

```bash
# Debian/Ubuntu
apt-get install protobuf-compiler gcc g++ pkg-config

# RHEL/AlmaLinux
yum install protobuf-compiler gcc gcc-c++ pkg-config
```

All other native libraries (lzma, zstd, zlib, sqlite) are vendored and compiled from source automatically. These tools are pre-installed in the devcontainer and CI images.

Building `jammi-db` with the `postgres` or `mysql` source feature on Linux also
compiles OpenSSL from source, which it links statically: OpenSSL's `Configure`
needs Perl with the `IPC::Cmd` and `Time::Piece` modules. No OpenSSL headers or
libraries are needed. macOS and Windows use the platform's TLS stack.

```bash
# Debian/Ubuntu
apt-get install perl make

# RHEL/AlmaLinux
yum install perl-IPC-Cmd perl-Time-Piece make
```

See [Connect to PostgreSQL / MySQL](./external-sources.md#what-ships).

## Python

```bash
pip install "jammi-ai[embedded]"            # the client and the in-process engine
pip install jammi-ai jammi-ai-native-cu12   # the same, with the CUDA engine (sm_80+)
pip install jammi-ai                        # the client alone, for a remote server
```

`jammi.connect("file://…")` runs the engine in-process and needs it installed;
`connect("grpc://…")` needs only the client. Requires Python 3.9+. Wheels are
built for Linux (x86_64 and aarch64, glibc 2.28+) and macOS (Apple Silicon and
Intel); the CUDA engine is Linux x86_64. Windows is not supported.

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

Set `HF_TOKEN` for gated models, or `HF_HOME` to control the cache location —
both are read as fallbacks when the config's own `[models]` section (see
[Configuration](./configuration.md#catalog-broker-signing-key-storage-and-model-source))
leaves `hub_token`/`hub_cache_dir` unset; a config value always wins over the
environment variable.
