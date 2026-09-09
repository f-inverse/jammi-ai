# BASE_IMAGE has no default ON PURPOSE, the same doctrine RUST_VERSION below
# is held to: the workflow passes the per-arch manylinux_2_28 image
# (`quay.io/pypa/manylinux_2_28_x86_64` / `quay.io/pypa/manylinux_2_28_aarch64`)
# explicitly for the leg it is building (buildx sets the runner's own
# TARGETARCH; there is no Dockerfile-side arch-to-image lookup). A default
# would let a build silently resolve to the wrong arch's userland instead of
# failing the instant the workflow forgets to pass it.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Platform SQLite runtime (`/lib64/libsqlite3.so.0`). The esc-073 harness
# (`crates/jammi-db/tests/it/esc_073_foreign_sqlite_library.rs`) `dlopen`s it to
# get a SECOND SQLite library instance in one process alongside the statically
# bundled `libsqlite3-sys`; with no platform library the harness has nothing to
# collide with and reports itself vacuous. The base image happens to carry
# `sqlite-libs` today, but NO package in it requires that package, so a base
# rebuild could drop it and silently hollow out the harness. Named explicitly
# here instead of inherited by luck (a no-op when already present).
RUN yum install -y sqlite-libs \
    && yum clean all

# Per-arch download variables. TARGETARCH is set by buildx per platform
# (`linux/amd64` -> `amd64`, `linux/arm64` -> `arm64`) and is NOT the same
# spelling any of the four upstream releases use in their own asset names, so
# one prelude maps it once into the four vendor-specific arch strings the
# download lines below need. Written to a file (not `ENV`, which cannot run a
# shell) and sourced by every subsequent RUN that needs them, so the mapping
# lives in exactly one place instead of four repeated case statements. Any
# TARGETARCH other than amd64/arm64 fails the build by name — silently
# defaulting to an arch string would produce a 404 on the download instead of
# a clear "this platform is not supported" message.
ARG TARGETARCH
RUN set -eu; \
    case "${TARGETARCH}" in \
      amd64) \
        printf '%s\n' \
          'PROTOC_ARCH=linux-x86_64' \
          'MOLD_ARCH=x86_64-linux' \
          'SCCACHE_ARCH=x86_64-unknown-linux-musl' \
          'QUARTO_ARCH=linux-amd64' \
          > /etc/ci-arch-env \
        ;; \
      arm64) \
        printf '%s\n' \
          'PROTOC_ARCH=linux-aarch_64' \
          'MOLD_ARCH=aarch64-linux' \
          'SCCACHE_ARCH=aarch64-unknown-linux-musl' \
          'QUARTO_ARCH=linux-arm64' \
          > /etc/ci-arch-env \
        ;; \
      *) \
        echo "unsupported TARGETARCH: '${TARGETARCH}' (expected amd64 or arm64)" >&2; \
        exit 1 \
        ;; \
    esac

# protoc: prost-build (via substrait) invokes protoc at build time.
# Not in AlmaLinux 8 repos — install from GitHub release.
ARG PROTOC_VERSION=28.3
RUN . /etc/ci-arch-env \
    && curl -fsSL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-${PROTOC_ARCH}.zip" \
        -o /tmp/protoc.zip \
    && unzip /tmp/protoc.zip -d /usr/local bin/protoc 'include/*' \
    && rm /tmp/protoc.zip

# mold linker for faster linking
ARG MOLD_VERSION=2.35.1
RUN . /etc/ci-arch-env \
    && curl -fsSL "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-${MOLD_ARCH}.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local

# Rust toolchain (manylinux ships no Rust).
#
# RUST_VERSION has no default ON PURPOSE. The build context is `.docker/`, so
# rust-toolchain.toml cannot be COPYed in; the workflow reads the pin from that
# file and passes it here. A default would let the image build succeed at a
# stale version while the repo pin moved — silently producing an image whose
# rustc disagrees with every checkout that uses it. Failing closed is the point.
ARG RUST_VERSION
RUN test -n "${RUST_VERSION}" || { echo "RUST_VERSION build-arg is required (from rust-toolchain.toml)"; exit 1; }
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain "${RUST_VERSION}" --profile minimal \
    && rustup component add rustfmt clippy \
    && rustc --version | grep -qF "${RUST_VERSION}"

# sccache — pre-built static binary (avoids openssl-devel in the image,
# which could cause manylinux-incompatible linkage in wheel builds)
ARG SCCACHE_VERSION=0.14.0
RUN . /etc/ci-arch-env \
    && curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local/cargo/bin \
        "sccache-v${SCCACHE_VERSION}-${SCCACHE_ARCH}/sccache"

# mdbook — documentation builds
RUN cargo install mdbook --locked --version 0.5.2 \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

# maturin — PyO3 wheel builds. Pinned to a specific release (not an unpinned
# `cargo install`) so every image build resolves the same maturin, keeping
# wheel builds reproducible across CI runs and local dev.
RUN cargo install maturin --locked --version 1.14.1 \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

# Quarto — renders + executes the in-repo cookbook (cookbook/book/). The binary
# only; jupyter (its execution engine) is a per-job pip install from the book's
# `[book]` extra against the freshly-built wheel, so the engine version it runs
# against is never baked into the image. The arm64 and amd64 tarballs both
# unpack to the same `quarto-${QUARTO_VERSION}/` top-level directory (verified
# against both upstream releases), so the symlink line below needs no
# per-arch branch.
ARG QUARTO_VERSION=1.6.40
RUN . /etc/ci-arch-env \
    && curl -fsSL "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-${QUARTO_ARCH}.tar.gz" \
        | tar -xz -C /opt \
    && ln -s "/opt/quarto-${QUARTO_VERSION}/bin/quarto" /usr/local/bin/quarto
