FROM quay.io/pypa/manylinux_2_28_x86_64

# protoc: prost-build (via substrait) invokes protoc at build time.
# Not in AlmaLinux 8 repos — install from GitHub release.
ARG PROTOC_VERSION=28.3
RUN curl -fsSL "https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOC_VERSION}/protoc-${PROTOC_VERSION}-linux-x86_64.zip" \
        -o /tmp/protoc.zip \
    && unzip /tmp/protoc.zip -d /usr/local bin/protoc 'include/*' \
    && rm /tmp/protoc.zip

# mold linker for faster linking
ARG MOLD_VERSION=2.35.1
RUN curl -fsSL "https://github.com/rui314/mold/releases/download/v${MOLD_VERSION}/mold-${MOLD_VERSION}-x86_64-linux.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local

# Rust toolchain (manylinux ships no Rust)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.88.0 --profile minimal \
    && rustup component add rustfmt clippy

# OpenSSL headers — needed by sccache (native-tls) at compile time
RUN yum install -y openssl-devel && yum clean all

# sccache — compilation caching (local disk or GHA cache backend in CI)
RUN cargo install sccache --locked \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

# mdbook — documentation builds
RUN cargo install mdbook --locked --version 0.5.2 \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git

# maturin — PyO3 wheel builds
RUN cargo install maturin --locked \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git
