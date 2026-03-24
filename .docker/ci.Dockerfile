FROM rust:1.88.0-bookworm

# System dependencies for building jammi-ai crates
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        protobuf-compiler \
        libprotobuf-dev \
        libonig-dev \
        liblzma-dev \
        pkg-config \
        mold \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Rust components (no rust-src — that's dev-only for rust-analyzer)
RUN rustup component add rustfmt clippy

# mdbook for documentation builds (pinned to match local dev)
RUN cargo install mdbook --locked --version 0.5.2 \
    && rm -rf /usr/local/cargo/registry /usr/local/cargo/git
