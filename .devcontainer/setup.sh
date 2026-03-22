#!/bin/bash
set -euo pipefail

# --- System dependencies ---
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    protobuf-compiler \
    libprotobuf-dev \
    libonig-dev \
    liblzma-dev \
    pkg-config

# --- Rust toolchain (from rust-toolchain.toml) ---
rustup show  # triggers automatic install of pinned toolchain

# --- Python packages ---
pip install --upgrade pip
pip install \
    maturin \
    pyarrow \
    pytest \
    pre-commit

# --- Rust tools ---
cargo install mdbook --locked

# --- Pre-commit hooks ---
pre-commit install

echo "✓ Development environment ready"
