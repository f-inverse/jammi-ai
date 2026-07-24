FROM ghcr.io/f-inverse/jammi-ai-ci:latest

# Dev-only: rust-src for rust-analyzer
RUN rustup component add rust-src

# Pinned rich-symbol-graph toolchain: nightly-2026-06-07 + its rust-analyzer
# component, and build-graph 0.1.0. Installed as root, HERE — right beside
# the other root-user toolchain addition above (`rustup component add
# rust-src`) — and BEFORE the `cp -r /usr/local/rustup/.` → vscode's
# RUSTUP_HOME below, so the copy carries the pinned nightly + component into
# vscode's home in the same step it already carries rust-src. Installing it
# instead after `USER vscode` would also reach a user-writable
# RUSTUP_HOME/CARGO_HOME, but would split "add a toolchain component" across
# two different users for no reason — this Dockerfile's existing structure is
# one root-user toolchain-setup phase, then one copy, then `USER vscode`.
# Kept in lockstep with the `NIGHTLY`/`BUILD_GRAPH_VERSION` pins in
# `ci/scripts/build_graph_rich.sh`, which names this file back — bump both in
# the same PR. The dated nightly coexists with rust-toolchain.toml's 1.94.0
# stable default; ci/scripts/build_graph_rich.sh invokes the dated nightly
# explicitly via `--nightly`, it is never the ambient toolchain.
RUN rustup toolchain install nightly-2026-06-07 --profile minimal \
    && rustup component add rust-analyzer --toolchain nightly-2026-06-07 \
    && cargo install build-graph --version 0.1.0 --locked

# Python: use manylinux's bundled CPython 3.12
RUN ln -s /opt/python/cp312-cp312/bin/python3 /usr/local/bin/python3

# Dev-only system packages (AlmaLinux 8 / yum)
RUN yum install -y 'dnf-command(config-manager)' \
    && yum config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo \
    && yum install -y \
        gh \
        sudo \
    && yum clean all

RUN useradd -m -s /bin/bash vscode \
    && usermod -aG wheel vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && mkdir -p /home/vscode/.cargo /home/vscode/.rustup \
    && cp -r /usr/local/cargo/. /home/vscode/.cargo/ \
    && cp -r /usr/local/rustup/. /home/vscode/.rustup/ \
    && chown -R vscode:vscode /home/vscode

# Create Python venv owned by vscode
RUN /usr/local/bin/python3 -m venv /home/vscode/.venv \
    && chown -R vscode:vscode /home/vscode/.venv

USER vscode
ENV CARGO_HOME=/home/vscode/.cargo
ENV RUSTUP_HOME=/home/vscode/.rustup
ENV PATH="/home/vscode/.local/bin:/home/vscode/.venv/bin:/home/vscode/.cargo/bin:${PATH}"
ENV SCCACHE_DIR="/home/vscode/.cache/sccache"
