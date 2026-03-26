FROM ghcr.io/f-inverse/jammi-ai-ci:latest

# Dev-only: rust-src for rust-analyzer
RUN rustup component add rust-src

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
