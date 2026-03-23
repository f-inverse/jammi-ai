FROM ghcr.io/f-inverse/jammi-ai-ci:latest

# Dev-only: rust-src for rust-analyzer
RUN rustup component add rust-src

# Python 3.12 + venv (PEP 668 on bookworm blocks system-wide pip)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for devcontainer
RUN useradd -m -s /bin/bash vscode \
    && mkdir -p /home/vscode/.cargo /home/vscode/.rustup \
    && cp -r /usr/local/cargo/. /home/vscode/.cargo/ \
    && cp -r /usr/local/rustup/. /home/vscode/.rustup/ \
    && chown -R vscode:vscode /home/vscode

# Create Python venv owned by vscode
RUN python3 -m venv /home/vscode/.venv \
    && chown -R vscode:vscode /home/vscode/.venv

USER vscode
ENV CARGO_HOME=/home/vscode/.cargo
ENV RUSTUP_HOME=/home/vscode/.rustup
ENV PATH="/home/vscode/.venv/bin:/home/vscode/.cargo/bin:${PATH}"
