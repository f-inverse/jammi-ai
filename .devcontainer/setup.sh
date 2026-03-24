#!/bin/bash
set -euo pipefail

# --- Fix volume ownership (Docker mounts as root) ---
sudo chown -R vscode:vscode target/ 2>/dev/null || true
sudo chown -R vscode:vscode /home/vscode/.cache/sccache 2>/dev/null || true

# --- Python packages (dev-only) ---
pip install --upgrade pip
pip install \
    maturin \
    pyarrow \
    pytest \
    pre-commit

# --- Claude Code (native installer) ---
curl -fsSL https://claude.ai/install.sh | bash

# --- Pre-commit hooks ---
pre-commit install

echo "Development environment ready"
