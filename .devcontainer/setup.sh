#!/bin/bash
set -euo pipefail

# --- Fix volume and cache ownership (Docker mounts as root) ---
sudo chown -R vscode:vscode target/ 2>/dev/null || true
sudo chown -R vscode:vscode /home/vscode/.cache 2>/dev/null || true

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
# Claude Code installer sets core.hooksPath; unset it so pre-commit can
# manage hooks in the standard .git/hooks directory.
git config --unset-all core.hooksPath 2>/dev/null || true
pre-commit install

echo "Development environment ready"
