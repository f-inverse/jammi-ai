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
    pytest

# --- Claude Code (native installer) ---
curl -fsSL https://claude.ai/install.sh | bash

echo "Development environment ready"
