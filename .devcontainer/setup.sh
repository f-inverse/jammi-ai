#!/bin/bash
set -euo pipefail

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
