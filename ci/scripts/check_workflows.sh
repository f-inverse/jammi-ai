#!/usr/bin/env bash
# Every workflow parses as GitHub parses it — job graph, `needs`, expressions,
# action inputs — so a broken workflow fails here rather than on push, where
# GitHub rejects the whole file and runs none of its jobs. Shell and Python
# bodies are checked by their own guards, not here.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
actionlint -shellcheck= -pyflakes= .github/workflows/*.yml
