#!/usr/bin/env bash
# Every workflow parses as GitHub parses it — job graph, `needs`, expressions,
# action inputs — so a broken workflow fails here rather than on push, where
# GitHub rejects the whole file and runs none of its jobs; and every `run:`
# block passes shellcheck, the lint the scripts under ci/ are held to. A
# Python body runs through its own script, never inline.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
actionlint -pyflakes= .github/workflows/*.yml
