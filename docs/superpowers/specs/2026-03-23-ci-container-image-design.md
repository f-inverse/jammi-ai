# CI Container Image

**Date:** 2026-03-23
**Status:** Approved

## Problem

CI runs on bare `ubuntu-latest` and rebuilds the entire toolchain every run — duplicating
the same apt-get block across 4 jobs. The devcontainer is the declared single source of
truth for the build environment (per CLAUDE.md), but CI doesn't use it. This has already
caused drift: mdbook version mismatch breaks the docs job, and `-D warnings` catches an
unused import that local dev doesn't flag.

## Design

### Two-layer image strategy

**Layer 1 — CI base image** (`rust:1.88.0-bookworm`, patch-pinned)

A lean image with only what's needed to build, test, and document the project:

- System deps: `protobuf-compiler`, `libprotobuf-dev`, `libonig-dev`, `liblzma-dev`, `pkg-config`
- Rust 1.88.0 with components: `rustfmt`, `clippy` (no `rust-src` — only needed for rust-analyzer in dev)
- `mdbook` pinned: `cargo install mdbook --locked --version 0.4.44`
- Runs as root (GitHub Actions `container:` jobs expect root)
- Layer cleanup: `apt-get clean && rm -rf /var/lib/apt/lists/*`

Published to `ghcr.io/f-inverse/jammi-ai-ci` with tags:
- `latest` (from main branch builds)
- Git commit SHA (for pinning/rollback)

**Failure mode:** If the image build fails, CI continues using the most recent `latest` tag.
To roll back, update `ci.yml` to pin a specific SHA tag.

**Layer 2 — Devcontainer**

References the CI base image via a devcontainer `Dockerfile` (not features, since devcontainer
features require Microsoft base image conventions that `rust:*` images don't have):

- Devcontainer `Dockerfile` installs Python 3.12, GitHub CLI directly
- `postCreateCommand` installs dev-only tools: maturin, pyarrow, pytest, pre-commit, Claude Code
- Adds `rust-src` component for rust-analyzer

### File layout

```
.docker/
  ci.Dockerfile           # CI base image definition

.github/workflows/
  build-ci-image.yml      # Build + push image on .docker/ or rust-toolchain.toml changes
  ci.yml                  # Updated — all jobs use container: ghcr.io/f-inverse/jammi-ai-ci

.devcontainer/
  Dockerfile               # FROM ghcr.io/f-inverse/jammi-ai-ci, adds Python/gh/rust-src
  devcontainer.json         # build.dockerfile → Dockerfile
  setup.sh                  # Reduced to dev-only pip/cargo tools
```

### CI workflow changes

All four jobs (`check`, `test`, `test-live`, `docs`) use `container: ghcr.io/f-inverse/jammi-ai-ci:latest`.
Remove all `dtolnay/rust-toolchain`, `Swatinem/rust-cache`, apt-get, and `cargo install mdbook` steps.
Jobs become checkout + run.

### Image build workflow

Triggers on push to `main` when paths change:
- `.docker/**`
- `rust-toolchain.toml`

Uses `docker/build-push-action` to build and push to GHCR.
Requires `permissions: packages: write` (`GITHUB_TOKEN` is sufficient for same-repo GHCR pushes).

When a new system dependency is needed, update `ci.Dockerfile` — the `.docker/**` path trigger
handles the rebuild automatically.

### Bug fixes (included in this work)

- Remove unused `Float32Array` import in `crates/jammi-ai/tests/phase04_inference.rs:153`
- Remove `multilingual = false` from `docs/guide/book.toml` (field removed in mdbook 0.4.21+, now causes a parse error)
