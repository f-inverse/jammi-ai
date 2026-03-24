# Jammi AI — Engineering Standards

## Development Environment

You are running inside a devcontainer. The workspace is at `/workspaces/jammi-ai`. Do not use host paths, `docker exec`, or attempt to install system packages directly.

**Two-layer Docker image:** The CI base image (`.docker/ci.Dockerfile`) owns the build toolchain (Rust 1.88.0, protobuf, libonig, liblzma, mold, sccache, mdbook). The devcontainer Dockerfile extends it with dev-only tools (rust-src, Python, gh CLI, sudo). `setup.sh` runs at container start for user-space setup (pip packages, Claude Code). When a phase introduces a new system dependency, add it to the appropriate layer — CI image for build deps, devcontainer Dockerfile for dev-only tools. Do not install dependencies ad-hoc in the running container — they won't survive a rebuild.

**Named volumes:** `target/` and sccache cache are mounted as Docker named volumes. They survive container rebuilds but not `docker volume rm`. If you see stale artifact errors after a Rust toolchain change, run `cargo clean`.

## Build Tooling Gotchas

### sccache

`rustc-wrapper = "sccache"` is set in `.cargo/config.toml`. This applies globally to all cargo commands.

- **sccache disables incremental compilation.** This is by design — sccache and incremental are mutually exclusive. sccache compensates by caching full crate compilations across clean builds. Do not try to enable incremental alongside sccache.
- **If sccache is not installed, cargo will fail** with a cryptic error like `could not execute process sccache`. Fix: `cargo install sccache --locked`. The CI base image and devcontainer both have it pre-installed.
- **If sccache server is stuck**, builds fail with connection errors. Fix: `sccache --stop-server` then retry.
- **First `cargo test` after only running `cargo check` is a full recompile** because test binaries have different compiler flags than check mode. sccache will cache the test compilation for next time.

### mold linker

`-fuse-ld=mold` is configured for Linux targets in `.cargo/config.toml`. It is not used on macOS (Apple linker used instead).

- **mold is Linux-only (ELF).** The config is scoped to `[target.x86_64-unknown-linux-gnu]` and `[target.aarch64-unknown-linux-gnu]`. It does not affect macOS host builds.
- **If mold is not installed in the container**, linking fails with `cannot find -fuse-ld=mold`. The CI base image has mold pre-installed. If you see this on a custom setup, install mold via `apt-get install mold`.
- **jammi-python (PyO3 cdylib) is excluded from default-members** and may produce different linker behavior. Build it via `maturin`, not `cargo build --workspace`.

## Code Principles (non-negotiable)

These apply to every line of code written in this project. No exceptions. No "we'll clean it up later."

### Clean, functional style
- Favor composition over inheritance
- Use iterators, combinators, and pattern matching over imperative loops
- Prefer pure functions — inputs in, outputs out, no side effects where avoidable
- Use `Result` propagation (`?`) over panics. `unwrap()` only in tests

### Clear boundaries and separation of concerns
- Every module has one responsibility. If you can't state it in one sentence, split it
- Traits define boundaries. Concrete types live behind traits at module edges
- No module may reach into another module's internals. Public API only
- Lifecycle (load/cache/evict) and execution (infer/batch/adapt) are separate concerns — never mix them

### DRY
- If logic appears twice, extract it. No copy-paste code
- Shared behavior goes into traits or utility functions, not duplicated match arms
- Configuration constants live in one place

### No backwards compatibility
- This is a greenfield project. No shims, no deprecated paths, no "keep the old way around"
- If something needs to change, change it everywhere. Break and rebuild correctly
- No `#[deprecated]`, no `_unused` renames, no compatibility re-exports

### Type-driven design
- Use newtypes (`ModelId`, `GpuPermit`) to make invalid states unrepresentable
- Use enums over stringly-typed parameters
- RAII for resource management (permits, guards, connections)
- Builder pattern where construction has more than 3 parameters

## Test Discipline

Default `cargo test` must be fully hermetic. No live network calls. Live tests are gated behind `#[cfg(feature = "live-hub-tests")]`.

## Self-Check Before Completing Any Task

Before declaring work done, verify:
- [ ] No duplicated logic introduced
- [ ] Every new public type/trait has a clear single responsibility
- [ ] No module reaches into another module's internals
- [ ] No temporary APIs or compatibility shims
- [ ] New interfaces are downstream-driven (only what consumers need)
- [ ] Tests are in the right category (unit/contract/integration/live)
- [ ] `cargo clippy` and `cargo fmt` pass
- [ ] Code reads as idiomatic Rust, not translated Java/Python
