# W00 — Rename `jammi-engine` Rust crate to `jammi-db` + Python module `jammi` to `jammi_ai`

**Status:** spec — pending review
**Owner:** TBD
**Estimated effort:** 1 week
**Workstream dependencies:** none (first workstream in the plan)
**Workstreams blocked by this:** W0 (jammi-numerics references `jammi-db`), W1 (Postgres backend touches the renamed catalog crate), W5.5 (Monitor refactor uses the renamed crate). Other workstreams indirectly affected through transitive deps.

## Motivation

The product is repositioning around two customer-facing brands:
- **Jammi AI** — OSS engine + AI primitives (free, Apache 2.0)
- **Jammi AI Platform** — commercial managed offering (paid, SaaS)

Two code-level names are out of step with that positioning:

1. **`jammi-engine` Rust crate** — the name "engine" understates what it is: a vector database + SQL federation + mutable companion tables + trigger broker. The product narrative talks about *managed jammi db*; the underlying OSS crate should match. Renaming `jammi-engine` → `jammi-db` aligns code with positioning.

2. **`jammi` Python module** — the PyPI distribution is already named `jammi-ai` (verified in `pyproject.toml:6`), but the importable Python module is `jammi`. This mismatch (install as `jammi-ai`, import as `jammi`) is technically valid but confusing. Renaming the Python module to `jammi_ai` makes install name and import name match — the standard Python convention.

Doing both renames in one coordinated PR cycle minimises downstream churn: the OSS workspace ships one breaking release; downstream consumers (jammi-enterprise) pin the new names in one follow-up PR.

## Current state (verified at spec time)

| Surface | Current value | Source of truth |
|---|---|---|
| PyPI distribution name | `jammi-ai` | `pyproject.toml:6` — already correct, **no change needed** |
| Python import module | `jammi` | `python/jammi/__init__.py` |
| Maturin `module-name` | `jammi._native` | `pyproject.toml [tool.maturin]` |
| Rust crate (engine) | `jammi-engine` | `crates/jammi-engine/Cargo.toml:2` |
| Workspace member | `crates/jammi-engine` | Root `Cargo.toml:4` |
| Workspace dep | `jammi-engine = { version = "0.5.9", path = "crates/jammi-engine" }` | Root `Cargo.toml` |
| jammi-enterprise pin | `jammi-engine = { git = "https://github.com/f-inverse/jammi-ai", tag = "v0.5.8" }` | `jammi-enterprise/Cargo.toml` |

## Scope

### Rename 1 — Rust crate `jammi-engine` → `jammi-db`

**Directory move:**
- `crates/jammi-engine/` → `crates/jammi-db/`

**Crate metadata:**
- `crates/jammi-db/Cargo.toml`:
  - `[package] name = "jammi-engine"` → `name = "jammi-db"`
  - `description` updated to reflect "vector database + SQL federation" framing
  - All `path = ...` references inside that file unchanged (they reference relative paths)

**Workspace root `Cargo.toml`:**
- `workspace.members`: `"crates/jammi-engine"` → `"crates/jammi-db"`
- `workspace.default-members`: same
- `workspace.dependencies`: `jammi-engine = { version = "0.5.9", path = "crates/jammi-engine" }` → `jammi-db = { version = "0.5.9", path = "crates/jammi-db" }`

**Callers within the jammi-ai workspace** (all `use jammi_engine::*` → `use jammi_db::*`, all `jammi-engine` in `Cargo.toml` → `jammi-db`):

By crate, in dependency order:
- `crates/jammi-ai/` — `Cargo.toml` dep + Rust source (most affected; ~40 files include `session.rs`, `catalog/*`, `inference/*`, eval, fine-tune)
- `crates/jammi-encoders/` — `Cargo.toml` dep + any direct uses
- `crates/jammi-lora/` — `Cargo.toml` dep
- `crates/jammi-server/` — `Cargo.toml` dep + Flight SQL implementation
- `crates/jammi-cli/` — `Cargo.toml` dep + all command files in `src/commands/` + `src/main.rs`
- `crates/jammi-python/` — `Cargo.toml` dep + PyO3 wrapper source
- `crates/jammi-test-utils/` — `Cargo.toml` dep + test helpers

Total estimated touch surface: ~180 file modifications across `.rs` and `.toml` files (verified via `grep -rln 'use jammi_engine\|jammi_engine::\|jammi-engine' crates/ --include='*.rs' --include='*.toml' | wc -l`).

**Documentation:**
- `README.md` at workspace root
- `docs/guide/` content referring to "engine"
- `docs/IMPLEMENTATION.md`
- Rustdoc on public items in the renamed crate (no journey-shaped writing per CLAUDE.md; just current-state names)
- Any `CLAUDE.md` in the workspace that mentions `jammi-engine` (search and update)

**crates.io publication:**
- Publish `jammi-db` v0.5.9 (matches workspace version)
- Mark `jammi-engine` deprecated on crates.io with a notice: `"renamed to jammi-db; install jammi-db instead"`. crates.io supports the `[badges.maintenance]` deprecation marker via `Cargo.toml`, OR yank the latest version and point to `jammi-db` from the repository README.
- Do NOT delete or fully yank historical `jammi-engine` versions — downstream consumers pinned at older versions should keep working.

**Downstream pin update** (`jammi-enterprise` repo — follow-up PR after the new jammi-db version publishes):
- `jammi-enterprise/Cargo.toml` workspace deps: `jammi-engine = { git = "...", tag = "v0.6.0" }` → `jammi-db = { git = "...", tag = "v0.6.0" }`
- All `use jammi_engine::*` → `use jammi_db::*` across the jammi-enterprise crates (verified ~20 caller files)

### Rename 2 — Python module `jammi` → `jammi_ai`

**Directory move:**
- `python/jammi/` → `python/jammi_ai/`

**Module init:**
- `python/jammi_ai/__init__.py`: `from jammi._native import connect, Database, SearchBuilder, FineTuneJob, ModelTask` → `from jammi_ai._native import ...`
- `__all__` list unchanged

**Maturin config in `pyproject.toml`:**
- `[tool.maturin] module-name = "jammi._native"` → `module-name = "jammi_ai._native"`

**PyPI distribution name** (`pyproject.toml [project] name`):
- **Unchanged.** Already `jammi-ai`.

**Python test imports** (verified callers via `grep -rn '^import jammi\|^from jammi' --include='*.py'`):
- `crates/jammi-python/tests/test_*.py` (4+ test files): `import jammi` → `import jammi_ai`
- `python/jammi_ai/__init__.py`: internal import (above)
- `tests/uat/shape_b_*.py`, `shape_c_isolation.py`: top-level imports
- `tests/cookbook_smoke_test.py`, `tests/fixtures/generate*.py`: any `import jammi` usage

**Documentation:**
- `README.md` Python install/usage examples
- `docs/guide/` Python sections
- Cookbook entries that show `import jammi` → update to `import jammi_ai`

### Backwards-compatibility shim

Per CLAUDE.md "no backwards compatibility — no shims, no deprecated paths, no keep-the-old-way-around": **do not ship a compatibility `jammi` Python module that re-exports from `jammi_ai`.** Customers who pin to older versions of `jammi-ai` PyPI keep working against the old `import jammi` module. Customers who upgrade past this release get a clean break with a release-note migration step (`s/import jammi/import jammi_ai/`).

The `0.5.x` line of releases continues to ship `import jammi`; the `0.6.0` release ships `import jammi_ai`. Major-version bump signals the breaking change.

## File-by-file change summary

The change is mechanical. A focused PR description should enumerate:

```
Rust crate rename (one PR in jammi-ai):
- Move:    crates/jammi-engine/         → crates/jammi-db/
- Edit:    crates/jammi-db/Cargo.toml   (package name + description)
- Edit:    Cargo.toml                   (workspace.members + workspace.dependencies)
- Edit:    crates/jammi-ai/Cargo.toml + ~40 .rs files
- Edit:    crates/jammi-encoders/Cargo.toml + .rs files
- Edit:    crates/jammi-lora/Cargo.toml + .rs files
- Edit:    crates/jammi-server/Cargo.toml + .rs files
- Edit:    crates/jammi-cli/Cargo.toml + ~10 .rs files
- Edit:    crates/jammi-python/Cargo.toml + .rs files
- Edit:    crates/jammi-test-utils/Cargo.toml + .rs files

Python module rename (same PR):
- Move:    python/jammi/                → python/jammi_ai/
- Edit:    python/jammi_ai/__init__.py
- Edit:    pyproject.toml [tool.maturin] module-name
- Edit:    crates/jammi-python/tests/test_*.py
- Edit:    tests/uat/shape_*.py
- Edit:    tests/cookbook_smoke_test.py + tests/fixtures/generate*.py

Docs (same PR):
- Edit:    README.md
- Edit:    docs/guide/**/*.md  (search/replace jammi-engine → jammi-db and import jammi → import jammi_ai)
- Edit:    docs/IMPLEMENTATION.md  (if it mentions jammi-engine)

Downstream (separate PR in jammi-enterprise after new jammi-ai version publishes):
- Edit:    jammi-enterprise/Cargo.toml (workspace deps pin jammi-db)
- Edit:    crates/jammi-enterprise/**/*.rs  (~10-15 files)
- Edit:    crates/jammi-enterprise-server/**/*.rs  (~5 files)
- Edit:    crates/jammi-enterprise-test-utils/**/*.rs (~2 files)
- Note:    crates/jammi-enterprise-python/ is being deleted in W6; do not touch
```

## Success criteria

The PR is correct when ALL of:

1. `cargo build --workspace` succeeds in the jammi-ai workspace
2. `cargo test --workspace --exclude jammi-python` passes (existing OSS test suite)
3. `maturin develop` succeeds; `python -c "import jammi_ai; jammi_ai.connect(artifact_dir='/tmp/test')"` works
4. `pytest crates/jammi-python/tests/` passes (with imports updated)
5. `cargo clippy --workspace -- -D warnings` clean
6. `cargo fmt --check` clean
7. No occurrence of `jammi_engine` or `jammi-engine` remains in source (`grep -rn 'jammi[_-]engine' crates/ python/ --include='*.rs' --include='*.toml' --include='*.py' --include='*.md'` returns empty)
8. No occurrence of `import jammi$` (bare) or `from jammi import` remains (`grep` similar)
9. crates.io shows `jammi-db v0.6.0` published; old `jammi-engine` marked deprecated
10. PyPI shows new `jammi-ai v0.6.0` with the renamed module; `pip install jammi-ai==0.6.0 && python -c "import jammi_ai"` works
11. Release notes published explaining the breaking change and migration steps

Downstream `jammi-enterprise` PR is correct when:
12. `cargo build --workspace` succeeds in jammi-enterprise (after the workspace dep is bumped to the new jammi-ai version)
13. `cargo test --workspace` passes (the existing 271 enterprise tests still green; this is a pure rename refactor so no behaviour changes)

## Rollback strategy

If the rename PR lands and a critical regression surfaces post-publish:
- crates.io: yank the `jammi-db v0.6.0` release, un-deprecate `jammi-engine`
- PyPI: yank the new `jammi-ai v0.6.0` release; the prior `jammi-ai v0.5.x` (with `import jammi`) remains the latest unyanked
- jammi-enterprise pin reverts to the prior `jammi-ai` tag

The rename is mechanical, so regressions are very unlikely. The rollback is a safety net, not an expected path.

## Out of scope (explicitly)

- Renaming the Rust workspace folder `jammi-ai/` to anything else (unchanged — that's the repo name)
- Renaming the Rust crate `jammi-ai` (within the workspace) to anything else (unchanged)
- Renaming Cargo.lock or git history
- Renaming the GitHub repo
- Any code logic changes — this is a pure refactor; behaviour is identical

## CLAUDE.md self-check

- [x] No duplicated logic introduced (pure refactor)
- [x] No band-aids — no `#[allow(...)]` added to silence warnings; no `// TODO: rename later`
- [x] No compatibility shims (`jammi` re-export module explicitly rejected)
- [x] Atomic across workspace: one PR per repo
- [x] Names match shape — `jammi-db` describes the engine's role; `jammi_ai` matches the PyPI distribution
- [x] Docs reflect the renamed state — no "added in PR #N" / "since v0.6" markers in rustdoc; the only journey-shaped writing is in `docs/IMPLEMENTATION.md` (the allowed location per CLAUDE.md) and this spec
- [x] User-facing docs (README, guide) describe the system as it IS after the rename, not "the old name was..."
