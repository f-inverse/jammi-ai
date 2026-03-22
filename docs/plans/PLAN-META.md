# Jammi AI — Implementation Plans

## Prerequisites

| Plan | Delivers |
|------|----------|
| [PLAN-00 Prerequisites](00-prerequisites/PLAN-00-prerequisites.md) | Devcontainer, system deps, pre-commit hooks |
| [PLAN-00 Test Fixtures](00-prerequisites/PLAN-00-test-fixtures.md) | Shared test data files (patents.parquet, golden sets, training data) |

Both are implemented before any phase begins.

## Phase Sequence

| # | Phase | Implementation | Tests | Spec Sections | Migration | Depends On |
|---|-------|---------------|-------|---------------|-----------|------------|
| 01 | Project Foundation | [PLAN-01](cp1-data-flows/PLAN-01-foundation.md) | [TEST-01](cp1-data-flows/TEST-01-foundation.md) | §12, §14, §9.1 | 001 | — |
| 02 | Source Registry & DataFusion Session | [PLAN-02](cp1-data-flows/PLAN-02-sources-datafusion.md) | [TEST-02](cp1-data-flows/TEST-02-sources-datafusion.md) | §4.1–4.3, §11 | — | 01 |
| 03 | Model Loading & Registry | [PLAN-03](cp2-intelligence/PLAN-03-model-loading.md) | [TEST-03](cp2-intelligence/TEST-03-model-loading.md) | §8.1–8.3 | — | 01 |
| 04 | InferenceExec Operator | [PLAN-04](cp2-intelligence/PLAN-04-inference-exec.md) | [TEST-04](cp2-intelligence/TEST-04-inference-exec.md) | §5.2, §7.1–7.4 | — | 02, 03 |
| 05 | Embedding Pipeline | [PLAN-05](cp3-core-loop/PLAN-05-embedding-pipeline.md) | [TEST-05](cp3-core-loop/TEST-05-embedding-pipeline.md) | §2.2, §5.2 (embedding mode), §9 | — | 04 |
| 06 | Vector Search & Evidence Model | [PLAN-06](cp3-core-loop/PLAN-06-vector-search-evidence.md) | [TEST-06](cp3-core-loop/TEST-06-vector-search-evidence.md) | §2.5–2.6, §3, §5.1 | — | 05 |
| 07 | Fine-Tuning | [PLAN-07](cp4-learning/PLAN-07-fine-tuning.md) | [TEST-07](cp4-learning/TEST-07-fine-tuning.md) | §2.4, §5.3 | — | 03, 04 |
| 08 | Evaluation | [PLAN-08](cp4-learning/PLAN-08-evaluation.md) | [TEST-08](cp4-learning/TEST-08-evaluation.md) | §2.7, §5.4 | 002 | 06, 07 |
| 09 | GPU Scheduling & Concurrency | [PLAN-09](cp5a-python-api/PLAN-09-gpu-scheduling.md) | [TEST-09](cp5a-python-api/TEST-09-gpu-scheduling.md) | §6 | — | 04, 05, 07, 08 |
| 10 | Python Bindings | [PLAN-10](cp5a-python-api/PLAN-10-python-bindings.md) | [TEST-10](cp5a-python-api/TEST-10-python-bindings.md) | §2, §13 | — | 01–09 |
| 11 | Server & CLI | [PLAN-11](cp5b-server-hardening/PLAN-11-server-cli.md) | [TEST-11](cp5b-server-hardening/TEST-11-server-cli.md) | §13 | — | 01–09 |
| 12a | External Sources & Backends | [PLAN-12a](cp5b-server-hardening/PLAN-12a-external-sources-backends.md) | [TEST-12a](cp5b-server-hardening/TEST-12a-external-sources-backends.md) | §4.2–4.3, §8.2, §11 | — | 02, 04 |
| 12b | Caching & Production Hardening | [PLAN-12b](cp5b-server-hardening/PLAN-12b-caching-hardening.md) | [TEST-12b](cp5b-server-hardening/TEST-12b-caching-hardening.md) | §7, §10 | — | 02, 04, 06 |
| 13 | Enterprise Platform | [PLAN-13](cp6-enterprise/PLAN-13-enterprise-platform.md) | [TEST-13](cp6-enterprise/TEST-13-enterprise-platform.md) | §2.8–2.9, §5.5–5.6, §9.2 | — | 01–12b |
| 14 | Enterprise Dashboard | [PLAN-14](cp7-dashboard/PLAN-14-dashboard.md) | [TEST-14](cp7-dashboard/TEST-14-dashboard.md) | §9.2 | — | 13 |

**Phases 01–12b:** Open-source engine (Apache 2.0). Complete product — federation, embeddings, inference, fine-tuning, search, evidence, evaluation, GPU scheduling, Python API, server, CLI, advanced sources, caching, production hardening.

**Phases 13–14:** Enterprise (separate private repo: `jammi-enterprise`). Autonomous experimentation, production monitoring, quality gates, and React dashboard — all in one private repo. The enterprise Rust crate uses the open-source engine as a standard library dependency. The React dashboard lives alongside it in the same repo. No enterprise-specific abstractions in OSS — the open-source crates expose stable, general-purpose interfaces (traits, hooks, observers) that enterprise consumes without special casing. No feature flags, no conditional compilation for enterprise.

The enterprise repo is created at the start of CP6 (Phase 13). Phases 01–12b are entirely in the OSS repo.

## Dependency Graph

```
Phase 01 (Foundation)
├── Phase 02 (Sources + DataFusion)
│   ├── Phase 04 (InferenceExec) ← also depends on 03
│   │   ├── Phase 05 (Embedding Pipeline)
│   │   │   └── Phase 06 (Vector Search + Evidence)
│   │   │       ├── Phase 08 (Evaluation) ← also depends on 07
│   │   │       └── Phase 12b (Caching & Hardening) ← also depends on 02, 04
│   │   └── Phase 07 (Fine-Tuning) ← also depends on 03
│   └── Phase 12a (External Sources & Backends) ← also depends on 04
├── Phase 03 (Model Loading)
├── Phase 09 (GPU Scheduling) ← depends on 04, 05, 07, 08
├── Phase 10 (Python Bindings) ← depends on 01–09
├── Phase 11 (Server + CLI) ← depends on 01–09
└── Phase 13 (Enterprise Platform) ← depends on 01–12b
    └── Phase 14 (Enterprise Dashboard) ← depends on 13
```

12a and 12b are independent of each other — they can be implemented in either order or in parallel within CP5b.

## Checkpoints & User Acceptance Testing

Implementation is verified at 7 checkpoints. Each checkpoint has a focused UAT guide that tests only new functionality. Regression is automated via `cargo test --workspace` and a growing smoke test (`cargo test --test smoke`, introduced at CP3).

| CP | Phases | Name | UAT Guide |
|----|--------|------|-----------|
| 1 | 01–02 | Data flows | [UAT-CP1](cp1-data-flows/UAT-CP1-data-flows.md) |
| 2 | 03–04 | Intelligence runs | [UAT-CP2](cp2-intelligence/UAT-CP2-intelligence.md) |
| 3 | 05–06 | Core product loop | [UAT-CP3](cp3-core-loop/UAT-CP3-core-loop.md) |
| 4 | 07–08 | Learning & measuring | [UAT-CP4](cp4-learning/UAT-CP4-learning.md) |
| 5a | 09–10 | Python API | [UAT-CP5a](cp5a-python-api/UAT-CP5a-python-api.md) |
| 5b | 11, 12a, 12b | Server & hardening | [UAT-CP5b](cp5b-server-hardening/UAT-CP5b-server-hardening.md) |
| 6 | 13 | Enterprise platform | [UAT-CP6](cp6-enterprise/UAT-CP6-enterprise.md) | [Enterprise repo setup](cp6-enterprise/PLAN-00-enterprise-prerequisites.md) before Phase 13 |
| 7 | 14 | Enterprise dashboard | [UAT-CP7](cp7-dashboard/UAT-CP7-dashboard.md) |

**Workflow at each checkpoint:**

```
1. cargo test --workspace          ← automated regression
2. cargo test --test smoke         ← end-to-end integration (from CP3)
3. python3 tests/smoke_test.py     ← Python regression (from CP5b)
4. [Manual UAT items]              ← new functionality only
```

## TDD Workflow

Every phase is implemented test-first. For each phase:

```
1. Read TEST-XX     → write test file(s) from the test plan
2. Write stubs      → minimal types + signatures so tests compile (return todo!())
3. Run tests        → all fail (red)
4. Read PLAN-XX     → implement until tests pass (green)
5. Refactor         → clean up, cargo clippy, cargo fmt
6. Document         → rustdoc on all new public items, update guide if checkpoint
```

Each TEST-XX file specifies:
- Test file path(s)
- Stubs needed to make tests compile
- Complete test code with assertions

Each PLAN-XX file specifies:
- **Goal**: What this phase delivers (testable outcome).
- **Files**: Exact files to create or modify, with purpose.
- **Implementation Details**: Key decisions, data structures, trait implementations, with code sketches where clarity demands it.
- **Acceptance Criteria**: How to verify the phase is complete.
- **Cross-Phase Dependencies**: How this phase's interfaces connect to other phases.

## Interface Discipline

Five rules govern how phases expose interfaces to each other:

1. **No temporary APIs.** No phase may expose an API that a later phase is expected to replace. If a subsystem's interface will evolve, define the final interface early with a minimal implementation. Example: `GpuScheduler` is defined in Phase 03 with an unlimited implementation; Phase 09 upgrades it to memory-budget admission. One interface, two implementations — no dead code path to remove.

2. **Downstream-driven.** Only add methods to a trait or public type when a dependent phase actually consumes them. "Might be useful later" is not a reason to expand an interface.

3. **Trait semantics are frozen once shipped.** Concrete types behind a trait may evolve (new fields, better algorithms), but the trait's method signatures and behavioral contracts may not change without updating all dependent PLAN and TEST docs.

4. **Cross-phase boundaries require contract tests.** A downstream phase depending on an interface must have tests that prove the interface's behavioral invariants — not only end-to-end smoke tests that happen to exercise the path.

5. **Tests do not justify API expansion.** If a test needs access to internals, add a `#[cfg(test)]` helper or a test-only module. Do not add production methods to satisfy test needs.

### Scheduler Interface Rule

By the end of Phase 03, all GPU-sensitive components depend on the `GpuScheduler` interface (defined in `concurrency/gpu_scheduler.rs`). Phase 03 provides `GpuScheduler::new_unlimited()` — a trivial implementation that always grants permits immediately. Phase 09 adds `GpuScheduler::new(total_memory, headroom)` with CAS-based memory-budget admission. `ModelCache` always receives a scheduler (no `Option` wrapping, no dual code paths). Operators always acquire activation permits through the same interface.

## Test Taxonomy

Every test belongs to exactly one category. The category determines where it lives, how it runs, and what it may depend on.

### Unit tests

- Pure logic, no network, no filesystem except `tempdir()`
- Deterministic and fast
- Run on every PR

Examples: backend selection heuristics, schema construction, batch formation, evidence merge logic, config parsing, model dimension estimation, adapter output schemas.

### Contract tests

- Validate behavioral invariants of trait implementations
- Named `*_contract_*` in the test file
- May use fixtures but no network
- Run on every PR

Examples: every `ModelBackend` impl satisfies load/estimate_memory semantics. Every `OutputAdapter` produces columns matching its `output_schema()`. Every source provider returns expected schema/scan behavior. Cache residency transitions are valid state machine transitions.

### Integration tests

- Filesystem, Parquet, SQLite/catalog, model fixtures, local HTTP mocks
- Multiple crates/components together
- May be slower
- **Network-dependent integration tests** (HF Hub, real model downloads) are gated behind `#[cfg(feature = "live-hub-tests")]` and do NOT run on default `cargo test`

Examples: `JammiSession` + source + resolver + operator. Cache + scheduler + load lifecycle. Vector search + evidence flow. Full inference pipeline with real model weights (feature-gated).

### Smoke tests

- Cross-phase regression gate
- One realistic end-to-end workflow per checkpoint boundary
- Introduced at CP3: `cargo test --test smoke`

### Live canary tests

- Real HF Hub downloads, real model inference, real external services
- Behind `--features live-hub-tests`
- Run nightly and pre-release, not on every PR

## CI Matrix

| Trigger | What runs |
|---------|-----------|
| PR | `cargo fmt --check`, `cargo clippy --workspace`, `cargo test --workspace` (hermetic: unit + contract + hermetic integration + smoke) |
| Nightly | PR suite + `cargo test --workspace --features live-hub-tests` (live canary tests, heavier concurrency stress) |
| Release | Full nightly suite + UAT checklist |

## Catalog Migration Protocol

The catalog schema uses `rusqlite_migration` with the `user_version` pragma. Migrations are numbered sequentially and registered in `crates/jammi-engine/src/catalog/migrations.rs`. `Catalog::open()` calls `to_latest()`, which applies only unapplied migrations in order.

**Rules:**

1. Each phase that modifies the catalog schema adds the next numbered migration.
2. Migration numbers are assigned in phase order and recorded in the Phase Sequence table above (Migration column).
3. A user upgrading from CP3 to CP5 gets all intermediate migrations applied automatically on first `Catalog::open()`.
4. Migrations are append-only — never modify a shipped migration. Schema fixes go in the next migration number.
5. Phase 01's `MIGRATION_001_CORE_TABLES` creates all initial tables. Later phases add columns or new tables as needed.

**Current migration sequence:**

| Migration | Phase | What it does |
|-----------|-------|-------------|
| 001 | 01 | Core tables: sources, embedding_sets, models, fine_tune_jobs, eval_runs, evidence_channels |
| 002 | 08 | Adds `golden_source` and `k` columns to eval_runs for comparison tracking |

Enterprise has its own migration sequence in `enterprise.db` (Phase 13) — completely separate from the OSS catalog.

## Documentation

Documentation is built incrementally alongside implementation — never deferred to the end. Three layers, different cadences:

### Layer 1: Rustdoc (every phase)

Every public type, trait, method, field, and enum variant gets a `///` doc comment. No phase is complete without this.

- One-line for simple items, 2-3 lines max for complex ones. Imperative mood.
- Config fields include default values in the doc comment.
- Crate-level `//!` doc comments describe each crate's purpose.
- Verified: `cargo doc --workspace --no-deps` must produce zero warnings.

### Layer 2: mdBook guide (per checkpoint)

The user guide lives at `docs/guide/` (mdBook). Add or update a chapter when a checkpoint lands with new user-facing capability.

- Chapters are 1-2 pages, not novels. Code examples over prose.
- Current chapters: Introduction, Getting Started, Configuration, Data Sources, Embedding Inference, Architecture.
- Planned additions: Vector Search (CP3), Fine-Tuning & Evaluation (CP4), Python API (CP5a), Server & CLI (CP5b).

### Layer 3: Examples (from CP3 onward)

Runnable Rust examples in `examples/` directory (`cargo run --example`). Start when the core loop (source → embed → search) is complete. More valuable than prose for users.

### What NOT to document

- Internal modules — only public API boundaries
- Architecture that duplicates PLAN files
- Reference docs by hand — rustdoc handles that
- Phases that aren't built yet

## Checkpoint Exit Criteria

Each checkpoint is an engineering gate, not just a milestone. A checkpoint is complete when:

1. **Interface boundaries are frozen.** Public traits, types, and method signatures consumed by later phases are stable. Any breaking change requires updating dependent PLAN and TEST docs.
2. **Invariants are enumerated.** Each invariant is a testable statement about the system's behavior at this checkpoint.
3. **Contract tests prove the invariants.** Not just end-to-end tests that happen to pass — explicit tests for each enumerated invariant.
4. **Regression passes.** `cargo test --workspace` + `cargo test --test smoke` (from CP3).
5. **Documentation updated.** Rustdoc on all new public items (`cargo doc` zero warnings). mdBook guide chapter added or updated if the checkpoint introduces user-facing capability.

Each UAT file includes an "Exit Criteria" section with the frozen interfaces, enumerated invariants, and the contract tests that prove them.
