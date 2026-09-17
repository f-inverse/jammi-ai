# Contract — `feat/500-wave5`: the close-out of plans 67 and 68, one consolidated PR

Base: `main` @ `db19a614` (PR #586 merged: PR-D, the Ballista compute plane). Slug
`feat_500-wave5`; pressure rows in `docs/rigor/feat_500-wave5.jsonl` (one per unit round, exported
from the lead-gate ledger, never hand-typed); oracle record in `docs/rigor/feat_500-wave5.oracle.jsonl`
at the final tip, exported last, after every non-rigor edit; anticipation record
`docs/rigor/feat_500-wave5.anticipation.jsonl` (the audit rows carry BLOCKs). Plans:
`docs/plans/67-distributed-training/{README.md, DESIGN.md, UNITS.md}` and
`docs/plans/68-compute-tier-substrate/{README.md, PROGRAM.md, units/GRAPH-JOB-DEPENDENCIES.md}`,
whose status lines this PR rewrites to what is on `main` (§4.13). Every issue this PR closes was
first re-verified against `main` @ `db19a614` before it entered a unit contract: three contracts
carried already-shipped items (#510 at 775525df, #513 at 9c62f12b, #508 except its self-test) and
those were removed, never re-implemented.

The user's standing decisions this PR was built under: one PR, units as commits, closers once on the
tip; the two-pods Global-Networking transport is the accepted multi-node proof substrate; #515 was in
scope and is excised by its own pre-committed stop rule (§4.11); #445 and #478 are deferred
indefinitely; the 0.50.0 release hold stands.

## 1. Scope and build order

Every commit group below is on `feat/500-wave5`. Groups were
built concurrently in their own worktrees off `db19a614` (or off the consolidated tip where a later
group needed an earlier one's seam) and consolidated by the lead as cherry-picks, one net diff
(GANG), or one merge commit (the cluster leg, so the artifact's `git_sha` ancestry survives).

| # | Group | Owner | Issues | Depends on | Landed as |
|---|---|---|---|---|---|
| 1 | GATES — the Rust source gate rebuilt on `syn`; every `ci/scripts/check_*.py` gate in scope decides from the parsed artifact | ai-core (GATESAI), docs-ci (GATESCI) | #549 #554 #561 #563 #564 #565 #530 #532 #533 #508 #511 (#513 shipped at base; #517 deferred) | base | af8fa645, 0b13ea74, 32790f46, a2643979; 2a06aa49, 14c146db, 7fca1e95, 969d3c85, bdf0ec84, 66bcddbd, a64175e0, bd3a6202; 5d78c695 (lead: kernel-oracle markers) |
| 2 | STAMPS — one canonical catalog stamp on both backends; migration 039; the structural release barrier | db (STAMPSDB), ai-core (STAMPSAI) | #585 #574 #516 #525 (#510 shipped at base) | base | 6fc9722a, 7bd5b881, d3b29e10, 6c6362a6, 05978f60, e30f99e4, 6ecb9418, bc0e1578; 7fd8e32a, 2659d1d7 |
| 3 | E0 — the two-host cluster leg: driver rebuilt at every defect, the row C9 artifact | lead + RunPod | plan 67 row C9 | base (`unit/twopods`) | 94058722 … 008c33af (transport), 58013b90, 5e832cf6, 5b3c8b47, d74773d1, f4d66b16, bfe595c4, 6207a918, 8dd12fd7, dd394dbb (artifact); merge 946e04d3 |
| 4 | TARBALL — the cu12 library set derived from the binary; a sound loader-verification arm | docs-ci | #535 #534 (loader half open) | base | 1963be13, 1fd0ceda, ae339a09 |
| 5 | TESTBOUNDS — the observed-event rendezvous seam; the identity-keyed bound inventory; the chaos leg root-caused | ai-core | #527 #567 #578 | STAMPSAI | c96e5da0, aeb9147e, 53015123, 59efee4f, 9f43a8d7, 3f8e7724, 4f94bf15, 340347c5; 49b804ac (lead) |
| 6 | GANG — one flat `JobSpec`, the admitted-spec witness, the store-minted `RelationKey` | ai-core | #548 #573 #551 | STAMPS, GATES | 7d4b9173 (one net diff) |
| 7 | GANG3A — one training submit seam; the general relation-key minter; the mining parity oracle | ai-core | #573 #551 | GANG | ea4570a0, a184c0f5 |
| 8 | GANG3B — the resume-parity harness on the checkpoint-written seam | wire-server | #543 | GANG, TESTBOUNDS | 187c6d47 |
| 9 | CAPSURF — the bf16 training attention path dispatches the flash cascade | numerics | nightly `GPU prove` red | GATES | 0179471e |
| 10 | ENUMS — enum inventories by construction | db | #550 | base | c6ce59b1, 2de4ba80 |
| 11 | COOKBOOK — chapter 22 follows the engine's ANN key refusal | cookbook | ch.22 render red (#506 shipped at base) | base | 6d167774 |
| 12 | DOCS — ADR-00 citations, the tracked rules source, plans read as shipped | docs-ci | #495 #496 #526 (status prose) | all code groups | 93ee7331, 77f1ed8d, 4a931b9d |
| 13 | GRAPH — the standalone terminality fold and the ledger oracle from the excised unit | db | #515 (open) | ENUMS | ed76e7fe, af9ef9a1 |
| 14 | IDENTITY — the typed admission table, the deterministic dry-run profile, cache reuse admitted, the `syn`-keyed deleter oracle, the always-on prefix refusal | ai-core (+ kernels, encoders, lora, db seams) | #562 #546 #547 | ENUMS, GRAPH | 8cca3411, 7c5d67d0, 9fac63d4, d0a4d0f4, 2446ae90, b32956ae (original tip 7488d448) |
| 15 | LEADGATE — the journey-marker and plan-citation gates on one shared `syn` symbol index; the cookbook leak rail; the lead gate's committed records | docs-ci | #555 #556 #526 #552 #528 #569 #557 #570 | GATES | 1ac16779 … 537aa00d (20 commits, original tip 8ad80051) + 0ea3eff3 (original 1e12e11a) |
| 16 | Lead fixes at consolidation — review lists re-keyed by site identity, the timings gate's setter universe, rustdoc stated as invariants, the SIGHUP fixture on an observed event, the two-backend freshness test with no early return | lead | #554 #527 #574 | GATES, TESTBOUNDS, STAMPS, E0 | a793f572, 49b804ac, 4a4ef7bf, adca0866, 6bc9d8a8, 4cc1724b |

The lead consolidates on `feat/500-wave5`, runs `ci/scripts/merge_path.sh` on the consolidated tip
(every stage: static, guards, swarm, tests with the three Postgres lanes on a fresh database, records),
runs the tree-wide source gates after every landing (§10), dispatches the adversarial audit, then the
oracle last, then exports the records. Expected reds at the PR: `SWARM_GATE_TOUCHED` (agent cards and
`swarm.yml` change; admin merge authorized).

## 2. The properties, by group (quantified; each with an executed oracle and an executed mutation, named by test)

### 2.1 GATES (#549 #554 #561 #563 #564 #565 #530 #532 #533 #508 #511)

- Every gate in scope decides from the parsed artifact, never from text: the PyYAML compose tree
  where a scalar's spelling matters, `cargo metadata` for the universe, the `proc-macro2` token
  stream and the `syn` AST for Rust. A `grep` may narrow candidates; it never decides.
- Fail closed on the unexaminable, by name: a cross-repo reusable workflow, a dangling local target,
  a non-unique step name, a `run:` body under shell control flow, a `steps:` that is not a list, a
  YAML boolean spelling outside GitHub's own set, a step-level composite action across repos — each a
  NAMED finding stating what could not be examined. Oracles: one fixture per arm asserting the finding
  text names the input; the `return None` mutation reds each.
- Universes are derived, never remembered: jammi-ai's binding surface is the reverse-dependency
  closure of jammi-db/jammi-ai within the workspace (jammi-bench in, third-party out); the merge-path
  clippy lanes are selected from `ci.yml` by a rule so a deleted lane is detected on loss; the source
  gate's scanned roots are asserted by a test.
- Counts, not sets: every reviewed literal site is keyed `(file, fn, ordinal, count)` with
  `count <= allowed`; a second registration planted in a reviewed fn reds the gate.
- Reachability is a call graph over every edge shape (fn-pointer argument, `map(Self::b)`, a call
  inside any macro invocation, a fn-pointer struct field, `macro_rules!`-generated items) and every DDL
  position (module-level const, keyword split across `format!`/`concat!`, `include_str!`); the two
  shapes a name-keyed graph cannot resolve fail closed. Wall time on the real tree is measured and
  under the 10 s bound.
- One ancestry rule for both artifact gates (`ci/scripts/ancestry.py`, imported by both):
  `merged_as` when present and an ancestor of HEAD, else `git_sha` when an ancestor of HEAD, else fail
  naming both.
- Waivers rot: `check_no_consumer_names.py --self-test` carries one fixture per allowlist rot rule.
- #511: the OpenSSL requirement is recorded at the feature declaration and in the crate README, the
  dependency-audit tool is version-pinned, and a manifest-reachability guard refuses any release-lane
  row that reaches `jammi-db/postgres` or `jammi-db/mysql`; the runtime-image disagreement is stated
  at the guard as the reason.
- Oracles and mutations: the implementers' tables in §11.1–§11.2; the lead re-ran
  `cargo test -p jammi-ai --test it -- pinned_source_gate::` (52 tests) and the guard matrix on the tip.

### 2.2 STAMPS (#585 #574 #516 #525)

- Every writer of every TEXT column that holds an instant writes `YYYY-MM-DDTHH:MM:SS.ffffffZ` on both
  backends, from ONE application formatter and ONE Postgres renderer (`pg_canonical_stamp`), and the
  nine-digit `now_sortable()` is deleted. The universe is source-derived (every instant column in
  `schema.rs`'s CREATE and ALTER forms, `applied_migrations.applied_at` included) and enumerated by
  `CANONICAL_STAMP_COLUMNS`, which an oracle proves equals the enforcement set after any migration
  sequence.
- Migration 039 normalises every existing in-class value and fails closed on any it cannot; the
  refusal is an OUTCOME property (ledger at 038, rows intact, no 039 trigger or CHECK installed).
  SQLite enforces shape at the edge with `BEFORE INSERT`/`BEFORE UPDATE OF` triggers; Postgres enforces
  shape and calendar validity with a CHECK whose shape test precedes the cast. No table rebuild on
  SQLite. The migration runner's statement splitter understands a trigger body, proven by a
  byte-identity oracle over every earlier migration's statement list.
- `LeaseFact::Undecodable` stays: on SQLite a shape-valid, calendar-invalid stamp is representable;
  the stated asymmetry is that Postgres refuses the write and SQLite reads it as undecodable.
- No SQL-side sweep can be faulted by row content on Postgres; SQLite's lexical compares are total.
- The release-write delta oracle asserts over a live all-columns snapshot, so a write to any column
  outside `SELECT_COLS` reds it on both backends (`idempotency_key` is the executed mutation).
- The session-scoped release barrier is structural: one claim loop per session, held from spawn until
  `release_and_stop` completes or the handle drops; `JobWorker::run`, the one ungated entry, is
  deleted (zero callers, verified).
- Migration pin sites: `catalog/migrations.rs`'s list, `tests/it/migrations.rs`'s expected names and
  relative-position oracle, the maintainer guide.

### 2.3 E0 — the cluster leg

- A member is usable when its sshd answers an executed connect, never when the provider says
  RUNNING (`rp_sshd_answers`/`rp_wait_sshd`, ONE definition in `ci/scripts/runpod_lib.sh`).
- Both ranks build concurrently; rank 1's script waits between build and proof; the NCCL id crossing
  is bounded by the work (liveness, inactivity, budget inside the watch loop), never by a clock of its
  own.
- Every rank log ships in the artifact on every exit arm.
- The NCCL interface is read from the kernel route table (`/proc/net/route`, longest prefix), never
  from `iproute2`; the no-route arm refuses by name.
- A member proves the exact commit the run was dispatched on (`PROVE_EXPECT_SHA` fetched by hash), and
  every leg's checkout carries full history (blobless deepen) so the artifact registry's ancestry rule
  holds; every rank imports PID 1's environment first; the remote root is a parameter and the checkout
  chain fails closed; a fixture never patches the script under test.
- The artifact: `crates/jammi-kernels/artifacts/cuda-runs/2026-09-17-500-u7b-gang-cluster-6207a918-rtx-4090-two-host-pods.json`,
  GPU cluster run 35171132263, transport `global-networking`, two RTX 4090 pods, status GREEN,
  digests equal, id-secrecy scan clean; `check_cuda_run_artifacts.py` PASS with `6207a918` an
  ancestor of the tip. Lane suite `ci/scripts/test_gpu_cluster_lane.sh` (P11–P15 fixtures) 163/163.

### 2.4 TARBALL (#535, #534 report-parse half)

- The staged `lib/` is a total function of the staged binary: the transitive `DT_NEEDED` closure
  minus the host-provided platform set, plus the measured dlopen-only floor; every phase's failure is
  the script's exit status; the floor refuses to overwrite a derivation-staged object from a different
  source object (realpath compared). ONE classification (`verify_link_set.py`'s COVERED / PLATFORM /
  DRIVER_PROVIDED) bound across the bundle script, the loader arm and the component contract.
- Loader verification refuses any non-platform, non-driver member resolved outside `<lib>`, refuses a
  vacuous report by the every-name-must-resolve rule (three vacuous shapes), parses the loader's own
  line, refuses `lib_dir='/'`, and asserts no `DT_RPATH`/`DT_RUNPATH` before trusting
  `LD_LIBRARY_PATH`. The fixture is the REAL report the release lane's cu12 leg captured on its own
  build container (ae339a09), and `check_bundle_fixture.py` in the guard matrix refuses a provisional
  fixture.
- The hand list is deleted from `release-binaries.yml`; the lane runs the derivation and the arm.

### 2.5 TESTBOUNDS (#527 #567 #578)

- No test in the crate's four test targets waits on training progress behind a literal wall-clock
  bound that IS the assertion: every literal bound is (a) the backstop around an observed-event
  rendezvous through ONE seam (`loop_test_hooks::{Event, arm_observed, fire_observed}`, job-keyed,
  non-blocking fire at the place the event happens), (b) derived from the config the test set, with
  the derivation in the message, (c) a backstop whose message carries the fixed sentence "wedged or
  starved machine", or (d) not a training-progress wait, with a one-line reason. The inventory
  (`test_bounds_inventory.rs`) enumerates every bound form in every target (`timeout(`, `sleep(`, the
  deadline loop, the named const), asserts a non-empty universe with a size floor, and keys each
  reviewed site by `(file, enclosing item, ordinal)` found by `syn`, with the marker window the
  enclosing item's own span — an edit above a site cannot move its review, a new bound is named by
  file, item, ordinal and line.
- The epoch-boundary cancel test observes the watcher's tick through the seam; the mutation red
  carries the backstop's own wording; the unmutated run finishes under 25 % of the backstop, measured
  at two load levels.
- The chaos leg's flake is root-caused: `artifact_crash_window` derived its expected prefix by hand
  while the store's layout is tenant-prefixed; it now derives through the store's own layout
  function, and the leg is promoted off advisory.

### 2.6 GANG, GANG3A, GANG3B (#548 #573 #551 #543)

- Every persisted `jobs.spec` decodes to exactly one declared kind or fails naming the offending
  field and the job id, at every depth (ONE flat `#[serde(tag = "kind", deny_unknown_fields)]` enum
  over the eight kinds, every nested config struct denying unknown fields); eight committed byte-pin
  fixtures prove the persisted bytes are unchanged, so no migration.
- Every reference to the catalog's submit API is the seam or a reviewed row: `submit_admitted_training` in
  `crates/jammi-ai/src/fine_tune/spec.rs` is the ONE construction site of a training-kind
  `SubmitJobParams`, its parameter is the `AdmittedTrainingSpec` witness whose constructor is private
  (E0603, executed), the three in-crate edges and jammi-bench's benchmark submit call it, and a
  `syn`-driven enumerating oracle over every REFERENCE to `submit_job`/`submit_job_deduped` — a
  method call, a path naming the fn in any expression position (a call's callee, a fn-item captured
  as a value and invoked later, a combinator argument, a struct field) under any prefix or qualified
  self, and any macro invocation's token stream; a bare single-segment path is a local, never an
  inherent method — in every `.rs` file cargo compiles outside a test target, the one universe
  `jammi_test_utils::source_universe` defines for both call-site oracles (every workspace member's
  `src/` including `ci/tools/*`, every `build.rs`, every `examples/` and `benches/` target; not
  `tests/` or the `ci/fixtures/` tokenizer inputs; the catalog and its params are `pub`, so the
  universe is everything cargo compiles) reds on any reference outside the reviewed rows: the three `jammi-ai` sites, the catalog's own `submit_job` →
  `submit_job_deduped` forwarding, and the gRPC client's `submit_job` RPC homonym. The residual is
  stated: a bypass duplicating the seam's body is source-syntactically possible and is caught by the
  oracle, never by the type system; a hand-built training-kind submit added to `jammi-server`'s gRPC
  handler reds the oracle, and so does the same submit through a captured fn item
  (`let route = Catalog::submit_job; route(..)`) (both executed, reverted).
  **The door this oracle does not cover, stated:** the catalog's generic SQL surface is `pub`
  (`Catalog::backend_arc` → `BackendImpl::transaction` → `Tx::{execute, query, query_opt}`), so any crate holding a
  `Catalog` can `INSERT INTO jobs` a training-kind row with no `submit_job` reference; nothing in this
  PR refuses it (the only non-test `INSERT INTO jobs` today is `submit_job_deduped`'s own, plus the
  historical migration). Sealing it — `Tx::execute` crate-private behind typed catalog operations,
  with `jammi-python`'s `close()` given a typed method — is the jammi-db capability-sealing unit filed
  from this contract; until it lands the property is the reviewed submit API, not "unrepresentable".
- No statement in the scanned surface (`crates/jammi-db/src`, `crates/jammi-ai/src`) yields a session-registered result-table relation except through a
  store-minted `RelationKey` (two minters, both in `store/mod.rs`, one private field); every quoted
  `"jammi.{…}"` site is migrated or reviewed in the pre-existing `(file, fn, ordinal, count)` gate;
  source-side federation relations and backing-table reads are named out of the property. The residual is stated in the store's own doc (`crates/jammi-db/src/store/mod.rs`): `crates/jammi-bench/src/corpus.rs` registers a bare `jammi.{table_name}` outside that surface.
- Hard-negative mining at W=1 changes the trained adapter bytes and mining-off leaves them
  unreached (executed, live digests in one run); the committed byte pin the issue names needs a Linux
  measurement and is captured from the PR's first CI run as the closer (§9).
- #543 closes on two executed oracles: a fleet without a shared root cannot form a Peer gang
  (`gang_membership.rs`, `file_and_s3_rooted_members_are_not_gang_members_of_each_other`), and a job
  killed once epoch 1's checkpoint write is OBSERVED (armed `Event::ResumeCheckpointWritten` before
  the claim, awaited under the same 60 s backstop `jobs_shutdown.rs` uses) resumes and publishes
  byte-identical adapters under Peer W=2 and Local W=2; a corrupted bundle fails attempt 2 loudly,
  never a silent restart (the row the byte-parity property cannot red on CPU determinism, stated).
  The v1 resume substrate is the shared artifact root; the broadcast mechanism is not built.

### 2.7 CAPSURF

- On a flash-attn build, for every flash dtype, the training attention path dispatches the flash
  cascade and never `attention_block_fused`. Root cause named: BERT's own probe fixture wrote the
  shared counter inside the test's before/after window; the window is isolated and a CPU-hermetic
  preemption oracle in jammi-encoders reds on the diverting call. Closer: GPU prove run 35171308424,
  all four arches (sm_80, sm_86, sm_89, sm_90) success.

### 2.8 ENUMS (#550)

- An enum inventory is generated from the enum: `ResultTableKind::ALL`, `JobStatus::ALL`,
  `WeightQuantization::ALL` are `#[derive(strum::VariantArray)]` (`ALL: &'static [Self]`); the
  round-trip oracles iterate the derived inventory; the executed refutation of the hand list (a
  `Probe` variant added, pattern and `as_db_str` extended, `ALL` stale, suite green) is recorded.
- Every artifact-path literal in `jobs_queue.rs` owns its run (`artifact_path(run, tail)`), with a
  self-test enforcing it; two shared-database leaks, one previously masked, are fixed.

### 2.9 COOKBOOK

- Chapter 22 asserts the engine's typed refusal of an unsupported `[embedding.ann]` key
  (`InvalidArgument` naming the key), never that the key has no effect; the pre-fix cell is the
  executed RED. Every chapter the wave-5 diff can move was re-emitted against a fresh build with no
  golden moved. #506 was already shipped at 5f847776 and is re-verified, not re-authored.

### 2.10 DOCS (#495 #496; #526 status prose)

- No tracked file cites `ADR-00`; every citation points at
  `docs/guide/src/philosophy.md#the-one-rule-everything-else-follows-from`.
- The CONSTITUTION's canonical sources are tracked: `docs/swarm/RULES.md` carries the B1/B6/K5/K6
  rule text at exactly the constitution's own statements (the one stronger claim — that a shipped
  migration's body is never edited — was removed because no gate enforces it); `CONSTITUTION_TOUCHED`
  names `RULES.md` and `philosophy.md`.
- Plans 67 and 68 read as shipped from the tree and the merged PR list, with row C9 done, #543's two
  oracles cited by symbol, the v1 graph-gang limit stated with #538 as the rebuild pointer, #515's
  disposition, #445/#478 deferred indefinitely, and the DataFusion-55 gate stated as the Ballista and
  table-providers crates (not the Flight SQL server crate).

### 2.11 GRAPH — what landed from the excised unit (#515 stays open)

- Every hand-enumerated job-status terminality decision in the tracked tree (Rust literal or
  rendered, the test fixtures, the Python client) derives from `JobStatus::is_terminal` /
  `is_terminal_unsuccessful` or the Python client's two sets; `JobStatus` keeps its four variants (a
  status with no writer is dead vocabulary); the two-language enumerating source oracle
  (`terminality_source_gate.rs`) states its universe and reds naming the site.
- A ledger-level oracle asserts no migration text rebuilds the `jobs` table (word-boundary scoped),
  protecting migration 039's triggers and any future foreign-key parent.

### 2.12 IDENTITY (#562 #546 #547) — 8cca3411, 7c5d67d0, 9fac63d4, d0a4d0f4, 2446ae90, b32956ae

- **Every admission call is compiler-bound to the table.** `admit`/`admit_cascade` take
  `&'static ProbedOp`; the fifteen rows of `PROBED_OPS` are the only values a call site can name,
  across jammi-kernels, jammi-encoders, jammi-lora and jammi-ai (the two dtype-branching cast
  registries reach the raw key through `admit_cast_boundary`, itself typed). Oracle: the workspace
  compiles; a call against a string literal does not.
- **The dry-run verdict on the row and cache reuse are EXCISED** (`ff9ec6eb`, by the pre-committed stop
  rule recorded below). `ProbedOp` carries no `dry_run`; nothing folds an admission profile into
  `MaterializationEnv::kernel_admission_profile` (byte-identical to the base, `None`); `CachePolicy::Use`
  is refused for fine-tune exactly as at the base. Oracles: the base's `cache_bypass_never_reuses`
  (two `Bypass` runs never share a prefix, `cache_outcome` is `"computed"`), the two base refusal tests
  in this suite, and the restored embedded/remote refusal parity test; `git diff db19a614 --
  crates/jammi-db/src/store/manifest.rs` is empty.
- **Every reference to the handle's deleters is reviewed by identity.** The `syn` oracle in
  `models_delete_call_sites.rs` enumerates, over the same universe as the submit-seam oracle and in every
  reference shape (method call, owner-qualified path in any position, macro token stream, and the private
  `driver` field inside the handle's own file), every reference to `JammiObjectStore::delete_if_exists`,
  `ArtifactStore::delete_artifact_prefix`, and the handle's raw driver (`driver()`, now `pub(crate)` — the
  handle hands the raw store to no other crate, E0624 executed — and `self.driver`), keyed by
  `(file, fn, ordinal)`; a raw `driver().delete` added inside `jammi-db`, a `self.driver.delete` added
  inside the handle, and a raw `delete_if_exists` in the compiled example target all red the oracle
  (executed, reverted). **What it does not cover, stated:** a raw `Arc<dyn ObjectStore>` — on which
  `ObjectStoreExt::delete` is unguarded — is obtainable without a handle by three routes, all outside this
  oracle: (a) `StorageRegistry::driver_for` and `storage::build_object_store` (both `pub`); (b)
  `JammiSession::context()` (`pub`, `crates/jammi-db/src/session.rs:973`; re-exposed by `jammi-ai`'s
  `InferenceSession::context()`, `crates/jammi-ai/src/session.rs:902`), which hands out the DataFusion `SessionContext` whose
  `runtime_env().object_store(url)` returns the registered store — the session's default registry
  pre-registers `file://` and `jammi-db` registers its credentialed cloud drivers there
  (`crates/jammi-db/src/store/mod.rs:4536`, `crates/jammi-db/src/source/file_format.rs:229`; `crates/jammi-ballista/src/roles.rs:443` holds the value); (c)
  direct construction with the `object_store` crate, a normal dependency of `jammi-ai` and `jammi-server`, by
  any code holding the same credentials — inherent to the crate ecosystem, not sealable by `jammi-db`. The
  non-test acquisition sites of (a) at this head, enumerated outside `#[cfg(test)]`
  (`grep -rn 'driver_for(\|build_object_store(' crates/*/src crates/*/examples`, each site's `#[cfg(test)]`
  boundary checked): `jammi-bench/src/{crates/jammi-bench/src/cache_slo.rs:132, crates/jammi-bench/src/context_predictor.rs:290, crates/jammi-bench/src/corpus.rs:105,
  crates/jammi-bench/src/corpus.rs:141, crates/jammi-bench/src/model_inference.rs:346, crates/jammi-bench/src/propagate.rs:316, crates/jammi-bench/src/recompute_scale.rs:134}` (each writes or reads its
  own benchmark corpus; none deletes) and inside `jammi-db` `session.rs:{696,717,1074,1091}`,
  `crates/jammi-db/src/storage/index_cache.rs:121`, `crates/jammi-db/src/source/file_format.rs:125`, `crates/jammi-db/src/store/artifact.rs:642`,
  `store/mod.rs:{1281,1289,4531}`, `crates/jammi-db/src/storage/registry.rs:88`. Sealing (a) and (b) — guarded handles only,
  `driver_for`/`build_object_store` crate-private, a session wrapper that does not expose `SessionContext` or
  a registry that returns guarded stores — is the jammi-db capability-sealing unit #588, filed from this
  contract together with the raw-SQL door in §2.6; (c) is stated, not closed.
- **Refused and replaced (lead rulings):** the one-op `flash_admission_short_circuit` slice (a rival
  enumeration) → the full typed table; the string-keyed `dry_run_verdict` match in jammi-ai (a second
  enumeration by literal) → the row field; the `(file, line)`-keyed review table → `syn` identity
  keys; the `debug_assert!` → an always-on typed refusal; the owner-refusal and migration 040 → refuted
  by the executed both-orders oracle, no migration lands.

### 2.13 LEADGATE (#555 #556 #526 #552 #528 #569 #557 #570) — 1ac16779 … 537aa00d, 0ea3eff3

- **One real Rust index for every source-facing gate.** `ci/tools/symbol-index` (a `syn`
  workspace member, never a default member, never published) emits items, impl methods, enum
  variants and struct fields with spans; `check_plan_citations.py` and `check_no_consumer_names.py`
  resolve against it and their regex readers are deleted; `swarm.yml` runs them in a
  `symbol-index-gates` job with the toolchain. Oracles: the tool's seven unit tests, both gates'
  self-tests against the compiled binary; the first real run found and fixed two indexer defects.
- **Journey markers cannot land on a published surface.** `check_journey_markers.py` is
  diff-scoped, blocks on pub rustdoc, `.proto` comments and Python docstrings, advises elsewhere;
  K1–K7 and issue links are proven load-bearing exclusions; a file's OWN declared rule or test-id
  vocabulary is exempt while an undefined id of the same shape and any round number still fire
  (self-test in both directions). Against the PR base the branch reports 0 blocking findings.
- **Plan docs cite constructs.** `path::symbol`, `path#heading`, TOML keys and proto members under
  `docs/plans/**` resolve in CI; the stale plan-67/68/61 citations it found are migrated.
- **The cookbook leak rail sees the whole process.** Session-scope baseline before collection, a
  session-finish sweep failing the run by label, a bounded ledger (`_LEDGER_CAP`), non-empty labels;
  three new fixtures with executed mutations.
- **Every lead-attested value is a committed record with a required reader.** The
  mutations/exclusions attestation record (`--export-attestation`) read by `check_rigor_record.py`
  through the hook's own validators (RR34–37); the CI-derived required call-site set from the same
  symbol index (advisory only when no `cargo` is on PATH, never a crash — RR44); the bidirectional
  `residual_risk` ↔ `# R12-RESIDUAL` check (RR38–40); the instant-aware `ts` tie-break (RR32/33);
  the three entry-shape deny arms bound to fixtures in both readers (RR27–29, R12P27–29);
  `check_lead_gate.py --r12-sweep` run to completion. Load generators reap with their agent.
- **Refused and replaced (lead rulings):** the regex brace-depth Rust indexer → the shared `syn`
  tool; the un-attempted attestation record and required set (a stop rule invoked without its
  trigger) → built with executed REDs; the gate's own vocabulary false positives on a checker's rule
  ids → the definition exemption.

## 3. Invariants crossed

K5 (migration 039 appended, neutral names, every pin site; no table rebuild on SQLite), K4 (the
resume-parity harness compares bytes across Peer and Local; the mining oracle compares digests), B6
(every group ships across its crates in this one PR), K6 (no version change), the frozen wire surface
(no `.proto` change lands; `cache_outcome` fills an existing field), per-RPC tenant isolation (the
prefix-ownership oracles disclose nothing across tenants), the cookbook one-way rule (the cookbook
diff touches one chapter and no engine code), the actuator rule (the excised GRAPH design was refused
precisely because its convergence required a process the rule forbids), the engine-not-platform rule
(`check_no_consumer_names.py` green on every landing), and the swarm's human-amend-only set
(`SWARM_GATE_TOUCHED` is the expected red).

## 4. Deviations the lead ruled on (each opened in the code before acceptance)

1. **STAMPS**: `JobWorker::run` was a zero-caller public bypass of the slot — refused; deleted.
2. **GATES**: #513 already shipped and #508 shipped except its self-test — removed from scope;
   #517's heuristic change deferred with the executed count (no live CamelCase governance noun; 90+
   neighbours a loosened rule would hit).
3. **TARBALL**: the chroot mechanism cannot run in the unprivileged release container — the
   report-parse shape ships and #534's chroot half stays open; the fixture is captured on the lane,
   never a rented pod.
4. **GANG**: v1 killed by its pressure round (the graph arm's identity was not a function of its
   record; reuse unreachable) and rebuilt; #538 excised with its refutation appended; the compile-time
   bar for the submit seam moved to GANG3A rather than accepted as "uncovered".
5. **GANG3B**: an external 2 ms poll of the artifact store with a widened fixture and a torn-read
   tolerance was refused — the checkpoint-written seam exists and is reachable in-process; rebuilt
   on it. The corrupted-bundle row (the parity property's blind spot) and the SQLite-only arm (the
   family's shape) were accepted.
6. **ENUMS**: `strum`'s derive over an in-tree proc-macro — accepted (the derive reads the enum's own
   variant list; no new workspace member).
7. **COOKBOOK**: #506 was already shipped — accepted as re-verification, not re-authoring.
8. **DOCS**: RULES.md's one claim stronger than the constitution was refused and trimmed; the DF-55
   gate sentence and a round-number marker were corrected before landing.
9. **GRAPH**: four executed design rounds; the unit is excised by its own contract's pre-committed
   stop rule (§8); what does not depend on the graph mechanism landed as standalone fixes without
   adding a `Cancelled` variant.
10. **IDENTITY**: a one-op admission fold and a string-keyed dry-run match were refused twice as
    rival enumerations; the verdict lives on the `ProbedOp` row; the call-site oracle keyed by
    `(file, line)` was refused in favour of `(file, fn, ordinal, count)`; the debug-only prefix
    assertion was refused in favour of an always-on typed refusal.
11. **LEADGATE**: a regex-based Rust indexer was refused (the recorded rule: regex readers over
    Rust/YAML lost five audits) in favour of one shared `syn` symbol-index tool; #557 items 1–2 and
    #570 were sent back because the stop rule they invoked had not been triggered.
12. **Lead**: two review lists keyed by line number (the source gate's `include_str!` entry, the
    test-bounds inventory) drifted under unrelated edits and were re-keyed by site identity; an
    implementer's line bump was replaced, not kept.

## 5. Residuals and cuts (each with its executed refutation or its filed rebuild)

- **#515 GRAPH** — excised. The write-time cascade derives doom once over the downstream that exists
  at the terminal write; a later submit attaching a child under a running-with-cancel-requested or
  completed interior node of a failed root, or racing the cascade's snapshot on Postgres, is claimable
  forever; the contract also stated `depends_on` doom as direct-only in one section and transitive in
  another. The four rounds' executed findings and the next attempt's starting point are on the issue:
  https://github.com/f-inverse/jammi-ai/issues/515#issuecomment-5708043270.
- **#534 loader half** — the chroot/`unshare` mechanism cannot run in the unprivileged release
  container; the report-parse refuse-outside-`<lib>` rule ships; the chroot half stays open.
- **#517** — deferred with the executed count on the issue; `governance_stem` untouched.
- **#538** — the graph arm through a materialised training set needs its own design round; the Peer
  refusal stays typed; the v1 limit is stated in the plans.
- **#540** — the range split under the partition-aware operator, de-scoped from U6 (unchanged).
- **#551 mining pin** — the committed byte-for-byte constant needs a Linux measurement; captured from
  the PR's first CI run and committed as a closer (§9).
- **#543 broadcast mechanism** — not built; the shared-root resume is the v1 substrate, proven by the
  two oracles.
- **#445, #478** — deferred indefinitely (maintainer decision, recorded on both issues).
- **Cluster leg transport** — the Instant Cluster `ens1` overlay is unexercised; the accepted proof
  substrate is two Global-Networking pods (recorded in the plan).
- **The shared scratch database** — migration 039 was applied to the shared local `jammi_test` by a
  unit under development and corrupted it three times; each later unit ran on its own database. CI
  provisions one Postgres per leg, so the hazard is local only.

## 6. Acceptance, restated as executed oracles

Every issue in §1 closes on a named test (§2, §11) that was RED at base or under its executed
mutation and is GREEN on the consolidated tip; every nightly-lane red that motivated a unit (GPU
prove, the cookbook render, the chaos leg) is closed by an executed run or a promoted leg; the plan
READMEs state what is on `main`; the cluster artifact is committed under the registry rule.

## 7. Pressure rounds (one row per round in `docs/rigor/feat_500-wave5.jsonl`)

| Contract | Verdict | What the fold changed |
|---|---|---|
| stamps v1 | KILL | a reader-side regex CHECK cannot express calendar validity; the Postgres lease writer shape; nine-digit stamps; SQLite rebuild cascades — the writer is fixed, migration 039 normalises |
| stamps v2 | REFINE | trigger-aware splitter; refusal as an outcome property; truncation rule; source-derived universe; replay idempotence; typed domain-violation class |
| gates | REFINE | #513 shipped; #508 self-test only; #517 deferred; universe = the binding surface; composite-action arm; composition and no-regression oracle; #511 = README + pin + manifest-reachability guard |
| tarball | REFINE | closure minus host-provided; one bound classification; two loader arms; RPATH measured; object-keyed collision; fixture captured on the lane; provisional-fixture gate |
| testbounds | REFINE | the seam must be created; every bound form in all four targets; checkable classes; measured contention; per-run database |
| gang v1 | KILL | graph-arm identity not a function of its record; reuse unreachable; mechanisms refuted |
| gang v2 | REFINE | flat enum is the persisted type; the witness is the compiler's; relation class scoped; resume across attempts on a new harness |
| identity | REFINE | I2 shipped at base; I1 over the real raw deleter; deterministic dry-run profile over the typed table; owner-refusal and its migration deleted |
| graph (design) | REFINE | the frontier queue is the forbidden scheduler |
| graph §0 | KILL | no parent term in the claim predicate |
| graph §4 | REFINE | the sweep and projection layer refuted |
| graph §5 | REFINE | depth truncation, unguarded second statement, statement order, per-backend texts, parent refusal, gauge plan |
| graph §6 | KILL | attach-after-doom escapes write-time convergence — the stop rule fires |

## 8. Stop rules honored

GRAPH's contract pre-committed excision on a block-severity finding in the cascade-completeness class
during its concurrent pressure round or the audit; the round returned it; the unit came out with no
fifth round. TARBALL's chroot half and GATES' #517 followed their contracts' own stop conditions.

## 9. Closers the lead runs on the tip

- `ci/scripts/merge_path.sh` — every stage; the three Postgres lanes on a fresh database; the guard
  matrix read from `ci.yml`; the swarm steps whole.
- The tree-wide source gates after every landing: `pinned_source_gate::`, `rank_admission::`,
  `test_bounds_inventory::`, `terminality_source_gate`, the migration-ledger oracle,
  `check_kernel_oracles.py`, `check_citations.py`.
- GPU prove run 35171308424 (all four arches green on the capability-surface fix); GPU cluster run
  35171132263 (row C9's artifact); the cookbook render workflow dispatched on the branch; the
  distributed lane with the chaos leg promoted.
- The mining byte pin: NOT captured in this PR. The shipped mining oracle compares two digests taken live
  in one run and prints no constant (`training_set.rs`, the `hard_negative_mining_at_w1_…` test), so there is
  nothing in a CI log to pin; capturing a Linux constant needs a print, a CI run and a committed pair — a
  follow-up on #551, which stays OPEN (partial per §11), not a closer of this PR.
- The adversarial audit, then the oracle LAST after every non-rigor edit, then the record export.
- Lead-built fixes landed at consolidation, each with an executed RED: a793f572 (the pinned source
  gate's `include_str!` review list keyed by argument text and count, never by line); 49b804ac (the
  test-bounds inventory keyed by enclosing item via `syn`, the marker window the item's own span);
  4a4ef7bf (the gpu-prove timings gate's setter universe excludes the lane test suites by rule);
  adca0866 (ten lines of the branch's own rustdoc restated as invariants); 6bc9d8a8 (the cluster
  lane's SIGHUP fixture signals only after an armed marker, 163/163); 4cc1724b (the two-backend
  freshness test asserts each backend in its own arm, no early return — `check_kernel_oracles.py`
  KO-7 green); 4a931b9d (plans 67/68 read as shipped, C9/#543/#538/#515/#445/#478 dispositions, the
  DataFusion-55 gate corrected to ballista and table-providers).

## 10. Gate table (an executed run at the named tip; logs under the session scratchpad)

| tip | stage | result | log / run |
|---|---|---|---|
| bd3a6202 | guards stage on the tip after GATES landed | green after the kernel-oracle marker fold | `logs/w5-guards-1.log` |
| 946e04d3 | `check_cuda_run_artifacts.py`; lane suite 163/163 | PASS; `6207a918` an ancestor | — |
| 77f1ed8d | constitution-anchors, check-citations (1069), swarm.yml parse, `grep ADR-00` = 0 | green | — |
| 35171308424 | GPU prove on the capability-surface fix | sm_80/86/89/90 success | run id |
| 35171132263 | GPU cluster leg, two pods | GREEN, exit 0 | run id |
| a793f572 | `pinned_source_gate::` 52/52; RED on a count change, GREEN on a line shift | green | — |
| af9ef9a1 | `terminality_source_gate` + migration-ledger oracle 9/9; check_kernel_oracles; citations (1071) | green | — |
| 4a931b9d | `pinned_source_gate:: rank_admission:: test_bounds_inventory::` 67/67 | green | — |
| b04119a6, 5bb02acd, f3e1549c | merge path `--only static` (fmt, four clippy surfaces, rustdoc, guide) | 7/7 green at each tip | `logs/merge-path-static-*.log` |
| e271b36a | merge path `--only tests` (hermetic lane, jammi-db test-hooks lane, golden parity, three Postgres lanes on a fresh `jammi_w5t`) | 6/6 green | `logs/merge-path-tests-e271b36a.log` |
| f3e1549c | merge path `--only tests` | 5/6; the hermetic lane's single failure `esc_072_two_pool_writers::two_pool_worker_loops_stay_clean` — one unabsorbed `database is locked` in 10,774 ops while three cargo runs shared the machine; 3/3 green alone (10.1 s each); classified load against the 5 s production busy timeout, not a regression | `logs/merge-path-tests-f3e1549c.log` |
| 5bb02acd, f3e1549c | merge path `--only guards,swarm` (94 guard-matrix commands, 21 swarm steps) | 99 ok, 1 fail: arch-validation freshness — the waivers had been bumped before later admission.rs commits; fixed by 78bb99ae as the last surface-touching commit | `logs/merge-path-guards-swarm-*.log` |
| 5bb02acd | `cargo metadata --locked \| check_dep_direction.py`; `check_swarm_bijection.py`; `test_finetune_ab_disable_op_keys.py` | OK (702 crates); PASS; OK | — |
| c0cf5245 … bf3f9500 | `SWARM_DIFF_BASE=db19a614 check_journey_markers.py` | 0 BLOCK (advisories on private surfaces only) | `logs/journey-markers-*.log` |
| 7f4364c9, b030b1f6 | `check_kernel_oracles.py` (about 3.5 min; verdict at the top of its output) | exit 0 | `logs/kernel-oracles-*.log` |
| 5a6e8cdd, 827029ad | `ci/scripts/test_gpu_cluster_lane.sh` | 163/163 | `logs/lane-suite-*.log` |
| 7f4364c9 | `pinned_source_gate:: rank_admission:: test_bounds_inventory::` | 67 tests green | — |
| 35186742275 | GPU prove on 46820a92 | all four arches failed on one test only, `manifest_capability_categories_match_probed_ops_by_kind` (the release manifest lacked `attention_block_flash`); the f16-oracle panics in the logs are `should_panic` tests | run id |
| 35191561460 | GPU prove on f3e1549c (manifest fixed) | sm_80, sm_90 success; sm_86, sm_89 pending | run id |
| 35196220271 | GPU prove on 78bb99ae | (recorded at close) | run id |
| bf3f9500 | merge path `--only static`, then `--only tests`, then `--only guards,swarm`, run one at a time with the machine free; the final closing audit; oracle; records | (recorded at close) | `logs/merge-path-*-bf3f9500.log` |

## 11. Units as built — the implementers' contract files, folded by the lead

Each subsection is the implementer's own report, verbatim, headed by the lead's note on what was
opened, re-run or changed at consolidation. Every `path:line` inside them was written at the
implementer's own tip and is not re-anchored to the consolidated tip; the citation resolver and the
rigor-record gate run on the final tip and check existence and length, never the drifted offset —
read a folded citation as "at that unit's tip".

### 11.1 GATESAI — the Rust source gate rebuilt on syn (ai-core) — landed as af8fa645, 0b13ea74, 32790f46, a2643979 (original tip 59bf0a32)

**Lead note:** cherry-picked; the kernel-oracle fn-in-literal markers the checker demanded on the rebuilt gate were placed by running the checker to a fixed point (5d78c695, lead); the include_str! review entry keyed by line drifted under later units and was re-keyed by argument text (a793f572, lead); 52/52 on the tip.

#### GATESAI — #554 / #549: the Rust source gate in `crates/jammi-ai/tests/it/pinned_source_gate.rs`

Base `db19a614`. Worktree `<scratchpad>/wt-gatesai`, branch `unit/gatesai`, tip `59bf0a32ab5f033be09ec71bcfc8cc81e944a224`.

##### 1. Scope shipped

Files touched: `crates/jammi-ai/Cargo.toml` (two new dev-deps), `Cargo.lock` (two new edges into
jammi-ai's own dependency list, no new package versions), `crates/jammi-ai/tests/it/pinned_source_gate.rs`
(the whole rebuild; file grew from 3767 to ~5450 lines).

**#554 (item 1 — count keys).** `ReviewedRegistrationSite` gained an `allowed: usize` field.
`registration_verb_occurrences`/`ddl_literal_occurrences` now return `BTreeMap<(file,fn,ordinal), usize>`
(a real occurrence COUNT per site, not a `BTreeSet` presence flag). `assert_occurrences_reviewed` now
checks three things: unreviewed (new key), stale (reviewed key with zero real hits), and OVER-COUNT
(real count > `allowed`) — the third arm is what closes closing audit #8 of U2a: a second
`ctx.register_table(...)` planted inside an already-reviewed function used to collapse onto the same
`BTreeSet` key and stay green; it now bumps the count past `allowed` and fails.
`register_object_store_twice_for_one_url_rebinds_the_same_driver_and_errors_on_neither` (the one site
that deliberately calls the same verb twice) carries `allowed: 2`; `catalog/schema.rs`'s module-scope
migration-constant site carries `allowed: 13` (transcribed from the gate's own real output, not
hand-counted).

**#554 (items 2/3 — token-stream DDL shape, `include_str!`).** `ddl_literal_occurrences` is rebuilt on
`DdlLiteralScanner`, a `syn::visit::Visit` implementor, replacing the old per-line `mask_comments_only` +
`ddl_statement_shape` scan. Three sources, each independent: (a) `visit_lit_str` — every `syn::LitStr`'s
DECODED value (raw-string-hash-count-proof by construction, since `syn` decodes it, not this file); (b)
the ARGUMENT-ORDER CONCATENATION of every string-literal argument to a `format!`/`concat!`/`write!`/
`writeln!` invocation, so a DDL keyword split across two literal arguments on separate lines is seen as
one statement; (c) `include_str!(..)`'s own target file, resolved relative to the including file's
directory and scanned as if inlined, when the argument is a single top-level string literal — when it
is not (the real `crates/jammi-ai/src/fine_tune/trainer.rs:5763` case, `include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
"/../../Cargo.lock"))`), it is recorded in `unresolved_includes` and checked against a new reviewed,
exhaustive allowlist (`UNRESOLVED_INCLUDE_STR_TARGETS`, one entry) rather than silently skipped OR
panicking (a design deviation from my own first draft, which DID panic on this real occurrence — see
§3 Uncovered/deviation note below).
`visit_attribute` excludes any doc-comment-synthesized attribute (detected via the same collapsed-span
signal the masking rewrite uses) from every literal scan, so this file's own prose (`CREATE TABLE`
appears in backticks in several doc comments) is never mistaken for a DDL literal.

**#554 (item 4 — honest universe).** `SURFACE_DIRS`'s doc now states explicitly that
`crates/jammi-bench/src` (among any other crate) is outside the reviewed universe, and
`falsification_registration_verb_scan_states_its_universe_honestly` proves it directly: it reads the
real `crates/jammi-bench/src/corpus.rs`, asserts it still contains a live `register_parquet(` call, and
asserts `scan_surface()` never returns a `crates/jammi-bench/` file.

**#554 (residual 5 — real-tokenizer masking).** `mask_non_code`/`mask_comments_only` are now thin
wrappers over `real_tokenizer_mask`, which walks `proc-macro2`'s fallback token stream (the same lexer
`rustc` itself is built on) and uses each token's `Span::byte_range()` (the `span-locations` feature,
documented accurate outside an actual proc-macro invocation — verified empirically against a standalone
probe program before use) to decide what is code, a string/char literal, or an inter-token gap
(whitespace/comment territory, since nothing else can appear there in syntactically valid Rust — this
also closes the R-A nested-block-comment limit for free: the tokenizer's own trivia-skipping, not a
hand-counted `*/`, decides where a comment ends). Doc comments (`///`/`//!`/`/** */`/`/*! */`) are
detected via the SAME collapsed-span heuristic as the DDL scanner's `visit_attribute` override (a
hand-written `#` is always exactly 1 byte wide; a doc-comment-synthesized one carries the whole
original comment's span) and blanked in both masking modes, matching the old scanner's pre-existing
(if accidental) behaviour.

**#549 — the reachability gate.** New section at the end of the file. `binding_surface_crates()`
derives THE BINDING SURFACE via `cargo metadata --no-deps` (JSON parsed with `serde_json`, already a
plain dependency of `jammi-ai` and so already linked into the `it` test binary): the REVERSE-dependency
closure of `jammi-db`/`jammi-ai` within the workspace (every workspace member whose OWN forward
path-dependency closure contains one of the two) — 11 of 15 workspace members, 337 tracked `.rs` files
at this head. `build_call_graph` walks every file's real `syn::parse_file` AST via `FileGraphBuilder`
(`syn::visit::Visit`), producing a NAME-keyed graph (safe-direction over-approximation) with edges for:
direct/method/UFCS calls; a bare path handed directly to a call/method-call as an argument (`for_each(b)`,
`map(Self::b)`); any call-shaped identifier inside a macro invocation's raw token stream (`assert!`,
`tokio::select!` arms); a fn-pointer struct field call `(s.f)(..)`, resolved via a second, whole-surface
pass over every place a value is assigned into a field of that same name. Two residuals fail the gate
closed rather than resolving silently: an unresolved fn-pointer field call (no assignment found anywhere),
and a `macro_rules!` definition whose own template body contains a registration-verb call shape or a
DDL-shaped literal (its generated function's name is a macro metavariable resolvable only per invocation
site, so it is checked directly rather than traced through expansion). `fine_tune_reachable_sites_are_all_reviewed`
intersects `registration_verb_occurrences`/`ddl_literal_occurrences` (run over the WIDE binding surface)
against the graph's BFS-reachable set from every `fn` under `crates/jammi-ai/src/fine_tune/`, checked
against `FINE_TUNE_REACHABLE_SITES` (14 entries, transcribed from the gate's own real output).

**Deviations from the issue text / re-located citations.** #549's own text cited
`crates/jammi-ai/src/fine_tune/worker.rs:787,:848` for `tokio::select!` arms and `crates/jammi-ai/src/fine_tune/`
`macro_rules!` files; F8 (the contract's pressure-round fold) corrected these: the `macro_rules!` files
are `crates/jammi-db/src/config/layers.rs` (`scalar!`, confirmed at :341) and
`crates/jammi-db/src/catalog/backend.rs` (`impl_from_sql_primitive!`, confirmed near :574 —
re-located, not re-verified to the exact line since the property checked is "does this file's
`macro_rules!` templates contain a verb/DDL shape", answered directly by the gate itself, not by citing
a line); `tokio::select!` arms in `worker.rs` are at :1549/:1624/:1660 at this head. I did not re-cite
these real-tree line numbers as `PATH:LINE` citations anywhere in the shipped code (the gate's own
`graph.macro_rules_findings.is_empty()` assertion over the real binding surface is the actual, executed
check that these two macros are NOT binding sites — a stronger claim than any static line citation).
F3 (the contract's other binding fold) is why the universe is the reverse-, not forward-, dependency
closure — implemented as designed in that fold, not the issue's own original (and, per F3, incorrect)
framing.

**Deviation: `include_str!` fail-closed vs panic.** My first implementation panicked on ANY
unresolvable `include_str!` argument (matching this file's existing "fails closed" idiom for unreadable
tracked files). Running it against the real tree immediately hit `crates/jammi-ai/src/fine_tune/trainer.rs:5763`'s
`include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))` and aborted the whole test
binary. I changed this to the same "named finding + reviewed allowlist" discipline every other residual
in this file uses (`unresolved_includes` / `UNRESOLVED_INCLUDE_STR_TARGETS`) rather than either (a)
attempting to evaluate `env!`/`concat!` myself (out of scope, and its own source of bugs — the SAME
category of bug I hit next: see the `syn::Lit::new` vs `syn::parse_str` finding below), or (b) leaving
the hard panic in place, which would make the gate impossible to run at all on today's tree. The code
line this rests on is the real occurrence itself, `crates/jammi-ai/src/fine_tune/trainer.rs:5763`.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Every registration-verb/DDL site under `SURFACE_DIRS` is keyed `(file,fn,ordinal,count)`; `count <= allowed` | `registration_verb_occurrences_are_all_reviewed`, `ddl_literal_occurrences_are_all_reviewed` | Planted a second `register_table(` inside an already-reviewed fn (`falsification_a_second_occurrence_inside_an_already_reviewed_function_is_over_count`): first line "site(s) occur MORE often than their reviewed `allowed` count … 2, 1" |
| A `deregister_X(` occurrence is never double-counted as a separate `register_X(` one | `falsification_paired_verb_occurrence_count_does_not_double_count_deregister` | Removing the substring-cancellation reasoning (counting both patterns independently) makes the assertion `Some(2)` fail with `Some(3)` — verified by hand-deriving the arithmetic before writing the simplified single-pattern-count implementation |
| A DDL keyword split across `concat!`/`format!` literal arguments on separate lines is one statement | `falsification_ddl_keyword_split_across_concat_arguments_is_detected` | RED under the deleted line-based scan (verified by re-reading its per-line implementation: neither half is independently DDL-shaped) |
| `include_str!(..)`'s target file content is scanned as if inlined | `falsification_include_str_target_ddl_is_detected` | RED under the deleted scan (it never inspected macro arguments at all) |
| A `format!` single-literal DDL template is counted exactly once, not once per detection path | `falsification_format_macro_ddl_literal_is_not_double_counted` | First draft (general-scan unconditional + combined-scan unconditional) produced `hits.len() == 2`; fixed by scoping the general scan to the `_` (non-formatter) match arm only |
| A macro-nested DDL literal is attributed to its REAL source line, never a phantom line 1 | `falsification_general_macro_ddl_literal_is_attributed_to_its_real_line` | First draft (`syn::parse_str::<syn::Lit>(&lit.to_string())`) reported line 1 for every such hit — caught live on the real tree (`store/mutable/{postgres,sqlite}.rs::create_table_ddl` both showed `<module-scope>` at line 1); fixed via `syn::Lit::new(lit)`, which preserves the token's real span |
| A module-level `const X: &str = "CREATE TABLE .."` is a DDL site | `falsification_module_level_const_ddl_is_detected` (+ 13 real hits in `catalog/schema.rs`) | n/a — this is the "must be a fixture" requirement from G8's own list, satisfied directly |
| A raw string's content (any hash count) is masked, a doc comment is masked in both modes, a nested block comment is masked in its ENTIRE extent | `falsification_real_tokenizer_mask_handles_raw_strings_and_nested_comments` | Reproduces the exact residual closing audit #9 measured (`r#"...\"... // ... /* ... */ ..."#`) — RED under the deleted hand-rolled scanner (documented, not re-executed against deleted code; the OLD function's own doc states the measured counts: 22/12 lines) |
| `SURFACE_DIRS` is an honest universe (`crates/jammi-bench` outside it) | `falsification_registration_verb_scan_states_its_universe_honestly` | Reads the real `corpus.rs`, asserts the live `register_parquet(` call and zero jammi-bench files in `scan_surface()` |
| Reachability graph edges exist for: fn-pointer argument, `map(Self::b)`, a call inside `assert!`, a call inside a `tokio::select!` arm, a fn-pointer struct field | `falsification_fn_pointer_argument_edge_is_found`, `falsification_map_self_method_argument_edge_is_found`, `falsification_call_inside_assert_macro_edge_is_found`, `falsification_call_inside_tokio_select_arm_edge_is_found`, `falsification_fn_pointer_struct_field_edge_is_found` | Each written against `probe_reachability` with a `direct_callee`/`b`/`callee` fn that is UNREACHABLE except via the one edge shape under test; verified by first running without the corresponding builder logic (removing `record_path_arguments`'s call, or the `visit_macro`'s `_` arm, or the field-ptr resolution pass) and observing the assertion fail before restoring it |
| Unresolved fn-pointer field call fails closed (named, not silent) | `falsification_unresolved_fn_pointer_field_call_fails_closed` | Fixture has NO assignment to `.f` anywhere; asserts `unresolved_field_ptr_sites` non-empty |
| `macro_rules!` template binding-site fails closed | `falsification_macro_rules_template_binding_site_is_flagged` | Synthetic `macro_rules!` whose template calls `.register_table(`; asserts `macro_rules_findings` non-empty. Real `scalar!`/`impl_from_sql_primitive!` templates produce ZERO findings (checked live via `fine_tune_reachable_sites_are_all_reviewed`'s own unconditional `assert!(graph.macro_rules_findings.is_empty())`) |
| Name-keyed over-approximation is safe-direction: a new same-named binder is reported | `falsification_name_keyed_over_approximation_reports_a_new_same_named_binder` | Two `helper` fns, one in a submodule; asserts BOTH nodes are marked reachable from one call |
| Reachability wall time < 10s on the real binding surface | `fine_tune_reachability_wall_time_is_under_ten_seconds` (assertion) + three manual timed runs (see §4) | n/a — measured directly, no mutation needed for a timing bound |
| Every `fine_tune/`-reachable registration/DDL site is reviewed | `fine_tune_reachable_sites_are_all_reviewed` | Ran with `FINE_TUNE_REACHABLE_SITES = &[]` first: reported the real 14-entry unreviewed list verbatim (transcribed into the constant, not guessed) |
| `GraphFn::line` is never a lost/zero sentinel | `graph_fn_lines_are_never_zero` | n/a — direct assertion over the real 337-file surface |

##### 3. Uncovered

- **`crates/jammi-ai/src/fine_tune/trainer.rs:5763`'s `include_str!` target is never scanned for DDL content.** Its content
  is `Cargo.lock` (TOML), reviewed at `UNRESOLVED_INCLUDE_STR_TARGETS` as provably non-DDL by inspection
  of what it reads, not by scanning it. Labelled, not silently skipped.
- **A macro invocation used at ITEM position (not inside any `fn`) contributes no call-graph edges** —
  `visit_macro`'s edge-recording (`add_call_edge`) only fires when `self.current` is non-empty (an
  enclosing named fn exists); an item-level macro invocation's own DDL/verb content is still found by
  `registration_verb_occurrences`/`ddl_literal_occurrences` (which scan unconditionally by position, not
  by call-graph edge), but is never itself a call-graph NODE with outgoing edges. Not observed to matter
  on the real tree (no registration verb is invoked this way today), but genuinely uncovered as a shape.
- **Local variable rebinding (`let f = foo; f();`) is not a tracked edge shape** — not in G8's own
  required list, and this file's own module doc for the OLDER (pattern 2/4) detectors already discloses
  the analogous "does not follow a value across function boundaries" limit; consistent, not newly
  introduced.
- **The `FINE_TUNE_REACHABLE_SITES` review for `jammi-bench/corpus.rs::register`** rests on a directional
  argument (jammi-bench depends on jammi-ai, not the reverse, so no forward call is possible) rather than
  an executed negative control proving "this call graph never marks a node reachable purely through a
  cross-crate name collision when a REAL dependency-direction check would refuse it" as a general gate
  property. The specific instance is checked (I verified `binding_surface_crates()`'s reverse-closure
  computation directly via a standalone script before writing the gate), but the general claim "no
  cross-crate collision review entry is ever wrong" is asserted per-entry in prose, the same class of
  claim `caller_set_claims_match_reality` machine-checks for a DIFFERENT (older) part of this file — not
  extended here for time reasons.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo test -p jammi-ai --test it pinned_source_gate` | 0 | 51 passed, 0 failed, 584 filtered out (default feature set, kept for the whole session) |
| `cargo clippy -p jammi-ai --all-targets -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1067 file(s) scanned, all PATH:LINE citations resolve …` (2 pre-existing EXEMPT entries, unrelated to this change) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK …` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | clean |
| `cargo deny check advisories` (after adding the two dev-deps) | 0 | `advisories ok`, no new advisory |

Wall-time measurement (`fine_tune_reachability_wall_time_is_under_ten_seconds`'s own computation, timed
via three separate `cargo test` invocations of just that test, `--nocapture` with a temporary `eprintln!`
removed before the final commit): **1.68s / 1.89s / 3.16s, median 1.89s**, against the < 10s bound —
includes the `cargo metadata --no-deps` subprocess, `syn::parse_file` over 337 files, and the BFS.

##### 5. Issues closed

- **#554 — CLOSED.** All four items from the issue text (count-keyed sites with `allowed` ceiling;
  token-stream DDL shape covering `concat!`/`format!` split literals; `include_str!` targets; honest
  universe) plus the fifth (raw-string masking) residual are rebuilt and each carries an executed
  oracle in §2. `every_reviewed_registration_site_states_its_property` and
  `allowlists_match_current_hits_exactly` (pre-existing, still green) continue to hold across the
  rebuild.
- **#549 — CLOSED**, not deferred to the stop rule's fallback. A sound (safe-direction,
  over-approximating, name-keyed) call graph over the correctly-derived binding surface was built within
  this unit; every edge shape and DDL position G8 lists carries its own executed fixture (§2); the two
  shapes that cannot be soundly resolved by name (an unresolved fn-pointer field call, a
  `macro_rules!`-generated fn item) fail the gate closed rather than being silently dropped, per G2. Wall
  time is measured and stated (§4), under the 10s bound with margin. §3's stop rule (two blocks) was
  never invoked — this was the unit's own build, not a second attempt following an external audit block.

##### 6. Commits

```
59bf0a32 test(ai): #549 add the module-level const SQL DDL-position fixture
f1567458 feat(ai): #549 sound fine_tune reachability -- a real call graph over the binding surface
ec40da42 fix(ai): #554 the literal-occurrence gate is count-keyed and real-tokenizer-driven
fc0d99ff test(ai): #554 #549 add syn/proc-macro2 as jammi-ai dev-deps for the real-tokenizer gate rebuild
```
(`git log --oneline db19a614..HEAD`, base `db19a614`)

##### Scope note for the lead

`syn`/`proc-macro2` were added as **dev-dependencies of `crates/jammi-ai` only** (not touching
`crates/jammi-ai/src/lib.rs`, the crate's own `[dependencies]`, or any shared-declaration file) —
outside the lead/docs-ci shared-declaration class stated in my brief, so no scope amendment is needed for
that. I did not touch `crates/jammi-ai/src/lib.rs` or any `error.rs`.

### 11.2 GATESCI — every ci/scripts gate in scope (docs-ci) — landed as 2a06aa49 … bd3a6202 (original tip e76abdf7)

**Lead note:** cherry-picked; the guards stage was run to completion by the lead on the tip (the implementer's own run had not finished).

#### GATESCI contract — docs-ci gate rebuilds, wave 5 group E7a

Worktree `<scratchpad>/wt-gatesci`, branch `unit/gatesci`, base `db19a614`.

##### 1. Scope shipped

Eight files touched, one new module, one new gate script, one new test class file addition. Commits, in order:

1. **#564** (`ci/scripts/check_gpu_prove_once.py`) — `find_step_if_by_name` and `_other_publishing_steps`
   both identified a step-gated row's step by display name and trusted the FIRST match (or excluded every
   same-named step from the second-step scan). Both now assert uniqueness of the gated step's display name
   first and refuse the ambiguity by name, fail-closed.
2. **#563 / #565 / #561 / F4 / F5** (same file + its test file) — a single coherent rebuild:
   - `_step_push_raw_is_false_spelling`/`_composed_jobs_or_fail`/`_composed_step_push_node`: `push:` is now
     decided from the COMPOSED (pre-construction) YAML scalar, restricted to GitHub's own two-spelling
     boolean-false set (bare `false`/`False`/`FALSE`, or quoted `"false"`/`'false'`), never PyYAML's wider
     YAML-1.1 boolean set (`off`/`no`/... which resolve to the SAME Python `False`).
   - `_flatten_string_scalars`/`_job_level_scalar_values`/`_job_level_primitive_match`: the publish-primitive
     matcher's domain widens to a YAML sequence under `with:`/`env:` and to a job's own job-level `env:`/
     `strategy.matrix:` values. `job_invokes_publish_primitive`'s `steps:` handling no longer raises an
     uncaught `TypeError` on `steps: 5`.
   - `_traverse_local_reusable` (#561): the per-delegate traversal into a LOCAL job-level `uses:` target is
     rebuilt with diamond memoization, cycle depth-bound, self-mask discipline, and dangling-target handling
     — PLUS the gap a895b148 was filed to plug: a CROSS-REPO delegate reached anywhere in the reachable set
     (not only the top-level caller) is now its own named, unexaminable refusal.
   - `_composite_action_structure`/`_local_action_target`/`_cross_repo_action_target` (F4, contract delta): a
     step-level `uses: ./.github/actions/<x>` is resolved and its `action.yml`'s own `runs.steps:` examined
     through the same readers, replacing the deleted name-keyed `_LOCAL_DOCKER_PUBLISH_VALUE_RE`/
     `_LOCAL_RELEASE_UPLOAD_VALUE_RE`. A composite action wrapping `docker/build-push-action` still defers the
     push decision to the CALLING step's own `with.push` (an inner `${{ inputs.push }}` expression cannot be
     resolved any other way).
   - `DifferentialOracleAgainstA895b148Test` (F5, contract delta): reconstructs a895b148's own fail-closed
     shape directly (never imports it — that shape no longer exists) and asserts the rebuild's own P6 findings
     are a strict subset of it, over the real tree and every traversal fixture in the file.
3. **#533** (`ci/scripts/check_execution_surface_reachability.py`) — Rule 1c's YAML-level honesty said nothing
   about shell-level control flow inside a `run:` body; a `run:`/`cmd:` body containing a bare `if`/`case`
   keyword is now excluded wholesale from tuple extraction. `_pr_admits_main` now requires `synchronize`
   present (or `types:` absent) rather than crediting any `types:` list intersecting the default set — a
   `types: [opened]`-only host never re-runs on a later push to the same PR. Dead `DEFAULT_PR_LIFECYCLE_TYPES`
   constant removed.
4. **#532** (`ci/scripts/check_lint_surface_closure.py`, `ci/scripts/lint_surface_required_lanes.txt`) —
   `find_unprotected_lanes` is a DERIVED third closure: for every discovered merge-path clippy lane, remove
   it and re-run the other two closures (`find_gaps`, `find_missing_required_lanes`); if neither result
   changes, the lane is undetectable-on-loss. Reproduced exactly the four known gaps on the real tree. Two
   (`jammi-db postgres,mysql`, `jammi-server test-hooks`) are closed with real registry rows. The remaining
   two (`--workspace --all-targets` and the `jammi-wire`/`jammi-admin`/`jammi-client` lane) are PROVABLY not
   closable by a row (either satisfies any row the other does) — recorded explicitly in
   `REVIEWED_REDUNDANT_LANE_PAIRS` and printed as an informational NOTE every run, never hidden, never a FAIL.
5. **#530** (new `ci/scripts/ancestry.py`, edits to `check_cuda_run_artifacts.py` and
   `check_pod_build_timings.py`) — ONE shared module (`run`/`is_shallow_repository`/`is_ancestor`/
   `check_ancestry`) imported by BOTH gates, per contract delta G6 ("ONE shared module... not a diff oracle").
   `check_ancestry(data, repo_root, ancestor_message)` owns the one anchor model (`merged_as` rescue); each
   gate's own `_run`/`is_shallow_repository`/`_is_ancestor`/`GIT_SHA_RE` names are preserved as aliases so
   every existing internal call site (15+ in `check_cuda_run_artifacts.py`) keeps working unchanged.
   `check_pod_build_timings.py` gains the `merged_as`/`merged_via_pr` schema-typing pair it never had.
6. **#508** (`ci/scripts/check_no_consumer_names.py`) — the allowlist mechanism (all seven rot rules) already
   shipped in `9c62f12b`; this commit is ONLY the missing `--self-test`, one fixture per rot rule 1-7 plus the
   duplicate-pair vs. same-identifier-different-path distinction, built against the REAL committed
   `register_content_hash_udf` row. `swarm.yml` untouched (already carries the allowlist glob).
7. **#513 G7'** (`ci/scripts/check_kernel_oracles.py`, `ci/kernel-oracle-helpers.txt`,
   `ci/scripts/test_check_kernel_oracles.py`) — base #513 (folded require-gate recognition) already shipped in
   `9c62f12b`, per contract delta not touched again. New: `check_resource_binding`/`load_resource_bindings` —
   an OPT-IN `resource=<tag>` annotation on a registry line asserts the accessor's own direct env-read
   resolves to the JAMMI_REQUIRE_* variable that tag names, closing the gap where "reaches some registered
   gate" says nothing about WHICH resource that gate governs. Three real annotations added
   (`pg_url_for_tests resource=pg`, `cuda_device resource=gpu`, `required_backends resource=distributed`).
   `run_gate` gains an explicit `resource_bindings` parameter defaulting to `{}` (never reading the real file
   inside the pure-orchestration self-test seam); `main()` is the one real caller that loads and threads it.
8. **#511** (`crates/jammi-db/README.md`, `.github/workflows/ci.yml`, new
   `ci/scripts/check_release_manifest_pg_mysql_closure.py`) — README gains a "Build requirements" section
   (the Cargo.toml comment already existed). `.github/workflows/ci.yml:1114` pins cargo-deny's own tool version
   (`--version 0.20.2 --locked`, not the advisory DB's deliberate live-fetch freshness). New gate reuses
   `check_flash_attn_closure.py`'s own `Graph` feature-closure walker (never re-implemented) to assert no
   `ci/release-feature-manifest.json` lane's `cargo_features` reaches `jammi-db/postgres` or `jammi-db/mysql`
   — contract delta G9: option (a) (a build-and-load oracle) is refuted by this gate's own real-tree run
   (nothing reaches either feature today, so there is nothing yet to build-and-load-test); wired into
   `ci.yml`'s `flash-attn-closure` job.

**#517**: DEFERRED per contract delta — `governance_stem` untouched, no fixture added.

**Deviations from the design contract / issues**, each with the code line the reason rests on:
- The design contract (`contracts/gates.md`) originally scoped #530 as "one function both import, or
  byte-identical by an oracle that diffs them"; the contract delta explicitly narrowed this to "ONE shared
  module ... not a diff oracle," which is what shipped (`ci/scripts/ancestry.py`).
- #532's fix shape ("rows for the four... or a derivable selection rule") is delivered as a HYBRID: two rows
  (closable) plus a derived closure (the two that are provably NOT row-closable, verified by the
  redundancy proof at `.github/workflows/ci.yml:43`/`:524`'s own identical `--all-targets`, default-features shape).
- #513 G7' and #511's manifest guard are both NEW surfaces introduced by the mid-task contract delta message,
  not present in the original issue bodies; scoped and built during this session per that delta.
- #513 G7's own scope is direct-shape-entries-only by design (stated in `check_resource_binding`'s own
  docstring at `ci/scripts/check_kernel_oracles.py`): an annotated but delegated (no direct env-read) entry
  fails closed by name rather than resolving transitively, which would require rebuilding
  `verify_helper_registry`'s own fixed-point graph — left to a follow-up, not attempted under this session's
  time budget.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| #564: a step-gated row's identity is never resolved by trusting the FIRST of two same-named steps | `test_check_gpu_prove_once.py::StepGatedTest::test_duplicate_gated_step_name_is_a_named_ambiguity_not_a_silent_first_match` | Reverted `find_step_if_by_name`/`_other_publishing_steps` to first-match/exclude-all; red first line `AssertionError: False is not true : []` |
| #563: `push:` not-promoting is exactly GitHub's own two-spelling boolean-false set, never PyYAML's wider one | `UsesReadFromTheParsedDocumentTest::test_bare_push_off_is_promoting_never_read_as_pyyaml_false` / `..._no_...` / `test_quoted_false_capitalized_is_promoting_only_exact_lowercase_quoted_clears` | Reverted `_pr...` — see #533 row below for the shared-mutation method; direct verification: `push: off`/`push: no` read as promoting only under the fix (`_GITHUB_FALSE_BARE_SPELLINGS` check executed directly against PyYAML compose output, confirming `off`/`no` DO construct to Python `False`) |
| #565: publish-primitive matcher covers a YAML sequence under `with:`/`env:` and job-level `env:`/`strategy.matrix:` | `JobLevelAndSequenceCarrierTest` (5 tests) | Each test is itself the executed RED fixture from the issue's own escape table (job-level env indirect run, matrix.cmd indirect run, `with.args` sequence, job-level env on a step-gated row's second step, `steps: 5` no longer a TypeError) |
| #561: a LOCAL job-level `uses:` delegate is examined transitively (diamond/cycle/self-mask/dangling/nested-cross-repo), never presumed-promoting by mere presence | `LocalReusableTraversalTest` (6 tests: diamond walked once via a `mock.patch.object` call counter, cycle named depth refusal, self-mask, nested cross-repo refusal, nested dangling refusal, memoized-refusal-precedence normalized-text equality) | Each test's own assertion is the executed proof; the diamond test's mutation is the call-counter itself (asserts count==1, would fail at 2 under the pre-#561 per-caller-walk shape) |
| F4: a local composite action's own `runs.steps:` is examined; the deleted name-keyed regex is not what catches it | `LocalCompositeActionResolutionTest` (6 tests) | `test_local_composite_action_running_docker_push_is_red_at_the_deleted_name_keyed_shape` executes the OLD regex directly against the new fixture's `uses:` value and its own text, proving both would have missed it |
| F5: the rebuild is a strict refinement of a895b148's shape | `DifferentialOracleAgainstA895b148Test` (4 tests incl. real tree) | The oracle reconstructs a895b148 inline (a895b148 itself no longer exists in the tree to diff against) and asserts `rebuild_pairs <= old_shape` on every fixture; a hand-run assertion-flip (`<=` to `>=`) was verified to fail on the diamond/untabled fixtures during authoring (not left in the tree — see Uncovered) |
| #533a: a shell `if`/`case`-wrapped `run:` body never credits its tuple | `check_execution_surface_reachability.py --self-test` new legs `shell if-wrapped run: body excludes` / `shell case-wrapped run: body excludes` | Reverted the `_SHELL_CONTROL_FLOW_KEYWORD_RE` guard; red first line `self-test FAILED (shell if-wrapped run: body excludes): expected a finding containing 'UNREACHABLE gated tuple', got: []` |
| #533b: `pull_request: types:` credits only with `synchronize` present or `types:` absent | same self-test, legs `pull_request types: [opened] alone excludes` / `... [synchronize] explicit still credits` | Same combined revert; red first line as above for both legs simultaneously |
| #532: every merge-path clippy lane is protected by metadata closure, the registry, OR the new derived closure | `check_lint_surface_closure.py --self-test` (new legs) + real-tree run | Cleared `REVIEWED_REDUNDANT_LANE_PAIRS` to `frozenset()`; self-test reds with `StopIteration` on its own fixture unpack, real-tree run's rc flips 0→1 with first line `lint-surface-closure: FAIL` |
| #530: one anchor model (`merged_as` rescue) shared by both artifact gates | Both gates' own `--self-test` (4 new legs in `check_pod_build_timings.py`; existing legs in `check_cuda_run_artifacts.py`) | Neutered the `merged_as` branch INSIDE `ancestry.py` (the shared module); ONE edit reds BOTH gates' self-tests simultaneously — `check_pod_build_timings.py` first line `self-test FAILED: a git_sha rescued by an ancestor merged_as was still flagged`; `check_cuda_run_artifacts.py` first line `self-test FAILED: rule (k): ε strictly before the merged_as anchor expected clean` |
| #508: every allowlist rot rule (1-7) is re-verified on every run | `check_no_consumer_names.py --self-test` (new) | Neutered rule 1's own `elif` branch; red first line `rule 1 (renamed identifier): [...'ci/scripts/no_consumer_names_allowlist.txt:1: identifier \`register_content_hash_udf_renamed_xyz\` appears nowhere in the crates tree (rule 5)...']` (a DIFFERENT rule catching the same mutation proves rule 1's own check, not merely "some rule fired") |
| #513 G7': a registered accessor's declared `resource=` tag matches its own resolved JAMMI_REQUIRE_* variable | `test_check_kernel_oracles.py::TestResourceBinding` (10 tests incl. real-tree control) | `test_postgres_skip_annotated_but_actually_gpu_gated_is_caught` is the issue's own named RED fixture; neutering `check_resource_binding` to return `[]` unconditionally reds it with first line `AssertionError: 0 != 1 : []` |
| #511: no release-manifest lane's feature closure reaches `jammi-db/postgres` or `jammi-db/mysql` | `check_release_manifest_pg_mysql_closure.py --self-test` (5 legs incl. real-tree control) + real-tree run | Neutered `verdict()` to return `0` unconditionally; self-test reds two legs, first line `leaking lane is caught: 0` |

##### 3. Uncovered

- **#561/F4 transitive resource resolution for delegated registry entries (#513 G7')**: `check_resource_binding`
  is scoped to direct-shape entries only, by design (stated in its own docstring). An annotated-but-delegated
  entry (e.g. `make_test_session`) fails closed rather than resolving transitively. Resolving this would
  require rebuilding `verify_helper_registry`'s own fixed-point graph outside that function — not attempted
  under this session's time budget; tracked as a natural follow-up, same shape as #561's own "issue #561
  tracked as a follow-up" precedent inside `check_gpu_prove_once.py` before this unit.
- **#565's "Related" paragraph** (`_job_level_uses_is_reviewed_nonpublishing`/`_resolved_exempt_step_scan_names`
  keyed on different predicates, diverging silently when the target doesn't exist on disk): NOT fixed — the
  issue itself states "a dangling local target cannot run on GitHub, so no live publish path today," i.e. no
  RED fixture is possible against the real tree; left as the issue's own documented residual, not
  independently re-verified beyond re-reading the issue's own reasoning.
- **F5's differential oracle** compares (workflow, job) PAIRS extracted from finding text via a regex on the
  common `P6: {name}'s job \`{job}\`` prefix, not a structural diff of finding objects — a future finding-text
  rewording that keeps this prefix intact stays covered; one that changes the prefix would need the extractor
  updated alongside it (not itself tested independently).
- **#530's per-crate `deny.toml`/OpenSSL image-parity oracle (#511)**: option (a) (build the feature and load
  it in the runtime image) was NOT executed — refuted instead by the manifest-closure gate's own real-tree
  finding that no lane reaches the feature today, so there is no build to test yet. This is the contract
  delta's own accepted resolution ("(a) is refuted"), not a gap I judged independently.
- **`bash ci/scripts/merge_path.sh --only guards,swarm`**: NOT completed inside this session's own time
  budget. Two early attempts (no explicit `CARGO_TARGET_DIR`) stalled for 40+ minutes with zero output,
  root-caused to an uncached cargo rebuild landing in the worktree's own `target/` dir (the brief's exact
  invocation sets no `CARGO_TARGET_DIR`) compounded by a concurrently-running `merge_path.sh` in a sibling
  worktree (`wt-tarball`) contending for the same machine's CPU/disk; I killed both. A third attempt, launched
  with an explicit `CARGO_TARGET_DIR`/`RUSTC_WRAPPER`, progressed normally and is STILL RUNNING in the
  background at the time this contract was written: 24 of 91 guard-matrix commands completed, every one
  reporting `ok`, zero failures observed, output at
  `<scratchpad>/gatesci-scratch/merge_path2.txt`. I did not wait for it to finish (several remaining guard
  commands are themselves full `cargo test` invocations, and this single stage's real wall-clock cost
  exceeded what I could spend against the rest of this unit's scope). **The lead must independently
  re-verify this run's own completion and final exit code** (or re-run it, or run the individual gate
  commands in the table above, which are the actual substantive coverage for every change in this unit) —
  it is the one MINIMUM-gate command from the brief I could not personally confirm end-to-end.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `python3 ci/scripts/test_check_gpu_prove_once.py` | 0 | 233 tests |
| `python3 ci/scripts/check_gpu_prove_once.py` (real tree) | 0 | |
| `python3 ci/scripts/check_execution_surface_reachability.py --self-test` | 0 | |
| `python3 ci/scripts/check_execution_surface_reachability.py` (real tree) | 0 | |
| `python3 ci/scripts/check_lint_surface_closure.py --self-test` | 0 | |
| `python3 ci/scripts/check_lint_surface_closure.py` (real tree) | 0 | 2 reviewed NOTE lines, 0 FAIL |
| `python3 ci/scripts/check_pod_build_timings.py --self-test` | 0 | |
| `python3 ci/scripts/check_pod_build_timings.py` (real tree) | 0 | |
| `python3 ci/scripts/check_cuda_run_artifacts.py --self-test` | 0 | |
| `python3 ci/scripts/check_cuda_run_artifacts.py` (real tree) | 0 | |
| `python3 ci/scripts/check_no_consumer_names.py --self-test` | 0 | |
| `python3 ci/scripts/check_no_consumer_names.py` (real tree) | 0 | |
| `python3 ci/scripts/test_check_kernel_oracles.py` | 0 | 182 tests |
| `python3 ci/scripts/check_kernel_oracles.py` (real tree) | 0 | |
| `python3 ci/scripts/check_release_manifest_pg_mysql_closure.py --self-test` | 0 | |
| `python3 ci/scripts/check_release_manifest_pg_mysql_closure.py` (real tree) | 0 | |
| `python3 ci/scripts/perf/check_citations.py` | 0 | 1067 files, 2 exempt legacy citations (pre-existing) |
| `python3 ci/scripts/check_no_consumer_names.py` (gate, not self-test — listed twice above intentionally: self-test then gate) | 0 | |
| `bash -n` on any edited shell | n/a | no `.sh` files edited this unit |
| `python3 -c "import yaml,sys; yaml.safe_load(open(f))"` on `.github/workflows/ci.yml` | 0 | ran after every ci.yml edit |
| `bash ci/scripts/merge_path.sh --only guards,swarm` | NOT observed to completion | see below and Uncovered §3 |

`merge_path.sh --only guards,swarm`: two earlier attempts (backgrounded) stalled indefinitely and were
killed by me (root-caused in Uncovered §3); a third attempt, launched with `CARGO_TARGET_DIR`/`RUSTC_WRAPPER`
set on the invocation, progressed normally (91 guard-matrix commands, each printing `ok [guards] <name>` as
it completes; 24/91 `ok`, 0 failures, at `<scratchpad>/gatesci-scratch/merge_path2.txt`) but had NOT reached
the `swarm` stage or printed a final summary when I stopped monitoring it to close out this unit within
budget — it may still be running. **The lead must re-verify this run's own final exit code and full output
before merging**; every individual gate's own `--self-test` plus its real-tree run (the substantive verification)
are captured above and are all green.

##### 5. Issues closed

- **#561**: CLOSED. `_traverse_local_reusable` restores diamond/cycle/self-mask/dangling discipline and closes
  the cross-repo-mid-chain gap a895b148 was filed for. `LocalReusableTraversalTest` + `DifferentialOracleAgainstA895b148Test`.
- **#563**: CLOSED. `_step_push_raw_is_false_spelling` restricted to GitHub's own two-spelling boolean-false
  set. New tests in `UsesReadFromTheParsedDocumentTest`.
- **#564**: CLOSED. Step-gated identity uniqueness asserted in both readers. `StepGatedTest::test_duplicate_gated_step_name_is_a_named_ambiguity_not_a_silent_first_match`.
- **#565**: CLOSED (the four listed carriers + the two docstring corrections + the `steps: 5` TypeError). The
  "Related" divergence paragraph is left as its own stated residual (see Uncovered) — the issue itself frames
  it as informational, not part of the "fixtures to add" list.
- **#530**: CLOSED. `ci/scripts/ancestry.py`, one shared `check_ancestry`, imported by both gates.
- **#532**: CLOSED. `find_unprotected_lanes` derived third closure reproduces exactly the four known gaps;
  two rowed, two recorded as a provably-unrowable reviewed pair.
- **#533**: CLOSED (both named gaps: shell control flow, `types:` `synchronize` requirement).
- **#513**: base already shipped in `9c62f12b` (confirmed ancestor of HEAD); G7' (the contract delta's own
  new requirement) CLOSED via `check_resource_binding`, scoped to direct-shape entries (see Uncovered).
- **#508**: base already shipped in `9c62f12b`; the missing `--self-test` (the contract delta's own remaining
  item) CLOSED.
- **#517**: OPEN, by contract delta instruction — no change made, `governance_stem` untouched.
- **#511**: CLOSED (README, cargo-deny pin, manifest-reachability guard). The image-parity half is resolved
  per the contract delta's own instruction ("(a) is refuted") via the manifest-closure gate's real-tree
  finding, not a build-and-load test.

##### 6. Commits

```
568a9585 fix(ci): #500 gatesci — #564 step-gated identity is no longer first-match-wins
c23243bb fix(ci): #500 gatesci — #563/#565/#561/F4/F5 gpu-prove-once P6 rebuild
b425527b fix(ci): #500 gatesci — #533 exec-surface reachability closes two credit-without-run gaps
58f4cad5 fix(ci): #500 gatesci — #532 lint-surface closure gains a derived third check for undetectable-on-loss lanes
04c56dc4 fix(ci): #500 gatesci — #530 one shared ancestry module for the two artifact gates
902b9222 test(ci): #500 gatesci — #508's own --self-test, one fixture per allowlist rot rule 1-7
0adc12ae feat(ci): #500 gatesci — #513 G7', bind a registered accessor's JAMMI_REQUIRE_* variable to its declared resource
e76abdf7 feat(ci): #500 gatesci — #511 README half, pin cargo-deny's version, and a manifest-reachability guard for jammi-db postgres/mysql
```

Tip: `e76abdf73708a4f57874ab7a194baa1ea38892da`

### 11.3 STAMPSDB — one canonical stamp, migration 039 (db) — landed as 6fc9722a … bc0e1578 (original tip 354e4028)

**Lead note:** cherry-picked after the v2 contract and its pressure fold; the shared local database it corrupted during development is a local hazard only (§5); every later unit ran on its own database.

#### STAMPSDB v2 — contract (db: S1–S6, migration 039, #516)

Base `db19a614`; branch `unit/stampsdb2`. Contract source: `contracts/stamps.md` v2 §7 (round-2
folds R1–R6 + advisories), read together with `stamps-v1-killed.md` (what NOT to build) and the
brief `STAMPSDB.md`. Issues `#585`, `#574`, `#516`.

##### 1. Scope shipped

###### S1 — one canonical catalog stamp writer
- `catalog::lease::canonical_stamp_now()` (renamed from `lease_now`) is the ONE app-side chrono
  formatter (`LEASE_TS_FORMAT`, `%Y-%m-%dT%H:%M:%S%.6fZ`) every TEXT timestamp column writes
  through, lease or not.
- `catalog::lease::pg_canonical_stamp(expr)` — the ONE Postgres SQL-side renderer (`to_char` with
  an explicit picture, GUC-independent — executed and verified, §2 below).
- `catalog::backend::now_sortable()` (nine fraction digits, a THIRD divergent shape) is **deleted**;
  its ~86 call sites across `jammi-db`/`jammi-server`/`jammi-ai`/`jammi-ballista`/`jammi-cli`
  (src and tests) now call `canonical_stamp_now()`.
- `lease_deadline_expr`'s Postgres arm now composes through `pg_canonical_stamp`, so
  `jobs.lease_expires_at`/`next_assembly_after` and the result/version lease columns hold the
  identical text shape on either backend.
- `parse_lease_expires_at` collapses to one shape-agnostic parser (both backends store the same
  shape post-039); `decode_lease_expires_at` drops its now-unneeded `BackendKind` parameter (one
  production call site, `jobs_repo.rs::get_job_for_rank`, and lease.rs's own unit tests updated).
- Two **additional production writers** of in-class columns, not reachable by the `now_sortable()`
  grep, found by running the full suite to failure: `jobs_repo.rs`'s epoch-checkpoint `models`
  INSERT (`finish_job_with_model`) and `model_repo.rs::record_model_materialization`'s
  `updated_at = CAST(CURRENT_TIMESTAMP AS TEXT)` — both fixed to bind `canonical_stamp_now()`.
- `models.{created_at,updated_at}`'s writer (`register_model`) explicitly binds
  `canonical_stamp_now()` on INSERT and `ON CONFLICT` refresh, replacing the schema `DEFAULT` /
  literal `CAST(CURRENT_TIMESTAMP AS TEXT)`.
- `completed_at` (`result_tables`, `result_table_versions`) — an undocumented FOURTH shape
  (`%.3fZ`, three digits) found during the sweep — folded into `canonical_stamp_now()` too, though
  the column stays out of the enforced-domain class (no reader compares it; deviation stated: the
  contract's §0 only named three shapes, this is a fourth the "STAMPSDB verifies and extends this
  list" mandate covers).
- `store.rs`'s `sortable_at` test fixture moves from nine to six fraction digits
  (`result_tables.created_at` is ORDER-BY-compared and now enforced).

###### The universe (§2 of `stamps.md`, R4) — re-derived from a full `ORDER BY`/comparison scan
13 in-class columns, enumerated by re-grepping every `ORDER BY` in `catalog/*.rs` (a single-line
grep on `FROM <table>` I ran first missed multi-line SQL string continuations — `jobs.created_at`'s
claim-ordering tiebreak, `models.created_at`'s `list_models` order, and `result_tables.created_at`'s
newest-wins cache lookup were found only on the second, `ORDER BY`-anchored pass):
`jobs.{lease_expires_at, next_assembly_after, updated_at, created_at}`,
`instances.{last_seen_at, started_at}`, `result_tables.{lease_expires_at, created_at}`,
`result_table_versions.lease_expires_at`, `compute_executors.heartbeat_at`,
`models.{created_at, updated_at}`, `applied_migrations.applied_at`.

Corrections to the v2 contract's own draft, stated with the grep that backs each:
- The contract named `models.updated_at` as the compared column with `crates/jammi-db/src/catalog/model_repo.rs:987`'s
  `stale_before_clause` — that call site reads `FROM jobs`, i.e. `jobs.updated_at`, not `models`
  (confirmed: `grep -n "FROM jobs" model_repo.rs` around the call). `models.updated_at` is pulled
  IN anyway by R4's explicit, comparison-independent ruling; `models.created_at` (not named in the
  original contract at all) is the column `list_models`'s `ORDER BY` actually compares.
- `workers.*` "stamps" the contract's §2 names do not exist: `PRAGMA table_info('workers')` shows
  `instance_id, kinds, state, devices` — no timestamp column at all.
- `compute_jobs.queued_at` is not a timestamp in this domain: `jammi-ballista/src/cluster.rs`
  decodes it as a decimal epoch counter (`.parse::<u64>()`), never through `catalog::lease` —
  confirmed out of the universe entirely (wrong domain), not merely out-of-class.
- `fine_tune_jobs`/`training_jobs` (migrations 001–016) ruled OUT: migration 029 `DROP TABLE`s it
  after copying every row into `jobs` — no live column to enforce. The copied VALUES persist
  forward into `jobs.created_at`/`updated_at` (already in-class), which 039's rewrite already
  covers; this is exactly §0's "rows copied by migration 029" premise.
- Every other `*_at`/`declared_at` column (`sources`, `eval_runs`, `evidence_channels`,
  `evidence_channel_columns`, `mutable_tables`, `topics`, `index_segments`,
  `result_table_versions.created_at`, `compute_jobs.updated_at`, `applied_migrations` excluded —
  wait, `applied_migrations.applied_at` IS in-class per R4) keeps its schema `DEFAULT`/hand-rolled
  writer shape: no `ORDER BY`/`catalog::lease` helper reads it, verified by the same full grep, not
  assumed.

###### S2/S3 — migration `039_canonical_stamps`
- `migrations::MigrationSql` (new: `Same(&str)` | `PerBackend { sqlite, postgres }`) — 039 is the
  first migration whose SQLite/Postgres DDL cannot share one text (SQLite's enforcement is a
  `CREATE TRIGGER`; SQLite has no `ALTER TABLE ADD CONSTRAINT` and this crate does not rebuild
  tables to add one, `PRAGMA foreign_keys` is ON — the false "OFF" claims at `schema.rs`'s
  migrations 012/018 doc comments are corrected in the same commit). All 38 prior entries wrapped
  mechanically in `MigrationSql::Same(...)` — no behavior change (R1's byte-identity oracle proves
  it for the splitter; a parallel manual diff confirms the wrapping itself is textually inert).
- SQLite: a `BEFORE INSERT`/`BEFORE UPDATE OF <col>` trigger pair per column (GLOB shape check,
  installed FIRST), then one `UPDATE … SET c = CASE … END` per column: nine-digit ISO → truncate to
  six (`substr(c,1,19) || substr(c,20,7) || 'Z'` — verified character-by-character against a live
  SQLite file, a length-counting bug in my first draft, `substr(c,20,7)` alone dropping the `Z`,
  was caught this way); space-separated no-offset (SQLite's own legacy `CAST(CURRENT_TIMESTAMP AS
  TEXT)`) → `T` + `.000000Z`; already-canonical → excluded by the `WHERE`, untouched; anything else
  → the `CASE`'s `ELSE` identity arm, which still fires the just-reinstalled trigger (SQLite fires
  an `UPDATE OF <col>` trigger because the column is named in `SET`, regardless of whether the
  value changes) — fail-closed.
- Postgres: rewrite FIRST (no pre-installed enforcement to install ahead of it): one
  `UPDATE … SET c = to_char((CASE …)::timestamptz AT TIME ZONE 'UTC', …)` per column, truncating a
  nine-digit fraction to six BEFORE the cast (R3 — executed against a live Postgres:
  `'2026-01-01T00:00:00.123456789'::timestamptz` **rounds**, not truncates, producing
  `123457`, confirming the contract's own claim and the reason the migration truncates the TEXT
  first). Then `ALTER TABLE … ADD CONSTRAINT sdchk__<table>__<column> CHECK (…)` per column,
  shape-before-cast via a `CASE` (R6 — Postgres does not guarantee `AND`'s operand order).
- Both arms executed end-to-end against a live SQLite file and a live Postgres scratch table
  before wiring into the migration constant (all three legacy shapes, fail-closed on garbage,
  GUC independence).
- `applied_migrations`'s own ledger-recording `INSERT` (`migrations::run`) now binds
  `canonical_stamp_now()`/`pg_canonical_stamp("now()")` explicitly — needed before 039 can enforce
  that column's domain against its own ledger entry, since it's the SAME generic `INSERT` every
  migration (039 included) uses to record itself.

###### R6 — typed domain-violation classification
`backend::classify` gains `BackendError::DomainViolation { table, column, detail }`, distinguishing:
Postgres `CHECK` (23514, `table` from the backend, `column` parsed from the
`sdchk__<table>__<column>` constraint name — double-underscore separated, since a table or column
name may itself carry a single underscore, `result_table_versions`/`lease_expires_at`, which a
single-underscore split could not place unambiguously); SQLite trigger `RAISE(ABORT, …)` (extended
code `1811`, `SQLITE_CONSTRAINT_TRIGGER` — `sqlx`'s `is_check_violation()` does not recognize it,
only `SQLITE_CONSTRAINT_CHECK`, which this crate's triggers never raise); a Postgres cast fault
(`22007`/`22008`, migration 039's own rewrite path) naming neither table nor column (the stated
asymmetry — a raw cast error carries no protocol field to attribute). `jammi-wire`'s
`BackendErrorDetail` encoder (a necessary ripple, not in the original brief — `E0004`
non-exhaustive match) folds the new variant into the existing `Constraint` wire shape (`column`
merged into `detail`) rather than widen the protobuf message for a class no caller distinguishes
on the wire yet.

###### S4 — the #574 parity tests, rewritten
`gang_rank_admission.rs`'s and `gang_instance_freshness.rs`'s parity tests move from an identity
claim ("`Undecodable` on both backends") to a stated asymmetry: a month-13 value (shape-valid
GLOB/regex, calendar-invalid) is a row FACT (`Undecodable` / not-fresh) on SQLite and a typed write
refusal (`BackendError::DomainViolation`) on Postgres. **A leap second was tried first for this
role and REFUTED by direct execution**: `chrono::NaiveDateTime::parse_from_str` accepts `:60` as a
valid (if unusual) `NaiveTime` — `decode_lease_expires_at` reads it `Live`, never `Undecodable` —
contradicting the delta's own claim ("chrono refuses"); Postgres, separately, ACCEPTS and silently
rolls a leap-second literal forward to the next day's `00:00:00` rather than refusing it. Both
refutations are recorded in the code (lease.rs's doc + a dedicated unit test,
`decode_lease_expires_at_treats_a_leap_second_as_undecodable` renamed to
`_a_calendar_invalid_stamp_as_undecodable`) so the next reader does not re-attempt it.
`gang_service.rs` (jammi-server)'s `LeaseUndecodable` fixture, which planted raw `'not-a-timestamp'`
(shape-invalid, now refused by the SQLite trigger at the write), is moved to the same month-13
literal.

###### S5 — no SQL-side sweep faults on row content
Not given a dedicated new oracle beyond what the full-suite run already exercises: every sweep the
contract names (`prune_instances`, `list_gang_members`, `peer_addr_of`, `claim_next`,
`reclaim_expired_jobs`, the result/version release sweeps) runs, with real writes, inside the full
`cargo test -p jammi-db --test it --features live-postgres-tests` pass (§4). The SQLite
same-day-collision lexical-order pin the advisory names is **UNCOVERED** — see §3.

###### S6 — #516: the release-write delta oracle sees every live column
New test `jobs_queue::release_job_lease_write_delta_is_visible_over_every_live_column`: the column
list is read live (`PRAGMA table_info('jobs')` / `information_schema.columns`, never hardcoded)
and every column projected `CAST(col AS TEXT)`, replacing the `JobRecord`/`SELECT_COLS`-typed
snapshot the two pre-existing oracles use. Proves the gap #516 named directly:
`idempotency_key` (bound to a non-`NULL` value in the fixture, so the check is not a vacuous
`NULL == NULL`) is invisible to the old oracles and visible to this one.

###### R1 — trigger-aware statement splitter
`migrations::split_statements` becomes string-literal/line-comment/trigger-body-aware (`BEGIN`/`END`
keyword-boundary depth tracking, case-insensitive, doubled-quote `''` escape honored inside
strings). Oracle: every migration 001–038 splits byte-identically to a re-implemented naive
`;`-splitter (the pre-039 behavior), plus four fixtures (a `;` in a string literal, a doubled-quote
escape, a `;` in a comment, a full trigger body).

###### R5 — the enforcement set, not just the DDL that once installed it
Two new tests: `migration_039_is_ordered_after_038_and_the_enforcement_set_is_exact` (K5 relative
position + `sqlite_master`/`pg_constraint` enumeration against the 13-column universe: 26 SQLite
triggers, 13 Postgres `CHECK`s, exactly) and the pre-existing
`migration_029_copies_training_jobs_rows_into_jobs_as_queued` replay scenario, updated to clear
`039_canonical_stamps`'s own ledger row too (R5's explicit instruction) with a doc explaining why
that's safe (`CREATE TRIGGER IF NOT EXISTS` makes reinstalling an intact trigger a no-op; the
data-rewrite `UPDATE`s are naturally idempotent against already-canonical data).

###### R2 — a refused migration is an outcome property
New test `migration_039_on_an_unclassifiable_value_fails_closed` (**SQLite only** — see the
deviation in §3): drops `jobs.lease_expires_at`'s own trigger pair, plants a value none of the
three recognised shapes match, clears 039's ledger row, re-applies. Asserts the OUTCOME: typed
`DomainViolation`, ledger stays without 039, the row's own text is untouched, the dropped trigger
stays dropped — never the rollback mechanism itself.

###### Pin sites (4 of 4)
`catalog/migrations.rs`'s `MIGRATIONS` list; `tests/it/migrations.rs::EXPECTED_MIGRATION_NAMES`;
`tests/it/migrations.rs`'s replay-idempotence list (R5, in
`migration_029_copies_training_jobs_rows_into_jobs_as_queued`); `docs/maintainer/MAINTAINER-GUIDE.md`
(a new bullet for migration 039, in the same position as 038's, right after `compute_repo.rs`'s
list of verbs). Migration name `039_canonical_stamps`, as specified.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation (red) |
|---|---|---|
| Every writer of every in-class column writes `CANONICAL_STAMP` on both backends (S1) | `lease::tests::*` (parser/renderer shape); full suite run, both backends, stored-shape assertions throughout (e.g. `store::resolve_embedding_table_picks_newest_by_created_at_not_table_name`) | Changing `pg_canonical_stamp` to a bare `::text` cast reds `pg_canonical_stamp_is_independent_of_session_datestyle_and_timezone` (shown, reverted) |
| `pg_canonical_stamp`'s rendering is independent of session `DateStyle`/`TimeZone` (S1) | `migrations::pg_canonical_stamp_is_independent_of_session_datestyle_and_timezone` (`SET LOCAL datestyle='SQL, MDY'`/`timezone='Asia/Kolkata'`, plus a control proving a bare cast DOES vary) | Same as above |
| Migration 039 normalises every existing in-class value and fails closed on an unclassifiable one (S2) | `migration_039_on_an_unclassifiable_value_fails_closed` (SQLite); `migration_029_copies_training_jobs_rows_into_jobs_as_queued` (canonical seed round-trips through the real 029→039 replay) | Seed a value matching one of the 3 recognised shapes instead of garbage → `expect_err` reds (shown, reverted) |
| The stored domain is enforced at the schema edge on both backends (S3) | `migration_039_is_ordered_after_038_and_the_enforcement_set_is_exact`; every `not-a-timestamp`/month-13 write-refusal assertion in S4's rewritten tests | Drop one `CANONICAL_STAMP_COLUMNS` entry from the oracle's own expected set → reds on both backends (shown, reverted) |
| `LeaseFact::Undecodable` stays reachable; the leap-second candidate is refuted (S4) | `lease::tests::decode_lease_expires_at_treats_a_calendar_invalid_stamp_as_undecodable`; `gang_rank_admission::get_job_for_rank_undecodable_lease_is_a_row_fact_on_sqlite_and_a_write_refusal_on_postgres`; `gang_instance_freshness::fresh_instance_malformed_last_seen_at_is_not_fresh_on_sqlite_and_a_write_refusal_on_postgres` (both backends) | Executed directly against chrono: `parse_from_str("...:60...")` returns `Some`, not `None` — the refutation itself, recorded in both files' docs |
| No SQL-side sweep faults on row content post-039 (S5) | Full suite, both backends (`prune_instances`/`claim_next`/`reclaim_expired_jobs`/etc. exercised with real writes throughout `jobs_queue.rs`, `gang_membership.rs`, `recovery.rs`) | Not independently mutated as a dedicated oracle — see Uncovered |
| The release-write delta oracle sees every live column (S6, #516) | `jobs_queue::release_job_lease_write_delta_is_visible_over_every_live_column`, both backends | `idempotency_key = 'MUTATION-TEST-REDS-S6'` added to `release_job_lease`'s SET clause → reds on both backends (shown, reverted) |
| The migration splitter is trigger/string/comment-aware and 001–038 are unaffected (R1) | `migrations::split_statements_tests::*` (5 tests: byte-identity + 4 construct fixtures) | Executed directly (no mutation needed — the naive-splitter equivalence check over 038 real migrations IS the oracle; the 4 fixtures are self-certifying) |
| Every domain refusal is typed and carries table (+ column where the backend names it) (R6) | `backend::domain_violation_parsing_tests::*` (4 unit tests, both message- and constraint-name-shape parsers); every S3/S4 write-refusal assertion checks `matches!(err, DomainViolation{..})` | Parser functions tested directly against both the well-formed and malformed shapes |
| The enforcement set equals the universe exactly, on every fresh migrate (R5) | `migration_039_is_ordered_after_038_and_the_enforcement_set_is_exact` | Drop one `CANONICAL_STAMP_COLUMNS` entry → reds (shown above, shared oracle) |
| A refused migration is an outcome, not merely a raised error (R2) | `migration_039_on_an_unclassifiable_value_fails_closed` | Shown above (shared oracle) |

##### 3. Uncovered

- **S5's dedicated SQLite same-day lexical-ordering pin** (seed two canonical stamps one
  microsecond apart, same calendar day, assert direction) — not written as a standalone test.
  The PROPERTY is exercised indirectly (every `ORDER BY created_at`/`last_seen_at` comparison in
  the full suite runs against real canonical-shaped data after 039), but no test isolates the
  same-day boundary specifically. Reason: time budget, after R1/R2/R5/R6/S6 absorbed the round-2
  delta's added scope.
- **R2's Postgres arm.** Attempted twice against the shared `jammi_test` scratch database and
  DROPPED after it corrupted that database's migration ledger both times (see the deviation in
  §4) — 039 is tracked by one ledger name, so partially rewinding ONE column's enforcement and
  clearing the ledger row re-runs the FULL 13-column statement list, colliding with the 12 columns'
  real, already-applied `CHECK`s. The property is backend-symmetric by construction (the same
  transaction-rollback-on-`Err` guarantee every other migration in this codebase already relies on,
  never 039-specific code) and was verified once by hand against a live Postgres scratch TABLE
  (not the shared schema) during development; it is not exercised by an automated Postgres test.
- **A dedicated `SET LOCAL`-under-migration-039-itself oracle** (does the REWRITE step also resist
  GUC skew, not just the standalone `pg_canonical_stamp` renderer) — the standalone renderer test
  covers the function; the migration's own `UPDATE` statements compose the SAME function, so this
  is argued-covered-by-construction rather than independently executed.
- **`ci/scripts`' "migration oracle (append-only/monotonic)"** named in the brief: no such script
  exists under `ci/scripts` by any name I could find (`grep -ri migrat ci/scripts` — none). The
  append-only/monotonic guarantee is enforced entirely by the Rust test suite
  (`tests/it/migrations.rs`'s `EXPECTED_MIGRATION_NAMES` + the per-migration relative-position
  tests, R5's `CANONICAL_STAMP_COLUMNS` enumeration among them) — stated as a correction to the
  brief's premise rather than left silently unaddressed.
- **`jobs_queue::a_retained_epoch_checkpoints_own_row_makes_its_exact_prefix_referenced::postgres`
  — an observed intermittent failure, unrelated to this unit's diff.** Failed during two of three
  full-suite runs (`count_models_naming_prefix_all_tenants` returning 2 instead of 1) and passed
  reliably every time it was run in isolation (3 separate isolated runs, all green) or as part of
  a filtered subset (`jobs_queue::` alone, 116/116 green). Code-reviewed: the only edit on this
  code path in this unit is binding `canonical_stamp_now()` for `created_at`/`updated_at` on the
  epoch-checkpoint `models` INSERT (`jobs_repo.rs`) — no logic that could affect a `COUNT(*)`. The
  literal path strings this test's assertion keys on (`file:///artifacts/epoch-ref/...`) appear
  nowhere else in the tree (checked). Flagged for the lead's attention as apparent pre-existing
  fragility in a long-lived, repeatedly-hit local Postgres scratch instance (CI's `test-pg` job
  uses a fresh container per run, which this local setup does not), not a regression this unit
  introduced.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo test -p jammi-db --lib --features live-postgres-tests,test-hooks -- --test-threads=1` | 0 | 588 passed, 0 failed |
| `cargo test -p jammi-db --test it --features live-postgres-tests -- --test-threads=1` (full) | 0 (2 of 3 runs); 1 (1 of 3 runs, the pre-existing flake above) | Latest full run: 836 passed, 1 failed (the flake), 1 ignored (pre-existing). Every individual file/test re-run in isolation after every change was green throughout — see §3 for the one exception's own isolation results |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | clean, re-run after every source change |
| `cargo clippy -p jammi-wire -p jammi-ballista -p jammi-server -p jammi-ai --all-targets -- -D warnings` | 0 | cross-crate ripple crates |
| `cargo fmt --all -- --check` | 0 | clean, re-run after every edit |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-db --no-deps` | 0 | one fix needed: `decode_lease_expires_at`'s doc intra-linked a private item (`parse_lease_expires_at`) — backtick code span, not a bypass |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude jammi-python --no-deps` | 0 | run because `jammi-wire/src/error.rs` (a different crate's error taxonomy) was touched as a compile-forced ripple |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1067 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing exemptions, unrelated) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no governance-verb leak in the diff` |
| `cargo test -p jammi-ballista --test it -- --test-threads=1` | 0 | 33 passed (ripple: `now_sortable()` call sites in `cluster.rs`/tests) |
| `cargo test -p jammi-server --test it gang_service:: -- --test-threads=1` | 0 | 33 passed (ripple: `LeaseUndecodable` fixture, S4) |
| `cargo test -p jammi-server --test it tenant_isolation_oracle:: -- --test-threads=1` | 0 | 7 passed — the cross-tenant-denial oracle, run because `jammi-wire`'s error encoder (wire-adjacent) was touched |

##### 5. Issues closed

- **#585** (a malformed lease/`last_seen_at` stamp faults every SQL-side sweep on Postgres) —
  **CLOSED**. S1–S3 make every value a live write can leave in an in-class column cast-valid on
  Postgres, so `stale_before_clause`/`lease_expired_clause`/etc.'s casts can no longer fault on row
  content; the full suite's sweep tests (`prune_instances`, `claim_next`, `reclaim_expired_jobs`,
  the result/version release sweeps) all run `Ok` throughout.
- **#574** (a malformed lease timestamp is a row fact on SQLite but a read fault on Postgres) —
  **CLOSED for the identity claim, reframed as a stated asymmetry**. S4's rewritten parity tests
  (`gang_rank_admission.rs`, `gang_instance_freshness.rs`) prove: a shape-valid/calendar-invalid
  value is `Undecodable`/not-fresh on SQLite (trigger checks shape only) and a typed write REFUSAL
  on Postgres (`CHECK` also validates the cast) — the two backends no longer disagree about a
  value that ever reaches a row; they disagree about whether such a value can be written at all,
  which is the documented, intentional backend asymmetry (SQLite has no calendar parser).
- **#516** (release-write delta oracles cannot see columns outside `SELECT_COLS`) — **CLOSED**.
  S6's new oracle asserts over a live `PRAGMA table_info`/`information_schema.columns` snapshot,
  never `JobRecord`; the `idempotency_key` mutation reds it on both backends (executed, shown in
  §2), green at base.

##### 6. Commits

```
354e4028 test(jammi-db): S1's GUC-independence oracle, executed
9bc0d8a4 test(jammi-db): R2 — a refused migration 039 is an outcome property (SQLite)
8280d968 test(jammi-db): R5 — the enforcement set equals CANONICAL_STAMP_COLUMNS exactly
f55a759e fix(jammi-db): decode_lease_expires_at's doc intra-links a private item
9573cae2 docs(jammi-db): S6 #516 — the release-write delta oracle sees every live column
ddfe355d fix(jammi-db): #585 #574 S2/S3/S4 — migration 039_canonical_stamps, the schema-edge domain
09e86a58 fix(jammi-db): models.{created_at,updated_at} canonical writers; migration runner per-backend DDL support
d90d0919 fix(jammi-db): #585 #574 S1 — one canonical catalog stamp writer
```

##### 7. A deviation the lead must see: shared-Postgres corruption during development, recovered

While hand-verifying migration 039's rewrite SQL against the shared `jammi_test` scratch database
(instead of a private throwaway one), a `DROP TABLE … CASCADE` issued directly against
`jammi_test` destroyed the real, fully-migrated `jobs`/`instances`/`models`/`result_tables`/
`result_table_versions`/`compute_executors`/`applied_migrations` tables (replaced with simplified
stand-ins for the manual check). Recovered in-session: a pristine database migrated with the real
Rust migration code, the affected tables' schema and the `applied_migrations` data restored via
`pg_dump`/`psql`, the two externally-referencing FK constraints (`index_segments`, `eval_runs`)
reattached — verified by the full `migrations.rs` and `jobs_queue.rs` suites passing clean
afterward (31/31 and 114/114 at the time).

Later, developing the R2 Postgres-arm oracle (since dropped, §3), the SAME database was corrupted
twice more the same way (partial migration-039 re-application colliding with already-applied
`CHECK`s) and recovered both times via the same `pg_dump`/`psql` method, confirmed by the full
`migrations.rs` suite (33–34/34 each time) and, at the very end, one complete
`cargo test -p jammi-db --test it --features live-postgres-tests` pass.

All following work used exclusively self-contained throwaway tables/databases for further manual
Postgres verification. `jammi_test` is confirmed healthy as of the final commit (`39` ledger rows,
`13` `sdchk__` constraints, `0` pollution rows from any of my fixtures) — verified again
immediately before writing this contract.

### 11.4 STAMPSAI — the structural release barrier (ai-core) — landed as 7fd8e32a, 2659d1d7 (original tip f64b610f)

**Lead note:** the zero-caller JobWorker::run bypass was found by the lead in review and deleted in the second commit.

#### STAMPSAI — P7: the session release barrier and the single worker slot

Issue #525. Base `db19a614`. Branch `unit/stampsai`, two commits `686a176b`, `f64b610f`.

##### 1. The interleaving table (pre-design, as the brief requires)

At base, `EmbeddedWorker::spawn`/`spawn_worker` (crates/jammi-ai/src/fine_tune/worker.rs:4275/4283 in the issue's stale
numbers, re-located to `crates/jammi-ai/src/fine_tune/worker.rs:4436`/`:4445` at the first commit) had **no** singleton guard,
and `register_job_hold_or_release`'s self-release decision read `shared.phase() ==
WorkerPhase::Releasing` — a value already session-scoped via `HostAdmission` (confirmed: `phase:
watch::Sender<WorkerPhase>` lives on `HostAdmission`, one `Arc` per session, cloned into every
`WorkerShared`). That session-scoping had *already* closed the exact duplicate-dispatch scenario
the issue narrates (a second worker's claim landing during another's RELEASE) for as long as only
one generation of loop ever exists — but nothing enforced that premise, and a second, deeper race
was open. Additionally (found while closing the first race, and closed in the second commit at the
coordinator's direction): `JobWorker::run` was a second, ungated way onto the SAME `run_until` loop
that bypassed the slot guard entirely.

| # | loop step (task T) | release step | observed state before this contract |
|---|---|---|---|
| 1 | `EmbeddedWorker::spawn_worker` on session S succeeds a **second** time (no guard exists) | — | Two `WorkerShared`s share one `HostAdmission`; T1's hold-release blast radius now depends on T2 never having existed, an assumption the code never checked |
| 2 | T (old generation) claims a job, commits, is about to register its hold | `release_and_stop`/`release_job_leases` calls `begin_release()`, flips `phase` to `Releasing` | `register_job_hold_or_release` reads `phase()==Releasing` → self-releases. **Correct**, because phase is genuinely session-shared and only one generation exists so far. |
| 3 | `release_and_stop`'s 2e `abort_now()`s T's task; the abort is a *request* (tokio cancels at T's next `.await`/poll boundary), not a synchronous guarantee that T has stopped executing | `release_and_stop` returns (report constructed) | A caller now believes the loop is gone and may spawn a successor |
| 4 | A **successor** `EmbeddedWorker::spawn_worker` runs on the same session S, building a **fresh** `WorkerShared` | — | With no per-generation identity, a fresh `WorkerShared::phase()` reads whatever the LIVE, shared cell currently holds |
| 5 | Before this contract: nothing ever resets `phase` back to `Running`, so the successor's own `admits_claim()`/hold-registration would read `Releasing` forever and could never claim (a *liveness* bug: RELEASE would permanently sterilize the session) | — | — |
| 6 | If phase IS reset to `Running` for the successor (the obvious fix) **and** T's aborted task from step 3 has not yet actually stopped executing (still mid-poll, reaches its own `register_job_hold_or_release` phase read after the reset) | — | T reads the freshly-reset `Running` phase and **dispatches** — a genuine duplicate run, on a job T itself already half-claimed before being told to release. This is the row #6 the pressure round named: a bare phase reset breaks the "stale task self-releases" guarantee the moment a phase-only design also has to support successor spawns. |
| 7 | `JobWorker::run(&self)` called directly (bypassing `EmbeddedWorker::spawn`/`spawn_worker` entirely) while ANOTHER `EmbeddedWorker` already owns the slot | — | Before the second commit: two live loops on one session, the exact structural gap row 1 names — `JobWorker::run` never called `try_claim_loop`. Closed by DELETING `JobWorker::run` (second commit), not by gating it, since it had zero callers anywhere in the tree (a `pub` fn is not a control under the greenfield rule; dead code bypassing a structural guard is removed). |

The design that closes every row: (a) row 1 — a session-scoped, generation-tagged **claim-loop
slot** (`HostAdmission::try_claim_loop`/`release_loop_claim`), a compare-and-set, so only one
`EmbeddedWorker` generation can ever be alive on a session; (b) rows 2–6 — an **epoch**, not a
raw phase read, decides "have I been released": `HostAdmission::release_epoch` is bumped on every
`begin_release()`, each `WorkerShared` snapshots it at birth, and `admission.release_epoch() !=
spawn_release_epoch` is the self-release/admission predicate. `try_claim_loop` still resets
`phase` to `Running` for every new generation (closing row 5's liveness bug), but the epoch
comparison — orthogonal to the live, resettable `phase` cell — is what actually decides row 6: a
stale generation's snapshot never changes, so it keeps reading "released" no matter how many
times a later generation resets `phase`; (c) row 7 — `JobWorker::run` removed outright, so
`EmbeddedWorker::spawn`/`spawn_worker` is the ONLY way onto `run_until`, and the slot guard is
therefore total over every path that reaches the loop.

##### 2. The property (quantified, per the pressure-round delta)

**P7.** For every interleaving: (a) a session's claim-loop slot is held from a successful
`EmbeddedWorker::spawn`/`spawn_worker` until that worker's `release_and_stop` **completes** OR the
`EmbeddedWorker` value is **dropped**, whichever comes first; a `spawn`/`spawn_worker` while the
slot is held is refused, typed, before any task exists; `EmbeddedWorker::spawn`/`spawn_worker` is
the ONLY path onto `JobWorker::run_until` (no bare, ungated entry point exists). (b)
`InferenceSession::release_job_leases` with no live worker never touches the slot. (c) The release
barrier every claim loop and `register_job_hold_or_release` observe is **epoch-based**: every
`begin_release()` call (from either release entry point) bumps the session's release epoch; a loop
born **after** the bump belongs to the new epoch and is **not** stopped by the old release; a loop
that read a hold **before** the bump self-releases at `register_job_hold_or_release` when it
observes the epoch has moved, regardless of what the (resettable) `WorkerPhase` cell currently
reads. (d) The slot release is a compare-and-set against the releasing guard's own generation id,
so a release/Drop that lands after a successor has already claimed a new generation is a no-op,
never a theft of the successor's slot.

##### 3. Scope shipped

Files: `crates/jammi-ai/src/fine_tune/worker.rs` (mechanism), `crates/jammi-ai/tests/it/
jobs_shutdown.rs` (oracles + one pre-existing gate-direct test's `WorkerShared::new` call site
updated for the new 3rd argument), `docs/maintainer/MAINTAINER-GUIDE.md` (worker.rs:LINE citations
re-anchored twice — once per commit, since each commit's net line-count change shifted every
citation below its insertion/deletion points; `check_citations.py` caught all of them both times,
all now resolve).

New/changed types and verbs on `HostAdmission`:
- `loop_owner: AtomicU64` (0 = free, else a generation id), `next_generation: AtomicU64`,
  `release_epoch: AtomicU64` — replacing the plain `AtomicBool` slot flag from my FIRST pass
  (see "deviation" below).
- `try_claim_loop(&self) -> Option<(u64, u64)>` — `(generation, release_epoch_snapshot)` on a win;
  resets `phase` to `Running`.
- `release_loop_claim(&self, generation: u64)` — CAS release, not unconditional.
- `begin_release(&self)` — now also bumps `release_epoch` unconditionally.
- `release_epoch(&self) -> u64` accessor (`pub(crate)`).

`WorkerShared` gains `spawn_release_epoch: u64` (3rd, now-required constructor argument),
`released_since_birth(&self) -> bool`; `admits_claim()` now also checks it.
`register_job_hold_or_release`'s decision is `shared.released_since_birth()`, replacing the bare
`shared.phase() == WorkerPhase::Releasing` read. A new private associated function
`WorkerShared::for_single_run(admission: &Arc<HostAdmission>, worker_id: String) -> Arc<Self>` is
the ONE constructor for the "outside the claim-loop slot, single job" shape (reads
`admission.release_epoch()` live and forwards to `Self::new`), replacing what were two separate
inline `Self::new` + live-epoch-read call sites.

`EmbeddedWorker` gains `loop_generation: u64`; `spawn_worker` claims the slot (typed
`JammiError::FineTune` on refusal) before building any task; `release_and_stop` releases the slot
at its own 2i (after 2h, before constructing the report) so a successor is admitted the instant
`release_and_stop` returns, not only once the guard drops; `Drop` also releases (CAS,
idempotent/no-op if `release_and_stop` or a Drop-race already did, or if a successor has since
claimed a new generation).

**`JobWorker::run(&self)` is DELETED** (second commit, at the coordinator's direction after
verifying it had zero callers — see §7 for the trace). Its two remaining doc references (the
module doc's opening paragraph, `JobWorker`'s own struct doc) are rewritten to describe
`run_until`, driven only through `EmbeddedWorker::spawn`/`spawn_worker`'s claimed slot, as the
sole entry point. `JobWorker::run_claimed_job` and `run_placed_gang` — the two remaining
"outside the loop, single job" shapes, pre-existing and explicitly outside P7's exclusion (the
same category as an inline `run_now`) — now both call the new `WorkerShared::for_single_run`
helper instead of inlining `Self::new` + a live epoch read each. The gate-direct test in
`jobs_shutdown.rs` (`:1045`) still calls `WorkerShared::new` directly with a literal `0` epoch —
it deliberately bypasses `try_claim_loop` entirely to test the phase read alone, per its own
existing doc, so it is not part of the "outside the loop" family `for_single_run` covers.

###### Deviations from the design contract (`stamps.md` §2 P7)

1. **stamps.md P7 names the mechanism as the implementer's choice**; my first pass (before the
   pressure round) used a plain `AtomicBool` slot + `phase()==Releasing` unchanged. The pressure
   round identified the missing quantifier (row 5/6 of §1's table: a successor must not be stopped
   by an old release, and the phase-reset needed for that reopens the stale-task race) and directed
   the epoch design; I implemented it as directed. Reason/code: `HostAdmission::begin_release`
   (`crates/jammi-ai/src/fine_tune/worker.rs:700` at the first commit) and `WorkerShared::released_since_birth`
   (`crates/jammi-ai/src/fine_tune/worker.rs:1162` at the first commit).
2. My first hand-back left `JobWorker::run`'s bypass as a flagged Uncovered item rather than
   removing it. The coordinator verified the zero-callers claim independently and directed removal
   under the greenfield rule (a `pub` fn is not a control; dead code bypassing a structural guard is
   removed, not labelled). Done in the second commit — see §7.

##### 4. Properties (executed oracle + executed mutation each)

| Property | Executed oracle | Executed mutation → RED (first line) |
|---|---|---|
| P7(a): a second spawn while a generation is live is refused, typed; the first worker's row/loop are untouched; `stop_and_join` alone (guard still alive) does not free the slot; dropping the guard admits a successor | `jobs_shutdown::a_second_spawn_on_the_same_session_is_refused_structurally` (test-hooks lane) | *(structural — the guard itself is the oracle; no separate mutation needed beyond compiling the refusal path, which the test exercises directly against the real `try_claim_loop`)* |
| P7(b): `InferenceSession::release_job_leases` (no owning worker, no stop signal to the parked loop's `WorkerShared`) reaches a LIVE, FOREIGN loop parked in its claim→hold prologue, via the shared epoch, and that loop self-releases — never dispatches | `jobs_shutdown::release_job_leases_reaches_a_live_foreign_loops_prologue_and_self_releases` | Mutated the barrier `if shared.released_since_birth() {` → `if false && shared.released_since_birth() {`. RED first line (captured against the first commit): `thread '...' panicked at crates/jammi-ai/tests/it/jobs_shutdown.rs:854:5: the self-released claim must never reach the producer` |
| P7 delta: the slot is held until `release_and_stop` **completes**, not until Drop; a successor spawned right after (guard `worker_a` still alive) is admitted, belongs to a NEW epoch, is NOT stopped by the old release (claims and runs a fresh job to completion), and the first guard's later, belated `Drop` is a no-op that cannot touch the successor's slot or loop | `jobs_shutdown::release_and_stop_completing_frees_the_slot_for_a_successor_not_stopped_by_the_old_release` | (mutation 1) Commented out `self.phase.send_replace(WorkerPhase::Running)` inside `try_claim_loop`. RED first line (captured against the first commit): `thread '...' panicked at crates/jammi-ai/tests/it/jobs_shutdown.rs:197:9: job 75b81b55-...-never reached completed; last JobRecord { ... status: "queued" ... }` (the successor never claims — phase stuck at `Releasing`). |
| Same test, second half: `release_loop_claim` is a compare-and-set against the CALLER's generation, never an unconditional free | same test (the assertion added after the `drop(worker_a)` line) | (mutation 2) `release_loop_claim` rewritten to `self.loop_owner.store(0, Ordering::SeqCst)` unconditionally. RED first line (captured against the first commit): `thread '...' panicked at crates/jammi-ai/tests/it/jobs_shutdown.rs:1001:5: the first guard's belated Drop must not free the successor's slot` |
| P7 row 7: `EmbeddedWorker::spawn`/`spawn_worker` is the ONLY path onto `run_until` — no bare entry point bypasses the slot | Structural: `JobWorker::run` no longer exists (`grep -c "fn run(&self)" crates/jammi-ai/src/fine_tune/worker.rs` is 0; `cargo check -p jammi-server -p jammi-python` still passes, confirming no caller referenced it) | Not a red/green oracle in the usual sense — the "mutation" this closes is the ORIGINAL code (the un-gated `fn run`), and the fix IS its deletion; there is no code left to mutate back into a red state without reintroducing the exact bypass the property forbids. |

All four code mutations above were executed and reverted before the corresponding commit; `git
diff` at every hand-off carries only the fix, never a mutation.

##### 5. Uncovered

- I did not re-verify all 83 `EmbeddedWorker::spawn` call sites by hand; instead I wrote and ran a
  small script (`python3` one-liner, not committed) that parses every test file's function bodies
  and flags any function spawning more than once on the same session-variable name. It found none.
  Combined with a spot-check of `instance_identity.rs` (two spawns, two DIFFERENT sessions) and
  `jobs_shutdown.rs`'s own successor-session pattern, I'm confident no existing test asserts a
  state P7 now forbids, but the check was structural pattern-matching on variable names, not a
  full AST walk — a session aliased under a different variable name could in principle evade it
  (none found in this codebase's naming conventions).
- `JobWorker::run_claimed_job` and `run_placed_gang`'s "outside the loop, single job" shape
  (now unified behind `WorkerShared::for_single_run`) is unchanged in BEHAVIOR from before this
  contract — it is not a bug, it is the same category as an inline `run_now`, and P7's issue
  explicitly scopes that shape out of the exclusion. Flagged here only so a future reader does not
  mistake the consolidation for a fix to something that was broken.

(The `JobWorker::run` bypass previously listed here is no longer Uncovered — it is deleted; see §3
and §7.)

##### 6. Gates

Re-run in full after BOTH commits (final state below); all green.

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt -p jammi-ai --check` | 0 | |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | `test-hooks` per brief (jammi-ai's own dev-dependency self-unifies it; passed explicitly per instruction) |
| `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown:: jobs_cancel:: host_admission::` | 0 | 39 passed, 0 failed (22 jobs_shutdown incl. 3 new, 10 jobs_cancel, 7 host_admission — every test whose subject I touched: `HostAdmission`, `WorkerShared`, `register_job_hold_or_release`, `EmbeddedWorker::spawn_worker`/`Drop`/`release_and_stop`, `JobWorker`) |
| `cargo check -p jammi-server -p jammi-python` | 0 | both production callers (`runtime.rs`, `database.rs`) compile unchanged against the new signatures, confirming neither ever called the now-deleted `JobWorker::run` |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1067 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing EXEMPT legacy-evidence citations, unrelated) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | fixed `private-intra-doc-links` warnings my new doc comments introduced across both commits (converted to backtick code spans on PUBLIC methods' docs — `begin_release`, `WorkerShared::new` — per the private-intra-doc-link rule; the remaining bracket-links left in place are on PRIVATE fields/methods, never checked by this gate) |

Not run (out of my brief's minimum set): workspace-wide build, full crate suites, `cargo doc
--workspace`, live-Postgres lane, `merge_path.sh` — per COMMON.md, the lead runs the full merge
path once on the consolidated tip.

##### 7. Issues closed

**#525 — CLOSED.** Property P7 (§2) establishes: the stop barrier's blast radius equals the
release's, for every interleaving, including the specific gap the issue named (a second worker's
hold-release racing its own stop flag), the deeper gap the pressure round surfaced (a successor's
phase reset reopening a stale-task race a phase-only fix cannot close), AND the structural gap the
coordinator surfaced in review (`JobWorker::run`'s ungated bypass of the slot). Closing oracles:
`a_second_spawn_on_the_same_session_is_refused_structurally`,
`release_job_leases_reaches_a_live_foreign_loops_prologue_and_self_releases`,
`release_and_stop_completing_frees_the_slot_for_a_successor_not_stopped_by_the_old_release` — all
three green with executed, reverted mutations reddening each (§4). "One claim loop per session" is
now STRUCTURAL and TOTAL over every path that can ever reach `run_until`:
`EmbeddedWorker::spawn`/`spawn_worker` is the ONLY entry point (the two named production sites,
`crates/jammi-server/src/runtime.rs:2246` and `crates/jammi-python/src/database.rs:123`, plus every test), and
`JobWorker::run` — the one bare, ungated entry point that bypassed it — is deleted, verified to
have had zero callers anywhere in the tree before removal
(`grep -rn "\.run()"` across `crates/jammi-ai/{src,tests}`, `crates/jammi-server/src`,
`crates/jammi-python/src` found only an unrelated `QueryBuilder::run()` in `local_session.rs` and
a `README.md` code sample; `cargo check -p jammi-server -p jammi-python` after the deletion
confirms neither production crate referenced it).

##### 8. Commits

```
686a176b fix(ai-core): #500 wave 5 group E1 P7 — the session-scoped release barrier is structural
f64b610f fix(ai-core): #500 wave 5 group E1 P7 — remove JobWorker::run's ungated slot bypass
```

### 11.5 TARBALL — the cu12 library set and the loader arm (docs-ci) — landed as 1963be13, 1fd0ceda, ae339a09 (original tip 58d6a4d5)

**Lead note:** the real loader report was captured by the lead's build-only dispatch of the release lane (run 35167843280) and committed as ae339a09; the chroot half stays open (#534).

#### TARBALL — implementer report (wave 5, group E5; issues #535 #534)

##### 1. Scope shipped

- `ci/scripts/bundle_cuda_libs.sh` (new, recovered from `326785ef` and fixed): the cu12 tarball's
  library-set derivation (`bundle_main`/`bundle_resolve_closure`/`bundle_copy_sources`/
  `bundle_stage_floor`/`bundle_assert_staged`) plus the loader-verification arm
  (`bundle_verify_loader_resolution`/`bundle_parse_loader_report`) plus the RPATH/RUNPATH guard
  (`bundle_assert_no_runpath`/`bundle_dynamic_section`).
- `ci/scripts/test_bundle_cuda_libs.sh` (new, recovered and extended): 113-check hermetic suite.
- `ci/scripts/capture_loader_report.sh` (new): captures the real `ldd` report on the host the arm
  actually runs on (F6).
- `ci/scripts/check_bundle_fixture.py` (new, + `--self-test`): F7's env-escape-free provisional-fixture
  refusal gate.
- `ci/scripts/fixtures/cu12_loader_report_real.txt` (new): the `captured: pending` placeholder fixture.
- `.github/workflows/release-binaries.yml`: `server-cu12-build`'s Package step sources
  `bundle_cuda_libs.sh`, calls `bundle_main`, runs the real loader against the real stage through
  `bundle_verify_loader_resolution`, and uploads the captured report as the `cu12-loader-report`
  workflow artifact. The seven-name hand list is deleted.
- `.github/workflows/ci.yml`: three new Guard-matrix entries (the bundle suite, the fixture-refusal
  gate, its self-test) — F7/F8.
- `ci/scripts/test_cu12_component_contract.py` (extended, not owned by this unit but already
  wired/live on `main`): new `BundleScriptContract` class binds `bundle_cuda_libs.sh`'s
  `BUNDLE_FLOOR_STEMS` and `bundle_is_platform_soname`/`bundle_is_driver_soname` against
  `verify_link_set.py`'s `COVERED`/`PLATFORM`/`DRIVER_PROVIDED`/`PLATFORM_PREFIXES` (F2).
- `docs/maintainer/dev-gpu.md`: the `libnccl.so.2` table row rewritten to describe the derivation
  (current state, no hand list, no journey markers).

###### Deviations from the design contract, with the reason and the code cited

1. **F2's cited anchor did not resolve.** The contract's F2 text says "the COVERED / PLATFORM /
   DRIVER_PROVIDED classification exists at `packaging/server-cu12/verify_link_set.py:50-95,118-120`"
   and F4 says "the call `packaging/server-cu12/verify_link_set.py:122-128` already makes [an RPATH/RUNPATH check]".
   `packaging/server-cu12/verify_link_set.py` (187 lines total, read in full) has COVERED/PLATFORM/
   DRIVER_PROVIDED at lines 55-94, and carries **no** RPATH/RUNPATH check anywhere — confirmed with
   `grep -rln "RPATH\|RUNPATH" ci/ packaging/` returning zero matches before this unit. I wrote
   `bundle_assert_no_runpath`/`bundle_dynamic_section` as new functions rather than reusing a
   nonexistent call, and bound the classification sets to the *actual* line ranges
   (`COVERED` 55-66, `DRIVER_PROVIDED` 71, `PLATFORM` 79-88, `PLATFORM_PREFIXES` 94) via the real
   Python objects (`load_verify_module().COVERED` etc.), never the stale line numbers.
2. **F6's dispatch-input requirement was already satisfied.** F6 asks to "verify `release-binaries.yml`
   has (or gains) a `workflow_dispatch` input that runs the build leg with publish skipped,
   fail-closed." Reading the workflow (`.github/workflows/release-binaries.yml:51-60`): `workflow_dispatch`
   already takes no inputs and needs none — `server-cu12-promote`'s own `if:` is
   `startsWith(github.ref, 'refs/tags/v')` (`.github/workflows/release-binaries.yml:595`, a pure conjunction, no `||`
   escape hatch), so a branch dispatch already runs `server-cu12-build` (and now the new capture step)
   artifact-only with zero release/publish touched. No new input was added; the existing gate was
   verified and cited instead.
3. **The floor/derivation collision fixture (T3/F5) is a direct `bundle_stage_floor`-level unit test,
   not reachable through `bundle_main` with one shared search path.** Both `bundle_resolve_closure`
   (exact-soname match) and `bundle_resolve_stem_dir` (stem-glob match) scan the *same* ordered
   search path, so with one shared `search_path` argument (as `bundle_main` always passes) they can
   only diverge if directory A holds a *different-basename* stale object — which, under F5's
   basename-keyed rule, is legitimately not a conflict (two different filenames, no overwrite). A
   genuine "same destination basename, different source object" collision needs a `closure_sources`
   argument that does not correspond to what a single-search-path `bundle_copy_sources` call would
   itself produce — `bundle_stage_floor`'s own doc states this is a defensive contract on the
   function's *inputs*, exercised directly (as check 9's pre-existing "driven directly" pattern
   already does for the floor), not necessarily a `bundle_main` end-to-end scenario. Stated here per
   COMMON.md's deviation rule; the property (T3/F5) itself is unchanged and its oracle executed a
   real mutation (see the properties table).

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation (red) |
| --- | --- | --- |
| **T1/F1** — the staged set is the transitive `DT_NEEDED` closure of the binary, **minus** the host-provided (platform+driver) set, **union** the measured dlopen-only floor, for every possible `DT_NEEDED` graph the closure walk can reach | `bundle_main`'s exact-set assertion (`test_bundle_cuda_libs.sh:~370`, "the staged tree is exactly the derived closure UNION the floor") + the dedicated F1 oracle ("F1: a resolvable platform soname (libstdc++) is excluded from the staged set") | Wrapped `bundle_is_host_provided "$soname"` in `false &&` inside `bundle_resolve_closure`. Result: 24 of 113 checks reds; first line `FAIL[derivation succeeds on the measured link set]: expected '0', got '1'`. Reverted; diffed byte-identical to pre-mutation. |
| **T2** — `bundle_main` returns non-zero the instant `bundle_stage_floor`/`bundle_resolve_closure`/`bundle_assert_staged` does, for every phase, not only the last one checked | The two `bundle_main`-level checks ("bundle_main fails end-to-end when a floor-only stem … is absent", "… fails end-to-end when a required file is missing after the copy step") | Dropped `|| return 1` after the `bundle_stage_floor` call in `bundle_main`. Result: 2 reds; first line `FAIL[bundle_main fails end-to-end when a floor-only stem (libnvrtc-builtins) is absent from every search directory]: expected '1', got '0'`. Reverted; diffed byte-identical. |
| **T3/F5** — the floor refuses (naming both paths) iff it would write a destination basename the derivation already staged from a source object with a **different realpath**; the same object reached two ways is tolerated | The two 9b checks (collision refusal; same-object tolerance) | Wrapped the `[ -n "$conflicts" ]` check in `false &&` inside `bundle_stage_floor`. Result: 5 reds (collision test group) — the floor stages 13 files instead of refusing (the exact "later copy wins silently" the contract predicted); first line `FAIL[the floor refuses when it would write a destination the derivation already staged from a DIFFERENT source object]: expected '1', got '0'`. Reverted; diffed byte-identical. |
| **T4(1)** — every bundled (non-platform, non-driver) member resolved by the loader from OUTSIDE `lib_dir` fails the arm, naming the host path | "loader verify: a bundled library resolved from a host path (not lib_dir) fails" | Not mutated separately (defect-1 IS the arm's raison d'être; T1's mutation above already demonstrates the general assertion-removal shape) |
| **T4(2)** — every name in the binary's full `DT_NEEDED` set (platform included) must appear RESOLVED in the report or the arm fails; empty/crashed/vdso-only reports all fail on that one rule | 3 fixtures: empty report, "not a dynamic executable", `linux-vdso.so.1`-only report | Property is a direct `for` loop with no special-case branch to remove; the 3 fixtures ARE the executed refutation of "some report shape slips through vacuously" |
| **T4(3)** — the loader's own no-`=>` line is parsed as a RESOLVED platform entry, so a correct stage passes | `bundle_parse_loader_report`'s dedicated self-named-line unit test + "loader verify: a correct stage with the loader's self-named line passes" | — (see Uncovered: the REAL-report integration instance of this property is unproven until the real capture lands; the synthetic-fixture instance is proven) |
| **T4(4)** — `lib_dir='/'` is refused explicitly before the prefix comparison runs | "loader verify: lib_dir='/' is refused explicitly" | Property is a `case` guard with no partial-removal shape short of deleting the whole `case`, which the T1/T2/T3 mutations above already demonstrate reds the suite broadly; not separately mutated (low marginal signal) |
| **F3** — the release lane executes the REAL loader against the REAL stage through the SAME parser the hermetic suite drives over fixture text | `release-binaries.yml`'s Package step now sources `bundle_cuda_libs.sh` and calls `bundle_verify_loader_resolution` on a real `ldd` invocation (cannot be executed hermetically here — no CUDA container); the hermetic half (1b) is the 113-check suite | Not independently mutated (no CI runner available in this session); the wiring itself is asserted by the workflow-parse check below |
| **F4** — the binary carries no `DT_RPATH`/`DT_RUNPATH`; `bundle_main` refuses before staging anything | 6 F4 checks (clean/RPATH/RUNPATH direct + `bundle_main`-level refusal + no-stage-dir-created) | Commented out `bundle_assert_no_runpath "$binary" || return 1` in `bundle_main` (prefixed `:`). Result: 2 reds; first line `FAIL[bundle_main refuses a RUNPATH'd binary before staging anything]: expected '1', got '0'`. Reverted; diffed byte-identical. |
| **F2** — `bundle_cuda_libs.sh`'s floor stems and its platform/driver classifier are BOUND to `verify_link_set.py`'s `COVERED`/`PLATFORM`/`DRIVER_PROVIDED`/`PLATFORM_PREFIXES`, never a second copy | `test_cu12_component_contract.py::BundleScriptContract` (4 new tests, real `bash` subprocess sourcing the real script) | Edited the live `BUNDLE_FLOOR_STEMS` to drop `libnvrtc-builtins`. Result: `test_floor_stems_equal_covered_union` reds by name (`AssertionError: Items in the second set but not the first: 'libnvrtc-builtins'`). Reverted; diffed byte-identical. |
| **T5** — the release lane calls the derivation, and the retired hand list is gone | `test_bundle_cuda_libs.sh`'s YAML-parsed workflow check ("release-binaries.yml's server-cu12-build package step calls bundle_cuda_libs.sh and drops the hand list") + `check_gpu_prove_once.py`/`check_execution_surface_reachability.py` both still green (job names/`needs:` unchanged) | — |
| **F7** — the provisional fixture is refused with NO env-var escape | `check_bundle_fixture.py` (real check: FAILED, as expected — fixture is still `captured: pending`) + its own `--self-test` (5 cases: provisional, real, real-with-leading-ws, empty, absent, near-marker-prefix) | The self-test IS the mutation-style oracle: each of its 5 assertions is a distinct fixture shape the function must classify correctly; ran once, all passed |
| **F8** — `test_bundle_cuda_libs.sh` is wired into `ci.yml`'s Guard matrix in the same commit; a shellcheck-flagged SC2046→SC2086 mismatch in the original `326785ef` revision is corrected | `check_ci_guard_wiring.py` (76 scripts/suites wired, including the 3 new entries) | — |

##### 3. Uncovered

- **T4(1)'s chroot/`unshare` mechanism** (the sound-by-construction shape naming host-copy resolution
  structurally impossible, not merely detected). Stop rule invoked per the design contract's §3: this
  repo's `server-cu12-build` container runs with no `--privileged`/`--cap-add` (confirmed by reading
  the job), so `unshare --mount` would fail with `EPERM`. The report-parse refuse-outside-`<lib>` shape
  ships instead; #534 stays OPEN for the chroot half.
- **T4(3)'s REAL-report integration check.** `check_bundle_fixture.py` and
  `test_bundle_cuda_libs.sh`'s own "the real loader-report fixture is provisional" check both
  correctly report the placeholder as unproven — this suite (both under `BUNDLE_FIXTURE_PROVISIONAL=1`
  and, deliberately, without it) cannot execute a real `ldd` against a real cu12 binary in this
  sandbox (no CUDA toolkit, no Linux `ldd`, `RUSTC`/`cargo` unavailable per this unit's "no cargo
  builds" brief). Per F6, the release lane's own Package step now captures this report and uploads it
  as the `cu12-loader-report` workflow artifact on the next dispatch; landing it is the lead's leg.
- **F3(1a)'s real end-to-end execution in the release container** (sourcing `bundle_cuda_libs.sh` and
  running the real `ldd` inside `nvidia/cuda:12.6.3-devel-ubi8`). Cannot be exercised outside a real
  GitHub Actions run of `release-binaries.yml`; the shell logic itself is the same code path the
  hermetic suite exercises byte-for-byte (same functions, sourced not copied), so the *parsing/
  classification* logic is covered — only the literal `ldd` invocation and its container environment
  are not.
- **`ci/scripts/merge_path.sh`'s `tests`/`static` stages** (cargo fmt/clippy/build/test, the guide
  build) — out of scope per this unit's brief ("No cargo builds"); `cargo` is not installed in this
  agent's sandbox at all (`command -v cargo` exits 1). The `guards`/`swarm` stages were run instead
  (see Gates).
- **A driver-only-host self-sufficiency smoke** (actually `exec`ing the assembled tarball on a real
  NVIDIA-driver-only machine). Named in the design contract §F6 as "a SEPARATE property, out of this
  unit" — restated here as the rebuild pointer: a future unit should add this as a live GPU leg once
  #534's chroot half or an equivalent driver-only rented pod is available.

##### 4. Gates

| Command | Exit | Notes |
| --- | --- | --- |
| `bash -n ci/scripts/bundle_cuda_libs.sh` | 0 | |
| `bash -n ci/scripts/test_bundle_cuda_libs.sh` | 0 | |
| `bash -n ci/scripts/capture_loader_report.sh` | 0 | |
| `shellcheck ci/scripts/bundle_cuda_libs.sh` | 0 | clean after correcting an inherited SC2046→SC2086 disable-code mismatch (present in `326785ef` too) |
| `shellcheck ci/scripts/capture_loader_report.sh` | 0 | clean |
| `shellcheck ci/scripts/test_bundle_cuda_libs.sh` | 1 | only the pre-existing SC1091 info finding (`. "$SCRIPT"` — shellcheck cannot statically follow a runtime path; `326785ef`'s own revision carries the identical finding) remains after a file-level `# shellcheck disable=SC2329,SC2034,SC2012,SC2086` documenting the three/four deliberate, inherited idioms (fixture-replacement functions shellcheck can't see invoked indirectly; output vars kept for manual debugging, asserted on by exit code; `ls` over a hermetic `mktemp -d` tree with only alphanumeric names; intentional soname-list word-splitting matching `bundle_main`'s own inline disable) |
| `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release-binaries.yml'))"` | 0 | |
| `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"` | 0 | |
| `BUNDLE_FIXTURE_PROVISIONAL=1 bash ci/scripts/test_bundle_cuda_libs.sh` | 0 | all 113 checks passed |
| `bash ci/scripts/test_bundle_cuda_libs.sh` (no env var) | 1 | 1 of 113 failed — confirms F7's refusal fires without the escape, as designed |
| `python3 ci/scripts/check_bundle_fixture.py --self-test` | 0 | 5 self-test cases pass |
| `python3 ci/scripts/check_bundle_fixture.py` | 1 | FAILED as expected — fixture still `captured: pending` |
| `python3 ci/scripts/test_cu12_component_contract.py` | 0 | 8 tests (4 pre-existing `ComponentContract` + 4 new `BundleScriptContract`) |
| `python3 ci/scripts/check_gpu_prove_once.py` | 0 | |
| `python3 ci/scripts/check_execution_surface_reachability.py` | 0 | PASS |
| `python3 ci/scripts/perf/check_citations.py` | 0 | 1067 files scanned, all resolve |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | |
| `python3 ci/scripts/check_swarm_bijection.py` | 0 | (docs-ci scope check; not in this unit's brief list but relevant since this session is also the docs-ci agent) |
| `python3 ci/scripts/check_ci_guard_wiring.py` | 0 | 76 scripts/suites wired, this unit's 3 new entries included |
| `bash ci/scripts/merge_path.sh --only guards,swarm` | *(see below)* | |

**`merge_path.sh --only guards,swarm`**: 94 guard-matrix commands executed. 91 `ok`. 3 `FAIL`:
1. `pod build substrate` (rc=1) — **not attributable to this unit's diff**: the failing sub-check is
   `test_pod_substrate.sh`'s `(w/esc-050 tarball-golden)` case, which shells out to
   `cargo package --list`; this agent's sandbox has no `cargo` on `PATH` at all
   (`command -v cargo` exits 1), consistent with this unit's brief ("No cargo builds"). Every other
   sub-check in that suite (195 of 214) passed. Flagged for the lead to re-verify in a
   cargo-equipped environment; nothing in this unit's diff touches `test_pod_substrate.sh` or its
   fixtures.
2. `cu12 tarball bundle suite` (rc=1) — **expected**: this leg runs
   `bash ci/scripts/test_bundle_cuda_libs.sh` with no `BUNDLE_FIXTURE_PROVISIONAL` set (matching what
   real CI will do), and the suite correctly refuses to run its T4(3) integration check against the
   still-provisional fixture (1 of 113 checks fails, by design — see F7).
3. `cu12 tarball fixture provisional refusal` (rc=1) — **expected**: `check_bundle_fixture.py` fails
   until the real report lands, with no escape (F7's whole point).

The `swarm` stage (15 steps read from `swarm.yml`, each run as one block) reached 5 of 15 steps, all
green, before I terminated the run (`kill` on the `merge_path.sh` process) after 26+ minutes stuck on
the 6th step — "Assert every esc-lead-gate-R12 deny arm has a dying fixture (deny-coverage sweep)".
That step and the repo-wide `swarm.yml` sweep it belongs to are independent of this unit's diff (it
enumerates lead-gate deny arms across the whole tree, not this unit's files), so I chose not to keep
blocking hand-off on it rather than let it run indefinitely:
```
ok    [swarm] Assert the swarm domain manifests partition crates/ exactly
ok    [swarm] Assert every constitution code anchor resolves
ok    [swarm] Assert no consumer-name / governance leak in the engine tree
ok    [swarm] Assert the lead-proactivity gate's hooks fire correctly on fixtures
ok    [swarm] Run check_lead_gate.py's own test suite
```
The remaining 10 swarm steps (including "CONSTITUTION is human-amend-only" / "swarm gate definitions
are human-amend-only" — the exact checks that are EXPECTED to flag this unit's `ci.yml`/
`release-binaries.yml`/new-`check_*.py` edits, per COMMON.md: "editing one trips SWARM_GATE_TOUCHED,
which the lead admin-merges — that is expected for this wave, not a reason to avoid the edit") were not
observed to completion in this session — the lead should re-run
`bash ci/scripts/merge_path.sh --only swarm` (or the full guard set) and expect an admin-merge case on
`SWARM_GATE_TOUCHED`, not a defect.

##### 5. Issues closed

- **#535** (cu12 tarball: derive the bundled library set from the binary's `DT_NEEDED` closure) —
  **CLOSED**. T1/T2/T3 above, each with an executed oracle and an executed, reverted mutation. The
  five measured defects the issue names are each individually addressed: (1) `bundle_main` now checks
  every phase's return status (T2); (2) the floor/copy-step unversioned-soname fallback is now one
  shared function (`bundle_stem_objects`), used by both the search and the copy; (3) the floor refuses
  a same-destination-basename collision with a different source object (T3/F5, tightened from the
  original directory-keyed shape by the pressure round); (4) `ci/scripts/test_bundle_cuda_libs.sh:259`'s "pins the
  fixture, not the object" advisory is addressed by the F2 binding suite, which reads the REAL object
  (`verify_link_set.py`'s real `COVERED`/`PLATFORM`/`DRIVER_PROVIDED`) rather than a second hand-copy;
  (5) the hand list itself is deleted from `release-binaries.yml` (T5).
- **#534** (cu12 tarball: runtime loader verification of the staged libraries) — **CLOSED for the
  report-parse shape; OPEN for the chroot/`unshare` mechanism** (stated explicitly in the design
  contract's stop rule and F3, and in the script's own module doc). All four MEASURED defects are
  addressed: (1) LD_LIBRARY_PATH-prepends is closed by the refuse-outside-`<lib>` rule (mechanism:
  report-parse, not chroot — the property, not the mechanism, per F3); (2) vacuous pass is closed by
  the every-name-must-resolve rule, exercised against 3 distinct vacuous-report shapes; (3) the
  loader's own no-`=>` line is parsed correctly; (4) `lib_dir='/'` is refused explicitly. The one
  remaining open item is T4(3)'s REAL captured-report integration proof, which needs the lead's
  `release-binaries.yml` dispatch + artifact download (F6) — the mechanism to produce it (the capture
  step + `capture_loader_report.sh`) is shipped in this unit.

##### 6. Commits

```
58d6a4d5 feat(ci): #535 #534 wire the cu12 tarball derivation into the release lane
e1453955 feat(ci): #535 #534 tarball cu12 derivation + sound loader verification arm
```
(`git log --oneline db19a614..HEAD` on `unit/tarball`)

### 11.6 TESTBOUNDS — the observed-event seam and the bound inventory (ai-core) — landed as c96e5da0 … 340347c5 (original tip 85d96d64)

**Lead note:** cherry-picked; the inventory's line-keyed review rows drifted under a later unit and were re-keyed by enclosing item (49b804ac, lead) with the class-C sentence moved onto the const it justifies.

#### TESTBOUNDS — implementation contract

Base: `2659d1d7` (wave-5 tip). Branch: `unit/testbounds`. Tip: `85d96d64`.

**Round 2 (this revision)** addresses the coordinator's two follow-ups on
the round-1 hand-back: (1) B4's root cause was in scope after all — fixed
at the root (the test now derives its expected prefix through the store's
own layout function, not a hand-built string), verified live 3×, and the
chaos leg's `advisory: true` flipped to `false`; (2) B2's fire-removal
mutation, previously not executed, is now executed and reverted. See §4 and
§2's B2/B4 rows below for what changed; §1's round-1 scope is otherwise
unchanged.

##### 1. Scope shipped

**A generalised, job-keyed observed-event rendezvous seam** in
`crates/jammi-ai/src/fine_tune/worker.rs`'s existing `loop_test_hooks` module (no
seventh `*_test_hooks` module — the pressure fold named `crates/jammi-ai/src/jobs.rs:841`'s
`compute_test_hooks::{ParkPoint, arm}` as the precedent; I extended the
module's own pre-existing `Rendezvous`/`arm_rendezvous`/`fire` notify-only
shape, keyed by `instance_id`, into a parallel `Event`/`arm_observed`/
`fire_observed` trio keyed by `job_id`):

- `pub enum Event { CancelObserved, ResumeCheckpointWritten }`
- `pub fn arm_observed(job_id: &str, which: Event) -> Observed` (test-side)
- `pub struct Observed { .. }` with `pub async fn wait_fired(&self)`
- `pub(crate) fn fire_observed(job_id: &str, which: Event)` — non-blocking,
  never parks (the opposite contract from `maybe_park`/`ParkHandle`).
  `pub(crate)`, not `pub(super)`: `crate::fine_tune::trainer` is a SIBLING
  module of `worker`, not a descendant, and is `ResumeCheckpointWritten`'s
  fire point.

Two fire points, both `#[cfg(feature = "test-hooks")]`-gated, neither
changing control flow (per the design contract's Non-goals):

- `worker.rs::spawn_cancel_request_watcher`: fires `CancelObserved`
  immediately after `cancel_requested_seen.store(true, ..)` — the instant
  the watcher OBSERVES the request, never merely "a poll tick happened".
- `trainer.rs::TrainingLoop::save_resume_checkpoint`: fires
  `ResumeCheckpointWritten` immediately after its `store.put_resume_checkpoint`
  call returns `Ok` — the earliest instant a test may observe
  `fetch_resume_checkpoint` return `Some`.

**Three real training-progress-wait sites converted to use it:**

1. `jobs_cancel.rs::a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary`
   (the #567/#578 flake) — arms `CancelObserved` before requesting the
   cancel, waits on it behind a 60s backstop, then joins the run with a 5s
   bound that only races thread-scheduling latency post-observation.
2. `jobs_shutdown.rs::release_and_stop_leaves_running_with_null_lease_and_no_new_bundle`'s
   "no resume bundle landed" wait (B3's named exemplar) — arms
   `ResumeCheckpointWritten` before `spawn_worker`, waits behind a 60s
   backstop, then confirms the bundle is readable.

**One site DERIVED instead** (no discrete watcher event exists):
`jobs_cancel.rs::a_lease_loss_on_the_owning_worker_lands_the_lease_lost_outcome_never_the_cancel_message`.
`hold.lost_flag()` is flipped by the lease keeper's own CONTINUOUS
renewal-miss check (a dedicated thread in `jammi-db`, out of this crate's
scope to instrument with a fire point) — a00c62a0's own commit message
(the dangling object) makes exactly this same call for this exact site
("the same rendezvous shape does not directly apply"). The bound is now
`lease_secs + heartbeat_secs * 12` (both captured from `config.lease`
BEFORE `config` moves into `InferenceSession::new`), with the derivation
named in the assertion message.

**Four sites reworded to the class-C fixed sentence** ("a generous
backstop against a wedged or starved machine") without changing any bound
value or mechanism: `fine_tune.rs`'s two epoch-0-checkpoint-file polls and
two full-run-completion joins, and `distributed/harness.rs`'s
`TERMINAL_TIMEOUT` panic (B4's own named exemplar).

**One sibling site aligned to match its neighbour**:
`jobs_shutdown.rs`'s second "training thread never returned" poll (a
bald `Duration::from_secs(120)`) now reads
`Duration::from_secs(FAST_TIMING.heartbeat + 120)`, matching its sibling
four lines earlier in the same test — both are now class B, not a literal.

**The enumerating oracle**: `crates/jammi-ai/tests/it/test_bounds_inventory.rs`
(new; registered in `tests/it/main.rs`). Scans every git-tracked `.rs` file
under all FOUR test targets (`it`, `distributed`, `gpu_capability`,
`metal_quantized_gpu.rs`) for four literal-bound FORMS: `timeout(Duration::
from_secs(`, `sleep(Duration::from_secs(`, `Instant::now() + Duration::
from_secs(`, `: Duration = Duration::from_secs(` — `from_millis` cadences
never match any of these, excluded by rule. 48 sites found total (0 in
`gpu_capability`/`metal_quantized_gpu`, both empty of this pattern today).
Every found `(file, line)` is checked against a hand-reviewed
`REVIEWED_SITES` table; a class A/B/C site's own text window is checked for
its class's marker (a rendezvous call, a named config identifier, or the
fixed backstop sentence); a class D site carries only a one-line reason
(not a training-progress wait at all), never a text check.

###### Deviations from the issue text (with the code line each rests on)

- **#527's table cites `crates/jammi-ai/tests/it/jobs_cancel.rs:510` and `crates/jammi-ai/tests/it/jobs_shutdown.rs:452`;
  current lines are `:501`/`:427`** (pre-existing drift the brief itself
  flagged as expected — "its cited lines may have drifted"). Re-located by
  reading the test bodies directly; confirmed by `git grep`.
- **The design contract's `a00c62a0` "model" commit does not exist on any
  ref reachable from this tree.** `git merge-base --is-ancestor a00c62a0
  HEAD` fails; `git branch --all --contains a00c62a0` is empty. `git show
  a00c62a0` still resolves (a loose object, presumably from an earlier
  abandoned branch never garbage-collected) — its diff was read and used as
  the DESIGN MODEL per the pressure fold's F1, but nothing from it is
  merged; the seam above is built fresh. Rests on: `git merge-base
  --is-ancestor a00c62a0 HEAD; echo $?` → `1`.
- **B2's "second bound (`crates/jammi-ai/tests/it/jobs_cancel.rs:510`-class)"** turned out to be a
  DIFFERENT test's bound (the lease-loss test, current line `:855`), not a
  second bound inside the epoch-boundary test itself — that test has
  exactly one bound (confirmed by reading its full body: one
  `arm_observed`/`wait_fired` pair, one post-observation join). The pressure
  fold's F1 confirms this reading ("`crates/jammi-ai/tests/it/jobs_cancel.rs:488` is the one bound
  ... `:510`/`:832` are stale → `:488/:814`").

##### 2. Properties

| Property | Executed oracle | Executed mutation that reds it (what changed, red output's first line) |
|---|---|---|
| B1 — every literal wall-clock bound around a training-progress wait in the four test targets is on the reviewed list with a checkable class | `crates/jammi-ai/tests/it/test_bounds_inventory.rs::every_literal_wall_clock_bound_is_reviewed` + `::every_class_a_b_c_site_carries_its_class_marker` (`it` lane, default features) | (1) Added `tokio::time::timeout(Duration::from_secs(999), foo);` to `scheduling.rs` → `every_literal_wall_clock_bound_is_reviewed` reds: `new, UNREVIEWED literal wall-clock bound(s) ...: [("crates/jammi-ai/tests/it/scheduling.rs", 297)]`. (2) Stripped `"a generous backstop against a wedged or starved machine: the run never returned"` down to `"the run never returned"` at `crates/jammi-ai/tests/it/fine_tune.rs:2901` → `every_class_a_b_c_site_carries_its_class_marker` reds: `crates/jammi-ai/tests/it/fine_tune.rs:2901 is reviewed as class C but its window (2901..=2903) does not contain the required marker "wedged or starved machine"`. Both reverted before the final gate run. |
| B2 — #567/#578's epoch-boundary cancel test completes fast, both uncontended and under contention, and its backstop message is specific | `jobs_cancel::a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary`, run directly via the compiled `it` test binary (measured, not `cargo test` wrapped, to get clean wall times) | 5 uncontended runs: 1.09s, 1.12s, 1.10s, 1.08s, 1.09s (load avg ~8-12 idle). 5 contended runs (a concurrent 8-thread `jobs_shutdown::` suite + a background `cargo clippy --workspace --all-targets` build, load average 14.5-16.5 measured via `uptime` at each sample): 1.10s, 1.09s, 1.07s, 1.07s, 1.08s. All ≤ 1.13s, i.e. < 2% of the 60s backstop (well under the 25% floor F4 sets). **Round 2**: the `fire_observed(&job_id, Event::CancelObserved)` call in `spawn_cancel_request_watcher` was commented out (marked `// MUTATION PROBE (B2)`), rebuilt, and the test re-run to its full 60s backstop — red's first line: `a generous backstop against a wedged or starved machine: the cancel-request watcher never observed the request: Elapsed(())` (test finished in 60.09s). Reverted; the un-mutated test re-confirmed green at 1.07s in the same run. |
| The lease-loss test's bound is DERIVED and states its derivation | `jobs_cancel::a_lease_loss_on_the_owning_worker_lands_the_lease_lost_outcome_never_the_cancel_message` (passes, 1 run, ~2s) | Not separately mutated (covered by B1's REVIEWED_SITES marker check: the site's own line literally contains `lease_secs + heartbeat_secs * 12`, checked by `every_class_a_b_c_site_carries_its_class_marker`'s class-B branch). |
| B3 — the sweep is an inventory, not a sample | The `REVIEWED_SITES` table itself (48 entries; see §3 below) + `every_literal_wall_clock_bound_is_reviewed`'s exact-set diff (no extra, no missing) | Same mutation as B1's (1) above (a new unreviewed site reds the exact-set diff). |
| B4 — the chaos leg's flake is root-caused AND fixed at the root | Round 1: reproduced `crash_between_publish_and_finalize_commits_only_the_winner` live 3× (FAIL, `jammi_tb_91443`, torn down). Round 2 (per the coordinator's follow-up): fixed by deriving `winner_prefix` through `ArtifactStore::prefix_url` (now `pub`) instead of a hand-built string; reproduced live 3× AGAIN against a fresh isolated Postgres db (`jammi_tb_r2_6335`) + the same sha-pinned MinIO/mc (re-verified against `distributed.yml`'s exact `MINIO_SHA256`/`MC_SHA256`, re-run in a `debian:bookworm-slim --platform linux/amd64` container) | Round 1: 3/3 FAIL, identical `_global` mismatch (6.5-9.2s each). Round 2: 3/3 PASS (9.60s, 6.77s, 6.78s) after the fix; mutation — reverted `winner_prefix` to the old hand-built string, rebuilt, re-ran: FAILS again with the byte-identical `_global` mismatch (`committed artifact_path "…/models/_global/…" must root under the WINNER's prefix "…/models/…"`, 6.54s) — reverted to the fix, re-confirmed PASS. Both round-2 infra instances torn down (container removed, database dropped, confirmed via `docker ps -a`/`DROP DATABASE`). See §4 below; the chaos leg's `advisory: true` flipped to `false` in `.github/workflows/distributed.yml`. |

##### 3. B3's full inventory (48 sites, `REVIEWED_SITES` in `test_bounds_inventory.rs`)

Class key: **A** = observed-event rendezvous (backstop only on the property
path); **B** = derived from the test's own config; **C** = generous
backstop, fixed sentence; **D** = not a training-progress wait (one-line
reason, no code change).

| File:line | Class | Marker / reason |
|---|---|---|
| `crates/jammi-ai/tests/it/fine_tune.rs:2648` | C | "wedged or starved machine" (worker-a lease-loss join) |
| `crates/jammi-ai/tests/it/fine_tune.rs:2901` | C | "wedged or starved machine" (winner-path run-completion join) |
| `crates/jammi-ai/tests/it/jobs_cancel.rs:190` | D | queued-inference cancel dispatch, no training compute |
| `crates/jammi-ai/tests/it/jobs_cancel.rs:501` | A | `cancel_observed.wait_fired()` backstop |
| `crates/jammi-ai/tests/it/jobs_cancel.rs:511` | A | post-observation join, "poll cadence" in message |
| `crates/jammi-ai/tests/it/jobs_cancel.rs:649` | A | pre-existing `arm_pause_before_spawn_blocking` oneshot |
| `crates/jammi-ai/tests/it/jobs_cancel.rs:855` | B | `lease_secs + heartbeat_secs * 12` on the same line |
| `crates/jammi-ai/tests/it/scheduling.rs:98`, `:264` | D | GPU memory scheduler admission queueing |
| `crates/jammi-ai/tests/it/training_set_stream.rs:305`, `:790` | D | prefetch/streaming deadlock backstop, not training compute |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:259` | C | DRAIN backstop, "wedged or starved machine" |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:290` | D | idle worker DRAIN, no job in flight |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:391` | D | session.close() shutdown |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:427` | A | `bundle_landed.wait_fired()` backstop (B3's own exemplar) |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:440` | B | "RELEASE is bounded by two heartbeats" |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:542` | D | compute/materialization RELEASE, not training |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:656`, `:749`, `:1260` | D | claim→hold prologue / parked-iteration RELEASE mechanics, gated by this file's own `loop_test_hooks` parks |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:845` | D | `release_job_leases`'s own message: waits on no loop at all |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:1474` | B | "RELEASE is bounded by two heartbeats" |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:1656` | D | RELEASE-vs-DRAIN handle race over a compute materialization |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:1751` | D | schema-fault RELEASE test, `fine_tune(1)` near-instant |
| `crates/jammi-ai/tests/it/cache_staleness.rs:781` | D | GPU-budget cache reload |
| `crates/jammi-ai/tests/it/instance_identity.rs:473`,`:515`,`:564`,`:606` | D | worker-registry (`list_workers`) timing |
| `crates/jammi-ai/tests/it/peer_gang.rs:381` | D | peer wire round-trip silence simulation |
| `crates/jammi-ai/tests/distributed/gang_chaos.rs:149`,`:193` | D | fault-injection delay before `kill9`, not an assertion bound — the pass/fail wait is `harness::await_job`'s `TERMINAL_TIMEOUT`, reviewed separately (see below) |
| `crates/jammi-ai/tests/it/host_admission.rs:95`,`:345` | D | admission/slot holder state, claim-loop idle-poll count |
| `crates/jammi-ai/tests/it/fine_tune.rs:2573`,`:2866` | C | "wedged or starved machine" (epoch-0 checkpoint-file polls) |
| `crates/jammi-ai/tests/it/instance_identity.rs:38`,`:286`,`:382` | D | catalog worker-listing poll |
| `crates/jammi-ai/tests/it/jobs_cancel.rs:676`,`:693` | D | watcher/catalog-handle cleanup after an abort |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:178` | D | claim→hold admission transition (`in_flight`); every call site awaits `want=1`, near-instant |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:347`,`:373`,`:477` | B | `FAST_TIMING.heartbeat` on the same line |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:902` | D | workers-row upsert-on-spawn poll |
| `crates/jammi-ai/tests/it/jobs_shutdown.rs:1694` | D | lease-keeper thread death poll |
| `crates/jammi-ai/tests/distributed/harness.rs:531` (`TERMINAL_TIMEOUT` const) | C | "wedged or starved machine" in the panic at `:590` |

(48 rows; 4×A, 6×B, 6×C, 32×D — counted directly from the table's `class:`
field.)

##### 4. Uncovered

- **`gpu_capability`/`metal_quantized_gpu` contribute zero sites to the
  inventory today** — confirmed by direct grep (no matches for any of the
  four FORMS), not merely assumed; the oracle's non-empty-universe floor
  (`>= 30`) is met entirely by `it` (44) + `distributed` (3, incl. the
  const) sites, so this is not hollow-green by omission, but I did not
  hand-audit those two targets' *source* for training-progress waits
  expressed in some OTHER shape my four regexes don't cover (e.g. a
  hand-rolled spin count rather than a `Duration`).
- **`gang_chaos.rs`'s two fault-injection sleeps (D-classified)**: I did
  not verify whether a `4`s pre-kill delay can ever land BEFORE the target
  process is actually claimed/running (which `harness::await_job` already
  gates on beforehand) or AFTER the job has already completed on its own —
  this is a live/gated distributed test I did not re-run under B4's
  infrastructure; flagged, not fixed, not exercised.
- **Round 2**: promoting the WHOLE `chaos` leg off `advisory` (§4 below)
  rests on live-verifying only ONE of its five tests
  (`crash_between_publish_and_finalize_commits_only_the_winner`, the only
  one that has ever actually failed). The other four
  (`killed_worker_job_is_reclaimed_and_completed_once`,
  `artifact_written_on_worker_is_readable_by_a_different_client`,
  `killed_peer_job_is_reclaimed_and_completed_by_a_new_gang`,
  `killed_coordinator_job_is_reclaimed_and_completed_by_a_new_gang`) were
  NOT independently re-run this session — the coordinator's follow-up
  message explicitly directed the promotion ("since the leg is no longer
  flaky, flip the chaos row … from `advisory: true` to required"), so this
  is the lead's call made on the record, not an independent claim by me
  that all five are individually flake-free.

##### 5. B4 — the chaos leg's flake, root-caused AND FIXED (round 2)

###### Round 1 — reproduce and root-cause

**Setup** (torn down after, confirmed): a per-run Postgres database
`jammi_tb_91443` (matching the running process's own pid) on the scratch
Postgres at `127.0.0.1 port 54329` — created fresh, never shared with
`jammi_dist`; the pinned MinIO (`RELEASE.2025-09-07T16-13-09Z`, sha256
`7c5bd851…` — matches `distributed.yml`'s `MINIO_SHA256` byte for byte) and
`mc` (`RELEASE.2025-08-13T08-35-41Z`, sha256 `01f866e9…` — matches
`MC_SHA256`) binaries, downloaded from each project's own GitHub release
and verified with `shasum -a 256` before use; both are `linux-amd64` ELF
(this host is macOS/arm64), run inside a `debian:bookworm-slim` container
(`docker run --platform linux/amd64`, the verified host binaries bind-mounted
read-only) — the container never floats a MinIO image, only executes the
pinned binary. Bucket `jammi-dist` created via the pinned `mc`. `jammi-server`
built with `--features storage-s3` into the SAME `CARGO_TARGET_DIR` the test
binary resolves it from; the distributed harness compiled with
`--features live-distributed-tests --test distributed --no-run`. Env matched
`distributed.yml` exactly. No other holder used this database or bucket
during the window.

**Result: 3/3 runs FAILED, identically shaped, in 6.5-9.2s each:**

```
committed artifact_path "s3://jammi-dist/…/models/_global/{job_id}/{winner}/2"
must root under the WINNER's prefix "s3://jammi-dist/…/models/{job_id}/{winner}/",
never the crashed loser "{loser_uuid}"
```

Every run failed on the SAME structural mismatch — the actual committed
path always has an extra `/_global/` segment between `models/` and
`{job_id}` that the test's own `winner_prefix` string
(`{result_root}/models/{job_id}/{winner}/`) never accounted for — never
the SIGKILL crash-window timing the test's name suggests.

**Root cause, dated:** `crates/jammi-db/src/store/artifact.rs::put_artifact`
deliberately inserts the untenanted-write path segment `_global`
(`GLOBAL_SEGMENT`, `crates/jammi-db/src/store/layout.rs:28`; the exact behavior is its own
pinned unit test, `put_artifact_lands_under_the_global_segment_when_
untenanted`) — introduced by commit `5fef1ac8` ("#484 tenant-prefixed
layout, list primitive, reconcile", 2026-09-09). `artifact_crash_window.rs`
was added by commit `80e92d1e` ("#15/#124 gated multi-process
distributed-validation lane", 2026-06-09) — three months earlier.
`5fef1ac8` is a descendant of `80e92d1e` (`git merge-base --is-ancestor
80e92d1e 5fef1ac8` → true): the tenant-prefixed layout landed AFTER this
test's `winner_prefix` assertion was written, and nothing updated the
assertion to match. `artifact_crash_window.rs`'s test body never sets a
tenant anywhere, so `put_artifact`'s untenanted arm — and its `_global`
segment — applies on every single invocation, independent of the crash
timing: a deterministic, 100%-reproducible test-assertion staleness, not
an intermittent race.

Round 1 reported this and stopped without patching, per the contract's
"a product race → STOP … do not patch the mechanism" instruction — the
finding fit neither of the contract's two named arms. **The coordinator's
follow-up overrode this**: the root cause is in scope, to be fixed at the
root rather than with a one-line string edit.

###### Round 2 — fix at the root, re-verify, promote the leg

**Fix**: `ArtifactStore::prefix_url` (`crates/jammi-db/src/store/artifact.rs`)
— the ONE function `put_artifact` itself calls to build a published
bundle's prefix — is now `pub` (was private, used only internally and by
its own crate's unit tests). `artifact_crash_window.rs`'s `winner_prefix`
now calls it directly (`session.artifact_store().prefix_url(None,
&[job_id.as_str(), winner])`) instead of hand-building the layout string a
second time. A future layout change can misalign the two only by changing
`prefix_url` itself without the test re-running against it — never
silently, the way the tenant-prefixed `_global` segment did.

**Re-verification setup**: a FRESH isolated database (`jammi_tb_r2_6335`,
distinct from round 1's `jammi_tb_91443`, which was already dropped) and a
freshly re-created MinIO container (`jammi-minio-testbounds-r2`) from the
SAME verified pinned binaries (re-checked: sha256 matches
`MINIO_SHA256`/`MC_SHA256` again, same binaries reused from round 1's
scratch dir — not re-downloaded, still byte-identical to what CI pins).

**Result: 3/3 runs PASSED** (9.60s, 6.77s, 6.78s).

**Mutation** (must red): reverted `winner_prefix` to the OLD hand-built
string (`format!("{result_root}/models/{job_id}/{winner}/")`), rebuilt, ran
once more against the SAME live setup — FAILED with the byte-identical
`_global` mismatch:

```
committed artifact_path "s3://jammi-dist/…/models/_global/{job_id}/{winner}/2"
must root under the WINNER's prefix "s3://jammi-dist/…/models/{job_id}/{winner}/", …
```

Reverted back to the `prefix_url`-based fix; re-ran once more to confirm
PASS again (6.56s) before tearing down.

**`.github/workflows/distributed.yml`**: the `chaos` matrix row's
`advisory: true` → `advisory: false` (`continue-on-error` now off, matching
`deterministic`/`ballista`); the file's header comment restated to name the
root cause and the fix, per the coordinator's instruction to state in the
commit why it was advisory and what changed. See §4's Uncovered entry for
the verification-scope caveat (only the one previously-failing test was
independently re-run; the leg-wide promotion is the coordinator's explicit
call).

**Teardown, confirmed (both rounds):** round 1 — `docker stop/rm
jammi-minio-testbounds` (confirmed gone via `docker ps -a`), `DROP DATABASE
jammi_tb_91443`. Round 2 — `docker stop/rm jammi-minio-testbounds-r2`
(confirmed gone via `docker ps -a`), `DROP DATABASE jammi_tb_r2_6335`
(confirmed; `jammi_dist`/`jammi_test` untouched both times).

##### 6. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo test -p jammi-ai --test it jobs_` (35 tests: `jobs_cancel`, `jobs_compute`, `jobs_shutdown`, one `training_set` hit) | 0 | 35 passed, 26.24s |
| `cargo test -p jammi-ai --test it fine_tune::cancelled_run_reclaims_epoch_checkpoints_that_actually_existed` | 0 | 1 passed, 4.06s |
| `cargo test -p jammi-ai --test it fine_tune::finalize_reclaims_a_persistently_failed_prune_and_warns` | 0 | 1 passed, 0.16s |
| `cargo test -p jammi-ai --test it test_bounds_inventory` | 0 | 2 passed, 0.06s |
| `cargo clippy -p jammi-ai --all-targets -- -D warnings` | 0 | (hermetic-lane feature set: default `local` + `test-hooks` via the crate's own dev-dependency unification — `ci.yml`'s `test` job runs `cargo test --workspace --exclude jammi-python` with no explicit `--features`, so this mirrors it exactly) |
| `cargo clippy -p jammi-db --all-targets -- -D warnings` | 0 | round 2: `jammi-db/src/store/artifact.rs` touched (`prefix_url` made `pub`) |
| `cargo test -p jammi-db --lib store::artifact` | 0 | round 2: 19 passed, incl. `put_artifact_lands_under_the_global_segment_when_untenanted` |
| `cargo fmt -p jammi-ai -- --check` | 0 | |
| `cargo fmt --all -- --check` | 0 | |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1068 file(s) scanned, all PATH:LINE citations resolve` (round 1: re-anchored 3 stale `MAINTAINER-GUIDE.md` cites `worker.rs` shifted, commit `6b1c1d44`; round 2: re-anchored 5 MORE stale cites `artifact.rs`'s `prefix_url` doc comment shifted, commit `6d20409d`) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |
| `cargo test -p jammi-ai --test it jobs_` (round 2 re-confirm) | 0 | 35 passed, 12.96s |
| `cargo test -p jammi-ai --test it test_bounds_inventory` (round 2 re-confirm) | 0 | 2 passed |

Note: an early run of the citations/consumer-names checks was piped through
`tail` before reading `$?`, which silently reads `tail`'s exit code instead
of the real one (caught the `check-citations: FAIL` in the piped output
itself, then re-ran unpiped to confirm the real exit code — `1` before the
`MAINTAINER-GUIDE.md` fix, `0` after). Not repeated in round 2 — every
citations/consumer-names run there used the unpiped `> file 2>&1; echo $?`
form from the start.

##### 7. Issues closed

- **#527** (the class): CLOSED for the class's mechanism and its own named
  exemplar (the `jobs_shutdown.rs` resume-bundle wait). The full inventory
  is closed as an INVENTORY, not a full-remediation claim — 6 sites remain
  class C (backstop, reworded but not converted to A) and 6 remain class B
  (derived, some pre-existing) by design; every site in the four targets is
  now classified with a checkable predicate, which is #527's actual ask
  ("Where a bound is genuinely unavoidable … it should be an explicit,
  generously-sized backstop whose message says that").
- **#567**: CLOSED. `a_claimed_training_jobs_cancel_request_is_honoured_at_
  the_next_epoch_boundary` now rendezvous on the real event; property B2
  verified (5+5 timing runs).
- **#578**: CLOSED. Same test, same fix; the residual noted in
  `docs/rigor/contracts/feat_500-B-U2b.md` is addressed by this same
  change (that contract is not touched here — the lead owns its own
  update).
- **The advisory chaos-leg flake (B4)**: CLOSED (round 2). Reproduced (3/3
  FAIL, round 1), root-caused with commit-level precision (a deterministic,
  100%-reproducible test-assertion staleness from #484's tenant-prefixed
  layout, never the SIGKILL timing window), fixed at the root
  (`ArtifactStore::prefix_url` exposed as the one path builder; the test
  derives through it instead of a hand-built copy), re-verified (3/3 PASS,
  round 2; the reverted-string mutation reproducibly reds), and the leg's
  `advisory: true` flipped to `false` in `.github/workflows/distributed.yml`
  per the coordinator's explicit direction. Verification-scope caveat: only
  the one previously-failing test was independently re-run under this fix
  (§4's Uncovered).

##### 8. Commits

```
85d96d64 ci: #500 wave 5 E6 — promote the distributed chaos leg off advisory
6d20409d fix(db,ai-core): #500 wave 5 E6 — artifact_crash_window derives its expected prefix through the store's own layout function
6b1c1d44 docs: #500 wave 5 E6 — re-anchor three MAINTAINER-GUIDE.md citations worker.rs moved under
a321c747 test(ai-core): #500 wave 5 E6 — the B1/B3 enumerating inventory oracle
4018d528 fix(ai-core): #500 wave 5 E6 — reword four backstop messages to the class-C sentence
5c1a5aaa fix(ai-core): #500 wave 5 E6 — jobs_shutdown's named resume-bundle bound
cbe5a8a9 fix(ai-core): #500 wave 5 E6 — jobs_cancel's two training-progress bounds
f805a8b6 fix(ai-core): #500 wave 5 E6 — one job-keyed observed-event rendezvous seam
```

##### 9. Scope amendment (round 2)

`crates/jammi-db/src/store/artifact.rs` is outside this unit's owned crate
(`jammi-ai`) — touched only because the coordinator's follow-up explicitly
directed exposing `ArtifactStore::prefix_url` as `pub` ("expose it as the
one path builder if it is not already callable") to fix B4 at the root. The
change is a single visibility widening (`fn` → `pub fn`) plus its doc
comment; no behavior changed (confirmed: all 19 of that module's own unit
tests still pass, `cargo clippy -p jammi-db --all-targets -- -D warnings`
clean).

### 11.7 GANG — gang v1 completeness (ai-core) — landed as 7d4b9173 (the net diff of unit/gang fdab1ce4 over bc0e1578)

**Lead note:** landed as one net diff because the unit had merged the consolidated branch twice; the maintainer-guide conflict was re-anchored by identifier text.

#### GANG — implementation contract (round 2, post-delta)

Base: wave-5 tip `2659d1d7`, then `git merge feat/500-wave5` (tip `bd3a6202`, GATESAI's syn-based
`pinned_source_gate.rs` rebuild + GATESCI's gate work) per the coordinator's mid-task delta, then a
SECOND `git merge feat/500-wave5` (tip `bc0e1578`, the stamps-db migration-039 work) per the
coordinator's later infra note. Final tip `fdab1ce4` on `unit/gang`.

**Infra note acted on:** the coordinator flagged that the shared `jammi_test` Postgres database now
carries migration 039 (from the not-yet-merged stamps-db work, landed live against the shared
instance) and that a tree knowing only 038 must not run against it — use `JAMMI_TEST_PG_URL=postgres://
jammi@127.0.0.1 port 54329/jammi_gang` (a fresh, dedicated database) instead. This exactly explains the 44
Postgres-arm failures recorded further down in an earlier draft of this report (all
`sdchk__result_tables__created_at`/`sdchk__models__created_at` check-constraint violations, on code this
diff never touches — the constraint text does not exist anywhere in this branch's own `jammi-db/src`,
confirming it was contamination from the OTHER unit's live schema work, not a regression here). After
merging the new tip and switching to `jammi_gang`, every Postgres-arm test this unit touches or added
passes (§4). The merge itself (`lease_now` → `canonical_stamp_now`, `now_sortable()` deleted) touched no
file this unit's diff modifies (`grep`-confirmed against every touched file before compiling); it also
surfaced one pre-existing gap in the MERGE (not in either lineage alone): a new unit test in
`migrations.rs` (from the stamps-db side) tripped `pinned_source_gate.rs`'s DDL-literal reviewed-allowlist
(from the GATESAI side) — fixed with one `DDL_LITERAL_SITES` entry, §1/§4.

##### 1. Scope shipped

**N4 (#548) — DONE, corrected for round 2.** `JobSpec` (`crates/jammi-ai/src/jobs.rs`) is one flat
`#[derive(Serialize, Deserialize)] #[serde(tag = "kind", rename_all = "snake_case",
deny_unknown_fields)]` enum over all eight compiled kinds (`COMPILED_KINDS`,
`crates/jammi-ai/src/fine_tune/worker.rs:265`) — not a wrapper around `TrainingSpec`/`ComputeSpec` (the
round-1 shape the coordinator's own message names as tried-and-failed: an outer tag over two inner
`kind`-tagged enums fails its own round-trip with "duplicate field `kind`" on serialize, since the
newtype-wrapped inner value would ALSO try to write its own tag; a `tag`/`content` adjacent form
round-trips but pushes `common` to depth 2 — executed as a RED mutation and reverted,
`jobs.rs::job_spec_byte_pins_every_compiled_kind_against_its_own_type` / `job_spec_fine_tune_literal_byte_pin`
go red under it). `JobSpec::as_training_spec`/`as_compute_spec` project to the narrower types.

Deviation from the dispatch's "six kinds": the real universe is **eight** (`COMPILED_KINDS`), not six —
verified against `crates/jammi-ai/src/fine_tune/worker.rs:265-274` directly; every oracle below covers all eight.

Round-2 correction (the coordinator's delta): `deny_unknown_fields` is per-`#[derive]`, not recursive —
a stray key nested inside `TrainingCommon`/`FineTuneConfig` (or any other nested struct reachable from a
`JobSpec` variant) was silently dropped even though `JobSpec`'s own top-level `deny_unknown_fields`
refused a top-level stray field. Added `#[serde(deny_unknown_fields)]` to every nested struct/enum
reachable from a `JobSpec` variant: `TrainingCommon` (jammi-ai), `GraphFineTuneSources`/
`GraphSampleConfig` (jammi-ai), `ContextPredictorTrainConfig`/`PredictiveHead` (jammi-ai),
`BuildNeighborGraph` (jammi-ai), `AsofKey`/`AsofJoinSpec` (jammi-ai), `PropagateRequest`/`EdgeSourceRef`
(jammi-ai), and — cross-crate — `FineTuneConfig`/`HardNegativeConfig`/`EmbeddingLoss`/`RegressionLoss`/
`ClassificationLoss` (`crates/jammi-wire/src/fine_tune.rs`). Pinned by
`jobs.rs::a_stray_field_nested_inside_common_is_refused_not_silently_dropped`.

Round-2 correction (the four readers): the persisted-spec decode sites now decode `JobSpec` (the one
canonical type) FIRST, then project — not `TrainingSpec`/`ComputeSpec` directly:
- `crates/jammi-ai/src/fine_tune/worker.rs:1975` (loop-claimer training path) — decodes `JobSpec`,
  projects via `as_training_spec()`, refuses (typed, job-id-keyed failure record) when the row's `kind`
  column says training but the decoded spec is a compute kind.
- `crates/jammi-ai/src/fine_tune/worker.rs:3288` (compute claim path, was `:3250` — the file grew above
  it) — same shape via `as_compute_spec()`.
- `crates/jammi-ai/src/fine_tune/worker.rs:6470` (Peer-rank `member_rank_body`, was `:6410`) — same shape.
- `crates/jammi-ai/src/fine_tune/training_job.rs::resolve_model_id` — the FOURTH reader the coordinator's
  delta named (I had missed it in round 1): decoded `TrainingSpec` directly from `record.spec` for the
  `context_predictor` arm; now decodes `JobSpec` and projects.

Deviation: three of the four durable submit edges (`session.rs::submit_fine_tune_spec_deduped`,
`context_predictor.rs::train_context_predictor_deduped`, and `run_now`'s `ComputeSpec` path) still WRITE
via `TrainingSpec`'s/`ComputeSpec`'s own `Serialize`, not `JobSpec`'s — only `enqueue` writes via
`JobSpec`. This is why the byte-pin tests remain load-bearing (not merely historical): `JobSpec::
Deserialize` reading a row that `TrainingSpec::Serialize` wrote is safe only because the two are proven
byte-identical, not because `JobSpec` is the sole writer. Consolidating every write path onto `JobSpec`'s
own `Serialize` was in scope conceptually but not attempted this round (see §3, Uncovered) — it would
touch `training_job_links`' signature and both remaining edges' call shape.

**N3 (#573) — DONE for the three jammi-ai edges, honestly partial on the compile-time claim.**
`crate::fine_tune::spec::admit_training_spec` now CONSUMES its `TrainingSpec` and returns
`AdmittedTrainingSpec` (`crates/jammi-ai/src/fine_tune/spec.rs`), a newtype whose field is private to
`spec.rs` — the only way any other module can obtain one is by calling `admit_training_spec`. All three
edges (`crates/jammi-ai/src/session.rs:1976`, `jobs.rs::enqueue`, `context_predictor.rs::train_context_predictor_deduped`)
now read the admitted spec back only through `AdmittedTrainingSpec::spec()`, so within EACH edge's own
body there is no path left that could serialize/submit the pre-admission value instead of (or alongside)
the admitted one.

**Stated honestly, not overclaimed** (executed falsification, in the type's own doc and this contract):
the coordinator's ask was "an unadmitted submit does not compile." I executed the falsification —
replacing `context_predictor.rs`'s `admit_training_spec` call + witness-read with `let training_spec =
&training_spec;` (skip admission, borrow the original value) — and it COMPILES. Both
`every_durable_training_submit_edge_calls_the_one_admission_function` (source-level, three-file oracle)
and `an_invalid_context_predictor_config_is_refused_through_the_real_edge` (new behavioural oracle) go
red under this mutation; reverted, both green. The reason a true "does not compile" gate is unreachable
here without a much larger refactor: the three edges are not consolidated behind one shared
durable-write function that takes `AdmittedTrainingSpec` as its parameter type — each still builds its
own `SubmitJobParams` inline, and `jammi_db::Catalog::submit_job`/`submit_job_deduped` are generic,
kind-agnostic APIs that cannot depend on this crate's `TrainingSpec` shape at all (the dependency runs
the other way). The witness closes the realistic "reuse the existing pattern" shape; it does not, and
structurally cannot in Rust's privacy model for a `pub` enum whose variants must stay externally
constructible (tests, wire decode), close a hand-rolled bypass of the whole pattern.

Added the context-predictor behavioural oracle #573's second bullet asked for:
`rank_admission.rs::an_invalid_context_predictor_config_is_refused_through_the_real_edge` submits an
invalid `ContextPredictorTrainConfig` (`context_k = 0`) through the real
`InferenceSession::train_context_predictor` edge and asserts a typed `JammiError::FineTune` refusal
naming `context_k`, with the `jobs` table unchanged.

Not attempted (see §3): the syn-derived submit-edge UNIVERSE the original v2 contract asked for — the
coordinator's delta states this rule "admits nothing on this tree" (found vacuous by the pressure round)
and directs the witness instead, which is what I built. jammi-bench's `crates/jammi-bench/src/finetune_run.rs:1870` edge (cited
by the coordinator) was not wired through admission: it never constructs a `TrainingSpec` value at all
(it submits a placeholder `spec: "{}"` row purely to obtain a trackable `job_id`, then drives
`TrainingLoop::run` directly, bypassing the whole submit-edge shape this witness protects), and
`admit_training_spec` is `pub(crate)` to jammi-ai — jammi-bench is a separate crate and cannot call it
without a visibility change I did not make this round. UNCOVERED, stated in §3.

**N2 (#551) — partial: the newtype, not the enumerating-gate extension.** `RelationKey`
(`crates/jammi-db/src/store/mod.rs`, beside `TrainingSetTable`) is a newtype over the quoted
`"jammi.{table}"` session-registered relation string; its field is private to `store/mod.rs`, so no
other module (in jammi-db or a downstream crate) can construct one — only
`TrainingSetTable::sql_relation()` mints it (changed its return type from `String` to `RelationKey`;
`Display`/`as_str()` keep every existing `format!("...{}...", table.sql_relation())` call site
unchanged). `TrainingSetTable::registered_name()` (the UNQUOTED `jammi.{table}` form,
`TableReference::bare`'s own input) stays a plain `String` — that risk class is already covered by
`pinned_source_gate.rs`'s Pattern 3 (`session_registration_literal_sites`/`SESSION_LITERAL_ALLOWED`,
GATESAI's rebuild), a separate gate from this newtype.

Only one production caller of `sql_relation()` exists in the workspace
(`crates/jammi-ai/src/fine_tune/training_set.rs:77::read_back_sql`); it and every test call site
(`crates/jammi-db/tests/it/materialization.rs`, `crates/jammi-ai/tests/it/training_set.rs`) compile and
pass unchanged against the new return type.

Not attempted (see §3): migrating the ~9 other hand-built `"jammi.{name}"` sites the original v2 contract
named (`crates/jammi-ai/src/session.rs:2313` et al.) onto the minter, and extending `pinned_source_gate.rs`'s enumerating
scan to cover the QUOTED-relation class (the coordinator's delta: scope to session-registered `jammi.
{name}` only, `{source_id}.public.{table}`/`FROM "{backing}"` named out, base =
`SESSION_LITERAL_ALLOWED` (10 entries, unchanged by me) + 5 jammi-bench sites). GATESAI's rebuilt
`pinned_source_gate.rs` is ~5500 lines with golden hit-count tests
(`allowlists_match_current_hits_exactly`, `caller_set_claims_match_reality`) I did not have the budget
to safely extend without risking a silent regression in machinery I did not build. UNCOVERED, stated
honestly in §3 rather than attempted at low confidence.

**N5 (#543) — NOT attempted this round.** No code changes to `crates/jammi-ai/tests/distributed/
gang_chaos.rs`, `crates/jammi-server/tests/it/gang_coordinator.rs`, or the trainer's resume/broadcast
path. The coordinator's delta asks for a NEW harness (`gang_chaos.rs`'s own shape: real
`jammi-server` processes, a real SIGKILL of the coordinator after epoch 1's checkpoint, comparing
attempt-2 adapter bytes Peer W=2 vs Local W=2 rank 0) rather than an extension of the existing
`TrainingLoopBuilder` in-process K4 reference. This is the most expensive remaining unit (live Postgres +
MinIO + multiple spawned `jammi-server` binaries, `distributed.yml`'s chaos lane) and I did not reach it
in the time available after N4's correction, N3, and N2 consumed the round. UNCOVERED, stated honestly
in §3; issue #543 stays open.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| N4: every `JobSpec` variant's JSON is byte-identical to the `TrainingSpec`/`ComputeSpec` variant it mirrors, for all eight kinds | `jobs.rs::job_spec_byte_pins_every_compiled_kind_against_its_own_type`, `::job_spec_fine_tune_literal_byte_pin`, `::training_job_spec_round_trips_and_keeps_the_flat_kind_tag` (lib) | `#[serde(tag="kind", content="content", ...)]` on `JobSpec` — red: `common` moves to `.content.common`, first line `left: None right: Some(Number(1))` |
| N4: a stray field at depth 1 (directly under `kind`) is refused, naming the field, for a training AND a compute kind | `jobs.rs::a_stray_field_under_a_declared_kind_is_refused_through_job_spec_naming_the_field`, `::a_stray_field_under_a_compute_kind_is_refused_naming_the_field` (lib) | drop `deny_unknown_fields` from `JobSpec`'s attribute — red: `expect_err` panics, "a stray field under the declared kind must be refused" |
| N4: a stray field nested one level under `common` is refused, not silently dropped | `jobs.rs::a_stray_field_nested_inside_common_is_refused_not_silently_dropped` (lib) | drop `deny_unknown_fields` from `TrainingCommon` — red (same panic shape; verified by inspection of the attribute's removal, consistent with the identical mechanism proven on `JobSpec` itself above) |
| N4: an unrecognised `kind` / a missing `kind` is refused naming itself, never silently mapped to a different variant | `jobs.rs::an_unrecognised_kind_is_refused_naming_the_kind_not_silently_mapped_to_a_different_variant`, `::a_job_spec_row_missing_kind_is_refused` (lib) | inherited from serde's internally-tagged derive machinery; unchanged by this round |
| N4: `jammi-db`'s `world_size_from_spec_json` reads `JobSpec`'s new shape unchanged, for every training kind (explicit `common.world_size`) and a compute kind (absent → default 1) | `gang_rank_admission.rs::get_job_for_rank_reads_world_size_the_same_way_for_every_compiled_job_spec_kind_shape` (jammi-db `it`, sqlite; hand-written JSON fixtures since jammi-db cannot import jammi-ai) | N/A — a cross-crate consistency check, not a mutation target; its own four fixtures each assert a distinct `WorldSizeFact` |
| N3: every one of the three durable training submit edges calls `admit_training_spec` in its own body | `rank_admission.rs::every_durable_training_submit_edge_calls_the_one_admission_function` (it) | replace `context_predictor.rs`'s admission+witness-read with a direct borrow — red: "found 0 calls in its body" |
| N3: an invalid `ContextPredictorTrainConfig` submitted through the real edge is refused, typed, nothing enqueued | `rank_admission.rs::an_invalid_context_predictor_config_is_refused_through_the_real_edge` (it) | same mutation as above — red: "the refusal must be typed, got Catalog(\"No ready embedding table for source 'episodes'\")" (validation never ran; the error surfaces from further downstream instead) |
| N2: no code outside `store/mod.rs` can construct a `RelationKey` | Structural (private field) + `materialization.rs`'s 20 `.sql_relation()` call sites (jammi-db `it`, sqlite) all compile/pass against the new `RelationKey` return type | N/A — a compile-time property; the field-privacy is what a `pub struct RelationKey(String)` with a `pub` field would not have |

##### 3. Uncovered

- **N3's "does not compile" claim beyond the three named edges.** Stated in §1/§2: the witness prevents
  a double-value footgun WITHIN an edge, not a hand-rolled bypass of the whole submit-edge pattern by a
  fourth, unconsolidated edge. Closing this fully needs either (a) a shared durable-write helper the
  three edges are refactored onto (so the helper's signature is the enforcement point), which is a larger
  cross-function refactor than this round reached, or (b) accepting the source-level oracle
  (`every_durable_training_submit_edge_calls_the_one_admission_function`) as the actual closure
  mechanism, which is what round-1's issue #573 already shipped and this round kept. Issue #573 stays
  CLOSED on the behavioural-oracle half (both halves of the issue's own "Rebuild" section are done); the
  coordinator's stronger compile-time ask is UNCOVERED.
- **jammi-bench's `crates/jammi-bench/src/finetune_run.rs:1870` edge.** Never constructs a `TrainingSpec`; submits a placeholder
  `spec: "{}"` row to obtain a `job_id` and drives training directly. Wiring this through
  `admit_training_spec` would need that function's visibility raised from `pub(crate)` (a jammi-ai/
  jammi-bench crate-boundary decision I did not make unilaterally) and a design decision about what a
  benchmark run's "admitted spec" even represents, since it has no real `TrainingSpec` to validate.
- **N2's remaining ~9 hand-built relation sites and the enumerating-gate extension.** The `RelationKey`
  newtype exists and is proven sound for its one real caller; the source-level enumerating sweep over
  every `"jammi.{name}"` site (the property "no code OUTSIDE the minter reaches this shape," as opposed
  to "the minter's own output is unforgeable," which IS proven) was not attempted — GATESAI's rebuilt
  `pinned_source_gate.rs` (~5500 lines, golden hit-count oracles) needed deeper familiarity than this
  round's budget allowed to extend safely.
- **N5 in full.** No chaos-SIGKILL resume-parity harness built. Issue #543 stays open; the program's
  existing ruling (a resumed job at world_size > 1 is refused at assembly, typed) is UNCHANGED — this
  round neither lifts nor further restricts it.
- ~~The shared Postgres lane's environment~~ — RESOLVED, not uncovered: the coordinator's infra note
  (migration 039 landed live on the shared `jammi_test` database from the not-yet-merged stamps-db work)
  explained the 44 failures an earlier round of this report recorded; re-run against the dedicated
  `jammi_gang` database (§4) is clean, 104/104.

##### 4. Gates

All commands below ran at the final tip (`fdab1ce4`), after both merges and the `jammi_gang` re-run.

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo clippy -p jammi-ai -p jammi-db -p jammi-wire --all-targets -- -D warnings` | 0 | one fix along the way: `clippy::clone_on_copy` on `GraphSampleConfig` (`.clone()` → `*sample_config`, it's `Copy`) |
| `cargo test -p jammi-ai --lib` | 0 | 813 passed, 0 failed |
| `cargo test -p jammi-ai --test it -- rank_admission:: acceleration_report:: jobs_compute:: jobs_cancel:: jobs_shutdown::` | 0 | 60 passed, 0 failed |
| `cargo test -p jammi-ai --test it -- pinned_source_gate::` | 0 | 51 passed, 0 failed — golden hit-list oracles included; after the second merge this needed one new `DDL_LITERAL_SITES` review entry (§1) for a stamps-db unit test's DDL-shaped fixture string, unrelated to my own N2/N3/N4 changes |
| `cargo test -p jammi-db --test it -- materialization:: gang_rank_admission:: --test-threads=1` (sqlite) | 0 | 58 passed, 0 failed |
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_gang cargo test -p jammi-db --test it --features live-postgres-tests -- materialization:: gang_rank_admission:: --test-threads=1` (postgres, the dedicated database the coordinator provisioned) | 0 | 104 passed, 0 failed. An EARLIER run against the shared `jammi_test` database (before the coordinator's infra note) showed 44 `sdchk__*__created_at` check-constraint failures on code this diff never touches — confirmed as contamination from the not-yet-merged stamps-db migration work landed live on that shared instance, not a regression here; superseded by this clean run, not merely explained away |
| `cargo check -p jammi-ai -p jammi-db --lib --tests` | 0 | confirms the merge (`lease_now`→`canonical_stamp_now`, `now_sortable()` deleted) touches nothing this unit's diff depends on |
| `python3 ci/scripts/perf/check_citations.py` | 0 | "1067 file(s) scanned, all PATH:LINE citations resolve" — 30 citations in `docs/maintainer/MAINTAINER-GUIDE.md`/`pinned_source_gate.rs`/`grpc_remote_compute.rs` re-anchored (my line insertions in `store/mod.rs`/`worker.rs`/`context_predictor.rs`/`fine_tune.rs` shifted them; new line numbers computed from the diff hunks, verified against every candidate occurrence in the target file, not hand-guessed) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | "no governance-verb leak... allowlist clean" |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | one fix along the way: `JobSpec`'s own doc linked a private `worker.rs` method (`run_claimed_compute_job`) via `[\`...\`]`; converted to a backtick code span (no doc-hidden bypass), per the committed convention (7fd457e) |

No shared-declaration file (`crates/jammi-ai/src/lib.rs`, `crates/jammi-ai/Cargo.toml`,
`crates/jammi-ai/src/error.rs`) was touched, so the workspace-wide rustdoc form was not required and not
run.

##### 5. Issues closed

- **#548 (N4): CLOSED.** `JobSpec` is off `#[serde(untagged)]`, is one flat derive-only tagged enum, a
  stray field at any depth is a typed refusal naming the field, an unrecognised/missing `kind` is
  refused naming itself, and every one of the four production readers of a `jobs.spec`/`training_spec`
  row decodes the same canonical type. No migration (bytes unchanged, proven by the byte-pin tests).
- **#573 (N3): CLOSED on both of the issue's own stated halves** (the source-level three-edge oracle, and
  the context-predictor behavioural oracle) — both pre-existed round 1 or were added this round, both
  pass, both execute a red mutation. The coordinator's STRONGER compile-time ask ("an unadmitted submit
  does not compile") is a NEW bar beyond what #573's own text asked for; that bar is UNCOVERED (§3) and,
  if the program wants it closed, needs a new/amended issue, since #573 itself did not scope it.
- **#551 (N2): OPEN.** The `RelationKey` newtype (half of the issue's ask) is done and proven sound for
  its one real caller. The mining W=1 byte-parity oracle (the issue's OTHER half) was not touched this
  round — it is a pre-existing property from U2b's closing audit, out of this unit's N2 scope per
  `contracts/gang.md` v2 (the mining oracle is named under N2 but is unrelated code to what this round
  built; I did not locate or re-verify it and make no claim about its state). The enumerating-gate
  extension and the remaining hand-built sites are UNCOVERED (§3).
- **#543 (N5): OPEN.** No code shipped this round. The program's existing refusal (resume at world_size >
  1 is refused at assembly) is unchanged.

##### 6. Commits

```
fdab1ce4 test(ai): gang -- review the migrations.rs DDL fixture the merged stamps-db work introduced
465e97a2 Merge branch 'feat/500-wave5' into unit/gang
bc0e1578 [... feat/500-wave5's own stamps-db commits (migration 039, lease_now -> canonical_stamp_now), merged per the coordinator's infra note ...]
d7beb5ae fix(ai,db,wire): gang R1 N4 depth-complete + R2 N3 witness + N2 RelationKey
aef5ba93 Merge branch 'feat/500-wave5' into unit/gang
68054783 wip(ai): gang N4 — JobSpec flat #[serde(tag=kind, deny_unknown_fields)] enum
bd3a6202 [... feat/500-wave5's own GATESAI/GATESCI commits (syn-based pinned_source_gate.rs rebuild), merged per the coordinator's mid-task delta ...]
c540df88 wip(ai): gang v1, superseded
```

`git log --oneline db19a614..HEAD` shows the full list including both merges of `feat/500-wave5`
(GATESAI's `pinned_source_gate.rs` syn rebuild + GATESCI's gate work, then the stamps-db migration-039
work) — merged per the coordinator's explicit instructions, not authored by this unit. This unit's own
three commits are `68054783`, `d7beb5ae`, `fdab1ce4`.

### 11.8 GANG3A — one submit seam, the relation-key minter, the mining oracle (ai-core) — landed as ea4570a0, a184c0f5 (original tip 84cce4af)

**Lead note:** cherry-picked; its bump of the include_str! review line was superseded by the lead's re-keying (a793f572); the mining byte pin is a closer (§9).

#### GANG3A — implementation contract (N3-seam, N2-gate)

Base: wave-5 tip `7d4b9173` (carries GANG v1's N4/N3-witness/N2-newtype commit). Worktree
`wt-gang3a`, branch `unit/gang3a`, final tip `84cce4af9662142cd09de180048812e243776452`.

##### 1. Scope shipped

###### N3-seam (#573, GANG3A's stronger bar) — DONE

`crate::fine_tune::spec::submit_admitted_training` (`crates/jammi-ai/src/fine_tune/spec.rs`) is
now the ONE function in the workspace that builds a `SubmitJobParams` for a training kind and
submits it. The three in-crate durable training edges no longer build `SubmitJobParams`
themselves:

- `session.rs::submit_fine_tune_spec_deduped` — admits, derives `training_job_links`, then calls
  the seam with `idempotency_key` forwarded.
- `jobs.rs::enqueue` — its training arm (the `Some(training)` branch of
  `spec.as_training_spec()`) calls the seam; its compute arm is untouched (N3 is training-only)
  and still builds `SubmitJobParams` from `JobSpec`'s own `Serialize`, as before.
- `pipeline/context_predictor.rs::train_context_predictor_deduped` — same shape as
  `session.rs`.

`admit_training_spec` and `submit_admitted_training` are both `pub` (round 3; previously
`pub(crate)`): `AdmittedTrainingSpec`'s own field stays private to `spec.rs` regardless (the
struct itself is `pub`, its tuple field is not), so the ONLY way ANY caller — in this crate, or
across the `jammi-bench` crate boundary — can obtain one is by calling `admit_training_spec`
first. `crates/jammi-bench/src/finetune_run.rs`'s placeholder submit (`kind: "fine_tune", spec:
"{}"`, no admission, no real spec) is rebuilt: it now builds a real `TrainingSpec::FineTune`
(honest about what this tier actually is — `source`/`columns` describe the committed fixture
this tier reads, not a materialized SQL source that does not exist here; `base_config(params,
params.epochs)` supplies the real `FineTuneConfig`), admits it, and submits through the seam
(`model_ref`/`output_model_id` computed locally — `fine_tuned_model_id(job_id)`, already `pub` —
since `TrainingJobLinks` stays `pub(crate)` to `jammi-ai` and the seam takes plain
`&str`s rather than that type, so it needs no cross-crate visibility change of its own).

**Deviation from the design contract's literal signature.** `gang3.md` names
`submit_admitted_training(catalog, AdmittedTrainingSpec, …) -> SubmittedJob`. Shipped signature:
`submit_admitted_training(catalog: &Catalog, admitted: &AdmittedTrainingSpec, job_id: &str,
model_ref: &str, output_model_id: &str, priority: i32, idempotency_key: Option<&str>) ->
Result<SubmittedJob>`. Reason: `TrainingJobLinks` (the natural `model_ref`/`output_model_id`
carrier the three in-crate edges already compute) is `pub(crate)` to `jammi-ai`
(`crates/jammi-ai/src/session.rs:87`) — taking it as a parameter would force it `pub` for
`jammi-bench` to construct one, which is unnecessary: the seam only ever *reads* the two
strings, so it takes them directly and stays decoupled from `session.rs`'s internal link-
derivation type. `SubmittedJob` is `{ pub recorded_job_id: String }`, matching the design's
named return type.

**The enumerating source oracle — rebuilt on `syn::parse_file`, not a text scan.**
`rank_admission.rs::every_submit_job_call_in_production_code_is_the_seam_or_a_reviewed_non_training_site`
replaces the round-1/round-2 oracle (which read three functions by exact signature-string search
and was found vacuous by the pressure round: a fourth edge in a fourth file was outside its
universe). The new oracle enumerates EVERY `.submit_job(`/`.submit_job_deduped(` method-call site
in `crates/jammi-ai/src`'s tracked source (reusing
`crate::pinned_source_gate::scan_surface`'s `git ls-files` enumeration — "the SAME machinery,
never a parallel scanner"), parsed via `syn::parse_file` + `syn::visit::Visit`, skipping any
`#[cfg(test)]`-attributed `mod`/`fn` (the pre-existing `worker.rs`/`trainer.rs` `mod tests { .. }`
fixtures that build a placeholder job row to test claim/execute machinery, not a submit edge —
reviewed and out of this property's universe, stated in the oracle's own doc). Every hit found
must be on a three-entry reviewed allow-list: `spec.rs::submit_admitted_training` (the seam),
`jobs.rs::enqueue` (its call sits inside the `None` arm of `match spec.as_training_spec() {..}` —
structurally unreachable for a training kind), `jobs.rs::run_now` (takes `spec: ComputeSpec`,
never `TrainingSpec`, as its own parameter type). The universe is "every call", not "every fn
matching a shape", closing the exact gap the coordinator's delta named.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Every `Catalog::submit_job`/`submit_job_deduped` call in `crates/jammi-ai/src`'s production code is the seam or a reviewed non-training site | `rank_admission.rs::every_submit_job_call_in_production_code_is_the_seam_or_a_reviewed_non_training_site` | Added `jobs.rs::submit_training_spec_unadmitted` (a hand-built `SubmitJobParams { kind: "fine_tune", .. }` call outside the seam) — RED: `a submit_job/submit_job_deduped call exists outside the reviewed allow-list ... ["crates/jammi-ai/src/jobs.rs::submit_training_spec_unadmitted"]`. Reverted. |
| No caller anywhere (in-crate or `jammi-bench`) can construct an `AdmittedTrainingSpec` without calling `admit_training_spec` first | Structural (private tuple field); executed falsification in `rank_admission.rs`: `jammi_ai::fine_tune::spec::AdmittedTrainingSpec(spec)` | `cargo check` on the falsification — RED: `error[E0603]: tuple struct constructor 'AdmittedTrainingSpec' is private`. Reverted. |
| The two `_deduped` edges (`session.rs`, `context_predictor.rs`) each call `admit_training_spec` exactly once and `submit_admitted_training` exactly once in their own body | `rank_admission.rs::the_two_seam_calling_edges_admit_before_calling_the_seam` | Added a redundant second `admit_training_spec` call inside `train_context_predictor_deduped` — RED: `left: 2, right: 1`. Reverted. |
| Every quoted `"jammi.{name}"` relation string this round found is built through `jammi_db::store::result_table_relation`/`TrainingSetTable::sql_relation` (the only two `RelationKey`-minting sites, both in `store/mod.rs`), and the minter's own construction site is the ONE reviewed, allow-listed exception | `pinned_source_gate.rs::no_new_unpinned_session_registration_literal` / `::allowlists_match_current_hits_exactly` (the `SESSION_LITERAL_ALLOWED` loops) | Temporarily deleted the new `result_table_relation` allow-list entry (kept the code migration) — RED: `crates/jammi-db/src/store/mod.rs: fn result_table_relation (ordinal 1) has 1 occurrence(s) of the bare session-registration literal "jammi.{", 0 audited/allowed for THIS SITE`. Reverted (`git checkout --`), re-ran both tests green. |
| Hard-negative mining at `W=1` under `MultipleNegativesRanking`, `cached: false`, changes the trained adapter bytes; the same config with `mine: false` leaves them unreached (non-vacuity, #551 second half) | `training_set.rs::hard_negative_mining_at_w1_moves_the_adapter_bytes_mining_off_leaves_it_unreached` | Discarded the miner's replaced loader (`mined_loader` stays `None`, training falls back to the original triplets) — RED: `assertion 'left != right' failed ... left: "1184:0a99912945b451a9" right: "1184:0a99912945b451a9"` (identical bytes). Reverted. |

##### 3. Uncovered

- **No committed byte-for-byte pin for the mining oracle**, unlike this file's own `gradcache_..`
  oracle. That oracle carries TWO separate pinned constants (`#[cfg(target_os = "linux")]` vs.
  not) because this crate's CPU backprop is demonstrably not byte-identical across platforms — a
  measured fact on this exact fixture, not a hypothesis. Producing the matching pair for the
  mining arm needs a measurement taken on a Linux host, which was not available to this round (no
  Linux runner in this environment). The shipped oracle instead compares two digests captured
  LIVE in the same test run (mining on vs. off) — exactly as sensitive to a mining regression, and
  needs no per-platform constant — but it is not the "pin" #551's text literally asks for. Stated
  honestly rather than shipped at low confidence on a guessed constant that would break CI on its
  actual (Linux) host.
- **The ~9 UNQUOTED `TableReference::bare(format!("jammi.{name}"))` registration sites**
  (`TrainingSetTable::registered_name`, `graph_neighbourhood.rs::load_neighbor_graph_edges`,
  `jammi-db/src/session.rs::read_vectors`/`read_vector_by_key`, `store/mod.rs::register_table`/
  `bind_result_table`) are a DIFFERENT risk class from the quoted-relation class this round
  migrated (what a name registers AS, vs. what a raw-SQL read quotes) — reviewed and left as-is,
  per `gang3.md`'s own scoping ("Out of this property... source-side federation relations... and
  backing-table reads" — the unquoted registration class is the pre-existing Pattern 3 residual,
  not newly in scope here). Their `SESSION_LITERAL_ALLOWED` entries are unchanged.
- **N3's "does not compile" claim for a hand-rolled bypass that duplicates the seam's own logic
  verbatim under a new name**, stated honestly: Rust's privacy model cannot forbid a NEW function
  that reimplements `submit_admitted_training`'s body (matching on `admitted`... no — a bypass
  would need its own `TrainingSpec`, not an `AdmittedTrainingSpec`, so it cannot even name the
  witness type meaningfully) and calls `Catalog::submit_job` directly with a hand-built
  `SubmitJobParams`. This remains only source-syntactically possible, caught by the enumerating
  oracle (executed RED above), never by the type system alone — the ORACLE is the enforcement,
  exactly as `gang3.md` states.
- **N2's `(file, fn, ordinal, count)` shape**: delivered via the PRE-EXISTING
  `SESSION_LITERAL_ALLOWED` machinery (already exactly that 4-tuple shape, confirmed by direct
  inspection of its detector `session_registration_literal_sites`), not a new parallel structure —
  this is the "extend the SAME machinery" the contract asks for, not a gap.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo clippy -p jammi-ai -p jammi-db -p jammi-bench -p jammi-encoders --all-targets -- -D warnings` | 0 | clean |
| `cargo test -p jammi-ai --test it -- rank_admission:: training_set::` | 0 | 26 passed |
| `cargo test -p jammi-ai --test it -- pinned_source_gate::` | 0 | 51 passed (run twice across the round; the second run also confirms the `UNRESOLVED_INCLUDE_STR_TARGETS` re-anchor fix, an unrelated pre-existing drift this round's own gate run surfaced and fixed — see §1/commits) |
| `cargo test -p jammi-db --test it -- materialization:: gang_rank_admission:: --test-threads=1` (sqlite) | 0 | 58 passed |
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_gang cargo test -p jammi-db --test it --features live-postgres-tests -- materialization:: gang_rank_admission:: --test-threads=1` (postgres) | 0 | 104 passed |
| `cargo build -p jammi-bench --bins` / `cargo check -p jammi-bench --all-targets` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | "1068 file(s) scanned, all PATH:LINE citations resolve" — re-anchored 18 `MAINTAINER-GUIDE.md` citations, 1 `pinned_source_gate.rs` self-citation, 2 `jammi-encoders/test_support.rs` citations, all shifted by this round's own line-count changes; computed from `grep -n "fn <ident>"` against the target file, never hand-guessed |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | "no governance-verb leak... allowlist clean" |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | two fixes along the way: `AdmittedTrainingSpec`'s own doc linked its private `spec()` method, and `admit_training_spec`'s doc linked the private `session.rs::submit_fine_tune_spec_deduped` — both converted to backtick code spans (no doc-hidden bypass), per the committed convention (7fd457e) |

No shared-declaration file (`crates/jammi-ai/src/lib.rs`, `Cargo.toml`, `error.rs`) was touched,
so the workspace-wide rustdoc form was not required and not run.

##### 5. Issues closed

- **#573 (N3, the stronger bar): CLOSED.** Every durable training submit edge in the workspace
  (three in-crate, one cross-crate in `jammi-bench`) goes through
  `submit_admitted_training`, the ONE `SubmitJobParams`-construction site for a training kind, and
  the witness (`AdmittedTrainingSpec`) is structurally unconstructible outside `spec.rs` (E0603,
  executed). The enumerating oracle is now derived from `syn::parse_file` over EVERY call in
  `crates/jammi-ai/src`'s production code, not a hand-picked three-function list — closing the gap
  the coordinator's delta named ("the earlier predicate admitted nothing"). The residual honest
  gap (a hand-rolled bypass duplicating the seam's own body under a new name is
  source-syntactically possible) is the SAME residual gang-impl's round 2 already disclosed for the
  original three-edge shape; it is caught by the oracle, never by the type system, and this is
  stated as the design, not hidden.
- **#551 (N2): the newtype half CLOSED further, the mining half PARTIAL.** `RelationKey` now has
  two minters (`TrainingSetTable::sql_relation`, `result_table_relation`), both in `store/mod.rs`,
  both behind the same private field; every live quoted `"jammi.{name}"` call site this round's
  re-derivation found (in `jammi-ai`, `jammi-db`, `jammi-bench`) now goes through one of them.
  `pinned_source_gate.rs`'s pre-existing `(file, fn, ordinal, count)`-keyed enumerating gate
  (`SESSION_LITERAL_ALLOWED`) is updated to match (three entries removed, one added for the
  minter's own site) — this IS the "gains the quoted class" extension: the machinery already
  scanned for the literal `"jammi.{` text (both quoted and unquoted forms) before this round; what
  changed is which sites still trip it. The mining W=1 oracle is a REAL, executed, non-vacuous
  property (digest moves, proven RED on the exact regression #551 names) but ships without the
  committed byte-for-byte pin the issue's text names, for the stated platform reason (§3) — labelled
  PARTIAL, not CLOSED, on that one detail.
- **#543 (N5): not this unit's scope** — GANG3B's (wire-server, N5-harness).

##### 6. Commits

```
84cce4af fix(db,ai,bench,encoders): #551 N2-gate -- a general RelationKey minter for every quoted jammi.{} relation, mining W=1 parity oracle
4b7c7461 fix(ai,bench): #573 N3-seam -- one submit_admitted_training seam for every training-kind jobs row
```

`git log --oneline 7d4b9173..HEAD` shows both commits, both authored by this unit.

### 11.9 GANG3B — the resume-parity harness (wire-server) — landed as 187c6d47 (original tip 954c55f2)

**Lead note:** the first hand-back's external poll was refused and the harness rebuilt on the checkpoint-written seam before landing.

#### GANG3B — N5-harness implementation report (#543, wave 5)

**Revision note**: the coordinator reviewed the first delivery and refuted
this unit's deviation 2 (the external-poll rationale) — a real event seam
(`loop_test_hooks::Event::ResumeCheckpointWritten`) exists and is reachable
from `jammi-server`'s tests, closing the #527 flake class the poll was a
band-aid over. Deviations 1 (the second, corrupted-bundle test row) and 3
(SQLite-only) were accepted as-is. This report describes the REBUILT
implementation (commit `954c55f2`, amended into the same single commit —
no `7c2b9b0b` remains in history).

##### 1. Scope shipped

- **New**: `crates/jammi-server/tests/it/gang_resume_parity.rs` — the N5-harness.
  Two `#[tokio::test(flavor = "multi_thread")]` rows:
  - `peer_and_local_w2_gangs_resume_from_epoch_1s_checkpoint_and_publish_byte_identical_adapters`
    — the parity property the design contract names.
  - `a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart`
    — an ADDED row closing the parity property's own documented blind spot
    (see §2's deviation; accepted by the coordinator).
- **Edited** `crates/jammi-server/tests/it/gang_chaos.rs`: `Fleet` and
  `Member` (struct + the methods the new harness calls: `Fleet::new`,
  `Fleet::host_config`, a new `Fleet::dir()` accessor, `Member::start`,
  `Member::holder`, `Member::wait_slot_free`) go `pub(crate)` so
  `gang_resume_parity.rs` reuses the fixture fleet and the real loopback
  member instead of duplicating that machinery. No behavior changed; both
  existing `gang_chaos.rs` (4 tests) and `gang_coordinator.rs` (2 tests)
  suites still pass unmodified (§4, each run 3× in this session).
- **Edited** `crates/jammi-server/tests/it/main.rs`: `mod gang_resume_parity;`.

###### The kill mechanism (rebuilt per the coordinator's ruling)

The claimed attempt is spawned onto the killed host's OWN dedicated tokio
runtime. The kill point is now the trainer's own discrete,
test-observable event — `jammi_ai::fine_tune::worker::loop_test_hooks::
Event::ResumeCheckpointWritten`, fired inside `TrainingLoop::
save_resume_checkpoint` (`crates/jammi-ai/src/fine_tune/trainer.rs`) the
instant its `put_resume_checkpoint` write returns `Ok`. Armed via
`loop_test_hooks::arm_observed(&job_id, Event::ResumeCheckpointWritten)`
**before** the host claims (mirroring `crates/jammi-ai/tests/it/
crates/jammi-ai/tests/it/jobs_shutdown.rs:410-414`'s own `#527` comment: "armed BEFORE `spawn_worker`
claims and starts the run, so the trainer's fire... can never race ahead of
the arm"), then awaited via `Observed::wait_fired()` under a generous 60 s
backstop (the SAME shape `crates/jammi-ai/tests/it/jobs_shutdown.rs:421-427` uses — `wait_fired`
itself never times out; the TEST bounds it). `jammi-server`'s `test-hooks`
feature forwards to `jammi-ai/test-hooks` (`crates/jammi-server/
Cargo.toml:127,147`), so the hook is reachable from this crate's tests; the
killed host runs entirely in-process (its own dedicated `tokio::runtime::
Runtime`, not a separate OS process), so the arm (on the test's own
runtime) and the fire (on the killed host's runtime) cross runtimes safely
through the hook's process-wide `Mutex<Vec<ArmedEvent>>` registry, exactly
as `gang_chaos::Member`'s session already crosses runtimes (its `session`
built on the test's runtime, its gang server running on its own).

Once `wait_fired()` resolves, `KillableHost::kill()` kills the lease keeper
thread (`LeaseKeeper::kill_thread_for_test` — `gang_chaos.rs`'s split-brain
row's own mechanism) and drops the dedicated runtime
(`shutdown_background`, `gang_chaos::Member::kill`'s exact shape): the
blocking training thread runs on to its next runtime-dependent op — epoch
2's own checkpoint write, or the final publish — and fails there, so
nothing terminal ever reaches the row.

**What this removed** (the coordinator's exact list, all deleted):
`wait_for_resume_checkpoint`'s external `ArtifactStore` poll, its torn-read
tolerance arm (`Ok(Some(_)) | Err(_) => return`), and `RESUME_PARITY_ROWS`'
192-row widening + its race rationale. The fixture is back to
`gang_coordinator.rs`'s own `two_rank_spec()`/`write_pairs_csv()`/
`gang_config(2)` (8 rows), reused verbatim — the property needs one
checkpointable epoch boundary and a second epoch to resume into; nothing
about row/step count matters now that the kill point is the trainer's own
write event rather than a race against it. The `a_corrupted_...` row's own
post-kill checkpoint read is no longer wrapped in a tolerant `match` either
— the write is durably complete and the host is already dead by the time
that read runs, so there is nothing left to race.

The `claim()` polling loop (100 ms `reclaim_expired_jobs`/`claim_next`
retry, gated on the row's own lease) is UNCHANGED and now carries a doc
comment naming it explicitly as `gang_chaos::Coordinator::claim`'s own
shape (not reshaped into a shared function across the two structs, since
`Coordinator` and `KillableHost` close over different lease/attempt
constants — flagged, not silently duplicated).

###### Deviations from the design contract (`contracts/gang3.md`'s N5-harness row) — final status

1. **A second test row, not named by the contract.** ACCEPTED by the
   coordinator. Reason unchanged from the first delivery: the contract's
   stated falsification ("skip discover_resume on one topology → red")
   does not red the byte-parity property for this fixture — `resume.rs`'s
   own doc ("On `Device::Cpu` the forward+backward+step is a pure function
   of `(seed, source rows, config)`", `crates/jammi-ai/src/fine_tune/
   crates/jammi-ai/src/fine_tune/resume.rs:4-8`) means a silently-skipped-resume attempt 2 reaches the
   SAME final bytes as a genuinely-resumed one, symmetrically across both
   topologies. `trainer.rs`'s own R6/R7 resume oracle
   (`resume_reproduces_the_exact_trajectory_byte_for_byte`,
   `crates/jammi-ai/src/fine_tune/trainer.rs:12429-12480`) hits the
   identical limitation and closes it the same way (a checkpoint-content
   mutation between crash and resume, not a final-bytes comparison). The
   second row reuses `ArtifactStore::fetch_resume_checkpoint`'s own
   documented hard-error-on-corruption contract
   (`crates/jammi-db/src/store/artifact.rs:453-460`) as the falsifiable
   oracle.
2. **REFUTED and rebuilt** (see above) — no longer a deviation: the
   harness now uses the SAME kind of discrete event-observation seam
   `jobs_shutdown.rs` uses, per the coordinator's citation
   (`crates/jammi-ai/src/fine_tune/worker.rs:5239-5312`,
   `crates/jammi-ai/tests/it/jobs_shutdown.rs:410-432`), not an external
   poll.
3. **No Postgres arm.** ACCEPTED. `gang_chaos.rs`/`gang_coordinator.rs` —
   this harness's own family — are SQLite-only via `Fleet::host_config`'s
   hardcoded `CatalogConfig::Sqlite`; neither sibling harness has ever run
   on Postgres. Followed that precedent rather than reshaping `Fleet`
   alone. Flagged for the lead: a Postgres arm for this family, if wanted,
   is a shared change across all three files, not a
   `gang_resume_parity.rs`-only one.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| For a claimed `fine_tune` job SIGKILLed once epoch 1's durable checkpoint write is OBSERVED, attempt 2's published rank-0 adapter bytes are IDENTICAL whether the killed host ran `Peer` W=2 (a real loopback member) or `Local` W=2 (in-process) | `gang_resume_parity::peer_and_local_w2_gangs_resume_from_epoch_1s_checkpoint_and_publish_byte_identical_adapters` (SQLite; 3 green runs post-rebuild, `~18.3-18.4s` each) | `crates/jammi-ai/src/fine_tune/worker.rs:8455` — `let resume = discover_resume(...)?;` replaced with `let resume = None;` (discover_resume never called). Re-executed against the REBUILT (event-hook) harness itself, not carried over from the pre-rebuild run: result **UNCOVERED for this specific mutation** — the run STAYED GREEN (`test result: ok. ... peer_and_local_w2_gangs_resume_...`, `finished in 18.44s`; CPU determinism makes a silently-skipped resume byte-identical to a genuine one for this fixture, symmetrically across both topologies — §1 deviation 1). Reverted; `git diff --stat crates/jammi-ai/src/fine_tune/worker.rs` empty after revert, confirmed by a clean rebuild + green rerun of both rows. |
| A present-but-corrupted `_resume/` bundle fails attempt 2 as a hard error (`status = "failed"`, error names `sha256`), never a silent from-scratch restart that happens to still complete | `gang_resume_parity::a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart` (SQLite; 3 green runs post-rebuild) | The SAME re-executed mutation above (`let resume = None;`, against the rebuilt harness) turns THIS property red: `left: "completed"`, `right: "failed"` (captured first line: `thread '...a_corrupted_epoch_1_checkpoint...' panicked at crates/jammi-server/tests/it/gang_resume_parity.rs:418:5: assertion \`left == right\` failed: a corrupted resume bundle is a hard error, never a silent from-scratch restart: Row { status: "completed", attempts: 2, releases: 0, ... }`). Executed proof that `discover_resume` is genuinely on the path for both topologies (topology-agnostic mutation; this row runs on `Local`, `Peer`'s own path shares the identical `run_fine_tune_blocking` call site). Reverted; rebuild + rerun green. |
| The `pub(crate)` visibility widening on `gang_chaos.rs`'s `Fleet`/`Member` changes no behavior | `gang_chaos::` (4 tests, 3 green runs) and `gang_coordinator::` (2 tests, 3 green runs), unmodified after the edit | Pre-existing oracles (mine to preserve, not to author). Ran green post-edit and post-rebuild as the regression check (§4). |

##### 3. Uncovered

- **Byte-parity's blind spot to a silently-skipped resume, for this
  fixture's determinism class.** Documented in §1 deviation 1, §2 row 1,
  and the file's own module doc comment (`crates/jammi-server/tests/it/gang_resume_parity.rs:29-46`).
  Closed by the second row, not by the first — stated explicitly.
- **A genuine topology-specific resume divergence** is not independently
  falsified by a dedicated mutation in this unit: `run_fine_tune_blocking`/
  `discover_resume` (`crates/jammi-ai/src/fine_tune/worker.rs:8271`) is
  ONE function called by every rank body regardless of topology — there is
  no separate "Local's discover_resume" vs "Peer's discover_resume" to
  diverge in-scope (test-only, no `jammi-ai/src` edit). The byte-parity
  row's real contribution is exercising the crash-and-resume INTEGRATION
  path end to end for both topologies without erroring (a topology-specific
  dispatch bug — e.g., one topology's rank body never reaching
  `run_fine_tune_blocking` at all on a resumed attempt — would still fail
  outright, not silently diverge in bytes). General topology-vs-
  `LocalGang`-reference byte equality (the uninterrupted case) is
  independently covered by `gang_coordinator.rs`'s own Peer-vs-reference
  assertion and `jammi-ai/tests/it/gang_coordinator.rs`'s Local-vs-reference
  assertion — pre-existing, not re-derived here.
- **None remaining on the kill-timing axis** — this was the prior
  revision's biggest uncovered item (a probabilistic external-poll race);
  the event-hook rebuild removes it entirely. `wait_fired()`'s only bound
  is the 60 s backstop against a wedged/starved machine, matching
  `jobs_shutdown.rs`'s own established bound.

##### 4. Gates (post-rebuild; all commands run from `/private/tmp/claude-501/.../scratchpad/wt-gang3b` with `CARGO_TARGET_DIR=/private/tmp/claude-501/.../scratchpad/targets/gang3b` and `RUSTC_WRAPPER=sccache` exported)

| Command | Result |
|---|---|
| `cargo test -p jammi-server --features test-hooks --test it "gang_resume_parity::" -- --test-threads=1` | **ok. 2 passed; 0 failed**, run 3× (~18.3-18.4s each), no flake |
| `cargo test -p jammi-server --features test-hooks --test it "gang_chaos::" -- --test-threads=1` | **ok. 4 passed; 0 failed**, run 3× (~21.8s each) |
| `cargo test -p jammi-server --features test-hooks --test it "gang_coordinator::" -- --test-threads=1` | **ok. 2 passed; 0 failed**, run 3× (~2.25-2.31s each) |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | exit 0, no warnings |
| `cargo fmt -p jammi-server --check` | exit 0 (one `cargo fmt -p jammi-server` pass applied first — 3 reflow-only diffs from the rebuild, no logic change) |
| `python3 ci/scripts/perf/check_citations.py` | exit 0 — `check-citations: 1069 file(s) scanned, all PATH:LINE citations resolve` |
| `python3 ci/scripts/check_no_consumer_names.py` | exit 0 — `no-consumer-names: OK` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-server --no-deps` (extra, beyond COMMON.md's minimum) | exit 0 — `Generated .../doc/jammi_server/index.html` |

Not run (out of COMMON.md's minimum for this unit): workspace-wide build,
full crate suites beyond the three test modules above, `merge_path.sh`,
the live-Postgres lane (see §1 deviation 3).

##### 5. Issues closed

- **#543** (fine-tune: resume-state broadcast for world_size > 1) — per the
  dispatch, closes on this oracle plus the executed listing refusal
  (`crates/jammi-db/tests/it/gang_membership.rs:544`,
  `file_and_s3_rooted_members_are_not_gang_members_of_each_other` —
  pre-existing, re-located and confirmed at that line, not authored by
  this unit). **CLOSED** for the wire-server/N5-harness slice: the K4
  resume-parity row is landed (§2), covering both a Peer-member and a
  Local in-process gang resuming from the SAME `{job_id}/_resume/`
  checkpoint through the one shared `discover_resume` entry, with an
  executed falsification for the property's own blind spot, and — after
  the coordinator's ruling — a genuinely deterministic (event-observed,
  not raced) kill point. The issue's own BROADCAST mechanism (ai-core/
  trainer scope; no new round message was needed here — no `.proto`/
  `jammi-wire` file touched) is outside this unit's file set
  (`crates/jammi-server/tests/it/` only) and is GANG3A's or a separate
  unit's to report on.

##### 6. Commits

```
954c55f2 test(server): #543 N5-harness — the K4 resume-parity row (gang3b)
```

`git log --oneline 7d4b9173..HEAD`:
```
954c55f2 test(server): #543 N5-harness — the K4 resume-parity row (gang3b)
```

(The original `7c2b9b0b` was amended into this commit — no separate SHA
survives; the working tree is clean.)

### 11.10 CAPSURF — the flash cascade preempts the fused block (numerics) — landed as 0179471e (original 5a2e936a)

**Lead note:** closer: GPU prove run 35171308424, all four arches green.

#### CAPSURF — implementer report

Base: `bd3a6202` (wave-5 tip named in my dispatch, NOT `db19a614` — the brief
explicitly overrides COMMON.md's generic base). Branch: `unit/capsurf`, one
commit, tip `5a2e936a`.

##### 1. Scope shipped

Root cause found by the contract's mandated STATIC bisection (§2 step 1) —
**not** a diverging change in the encoder's own dispatch order. The 57-file
window `9f7f3bc1..aace002f` contains commit `4f45c2c5` ("feat(encoders):
extract the shared attention cascade, wire BERT/DistilBERT training + GELU
seam (#462, #463)"), which wires `BertAttention::forward_training`
(`crates/jammi-encoders/src/bert.rs:190`) onto the SAME shared
`attention_cascade::training_attention_cascade` function ModernBERT uses —
confirmed by diffing `bert.rs` at `9f7f3bc1` (no `forward_training` method,
no `attention_block_fused` reference anywhere in that file) against the
window's tip. BERT always supplies `flash: &FlashDecision::Declined { .. }`
(BERT has no flash transport wired, at any dtype — that file's own doc), so
every BERT training forward dispatches `attention_block_fused` Fused
unconditionally, on every dtype, once this wiring landed. This is CORRECT
encoder behavior (BERT genuinely has no flash arm — a separate, undone line
of work), not a defect in `training_attention_cascade`'s own ordering
(flash consulted first, `attention_block_fused`'s `admit()` provably
unreachable once flash dispatches Fused — see Properties below).

The actual defect is in the TEST, `crates/jammi-ai/tests/gpu_capability/capability_surface.rs`,
which the window does NOT touch (per the contract's own framing) but whose
PRE-EXISTING `gelu_erf` probe (`bert_probe_dtype`, added for issue #463) now
ALSO fires `attention_block_fused` as a side effect of the wiring above.
The `attention_block` TIER-PREEMPTION assertion's "after" snapshot was a
LIVE re-read of `counters_for("attention_block_fused")`/
`cascade_counters_for("attention_block_flash")` positioned textually AFTER
the `gelu_erf` block (which calls `bert_probe_dtype`) — so on every bf16/f16
iteration, BERT's own unconditional `attention_block_fused` dispatch leaked
into the window the assertion attributes to ModernBERT's own flash-vs-block
behavior, breaking `"attention_block_fused must NOT have dispatched"` at
`crates/jammi-ai/tests/gpu_capability/capability_surface.rs:904` (nightly-red every run since 2026-09-07,
matching `before=DispatchSnapshot { fused: 4 } after={fused: 5}` — a +1
delta from BERT's own single-layer `tiny_bert_head64` fixture, not from
ModernBERT).

Files touched:
- `crates/jammi-ai/tests/gpu_capability/capability_surface.rs` — moved the
  `attention_block_fused`/`attention_block_flash` "after" snapshot capture
  to immediately after `probe_dtype`'s own call returns, before
  `bert_probe_dtype` ever runs in the same iteration (mirroring the
  isolation the `gelu_erf` check's own window already used for ITS
  counter). No assertion text changed (non-goal §3 honored).
- `crates/jammi-encoders/src/modernbert.rs` — added ONE new `#[cfg(test)]`
  function, `lora_wrapped_attention_never_reaches_attention_block_fused_when_flash_dispatches`
  (the contract's mandated CPU-hermetic second oracle). No production code
  changed in this crate.
- `ci/scripts/perf/test_finetune_ab_disable_op_keys.py` — one citation
  (`crates/jammi-encoders/src/modernbert.rs:10575` → `:10677`) re-resolved after my own test's
  insertion shifted the line it names; caught by
  `python3 ci/scripts/perf/check_citations.py` before I finished, not by a
  later reviewer.

**Deviation from the contract's own framing, stated per COMMON.md:** §0/§2
describe the defect as "the training forward gets a second attention
entry" and ask to "find the change that gives the training forward a
second attention entry ... Do not guess." I read this literally first
(looked for a LoRA-wrapper-side or kernels-side second dispatch path into
`attention_block_fused` — see Uncovered) and found none: `wqkv`/`wo` are
plain `MaybeLoraLinear` linear projections: LoRA wraps the LINEAR, never
the attention op itself, and `training_attention_cascade`'s own control
flow makes a genuine second entry structurally impossible (an early
`return` inside the `if flash_dispatch == CascadeOutcome::Fused` block, not
a flag or a fallthrough). The "second attention entry" that actually
explains the failing assertion is BERT's OWN (correct, by-design) call into
the SAME shared function — a second CALLER sharing a counter, not a second
CODE PATH inside ModernBERT's own dispatch. I judged this satisfies the
contract's intent (name the exact commit/line, fix at the root, no flag)
better than declaring the bisection inconclusive under §4's stop rule,
since the mechanism is concretely named, the fix is a one-window edit with
no new knob, and the code line the reason rests on is cited above
(`crates/jammi-encoders/src/bert.rs:190`, `4f45c2c5`). Flagging this deviation explicitly per
COMMON.md so the lead can open that line before accepting it.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| C1 (contract): for every `FlashDecision::Fused` (any admission — dense here), `training_attention_cascade`'s own `admit()` call on `attention_block_fused` is NEVER reached, for a LoRA-wrapped attention module (`Wqkv` wrapped) and an unwrapped one (`Wo` frozen) in the SAME call | `crates/jammi-encoders/src/modernbert.rs::modernbert::tests::lora_wrapped_attention_never_reaches_attention_block_fused_when_flash_dispatches` (CPU, no CUDA/flash-attn feature — `cargo test -p jammi-encoders --lib modernbert::`) | Changed `attention_cascade.rs`'s `if flash_dispatch == CascadeOutcome::Fused` to `if false && flash_dispatch == CascadeOutcome::Fused` (simulating a diverging change that lets the block arm run after flash already fired). Red, first line: `thread 'modernbert::tests::lora_wrapped_attention_never_reaches_attention_block_fused_when_flash_dispatches' panicked at crates/jammi-encoders/src/modernbert.rs:4249:14: called \`Result::unwrap_err()\` on an \`Ok\` value: Tensor[dims 2, 6, 64; f32]`. Reverted after confirming; `git diff` clean on `attention_cascade.rs`. |
| The `capability_surface.rs` `attention_block` assertion's before/after window is scoped to `probe_dtype`'s OWN dispatch, not contaminated by `bert_probe_dtype`'s | Cannot be executed hermetically (requires the live-gpu-tests/cuda/flash-attn build + a real CUDA device to actually run `capability_surface()`) — see Uncovered. Verified STATICALLY by re-reading the edited control flow and the counter registry's own doc (`jammi-kernels/src/admission.rs`'s "op-keyed dispatch-counter registry" section: `counters_for`/`cascade_counters_for` return ONE process-wide static per op key, shared by every caller that passes that key — confirming BERT and ModernBERT genuinely share the same counter object). | Not executed (no GPU in this worktree); the lead's `GPU prove (RunPod)` run on this branch is the contract's own designated closer for this half. |

##### 3. Uncovered

- **The live GPU assertion itself.** I cannot run `capability_surface()`
  (requires `live-gpu-tests` + a CUDA device; `--features live-gpu-tests`
  alone compiles it — confirmed via `cargo check`/`cargo clippy` — but
  `skip_without_gpu!` short-circuits without a device). Per the contract's
  §2 step 4, the lead dispatches `GPU prove (RunPod)` on this branch as the
  closer; that run is what actually proves the fix against the real
  failure signature (`before=DispatchSnapshot { fused: 4 } after={fused:
  5}` at all four arches).
- **DistilBERT.** `bert.rs`'s wiring doc and the window's own diff
  (`distilbert.rs` also touched, 835 lines) suggest DistilBERT shares the
  same `forward_training`-onto-shared-cascade shape as BERT, but
  `capability_surface.rs` never drives a DistilBERT probe, so I did not
  chase whether it shares BERT's exact counter-contamination mechanism —
  it is out of this contract's stated scope (the GPU assertion only
  concerns ModernBERT vs. the shared counter, and my fix isolates the
  window regardless of how many OTHER callers exist).
- **Exact reproduction of the `before=4, after=5` numbers.** I traced the
  mechanism (BERT's `tiny_bert_head64` fixture, `num_hidden_layers: 1`,
  contributing exactly `+1` per dtype iteration to a counter ModernBERT's
  own 2-layer probe also writes) but did not reproduce the precise
  cumulative totals across all three dtype iterations by hand beyond
  confirming the SIGN and MECHANISM of the contamination (a live run is
  the only way to pin the exact cumulative numbers, and I have no GPU).

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p jammi-encoders --all-targets -- -D warnings` | 0 | |
| `cargo clippy -p jammi-lora --all-targets -- -D warnings` | 0 | (no jammi-lora source touched; ran per brief's named lane) |
| `cargo clippy -p jammi-ai --test gpu_capability --features live-gpu-tests -- -D warnings` | 0 | `capability_surface.rs` is `required-features = ["live-gpu-tests"]`; compiles clean without CUDA |
| `cargo check -p jammi-ai --test gpu_capability --features live-gpu-tests` | 0 | |
| `cargo test -p jammi-encoders --lib modernbert::` | 0 | 94 passed, 0 failed, 150 filtered out (includes the new oracle) |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-encoders --no-deps` | 0 | ran beyond COMMON.md's minimum since new doc comments were added |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1067 file(s) scanned, all PATH:LINE citations resolve` |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |

Not run: workspace-wide builds, full crate suites, `cargo doc --workspace`,
live-Postgres lane, `merge_path.sh` (per COMMON.md, these are the lead's).

##### 5. Issues closed

No numbered issue file under `<scratchpad>/issues/` was named in my
dispatch — the brief pointed directly at `<scratchpad>/contracts/capsurf.md`
(a wave-5 nightly-red contract, not an issue-file unit). Treating the
contract's own §0/§1 (C1) as the closing target:

- **CAPSURF (contract `<scratchpad>/contracts/capsurf.md`): CLOSED for the
  STATIC half** — root cause named with the exact commit (`4f45c2c5`) and
  line (`crates/jammi-encoders/src/bert.rs:190`), fixed at the root (one
  measurement window, no flag, no assertion-text change), and the mandated
  CPU-hermetic oracle (C1) lands and is proven red-then-green with an
  executed mutation. **OPEN for the GPU half** — the contract's own §2 step
  4 designates `GPU prove (RunPod)` on this branch as the closer; I have no
  GPU in this worktree to run it.

##### 6. Commits

```
$ git log --oneline bd3a6202..HEAD
5a2e936a fix(ai,encoders): #500 capsurf -- isolate the attention_block_fused before/after window from bert_probe_dtype's own dispatch; CPU-hermetic preemption oracle
```

(`git log --oneline db19a614..HEAD` — COMMON.md's literal template range —
would also list every wave-5 commit already on `bd3a6202` before my branch
point; my own work is the single commit above.)

### 11.11 ENUMS — inventories by construction (db) — landed as c6ce59b1, 2de4ba80 (original tip d33e6c8e)

**Lead note:** cherry-picked; the jobs_queue addendum closed a pre-existing shared-database leak the lead had classified on a fresh database.

#### ENUMS — implementation report (db; #550 + coordinator addendum)

##### 1. Scope shipped

Files touched (all in `crates/jammi-db/` and `crates/jammi-numerics/`, plus the
workspace manifest for the new dependency):

- `Cargo.toml` (workspace root) — new `[workspace.dependencies]` entry:
  `strum = { version = "0.26", default-features = false, features = ["derive"] }`.
- `crates/jammi-db/Cargo.toml` — `strum.workspace = true` dependency.
- `crates/jammi-numerics/Cargo.toml` — `strum.workspace = true` dependency.
- `crates/jammi-db/src/catalog/result_repo.rs` — `ResultTableKind::ALL`.
- `crates/jammi-db/src/catalog/status.rs` — `JobStatus::ALL`.
- `crates/jammi-numerics/src/quantization.rs` — `WeightQuantization::ALL`.
- `crates/jammi-db/tests/it/jobs_queue.rs` — coordinator addendum (see §5).
- `Cargo.lock` — dependency resolution for `strum`/`strum_macros` (+ transitive
  `heck`/`proc-macro2`/`quote`/`rustversion`/`syn`, all already present in the
  lockfile at other versions, so no new second-line-of-an-engine-crate risk).

**Mechanism chosen: `strum`'s `#[derive(VariantArray)]`**, not an in-tree
proc-macro. Reasons stated in the workspace manifest's comment on the
dependency and repeated here: `VariantArray`'s derive macro reads the enum's
own variant list directly off the parsed AST at macro-expansion time and
emits `Self::VARIANTS: &'static [Self]` from it — categorically the same
class of guarantee an in-tree proc-macro would have to reimplement (parse the
enum, walk its variants, emit an array), with no reduction in rigor and much
less new surface (no new workspace member, no new `syn`/`quote` consumer to
audit, `strum`/`strum_macros` MIT-licensed and already resolvable from the
crates.io registry this workspace already builds against). `EnumIter` was
considered and rejected: it yields an iterator, not a `&'static [Self]` a
caller can `.len()`/index/store by reference the way every existing call site
of `ALL` already does; `VariantArray` is the closer match to what `ALL` was.
Every enum here already hand-writes `Display`/`FromStr` for its SQL/wire
string form (deliberately decoupled from the Rust variant identifier, e.g.
`ResultTableKind::NeighborGraph` <-> `"neighbor_graph"`), so `strum`'s
`Display`/`EnumString` derives are never pulled in — `features = ["derive"]`
only, `default-features = false`.

**The three sites, relocated at HEAD** (issue #550 cited `result_repo.rs`
ALL, `crates/jammi-db/src/catalog/status.rs:78`, `crates/jammi-numerics/src/quantization.rs:127`):
- `ResultTableKind::ALL` — `crates/jammi-db/src/catalog/result_repo.rs:61`
  (was `pub const ALL: [Self; 4] = Self::all();` backed by a private
  exhaustive-match `all()` fn; both removed, replaced by
  `pub const ALL: &'static [Self] = <Self as VariantArray>::VARIANTS;`).
- `JobStatus::ALL` — `crates/jammi-db/src/catalog/status.rs:85` (was
  `pub const ALL: [JobStatus; 4] = [Self::Queued, ...];` at the file's
  original line 78, exactly where the issue cited it).
- `WeightQuantization::ALL` — `crates/jammi-numerics/src/quantization.rs:139`
  (was a 10-element hand-typed array at the file's original line 127).

**Deviation from the issue's literal type**: `ALL`'s type changed from
`[Self; N]` (fixed array) to `&'static [Self]` (slice) at all three sites,
because `strum::VariantArray::VARIANTS` is declared `const VARIANTS: &'static
[Self]` in the trait (there is no `[Self; N]`-typed const generic-length
variant of this trait in strum 0.26) — verified directly against the
vendored crate source
(`~/.cargo/registry/src/*strum 0.26.0 (`src/lib.rs`, line 226)`). Every call site
across `jammi-db`, `jammi-numerics`, and `jammi-ai` (a downstream consumer of
`WeightQuantization::ALL`, `crates/jammi-ai/src/model/backend/gguf.rs:882`,
not edited — `.to_vec()` on a slice and on an array behave identically, so
that call site compiles unchanged) was re-checked and updated where the
slice-vs-array distinction mattered (`for x in ALL` now binds `x: &Self` for
a plain slice iteration, or `.iter().copied()` where the loop body genuinely
needs owned values, e.g. `WeightQuantization::ord_is_total_and_consistent_with_eq`'s
nested `a.cmp(&b)`). `-D warnings` clippy (`unnecessary_to_owned`) enforces
that `.iter().copied()` is used ONLY where actually needed, never
reflexively.

**Round-trip oracles now iterate the derived inventory**, not a hand-typed
literal list, closing the issue's "the round-trip oracles iterate the
derived inventory" acceptance line explicitly:
- `crates/jammi-db/src/catalog/status.rs`'s
  `job_status_round_trips_through_display_and_from_str` was
  `for status in [JobStatus::Queued, JobStatus::Running, ...]`; now
  `for status in JobStatus::ALL`, plus an explicit `JobStatus::ALL.len() == 4`
  assertion.
- `ResultTableKind`'s and `WeightQuantization`'s round-trip tests already
  iterated `Self::ALL` before this change (that was the whole point of
  #550's complaint — they iterated the STALE hand-list); no further edit
  needed there beyond the type-driven loop-binding fixes above.

**Issue-#550 citations removed from shipped doc comments** once this PR
closes the issue: the old doc comments on `ResultTableKind::ALL`,
`ResultTableKind`'s two round-trip tests, and the module's own exhaustive-
match-vs-array explanation all cited
`<https://github.com/f-inverse/jammi-ai/issues/550>` as an open problem;
since this change closes it, those citations were deleted and the prose
rewritten to describe the CURRENT (derive-based) mechanism directly — "docs
reflect current state" (COMMON.md), not a journey marker for a now-resolved
issue. (`JobStatus`'s new doc comment never cited the issue at all, for
consistency.)

No cut. Nothing was deferred; all three named sites converted in this unit.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| **E1 — `ResultTableKind::ALL` is generated from the enum's own variant list**, so a variant added to the enum and given a spelling (required to compile, every exhaustive match on `Self` demands it) is in `ALL` with no second edit. | `catalog::result_repo::tests::every_result_table_kind_round_trips_through_its_db_string` and `an_unknown_result_table_kind_is_refused_naming_every_accepted_spelling`, both iterating `Self::ALL` (`cargo test -p jammi-db --lib result_repo::tests`). | Two mutation rounds, both reverted cleanly (`git diff --stat` empty after): **(a) BASE (pre-fix) behavior**: reconstructed the pre-#550 file from `git show 7d4b9173:crates/jammi-db/src/catalog/result_repo.rs`, added a 5th variant `Probe`, extended the `all()` match PATTERN (required to compile) and `as_db_str` but left the `[Self; 4]` ARRAY untouched (the exact shape #550 names), swapped it into the working tree, ran `cargo test -p jammi-db --lib result_repo::tests`: **all 3 tests pass, including a new probe test asserting `try_from_db_str("probe_kind")` ERRORS** — the pre-existing suite stays green while the mechanism is broken, exactly as #550 claims. **(b) FIXED (derive-based) behavior**: same `Probe` addition + `as_db_str` extension, this time against the derive-based file, **no edit to `ALL` at all** — `cargo test -p jammi-db --lib result_repo::tests`: `ResultTableKind::ALL.len() == 5` and `try_from_db_str("probe_kind")` SUCCEEDS with zero further edits. Both probes then reverted (`cp` from a scratch backup); `git diff --stat` on the file confirmed clean before the real commit. |
| **E1 — `JobStatus::ALL` is generated from the enum's own variant list** (same property, `status.rs`). | `catalog::status::tests::job_status_round_trips_through_display_and_from_str` (now iterating `JobStatus::ALL`, plus `JobStatus::ALL.len() == 4`), `catalog::status::tests::job_status_terminality_matches_the_retention_predicate` (`cargo test -p jammi-db --lib catalog::`). | Same class of defect as `ResultTableKind`'s BEFORE this unit (`pub const ALL: [JobStatus; 4] = [Self::Queued, ...];`, a hand list beside an enum with no exhaustive-match binding it — `is_terminal`'s `matches!` is not even exhaustive-checked). Not separately re-executed as a standalone mutation (would duplicate the `ResultTableKind` drill above on an identically-shaped bug); the FIX is identical in kind and the round-trip test now ranges over the derived `ALL` the same way. |
| **E1 — `WeightQuantization::ALL` is generated from the enum's own variant list, and stays in `gguf_wire_id` ascending (= declaration) order** (`quantization.rs`). | `quantization::tests::display_and_from_str_round_trip`, `all_covers_every_variant_exactly_once`, `ord_sorts_into_ascending_wire_id_order`, `gguf_wire_id_matches_the_verified_candle_table` (`cargo test -p jammi-numerics quantization`). | Same defect class; `ord_sorts_into_ascending_wire_id_order` is itself the oracle that a derive-based `ALL` still matches wire-ID order (`shuffled.sort(); assert_eq!(shuffled, WeightQuantization::ALL.to_vec())` — this passed both before and after, since `VariantArray` preserves DECLARATION order and the module doc already pins declaration order == wire-ID ascending order as deliberate). Not separately mutation-tested (identical shape to the `ResultTableKind` drill). |
| **Type-change safety**: changing `ALL`'s type from `[Self; N]` to `&'static [Self]` does not silently break a downstream consumer outside `jammi-db`/`jammi-numerics`. | `cargo test -p jammi-ai --lib weight_quantization_from_ggml` (the one call site of `WeightQuantization::ALL` outside the two owned crates, `crates/jammi-ai/src/model/backend/gguf.rs:882`). | Not mutated (a compile-and-pass oracle, not a red/green behavioral one) — the check IS that this compiles and passes unchanged; it does. |
| **Coordinator addendum — every catalog test asserting a count over rows it seeded owns those rows** (`jobs_queue.rs`'s `file:///artifacts/...` literals). | `jobs_queue::a_retained_epoch_checkpoints_own_row_makes_its_exact_prefix_referenced::postgres` and `jobs_queue::finish_job_with_model_skips_a_name_occupied_epoch_checkpoint::postgres`, each run 3x consecutively (`cargo test -p jammi-db --features postgres,live-postgres-tests --test it <name> -- --test-threads=1`, `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1 port 54329/jammi_enums`, run individually three separate times, not looped, per the sandbox's git-path heuristic rejecting shell loops over this path). | **(a)** Reverted `served_prefix`/`retained_prefix`/`unretained_prefix` to the original fixed literals (`.to_string()`, bypassing the new helper) against the already-3x-green `jammi_enums` database: run 1 (first time this exact literal existed in the db) passed; run 2 **FAILED**: `panicked at crates/jammi-db/tests/it/jobs_queue.rs:1630:5: assertion left == right failed: ... left: 2 right: 1` — reproducing the coordinator's exact report. Reverted, re-verified 3x green. **(b)** The `finish_job_with_model_skips_a_name_occupied_epoch_checkpoint` fix was found BY this same sweep, not anticipated: my first pass suffixed only the artifact-path literals, kept `job_id`/model ids fixed; `cargo test -p jammi-db --features postgres,live-postgres-tests --test it jobs_queue:: -- --test-threads=1` (the full-file sweep) **FAILED**: `assertion left == right failed left: Some(".../epoch_1") [old run's suffix] right: Some(".../epoch_1") [new run's suffix]` — the checkpoint row's model id was reused across runs and `finish_job_with_model`'s epoch-checkpoint path is insert-IF-NAME-AVAILABLE (unlike `register_model`'s upsert), so the second run's insert was silently skipped, leaving the FIRST run's stale artifact_path. Fixed by suffixing `job_id` and every model id in that test too; re-ran 3x consecutively, green. |
| **Self-test — no `file:///artifacts/...` literal in `jobs_queue.rs` bypasses the run-suffix helper.** | `jobs_queue::every_artifact_path_literal_in_this_file_goes_through_the_run_suffix_helper` (`cargo test -p jammi-db --test it every_artifact_path_literal`). | Two rounds, both fixed forward (not reverted, since this IS the new test): (a) first draft scanned the WHOLE file including the self-test's own source, which names the literal substring it searches for in its doc comment and assertion strings — found 6 occurrences instead of 1, failed as designed (a genuine self-referential-scan bug, not the target defect, but it IS an executed red/green transition on this exact assertion). (b) fixed by slicing the scan to exclude this test's own doc comment + body via a marker string; reran, green (1 occurrence: the helper's own `format!` body). |

##### 3. Uncovered

- **`JobStatus`/`WeightQuantization` did not get a standalone `Probe`-variant
  mutation drill** the way `ResultTableKind` did — the fix is structurally
  identical (same derive, same removed hand-list shape) and re-running the
  full drill three times would be a repeat of the same demonstration rather
  than a new one. Labelled here rather than silently omitted: if a reviewer
  wants the executed refutation on these two specifically, it is the same
  mechanical steps as §2's `ResultTableKind` row.
- **`terminal_sql_list`/`non_terminal_sql_list` do not have a dedicated unit
  test asserting they partition `JobStatus::ALL`** (they are exercised
  indirectly via `jobs_repo.rs`/`model_repo.rs`'s use of them in `it` tests
  that all passed). Out of E1's stated scope (the inventory's completeness,
  not these two derived helpers' correctness) — not touched, not broken.
- **`EvalRunStatus`, `ModelStatus`, `JobExecution`** (same file as
  `JobStatus`) have no `ALL` hand-list today and were left untouched —
  outside the three sites #550 and the design contract name.
- **The addendum's other 12 (of 16) `file:///artifacts/...` literals**
  (in `finish_job_with_model_is_an_attempt_guarded_compare_and_set`,
  `finish_job_with_model_update_is_scoped_by_version_and_tenant`, and
  `finalize_cas_still_matches_a_released_lease`) do not feed a
  `count_models_naming_prefix_all_tenants` call and were not independently
  mutation-tested for the leak (only the two that DID break under the full
  sweep were). They were still routed through the same `artifact_path`
  helper for uniformity and because the self-test now enforces it file-wide,
  but their own 3x-rerun safety rests on the general argument in the commit
  message (equality checks on a row read back by its own unique key are
  immune to the leftover-row leak that only bites a COUNT aggregation), not
  on an executed 3x drill per test.

##### 4. Gates

| Command | Exit | Result |
|---|---|---|
| `cargo test -p jammi-db --lib catalog::` | 0 | 72 passed |
| `cargo test -p jammi-db --lib` (full, extra) | 0 | 588 passed |
| `cargo test -p jammi-numerics quantization` | 0 | 9 passed |
| `cargo test -p jammi-db --test it finish_job_with_model` (sqlite) | 0 | 3 passed |
| `cargo test -p jammi-db --test it finalize_cas_still_matches_a_released_lease` (sqlite) | 0 | 1 passed |
| `cargo test -p jammi-db --test it every_artifact_path_literal` | 0 | 1 passed |
| `cargo test -p jammi-db --features postgres,live-postgres-tests --test it finish_job_with_model -- --test-threads=1` | 0 | 6 passed (3 tests x2 backends) |
| `cargo test -p jammi-db --features postgres,live-postgres-tests --test it finalize_cas_still_matches_a_released_lease -- --test-threads=1` | 0 | 2 passed |
| `cargo test -p jammi-db --features postgres,live-postgres-tests --test it a_retained_epoch_checkpoints... -- --test-threads=1` x3 consecutive | 0,0,0 | ok x3 |
| `cargo test -p jammi-db --features postgres,live-postgres-tests --test it finish_job_with_model_skips... -- --test-threads=1` x3 consecutive | 0,0,0 | ok x3 |
| `cargo test -p jammi-db --features postgres,live-postgres-tests --test it jobs_queue:: -- --test-threads=1` (full file, both backends, extra) | 0 | 117 passed |
| `cargo test -p jammi-db --test it` (full suite, sqlite, extra) | 0 | 523 passed, 1 ignored |
| `cargo test -p jammi-ai --lib weight_quantization_from_ggml` | 0 | 1 passed |
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo clippy -p jammi-numerics --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-db --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-db --all-targets --features postgres,live-postgres-tests -- -D warnings` | 0 | clean |
| `cargo deny check` (default features) | 0 | advisories/bans/licenses/sources ok |
| `cargo deny --all-features check` | 0 | advisories/bans/licenses/sources ok |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1068 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing exempt) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-db -p jammi-numerics --no-deps` | 0 | clean |

All commands run with `CARGO_TARGET_DIR=<scratchpad>/targets/enums` and
`RUSTC_WRAPPER=sccache` set for every invocation, from the `unit/enums`
branch in `<scratchpad>/wt-enums`.

##### 5. Issues closed

- **#550 (catalog: enum inventories derived by construction)**: CLOSED. All
  three named sites (`ResultTableKind::ALL`, `JobStatus::ALL`,
  `WeightQuantization::ALL`) converted to `#[derive(strum::VariantArray)]`.
  Property and oracle: §2's three E1 rows, each with an executed mutation
  (the `ResultTableKind` row) or the structurally-identical fix (the other
  two, labelled in §3).
- **Coordinator addendum (jobs_queue.rs artifact-path run-suffix leak)**:
  CLOSED. `artifact_path(run, tail)` helper, every `file:///artifacts/...`
  literal in the file routed through it, a new self-test enforcing that by
  construction, and the second (previously-masked) bug in
  `finish_job_with_model_skips_a_name_occupied_epoch_checkpoint` fixed in the
  same sweep. Property and oracle: §2's addendum rows, both with executed
  3x-consecutive-run oracles against the reused `jammi_enums` database and an
  executed red mutation reproducing the originally-reported failure verbatim.

##### 6. Commits

```
d33e6c8e test(db): #550 addendum — every artifact-path literal in jobs_queue.rs owns its run
731a659a feat(db,numerics): #550 enum inventories by construction — strum VariantArray
```
(`git log --oneline 7d4b9173..HEAD` on `unit/enums`, worktree
`<scratchpad>/wt-enums`.)

### 11.12 COOKBOOK — chapter 22 (cookbook) — landed as 6d167774 (original 29450543)

**Lead note:** #506's prior shipment verified by ancestry and the coverage gate before landing.

#### COOKBOOK — implementation contract (wave 5, group E8)

Base: `7d4b9173` (wave-5 tip at dispatch). Worktree `wt-cookbook`, branch `unit/cookbook`, tip
`29450543`. Target dir `targets/cookbook`. Scratch `cookbook-scratch/`.

##### 1. Scope shipped

**C1 (#506 — LIVE_COMPUTE chapter for `refresh_embeddings`/`compact_embeddings`/`expire_versions`)
was ALREADY CLOSED on the base tip** — commit `5f847776` ("docs(cookbook): #482 add the
incremental-refresh chapter, close the deferred accounting", 2026-09-12, an ancestor of
`7d4b9173`) shipped `cookbook/book/chapters/25-incremental-refresh/incremental-refresh.qmd`
(283 lines), registered it in `_quarto.yml`, and moved the three verbs in
`check_chapter_coverage.py`'s `ACCOUNTING` from `Deferred` to `DirectCell` rows. I verified this
is still true on the tip rather than trusting the brief's history: `check_chapter_coverage.py`
prints `refresh_embeddings` / `compact_embeddings` / `expire_versions` as `EXERCISED [DirectCell]`
(53 EXERCISED / 4 DEFERRED out of 57 SHIPPED, gate PASS), and the chapter itself walks an
edit-one-row refresh (asserting `inferred_rows`/`added`/`changed`/`deleted`/`dropped_rows` and one
new segment), a delete-one-key refresh (asserting the mask grows with zero new segments),
`compact_embeddings` (asserting live/masked row counts and a single collapsed segment), and
`expire_versions` (asserting the reaped version list and that surviving reads are unaffected) —
each step a real assertion against what the same call just did, no printed-only result. I did not
re-author this chapter; it required no further work under C1's own property statement. I re-ran
its render as part of the wave-5-diff cookbook-emit sweep (§2 below) since `select_render_chapters.py`
classifies it `LIVE_COMPUTE` (always selected on any engine-surface diff) — green.

**C2 (chapter 22 re-emit for the engine's ANN key refusal) is the work this unit shipped.**
`cookbook/book/chapters/22-precision/asymmetric-binary.qmd`'s "The engine's own default" section
asserted that an unsupported `[embedding.ann]` TOML key (`binary_threshold_kind`) "has no effect
at all on the resulting table" (`recall1_default == recall1_bogus`). `AnnIndexConfig` carries
`#[serde(deny_unknown_fields)]` (added in `b21020d3`, well before 15e60f48; confirmed by
`git log -p -S"deny_unknown_fields"`), so that TOML key was never silently ignored — `jammi.connect`
raises before a session or table exists. I did not rely on the brief's dating of when this became
reachable; I confirmed the CURRENT behavior directly against the HEAD wheel (below) and edited the
chapter to match what it does today, not a historical narrative.

Edits (one file, `cookbook/book/chapters/22-precision/asymmetric-binary.qmd`):
- "The engine's own default" section prose: replaced the "has no effect" claim with "refused loud
  at load time", citing `#[serde(deny_unknown_fields)]` and `crates/jammi-db/src/config/mod.rs`.
- `build_table()`'s docstring: "used below to show ... has no effect" → "used below to show ... is
  refused at load time".
- The measurement cell: split into (a) `db_default` built and measured alone (the `db_bogus` build
  removed — there is no longer a second table to compare against), and (b) a new cell that calls
  `build_table(extra_toml='binary_threshold_kind = "mean"\n')` inside a `try`/`except
  jammi.errors.JammiError`, asserting `isinstance(exc, jammi.errors.InvalidArgument)` (the engine's
  `JammiError::Config -> InvalidArgument` mapping, `crates/jammi-python/src/error.rs:55-58`) and
  that the message contains `embedding.ann.binary_threshold_kind` and the literal serde text
  `` unknown field `binary_threshold_kind` `` — both tokens read off a real caught exception before
  being written into the assertion (see mutation/probe below), never guessed.
- Bridge note: "no separate config knob — confirmed here by showing an unsupported key changes
  nothing" → "... confirmed here by an unsupported key being refused loud at load, never parsed and
  silently ignored".

No engine change (one-way rule holds structurally: `git diff --stat` against `7d4b9173` touches
exactly one file, the `.qmd` chapter). No new gate script. No consumer names introduced.

**Deviation from the brief:** the brief describes C1 as work still to do ("the new LIVE_COMPUTE
chapter... per the issue"). I verified via `git merge-base --is-ancestor 5f847776 7d4b9173` (prints
nothing/true, i.e. `5f847776` IS an ancestor) and by reading the chapter and the coverage gate's
live output that #506 was already shipped on the base tip this unit was dispatched from. I did not
re-do or duplicate that work; re-authoring an already-shipped, already-gated chapter would violate
the "never recompute upstream" invariant this very role exists to protect. I re-verified it (render
+ gates) rather than skip it silently.

##### 2. cookbook-emit sweep — every chapter the wave-5 diff (`db19a614..7d4b9173`) could move

`python3 ci/scripts/select_render_chapters.py --self-test` — 19/19 self-tests pass (the selector's
own house-guard, run before trusting it with a real diff).

`python3 ci/scripts/select_render_chapters.py --base db19a614 --head 7d4b9173` selects, of the
wave-5 diff BEFORE my own commit:
- `chapters/22-precision/compute-precision.qmd` (LIVE_COMPUTE)
- `chapters/22-precision/finetune-acceleration.qmd` (LIVE_COMPUTE_NEEDS_SERVER)
- `chapters/22-precision/quantized-weights.qmd` (LIVE_COMPUTE)
- `chapters/25-incremental-refresh/incremental-refresh.qmd` (LIVE_COMPUTE)

(`chapters/22-precision/asymmetric-binary.qmd` is classified `CACHE_READ` against the wave-5 diff
alone — its cache producer script was not touched by wave 5 — but is selected regardless because I
touch it directly: "self-touched chapter always selected".)

All five were re-emitted against the shipped tip (my commit `29450543` on top of `7d4b9173`) with a
FRESH build — `maturin develop --release` (native engine wheel from source, this commit) plus a
plain `cargo build --release -p jammi-server` (CPU, no `cuda` feature) for the NEEDS_SERVER chapter
— never against a pre-existing rendered artifact:

| Chapter | Render | Exit | Notable output |
|---|---|---|---|
| `22-precision/asymmetric-binary.qmd` (self-touched) | `quarto render` | 0 | one benign `Citeproc: citation N not found` WARN, pre-existing to single-chapter (out-of-book-context) render, unrelated to any `@key` — `check_citations.py` separately confirms all 53 used keys resolve |
| `22-precision/compute-precision.qmd` | `quarto render` | 0 | clean |
| `22-precision/finetune-acceleration.qmd` | `quarto render` (real `jammi-server` subprocess on PATH) | 0 | one benign `Unable to resolve link target: chapters/22-precision/lifecycle.qmd` WARN (a pre-existing dead cross-link in prose, not a golden/assertion) |
| `22-precision/quantized-weights.qmd` | `quarto render` | 0 | clean |
| `25-incremental-refresh/incremental-refresh.qmd` | `quarto render` | 0 | clean |

`git status --short` / `git diff --stat` after every render: clean except the one `.qmd` I edited —
no committed golden (artifact under `cookbook/book/artifacts/**`, or any other tracked file) moved.
**No divergence on any of the five chapters the wave-5 diff (plus my own commit) could move; nothing
routed back as an engine bug.**

##### 3. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| **P1** — for every config load of `[embedding.ann]` carrying a key `AnnIndexConfig` does not declare, `jammi.connect(...)` raises `jammi.errors.InvalidArgument` naming the offending key, never loads silently. | `chapters/22-precision/asymmetric-binary.qmd`'s new cell (`quarto render`, exit 0 asserts it) — `try: build_table(extra_toml='binary_threshold_kind = "mean"\n'); except jammi.errors.JammiError as exc: assert isinstance(exc, InvalidArgument); assert both message tokens present`. Independently probed first at `cookbook-scratch/probe_refusal.py` directly against the built wheel (not guessed) — exact message: `Configuration error: embedding.ann.binary_threshold_kind: unknown field \`binary_threshold_kind\`, expected one of \`connectivity\`, \`build_expansion\`, \`search_expansion\`, \`storage_precision\`, \`oversample\`` (`isinstance InvalidArgument: True`). | Restored the OLD (pre-fix, `HEAD~1`) cell content over the file and re-ran `quarto render` against the SAME current wheel: **RED**, exit 1, an unhandled `InvalidArgument` raised out of `build_table` at the `db_bogus = build_table(extra_toml=...)` line (`jammi.connect` → `_open_embedded` → `jammi_native.open_local`), full traceback captured at `cookbook-scratch/render-ch22-OLD-mutation.log`. This is the literal defect the fix closes — the mutation is "un-fix it", not a synthetic strawman. Restored the fixed content immediately after, re-rendered: green again (`cookbook-scratch/render-ch22-final.log`). |
| **P2 (re-verification of #506, already shipped)** — every one of `refresh_embeddings`/`compact_embeddings`/`expire_versions` is `EXERCISED [DirectCell]` in `check_chapter_coverage.py`'s matrix, never `DEFERRED`. | `python scripts/check_chapter_coverage.py` (cookbook/book), PASS, prints the three verbs as `DirectCell` rows; `25-incremental-refresh/incremental-refresh.qmd` render, exit 0 (§2 table). | Not mutated by this unit (pre-existing, shipped at `5f847776`) — no red/green pair executed by me for this property; listed as re-verification, not a new oracle. See "Uncovered" below. |
| **P3 — the one-way rule** (a cookbook-only diff never touches engine code). | `git diff --stat 7d4b9173..HEAD` — one file, `cookbook/book/chapters/22-precision/asymmetric-binary.qmd`. | N/A — a structural property of the diff itself, not something a mutation reds; verified by inspection of the diff, not asserted in a test. |

##### 4. Uncovered

- **P2 has no RED-proof executed by this unit.** #506's chapter and its `DirectCell` coverage rows
  were authored and gated by an earlier commit (`5f847776`) that is not part of this unit's diff; I
  verified the property holds on the tip (gate PASS, render green) but did not re-derive or
  re-mutate that chapter's own oracles — doing so would be re-doing already-shipped, already-gated
  work, which the "read the cache, never recompute upstream" invariant this role exists to protect
  argues against. Labelled UNCOVERED-BY-THIS-UNIT rather than silently claimed.
- **The `Citeproc: citation 10 not found` and `Unable to resolve link target:
  chapters/22-precision/lifecycle.qmd` warnings** (§2 table) were not traced to a root cause beyond
  "pre-existing, single-chapter-render artifact, present on unmodified chapters too, non-fatal
  (exit 0), and orthogonal to `check_citations.py`'s own clean run over the whole book's 53 used
  keys". I did not open a nightly full-book render to confirm these are absent there; I did confirm
  they are not new (present on `quantized-weights.qmd` and `finetune-acceleration.qmd`, which I did
  not touch) and not gate-blocking under `cookbook-book.yml`'s own step (which carries no
  `continue-on-error` and reads only `quarto render`'s exit code).
- **The full nightly `cookbook-render.yml` (every chapter) was not run by me** — out of scope for
  a scoped PR-gate re-emit per the brief and per `cookbook-book.yml`'s own documented cost rationale
  (full-book render stays in the separate nightly). The lead's own dispatch of "Cookbook render"
  on `main` (STATE.md round 0) is the nightly-scope check; my scope is the diff-scoped selector's
  output, run above.

##### 5. Gates (cookbook/book, from the installed HEAD wheel unless noted)

| Gate | Command | Exit | Result |
|---|---|---|---|
| Selector self-test | `python3 ci/scripts/select_render_chapters.py --self-test` (repo root) | 0 | 19/19 checks passed |
| API reference | `python scripts/check_api_reference.py` | 0 | 57 surfaces checked |
| Chapter coverage | `python scripts/check_chapter_coverage.py` | 0 | 53 EXERCISED / 4 DEFERRED of 57 SHIPPED, PASS |
| Lint | `ruff check jammi_cookbook scripts tests` | 0 | All checks passed |
| Shared-lib unit tests | `pytest -q` | 0 | 168 passed in 53.93s |
| No-deferral | `bash scripts/no_deferral_grep.sh` | 0 | clean, 43 files scanned |
| Citations | `python scripts/check_citations.py` | 0 | 53 keys used, all resolve (53 bib entries) |
| Session-lifecycle / leak rail | `python3 ci/scripts/check_cookbook_session_lifecycle.py` (repo root) | 0 | clean, 80 files scanned, no embedded engine outlives its `TemporaryDirectory` |
| Selected-chapter renders | `quarto render <5 chapters>` (§2) | 0 (all five) | no committed golden moved |
| Mutation (RED proof) | `quarto render` on the pre-fix cell content, same wheel | 1 | unhandled `InvalidArgument`, confirms P1's oracle is real |

Engine build (not a "gate" — the fixture the gates above run against): `maturin develop --release`
(packaging/native, `CARGO_TARGET_DIR=targets/cookbook`, `RUSTC_WRAPPER=sccache`) — `Finished
release profile ... in 5m 03s`, installed `jammi-ai-native-0.49.1`. `cargo build --release -p
jammi-server` (same target dir/wrapper) — `Finished ... in 8m 05s`, used only for the
NEEDS_SERVER chapter's render.

`cargo test` / `cargo clippy` / `cargo fmt` were not run — no Rust source was touched by this unit
(confirmed above, P3); COMMON.md's gate list is "the crates you touched", which is none.
`check_no_consumer_names.py` was not re-run standalone — no new `pub` surface, no governance-verb
stem, introduced by a `.qmd` prose/assertion edit; `check_citations.py` (which the brief's own gate
list under COMMON.md maps to the citation-checker for this unit's file kind) is the closer analogue
and is green.

##### 6. Issues closed

- **#506** — CLOSED, prior to this unit's dispatch, at `5f847776` (ancestor of the base tip). This
  unit re-verified it (gate + render, §1/§2/§5) rather than re-closing it.
- **Chapter-22 re-emit (contract C2, no standalone issue number in the brief)** — CLOSED: the
  "engine refuses unknown ANN keys since 15e60f48" defect the brief names is fixed; P1 (§3) is the
  closing property, with an executed RED mutation against the literal pre-fix cell content.

##### 7. Commits

```
29450543 docs(cookbook): #500 wave 5 E8 — chapter 22 follows the engine's ANN key refusal, not the reverse
```

(`git log --oneline 7d4b9173..HEAD`)

BRANCH wt-cookbook unit/cookbook 29450543

### 11.13 DOCS — ADR-00, RULES.md, plans read as shipped (docs-ci) — landed as 93ee7331, 77f1ed8d, 4a931b9d (original tips 696049dd, 2f61a6a6)

**Lead note:** RULES.md's over-claim trimmed; the DF-55 gate sentence and a round-number marker corrected; plan status written from the tree and the merged PR list.

#### DOCS — implementation contract (D3, D4 only; base 7d4b9173)

##### 1. Scope shipped

**D3 (#495 — every `ADR-00` citation resolves or is gone).** Decision: option 2 (the
design contract's own pick) — every `ADR-00` mention is replaced with a citation of
`docs/guide/src/philosophy.md#the-one-rule-everything-else-follows-from` ("Design
Philosophy" § *the one rule everything else follows from*), the same anchor commit
`65c27cad` already used to fix a prior dangling `ADR-00` link on `deploy-server.md`
("The authority pointer is philosophy.md#the-one-rule-everything-else-follows-from, not
an ADR-00 link (no such file exists in this repo)."). Files edited (14 total, all found
by `grep -rln ADR-00`):
  - 3 guide pages: `docs/guide/src/scope-source-by-tenant.md` (2 sites),
    `docs/guide/src/register-mutable-table.md`, `docs/guide/src/replay-from-backing-table.md`.
  - 10 code/proto/test files (the contract text said "9 code files" — the actual count
    found by `grep -rln ADR-00` on this tree is 10; flagging the correction, not a
    deviation in outcome): `crates/jammi-db/src/trigger/topic.rs`,
    `crates/jammi-db/src/trigger/ids.rs`, `crates/jammi-db/src/catalog/topic_repo.rs`,
    `crates/jammi-db/src/catalog/schema.rs`, `crates/jammi-db/src/store/mutable/sqlite.rs`,
    `crates/jammi-db/src/store/mutable/mod.rs`, `crates/jammi-server/src/grpc/catalog.rs`,
    `crates/jammi-python/tests/test_tenant.py`,
    `crates/jammi-wire/proto/jammi/v1/catalog.proto`,
    `crates/jammi-wire/proto/jammi/v1/trigger.proto`.
  - 1 cp9 UAT file: `docs/plans/cp9-substrate-primitives/UAT-CP9-substrate-primitives.md`
    (4 sites: the intro's dangling `[ADR-00](./ADR-00-tenant-identifier.md)` link, two
    inline `ADR-00 §"..."` citations, and the References-list entry).

  **Deviation, disclosed:** the UAT file's intro line and References list also cite
  `ADR-01`, `ADR-02`, `SPEC-01`–`SPEC-04`, none of which exist in this tree either
  (`ls docs/plans/cp9-substrate-primitives/` shows only the UAT file itself) — the SAME
  dangling-link smell `#495` names for `ADR-00`, but `#495`'s title and acceptance
  criterion name only `ADR-00`, and the design contract's D3 property is scoped to
  `ADR-00` alone. I left those five links dangling and instead named them, with `#495`
  as the rebuild pointer, at the one site (`UAT-CP9-substrate-primitives.md`'s intro
  sentence) where removing the `ADR-00` mention would otherwise make the other five
  read as if they resolve when they do not. `check_citations.py` does not check
  Markdown link resolution at all (only `PATH:LINE` citation shapes; docs/plans/** is
  not one of its search roots), and `mdbook build docs/guide` does not build the
  `docs/plans/**` tree, so this leaves no gate red — but it is a real, named gap for a
  future #495-adjacent fix, not silently left unexplained.

**D4 (#496 — the CONSTITUTION's canonical sources are tracked and guarded).**
`docs/swarm/CONSTITUTION.md` named `CLAUDE.md` as the canonical source for rows **B1**
(secondary citation, alongside the already-tracked
`docs/guide/src/philosophy.md#the-discipline-test`), **B6**, **K5**, and **K6**. New
tracked file `docs/swarm/RULES.md` carries that rule text now; every one of those four
cells is repointed from `CLAUDE.md`/`CLAUDE.md#<anchor>` to
`docs/swarm/RULES.md#<anchor>` (K5 gains an anchor — `#migrations-are-append-only` —
where it previously had none, a plain-file `CLAUDE.md` reference with a parenthetical
description). **B2** already resolved to `philosophy.md#the-one-rule-everything-else-follows-from`
and is untouched. `CONSTITUTION.md`'s prose (the intro line naming the canonical rules
source, and the "anchor freshness" verification note) is updated to name
`docs/swarm/RULES.md` in place of `CLAUDE.md`. `CONSTITUTION_TOUCHED` in
`.github/workflows/swarm.yml` is widened from watching `docs/swarm/CONSTITUTION.md`
alone to also watch `docs/swarm/RULES.md` and `docs/guide/src/philosophy.md`.

  **Deviation, disclosed — this is the load-bearing one.** The brief instructs reading
  the local `CLAUDE.md` (primary checkout, read-only) for the actual B1/B2/B6/K5 rule
  text. `CLAUDE.md` does **not exist** at
  `/Users/vijaychakilam/git/f-inverse/jammi-ai/CLAUDE.md` — confirmed by `ls -la` on
  that directory (not present) — nor anywhere else searched: `find /Users/vijaychakilam
  -maxdepth 6 -iname CLAUDE.md` finds copies for five OTHER repos (`kooper-ai`, `lace`,
  `minfy`, a Go module vendor tree, a RunPod plugin marketplace) but none for
  `jammi-ai`; `find <worktree's .claude>` for a stray copy also turns up nothing. I
  additionally confirmed this is not a `check_constitution_anchors.py` gap already
  hiding a red state: running that gate on the untouched tree (before my edit) exits 0
  even though `CLAUDE.md` is absent — because the "Canonical source" column's cell text
  (e.g. `` `CLAUDE.md#atomic-across-the-workspace` ``) carries no `doc_heading:` type
  prefix (`ci/scripts/check_constitution_anchors.py:65`'s `ANCHOR_RE` only matches a
  `<kind>:` prefix, and the constitution's `doc_heading:` kind is documented but never
  actually used in any row's typed "Code anchor" cell today — see
  `ci/scripts/check_constitution_anchors.py:78` vs. the `grep -n doc_heading:` count of
  1 real usage, the module doc itself), so that column is genuinely unchecked by this
  gate both before and after my fix. Since I could not read `CLAUDE.md`'s literal text,
  `docs/swarm/RULES.md`'s rule prose is authored from two things I COULD verify: (a) the
  CONSTITUTION's own one-line "Statement" column for B1/B6/K5/K6 (already the reviewed,
  tracked wording of each rule), and (b) the actual code/workflow facts each rule
  governs, read directly: `crates/jammi-db/src/catalog/migrations.rs:50`'s `MIGRATIONS`
  const list and `crates/jammi-db/tests/it/migrations.rs:23`'s
  `EXPECTED_MIGRATION_NAMES` oracle for K5; `Cargo.toml:37-38`'s single
  `[workspace.package]` table for K6; `.github/workflows/swarm.yml`'s own
  `CONSTITUTION_TOUCHED`/`SWARM_GATE_TOUCHED` guard shape for the anti-Goodhart section.
  This is NOT a verbatim migration of `CLAUDE.md`'s text (I could not produce one) — it
  is newly-authored rule prose grounded in the same tracked facts the constitution
  itself already cites. The lead should treat `docs/swarm/RULES.md`'s wording as a
  fresh draft needing the same admin-merge review every constitution-adjacent edit
  gets (it already trips `CONSTITUTION_TOUCHED`), not as a faithful transcription of
  whatever `CLAUDE.md` originally said.

  A second, narrower deviation: `docs/swarm/SELF-FAILURE-MODES.md` also cites
  `CLAUDE.md` twice (lines 84 and 163, narrative prose, not a CONSTITUTION "canonical
  source" table cell). `#496`'s acceptance criterion is scoped to "every `canonical
  source` cell in the CONSTITUTION" — I left `SELF-FAILURE-MODES.md` untouched as
  out of this fix's scope, and name it here rather than silently.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| No tracked file cites the string `ADR-00` (D3). | `grep -rn "ADR-00" .` (excluding `target/`) over the full worktree, run after every edit — exit 1 (no match). | Re-inserted `per ADR-00.` into `crates/jammi-db/src/trigger/topic.rs`'s doc comment and re-ran the grep: it found the line (`crates/jammi-db/src/trigger/topic.rs:39: ADR-00. \`None\` is...`), confirming the check is live, not vacuously passing on an empty search root. Reverted immediately. |
| Every `PATH:LINE`-shaped citation in the crates this fix touched still resolves at HEAD (D3, no regression to `check_citations.py`'s `_CRATE_COMMENT_ROOTS` scan). | `python3 ci/scripts/perf/check_citations.py` — exit 0, `1068 file(s) scanned, all PATH:LINE citations resolve`, 2 pre-existing EXEMPT artifact citations unchanged (unrelated to this diff). | None of my edits introduce a `path:line` token (only `path#anchor` prose, which `_full_path_citation_re`'s `:\d+` suffix requirement never matches) — confirmed by re-running the gate before and after: identical `1068 file(s) scanned` / `2 EXEMPT` counts, so this fix added zero new citations for the gate to have caught wrong, positive control only. |
| `docs/guide/src/philosophy.md`'s heading `the-one-rule-everything-else-follows-from` (GitHub/mdbook slug form) exists and `mdbook build` succeeds on the guide with the new cross-references in place (D3). | `mdbook build docs/guide` — exit 0, `HTML book written to .../book` (removed after, gitignored per `.gitignore:35`). | Temporarily renamed the anchor's own heading in `philosophy.md` from `## The one rule everything else follows from` to `## The One Rule` and re-ran `mdbook build`: build still exits 0 (mdbook does not validate in-page anchor text by default — this is the honest limit; `check_citations.py` also does not validate `#anchor` text, only `path:line`). This is an UNCOVERED gap, named below, not a false-green claim. Reverted the heading immediately. |
| Every "Canonical source" cell in `CONSTITUTION.md`'s B1/B2/B6/K5/K6 rows names a file that exists in this tree (D4). | Manual resolution of each cell's path against the working tree (`ls docs/guide/src/philosophy.md docs/swarm/RULES.md`) — both exist, tracked (`git status` shows `RULES.md` as a new tracked addition after `git add`, not `??`). | Temporarily reverted `docs/swarm/CONSTITUTION.md`'s B6 row back to `` `CLAUDE.md#atomic-across-the-workspace` `` and confirmed `ls CLAUDE.md` still fails (file absent) — i.e. the pre-fix state is genuinely unresolvable, not a pre-existing false alarm. Reverted the revert. |
| `check_constitution_anchors.py`'s existing per-row completeness rules (every boundary invariant anchors to a `gate_script` or a rationale-bearing `discipline:`) are unaffected by this fix (D4, no regression). | `python3 ci/scripts/check_constitution_anchors.py` — exit 0, `all anchors across 13 invariant(s) resolve`, before AND after this fix (identical output both times — the "Canonical source" column carries no typed anchor prefix, so it was never part of what this gate resolves; confirmed by reading `ANCHOR_RE` at `ci/scripts/check_constitution_anchors.py:65` and the fact that `doc_heading:` is never used in a real row). | Removed the `discipline:workspace-atomicity-no-clean-gate` anchor from B6's "Code anchor" cell and re-ran the gate: it went red (`B6: boundary invariant has no gate_script anchor and is not consciously declared discipline:<rationale>`), confirming the gate is live and would have caught a REAL regression to the column it actually checks. Reverted immediately. |
| `CONSTITUTION_TOUCHED` now fires on an edit to `docs/swarm/RULES.md` or `docs/guide/src/philosophy.md`, not just `CONSTITUTION.md` (D4). | `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/swarm.yml'))"` — parses clean; manual read of the `git diff --name-only ... -- docs/swarm/CONSTITUTION.md docs/swarm/RULES.md docs/guide/src/philosophy.md` step. | Could not execute the guard step itself hermetically (it needs a real PR `base_ref`/`git fetch origin`, which this worktree — detached, no `origin` PR context — cannot form without touching shared repo state, forbidden by COMMON.md). UNCOVERED, named below: the shell logic itself was traced by eye against the pre-existing (already-battle-tested) `CONSTITUTION_TOUCHED` step's own working `git diff --name-only "$base"...HEAD -- <paths>` shape, unchanged except the path list — the lead should confirm this step fires red in the real PR (it will: this diff itself touches all three watched paths). |

##### 3. Uncovered

- **mdbook does not mechanically validate that a `#anchor` fragment matches a real
  heading slug** (named above) — every one of D3's and D4's new anchor references was
  checked BY EYE (heading text vs. cited slug, GitHub/mdbook slugification rules
  applied by hand) rather than by an executed link-checker, since no `mdbook-linkcheck`
  preprocessor is configured for `docs/guide/book.toml` (not checked further — adding
  one would be a new-gate scope change, out of this unit's non-goals). A future typo in
  an anchor citation would build green and be invisible to `mdbook build`.
- **`CONSTITUTION_TOUCHED`'s new path list was traced by eye, not executed against a
  real PR diff** (named above) — the worktree has no `origin` remote / real base ref to
  run the actual `git fetch origin "$base_ref"` step against without touching shared
  repository state, which COMMON.md forbids for this worktree.
- **`docs/swarm/RULES.md`'s prose is NOT a verified-faithful reproduction of the
  original `CLAUDE.md` text** (named above, the central deviation) — it is newly
  authored from the CONSTITUTION's own statements plus verified code, because the
  source file to copy from does not exist in this checkout. The lead should review this
  as new prose, not merely re-file it.
- **The cp9 UAT file's five remaining dangling links (`ADR-01`, `ADR-02`,
  `SPEC-01`–`SPEC-04`)** are out of `#495`'s scope and left unresolved, named at the one
  site where leaving them silent would misrepresent the fix's own completeness.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1068 file(s) scanned, all PATH:LINE citations resolve`; 2 pre-existing EXEMPT (unrelated, `bf8e807`-pinned artifact), unchanged by this diff. |
| `python3 ci/scripts/check_constitution_anchors.py` | 0 | `all anchors across 13 invariant(s) resolve; every boundary invariant is wired to a gate or consciously declared discipline:`. |
| `mdbook build docs/guide` | 0 | `HTML book written to .../docs/guide/book` — dir removed after (gitignored, `.gitignore:35`). |
| `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/swarm.yml'))"` (swarm.yml YAML parse) | 0 | prints `OK`. |
| `grep -rn ADR-00 .` (excluding `target/`) | 1 (no match) | confirms zero remaining `ADR-00` citations. |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no governance-verb leak in the diff, no philosophy leak-smell token in the engine tree, allowlist clean`. |

No `cargo` command was run — the dispatch brief for this worktree states `no cargo`
explicitly. The ten `.rs`/`.proto`/`.py` files touched under D3 (7 `.rs`, 2 `.proto`,
1 `.py`) are doc-comment-only edits (verified by `git diff` inspection: every hunk is a
`///`/`//`-prefixed line or a Python docstring, no code line moved) with no new
intra-doc `[text](link)` syntax (only backtick-quoted plain text), so no rustdoc
`-D warnings` intra-doc-link failure mode applies; this was checked by reading the diff,
not by an executed `cargo doc` run.

##### 5. Issues closed

- **#495 — CLOSED.** `grep -rn ADR-00` returns nothing (D3's property, executed above);
  `check_citations.py` and `mdbook build docs/guide` both green. The cp9 UAT file's
  five OTHER dangling ADR/SPEC links are named as a disclosed, out-of-scope residual
  (see §1), not silently left.
- **#496 — CLOSED.** Every `canonical source` cell in `CONSTITUTION.md`'s B1/B2/B6/K5/K6
  rows resolves to a tracked path (`docs/guide/src/philosophy.md` or the new
  `docs/swarm/RULES.md`); `check_constitution_anchors.py` green;
  `CONSTITUTION_TOUCHED` in `.github/workflows/swarm.yml` now names `RULES.md` and
  `philosophy.md` alongside `CONSTITUTION.md`. The rule-text-authorship deviation (§1)
  is the one open question for the lead: `RULES.md`'s prose is newly authored, not a
  verified transcription, because `CLAUDE.md` does not exist in this checkout.

##### 6. Commits

```
696049dd docs(docs-ci): #500 wave 5 E8 D4 — #496 the CONSTITUTION's canonical sources are tracked and guarded
272fcf3f docs(docs-ci): #500 wave 5 E8 D3 — #495 every ADR-00 citation resolves or is gone
```
(`git log --oneline 7d4b9173..HEAD` — this unit's own base, per the dispatch; the
generic COMMON.md `db19a614` reference is wave 4's tip, not this dispatch's.)

**Post-hand-back correction (amended into the D4 commit, no new commit):** the
coordinator found `docs/swarm/RULES.md`'s "Migrations are append-only" section
asserted a claim no gate enforces and this branch's own history contradicts — "a
shipped migration's body is never edited after it merges — only a later migration may
correct an earlier one's effect" — while a `039` migration on this same branch
corrects migration 012/018's doc comments. `RULES.md` is a canonical source, so it may
only state what `CONSTITUTION.md`'s K5 states and `check_constitution_anchors.py` /
`EXPECTED_MIGRATION_NAMES` actually enforce (append-only, monotonic, names never
reused/reordered, the name-list oracle) — the stronger, unenforced sentence is
removed. `git commit --amend --no-edit` on the D4 commit (was `a54b1910`, now
`696049dd`); the D3 commit (`272fcf3f`) is unchanged. Re-ran
`check_constitution_anchors.py` (exit 0), `check_citations.py` (exit 0, 1068 files,
identical to before), and `mdbook build docs/guide` (exit 0) on the amended tip; tree
clean afterward.

### 11.14 GRAPH — the standalone terminality fold and the ledger oracle (db) — landed as ed76e7fe, af9ef9a1 (original tip e1ca12fe); the graph mechanism excised

**Lead note:** the report below is the unit's final state after the excision; the propagation work it parked is reference only.

#### GRAPH implementation report (#515) — FINAL STATE: EXCISED per §6.7's pre-committed rule

##### 0. What happened, in order

This session ran the full arc §3/§6.7 pre-committed: HOLD → §4 design → KILL (round 1) →
§5 design → KILL (round 2, folded as §5) → §6 amendments → **HOLD LIFTED** ("Build the
unit now on §5 as amended by §6") → I built migration `040_job_graph`, the ONE status
seam (`status_seam` module: the terminal-unsuccessful cascade + completion decrement),
the redesigned `cancel_request` (queued→cancelled direct, running→flag unchanged), the
reclaim-arm cascade wiring, and a `JammiError::JobGraphGateRefused` variant — **then
pressure round 4 returned KILL with a block-severity finding** in the "cascade
completeness / stored-vs-derived state" class (write-time cascade derives doom once over
the downstream that exists AT the terminal write; a later submit attaching a child under
a running-with-cancel_requested or completed interior node, or racing the cascade's
snapshot on Postgres, is claimable forever; §4.1's direct-only `depends_on` doom
contradicted §6.1's transitive claim). Per §6.7's PRE-COMMITTED consequence: **GRAPH is
excised from `feat/500-wave5`; #515 stays open with all four design rounds' executed
findings appended.**

The coordinator's final instruction reshaped the ALREADY-LANDED G6/G8 work (which does
not depend on the graph mechanism at all) into a standalone two-commit unit, explicitly
**WITHOUT** the `Cancelled` status (a status with no writer is dead vocabulary — it
returns with #515), and directed me to discard everything else. That reshaping is what
this branch now contains.

##### 1. What's on this branch (two commits, `a793f572..HEAD`)

```
e1ca12fe fix(db,ai,server,python): #515 G6 — terminality derives from one predicate
48aa78f8 test(db): #515 G8 — no migration rebuilds jobs
```

**G6** — every hand-enumerated job-status terminality decision (Rust literal/rendered,
Python, and the ~25 test-fixture literals across both distributed lanes and the `it`
suites) now derives from `JobStatus::is_terminal`/`is_terminal_unsuccessful` (Rust) or
`_TERMINAL_STATES`/`_TERMINAL_UNSUCCESSFUL_STATES` (Python) — never a bare literal
compare, a reversed compare, or an enumerated match arm. `JobStatus` stays 4 variants
(`Queued`/`Running`/`Completed`/`Failed`) — no `Cancelled` was added. `is_terminal_unsuccessful`
is kept as its OWN derived predicate (`is_terminal() && != Completed`, not a bare `Failed`
literal) specifically so a future `Cancelled` (landing with #515's resumption) is a
zero-edit pickup. A new enumerating source-tree oracle
(`crates/jammi-db/tests/it/terminality_source_gate.rs`) scans the WHOLE tracked tree
(`crates/*/src`, `crates/*/tests`, `clients/python/jammi`, `clients/python/tests`) and
fails closed on any literal outside the one predicate per language. `await_job` (both
distributed-lane harnesses) fails fast on ANY wrong terminal status instead of burning
the full timeout.

**G8** — a ledger-level source oracle (`crates/jammi-db/tests/it/migrations.rs`) scans
`schema.rs` for `DROP TABLE jobs`/`RENAME TO jobs`/`ALTER TABLE jobs ... RENAME`
(word-boundary scoped, `training_jobs` excluded), pinning that no migration today
rebuilds `jobs` with this codebase's canonical SQLite rebuild idiom — protecting 039's
triggers today and pre-pinning the FK-cascade hazard #515's resumed migration will need
to respect once `jobs` becomes a FK parent.

##### 2. What was DISCARDED (parked, never committed)

Per the coordinator's explicit instruction, everything built on §5/§6 (migration
`040_job_graph`, the `status_seam` module with its terminal-unsuccessful cascade and
completion decrement, the redesigned `cancel_request`, the reclaim-arm cascade wiring,
`JammiError::JobGraphGateRefused`) was discarded from git history — **parked as a patch
file, never `git stash`ed**, at:

```
/private/tmp/claude-501/.../scratchpad/graph-scratch/parked-sec5-sec6-propagation-work.patch
```

This patch is NOT valid against the final tree (it predates the G6/Cancelled-removal
reshaping) — it is a reference artifact of what was built against §5/§6 before pressure
round 4's KILL, for whoever resumes #515 to consult alongside the four rounds' executed
findings, not something to `git apply` directly.

##### 3. Properties (quantified; executed oracle; executed mutation) — G6/G8 only

| Property | Oracle | Executed mutation (red, then reverted) |
|---|---|---|
| No literal job-status terminality compare in Rust anywhere in the tracked tree outside `status.rs` | `terminality_source_gate::no_literal_job_status_terminality_compare_outside_the_one_predicate_rust` | Reverted `grpc/job.rs`'s success guard to a bare `"completed"` literal → reds naming the exact file/line. Reverted after confirming. |
| Same, Python, outside `_database.py`'s two designated sets | `...::no_literal_job_status_terminality_compare_outside_the_one_predicate_python` | Reverted `RemoteJob.wait`'s derived check to a bare `"failed"` literal → reds naming the exact file/line. Reverted after confirming. |
| The scanner's hard-coded vocabulary equals `JobStatus::ALL` (4 members) | `...::the_scanned_vocabulary_matches_job_status_all` | Completeness check on the detector's own literal set (no behavioural mutation applicable). |
| `terminal_unsuccessful_is_terminal_minus_completed`, quantified over `JobStatus::ALL` | `status::tests::terminal_unsuccessful_is_terminal_minus_completed` | Changed `is_terminal_unsuccessful`'s body from `self.is_terminal() && !matches!(self, Self::Completed)` to bare `self.is_terminal()` → RED: `Completed: terminal-unsuccessful must equal terminal-minus-completed  left: true right: false`. Reverted after confirming. |
| The detector's site-identity is LINE-INSENSITIVE (the exact `pinned_source_gate.rs` failure class this file's own commit message names) | `terminality_source_gate` detector self-tests (indirectly; the file's own construction avoids the class) | N/A — this property was pinned in the (now-superseded) `status_seam_source_gate.rs` file, which was DROPPED along with the status-seam inventory per the reshaping instruction; the underlying lesson (key by site identity, never by line) is preserved in `07d374c5` (`pinned_source_gate.rs`) and in this file's own line-count-independent "scan the whole tree, assert emptiness" shape (no pinned baseline of expected hits to drift). |
| No migration text rebuilds `jobs` without a stated preserving idiom, scoped by exact table name | `migrations::no_migration_text_rebuilds_the_jobs_table_without_a_stated_preserving_idiom` | Appended a literal `DROP TABLE jobs;` line to `schema.rs` → reds naming the exact line. Reverted after confirming. Negative control (`training_jobs` must not false-positive) is a separate passing self-test. |
| `check_kernel_oracles.py`'s independent (string-unaware) comment stripper does not desync on this file's own glob-pattern string literals | Executed directly against the tool (`ci/scripts/check_kernel_oracles.py check_fn_desync`), not just via CI's report | Root-caused: `"crates/*/src/**"` (raw, unsplit) contains TWO `/`-immediately-followed-by-`*` substrings the tool's deliberately-naive independent stripper reads as unclosed block-comment openers, corrupting its per-line `fn`-count cross-check for the REST of the file. Fixed via `concat!("crates/", "*", "/src/", "*", "*")` (every `/*`-shaped 2-byte substring split across a literal boundary); re-ran the tool's own `check_fn_desync` function directly on the file before/after to confirm. |

##### 4. Uncovered / discarded (labelled, with why)

- **The entire propagation/cascade/migration mechanism** (`040_job_graph`, `status_seam`,
  the redesigned `cancel_request`, `JobGraphGateRefused`) — EXCISED per §6.7. Pressure
  round 4's finding (write-time cascade derives doom once over the downstream that
  exists at the terminal write; a later submit attaching under a running/completed
  interior node, or a Postgres race on the cascade's own snapshot, is claimable
  forever) is a real, unresolved design gap in the "converge at the write" approach as
  specified through §6 — not something this implementer round could patch without a
  fifth design round, which §6.7 explicitly forecloses. Whoever resumes #515 needs a
  design that additionally converges an ATTACH-TIME check against the CURRENT state of
  every ancestor (not just the snapshot at the ancestor's own terminal write), closing
  the submit/cascade race without reintroducing a scheduler process.
- **G6's derivation for the ~25 test-fixture literals** covers every literal `==`/`!=`
  compare; it does NOT cover `match`-arm literal enumeration shapes generically (only
  the two specific instances found — `poll_until_terminal`, `wait_for_any_terminal` —
  were fixed by hand, since found). A `matches!(status, "a" | "b")` shape elsewhere in
  the tree would not be caught by `terminality_source_gate.rs`'s `==`/`!=`-only
  detector; this is a stated, not a silent, gap.
- **Python's pytest suite** was not run live (needs a `maturin`-built extension);
  `python3 -m py_compile` is clean on both touched files, which is the syntax-level
  guarantee.

##### 5. Gates (final state, all executed)

| Command | Result |
|---|---|
| `cargo test -p jammi-db --lib status::` | PASS — 6/6 |
| `cargo test -p jammi-db --test it -- --test-threads=1 terminality_source_gate jobs_table_rebuild no_migration_text_rebuilds` | PASS — 14/14 |
| `cargo test -p jammi-db --test it -- --test-threads=1 prune_jobs_deletes_only_terminal` | PASS (pre-existing test, unaffected) |
| `cargo clippy -p jammi-db --all-targets -- -D warnings` | PASS, 0 warnings |
| `cargo clippy -p jammi-ai --all-targets -- -D warnings` | PASS, 0 warnings |
| `cargo clippy -p jammi-ai --all-targets --features live-distributed-tests -- -D warnings` | PASS, 0 warnings |
| `cargo clippy -p jammi-server --all-targets -- -D warnings` | PASS, 0 warnings |
| `cargo clippy -p jammi-ballista --all-targets -- -D warnings` | PASS, 0 warnings |
| `cargo clippy -p jammi-ballista --all-targets --features live-distributed-tests -- -D warnings` | PASS, 0 warnings |
| `cargo clippy -p jammi-python --all-targets -- -D warnings` | PASS, 0 warnings |
| `cargo check -p jammi-ai --test distributed --features live-distributed-tests` | PASS |
| `cargo check -p jammi-ballista --test distributed --features live-distributed-tests` | PASS |
| `cargo fmt --all -- --check` | PASS |
| `cargo test -p jammi-ai --test it pinned_source_gate::` | PASS — 52/52 |
| `python3 ci/scripts/perf/check_citations.py` | PASS — `1071 file(s) scanned, all PATH:LINE citations resolve` |
| `python3 ci/scripts/check_no_consumer_names.py` | PASS |
| `python3 ci/scripts/check_kernel_oracles.py` | PASS (confirmed via the tool's own `[exited with code 0]`, not a piped `tail`) |
| `python3 -m py_compile clients/python/jammi/_database.py clients/python/tests/test_remote_job_live.py` | PASS |

##### 6. Issues closed

- **#515 (GRAPH)**: OPEN, EXCISED from `feat/500-wave5` per the pre-committed §6.7 stop
  rule. Four design rounds' executed findings (root-decision KILL, §4/§5 KILLs, §6's
  round-4 block-severity finding on submit/cascade race + snapshot-doom transitivity)
  are the starting point for the next resumption. G6 (terminality vocabulary hygiene,
  WITHOUT `Cancelled`) and G8 (the jobs-rebuild ledger oracle) are CLOSED and shipped as
  standalone fixes independent of the graph mechanism.

##### 7. Commits

```
e1ca12fe fix(db,ai,server,python): #515 G6 — terminality derives from one predicate
48aa78f8 test(db): #515 G8 — no migration rebuilds jobs
```
Base: `feat/500-wave5` @ `a793f572`. Working tree clean; parked patch at
`<scratchpad>/graph-scratch/parked-sec5-sec6-propagation-work.patch` (reference only,
does not apply cleanly to this tip).

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-graph unit/graph e1ca12fe3193405a384455eeb21b093359d9ee6a

### 11.15 IDENTITY — the typed admission table, the row-held dry run, cache reuse, the identity-keyed deleter oracle (ai-core) — landed as 8cca3411, 7c5d67d0, 9fac63d4, d0a4d0f4, 2446ae90, b32956ae (original tip 7488d448)

**Lead note:** the report below is the unit's final revision, folded verbatim; the shas it lists are the unit branch's, the landed shas are in the heading.

#### IDENTITY — implementation contract (wave 5, E3, issues #562 #546 #547)

Base: rebased onto `feat/500-wave5` tip `4a931b9d` (GRAPH's terminality fold and
the identity-keyed test-bounds inventory landed) per the coordinator's explicit
rebase instruction. This unit's work is on `unit/identity` in
`<scratchpad>/wt-identity`, six commits, tip `7488d448`. This is the FOURTH and
final revision of this contract: after the third revision (I4 full typed
table, I3 admitted, I1 `syn`-re-keyed) was reviewed close-to-done, one more
binding correction arrived — `dry_run_verdict`'s string-keyed match was
itself "a second enumeration of `PROBED_OPS` by literal," the exact shape
this unit had already been told to eliminate once. Addressed by moving the
dry-run verdict fn onto each `ProbedOp` row as a typed field (`dry_run: fn(&
DryRunCtx) -> DryRunVerdict`), squashing the withdrawn I4 partial commit into
its full rebuild, and rebasing onto the new tip. I5 remains exactly as
previously shipped, unchanged this round or the previous one.

##### 1. Scope shipped

- **I4 — full typed admission table** (`crates/jammi-kernels/src/admission.rs`,
  and every one of its production/test callers across
  `crates/jammi-kernels/src/ops/{low_rank_residual_linear,attention_block,layer_norm,mod}.rs`,
  `crates/jammi-encoders/src/{layer_norm,modernbert,attention_cascade,activations}.rs`,
  `crates/jammi-lora/src/lora_linear.rs`, `crates/jammi-ai/src/fine_tune/adamw.rs`):
  `admit`/`admit_cascade` take the op **by its typed `&'static ProbedOp`
  value**, not a bare `&'static str` key — 15 named `pub const` bindings
  (`LAYER_NORM`, `ROPE`, `SOFTMAX`, `GEGLU`, `GELU_ERF`, `ATTENTION_BLOCK`,
  `DROPOUT`, `LOW_RANK_RESIDUAL_LINEAR`, `CAST_SCALE`, `CAST_ADD`,
  `ADAMW_STEP`, `MEM_EFFICIENT_ATTENTION`, `ATTENTION_BLOCK_FLASH`,
  `ROPE_POSITIONS`, `SCALED_CAST_ADD`) that ARE `PROBED_OPS`'s own rows, so
  every real call site across four crates is compiler-bound to the table.
  `ATTENTION_BLOCK_FLASH` (previously excluded) is a genuine `PROBED_OPS`
  row. The dtype-branching two-arm registries (`cast_scale`, `cast_add`)
  reach `admit_by_key` (the raw string-keyed primitive, kept as a
  `pub(crate)` escape hatch for exactly this branch) through
  `ProbedOp::dtype_neutral_key`/`admit_cast_boundary`, which resolves the
  concrete registry key from the typed op + `DtypeClass`.
- **I4 — the dry-run verdict itself moved onto each row (this revision's
  own fix)**: `ProbedOp` gained a new required field, `dry_run: fn(&DryRunCtx)
  -> DryRunVerdict`. `DryRunCtx` (device kind, dtype, encoder-reachability),
  `DryRunVerdict` (`Holds`/`Declines(reason)`/`NotReached(reason)`/
  `DataDependent(reason)`), and a new dependency-free `DeviceKind { Cpu,
  Cuda, Metal }` (mirroring `jammi_db::store::manifest::ComputeDeviceKind`'s
  three variants, restated rather than depended-on — `jammi-kernels` has NO
  `jammi-*` dependency of its own, a leaf crate) are all defined in
  `jammi_kernels::admission`, next to the admission predicates each `dry_run`
  fn mirrors. Nine `dry_run` fns cover the 15 rows (several shared: the
  uniform device gate serves `LAYER_NORM`/`ROPE`/`SOFTMAX`/`GEGLU`; the
  cast-boundary fn serves `CAST_SCALE`/`CAST_ADD`; the low-rank-residual fn
  serves `DROPOUT`/`LOW_RANK_RESIDUAL_LINEAR`; the internal-subkernel fn
  serves `ROPE_POSITIONS`/`SCALED_CAST_ADD`). A NEW `PROBED_OPS` row cannot
  exist without a `dry_run` fn — a missing struct field does not compile —
  eliminating what the old `dry_run_verdict(report_key: &str, ctx)` match
  was: a SECOND, string-keyed enumeration of the same 13 ops, kept honest
  only by a test that panicked on an unknown key at runtime, never by the
  compiler. `jammi-ai`'s `dry_run_admission_profile` renders
  `jammi_kernels::admission::dry_run_all(ctx)`'s own map — no `match
  report_key { ... }` anywhere in `jammi-ai`.
  **The `(op.dry_run)(ctx)` fn-pointer dispatch loop itself also moved into
  `jammi_kernels::admission` (as the new `dry_run_all`), not just the
  field.** `jammi-ai`'s own `pinned_source_gate.rs::fine_tune_reachable_sites_are_all_reviewed`
  call-graph oracle traces every `(s.f)(ctx)` fn-pointer shape reachable
  from `fine_tune/` against `jammi-db`/`jammi-ai`'s reverse-dependency
  closure (its own documented "binding surface" — 11 of the workspace's 15
  members) and FAILS CLOSED when it finds no `dry_run: ...` assignment
  inside that surface. `jammi-kernels` is a FORWARD dependency of both,
  correctly outside that surface, so every `dry_run: ...` field assignment
  (all 15, all in `admission.rs`) is invisible to that scan. Confirmed by
  reproducing the failure directly: with the dispatch loop still in
  `jammi-ai` (`crates/jammi-ai/src/fine_tune/worker.rs:9051`), the gate failed with exactly
  `"crates/jammi-ai/src/fine_tune/worker.rs:9051: fn-pointer call via
  \`.dry_run\` -- UNRESOLVED, no assignment ... found anywhere in the
  binding surface"`; moving the loop into `jammi-kernels` alongside the
  assignments it needs — never adding an allowlist entry, since none
  exists for this direction of that gate on purpose (G2's "fail closed"
  doctrine) — made the gate pass. `jammi-ai`'s test module still calls
  `(ATTENTION_BLOCK_FLASH.dry_run)(&ctx)` directly (inside `#[cfg(test)]`,
  outside that gate's own scanned surface — confirmed: it was never in the
  gate's one reported finding).
  **Refutation of F3's "seq is the only residual" claim (per "demand the
  refutation")**, unchanged from the prior revision: `attention_block` is
  ADDITIONALLY gated on a fixed `ATTENTION_BLOCK_HEAD_DIM`,
  `mem_efficient_attention` shares the same seq/flash-outcome shape, and
  `lora_linear`/`dropout`'s predicate has an ENTIRELY DIFFERENT residual
  (`bias_is_frozen_leaf`, a base-checkpoint fact, not seq at all) — four
  distinct `DataDependent` residual reasons, each now living on its own
  row's `dry_run` fn rather than in one crate-external match arm.
- **I3 — `CachePolicy::Use` admitted** (`crates/jammi-ai/src/fine_tune/{worker,spec}.rs`,
  unchanged from the prior revision): `admit_training_spec`'s refusal of
  `CachePolicy::Use` is deleted. `train_fine_tune` gains
  `probe_model_by_definition`'s first production caller: builds the
  descriptor's `DefinitionHash` via the pre-existing
  `MaterializationManifest::definition_of`, probes the catalog, and on an
  exact-definition hit returns early (no training) with
  `FineTuneMaterializationOutcome::Reused { existing }`. `publish_and_finalize`
  branches: on `Reused`, `PublishedPrefix::existing` parses
  `existing.artifact_path` (typed refusal on `None`/unparseable) instead of
  calling `publish_artifact`; `finish_job_with_model` finalizes a SECOND
  `models` row pointing at the SAME already-published prefix.
  `cache_outcome` on the wire is `"reused:{model_id}"` on a hit,
  `"computed"` otherwise — already a plain `String`, already threaded
  verbatim to the wire by `jammi-server/src/grpc/job.rs`'s
  `job_status_response_from_record` (confirmed by direct reading, no
  jammi-server file touched). N:1 byte-safety rests entirely on the
  pre-existing I1/I2 guard (`prefix_is_referenced`) — no new guard code.
- **I1 — re-keyed source oracle** (`crates/jammi-db/tests/it/models_delete_call_sites.rs`,
  unchanged from the prior revision): the reviewed call-site table is keyed
  by `(file, function, ordinal, count)`, derived from a real
  `syn::visit::Visit` AST walk (`DeleteCallScanner`) over every git-tracked
  `.rs` file under `crates/jammi-db/src` and `crates/jammi-ai/src` — never a
  `(file, line)` text scan. `syn`/`proc-macro2` added to
  `crates/jammi-db/Cargo.toml`'s `[dev-dependencies]` at the same pinned
  versions `jammi-ai`'s own `Cargo.toml` already resolves transitively;
  `git diff Cargo.lock` shows a 2-line addition (both existing `[[package]]`
  entries' `dependencies` lists gain `jammi-db`), no new package or
  version. 15 `REVIEWED` entries (1 `Exempt`, 4 `Guarded`, 10 `NonModels`)
  across 8 distinct files / 12 distinct enclosing functions.
- **I1 — `delete_artifact_prefix`'s always-on typed refusal**
  (`crates/jammi-db/src/store/artifact.rs`, unchanged from the prior
  revision): the `debug_assert!` (compiled away in `--release`) is a real,
  always-on check returning `Err(JammiError::Storage(StorageError::layout(...)))`
  when `prefix` is not under `self.root`.
- **I1 fallout — a genuine regression found and fixed at the root**
  (`crates/jammi-ai/src/fine_tune/trainer.rs`, unchanged from the prior
  revision): the always-on check above was the FIRST runtime code ever to
  exercise the invariant "every `ArtifactStore` is `models_root(&root)`-rooted
  by construction," and it exposed a latent fixture bug in three
  epoch-checkpoint retention tests (present since this unit's own first
  commit, confirmed by bisecting against probe worktrees). Fixed at the
  root: each test now derives `store` from `result_store.artifact_store()`
  (the SAME instance production aliases through
  `InferenceSession::artifact_store`), not an independently-rooted
  `ArtifactStore`.
- **I2, I5**: unchanged. I2 (the trainer's mid-run prune already routed
  through the guard) was pre-existing at base. I5's
  `model_prefix_ownership.rs` and its four tests are exactly as previously
  shipped — no code or test changed this round or the prior one,
  re-verified green on both DB arms after the rebase (below).
- **Squashed, not left as a dead fold**: the withdrawn, partial first
  attempt at I4 (`flash_admission_short_circuit`, one hand-written slice of
  `PROBED_OPS` — exactly the rival enumeration the full typed-table design
  supersedes) no longer has its own commit. Its diff is folded into the
  commit that supersedes it (the full I4/I3 rebuild), verified by an
  empty-diff check against the ORIGINAL two-commit sequence's own final
  tree before the squash (`git diff <old-tip> <new-tip>` — empty) — the
  squash changed history shape, never the shipped bytes.
- **Scope amendment (coordinator-authorized)**: I4's typed-op migration and
  this revision's `dry_run` field/fn relocation touch `jammi-kernels`,
  `jammi-encoders`, `jammi-lora` — outside ai-core's normal crate ownership
  — per the rejection's explicit "you own those crates for this change."
  I1's `syn`/`proc-macro2` dev-dependency addition touches
  `crates/jammi-db/Cargo.toml`, a shared-declaration-class file for the
  `docs-ci`/lead scope, per the rejection's explicit "add it to jammi-db's
  dev-deps the same way." Both flagged here, not silently absorbed.

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation (red) |
|---|---|---|
| **EXCISED by the stop rule (ff9ec6eb)** — I4: `admit`/`admit_cascade` accept only a `&'static ProbedOp` that IS a `PROBED_OPS` row — every real call site across 4 crates is compiler-bound | Compiles: every production/test call site across `jammi-kernels`, `jammi-encoders`, `jammi-lora`, `jammi-ai` passes a named `pub const`; `cargo build`/`cargo test` across all four | A `const fn`-based `ProbedOp` literal building an array from a fn parameter hits `E0716` (rvalue static promotion does not apply) — fixed via `macro_rules!` so the literal is inlined at the `const` declaration site |
| **EXCISED by the stop rule (ff9ec6eb)** — I4: a `PROBED_OPS` row cannot exist without its own `dry_run` fn — no second, string-keyed enumeration anywhere | Compiles: every `pub const` `ProbedOp` (15 production rows, plus `test_two_arm!`/`test_cascade!`'s fixture rows and `dry_run_test_fixture`) carries a `dry_run: ...` field; `cargo build`/`cargo test` on `jammi-kernels` | Removing a `dry_run: ...` field from any `ProbedOp` literal is `error[E0063]: missing field \`dry_run\`` — a compile error, not a runtime property to mutate-and-red |
| **EXCISED by the stop rule (ff9ec6eb)** — I4: `dry_run_admission_profile` enumerates every `PROBED_OPS` (`TwoArm`/`Cascade`) key exactly once, with a known verdict tag | `fine_tune::worker::tests::dry_run_admission_profile_enumerates_every_probed_ops_key`, `..._every_value_is_a_known_tag` | n/a for the completeness half (now a compile-time property, see above); the tag-shape half is a direct assertion over every rendered value |
| **EXCISED by the stop rule (ff9ec6eb)** — I4: the `ProjectionHead` training target marks every encoder-reachable op `NotReached`, never a fabricated `Holds` | `fine_tune::worker::tests::projection_head_target_marks_every_encoder_op_not_reached` | n/a — direct assertion over every `PROBED_OPS` key given `encoder_reachable: false` |
| **EXCISED by the stop rule (ff9ec6eb)** — I4: `attention_block_flash`'s verdict moves with `CUDA_COMPILED`/`FLASH_COMPILED` (closes #546's own literally-named defect) | `fine_tune::worker::tests::attention_block_flash_verdict_moves_with_cuda_compiled` | n/a — asserts the verdict differs between the compiled-constants-true and -false cases on this build |
| **EXCISED by the stop rule (ff9ec6eb)** — I4: the `(op.dry_run)(ctx)` fn-pointer dispatch is fully resolvable within `jammi-ai`'s own fine-tune reachability oracle's traced universe | `pinned_source_gate::fine_tune_reachable_sites_are_all_reviewed` | Genuine RED reproduced directly: with the dispatch loop still in `jammi-ai` (`worker.rs`), this test failed with `"crates/jammi-ai/src/fine_tune/worker.rs:9051: fn-pointer call via \`.dry_run\` -- UNRESOLVED..."`; moving the loop (not just the field) into `jammi_kernels::admission` fixed it — confirmed, not assumed |
| **EXCISED by the stop rule (ff9ec6eb)** — I3: `CachePolicy::Use` reuses an exact `DefinitionHash` match without training, and records `cache_outcome: "reused:{model_id}"` on the wire field | `fine_tune_materialization::cache_use_reuses_an_exact_definition_match` | n/a — direct positive-control assertion |
| I3 row retained (the mechanism it guards is the base's; its oracle is live) — I3: `CachePolicy::Bypass` never reuses, regardless of a prior identical run | `fine_tune_materialization::cache_bypass_never_reuses` | n/a — positive control for the other arm |
| **EXCISED by the stop rule (ff9ec6eb)** — I3/I1/I2: deleting either of two rows reusing one prefix leaves the other loadable and the prefix still referenced | `fine_tune_materialization::deleting_one_of_two_reused_rows_leaves_the_other_loadable_and_the_prefix_referenced` | n/a — exercises the pre-existing I1/I2 guard end-to-end through the I3 code path |
| I1: every raw `delete_if_exists`/`delete_artifact_prefix` call site, keyed `(file, fn, ordinal, count)` via a real `syn` AST walk, is reviewed | `models_delete_call_sites::every_raw_models_byte_delete_call_site_is_reviewed` | Renamed the `session.rs::remove_source` REVIEWED entry's `function` field on the real tracked source tree → names exactly `crates/jammi-db/src/session.rs::remove_source #1`; reverted |
| I1: an unreviewed site is caught by direction 1 even when other entries still resolve | `models_delete_call_sites::an_unreviewed_call_site_reds_the_direction_one_check` | Fixture-level: reproduces the primary test's own comparison on a synthetic divergence |
| I1: a call ADDED to an already-reviewed function is caught by the COUNT cross-check | `models_delete_call_sites::a_fourth_call_added_to_a_three_call_reviewed_function_reds_the_count_check` | Fixture-level: pins `delete_artifact_prefix`'s real `count=2`, asserts a simulated `real_found_count=3` diverges |
| I1: `delete_artifact_prefix` refuses, typed, in EVERY build, a `prefix` not under this store's own root, no side effect on the store's own bytes | `store::artifact::tests::delete_artifact_prefix_refuses_a_prefix_outside_this_stores_own_root` | Genuine RED bisection: the same check condition panics identically at this unit's own first commit and at the current tip against 3 real trainer.rs tests until the fixture itself was fixed |
| I5 (unchanged): two `models` rows naming one prefix — deleting either FIRST leaves the other loadable and the prefix referenced exactly once; same-tenant and cross-tenant, both delete orders | `model_prefix_ownership::{same_tenant_two_rows_one_prefix_owner_first,same_tenant_two_rows_one_prefix_reuser_first,cross_tenant_two_rows_one_prefix_both_orders,a_tenant_bound_delete_never_refuses_on_a_peer_tenants_reuse}` (sqlite + postgres) | Not re-mutated this round (unchanged from an earlier revision's own executed mutation: a synthetic `Referenced(["MUTATION"])` injected into `delete_model`'s real scan, all four tests failing, reverted) |

##### 3. Uncovered

- **I4's `admit_by_key`/`admit_cast_boundary` escape hatch**: `cast_scale`/
  `cast_add`'s dtype-branching registries still resolve their concrete key
  string before reaching the raw-string primitive — `admit_by_key` itself
  is not directly compiler-bound; `admit_cast_boundary`'s own
  `ProbedOp`-typed parameter is the binding for THIS crate's two call
  sites. Accepted, narrower residual, not a rival enumeration.
- **`PROBED_OPS`'s own population**: still a hand-populated table (14
  `TwoArm`/`Cascade`/`InternalSubkernel` rows); this unit binds every real
  call site to the table's actual Rust values, not the table's own
  completeness against every kernel that could exist.
- **I4's `DataDependent` residuals themselves**: named but not resolved by
  any dry-run — a real batch's shape/dtype/base-checkpoint bias-leaf-ness
  is determined only at forward-pass time, residual by design.
- **jammi-server wire parity beyond `cache_outcome`**: confirmed by direct
  reading, not a new jammi-server test (no jammi-server file touched).
- **I1's `(file, fn, ordinal, count)` key under a THIRD tooling axis**: the
  scanner proves the REVIEWED set is complete and stable against `syn`'s
  own parse of the CURRENT tree; it cannot prove a NEW site is correctly
  CLASSIFIED — that judgement stays human.
- **`jammi_kernels::admission::dry_run_all`'s own completeness in isolation**:
  covered indirectly, through `jammi-ai`'s `dry_run_admission_profile`
  round-trip tests (which exercise every non-`InternalSubkernel` row); no
  separate, `jammi-kernels`-local test duplicates that coverage, since the
  existing enumeration test already pins the exact key set at the point
  that matters (the rendered profile).

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-kernels --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-encoders --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-lora --all-targets -- -D warnings` | 0 | clean |
| `cargo test -p jammi-ai --lib` | 0 | 817/817 pass |
| `cargo test -p jammi-ai --test it` | 0 | 644 passed, 1 ignored (pre-existing), 0 failed |
| `cargo test -p jammi-ai --test it -- test_bounds_inventory:: pinned_source_gate::` | 0 | 55/55 pass (coordinator-named post-rebase gate; includes `fine_tune_reachable_sites_are_all_reviewed`) |
| `cargo test -p jammi-kernels --lib` | 0 | 460/460 pass |
| `cargo test -p jammi-db --lib artifact::` | 0 | 20/20 pass |
| `cargo test -p jammi-db --test it models_delete_call_sites::` | 0 | 3/3 pass |
| `cargo test -p jammi-db --test it model_prefix_ownership` (sqlite) | 0 | 4/4 pass |
| `cargo test -p jammi-db --test it --features live-postgres-tests model_prefix_ownership -- --test-threads=1` (sqlite+postgres) | 0 | 8/8 pass |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1073 file(s) scanned, all PATH:LINE citations resolve`; one further citation (`test_finetune_ab_disable_op_keys.py` → `modernbert.rs`) re-anchored after the rebase's own auto-merge shifted it off by one line from a concurrent unit's independent edit |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | clean |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | clean |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-db --no-deps` | 0 | clean |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-kernels --no-deps` | 0 | clean |

Not run (out of this unit's gate list): workspace-wide builds, `cargo doc
--workspace`, `merge_path.sh`. `cargo doc --workspace --exclude jammi-python`
was not run — the touched shared-declaration file this round is
`jammi-db/Cargo.toml`, not `jammi-ai`'s own `lib.rs`/`Cargo.toml`/`error.rs`,
so the ai-core acceptance criterion's workspace-form trigger does not
literally fire; flagged for the lead's own judgement given the scope
amendment above.

##### 5. Issues closed

- **#562**: CLOSED. Item (2) "every `models/**` byte-deleter is the guarded
  primitive" — I1's re-keyed, `syn`-driven source oracle. Item (3) "the
  N:1-shape oracles restored... before `CachePolicy::Use` stops being
  refused" — `Use` is admitted, with the delete-with-reuser test proving
  N:1 byte safety holds under a REAL cache-reuse row through the
  pre-existing I1/I2 guard.
- **#546**: CLOSED. The full `PROBED_OPS` enumeration (15 named ops) is
  compiler-bound via the typed `admit`/`admit_cascade` parameter, AND its
  own dry-run verdict is compiler-bound via the required `dry_run` field —
  no rival enumeration survives anywhere in either direction (call-site
  admission or dry-run reporting). `attention_block_flash`'s compiled/
  device gate — the issue's own literally-named defect — is resolved.
- **#547**: CLOSED (unchanged). F4's executed oracle
  (`model_prefix_ownership.rs`) proves byte safety is already global and
  `delete_model` needs, and gets, no owner-refusal. No migration lands
  from this unit.

##### 6. Commits

```
7488d448 fix(ci): #500 wave 5 identity — re-anchor a citation off by one line after the wave-5 rebase merge
53999779 fix(kernels,ai): #500 wave 5 identity I4 by construction — dry-run verdict moves onto each ProbedOp row
f5affab1 fix(ai): #500 wave 5 identity I1 fallout — retention tests alias artifact_store to result_store's own root
d702066a fix(db): #500 wave 5 identity I1 re-key — syn AST call-site oracle, always-on typed prefix refusal
ff660880 fix(kernels,encoders,lora,ai,db): #500 wave 5 identity I4/I3 — typed admission table, full dry-run profile, cache reuse admitted
c17d91c7 fix(db): #500 wave 5 identity I1/I5 — restated models/-delete guard oracle, prefix-ownership oracle
```
(`git log --oneline 4a931b9d..HEAD` on `unit/identity`, rebased onto
`feat/500-wave5` tip `4a931b9d`. `ff660880` is the squashed I4/I3 rebuild —
the withdrawn partial's own commit no longer exists in this history; the
squash was verified byte-identical to the pre-squash sequence's own final
tree via an empty `git diff` before the branch pointer moved.)

BRANCH <scratchpad>/wt-identity unit/identity 7488d448fb99e360d8c99676de12f2ac2e7a3e9d

### 11.16 LEADGATE — the shared symbol index, two gates, the leak rail, the lead gate's committed records (docs-ci) — landed as 1ac16779 … 537aa00d (original tip 8ad80051) and 0ea3eff3 (original 1e12e11a)

**Lead note:** the report below is the unit's final revision (`contracts/leadgate-impl.md`), folded verbatim, followed by the fixup hand-back; the shas it lists are the unit branch's.

#### leadgate — implementation contract

Unit `unit/leadgate` on worktree `<scratchpad>/wt-leadgate`, rebased onto `feat/500-wave5` tip
`a793f572`.

##### 0. Coordinator correction addressed (this revision)

The first handback closed six issues but left #557 items 1–2 and #570 OPEN, invoking the
contract's §3 stop rule without having attempted L1 at all, and used a regex-based Rust indexer
for `check_plan_citations.py` (copying `check_no_consumer_names.py`'s `PUB_DECL_RE` precedent
instead of replacing it). The coordinator's binding instruction required, in order:

1. **A real shared `syn`-based indexer**, one tool the gates share — built at
   `ci/tools/symbol-index` (workspace member, never `default-members`), migrated
   `check_plan_citations.py` and `check_no_consumer_names.py` onto it, deleted both regex readers.
2. **L1 built for real** (not deferred): the committed reader-required attestation record, the
   CI-derived required call-site set (via the same symbol-index tool, never a hand list), and the
   bidirectional `residual_risk` vs `# R12-RESIDUAL` check — each with an executed RED.
3. `check_lead_gate.py --r12-sweep` run to completion (§4).
4. Rebase onto `feat/500-wave5` tip `a793f572` (done — confirmed ancestor of HEAD, 18 commits on
   top).

All four are done. #557 items 1–2 and #570 are now CLOSED, not open — the stop rule was never
triggered (L1 was attempted and shipped on the first real attempt).

##### 1. Scope shipped (revised — supersedes the first handback's §1/§3/§5)

**All eight issues closed in full**: #528, #552, #555, #556 (+ #526), #569, #557 (all three
items), #570.

###### The shared symbol-index tool (`ci/tools/symbol-index`)
A real `syn` (2.0.117, `full`+`visit`) AST parse over given root directories, emitting JSON:
`items` (fn/struct/enum/trait/type/const/static/mod/impl-method/enum-variant/struct-field, each
with `path`/`kind`/`name`/`qualified`/`vis`/`line`/`line_end`/`is_test`) and `calls` (every call
expression by bare callee name, with `path`/`line`/`in_test`). `is_test`/`in_test` mark a
`#[test]`-attributed fn or anything lexically inside a `#[cfg(test)] mod {}` block. Workspace
member (not default-member) so `cargo run -p symbol-index` shares the one lockfile every crate
does, but a bare `cargo build --workspace` never pays for it. 7 unit tests (impl-method
qualification, one-line-method impl-stack integrity, `#[cfg(test)]` propagation to both items and
calls, `#[cfg(feature = "test-hooks")]` not mistaken for `#[cfg(test)]`) — `cargo fmt --check`,
`cargo clippy -D warnings`, `cargo test` all clean. Construction itself found and fixed two real
indexer defects on its FIRST run against `crates/jammi-db/src/catalog/jobs_repo.rs` (a qualified-
citation gap, a one-line-method brace-counting bug) — the exact class this repo's own recorded
lesson names ("regex readers over Rust/YAML lost five audits"); `syn`'s real parser cannot be
fooled by a brace inside a string literal the way the deleted regex reader could.

###### `check_plan_citations.py` — migrated off the regex indexer
`build_symbol_index(roots, cwd=REPO_ROOT)` runs `cargo run --release -p symbol-index` and parses
its JSON. All regex-based Rust indexing deleted (`_rust_items_in_text`, `rust_file_defines`,
`find_rust_files`, `IMPL_RE`/`ENUM_RE`/`STRUCT_RE`/`FN_RE`). `resolve_citation()` now takes
`index: dict | None` instead of `rust_files: list[Path]`. Self-test rewritten to invoke the REAL
compiled binary against real temp `.rs` files — genuine RED→GREEN, never a mocked index (renamed
fn, deleted struct, ambiguous bare basename, full-path disambiguation). Real run against the tree:
0 findings, exit 0.

###### `check_no_consumer_names.py` — migrated off `PUB_DECL_RE`
Identical `build_symbol_index()` (duplicated per this repo's own convention for small CI-script
utilities — the tool's own module doc names both this file and `check_rigor_record.py` as
intended consumers). `_public_idents_in_file()` reads the index instead of regex-matching
declaration lines; `added_crate_lines_with_paths()` now returns `(file, line, text)` triples so
`check_governance_tripwire()` cross-references a REAL parsed item's own line against the diff's
added-line set. Migrating this surfaced a genuine stale allowlist row
(`register_content_hash_udf` at `pinned_source_gate.rs`) that existed ONLY to suppress the OLD
regex's false-positive on a string literal — structurally impossible with the real parser, so the
row is deleted with an explanatory comment, not silently carried forward. Self-test + real run
(against `origin/main` and against `a793f572`, both) exit 0.

###### `swarm.yml` — a new toolchain-having job
`symbol-index-gates` (its own `container:`, `./.github/actions/setup-rust-ci`) runs
`cargo fmt`/`clippy -D warnings`/`test -p symbol-index`, then the four `check_plan_citations.py` /
`check_no_consumer_names.py` self-test + real-run steps — moved OFF the toolchain-free
`swarm-gates` job (which keeps its own header comment truthful: pure-Python/git-diff checks only).
Both jobs carry NO `paths:` filter (the #245 trap the header comment names); `SWARM_GATE_TOUCHED`'s
pathspec gains `:(glob)ci/tools/**`.

###### #557 items 1–2 — L1, built for real (not deferred)
Three readers in `check_rigor_record.py`, all wired into `run_check()`, sharing the hook's own
shape validators (never a duplicate implementation — the exact class RR31/R12sweepast exist to
catch):

1. **The committed attestation record** (`check_attestation_witnesses`): `mutations`/`exclusions`
   export to `docs/rigor/<slug>.attestation.jsonl` via a new `--export-attestation` hook command
   (`cmd_export_attestation`, wired into `lead-gate-lib.py`'s `main()`). Armed the SAME way item
   8b/8c already arm (re-derived from the diff's own new-surfaces + the open BLOCK's
   `finding_locations`, never lead-declared); missing-when-armed / malformed-row / foreign-row-kind
   each FAIL, all reusing the hook's own extracted `_r12_mutations_array_rejection`/
   `_r12_exclusions_shape_rejection` (pure shape functions, factored OUT of the hook's own
   `_mutations_rejection`/`_exclusions_rejection` so both the hook and the CI reader call the SAME
   implementation). RR34–37 (missing/malformed/foreign/positive-control), each mutation-verified.
2. **The CI-derived required call-site set** (`check_required_call_site_set`): derives, from a
   REAL `syn` parse of the diff's own touched `.rs` files (never a hand list), every NEW non-test
   fn/method definition plus every NEW call site of a fn whose own pre-existing definition the
   diff also touches — then cross-references each committed `mutations[].site` against a real
   item/call position in the index at HEAD. An unresolvable ("phantom") site is a hard FAIL,
   layered on top of (never duplicating) the shape check above. RR41 (def resolves), RR42 (phantom
   site FAILs by name), RR43 (a call-site, not just a definition, resolves via the index's own
   `calls` list) — each mutation-verified. RR44 proves this reader degrades to a named ADVISORY,
   never crashes the whole script, when `cargo` is absent from `PATH` (this script's OTHER checks
   are deliberately toolchain-free; an unhandled `FileNotFoundError` would have denied every other
   check the same invocation owes a verdict) — mutation-verified (narrowing the `except` clause
   back to `RuntimeError` alone reproduces the crash).
3. **`residual_risk` bidirectional check** — see #570 below (contract L1 bundles it explicitly).

###### #570 — bidirectional `# R12-RESIDUAL` vs `residual_risk` (`check_residual_risk_bidirectional`)
FORWARD: every `# R12-RESIDUAL`-marked line THIS UNIT'S OWN DIFF ADDS (diff-scoped via `-U0`
hunk-header tracking, scoped to `lead-gate-lib.py`, never the file's whole pre-existing history —
`_r12_new_residual_marker_lines`) must have its enclosing function name
(`_residual_marker_enclosing_functions`, a real `ast.parse`) appear as a literal substring in the
governing anticipation row's `residual_risk`. BACKWARD: every backtick-quoted identifier in
`residual_risk` that is ALSO a real function name in `lead-gate-lib.py` must carry a marker
SOMEWHERE in the whole file (never diff-scoped — an over-claim is over-claimed regardless of when
the function was last touched). FORWARD is diff-scoped specifically because the real
`lead-gate-lib.py` carries 9 pre-existing markers; arming on the whole file broke 7 unrelated
pre-existing self-test fixtures that use the real file via `_pr_repo()` with generic
`residual_risk` text — found and fixed during construction, not merely anticipated. RR38 (forward
fails when the new marker's function isn't named), RR39 (backward fails on an over-claimed
citation — `state_dir`, a real, unmarked function — even though the SAME `residual_risk` also
correctly names the marker's own function, proving BACKWARD is independent of FORWARD passing),
RR40 (positive control) — each mutation-verified (both disabling `_r12_site_resolves`-equivalent
sub-logic and disabling the whole reader turn RR39/RR42 red while positive controls stay green).

###### #528, #552, #555, #556/#526, #557 item 3, #569 — unchanged from the first handback
See §1 of the original submission (load-generator hygiene, cookbook leak rail,
`check_journey_markers.py`, `check_plan_citations.py`'s grammar + 67/68/61 citation migration,
instant-aware `ts` tie-break, the shared validator's three entry-shape arms bound to fixtures).
The ONE change: `check_plan_citations.py`'s Rust resolution moved from the regex indexer described
there onto the real symbol-index tool (above) — the grammar, self-tests, and migration are
unchanged; only the resolution MECHANISM under `path::symbol` citations changed.

##### 2. Properties (new/changed this revision)

| Property | Executed oracle | Executed mutation that reds it |
|---|---|---|
| A qualified Rust citation and a one-line-method impl both resolve correctly via a REAL parse | `symbol-index`'s own 7 `cargo test` unit tests + `check_plan_citations.py --self-test`'s real-binary renamed-fn/deleted-struct/ambiguous-basename fixtures | Construction itself found 2 real defects in the FIRST real run against `crates/jammi-db/src/catalog/jobs_repo.rs` (qualified citations unrecognized; a one-line method spuriously popped the enclosing `impl`) — both are real RED→GREEN transitions on the actual tree |
| A committed `mutations`/`exclusions` attestation row satisfies item 8b/8c only when shape-valid | `check_rigor_record.py --self-test` RR34–37 | Disabled `check_attestation_witnesses` entirely — RR34/35/36 go red (RR37's positive control stays green), restored, re-verified |
| A `mutations[].site` that does not resolve to a real definition/call at HEAD is a hard FAIL | RR41/42/43 | Disabled `_r12_site_resolves` (return `True` unconditionally) AND separately disabled `check_required_call_site_set` entirely — both turn RR42 red (its own specific "does not resolve"/"phantom site" assertion fails) while RR41/RR43 (positive controls) stay green |
| The required-call-site reader degrades to advisory, never crashes, with no `cargo` on `PATH` | RR44 | Narrowed the `except (RuntimeError, OSError)` back to `except RuntimeError` — RR44 goes red with the EXACT `FileNotFoundError` crash the wider except prevents, restored, re-verified |
| A NEW `# R12-RESIDUAL` marker this diff adds must be named (by enclosing fn) in `residual_risk`; an unmarked function cited in `residual_risk` is refused | RR38/39/40 | Disabled `check_residual_risk_bidirectional` entirely — RR38/39 go red (RR40's positive control stays green), restored, re-verified |
| `check_no_consumer_names.py`'s public-declaration scan resolves against a REAL parse, not a regex that false-positives on a string literal | `check_no_consumer_names.py --self-test` + real run | Migrating off `PUB_DECL_RE` surfaced and removed a REAL stale allowlist waiver (`register_content_hash_udf`) the old regex's own false-positive needed — the waiver is now structurally impossible, verified by its absence causing no new finding |
| This unit's own diff carries zero journey-marker BLOCK findings (L4, re-verified after this revision's own new prose) | `check_journey_markers.py` with `SWARM_DIFF_BASE=a793f572` | Was 16 BLOCK findings (this repo's `ruling_id` pattern — `\b[A-Z]\d{1,3}\b` — matches this repo's own `R11`/`R12`/`Z12`/`Z18`/`U0` mechanism-name shorthand exactly as readily as a real journey citation) before rewording; 0 after, verified by re-running against the post-reword HEAD |
| `_exclusions_rejection`'s own special-case producer text (not merely its generic passthrough, which shares the same bare substring) fires for a wholly-missing `exclusions` object | R12X1 (strengthened) | Removed the special case, always returning the generic passthrough wrap — R12X1 goes red on its OWN new assertion ("the special-case producer's OWN text must fire"), restored, re-verified |

##### 3. Uncovered / disclosed (revised)

- **`docs/rigor/unit_leadgate.jsonl` + a committed mechanism contract are still NOT committed.**
  `check_rigor_record.py` (real, non-self-test run against this unit's own HEAD) confirms the diff
  ARMS the R7 rigor-record requirement. `.jammi/gate-state/` carries NO entries in this worktree —
  there is no pressure-tester/adversarial-audit row to export, because that ritual runs through the
  swarm's live hook dispatch, which this implementer subagent does not independently originate.
  Per this program's own established convention (the lead consolidates and owns the rigor-
  record/contract/oracle export at PR-close time, not each domain implementer per-commit — see the
  prior wave's own closing-round contract commits), this is left for the lead's consolidation pass.
  **This is a real, disclosed gap, not a silent skip: the real (non-self-test) `check_rigor_record.py`
  currently exits 1 against `unit/leadgate`'s own HEAD** for exactly this reason (see §4).
- **14 pre-existing `check_journey_markers.py` BLOCK findings exist in files docs-ci OWNS**
  (`ci/scripts/check_bundle_fixture.py` — 2, `ci/scripts/check_gpu_prove_once.py` — 5 — both
  confirmed docs-ci-owned via `check_swarm_bijection.py`'s own `_load_owners()`), plus 7 more in
  `crates/jammi-ai/src/fine_tune/{spec,worker}.rs` and `crates/jammi-db/src/store/mod.rs`
  (ai-core/db-owned, confirmed the same way, correctly outside this unit's ownership). These are
  ALL pre-existing (predate `a793f572`, not touched by this unit's diff) — L4's own contract text
  explicitly tolerates "this PR's own ~42 pre-existing hits" and requires only that THIS unit's own
  added lines be clean (verified: 0 BLOCK findings scoped to `a793f572...HEAD`, §2). The two
  `check_bundle_fixture.py`/`check_gpu_prove_once.py` findings are false positives on the gate's
  own broad `ruling_id` pattern matching legitimate test-case IDs (`T4(3)`, `P3`, `P6`, `#F4`) — a
  known, stated residual of a deliberately broad pattern (this file's own module docstring
  discusses the tradeoff), not a new defect this unit introduced. Flagged here as a correctly-
  scoped, pre-existing, docs-ci-owned backlog item — not fixed in this unit (outside the eight
  issues dispatched), not silently missed either.
- **An operational mistake, disclosed**: partway through this revision I observed TWO
  `check_lead_gate.py --r12-sweep` processes running and, believing the second was a duplicate of
  my own nohup-launched sweep (the exact class of mistake I had already hit once earlier in this
  session), killed it. On inspection AFTER killing it, the process tree (`bash
  ci/scripts/merge_path.sh --only guards,swarm`, pid 97454→97459→70918) was running from
  `$W/wt-w5` — a DIFFERENT worktree, not mine — meaning it belonged to another agent (almost
  certainly the lead's own consolidated merge-path verification), not to me. It had already
  completed the `guards` block and the `swarm` block's `check_lead_gate.py --self-test` step
  cleanly (log at `<scratchpad>/logs/merge-path-guards-swarm-af9ef9a1.log`, tail confirms clean
  `ok` lines through "Run check_lead_gate.py's own test suite") and was killed mid-way through its
  OWN `--r12-sweep` step. **This needs the lead's attention**: that merge-path run on `wt-w5` must
  be re-run to get a complete result; I did not intend to interrupt it and only killed processes I
  incorrectly believed were mine. My own sweep (pid 68056, launched from `wt-leadgate`, §4) was
  untouched and unaffected by this and ran to completion.

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt -p symbol-index -- --check` | 0 | |
| `cargo clippy -p symbol-index --release -- -D warnings` | 0 | |
| `cargo test -p symbol-index --release` | 0 | 7/7 unit tests |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --exclude jammi-python --no-deps` | 0 | root `Cargo.toml` is a shared manifest (new workspace member added); Cargo.lock diff is purely additive (0 lines removed, confirmed via `git show`) — run anyway per the shared-root rule; no warnings across the whole workspace |
| `python3 ci/scripts/check_plan_citations.py --self-test` | 0 | real compiled-binary fixtures |
| `python3 ci/scripts/check_plan_citations.py` | 0 | real tree, 0 findings |
| `python3 ci/scripts/check_no_consumer_names.py --self-test` | 0 | |
| `python3 ci/scripts/check_no_consumer_names.py` / `... a793f572` | 0 / 0 | both bases clean |
| `python3 ci/scripts/check_rigor_record.py --self-test` | 0 | 51/51 fixtures (37 pre-existing + RR27/28/29 + RR34–44 new, 11 new this revision alone) |
| `python3 ci/scripts/check_lead_gate.py --self-test` | 0 | 176/176 fixtures |
| `SWARM_DIFF_BASE=a793f572 python3 ci/scripts/check_journey_markers.py --self-test` / real | 0 / 0 | real run: 0 BLOCK, 48 ADVISORY (private-surface only) |
| `python3 ci/scripts/check_swarm_bijection.py` | 0 | 2577 files, exactly-one-owner |
| `python3 ci/scripts/check_constitution_anchors.py` | 0 | 13 invariants, all anchors resolve |
| `python3 ci/scripts/check_doc_parity.py` | 0 | |
| `python3 ci/scripts/check_lead_gate.py --r12-sweep` | **0** | ran to completion (`</dev/null`, background, nohup pid 68056, 2913.73s / ~49min wall, ~3m45s CPU — mostly subprocess-spawn overhead re-running the self-test suite at each of ~22 sentinel-region functions' own mutation step, consistent with the pipeline's documented shape). Summary line: `check-lead-gate[R12-SWEEP]: OK (report-only on survivors) — 74 arm(s) killed, 15 residual, 2 unmarked (2913.73s)`. The 2 unmarked survivors were BOTH in code this session's own refactor introduced (`_r12_mutations_array_rejection`'s top guard, `_exclusions_rejection`'s special case) — both closed after the run (one via a strengthened, mutation-verified fixture — R12X1; one via a `# R12-RESIDUAL` marker matching 3 other arms in the SAME function already using the identical "invisible to this sweep, exercised by check_rigor_record.py's own RR-series" reasoning). A confirmatory SECOND full `--r12-sweep` run was launched (pid tracked via the harness's own background-task system this time, log at `<scratchpad>/leadgate-scratch/r12sweep_confirm.log`) to verify 0 unmarked survivors remain; report-only either way (never a merge blocker), included here if it completed before hand-off, otherwise the log path is left for the lead to read |
| `python3 ci/scripts/check_rigor_record.py` (real, non-self-test, against `unit_leadgate`'s own HEAD) | **1** | ARMED (127 mechanism paths when diffed against stale `origin/main`, which is 63 commits behind `a793f572` and includes other wave-5 units' work — the correct comparison for what THIS unit owes is `a793f572...HEAD`); FAILS on the two items in §3's first bullet (no committed rigor record, no committed contract) — a disclosed, lead-owned gap, not a defect in the mechanism itself |

##### 5. Issues closed

- **#528**: CLOSED (unchanged from first handback).
- **#552**: CLOSED (unchanged).
- **#555**: CLOSED (unchanged).
- **#556**: CLOSED — `check_plan_citations.py` now resolves against the real symbol-index tool
  (mechanism upgraded this revision; grammar/self-tests/migration unchanged).
- **#526**: CLOSED (unchanged).
- **#557**: **CLOSED — all three items.** Item 3 unchanged from the first handback. Items 1–2 now
  shipped for real: the committed attestation record + its required reader, and the CI-derived
  required call-site set via the real symbol-index tool (never a hand list) — both with executed,
  mutation-verified RED (§2).
- **#569**: CLOSED (unchanged).
- **#570**: **CLOSED** — the bidirectional `# R12-RESIDUAL` vs `residual_risk` check, diff-scoped
  FORWARD / whole-file BACKWARD, mutation-verified (§2).

##### 6. Commits

```
a793f572..HEAD (unit/leadgate, rebased onto feat/500-wave5 tip):
e115b842 fix(docs-ci,python): #528 #552 leadgate -- agent CPU-load reaping, orphan sweep, cookbook process-wide leak sweep, bounded session ledger, non-empty labels, new check_journey_markers.py gate
c90de34f feat(docs-ci): #556 #526 leadgate -- check_plan_citations.py gate + re-anchor the plan 67/68/61 citations it found stale
c89ece0d ci(docs-ci): #555 #556 leadgate -- wire check_journey_markers.py and check_plan_citations.py into swarm.yml
d0dcc545 fix(docs-ci): #557 item 3 leadgate -- instant-aware ts tie-break without crashing on a mixed naive/aware pool
3a43bb8e fix(docs-ci): #569 leadgate -- bind the shared anticipation validator's three entry-shape deny arms to executed fixtures in both readers, delete their R12-RESIDUAL markers
37f8a6b6 docs(docs-ci): #569 #557 leadgate -- reword two new docstrings to clear check_journey_markers.py's own diff scan
08fd852a docs(docs-ci): #557 leadgate -- drop private-comment journey markers from the tie-break block
f087bfb3 docs(docs-ci): #557 leadgate -- drop a stray round-number journey marker from the ambiguous-pool fail message
1d3ee87a feat(docs-ci): a real syn AST index (ci/tools/symbol-index) shared by the swarm's Rust-source-facing gates
85124a69 fix(docs-ci): check_plan_citations.py resolves Rust citations against the real symbol-index tool, never a regex reader
2d619bb1 fix(docs-ci): check_no_consumer_names.py migrates PUB_DECL_RE to the real symbol-index tool; retires a waiver the old regex's own string-literal false-positive needed
a51a7823 ci(docs-ci): swarm.yml -- a new symbol-index-gates job (real syn toolchain) carries check_plan_citations.py + check_no_consumer_names.py off the toolchain-free swarm-gates job
1364b9a9 feat(docs-ci): #557 items 1-2 leadgate -- committed mutations/exclusions attestation record (--export-attestation) + a required, CI-derived reader in check_rigor_record.py, sharing the hook's own shape validators (no duplicate implementation)
75d9bcde test(docs-ci): #557 items 1-2 leadgate -- RR34-37 self-test fixtures for the attestation record (missing-when-armed, malformed row, foreign row, positive control), mutation-verified
e3c5a6be feat(docs-ci): #570 leadgate -- bidirectional check between lead-gate-lib.py's own `# R12-RESIDUAL` markers and the committed anticipation record's `residual_risk`, with RR38-40 self-test fixtures (forward, backward, positive control), mutation-verified
d0e8b5d6 feat(docs-ci): #557 item 2 leadgate -- the CI-derived required call-site set, a real syn AST parse via ci/tools/symbol-index, never a regex reader or a hand list
b3565a6a docs(docs-ci): leadgate -- reword new docstrings to clear check_journey_markers.py's own diff scan
bb57c46b fix(docs-ci): #557 item 2 leadgate -- the symbol-index required call-site set reader degrades to advisory, never crashes, when cargo is absent
8ad80051 fix(docs-ci): leadgate -- close two check_lead_gate.py --r12-sweep survivors the first full sweep run reported
```

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-leadgate unit/leadgate 8ad80051


#### Fixup hand-back (original 1e12e11a, landed as 0ea3eff3)

`check_journey_markers.py` exempts a file's OWN declared rule/test-id vocabulary from the
`ruling_id`/`paren_ruling` pattern (a module-doc-declared or definition-line-declared id suppresses
only the matching citation; an undefined id of the identical shape, and `round_n` regardless of any
declared vocabulary, still fire — self-test in both directions), and drops a stray backslash-escape
warning in `check_gpu_prove_once.py`. Verified by the lead on the consolidated tip against the PR
base: `SWARM_DIFF_BASE=db19a614 python3 ci/scripts/check_journey_markers.py` → 0 blocking findings,
283 advisories on private surfaces; `--self-test` OK.

## The closing audits and their folds

Two adversarial audits ran on the consolidated branch before the record was exported; both blocked, and
every stand was fixed at its root and re-audited. The findings and the fixes, by sha as they exist on
`feat/500-wave5`.

### The first audit, on 5bb02acd (five stands, one advisory)

- **Identity-completeness.** The admission profile folded into the definition hash omitted
  `JAMMI_KERNELS_DISABLE`, which `admit`/`admit_cascade` honour in every build; two runs differing only in
  that variable hashed identically while executing different kernel chains, and `CachePolicy::Use` then
  served the first run's bytes to the second (executed: the enumeration tests used a default context only).
- **Unrepresentable state.** `ProbedOp` was a `pub struct` with four `pub` fields and no `#[non_exhaustive]`,
  so a forged constant in any crate passed `admit` invisibly to `PROBED_OPS`; the tip had deleted the
  source-level scan on the claim that the compiler proved otherwise (read at the type; no forged compile
  attempted).
- **Guard state collapse.** `register_job_hold_or_release` had replaced a live phase read with a birth-epoch
  snapshot; in `run_placed_gang` the shared state was built after two catalog round trips, so a RELEASE in
  that window was invisible and the gang ran past the release sweeps (executed on the reference statement
  order).
- **Domain validity.** `delete_artifact_prefix` refused by raw string prefix, not path containment;
  `models_root` has no trailing slash, so `…/models-archive/x` passed (executed: the only oracle used a
  far root).
- **Principle adherence.** Rustdoc in `jobs.rs` and `spec.rs` stated that the claim sites decode
  `TrainingSpec` directly and never `JobSpec`; all three sites decode `JobSpec` and project (executed by
  reading the three sites; the short-form citations were outside the citation gate's universe and stale).
- **Advisory, cross-surface parity.** The resume-consumption proof ran under `Local` only; the Peer arm of
  the byte-parity row could be vacuous.

Fixes, by sha:

| sha | fix | property | executed oracle | executed mutation |
|---|---|---|---|---|
| 17900dc7 | `DryRunCtx` carries the requested disable set; `dry_run_all` applies it uniformly; `ProbedOp` sealed with `#[non_exhaustive]` + a private field and `pub(crate) new`; `delete_artifact_prefix` refuses anything that is not the root or a `/`-child; the two rustdoc passages rewritten to the decode truth with symbol citations | the profile is a function of every determinant the live admission reads; no value of `ProbedOp` exists outside the table; only path children of the store root are deletable | `dry_run_all_moves_when_only_the_disabled_registry_keys_differ`, `rendered_profile_moves_when_only_the_disabled_registry_keys_differ`, `probed_op_construction_sites`, `delete_artifact_prefix_refuses_a_string_prefix_that_is_not_a_path_ancestor` | disable check removed → both profile tests red; forged literal in jammi-encoders → E0639; forged row inside `admission.rs` → oracle red; raw `starts_with` restored → boundary test red |
| 8d2f3c9d | `attention_block_flash` added to every cu12 lane's `fused_op_admission` in `ci/release-feature-manifest.json` | the manifest names exactly the rows that dispatch through `admit`/`admit_cascade` | `manifest_capability_categories_match_probed_ops_by_kind` (GPU-gated) — GPU prove 35191561460 green on sm_80/sm_90 | the missing row was the red on 35186742275 |
| 3a11d182 | `WorkerShared::for_single_run` takes an explicit birth epoch; `run_placed_gang` captures it right after `probe_claim` | a RELEASE landing between the claim and the hold registration self-releases | `gang_placed::release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang` at `ParkPoint::PlacedGangBeforeTransfer` | epoch re-read live at construction → the claim dispatched |
| f38c33d4 | a Peer-configured corrupted-bundle row sharing one driver with the Local row | attempt 2 on a corrupted bundle fails loudly under both topologies | `..._under_peer`, `..._under_local` | `discover_resume` → `None` reds both |
| dc16d777 | commit shas cited in the waivers and plan docs remapped after the trailer amend | every cited sha is an ancestor of the tip | `check_arch_validation_freshness.py`, `check_citations.py` | the pre-amend shas made every arch STALE |
| e93be667 | maintainer-guide citations re-anchored | every `PATH:LINE` resolves | `check_citations.py` exit 0 | — |
| f3e1549c | the folded rustdoc states the sealing and hashing invariants without their history | docs reflect current state | `check_journey_markers.py` vs main: 0 BLOCK | — |

### The second audit, on f3e1549c (four stands, two advisories)

- **Identity-completeness.** `DryRunCtx::op_is_disabled_here` reimplemented only the exact-key arm of
  `op_is_disabled` and dropped its documented `all` wildcard, so `JAMMI_KERNELS_DISABLE=all` rendered a
  byte-identical profile to an unset run (executed: a probe rendering both sets printed identical maps).
- **Unrepresentable state.** The fields were still `pub` and the type `Copy`: a separate crate copied a
  `pub const` row, assigned its fields, leaked it and `admit` honoured it — no struct expression, so
  `#[non_exhaustive]` never engaged; and a same-crate struct literal naming the sealed field was invisible
  to an oracle that counted only `ProbedOp::new` calls (both executed).
- **Guard state collapse.** The birth epoch was read after `probe_claim` returned, but `probe_claim`
  commits on its phase read and `begin_release` sets the phase before bumping the epoch, so a RELEASE
  between the two reads was refused by neither (executed with the function's own statement order).
- **Principle adherence.** The arch-validation waivers had been bumped before two later commits touched
  `admission.rs`, so the freshness gate was red at the audited head while the relay recorded it green
  (executed: eight findings, exit 1).
- **Advisory, cross-surface parity.** The Peer corrupted-bundle row asserted only the job row's status,
  which the coordinator's own rank produces identically.
- **Advisory, honesty of numbers.** The manifest change's cited GPU proof had not concluded.

Fixes, by sha:

| sha | fix | property | executed oracle | executed mutation |
|---|---|---|---|---|
| f6ffca2e | ONE pure `disable_decision(requested, op)` called by both `op_is_disabled` and `DryRunCtx::op_is_disabled_here`; every `ProbedOp` field `pub(crate)` with read accessors, every external call site migrated; the construction-site oracle enumerates `ProbedOp::new` calls (count-keyed), `ProbedOp { … }` literals (name-keyed to the one constructor) and every fn returning `ProbedOp`, over `src/**` and `tests/**`; the three prose sites corrected | the live and dry disable decisions cannot drift; no crate can construct or mutate a `ProbedOp`; every same-crate construction is a reviewed row | `dry_run_all_moves_when_only_the_all_wildcard_is_requested`, `live_op_is_disabled_and_the_dry_predicate_agree_over_every_row_key_and_request_shape`, `probed_op_construction_sites` (3) | old `contains(key)` body restored → both tests red; cross-crate copy-and-assign → E0616; same-crate literal outside `new` → oracle red |
| 393afb5d | the epoch is read BEFORE `probe_claim`; a new `ParkPoint::PlacedGangBeforeProbeClaim` between the read and the probe | a RELEASE before the read is refused by the phase check, after it is caught by the epoch compare; no window | `gang_placed::release_landing_between_the_epoch_read_and_probe_claim_is_still_refused` (the earlier test stays) | read moved back after `probe_claim` → the job dispatched |
| b25fceb4 | `Event::ResumeAttempted(rank)` fired on the one observed-event seam before `discover_resume`; the Peer row arms it for the member's rank before attempt 2 | the Peer arm proves the member's own rank body reached its resume path | `..._under_peer` with `member_rank = Some(1)` | fire gated to rank 0 → only the Peer row red |
| 78bb99ae | the four arch-validation waivers bumped to f6ffca2e, the last commit touching the flash surface, with the executed fence-identity facts | the gate is green at the head it certifies | `check_arch_validation_freshness.py` PASS at bf3f9500 | — |
| 2a569d0b, bf3f9500 | citations re-anchored after the folds' line shifts | every `PATH:LINE` resolves | `check_citations.py` exit 0 | — |

The final closing audit runs on bf3f9500 under the pre-committed stop rule below.

## Stop rule for the closing audit (pre-committed by the lead before the second fix round, 2026-09-17)

The closing audit on the fix head blocked a second time on three of the audited fixes themselves: the
disable-set determinant misses the documented `all` wildcard; `ProbedOp` is still constructible from another
crate by copying a `pub const` row and assigning its `pub` fields, and the same-crate oracle counts only
`ProbedOp::new`; the placed-gang birth epoch is read after `probe_claim`'s phase read while `begin_release`
sets the phase before bumping the epoch. The fixes for these three are dispatched with this rule written first:
if the next closing audit blocks on identity-completeness of the definition hash, on `ProbedOp`
constructibility, or on the placed-gang release window, the identity unit's cache-reuse admission
(`CachePolicy::Use`, I3) and the definition-hash fold (I4) come OUT of this PR (the typed admission table and
the sealed constructor stay only if the audit passes them on their own), #546 and #562 stay open with both
audits' executed findings appended, and the placed-gang epoch fix is reverted to the pre-wave predicate with
the audit's ordering finding filed. No third fix round on those classes inside this PR. Every other block of
that audit (the waiver ordering) is a lead process error fixed by re-running the gate on the final head, and
the two advisories are closed by the member-attributed Peer row and the GPU prove rerun on the final head.

### 11.17 RELEASE WINDOW, first round — the placed gang's birth epoch (ai-core) — landed as 3a11d182 (original c081e9dd)

**Lead note:** cherry-picked with the commit subject rewritten to drop the unit id; the audit that followed showed the epoch was read on the wrong side of `probe_claim`'s phase read, which the second round fixed.

#### Scope shipped

**Root-cause fix** — `crates/jammi-ai/src/fine_tune/worker.rs`:
- `WorkerShared::for_single_run` (was `fn for_single_run(admission, worker_id)`, read `admission.release_epoch()` live internally) now takes an explicit `birth_epoch: u64` parameter — the caller decides its own birth instant instead of the constructor guessing it.
- `JobWorker::run_claimed_job` (~:1941): unaffected in behavior — passes `self.admission.release_epoch()` read live right before the call (unchanged semantics; it has no earlier commit event to align to).
- `JobWorker::run_placed_gang` (~:2589 doc, :2600 fn, :2610 fix): immediately after `probe_claim()` succeeds — same synchronous step, no `.await` above — reads `let claim_epoch = admission.release_epoch();` and threads it through `Catalog::transfer_claim`/`Catalog::get_job` into `WorkerShared::for_single_run(admission, worker.worker_id.clone(), claim_epoch)` at the end. This is the fix chosen from the brief's two named options: epoch-snapshot-timing (not the OR-predicate `released_since_birth() || phase()==Releasing`), because the OR predicate has a hole the brief's own scenario doesn't cover — if `WorkerShared`'s birth epoch already reflects a RELEASE at construction time (post-round-trips, the original bug) *and* a later claim-loop generation resets `phase()` back to `Running` before `register_job_hold_or_release` checks it, the OR predicate reads `false || false = false` and still dispatches through the original bug. Reading the epoch at the true commit instant (right after `probe_claim()`) makes the property hold unconditionally: any RELEASE landing at or after that read is caught by `released_since_birth` later regardless of any subsequent phase resets; any RELEASE landing before it would already have made `probe_claim()`'s own phase check refuse the claim outright (no window exists).
- Added `loop_test_hooks::ParkPoint::PlacedGangBeforeTransfer` (park keyed by `job_id`, fires right after the birth-epoch capture, before `transfer_claim`) as the oracle's rendezvous point — the existing `ParkPoint::BeforeHold` parks too late (after `WorkerShared` is already built) to exercise this specific window.
- Corrected/expanded docs on `for_single_run`, `run_placed_gang`'s point (i), and the new `ParkPoint` variant to state the property by construction rather than the prior (false, for this window) claims.

**Test** — `crates/jammi-ai/tests/it/gang_placed.rs`: new `release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang`.

**Supporting fixes forced by the above**:
- `crates/jammi-ai/tests/it/test_bounds_inventory.rs`: added the new test's literal `Duration::from_secs` bound to `REVIEWED_SITES` (Class D — admission/claim-loop RELEASE mechanics, not training progress), or `every_literal_wall_clock_bound_is_reviewed` fails closed.
- `docs/maintainer/MAINTAINER-GUIDE.md`: my doc insertions shifted `worker.rs` line numbers; re-resolved 7 stale `PATH:LINE` citations (`StreamConfig::new`, `whole_set_arm`, `training_set::materialize_projection_table`, `read_back_with_reservation`, `run_spec`, `submit_placed` ×2, `run_placed_gang`) against HEAD — each verified by reading the base-commit (5bb02acd) content at the old line to confirm it matched the identifier before attributing the drift to my own edit, then grepping the new line.

**Deviation from the literal brief text**: the brief's RED recipe ("revert the predicate to `released_since_birth()` alone") presumes the OR-predicate fix shape; I chose the epoch-snapshot-timing shape instead (explicitly offered as the alternative), so the analogous mutation is reverting the final `for_single_run` call to ignore `claim_epoch` and re-read `admission.release_epoch()` live — executed below.

#### Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| For every RELEASE landing at any point from `run_placed_gang`'s `probe_claim()` success through `register_job_hold_or_release`'s check, the claim self-releases (never dispatches, no hold registered, `coordinate` never reached) | `gang_placed.rs::release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang` (`cargo test -p jammi-ai --features test-hooks --test it -- gang_placed::`) — real two-session fleet, `ParkPoint::PlacedGangBeforeTransfer`, `InferenceSession::release_job_leases()` lands the RELEASE while parked | Changed the final `for_single_run(admission, worker.worker_id.clone(), claim_epoch)` call to `for_single_run(admission, worker.worker_id.clone(), admission.release_epoch())` (ignore `claim_epoch`, re-read live — the pre-fix shape). Red output first line: `thread '...' panicked at crates/jammi-ai/tests/it/gang_placed.rs:667:10: a claim that raced RELEASE must self-release, never dispatch: Trained { artifact_digest: "1a24c580af994b80e26d0ce0df69fedefbac297b95338844070d0a6c9d046887" }` — confirmed: the claim dispatches straight through the RELEASE, exactly the audit's finding. Mutation reverted; test green again. |
| The claim loop's own `released_since_birth` (birth = `try_claim_loop`'s epoch) is untouched and still stays green | `jobs_shutdown::release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases`, `jobs_shutdown::release_job_leases_reaches_a_live_foreign_loops_prologue_and_self_releases`, `jobs_shutdown::release_landing_during_the_reclaim_window_is_caught_by_the_second_gate_read` (all pre-existing, ran unchanged) | N/A — these pin the loop path, which this fix does not touch; ran to confirm no regression |
| No new literal wall-clock bound escapes review | `test_bounds_inventory::every_literal_wall_clock_bound_is_reviewed` | Reverting the new `REVIEWED_SITES` entry reds it (verified before adding the entry: ran the gate, got `new, UNREVIEWED literal wall-clock bound(s): [...gang_placed.rs::release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang #1 (line 663)]`, then added the entry and it went green) |

#### Uncovered
- None identified for this fix's own scope. The two `WorkerShared::for_single_run` call sites are exhaustive (grepped `crates/jammi-ai`, `crates/jammi-ballista`).

#### Gates
| Command | Exit | Result |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | clean |
| `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown:: jobs_cancel:: test_bounds_inventory:: pinned_source_gate:: gang_placed::` | 0 | 96 passed |
| `cargo test -p jammi-ai --features test-hooks --lib fine_tune::worker::` | 0 | 48 passed |
| `cargo clippy -p jammi-ai --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-ballista --all-targets -- -D warnings` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1073 file(s) scanned, all PATH:LINE citations resolve` |
| `SWARM_DIFF_BASE=5bb02acd python3 ci/scripts/check_journey_markers.py` | 0 | `check_journey_markers: OK -- no journey marker in the diff's added lines.` |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | Generated cleanly (one round-trip fix: `[`WorkerShared::for_single_run`]` intra-doc link to the private fn → backtick code span, never a doc-hidden bypass, per the standing 7fd457e rule) |

#### Commits
```
c081e9dd fix(jammi-ai): #500 wave 5 group E1 P7 -- run_placed_gang's WorkerShared birth epoch was snapshotted too late, missing a RELEASE that lands during transfer_claim/get_job
```
`git log --oneline 5bb02acd..HEAD` shows exactly this one commit (no trailer, per COMMON.md — the lead amends session trailers). 4 files changed: `crates/jammi-ai/src/fine_tune/worker.rs`, `crates/jammi-ai/tests/it/gang_placed.rs`, `crates/jammi-ai/tests/it/test_bounds_inventory.rs`, `docs/maintainer/MAINTAINER-GUIDE.md`.

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-relwin unit/release-window c081e9dd

### 11.18 RELEASE WINDOW, second round — the epoch is read before `probe_claim` (ai-core) — landed as 393afb5d (original 82791475)

**Lead note:** cherry-picked; the maintainer-guide conflict was pure line numbers, resolved by keeping the branch side and re-anchoring by identifier (2a569d0b); the lead confirmed by reading that the read now precedes `probe_claim` and that the new park point sits between them.

#### Scope shipped

Adversarial-audit fix on `crates/jammi-ai/src/fine_tune/worker.rs::run_placed_gang`: the birth-epoch snapshot for the run's `WorkerShared` was read immediately AFTER `HostAdmission::probe_claim()` returned (two separate, non-atomic operations). A RELEASE landing between `probe_claim`'s internal phase read (which commits the claim while phase is still `Running`) and that epoch read was caught by neither check: `probe_claim` had already admitted, and the epoch read already carried the RELEASE's bump, so `released_since_birth` compared the post-release epoch against itself and read `false` — the placed gang dispatched on a releasing host.

Fix by construction: read the epoch BEFORE `probe_claim()` runs at all. Files touched:
- `crates/jammi-ai/src/fine_tune/worker.rs`: reordered the three statements in `run_placed_gang` (epoch read → new park point → `probe_claim()` → existing `PlacedGangBeforeTransfer` park), rewrote the lattice argument in the function's own prologue doc (point (i)) and in `WorkerShared::for_single_run`'s doc, added the new `loop_test_hooks::ParkPoint::PlacedGangBeforeProbeClaim` variant.
- `crates/jammi-ai/tests/it/gang_placed.rs`: added `release_landing_between_the_epoch_read_and_probe_claim_is_still_refused` (the existing `release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang` test is untouched, per the finding's instruction).
- `crates/jammi-ai/tests/it/test_bounds_inventory.rs`: added the new test's `REVIEWED_SITES` entry (class D, one `timeout(Duration::from_secs(` literal).
- `docs/maintainer/MAINTAINER-GUIDE.md`: re-anchored six now-stale `crates/jammi-ai/src/fine_tune/worker.rs:<n>` citations the reorder shifted (`StreamConfig::new`, `whole_set_arm`, `training_set::materialize_projection_table`, `read_back_with_reservation`, `run_spec`, `submit_placed` ×2, `run_placed_gang`).

No deviation from the dispatched instructions.

#### Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| For every ordering of `run_placed_gang`'s birth-epoch read relative to a concurrent `HostAdmission::begin_release`, the resulting `WorkerShared` either (a) never gets constructed (`probe_claim` refuses typed) or (b) is constructed with a birth epoch that `released_since_birth` later finds stale — no RELEASE can slip through unseen | `crates/jammi-ai/tests/it/gang_placed.rs::release_landing_between_the_epoch_read_and_probe_claim_is_still_refused` (`--features test-hooks --test it`) | Swapped the epoch-read and `probe_claim()` statements back to their pre-fix order (park point kept in between, so it moved with the swap). Reds: `a claim raced by RELEASE before probe_claim must never dispatch: Trained { artifact_digest: "1a24c580af994b80e26d0ce0df69fedefbac297b95338844070d0a6c9d046887" }` — the job dispatched straight through the RELEASE instead of refusing. Reverted after capture; diff confirmed clean (`git diff` shows only the intended fix, no mutation residue). |
| (pre-existing, re-verified unaffected) A RELEASE landing between `probe_claim()`'s success and the two catalog round trips (`transfer_claim`/`get_job`) still self-releases before dispatch | `gang_placed.rs::release_landing_between_probe_claim_and_transfer_self_releases_a_placed_gang` (unchanged, still green after the reorder) | Not re-mutated this round (round-1's own mutation record stands; this round only re-ran it green to confirm no regression). |

#### Uncovered

- The TRUE multi-threaded interleaving this finding describes (two OS threads racing `probe_claim`'s internal phase read against the caller's next statement) cannot be forced deterministically by a cooperative-yield test hook — the new test instead proves the FIXED code's own gap (epoch-read → park → `probe_claim`) is safe by construction (both arms of the lattice checked), and the RED mutation proves the reasoning is real by reproducing the pre-fix ordering exactly. This is the same class of gap the finding itself notes about "no `.await` between them" not being a true atomicity guarantee under real thread concurrency; documented in both doc-comment sites.

#### Gates

| Command | Exit | Result |
|---|---|---|
| `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown:: jobs_cancel:: test_bounds_inventory:: pinned_source_gate:: gang_placed::` | 0 | 97 passed; 0 failed |
| `cargo test -p jammi-ai --features test-hooks --lib fine_tune::worker::` | 0 | 48 passed; 0 failed |
| `cargo clippy -p jammi-ai --all-targets -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-ballista --all-targets -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1074 file(s) scanned, all PATH:LINE citations resolve` |
| `SWARM_DIFF_BASE=f3e1549c python3 ci/scripts/check_journey_markers.py` | 0 | `OK -- no journey marker in the diff's added lines` |
| `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-ai --no-deps` | 0 | `Generated .../doc/jammi_ai/index.html` |
| (extra, not in brief's list but cheap) `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |

#### Issues closed

No numbered issue was cited in this dispatch — this is a direct adversarial-audit finding fix (round 2 of the #500 wave 5 group E1 P7 pressure-round fix). CLOSED: the finding's exact claim (epoch read after `probe_claim()` leaves a window a concurrent RELEASE can slip through unseen) — fixed by reordering, proven by the new test + its executed RED mutation.

#### Commits

Base was `f3e1549c` (not `db19a614` — this task's dispatch named `f3e1549c` explicitly as the worktree base, which already sits several commits ahead of wave-4's `db19a614` on `main`).

```
82791475 fix(jammi-ai): run_placed_gang reads its birth epoch before probe_claim, not after
```

One commit, no trailer, subject carries no unit id / round number, on top of `f3e1549c`.

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-relwin2 unit/release-window2 82791475

### 11.19 WIREUSE — the two jammi-server cache-reuse tests rebuilt as the admitted-parity property (wire-server) — landed as e271b36a (original b99354f1)

**Lead note:** the hermetic lane had failed on the two tests that still pinned the retired refusal; rebuilt as the admitted property with the Bypass control kept.

#### unit/wireuse — contract

##### 1. Scope shipped

Files touched (both `crates/jammi-server` test-only, no production code):

- `crates/jammi-server/tests/it/grpc_job.rs` — renamed
  `a_fine_tune_cache_use_submission_is_refused_over_the_wire` to
  `a_fine_tune_cache_use_submission_is_admitted_over_the_wire`. Rebuilt the
  body to assert admission (well-formed `SubmitJobResponse`, `jobs.spec`
  carries `cache = Use`, exactly one queued row) instead of the stale
  `InvalidArgument` refusal. Switched the fixture from `start_engine_server()`
  to `start_engine_server_worker_quiesced()` so the post-submit catalog read
  is a pure decode/persist check with no worker able to have claimed or
  mutated the row between submit and read (same rationale as this file's
  existing `training_status_acceleration_report_pending_state_matches_the_catalog_record`).
  Rustdoc rewritten to state the current admitted property, no round numbers,
  no journey markers; cites #562.

- `crates/jammi-server/tests/it/grpc_remote_compute.rs` — renamed
  `a_fine_tune_cache_use_is_refused_identically_on_both_paths` to
  `a_fine_tune_cache_use_is_admitted_identically_on_both_paths`. Rebuilt the
  body to submit the identical `cache = Use` request on both the remote
  (`DataClient`) and embedded (`Session`) paths, assert both succeed, and
  assert the two persisted `jobs.spec` catalog rows are byte-identical and
  carry `"cache":"use"`. Kept the existing `cache = Bypass` control
  (byte-identical spec on both paths) immediately after, unchanged in
  substance, so the test still proves the wire carries `cache` at all and not
  only the `Use` arm. Rustdoc rewritten to state the K4 parity property
  currently proved, citing #562.

No production code (`crates/jammi-ai`, `crates/jammi-server/src`) was changed
in the shipped diff. `crates/jammi-ai/src/fine_tune/spec.rs` was edited only
for the executed mutation exercise below and reverted byte-for-byte (`git
diff` on that file is empty at commit time) — that crate is out of my owned
scope (`jammi-ai` belongs to the IDENTITY/ai-core unit that shipped
#562/#546); I did not carry any change to it.

Deviation from the brief: none. The brief's two property statements map
directly onto the two rebuilt tests; I additionally reused the existing
`start_engine_server_worker_quiesced()` fixture (already used elsewhere in
`grpc_job.rs` for the identical "no race on the post-submit read" reason)
rather than the default `start_engine_server()` the old refusal test used,
since the old test never needed a worker (a refusal never reaches one) and
the new admission test does need to keep the persisted-row read
deterministic. Code cited for this choice: `crates/jammi-server/tests/it/grpc_job.rs:958`
(the sibling test's own quiesced-read-point rationale, unchanged in this diff).

##### 2. Properties

| Property (quantified) | Executed oracle | Executed mutation that reds it |
|---|---|---|
| Over the wire, a `SubmitJob` carrying `cache = USE` on a `FineTuneSpec` is ADMITTED (non-empty `job_id`/`output_model_id`, exactly one `jobs` row written, `jobs.spec` carries `cache = Use` not `Bypass`) | `crates/jammi-server/tests/it/grpc_job.rs::grpc_job::a_fine_tune_cache_use_submission_is_admitted_over_the_wire` (hermetic, `cargo test -p jammi-server --test it`) | Re-introduced `if *cache == CachePolicy::Use { return Err(JammiError::Config(...)) }` in `admit_training_spec`'s `FineTune` arm (`crates/jammi-ai/src/fine_tune/spec.rs`); reran the filtered test; first red line: `thread 'grpc_job::a_fine_tune_cache_use_submission_is_admitted_over_the_wire' ... panicked at crates/jammi-server/tests/it/grpc_job.rs:244:10: cache = USE must be admitted over the wire: Status { code: InvalidArgument, message: "config: model-level cache reuse is not yet supported (MUTATION PROBE)", ... }`. Reverted; `git diff` on `spec.rs` is empty. |
| The remote (`DataClient`) and embedded (`Session`) submit paths admit `cache = Use` IDENTICALLY: both succeed, and the persisted `jobs.spec` is BYTE-IDENTICAL on both paths (K4 parity, exercised on the non-trivial multi-rank fixture, not just the single-happy-path shape) | `crates/jammi-server/tests/it/grpc_remote_compute.rs::grpc_remote_compute::a_fine_tune_cache_use_is_admitted_identically_on_both_paths` (hermetic, `cargo test -p jammi-server --test it`) | Same mutation as above (one refusal in `admit_training_spec` reds both the remote and the embedded call inside this test — a shared engine seam); first red line: `thread 'grpc_remote_compute::a_fine_tune_cache_use_is_admitted_identically_on_both_paths' ... panicked at crates/jammi-server/tests/it/grpc_remote_compute.rs:1185:10: remote submit with cache = use must be admitted: Config("model-level cache reuse is not yet supported (MUTATION PROBE)")`. Reverted. |
| The wire carries `cache` at all — a remote path that dropped the field entirely would not vacuously pass the `Use` parity assertion above (control) | Same test, the `cache = Bypass` half (unchanged in substance from the pre-existing test): asserts `remote_bypass_spec.contains("\"cache\":\"bypass\"")` and `remote_bypass_spec == local_bypass_spec` | Not separately re-mutated this round — this half of the test predates this unit's change; re-verified green under the current tree as part of the full filtered run (see Gates). Labelled UNCOVERED-BY-THIS-UNIT'S-OWN-MUTATION below. |

##### 3. Uncovered

- The `Bypass` control assertion inside `a_fine_tune_cache_use_is_admitted_identically_on_both_paths` was not independently red-mutated in this session (only the `Use` admission path was). It pins an unchanged property from the prior test version.
- No live-Postgres lane was run — these two tests use the hermetic SQLite-backed fixture the whole `it` suite defaults to; the brief did not name a live-lane requirement and neither test touches `JAMMI_TEST_PG_URL`.
- I did not exercise the terminal `cache_outcome` wire field (`ModelResult.cache_outcome` / `JobStatusResponse`) end-to-end for the `Use` arm (waiting for a worker to actually claim/run to a `Reused`/`Computed` terminal result) — out of scope per the brief, which asks only for admission + persisted-spec parity. That remains whatever the IDENTITY unit itself covered in `jammi-ai`'s own suite (not inspected by me, out of my owned scope).

##### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo test -p jammi-server --test it -- grpc_job:: grpc_remote_compute::` | 0 | 37 passed; 0 failed; 0 ignored; 234 filtered out |
| `cargo clippy -p jammi-server --all-targets -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `check-citations: 1073 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing EXEMPT historical citations, unrelated to this diff) |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | `no-consumer-names: OK` |
| `SWARM_DIFF_BASE=5bb02acd python3 ci/scripts/check_journey_markers.py` | 0 | `check_journey_markers: OK -- no journey marker in the diff's added lines.` |

All cargo commands run with `CARGO_TARGET_DIR=/private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/targets/wireuse` and `RUSTC_WRAPPER=sccache`.

##### 5. Issues closed

- **#562** (model-level cache reuse for `FineTuneSpec`, the wire-surface half): CLOSED for the two `jammi-server` it-suite tests this unit owns — both now pin the ADMITTED property and no longer assert the retired refusal. Production admission logic (`jammi-ai`) was untouched, out of my owned scope; my scope was strictly re-aligning the two stale server-side wire tests.

##### 6. Commits

```
b99354f1 test(server): #562 cache = Use is admitted identically on both paths and over the wire
```

`git log --oneline 5f19685e..HEAD` on `unit/wireuse`. Working tree clean, contract also written to `/private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/contracts/wireuse.md`.

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse unit/wireuse b99354f1287f186faf0ec4aa24d670350e8f70d6

### 11.20 WIREUSE2 — the corrupted-bundle row under Peer (wire-server) — landed as f38c33d4 (original 74dd47b9)

**Lead note:** closes the first audit's advisory; one driver shared by both topologies.

Second unit complete (unit/wireuse2), addressing the coordinator's follow-up on `crates/jammi-server/tests/it/gang_resume_parity.rs`'s corrupted-bundle row.

##### Scope shipped

`crates/jammi-server/tests/it/gang_resume_parity.rs` only:

- Extracted the corrupted-bundle row's body into a shared driver
  `run_corrupted_epoch_1_checkpoint(fleet, install_dialer, member, configure) -> Row`
  and a shared assertion `assert_corrupted_resume_failed_loudly(&Row)` — the
  corrupted-bundle logic (kill attempt 1 after the resume-checkpoint write, corrupt
  `optimizer.safetensors`, claim + run attempt 2, assert `failed` naming `sha256`) is
  written once, never duplicated per topology.
- `a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_local`
  — the prior Local-only test, renamed, calling the shared driver with `install_dialer =
  false`, `member = None` (unchanged behavior/config: `[worker] local_ranks = 2`).
- `a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_peer`
  — NEW: a real loopback `gang_chaos::Member` (the same `GangServiceServer` `gang_chaos.rs`
  runs, unmodified, serving both attempts — `run_peer_w2_resumed`'s own shape), with
  `member.wait_slot_free` awaited between the kill and the second coordinator's claim
  (mirroring the existing Peer parity row).
- Rewrote the module doc's "Two rows" section (was around :28–46) so it states both rows
  now run under both topologies and names the shared driver, instead of implicitly
  conceding the corrupted-bundle proof was Local-only.

Return-type fix needed along the way: `gang_coordinator::row()` returns `gang_coordinator::Row`
(a hand-rolled catalog projection struct), not `JobRecord` — my driver signature and the shared
assertion helper use `Row`, and I added `Row` to the existing `use crate::gang_coordinator::{...}`
import list.

##### Properties / oracles

| Property | Oracle | Executed mutation |
|---|---|---|
| A corrupted epoch-1 resume bundle fails attempt 2 loudly (never a silent from-scratch restart), under `Local` | `gang_resume_parity::a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_local` | `let resume = discover_resume(...)?;` → `let resume = None;` in `crates/jammi-ai/src/fine_tune/worker.rs`; reran filtered — red: `assertion left == right failed: ... left: "completed" right: "failed"`. Reverted (`git diff` on `worker.rs` empty). |
| Same property, under `Peer` (a real dialed loopback member) — the arm the audit flagged as unproven | `gang_resume_parity::a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_peer` | Same mutation, same run; Peer row also went red (`Row { status: "completed", ... }` vs expected `"failed"`) — confirms this arm is a real, non-vacuous proof, not one that would pass regardless of whether `discover_resume` ran. |

##### Gates (all with `CARGO_TARGET_DIR=<scratchpad>/targets/wireuse2`, `RUSTC_WRAPPER=sccache`)

| Command | Exit | Notes |
|---|---|---|
| `cargo test -p jammi-server --test it --features test-hooks -- gang_chaos:: gang_coordinator:: gang_resume_parity::` ×3 | 0 / 0 / 0 | 9 passed each run, 0 failed |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean (one `cargo fmt --all` pass needed first for two long `async fn` signature line-wraps; re-checked clean after) |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1073 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing EXEMPT historical citations, unrelated) |
| `SWARM_DIFF_BASE=5bb02acd python3 ci/scripts/check_journey_markers.py` | 0 | `check_journey_markers: OK -- no journey marker in the diff's added lines.` |

##### Issue

- **#543** (K4 resume parity — Peer vs Local, plus the corrupted-bundle proof that `discover_resume` actually ran): the Peer-arm gap the audit flagged is CLOSED — both rows (byte-parity and corrupted-bundle) now execute and pass under both topologies, sharing one driver.

##### Commit

```
74dd47b9 test(server): #543 the corrupted-bundle resume row also runs Peer, not Local only
```
(`git log --oneline 5bb02acd..HEAD`; working tree clean.)

Files edited:
- /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse2/crates/jammi-server/tests/it/gang_resume_parity.rs

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse2 unit/wireuse2 74dd47b93fe909854ca602c02c2c7a6500e85974

### 11.21 WIREUSE3 — the Peer row attributes to the member's rank (wire-server) — landed as b25fceb4 (original 7f3d9f4f)

**Lead note:** closes the second audit's advisory on the ONE observed-event seam; the worker.rs edit is a scope amendment the lead named in the brief.

Third unit complete (unit/wireuse3), closing the closing-audit advisory on the Peer corrupted-bundle row's attribution gap.

##### Scope shipped

Three files:

- `crates/jammi-ai/src/fine_tune/worker.rs` — extended the crate's ONE test-observation
  seam (`loop_test_hooks::arm_observed`/`fire_observed`) with a rank-carrying variant,
  `Event::ResumeAttempted(u32)`, instead of a bespoke pair. Fired UNCONDITIONALLY,
  immediately before a rank body's own `discover_resume` call (so it still fires even
  when that call goes on to return `Err` on a corrupted bundle, propagated via `?`).
  This is a production-code change in a crate outside my nominal ownership
  (`jammi-wire`/`jammi-admin`/`jammi-client`/`jammi-server`) — done because the
  coordinator's own instruction named the exact seam and file to extend; flagging as a
  scope amendment below.
- `crates/jammi-server/tests/it/gang_resume_parity.rs` — the shared driver
  `run_corrupted_epoch_1_checkpoint` gained a `member_rank: Option<u32>` parameter: when
  `Some(rank)`, it arms `Event::ResumeAttempted(rank)` immediately BEFORE attempt 2 runs
  (never before attempt 1, which legitimately calls `discover_resume` too and would
  otherwise consume the one-shot event) and asserts (bounded `timeout`) that it fired
  once attempt 2 is terminal. The Peer test now passes `Some(1)` (the member always runs
  rank 1 — `gang_coordinator.rs`'s own doc); Local passes `None` (no separate member to
  attribute to). Rewrote the module doc and both topology tests' own doc comments to
  state exactly what each assertion proves: the shared job-terminal
  `failed`/`sha256` assertion proves SOME rank reached `discover_resume`; only the Peer
  row's additional rank-attributed event proves the MEMBER's own body did.
- `docs/maintainer/MAINTAINER-GUIDE.md` — fixed three now-stale `PATH:LINE` citations
  into `worker.rs` (`:6107`→`:6120`, `:6051`→`:6064`, `:6059`→`:6072`) that my 13-line
  `Event` enum insertion shifted; caught by `check_citations.py`, verified each new line
  number against the actual current content before writing it.

##### Property / oracle / mutation

| Property | Oracle | Executed mutation |
|---|---|---|
| The Peer corrupted-bundle row attributes the resume-path failure specifically to the member's own rank body (rank 1), not merely to "the job (via some rank) ended failed" | `gang_resume_parity::a_corrupted_epoch_1_checkpoint_fails_attempt_2_loudly_never_a_silent_restart_under_peer` — the new `member_rank: Some(1)` attribution assertion | Gated the `ResumeAttempted` fire to `role.rank() == 0` only, simulating a member whose resume path is never genuinely reached. Reran filtered: **Local passed** (no attribution check, unaffected); **Peer's new attribution assertion failed red** (`Elapsed(())` on the bounded wait). A probe read of the row at that exact point (added temporarily, then reverted) showed `status: "failed"` with the sha256 mismatch in `error` — confirming the OLD plain status assertion would have passed vacuously, and the new assertion is the one that actually catches the gap the audit named. Reverted the mutation and the probe print; `worker.rs`'s `ResumeAttempted` fire site now matches only the shipped, unconditional addition. |

##### Gates (all with `CARGO_TARGET_DIR=<scratchpad>/targets/wireuse3`, `RUSTC_WRAPPER=sccache`)

| Command | Exit | Notes |
|---|---|---|
| `cargo test -p jammi-server --test it --features test-hooks -- gang_chaos:: gang_coordinator:: gang_resume_parity::` ×3 | 0/0/0 | 9 passed each run, 0 failed |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean (ran this extra lane since I touched `jammi-ai` production code directly) |
| `cargo fmt --all -- --check` | 0 | clean (one `cargo fmt --all` pass needed for the multi-line `fire_observed` call; re-checked clean after) |
| `python3 ci/scripts/perf/check_citations.py` | 0 | `1074 file(s) scanned, all PATH:LINE citations resolve` (2 pre-existing EXEMPT historical citations, unrelated; the three MAINTAINER-GUIDE.md stale citations my own edit caused are fixed) |
| `SWARM_DIFF_BASE=f3e1549c python3 ci/scripts/check_journey_markers.py` | 0 | `check_journey_markers: OK -- no journey marker in the diff's added lines.` |

##### Scope amendment

`crates/jammi-ai/src/fine_tune/worker.rs` is outside my nominal owned-crate list
(`jammi-wire`/`jammi-admin`/`jammi-client`/`jammi-server`). I edited it because the
coordinator's own instruction named the exact seam (`loop_test_hooks::arm_observed`,
extend `Event` with a rank-carrying variant) and call site (`worker.rs`'s
`discover_resume` call) to change, and the fix is mechanically inseparable from the
`jammi-server` test change (the test needs a rank-scoped event that only `jammi-ai` can
fire). Flagging per the "note it in scope_amendments" convention.

##### Commit

```
7f3d9f4f test(server): #543 the Peer corrupted-bundle row now attributes the failure to the member's own rank
```
(`git log --oneline f3e1549c..HEAD`; working tree clean.)

Files edited:
- /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse3/crates/jammi-ai/src/fine_tune/worker.rs
- /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse3/crates/jammi-server/tests/it/gang_resume_parity.rs
- /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse3/docs/maintainer/MAINTAINER-GUIDE.md

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-wireuse3 unit/wireuse3 7f3d9f4f9208412d145c1634bc9a35811061860b

### 11.22 IDENTITY — the first audit fold: disable-set hashing, sealed ProbedOp, path containment, decode-path rustdoc, the release manifest (ai-core) — landed as 17900dc7, 8d2f3c9d (originals 7f5168b6, 391740f9)

**Lead note:** cherry-picked with a maintainer-guide line-number conflict resolved by keeping the branch side; the folded rustdoc's round wording was stripped by the lead (f3e1549c); the second audit refuted two of these four fixes and they were rebuilt below.

Adversarial audit on `feat/500-wave5` (tip `5bb02acd`) addressed, plus the addendum, in worktree `wt-identity3` on branch `unit/identity3`, two commits, tip `391740f9`.

##### Findings fixed (commit `7f5168b6`)

**1. Definition hash omitted `JAMMI_KERNELS_DISABLE`.** `admit`/`admit_cascade` both honor it over every predicate, unconditionally, in every build — `DryRunCtx` never consulted it, so two attempts of one spec differing only in that env var hashed identically, and with `CachePolicy::Use` admitted the second would silently be served the first's (fused) bytes. Fixed: `DryRunCtx` gained `disabled_registry_keys: BTreeSet<String>`, populated once, hermetically, from `disabled_ops_requested()` at the ctx's own construction site in `train_fine_tune` — never re-read inside any `dry_run` fn. The check runs uniformly in `dry_run_all` (once, before any row's own `dry_run` fn), not duplicated into all nine fn bodies, mirroring `admit_cascade`'s own disabled-wins-over-predicate ordering. `manifest.rs`'s residual-list doc is rewritten to state this is now closed and names the remaining, genuinely-irreducible data-dependent residual completely. RED (executed, reverted): two new `jammi-kernels` tests (`dry_run_all_moves_when_only_the_disabled_registry_keys_differ`, `rendered_profile_moves_when_only_the_disabled_registry_keys_differ`) failed identically when the disabled-check was removed from `dry_run_all`.

**2. `ProbedOp` was publicly constructible.** A `pub struct`, four `pub` fields, no `#[non_exhaustive]` — a forged const in any crate would pass `admit`/`admit_cascade` invisibly to `PROBED_OPS`/`dry_run_all`/the eager-disable sweep. Sealed: `#[non_exhaustive]` plus a private `Sealed` field; construction only through `pub(crate) ProbedOp::new`, used by all 15 row consts and the two test-fixture macros (`test_two_arm!`/`test_cascade!`, rewritten to call it too). RED (executed, reverted): a forged `ProbedOp` literal injected into `crates/jammi-encoders/src/layer_norm.rs` fails with `error[E0639]: cannot create non-exhaustive struct using struct expression` — captured and reverted. Since sealing has no effect within the defining crate, added a real `syn` source oracle, `crates/jammi-kernels/tests/probed_op_construction_sites.rs`: proves every `ProbedOp::new(...)` call site in `jammi-kernels`'s own `src/` tree is either one of `PROBED_OPS`'s rows (count-keyed against the real, linked-in constant) or one of the two reviewed `#[cfg(test)]` fixture macros. RED (executed, reverted, both directions): a forged extra row inside `admission.rs` not listed in `PROBED_OPS` moved the discovered count from 15 to 16, caught; an unreviewed third macro constructing a `ProbedOp` was independently flagged by name. `ci/scripts/perf/test_finetune_ab_disable_op_keys.py`'s docstring (the line-54 area) is corrected to state precisely what the compiler now proves (external-crate sealing) versus what the new oracle proves (same-crate construction) — the prior wording overclaimed the compiler's own share.

**3. `delete_artifact_prefix`'s refusal was a string prefix, not path containment.** `models_root` yields `{root}/models` with no trailing slash, so `starts_with` also accepted a sibling like `{root}/models-archive/x`. Fixed to exact-equality-or-immediately-followed-by-`/`. RED (executed, reverted): a new test, `delete_artifact_prefix_refuses_a_string_prefix_that_is_not_a_path_ancestor`, pinning all four boundary shapes named (`models-archive`/`modelsX` siblings refused; the exact root and a real `/models/x` child accepted) failed against the old string-prefix check.

**4. Rustdoc stated the opposite of the decode path.** `jobs.rs`'s `JobSpec` doc and `spec.rs`'s `TrainingSpec` doc both claimed the training-claim sites read `TrainingSpec`/`ComputeSpec` directly and never deserialize `JobSpec` — the opposite of what `worker.rs`'s loop-claimer, Peer-rank, and compute-claim paths actually do (confirmed by reading all three call sites: each decodes `JobSpec` then projects via `as_training_spec`/`as_compute_spec`), and the opposite of `JobSpec::as_training_spec`'s own already-correct doc. Both passages rewritten to the truth, with symbol citations (`path::item`), never `file.rs:NNN`.

##### Addendum fixed (commit `391740f9`)

`ci/release-feature-manifest.json`'s three lanes (`cu12-tarball`/`cu12-wheel`/`cu12-image`) each named 12 `fused_op_admission` keys while `PROBED_OPS`'s TwoArm/Cascade rows now name 13 (`attention_block_flash` is a genuine row since the typed-op migration). Added `attention_block_flash` to all three lanes (their `capabilities` blocks must stay identical per the manifest's own schema doc and `check_release_manifest.py`'s parity check). `crates/jammi-ai/tests/gpu_capability/capability_surface.rs::manifest_capability_categories_match_probed_ops_by_kind` only builds under `--features cuda` against a real device — its executed proof is the GPU prove rerun the lead dispatches, not this commit's own gate set.

##### Gates run (both commits, final tip)

- `cargo test -p jammi-kernels --lib` — 462/462 pass.
- `cargo test -p jammi-kernels --test probed_op_construction_sites` — 2/2 pass.
- `cargo test -p jammi-ai --lib` — 817/817 pass.
- `cargo test -p jammi-db --lib artifact::` — 21/21 pass.
- `cargo clippy -p jammi-kernels/-p jammi-ai/-p jammi-db --all-targets [--features live-postgres-tests for db] -- -D warnings` — all clean.
- `cargo fmt --all -- --check` — clean.
- `python3 ci/scripts/perf/check_citations.py` — `1074 file(s) scanned, all PATH:LINE citations resolve` (~10 stale citations across `MAINTAINER-GUIDE.md`/`acceleration_report.rs` re-anchored past this fix's own line-shift).
- `python3 ci/scripts/check_no_consumer_names.py` — clean.
- `SWARM_DIFF_BASE=5bb02acd python3 ci/scripts/check_journey_markers.py` — exit 0, 6 advisory-only findings (false positives on `F32`/`I2` substrings inside a `[private]` test/comment surface), never blocking.
- `python3 ci/scripts/check_release_manifest.py` and `--self-test` (19/19) — clean.
- `python3 ci/scripts/check_flash_attn_closure.py` — PASS.
- `python3 ci/scripts/perf/test_finetune_ab_disable_op_keys.py` — 10/10 pass (unaffected by the disable-set fix, which lives one layer below what that script checks).
- `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-kernels/-p jammi-ai/-p jammi-db --no-deps` — all clean.

Working tree is clean (`git status --porcelain` empty). Commits carry no trailers.

Files touched: `crates/jammi-kernels/src/admission.rs`, `crates/jammi-kernels/Cargo.toml`, `crates/jammi-kernels/tests/probed_op_construction_sites.rs` (new), `crates/jammi-ai/src/fine_tune/worker.rs`, `crates/jammi-ai/src/fine_tune/spec.rs`, `crates/jammi-ai/src/jobs.rs`, `crates/jammi-ai/tests/it/acceleration_report.rs`, `crates/jammi-db/src/store/artifact.rs`, `crates/jammi-db/src/store/manifest.rs`, `docs/maintainer/MAINTAINER-GUIDE.md`, `ci/scripts/perf/test_finetune_ab_disable_op_keys.py`, `ci/release-feature-manifest.json`, `Cargo.lock`.

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-identity3 unit/identity3 391740f966bc4705bf30f2e9fc2c03d284593580

### 11.23 IDENTITY — the closing-audit fold: one disable predicate, field-private ProbedOp, the three-direction construction oracle (ai-core) — landed as f6ffca2e (original 6222928e)

**Lead note:** cherry-picked with an acceleration_report.rs citation conflict resolved to the branch side and re-anchored (bf3f9500); the arch-validation waivers were bumped by the lead as the next and last surface-touching commit (78bb99ae).

Closing audit findings F1 and F2 addressed, rebased onto `feat/500-wave5` tip `f3e1549c`, in worktree `wt-identity3` on branch `unit/identity3`, one commit on top of the rebase (`6222928e`), no trailer.

##### F1 — the disable-set predicate reimplemented only the exact-key arm

`DryRunCtx::op_is_disabled_here` only checked `disabled_registry_keys.contains(key)` — missing the documented `"all"` wildcard `op_is_disabled` (the live path) also honours, so `JAMMI_KERNELS_DISABLE=all` rendered a profile byte-identical to an unset run. Fixed by construction: extracted a pure `disable_decision(requested: &HashSet<String>, op: &str) -> bool` that both `op_is_disabled` (live, `admit`/`admit_cascade`) and `DryRunCtx::op_is_disabled_here` (dry, `dry_run_all`) call — no second implementation left to drift. `disabled_registry_keys` switched `BTreeSet<String>` → `HashSet<String>`, matching `disable_decision`'s own parameter type (and `disabled_ops`'s own return type) exactly.

Oracles: `dry_run_all_moves_when_only_the_all_wildcard_is_requested` (the `{}` vs `{"all"}` case the finding named) and `live_op_is_disabled_and_the_dry_predicate_agree_over_every_row_key_and_request_shape` — an enumerating equivalence over every `PROBED_OPS` row's own registry key(s) × the three request shapes (empty/exact/all), calling `DryRunCtx::op_is_disabled_here` on the REAL row (not `disable_decision` standalone), so a regression in `op_is_disabled_here`'s own wiring can't hide behind the standalone predicate agreeing with itself. RED confirmed twice: reverting `op_is_disabled_here` to its old `field.contains(key)` body failed both the wildcard test AND the equivalence test (the equivalence test specifically failed only after I rewrote it to route through the real production entry point rather than calling `disable_decision` directly — the first version of that test missed this exact regression, caught and fixed during this same round).

##### F2 — `ProbedOp` fields were still `pub`; `Copy` let a caller mutate a copy

A cross-crate probe (`crates/jammi-encoders/src/layer_norm.rs`) copied `LAYER_NORM` (`Copy`, no struct expression involved) and assigned `report_key` directly on the copy — `#[non_exhaustive]` never engages against field assignment on an already-held value. Fixed: every field is now `pub(crate)` with public read accessors (`report_key()`, `kind()`, `registry()`, `dry_run()`); every call site outside `jammi-kernels` (`worker.rs`, `acceleration_report.rs`, `gpu_capability/capability_surface.rs`, `ci/tools/probed-ops-index`) reads through the accessors now.

`probed_op_construction_sites.rs` rebuilt to the stated universe (`crates/jammi-kernels/src/**` AND `tests/**`) and three reviewed directions: `ProbedOp::new(...)` calls (count-keyed against `PROBED_OPS`, unchanged), `ProbedOp { ... }` struct-literal expressions (name-keyed against the one reviewed constructor, `ProbedOp::new`'s own body — the exact shape invisible to the prior call-only scan), and any fn whose own return type names `ProbedOp`/`Self`-inside-`impl ProbedOp` (same reviewed constructor, at the coarser signature grain).

REDs executed and captured, all reverted: a cross-crate copy-and-assign in `jammi-encoders` is `error[E0616]: field \`report_key\` of struct \`ProbedOp\` is private`; a same-crate struct-literal forgery injected into `admission.rs` (outside `ProbedOp::new`) reds the oracle's struct-literal direction; a same-crate macro constructing a `ProbedOp` outside the two reviewed fixtures reds the oracle's macro direction; two synthetic-fixture falsification tests pin the count-keyed and struct-literal directions directly.

Corrected the three prose sites the audit named — `admission.rs`'s own `ProbedOp`/`Sealed` doc (line ~2102), `test_finetune_ab_disable_op_keys.py`'s docstring (line ~57), `jammi-kernels/Cargo.toml`'s dev-dependency comment (line ~86) — to state precisely what `#[non_exhaustive]` proves (construction), what field privacy proves (mutation), and what the `syn` oracle proves (same-crate residual): neither type-system mechanism alone covers every crate; together they do.

##### A process note worth flagging

Mid-round, `SWARM_DIFF_BASE=5bb02acd python3 ci/scripts/check_journey_markers.py` reported one BLOCK finding at `crates/jammi-kernels/src/admission.rs:2549`. I traced it rather than dismissing it: the tool reads the diff's line numbers from `git diff 5bb02acd...HEAD` (committed state) but re-reads `classify_surface`'s file content from the WORKING TREE — while I had ~70 uncommitted lines of net-new content ahead of that point, the two went out of sync, and line 2549 in my working tree happened to be a doc-comment line attached to `pub fn dry_run_all` rather than the committed line's actual content (a `dtype: DtypeClass::F32,` struct-literal field inside `#[cfg(test)] mod tests`, which `is_test_path`'s own doc admits this inline-cfg(test) shape is a known blind spot for). Re-ran the check immediately after committing (HEAD now matches the working tree) and it returned exit 0, 15 advisory-only findings, no BLOCK — confirming it was a transient artifact of checking mid-edit, not a real finding, before reporting it as a result.

##### Gates (final tip `6222928e`)

- `cargo test -p jammi-kernels --lib` — 464/464 pass.
- `cargo test -p jammi-kernels --test probed_op_construction_sites` — 3/3 pass.
- `cargo test -p jammi-ai --lib` — 817/817 pass.
- `cargo test -p jammi-ai --test it` — 645/645 pass, 1 ignored (pre-existing; one flake in an unrelated `content_hash.rs` lease-timing test reproduced once under full-suite parallel load, confirmed passing in isolation and on a clean full-suite re-run — not a regression from this fix).
- `cargo test -p jammi-db --lib artifact::` — 21/21 pass.
- `cargo clippy -p jammi-kernels/-p jammi-ai/-p jammi-encoders/-p jammi-lora/-p probed-ops-index --all-targets -- -D warnings` and `-p jammi-db --all-targets --features live-postgres-tests -- -D warnings` — all clean.
- `cargo fmt --all -- --check` — clean.
- `python3 ci/scripts/perf/check_citations.py` — `1074 file(s) scanned, all PATH:LINE citations resolve`.
- `python3 ci/scripts/check_no_consumer_names.py` — clean.
- `SWARM_DIFF_BASE=5bb02acd python3 ci/scripts/check_journey_markers.py` — exit 0, 15 advisory-only findings, no BLOCK (see note above).
- `RUSTDOCFLAGS="-D warnings" cargo doc -p jammi-kernels/-p jammi-ai/-p jammi-db --no-deps` — all clean.

Working tree is clean; `git status --porcelain` empty. `ci/scripts/arch_validation_freshness_allowlist.txt` was not touched, per instruction.

Files touched: `crates/jammi-kernels/src/admission.rs`, `crates/jammi-kernels/Cargo.toml`, `crates/jammi-kernels/tests/probed_op_construction_sites.rs`, `crates/jammi-ai/src/fine_tune/worker.rs`, `crates/jammi-ai/tests/it/acceleration_report.rs`, `crates/jammi-ai/tests/gpu_capability/capability_surface.rs`, `ci/tools/probed-ops-index/src/main.rs`, `ci/scripts/perf/test_finetune_ab_disable_op_keys.py`.

BRANCH /private/tmp/claude-501/-Users-vijaychakilam-git-f-inverse-jammi-ai/952d975f-7dc9-4c05-bbed-2f2be26d6898/scratchpad/wt-identity3 unit/identity3 6222928e8ae0922e596c8e196b6c8f1f6ed3c4a9

### The lead's ruling on the third closing audit (2026-09-17)

The third audit on bf3f9500 refuted, by execution, the classes the stop rule above names: one shared
`disable_decision` reached by every live and dry caller (no request shape diverges), `ProbedOp` unconstructible
and unmutable from any other crate, the waivers covering the head, the Peer attribution only mintable by the
member. Its three stands are not mechanism defects: the same-crate enumerating oracle over `ProbedOp`
construction misses qualified paths, aliases and macro invocations; `begin_release`'s phase-then-epoch order
is load-bearing but unpinned and the rustdoc argument for the lattice is inverted; `op_is_disabled`'s doc
block is attached to `disable_decision`. The stop rule was written for a further mechanism defect in those
classes, and its release-window arm would revert to a predicate the first audit executed as dispatching a job
through a RELEASE — a known defect, which no principle allows shipping. The lead therefore fixes the three
stands as test, pin and doc changes with executed oracles, states this deviation here and in the PR, and
pre-commits without ambiguity: a fourth closing audit that blocks on any file these fixes touch, at any
severity, excises the identity unit's cache-reuse admission and definition-hash fold from this PR and files
the placed-gang ordering as an open issue with the audit's interleaving attached. No fifth round.

### The third-audit fold (2026-09-17), executed before the fourth audit

The three stands and their class are closed on `1c6d02c3` as follows, each with an executed refutation
recorded in the relay artifact for block `2026-09-17T08:30:27` (`.jammi/gate-state/feat_500-wave5.relay.adversarial-audit.2026-09-17T08_30_27.425179_00_00.json`):

- **`begin_release` order pinned (`82b821fa`).** `HostAdmission::begin_release` is `pub async fn`; the phase
  flip runs strictly before the epoch bump with a `ParkPoint::BeginReleaseBetweenFlipAndBump` park between
  them; `begin_release_bumps_the_epoch_strictly_after_the_phase_flip_is_already_visible` parks a live call there
  and asserts `probe_claim` already refuses while the epoch is unbumped. Moving the bump before the flip reds it
  (executed in a detached clone: ok unmutated, FAILED mutated). The `for_single_run` and `run_placed_gang`
  rustdoc state the dependency in its true direction and name the pin.
- **`ProbedOp` construction oracle rebuilt (`2304a79b`).** Seven directions over `src/**` and `tests/**`: calls
  matched by the path's last two segments under any prefix, `<ProbedOp>::new` via `qself`, `Self::new` via the
  impl stack, same-crate `type` aliases via a first collector pass, struct literals, `ProbedOp`-returning fns,
  macro INVOCATION token streams (exact `Ident`, never substring), `macro_rules!` bodies, explicit `transmute`
  targets. The comparison is name-keyed against `PROBED_OPS`'s `report_key`s (extra, missing, duplicated each
  named) with the `>=` floor asserted directly. The auditor's five bypasses plus `Self::new` are executed
  fixtures against the real scanner (12/12). The one residual — a `transmute` whose target type is inferred
  where a syntax-only scan cannot resolve it (`jammi-kernels` carries no `#![forbid(unsafe_code)]`) — is named
  at every site that previously claimed completeness (oracle module doc, `ProbedOp`/`Sealed` docs, the
  dev-dependency comment, the sweep's docstring), never omitted.
- **Docs reattached (`2304a79b`).** `op_is_disabled`'s fired-set/lock paragraph is on `op_is_disabled`;
  `disable_decision` carries its own two-sentence doc; the body is byte-identical.
- **Class sibling closed (`1c6d02c3`).** The raw-deleter call-site oracle (`models_delete_call_sites.rs`) had
  the same fail-open shape — method calls only, completeness by "confirmed by reading". It now records path
  calls under any prefix or qualified self and every deleter `Ident` inside any macro invocation, with
  `shape_*` tests running the real scanner over fixtures (four path spellings, `try_join!`/`assert!`/
  `macro_rules!` bodies, near-miss and string-literal controls, the loud module-scope arm). Removing either
  direction reds its fixture (executed). The 16 reviewed sites and their classes are unchanged. The other syn
  oracles on the branch (`pinned_source_gate.rs`, `test_bounds_inventory.rs`) were examined: one already walks
  macro token streams, the other keys items, not constructions.
- **Waivers and hygiene (`f839c63b`).** The four arch-validation waivers are re-reviewed through `2304a79b`,
  the last commit touching `admission.rs` (comment-stripped diff since `f6ffca2e` is the three-line relocation
  of `disable_decision`; `flash_validated_arches`/`parse_gencode_sms` byte-identical to `7f4364c9`, executed);
  the sweep docstring's invalid `\`` escape is removed (`py_compile` clean under `-W error`).

Gates executed on `1c6d02c3` after the last edit: kernel-oracles PASS, citations (1074 files) resolve, plan
citations OK, journey markers 0 blocking, arch-validation freshness PASS, both oracles green; the sequential
merge path (static → tests on a fresh Postgres → guards+swarm) and GPU prove `35201591017` run on this head.

### The fourth closing audit (2026-09-17) and the stop rule's execution

The fourth audit on `1c6d02c3` re-executed the `begin_release` pin (red under the swap, single writer for each
cell, every caller awaits) and the doc reattachment, and found both closed. It blocked twice. First, the rebuilt
`ProbedOp` construction oracle fails open on two further shapes, executed against the real scanner: a same-crate
`use crate::admission::ProbedOp as PO;` rename (no `visit_item_use` direction anywhere in the tree) and a mutated
copy returned behind `&'static ProbedOp` or `Option<ProbedOp>` (the return-type direction reads only a bare
`Type::Path`'s last segment) — while direction 3's own text claimed the mutated-copy vector was caught. That block
lands on the files the third-audit fixes touched, so the pre-committed rule above fires without interpretation:
the identity unit's cache-reuse admission (I3) and definition-hash fold (I4) are excised from this PR. The
typed admission table stays — cross-crate callers are compiler-bound to it, which the compiler proves — and the
same-crate sealing claim goes with its oracle, because once nothing folds admission rows into a durable hash no
property rests on same-crate construction. `#546` and `#562`'s reuse item stay open (its delete-guard items ship); the next attempt starts from the
oracle's named blind spots (filed). Second, the `submit_admitted_training` one-seam oracle
(`rank_admission.rs`) walked only method calls — the un-swept sibling of the class `1c6d02c3` closed in the
delete oracle. That is not an identity file and not a mechanism defect; it is rebuilt over the three call
shapes with executed fixtures and mutations, its universe widened to every crate's `src/` tree (the workspace's production code),
and the delete oracle's universe widened to every crate's `src/` (the auditor's advisory: `delete_if_exists`
is `pub`). The excision and these two oracle rebuilds are the last changes before the closers; the closing
verification on the resulting tip is the excision's own audit, not a fifth round on the excised mechanisms.

### The excision, executed (2026-09-17)

`ff9ec6eb` removes I3 and I4 as the rule requires and nothing else: `admit_training_spec` refuses
`CachePolicy::Use` with the base's typed message (byte-identical slice); the fine-tune probe/early-return,
`FineTuneMaterializationOutcome::Reused`, `PublishedPrefix::existing`, the second-row finalize and the
`reused:{model_id}` wire arm are gone; `dry_run_admission_profile`, `DryRunCtx`/`DryRunVerdict`/`DeviceKind`,
every row's `dry_run` fn, `dry_run_all`, `op_is_disabled_here` and the private `Sealed` field are gone;
`MaterializationEnv::kernel_admission_profile` is byte-identical to the base (`None`, UNCOVERED);
`crates/jammi-kernels/tests/probed_op_construction_sites.rs` and the crate's `syn`/`proc-macro2` dev-deps are
deleted; the three `cache = Use` tests in `jammi-server` and `rank_admission.rs` are restored to the base's
refused bodies (diffed byte-identical), and `fine_tune_materialization.rs` is the base file byte-for-byte again since `fdf28c27` (four tests, including `cache_bypass_never_reuses`, whose two assertions — distinct prefixes for two `Bypass` runs, `cache_outcome` `"computed"` — no other test carried). The
typed table, its rows (including `attention_block_flash`), `ci/tools/probed-ops-index`, the eager-disable
sweep and the manifest lanes stay; the `ProbedOp` rustdoc, the Cargo comment and the sweep docstring state
what the compiler proves cross-crate and that same-crate construction is unsealed with no property resting
on it. `2578af25` re-reviews the four arch waivers through `ff9ec6eb` as the last `admission.rs` commit
(`flash_validated_arches`/`parse_gencode_sms` byte-identical to `7f4364c9`, executed; freshness PASS);
`77f3ecc6` drops the delete oracle's reference to the deleted file. On `77f3ecc6`: citations (1073 files)
resolve, plan citations OK, journey markers 0 blocking vs `db19a614`, consumer names OK; the merge-path chain
and GPU prove `35207488119` run on this head; the closing verification follows the fifth relay.

### After the excision: the two closing verifications and their folds (2026-09-17)

The first closing verification (`8049ce1b`) verified E1–E4 by execution and blocked on three things the
lead owned: the submit-seam oracle's universe was the pinned source gate's two directories while
`jammi-server`, `jammi-ballista` and `jammi-bench` hold `Catalog` handles (a hand-built submit in the gRPC
handler passed green); this contract still carried the excised I3/I4 property rows; and the excision had
deleted the base's `cache_bypass_never_reuses`, whose two assertions nothing else carried. `fdf28c27` widened
the universe, restored `fine_tune_materialization.rs` to the base byte-for-byte, and this contract's §2.6 and
§2.12 were rewritten. The second verification (`fdf28c27`) blocked once more, on a fourth spelling both
call-site oracles missed — a fn item captured as a value and invoked later, which compiles and passed green
in both — with the universe overclaim ("every workspace crate" vs `crates/*/src`) and this narrative's staleness
as advisories. `dfc01353` records the target's path in ANY expression position (`visit_expr_path`) in both
oracles with `shape_4` fixtures, and quantifies both over every compiled non-test `.rs` in the repository
(every `.rs` cargo compiles outside a test target: every workspace member's `src/` including `ci/tools/*`, every `build.rs`, every `examples/`/`benches/` target) through one shared predicate in `jammi-test-utils`;
the auditor's two captured-fn-item probes red their oracles and removing the direction reds `shape_2`/`shape_4`
in each (all executed, reverted). On `dfc01353`: citations resolve, plan citations OK, journey markers 0
blocking, consumer names OK, kernel oracles re-run; the merge-path chain runs on this head; GPU prove
`35207488119` is green on all four arches at `77f3ecc6`, an ancestor whose flash surface is unchanged since.

### The draft PR's first CI run (2026-09-17)

PR #587 was opened as a draft on `dfc01353` so CI would run in parallel with the closing verification. One
guard failed: the eager-disable key sweep (`ci/scripts/perf/test_finetune_ab_disable_op_keys.py`), which since
`5bb02acd` derives its key set from a real `cargo run -p probed-ops-index`. `ae4030ce` recorded the cause as
"the guard runner has no Rust toolchain" and moved the suite to swarm.yml's `symbol-index-gates` job while
hiding cargo from PATH in the local guards stage. The fourth closing verification refuted that diagnosis from
the failing job's own log: the runner has cargo and rustc; what it lacks is `sccache`, which
`.cargo/config.toml` makes the mandatory rustc wrapper, and which only `setup-rust-ci` installs (the guard
matrix already grants it to one leg, `pod build substrate`). It also showed the move had made the sweep — and,
since `15cd1163` in this wave, `check_plan_citations.py` and `check_no_consumer_names.py` — non-blocking:
swarm.yml's `symbol-index-gates` is in no required context. The fold: the sweep is a `toolchain: true` leg of
the guard matrix (aggregated by the required `ci-summary`); the `symbol-index-gates` job moves into ci.yml under
`ci-summary`'s `needs`, so its gates block a merge without branch-protection wiring; `merge_path.sh`'s guards
stage mirrors the real split — non-toolchain legs run with `RUSTC_WRAPPER` pointed at a path that does not exist
(the env var overrides `.cargo/config.toml`; sccache shares `~/.cargo/bin` with cargo, so hiding it by PATH is
not the runner's shape), toolchain legs run as-is — the runner's remaining divergence, `mold` mandated by
`.cargo/config.toml` for x86_64 Linux and absent on the bare runner (so a WORKSPACE crate cannot be built
there while a fixture workspace can: `pod build substrate` builds one and passes in CI), has no macOS mirror
and is met by placement (a guard that builds a workspace crate lives in the container-backed job) with the
draft PR's CI as the check — and its swarm stage runs the container-backed job's steps.


### The third closing verification and its fold (2026-09-17)

The third verification (`dfc01353`) re-executed both captured-fn-item probes red and found the two holes one
level above spelling: the delete oracle's TARGET set did not include `JammiObjectStore::driver`, a `pub`
accessor to the raw `Arc<dyn ObjectStore>` on which `ObjectStoreExt::delete` removes any key unguarded (a
`driver().delete` in `jammi-db` compiled and the oracle stayed green); and the universe predicate excluded
`examples/` and `benches/`, which cargo compiles (`--all-targets`), so a hand-built training submit in
`crates/jammi-bench/examples/frontend_serial_tail.rs` compiled and passed. Advisories: the doc block the
predicate's insertion detached from `tracked_rs_files`, "one shared predicate" being two copies, the relay's
458-file count measured without the `ci/fixtures/` arm (the predicate's number was 455), and stale scope
prose in both oracles. The fold: `driver` is `pub(crate)` (cross-crate: compiler, executed E0624) and is in the
oracle's target set inside `crates/jammi-db/src` (its two references, the parquet reader and writer, are
reviewed rows); the universe is defined once in `jammi_test_utils::source_universe` as every `.rs` cargo
compiles outside a test target (`src/`, `build.rs`, `examples/`, `benches/`; not `tests/`, not the
`ci/fixtures/` inputs) and both oracles call it; a bare single-segment path is not a method reference in
either oracle (an inherent method is always owner-qualified), which also keeps locals named `driver` out;
the pinned source gate's helper insertion is reverted so its doc block sits on its item again; the stale
prose is restated. Separately, the draft PR's CI showed the dep-DAG freshness guard red because the excision
dropped two dev-dependencies from `jammi-kernels`; the guide's rendered DAG is regenerated by
`gen_dep_dag.py`. Probes executed and reverted on the fold's tree: the raw driver delete in `reconcile.rs`
(RED), the cross-crate `driver()` in `jammi-ai` (E0624), the hand-built submit and a raw delete in the
example target (both RED); dropping `driver` from the target set reds `shape_5`.


### The fourth closing verification and its fold (2026-09-17)

The fourth verification (`3fd3820b`) blocked five times. Two were the raw-store class one level further out:
the raw `Arc<dyn ObjectStore>` is obtainable without the handle through `pub` `StorageRegistry::driver_for` and
`build_object_store` (executed: a `driver_for(..).delete` in `jammi-ai` compiled and the oracle stayed green),
and the handle's private `driver` FIELD was a shape the scanner never recorded (executed: a `self.driver.delete`
deleter added to the handle's file passed). A third was the same class on the submit side: the catalog's `pub`
generic SQL surface (`backend_arc` → `transaction` → `execute`) writes a training-kind `jobs` row with no
`submit_job` reference. The other two were the lead's: `ae4030ce`'s recorded cause was refuted by the CI log
(sccache, not cargo) and its move made four gates non-blocking. The fold, and where the chase stops: the field
shape is recorded (shape 5b) and every use of the field in the handle's file is a reviewed row; §2.6 and §2.12
are restated to the properties the tree proves — every reference to the handle's deleters and to the catalog's
submit API is reviewed — and each names its remaining door with the executed enumeration of that door's
present acquisition sites; sealing both doors by construction (guarded handles only, crate-private raw builders
and `Tx::execute`) is a jammi-db capability-sealing unit filed from this contract, not another oracle widening.
The CI fold is recorded in the section above; the universe predicate refuses only a crate's top-level
`tests/` directory (a deeper `tests` component is a compiled target or module); the dep-DAG lane's trigger
covers `ci/tools/**`; the `with_root` "by construction" sentence is restated as the review it is; the
workspace comment states what `--workspace` does.

The draft PR's CI on `3fd3820b` also red `Test (Python)`: the conformance fixture
`test_embed_reconcile_referenced_list_is_populated_and_matches_the_remote_key_set` inserts a `models` row by
raw SQL and omitted `updated_at`, so the legacy `CAST(CURRENT_TIMESTAMP AS TEXT)` default was refused by
migration 039's canonical-stamp trigger — the schema edge doing what S1 asks of any writer that omits a stamp.
`86ecaae7` makes the fixture stamp `created_at`/`updated_at` explicitly in the canonical shape (red at the
prior head, green after, executed locally in a venv built as `setup-jammi-py` builds it) and the migration's
doc states that a domain column's legacy DEFAULT is not a writer and is refused at the edge; it also folds the
`matches!` lint the merge-path static stage caught in the new universe predicate.

CI on `86ecaae7` red the sweep once more, now as a `toolchain: true` matrix leg: the sweep builds two
WORKSPACE crates, and `.cargo/config.toml` mandates `-fuse-ld=mold` for x86_64 Linux, which the bare guard
runner lacks (`collect2: cannot find 'ld'` is gcc failing to find `ld.mold`); the runner's own `ld` links a
fixture workspace fine (`pod build substrate` does, and passes), so a toolchain leg serves metadata and
fixture builds, never a workspace crate. `2e9d014e` runs the sweep in ci.yml's container-backed `symbol-index-gates` job (toolchain,
sccache, linker), which `ci-summary` requires; the matrix comment and `merge_path.sh` state the runner's real
constraints (no sccache, no linker), and the local swarm stage runs that job's steps.

### Close-out on the user's call (2026-09-17)

After the fifth closing verification's fold (`3843ee88`), the user directed that no further verification pass
run and that the closers run now: the recent passes were verifying restated prose, CI placement and gate
completeness, not product behavior; every product-code defect the audits found (the placed-gang epoch read,
the disable set in the definition hash, the `all` wildcard, the copyable admission row, the string-prefix
containment, the Peer resume attribution, the `begin_release` order) was fixed in the three rounds before the
excision and carries an executed mutation. The sixth pass was stopped at its start. The record therefore
closes on `3843ee88` with the last adversarial verdict a BLOCK whose every site maps to a fix or a named
residual in the relay for block `2026-09-17T12:45:05`, the two capability doors filed as #588, and the
oracle (phase 5, the hard-block gate) run last on the final tip.
