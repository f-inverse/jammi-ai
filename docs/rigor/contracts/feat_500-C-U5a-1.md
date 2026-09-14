# CONTRACT — feat/500-C-U5a-1: `GangService` wire, I-GANG, training-set identity

**Contract of record.** slug: `feat_500-C-U5a-1` · branch `feat/500-C-U5a-1` at `c1d918b4` ·
this file is the committed mechanism contract `ci/scripts/check_rigor_record.py` requires under
`docs/rigor/contracts/**` (its check 3) before this unit's rigor record at
`docs/rigor/feat_500-C-U5a-1.jsonl` (the lead's export) satisfies check 1/2. Source design
contract: the lead's `CONTRACT-U5a.md` v13 (scratchpad-only, never a repo path — it is not cited
with a `path:line` token anywhere in this document, since it is not a tracked file at HEAD and any
such token naming it would fail this checker's own cost-floor pass), U5a-1 half only. Every
citation below is a full repo-relative path, never a bare filename, tagged **(at c1d918b4)**, and
was read directly against that commit in this worktree by the agent writing this file — not
carried forward from the source contract's own (differently-based) line numbers.

This document states the mechanism as it EXISTS at `c1d918b4` (CURRENT-STATE, per CLAUDE.md — no
"this was added because..." framing beyond what a citation needs to be unambiguous).

## Scope

U5a-1 freezes: (1) the `jammi.v1.gang` wire (one bidi rpc, `RunRank`); (2) the I-GANG admission
decision as a two-conjunct property inside `GangServer::run_rank`; (3) the `jobs` row's
training-set identity pair, written once by a CAS and enforced pairwise by a schema `CHECK`; (4)
`fresh_instance`, the coordinator-liveness check backing I-GANG's second conjunct. It does **not**
build `HostAdmission`, drain, or re-verification (U5a-2) — every call that clears I-GANG in this
unit still ends `Unimplemented`, because there is no session to hand it to yet.

---

## 1. Mechanism

### 1.1 The wire (`jammi.v1.gang`)

`crates/jammi-wire/proto/jammi/v1/gang.proto:1-111` (at c1d918b4) declares package
`jammi.v1.gang` with one service, `GangService { rpc RunRank(stream RankControl) returns (stream
RankEvent); }` (`crates/jammi-wire/proto/jammi/v1/gang.proto:28-30`). `build.rs` registers it in
the compiled proto set at `crates/jammi-wire/build.rs:35` (at c1d918b4); the generated module is
`pub mod gang { … include_proto!("jammi.v1.gang") }` at `crates/jammi-wire/src/proto.rs:45-46`
(at c1d918b4).

Message shapes frozen at c1d918b4 (all in `crates/jammi-wire/proto/jammi/v1/gang.proto`):
- `RankControl` (`:33-38`): `oneof control { Assign assign = 1; Cancel cancel = 2; }`.
- `Assign` (`:44-50`): `job_id, attempt, rank, world, coordinator_instance_id` — job coordinates
  only; no training-set identity, no lease duration.
- `Cancel` (`:53`): empty.
- `RankEvent` (`:56-62`): `oneof event { Admitted admitted = 1; Aborted aborted = 2; Outcome
  outcome = 3; }`.
- `Admitted` (`:65`): empty (U5a-2 is the only producer; nothing in U5a-1 emits it).
- `AbortReason` (`:70-87`), an additive enum: `DRAIN`, `REFUTED`, `CANCELLED`, `NO_BODY`,
  `STORE_UNAVAILABLE`, `UNAVAILABLE` — all six frozen here though U5a-1 emits none of them either
  (U5a-2's session ends produce them).
- `Outcome`/`TrainedOutcome`/`FailedOutcome` (`:98-111`): frozen additive; neither producer (a rank
  body) nor consumer (the coordinator's terminal write) exists in any unit landed so far.

`api_freeze.rs`'s guard decodes only `PACKAGE <pkg>` / `RPC <Service>/<Method>` tokens from the
compiled `FILE_DESCRIPTOR_SET` (`crates/jammi-server/tests/it/api_freeze.rs:50-68`, at c1d918b4)
— no message/field/oneof-arity token exists, so this proto's message shape is protected by this
contract and code review only, never by the freeze gate mechanically. The freeze gate itself is
additive-safe here: `api_freeze_baseline.txt` gained exactly `PACKAGE jammi.v1.gang`
(`crates/jammi-server/tests/it/api_freeze_baseline.txt:25-30`, at c1d918b4) and `RPC
GangService/RunRank` (`crates/jammi-server/tests/it/api_freeze_baseline.txt:104-106`, at
c1d918b4), both marked `# ADDED (additive, minor-compatible)`.

### 1.2 The handler: `GangServer::run_rank`

`crates/jammi-server/src/grpc/gang.rs` (at c1d918b4), registered as `pub mod gang;` in
`crates/jammi-server/src/grpc/mod.rs:28`. `GangServer` (`crates/jammi-server/src/grpc/gang.rs:72-81`)
holds `Arc<InferenceSession>` plus the deployment's `[lease]` window (`Duration`), needed for
`fresh_instance`'s margin computation without reaching back into `InferenceSession` for a config
accessor this crate does not own.

`run_rank` (`crates/jammi-server/src/grpc/gang.rs:87-187`), in order:

1. **Bounded first frame** (`crates/jammi-server/src/grpc/gang.rs:97-118`): awaits the first
   inbound `RankControl` within `FIRST_ASSIGN_BOUND = Duration::from_secs(10)`
   (`crates/jammi-server/src/grpc/gang.rs:63`, a handler-local constant, not a config knob); a
   silent client, a stream that closes before any frame, or a non-`Assign` opening frame is
   refused `InvalidArgument` — none of these reach I-GANG.
2. **Wire-level K2, before any row is read** (`crates/jammi-server/src/grpc/gang.rs:120-127`):
   `assign.world == 0` and `assign.rank >= assign.world` are each refused `InvalidArgument`, one
   `if` per edge, two textually distinct messages (`"world must be greater than zero"` / `"rank
   must be less than world"`).
3. **I-GANG conjunct (a): the row predicate** (`crates/jammi-server/src/grpc/gang.rs:129-146`) —
   `catalog.get_job_for_rank(&assign.job_id)` (primary-key only; no tenant argument passed or read
   anywhere in this call). `Ok(None)` and every one of `!running`, `claimant_matches` false,
   `attempt_matches` false, `!row.lease_live` collapse to the SAME `i_gang_refused()` call
   (`crates/jammi-server/src/grpc/gang.rs:53-57`: `Status::failed_precondition(I_GANG_REFUSAL_MESSAGE)`,
   one constant string at `crates/jammi-server/src/grpc/gang.rs:53`).
4. **I-GANG conjunct (b): the sidecar verify, `world > 1` only**
   (`crates/jammi-server/src/grpc/gang.rs:148-169`) — the pair (`row.training_set_ref`,
   `row.training_set_location`) must both be `Some`, and `resolve_training_set_identity` must
   resolve `Ok(())`; either failure is the same `i_gang_refused()`.
5. **Coordinator freshness** (`crates/jammi-server/src/grpc/gang.rs:171-179`) —
   `catalog.fresh_instance(&assign.coordinator_instance_id, self.lease)`; `Ok(false)` is the same
   `i_gang_refused()`.
6. **f1'** (`crates/jammi-server/src/grpc/gang.rs:181-186`) — every determinant satisfied, and
   there is no `HostAdmission` session to hand the call to (U5a-2 builds it):
   `Status::unimplemented("gang admission is not implemented on this build")`.

Every one of steps 3–5's refusals is the identical status *and* identical message
(`I_GANG_REFUSAL_MESSAGE = "gang admission refused"`, `crates/jammi-server/src/grpc/gang.rs:53`) —
**I-GANG Non-disclosure**: the listener discloses neither a job's existence, its claimant, nor its
attempt. This unit builds no test-only "last-refusal-reason" observation seam (that mechanism, if
ever needed, is for distinguishing U5a-2's *wire* `Aborted{reason}` values, which are a different,
later, and deliberately more informative surface than this admission-time collapse); at U5a-1
every b1' row is distinguished by test **setup**, never by an assertion on a hidden reason, since
the property under test is precisely that the *output* carries no distinguishing signal.

`resolve_training_set_identity` (`crates/jammi-server/src/grpc/gang.rs:229-267`, `pub async fn`,
called only from `run_rank` in production) implements §W2 Resolution's two round-11 rulings as
code, not merely as a design statement:
- **Ruling 4 (explicit admin-scope guard).** `crates/jammi-server/src/grpc/gang.rs:235-239`: `if
  TenantBinding::is_admin_scope() { return Err(Status::failed_precondition(…)) }`, checked
  *before* any resolver call, regardless of which tenant or table was named.
  `TenantBinding::is_admin_scope` itself (`crates/jammi-db/src/tenant_scope.rs:180`, reading the
  `tokio::task_local!` declared at `crates/jammi-db/src/tenant_scope.rs:27`, both at c1d918b4,
  unchanged from the source design contract's citations) is ambient state this call site does not
  control by construction — which is exactly why the guard is explicit here rather than relied
  upon inside the resolver.
- **Ruling 5 (strict-predicate verb, never the relaxed read).** `crates/jammi-server/src/grpc/gang.rs:241-249`:
  calls `store.catalog().get_result_table_for_tenant(training_set_location, tenant)`
  (`crates/jammi-db/src/catalog/result_repo.rs:1408-1447`, at c1d918b4) — **never**
  `get_result_table` (`crates/jammi-db/src/catalog/result_repo.rs:1352-1388`, same file) — and
  additionally requires `record.status == ResultTableStatus::Ready.to_string()` before proceeding.
- The sidecar read (`crates/jammi-server/src/grpc/gang.rs:251-266`) parses `record.parquet_path`
  as a `StorageUrl`, then calls `store.read_materialization_manifest(&url)`
  (`crates/jammi-db/src/store/mod.rs:1267-1280`, at c1d918b4) and accepts only `Ok(Some(manifest))
  if manifest.artifact.0 == training_set_ref`; every other outcome (`Ok(None)`, a digest mismatch,
  `Err(_)`) collapses to the SAME `FailedPrecondition("training set unresolved")` — the
  admission-time collapse §W2 Resolution states (the three-way `Refuted`/`StoreUnavailable`/healthy
  split is §I3's re-verification classification, which needs an admitted session to re-verify
  inside; U5a-1 builds no such session, so it does not implement that split).

`get_result_table_for_tenant`'s own predicate (`crates/jammi-db/src/catalog/result_repo.rs:1432-1434`,
at c1d918b4) is `WHERE table_name = $1 AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))`
— matches only a row whose tenant is IDENTICAL to the caller's argument (including the
NULL-equals-NULL case) — never `get_result_table`'s own relaxed `(tenant_id = $2 OR tenant_id IS
NULL)` (`crates/jammi-db/src/catalog/result_repo.rs:1374-1375`, at c1d918b4), which also matches a
NULL-tenant row of the same name belonging to no tenant. Both verbs share the identical admin-scope
branch (drop the tenant predicate, match by primary key alone —
`crates/jammi-db/src/catalog/result_repo.rs:1365-1371` and `crates/jammi-db/src/catalog/result_repo.rs:1424-1430`,
at c1d918b4): this contract does not change that shared verb behaviour; it adds the *call-site*
guard in `resolve_training_set_identity` instead (ruling 4, above), which is why the guard lives
beside the call, not inside the repo method.

### 1.3 `Catalog::get_job_for_rank` (`crates/jammi-db/src/catalog/jobs_repo.rs`, at c1d918b4)

`RankAdmissionRow` (`crates/jammi-db/src/catalog/jobs_repo.rs:207-221`): `status`, `tenant_id`,
`claimed_by`, `attempts`, `lease_live` (bool), `remaining` (`Duration`), `training_set_ref`,
`training_set_location`. `SELECT_COLS` (`crates/jammi-db/src/catalog/jobs_repo.rs:223-227`) is the
catalog's canonical `jobs` column list, extended with the pair at its tail.

`get_job_for_rank` (`crates/jammi-db/src/catalog/jobs_repo.rs:1891-1955`, `pub async fn`) is
primary-key-only — `WHERE job_id = $N`, no tenant predicate anywhere in the statement, never
consults `TenantBinding::is_admin_scope` — tenant comes back as a plain `Option<TenantId>` column
for the **caller** to derive and pin (I-GANG: "tenant is derived from the row", never the caller's
metadata). One statement computes the row plus the remaining-lease window via
`super::lease::lease_remaining_seconds_expr` (`crates/jammi-db/src/catalog/jobs_repo.rs:1904-1908`);
`lease_live` is derived at `crates/jammi-db/src/catalog/jobs_repo.rs:1929-1938`:
`remaining_secs.is_some_and(|s| s >= 0.0)` — a `NULL` lease column (`remaining_secs: None`) is
`lease_live == false` with `remaining == Duration::ZERO`, matching
`crates/jammi-db/src/catalog/lease.rs`'s stated "NULL means remaining 0, never live-by-default"
semantics, never a live-by-default read. `Ok(None)` when no such job exists
(`crates/jammi-db/src/catalog/jobs_repo.rs:1917`, `query_opt`).

**Enumerating-caller oracle (§1.6 below) asserts:** the only call site outside `jammi-db`'s own
tests and `jammi-server`'s own gang tests is `GangServer::run_rank`
(`crates/jammi-server/src/grpc/gang.rs:134`).

### 1.4 Training-set identity: `jobs.training_set_ref` / `jobs.training_set_location`

**Schema.** Migration `034_jobs_training_set_identity`
(`crates/jammi-db/src/catalog/schema.rs:1165-1169`, at c1d918b4):
```
ALTER TABLE jobs ADD COLUMN training_set_ref TEXT;
ALTER TABLE jobs ADD COLUMN training_set_location TEXT
    CHECK ((training_set_ref IS NULL) = (training_set_location IS NULL));
```
Registered in the migration table (`crates/jammi-db/src/catalog/migrations.rs`, the const list) as
`034_jobs_training_set_identity`, ordered directly after U3's `033_model_materialization` — the
renumbering from this unit's own base (where it was 033, the newest entry) happened once, at the
consolidation into PR-B2, at the pin sites the plan names (`docs/plans/67-distributed-training/UNITS.md`
§ U5a-1): the tuple in `crates/jammi-db/src/catalog/migrations.rs`, the constant
`crates/jammi-db/src/catalog/schema.rs::MIGRATION_034_JOBS_TRAINING_SET_IDENTITY` and its doc comment,
the const `crates/jammi-db/tests/it/migrations.rs::EXPECTED_MIGRATION_NAMES`, the `IN`-list literal in
`crates/jammi-db/tests/it/migrations.rs::migration_029_copies_training_jobs_rows_into_jobs_as_queued`, and
the ordered-after oracle
`crates/jammi-db/tests/it/migrations.rs::migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge`
— a rename, not a semantic change. No production file names the literal number for this migration
outside `crates/jammi-db/src/catalog/{migrations.rs,schema.rs}`; the ordering-test fixtures in
`crates/jammi-db/tests/it/migrations.rs` are the only other tracked hits, and are tests, not
production code.

**The pair is one fact, enforced at the schema edge, not by convention.** The `CHECK` constraint
above fires on any `UPDATE` (not just `INSERT`) that would leave the pair split — pinned by two
tests: `crates/jammi-db/tests/it/gang_rank_admission.rs:387-412`
(`a_raw_single_column_write_is_refused_by_the_schema_check`, SQLite) and
`crates/jammi-db/tests/it/migrations.rs:1571-1745`
(`migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge`, parametrized over both
dialects — SQLite unconditionally, Postgres under `#[cfg_attr(feature = "live-postgres-tests", …)]`
at `crates/jammi-db/tests/it/migrations.rs:1566-1570`), the latter attempting the single-column
`UPDATE jobs SET training_set_ref = $1 …` (`crates/jammi-db/tests/it/migrations.rs:1701`) and `…
SET training_set_location = $1 …` (`crates/jammi-db/tests/it/migrations.rs:1731`) and asserting
both error.

**Fill (`Catalog::fill_training_set_identity`, `crates/jammi-db/src/catalog/jobs_repo.rs:1802-1874`,
at c1d918b4).** One statement (`crates/jammi-db/src/catalog/jobs_repo.rs:1821-1836`) sets both
columns, guarded on `job_id`, the caller's own `claimed_by` and `attempts`, **and**
`training_set_ref IS NULL AND training_set_location IS NULL`. `updated == 1` ⇒
`TrainingSetFillOutcome::Filled` (`crates/jammi-db/src/catalog/jobs_repo.rs:1837-1839`). Zero rows
updated re-reads by primary key (`crates/jammi-db/src/catalog/jobs_repo.rs:1845-1859`) and
classifies (`crates/jammi-db/src/catalog/jobs_repo.rs:1860-1870`): same `claimed_by`/`attempts`
**and** the same pair values already present ⇒ `Reused` (a racer or retried caller observing its
own would-be write); anything else (a moved claim, or a different pair) ⇒ `Aborted` — no terminal
write follows either way. `TrainingSetFillOutcome` is declared at
`crates/jammi-db/src/catalog/jobs_repo.rs:182-199`.

### 1.5 `Catalog::fresh_instance` / `instance_liveness_margin`

`instance_liveness_margin(lease: Duration) -> Duration` (`crates/jammi-db/src/catalog/lease.rs:269-271`,
at c1d918b4): `lease.saturating_mul(2)` — a **new** function this unit creates (per the lead's
2026-09-14 amendment folded into the source design contract), named so a second caller
(`fresh_instance` here; U5b-1a's own freshness check later) shares one margin definition rather
than re-deriving the `2 *` factor inline the way `reclaim_expired_jobs`'s inline-execution arm
already did before this extraction. Self-test at `crates/jammi-db/src/catalog/lease.rs:277-288`
pins both the `2×` factor and the `saturating_mul` (never wrapping/panicking) overflow shape at
`Duration::MAX`.

`Catalog::fresh_instance` (`crates/jammi-db/src/catalog/jobs_repo.rs:2020-2047`, at c1d918b4):
primary-key lookup against `instances` (no tenant column on that table, so no tenant predicate to
drop or keep), `true` iff a row exists **and** `NOT (stale_before_clause("last_seen_at", kind,
margin, …))` where `margin = instance_liveness_margin(lease)`. An absent row and a stale row both
read `false` — the caller (a rank checking its coordinator) maps either to the same member-scoped
refusal, disclosing which determinant failed to no one.

### 1.6 Enumerating-caller oracles (`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs`, at c1d918b4)

Two grep-shaped, MASKED, code-only assertions over the **whole tracked tree** (`git ls-files`,
never a hand-rolled walk —
`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::git_ls_files`):
- `only_the_gang_run_rank_handler_calls_get_job_for_rank`
  (`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::only_the_gang_run_rank_handler_calls_get_job_for_rank`): the token
  `get_job_for_rank(` occurs, as CODE (comments and string/char literals masked first,
  `mask_non_code`, `crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::mask_non_code`; an
  identifier-boundary check, `contains_code_token`,
  `crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::contains_code_token`, so a longer identifier
  merely ending in the token — this file's own test-fn names — is not mistaken for a call), only
  in `crates/jammi-db/src/catalog/jobs_repo.rs` (the definition), `crates/jammi-server/src/grpc/gang.rs`
  (the one production caller), and `crates/jammi-db/tests/it/gang_rank_admission.rs` (jammi-db's
  own unit tests) — a fixed allowlist checked BOTH directions (no unexpected hit; no stale
  allowlist entry that no longer hits).
- `only_resolve_training_set_identity_calls_get_result_table_for_tenant` — deleted with the
  world>1 conjunct (Addendum 2; #566): the gang `RunRank` handler no longer
  reaches `get_result_table_for_tenant` at all (Addendum 2 §B1), so there is no gang-adjacent
  caller surface left for this oracle to enumerate. `get_result_table_for_tenant` itself and the
  strict-predicate test that measured it are deleted with that conjunct; the property and its
  rebuild are `HostAdmission`'s (UNITS.md § U5a-2;
  <https://github.com/f-inverse/jammi-ai/issues/566>).

The surviving oracle is a **measured claim**, not prose: `impossibility_claims` (§6, below) states
"no caller other than the gang `RunRank` handler calls `get_job_for_rank`" with this executed
enumeration as the refutation attempt — not an assertion nobody tried to break. (The paired
`get_result_table_for_tenant` claim this section originally measured alongside it is deleted with
the oracle that measured it — Addendum 2 §B1; #566.)

Masking self-tests (never load-bearing on the oracle above passing today, but load-bearing on that
oracle never *silently* stopping being one): `mask_non_code_hides_comments_and_strings_but_not_code`
(`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::mask_non_code_hides_comments_and_strings_but_not_code`) and
`contains_code_token_rejects_a_same_tokened_longer_identifier`
(`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::contains_code_token_rejects_a_same_tokened_longer_identifier`).
Fixture strings in both are annotated `// kernel-oracles: fn-in-literal reviewed: …` — the
c1d918b4 follow-up commit's own concern: a string literal fixture containing the substring
`fn ... {` looks, to a naive scanner, like a function declaration; these comments record that each
such fixture was reviewed and is intentional scaffolding, not a real declaration
`check_kernel_oracles.py` should flag.

### 1.7 Mount, observability, tenant-isolation allowlist

**Mount.** `crates/jammi-server/src/runtime.rs:585-591` (at c1d918b4):
`GangServiceServer::new(GangServer::new(…))` is `.add_service()`d onto the SAME
`tonic::service::Routes` as `PeerServiceServer`, on the internal `[server] peer_bind` listener only
— never the public listener's own `Routes` (built separately, around
`crates/jammi-server/src/runtime.rs:1945`). The `[lease]` window is read once at bind time
(`crates/jammi-server/src/runtime.rs:568-576`) from the already-validated config.

**Counter.** `GANG_PREFIX = "/jammi.v1.gang.GangService/"` (`crates/jammi-server/src/metrics_layer.rs:58`)
increments `jammi_gang_requests_total{rpc}` (field `gang_requests`,
`crates/jammi-server/src/routes/health.rs:156,230-238,250`) at
`crates/jammi-server/src/metrics_layer.rs:119-121` — counted regardless of how the call is
ultimately decided (proven by `run_rank_increments_gang_requests_metric`,
`crates/jammi-server/tests/it/gang_service.rs:398-425`, which counts a call refused at the
wire-level K2 edge, before I-GANG ever runs). Documented at `docs/guide/src/operability.md:48`.

**Tenant-isolation allowlist.** `GANG_LISTENER_ALLOWLIST`
(`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:181-185`) is its own bucket, deliberately
**not** appended to `PEER_LISTENER_ALLOWLIST`
(`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:156-167`, whose own doc calls that bucket
"deliberately tenant-free" — the opposite shape from I-GANG, which derives tenant from the row).
`covered_on_wire` (`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:3119-3137`) and
`allowlist_and_cases_partition_the_wire_surface`
(`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:3181-3210`) both union the new bucket in;
`gang_service_is_unimplemented_on_the_public_listener`
(`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:3316-3346`) proves the exemption's own
premise — the public listener answers `Unimplemented` for `/jammi.v1.gang.GangService/*` — the
same way `peer_service_is_unimplemented_on_the_public_listener`
(`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:3274-3302`) proves it for `PeerService`.

**Docs (one commit set, `efb98d54`, at c1d918b4).** `docs/guide/src/api-stability.md:51,65`
(the wire package table, thirteen packages now); `docs/guide/src/security.md:77-80,111-140`
(new "The gang listener (I-GANG)" section, its own non-disclosure/tenant-derivation/Postgres-only
bullets); `docs/guide/src/configuration.md:177,191` (`peer_bind` key doc names `GangService`
beside `PeerService`); `docs/guide/src/deploy-server.md:468` ("The peer listener (I-PEER /
I-GANG)"); `docs/guide/src/reference-topologies.md:292-297` (`peer_bind` bullet renamed "makes a
replica an owner (and a gang admission member)"); `docs/guide/src/operability.md:48` (the counter
row, above). `check_doc_parity.py` and `check_no_consumer_names.py` both pass on this diff (§7,
below).

---

## 2. Properties (quantified, never a single-input claim)

**P1 — I-GANG is served iff a two-conjunct property, never one.** For every `RunRank` call whose
wire-level K2 edges pass: `Admitted`-eligible (in U5a-2's terms) / `Unimplemented` (in this unit)
iff (a) `get_job_for_rank(job_id)` returns a row with `status = running`, `claimed_by =
assign.coordinator_instance_id`, `attempts == assign.attempt`, `lease_live`, and — for `world > 1`
only — both training-set columns `Some` — **and** (b), for `world > 1` only,
`resolve_training_set_identity` resolves `Ok(())` against that row's own pair. Neither conjunct
alone is sufficient; §1.2 traces every branch of the handler to show both are actually evaluated
(not short-circuited past) whenever the input reaches that far, and
`run_rank_every_i_gang_determinant_satisfied_is_unimplemented` (§3) exercises the case where every
conjunct holds.

**P2 — Non-disclosure.** Every I-GANG refusal (conjunct (a)'s four sub-checks, conjunct (b), and
coordinator freshness) is the SAME `Status::failed_precondition` with the SAME fixed message
string (`I_GANG_REFUSAL_MESSAGE`, one constant, `crates/jammi-server/src/grpc/gang.rs:53`) —
quantified over every reachable refusal branch in `run_rank`, not just the ones this file's tests
happen to name.

**P3 — Tenant is derived, never accepted.** No code path in `run_rank` or
`resolve_training_set_identity` reads a tenant value from `RankControl`/`Assign` (neither message
declares one, §1.1) or from any per-call caller metadata; the only tenant value used is
`row.tenant_id`, read from `get_job_for_rank`'s own row.

**P4 — The training-set identity pair is one fact.** For every `UPDATE jobs` statement this
program issues against `training_set_ref`/`training_set_location` (the CAS at
`crates/jammi-db/src/catalog/jobs_repo.rs:1821-1836` is the *only* production writer, per the
enumerating-caller-oracle-shaped methodology §1.6 applies to the two verbs it names — no second
writer of this pair exists in this crate), the pair transitions NULL→(v1,v2) atomically or not at
all; independently, the schema `CHECK` (§1.4) refuses ANY statement, from ANY writer present or
future, that would leave the pair split — a property enforced at two independent layers (the sole
writer's own predicate, and the schema itself), never one alone.

**P5 — `fresh_instance`'s margin is `2×` the deployment lease, shared with `reclaim_expired_jobs`'s
existing inline tolerance**, never re-derived at this call site
(`crates/jammi-db/src/catalog/lease.rs:262-271`'s own stated purpose).

**P6 — Additive freeze.** Every new proto symbol (§1.1) is additive to the compiled descriptor set;
`api_freeze`'s two tests (`wire_surface_equals_the_frozen_baseline`,
`manifest_format_version_is_frozen`, `crates/jammi-server/tests/it/api_freeze.rs:82-104`) both pass
unmodified in shape (only the baseline file's data grew) — no existing `PACKAGE`/`RPC` line was
renamed or removed.

---

## 3. Oracles, by name, and what each EXCLUDES

All paths below are `crates/jammi-server/tests/it/gang_service.rs` unless stated otherwise; all at
c1d918b4.

| Oracle | Asserts | Excludes |
|---|---|---|
| `run_rank_refuses_world_zero` (`:366-390`) | `world == 0` ⇒ `InvalidArgument`, message names the SPECIFIC edge (not the `rank>=world` message this exact input also trips) | Does not test `rank >= world` with a nonzero world; does not reach I-GANG at all (refused before any DB read) |
| `run_rank_refuses_rank_at_world_boundary` (`:430-442`) | `rank == world` (the boundary, not just wildly out of range) ⇒ `InvalidArgument` | Does not test `rank < world` (the admitted shape, covered elsewhere) |
| `run_rank_refuses_a_stream_opening_with_cancel` (`:447-…`, same file) | A stream whose first frame is `Cancel`, not `Assign` ⇒ `InvalidArgument` | Does not test a `Cancel` AFTER a valid `Assign` (that is U5a-2's admitted-stream K2 row, not built here) |
| `run_rank_refuses_a_stream_closed_before_assign` (`:467-…`, same file) | Stream ends (`None`) before any frame ⇒ `InvalidArgument`, distinct from the timeout path | Does not test the bounded-timeout arm (silent client, no close) directly — that is the `Err(_)` timeout branch, `crates/jammi-server/src/grpc/gang.rs:105-109`, exercised implicitly by the 10s bound never firing in these fast tests, not explicitly timed here |
| `run_rank_every_i_gang_determinant_satisfied_is_unimplemented` (`:495-522`) | Every I-GANG conjunct genuinely satisfied ⇒ `Unimplemented` (f1') — proves determinants are DECIDED, not skipped past on an easier earlier refusal | Excludes `world > 1` (this row's fixture is `world=1`); the `world > 1` sidecar-verify success path is not exercised by an end-to-end `RunRank` call anywhere in this file — only via the two b1'/§1.2 unit-level calls to `resolve_training_set_identity` directly (`:63-130`, `:144-281`) |
| `run_rank_refuses_when_job_not_found` (`:527-540`) | Absent job ⇒ `FailedPrecondition`, SAME status/message as every other determinant | Does not distinguish "not found" from any other refusal by message — that IS the property |
| `run_rank_refuses_when_job_not_running` (`:544-…`, same file) | `queued` (never claimed) row ⇒ refused, isolated from freshness (coordinator upserted first) | Does not cover `completed`/`failed` terminal rows explicitly — only the `queued` non-running case |
| `run_rank_refuses_when_lease_expired` (`:601-…`, same file) | Expired lease ⇒ refused | Does not cover the NULL-lease sub-case as its own row inside this file (that shape is covered at the DB layer, `crates/jammi-db/tests/it/gang_rank_admission.rs`'s own `get_job_for_rank_treats_a_null_lease_as_not_live`, §3 below, not re-proven through the wire here) |
| `run_rank_refuses_when_coordinator_not_fresh` (`:643-…`, same file) | Coordinator's `instances` row absent/stale ⇒ refused, decided AFTER the row predicate passes | Does not distinguish absent-row from stale-row (both are `fresh_instance == false`, by design, §1.5) |
| `run_rank_refuses_when_attempt_does_not_match` (`:679-…`) / `run_rank_refuses_when_claimed_by_a_different_coordinator` (`:718-…`, same file) | Each isolates ONE conjunct of §I1(a) at a time | Neither combines two false conjuncts in one call (each row is a single-determinant probe by construction) |
| `run_rank_increments_gang_requests_metric` (`:398-425`) | The counter increments on a call refused at the WIRE level (before I-GANG runs) | Does not assert the counter for an I-GANG-refused or a would-be-admitted call — the "regardless of outcome" claim is proven at the cheapest refusal point, not re-proven at every later one |
| `strict_resolver_never_matches_a_null_tenant_row_for_a_real_tenant` (`:63-130`) | (i) the RELAXED `get_result_table` DOES leak a NULL-tenant row to a real tenant (the pre-existing hazard, still true); (ii) the STRICT verb does NOT; (iii) the resolution wrapper refuses `FailedPrecondition` | Does not exercise this through the wire (`RunRank`) — calls `resolve_training_set_identity` directly; does not cover a NULL-tenant CALLER (only a real tenant A resolving a NULL-tenant row) |
| `resolution_site_refuses_under_admin_scope_even_when_the_raw_verb_would_resolve` (`:144-281`) | (i) the raw strict verb's OWN admin-scope behaviour is unchanged (still resolves cross-tenant under admin scope — the hazard ruling 4 names, proven still present); (ii) the resolution-site GUARD refuses unconditionally, before the verb is ever called, even though every other conjunct (status `ready`, matching digest) genuinely holds | Does not test admin scope with a table that would ALSO fail verification for other reasons (the fixture is a genuinely valid, ready, correctly-materialized table for tenant B, specifically so the guard — not a downstream check — is what refuses) |
| `a_raw_single_column_write_is_refused_by_the_schema_check` (`crates/jammi-db/tests/it/gang_rank_admission.rs:387-412`) | SQLite refuses a lone `training_set_ref` write via the `CHECK` | Does not test the `training_set_location`-alone direction (covered instead by the migrations test, next row) nor Postgres |
| `migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge` (`crates/jammi-db/tests/it/migrations.rs:1571-1745`) | Ordering (033 after 032, by relative position — K5, never `.last()`); BOTH single-column directions refused; parametrized over SQLite (always) and Postgres (`--features live-postgres-tests`, skips cleanly without `JAMMI_TEST_PG_URL`) | Postgres arm is a no-op locally without a live Postgres; CI's own `live-postgres-tests` lane is what actually exercises it there |
| `get_job_for_rank_returns_none_for_an_absent_job` / `_reflects_a_live_claim` / `_treats_a_null_lease_as_not_live` / `_treats_an_expired_lease_as_not_live` / `_returns_the_filled_pair` (`crates/jammi-db/tests/it/gang_rank_admission.rs:62-175`) | The five row-predicate edges at the DB layer directly, including the lease-boundary pair (NULL vs. expired, both `lease_live == false`) | None of these five drive the wire; the wire-level lease determinant is proven ONCE end-to-end (`run_rank_refuses_when_lease_expired`) and not re-proven per lease sub-case through gRPC |
| `fill_training_set_identity_first_call_fills` / `_second_call_same_values_reuses` / `_concurrent_racer_reuses_never_overwrites` (`crates/jammi-db/tests/it/gang_rank_admission.rs:181-282`) / `_moved_claim_aborts_without_a_terminal_write` (`crates/jammi-db/tests/it/gang_rank_admission.rs:287-334`) / `_matching_claim_different_pair_aborts` (`crates/jammi-db/tests/it/gang_rank_admission.rs:341-…`) | Filled / Reused (idempotent retry) / Reused (genuine `tokio::spawn` race, asserting exactly one `Filled` and one `Reused` outcome) / Aborted-on-moved-claim (no terminal write, status untouched) / Aborted-on-differing-pair (isolates "different pair" from "claim moved" — a shape this program's own CAS can never construct via its own writes, since the predicate always requires the pair NULL first, so this row is manufactured directly) | The concurrent racer test proves the DB-level CAS is race-safe; it does not drive two concurrent `RunRank` calls through the wire (no such concurrent-caller path exists at U5a-1 — the CAS's own caller, U5b-1b-ii's materialization step, is not built yet) |
| `fresh_instance_true_for_a_recently_seen_instance` / `_false_for_an_absent_instance` / `_false_for_a_stale_instance` / `_true_just_inside_the_liveness_margin` (`crates/jammi-db/tests/it/gang_instance_freshness.rs:63-125`) | The margin's own boundary (`2×lease` exactly, both sides) | Does not go through `GangServer::run_rank`'s own freshness call — that is `run_rank_refuses_when_coordinator_not_fresh`, above, which does not re-probe the exact `2×` boundary (only absent/stale, not the borderline-inside case) |
| `only_the_gang_run_rank_handler_calls_get_job_for_rank` (`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::only_the_gang_run_rank_handler_calls_get_job_for_rank`) | No caller of `get_job_for_rank` exists outside the named allowlist, TODAY, over the whole tracked tree | Cannot prove no FUTURE caller will be added silently — it re-fails the moment one is, which is the property it actually offers (a standing tripwire, not a proof about the future); `only_resolve_training_set_identity_calls_get_result_table_for_tenant`, this row's original pair, is deleted with the world>1 conjunct (Addendum 2; #566) |
| `mask_non_code_hides_comments_and_strings_but_not_code` / `contains_code_token_rejects_a_same_tokened_longer_identifier` (`crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::mask_non_code_hides_comments_and_strings_but_not_code` / `crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::contains_code_token_rejects_a_same_tokened_longer_identifier`) | The oracle above cannot be defeated by a comment, a string literal, or a same-tokened longer identifier (this file's own test-fn names) | Does not cover every conceivable Rust lexical edge case (e.g., a token split across a raw-string continuation) — masks line comments, block comments (non-nested), plain and raw string/char literals only, matching the `whose_fault_gate.rs` precedent's own stated limit |
| `wire_surface_equals_the_frozen_baseline` / `manifest_format_version_is_frozen` (`crates/jammi-server/tests/it/api_freeze.rs:82-104`) | The compiled descriptor set matches the baseline set-equal (as SETS, order-independent) | Does not check message/field shape (the freeze guard's own stated limit, §1.1) |
| `gang_service_is_unimplemented_on_the_public_listener` (`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:3316-3346`) | The public listener refuses `Unimplemented` for `GangService/RunRank`; the allowlist entry's own `why` text carries the required I-GANG sentence | Does not test any OTHER method on a hypothetical future `GangService` extension — only `RunRank`, the only rpc that exists |
| `allowlist_and_cases_partition_the_wire_surface` (`crates/jammi-server/tests/it/tenant_isolation_oracle.rs:3181-3210`) | Every case + both allowlists partition the wire surface with NO overlap | Does not itself prove either allowlist's `why` text is accurate — that is the two `*_is_unimplemented_on_the_public_listener` tests' job |

---

## 4. Mutations executed / implied by the oracle shapes above

- **Running conjunct.** Flip a claimed job's `status` off `running` (test setup, not a code
  mutation — `run_rank_refuses_when_job_not_running`) — the conjunct is what refuses, not a
  coincidental earlier check; `run_rank_every_i_gang_determinant_satisfied_is_unimplemented`'s own
  doc comment (`crates/jammi-server/tests/it/gang_service.rs:490-494`) states the design intent
  explicitly: a call satisfying every OTHER conjunct must not stop short on an easier one, proven
  by satisfying ALL of them together and observing the SAME success shape (`Unimplemented`, never
  an earlier refusal that would indicate a determinant was skipped).
- **Schema `CHECK` clause.** Deleting or weakening the `CHECK` at
  `crates/jammi-db/src/catalog/schema.rs:1167-1168` is exactly what
  `a_raw_single_column_write_is_refused_by_the_schema_check` and
  `migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge` are built to catch —
  both assert the single-column `UPDATE` **errors**; removing the constraint flips both green
  assertions to a silent success, i.e. both tests fail loudly (`err.is_err()` / the explicit
  `assert!` on the pg/sqlite error) rather than passing vacuously.
- **`margin = lease` (not `2×lease`).** `instance_liveness_margin_is_twice_the_lease`
  (`crates/jammi-db/src/catalog/lease.rs:277-288`) pins the factor numerically
  (`Duration::from_secs(5)` → `Duration::from_secs(10)`, never `Duration::from_secs(5)`);
  `fresh_instance_true_just_inside_the_liveness_margin`
  (`crates/jammi-db/tests/it/gang_instance_freshness.rs:110-125`, staleness `50s` against
  `lease=30s`, margin `60s`) would flip to `false` under a `margin = lease` (30s) mutation, since
  `50s > 30s`. Both together bound the factor from two independent test files, not one.
- **Planted code caller.** The enumerating-caller oracle's own allowlist-both-directions shape
  (§1.6) IS the executed mutation-detector: adding any new call site to `get_job_for_rank`,
  anywhere in the tracked tree outside the allowlist, flips
  `only_the_gang_run_rank_handler_calls_get_job_for_rank` from green to a named
  failure identifying the offending file — exercised at authorship time by temporarily adding a
  throwaway `get_job_for_rank(` call to a fourth file and observing the named failure (the standard
  `whose_fault_gate.rs`-precedent methodology this oracle file's own module-level doc comment
  cites, `crates/jammi-server/tests/it/gang_rank_admission_oracle.rs`), then reverting it; the
  c1d918b4 follow-up commit is the fixture-review pass that confirmed every remaining `fn`-shaped
  string literal in the masking self-tests is reviewed scaffolding, not a missed real call site.
  (The paired `only_resolve_training_set_identity_calls_get_result_table_for_tenant` oracle this
  paragraph originally also named is deleted with the world>1 conjunct — Addendum 2; #566.)
- **Claimant / attempt / lease / not-found / not-fresh conjuncts.** Each has its OWN isolated test
  (§3 table) that holds every other conjunct at its satisfied value and flips exactly one — the
  mutation is the test fixture's own single changed field (wrong `claimed_by`, wrong `attempts`,
  expired/NULL lease, absent job, stale/absent `instances` row), never a combined multi-conjunct
  probe that could hide which one actually did the refusing.
- **The CAS predicate.** `fill_training_set_identity_matching_claim_different_pair_aborts`
  manufactures, via a raw `UPDATE` outside the CAS, the one state the CAS's own predicate can never
  reach through its own writes (a pair already set to DIFFERENT values under the SAME
  `claimed_by`/`attempts`) — proving the re-read's value-comparison arm
  (`crates/jammi-db/src/catalog/jobs_repo.rs:1864-1865`, the `tsref == training_set_ref && tsloc ==
  training_set_location` conjunct), not just the claimant/attempt comparison, is load-bearing:
  deleting that conjunct alone (leaving only the claimant/attempt check) would flip this specific
  test's expected `Aborted` to `Reused`.

---

## 5. What is NOT in this unit

- **U5a-2's entire surface**: `HostAdmission`, the `holder` cell, drain, re-verification, the
  `select!`-driven bidi hold loop, admit-and-hold, the park bound, and every `Aborted{reason}`
  EMISSION (the reasons are frozen on the wire, §1.1, but nothing in this unit ever constructs
  one). Every call that clears I-GANG in this unit ends `Unimplemented` (§1.2 step 6), never
  `Admitted`.
- **U5b-1a**: `instances.peer_addr`, `canonicalize_result_root`, `list_gang_members`, the coarse
  root pre-filter at listing time.
- **U5b-1b-i/ii**: the round protocol, body-level byte attestation, the coordinator (membership →
  assignment → dispatch → assembly), `RankEvent::Outcome`'s producer and consumer, the
  `world_size == 1` rank body, the writer-role split, `[worker] assembly_attempts`.
- **The second-`Assign`-on-an-admitted-stream K2 row**: requires the admitted-stream state U5a-2
  builds; not testable here (no stream is ever admitted in this unit).
- **A test-hooks last-refusal-reason observation seam**: not built, and not needed at this unit —
  I-GANG's own non-disclosure property (P2) means there is nothing meaningful to observe yet;
  every b1' row here is distinguished by test setup, never by an assertion on a hidden reason.

---

## 6. Report

`impossibility_claims`:
- "no caller other than the gang `RunRank` handler calls `get_job_for_rank`" — executed
  enumeration: `only_the_gang_run_rank_handler_calls_get_job_for_rank`,
  `crates/jammi-server/tests/it/gang_rank_admission_oracle.rs::only_the_gang_run_rank_handler_calls_get_job_for_rank`
  (at c1d918b4), scanning `git ls-files` over the whole tracked tree, masked to code only. Result
  set at c1d918b4: exactly `crates/jammi-db/src/catalog/jobs_repo.rs`,
  `crates/jammi-server/src/grpc/gang.rs`, `crates/jammi-db/tests/it/gang_rank_admission.rs`.
- "no caller other than the gang `RunRank` handler resolves `training_set_location`" (i.e. calls
  `get_result_table_for_tenant`) — the oracle that measured this claim,
  `only_resolve_training_set_identity_calls_get_result_table_for_tenant`, is deleted with the
  world>1 conjunct (Addendum 2; #566): the gang `RunRank` handler no longer calls
  `get_result_table_for_tenant` at all in this unit (Addendum 2 §B1), so there is no
  gang-adjacent caller surface left to enumerate here; `get_result_table_for_tenant` itself and
  the strict-predicate test that measured it are deleted with the same conjunct, and the property
  and its rebuild are `HostAdmission`'s (UNITS.md § U5a-2;
  <https://github.com/f-inverse/jammi-ai/issues/566>).
- "the write-once training-set pair is ever set with one column null and the other not" —
  executed attempt: `a_raw_single_column_write_is_refused_by_the_schema_check`
  (`crates/jammi-db/tests/it/gang_rank_admission.rs:387-412`) and
  `migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge`
  (`crates/jammi-db/tests/it/migrations.rs:1571-1745`, both dialects) — both directions attempted
  directly against the schema, both refused.
- "an admitted `RunRank` session reaches a run path in this unit" — executed attempt: none
  possible by construction — `run_rank` has exactly one success terminal
  (`Status::unimplemented`, `crates/jammi-server/src/grpc/gang.rs:184-186`) and no
  `HostAdmission` type exists in this unit's dependency graph to construct one from (`grep -rn
  HostAdmission crates/` at c1d918b4 returns no hits under `crates/jammi-server/src` or
  `crates/jammi-ai/src` — confirmed at authorship time).

`citations_reanchored`: every citation in this document was read directly against `c1d918b4` in
this worktree by the writing agent, not carried from the source design contract's own (`wt-B`)
line numbers — where a number matches the source contract exactly (e.g.
`crates/jammi-db/src/store/mod.rs:1267`, `:3668`) it is because this branch's history runs through
the same `wt-B`-equivalent base for those files, confirmed independently rather than assumed
identical (§1.2 traces the exact call chain through `resolve_training_set_identity`).

---

## 7. Gates

`cargo test -p jammi-server --test it` (whole suite); `cargo test -p jammi-server --features
test-hooks --test it` (same suite, the test-hooks feature union, `crates/jammi-server/Cargo.toml:122,136`);
`cargo test -p jammi-db --features postgres`; `cargo test -p jammi-db --test it` (the
`gang_rank_admission*` / `gang_instance_freshness` / `migrations` targets); workspace clippy incl.
gated lanes; `cargo fmt --check`; `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p jammi-wire -p
jammi-server -p jammi-db`; `check_doc_parity.py`; `check_no_consumer_names.py`;
`check_dep_direction.py`; `check_swarm_bijection.py`.

---

## Addendum — the shipped shape at `77c15f10`

Sections 1–7 above are the pinned record of what `c1d918b4` stated and are left
unedited. This addendum states the mechanism as it EXISTS at `77c15f10` — every
sentence below was read directly against that commit in this worktree by the
agent writing this addendum, cited by crate-qualified `path::item`.

### A1. The lattice is keyed on the row's own `world_size`, never `assign.world`

`crates/jammi-db/src/catalog/jobs_repo.rs::RankAdmissionRow` carries the
field `world_size: u32`, decoded by the module-private
`crates/jammi-db/src/catalog/jobs_repo.rs::world_size_from_spec_json` from
the job's own `spec` JSON: a top-level `world_size` key, or one nested under
`common`; absent either way ⇒ the const
`crates/jammi-db/src/catalog/jobs_repo.rs::WORLD_SIZE_IF_ABSENT` (`= 1`);
present but not decodable as a `u32` ⇒ `BackendError::TypeConversion`, never
a silent `1`. Pinned by three oracles in
`crates/jammi-db/tests/it/gang_rank_admission.rs`:
`get_job_for_rank_reflects_the_row_world_size` (a spec naming `world_size: 2`
round-trips `2`), `get_job_for_rank_defaults_world_size_when_absent_from_spec`
(a spec naming nothing round-trips `1`), and
`get_job_for_rank_malformed_world_size_is_a_typed_error` (a non-numeric
`world_size` is a typed `Err`, never a silent `1`).

`crates/jammi-db/src/catalog/jobs_repo.rs::Catalog::get_job_for_rank` is
primary-key-only (`WHERE job_id = $N`) and decides nothing beyond that
lookup — its own doc comment states explicitly that every determinant the
returned row feeds, including `world_size`, is the CALLER's
(`GangServer::run_rank`) to decide, never a caller-supplied `Assign.world`
standing in for it.

`crates/jammi-server/src/grpc/gang.rs::GangServer::run_rank` (~150-line
`impl GangService` method) keys the lattice on `row.world_size`: after the
row's `status`/`claimed_by`/`attempts`/`lease_live` conjuncts,
`assign.world != row.world_size` is itself a refusal with the SAME fixed
message (an inline comment states why: keying the pair conjunct on
`assign.world` instead would let a `world_size > 1` job admit under a
caller-supplied `world = 1`, skipping the pair conjunct and the sidecar
verify entirely). Only once that conjunct holds does `row.world_size > 1`
gate the training-set pair conjunct and the sidecar verify — never
`assign.world`.

The pairwise conjunct (`world > 1`) is described the same way in
`docs/maintainer/MAINTAINER-GUIDE.md#28a-gangservice--multi-host-gang-admission-i-gang`
("2.8a"): "the ROW-keyed lattice" — rewritten by this round from the pinned
contract's assign-keyed description; that same section's `get_job_for_rank`
entry states it "decides nothing itself and returns `Ok(None)` only when no
job with that id exists."

### A2. `TrainingSetOutcome` and the six RPC oracles a–f

`crates/jammi-server/src/grpc/gang.rs::resolve_training_set_identity_classified`
classifies into `crates/jammi-server/src/grpc/gang.rs::TrainingSetOutcome`
(`Verified`, `AdminScopeRefused`, `OtherTenant`, `NotReady`,
`DigestMismatch`). `crates/jammi-server/src/grpc/gang.rs::resolve_training_set_identity`
is a thin wire wrapper collapsing every non-`Verified` outcome to the fixed
`FailedPrecondition` (§I1 Non-disclosure), except `AdminScopeRefused`'s own
message (a distinct wire message this outcome alone carries — a resolution
wrapped in `with_admin_scope`, never reachable through `run_rank`'s own
tenant-derived path in production, per the `TrainingSetOutcome::AdminScopeRefused`
variant's own doc comment).

The six determinants R2 named, through the wire, by test name (all
`crates/jammi-server/tests/it/gang_service.rs`):
- (a) pair NULL at `world_size` 2 →
  `run_rank_refuses_world_gt_one_when_training_set_pair_missing`.
- (b) pair set, another tenant's / NULL-tenant row →
  `run_rank_refuses_a_training_set_another_tenant_owns` and
  `run_rank_refuses_a_null_tenant_training_set_for_a_tenant_bound_job`.
- (c) own tenant + Ready + digest match →
  `run_rank_world_two_own_tenant_training_set_reaches_unimplemented`.
- (d) own tenant + digest MISMATCH →
  `run_rank_refuses_world_gt_one_when_training_set_digest_mismatches`.
- (e) own tenant + status not Ready →
  `run_rank_refuses_world_gt_one_when_training_set_not_ready`.
- (f) `assign.world` (1) ≠ `row.world_size` (2) →
  `run_rank_refuses_when_assign_world_mismatches_row_world_size`.

### A3. Non-disclosure: `GangRefusalReason`, the pairwise oracle, the test-hooks seam

`crates/jammi-server/src/grpc/gang.rs::GangRefusalReason` names eleven
determinants (`NotRunning`, `WrongClaimant`, `WrongAttempt`, `LeaseDead`,
`NotFound`, `WorldMismatch`, `TrainingSetPairMissing`,
`TrainingSetOtherTenant`, `TrainingSetNotReady`, `TrainingSetDigestMismatch`,
`CoordinatorNotFresh`) — defined unconditionally (no `#[cfg]` on the enum
itself); `crates/jammi-server/src/grpc/gang.rs::GangServer::record_refusal`
stamps one at every refusal site and is a no-op outside `test-hooks` (only
the `GangServer::last_refusal` field and the `GangServer::last_refusal_reason`
getter are gated). The pairwise non-disclosure oracle,
`run_rank_refusal_is_non_disclosing_across_every_determinant`
(`crates/jammi-server/tests/it/gang_service.rs`), drives all eleven scenarios
(the `every_gang_refusal_reason` helper, same file) via the shared fixture
builder `refusal_scenario` (~290-line fixture builder, same file) and
asserts `(code, message)` pairwise identical across every pair. The
`test-hooks`-only reason-distinguishing oracle,
`run_rank_last_refusal_reason_distinguishes_every_determinant`
(`crates/jammi-server/tests/it/gang_service.rs`, `#[cfg(feature =
"test-hooks")]`), drives the SAME eleven scenarios and asserts the served
instance's recorded reason matches, through
`crates/jammi-server/tests/it/common/grpc.rs::PeerEngineServer::gang_last_refusal_reason`,
which reads `crates/jammi-server/src/grpc/gang.rs::GangRefusalHandle::get` —
a handle `crates/jammi-server/src/grpc/gang.rs::GangServer::refusal_reason_handle`
clones out of the SAME `Arc<Mutex<..>>` the mounted, serving instance
mutates, cloned BEFORE `GangServer` moves by value into
`GangServiceServer::new` (inside
`crates/jammi-server/src/runtime.rs::OssServer::bind`) — threaded as
`Option<GangRefusalHandle>` through
`crates/jammi-server/src/runtime.rs::BoundServer` (field
`gang_refusal_handle`, getter `BoundServer::gang_refusal_handle`) into
`crates/jammi-server/tests/it/common/grpc.rs::PeerEngineServer` (field
`gang_refusal_handle`).

The plain `cargo test -p jammi-server --test it -- gang` lane executes 27
gang-named cases; the `--features test-hooks` lane executes 28 — one more,
`run_rank_last_refusal_reason_distinguishes_every_determinant`, invisible to
the plain lane by construction (its `#[cfg(feature = "test-hooks")]` gate
compiles it away otherwise). The 27/28 split is over every test-fn whose
qualified name contains `gang`: 22 plain + 1 `test-hooks`-gated in
`gang_service.rs`, 4 in `gang_rank_admission_oracle.rs`, and 1
(`gang_service_is_unimplemented_on_the_public_listener`,
`crates/jammi-server/tests/it/tenant_isolation_oracle.rs`) in
`tenant_isolation_oracle.rs` — 22+4+1 = 27 plain, +1 test-hooks-only = 28.

### A4. Admission catalog faults → `Unavailable`; no injectable fault seam

`crates/jammi-server/src/grpc/gang.rs::admission_catalog_fault` maps a
genuine catalog fault to `Status::unavailable(..)`, never
`map_engine_error`'s generic mapping. Two call sites: `get_job_for_rank`
erroring inside `crates/jammi-server/src/grpc/gang.rs::GangServer::run_rank`
(`Err(e) => admission_catalog_fault(e)`) and `get_result_table_for_tenant`
erroring inside
`crates/jammi-server/src/grpc/gang.rs::resolve_training_set_identity_classified`
(`.map_err(admission_catalog_fault)`). No test in this tree exercises either
call site's error arm — `grep -rn admission_catalog_fault
crates/jammi-server/tests/` (confirmed at this commit) returns no hits —
because no fault-injection seam exists in `jammi-db`'s catalog backend
reachable from this crate's `it` harness without touching `jammi-db`, out of
this round's wire-server scope; the mapping is stated, not tested, matching
`docs/maintainer/MAINTAINER-GUIDE.md#28a-gangservice--multi-host-gang-admission-i-gang`'s
own wording ("A genuine catalog fault during admission is `Unavailable`, not
`FailedPrecondition`").

### A5. `RunRank` on the streaming-path allowlist, its own `MethodClass` arm

`crates/jammi-server/src/limits.rs::RUN_RANK_PATH` is
`"/jammi.v1.gang.GangService/RunRank"`;
`crates/jammi-server/src/limits.rs::is_streaming_path` recognizes it as a
third server-streaming path beside `WAIT_JOB_PATH`/`SUBSCRIBE_PATH`, pinned
equal to the descriptor-derived server-streaming set by
`crates/jammi-server/src/limits.rs::is_streaming_path_allowlist_matches_the_descriptor_derived_server_streaming_set`,
deriving the set from `jammi_wire::FILE_DESCRIPTOR_SET`'s
`MethodDescriptorProto::server_streaming` flag.
`crates/jammi-server/src/limits.rs::MethodClass::call`'s own arm for `path ==
RUN_RANK_PATH` applies ONLY the `deadline` treatment every streaming path
gets (`wait_timeout_secs`) — no stream-count budget (no `server.limits.max_*`
knob exists for it, per that arm's own comment), never falling into
`Subscribe`'s or `WaitJob`'s budget. In production this arm never runs:
`GangService` is mounted only on the internal `[server] peer_bind` listener,
built LAYER-FREE — the peer-listener arm inside
`crates/jammi-server/src/runtime.rs::BoundServer::serve_with_signals` builds
its `Server` carrying only `MetricsLayer` (a comment there states "No tenant
layer, no gRPC-web framing, no `[server.limits]` stack"), never
`MethodClassLayer` — matching `crates/jammi-server/src/limits.rs`'s own
module doc (immediately above `is_streaming_path`) statement of this fact.

### A6. Stated limits carried forward unchanged by this round

- `world > 1`'s pair-and-verify conjunct remains U5a-2's own `HostAdmission`
  session to hold and re-verify (§5, above) — this round only changed WHICH
  field gates it (the row's `world_size`, never `assign.world`); U5a-2's scope
  is otherwise unchanged.
- `crates/jammi-db/src/catalog/lease.rs::lease_remaining_seconds_expr`'s
  SQLite `julianday` arm carries roughly sub-100-microsecond rounding
  relative to `lease_expired_clause`'s exact string compare (documented on
  that same function) — negligible at deployment lease scales, stated
  honestly rather than claimed bit-exact.
- The migration is `034_jobs_training_set_identity`: it was `033` on this unit's own base
  only, and the consolidation into PR-B2 (where U3's `033_model_materialization` precedes it)
  renumbered it once, at the pin sites and the ordered-after oracle named in § 1.4 above —
  `migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge`
  (`crates/jammi-db/tests/it/migrations.rs`) asserts the position relative to `033`, never
  `.last()`.

### A7. Mutations executed, as properties, by named test killed

- **A caller-keyed gate (`assign.world` instead of `row.world_size`)** flips
  `run_rank_refuses_when_assign_world_mismatches_row_world_size`
  (`crates/jammi-server/tests/it/gang_service.rs`) from refused to admitted
  (the row's own `world_size = 2`, `assign.world = 1`, no longer caught).
- **Deleting the `row.world_size > 1` block entirely** flips at least three
  tests green-to-broken (all `crates/jammi-server/tests/it/gang_service.rs`):
  `run_rank_refuses_world_gt_one_when_training_set_pair_missing`,
  `run_rank_refuses_world_gt_one_when_training_set_not_ready`, and
  `run_rank_refuses_world_gt_one_when_training_set_digest_mismatches` — each
  fixture's pair/status/digest fault is never reached, so the call instead
  admits (`Unimplemented`) where it must refuse.
- **The sidecar verify made vacuously `Ok(TrainingSetOutcome::Verified)`**
  kills `run_rank_refuses_world_gt_one_when_training_set_digest_mismatches`
  specifically — a genuine digest mismatch would no longer refuse.
- **The `Ready` conjunct dropped** (`ResultTableStatus::Ready` check removed
  from
  `crates/jammi-server/src/grpc/gang.rs::resolve_training_set_identity_classified`)
  kills `run_rank_refuses_world_gt_one_when_training_set_not_ready`
  specifically.
- **`row.tenant_id` forced to `None`** (or the strict predicate's tenant bind
  dropped) kills `run_rank_refuses_a_training_set_another_tenant_owns`
  (`crates/jammi-server/tests/it/gang_service.rs`) — a NULL-tenant caller
  would then resolve a real tenant's row.
- **A leaking refusal message** (any one `GangRefusalReason` arm's status
  interpolating a job id, claimant, or reason into `I_GANG_REFUSAL_MESSAGE`)
  kills `run_rank_refusal_is_non_disclosing_across_every_determinant`
  (`crates/jammi-server/tests/it/gang_service.rs`), naming the exact
  differing pair.
- **A `world_size` decode that silently coerces a malformed value to `1`**
  kills `get_job_for_rank_malformed_world_size_is_a_typed_error`
  (`crates/jammi-db/tests/it/gang_rank_admission.rs`); **a decode that
  defaults an ABSENT field to anything but `1`** kills
  `get_job_for_rank_defaults_world_size_when_absent_from_spec` (same file);
  **a decode that fails to read a genuinely present `world_size`** kills
  `get_job_for_rank_reflects_the_row_world_size` (same file).

---

## Addendum 2 — the shipped shape at `1f2ab6ba`

The addendum above states the mechanism as it existed at `77c15f10`; the
sentences below supersede it on the training-set world-gate and the
`world_size` decode — every other sentence there (the wire freeze, §A5's
streaming-allowlist arm, §A6's carried-forward limits) is unaffected and
remains its own record. This addendum states the mechanism as it EXISTS at
`1f2ab6ba` — every sentence below was read directly against that commit in
this worktree by the agent writing this addendum, cited by crate-qualified
`path::item`.

### B1. The training-set pair conjunct and its sidecar verify are gone from this unit

`crates/jammi-server/src/grpc/gang.rs::GangServer::run_rank` no longer
resolves training-set identity at all: `resolve_training_set_identity`,
`resolve_training_set_identity_classified`, and the `TrainingSetOutcome`
type they returned do not exist anywhere in `crates/` — confirmed by
`grep -rn 'TrainingSetOutcome\|resolve_training_set_identity' crates/`
returning no hits. `GangRefusalReason` accordingly drops
`TrainingSetPairMissing`, `TrainingSetOtherTenant`, `TrainingSetNotReady`,
and `TrainingSetDigestMismatch`, and gains none in their place: the enum now
names exactly ten determinants — `AdminScope`, `NotFound`, `NotRunning`,
`WrongClaimant`, `WrongAttempt`, `LeaseDead`, `SpecUndecodable`,
`WorldMismatch`, `MultiHostUnsupported`, `CoordinatorNotFresh`. A row whose
own `world_size` decodes successfully to a value other than `1` refuses
under `MultiHostUnsupported` unconditionally, whether or not the caller's
`assign.world` happens to agree with it — the pair conjunct and the sidecar
verify that would admit a genuine multi-host row are recorded, on the
variant's own doc comment and in
`docs/plans/67-distributed-training/UNITS.md` § U5a-2, as `HostAdmission`'s
to build, filed at <https://github.com/f-inverse/jammi-ai/issues/566>.
`crates/jammi-server/tests/it/gang_service.rs::every_gang_refusal_reason`
forces every `GangRefusalReason` variant into `assert_every_variant_is_a_witness`'s
own exhaustive match, which has no wildcard arm over the enum's real variant
set: a future variant fails that file to compile, naming the missing match
arm, until a matching arm is added. That match re-validates each of
`WITNESSES`'s own hand-listed entries one way — every witness IS a real
variant — but does not, by itself, force a new variant INTO `WITNESSES`:
executed check — adding a variant with only its match arm, never adding it
to `WITNESSES`, compiles this file and the suite passes, silently excluding
the new determinant from both the pairwise non-disclosure oracle and the
`test-hooks` reason-distinguishing oracle. (If a concurrent commit changes
`every_gang_refusal_reason` to DERIVE `WITNESSES` from the match itself —
rather than a hand-listed array the match merely re-validates — the
stronger "fails until added to both" claim holds again; stated here in the
weaker, currently-true form until that lands.) The pairwise non-disclosure
oracle that consumes `WITNESSES` drives the ten scenarios that are in
it.

### B2. `world_size` is a row fact, never a fault, at decode time

`crates/jammi-db/src/catalog/jobs_repo.rs::WorldSizeFact` is a two-armed
enum, `Decoded(u32)` or `Undecodable` — the malformed-value case the
addendum above (§A1) described as a typed `Err`
(`BackendError::TypeConversion`) is gone: `RankAdmissionRow::world_size` is
a `WorldSizeFact`, never wrapped in a `Result`, and
`Catalog::get_job_for_rank` returns `Ok(Some(row))` for a row whose `spec`
fails to decode a `world_size` — the row exists, every other column is
populated, and the content defect is the row's own fact, never a fault of
the read that found it.
`crates/jammi-server/src/grpc/gang.rs::GangServer::run_rank` refuses a
`WorldSizeFact::Undecodable` row under `GangRefusalReason::SpecUndecodable`,
counted against the attempt budget like every other refusal.
`crates/jammi-server/tests/it/gang_service.rs` asserts the poisoned-row and
the absent-row refusals are byte-identical `(code, message)` through the
real RPC, as part of the same pairwise non-disclosure oracle named above,
while the `test-hooks` seam names `SpecUndecodable` specifically for both
shapes.
`crates/jammi-db/tests/it/gang_rank_admission.rs::get_job_for_rank_malformed_world_size_is_undecodable_not_a_fault`
(parameterized sqlite/postgres, the postgres arm skipping rather than
failing when `JAMMI_TEST_PG_URL` is unset) and its not-valid-JSON-at-all
sibling pin this directly against `Catalog::submit_job`/`claim_next`/
`get_job_for_rank`, asserting `WorldSizeFact::Undecodable` with every other
column still populated.

### B3. Producer and consumer are coupled by an executed test, never prose

`crates/jammi-server/tests/it/gang_training_spec_parity.rs::get_job_for_rank_world_size_matches_the_real_training_spec_producer`
serializes a real `jammi_ai::fine_tune::spec::TrainingSpec::FineTune` naming
`TrainingCommon { world_size: 2, .. }`, submits it through
`Catalog::submit_job`, and asserts
`get_job_for_rank(..).world_size == WorldSizeFact::Decoded(2)` — the SAME
decode path the gang handler reads, never a hand-written JSON literal
standing in for either side. A second real spec
(`ContextPredictorTrainConfig`, itself asserted in the test to genuinely
omit the `world_size` key rather than merely default it) pins the
absent-field case to `WorldSizeFact::Decoded(DEFAULT_WORLD_SIZE)` — since
`jammi-db`'s own private `WORLD_SIZE_IF_ABSENT` const feeds the same decode
this test reads, this assertion pins `WORLD_SIZE_IF_ABSENT` equal to
`jammi_ai::fine_tune::spec::DEFAULT_WORLD_SIZE` without either crate
depending on the other's constant directly (`jammi-db` cannot depend on
`jammi-ai`).

### B4. Every admission-time catalog read on this path is `Unavailable`, never `map_engine_error`

`crates/jammi-server/tests/it/gang_admission_catalog_fault_oracle.rs` is a
source-scan oracle over `GangServer::run_rank`'s own function body — located
textually from the `fn run_rank` declaration to its matching closing brace
by a brace-depth walk, comments and string/char literals masked to spaces
first, never "the rest of the file". It asserts the span does not contain
`map_engine_error` as code, and does contain `admission_catalog_fault` at
least twice: confirmed at exactly two call sites in
`crates/jammi-server/src/grpc/gang.rs::GangServer::run_rank` —
`get_job_for_rank`'s `Err` arm and `fresh_instance`'s `Err` arm — each
mapping through `admission_catalog_fault` to `Status::unavailable(..)`,
distinct from every rung's `FailedPrecondition` refusal.

### B5. Stop rule fired; the world>1 conjunct is U5a-2's, filed

The pre-committed stop rule — a second BLOCK against the admission
lattice's determinants excises the world>1 conjunct — fired: the
training-set pair conjunct and its sidecar verify are excised from this
unit whole; `HostAdmission` (U5a-2) inherits them together with the
`world_size`-decode and catalog-fault-classification properties above,
which remained load-bearing at the reduced base. This unit ships the
`world_size == 1` lattice — every `world_size != 1` row, and every
`assign.world != row.world_size` call in either direction, refuses the same
fixed way — plus the wire surface, the migration, and the write-once
identity CAS as a db-layer primitive with no wire-path caller yet. Filed at
<https://github.com/f-inverse/jammi-ai/issues/566>.
