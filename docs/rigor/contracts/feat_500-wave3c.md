# Contract — `feat/500-wave3c`: the remainder of plan 67's wave 3, one consolidated PR

Base: `main` @ `b5a7aab9` (PR #580 merged: U5b-1a-A2 root identity, U5b-0 leaf inventory,
the merge-path runner). Slug `feat_500-wave3c`; pressure row in
`docs/rigor/feat_500-wave3c.jsonl`; oracle record in `docs/rigor/feat_500-wave3c.oracle.jsonl`
at the final tip, exported last, after every non-rigor edit.

## 1. Scope and build order

The plan's sizing doc pins two hard orders inside the remainder (U4b before U5b-1b-ii;
U5b-1b-ii before U5b-1b-iii and U5b-2); everything else is independent and was built
concurrently, one implementer per unit in its own worktree off `856ec8dd`, consolidated here
as one commit per unit:

| # | Unit | Built | Depends on |
|---|---|---|---|
| 1 | U4b — rank context; gather rule; lockstep; single-node `Local` gang; `[worker] local_ranks` | concurrently (wave A) | main |
| 2 | U5b-1b-i — `Peer` collective + two-phase round protocol; peer-listener decode cap | concurrently (wave A) | main |
| 3 | U5a-2 — `HostAdmission`, drain, re-verification, admit-and-hold; the world>1 admission conjunct (#566) | concurrently (wave A) | main |
| 4 | U5b-1b-ii (db) — `[distributed] max_world_size`; migration 037; assembly reason table, cooldown/counter; claim-side cooldown term; CAS call-site helper | concurrently (wave A) | main |
| 5 | U5b-1b-ii (body) — coordinator: membership → assignment → dispatch → assembly; `serveable_world` at submit | after 1–4 | 1, 2, 3, 4 |
| 6 | U5b-1b-iii — `world_size == 1` rank body; runner-role writer split; `Outcome`; resume pin | after 5 | 5 |
| 7 | U5b-2 — watchdog; abort with no terminal write; released-vs-failed; chaos | after 6 | 5, 6 |

Not in this PR: U7b-A2b live (RunPod cluster runs, with the user); the pod and cluster legs'
artifacts (label-gated, run after merge).

Each unit's section below is the implementer's contract file, folded by the lead after the
lead opened every cited line and re-ran every named oracle on the consolidated tree; every
deviation from `docs/plans/67-distributed-training/UNITS.md` is stated with its reason and the
plan rows carry a dated correction in the same commit.

## 2. U4b

(folded at consolidation)

## 3. U5b-1b-i

(folded at consolidation)

## 4. U5a-2 (landed as three commits on this branch; originals `8332a289`, `db82aef9`, `54bbe341`)

The implementer's contract, folded by the lead after opening the cited lines: the strict resolver takes an explicit tenant with the strict predicate and no admin arm; the holder flip is at the hold site and the prologue self-release test is intact; `dispatch_round_frame` is the one wiring site U5b-1b-i's round machinery joins; `in_flight`/`InFlightGuard` are gone. The pressure round's block 7 (no route to the job's tenant) is closed by the rebuilt tenant carrier; its advisory on the refusal-vs-abort observable split is closed by P1/P9.

Branch `unit/u5a2` in worktree `wt-u5a2`, cut from `856ec8dd`. Crates touched: `jammi-db`,
`jammi-ai`, `jammi-server` (+ one proto comment in `jammi-wire`, docs). Every claim below is
stated as it EXISTS on the branch tip; every path is repo-relative; test names are
`file::fn`.

### 1. Scope shipped

#### jammi-db (edits local to `get_job_for_rank`/`RankAdmissionRow` and the strict resolver)
- `crates/jammi-db/src/catalog/jobs_repo.rs::RankAdmissionRow` gains `tenant_id: Option<String>`
  (the row's `jobs.tenant_id` as RAW TEXT), `training_set_ref: Option<String>`,
  `training_set_location: Option<String>`; `Catalog::get_job_for_rank`'s one statement selects
  the three columns. **Deviation, stated:** the tenant is carried as text, never parsed in the row
  mapper — the U5a-1 round-2 excision deleted `tenant_id` precisely because its `parse::<TenantId>()`
  turned a found row into `Err` (indistinguishable from a driver fault, contradicting the method's
  own "infallible on content" contract, `docs/rigor/contracts/feat_500-C-U5a-1.md` Addendum 3).
  The handler parses it; an unparseable value is a row fact (`GangRefusalReason::TenantUndecodable`),
  exactly like an undecodable `world_size`. Pinned by
  `crates/jammi-db/tests/it/gang_rank_admission.rs::get_job_for_rank_carries_the_tenant_text_and_the_filled_pair`
  (sqlite + postgres), whose last arm plants `tenant_id = 'not-a-uuid'` by raw SQL and asserts
  `Ok(Some(row))` with the text verbatim.
- `crates/jammi-db/src/catalog/result_repo.rs::Catalog::get_result_table_for_tenant(name, tenant:
  Option<TenantId>)` — the strict predicate `tenant_id = $t OR (tenant_id IS NULL AND $t IS NULL)`,
  an explicit tenant argument, **no admin arm and no read of `current_tenant()`** (deviation from
  the excised verb, which kept the repo-wide admin-scope branch: a resolver whose tenant is an
  explicit argument must not be widened by ambient scope; the call site guard is kept on top of it).
  Tests: `crates/jammi-db/tests/it/result_tables.rs::get_result_table_for_tenant_never_matches_a_null_tenant_row_for_a_real_tenant`
  and `::get_result_table_for_tenant_resolves_only_the_owning_tenant`, both `test_case`-parameterized
  sqlite/postgres, each with a `with_admin_scope` arm proving ambient scope does not widen it.

#### jammi-ai (`crates/jammi-ai/src/fine_tune/worker.rs`, `crates/jammi-ai/src/session.rs`)
- `HostAdmission { phase: watch<WorkerPhase>, holder: watch<Holder>, registry: Arc<InstanceRegistration> }`
  owned by `InferenceSession` (`InferenceSession::host_admission()`); `instance_registration()`
  delegates to `registry`. `phase` moves out of `WorkerShared` (`WorkerShared::phase()` delegates;
  `set_phase_for_test` delegates). `HostAdmission::{begin_drain (Running→Draining, never regresses
  Releasing), begin_release, phase_receiver, holder, holder_receiver, probe_claim, job_running
  (pub(crate)), try_hold_rank, admit_rank, hold_for_test (test-hooks)}`.
- `Holder = Free | ClaimProbe | JobRun | Rank{job_id, attempt}`; `HolderBusy = ClaimProbe | JobRun |
  Rank{..}`; guards `ClaimGuard` (ClaimProbe/JobRun → Free on drop) and `RankHold` (→ Free on drop
  only if the cell still names this exact `(job_id, attempt)`).
- The claim loop (`JobWorker::run_until`): `probe_claim()` (Free→ClaimProbe) immediately before
  `claim_next`; a held slot skips the claim and sleeps the idle poll; `register_job_hold_or_release`
  flips ClaimProbe→JobRun once the lease hold is registered; the guard drops after
  `run_claimed_job_under` returns. **Deviation from README r27's sketch** ("`claim_next`'s `Some`
  arm flips ClaimProbe→JobRun"): the flip is at the HOLD SITE, so the claim→hold prologue stays a
  `ClaimProbe` and `release_and_stop`'s 2e still waits one heartbeat for the prologue's own
  self-release (zero net attempts, OPS D10) instead of aborting a claim whose lease would only fall
  to expiry. Pinned by `crates/jammi-ai/tests/it/host_admission.rs::the_claim_loop_moves_the_holder_free_probe_run_free`
  (the prologue park reads `ClaimProbe`) and the unchanged
  `crates/jammi-ai/tests/it/jobs_shutdown.rs::release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases`.
- `WorkerShared::in_flight`, `InFlightGuard` DELETED (cut, no rebuild owed: the holder kind is the
  fact they approximated). `EmbeddedWorker` holds `admission`; `begin_drain` → `admission.begin_drain()`;
  `release_and_stop` 2a → `begin_release()`, 2e reads the holder KIND (`JobRun` aborts now; anything
  else waits one heartbeat). `JobWorker` holds `admission` (its own `Arc`, keeps no session alive) so
  `run()`/`run_claimed_job` build `WorkerShared` over it; a direct `run_claimed_job` and an inline
  `run_now` never touch the holder. `InferenceSession::release_job_leases` flips `begin_release()`
  first. `/metrics` `jammi_worker_jobs_in_flight` = `holder == JobRun` (`crates/jammi-server/src/routes/health.rs`).

#### jammi-server (`crates/jammi-server/src/grpc/gang.rs`, `runtime.rs`, tests, docs)
- `GangServer::new(session, lease, heartbeat)`; `runtime.rs` passes `LeaseIntervals::heartbeat()`
  and calls `session.host_admission().begin_drain()` on both DRAIN arms (a worker-less server has no
  `EmbeddedWorker::begin_drain` to flip the phase).
- `run_rank` order: bounded first frame → wire K2 → ambient admin scope → `get_job_for_rank`
  (`admission_catalog_fault` on `Err`) → status/claimant/attempt/lease → `WorldSizeFact` →
  `assign.world != row.world_size` → **world>1 conjunct on `row.world_size > 1`**: pair filled
  (`TrainingSetPairMissing`), tenant text parses (`TenantUndecodable`), then
  `resolve_training_set_identity(store, tenant, ref, location) -> Result<TrainingSetOutcome, JammiError>`
  (`Err` = the strict resolver's catalog read faulting → `admission_catalog_fault`; outcomes
  `Verified | AdminScopeRefused | Unresolved | NotReady | SidecarAbsent | DigestMismatch | StoreFault`
  → `GangRefusalReason::{AdminScope, TrainingSetUnresolved, TrainingSetNotReady,
  TrainingSetSidecarAbsent, TrainingSetDigestMismatch, TrainingSetStoreFault}`) → `fresh_instance`
  (`admission_catalog_fault` on `Err`) → **only then** `HostAdmission::admit_rank(job_id,
  row.attempts, heartbeat)` (`Err(HolderBusy)` → one fixed `Unavailable("gang admission: this
  host's job slot is busy")`) → `Admitted` (`mpsc` channel, `ReceiverStream`) → `tokio::spawn(HeldSession::hold())`.
  `GangRefusalReason::MultiHostUnsupported` is DELETED; the enum names sixteen determinants.
- `HeldSession::hold`: FOUR `select!` arms — inbound (`on_control_frame`: `Cancel` →
  `Aborted{Cancelled}`; second `Assign` → `InvalidArgument` trailer; everything else →
  `dispatch_round_frame`, the ONE wiring site for the round protocol, today only an empty frame →
  trailer; the client's half-close disables the arm, the session stays held; a transport error
  ends silently), phase watch (`!= Running` or sender gone → `Aborted{Drain}`),
  `interval_at(now + heartbeat, heartbeat)` tick → `reverify` (`ReverifyEnd::{Refuted, Unavailable,
  StoreUnavailable}` with `abort_reason()`, `scope()`, `counts_toward_assembly_attempts()`), park
  `sleep(lease)` → `Aborted{NoBody}`. One event (or one trailer) then the stream closes and the
  `RankHold` drops.
- **Observables split at admission (pressure-round fold 2):** pre-admission = `Err(Status)` (the
  one fixed `FailedPrecondition` for every I-GANG determinant; `Unavailable` for catalog fault /
  busy slot; `InvalidArgument` for wire K2); post-admission = in-stream `Admitted` then one
  `Aborted{reason}`, or a status trailer for the admitted-stream K2 violation. R3 is not weakened:
  a reason is named only on a session the caller was admitted to on the job's own coordinates.
- Docs (same commit set): `docs/maintainer/MAINTAINER-GUIDE.md` §2.8a rewritten (the lattice incl.
  the world>1 conjunct, `HostAdmission`, the hold loop, observables split, config);
  `docs/guide/src/security.md` I-GANG (derivation, non-disclosure incl. the post-admission reasons,
  admit-and-hold); `configuration.md`, `deploy-server.md`, `api-stability.md`, `gang.proto`
  comment, `api_freeze_baseline.txt` comment, DESIGN.md/README.md allowlist sentences; seven
  pre-existing `PATH:LINE` citations re-anchored after the line shifts.
- Tests: `crates/jammi-server/tests/it/gang_service.rs` (rewritten around the restored world>1
  fixtures), `gang_terminal_write_oracle.rs` (new), `gang_rank_admission_oracle.rs` (+ the strict
  resolver's caller oracle), `gang_admission_catalog_fault_oracle.rs` (floor 2→3),
  `tenant_isolation_oracle.rs` (derivation claim + assertion), `gang_training_spec_parity.rs`
  (sqlite + postgres arms as two named fns over one body — this crate has no `test_case` dev-dep;
  no dependency added), `health.rs` (gauge read off the holder).

#### Cuts and their rebuilds
- `WorkerShared::in_flight`/`InFlightGuard`: cut; rebuilt as the holder kind (above).
- `GangRefusalReason::MultiHostUnsupported` and the `world_size != 1` refusal: cut; rebuilt as the
  world>1 conjunct (rows in §2).
- `run_rank_refuses_world_gt_one_when_caller_world_matches_the_row`: deleted with its determinant;
  its role (the direction-(a) control) is now `run_rank_refuses_when_assign_world_mismatches_row_world_size`'s
  `TrainingSetPairMissing` control.
- The excised admin-scope arm of the strict verb: not restored (deviation above); its property is
  the db tests' `with_admin_scope` arms.

### 2. Properties (quantified) — executed oracle — executed mutation that reds it

Lanes: SRV = `cargo test -p jammi-server --features test-hooks --test it -- <filter>`; SRV-plain =
the same without `--features`; AI = `cargo test -p jammi-ai --features test-hooks --test it --`;
DB = `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- --test-threads=1`
(sqlite arm always; postgres arm with `JAMMI_TEST_PG_URL`). Mutation ids M1–M13 are the executed
runs recorded in `u5a2-scratch/mutations.txt` (each: applied → one filtered test → reverted).

| # | Property (over every input / exit arm) | Executed oracle | Executed mutation → red (first line) |
|---|---|---|---|
| P1 | For every `RunRank` call, the admission decision (every I-GANG determinant incl. the world>1 conjunct and freshness) is complete BEFORE the holder is consulted; a refused call never touches the holder; `Admitted` is emitted only after a successful CAS (a busy slot is `Err(Status)`, never a stream). | SRV `gang_service::run_rank_refuses_unavailable_at_once_while_a_loop_job_runs` (busy slot → `Unavailable` + no reason recorded; an absent job under a busy slot → `FailedPrecondition`/`NotFound`) | M3: slot consulted before the row read → M3 (holder checked before the row read): `gang_service.rs:2192: assertion left == right failed` (an absent job answered `Unavailable`, not `FailedPrecondition`) |
| P2 | For every `row.world_size > 1` call and NEVER for `world_size == 1`: the pair must be filled, the row's tenant must parse, the strict resolver under the ROW's tenant must find a `ready` row, its sidecar must verify `artifact == training_set_ref`; every determinant refuses the one fixed `FailedPrecondition`; keyed on the row's decoded fact, never `assign.world`. | SRV `gang_service::run_rank_refuses_when_assign_world_mismatches_row_world_size` (both directions + controls: direction (a)'s control reaches `TrainingSetPairMissing`, direction (b)'s control is `Admitted`), `::run_rank_refuses_a_training_set_another_tenant_owns`, `::run_rank_refuses_a_null_tenant_training_set_for_a_tenant_bound_job`, `::run_rank_refuses_world_gt_one_when_the_sidecar_predates_the_leaf_inventory`, `::run_rank_world_two_own_tenant_training_set_is_admitted_held_and_parks_no_body` (the admitting control) | M1: world gate deleted → M1 (`if false && assign.world != world_size`): `gang_service.rs:1713: assertion left == right failed: must record WorldMismatch specifically, not the pair determinant`; M2: tenant forced `None` → M2 (`let tenant = None` before the resolver): `gang_service.rs:2019: the strict resolver must not resolve the NULL-tenant row (a later determinant refusing instead would mean it did)`; M7: leaf-less sidecar accepted → M7 (`Ok(None) => Verified`): `gang_service.rs:1583: every fixture this function builds must refuse` (the leaf-less sidecar admitted) |
| P3 | Non-disclosure: all sixteen determinants refuse with pairwise-identical `(code, message)` on the wire; the `test-hooks` seam distinguishes every one. | SRV-plain + SRV `gang_service::run_rank_refusal_is_non_disclosing_across_every_determinant`; SRV `::run_rank_last_refusal_reason_distinguishes_every_determinant`; `every_gang_refusal_reason` re-validated by an exhaustive match | (shape inherited from U5a-1, unchanged: any one arm interpolating a reason reds the pairwise oracle naming the pair; not re-executed this round — the sixteen-row scenario table is the executed novelty) |
| P4 | Tenant is derived from the job row, never accepted: no caller metadata is read; the strict resolver never resolves another tenant's or a NULL-tenant row for a real tenant, under ambient admin scope or not; the resolution site refuses ambient admin scope before the resolver runs. | SRV `gang_service::run_rank_never_reads_a_caller_supplied_tenant` (`jammi-session-id` metadata ignored → `Admitted`), `::resolution_site_refuses_under_admin_scope_before_the_strict_resolver_runs` (control `Verified` outside, `AdminScopeRefused` inside); DB `result_tables::get_result_table_for_tenant_never_matches_a_null_tenant_row_for_a_real_tenant::{sqlite,postgres}`, `::get_result_table_for_tenant_resolves_only_the_owning_tenant::{sqlite,postgres}`; the derivation claim in `tenant_isolation_oracle::gang_service_is_unimplemented_on_the_public_listener` | M11: strict predicate relaxed → M11 (predicate relaxed to `OR tenant_id IS NULL`): `result_tables.rs:550: the strict resolver must never match a NULL-tenant row for a real tenant` (sqlite arm); M2 (above) |
| P5 | The admission row carries the row's own tenant as text and the filled pair on both backends; the read is infallible on content (a garbage tenant is `Ok(Some)`). | DB `gang_rank_admission::get_job_for_rank_carries_the_tenant_text_and_the_filled_pair::{sqlite,postgres}` | M12: tenant parsed in the mapper → M12 (`parse::<TenantId>().expect(..)` in the mapper): `jobs_repo.rs:2087: panicked` (the garbage-tenant arm is no longer `Ok(Some)`) (sqlite arm) |
| P6 | Holder lattice (c2'): `Free` admits; `JobRun`/another `Rank` refuse at once; the same job at an equal attempt refuses, at a greater attempt supersedes in place and the elder's drop leaves the successor's hold; `ClaimProbe` is waited ≤ one bound then admits-if-freed or refuses; a hold's drop frees only its own cell. | AI `host_admission::{free_admits_a_rank_and_dropping_the_hold_frees_the_slot, a_job_run_or_another_rank_refuses_at_once, the_same_job_at_a_greater_attempt_takes_the_slot_and_the_elder_leaves_it, a_claim_probe_is_waited_out_then_admitted_if_freed_or_refused_at_the_bound}`; SRV `gang_service::{run_rank_refuses_unavailable_at_once_while_a_loop_job_runs, run_rank_waits_out_a_claim_probe_then_admits_if_freed_or_refuses_unavailable}` (test-hooks), `::a_held_rank_refuses_other_ranks_and_the_same_job_at_a_greater_attempt_takes_the_slot` (plain) | M9: `RankHold::drop` frees regardless of identity → M9 (`RankHold::drop` frees any `Rank`): `host_admission.rs:197: the superseded elder's drop must not free the successor's slot` |
| P7 | Exclusion (d2', OPS D6): the loop moves the holder `Free→ClaimProbe→JobRun→Free` around every claim (the flip at the hold site, the prologue a probe); an idle loop never calls `claim_next` while a rank is held and claims the moment it is freed; an inline `run_now` is outside the exclusion; a `JobRun`-holding peer refuses a rank, an idle peer admits. | AI `host_admission::{the_claim_loop_moves_the_holder_free_probe_run_free, an_idle_loop_never_claims_while_a_rank_is_held, an_inline_run_now_never_touches_the_holder}`; the gauge `health::gauges::in_flight_gauge_is_one_during_a_loop_claimed_job_and_zero_during_run_now`; the reshaped `jobs_shutdown` suite (20 rows) | M8: flip at `claim_next`'s `Some` arm → M8 (`job_running()` at `claim_next`'s `Some` arm): `host_admission.rs:306: the claim committed but the hold is not registered: still a probe` (read `JobRun`); M10: probe ignores a held slot → M10 (`probe_claim` overwrites a held slot): `host_admission.rs:365: claim_next must not be called while a rank is held` |
| P8 | OPS D10: RELEASE's abort decision reads the holder kind, never a count — a `Rank` beside an idle loop is never loop work (cooperative `Stopped`, hold untouched, phase `Releasing`); `JobRun` aborts now. | AI `host_admission::release_and_stop_beside_a_held_rank_exits_cooperatively_and_flips_the_phase`; `jobs_shutdown::release_with_the_loop_paused_in_the_claim_to_hold_prologue_self_releases` (the prologue still self-releases) | M13: a held `Rank` treated like `JobRun` → M13 (`holder == Holder::Free` → a held `Rank` aborts now): first executed against the ORIGINAL oracle (an idle loop) it stayed GREEN — the loop had exited at 2a before 2e ran, so the arms were indistinguishable; the oracle was rewritten to park the loop after reclaim, re-executed green, then the same mutation re-executed: `host_admission.rs:457: a held Rank is not loop work: 2e waits for the cooperative exit, never aborts` (`Aborted` ≠ `Stopped`) |
| P9 | The hold loop has exactly four arms and every end is one stream event: `Cancel` → `Cancelled`; a second `Assign` → `InvalidArgument` TRAILER (K2); phase leaving `Running` → `Drain` at once (both via the session cell and via the real server shutdown path, worker-less); park bound → `NoBody` after ≥ two ticks. | SRV `gang_service::{run_rank_cancel_on_an_admitted_stream_ends_cancelled, run_rank_second_assign_on_an_admitted_stream_is_invalid_argument, run_rank_held_session_ends_drain_when_the_host_drains, run_rank_held_session_ends_drain_on_server_shutdown, run_rank_every_i_gang_determinant_satisfied_is_admitted_held_and_parks_no_body}` | M4: drain arm never fires → M4 (phase arm replaced by `pending()`): `gang_service.rs:570: expected Aborted{Drain}, got Aborted { reason: NoBody }` (parks to `NoBody` instead) |
| P10 | Re-verification (i2'): the three ends are pairwise distinct on the wire, in scope and in the count rule; a row fact moving → `Refuted`; the catalog faulting → `Unavailable`; THIS host's store faulting (`Storage`/`Io`) → `StoreUnavailable`; the artifact's sidecar no longer verifying → `Refuted` (never `StoreUnavailable`). | SRV `gang_service::{run_rank_held_session_ends_refuted_when_the_row_no_longer_holds, run_rank_held_session_ends_unavailable_when_the_catalog_faults, run_rank_held_session_ends_store_unavailable_when_this_hosts_store_faults, run_rank_held_session_ends_refuted_when_the_sidecar_stops_verifying}`; lib `grpc::gang::tests::reverify_ends_are_pairwise_distinguishable_in_reason_scope_and_count` | M5: store fault classified `Refuted` → M5 (`StoreFault → ReverifyEnd::Refuted`): `gang_service.rs:570: expected Aborted{StoreUnavailable}, got Aborted { reason: Refuted }` |
| P11 | Terminal-write scope (g2'): the peer names no `jobs` writer (the set derived from `jobs_repo.rs` itself) and every end (cancel, K2 trailer, drain, refuted, unavailable, store-unavailable, park, supersession) leaves the job row byte-identical to its pre-admission snapshot. | SRV `gang_terminal_write_oracle::the_gang_handler_names_no_jobs_writer` (+ its two self-tests); the `row_facts` before/after equality in every hold-loop row above | M6: `fail_job(` named as code in `run_rank` → M6 (`let _ = stringify!(fail_job());` in `run_rank`): `gang_terminal_write_oracle.rs:283: gang.rs names the jobs writer fail_job( as code` |
| P12 | Every admission-time catalog read maps `Err` through `admission_catalog_fault` (three sites), never `map_engine_error`; `get_job_for_rank`'s and `get_result_table_for_tenant`'s only production callers are the gang handler. | SRV `gang_admission_catalog_fault_oracle::run_rank_never_calls_map_engine_error` (floor 3), `gang_rank_admission_oracle::{only_the_gang_run_rank_handler_calls_get_job_for_rank, only_the_gang_resolution_site_calls_get_result_table_for_tenant}` | (allowlist-both-directions shape: a fourth caller file reds the oracle naming it — the U5a-1 methodology; the raised floor was exercised by the `2 → 3` self-test fixture change) |
| P13 | Producer→consumer `world_size` parity holds on both backends. | SRV `gang_training_spec_parity::get_job_for_rank_world_size_matches_the_real_training_spec_producer_{sqlite,postgres}` (postgres executed with `JAMMI_TEST_PG_URL`) | (decode mutations inherited from U5a-1's db tests) |

### 3. Uncovered

- **UNCOVERED — a `StoreUnavailable` from a genuine I/O fault on a local file.** The executed
  member-scoped store fault is `JammiError::Storage(SchemeNotEnabled)` from `open_parquet` on an
  `s3://` URL this build compiles no driver for (hermetic, network-free). A permission/I-O error
  on a `file://` sidecar is the same `JammiError::Storage`/`Io` arm by construction
  (`ManifestError::Storage → JammiError::Storage`, `object_store` errors → `StorageError`) but is
  not executed: a `chmod`-based fixture silently passes under a root CI lane (memory note
  `jammi-ci-root-permission-fault-tests`) and no fault-injection seam exists in the store; adding
  one was out of scope.
- **UNCOVERED — the `FIRST_ASSIGN_BOUND` timeout arm** (a silent client for 10 s): inherited from
  U5a-1, still not timed in the suite.
- **UNCOVERED — `AbortReason::Drain` on the sender-dropped branch** (the `InferenceSession` dropping
  while a rank is held): the session outlives every server in the fixtures; the arm is the same
  `wait_for` `Err` path as a phase flip and is stated, not executed.
- **UNCOVERED — the round-frame dispatch point receiving a real round frame**: no such frame exists
  in `RankControl`'s oneof at this head; only the empty-frame protocol violation reaches
  `dispatch_round_frame` and is not separately executed (a `control: None` frame cannot be built
  by the generated client without a raw codec; the second-`Assign` row executes the same
  end-with-trailer path).
- **Non-disclosure over the sixteen determinants on the plain lane** is executed; the
  "a leaking arm reds the pairwise oracle" mutation is inherited from U5a-1 and not re-executed.
- The world>1 admission row from a REAL coordinator's `fill_training_set_identity` call on the wire
  path (U5b-1b-ii's materialization step) does not exist yet; every world>1 fixture fills the pair
  through the db CAS directly.

### 4. Gates (trimmed set per the lead; exit codes and counts)

All with `CARGO_TARGET_DIR=…/targets/u5a2`; one `--features` set per crate for the whole session
(`jammi-db`: `live-postgres-tests,test-hooks`; `jammi-ai`/`jammi-server`: `test-hooks`). The final
tree (tip below) is exactly the tree these ran on: every mutation was reverted by `git checkout`
and `git status` is clean; the only edit after the server runs was the ai TEST rewrite of the
RELEASE-beside-a-rank row, re-run green with its mutation red, then `cargo fmt --check` and
`cargo clippy -p jammi-ai` re-run green.

| Command | exit | result |
|---|---|---|
| `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:54329/jammi_test cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- --test-threads=1 get_result_table_for_tenant get_job_for_rank_carries_the_tenant_text` | 0 | 6 passed (3 tests × sqlite + postgres); 0 failed |
| `cargo test -p jammi-ai --features test-hooks --test it -- jobs_shutdown host_admission` | 0 | 28 passed (20 `jobs_shutdown` + 8 `host_admission`); 0 failed |
| `cargo test -p jammi-ai --features test-hooks --test it -- release_and_stop_beside_a_held_rank` (after the oracle rewrite) | 0 | 1 passed |
| `cargo test -p jammi-server --features test-hooks --test it -- --test-threads=4 gang tenant_isolation_oracle::gang_service_is_unimplemented` | 0 | 47 passed (gang_service 37, gang_rank_admission_oracle 4, gang_admission_catalog_fault_oracle 3, gang_terminal_write_oracle 3, gang_training_spec_parity 2 — postgres arm skipped here —, tenant_isolation_oracle 1); 0 failed |
| `JAMMI_TEST_PG_URL=… cargo test -p jammi-server --features test-hooks --test it -- --test-threads=1 in_flight_gauge gang_training_spec_parity` | 0 | 3 passed (the gauge row; parity sqlite + postgres, postgres EXECUTED) |
| `cargo test -p jammi-server --features test-hooks --lib -- gang` | 0 | 1 passed (`reverify_ends_are_pairwise_distinguishable_in_reason_scope_and_count`) |
| `cargo test -p jammi-server --test it -- gang tenant_isolation_oracle::gang_service_is_unimplemented` (PLAIN lane, the count) | 0 | 44 passed; 0 failed — the `test-hooks` lane runs 3 more (`run_rank_last_refusal_reason_distinguishes_every_determinant`, `run_rank_refuses_unavailable_at_once_while_a_loop_job_runs`, `run_rank_waits_out_a_claim_probe_then_admits_if_freed_or_refuses_unavailable`), invisible to the plain lane by `#[cfg]` |
| `cargo clippy -p jammi-server --all-targets --features test-hooks -- -D warnings` | 0 | clean |
| `cargo clippy -p jammi-ai --all-targets --features test-hooks -- -D warnings` | 0 | clean (a pre-existing 4-space doc continuation in the 2e bullet tripped `doc_overindented_list_items` once I restructured that bullet; re-indented) |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | clean |
| `cargo fmt --all -- --check` | 0 | clean |
| `python3 ci/scripts/perf/check_citations.py` (not in the trimmed set; run because my insertions shifted lines) | 0 | `1030 file(s) scanned, all PATH:LINE citations resolve` — after re-anchoring seven citations (six in `MAINTAINER-GUIDE.md`, one in `pinned_source_gate.rs`) that my worker.rs/runtime.rs/session.rs insertions had moved |
| `python3 ci/scripts/check_no_consumer_names.py` | 0 | OK |

Executed mutations: M1–M12 red on the first run; M13 green on the first run (vacuous oracle — an
idle loop exits at 2a before 2e; recorded honestly), the oracle rewritten, then M13 red — see §2.
`u5a2-scratch/mutations.txt` and `mut-M*.log` hold every run's output.

### 5. Commits (`git log --oneline 856ec8dd..HEAD`)

```
54bbe341 feat(wire-server): #500 U5a-2 — admit-and-hold: the world>1 conjunct (#566), the holder CAS, the four-arm hold loop, re-verification's three ends
db82aef9 feat(ai-core): #500 U5a-2 — HostAdmission: session-owned phase, the holder CAS lattice, the claim loop's probe
8332a289 feat(db): #500 U5a-2 — admission row carries the job's tenant text and training-set pair; strict tenant-pinned result-table resolver
```
(no trailers, per the brief; 30 files, +4331/−785 against `856ec8dd`; `git status` clean.)


## 5. U5b-1b-ii — database slice (landed as one commit on this branch; original `6dab678b`)

The implementer's contract, folded by the lead after opening the cited lines (the cooldown term is `lease_expired_clause("next_assembly_after", ..)` verbatim; `lease.rs` carries a zero diff; `AssemblyOutcome::effect` has no wildcard arm; the pressure round's block 8 is closed by this shape).

### 1. Scope shipped

Files touched (all inside `crates/jammi-db` and its docs; `[worker]` untouched, per the brief):

- `crates/jammi-db/src/config/mod.rs` — new `DistributedConfig { max_world_size: u32 }`
  section (default 1, `deny_unknown_fields`), `JammiConfig.distributed` field, `Default`
  wiring, and a `config.distributed.validate()?` call in `load_from` (refuses `0`).
- `crates/jammi-db/src/config/tests.rs` — 4 tests for the new section (default, TOML
  round-trip, zero-refusal at `validate()` and at `load_from`, independence from
  `[worker]`).
- `crates/jammi-db/src/catalog/schema.rs` — `MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER`:
  `jobs.assembly_failures INTEGER NOT NULL DEFAULT 0`, `jobs.next_assembly_after` (nullable
  `TEXT`, same representation `lease_expires_at` uses).
- `crates/jammi-db/src/catalog/migrations.rs` — the `("037_jobs_assembly_failures_next_after",
  schema::MIGRATION_037_JOBS_ASSEMBLY_FAILURES_NEXT_AFTER)` tuple appended.
- `crates/jammi-db/src/catalog/jobs_repo.rs`:
  - `AssemblyOutcome` (9 variants: `Refuted`, `AllRootDivergent`, `Unavailable`,
    `StoreUnavailable`, `ShortListed`, `NoBody`, `Drain`, `Cancelled`, `Success`) and a
    private `AssemblyEffect` (`Neither | CooldownOnly | CooldownAndCounted | Reset`) — a
    proper 4-way enum rather than a `(bool, bool)` pair, so "counted but not cooled" (a
    combination the design never calls for) is unrepresentable. `AssemblyOutcome::effect`
    is an exhaustive match (no wildcard): a new variant with no rule is a compile error.
  - `assembly_backoff(failures: u32) -> Duration` — bounded exponential (`2s * 2^k`,
    `k` clamped to 8 before the shift to avoid overflow regardless of `failures`, final
    ceiling 300s / 5 minutes, both named constants).
  - `Catalog::record_assembly_outcome(job_id, attempt, outcome) -> Result<bool>` — a
    locking read of the current `assembly_failures` (guarded by `job_id`+`attempts`, `FOR
    UPDATE` on Postgres; SQLite's `BEGIN IMMEDIATE` already serializes), then ONE `UPDATE`
    that writes both columns per the rule, using `super::lease::lease_deadline_expr` (the
    SAME helper `claim_next`'s own lease stamp uses) for the cooldown deadline. Returns
    `false` with no write when `attempt` no longer names the row's current attempt.
  - `claim_next`'s candidate subselect gains `AND {cooldown_clause}`, where
    `cooldown_clause = lease_expired_clause("next_assembly_after", kind, &mut params)` —
    **reused verbatim**, not a new function (see the pressure-round correction below).
  - `TrainingSetAssembly { Won, Reused, Moved }` and
    `Catalog::materialize_or_reuse_training_set(job_id, claimed_by, attempts, ref,
    location) -> Result<TrainingSetAssembly>` — a thin, DESIGN.md-vocabulary wrapper of
    `fill_training_set_identity`, mapping `Filled → Won`, `Reused → Reused`, `Aborted →
    Moved`.
- `crates/jammi-db/tests/it/migrations.rs` — `037_...` appended to `EXPECTED_MIGRATION_NAMES`
  and the 029-replay `DELETE FROM applied_migrations ... IN (...)` list (037 ALTERs `jobs`);
  a new column-presence assertion after the replayed reopen (the fifth K5 pin site's teeth);
  a new `migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after` oracle
  (sqlite+postgres) mirroring 034/035/036's own shape, plus a default-value assertion
  (`assembly_failures = 0`, `next_assembly_after = NULL` on a row naming neither column).
- `crates/jammi-db/tests/it/assembly_outcome.rs` (new) — every behavioural oracle below.
- `crates/jammi-db/tests/it/main.rs` — `mod assembly_outcome;`.
- `docs/guide/src/configuration.md`, `docs/maintainer/MAINTAINER-GUIDE.md` — `[distributed]
  max_world_size` documented (the guide fence is inside the ONE big TOML fence
  `docs_config_fences.rs`'s real-loader oracle already walks, so it is exercised, not just
  asserted in prose).

#### Deviations from the brief, with the cited reason

1. **No `cooldown_ready_clause` function.** The brief (`U5b-1b-ii-db.md`) asked for a new
   `(next_assembly_after IS NULL OR next_assembly_after <= <backend now>)` clause. Mid-task
   the lead relayed a pressure-round finding, verified against `crates/jammi-db/src/catalog/lease.rs:13-26,
   47-56, 60-73, 147-158`: SQLite's arm deliberately binds the APPLICATION clock (never a
   SQL clock function, which truncates precision) while Postgres's reads the DATABASE's
   `now()`, and the two backends store different representations. The binding correction:
   reuse `lease_expired_clause` VERBATIM (its `<` predicate, not a new `<=` one) rather than
   invent a second, near-duplicate clock helper. I cut the `cooldown_ready_clause` function
   and its lease.rs tests entirely and wired `claim_next`'s cooldown term straight to
   `lease_expired_clause("next_assembly_after", kind, &mut params)` — `lease.rs` ends the
   unit with a NET-ZERO diff (`git diff --stat crates/jammi-db/src/catalog/lease.rs` is
   empty), so it is not part of this unit's shipped diff at all.
2. **Acceptance (b) restated, per the same relay:** "no second clock source: the cooldown
   predicate is rendered by the lease module's helpers" (a source-scan oracle), plus the
   process-clock-skew oracle scoped to the Postgres arm only. Both built exactly as
   directed — see properties 5 and 6 below.
3. **`record_assembly_outcome` is a locking SELECT + one UPDATE, not a single bare UPDATE.**
   The brief's prose says "one UPDATE applying that rule"; I read that as "one coherent,
   atomically-guarded write" rather than "zero reads permitted" — embedding the exponential-
   backoff arithmetic (`2^k` clamped, ceiling, per-dialect `LEAST`/`MIN`) directly in raw SQL
   for two dialects risked exactly the overflow/precision class of bug `assembly_backoff`'s
   own doc comment warns about, for no benefit: the read is a locked, guarded, same-
   transaction SELECT (`FOR UPDATE` on Postgres; SQLite's `BEGIN IMMEDIATE` already
   serializes per `catalog::migrations`' own module doc and the concurrent-migrate tests),
   and the SUBSEQUENT write is exactly one `UPDATE` statement per call, guarded by the same
   `job_id`/`attempts` CAS `fill_training_set_identity` uses. `assembly_backoff` itself is a
   pure, independently testable Rust function (see property 4's escalation oracle), not
   duplicated per-dialect SQL.
4. **`materialize_or_reuse_training_set` maps `Aborted → Moved`, not a bespoke third state.**
   `fill_training_set_identity` (existing, `crates/jammi-db/src/catalog/jobs_repo.rs`) already
   distinguishes "won"/"reused"/"aborted, no write" exhaustively; the brief's ask ("returns
   whether it WON or REUSED; a moved claim ... aborts with NO write, a distinct typed
   outcome") is satisfied by a thin rename into the coordinator's own vocabulary
   (`TrainingSetAssembly`), not a new state machine.

### 2. Properties

| Property (quantified) | Executed oracle (test path::name, lane/features) | Executed mutation that reds it (change, red output first line) |
|---|---|---|
| (a) A higher-priority job inside its cooldown never blocks a lower-priority ready job, on either backend. | `assembly_outcome::cooldown_job_never_blocks_a_lower_priority_ready_job::{sqlite,postgres}` (`--features live-postgres-tests,test-hooks`) | Removed `AND {cooldown_clause}` from both `candidate` branches in `claim_next` (jobs_repo.rs). Red (both dialects, verified separately with/without `JAMMI_TEST_PG_URL`): `assertion \`left == right\` failed: the higher-priority job must be skipped while cooling down ... left: "cooling-high" right: "ready-low"`. |
| (b1) No second clock source: every cooldown-bearing SQL fragment in `jobs_repo.rs` is rendered through `catalog::lease`'s helpers, never a hand-written `CURRENT_TIMESTAMP`/`datetime('now'`/`chrono::Utc::now()`. | `assembly_outcome::cooldown_sql_has_no_second_clock_source` (hermetic, no backend) | Added `const MUTATION_PROBE_TOKEN: &str = "CURRENT_TIMESTAMP";` to jobs_repo.rs. Red: `jobs_repo.rs must never hand-write a clock-bearing SQL fragment ... found the literal "CURRENT_TIMESTAMP"`. |
| (b2, Postgres-only) The cooldown predicate is governed ENTIRELY by the Postgres SERVER's own clock — a value set via `now() ± interval` and compared via the same `now()`, never a value this test computes from its own process clock. | `assembly_outcome::postgres_cooldown_is_governed_by_the_server_clock_alone` (`--features live-postgres-tests`, requires `JAMMI_TEST_PG_URL`) | Same mutation as (a) (cooldown clause removed). Red: `the hour-ahead cooldown must still hold the second row back`. |
| (c1) The assembly outcome table is exhaustive and total: every one of the 8 non-success variants gets exactly its documented (counted?, cooled?) pair, on both backends. | `assembly_outcome::assembly_outcome_table_is_exhaustive_and_each_rule_fires_once::{sqlite,postgres}` | Reassigned `AllRootDivergent` from `CooldownAndCounted` to `CooldownOnly` in `AssemblyOutcome::effect`. Red: `assertion \`left == right\` failed: all-root-divergent: counted=true left: 0 right: 1`. |
| (c2) Consecutive COUNTED failures escalate the backoff (bounded exponential on the counter), on both backends. | `assembly_outcome::consecutive_counted_failures_escalate_the_backoff::{sqlite,postgres}` | Pinned `assembly_backoff`'s exponent to `0` regardless of `failures` (constant backoff). Red: `the second counted failure's backoff (1.999...s remaining) must exceed the first's (1.999...s remaining)`. |
| (c3) A success resets BOTH the counter and the cooldown, even after a counted failure armed both. | `assembly_outcome::success_resets_the_counter_and_the_cooldown::{sqlite,postgres}` | Remapped `Success` to `AssemblyEffect::Neither` (no reset). Red: `assertion \`left == right\` failed: success must reset the counter to 0 left: 1 right: 0`. |
| (d) `record_assembly_outcome`: a moved claim (stale `attempts`) writes NOTHING, on either backend. | `assembly_outcome::record_assembly_outcome_moved_claim_aborts_without_a_write::{sqlite,postgres}` | Dropped the `attempts = $2` guard from the internal locking SELECT (job_id-only). Red: `a stale attempt number must write nothing at all`. |
| Migration 037 ordered after 036; adds `jobs.assembly_failures` (NOT NULL) / `jobs.next_assembly_after` (nullable TEXT), on both backends. | `migrations::migration_037_is_ordered_after_036_and_adds_assembly_failures_next_after::{sqlite,postgres}` | RED at base: the migration, the const, and the test are all new in this diff — the test cannot even compile against the pre-change tree (`schema::MIGRATION_037_...` does not exist). |
| The fifth K5 pin site: 037 is on the 029-replay's ledger-clearing `IN (...)` list, so a manufactured pre-029 reopen replays it and the recreated `jobs` table carries both columns. | `migrations::migration_029_copies_training_jobs_rows_into_jobs_as_queued` | Dropped `'037_jobs_assembly_failures_next_after'` from the `DELETE FROM applied_migrations ... IN (...)` list. Red: `reopened jobs must carry 'assembly_failures' (migration 037 replayed): []`. |
| (f1) `materialize_or_reuse_training_set`: first call WINS; a repeat under the same claim, same values, REUSEs; a concurrent racer REUSEs (never a second write); a moved claim aborts with NO write. | `assembly_outcome::materialize_or_reuse_training_set_{first_call_wins,second_call_same_values_reuses,concurrent_racer_reuses_never_overwrites,moved_claim_aborts_without_a_write}::{sqlite,postgres}` | Remapped `TrainingSetFillOutcome::Aborted → TrainingSetAssembly::Reused` (instead of `Moved`). Red: `assertion \`left == right\` failed left: Reused right: Moved`. |
| `[distributed] max_world_size` defaults to 1, round-trips through TOML, refuses `0` at both `validate()` and `load_from`, and loads independently of `[worker]`. | `config::tests::distributed_config_{default_is_single_rank,toml_round_trips_max_world_size,zero_max_world_size_is_refused_at_load,loads_independently_of_worker}` (`--features test-hooks`, `--lib`) | Disabled the `max_world_size == 0` branch (`if false && ...`) in `DistributedConfig::validate`. Red: `called \`Result::unwrap_err()\` on an \`Ok\` value: ()`. |
| The `[distributed]` guide fence parses under the real loader (doc-parity). | `docs_config_fences::docs_toml_fences_parse_under_the_real_loader` (pre-existing oracle, exercised by the new fence content) | Not separately mutated — this is the SAME generic oracle every guide fence already rides; a broken `[distributed]` fence would fail it exactly as any other section's would (verified green with the addition in place). |

### 3. Uncovered

- **The body's own call sites** (`jammi-ai`'s `fine_tune/worker.rs`/`spec.rs`, `jammi-server`'s
  `grpc/gang.rs`) that will actually call `record_assembly_outcome` and
  `materialize_or_reuse_training_set`, and the `serveable_world`/`[distributed]
  max_world_size` submit-time cross-check — explicitly out of scope for this unit (U4b,
  U5b-1b-i, U5a-2 build them); labelled DEFERRED per the brief, not a gap in this unit's own
  verb-level oracles.
- **A true OS-clock-skew injection** (actually setting the test process's wall clock away
  from the Postgres server's) is not attempted — no clock-mocking harness exists in this
  crate, and the lead's relayed correction explicitly reframes acceptance (b) as the
  structural "no second clock source" source-scan plus the Postgres-server-clock-authority
  behavioural test, which together are the achievable, hermetic proxy for the same property.
  Labelled, not silently assumed.
- **`assembly_backoff`'s behaviour beyond the clamp exponent (8)** — failures counts above
  8 all saturate at the same 300s ceiling; not separately probed at e.g. `failures = 1000`
  for overflow-freedom beyond the `checked_shl`/`expect` invariant already encoded (the
  `expect` message states why it cannot panic: the exponent is clamped to `< 32` before the
  shift). No test drives `failures` near `u32::MAX`; the `unwrap_or(u32::MAX)` fallback in
  `record_assembly_outcome`'s `u32::try_from(new_failures)` (i32 → u32) is likewise
  unexercised at that boundary (`new_failures` is bounded by `i32::MAX` well before it, via
  `saturating_add`, so the fallback arm is dead in practice but retained as a defensive
  total function).

### 4. Gates

| Command | Exit | Notes |
|---|---|---|
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p jammi-db --all-targets --features live-postgres-tests,test-hooks -- -D warnings` | 0 | |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --test it -- assembly_outcome migrations docs_config_fences --test-threads=1` | 0 | 53 passed (20 `assembly_outcome`, 30 `migrations`, 3 `docs_config_fences`), 0 failed, `JAMMI_TEST_PG_URL=postgres://jammi@127.0.0.1:54329/jammi_test` set (live scratch Postgres 16, reachable on 54329) |
| `cargo test -p jammi-db --features live-postgres-tests,test-hooks --lib config::tests::distributed -- --test-threads=1` | 0 | 4 passed |
| Every mutation in §2 | red confirmed, then reverted | `git status --short` clean of mutation artifacts after each revert; `git diff --stat crates/jammi-db/src/catalog/lease.rs` empty (net zero touch, per the pressure-round correction) |

One test binary reused for the whole session (`CARGO_TARGET_DIR=<scratch>/targets/u5b1bii-db`,
one `--features live-postgres-tests,test-hooks` set throughout).

### 5. Commits

```
6dab678b feat(db): #500 U5b-1b-ii (db) — assembly cooldown/counter, [distributed] max_world_size, the CAS call-site helper
```
(one commit on `unit/u5b1bii-db`, off tip `856ec8dd`)


## 6. U5b-1b-ii — coordinator body

(built after wave A lands)

## 7. U5b-1b-iii

(built after §6)

## 8. U5b-2

(built after §7)

## 9. Pressure round (phase 1, executed at `856ec8dd` before the code landed) — REFINE, eight blocks folded

The design of the four wave-A units was attacked read-only against the base tree; the lead
opened every cited line before relaying. Each block below names the decision that folds it,
sent to the implementer while its unit was still being built; the commit closing each is
recorded at consolidation.

| # | Finding (cited) | Decision folded |
|---|---|---|
| 1 | U4b (b) "bit-for-bit summed adapter gradient vs W=1" is unachievable under f32 reassociation (`batch_bucket.rs:304-315` already documents 1e-5 for the padding half) and is satisfiable only by every rank computing the whole global gradient and skipping the reduce | (b) restated as a pre-registered ε whose discriminator is the W× hazard; bit-identity kept only for W=2-twice and resume-vs-uninterrupted; the "trainable op after the gather" mutation stays the hazard oracle |
| 2 | Zero ≠ absent for AdamW (`adamw.rs:245` skips an absent Var, steps a zero-gradient one; `optimizer.rs:626-633` documents absence as designed) | the reduce reduces the presence set (union across ranks) and restores "absent on every rank ⇒ absent" before `clip_and_step`; never a `world > 1` guard |
| 3 | Both epoch loops end on THIS rank's first empty chunk (`trainer.rs:1400-1404`, `1453-1456`); the optimizer-step boundary counts only non-diverged batches (`2881-2893`, `2903`) | epoch end = the global step count from the partition rule; a zero-row rank takes the step with a synthesized `[0, hidden]` tensor; the divergence decision is flag-reduced BEFORE accumulation so all ranks skip together (at W=1 `Noop` returns the rank's own flags — the W=1 window is unchanged); the one-rank divergence oracle uses the `cfg(test)` poke seam and says so |
| 4 | Hard-negative mining (`1303-1315`), GradCache (`1344-1362`) and the Precomputed arm (`1317-1343`) have no gather story; `gradcache.rs`/`hard_negative_miner.rs` in no scope | typed refusal at the spec admission edge for `world_size > 1` with mining, GradCache or a precomputed loader; their gather is a named follow-up; (b)/(d) oracles use the real loaders |
| 5 | The lifted `Descriptor` drops the round generation `Local` keeps outside it (`local.rs:174-193`); a commit-phase fault leaves one rank un-applied and the next fold is silently wrong | the round index is bound into the agreed descriptor; any commit-phase fault is fatal on every rank |
| 6 | `tests/distributed` already exists behind `live-distributed-tests` with a Postgres+MinIO harness and never gates a PR (`Cargo.toml:335-338`, `distributed.yml:24-26`) — (a),(c),(e),(f) would report green without executing | the Peer-vs-Local fold, deadline, commit-phase fault, descriptor disagreement, corrupted leaf and decode cap move to a hermetic two-process target in the ordinary test lane; fleet-dependent rows join the matrix by name |
| 7 | `RankAdmissionRow` carries no tenant (`jobs_repo.rs:212-250`, deleted in #566 round 2); `get_job` is tenant-filtered; admin scope forbidden — no route to the job's tenant for R2(b) | the admission row carries the row's OWN `jobs.tenant_id` (rebuilt carrier, its own oracle on both backends); the strict resolver pins to it |
| 8 | "Backend SQL clock on both dialects" contradicts the lease discipline: SQLite binds the application clock by design (`lease.rs:13-26`), and the two backends store different representations (`47-56`) | the cooldown reuses `lease_expired_clause`/`lease_deadline_expr`/`lease_now`/`lease_deadline` — no second clock source; the process-skew oracle is Postgres-only; a source-scan oracle pins "no literal clock outside the lease helpers" |

Advisories folded in the same messages: no rung exchange and no pin in U4b (the gather is dim 0
over pooled `[rows, hidden]`; `batch_bucket.rs:96-104`'s rationale rewritten; (c)'s ε covers
padding variance); the `agreement` slot is bound by U4b's canonical key-name digest; the
`BlockingCall` witness is a per-call trait argument owned by U5b-1b-i, threaded through U4b's
single collective helper at consolidation; an older resume bundle reads as no checkpoint (warn),
and W=1's byte-identity property is the artifact, not the bundle; the decode cap is per service
(both services on the peer listener, and the client recv side), pinned at exactly
`max_message_bytes`; classification logits are gathered by moving `classify()` into batch
construction (`TrainingBatch::Classification` reshaped, `data.rs` in scope); the
`[worker] world_size` → `local_ranks` rename touches the config field and accessors only and
the guide block is an ADD; stale premises corrected (U5a-1 froze no round messages;
`get_result_table_for_tenant` is deleted); the run_rank seam is pinned — U5a-2 owns admission,
the holder CAS and the hold loop; U5b-1b-i's round machinery lives in its own module and is wired
into the inbound arm at consolidation.

## 10. Gates (the merge path)

`bash ci/scripts/merge_path.sh` on the consolidated tip with `JAMMI_TEST_PG_URL` pointing at a
local PostgreSQL 16 in the CI lane's shape — run ONCE by the lead, never per implementer; the
phase-5 oracle dispatched after every other stage is green; only `docs/rigor/**` committed
after it.
