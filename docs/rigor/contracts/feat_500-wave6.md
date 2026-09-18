# feat/500-wave6 — waves 6 and 7 of plans 67/68: the residue units, driven to close

Branch `feat/500-wave6`, base `b8052978` (main after PR #587). One consolidated PR; each unit lands as one
squashed commit whose tree is byte-identical to the unit branch's audited tip. Rigor record:
`docs/rigor/feat_500-wave6.jsonl` (every verifier row of every unit, landed and excised, composed from the
per-unit ledgers `unit_<x>.jsonl` plus the design-round rows bound to this branch);
`feat_500-wave6.anticipation.jsonl` (one re-executable attack per file the open audit BLOCK rows name);
`feat_500-wave6.attestation.jsonl`; `feat_500-wave6.oracle.jsonl`.

## 1. Scope and build order

Wave 6 (eleven program-filed issues left open by PR #587) and wave 7 (three design-first units). Units were built
in parallel on `unit/<name>` worktrees from `b8052978`, each under a pre-committed contract (this scratchpad's
`contracts/<name>.md`, folded into §11 below), a pressure round, a closing adversarial audit, one fix round, and
one final re-audit; from 2026-09-18 the lead applied the stop rules per property (only a mechanism block counts
toward excision) and, after the user's directive of the same day, landed the last units on the lead's own executed
probes instead of further verifier rounds.

Landing order on `feat/500-wave6` (each verified: unit files byte-identical to the unit tip; overlapping files carry
both units' hunks; the seven tree-wide source gates green on the tip):

| commit | unit | unit tip | issues |
|---|---|---|---|
| 2262f2d4 | SEAL S1 | keep/seal-final = 1018e41e | #588 |
| 94136dae | WIDTHFIX | keep/widthfix-final = 76383e84 | #519 |
| 994de691 | GATES | keep/gates-final = 459020a6 | #589 #590 #517 #507 |
| 4c3b91b4 | RELKEY | keep/relkey-final = a331f617 | #551 |
| (pending) | RANGESPLIT | | #540 |
| (pending) | RENDEZVOUS | | plan 68 DIST-2 |
| (pending) | GRAPHARM | | #538 |

Excised by their pre-committed stop rules, branches kept audit-ready, records posted on their issues: COOKGATE
(#539, `unit/cookgate` @ 2a18a1e4), LOADER (#534, `unit/loader` @ 6fb49ec4), PROFILE (#546, third excision,
`unit/profile` @ 82693363), TYPESTATE (#519, two pressure KILLs; its three real findings shipped as WIDTHFIX), SEAL
S2 (the read-only DataFusion registry, `keep/seal-s2-built` = a13ad1cb), GRAPH (#515, two design KILLs).

## 2. The properties, by group (quantified; each with an executed oracle and an executed mutation, named by test)

### 2.1 SEAL S1 (#588) — 2262f2d4
- No crate outside `jammi-db` can obtain a raw `Arc<dyn ObjectStore>` from the registry or the builder:
  `build_object_store` is `pub(crate)` at its definition, `StorageRegistry::driver_for` is `pub(crate)` (a doctest compiles as an outside crate, so `pub(crate)` is what yields the pinned E0624),
  `JammiObjectStore::{new, open, handle_for}` are the only doors and none yields the driver. Oracles: three
  `compile_fail` doctests in `crates/jammi-db/src/storage/mod.rs` pinning the measured codes (E0624, E0603, E0425);
  each red under its re-`pub` mutation (executed by the implementer and by audit #2 from an external probe crate that
  enumerated every `pub` item of jammi-db with the measured rustc code per door).
- The delete oracle `crates/jammi-db/tests/it/models_delete_call_sites.rs` re-anchors its residual routes: route 1
  (registry/builder) closed; route 2 (the DataFusion context) OPEN — the default `RuntimeEnv` pre-registers a
  `LocalFileSystem` rooted at `/` for `file://`, and a registry-level wrapper cannot close it because
  `SessionContext::state_ref` is `pub` (datafusion 54.1.0, `datafusion/core/src/execution/context/mod.rs` line 2043 in that crate) and rebinds the whole
  env with DataFusion types only (executed by audit #2: sealed store → swap → `delete` Ok, file gone); route 4
  (direct `object_store` construction) open by design.

### 2.2 WIDTHFIX (#519) — 94136dae
- Four downstream call sites that hold a width authority apply `ValidatedQuery::require_authority_width`: the
  placement entry's Mixed shape (catalog width, else the first resident local segment), its all-local shape,
  `exact_vector_search`'s no-catalog-width fallback, and `ResultStore::search_vectors_local` (checked
  unconditionally, matching the placed AllLocal arm). Oracles: `mixed_with_local_segment_and_no_catalog_width_…`
  (mutation `None => None` → red, executed by the lead and by audit #2),
  `search_vectors_local_with_a_catalog_width_still_checks_a_deferred_stored_query` (mutation: restore the old
  `if catalog_width(table).is_none()` guard → red naming `segment 0.vector`), the sibling no-catalog-width test.
- `decode_ann_search` boxes every refusal's `JammiError` directly, so an in-process caller recovers the class by
  downcast (test in `crates/jammi-ballista/tests/it/codec.rs`); across a real Ballista job the class is lost to
  `TaskStatus` stringification and a remote client sees `Code::Internal` — named as a residual on #519 (comment
  posted 2026-09-18 before the final audit read it).
- `docs/guide/src/api-stability.md` and `jammi_numerics::query` state the count (four) with a guard clause against
  drifting lower; the entries name the sites without asserting which authority each resolves first.

### 2.3 GATES (#589 #590 #517 #507) — 994de691
- P1 (#589): the R12 deny-coverage sweep runs one process per arm with the hooks dir passed as a parameter; workers
  are reused with `_r12_run_fixture_subset_against`'s `finally` restore as the cross-arm guarantee
  (`R12sweepworker` reds when the restore is deleted); every tempdir is cleaned per call (`R12sweepresidue` reds
  with a named count of 368 when the cleanup is deleted); timeouts scale with the degree, a timeout is a named
  per-arm error; the shipped degree is `max(2, min(cores, 4))` and `R12sweepconc` runs at 1, 2 and 4.
- P2 (#590): `--require-symbol-index` turns any fixture whose call-site set could not be built into a named
  failure; `--only` refuses an empty or unknown selection (rc 1, executed by the lead); the single flag exemption is
  pinned by the self-test; RR45's recursion guard skips loudly.
- P3 (#517): CamelCase NOUN forms of the governance verbs and their plurals (incl. `Transitions`) are findings in
  `pub` declarations; census committed as a floor (6 838 pub items, 29 flagged, 0 new), recomputed by audit #2.
- P4 (#507): `ci/release-feature-manifest.json` is the one place every shipped artifact family's cargo feature list
  lives; `check_release_manifest.py` decides `capabilities` syntactically and `check_flash_attn_closure.py`
  cross-checks the derived closure for every lane (both directions red under mutation, executed by audit #2); the
  Dockerfile's `ARG CARGO_FEATURES` carries no default in either stage and each builder RUN is guarded by
  `${CARGO_FEATURES:?}`, asserted by `shipped_feature_exposure.rs` (the lead executed the added-default mutation at
  Dockerfile:63 → red by name); every workflow build site reads the manifest (nine sites over six lanes, listed in
  that test's doc). The PyYAML gate over workflow scalars was built and excised by the pre-committed fallback after
  blocking twice (§5).

### 2.4 RELKEY (#551) — 4c3b91b4
- A training set's read-back order is a property of the relation: `TrainingSetRelation` (private constructor;
  an empty key is a typed artifact-class refusal — `training_set_relation_with_no_order_columns_is_unconstructible`,
  mutation `if false &&` → red, executed by the lead) renders `ORDER BY` from the row's recorded descriptor;
  `read_back_sql(table)` takes no projection.
- `TrainingSetTable::from_record(record, manifest, outcome)` binds the manifest to the catalog row's recorded
  `definition_hash` before reading the descriptor (`from_record_refuses_a_manifest_whose_hash_disagrees_with_the_records_own`,
  mutation → red, executed by the lead) and refuses an empty descriptor column list before minting
  (`from_record_refuses_a_training_set_descriptor_with_an_empty_column_list`, mutation → red, executed by the lead).
- The whole-row accessor is deleted; the reader-class scan in `crates/jammi-ai/src/fine_tune/training_set.rs`
  counts every hand-built relation read per (path, fn, count) with a syn-parsed accessor-impl exclusion (the lead
  executed the count 2→1 mutation on `pool_context_vectors` → red naming the site; audit #3 executed six span plants,
  all loud).
- The hard-negative mining W=1 adapter bytes are pinned per platform; the Linux value is left to a native CI run by
  an explicit re-pin (`MINING_ADAPTER_PRINTS`, panic-by-name, no `#[ignore]`).

### 2.5 RANGESPLIT (#540) — 700744ab (keep/rangesplit-final = d8013f20)
- RS1: on the optimized plan nothing sits between `OrdinalSplitExec` and `InferenceExec` across the 120-cell grid
  ({PIPE, UDTF} × {bare, GROUP BY, ORDER BY, WHERE, LIMIT} × N ∈ {1,2,4} × target_partitions ∈ {1,2,4,8}); every
  cell contains both nodes (asserted); dropping `benefits_from_input_partitioning` reds 60 of 120 cells (measured on
  the tip, stated as such).
- RS2: the row sequence at N ∈ {1,2,4} is identical per row over every column but `_latency_ms` on both input
  shapes (unsorted `SELECT *`); the merge keyed `[_row_id, _ordinal]` is refused by `SanityCheckPlan` (mutation
  executed, reds the RS2 oracle itself). A second execution of the same plan sees every row again and two
  interleaved runs never share a batch: generations are keyed on the run's `Arc<TaskContext>` identity (a map of live
  generations; the lead executed the constant-key mutation → red) — a double execute of one partition within one
  context is a typed refusal.
- RS3: the split holds no channel and spawns nothing; each partition's poll pulls the next batch under one lock, so
  any subset of partitions polled to completion conserves every row exactly once (concurrent-polling oracle; a
  duplicate-delivery mutation reds it); residency is structural (the shared pull holds a stream handle, counters and
  flags, never a batch) and no instrument claims otherwise; audit #1's three wedge shapes are committed tests.
- RS4: the key cast is a typed refusal naming the column and its type; a typed refusal raised below the split
  reaches the caller as that variant at every N (the observing partition returns the owned `DataFusionError`
  unchanged; the lead executed the stringifying mutation → red; end-to-end over `generate_text_embeddings` with one
  NULL id at partitions ∈ {1,2} → `InvalidKey` both).
- RS5: exactly four roots under `crates/jammi-ai/src` (the oracle's stated universe, `crates/jammi-ai/tests/it/rangesplit.rs`) build an `InferenceExec` through `wrap_with_split_and_merge` (`annotate_plan`,
  `infer_materialize`, `build_embedding_plan`, the embedding refresh) — a syn oracle over every `.rs` under
  `crates/jammi-ai/src` on disk with a crate-wide alias table (use-as, cross-file `pub use … as`, `type` alias,
  nested modules, macro definitions, glob imports, untracked files all caught; audit #3's seven plus two shapes);
  `AnnSearchExec`/`GangExec` state why they keep one partition.
- RS6: no wire form — a plan carrying the split is refused typed for distributed submission (the contract's
  pre-committed exit, taken after the audit read ballista 54.1's stage cut at the merge; recorded on #540).
- RS7: forwards are admitted by a per-exec permit shared by every partition of that exec (CPU: available
  parallelism; GPU: one, the device-wide scheduler seam named, not built); the counter oracle reds under the
  no-permit mutation.
- RS8: `ResultSink::batch_num`/`checkpoint` count merged batches; the oracle under partitions = 2 with a one-row
  batch size measures three merged batches (the cause stated: the batch size, not the file count); no
  resume-by-checkpoint feature exists to name (`get_checkpoint` has no caller — stated).
- `[inference] partitions` is refused outside 1..=1024 at load and at session construction (a struct-literal config
  cannot bypass it); at the default a multi-partition input is coalesced so the exec declares one partition and
  reads every row (oracle through the real `annotate_plan`; the lead executed the coalesce-removal mutation → red).
### 2.6 RENDEZVOUS (plan 68 DIST-2) — 7ccda2bf (keep/rendezvous-final = a146a78a)
- RV1: `SegmentPlacement::plan(table, segments) -> Result<Vec<Vec<PeerAddr>>>` replaces the per-segment `owners`
  call, called once above `resolve_search_mode`'s loop; an arity mismatch is a typed `Catalog` refusal
  (`arity_guard::placement_reply_shorter_than_segment_count_is_a_named_arity_refusal`; guard removed → red,
  executed by audit #2 — a short answer silently truncated eight placement tests before the guard).
- RV2: the ring is the `instances` rows with `peer_addr`, live under the liveness margin, sharing this replica's
  result-root identity, self included; the liveness + root conjuncts are one SQL fragment shared with
  `list_gang_members`; an empty or self-less ring falls back to all-local and counts
  `jammi_placement_ring_empty_total` — registered with the server's scrape registry (`install_ring_empty`,
  `RingEmptyCollector`) and proven 0→1 through a real `OssServer` (registration deleted → red, executed by the lead
  and by audit #2); present only under `placement = "rendezvous"`, stated in operability.md. 17 ring cases on SQLite
  and live Postgres single-threaded (the lead re-ran the live lane on the landing head).
- RV3: the Postgres `stale_before_clause` compares canonical TEXT with no cast (Seq/Index scan measured; one bind;
  database clock); the SQLite triggers guard the stamp's shape, the Postgres CHECK also calendar validity —
  `last_seen_at_is_fresh`'s doc states exactly that after audit #2.
- RV4: `[server] placement` refused typed without `peer_advertise` at `MembershipConfig::validate`, reached from
  `load_from` and `from_config`, unbypassable on the embedded path.
- RV5: `domain_hash(domain, parts)` extracted; `content_hash_row` byte-identical (independent hand-rolled fold);
  the placement score (big-endian u64 of bytes 0..8 over the length-prefixed instance id, table, segment id) pinned
  by three golden u64s recomputed independently in Python by audit #2 (byte-window/endianness mutation → red,
  executed by the lead and the auditor); the domain set is prefix-free and GATED tree-wide: a syn walk over every
  `.rs` file `git ls-files -- crates/` lists resolves every `domain_hash` call's first argument (literal,
  `.as_bytes()`, same-crate const) into a reviewed set, names any unresolvable one, and pins the two hand-rolled
  domains (`manifest.rs`, `version.rs`) — the lead planted `domain_hash(b"jammi", ..)` in a tracked test file →
  red naming the file.
- RV6: one untransacted statement per placed search (`BackendImpl::query_untransacted`, `pub(crate)`, one caller);
  measured 3.9–5.0 ms at 101 rows and 10.2–10.9 ms at 10,101 rows on the scratch host, reproduced independently by
  audit #2 (2.07 / 8.43 ms), with the EXPLAIN reason (Seq Scan; `idx_instances_seen` is on `last_seen_at` alone and
  `result_root_identity` is unindexed) stated in the doc.
- RV7: the distributed lane's placed-search leg (per-fleet placement knob, shared-MinIO embedded table, owner-side
  `/metrics` read, K4 byte parity, D8 retry_ok + UNAVAILABLE with real kills) run live 4/4 by the implementer on
  native darwin MinIO + the scratch Postgres; the timing coupling (worker-2's row must still be live while the
  parity loop runs) is stated and guarded by an elapsed assertion before the second kill.
- RV8: no new task or thread anywhere in the diff (swept). RV9: docs re-anchored (session.rs's stale root-identity
  sentence, deploy-server.md's twin, security.md I-PEER delta, configuration.md, reference-topologies.md Shape D,
  DIST-DATA-PLANE.md §9, PROGRAM.md); CHANGELOG entries under [Unreleased].
### 2.7 GRAPHARM (#538) — 869f20db (keep/grapharm-final = cd679125, plus two lead integration edits)
- GA1: the sample is a function of the input SET — both scans read under `GRAPH_READ_ORDER_RULE_V1` (full projected
  tuple ascending, NULLs first); a duplicate node id is refused naming it (`graph_sample_is_a_function_of_the_set_not_the_scan_order`;
  audit #1 reverted the two ORDER BYs → the reference-sampler tests diverged). Every reference sampler in jammi-ai
  and jammi-server tests goes through one `sort_into_graph_read_order` helper (the class fix for the two tests GA1
  reddened); `cargo test -p jammi-ai --test it` full: 658 green on the unit tip.
- GA2: `ProducingDescriptor::GraphTrainingSet { node_source, edge_source, id_column, text_column, read_order_rule,
  sample… }` folds every output-affecting sampling determinant (K7 test destructures without `..`; a new
  `GraphSampleConfig` knob is a compile error at the conversion site — mutation executed by the implementer).
- GA3: the format is decided from the config; an anchor whose whole candidate pool lies inside its excluded
  neighbourhood is refused at SAMPLE time naming it, before any write; NULL negatives refused at both text decode
  sites and at the media-triplet decode.
- GA4: leading `_ordinal`; the loader reads the table through `TrainingSetTable::relation()` (RELKEY's
  relation-owned committed order — the lead's integration edit on the consolidated branch, removing the loader's
  hand-built `ORDER BY`); the Batches arm never re-imposes the sort; the pinned adapter bytes move if the order does.
- GA5: one input seam (`TrainingSetInput::{Sql, Batches}`) through one `materialize_training_set`; the graph
  producer is the sole writer of a graph training set and `recompute` calls it.
- GA6: two attempts of one job id never displace each other (unpinned anchors short-circuit reuse; per-attempt
  table names; two distinct `ready` rows asserted).
- GA7: `GraphSampler::sample_into(emit)` streams pairs into bounded batches — no `Vec<SampledPair>` of the whole
  output on the emit path (a source oracle over `materialize_graph_training_set`'s body asserts `.sample_into(`
  and never `.sample()`; mutation → red); the named `training_set_graph_sample` reservation is sized from the
  measured allocation (text + per-`String`/bucket overhead) and held by a `ReservationGuard` until the write commits
  (`fine_tune_graph_reservation_is_released_after_the_write_commits`; the lead executed the early-release mutation →
  red).
- GA8: EXITED per §3 after the closing audit executed two refutations (a `Peer` member read `ORDER BY anchor,
  positive` while rank 0 read `_ordinal`; the W=2 parity oracle was unchanged by garbage member rows). Both `Peer`
  refusal sites are restored, typed, naming #538 and the reason (`graph_fine_tune_peer_gang_is_refused_by_name`);
  every multi-host claim narrowed in CHANGELOG, the guide, DESIGN.md §8 and README. The next attempt starts from
  the member's committed-order read and an executed gradient-propagation proof (comment on #538).
- GA9: `recompute` replays the sample from the recorded descriptor byte-identically; the recorded `task`/`format`
  are threaded through and asserted equal to the replay's derivation (mutation → typed refusal).
- GA10: docs re-anchored; MAINTAINER-GUIDE's PRODUCING-DESCRIPTOR-VARIANTS block lists the new variant (doc parity
  green); `source_id` is an identifier (`graph__node-X__edge-Y`), never a sentence; replay refuses non-finite
  `return_p`/`in_out_q` bit patterns; the Linux byte pin (measured under QEMU) moves only by an explicit commit.
- Disclosed residual: a graph fine-tune's `FineTuneRun` carries `materialization_source: None` — nothing links the
  trained model's row to the `GraphTrainingSet` table it trained from (on #538).
- Integration on the consolidated branch: `recompute_graph_training_set` returns the table name through the
  accessor and is a reviewed count-1 entry of RELKEY's reader-class allow-list.

### 2.8 LOADER (#534) — eb37984f (keep/loader-final = 686e05c9; revived and folded after its excision)
- The cu12 tarball's self-sufficiency is proven in a REAL chroot jail on the release lane: bundle-able members
  hardlinked under `/lib` (a jail write cannot reach the stage's inodes — executed oracle, md5 identical
  before/after a real `bundle_build_jail`), platform members byte-copied under `/platform`, the loader at its own
  `PT_INTERP` path; `bundle_assert_jail_class_provenance` runs inside `bundle_build_jail` and refuses a member in
  the wrong class directory by soname AND by inode provenance (three failure shapes executed in a real jail).
- The trace is a tolerant `LD_TRACE_LOADED_OBJECTS` run (python fork → chroot → execve with a pinned minimal env);
  `bundle_verify_jail_report` judges EVERY report line by class (bundle-able ⇒ jail lib dir, platform ⇒ platform
  dir, driver ⇒ `not found`, anything else ⇒ named failure); the vdso self-line is admitted only by state, path and
  raw-text shape (three spoofs refused); RPATH/RUNPATH refused on every staged object except `$ORIGIN`-only entries
  by component (the release lane found NVIDIA's `RUNPATH [$ORIGIN]`); an `LD_TRACE_LOADED_OBJECTS` leak is refused
  in the workflow before python is exec'd (structurally undetectable in-script, proven under python:3.12-slim).
- The workflow-shape gate strips inline comments quote-aware (the trailing-comment dodge is its own fixture);
  the lead's real-jail probe on the final head: build 0, file-set 0, happy 0, missing-transitive 1, host-path 1.
- The real jail report fixture (`ci/scripts/fixtures/cu12_jail_report_real.txt`) is captured from the lane's own
  dispatched run on this branch and committed (provisional flag dropped in that commit).

### 2.9 PROFILE (#546) — c9ecfc8d (keep/profile-final = 17d91ddd; revived and folded after its third excision)
- K1: `ProbedOpId` is the closed enum of probed ops; no `ProbedOp` value ever enters the fold; the index tool's
  output is byte-identical to base (verified in both audits and on the final head).
- K2': the profile is EX ANTE — `BUILD_FACTS`, `admission_mode()`, and the `JAMMI_KERNELS_DISABLE` set resolved per
  row at the JOB's backbone dtype class through the same registry key `admit` reads (`registry_keys_for(dtype).next()`;
  1,200-case equivalence against the real `admit()` outcome executed by audit #2); three states per row
  (`disabled` / `enabled` / `n/a`), the `all` wildcard equal to the explicit full list at a dtype, `DtypeClass::Any`
  refused. The fold site passes `dtype_class_of(config.backbone_dtype)` — audit #2's block was the inference
  precision — proven by `kernel_admission_profile_f16_backbone_moves_the_hash_under_its_own_cast_key` (the lead
  executed the fold mutation back to `compute_precision()` → red with identical hashes); the e2e oracle carries the
  dtype across the child-process boundary explicitly and touches no model.
- K4/K5: the jammi-db manifest field is hash-complete over any line-shaped content; every producer that invokes a
  model records the profile or `None` (pipeline/embedding.rs stated); the residual hardware arm (the env folds
  `ComputeDevice::Cuda{ordinal}`, never the compute capability) is stated in the manifest doc.
- Found and fixed en route: the e2e harness's profile transport truncated a multi-line profile to its first line
  (invisible while every assertion checked the first row); encoded on one line now.
- The journey-marker gate's `[A-Z]\d` pattern fires on `F32`/`F16` in pub rustdoc; the renderer's doc names the
  precision classes in lowercase prose (lead edit on the unit branch).

## 3. Invariants crossed
K1 (compiler-forced enumeration): the ProbedOp table stays typed (PROFILE excised, nothing folds rows); RELKEY's
descriptor match is exhaustive. K4 (byte parity): WIDTHFIX's bench floors and RELKEY's mining pin unchanged; GRAPHARM's
graph K4 pin pending. K5 (append-only migrations): no migration in this wave. K7 (identity completeness): RELKEY's
hash binding; RENDEZVOUS's placement hash golden vectors (pending landing).

## 4. Deviations the lead ruled on (each opened in the code before acceptance)
- WIDTHFIX A4: the implementer made `search_vectors_local`'s check unconditional instead of correcting the comment —
  accepted after reading the placed AllLocal arm it mirrors; the added test's class expectation (Stored-provenance
  mismatch is `IncompatibleFormat` by design) corrected the lead's own brief, verified in `error.rs`.
- RELKEY: `from_record`'s hash binding is to the catalog row's attestation; a caller who authors the record itself
  authors both sides (the §0-disclosed hand-built-record residual) — the doc states exactly that after audit #3.
- GATES: the no-default `ARG` + `:?` guard is kept as the fail-closed form under the P4 fallback (the contract's
  fallback text predated the §4(ii) fold that removed the default).
- SEAL: S2 excised by F7 on the second S2 block; the reword at `crates/jammi-db/src/storage/config.rs:24` avoided the journey-marker gate's
  false positive on the `S3` token without widening the gate's vocabulary.

## 5. Residuals and cuts (each with its executed refutation or its filed rebuild)
- #588 (stays open, scoped to route 2): route 2 (the DataFusion context) is open; the next attempt is a context facade that never exposes
  `state_ref`/the runtime env.
- #507: no gate covers a feature literal placed in a workflow scalar (`build-args`, `--features=`, input
  defaults); the next attempt decides by value over the whole parsed document — rebuild unit #593.
- #519 (stays open, scoped to the entry half): the Ballista wire loses the caller class (`TaskStatus` stringification); the entry-half type-state starts
  from the 32 `validate_query(` sites.
- #551: the hand-built-record route and `CacheOutcome::Reused { table }` still yield a bare name — rebuild unit #594.
- #546: revived and folded (§2.9) — the fold site now resolves on `config.backbone_dtype`, `DtypeClass::Any` is
  refused, the F16 leg is executed; no residual.
- #534: revived and folded (§2.8) — the chroot-jail arm, the class-provenance assert and the REAL jail report
  captured from lane run 35300438931 (`ci/scripts/fixtures/cu12_jail_report_real.txt`); residual R-MARKER below.
- #539: one bounded round remains, by the auditor's own account.
- #540: RS6 (the wire form) refused typed — ballista cuts a stage at the merge; a distributable form must live in
  Ballista's partition model — rebuild unit #595.
- #538: GA8 (the Peer gang) refused for graph fine-tunes — the member's read order and the missing
  gradient-propagation proof. Rebuild unit filed and scheduled as #591 (P1 one committed order for every rank
  with a displacement-sensitive parity oracle, P2 the graph descriptor admitted by the member's bind path, P3 an
  executed W=2 gradient-propagation proof, P4 the refusal deleted only when P1–P3 are green).
- R-MARKER: the lead-gate library's committed test-failure marker set names cargo's `test result: FAILED`, the lead-gate self-test's `FAIL —` and pytest's `= FAILURES =`; the bash test harness `ci/scripts/test_bundle_cuda_libs.sh` reports `FAIL[<check>]:` and `N of M check(s) FAILED`, so a mutation witnessed only by that harness cannot enter an attestation row. Rebuild unit: extend the marker set from the harness's own committed failure line (one gate, one fixture), on its own PR — #596; until then LOADER's executed mutations live in §2.8.
- R-RS5-UNIVERSE (oracle advisory A3): the RS5 source oracle enumerates `crates/jammi-ai/src` only; a fifth `InferenceExecBuilder::new` lives outside it at `crates/jammi-ballista/src/codec.rs:329` (the wire decode — no divergence, the split has no wire form and encoding one is refused typed). An `InferenceExec` root added later under `jammi-ballista`/`jammi-server` would escape RS5; rebuild unit: widen the oracle's universe to every crate that depends on `jammi-ai` (one gate, one planted counterexample) — #597.
- R-STALE-CLAUSE (oracle advisory A4): `stale_before_clause`'s Postgres arm is now a cast-free lexical comparison (`crates/jammi-db/src/catalog/lease.rs:288-295`) with six call sites (`instance.rs`, `model_repo.rs`, `jobs_repo.rs` ×4); the executed EXPLAIN + row-set differential oracle (`crates/jammi-db/src/catalog/lease.rs:918-1060`) covers an `instances`-shaped fixture only. The other in-class columns hold canonical shape through the Postgres schema-edge CHECKs (`sdchk__jobs__updated_at`, `sdchk__models__updated_at`), which are not retroactive over rows written before them. Bounded residual; rebuild unit: the differential oracle over every `stale_before_clause` caller's table — #598.

## 6. Acceptance, restated as executed oracles
(per unit, the tests named in §2 with their mutation lines; the lead's own executed probes are recorded in the
session STATE log and in the relay artifacts' `probe` lists.)

## 7. Pressure rounds (one row per round in `docs/rigor/feat_500-wave6.jsonl`)
Wave 6: seven contracts pressure-tested (KILL ×4 → TYPESTATE excised after a second KILL; COOKGATE, RELKEY, SEAL
rebuilt as v2 and re-pressured; REFINE ×3 folded: LOADER, PROFILE, GATES). Wave 7: two design rounds each for GRAPH
(KILL, KILL → excised), GRAPHARM, RANGESPLIT, RENDEZVOUS (promoted to implementation contracts). The first
RENDEZVOUS design verdict carried no JSON fence and is recorded UNBOUND; its second round is the pressure row.
Phase-5 oracle on 714cfb41: PASS (`docs/rigor/feat_500-wave6.oracle.jsonl`), mechanical gates all 0, four advisories folded above (A1 §2.1 wording, A2 the lead's own brief named a migration this wave does not carry — §3 was already right, A3/A4 as §5 residuals).

## 8. Stop rules honored
Second-block excisions: TYPESTATE (pressure), COOKGATE, LOADER, PROFILE (closing audits), SEAL S2 (F7), GRAPH
(design). Pre-committed fallbacks taken: GATES P4 (lane-sites gate excised), RANGESPLIT RS6 (wire refusal),
GRAPHARM GA8 (Peer refusal kept). From 2026-09-18 the rule counts per property and only mechanism blocks
(user decision); prose-only findings were folded and verified by the closers.

## 9. Closers the lead runs on the tip
The committed merge path (`bash ci/scripts/merge_path.sh`: static, guards, swarm, tests incl. the Postgres lane on
the scratch Postgres 16 server on loopback port 54329, records), the swarm-gate set after every landing (seven tree-wide gates,
green after each of the nine landings), the lane dispatches on the pushed branch — `release-binaries.yml` run
35300438931 (every build job green; promote skipped by construction, no tag; its `cu12-jail-report` artifact is
the committed LOADER fixture) and `server-image.yml` run 35300440387 with `selfcontained=false` (success) — which
are GATES P4's §4(iii) precondition; the rigor record composed from every unit ledger, the anticipation and
attestation exports, the oracle record last, then the PR.

Consolidated-tip integration the merge path found and the lead fixed (151fe082, a1fd5bf0): three test reads of the
record field RELKEY made private (GRAPHARM's tests) routed through accessors and the catalog; about thirty
maintainer-guide citations re-resolved to the definitions' current lines (the units moved `store/mod.rs` by ~800
lines); the exposure test's manifest-read citations rewritten with the lane key at each line; synthetic Rust source
fixtures (RANGESPLIT's alias oracle, RENDEZVOUS's domain gate) restructured so the kernel-oracle stripper's
reviewed-literal marker sits directly above each `fn`-shaped literal; the placed-search leg's readiness deadline
reviewed in the wall-clock-bound inventory.
- Attestation sample (`docs/rigor/feat_500-wave6.attestation.jsonl`): eight rows, one per landed unit except LOADER, each carrying at most three executed mutations (`rc_before` 0, `rc_after` non-zero, the committed test-failure marker) and, on the GATES row, the exclusion sentence for every one of the wave's 218 new test surfaces. LOADER's mutations are witnessed by `ci/scripts/test_bundle_cuda_libs.sh`, whose failure marker (`FAIL[<check>]: …` / `N of M check(s) FAILED`) is outside the committed test-failure marker set, so they are recorded in §2.8 and not sampled here (residual R-MARKER in §5).

## 10. Gate table (an executed run at the named tip; logs under the session scratchpad)

`bash ci/scripts/merge_path.sh --only <stages>` on the consolidated branch, Postgres 16 lanes against a scratch server on loopback port 54329; each row is one executed run, newest last. Reds were fixed at the root and the stage re-run on the fixed tip; the one red that remains is by construction.

| tip | stages | ran | red | disposition |
|---|---|---|---|---|
| 7ccda2bf | static | 7 | 1 — rustdoc `-D warnings` | fixed in the GRAPHARM fold (869f20db) |
| 869f20db | static, guards, swarm | 124 | 5 — clippy workspace + postgres (private `record` reads), citation resolver, kernel-oracle standard, SWARM_GATE_TOUCHED | 151fe082 (accessors, ~30 guide citations, fixture markers); swarm red by construction |
| 151fe082 | tests | 6 | 1 — hermetic workspace lane (bounds inventory) | a1fd5bf0 (REVIEWED_SITES entry, Class::D) |
| c9ecfc8d | static, guards, swarm, tests | 131 | 6 — cu12 bundle suite + provisional-fixture refusal, citation resolver, kernel-oracle standard, arch-validation freshness, SWARM_GATE_TOUCHED | 0805b557 (the real jail report) + 5a066041 (KO-7 require-gates registered, arch waivers bumped with executed notes, 11 citations); static and tests green |
| 5a066041 | guards, swarm | 118 | 1 — SWARM_GATE_TOUCHED | by construction: the diff amends `.claude/hooks/lead-gate-lib.py` and `ci/scripts/check_lead_gate.py` (GATES); admin-merge pre-authorized for this red alone |
| 5a066041 | tests | 6 | 0 | hermetic workspace, jammi-db test-hooks, encoders golden-parity, three Postgres lanes |
| f813aee5 | static, records | 9 | 1 — oracle gate (no oracle record yet) | static all green (fmt, clippy workspace + gated surfaces + postgres, rustdoc, mdbook); `check_rigor_record.py` OK |
| 1845253c | static, guards, swarm, tests, records | — | — | the doc-claims lens fold (20 prose claims re-anchored, two mechanisms aligned to their stated property; three read-only auditors, 145 files, 408 claims); the closing run on the final tip, result on the PR |

Lanes dispatched from the branch: release-binaries 35300438931 (success, the jail report captured — GATES P4 §4(iii) precondition), server-image 35300440387 (success), gpu-prove 35304096937 (four legs, result recorded on the PR).

## 11. Units as built — the unit contracts, folded by the lead

### 11.1 SEAL (S1 shipped; S2 built and excised) — the unit contract as pressure-tested and folded

## SEAL v2 — SEAL-STORE: no crate outside jammi-db can delete under a catalog-managed root (wave 6; #588; round-1 KILL folded)

Supersedes `seal.md` in full. Implementer: db (cross-crate scope: jammi-bench, jammi-ai tests + peer_tests.rs,
jammi-db tests, `crates/jammi-ai/src/session.rs:902`, `crates/jammi-ballista/src/roles.rs:443`).
Worktree `/Users/vijaychakilam/git/f-inverse/wt-seal`, branch `unit/seal`, base `b8052978`. Targets
`/Users/vijaychakilam/git/f-inverse/targets/seal`. Rules: `RULES-implementer.md`.

### 0. Round-1 record (executed on b8052978; binding)
- Raw catalog SQL cannot be sealed by visibility: `BackendImpl` is a `pub` enum with `pub` variants, `SqliteBackend::open`
  / `PostgresBackend::open_with_options` are `pub` (121 direct constructions outside jammi-db/src; crates/jammi-ai/tests/it/jobs_shutdown.rs:1735
  opens its own backend on the same file), and a foreign SQLite library reaches the same file in-process
  (esc_073). 113 `backend_arc()` sites and 192 `tx.{execute,query,query_opt}` sites live in tests across five crates;
  `crates/jammi-db/tests/it` is a separate crate; negative-path fixtures (crates/jammi-db/tests/it/migrations.rs:839-895 pre-migration schema,
  crates/jammi-server/tests/it/grpc_job.rs:891 / crates/jammi-cli/tests/it/jobs.rs:132 hand-built `jobs` rows) exist precisely because the engine's verbs refuse those states.
  **Disposition of #588's SQL half:** it is NOT a sealing unit. The property "a training-kind `jobs` row is writable
  only through `submit_job_deduped`" is carried, for production code, by the enumerating submit-seam oracle that already
  scans every crate's `src/` (wave 5, fdf28c27) — and the engine is "on a trusted network, not a security boundary"
  (`docs/guide/src/security.md:3`). Tests keep raw SQL by design. This unit records that disposition on #588 (the
  issue's SQL half closes as "by oracle over production sources; raw SQL from tests and direct backend construction
  are by design; foreign-library access is outside any seal") and ships no SQL change.
- Raw object store: every external acquisition (7 jammi-bench, jammi-ai/jammi-db tests) immediately wraps the driver
  in `JammiObjectStore::new(driver, url)` — the migration is mechanical. Route 3 (#588's first comment):
  `JammiSession::context()` (crates/jammi-db/src/session.rs:973), re-exposed by jammi-ai crates/jammi-ai/src/session.rs:902 and held by ballista crates/jammi-ballista/src/roles.rs:443,
  yields the DataFusion `SessionContext` whose `runtime_env().object_store(url)` returns the registered credentialed
  driver — an external probe compiled `ObjectStoreExt::delete` on it. The correct error codes: an inherent method
  sealed to `pub(crate)` is **E0624**; a free fn reached by path is **E0603**; `crates/jammi-db/src/storage/mod.rs:28`'s `pub use
  builder::{build_object_store, …}` must become `pub(crate) use` (else E0365). rustdoc DOES enforce a declared code
  (a `compile_fail,E0603` block whose real error is E0624 FAILS) — so the fence codes are load-bearing and must be
  the measured ones. The API-freeze baseline is a gRPC wire set and records no Rust item.

### 1. Properties (binding)

**S1 — No crate outside `jammi-db` can obtain a raw `Arc<dyn ObjectStore>` from the registry or the builder.**
`StorageRegistry::driver_for` → `pub(crate)`; `storage::build_object_store` → `pub(crate)` (+ the re-export at
`crates/jammi-db/src/storage/mod.rs:28` → `pub(crate) use`); new `StorageRegistry::handle_for(&url, cloud) -> Result<JammiObjectStore>` is the
public acquisition (promote `JammiObjectStore::list` to `pub` if a migrated site needs it). Every external site
migrates to `handle_for`. Oracle: rustdoc `compile_fail,E0624` (method) and `compile_fail,E0603` (free fn) doctests in
`storage/mod.rs`, codes measured not assumed. Mutation: re-`pub` `driver_for` → the E0624 doctest FAILS (the code is
not emitted).

**S2 — The DataFusion session context cannot delete under a catalog-managed root.** Every object store the engine
registers on the shared `SessionContext`'s `RuntimeEnv` is wrapped in a `ReadOnlyObjectStore` newtype (implements
`object_store::ObjectStore`; `get`/`get_opts`/`get_range`/`head`/`list`/`list_with_delimiter` delegate; `put*`,
`delete`, `delete_stream`, `copy`, `rename`, `copy_if_not_exists`, `rename_if_not_exists`, `abort_multipart`/
`put_multipart*` return `object_store::Error::NotSupported` naming the root). The engine's own writes (Parquet
result tables, sidecars, manifests, models) go through `JammiObjectStore` — the implementer PROVES this by grepping
every `runtime_env().register_object_store` / `object_store(` site and every DataFusion write path (`COPY TO`,
`INSERT INTO` on listing tables, `write_parquet`) in the tree, and lists them; any engine write that DOES go through
the context is re-routed to the handle (or, if a DataFusion writer must be used, the contract's pressure round 2
rules on the exception — do not decide it alone). Oracle (runtime, RED at base): an integration test in jammi-ai
obtains `session.context().runtime_env().object_store(&url)` and calls `delete` on a live `models/` key → `Err(NotSupported)`
and the key still exists; a control that the same path can `get`. Mutation: register the raw driver → RED.

**S3 — The delete oracle's universe is re-anchored.** `models_delete_call_sites.rs`'s module doc names no residual
raw route: the registry/builder route is sealed (S1), the context route is read-only (S2); the remaining residual —
constructing an `object_store` driver directly from the filesystem/cloud credentials outside jammi-db — is named as
by-design under the trusted-network posture (security.md). Mutation fixture kept.

**S4 — Public surface diff enumerated.** Every `pub` item of jammi-db removed/narrowed, listed in the hand-back
against the crate's terminal-0.x semver posture (the baseline's own header); the proto freeze is untouched (say so).

**S5 — Nothing observable changes for embedded/remote users.** K4 parity tests unchanged; the full jammi-db and
jammi-ai it-suites the change touches run green (filtered runs listed).

### 2. Non-goals
No SQL/backend visibility change (see §0 disposition). No change to the deleter's guard logic. No new drivers.

### 3. Stop rule (pre-committed)
If S2 blocks in pressure round 2 or the closing audit on an engine write that genuinely must flow through the
DataFusion context (measured, cited), S2 ships as: read-only wrapper for every root EXCEPT that one, and the exception
is a named residual on #588 with the write path cited. A second block on S1 excises the unit (record on #588).

### 4. Pressure-round-2 folds
(appended by the lead after the verdict)

### 4. Pressure-round-2 folds (2026-09-17, REFINE — executed against datafusion =54.1.0 / object_store =0.13.2; BINDING, override §1 where they differ; probe sources kept at /Users/vijaychakilam/git/f-inverse/targets/pt-seal2/{probe,fence})
F1 — **S2's mechanism is the RuntimeEnv's object-store REGISTRY, not the two registration sites.** Both engine registrations (crates/jammi-db/src/store/mod.rs:4536, crates/jammi-db/src/source/file_format.rs:229) skip `Scheme::File | Memory`; the default models root is `file://`, answered by DataFusion's own pre-registered `LocalFileSystem` rooted at `/` — executed: `delete(models/weights.bin) = Ok(())` through a plain context. Fix (proven green): install a wrapping `ObjectStoreRegistry` via `RuntimeEnvBuilder::with_object_store_registry` at crates/jammi-db/src/session.rs:209, wrapping in BOTH `register_store` and `get_store` (`ReadOnlyObjectStore(LocalFileSystem)` → delete/copy/rename/put → `NotSupported`, key survives; get/head/list and a listing+parquet read work; `COPY … TO` through the context → NotSupported and no file). One seam covers both registration sites, `single_partition_context` (crates/jammi-db/src/session.rs:1180), Ballista's `runtime_producer` (crates/jammi-ballista/src/roles.rs:443), every `TaskContext`, and every future site. No engine write flows through the context (rg over every write verb; engine parquet goes via storage/writer.rs over the handle; ballista shuffle via std::fs; INSERT INTO a single-file ListingTable is refused by DF) — no exception needed.
F2 — **The wrapper implements exactly the SEVEN required methods** (`put_opts`, `put_multipart_opts`, `get_opts`, `delete_stream`, `list`, `list_with_delimiter`, `copy_opts`); `get/head/delete/copy/rename/put` are `ObjectStoreExt` blanket methods (E0407 if written); no `abort_multipart` exists. `ObjectStoreExt::delete` demands `delete_stream` yield exactly ONE item per input location — the refusal is one `Err(NotSupported{..})` per location (an empty stream becomes `Generic{store:"ext"}`); refusing `copy_opts` + `delete_stream` covers `rename`.
F3 — **S1 seals the free fn AT ITS DEFINITION** (`pub(crate) fn build_object_store` in crates/jammi-db/src/storage/builder.rs:23 — `crates/jammi-db/src/storage/mod.rs:9` is `pub mod builder;`, so narrowing only the re-export at :28 leaves the module path public: executed, compiles clean and runs) and fences BOTH paths: `compile_fail,E0603` over `jammi_db::storage::build_object_store` and over `jammi_db::storage::builder::build_object_store`; `compile_fail,E0624` for `driver_for`. `pub(crate) use builder::{build_object_store, DynObjectStore};` compiles. `JammiObjectStore::list` needs no promotion (no migrating site calls it).
F4 — **Migration universe (executed):** `driver_for` outside jammi-db/src = 20 (bench 7, jammi-db tests 11, jammi-ai tests 2); `build_object_store` = 2 (crates/jammi-ai/tests/it/peer_gang.rs:443, crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:1246). All one-liners except `crates/jammi-ai/src/fine_tune/collective/peer_tests.rs:1237-1248` (a `#[cfg(test)]` mod inside jammi-ai/src building a `RecordingStore` over the raw driver): it constructs `object_store::local::LocalFileSystem` itself — residual route 4, by design, named.
F5 — **S3 lands only after F1** (crates/jammi-db/tests/it/models_delete_call_sites.rs:20-32 today truthfully says the default registry pre-registers `file://`; re-anchor to "the registry wraps every store read-only" and keep route 4 in the residual list). `crates/jammi-ai/tests/it/pinned_source_gate.rs:3928-3960, 4047-4051, 5172-5177` record `register_driver_for_url`'s `Arc::ptr_eq` identity property — falsified by wrapping; update that recorded text in the same change; `crates/jammi-db/src/store/mod.rs:4836`'s unit test survives (own `SessionContext::new()`) — state it.
F6 — **S5 restated:** the seal REMOVES a user-reachable capability — `JammiSession::sql` (crates/jammi-db/src/session.rs:776) hands arbitrary SQL to DataFusion, so a Flight SQL user could `COPY … TO 'file:///…'` as the server user; after the seal it is `NotSupported`. Stated on #588 and in `docs/guide/src/security.md` as a deliberate behaviour change, with a test asserting the refusal (RED at base: the file is written).
F7 — **Stop rule symmetric:** a second block on S2, from any cause, excises S2 and records the residual on #588.

### 11.2 WIDTHFIX — the unit contract as pressure-tested and folded

## WIDTHFIX — three width-authority sites that hold the authority and skip it (wave 6; from #519's excised TYPESTATE)

TYPESTATE was excised by its pre-committed stop rule after two KILLs (records: the two pressure verdicts on
`unit/typestate`; the executed refutations — a v2-exact scratch crate compiles #519's defect; `defer()`/artifact-check
inter-convert the states; `Node.vector`'s corpus role; the wire needs a readable deferred query — go on #519 at close-out).
What survives is the unit's real teeth: three sites found by reading, each a genuine authority-in-hand-but-unchecked
defect with a constructible oracle. No typestate, no clone regime, no seam reshape.

Implementer: numerics (cross-crate scope granted; the same agent that read the seams). Worktree
`/Users/vijaychakilam/git/f-inverse/wt-typestate`, NEW branch `unit/widthfix` created from base `b8052978` in that
worktree (it is clean at base). Targets `/Users/vijaychakilam/git/f-inverse/targets/typestate`. Rules: `RULES-implementer.md`.
Scope: `crates/jammi-ballista/src/codec.rs` (+ `error.rs` mapping, `tests/it/codec.rs`), `crates/jammi-db/src/index/placed.rs`
(`search_mixed`, + its `mixed_with_fake*` harnesses), `crates/jammi-db/src/store/mod.rs` (`search_vectors_local`),
`crates/jammi-db/tests/it/**` for the store oracle, `crates/jammi-numerics/src/query.rs` docs only (the module doc's
"three production call sites" sentence for `require_authority_width` is re-counted after this unit — never a typed number
without the grep in the same commit).

### 1. Properties (binding; each RED at base, GREEN after; each mutation executed)

**W1 — `decode_ann_search` checks the width against the authority it holds.** `crates/jammi-ballista/src/codec.rs:359`
passes `None` while the `ResultTableRecord` (with `dimensions`) is resolved at :346-357; it passes `table.dimensions()`.
Oracle: `crates/jammi-ballista/tests/it/codec.rs:470-536`'s hand-built `pb::AnnSearchExecNode` against a table with
`dimensions: Some(4)` and a 3-wide query → refused at decode (RED at base: decodes). **Fold (binding):** the refusal must
surface at the client boundary in the CALLER class (`Schema`/`InvalidArgument`), never as `DataFusionError::External`
plan-deserialisation noise — map the decode-time `QueryValidationError::Width` through the existing caller-fault
mapping (prove it with a test through the ballista error mapping, or through the lane's client if a harness exists;
cite the mapping site `crates/jammi-ballista/src/error.rs`). This RED is defense-in-depth (a conforming coordinator
checks first in `QueryBuilder::new`) — the test's doc says so. Mutation: revert to `None` → oracle reds.

**W2 — `search_mixed`'s non-empty-`local` branch performs the authority check before any unit search.**
`crates/jammi-db/src/index/placed.rs:347-392`: when `catalog_dimensions` is `None`, the branch resolves the authority
exactly as the AllLocal arm does (the set's own first segment, crates/jammi-db/src/index/placed.rs:280-287) and calls `require_authority_width`
BEFORE the units loop, so a caller's wrong width is `Width{source: Caller}`, not `ArtifactMismatch("segment N")`. Oracle:
the `mixed_with_fake`/`mixed_with_fake_n_remote` harnesses (crates/jammi-db/src/index/placed.rs:1254/:1264) with a wrong-width query → caller
fault (RED at base: artifact fault). Mutation: drop the check → reds.

**W3 — `search_vectors_local`'s `Some(index)` branch resolves the authority when the catalog records none.**
`crates/jammi-db/src/store/mod.rs:2555-2568` calls `index.search_final` directly; when `catalog_dimensions` is `None` it
first resolves the authority as `search_final_placed`'s AllLocal arm does. Oracle: a table with NO recorded `dimensions`
(binding — with `Some` the oracle is green at base) and a wrong-width query → caller fault. Mutation: bypass → reds.

**W4 — Every existing width-error control stays green** (the list in the round-2 verdict §5: crates/jammi-db/src/index/placed.rs:993/:1501,
crates/jammi-db/src/index/segment.rs:1154/:1135/:1191, crates/jammi-db/tests/it/index.rs:307/:323/:331/:396/:444, crates/jammi-numerics/src/query.rs:368/:409, crates/jammi-server/tests/it/peer_placement.rs:557/:1172,
crates/jammi-db/tests/it/register_computed_embedding.rs:231, crates/jammi-ai/src/fine_tune/hard_negative_miner.rs:431) — run them, list them.

**W5 — Docs reflect the count.** `query.rs`'s `require_authority_width` doc sentence names the new production call
count with the grep that derives it stated in the commit; no journey markers.

### 2. Non-goals
No typestate, no change to `ValidatedQuery`'s API, no seam reshaping, no error-string change outside the three sites.

### 3. Stop rule (pre-committed)
A closing-audit block on W1's error-class fold ships W2+W3 with W1 as a decode-time check whose class is named as a
residual on #519; a second block on W2/W3 excises the unit.

### 11.3 GATES — the unit contract as pressure-tested and folded

## GATES — swarm/CI gate residue (wave 6; issues #589 #590 #517 #507)

Implementer: docs-ci. Worktree `/Users/vijaychakilam/git/f-inverse/wt-gates`, branch `unit/gates`, base `b8052978`.
Targets `/Users/vijaychakilam/git/f-inverse/targets/gates`. Rules: `RULES-implementer.md` (same directory).
Scope: `ci/scripts/check_lead_gate.py` (+ its test file), `ci/scripts/check_rigor_record.py`,
`ci/scripts/check_no_consumer_names.py`, `.github/workflows/{ci.yml,swarm.yml,pypi-server.yml,release-binaries.yml,
server-image.yml}`, `Dockerfile`, `ci/release-feature-manifest.json`, `ci/scripts/check_release_manifest.py`,
`crates/jammi-db/tests/it/shipped_feature_exposure.rs`, `ci/scripts/check_flash_attn_closure.py`,
`ci/scripts/runpod_gpu_prove.sh` (readers of the manifest). Every gate script here is human-amend-only
(`SWARM_GATE_TOUCHED`); the lead admin-merges with that red — do not add allowlist entries to dodge it.

### 0. Premises (verified on main by the lead, 2026-09-17)
- #589: `run_r12_deny_coverage_sweep` (`check_lead_gate.py`) mutates one arm at a time into its own hooks dir and
  runs the R12 fixture subset per arm, serially; ~50 arms × ~60 s ≈ 50 min per CI run (run 35238230860).
- #590: `_symbol_index_unavailable(r)` makes RR42/RR43 a named skip whenever the reader reports "could not build
  the symbol-index required call-site set"; `ci.yml`'s container job `symbol-index-gates` runs
  `check_rigor_record.py --self-test` (line ~2839) with the toolchain present, so a skip there is a silent loss.
- #517: `governance_stem` matches `retire_x`, `RetireX`, `retire2`, never `Retirement`/`Registration`; the wave-5
  pressure round measured that a naive participle rule would newly flag `Registered`(54) `Gated`(29)
  `Registering`(3) `Registers`(4) `Gates`(2) `Promoting`/`Promotes`(1) — legitimate engine vocabulary.
- #507: four literal feature lists remain — `.github/workflows/pypi-server.yml:71,:86` (`cargo_features: jetstream-broker,storage-cloud`),
  `.github/workflows/release-binaries.yml:300` (`--features jetstream-broker,storage-cloud`), `Dockerfile:102`
  (`ARG CARGO_FEATURES=cuda,jetstream-broker,storage-cloud`; `Dockerfile:71` the CPU ARG). The CUDA lanes already read
  `ci/release-feature-manifest.json` `.lanes.<key>.cargo_features`; `check_release_manifest.py` requires every
  `lanes` entry to carry an identical `capabilities` block, which is why the CPU families were kept out of the manifest
  and pinned by `shipped_feature_exposure.rs` check 1 instead.

### 1. Properties (binding; each with an executed oracle and an executed mutation)

**P1 (#589) — The sweep's result is independent of arm execution order and concurrency, and its wall time is
bounded by the slowest arm plus the meta-check, not by the arm count.** Arms run concurrently (each arm already owns
its own mutant hooks dir and its own process group); the per-arm results `per_arm[pos]` are byte-identical to the
serial run's. Oracle: a fixture that runs the sweep on a mutated COPY (the existing `R12sweepmeta` shape) both
serially and concurrently and asserts identical `positions`/`per_arm`; a second fixture proves a genuinely silent arm
is still flagged under concurrency. Mutation: force the concurrent path to drop one arm's result → the equality
fixture reds. Measured: the CI step's wall time on this branch printed in the hand-back (target < 10 min on the
runner; the number is measured, never asserted). The existing `R12sweepmeta`-first order and the report-only
survivor semantics (Z10) are unchanged. Concurrency degree derives from `os.cpu_count()` with a floor of 2; the
self-alarm and kill-path fixtures (`R12alarmkill`) stay excluded from the mutation subset exactly as today.

**P2 (#590) — Where the symbol index CAN be built, an index-backed fixture that cannot run FAILS by name; it never
self-skips.** `check_rigor_record.py --self-test --require-symbol-index` turns the `_symbol_index_unavailable` arm of
every index-backed fixture (RR42, RR43, and any other fixture that consults it — enumerate them in the code, the
list is derived by grepping the callers, stated in the flag's doc) into a `Failure` naming the fixture and the
reader's own warning text. `ci.yml`'s `symbol-index-gates` job passes the flag; the toolchain-less swarm lane does
not. Oracle: a fixture (`RR45`) that runs the self-test in a subprocess with the index made unbuildable (an
unwritable/absent `ci/tools/symbol-index` path via the reader's own override) under the flag → exit non-zero with the
fixture name in stderr; without the flag → the named skip line. Mutation: remove the flag's check → RR45 reds.

**P3 (#517) — A CamelCase NOUN form of a governance verb in a `pub` declaration is a finding; a participle,
agentive or plural form of the same verb is not.** `governance_stem` gains a per-verb morphology table (a closed
mapping, one row per verb in `GOVERNANCE_VERBS`: `promote→Promotion`, `retire→Retirement`, `register→Registration`,
`approve→Approval`, `gate→Gating` is NOT a noun row (ambiguous with the engine's own `Gated`/`Gates` vocabulary —
`gate` keeps only its verb-head rule), `stage→Staging` is NOT a noun row (same reason), `transition` is already
noun-shaped and keeps the existing rule). A noun row matches as a whole CamelCase head token (`Retirement`,
`RetirementPolicy`, `retirement_policy`), never `Retired`/`Retiring`/`Retires`/`Registered`. Oracle: the self-test
pair (`Retirement` → `retire`; `Registered` → None; `RegistrationTable` → `register`; `Registers` → None), plus the
gate run on the REAL tree reports zero NEW findings (the number printed in the hand-back; if a real finding appears,
it is either a genuine leak fixed in this unit with its own commit, or a waiver in the allowlist with all seven rot
rules satisfied — never a rule loosened to hide it). Mutation: delete the noun table → the pair reds.

**P4 (#507) — Every shipped artifact family's cargo feature list lives in exactly ONE committed place, and every
build site reads it.** `ci/release-feature-manifest.json` is reshaped: a top-level `families` object with one entry
per shipped family (the six the exposure test names: cuda tarball, cuda wheel, cuda image, cpu wheel, cpu tarball, cpu
image), each `{ "cargo_features": [...], "capabilities": {...} | absent }`; `capabilities` is REQUIRED on every family
whose features reach `jammi-kernels/flash-attn` or `cuda` (derived, as `check_flash_attn_closure.py` already does)
and FORBIDDEN on every other family. `check_release_manifest.py` enforces that rule instead of "every lane identical"
(the three CUDA families still must carry identical `capabilities`). Every workflow build site reads its family via
`jq`; the `Dockerfile` carries NO default feature list — `ARG CARGO_FEATURES` is required and the image workflow
passes it from the manifest (a build without it fails at `RUN` with a named error, the oracle asserts the message).
`shipped_feature_exposure.rs` check 1 is rebuilt: it parses the workflows (PyYAML-equivalent in Rust: `serde_yaml`
is NOT a dependency — use the manifest + a structural assertion that no `--features`/`cargo_features:` literal
containing a feature name appears at any build site; a literal is a FINDING) and derives the six families from the
manifest alone. Oracle: the exposure test RED on a re-added literal (mutation: put `--features jetstream-broker` back
at `release-binaries.yml`'s CPU site) and RED on a family missing from the manifest; `check_release_manifest.py`
self-test covers: capabilities on a CPU family (finding), missing on a CUDA family (finding), the three CUDA
families diverging (finding). The `deny.toml` exposure line generator keeps producing a byte-identical line for
today's data (assert it).

**P5 — No gate in scope decides from free text where the artifact is parseable.** Any new parsing in this unit
uses PyYAML compose/construct or the JSON parser; a regex may narrow candidates, never decide (a lint the
pressure-tester will apply).

### 2. Non-goals
No new gates. No change to the R12 deny arms themselves or to `lead-gate-lib.py`. No change to the CUDA lanes'
capability data. No edit to `.claude/**`.

### 3. Stop rule (pre-committed)
If P4's Dockerfile/workflow reshaping blocks twice in audit on the release-lane surface, P4 ships as: manifest
`families` + exposure-test derivation + `check_release_manifest.py` rule, with the CPU workflow literals replaced
and the Dockerfile default kept and asserted equal to the manifest by the exposure test (the residual named on
##507). P1–P3 have no fallback: a second block excises that property with its executed record on its issue.

### 4. Pressure-round folds
(appended by the lead after the pressure-tester's verdict)

### 4. Pressure-round folds (2026-09-17, REFINE — executed; BINDING, override §0/§1 where they differ; probes at /Users/vijaychakilam/git/f-inverse/targets/pt-gates/scratch/p{1,2,3,4}_*.py)
**P1 (#589):** real numbers on b8052978 (run 35249216522): 91 arms, 1763.74 s, ~19.4 s/arm on the runner (54.8 s/arm locally); the cited run 35238230860 was CANCELLED and printed nothing. `HOOKS_DIR` is a module global mutated by `_r12_run_fixture_subset_against` (:4176-4190) and read by `_r12_mutant_hooks_dir` (:4142) — a thread pool mis-attributes (executed: arm A's fixtures ran against arm B's mutant; arm B against the REAL lib; global left dead). Mechanism: PROCESS-level isolation (one process per arm; 4 arms/4 processes measured 35.7 s wall) with the hooks dir passed as a PARAMETER through `_run`/`_r12_mutant_hooks_dir` (delete the global mutation); the "own process group" clause is dropped (it describes the attack subprocess). Timeouts (`timeout=10` at :316/:398, `_GIT_TIMEOUT_S_FIXTURE`) scale with the degree, and a `TimeoutExpired` is a LOUD named per-arm error (never the silent non-credit `except Exception: pass` at :4180-4191 produces today — executed). The bound is restated: wall ≈ ceil(91/degree) × per-arm + meta (≈57 s) + the two new oracles (each ≈ one meta run) — at degree 4 on ubuntu-latest ≈ 10–11 min; the hand-back reports the measured number, no target is asserted. The fixed marker paths at :1770/:2028 (`/tmp/should-never-run-*`) become per-arm unique.
**P2 (#590):** the flag is a property of the RESULT, not an enumeration: under `--require-symbol-index`, ANY `run_check` result whose warnings carry "could not build the symbol-index required call-site set" is a `Failure` naming the fixture — this covers RR41 (index-backed, negative assertion, passes VACUOUSLY today with no skip line — executed) and every future fixture by construction. RR45 uses RR44's no-cargo-`PATH` idiom (there is no path override in `build_symbol_index`, :139-161) and `self_test()` gains a fixture exclusion + its own selection flag so RR45's subprocess `--self-test` does not recurse (the `R12sweepmeta`-style exclusion).
**P3 (#517):** SOUND as written; the oracle is the tree-wide census, not the diff-scoped tripwire (`check_governance_tripwire` examines only pub items on this diff's added lines, :534-554): over the real `symbol-index` parse of `crates` (22 900 items, 6 838 pub) the noun table newly flags 0 (today's rule flags 29; `Retirement*` occurs as no indexed item). The self-test carries that census as a committed-literal floor (6838 pub items scanned, 0 new) recomputed in CI's container job where the index builds; the pair fixtures stay. Note in the gate's doc that a finding fails the step (`main()` returns 1).
**P4 (#507):** (i) the manifest keeps its `lanes` key (renaming breaks five out-of-scope readers: ci/scripts/prove_surface.py:111, crates/jammi-ai/tests/gpu_capability/capability_surface.rs:471/498/1140, ci/scripts/check_gpu_prove_once.py:284 + PROMOTION_TABLE, ci/scripts/check_gpu_prove_timings.py:601/736/757, ci/scripts/test_check_gpu_prove_once.py:348 — a rename buys no property); the three CPU families join `lanes` with `capabilities` ABSENT; `check_release_manifest.py`'s closed top-level key set (check 5) is unchanged; its capabilities rule becomes SYNTACTIC where it runs (the guard matrix has no toolchain): `capabilities` present iff the lane's declared `cargo_features` names `cuda` or `flash-attn`, forbidden otherwise, identical across every present block; the DERIVED closure stays in `check_flash_attn_closure.py` (toolchain-bearing), which skips lanes without `capabilities` by rule instead of `sys.exit(2)` (:234). (ii) Dockerfile: the CPU builder has NO ARG today (:71 is a hardcoded prefixed `--features` in its RUN); it gains a post-FROM `ARG CARGO_FEATURES` with NO default, a `: "${CARGO_FEATURES:?…}"` guard in the RUN (an empty build-arg otherwise builds successfully — the fail-open #507 names), and the `jammi-server/`-prefix `awk` transform the CUDA stage already does; the three CPU image jobs (.github/workflows/server-image.yml:151/:398/:627) pass it via `jq` from the manifest; the CUDA stage's ARG default is removed the same way. (iii) Verification (executed, not asserted): `ci/release-feature-manifest.json` joins pypi-server.yml's PR `paths` (:20-25); the LEAD dispatches release-binaries.yml build-only and server-image.yml's dispatchable CPU build on the branch and cites the run ids; the guard-refusal oracle is a hermetic `docker build --target <cpu-builder>` with the arg absent, run by the implementer locally in the CI image (or the exact Dockerfile snippet in a scratch image) — cite the output. (iv) The exposure test keeps a POSITIVE control per family: the site is found and reads its own family key from the manifest; the reasoned `NOT_SHIPPED` literals stay legal; a deleted/renamed site is a finding. The YAML sites are read by a REAL parser: a new `ci/scripts/check_release_lane_sites.py` (PyYAML compose nodes, guard matrix, self-test) owns the workflow half; `shipped_feature_exposure.rs` keeps the manifest + deny.toml halves and asserts the families from the manifest alone (P5 honoured; no Rust text scan over YAML).
**§3 stop rule** unchanged in shape; P4's fallback now names the executed dispatches as its precondition.

### 11.4 RELKEY — the unit contract as pressure-tested and folded

## RELKEY v2 — a training-set relation value that carries its own order; the mining W=1 pin (wave 6; #551; round-1 KILL folded)

Supersedes `relkey.md` in full. Implementer: ai-core (cross-scope for `crates/jammi-db/src/store/mod.rs` `TrainingSetTable`
and `crates/jammi-db/tests/it/materialization.rs`). Worktree `/Users/vijaychakilam/git/f-inverse/wt-relkey`, branch
`unit/relkey`, base `b8052978`. Targets `/Users/vijaychakilam/git/f-inverse/targets/relkey`. Rules: `RULES-implementer.md`.
Scope: `crates/jammi-db/src/store/mod.rs` (`TrainingSetTable` :288-360, its `sql_relation`, `registered_name`, `record`),
`crates/jammi-ai/src/fine_tune/{training_set.rs,stream.rs,worker.rs}`, `crates/jammi-ai/tests/it/{training_set.rs,
training_set_stream.rs,pinned_source_gate.rs,common/mod.rs}`, `crates/jammi-db/tests/it/materialization.rs`.

### 0. Premises (round 1, executed on b8052978 — binding; they replace the killed contract's §0)
- `pub struct RelationKey(String)` EXISTS (crates/jammi-db/src/store/mod.rs:380, shipped 7d4b9173 in wave 5): the workspace-general quoted
  relation newtype with two minters and nine general consumers interpolating it via `Display` with their own (or no)
  order. It is NOT training-set-specific and must not be changed.
- `training_set_order_by` is already in jammi-db (crates/jammi-db/src/store/mod.rs:442). `record` is `pub` (crates/jammi-db/src/store/mod.rs:288); its
  `table_name` is protocol-load-bearing (`TrainingSetIdentityPair::training_set_location`, crates/jammi-ai/src/fine_tune/worker.rs:3696, re-resolved
  at :6086 via `Catalog::get_result_table` → `pub table_name`), and the struct is constructed by literal from jammi-ai
  (crates/jammi-ai/src/fine_tune/worker.rs:6118-6125). The bare name is therefore reachable from the public catalog surface without
  `TrainingSetTable` at all — a type on `TrainingSetTable` cannot make the hand-built route unrepresentable.
- The spelling scan (`every_production_sql_relation_call_site_is_on_the_allow_list`, crates/jammi-ai/src/fine_tune/training_set.rs:358-364) is the
  only workspace-wide net over hand-built readers; today's reader class is closed at ONE production site
  (`read_back_sql`; crates/jammi-ai/src/fine_tune/stream.rs:401/779/914 compose on it). `registered_name` has zero production callers.
- The retained order test (crates/jammi-ai/tests/it/training_set.rs:962-982) string-slices the ordered SQL at `" ORDER BY "` to prove the
  order matters — a `String` return can always be sliced; "no unordered read is representable" is not a property a
  `String`-returning value can carry.
- `recompute_training_set` (crates/jammi-ai/src/pipeline/recompute.rs:533-592) re-MATERIALISES via the producer (which sorts at crates/jammi-db/src/store/mod.rs:4375);
  there is no replay reader and no order clause to drop. `crates/jammi-ai/tests/it/pinned_source_gate.rs:1630-1635` pins
  `registered_name` count = 1 and carries a `result_table_relation` entry (:1746).
- The mining oracle prints no constant (executed twice: zero output); the GradCache pin is a per-`target_os` pair of
  plain slices (crates/jammi-ai/src/fine_tune/training_set.rs:398-412); the hermetic `test` job runs `--workspace` on Linux in the CI image, so a
  `cfg!(target_os = "linux")` failing arm reds that lane. ci.yml's K4 note records heterogeneous CPU hardware within
  one (arch, os) pair.

### 1. Properties (binding)

**R1 — Every reader in jammi-ai's fine-tune path reaches the training-set relation only through a value that
renders the committed order from the table's OWN recorded columns.** A training-set-specific value,
`TrainingSetRelation`, minted only by `TrainingSetTable` (private field, no `Display`, no bare-name accessor), whose
only SQL conversion is `select_ordered(&self, projection) -> String` rendering `ORDER BY` from the columns the
producer recorded in `ProducingDescriptor::TrainingSet { columns, .. }` (crates/jammi-db/src/store/mod.rs:3053-3055 already recovers them
from the manifest) — never a caller-supplied key, so a present-but-wrong key is unrepresentable. `read_back_sql` and
stream.rs's composed reads consume it. `RelationKey` and its nine general consumers are untouched. Oracle:
`compile_fail,E0616` (private field) and `compile_fail,E0599` (no `as_str`/`Display`) doctests WITH the error code on
the fence (in-tree precedent crates/jammi-ai/src/fine_tune/collective/mod.rs:189) — the round measured that a bare fence passes on any error;
plus a positive test that `select_ordered`'s ORDER BY equals `training_set_order_by(record columns)`. The retained
row-group order test keeps its string-slice control (it proves the order is load-bearing; the property is about what
a reader can OBTAIN, not about what a test can slice). Mutation: expose `as_str` → doctest reds.

**R2 — The spelling scan stays and widens.** The allow-list scan keeps running; its needle set gains
`result_table_relation(` and `.record.table_name` / `.table_name()` (the catalog-record route), and its doc states the
honest guarantee: "every production reader that names the relation goes through `TrainingSetRelation` or is on this
list"; the hand-built/catalog-record route is a disclosed residual on #551, never claimed closed.

**R3 — `record` is private with named accessors and one constructor.** `TrainingSetTable::record` becomes private;
`table_name()` (the wire needs it), `parquet_path()`, `row_count()`, `kind()` accessors; `TrainingSetTable::from_record(..)`
for crates/jammi-ai/src/fine_tune/worker.rs:6118. `registered_name` is deleted (zero production callers); `pinned_source_gate.rs`'s
`SESSION_LITERAL_ALLOWED` entry for it is removed by rule and the `result_table_relation` entry re-keyed
(file, fn, ordinal, count — never line). Every out-of-crate test site listed in §0 is migrated. crates/jammi-ai/tests/it/training_set.rs:1029
(`a_result_table_cannot_be_a_fine_tune_source`) is reshaped to the accessors.

**R4 — The mining W=1 byte pin, captured honestly.** The mining oracle gains a `println!` of the mining-ON adapter
digest pair BEFORE its assert (a failing test's captured stdout is the lead's only read), under the GradCache shape:
`#[cfg(target_os = "linux")] const MINING_ADAPTER_PRINTS: &[(&str, &str)]` twin. On this branch the Linux arm is a
by-name failing assertion (`unimplemented pin: capture from CI` — `#[ignore]` forbidden); the macOS arm is pinned from
the implementer's two local runs (must agree). The LEAD commits the Linux constant only after it agrees across TWO CI
runs (the K4 heterogeneity note); if the two runs disagree, the pin adopts the K4 `JAMMI_REQUIRE_*`-gated shape and the
disagreement is recorded on #551. Mining-off stays the live control.

**R5 — Docs reflect the state.** training_set.rs's module doc describes `TrainingSetRelation` + the widened scan + the
disclosed residual; the "one exclusion" paragraph about `registered_name` is deleted. No edits to docs/plans history or
docs/rigor records.

### 2. Non-goals
No change to `RelationKey`, `TRAINING_SET_ORDER_RULE_V1`, the digest, the partition rule, or recompute.

### 3. Stop rule (pre-committed)
R1–R3 ship together or not at all; a second closing-audit block on them excises R1–R3 (record on #551) and R4 ships
alone. R4 has no fallback beyond its own two-run rule.

### 4. Round-1 record
KILL on the unit-as-written (six blocks: property refuted by its own retained test; catalog route; shared-type
mutation; `record` protocol-load-bearing; pinned_source_gate unscoped; R2 against a non-existent reader). Folded here.

### 11.5 RANGESPLIT — the unit contract as pressure-tested and folded

## RANGESPLIT — N-way inference below one merge, `_ordinal`-keyed (wave 7; implementation contract; #540)

Two design rounds executed (rangesplit-design.md → REFINE; rangesplit-design-v2.md → REFINE, every fold executed; 240-cell
plan grid, execution equivalence, residency, deadlock arms; probes at /Users/vijaychakilam/git/f-inverse/targets/
pt-rangesplit: src/bin/{v2plan,v2exec,estruct}.rs, v2plan.txt, v2exec.txt, estruct.txt). This is the implementation
contract; §0 is the executed record. Implementer: ai-core (cross-crate scope: `crates/jammi-ballista/src/codec.rs` +
`plan.proto` + its round-trip tests + `crates/jammi-ballista/tests/distributed/main.rs:53` doc, `crates/jammi-db/src/inference/**` if the
schema helper lives there). Worktree `/Users/vijaychakilam/git/f-inverse/wt-rangesplit`, branch `unit/rangesplit`, base
`b8052978` (rebased onto the wave-6 tip at cut). Targets `/Users/vijaychakilam/git/f-inverse/targets/rangesplit`.
Rules: `RULES-implementer.md`.

### 0. Executed record (binding premises)
- `InferenceSession::annotate_plan` (crates/jammi-ai/src/session.rs:1018-1048) builds `InferenceExec` DIRECTLY on the scan — no
  `ordered_input`, no sort below. The three `ordered_input` sites are crates/jammi-ai/src/session.rs:1475, crates/jammi-ai/src/pipeline/embedding.rs:98,
  crates/jammi-ai/src/pipeline/embedding_refresh.rs:937 (the materialising paths). The four InferenceExec roots: crates/jammi-ai/src/session.rs:1565,
  crates/jammi-ai/src/pipeline/embedding.rs:263, crates/jammi-ai/src/pipeline/embedding_refresh.rs:966, crates/jammi-ai/src/query/builder.rs:306.
- The declaration set (split: `required_input_distribution = SinglePartition`; InferenceExec: `UnspecifiedDistribution`,
  `benefits_from_input_partitioning = [false]`, `output_partitioning` = the child's N, `execute(partition)` forwarding)
  keeps EnforceDistribution from inserting anything BETWEEN split and inference in 240/240 optimized cells on the REAL
  annotate shape (it inserts one CoalescePartitionsExec BELOW the split, which also repairs today's silent N-1
  partition drop on that path).
- A merge keyed `[_row_id, _ordinal]` DIVERGES from the 1-partition sequence on unsorted (annotate) input (row 1 at N=4)
  and its published `[_row_id ASC]` ordering is FALSE there — DataFusion deleted the user's ORDER BY SortExec on the
  strength of it and the collect DEADLOCKED. A merge keyed `[_ordinal]` alone is identical at every N on both input
  shapes, drains partitions in fan-out order (peak 2N+1 batches under a bounded split), and the optimizer keeps a real
  SortExec for a user ORDER BY. `SanityCheckPlan` rejects an SPM whose child does not publish the merge key.
- Literal byte equality is unachievable: `_latency_ms` (crates/jammi-ai/src/inference/schema.rs:33) is a per-sub-batch wall-clock value;
  the merge re-batches.
- `_ordinal` is produced by the runner today (crates/jammi-ai/src/inference/runner.rs:188 `next_ordinal` → crates/jammi-ai/src/inference/schema.rs:159 `ordinal_start`) at prefix
  index 1 of every output batch; `passthrough` columns are appended at the END (crates/jammi-ai/src/inference/schema.rs:45-50). The canonical
  read-back is `ORDER BY _row_id, _ordinal` (crates/jammi-ai/src/session.rs:2307-2311).
- The optimizer removes the provider's SPM in 48/240 cells (ORDER BY / GROUP BY parents): the merge is an invariant of
  the four hand-built roots only, never of an optimized plan.
- `build_embedding_plan`'s plan is submitted over the wire by the committed distributed oracle
  (crates/jammi-ballista/tests/distributed/main.rs:260, 788, 976); `JammiCodec::try_encode` (crates/jammi-ballista/src/codec.rs:164-186) has no arm for
  a new node ("Unsupported plan node").
- N partitions run N concurrent `LoadedModel::forward(&self)` on one `Arc<LoadedModel>` (crates/jammi-ai/src/model/mod.rs:567); device
  reservation is taken at LOAD (`GpuPermit`), not per forward; the runner's OOM-halving recovers per partition.
- Struct/Map keys fail the Utf8 cast and the fallback surfaces a misattributed schema-shape error naming `_row_id`
  (annotate path only: no KeyCheckExec, no `safe:false` cast); a `List<Int32>` key casts SUCCESSFULLY in arrow 58.3 —
  the refusal keys on "the cast returned Err", never a type list.
- An SPM root re-batches: `ResultSink::batch_num` and the catalog `checkpoint` column (crates/jammi-ai/src/pipeline/result_sink.rs:74-96,
  crates/jammi-db/src/store/building.rs:190, crates/jammi-db/src/catalog/result_repo.rs:1084/1862) count merged batches after this change.

### 1. Properties (binding; each with an executed oracle and mutation)
**RS1 — Nothing between the split and the operator on the optimized UDTF plan.** `OrdinalSplitExec` + the declaration set
above; oracle: the 240-cell grid ({PIPE, UDTF} × {bare, GROUP BY, ORDER BY, WHERE, LIMIT} × N ∈ {1,2,4} ×
target_partitions ∈ {1,2,4,8}) asserts no RepartitionExec/SortExec/CoalescePartitionsExec between split and
InferenceExec; mutation: drop `benefits_from_input_partitioning=[false]` → cells red.

**RS2 — Row-sequence equivalence.** For N ∈ {1,2,4}, the row SEQUENCE over every column but `_latency_ms` is identical to
N=1, on BOTH input shapes (sorted materialising path; unsorted annotate path), on a fixture with duplicate `_row_id`s.
Mechanism: the split assigns a GLOBAL `_ordinal` before fan-out (single writer; the runner's `next_ordinal`/
`ordinal_start` producer moves to the split — the split's OUTPUT schema carries `_ordinal` as a named input column the
operator consumes and re-emits at prefix index 1, never via `passthrough`); InferenceExec publishes `[_ordinal ASC]`
ONLY (true by construction) and `maintains_input_order = true`; the four in-process roots wrap `SortPreservingMergeExec
([_ordinal ASC])`. Mutation: merge on `[_row_id, _ordinal]` → the unsorted-shape oracle diverges.

**RS3 — Bounded residency AND liveness.** Under the merge's drive with a bounded split channel (cap 2), peak in-flight
batches ≤ 2N+1 (measured, asserted); a 10 s liveness timeout on every equivalence arm; the backpressure/cancellation
protocol is stated: a consumer that stops polling one partition (LIMIT, error) must not wedge the driver — the split's
sender detects a dropped receiver and drops that partition's batches (oracle: `LIMIT 1` over N=4 completes).

**RS4 — The key cast is a typed refusal.** `crates/jammi-ai/src/inference/schema.rs:148`'s fallback becomes `Err(JammiError::Inference)` naming
the key column, its type and "cannot be cast to Utf8" whenever `cast(...)` returns Err (never a type list); fixture: a
Struct-typed key through `annotate()` — RED at base with the misattributed message; the pipeline paths' earlier
`safe:false` cast site is named as the second refusal site.

**RS5 — The four roots, count-keyed.** A source oracle asserts the four InferenceExec roots (file+fn) each execute a
`SortPreservingMergeExec` root; `AnnSearchExec`/`GangExec` keep `UnknownPartitioning(1)` with the doc sentence why.

**RS6 — The wire.** `OrdinalSplitExec` gets a `NodeTag`, a `plan.proto` message, encode/decode arms in `JammiCodec`, a
round-trip test, and `crates/jammi-ballista/tests/distributed/main.rs:53`'s documented plan shape is re-anchored; the distributed oracle's
submitted plan round-trips (the split's fan-out is process-local: the stage/task meaning of an N-partition InferenceExec
on an executor is stated in the codec doc).

**RS7 — Forward concurrency is admitted, not assumed.** In-flight `forward` calls across partitions are bounded by a
per-device permit (CPU: N ≤ available parallelism; GPU: the device scheduler's admission — default 1 concurrent forward
per device unless the scheduler grants more), so peak activation memory never scales with N unadmitted; the OOM-halving
recovery is per partition and stated. Oracle: a counter of concurrent forwards never exceeds the permit under N=4 (CPU).
Speedup measured on CPU at N=4 on the bench corpus and reported; the GPU statement is "N=1 concurrent forward by
default" with the admission seam named.

**RS8 — Checkpoint semantics named.** `ResultSink::batch_num`/`checkpoint` count merged batches; the doc states it and the
resume path is exercised across the change (an existing resume test stays green, named).

### 2. Non-goals
No contiguous key ranges; no change to the ANN sidecar; the must-not-touch list from #540 is binding (job row,
`finish_job_with_model` CAS, `PublishedPrefix`, `record_failed`, `put_resume_checkpoint`).

### 3. Stop rule (pre-committed)
A closing-audit block on RS6 (wire) ships RS1–RS5 + RS7–RS8 with the split refused for wire submission (typed
"not distributable in v1", mirroring `MaskExec`) and RS6 named on #540; a second block on RS2/RS3 excises the unit.

### 11.6 RENDEZVOUS — the unit contract as pressure-tested and folded

## RENDEZVOUS — plan 68 DIST unit 2: `RendezvousPlacement` over the live instances ring (wave 7; implementation contract)

Two design rounds executed (contracts/rendezvous-design.md → REFINE; rendezvous-design-v2.md → REFINE with every
correction executed and quoted). Under the two-round rule this is the implementation contract; §0 is the executed record.
Implementer: db (cross-crate scope granted for `crates/jammi-ai/src/session.rs`, `crates/jammi-server/src/runtime.rs`,
`crates/jammi-server/tests/it/peer_placement.rs`, `crates/jammi-ai/tests/distributed/**`, `docs/guide`). Worktree
`/Users/vijaychakilam/git/f-inverse/wt-rendezvous`, branch `unit/rendezvous`, base `b8052978` (rebased onto the wave-6
tip at cut). Targets `/Users/vijaychakilam/git/f-inverse/targets/rendezvous`. Rules: `RULES-implementer.md`.

### 0. Executed record (binding premises)
- `SegmentPlacement::owners` is called per segment in a loop (crates/jammi-db/src/store/mod.rs:2600-2607) and the result is captured into
  `SegmentSource::Remote`, so phase 2 already reuses phase 1's snapshot; one `plan()` above the loop makes the whole
  two-phase query see one ring. The seam reshape `plan(&self, table, segments: &[SegmentId]) -> Result<Vec<Vec<PeerAddr>>>`
  was applied in a probe and `cargo check --workspace --all-targets` passed (3 files, +58/−25).
- `list_gang_members` (crates/jammi-db/src/catalog/jobs_repo.rs:2775-2843): SQL = `instances JOIN workers … WHERE peer_addr IS NOT NULL AND
  result_root_identity = $n AND NOT (stale)`; `state='claiming'` + kinds are filtered in RUST (:2817-2830). The
  extractable shared fragment is the whole WHERE clause. Root identity (migration 036) is the comparison; crates/jammi-ai/src/session.rs:151-155
  carries a stale doc saying otherwise.
- `instances.peer_addr` is written ONLY when `peer_advertise` is set (crates/jammi-db/src/catalog/instance.rs:461-464, 579-585; crates/jammi-db/src/config/mod.rs:1788-1796);
  PeerService and GangService share the `peer_bind` listener (crates/jammi-server/src/runtime.rs:571-580, 636-643); `MembershipConfig::validate`
  already requires `peer_bind` when `peer_advertise` is set. There is NO separable retrieval key.
- Stamps are canonical fixed-width UTC (`LEASE_TS_FORMAT`, crates/jammi-db/src/catalog/lease.rs:47; CHECK constraint crates/jammi-db/src/catalog/schema.rs:1893-1895 + SQLite
  triggers): lexical order == chronological (0 disagreeing ranks over 200k rows). `stale_before_clause`'s cast form is
  a Seq Scan (17.9 ms at 200k); the DB-side canonical cutoff `last_seen_at >= to_char((now() - make_interval(secs => $1))
  AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')` is an Index Scan (0.040 ms), row-set identical, one bind, DATABASE
  clock preserved. crates/jammi-db/src/catalog/lease.rs:829-843 pins the clause STRING (mechanism) and must be rewritten to pin the property.
  The same rewrite makes `list_gang_members` 87× cheaper with an identical row set.
- `content_hash_row` hardcodes `CONTENT_HASH_DOMAIN` and takes `ContentValue` (crates/jammi-db/src/store/content_hash.rs:37, 83-85). HRW over
  the real fold (100 segments, N=3→2: moved 31/100, none not owned by the leaver; N=3→4: 32 moved, all to the newcomer;
  plan invariant under ring order) — minimal disruption holds.
- `/metrics` on the health listener (crates/jammi-server/src/lib.rs:84); `jammi_peer_requests_total{rpc}` already counts every
  PeerService call (crates/jammi-server/src/metrics_layer.rs:52-55, 118); the lane spawns real `jammi-server` processes with per-worker health
  ports (crates/jammi-ai/tests/distributed/harness.rs:374, 486-492) from ONE shared `worker_toml`; `SearchRequest` carries `query_vector` (no model needed).
- `ServerConfig::validate` is called only by `OssServer::new` (crates/jammi-server/src/runtime.rs:459-462), not by `load_from`;
  `MembershipConfig::validate` is called by `load_from` and by `InstanceRegistration::from_config` — the membership
  choke point.

### 1. Properties (binding; each with an executed oracle and mutation)
**RV1 — One ring per query.** `SegmentPlacement::plan(&self, table, segments) -> Result<Vec<Vec<PeerAddr>>>` (arity
invariant `len == segments.len()` checked; a catalog read failure is an `Err`, never silently all-local) replaces
`owners`; `resolve_search_mode` calls it once above the loop; `AllLocal`/`StaticPlacement`/`TestPlacement` migrate.
Oracle: the plan for a fixed ring is a pure function (table-driven, expected HRW order computed by an INDEPENDENT
second implementation in the test); mutation: reorder the ring → identical plan.

**RV2 — Ring predicate (placement-generic).** `RendezvousPlacement { catalog, self_instance_id, margin }`: ring = `instances`
rows with `peer_addr IS NOT NULL`, live under `instance_liveness_margin`, `result_root_identity` == self's, **self
INCLUDED** (its own row by `instance_id`); no `workers` join, no job-kind vocabulary. The liveness + root conjuncts are ONE
shared SQL fragment used by both this predicate and `list_gang_members` (one definition of "live with my root"; the
gang predicate keeps its `workers` join and Rust filters). Arms: `first == self → local (empty)`; else `[first, second]`;
**empty ring or self-row absent → all-local, counted** (`jammi_placement_ring_empty_total` / a typed metric, never
silent). Oracle: a stale row is excluded; a root-identity-mismatched row is excluded and counted; a `[worker]
enabled=false` advertising replica IS a member; a process without `peer_advertise` is NOT; self appears as a candidate.

**RV3 — Sargable liveness on both backends, database clock.** `stale_before_clause` (Postgres arm) renders the cutoff
DB-side via `pg_canonical_stamp` and compares TEXT `>=` (Index Scan on `idx_instances_seen`); SQLite arm equivalent; the
pinned test at crates/jammi-db/src/catalog/lease.rs:829-843 is rewritten to the PROPERTY (one bind = margin seconds; database clock on Postgres;
`EXPLAIN` shows the index; row-set identity vs the cast form over a fixture at one instant incl. the exact-microsecond
boundary). Every caller (jobs reclaim, instance sweep, retention) keeps its row set — a differential oracle over each.

**RV4 — Explicit opt-in.** `[server] placement = "local" | "rendezvous"` (default `local`); `rendezvous` requires
`peer_advertise`, refused typed at the MEMBERSHIP choke point (`MembershipConfig::validate`), so struct-literal configs
and the embedded path cannot skip it. `peer_advertise`/`peer_bind` semantics untouched. Config doc + upgrade note: no
existing deployment changes behaviour. Oracle: a `rendezvous` config without `peer_advertise` is refused by name from
`load_from` AND from `from_config`.

**RV5 — Hash.** Extract `domain_hash(domain: &[u8], parts: &[&[u8]]) -> [u8; 32]` (the length-prefixed, domain-separated
SHA-256 fold) as the primitive; `content_hash_row` calls it with `CONTENT_HASH_DOMAIN` (byte-identical output — an oracle
over the existing content-hash goldens) and placement calls it with `b"jammi.placement.v1"` over
`(instance_id, table, segment_id)`; score = big-endian u64 of bytes 0..8; tie-break by `instance_id` byte order.
Minimal-disruption oracle: 100 segments, N=3→2 moved ≈ 1/N and every moved segment was owned by the leaver; N=3→4 every
moved segment goes to the newcomer.

**RV6 — Cost stated.** One ring read per placed search, a single statement (no READ ONLY transaction wrapper unless
measured necessary); the per-search cost measured (ms at 100 and 10k instance rows on the scratch Postgres) and stated
in the doc as a property with a budget (≤ 2 ms at 10k rows on the scratch host, or the measured number with the reason).

**RV7 — Lane proof.** `crates/jammi-ai/tests/distributed/` gains a placed-search leg: a per-test placement knob on
`spawn_worker`/`WorkerPorts` (the shared `worker_toml` stays `local` for the existing job-shaped tests), a shared-MinIO
embedded table, a `Search` client fixture, the owner-side observable read from the OTHER worker's `/metrics`
`jammi_peer_requests_total{rpc="SegmentSearch"}` (no new counter), K4 byte parity placed-vs-local, and the D8 ladder's
`retry_ok` + `UNAVAILABLE` arms with one process killed.

**RV8 — No new task/thread** anywhere in the diff (`tokio::spawn`, `spawn_blocking`, `std::thread::spawn`,
`spawn_local`): source oracle over the whole diff.

**RV9 — Docs.** crates/jammi-ai/src/session.rs:151-155's stale sentence corrected; `security.md` states the delta (the owner set becomes
self-asserted through the catalog under trust class I-PEER); the deploy guide's placement section; plan 68
`DIST-DATA-PLANE.md` §9 updated to "shipped"; `PROGRAM.md` "later" row updated.

### 2. Non-goals
No DIST-3 (`datafusion-distributed`), no change to the D8 ladder, no background membership loop (D5), no change to
`peer_advertise`/`peer_bind` semantics.

### 3. Stop rule (pre-committed)
A closing-audit block on RV7 (lane) ships RV1–RV6 + RV8–RV9 with the lane leg as a follow-up named on the plan's §10;
a second block on RV1–RV5 excises the unit (record on the plan's §10 and a new issue).

### 11.7 GRAPHARM — the unit contract as pressure-tested and folded

## GRAPHARM — the graph fine-tune arm through a materialised training set (wave 7; implementation contract; #538)

Two design rounds executed (grapharm-design.md → REFINE; grapharm-design-v2.md → REFINE, root survived, four local blocks
with fixes executed). This is the implementation contract; §0 is the executed record. Implementer: ai-core (cross-crate
scope: `crates/jammi-db/src/store/{manifest.rs,mod.rs}` materialisation + descriptor, `crates/jammi-server/tests/it/gang_
coordinator.rs`, docs/plans). Worktree `/Users/vijaychakilam/git/f-inverse/wt-grapharm`, branch `unit/grapharm`, base
`b8052978` (rebased onto the wave-6 tip at cut — RELKEY also edits training_set.rs/TrainingSetTable). Targets
`/Users/vijaychakilam/git/f-inverse/targets/grapharm`. Rules: `RULES-implementer.md`. Harness with executed probes:
/Users/vijaychakilam/git/f-inverse/targets/pt-grapharm/{harness,dfprobe} (E1–E12, P1–P6) — reuse them as fixtures' seeds.

### 0. Executed record (binding premises)
- Sample is a function of input ROW ORDER today (E1: 290 vs 303 rows across layouts); nodes/edges are read with no ORDER BY
  (crates/jammi-ai/src/fine_tune/worker.rs:3861-3866, :3894-3899). Full-tuple order on both scans makes two layouts sample identically (E2); a duplicate
  NODE id is NOT benign (E3: the sampler keeps the last text); duplicate EDGES are order-benign but output-affecting vs a
  de-duplicated set (E12) — they must be KEPT.
- The training-set write sorts by the full projected tuple (crates/jammi-db/src/store/mod.rs:4368-4380) and nothing shuffles, so a leading
  `_ordinal` column reproduces the emission order under `full_tuple_v1` (E8', P2) and the read-back re-applies it
  (crates/jammi-ai/src/fine_tune/training_set.rs:69-80); the decoder is by name (crates/jammi-ai/src/fine_tune/decode.rs:376-395, 494-540) so `_ordinal` is inert; batch composition
  is unchanged (E9).
- The producer is stream-shaped: a nameless one-shot `PartitionStream`/`StreamingTable` through `ctx.read_table(..)
  .select(..).sort(..).create_physical_plan()` plans and executes exactly like `plan_training_set_rows` (P1/P2), single
  partition, no registration (the fine_tune/** zero-hit gate crates/jammi-ai/tests/it/pinned_source_gate.rs:3015 stays green).
- `replay_descriptor` has no wildcard arm (crates/jammi-ai/src/pipeline/recompute.rs:192-249; K1 = compiler exhaustiveness); the manifest enum is
  internally tagged (crates/jammi-db/src/store/manifest.rs:411-413): no MANIFEST_VERSION bump, no existing hash moves. GA2's mechanism is the
  in-tree idiom: exhaustive destructuring with no `..` (crates/jammi-db/src/store/manifest.rs:2666-2679, 2802-2810) — NOT a field-count derive
  (strum's VariantArray is enum-only).
- node2vec needs random access (`biased_walk` over `out_adj`, `undirected.neighbours`, uniform negatives over `node_ids`):
  the resident set is O(|V|+|E|); the pool is `InferenceSession::memory_pool()` (crates/jammi-ai/src/session.rs:654-657 → a bounded
  GreedyMemoryPool, jammi-db crates/jammi-db/src/session.rs:210); the in-tree idiom is a NAMED consumer per site (`training_set_eager`,
  crates/jammi-ai/tests/it/training_set.rs:263-266; `training_set_stream[..]`, crates/jammi-ai/src/fine_tune/stream.rs:405-411). A full-tuple SortExec buffers the whole output
  before emitting (P6: 4.25 MB reserved before the first row at 20k rows) — the pool's peak is not the oracle.
- Empty negative pool: on a star graph 3 of 23 rows carry no negative (E10); today `from_graph` demotes the whole set to
  Pairs from `pairs.first()` (crates/jammi-ai/src/fine_tune/data.rs:528-533) and a Triplet set raises a typed error for such a row (crates/jammi-ai/src/fine_tune/data.rs:534-541);
  the shared decoder reads a NULL text cell as "" (crates/jammi-ai/src/fine_tune/decode.rs:118-129; P5) at BOTH decode sites (:494-540 eager,
  :929-931 stream) and `detect_training_format` keys on column presence.
- The Peer refusal is TWO sites: the coordinator (crates/jammi-ai/src/fine_tune/worker.rs:3799-3810) and the member body (crates/jammi-ai/src/fine_tune/worker.rs:6706-6724), and the
  member needs `columns`/`task` for `detect_training_format`/`bind_training_source` (:6775-6783), which
  `TrainingSpec::GraphFineTune` does not carry.
- Reclaim shape for the two-attempts-one-job-id oracle: `loser_prefix_is_never_the_committed_artifact` (crates/jammi-ai/tests/it/fine_tune.rs:2337)
  / `configured_short_lease_drives_reclaim` (:2220); per-call unique table names pinned by crates/jammi-db/tests/it/materialization.rs:1613.
- Citations: unpinned-reuse oracle crates/jammi-db/tests/it/materialization.rs:1642; K4 assertion jammi-server crates/jammi-server/tests/it/gang_coordinator.rs:587-593;
  W=1 in-memory reference jammi-ai crates/jammi-ai/tests/it/gang_coordinator.rs:622-631; `EmptyTrainingSet { source_query }` at crates/jammi-db/src/store/mod.rs:4225-4231.

### 1. Properties (binding; each RED at base or under its named mutation; mutation executed)
**GA1 — Sample is a function of the input SET.** `GRAPH_READ_ORDER_RULE_V1` = full projected tuple ascending, NULLs first,
on both scans; duplicate node id → typed refusal at build (names the id); duplicate edges kept (doc states the asymmetry).
Oracle: two layouts → identical bytes; duplicate-id fixture → refusal; mutation: drop the edge ORDER BY → bytes differ.

**GA2 — Identity is complete.** `ProducingDescriptor::GraphTrainingSet { node_source, edge_source, id_column, text_column,
src_column, dst_column, task, format, sample: { seed, walk_length, walks_per_node, return_p_bits: u64, in_out_q_bits: u64,
hard_negatives, exclude_hops }, read_order_rule }`; `min_negatives` is NOT in the descriptor (E5: non-output-affecting) and
the doc says why; hash-mutation tests per field via exhaustive destructuring with no `..`; a `replay_descriptor` arm
(compile-forced). Mutation: change one sample knob → different hash (table over every field).

**GA3 — Format is decided from the config, and an empty negative pool is a typed refusal at SAMPLE time.** `format =
graph_triplet iff hard_negatives > 0`; under `graph_triplet` every emitted anchor carries ≥ 1 negative or sampling refuses
typed naming the anchor (no NULL negative is ever written); defense at BOTH decode sites: a NULL `negative` cell is a
typed refusal, never "" (RED at base: P5 reads ""). Oracle: the E6 star with `hard_negatives=2` → refusal; the same star
with `hard_negatives=0` → `graph_pairs` and trains. A1: the W=1 parity fixture is one on which today's `pairs.first()` and
the config decision AGREE; the flipped class (star) has its own before/after assertion.

**GA4 — Emission order is recorded and reproduced.** Leading `_ordinal: u64`; the Batches arm does NOT re-impose the sort
(the stream is in ordinal order by construction; a per-batch sortedness assertion replaces the SortExec); the read-back
order == emission order (E8'/P2 as a test). W=1 byte pin: attempted against main's in-memory arm on the agreeing fixture;
if bytes differ for a measured reason, a NEW per-OS pin at this base with the reason recorded.

**GA5 — Input seam.** `TrainingSetInput::{Sql(..), Batches { schema, stream }}` through the same `materialize_training_set`;
nameless provider; refusals render honestly (`EmptyTrainingSet` and the single-partition breach name the node/edge source
ids, never a fabricated SQL string — A2). Oracle: pinned_source_gate's zero-hit registration test stays green; the empty
graph → typed `EmptyTrainingSet` naming the sources.

**GA6 — Two attempts of one job id never displace each other** (reclaim shape from crates/jammi-ai/tests/it/fine_tune.rs:2337): both attempts
materialise their own row; the loser's prefix is never the committed artifact.

**GA7 — Residency.** A NAMED `MemoryConsumer` (`training_set_graph_sample`) reserves the adjacency + node-text bytes for
the sampler's lifetime; the oracle asserts THAT consumer's reservation ≥ the measured resident bytes and that no
`Vec<SampledPair>` of the whole output exists (source oracle over the emit path). Mutation: skip the reservation → red.

**GA8 — Peer gang.** BOTH refusal sites deleted; `TrainingSpec::GraphFineTune` carries `task` and the member binds the
training source from the MATERIALISED TABLE (the descriptor's recorded columns; `detect_training_format` on the table),
so the member needs no column vocabulary from the spec; K4 oracle gains a graph case (Peer W=2 == Local W=2 rank-0
bytes) through the streamed loader.

**GA9 — Recompute arm** re-samples from the descriptor, re-anchoring both relations (crates/jammi-ai/src/pipeline/recompute.rs:548-571) — byte parity.

**GA10 — Docs:** DESIGN.md §8 / README "NOT in v1" sentences for #538 replaced; `graph_fine_tune` docs state the table
path, the order rule, the empty-pool refusal; `training_set.rs` module doc gains the graph arm.

### 2. Non-goals
No change to the tabular arm's order rule, digest, or partition rule; no change to node2vec's sampling semantics beyond
the ordered read and the empty-pool refusal.

### 3. Stop rule (pre-committed)
A closing-audit block on GA8 (the member seam) ships GA1–GA7 + GA9–GA10 with the Peer refusal kept and #538 narrowed to
the W>1 gang; a second block on GA1–GA4 excises the unit (record on #538).

### 11.8 LOADER — the unit contract as pressure-tested and folded

## LOADER — the cu12 tarball's self-sufficiency proven under a jail, in the release lane (wave 6; #534 chroot half)

Implementer: docs-ci. Worktree `/Users/vijaychakilam/git/f-inverse/wt-loader`, branch `unit/loader`, base `b8052978`.
Targets `/Users/vijaychakilam/git/f-inverse/targets/loader`. Rules: `RULES-implementer.md`.
Scope: `ci/scripts/bundle_cuda_libs.sh`, `ci/scripts/test_bundle_cuda_libs.sh`, `ci/scripts/fixtures/cu12_loader_*`,
`.github/workflows/release-binaries.yml` (the `server-cu12-build` job ONLY — GATES edits the CPU tarball job in the
same file; keep your hunk inside the cu12 job), `packaging/server-cu12/**` docs.

### 0. Premises (verified 2026-09-17)
- Wave 5 shipped arm (1a): the release lane runs the REAL `ldd` with `LD_LIBRARY_PATH=<lib>` in the CUDA container as
  root and refuses any non-platform, non-driver member resolved from OUTSIDE `<lib>` (detection). The stop rule fired
  on the `unshare --mount` mechanism because the job has no `--privileged`/`--cap-add`.
- The lead probed the default Docker capability set on 2026-09-17: in `docker run --rm ubuntu:24.04`, `chroot / /bin/true`
  SUCCEEDS (CAP_SYS_CHROOT is in Docker's default bounding set) and `unshare -Urm` is DENIED. The GitHub-hosted
  runner's `container:` jobs run with Docker's defaults. So a plain `chroot` jail needs no privilege the lane lacks.
  Reference: Docker Engine security docs, "Runtime privilege and Linux capabilities" (default capability list includes
  SYS_CHROOT); glibc `ld.so(8)` `--library-path`/`--list`.

### 1. Properties (binding)

**L1 — Every bundle-able member resolves from `<lib>` and nothing else, proven with host copies hidden, in the
release lane.** A jail directory is built from NOTHING but: the staged `<lib>/`, the real `jammi-server` binary, the
platform loader (`ld-linux-x86-64.so.2`, copied from the binary's `PT_INTERP`) and the platform members the
manifest already classifies as host-provided (glibc's own: `libc`, `libm`, `libpthread`, `libdl`, `librt`, `libgcc_s`,
`libstdc++` — the CLOSED list the script already carries as the platform floor), and the driver members
(`libcuda.so.1`, `libnvidia-ml.so.1`) are DELIBERATELY ABSENT. Then `chroot <jail> /ld.so --list /jammi-server`
(the loader invoked explicitly, `--library-path /lib`) runs as root in the CUDA container. Rule: every bundle-able
member MUST resolve to a path under `/lib`; every driver member MUST be `not found` (proof they are not silently
bundled or host-satisfied); every platform member MUST resolve from the jail's platform copies; any other `not
found` is a FAIL by name. The report is parsed by the SAME `bundle_parse_loader_report` the hermetic suite already
tests. Oracle (hermetic, `test_bundle_cuda_libs.sh`): the jail builder on a fixture stage + fixture ELF-free inputs
is not executable on macOS — so the hermetic suite tests (a) the jail-report RULE over committed report fixtures:
a real jail report captured from the lane (the implementer adds a `workflow_dispatch` build-only capture exactly as
wave 5 did for `cu12_loader_report_real.txt`; the LEAD dispatches it and commits the fixture), a mutated copy with a
bundle-able member `not found` (FAIL), a copy with a driver member RESOLVED (FAIL — a bundled driver is a defect),
a copy where a member resolves from `/usr/lib` (FAIL, impossible in the jail but the rule still refuses it);
(b) the jail BUILDER's file-set rule as a pure function over a fixture listing: the jail contains exactly the
allowed set, nothing from the host (mutation: add `/usr/lib/libnccl.so.2` to the jail → FAIL).

**L2 — The lane fails closed when the jail cannot be entered.** If `chroot` itself fails (EPERM), the step FAILS
naming the missing capability; it never falls back to the detection-only arm silently (the detection arm 1a keeps
running FIRST as its own step — both arms are required; state in the workflow why each exists).

**L3 — The rule text and the fixture list are derived from the binary's `DT_NEEDED`, never typed.** As today's
rule (2): the required-resolved set is the binary's own `DT_NEEDED` list (measured in the lane by `readelf -d`,
committed as the fixture `BINARY_NEEDED`), so a new dependency cannot be forgotten.

**L4 — Docs reflect current state.** `bundle_cuda_libs.sh`'s long comment block about the excised arm is rewritten
to describe the two arms that exist (detection + jail); #534's "OPEN chroot half" sentence is deleted; the
packaging README states what the tarball guarantees and how it is proven.

### 2. Non-goals
No change to the staging derivation (`DT_NEEDED` walk + floor). No `unshare`. No change to the CPU tarball job.

### 3. Stop rule (pre-committed)
If the jail cannot run the loader in the CUDA image for a reason the lane cannot fix (e.g. the loader refuses
`--list` under chroot for a reason measured in the lane log), L1 ships as the hermetic RULE + a lane step that
captures the jail report and uploads it as an artifact WITHOUT enforcing, the reason is quoted on #534 with the run
id, and the enforcement is the next unit. One attempt of that shape only.

### 4. Pressure-round folds
(appended by the lead after the pressure-tester's verdict)

### 4. Pressure-round folds (2026-09-17, REFINE — executed on glibc 2.28 (ubi8, the lane's manylinux_2_28 family) and 2.39; BINDING, they override §1 where they differ; probe scripts at /Users/vijaychakilam/git/f-inverse/targets/pt-loader/{jail*.sh,jail8.py,parse_demo2.sh})
F1 — `ld.so --list` is FATAL on the first missing library (rc=127, one error line, no report) — the "drivers must be `not found`" proof is unobtainable with `--list`, and a driver miss masks a bundle-able miss. Mechanism: tolerant trace mode INSIDE the jail — a python driver (python3 is already a hard dependency of these scripts): `fork` → `os.chroot(jail)` → `os.environ["LD_TRACE_LOADED_OBJECTS"]="1"` → `os.execve("/lib64/ld-linux-x86-64.so.2", [..., "--library-path", "/lib", "/jammi-server"])`; measured: `=> not found` for absent members, `/lib/…` for the rest, loader self-line in the fixture shape, exit 0. No `env` binary in the jail; no extra file.
F2 — The loader must sit in the jail AT its PT_INTERP path (`/lib64/ld-linux-x86-64.so.2`) and be invoked there; copied to `/ld.so` its self-line reads `/lib64/ld-linux-x86-64.so.2 => /ld.so (…)` and the shipped `bundle_verify_loader_resolution` FAILS a correct jail (defect (3) mirrored). Measured: at the PT_INTERP path the self-line is the no-`=>` fixture shape and the shipped rule passes.
F3 — The platform set is DERIVED by calling `bundle_is_platform_soname` (the script's list also carries `libmvec.so.*` and `ld-linux-*`; `libmvec.so.1` is a measured DT_NEEDED member), never restated. The driver set is DERIVED: `{n ∈ measured DT_NEEDED : bundle_is_driver_soname n}` (`libnvidia-ml` appears in no artifact; `packaging/server-cu12/verify_link_set.py:71` names libcuda, libnvidia-ptxjitcompiler, libnvidia-nvvm).
F4 — L2 becomes a POSITIVE property: the report is accepted only if EVERY resolved path lies inside the jail (`/lib/…` or the in-jail loader path); any host path is a FAIL. (Setting `LD_TRACE_LOADED_OBJECTS` on the `chroot` command line traces `chroot`'s own loader on the HOST and exits 0 — a silent vacuous pass; exit codes 125/127 do not separate EPERM from a jail-construction bug from a missing library.)
F5 — Quantifier: "bundle-able member" = `DT_NEEDED ∩ non-host` (the loader-reachable set). `libnvrtc-builtins.so.12.6` is staged by the floor only, dlopen'd, never in any report — its self-sufficiency stays UNPROVEN by this arm and the doc says so.
F6 — The jail's FILE set = the binary + the staged `<lib>` (hardlinked, `cp -al`, on the container's own filesystem, not the bind-mounted workspace — the stage is multi-GB) + the platform closure of the binary AND every staged object (derived via readelf over all of them) + the loader at its PT_INTERP path; drivers absent. `(report, BINARY_NEEDED)` are captured in the SAME lane run and committed together.
F7 — The jail rule is a NEW named predicate (`bundle_verify_jail_report`): bundle-able ⇒ resolved under `/lib`; driver ⇒ `not found`; platform ⇒ resolved from the jail's platform copies; any other `not found` ⇒ FAIL; any host path ⇒ FAIL. The shipped detection verifier (arm 1a, which `continue`s on drivers) is NOT modified. The jail fixture is a second committed fixture with `lib_dir=/lib`.
F8 — Stop rule §3 is re-aimed: it fires only on a limit measured IN THE LANE that survives F1–F7 (never on `--list`'s behaviour).

### 11.9 PROFILE — the unit contract as pressure-tested and folded

## PROFILE — the kernel-admission profile as a definition-hash determinant, by construction (wave 6; #546, third attempt)

Implementer: ai-core (cross-crate scope granted for `crates/jammi-kernels/src/admission.rs`, `crates/jammi-db/src/store/
manifest.rs`). Worktree `/Users/vijaychakilam/git/f-inverse/wt-profile`, branch `unit/profile`, base `b8052978`.
Targets `/Users/vijaychakilam/git/f-inverse/targets/profile`. Rules: `RULES-implementer.md`.
Scope: `crates/jammi-kernels/src/admission.rs` (`ProbedOp`, `PROBED_OPS`, `admit`/`admit_cascade`, dispatch counters),
`crates/jammi-ai/src/fine_tune/worker.rs` (the materialization env at ~:3990), `crates/jammi-db/src/store/manifest.rs`
(`MaterializationEnv::kernel_admission_profile`), `ci/tools/probed-ops-index` (if the index must learn the enum),
the hash-completeness tests in `jammi-db`, `crates/jammi-ai/tests/it/fine_tune.rs` identity tests.

### 0. Premises (verified 2026-09-17; the executed record is on #546's 2026-09-17 comment)
- Two attempts excised. The wave-5 shape (a typed `PROBED_OPS` table + `dry_run` per row + a syn oracle that no forged
  `ProbedOp` enters the fold) failed because a syn oracle cannot enumerate every construction shape (UFCS, alias,
  macro invocation, non-path return types, transmute) — an enumerating oracle over an OPEN type. The record's own
  closing question: make the determinant BY CONSTRUCTION — "the profile computed from a closed enum whose variants
  are the rows, so a forged row cannot enter the fold because the fold never takes a `ProbedOp` value at all".
- `MaterializationEnv::kernel_admission_profile: Option<String>` (crates/jammi-db/src/store/manifest.rs:267) is declared, hash-affecting when
  `Some`, written by no producer (UNCOVERED; crates/jammi-ai/src/fine_tune/worker.rs:3998). K7: a content-addressable identity folds the complete
  output-affecting parameter set.
- What ships today: `PROBED_OPS: &[ProbedOp]` (crates/jammi-kernels/src/admission.rs:2103) with `report_key`, `kind`, `registry`; the
  eager-disable sweep derived from the table; per-op dispatch counters (`counters_for(key).snapshot()` — fused/eager).

### 1. Properties (binding)

**K1 — The set of admission-relevant ops is a CLOSED enum, and the profile is a total function of it.**
`ProbedOpId` (`#[derive(Copy, Clone, Eq, Ord, Hash, strum::VariantArray)]`, one variant per row of today's table;
`#[non_exhaustive]` FORBIDDEN) replaces the open table as the identity of an op: `PROBED_OPS` becomes
`ProbedOpId::row(self) -> &'static ProbedOp` (a `match` over every variant — the compiler enforces totality), and
`admit`/`admit_cascade` take `ProbedOpId`. The fold reads `ProbedOpId::VARIANTS` only. Oracle: a test that
`ProbedOpId::VARIANTS.len()` equals the number of `match` arms/rows and that every variant's row `report_key` is
unique; a `compile_fail` doctest showing a `ProbedOp` value cannot be constructed outside the crate (fields private,
no constructor) AND that the profile function does not accept a `ProbedOp` (its signature takes nothing but the
observed outcomes keyed by `ProbedOpId`). Mutation: add a variant without a row → compile error (state it).

**K2 — The profile folds the OBSERVED per-op admission outcome of the training run, not a prediction.** After
training, the worker reads, for every `ProbedOpId` variant, the dispatch outcome class the run actually took
(`Fused`/`Eager`/`Unreached`, derived from the counters' before/after window that brackets the training loop,
the same mechanism `capability_surface.rs` uses), plus the build facts that gate admission (the closed set of cargo
features: `cuda`, `flash-attn`, jammi-kernels' own — derived from `cfg!` at compile time, listed in the enum's own
module), plus `admission_mode` and the disabled-op set. The rendered profile is a canonical string (sorted by
variant order, one line per variant, `report_key=Outcome`), written via `with_kernel_admission_profile`. Oracle:
(a) a hash-completeness test in `jammi-db`: two envs differing only in one variant's outcome hash differently;
(b) an end-to-end identity test in `jammi-ai` (CPU): a fine-tune with one op disabled (via the existing disable
sweep) vs the same spec with it enabled produce DIFFERENT definition hashes and the manifest's profile lines name
the flip; (c) determinism: the same spec twice on the same host → identical profile string. Mutation: drop one
variant from the render loop → (a)'s per-variant test reds (the test iterates `VARIANTS`).

**K3 — `CachePolicy::Use` semantics stay honest.** With the profile written, a cache hit requires equal profiles by
construction; the manifest doc's UNCOVERED paragraph is deleted; the fine-tune `cache = Use` refusal (REUSE unit)
is unchanged by this unit.

**K4 — The index tool and the sweep read the enum.** `ci/tools/probed-ops-index` and the eager-disable sweep derive
their universe from `ProbedOpId::VARIANTS` (or the rows through it); the syn-based construction oracle from wave 5,
if any remnant exists, is DELETED (the property no longer rests on it) and its deletion named.

**K5 — Prior manifests stay readable.** `kernel_admission_profile` stays `Option<String>`, serde-default; a manifest
without it deserialises; a row without it never matches a row with it (the hash differs) — a test asserts both.

### 2. Non-goals
No new fused kernels. No change to admission policy. No GPU-only test (CPU-hermetic oracles; the GPU prove lane
picks up `capability_surface` unchanged).

### 3. Stop rule (pre-committed)
If K2's observed-outcome fold blocks twice in audit (an op whose outcome cannot be observed from the counters
without a per-op probe window), the unit ships K1 + K4 + K5 with the profile folding ONLY the build facts and the
disabled set (both by construction) and the outcome column labelled UNCOVERED per variant in the rendered string
(`report_key=Unobserved`), and #546 stays open with the executed record. A third block excises the unit.

### 4. Pressure-round folds
(appended by the lead after the pressure-tester's verdict)

### 4. Pressure-round folds (2026-09-17, REFINE — executed; BINDING, they override §1/§3 where they differ)
F1 — **K2 is DELETED, not demoted.** A `DefinitionHash` is computed BEFORE the work (`Catalog::probe_model_by_definition`) to find work already done; a profile known only after training makes that key non-computable ex ante — the identity's only consumer becomes impossible. The env is built before training at crates/jammi-ai/src/fine_tune/worker.rs:3993-4073 because an identity is an input fact. Also executed: the counter registry is process-global and `counters_for(..).record(..)` is `pub` and writable from a crate outside the workspace (the open universe relocates); Local-rank gangs run every rank as an in-process thread over the same counters; the pre-loop acceleration probe moves them; BERT/DistilBERT record `attention_block_flash` as `declined` identically with and without `flash-attn` (the observed column does not discriminate the case #546 was opened on); {Fused,Eager,Unreached} is not closed (both-moved exists); 4 of 15 rows have no dtype-neutral key and 2 share `lora_linear_fused`.
F2 — **K2' (the unit):** the profile is a canonical string over facts that are all EX ANTE and all by construction: `jammi_kernels::admission::{CUDA_COMPILED, FLASH_COMPILED}` plus every further build fact jammi-kernels itself publishes (a `BUILD_FACTS: &[(&str, bool)]` const in jammi-kernels — never a hand-list at the fold site, never `cfg!` in jammi-ai: `cargo tree` shows jammi-kernels can be built with `cuda` while jammi-ai's own `cuda` feature is off; `metal` must be in it), `admission_mode()`, the sorted `disabled_ops_requested()` ∩ `ProbedOpId::VARIANTS` rendered by variant, and the job's `DtypeClass`. Computed at worker.rs:~3990 where the env is built. Oracle (a) stays (hash-completeness: two envs differing in one fact hash differently — table-driven over every fact); oracle (b) becomes: a fine-tune under `disabled_ops` containing one variant vs none produce DIFFERENT definition hashes and the profile line names it — the two legs live in SEPARATE `[[test]]` targets (or a child-process harness) because `disabled_ops()`/`admission_mode()` are process-wide `OnceLock`s over env vars; oracle (c) determinism: same spec twice → identical profile.
F3 — **K1 totality is stated over the ROW, not over `dtype_neutral_key`:** name the 4 rows without a dtype-neutral key (`cast_scale`, `cast_add`, `rope_positions`, `scaled_cast_add`) and the 2 sharing `lora_linear_fused`; the profile renders every variant (a row's line is its `report_key` and whether it is in the disabled set — no key needed). Derive list: `#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, strum::VariantArray)]` (strum 0.26 already a workspace dep).
F4 — **K3 restated:** `cache = Use` stays refused (REUSE unit); K3 says only that the manifest doc's UNCOVERED paragraph is deleted and that the determinant is now complete over build facts + admission policy; observed per-op outcomes stay in the esc-075 acceleration report (non-durable, carries its attribution caveat).
F5 — Scope gains the prose sites that go false on landing: `crates/jammi-kernels/src/admission.rs:1774`, `crates/jammi-db/src/store/manifest.rs:250-266` and `:730`, `crates/jammi-ai/src/fine_tune/worker.rs:3997`, `ci/scripts/perf/test_finetune_ab_disable_op_keys.py:51-66` (the sealing argument + "no durable artifact folds this").
F6 — **§3 stop rule rewritten:** there is no fallback to fall to (F2 IS the unit); a second closing-audit block on K1/K2'/K4/K5 excises the unit (record on #546).

### 11.10 COOKGATE (excised; own PR later) — the unit contract as pressure-tested and folded

## COOKGATE v2 — amend the shipped session-lifecycle gate to the property it claims (wave 6; #539; round-1 KILL folded)

Supersedes `cookgate.md` in full. Round 1 (executed) established: the AST gate already exists on main
(`ci/scripts/check_cookbook_session_lifecycle.py`, e464ec1b, 13/13 self-tests, wired at .github/workflows/ci.yml:1410/:1412 in the
toolchain-free guard matrix on every PR). #539 is stale, not the tree. The home STAYS the guard matrix. What is
genuinely undone, each executed by the round as a live false negative on main:

Implementer: cookbook (cross-scope granted for `ci/scripts/check_cookbook_session_lifecycle.py` — human-amend-only,
SWARM_GATE_TOUCHED; the lead admin-merges — and `ci/scripts/select_render_chapters.py::_executed_python_cells`
import only). Worktree `/Users/vijaychakilam/git/f-inverse/wt-cookgate`, branch `unit/cookgate`, base `b8052978`.
Rules: `RULES-implementer.md`.

### 1. Properties (binding; each with an executed RED fixture on the shipped gate → GREEN after)

**G1 — Taint propagates through the block's dataflow, not one expression subtree.** Within the tempdir's block,
a Name bound (by `=`, walrus, `for`, `with … as`) from an expression that mentions a tainted Name is tainted;
`Path(d) / "x"`, `os.path.join(d, ..)`, `str(d)`, `d.name`, `f"{d}/x"` all propagate. Fixtures (RED at base): the
round's D (`root = d; connect(f"file://{root}")`) and D' (`cat = Path(d) / "catalog"; connect(f"file://{cat}")`),
plus a two-hop chain. Mutation: drop the assignment rule → D reds.

**G2 — Every tempdir whose removal is reachable is in scope, not only `with TemporaryDirectory`.** Sources:
`with tempfile.TemporaryDirectory(...) as d` (any import/alias form — `from tempfile import TemporaryDirectory
as T` resolved through the module's import bindings), `td = TemporaryDirectory()` + `td.name` (removed by
`td.cleanup()` or `with td`), `stack.enter_context(TemporaryDirectory())`, and `d = tempfile.mkdtemp(...)` whose
removal is a `shutil.rmtree(d…)` / `os.rmdir` / `Path(d).rmdir()` anywhere in the same function (a held,
never-removed mkdtemp stays out of scope — state the rule in the docstring). Fixtures (RED at base): A, B, C, E
from the round; a held-mkdtemp control (GREEN). Mutation: drop the mkdtemp+rmtree source → C reds.

**G3 — The universe is every `cookbook/**` `.py` and every python cell of every `cookbook/**` `.qmd`, with ONE
extractor and a committed literal floor.** `_PY_ROOTS`/`_QMD_ROOTS` widen to `cookbook/`; the cell extractor is
imported from `select_render_chapters._executed_python_cells` (delete the gate's own regex); the universe floor is
a COMMITTED LITERAL pair (files, cells) asserted `>=` in the self-test (today 82 / 364 — measure, don't copy), so an
extractor regression that drops cells reds. Cells parse CONCATENATED per file (the shipped soundness argument) with
a per-cell line-offset map so a finding reports the real `.qmd` line. The runtime rail's own fixture gallery
(`cookbook/book/tests/test_session_lifecycle_guard.py`) is inside the universe and MUST stay clean by the rules
(it has zero tempdir sources) — assert it explicitly, never exclude it.

**G4 — Sink and credit rules stated as they behave.** Sink: `jammi.connect(...)` and any binding of it the
module's imports create (`from jammi import connect`, `connect as c`), `target` positional or keyword. Credit: the
session is the context manager of a `with`/`async with` item in the same statement or a nested one; everything
else is an offender (existing rules kept; fixtures kept).

**G5 — The live offender is fixed.** `cookbook/book/scripts/build_media_tower_lora_cache.py:833/842` (mkdtemp +
rmtree with `jammi.connect` sessions at :292/:507/:675 never closed): every session closed before the rmtree
(the gate must RED on it at base under G2, GREEN after the fix — this is the unit's real-tree RED).

**G6 — Docs reflect the state.** `cookbook/book/tests/test_session_lifecycle_guard.py:13`'s "filed as issue #539" sentence becomes the
complementarity statement (runtime rail = pytest lanes; static gate = every other lane); the gate's module doc
states G1–G4's rules and the literal floor. #539 closes on merge.

### 2. Non-goals
No new gate, no new CI home, no change to the runtime rail.

### 3. Stop rule (pre-committed)
If G1's dataflow rule blocks in the closing audit on a shape it cannot resolve (dynamic attribute, comprehension
scope), the shape is listed as UNCOVERED by name in the module doc with its executed fixture marked expected-MISS,
and the unit ships; a second block excises the unit (#539 stays open with the record).

### 4. Round-1 record
Verdict KILL on the unit-as-scoped (rebuild of a shipped gate); folded here as the amendment the round derived.

### 11.11 TYPESTATE (excised in design; shipped as WIDTHFIX) — the unit contract as pressure-tested and folded

## TYPESTATE v2 — "no kernel sees a query that met zero width checks", by type (wave 6; issue #519; round-1 KILL folded)

Supersedes `typestate.md` in full. Implementer: numerics (cross-crate scope for every producer/consumer of
`ValidatedQuery`). Worktree `/Users/vijaychakilam/git/f-inverse/wt-typestate`, branch `unit/typestate`, base `b8052978`.
Targets `/Users/vijaychakilam/git/f-inverse/targets/typestate`. Rules: `RULES-implementer.md`.
Scope: `crates/jammi-numerics/src/{query.rs,distance.rs}` (+ a `trybuild` dev-dependency and `tests/ui/**`),
`crates/jammi-db/src/index/{exact.rs,placed.rs,mod.rs,segment.rs}`, `crates/jammi-db/src/store/mod.rs` (`search_vectors`,
`search_vectors_local`), `crates/jammi-db/tests/it/whose_fault_gate.rs` (its ALLOWED keys move; re-anchor by
(file, fn, ordinal, occurrence) — never by line), `crates/jammi-ai/src/{query/builder.rs,eval/runner.rs,pipeline/
context_set.rs,pipeline/neighbor_graph.rs,operator/ann_search_exec.rs}`, `crates/jammi-server/src/grpc/peer.rs`,
`crates/jammi-ballista/src/codec.rs`, `crates/jammi-bench/src/{operator_mirror,recall,search_rss,sweep}.rs`,
`crates/jammi-test-utils/src/lib.rs:366` (the shared `vq` helper).

### 0. Premises (round 1, executed on b8052978 — binding)
- 16 production `validate_query` entries defer (`None`): 12 literal, 4 runtime-`Option` (crates/jammi-ai/src/query/builder.rs:81, crates/jammi-ai/src/eval/runner.rs:160,
  crates/jammi-ai/src/pipeline/context_set.rs:424, crates/jammi-ai/src/pipeline/neighbor_graph.rs:349) feeding single-typed consumers (`AnnSearchExec.query_vector`, `Node.vector`).
- 3 production `require_authority_width` sites (crates/jammi-db/src/index/placed.rs:286, :352, crates/jammi-db/src/index/exact.rs:211); four consumers take `&ValidatedQuery`
  and choose the check at runtime from `catalog_dimensions: Option<usize>` (crates/jammi-db/src/index/exact.rs:135-213, crates/jammi-db/src/index/placed.rs:264-291,
  :328-363, crates/jammi-db/src/store/mod.rs:2520-2573).
- crates/jammi-server/src/grpc/peer.rs:162-176 checks width through `Deref` and raises `Status::failed_precondition(...)` per segment (a loop).
- The `Stored` exception (crates/jammi-numerics/src/query.rs:51-66, crates/jammi-ai/src/query/builder.rs:57-81) passes `None` with the authority in hand, on purpose.
- rustdoc `compile_fail,E0xxx` annotations do not pin the error on rustc 1.94.0; only `trybuild` with a committed
  `.stderr` does. E0277 never arises for this design; E0308/E0599 do.

### 1. Properties (binding)

**V1 — The unconstructible state is "a kernel receives a query that has met zero width checks".**
`ValidatedQuery<S>` with `S ∈ {Deferred, Authorized}` (sealed marker trait). `validate_query(values, Some(w), src)
-> Authorized` (the construction-time authority check, error unchanged); `validate_query(values, None, src) ->
Deferred`. The transitions, ALL consuming `self` and returning `Authorized`, with errors byte-identical to today's:
`Deferred::require_width(self, expected, artifact) -> Result<Authorized, QueryValidationError>` (ArtifactMismatch),
`Deferred::require_authority_width(self, expected) -> Result<Authorized, QueryValidationError>` (Width{source}),
and `Deferred::authorize_with<E>(self, expected, on_mismatch: impl FnOnce(usize /*actual*/) -> E) -> Result<Authorized, E>`
(the peer entry keeps its own `Status::failed_precondition` bytes; the per-segment loop checks every segment's
width with `authorize_with` on a clone, or checks all widths then transitions once — either way every byte of
every existing error is unchanged, proven by the existing tests staying green). `Authorized::require_width(&self,
expected, artifact)` re-checks against a later artifact (today's downstream semantics, unchanged).
`Authorized::defer(self) -> Deferred` exists for the runtime-`Option` entries (unifying the two arms; sound because a
Deferred always meets a check before any kernel). `Deref<Target=[f32]>`, `as_slice`, `into_inner`, and every kernel
signature (`crates/jammi-numerics/src/distance.rs:29, :58`, `index/segment.rs`, `crates/jammi-db/src/index/mod.rs:80 VectorIndex::search`) exist ONLY for
`Authorized`. Consumer seams that today take `&ValidatedQuery` and choose the check at runtime (`search_vectors`,
`search_vectors_local`, `exact_vector_search`, the two placed arms) take `ValidatedQuery<Deferred>` BY VALUE, perform
the transition they already perform (same call, same error), then call kernels with `&Authorized`. Callers holding
an `Authorized` (the `Some` arm) pass `.defer()`; callers holding a `Deferred` pass it through.
Oracle: `trybuild` UI tests in `crates/jammi-numerics/tests/ui/` with committed `.stderr`: (a) `Deferred` passed to a
kernel (`&[f32]`/`&Authorized` parameter); (b) `.as_slice()`/`.len()`/`into_inner()` on `Deferred`; (c) a `Deferred`
stored in `AnnSearchExec`'s field type is impossible? — NO: that field IS `Deferred` (the seam is by value); instead
(c) `Authorized::require_authority_width` does not exist (an authorized query cannot be re-attributed to the caller,
today's property). Mutation: implement `Deref` for `Deferred` → (a)/(b) `.stderr` diverge → RED.

**V2 — Every width check that exists today exists after, at the same site, with the same error bytes.** The
implementer lists each of the 3 authority checks, every `require_width`, and peer.rs's check with before/after
lines; no check is added or removed (the transition IS the existing check). Oracle: every existing test that
asserts a width error message stays green (controls, named), and `cargo test --workspace` on the touched crates'
filtered suites.

**V3 — The `Stored` exception is unchanged.** builder.rs's `Stored` arm still passes `None`; it reaches the store's
by-value seam as `Deferred` and meets the SAME check it meets today. State this in the module doc.

**V4 — The whose-fault gate stays sound.** `whose_fault_gate.rs`'s ALLOWED entries are re-keyed to the new
(file, fn, ordinal, occurrence) tuples; the inverse control at `:851` stays green; the entry note at `:123` describes
the new shape. Never a line-keyed entry.

**V5 — Public surface diff enumerated.** Every `pub` item added/removed/renamed in `jammi_numerics::query` and every
changed signature across crates, listed in the hand-back (there is no Rust API baseline; the proto freeze is
untouched — say so).

### 2. Non-goals
No change to width semantics, attribution rules, finite-check, `QuerySource`, or any error string.

### 3. Stop rule (pre-committed)
If the by-value seam reshaping blocks in the second pressure round on a consumer that must hold a query across
multiple artifacts without cloning (a measured cost, not a preference), V1 ships with `Authorized` only at the
kernel boundary and the seams keeping `&ValidatedQuery<Deferred>` plus `Deferred::require_width(&self, ..) ->
Authorized` returning a CLONE (cost: one `Vec<f32>` copy per search, measured and stated), residual on #519. A
second KILL excises TYPESTATE from wave 6 (record on #519; it joins wave 7's design list).

### 4. Pressure-round-2 folds
(appended by the lead after the verdict)

### 5. Fold from the implementer's own reading (2026-09-17, before any code) — binding
**V6 — Sites that hold the authority and pass `None` are corrected, each with a RED-at-base oracle (the unit's
runtime teeth).** Found: `crates/jammi-ballista/src/codec.rs` `decode_ann_search` (hardcodes `expected_width: None`
while `table.dimensions()` is resolved lines above); `crates/jammi-db/src/index/placed.rs` `search_mixed`'s
non-empty-`local` branch (no authority check; the first local segment's artifact check misattributes a caller
mistake); `crates/jammi-db/src/store/mod.rs` `search_vectors_local`'s `Some(index)` branch (calls
`index.search_final` directly, bypassing `search_final_placed`'s authority resolution). Under v2 each becomes a
compile error (a `Deferred` reaching a kernel) and is fixed by performing the authority check the site can already
perform; the oracle for each: a width-mismatched query at that entry is refused with `Width{source}` (caller fault)
where today it is refused downstream as `ArtifactMismatch` or not at all — RED at base, GREEN after. These three
are exceptions to V2's "same error bytes at the same site" and are listed as such in the hand-back (K2: validate at
the input edge). The `encode` side of the codec stays state-agnostic (`into_inner` on `Authorized` after the
check; a query that cannot be authorized at decode is a typed decode refusal — a NEW refusal path, with its own
round-trip test).

