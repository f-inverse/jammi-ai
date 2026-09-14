# CONTRACT — feat/500-B-U2b as PR-B2: three closed units consolidated onto main, and the consolidation itself examined

**Contract of record.** slug: `feat_500-B-U2b` · this file is the committed mechanism contract
`ci/scripts/check_rigor_record.py` requires under `docs/rigor/contracts/**` for the rigor record at
`docs/rigor/feat_500-B-U2b.jsonl` (the lead's export, landed as the last, docs-only commit). The PR's head
branch is `feat/500-B-U2b` (PR #575, retitled), which is why the slug names one unit while the branch
carries three: **U2b** (own head `acbaad9e`, PR #575), **U3** (own head `d4100302`, PR #576) and **U5a-1**
(own head `2f7dbe68`, PR #577), each closed on its own branch by the rigor chain with its remaining
findings filed as Stands, then cherry-picked commit by commit onto `main` @ `4c31ef9e` (the PR-B1 merge).
Every citation names a construct by path and item (`path::item`), never a `path:line` offset.

## Scope of PR-B2

`git diff --stat main` at the final head: 107 files — `crates/jammi-ai` (29), `crates/jammi-server` (25),
`crates/jammi-db` (18), `crates/jammi-wire` (9), `crates/jammi-cli` (3), `crates/jammi-python` (3),
`crates/jammi-client` (1), `crates/jammi-admin` (1), `clients/python` (6), `docs/guide` (7),
`docs/plans/67-distributed-training` (3), `docs/maintainer/MAINTAINER-GUIDE.md`, this directory.
Commit groups, in order: U2b's 23 own commits (11 plan-doc commits dropped because `main` already
carried their later twins — the #556 construct-citation conversion and the A4/A14 dated rows), U3's 76,
U5a-1's 35 (with its migration renumbered at pick time), then five consolidation commits:
the W=1 digest-pin helper, the plan/contract normalization, two `jammi-db` fixes for hand-resolved
migration hunks, and the gang spec-parity fixture's `cache` field — plus this contract and its record
commits. The two `pre-consolidation` squash commits on the unit branches are not picked.

## U2b — eager loader, partition rule, scaler, whole-set arms (as shipped: the streaming arm excised)

**Mechanism.** The trainer's only partition constructor is
`crates/jammi-ai/src/fine_tune/partition.rs::PartitionSpec::single_rank` (rank 0 of world 1);
`crates/jammi-ai/src/fine_tune/partition.rs::batches_per_epoch` is `ceil(train_count / (W·B))` and every
step quantity is indexed by global batch; `crates/jammi-ai/src/fine_tune/data.rs::text_chunk_for_rank` selects
a rank's chunk; the eager `TextRows` path reads a committed training set back through
`crates/jammi-ai/src/fine_tune/training_set.rs::read_back_sql`, which applies `training_set_order_by`;
`crates/jammi-ai/src/fine_tune/training_set.rs::training_set_spec` is the one `TrainingSetSpec` constructor
every production site builds through; `crates/jammi-ai/src/fine_tune/batch_bucket.rs::resolve_bucket_rung`
carries the rung-pinning option (unused, `None`, at the sole W=1 call site). The streaming reader
(`read_back_range_sql`, `open_row_range_stream`, `BatchChunker`, `ChunkLease`) was built and then EXCISED by
the unit's own pre-committed stop rule (fix round 3); the history keeps both the build and the revert.
The writer's row-group size is the fixed default (`crates/jammi-db/src/storage/writer.rs::ObjectParquetWriter::open`,
no test-hook knob).

**Properties (over every input).**
- P1 For every committed training set read through `read_back_sql`, the row order equals the committed order,
  on a multi-row-group table at any `target_partitions` — the `ORDER BY` is re-applied, never assumed.
- P2 Every production call site that reaches a training-set table's relation key, by any of the three
  routes (`.sql_relation(`, the UFCS form, `registered_name(`), is on a reviewed allow-list with a named
  behavioural order assertion; a new caller anywhere under `crates/*/src` fails the oracle.
- P3 The adapter bytes of the contrastive, regression and GradCache W=1 runs are byte-identical to their pinned
  fixtures (K3: the scaler is computed once over the train prefix), so the refactor moved no bytes.
- P4 `training_set_spec` is a pure pass-through: its `definition_hash` equals a hand-built literal's.

**Oracles, by name, and what each excludes.**
- `crates/jammi-ai/tests/it/training_set.rs::read_back_re_applies_the_committed_order_across_row_groups` (P1;
  excludes: a reader that bypasses `read_back_sql`).
- `crates/jammi-ai/src/fine_tune/training_set.rs::reader_class_allow_list::every_production_sql_relation_call_site_is_on_the_allow_list`
  (P2; its universe is the three named routes — a fourth route is #551's `RelationKey` newtype).
- `crates/jammi-ai/tests/it/training_set.rs::refactor_parity`, `::regression_refactor_parity`,
  `::gradcache_completes_at_w1_with_a_pinned_adapter_digest` (P3; the fixtures are platform-pinned prints;
  mining's W=1 digest is NOT pinned — #551 — because no oracle executes mining to completion).
- `crates/jammi-ai/src/fine_tune/training_set.rs::tests::training_set_spec_matches_a_hand_built_spec_byte_for_byte` (P4).
- `crates/jammi-ai/src/fine_tune/partition.rs::tests::single_rank_is_rank_zero_of_world_one`.

**Filed.** #544 (the residency-bounded streaming reader, U2c), #550 (enum inventories by construction),
#551 (`RelationKey` newtype; mining W=1 parity oracle).

## U3 — `FineTune` producer, `model_materialization` migration, cache policy refused, prune guard

**Mechanism.** `crates/jammi-db/src/store/manifest.rs::ProducingDescriptor::FineTune` carries an opaque,
versioned canonical encoding (`spec_canonical` + `spec_schema_version`, K7 exhaustive destructuring of
`FineTuneConfig` and `TrainingCommon`). Migration `033_model_materialization`
(`crates/jammi-db/src/catalog/schema.rs::MIGRATION_033_MODEL_MATERIALIZATION`) adds the nullable
`models.definition_hash` / `models.input_anchors_json` and two indexes; no `manifest_path` column — the
sidecar is the fixed name `materialization.json` under the model's artifact prefix, written last after
`manifest.json`. `crates/jammi-db/src/catalog/model_repo.rs::Catalog::probe_model_by_definition` never matches
`NULL` and restricts to the servable set; `::record_model_materialization` writes only after the finalize
CAS. `crates/jammi-ai/src/fine_tune/spec.rs::admit_training_spec` is the ONE admission function on every
durable training submit edge (`InferenceSession::enqueue`, `submit_fine_tune_spec_deduped`,
`train_context_predictor_deduped`): per-kind validate, rank admission, and the `CachePolicy::Use` refusal —
model-level cache reuse is not implemented (#562); `cache` lives on `TrainingSpec::FineTune`, so
`GraphFineTune` cannot carry one. Every `models/`-prefix byte-delete consults
`crates/jammi-db/src/store/reconcile.rs::ResultStore::prefix_is_referenced` (admin-scoped, exact key or
immediate parent) through `::delete_unreferenced_prefix` — reconcile's reap, the worker's abandon path, the
epoch-checkpoint sweep, and the trainer's mid-run retention prune
(`crates/jammi-ai/src/fine_tune/trainer.rs::TrainingLoop::delete_epoch_checkpoint_guarded`); the unguarded
`ArtifactStore::delete_epoch_checkpoint` is deleted. `ReconcileReport.referenced`/`referenced_count` ride
the wire (`catalog.proto` tags 15/16), rendered by `jammi reconcile`. The recompute arm for `FineTune` is
retrain (`crates/jammi-ai/src/pipeline/recompute.rs::InferenceSession::recompute_fine_tune`), and a recorded
PINNED anchor is re-resolved pinned, never downgraded (`::reresolve_recorded_anchor`).

**Properties.**
- P5 For every durable submit edge and both transports, `CachePolicy::Use` on either LoRA kind is refused,
  typed and identically; `Bypass` (or unset) trains. Unset encodes as the absent field on both transports.
- P6 For every `models/` byte-delete path, a prefix named by any live row (any tenant, exact key or
  immediate parent) is never deleted; the refusal is typed (`StorageError::Referenced { prefix, count }`).
- P7 Migration 033 is present and ordered after 032 by relative position (never `.last()`), on both backends.
- P8 A `FineTune` descriptor round-trips; a variant field added to `FineTuneConfig` or `TrainingCommon`
  without touching the canonical producer fails to compile.

**Oracles.** `crates/jammi-server/tests/it/grpc_remote_compute.rs::a_fine_tune_cache_use_is_refused_identically_on_both_paths`
and `crates/jammi-ai/tests/it/rank_admission.rs::a_fine_tune_cache_use_is_refused_through_enqueue_too` (P5; the
structural oracle in `rank_admission.rs` enumerates the three named edges — deriving that universe from the
source tree, and a behavioural oracle for the context-predictor edge, are #573);
`crates/jammi-db/tests/it/reconcile.rs::delete_unreferenced_prefix_refuses_a_referenced_prefix_and_deletes_an_unreferenced_one`
and the one-level checkpoint-reclaim oracle (P6);
`crates/jammi-db/tests/it/migrations.rs::migration_033_is_ordered_after_032_and_adds_model_materialization_columns` (P7);
`crates/jammi-db/tests/it/materialization.rs::probe_model_by_definition_finds_a_row_with_matching_pinned_anchors`,
`::probe_model_by_definition_tenant_fan_out`, `::record_model_materialization_refuses_a_row_the_finalize_cas_has_not_committed`
(P8 and the servable-set predicate). **Filed.** #573, #562, #566, #546, #547, #548.

## U5a-1 — `GangService` wire, I-GANG, training-set identity (W=1 lattice)

Its own committed contract is `docs/rigor/contracts/feat_500-C-U5a-1.md` (mechanism, properties, oracles,
addenda at the shipped shape); this file adds only what the consolidation changed: its migration is
`034_jobs_training_set_identity`, ordered after 033, pinned at the tuple in
`crates/jammi-db/src/catalog/migrations.rs`, the constant
`crates/jammi-db/src/catalog/schema.rs::MIGRATION_034_JOBS_TRAINING_SET_IDENTITY`, the const
`crates/jammi-db/tests/it/migrations.rs::EXPECTED_MIGRATION_NAMES`, the `IN`-list literal in
`crates/jammi-db/tests/it/migrations.rs::migration_029_copies_training_jobs_rows_into_jobs_as_queued`, and the
ordered-after oracle
`crates/jammi-db/tests/it/migrations.rs::migration_034_is_ordered_after_033_and_pins_the_pair_at_the_schema_edge`,
whose position assertion is relative to `033_model_materialization`. Its contract's two renumbering notes
state that as the shipped fact. **Filed.** #574, #566.

## The consolidation — every hand-resolved hunk, and the property each preserves

The picks were replayed in each unit's own order; a conflict was resolved by rebuilding the file from
`main`'s side and applying the pick's own hunks, never by taking a unit's copy whole (the one exception,
`DESIGN.md`, was caught and rebuilt — C8). Fidelity check: `git merge-tree --write-tree --merge-base=<unit base>
main <unit head>` gave the three-way target for U2b; after the picks, only the two hand-resolved files
differed from that target, by exactly the resolutions below.

- C1 `crates/jammi-ai/src/fine_tune/training_set.rs` — `main` (PR-B1) had excised the graph arm's
  `materialize_sampled_pairs`/`RegisteredPairs` and their tests; every U2b hunk targeting that arm was dropped,
  every hunk targeting surviving code applied (`read_back_range_sql` added, then removed by U2b's own revert;
  the `reader_class_allow_list` module; `training_set_spec`; its one surviving test in a new `mod tests`).
  Property: the graph arm stays `main`'s (#538); U2b's shipped surface is complete (P2, P4 oracles run).
- C2 `crates/jammi-db/src/catalog/result_repo.rs` — `main`'s #550-corrected `ResultTableKind::ALL` doc kept
  over U2b's earlier wording of the same fact; U2b's later refusal-message doc applied. Property: the doc
  states what E0004 enforces (the pattern), not the array.
- C3 `crates/jammi-ai/src/pipeline/recompute.rs` — `main`'s missing-sidecar refusal bullet and U3's pinned-anchor
  bullet both kept; the heading counts four refusals. Property: every refusal the code makes is documented.
- C4 `crates/jammi-db/tests/it/materialization.rs` — both sides' imports merged.
- C5 `docs/guide/src/fine-tuning.md` — `main`'s graph-arm paragraph (#538) kept above U3's cache section, which
  its own later commit restates as "not yet supported".
- C6 `crates/jammi-ai/src/fine_tune/worker.rs` — U2b's `single_rank` comment on the builder chain (it is
  not on `main`; U2b's own head carries it) kept; U3's `.result_store(result_store)` builder call added
  (the guarded prune port, P6).
- C7 `crates/jammi-db/src/catalog/{migrations.rs,schema.rs}`, `tests/it/migrations.rs` — 033 (U3) then 034
  (U5a-1); the two `jammi-db` consolidation fixes close 033's raw-string literal before 034's declaration and
  split the ledger into one tuple per migration (both were compile errors, caught by the workspace build at
  the head, never by a test — P7's oracle and the 034 oracle then ran green).
- C8 `docs/plans/67-distributed-training/DESIGN.md` — rebuilt as `main`'s v4.6 file with U3's § 3 and its § 6
  oracle row three-way merged in; `UNITS.md` § U5a spliced from the U5a-1 branch (U5a-1/U5a-2 split) with
  `main`'s #540 fold restored; plan citations stay in construct form (zero `path:line` in the plan docs, as on
  `main`). Property: the plan on `main` is the plan on this branch plus what the three units shipped.
- C9 `crates/jammi-ai/tests/it/training_set.rs::pinned_prints` — U2b's two digest pins enumerated every file
  in the prefix and U3 now writes `materialization.json` there (wall-clock `produced_at`, per-process
  `produced_by`); the three readers collapse to one helper that excludes the sidecar, as the contrastive pin
  already did. Property: P3 compares adapter bytes, not provenance metadata; the adapter prints were identical
  before and after (the failing assertion's `left`/`right` differed only by the sidecar entry).
- C10 `crates/jammi-server/tests/it/gang_training_spec_parity.rs::fine_tune_spec_world_two` — U5a-1's
  producer→consumer parity fixture builds a real `TrainingSpec::FineTune` literal, and U3 added the `cache`
  field to that variant on its own branch, so the literal did not compile on the consolidated head (the one
  hunk the build, not a test, caught across units). `cache: CachePolicy::Bypass` is added. Property: `cache`
  is a call-time dial excluded from `spec_canonical`, so the `world_size` decode the oracle asserts is
  unaffected; `Bypass` is the serde default (`#[serde(default)]` on the field, `#[default]` on the enum) a
  row persisted before the field existed decodes to, so `get_job_for_rank` reads U3's `TrainingCommon` shape
  either way — the parity test executes green at this head.

## Gates at the final head

`cargo build --tests --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
`cargo fmt --all -- --check`, `cargo test --workspace --no-fail-fast` (hermetic; no `JAMMI_TEST_PG_URL`),
and every `ci.yml` guard-matrix command (86 entries, run through PyYAML) on both `main` and this head.
The rigor record concatenates the three units' gate-state rows and the closers taken at this head:
a `pressure-tester` pass over this consolidation and an `oracle` PASS (the oracle record's `head_sha` tree
equals the head outside `docs/rigor/**`). No further adversarial round was opened on the consolidated head:
each unit's closing audit ended with its findings filed (#573, #574, #566, #562) under the program's closing
rule, and the consolidation's own changes are the ten resolutions above.

## Residuals recorded UNCOVERED

- `crates/jammi-ai/tests/it/jobs_cancel.rs::a_claimed_training_jobs_cancel_request_is_honoured_at_the_next_epoch_boundary`
  (a `main` test, unchanged here but for the `cache` field) timed out at its 15 s bound in both full
  `cargo test --workspace` runs on this branch — each of which ran while the 86-command guard matrix was
  executing on the same machine — and passed in every uncontended shape: alone (1.3 s), inside the
  `jammi-ai` `it` suite alone at this head (552/552), and under the workspace-unified feature set alone at
  this head and on `main` (1.07 s each). A load-sensitive bound, not a consolidation defect — recorded so a
  CI recurrence is read as such and fixed at the test (widen or make the watcher cadence explicit), never
  by touching the mechanism. (`main`'s own `jammi-ai` `it` suite fails
  `acceleration_report::probed_ops_bind_to_the_real_registry_and_key_sets_are_dtype_deterministic` on this
  machine; that test is untouched by this PR and passes at this head.)
- The kernel-admission profile determinant (#546), the model→prefix ownership edge (#547), the untagged
  `JobSpec` (#548), and the W=1 mining parity oracle (#551) stand as filed by the units.
