# Contract — `feat/500-wave3b`: U5b-1a-A2 (root identity), U5b-0 (leaf inventory), the merge path

Plans 67/68, wave 3 remainder, lead-built (2026-09-15 directive: the lead
implements; agents run only for the CI-required records — this contract's
pressure row and the phase-5 oracle at the final tip). This contract is
EXECUTABLE: every property below names the test that asserts it, every
"impossible" claim names the executed attempt to falsify it, and every new
call site names the oracle that goes red when it is deleted.

## 1. Scope

1. **U5b-1a-A2 — result-root identity across spellings, and the membership
   predicate built on it.** Filed by U5b-1a's round-3 stop rule, widened by
   round-5's (`docs/rigor/contracts/feat_500-C-U5b-1a.md` §6, §10, §12); a
   precondition of U5b-1b-ii. BUILT on this branch.
2. **U5b-0 — partitioned attestation inventory** (`docs/plans/67-distributed-
   training/UNITS.md` § U5b-0): per-row-group leaf digests in
   `MaterializationManifest`, `artifact` the fold over the leaves,
   `MANIFEST_VERSION` bump, an old sidecar a cache MISS. TO BUILD on this
   branch (§4 states the design and its oracles before the code exists).
3. **The merge path, locally** — `ci/scripts/merge_path.sh`: the CI matrix
   read from the workflow files at run time, in the gate-safe order
   (records last). Documented in the maintainer guide's PR-gate paragraph.
4. **The live lane's container** — `.github/workflows/ci.yml` `test-live`
   gains the `safe.directory` step every other container job has; the
   `pinned_source_gate` tests shelled out to `git ls-files` and died with
   "dubious ownership" on every main run (nine on 937d72e6, six on ad2f5b7b).

Out of scope: U5b-1b-i/ii/iii, U4b, U5a-2, U5b-2, U7b-A2b (their own
branches); a shared refusal-message constructor for the eager and streamed
decoders (named by U2c's contract, still not done).

## 2. U5b-1a-A2 — the design that survives the three failed rounds

U5b-1a's rounds 1–3 failed on the same fault line each time: identity was
computed where it could disagree with what the store actually rooted at (a
URL VIEW of `artifact_dir` instead of the literal path; an alias fold that
produced a string the store never rooted at). The design here keeps the
verbatim root untouched everywhere and adds identity as a SEPARATE value
that is never used to root anything:

- `RootIdentity::of(root: &str) -> Result<RootIdentity>` — one total
  function of the verbatim root string, run by the process that OWNS the
  root, at registration, on its own filesystem
  (`crates/jammi-db/src/catalog/instance.rs`). Rules, by the scheme the
  store's own parser (`StorageUrl::parse`, the one alias table) assigns:
  object stores → `{canonical scheme}://{authority lowercased}/{key}` with
  trailing `/`s trimmed and key case preserved (`r2://` ≠ `s3://`); local
  roots → `file://{longest existing prefix, canonicalised}{remainder,
  lexical}` (symlinks, `.`/`..`, the filesystem's case; a root the store has
  not created yet has the identity it will have); `memory://` → refused,
  typed, naming the root and the way out (an in-memory store is never
  shareable); an unknown or upper-case scheme → refused exactly as the
  store refuses it when rooting.
- `MemberRoot { root, identity }` — the verbatim string PAIRED with its
  identity, built only by `MemberRoot::resolved(config)` in production
  (`resolved_result_root()` then `RootIdentity::of`); the `test-hooks`
  `MemberRoot::new(root)` derives the identity the same way and panics on
  an unshareable root (a fixture bug).
- `instances.result_root_identity TEXT` — migration
  `036_instances_result_root_identity`, appended, nullable, pinned at the
  four K5 sites; written by `upsert_instance`/`reregister_instance` from
  the one `MemberRoot`.
- `GangListing.root_identity: &RootIdentity` — the caller's own;
  `list_gang_members` adds `AND i.result_root_identity = $n` to its SQL
  (a byte-exact `=` on a string this engine wrote; NULL never matches).
  The verbatim `result_root` column is never compared.

## 3. U5b-1a-A2 — properties, each with its executed oracle

| Property | Oracle (all GREEN at this head) |
|---|---|
| P-A1 alias/authority-case/trailing-slash fold on object stores; key case, bucket, backend stay distinct | `catalog::instance::root_identity_tests::object_store_aliases_authority_case_and_trailing_slashes_fold`, `::object_store_key_case_buckets_and_backends_stay_distinct` |
| P-A2 local roots: symlink ≡ target, `.`/`..`, trailing slash, `file://` ≡ bare, not-yet-existing leaf, relative against cwd; distinct dirs distinct | `::a_local_root_folds_symlinks_dot_segments_trailing_slashes_and_the_file_scheme`, `::a_local_root_the_store_has_not_created_yet_has_the_identity_it_will_have`, `::a_relative_local_root_is_taken_against_the_working_directory`, `::distinct_local_roots_stay_distinct` |
| P-A3 `memory://` and an unknown scheme refused naming the root | `::a_memory_root_and_an_unknown_scheme_are_refused_naming_the_root`; `config::tests::from_config_refuses_a_memory_result_root_for_a_member_only` (a library config with the same root is untouched; an upper-case scheme is refused on both paths) |
| P-A4 the row carries the verbatim spelling AND `RootIdentity::of` of it, over every advertising arm | `config::tests::from_config_member_root_is_resolved_result_root_verbatim_over_every_arm`, `::from_config_root_identity_is_of_the_resolved_root_over_every_arm`, `::from_config_never_aliases_gcs_and_gs_result_root_spellings_but_their_identities_are_one`; real sessions: `crates/jammi-ai/tests/it/storage_root.rs::member_row_matches_resolved_root_*` (identity column asserted), `::a_member_with_a_memory_result_root_is_refused_at_session_construction` |
| P-A5 the predicate: same-location spellings ARE members (gcs/gs; symlink/target; authority case; trailing slash); different locations and NULL identities are NOT | `crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_gang_members_of_each_other`, `::a_symlinked_local_root_and_its_target_are_the_same_gang`, `::list_folds_authority_case_and_trailing_slash_but_not_key_case`, `::file_and_s3_rooted_members_are_not_gang_members_of_each_other`, `::list_excludes_a_member_with_peer_addr_set_and_no_root`, `::a_row_with_a_root_but_no_identity_is_never_a_member` (sqlite + postgres arms) |
| P-A6 migration 036 appended after 035, nullable on both dialects, replayed with the instances family | `crates/jammi-db/tests/it/migrations.rs::migration_036_is_ordered_after_035_and_adds_instances_result_root_identity` (both arms); the 029-replay DELETE list and the `pragma_table_info` teeth include the column |

**Mutation (executed).** With the `AND i.result_root_identity = $n` clause
removed from `list_gang_members`, exactly the four exclusion oracles of
P-A5 go RED (`list_excludes_a_member_with_peer_addr_set_and_no_root`,
`list_folds_authority_case_and_trailing_slash_but_not_key_case`,
`file_and_s3_rooted_members_are_not_gang_members_of_each_other`,
`a_row_with_a_root_but_no_identity_is_never_a_member`) and the rest stay
green — the predicate is load-bearing, and the inclusion oracles alone
would not have caught its absence (they are the control, not the test).

**What the oracles exclude.** A Windows drive-letter root (the parser
accepts one; no fixture here runs on Windows). Two hosts whose local roots
canonicalise to the same path on UNSHARED filesystems — indistinguishable
to this predicate by design (necessary, never sufficient; sufficiency is the
attestation VERIFY). Postgres arms ran against a local PostgreSQL 16 in the
CI lane's shape (§6).

## 4. U5b-0 — design and oracles (stated BEFORE the code)

- `LeafDigest { index: u32, digest: ArtifactDigest }`;
  `MaterializationManifest.leaves: Vec<LeafDigest>` (a REQUIRED field);
  `MaterializationManifest.artifact` becomes `ArtifactDigest::fold(&leaves)`
  — ONE fold function over the leaf digests in index order, never a second
  independently computed whole-artifact digest.
- Result table: one leaf per Parquet row group, its digest the SHA-256 of
  that row group's byte range in the file (from the footer's column-chunk
  offsets), computed by the writer as each row group is written. Model
  bundle (`ArtifactStore::write_model_materialization`): one leaf per file
  in the bundle manifest's name-sorted order, its digest the file's own
  sha256 the bundle manifest already records.
- `MANIFEST_VERSION` 3 → 4. An old sidecar (no `leaves`, version 3) is
  rejected by `from_json_bytes` (serde-first, then the version guard), and
  every reader that consults a manifest treats that rejection as a MISS —
  re-materialise — never a whole-artifact read accepted in its place.
- `verify_materialization` recomputes the leaves from the artifact bytes +
  footer and folds; the freshness reader's `CurrentAnchor::ResultDigest`
  keeps reading `manifest.artifact` (now the fold).

| Property | Oracle (to be written RED first, then GREEN) |
|---|---|
| P-B1 leaf count == row-group count (the footer's own count, read independently by the test) over a multi-row-group fixture | `store::manifest` or `tests/it/materialization.rs::leaf_count_equals_the_footer_row_group_count` |
| P-B2 `artifact == fold(leaves)`, the fold recomputed independently by the test | `::artifact_digest_is_the_fold_over_the_leaves` |
| P-B3 a v3 sidecar (no `leaves`) is a typed rejection in `from_json_bytes` and a MISS in the freshness/probe reader — never a hit | `::a_pre_leaves_sidecar_is_a_cache_miss_not_a_whole_artifact_hit` |
| P-B4 partition property: flipping one byte inside row group k changes leaf k's digest and no other leaf's; `verify_materialization` names row group k | `::a_corrupted_row_group_is_named_by_its_leaf` |
| P-B5 a model bundle's leaves are its files, name-sorted, and `artifact` is their fold; `write_model_materialization` and `read_model_materialization` round-trip | `::a_model_bundle_attests_one_leaf_per_file` |

Mutation to execute before closing: make the writer emit ONE leaf for the
whole file → P-B1 and P-B4 must go red; make `artifact` an independent
whole-file digest → P-B2 must go red.

## 5. Also on this branch

- `ci/scripts/merge_path.sh` (§1.3). Validated by running its `guards` and
  `swarm` stages on this tree (`scratchpad/logs/mp-guards-swarm.log`) and
  the full script before the PR.
- `.github/workflows/ci.yml` `test-live`: `safe.directory` (§1.4). Verified
  only by the next main run of that lane (it does not run on PRs); the fix
  is the same step the hermetic job carries at its own `git` sites.

## 6. Gates (the merge path)

`bash ci/scripts/merge_path.sh` on the final tip, with `JAMMI_TEST_PG_URL`
pointing at a local PostgreSQL 16 in the CI lane's shape; the phase-5 oracle
dispatched after every other stage is green; only `docs/rigor/**` committed
after it.
