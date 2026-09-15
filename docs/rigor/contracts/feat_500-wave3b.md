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
4. **The container jobs' git** — `.github/actions/setup-rust-ci` (the
   composite every container job runs right after checkout) now sets the
   `safe.directory` mark ONCE; the five per-job copies in `ci.yml` are
   removed. The `pinned_source_gate` tests shelled out to `git ls-files`
   and died with "dubious ownership" on every main run of the live lane
   (nine on 937d72e6, six on ad2f5b7b), which had no copy; eight container
   jobs had none. Placed in the composite on the pressure round's finding
   (F16), not as a sixth copy.

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

- `RootIdentity::of(root, cloud)` (crate-private) — the derivation, run
  by the process that OWNS the root, at registration, on its own
  filesystem, from the SAME config the store roots itself from
  (`crates/jammi-db/src/catalog/instance.rs`). Rules, by the scheme the
  store's own parser (`StorageUrl::parse`, the one alias table) assigns:
  object stores → `{canonical scheme}://{bucket}/{key}` (the bucket verbatim —
  the store hands it to the driver as spelled and the driver dials it as
  spelled; the fifth oracle round retired the lowercase fold) where
  the key is normalised by the SAME `object_store::path::Path::parse` the
  store hands its keys to (one leading `/` stripped, a trailing one
  dropped, an empty segment refused exactly as the store refuses it — so
  `s3://b//p` ≡ `s3://b/p` and `s3://b/p//` is no root), then
  `@{key=value;…}` — the location determinants READ BACK from the very
  builder the store constructs for that root
  (`storage::location_determinants`: the process environment via
  `from_env()` first, `[storage.cloud]` on top, the order `build_*`
  applies): for S3/R2 the BUCKET ENDPOINT `build()` dials, computed by the
  driver's own expression over inputs read back from the builder — the
  S3-specific endpoint over the generic one whatever spelling set either
  (`AWS_ENDPOINT_URL`, `AWS_ENDPOINT`, `AWS_ENDPOINT_URL_S3`, the config;
  for R2 a stray `AWS_ENDPOINT_URL_S3` overrides the configured endpoint in
  the store and therefore here), dialled verbatim under virtual-hosted
  style or as `endpoint/bucket` path-style (the fifth oracle round: the two
  styles are two key namespaces), the S3 Express zonal host, or the
  regional AWS host (region is part of that URL: two regions for one
  bucket are two identities, a split, never a merge) — the Azure account
  and endpoint — or, in emulator mode,
  the Azurite host `AZURITE_BLOB_STORAGE_URL` object_store reads with a
  bare `std::env::var` outside its key tables (default included) and the
  emulator account — the Fabric switch, the GCS base URL. The identity
  spells exactly the variables the driver spells: none through the key
  tables, and the one bare read the same way; and it reads every VALUE as
  the driver reads it — the boolean parser's `1`/`true`/`on`/`yes`/`y` in
  any case and nothing else (mirrored, since that parser is private, with
  an oracle over every spelling it accepts and several it rejects; a value
  it rejects is one it refuses to build a store on, and takes the false
  arm here), the emulator and Azure account URLs parsed and the S3/R2
  endpoint's trailing slashes trimmed as the driver does before appending
  the bucket. Two oracle rounds found a
  hand-written variable list drifting, a third found the bare read, a
  fourth found the boolean spellings — each closed by mirroring the
  driver's own resolution, not by listing. So nothing the driver honours
  can be missed. In a build without a scheme's storage
  feature the store cannot
  dial that scheme at all and there are no determinants — two buckets of
  one name behind two endpoints or accounts are two locations; local
  roots → the directory is CREATED first (`create_dir_all`, what the store
  does at open, idempotent) and then canonicalised (symlinks, `.`/`..`, the
  filesystem's own spelling — on a case-insensitive filesystem two
  spellings of one directory settle to the on-disk one; a relative root
  against the working directory); any filesystem error is a typed refusal
  naming the root, never a fallback to a different identity; `memory://`
  → refused, typed, naming the root and the way out; an unknown or
  upper-case scheme → refused exactly as the store refuses it.
  The pressure round's five findings on the first cut are each closed by
  one of these rules: the store's key parser (F1), refusal on every
  non-ENOENT error (F2), create-then-canonicalise instead of a lexical tail
  (F3), the endpoint in the identity (F4), and the listing taking the
  member's own `MemberRoot` with the derivation private (F5).
- `MemberRoot { root, identity }` — the verbatim string PAIRED with its
  identity, built only by `MemberRoot::resolved(config)` in production
  (`resolved_result_root()` then `RootIdentity::of`); the `test-hooks`
  `MemberRoot::new(root)` derives the identity the same way and panics on
  an unshareable root (a fixture bug).
- `instances.result_root_identity TEXT` — migration
  `036_instances_result_root_identity`, appended, nullable, pinned at the
  four K5 sites; written by `upsert_instance`/`reregister_instance` from
  the one `MemberRoot`.
- `GangListing.root: &MemberRoot` — the caller's own registration value,
  never a bare identity (an identity exists only inside a `MemberRoot`,
  derived once); `list_gang_members` adds `AND i.result_root_identity =
  $n` to its SQL (a byte-exact `=` on a string this engine wrote; NULL
  never matches). The verbatim `result_root` column is never compared.

## 3. U5b-1a-A2 — properties, each with its executed oracle

| Property | Oracle (all GREEN at this head) |
|---|---|
| P-A1 alias fold and the store's own key normalisation on object stores (a key the store refuses is refused; keys the store equates are equated — the store's parser IS the oracle); bucket case, key case, bucket, backend stay distinct | `catalog::instance::root_identity_tests::object_store_aliases_and_the_stores_key_normalisation_fold`, `::a_key_the_store_refuses_is_refused_and_keys_the_store_equates_are_equated`, `::object_store_key_case_buckets_and_backends_stay_distinct` |
| P-A1b the location determinants the store's builder dials are part of a cloud identity, read back from that builder: for S3/R2 the bucket endpoint `build()` dials — every endpoint spelling object_store accepts (`AWS_ENDPOINT_URL`, `AWS_ENDPOINT`, `AWS_ENDPOINT_URL_S3` — the S3-specific one winning as `build()` dials it), virtual-hosted vs path style, S3 Express (a bucket without a zone suffix refused as the driver refuses), the region, config on top of the environment, an empty value unset; R2's configured endpoint and the stray `AWS_ENDPOINT_URL_S3` override; Azure account, endpoint (both spellings), the Fabric switch, and in emulator mode the Azurite host (`AZURITE_BLOB_STORAGE_URL`, its default, URL-parsed, the endpoint ignored as `build()` ignores it) — both switches read through every spelling the driver's boolean parser accepts and none it rejects; the GCS base URL | `catalog::instance::root_identity_tests::the_endpoint_the_store_would_dial_is_part_of_a_cloud_identity`, `::every_endpoint_spelling_the_store_honours_is_part_of_the_identity` (with `--features storage-cloud`); `storage::builder::tests::s3_determinants_are_the_bucket_endpoint_the_builder_dials`, `::r2_determinants_are_the_configured_endpoint_unless_the_s3_env_url_overrides_it`, `::azure_determinants_carry_account_endpoint_emulator_and_fabric_as_the_builder_reads_them`, `::gcs_determinants_carry_the_base_url_the_builder_reads` — run by the new `storage-cloud` step of ci.yml's hermetic job |
| P-A2 local roots: symlink ≡ target, `.`/`..`, trailing slash, `file://` ≡ bare, relative against cwd; the root is CREATED and its identity is stable afterwards, a case-divergent spelling settles to the on-disk one on a case-insensitive filesystem (two directories on a case-sensitive one); a root that cannot be created (a FILE where a directory is needed) is refused naming it; distinct dirs distinct | `::a_local_root_folds_symlinks_dot_segments_trailing_slashes_and_the_file_scheme`, `::a_local_root_is_created_and_a_case_divergent_spelling_settles_to_the_on_disk_one`, `::a_local_root_that_cannot_be_created_is_refused_naming_it`, `::a_relative_local_root_is_taken_against_the_working_directory`, `::distinct_local_roots_stay_distinct` |
| P-A3 `memory://` and an unknown scheme refused naming the root | `::a_memory_root_and_an_unknown_scheme_are_refused_naming_the_root`; `config::tests::from_config_refuses_a_memory_result_root_for_a_member_only` (a library config with the same root is untouched; an upper-case scheme is refused on both paths) |
| P-A4 the row carries the verbatim spelling AND `RootIdentity::of` of it, over every advertising arm | `config::tests::from_config_member_root_is_resolved_result_root_verbatim_over_every_arm`, `::from_config_root_identity_is_of_the_resolved_root_over_every_arm`, `::from_config_never_aliases_gcs_and_gs_result_root_spellings_but_their_identities_are_one`; real sessions: `crates/jammi-ai/tests/it/storage_root.rs::member_row_matches_resolved_root_*` (identity column asserted), `::a_member_with_a_memory_result_root_is_refused_at_session_construction` |
| P-A5 the predicate: same-location spellings ARE members (gcs/gs; symlink/target; authority case; trailing slash); different locations and NULL identities are NOT | `crates/jammi-db/tests/it/gang_membership.rs::gcs_and_gs_spelled_members_are_gang_members_of_each_other`, `::a_symlinked_local_root_and_its_target_are_the_same_gang`, `::list_folds_a_trailing_slash_but_neither_bucket_nor_key_case`, `::file_and_s3_rooted_members_are_not_gang_members_of_each_other`, `::list_excludes_a_member_with_peer_addr_set_and_no_root`, `::a_row_with_a_root_but_no_identity_is_never_a_member` (sqlite + postgres arms) |
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
attestation VERIFY). A permission-denied ancestor (EACCES) is refused by
the same arm the ENOTDIR oracle exercises but has no oracle of its own
(CI's lanes run as root, where chmod is bypassed). An Azure
`https://…blob.core.windows.net` spelling (the parser's doc mentions it;
`parse_scheme` accepts only `azure`/`abfss`, so it is refused on both
paths). Postgres arms ran against a local PostgreSQL 16 in the CI lane's
shape (§6).

## 4. U5b-0 — design and oracles (stated before the code, rewritten on the pressure round; BUILT)

The first cut folded the leaves INTO `artifact`. The pressure round showed
that to be wrong at the root: `manifest.artifact` is the base of the
version-identity chain (`store/version.rs`: the base version's identity IS
the artifact hex; every downstream `InputAnchor::result_digest` records it),
`verify_materialization` reuses the whole-object digest as the base
fragment's digest beside `of_bytes` over the deletes object, and a leaf set
over row groups does not partition a Parquet file (footer, page index,
bloom filters, magic belong to no row group — a footer-only mutation would
change no leaf). So:

- `MaterializationManifest.artifact` is UNCHANGED: the whole-object SHA-256,
  the in-toto subject, the root of the version chain, recomputable by any
  verifier holding the bytes.
- `MaterializationManifest.leaves: Vec<LeafDigest>` is ADDITIVE — a keyed
  inventory, `LeafDigest { key: LeafKey, digest: ArtifactDigest }` with
  `LeafKey::RowGroup { index, offset, length }` for a result table (the
  digest of that byte range, read from the footer's column-chunk offsets)
  and `LeafKey::File { name }` for a model bundle (the file's own sha256
  the bundle manifest already records, keyed by NAME so adding a file
  changes no other leaf and `Manifest::combined_hash` stays the bundle's
  content address). The inventory is what a peer verifies one partition
  against without reading the file; it never stands in for `artifact`.
- No `MANIFEST_VERSION` bump: the number is reserved for a change to an
  existing descriptor variant's determinant set, and a since-added
  REQUIRED field is already rejected serde-first. `leaves` is required.
- The MISS rule, scoped: `read_materialization_manifest` maps a SHAPE
  rejection (serde: a sidecar written before `leaves` existed) to
  `Ok(None)` — a pre-`leaves` table reads as a pre-contract table, which
  every reader already handles (`MatchVerdict::MissingManifest`, the
  recompute-from-bytes anchor arm, the cache probe's miss) — while a
  VERSION rejection (`UnsupportedManifestVersion`, a newer engine's
  sidecar) stays an error: an older binary must never re-materialise over
  a newer engine's table.
- `verify_materialization` is unchanged in what it compares; a new
  `verify_partitions(&record) -> PartitionVerdict` recomputes every leaf
  from the bytes and the footer and names the first divergent leaf.

| Property | Oracle (GREEN at this head) |
|---|---|
| P-B1 leaf count == the footer's row-group count, read independently by the test with the `parquet` crate — over a three-row-group object at the unit level, and through the funnel on a real table | `store::manifest::tests::leaves::one_leaf_per_row_group_in_footer_order_each_the_footers_byte_range_digest`; `tests/it/materialization.rs::the_funnel_writes_one_leaf_per_row_group_and_verify_partitions_matches` (sqlite + postgres) |
| P-B2 each leaf's digest == SHA-256 over the byte range the test reads from the footer itself (`ColumnChunkMetaData::byte_range`), not from the leaf | the same unit oracle (the footer is read by the test, the leaf compared to it) |
| P-B3 a sidecar without `leaves` at the current version is `PreLeavesSidecar`, reads as `Ok(None)`, and both verbs say `MissingManifest`; a NEWER version is an error, never a miss; garbage stays a serde error | `store::manifest::tests::leaves::a_pre_leaves_sidecar_is_a_typed_pre_leaves_rejection_and_nothing_else_is`; `tests/it/materialization.rs::a_pre_leaves_sidecar_reads_as_absent_on_both_verbs` |
| P-B4 a byte flipped inside row group k changes leaf k and no other, and `verify_partitions` names leaf k; a FOOTER-only mutation changes no leaf while `verify_materialization` reports `Mismatch` (the whole-object digest is the subject) | `store::manifest::tests::leaves::a_byte_flipped_inside_row_group_k_changes_leaf_k_only_and_a_footer_flip_changes_no_leaf`; `tests/it/materialization.rs::a_corrupted_row_group_is_named_by_its_leaf_and_a_footer_mutation_by_the_artifact` |
| P-B5 a model bundle's leaves are its files by name with the bundle manifest's own sha256, `artifact` is still `combined_hash`, and a bundle with one more file carries every existing leaf unchanged | `store::artifact::tests::write_model_materialization_round_trips_and_folds_the_bundle_digest` |

**Mutations (executed, each restored).** The writer emitting ONE leaf for
the whole file: P-B1 and P-B4's unit oracles red, P-B3 green (as it should
be — it does not depend on the walk). Bundle leaves keyed by position: P-B5
red. A version rejection mapped to a miss: P-B3 red. Every oracle green
again after the restore.

**What the oracles exclude.** A Parquet object whose footer is unreadable
(the walk refuses with `ParquetFooter`; `verify_partitions` propagates it —
the footer-mutation arm accepts either `Match` or that error, since a bit
flip in the metadata may or may not break its decoding). Page indexes and
bloom filters are covered only by the whole-object digest, by design.

## 5. Also on this branch

- `ci/scripts/merge_path.sh` (§1.3). The first cut split each Swarm-gates
  step into lines (so the two multi-line human-amend-only guards were never
  evaluated), substituted two workflow expressions and passed the rest
  through, and ran diff-scoped gates over an empty diff when HEAD was the
  base. Rewritten: each step's `run:` block executes whole, every `${{ }}`
  expression is expanded or the run stops naming it, HEAD == base or a
  dirty tree is a refusal, a missing `mdbook` fails like a missing
  Postgres, and the `ci.yml` jobs it does not cover are printed up front.
  Validation: the `guards` and `swarm` stages run to their summary on the
  committed tree (`scratchpad/logs/mp-guards-swarm-2.log`), and the full
  script before the PR.
- `.github/actions/setup-rust-ci` sets `safe.directory` (§1.4). Verified by
  every container job's next run; the live lane's `pinned_source_gate`
  tests are the observable.

## 6. Gates (the merge path)

`bash ci/scripts/merge_path.sh` on the final tip, with `JAMMI_TEST_PG_URL`
pointing at a local PostgreSQL 16 in the CI lane's shape; the phase-5 oracle
dispatched after every other stage is green; only `docs/rigor/**` committed
after it.
