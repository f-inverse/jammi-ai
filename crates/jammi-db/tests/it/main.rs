mod assembly_outcome;
mod audit;
mod backup_recipe;
mod bans_names_resolve;
mod broker_parity;
mod broker_postgres_config;
mod caching;
mod catalog_ping;
mod channels;
mod common;
mod concurrent_writers;
mod datafusion_version;
mod docs_config_fences;
mod ephemeral;
mod esc_071_cross_session_visibility;
mod esc_072_two_pool_writers;
mod esc_073_foreign_sqlite_library;
mod esc_099_multi_replica_offset_collision;
mod esc_100_lossy_replay_types;
mod esc_101_intra_batch_row_order;
mod eval_per_query;
mod exact_search;
mod foundation;
mod freshness;
mod gang_instance_freshness;
// `MemberRoot::new` (the arbitrary-string fixture constructor this file's
// helpers build every member root through) only exists under
// `feature = "test-hooks"` (P-X4: the production constructor is
// `MemberRoot::resolved`, config-only) — CI's "test-hooks lane"
// (`.github/workflows/ci.yml`'s `Run tests (test-hooks lane)` step) compiles
// and runs this whole file on every PR; the plain `cargo test --workspace`
// step does not include it, same as `materialization_crash_recovery`/
// `mutable_crash_recovery` below.
#[cfg(feature = "test-hooks")]
mod gang_membership;
mod gang_rank_admission;
mod index;
mod jobs_queue;
mod lease_keeper;
mod masked_read;
mod materialization;
#[cfg(feature = "test-hooks")]
mod materialization_crash_recovery;
mod member_root_constructor;
mod memory_pool;
mod migrations;
mod model_lifecycle;
#[cfg(feature = "test-hooks")]
mod mutable_crash_recovery;
mod mutable_federation;
mod mutable_tables;
mod read_vectors;
mod reconcile;
mod recovery;
mod register_computed_embedding;
mod result_tables;
mod segment;
mod serde_json_preserve_order;
mod shipped_feature_exposure;
mod sources;
mod sqlite_single_process_seam;
mod storage_cloud;
mod store;
mod tenant_scope;
mod trigger;
#[cfg(feature = "live-broker-tests")]
mod trigger_jetstream;
mod whose_fault_gate;
