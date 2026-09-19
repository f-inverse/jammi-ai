mod arity_guard;
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
mod compute_repo;
mod concurrent_writers;
mod datafusion_version;
mod docs_config_fences;
mod domain_hash_prefix_free_gate;
mod ephemeral;
mod eval_per_query;
mod exact_search;
mod foundation;
mod freshness;
mod gang_instance_freshness;
// Every member root in this file is built through `MemberRoot::new`, the
// arbitrary-string fixture constructor, which exists only under `test-hooks`
// (the production constructor, `MemberRoot::resolved`, is config-only).
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
mod model_artifacts;
mod model_finalize;
mod model_lifecycle;
mod model_reuse;
mod models_delete_call_sites;
#[cfg(feature = "test-hooks")]
mod mutable_crash_recovery;
mod mutable_federation;
mod mutable_tables;
mod read_vectors;
mod reconcile;
mod reconcile_artifacts;
mod recovery;
mod register_computed_embedding;
mod registry_read_only;
#[cfg(feature = "test-hooks")]
mod rendezvous_ring;
mod result_tables;
mod segment;
mod serde_json_preserve_order;
mod shipped_feature_exposure;
mod sources;
mod sqlite_cross_session_visibility;
mod sqlite_foreign_library;
mod sqlite_single_process_seam;
mod sqlite_two_pool_writers;
mod storage_cloud;
mod store;
mod tenant_scope;
mod terminality_source_gate;
mod trigger;
#[cfg(feature = "live-broker-tests")]
mod trigger_jetstream;
mod trigger_multi_replica_offsets;
mod trigger_replay_column_types;
mod trigger_replay_row_order;
