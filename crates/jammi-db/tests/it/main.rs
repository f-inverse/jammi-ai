mod audit;
mod caching;
mod catalog_ping;
mod channels;
mod common;
mod concurrent_writers;
mod ephemeral;
mod esc_071_cross_session_visibility;
mod esc_072_two_pool_writers;
mod esc_073_foreign_sqlite_library;
mod eval_per_query;
mod exact_search;
mod fine_tune_queue;
mod foundation;
mod freshness;
mod index;
mod materialization;
#[cfg(feature = "test-hooks")]
mod materialization_crash_recovery;
mod migrations;
mod model_lifecycle;
#[cfg(feature = "test-hooks")]
mod mutable_crash_recovery;
mod mutable_federation;
mod mutable_tables;
mod read_vectors;
mod recovery;
mod register_computed_embedding;
mod segment;
mod sources;
mod sqlite_single_process_seam;
mod storage_cloud;
mod store;
mod tenant_scope;
mod trigger;
#[cfg(feature = "live-broker-tests")]
mod trigger_jetstream;
