mod api_freeze;
mod common;
mod composability_seam;
mod flight_annotate;
mod flight_tenant;
mod grpc_byo_auth;
mod grpc_embedding;
// The client/server-topology GPU proof: compiled only under `live-gpu-tests`,
// a meaningful run also needs `cuda` + a visible GPU (it skips otherwise).
#[cfg(feature = "live-gpu-tests")]
mod grpc_embedding_gpu;
mod grpc_eval;
mod grpc_inference;
mod grpc_introspection;
mod grpc_mutable_topic_audit;
mod grpc_pipeline;
mod grpc_remote_compute;
mod grpc_remote_list;
mod grpc_remote_session;
// K4 transport-only DEVICE leg (unit 62 / CONTRACT.md E5) — GPU coverage of
// grpc_remote_session.rs's CPU bitwise remote-vs-local assertion. Same gating
// as grpc_embedding_gpu above: compiled only under `live-gpu-tests`, skips
// cleanly without a visible GPU.
#[cfg(feature = "live-gpu-tests")]
mod grpc_remote_session_gpu;
mod grpc_session;
mod grpc_tracing_span;
mod grpc_training;
mod grpc_trigger;
mod grpc_web;
mod health;
mod mount_tenant_scoped;
mod serve_bind_race;
mod serve_e2e;
mod server;
mod service_tiers;
mod tenant_isolation_oracle;
mod uat_shapes;
