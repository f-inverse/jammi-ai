//! The served proofs that need CUDA: compiled only under `live-gpu-tests`
//! (device 0) — the topology proofs additionally under
//! `live-gpu-gang-tests` (devices 0 and 1, and NCCL) — and selected on a GPU
//! host by this module's path.

mod grpc_embedding;
mod grpc_remote_session;
#[cfg(feature = "live-gpu-gang-tests")]
mod topology;
