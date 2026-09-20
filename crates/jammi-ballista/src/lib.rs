//! `jammi-ballista` — the Ballista compute plane.
//!
//! Sits between `jammi-ai`/`jammi-db` and `jammi-server`: it encodes jammi's
//! own physical operators across a scheduler/executor boundary
//! ([`codec::JammiCodec`]), adapts Ballista's per-stage execution
//! ([`engine::JammiExecutionEngine`]), hosts the scheduler/executor roles
//! ([`roles`]), and submits a plan to a hosted scheduler
//! ([`client::submit_physical_plan`]). A placed task's typed failure crosses
//! the scheduler as `jammi_wire::TaskErrorEnvelope` — written by the engine
//! (`engine::envelope_task_error`), restored by the client
//! (`client::restore_task_error`) — so a placed refusal classifies exactly
//! as the in-process one. Publishable, lockstep with the rest
//! of the workspace, no cargo feature gates any of this — roles are
//! config-shaped: `jammi-server` depends on this crate unconditionally
//! and decides at runtime, from `[ballista]`, whether a process hosts
//! either role.
//!
//! Dependency direction: this crate depends on `jammi-ai`/`jammi-db`/
//! `jammi-wire`; neither `jammi-ai` nor `jammi-db` depends on it. The
//! `PlacedGangSubmitter`/`PlacedGangRunner` seams `jammi-ai` exposes are
//! INSTALLED by this crate's roles (`roles::host_scheduler`/
//! `roles::host_executor`), never called from `jammi-ai`'s own dependency
//! graph — the same shape `MemberDialer` already uses
//! (`crates/jammi-ai/src/fine_tune/worker.rs`).

pub mod client;
pub mod cluster;
pub mod codec;
pub mod engine;
pub mod error;
pub mod placement;
pub mod roles;
