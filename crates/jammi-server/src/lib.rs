//! Library surface for the OSS `jammi-server` binary.
//!
//! The binary's orchestration entry-point is [`runtime::OssServer`].
//! The library also re-exports building blocks (`build_router`,
//! `flight::*`, `grpc::*`) so test fixtures and downstream binaries
//! (e.g. the `jammi` CLI's `serve` subcommand) can compose the same
//! pieces without reimplementing them.
//!
//! For downstream *composition*, [`runtime::assemble_grpc_chain`] returns an
//! [`runtime::AssembledChain`] with the engine's services pre-mounted; a
//! downstream chains [`runtime::AssembledChain::mount`] to add its own services
//! beside the engine's on one listener, then serves — or splits via
//! [`runtime::AssembledChain::into_axum_router`] to compose a single axum
//! listener of its own. `serve_grpc_chain` is the thin OSS-only path over the
//! same seam.

pub mod error;
pub mod flight;
pub mod grpc;
pub mod grpc_web_trailers;
pub mod metrics_layer;
pub mod routes;
pub mod runtime;
pub mod telemetry;
pub mod tenant_resolver_layer;
pub mod tiers;

use std::sync::Arc;

use axum::routing::get;
use axum::Router;

use jammi_db::catalog::topic_repo::TopicRepo;
use jammi_db::trigger::{Publisher, Subscriber};

use crate::error::fallback_handler;
use crate::routes::health::{self, MetricsRegistry};
use crate::runtime::ReadinessProbe;

/// Trigger-stream handles attached to the gRPC server. The caller
/// constructs these once per deployment (sharing one broker, publisher,
/// subscriber, and topic catalog repo across every connection).
///
/// Kept as a public struct because integration tests still wire trigger
/// handles manually for fixtures that need to drive a stubbed broker.
/// Production code goes through [`runtime::OssServer`] which derives
/// the same handles from the engine session.
pub struct TriggerHandles {
    pub topic_repo: Arc<TopicRepo>,
    pub publisher: Arc<Publisher>,
    pub subscriber: Arc<Subscriber>,
}

/// Build the side-channel router with `/healthz` only — no readiness
/// or metrics state attached. Callers that need the full surface
/// construct the router through [`runtime::OssServer`].
pub fn build_router() -> Router {
    Router::new()
        .route("/healthz", get(health::healthz))
        .fallback(fallback_handler)
}

/// Build the full side-channel router exposing `/healthz`, `/readyz`,
/// and `/metrics`. The readiness probe and metrics registry are passed
/// in as `Arc`s so test fixtures can substitute stubs.
pub fn build_health_router(
    readiness: Arc<ReadinessProbe>,
    metrics: Arc<MetricsRegistry>,
) -> Router {
    let readyz = Router::new()
        .route("/readyz", get(health::readyz))
        .with_state(readiness);
    let metrics = Router::new()
        .route("/metrics", get(health::metrics))
        .with_state(metrics);
    Router::new()
        .route("/healthz", get(health::healthz))
        .merge(readyz)
        .merge(metrics)
        .fallback(fallback_handler)
}
