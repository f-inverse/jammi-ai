//! Gated, multi-process distributed-validation lane for the durable
//! training-worker fleet.
//!
//! This harness proves the fleet-safety claims under *real* distribution: N
//! worker **processes** (`jammi serve`, train tier) claiming from a **shared**
//! Postgres catalog and a **shared** object store (MinIO) under lease-based
//! reclaim. Unlike the in-process `it::fine_tune` durability tests — which drive
//! one `TrainingWorker` per `tokio` task against a local SQLite catalog and a
//! `file://` artifact root — every assertion here crosses a process boundary and
//! a network round-trip, so the catalog's `FOR UPDATE SKIP LOCKED` claim, the
//! lease heartbeat/reclaim, the finalize compare-and-set, and the
//! content-addressed object-store artifact path are all exercised as a deployed
//! fleet would exercise them.
//!
//! The lane is **off by default**: it compiles and runs only under the
//! `live-distributed-tests` cargo feature, and every test early-returns (with a
//! `tracing::warn`, never `#[ignore]`) when the shared backends are not
//! configured. The driving env vars are:
//!
//! - `JAMMI_TEST_PG_URL`        — the shared Postgres catalog URL.
//! - `JAMMI_TEST_S3_ENDPOINT`   — the MinIO S3 endpoint (e.g. `http://127.0.0.1:9000`).
//! - `JAMMI_TEST_S3_BUCKET`     — a pre-created bucket the lane roots artifacts under.
//! - `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` — MinIO credentials.
//! - `AWS_REGION`               — optional; defaults to `us-east-1` for MinIO.
//!
//! The four properties live in their own modules; `harness` holds the shared
//! spawn / poll / teardown machinery they all build on.

mod harness;

mod artifact_crash_window;
mod cross_tenant_isolation;
mod exactly_one_claim;
mod kill9_reclaim;
