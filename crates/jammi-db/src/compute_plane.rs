//! Where a materialization's plan runs: on the compute plane when this
//! process holds the client role and a live executor can hold the plan, in
//! this process otherwise.
//!
//! [`ComputePlane`] is the seam: this crate declares it, the compute-plane
//! crate implements it over its submit client, and the server installs it
//! on the session through [`ComputePlaneSlot::install`] when the process is
//! configured as a client — the engine never depends on the implementation.
//! The slot rides in the session's `SessionConfig` as an extension, so it
//! reaches every plan executed under a context derived from the session's —
//! a per-request Flight SQL state, a single-partition derivation — through
//! the `TaskContext` alone, and a context a caller built for itself carries
//! no slot and never routes.

use std::fmt;
use std::sync::{Arc, OnceLock};

use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::ExecutionPlan;
use futures::future::BoxFuture;

use crate::error::Result;
use crate::store::manifest::ComputeDeviceKind;

/// A compute plane a physical plan is submitted to: the plan's stages run
/// on the plane's executors and its output streams back. The plan crosses
/// as it is — the same operators the in-process run would execute —
/// and a failure on the plane reaches the caller as the same typed error
/// the in-process run would raise.
pub trait ComputePlane: Send + Sync {
    /// Submit `plan`, or refuse it typed before any task is scheduled.
    fn submit(&self, plan: Arc<dyn ExecutionPlan>) -> BoxFuture<'static, Result<Submission>>;
}

/// What became of a submission.
pub enum Submission {
    /// Every stage was bound; the stream is the plan's own output.
    Placed(SendableRecordBatchStream),
    /// Refused before any task was scheduled: no live executor can hold the
    /// plan. The plan runs where it was issued instead — never parked on a
    /// scheduler that cannot bind it.
    Unheld(Unheld),
}

/// Why the compute plane cannot hold a plan right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unheld {
    /// No registered executor is live.
    NoLiveExecutor,
    /// No live executor lists a device of the kind the plan requires.
    NoExecutorOfKind(ComputeDeviceKind),
}

impl fmt::Display for Unheld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLiveExecutor => f.write_str("no live registered compute executor"),
            Self::NoExecutorOfKind(kind) => write!(
                f,
                "this plan requires device_kind {kind:?} but no live registered compute \
                 executor lists a {} device",
                kind.wire_str()
            ),
        }
    }
}

/// The session's write-once slot for its [`ComputePlane`]. Empty on a
/// process holding no client role — every materialization runs in-process.
#[derive(Default)]
pub struct ComputePlaneSlot(OnceLock<Arc<dyn ComputePlane>>);

impl fmt::Debug for ComputePlaneSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputePlaneSlot")
            .field("installed", &self.0.get().is_some())
            .finish()
    }
}

impl ComputePlaneSlot {
    /// Install the process's [`ComputePlane`] — once. `false` when one is
    /// already installed (the first stays).
    pub fn install(&self, plane: Arc<dyn ComputePlane>) -> bool {
        self.0.set(plane).is_ok()
    }

    /// The installed plane, if this process holds the client role.
    pub fn plane(&self) -> Option<Arc<dyn ComputePlane>> {
        self.0.get().cloned()
    }
}
