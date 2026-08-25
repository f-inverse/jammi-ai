//! Runtime admission scaffolding shared by every fused op's call site: a
//! CUDA compute-capability probe, per-op fused/eager dispatch counters, a
//! log-once-per-process WARN helper, and a `Strict` mode that turns a failed
//! domain check into a hard error instead of a silent fallback.
//!
//! This module contains no fusion POLICY (no op decides here whether it
//! *should* fuse) — it is the shared mechanism a call site's own domain
//! check (dtype, shape, contiguity, device capability) reports its outcome
//! through, so every op's fallback is observable the same way.
//!
//! ## The op-keyed dispatch-counter registry
//!
//! [`device_is_supported`] and [`admission_mode`] are the CANONICAL home for
//! two predicates every fused op's call site needs (moved here from
//! `jammi-encoders::layer_norm` by the C6 commit, which found them
//! duplicated/reached-through-`crate::layer_norm::` by every one of C2-C5's
//! four ops): `jammi-encoders` re-exports both names from `crate::layer_norm`
//! so its existing call sites (`crate::layer_norm::admission_mode()`, etc.)
//! keep compiling unchanged.
//!
//! [`counters_for`] generalizes the OTHER half of the duplication: C2-C5
//! each hand-declared their own `pub(crate) static X_DISPATCH_COUNTERS:
//! DispatchCounters = DispatchCounters::new();` in their own module, one per
//! op. A NEW fused op (this crate's or a downstream crate's) does not need
//! to repeat that — it calls `counters_for("its_op_name")` and gets back a
//! `&'static DispatchCounters` looked up (or, on first use, lazily created
//! and leaked) from ONE process-wide, op-keyed table. This is additive: the
//! four existing per-op statics are left as they are (a live, working,
//! independently-tested mechanism — migrating them is a separate, higher-
//! blast-radius change this commit does not make), but every op added after
//! this one — starting with the LoRA epilogue's `"lora_epilogue"` counters —
//! uses the registry instead of adding a fifth hand-declared static.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use candle_core::Device;

use crate::error::{KernelError, Result};

/// Minimum CUDA compute capability the fused kernels require: bf16 tensor
/// cores need Ampere (`sm_80`) or newer. Below this, a call site's domain
/// check fails and `admit` records/reports the eager fallback.
pub const MIN_CUDA_COMPUTE_CAP: (usize, usize) = (8, 0);

/// A CUDA device's reported compute capability, as `(major, minor)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComputeCapability {
    pub major: usize,
    pub minor: usize,
}

impl ComputeCapability {
    pub fn new(major: usize, minor: usize) -> Self {
        Self { major, minor }
    }

    /// Whether this capability meets [`MIN_CUDA_COMPUTE_CAP`].
    pub fn meets_minimum(&self) -> bool {
        (self.major, self.minor) >= MIN_CUDA_COMPUTE_CAP
    }
}

/// Probes the compute capability of a candle [`candle_core::Device`], if it
/// is a CUDA device this build can query.
///
/// Returns `None` for a non-CUDA device (CPU / Metal — the capability
/// predicate simply does not apply, not a failure) and, without the `cuda`
/// feature, for every device (there is no CUDA runtime to query). A query
/// failure on an actual CUDA device (feature on, driver call errors) also
/// degrades to `None` — the safe default under uncertainty is "capability
/// not established", which every call site treats as "does not meet the
/// minimum" and falls back accordingly (family D: default to the side that
/// cannot silently compute a wrong number).
///
/// This reads the compute capability off the CONTEXT candle's own
/// `CudaDevice` already holds (`CudaDevice::cuda_stream().context()`)
/// rather than constructing a fresh `CudaContext::new(ordinal)`: the
/// latter retains the device's primary context and — per cudarc's own
/// docs, "All safe apis call `CudaContext::bind_to_thread()` before doing
/// work in a certain context" — binds the CALLING THREAD to it as a side
/// effect, which is not something a mere capability "probe" should do.
/// Reusing the `Arc<CudaContext>` candle already owns has no such effect:
/// it borrows a handle that is already bound/retained for the device
/// candle is using, rather than retaining/binding a new one.
pub fn probe_cuda_compute_capability(device: &candle_core::Device) -> Option<ComputeCapability> {
    #[cfg(feature = "cuda")]
    {
        if let candle_core::Device::Cuda(cuda_device) = device {
            if let Ok((major, minor)) = cuda_device.cuda_stream().context().compute_capability() {
                return Some(ComputeCapability::new(
                    major.max(0) as usize,
                    minor.max(0) as usize,
                ));
            }
        }
        None
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        None
    }
}

/// Whether a fused-op call site falls back to the eager composition when
/// its domain check fails, or treats the failure as a hard error.
///
/// The bench tier and a capability-check lane run `Strict` so "fell back
/// everywhere" can never pass as a green measurement of the fused path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AdmissionMode {
    #[default]
    Fallback,
    Strict,
}

/// The outcome of one fused-op call site's domain check: did the fused
/// kernel run, or did the eager composition?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOutcome {
    Fused,
    Eager,
}

/// A read-only snapshot of [`DispatchCounters`], cheap to copy and log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DispatchSnapshot {
    pub fused: u64,
    pub eager: u64,
}

/// Per-op fused/eager dispatch counters. Atomics so any thread can record a
/// dispatch outcome without a lock; a run's durable job record reads
/// [`Self::snapshot`] to state which kernels actually executed.
#[derive(Debug, Default)]
pub struct DispatchCounters {
    fused: AtomicU64,
    eager: AtomicU64,
}

impl DispatchCounters {
    pub const fn new() -> Self {
        Self {
            fused: AtomicU64::new(0),
            eager: AtomicU64::new(0),
        }
    }

    pub fn record(&self, outcome: DispatchOutcome) {
        let counter = match outcome {
            DispatchOutcome::Fused => &self.fused,
            DispatchOutcome::Eager => &self.eager,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> DispatchSnapshot {
        DispatchSnapshot {
            fused: self.fused.load(Ordering::Relaxed),
            eager: self.eager.load(Ordering::Relaxed),
        }
    }
}

/// Emits a `tracing::warn!` at most once per process for a given
/// `(op, predicate)` pair — the log-once-per-process WARN naming the op AND
/// the failed predicate.
pub fn warn_fallback_once(op: &'static str, predicate: &'static str) {
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, &'static str)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if seen.insert((op, predicate)) {
        tracing::warn!(
            op,
            predicate,
            "fused-kernel domain check failed; falling back to the eager composition"
        );
    }
}

/// Applies one fused-op call site's admission decision: records the
/// dispatch outcome, and in `Strict` mode turns a failed predicate into a
/// typed error instead of a silent fallback.
///
/// `predicate_holds` is the call site's own domain check (already evaluated
/// — this function does not know what the predicate means, only whether it
/// held); `predicate_name` is what gets logged/erred on failure.
pub fn admit(
    mode: AdmissionMode,
    op: &'static str,
    predicate_name: &'static str,
    predicate_holds: bool,
    counters: &DispatchCounters,
) -> Result<DispatchOutcome> {
    if predicate_holds {
        counters.record(DispatchOutcome::Fused);
        return Ok(DispatchOutcome::Fused);
    }
    counters.record(DispatchOutcome::Eager);
    match mode {
        AdmissionMode::Fallback => {
            warn_fallback_once(op, predicate_name);
            Ok(DispatchOutcome::Eager)
        }
        AdmissionMode::Strict => Err(KernelError::StrictModeFallback {
            op,
            predicate: predicate_name,
        }),
    }
}

/// `Strict` mode (an explicit fused-path request errors instead of falling
/// back) is switched on by the `JAMMI_KERNELS_STRICT` environment variable,
/// read once per process — the bench tier and a `gpu_capability`-style lane
/// are the intended callers, set before the process starts so "fell back
/// everywhere" can never read as a green measurement of the fused path (K2,
/// scope decision 6 of the fused-kernels plan). `Fallback` (the default) is
/// what every ordinary training run uses.
///
/// ONE env var governs strictness uniformly across every fused kernel every
/// crate in this workspace dispatches through `admit`, rather than one env
/// var per op or per crate — moved here (from `jammi-encoders::layer_norm`,
/// where C2-C5 all reached it via `crate::layer_norm::admission_mode()`) so
/// a crate with no dependency on `jammi-encoders` at all (`jammi-lora`) can
/// read the exact same switch.
pub fn admission_mode() -> AdmissionMode {
    static MODE: OnceLock<AdmissionMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        if std::env::var_os("JAMMI_KERNELS_STRICT").is_some() {
            AdmissionMode::Strict
        } else {
            AdmissionMode::Fallback
        }
    })
}

/// Whether `d` is a device every fused CPU/CUDA-backed op in this crate can
/// actually run on: CPU always, and CUDA only when THIS BUILD compiled
/// jammi-kernels' `cuda` feature (`cfg!(feature = "cuda")`).
///
/// Metal is refused unconditionally — no op in this crate has a `metal_fwd`,
/// and candle's default `metal_fwd` ERRORS rather than falling back, so a
/// Metal tensor reaching `apply2`/`apply3` would turn a working eager
/// forward into a hard error rather than a clean fallback; refusing it here,
/// before the tensor ever reaches an op, is what keeps the fallback clean.
///
/// The `cfg!(feature = "cuda")` half exists for a narrower reason than
/// Metal's: candle's `CustomOp2::cuda_fwd` ALSO has a default impl (a typed
/// `Err`, not a panic), so a CUDA tensor reaching `apply2` while this
/// crate's own `cuda` feature is OFF (e.g. some other crate in the same
/// workspace build enabled `candle-core/cuda` via feature unification,
/// without going through this crate's `cuda` feature) would still fail
/// SAFELY today — but `cfg!(feature = "cuda")` makes that structurally
/// impossible rather than merely unreached, at zero runtime cost (the whole
/// expression folds to a compile-time constant).
///
/// Moved here from `jammi-encoders::layer_norm::device_is_supported` (the
/// C6 commit): that crate now re-exports this function under its old path
/// so `crate::layer_norm::device_is_supported(..)` call sites in
/// `jammi-encoders` (including `crate::modernbert`'s RoPE/softmax/GeGLU
/// admission predicates) keep compiling unchanged, and `jammi-lora` — which
/// has no dependency on `jammi-encoders` at all — reaches the identical,
/// once-audited clause directly.
pub fn device_is_supported(d: &Device) -> bool {
    d.is_cpu() || (cfg!(feature = "cuda") && d.is_cuda())
}

/// The op-keyed dispatch-counter registry: one process-wide table from an
/// op's name to its `DispatchCounters`. See the module doc's "op-keyed
/// dispatch-counter registry" section for why this exists alongside (not
/// instead of) the four hand-declared per-op statics C2-C5 already shipped.
///
/// Looks up `op`'s counters, creating (and leaking — a `'static` handle,
/// same lifetime class as a hand-declared `static`, is the whole point) a
/// fresh zeroed `DispatchCounters` the first time a given `op` name is seen.
/// Every subsequent call with the SAME `op` string returns the SAME
/// counters — `admit`'s `Relaxed` atomics accumulate across the process's
/// lifetime exactly as a hand-declared `static DispatchCounters` would.
///
/// `op` is `&'static str` (a string literal at every call site in this
/// codebase — the op's own compile-time name, e.g. `"lora_epilogue"`), so
/// the registry can hand back a genuine `&'static DispatchCounters` without
/// any unsafe lifetime extension.
pub fn counters_for(op: &'static str) -> &'static DispatchCounters {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry(op)
        .or_insert_with(|| Box::leak(Box::new(DispatchCounters::new())))
}

/// The registry [`counters_for`] reads/writes — split out so [`snapshot_all`]
/// can take the SAME lock rather than maintaining a second table.
fn registry() -> &'static Mutex<HashMap<&'static str, &'static DispatchCounters>> {
    static REGISTRY: OnceLock<Mutex<HashMap<&'static str, &'static DispatchCounters>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A snapshot of every op currently registered in [`counters_for`]'s
/// table, keyed by op name — the read API a durable job record or a bench
/// report uses to state which kernel paths ran during a measured run
/// WITHOUT needing to know each op's name ahead of time (unlike a
/// per-op `*_dispatch_snapshot()` function, which does). `BTreeMap` (not
/// `HashMap`): a deterministic iteration order for anything that logs or
/// serializes this snapshot (family J — hashmap iteration order is not a
/// fold order this codebase relies on for a durable artifact).
///
/// Only reflects ops that have been looked up via [`counters_for`] at
/// least once (an op with zero forwards taken through it — e.g. a code
/// path never reached in this run — is simply ABSENT, not present with
/// zero counts); a caller that needs to distinguish "never registered"
/// from "registered but never dispatched" should track that separately.
pub fn snapshot_all() -> std::collections::BTreeMap<&'static str, DispatchSnapshot> {
    registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .map(|(&op, counters)| (op, counters.snapshot()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_capability_meets_minimum_is_lexicographic() {
        assert!(ComputeCapability::new(8, 0).meets_minimum());
        assert!(ComputeCapability::new(9, 0).meets_minimum());
        assert!(ComputeCapability::new(8, 6).meets_minimum());
        assert!(!ComputeCapability::new(7, 5).meets_minimum());
        assert!(!ComputeCapability::new(0, 0).meets_minimum());
    }

    #[test]
    fn probe_on_cpu_device_is_not_applicable() {
        // The predicate does not apply to a non-CUDA device; `None` is the
        // documented "not applicable", not a probe failure.
        assert_eq!(
            probe_cuda_compute_capability(&candle_core::Device::Cpu),
            None
        );
    }

    #[test]
    fn admit_fallback_mode_records_and_never_errors() {
        let counters = DispatchCounters::new();
        let outcome = admit(
            AdmissionMode::Fallback,
            "test_op",
            "always_false",
            false,
            &counters,
        )
        .expect("Fallback mode never errors");
        assert_eq!(outcome, DispatchOutcome::Eager);
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }

    #[test]
    fn admit_strict_mode_errors_on_a_failed_predicate() {
        let counters = DispatchCounters::new();
        let err = admit(
            AdmissionMode::Strict,
            "test_op",
            "always_false",
            false,
            &counters,
        )
        .expect_err("Strict mode must error, never silently fall back");
        assert!(matches!(
            err,
            KernelError::StrictModeFallback {
                op: "test_op",
                predicate: "always_false"
            }
        ));
        // STRICT still records the attempted (eager) outcome — the
        // dispatch-count provenance is not skipped just because the call
        // errors.
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }

    #[test]
    fn admit_records_fused_when_predicate_holds() {
        let counters = DispatchCounters::new();
        let outcome = admit(
            AdmissionMode::Strict,
            "test_op",
            "always_true",
            true,
            &counters,
        )
        .expect("a satisfied predicate never errors, in either mode");
        assert_eq!(outcome, DispatchOutcome::Fused);
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 1, eager: 0 });
    }

    #[test]
    fn warn_fallback_once_is_idempotent_per_process() {
        // Not directly observable from outside (the log line itself isn't
        // captured here), but calling it twice with the same key must not
        // panic and the internal set must not grow unbounded per call —
        // exercised for the side-effect-free contract, not the log output.
        warn_fallback_once("dedup_test_op", "dedup_predicate");
        warn_fallback_once("dedup_test_op", "dedup_predicate");
    }

    #[test]
    fn device_is_supported_rejects_metal() {
        // No `metal` feature exists on this crate at all — the predicate
        // must reject `Device::Metal` structurally, not via a cfg this
        // crate doesn't even define. Mirrors
        // `jammi_encoders::layer_norm::tests::device_is_supported_rejects_metal`.
        let metal = Device::Metal(candle_core::MetalDevice);
        assert!(!device_is_supported(&metal));
        // CPU is unconditionally supported regardless of build features.
        assert!(device_is_supported(&Device::Cpu));
    }

    #[test]
    fn admission_mode_defaults_to_fallback_without_the_env_var() {
        // `admission_mode` memoizes into a process-wide `OnceLock`, so this
        // only asserts the DEFAULT value observed by a fresh process (no
        // other test in this binary sets `JAMMI_KERNELS_STRICT` before this
        // one runs — `cargo test`'s default per-test-thread model still
        // shares one process-wide env and one `OnceLock`, so this is a
        // documentation-level assertion about the default, not a hermetic
        // unit test of the env-var branch itself).
        if std::env::var_os("JAMMI_KERNELS_STRICT").is_none() {
            assert_eq!(admission_mode(), AdmissionMode::Fallback);
        }
    }

    #[test]
    fn counters_for_returns_the_same_static_instance_for_the_same_op_name() {
        let a = counters_for("registry_test_op_a");
        let b = counters_for("registry_test_op_a");
        // Same `op` name -> same underlying `DispatchCounters`: a record
        // through one handle is visible through the other.
        a.record(DispatchOutcome::Fused);
        assert_eq!(b.snapshot(), DispatchSnapshot { fused: 1, eager: 0 });
        assert!(std::ptr::eq(a, b), "must be the identical instance");
    }

    #[test]
    fn snapshot_all_contains_every_registered_op_name() {
        // Distinct, test-local op names so this assertion is not racy
        // under `cargo test`'s parallel execution (other tests in this
        // binary register their OWN op names into the same process-wide
        // registry — additive-only, never removed, so checking for
        // PRESENCE of these specific keys is safe; asserting the full map
        // is exactly these entries would not be).
        let a = counters_for("registry_test_op_snapshot_all_a");
        let b = counters_for("registry_test_op_snapshot_all_b");
        a.record(DispatchOutcome::Fused);
        b.record(DispatchOutcome::Eager);
        b.record(DispatchOutcome::Eager);

        let all = snapshot_all();
        assert_eq!(
            all.get("registry_test_op_snapshot_all_a"),
            Some(&DispatchSnapshot { fused: 1, eager: 0 })
        );
        assert_eq!(
            all.get("registry_test_op_snapshot_all_b"),
            Some(&DispatchSnapshot { fused: 0, eager: 2 })
        );
        // Non-vacuity: an op name never passed to `counters_for` must be
        // absent, not present-with-zeros.
        assert!(!all.contains_key("registry_test_op_never_registered"));
    }

    #[test]
    fn counters_for_keys_by_op_name_not_by_call_site() {
        // Two DIFFERENT op names never share counters, even though both
        // route through the same registry.
        let a = counters_for("registry_test_op_b1");
        let b = counters_for("registry_test_op_b2");
        a.record(DispatchOutcome::Fused);
        b.record(DispatchOutcome::Eager);
        assert_eq!(a.snapshot(), DispatchSnapshot { fused: 1, eager: 0 });
        assert_eq!(b.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }
}
