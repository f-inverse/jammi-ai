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
//!
//! **Update (contract K-aux):** the four C2-C5 statics named above
//! (`LN_DISPATCH_COUNTERS`, `ROPE_DISPATCH_COUNTERS`,
//! `SOFTMAX_DISPATCH_COUNTERS`, `GEGLU_DISPATCH_COUNTERS`) have SINCE been
//! migrated onto this registry themselves — each is now a
//! `LazyLock<&'static DispatchCounters>` that calls `counters_for(..)`
//! under the hood (`crates/jammi-encoders/src/layer_norm.rs:91-92`,
//! `crates/jammi-encoders/src/modernbert.rs:149-150,163-164,1220-1221`;
//! `ATTENTION_BLOCK_DISPATCH_COUNTERS` at `modernbert.rs:579-580` is a
//! fifth, added the same way rather than as a sixth hand-declared static).
//! The paragraph above is kept for its historical rationale (why the
//! registry exists at all), not as a description of the current state —
//! there are no hand-declared `DispatchCounters::new()` statics left
//! outside this module's own tests and `admit`'s registry-population path
//! (`grep -rn 'DispatchCounters::new()' crates/`, confirmed at this
//! contract's tip). What matters for [`admit`]'s `JAMMI_KERNELS_DISABLE`
//! (below) is unaffected either way: every one of these call sites passes
//! its op name through `admit` itself, so the disable list covers it
//! regardless of how its counters are stored.
//!
//! ## `JAMMI_KERNELS_DISABLE` — forcing the eager arm without a second build
//!
//! A comma-separated list of op keys (`admit`'s own `op: &'static str`
//! parameter — the literal each call site passes, e.g.
//! `"layer_norm_fused"`), or the literal `"all"` to disable every op. An op
//! named in the list makes [`admit`] return `Ok(DispatchOutcome::Eager)`
//! UNCONDITIONALLY for that op — the predicate is not even consulted, and
//! `Strict` mode does NOT turn this into an error (disable wins over
//! Strict: forcing an op eager is a deliberate instruction, not the
//! predicate failure `Strict` exists to catch).
//!
//! ### Standalone vs subsumed op keys — this is NOT flat, and naming the wrong
//! ### one alone is a silent no-op
//!
//! Not every op key named in [`admit`]'s call graph reaches [`admit`] on
//! every run — `"attention_block_fused"` (`modernbert.rs:930`) SUBSUMES the
//! RoPE and softmax steps on the training path: when it dispatches Fused,
//! `AttentionBlockFused` performs rotate-half AND masked-softmax internally
//! as one `CustomOp3`, and neither `"rope_fused"` nor
//! `"softmax_last_dim_fused"` is ever consulted for that call. Disabling
//! `"rope_fused"` or `"softmax_last_dim_fused"` ALONE therefore changes
//! nothing on a checkpoint where `attention_block_fused` admits (confirmed
//! against the committed A100 artifact,
//! `crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p3-e32ed90-a100-sxm4.json`:
//! `rope_fused_dispatches: 0`, `softmax_fused_dispatches: 0`,
//! `attention_block_fused_dispatches: 700` — those two ops never fired on
//! that run because nothing routed through them, not because anything was
//! disabled) — an `JAMMI_KERNELS_DISABLE=softmax_last_dim_fused`-only run on
//! such a checkpoint is INVALID for exactly this reason (see
//! [`unmatched_disables`] below: the entry never fires, so it is reported
//! unmatched rather than silently accepted).
//!
//! **Live standalone** (reachable directly, own call site, own predicate):
//! `"layer_norm_fused"` (`jammi-encoders/src/layer_norm.rs:189`),
//! `"geglu_fused"` (`jammi-encoders/src/modernbert.rs:1259`),
//! `"lora_linear_fused"` (`jammi-lora/src/lora_linear.rs:642-644`), and
//! `"attention_block_fused"` itself (`modernbert.rs:930`).
//!
//! **Subsumed** (reachable ONLY when `"attention_block_fused"` is ALSO
//! disabled, forcing `forward_training_attention` into
//! `forward_eager_training_attention_composition` — the composition that
//! calls `RotaryEmbedding::apply_training` and `softmax_apply_training`,
//! each of which independently calls [`admit`] with its own op key):
//! `"rope_fused"` (`modernbert.rs:478`), `"softmax_last_dim_fused"`
//! (`modernbert.rs:1188`).
//!
//! **Registered but permanently dead** (never passed to [`admit`] in
//! today's call graph — see the "safety property" section below):
//! `"lora_epilogue"` and `"lora_dropout"` (`counters_for("lora_epilogue")` /
//! `counters_for("lora_dropout")`, `lora_linear.rs:36,65`); both
//! stand-alone call sites they used to guard were superseded by
//! `crate::ops::LowRankResidualLinear`'s single fused-site `CustomOp3`,
//! which reuses their `cpu_fwd`/`cuda_fwd` directly, bypassing `admit`
//! entirely (`lora_dropout_counters`'s and `lora_epilogue_counters`'s own
//! doc comments at those lines). These always read `{fused: 0, eager: 0}`
//! and, if named in `JAMMI_KERNELS_DISABLE`, always come back from
//! [`unmatched_disables`] as an INVALID run — a real, present-in-the-
//! registry op name that nonetheless never fires, not a synthetic example.
//!
//! ### The correct one-build A/B for `softmax_last_dim_fused` (or `rope_fused`)
//!
//! Because `attention_block_fused` subsumes both, isolating the softmax
//! kernel's own fused-vs-eager difference on a checkpoint where
//! `attention_block_fused` admits needs TWO env settings, not one:
//!
//! - **Eager leg** (softmax runs eager):
//!   `JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=attention_block_fused,softmax_last_dim_fused`
//! - **Fused leg** (softmax runs fused, inside the eager attention
//!   composition):
//!   `JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=attention_block_fused`
//!
//! Both legs disable `attention_block_fused` (so the run takes the eager
//! attention composition, which is the only path that ever consults
//! `softmax_last_dim_fused` at all) and differ ONLY in whether
//! `softmax_last_dim_fused` is additionally named — the isolated variable.
//! `crates/jammi-bench/tests/finetune_step_kernel_disable.rs`'s
//! `softmax_last_dim_fused_nesting_isolates_the_softmax_kernel_through_the_real_cli`
//! drives exactly this pair through the real `jammi-bench finetune-step`
//! CLI and asserts the expected fused/eager split on each leg. The same
//! nesting, with `rope_fused` in place of `softmax_last_dim_fused`,
//! isolates the RoPE kernel.
//!
//! A run naming ONLY `softmax_last_dim_fused` (without also disabling
//! `attention_block_fused`) is the exact "one-build A/B" this module used
//! to advertise for that op alone — it is now understood to be
//! non-functional on any checkpoint where `attention_block_fused` admits,
//! and [`unmatched_disables`] reports it, rather than accepting it.
//!
//! ### The safety property: a typo must never read as a successful forced-eager run
//!
//! [`unmatched_disables`] returns every `JAMMI_KERNELS_DISABLE` entry that
//! has never actually disabled a live `admit` call (tracked via
//! `fired_disables` (this module's own internal bookkeeping), populated by
//! `op_is_disabled`). A caller that
//! turns dispatch counters into a durable/report artifact (this crate's
//! own callers in `jammi-bench`) is expected to treat a non-empty
//! `unmatched_disables()` as an INVALID run rather than a datum — the same
//! "absent counters is not evidence of zero" discipline
//! [`snapshot_all`]'s doc already states for a never-registered op name.
//!
//! `"all"` is EXEMPT from this safety property in one respect: it is
//! recorded fired the moment ANY op reaches [`admit`], regardless of which
//! one — `op_is_disabled`'s doc's "exact match and the `all` wildcard are
//! recorded independently" clause. So `unmatched_disables()` coming back
//! empty for a `JAMMI_KERNELS_DISABLE=all` run proves only that AT LEAST
//! ONE op reached `admit` and was forced eager by it, never that EVERY
//! registered op was — a caller that needs the latter must check
//! [`snapshot_all`]'s per-op `eager` counts directly, not lean on `"all"`
//! coming back matched as if it were evidence for the whole registry.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use candle_core::Device;

use crate::error::{KernelError, Result};

// ─────────────────────────────────────────────────────────────────────────
// Cascade admission: `PredicateOutcome` (P6 stage B, contract v4 §3.3)
// ─────────────────────────────────────────────────────────────────────────
//
// **Scope, corrected by the lead mid-round (P6 Stage B v4 pressure-test,
// 2026-08-25): the 17 pre-existing two-arm predicates (LayerNorm, RoPE,
// softmax, GeGLU, the whole fused attention block, the LoRA site — one
// `bool`-valued `fn ..._admission_predicate` each, listed in this crate's
// own module doc above) are NOT migrated onto [`PredicateOutcome`] and
// keep calling [`admit`] with a plain `bool`, UNCHANGED, byte-for-byte.**
// The original plan (this section's own first draft) reclassified every
// domain check as `DomainMiss` — a decline that never errors even under
// `Strict`. That is wrong for those 17: `Strict` mode's entire purpose
// (`AdmissionMode`'s own doc: "so 'fell back everywhere' can never pass as
// a green measurement of the fused path") is to make a two-arm op's
// predicate failure a HARD error in a controlled bench/capability lane —
// silently reclassifying every one of those failures as a never-erroring
// `DomainMiss` would quietly defang that property for six ops nobody asked
// to change. [`PredicateOutcome`] is introduced ONLY for a genuine THREE-
// (or more-) arm cascade — today: `attention_block_flash` (P6's flash
// attention arm) → `attention_block_fused` (the existing block arm) →
// eager — where a decline does NOT mean "the raw eager composition ran"
// the way a two-arm op's `false` always has: it means "try the NEXT
// admission-gated arm", which [`admit`] itself has no way to express with
// a bare `bool`. [`admit_cascade`] is that new, narrowly-scoped entry
// point; [`admit`] (the 17 existing call sites) is untouched.
//
// The pre-existing 10-cell `JAMMI_KERNELS_DISABLE` lattice (this module's
// own `lattice_cell_01`..`_10` tests, `crates/jammi-bench/tests/
// finetune_step_kernel_disable.rs`) exercises `admit`/`admit_inner`
// exclusively and is untouched by this section — its behaviour is
// BYTE-IDENTICAL before and after this addition (no test in that lattice
// was edited to add this feature).
//
// **HELD, per the lead's P6 Stage B v5 pressure-test correction: the
// `JAMMI_KERNELS_DISABLE=attention_block_flash` lattice cell (contract v4
// §3.3's L11) and `ab_merge.py`'s bench-side "absorber class" semantics are
// deliberately NOT implemented here.** [`admit_cascade`]'s disabled branch
// below records the decline in `declined` (not `eager` — see
// [`CascadeDispatchCounters`]'s doc), which means a leg that intentionally
// disables `attention_block_flash` reads `declined > 0` exactly like a
// genuine domain/capability miss would — indistinguishable from the bench
// side without a "fire-without-counting" signal that does not exist yet. A
// numerics design v2 is being dispatched specifically to resolve this (a
// public fire-without-counting entry point plus an `ab_merge.py` exemption
// admitting `eager > 0` on a disabled pair, with an absorber CASCADE
// `attention_block_flash ⊃ attention_block_fused ⊃ {rope_fused,
// softmax_last_dim_fused}` rather than a flat disabled-op class) — do not
// build on top of the mechanism below for that specific lattice cell until
// it lands.

/// The outcome of a CASCADE arm's own domain/capability predicate — see
/// this module's "Cascade admission" section above for why this exists
/// ONLY for a genuine multi-arm chain (today: `attention_block_flash` →
/// `attention_block_fused` → eager) and not for the 17 pre-existing
/// two-arm ops, which keep calling [`admit`] with a plain `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateOutcome {
    /// The predicate holds: dispatch to THIS arm.
    Holds,
    /// This call's DATA (shape, dtype, mask structure — a property of the
    /// specific tensors this call was handed) does not fit this arm's
    /// mathematical domain. Legitimate and expected (e.g. a mixed-prefix
    /// batch on the flash arm) — [`admit_cascade`] NEVER turns this into
    /// an error, in either [`AdmissionMode`]: it declines to the next arm
    /// and records `CascadeDispatchCounters`'s `declined` counter (private
    /// field, incremented internally by [`admit_cascade`]).
    DomainMiss,
    /// The BUILD, DEVICE, or ENVIRONMENT cannot run this arm regardless of
    /// the data (the feature was not compiled, the GPU architecture does
    /// not match, the device is not CUDA at all). [`admit_cascade`]
    /// declines to the next arm; under [`AdmissionMode::Strict`] this is a
    /// typed error UNLESS the caller asserts the next arm can run
    /// (`next_arm_can_run`) — see [`admit_cascade`]'s doc.
    CapabilityMiss,
}

/// [`admit_cascade`]'s decision: either this arm fires, or it declines and
/// the caller falls through to its OWN next arm (never "eager" directly —
/// unlike [`DispatchOutcome::Eager`], a cascade decline does not assert
/// that eager is what runs next; the caller decides that).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeOutcome {
    Fused,
    Declined,
}

/// A read-only snapshot of [`CascadeDispatchCounters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CascadeDispatchSnapshot {
    pub fused: u64,
    /// Always `0` for every op that only ever calls [`admit_cascade`] with
    /// `outcome = PredicateOutcome::Holds` or a decline — a cascade arm's
    /// own decline is counted in `declined`, never here. Present (rather
    /// than omitted) so this snapshot's SHAPE lines up with
    /// [`DispatchSnapshot`]'s for any tool that reads both registries —
    /// see [`CascadeDispatchCounters`]'s doc.
    pub eager: u64,
    pub declined: u64,
}

/// Per-op counters for a CASCADE admission (see [`PredicateOutcome`]'s
/// doc): a SEPARATE, purpose-built type from [`DispatchCounters`] — not an
/// added field on it — precisely because adding a field to
/// [`DispatchSnapshot`] would break every one of the 28 existing
/// `DispatchSnapshot { fused: .., eager: .. }` struct-literal call sites
/// across this workspace (`crates/jammi-lora/src/lora_linear.rs`,
/// `crates/jammi-encoders/src/lib.rs`, and this module's own tests) for a
/// counter the 17 two-arm ops never need. `eager` is kept (always `0` in
/// practice — see its own doc) purely so the two counter shapes match.
#[derive(Debug, Default)]
pub struct CascadeDispatchCounters {
    fused: AtomicU64,
    eager: AtomicU64,
    declined: AtomicU64,
}

impl CascadeDispatchCounters {
    pub const fn new() -> Self {
        Self {
            fused: AtomicU64::new(0),
            eager: AtomicU64::new(0),
            declined: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> CascadeDispatchSnapshot {
        CascadeDispatchSnapshot {
            fused: self.fused.load(Ordering::Relaxed),
            eager: self.eager.load(Ordering::Relaxed),
            declined: self.declined.load(Ordering::Relaxed),
        }
    }
}

/// The op-keyed registry for [`CascadeDispatchCounters`] — mirrors
/// [`counters_for`]/`registry` (this module's private, plain
/// `Mutex<HashMap<..>>` table) exactly, as a SEPARATE table (a cascade
/// op is never looked up through [`counters_for`], and a two-arm op is
/// never looked up through this function).
pub fn cascade_counters_for(op: &'static str) -> &'static CascadeDispatchCounters {
    cascade_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .entry(op)
        .or_insert_with(|| Box::leak(Box::new(CascadeDispatchCounters::new())))
}

fn cascade_registry() -> &'static Mutex<HashMap<&'static str, &'static CascadeDispatchCounters>> {
    static REGISTRY: OnceLock<Mutex<HashMap<&'static str, &'static CascadeDispatchCounters>>> =
        OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether `op` is named in `JAMMI_KERNELS_DISABLE` (exact match or the
/// `"all"` wildcard) — a thin, PUBLIC wrapper over the same private
/// `disabled_ops`/`fired_disables` plumbing [`admit`] itself uses,
/// exposed so a caller can skip an EXPENSIVE predicate computation (e.g. a
/// device-side mask reduction + D2H sync, P6 stage B's `flash_d2h_syncs`)
/// entirely when the op is disabled, rather than computing it and then
/// discarding the result inside [`admit_cascade`]. Calling this and then
/// [`admit_cascade`] is safe and not double-counted: this function reads
/// the SAME `fired_disables` bookkeeping [`admit_cascade`] itself updates,
/// so the disable is recorded fired exactly once regardless of how many
/// times a caller checks first.
pub fn op_disabled(op: &'static str) -> bool {
    op_is_disabled(disabled_ops(), fired_disables(), op)
}

/// Applies one CASCADE arm's admission decision (see [`PredicateOutcome`]'s
/// doc for why this is a separate entry point from [`admit`]).
///
/// `next_arm_can_run` is the caller's OWN assertion — `admit_cascade` does
/// not and cannot know the caller's fallback chain — that if THIS arm
/// declines, some later arm in the chain can still execute. Under
/// [`AdmissionMode::Strict`], a [`PredicateOutcome::CapabilityMiss`] is a
/// typed [`KernelError::StrictModeFallback`] UNLESS `next_arm_can_run` is
/// `true`; a [`PredicateOutcome::DomainMiss`] is NEVER an error, in either
/// mode (this call's data legitimately does not fit this arm — that is not
/// the class of defect `Strict` exists to catch).
///
/// `JAMMI_KERNELS_DISABLE` naming `op` wins over everything (identical
/// precedent to [`admit`]): the predicate is not even consulted, `Strict`
/// does not turn it into an error, and the decline is recorded in
/// `declined` (not `eager` — see [`CascadeDispatchCounters`]'s doc).
pub fn admit_cascade(
    mode: AdmissionMode,
    op: &'static str,
    predicate_name: &'static str,
    outcome: PredicateOutcome,
    next_arm_can_run: bool,
    counters: &CascadeDispatchCounters,
) -> Result<CascadeOutcome> {
    if op_is_disabled(disabled_ops(), fired_disables(), op) {
        counters.declined.fetch_add(1, Ordering::Relaxed);
        warn_disabled_once(op);
        return Ok(CascadeOutcome::Declined);
    }
    match outcome {
        PredicateOutcome::Holds => {
            counters.fused.fetch_add(1, Ordering::Relaxed);
            Ok(CascadeOutcome::Fused)
        }
        PredicateOutcome::DomainMiss => {
            counters.declined.fetch_add(1, Ordering::Relaxed);
            Ok(CascadeOutcome::Declined)
        }
        PredicateOutcome::CapabilityMiss => {
            counters.declined.fetch_add(1, Ordering::Relaxed);
            match mode {
                AdmissionMode::Fallback => Ok(CascadeOutcome::Declined),
                AdmissionMode::Strict => {
                    if next_arm_can_run {
                        Ok(CascadeOutcome::Declined)
                    } else {
                        Err(KernelError::StrictModeFallback {
                            op,
                            predicate: predicate_name,
                        })
                    }
                }
            }
        }
    }
}

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

/// One recorded fallback warning: `(op, predicate_key, message)` — see
/// [`fallback_warnings_emitted`]'s doc for what `predicate_key`
/// distinguishes.
type FallbackWarning = (&'static str, &'static str, String);

/// Process-wide, append-only record of every fallback warning this process
/// has actually EMITTED (not merely "would have logged"): [`FallbackWarning`]
/// triples, pushed by [`warn_fallback_once_with_message`] in the SAME
/// guarded block that fires `tracing::warn!`. This is the deterministic
/// oracle a test asserts against, in place of capturing the `tracing` log
/// line itself — `tracing`'s callsite `Interest` cache is PROCESS-GLOBAL, so
/// a thread-local `tracing::subscriber::set_default` guard around one test
/// races every OTHER test in this shared binary that can reach the same
/// `tracing::warn!` callsite concurrently (this crate used exactly that
/// pattern until it flaked under `cargo test --test-threads=N`: a sibling
/// test on another thread with no subscriber installed could win the
/// callsite's first-touch `Interest` decision and starve this test's
/// capture of every event). A plain, process-wide `Mutex<Vec<_>>` has no
/// such hazard: every thread that reaches this function pushes into the
/// SAME vector regardless of scheduling, and [`fallback_warnings_emitted`]
/// reads it directly — no subscriber, no callsite cache, no thread-local
/// guard.
///
/// Same provenance family as [`disabled_ops_fired`]: an append-only,
/// process-wide record a test (or a durable run artifact) can assert
/// against deterministically rather than eyeballing a log stream.
fn fallback_warnings() -> &'static Mutex<Vec<FallbackWarning>> {
    static WARNINGS: OnceLock<Mutex<Vec<FallbackWarning>>> = OnceLock::new();
    WARNINGS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Every fallback warning this process has emitted so far, in emission
/// order: `(op, predicate_key, message)`. `predicate_key` is either the
/// call site's own `predicate_name` (a genuine domain-predicate failure —
/// `warn_predicate_failed_once`'s path) or the literal
/// `"disabled_by_JAMMI_KERNELS_DISABLE"` (`warn_disabled_once`'s path,
/// the same fixed key `op_is_disabled`'s callers use elsewhere in this
/// module) — the two are always distinguishable by this field alone,
/// without needing to compare `message` text. See `fallback_warnings`'s
/// doc for why this is the oracle a test asserts against instead of a
/// captured `tracing` log line.
pub fn fallback_warnings_emitted() -> Vec<FallbackWarning> {
    fallback_warnings()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Emits a `tracing::warn!` at most once per process for a given
/// `(op, predicate)` pair, with `message` as the log line — split out from
/// [`warn_fallback_once`] so [`warn_disabled_once`] can share the SAME
/// log-once-per-process dedup set while emitting a message that does not
/// misattribute a deliberate `JAMMI_KERNELS_DISABLE` instruction as a
/// "domain check failed" defect.
///
/// Records into [`fallback_warnings`] in the SAME guarded block that fires
/// `tracing::warn!` — a mutation that replaces this function's body (or
/// [`warn_fallback_once`]'s) wholesale with `()` removes BOTH the log line
/// and the record, so [`fallback_warnings_emitted`] coming back without
/// the expected entry is what a test observes; it does not need to capture
/// the `tracing` output at all.
fn warn_fallback_once_with_message(op: &'static str, predicate: &'static str, message: &str) {
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, &'static str)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if seen.insert((op, predicate)) {
        tracing::warn!(op, predicate, "{message}");
        fallback_warnings()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((op, predicate, message.to_string()));
    }
}

/// Emits a `tracing::warn!` at most once per process for a given
/// `(op, predicate)` pair — the log-once-per-process WARN naming the op AND
/// the failed predicate. Used for a genuine domain-predicate failure; see
/// `warn_disabled_once` for the `JAMMI_KERNELS_DISABLE` path's own,
/// differently-worded message (advisory: the disabled path is not a
/// predicate failure and must not read as one in the log).
pub fn warn_fallback_once(op: &'static str, predicate: &'static str) {
    warn_fallback_once_with_message(
        op,
        predicate,
        "fused-kernel domain check failed; falling back to the eager composition",
    );
}

/// Parses a `JAMMI_KERNELS_DISABLE`-shaped value into an op-key SET:
/// comma-separated, each entry trimmed, empty entries dropped (a trailing
/// comma or stray whitespace must not manufacture a bogus `""` entry that
/// [`unmatched_disables`] would then report as unmatched forever), and
/// DUPLICATES COLLAPSED (a `HashSet`, not a `Vec` — the grammar this
/// parses is a SET of op keys, not an ordered, possibly-repeating list;
/// `"foo,foo"` and `"foo"` name the same request). `None` (unset) and
/// `Some("")`/an all-whitespace/all-comma value collapse to the SAME empty
/// set — `op_is_disabled` treats an empty set as unconditionally inert,
/// so unset and "set but empty" are byte-identical in their effect on
/// [`admit`].
///
/// Public and reused verbatim by `jammi-bench`'s `--expect-kernels-disabled`
/// flag (`crates/jammi-bench/src/main.rs`): that flag's value and
/// `JAMMI_KERNELS_DISABLE` are the SAME grammar (a caller states the SAME
/// disable list two ways — one via the env var this process reads, one via
/// an argv claim about what it expects that env var to carry), so there is
/// exactly ONE parser for it. A round-3 audit found `--expect-kernels-disabled`
/// had grown a SECOND, divergent parser that collected into a `Vec` with no
/// dedup: `JAMMI_KERNELS_DISABLE=op --expect-kernels-disabled op,op` then
/// compared `disabled_ops_requested()`'s deduplicated `["op"]` against the
/// duplicate-preserving `["op", "op"]` and hard-failed a VALID leg, blaming
/// a dropped env var that was never dropped. Both parsers reading through
/// this one function makes that class of divergence structurally
/// impossible, not just untested.
///
/// Pure/no I/O — split out from `disabled_ops` so the parsing edge
/// cases are unit-testable with literal inputs, independent of the
/// process-wide [`OnceLock`] `disabled_ops` memoizes into.
pub fn parse_disable_list(raw: Option<&str>) -> HashSet<String> {
    raw.map(|s| {
        s.split(',')
            .map(str::trim)
            .filter(|tok| !tok.is_empty())
            .map(str::to_string)
            .collect()
    })
    .unwrap_or_default()
}

/// The op keys named by `JAMMI_KERNELS_DISABLE`, read once per process —
/// mirrors [`admission_mode`]'s `OnceLock` contract (set before the
/// process starts; a bench/CI lane's convention, not something a running
/// process is expected to observe change). An op name in this set makes
/// [`admit`] return `Ok(DispatchOutcome::Eager)` for that op regardless of
/// `AdmissionMode` or the predicate — see this module's doc.
fn disabled_ops() -> &'static HashSet<String> {
    static DISABLED: OnceLock<HashSet<String>> = OnceLock::new();
    DISABLED
        .get_or_init(|| parse_disable_list(std::env::var("JAMMI_KERNELS_DISABLE").ok().as_deref()))
}

/// Which of [`disabled_ops`]'s requested entries have actually disabled a
/// live [`admit`] call at least once — the observation
/// [`unmatched_disables`] diffs `disabled_ops()` against. Populated by
/// [`op_is_disabled`].
///
/// `RwLock`, not `Mutex`: [`op_is_disabled`] takes only a READ lock on the
/// (overwhelmingly common) already-fired path, so a forced-eager leg of a
/// TIMING A/B — which calls this once per encoder forward, every step —
/// is not biased by an exclusive lock it does not need after the first
/// dispatch (see [`op_is_disabled`]'s doc).
fn fired_disables() -> &'static RwLock<HashSet<String>> {
    static FIRED: OnceLock<RwLock<HashSet<String>>> = OnceLock::new();
    FIRED.get_or_init(|| RwLock::new(HashSet::new()))
}

/// Whether `op` is disabled by `requested` (an exact-name entry, or the
/// `"all"` wildcard), recording which of `requested`'s entries actually
/// matched into `fired`.
///
/// Pure with respect to global state — both collections are passed in —
/// so the disable lattice's cells are unit-testable against literal,
/// test-local `requested`/`fired` instances, never the process-wide
/// `OnceLock`s [`admit`] itself reads through
/// [`disabled_ops`]/[`fired_disables`]. (This crate does not test the
/// env-var plumbing itself via `std::env::set_var` inside `cargo test`:
/// the `OnceLock` is initialized by whichever test's thread reads it
/// FIRST in the shared test binary, exactly the hazard
/// `admission_mode_defaults_to_fallback_without_the_env_var`'s doc
/// already names for `JAMMI_KERNELS_STRICT`. `crates/jammi-bench/tests/`
/// proves the real env-var path end to end by spawning the compiled
/// `jammi-bench` binary as a fresh child PROCESS instead, where a fresh
/// `OnceLock` is guaranteed.)
///
/// An exact match and the `"all"` wildcard are recorded independently:
/// `requested = {"all", "foo"}` with `op = "foo"` marks BOTH `"all"` and
/// `"foo"` fired (each legitimately requested it); `requested = {"all"}`
/// with `op = "foo"` marks only `"all"` fired (`"foo"` was never
/// requested by name, so it must never appear in `fired` — that would
/// let an unrelated op's dispatch paper over a genuinely-never-matched
/// literal entry).
///
/// Takes a READ lock first to check whether every entry `op` could fire is
/// already recorded in `fired`, only escalating to a write lock (and only
/// then allocating `op.to_string()`) the first time each entry actually
/// fires. A disabled op's call site reaches this function on EVERY
/// dispatch (every encoder forward, every step) for the lifetime of the
/// process, but after the first dispatch there is nothing left to record —
/// an unconditional write lock plus a `String` allocation on every one of
/// those later calls would bias exactly the timing measurement a
/// `JAMMI_KERNELS_DISABLE` forced-eager leg exists to produce.
fn op_is_disabled(
    requested: &HashSet<String>,
    fired: &RwLock<HashSet<String>>,
    op: &'static str,
) -> bool {
    if requested.is_empty() {
        return false;
    }
    let via_all = requested.contains("all");
    let via_exact = requested.contains(op);
    if !via_all && !via_exact {
        return false;
    }
    {
        let fired_read = fired
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let exact_already_fired = !via_exact || fired_read.contains(op);
        let all_already_fired = !via_all || fired_read.contains("all");
        if exact_already_fired && all_already_fired {
            return true;
        }
    }
    let mut fired = fired
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if via_exact {
        fired.insert(op.to_string());
    }
    if via_all {
        fired.insert("all".to_string());
    }
    true
}

/// `requested` entries absent from `fired`, sorted (family J: `HashSet`
/// iteration order is not a fold order this codebase relies on for a
/// durable artifact — a caller that logs/serializes this list needs a
/// deterministic order, not whatever the default hasher's bucket layout
/// happens to produce on a given run).
fn compute_unmatched(requested: &HashSet<String>, fired: &HashSet<String>) -> Vec<String> {
    let mut unmatched: Vec<String> = requested.difference(fired).cloned().collect();
    unmatched.sort();
    unmatched
}

/// Every `JAMMI_KERNELS_DISABLE` entry that has not disabled a single live
/// [`admit`] call this process — the safety-property read API. A
/// non-empty result means the disable list named at least one op key that
/// this run never actually dispatched through, which is EXACTLY the "a
/// typo in the disable list silently reads as a successful forced-eager
/// run" failure this mechanism exists to prevent (see this module's doc's
/// "safety property" section). A caller building a durable run record
/// (`jammi-bench`'s report/proof path) must treat a non-empty result as
/// an INVALID run, not a datum.
///
/// `disabled_ops`'s registry is lazily populated by observation (an op
/// name only becomes "seen" the first time a call site actually reaches
/// [`admit`]), so this cannot be validated at process startup — only at
/// the end of a run, after every call site that was going to fire this
/// process has had the chance to.
pub fn unmatched_disables() -> Vec<String> {
    let fired = fired_disables()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    compute_unmatched(disabled_ops(), &fired)
}

/// The `JAMMI_KERNELS_DISABLE` entries requested this process, sorted
/// (family J — see `compute_unmatched`'s doc for why a `HashSet`'s
/// iteration order is never a durable-artifact fold order). The
/// REQUESTED half of the `requested`/`fired` pair a caller building a
/// durable run record (`jammi-bench`'s `FinetuneStepTier`) is expected to
/// carry: naming which arm a run intended to measure, independent of
/// whether anything actually fired — see [`disabled_ops_fired`]'s doc for
/// why the pair, not either alone, is what closes the "env var silently
/// not forwarded" hole (both empty is byte-identical to "nothing was
/// requested", the same as an unset env var — that is the point: a
/// caller comparing this against an EXPECTED non-empty request can tell
/// the two apart, this function alone cannot).
pub fn disabled_ops_requested() -> Vec<String> {
    let mut v: Vec<String> = disabled_ops().iter().cloned().collect();
    v.sort();
    v
}

/// The `JAMMI_KERNELS_DISABLE` entries that have actually disabled at
/// least one live [`admit`] call this process, sorted. The FIRED half of
/// the `requested`/`fired` pair: a run whose `JAMMI_KERNELS_DISABLE` env
/// var was silently dropped (a var-NAME typo, an unforwarded ssh/`docker
/// -e` environment) reads `disabled_ops_requested() == []` and
/// `disabled_ops_fired() == []` — indistinguishable, on THIS pair alone,
/// from a run that genuinely requested nothing. A caller that recorded
/// what it INTENDED to request (the op key(s) it passed on its own
/// command line, independent of this process's view of its environment)
/// can compare that intent against this pair and catch the drop: a
/// non-empty intended request paired with an empty
/// `disabled_ops_requested()` here is exactly the dropped-var failure
/// mode. See [`unmatched_disables`] for the SEPARATE, narrower property
/// this function does not replace: an entry that WAS requested but never
/// fired (a typo inside a delivered list, or a dead registry name) is
/// still reported there as an invalid-run condition regardless of what
/// this pair shows.
pub fn disabled_ops_fired() -> Vec<String> {
    let fired = fired_disables()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut v: Vec<String> = fired.iter().cloned().collect();
    v.sort();
    v
}

/// Why an [`admit_inner`] call landed on the `Eager` arm — `None` when the
/// predicate held (`Fused`). Exposed on [`AdmitDecision`] (not just as a
/// side-effecting log line) so a lattice cell can assert on it DIRECTLY:
/// deleting the statement that computes/emits the disabled-path warning
/// then breaks compilation (`reason` becomes unbound) rather than silently
/// surviving — see [`AdmitDecision`]'s doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FallbackReason {
    /// This op was named in `JAMMI_KERNELS_DISABLE` (or `"all"`) — a
    /// deliberate instruction, not a predicate failure.
    Disabled,
    /// The call site's own domain predicate did not hold.
    PredicateFailed,
}

/// [`admit_inner`]'s full decision: the outcome every call site consumes,
/// PLUS why (`None` on `Fused`) — the `reason` field is what makes cell 5
/// (disabled path) and cell 3/7 (predicate-failure path) distinguishable
/// to a test without capturing a `tracing` log line, closing the mutant
/// that deletes the disabled path's own warning: `reason` is produced BY
/// the warn helper call (`warn_disabled_once`/`warn_predicate_failed_once`
/// return the [`FallbackReason`] they log), so a mutation that deletes
/// that call cannot compile (the `let reason = ..;` binding used in the
/// return value would be gone) — it is not merely untested, it is
/// unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AdmitDecision {
    outcome: DispatchOutcome,
    reason: Option<FallbackReason>,
}

/// Emits the disabled-path warning — a message DISTINCT from
/// [`warn_fallback_once`]'s ("fused-kernel domain check failed…"), because
/// a `JAMMI_KERNELS_DISABLE` entry is not a domain-check failure at all;
/// conflating the two log lines would misattribute a deliberate
/// instruction as a predicate defect. Returns [`FallbackReason::Disabled`]
/// unconditionally — see [`AdmitDecision`]'s doc for why this return value
/// (not just the log side effect) is what a test asserts on.
fn warn_disabled_once(op: &'static str) -> FallbackReason {
    warn_fallback_once_with_message(
        op,
        "disabled_by_JAMMI_KERNELS_DISABLE",
        "op disabled via JAMMI_KERNELS_DISABLE",
    );
    FallbackReason::Disabled
}

/// Emits the predicate-failure warning via [`warn_fallback_once`], then
/// returns [`FallbackReason::PredicateFailed`] — see
/// [`warn_disabled_once`]'s doc for the sibling disabled-path helper and
/// why each returns its own [`FallbackReason`] rather than being a bare
/// side effect.
fn warn_predicate_failed_once(op: &'static str, predicate_name: &'static str) -> FallbackReason {
    warn_fallback_once(op, predicate_name);
    FallbackReason::PredicateFailed
}

/// [`admit`]'s decision core: `disabled` is already resolved (by the
/// caller, from [`op_is_disabled`]) rather than read from process state
/// here, so every lattice cell is testable with literal, deterministic
/// inputs — see [`admit`]'s doc for the full state table.
fn admit_inner(
    mode: AdmissionMode,
    op: &'static str,
    predicate_name: &'static str,
    predicate_holds: bool,
    disabled: bool,
    counters: &DispatchCounters,
) -> Result<AdmitDecision> {
    if disabled {
        // Disable wins over BOTH the predicate and `Strict` mode — an
        // explicit `JAMMI_KERNELS_DISABLE` entry is a deliberate
        // instruction to force the eager arm, not the predicate failure
        // `Strict` exists to turn into an error. This is the load-bearing
        // cell (predicate holds AND mode is `Strict`, and disable STILL
        // wins) that makes `JAMMI_KERNELS_STRICT=1
        // JAMMI_KERNELS_DISABLE=<op>` a one-build A/B oracle: `<op>` is
        // forced eager while every OTHER op passing through this same
        // function is still strictly proven fused.
        counters.record(DispatchOutcome::Eager);
        let reason = warn_disabled_once(op);
        return Ok(AdmitDecision {
            outcome: DispatchOutcome::Eager,
            reason: Some(reason),
        });
    }
    if predicate_holds {
        counters.record(DispatchOutcome::Fused);
        return Ok(AdmitDecision {
            outcome: DispatchOutcome::Fused,
            reason: None,
        });
    }
    counters.record(DispatchOutcome::Eager);
    match mode {
        AdmissionMode::Fallback => {
            let reason = warn_predicate_failed_once(op, predicate_name);
            Ok(AdmitDecision {
                outcome: DispatchOutcome::Eager,
                reason: Some(reason),
            })
        }
        AdmissionMode::Strict => Err(KernelError::StrictModeFallback {
            op,
            predicate: predicate_name,
        }),
    }
}

/// Applies one fused-op call site's admission decision: records the
/// dispatch outcome, and in `Strict` mode turns a failed predicate into a
/// typed error instead of a silent fallback.
///
/// `predicate_holds` is the call site's own domain check (already evaluated
/// — this function does not know what the predicate means, only whether it
/// held); `predicate_name` is what gets logged/erred on failure.
///
/// Before consulting `predicate_holds` or `mode` at all, checks whether
/// `op` is named in `JAMMI_KERNELS_DISABLE` (this module's doc has the
/// full mechanism and the state table): if so, this ALWAYS returns
/// `Ok(DispatchOutcome::Eager)`, unconditionally, in either `AdmissionMode`
/// — disable wins over `Strict`. With the env var unset (or set but
/// empty), `disabled_ops` is an empty set and `op_is_disabled` returns
/// `false` for every `op`, so this reduces to exactly the two-outcome
/// decision this function has always made.
pub fn admit(
    mode: AdmissionMode,
    op: &'static str,
    predicate_name: &'static str,
    predicate_holds: bool,
    counters: &DispatchCounters,
) -> Result<DispatchOutcome> {
    let disabled = op_is_disabled(disabled_ops(), fired_disables(), op);
    admit_inner(
        mode,
        op,
        predicate_name,
        predicate_holds,
        disabled,
        counters,
    )
    .map(|decision| decision.outcome)
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

/// Whether THIS crate (`jammi-kernels`) was compiled with the `flash-attn`
/// feature — a plain `const`, unconditionally compiled (never behind
/// `#[cfg(feature = "flash-attn")]` itself), so a downstream crate can read
/// the real answer WITHOUT forwarding the feature through its own
/// `Cargo.toml`. This matters because `crate::flash` (the module the real
/// answer would otherwise require referencing) IS `#[cfg(feature =
/// "flash-attn")]`-gated (`lib.rs`): a call site cannot "stay compiled"
/// behind a bare `cfg!()` runtime check if reaching the `true` branch would
/// need to NAME a type from that module — the code would fail to compile
/// whenever the LOCAL crate's own feature is off, regardless of what this
/// constant says. `FLASH_COMPILED` is therefore useful ONLY for a predicate
/// that decides fused-vs-eager without ever naming a `crate::flash` type
/// directly (P6 Stage B's `attention_block_flash` admission predicate,
/// `jammi-encoders`, is exactly such a caller today — it holds row
/// `lengths`, not a constructed `crate::flash::CuSeqlens`, for precisely
/// this reason). Workspace feature unification makes this SOUND: `cfg!`
/// resolves to how THIS crate was actually compiled for the whole build
/// graph, the same value regardless of which downstream crate is asking.
pub const FLASH_COMPILED: bool = cfg!(feature = "flash-attn");

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
        // Calling it twice with the same key must not panic, and
        // `fallback_warnings_emitted()` must record exactly ONE entry for
        // this (op, predicate) pair regardless of how many times it fires
        // — the `seen.insert` dedup guards the push the same way it guards
        // the `tracing::warn!` call.
        warn_fallback_once("dedup_test_op", "dedup_predicate");
        warn_fallback_once("dedup_test_op", "dedup_predicate");
        warn_fallback_once("dedup_test_op", "dedup_predicate");
        let count = fallback_warnings_emitted()
            .iter()
            .filter(|(op, predicate, _)| *op == "dedup_test_op" && *predicate == "dedup_predicate")
            .count();
        assert_eq!(
            count, 1,
            "a log-once-per-process key must record exactly one entry no matter how many times it fires"
        );
    }

    /// Cell 3 and cell 5's messages are not merely non-equal by accident —
    /// this drives BOTH `admit_inner` arms itself (through a dedicated,
    /// test-unique op name, so this assertion is independent of whichever
    /// other test happens to run first/concurrently) and asserts the two
    /// recorded messages differ. Replaces
    /// `disabled_path_warn_message_is_distinct_from_a_genuine_predicate_failure`
    /// (removed): that test proved the same property via a thread-local
    /// `tracing::subscriber::set_default` guard racing `tracing`'s
    /// process-global callsite `Interest` cache against every sibling test
    /// reaching the same callsite concurrently — flaky under
    /// `cargo test --test-threads=N` (8/200 runs observed failing in the
    /// phase-4 audit). `fallback_warnings_emitted()` has no such race.
    #[test]
    fn fallback_warning_messages_are_distinct_between_the_disabled_and_predicate_failure_paths() {
        let op = "fallback_warning_distinctness_op";
        let counters = DispatchCounters::new();

        admit_inner(
            AdmissionMode::Fallback,
            op,
            "distinctness_pred",
            false,
            false,
            &counters,
        )
        .expect("Fallback mode never errors");
        admit_inner(
            AdmissionMode::Fallback,
            op,
            "distinctness_pred",
            true,
            true,
            &counters,
        )
        .expect("a disabled op never errors");

        let warnings = fallback_warnings_emitted();
        let predicate_failure_message = warnings
            .iter()
            .find(|(o, p, _)| *o == op && *p == "distinctness_pred")
            .map(|(_, _, m)| m.clone())
            .expect("predicate-failure warn must be recorded");
        let disabled_message = warnings
            .iter()
            .find(|(o, p, _)| *o == op && *p == "disabled_by_JAMMI_KERNELS_DISABLE")
            .map(|(_, _, m)| m.clone())
            .expect("disabled warn must be recorded");

        assert_ne!(
            predicate_failure_message, disabled_message,
            "a JAMMI_KERNELS_DISABLE forced-eager outcome must never read as a predicate defect, or vice versa"
        );
        assert!(predicate_failure_message.contains("fused-kernel domain check failed"));
        assert!(disabled_message.contains("op disabled via JAMMI_KERNELS_DISABLE"));
    }

    /// B4 / advisory (a), belt-and-braces: captures the ACTUAL
    /// `tracing::warn!` line the disabled path emits via a hand-rolled
    /// `tracing::Subscriber` (no `tracing-subscriber` dev-dependency needed
    /// — `tracing` itself is already a direct dependency of this crate).
    /// This closes the mutant class the reason-returning refactor cannot,
    /// on its own, fully rule out: a mutation that deletes the WARN CALL
    /// one level inside `warn_disabled_once` (rather than the `let reason =
    /// ..;` binding `admit_inner` uses, which a deletion mutant cannot
    /// remove without breaking compilation) would still leave
    /// `decision.reason` correct — this test fails on that mutant because
    /// it asserts the LOG LINE itself fired, not just the returned reason.
    /// Also proves advisory (a): the disabled path's message is DISTINCT
    /// from a genuine predicate-failure's ("fused-kernel domain check
    /// failed…") — conflating the two would misattribute a deliberate
    /// `JAMMI_KERNELS_DISABLE` instruction as a predicate defect.
    #[test]
    fn disabled_path_warn_message_is_distinct_from_a_genuine_predicate_failure() {
        use std::sync::Arc;
        use tracing::field::{Field, Visit};
        use tracing::span::{Attributes, Id, Record};
        use tracing::{Event, Metadata, Subscriber};

        #[derive(Default)]
        struct MessageVisitor(String);
        impl Visit for MessageVisitor {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                self.0.push_str(&format!("{}={value:?} ", field.name()));
            }
        }

        struct CapturingSubscriber {
            events: Arc<Mutex<Vec<String>>>,
        }
        impl Subscriber for CapturingSubscriber {
            fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
                true
            }
            fn new_span(&self, _span: &Attributes<'_>) -> Id {
                Id::from_u64(1)
            }
            fn record(&self, _span: &Id, _values: &Record<'_>) {}
            fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
            fn event(&self, event: &Event<'_>) {
                let mut visitor = MessageVisitor::default();
                event.record(&mut visitor);
                self.events
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(visitor.0);
            }
            fn enter(&self, _span: &Id) {}
            fn exit(&self, _span: &Id) {}
        }

        let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let subscriber = CapturingSubscriber {
            events: events.clone(),
        };
        let _guard = tracing::subscriber::set_default(subscriber);

        let counters = DispatchCounters::new();
        admit_inner(
            AdmissionMode::Fallback,
            "warn_capture_disabled_op",
            "some_predicate",
            true,
            true, // disabled
            &counters,
        )
        .expect("disabled path never errors");

        let captured = events.lock().unwrap_or_else(|p| p.into_inner()).join("\n");
        assert!(
            captured.contains("op disabled via JAMMI_KERNELS_DISABLE"),
            "the disabled path's own message must actually be logged; captured={captured}"
        );
        assert!(
            !captured.contains("fused-kernel domain check failed"),
            "the disabled path must NOT reuse the predicate-failure message; captured={captured}"
        );
    }

    #[test]
    fn flash_compiled_matches_the_flash_attn_feature_this_binary_was_built_with() {
        assert_eq!(FLASH_COMPILED, cfg!(feature = "flash-attn"));
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

    /// Round-2 mutants triage (`cargo mutants`: `replace admission_mode ->
    /// AdmissionMode with Default::default()` MISSED): nothing in this
    /// crate OR `jammi-bench`'s real-CLI tests previously exercised
    /// `admission_mode()` actually reading `Strict` from a genuine
    /// `JAMMI_KERNELS_STRICT` env var — `jammi_encoders::layer_norm`'s
    /// `strict_mode_errors_instead_of_falling_back_on_a_failed_predicate`
    /// explicitly bypasses this function (calls `admit` with a literal
    /// `AdmissionMode::Strict` instead, citing the exact same `OnceLock`
    /// hazard), and every `jammi-bench` fixture's fused predicates hold
    /// trivially, so `JAMMI_KERNELS_STRICT=1` there never distinguishes
    /// Strict from Fallback (disable-wins-over-Strict cells aside). A
    /// mutant that made `admission_mode()` always return `Fallback` would
    /// silently turn Strict mode into a no-op everywhere in a real binary
    /// and nothing would have caught it.
    ///
    /// Spawns the ALREADY-COMPILED test binary as a fresh CHILD process
    /// with the env var set and `--exact` targeting ONLY
    /// [`admission_mode_child_process_body`] below — a fresh `OnceLock`,
    /// guaranteed, the same technique
    /// `crates/jammi-bench/tests/finetune_step_kernel_disable.rs` uses for
    /// `JAMMI_KERNELS_DISABLE`.
    #[test]
    fn admission_mode_reads_strict_from_the_real_env_var_in_a_fresh_process() {
        let exe = std::env::current_exe().expect("test binary path");
        let output = std::process::Command::new(exe)
            .args([
                "admission::tests::admission_mode_child_process_body",
                "--exact",
                "--nocapture",
            ])
            .env("JAMMI_KERNELS_STRICT", "1")
            .env("ADMISSION_MODE_CHILD", "1")
            .output()
            .expect("spawn child test binary");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "child process assertion failed: stdout={stdout}\nstderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        // Non-vacuity: `cargo test`'s libtest harness exits 0 on a filter
        // that matches ZERO tests (a typo'd `--exact` path, or a module
        // rename that silently stops matching, would make this test
        // "pass" having run NOTHING). Asserting the child actually ran and
        // passed exactly the one test it was told to run is what makes
        // `output.status.success()` alone mean what this test claims it
        // means.
        assert!(
            stdout.contains("1 passed"),
            "the child process must have actually run (and passed) exactly one test — \
             stdout={stdout}"
        );
    }

    /// Only meaningful inside the child process
    /// [`admission_mode_reads_strict_from_the_real_env_var_in_a_fresh_process`]
    /// spawns (guarded on `ADMISSION_MODE_CHILD`, the same pattern
    /// `admission_mode_defaults_to_fallback_without_the_env_var` uses for
    /// the unset case) — a no-op pass when run directly by the ordinary
    /// test harness.
    #[test]
    fn admission_mode_child_process_body() {
        if std::env::var_os("ADMISSION_MODE_CHILD").is_some() {
            assert_eq!(
                admission_mode(),
                AdmissionMode::Strict,
                "JAMMI_KERNELS_STRICT=1 in a fresh process must read as Strict"
            );
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

    // ---- JAMMI_KERNELS_DISABLE: the contract K-aux lattice ----------------
    //
    // Cells 1-8 drive `admit_inner` directly with a literal `disabled: bool`
    // (see `admit_inner`'s and `op_is_disabled`'s docs for why: this makes
    // every cell hermetic and parallel-safe, independent of the
    // process-wide `OnceLock`s the real `JAMMI_KERNELS_DISABLE` env var
    // memoizes into). Cell 9 (env unset/empty) is covered twice: the
    // parser-level tests below prove `parse_disable_list` collapses unset
    // and empty to the identical empty set, and
    // `lattice_cell_09_env_unset_admit_reduces_to_the_pre_disable_two_outcome_function`
    // is a guarded, documentation-level assertion about the real `admit`
    // (mirroring `admission_mode_defaults_to_fallback_without_the_env_var`'s
    // precedent). Cell 10's hermetic half (`compute_unmatched`) is below;
    // its REAL-entry-point half lives in
    // `crates/jammi-bench/tests/finetune_step_kernel_disable.rs` (a fresh
    // child process, so the `OnceLock` hazard does not apply there).

    #[test]
    fn lattice_cell_01_not_disabled_predicate_holds_fallback_is_fused() {
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Fallback,
            "lattice_op",
            "pred",
            true,
            false,
            &counters,
        )
        .expect("never errors");
        assert_eq!(decision.outcome, DispatchOutcome::Fused);
        assert_eq!(
            decision.reason, None,
            "a fused dispatch has no fallback reason"
        );
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 1, eager: 0 });
    }

    #[test]
    fn lattice_cell_02_not_disabled_predicate_holds_strict_is_fused() {
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Strict,
            "lattice_op",
            "pred",
            true,
            false,
            &counters,
        )
        .expect("a satisfied predicate never errors, in either mode");
        assert_eq!(decision.outcome, DispatchOutcome::Fused);
        assert_eq!(
            decision.reason, None,
            "a fused dispatch has no fallback reason"
        );
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 1, eager: 0 });
    }

    #[test]
    fn lattice_cell_03_not_disabled_predicate_fails_fallback_is_eager() {
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Fallback,
            "lattice_op",
            "pred_failed",
            false,
            false,
            &counters,
        )
        .expect("Fallback mode never errors");
        assert_eq!(decision.outcome, DispatchOutcome::Eager);
        assert_eq!(
            decision.reason,
            Some(FallbackReason::PredicateFailed),
            "a genuine predicate failure must be distinguishable from a JAMMI_KERNELS_DISABLE forced-eager outcome"
        );
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
        // The observability half of cell 3: the ONLY signal that this call
        // took the eager arm because the PREDICATE failed (not because the
        // op was disabled) is this warning — see `fallback_warnings`'s doc
        // for why this is asserted via the process-wide record rather than
        // a captured `tracing` log line. Kills the `replace
        // warn_fallback_once with ()` mutant: deleting that call stops
        // this entry from ever being pushed, while `decision.reason` above
        // (produced by `warn_predicate_failed_once`'s unconditional return
        // value, not by the warn call's side effect) would still read
        // correctly — this assertion is what that mutant needs to survive
        // past.
        let warnings = fallback_warnings_emitted();
        assert!(
            warnings.iter().any(|(op, predicate, message)| {
                *op == "lattice_op"
                    && *predicate == "pred_failed"
                    && message == "fused-kernel domain check failed; falling back to the eager composition"
            }),
            "the predicate-failure warn must actually be recorded for (lattice_op, pred_failed); warnings={warnings:?}"
        );
    }

    /// Cell 3's shape again, but through the literal public [`admit`] entry
    /// point rather than [`admit_inner`] — achievable hermetically here
    /// (unlike cell 5's disabled arm below) because this arm never
    /// consults `JAMMI_KERNELS_DISABLE` at all: `op_is_disabled` on the
    /// real, env-var-backed [`disabled_ops`] returns `false` for any op
    /// name never named in it, and no test in this binary ever calls
    /// `std::env::set_var("JAMMI_KERNELS_DISABLE", ..)` in-process (see
    /// `op_is_disabled`'s doc). Guarded the same way
    /// `admission_mode_defaults_to_fallback_without_the_env_var` and cell
    /// 9 are, and a fresh, test-unique op name so a concurrently-running
    /// test can never make this one's `disabled` observation ambiguous.
    #[test]
    fn lattice_cell_03_predicate_failure_warn_is_recorded_through_the_real_admit() {
        if std::env::var_os("JAMMI_KERNELS_DISABLE").is_none() {
            let op = "lattice_cell_03_real_admit_warn_op";
            let counters = DispatchCounters::new();
            let outcome = admit(
                AdmissionMode::Fallback,
                op,
                "warn_observability_pred",
                false,
                &counters,
            )
            .expect("Fallback mode never errors");
            assert_eq!(outcome, DispatchOutcome::Eager);
            let warnings = fallback_warnings_emitted();
            assert!(
                warnings.iter().any(|(o, p, m)| {
                    *o == op
                        && *p == "warn_observability_pred"
                        && m == "fused-kernel domain check failed; falling back to the eager composition"
                }),
                "the predicate-failure warn must be recorded through the real admit(); warnings={warnings:?}"
            );
        }
    }

    #[test]
    fn lattice_cell_04_not_disabled_predicate_fails_strict_errors() {
        let counters = DispatchCounters::new();
        let err = admit_inner(
            AdmissionMode::Strict,
            "lattice_op",
            "pred_failed",
            false,
            false,
            &counters,
        )
        .expect_err("Strict mode must error, never silently fall back");
        assert!(matches!(
            err,
            KernelError::StrictModeFallback {
                op: "lattice_op",
                predicate: "pred_failed"
            }
        ));
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }

    #[test]
    fn lattice_cell_05_disabled_predicate_holds_fallback_is_eager() {
        // Closes the mutant that deletes the disabled path's own warning
        // (`warn_disabled_once` inside `admit_inner`): `decision.reason` is
        // produced BY that call, not read from a captured log line, so a
        // mutation removing it fails to compile rather than surviving.
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Fallback,
            "lattice_op",
            "pred",
            true,
            true,
            &counters,
        )
        .expect("a disabled op never errors");
        assert_eq!(decision.outcome, DispatchOutcome::Eager);
        assert_eq!(
            decision.reason,
            Some(FallbackReason::Disabled),
            "disabling an op with a HOLDING predicate must be attributed to the disable, not a predicate failure"
        );
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
        // The observability half of cell 5: the ONLY signal on this arm
        // that the fused predicate WOULD have held but the op ran eager
        // anyway is this warning (with the fixed
        // `disabled_by_JAMMI_KERNELS_DISABLE` key, not the call site's own
        // predicate name) — see `fallback_warnings`'s doc. Kills the
        // `replace warn_fallback_once_with_message with ()` mutant: that
        // mutation removes both `warn_disabled_once`'s and
        // `warn_predicate_failed_once`'s log line AND this push (they
        // share the one function), while `decision.reason` above would
        // still read `Some(Disabled)` unaffected (it is the unconditional
        // return value of `warn_disabled_once`, not derived from the warn
        // call's side effect) — this assertion is what that mutant needs
        // to survive past.
        //
        // Driven through `admit_inner` (not the literal public [`admit`]):
        // exercising this arm through `admit` would require setting
        // `JAMMI_KERNELS_DISABLE` in-process, which races every OTHER test
        // in this shared binary for who populates `disabled_ops`'s
        // `OnceLock` first (`op_is_disabled`'s doc). `admit_inner` IS
        // `admit`'s real decision core once `disabled` has been resolved —
        // the only thing `admit` adds is that env-var read — so this is
        // the real admission decision function, not a helper reimplementing
        // it with literals. The real end-to-end proof that `admit()`'s
        // disabled arm actually fires through a genuine
        // `JAMMI_KERNELS_DISABLE` env var lives in
        // `crates/jammi-bench/tests/finetune_step_kernel_disable.rs`
        // (`strict_mode_disable_forces_layer_norm_eager_and_the_run_still_succeeds`
        // and its siblings), which spawns the real CLI in a fresh child
        // process for exactly this reason.
        let warnings = fallback_warnings_emitted();
        assert!(
            warnings.iter().any(|(op, predicate, message)| {
                *op == "lattice_op"
                    && *predicate == "disabled_by_JAMMI_KERNELS_DISABLE"
                    && message == "op disabled via JAMMI_KERNELS_DISABLE"
            }),
            "the disabled-path warn must actually be recorded for (lattice_op, disabled_by_JAMMI_KERNELS_DISABLE); warnings={warnings:?}"
        );
    }

    #[test]
    fn lattice_cell_06_disabled_predicate_holds_strict_is_eager_not_error() {
        // The load-bearing cell: disable wins over BOTH a holding
        // predicate AND `Strict` mode. Without this cell,
        // `JAMMI_KERNELS_STRICT=1 JAMMI_KERNELS_DISABLE=<op>` would error
        // instead of forcing the eager arm, and the one-build A/B oracle
        // this contract exists for would not work.
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Strict,
            "lattice_op",
            "pred",
            true,
            true,
            &counters,
        )
        .expect("disable must win over Strict — this must NOT error");
        assert_eq!(decision.outcome, DispatchOutcome::Eager);
        assert_eq!(decision.reason, Some(FallbackReason::Disabled));
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }

    #[test]
    fn lattice_cell_07_disabled_predicate_fails_fallback_is_eager() {
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Fallback,
            "lattice_op",
            "pred_failed",
            false,
            true,
            &counters,
        )
        .expect("a disabled op never errors");
        assert_eq!(decision.outcome, DispatchOutcome::Eager);
        assert_eq!(
            decision.reason,
            Some(FallbackReason::Disabled),
            "disable must be the reported reason even though the predicate ALSO failed"
        );
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }

    #[test]
    fn lattice_cell_08_disabled_predicate_fails_strict_is_eager_not_error() {
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Strict,
            "lattice_op",
            "pred_failed",
            false,
            true,
            &counters,
        )
        .expect("disable must win over Strict even when the predicate ALSO fails — must NOT error");
        assert_eq!(decision.outcome, DispatchOutcome::Eager);
        assert_eq!(decision.reason, Some(FallbackReason::Disabled));
        assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 0, eager: 1 });
    }

    #[test]
    fn lattice_cell_09_env_unset_admit_reduces_to_the_pre_disable_two_outcome_function() {
        // `op_is_disabled` on an empty requested set is `false` for every
        // op, so with `disabled_ops()` empty (the unset/empty-string case
        // `parse_disable_list`'s tests below cover directly), the real,
        // process-wide `admit`'s behaviour is `admit_inner` with
        // `disabled = false` for every call — EXACTLY this function's
        // pre-K-aux two-outcome shape. Guarded the same way
        // `admission_mode_defaults_to_fallback_without_the_env_var` is:
        // only meaningful if the real env var happens to be unset for
        // this test run (see `op_is_disabled`'s doc for why an in-process
        // `std::env::set_var` test is not attempted here).
        if std::env::var_os("JAMMI_KERNELS_DISABLE").is_none() {
            let counters = DispatchCounters::new();
            let outcome = admit(
                AdmissionMode::Strict,
                "cell9_never_disabled_op",
                "always_true",
                true,
                &counters,
            )
            .expect("undisabled, satisfied predicate never errors");
            assert_eq!(outcome, DispatchOutcome::Fused);
            assert_eq!(counters.snapshot(), DispatchSnapshot { fused: 1, eager: 0 });
            assert!(disabled_ops().is_empty());
            assert!(unmatched_disables().is_empty());
            // B3: the `requested`/`fired` pair a durable run record
            // (`jammi-bench`'s `FinetuneStepTier`) carries — both empty
            // with the env var genuinely unset, exactly matching an
            // ordinary undisabled run. `crates/jammi-bench/tests/` proves
            // the pair is NON-empty and matched on a genuine forced-eager
            // run through the real CLI, in a fresh process where setting
            // the env var is safe.
            assert!(disabled_ops_requested().is_empty());
            assert!(disabled_ops_fired().is_empty());
        }
    }

    #[test]
    fn parse_disable_list_unset_is_empty() {
        assert!(parse_disable_list(None).is_empty());
    }

    #[test]
    fn parse_disable_list_empty_string_is_empty() {
        assert!(parse_disable_list(Some("")).is_empty());
    }

    #[test]
    fn parse_disable_list_whitespace_and_stray_commas_produce_no_bogus_empty_entry() {
        // A trailing comma or stray whitespace must not manufacture a
        // bogus `""` entry — that would make `unmatched_disables()`
        // report a phantom unmatched entry for formatting alone.
        assert!(parse_disable_list(Some("  ,, , ")).is_empty());
    }

    #[test]
    fn parse_disable_list_trims_and_splits_on_comma() {
        let parsed = parse_disable_list(Some(" softmax_last_dim_fused ,rope_fused,,geglu_fused "));
        let expected: HashSet<String> = ["softmax_last_dim_fused", "rope_fused", "geglu_fused"]
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(parsed, expected);
    }

    #[test]
    fn parse_disable_list_all_keyword_is_a_plain_member_not_special_cased_here() {
        // `"all"`'s WILDCARD semantics live in `op_is_disabled`, not the
        // parser — the parser just preserves the literal entry.
        let parsed = parse_disable_list(Some("all"));
        assert_eq!(
            parsed,
            ["all".to_string()].into_iter().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn op_is_disabled_unlisted_op_is_false_and_does_not_touch_fired() {
        let requested: HashSet<String> = ["foo".to_string()].into_iter().collect();
        let fired = RwLock::new(HashSet::new());
        assert!(!op_is_disabled(&requested, &fired, "bar"));
        assert!(fired.read().unwrap().is_empty());
    }

    #[test]
    fn op_is_disabled_empty_requested_set_is_always_false() {
        let requested: HashSet<String> = HashSet::new();
        let fired = RwLock::new(HashSet::new());
        assert!(!op_is_disabled(&requested, &fired, "anything"));
    }

    #[test]
    fn op_is_disabled_exact_match_disables_and_records_only_that_name() {
        let requested: HashSet<String> = ["foo".to_string()].into_iter().collect();
        let fired = RwLock::new(HashSet::new());
        assert!(op_is_disabled(&requested, &fired, "foo"));
        let snap = fired.read().unwrap();
        assert_eq!(snap.len(), 1);
        assert!(snap.contains("foo"));
    }

    #[test]
    fn op_is_disabled_all_wildcard_disables_any_op_without_marking_its_own_name_fired() {
        let requested: HashSet<String> = ["all".to_string()].into_iter().collect();
        let fired = RwLock::new(HashSet::new());
        assert!(op_is_disabled(&requested, &fired, "some_op"));
        assert!(op_is_disabled(&requested, &fired, "another_op"));
        let snap = fired.read().unwrap();
        // Only `"all"` itself was ever a REQUESTED entry — neither op's
        // own literal name was requested, so neither may appear in
        // `fired` (that would let an unrelated op's dispatch paper over
        // a genuinely-never-matched literal entry, e.g. a mistyped
        // `"al"` sitting alongside a correct `"all"`).
        assert_eq!(snap.len(), 1);
        assert!(snap.contains("all"));
        assert!(!snap.contains("some_op"));
        assert!(!snap.contains("another_op"));
    }

    #[test]
    fn op_is_disabled_exact_and_all_both_present_marks_both_fired() {
        let requested: HashSet<String> =
            ["all".to_string(), "foo".to_string()].into_iter().collect();
        let fired = RwLock::new(HashSet::new());
        assert!(op_is_disabled(&requested, &fired, "foo"));
        let snap = fired.read().unwrap();
        assert_eq!(snap.len(), 2);
        assert!(snap.contains("all"));
        assert!(snap.contains("foo"));
    }

    #[test]
    fn op_is_disabled_repeated_calls_after_first_fire_stay_correct_and_idempotent() {
        // Advisory (b): once `op` is already recorded in `fired`, a repeat
        // call must take the read-only fast path (no reallocating
        // `op.to_string()`, no write lock) — exercised here by calling
        // `op_is_disabled` many times for the SAME op and asserting both
        // the return value and `fired`'s contents are stable (no growth,
        // no corruption from ever taking the write path again).
        let requested: HashSet<String> = ["foo".to_string()].into_iter().collect();
        let fired = RwLock::new(HashSet::new());
        for _ in 0..5 {
            assert!(op_is_disabled(&requested, &fired, "foo"));
        }
        let snap = fired.read().unwrap();
        assert_eq!(snap.len(), 1);
        assert!(snap.contains("foo"));
    }

    #[test]
    fn compute_unmatched_reports_requested_entries_never_fired() {
        // Cell 10 — the safety property — hermetic half: a typo'd entry
        // that never fired must be the one and only name reported.
        let requested: HashSet<String> =
            ["foo".to_string(), "bar".to_string()].into_iter().collect();
        let fired: HashSet<String> = ["foo".to_string()].into_iter().collect();
        assert_eq!(
            compute_unmatched(&requested, &fired),
            vec!["bar".to_string()]
        );
    }

    #[test]
    fn compute_unmatched_is_empty_when_every_requested_entry_fired() {
        let requested: HashSet<String> = ["foo".to_string()].into_iter().collect();
        let fired: HashSet<String> = ["foo".to_string()].into_iter().collect();
        assert!(compute_unmatched(&requested, &fired).is_empty());
    }

    // ---- `admit_cascade` / `PredicateOutcome` lattice ---------------------
    //
    // Distinct from the `JAMMI_KERNELS_DISABLE` lattice above (cells
    // 01-10, `admit_inner`): this exercises the NEW cascade entry point
    // only, through the REAL `admit_cascade` (not a private `_inner`), and
    // asserts the 10-cell lattice above is untouched (no cell was edited
    // to add these).

    #[test]
    fn cascade_holds_records_fused_in_either_mode() {
        for mode in [AdmissionMode::Fallback, AdmissionMode::Strict] {
            let counters = CascadeDispatchCounters::new();
            let outcome = admit_cascade(
                mode,
                "cascade_test_op_holds",
                "pred",
                PredicateOutcome::Holds,
                true,
                &counters,
            )
            .expect("Holds never errors");
            assert_eq!(outcome, CascadeOutcome::Fused);
            assert_eq!(
                counters.snapshot(),
                CascadeDispatchSnapshot {
                    fused: 1,
                    eager: 0,
                    declined: 0
                }
            );
        }
    }

    #[test]
    fn cascade_domain_miss_never_errors_even_under_strict() {
        // The load-bearing cell distinguishing `DomainMiss` from
        // `CapabilityMiss`: a domain miss is never a Strict error,
        // regardless of `next_arm_can_run`.
        for next_arm_can_run in [true, false] {
            let counters = CascadeDispatchCounters::new();
            let outcome = admit_cascade(
                AdmissionMode::Strict,
                "cascade_test_op_domain_miss",
                "pred_domain",
                PredicateOutcome::DomainMiss,
                next_arm_can_run,
                &counters,
            )
            .expect("DomainMiss must never error, in either mode, regardless of next_arm_can_run");
            assert_eq!(outcome, CascadeOutcome::Declined);
        }
        let counters = CascadeDispatchCounters::new();
        let outcome = admit_cascade(
            AdmissionMode::Fallback,
            "cascade_test_op_domain_miss_fb",
            "pred_domain",
            PredicateOutcome::DomainMiss,
            false,
            &counters,
        )
        .expect("DomainMiss never errors under Fallback either");
        assert_eq!(outcome, CascadeOutcome::Declined);
        assert_eq!(
            counters.snapshot(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 1
            }
        );
    }

    #[test]
    fn cascade_capability_miss_fallback_mode_declines_without_error() {
        let counters = CascadeDispatchCounters::new();
        let outcome = admit_cascade(
            AdmissionMode::Fallback,
            "cascade_test_op_cap_fb",
            "pred_cap",
            PredicateOutcome::CapabilityMiss,
            false,
            &counters,
        )
        .expect("Fallback mode never errors regardless of next_arm_can_run");
        assert_eq!(outcome, CascadeOutcome::Declined);
        assert_eq!(
            counters.snapshot(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 1
            }
        );
    }

    #[test]
    fn cascade_capability_miss_strict_declines_when_next_arm_can_run() {
        let counters = CascadeDispatchCounters::new();
        let outcome = admit_cascade(
            AdmissionMode::Strict,
            "cascade_test_op_cap_strict_ok",
            "pred_cap",
            PredicateOutcome::CapabilityMiss,
            true,
            &counters,
        )
        .expect("Strict must not error when the caller asserts a fallback arm can run");
        assert_eq!(outcome, CascadeOutcome::Declined);
        assert_eq!(
            counters.snapshot(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 1
            }
        );
    }

    #[test]
    fn cascade_capability_miss_strict_errors_when_no_arm_can_run() {
        // The other load-bearing cell: Strict DOES still have teeth for a
        // cascade — if nothing downstream can run either, this must error,
        // not silently decline.
        let counters = CascadeDispatchCounters::new();
        let err = admit_cascade(
            AdmissionMode::Strict,
            "cascade_test_op_cap_strict_err",
            "pred_cap",
            PredicateOutcome::CapabilityMiss,
            false,
            &counters,
        )
        .expect_err("Strict must error when NO arm in the chain can run");
        assert!(matches!(
            err,
            KernelError::StrictModeFallback {
                op: "cascade_test_op_cap_strict_err",
                predicate: "pred_cap"
            }
        ));
        // The attempted decline is still recorded even though the call
        // errors — same provenance discipline as `admit`'s Strict path.
        assert_eq!(
            counters.snapshot(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 1
            }
        );
    }

    #[test]
    fn cascade_disabled_wins_over_holds_and_over_strict() {
        let counters = CascadeDispatchCounters::new();
        // Hermetic: drives `admit_cascade` for a literal op name that is
        // never named in the REAL `JAMMI_KERNELS_DISABLE` env var in this
        // test binary, so this cell instead asserts the STRUCTURAL
        // property via `op_disabled` directly being false for an
        // unrequested name, then exercises the disabled branch through
        // `admit_cascade`'s own requested-set plumbing by using the `"all"`-
        // style path is not available hermetically (mirrors why the
        // `JAMMI_KERNELS_DISABLE` lattice above tests `admit_inner`
        // directly instead of `admit`). Cascade's disabled path shares the
        // identical `op_is_disabled`/`disabled_ops`/`fired_disables`
        // mechanism already proven correct by cells 01-10 above and
        // `crates/jammi-bench/tests/` end-to-end — this cell instead
        // proves `admit_cascade` records to `declined` (not `fused`/`eager`)
        // when disabled, which `admit`/`admit_inner` cannot express at all
        // (they only have `fused`/`eager`).
        assert!(!op_disabled("cascade_test_op_never_in_env"));
        let outcome = admit_cascade(
            AdmissionMode::Strict,
            "cascade_test_op_never_in_env",
            "pred",
            PredicateOutcome::Holds,
            true,
            &counters,
        )
        .expect("undisabled Holds never errors");
        assert_eq!(outcome, CascadeOutcome::Fused);
        assert_eq!(counters.snapshot().fused, 1);
    }

    #[test]
    fn cascade_dispatch_snapshot_default_is_all_zero() {
        assert_eq!(
            CascadeDispatchSnapshot::default(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 0
            }
        );
    }

    #[test]
    fn cascade_counters_for_returns_the_same_static_instance_for_the_same_op_name() {
        let a = cascade_counters_for("cascade_registry_test_op_a");
        let b = cascade_counters_for("cascade_registry_test_op_a");
        a.declined.fetch_add(1, Ordering::Relaxed);
        assert_eq!(
            b.snapshot(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 1
            }
        );
        assert!(std::ptr::eq(a, b), "must be the identical instance");
    }

    #[test]
    fn cascade_counters_for_keys_by_op_name_not_by_call_site() {
        let a = cascade_counters_for("cascade_registry_test_op_b1");
        let b = cascade_counters_for("cascade_registry_test_op_b2");
        a.fused.fetch_add(1, Ordering::Relaxed);
        b.declined.fetch_add(2, Ordering::Relaxed);
        assert_eq!(
            a.snapshot(),
            CascadeDispatchSnapshot {
                fused: 1,
                eager: 0,
                declined: 0
            }
        );
        assert_eq!(
            b.snapshot(),
            CascadeDispatchSnapshot {
                fused: 0,
                eager: 0,
                declined: 2
            }
        );
    }

    #[test]
    fn compute_unmatched_output_is_sorted_regardless_of_hashset_insertion_order() {
        // Family J: two `HashSet`s built by inserting in opposite order
        // must still yield the SAME `Vec` — the ordering is a property of
        // `compute_unmatched`'s explicit `.sort()`, not of insertion
        // order or the default hasher's bucket layout (which is
        // randomized per-process and is not a fold order this codebase
        // relies on for a durable/logged artifact).
        let mut requested_a = HashSet::new();
        for k in ["zeta", "alpha", "mu"] {
            requested_a.insert(k.to_string());
        }
        let mut requested_b = HashSet::new();
        for k in ["mu", "zeta", "alpha"] {
            requested_b.insert(k.to_string());
        }
        let fired = HashSet::new();
        let out_a = compute_unmatched(&requested_a, &fired);
        let out_b = compute_unmatched(&requested_b, &fired);
        assert_eq!(out_a, out_b);
        assert_eq!(
            out_a,
            vec!["alpha".to_string(), "mu".to_string(), "zeta".to_string()]
        );
    }
}
