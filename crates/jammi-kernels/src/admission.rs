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
//! under the hood: `LN_DISPATCH_COUNTERS`, `crates/jammi-encoders/src/layer_norm.rs:128`;
//! `ROPE_DISPATCH_COUNTERS`, `crates/jammi-encoders/src/modernbert.rs:180`;
//! `SOFTMAX_DISPATCH_COUNTERS`, `crates/jammi-encoders/src/modernbert.rs:194`;
//! `GEGLU_DISPATCH_COUNTERS`, `crates/jammi-encoders/src/modernbert.rs:1379`
//! (`ATTENTION_BLOCK_DISPATCH_COUNTERS`, `crates/jammi-encoders/src/modernbert.rs:586`
//! is a fifth, added the same way rather than as a sixth hand-declared static).
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
//! `"layer_norm_fused"` (`jammi-encoders/src/layer_norm.rs:583`),
//! `"geglu_fused"` (`jammi-encoders/src/modernbert.rs:1424`),
//! `"lora_linear_fused"` (`jammi-lora/src/lora_linear.rs:1007`), and
//! `"attention_block_fused"` (`jammi-encoders/src/attention_cascade.rs:914`) itself.
//!
//! **Subsumed** (reachable ONLY when `"attention_block_fused"` is ALSO
//! disabled, forcing `forward_training_attention` into
//! `forward_eager_training_attention_composition` — the composition that
//! calls `RotaryEmbedding::apply_training` and `softmax_apply_training`,
//! each of which independently calls [`admit`] with its own op key):
//! `"rope_fused"` (`modernbert.rs:478`), `"softmax_last_dim_fused"`
//! (`modernbert.rs:1188`).
//!
//! **Subsumed by `"lora_linear_fused"`** (reachable ONLY when
//! `"lora_linear_fused"` (`jammi-lora/src/lora_linear.rs:1007`) itself
//! admits Fused — `crate::ops::LowRankResidualLinear::bwd` is the sole
//! call site that ever passes either key to [`admit`], and
//! `LowRankResidualLinear` is only constructed on the branch where
//! `lora_linear_fused` already admitted; see `ops::cast_scale`'s
//! module doc's "cast-boundary lever"): `"cast_scale_bf16_f32"` and
//! `"cast_add_bf16"` (`ops/low_rank_residual_linear.rs`'s
//! `admit_cast_boundary`, called at its B1/B3 sites). Each ALSO has its
//! own runtime dtype gate above the `admit` call (`grad_res`/`base_dtype`
//! must be `BF16`, not merely `lora_linear_fused` admitting) — a
//! `JAMMI_KERNELS_DISABLE=cast_scale_bf16_f32`-only run on a checkpoint
//! where `lora_linear_fused` never admits, OR where the base/grad_res
//! dtype is `F32` (nothing to fuse, see that op's own module doc), is
//! INVALID for the same reason as the attention-block case above.
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

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock, RwLock};

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
    /// the data. Four named classes, each reported with its own reason
    /// string (never collapsed into one generic "capability" text, so a
    /// probe/report reader can tell them apart): the feature was not
    /// compiled; the GPU architecture does not match; the device is not
    /// CUDA at all; or — the fourth class, added for the `attention_block_flash`
    /// cascade's BERT/DistilBERT callers — **the CALLING FORWARD has not
    /// wired this arm's transport protocol** (reason string
    /// `flash_transport_not_wired`): the arm's DEVICE and BUILD are both
    /// capable, but the caller's own forward function has not implemented
    /// the rank-2/unpadded transport this arm's dense path requires (BERT's
    /// and DistilBERT's per-layer forward stays rank-4 `[batch, seq, ...]`
    /// throughout, unlike ModernBERT's `forward_padded_transport_attention`
    /// — see `crates/jammi-encoders/src/attention_cascade.rs`'s module doc
    /// and the plan's R1'/R2' rulings). This is a caller-side gap, not a
    /// device/build fact, so it is counted on the SAME `declined` counter
    /// the other three classes use (never silent — R1' names this as the
    /// unit's own remaining scope gap, not swept under a coarser miss).
    /// [`admit_cascade`] declines to the next arm for every one of these
    /// four; under [`AdmissionMode::Strict`] this is a typed error UNLESS
    /// the caller asserts the next arm can run (`next_arm_can_run`) — see
    /// [`admit_cascade`]'s doc.
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
///
/// **Every decline also records `(op, predicate)` into the SAME
/// probe-capture window `admit_inner` uses** (`record_probe_miss`, this
/// module's "probe-window capture sink" section) — a channel this function
/// used to leave closed, recording declines as counter increments only.
/// The disabled branch records [`DISABLED_PREDICATE_KEY`]; `DomainMiss` and
/// `CapabilityMiss` record `predicate_name`, `CapabilityMiss` BEFORE the
/// `mode` match so a `Strict`-mode hard error still lands an entry (mirrors
/// `admit_inner`'s own placement). A caller with an armed
/// [`probe_capture_begin`] window can therefore read
/// [`probe_capture_reason_for`] for a cascade op exactly as it would for a
/// two-arm [`admit`] op.
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
        // Campaign #446 finding 3's channel, closed here too (audit round,
        // item 3): the SAME probe-capture window `admit_inner` records
        // into, independent of `warn_disabled_once`'s log-once dedupe.
        record_probe_miss(op, DISABLED_PREDICATE_KEY);
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
            record_probe_miss(op, predicate_name);
            Ok(CascadeOutcome::Declined)
        }
        PredicateOutcome::CapabilityMiss => {
            counters.declined.fetch_add(1, Ordering::Relaxed);
            // Recorded BEFORE the `mode` match, mirroring `admit_inner`'s
            // own placement: a `Strict`-mode hard error returns from the
            // arm below without ever reaching a log call, and a window
            // that saw `declined` move but has no entry for why would be
            // exactly the blind spot this sink exists to close.
            record_probe_miss(op, predicate_name);
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

/// Probes the driver-reported name of a candle [`candle_core::Device`]'s
/// underlying CUDA device (e.g. `"NVIDIA L40S"`), if it is a CUDA device
/// this build can query.
///
/// Mirrors [`probe_cuda_compute_capability`] exactly: same
/// `CudaDevice::cuda_stream().context()` handle (no fresh
/// `CudaContext::new(ordinal)`, so no extra thread-binding side effect —
/// see that function's doc for why), same `None`-on-non-CUDA /
/// `None`-without-the-`cuda`-feature / `None`-on-query-failure collapse.
/// This is identification metadata for a print/log header, not an
/// admission predicate: no call site should branch dispatch on it (unlike
/// [`ComputeCapability`], which several call sites gate on).
pub fn probe_cuda_device_name(device: &candle_core::Device) -> Option<String> {
    #[cfg(feature = "cuda")]
    {
        if let candle_core::Device::Cuda(cuda_device) = device {
            if let Ok(name) = cuda_device.cuda_stream().context().name() {
                return Some(name);
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

// =============================================================================
// The probe-window capture sink (campaign #446 finding 3)
// =============================================================================
//
// [`fallback_warnings_emitted`] above is a LOG-ONCE record: every entry is
// pushed inside `warn_fallback_once_with_message`'s
// `seen.insert((op, predicate))` guard, so a given `(op, predicate)` pair
// appears AT MOST ONCE per process. That makes it a fine oracle for "did this
// process ever warn about X", and a WRONG source for "why did THIS job's
// probe miss": a job whose miss repeats a pair some earlier job already
// burned pushes nothing, and a reader taking the most recent entry for the op
// gets the most recent DIFFERENT predicate — a fabricated reason, persisted
// durably on the job record.
//
// A naive before/after window over `fallback_warnings_emitted()` cannot fix
// that either: the dedupe is UPSTREAM of the record, so the window is empty
// in exactly the repeat case that needs it. The sink below is therefore a
// SECOND, independent channel: while armed, `admit_inner`'s miss path ALWAYS
// records `(op, predicate)` here, whatever the log-once dedupe decides (which
// stays exactly as it is, for logging).
//
// **`admit_cascade`'s decline path uses this SAME sink (audit round,
// jammi-kernels item 3).** `admit_cascade` used to record a decline as a
// counter increment only — `probe_capture_reason_for` had no entry for a
// cascade op at all, so a caller (e.g. `jammi-ai`'s esc-075 probe) could see
// `attention_block_flash`'s `declined` counter move but never learn WHY.
// Every one of `admit_cascade`'s three decline shapes (disabled,
// `DomainMiss`, `CapabilityMiss` — including the `Strict`+`CapabilityMiss`
// hard-error case, recorded BEFORE the mode match for the same reason
// `admit_inner` records before ITS mode match) now calls
// [`record_probe_miss`] with the identical `(op, predicate)` shape, so
// `probe_capture_reason_for("attention_block_flash")` reads
// `flash_transport_not_wired` for a BERT-family forward exactly the way it
// reads a two-arm op's predicate name. The counters this closes NOTHING
// about: `CascadeDispatchCounters`'s `fused`/`declined` fields are
// unchanged, byte-identical to before this addition.

/// The predicate key `warn_disabled_once` logs and the sink records for a
/// `JAMMI_KERNELS_DISABLE`-forced eager arm. Hoisted to a `const` so the two
/// producers cannot drift: a consumer distinguishing "deliberate instruction"
/// from "domain-predicate failure" compares against this ONE spelling.
pub const DISABLED_PREDICATE_KEY: &str = "disabled_by_JAMMI_KERNELS_DISABLE";

/// One captured miss: `(op, predicate_key)` — the same two fields
/// [`fallback_warnings_emitted`] carries, without the formatted message
/// (a consumer reading this wants the verbatim key, not log prose).
pub type ProbeMiss = (&'static str, &'static str);

thread_local! {
    /// The armed probe window for THIS thread, or `None` when unarmed.
    ///
    /// **Thread-local, not process-wide, on purpose**: a process-wide sink
    /// would mix a concurrently-probing second worker's misses into this
    /// job's window, which is the same misattribution the process-lifetime
    /// warn list already commits. Thread-local means a window captures
    /// exactly the misses raised by the thread that armed it.
    ///
    /// `None` when unarmed — the hot path pays one TLS access plus an
    /// `Option` check and allocates nothing (pinned by
    /// `unarmed_admit_records_nothing_and_keeps_the_sink_none`).
    static PROBE_CAPTURE: RefCell<Option<Vec<ProbeMiss>>> = const { RefCell::new(None) };

    /// The identity of the window currently armed on this thread, or
    /// [`NO_WINDOW`] when unarmed — the token a [`ProbeCaptureGuard`] checks
    /// itself against before it restores anything (campaign #446 round-1
    /// advisory: nested windows finished OUT OF ORDER would otherwise
    /// misattribute entries; see [`ProbeCaptureGuard::restore`]).
    ///
    /// Read and written ONLY by `probe_capture_begin`/`ProbeCaptureGuard` —
    /// never by `record_probe_miss`, so the armed and unarmed hot paths are
    /// byte-for-byte what they were.
    static ARMED_WINDOW: Cell<u64> = const { Cell::new(NO_WINDOW) };

    /// This thread's monotonic window-token source. Per-thread and
    /// deterministic (never an RNG, family J): tokens are only ever compared
    /// for equality against a guard armed on the SAME thread, so two threads
    /// minting the same number is not a collision.
    static NEXT_WINDOW_TOKEN: Cell<u64> = const { Cell::new(FIRST_WINDOW) };
}

/// The [`ARMED_WINDOW`] value meaning "no window armed on this thread". Never
/// minted as a token, so a guard can never match it.
const NO_WINDOW: u64 = 0;

/// The first token [`NEXT_WINDOW_TOKEN`] mints.
const FIRST_WINDOW: u64 = 1;

/// The refusal message [`ProbeCaptureGuard`] panics with when a window is
/// finished or dropped out of order. Hoisted so the `finish` and `Drop`
/// spellings cannot drift, and so the test that pins the shape asserts
/// against the one text.
const OUT_OF_ORDER_WINDOW: &str =
    "jammi-kernels: probe-capture windows must be finished innermost-first — this guard armed a \
     window that is no longer the one armed on this thread, so restoring it would hand a NESTED \
     window's entries to the wrong probe (and destroy the inner window's own). Finish or drop \
     the inner ProbeCaptureGuard first.";

/// Records `(op, predicate)` into this thread's armed probe window, if one is
/// armed. A no-op (and non-allocating) otherwise.
///
/// **Bounded by DISTINCT pairs, not by miss COUNT** (family E: bound the term
/// that grows). The number of miss EVENTS in a window is caller-controlled —
/// it scales with the probed model's layer count, which comes from the job
/// spec. The number of DISTINCT `(op, predicate)` pairs does not: both fields
/// are `&'static str` literals from a finite, compile-time set of admission
/// call sites, so deduplicating on insert bounds the sink by the workspace's
/// own op/predicate cardinality regardless of how large a model a caller
/// submits. Duplicates carry no information for the consumer either — it asks
/// "which predicate failed for this op", not "how many times".
///
/// The linear `contains` scan is over that same tiny set (single-digit
/// entries in every real window); a `HashSet` would cost more to allocate
/// than it saves, and would lose the insertion order
/// [`probe_capture_reason_for`] resolves ties with.
fn record_probe_miss(op: &'static str, predicate: &'static str) {
    PROBE_CAPTURE.with(|slot| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            if let Some(sink) = slot.as_mut() {
                if !sink.contains(&(op, predicate)) {
                    sink.push((op, predicate));
                }
            }
        }
    });
}

/// Whether this thread currently has an armed probe window — the oracle a
/// test asserts the hot path stays clean against (`None` when unarmed), and
/// the honest answer to "would a miss right now be captured".
pub fn probe_capture_is_armed() -> bool {
    PROBE_CAPTURE.with(|slot| slot.borrow().is_some())
}

/// Arms a probe-window capture sink on THIS thread and returns the guard that
/// owns it. While armed, EVERY [`admit`] miss on this thread records its
/// `(op, predicate)` pair into the window, independent of the log-once dedupe
/// [`fallback_warnings_emitted`] applies (which is left exactly as it is —
/// this does not change what gets logged).
///
/// **Thread-locality is a real constraint on the caller.** The window
/// captures only misses raised on the arming thread. `jammi-ai`'s esc-075
/// probe satisfies this: `run_fine_tune_blocking` runs inside one
/// `tokio::task::spawn_blocking` closure, and the probe's encoder forward,
/// `Tensor::backward()` graph walk and `AdamW::step` are all synchronous
/// calls on that single thread — candle's own intra-kernel parallelism sits
/// BELOW the `admit()` call sites (inside gemm/rayon kernels), never around
/// them. It would break if a future caller (a) armed the window and then
/// awaited across a runtime yield point, so the probe resumed on a different
/// worker thread, or (b) dispatched an admission-gated op from inside a
/// `rayon`/`std::thread::spawn` closure — a data-parallel or multi-GPU arm is
/// the realistic shape. Either case degrades HONESTLY, not silently: the
/// window simply has no entry for that op and the consumer writes its own
/// "reason unavailable" marker rather than a guess.
///
/// Dropping the guard without calling [`ProbeCaptureGuard::finish`] disarms
/// and discards the window.
///
/// **Windows nest strictly.** Each call mints a fresh per-thread token and
/// records it as this thread's armed window; the returned guard restores only
/// while it still owns that token. Finishing or dropping guards out of order
/// is REFUSED with a panic rather than silently misattributing entries — see
/// [`ProbeCaptureGuard::finish`], and `ProbeCaptureGuard::restore` (private)
/// for why restoring out of order would misattribute twice over.
#[must_use = "the window is disarmed as soon as the guard drops; bind it for the probe's \
              duration and call finish() to read it"]
pub fn probe_capture_begin() -> ProbeCaptureGuard {
    let token = NEXT_WINDOW_TOKEN.with(|next| {
        let token = next.get();
        // Saturating, not wrapping: a wrap could re-mint a token an
        // outer guard still holds, which is exactly the aliasing the token
        // exists to detect. `u64::MAX` windows on one thread is not a
        // reachable count, and pinning at the ceiling degrades to "every
        // further window shares one token" — loud (an inner guard would then
        // wrongly pass the check) only in a scenario that cannot occur, and
        // never silently wrong for the first 2^64 - 1 windows.
        next.set(token.saturating_add(1));
        token
    });
    let previous = PROBE_CAPTURE.with(|slot| slot.borrow_mut().replace(Vec::new()));
    let previous_token = ARMED_WINDOW.with(|armed| armed.replace(token));
    ProbeCaptureGuard {
        previous,
        previous_token,
        token,
        restored: false,
    }
}

/// The RAII owner of an armed probe window — see [`probe_capture_begin`].
pub struct ProbeCaptureGuard {
    /// Whatever window was armed on this thread when this guard armed its own
    /// (`None` in every real use — nested windows are not a shape this
    /// codebase has). Restored on `finish`/drop so a nested window cannot
    /// silently destroy its parent's; an inner window's entries are NOT
    /// merged into the outer one (the inner probe's misses are the inner
    /// probe's, not the outer's).
    previous: Option<Vec<ProbeMiss>>,
    /// The [`ARMED_WINDOW`] token in force when this guard armed its own —
    /// restored alongside `previous`, so the token and the sink always move
    /// together.
    previous_token: u64,
    /// This guard's own window identity, minted by [`probe_capture_begin`].
    /// A restore is legal only while [`ARMED_WINDOW`] still equals this.
    token: u64,
    /// Set by the first `restore` so `finish` followed by `drop` restores
    /// once, not twice (a second restore would clobber `previous` back to
    /// `None` after `finish` had just put it back). Also set by a REFUSED
    /// restore, so a refusal cannot repeat from `Drop` during its own unwind.
    restored: bool,
}

impl ProbeCaptureGuard {
    /// Whether the window this guard armed is still the one armed on this
    /// thread — false exactly in the out-of-order shape (an inner window was
    /// armed after this one and has not been finished/dropped yet).
    fn owns_the_armed_window(&self) -> bool {
        ARMED_WINDOW.with(|armed| armed.get()) == self.token
    }

    /// Disarms this window and returns its entries, or `None` if already
    /// disarmed.
    ///
    /// # Panics
    ///
    /// Panics with [`OUT_OF_ORDER_WINDOW`] when this guard no longer owns the
    /// window armed on this thread — i.e. a nested window was armed after it
    /// and is still live. Restoring here would take the INNER window's
    /// entries (the sink holds the innermost window, not this guard's) and
    /// then overwrite the sink with this guard's `previous`, destroying the
    /// inner window outright: two misattributions in one move. The advisory
    /// that prompted this is not reachable today — `jammi-ai`'s esc-075 probe
    /// is the only caller and never nests — so this is a REFUSAL that makes
    /// the shape impossible to introduce silently, not a recovery from a live
    /// bug. The sink is deliberately left untouched on refusal: the inner
    /// window keeps its own entries and its own guard still restores
    /// correctly.
    fn restore(&mut self) -> Option<Vec<ProbeMiss>> {
        if self.restored {
            return None;
        }
        // Marked spent BEFORE the refusal, so this guard's own `Drop` (which
        // runs during the panic's unwind) finds nothing to do rather than
        // panicking a second time — a panic-in-panic aborts the process,
        // which would replace a legible refusal with a bare SIGABRT.
        self.restored = true;
        assert!(self.owns_the_armed_window(), "{OUT_OF_ORDER_WINDOW}");
        ARMED_WINDOW.with(|armed| armed.set(self.previous_token));
        PROBE_CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            let taken = slot.take();
            *slot = self.previous.take();
            taken
        })
    }

    /// Disarms the window and returns every DISTINCT `(op, predicate)` miss
    /// recorded on this thread while it was armed, in first-occurrence order.
    ///
    /// # Panics
    ///
    /// Panics when this window is not the innermost one armed on this thread:
    /// the sink holds the INNER window, so restoring here would hand the
    /// inner probe's entries to this one and destroy the inner window in the
    /// same move. The refusal touches nothing (see the private
    /// `ProbeCaptureGuard::restore`).
    pub fn finish(mut self) -> Vec<ProbeMiss> {
        self.restore().unwrap_or_default()
    }
}

impl Drop for ProbeCaptureGuard {
    /// Restores this thread's window, refusing the out-of-order shape the
    /// same way [`ProbeCaptureGuard::finish`] does — with one concession the
    /// `finish` path does not need: a `Drop` that is ALREADY running inside
    /// someone else's unwind cannot panic (that aborts the process), so it
    /// refuses silently there. Refusing means leaving the sink alone, which
    /// is the fail-safe direction in both cases.
    fn drop(&mut self) {
        if self.restored {
            return;
        }
        if !self.owns_the_armed_window() {
            self.restored = true;
            if std::thread::panicking() {
                return;
            }
            panic!("{OUT_OF_ORDER_WINDOW}");
        }
        let _ = self.restore();
    }
}

/// The verbatim predicate key `window` recorded for `op`, or `None` if this
/// window has no entry for it.
///
/// **First occurrence, not last.** A window CAN in principle hold more than
/// one distinct predicate for one op (heterogeneous layers reaching different
/// branches of the same predicate); the earliest miss is the one this
/// reports, deterministically, and this fn never invents a summary key for
/// the multi-predicate case. `None` is the honest answer a caller must
/// surface as its own "unavailable" marker rather than filling in — the
/// realistic causes are the thread-locality limits in
/// [`probe_capture_begin`]'s doc and a concurrent thread's dispatch moving
/// the counter this window's owner then attributed to itself.
pub fn probe_capture_reason_for(window: &[ProbeMiss], op: &str) -> Option<&'static str> {
    window
        .iter()
        .find(|(recorded_op, _)| *recorded_op == op)
        .map(|&(_, predicate)| predicate)
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
        DISABLED_PREDICATE_KEY,
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
        // Campaign #446 finding 3: the probe window records EVERY miss,
        // independent of `warn_disabled_once`'s log-once dedupe below — a
        // repeat of an already-warned `(op, predicate)` pair still belongs to
        // the job whose window is armed right now.
        record_probe_miss(op, DISABLED_PREDICATE_KEY);
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
    // Recorded BEFORE the mode match, so a `Strict` run's hard error is
    // captured too: `Strict` returns before `warn_predicate_failed_once` is
    // ever reached, and a window that saw the counter move but has no entry
    // for why would be exactly the blind spot this sink exists to close.
    record_probe_miss(op, predicate_name);
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

/// Whether `d` is a device every fused CPU/CUDA-backed `apply2`/`apply3`
/// site in this crate can actually run on: CPU always, and CUDA only when
/// THIS BUILD compiled jammi-kernels' `cuda` feature (`cfg!(feature =
/// "cuda")`).
///
/// **This function gates `apply2`/`apply3` sites specifically — it is not
/// "does ANY op in this crate have a `metal_fwd`".** Metal is refused
/// unconditionally here because no BINARY/TERNARY op this function gates
/// (`LowRankResidualLinear`, `ScaledCastAdd`, and every other
/// `CustomOp2`/`CustomOp3` this crate ships) has a `metal_fwd`, and
/// candle's default `metal_fwd` ERRORS rather than falling back, so a
/// Metal tensor reaching `apply2`/`apply3` would turn a working eager
/// forward into a hard error rather than a clean fallback; refusing it here,
/// before the tensor ever reaches one of those ops, is what keeps the
/// fallback clean. `ops::DropoutFused` (a UNARY `CustomOp1`, reached only
/// through `apply1`, never through this predicate) is the one exception in
/// this crate — see its module doc's "Metal: a device-scoped deterministic
/// host fallback" section (issue #433) — but that does not change this
/// function's answer for `apply2`/`apply3`'s own device set, which stays
/// CPU/CUDA-only until one of those ops grows a real `metal_fwd` too.
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

/// Whether THIS crate (`jammi-kernels`) was compiled with the `cuda`
/// feature — a plain `const`, unconditionally compiled (never behind
/// `#[cfg(feature = "cuda")]` itself), for exactly the reason
/// [`FLASH_COMPILED`] is: a downstream crate (or a process-identity report
/// a build-features identity field feeds) can read the real answer without
/// forwarding the feature through its own `Cargo.toml`. `flash-attn`
/// DEPENDS on `cuda` (`Cargo.toml`'s `flash-attn = ["cuda"]`), so
/// `FLASH_COMPILED` implies `CUDA_COMPILED` — never the reverse — for every
/// build this crate can produce.
pub const CUDA_COMPILED: bool = cfg!(feature = "cuda");

/// Parses a comma-joined `JAMMI_FLASH_GENCODE_SMS`-shaped value (e.g.
/// `"80,86,89,90"`) into the compute capabilities it names, in the SAME
/// order the comma list gives them. Panics on a malformed token — mirrors
/// `crate::flash`'s own per-token parser's "this is compile-time-pinned
/// `build.rs` output, not untrusted user input" contract: a malformed
/// value here can only mean `build.rs` and this function have drifted,
/// which must fail loud, not silently produce an empty or partial set.
/// Pure and free of any env/feature read — [`flash_built_arches`] is the
/// only caller, and it is what decides WHETHER to call this at all.
fn parse_gencode_sms(sms: &str) -> Vec<ComputeCapability> {
    sms.split(',')
        .map(|sm| {
            if sm.len() < 2 || !sm.chars().all(|c| c.is_ascii_digit()) {
                panic!(
                    "jammi-kernels admission: JAMMI_FLASH_GENCODE_SMS token {sm:?} is not at \
                     least two ASCII digits — build.rs and admission.rs have drifted"
                );
            }
            let split = sm.len() - 1;
            let (major_s, minor_s) = sm.split_at(split);
            let major: usize = major_s.parse().unwrap_or_else(|_| {
                panic!(
                    "jammi-kernels admission: JAMMI_FLASH_GENCODE_SMS token {sm:?}: major \
                     digits {major_s:?} do not parse as usize"
                )
            });
            let minor: usize = minor_s.parse().unwrap_or_else(|_| {
                panic!(
                    "jammi-kernels admission: JAMMI_FLASH_GENCODE_SMS token {sm:?}: minor \
                     digit {minor_s:?} does not parse as usize"
                )
            });
            ComputeCapability::new(major, minor)
        })
        .collect()
}

/// The full set of compute capabilities `build.rs::build_flash_attn`
/// compiled native cubins for — parsed from `JAMMI_FLASH_GENCODE_SMS`
/// (`cargo:rustc-env`, emitted UNCONDITIONALLY by `build.rs::main` in
/// every feature configuration, so `env!()` always compiles regardless of
/// this crate's own feature set — see that emission's own doc comment).
///
/// Returns `&[]` whenever [`FLASH_COMPILED`] is `false`: even though the
/// env var is always PRESENT (it is a `build.rs`-time string, never itself
/// feature-gated), an arch this build never actually compiled a cubin for
/// is not "built" in any sense a caller should trust — this mirrors
/// [`FLASH_COMPILED`]'s own "truthful in every cfg" contract (a plain,
/// unconditionally-compiled accessor whose ANSWER still reflects the real
/// feature state, M3 plan v2 delta 3) rather than [`CUDA_COMPILED`]-style
/// blind trust in a string that may not correspond to anything this build
/// actually produced.
///
/// `admitted != compiled` in general (M3 plan D4). ROUND-2 AUDIT FINDING
/// C: an earlier revision of this crate had NO type distinguishing
/// "compiled" from "validated" at all — `crate::flash::check_arch` and
/// every fence site in `jammi-encoders`/`jammi-bench` read THIS function
/// directly as their admission set, so any arch added to `build.rs`'s
/// `GENCODE_ARCHES` was ADMITTED the moment it compiled, with zero pod
/// evidence required (the auditor proved this concretely by adding a
/// hypothetical `sm_100` entry and watching the entire hermetic battery
/// stay green). This function is now DELIBERATELY not an enforcement
/// point: it answers ONLY "did `build.rs` compile a cubin for this arch",
/// which is necessary but not sufficient for admission — no fence site
/// reads it directly anymore. [`flash_validated_arches`] (below) is the
/// actual enforcement point every fence reads; it is asserted (by a
/// hermetic test) to be a SUBSET of what this function returns.
pub fn flash_built_arches() -> &'static [ComputeCapability] {
    static ARCHES: LazyLock<Vec<ComputeCapability>> =
        LazyLock::new(|| parse_gencode_sms(env!("JAMMI_FLASH_GENCODE_SMS")));
    if FLASH_COMPILED {
        ARCHES.as_slice()
    } else {
        &[]
    }
}

/// The SUBSET of [`flash_built_arches`] with an actual green per-arch pod
/// parity leg — parsed from `JAMMI_FLASH_VALIDATED_SMS` (`build.rs`'s
/// `VALIDATED_SMS` const, emitted the SAME unconditional way as
/// `JAMMI_FLASH_GENCODE_SMS` — see that emission's own doc comment).
///
/// **THIS is the actual admission gate** (round-2 audit finding C):
/// `crate::flash::check_arch`, `jammi-encoders::modernbert`'s
/// `flash_arch_ok`, and `jammi-bench`'s `flash_capable_cuda` all read
/// THIS function, not [`flash_built_arches`] — a device outside this set
/// is refused even if its arch IS compiled (`FlashError::Arch` /
/// `"arch_in_flash_validated_set"`), because "compiled" alone was proven
/// (by the round-2 audit's own `sm_100` experiment) to be an insufficient
/// admission criterion on its own. Same `FLASH_COMPILED`-gated `&[]`
/// degrade as [`flash_built_arches`], for the same reason (M3 plan v2
/// delta 3's "truthful in every cfg" contract).
///
/// Widening this set is its OWN, separately-reviewable commit — never
/// bundled with a `GENCODE_ARCHES` addition in the same diff — because
/// widening it is exactly the step that MUST be gated on a green pod
/// parity artifact actually landing (`build.rs::VALIDATED_SMS`'s own doc;
/// `third_party/flash-attention/VENDORED.md`'s "Supported archs" per-arch
/// table names the evidence for each currently-validated entry).
pub fn flash_validated_arches() -> &'static [ComputeCapability] {
    static ARCHES: LazyLock<Vec<ComputeCapability>> =
        LazyLock::new(|| parse_gencode_sms(env!("JAMMI_FLASH_VALIDATED_SMS")));
    if FLASH_COMPILED {
        ARCHES.as_slice()
    } else {
        &[]
    }
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

// =============================================================================
// The probed-op table (campaign #446 finding 2)
// =============================================================================
//
// ONE static fact about which ops a per-job acceleration report / capability
// probe can attribute to a real dispatch decision, and under which registry
// key. Before this table the same facts were hand-encoded in five unsynced
// places (`crates/jammi-ai/src/fine_tune/worker.rs`'s
// `PROBED_ACCELERATION_OPS` + its `AdmissionProbeSnapshot` struct fields +
// its `two_arm` match, and `crates/jammi-ai/tests/gpu_capability/
// capability_surface.rs`'s `TWO_ARM_OPS` + `KNOWN_NO_DISPATCH_SITE_OPS`),
// which is exactly how the f16 cast-epilogue keys came to be missing from
// the shipped report on the headline dtype.
//
// **Why not `snapshot_all`.** `snapshot_all()` reflects only ops that have
// been looked up via `counters_for` AT LEAST ONCE in this process (its own
// doc says so), so a key set derived from it varies with process history: a
// job that happens to run first in a fresh process would report a strictly
// smaller `ops` key set than the identical job running second. A durable
// per-job artifact whose SHAPE depends on what else the process did is not a
// measurement (family F). This table is a compile-time constant instead, so
// the candidate key set is a pure function of the job's dtype class.

/// The dtype family a [`ProbedOp`]'s registry key is resolved under.
///
/// A probed op's REPORT key is deliberately dtype-NEUTRAL (`"cast_scale"`,
/// never `"cast_scale_bf16_f32"`) because it names the *capability* a
/// consumer asks about; the registry key the kernel's own `admit()` call
/// site passes is NOT dtype-neutral, because each 16-bit cast-boundary
/// kernel is a genuinely independent type (`CastScaleBf16F32` vs
/// `CastScaleF16F32` — see `crate::ops::cast_scale`'s module doc on why an
/// `F16` analog is real kernel authoring, not a reinterpretation), dispatched
/// under its OWN key so `JAMMI_KERNELS_DISABLE` can force each back to its
/// two-kernel chain independently. The registry key is therefore RESOLVED
/// from the job's backbone dtype at probe time, never spelled into the
/// report.
///
/// [`DtypeClass::Any`] on a table ENTRY means "this op dispatches under one
/// key regardless of dtype"; passing `Any` as the QUERY dtype to
/// [`ProbedOp::registry_keys_for`] therefore matches only the dtype-neutral
/// entries — see that method's doc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DtypeClass {
    /// One registry key covering every backbone dtype.
    Any,
    /// `f32` backbones only.
    F32,
    /// `bf16` backbones only.
    Bf16,
    /// `f16` backbones only.
    F16,
}

/// How a [`ProbedOp`]'s dispatch decision is (or is not) observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbedOpKind {
    /// A two-arm [`admit`] site: [`counters_for`]'s `fused`/`eager` split is
    /// the observable, and a miss carries a verbatim predicate key.
    TwoArm,
    /// An [`admit_cascade`] site: [`cascade_counters_for`]'s
    /// `fused`/`declined` split is the observable. `admit_cascade` has NO
    /// `fallback_warnings`-shaped reason channel, so a decline can only ever
    /// be reported at the coarse `capability_or_domain_miss` grain.
    Cascade,
    /// A kernel with NO admission gate of its own: it is launched
    /// unconditionally from INSIDE `parent`'s already-admitted fused arm (a
    /// bare launcher call at the storage level, not a tracked `Tensor` op),
    /// so it has no registry key and no probe can read a delta for it. Its
    /// execution is implied by `parent` dispatching fused — which is what
    /// makes it provable at all, and why it must never be claimed as an
    /// independently-admitting op.
    InternalSubkernel {
        /// The [`ProbedOp::report_key`] of the op whose fused arm launches
        /// this kernel.
        parent: &'static str,
    },
}

/// One row of [`PROBED_OPS`]: a dtype-neutral report key, how its dispatch
/// is observable, and the registry key(s) its kernel's own `admit()` /
/// `admit_cascade()` call site passes, per dtype class.
#[derive(Debug, Clone, Copy)]
pub struct ProbedOp {
    /// The dtype-NEUTRAL key this op appears under in a durable acceleration
    /// report and in `ci/release-feature-manifest.json`'s capability lists.
    pub report_key: &'static str,
    /// How (or whether) this op's dispatch decision can be observed.
    pub kind: ProbedOpKind,
    /// `(dtype class, registry key)` — the key the kernel's own call site
    /// passes to [`admit`]/[`admit_cascade`], reused VERBATIM. Empty for a
    /// [`ProbedOpKind::InternalSubkernel`], which has no key at all.
    pub registry: &'static [(DtypeClass, &'static str)],
}

impl ProbedOp {
    /// The registry key(s) this op dispatches under for a job whose backbone
    /// dtype is `dtype`: every entry whose class is [`DtypeClass::Any`] or
    /// exactly `dtype`.
    ///
    /// Passing [`DtypeClass::Any`] yields ONLY the dtype-neutral entries —
    /// the honest reading of "this caller has no concrete dtype in hand", not
    /// a wildcard that would silently claim both 16-bit keys at once. Use
    /// [`ProbedOp::all_registry_keys`] for the every-dtype enumeration.
    ///
    /// Today every [`ProbedOpKind::TwoArm`]/[`ProbedOpKind::Cascade`] row
    /// yields AT MOST ONE key for any concrete dtype class — pinned by
    /// `probed_ops_resolve_to_at_most_one_registry_key_per_dtype_class`, not
    /// assumed.
    pub fn registry_keys_for(&self, dtype: DtypeClass) -> impl Iterator<Item = &'static str> + '_ {
        self.registry
            .iter()
            .filter(move |(class, _)| *class == DtypeClass::Any || *class == dtype)
            .map(|&(_, key)| key)
    }

    /// Every registry key this op can dispatch under, across all dtype
    /// classes — the enumeration a "is this table's key set closed over the
    /// workspace's real call sites" audit reads.
    pub fn all_registry_keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.registry.iter().map(|&(_, key)| key)
    }
}

/// The ONE static fact about which ops an acceleration report / capability
/// probe can attribute, and under which registry key.
///
/// **Populated by reading every `counters_for("...")` /
/// `cascade_counters_for("...")` / `admit_cast_boundary("...")` literal in
/// `crates/jammi-kernels`, `crates/jammi-encoders`, `crates/jammi-lora` and
/// `crates/jammi-ai` — never from memory.** Row-by-row provenance:
///
/// | report key | kind | registry key(s) | call site |
/// |---|---|---|---|
/// | `layer_norm` | TwoArm | `layer_norm_fused` | `layer_norm_fused`, `crates/jammi-encoders/src/layer_norm.rs:129` |
/// | `rope` | TwoArm | `rope_fused` | `rope_fused`, `crates/jammi-encoders/src/modernbert.rs:181` |
/// | `softmax` | TwoArm | `softmax_last_dim_fused` | `softmax_last_dim_fused`, `crates/jammi-encoders/src/attention_cascade.rs:404` |
/// | `geglu` | TwoArm | `geglu_fused` | `geglu_fused`, `crates/jammi-encoders/src/modernbert.rs:1380` |
/// | `gelu_erf` | TwoArm | `gelu_erf_fused` | `gelu_erf_fused`, `crates/jammi-kernels/src/ops/gelu_erf.rs`'s `GeluErfFused::name()`; the `admit()` CALL SITE is `crates/jammi-encoders/src/activations.rs`'s `gelu_erf(x, training)`, reachable on every training-mode erf-GELU call for a head_dim-agnostic BERT-family MLP — `BertIntermediate::forward` (`bert.rs:296`) and `DistilBertFfn::forward` (`distilbert.rs:211`) both call it. |
/// | `attention_block` | TwoArm | `attention_block_fused` | `attention_block_fused`, `crates/jammi-encoders/src/attention_cascade.rs:399` |
/// | `dropout` | TwoArm | `lora_linear_fused` | `lora_linear_fused`, `crates/jammi-lora/src/lora_linear.rs:1007` (`admit` call site) |
/// | `low_rank_residual_linear` | TwoArm | `lora_linear_fused` | `lora_linear_fused`, `crates/jammi-lora/src/lora_linear.rs:1007` (`admit` call site) |
/// | `cast_scale` | TwoArm | bf16 → `cast_scale_bf16_f32`, f16 → `cast_scale_f16_f32` | `cast_scale_bf16_f32`, `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:1054`; `cast_scale_f16_f32`, `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:1068` |
/// | `cast_add` | TwoArm | bf16 → `cast_add_bf16`, f16 → `cast_add_f16` | `cast_add_bf16`, `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:1153`; `cast_add_f16`, `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:1165` |
/// | `adamw_step` | TwoArm | `adamw_step_fused` | `adamw_step_fused`, `crates/jammi-ai/src/fine_tune/adamw.rs:33` (`admit`, `crates/jammi-ai/src/fine_tune/adamw.rs:257`) |
/// | `mem_efficient_attention` | Cascade | `mem_efficient_attention` | `mem_efficient_attention`, `crates/jammi-encoders/src/attention_cascade.rs:868` |
/// | `rope_positions` | InternalSubkernel(`attention_block_flash`) | — | `rope_positions`, `crates/jammi-kernels/src/ops/flash_attention.rs:645` |
/// | `scaled_cast_add` | InternalSubkernel(`low_rank_residual_linear`) | — | `ScaledCastAdd`, `crates/jammi-kernels/src/ops/low_rank_residual_linear.rs:946` (CPU), `ScaledCastAdd`, `crates/jammi-kernels/src/cuda/low_rank_residual_linear.rs:142` (CUDA) |
///
/// **`adamw_step`: the optimizer's dtype DOMAIN is not a dtype CLASS.**
/// `adamw_step_fused`'s own admission predicate requires
/// `theta`/`first_moment`/`second_moment`/`grad` to be `F32`
/// (`crates/jammi-ai/src/fine_tune/adamw.rs`'s `fused_admission_predicate`,
/// `dtype_f32`), and `crates/jammi-kernels/src/ops/adamw_step.rs`'s module
/// doc names `F32` as the op's only implemented dtype. It is tempting to
/// encode that as `(DtypeClass::F32, "adamw_step_fused")`. **That would be
/// wrong**, and wrong in exactly finding 2's own shape.
///
/// [`DtypeClass`] selects on the JOB'S BACKBONE dtype — it is what a caller
/// resolves a registry key WITH (`crates/jammi-ai/src/fine_tune/worker.rs`'s
/// `dtype_class_of(config.backbone_dtype)`). The optimizer's `F32` domain is
/// a fact about a DIFFERENT tensor set: the TRAINABLE variables, which are
/// `F32` on every backbone. `jammi-lora` refuses a non-`F32` adapter outright
/// (`lora_linear.rs`'s `lora_ab_dtype_f32` predicate), and the trainer builds
/// its `VarBuilder` at `DType::F32` regardless of `backbone_dtype`. So an
/// `f16`- or `bf16`-backbone job's optimizer step admits and dispatches
/// `adamw_step_fused` exactly as an `f32` job's does.
///
/// Gating the row on `DtypeClass::F32` would therefore OMIT `adamw_step` from
/// every `bf16`/`f16` job's report while the op demonstrably dispatched —
/// a silent-eager invisibility on the headline dtype, which is the defect
/// this whole table exists to retire. `Any` is the honest encoding: one
/// registry key covering every backbone dtype. The `F32`-only domain still
/// shows up where it belongs — as this op's own admission PREDICATE, whose
/// failure would be reported as `holds: false` with the verbatim `dtype_f32`
/// key, never as an absent row.
///
/// **Registry keys that exist but are deliberately NOT rows** (each read at
/// the cited call site during this population, and excluded for a stated
/// reason — an omission with no reason is how finding 2 happened):
///
/// - `attention_block_flash` (`crates/jammi-encoders/src/modernbert.rs:1098`,
///   `:1990`) — a real cascade, but the esc-075 report surfaces it through
///   its OWN dedicated top-level `flash` field (with the compiled/device
///   short-circuit reasons a plain `ops` entry cannot express), and
///   `ci/release-feature-manifest.json` declares it as `flash_compiled` +
///   `flash_dtypes`, not as a `fused_op_admission` entry. Adding it here
///   would make the same fact appear twice in one artifact.
/// - `lora_dropout` (`crates/jammi-lora/src/lora_linear.rs:37`) and
///   `lora_epilogue` (`:66`) — registry entries with NO `admit()` call site
///   anywhere: both are documented as "permanently `{fused: 0, eager: 0}`",
///   superseded by `lora_linear_fused`, and kept only for snapshot-schema
///   compatibility. A row for either would put a permanently-unmoving
///   counter in the report.
///
/// **`registry` is empty for a [`ProbedOpKind::InternalSubkernel`] row on
/// purpose**: `rope_positions` and `scaled_cast_add` are launched by a bare
/// call into `crate::cuda::rope_positions::cuda_fwd` /
/// `ScaledCastAdd::{cpu_fwd,cuda_fwd}` from INSIDE an already-admitted parent
/// arm, deliberately bypassing the tracked-`Tensor` op path (and therefore
/// [`admit`]) — see `FlashVarlenAttentionRope`'s own doc for why (candle's
/// tape would otherwise retain the rotated buffer for the whole backward
/// pass). They have no key for any probe to read a delta from; their
/// execution is proven by the PARENT dispatching fused, and must never be
/// claimed as an independent admission.
pub const PROBED_OPS: &[ProbedOp] = &[
    ProbedOp {
        report_key: "layer_norm",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "layer_norm_fused")],
    },
    ProbedOp {
        report_key: "rope",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "rope_fused")],
    },
    ProbedOp {
        report_key: "softmax",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "softmax_last_dim_fused")],
    },
    ProbedOp {
        report_key: "geglu",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "geglu_fused")],
    },
    ProbedOp {
        report_key: "gelu_erf",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "gelu_erf_fused")],
    },
    ProbedOp {
        report_key: "attention_block",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "attention_block_fused")],
    },
    ProbedOp {
        report_key: "dropout",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "lora_linear_fused")],
    },
    ProbedOp {
        report_key: "low_rank_residual_linear",
        kind: ProbedOpKind::TwoArm,
        registry: &[(DtypeClass::Any, "lora_linear_fused")],
    },
    ProbedOp {
        report_key: "cast_scale",
        kind: ProbedOpKind::TwoArm,
        registry: &[
            (DtypeClass::Bf16, "cast_scale_bf16_f32"),
            (DtypeClass::F16, "cast_scale_f16_f32"),
        ],
    },
    ProbedOp {
        report_key: "cast_add",
        kind: ProbedOpKind::TwoArm,
        registry: &[
            (DtypeClass::Bf16, "cast_add_bf16"),
            (DtypeClass::F16, "cast_add_f16"),
        ],
    },
    ProbedOp {
        report_key: "adamw_step",
        kind: ProbedOpKind::TwoArm,
        // `DtypeClass::Any`, NOT `F32` — see this table's "the optimizer's
        // dtype domain is not a dtype CLASS" note. The op's own tensors are
        // F32-only; the JOB's backbone dtype is a different axis, and it is
        // the job's that `DtypeClass` selects on.
        registry: &[(DtypeClass::Any, "adamw_step_fused")],
    },
    ProbedOp {
        report_key: "mem_efficient_attention",
        kind: ProbedOpKind::Cascade,
        registry: &[(DtypeClass::Any, "mem_efficient_attention")],
    },
    ProbedOp {
        report_key: "rope_positions",
        kind: ProbedOpKind::InternalSubkernel {
            parent: "attention_block_flash",
        },
        registry: &[],
    },
    ProbedOp {
        report_key: "scaled_cast_add",
        kind: ProbedOpKind::InternalSubkernel {
            parent: "low_rank_residual_linear",
        },
        registry: &[],
    },
];

/// The [`PROBED_OPS`] row with this `report_key`, or `None`.
pub fn probed_op(report_key: &str) -> Option<&'static ProbedOp> {
    PROBED_OPS.iter().find(|op| op.report_key == report_key)
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

    /// Panics if `JAMMI_REQUIRE_FLASH` is set, since a caller in that lane
    /// must not be allowed to silently skip the flash-arm assertions.
    #[cfg(test)]
    fn require_flash_compiled_or_skip(test_name: &str) {
        if std::env::var_os("JAMMI_REQUIRE_FLASH").is_some() {
            panic!(
                "{test_name}: JAMMI_REQUIRE_FLASH is set but this build's jammi-kernels was \
                 compiled without the flash-attn feature (FLASH_COMPILED=false) -- this lane \
                 must run the real flash arm, not skip it"
            );
        }
    }

    /// The `flash_built_arches()` ACCESSOR's own behavior under this crate's
    /// default (no `flash-attn`) test build: `arches.is_empty()` here proves
    /// only that the `FLASH_COMPILED` gate degrades correctly (M3 plan v2
    /// delta 3) — it does NOT, by itself, prove `GENCODE_ARCHES` still pins
    /// the intended sm80/86/89/90 set, because the early `return` below
    /// skips the pinned-set assertion entirely whenever this crate's own
    /// `flash-attn` feature is off (every hermetic default-feature lane,
    /// which is every lane this agent's own local run and most of CI take).
    /// A round-2 audit (mutant: `GENCODE_ARCHES` rewritten to
    /// `sm_70/sm_80/sm_86` — a REGRESSION, dropping a pre-Ampere floor
    /// violation in AND dropping 89/90) proved this test alone stayed GREEN
    /// against that mutant in the hermetic lane: an earlier revision of
    /// this doc comment claimed this test was what pins the set "in every
    /// hermetic CI/laptop run" — that claim was WRONG. The actual hermetic
    /// pin, which DOES run (and DOES go red on that exact mutant) in every
    /// feature configuration including this crate's default build, is
    /// [`gencode_smss_env_var_matches_the_pinned_build_rs_set`] below — see
    /// that test's own doc for why `env!()` makes it possible.
    ///
    /// Panics rather than silently letting this test skip its
    /// exact-pinned-arch-set assertions when `JAMMI_REQUIRE_FLASH` is set
    /// but this build was not compiled with the `flash-attn` feature
    /// (`FLASH_COMPILED == false`) — mirrors `jammi_encoders::modernbert`'s
    /// own `flash_compiled_or_skip` gate (same env var, same "this lane
    /// must run the real flash arm, not skip it" rationale), narrowed here
    /// to the feature-compilation check alone via
    /// [`require_flash_compiled_or_skip`]: this test has no device to
    /// probe, `flash_built_arches()` is a pure compile-time accessor, so
    /// there is no arch-membership half to check.
    #[test]
    fn flash_built_arches_degrades_to_empty_without_flash_compiled() {
        let arches = flash_built_arches();
        if !FLASH_COMPILED {
            assert!(
                arches.is_empty(),
                "flash-attn not compiled: flash_built_arches() must be empty, not {arches:?}"
            );
            require_flash_compiled_or_skip(
                "flash_built_arches_degrades_to_empty_without_flash_compiled",
            );
            return;
        }
        let want = [
            ComputeCapability::new(8, 0),
            ComputeCapability::new(8, 6),
            ComputeCapability::new(8, 9),
            ComputeCapability::new(9, 0),
        ];
        assert_eq!(arches, want);
        for arch in arches {
            assert!(
                arch.meets_minimum(),
                "{arch:?} must meet MIN_CUDA_COMPUTE_CAP -- every compiled arch is Ampere-or-newer"
            );
        }
        assert_eq!(
            arches.iter().min().copied(),
            Some(ComputeCapability::new(8, 0)),
            "sm80 is the true floor of the compiled set"
        );
    }

    /// THE hermetic pin on `build.rs::GENCODE_ARCHES` — round-2 audit
    /// finding F1's fix. `env!("JAMMI_FLASH_GENCODE_SMS")` reads the REAL
    /// value `build.rs`'s `main()` emitted for THIS crate's OWN
    /// compilation, and `main()` emits it UNCONDITIONALLY (every feature
    /// configuration, not only under `flash-attn` — see that emission's
    /// own doc comment in `build.rs`) — so this assertion is meaningful,
    /// and actually RUNS, in the default hermetic lane, unlike
    /// [`flash_built_arches_degrades_to_empty_without_flash_compiled`]'s
    /// early-return above (which the `flash_built_arches()` ACCESSOR's own
    /// `FLASH_COMPILED` gate short-circuits before ever comparing against
    /// `want` in that same lane). Verified against the audit's own mutant
    /// (`GENCODE_ARCHES` rewritten to a pre-Ampere-inclusive,
    /// 89/90-dropping `sm_70/sm_80/sm_86` set): this test goes RED against
    /// that mutant in a scratch copy — the ONLY one of the three sites the
    /// audit named (this test, `build_rs_unit.rs`'s parse tests,
    /// `flash/mod.rs`'s pin) that actually catches it in a lane this repo's
    /// hermetic gate runs.
    #[test]
    fn gencode_smss_env_var_matches_the_pinned_build_rs_set() {
        assert_eq!(env!("JAMMI_FLASH_GENCODE_SMS"), "80,86,89,90");
        let want = vec![
            ComputeCapability::new(8, 0),
            ComputeCapability::new(8, 6),
            ComputeCapability::new(8, 9),
            ComputeCapability::new(9, 0),
        ];
        let got = parse_gencode_sms(env!("JAMMI_FLASH_GENCODE_SMS"));
        assert_eq!(got, want);
        for arch in &got {
            assert!(
                arch.meets_minimum(),
                "{arch:?} must meet MIN_CUDA_COMPUTE_CAP -- every compiled arch is Ampere-or-newer"
            );
        }
        assert_eq!(
            got.iter().min().copied(),
            Some(ComputeCapability::new(8, 0)),
            "sm80 is the true floor of the compiled set"
        );
    }

    /// Round-2 audit finding C's own hermetic pin: reads
    /// `env!("JAMMI_FLASH_VALIDATED_SMS")` directly (the value
    /// `build.rs::VALIDATED_SMS` produces via `main`'s unconditional
    /// emission — see [`flash_validated_arches`]'s own doc), so this runs,
    /// and is meaningful, in the default hermetic lane, the same way
    /// [`gencode_smss_env_var_matches_the_pinned_build_rs_set`] does for
    /// the compiled set.
    ///
    /// THREE properties, all load-bearing:
    /// 1. Validated is a SUBSET of compiled (`validated ⊆ compiled`) — a
    ///    validated arch that was somehow never even compiled would be an
    ///    impossible, self-contradictory state.
    /// 2. Validated matches its OWN pinned value exactly.
    /// 3. TODAY's additional invariant: validated == compiled (every
    ///    currently-compiled arch also has a green pod parity leg, per
    ///    the lead's own pod-run confirmation for this PR's four arches).
    ///    This is NOT a permanent guarantee — a future `-gencode` addition
    ///    to `GENCODE_ARCHES` legitimately breaks it (the arch stays
    ///    compiled-but-unvalidated until its OWN artifact lands) — but
    ///    it IS what makes the auditor's own `sm_100` mutant (add a
    ///    `-gencode` entry, update `GENCODE_ARCHES`'s own pin, but do
    ///    NOT touch `VALIDATED_SMS`) go RED here: `compiled` grows to 5
    ///    entries, `validated` stays at 4, and property 3's equality
    ///    assertion fails — proving admission cannot silently follow a
    ///    `GENCODE_ARCHES`-only edit anymore. Verified directly against
    ///    the auditor's exact mutant in a scratch copy (see hand-off).
    #[test]
    fn flash_validated_arches_env_var_is_a_pinned_subset_of_compiled() {
        assert_eq!(env!("JAMMI_FLASH_VALIDATED_SMS"), "80,86,89,90");
        let validated = parse_gencode_sms(env!("JAMMI_FLASH_VALIDATED_SMS"));
        let compiled = parse_gencode_sms(env!("JAMMI_FLASH_GENCODE_SMS"));
        let want = vec![
            ComputeCapability::new(8, 0),
            ComputeCapability::new(8, 6),
            ComputeCapability::new(8, 9),
            ComputeCapability::new(9, 0),
        ];
        assert_eq!(validated, want, "validated set must match its own pin");
        for v in &validated {
            assert!(
                compiled.contains(v),
                "{v:?} is claimed VALIDATED but is not even in the COMPILED set -- impossible \
                 state (a validated arch must first be compiled)"
            );
        }
        assert_eq!(
            compiled, validated,
            "TODAY's invariant: every currently compiled arch is ALSO validated (per the lead's \
             own pod-run confirmation for sm80/86/89/90). A compiled arch with no matching \
             VALIDATED_SMS entry means build.rs::GENCODE_ARCHES grew without its own validation \
             entry landing in the SAME commit -- see build.rs::VALIDATED_SMS's own doc for the \
             M3 plan D4 obligation this enforces (round-2 audit finding C)"
        );
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
    fn device_name_probe_on_cpu_device_is_not_applicable() {
        // Same boundary oracle as `probe_on_cpu_device_is_not_applicable`,
        // for the sibling name probe: a non-CUDA device is "not
        // applicable", never a probe failure masquerading as a name.
        assert_eq!(probe_cuda_device_name(&candle_core::Device::Cpu), None);
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

    #[test]
    fn flash_compiled_matches_the_flash_attn_feature_this_binary_was_built_with() {
        assert_eq!(FLASH_COMPILED, cfg!(feature = "flash-attn"));
    }

    #[test]
    fn cuda_compiled_matches_the_cuda_feature_this_binary_was_built_with() {
        assert_eq!(CUDA_COMPILED, cfg!(feature = "cuda"));
    }

    #[test]
    // `#[cfg(not(feature = "cuda"))]`-gated: on a `--features cuda` build
    // (the GPU prove lane, `ci/scripts/runpod_gpu_prove.sh`)
    // `CUDA_COMPILED` is `true`, so this assertion would fail
    // deterministically there — the twin below
    // (`cuda_compiled_is_true_on_a_cuda_build`) pins that arm instead.
    #[cfg(not(feature = "cuda"))]
    // Compile-time-known `false` for THIS (non-cuda) test target; that is
    // precisely the value being pinned, so `assertions_on_constants` is
    // suppressed here rather than obscured behind a non-`assert!` rewrite.
    #[allow(clippy::assertions_on_constants)]
    fn cuda_compiled_is_false_on_a_non_cuda_build() {
        // This test target is built by a plain `cargo test -p jammi-kernels`
        // (no `--features cuda`), so this pins the value a downstream
        // identity field reads on that default build path, not merely that
        // it round-trips `cfg!`.
        assert!(!CUDA_COMPILED);
    }

    #[test]
    // Twin of `cuda_compiled_is_false_on_a_non_cuda_build`, gated the other
    // way: only compiled (and only true) under `--features cuda`.
    #[cfg(feature = "cuda")]
    #[allow(clippy::assertions_on_constants)]
    fn cuda_compiled_is_true_on_a_cuda_build() {
        assert!(CUDA_COMPILED);
    }

    #[test]
    // Valid under both feature sets (an implication, not a fixed value), so
    // this one carries no `#[cfg(...)]` gate — it runs, and holds, on both
    // the CPU lane and the GPU prove lane.
    #[allow(clippy::assertions_on_constants)]
    fn flash_compiled_implies_cuda_compiled() {
        // `flash-attn = ["cuda"]` in `Cargo.toml`: every build that turns on
        // `flash-attn` also turns on `cuda`, so `FLASH_COMPILED` can never
        // be true while `CUDA_COMPILED` is false.
        assert!(!FLASH_COMPILED || CUDA_COMPILED);
    }

    /// Acquire a Metal device for [`device_is_supported_rejects_metal`]'s
    /// own `metal`-feature-only leg, or `None` to skip — unless
    /// `JAMMI_REQUIRE_METAL` is set, in which case a device-acquisition
    /// failure PANICS. Wraps `Device::new_metal(0)` in
    /// `std::panic::catch_unwind`, mirroring `tests/metal_parity.rs`'s own
    /// `metal_device_or_skip`: on at least one real GH `macos-14` runner
    /// `Device::new_metal(0)` does not merely return `Err` on a
    /// missing/broken device — an `objc2` class lookup inside
    /// candle-metal-kernels' `residency_set.rs:18`
    /// (`MTLResidencySetDescriptor`) can PANIC instead, a probe-time
    /// failure mode a bare `Result` cannot model. Catching that panic here
    /// is sound for the same reason `tests/metal_parity.rs`'s own doc
    /// gives: the probe owns no lock and mutates no shared state before
    /// failing, so unwinding out of it leaves nothing poisoned to clean
    /// up. Both failure shapes (a returned `Err`, or a caught panic) fold
    /// into the same skip/require decision below.
    #[cfg(all(test, feature = "metal"))]
    fn metal_device_or_skip(test_name: &str) -> Option<Device> {
        let outcome: std::result::Result<Device, String> =
            match std::panic::catch_unwind(|| Device::new_metal(0)) {
                Ok(Ok(d)) => Ok(d),
                Ok(Err(e)) => Err(e.to_string()),
                Err(payload) => {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "<non-string panic payload>".to_string()
                    };
                    Err(format!("Device::new_metal(0) panicked: {msg}"))
                }
            };
        match outcome {
            Ok(d) => Some(d),
            Err(msg) => {
                if std::env::var_os("JAMMI_REQUIRE_METAL").is_some() {
                    panic!(
                        "{test_name}: JAMMI_REQUIRE_METAL is set but no Metal device is \
                         available: {msg}"
                    );
                }
                eprintln!(
                    "{test_name}: no Metal device available in this build/host -- skipping the \
                     Metal leg"
                );
                None
            }
        }
    }

    #[test]
    fn device_is_supported_rejects_metal() {
        // `device_is_supported` must reject `Device::Metal` STRUCTURALLY —
        // unconditionally, with no `cfg(feature = "metal")` branch of its
        // own body (see its doc: this crate's `apply2`/`apply3` sites have
        // no `metal_fwd`, regardless of whether candle-core itself was
        // compiled with Metal support). This crate's own `metal` feature
        // (added for issue #433, gating ONLY `tests/metal_parity.rs` and
        // `ops::DropoutFused`'s UNARY `metal_fwd`) changes what
        // `candle_core::MetalDevice` even IS at compile time — a real,
        // non-unit struct when active, the dummy unit struct otherwise —
        // so constructing a value of it (not `device_is_supported` itself)
        // is the one place that legitimately needs a `cfg` branch. Mirrors
        // `jammi_encoders::layer_norm::tests::device_is_supported_rejects_metal`.
        #[cfg(feature = "metal")]
        let Some(metal) = metal_device_or_skip("device_is_supported_rejects_metal") else {
            return;
        };
        #[cfg(not(feature = "metal"))]
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

    /// Audit round, item 3: `admit_cascade`'s decline path used to be a
    /// counter-increment-only channel — `probe_capture_reason_for` had no
    /// entry for a cascade op at all, however armed the window was.
    /// Mirrors `armed_window_records_a_miss_even_when_the_log_once_dedupe_
    /// suppresses_it`'s shape for `admit_inner`, one level up: window OPEN
    /// -> a `DomainMiss` decline -> the reason is readable afterward;
    /// window CLOSED -> a later decline records nowhere (the same "hot
    /// path allocates/records nothing while unarmed" property
    /// `unarmed_admit_records_nothing_and_keeps_the_sink_none` pins for
    /// `admit_inner`).
    #[test]
    fn cascade_decline_feeds_the_same_probe_capture_window_admit_inner_uses() {
        let counters = CascadeDispatchCounters::new();

        // Window OPEN.
        let guard = probe_capture_begin();
        let outcome = admit_cascade(
            AdmissionMode::Fallback,
            "cascade_probe_test_op",
            "cascade_probe_test_predicate",
            PredicateOutcome::DomainMiss,
            false,
            &counters,
        )
        .expect("DomainMiss never errors under Fallback");
        assert_eq!(outcome, CascadeOutcome::Declined);
        let window = guard.finish();
        assert_eq!(
            probe_capture_reason_for(&window, "cascade_probe_test_op"),
            Some("cascade_probe_test_predicate"),
            "an armed window must capture a cascade decline's (op, predicate) pair exactly \
             the way it captures admit_inner's"
        );

        // Window CLOSED: a decline with no window armed must not leak into
        // a window armed afterward.
        assert!(!probe_capture_is_armed());
        let outcome = admit_cascade(
            AdmissionMode::Fallback,
            "cascade_probe_test_op_unarmed",
            "cascade_probe_test_predicate",
            PredicateOutcome::CapabilityMiss,
            true,
            &counters,
        )
        .expect("CapabilityMiss with next_arm_can_run never errors under Fallback");
        assert_eq!(outcome, CascadeOutcome::Declined);
        let later_window = probe_capture_begin().finish();
        assert!(
            later_window.is_empty(),
            "a decline recorded with no window armed must not resurface in a window armed \
             afterward, got {later_window:?}"
        );
    }

    /// The disabled branch records `DISABLED_PREDICATE_KEY`, not
    /// `predicate_name` — mirrors `admit_inner`'s own disabled-path record
    /// (`record_probe_miss(op, DISABLED_PREDICATE_KEY)`), so a probe
    /// reader distinguishes "deliberately disabled" from "domain predicate
    /// failed" for a cascade op the identical way it does for a two-arm
    /// one. Hermetic like `cascade_disabled_wins_over_holds_and_over_strict`
    /// above: cannot drive the REAL disabled branch without the env var, so
    /// this proves the reachable half (`op_disabled` is false, `Holds`
    /// records nothing into the window) and leaves the disabled-record
    /// line's correctness to code inspection plus `admit_inner`'s own
    /// identically-shaped, already-tested call.
    #[test]
    fn cascade_holds_records_nothing_into_an_armed_probe_window() {
        let counters = CascadeDispatchCounters::new();
        let guard = probe_capture_begin();
        let outcome = admit_cascade(
            AdmissionMode::Strict,
            "cascade_probe_test_op_holds",
            "cascade_probe_test_predicate_holds",
            PredicateOutcome::Holds,
            true,
            &counters,
        )
        .expect("Holds never errors");
        assert_eq!(outcome, CascadeOutcome::Fused);
        let window = guard.finish();
        assert!(
            window.is_empty(),
            "a Holds outcome (this arm fires) must not record a miss, got {window:?}"
        );
    }

    /// `CapabilityMiss` under `Strict` with no next arm returns `Err` --
    /// this cell proves the probe window still captures the reason BEFORE
    /// that error propagates, matching this function's own doc ("recorded
    /// BEFORE the mode match, mirroring `admit_inner`'s own placement").
    #[test]
    fn cascade_strict_capability_miss_hard_error_still_records_into_the_probe_window() {
        let counters = CascadeDispatchCounters::new();
        let guard = probe_capture_begin();
        let err = admit_cascade(
            AdmissionMode::Strict,
            "cascade_probe_test_op_strict_err",
            "cascade_probe_test_predicate_strict_err",
            PredicateOutcome::CapabilityMiss,
            false,
            &counters,
        )
        .expect_err("Strict CapabilityMiss with no next arm must hard-error");
        assert!(matches!(
            err,
            KernelError::StrictModeFallback {
                op: "cascade_probe_test_op_strict_err",
                predicate: "cascade_probe_test_predicate_strict_err"
            }
        ));
        let window = guard.finish();
        assert_eq!(
            probe_capture_reason_for(&window, "cascade_probe_test_op_strict_err"),
            Some("cascade_probe_test_predicate_strict_err"),
            "a Strict-mode hard error must not skip the probe-capture record — the window \
             would otherwise see `declined` move (via the counter) with no entry for why"
        );
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

    /// Every concrete dtype class. [`DtypeClass::Any`] is deliberately NOT
    /// here: it is a table-ENTRY class ("one key for every dtype"), never a
    /// job's resolved backbone dtype.
    const CONCRETE_DTYPE_CLASSES: &[DtypeClass] =
        &[DtypeClass::F32, DtypeClass::Bf16, DtypeClass::F16];

    /// The invariant [`ProbedOp::registry_keys_for`]'s doc claims and every
    /// consumer relies on: for a CONCRETE dtype class, a two-arm/cascade row
    /// resolves to at most one registry key, so "the key for this op at this
    /// dtype" is well defined and a consumer never has to guess which of two
    /// counters is the op's real dispatch signal.
    #[test]
    fn probed_ops_resolve_to_at_most_one_registry_key_per_dtype_class() {
        for op in PROBED_OPS {
            if matches!(op.kind, ProbedOpKind::InternalSubkernel { .. }) {
                continue;
            }
            for &dtype in CONCRETE_DTYPE_CLASSES {
                let keys: Vec<&str> = op.registry_keys_for(dtype).collect();
                assert!(
                    keys.len() <= 1,
                    "PROBED_OPS row {:?} resolves to {} registry keys at {dtype:?} ({keys:?}) — \
                     a report/capability consumer would have to guess which one is this op's \
                     dispatch signal",
                    op.report_key,
                    keys.len()
                );
            }
        }
    }

    /// A report key must be dtype-NEUTRAL and unique. The `!key.ends_with`
    /// checks are the direct regression guard on finding 2's root cause: the
    /// shipped table spelled a bf16-specific REGISTRY key (`cast_add_bf16`)
    /// into the report's dtype-neutral vocabulary, which is what made the f16
    /// job's report structurally unable to name its own cast epilogue.
    #[test]
    fn probed_ops_report_keys_are_dtype_neutral_and_unique() {
        let mut seen = HashSet::new();
        for op in PROBED_OPS {
            assert!(
                seen.insert(op.report_key),
                "duplicate PROBED_OPS report key {:?}",
                op.report_key
            );
            for suffix in ["_bf16", "_f16", "_f32", "_bf16_f32", "_f16_f32"] {
                assert!(
                    !op.report_key.ends_with(suffix),
                    "PROBED_OPS report key {:?} carries a dtype suffix {suffix:?} — report keys \
                     are dtype-neutral (campaign #446 finding 2); the dtype lives in the \
                     `registry` column, resolved at probe time",
                    op.report_key
                );
            }
        }
    }

    /// An internal-subkernel row has NO registry key (there is no `admit()`
    /// site to read a delta from) and names a PARENT that is itself either a
    /// probed row or the flash cascade the acceleration report surfaces
    /// through its own `flash` field — a parent no consumer can resolve would
    /// make "proven via the parent's dispatch" unprovable.
    #[test]
    fn internal_subkernel_rows_have_no_registry_key_and_a_resolvable_parent() {
        for op in PROBED_OPS {
            let ProbedOpKind::InternalSubkernel { parent } = op.kind else {
                continue;
            };
            assert!(
                op.registry.is_empty(),
                "internal subkernel {:?} must have no registry key — it is launched from inside \
                 {parent:?}'s already-admitted arm, never through admit()",
                op.report_key
            );
            let resolvable = probed_op(parent).is_some() || parent == "attention_block_flash";
            assert!(
                resolvable,
                "internal subkernel {:?} names parent {parent:?}, which is neither a PROBED_OPS \
                 row nor the flash cascade — its execution would be unprovable",
                op.report_key
            );
        }
    }

    /// Every two-arm/cascade row DOES carry at least one registry key (the
    /// mirror of the test above: a probed row with no key is a row a probe
    /// silently drops — exactly the shape `lora_dropout`/`lora_epilogue`
    /// have, which is why neither is a row at all; see this constant's own
    /// doc for that exclusion).
    #[test]
    fn two_arm_and_cascade_rows_all_carry_a_registry_key() {
        for op in PROBED_OPS {
            if matches!(op.kind, ProbedOpKind::InternalSubkernel { .. }) {
                continue;
            }
            assert!(
                !op.registry.is_empty(),
                "PROBED_OPS row {:?} is {:?} but names no registry key — a probe would silently \
                 drop it",
                op.report_key,
                op.kind
            );
        }
    }

    /// [`DtypeClass::Any`] as the QUERY dtype yields only the dtype-neutral
    /// entries, never both 16-bit keys at once — the documented, non-wildcard
    /// reading. A wildcard here would let a caller with no dtype in hand
    /// snapshot `cast_add_bf16` AND `cast_add_f16` and then report whichever
    /// moved, which is the fabrication this table exists to prevent.
    #[test]
    fn registry_keys_for_any_yields_only_dtype_neutral_entries() {
        let cast_add = probed_op("cast_add").expect("cast_add is a PROBED_OPS row");
        assert_eq!(
            cast_add.registry_keys_for(DtypeClass::Any).count(),
            0,
            "cast_add has no dtype-neutral key, so an Any query must yield none"
        );
        assert_eq!(
            cast_add
                .registry_keys_for(DtypeClass::F16)
                .collect::<Vec<_>>(),
            vec!["cast_add_f16"]
        );
        assert_eq!(
            cast_add
                .registry_keys_for(DtypeClass::Bf16)
                .collect::<Vec<_>>(),
            vec!["cast_add_bf16"]
        );
        assert_eq!(
            cast_add.registry_keys_for(DtypeClass::F32).count(),
            0,
            "an f32 backbone takes low_rank_residual_linear's admit()-free \"nothing to fuse\" \
             branch — there is no cast_add key for it, and the report must omit the op rather \
             than claim a miss"
        );

        let layer_norm = probed_op("layer_norm").expect("layer_norm is a PROBED_OPS row");
        for &dtype in CONCRETE_DTYPE_CLASSES {
            assert_eq!(
                layer_norm.registry_keys_for(dtype).collect::<Vec<_>>(),
                vec!["layer_norm_fused"],
                "a dtype-neutral row resolves to the same key at every dtype"
            );
        }
    }

    /// (iv) The hot path must not grow: with no window armed, `admit()`
    /// records nothing and the sink stays `None` — no `Vec` is ever
    /// allocated for the overwhelming majority of dispatches (every
    /// production forward outside a probe). Asserting `probe_capture_is_armed
    /// () == false` AFTER a real miss is the non-vacuous form: a naive
    /// implementation that lazily created the sink on first miss would leave
    /// it `Some` here.
    #[test]
    fn unarmed_admit_records_nothing_and_keeps_the_sink_none() {
        assert!(
            !probe_capture_is_armed(),
            "no window may be armed on a fresh thread"
        );
        let counters = DispatchCounters::new();
        let decision = admit_inner(
            AdmissionMode::Fallback,
            "probe_sink_unarmed_op",
            "probe_sink_unarmed_predicate",
            false,
            false,
            &counters,
        )
        .expect("Fallback mode never errors on a predicate miss");
        assert_eq!(decision.outcome, DispatchOutcome::Eager);
        assert!(
            !probe_capture_is_armed(),
            "an unarmed admit() miss must NOT lazily create a sink — the hot path allocates \
             nothing"
        );
        // And a window armed AFTERWARDS is empty: the earlier miss was
        // genuinely dropped, not buffered somewhere and replayed.
        let window = probe_capture_begin().finish();
        assert!(
            window.is_empty(),
            "a window armed after an unarmed miss must be empty, got {window:?}"
        );
    }

    /// An armed window records the miss AND survives the log-once dedupe: the
    /// SAME `(op, predicate)` pair captured twice, in two successive windows,
    /// even though `fallback_warnings_emitted()` records it only the first
    /// time. This is finding 3's whole mechanism in one test — the second
    /// window is exactly the case a before/after diff of the warn list
    /// reports as empty.
    #[test]
    fn armed_window_records_a_miss_even_when_the_log_once_dedupe_suppresses_it() {
        let counters = DispatchCounters::new();
        let miss = || {
            admit_inner(
                AdmissionMode::Fallback,
                "probe_sink_dedupe_op",
                "probe_sink_dedupe_predicate",
                false,
                false,
                &counters,
            )
            .expect("Fallback mode never errors on a predicate miss");
        };

        let first_guard = probe_capture_begin();
        miss();
        let first = first_guard.finish();
        assert_eq!(
            first,
            vec![("probe_sink_dedupe_op", "probe_sink_dedupe_predicate")]
        );

        let warns_after_first = fallback_warnings_emitted()
            .into_iter()
            .filter(|(op, _, _)| *op == "probe_sink_dedupe_op")
            .count();
        assert_eq!(
            warns_after_first, 1,
            "the log-once record holds exactly one entry for this pair"
        );

        let second_guard = probe_capture_begin();
        miss();
        let second = second_guard.finish();
        assert_eq!(
            second,
            vec![("probe_sink_dedupe_op", "probe_sink_dedupe_predicate")],
            "the SECOND window must carry the pair too — the log-once dedupe governs logging \
             only, never what a probe window may attribute to its own job"
        );
        assert_eq!(
            fallback_warnings_emitted()
                .into_iter()
                .filter(|(op, _, _)| *op == "probe_sink_dedupe_op")
                .count(),
            warns_after_first,
            "and the dedupe itself is UNCHANGED — the sink is a second channel, not a \
             relaxation of the log-once contract"
        );
    }

    /// Distinct pairs only, in first-occurrence order: the sink is bounded by
    /// the workspace's finite `(op, predicate)` cardinality, not by the
    /// caller-controlled number of miss EVENTS (family E). Ten misses over
    /// two pairs yield two entries.
    #[test]
    fn armed_window_deduplicates_pairs_and_keeps_first_occurrence_order() {
        let counters = DispatchCounters::new();
        let guard = probe_capture_begin();
        for i in 0..10 {
            let predicate = if i % 2 == 0 {
                "probe_sink_bound_predicate_a"
            } else {
                "probe_sink_bound_predicate_b"
            };
            admit_inner(
                AdmissionMode::Fallback,
                "probe_sink_bound_op",
                predicate,
                false,
                false,
                &counters,
            )
            .expect("Fallback mode never errors on a predicate miss");
        }
        let window = guard.finish();
        assert_eq!(
            window,
            vec![
                ("probe_sink_bound_op", "probe_sink_bound_predicate_a"),
                ("probe_sink_bound_op", "probe_sink_bound_predicate_b"),
            ],
            "the sink must grow with DISTINCT (op, predicate) pairs, not with miss events"
        );
        assert_eq!(
            probe_capture_reason_for(&window, "probe_sink_bound_op"),
            Some("probe_sink_bound_predicate_a"),
            "first occurrence wins, deterministically"
        );
        assert_eq!(
            probe_capture_reason_for(&window, "an_op_this_window_never_saw"),
            None,
            "an op with no entry must be None — the caller's cue to write its own honest \
             \"unavailable\" marker, never a guess"
        );
    }

    /// (iii) Two threads with concurrent windows do not mix: each captures
    /// only the misses raised on its OWN thread. A process-wide sink would
    /// put both ops in both windows.
    #[test]
    fn concurrent_windows_on_two_threads_do_not_mix() {
        use std::sync::mpsc;

        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let (go_tx, go_rx) = mpsc::channel::<()>();

        let handle = std::thread::spawn(move || {
            let counters = DispatchCounters::new();
            let guard = probe_capture_begin();
            admit_inner(
                AdmissionMode::Fallback,
                "probe_sink_thread_b_op",
                "probe_sink_thread_b_predicate",
                false,
                false,
                &counters,
            )
            .expect("Fallback mode never errors on a predicate miss");
            // Signal that B's window is armed AND populated, then hold it
            // open until A has done its own miss — so the two windows really
            // are concurrent, not merely sequential.
            ready_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            guard.finish()
        });

        let counters = DispatchCounters::new();
        let guard = probe_capture_begin();
        ready_rx.recv().unwrap();
        admit_inner(
            AdmissionMode::Fallback,
            "probe_sink_thread_a_op",
            "probe_sink_thread_a_predicate",
            false,
            false,
            &counters,
        )
        .expect("Fallback mode never errors on a predicate miss");
        let a = guard.finish();
        go_tx.send(()).unwrap();
        let b = handle.join().expect("thread B must not panic");

        assert_eq!(
            a,
            vec![("probe_sink_thread_a_op", "probe_sink_thread_a_predicate")],
            "thread A's window must hold ONLY A's miss, got {a:?}"
        );
        assert_eq!(
            b,
            vec![("probe_sink_thread_b_op", "probe_sink_thread_b_predicate")],
            "thread B's window must hold ONLY B's miss, got {b:?}"
        );
    }

    /// Dropping the guard without `finish` disarms the window — a probe that
    /// panics or returns early must not leave a sink armed for whatever runs
    /// next on this thread (which would then attribute unrelated misses to a
    /// job that is already over).
    #[test]
    fn dropping_the_guard_without_finish_disarms() {
        {
            let _guard = probe_capture_begin();
            assert!(probe_capture_is_armed());
        }
        assert!(
            !probe_capture_is_armed(),
            "the window must be disarmed by Drop, not only by finish()"
        );
    }

    /// The POSITIVE control for the out-of-order refusal below: properly
    /// nested windows (inner finished first) still work exactly as
    /// documented. Without this, a `restore` that refused unconditionally
    /// would "pass" the refusal test while breaking every real caller.
    ///
    /// Also pins the nesting semantics the guard's own doc states: the
    /// inner window's entries are the INNER probe's (never merged upward),
    /// and the outer window's own entries survive the nested window intact.
    #[test]
    fn properly_nested_windows_finish_innermost_first_and_keep_their_own_entries() {
        std::thread::spawn(|| {
            let counters = DispatchCounters::new();
            let miss = |op: &'static str| {
                admit_inner(
                    AdmissionMode::Fallback,
                    op,
                    "probe_sink_nesting_predicate",
                    false,
                    false,
                    &counters,
                )
                .expect("Fallback mode never errors on a predicate miss");
            };

            let outer = probe_capture_begin();
            miss("probe_sink_nesting_outer_op");
            let inner = probe_capture_begin();
            miss("probe_sink_nesting_inner_op");
            assert_eq!(
                inner.finish(),
                vec![(
                    "probe_sink_nesting_inner_op",
                    "probe_sink_nesting_predicate"
                )],
                "the inner window holds ONLY the misses raised while it was armed"
            );
            assert!(
                probe_capture_is_armed(),
                "finishing the inner window must restore the OUTER one, not disarm the thread"
            );
            miss("probe_sink_nesting_outer_op_again");
            assert_eq!(
                outer.finish(),
                vec![
                    (
                        "probe_sink_nesting_outer_op",
                        "probe_sink_nesting_predicate"
                    ),
                    (
                        "probe_sink_nesting_outer_op_again",
                        "probe_sink_nesting_predicate"
                    ),
                ],
                "the outer window keeps its pre-nesting entries AND records again after the \
                 inner window closed; the inner probe's miss is never merged into it"
            );
            assert!(!probe_capture_is_armed());
        })
        .join()
        .expect("the properly nested shape must not panic");
    }

    /// Campaign #446 round-1 advisory: a window finished OUT OF ORDER (an
    /// outer guard finished while an inner one is still armed) must be
    /// REFUSED, loudly, rather than handing the inner window's entries to the
    /// outer probe and destroying the inner window in the same move.
    ///
    /// Not reachable from today's callers — `jammi-ai`'s esc-075 probe is the
    /// only one and never nests — so this test constructs the shape directly.
    /// It runs on its OWN thread for two reasons: the sink is thread-local,
    /// and the refusal deliberately leaves this thread's sink ARMED (the
    /// fail-safe: it touches nothing), which must not leak into a sibling
    /// test sharing the harness thread.
    #[test]
    fn out_of_order_window_finish_is_refused_and_leaves_the_inner_window_intact() {
        std::thread::spawn(|| {
            let counters = DispatchCounters::new();
            let outer = probe_capture_begin();
            let inner = probe_capture_begin();
            admit_inner(
                AdmissionMode::Fallback,
                "probe_sink_out_of_order_op",
                "probe_sink_out_of_order_predicate",
                false,
                false,
                &counters,
            )
            .expect("Fallback mode never errors on a predicate miss");

            // Silence the default hook for the expected panic only, so a
            // refusal this test EXPECTS does not print a scary backtrace
            // while a genuinely unexpected panic elsewhere still does.
            let previous_hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let refused =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || outer.finish()));
            std::panic::set_hook(previous_hook);

            let payload = refused.expect_err(
                "finishing the OUTER window while the inner one is still armed must be refused \
                 — silently restoring would hand the inner probe's entries to the outer one",
            );
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic payload>");
            assert!(
                message.contains("innermost-first"),
                "the refusal must say WHAT is wrong and HOW to fix it, got: {message}"
            );

            // The refusal touched nothing: the inner window is still armed
            // and still owns its entry, so the inner probe's own `finish`
            // remains correct.
            assert!(
                probe_capture_is_armed(),
                "a refused restore must leave the sink alone, not disarm the inner window"
            );
            assert_eq!(
                inner.finish(),
                vec![(
                    "probe_sink_out_of_order_op",
                    "probe_sink_out_of_order_predicate"
                )],
                "the inner window keeps its own entries through the outer guard's refusal"
            );
        })
        .join()
        .expect("the refusal must be a catchable panic on the guard, not a thread-killing one");
    }

    /// The `JAMMI_KERNELS_DISABLE` arm records too, under the ONE hoisted
    /// [`DISABLED_PREDICATE_KEY`] spelling — so a consumer can tell "you
    /// disabled this op" apart from "the domain check failed" without
    /// comparing log prose.
    #[test]
    fn disabled_arm_records_the_hoisted_disabled_predicate_key() {
        let counters = DispatchCounters::new();
        let guard = probe_capture_begin();
        admit_inner(
            AdmissionMode::Strict,
            "probe_sink_disabled_op",
            "a_predicate_that_holds",
            true,
            true,
            &counters,
        )
        .expect("disable wins over Strict");
        let window = guard.finish();
        assert_eq!(
            probe_capture_reason_for(&window, "probe_sink_disabled_op"),
            Some(DISABLED_PREDICATE_KEY),
            "a deliberate disable must be recorded as such, never as a domain-predicate failure"
        );
    }

    /// A `Strict`-mode predicate failure records into the window even though
    /// it returns `Err` before any warn helper runs — the arm that has NO
    /// entry in `fallback_warnings_emitted()` at all.
    #[test]
    fn strict_mode_predicate_failure_still_records_into_the_window() {
        let counters = DispatchCounters::new();
        let guard = probe_capture_begin();
        let err = admit_inner(
            AdmissionMode::Strict,
            "probe_sink_strict_op",
            "probe_sink_strict_predicate",
            false,
            false,
            &counters,
        )
        .expect_err("Strict turns a predicate miss into a hard error");
        assert!(matches!(err, KernelError::StrictModeFallback { .. }));
        let window = guard.finish();
        assert_eq!(
            probe_capture_reason_for(&window, "probe_sink_strict_op"),
            Some("probe_sink_strict_predicate"),
            "Strict returns before warn_predicate_failed_once, so the warn list has nothing — \
             the window must still know why"
        );
        assert!(
            !fallback_warnings_emitted()
                .into_iter()
                .any(|(op, _, _)| op == "probe_sink_strict_op"),
            "control: the warn list genuinely has no entry for this op, so the assertion above \
             is not passing through the old channel by accident"
        );
    }

    /// `all_registry_keys` is the every-dtype enumeration (distinct from
    /// `registry_keys_for`): a "does this table's key set cover the
    /// workspace's real call sites" audit reads it, and it must not collapse
    /// the two 16-bit keys into one.
    #[test]
    fn all_registry_keys_enumerates_every_dtype_variant() {
        let cast_scale = probed_op("cast_scale").expect("cast_scale is a PROBED_OPS row");
        assert_eq!(
            cast_scale.all_registry_keys().collect::<Vec<_>>(),
            vec!["cast_scale_bf16_f32", "cast_scale_f16_f32"]
        );
    }
}
