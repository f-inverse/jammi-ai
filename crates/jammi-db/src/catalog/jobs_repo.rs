//! The `jobs` / `instances` / `workers` catalog tables (migration 029):
//! kind-agnostic durable compute jobs, process liveness, and claim-loop
//! membership.
//!
//! `jobs` generalises the training queue's claim/lease/reclaim machinery
//! (the former `training_jobs` table, `catalog::training_repo`, now
//! removed with no shim) to every kind of durable work a worker can claim —
//! training AND the compute verbs that opt into the queue. Every
//! lease-guarded write on a `running` row carries the full attempt guard
//! `WHERE job_id AND claimed_by = $instance AND status = 'running' AND
//! attempts = $n`: `claimed_by` alone cannot distinguish "the current
//! claimant, mid-run" from "a zombie of a prior attempt this job already
//! moved past via reclaim" when an instance id can, in principle, recur
//! (the same defence the removed `training_repo::record_acceleration_report`
//! documented for the training queue).
//!
//! Two execution modes share the one table (`jobs.execution`):
//!
//!   - **`queued`** — claimed by the poll loop
//!     ([`Catalog::claim_next`]), the only rows `claim_next`'s
//!     `WHERE execution = 'queued'` predicate ever selects.
//!   - **`inline`** — claimed exactly once, by id, in the submitting call
//!     itself ([`Catalog::claim_by_id`]); never selected by `claim_next`.
//!     An inline job has no requeue arm: if its executing process dies, the
//!     row is failed once its owning `instances` row is stale or absent
//!     ([`Catalog::reclaim_expired_jobs`]'s liveness arm), never re-claimed.
//!
//! `instances` is the process-liveness table every process upserts at
//! construction and heartbeats; `workers` is the claim-loop-membership table
//! — only a process actually running the claim loop has a row, naming the
//! `kind`s it claims.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::backend::{BackendError, BackendKind, Row, SqlValue, TxOptions};
use super::instance::{
    decode_devices_json, DeviceFact, GangListing, GangMember, InstanceRegistration, PeerAddr,
};
use super::lease::{
    canonical_stamp_now, instance_liveness_margin, lease_deadline_expr, lease_expired_clause,
    lease_live_clause, stale_before_clause,
};
use super::status::{JobExecution, JobStatus};
use super::Catalog;
use crate::error::{JammiError, Result};
use crate::tenant::TenantId;
use crate::tenant_scope::TenantBinding;

/// A row from the `jobs` catalog table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub kind: String,
    pub tenant_id: Option<TenantId>,
    pub status: String,
    pub execution: String,
    /// The self-contained, producer-owned tagged JSON specification a worker
    /// reconstructs the run from on a fresh process. Stored opaquely — the
    /// typed shape lives in the engine crate that produces and consumes it.
    pub spec: String,
    /// The name of the `result_tables` row this attempt is (or already has)
    /// materialising into, written inside the SAME transaction as that row's
    /// own INSERT (`Catalog::create_result_table`'s job-CAS). `None` until a
    /// producer that stages through a result table sets it.
    pub partial_result: Option<String>,
    /// The tagged JSON terminal payload a successful attempt writes at
    /// finish. `None` until the job completes.
    pub result: Option<String>,
    /// The terminal failure message. `None` until the job fails.
    pub error: Option<String>,
    pub progress_rows_done: Option<i64>,
    pub progress_rows_total: Option<i64>,
    pub progress_phase: Option<String>,
    /// Set by a cancel request; the executor observes it at a checkpoint
    /// boundary. Never cleared back to `false` once set.
    pub cancel_requested: bool,
    /// The PK-keyed base model a training kind fine-tunes from. `None` for
    /// every compute kind.
    pub model_ref: Option<String>,
    /// The NAME-keyed model a training kind registers on finish. `None`
    /// until (and unless) a training kind finalizes.
    pub output_model_id: Option<String>,
    /// The NAME-keyed, FK-free `ModelSource` string a compute verb resolves
    /// its model against. `None` for a training kind.
    pub model_source: Option<String>,
    /// Id of the instance holding the lease, or `None` while queued/unclaimed.
    pub claimed_by: Option<String>,
    pub attempts: u32,
    /// How many times a claimant handed this job's lease back ON PURPOSE
    /// ([`Catalog::release_job_lease`] / [`Catalog::release_jobs_claimed_by`]
    /// — a two-mode shutdown's RELEASE arm), as opposed to letting it
    /// expire. Offsets the reclaim cap: [`Catalog::reclaim_expired_jobs`]
    /// compares `attempts - releases` against its limit, so a deploy storm
    /// of releases never burns the attempts a genuine crash consumes.
    /// Bumped at most once per attempt by construction (every release
    /// statement carries `lease_expires_at IS NOT NULL`, so a second
    /// release of the same lease matches zero rows).
    pub releases: u32,
    /// Lease deadline as a canonical UTC timestamp, or `None` when not leased
    /// — including a `running` row whose holder RELEASED it (see
    /// [`Self::releases`]): such a row is reclaimable at once.
    pub lease_expires_at: Option<String>,
    pub priority: i32,
    /// Migration 024's temporary operator hold: a `false` row is excluded
    /// from `claim_next` without being deleted or given a new status.
    pub claimable: bool,
    /// The job's per-attempt acceleration determination (esc-075), as an
    /// opaque, self-describing JSON payload whose vocabulary the payload's
    /// *producer* owns — matching `spec`'s and `result`'s schema-at-the-
    /// producer deferral, not a closed enum pinned here. Carries forward the
    /// exact contract the removed `training_repo::TrainingJobRecord` field
    /// of the same name documented, generalised from training-only to every
    /// job kind:
    ///
    ///   - `None` (SQL `NULL`) — unknown: a row this code never touched
    ///     (there is no such row on a fresh catalog, since [`Catalog::submit_job`]
    ///     always stamps the pending marker below at insert; a pre-migration-026
    ///     row copied by migration 029 is the one surviving source). Never
    ///     read as "accelerated" or "eager" — it is an honest absence of
    ///     information, not a claim.
    ///   - `Some(json)` — a payload landed. The catalog itself writes exactly
    ///     the pending marker (at [`Catalog::submit_job`] and again on the
    ///     requeue arm of [`Catalog::reclaim_expired_jobs`]) plus the
    ///     `"undetermined"` retirements of that marker — one per terminal
    ///     write. Every other payload — commonly `{"state":"determined", ...}`
    ///     from the claiming instance via
    ///     [`Catalog::record_acceleration_report`], but not limited to that
    ///     one shape — is the producer's to define; the catalog stores it
    ///     byte-for-byte and never inspects, validates, or enumerates it.
    ///
    /// # The pending marker's lifecycle
    ///
    /// `{"state":"pending"}` asserts one sentence: *the job exists and no
    /// claimant has computed a determination YET*. "Yet" is the load-bearing
    /// word — it is only true while the job can still reach the probe. The
    /// moment the row goes TERMINAL, `pending` describes a state that will
    /// never resolve, so the catalog retires it INSIDE each terminal write's
    /// own UPDATE, never at N caller sites: [`Catalog::finish_job`],
    /// [`Catalog::finish_job_with_model`], [`Catalog::fail_job`], and both
    /// terminal arms of [`Catalog::reclaim_expired_jobs`] (lease-exhausted
    /// and inline-executor-dead). The non-terminal requeue arm of
    /// [`Catalog::reclaim_expired_jobs`] instead RESETS the column to
    /// `{"state":"pending"}` unconditionally: the row returns to `queued`
    /// for a NEW attempt that will re-probe, so a `determined` payload from
    /// the dead attempt (which described the hardware/config THAT attempt
    /// saw) must not survive onto a row whose next attempt has not started.
    ///
    /// Every terminal rewrite is strictly `pending`-valued: a payload that
    /// is already `determined` / `not_applicable` / `undetermined` is the
    /// last true thing known about the job and is preserved byte-for-byte,
    /// and a legacy SQL `NULL` stays `NULL` (three-valued `NULL = '…'` is
    /// not true, so the retirement `CASE` falls through) — "unknown" is
    /// never fabricated into a state. Each terminal reason is distinct, so
    /// the retired marker names WHICH edge retired it.
    pub acceleration_report: Option<String>,
    /// The `ArtifactDigest` of the coordinator's
    /// materialized `TrainingSet` — job-scoped, write-once (see
    /// [`Catalog::fill_training_set_identity`]), `NULL` until filled and for
    /// every `world_size == 1` job. Never `Some` while
    /// [`Self::training_set_location`] is `None`, or vice versa — enforced
    /// by migration 033's `CHECK` constraint at the schema edge, not merely
    /// by convention.
    pub training_set_ref: Option<String>,
    /// The `result_tables` NAME the coordinator
    /// materialized the training set under — the one coordinate a rank
    /// resolves with a single tenant-pinned lookup, the strict resolver
    /// [`Catalog::get_result_table_for_tenant`] under the job's own
    /// `tenant_id`. See [`Self::training_set_ref`]'s pairing note.
    pub training_set_location: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

impl JobRecord {
    /// Whether this row is terminal (`completed` or `failed`) — no further
    /// lease-guarded write is accepted, and the row is eligible for
    /// retention once past `[jobs] retention_days`. Derived from
    /// [`JobStatus::is_terminal`] (the ONE terminality predicate,
    /// `status.rs:70-72`) rather than a hand-typed literal set, so a status
    /// added to [`JobStatus::ALL`] is reflected here without a second edit.
    /// An unparseable `status` (never written by this crate) is treated as
    /// non-terminal — fail OPEN on lease-guard eligibility, never silently
    /// terminal.
    pub fn is_terminal(&self) -> bool {
        self.status
            .parse::<JobStatus>()
            .map(|s| s.is_terminal())
            .unwrap_or(false)
    }

    /// Whether this row is terminal AND unsuccessful (`failed`). See
    /// [`JobStatus::is_terminal_unsuccessful`].
    pub fn is_terminal_unsuccessful(&self) -> bool {
        self.status
            .parse::<JobStatus>()
            .map(|s| s.is_terminal_unsuccessful())
            .unwrap_or(false)
    }
}

/// The outcome of [`Catalog::fill_training_set_identity`]'s write-once CAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingSetFillOutcome {
    /// This call's own statement won the CAS: the pair is now set to the
    /// values it passed.
    Filled,
    /// Zero rows updated, but a re-read under the SAME `claimed_by`/
    /// `attempts` found the pair already set to the SAME values this call
    /// passed — a concurrent racer (or a retried caller) observing its own
    /// would-be write, never a second write.
    Reused,
    /// Zero rows updated, and the re-read found either a different
    /// `claimed_by`/`attempts` (the claim moved) or a different pair value —
    /// the attempt ends here with NO terminal write ("no supersession"), a
    /// normal event, not a failure.
    Aborted,
}

/// The coordinator's call-site wrapper of [`TrainingSetFillOutcome`], named
/// in DESIGN.md § 4's own vocabulary ("the coordinator materializes or
/// reuses the training set") — see
/// [`Catalog::materialize_or_reuse_training_set`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingSetAssembly {
    /// This call's own values won the write-once CAS.
    Won,
    /// A pair already existed, set to the SAME values this call passed — a
    /// concurrent racer or a retry observing its own would-be write.
    Reused,
    /// The claim moved (a different claimant/attempt now holds the row, or
    /// an existing pair holds DIFFERENT values) between this call being
    /// issued and its CAS running — NO write happened. The coordinator that
    /// issued this call is no longer the row's current attempt and must
    /// abandon assembly rather than proceed under a pair it does not
    /// actually own.
    Moved,
}

/// The closed set of reasons a coordinator's ASSEMBLY attempt (membership →
/// dispatch, DESIGN.md § 4) did not proceed to a run, each carrying its own
/// cooldown/counting rule
/// (`docs/plans/67-distributed-training/UNITS.md` § U5b-1b-ii). A new
/// variant with no rule is a COMPILE error — `AssemblyOutcome::effect`
/// matches every variant explicitly, never a wildcard arm.
///
/// The rule, by variant (see `AssemblyEffect`):
///
/// - [`Self::Refuted`] / [`Self::AllRootDivergent`] — a genuine, terminal-
///   looking refusal of THIS attempt: counted AND cooled down.
/// - [`Self::Unavailable`] / [`Self::StoreUnavailable`] / [`Self::ShortListed`]
///   — a member-scoped or transient condition (OPS D10): cooled down so the
///   next attempt does not immediately repeat the same failed dispatch, but
///   NOT counted — the job itself did nothing wrong.
/// - [`Self::NoBody`] / [`Self::Drain`] / [`Self::Cancelled`] — assembly
///   never actually ran (no coordinator body to dispatch to, a draining
///   host, an external cancel): NEITHER counted nor cooled down, since no
///   assembly ATTEMPT happened at all.
/// - [`Self::Success`] — assembly proceeded: RESETS both the counter and
///   the cooldown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyOutcome {
    /// The coordinator's own row was refuted at re-verification.
    Refuted,
    /// Every listed member's result-root identity diverged from the
    /// coordinator's own (U5b-1a-A2's root predicate): no admissible member
    /// existed for this attempt at all.
    AllRootDivergent,
    /// A member the listing named answered unavailable/unreachable at
    /// dispatch.
    Unavailable,
    /// The shared object store this attempt's training set would
    /// materialize into was itself unavailable.
    StoreUnavailable,
    /// Fewer members answered than `world_size - 1` requires.
    ShortListed,
    /// No coordinator body exists to run this attempt at all.
    NoBody,
    /// The host is draining (68 OPS) and refuses new assembly.
    Drain,
    /// The job was cancelled before assembly completed.
    Cancelled,
    /// Assembly proceeded to a run.
    Success,
}

/// The effect class `AssemblyOutcome::effect` maps every variant onto — a
/// proper enum, not a `(bool, bool)` pair, so "counted but not cooled down"
/// (a combination the design never calls for) is not even representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssemblyEffect {
    /// Neither counted nor cooled down — the counter and cooldown columns
    /// are left exactly as they were.
    Neither,
    /// Cooled down (a fresh `next_assembly_after`, on the current counter
    /// value), NOT counted.
    CooldownOnly,
    /// Cooled down AND counted: `assembly_failures` bumps by one, and the
    /// fresh cooldown is computed on the NEW (post-bump) count.
    CooldownAndCounted,
    /// Success: both columns reset (`assembly_failures = 0`,
    /// `next_assembly_after = NULL`).
    Reset,
}

impl AssemblyOutcome {
    /// Whether this outcome COUNTS toward `assembly_failures` — the
    /// terminal-refusal class ([`Self::Refuted`], [`Self::AllRootDivergent`]),
    /// which the coordinator leaves for reclaim on its lease (an attempt
    /// spent), as opposed to every other outcome, which costs the job no
    /// attempt (the coordinator hands its lease back at once, OPS D10).
    /// Derived from `Self::effect` — never a second table.
    pub fn counts_toward_failures(self) -> bool {
        self.effect() == AssemblyEffect::CooldownAndCounted
    }

    fn effect(self) -> AssemblyEffect {
        match self {
            Self::Refuted | Self::AllRootDivergent => AssemblyEffect::CooldownAndCounted,
            Self::Unavailable | Self::StoreUnavailable | Self::ShortListed => {
                AssemblyEffect::CooldownOnly
            }
            Self::NoBody | Self::Drain | Self::Cancelled => AssemblyEffect::Neither,
            Self::Success => AssemblyEffect::Reset,
        }
    }
}

/// Bounded exponential backoff on a count of COUNTED assembly failures:
/// `BASE * 2^min(failures, CLAMP_EXPONENT)`, capped at [`CEILING`] — the
/// clamp on the exponent guards the `u32` shift against overflow
/// regardless of how large `failures` grows (an 8-bit exponent already
/// saturates the ceiling: `2s * 2^8 = 512s > 300s`), and the final `.min`
/// pins the documented ceiling exactly: no cooldown this function computes
/// ever exceeds five minutes, however many counted failures accumulate.
const ASSEMBLY_BACKOFF_BASE: Duration = Duration::from_secs(2);
const ASSEMBLY_BACKOFF_CLAMP_EXPONENT: u32 = 8;
/// The documented backoff ceiling: five minutes.
const ASSEMBLY_BACKOFF_CEILING: Duration = Duration::from_secs(300);

fn assembly_backoff(failures: u32) -> Duration {
    let exponent = failures.min(ASSEMBLY_BACKOFF_CLAMP_EXPONENT);
    let multiplier = 1u32
        .checked_shl(exponent)
        .expect("exponent clamped to ASSEMBLY_BACKOFF_CLAMP_EXPONENT, always < 32");
    ASSEMBLY_BACKOFF_BASE
        .saturating_mul(multiplier)
        .min(ASSEMBLY_BACKOFF_CEILING)
}

/// The row [`Catalog::get_job_for_rank`] returns — every field the I-GANG
/// row predicate needs (`docs/rigor/contracts/feat_500-C-U5a-1.md` § A1),
/// computed in ONE statement, plus the three row facts the `world_size > 1`
/// conjunct reads ([`Self::tenant_id`], [`Self::training_set_ref`],
/// [`Self::training_set_location`]). Every column decode here is
/// INFALLIBLE by construction — `tenant_id` is carried as the row's raw
/// text, never parsed here (see that field) — so `Err` from
/// `get_job_for_rank` means the read itself faulted, never that this
/// row's content failed to decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankAdmissionRow {
    pub status: String,
    pub claimed_by: Option<String>,
    pub attempts: u32,
    /// The `jobs.tenant_id` column's raw text (`None` for a job submitted
    /// outside any tenant binding — a GLOBAL job). Carried as TEXT, never
    /// parsed into a [`crate::TenantId`] here, so a value that does not
    /// parse is a ROW FACT the caller (the gang admission handler) refuses,
    /// exactly like an undecodable `world_size` — never an `Err` from the
    /// read that found it. The caller pins every training-set read to the
    /// tenant this text names (I-GANG: tenant is derived from the row, never
    /// accepted from the caller) through the strict resolver
    /// [`Catalog::get_result_table_for_tenant`].
    pub tenant_id: Option<String>,
    /// [`JobRecord::training_set_ref`] as it stands on the row — the
    /// `ArtifactDigest` the coordinator's write-once CAS
    /// ([`Catalog::fill_training_set_identity`]) recorded, `None` until
    /// filled. Read only for a `world_size > 1` row; paired with
    /// [`Self::training_set_location`] at the schema edge (migration 034's
    /// `CHECK`), so the two are `Some` together or `None` together.
    pub training_set_ref: Option<String>,
    /// [`JobRecord::training_set_location`] as it stands on the row — the
    /// `result_tables` NAME a rank resolves with ONE strict tenant-pinned
    /// lookup ([`Catalog::get_result_table_for_tenant`]) under
    /// [`Self::tenant_id`], then verifies against the sidecar manifest's
    /// `artifact` digest (must equal [`Self::training_set_ref`]).
    pub training_set_location: Option<String>,
    /// [`super::lease::LeaseFact`], decoded in RUST from `lease_expires_at`'s
    /// raw stored text (never a SQL-side `col::timestamptz` cast) via
    /// [`super::lease::decode_lease_expires_at`] — infallible by
    /// construction on EITHER backend, exactly like [`Self::world_size`]:
    /// text that does not parse is [`super::lease::LeaseFact::Undecodable`],
    /// a ROW FACT the caller refuses, never a fault of the read that found
    /// it (the resolution of
    /// <https://github.com/f-inverse/jammi-ai/issues/574>).
    pub lease: super::lease::LeaseFact,
    /// The row's `spec` column VERBATIM — the same JSON the claiming worker
    /// reconstructs its run from, and what an admitted member's rank body
    /// reconstructs ITS run from (`jammi-ai`'s `run_member_rank`): the
    /// job's training spec is a row fact a member reads through the
    /// admission it was granted, never a value the coordinator sends on the
    /// wire (DESIGN.md §4: no URL and no spec travels in the `Assign`).
    pub spec: String,
    /// The ROW's own rank count, decoded from the SAME `spec` JSON the
    /// claiming worker reconstructs its run from — never the caller's own
    /// `Assign.world`. A gang admission gate keyed on the caller's claim
    /// instead of this field lets a `world_size > 1` job admit under a
    /// caller-supplied `world = 1`, skipping the training-set pair conjunct
    /// and the sidecar verify entirely — a hazard this field's row-keying
    /// closes.
    ///
    /// Decoded via `world_size_from_spec_json` (module-private): a top-level
    /// `world_size` key, or (the shape `jammi-ai`'s `TrainingCommon` actually
    /// persists) a `world_size` key nested one level under `common`. Absent
    /// either way ⇒ [`WorldSizeFact::Decoded`]`(1)` (the single-rank default
    /// every job kind that never names a rank count implicitly is —
    /// `jammi-db` cannot depend on `jammi-ai` to read its serde default
    /// directly, so this mirrors it independently). A `world_size` key
    /// present but not a valid non-negative rank count (non-numeric,
    /// negative, fractional, or overflowing `u32`), or `spec` text that is
    /// not valid JSON at all, is [`WorldSizeFact::Undecodable`] — a ROW FACT
    /// about the content this claimant wrote, never a fault of the read that
    /// found it: [`Catalog::get_job_for_rank`] still returns `Ok(Some(row))`
    /// with every other column populated, and the caller (the gang admission
    /// handler) decides how to refuse it. `Err` from `get_job_for_rank` means
    /// the read itself faulted — never that this row's `spec` failed to
    /// decode a `world_size`, and never (since
    /// <https://github.com/f-inverse/jammi-ai/issues/574>) that this row's
    /// `lease_expires_at` failed to decode a lease: see [`Self::lease`].
    pub world_size: WorldSizeFact,
}

/// The outcome of decoding [`RankAdmissionRow::world_size`] from a job's
/// `spec` JSON — a row FACT the caller matches on, never a sentinel integer
/// and never an `Option` (whose `None` would read as "absent" rather than
/// "present but unreadable"). A row whose spec does not decode a
/// `world_size` is exactly as real a row as one that does: `Undecodable`
/// carries no further detail (the caller refuses the row outright; the
/// decode failure's own text is not part of the I-GANG row predicate), and
/// deciding what to do with it is the caller's (the gang admission
/// handler's).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldSizeFact {
    /// The row's own rank count, decoded successfully — including the
    /// absent-field default, `WORLD_SIZE_IF_ABSENT`.
    Decoded(u32),
    /// The row's `spec` column did not decode a `world_size`: either the
    /// text is not valid JSON at all, or a `world_size` key (top-level or
    /// nested under `common`) is present but not a valid non-negative rank
    /// count. The row otherwise exists and every other column is populated.
    Undecodable,
}

/// The single-rank default this module stamps when a job's `spec` JSON names
/// no `world_size` at all — every non-training job kind, and every training
/// spec predating the field. Mirrors (independently: `jammi-db` must not
/// depend on `jammi-ai`) the wire default `jammi_ai::fine_tune::spec::
/// TrainingCommon`'s own `#[serde(default = "default_world_size")]` hook
/// falls back to when its JSON carries no count.
const WORLD_SIZE_IF_ABSENT: u32 = 1;

/// Decodes [`RankAdmissionRow::world_size`] from a job's raw `spec` JSON — a
/// targeted field read, not a typed deserialization of the whole spec
/// (`jammi-db` does not know, and must not depend on, `jammi-ai`'s
/// `TrainingSpec` shape). Checks a top-level `world_size` key first, then a
/// `world_size` key nested one level under `common` (the shape
/// `TrainingCommon` actually persists for the two fine-tune spec variants),
/// so a future producer that flattens the field is read the same way a
/// current nested one is. Neither key present ⇒
/// [`WorldSizeFact::Decoded`]`(`[`WORLD_SIZE_IF_ABSENT`]`)`. The key present
/// but not decodable as a `u32` (a string, a negative number, a fraction, or
/// a value overflowing `u32`), or `spec` text that is not valid JSON at all,
/// ⇒ [`WorldSizeFact::Undecodable`] — infallible: a job row's `spec` column
/// is the worker's own reconstruction input, so a value that fails to parse
/// or names something un-decodable is a fact about THIS row's content, never
/// a fault of the read that found it, and this function never returns an
/// `Err` a caller could conflate with a genuine backend/driver failure.
fn world_size_from_spec_json(spec: &str) -> WorldSizeFact {
    let value: serde_json::Value = match serde_json::from_str(spec) {
        Ok(v) => v,
        Err(_) => return WorldSizeFact::Undecodable,
    };
    let field = value
        .get("world_size")
        .or_else(|| value.get("common").and_then(|c| c.get("world_size")));
    match field {
        None => WorldSizeFact::Decoded(WORLD_SIZE_IF_ABSENT),
        Some(v) => match serde_json::from_value::<u32>(v.clone()) {
            Ok(n) => WorldSizeFact::Decoded(n),
            Err(_) => WorldSizeFact::Undecodable,
        },
    }
}

const SELECT_COLS: &str = "job_id, kind, tenant_id, status, execution, spec, partial_result, \
     result, error, progress_rows_done, progress_rows_total, progress_phase, cancel_requested, \
     model_ref, output_model_id, model_source, claimed_by, attempts, releases, \
     lease_expires_at, priority, claimable, acceleration_report, \
     training_set_ref, training_set_location, created_at, updated_at";

/// The explicit submission-time marker [`Catalog::submit_job`] writes into
/// `acceleration_report`: the job exists but no claimant has yet computed an
/// acceleration determination for it (esc-075). Distinct from SQL `NULL` (a
/// row this code never touched) — see [`JobRecord::acceleration_report`]'s
/// producer-owned-payload contract. This is the only value the catalog ever
/// writes that is not a retirement of itself: every terminal write matches
/// these exact bytes and replaces them, so `pending` never survives onto a
/// terminal row.
const ACCELERATION_REPORT_PENDING: &str = r#"{"state":"pending"}"#;

/// The marker [`Catalog::fail_job`] substitutes for a still-`pending` report,
/// in the same attempt-guarded UPDATE that stamps `failed`: the attempt died
/// between the claim and the acceleration probe, so no determination exists
/// and none ever will for this attempt. Byte-for-byte vocabulary; see
/// [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_FAILED_BEFORE_PROBE: &str =
    r#"{"state":"undetermined","reason":"failed_before_probe"}"#;

/// The marker [`Catalog::reclaim_expired_jobs`]'s queued-execution,
/// attempts-exhausted arm substitutes for a still-`pending` report: the
/// claimant is gone (its lease expired) and the job has no attempts left, so
/// the row is terminal with no determination — and, unlike the `fail` path,
/// with no live worker that could ever compensate. Byte-for-byte vocabulary;
/// see [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_LEASE_EXPIRED_EXHAUSTED: &str =
    r#"{"state":"undetermined","reason":"lease_expired_attempts_exhausted"}"#;

/// The marker [`Catalog::reclaim_expired_jobs`]'s inline-execution,
/// instance-dead arm substitutes for a still-`pending` report: an inline job
/// has no requeue arm, so once its owning `instances` row is stale or absent
/// the row is failed outright with no determination and no live worker that
/// could ever compensate. Byte-for-byte vocabulary; see
/// [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_INLINE_EXECUTOR_DIED: &str =
    r#"{"state":"undetermined","reason":"inline_executor_died"}"#;

/// The marker [`Catalog::finish_job`]/[`Catalog::finish_job_with_model`]
/// substitute for a still-`pending` report, in the same attempt-guarded
/// UPDATE that stamps `completed`: the run SUCCEEDED, but nothing ever
/// recorded a determination for it — the claimant's report-persist step is
/// best-effort (it can lose its lease guard, or hit a catalog error, and the
/// run still proceeds to finish), so success is not evidence that a
/// determination exists. Byte-for-byte vocabulary; see
/// [`JobRecord::acceleration_report`]'s lifecycle section.
const ACCELERATION_REPORT_FINALIZED_WITHOUT_DETERMINATION: &str =
    r#"{"state":"undetermined","reason":"finalized_without_determination"}"#;

/// A backend-portable `SET acceleration_report = …` clause that retires the
/// submission-time pending marker and leaves every other payload alone:
/// `$pending_param` is the pending marker to match, `$terminal_param` the
/// marker to write in its place. Rendered into the terminal UPDATEs so the
/// rewrite rides the SAME statement as the status transition — never a
/// read-then-write outside the transaction, and never a second UPDATE whose
/// predicate could drift from the first's.
///
/// The three-valued comparison is the point: a legacy SQL `NULL` is neither
/// equal nor unequal to the marker, so `CASE WHEN NULL = '…'` is not true and
/// the `ELSE` arm preserves the `NULL`. `CASE`/`WHEN`/`ELSE` is core SQL —
/// identical on SQLite and Postgres, so no dialect branch is needed.
fn retire_pending_report_clause(pending_param: usize, terminal_param: usize) -> String {
    format!(
        "acceleration_report = CASE WHEN acceleration_report = ${pending_param} \
         THEN ${terminal_param} ELSE acceleration_report END"
    )
}

fn parse_row(row: &Row<'_>) -> std::result::Result<JobRecord, super::backend::BackendError> {
    let tenant_id = row
        .try_get::<String>("tenant_id")?
        .map(|s| {
            s.parse::<TenantId>()
                .map_err(|e| super::backend::BackendError::TypeConversion {
                    column: "tenant_id".to_string(),
                    detail: e.to_string(),
                })
        })
        .transpose()?;
    Ok(JobRecord {
        job_id: row.get("job_id")?,
        kind: row.get("kind")?,
        tenant_id,
        status: row.get("status")?,
        execution: row.get("execution")?,
        spec: row.get("spec")?,
        partial_result: row.try_get("partial_result")?,
        result: row.try_get("result")?,
        error: row.try_get("error")?,
        progress_rows_done: row.try_get("progress_rows_done")?,
        progress_rows_total: row.try_get("progress_rows_total")?,
        progress_phase: row.try_get("progress_phase")?,
        cancel_requested: row.get("cancel_requested")?,
        model_ref: row.try_get("model_ref")?,
        output_model_id: row.try_get("output_model_id")?,
        model_source: row.try_get("model_source")?,
        claimed_by: row.try_get("claimed_by")?,
        attempts: row.get::<i32>("attempts")? as u32,
        releases: row.get::<i32>("releases")? as u32,
        lease_expires_at: row.try_get("lease_expires_at")?,
        priority: row.get("priority")?,
        claimable: row.get("claimable")?,
        acceleration_report: row.try_get("acceleration_report")?,
        training_set_ref: row.try_get("training_set_ref")?,
        training_set_location: row.try_get("training_set_location")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
    })
}

/// Input parameters for [`Catalog::submit_job`]. Grouped into one struct (the
/// `RegisterModelParams`/`CreateTrainingJobParams` pattern) so the insert
/// surface has one place to grow.
#[derive(Debug, Clone)]
pub struct SubmitJobParams<'a> {
    pub job_id: &'a str,
    pub kind: &'a str,
    /// `Queued` — claimable by the poll loop. `Inline` — claimed once, by id,
    /// in the submitting call itself; every job is inserted `status =
    /// 'queued'` regardless (the row's very first claim, whichever path
    /// claims it, is what performs the `queued -> running` transition).
    pub execution: JobExecution,
    pub spec: &'a str,
    /// The PK-keyed base model a training kind fine-tunes from.
    pub model_ref: Option<&'a str>,
    /// The NAME-keyed model a training kind will register on finish.
    pub output_model_id: Option<&'a str>,
    /// The NAME-keyed, FK-free `ModelSource` string a compute verb resolves
    /// its model against.
    pub model_source: Option<&'a str>,
    /// Claim-ordering tie-break before `created_at`. `0` reproduces plain
    /// oldest-first FIFO — see migration 024.
    pub priority: i32,
}

/// Input parameters for [`Catalog::finish_job`].
#[derive(Debug, Clone)]
pub struct FinishJobParams<'a> {
    pub job_id: &'a str,
    /// The instance the lease-guarded CAS matches against `claimed_by`.
    pub instance_id: &'a str,
    /// The attempt the lease-guarded CAS matches against `attempts`.
    pub attempts: u32,
    /// The tagged JSON terminal payload.
    pub result: &'a str,
}

/// One retained epoch checkpoint's catalog row, inserted inside the same
/// attempt-guarded finish transaction that commits the output model's served
/// path ([`Catalog::finish_job_with_model`]) — ported from the removed
/// `training_repo::EpochCheckpointRow` (unit 348) onto the generalised `jobs`
/// schema.
///
/// Lifecycle: a row is inserted **only** when the finish CAS this call rides
/// on actually wins — the insert sits inside the same `if job_updated == 1`
/// guard as the output model's `artifact_path` write. A zombie or
/// lease-lost caller's `finish_job_with_model` therefore matches zero job
/// rows and never reaches this insert at all, so it can never register an
/// epoch-checkpoint row for an attempt that lost the race — mirroring the
/// output model's own "served path is written by exactly one writer"
/// contract.
#[derive(Debug, Clone)]
pub struct EpochCheckpointRow<'a> {
    /// Distinct catalog name: `jammi:fine-tuned:{job_id}:epoch_{N}` — never
    /// an additional VERSION of the output model's name (that would let a
    /// later finish's version-scoped CAS clobber a different row's path; see
    /// [`Catalog::finish_job_with_model`]'s version predicate).
    pub model_id: &'a str,
    /// Catalog `model_type`, mirroring the output model row's convention
    /// (e.g. `"fine-tuned"`).
    pub model_type: &'a str,
    /// Same task the output model row carries.
    pub task: crate::model_task::ModelTask,
    /// Same base-model lineage the output model row carries.
    pub base_model_id: Option<&'a str>,
    /// The attempt-unique object-store prefix these bytes were published
    /// under — the bytes are already complete (manifest-last) by the time
    /// this row is built, unlike the output model row, which registers with
    /// no path and gains it only from the CAS below.
    pub artifact_path: &'a str,
}

/// Input parameters for [`Catalog::finish_job_with_model`] — grouped (the
/// `RegisterModelParams`/`FinishJobParams` pattern) so the attempt-guarded
/// call site names each field, and so a future addition to the finish
/// surface grows this struct rather than clippy's `too_many_arguments` limit.
pub struct FinishJobWithModelParams<'a> {
    /// The job id to finish.
    pub job_id: &'a str,
    /// The instance the attempt-guarded CAS matches against `claimed_by`.
    pub instance_id: &'a str,
    /// The attempt the attempt-guarded CAS matches against `attempts`.
    pub attempts: u32,
    /// The tagged JSON terminal payload, exactly as [`FinishJobParams::result`].
    pub result: &'a str,
    /// The output model's catalog NAME (`jobs.output_model_id`, already
    /// written at [`Catalog::submit_job`] time for a training kind — see
    /// [`JobRecord::output_model_id`]).
    pub output_model_id: &'a str,
    /// The output model's catalog VERSION — together with `output_model_id`
    /// and the tenant, the exact row the served-path `UPDATE` must touch and
    /// no other (B5, unit 348).
    pub output_model_version: i32,
    /// The object-store prefix this caller published the output artifact
    /// under — committed as the model row's `artifact_path`.
    pub artifact_path: &'a str,
    /// Every RETAINED epoch checkpoint to register alongside the output
    /// model (unit 348) — empty for a training kind with no per-epoch
    /// checkpointing.
    pub epoch_checkpoints: &'a [EpochCheckpointRow<'a>],
}

/// A row from the `workers` table joined with its owning `instances` row —
/// `ListWorkers`' wire shape (N12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRecord {
    pub instance_id: String,
    pub label: Option<String>,
    pub host: Option<String>,
    pub kinds: String,
    /// The claim loop's lifecycle state as the row carries it — one of
    /// [`WorkerState`]'s spellings (`warming` / `claiming` / `draining`),
    /// written by the loop task in that order (`upsert_worker` with
    /// `warming` as its first statement and again at each transition —
    /// every lifecycle write is an upsert) so no reader ever sees
    /// `claiming` before the row exists.
    pub state: String,
    pub started_at: String,
    pub last_seen_at: String,
    /// This worker's device inventory, as `Catalog::upsert_worker` wrote
    /// it. A `ListWorkers` MIRROR only (mapped verbatim onto
    /// `jammi.v1.job.WorkerSummary.devices`, field 8, an additive field on
    /// the frozen RPC surface — `crates/jammi-wire/proto/jammi/v1/job.proto`)
    /// — never the placement policy's authority; see
    /// `super::compute_repo::ComputeExecutorRecord::devices`'s doc. A
    /// malformed stored value decodes to an empty list with a
    /// `tracing::warn!` naming `instance_id` (the #574 row-fact rule; see
    /// [`super::instance::decode_devices_json`]), never a read fault.
    pub devices: Vec<DeviceFact>,
}

fn parse_worker_row(
    row: &Row<'_>,
) -> std::result::Result<WorkerRecord, super::backend::BackendError> {
    let instance_id: String = row.get("instance_id")?;
    let devices_json: String = row.get("devices")?;
    let devices = decode_devices_json(&devices_json, &instance_id);
    Ok(WorkerRecord {
        label: row.try_get("label")?,
        host: row.try_get("host")?,
        kinds: row.get("kinds")?,
        state: row.get("state")?,
        started_at: row.get("started_at")?,
        last_seen_at: row.get("last_seen_at")?,
        instance_id,
        devices,
    })
}

/// One `instances JOIN workers` candidate row, before
/// [`Catalog::list_gang_members`]'s own Rust-side exclusion filters run —
/// a named struct rather than a five-element tuple (A5: `clippy::
/// type_complexity` is a signal to name the shape, not to suppress it).
struct GangCandidateRow {
    instance_id: String,
    peer_addr: String,
    kinds: String,
    state: String,
}

/// The `workers.state` vocabulary (migration 031, CHECK-constrained at the
/// SQL edge): what THIS process's claim loop is doing right now, as another
/// process can read it off the row (`ListWorkers`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// The loop task exists but is parked on its worker gate (the process is
    /// still preloading models): it has claimed nothing and will not until
    /// the gate opens.
    Warming,
    /// The claim loop is live.
    Claiming,
    /// A DRAIN is in progress: the loop finishes its in-flight job (if any)
    /// and stops claiming; the row is deleted once the loop has exited.
    Draining,
}

impl WorkerState {
    /// The exact string the `workers.state` column carries.
    pub fn as_db_str(self) -> &'static str {
        match self {
            WorkerState::Warming => "warming",
            WorkerState::Claiming => "claiming",
            WorkerState::Draining => "draining",
        }
    }
}

impl std::fmt::Display for WorkerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_db_str())
    }
}

/// The maximum byte length `submit_job_deduped`'s `idempotency_key` accepts
/// (migration 030: `jobs.idempotency_key TEXT` + the partial unique index
/// `idx_jobs_tenant_idempotency_key` — see [`Catalog::submit_job_deduped`]'s
/// doc). Bounded at the edge (the `SubmitJob` gRPC handler) AND asserted
/// again here (defence in depth): a live reproducer on Postgres was `index
/// row size 5136 exceeds btree version 4 maximum 2704` for an oversize key —
/// SQLite silently accepted the same 8 KiB key, so the two backends diverged
/// on the exact same input without this bound. Cited by `job.proto`'s
/// `idempotency_key` field doc and the deployment guide.
pub const MAX_IDEMPOTENCY_KEY_BYTES: usize = 256;

impl Catalog {
    /// Submit a new job, `status = 'queued'` — the row's own claim (by
    /// [`Self::claim_next`] for `execution = 'queued'`, or by
    /// [`Self::claim_by_id`] for `execution = 'inline'`) performs the
    /// `queued -> running` transition. Tenant bound + asserted (SPEC-03 §7).
    /// Never deduped — a thin `idempotency_key: None` call onto
    /// [`Self::submit_job_deduped`], kept as the plain entry point every
    /// pre-existing caller (every compute verb, every training kind's
    /// internal submit) uses unchanged.
    pub async fn submit_job(&self, p: SubmitJobParams<'_>) -> Result<()> {
        self.submit_job_deduped(p, None).await?;
        Ok(())
    }

    /// Submit a new job exactly like [`Self::submit_job`], additionally
    /// deduped by an optional per-tenant `idempotency_key` (migration 030:
    /// `jobs.idempotency_key` + the partial unique index
    /// `idx_jobs_tenant_idempotency_key` — see that migration's doc for why
    /// the index keys on `COALESCE(tenant_id, '')`).
    ///
    /// `idempotency_key` of `None` or `""` never dedupes: a plain
    /// unconditional insert, byte-identical to [`Self::submit_job`] (a `NULL`
    /// key row is excluded from the partial index entirely, so it can never
    /// participate in a conflict). `Some(key)` non-empty makes the insert a
    /// single atomic compare-and-set: `INSERT ... ON CONFLICT ... DO NOTHING`
    /// inside the SAME statement that creates the row, so two concurrent
    /// submissions racing the same `(tenant, key)` pair can never both land a
    /// row — the loser's insert affects zero rows, and this call re-reads the
    /// winner's `job_id` inside the SAME transaction (no separate
    /// lookup-then-insert race window a caller could interleave a THIRD write
    /// into). This is the only durable dedupe surface — there is no
    /// in-process map to resurrect a pruned row's identity or to forget it on
    /// a process restart.
    ///
    /// Returns the job id OF RECORD: `p.job_id` on a fresh insert (including
    /// every `None`-key call, which always inserts fresh), or the winning
    /// prior row's id when `Some(key)` collided — the caller must not assume
    /// the returned id equals `p.job_id`.
    pub async fn submit_job_deduped(
        &self,
        p: SubmitJobParams<'_>,
        idempotency_key: Option<&str>,
    ) -> Result<String> {
        let job_id = p.job_id.to_string();
        let kind = p.kind.to_string();
        let execution = p.execution.to_string();
        let spec = p.spec.to_string();
        let model_ref = p.model_ref.map(str::to_string);
        let output_model_id = p.output_model_id.map(str::to_string);
        let model_source = p.model_source.map(str::to_string);
        let priority = p.priority as i64;
        let tenant = self.current_tenant();
        let now = canonical_stamp_now();
        let key = idempotency_key
            .filter(|k| !k.is_empty())
            .map(str::to_string);
        // Defence in depth: the `SubmitJob` gRPC handler already refuses an
        // over-bound key at the edge with a typed `InvalidArgument`, but this
        // in-process entry point (every embedded caller, and any future wire
        // surface) must not rely solely on that -- see
        // [`MAX_IDEMPOTENCY_KEY_BYTES`]'s doc for the live Postgres/SQLite
        // divergence this closes.
        if let Some(k) = &key {
            if k.len() > MAX_IDEMPOTENCY_KEY_BYTES {
                return Err(JammiError::Config(format!(
                    "idempotency_key exceeds the maximum length \
                     (MAX_IDEMPOTENCY_KEY_BYTES = {MAX_IDEMPOTENCY_KEY_BYTES} bytes)"
                )));
            }
        }

        let recorded_job_id = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    tx.assert_tenant_matches(tenant, "jobs")?;
                    let sql = format!(
                        "INSERT INTO jobs \
                         (job_id, kind, tenant_id, status, execution, spec, model_ref, \
                          output_model_id, model_source, priority, acceleration_report, \
                          idempotency_key, created_at, updated_at) \
                         VALUES ($1, $2, $3, '{queued}', $4, $5, $6, $7, $8, $9, $10, $11, $12, $12) \
                         ON CONFLICT (COALESCE(tenant_id, ''), idempotency_key) \
                           WHERE idempotency_key IS NOT NULL \
                         DO NOTHING \
                         RETURNING job_id",
                        queued = JobStatus::Queued
                    );
                    // The insert is attempted at most twice. The re-read
                    // below (after `ON CONFLICT DO NOTHING` excludes our own
                    // row) CAN legitimately find nothing under
                    // `ReadCommitted`: a concurrent `PruneJobs` can delete
                    // the winning row between the INSERT's own conflict
                    // check and this re-read, freeing the (tenant, key) pair
                    // again for the instant in between -- that is not "never
                    // happens", it is a real, if rare, race this retry
                    // closes. A second miss in a row (the retry ALSO losing
                    // to a conflict whose winner is then ALSO gone by the
                    // time of ITS re-read) is the genuine, honestly-named
                    // failure mode below.
                    for _attempt in 0..2 {
                        let inserted = tx
                            .query_opt(
                                &sql,
                                &[
                                    SqlValue::TextOwned(job_id.clone()),
                                    SqlValue::TextOwned(kind.clone()),
                                    SqlValue::from(tenant.map(|t| t.to_string())),
                                    SqlValue::TextOwned(execution.clone()),
                                    SqlValue::TextOwned(spec.clone()),
                                    SqlValue::from(model_ref.clone()),
                                    SqlValue::from(output_model_id.clone()),
                                    SqlValue::from(model_source.clone()),
                                    SqlValue::Int(priority),
                                    SqlValue::Text(ACCELERATION_REPORT_PENDING),
                                    SqlValue::from(key.clone()),
                                    SqlValue::TextOwned(now.clone()),
                                ],
                                |row| row.get::<String>("job_id"),
                            )
                            .await?;
                        if let Some(id) = inserted {
                            return Ok(id);
                        }
                        // The insert's own row was excluded by the ON
                        // CONFLICT's DO NOTHING only when `key` is `Some` (a
                        // `NULL`-keyed row is outside the partial index and
                        // can never conflict) — so a `None` reaching this
                        // arm would be a backend bug, not a caller error.
                        let key_str = key.clone().expect(
                            "a None idempotency_key row is excluded from the partial index and \
                             can never hit ON CONFLICT DO NOTHING",
                        );
                        let (select_sql, params): (&str, Vec<SqlValue<'static>>) = match tenant {
                            Some(t) => (
                                "SELECT job_id FROM jobs \
                                 WHERE idempotency_key = $1 AND tenant_id = $2",
                                vec![
                                    SqlValue::TextOwned(key_str),
                                    SqlValue::TextOwned(t.to_string()),
                                ],
                            ),
                            None => (
                                "SELECT job_id FROM jobs \
                                 WHERE idempotency_key = $1 AND tenant_id IS NULL",
                                vec![SqlValue::TextOwned(key_str)],
                            ),
                        };
                        let existing = tx
                            .query_opt(select_sql, &params, |row| row.get::<String>("job_id"))
                            .await?;
                        if let Some(id) = existing {
                            return Ok(id);
                        }
                        // The first miss loops back and retries the insert
                        // once (the winner was deleted between the conflict
                        // check and this re-read, freeing the pair again); a
                        // second miss falls through to the honest failure
                        // below.
                    }
                    Err(BackendError::Execution(
                        "submit_job_deduped: ON CONFLICT DO NOTHING fired twice with no row \
                         found for the (tenant, idempotency_key) pair either time -- a \
                         concurrent PruneJobs deleted the winner between the conflict check and \
                         the re-read on BOTH the original attempt and its retry, under \
                         ReadCommitted; a genuine (rare) race, not a bug that never happens"
                            .to_string(),
                    ))
                })
            })
            .await?;
        Ok(recorded_job_id)
    }

    /// Get a job by id. Tenant-filtered; inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure the tenant
    /// predicate is dropped and the row resolves by its primary key alone.
    pub async fn get_job(&self, job_id: &str) -> Result<JobRecord> {
        let admin = TenantBinding::is_admin_scope();
        let sql = if admin {
            format!("SELECT {SELECT_COLS} FROM jobs WHERE job_id = $1")
        } else {
            format!(
                "SELECT {SELECT_COLS} FROM jobs WHERE job_id = $1 \
                   AND (tenant_id = $2 OR tenant_id IS NULL)"
            )
        };
        let id = job_id.to_string();
        let id_for_err = id.clone();
        let tenant = self.current_tenant();
        let found = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let mut params = vec![SqlValue::TextOwned(id)];
                        if !admin {
                            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                        }
                        tx.query_opt(&sql, &params, parse_row).await
                    })
                },
            )
            .await?;
        found.ok_or_else(|| JammiError::Catalog(format!("Job '{id_for_err}' not found")))
    }

    /// List jobs visible to the session tenant, most recent first. Inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure the tenant
    /// predicate is dropped and every tenant's jobs are returned.
    pub async fn list_jobs(&self) -> Result<Vec<JobRecord>> {
        let admin = TenantBinding::is_admin_scope();
        let sql = if admin {
            format!("SELECT {SELECT_COLS} FROM jobs ORDER BY created_at DESC")
        } else {
            format!(
                "SELECT {SELECT_COLS} FROM jobs \
                 WHERE tenant_id = $1 OR tenant_id IS NULL \
                 ORDER BY created_at DESC"
            )
        };
        let tenant = self.current_tenant();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let params: Vec<SqlValue<'static>> = if admin {
                            Vec::new()
                        } else {
                            vec![SqlValue::from(tenant.map(|t| t.to_string()))]
                        };
                        tx.query(&sql, &params, parse_row).await
                    })
                },
            )
            .await?)
    }

    /// Atomically claim the highest-priority `queued`-execution job of one of
    /// `kinds` for `instance_id`, leasing it for `lease`. Mirrors
    /// `training_repo::claim_next_training_job`'s ordering and per-backend
    /// concurrency (`FOR UPDATE SKIP LOCKED` on Postgres; SQLite's single
    /// serialised writer), generalised with a `kind IN (...)` filter and
    /// `execution = 'queued'` so an `inline` row — however it got stuck at
    /// `queued` — is never selected here. `Ok(None)` when `kinds` is empty or
    /// no matching job is claimable.
    pub async fn claim_next(
        &self,
        instance_id: &str,
        kinds: &[&str],
        lease: Duration,
    ) -> Result<Option<JobRecord>> {
        if kinds.is_empty() {
            return Ok(None);
        }
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let queued_execution = JobExecution::Queued.to_string();
        let instance_id = instance_id.to_string();
        #[cfg(feature = "test-hooks")]
        let instance_for_hook = instance_id.clone();
        let now = canonical_stamp_now();
        let kind = self.backend().backend_kind();

        let mut params: Vec<SqlValue<'static>> = vec![
            SqlValue::TextOwned(running),
            SqlValue::TextOwned(instance_id),
            SqlValue::TextOwned(queued.clone()),
        ];
        let mut kind_placeholders = Vec::with_capacity(kinds.len());
        for k in kinds {
            params.push(SqlValue::TextOwned((*k).to_string()));
            kind_placeholders.push(format!("${}", params.len()));
        }
        let kind_in = kind_placeholders.join(", ");
        params.push(SqlValue::TextOwned(queued_execution));
        let execution_bind = params.len();
        // The cooldown term (migration 037's `jobs.next_assembly_after`):
        // in the CANDIDATE subselect only, never the outer CAS `UPDATE`'s
        // own guard, so a job cooling down after a non-proceeding assembly
        // outcome is simply never a candidate a higher-priority row could
        // block a lower-priority ready one behind — the ORDER BY / LIMIT 1
        // never even considers it. `lease_expired_clause` is reused
        // VERBATIM (never a second clock helper, never a new stored
        // representation): "the deadline is absent or has passed" is
        // exactly the same predicate for a cooldown as for a lease, on the
        // SAME backend clock the lease columns use (`super::lease`'s
        // module docs) — no bind at all on Postgres, so a skewed calling
        // process's clock cannot affect that arm.
        let cooldown_clause = lease_expired_clause("next_assembly_after", kind, &mut params);

        let candidate = match kind {
            BackendKind::Postgres => format!(
                "(SELECT job_id FROM jobs WHERE status = $3 AND execution = ${execution_bind} \
                  AND claimable AND kind IN ({kind_in}) AND {cooldown_clause} \
                  ORDER BY priority DESC, created_at LIMIT 1 FOR UPDATE SKIP LOCKED)"
            ),
            BackendKind::Sqlite => format!(
                "(SELECT job_id FROM jobs WHERE status = $3 AND execution = ${execution_bind} \
                  AND claimable AND kind IN ({kind_in}) AND {cooldown_clause} \
                  ORDER BY priority DESC, created_at LIMIT 1)"
            ),
        };
        // The lease deadline is the DATABASE's own clock on Postgres (never
        // this process's — `catalog::lease`'s module docs).
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        let updated_at_bind = params.len();
        let sql = format!(
            "UPDATE jobs \
             SET status = $1, claimed_by = $2, lease_expires_at = {deadline_expr}, \
                 attempts = attempts + 1, updated_at = ${updated_at_bind} \
             WHERE job_id = {candidate} AND status = $3 \
             RETURNING {SELECT_COLS}"
        );

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let claimed = tx.query_opt(&sql, &params, parse_row).await?;
                    // Test-only: hold a landed claim open before COMMIT (the
                    // shutdown oracles' "claim in flight" window). A no-op
                    // unless armed for this instance; never parks an empty
                    // claim, so an idle loop's ticks cannot consume the arm.
                    #[cfg(feature = "test-hooks")]
                    if claimed.is_some() {
                        super::claim_test_hooks::maybe_park(
                            &instance_for_hook,
                            super::claim_test_hooks::ParkPoint::ClaimBeforeCommit,
                        )
                        .await;
                    }
                    Ok(claimed)
                })
            })
            .await
            .map_err(Into::into)
    }

    /// Atomically claim one `inline`-execution job by id for `instance_id`
    /// (`queued -> running`, CAS on `job_id AND status = 'queued' AND
    /// execution = 'inline'`), leasing it for `lease`. `Ok(None)` when the
    /// row is absent, not `inline`, or already claimed.
    pub async fn claim_by_id(
        &self,
        job_id: &str,
        instance_id: &str,
        lease: Duration,
    ) -> Result<Option<JobRecord>> {
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let inline = JobExecution::Inline.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let now = canonical_stamp_now();
        let kind = self.backend().backend_kind();

        let mut params: Vec<SqlValue<'static>> = vec![
            SqlValue::TextOwned(running),
            SqlValue::TextOwned(instance_id),
        ];
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        let updated_at_bind = params.len();
        params.push(SqlValue::TextOwned(job_id));
        let job_id_bind = params.len();
        params.push(SqlValue::TextOwned(queued));
        let queued_bind = params.len();
        params.push(SqlValue::TextOwned(inline));
        let inline_bind = params.len();
        let sql = format!(
            "UPDATE jobs \
             SET status = $1, claimed_by = $2, lease_expires_at = {deadline_expr}, \
                 attempts = attempts + 1, updated_at = ${updated_at_bind} \
             WHERE job_id = ${job_id_bind} AND status = ${queued_bind} \
               AND execution = ${inline_bind} \
             RETURNING {SELECT_COLS}"
        );

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.query_opt(&sql, &params, parse_row).await })
            })
            .await
            .map_err(Into::into)
    }

    /// Renew the lease on a running job the caller still owns. `false` when
    /// the guard misses: the lease was lost, the job is not running,
    /// `attempts` is stale (a successor already claimed this job), or the
    /// lease was RELEASED (`lease_expires_at IS NULL` — [`Self::release_job_lease`]
    /// / [`Self::release_jobs_claimed_by`]): the `IS NOT NULL` arm is what
    /// makes a release final at the SQL edge, so no holder — this process's
    /// own keeper included — can re-arm a lease its instance handed back.
    /// Every `running` row this catalog's own claim path produces carries a
    /// non-NULL lease ([`Self::claim_next`] / [`Self::claim_by_id`] both
    /// stamp the deadline), so the arm changes nothing for a live holder.
    pub async fn heartbeat_job(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        lease: Duration,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let now = canonical_stamp_now();
        let kind = self.backend().backend_kind();
        let mut params: Vec<SqlValue<'static>> = Vec::new();
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        params.push(SqlValue::TextOwned(job_id));
        params.push(SqlValue::TextOwned(running));
        params.push(SqlValue::TextOwned(instance_id));
        params.push(SqlValue::Int(attempts));
        let sql = format!(
            "UPDATE jobs SET lease_expires_at = {deadline_expr}, updated_at = $2 \
             WHERE job_id = $3 AND status = $4 AND claimed_by = $5 AND attempts = $6 \
               AND lease_expires_at IS NOT NULL"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(updated == 1)
    }

    /// The placed-gang hand-off (design contract `feat_500-wave4.md` §
    /// 2.4): move `job_id`'s claim from `from_instance` to `to_instance` —
    /// `claimed_by = $to`, a fresh `lease` deadline, `updated_at` — WITHOUT
    /// touching `attempts` or `releases` (zero net attempts: this is a
    /// hand-off, never a re-claim). `Ok(false)` when the guard misses:
    ///
    /// - `claimed_by != from_instance` — a stale runner (a superseded
    ///   attempt, or a SECOND launch of the same task, Ballista's own
    ///   reset-on-`ExecutorLost`) cannot transfer a claim it does not hold;
    ///   this is also the bind-time re-launch guard's second half (the
    ///   first half is the placement policy refusing to bind a task whose
    ///   row is already `claimed_by` an executor);
    /// - `attempts` does not match — a stale runner of an OLDER attempt
    ///   cannot transfer a claim a newer attempt already moved past;
    /// - `status != 'running'` — a terminal or queued row has no claim to
    ///   hand off;
    /// - the lease is not LIVE — [`lease_live_clause`], a POSITIVE
    ///   comparison (`lease_expires_at IS NOT NULL AND lease_expires_at >
    ///   now`), never [`lease_expired_clause`]'s `IS NULL OR …` shape: a
    ///   RELEASE ([`Self::release_job_lease`]) sets `lease_expires_at =
    ///   NULL`, and that NULL must make a transfer FAIL, the opposite of
    ///   what `lease_expired_clause`'s own `OR` would read a NULL as for a
    ///   RECLAIM sweep's purposes. See [`lease_live_clause`]'s doc for why
    ///   this is its own predicate, not `NOT lease_expired_clause(..)`.
    pub async fn transfer_claim(
        &self,
        job_id: &str,
        from_instance: &str,
        to_instance: &str,
        attempts: u32,
        lease: Duration,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let from_instance = from_instance.to_string();
        let to_instance = to_instance.to_string();
        let attempts = attempts as i64;
        let now = canonical_stamp_now();
        let kind = self.backend().backend_kind();

        let mut params: Vec<SqlValue<'static>> = Vec::new();
        params.push(SqlValue::TextOwned(to_instance));
        let to_bind = params.len();
        let deadline_expr = lease_deadline_expr(kind, lease, &mut params);
        params.push(SqlValue::TextOwned(now));
        let updated_at_bind = params.len();
        params.push(SqlValue::TextOwned(job_id));
        let job_id_bind = params.len();
        params.push(SqlValue::TextOwned(from_instance));
        let from_bind = params.len();
        params.push(SqlValue::Int(attempts));
        let attempts_bind = params.len();
        params.push(SqlValue::TextOwned(running));
        let status_bind = params.len();
        let live_clause = lease_live_clause("lease_expires_at", kind, &mut params);

        let sql = format!(
            "UPDATE jobs SET claimed_by = ${to_bind}, lease_expires_at = {deadline_expr}, \
                 updated_at = ${updated_at_bind} \
             WHERE job_id = ${job_id_bind} AND claimed_by = ${from_bind} \
               AND attempts = ${attempts_bind} AND status = ${status_bind} \
               AND {live_clause}"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Finish a job the caller still owns: `running -> completed`, recording
    /// the terminal JSON `result`. `false` when the attempt guard misses.
    ///
    /// This is a TERMINAL write, so the same attempt-guarded UPDATE also
    /// retires a still-`pending` `acceleration_report` to
    /// `{"state":"undetermined","reason":"finalized_without_determination"}`
    /// (esc-075) — see [`JobRecord::acceleration_report`]'s lifecycle
    /// section. A training kind that also needs to commit its output
    /// model's served path and epoch checkpoints atomically with this
    /// transition should call [`Catalog::finish_job_with_model`] instead,
    /// never this method followed by a second write.
    pub async fn finish_job(&self, p: FinishJobParams<'_>) -> Result<bool> {
        let completed = JobStatus::Completed.to_string();
        let running = JobStatus::Running.to_string();
        let job_id = p.job_id.to_string();
        let instance_id = p.instance_id.to_string();
        let attempts = p.attempts as i64;
        let result = p.result.to_string();
        let now = canonical_stamp_now();
        let retire = retire_pending_report_clause(3, 4);
        let sql = format!(
            "UPDATE jobs SET status = $1, result = $2, {retire}, lease_expires_at = NULL, \
             updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8 AND attempts = $9"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        &sql,
                        &[
                            SqlValue::TextOwned(completed),
                            SqlValue::TextOwned(result),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                            SqlValue::Text(ACCELERATION_REPORT_FINALIZED_WITHOUT_DETERMINATION),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Finish a job the caller still owns as a single attempt-guarded
    /// compare-and-set that ALSO commits the output model's served artifact
    /// path and registers every surviving epoch-checkpoint row (unit 348) —
    /// the atomic finish-with-model-registration peer of [`Self::finish_job`],
    /// ported from the removed `training_repo::finalize_training_job` onto
    /// the generalised `jobs` schema's full attempt guard (`job_id AND
    /// claimed_by AND status = 'running' AND attempts = $n`, where the old
    /// method carried only `job_id AND claimed_by AND status`).
    ///
    /// In one transaction it flips the job row to `completed` and writes
    /// `result` — and, only if that job-row CAS matched, records
    /// `artifact_path` on the output model's row and inserts one row per
    /// `epoch_checkpoints` entry. The job-row CAS lands **only** while the
    /// row is still `running`, `claimed_by == instance_id`, and `attempts ==
    /// attempts`. Returns `true` when the caller held the lease and is the
    /// sole finisher, `false` when it was not (the lease was lost, the row
    /// is not `running`, or `attempts` is stale — a successor already
    /// claimed this job). A `false` return means the caller must not act as
    /// the finisher; the job is left for [`Self::reclaim_expired_jobs`] and
    /// whichever instance re-claims it.
    ///
    /// The model row is matched by `name = output_model_id AND version =
    /// output_model_version`, tenant-scoped with the same STRICT predicate
    /// [`Self::delete_model`] uses (`tenant_id = $t OR (tenant_id IS NULL AND
    /// $t IS NULL)`) — never the relaxed `OR tenant_id IS NULL` a *read*
    /// resolver uses to also see a global row. The model-row update is gated
    /// on the job-row CAS matching (it runs in the same transaction and is
    /// skipped when the CAS matched zero rows), so the finish CAS is the
    /// **sole writer** of the served path: a loser's finish matches no job
    /// row and therefore writes neither the job status nor the model's
    /// served path.
    ///
    /// The `jobs` CAS itself stays NOT tenant-scoped, matching every other
    /// lease-identity write in this module: the lease identity (`claimed_by`,
    /// `attempts`) is the authority there, not the session tenant, and
    /// `job_id` is a global unique PK so no cross-tenant collision is
    /// possible on that row. The `models` predicate above is the one that
    /// needed tenant scoping, because `models.name` is NOT globally unique
    /// the way `job_id` is.
    ///
    /// This is a TERMINAL write, so the same CAS also retires a still-pending
    /// `acceleration_report` (esc-075) exactly as [`Self::finish_job`] does
    /// — see [`JobRecord::acceleration_report`]'s lifecycle section.
    ///
    /// An epoch-checkpoint whose catalog NAME is already occupied by another
    /// row is skipped (logged), never failing the whole job over one name
    /// collision: its bytes remain durable on the object store but
    /// permanently unregistered. The pre-check is by NAME ALONE (every
    /// version, the same strict tenant predicate the output model's UPDATE
    /// uses) — never a bare `ON CONFLICT(model_id) DO NOTHING`, which only
    /// catches an EXACT `(tenant, name, version=1)` PK collision and would
    /// silently let a same-name-different-version row SHADOW the checkpoint
    /// from every reader via `ORDER BY version DESC` resolution.
    pub async fn finish_job_with_model(&self, p: FinishJobWithModelParams<'_>) -> Result<bool> {
        let FinishJobWithModelParams {
            job_id,
            instance_id,
            attempts,
            result,
            output_model_id,
            output_model_version,
            artifact_path,
            epoch_checkpoints,
        } = p;
        let completed = JobStatus::Completed.to_string();
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let result = result.to_string();
        let output_model_id = output_model_id.to_string();
        let output_model_version = output_model_version as i64;
        let artifact_path = artifact_path.to_string();
        let now = canonical_stamp_now();
        let tenant = self.current_tenant();
        let retire = retire_pending_report_clause(3, 4);
        let job_sql = format!(
            "UPDATE jobs SET status = $1, result = $2, {retire}, lease_expires_at = NULL, \
             updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8 AND attempts = $9"
        );
        // Owned, 'static-lifetime copies of the epoch rows so the transaction
        // closure (which must be `'static`) can move them without borrowing
        // `epoch_checkpoints`.
        let epoch_rows: Vec<(String, String, String, Option<String>, String)> = epoch_checkpoints
            .iter()
            .map(|r| {
                (
                    r.model_id.to_string(),
                    r.model_type.to_string(),
                    r.task.as_db_str().to_string(),
                    r.base_model_id.map(str::to_string),
                    r.artifact_path.to_string(),
                )
            })
            .collect();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.set_tenant(tenant);
                    let job_updated = tx
                        .execute(
                            &job_sql,
                            &[
                                SqlValue::TextOwned(completed),
                                SqlValue::TextOwned(result),
                                SqlValue::Text(ACCELERATION_REPORT_PENDING),
                                SqlValue::Text(ACCELERATION_REPORT_FINALIZED_WITHOUT_DETERMINATION),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(job_id),
                                SqlValue::TextOwned(instance_id),
                                SqlValue::TextOwned(running),
                                SqlValue::Int(attempts),
                            ],
                        )
                        .await?;
                    // The gate every model-side write below hangs off: only
                    // the attempt that won the job-row CAS commits the
                    // served path and the epoch rows — pinned by
                    // `jobs_queue.rs::finish_job_with_model_is_an_attempt_guarded_compare_and_set`,
                    // the finish-side sibling of esc-107's `create_result_table`
                    // control.
                    if job_updated == 1 {
                        tx.assert_tenant_matches(tenant, "models")?;
                        let tenant_val = SqlValue::from(tenant.map(|t| t.to_string()));
                        tx.execute(
                            "UPDATE models SET artifact_path = $1, updated_at = $2 \
                             WHERE name = $3 AND version = $4 \
                               AND (tenant_id = $5 OR (tenant_id IS NULL AND $5 IS NULL))",
                            &[
                                SqlValue::TextOwned(artifact_path),
                                SqlValue::TextOwned(now.clone()),
                                SqlValue::TextOwned(output_model_id),
                                SqlValue::Int(output_model_version),
                                tenant_val.clone(),
                            ],
                        )
                        .await?;

                        for (model_id, model_type, task_db_str, base_model_id, path) in epoch_rows {
                            let occupied = tx
                                .query_opt(
                                    "SELECT COUNT(*) AS n FROM models \
                                     WHERE name = $1 \
                                       AND (tenant_id = $2 OR (tenant_id IS NULL AND $2 IS NULL))",
                                    &[SqlValue::TextOwned(model_id.clone()), tenant_val.clone()],
                                    |row| row.get::<i64>("n"),
                                )
                                .await?
                                .unwrap_or(0)
                                > 0;
                            if occupied {
                                tracing::warn!(
                                    occupied_name = %model_id,
                                    skipped_artifact_path = %path,
                                    "epoch-checkpoint catalog name already occupied by another \
                                     row; skipping registration — the checkpoint's bytes remain \
                                     durable but unregistered"
                                );
                                continue;
                            }

                            let pk = super::model_repo::model_pk(tenant, &model_id, 1);
                            let metadata = serde_json::json!({
                                "base_model_id": base_model_id,
                                "config_json": serde_json::Value::Null,
                            })
                            .to_string();
                            let now = canonical_stamp_now();
                            tx.execute(
                                "INSERT INTO models \
                                 (model_id, name, model_type, task, backend, version, \
                                  status, metadata, artifact_path, tenant_id, created_at, updated_at) \
                                 VALUES ($1, $2, $3, $4, 'candle', 1, 'checkpoint', $5, $6, $7, $8, $8)",
                                &[
                                    SqlValue::TextOwned(pk),
                                    SqlValue::TextOwned(model_id),
                                    SqlValue::TextOwned(model_type),
                                    SqlValue::TextOwned(task_db_str),
                                    SqlValue::TextOwned(metadata),
                                    SqlValue::TextOwned(path),
                                    tenant_val.clone(),
                                    SqlValue::TextOwned(now),
                                ],
                            )
                            .await?;
                        }
                    }
                    Ok(job_updated)
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Fail a job the caller still owns: `running -> failed`, recording
    /// `error`. `false` when the attempt guard misses.
    ///
    /// This is a TERMINAL write, so the same attempt-guarded UPDATE also
    /// retires a still-`pending` `acceleration_report` to
    /// `{"state":"undetermined","reason":"failed_before_probe"}` (esc-075) —
    /// see [`JobRecord::acceleration_report`]'s lifecycle section.
    pub async fn fail_job(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        error: &str,
    ) -> Result<bool> {
        let failed = JobStatus::Failed.to_string();
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let error = error.to_string();
        let now = canonical_stamp_now();
        let retire = retire_pending_report_clause(3, 4);
        let sql = format!(
            "UPDATE jobs SET status = $1, error = $2, {retire}, lease_expires_at = NULL, \
             updated_at = $5 \
             WHERE job_id = $6 AND claimed_by = $7 AND status = $8 AND attempts = $9"
        );
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        &sql,
                        &[
                            SqlValue::TextOwned(failed),
                            SqlValue::TextOwned(error),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                            SqlValue::Text(ACCELERATION_REPORT_FAILED_BEFORE_PROBE),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record a checkpoint's progress on a job the caller still owns. `false`
    /// when the attempt guard misses.
    pub async fn progress_job(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        rows_done: Option<i64>,
        rows_total: Option<i64>,
        phase: Option<&str>,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let phase = phase.map(str::to_string);
        let attempts = attempts as i64;
        let now = canonical_stamp_now();
        let sql = "UPDATE jobs SET progress_rows_done = $1, progress_rows_total = $2, \
                   progress_phase = $3, updated_at = $4 \
                   WHERE job_id = $5 AND claimed_by = $6 AND status = $7 AND attempts = $8";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::from(rows_done),
                            SqlValue::from(rows_total),
                            SqlValue::from(phase),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// RELEASE this instance's lease on one running, loop-claimed job: the
    /// heartbeat CAS (`job_id / status = 'running' / claimed_by / attempts`)
    /// plus `execution = 'queued'`, setting `lease_expires_at = NULL` and
    /// `releases = releases + 1`. The row stays `running` and `claimed_by`
    /// this instance — no status is invented for shutdown — but with a NULL
    /// lease it is immediately reclaimable ([`Self::reclaim_expired_jobs`]'s
    /// arm 1a treats `IS NULL` as expired), so a successor claims it within
    /// one idle poll instead of one lease window; the reclaim cap counts
    /// `attempts - releases`, so the release costs the job no attempt.
    ///
    /// `Ok(false)` when the guard misses: not this instance's live lease
    /// (a peer reclaimed it, it went terminal, `attempts` is stale), an
    /// INLINE row (`execution = 'inline'` — an inline job has no requeue
    /// arm and is never released; arm 2 fails it by instance staleness), or
    /// the lease was ALREADY released — the `lease_expires_at IS NOT NULL`
    /// arm makes every release statement idempotent, so `releases` is
    /// bumped at most once per attempt by construction.
    pub async fn release_job_lease(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let queued_execution = JobExecution::Queued.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let attempts = attempts as i64;
        let now = canonical_stamp_now();
        let sql = "UPDATE jobs SET lease_expires_at = NULL, releases = releases + 1, \
                   updated_at = $1 \
                   WHERE job_id = $2 AND status = $3 AND execution = $4 AND claimed_by = $5 \
                     AND attempts = $6 AND lease_expires_at IS NOT NULL";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(running),
                            SqlValue::TextOwned(queued_execution),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// The sweep form of [`Self::release_job_lease`]: RELEASE every running,
    /// loop-claimed (`execution = 'queued'`) job this instance still holds a
    /// live lease on — the same `SET`, `WHERE claimed_by = $me AND status =
    /// 'running' AND execution = 'queued' AND lease_expires_at IS NOT NULL`.
    /// Returns the number of rows released. Idempotent by the `IS NOT NULL`
    /// arm: a second sweep matches nothing; inline rows are never matched.
    /// A shutdown's RELEASE arm runs it twice — once before the loop is
    /// stopped (covering a claim that committed after the keeper's per-hold
    /// pass snapshotted) and once after (covering a claim that committed
    /// after the first sweep).
    pub async fn release_jobs_claimed_by(&self, instance_id: &str) -> Result<usize> {
        let running = JobStatus::Running.to_string();
        let queued_execution = JobExecution::Queued.to_string();
        let instance_id = instance_id.to_string();
        let now = canonical_stamp_now();
        let sql = "UPDATE jobs SET lease_expires_at = NULL, releases = releases + 1, \
                   updated_at = $1 \
                   WHERE claimed_by = $2 AND status = $3 AND execution = $4 \
                     AND lease_expires_at IS NOT NULL";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::TextOwned(queued_execution),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated as usize)
    }

    /// Clear `partial_result` on a job the caller still owns — the
    /// attempt-guarded inverse of [`Self::record_partial_result`] and of
    /// `create_result_table`'s in-transaction CAS: `UPDATE jobs SET
    /// partial_result = NULL WHERE job_id AND claimed_by AND status =
    /// 'running' AND attempts AND partial_result = $table`. `false` when the
    /// guard misses (a peer already cleared it, a successor superseded this
    /// attempt, or the column names a different table) — idempotent.
    ///
    /// Called by the compute path's partial-result dispatch on its
    /// claim-and-fail arm, BEFORE it materializes anew: `partial_result` is
    /// otherwise written exactly once (`create_result_table`'s CAS carries
    /// `partial_result IS NULL`) and never cleared, so every attempt >= 2
    /// that failed its predecessor's table and then created its own would
    /// find that CAS matching zero rows and land `JobAttemptSuperseded` —
    /// a terminal `failed` for a job whose successor did everything right
    /// (escape `esc-110`).
    pub async fn clear_partial_result(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        table_name: &str,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let table_name = table_name.to_string();
        let attempts = attempts as i64;
        let now = canonical_stamp_now();
        let sql = "UPDATE jobs SET partial_result = NULL, updated_at = $1 \
                   WHERE job_id = $2 AND claimed_by = $3 AND status = $4 AND attempts = $5 \
                     AND partial_result = $6";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                            SqlValue::TextOwned(table_name),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record `partial_result` on a job the caller still owns, outside a
    /// `create_result_table` transaction (e.g. adopting a predecessor's
    /// already-`ready` table on a reclaimed attempt — N1). `false` when the
    /// attempt guard misses. The narrower, `job_id`-only CAS
    /// [`Self::create_result_table`] performs at table-CREATION time is a
    /// separate, deliberately weaker write — see its doc.
    pub async fn record_partial_result(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        table_name: &str,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let table_name = table_name.to_string();
        let attempts = attempts as i64;
        let now = canonical_stamp_now();
        let sql = "UPDATE jobs SET partial_result = $1, updated_at = $2 \
                   WHERE job_id = $3 AND claimed_by = $4 AND status = $5 AND attempts = $6";
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        sql,
                        &[
                            SqlValue::TextOwned(table_name),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Record the claiming instance's acceleration determination (esc-075)
    /// for this attempt of a job the caller still owns — the report-writing
    /// peer of [`Self::progress_job`]. Replaces `acceleration_report`
    /// **only** while the row is still `running`, `claimed_by == instance_id`,
    /// **and `attempts == attempt`**.
    ///
    /// The `attempts` guard is mandatory, not a defensive extra:
    /// `claimed_by` can carry an instance id that is stable across process
    /// restarts, so `(job_id, claimed_by, status)` alone cannot tell "the
    /// current claimant, mid-run" from "a zombie of the same instance
    /// identity, from an attempt this job already moved past via reclaim". A
    /// reclaim always bumps `attempts` on re-claim ([`Self::claim_next`]'s
    /// and [`Self::claim_by_id`]'s `attempts = attempts + 1`), so pinning the
    /// exact attempt closes precisely that gap: a zombie presenting its own
    /// stale `attempt` value matches zero rows and can never overwrite the
    /// current claimant's report, even when it shares the current claimant's
    /// instance id and the row happens to be `running` again under a *later*
    /// attempt.
    ///
    /// Returns `true` when the write landed (the guard matched) and `false`
    /// when it did not — the lease was lost, the job is not running, or
    /// the caller's `attempt` is stale. Not tenant-scoped, matching the other
    /// lease-identity operations ([`Self::heartbeat_job`],
    /// [`Self::progress_job`], [`Self::finish_job`], [`Self::fail_job`]).
    pub async fn record_acceleration_report(
        &self,
        job_id: &str,
        instance_id: &str,
        attempts: u32,
        report_json: &str,
    ) -> Result<bool> {
        let running = JobStatus::Running.to_string();
        let job_id = job_id.to_string();
        let instance_id = instance_id.to_string();
        let report_json = report_json.to_string();
        let attempts = attempts as i64;
        let now = canonical_stamp_now();

        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE jobs SET acceleration_report = $1, updated_at = $2 \
                         WHERE job_id = $3 AND claimed_by = $4 AND status = $5 AND attempts = $6",
                        &[
                            SqlValue::TextOwned(report_json),
                            SqlValue::TextOwned(now),
                            SqlValue::TextOwned(job_id),
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(running),
                            SqlValue::Int(attempts),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Request cancellation of a non-terminal job. Sets `cancel_requested`;
    /// the executor observes it at a checkpoint boundary. Tenant-scoped with
    /// the STRICT predicate [`Self::delete_model`] uses; inside
    /// [`crate::session::JammiSession::with_admin_scope`] the predicate is
    /// dropped. `false` when the row is absent, out of scope, or already
    /// terminal (a terminal job cannot be cancelled retroactively).
    pub async fn cancel_request(&self, job_id: &str) -> Result<bool> {
        let admin = TenantBinding::is_admin_scope();
        let job_id_s = job_id.to_string();
        let tenant = self.current_tenant();
        let now = canonical_stamp_now();
        let non_terminal = JobStatus::non_terminal_sql_list();
        let sql = if admin {
            format!(
                "UPDATE jobs SET cancel_requested = TRUE, updated_at = $1 \
                 WHERE job_id = $2 AND status IN ({non_terminal})"
            )
        } else {
            format!(
                "UPDATE jobs SET cancel_requested = TRUE, updated_at = $1 \
                 WHERE job_id = $2 AND status IN ({non_terminal}) \
                   AND (tenant_id = $3 OR (tenant_id IS NULL AND $3 IS NULL))"
            )
        };
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let mut params = vec![SqlValue::TextOwned(now), SqlValue::TextOwned(job_id_s)];
                    if !admin {
                        params.push(SqlValue::from(tenant.map(|t| t.to_string())));
                    }
                    tx.execute(&sql, &params).await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Reclaim every expired job across the catalog. Two arms, both scoped to
    /// `status = 'running'`:
    ///
    ///   - **queued-execution, lease expired** (absent-or-expired
    ///     `lease_expires_at`, the same `lease_expired_clause` predicate
    ///     `training_repo::reclaim_expired_training_jobs` used): requeue when
    ///     `attempts - releases < max_attempts` (a lease the holder RELEASED
    ///     on purpose — [`Self::release_job_lease`] — is reclaimable at once
    ///     and costs the job no attempt), otherwise fail with
    ///     `"job lease expired after exhausting max attempts"`.
    ///   - **inline-execution, owning instance dead**: failed with
    ///     `"inline executor died"` when no `instances` row for `claimed_by`
    ///     has `last_seen_at` within `2 * lease` of now — covering BOTH an
    ///     absent instance row and a present-but-stale one with one `NOT
    ///     EXISTS (… live …)` predicate. Never requeued: an inline job has no
    ///     poll loop to re-claim it.
    ///
    /// `lease` is the deployment's lease WINDOW (the same value passed to
    /// [`Self::claim_next`]/[`Self::heartbeat_job`]) — used both to interpret
    /// `jobs.lease_expires_at` and, doubled, as the inline-liveness margin
    /// against `instances.last_seen_at`. Returns the number of jobs actioned
    /// across both arms. Not tenant-scoped — sweeps every tenant.
    ///
    /// Every arm also moves `acceleration_report`, in its OWN `UPDATE` (never
    /// a read-then-write): a requeue resets it to the pending marker
    /// (esc-075) — the row will re-probe under its next attempt — while both
    /// terminal arms (attempts-exhausted, inline-executor-dead) retire a
    /// still-pending marker with their own distinct reason. See
    /// [`JobRecord::acceleration_report`]'s lifecycle section.
    pub async fn reclaim_expired_jobs(&self, lease: Duration, max_attempts: u32) -> Result<usize> {
        let reclaimed = self
            .reclaim_expired_jobs_inner(lease, max_attempts, None)
            .await?;
        Ok(reclaimed.len())
    }

    /// The row-scoped peer of [`Self::reclaim_expired_jobs`]: given a job
    /// already read (by [`Self::get_job`]/[`Self::list_jobs`] or a wire
    /// `JobStatus`/`WaitJob` call), reclaim it in place if — and only if — it
    /// currently qualifies under either arm. `Ok(None)` when the row does not
    /// qualify (including: it is not `running` by the time this runs,
    /// or `record.job_id` does not exist) — the steady-state case, which
    /// issues the SAME `UPDATE … WHERE …` predicate as the bulk sweep scoped
    /// by `job_id`, so a non-qualifying row costs one no-op statement, never
    /// an unconditional write.
    pub async fn reclaim_job_on_read(
        &self,
        record: &JobRecord,
        lease: Duration,
        max_attempts: u32,
    ) -> Result<Option<JobRecord>> {
        if record.status != JobStatus::Running.to_string() {
            return Ok(None);
        }
        let mut reclaimed = self
            .reclaim_expired_jobs_inner(lease, max_attempts, Some(&record.job_id))
            .await?;
        Ok(reclaimed.pop())
    }

    async fn reclaim_expired_jobs_inner(
        &self,
        lease: Duration,
        max_attempts: u32,
        only_job_id: Option<&str>,
    ) -> Result<Vec<JobRecord>> {
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let failed = JobStatus::Failed.to_string();
        let queued_execution = JobExecution::Queued.to_string();
        let inline_execution = JobExecution::Inline.to_string();
        let max_attempts_i = max_attempts as i64;
        let now = canonical_stamp_now();
        let kind = self.backend().backend_kind();
        let scope_job = only_job_id.map(str::to_string);
        let exhausted_error = "job lease expired after exhausting max attempts".to_string();
        let inline_error = "inline executor died".to_string();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let mut out = Vec::new();

                    // Arm 1a: queued-execution, lease expired, attempts left -> requeue.
                    // Unconditionally RESETS acceleration_report to the pending
                    // marker (esc-075): the row returns to `queued` for a NEW
                    // attempt that will re-probe, so "no claimant has computed a
                    // determination yet" is once again exactly true, even when
                    // the dead attempt had already recorded a `determined`
                    // payload that describes hardware/config THAT attempt saw
                    // and must not be attributed to the next, not-yet-started
                    // attempt. See `JobRecord::acceleration_report`'s lifecycle
                    // section.
                    {
                        let mut params: Vec<SqlValue<'static>> = vec![
                            SqlValue::TextOwned(queued.clone()),
                            SqlValue::TextOwned(now.clone()),
                            SqlValue::Text(ACCELERATION_REPORT_PENDING),
                        ];
                        let pending_bind = params.len();
                        let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                        params.push(SqlValue::TextOwned(running.clone()));
                        let running_bind = params.len();
                        params.push(SqlValue::TextOwned(queued_execution.clone()));
                        let execution_bind = params.len();
                        params.push(SqlValue::Int(max_attempts_i));
                        let max_bind = params.len();
                        let mut scope_sql = String::new();
                        if let Some(job_id) = &scope_job {
                            params.push(SqlValue::TextOwned(job_id.clone()));
                            scope_sql = format!(" AND job_id = ${}", params.len());
                        }
                        let sql = format!(
                            "UPDATE jobs SET status = $1, claimed_by = NULL, \
                                 lease_expires_at = NULL, acceleration_report = ${pending_bind}, \
                                 updated_at = $2 \
                             WHERE status = ${running_bind} AND execution = ${execution_bind} \
                               AND {expired} AND attempts - releases < ${max_bind}{scope_sql} \
                             RETURNING {SELECT_COLS}"
                        );
                        out.extend(tx.query(&sql, &params, parse_row).await?);
                    }

                    // Arm 1b: queued-execution, lease expired, attempts exhausted -> fail.
                    // TERMINAL, so it retires a still-pending acceleration_report
                    // (esc-075) — see `JobRecord::acceleration_report`'s lifecycle
                    // section.
                    {
                        let mut params: Vec<SqlValue<'static>> = vec![
                            SqlValue::TextOwned(failed.clone()),
                            SqlValue::TextOwned(exhausted_error.clone()),
                        ];
                        params.push(SqlValue::TextOwned(now.clone()));
                        let now_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_PENDING));
                        let pending_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_LEASE_EXPIRED_EXHAUSTED));
                        let terminal_bind = params.len();
                        let retire = retire_pending_report_clause(pending_bind, terminal_bind);
                        let expired = lease_expired_clause("lease_expires_at", kind, &mut params);
                        params.push(SqlValue::TextOwned(running.clone()));
                        let running_bind = params.len();
                        params.push(SqlValue::TextOwned(queued_execution.clone()));
                        let execution_bind = params.len();
                        params.push(SqlValue::Int(max_attempts_i));
                        let max_bind = params.len();
                        let mut scope_sql = String::new();
                        if let Some(job_id) = &scope_job {
                            params.push(SqlValue::TextOwned(job_id.clone()));
                            scope_sql = format!(" AND job_id = ${}", params.len());
                        }
                        let sql = format!(
                            "UPDATE jobs SET status = $1, error = $2, \
                                 lease_expires_at = NULL, {retire}, updated_at = ${now_bind} \
                             WHERE status = ${running_bind} AND execution = ${execution_bind} \
                               AND {expired} AND attempts - releases >= ${max_bind}{scope_sql} \
                             RETURNING {SELECT_COLS}"
                        );
                        out.extend(tx.query(&sql, &params, parse_row).await?);
                    }

                    // Arm 2: inline-execution, owning instance absent or stale -> fail.
                    // TERMINAL (an inline job has no requeue arm), so it retires
                    // a still-pending acceleration_report (esc-075) — see
                    // `JobRecord::acceleration_report`'s lifecycle section.
                    {
                        let margin = instance_liveness_margin(lease);
                        let mut params: Vec<SqlValue<'static>> = vec![
                            SqlValue::TextOwned(failed.clone()),
                            SqlValue::TextOwned(inline_error.clone()),
                        ];
                        params.push(SqlValue::TextOwned(now.clone()));
                        let now_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_PENDING));
                        let pending_bind = params.len();
                        params.push(SqlValue::Text(ACCELERATION_REPORT_INLINE_EXECUTOR_DIED));
                        let terminal_bind = params.len();
                        let retire = retire_pending_report_clause(pending_bind, terminal_bind);
                        let stale =
                            stale_before_clause("i.last_seen_at", kind, margin, &mut params);
                        params.push(SqlValue::TextOwned(running.clone()));
                        let running_bind = params.len();
                        params.push(SqlValue::TextOwned(inline_execution.clone()));
                        let execution_bind = params.len();
                        let mut scope_sql = String::new();
                        if let Some(job_id) = &scope_job {
                            params.push(SqlValue::TextOwned(job_id.clone()));
                            scope_sql = format!(" AND job_id = ${}", params.len());
                        }
                        let sql = format!(
                            "UPDATE jobs SET status = $1, error = $2, \
                                 lease_expires_at = NULL, {retire}, updated_at = ${now_bind} \
                             WHERE status = ${running_bind} AND execution = ${execution_bind}\
                                 {scope_sql} \
                               AND NOT EXISTS ( \
                                 SELECT 1 FROM instances i \
                                 WHERE i.instance_id = jobs.claimed_by AND NOT ({stale}) \
                               ) \
                             RETURNING {SELECT_COLS}"
                        );
                        out.extend(tx.query(&sql, &params, parse_row).await?);
                    }

                    Ok(out)
                })
            })
            .await
            .map_err(Into::into)
    }

    /// The write-once CAS for the training-set
    /// identity pair. ONE statement sets BOTH `training_set_ref` and
    /// `training_set_location`, guarded on `job_id`, the CALLER'S OWN
    /// `claimed_by`/`attempts`, and the pair still being unset —
    /// `training_set_ref IS NULL AND training_set_location IS NULL`. A
    /// concurrent second coordinator's CAS (the same attempt, racing this
    /// one) sees zero rows updated and re-reads: if the pair landed with
    /// the SAME values under the SAME claim, this is [`TrainingSetFillOutcome::Reused`]
    /// (the loser observes its own would-be write already there, never a
    /// second write); if the claim moved (a different `claimed_by`/
    /// `attempts`) or the pair holds different values, this is
    /// [`TrainingSetFillOutcome::Aborted`] — a normal event, not a failure,
    /// and NO terminal write follows from it ("no supersession").
    ///
    /// The pair is one fact, not two independently nullable columns
    /// (migration 033's `CHECK` constraint pins this at the schema edge);
    /// this is the ONLY method in this crate that writes either column.
    pub async fn fill_training_set_identity(
        &self,
        job_id: &str,
        claimed_by: &str,
        attempts: u32,
        training_set_ref: &str,
        training_set_location: &str,
    ) -> Result<TrainingSetFillOutcome> {
        let job_id = job_id.to_string();
        let claimed_by = claimed_by.to_string();
        let training_set_ref = training_set_ref.to_string();
        let training_set_location = training_set_location.to_string();
        let attempts_i = attempts as i64;
        let now = canonical_stamp_now();

        Ok(self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let updated = tx
                        .execute(
                            "UPDATE jobs SET training_set_ref = $1, training_set_location = $2, \
                                 updated_at = $3 \
                             WHERE job_id = $4 AND claimed_by = $5 AND attempts = $6 \
                               AND training_set_ref IS NULL AND training_set_location IS NULL",
                            &[
                                SqlValue::TextOwned(training_set_ref.clone()),
                                SqlValue::TextOwned(training_set_location.clone()),
                                SqlValue::TextOwned(now),
                                SqlValue::TextOwned(job_id.clone()),
                                SqlValue::TextOwned(claimed_by.clone()),
                                SqlValue::Int(attempts_i),
                            ],
                        )
                        .await?;
                    if updated == 1 {
                        return Ok(TrainingSetFillOutcome::Filled);
                    }

                    // Zero rows: re-read by primary key alone (this is an
                    // internal consistency check on a job id the caller
                    // already holds, not a tenant-scoped external read) and
                    // decide REUSE vs ABORT.
                    let row = tx
                        .query_opt(
                            "SELECT claimed_by, attempts, training_set_ref, training_set_location \
                             FROM jobs WHERE job_id = $1",
                            &[SqlValue::TextOwned(job_id)],
                            |row| {
                                Ok((
                                    row.try_get::<String>("claimed_by")?,
                                    row.get::<i32>("attempts")? as u32,
                                    row.try_get::<String>("training_set_ref")?,
                                    row.try_get::<String>("training_set_location")?,
                                ))
                            },
                        )
                        .await?;
                    Ok(match row {
                        Some((cb, att, Some(tsref), Some(tsloc)))
                            if cb.as_deref() == Some(claimed_by.as_str())
                                && att == attempts
                                && tsref == training_set_ref
                                && tsloc == training_set_location =>
                        {
                            TrainingSetFillOutcome::Reused
                        }
                        _ => TrainingSetFillOutcome::Aborted,
                    })
                })
            })
            .await?)
    }

    /// The coordinator's call site into [`Self::fill_training_set_identity`]
    /// — DESIGN.md § 4's own vocabulary ("the coordinator materializes or
    /// reuses the training set"), returned as [`TrainingSetAssembly`] rather
    /// than the lower-level [`TrainingSetFillOutcome`] so the coordinator's
    /// own call site never has to re-derive "moved" from "aborted".
    ///
    /// This crate ships the verb and its own oracle here; the production
    /// caller is the coordinator body (`jammi-ai`'s `fine_tune/worker.rs`,
    /// U5b-1b-i / U5a-2) — DEFERRED, not built in this crate.
    pub async fn materialize_or_reuse_training_set(
        &self,
        job_id: &str,
        claimed_by: &str,
        attempts: u32,
        training_set_ref: &str,
        training_set_location: &str,
    ) -> Result<TrainingSetAssembly> {
        Ok(
            match self
                .fill_training_set_identity(
                    job_id,
                    claimed_by,
                    attempts,
                    training_set_ref,
                    training_set_location,
                )
                .await?
            {
                TrainingSetFillOutcome::Filled => TrainingSetAssembly::Won,
                TrainingSetFillOutcome::Reused => TrainingSetAssembly::Reused,
                TrainingSetFillOutcome::Aborted => TrainingSetAssembly::Moved,
            },
        )
    }

    /// Apply `AssemblyOutcome::effect`'s rule to `job_id`'s attempt
    /// `attempt`, in ONE `UPDATE`: `assembly_failures` bumps by one only for
    /// `AssemblyEffect::CooldownAndCounted` reasons, `next_assembly_after`
    /// is stamped `now + backoff(k)` (`k` the NEW, post-bump count for a
    /// counted reason, the UNCHANGED current count for a cooldown-only
    /// reason) for every reason that carries a cooldown at all, and a
    /// [`AssemblyOutcome::Success`] resets both columns — always on the
    /// BACKEND's own clock (`super::lease::lease_deadline_expr`, never a
    /// bound application timestamp on Postgres).
    ///
    /// A guard read (`SELECT … WHERE job_id = $1 AND attempts = $2`,
    /// row-locked on Postgres via `FOR UPDATE` — SQLite is already
    /// serialized for the whole transaction under `BEGIN IMMEDIATE`) learns
    /// the CURRENT `assembly_failures` count before the one `UPDATE` writes
    /// it (never a second, independent write): a moved claim — `attempt` no
    /// longer names the row's current attempt, or the job is gone — is
    /// caught here and the call returns `Ok(false)` with NO write at all,
    /// the exact "moved claim aborts with no write" shape
    /// [`Self::fill_training_set_identity`]'s own CAS uses.
    ///
    /// Returns `true` when the guarded row was found and updated, `false`
    /// when the attempt had already moved.
    pub async fn record_assembly_outcome(
        &self,
        job_id: &str,
        attempt: u32,
        outcome: AssemblyOutcome,
    ) -> Result<bool> {
        let job_id = job_id.to_string();
        let attempt_i = attempt as i64;
        let now = canonical_stamp_now();
        let kind = self.backend().backend_kind();
        let effect = outcome.effect();

        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    let select_sql = match kind {
                        BackendKind::Postgres => {
                            "SELECT assembly_failures FROM jobs \
                             WHERE job_id = $1 AND attempts = $2 FOR UPDATE"
                        }
                        BackendKind::Sqlite => {
                            "SELECT assembly_failures FROM jobs \
                             WHERE job_id = $1 AND attempts = $2"
                        }
                    };
                    let current: Option<i32> = tx
                        .query_opt(
                            select_sql,
                            &[
                                SqlValue::TextOwned(job_id.clone()),
                                SqlValue::Int(attempt_i),
                            ],
                            |row| row.get::<i32>("assembly_failures"),
                        )
                        .await?;
                    let Some(current) = current else {
                        // The attempt already moved (or the job never
                        // existed): no write, matching the CAS's own
                        // "moved claim aborts with no write" shape.
                        return Ok(false);
                    };

                    let new_failures: i32 = match effect {
                        AssemblyEffect::CooldownAndCounted => current.saturating_add(1),
                        AssemblyEffect::CooldownOnly | AssemblyEffect::Neither => current,
                        AssemblyEffect::Reset => 0,
                    };

                    let mut params: Vec<SqlValue<'static>> =
                        vec![SqlValue::Int(i64::from(new_failures))];
                    let failures_bind = params.len();
                    let cooldown_expr = match effect {
                        AssemblyEffect::Neither | AssemblyEffect::Reset => "NULL".to_string(),
                        AssemblyEffect::CooldownOnly | AssemblyEffect::CooldownAndCounted => {
                            let backoff =
                                assembly_backoff(u32::try_from(new_failures).unwrap_or(u32::MAX));
                            lease_deadline_expr(kind, backoff, &mut params)
                        }
                    };
                    params.push(SqlValue::TextOwned(now));
                    let updated_at_bind = params.len();
                    params.push(SqlValue::TextOwned(job_id));
                    let job_id_bind = params.len();
                    params.push(SqlValue::Int(attempt_i));
                    let attempt_bind = params.len();
                    let sql = format!(
                        "UPDATE jobs SET assembly_failures = ${failures_bind}, \
                             next_assembly_after = {cooldown_expr}, \
                             updated_at = ${updated_at_bind} \
                         WHERE job_id = ${job_id_bind} AND attempts = ${attempt_bind}"
                    );
                    let n = tx.execute(&sql, &params).await?;
                    Ok(n == 1)
                })
            })
            .await
            .map_err(Into::into)
    }

    /// The row `GangService::run_rank`'s I-GANG row predicate reads
    /// (`docs/rigor/contracts/feat_500-C-U5a-1.md` § A1; `GangService` is a
    /// `jammi-server` type this crate has no visibility into — named here
    /// only in prose, never as an intra-doc link).
    /// Primary-key only (`WHERE job_id = $1`) — no tenant
    /// predicate, never [`TenantBinding::is_admin_scope`] (this method does
    /// not consult it at all). The row's OWN `tenant_id` comes back as raw
    /// text ([`RankAdmissionRow::tenant_id`]) for the CALLER to derive and
    /// pin — "tenant is derived, never accepted"
    /// (`docs/rigor/contracts/feat_500-C-U5a-1.md` §2 (P3)) — together with
    /// the training-set identity pair the `world_size > 1` conjunct
    /// resolves under that tenant.
    /// ONE statement: the row's `status`/`claimed_by`/`attempts`/`spec`/
    /// `tenant_id`/training-set pair, alongside `lease_expires_at`'s RAW
    /// stored text — no SQL-side `col::timestamptz` cast or `EXTRACT(...)`
    /// here, unlike the CLAIM/RECLAIM path's
    /// [`super::lease::lease_expired_clause`] /
    /// [`super::lease::lease_remaining_seconds_expr`]. The text is decoded in
    /// RUST, after the read, via [`super::lease::decode_lease_expires_at`]
    /// into [`RankAdmissionRow::lease`] — never a second round trip. `Ok(None)`
    /// when no such job exists. `Err` means the READ ITSELF faulted — never
    /// that a row's content failed to decode: neither `spec` failing to
    /// decode a `world_size` ([`RankAdmissionRow::world_size`] as
    /// [`WorldSizeFact::Undecodable`]) nor (since
    /// <https://github.com/f-inverse/jammi-ai/issues/574>) `lease_expires_at`
    /// failing to parse ([`RankAdmissionRow::lease`] as
    /// [`super::lease::LeaseFact::Undecodable`]) is ever conflated with the
    /// read faulting — both are returned `Ok(Some(row))` with every other
    /// column populated (see each type's docs).
    ///
    /// This method decides NOTHING beyond that lookup — it is a plain
    /// row-by-primary-key read, never itself the I-GANG decider. Every
    /// determinant the returned [`RankAdmissionRow`] feeds (`status`,
    /// `claimed_by`, `attempts`, [`RankAdmissionRow::lease`], and — critically
    /// — [`RankAdmissionRow::world_size`], the ROW's own rank count, never a
    /// caller-supplied `Assign.world`) is decided by the caller
    /// (`GangService::run_rank`, the gang admission handler). A gate keyed
    /// on the caller's own claim about `world` rather than this field's
    /// value lets a mismatched claim skip the `world_size > 1` conjuncts
    /// entirely — the row is the only authority; the caller is never trusted
    /// to state its own admission class.
    ///
    /// **Enumerating-caller oracle** (`crates/jammi-server/tests/it/
    /// gang_rank_admission_oracle.rs`): the only call site outside this
    /// crate's own tests is the gang `RunRank` handler.
    pub async fn get_job_for_rank(&self, job_id: &str) -> Result<Option<RankAdmissionRow>> {
        let job_id = job_id.to_string();
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let params: Vec<SqlValue<'static>> = vec![SqlValue::TextOwned(job_id)];
                        let sql = "SELECT status, claimed_by, attempts, spec, tenant_id, \
                                 training_set_ref, training_set_location, lease_expires_at \
                             FROM jobs WHERE job_id = $1";
                        tx.query_opt(sql, &params, |row| {
                            let spec: String = row.get("spec")?;
                            let world_size = world_size_from_spec_json(&spec);
                            let lease_expires_at: Option<String> =
                                row.try_get("lease_expires_at")?;
                            let lease = super::lease::decode_lease_expires_at(
                                lease_expires_at.as_deref(),
                                super::lease::app_clock_now(),
                            );
                            Ok(RankAdmissionRow {
                                status: row.get("status")?,
                                claimed_by: row.try_get("claimed_by")?,
                                attempts: row.get::<i32>("attempts")? as u32,
                                tenant_id: row.try_get("tenant_id")?,
                                training_set_ref: row.try_get("training_set_ref")?,
                                training_set_location: row.try_get("training_set_location")?,
                                spec,
                                lease,
                                world_size,
                            })
                        })
                        .await
                    })
                },
            )
            .await?)
    }

    /// Upsert this process's `instances` row from `reg`: insert on first
    /// call (`started_at` stamped once), refresh `label`/`host`/`peer_addr`/
    /// `result_root`/`last_seen_at` on every later call. `reg.peer_addr` /
    /// `reg.member_root` write `NULL` for a process that never joins a
    /// gang — `reg` (produced by
    /// [`InstanceRegistration::from_config`](super::instance::InstanceRegistration::from_config))
    /// is the ONLY shape this and [`Self::reregister_instance`] accept.
    pub async fn upsert_instance(&self, reg: &InstanceRegistration) -> Result<()> {
        let instance_id = reg.instance_id.clone();
        let label = reg.label.clone();
        let host = reg.host.clone();
        let peer_addr = reg.peer_addr.as_ref().map(|p| p.as_str().to_string());
        let result_root = reg.member_root.as_ref().map(|c| c.as_str().to_string());
        let result_root_identity = reg
            .member_root
            .as_ref()
            .map(|c| c.identity().as_str().to_string());
        let now = canonical_stamp_now();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO instances \
                             (instance_id, label, host, peer_addr, result_root, \
                              result_root_identity, started_at, last_seen_at) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $7) \
                         ON CONFLICT(instance_id) DO UPDATE SET \
                             label = excluded.label, host = excluded.host, \
                             peer_addr = excluded.peer_addr, result_root = excluded.result_root, \
                             result_root_identity = excluded.result_root_identity, \
                             last_seen_at = excluded.last_seen_at",
                        &[
                            SqlValue::TextOwned(instance_id),
                            SqlValue::from(label),
                            SqlValue::from(host),
                            SqlValue::from(peer_addr),
                            SqlValue::from(result_root),
                            SqlValue::from(result_root_identity),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// The lease keeper's re-upsert on a MISSED touch (`Catalog::
    /// touch_instance` returning `Ok(false)`): re-writes the WHOLE
    /// membership tuple — the `instances` row from `reg` exactly like
    /// [`Self::upsert_instance`], AND, when `reg`'s worker half
    /// ([`InstanceRegistration::worker_snapshot`]) is `Some`, the `workers`
    /// row too — in ONE transaction, so a process whose row was pruned
    /// during a transient outage rejoins its gang (and its claim-loop
    /// membership, if any) atomically on its next successful heartbeat, with
    /// no restart. `touch_instance` itself stays a pure `UPDATE` that can
    /// never resurrect a pruned row; this is the ONLY verb that can.
    pub async fn reregister_instance(&self, reg: &InstanceRegistration) -> Result<()> {
        let instance_id = reg.instance_id.clone();
        let label = reg.label.clone();
        let host = reg.host.clone();
        let peer_addr = reg.peer_addr.as_ref().map(|p| p.as_str().to_string());
        let result_root = reg.member_root.as_ref().map(|c| c.as_str().to_string());
        let result_root_identity = reg
            .member_root
            .as_ref()
            .map(|c| c.identity().as_str().to_string());
        // `devices` is JSON-encoded up front (outside the transaction, the
        // same shape `upsert_worker` uses) so a re-upsert here writes the
        // snapshot's device inventory too, never leaving `workers.devices`
        // behind while the rest of the row rejoins.
        let worker = reg
            .worker_snapshot()
            .map(|w| {
                let devices_json = serde_json::to_string(&w.devices)
                    .map_err(|e| JammiError::Catalog(format!("devices encode: {e}")))?;
                Ok::<_, JammiError>((w, devices_json))
            })
            .transpose()?;
        let now = canonical_stamp_now();
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO instances \
                             (instance_id, label, host, peer_addr, result_root, \
                              result_root_identity, started_at, last_seen_at) \
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $7) \
                         ON CONFLICT(instance_id) DO UPDATE SET \
                             label = excluded.label, host = excluded.host, \
                             peer_addr = excluded.peer_addr, result_root = excluded.result_root, \
                             result_root_identity = excluded.result_root_identity, \
                             last_seen_at = excluded.last_seen_at",
                        &[
                            SqlValue::TextOwned(instance_id.clone()),
                            SqlValue::from(label),
                            SqlValue::from(host),
                            SqlValue::from(peer_addr),
                            SqlValue::from(result_root),
                            SqlValue::from(result_root_identity),
                            SqlValue::TextOwned(now),
                        ],
                    )
                    .await?;
                    if let Some((w, devices_json)) = worker {
                        tx.execute(
                            "INSERT INTO workers (instance_id, kinds, state, devices) \
                             VALUES ($1, $2, $3, $4) \
                             ON CONFLICT(instance_id) DO UPDATE \
                             SET kinds = excluded.kinds, state = excluded.state, \
                                 devices = excluded.devices",
                            &[
                                SqlValue::TextOwned(instance_id),
                                SqlValue::TextOwned(w.kinds),
                                SqlValue::Text(w.state.as_db_str()),
                                SqlValue::TextOwned(devices_json),
                            ],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(())
    }

    /// Heartbeat this process's `instances` row. `false` when the row is
    /// absent (recovery pruned it before this beat landed).
    pub async fn touch_instance(&self, instance_id: &str) -> Result<bool> {
        let instance_id = instance_id.to_string();
        let now = canonical_stamp_now();
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE instances SET last_seen_at = $1 WHERE instance_id = $2",
                        &[SqlValue::TextOwned(now), SqlValue::TextOwned(instance_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// Is `instance_id`'s `instances` row FRESH —
    /// present, and last seen within [`super::lease::instance_liveness_margin`]
    /// (`2 * lease`) on the DB clock? Primary-key lookup; `instances` carries
    /// no tenant column, so there is no tenant predicate to drop or keep.
    /// `false` for an absent, a malformed, OR a stale row alike — the caller
    /// (a gang rank checking its coordinator) maps every one to the same
    /// member-scoped `FailedPrecondition`, disclosing nothing about which.
    /// `last_seen_at`'s raw stored text is decoded in RUST
    /// ([`super::lease::last_seen_at_is_fresh`]) — never a SQL-side
    /// `col::timestamptz` cast — so a malformed value never faults the read
    /// (the resolution of
    /// <https://github.com/f-inverse/jammi-ai/issues/574> for this column);
    /// this column is ALWAYS an application-clock stamp on either backend
    /// (`Catalog::upsert_instance` / `Catalog::reregister_instance` /
    /// `Catalog::touch_instance` never write the database clock here), so
    /// the decode needs no [`BackendKind`] at
    /// all.
    pub async fn fresh_instance(&self, instance_id: &str, lease: Duration) -> Result<bool> {
        let instance_id = instance_id.to_string();
        let margin = instance_liveness_margin(lease);
        // `last_seen_at TEXT NOT NULL` (`schema.rs`) — a present row always
        // carries SOME text; an absent row is the only `None` here.
        let last_seen_at: Option<String> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let sql = "SELECT last_seen_at FROM instances WHERE instance_id = $1";
                        tx.query_opt(sql, &[SqlValue::TextOwned(instance_id)], |row| {
                            row.get::<String>("last_seen_at")
                        })
                        .await
                    })
                },
            )
            .await?;
        Ok(last_seen_at.is_some_and(|last_seen_at| {
            super::lease::last_seen_at_is_fresh(
                &last_seen_at,
                margin,
                super::lease::app_clock_now(),
            )
        }))
    }

    /// The ONE by-id peer-address resolution verb (DESIGN.md § 4): `Some`
    /// iff `instance_id`'s row is present, its `peer_addr` is non-NULL, and
    /// it is fresh under [`super::lease::instance_liveness_margin`] on the
    /// DB clock. No kind / root / self filter — a rank resolving its own
    /// coordinator (or a peer resolving ANY other member by id, including a
    /// busy or other-kind one) uses this, never `list_gang_members`.
    /// `instances` carries no tenant column, so there is no tenant predicate
    /// to drop or keep — the same answer under a scoped tenant binding and
    /// under none.
    ///
    /// # Errors
    ///
    /// A stored `peer_addr` that fails [`PeerAddr::parse`] is a typed
    /// [`JammiError::Catalog`] — a corrupted row fact, never silently mapped
    /// to `None` the way an absent or stale row is.
    pub async fn peer_addr_of(
        &self,
        instance_id: &str,
        lease: Duration,
    ) -> Result<Option<PeerAddr>> {
        let instance_id_owned = instance_id.to_string();
        let kind = self.backend().backend_kind();
        let margin = instance_liveness_margin(lease);
        let raw = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let mut params: Vec<SqlValue<'static>> = Vec::new();
                        let stale = stale_before_clause("last_seen_at", kind, margin, &mut params);
                        params.push(SqlValue::TextOwned(instance_id_owned));
                        let id_bind = params.len();
                        let sql = format!(
                            "SELECT peer_addr FROM instances \
                             WHERE instance_id = ${id_bind} AND peer_addr IS NOT NULL \
                               AND NOT ({stale})"
                        );
                        tx.query_opt(&sql, &params, |row| row.get::<String>("peer_addr"))
                            .await
                    })
                },
            )
            .await?;
        match raw {
            None => Ok(None),
            Some(addr) => PeerAddr::parse(&addr).map(Some).map_err(|e| {
                JammiError::Catalog(format!(
                    "instance '{instance_id}' has a corrupted peer_addr row: {e}"
                ))
            }),
        }
    }

    /// The gang-membership listing verb (DESIGN.md § 4, M3): every FRESH,
    /// `claiming` worker whose `kinds` contains `listing.kind` as a whole,
    /// trimmed, comma-split token, excluding `listing.self_instance` —
    /// sorted by `instance_id` BYTE ORDER, in Rust, never a SQL `ORDER BY`
    /// (backend-dependent collation). An INNER join on `workers`: a member
    /// is a fleet worker with a claim-loop slot, not merely a live process.
    /// `instances` carries no tenant column: the same answer under a scoped
    /// tenant binding and under none.
    ///
    /// **Root identity is part of this predicate** (unit U5b-1a-A2): a
    /// candidate's `instances.result_root_identity` must EQUAL the identity
    /// of `listing.root`, the caller's own [`super::instance::MemberRoot`]
    /// (the value its row carries — never a bare identity; the identity is
    /// derived once by each member from its verbatim root and the store's
    /// cloud config). Two members rooted at byte-DIFFERENT spellings of
    /// the SAME location (`gcs://b/p` and `gs://b/p`, a symlinked local
    /// root and its target) ARE gang members of each other; two members
    /// rooted at different locations are not; a row whose identity is NULL
    /// (written before the column existed, or by a process with no
    /// membership) never matches. The verbatim `result_root` column is
    /// never compared — it is carried for humans, not for this predicate.
    ///
    /// Filtering is split deliberately: freshness, NULL-ness and the
    /// identity equality are pushed into SQL (an index-backed predicate
    /// over a potentially large table; the identity is a byte-exact `=` on a
    /// string this engine wrote, never a collation question); everything
    /// else — the self exclusion, the worker state, and the kind token
    /// match — runs in Rust, over the already-narrowed row set, so no SQL
    /// dialect's string/collation semantics can silently diverge from what
    /// this verb promises.
    ///
    /// # Errors
    ///
    /// A stored `peer_addr` that fails [`PeerAddr::parse`] is a typed
    /// [`JammiError::Catalog`] naming the corrupted instance.
    pub async fn list_gang_members(&self, listing: GangListing<'_>) -> Result<Vec<GangMember>> {
        let kind = self.backend().backend_kind();
        let margin = instance_liveness_margin(listing.lease);
        let root_identity = listing.root.identity().as_str().to_string();
        let rows: Vec<GangCandidateRow> = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let mut params: Vec<SqlValue<'static>> = Vec::new();
                        let stale =
                            stale_before_clause("i.last_seen_at", kind, margin, &mut params);
                        params.push(SqlValue::TextOwned(root_identity));
                        let identity_bind = params.len();
                        let sql = format!(
                            "SELECT i.instance_id AS instance_id, i.peer_addr AS peer_addr, \
                                    w.kinds AS kinds, w.state AS state \
                             FROM instances i JOIN workers w ON w.instance_id = i.instance_id \
                             WHERE i.peer_addr IS NOT NULL \
                               AND i.result_root_identity = ${identity_bind} \
                               AND NOT ({stale})"
                        );
                        tx.query(&sql, &params, |row| {
                            Ok(GangCandidateRow {
                                instance_id: row.get::<String>("instance_id")?,
                                peer_addr: row.get::<String>("peer_addr")?,
                                kinds: row.get::<String>("kinds")?,
                                state: row.get::<String>("state")?,
                            })
                        })
                        .await
                    })
                },
            )
            .await?;

        let claiming = WorkerState::Claiming.as_db_str();
        let mut members = Vec::new();
        for row in rows {
            if row.instance_id == listing.self_instance {
                continue;
            }
            if row.state != claiming {
                continue;
            }
            if !row
                .kinds
                .split(',')
                .map(str::trim)
                .any(|token| token == listing.kind)
            {
                continue;
            }
            let peer_addr = PeerAddr::parse(&row.peer_addr).map_err(|e| {
                JammiError::Catalog(format!(
                    "instance '{}' has a corrupted peer_addr row: {e}",
                    row.instance_id
                ))
            })?;
            members.push(GangMember {
                instance_id: row.instance_id,
                peer_addr,
            });
        }
        members.sort_by(|a, b| a.instance_id.as_bytes().cmp(b.instance_id.as_bytes()));
        Ok(members)
    }

    /// Upsert this process's `workers` row — present only while the process
    /// runs the claim loop. `kinds` is the `,`-joined kind set this worker
    /// claims — the ONLY encoding this column carries, since
    /// [`Self::list_gang_members`] splits it on `,` and trims each token
    /// (above, where each row's `kinds` is matched against
    /// `listing.kind`); `state` is the loop's lifecycle state at this
    /// instant (the loop task writes `warming` as its FIRST statement and
    /// re-upserts `claiming` once its gate opens — never a bare `UPDATE`,
    /// so a row the first write failed to create is created at the
    /// transition); `devices` is this `[worker]` process's own device
    /// inventory (`ListWorkers` mirror only, read back verbatim on
    /// `jammi.v1.job.WorkerSummary.devices` — see [`WorkerRecord::devices`]'s
    /// doc), JSON-encoded verbatim into `workers.devices`. A re-upsert on an
    /// existing row resets all three.
    ///
    /// # Errors
    ///
    /// Under `feature = "test-hooks"`, a failure armed for `instance_id`
    /// through `worker_test_hooks::arm_upsert_worker_failure` (not an
    /// intra-doc link: that module exists only under `feature =
    /// "test-hooks"`, so a link to it fails `cargo doc`'s default-feature
    /// pass) is returned here, once, with no write attempted — the
    /// deterministic fixture P-Y4 (contract `feat_500-C-U5b-1a` §12) needs
    /// to prove the caller's registration cell stays cleared when this call
    /// fails.
    pub async fn upsert_worker(
        &self,
        instance_id: &str,
        kinds: &str,
        state: WorkerState,
        devices: &[DeviceFact],
    ) -> Result<()> {
        #[cfg(feature = "test-hooks")]
        if super::worker_test_hooks::take_armed(instance_id) {
            return Err(JammiError::Catalog(format!(
                "test-hooks: injected upsert_worker failure for instance '{instance_id}'"
            )));
        }
        let instance_id = instance_id.to_string();
        let kinds = kinds.to_string();
        let state = state.as_db_str();
        let devices_json = serde_json::to_string(devices)
            .map_err(|e| JammiError::Catalog(format!("devices encode: {e}")))?;
        self.backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO workers (instance_id, kinds, state, devices) \
                         VALUES ($1, $2, $3, $4) \
                         ON CONFLICT(instance_id) DO UPDATE \
                         SET kinds = excluded.kinds, state = excluded.state, \
                             devices = excluded.devices",
                        &[
                            SqlValue::TextOwned(instance_id),
                            SqlValue::TextOwned(kinds),
                            SqlValue::Text(state),
                            SqlValue::TextOwned(devices_json),
                        ],
                    )
                    .await
                })
            })
            .await?;
        Ok(())
    }

    /// Flip this process's `workers.state` (`claiming` when the worker gate
    /// opens, `draining` when a DRAIN begins). `false` when no row exists —
    /// the loop never ran, or the row was already deleted.
    pub async fn set_worker_state(&self, instance_id: &str, state: WorkerState) -> Result<bool> {
        let instance_id = instance_id.to_string();
        let state = state.as_db_str();
        let updated = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "UPDATE workers SET state = $1 WHERE instance_id = $2",
                        &[SqlValue::Text(state), SqlValue::TextOwned(instance_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(updated == 1)
    }

    /// The gauge sampler's one query: `(kind, status, count)` over every
    /// loop-claimable row — `execution = 'queued'` and `status IN ('queued',
    /// 'running')`, grouped by kind and status — an index-only aggregate on
    /// `idx_jobs_kind_status` (migration 031). Inline rows and terminal rows
    /// are excluded; a held (`claimable = false`) row is still counted as
    /// queued. Tenant-unscoped by design (a queue depth is a process-level
    /// signal, not a tenant dimension). Sorted in Rust — the backend's
    /// `GROUP BY` order is never trusted.
    pub async fn count_jobs_by_kind_status(&self) -> Result<Vec<(String, String, i64)>> {
        let queued_execution = JobExecution::Queued.to_string();
        let queued = JobStatus::Queued.to_string();
        let running = JobStatus::Running.to_string();
        let mut rows = self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT kind, status, COUNT(*) AS n FROM jobs \
                             WHERE execution = $1 AND status IN ($2, $3) \
                             GROUP BY kind, status",
                            &[
                                SqlValue::TextOwned(queued_execution),
                                SqlValue::TextOwned(queued),
                                SqlValue::TextOwned(running),
                            ],
                            |row| {
                                Ok((
                                    row.get::<String>("kind")?,
                                    row.get::<String>("status")?,
                                    row.get::<i64>("n")?,
                                ))
                            },
                        )
                        .await
                    })
                },
            )
            .await?;
        rows.sort();
        Ok(rows)
    }

    /// Delete this process's `workers` row — the claim loop has stopped, so
    /// the process must not advertise itself as a claimant (a
    /// `ListWorkers` read after an `EmbeddedWorker` drop shows no row for
    /// it, rather than a row that lingers until its `instances` row goes
    /// stale and cascades). The `instances` row itself is untouched: the
    /// process is still alive and may still hold inline jobs. `false` when
    /// no row existed.
    pub async fn delete_worker(&self, instance_id: &str) -> Result<bool> {
        let instance_id = instance_id.to_string();
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    tx.execute(
                        "DELETE FROM workers WHERE instance_id = $1",
                        &[SqlValue::TextOwned(instance_id)],
                    )
                    .await
                })
            })
            .await?;
        Ok(deleted == 1)
    }

    /// List every worker, joined with its owning `instances` row (N12).
    pub async fn list_workers(&self) -> Result<Vec<WorkerRecord>> {
        Ok(self
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        tx.query(
                            "SELECT w.instance_id AS instance_id, i.label AS label, \
                                    i.host AS host, w.kinds AS kinds, w.state AS state, \
                                    i.started_at AS started_at, i.last_seen_at AS last_seen_at, \
                                    w.devices AS devices \
                             FROM workers w JOIN instances i ON w.instance_id = i.instance_id \
                             ORDER BY i.started_at",
                            &[],
                            parse_worker_row,
                        )
                        .await
                    })
                },
            )
            .await?)
    }

    /// Delete `instances` rows stale for at least `stale_after` — their
    /// `workers` row (if any) cascades with them. Returns the count deleted.
    pub async fn prune_instances(&self, stale_after: Duration) -> Result<usize> {
        let kind = self.backend().backend_kind();
        let mut params: Vec<SqlValue<'static>> = Vec::new();
        let stale = stale_before_clause("last_seen_at", kind, stale_after, &mut params);
        let sql = format!("DELETE FROM instances WHERE {stale}");
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(deleted as usize)
    }

    /// Delete terminal (`completed`/`failed` — [`JobStatus::
    /// terminal_sql_list`], never a hand-typed pair) job rows whose
    /// `updated_at` is at least `retention` in the past. A non-terminal job
    /// is never pruned, regardless of age. Returns the count deleted.
    ///
    /// Tenant-scoped with the same STRICT predicate [`Self::cancel_request`]
    /// uses: a caller prunes only rows whose `tenant_id` equals its own (or
    /// is `NULL`, the global namespace) — never a peer tenant's terminal
    /// rows, however old. Inside a
    /// [`crate::session::JammiSession::with_admin_scope`] closure the
    /// predicate is dropped and every tenant's terminal rows are eligible;
    /// `InferenceSession::wrap`'s construction-time sweep is the one
    /// production caller that runs under that admin bypass (it is not an
    /// RPC — `JobService::PruneJobs` always calls this under the caller's
    /// own `scoped(...)` tenant, exactly like every other job RPC).
    pub async fn prune_jobs(&self, retention: Duration) -> Result<usize> {
        let admin = TenantBinding::is_admin_scope();
        let kind = self.backend().backend_kind();
        let mut params: Vec<SqlValue<'static>> = Vec::new();
        let stale = stale_before_clause("updated_at", kind, retention, &mut params);
        let terminal = JobStatus::terminal_sql_list();
        let tenant = self.current_tenant();
        let sql = if admin {
            format!("DELETE FROM jobs WHERE status IN ({terminal}) AND {stale}")
        } else {
            params.push(SqlValue::from(tenant.map(|t| t.to_string())));
            let tenant_bind = params.len();
            format!(
                "DELETE FROM jobs WHERE status IN ({terminal}) AND {stale} \
                   AND (tenant_id = ${tenant_bind} OR (tenant_id IS NULL AND ${tenant_bind} IS NULL))"
            )
        };
        let deleted = self
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move { tx.execute(&sql, &params).await })
            })
            .await?;
        Ok(deleted as usize)
    }
}
