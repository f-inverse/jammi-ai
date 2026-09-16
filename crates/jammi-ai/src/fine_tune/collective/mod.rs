//! The collective: what a gang of training ranks does at a step boundary.
//!
//! One trait, one implementation per transport, selected by CONFIGURATION —
//! `[worker] collective` and `[worker] local_ranks`, never a cargo feature. The
//! trainer holds a `&dyn Collective` and is never `cfg`-forked: a single-rank
//! run holds a [`Noop`], a multi-rank run on one host holds a [`Local`], a
//! multi-host run holds a [`Peer`] (rank 0 in the coordinator's process, every
//! other rank on the far end of one admitted `RunRank` stream), and — on a
//! CUDA build, where the `nccl` submodule exists — an `nccl::Nccl`; the
//! trainer's own code is the same code in every case. (A link rather than a
//! code span would resolve only on a CUDA build, and fail the docs lane on
//! every other.)
//!
//! # The blocking-call witness
//!
//! Every verb takes a [`BlockingCall`]: a thread-bound witness that the
//! caller is on a thread that MAY block — a `spawn_blocking` thread or a
//! plain OS thread — never a runtime worker thread. [`Peer`] drives async
//! stream I/O under `Handle::block_on`, which is a panic on a worker thread;
//! the witness turns that from a runtime failure into a COMPILE error: it has
//! no public constructor, it is minted only inside the closures
//! [`BlockingCall::spawn_blocking`] / [`BlockingCall::spawn_thread`] /
//! [`BlockingCall::spawn_scoped`] run on the thread they create, and it is
//! `!Send + !Sync`, so it cannot be carried into a `tokio::spawn`ed future
//! or stored anywhere a worker thread could reach it. The witness lives on
//! the TRAIT, not on `Peer` alone, because the trainer holds a `&dyn
//! Collective` and never names `Peer`: a guarantee on `Peer`'s inherent
//! methods would be invisible at the one call site that matters. [`Noop`],
//! [`Local`] and `Nccl` accept the witness and ignore it — one ignored
//! parameter each is the whole cost of a discipline that is compile-checked
//! on every arm.
//!
//! # The five operations
//!
//! - [`Collective::all_gather`] — every rank's slice of one step's batch,
//!   concatenated in RANK ORDER, so every rank computes the identical global
//!   loss over the identical rows.
//! - [`Collective::all_reduce_sum`] — the gradient sum at the optimizer-step
//!   boundary, over tensors laid out in the canonical trainable-variable
//!   order.
//! - [`Collective::all_reduce_max_flags`] — the lockstep control word
//!   (divergence, early stop, epoch exit): one `u32` of bit flags, maxed, so
//!   every rank leaves the boundary with the same decision.
//! - [`Collective::broadcast`] — one rank's tensor to every rank (the
//!   validation-loss stop signal, a scaler).
//! - [`Collective::barrier`] — the ordering point with no payload.
//!
//! # What every implementation guarantees
//!
//! - **Rank-ordered, fixed-fold results.** A gather is the rank-ordered
//!   concatenation and a sum is the rank-ordered fold; both are a pure
//!   function of the ranks' inputs, so two runs of the same gang over the
//!   same inputs produce byte-identical outputs, and the result equals the
//!   serial reference bit-for-bit on a host device.
//! - **Counts are derived, never exchanged.** [`Collective::all_gather`]
//!   takes the counts every rank derives locally from the partition rule; no
//!   implementation performs a counts exchange, so no implementation can
//!   deadlock on one.
//! - **Every implementation checks its own caller's arguments.** A count
//!   vector that is not `world`-long, a local tensor whose row count
//!   contradicts its own entry, a root outside the gang: each is a typed
//!   error at the seam, never a panic and never a silently wrong reduction.
//!   These are decided from what THIS rank passed, so every arm decides them
//!   the same way, through the same two helpers.
//!
//! # What the arms that can see every rank's arguments guarantee
//!
//! A round is agreed on its [`Descriptor`], the round index included.
//!
//! A check on what a PEER passed needs the peer's arguments. [`Local`] has
//! them — the rendezvous carries each rank's whole contribution — and so
//! does [`Peer`] — every member's contribution reaches the coordinator with
//! its descriptor. On both, a rank deposits a [`Descriptor`] alongside its
//! contribution: the verb, and every argument other than the tensor bytes
//! that determines the round's result (`root` for `broadcast`, the full
//! `counts` vector for `all_gather`, one signature per tensor the verb
//! carries — for `all_gather` the TRAILING shape only, since dim 0 is
//! already the `counts` vector and legitimately differs by rank — the world
//! size, and the caller-bound [`Descriptor::agreement`] digest). A round is
//! published ONLY once every rank's descriptor for it is equal; on any
//! disagreement BEFORE the round publishes, no rank is ever handed a result
//! — every rank gets a typed error naming both descriptors, and the gang is
//! faulted before any of them returns. So on these arms, **no rank can ever
//! return `Ok` from a round any other rank rejects before that round
//! publishes**: two ranks each naming themselves root, or deriving different
//! partition counts, are a symmetric typed error on both, never an `Ok` on
//! one and a wrong answer (or a different error) on the other. A rank-local
//! failure AFTER a round has published (a `to_device` copy that fails, a
//! concatenation the backend refuses) is a different case: the round has
//! already handed every rank its agreed result, so that failure faults the
//! gang for every collective AFTER this one, but it does not — and cannot —
//! retract the `Ok` peers already hold for this one. [`Noop`] has no peer to
//! disagree with (`world` is always 1), so this is vacuous there.
//!
//! Publish-then-fault is a real state, not a theoretical one: a round can
//! publish (every rank's descriptor agreed) and then fault before every rank
//! has taken its result — a rank that already took the published value keeps
//! its `Ok`, a rank that had not gets the gang's fault instead, and no rank
//! is ever handed a wrong result. [`Local`] can afford to let that asymmetry
//! stand because every rank is a thread of one process sharing one `Round`.
//! [`Peer`] decides it explicitly with a two-phase round (see
//! [`peer`]'s module doc): a result is held UNAPPLIED on every member until
//! the coordinator, having observed every member's ACK, commits it — a fault
//! before the last ACK leaves no rank applied, and the commit point (the last
//! ACK observed by the coordinator) is the one state a later fault cannot
//! retract.
//!
//! The `Nccl` arm has none of that. NCCL exchanges the buffers a collective
//! names and nothing else: there is no counts exchange (by design — see
//! above), so a peer's counts, its tensor-list length and its idea of the
//! root are not observable to this rank, and a gang whose ranks disagree
//! about any of them produces a wrong result or a hang rather than a typed
//! error. The failure signal for that arm is the watchdog abort — a peer
//! that never answers leaves this rank in `synchronize`, another thread
//! calls `nccl::Nccl::abort`, and the abort FLAG (never the collective's
//! return value, which is `Ok` over a garbage buffer) is what says the
//! attempt failed. A gang is kept in agreement upstream of the collective,
//! by every rank deriving its counts from the same partition rule and
//! walking the same canonical trainable-variable order, and not by this
//! seam.

use std::fmt;
use std::marker::PhantomData;

use jammi_db::error::{JammiError, Result};

use candle_core::{DType, Tensor};

pub mod local;
#[cfg(feature = "cuda")]
pub mod nccl;
pub mod noop;
pub mod peer;

#[cfg(test)]
mod peer_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::witness;

pub use local::{Local, LocalGang};
pub use noop::Noop;
pub use peer::{CoordinatorLink, LinkFault, MemberEnd, MemberLink, Peer, RankReadFault};

/// A thread-bound witness that the current thread may block.
///
/// Minted ONLY inside the closure one of [`Self::spawn_blocking`],
/// [`Self::spawn_thread`] or [`Self::spawn_scoped`] runs on the thread it
/// creates — a tokio blocking-pool thread or a plain OS thread, never a
/// runtime worker thread — and `!Send + !Sync`, so it cannot be moved into
/// a `tokio::spawn`ed future, stored in a `Send` value, or otherwise reach a
/// worker thread. Every [`Collective`] verb takes one; [`Peer`] is the arm
/// that needs it (it blocks on stream I/O), and every arm is compile-checked
/// by it: a verb called from a worker thread has no witness to pass.
///
/// # The compile-time claim, as doctests
///
/// The four blocks below are the oracle for that claim: rustdoc compiles
/// each one, requires the three `compile_fail` blocks to FAIL with the
/// error code named on the fence, and requires the control to pass
/// (`cargo test -p jammi-ai --doc -- BlockingCall`). Making the witness
/// `Send` (a `PhantomData<()>`) turns the `E0277` block into a successful
/// compile, which fails the doctest.
///
/// The control — the same verb from a `spawn_blocking` closure compiles
/// and runs:
///
/// ```
/// use jammi_ai::fine_tune::collective::{BlockingCall, Collective, Noop};
///
/// let rt = tokio::runtime::Builder::new_current_thread()
///     .enable_all()
///     .build()
///     .unwrap();
/// rt.block_on(async {
///     let noop = Noop::new();
///     BlockingCall::spawn_blocking(move |call| noop.barrier(&call))
///         .await
///         .unwrap()
///         .unwrap();
/// });
/// ```
///
/// A witness minted on a blocking thread cannot be carried into a
/// `tokio::spawn`ed future — it is not `Send` — so the verb cannot run on a
/// runtime worker thread:
///
/// ```compile_fail,E0277
/// use jammi_ai::fine_tune::collective::{BlockingCall, Collective, Noop};
///
/// let rt = tokio::runtime::Runtime::new().unwrap();
/// rt.block_on(async {
///     let noop = Noop::new();
///     let handle = BlockingCall::spawn_blocking(move |call| {
///         tokio::spawn(async move { noop.barrier(&call) })
///     });
///     let _ = handle;
/// });
/// ```
///
/// On a worker thread there is no witness to pass, so the verb cannot be
/// called at all:
///
/// ```compile_fail,E0061
/// use jammi_ai::fine_tune::collective::{Collective, Noop};
///
/// let rt = tokio::runtime::Runtime::new().unwrap();
/// rt.block_on(async {
///     let noop = Noop::new();
///     noop.barrier().unwrap();
/// });
/// ```
///
/// The constructor is private — there is no fourth minting site:
///
/// ```compile_fail,E0624
/// use jammi_ai::fine_tune::collective::{BlockingCall, Collective, Noop};
///
/// let call = BlockingCall::mint();
/// Noop::new().barrier(&call).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BlockingCall {
    _thread_bound: PhantomData<*const ()>,
}

impl BlockingCall {
    /// The one constructor, private: reachable only from the three minting
    /// sites below, each of which is by construction on a thread that is not
    /// a runtime worker. Production mints at three places, all in
    /// `worker.rs`: rank 0's `spawn_blocking` and a `Local` gang's per-rank
    /// `spawn_thread` (both in `train_fine_tune`), and an admitted member's
    /// `spawn_blocking` in `run_member_rank` (the rank body).
    fn mint() -> Self {
        Self {
            _thread_bound: PhantomData,
        }
    }

    /// Run `f` on tokio's blocking pool with a fresh witness — the first
    /// and the third of the three production minting sites: the worker
    /// spawns rank 0's training thread (the single rank, a `Local` gang's
    /// rank 0, or a `Peer` gang's coordinator) here (`worker.rs`,
    /// `train_fine_tune`), and an admitted member's rank body spawns ITS
    /// training thread here too (`worker.rs`, `run_member_rank` — the
    /// member's `TrainingLoop::run` receives the witness minted at its own
    /// boundary, exactly as rank 0 does).
    pub fn spawn_blocking<F, T>(f: F) -> tokio::task::JoinHandle<T>
    where
        F: FnOnce(BlockingCall) -> T + Send + 'static,
        T: Send + 'static,
    {
        tokio::task::spawn_blocking(move || f(Self::mint()))
    }

    /// Run `f` on a fresh OS thread with a fresh witness — the second
    /// production minting site: the worker spawns every OTHER rank of an
    /// in-process `Local` gang here, one thread pinned to one device, each
    /// rank's `TrainingLoop::run` receiving the witness minted at its own
    /// boundary (`worker.rs`, `train_fine_tune`). An OS thread has no
    /// runtime context of its own, so it can block; [`Peer`] blocks on the
    /// [`tokio::runtime::Handle`] its links captured, which is allowed from
    /// any thread that is not one of that runtime's workers (the worker's
    /// rank threads enter the runtime's handle for the same reason).
    pub fn spawn_thread<F, T>(f: F) -> std::thread::JoinHandle<T>
    where
        F: FnOnce(BlockingCall) -> T + Send + 'static,
        T: Send + 'static,
    {
        std::thread::spawn(move || f(Self::mint()))
    }

    /// [`Self::spawn_thread`] inside a [`std::thread::scope`].
    pub fn spawn_scoped<'scope, F, T>(
        scope: &'scope std::thread::Scope<'scope, '_>,
        f: F,
    ) -> std::thread::ScopedJoinHandle<'scope, T>
    where
        F: FnOnce(BlockingCall) -> T + Send + 'scope,
        T: Send + 'scope,
    {
        scope.spawn(move || f(Self::mint()))
    }
}

/// The five collectives, as a round's descriptor names them. CLOSED — the
/// wire's `RoundVerb` mirrors it value for value, and a wire value outside
/// this set is a refusal, never a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verb {
    AllGather,
    AllReduceSum,
    AllReduceMaxFlags,
    Broadcast,
    Barrier,
}

impl Verb {
    /// The trait method's name — what every error message is prefixed with.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AllGather => "all_gather",
            Self::AllReduceSum => "all_reduce_sum",
            Self::AllReduceMaxFlags => "all_reduce_max_flags",
            Self::Broadcast => "broadcast",
            Self::Barrier => "barrier",
        }
    }
}

impl fmt::Display for Verb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One rank's declaration of what a round computes: the verb, and every
/// per-call argument other than the tensor bytes that determines the round's
/// result.
///
/// An arm that can see every rank's descriptor ([`Local`]'s rendezvous,
/// [`Peer`]'s coordinator) publishes a round ONLY once every rank's
/// descriptor for it is equal (checked by [`Descriptor::agrees_with`]); on
/// any disagreement the round is never published, and every rank gets a
/// typed error naming both descriptors instead. This is the ONE place a
/// cross-rank agreement check lives — no verb's trait method runs its own
/// peer-specific check outside it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Descriptor {
    /// The round this rank is entering, counted from 0 on every rank of the
    /// gang. A determinant like every other field: the same verb and
    /// arguments in a different round is a different round, so a
    /// contribution that outlived its round can never agree with the round
    /// being completed. [`Local`] stamps it under its rendezvous lock from
    /// the shared round's generation; [`Peer`] stamps it from each rank's
    /// own round counter, and the wire carries it.
    pub round: u64,
    /// The operation.
    pub verb: Verb,
    /// [`Collective::world`] as this rank sees it.
    pub world: usize,
    /// [`Collective::broadcast`]'s `root`; `None` for every other verb.
    pub root: Option<u32>,
    /// [`Collective::all_gather`]'s full `counts` vector; `None` for every
    /// other verb.
    pub counts: Option<Vec<usize>>,
    /// One entry per tensor this call carries, in the order the verb defines
    /// it: `all_gather`'s single `local`, `all_reduce_sum`'s slice in
    /// canonical order, or `broadcast`'s `t`. Empty for
    /// `all_reduce_max_flags` and `barrier`, which carry no tensor.
    pub tensors: Vec<TensorSignature>,
    /// An opaque digest the CALLER binds on its rank's collective (a trainer
    /// binds the digest of its canonical trainable-variable key order —
    /// `Local::with_agreement` / `Peer::with_agreement`); `None` when nothing
    /// is bound. The collective computes nothing from it and compares it
    /// like every other field, so two ranks bound to different values — or
    /// one bound and one not — disagree.
    pub agreement: Option<String>,
}

/// One tensor's shape and dtype, as far as a round's descriptor cares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSignature {
    pub dims: Vec<usize>,
    pub dtype: DType,
}

impl TensorSignature {
    /// The full shape and dtype of `t`.
    pub fn of(t: &Tensor) -> Self {
        Self {
            dims: t.dims().to_vec(),
            dtype: t.dtype(),
        }
    }

    /// [`Self::of`] with dim 0 dropped: `all_gather`'s row count is already
    /// the descriptor's [`Descriptor::counts`] and legitimately differs by
    /// rank (a zero-row rank, an uneven partition), so only the TRAILING
    /// shape is a determinant of agreement for a gathered tensor.
    ///
    /// Every caller reaches this only after `checked_gather_counts` has
    /// already refused a 0-dim tensor for this exact call, so `dims` always
    /// has at least one entry here. This does not fall back to signing an
    /// empty trailing shape for a shape it cannot honestly sign — a 0-dim
    /// tensor used to reach this via `unwrap_or_default()` and be signed
    /// exactly like a 1-D tensor of the same (empty) trailing shape, which
    /// was the defect.
    pub fn of_gather_slice(t: &Tensor) -> Self {
        let dims = t.dims();
        let trailing = dims.get(1..).expect(
            "checked_gather_counts already refused a 0-dim tensor before this call reaches \
             of_gather_slice",
        );
        Self {
            dims: trailing.to_vec(),
            dtype: t.dtype(),
        }
    }
}

impl Descriptor {
    /// `true` when every field this round's result depends on agrees with
    /// `other`.
    ///
    /// Delegates to the derived [`PartialEq`] rather than repeating a
    /// hand-written per-field comparison: [`Descriptor`] carries no field
    /// that is not itself a determinant of a round's result (a different
    /// root, a different partition, a different trainable-variable count, a
    /// differently shaped or typed tensor, a different world size, or a
    /// different caller-bound agreement — see the struct's own field docs),
    /// so the derive already computes exactly the comparison this round
    /// needs. A hand-written comparison is exactly what a NEW field could be
    /// added without — silently exempting it from agreement;
    /// `agrees_with_matches_derived_equality_over_a_per_field_mutation_sweep`
    /// in `local.rs` destructures [`Descriptor`] field-by-field with no `..`,
    /// so a field added to the struct without a matching arm there fails to
    /// COMPILE.
    pub fn agrees_with(&self, other: &Descriptor) -> bool {
        self == other
    }
}

/// The gang's collective operations, as the trainer sees them.
///
/// `Send + Sync` because the trainer runs on a blocking thread and holds the
/// implementation behind an `Arc`; the CUDA arm carries the single-thread
/// discipline NCCL requires in its own documentation rather than in the
/// trait's bounds. Every verb takes a [`BlockingCall`] — see the module doc.
pub trait Collective: Send + Sync {
    /// Concatenate every rank's slice of this step's batch along dim 0, in
    /// RANK ORDER, and return the identical tensor on every rank.
    ///
    /// `counts[r]` is the number of rows rank `r` contributes, derived by
    /// EVERY rank from the partition rule — nothing is exchanged to discover
    /// them. `counts.len()` must equal [`Self::world`] and `counts[self.rank()]`
    /// must equal `local`'s dim-0 extent; either mismatch is a typed error
    /// rather than a wrong layout. A zero-count rank is a normal case (a
    /// batch whose row count is not a multiple of `world · batch`) and
    /// contributes no rows.
    ///
    /// The result has `counts.iter().sum()` rows: the padding an
    /// unequal-count NCCL gather needs internally is never visible here, so
    /// every implementation returns the layout a serial concatenation would.
    ///
    /// **Backward.** Only the calling rank's own slot carries a gradient;
    /// the remote slots are detached, so no gradient crosses to a peer and
    /// the summed gradient of a trainable parameter is not `world` times too
    /// large.
    fn all_gather(&self, call: &BlockingCall, local: &Tensor, counts: &[usize]) -> Result<Tensor>;

    /// Replace each tensor with the rank-ordered sum of every rank's tensor
    /// at that index. The slice is the canonical trainable-variable order,
    /// identical on every rank, and every rank must pass the same length and
    /// the same per-index shape/dtype.
    fn all_reduce_sum(&self, call: &BlockingCall, tensors: &mut [Tensor]) -> Result<()>;

    /// The bitwise-flag control word for the lockstep boundary: returns the
    /// maximum over every rank's `flags`, so a flag set on ANY rank is seen
    /// by all of them.
    fn all_reduce_max_flags(&self, call: &BlockingCall, flags: u32) -> Result<u32>;

    /// Replace `t` with rank `root`'s `t`. `root` must be a rank of this
    /// gang. Every rank — root or not — passes a `t` of the same shape and
    /// dtype: an arm that can see every rank's arguments (`Local`, `Peer`)
    /// refuses a mismatch symmetrically on EVERY rank, never only on the
    /// non-root rank whose placeholder happened to differ from the root's.
    fn broadcast(&self, call: &BlockingCall, t: &mut Tensor, root: u32) -> Result<()>;

    /// Return only once every rank has reached this call.
    fn barrier(&self, call: &BlockingCall) -> Result<()>;

    /// This rank's index in `0..world`.
    fn rank(&self) -> u32;

    /// How many ranks are in the gang.
    fn world(&self) -> u32;

    /// Bind the opaque agreement digest this rank signs every round's
    /// [`Descriptor`] with — the trainer's canonical trainable-variable
    /// layout (`RankContext::bind_agreement`), bound once the target is
    /// built and before the first collective, so two ranks whose layouts
    /// differ are a typed descriptor disagreement on every rank rather
    /// than a wrong fold. Binding the SAME digest again is a no-op; a
    /// DIFFERENT digest on a rank already bound is a typed error (a rank
    /// has exactly one layout per run). [`Noop`] has no peer to disagree
    /// with and `Nccl` carries no descriptor at all (the module doc's last
    /// paragraph): both accept and ignore it.
    fn bind_agreement(&self, digest: String) -> Result<()>;
}

/// The ONE rule behind [`Collective::bind_agreement`] on the arms that carry
/// a descriptor (`Local`, `Peer`): a slot binds once; the same digest again
/// is a no-op; a different digest on a bound slot is a typed error naming
/// both, since a rank has exactly one canonical layout per run.
pub(crate) fn bind_agreement_once(
    slot: &std::sync::OnceLock<String>,
    digest: String,
) -> Result<()> {
    match slot.set(digest) {
        Ok(()) => Ok(()),
        Err(rejected) => {
            let bound = slot.get().expect("set failed because a value is bound");
            if *bound == rejected {
                Ok(())
            } else {
                Err(JammiError::FineTune(format!(
                    "bind_agreement: this rank already signs its rounds with agreement {bound} \
                     and cannot be rebound to {rejected} — one canonical layout per run"
                )))
            }
        }
    }
}

/// Check `counts` against the gang's shape and the caller's own tensor —
/// shared by every implementation (`Noop`, `Local`, `Peer`, `Nccl`) so ONE
/// seam decides what a well-formed gather request is; no arm carries a
/// scalar check of its own.
///
/// A 0-dim tensor (a scalar) has no row count to check against `counts` at
/// all — `dims()` is empty, not `[0]` — so it is refused here, naming the
/// rank and the shape, BEFORE any arm's own gather logic ever runs. Without
/// this, a caller-derived `unwrap_or(0)` default would read a scalar as
/// though it claimed zero rows, which is a different — and valid — case (a
/// 1-D or higher tensor whose dim-0 extent happens to be zero).
///
/// Returns the total row count the gather produces.
pub(crate) fn checked_gather_counts(
    rank: u32,
    world: u32,
    local: &Tensor,
    counts: &[usize],
) -> Result<usize> {
    if counts.len() != world as usize {
        return Err(JammiError::FineTune(format!(
            "all_gather: counts has {} entries for a gang of {world} ranks — every rank \
             derives one count per rank from the partition rule",
            counts.len()
        )));
    }
    let dims = local.dims();
    let Some(&local_rows) = dims.first() else {
        return Err(JammiError::FineTune(format!(
            "all_gather: rank {rank} passed a 0-dim tensor (shape {dims:?}) — a gather \
             concatenates along dim 0, so a scalar has no row count to check against the \
             partition rule and cannot be a gather slice"
        )));
    };
    // `rank` is never checked against `world` here: every arm's constructor
    // already guarantees `rank < world` before a `Collective` value exists at
    // all — `Noop` hardcodes rank 0 of world 1, `LocalGang::rank` refuses a
    // rank outside the gang before handing out a `Local`, `Peer::member`
    // refuses the same before handing out a `Peer`, and `Nccl::from_rank`
    // before handing out an `Nccl`. So `counts.get(rank as usize)` returning
    // `None` here is unreachable
    // through any of the four arms today; it is still a typed error rather
    // than an index panic, for defense in depth, and no second refusal site
    // for `world == 0` / `rank >= world` is added anywhere else in this
    // module — the length check above and this one are the only two.
    let Some(&claimed) = counts.get(rank as usize) else {
        return Err(JammiError::FineTune(format!(
            "all_gather: rank {rank} is not an index into a {world}-entry counts vector — every \
             arm's constructor guarantees rank < world before a collective ever runs, so this \
             should be unreachable"
        )));
    };
    if local_rows != claimed {
        return Err(JammiError::FineTune(format!(
            "all_gather: rank {rank} holds {local_rows} rows but the partition rule says \
             {claimed} — the ranks disagree about the partition, so the gathered layout \
             would not be the one the peers assume"
        )));
    }
    Ok(counts.iter().sum())
}

/// Reject a `root` that is not a rank of this gang, rather than indexing a
/// slot that does not exist.
pub(crate) fn checked_root(world: u32, root: u32) -> Result<usize> {
    if root >= world {
        return Err(JammiError::FineTune(format!(
            "broadcast: root {root} is not a rank of a gang of {world}"
        )));
    }
    Ok(root as usize)
}
