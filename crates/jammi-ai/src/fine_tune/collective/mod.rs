//! The collective: what a gang of training ranks does at a step boundary.
//!
//! One trait, one implementation per transport, selected by CONFIGURATION —
//! `[worker] collective` and `[worker] world_size`, never a cargo feature. The
//! trainer holds a `&dyn Collective` and is never `cfg`-forked: a single-rank
//! run holds a [`Noop`], a multi-rank run on one host holds a [`Local`] or —
//! on a CUDA build, where the `nccl` submodule exists — an `nccl::Nccl`, and
//! the trainer's own code is the same code in every case. (A link rather than
//! a code span would resolve only on a CUDA build, and fail the docs lane on
//! every other.)
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
//! # What only the host arms guarantee
//!
//! A check on what a PEER passed needs the peer's arguments. [`Local`] has
//! them — the rendezvous carries each rank's whole contribution — so a rank
//! deposits a **descriptor** alongside it: the verb, and every argument other
//! than the tensor bytes that determines the round's result (`root` for
//! `broadcast`, the full `counts` vector for `all_gather`, one signature per
//! tensor the verb carries — for `all_gather` the TRAILING shape only, since
//! dim 0 is already the `counts` vector and legitimately differs by rank —
//! and the world size). A round is published ONLY once every rank's
//! descriptor for it is equal; on any disagreement BEFORE the round
//! publishes, no rank is ever handed a result — every rank gets a typed error
//! naming both descriptors, and the gang is faulted before any of them
//! returns. So on [`Local`], **no rank can ever return `Ok` from a round any
//! other rank rejects before that round publishes**: two ranks each naming
//! themselves root, or deriving different partition counts, are a symmetric
//! typed error on both, never an `Ok` on one and a wrong answer (or a
//! different error) on the other. A rank-local failure AFTER a round has
//! published (a `to_device` copy that fails, a concatenation the backend
//! refuses) is a different case: the round has already handed every rank its
//! agreed result, so that failure faults the gang for every collective AFTER
//! this one, but it does not — and cannot — retract the `Ok` peers already
//! hold for this one. [`Noop`] has no peer to disagree with (`world` is
//! always 1), so this is vacuous there.
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

use jammi_db::error::{JammiError, Result};

use candle_core::Tensor;

pub mod local;
#[cfg(feature = "cuda")]
pub mod nccl;
pub mod noop;

#[cfg(test)]
mod tests;

pub use local::{Local, LocalGang};
pub use noop::Noop;

/// The gang's collective operations, as the trainer sees them.
///
/// `Send + Sync` because the trainer runs on a worker thread and holds the
/// implementation behind an `Arc`; the CUDA arm carries the single-thread
/// discipline NCCL requires in its own documentation rather than in the
/// trait's bounds.
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
    fn all_gather(&self, local: &Tensor, counts: &[usize]) -> Result<Tensor>;

    /// Replace each tensor with the rank-ordered sum of every rank's tensor
    /// at that index. The slice is the canonical trainable-variable order,
    /// identical on every rank, and every rank must pass the same length and
    /// the same per-index shape/dtype.
    fn all_reduce_sum(&self, tensors: &mut [Tensor]) -> Result<()>;

    /// The bitwise-flag control word for the lockstep boundary: returns the
    /// maximum over every rank's `flags`, so a flag set on ANY rank is seen
    /// by all of them.
    fn all_reduce_max_flags(&self, flags: u32) -> Result<u32>;

    /// Replace `t` with rank `root`'s `t`. `root` must be a rank of this
    /// gang. Every rank — root or not — passes a `t` of the same shape and
    /// dtype: a host arm that can see every rank's arguments (`Local`)
    /// refuses a mismatch symmetrically on EVERY rank, never only on the
    /// non-root rank whose placeholder happened to differ from the root's.
    fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()>;

    /// Return only once every rank has reached this call.
    fn barrier(&self) -> Result<()>;

    /// This rank's index in `0..world`.
    fn rank(&self) -> u32;

    /// How many ranks are in the gang.
    fn world(&self) -> u32;
}

/// Check `counts` against the gang's shape and the caller's own tensor —
/// shared by every implementation (`Noop`, `Local`, `Nccl`) so ONE seam
/// decides what a well-formed gather request is; no arm carries a scalar
/// check of its own.
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
    // rank outside the gang before handing out a `Local`, and
    // `Nccl::from_rank` refuses the same before handing out an `Nccl`. So
    // `counts.get(rank as usize)` returning `None` here is unreachable
    // through any of the three arms today; it is still a typed error rather
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
