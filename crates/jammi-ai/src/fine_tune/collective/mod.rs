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
//! `broadcast`, the full `counts` vector for `all_gather`, how many tensors
//! and each one's shape and dtype, and the world size). A round is published
//! ONLY once every rank's descriptor for it is equal; on any disagreement no
//! rank is ever handed a result — every rank gets a typed error naming both
//! descriptors, and the gang is faulted before any of them returns. So on
//! [`Local`], **no rank can ever return `Ok` from a round any other rank
//! rejects**: two ranks each naming themselves root, or deriving different
//! partition counts, are a symmetric typed error on both, never an `Ok` on
//! one and a wrong answer (or a different error) on the other. [`Noop`] has
//! no peer to disagree with (`world` is always 1), so this is vacuous there.
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
    /// gang.
    fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()>;

    /// Return only once every rank has reached this call.
    fn barrier(&self) -> Result<()>;

    /// This rank's index in `0..world`.
    fn rank(&self) -> u32;

    /// How many ranks are in the gang.
    fn world(&self) -> u32;
}

/// Check `counts` against the gang's shape and the caller's own tensor —
/// shared by every implementation so one seam decides what a well-formed
/// gather request is.
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
    let local_rows = local.dims().first().copied().unwrap_or(0);
    let claimed = counts[rank as usize];
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
