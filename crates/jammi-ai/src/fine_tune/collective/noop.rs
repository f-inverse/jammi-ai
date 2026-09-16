//! The single-rank collective: every operation is the identity.

use jammi_db::error::Result;

use candle_core::Tensor;

use super::{checked_gather_counts, checked_root, BlockingCall, Collective};

/// The collective of a gang of one.
///
/// A `world_size = 1` run holds this, so the trainer's collective calls are
/// present in the single-rank path too — the multi-rank arms differ from it
/// in what they do, never in whether the trainer calls them. Identity in the
/// strict sense: `all_gather` returns the caller's own tensor, `all_reduce_*`
/// returns the caller's own values, `broadcast` leaves the caller's tensor
/// alone, and `barrier` returns immediately. Nothing is copied, so a
/// single-rank run pays nothing for the seam.
///
/// The domain checks still run: a `counts` vector that is not one entry long,
/// or one that contradicts the local row count, is a partition-rule bug that
/// would be a wrong layout at `world > 1`, and it is refused here rather than
/// passing silently on the one topology that cannot expose it.
#[derive(Debug, Default, Clone, Copy)]
pub struct Noop;

impl Noop {
    /// The single-rank collective.
    pub fn new() -> Self {
        Self
    }
}

impl Collective for Noop {
    fn all_gather(&self, _call: &BlockingCall, local: &Tensor, counts: &[usize]) -> Result<Tensor> {
        checked_gather_counts(0, 1, local, counts)?;
        Ok(local.clone())
    }

    fn all_reduce_sum(&self, _call: &BlockingCall, _tensors: &mut [Tensor]) -> Result<()> {
        Ok(())
    }

    fn all_reduce_max_flags(&self, _call: &BlockingCall, flags: u32) -> Result<u32> {
        Ok(flags)
    }

    fn broadcast(&self, _call: &BlockingCall, _t: &mut Tensor, root: u32) -> Result<()> {
        checked_root(1, root)?;
        Ok(())
    }

    fn barrier(&self, _call: &BlockingCall) -> Result<()> {
        Ok(())
    }

    fn rank(&self) -> u32 {
        0
    }

    fn world(&self) -> u32 {
        1
    }

    fn bind_agreement(&self, _digest: String) -> Result<()> {
        // A gang of one has no peer to disagree with.
        Ok(())
    }
}
