//! One round's arithmetic, shared by every arm and every transport: what a
//! rank contributes, the rank-ordered fold, and how each verb hands the
//! folded result back to its caller.
//!
//! The control planes ([`Local`](super::Local)'s rendezvous,
//! [`Peer`](super::Peer)'s two-phase round) decide WHEN a round's
//! contributions may be folded — once every rank's descriptor agrees — and
//! the transport decides HOW the contributions' bytes reached the folding
//! rank. Neither decides WHAT the fold computes: that is this module, once,
//! so a round folded by `Local` on rank 0's device, by `Peer`'s coordinator,
//! or by every rank of a device-transport gang on its own device is the same
//! sequence of operations over the same rank-ordered inputs, and the result
//! bytes are identical across arms and transports on one device kind.

use candle_core::{Device, Tensor};
use jammi_db::error::{JammiError, Result};

use super::{Descriptor, Verb};

/// What one rank contributes to one round.
#[derive(Clone, Debug)]
pub(crate) enum Contribution {
    /// [`Collective::all_gather`](super::Collective::all_gather): this rank's
    /// slice.
    Gather(Tensor),
    /// [`Collective::all_reduce_sum`](super::Collective::all_reduce_sum):
    /// this rank's tensors, canonical order.
    ReduceSum(Vec<Tensor>),
    /// [`Collective::all_reduce_max_flags`](super::Collective::all_reduce_max_flags):
    /// this rank's control word.
    MaxFlags(u32),
    /// [`Collective::broadcast`](super::Collective::broadcast): the root's
    /// tensor; `None` off the root.
    Broadcast(Option<Tensor>),
    /// [`Collective::barrier`](super::Collective::barrier): no payload.
    Barrier,
}

impl Contribution {
    /// The operation — [`Descriptor::verb`], so a round whose ranks run
    /// different collectives is caught by the descriptor agreement like every
    /// other disagreement.
    pub(crate) fn verb(&self) -> Verb {
        match self {
            Self::Gather(_) => Verb::AllGather,
            Self::ReduceSum(_) => Verb::AllReduceSum,
            Self::MaxFlags(_) => Verb::AllReduceMaxFlags,
            Self::Broadcast(_) => Verb::Broadcast,
            Self::Barrier => Verb::Barrier,
        }
    }

    /// The tensors this contribution carries, in the verb's order.
    pub(crate) fn tensors(&self) -> Vec<&Tensor> {
        match self {
            Self::Gather(t) => vec![t],
            Self::ReduceSum(ts) => ts.iter().collect(),
            Self::MaxFlags(_) | Self::Barrier => Vec::new(),
            Self::Broadcast(t) => t.iter().collect(),
        }
    }

    /// The control word this contribution carries; `0` for every verb but
    /// `all_reduce_max_flags`.
    pub(crate) fn flags(&self) -> u32 {
        match self {
            Self::MaxFlags(flags) => *flags,
            _ => 0,
        }
    }
}

/// A folded round result: the verb's tensors, and the control word.
#[derive(Debug)]
pub(crate) struct Folded {
    pub(crate) tensors: Vec<Tensor>,
    pub(crate) flags: u32,
}

/// The fold, on `device`, in rank order: `Tensor::cat` of the non-empty
/// slices for a gather, a left fold of `acc.add(term)` for a sum, `max` for
/// the control word, the root's tensor for a broadcast.
///
/// `contributions[r]` is rank `r`'s, every one under the agreed `descriptor`
/// — the caller's control plane proved that before calling, so a
/// contribution of another verb is unreachable here.
pub(crate) fn fold(
    verb: Verb,
    descriptor: &Descriptor,
    contributions: &[Contribution],
    device: &Device,
) -> Result<Folded> {
    match verb {
        Verb::AllGather => {
            let mut slices = Vec::with_capacity(contributions.len());
            for contribution in contributions {
                let Contribution::Gather(tensor) = contribution else {
                    unreachable!("every contribution of a round carries the round's verb");
                };
                if tensor.dims().first().copied().unwrap_or(0) == 0 {
                    continue;
                }
                slices.push(
                    tensor
                        .to_device(device)
                        .map_err(|e| JammiError::FineTune(format!("all_gather: to_device: {e}")))?,
                );
            }
            let gathered = if slices.is_empty() {
                // Every rank contributed zero rows: rank 0's own (empty)
                // tensor already carries the agreed trailing shape and dtype.
                let Contribution::Gather(own) = &contributions[0] else {
                    unreachable!("rank 0's contribution carries the round's verb");
                };
                own.clone()
            } else if slices.len() == 1 {
                slices.into_iter().next().expect("length checked")
            } else {
                Tensor::cat(&slices, 0)
                    .map_err(|e| JammiError::FineTune(format!("all_gather: cat: {e}")))?
            };
            Ok(Folded {
                tensors: vec![gathered],
                flags: 0,
            })
        }
        Verb::AllReduceSum => {
            let count = descriptor.tensors.len();
            let mut sums = Vec::with_capacity(count);
            for index in 0..count {
                let mut sum: Option<Tensor> = None;
                for (peer, contribution) in contributions.iter().enumerate() {
                    let Contribution::ReduceSum(tensors) = contribution else {
                        unreachable!("every contribution of a round carries the round's verb");
                    };
                    let term = tensors[index].to_device(device).map_err(|e| {
                        JammiError::FineTune(format!("all_reduce_sum: to_device: {e}"))
                    })?;
                    sum = Some(match sum {
                        None => term,
                        Some(acc) => acc.add(&term).map_err(|e| {
                            JammiError::FineTune(format!(
                                "all_reduce_sum: adding rank {peer}'s tensor {index}: {e}"
                            ))
                        })?,
                    });
                }
                sums.push(sum.expect("a gang has at least one rank"));
            }
            Ok(Folded {
                tensors: sums,
                flags: 0,
            })
        }
        Verb::AllReduceMaxFlags => Ok(Folded {
            tensors: Vec::new(),
            flags: contributions
                .iter()
                .map(Contribution::flags)
                .max()
                .unwrap_or(0),
        }),
        Verb::Broadcast => {
            let root = descriptor
                .root
                .expect("a broadcast descriptor names its root") as usize;
            let Contribution::Broadcast(from_root) = &contributions[root] else {
                unreachable!("every contribution of a round carries the round's verb");
            };
            let from_root = from_root
                .as_ref()
                .ok_or_else(|| {
                    JammiError::FineTune(format!(
                        "broadcast: rank {root} is the agreed root but contributed no tensor"
                    ))
                })?
                .to_device(device)
                .map_err(|e| JammiError::FineTune(format!("broadcast: to_device: {e}")))?;
            Ok(Folded {
                tensors: vec![from_root],
                flags: 0,
            })
        }
        Verb::Barrier => Ok(Folded {
            tensors: Vec::new(),
            flags: 0,
        }),
    }
}

/// The one tensor of a one-tensor result (a gather's concatenation, a
/// broadcast's root tensor).
pub(crate) fn single(mut tensors: Vec<Tensor>) -> Tensor {
    tensors
        .pop()
        .expect("a one-signature result carries exactly one tensor")
}

/// Hand a folded gather back to rank `rank`, on `device`: the row count must
/// be the counts' sum, and only this rank's own slot carries a gradient — the
/// remote slots are the published values, detached, and the caller's own
/// `local` is spliced back in at its rows, byte-equal to what was folded
/// there.
pub(crate) fn apply_gather(
    folded: Folded,
    local: &Tensor,
    counts: &[usize],
    rank: usize,
    device: &Device,
) -> Result<Tensor> {
    let total: usize = counts.iter().sum();
    let own_rows = counts[rank];
    let rows_before: usize = counts[..rank].iter().sum();
    let gathered = single(folded.tensors)
        .to_device(device)
        .map_err(|e| JammiError::FineTune(format!("all_gather: to_device: {e}")))?;
    let rows = gathered.dims().first().copied().unwrap_or(0);
    if rows != total {
        return Err(JammiError::FineTune(format!(
            "all_gather: gathered {rows} rows where the counts sum to {total}"
        )));
    }
    if own_rows == 0 {
        return Ok(gathered.detach());
    }
    if rows == own_rows {
        return Ok(local.clone());
    }
    let mut parts = Vec::with_capacity(3);
    if rows_before > 0 {
        parts.push(
            gathered
                .narrow(0, 0, rows_before)
                .map_err(|e| JammiError::FineTune(format!("all_gather: narrow: {e}")))?
                .detach(),
        );
    }
    parts.push(local.clone());
    let after = rows - rows_before - own_rows;
    if after > 0 {
        parts.push(
            gathered
                .narrow(0, rows_before + own_rows, after)
                .map_err(|e| JammiError::FineTune(format!("all_gather: narrow: {e}")))?
                .detach(),
        );
    }
    Tensor::cat(&parts, 0).map_err(|e| JammiError::FineTune(format!("all_gather: cat: {e}")))
}

/// Hand a folded sum back: each slot becomes its rank-ordered sum, on
/// `device`.
pub(crate) fn apply_sum(folded: Folded, tensors: &mut [Tensor], device: &Device) -> Result<()> {
    for (slot, sum) in tensors.iter_mut().zip(folded.tensors) {
        *slot = sum
            .to_device(device)
            .map_err(|e| JammiError::FineTune(format!("all_reduce_sum: to_device: {e}")))?;
    }
    Ok(())
}

/// Hand a folded broadcast back: `t` becomes the root's tensor, on `device`,
/// detached — the root's graph is not a path a gradient takes to this rank.
pub(crate) fn apply_broadcast(folded: Folded, t: &mut Tensor, device: &Device) -> Result<()> {
    *t = single(folded.tensors)
        .to_device(device)
        .map_err(|e| JammiError::FineTune(format!("broadcast: to_device: {e}")))?
        .detach();
    Ok(())
}
