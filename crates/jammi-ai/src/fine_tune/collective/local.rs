//! The in-process collective: ranks are threads of one process, one per
//! device of one host.
//!
//! Every operation is a rendezvous — each rank deposits its contribution,
//! the last arrival publishes the rank-ordered vector of contributions, and
//! each rank then computes the SAME result from the SAME rank-ordered inputs.
//! Two properties follow from that shape rather than from care taken at each
//! call site:
//!
//! - **Determinism.** A gather is a rank-ordered concatenation and a sum is a
//!   rank-ordered fold on ONE device (rank 0's), so the result is a pure
//!   function of the ranks' inputs. Two runs of the same gang over the same
//!   inputs produce byte-identical outputs, and each equals the serial
//!   reference bit-for-bit.
//! - **Lockstep.** The rendezvous admits exactly one contribution per rank
//!   per round and will not start a round until the previous one has been
//!   consumed by every rank, so a rank that reaches a collective a different
//!   number of times than its peers cannot silently pair up with the wrong
//!   round: it waits, and the gang's deadline expires with a typed error
//!   naming the round.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use jammi_db::error::{JammiError, Result};

use candle_core::{Device, Tensor};

use super::{checked_gather_counts, checked_root, Collective};

/// How long a rank waits at a rendezvous before the round is declared failed.
///
/// The same 120 seconds `[worker] rank_timeout_secs` defaults to. A deadline
/// rather than an unbounded wait because the alternative to a typed error
/// here is a hung process with no statement of what it is waiting for: an
/// in-process gang whose peer thread panicked, exited early, or took a
/// different branch would otherwise park forever.
pub const DEFAULT_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(120);

/// What one rank contributes to one round of one collective.
///
/// The kind is part of the contribution so a round in which the ranks are
/// executing DIFFERENT collectives is a typed error rather than a
/// nonsensical result: a mismatch means the gang has left lockstep.
#[derive(Clone, Debug)]
enum Contribution {
    /// [`Collective::all_gather`]: this rank's slice.
    Gather(Tensor),
    /// [`Collective::all_reduce_sum`]: this rank's tensors, canonical order.
    ReduceSum(Vec<Tensor>),
    /// [`Collective::all_reduce_max_flags`]: this rank's control word.
    MaxFlags(u32),
    /// [`Collective::broadcast`]: the root's tensor; `None` off the root.
    Broadcast(Option<Tensor>),
    /// [`Collective::barrier`]: no payload.
    Barrier,
}

impl Contribution {
    /// The operation name, for the error a kind mismatch raises.
    fn kind(&self) -> &'static str {
        match self {
            Self::Gather(_) => "all_gather",
            Self::ReduceSum(_) => "all_reduce_sum",
            Self::MaxFlags(_) => "all_reduce_max_flags",
            Self::Broadcast(_) => "broadcast",
            Self::Barrier => "barrier",
        }
    }
}

/// One rendezvous round.
#[derive(Debug)]
struct Round {
    /// Rank-indexed contributions of the round being assembled.
    slots: Vec<Option<Contribution>>,
    /// How many ranks have deposited into `slots`.
    arrived: usize,
    /// The completed round, shared by every rank until all have taken it.
    /// `Some` blocks the NEXT round from starting, which is what keeps a
    /// fast rank from overtaking a slow one by a whole collective.
    published: Option<Arc<Vec<Contribution>>>,
    /// How many ranks have taken `published`.
    taken: usize,
}

/// The state a [`LocalGang`]'s ranks share.
#[derive(Debug)]
struct Shared {
    world: usize,
    devices: Vec<Device>,
    timeout: Duration,
    round: Mutex<Round>,
    signal: Condvar,
}

impl Shared {
    /// Deposit this rank's contribution and return every rank's, in rank
    /// order, once the round completes.
    fn exchange(&self, rank: usize, contribution: Contribution) -> Result<Arc<Vec<Contribution>>> {
        let kind = contribution.kind();
        let deadline = Instant::now() + self.timeout;
        let mut round = self.round.lock().unwrap_or_else(PoisonError::into_inner);

        // The previous round is still being read by a slower peer: this rank
        // is a whole collective ahead, so it waits here rather than
        // overwriting a slot the peer has not read.
        while round.published.is_some() {
            round = self.wait(round, deadline, kind, "the previous round to be consumed")?;
        }

        if round.slots[rank].is_some() {
            return Err(JammiError::FineTune(format!(
                "{kind}: rank {rank} entered the same round twice — the gang is not in lockstep"
            )));
        }
        round.slots[rank] = Some(contribution);
        round.arrived += 1;

        if round.arrived == self.world {
            let contributions: Vec<Contribution> = round
                .slots
                .iter_mut()
                .map(|slot| slot.take().expect("every rank deposited into this round"))
                .collect();
            round.published = Some(Arc::new(contributions));
            round.arrived = 0;
            round.taken = 0;
            self.signal.notify_all();
        } else {
            while round.published.is_none() {
                round = self.wait(round, deadline, kind, "every peer to arrive")?;
            }
        }

        let published = Arc::clone(
            round
                .published
                .as_ref()
                .expect("the round is published on every path that reaches here"),
        );
        round.taken += 1;
        if round.taken == self.world {
            round.published = None;
            self.signal.notify_all();
        }
        drop(round);

        // Every rank ran the same collective, or the gang has left lockstep
        // and any result computed from these contributions would be
        // meaningless.
        for (peer, contribution) in published.iter().enumerate() {
            if contribution.kind() != kind {
                return Err(JammiError::FineTune(format!(
                    "{kind}: rank {peer} is at {} instead — the gang has left lockstep",
                    contribution.kind()
                )));
            }
        }
        Ok(published)
    }

    /// One bounded wait on the round's condition variable.
    fn wait<'a>(
        &'a self,
        round: std::sync::MutexGuard<'a, Round>,
        deadline: Instant,
        kind: &str,
        waiting_for: &str,
    ) -> Result<std::sync::MutexGuard<'a, Round>> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(JammiError::FineTune(format!(
                "{kind}: timed out after {:?} waiting for {waiting_for}",
                self.timeout
            )));
        }
        let (round, wait) = self
            .signal
            .wait_timeout(round, remaining)
            .unwrap_or_else(PoisonError::into_inner);
        if wait.timed_out() && Instant::now() >= deadline {
            return Err(JammiError::FineTune(format!(
                "{kind}: timed out after {:?} waiting for {waiting_for}",
                self.timeout
            )));
        }
        Ok(round)
    }
}

/// A gang of in-process ranks over the devices of one host.
///
/// Build one, take one [`Local`] per rank with [`Self::rank`], and drive each
/// on its own thread. The gang is the shared rendezvous; the [`Local`] values
/// are the per-rank handles the trainer holds.
#[derive(Debug)]
pub struct LocalGang {
    shared: Arc<Shared>,
}

impl LocalGang {
    /// A gang with one rank per entry of `devices`, in that order: rank `r`
    /// trains on `devices[r]` and `devices[0]` is the reduction device.
    ///
    /// An empty device list is refused — a gang of no ranks has no rank 0 to
    /// reduce on, and every collective over it would be a division by an
    /// empty fold.
    pub fn new(devices: Vec<Device>) -> Result<Self> {
        Self::with_timeout(devices, DEFAULT_RENDEZVOUS_TIMEOUT)
    }

    /// [`Self::new`] with an explicit rendezvous deadline, for a caller that
    /// knows its own bound (a test that asserts the deadline fires, a
    /// deployment whose `[worker] rank_timeout_secs` differs from the
    /// default).
    pub fn with_timeout(devices: Vec<Device>, timeout: Duration) -> Result<Self> {
        if devices.is_empty() {
            return Err(JammiError::FineTune(
                "a local gang needs at least one device: rank 0's device is the one every \
                 reduction folds on"
                    .into(),
            ));
        }
        if timeout.is_zero() {
            return Err(JammiError::FineTune(
                "a local gang's rendezvous timeout must be > 0 (a zero deadline expires \
                 before any peer can arrive)"
                    .into(),
            ));
        }
        let world = devices.len();
        Ok(Self {
            shared: Arc::new(Shared {
                world,
                devices,
                timeout,
                round: Mutex::new(Round {
                    slots: (0..world).map(|_| None).collect(),
                    arrived: 0,
                    published: None,
                    taken: 0,
                }),
                signal: Condvar::new(),
            }),
        })
    }

    /// How many ranks this gang has.
    pub fn world(&self) -> u32 {
        self.shared.world as u32
    }

    /// The handle for `rank`, or a typed error when `rank` is not a rank of
    /// this gang.
    pub fn rank(&self, rank: u32) -> Result<Local> {
        if rank as usize >= self.shared.world {
            return Err(JammiError::FineTune(format!(
                "rank {rank} is not a rank of a gang of {}",
                self.shared.world
            )));
        }
        Ok(Local {
            rank,
            shared: Arc::clone(&self.shared),
        })
    }
}

/// One rank's handle on a [`LocalGang`].
#[derive(Debug)]
pub struct Local {
    rank: u32,
    shared: Arc<Shared>,
}

impl Local {
    /// The device this rank trains on.
    pub fn device(&self) -> &Device {
        &self.shared.devices[self.rank as usize]
    }

    /// Rank 0's device: the one every reduction folds on, so the fold's
    /// arithmetic happens in one place rather than once per rank.
    fn reduce_device(&self) -> &Device {
        &self.shared.devices[0]
    }
}

impl Collective for Local {
    fn all_gather(&self, local: &Tensor, counts: &[usize]) -> Result<Tensor> {
        let total = checked_gather_counts(self.rank, self.world(), local, counts)?;
        let contributions = self
            .shared
            .exchange(self.rank as usize, Contribution::Gather(local.clone()))?;

        // Rank order, this rank's device, and only this rank's own slot
        // attached to the graph: the remote slots are values, not a path a
        // gradient can take to a peer's parameters.
        let mut slices: VecDeque<Tensor> = VecDeque::with_capacity(contributions.len());
        for (peer, contribution) in contributions.iter().enumerate() {
            let Contribution::Gather(tensor) = contribution else {
                unreachable!("exchange checked every contribution's kind");
            };
            let rows = tensor.dims().first().copied().unwrap_or(0);
            if rows != counts[peer] {
                return Err(JammiError::FineTune(format!(
                    "all_gather: rank {peer} contributed {rows} rows where the partition rule \
                     says {} — the ranks disagree about the partition",
                    counts[peer]
                )));
            }
            if rows == 0 {
                continue;
            }
            let slice = if peer == self.rank as usize {
                tensor.clone()
            } else {
                tensor
                    .to_device(self.device())
                    .map_err(|e| JammiError::FineTune(format!("all_gather: to_device: {e}")))?
                    .detach()
            };
            slices.push_back(slice);
        }

        if slices.is_empty() {
            // Every rank contributed zero rows: the gather is this rank's own
            // (empty) tensor, which already carries the trailing shape and
            // dtype the caller expects.
            return Ok(local.clone());
        }
        let slices: Vec<Tensor> = slices.into();
        let gathered = if slices.len() == 1 {
            slices.into_iter().next().expect("length checked")
        } else {
            Tensor::cat(&slices, 0)
                .map_err(|e| JammiError::FineTune(format!("all_gather: cat: {e}")))?
        };
        let rows = gathered.dims().first().copied().unwrap_or(0);
        if rows != total {
            return Err(JammiError::FineTune(format!(
                "all_gather: gathered {rows} rows where the counts sum to {total}"
            )));
        }
        Ok(gathered)
    }

    fn all_reduce_sum(&self, tensors: &mut [Tensor]) -> Result<()> {
        let contributions = self.shared.exchange(
            self.rank as usize,
            Contribution::ReduceSum(tensors.to_vec()),
        )?;

        for (peer, contribution) in contributions.iter().enumerate() {
            let Contribution::ReduceSum(peer_tensors) = contribution else {
                unreachable!("exchange checked every contribution's kind");
            };
            if peer_tensors.len() != tensors.len() {
                return Err(JammiError::FineTune(format!(
                    "all_reduce_sum: rank {peer} reduced {} tensors where rank {} reduced {} \
                     — the canonical trainable-variable order must be identical on every rank",
                    peer_tensors.len(),
                    self.rank,
                    tensors.len()
                )));
            }
        }

        for (index, slot) in tensors.iter_mut().enumerate() {
            // The fold runs on rank 0's device, in rank order, so the sum is
            // one fixed sequence of additions rather than one per rank.
            let mut sum: Option<Tensor> = None;
            for (peer, contribution) in contributions.iter().enumerate() {
                let Contribution::ReduceSum(peer_tensors) = contribution else {
                    unreachable!("exchange checked every contribution's kind");
                };
                let term = peer_tensors[index]
                    .to_device(self.reduce_device())
                    .map_err(|e| JammiError::FineTune(format!("all_reduce_sum: to_device: {e}")))?;
                sum = Some(match sum {
                    None => term,
                    Some(acc) => acc.add(&term).map_err(|e| {
                        JammiError::FineTune(format!(
                            "all_reduce_sum: adding rank {peer}'s tensor {index}: {e}"
                        ))
                    })?,
                });
            }
            let sum = sum.expect("a gang has at least one rank");
            *slot = sum
                .to_device(self.device())
                .map_err(|e| JammiError::FineTune(format!("all_reduce_sum: to_device: {e}")))?;
        }
        Ok(())
    }

    fn all_reduce_max_flags(&self, flags: u32) -> Result<u32> {
        let contributions = self
            .shared
            .exchange(self.rank as usize, Contribution::MaxFlags(flags))?;
        let mut max = 0u32;
        for contribution in contributions.iter() {
            let Contribution::MaxFlags(peer_flags) = contribution else {
                unreachable!("exchange checked every contribution's kind");
            };
            max = max.max(*peer_flags);
        }
        Ok(max)
    }

    fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()> {
        let root_index = checked_root(self.world(), root)?;
        let payload = (self.rank == root).then(|| t.clone());
        let contributions = self
            .shared
            .exchange(self.rank as usize, Contribution::Broadcast(payload))?;
        let Contribution::Broadcast(from_root) = &contributions[root_index] else {
            unreachable!("exchange checked every contribution's kind");
        };
        let from_root = from_root.as_ref().ok_or_else(|| {
            JammiError::FineTune(format!(
                "broadcast: rank {root} entered the round as a non-root — the ranks disagree \
                 about which of them is the root"
            ))
        })?;
        *t = from_root
            .to_device(self.device())
            .map_err(|e| JammiError::FineTune(format!("broadcast: to_device: {e}")))?
            .detach();
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        self.shared
            .exchange(self.rank as usize, Contribution::Barrier)?;
        Ok(())
    }

    fn rank(&self) -> u32 {
        self.rank
    }

    fn world(&self) -> u32 {
        self.shared.world as u32
    }
}
