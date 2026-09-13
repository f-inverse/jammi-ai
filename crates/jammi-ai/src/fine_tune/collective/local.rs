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
//! - **A failure is the gang's, and it is permanent.** A rank cannot fail a
//!   collective alone: its peers are mid-round with it. So every error exit
//!   records a FAULT on the gang and abandons the round in progress —
//!   dropping every deposited contribution, including the failing rank's own
//!   — and every later entry on every rank refuses with a typed error that
//!   quotes that fault. Without this, a peer arriving after a deadline would
//!   complete the already-failed round over a timed-out rank's stale
//!   contribution and be handed `Ok`, and the two ranks would leave one round
//!   with opposite verdicts.

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
    /// Which round this is. Incremented every time the round in progress
    /// ends — by publishing, or by being abandoned after a failure — so the
    /// round a contribution was deposited into is nameable after the fact.
    ///
    /// A slot carries the generation it was deposited at, which makes
    /// "assembling a round out of contributions from two different rounds"
    /// a checkable condition rather than an assumption: the publishing rank
    /// refuses a round whose slots do not all name the round it is
    /// completing, so a contribution that outlived its round can never be
    /// folded into a result.
    generation: u64,
    /// Rank-indexed contributions of the round being assembled, each tagged
    /// with the [`Self::generation`] it was deposited at.
    slots: Vec<Option<(u64, Contribution)>>,
    /// How many ranks have deposited into `slots`.
    arrived: usize,
    /// The completed round, shared by every rank until all have taken it.
    /// `Some` blocks the NEXT round from starting, which is what keeps a
    /// fast rank from overtaking a slow one by a whole collective. The `u64`
    /// is the generation the contributions were assembled from.
    published: Option<(u64, Arc<Vec<Contribution>>)>,
    /// How many ranks have taken `published`.
    taken: usize,
}

impl Round {
    /// Drop the round in progress and move to the next generation.
    ///
    /// Every deposited contribution goes, including the abandoning rank's
    /// own: a contribution whose owner has been told the collective failed
    /// must not be readable by anyone. `published` goes too — a vector no
    /// rank is going to finish taking would otherwise block the gang's next
    /// round forever, and the fault recorded alongside this call is what
    /// every rank reads instead.
    ///
    /// A round with nothing in it has nothing to abandon, and the generation
    /// stays put: it counts rounds that ENDED, so a second failure reported
    /// against an already-abandoned round must not make it look as though
    /// another round came and went.
    fn abandon(&mut self) {
        if self.arrived == 0 && self.published.is_none() {
            return;
        }
        for slot in &mut self.slots {
            *slot = None;
        }
        self.arrived = 0;
        self.published = None;
        self.taken = 0;
        self.generation += 1;
    }
}

/// The state a [`LocalGang`]'s ranks share.
#[derive(Debug)]
struct Shared {
    world: usize,
    devices: Vec<Device>,
    timeout: Duration,
    round: Mutex<Round>,
    /// The first failure any rank reported, and the gang's permanent state
    /// from then on. Written and read only while the [`Self::round`] guard
    /// is held (lock order: `round` then `fault`, never the reverse), so a
    /// rank cannot deposit into a round another rank is abandoning.
    fault: Mutex<Option<String>>,
    signal: Condvar,
}

impl Shared {
    /// Deposit this rank's contribution and return every rank's, in rank
    /// order, once the round completes.
    fn exchange(&self, rank: usize, contribution: Contribution) -> Result<Arc<Vec<Contribution>>> {
        let kind = contribution.kind();
        let deadline = Instant::now() + self.timeout;
        let mut round = self.round.lock().unwrap_or_else(PoisonError::into_inner);

        // Under the round's own lock, so this gate and a peer's abandonment
        // of the round cannot interleave: a rank either sees the fault, or
        // deposits into a round no one is abandoning.
        if let Some(refusal) = self.refusal(kind) {
            return Err(refusal);
        }

        // The previous round is still being read by a slower peer: this rank
        // is a whole collective ahead, so it waits here rather than
        // overwriting a slot the peer has not read.
        loop {
            // The predicate is re-checked before every wait and never after
            // the wait's own return: a wake that satisfied the predicate at
            // the same instant the deadline passed proceeds, so a round the
            // gang DID complete is never reported as a timeout on one rank
            // and a success on another.
            if round.published.is_none() {
                break;
            }
            round = self.wait(round, deadline, kind, "the previous round to be consumed")?;
        }

        if round.slots[rank].is_some() {
            let reason = format!(
                "{kind}: rank {rank} entered the same round twice — the gang is not in lockstep"
            );
            return Err(self.fail(&mut round, reason));
        }
        let generation = round.generation;
        round.slots[rank] = Some((generation, contribution));
        round.arrived += 1;

        if round.arrived == self.world {
            // Every slot must name the round being completed. An abandoned
            // round drops its slots, so in this module a contribution cannot
            // outlive its round — this is the check that says so rather than
            // assuming it, and it is what refuses the `Ok` a late peer would
            // otherwise be handed over a departed rank's stale contribution.
            let mut contributions = Vec::with_capacity(self.world);
            let mut superseded: Option<(usize, u64)> = None;
            for (peer, slot) in round.slots.iter_mut().enumerate() {
                let (deposited, contribution) =
                    slot.take().expect("every rank deposited into this round");
                if deposited != generation {
                    superseded = Some((peer, deposited));
                }
                contributions.push(contribution);
            }
            if let Some((peer, deposited)) = superseded {
                let reason = format!(
                    "{kind}: rank {peer}'s contribution is from round {deposited}, not the \
                     round {generation} this rank is completing — it outlived its own round, \
                     so folding it would produce a result no rank agreed to"
                );
                return Err(self.fail(&mut round, reason));
            }
            round.published = Some((generation, Arc::new(contributions)));
            round.generation += 1;
            round.arrived = 0;
            round.taken = 0;
            self.signal.notify_all();
        } else {
            loop {
                if round.published.is_some() {
                    break;
                }
                round = self.wait(round, deadline, kind, "every peer to arrive")?;
            }
        }

        let (published_generation, published) = round
            .published
            .as_ref()
            .map(|(generation, contributions)| (*generation, Arc::clone(contributions)))
            .expect("the round is published on every path that reaches here");
        if published_generation != generation {
            let reason = format!(
                "{kind}: rank {rank} deposited into round {generation} but round \
                 {published_generation} is the one published — the round this rank \
                 contributed to was superseded"
            );
            return Err(self.fail(&mut round, reason));
        }
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
                let reason = format!(
                    "{kind}: rank {peer} is at {} instead — the gang has left lockstep",
                    contribution.kind()
                );
                self.fail_detached(reason.clone());
                return Err(JammiError::FineTune(reason));
            }
        }
        Ok(published)
    }

    /// One bounded wait on the round's condition variable.
    ///
    /// A wake finds the gang faulted, the deadline passed, or neither; only
    /// the third case returns a guard to the caller's loop for another look
    /// at its predicate.
    fn wait<'a>(
        &'a self,
        mut round: std::sync::MutexGuard<'a, Round>,
        deadline: Instant,
        kind: &str,
        waiting_for: &str,
    ) -> Result<std::sync::MutexGuard<'a, Round>> {
        if let Some(refusal) = self.refusal(kind) {
            return Err(refusal);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            let reason = format!(
                "{kind}: timed out after {:?} waiting for {waiting_for}",
                self.timeout
            );
            return Err(self.fail(&mut round, reason));
        }
        let (round, _) = self
            .signal
            .wait_timeout(round, remaining)
            .unwrap_or_else(PoisonError::into_inner);
        Ok(round)
    }

    /// Record `reason` as the gang's fault, abandon the round in progress,
    /// wake every parked rank, and hand the caller the error to return.
    ///
    /// The two halves are one step: a rank that is told its collective
    /// failed must not leave a contribution behind for a peer to fold, and
    /// the peers must be able to find out why they are being refused.
    fn fail(&self, round: &mut Round, reason: String) -> JammiError {
        self.record(reason.clone());
        round.abandon();
        self.signal.notify_all();
        JammiError::FineTune(reason)
    }

    /// [`Self::fail`] for a failure raised outside the round's lock — the
    /// domain checks a rank runs on its own inputs before the rendezvous,
    /// and the ones it runs on the published contributions after it.
    fn fail_detached(&self, reason: String) {
        let mut round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        self.fail(&mut round, reason);
    }

    /// The gang's fault as the error an entering rank gets, or `None` while
    /// the gang is healthy.
    ///
    /// Every caller holds the [`Self::round`] guard, which is what orders
    /// this read against [`Self::fail`]'s write. The fault's own words are
    /// quoted so a refused rank learns which collective failed and how,
    /// rather than only that something did.
    fn refusal(&self, kind: &str) -> Option<JammiError> {
        let fault = self.fault.lock().unwrap_or_else(PoisonError::into_inner);
        fault.as_ref().map(|reason| {
            JammiError::FineTune(format!(
                "{kind}: the gang has already failed — {reason}. A collective after a failure \
                 is refused rather than run: the peers it would rendezvous with are not \
                 coming, and any result assembled from what they left behind is stale."
            ))
        })
    }

    /// Record the gang's FIRST failure; later ones leave it alone. The first
    /// one is the fact that explains all the rest, and a fault that kept
    /// being overwritten would end up naming a consequence instead of a
    /// cause.
    fn record(&self, reason: String) {
        let mut fault = self.fault.lock().unwrap_or_else(PoisonError::into_inner);
        if fault.is_none() {
            *fault = Some(reason);
        }
    }

    /// Refuse at the very start of a collective, before the rank touches its
    /// own inputs, when the gang has already failed.
    fn check_entry(&self, kind: &str) -> Result<()> {
        let _round = self.round.lock().unwrap_or_else(PoisonError::into_inner);
        match self.refusal(kind) {
            Some(refusal) => Err(refusal),
            None => Ok(()),
        }
    }
}

/// The words to record as a gang's fault for `error`.
///
/// A [`JammiError::FineTune`]'s own message rather than its `Display`, so a
/// fault quoted inside a later refusal does not carry a second copy of the
/// "Fine-tune error" prefix.
fn reason_of(error: &JammiError) -> String {
    match error {
        JammiError::FineTune(message) => message.clone(),
        other => other.to_string(),
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
                    generation: 0,
                    slots: (0..world).map(|_| None).collect(),
                    arrived: 0,
                    published: None,
                    taken: 0,
                }),
                fault: Mutex::new(None),
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

    /// Run one collective inside the gang's fault state.
    ///
    /// Refuse before touching anything if the gang has already failed, and
    /// make any failure of `body` the GANG's failure. A rank never fails a
    /// collective alone — its peers are in the same round — so recording the
    /// failure here is what turns a peer's otherwise unbounded wait into a
    /// prompt typed error, and what stops a peer from completing the round
    /// this rank has just walked away from.
    ///
    /// This wraps the whole operation, not just the rendezvous, so the
    /// domain checks a rank runs on its own inputs (a count vector that does
    /// not describe the gang, a root outside it) and the ones it runs on the
    /// published contributions (a peer whose row count contradicts the
    /// partition rule) fault the gang too: each of those is a disagreement
    /// this rank can see and its peers cannot.
    fn guarded<T>(&self, kind: &str, body: impl FnOnce() -> Result<T>) -> Result<T> {
        self.shared.check_entry(kind)?;
        match body() {
            Ok(value) => Ok(value),
            Err(error) => {
                self.shared.fail_detached(reason_of(&error));
                Err(error)
            }
        }
    }
}

impl Collective for Local {
    fn all_gather(&self, local: &Tensor, counts: &[usize]) -> Result<Tensor> {
        self.guarded("all_gather", || {
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
        })
    }

    fn all_reduce_sum(&self, tensors: &mut [Tensor]) -> Result<()> {
        self.guarded("all_reduce_sum", || {
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
                        .map_err(|e| {
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
                let sum = sum.expect("a gang has at least one rank");
                *slot = sum
                    .to_device(self.device())
                    .map_err(|e| JammiError::FineTune(format!("all_reduce_sum: to_device: {e}")))?;
            }
            Ok(())
        })
    }

    fn all_reduce_max_flags(&self, flags: u32) -> Result<u32> {
        self.guarded("all_reduce_max_flags", || {
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
        })
    }

    fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()> {
        self.guarded("broadcast", || {
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
        })
    }

    fn barrier(&self) -> Result<()> {
        self.guarded("barrier", || {
            self.shared
                .exchange(self.rank as usize, Contribution::Barrier)?;
            Ok(())
        })
    }

    fn rank(&self) -> u32 {
        self.rank
    }

    fn world(&self) -> u32 {
        self.shared.world as u32
    }
}

/// White-box oracles for the rendezvous state the public API cannot reach on
/// its own.
#[cfg(test)]
mod rendezvous_state_tests {
    use super::*;

    /// The round a rank walks away from holds NOTHING afterwards.
    ///
    /// This is the state half of the failure: the fault tells a late peer to
    /// stop, and the empty round means that even a peer that somehow got
    /// past the fault would have nothing of the departed rank's to fold. A
    /// timeout exit that left `slots[rank]` filled and `arrived`
    /// incremented is exactly what let a late peer complete a failed round.
    #[test]
    fn a_timed_out_round_keeps_no_contribution_and_no_arrival() {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_millis(50)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        rank0
            .all_reduce_max_flags(0b1011)
            .expect_err("rank 1 never arrives inside the deadline");

        let round = gang.shared.round.lock().expect("round");
        assert!(
            round.slots.iter().all(Option::is_none),
            "the abandoned round still holds a contribution: {:?}",
            round.slots
        );
        assert_eq!(
            round.arrived, 0,
            "the abandoned round still counts arrivals"
        );
        assert!(round.published.is_none());
        assert_eq!(
            round.generation, 1,
            "abandoning the round moves the gang past it, so nothing deposited into it can be \
             mistaken for part of the next one"
        );
        assert!(
            gang.shared
                .fault
                .lock()
                .expect("fault")
                .as_deref()
                .is_some_and(|reason| reason.contains("timed out")),
            "the gang must carry the fault that explains every later refusal"
        );
    }

    /// A round assembled out of contributions from two different rounds is
    /// refused by the rank that would complete it.
    ///
    /// **This is not reachable through the public API**, and the attempt to
    /// reach it is this test: abandoning a round drops every slot
    /// (`a_timed_out_round_keeps_no_contribution_and_no_arrival`) and the
    /// fault refuses every later entry
    /// (`every_collective_after_a_fault_errs_promptly_on_every_rank`), so
    /// there is no public sequence that leaves a stale contribution in a
    /// live round. The test reaches the condition by moving the round's
    /// generation on under the lock — the one thing an abandonment does
    /// that a stale-slot bug would not — and asserts the second, independent
    /// defence holds on its own: the completing rank refuses rather than
    /// folding a contribution that outlived its round.
    #[test]
    fn a_round_holding_a_superseded_contribution_is_refused_by_the_rank_completing_it() {
        let gang =
            LocalGang::with_timeout(vec![Device::Cpu; 2], Duration::from_secs(5)).expect("gang");
        let rank0 = gang.rank(0).expect("rank 0");
        let rank1 = gang.rank(1).expect("rank 1");

        std::thread::scope(|scope| {
            let parked = scope.spawn(move || rank0.barrier());

            // Wait until rank 0's contribution is in the round, then move the
            // round on without taking that contribution with it.
            loop {
                let mut round = gang.shared.round.lock().expect("round");
                if round.arrived == 1 {
                    round.generation += 1;
                    break;
                }
                drop(round);
                std::thread::yield_now();
            }

            let completing = rank1
                .barrier()
                .expect_err("rank 0's contribution is from the round before this one");
            let message = completing.to_string();
            assert!(
                message.contains("rank 0's contribution is from round 0")
                    && message.contains("outlived its own round"),
                "unexpected message: {message}"
            );

            // And the departed rank is told the gang failed, rather than
            // waiting out its own deadline.
            let stale = parked
                .join()
                .expect("rank 0 thread")
                .expect_err("the round rank 0 deposited into was never completed");
            assert!(
                stale.to_string().contains("outlived its own round"),
                "unexpected message: {stale}"
            );
        });
    }
}
