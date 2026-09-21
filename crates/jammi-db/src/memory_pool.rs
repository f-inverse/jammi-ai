//! [`ActiveSpillPool`] — the session's memory pool: `[engine] memory_limit`
//! shared fairly among the spilling consumers that are *using* it.
//!
//! # Why the session has its own pool
//!
//! DataFusion ships two bounded pools, and a plan built to run out of core
//! fits neither. `GreedyMemoryPool` is first come, first served: a sort that
//! fit in memory early holds the pool while it streams its result out, and
//! the next operator's single batch is refused at a pool size spilling would
//! have carried. `FairSpillPool` holds each spilling consumer to an equal
//! share — of the consumers *registered*, and an operator registers when its
//! stream is built, which is when the plan starts. A pipeline of hops
//! registers every sort and join of every hop on every partition at once, so
//! each share is `pool / (partitions × operators)`: a roomy pool spills, and
//! the floor below which the plan is refused grows with the plan's depth,
//! though only one hop's operators ever hold memory at a time.
//!
//! # The rule
//!
//! A consumer is **active** while it holds a non-zero reservation. A spilling
//! consumer asking to grow is held to an equal share of what the unspillable
//! consumers leave, divided among the active spilling consumers — itself
//! included, so a consumer asking for its first byte counts:
//!
//! ```text
//! share = (pool − unspillable) / max(1, active spillers ∪ {the asker})
//! grant   iff   held(asker) + additional ≤ share
//! ```
//!
//! An idle registered consumer takes no share; a lone active spiller may take
//! the whole pool; two split it evenly. An unspillable consumer is granted
//! what is actually free (`pool − unspillable − spillable`), as in both of
//! DataFusion's pools. A refusal is DataFusion's own
//! `ResourcesExhausted`, naming the pool size, which the engine types as
//! [`crate::error::JammiError::ResourcesExhausted`].
//!
//! # What the rule does not promise
//!
//! `try_grow` is synchronous and a consumer can only be asked to spill by
//! refusing *its own* next request. So when a spiller that took the pool
//! alone is joined by a second, the second is granted its half at once and
//! the first gives its excess back only when it next asks to grow (it is
//! refused, and spills) or finishes. Until then the spilling reservations
//! sum past the pool — by at most the newcomers' shares, `pool · (1/2 + 1/3 +
//! …)` over the consumers that became active since. `FairSpillPool` has the
//! same property for a consumer that registers late; refusing the newcomer
//! instead is the greedy pool's failure. [`ActiveSpillPool::reserved`]
//! reports the true sum, over-commitment included.
//!
//! Holdings are tracked per consumer, not per reservation: a reservation can
//! be split, and the halves are one consumer's memory.

use std::collections::HashMap;
use std::fmt::{self, Display, Formatter};

use datafusion::common::{resources_datafusion_err, Result};
use datafusion::execution::memory_pool::{
    human_readable_size, MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use parking_lot::Mutex;

/// See the module doc.
#[derive(Debug)]
pub struct ActiveSpillPool {
    pool_size: usize,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// Bytes held, per registered consumer id.
    held: HashMap<usize, usize>,
    /// The sum held by consumers that cannot spill.
    unspillable: usize,
    /// The sum held by consumers that can.
    spillable: usize,
    /// The spilling consumers holding a non-zero reservation.
    active_spillers: usize,
}

impl State {
    fn add(&mut self, consumer: &MemoryConsumer, bytes: usize) {
        let held = self.held.entry(consumer.id()).or_default();
        if consumer.can_spill() {
            if *held == 0 && bytes > 0 {
                self.active_spillers += 1;
            }
            self.spillable += bytes;
        } else {
            self.unspillable += bytes;
        }
        *held += bytes;
    }

    fn remove(&mut self, consumer: &MemoryConsumer, bytes: usize) {
        let held = self.held.entry(consumer.id()).or_default();
        let released = bytes.min(*held);
        *held -= released;
        if consumer.can_spill() {
            self.spillable -= released.min(self.spillable);
            if *held == 0 && released > 0 {
                self.active_spillers -= 1;
            }
        } else {
            self.unspillable -= released.min(self.unspillable);
        }
    }
}

impl ActiveSpillPool {
    /// A pool of `pool_size` bytes.
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool_size,
            state: Mutex::new(State::default()),
        }
    }
}

impl Display for ActiveSpillPool {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}(pool_size: {})",
            self.name(),
            human_readable_size(self.pool_size)
        )
    }
}

impl MemoryPool for ActiveSpillPool {
    fn name(&self) -> &str {
        "active_spill"
    }

    fn register(&self, consumer: &MemoryConsumer) {
        self.state.lock().held.insert(consumer.id(), 0);
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        let mut state = self.state.lock();
        if let Some(held) = state.held.get(&consumer.id()).copied() {
            state.remove(consumer, held);
        }
        state.held.remove(&consumer.id());
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.state.lock().add(reservation.consumer(), additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.state.lock().remove(reservation.consumer(), shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let consumer = reservation.consumer();
        let mut state = self.state.lock();
        let held = state.held.get(&consumer.id()).copied().unwrap_or(0);
        let available = if consumer.can_spill() {
            // The asker counts among the active even at its first byte.
            let sharers = state.active_spillers + usize::from(held == 0);
            let share = self.pool_size.saturating_sub(state.unspillable) / sharers.max(1);
            share.saturating_sub(held)
        } else {
            self.pool_size
                .saturating_sub(state.unspillable + state.spillable)
        };
        if additional > available {
            return Err(resources_datafusion_err!(
                "Failed to allocate additional {} for {} with {} already allocated for this \
                 consumer - {} remain available for it in the memory pool: {}",
                human_readable_size(additional),
                consumer.name(),
                human_readable_size(held),
                human_readable_size(available),
                self
            ));
        }
        state.add(consumer, additional);
        Ok(())
    }

    fn reserved(&self) -> usize {
        let state = self.state.lock();
        state.spillable + state.unspillable
    }

    fn memory_limit(&self) -> MemoryLimit {
        MemoryLimit::Finite(self.pool_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const MIB: usize = 1 << 20;

    fn pool(bytes: usize) -> Arc<dyn MemoryPool> {
        Arc::new(ActiveSpillPool::new(bytes))
    }

    fn spiller(name: &str, pool: &Arc<dyn MemoryPool>) -> MemoryReservation {
        MemoryConsumer::new(name)
            .with_can_spill(true)
            .register(pool)
    }

    #[test]
    fn an_idle_registered_consumer_takes_no_share() {
        let pool = pool(100 * MIB);
        let _idle: Vec<_> = (0..50)
            .map(|i| spiller(&format!("idle-{i}"), &pool))
            .collect();
        let busy = spiller("busy", &pool);
        busy.try_grow(100 * MIB)
            .expect("fifty idle registrations leave the whole pool to the one that asks");
    }

    #[test]
    fn a_lone_active_spiller_may_take_the_whole_pool_and_not_a_byte_more() {
        let pool = pool(100 * MIB);
        let lone = spiller("lone", &pool);
        lone.try_grow(100 * MIB).unwrap();
        let refused = lone.try_grow(1).unwrap_err().to_string();
        assert!(refused.contains("pool_size: 100.0 MB"), "{refused}");
        assert_eq!(pool.reserved(), 100 * MIB);
    }

    #[test]
    fn two_active_spillers_split_evenly() {
        let pool = pool(100 * MIB);
        let (a, b) = (spiller("a", &pool), spiller("b", &pool));
        a.try_grow(10 * MIB).unwrap();
        b.try_grow(50 * MIB).expect("b's half");
        b.try_grow(1).expect_err("not past its half");
        a.try_grow(40 * MIB).expect("a's half");
        a.try_grow(1).expect_err("not past its half");
        // One gives everything back: the other is alone again.
        drop(a);
        b.try_grow(50 * MIB).expect("the whole pool, alone");
    }

    #[test]
    fn a_consumer_is_active_by_what_it_holds_across_split_reservations() {
        let pool = pool(100 * MIB);
        let (a, b) = (spiller("a", &pool), spiller("b", &pool));
        a.try_grow(20 * MIB).unwrap();
        let part = a.split(10 * MIB);
        drop(part);
        // `a` still holds ten: it is still active, and `b` still gets half.
        b.try_grow(50 * MIB).unwrap();
        b.try_grow(1)
            .expect_err("a holds a share while it holds memory");
        a.free();
        b.try_grow(50 * MIB).expect("a holds nothing now");
    }

    #[test]
    fn an_unspillable_consumer_gets_what_is_free_and_shrinks_the_spillers_share() {
        let pool = pool(100 * MIB);
        let fixed = MemoryConsumer::new("fixed").register(&pool);
        fixed.try_grow(40 * MIB).unwrap();
        let sort = spiller("sort", &pool);
        sort.try_grow(60 * MIB)
            .expect("what the unspillable one leaves");
        sort.try_grow(1).expect_err("and no more");
        fixed.try_grow(1).expect_err("nothing is free");
    }

    /// The module doc's caveat, pinned: a newcomer is granted its share at
    /// once, and the pool reports the over-commitment rather than hiding it.
    #[test]
    fn a_newcomer_is_granted_its_share_before_the_holder_gives_its_excess_back() {
        let pool = pool(100 * MIB);
        let (first, second) = (spiller("first", &pool), spiller("second", &pool));
        first.try_grow(100 * MIB).unwrap();
        second.try_grow(50 * MIB).expect("its half, at once");
        assert_eq!(pool.reserved(), 150 * MIB);
        first
            .try_grow(1)
            .expect_err("the holder is over its share: it must spill");
        first.shrink(60 * MIB);
        assert_eq!(pool.reserved(), 90 * MIB);
    }
}
