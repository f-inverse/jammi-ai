//! A keyed memo whose values are computed once for every concurrent caller
//! of a key, re-probed for freshness on every read, and evicted
//! least-recently-used when idle.
//!
//! The model cache holds two: the loaded models (whose entries carry a
//! device reservation) and the descriptions a model is materialized from.
//! Both are this one shape — snapshot, probe, re-validate, single-flight —
//! so the lost-wakeup and stale-entry reasoning lives here once.

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::Arc;

use jammi_db::error::Result;
use tokio::sync::{Notify, RwLock};

/// A value a [`Memo`] holds: how it is probed, handed out and released.
pub(crate) trait MemoEntry: Send + Sync {
    /// What a lookup clones out of an entry under the read lock, to probe
    /// outside it. It must not hold anything [`Self::is_idle`] counts: a
    /// clone made before the entry is handed out would be invisible to
    /// eviction.
    type Probe: Send;
    /// What a caller receives.
    type Handle;

    /// The probe handle for this entry.
    fn probe_handle(&self) -> Self::Probe;
    /// Whether `probe` was taken from this very entry — false once the entry
    /// was evicted or replaced while the probe ran.
    fn is(&self, probe: &Self::Probe) -> bool;
    /// Re-`stat` what the value was computed from: `Ok(true)` fresh,
    /// `Ok(false)` stale, `Err` unprobeable.
    fn probe_freshness(probe: &Self::Probe) -> Result<bool>;
    /// Hand the value out. Called under the memo's write lock, in the same
    /// critical section that found the entry current, so eviction never
    /// observes the entry idle while a handle is being made.
    fn hand_out(&self) -> Self::Handle;
    /// Whether removing this entry now releases what it holds.
    fn is_idle(&self) -> bool;
}

/// The memo's state: its entries, their recency, and the computations in
/// flight. One lock guards all three, so a key is never both in flight and
/// present to a reader that holds the lock.
pub(crate) struct MemoState<K, E> {
    entries: HashMap<K, E>,
    lru: VecDeque<K>,
    in_flight: HashMap<K, Arc<Notify>>,
}

impl<K, E> Default for MemoState<K, E> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            in_flight: HashMap::new(),
        }
    }
}

impl<K: Clone + Eq + Hash, E: MemoEntry> MemoState<K, E> {
    fn insert(&mut self, key: K, entry: E) {
        self.lru.retain(|k| k != &key);
        self.lru.push_back(key.clone());
        self.entries.insert(key, entry);
    }

    fn touch(&mut self, key: &K) {
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }

    fn remove(&mut self, key: &K) -> Option<E> {
        self.lru.retain(|k| k != key);
        self.entries.remove(key)
    }

    /// Remove the least-recently-used idle entry whose key satisfies
    /// `eligible`, returning it so the caller drops it — and releases what
    /// it holds — outside any lock. `None` means nothing eligible can be
    /// released: an entry that is not idle is skipped, never removed, so a
    /// `Some` is always real progress.
    pub(crate) fn evict_one(&mut self, eligible: impl Fn(&K) -> bool) -> Option<(K, E)> {
        let key = self
            .lru
            .iter()
            .find(|k| eligible(k) && self.entries.get(*k).is_some_and(E::is_idle))
            .cloned()?;
        self.remove(&key).map(|entry| (key, entry))
    }

    /// Evict idle entries, oldest first, until at most `capacity` remain.
    /// Entries in use are never evicted, so the count can exceed the
    /// capacity while they are held.
    fn shed_to(&mut self, capacity: NonZeroUsize) -> Vec<(K, E)> {
        std::iter::from_fn(|| {
            (self.entries.len() > capacity.get())
                .then(|| self.evict_one(|_| true))
                .flatten()
        })
        .collect()
    }

    /// The keys held, in recency order (oldest first).
    pub(crate) fn keys(&self) -> impl Iterator<Item = &K> {
        self.lru.iter()
    }

    #[cfg(test)]
    pub(crate) fn get(&self, key: &K) -> Option<&E> {
        self.entries.get(key)
    }

    #[cfg(test)]
    pub(crate) fn insert_for_test(&mut self, key: K, entry: E) {
        self.insert(key, entry);
    }

    #[cfg(test)]
    pub(crate) fn touch_for_test(&mut self, key: &K) {
        self.touch(key);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub(crate) fn in_flight_for_test(&mut self) -> &mut HashMap<K, Arc<Notify>> {
        &mut self.in_flight
    }
}

/// A test's half of a deterministic pause: signalled when the paused task
/// has arrived, and released by the test.
#[cfg(test)]
pub(crate) struct PauseHandle {
    /// Signalled once the paused task has reached the pause point.
    pub(crate) arrived: Arc<Notify>,
    /// The test calls `.notify_one()` on this to resume the paused task.
    pub(crate) release: Arc<Notify>,
}

/// One installable pause point. Consumed the first time it is reached, so
/// it pauses once per installation; a no-op when none is installed.
#[cfg(test)]
#[derive(Default)]
struct PausePoint(std::sync::Mutex<Option<PauseHandle>>);

#[cfg(test)]
impl PausePoint {
    fn install(&self) -> PauseHandle {
        let arrived = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        *self.0.lock().expect("the pause point is never poisoned") = Some(PauseHandle {
            arrived: Arc::clone(&arrived),
            release: Arc::clone(&release),
        });
        PauseHandle { arrived, release }
    }

    async fn reach(&self) {
        let handle = self
            .0
            .lock()
            .expect("the pause point is never poisoned")
            .take();
        if let Some(handle) = handle {
            handle.arrived.notify_one();
            handle.release.notified().await;
        }
    }
}

/// A keyed memo: one computation per key for every concurrent caller, a
/// freshness probe on every read, and least-recently-used eviction of idle
/// entries past an optional capacity.
///
/// The guarantee is BOUNDED staleness: a handle was fresh at some instant
/// before the call that returned it began, and is never revalidated after.
pub(crate) struct Memo<K, E> {
    state: RwLock<MemoState<K, E>>,
    capacity: Option<NonZeroUsize>,
    #[cfg(test)]
    probe_pause: PausePoint,
    #[cfg(test)]
    wait_pause: PausePoint,
}

impl<K, E> Memo<K, E>
where
    K: Clone + Eq + Hash + std::fmt::Debug + Send + Sync,
    E: MemoEntry,
{
    /// An empty memo holding at most `capacity` idle entries (`None`:
    /// unbounded).
    pub(crate) fn new(capacity: Option<NonZeroUsize>) -> Self {
        Self {
            state: RwLock::new(MemoState::default()),
            capacity,
            #[cfg(test)]
            probe_pause: PausePoint::default(),
            #[cfg(test)]
            wait_pause: PausePoint::default(),
        }
    }

    /// The value under `key`, computing it with `compute` when no fresh one
    /// is held.
    ///
    /// A held entry is probed OUTSIDE the lock (a probe `stat`s files, and
    /// holding the lock across it would stall every other key), then
    /// re-validated under the write lock by identity: an entry replaced
    /// while the probe ran is retried against, never handed out. A stale or
    /// unprobeable entry is evicted — only if it is still the entry probed —
    /// and an unprobeable one's error is returned, so the next call takes
    /// the cold path instead of re-probing a dead entry forever.
    ///
    /// A caller that finds the key in flight waits for it. Its `Notified`
    /// future is `enable()`d while the write lock is still held, and the
    /// computing caller must take that lock to leave `in_flight` before it
    /// calls `notify_waiters`, so registration happens-before any wakeup
    /// that applies to it: the lost-wakeup window is closed structurally,
    /// not by a timeout. The waiter then retries from the top — the
    /// computation may have failed.
    pub(crate) async fn get_or_compute<F, Fut>(&self, key: &K, compute: F) -> Result<E::Handle>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<E>>,
    {
        loop {
            let snapshot = self
                .state
                .read()
                .await
                .entries
                .get(key)
                .map(E::probe_handle);
            if let Some(probe) = snapshot {
                #[cfg(test)]
                self.probe_pause.reach().await;

                match E::probe_freshness(&probe) {
                    Ok(true) => {
                        let mut state = self.state.write().await;
                        if let Some(handle) = state
                            .entries
                            .get(key)
                            .filter(|entry| entry.is(&probe))
                            .map(E::hand_out)
                        {
                            state.touch(key);
                            return Ok(handle);
                        }
                        continue;
                    }
                    Ok(false) => drop(self.evict_if_current(key, &probe).await),
                    Err(e) => {
                        drop(self.evict_if_current(key, &probe).await);
                        return Err(e);
                    }
                }
            }

            let mut state = self.state.write().await;
            if state.entries.contains_key(key) {
                continue;
            }
            if let Some(notify) = state.in_flight.get(key).map(Arc::clone) {
                let notified = notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                drop(state);
                #[cfg(test)]
                self.wait_pause.reach().await;
                notified.await;
                continue;
            }
            let notify = Arc::new(Notify::new());
            state.in_flight.insert(key.clone(), Arc::clone(&notify));
            drop(state);

            let computed = compute().await;

            let mut state = self.state.write().await;
            state.in_flight.remove(key);
            let outcome = computed.map(|entry| {
                let handle = entry.hand_out();
                state.insert(key.clone(), entry);
                let shed = self
                    .capacity
                    .map(|capacity| state.shed_to(capacity))
                    .unwrap_or_default();
                (handle, shed)
            });
            drop(state);
            notify.notify_waiters();
            return outcome.map(|(handle, shed)| {
                for (evicted, _) in &shed {
                    tracing::info!(key = ?evicted, "evicted an idle entry past the memo's capacity");
                }
                handle
            });
        }
    }

    /// Remove `key`'s entry if — and only if — it is still the entry `probe`
    /// was taken from; a concurrent caller may already have replaced it.
    /// Unconditional on idleness: serving a stale value is a correctness
    /// bug, not a capacity one. The removed entry is returned to be dropped
    /// outside the lock.
    async fn evict_if_current(&self, key: &K, probe: &E::Probe) -> Option<E> {
        let mut state = self.state.write().await;
        if state.entries.get(key).is_some_and(|entry| entry.is(probe)) {
            state.remove(key)
        } else {
            None
        }
    }

    /// Evict the least-recently-used idle entry whose key satisfies
    /// `eligible` — see [`MemoState::evict_one`].
    pub(crate) async fn evict_one(&self, eligible: impl Fn(&K) -> bool) -> Option<(K, E)> {
        self.state.write().await.evict_one(eligible)
    }

    /// Read the memo's state.
    pub(crate) async fn read(&self) -> tokio::sync::RwLockReadGuard<'_, MemoState<K, E>> {
        self.state.read().await
    }

    #[cfg(test)]
    pub(crate) async fn write_for_test(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, MemoState<K, E>> {
        self.state.write().await
    }

    /// Test seam: pause the next lookup between its snapshot and its probe.
    #[cfg(test)]
    pub(crate) fn install_probe_pause(&self) -> PauseHandle {
        self.probe_pause.install()
    }

    /// Test seam: pause the next single-flight waiter after it registered
    /// and before it awaits.
    #[cfg(test)]
    pub(crate) fn install_wait_pause(&self) -> PauseHandle {
        self.wait_pause.install()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A value with a switchable freshness and an in-use count — enough to
    /// drive every arm of the memo without a model.
    struct Probed {
        fresh: Arc<AtomicBool>,
        in_use: Arc<AtomicUsize>,
        value: usize,
    }

    impl MemoEntry for Probed {
        type Probe = (Arc<AtomicBool>, usize);
        type Handle = usize;
        fn probe_handle(&self) -> Self::Probe {
            (Arc::clone(&self.fresh), self.value)
        }
        fn is(&self, probe: &Self::Probe) -> bool {
            Arc::ptr_eq(&self.fresh, &probe.0)
        }
        fn probe_freshness(probe: &Self::Probe) -> Result<bool> {
            Ok(probe.0.load(Ordering::SeqCst))
        }
        fn hand_out(&self) -> usize {
            self.value
        }
        fn is_idle(&self) -> bool {
            self.in_use.load(Ordering::SeqCst) == 0
        }
    }

    fn probed(value: usize) -> Probed {
        Probed {
            fresh: Arc::new(AtomicBool::new(true)),
            in_use: Arc::new(AtomicUsize::new(0)),
            value,
        }
    }

    /// Concurrent callers of one key share one computation.
    #[tokio::test]
    async fn one_computation_serves_every_concurrent_caller() {
        let memo = Arc::new(Memo::<&str, Probed>::new(None));
        let computations = Arc::new(AtomicUsize::new(0));
        let callers = (0..8).map(|_| {
            let memo = Arc::clone(&memo);
            let computations = Arc::clone(&computations);
            tokio::spawn(async move {
                memo.get_or_compute(&"k", || async move {
                    computations.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                    Ok(probed(7))
                })
                .await
                .unwrap()
            })
        });
        for caller in callers.collect::<Vec<_>>() {
            assert_eq!(caller.await.unwrap(), 7);
        }
        assert_eq!(computations.load(Ordering::SeqCst), 1);
    }

    /// A stale entry is recomputed; a fresh one is served as held.
    #[tokio::test]
    async fn a_stale_entry_is_recomputed() {
        let memo = Memo::<&str, Probed>::new(None);
        let first = probed(1);
        let fresh = Arc::clone(&first.fresh);
        assert_eq!(
            memo.get_or_compute(&"k", || async { Ok(first) })
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            memo.get_or_compute(&"k", || async { Ok(probed(2)) })
                .await
                .unwrap(),
            1
        );
        fresh.store(false, Ordering::SeqCst);
        assert_eq!(
            memo.get_or_compute(&"k", || async { Ok(probed(2)) })
                .await
                .unwrap(),
            2
        );
    }

    /// Past its capacity the memo sheds its oldest idle entries, and never
    /// one in use.
    #[tokio::test]
    async fn capacity_sheds_the_oldest_idle_entry_and_never_one_in_use() {
        let memo = Memo::<&str, Probed>::new(NonZeroUsize::new(2));
        let held = probed(1);
        let in_use = Arc::clone(&held.in_use);
        in_use.store(1, Ordering::SeqCst);
        memo.get_or_compute(&"a", || async { Ok(held) })
            .await
            .unwrap();
        memo.get_or_compute(&"b", || async { Ok(probed(2)) })
            .await
            .unwrap();
        memo.get_or_compute(&"c", || async { Ok(probed(3)) })
            .await
            .unwrap();
        let keys: Vec<_> = memo.read().await.keys().copied().collect();
        assert_eq!(
            keys,
            vec!["a", "c"],
            "b is the oldest idle entry; a is in use"
        );

        memo.get_or_compute(&"d", || async { Ok(probed(4)) })
            .await
            .unwrap();
        let keys: Vec<_> = memo.read().await.keys().copied().collect();
        assert_eq!(keys, vec!["a", "d"]);

        in_use.store(0, Ordering::SeqCst);
        memo.get_or_compute(&"e", || async { Ok(probed(5)) })
            .await
            .unwrap();
        let keys: Vec<_> = memo.read().await.keys().copied().collect();
        assert_eq!(keys, vec!["d", "e"], "released, a is shed first");
    }

    /// A failed computation is not memoized, and a waiter retries it.
    #[tokio::test]
    async fn a_failed_computation_is_not_memoized() {
        let memo = Memo::<&str, Probed>::new(None);
        let failed = memo
            .get_or_compute(&"k", || async {
                Err(jammi_db::error::JammiError::Config("refused".into()))
            })
            .await;
        assert!(failed.is_err());
        assert_eq!(memo.read().await.keys().count(), 0);
        assert_eq!(
            memo.get_or_compute(&"k", || async { Ok(probed(9)) })
                .await
                .unwrap(),
            9
        );
    }
}
