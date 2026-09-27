//! Hermetic oracles for the device transport, on CPU devices.
//!
//! [`MemExchange`] is a [`DeviceExchange`] over shared memory: the same
//! contract a device library supplies (every rank passes one equal-length
//! 1-D buffer, gets the rank-ordered concatenation back; `abort` ends a
//! parked call), with a per-rank behaviour a test can make stall or fail.
//! The gangs under test are the real control planes driving it, so what is
//! proved here is the transport layer's packing, the agreement-before-bytes
//! order, the deadline and the fault propagation; only the device primitive
//! itself (NCCL) needs a GPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use jammi_db::error::{JammiError, Result};

use super::transport::{DeviceExchange, Transport};
use super::{BlockingCall, Collective, Local, LocalGang};

/// What one rank's [`MemExchange`] does when called.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Behaviour {
    /// Join the gather.
    Join,
    /// Park until aborted — a peer that never joins.
    Stall,
    /// Fail at once — a rank whose device errored.
    Fail,
}

#[derive(Default)]
struct MemState {
    bufs: Vec<Option<Tensor>>,
    arrived: usize,
    published: Option<Arc<Tensor>>,
    taken: usize,
    aborted: Vec<bool>,
}

struct MemGang {
    world: usize,
    state: Mutex<MemState>,
    signal: Condvar,
}

pub(super) struct MemExchange {
    gang: Arc<MemGang>,
    rank: usize,
    device: Device,
    behaviour: Behaviour,
    calls: AtomicUsize,
}

impl MemExchange {
    pub(super) fn gang(world: usize, behaviours: &[Behaviour]) -> Vec<Arc<MemExchange>> {
        let gang = Arc::new(MemGang {
            world,
            state: Mutex::new(MemState {
                bufs: vec![None; world],
                aborted: vec![false; world],
                ..MemState::default()
            }),
            signal: Condvar::new(),
        });
        (0..world)
            .map(|rank| {
                Arc::new(MemExchange {
                    gang: Arc::clone(&gang),
                    rank,
                    device: Device::Cpu,
                    behaviour: behaviours[rank],
                    calls: AtomicUsize::new(0),
                })
            })
            .collect()
    }

    /// How many times this rank called the primitive.
    pub(super) fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn aborted(&self) -> JammiError {
        JammiError::FineTune(format!("rank {}'s exchange was aborted", self.rank))
    }
}

impl DeviceExchange for MemExchange {
    fn all_gather(&self, _call: &BlockingCall, buf: &Tensor) -> Result<Tensor> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let gang = &self.gang;
        let mut state = gang.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.aborted[self.rank] {
            return Err(self.aborted());
        }
        match self.behaviour {
            Behaviour::Fail => {
                return Err(JammiError::Gpu(format!(
                    "rank {}'s device failed the gather",
                    self.rank
                )))
            }
            Behaviour::Stall => loop {
                if state.aborted[self.rank] {
                    return Err(self.aborted());
                }
                state = gang
                    .signal
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
            },
            Behaviour::Join => {}
        }
        // A rank a whole gather ahead waits for the previous result to be
        // taken by every rank before depositing into the next one.
        while state.published.is_some() {
            if state.aborted[self.rank] {
                return Err(self.aborted());
            }
            state = gang
                .signal
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.bufs[self.rank] = Some(buf.clone());
        state.arrived += 1;
        if state.arrived == gang.world {
            let parts: Vec<Tensor> = state
                .bufs
                .iter_mut()
                .map(|b| b.take().expect("every rank deposited"))
                .collect();
            state.published = Some(Arc::new(Tensor::cat(&parts, 0).expect("cat")));
            state.arrived = 0;
            gang.signal.notify_all();
        }
        loop {
            if state.aborted[self.rank] {
                return Err(self.aborted());
            }
            if let Some(published) = state.published.clone() {
                state.taken += 1;
                if state.taken == gang.world {
                    state.published = None;
                    state.taken = 0;
                }
                gang.signal.notify_all();
                return Ok((*published).clone());
            }
            state = gang
                .signal
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn abort(&self) {
        let mut state = self
            .gang
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        state.aborted[self.rank] = true;
        self.gang.signal.notify_all();
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// A `world`-rank CPU gang over `transports`, each rank's body on its own
/// thread; results in rank order.
fn run<T, F>(transports: Vec<Transport>, timeout: Duration, body: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(Local, BlockingCall) -> T + Send + Sync + 'static,
{
    let world = transports.len();
    let gang =
        LocalGang::with_transports(vec![Device::Cpu; world], transports, timeout).expect("gang");
    let body = Arc::new(body);
    let handles: Vec<_> = (0..world as u32)
        .map(|rank| {
            let local = gang.rank(rank).expect("rank");
            let body = Arc::clone(&body);
            BlockingCall::spawn_thread(move |call| body(local, call))
        })
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().expect("rank thread"))
        .collect()
}

fn inline(world: usize) -> Vec<Transport> {
    vec![Transport::Inline; world]
}

fn device(exchanges: &[Arc<MemExchange>]) -> Vec<Transport> {
    exchanges
        .iter()
        .map(|x| Transport::Device(Arc::clone(x) as Arc<dyn DeviceExchange>))
        .collect()
}

/// The bytes of a tensor, whatever its dtype — equality is bit-for-bit.
fn bytes(t: &Tensor) -> (Vec<usize>, DType, Vec<u8>) {
    let flat = t.flatten_all().expect("flatten");
    let raw: Vec<u8> = match t.dtype() {
        DType::F32 => flat
            .to_vec1::<f32>()
            .expect("f32")
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        DType::BF16 => flat
            .to_vec1::<half::bf16>()
            .expect("bf16")
            .into_iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect(),
        other => panic!("no byte reader for {other:?}"),
    };
    (t.dims().to_vec(), t.dtype(), raw)
}

fn ramp(rows: usize, cols: usize, base: f32, dtype: DType) -> Tensor {
    let data: Vec<f32> = (0..rows * cols).map(|i| base + i as f32 * 0.25).collect();
    Tensor::from_vec(data, (rows, cols), &Device::Cpu)
        .expect("tensor")
        .to_dtype(dtype)
        .expect("dtype")
}

/// One program every verb: an unequal gather (a zero-row rank among them),
/// a mixed-dtype sum, a broadcast from a non-zero root, the control word and
/// a barrier. What each rank ends with, as bytes.
fn every_verb(local: Local, call: BlockingCall) -> Vec<(Vec<usize>, DType, Vec<u8>)> {
    let rank = local.rank();
    let counts = [2usize, 0, 3];
    let slice = ramp(counts[rank as usize], 4, 10.0 * rank as f32, DType::F32);
    let gathered = local.all_gather(&call, &slice, &counts).expect("gather");

    let mut tensors = vec![
        ramp(3, 2, rank as f32, DType::F32),
        ramp(2, 5, 1.0 + rank as f32, DType::BF16),
        ramp(1, 3, -(rank as f32), DType::F32),
    ];
    local.all_reduce_sum(&call, &mut tensors).expect("sum");

    let mut announced = ramp(2, 2, 100.0 * (rank as f32 + 1.0), DType::F32);
    local
        .broadcast(&call, &mut announced, 2)
        .expect("broadcast");

    let flags = local.all_reduce_max_flags(&call, 1 << rank).expect("flags");
    local.barrier(&call).expect("barrier");

    let mut out = vec![bytes(&gathered)];
    out.extend(tensors.iter().map(bytes));
    out.push(bytes(&announced));
    out.push((vec![], DType::U32, flags.to_le_bytes().to_vec()));
    out
}

#[test]
fn a_device_transport_gang_ends_every_verb_with_the_inline_gangs_bytes() {
    let timeout = Duration::from_secs(30);
    let inline_result = run(inline(3), timeout, every_verb);
    let exchanges = MemExchange::gang(3, &[Behaviour::Join; 3]);
    let device_result = run(device(&exchanges), timeout, every_verb);
    assert_eq!(
        inline_result, device_result,
        "the device transport must end every rank with the inline transport's bytes"
    );
    // Every rank holds the same result.
    for rank in 1..3 {
        assert_eq!(
            device_result[0], device_result[rank],
            "rank {rank} diverged"
        );
    }
    // The tensor verbs moved by the exchange — one gather per dtype per
    // round: the gather (f32), the sum (f32 + bf16), the broadcast (f32).
    // The control verbs rode inline.
    for exchange in &exchanges {
        assert_eq!(exchange.calls.load(Ordering::SeqCst), 4);
    }
}

#[test]
fn a_disagreeing_round_is_refused_before_any_rank_touches_the_device_exchange() {
    let exchanges = MemExchange::gang(2, &[Behaviour::Join; 2]);
    let results = run(
        device(&exchanges),
        Duration::from_secs(30),
        |local, call| {
            // Rank 1 names a different root: the descriptors disagree.
            let root = local.rank();
            let mut t = ramp(2, 2, 1.0, DType::F32);
            local.broadcast(&call, &mut t, root)
        },
    );
    for (rank, result) in results.iter().enumerate() {
        let error = result.as_ref().expect_err("the round must be refused");
        assert!(
            error.to_string().contains("disagree"),
            "rank {rank}: {error}"
        );
    }
    for exchange in &exchanges {
        assert_eq!(
            exchange.calls.load(Ordering::SeqCst),
            0,
            "no rank may reach the device exchange before the round is agreed"
        );
    }
}

#[test]
fn a_peer_that_never_joins_the_exchange_ends_at_the_deadline_with_a_typed_timeout() {
    let timeout = Duration::from_millis(400);
    let exchanges = MemExchange::gang(3, &[Behaviour::Join, Behaviour::Join, Behaviour::Stall]);
    let started = Instant::now();
    let results = run(device(&exchanges), timeout, |local, call| {
        let mut t = vec![ramp(2, 3, 1.0, DType::F32)];
        let first = local.all_reduce_sum(&call, &mut t);
        // The gang is faulted from here on, on every rank.
        let after = local.barrier(&call);
        (first, after)
    });
    assert!(
        started.elapsed() < timeout * 10,
        "the gang must end near its deadline, not park: {:?}",
        started.elapsed()
    );
    for (rank, (first, after)) in results.iter().enumerate() {
        let error = first.as_ref().expect_err("the round cannot complete");
        let text = error.to_string();
        assert!(
            text.contains("did not complete within")
                || text.contains("already failed")
                || text.contains("aborted"),
            "rank {rank}: {text}"
        );
        after
            .as_ref()
            .expect_err("a collective after the fault is refused");
    }
    // The joining ranks' error names the deadline.
    assert!(results[..2].iter().any(|(first, _)| first
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("did not complete within")));
}

#[test]
fn a_rank_whose_exchange_fails_releases_its_parked_peers_at_once() {
    // A long deadline: the peers must be released by the fault, not by it.
    let timeout = Duration::from_secs(60);
    let exchanges = MemExchange::gang(3, &[Behaviour::Join, Behaviour::Fail, Behaviour::Join]);
    let started = Instant::now();
    let results = run(device(&exchanges), timeout, |local, call| {
        let mut t = vec![ramp(2, 3, 1.0, DType::F32)];
        local.all_reduce_sum(&call, &mut t)
    });
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "a failed exchange must fault the gang at once: {:?}",
        started.elapsed()
    );
    assert!(results[1]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("device failed"));
    for rank in [0, 2] {
        results[rank]
            .as_ref()
            .expect_err("a peer of a failed exchange cannot complete the round");
    }
}

#[test]
fn a_local_gang_refuses_mixed_transports_and_a_transport_count_that_is_not_the_world() {
    let exchanges = MemExchange::gang(2, &[Behaviour::Join; 2]);
    let mixed = vec![
        Transport::Inline,
        Transport::Device(Arc::clone(&exchanges[1]) as Arc<dyn DeviceExchange>),
    ];
    let error = LocalGang::with_transports(vec![Device::Cpu; 2], mixed, Duration::from_secs(1))
        .expect_err("a mixed gang has no common primitive");
    assert!(error.to_string().contains("all inline or all device"));

    let error = LocalGang::with_transports(vec![Device::Cpu; 3], inline(2), Duration::from_secs(1))
        .expect_err("one transport per rank");
    assert!(error.to_string().contains("one per rank"));
}

/// A round where EVERY rank's tensors of a dtype are empty — the gather of a
/// step no rank holds rows for — still hands every rank a tensor of its
/// agreed shape, with the inline transport's bytes, and moves nothing.
#[test]
fn a_round_whose_tensors_are_all_empty_ends_like_the_inline_round() {
    let program = |local: Local, call: BlockingCall| {
        let empty = ramp(1, 4, local.rank() as f32, DType::F32)
            .narrow(0, 0, 0)
            .expect("empty");
        let gathered = local.all_gather(&call, &empty, &[0, 0]).expect("gather");
        bytes(&gathered)
    };
    let inline_result = run(inline(2), Duration::from_secs(30), program);
    let exchanges = MemExchange::gang(2, &[Behaviour::Join; 2]);
    let device_result = run(device(&exchanges), Duration::from_secs(30), program);
    assert_eq!(inline_result, device_result);
    assert_eq!(device_result[0].0, vec![0, 4]);
    for exchange in &exchanges {
        assert_eq!(exchange.calls(), 0, "an all-empty round moves no byte");
    }
}
