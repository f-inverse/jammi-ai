//! How a round's tensor bytes move between ranks once the control plane has
//! agreed the round.
//!
//! A collective is a control plane × a transport. The control plane —
//! [`Local`](super::Local)'s rendezvous, [`Peer`](super::Peer)'s two-phase
//! round — owns everything a round's CORRECTNESS rests on: every rank's
//! [`Descriptor`] must agree before any rank is handed a result, a failure is
//! the gang's and permanent, a round applies only at its commit point, and
//! every wait expires at the gang deadline. The transport owns only how the
//! contributions' bytes reach the ranks that fold them:
//!
//! - [`Transport::Inline`] — the bytes ride the control round itself (the
//!   rendezvous slots, the `RunRank` payload frames).
//! - [`Transport::Device`] — the control round carries descriptors only;
//!   once they agree, every rank's tensors move by one device primitive,
//!   [`DeviceExchange::all_gather`], and EVERY rank folds the rank-ordered
//!   contributions on its own device through the one fold
//!   ([`super::round::fold`]), so the result bytes are the ones the inline
//!   transport produces.
//!
//! Agreement first is what makes a device transport safe: a device
//! collective (NCCL) cannot detect two ranks passing different sizes — that
//! is undefined behaviour, not an error — so no rank calls the primitive
//! until the control plane has proved every rank is about to call it with
//! the same packing.
//!
//! The control verbs (`all_reduce_max_flags`, `barrier`) carry a `u32` or
//! nothing and always ride inline; the device primitive moves tensors.
//!
//! # Packing
//!
//! The primitive is the smallest a device library must supply: every rank
//! passes one 1-D buffer of the same length and dtype, and gets the
//! rank-ordered concatenation back. [`exchange_contributions`] packs a
//! contribution's tensors into one buffer per dtype (in the agreed
//! descriptor's tensor order), zero-pads it to the longest rank's buffer
//! (a gather's ranks contribute different row counts; a broadcast's non-root
//! ranks contribute nothing), gathers, and cuts each rank's tensors back out
//! at the shapes the agreed descriptor implies for that rank. The padding
//! never reaches a fold.
//!
//! # The deadline
//!
//! A device collective has no timeout of its own: a rank parked in it against
//! a peer that died waits forever. Every exchange runs under a watchdog armed
//! at the gang deadline; expiry calls [`DeviceExchange::abort`], which ends
//! the parked call, and the exchange returns a typed timeout. A control plane
//! that records a gang fault aborts every rank's exchange it can reach, so a
//! peer parked in the primitive learns of the fault at once rather than at
//! the deadline.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

use candle_core::{DType, Device, Tensor};
use jammi_db::error::{JammiError, Result};

use super::round::{contribution_shapes, Contribution, Shape};
use super::{BlockingCall, Descriptor, Verb};

/// The one device primitive a [`Transport::Device`] needs.
pub trait DeviceExchange: Send + Sync {
    /// The rank-ordered concatenation of every rank's `buf`, on this rank's
    /// device. Every rank passes a 1-D tensor of the same length and dtype —
    /// [`exchange_contributions`]'s packing guarantees it — and gets back
    /// `world × len` elements.
    fn all_gather(&self, call: &BlockingCall, buf: &Tensor) -> Result<Tensor>;

    /// End an exchange in flight on this rank and refuse every later one.
    /// Idempotent, and callable from any thread while an exchange is parked.
    fn abort(&self);

    /// The device this rank's buffers live on.
    fn device(&self) -> &Device;
}

/// How a gang's round bytes move — see the module doc.
#[derive(Clone)]
pub enum Transport {
    /// The bytes ride the control round.
    Inline,
    /// The bytes move by this rank's device exchange, after agreement.
    Device(Arc<dyn DeviceExchange>),
}

impl fmt::Debug for Transport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline => f.write_str("Inline"),
            Self::Device(exchange) => write!(f, "Device({:?})", exchange.device()),
        }
    }
}

impl Transport {
    /// The device exchange a round of `verb` moves its tensors by, or `None`
    /// when the round rides inline — always for the control verbs, which
    /// carry no tensor.
    pub(crate) fn device_for(&self, verb: Verb) -> Option<&Arc<dyn DeviceExchange>> {
        match (self, verb) {
            (_, Verb::AllReduceMaxFlags | Verb::Barrier) => None,
            (Self::Inline, _) => None,
            (Self::Device(exchange), _) => Some(exchange),
        }
    }

    /// End this rank's exchange in flight, if it has one — how a control
    /// plane that recorded a gang fault releases a rank parked in the
    /// primitive.
    pub(crate) fn abort(&self) {
        if let Self::Device(exchange) = self {
            exchange.abort();
        }
    }
}

/// Every rank's contribution to an AGREED round, in rank order, moved by
/// `exchange` — this rank's own is `own`, the others are rebuilt from the
/// gathered bytes at the shapes the agreed `descriptor` implies for them.
///
/// Runs under the gang deadline `timeout`: see the module doc.
pub(crate) fn exchange_contributions(
    exchange: &dyn DeviceExchange,
    call: &BlockingCall,
    descriptor: &Descriptor,
    own: &Contribution,
    rank: u32,
    timeout: Duration,
) -> Result<Vec<Contribution>> {
    let verb = descriptor.verb;
    let world = descriptor.world;
    let shapes: Vec<Vec<Shape>> = (0..world as u32)
        .map(|r| contribution_shapes(descriptor, r))
        .collect();
    let mut per_rank: Vec<Vec<Option<Tensor>>> = shapes
        .iter()
        .map(|rank_shapes| vec![None; rank_shapes.len()])
        .collect();
    let own_tensors = own.tensors();
    if own_tensors.len() != shapes[rank as usize].len() {
        return Err(JammiError::FineTune(format!(
            "{verb}: rank {rank} contributes {} tensors where the agreed round names {}",
            own_tensors.len(),
            shapes[rank as usize].len()
        )));
    }

    for dtype in dtypes_in_order(descriptor) {
        // The element count every rank packs for this dtype, from the agreed
        // shapes alone — identical on every rank, so every rank pads to the
        // same length without exchanging it.
        let counts: Vec<usize> = shapes
            .iter()
            .map(|rank_shapes| {
                rank_shapes
                    .iter()
                    .filter(|(_, d)| *d == dtype)
                    .map(|(dims, _)| dims.iter().product::<usize>())
                    .sum()
            })
            .collect();
        let len = counts.iter().copied().max().unwrap_or(0);
        if len == 0 {
            continue;
        }
        let packed = pack(
            verb,
            &own_tensors,
            &shapes[rank as usize],
            dtype,
            len,
            exchange,
        )?;
        let gathered = within_deadline(exchange, verb, timeout, || {
            exchange.all_gather(call, &packed)
        })?;
        let expected = world * len;
        if gathered.dims() != [expected] {
            return Err(JammiError::FineTune(format!(
                "{verb}: the device exchange returned shape {:?} where {world} ranks × {len} \
                 elements is {expected}",
                gathered.dims()
            )));
        }
        for (peer, rank_shapes) in shapes.iter().enumerate() {
            let mut offset = peer * len;
            for (index, (dims, d)) in rank_shapes.iter().enumerate() {
                if *d != dtype {
                    continue;
                }
                let elements: usize = dims.iter().product();
                let tensor = gathered
                    .narrow(0, offset, elements)
                    .and_then(|t| t.reshape(dims.as_slice()))
                    .map_err(|e| {
                        JammiError::FineTune(format!("{verb}: unpack rank {peer}: {e}"))
                    })?;
                offset += elements;
                per_rank[peer][index] = Some(tensor);
            }
        }
    }

    per_rank
        .into_iter()
        .enumerate()
        .map(|(peer, slots)| {
            if peer == rank as usize {
                return Ok(own.clone());
            }
            let tensors = slots
                .into_iter()
                .map(|slot| slot.expect("every agreed tensor is unpacked from its dtype's buffer"))
                .collect();
            Contribution::from_tensors(verb, tensors)
        })
        .collect()
}

/// The dtypes a round's tensors carry, in the agreed descriptor's order —
/// the same order on every rank.
fn dtypes_in_order(descriptor: &Descriptor) -> Vec<DType> {
    let mut seen = Vec::new();
    for signature in &descriptor.tensors {
        if !seen.contains(&signature.dtype) {
            seen.push(signature.dtype);
        }
    }
    seen
}

/// This rank's tensors of `dtype`, flattened in order onto the exchange's
/// device and zero-padded to `len` elements.
fn pack(
    verb: Verb,
    tensors: &[&Tensor],
    shapes: &[Shape],
    dtype: DType,
    len: usize,
    exchange: &dyn DeviceExchange,
) -> Result<Tensor> {
    let device = exchange.device();
    let mut parts = Vec::new();
    let mut packed_len = 0usize;
    for (tensor, (dims, d)) in tensors.iter().zip(shapes) {
        if *d != dtype {
            continue;
        }
        if tensor.dims() != dims.as_slice() || tensor.dtype() != dtype {
            return Err(JammiError::FineTune(format!(
                "{verb}: a tensor of shape {:?} {:?} does not match the agreed {dims:?} {d:?}",
                tensor.dims(),
                tensor.dtype()
            )));
        }
        let flat = tensor
            .flatten_all()
            .and_then(|t| t.to_device(device))
            .map_err(|e| JammiError::FineTune(format!("{verb}: pack: {e}")))?;
        packed_len += flat.dims()[0];
        if flat.dims()[0] > 0 {
            parts.push(flat);
        }
    }
    if packed_len < len {
        parts.push(
            Tensor::zeros(len - packed_len, dtype, device)
                .map_err(|e| JammiError::FineTune(format!("{verb}: pad: {e}")))?,
        );
    }
    Tensor::cat(&parts, 0)
        .map(|t| t.detach())
        .map_err(|e| JammiError::FineTune(format!("{verb}: pack: {e}")))
}

/// Run `op` with a watchdog armed at `timeout`: expiry aborts `exchange`,
/// which ends a parked call, and the result is a typed timeout — never
/// whatever buffer the aborted call left behind.
fn within_deadline<T>(
    exchange: &dyn DeviceExchange,
    verb: Verb,
    timeout: Duration,
    op: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let done = (Mutex::new(false), Condvar::new());
    let fired = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            let (lock, signal) = &done;
            let guard = lock.lock().unwrap_or_else(PoisonError::into_inner);
            let (guard, wait) = signal
                .wait_timeout_while(guard, timeout, |finished| !*finished)
                .unwrap_or_else(PoisonError::into_inner);
            if wait.timed_out() && !*guard {
                fired.store(true, Ordering::SeqCst);
                exchange.abort();
            }
        });
        let result = op();
        let (lock, signal) = &done;
        *lock.lock().unwrap_or_else(PoisonError::into_inner) = true;
        signal.notify_all();
        result
    });
    if fired.load(Ordering::SeqCst) {
        return Err(JammiError::FineTune(format!(
            "{verb}: the device exchange did not complete within {timeout:?} — a peer never \
             joined it, so this rank's exchange was aborted"
        )));
    }
    result
}
