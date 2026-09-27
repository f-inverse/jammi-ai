//! The NCCL device exchange: a [`Transport::Device`](super::Transport)'s
//! bytes moved by the device interconnect.
//!
//! Reached through candle's own re-export
//! (`candle_core::cuda::cudarc::nccl`) — the `cuda` feature adds
//! `candle-core/nccl`, which is a pure pass-through to `cudarc/nccl`, so this
//! module needs no direct `cudarc` dependency. **Topology is configuration,
//! not a build feature**: this module is compiled on a CUDA build and SELECTED
//! by `[worker] collective` (`super::select_transport`), under the same
//! control planes the host transport runs under.
//!
//! [`Nccl`] is a [`DeviceExchange`] and nothing more: it moves one packed
//! buffer per call ([`DeviceExchange::all_gather`], `ncclAllGather`). The
//! round's agreement, fault, commit and deadline are the control plane's, and
//! the fold is [`super::round`]'s — so an NCCL gang computes the bytes a host
//! gang computes, and never calls NCCL with a size another rank did not agree
//! to (mismatched NCCL counts are undefined behaviour, not an error).
//!
//! # Three facts this exchange is shaped by
//!
//! **Joining a multi-process gang is bounded.** `ncclCommInitRank` blocks
//! until every rank has joined, and there is no communicator yet to abort, so
//! a peer that died before joining would park the rest forever. A
//! multi-process rank therefore joins through `ncclCommInitRankConfig` with a
//! NON-BLOCKING config, polls `ncclCommGetAsyncError` until the join
//! completes, and aborts the half-built communicator at the gang deadline.
//! The config names only the fields every NCCL since 2.17 carries (through
//! `splitShare`) and says so in its `size`: NCCL starts from its own
//! `NCCL_CONFIG_INITIALIZER` and copies at most `size` bytes over it, so every
//! later field keeps its default whatever layout this build's bindings carry.
//! The single-process gang (`ncclCommInitAll`) has no remote party to wait
//! for and joins blocking.
//!
//! **A dead peer is not detected, and the abort flag is the failure signal.**
//! The host side of a collective does not block (it enqueues on the comm's
//! stream); the wait is in `stream.synchronize()`, and NCCL will sit in it
//! indefinitely against a peer that has died. The escape is
//! `ncclCommAbort` from ANOTHER thread ([`DeviceExchange::abort`], which the
//! transport's deadline watchdog and the control plane's fault both call),
//! after which the blocked `synchronize` returns `Ok(())` with a GARBAGE
//! buffer. So the return value of the sync is not the failure signal —
//! [`Nccl::is_aborted`] is, and every exchange checks it after synchronizing
//! and refuses rather than handing back the garbage.
//!
//! **A communicator must never be aborted twice** (the second call
//! segfaults), and never used after its abort (the handle is freed). The
//! communicator is [`RawComm`], whose `Drop` IS `ncclCommAbort`, held in a
//! `Mutex<Option<RawComm>>`: [`Nccl::abort`] takes it out and drops it, so a
//! double abort is unrepresentable, and every use of the handle happens under
//! that lock, so no call can race the abort that frees it. Each such use is
//! brief — an enqueue, one `ncclCommGetAsyncError` poll — and the one long
//! wait, the stream synchronization, runs with the lock released.
//!
//! # Threading discipline
//!
//! A raw `ncclComm_t` is not `Send`, and NCCL requires one thread per
//! communicator for collectives. [`Nccl`] is `Send + Sync` because the
//! transport holds it behind an `Arc` and the watchdog aborts it from another
//! thread; the discipline that makes that sound is the `Mutex`: at most one
//! thread uses the handle at a time, and the only cross-thread call is the
//! abort, which NCCL documents as callable while a collective is in flight.

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use jammi_db::error::{JammiError, Result};

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{
    CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, ValidAsZeroBits,
};
use candle_core::cuda::cudarc::nccl::{result, sys, Id, NcclType};
use candle_core::cuda_backend::{CudaDType, CudaStorage};
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor};

use super::transport::DeviceExchange;
use super::BlockingCall;

/// The 128 opaque bytes that identify one NCCL communicator, minted by rank 0
/// and carried to every peer on the gang's own link.
pub type NcclIdBytes = [u8; 128];

/// How long a poll of an in-progress NCCL call sleeps before asking again.
const POLL_INTERVAL: Duration = Duration::from_micros(200);

/// NCCL's "this config field is unset" marker (`NCCL_CONFIG_UNDEF_INT`).
const CONFIG_UNDEF_INT: c_int = c_int::MIN;

/// `NCCL_CONFIG_INITIALIZER`'s magic.
const CONFIG_MAGIC: u32 = 0xcafe_beef;

/// A raw NCCL communicator; dropping it aborts it (`ncclCommAbort`).
struct RawComm(sys::ncclComm_t);

// SAFETY: the handle is only ever used under `Nccl::comm`'s lock (the module
// doc's "Threading discipline"); moving it between threads is what NCCL's
// cross-thread abort requires.
unsafe impl Send for RawComm {}

impl Drop for RawComm {
    fn drop(&mut self) {
        // SAFETY: the handle came from a successful init and, by the
        // `Option::take` discipline, is aborted exactly once.
        if let Err(e) = unsafe { result::comm_abort(self.0) } {
            tracing::warn!(status = ?e.0, "ncclCommAbort failed at teardown");
        }
    }
}

/// One rank's NCCL communicator.
pub struct Nccl {
    rank: u32,
    world: u32,
    device: Device,
    stream: Arc<CudaStream>,
    /// The communicator, or `None` once [`Self::abort`] has taken and dropped
    /// it. See the module docs: this `Option` is what makes a double abort
    /// unrepresentable, and its lock is what keeps a call off a freed handle.
    comm: Mutex<Option<RawComm>>,
    /// Set by [`Self::abort`] BEFORE the communicator is dropped, so a rank
    /// blocked in `synchronize` can tell an abort-unblocked return (garbage
    /// buffer) from a real completion.
    aborted: AtomicBool,
}

impl Nccl {
    /// Mint the communicator id on rank 0. The 128 bytes are an opaque
    /// secret: they are the capability to join this gang.
    pub fn new_id() -> Result<NcclIdBytes> {
        let id = Id::new().map_err(|e| nccl_error("ncclGetUniqueId", e))?;
        let mut bytes = [0u8; 128];
        for (out, c) in bytes.iter_mut().zip(id.internal().iter()) {
            *out = *c as u8;
        }
        Ok(bytes)
    }

    /// One communicator per device, all in THIS process — the single-process
    /// multi-GPU gang (`ncclCommInitAll`). Rank `r` is `devices[r]`.
    ///
    /// Each communicator runs on the candle device's OWN stream, the stream
    /// candle launches kernels on, so a collective orders after the kernels
    /// that produced its input without an extra fence.
    pub fn single_process(devices: &[Device]) -> Result<Vec<Self>> {
        if devices.is_empty() {
            return Err(JammiError::Gpu(
                "an NCCL gang needs at least one device".into(),
            ));
        }
        let streams = devices
            .iter()
            .map(cuda_stream)
            .collect::<Result<Vec<_>>>()?;
        let ordinals: Vec<c_int> = streams
            .iter()
            .map(|s| s.context().ordinal() as c_int)
            .collect();
        let mut comms: Vec<sys::ncclComm_t> = vec![std::ptr::null_mut(); devices.len()];
        // SAFETY: `comms` and `ordinals` are both `devices.len()` long.
        unsafe {
            result::comm_init_all(
                comms.as_mut_ptr(),
                devices.len() as c_int,
                ordinals.as_ptr(),
            )
        }
        .map_err(|e| nccl_error("ncclCommInitAll", e))?;
        let world = devices.len() as u32;
        Ok(comms
            .into_iter()
            .zip(devices.iter().zip(streams))
            .enumerate()
            .map(|(rank, (comm, (device, stream)))| Self {
                rank: rank as u32,
                world,
                device: device.clone(),
                stream,
                comm: Mutex::new(Some(RawComm(comm))),
                aborted: AtomicBool::new(false),
            })
            .collect())
    }

    /// This process's single rank of a multi-process gang, joined with the
    /// id rank 0 minted — bounded by `deadline` (the module doc's first
    /// fact): a join that has not completed by then is aborted and refused.
    pub fn from_rank(
        device: &Device,
        rank: u32,
        world: u32,
        id: NcclIdBytes,
        deadline: Duration,
    ) -> Result<Self> {
        if rank >= world {
            return Err(JammiError::Gpu(format!(
                "rank {rank} is not a rank of a gang of {world}"
            )));
        }
        let stream = cuda_stream(device)?;
        // The join binds the communicator to the CALLING thread's current
        // device; bind this rank's before calling.
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| JammiError::Gpu(format!("ncclCommInitRankConfig: bind device: {e}")))?;
        let mut internal = [0 as std::ffi::c_char; 128];
        for (out, b) in internal.iter_mut().zip(id.iter()) {
            *out = *b as std::ffi::c_char;
        }
        let version = result::get_nccl_version().map_err(|e| nccl_error("ncclGetVersion", e))?;
        // SAFETY: an all-zero config is a valid bit pattern for the struct
        // (integers and null pointers); every field NCCL reads — through
        // `size` — is set below.
        let mut config: sys::ncclConfig_t = unsafe { std::mem::zeroed() };
        config.size =
            std::mem::offset_of!(sys::ncclConfig_t, splitShare) + std::mem::size_of::<c_int>();
        config.magic = CONFIG_MAGIC;
        config.version = version as u32;
        config.blocking = 0;
        config.cgaClusterSize = CONFIG_UNDEF_INT;
        config.minCTAs = CONFIG_UNDEF_INT;
        config.maxCTAs = CONFIG_UNDEF_INT;
        config.netName = std::ptr::null();
        config.splitShare = CONFIG_UNDEF_INT;

        let mut comm: sys::ncclComm_t = std::ptr::null_mut();
        // SAFETY: `comm` and `config` outlive the call; the id is 128 bytes.
        let started = unsafe {
            sys::ncclCommInitRankConfig(
                &mut comm,
                world as c_int,
                sys::ncclUniqueId { internal },
                rank as c_int,
                &mut config,
            )
        };
        match started {
            sys::ncclResult_t::ncclSuccess | sys::ncclResult_t::ncclInProgress => {}
            other => {
                return Err(JammiError::Gpu(format!(
                    "ncclCommInitRankConfig: rank {rank} of {world}: nccl status {other:?}"
                )))
            }
        }
        let comm = RawComm(comm);
        let begun = Instant::now();
        loop {
            match async_status(&comm) {
                sys::ncclResult_t::ncclSuccess => break,
                sys::ncclResult_t::ncclInProgress => {
                    if begun.elapsed() >= deadline {
                        // Dropping the half-built communicator aborts it.
                        drop(comm);
                        return Err(JammiError::Gpu(format!(
                            "ncclCommInitRankConfig: rank {rank} of {world} did not complete \
                             joining within {deadline:?} — a peer never joined, so this rank's \
                             half-built communicator was aborted"
                        )));
                    }
                    std::thread::sleep(POLL_INTERVAL);
                }
                other => {
                    drop(comm);
                    return Err(JammiError::Gpu(format!(
                        "ncclCommInitRankConfig: rank {rank} of {world}: nccl status {other:?}"
                    )));
                }
            }
        }
        Ok(Self {
            rank,
            world,
            device: device.clone(),
            stream,
            comm: Mutex::new(Some(comm)),
            aborted: AtomicBool::new(false),
        })
    }

    /// Abort this communicator, unblocking a rank parked in `synchronize`
    /// against a gang member that will never answer.
    ///
    /// Idempotent: the first call takes the communicator out of the `Option`
    /// and drops it (`ncclCommAbort`); a second call finds `None` and does
    /// nothing, so the double abort that segfaults is unreachable. Every later
    /// exchange on this rank refuses with a typed error rather than reading
    /// the garbage buffer an aborted collective leaves behind.
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
        let taken = self
            .comm
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        drop(taken);
    }

    /// Whether [`Self::abort`] has run. This — never a collective's return
    /// value — is the failure signal for a gang whose peer died: an aborted
    /// `synchronize` returns `Ok(())` over a garbage buffer.
    pub fn is_aborted(&self) -> bool {
        self.aborted.load(Ordering::SeqCst)
    }

    /// Refuse once the communicator has been aborted.
    fn check_live(&self, op: &str) -> Result<()> {
        if self.is_aborted() {
            return Err(JammiError::Gpu(format!(
                "{op}: rank {}'s NCCL communicator was aborted — the gang's attempt has failed \
                 and any buffer a collective left behind is garbage",
                self.rank
            )));
        }
        Ok(())
    }

    /// Run `body` with the live handle, under the lock.
    fn with_handle<T>(&self, op: &str, body: impl FnOnce(&RawComm) -> Result<T>) -> Result<T> {
        let guard = self.comm.lock().unwrap_or_else(PoisonError::into_inner);
        let comm = guard.as_ref().ok_or_else(|| {
            JammiError::Gpu(format!(
                "{op}: rank {}'s NCCL communicator was aborted",
                self.rank
            ))
        })?;
        body(comm)
    }

    /// Wait until the enqueued call has left NCCL's in-progress state — one
    /// short lock per poll, so an abort can land between two of them.
    fn settle(&self, op: &str) -> Result<()> {
        loop {
            match self.with_handle(op, |comm| Ok(async_status(comm)))? {
                sys::ncclResult_t::ncclSuccess => return Ok(()),
                sys::ncclResult_t::ncclInProgress => std::thread::sleep(POLL_INTERVAL),
                other => {
                    return Err(JammiError::Gpu(format!(
                        "{op}: rank {}: nccl status {other:?}",
                        self.rank
                    )))
                }
            }
        }
    }

    /// Wait for the enqueued collective on the stream, then decide by the
    /// abort flag. With the lock RELEASED: a watchdog thread must be able to
    /// take it and [`Self::abort`] to end this wait.
    fn synchronize(&self, op: &str) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| JammiError::Gpu(format!("{op}: stream synchronize: {e}")))?;
        self.check_live(op)
    }
}

impl DeviceExchange for Nccl {
    fn all_gather(&self, _call: &BlockingCall, buf: &Tensor) -> Result<Tensor> {
        self.check_live("all_gather")?;
        let contiguous = buf
            .contiguous()
            .map_err(|e| JammiError::Gpu(format!("all_gather: contiguous: {e}")))?;
        let gathered = contiguous
            .apply_op1_no_bwd(&NcclAllGather { nccl: self })
            .map_err(|e| JammiError::Gpu(format!("all_gather: {e}")))?;
        self.settle("all_gather")?;
        self.synchronize("all_gather")?;
        Ok(gathered)
    }

    fn abort(&self) {
        Nccl::abort(self);
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

/// The CUDA stream candle launches `device`'s kernels on.
fn cuda_stream(device: &Device) -> Result<Arc<CudaStream>> {
    Ok(device
        .as_cuda_device()
        .map_err(|e| JammiError::Gpu(format!("NCCL needs a CUDA device: {e}")))?
        .cuda_stream())
}

/// The communicator's asynchronous status: `ncclInProgress` while a
/// non-blocking call is still being set up, `ncclSuccess` once it is not.
fn async_status(comm: &RawComm) -> sys::ncclResult_t {
    let mut status = sys::ncclResult_t::ncclSuccess;
    // SAFETY: a live handle (callers hold `Nccl::comm`'s lock, or own a
    // communicator no other thread can see yet) and a valid out-pointer.
    let polled = unsafe { sys::ncclCommGetAsyncError(comm.0, &mut status) };
    match polled {
        sys::ncclResult_t::ncclSuccess => status,
        failed => failed,
    }
}

/// Format a `cudarc` NCCL error: `NcclError` implements neither `Display` nor
/// `Error`, so its status code is what there is to report.
fn nccl_error(op: &str, e: result::NcclError) -> JammiError {
    JammiError::Gpu(format!("{op}: nccl status {:?}", e.0))
}

/// Refuse a dtype this exchange's `cuda_fwd` match does not cover, at this
/// seam rather than failing to compile a match arm deeper in.
///
/// The message says only that THIS EXCHANGE does not dispatch it — never that
/// NCCL lacks a data type for it, which is not always true: `I16` and the
/// `F8`/`F6`/`F4` family really have no `ncclDataType_t`, but `i8`, `i32` and
/// `u64` do and are refused here anyway, because the match arms below only
/// cover f32, f64, f16, bf16, u8, u32 and i64.
fn refuse_unsupported_dtype(op: &str, dtype: DType) -> candle_core::Result<()> {
    candle_core::bail!("{op}: this exchange does not dispatch {dtype:?}")
}

/// `ncclAllGather` of one packed 1-D buffer: one `sendcount` per rank,
/// `world × sendcount` out. Enqueued under the communicator's lock; the
/// caller settles and synchronizes it with the lock released.
struct NcclAllGather<'a> {
    nccl: &'a Nccl,
}

impl NcclAllGather<'_> {
    fn typed<T>(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)>
    where
        T: CudaDType + NcclType + DeviceRepr + ValidAsZeroBits,
    {
        let slice = storage.as_cuda_slice::<T>()?;
        let send = match layout.contiguous_offsets() {
            Some((start, end)) => slice.slice(start..end),
            None => candle_core::bail!("all_gather: input must be contiguous"),
        };
        let device = storage.device().clone();
        let elements = layout.shape().elem_count();
        let world = self.nccl.world as usize;
        let mut recv = device.alloc_zeros::<T>(elements * world)?;
        let stream = &self.nccl.stream;
        self.nccl
            .with_handle("all_gather", |comm| {
                let (src, _src_record) = send.device_ptr(stream);
                let (dst, _dst_record) = recv.device_ptr_mut(stream);
                // SAFETY: `src` holds `elements` and `dst` `world × elements`
                // of `T` on this rank's device; the handle is live under the
                // lock; the stream is the one both buffers are ordered on.
                let enqueued = unsafe {
                    result::all_gather(
                        src as _,
                        dst as _,
                        elements,
                        T::as_nccl_type(),
                        comm.0,
                        stream.cu_stream() as _,
                    )
                };
                match enqueued {
                    Ok(_) => Ok(()),
                    Err(e) if e.0 == sys::ncclResult_t::ncclInProgress => Ok(()),
                    Err(e) => Err(nccl_error("ncclAllGather", e)),
                }
            })
            .map_err(|e| candle_core::Error::Cuda(e.to_string().into()))?;
        Ok((
            CudaStorage::wrap_cuda_slice(recv, device),
            Shape::from(elements * world),
        ))
    }
}

impl CustomOp1 for NcclAllGather<'_> {
    fn name(&self) -> &'static str {
        "nccl-all-gather"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("nccl-all-gather is a CUDA collective; it has no CPU implementation")
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        match storage.dtype() {
            DType::F32 => self.typed::<f32>(storage, layout),
            DType::F64 => self.typed::<f64>(storage, layout),
            DType::F16 => self.typed::<half::f16>(storage, layout),
            DType::BF16 => self.typed::<half::bf16>(storage, layout),
            DType::U8 => self.typed::<u8>(storage, layout),
            DType::U32 => self.typed::<u32>(storage, layout),
            DType::I64 => self.typed::<i64>(storage, layout),
            other => {
                refuse_unsupported_dtype("all_gather", other)?;
                unreachable!("refuse_unsupported_dtype always returns Err")
            }
        }
    }
}
