//! The NCCL collective: ranks on CUDA devices, reduced by the device
//! interconnect.
//!
//! Reached through candle's own re-export
//! (`candle_core::cuda::cudarc::nccl`) — the `cuda` feature adds
//! `candle-core/nccl`, which is a pure pass-through to `cudarc/nccl`, so this
//! arm needs no direct `cudarc` dependency. **Topology is configuration, not
//! a build feature**: this module is compiled on a CUDA build and SELECTED by
//! `[worker] collective`, and the trainer that drives it holds the same
//! `&dyn Collective` it holds for the host arms.
//!
//! # Three facts this arm is shaped by
//!
//! **NCCL has no `allgatherv`.** One `sendcount`, identical on every rank.
//! Unequal counts are therefore pad-to-max on the way in and narrow on the
//! way out, so [`Collective::all_gather`]'s contract — the rank-ordered
//! concatenation of exactly the contributed rows — holds here as it does on
//! the host arms, and the padding is never visible to the trainer.
//!
//! **A dead peer is not detected, and the abort flag is the failure signal.**
//! The host side of a collective does not block (it enqueues on the comm's
//! stream); the wait is in `stream.synchronize()`, and NCCL will sit in it
//! indefinitely against a peer that has died. The escape is
//! `ncclCommAbort` from ANOTHER thread ([`Nccl::abort`], which a per-attempt
//! watchdog calls), after which the blocked `synchronize` returns `Ok(())`
//! with a GARBAGE buffer. So the return value of the sync is not the failure
//! signal — [`Nccl::is_aborted`] is, and every operation here checks it after
//! synchronizing and refuses rather than handing back the garbage.
//!
//! **A communicator must never be aborted twice** (the second call
//! segfaults). `cudarc`'s `Drop for Comm` IS an abort, and it exposes no
//! other way to abort, so "abort" here means "drop the communicator". The
//! double abort is made unrepresentable by the type rather than guarded by a
//! flag: the comm lives in a `Mutex<Option<Comm>>` and [`Nccl::abort`] takes
//! it out ([`Option::take`]) and drops it. After an abort the `Option` holds
//! no `Comm`, so dropping the [`Nccl`] has nothing to drop; and a
//! never-aborted [`Nccl`] drops its one `Comm` exactly once, through the
//! same `Drop` — which is the abort NCCL wants at teardown.
//!
//! # Threading discipline
//!
//! `Comm` is `!Send + !Sync` in `cudarc` (it is a raw `ncclComm_t`), and NCCL
//! requires one thread per communicator for collectives. [`Nccl`] carries
//! `unsafe impl Send + Sync` because the trainer holds it behind an `Arc` and
//! the watchdog aborts it from another thread; the discipline that makes that
//! sound is enforced by the `Mutex`: at most one thread is inside a
//! collective on a given communicator at a time, and the only cross-thread
//! call is the abort, which NCCL explicitly documents as callable from
//! another thread while a collective is in flight.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use jammi_db::error::{JammiError, Result};

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use candle_core::cuda::cudarc::nccl::{Comm, Id, NcclType, ReduceOp};
use candle_core::cuda_backend::{CudaDType, CudaStorage};
use candle_core::{CpuStorage, CustomOp1, DType, Device, Layout, Shape, Tensor};

use super::{checked_gather_counts, checked_root, Collective};

/// The 128 opaque bytes that identify one NCCL communicator, minted by rank 0
/// and carried to every peer out of band.
pub type NcclIdBytes = [u8; 128];

/// One rank's NCCL communicator.
pub struct Nccl {
    rank: u32,
    world: u32,
    device: Device,
    /// The communicator, or `None` once [`Self::abort`] has taken and dropped
    /// it. See the module docs: this `Option` is what makes a double abort
    /// unrepresentable.
    comm: Mutex<Option<Comm>>,
    /// Set by [`Self::abort`] BEFORE the communicator is dropped, so a rank
    /// blocked in `synchronize` can tell an abort-unblocked return (garbage
    /// buffer) from a real completion.
    aborted: AtomicBool,
}

// SAFETY: see the module docs' "Threading discipline". The `Mutex` admits one
// thread at a time to the communicator; the only concurrent access is
// `abort`, which NCCL documents as callable from another thread.
unsafe impl Send for Nccl {}
unsafe impl Sync for Nccl {}

impl Nccl {
    /// Mint the communicator id on rank 0. The 128 bytes are an opaque
    /// secret: they are the capability to join this gang, and they travel to
    /// the peers out of band.
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
    /// Each communicator is built on the candle device's OWN stream
    /// (`CudaDevice::cuda_stream()`), the stream candle allocates and
    /// launches kernels on, so a collective orders after the kernels that
    /// produced its input without an extra fence.
    pub fn single_process(devices: &[Device]) -> Result<Vec<Self>> {
        if devices.is_empty() {
            return Err(JammiError::Gpu(
                "an NCCL gang needs at least one device".into(),
            ));
        }
        let mut streams = Vec::with_capacity(devices.len());
        for device in devices {
            let cuda = device
                .as_cuda_device()
                .map_err(|e| JammiError::Gpu(format!("NCCL needs a CUDA device: {e}")))?;
            streams.push(cuda.cuda_stream());
        }
        let world = devices.len() as u32;
        let comms = Comm::from_devices(streams).map_err(|e| nccl_error("ncclCommInitAll", e))?;
        Ok(comms
            .into_iter()
            .zip(devices.iter())
            .enumerate()
            .map(|(rank, (comm, device))| Self {
                rank: rank as u32,
                world,
                device: device.clone(),
                comm: Mutex::new(Some(comm)),
                aborted: AtomicBool::new(false),
            })
            .collect())
    }

    /// This process's single rank of a multi-process gang
    /// (`ncclCommInitRank`), joined with the id rank 0 minted.
    pub fn from_rank(device: &Device, rank: u32, world: u32, id: NcclIdBytes) -> Result<Self> {
        if rank >= world {
            return Err(JammiError::Gpu(format!(
                "rank {rank} is not a rank of a gang of {world}"
            )));
        }
        let cuda = device
            .as_cuda_device()
            .map_err(|e| JammiError::Gpu(format!("NCCL needs a CUDA device: {e}")))?;
        let mut internal = [0 as std::ffi::c_char; 128];
        for (out, b) in internal.iter_mut().zip(id.iter()) {
            *out = *b as std::ffi::c_char;
        }
        let comm = Comm::from_rank(
            cuda.cuda_stream(),
            rank as usize,
            world as usize,
            Id::uninit(internal),
        )
        .map_err(|e| nccl_error("ncclCommInitRank", e))?;
        Ok(Self {
            rank,
            world,
            device: device.clone(),
            comm: Mutex::new(Some(comm)),
            aborted: AtomicBool::new(false),
        })
    }

    /// Abort this communicator, unblocking a peer rank that is parked in
    /// `synchronize` against a gang member that will never answer.
    ///
    /// Idempotent: the first call takes the `Comm` out of the `Option` and
    /// drops it (`cudarc`'s `Drop` is `ncclCommAbort`); a second call finds
    /// `None` and does nothing, so the double abort that segfaults is not
    /// something a caller can reach. Every subsequent collective on this rank
    /// refuses with a typed error rather than reading the garbage buffer an
    /// aborted collective leaves behind.
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

    /// The device this rank trains on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Refuse once the communicator has been aborted.
    fn check_live(&self, op: &str) -> Result<()> {
        if self.is_aborted() {
            return Err(JammiError::Gpu(format!(
                "{op}: this rank's NCCL communicator was aborted — the gang's attempt has \
                 failed and any buffer a collective left behind is garbage"
            )));
        }
        Ok(())
    }

    /// Wait for the enqueued collective, then decide by the abort flag.
    ///
    /// The synchronization deliberately happens with the communicator's lock
    /// RELEASED: the host enqueue does not block but this wait does, and a
    /// watchdog thread must be able to take the lock and [`Self::abort`] to
    /// end it.
    fn synchronize(&self, op: &str) -> Result<()> {
        let cuda = self
            .device
            .as_cuda_device()
            .map_err(|e| JammiError::Gpu(format!("{op}: {e}")))?;
        cuda.cuda_stream()
            .synchronize()
            .map_err(|e| JammiError::Gpu(format!("{op}: stream synchronize: {e}")))?;
        self.check_live(op)
    }

    /// Run `body` with the live communicator, then synchronize outside the
    /// lock and check the abort flag.
    fn with_comm<T>(&self, op: &str, body: impl FnOnce(&Comm) -> Result<T>) -> Result<T> {
        self.check_live(op)?;
        let out = {
            let guard = self.comm.lock().unwrap_or_else(PoisonError::into_inner);
            let comm = guard.as_ref().ok_or_else(|| {
                JammiError::Gpu(format!("{op}: this rank's NCCL communicator was aborted"))
            })?;
            body(comm)?
        };
        self.synchronize(op)?;
        Ok(out)
    }
}

impl Collective for Nccl {
    fn all_gather(&self, local: &Tensor, counts: &[usize]) -> Result<Tensor> {
        let total = checked_gather_counts(self.rank, self.world, local, counts)?;
        let max_rows = counts.iter().copied().max().unwrap_or(0);
        if max_rows == 0 {
            // Every rank contributed nothing: the gather is this rank's own
            // (empty) tensor, which already has the trailing shape and dtype.
            return Ok(local.clone());
        }

        // Pad to the maximum count so every rank sends the identical
        // `sendcount` NCCL requires, gather `world × max_rows`, then narrow
        // each rank's slot back to the rows it actually contributed. The
        // padding is an implementation detail of this arm and never reaches
        // the caller.
        let padded = pad_rows(local, max_rows)?;
        let gathered = self.with_comm("all_gather", |comm| {
            padded
                .apply_op1_no_bwd(&NcclAllGather {
                    comm,
                    world: self.world as usize,
                })
                .map_err(|e| JammiError::Gpu(format!("all_gather: {e}")))
        })?;

        let mut slices = Vec::with_capacity(counts.len());
        for (peer, rows) in counts.iter().copied().enumerate() {
            if rows == 0 {
                continue;
            }
            let slot = gathered
                .narrow(0, peer * max_rows, rows)
                .map_err(|e| JammiError::Gpu(format!("all_gather: narrow: {e}")))?;
            // Only this rank's own slot may carry a gradient; the gather
            // itself has no backward (`apply_op1_no_bwd`), so the local slot
            // is re-attached from the caller's own tensor.
            slices.push(if peer == self.rank as usize {
                local.clone()
            } else {
                slot
            });
        }
        let out = if slices.len() == 1 {
            slices.into_iter().next().expect("length checked")
        } else {
            Tensor::cat(&slices, 0).map_err(|e| JammiError::Gpu(format!("all_gather: cat: {e}")))?
        };
        let rows = out.dims().first().copied().unwrap_or(0);
        if rows != total {
            return Err(JammiError::Gpu(format!(
                "all_gather: gathered {rows} rows where the counts sum to {total}"
            )));
        }
        Ok(out)
    }

    fn all_reduce_sum(&self, tensors: &mut [Tensor]) -> Result<()> {
        for tensor in tensors.iter_mut() {
            let reduced = self.with_comm("all_reduce_sum", |comm| {
                tensor
                    .apply_op1_no_bwd(&NcclAllReduce {
                        comm,
                        op: ReduceOp::Sum,
                    })
                    .map_err(|e| JammiError::Gpu(format!("all_reduce_sum: {e}")))
            })?;
            *tensor = reduced;
        }
        Ok(())
    }

    fn all_reduce_max_flags(&self, flags: u32) -> Result<u32> {
        let local = Tensor::from_vec(vec![flags], 1, &self.device)
            .map_err(|e| JammiError::Gpu(format!("all_reduce_max_flags: {e}")))?;
        let maxed = self.with_comm("all_reduce_max_flags", |comm| {
            local
                .apply_op1_no_bwd(&NcclAllReduce {
                    comm,
                    op: ReduceOp::Max,
                })
                .map_err(|e| JammiError::Gpu(format!("all_reduce_max_flags: {e}")))
        })?;
        let values = maxed
            .to_vec1::<u32>()
            .map_err(|e| JammiError::Gpu(format!("all_reduce_max_flags: read back: {e}")))?;
        values
            .first()
            .copied()
            .ok_or_else(|| JammiError::Gpu("all_reduce_max_flags: empty result".into()))
    }

    fn broadcast(&self, t: &mut Tensor, root: u32) -> Result<()> {
        checked_root(self.world, root)?;
        let out = self.with_comm("broadcast", |comm| {
            t.apply_op1_no_bwd(&NcclBroadcast {
                comm,
                root: root as i32,
            })
            .map_err(|e| JammiError::Gpu(format!("broadcast: {e}")))
        })?;
        *t = out;
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        // NCCL has no barrier: the idiom is a one-element collective plus the
        // stream synchronization `with_comm` already performs.
        self.all_reduce_max_flags(0)?;
        Ok(())
    }

    fn rank(&self) -> u32 {
        self.rank
    }

    fn world(&self) -> u32 {
        self.world
    }
}

/// Zero-pad `t` to `rows` rows along dim 0. Every rank must send the same
/// element count, and the padding rows are narrowed away again on the far
/// side.
fn pad_rows(t: &Tensor, rows: usize) -> Result<Tensor> {
    let have = t.dims().first().copied().unwrap_or(0);
    if have == rows {
        return t
            .contiguous()
            .map_err(|e| JammiError::Gpu(format!("all_gather: contiguous: {e}")));
    }
    let mut shape = t.dims().to_vec();
    if shape.is_empty() {
        return Err(JammiError::Gpu(
            "all_gather: a scalar has no rows to gather along".into(),
        ));
    }
    shape[0] = rows - have;
    let pad = Tensor::zeros(shape, t.dtype(), t.device())
        .map_err(|e| JammiError::Gpu(format!("all_gather: pad: {e}")))?;
    let padded = if have == 0 {
        pad
    } else {
        Tensor::cat(&[t, &pad], 0).map_err(|e| JammiError::Gpu(format!("all_gather: pad: {e}")))?
    };
    padded
        .contiguous()
        .map_err(|e| JammiError::Gpu(format!("all_gather: contiguous: {e}")))
}

/// Format a `cudarc` NCCL error: `NcclError` implements neither `Display` nor
/// `Error`, so its status code is what there is to report.
fn nccl_error(op: &str, e: candle_core::cuda::cudarc::nccl::result::NcclError) -> JammiError {
    JammiError::Gpu(format!("{op}: nccl status {:?}", e.0))
}

/// The dtypes that have an `ncclDataType_t`, refused at this seam rather than
/// failing to compile a match arm deeper in.
fn refuse_unsupported_dtype(op: &str, dtype: DType) -> candle_core::Result<()> {
    candle_core::bail!(
        "{op}: dtype {dtype:?} has no NCCL data type (f32, f64, f16, bf16, u8, u32 and i64 do)"
    )
}

/// The contiguous device slice a collective sends from.
///
/// A non-contiguous input is refused rather than silently sending the wrong
/// elements: the caller makes it contiguous first, which is a copy it can see
/// rather than one hidden here.
macro_rules! send_view {
    ($op:expr, $storage:expr, $layout:expr, $t:ty) => {{
        let slice = $storage.as_cuda_slice::<$t>()?;
        match $layout.contiguous_offsets() {
            Some((start, end)) => slice.slice(start..end),
            None => candle_core::bail!("{}: input must be contiguous", $op),
        }
    }};
}

/// `ncclAllReduce` over one tensor, in place of a candle op.
struct NcclAllReduce<'a> {
    comm: &'a Comm,
    op: ReduceOp,
}

impl NcclAllReduce<'_> {
    fn typed<T>(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)>
    where
        T: CudaDType + NcclType + DeviceRepr + ValidAsZeroBits,
    {
        let send = send_view!("all_reduce", storage, layout, T);
        let device = storage.device().clone();
        let mut recv = device.alloc_zeros::<T>(layout.shape().elem_count())?;
        self.comm
            .all_reduce(&send, &mut recv, &self.op)
            .map_err(|e| candle_core::Error::Cuda(format!("nccl status {:?}", e.0).into()))?;
        Ok((
            CudaStorage::wrap_cuda_slice(recv, device),
            layout.shape().clone(),
        ))
    }
}

impl CustomOp1 for NcclAllReduce<'_> {
    fn name(&self) -> &'static str {
        "nccl-all-reduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("nccl-all-reduce is a CUDA collective; it has no CPU implementation")
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
                refuse_unsupported_dtype("all_reduce", other)?;
                unreachable!("refuse_unsupported_dtype always returns Err")
            }
        }
    }
}

/// `ncclAllGather`: one `sendcount` per rank, `world × sendcount` out.
struct NcclAllGather<'a> {
    comm: &'a Comm,
    world: usize,
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
        let send = send_view!("all_gather", storage, layout, T);
        let device = storage.device().clone();
        let elements = layout.shape().elem_count();
        let mut recv = device.alloc_zeros::<T>(elements * self.world)?;
        self.comm
            .all_gather(&send, &mut recv)
            .map_err(|e| candle_core::Error::Cuda(format!("nccl status {:?}", e.0).into()))?;
        let mut dims = layout.shape().dims().to_vec();
        if dims.is_empty() {
            candle_core::bail!("all_gather: a scalar has no rows to gather along");
        }
        dims[0] *= self.world;
        Ok((
            CudaStorage::wrap_cuda_slice(recv, device),
            Shape::from_dims(&dims),
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

/// `ncclBroadcast` from `root` to every rank.
struct NcclBroadcast<'a> {
    comm: &'a Comm,
    root: i32,
}

impl NcclBroadcast<'_> {
    fn typed<T>(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)>
    where
        T: CudaDType + NcclType + DeviceRepr + ValidAsZeroBits,
    {
        let send = send_view!("broadcast", storage, layout, T);
        let device = storage.device().clone();
        let mut recv = device.alloc_zeros::<T>(layout.shape().elem_count())?;
        self.comm
            .broadcast(Some(&send), &mut recv, self.root)
            .map_err(|e| candle_core::Error::Cuda(format!("nccl status {:?}", e.0).into()))?;
        Ok((
            CudaStorage::wrap_cuda_slice(recv, device),
            layout.shape().clone(),
        ))
    }
}

impl CustomOp1 for NcclBroadcast<'_> {
    fn name(&self) -> &'static str {
        "nccl-broadcast"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("nccl-broadcast is a CUDA collective; it has no CPU implementation")
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
                refuse_unsupported_dtype("broadcast", other)?;
                unreachable!("refuse_unsupported_dtype always returns Err")
            }
        }
    }
}
