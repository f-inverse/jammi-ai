//! FlashAttention-2 varlen forward/backward FFI — feature `flash-attn`.
//!
//! The Rust side of `third_party/flash-attention/jammi/flash_api_jammi.{h,cu}`,
//! jammi's torch-free C wrapper over the vendored upstream kernels
//! (`third_party/flash-attention/src/flash_{fwd,bwd}_hdim64_bf16_sm80.cu`,
//! tag v2.8.3.post1; provenance in `third_party/flash-attention/VENDORED.md`).
//! No `KernelOp` lives here: this module is the kernel BOUNDARY (raw
//! device buffers in, raw device buffers out) that a later op composes.
//!
//! # Layout served (the qkv-packed varlen case)
//!
//! | buffer | dtype | shape | role |
//! |---|---|---|---|
//! | `qkv` | bf16 | `[total_q, 3, H, 64]` | input; q/k/v are the three `[H, 64]` slabs of each row |
//! | `cu_seqlens` | i32 | `[B + 1]` | cumulative sequence starts, `cu[0] = 0`, `cu[B] = total_q` |
//! | `o` | bf16 | `[total_q, H, 64]` | forward output |
//! | `lse` | f32 | `[H, total_q]` | forward logsumexp per (head, row), natural log of the SCALED scores |
//! | `d_o` | bf16 | `[total_q, H, 64]` | backward input |
//! | `d_qkv` | bf16 | `[total_q, 3, H, 64]` | backward output, written in place with the packed strides |
//! | `softmax_d` | f32 | `[H, total_q + 128·B]` | backward scratch (`BwdScratch`) |
//! | `dq_accum` | f32 | `[splits, total_q + 128·B, H, 64]` | backward scratch (`BwdScratch`) |
//!
//! The `128·B` padding rows exist because the backward atomically
//! accumulates up to one 128-row block past each sequence's end rather
//! than bounds-checking (upstream `flash_api.cpp:1103-1111`); the i-th
//! sequence's rows live at `cu[i] + 128·i ..` in both scratch buffers
//! (`flash_bwd_kernel.h:120-126`).
//!
//! # Determinism (the recompute-soundness invariant)
//!
//! The forward has no atomics and no split-KV path reachable from this ABI
//! (the split kernel's translation unit is not compiled; the wrapper
//! asserts `num_splits <= 1`), and dropout is compiled out AND refused —
//! so two forwards on identical inputs are BIT-IDENTICAL in `o` and `lse`.
//! The backward is deterministic iff [`VarlenConfig::deterministic`]: it
//! then gives every (batch, head) its own `ceil(num_SM / (B·H))` private
//! `dq_accum` split buffers (no cross-block atomics on shared rows) and
//! reduces them in a fixed order (`flash_bwd_preprocess_kernel.h:243`);
//! those buffers MUST be zero on entry, which [`BwdScratch::alloc`] does.
//! With `deterministic = false` the kernel clears `dq_accum` itself and
//! accumulates with atomics — cheaper, order-dependent.
//!
//! # Refusal lattice
//!
//! Every predicate below is a cell with one test (`tests/flash_smoke.rs`
//! for the C-side cells, this module's unit tests for the pure ones).
//!
//! Rust side (before any FFI call):
//!
//! | predicate | outcome |
//! |---|---|
//! | `total_q == 0` / `batch == 0` / `num_heads == 0` / `max_seqlen == 0` | `FlashError::Geometry` |
//! | `max_seqlen > total_q` | `FlashError::Geometry` |
//! | any shape product `> i32::MAX` | `FlashError::Geometry` |
//! | `softmax_scale` not finite or `<= 0` | `FlashError::Geometry` |
//! | `window > i32::MAX` | `FlashError::Geometry` |
//! | a buffer's element count `!=` the shape's | `FlashError::Geometry` |
//! | `dq_accum` split count `!=` [`dq_accum_splits`] | `FlashError::Geometry` |
//! | args struct size `!=` the C struct's | `FlashError::Refused(Abi)` |
//!
//! C side (`flash_api_jammi.cu`, every upstream `TORCH_CHECK` that applies),
//! each mapped to a [`FlashStatus`]: `NullPointer`, `HeadDim`, `Dims`,
//! `DropoutUnsupported`, `CausalUnsupported`, `Scale`, `BufferLen`,
//! `ComputeCapability`, `Cuda`, `SplitKernel`, `Abi`, `DqAccumSplits`,
//! `Window`. The safe API cannot produce `DropoutUnsupported` (it has no
//! dropout parameter at all — the type system is the Rust-boundary
//! refusal) or `HeadDim` (64 is a constant); the smoke test drives those
//! two through [`raw`] to prove the C side refuses them on its own.
//!
//! # Window semantics
//!
//! `window = Some(w)` maps to upstream `window_size = (w, w)`: query row
//! `r` attends keys `c` with `r − w <= c <= r + w` within its own sequence
//! (`mask.h:172-173,194`) — a symmetric radius, `2·w + 1` keys. `None` maps
//! to `(-1, -1)` (every key in the sequence). A radius `>= max_seqlen` is
//! the same as `None` (upstream `flash_api.cpp:608-609`). A masked score is
//! `-INFINITY` in FA2, which `exp` takes to exactly `0.0` in f32.
//!
//! # Synchronisation
//!
//! Every call is asynchronous on the device's stream
//! (`CudaDevice::cuda_stream`); the cudarc `SyncOnDrop` guards from
//! `device_ptr`/`device_ptr_mut` record the read/write events so later
//! candle ops on the same buffers order correctly. Nothing here reads
//! device memory on the host: `total_q` and `max_seqlen` are host inputs
//! ([`VarlenGeometry`]), derived by the caller from the same lengths that
//! built `cu_seqlens`.

use std::ffi::{c_void, CStr};

use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut, DriverError,
};
use candle_core::CudaDevice;
use half::bf16;
use thiserror::Error;

/// The `extern "C"` surface of `flash_api_jammi.h`, verbatim. Public so the
/// smoke test can drive refusals the safe API makes unrepresentable
/// (`p_dropout != 0`, `head_dim != 64`, a wrong `struct_size`).
///
/// # Safety
///
/// Every pointer must be a device pointer valid on the given stream for
/// the element count in its `*_len`; the safe API above is the sanctioned
/// way to build these.
pub mod raw {
    use std::ffi::{c_char, c_void};

    /// Mirror of `jammi_flash_varlen_fwd_args`. Field order and types are
    /// the header's; `struct_size` must be `size_of::<FwdArgs>()`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct FwdArgs {
        pub qkv: *const c_void,
        pub o: *mut c_void,
        pub softmax_lse: *mut f32,
        pub cu_seqlens: *const i32,
        pub stream: *mut c_void,
        pub qkv_len: i64,
        pub o_len: i64,
        pub softmax_lse_len: i64,
        pub cu_seqlens_len: i64,
        pub struct_size: i32,
        pub total_q: i32,
        pub batch: i32,
        pub num_heads: i32,
        pub head_dim: i32,
        pub max_seqlen: i32,
        pub window_size_left: i32,
        pub window_size_right: i32,
        pub softmax_scale: f32,
        pub p_dropout: f32,
    }

    /// Mirror of `jammi_flash_varlen_bwd_args`.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct BwdArgs {
        pub qkv: *const c_void,
        pub o: *const c_void,
        pub softmax_lse: *const f32,
        pub d_o: *const c_void,
        pub d_qkv: *mut c_void,
        pub softmax_d: *mut f32,
        pub dq_accum: *mut f32,
        pub cu_seqlens: *const i32,
        pub stream: *mut c_void,
        pub qkv_len: i64,
        pub o_len: i64,
        pub softmax_lse_len: i64,
        pub d_o_len: i64,
        pub d_qkv_len: i64,
        pub softmax_d_len: i64,
        pub dq_accum_len: i64,
        pub cu_seqlens_len: i64,
        pub struct_size: i32,
        pub total_q: i32,
        pub batch: i32,
        pub num_heads: i32,
        pub head_dim: i32,
        pub max_seqlen: i32,
        pub window_size_left: i32,
        pub window_size_right: i32,
        pub softmax_scale: f32,
        pub p_dropout: f32,
        pub deterministic: i32,
        pub dq_accum_splits: i32,
    }

    unsafe extern "C" {
        pub fn jammi_flash_varlen_fwd(args: *const FwdArgs) -> i32;
        pub fn jammi_flash_varlen_bwd(args: *const BwdArgs) -> i32;
        pub fn jammi_flash_dq_accum_splits(
            batch: i32,
            num_heads: i32,
            deterministic: i32,
            out_splits: *mut i32,
        ) -> i32;
        pub fn jammi_flash_num_sms(out_num_sms: *mut i32) -> i32;
        pub fn jammi_flash_strerror(status: i32) -> *const c_char;
        pub fn jammi_flash_sizeof_fwd_args() -> usize;
        pub fn jammi_flash_sizeof_bwd_args() -> usize;
    }
}

/// Head dimension the vendored kernels are instantiated for (the only one
/// compiled: `run_mha_{fwd,bwd}_<bf16, 64, false>`).
pub const HEAD_DIM: usize = 64;

/// Rows of padding per sequence the backward scratch carries
/// (`flash_api.cpp:1100,1106`: "128 is the max block size on the seqlen_q
/// dimension").
pub const SEQLEN_Q_PAD: usize = 128;

/// The C wrapper's non-zero status codes (`jammi_flash_status` in
/// `flash_api_jammi.h`), one variant per code.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlashStatus {
    /// A required device pointer is NULL.
    NullPointer = 1,
    /// `head_dim != 64`.
    HeadDim = 2,
    /// A dimension is `<= 0`, `max_seqlen > total_q`, or an offset overflows i32.
    Dims = 3,
    /// `p_dropout != 0.0` — dropout is compiled out and would be silently ignored.
    DropoutUnsupported = 4,
    /// `(window_size_left < 0, window_size_right == 0)` selects the (uncompiled) causal kernel.
    CausalUnsupported = 5,
    /// `softmax_scale` not finite or `<= 0`.
    Scale = 6,
    /// A buffer's element count differs from the count its shape implies.
    BufferLen = 7,
    /// Device compute capability major `< 8`.
    ComputeCapability = 8,
    /// A CUDA runtime call failed.
    Cuda = 9,
    /// Internal: `num_splits > 1` (the split-KV path) — unreachable by construction.
    SplitKernel = 10,
    /// Args struct size mismatch between Rust and C.
    Abi = 11,
    /// `dq_accum_splits` differs from the device-derived split count.
    DqAccumSplits = 12,
    /// A window size below `-1`.
    Window = 13,
}

impl FlashStatus {
    /// The variant for a wrapper status code; `None` for `0` (ok) and for
    /// codes this build does not know.
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            1 => Self::NullPointer,
            2 => Self::HeadDim,
            3 => Self::Dims,
            4 => Self::DropoutUnsupported,
            5 => Self::CausalUnsupported,
            6 => Self::Scale,
            7 => Self::BufferLen,
            8 => Self::ComputeCapability,
            9 => Self::Cuda,
            10 => Self::SplitKernel,
            11 => Self::Abi,
            12 => Self::DqAccumSplits,
            13 => Self::Window,
            _ => return None,
        })
    }

    /// The wrapper's static message for this status.
    pub fn message(self) -> String {
        strerror(self as i32)
    }
}

/// Errors of the safe API.
#[derive(Debug, Error)]
pub enum FlashError {
    /// The C wrapper refused the call with a known status (its message is
    /// the wrapper's own static text).
    #[error("flash-attn wrapper refused ({status:?}, code {code}): {message}")]
    Refused {
        status: FlashStatus,
        code: i32,
        message: String,
    },
    /// The C wrapper returned a code this build does not know.
    #[error("flash-attn wrapper returned unknown status {code}: {message}")]
    UnknownStatus { code: i32, message: String },
    /// A Rust-side domain refusal: geometry, config, or buffer length.
    #[error("flash-attn: {0}")]
    Geometry(String),
    /// A cudarc driver call (context binding) failed.
    #[error("flash-attn: CUDA driver error: {0}")]
    Driver(#[from] DriverError),
    /// A candle device call (allocation) failed.
    #[error("flash-attn: {0}")]
    Candle(#[from] candle_core::Error),
}

impl FlashError {
    /// The wrapper status behind a [`FlashError::Refused`], else `None`.
    pub fn status(&self) -> Option<FlashStatus> {
        match self {
            Self::Refused { status, .. } => Some(*status),
            _ => None,
        }
    }
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, FlashError>;

fn strerror(code: i32) -> String {
    // SAFETY: `jammi_flash_strerror` returns a pointer to a static,
    // NUL-terminated C string for every input (unknown codes map to a
    // fixed string); it never returns NULL.
    let p = unsafe { raw::jammi_flash_strerror(code) };
    if p.is_null() {
        return "<null message>".to_string();
    }
    // SAFETY: non-null static NUL-terminated string, see above.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Maps a wrapper status code to `Ok(())` / the typed error.
pub fn check_status(code: i32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let message = strerror(code);
    Err(match FlashStatus::from_code(code) {
        Some(status) => FlashError::Refused {
            status,
            code,
            message,
        },
        None => FlashError::UnknownStatus { code, message },
    })
}

/// The `#[repr(C)]` mirrors must be exactly the C structs' size; checked
/// on every call (one FFI call returning a constant).
fn check_abi() -> Result<()> {
    // SAFETY: pure functions returning `sizeof` constants.
    let (fwd, bwd) = unsafe {
        (
            raw::jammi_flash_sizeof_fwd_args(),
            raw::jammi_flash_sizeof_bwd_args(),
        )
    };
    if fwd != std::mem::size_of::<raw::FwdArgs>() || bwd != std::mem::size_of::<raw::BwdArgs>() {
        let code = FlashStatus::Abi as i32;
        return Err(FlashError::Refused {
            status: FlashStatus::Abi,
            code,
            message: format!(
                "{} (fwd: C {fwd} vs Rust {}; bwd: C {bwd} vs Rust {})",
                strerror(code),
                std::mem::size_of::<raw::FwdArgs>(),
                std::mem::size_of::<raw::BwdArgs>()
            ),
        });
    }
    Ok(())
}

/// The batch's shape, host-side. `total_q` and `max_seqlen` are derived
/// by the caller from the same per-sequence lengths that built
/// `cu_seqlens` — nothing here reads `cu_seqlens` back from the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarlenGeometry {
    /// Sum of the sequence lengths (`cu_seqlens[batch]`).
    pub total_q: usize,
    /// Number of sequences (`cu_seqlens.len() - 1`).
    pub batch: usize,
    /// Attention heads (q, k and v share it — no GQA in this layout).
    pub num_heads: usize,
    /// The longest sequence in the batch. Sizes the kernel grid; a value
    /// below the true maximum would leave that sequence's tail rows
    /// unvisited, and cannot be checked without a device read — it is the
    /// caller's contract.
    pub max_seqlen: usize,
}

impl VarlenGeometry {
    /// `total_q · 3 · H · 64`.
    pub fn qkv_len(&self) -> usize {
        self.total_q * 3 * self.num_heads * HEAD_DIM
    }
    /// `total_q · H · 64` (also `d_o`).
    pub fn o_len(&self) -> usize {
        self.total_q * self.num_heads * HEAD_DIM
    }
    /// `H · total_q`.
    pub fn lse_len(&self) -> usize {
        self.num_heads * self.total_q
    }
    /// `batch + 1`.
    pub fn cu_seqlens_len(&self) -> usize {
        self.batch + 1
    }
    /// `total_q + 128 · batch` — the padded row count of both backward
    /// scratch buffers.
    pub fn rows_padded(&self) -> usize {
        self.total_q + SEQLEN_Q_PAD * self.batch
    }
    /// `H · rows_padded`.
    pub fn softmax_d_len(&self) -> usize {
        self.num_heads * self.rows_padded()
    }
    /// `splits · rows_padded · H · 64`.
    pub fn dq_accum_len(&self, splits: usize) -> usize {
        splits * self.rows_padded() * self.num_heads * HEAD_DIM
    }

    /// The pure refusal cells of the lattice in the module doc.
    pub fn validate(&self) -> Result<()> {
        if self.total_q == 0 || self.batch == 0 || self.num_heads == 0 || self.max_seqlen == 0 {
            return Err(FlashError::Geometry(format!(
                "every dimension must be > 0: {self:?}"
            )));
        }
        if self.max_seqlen > self.total_q {
            return Err(FlashError::Geometry(format!(
                "max_seqlen ({}) exceeds total_q ({})",
                self.max_seqlen, self.total_q
            )));
        }
        // The kernels index with 32-bit offsets (flash.h:165-166); the
        // largest offsets are into dq_accum (one split) and into qkv.
        let i32_max = i32::MAX as u128;
        let dq_accum_one_split =
            (self.rows_padded() as u128) * (self.num_heads as u128) * (HEAD_DIM as u128);
        let qkv = (self.total_q as u128) * 3 * (self.num_heads as u128) * (HEAD_DIM as u128);
        if dq_accum_one_split > i32_max || qkv > i32_max {
            return Err(FlashError::Geometry(format!(
                "shape offsets exceed the kernels' int32 index budget: {self:?}"
            )));
        }
        Ok(())
    }
}

/// Per-call configuration. There is deliberately NO dropout field: the
/// build compiles dropout out, so the only honest value is 0 and the type
/// makes any other unrepresentable (the C wrapper additionally refuses a
/// non-zero `p_dropout` for callers of [`raw`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VarlenConfig {
    /// Multiplier on `q·k` before the softmax (e.g. `1/sqrt(64)`); finite, `> 0`.
    pub softmax_scale: f32,
    /// Symmetric window radius `w` (keys `r − w ..= r + w`), `None` = all keys.
    pub window: Option<u32>,
    /// Deterministic backward (see the module doc).
    pub deterministic: bool,
}

impl VarlenConfig {
    /// `(window_size_left, window_size_right)` as upstream takes them.
    pub fn window_sizes(&self) -> Result<(i32, i32)> {
        match self.window {
            None => Ok((-1, -1)),
            Some(w) => {
                let w = i32::try_from(w).map_err(|_| {
                    FlashError::Geometry(format!("window radius {w} exceeds i32::MAX"))
                })?;
                Ok((w, w))
            }
        }
    }

    /// The pure refusal cells of the lattice in the module doc.
    pub fn validate(&self) -> Result<()> {
        if !self.softmax_scale.is_finite() || self.softmax_scale <= 0.0 {
            return Err(FlashError::Geometry(format!(
                "softmax_scale must be finite and > 0, got {}",
                self.softmax_scale
            )));
        }
        self.window_sizes().map(|_| ())
    }
}

fn check_len(name: &str, got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(FlashError::Geometry(format!(
            "{name}: buffer holds {got} elements, the shape needs exactly {expected}"
        )));
    }
    Ok(())
}

fn as_i32(name: &str, v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| FlashError::Geometry(format!("{name} = {v} exceeds i32::MAX")))
}

fn bind(dev: &CudaDevice) -> Result<()> {
    // The C wrapper uses the CUDA runtime API (cudaGetDevice, kernel
    // launches), which resolves the CURRENT driver context of the calling
    // thread; candle binds it on device creation, but not on every thread
    // that later holds a `CudaDevice` clone (test threads, a rayon pool).
    dev.cuda_stream().context().bind_to_thread()?;
    Ok(())
}

/// The current device's multiprocessor count.
pub fn num_sms(dev: &CudaDevice) -> Result<usize> {
    bind(dev)?;
    let mut n: i32 = 0;
    // SAFETY: `n` is a valid out-pointer for the call's duration.
    check_status(unsafe { raw::jammi_flash_num_sms(&mut n) })?;
    Ok(n as usize)
}

/// The `dq_accum` split count the backward uses for this (batch, heads,
/// deterministic) on the current device: `1` when not deterministic, else
/// `ceil(num_SM / (batch · num_heads))` (`flash_api.cpp:1115`,
/// `flash_bwd_launch_template.h:78-81`).
pub fn dq_accum_splits(
    dev: &CudaDevice,
    batch: usize,
    num_heads: usize,
    deterministic: bool,
) -> Result<usize> {
    bind(dev)?;
    let mut n: i32 = 0;
    let code = unsafe {
        // SAFETY: `n` is a valid out-pointer for the call's duration.
        raw::jammi_flash_dq_accum_splits(
            as_i32("batch", batch)?,
            as_i32("num_heads", num_heads)?,
            deterministic as i32,
            &mut n,
        )
    };
    check_status(code)?;
    Ok(n as usize)
}

/// Backward scratch, allocated per the module-doc table.
pub struct BwdScratch {
    /// f32 `[H, total_q + 128·B]`, uninitialised (the kernel writes every
    /// row it reads).
    pub softmax_d: CudaSlice<f32>,
    /// f32 `[splits, total_q + 128·B, H, 64]`: ZEROED when deterministic
    /// (the kernel accumulates into it without clearing,
    /// `flash_bwd_launch_template.h:84-88`), uninitialised otherwise (the
    /// kernel clears its own rows first).
    pub dq_accum: CudaSlice<f32>,
    /// The split count `dq_accum` was sized with.
    pub splits: usize,
}

impl BwdScratch {
    /// Allocates both buffers for `geom` on `dev`.
    pub fn alloc(dev: &CudaDevice, geom: &VarlenGeometry, deterministic: bool) -> Result<Self> {
        geom.validate()?;
        let splits = dq_accum_splits(dev, geom.batch, geom.num_heads, deterministic)?;
        // SAFETY: uninitialised device memory that the kernel fully
        // overwrites before reading (compute_dot_do_o writes dP_sum for
        // every row of every block it visits).
        let softmax_d = unsafe { dev.alloc::<f32>(geom.softmax_d_len()) }?;
        let n = geom.dq_accum_len(splits);
        let dq_accum = if deterministic {
            dev.alloc_zeros::<f32>(n)?
        } else {
            // SAFETY: as above — `flash_bwd_dot_do_o_kernel<Clear_dQaccum=true>`
            // zeroes every row before the accumulation kernel runs.
            unsafe { dev.alloc::<f32>(n) }?
        };
        Ok(Self {
            softmax_d,
            dq_accum,
            splits,
        })
    }
}

/// Forward into caller-provided `o` / `lse` views (sizes checked exactly).
pub fn flash_varlen_fwd_into(
    dev: &CudaDevice,
    qkv: CudaView<'_, bf16>,
    cu_seqlens: CudaView<'_, i32>,
    mut o: CudaViewMut<'_, bf16>,
    mut lse: CudaViewMut<'_, f32>,
    geom: &VarlenGeometry,
    cfg: &VarlenConfig,
) -> Result<()> {
    geom.validate()?;
    cfg.validate()?;
    check_len("qkv", qkv.len(), geom.qkv_len())?;
    check_len("cu_seqlens", cu_seqlens.len(), geom.cu_seqlens_len())?;
    check_len("o", o.len(), geom.o_len())?;
    check_len("lse", lse.len(), geom.lse_len())?;
    let (window_size_left, window_size_right) = cfg.window_sizes()?;
    bind(dev)?;
    check_abi()?;

    let stream = dev.cuda_stream();
    let (qkv_p, _g_qkv) = qkv.device_ptr(&stream);
    let (cu_p, _g_cu) = cu_seqlens.device_ptr(&stream);
    let (o_p, _g_o) = o.device_ptr_mut(&stream);
    let (lse_p, _g_lse) = lse.device_ptr_mut(&stream);
    let args = raw::FwdArgs {
        qkv: qkv_p as usize as *const c_void,
        o: o_p as usize as *mut c_void,
        softmax_lse: lse_p as usize as *mut f32,
        cu_seqlens: cu_p as usize as *const i32,
        stream: stream.cu_stream() as *mut c_void,
        qkv_len: geom.qkv_len() as i64,
        o_len: geom.o_len() as i64,
        softmax_lse_len: geom.lse_len() as i64,
        cu_seqlens_len: geom.cu_seqlens_len() as i64,
        struct_size: std::mem::size_of::<raw::FwdArgs>() as i32,
        total_q: as_i32("total_q", geom.total_q)?,
        batch: as_i32("batch", geom.batch)?,
        num_heads: as_i32("num_heads", geom.num_heads)?,
        head_dim: HEAD_DIM as i32,
        max_seqlen: as_i32("max_seqlen", geom.max_seqlen)?,
        window_size_left,
        window_size_right,
        softmax_scale: cfg.softmax_scale,
        p_dropout: 0.0,
    };
    // SAFETY: every pointer comes from a live cudarc view whose element
    // count was just checked against the shape; the guards above keep the
    // views' events recorded on the stream until the launch is enqueued.
    let code = unsafe { raw::jammi_flash_varlen_fwd(&args) };
    check_status(code)
}

/// Forward: allocates and returns `(o, lse)`.
pub fn flash_varlen_fwd(
    dev: &CudaDevice,
    qkv: &CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    geom: &VarlenGeometry,
    cfg: &VarlenConfig,
) -> Result<(CudaSlice<bf16>, CudaSlice<f32>)> {
    geom.validate()?;
    // SAFETY: uninitialised outputs the kernel fully overwrites (every
    // (row, head) of `o` and `lse` belongs to exactly one launched block).
    let mut o = unsafe { dev.alloc::<bf16>(geom.o_len()) }?;
    let mut lse = unsafe { dev.alloc::<f32>(geom.lse_len()) }?;
    flash_varlen_fwd_into(
        dev,
        qkv.as_view(),
        cu_seqlens.as_view(),
        o.as_view_mut(),
        lse.as_view_mut(),
        geom,
        cfg,
    )?;
    Ok((o, lse))
}

/// The backward's buffers, as views (see the module-doc table).
pub struct BwdBuffers<'a> {
    pub qkv: CudaView<'a, bf16>,
    pub cu_seqlens: CudaView<'a, i32>,
    pub o: CudaView<'a, bf16>,
    pub lse: CudaView<'a, f32>,
    pub d_o: CudaView<'a, bf16>,
    pub d_qkv: CudaViewMut<'a, bf16>,
    pub softmax_d: CudaViewMut<'a, f32>,
    /// Must be all-zero when `cfg.deterministic`.
    pub dq_accum: CudaViewMut<'a, f32>,
    /// The split count `dq_accum` was sized with ([`dq_accum_splits`]).
    pub dq_accum_splits: usize,
}

/// Backward into caller-provided buffers (sizes checked exactly).
pub fn flash_varlen_bwd_into(
    dev: &CudaDevice,
    bufs: BwdBuffers<'_>,
    geom: &VarlenGeometry,
    cfg: &VarlenConfig,
) -> Result<()> {
    geom.validate()?;
    cfg.validate()?;
    let BwdBuffers {
        qkv,
        cu_seqlens,
        o,
        lse,
        d_o,
        mut d_qkv,
        mut softmax_d,
        mut dq_accum,
        dq_accum_splits: splits,
    } = bufs;
    check_len("qkv", qkv.len(), geom.qkv_len())?;
    check_len("cu_seqlens", cu_seqlens.len(), geom.cu_seqlens_len())?;
    check_len("o", o.len(), geom.o_len())?;
    check_len("lse", lse.len(), geom.lse_len())?;
    check_len("d_o", d_o.len(), geom.o_len())?;
    check_len("d_qkv", d_qkv.len(), geom.qkv_len())?;
    check_len("softmax_d", softmax_d.len(), geom.softmax_d_len())?;
    let expected_splits = dq_accum_splits(dev, geom.batch, geom.num_heads, cfg.deterministic)?;
    if splits != expected_splits {
        return Err(FlashError::Geometry(format!(
            "dq_accum was sized for {splits} split(s) but this device/config uses {expected_splits}"
        )));
    }
    check_len("dq_accum", dq_accum.len(), geom.dq_accum_len(splits))?;
    let (window_size_left, window_size_right) = cfg.window_sizes()?;
    bind(dev)?;
    check_abi()?;

    let stream = dev.cuda_stream();
    let (qkv_p, _g_qkv) = qkv.device_ptr(&stream);
    let (cu_p, _g_cu) = cu_seqlens.device_ptr(&stream);
    let (o_p, _g_o) = o.device_ptr(&stream);
    let (lse_p, _g_lse) = lse.device_ptr(&stream);
    let (do_p, _g_do) = d_o.device_ptr(&stream);
    let (dqkv_p, _g_dqkv) = d_qkv.device_ptr_mut(&stream);
    let (sd_p, _g_sd) = softmax_d.device_ptr_mut(&stream);
    let (dqa_p, _g_dqa) = dq_accum.device_ptr_mut(&stream);
    let args = raw::BwdArgs {
        qkv: qkv_p as usize as *const c_void,
        o: o_p as usize as *const c_void,
        softmax_lse: lse_p as usize as *const f32,
        d_o: do_p as usize as *const c_void,
        d_qkv: dqkv_p as usize as *mut c_void,
        softmax_d: sd_p as usize as *mut f32,
        dq_accum: dqa_p as usize as *mut f32,
        cu_seqlens: cu_p as usize as *const i32,
        stream: stream.cu_stream() as *mut c_void,
        qkv_len: geom.qkv_len() as i64,
        o_len: geom.o_len() as i64,
        softmax_lse_len: geom.lse_len() as i64,
        d_o_len: geom.o_len() as i64,
        d_qkv_len: geom.qkv_len() as i64,
        softmax_d_len: geom.softmax_d_len() as i64,
        dq_accum_len: geom.dq_accum_len(splits) as i64,
        cu_seqlens_len: geom.cu_seqlens_len() as i64,
        struct_size: std::mem::size_of::<raw::BwdArgs>() as i32,
        total_q: as_i32("total_q", geom.total_q)?,
        batch: as_i32("batch", geom.batch)?,
        num_heads: as_i32("num_heads", geom.num_heads)?,
        head_dim: HEAD_DIM as i32,
        max_seqlen: as_i32("max_seqlen", geom.max_seqlen)?,
        window_size_left,
        window_size_right,
        softmax_scale: cfg.softmax_scale,
        p_dropout: 0.0,
        deterministic: cfg.deterministic as i32,
        dq_accum_splits: as_i32("dq_accum_splits", splits)?,
    };
    // SAFETY: as in `flash_varlen_fwd_into`.
    let code = unsafe { raw::jammi_flash_varlen_bwd(&args) };
    check_status(code)
}

/// Backward: allocates the scratch ([`BwdScratch::alloc`]) and `d_qkv`,
/// returns `d_qkv` bf16 `[total_q, 3, H, 64]`.
///
/// Eight parameters: the five device buffers are the backward's actual
/// arity (upstream `mha_varlen_bwd` takes the same five plus cu_seqlens);
/// folding them into a struct would only move the same names one level
/// down — [`flash_varlen_bwd_into`] is that shape for callers who own
/// the buffers.
#[allow(clippy::too_many_arguments)]
pub fn flash_varlen_bwd(
    dev: &CudaDevice,
    qkv: &CudaSlice<bf16>,
    cu_seqlens: &CudaSlice<i32>,
    o: &CudaSlice<bf16>,
    lse: &CudaSlice<f32>,
    d_o: &CudaSlice<bf16>,
    geom: &VarlenGeometry,
    cfg: &VarlenConfig,
) -> Result<CudaSlice<bf16>> {
    geom.validate()?;
    let mut scratch = BwdScratch::alloc(dev, geom, cfg.deterministic)?;
    // SAFETY: uninitialised output the kernel fully overwrites (dq via
    // convert_dQ over every row block, dk/dv over every key block).
    let mut d_qkv = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }?;
    flash_varlen_bwd_into(
        dev,
        BwdBuffers {
            qkv: qkv.as_view(),
            cu_seqlens: cu_seqlens.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: d_o.as_view(),
            d_qkv: d_qkv.as_view_mut(),
            softmax_d: scratch.softmax_d.as_view_mut(),
            dq_accum: scratch.dq_accum.as_view_mut(),
            dq_accum_splits: scratch.splits,
        },
        geom,
        cfg,
    )?;
    Ok(d_qkv)
}

#[cfg(test)]
mod tests {
    //! The pure cells of the refusal lattice (no device needed).
    use super::*;

    fn geom() -> VarlenGeometry {
        VarlenGeometry {
            total_q: 21,
            batch: 3,
            num_heads: 2,
            max_seqlen: 9,
        }
    }

    #[test]
    fn geometry_lengths_follow_the_module_table() {
        let g = geom();
        assert_eq!(g.qkv_len(), 21 * 3 * 2 * 64);
        assert_eq!(g.o_len(), 21 * 2 * 64);
        assert_eq!(g.lse_len(), 2 * 21);
        assert_eq!(g.cu_seqlens_len(), 4);
        assert_eq!(g.rows_padded(), 21 + 128 * 3);
        assert_eq!(g.softmax_d_len(), 2 * (21 + 384));
        assert_eq!(g.dq_accum_len(1), (21 + 384) * 2 * 64);
        assert_eq!(g.dq_accum_len(4), 4 * (21 + 384) * 2 * 64);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn geometry_refuses_each_zero_dimension() {
        for (i, g) in [
            VarlenGeometry {
                total_q: 0,
                ..geom()
            },
            VarlenGeometry { batch: 0, ..geom() },
            VarlenGeometry {
                num_heads: 0,
                ..geom()
            },
            VarlenGeometry {
                max_seqlen: 0,
                ..geom()
            },
        ]
        .iter()
        .enumerate()
        {
            assert!(
                matches!(g.validate(), Err(FlashError::Geometry(_))),
                "cell {i}: {g:?}"
            );
        }
    }

    #[test]
    fn geometry_refuses_max_seqlen_above_total_q() {
        let g = VarlenGeometry {
            max_seqlen: 22,
            ..geom()
        };
        assert!(matches!(g.validate(), Err(FlashError::Geometry(_))));
        // The boundary itself (one sequence of the whole batch) is legal.
        let g = VarlenGeometry {
            max_seqlen: 21,
            ..geom()
        };
        assert!(g.validate().is_ok());
    }

    #[test]
    fn geometry_refuses_int32_offset_overflow() {
        // qkv offsets: total_q * 3 * H * 64 > i32::MAX at H = 1 once
        // total_q > 11_184_810; dq_accum offsets overflow slightly later
        // (rows_padded * H * 64), so the qkv product is the binding cell.
        let g = VarlenGeometry {
            total_q: 11_184_811,
            batch: 1,
            num_heads: 1,
            max_seqlen: 11_184_811,
        };
        assert!(matches!(g.validate(), Err(FlashError::Geometry(_))));
        let g = VarlenGeometry {
            total_q: 11_184_810,
            max_seqlen: 11_184_810,
            ..g
        };
        assert!(g.validate().is_ok(), "{:?}", g.validate().err());
        // And a dq_accum-bound cell: the 128·B padding rows push it over
        // while qkv (which has no padding rows) stays under.
        // rows_padded = 1_000 + 128 · 262_144 = 33_555_432; × 64 =
        // 2_147_547_648 > i32::MAX, while qkv = 1_000 · 192 stays tiny.
        let g = VarlenGeometry {
            total_q: 1_000,
            batch: 262_144,
            num_heads: 1,
            max_seqlen: 1,
        };
        assert!(matches!(g.validate(), Err(FlashError::Geometry(_))));
    }

    #[test]
    fn config_window_maps_to_symmetric_pair_and_refuses_bad_scale() {
        let ok = VarlenConfig {
            softmax_scale: 0.125,
            window: Some(2),
            deterministic: true,
        };
        assert_eq!(ok.window_sizes().unwrap(), (2, 2));
        assert!(ok.validate().is_ok());
        let none = VarlenConfig { window: None, ..ok };
        assert_eq!(none.window_sizes().unwrap(), (-1, -1));
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let c = VarlenConfig {
                softmax_scale: bad,
                ..ok
            };
            assert!(
                matches!(c.validate(), Err(FlashError::Geometry(_))),
                "scale {bad}"
            );
        }
        let huge = VarlenConfig {
            window: Some(u32::MAX),
            ..ok
        };
        assert!(matches!(huge.validate(), Err(FlashError::Geometry(_))));
    }

    #[test]
    fn status_codes_round_trip_and_zero_is_ok() {
        for code in 1..=13 {
            let s = FlashStatus::from_code(code).expect("known code");
            assert_eq!(s as i32, code);
        }
        assert!(FlashStatus::from_code(0).is_none());
        assert!(FlashStatus::from_code(14).is_none());
        assert!(FlashStatus::from_code(-1).is_none());
    }

    #[test]
    fn args_struct_sizes_match_the_c_header() {
        // 5 pointers + 4 i64 + 10 × 4-byte = 40 + 32 + 40 = 112;
        // 9 pointers + 8 i64 + 12 × 4-byte = 72 + 64 + 48 = 184.
        assert_eq!(std::mem::size_of::<raw::FwdArgs>(), 112);
        assert_eq!(std::mem::size_of::<raw::BwdArgs>(), 184);
        // The linked library agrees (this is the same check the safe API
        // makes on every call).
        check_abi().unwrap();
    }
}
