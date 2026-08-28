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
//! The backward is deterministic iff `VarlenConfig::deterministic`: it
//! then gives every (batch, head) its own `ceil(num_SM / (B·H))` private
//! `dq_accum` split buffers (no cross-block atomics on shared rows) and
//! reduces them in a fixed order (`flash_bwd_preprocess_kernel.h:243`);
//! those buffers MUST be zero on entry, which `BwdScratch::alloc` does.
//! With `deterministic = false` the kernel clears `dq_accum` itself and
//! accumulates with atomics — cheaper, order-dependent.
//!
//! # Refusal lattice
//!
//! Every predicate reachable from this crate's own inputs on the SUPPORTED
//! hardware (an Ampere-or-newer GPU that this build's own driver can open)
//! is a cell with one test (`tests/flash_smoke.rs` for the C-side cells,
//! this module's unit tests for the pure ones). Five cells below are
//! UNREACHABLE by construction under that scope and are documented, not
//! tested, rather than claimed as covered:
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
//! | `dq_accum` split count `!=` `dq_accum_splits` | `FlashError::Geometry` |
//! | args struct size `!=` the C struct's | `FlashError::Refused(Abi)` |
//! | device compute capability not a MEMBER of the build's compiled arch set (`check_arch`) | `FlashError::Arch` |
//! | a sequence length `== 0`, or the batch is empty (`CuSeqlens::from_lengths`) | `FlashError::Geometry` |
//! | `total_q`/per-length `> i32::MAX` (`CuSeqlens::from_lengths`) | `FlashError::Geometry` |
//!
//! C side (`flash_api_jammi.cu`, every upstream `TORCH_CHECK` that applies),
//! each mapped to a `FlashStatus`: `NullPointer`, `HeadDim`, `Dims`,
//! `DropoutUnsupported`, `CausalUnsupported`, `Scale`, `BufferLen`,
//! `ComputeCapability`, `Cuda`, `SplitKernel`, `Abi`, `DqAccumSplits`,
//! `Window`. The safe API cannot produce `DropoutUnsupported` (it has no
//! dropout parameter at all — the type system is the Rust-boundary
//! refusal) or `HeadDim` (64 is a constant); the smoke test drives those
//! two through `raw` to prove the C side refuses them on its own.
//!
//! UNTESTED cells (honest accounting, not claimed as covered by the "one
//! test per cell" statement above):
//!
//! - **`FlashStatus::ComputeCapability`** (`check_device`, compute
//!   capability major `< 8`): unreachable on the CI/pod hardware this crate
//!   targets (the landing proof runs on an A100, cc 8.0); would need a
//!   pre-Ampere device to drive for real. Not simulated (faking
//!   `cudaDeviceGetAttribute`'s return would test the C `if`, not the
//!   actual refusal a pre-Ampere caller hits).
//! - **`FlashStatus::Cuda`** (a `cudaGetDevice`/`cudaDeviceGetAttribute`/
//!   post-launch `cudaGetLastError` failure): needs an actual CUDA runtime
//!   fault (OOM, driver reset, an invalid context) to trigger honestly;
//!   nothing in this crate's control flow can force one without mocking
//!   the CUDA runtime, which would test the mock, not the wrapper.
//! - **`FlashStatus::SplitKernel`** (internal: `params.num_splits > 1`):
//!   documented in the wrapper as unreachable by construction — `params =
//!   {}` zero-initialises `num_splits` to `0`, and the split-KV kernel's
//!   translation unit is not even linked into `libjammi_flash.a`. There is
//!   no code path, mutated or otherwise, that sets it to `> 1`; a test
//!   would have to hand-edit the C wrapper to fabricate one, which is not
//!   testing this crate.
//! - **`FlashError::UnknownStatus`** (a wrapper status code this Rust
//!   build's `FlashStatus::from_code` does not recognise): unreachable
//!   as long as the Rust `#[repr(i32)]` enum and the C header's status list
//!   are kept in lock-step (`status_codes_round_trip_and_zero_is_ok`
//!   exercises every KNOWN code both directions); would only fire if the
//!   two drifted, which `check_abi`'s struct-size check does not itself
//!   catch: a new status variant added to one side without the other is
//!   an unguarded hole (the enum count and the C header's status count
//!   are not cross-checked anywhere).
//! - **`stream == NULL`** (the header's documented "legacy default stream"
//!   arm): legal per `flash_api_jammi.h`, but neither this module's safe
//!   API nor the smoke test's `raw` calls ever pass a NULL stream — every
//!   call site has a real `CudaStream` from `dev.cuda_stream()`. The arm
//!   exists for a hypothetical caller of `raw` outside this crate; this
//!   crate itself never exercises it.
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
//! device memory on the host: `total_q` and `max_seqlen` are host inputs,
//! derived from the same lengths that build `cu_seqlens` — see
//! `CuSeqlens` below.
//!
//! # `CuSeqlens`: the only way to reach a launch
//!
//! Every kernel launch in this module indexes memory using `total_q` and
//! `max_seqlen` from the HOST while the actual row extents live in the
//! DEVICE `cu_seqlens` array — the kernels themselves do not bounds-check
//! `cu_seqlens` against them (upstream `flash_api.cpp:1103-1111`). If those
//! two ever disagree (a `VarlenGeometry` claiming `total_q = 21` paired with
//! a device array whose last entry is `4_000_000`), the launch reads/writes
//! past the buffers it was given — an illegal memory access from safe Rust,
//! not a panic or a `Result`, because it happens on the device after the
//! host-side call already returned `Ok`.
//!
//! `CuSeqlens` makes that disagreement unrepresentable: it is the ONLY
//! type that carries a device `cu_seqlens` array, and the only way to build
//! one is `CuSeqlens::from_lengths`, which computes `total_q` and
//! `max_seqlen` FROM the same host lengths that build the array — they are
//! never independent inputs. `VarlenGeometry` (below) is consequently a
//! private-field projection: the only way to obtain one is
//! `CuSeqlens::geometry`, so a geometry can never describe a different
//! array than the one it was derived from. Every public function in this
//! module that launches a kernel takes `&CuSeqlens`, never a raw
//! `cu_seqlens: CudaView<i32>` alongside a separate `&VarlenGeometry` — a
//! geometry from one batch paired with the device array of a different
//! batch would reintroduce the same disagreement even though both values
//! are individually legitimate.
//!
//! The one caller who legitimately owns a device `cu_seqlens` array already
//! (none in this codebase today; a future producer of one) has
//! `CuSeqlens::from_device_unchecked` — `unsafe`, with the exact contract
//! spelled out on the function.

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
    /// The device's compute capability is not a MEMBER of the SET of
    /// compute capabilities `build.rs` compiled this crate's cubins for
    /// (`check_arch`). Deliberately a distinct variant from `Geometry` (an
    /// architecture mismatch is a capability question, not a shape one)
    /// and distinct from `FlashStatus::ComputeCapability` (the C wrapper's
    /// own `cc_major < 8` guard, `flash_api_jammi.cu:151-156` — major-only,
    /// so it would NOT catch e.g. an sm_87 device on an sm_80/86/89/90-only
    /// build; this Rust-side check is exact SET membership and runs
    /// first). Membership is exact and enumerated, never `>=` and never
    /// major-compat (M3 plan D2): an sm_86 device is admitted only because
    /// `86` is literally in the compiled set, not because `8 >= 8`.
    #[error(
        "flash-attn: device compute capability {device:?} is not a member of this crate's \
         compiled arch set {built_for:?} (-gencode sm list: {built_for_sms}) — this build \
         embeds ONLY those cubins (no PTX), so a device outside the set cannot JIT a different, \
         unvalidated kernel variant; it would either fail to load the module or (if the C \
         wrapper's own major-only guard also missed it) silently run an unverified code path"
    )]
    Arch {
        built_for: Vec<(u32, u32)>,
        built_for_sms: String,
        device: (usize, usize),
    },
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

/// The mismatch message when the C library's `sizeof` for the two args
/// structs differs from the `#[repr(C)]` mirrors' — `None` when both agree.
/// Pure, so the cell table (fwd only / bwd only / both / neither) is
/// unit-tested without a way to inject a wrong C size.
fn abi_mismatch(c_fwd: usize, c_bwd: usize) -> Option<String> {
    let rust_fwd = std::mem::size_of::<raw::FwdArgs>();
    let rust_bwd = std::mem::size_of::<raw::BwdArgs>();
    if c_fwd == rust_fwd && c_bwd == rust_bwd {
        return None;
    }
    Some(format!(
        "fwd: C {c_fwd} vs Rust {rust_fwd}; bwd: C {c_bwd} vs Rust {rust_bwd}"
    ))
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
    match abi_mismatch(fwd, bwd) {
        None => Ok(()),
        Some(detail) => {
            let code = FlashStatus::Abi as i32;
            Err(FlashError::Refused {
                status: FlashStatus::Abi,
                code,
                message: format!("{} ({detail})", strerror(code)),
            })
        }
    }
}

/// Parses ONE `sm_XX` two-digit code into `(major, minor)` — the same
/// per-token rule [`parse_gencode_sms`] applies to each entry of the
/// comma-joined `JAMMI_FLASH_GENCODE_SMS` value. Two-digit `sm_XX` codes
/// split as (all digits but the last, last digit) = (major, minor) — true
/// of every NVIDIA compute capability this crate could plausibly target
/// (`sm_80`, `sm_86`, `sm_89`, `sm_90`); a future 3-digit code (e.g. a
/// hypothetical `sm_100`) would need this function to change, which
/// `gencode_sms_parses_the_pinned_build_value` below would catch (it pins
/// the exact CURRENT value, not just "parses without erroring").
fn parse_one_gencode_sm(sm: &str) -> Result<(u32, u32)> {
    if sm.len() < 2 || !sm.chars().all(|c| c.is_ascii_digit()) {
        return Err(FlashError::Geometry(format!(
            "JAMMI_FLASH_GENCODE_SMS token {sm:?}: expected at least two ASCII digits"
        )));
    }
    let split = sm.len() - 1;
    let (major_s, minor_s) = sm.split_at(split);
    let major: u32 = major_s.parse().map_err(|_| {
        FlashError::Geometry(format!(
            "JAMMI_FLASH_GENCODE_SMS token {sm:?}: major digits {major_s:?} do not parse as u32"
        ))
    })?;
    let minor: u32 = minor_s.parse().map_err(|_| {
        FlashError::Geometry(format!(
            "JAMMI_FLASH_GENCODE_SMS token {sm:?}: minor digit {minor_s:?} does not parse as u32"
        ))
    })?;
    Ok((major, minor))
}

/// `build.rs`'s comma-joined `-gencode` set, parsed from the
/// `JAMMI_FLASH_GENCODE_SMS` env var it emits beside those literals
/// (`cargo:rustc-env=JAMMI_FLASH_GENCODE_SMS=<sm>,<sm>,...`, e.g.
/// `"80,86,89,90"`) — never independently retyped, so this and the actual
/// compiled cubin set cannot drift apart. Widened from the deleted
/// singular `parse_gencode_sm`/`JAMMI_FLASH_GENCODE_SM` pair (M3 plan D2):
/// same per-token rule ([`parse_one_gencode_sm`]), applied to every
/// comma-separated entry rather than exactly one.
fn parse_gencode_sms(sms: &str) -> Result<Vec<(u32, u32)>> {
    sms.split(',').map(parse_one_gencode_sm).collect()
}

/// The set of compute capabilities this build's cubins were compiled for,
/// from `build.rs`'s `JAMMI_FLASH_GENCODE_SMS`.
fn built_for_compute_caps() -> Result<Vec<(u32, u32)>> {
    parse_gencode_sms(env!("JAMMI_FLASH_GENCODE_SMS"))
}

/// Pure core of [`check_arch`]: `None` iff `device` is a MEMBER of
/// `built_for`. Separated out so the mismatch cell is unit-testable
/// without a device — the same `abi_mismatch`/`check_abi` split this
/// module already uses. Membership, never `>=` and never major-compat
/// (M3 plan D2) — see [`FlashError::Arch`]'s own doc for why.
fn arch_mismatch(
    built_for: &[(u32, u32)],
    built_for_sms: &str,
    device: (usize, usize),
) -> Option<FlashError> {
    if built_for.contains(&(device.0 as u32, device.1 as u32)) {
        return None;
    }
    Some(FlashError::Arch {
        built_for: built_for.to_vec(),
        built_for_sms: built_for_sms.to_string(),
        device,
    })
}

/// Refuses a device whose compute capability is not a MEMBER of the set of
/// arches this crate's cubins were built for (see [`FlashError::Arch`]'s
/// doc for why set membership, never "at least"). Checked beside
/// [`check_abi`] at every launch (`flash_varlen_fwd_into`,
/// `flash_varlen_bwd_into`) — a cheap driver query, not a hot-path cost.
fn check_arch(dev: &CudaDevice) -> Result<()> {
    let built_for = built_for_compute_caps()?;
    // `compute_capability()` returns signed ints (cudarc mirrors the raw
    // CUDA driver attribute query, `cudaDeviceAttr`, which is `i32`);
    // `.max(0)` before the `usize` cast mirrors `admission.rs`'s own
    // `probe_cuda_compute_capability` — a negative value is not a real
    // capability, and clamping (not erroring) matches that precedent
    // exactly rather than introducing a second convention for the same
    // driver call.
    let (major, minor) = dev.cuda_stream().context().compute_capability()?;
    let device = (major.max(0) as usize, minor.max(0) as usize);
    match arch_mismatch(&built_for, env!("JAMMI_FLASH_GENCODE_SMS"), device) {
        None => Ok(()),
        Some(e) => Err(e),
    }
}

/// The batch's shape, host-side. Fields are PRIVATE: the only way to build
/// one is [`CuSeqlens::geometry`], which derives `total_q` and `max_seqlen`
/// from the very lengths that built the paired device `cu_seqlens` array —
/// see the module doc's "`CuSeqlens`: the only way to reach a launch"
/// section. A `VarlenGeometry` can therefore never disagree with the
/// `CuSeqlens` it came from; it is a read-only projection, not an
/// independent input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarlenGeometry {
    /// Sum of the sequence lengths (`cu_seqlens[batch]`).
    total_q: usize,
    /// Number of sequences (`cu_seqlens.len() - 1`).
    batch: usize,
    /// Attention heads (q, k and v share it — no GQA in this layout).
    num_heads: usize,
    /// The longest sequence in the batch. Sizes the kernel grid.
    max_seqlen: usize,
}

impl VarlenGeometry {
    /// Sum of the sequence lengths.
    pub fn total_q(&self) -> usize {
        self.total_q
    }
    /// Number of sequences.
    pub fn batch(&self) -> usize {
        self.batch
    }
    /// Attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    /// The longest sequence in the batch.
    pub fn max_seqlen(&self) -> usize {
        self.max_seqlen
    }
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

    /// The pure refusal cells of the lattice in the module doc. Called by
    /// [`CuSeqlens::geometry`] on every construction; not `pub` because a
    /// `VarlenGeometry` cannot be built without it (there is no other path
    /// to an unvalidated instance to call this on).
    fn validate(&self) -> Result<()> {
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

/// Host-only: builds the cumulative-sum array and validates it, without
/// touching any device. Pure so the whole lattice — including the class of
/// disagreement that produces an illegal memory access (host-claimed
/// `total_q`/`max_seqlen` smaller than the device array's real extent) — is
/// unit-tested here with no CUDA device required; `CuSeqlens::from_lengths`
/// is this plus the upload.
fn cu_seqlens_from_lengths(lengths: &[usize]) -> Result<(Vec<i32>, usize, usize, usize)> {
    if lengths.is_empty() {
        return Err(FlashError::Geometry(
            "cu_seqlens: batch must be non-empty (at least one sequence length)".to_string(),
        ));
    }
    let mut cu: Vec<i64> = Vec::with_capacity(lengths.len() + 1);
    cu.push(0);
    let mut max_seqlen: usize = 0;
    for (i, &len) in lengths.iter().enumerate() {
        // A zero-length sequence has no rows to attend from or to; the
        // downstream kernel grid is sized per-sequence and has no
        // representation for "this sequence contributes nothing" — refused
        // rather than silently producing an empty (and un-attended-to)
        // slice of the batch.
        if len == 0 {
            return Err(FlashError::Geometry(format!(
                "cu_seqlens: sequence {i} has length 0 — every sequence must be non-empty"
            )));
        }
        if len > i32::MAX as usize {
            return Err(FlashError::Geometry(format!(
                "cu_seqlens: sequence {i} length {len} exceeds i32::MAX"
            )));
        }
        max_seqlen = max_seqlen.max(len);
        // SAFETY-of-invariant note (not unsafe code): every `len` here is a
        // `usize` (never negative) checked `> 0` above, so `cu` is built by
        // repeated STRICT increase — it is non-decreasing (in fact strictly
        // increasing) BY CONSTRUCTION. A non-monotone cumulative array is
        // consequently unreachable through this function; see
        // `cu_seqlens_is_strictly_increasing_by_construction` below and
        // `CuSeqlens::from_device_unchecked`'s `# Safety` section, which is
        // where a non-monotone array becomes representable again (and is
        // exactly the misuse that contract forbids).
        let prev = *cu.last().expect("cu always has at least one element");
        let next = prev + len as i64;
        cu.push(next);
    }
    let total_q_i64 = *cu.last().expect("cu always has at least one element");
    if total_q_i64 > i32::MAX as i64 {
        return Err(FlashError::Geometry(format!(
            "cu_seqlens: total_q {total_q_i64} exceeds i32::MAX"
        )));
    }
    let batch = lengths.len();
    // batch <= total_q <= i32::MAX (every length >= 1), but check the
    // element count (batch + 1) explicitly with the module's own i32 guard
    // rather than relying on that inequality.
    as_i32("batch", batch)?;
    as_i32("cu_seqlens_len", batch + 1)?;
    let cu_i32: Vec<i32> = cu.iter().map(|&v| v as i32).collect();
    Ok((cu_i32, total_q_i64 as usize, batch, max_seqlen))
}

/// A validated, uploaded `cu_seqlens` array — the ONLY way any function in
/// this module accepts one. See the module doc's "`CuSeqlens`: the only way
/// to reach a launch" section for why: `total_q` and `max_seqlen` are
/// derived FROM the lengths that build the device array, so a `CuSeqlens`
/// can never claim extents its own array does not have.
pub struct CuSeqlens {
    cu: CudaSlice<i32>,
    total_q: usize,
    batch: usize,
    max_seqlen: usize,
}

impl CuSeqlens {
    /// Builds `cu_seqlens` from HOST sequence lengths: computes the prefix
    /// sum, validates it (non-empty batch, every length `> 0` and `<=
    /// i32::MAX`, `total_q <= i32::MAX`, the element-count guard the module
    /// already uses elsewhere), uploads the `i32` array, and derives
    /// `total_q`/`batch`/`max_seqlen` from the SAME lengths — they are never
    /// independent inputs. This is the sanctioned entry point; Stage B's
    /// encoder always has host lengths already (`BatchEncoding`), so this
    /// costs the real caller nothing beyond one small H2D copy per forward.
    pub fn from_lengths(lengths: &[usize], dev: &CudaDevice) -> Result<Self> {
        let (cu_i32, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(lengths)?;
        bind(dev)?;
        let cu = dev.clone_htod(&cu_i32)?;
        Ok(Self {
            cu,
            total_q,
            batch,
            max_seqlen,
        })
    }

    /// Builds a `CuSeqlens` from a device array the caller already owns,
    /// WITHOUT validating it against the device contents (that would need a
    /// synchronous device read, which this module never does).
    ///
    /// # Safety
    ///
    /// The caller must guarantee, for the exact array `cu` points to on
    /// `dev`'s stream:
    ///
    /// - `cu.len() == batch + 1`.
    /// - Every element is a valid `i32` (no reinterpreted garbage).
    /// - `cu[0] == 0`.
    /// - `cu` is strictly non-decreasing: `cu[i] <= cu[i + 1]` for every `i`.
    /// - `cu[batch] == total_q` EXACTLY.
    /// - `max_seqlen >= cu[i + 1] - cu[i]` for every `i` (the true maximum
    ///   sequence length in the batch; a value below it leaves that
    ///   sequence's tail rows unvisited by the kernel grid — silently wrong,
    ///   not a crash).
    /// - `cu` is valid on the same device/stream every later call in this
    ///   module is made with.
    ///
    /// Violating any of these produces an illegal memory access: the
    /// kernels index `cu[0..batch]` using the HOST `total_q`/`max_seqlen`
    /// and never bounds-check them against the array's actual contents, so
    /// a small `total_q`/`max_seqlen` claim paired with a device array
    /// whose real last entry is far larger reads and writes past the
    /// buffers the launch was given.
    pub unsafe fn from_device_unchecked(
        cu: CudaSlice<i32>,
        total_q: usize,
        batch: usize,
        max_seqlen: usize,
    ) -> Self {
        Self {
            cu,
            total_q,
            batch,
            max_seqlen,
        }
    }

    /// Sum of the sequence lengths.
    pub fn total_q(&self) -> usize {
        self.total_q
    }
    /// Number of sequences.
    pub fn batch(&self) -> usize {
        self.batch
    }
    /// The longest sequence in the batch.
    pub fn max_seqlen(&self) -> usize {
        self.max_seqlen
    }
    /// A view of the uploaded `[batch + 1]` `i32` array.
    pub fn as_view(&self) -> CudaView<'_, i32> {
        self.cu.as_view()
    }

    /// Refuses unless this batch is DENSE and UNIFORM at exactly `seq`:
    /// every one of its `batch` sequences has length exactly `seq` (`ops::
    /// rope_positions`'s own dense-only scope, module doc: "`cu_seqlens`
    /// uniform, every sequence length `== seq`" — the closed form
    /// `position = token % seq` `rope_positions_dims` compiles down to is
    /// only VALID under that premise). A HOST-only check — `total_q ==
    /// batch * seq` and `max_seqlen == seq` together are both NECESSARY
    /// and SUFFICIENT for "every length == seq": if the batch's `batch`
    /// non-negative integer lengths sum to `batch * seq` (their mean is
    /// exactly `seq`) but are not all equal to `seq`, at least one must
    /// exceed `seq` (a pigeonhole argument on integers below/at/above the
    /// mean), so `max_seqlen > seq` catches every non-uniform batch this
    /// sum-matches check alone would miss (e.g. `lengths = [3, 5], seq =
    /// 4`: `total_q = 8 = 2*4` passes the sum check alone, but
    /// `max_seqlen = 5 != 4` is refused here) — no device read of the
    /// per-sequence lengths (`cu`'s own array) is needed to prove it,
    /// consistent with this module's "nothing here reads device memory on
    /// the host" rule (module doc, "Synchronisation").
    pub(crate) fn check_dense_uniform(&self, seq: usize) -> Result<()> {
        check_dense_uniform_geometry(self.total_q, self.batch, self.max_seqlen, seq)
    }

    /// The batch's [`VarlenGeometry`] for `num_heads` attention heads —
    /// the ONLY way to construct one, so it can never disagree with `self`.
    /// Validates the num_heads-dependent cells (the int32 offset budget for
    /// `qkv`/`dq_accum`) that `from_lengths`/`from_device_unchecked` cannot
    /// check without knowing the head count.
    pub fn geometry(&self, num_heads: usize) -> Result<VarlenGeometry> {
        let g = VarlenGeometry {
            total_q: self.total_q,
            batch: self.batch,
            num_heads,
            max_seqlen: self.max_seqlen,
        };
        g.validate()?;
        Ok(g)
    }

    /// A deep, OWNED copy of this batch's device array. `CudaSlice<T>`'s
    /// own `Clone` impl (`cudarc` 0.17.8 `core.rs:619`) is a real
    /// device-to-device `memcpy` (`self.stream.clone_dtod(self)`, via the
    /// PANICKING `Clone::clone` — `try_clone().unwrap()`), not a refcount
    /// bump, so `CuSeqlens` cannot be cheaply `Arc`-shared by wrapping it
    /// directly; this crate-private, FALLIBLE method (never panics) is how
    /// [`crate::ops::flash_attention`]'s stateful op gives itself a
    /// `'static`, independently-owned handle that outlives the `&CuSeqlens`
    /// borrow its public entry point receives. The array itself is tiny
    /// (`batch + 1` `i32`s — single digits to low hundreds in practice), so
    /// the `memcpy` this performs is not a perf concern the way cloning
    /// `qkv`/`d_qkv` (megabytes) would be.
    pub(crate) fn try_duplicate(&self) -> Result<Self> {
        Ok(Self {
            cu: self.cu.try_clone()?,
            total_q: self.total_q,
            batch: self.batch,
            max_seqlen: self.max_seqlen,
        })
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

/// The pure geometry core of [`CuSeqlens::check_dense_uniform`] — HOST-only
/// integers in, no device access — so it is directly CPU-testable without
/// building a real `CuSeqlens` (which needs a `CudaDevice`). See that
/// method's own doc for the necessary-and-sufficient argument.
fn check_dense_uniform_geometry(
    total_q: usize,
    batch: usize,
    max_seqlen: usize,
    seq: usize,
) -> Result<()> {
    if total_q != batch * seq || max_seqlen != seq {
        return Err(FlashError::Geometry(format!(
            "cu_seqlens is not dense/uniform at seq={seq}: total_q={total_q}, batch={batch}, \
             max_seqlen={max_seqlen} -- expected total_q == batch*seq and max_seqlen == seq \
             (every sequence length exactly seq); a non-uniform batch would silently misindex \
             RopePositionsFused's dense-only position = token % seq"
        )));
    }
    Ok(())
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
    /// Allocates both buffers for `geom` on `dev`. `geom` (from
    /// [`CuSeqlens::geometry`]) is already validated by construction; this
    /// re-validates anyway (cheap, defense in depth for anyone who obtains
    /// one and holds it across an unrelated mutation).
    pub fn alloc(dev: &CudaDevice, geom: &VarlenGeometry, deterministic: bool) -> Result<Self> {
        geom.validate()?;
        let splits = dq_accum_splits(dev, geom.batch(), geom.num_heads(), deterministic)?;
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
/// `cu` is the ONLY source of `total_q`/`batch`/`max_seqlen` — see the
/// module doc.
pub fn flash_varlen_fwd_into(
    dev: &CudaDevice,
    qkv: CudaView<'_, bf16>,
    cu: &CuSeqlens,
    mut o: CudaViewMut<'_, bf16>,
    mut lse: CudaViewMut<'_, f32>,
    num_heads: usize,
    cfg: &VarlenConfig,
) -> Result<()> {
    let geom = cu.geometry(num_heads)?;
    cfg.validate()?;
    check_len("qkv", qkv.len(), geom.qkv_len())?;
    check_len("o", o.len(), geom.o_len())?;
    check_len("lse", lse.len(), geom.lse_len())?;
    let (window_size_left, window_size_right) = cfg.window_sizes()?;
    bind(dev)?;
    check_abi()?;
    check_arch(dev)?;

    let stream = dev.cuda_stream();
    let cu_seqlens = cu.as_view();
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
        total_q: as_i32("total_q", geom.total_q())?,
        batch: as_i32("batch", geom.batch())?,
        num_heads: as_i32("num_heads", geom.num_heads())?,
        head_dim: HEAD_DIM as i32,
        max_seqlen: as_i32("max_seqlen", geom.max_seqlen())?,
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
    cu: &CuSeqlens,
    num_heads: usize,
    cfg: &VarlenConfig,
) -> Result<(CudaSlice<bf16>, CudaSlice<f32>)> {
    let geom = cu.geometry(num_heads)?;
    // SAFETY: uninitialised outputs the kernel fully overwrites (every
    // (row, head) of `o` and `lse` belongs to exactly one launched block).
    let mut o = unsafe { dev.alloc::<bf16>(geom.o_len()) }?;
    let mut lse = unsafe { dev.alloc::<f32>(geom.lse_len()) }?;
    flash_varlen_fwd_into(
        dev,
        qkv.as_view(),
        cu,
        o.as_view_mut(),
        lse.as_view_mut(),
        num_heads,
        cfg,
    )?;
    Ok((o, lse))
}

/// The backward's buffers, as views (see the module-doc table). `cu_seqlens`
/// is NOT a field here: it is a separate `&CuSeqlens` parameter to
/// [`flash_varlen_bwd_into`] so it can never be paired with a geometry
/// derived from a different array.
pub struct BwdBuffers<'a> {
    pub qkv: CudaView<'a, bf16>,
    pub o: CudaView<'a, bf16>,
    pub lse: CudaView<'a, f32>,
    pub d_o: CudaView<'a, bf16>,
    pub d_qkv: CudaViewMut<'a, bf16>,
    pub softmax_d: CudaViewMut<'a, f32>,
    /// Must be all-zero when `cfg.deterministic` ; `flash_varlen_bwd_into`
    /// zeroes it itself when that holds, so callers do not need to (and a
    /// caller who reuses a poisoned buffer cannot silently produce a wrong
    /// `dQ`).
    pub dq_accum: CudaViewMut<'a, f32>,
    /// The split count `dq_accum` was sized with ([`dq_accum_splits`]).
    pub dq_accum_splits: usize,
}

/// Backward into caller-provided buffers (sizes checked exactly). `cu` is
/// the ONLY source of `total_q`/`batch`/`max_seqlen`.
pub fn flash_varlen_bwd_into(
    dev: &CudaDevice,
    cu: &CuSeqlens,
    num_heads: usize,
    bufs: BwdBuffers<'_>,
    cfg: &VarlenConfig,
) -> Result<()> {
    let geom = cu.geometry(num_heads)?;
    cfg.validate()?;
    let BwdBuffers {
        qkv,
        o,
        lse,
        d_o,
        mut d_qkv,
        mut softmax_d,
        mut dq_accum,
        dq_accum_splits: splits,
    } = bufs;
    check_len("qkv", qkv.len(), geom.qkv_len())?;
    check_len("o", o.len(), geom.o_len())?;
    check_len("lse", lse.len(), geom.lse_len())?;
    check_len("d_o", d_o.len(), geom.o_len())?;
    check_len("d_qkv", d_qkv.len(), geom.qkv_len())?;
    check_len("softmax_d", softmax_d.len(), geom.softmax_d_len())?;
    let expected_splits = dq_accum_splits(dev, geom.batch(), geom.num_heads(), cfg.deterministic)?;
    if splits != expected_splits {
        return Err(FlashError::Geometry(format!(
            "dq_accum was sized for {splits} split(s) but this device/config uses {expected_splits}"
        )));
    }
    check_len("dq_accum", dq_accum.len(), geom.dq_accum_len(splits))?;
    let (window_size_left, window_size_right) = cfg.window_sizes()?;
    bind(dev)?;
    check_abi()?;
    check_arch(dev)?;

    let stream = dev.cuda_stream();
    // `dq_accum` MUST be all-zero on entry when `cfg.deterministic`
    // (the deterministic launch selects `Clear_dQaccum=false` and
    // accumulates into whatever is already there,
    // `flash_bwd_launch_template.h:84-88`) — a caller that violates this
    // gets a silently WRONG `dQ`, not an error. `BwdScratch::alloc` already
    // zeroes it, but `BwdBuffers` is a public struct any caller can fill
    // with a reused/poisoned view, so this function re-zeroes it itself
    // rather than trusting the caller: one `cudaMemsetAsync` of
    // `dq_accum.num_bytes()` (a few MB at production geometry, e.g. ~63 MB
    // at B=24,S=512 per the module doc's memory line — already paid once by
    // `BwdScratch::alloc`'s own zeroing, so a `BwdScratch`-backed caller
    // pays it twice; the alternative, trusting the caller, was the
    // silently-wrong-gradient failure mode this closes).
    if cfg.deterministic {
        stream.memset_zeros(&mut dq_accum)?;
    }
    let cu_seqlens = cu.as_view();
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
        total_q: as_i32("total_q", geom.total_q())?,
        batch: as_i32("batch", geom.batch())?,
        num_heads: as_i32("num_heads", geom.num_heads())?,
        head_dim: HEAD_DIM as i32,
        max_seqlen: as_i32("max_seqlen", geom.max_seqlen())?,
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
    cu: &CuSeqlens,
    num_heads: usize,
    o: &CudaSlice<bf16>,
    lse: &CudaSlice<f32>,
    d_o: &CudaSlice<bf16>,
    cfg: &VarlenConfig,
) -> Result<CudaSlice<bf16>> {
    let geom = cu.geometry(num_heads)?;
    let mut scratch = BwdScratch::alloc(dev, &geom, cfg.deterministic)?;
    // SAFETY: uninitialised output the kernel fully overwrites (dq via
    // convert_dQ over every row block, dk/dv over every key block).
    let mut d_qkv = unsafe { dev.alloc::<bf16>(geom.qkv_len()) }?;
    flash_varlen_bwd_into(
        dev,
        cu,
        num_heads,
        BwdBuffers {
            qkv: qkv.as_view(),
            o: o.as_view(),
            lse: lse.as_view(),
            d_o: d_o.as_view(),
            d_qkv: d_qkv.as_view_mut(),
            softmax_d: scratch.softmax_d.as_view_mut(),
            dq_accum: scratch.dq_accum.as_view_mut(),
            dq_accum_splits: scratch.splits,
        },
        cfg,
    )?;
    Ok(d_qkv)
}

#[cfg(test)]
mod tests {
    //! The pure cells of the refusal lattice (no device needed).
    use super::*;

    #[test]
    fn gencode_sms_parses_the_pinned_build_value() {
        // Pins the CURRENT `build.rs` `GENCODE_ARCHES` literal exactly
        // (`80,86,89,90`) — a change to that literal without updating this
        // test is exactly the drift `built_for_compute_caps` exists to
        // prevent silently; this asserts the derivation, not just "it
        // parses something".
        assert_eq!(
            parse_gencode_sms("80,86,89,90").unwrap(),
            vec![(8, 0), (8, 6), (8, 9), (9, 0)]
        );
        assert_eq!(
            built_for_compute_caps().unwrap(),
            vec![(8, 0), (8, 6), (8, 9), (9, 0)]
        );
    }

    #[test]
    fn gencode_sm_parses_every_two_digit_ampere_hopper_code() {
        for (sm, want) in [
            ("80", (8, 0)),
            ("86", (8, 6)),
            ("89", (8, 9)),
            ("90", (9, 0)),
        ] {
            assert_eq!(parse_one_gencode_sm(sm).unwrap(), want, "sm_{sm}");
        }
    }

    #[test]
    fn gencode_sm_refuses_malformed_values() {
        for bad in ["", "8", "8a", "-80", "sm80", " 80"] {
            let e = parse_one_gencode_sm(bad).unwrap_err();
            assert!(matches!(e, FlashError::Geometry(_)), "{bad:?}: {e}");
        }
        // A malformed TOKEN inside an otherwise well-formed comma list
        // propagates the same error through `parse_gencode_sms` (the
        // `Iterator<Item = Result<_>>::collect()` short-circuit).
        let e = parse_gencode_sms("80,8a,90").unwrap_err();
        assert!(matches!(e, FlashError::Geometry(_)), "{e}");
    }

    #[test]
    fn arch_mismatch_cells() {
        let built_for = vec![(8, 0), (8, 6), (8, 9), (9, 0)];
        // Every compiled arch is admitted, exactly — set membership, never
        // major-compat (M3 plan D2): `(8, 6)`/`(8, 9)` admit only because
        // they are LITERALLY in the compiled set, not because `8 >= 8`.
        for &admit in &built_for {
            assert!(
                arch_mismatch(
                    &built_for,
                    "80,86,89,90",
                    (admit.0 as usize, admit.1 as usize)
                )
                .is_none(),
                "{admit:?} must be admitted -- it is in the compiled set"
            );
        }
        // Refused: same major, but a MINOR the compiled set does not
        // enumerate (sm_87) — the C wrapper's own major-only guard
        // (`cc_major < 8`) MISSES this but this Rust-side check must catch
        // it: membership is exact and enumerated, never "same major
        // admits".
        let e = arch_mismatch(&built_for, "80,86,89,90", (8, 7)).unwrap();
        assert!(matches!(e, FlashError::Arch { .. }), "{e}");
        // An OLDER device (sm_70) — the C wrapper WOULD catch this
        // (major < 8), but the Rust-side check catches it first either
        // way, with a message naming both the device and the compiled set.
        let e = arch_mismatch(&built_for, "80,86,89,90", (7, 0)).unwrap();
        let msg = e.to_string();
        assert!(msg.contains("(7, 0)") && msg.contains("(8, 0)"), "{msg}");
        // A NEWER, uncompiled major (a hypothetical sm100) — the typed
        // refusal D1 requires rather than an unvalidated PTX JIT.
        let e = arch_mismatch(&built_for, "80,86,89,90", (10, 0)).unwrap();
        assert!(matches!(e, FlashError::Arch { .. }), "{e}");
    }

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

    // -----------------------------------------------------------------
    // `CuSeqlens` / `cu_seqlens_from_lengths`. All pure (no
    // device): `cu_seqlens_from_lengths` is `CuSeqlens::from_lengths` minus
    // the upload, so the whole host-side lattice is testable here; the
    // device-touching half is exercised by `flash_smoke.rs` on the pod.
    // -----------------------------------------------------------------

    #[test]
    fn cu_seqlens_derives_total_q_and_max_seqlen_from_a_hand_computed_prefix_sum() {
        // Hand-computed: lengths [5, 9, 7] -> cu = [0, 5, 14, 21].
        let (cu, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(&[5, 9, 7]).unwrap();
        assert_eq!(cu, vec![0, 5, 14, 21]);
        assert_eq!(total_q, 21);
        assert_eq!(batch, 3);
        assert_eq!(max_seqlen, 9);
        // A second fixture with a different max position (first, not middle).
        let (cu, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(&[100, 1, 2, 3]).unwrap();
        assert_eq!(cu, vec![0, 100, 101, 103, 106]);
        assert_eq!(total_q, 106);
        assert_eq!(batch, 4);
        assert_eq!(max_seqlen, 100);
        // Singleton batch: total_q == max_seqlen == the one length.
        let (cu, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(&[42]).unwrap();
        assert_eq!(cu, vec![0, 42]);
        assert_eq!(total_q, 42);
        assert_eq!(batch, 1);
        assert_eq!(max_seqlen, 42);
    }

    /// [`check_dense_uniform_geometry`] — CPU-only, no `CudaDevice` needed
    /// (unlike a real `CuSeqlens`): `lengths = [3, 5], seq = 4` sums to
    /// `total_q = 8 = batch(2) * seq(4)` (the sum-matches check ALONE would
    /// pass this non-uniform batch) but `max_seqlen = 5 != seq = 4`, so it
    /// must be refused; `lengths = [4, 4], seq = 4` is genuinely dense and
    /// must pass.
    #[test]
    fn check_dense_uniform_geometry_refuses_non_uniform_and_accepts_dense() {
        let (_cu, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(&[3, 5]).unwrap();
        assert_eq!((total_q, batch, max_seqlen), (8, 2, 5));
        let e = check_dense_uniform_geometry(total_q, batch, max_seqlen, 4).unwrap_err();
        assert!(
            matches!(e, FlashError::Geometry(_)),
            "lengths=[3,5], seq=4 must be refused as non-uniform: {e}"
        );

        let (_cu, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(&[4, 4]).unwrap();
        assert_eq!((total_q, batch, max_seqlen), (8, 2, 4));
        check_dense_uniform_geometry(total_q, batch, max_seqlen, 4)
            .expect("lengths=[4,4], seq=4 is genuinely dense/uniform and must be accepted");
    }

    /// `total_q`/`max_seqlen` are ALWAYS derived from the device array's own
    /// construction — never an independent input a caller can mismatch
    /// against it. Concretely: the only lengths that produce the device
    /// array `[0, 5, 14, 4_000_000]` are `[5, 9, 3_999_986]`, which derive
    /// `total_q = 4_000_000`, `max_seqlen = 3_999_986` — a claim of
    /// `(total_q, max_seqlen) = (21, 9)` against that same array (small host
    /// numbers paired with an array whose real extent is 4,000,000 rows) can
    /// never come out of this function, so a kernel launch can never be
    /// handed that disagreement through the safe API.
    #[test]
    fn geometry_cannot_disagree_with_the_array_it_is_derived_from() {
        let lengths = [5usize, 9, 3_999_986];
        let (cu, total_q, batch, max_seqlen) = cu_seqlens_from_lengths(&lengths).unwrap();
        assert_eq!(cu, vec![0, 5, 14, 4_000_000]);
        assert_eq!(batch, 3);
        assert_eq!(
            total_q, 4_000_000,
            "derived total_q must be the array's real extent"
        );
        assert_eq!(
            max_seqlen, 3_999_986,
            "derived max_seqlen must be the array's real longest sequence"
        );
        assert_ne!(
            (total_q, max_seqlen),
            (21, 9),
            "a small mismatched (total_q, max_seqlen) claim must never come out of this function"
        );
    }

    #[test]
    fn cu_seqlens_refuses_empty_batch() {
        let e = cu_seqlens_from_lengths(&[]).unwrap_err();
        assert!(matches!(e, FlashError::Geometry(_)), "{e}");
        assert!(e.to_string().contains("non-empty"), "{e}");
    }

    #[test]
    fn cu_seqlens_refuses_zero_length_sequence() {
        // One cell per position: first, middle, last, and the singleton case.
        for lengths in [
            &[0usize, 5, 7][..],
            &[5, 0, 7][..],
            &[5, 7, 0][..],
            &[0][..],
        ] {
            let e = cu_seqlens_from_lengths(lengths).unwrap_err();
            assert!(matches!(e, FlashError::Geometry(_)), "{lengths:?}: {e}");
            assert!(e.to_string().contains("length 0"), "{lengths:?}: {e}");
        }
    }

    #[test]
    fn cu_seqlens_refuses_per_sequence_length_overflow() {
        let lengths = [5usize, i32::MAX as usize + 1, 7];
        let e = cu_seqlens_from_lengths(&lengths).unwrap_err();
        assert!(matches!(e, FlashError::Geometry(_)), "{e}");
        assert!(e.to_string().contains("exceeds i32::MAX"), "{e}");
        // The boundary itself (a single sequence of exactly i32::MAX) is
        // legal on its own (batch of one).
        assert!(cu_seqlens_from_lengths(&[i32::MAX as usize]).is_ok());
    }

    #[test]
    fn cu_seqlens_refuses_total_q_overflow() {
        // Each length individually fits i32::MAX, but the SUM does not.
        let half = i32::MAX as usize / 2 + 1;
        let lengths = [half, half];
        assert!(half <= i32::MAX as usize, "each length fits alone");
        let e = cu_seqlens_from_lengths(&lengths).unwrap_err();
        assert!(matches!(e, FlashError::Geometry(_)), "{e}");
        assert!(e.to_string().contains("total_q"), "{e}");
        assert!(e.to_string().contains("exceeds i32::MAX"), "{e}");
    }

    /// A non-monotone cumulative array is UNREACHABLE from
    /// `cu_seqlens_from_lengths`: every `length` is a `usize` (never
    /// negative) and is checked `> 0` before being added, so each partial
    /// sum is STRICTLY greater than the last — by construction, not by a
    /// runtime check. This is exactly the class of misuse
    /// `CuSeqlens::from_device_unchecked`'s `# Safety` section forbids for a
    /// caller-supplied array (its contract requires "strictly
    /// non-decreasing"): the safe constructor never needs to check it
    /// because it cannot produce anything else.
    #[test]
    fn cu_seqlens_is_strictly_increasing_by_construction() {
        for lengths in [
            &[5usize, 9, 7][..],
            &[1, 1, 1, 1][..],
            &[1000, 1, 1][..],
            &[7][..],
        ] {
            let (cu, ..) = cu_seqlens_from_lengths(lengths).unwrap();
            assert!(
                cu.windows(2).all(|w| w[0] < w[1]),
                "{lengths:?}: cu = {cu:?} is not strictly increasing"
            );
            assert_eq!(cu[0], 0, "{lengths:?}: cu[0] must be 0");
        }
    }

    /// A length mismatch — the derived `VarlenGeometry`/uploaded array
    /// disagreeing on element count — is likewise unreachable here: the
    /// device array `CuSeqlens::from_lengths` uploads is built from exactly
    /// this function's `cu` (length `lengths.len() + 1`), and
    /// `VarlenGeometry::cu_seqlens_len` (`batch + 1`) uses the SAME `batch`
    /// this function returns (`lengths.len()`) — the two can never drift
    /// apart because they are computed from the same host input in the same
    /// call.
    #[test]
    fn cu_seqlens_array_length_matches_geometrys_batch_plus_one_by_construction() {
        for lengths in [&[5usize, 9, 7][..], &[1][..], &[2, 2, 2, 2, 2][..]] {
            let (cu, _total_q, batch, _max_seqlen) = cu_seqlens_from_lengths(lengths).unwrap();
            assert_eq!(cu.len(), batch + 1);
            assert_eq!(batch, lengths.len());
        }
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
        // qkv cell: total_q · 3 · H · 64 > i32::MAX at H = 2 once
        // total_q > 5_592_405 (5_592_406 · 384 = 2_147_483_904); the
        // dq_accum product there ((5_592_406 + 128) · 2 · 64 ≈ 716M) stays
        // under, so this cell isolates the qkv term — and H = 2 (not 1) so
        // a dropped head factor changes the verdict.
        let g = VarlenGeometry {
            total_q: 5_592_406,
            batch: 1,
            num_heads: 2,
            max_seqlen: 5_592_406,
        };
        assert!(matches!(g.validate(), Err(FlashError::Geometry(_))));
        let g = VarlenGeometry {
            total_q: 5_592_405,
            max_seqlen: 5_592_405,
            ..g
        };
        assert!(g.validate().is_ok(), "{:?}", g.validate().err());
        // dq_accum cell: rows_padded = 1_000 + 128 · 131_072 = 16_778_216;
        // · 2 · 64 = 2_147_611_648 > i32::MAX, while qkv = 1_000 · 384 stays
        // tiny. Again H = 2 so the head factor is load-bearing.
        let g = VarlenGeometry {
            total_q: 1_000,
            batch: 131_072,
            num_heads: 2,
            max_seqlen: 1,
        };
        assert!(matches!(g.validate(), Err(FlashError::Geometry(_))));
        // Largest legal batch at this total_q / H: rows_padded must be
        // <= floor(i32::MAX / 128) = 16_777_215, i.e. batch <= 131_064
        // (1_000 + 128 · 131_064 = 16_777_192; · 128 = 2_147_480_576).
        let g = VarlenGeometry {
            batch: 131_064,
            ..g
        };
        assert!(g.validate().is_ok(), "{:?}", g.validate().err());
        // The comparison is `>`; `>=` would be indistinguishable here
        // because both products are multiples of 64 and i32::MAX is odd —
        // equality is unreachable, so that cell has no test by construction.
    }

    #[test]
    fn abi_mismatch_cells() {
        let fwd = std::mem::size_of::<raw::FwdArgs>();
        let bwd = std::mem::size_of::<raw::BwdArgs>();
        assert!(abi_mismatch(fwd, bwd).is_none());
        for (name, cf, cb) in [
            ("fwd only", fwd + 8, bwd),
            ("bwd only", fwd, bwd - 4),
            ("both", fwd + 8, bwd + 8),
        ] {
            let m = abi_mismatch(cf, cb).unwrap_or_else(|| panic!("{name}: no mismatch"));
            assert!(
                m.contains(&format!("C {cf}")) && m.contains(&format!("C {cb}")),
                "{m}"
            );
        }
    }

    #[test]
    fn status_message_is_the_wrapper_static_text() {
        // Each variant's message is the wrapper's own string (not empty,
        // not the unknown-code fallback), and the dropout one names the
        // parameter a caller would search for.
        for code in 1..=13 {
            let s = FlashStatus::from_code(code).unwrap();
            let m = s.message();
            assert!(!m.is_empty() && !m.contains("unknown"), "{s:?}: {m:?}");
        }
        assert!(FlashStatus::DropoutUnsupported
            .message()
            .contains("p_dropout must be 0.0"));
        assert!(FlashStatus::HeadDim.message().contains("64"));
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
