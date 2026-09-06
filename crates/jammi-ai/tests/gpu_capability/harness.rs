//! Shared machinery for the GPU-capability suite: the CUDA-availability skip
//! guard, the paired CPU / GPU session builders, the fixture paths, and the
//! parity tolerances + comparison helpers every property reuses.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::DataType;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::config::{GpuConfig, InferenceConfig, JammiConfig, LoggingConfig};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_numerics::ComputePrecision;

// ─── Parity tolerances ───────────────────────────────────────────────────────
//
// candle's CUDA and CPU backends are *not* bit-identical: matmul reductions
// run in a different order (and, on the GPU, may fold through cuBLAS), so the
// low bits of an embedding differ even when the math is correct. The decisive
// signal that the *kernel is correct* is that the two devices produce the same
// vector up to that low-bit reduction noise — so we assert on cosine similarity
// (direction is what every downstream `search` / context pool consumes) with a
// tight floor, plus a generous per-element absolute bound as a coarse guard
// against a single blown lane.
//
// COSINE_FLOOR = 0.9999: a correct fp32 forward over a tiny model diverges only
// in reduction order; 1e-4 of angular slack is orders of magnitude looser than
// the noise a correct kernel produces yet far tighter than any real bug (a
// transposed weight, a wrong dtype, an off-by-one in a kernel) could sneak
// under — such a bug collapses cosine well below 0.99.

/// Minimum cosine similarity between a CPU vector and its GPU counterpart for
/// the pair to count as parity. See the module note for the justification.
pub const COSINE_FLOOR: f64 = 0.9999;

/// Per-element absolute tolerance, a coarse backstop alongside the cosine
/// floor: no single lane may diverge by more than this regardless of direction.
pub const ELEMENTWISE_ABS_TOL: f64 = 1e-3;

// ─── CUDA-availability skip guard ──────────────────────────────────────────

/// Whether a CUDA device is usable for this build. `false` whenever the `cuda`
/// feature is off (the engine compiles no CUDA path) or no device opens, so the
/// suite skips cleanly on a CPU build / GPU-less host instead of failing.
#[cfg(feature = "cuda")]
pub fn gpu_available() -> bool {
    candle_core::Device::new_cuda(0).is_ok()
}

/// Without the `cuda` feature the engine has no CUDA path at all, so the suite
/// always skips.
#[cfg(not(feature = "cuda"))]
pub fn gpu_available() -> bool {
    false
}

// ─── The one-at-a-time GPU slot for DEVICE-GLOBAL oracles ──────────────────

/// This process's single GPU slot for a DEVICE-GLOBAL measurement. See
/// [`SerialGpu`] for why it exists, and for the precise boundary of what it
/// does and does not close.
///
/// Binary-scoped (here in `harness`, not private to one module) on purpose:
/// `gpu_capability` is ONE test binary with fourteen modules, so a per-file
/// static could only ever serialize a file against itself. A module that
/// grows a device-global oracle tomorrow must take THIS slot, not mint a
/// second one that the first cannot see.
static GPU_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A CUDA device that cannot be held without also holding [`GPU_SERIAL`].
///
/// Campaign #446, finding 9 (the class sibling of
/// `crates/jammi-encoders/tests/esc076_comparable_eager_control.rs`'s own
/// `SerialGpu`). `gguf_quantized_gpu.rs`'s admission-truthfulness oracle
/// reads DEVICE-GLOBAL memory (`nvidia-smi --query-gpu=memory.used`, a
/// whole-device figure) as a before/after DELTA around a model load. Any
/// concurrent allocation on the same device inside that window is charged to
/// the load; any concurrent free can even drive the signed delta negative.
///
/// **The remedy is structural, not documentary.** `serial_cuda_device` is the
/// only way to get a `Device` for such an oracle, and it hands back this
/// wrapper, which holds the slot for as long as the caller holds the device.
/// A test added tomorrow cannot forget to serialize, because it cannot obtain
/// the device without doing so.
///
/// # Why there is no `Deref<Target = Device>` (round-1 audit)
///
/// This type used to `impl Deref<Target = Device>`, which kept `&device` call
/// sites unchanged — and let `let d = (*serial_cuda_device().unwrap()).clone();`
/// type-check. That one-liner ENDS the serialization: the temporary guard
/// (and with it the slot) is dropped at the end of the statement, while `d`
/// is a live, owned `Device` the caller then measures on with no slot held —
/// the exact escape the wrapper exists to prevent, spelled as ordinary deref
/// usage.
///
/// [`Self::device`] replaces it: the borrow it returns is tied to `&self`, so
/// no `&Device` can outlive the guard. Every call site passes
/// `guard.device()` where it used to pass `&guard`.
///
/// **What is still open, stated rather than implied.** `candle_core::Device`
/// is `Clone`, so `guard.device().clone()` compiles and always will —
/// nothing an API of this shape can do prevents cloning a `Clone` type
/// reachable by reference (a `&DeviceRef` newtype does not help: if it
/// derefs to `Device` the clone resolves straight through it, and if it does
/// not, no call site can pass it where `&Device` is wanted). What changed is
/// that the escape is now an EXPLICIT, greppable `.device().clone()` rather
/// than an incidental consequence of deref — a reviewer looking for it has a
/// single spelling to grep for, and no correct call site needs it.
///
/// **What this does NOT close, stated plainly rather than implied.** Only the
/// callers of `serial_cuda_device` take the slot. The other thirteen modules
/// in this binary build GPU-pinned sessions through [`gpu_session`] and
/// allocate on the same device without taking it; today they are held off the
/// measurement window ONLY by `ci/scripts/runpod_gpu_prove.sh`'s
/// `--test-threads=1` on the `gpu_capability` invocation — a CI flag, i.e.
/// exactly the kind of convention finding 9 is about. Closing that residual
/// means routing every device acquisition in this binary through this slot
/// (a change to `gpu_session`'s signature at every call site), which is a
/// separate, larger unit; it is recorded here so the remaining exposure is
/// visible at the mechanism rather than only in a review note.
///
/// A poisoned lock is recovered with `into_inner` rather than unwrapped: one
/// leg panicking must fail THAT leg, not turn every sibling into a confusing
/// poison error that buries the original diagnosis.
pub struct SerialGpu {
    device: candle_core::Device,
    /// Held, never read — dropping it at the end of the test body is the
    /// entire mechanism.
    _slot: std::sync::MutexGuard<'static, ()>,
}

impl SerialGpu {
    /// This guard's device, borrowed for no longer than the guard itself —
    /// the ONLY way to reach it, and the reason there is no `Deref` (see the
    /// type doc). Deliberately NOT `into_device`/`to_device`: nothing in this
    /// binary has a legitimate reason to hold a `Device` past the slot.
    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }
}

/// Take this binary's one-at-a-time GPU slot, recovering a poisoned lock
/// rather than unwrapping it (see [`SerialGpu`]'s doc for why).
///
/// Split out from [`serial_cuda_device`] so the exclusion property itself is
/// testable on a GPU-less lane, where no `SerialGpu` can ever be constructed:
/// `gpu_slot_is_exclusive_while_held` below is the non-vacuous control that
/// the slot is a real mutex and not a decorative field.
fn take_gpu_slot() -> std::sync::MutexGuard<'static, ()> {
    GPU_SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A CUDA device bound to this binary's one-at-a-time [`GPU_SERIAL`] slot —
/// the ONLY device source a device-global oracle may use. `None` when no CUDA
/// device opens, which without the `cuda` feature is always: `Device::new_cuda`
/// exists in either build and simply errors there, so this needs no `cfg` arm
/// of its own (the caller's `skip_without_gpu!` has already returned in that
/// lane anyway).
pub fn serial_cuda_device() -> Option<SerialGpu> {
    // Taken BEFORE `Device::new_cuda`, so even device ACQUISITION (which
    // allocates a context on the device) is serialized against a sibling
    // leg's memory trace.
    let slot = take_gpu_slot();
    candle_core::Device::new_cuda(0)
        .ok()
        .map(|device| SerialGpu {
            device,
            _slot: slot,
        })
}

/// The slot is a real mutual exclusion, and it is released only when the
/// guard drops — the property every device-global oracle in this binary
/// leans on, pinned on EVERY lane (this one needs no GPU, so the CPU lane
/// proves it too, where no [`SerialGpu`] can be constructed at all).
///
/// Non-vacuous in both directions: `try_lock` must FAIL while the slot is
/// held (a decorative, always-available mutex fails here) and SUCCEED once it
/// is dropped (a slot that leaked would fail here instead).
#[test]
fn gpu_slot_is_exclusive_while_held() {
    {
        let _slot = take_gpu_slot();
        assert!(
            GPU_SERIAL.try_lock().is_err(),
            "a second holder must not be able to take the slot while it is held — without \
             this, `SerialGpu` would serialize nothing"
        );
    }
    assert!(
        GPU_SERIAL.try_lock().is_ok(),
        "dropping the guard must release the slot — a leaked slot would deadlock every \
         later device-global oracle in this binary"
    );
}

// ─── The one-at-a-time admission-counter slot for CPU-hermetic dispatch-counter tests ───

/// This process's serialization slot for any CPU-hermetic test that reads a
/// `jammi_kernels::admission` dispatch-registry counter as a before/after
/// DELTA. Binary-scoped for the same reason [`GPU_SERIAL`] is (`gpu_capability`
/// is ONE test binary with fourteen modules; a per-file static could only
/// serialize a file against itself): `crates/jammi-encoders`'s own
/// admission-counter tests already share ONE lock per counter family (e.g.
/// `attention_cascade::ATTENTION_BLOCK_COUNTER_TEST_LOCK`) precisely because
/// `cargo test` runs a binary's tests concurrently by default and a
/// process-wide registry counter has no per-test isolation of its own — a
/// sibling test dispatching the SAME counter mid-window would corrupt a
/// before/after delta. This binary's first CPU-hermetic counter test
/// (`capability_surface`'s
/// `gelu_erf_fused_bumps_on_a_bert_family_training_forward_cpu_hermetic`,
/// issue #463 follow-up) takes this lock; any sibling added later must take
/// the SAME one rather than minting a second the first cannot see.
pub static ADMISSION_COUNTER_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Early-return a test with a loud `tracing::warn` skip (never `#[ignore]`)
/// when no GPU is usable, so the GPU-less / CPU lane runs the suite as a no-op
/// rather than a failure. Returns `true` when the caller should skip.
#[macro_export]
macro_rules! skip_without_gpu {
    () => {{
        if !$crate::harness::gpu_available() {
            tracing::warn!(
                "SKIP: no usable CUDA device (build the suite with \
                 `--features cuda,live-gpu-tests` on a GPU host to run it)"
            );
            return;
        }
    }};
}

// ─── Fixture paths ───────────────────────────────────────────────────────────

/// Workspace root — three levels up from this test's manifest dir
/// (`crates/jammi-ai/tests/gpu_capability/` → workspace root).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Path to a `tests/fixtures/` fixture (e.g. `patents.parquet`).
pub fn fixture(name: &str) -> PathBuf {
    workspace_root().join("tests").join("fixtures").join(name)
}

/// `file://` URL for a `tests/fixtures/` fixture, suitable for source registration.
pub fn fixture_url(name: &str) -> String {
    format!("file://{}", fixture(name).display())
}

/// Path to a `cookbook/fixtures/` fixture (e.g. the `tiny_bert/` encoder dir).
pub fn cookbook_fixture(name: &str) -> PathBuf {
    workspace_root()
        .join("cookbook")
        .join("fixtures")
        .join(name)
}

/// `local:` model id for a cookbook encoder fixture — the same id the cookbook
/// recipes and the CPU `it` suite use for `tiny_bert`.
pub fn local_model_id(fixture_name: &str) -> String {
    format!("local:{}", cookbook_fixture(fixture_name).to_str().unwrap())
}

/// `file://` URL for a `cookbook/fixtures/` fixture, suitable for source
/// registration (e.g. the `tiny_ner_corpus.parquet` corpus).
pub fn cookbook_fixture_url(name: &str) -> String {
    format!("file://{}", cookbook_fixture(name).display())
}

/// `local:` model id for a `tests/fixtures/` encoder fixture — the same id
/// the CPU `it` suite uses for `tiny_modernbert` / `tiny_open_clip`, which are
/// committed under `tests/fixtures/` rather than `cookbook/fixtures/`.
pub fn local_fixture_model_id(fixture_name: &str) -> String {
    format!("local:{}", fixture(fixture_name).to_str().unwrap())
}

// ─── Session builders ─────────────────────────────────────────────────────────

/// A JammiConfig rooted at `artifact_dir`, pinned to `device`
/// (`-1` = CPU, `0` = first CUDA device), running inference at `precision`. The
/// GPU variant sets `require_gpu = true` so a build / host without a usable GPU
/// fails fast at session construction rather than silently degrading to CPU — a
/// parity test that runs to completion on `device = 0` therefore *did* run on
/// the GPU.
fn config_for(artifact_dir: &Path, device: i32, precision: ComputePrecision) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: GpuConfig {
            device,
            require_gpu: device >= 0,
            compute_precision: precision,
            ..Default::default()
        },
        inference: InferenceConfig {
            batch_size: 8,
            ..Default::default()
        },
        logging: LoggingConfig {
            level: "info".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Build a CPU-pinned (`gpu.device = -1`) session over a fresh artifact dir, at
/// the default `F32` precision.
pub async fn cpu_session(artifact_dir: &Path) -> Arc<InferenceSession> {
    Arc::new(
        InferenceSession::new(config_for(artifact_dir, -1, ComputePrecision::F32))
            .await
            .expect("cpu-pinned session"),
    )
}

/// Build a GPU-pinned (`gpu.device = 0`, `require_gpu = true`) session over a
/// fresh artifact dir, at the default `F32` precision. Only call after
/// [`gpu_available`] / `skip_without_gpu!`.
pub async fn gpu_session(artifact_dir: &Path) -> Arc<InferenceSession> {
    gpu_session_with_precision(artifact_dir, ComputePrecision::F32).await
}

/// Build a GPU-pinned (`gpu.device = 0`, `require_gpu = true`) session over a
/// fresh artifact dir at an explicit inference `precision` — the entry point
/// for exercising the compute-precision gate on a real device (e.g. `BF16`,
/// whose runtime compute-capability gate only resolves on a CUDA device). Only
/// call after [`gpu_available`] / `skip_without_gpu!`.
pub async fn gpu_session_with_precision(
    artifact_dir: &Path,
    precision: ComputePrecision,
) -> Arc<InferenceSession> {
    Arc::new(
        InferenceSession::new(config_for(artifact_dir, 0, precision))
            .await
            .expect("gpu-pinned session (require_gpu=true)"),
    )
}

// ─── Shared source / result-table fixtures ─────────────────────────────────

/// Register the `patents.parquet` fixture as a source named `"patents"` on
/// `session` — the shared text corpus every text-embedding / classification /
/// NER parity cell runs over.
pub async fn add_patents(session: &Arc<InferenceSession>) {
    session
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
}

/// Read an embedding result table's `(_row_id, vector)` rows into a map, so a
/// CPU and a GPU table compare row-exact even if their scan order differs.
/// Every embedding-generation verb (`generate_text_embeddings`,
/// `generate_image_embeddings`, `generate_audio_embeddings`) writes this same
/// `_row_id` + `vector` (`FixedSizeList<Float32>`) result-table shape.
pub async fn keyed_result_vectors(
    session: &Arc<InferenceSession>,
    table: &ResultTableRecord,
) -> HashMap<String, Vec<f32>> {
    let batches = session
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{}\"",
            table.table_name
        ))
        .await
        .unwrap();
    let mut out = HashMap::new();
    for batch in &batches {
        let ids = arrow::compute::cast(batch.column(0), &DataType::Utf8).unwrap();
        let ids = ids.as_any().downcast_ref::<StringArray>().unwrap();
        let list = batch
            .column(1)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            let cell = list.value(i);
            let floats = cell.as_any().downcast_ref::<Float32Array>().unwrap();
            out.insert(
                ids.value(i).to_string(),
                (0..floats.len()).map(|j| floats.value(j)).collect(),
            );
        }
    }
    out
}

// ─── Parity comparison helpers ─────────────────────────────────────────────

/// Cosine similarity between two equal-length vectors. Panics on a length
/// mismatch (a parity comparison over mismatched dims is itself a bug).
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "parity vectors must share a dimension");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Largest per-element absolute difference between two equal-length vectors.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "parity vectors must share a dimension");
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

// ─── Per-epoch loss capture (P2 / P3 learning proof) ───────────────────────
//
// The fine-tune trainer emits one `tracing::info!(epoch, avg_train_loss, …,
// "Epoch complete")` event per epoch. The run runs in an in-process worker task
// (same process, a different task), so a process-global `tracing` subscriber
// observes those events. We install one such subscriber once for the whole test
// binary and record `(epoch, avg_train_loss)` into a shared buffer; each test
// clears the buffer before its single `fine_tune` call and reads the captured
// curve after. `--test-threads=1` (the suite's mandated run mode) means no two
// fine-tune runs interleave, so the buffer holds exactly one run's epochs.

pub mod loss_capture {
    use std::sync::{Mutex, OnceLock};

    use tracing::field::{Field, Visit};
    use tracing::Event;
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::Layer;

    /// `(epoch, avg_train_loss)` rows captured since the last [`reset`].
    static EPOCHS: OnceLock<Mutex<Vec<(u64, f64)>>> = OnceLock::new();
    /// `(epoch, avg_val_loss)` rows — only pushed when the "Epoch complete"
    /// event actually carried an `avg_val_loss` field (tracing's
    /// `impl<T: Value> Value for Option<T>` skips recording a `None` field
    /// entirely, so this buffer is naturally empty on a `TrainLoss`-monitored
    /// run and populated on the default `ValLoss`-monitored one).
    static VAL_EPOCHS: OnceLock<Mutex<Vec<(u64, f64)>>> = OnceLock::new();
    static INSTALLED: OnceLock<()> = OnceLock::new();

    fn buffer() -> &'static Mutex<Vec<(u64, f64)>> {
        EPOCHS.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn val_buffer() -> &'static Mutex<Vec<(u64, f64)>> {
        VAL_EPOCHS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// A `tracing` layer that records the `epoch` + `avg_train_loss` +
    /// `avg_val_loss` fields of every "Epoch complete" event into the shared
    /// buffers.
    struct EpochLossLayer;

    struct EpochVisitor {
        epoch: Option<u64>,
        loss: Option<f64>,
        val_loss: Option<f64>,
        is_epoch_event: bool,
    }

    impl Visit for EpochVisitor {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" && format!("{value:?}").contains("Epoch complete") {
                self.is_epoch_event = true;
            }
        }
        fn record_u64(&mut self, field: &Field, value: u64) {
            if field.name() == "epoch" {
                self.epoch = Some(value);
            }
        }
        fn record_i64(&mut self, field: &Field, value: i64) {
            if field.name() == "epoch" {
                self.epoch = Some(value as u64);
            }
        }
        fn record_f64(&mut self, field: &Field, value: f64) {
            if field.name() == "avg_train_loss" {
                self.loss = Some(value);
            } else if field.name() == "avg_val_loss" {
                self.val_loss = Some(value);
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for EpochLossLayer {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut v = EpochVisitor {
                epoch: None,
                loss: None,
                val_loss: None,
                is_epoch_event: false,
            };
            event.record(&mut v);
            if v.is_epoch_event {
                if let (Some(e), Some(l)) = (v.epoch, v.loss) {
                    buffer().lock().unwrap().push((e, l));
                }
                if let (Some(e), Some(l)) = (v.epoch, v.val_loss) {
                    val_buffer().lock().unwrap().push((e, l));
                }
            }
        }
    }

    /// Install the per-epoch loss-capture layer once for the test binary,
    /// alongside a console fmt layer (filtered by `RUST_LOG`, default `info`) so
    /// the suite's parity / loss / delta reports print under `--nocapture`.
    /// Idempotent. Call from every test so the capture layer is live for the
    /// fine-tune runs and the reports surface for the parity runs.
    pub fn install() {
        INSTALLED.get_or_init(|| {
            use tracing_subscriber::EnvFilter;
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let fmt = tracing_subscriber::fmt::layer()
                .with_test_writer()
                .with_target(false);
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt)
                .with(EpochLossLayer)
                .init();
        });
    }

    /// Clear the buffers before a fresh fine-tune run.
    pub fn reset() {
        buffer().lock().unwrap().clear();
        val_buffer().lock().unwrap().clear();
    }

    /// The captured `(epoch, avg_train_loss)` rows, ordered by capture order.
    pub fn captured() -> Vec<(u64, f64)> {
        buffer().lock().unwrap().clone()
    }

    /// The captured `(epoch, avg_val_loss)` rows, ordered by capture order —
    /// empty unless the run's `early_stopping_metric` actually measured
    /// validation loss (the default, `ValLoss`).
    pub fn captured_val() -> Vec<(u64, f64)> {
        val_buffer().lock().unwrap().clone()
    }
}

/// Assert that a captured per-epoch loss curve genuinely *decreases* first→last
/// — the on-device-learning proof. Requires at least two epochs (a single-epoch
/// run carries no first→last signal). Returns `(first_loss, last_loss)` for
/// reporting.
pub fn assert_loss_decreases(label: &str, curve: &[(u64, f64)]) -> (f64, f64) {
    assert!(
        curve.len() >= 2,
        "{label}: need ≥2 epochs to prove a loss decrease, captured {curve:?}"
    );
    let first = curve.first().unwrap().1;
    let last = curve.last().unwrap().1;
    tracing::info!(label, first, last, epochs = curve.len(), "loss curve");
    assert!(
        first.is_finite() && last.is_finite(),
        "{label}: non-finite loss in {curve:?}"
    );
    assert!(
        last < first,
        "{label}: training loss did not decrease on GPU (first {first}, last {last}); \
         curve {curve:?}"
    );
    (first, last)
}

/// Assert every loss value across one or more captured curves is finite, BY
/// COUNT (family F9: never a vacuous "some finite" pass — every reported
/// value is checked and the tally is asserted, not merely the endpoints
/// [`assert_loss_decreases`] happens to touch).
pub fn assert_all_finite(label: &str, curves: &[&[(u64, f64)]]) {
    let mut total = 0usize;
    let mut finite = 0usize;
    for curve in curves {
        for &(_, l) in curve.iter() {
            total += 1;
            if l.is_finite() {
                finite += 1;
            }
        }
    }
    assert_eq!(
        finite, total,
        "{label}: expected every reported epoch loss finite, got {finite}/{total}"
    );
}

/// Assert CPU↔GPU parity for one named vector pair: cosine ≥ [`COSINE_FLOOR`]
/// and every lane within [`ELEMENTWISE_ABS_TOL`]. Returns `(cosine, max_abs)`
/// so the caller can report the achieved numbers.
pub fn assert_parity(label: &str, cpu: &[f32], gpu: &[f32]) -> (f64, f64) {
    let cos = cosine(cpu, gpu);
    let max_abs = max_abs_diff(cpu, gpu);
    tracing::info!(label, cos, max_abs, "CPU↔GPU parity");
    assert!(
        cos >= COSINE_FLOOR,
        "{label}: CPU↔GPU cosine {cos} below floor {COSINE_FLOOR} \
         (max |Δ| {max_abs}) — GPU output diverged, a real kernel/dtype bug"
    );
    assert!(
        max_abs <= ELEMENTWISE_ABS_TOL,
        "{label}: CPU↔GPU max |Δ| {max_abs} exceeds {ELEMENTWISE_ABS_TOL} \
         (cosine {cos}) — a single lane blew up, a real kernel bug"
    );
    (cos, max_abs)
}
