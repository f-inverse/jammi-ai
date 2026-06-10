//! Shared machinery for the GPU-capability suite: the CUDA-availability skip
//! guard, the paired CPU / GPU session builders, the fixture paths, and the
//! parity tolerances + comparison helpers every property reuses.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::config::{GpuConfig, InferenceConfig, JammiConfig, LoggingConfig};

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

// ─── Session builders ─────────────────────────────────────────────────────────

/// A JammiConfig rooted at `artifact_dir` and pinned to `device`
/// (`-1` = CPU, `0` = first CUDA device). The GPU variant sets
/// `require_gpu = true` so a build / host without a usable GPU fails fast at
/// session construction rather than silently degrading to CPU — a parity test
/// that runs to completion on `device = 0` therefore *did* run on the GPU.
fn config_for(artifact_dir: &Path, device: i32) -> JammiConfig {
    JammiConfig {
        artifact_dir: artifact_dir.to_path_buf(),
        gpu: GpuConfig {
            device,
            require_gpu: device >= 0,
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

/// Build a CPU-pinned (`gpu.device = -1`) session over a fresh artifact dir.
pub async fn cpu_session(artifact_dir: &Path) -> Arc<InferenceSession> {
    Arc::new(
        InferenceSession::new(config_for(artifact_dir, -1))
            .await
            .expect("cpu-pinned session"),
    )
}

/// Build a GPU-pinned (`gpu.device = 0`, `require_gpu = true`) session over a
/// fresh artifact dir. Only call after [`gpu_available`] / `skip_without_gpu!`.
pub async fn gpu_session(artifact_dir: &Path) -> Arc<InferenceSession> {
    Arc::new(
        InferenceSession::new(config_for(artifact_dir, 0))
            .await
            .expect("gpu-pinned session (require_gpu=true)"),
    )
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
    static INSTALLED: OnceLock<()> = OnceLock::new();

    fn buffer() -> &'static Mutex<Vec<(u64, f64)>> {
        EPOCHS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// A `tracing` layer that records the `epoch` + `avg_train_loss` fields of
    /// every "Epoch complete" event into the shared buffer.
    struct EpochLossLayer;

    struct EpochVisitor {
        epoch: Option<u64>,
        loss: Option<f64>,
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
            }
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for EpochLossLayer {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut v = EpochVisitor {
                epoch: None,
                loss: None,
                is_epoch_event: false,
            };
            event.record(&mut v);
            if v.is_epoch_event {
                if let (Some(e), Some(l)) = (v.epoch, v.loss) {
                    buffer().lock().unwrap().push((e, l));
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

    /// Clear the buffer before a fresh fine-tune run.
    pub fn reset() {
        buffer().lock().unwrap().clear();
    }

    /// The captured `(epoch, avg_train_loss)` rows, ordered by capture order.
    pub fn captured() -> Vec<(u64, f64)> {
        buffer().lock().unwrap().clone()
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
