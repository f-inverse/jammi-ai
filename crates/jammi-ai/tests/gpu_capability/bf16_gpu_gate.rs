//! P4 — bf16 inference is ADMITTED and correct on an Ampere+ (sm_80+) GPU.
//!
//! This closes the coverage gap the pure-fn unit test cannot reach: the runtime
//! compute-capability gate's *admit wiring*. On a CUDA build the gate acquires
//! the resolved device's real capability
//! (`as_cuda_device().cuda_stream().context().compute_capability()`), feeds it
//! to `ComputePrecision::is_supported_on_cuda`, and — on sm_80+ — admits bf16
//! as `DType::BF16`. The CPU suite can only exercise the `cfg(not(cuda))` reject
//! arm and the decision predicate in isolation; only a real Ampere+ device
//! (here the A10G, sm_86) drives acquire → decide → admit end-to-end.
//!
//! The proof that bf16 was genuinely admitted-as-bf16 (not silently down-graded
//! to f32) is structural: the gate's `BF16` arm either maps to `DType::BF16` or
//! returns a typed error — there is no silent-fallback path — so a session that
//! loads and encodes on a `compute_precision = BF16` config *did* run bf16.
//! A bf16-rejecting gate (e.g. the pre-gate code, which rejected bf16
//! unconditionally) would make `encode_text_query` return `Err`, failing the
//! `.expect` below.
//!
//! Gated exactly like the rest of the suite: `live-gpu-tests` + a meaningful run
//! needs `cuda` + a visible GPU; without them it skips loudly via
//! `skip_without_gpu!` (never `#[ignore]`).

use jammi_numerics::ComputePrecision;
use tempfile::TempDir;

use crate::harness;
use crate::skip_without_gpu;

/// bf16 is admitted on an Ampere+ GPU and encodes the same direction as f32.
#[tokio::test(flavor = "multi_thread")]
async fn bf16_admitted_and_matches_f32_on_ampere() {
    skip_without_gpu!();
    harness::loss_capture::install();
    let model = harness::local_model_id("tiny_bert");
    let query = "a method for quantum error correction in superconducting qubits";

    // bf16 GPU session. If the runtime gate wrongly rejected sm_86 — or the
    // acquire→decide→admit wiring were broken — the model load would fail and
    // `encode_text_query` would return `Err`, tripping this `.expect`.
    let bf16_dir = TempDir::new().unwrap();
    let bf16 = harness::gpu_session_with_precision(bf16_dir.path(), ComputePrecision::BF16).await;
    let bf16_vec = bf16.encode_text_query(&model, query).await.expect(
        "bf16 must be ADMITTED on an Ampere+ GPU (sm_80+): the runtime \
         compute-capability gate should query the device (sm_86) and allow it",
    );

    // f32 GPU session on the same device: the reference direction.
    let f32_dir = TempDir::new().unwrap();
    let f32 = harness::gpu_session_with_precision(f32_dir.path(), ComputePrecision::F32).await;
    let f32_vec = f32.encode_text_query(&model, query).await.unwrap();

    // A valid embedding: same dimension as f32, every lane finite, not all-zero.
    assert_eq!(
        bf16_vec.len(),
        f32_vec.len(),
        "bf16 and f32 query vectors must share a dimension"
    );
    assert!(
        bf16_vec.iter().all(|x| x.is_finite()),
        "every lane of a bf16-computed query vector must be finite: {bf16_vec:?}"
    );
    assert!(
        bf16_vec.iter().any(|&v| v != 0.0),
        "bf16 query vector must not be all-zero"
    );

    // Correct: bf16 preserves f32's 8-bit exponent range, so it encodes the same
    // direction as f32 (only the low mantissa bits differ). A wrong dtype path
    // (transposed weights, a bad cast) would collapse this cosine well below the
    // floor. `search` consumes direction, so cosine is the decisive signal.
    let cos = harness::cosine(&bf16_vec, &f32_vec);
    tracing::info!(
        cos,
        dim = bf16_vec.len(),
        "bf16↔f32 encode_query cosine on Ampere (A10G sm_86)"
    );
    assert!(
        (0.99..=1.0).contains(&cos),
        "bf16 must encode the same direction as f32 (cosine {cos} outside [0.99, 1.0]) — \
         a wrong bf16 dtype path would diverge"
    );
}
