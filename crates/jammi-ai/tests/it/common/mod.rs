pub use jammi_test_utils::*;

use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use jammi_ai::session::InferenceSession;
use jammi_db::error::{JammiError, Result as JammiResult};
use jammi_db::store::ArtifactStore;
use jammi_numerics::retrieval::AggregateMetrics;

/// The ANN index-segment bundle base URLs of `table_name`, in segment order —
/// the load-side handle for tests that inspect a table's on-disk sidecar bundle
/// now that a table's index is a set of segments rather than one `index_path`.
pub async fn segment_index_urls(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> Vec<String> {
    session
        .catalog()
        .list_index_segments(table_name)
        .await
        .unwrap()
        .into_iter()
        .map(|s| s.index_path)
        .collect()
}

/// The single segment's bundle base URL for a freshly-embedded table (one embed
/// pass writes exactly one segment). Panics unless there is exactly one.
pub async fn segment0_index_url(
    session: &jammi_ai::session::InferenceSession,
    table_name: &str,
) -> String {
    let mut urls = segment_index_urls(session, table_name).await;
    assert_eq!(
        urls.len(),
        1,
        "expected exactly one index segment for '{table_name}'"
    );
    urls.pop().unwrap()
}

/// Build an [`ArtifactStore`] rooted at a hermetic `memory://` URL with a fresh
/// local fetch cache, for resolver-level unit tests that construct a
/// `ModelResolver` directly (rather than through a full `InferenceSession`). The
/// cache dir leaks for the test binary's lifetime — acceptable in a test.
pub fn test_artifact_store() -> Arc<ArtifactStore> {
    let cache = tempfile::tempdir().unwrap().keep();
    Arc::new(
        ArtifactStore::with_root(
            jammi_db::storage::StorageUrl::memory("test-artifacts"),
            jammi_db::storage::StorageRegistry::new(),
            cache,
        )
        .unwrap(),
    )
}

/// Return the four aggregate retrieval metrics paired with their wire-format
/// snake_case names. Tests that need to iterate over every metric (range
/// checks, baseline-vs-candidate diffs, determinism comparisons) consume this
/// instead of indexing fields by string — the array literal makes adding a
/// new metric to [`AggregateMetrics`] a single-file edit that the compiler
/// guides exhaustively.
pub fn aggregate_named_metrics(agg: &AggregateMetrics) -> [(&'static str, f64); 4] {
    [
        ("recall_at_k", agg.recall_at_k),
        ("precision_at_k", agg.precision_at_k),
        ("mrr", agg.mrr),
        ("ndcg", agg.ndcg),
    ]
}

// =============================================================================
// esc-089's own `symptom_spec.control` (`.jammi/escapes.jsonl`), shared by
// every `*_serves_cold_after_restart` test: three cross-modal towers in
// `tower_adapters.rs` and the BERT-family peer in `fine_tune.rs`. The
// fix-verifier proved esc-089's RED->GREEN mechanism but ruled it
// NOT-SYMPTOM-FAITHFUL because none of the four tests implemented the
// escape's own control — this section is that control, in exactly one
// place, so every site runs the SAME oracle:
//
//   - positive controls, in the WARM session, before any cold assertion:
//     `v_base`/`v_warm` both fully finite and non-degenerate (L2 > 1e-6);
//     `max|v_warm - v_base| >= 1e-3` over finite pairs; a second
//     `serve(base_id)` bit-identical to the first (base-side determinism);
//   - the cold mechanism assertion: `v_cold == v_warm` bit-for-bit;
//   - the mechanism co-assertion: after the restart, the fine-tuned id's
//     catalog record (read through `Catalog::get_model`, the same db-API
//     read path the esc-089 RED dump used) still carries `model_type ==
//     "fine-tuned"`, `artifact_path.is_some()`, `base_model_id.is_some()`;
//   - the negative control: with the published bundle's `adapter.safetensors`
//     deleted, a COLD serve through a brand-new `InferenceSession` (the real
//     resolver, never a hand-built `ResolvedModel`) refuses with a typed
//     `JammiError::Model` naming the missing file and returns no vector.
// =============================================================================

/// A boxed, higher-ranked "serve this model" closure:
/// `Fn(&InferenceSession) -> future<Result<Vec<f32>>>`, generic over the
/// session's own borrow so the SAME closure re-applies unchanged to the warm
/// session, the cold session, and (for the negative control) a third
/// post-mutation session, without the caller re-borrowing anything it owns.
/// [`text_serve`]/[`image_serve`]/[`audio_serve`] build one per probe by
/// moving in an owned model id and payload.
pub type ServeFn = Box<
    dyn for<'s> Fn(
            &'s InferenceSession,
        ) -> Pin<Box<dyn Future<Output = JammiResult<Vec<f32>>> + Send + 's>>
        + Send
        + Sync,
>;

/// A [`ServeFn`] over [`InferenceSession::encode_text_query`].
pub fn text_serve(model_id: impl Into<String>, probe: &'static str) -> ServeFn {
    let model_id = model_id.into();
    Box::new(move |s: &InferenceSession| {
        let model_id = model_id.clone();
        Box::pin(async move { s.encode_text_query(&model_id, probe).await })
    })
}

/// A [`ServeFn`] over [`InferenceSession::encode_image_query`]. `bytes` is
/// `Arc`-shared rather than cloned per call — a probe image can be sizeable
/// and this closure is invoked at least three times (warm repeat, cold, the
/// negative control's post-mutation serve).
pub fn image_serve(model_id: impl Into<String>, bytes: Arc<Vec<u8>>) -> ServeFn {
    let model_id = model_id.into();
    Box::new(move |s: &InferenceSession| {
        let model_id = model_id.clone();
        let bytes = Arc::clone(&bytes);
        Box::pin(async move { s.encode_image_query(&model_id, &bytes).await })
    })
}

/// A [`ServeFn`] over [`InferenceSession::encode_audio_query`]. See
/// [`image_serve`] for why `bytes` is `Arc`-shared.
pub fn audio_serve(model_id: impl Into<String>, bytes: Arc<Vec<u8>>) -> ServeFn {
    let model_id = model_id.into();
    Box::new(move |s: &InferenceSession| {
        let model_id = model_id.clone();
        let bytes = Arc::clone(&bytes);
        Box::pin(async move { s.encode_audio_query(&model_id, &bytes).await })
    })
}

/// esc-089 positive control (a): every component of `v` is finite and `v`'s
/// L2 norm is `> 1e-6` — rules out a degenerate (NaN-laced, all-zero, or
/// near-zero) embedding making the downstream difference/bit-equality
/// checks pass vacuously.
pub fn assert_finite_and_nondegenerate(v: &[f32], label: &str) {
    for (i, x) in v.iter().enumerate() {
        assert!(
            x.is_finite(),
            "{label}: component {i} is non-finite ({x}) — every component of a served \
             embedding must be finite"
        );
    }
    let norm = v
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    assert!(
        norm > 1e-6,
        "{label}: L2 norm {norm} is not > 1e-6 — a degenerate (near-zero) embedding"
    );
}

/// esc-089 positive control (b): `max|other[i] - base[i]|`, taken ONLY over
/// index pairs where both components are finite (family F: a control must
/// fail on every bad path, including non-finite — `NaN > c` is `false`, so
/// letting a non-finite component silently poison the max would make this
/// assertion pass vacuously on exactly the input it exists to reject). Panics
/// if `base`/`other` share no finite pair at all, rather than reporting a
/// vacuous `max_diff = 0.0`.
pub fn assert_min_diff_over_finite_pairs(base: &[f32], other: &[f32], min_diff: f32, label: &str) {
    assert_eq!(
        base.len(),
        other.len(),
        "{label}: base and other embeddings must have the same width"
    );
    let mut max_diff = 0.0f32;
    let mut finite_pairs = 0usize;
    for (a, b) in base.iter().zip(other) {
        if a.is_finite() && b.is_finite() {
            finite_pairs += 1;
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert!(
        finite_pairs > 0,
        "{label}: no finite (base, other) pair to compare — at least one vector is entirely \
         non-finite"
    );
    assert!(
        max_diff >= min_diff,
        "{label}: max|Δ| over {finite_pairs} finite pairs = {max_diff}, expected >= {min_diff}"
    );
}

/// The cold-restart mechanism assertion, unchanged from before this unit:
/// `served` and `reference` must be bit-for-bit identical.
pub fn assert_bit_equal(served: &[f32], reference: &[f32], label: &str) {
    assert_eq!(
        served.len(),
        reference.len(),
        "{label}: served and reference embeddings must have the same width"
    );
    for (i, (a, b)) in served.iter().zip(reference).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}: component {i} differs bit-for-bit (served {a}, reference {b})"
        );
    }
}

/// esc-089 mechanism co-assertion: after a cold restart, `model_id`'s catalog
/// record — read through [`jammi_db::catalog::Catalog::get_model`], the same
/// db-API read path the esc-089 RED dump used to show the corrupted row —
/// still carries the three fields `ModelResolver::try_catalog_lookup`'s
/// fine-tuned arm depends on. This pins the ROW, not just the served vector:
/// a regression that clobbers `model_type`/`artifact_path`/`base_model_id`
/// (esc-089's actual root cause — `ModelCache::do_load`'s post-load
/// bookkeeping) but happens to leave both warm and cold resolving to the
/// SAME corrupted row would still pass a bit-equality-only oracle.
pub async fn assert_fine_tuned_record_intact(
    session: &InferenceSession,
    model_id: &str,
    label: &str,
) {
    let record = session
        .catalog()
        .get_model(model_id)
        .await
        .expect("catalog lookup")
        .unwrap_or_else(|| {
            panic!("{label}: fine-tuned model '{model_id}' is not registered in the catalog after restart")
        });
    assert_eq!(
        record.model_type, "fine-tuned",
        "{label}: catalog record's model_type must stay 'fine-tuned' after restart, got '{}'",
        record.model_type
    );
    assert!(
        record.artifact_path.is_some(),
        "{label}: catalog record's artifact_path must stay Some after restart"
    );
    assert!(
        record.base_model_id.is_some(),
        "{label}: catalog record's base_model_id must stay Some after restart"
    );
}

/// esc-089 negative control: with the published bundle's `adapter.safetensors`
/// deleted, a COLD serve through a brand-new [`InferenceSession`] opened over
/// `session_root` (the real resolver, never a hand-built `ResolvedModel`)
/// must refuse — a typed `JammiError::Model` whose message names the missing
/// file — and must never return a vector, finite or not.
///
/// `bundle_dir` is read straight off `model_id`'s catalog `artifact_path`: a
/// `file://` prefix resolves to that exact directory with no copy (see
/// `ArtifactStore::fetch_artifact`'s in-place path), so deleting the file
/// there deletes the SAME file every earlier serve in the test read.
pub async fn assert_deleted_adapter_refuses_by_name(
    session: &InferenceSession,
    session_root: &Path,
    model_id: &str,
    label: &str,
    serve_tuned: &ServeFn,
) {
    let record = session
        .catalog()
        .get_model(model_id)
        .await
        .expect("catalog lookup")
        .expect("fine-tuned model registered in catalog");
    let prefix = record.artifact_path.expect("artifact_path");
    let prefix_url = jammi_db::storage::StorageUrl::parse(&prefix).unwrap();
    let bundle_dir = std::path::PathBuf::from(prefix_url.path());
    let weights_path = bundle_dir.join("adapter.safetensors");
    std::fs::remove_file(&weights_path).unwrap_or_else(|e| {
        panic!(
            "{label}: failed to delete the published bundle's adapter.safetensors at \
             {weights_path:?}: {e}"
        )
    });

    let broken_session = InferenceSession::new(test_config(session_root))
        .await
        .unwrap();
    match serve_tuned(&broken_session).await {
        Ok(v) => panic!(
            "{label}: serving '{model_id}' after deleting its adapter.safetensors returned a \
             vector (len {}) instead of refusing — a broken bundle must never silently serve",
            v.len()
        ),
        Err(JammiError::Model { message, .. }) => {
            assert!(
                message.contains("adapter.safetensors"),
                "{label}: refusal message must name the missing file 'adapter.safetensors', \
                 got: {message}"
            );
        }
        Err(other) => panic!(
            "{label}: expected a typed JammiError::Model naming the missing file, got a \
             different error variant: {other:?}"
        ),
    }
}

/// Runs the FULL esc-089 `symptom_spec.control` set against one
/// already-completed cold-restart round trip. Called identically by
/// `tower_adapters.rs`'s three cross-modal `*_serves_cold_after_restart`
/// tests and `fine_tune.rs`'s BERT-family peer.
#[allow(clippy::too_many_arguments)]
pub async fn assert_esc089_cold_restart_controls(
    session_root: &Path,
    warm_session: &InferenceSession,
    cold_session: &InferenceSession,
    model_id: &str,
    label: &str,
    v_base: &[f32],
    v_warm: &[f32],
    v_cold: &[f32],
    serve_base: &ServeFn,
    serve_tuned: &ServeFn,
) {
    // Positive controls (warm session).
    assert_finite_and_nondegenerate(v_base, &format!("{label}: v_base"));
    assert_finite_and_nondegenerate(v_warm, &format!("{label}: v_warm"));
    assert_min_diff_over_finite_pairs(v_base, v_warm, 1e-3, &format!("{label}: v_warm vs v_base"));
    let base_repeat = serve_base(warm_session).await.unwrap_or_else(|e| {
        panic!("{label}: second serve(base_id) in the warm session failed: {e}")
    });
    assert_bit_equal(v_base, &base_repeat, &format!("{label}: base served twice"));

    // The cold mechanism assertion.
    assert_bit_equal(v_warm, v_cold, &format!("{label}: cold vs warm"));

    // Mechanism co-assertion, read through the cold (post-restart) session.
    assert_fine_tuned_record_intact(cold_session, model_id, label).await;

    // Negative control.
    assert_deleted_adapter_refuses_by_name(
        cold_session,
        session_root,
        model_id,
        label,
        serve_tuned,
    )
    .await;
}
