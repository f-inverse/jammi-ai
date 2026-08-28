//! esc-058 fix test (`closes_escape: esc-058`): a warm `ModelCache` must not
//! serve a pre-mutation `Arc<LoadedModel>` (stale digest + stale vectors)
//! after the underlying model directory is mutated in place — the SAME
//! session, SAME `ModelCache`, SAME `ModelId`, never dropped in between.
//!
//! Reuses `pooling_config.rs`'s hermetic `tiny_bert` fixture-copy helpers
//! (`build_local_model_dir`, `cls_pooling_config`, `mean_pooling_config`) —
//! the identical mutation `pooling_config.rs:150-192` already proves changes
//! the pooled vectors, so the RED/GREEN assertions below rest on a
//! production-proven trigger, not a hypothetical one.
//!
//! Reverting the esc-058 staleness probe (`ModelCache::get_or_load`'s
//! `probe_freshness` call on its fast path, back to the unconditioned
//! `cache.entries.get(&id)` hit origin/main shipped) makes
//! `warm_hit_after_in_place_mutation_reloads_fresh_digest_and_vectors`'s
//! post-fix assertions RED: the warm process would keep attesting `D1`/`V1`
//! after the mutation instead of `D2`/`V2`.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use jammi_ai::concurrency::GpuScheduler;
use jammi_ai::model::backend::DeviceConfig;
use jammi_ai::model::cache::ModelCache;
use jammi_ai::model::resolver::ModelResolver;
use jammi_ai::model::{LoadedModel, ModelSource, ModelTask};
use jammi_db::catalog::Catalog;
use jammi_db::store::manifest::ModelContentDigest;
use tempfile::tempdir;

use crate::pooling_config::{build_local_model_dir, cls_pooling_config, mean_pooling_config};

const TEXT: &str = "the quick brown fox jumps over the lazy dog";

/// A max-abs-difference tolerance well under the mean-vs-CLS pooling gap on
/// `tiny_bert` (`pooling_config.rs` shows those vectors are NOT even
/// approximately equal at `1e-5`) — any real content-triggered divergence
/// clears this by orders of magnitude, so this stays a strict, non-vacuous
/// bound rather than a threshold tuned to pass.
const DIVERGENCE_TOL: f32 = 1e-4;

fn new_cache(catalog: Arc<Catalog>) -> ModelCache {
    let resolver = ModelResolver::new(catalog, crate::common::test_artifact_store()).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let scheduler = Arc::new(GpuScheduler::new_unlimited());
    ModelCache::new(resolver, device_config, scheduler)
}

fn embed(model: &LoadedModel, text: &str) -> Vec<f32> {
    let content: Vec<ArrayRef> = vec![Arc::new(StringArray::from(vec![text])) as ArrayRef];
    let output = model.forward(&content, ModelTask::TextEmbedding).unwrap();
    output.float_outputs[0].clone()
}

/// (c)'s non-finite-safe divergence check: every element of BOTH vectors
/// must be finite before any comparison is drawn (a diverged-to-NaN load
/// would otherwise satisfy a naive `v1 != v2` and make a `<=` threshold
/// comparison silently false rather than surfacing the corruption), and the
/// mutation's effect is proven by a strictly positive, FINITE max-abs delta
/// — asserted as `delta.is_finite() && delta > tol`, never `!(delta <= tol)`.
fn assert_finite_and_diverged(label: &str, a: &[f32], b: &[f32], tol: f32) {
    assert!(
        a.iter().all(|x| x.is_finite()),
        "{label}: all elements of the first vector must be finite before comparison, got {a:?}"
    );
    assert!(
        b.iter().all(|x| x.is_finite()),
        "{label}: all elements of the second vector must be finite before comparison, got {b:?}"
    );
    assert_eq!(a.len(), b.len(), "{label}: vector width must match");
    let delta = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        delta.is_finite() && delta > tol,
        "{label}: expected a strictly positive, finite divergence > {tol}, got delta={delta} \
         (a={a:?}, b={b:?})"
    );
}

fn assert_finite_and_identical(label: &str, a: &[f32], b: &[f32]) {
    assert!(
        a.iter().all(|x| x.is_finite()),
        "{label}: all elements of the first vector must be finite, got {a:?}"
    );
    assert!(
        b.iter().all(|x| x.is_finite()),
        "{label}: all elements of the second vector must be finite, got {b:?}"
    );
    assert_eq!(
        a, b,
        "{label}: expected bitwise-identical vectors (no mutation occurred)"
    );
}

/// (a) `digest` must be the HASHED `Sha256` variant, never `Unavailable` —
/// otherwise `D1 == D2` could hold vacuously (a silently degraded digest
/// looks "unchanged" no matter what).
fn assert_hashed(label: &str, digest: &ModelContentDigest) -> String {
    match digest {
        ModelContentDigest::Sha256(hex) => hex.clone(),
        ModelContentDigest::Unavailable(reason) => panic!(
            "{label}: expected a hashed Sha256 content digest, got \
             Unavailable({reason:?}) — a degraded digest would make this control vacuous"
        ),
    }
}

fn tiny_bert_dir(root: &Path, name: &str, pooling: &serde_json::Value) -> std::path::PathBuf {
    let dir = root.join(name);
    build_local_model_dir(&dir, Some(pooling));
    dir
}

/// The esc-058 observable + all four control arms, in one session / one
/// `ModelCache` / one `ModelId`, exactly as the escape's `symptom_spec`
/// requires:
///
/// (1) embed `dir` under `ModelCache` → `V1`, `D1` (warm entry cached).
/// (2) mutate `dir`'s `1_Pooling/config.json` bytes IN PLACE (CLS → mean —
///     `pooling_config.rs` proves this changes the pooled vectors),
///     WITHOUT dropping the cache/session.
/// (3) `get_or_load` again on the SAME `ModelId`, SAME cache — the warm-hit
///     path under test.
/// (4) `ModelCache::load_owned_for_test` for the post-mutation COLD reading
///     — the non-vacuous control (a)–(c).
/// (5) a second, UNTOUCHED model dir proves arms (3)/(4) are not just
///     ambient nondeterminism — control (d).
#[tokio::test]
async fn warm_hit_after_in_place_mutation_reloads_fresh_digest_and_vectors() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    // --- mutated dir ---
    let dir = tiny_bert_dir(tmp.path(), "warm_model", &cls_pooling_config());
    let source = ModelSource::local(&dir);

    // (1) Warm the cache.
    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d1_raw = guard1.model.content_digest().unwrap();
    let d1 = assert_hashed("D1 (pre-mutation, warm)", &d1_raw);
    let v1 = embed(&guard1.model, TEXT);
    drop(guard1); // mirrors EmbeddingPipeline::run: guard dropped once digest/dims are read.

    // (2) In-place mutation — dir (and therefore ModelId) never changes.
    std::fs::write(
        dir.join("1_Pooling/config.json"),
        serde_json::to_string_pretty(&mean_pooling_config()).unwrap(),
    )
    .unwrap();

    // (3) Warm replay in the SAME session, SAME cache, SAME ModelId.
    let guard_warm = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d_warm_raw = guard_warm.model.content_digest().unwrap();
    let d_warm = assert_hashed("D (warm, post-mutation)", &d_warm_raw);
    let v_warm = embed(&guard_warm.model, TEXT);
    drop(guard_warm);

    // (4) Control: the post-mutation COLD reading, bypassing the LRU
    // entirely (independent re-resolve + re-load).
    let cold_model = cache
        .load_owned_for_test(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();
    let d2_raw = cold_model.content_digest().unwrap();
    let d2 = assert_hashed("D2 (post-mutation, cold)", &d2_raw);
    let v2 = embed(&cold_model, TEXT);

    // (a) both D1 and D2 are HASHED — already enforced by `assert_hashed`
    // above (not `Unavailable`, not `Err`).

    // (b) D2 != D1 as exact hex strings — independently established, so the
    // fixture genuinely produces different bytes post-mutation.
    assert_ne!(
        d1, d2,
        "D2 (cold, post-mutation) must differ from D1 (pre-mutation) as exact hex \
         strings — otherwise this test cannot distinguish a fixed digest from a \
         content-tracking one"
    );

    // (c) V1 vs V2 (cold): finite-safe, strictly-positive-delta divergence.
    assert_finite_and_diverged("V1 vs V2 (cold)", &v1, &v2, DIVERGENCE_TOL);

    // THE oracle: post-fix, the warm replay must match the fresh cold
    // reading (D2/V2), never the pre-mutation D1/V1. Pre-fix (the fast path
    // with no staleness probe), `d_warm == d1` and `v_warm == v1` — this
    // assertion is RED there.
    assert_eq!(
        d_warm, d2,
        "warm replay after an in-place mutation must record the CURRENT digest \
         (D2), not the stale load-time digest (D1) — got d_warm={d_warm}, D1={d1}, \
         D2={d2}"
    );
    assert_ne!(
        d_warm, d1,
        "warm replay must NOT still attest the pre-mutation digest D1"
    );
    assert_eq!(
        v_warm, v2,
        "warm replay after an in-place mutation must emit the CURRENT vectors \
         (bitwise equal to the fresh cold reading V2), not the stale pre-mutation \
         vectors"
    );
    assert_finite_and_diverged(
        "V1 vs V_warm (post-mutation warm replay)",
        &v1,
        &v_warm,
        DIVERGENCE_TOL,
    );

    // (d) No-mutation determinism control: a SEPARATE, UNTOUCHED model dir
    // in the SAME cache must report the SAME digest and BITWISE-IDENTICAL
    // finite vectors on a warm replay and a cold reading — proving the RED
    // above cannot be manufactured by ambient nondeterminism (a timestamp,
    // an unstable fold order, an unseeded RNG) rather than the actual
    // mutation.
    let untouched_dir = tiny_bert_dir(tmp.path(), "untouched_model", &mean_pooling_config());
    let untouched_source = ModelSource::local(&untouched_dir);

    let guard_u1 = cache
        .get_or_load(&untouched_source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let du1 = assert_hashed(
        "D (untouched, first load)",
        &guard_u1.model.content_digest().unwrap(),
    );
    let vu1 = embed(&guard_u1.model, TEXT);
    drop(guard_u1);

    let guard_u_warm = cache
        .get_or_load(&untouched_source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let du_warm = assert_hashed(
        "D (untouched, warm replay)",
        &guard_u_warm.model.content_digest().unwrap(),
    );
    let vu_warm = embed(&guard_u_warm.model, TEXT);
    drop(guard_u_warm);

    let cold_u_model = cache
        .load_owned_for_test(&untouched_source, ModelTask::TextEmbedding)
        .await
        .unwrap();
    let du_cold = assert_hashed(
        "D (untouched, cold reading)",
        &cold_u_model.content_digest().unwrap(),
    );
    let vu_cold = embed(&cold_u_model, TEXT);

    assert_eq!(
        du1, du_warm,
        "untouched dir: warm replay digest must match the first load's digest"
    );
    assert_eq!(
        du1, du_cold,
        "untouched dir: cold reading digest must match the first load's digest"
    );
    assert_finite_and_identical("untouched dir: first load vs warm replay", &vu1, &vu_warm);
    assert_finite_and_identical("untouched dir: first load vs cold reading", &vu1, &vu_cold);
}
