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

use crate::pooling_config::{
    build_local_model_dir, cls_pooling_config, mean_pooling_config, sample_preprocessor_config,
    with_length_marker, write_preprocessor_config,
};

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
    //
    // F-4a: deliberately LENGTH-CHANGING (via `with_length_marker`, not an
    // incidental `to_string` vs `to_string_pretty` formatting difference —
    // both writes here use the SAME compact serializer), and asserted as
    // such below. A straight cls_pooling_config -> mean_pooling_config swap
    // is byte-length-IDENTICAL (one `true`/one `false` just trade places),
    // which would make this test's staleness detection rest entirely on
    // sub-second mtime resolution — never asserted, and not portable to a
    // coarser filesystem clock.
    let pooling_config_path = dir.join("1_Pooling/config.json");
    let bytes_before_mutation = std::fs::read(&pooling_config_path).unwrap();
    let mutated_pooling_json =
        serde_json::to_string(&with_length_marker(mean_pooling_config())).unwrap();
    std::fs::write(&pooling_config_path, &mutated_pooling_json).unwrap();
    assert_ne!(
        bytes_before_mutation.len(),
        mutated_pooling_json.len(),
        "F-4a: the primary mutation must be byte-length-changing, not merely \
         content-changing — a length-identical mutation's detection would rest \
         entirely on sub-second mtime resolution, never asserted here"
    );

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

/// F-4a diagnostic (NOT a hard gate — `#[ignore]`d): the SAME scenario as
/// `warm_hit_after_in_place_mutation_reloads_fresh_digest_and_vectors`, but
/// with a byte-length-IDENTICAL mutation (a straight
/// `cls_pooling_config()` ⇄ `mean_pooling_config()` swap — each has exactly
/// one `true`/4 chars and five `false`/5 chars, just at different keys, so
/// the serialized length never changes). Detection here rests ENTIRELY on
/// the filesystem's mtime resolution: on a coarse-grained clock (some
/// container/network filesystems only tick whole seconds), a same-second
/// rewrite is indistinguishable from no rewrite at all to `stat`, and the
/// probe will keep serving the stale in-memory model — the fingerprint's
/// own documented residual (`ModelFingerprint`'s doc, `compute_model_content_digest`'s
/// sibling). This test is an honest, explicitly-labelled record of that
/// residual, not a correctness assertion this crate can guarantee on every
/// filesystem — it is `#[ignore]`d so a coarse-clock CI runner cannot flake
/// the gate on an environmental property this fix does not (and cannot)
/// control. Run it manually (`cargo test -- --ignored`) to observe whether
/// your local filesystem's mtime resolution happens to catch this case.
#[tokio::test]
#[ignore = "F-4a: detection here depends on the filesystem's mtime resolution, not on \
            anything this crate controls — diagnostic only, never a CI gate"]
async fn warm_hit_after_same_length_mutation_is_mtime_dependent_diagnostic() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    let dir = tiny_bert_dir(tmp.path(), "same_length_model", &cls_pooling_config());
    let source = ModelSource::local(&dir);

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d1 = assert_hashed(
        "D1 (pre-mutation, warm)",
        &guard1.model.content_digest().unwrap(),
    );
    drop(guard1);

    let pooling_config_path = dir.join("1_Pooling/config.json");
    let bytes_before = std::fs::read(&pooling_config_path).unwrap();
    let mutated = serde_json::to_string(&mean_pooling_config()).unwrap();
    assert_eq!(
        bytes_before.len(),
        mutated.len(),
        "this diagnostic's premise is a byte-length-IDENTICAL mutation"
    );
    std::fs::write(&pooling_config_path, &mutated).unwrap();

    let guard_warm = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d_warm = assert_hashed(
        "D (warm, post-mutation)",
        &guard_warm.model.content_digest().unwrap(),
    );

    assert_ne!(
        d_warm, d1,
        "on a filesystem with sub-second mtime resolution, the same-length \
         mutation is still detected via the changed mtime — if this fails, your \
         filesystem's mtime clock is coarser than the gap between the two writes \
         above, which is exactly the documented residual this diagnostic records"
    );
}

// ── F-4b (audit round 62): appearance of a newly-output-affecting file ──

/// F-4b, the `ModelCache` peer of
/// `model::backend::candle::digest_fingerprint_audit62_tests::pooling_config_appearing_after_fingerprint_trips_the_probe`:
/// `1_Pooling/config.json` APPEARING (not merely changing) between a warm
/// load and a warm replay — a bare BERT dir gets a `1_Pooling/` directory it
/// did not have at load time — must trip the staleness probe and reload,
/// switching the served pooling strategy from the load-time mean-fallback to
/// the newly-declared CLS strategy.
#[tokio::test]
async fn warm_hit_after_1_pooling_config_appearing_reloads_fresh() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    let dir = tmp.path().join("no_pooling_model");
    build_local_model_dir(&dir, None); // no 1_Pooling/ at all yet
    let source = ModelSource::local(&dir);

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d1 = assert_hashed(
        "D1 (pre-appearance, warm, mean fallback)",
        &guard1.model.content_digest().unwrap(),
    );
    let v1 = embed(&guard1.model, TEXT);
    drop(guard1);

    // 1_Pooling/config.json APPEARS — did not exist at load time.
    let pooling_dir = dir.join("1_Pooling");
    std::fs::create_dir_all(&pooling_dir).unwrap();
    std::fs::write(
        pooling_dir.join("config.json"),
        serde_json::to_string(&cls_pooling_config()).unwrap(),
    )
    .unwrap();

    let guard_warm = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d_warm = assert_hashed(
        "D (warm, post-appearance)",
        &guard_warm.model.content_digest().unwrap(),
    );
    let v_warm = embed(&guard_warm.model, TEXT);

    assert_ne!(
        d_warm, d1,
        "1_Pooling/config.json appearing after load must change the served \
         content digest on the next warm hit (F-4b)"
    );
    assert_finite_and_diverged(
        "V1 (mean fallback) vs V_warm (CLS, post-appearance)",
        &v1,
        &v_warm,
        DIVERGENCE_TOL,
    );
}

/// F-4b peer: `preprocessor_config.json` APPEARING between a warm load and a
/// warm replay must also trip the probe. `tiny_bert` is a plain BERT
/// text-embedding model — no CLAP audio tower ever reads this file — so
/// unlike the pooling case there is no pooled-vector divergence to assert;
/// the digest changing (and the warm replay picking up the CURRENT digest
/// rather than the stale load-time one) is the complete, honest observable
/// here.
#[tokio::test]
async fn warm_hit_after_preprocessor_config_appearing_reloads_fresh() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    let dir = tiny_bert_dir(tmp.path(), "no_preprocessor_model", &mean_pooling_config());
    let source = ModelSource::local(&dir);

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d1 = assert_hashed(
        "D1 (pre-appearance, warm)",
        &guard1.model.content_digest().unwrap(),
    );
    drop(guard1);

    // preprocessor_config.json APPEARS — did not exist at load time.
    write_preprocessor_config(&dir, &sample_preprocessor_config());

    let guard_warm = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d_warm = assert_hashed(
        "D (warm, post-appearance)",
        &guard_warm.model.content_digest().unwrap(),
    );

    assert_ne!(
        d_warm, d1,
        "preprocessor_config.json appearing after load must change the served \
         content digest on the next warm hit (F-4b)"
    );

    // Non-vacuous control: a COLD reading of the post-appearance dir must
    // match the warm replay's digest — proving `d_warm` is the CURRENT
    // digest, not merely "some other digest".
    let cold_model = cache
        .load_owned_for_test(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();
    let d_cold = assert_hashed(
        "D (cold, post-appearance)",
        &cold_model.content_digest().unwrap(),
    );
    assert_eq!(
        d_warm, d_cold,
        "the warm replay's digest must match a fresh cold reading of the \
         post-appearance directory"
    );
}

// ── F-3 (audit round 62): GPU permit accounting survives stale eviction \
//    while a guard is held ──

/// F-3: a stale-fingerprint eviction must NEVER release GPU memory that is
/// still genuinely resident — i.e. still reachable through a `ModelGuard`
/// that has not been dropped. Reproduces the exact defect: `warm_guard` is
/// held ACROSS the mutation and the second `get_or_load` (unlike the main
/// esc-058 test above, which drops its guard before mutating — the realistic
/// `EmbeddingPipeline::run` shape, but NOT the one that exposes this
/// accounting bug).
///
/// Budget sized to fit exactly TWO `tiny_bert` weight-file-sized models
/// (`2 * weights_len`) — tight enough that admission arithmetic is a
/// deterministic proxy for the scheduler's own internal counters (no
/// `cfg(test)`-only accessor needed): the moment BOTH the pre-mutation model
/// (still resident via `warm_guard`) and the freshly-reloaded post-mutation
/// model are admitted, `available()` must read EXACTLY 0. Pre-fix (a
/// non-`Arc`-shared permit dropped unconditionally on entry removal),
/// evicting the stale `CacheEntry` released `warm_guard`'s still-resident
/// `weights_len` bytes back to the scheduler immediately — `available()`
/// would read `weights_len` (nonzero) instead of `0`, silently reporting
/// room that does not physically exist (double-booking).
#[tokio::test]
async fn stale_eviction_never_double_books_gpu_memory_while_guard_held() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());

    let dir = tiny_bert_dir(tmp.path(), "gpu_accounting_model", &cls_pooling_config());
    let weights_len = std::fs::metadata(dir.join("model.safetensors"))
        .unwrap()
        .len() as usize;

    let scheduler = Arc::new(GpuScheduler::new(2 * weights_len, 0.0));
    let resolver =
        ModelResolver::new(Arc::clone(&catalog), crate::common::test_artifact_store()).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let cache = ModelCache::new(resolver, device_config, Arc::clone(&scheduler));
    let source = ModelSource::local(&dir);

    // (1) Warm the cache and KEEP THE GUARD (the bug-exposing shape).
    let warm_guard = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    assert_eq!(
        scheduler.available(),
        weights_len,
        "after one load, exactly one model's worth of budget should remain"
    );

    // (2) Length-changing in-place mutation (F-4a discipline) — same dir,
    // same ModelId.
    let pooling_config_path = dir.join("1_Pooling/config.json");
    let mutated = serde_json::to_string(&with_length_marker(mean_pooling_config())).unwrap();
    std::fs::write(&pooling_config_path, &mutated).unwrap();

    // (3) Second `get_or_load`, SAME session/cache/id, `warm_guard` STILL
    // HELD: the stale probe fires, evicts the `CacheEntry`, and reloads.
    let reload_guard = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();

    // Both models are now genuinely resident: the pre-mutation one through
    // `warm_guard` (not yet dropped) and the post-mutation one through
    // `reload_guard`. The scheduler must report NO remaining budget — never
    // `weights_len` (which would mean the pre-mutation model's bytes were
    // incorrectly released while still reachable).
    assert_eq!(
        scheduler.available(),
        0,
        "both the still-guard-held pre-mutation model and the freshly reloaded \
         post-mutation model are genuinely resident; the scheduler must never \
         report room that does not physically exist (F-3, no double-booking)"
    );

    // (4) Dropping the pre-mutation guard releases ITS reservation — proving
    // no PERMANENT leak (the permit eventually dies, just not while the
    // guard was live).
    drop(warm_guard);
    assert_eq!(
        scheduler.available(),
        weights_len,
        "dropping the pre-mutation guard must release its reservation exactly \
         once no more, no less (no leak, no double-release)"
    );

    // (5) Dropping `reload_guard` does NOT free further budget: unlike
    // `warm_guard`'s entry (evicted from `cache.entries` in step 3, so
    // `warm_guard` held the LAST clone of its permit), the reloaded model's
    // `CacheEntry` is still present in the warm cache — it keeps its own
    // `Arc<GpuPermit>` clone alive so a subsequent `get_or_load` can keep
    // serving it without re-reserving. This is the mirror check of step
    // (4): proves the fix does not over-release either (no double-release,
    // no premature drop of memory a still-cached entry legitimately holds).
    drop(reload_guard);
    assert_eq!(
        scheduler.available(),
        weights_len,
        "the reloaded model remains warm in the cache after its guard drops, so \
         its reservation must stay held — not released early"
    );

    // (6) No PERMANENT leak: dropping the cache itself (its last `CacheEntry`
    // clone of the reload permit) returns the scheduler to full budget.
    drop(cache);
    assert_eq!(
        scheduler.available(),
        2 * weights_len,
        "dropping the cache (its entry's last permit clone) must return the \
         scheduler to its full budget — the permit is never permanently leaked"
    );
}
