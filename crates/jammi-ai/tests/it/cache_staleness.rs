//! esc-058 fix test (`closes_escape: esc-058`): a warm `ModelCache` must not
//! serve a pre-mutation `Arc<LoadedModel>` (stale digest + stale vectors)
//! after the underlying model directory is mutated in place — the SAME
//! session, SAME `ModelCache`, SAME `ModelId`, never dropped in between.
//!
//! Reuses `pooling_config.rs`'s hermetic `tiny_bert` fixture-copy helpers
//! (`build_local_model_dir`, `cls_pooling_config`, `mean_pooling_config`) —
//! the identical mutation `pooling_config.rs:202-263`
//! (`cls_declared_pooling_differs_from_mean_declared_pooling`, key assertion
//! at `pooling_config.rs:246-251`) already proves changes the pooled
//! vectors, so the RED/GREEN assertions below rest on a production-proven
//! trigger, not a hypothetical one.
//!
//! Reverting the esc-058 staleness probe (`ModelCache::get_or_load`'s
//! `probe_freshness` call on its fast path, back to the unconditioned
//! `cache.entries.get(&id)` hit origin/main shipped) makes
//! `warm_hit_after_in_place_mutation_reloads_fresh_digest_and_vectors`'s
//! post-fix assertions RED: the warm process would keep attesting `D1`/`V1`
//! after the mutation instead of `D2`/`V2`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

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
///
/// **Deliberately NOT a test of item 2's admission-wait fallback (unit-62
/// design pressure-test).** `2 * weights_len` is the genuine MINIMUM budget
/// this test's own scenario requires — both the pre-mutation model (via
/// `warm_guard`, held open on purpose) and the freshly-reloaded
/// post-mutation model must be able to be resident AT THE SAME TIME for the
/// `available() == 0` assertion below to mean anything, so `do_load`'s
/// admission loop here always succeeds on its very first `try_acquire` and
/// never calls `evict_one` or falls back to `GpuScheduler::acquire`'s async
/// wait at all. See
/// `stale_reload_while_guard_live_waits_for_release_under_a_realistic_budget`
/// below for the item-2-specific coverage: a budget sized to exactly ONE
/// resident copy (not two), where the SAME "guard held across a stale
/// reload" shape must WAIT for the guard's release instead of either
/// double-booking (impossible at 1x) or hard-erroring (item 2's fix).
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

// ── Unit-62 design pressure-test, item 2 (block): double-reservation \
//    liveness — a stale reload while a guard is live must WAIT, never \
//    hard-error, under a realistically-sized budget ──

/// Item 2: the stale path (`get_or_load`'s `Ok(false)` arm) evicts the
/// `CacheEntry` while `warm_guard` still holds a live clone of its
/// `Arc<GpuPermit>` — the reload that follows transiently needs this
/// model's budget TWICE (the still-reserved pre-mutation bytes, held open
/// by `warm_guard`, plus the freshly-resolved post-mutation bytes) before
/// `warm_guard` is ever dropped. Budget here is sized to exactly ONE
/// resident copy (`weights_len`, NOT `2 * weights_len` —
/// [`stale_eviction_never_double_books_gpu_memory_while_guard_held`]'s
/// budget is deliberately generous for a DIFFERENT scenario; see its own
/// doc for why that test cannot exercise this path at all): a realistic
/// production sizing, exactly what a single resident copy of this model
/// needs and no more.
///
/// RED pre-fix (`do_load`'s admission loop hard-errors as soon as
/// `evict_one` finds nothing — the stale entry is already gone by the time
/// admission runs, so there is nothing left to evict even though the
/// request is perfectly satisfiable once `warm_guard` releases): the
/// spawned reload task resolves almost immediately with
/// `Err("Cannot acquire GPU memory: nothing to evict")`, regardless of
/// whether `warm_guard` has been dropped yet — the "must still be pending"
/// assertion below fails (the task has already completed), and the final
/// join fails too (`Err`, never `Ok`).
///
/// GREEN post-fix: `do_load` falls back to `GpuScheduler::acquire`'s async
/// wait once `evict_one` finds nothing AND the request is within the
/// scheduler's total usable budget (`memory_bytes <= usable_capacity()` —
/// here `memory_bytes == weights_len == usable_capacity()`, so this is NOT
/// the genuinely-over-budget case) — the reload genuinely blocks until
/// `warm_guard`'s `Drop` releases its permit clone and calls
/// `notify_waiters`, then succeeds.
#[tokio::test]
async fn stale_reload_while_guard_live_waits_for_release_under_a_realistic_budget() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());

    let dir = tiny_bert_dir(
        tmp.path(),
        "realistic_budget_wait_model",
        &cls_pooling_config(),
    );
    let weights_len = std::fs::metadata(dir.join("model.safetensors"))
        .unwrap()
        .len() as usize;

    // Realistic budget: exactly ONE resident copy, no slack for a second,
    // transient one.
    let scheduler = Arc::new(GpuScheduler::new(weights_len, 0.0));
    let resolver =
        ModelResolver::new(Arc::clone(&catalog), crate::common::test_artifact_store()).unwrap();
    let device_config = DeviceConfig {
        gpu_device: -1,
        memory_fraction: 1.0,
        require_gpu: false,
        compute_precision: jammi_numerics::ComputePrecision::F32,
    };
    let cache = Arc::new(ModelCache::new(
        resolver,
        device_config,
        Arc::clone(&scheduler),
    ));
    let source = ModelSource::local(&dir);

    // (1) Warm the cache and KEEP THE GUARD — the budget is now fully
    // reserved by `warm_guard` alone.
    let warm_guard = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    assert_eq!(
        scheduler.available(),
        0,
        "the realistic 1x budget is fully reserved by warm_guard alone"
    );

    // (2) Length-changing in-place mutation (F-4a discipline) — trips the
    // staleness probe on the next call.
    let pooling_config_path = dir.join("1_Pooling/config.json");
    let mutated = serde_json::to_string(&with_length_marker(mean_pooling_config())).unwrap();
    std::fs::write(&pooling_config_path, &mutated).unwrap();

    // (3) Spawn the reload WHILE `warm_guard` is still held — this is the
    // exact shape that needs the model's budget TWICE, transiently.
    let cache_for_reload = Arc::clone(&cache);
    let source_for_reload = source.clone();
    let mut reload_task = tokio::spawn(async move {
        cache_for_reload
            .get_or_load(&source_for_reload, ModelTask::TextEmbedding, None)
            .await
    });

    // (4) The reload must NOT complete yet: `warm_guard` is still held, the
    // budget cannot admit a second resident copy, and the fixed behavior
    // waits rather than erroring. A generous 500ms bound — far more than a
    // hermetic tiny_bert resolve+load needs, but nowhere near long enough
    // to look like a hang — turns a REGRESSION (the pre-fix hard error,
    // which returns almost immediately) into a fast, clear assertion
    // failure instead of relying on indefinite blocking to "prove" a wait.
    let still_pending = tokio::time::timeout(Duration::from_millis(500), &mut reload_task).await;
    match still_pending {
        Err(_elapsed) => {
            // Timed out waiting for completion — genuinely still pending,
            // exactly what a WAIT (not a hard error) looks like.
        }
        Ok(joined) => match joined.unwrap() {
            Ok(_guard) => panic!(
                "the reload completed (Ok) BEFORE warm_guard was dropped, while \
                 the realistic 1x budget could not possibly admit a second \
                 resident copy — this should be structurally impossible either \
                 way, fixed or not"
            ),
            Err(e) => panic!(
                "the reload completed BEFORE warm_guard was dropped, while the \
                 realistic 1x budget could not possibly admit a second resident \
                 copy — this can only mean the pre-fix hard error path ran \
                 instead of waiting (item 2's regression): got Err({e})"
            ),
        },
    }

    // (5) Release the budget: dropping `warm_guard` frees its reservation
    // and wakes the waiting reload via `GpuPermit::drop`'s
    // `notify_waiters()`.
    drop(warm_guard);

    // (6) The reload must now complete successfully, within a generous
    // bound.
    let joined = tokio::time::timeout(Duration::from_secs(10), reload_task).await;
    let result = match joined {
        Ok(joined) => joined.unwrap(),
        Err(_) => panic!(
            "the reload never completed within 10s after warm_guard's release — \
             a hang would mean GpuScheduler::acquire's wait never woke, which \
             would itself be a distinct regression from the hard-error bug this \
             test targets"
        ),
    };
    match result {
        Ok(guard) => drop(guard),
        Err(e) => panic!(
            "the reload must succeed once warm_guard's reservation is released \
             — never a hard error on a request that is perfectly satisfiable \
             once the outgoing guard's memory is actually freed (item 2) — got \
             Err({e})"
        ),
    }
}

// ── F-4' (audit round 62, adversarial round 3): a deleted OPTIONAL \
//    candidate must reload fresh, never wedge; a deleted REQUIRED \
//    candidate keeps the typed refusal, and stays a typed refusal — never \
//    a permanent wedge into a different failure mode ──

/// F-4' core: `1_Pooling/config.json` is OPTIONAL (the loader has a
/// documented mean-pooling fallback for its absence). Deleting it from a
/// LIVE, load-time-CLS-declared model directory must make the next warm
/// `get_or_load` reload FRESH — new digest, mean-fallback vectors — never a
/// typed refusal, and never a permanent wedge (two consecutive warm calls
/// after the deletion must both succeed with stable, matching output).
///
/// Pre-F-4', `ModelFingerprint::probe` collapsed this exact case
/// (present-at-load, `NotFound`-now) into `Err` regardless of optionality,
/// and `ModelCache::get_or_load`'s `Err` arm never evicts — so the SAME
/// unusable `CacheEntry` stayed cached forever, and every subsequent
/// `get_or_load` on this id re-hit the identical `Err`, even though a cold
/// reload (bypassing the cache) would have succeeded via the mean fallback
/// the whole time. RED there: this test's first post-deletion `get_or_load`
/// would return `Err`, never `Ok`.
#[tokio::test]
async fn warm_hit_after_optional_pooling_config_deleted_reloads_fresh_never_wedges() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    // Load-time: CLS-declared (1_Pooling/config.json present and read).
    let dir = tiny_bert_dir(tmp.path(), "optional_deleted_model", &cls_pooling_config());
    let source = ModelSource::local(&dir);

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let d1 = assert_hashed(
        "D1 (pre-deletion, CLS, warm)",
        &guard1.model.content_digest().unwrap(),
    );
    drop(guard1);

    // DELETE the optional candidate — present at load time, gone now.
    std::fs::remove_file(dir.join("1_Pooling/config.json")).unwrap();

    // First warm call after deletion: must reload FRESH (not error, not
    // silently keep serving the pre-deletion CLS-pooled vectors).
    let guard_warm_1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await;
    let guard_warm_1 = match guard_warm_1 {
        Ok(g) => g,
        Err(e) => panic!(
            "an OPTIONAL candidate (1_Pooling/config.json) deleted between load \
             and probe must reload fresh via the mean-pooling fallback, never a \
             typed refusal (F-4') — got Err({e})"
        ),
    };
    let d_warm_1 = assert_hashed(
        "D (warm, post-deletion, 1st call)",
        &guard_warm_1.model.content_digest().unwrap(),
    );
    let v_warm_1 = embed(&guard_warm_1.model, TEXT);
    drop(guard_warm_1);

    assert_ne!(
        d_warm_1, d1,
        "deleting the optional 1_Pooling/config.json must change the served \
         content digest (it drops out of the digest's gated input set)"
    );

    // Non-vacuous control: the post-deletion vectors must match an
    // independent, freshly-resolved model with NO 1_Pooling/ directory at
    // all — the exact mean-fallback shape `pooling_config.rs:202-263`
    // (`cls_declared_pooling_differs_from_mean_declared_pooling`, control
    // (3) at :253-262) already proves matches a mean-declared model.
    let reference_dir = tmp.path().join("reference_no_pooling_dir");
    build_local_model_dir(&reference_dir, None);
    let reference_model = cache
        .load_owned_for_test(
            &ModelSource::local(&reference_dir),
            ModelTask::TextEmbedding,
        )
        .await
        .unwrap();
    let v_reference = embed(&reference_model, TEXT);
    assert_eq!(
        v_warm_1, v_reference,
        "post-deletion vectors must be bitwise-identical to an independent \
         no-pooling-file (mean-fallback) reference model — proving the reload \
         actually took the mean-pooling fallback, not merely 'some other' \
         digest/vectors"
    );

    // Second non-vacuous control: a COLD reading of `source` taken AFTER
    // the deletion (bypassing the cache entirely, independent re-resolve +
    // re-load) must match the warm reload's digest and vectors — proving
    // `d_warm_1`/`v_warm_1` are the CURRENT (post-deletion) state, not an
    // artifact of the cache path specifically.
    let cold_model = cache
        .load_owned_for_test(&source, ModelTask::TextEmbedding)
        .await
        .unwrap();
    let d_cold = assert_hashed(
        "D (cold, post-deletion)",
        &cold_model.content_digest().unwrap(),
    );
    let v_cold = embed(&cold_model, TEXT);
    assert_eq!(
        d_cold, d_warm_1,
        "a cold reading taken after the deletion must match the warm reload's digest"
    );
    assert_eq!(
        v_cold, v_warm_1,
        "a cold reading taken after the deletion must match the warm reload's vectors"
    );

    // Never a permanent wedge: a SECOND warm call after the reload must
    // also succeed, with STABLE output (not merely 'succeeds once, then
    // starts failing' or vice versa).
    let guard_warm_2 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await;
    let guard_warm_2 = match guard_warm_2 {
        Ok(g) => g,
        Err(e) => panic!(
            "a SECOND warm get_or_load after the optional-candidate reload must \
             also succeed — never a permanent wedge (F-4') — got Err({e})"
        ),
    };
    let d_warm_2 = assert_hashed(
        "D (warm, post-deletion, 2nd call)",
        &guard_warm_2.model.content_digest().unwrap(),
    );
    assert_eq!(
        d_warm_2, d_warm_1,
        "the second warm call after the reload must report the SAME (now \
         stable, mean-fallback) digest — no further reload should be needed \
         since nothing changed on disk between the two calls"
    );
}

/// F-4' control, UPDATED for unit-62 design pressure-test item 3
/// (evict-on-`Err`, wedge elimination): `model.safetensors` is a REQUIRED
/// candidate — the loader has no fallback for missing weights. Deleting it
/// must keep the EXISTING typed refusal (K2) behavior on the FIRST
/// post-deletion call — the probe's own "no longer readable" message.
/// Calling `get_or_load` TWICE after the deletion must refuse BOTH times
/// (never silently start succeeding), but the two refusals are no longer
/// required — nor expected — to be byte-identical: `probe`'s `Err` arm now
/// evicts the `CacheEntry` before returning (see `ModelFingerprint::probe`'s
/// arm-(c) doc), so the SECOND call is cold-equivalent — it takes the full
/// resolve + load path instead of re-probing the identical dead entry, and
/// hits `ModelResolver::resolve_local`'s OWN typed "no weights found" error
/// instead of the probe's "no longer readable" one. Same observable
/// contract (refusal, deterministically, both times), different message —
/// proving there is no permanent wedge on this identical stale entry.
///
/// RED pre-fix (revert item 3's `evict_if_current` call in the `Err` arm):
/// the entry stays cached after the first `Err`, so the SECOND call
/// re-probes the SAME entry and re-hits the identical "no longer readable"
/// message — `msg_2` would equal `msg_1` and `msg_2` would still contain
/// "no longer readable", both assertions below inverted from what this test
/// now asserts.
#[tokio::test]
async fn warm_hit_after_required_weights_deleted_evicts_and_second_call_hits_the_load_path() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    let dir = tiny_bert_dir(tmp.path(), "required_deleted_model", &mean_pooling_config());
    let source = ModelSource::local(&dir);

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    drop(guard1);

    // DELETE the required candidate.
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();

    let result_1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await;
    let msg_1 = match result_1 {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "deleting the REQUIRED model.safetensors must refuse — a reload \
             cannot possibly succeed without the weights file"
        ),
    };
    assert!(
        msg_1.contains("no longer readable"),
        "expected the esc-058 staleness-probe's typed refusal message on the \
         FIRST post-deletion call (the probe itself is what detects the \
         deletion), got: {msg_1}"
    );

    // SECOND, CONSECUTIVE call: the wedge check. Pre-fix, this would hit
    // the identical still-cached, still-dead entry and re-`Err` with the
    // SAME probe message forever. Post-fix (item 3), the first call's `Err`
    // arm evicted the entry, so this call is cold-equivalent — it must
    // still refuse (the weights file is still gone), but via the LOADER's
    // own typed error, never the probe's.
    let result_2 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await;
    let msg_2 = match result_2 {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "a SECOND get_or_load after the first typed refusal must ALSO \
             refuse — the weights file is still missing, so a cold reload \
             cannot possibly succeed either — never silently start \
             succeeding with a corrupted or half-loaded state"
        ),
    };
    assert!(
        msg_2.contains("No model weights found"),
        "expected the SECOND call's refusal to come from \
         `ModelResolver::resolve_local`'s own typed error (proving the \
         entry was evicted and this call took the cold path), not the \
         probe's — got: {msg_2}"
    );
    assert_ne!(
        msg_1, msg_2,
        "the two refusals must come from DIFFERENT code paths (probe vs. \
         cold resolve) — evict-on-Err (item 3) means the second call never \
         re-probes the same dead entry, so it never reproduces the FIRST \
         call's exact message; a wedge would keep serving byte-identical \
         messages here forever"
    );
}

// ── Round 10 (audit round 62, adversarial round 10 — "the terminal class \
//    closure"): the weights slot's alternate-arm appearance/deletion, at \
//    the full ModelCache level ──

/// (a) appearance, end-to-end: `model.onnx` appears beside an already-loaded
/// `model.safetensors`. The warm path must NOT silently keep serving the
/// pre-appearance `Arc<LoadedModel>` forever: it must detect the staleness
/// and evict + genuinely reload.
///
/// **What this test can honestly assert at the `ModelCache` level**: by the
/// time this model's SECOND `get_or_load` runs, `do_load`'s first call has
/// already registered it in the catalog (`register_model`, with the
/// persisted `backend` it loaded at — Candle). `ModelResolver::resolve`
/// checks the catalog FIRST (`try_catalog_lookup`), so THIS reload reuses
/// the persisted Candle/`model.safetensors` record rather than re-deriving
/// `resolve_local`'s `has_onnx` heuristic from scratch — the backend-flip
/// itself only happens for an resolve with no catalog record to short-circuit
/// it, proven independently, unregistered, in `models.rs`'s
/// `resolve_local_prefers_onnx_once_it_appears_alongside_existing_safetensors`.
/// What THIS test proves is the piece that mechanism actually gates: the
/// stale-fingerprint eviction genuinely fires and a NEW `Arc<LoadedModel>` is
/// constructed (never the SAME cached instance silently handed back) —
/// proven by `Arc::ptr_eq` inequality between the two guards' `model`
/// handles, the direct, non-vacuous signal that a real evict+reload cycle
/// ran rather than the fast path short-circuiting on a stale-but-undetected
/// probe.
///
/// **Framing correction (unit-62 design pressure-test, item 4a).** The
/// "persisted record" language above must not be read as the catalog row
/// being some fixed, identity-pinned fact once written. `Catalog::register_model`
/// (`model_repo.rs:159-167`) is an UNCONDITIONAL UPSERT on every load:
/// `backend`, `task`, and `model_type` are overwritten with
/// `excluded.<col>` on every `ON CONFLICT`, last-writer-wins — only
/// `artifact_path` gets the `COALESCE(excluded, existing)` set-but-never-
/// clear treatment. So this test's observed pinning (the reload keeps
/// resolving Candle/`model.safetensors` rather than flipping to ORT/
/// `model.onnx`) is NOT a guarantee that the catalog row is immutable or
/// that re-registration is a no-op; it is a consequence of
/// `try_catalog_lookup`'s catalog-FIRST precedence over `resolve_local`'s
/// on-disk heuristic, for THIS specific model's second load — a different
/// sequence (e.g. a load with a different `backend_hint`, which
/// `try_catalog_lookup` honors over the persisted `backend`) can and does
/// re-derive a different backend on the very next call. Cache-key
/// narrower-than-resolve-key (backend_hint mismatch specifically) is one of
/// unit 65's classes, not something this test's pinning observation
/// contradicts or resolves.
///
/// RED pre-round-10: the weights slot's fingerprint only ever tracked the
/// SELECTED arm (`model.safetensors`) — `model.onnx` appearing was invisible
/// to `probe`, which reported fresh forever, so the second call would return
/// the IDENTICAL `Arc` as the first (this test's `Arc::ptr_eq` assertion
/// would be `true`, not `false`), silently masking the appearance entirely.
#[tokio::test]
async fn warm_hit_after_model_onnx_appearing_beside_safetensors_evicts_and_genuinely_reloads() {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    let dir = tiny_bert_dir(tmp.path(), "onnx_appears_model", &mean_pooling_config());
    let source = ModelSource::local(&dir);

    // (1) Cold load: only model.safetensors exists — Candle backend.
    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let model1 = Arc::clone(&guard1.model);
    drop(guard1);

    // (2) model.onnx APPEARS beside the existing model.safetensors — the
    // UNSELECTED arm of the weights slot.
    std::fs::write(dir.join("model.onnx"), b"fake-onnx-bytes").unwrap();

    // (3) The warm path must evict and genuinely reload rather than silently
    // keep serving the pre-appearance `Arc`.
    let guard2 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .expect(
            "model.onnx appearing must trip a stale-reload — which succeeds here \
             because try_catalog_lookup's persisted record keeps resolving Candle/ \
             model.safetensors (both still present and unchanged) — never a refusal",
        );
    assert!(
        !Arc::ptr_eq(&model1, &guard2.model),
        "model.onnx appearing beside model.safetensors must trip the staleness probe \
         and produce a GENUINELY NEW Arc<LoadedModel> on the next get_or_load — the \
         SAME Arc being handed back (Arc::ptr_eq == true) would mean the appearance \
         was never detected (the round-10 defect) and the fast path silently kept \
         serving the pre-appearance instance forever"
    );
}

/// (b) deletion-with-alternate, end-to-end: the SELECTED weights arm
/// (`model.safetensors`) is deleted while an alternate
/// (`open_clip_model.safetensors`) survives — a cold resolve's own
/// standard/open_clip fallback chain (`resolve_local`) succeeds via the
/// alternate. Two consecutive warm calls after the deletion must both
/// succeed — never wedge, never refuse.
///
/// RED pre-round-10: the weights slot's fingerprint only ever tracked the
/// SELECTED arm; deleting it fell into the sole "REQUIRED, no fallback"
/// treatment every weights path had (`optional: false` unconditionally,
/// with no per-slot alternate-exists carve-out), so this refused instead of
/// reloading via the alternate a cold resolve would happily use.
#[tokio::test]
async fn warm_hit_after_selected_weights_arm_deleted_with_alternate_present_reloads_via_alternate_never_refuses(
) {
    let tmp = tempdir().unwrap();
    let catalog_dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(catalog_dir.path()).await.unwrap());
    let cache = new_cache(Arc::clone(&catalog));

    let dir = tiny_bert_dir(
        tmp.path(),
        "weights_alternate_model",
        &mean_pooling_config(),
    );
    let source = ModelSource::local(&dir);

    // An alternate arm (open_clip_model.safetensors) ALSO exists on disk,
    // byte-identical to model.safetensors — untracked by the pre-round-10
    // fingerprint (only the SELECTED arm was ever a candidate).
    std::fs::copy(
        dir.join("model.safetensors"),
        dir.join("open_clip_model.safetensors"),
    )
    .unwrap();

    let guard1 = cache
        .get_or_load(&source, ModelTask::TextEmbedding, None)
        .await
        .unwrap();
    let model1 = Arc::clone(&guard1.model);
    drop(guard1);

    // DELETE the selected arm — the alternate survives.
    std::fs::remove_file(dir.join("model.safetensors")).unwrap();

    // Two consecutive warm calls must both succeed, never wedge and never
    // refuse — a cold resolve of this SAME directory would happily pick up
    // open_clip_model.safetensors via resolve_local's own fallback chain.
    //
    // esc-058 fix-verifier fold-in: `Ok(_)` alone is satisfiable by a bug
    // that silently keeps serving the pre-deletion `Arc` without ever
    // actually reloading (the staleness probe never tripping would look
    // identical to an observer that only checks `Ok`/`Err`) — pin the FIRST
    // post-deletion call to a genuinely NEW `Arc<LoadedModel>` via
    // `Arc::ptr_eq` inequality, proving a real reload occurred rather than
    // the assertion being vacuously satisfied by an untouched warm hit.
    for attempt in 0..2 {
        let result = cache
            .get_or_load(&source, ModelTask::TextEmbedding, None)
            .await;
        match result {
            Ok(guard) => {
                if attempt == 0 {
                    assert!(
                        !Arc::ptr_eq(&model1, &guard.model),
                        "the first post-deletion get_or_load must genuinely reload \
                         via the alternate arm, not silently keep serving the \
                         pre-deletion Arc — Arc::ptr_eq == true would mean the \
                         staleness probe never actually tripped, making the \
                         surrounding Ok/Err check vacuous"
                    );
                }
                drop(guard);
            }
            Err(e) => panic!(
                "attempt #{attempt}: deleting the SELECTED weights arm \
                 (model.safetensors) while an alternate (open_clip_model.safetensors) \
                 still exists must reload via the alternate, never refuse — a cold \
                 resolve would succeed via resolve_local's own fallback chain (round \
                 62, adversarial round 10) — got Err({e})"
            ),
        }
    }
}
