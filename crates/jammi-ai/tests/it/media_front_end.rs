//! Media front-end parallelization (#421 follow-on): the public
//! `arrow_to_images` / `arrow_to_audio` decode loops (serving) and the shared
//! `decode_*_batch` / `preprocess_*_batch` helpers they and the trainer's
//! `audio_encoder_input` / `image_encoder_input` both call.
//!
//! The pool-size {1, 5, 7, 24} bit-identity / determinism / typed-error /
//! lowest-index oracles for `preprocess_clap_fusion` and
//! `preprocess_image_batch` themselves live as crate-internal unit tests next
//! to the code (`inference::audio_preprocess` / `inference::image_preprocess`
//! `#[cfg(test)] mod tests`), because the release-mode length-check oracle
//! needs the private per-row helpers (`clap_fusion_row` / `image_row`). This
//! file covers the parts of the contract only reachable through the PUBLIC
//! surface: the null-bookkeeping / path-vs-bytes arrow arms, and the n=1
//! serving-latency measurement.

use std::sync::Arc;

use arrow::array::{ArrayRef, BinaryArray, StringArray};
use jammi_ai::fine_tune::media_front_end_pool_threads;
use jammi_ai::inference::{arrow_to_audio, arrow_to_images};

use crate::common;

fn tiny_image_corpus_dir() -> std::path::PathBuf {
    common::cookbook_fixture("tiny_image_corpus")
}

fn tiny_audio_corpus_dir() -> std::path::PathBuf {
    common::cookbook_fixture("tiny_audio_corpus")
}

// ─── `media_front_end_pool_threads` ─────────────────────────────────────────

#[test]
fn media_front_end_pool_threads_matches_rayon_current_num_threads() {
    // A trivial re-export check: the provenance field
    // (`FinetuneRunTier.rayon_pool_threads`) reads this fn, so it must be
    // exactly `rayon::current_num_threads()` — the pool SIZE, not a count of
    // threads that ran any particular batch.
    assert_eq!(media_front_end_pool_threads(), rayon::current_num_threads());
    assert!(media_front_end_pool_threads() >= 1);
}

// ─── `arrow_to_images` ───────────────────────────────────────────────────────

#[test]
fn arrow_to_images_reads_paths_and_bytes_and_preserves_nulls() {
    let corpus = tiny_image_corpus_dir();
    let path_a = corpus.join("img_circle_0.png");
    let path_b = corpus.join("img_square_0.png");
    assert!(path_a.exists(), "fixture must exist: {path_a:?}");
    assert!(path_b.exists(), "fixture must exist: {path_b:?}");

    // Path-valued column (Utf8): row 1 is null.
    let path_col: ArrayRef = Arc::new(StringArray::from(vec![
        Some(path_a.to_str().unwrap()),
        None,
        Some(path_b.to_str().unwrap()),
    ]));
    let out = arrow_to_images(&[path_col]).expect("path-valued arrow_to_images must decode");
    assert_eq!(out.len(), 3);
    assert!(out[0].is_some(), "row 0 must decode");
    assert!(out[1].is_none(), "row 1 (null) must stay None");
    assert!(out[2].is_some(), "row 2 must decode");

    // Bytes-valued column (Binary): same null in the middle.
    let bytes_a = std::fs::read(&path_a).unwrap();
    let bytes_b = std::fs::read(&path_b).unwrap();
    let binary_col: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(bytes_a.as_slice()),
        None,
        Some(bytes_b.as_slice()),
    ]));
    let out = arrow_to_images(&[binary_col]).expect("bytes-valued arrow_to_images must decode");
    assert_eq!(out.len(), 3);
    assert!(out[0].is_some());
    assert!(out[1].is_none());
    assert!(out[2].is_some());
}

#[test]
fn arrow_to_images_two_bad_binary_rows_surfaces_the_lowest_index() {
    let corpus = tiny_image_corpus_dir();
    let good = std::fs::read(corpus.join("img_circle_0.png")).unwrap();
    let bad = b"not an image at all".to_vec();

    let col: ArrayRef = Arc::new(BinaryArray::from(vec![
        good.as_slice(),
        good.as_slice(),
        bad.as_slice(), // row 2: bad
        good.as_slice(),
        bad.as_slice(), // row 4: bad
        good.as_slice(),
    ]));
    let err = arrow_to_images(&[col]).expect_err("two bad rows must error");
    let msg = err.to_string();
    assert!(
        msg.contains("row 2"),
        "expected the lowest-index (row 2) failure, got: {msg}"
    );
    assert!(
        !msg.contains("row 4"),
        "row 4's failure must not be the one surfaced: {msg}"
    );
}

// ─── `arrow_to_audio` ────────────────────────────────────────────────────────

#[test]
fn arrow_to_audio_reads_paths_and_bytes_and_preserves_nulls() {
    let corpus = tiny_audio_corpus_dir();
    let path_a = corpus.join("clip_sine_0.wav");
    let path_b = corpus.join("clip_harmonic_0.wav");
    assert!(path_a.exists(), "fixture must exist: {path_a:?}");
    assert!(path_b.exists(), "fixture must exist: {path_b:?}");

    let path_col: ArrayRef = Arc::new(StringArray::from(vec![
        Some(path_a.to_str().unwrap()),
        None,
        Some(path_b.to_str().unwrap()),
    ]));
    let out = arrow_to_audio(&[path_col]).expect("path-valued arrow_to_audio must decode");
    assert_eq!(out.len(), 3);
    assert!(out[0].is_some(), "row 0 must decode");
    assert!(out[1].is_none(), "row 1 (null) must stay None");
    assert!(out[2].is_some(), "row 2 must decode");

    let bytes_a = std::fs::read(&path_a).unwrap();
    let bytes_b = std::fs::read(&path_b).unwrap();
    let binary_col: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(bytes_a.as_slice()),
        None,
        Some(bytes_b.as_slice()),
    ]));
    let out = arrow_to_audio(&[binary_col]).expect("bytes-valued arrow_to_audio must decode");
    assert_eq!(out.len(), 3);
    assert!(out[0].is_some());
    assert!(out[1].is_none());
    assert!(out[2].is_some());
}

#[test]
fn arrow_to_audio_two_bad_binary_rows_surfaces_the_lowest_index() {
    let corpus = tiny_audio_corpus_dir();
    let good = std::fs::read(corpus.join("clip_sine_0.wav")).unwrap();
    let bad = b"not audio at all".to_vec();

    let col: ArrayRef = Arc::new(BinaryArray::from(vec![
        good.as_slice(),
        bad.as_slice(), // row 1: bad
        good.as_slice(),
        good.as_slice(),
        bad.as_slice(), // row 4: bad
        good.as_slice(),
    ]));
    let err = arrow_to_audio(&[col]).expect_err("two bad rows must error");
    let msg = err.to_string();
    assert!(
        msg.contains("row 1"),
        "expected the lowest-index (row 1) failure, got: {msg}"
    );
    assert!(
        !msg.contains("row 4"),
        "row 4's failure must not be the one surfaced: {msg}"
    );
}

// ─── n = 1 serving-latency measurement (before vs after) ───────────────────

/// The pre-unit sequential per-image write, reproduced here byte-for-byte
/// (the parallel writer's per-item math — `image_row` / `preprocess_one_image`
/// — is the SAME loop body; only the outer loop changed from sequential
/// `push` to a `par_chunks_mut` write). Used only to produce an honest
/// "before" timing baseline for the n=1 request path — the numeric behavior
/// is already covered bit-exactly by
/// `image_preprocess::tests::image_batch_parallel_matches_sequential_bit_identical_across_pool_sizes`.
fn sequential_preprocess_one(
    img: &image::DynamicImage,
    target_size: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    use image::imageops::{overlay, FilterType};
    let (w, h) = (img.width(), img.height());
    let padded = if w == h {
        img.clone()
    } else {
        let size = w.max(h);
        let mut canvas = image::DynamicImage::new_rgb8(size, size);
        if let Some(rgb) = canvas.as_mut_rgb8() {
            for pixel in rgb.pixels_mut() {
                *pixel = image::Rgb([255, 255, 255]);
            }
        }
        overlay(
            &mut canvas,
            img,
            ((size - w) / 2) as i64,
            ((size - h) / 2) as i64,
        );
        canvas
    };
    let resized = padded.resize_exact(target_size, target_size, FilterType::CatmullRom);
    let rgb = resized.to_rgb8();
    let mut out = Vec::with_capacity(3 * (target_size as usize) * (target_size as usize));
    for c in 0..3 {
        let m = mean[c];
        let s = std[c];
        for y in 0..target_size {
            for x in 0..target_size {
                let pixel = rgb.get_pixel(x, y)[c];
                out.push((pixel as f32 / 255.0 - m) / s);
            }
        }
    }
    out
}

/// Measures the n=1 image-embedding front-end request path before (the
/// sequential per-pixel loop reproduced above) and after (the shipped
/// `image_preprocess::preprocess_image_batch`, which now runs its per-item
/// write through `par_chunks_mut` — but at n=1 there is exactly one chunk, so
/// the only overhead the "after" path can add is rayon's own dispatch, not
/// contention). No `#[ignore]`: this always runs, and always prints both
/// numbers (`cargo test -p jammi-ai --test it n1_image -- --nocapture`).
#[test]
fn n1_image_request_latency_before_vs_after() {
    use jammi_ai::inference::image_preprocess::preprocess_image_batch;

    let corpus = tiny_image_corpus_dir();
    let img = image::open(corpus.join("img_circle_0.png")).expect("fixture image must decode");
    let target_size = 224u32;
    // Plausible normalization constants — this test measures latency, not
    // model-parity numerics, so full CLIP-precision constants are not needed.
    let mean = [0.481_5_f32, 0.457_8, 0.408_2];
    let std = [0.268_6_f32, 0.261_3, 0.275_8];
    let device = candle_core::Device::Cpu;

    const ITERS: u32 = 50;

    // Warm up (page faults, filter-table setup, allocator warm-up) so the
    // timed loop measures steady-state per-call cost, not one-time setup.
    sequential_preprocess_one(&img, target_size, &mean, &std);
    preprocess_image_batch(
        std::slice::from_ref(&img),
        target_size,
        &mean,
        &std,
        &device,
    )
    .unwrap();

    let before_start = std::time::Instant::now();
    for _ in 0..ITERS {
        let row = sequential_preprocess_one(&img, target_size, &mean, &std);
        assert_eq!(
            row.len(),
            3 * (target_size as usize) * (target_size as usize)
        );
    }
    let before = before_start.elapsed() / ITERS;

    let after_start = std::time::Instant::now();
    for _ in 0..ITERS {
        let t = preprocess_image_batch(
            std::slice::from_ref(&img),
            target_size,
            &mean,
            &std,
            &device,
        )
        .unwrap();
        assert_eq!(
            t.dims(),
            &[1, 3, target_size as usize, target_size as usize]
        );
    }
    let after = after_start.elapsed() / ITERS;

    println!(
        "n=1 image front-end latency (target_size={target_size}, {ITERS} iters): \
         before(sequential)={before:?}  after(par_chunks_mut, n=1)={after:?}"
    );

    // Not the pod's pre-registered ≤5% bar (that measurement is interleaved,
    // on the profile's real corpus, over N=100 steps) — this Mac-local n=1
    // check only guards against a gross regression (e.g. an accidental
    // thread-pool spin-up cost per call): "after" must stay within a
    // generous multiple of "before".
    assert!(
        after <= before * 3 + std::time::Duration::from_millis(5),
        "n=1 request latency regressed grossly: before={before:?} after={after:?}"
    );
}
