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

/// `arrow_to_images` marks each corrupt row's OWN status rather than
/// refusing the whole call — the per-row contract
/// `docs/guide/src/generate-image-embeddings.md`'s "Error handling" table
/// documents. Both bad rows surface, independently, with their OWN row
/// number; a good row that shares the batch with them still decodes.
#[test]
fn arrow_to_images_two_bad_binary_rows_each_surface_their_own_row() {
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
    let out = arrow_to_images(&[col]).expect("a per-row decode failure must not fail the batch");
    assert_eq!(out.len(), 6);
    assert!(out[0].as_ref().unwrap().is_ok(), "row 0 must decode");
    assert!(out[1].as_ref().unwrap().is_ok(), "row 1 must decode");
    let row2 = out[2].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(row2.contains("row 2"), "row 2's own error: {row2}");
    assert!(out[3].as_ref().unwrap().is_ok(), "row 3 must decode");
    let row4 = out[4].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(row4.contains("row 4"), "row 4's own error: {row4}");
    assert!(out[5].as_ref().unwrap().is_ok(), "row 5 must decode");
}

/// NULL rows AND a corrupt row in the SAME batch: the corrupt row's error
/// must name its ARROW row (4), not its position among the non-null rows (2,
/// since rows 1 and 3 are null and get compacted out before decoding).
#[test]
fn arrow_to_images_nulls_and_a_bad_row_together_report_the_arrow_row() {
    let corpus = tiny_image_corpus_dir();
    let good = std::fs::read(corpus.join("img_circle_0.png")).unwrap();
    let bad = b"not an image at all".to_vec();

    let col: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(good.as_slice()), // row 0: good
        None,                  // row 1: null
        Some(good.as_slice()), // row 2: good
        None,                  // row 3: null
        Some(bad.as_slice()),  // row 4: bad -- compacted position 2, Arrow row 4
        Some(good.as_slice()), // row 5: good
    ]));
    let out = arrow_to_images(&[col]).expect("nulls + one bad row must not fail the batch");
    assert_eq!(out.len(), 6);
    assert!(out[0].as_ref().unwrap().is_ok(), "row 0 must decode");
    assert!(out[1].is_none(), "row 1 (null) must stay None");
    assert!(out[2].as_ref().unwrap().is_ok(), "row 2 must decode");
    assert!(out[3].is_none(), "row 3 (null) must stay None");
    let row4 = out[4].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(
        row4.contains("row 4"),
        "must report the Arrow row (4), not the compacted position (2): {row4}"
    );
    assert!(
        !row4.contains("row 2:"),
        "must never report the compacted position instead of the Arrow row: {row4}"
    );
    assert!(out[5].as_ref().unwrap().is_ok(), "row 5 must decode");
}

/// The ONE documented per-row decode-failure shape
/// (`docs/guide/src/generate-image-embeddings.md`'s "Error handling" table:
/// `"Failed to decode image at row N: ..."`) must hold for a PATH-valued
/// corrupt row too, with the path appended AFTER that documented
/// prefix+cause, never spliced into the middle of it.
#[test]
fn arrow_to_images_path_valued_bad_row_keeps_the_documented_prefix_and_appends_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let bad_path = dir.path().join("corrupt.png");
    std::fs::write(&bad_path, b"not an image at all").unwrap();

    let col: ArrayRef = Arc::new(StringArray::from(vec![Some(bad_path.to_str().unwrap())]));
    let out = arrow_to_images(&[col]).expect("a decode failure is per-row, not a hard Err");
    let err = out[0].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(
        err.contains("Failed to decode image at row 0: "),
        "must keep the documented prefix verbatim: {err}"
    );
    let path_str = bad_path.to_str().unwrap();
    assert!(
        err.ends_with(&format!("(path '{path_str}')")),
        "must append the path AFTER the documented prefix+cause: {err}"
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

/// `arrow_to_audio` marks each corrupt row's OWN status rather than
/// refusing the whole call — mirroring `arrow_to_images`'s per-row contract.
#[test]
fn arrow_to_audio_two_bad_binary_rows_each_surface_their_own_row() {
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
    let out = arrow_to_audio(&[col]).expect("a per-row decode failure must not fail the batch");
    assert_eq!(out.len(), 6);
    assert!(out[0].as_ref().unwrap().is_ok(), "row 0 must decode");
    let row1 = out[1].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(row1.contains("row 1"), "row 1's own error: {row1}");
    assert!(out[2].as_ref().unwrap().is_ok(), "row 2 must decode");
    assert!(out[3].as_ref().unwrap().is_ok(), "row 3 must decode");
    let row4 = out[4].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(row4.contains("row 4"), "row 4's own error: {row4}");
    assert!(out[5].as_ref().unwrap().is_ok(), "row 5 must decode");
}

/// NULL rows AND a corrupt row in the SAME batch: the corrupt row's error
/// must name its ARROW row (4), not its position among the non-null rows (2,
/// since rows 1 and 3 are null and get compacted out before decoding).
#[test]
fn arrow_to_audio_nulls_and_a_bad_row_together_report_the_arrow_row() {
    let corpus = tiny_audio_corpus_dir();
    let good = std::fs::read(corpus.join("clip_sine_0.wav")).unwrap();
    let bad = b"not audio at all".to_vec();

    let col: ArrayRef = Arc::new(BinaryArray::from(vec![
        Some(good.as_slice()), // row 0: good
        None,                  // row 1: null
        Some(good.as_slice()), // row 2: good
        None,                  // row 3: null
        Some(bad.as_slice()),  // row 4: bad -- compacted position 2, Arrow row 4
        Some(good.as_slice()), // row 5: good
    ]));
    let out = arrow_to_audio(&[col]).expect("nulls + one bad row must not fail the batch");
    assert_eq!(out.len(), 6);
    assert!(out[0].as_ref().unwrap().is_ok(), "row 0 must decode");
    assert!(out[1].is_none(), "row 1 (null) must stay None");
    assert!(out[2].as_ref().unwrap().is_ok(), "row 2 must decode");
    assert!(out[3].is_none(), "row 3 (null) must stay None");
    let row4 = out[4].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(
        row4.contains("row 4"),
        "must report the Arrow row (4), not the compacted position (2): {row4}"
    );
    assert!(
        !row4.contains("row 2:"),
        "must never report the compacted position instead of the Arrow row: {row4}"
    );
    assert!(out[5].as_ref().unwrap().is_ok(), "row 5 must decode");
}

/// The audio peer of
/// `arrow_to_images_path_valued_bad_row_keeps_the_documented_prefix_and_appends_the_path`:
/// the documented shape (`"Failed to decode audio at row N: ..."`) and the
/// path-appended-after-the-prefix convention must hold identically for
/// audio, not just image — `arrow_to_audio` must attach a path-valued row's
/// source path to its decode failure exactly as `arrow_to_images` does.
#[test]
fn arrow_to_audio_path_valued_bad_row_keeps_the_documented_prefix_and_appends_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let bad_path = dir.path().join("corrupt.wav");
    std::fs::write(&bad_path, b"not audio at all").unwrap();

    let col: ArrayRef = Arc::new(StringArray::from(vec![Some(bad_path.to_str().unwrap())]));
    let out = arrow_to_audio(&[col]).expect("a decode failure is per-row, not a hard Err");
    let err = out[0].as_ref().unwrap().as_ref().unwrap_err().to_string();
    assert!(
        err.contains("Failed to decode audio at row 0: "),
        "must keep the documented prefix verbatim: {err}"
    );
    let path_str = bad_path.to_str().unwrap();
    assert!(
        err.ends_with(&format!("(path '{path_str}')")),
        "must append the path AFTER the documented prefix+cause: {err}"
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

/// [`sequential_preprocess_one`] plus the SAME `Tensor::from_vec` construction
/// `preprocess_image_batch`'s "after" path ends with — so "before" and
/// "after" do the same work end-to-end (pixel loop AND tensor build), not
/// just the pixel loop. Omitting the tensor build from "before" would make
/// "after" pay for work "before" never measured, silently flattering the
/// parallel path's apparent overhead.
fn sequential_preprocess_one_tensor(
    img: &image::DynamicImage,
    target_size: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &candle_core::Device,
) -> candle_core::Tensor {
    let row = sequential_preprocess_one(img, target_size, mean, std);
    let t = target_size as usize;
    candle_core::Tensor::from_vec(row, (1, 3, t, t), device)
        .expect("sequential reference tensor build must succeed")
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
    sequential_preprocess_one_tensor(&img, target_size, &mean, &std, &device);
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
        // Same work as "after": pixel loop AND tensor build (see
        // `sequential_preprocess_one_tensor`'s doc for why the tensor build
        // must be included on both sides).
        let t = sequential_preprocess_one_tensor(&img, target_size, &mean, &std, &device);
        assert_eq!(
            t.dims(),
            &[1, 3, target_size as usize, target_size as usize]
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

    // A tight 5% bar is a wall-clock, machine-load-sensitive measurement —
    // unsuitable as a hard, always-on assertion in the default suite (flaky
    // under CI contention). It is MEASURED and PRINTED on every run (see
    // above); the contract's pre-registered bar (the pod's interleaved A/B is
    // the authoritative ≤5% regression check, over N=100 steps on the real
    // corpus — this Mac-local n=1 check is a much coarser proxy for it) is
    // only ASSERTED when explicitly opted into via
    // `JAMMI_FRONTEND_N1_LATENCY=1`, so a noisy dev machine or CI runner
    // never reds the default suite on a wall-clock fluke. (This fn never
    // SKIPS — it always runs and always measures; the env var narrows only
    // the tight assertion, not reachability, so KO-7's require-gate registry
    // does not apply here.)
    //
    // A much LOOSER bar stays ALWAYS-ON, though: n=1 has no parallel work to
    // gain from (exactly one `par_chunks_mut` chunk), so "after" should never
    // be dramatically slower than "before" — a regression that big (a stray
    // per-call thread-pool install, a lock acquired every request, ...) is a
    // real bug the default suite should catch on every run, not just an
    // opted-in one. `before×3 + 5ms` is generous enough to absorb ordinary
    // CI-machine noise while still catching an order-of-magnitude regression.
    let gross_bar = before.mul_f64(3.0) + std::time::Duration::from_millis(5);
    assert!(
        after <= gross_bar,
        "n=1 request latency regressed far beyond a gross always-on bar: \
         before={before:?} after={after:?} bar(3x before + 5ms)={gross_bar:?}"
    );

    if std::env::var_os("JAMMI_FRONTEND_N1_LATENCY").is_some() {
        let bar = before.mul_f64(1.05);
        assert!(
            after <= bar,
            "n=1 request latency regressed beyond the pre-registered 5% bar: \
             before={before:?} after={after:?} bar(1.05x before)={bar:?}"
        );
    }
}
