//! Image preprocessing for vision models.
//!
//! Provides pad-to-square, resize, and normalize transforms.
//! Preprocessing parameters (mean, std) are model-driven, not hardcoded.

use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use image::DynamicImage;
use jammi_db::error::{JammiError, Result};
use rayon::prelude::*;

/// Decode a batch of encoded image byte buffers in PARALLEL, across whichever
/// rayon pool the call runs under (its global pool in production; candle
/// installs no private pool of its own, so this is the one pool the process
/// ever schedules media-batch work on).
///
/// Used by `fine_tune::trainer`'s `image_encoder_input`, whose items are
/// already the row space the caller wants reported (no compaction ever
/// happens ahead of it — every `MediaTriplet` blob is present by
/// construction), so this thin wrapper reports positions `0..items.len()`
/// via [`decode_image_batch_indexed`]. A caller that compacts rows first
/// (dropping nulls before decoding, as [`super::arrow_to_images`] and
/// `CandleBackend::forward_image_embedding` both do) must call
/// [`decode_image_batch_indexed`] directly with the ORIGINAL row ids, or
/// every error message after the first null row reports the wrong (compacted)
/// position instead of the caller's row.
pub fn decode_image_batch<T>(items: &[T]) -> Result<Vec<DynamicImage>>
where
    T: AsRef<[u8]> + Sync,
{
    let row_ids: Vec<usize> = (0..items.len()).collect();
    decode_image_batch_indexed(&row_ids, items)
}

/// [`decode_image_batch`], reporting `row_ids[i]` (not the position `i`) in
/// every error — for a caller that has already compacted its batch (dropped
/// null rows) before decoding, so the position in `items` and the row the
/// caller (and its own caller, in turn) means by "row N" have diverged.
/// `row_ids.len()` must equal `items.len()`.
///
/// One decode task per item — chunk count is `items.len()`, no thread-count
/// knob, so the effective parallelism is `min(pool_size, items.len())`,
/// emergent from whichever pool is installed. Errors are collected per row
/// and the LOWEST-INDEX failing row is the one surfaced, with a row-indexed
/// message — the same selection the pre-unit sequential decode loop made by
/// construction (it returned on the first failure it walked into, in row
/// order); `row_ids` only changes what number a row is called, never the
/// selection order.
pub fn decode_image_batch_indexed<T>(row_ids: &[usize], items: &[T]) -> Result<Vec<DynamicImage>>
where
    T: AsRef<[u8]> + Sync,
{
    crate::inference::lowest_index_result(decode_image_results(row_ids, items))
}

/// [`decode_image_batch_indexed`]'s per-row outcomes, WITHOUT collapsing to
/// the lowest-index failure — for a caller (serving's `arrow_to_images`) that
/// marks each row's OWN status rather than refusing the whole batch because
/// one row's bytes were corrupt. Every element is independently `Ok` or a
/// row-indexed `Err`, in the same order as `items`/`row_ids`.
pub fn decode_image_batch_per_row_indexed<T>(
    row_ids: &[usize],
    items: &[T],
) -> Vec<Result<DynamicImage>>
where
    T: AsRef<[u8]> + Sync,
{
    decode_image_results(row_ids, items)
}

/// The parallel per-row decode shared by [`decode_image_batch_indexed`] and
/// [`decode_image_batch_per_row_indexed`] — the only difference between the
/// two public entry points is whether the caller wants ALL per-row outcomes
/// or just the lowest-index failure.
fn decode_image_results<T>(row_ids: &[usize], items: &[T]) -> Vec<Result<DynamicImage>>
where
    T: AsRef<[u8]> + Sync,
{
    debug_assert_eq!(
        row_ids.len(),
        items.len(),
        "row_ids must have one entry per item"
    );
    items
        .par_iter()
        .zip(row_ids.par_iter())
        .map(|(item, &row)| {
            image::load_from_memory(item.as_ref()).map_err(|e| {
                JammiError::Inference(format!("Failed to decode image at row {row}: {e}"))
            })
        })
        .collect()
}

/// One image's normalized pixel row: pad to square → resize → normalize,
/// `[3, side, side]` row-major (channel-major / CHW), `side = target_size`.
fn preprocess_one_image(
    img: &DynamicImage,
    target_size: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Vec<f32> {
    let padded = pad_to_square(img);
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

/// [`preprocess_one_image`] plus a verification of its length against
/// `expected_len` — the caller's preallocated, fixed-stride chunk. This is
/// the release-mode per-item length check the parallel writer in
/// [`preprocess_image_batch`] depends on (a plain `if`, not `debug_assert!`,
/// so it still runs in release builds): `preprocess_one_image`'s output
/// length is always `expected_len` for a consistent `target_size` in
/// practice, but a future change to its pixel-loop arithmetic must fail
/// loudly here rather than silently truncate/short-`copy_from_slice`/misalign
/// another image's chunk.
fn image_row(
    row_index: usize,
    img: &DynamicImage,
    target_size: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    expected_len: usize,
) -> Result<Vec<f32>> {
    let row = preprocess_one_image(img, target_size, mean, std);
    if row.len() != expected_len {
        return Err(JammiError::Inference(format!(
            "Image preprocessing row {row_index}: produced {} values, expected {expected_len}",
            row.len()
        )));
    }
    Ok(row)
}

/// Preprocess a batch of images into a model-ready tensor.
///
/// Each image is: padded to square (white) → resized to `target_size` → normalized.
/// Returns tensor of shape `(batch, 3, target_size, target_size)`.
///
/// Reports row positions `0..images.len()` in any per-row error via
/// [`preprocess_image_batch_indexed`]; see that function's doc for the
/// compacted-batch case (`CandleBackend::forward_image_embedding` calls it
/// directly with the original Arrow row ids for exactly that reason).
pub fn preprocess_image_batch(
    images: &[DynamicImage],
    target_size: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &Device,
) -> Result<Tensor> {
    let row_ids: Vec<usize> = (0..images.len()).collect();
    preprocess_image_batch_indexed(&row_ids, images, target_size, mean, std, device)
}

/// [`preprocess_image_batch`], reporting `row_ids[i]` (not the position `i`)
/// in every per-row error — for a caller that has already compacted its
/// batch (dropped null rows) before preprocessing, so the position in
/// `images` and the row its own caller means by "row N" have diverged.
/// `row_ids.len()` must equal `images.len()`.
///
/// Preallocates the whole batch's flat buffer, then writes each image's
/// disjoint, fixed-stride `pixels_per_image` chunk in PARALLEL across
/// whichever rayon pool this call runs under (`par_chunks_mut` zipped with
/// the images — one task per image, no thread-count knob). `row_ids` only
/// changes what number a row is called in an error message; it never
/// reorders the chunk a given image writes to, so it cannot change the
/// numeric output.
pub fn preprocess_image_batch_indexed(
    row_ids: &[usize],
    images: &[DynamicImage],
    target_size: u32,
    mean: &[f32; 3],
    std: &[f32; 3],
    device: &Device,
) -> Result<Tensor> {
    if images.is_empty() {
        return Err(JammiError::Inference(
            "Cannot preprocess empty image batch".into(),
        ));
    }
    if target_size == 0 {
        return Err(JammiError::Inference(
            "Image target_size must be positive, got 0".into(),
        ));
    }
    debug_assert_eq!(
        row_ids.len(),
        images.len(),
        "row_ids must have one entry per image"
    );

    let pixels_per_image = 3 * (target_size as usize) * (target_size as usize);
    let mut flat = vec![0f32; images.len() * pixels_per_image];

    let results: Vec<Result<()>> = flat
        .par_chunks_mut(pixels_per_image)
        .zip(images.par_iter())
        .zip(row_ids.par_iter())
        .map(|((chunk, img), &row)| {
            let row_values = image_row(row, img, target_size, mean, std, chunk.len())?;
            chunk.copy_from_slice(&row_values);
            Ok(())
        })
        .collect();
    crate::inference::lowest_index_result(results)?;

    let t = target_size as usize;
    Tensor::from_vec(flat, (images.len(), 3, t, t), device)
        .map_err(|e| JammiError::Inference(format!("Failed to create image tensor: {e}")))
}

/// Pad an image to a square with white background, centered.
fn pad_to_square(img: &DynamicImage) -> DynamicImage {
    let (w, h) = (img.width(), img.height());
    if w == h {
        return img.clone();
    }

    let size = w.max(h);
    let mut canvas = DynamicImage::new_rgb8(size, size);

    // Fill with white
    if let Some(rgb) = canvas.as_mut_rgb8() {
        for pixel in rgb.pixels_mut() {
            *pixel = image::Rgb([255, 255, 255]);
        }
    }

    let paste_x = (size - w) / 2;
    let paste_y = (size - h) / 2;
    image::imageops::overlay(&mut canvas, img, paste_x as i64, paste_y as i64);
    canvas
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time `Send` assertion (K4/family-J precedent:
    /// `jammi-kernels/src/ops/saved.rs`'s `saved_is_send_and_sync`): the type
    /// that crosses the parallel decode/preprocess boundary in
    /// [`decode_image_batch`] and [`preprocess_image_batch`] must be `Send`,
    /// or the crate would not compile — this only names the requirement, it
    /// never silently loosens it.
    #[test]
    fn dynamic_image_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DynamicImage>();
        assert_send::<Result<DynamicImage>>();
    }

    #[allow(clippy::excessive_precision)]
    const TEST_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const TEST_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn test_image(w: u32, h: u32) -> DynamicImage {
        DynamicImage::new_rgb8(w, h)
    }

    #[test]
    fn test_pad_to_square_landscape() {
        let img = test_image(200, 100);
        let padded = pad_to_square(&img);
        assert_eq!(padded.width(), 200);
        assert_eq!(padded.height(), 200);
    }

    #[test]
    fn test_pad_to_square_portrait() {
        let img = test_image(100, 300);
        let padded = pad_to_square(&img);
        assert_eq!(padded.width(), 300);
        assert_eq!(padded.height(), 300);
    }

    #[test]
    fn test_pad_to_square_already_square() {
        let img = test_image(150, 150);
        let padded = pad_to_square(&img);
        assert_eq!(padded.width(), 150);
        assert_eq!(padded.height(), 150);
    }

    #[test]
    fn test_preprocess_batch_shape() {
        let images = vec![test_image(100, 200), test_image(300, 150)];
        let tensor =
            preprocess_image_batch(&images, 224, &TEST_MEAN, &TEST_STD, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[2, 3, 224, 224]);
    }

    #[test]
    fn test_preprocess_normalization_range() {
        // White image: pixel=255 -> (1.0 - mean) / std
        let img = {
            let mut d = DynamicImage::new_rgb8(10, 10);
            if let Some(rgb) = d.as_mut_rgb8() {
                for p in rgb.pixels_mut() {
                    *p = image::Rgb([255, 255, 255]);
                }
            }
            d
        };

        let tensor =
            preprocess_image_batch(&[img], 4, &TEST_MEAN, &TEST_STD, &Device::Cpu).unwrap();
        let vals = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let expected_ch0 = (1.0 - TEST_MEAN[0]) / TEST_STD[0];
        assert!((vals[0] - expected_ch0).abs() < 0.01);
    }

    #[test]
    fn test_preprocess_empty_batch_errors() {
        let result = preprocess_image_batch(&[], 224, &TEST_MEAN, &TEST_STD, &Device::Cpu);
        assert!(result.is_err());
    }

    // -- Media front-end parallelization (#421 follow-on) --------------------

    /// Encode a small solid-color RGB image as real PNG bytes, for decode
    /// tests that need genuine (not garbage) encoded bytes.
    fn png_bytes(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut img = DynamicImage::new_rgb8(w, h);
        if let Some(buf) = img.as_mut_rgb8() {
            for pixel in buf.pixels_mut() {
                *pixel = image::Rgb(rgb);
            }
        }
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    #[test]
    fn decode_image_batch_empty_is_ok_empty() {
        // Decode-stage empty is a no-op (K2's "empty batch refused" guard
        // lives at the PREPROCESS stage — see `test_preprocess_empty_batch_errors`
        // above — matching the pre-unit sequential decode loop, which also
        // never rejected zero rows).
        let items: Vec<Vec<u8>> = Vec::new();
        let decoded = decode_image_batch(&items).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn decode_image_batch_two_bad_rows_surfaces_the_lowest_index() {
        let good = png_bytes(4, 4, [10, 20, 30]);
        let items: Vec<Vec<u8>> = vec![
            good.clone(),
            b"not an image at all".to_vec(), // row 1: bad
            good.clone(),
            good.clone(),
            b"also not an image".to_vec(), // row 4: bad
            good,
        ];
        let err = decode_image_batch(&items).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("row 1"),
            "expected the lowest-index (row 1) failure, got: {msg}"
        );
        assert!(
            !msg.contains("row 4"),
            "row 4's failure must not be the one surfaced when row 1 also failed: {msg}"
        );
    }

    /// The row-index bug this fold closes: a caller that COMPACTS its batch
    /// (drops null rows) before decoding must get the ORIGINAL row back in
    /// the error, not the position in the compacted `items` slice. Position 1
    /// (the bad row) is deliberately given a row id far from its position.
    #[test]
    fn decode_image_batch_indexed_reports_the_given_row_id_not_the_position() {
        let good = png_bytes(4, 4, [1, 2, 3]);
        let items: Vec<Vec<u8>> = vec![good.clone(), b"not an image".to_vec(), good];
        let row_ids = vec![10usize, 42, 99];

        let err = decode_image_batch_indexed(&row_ids, &items).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("row 42"),
            "expected the caller's row id (42), got: {msg}"
        );
        assert!(
            !msg.contains("row 1:"),
            "must never report the compacted position instead of the row id: {msg}"
        );
    }

    #[test]
    fn preprocess_image_batch_zero_target_size_is_a_typed_error_not_a_panic() {
        // `par_chunks_mut` panics on a zero chunk size — `target_size == 0`
        // makes `pixels_per_image` zero, so this guard at the input edge is
        // load-bearing, not decorative (base returned a `[1, 3, 0, 0]`
        // tensor instead of refusing the request).
        let images = vec![test_image(4, 4)];
        let err = preprocess_image_batch(&images, 0, &TEST_MEAN, &TEST_STD, &Device::Cpu)
            .expect_err("target_size == 0 must be a typed error, not a panic");
        let msg = err.to_string();
        assert!(
            msg.contains("target_size") && msg.contains('0'),
            "error must name the field and the offending value: {msg}"
        );
    }

    #[test]
    fn image_row_wrong_expected_len_is_a_typed_error_not_a_panic() {
        // The release-mode length check `image_row` performs before handing
        // its row back to `copy_from_slice` — exercised directly (not via
        // `debug_assert!`, which would compile out in release) by asking for
        // a length the real pixel row can never have.
        let img = test_image(8, 8);
        let real_len = 3 * 8 * 8;
        let err = image_row(2, &img, 8, &TEST_MEAN, &TEST_STD, real_len + 1)
            .expect_err("a deliberately wrong expected_len must be a typed error");
        let msg = err.to_string();
        assert!(msg.contains("row 2"), "error must be row-indexed: {msg}");
    }

    /// One synthetic RGB image at batch position `idx`: a distinct solid
    /// color per index, with a landscape/portrait/square aspect cycled by
    /// index so the batch exercises every `pad_to_square` branch.
    fn synthetic_image(idx: usize) -> DynamicImage {
        let (w, h) = match idx % 3 {
            0 => (12u32, 8u32),  // landscape
            1 => (8u32, 12u32),  // portrait
            _ => (10u32, 10u32), // already square
        };
        let mut img = DynamicImage::new_rgb8(w, h);
        if let Some(buf) = img.as_mut_rgb8() {
            let r = ((idx * 37) % 256) as u8;
            let g = ((idx * 61 + 17) % 256) as u8;
            let b = ((idx * 89 + 53) % 256) as u8;
            for (x, y, pixel) in buf.enumerate_pixels_mut() {
                // A per-pixel offset (not just a solid fill) so the resize
                // filter sees real spatial content, not a flat color.
                let jitter = ((x + y) % 5) as u8;
                *pixel = image::Rgb([r.wrapping_add(jitter), g, b.wrapping_add(jitter)]);
            }
        }
        img
    }

    fn batch_24() -> Vec<DynamicImage> {
        (0..24).map(synthetic_image).collect()
    }

    /// Run [`preprocess_image_batch`] under a rayon pool of exactly
    /// `pool_size` threads and return its flattened `f32` output.
    fn run_at_pool_size(pool_size: usize, images: &[DynamicImage], target_size: u32) -> Vec<f32> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(pool_size)
            .build()
            .expect("build a fixed-size rayon pool");
        pool.install(|| {
            let tensor =
                preprocess_image_batch(images, target_size, &TEST_MEAN, &TEST_STD, &Device::Cpu)
                    .unwrap();
            tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        })
    }

    /// The PRE-UNIT sequential implementation, transcribed verbatim from
    /// `c1b0b0ba`'s `preprocess_image_batch` (before the `par_chunks_mut`
    /// rewrite) — kept ONLY as a test oracle, INDEPENDENT of the shipped
    /// parallel code path, so the bit-identity assertions below compare the
    /// new code against a second implementation rather than against itself
    /// at a different pool size (a pool-1-vs-pool-k comparison alone cannot
    /// catch a bug the SAME code makes at every pool size). Reuses
    /// `pad_to_square`, unchanged by this unit.
    fn preprocess_image_batch_reference_sequential(
        images: &[DynamicImage],
        target_size: u32,
        mean: &[f32; 3],
        std: &[f32; 3],
        device: &Device,
    ) -> Tensor {
        let pixels_per_image = 3 * (target_size as usize) * (target_size as usize);
        let mut flat = Vec::with_capacity(images.len() * pixels_per_image);
        for img in images {
            let padded = pad_to_square(img);
            let resized = padded.resize_exact(target_size, target_size, FilterType::CatmullRom);
            let rgb = resized.to_rgb8();
            for c in 0..3 {
                let m = mean[c];
                let s = std[c];
                for y in 0..target_size {
                    for x in 0..target_size {
                        let pixel = rgb.get_pixel(x, y)[c];
                        flat.push((pixel as f32 / 255.0 - m) / s);
                    }
                }
            }
        }
        let t = target_size as usize;
        Tensor::from_vec(flat, (images.len(), 3, t, t), device).unwrap()
    }

    /// Oracle 1 (K4): element-wise bit identity between the PRE-UNIT
    /// sequential reference above and the shipped parallel code at pool sizes
    /// {1, 5, 7, 24} (non-dividing counts included), on a 24-image batch that
    /// exercises every `pad_to_square` aspect-ratio branch, at a
    /// non-power-of-two `target_size` (17) so no accidental stride alignment
    /// hides a bug.
    #[test]
    fn image_batch_parallel_matches_sequential_bit_identical_across_pool_sizes() {
        let images = batch_24();
        let target_size = 17;

        let reference = preprocess_image_batch_reference_sequential(
            &images,
            target_size,
            &TEST_MEAN,
            &TEST_STD,
            &Device::Cpu,
        )
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        for &k in &[1usize, 5, 7, 24] {
            let got = run_at_pool_size(k, &images, target_size);
            assert_eq!(
                got.len(),
                reference.len(),
                "output length must match the pre-unit reference at pool size {k}"
            );
            for (idx, (&a, &b)) in got.iter().zip(reference.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "element {idx} differs from the pre-unit sequential reference at pool \
                     size {k}: {a} (bits {:x}) vs {b} (bits {:x})",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }

    /// `row_ids` only changes what number a row is called in an error
    /// message; it must never change the numeric output. Uses row ids that
    /// are neither `0..n` nor monotonic-by-one, so any accidental dependence
    /// on `row_ids` ordering or magnitude (e.g. using it as a chunk offset)
    /// would show up here.
    #[test]
    fn preprocess_image_batch_indexed_row_ids_do_not_affect_numeric_output() {
        let images = batch_24();
        let target_size = 16;

        let sequential =
            preprocess_image_batch(&images, target_size, &TEST_MEAN, &TEST_STD, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

        let row_ids: Vec<usize> = (0..images.len()).map(|i| i * 7 + 1000).collect();
        let indexed = preprocess_image_batch_indexed(
            &row_ids,
            &images,
            target_size,
            &TEST_MEAN,
            &TEST_STD,
            &Device::Cpu,
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

        assert_eq!(sequential.len(), indexed.len());
        for (idx, (&a, &b)) in sequential.iter().zip(indexed.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {idx} changed when row_ids were non-sequential"
            );
        }
    }

    /// Oracle 2 (determinism): two independent parallel runs at the same pool
    /// size reproduce the identical output.
    #[test]
    fn image_batch_parallel_is_deterministic_across_two_runs() {
        let images = batch_24();
        let target_size = 16;

        let first = run_at_pool_size(7, &images, target_size);
        let second = run_at_pool_size(7, &images, target_size);

        assert_eq!(first.len(), second.len());
        for (idx, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {idx} differs between two runs at the same pool size"
            );
        }
    }
}
