//! Image preprocessing for CLIP-family vision models.
//!
//! Provides pad-to-square, resize, normalize, and rotation transforms
//! matching the OpenCLIP preprocessing pipeline.

use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use image::DynamicImage;
use jammi_engine::error::{JammiError, Result};

/// CLIP image normalization mean (per-channel, RGB order).
#[allow(clippy::excessive_precision)]
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];

/// CLIP image normalization std (per-channel, RGB order).
#[allow(clippy::excessive_precision)]
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Preprocess a batch of images into a CLIP-ready tensor.
///
/// Each image is: padded to square (white) → resized to `target_size` → normalized.
/// Returns tensor of shape `(batch, 3, target_size, target_size)`.
pub fn preprocess_clip_batch(
    images: &[DynamicImage],
    target_size: u32,
    device: &Device,
) -> Result<Tensor> {
    if images.is_empty() {
        return Err(JammiError::Inference(
            "Cannot preprocess empty image batch".into(),
        ));
    }

    let pixels_per_image = 3 * (target_size as usize) * (target_size as usize);
    let mut flat = Vec::with_capacity(images.len() * pixels_per_image);

    for img in images {
        let padded = pad_to_square(img);
        let resized = padded.resize_exact(target_size, target_size, FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        // Convert to CHW float32 with CLIP normalization
        for c in 0..3 {
            let mean = CLIP_MEAN[c];
            let std = CLIP_STD[c];
            for y in 0..target_size {
                for x in 0..target_size {
                    let pixel = rgb.get_pixel(x, y)[c];
                    flat.push((pixel as f32 / 255.0 - mean) / std);
                }
            }
        }
    }

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

/// Rotate an image by a multiple of 90 degrees.
///
/// Only supports 0, 90, 180, 270. Other angles are ignored (returns original).
pub fn rotate_image(img: &DynamicImage, angle: u16) -> DynamicImage {
    match angle % 360 {
        0 => img.clone(),
        90 => img.rotate90(),
        180 => img.rotate180(),
        270 => img.rotate270(),
        _ => img.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let tensor = preprocess_clip_batch(&images, 224, &Device::Cpu).unwrap();
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

        let tensor = preprocess_clip_batch(&[img], 4, &Device::Cpu).unwrap();
        let vals = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // White pixel channel 0: (1.0 - 0.48145466) / 0.26862954 ≈ 1.93
        let expected_ch0 = (1.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!((vals[0] - expected_ch0).abs() < 0.01);
    }

    #[test]
    fn test_rotate_image() {
        let img = test_image(100, 200);

        let r0 = rotate_image(&img, 0);
        assert_eq!((r0.width(), r0.height()), (100, 200));

        let r90 = rotate_image(&img, 90);
        assert_eq!((r90.width(), r90.height()), (200, 100));

        let r180 = rotate_image(&img, 180);
        assert_eq!((r180.width(), r180.height()), (100, 200));

        let r270 = rotate_image(&img, 270);
        assert_eq!((r270.width(), r270.height()), (200, 100));
    }

    #[test]
    fn test_preprocess_empty_batch_errors() {
        let result = preprocess_clip_batch(&[], 224, &Device::Cpu);
        assert!(result.is_err());
    }
}
