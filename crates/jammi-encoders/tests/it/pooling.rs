//! Tests for the shared pooling + mask helpers. CPU-only, hermetic.

use candle_core::{Device, Tensor};
use jammi_encoders::{pool_and_normalize, Pooling};

fn cpu() -> Device {
    Device::Cpu
}

fn hidden_fixture(device: &Device) -> Tensor {
    // [batch=1, seq=3, hidden=2]
    Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 2.0], (1, 3, 2), device).unwrap()
}

#[test]
fn extended_attention_mask_values() {
    // The helper is crate-private, so we exercise it indirectly by checking
    // that mean-pool with a partially-masked input ignores the padding token.
    let device = cpu();
    let hidden = hidden_fixture(&device);
    let mask = Tensor::new(&[[1u32, 1, 0]], &device).unwrap();
    let pooled = pool_and_normalize(&hidden, &mask, Pooling::Mean).unwrap();
    // Mean of (1,0) and (0,1) is (0.5, 0.5); L2-normalised is (1/√2, 1/√2).
    let row: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();
    assert!((row[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!((row[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn pool_and_normalize_mean_correctness() {
    let device = cpu();
    let hidden = hidden_fixture(&device);
    let mask = Tensor::new(&[[1u32, 1, 1]], &device).unwrap();
    let pooled = pool_and_normalize(&hidden, &mask, Pooling::Mean).unwrap();
    // Mean over all three rows: (1+0+2)/3 = 1.0, (0+1+2)/3 = 1.0. L2 → (√½, √½).
    let row: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();
    let norm = (row[0].powi(2) + row[1].powi(2)).sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
    assert!((row[0] - row[1]).abs() < 1e-6, "expected equal coords");
}

#[test]
fn pool_and_normalize_cls_correctness() {
    let device = cpu();
    let hidden = hidden_fixture(&device);
    let mask = Tensor::new(&[[1u32, 1, 1]], &device).unwrap();
    let pooled = pool_and_normalize(&hidden, &mask, Pooling::Cls).unwrap();
    // First token is (1.0, 0.0); already unit-norm.
    let row: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();
    assert!((row[0] - 1.0).abs() < 1e-6);
    assert!(row[1].abs() < 1e-6);
}

#[test]
fn pool_and_normalize_max_correctness() {
    // [1, 3, 2] with mask [1,1,0]. Per-feature max over real tokens is
    // max((1,0), (0,1)) = (1, 1). Padding row (2,2) must NOT be selected.
    let hidden =
        Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 2.0], (1, 3, 2), &cpu()).unwrap();
    let mask = Tensor::new(&[[1u32, 1, 0]], &cpu()).unwrap();
    let pooled = pool_and_normalize(&hidden, &mask, Pooling::Max).unwrap();
    let row: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();
    let norm = (row[0].powi(2) + row[1].powi(2)).sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    assert!((row[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    assert!((row[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn pool_and_normalize_weighted_mean_correctness() {
    let device = cpu();
    // hidden = [(1,0), (0,1), (2,2)], mask all-1, weights [1,2,3].
    // numerator = 1*(1,0) + 2*(0,1) + 3*(2,2) = (1+0+6, 0+2+6) = (7, 8)
    // denominator = 1+2+3 = 6
    // pooled = (7/6, 8/6); L2-norm → (7, 8) / √(49+64) = (7, 8) / √113.
    let hidden = hidden_fixture(&device);
    let mask = Tensor::new(&[[1u32, 1, 1]], &device).unwrap();
    let pooled = pool_and_normalize(&hidden, &mask, Pooling::WeightedMean).unwrap();
    let row: Vec<f32> = pooled.squeeze(0).unwrap().to_vec1().unwrap();
    let denom = (49f32 + 64f32).sqrt();
    assert!((row[0] - 7.0 / denom).abs() < 1e-5, "x = {}", row[0]);
    assert!((row[1] - 8.0 / denom).abs() < 1e-5, "y = {}", row[1]);
}
