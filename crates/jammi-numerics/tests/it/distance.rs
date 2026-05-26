use approx::assert_abs_diff_eq;
use jammi_numerics::distance::{cosine_distance, cosine_similarity, vector_norm};

#[test]
fn cosine_distance_of_self_is_zero() {
    let v = vec![1.0_f32, 2.0, 3.0];
    assert_abs_diff_eq!(cosine_distance(&v, &v), 0.0_f32, epsilon = 1e-6);
}

#[test]
fn cosine_distance_anti_parallel_is_two() {
    let v = vec![1.0_f32, 2.0, 3.0];
    let neg: Vec<f32> = v.iter().map(|x| -x).collect();
    assert_abs_diff_eq!(cosine_distance(&v, &neg), 2.0_f32, epsilon = 1e-6);
}

#[test]
fn cosine_distance_with_zero_vector_returns_one() {
    let v = vec![1.0_f32, 0.0, 0.0];
    let zero = vec![0.0_f32, 0.0, 0.0];
    assert_eq!(cosine_distance(&v, &zero), 1.0);
}

#[test]
fn similarity_and_distance_sum_to_one_for_non_degenerate() {
    let a = vec![1.0_f32, 0.5, -0.3];
    let b = vec![0.8_f32, 1.2, 0.1];
    let sum = cosine_similarity(&a, &b) + cosine_distance(&a, &b);
    assert_abs_diff_eq!(sum, 1.0_f32, epsilon = 1e-6);
}

#[test]
fn vector_norm_matches_known_values() {
    assert_abs_diff_eq!(vector_norm(&[3.0, 4.0]), 5.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vector_norm(&[]), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(vector_norm(&[1.0, 0.0, 0.0]), 1.0, epsilon = 1e-12);
}
