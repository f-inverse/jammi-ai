use approx::assert_abs_diff_eq;
use jammi_numerics::classification::ClassificationMetrics;

#[test]
fn perfect_predictions() {
    let predicted: Vec<String> = vec!["a".into(), "b".into(), "a".into()];
    let actual: Vec<String> = vec!["a".into(), "b".into(), "a".into()];
    let r = ClassificationMetrics::compute(&predicted, &actual);
    assert_abs_diff_eq!(r.accuracy, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r.f1, 1.0, epsilon = 1e-12);
}

#[test]
fn all_wrong_predictions() {
    let predicted: Vec<String> = vec!["b".into(), "a".into(), "b".into()];
    let actual: Vec<String> = vec!["a".into(), "b".into(), "a".into()];
    let r = ClassificationMetrics::compute(&predicted, &actual);
    assert_abs_diff_eq!(r.accuracy, 0.0, epsilon = 1e-12);
}

#[test]
fn per_class_metrics_populated() {
    let predicted: Vec<String> = vec!["a".into(), "a".into(), "b".into(), "c".into()];
    let actual: Vec<String> = vec!["a".into(), "b".into(), "b".into(), "c".into()];
    let r = ClassificationMetrics::compute(&predicted, &actual);
    assert!(r.per_class.contains_key("a"));
    assert!(r.per_class.contains_key("b"));
    assert!(r.per_class.contains_key("c"));
    // accuracy = 3/4
    assert_abs_diff_eq!(r.accuracy, 0.75, epsilon = 1e-12);
}
