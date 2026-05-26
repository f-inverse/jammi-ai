use jammi_numerics::pareto::{dominates, frontier};

#[test]
fn strict_dominance_under_minimisation() {
    // a beats b on every dim except one tie ⇒ dominates
    assert!(dominates(&[1.0, 2.0], &[1.0, 3.0]));
}

#[test]
fn equal_points_do_not_dominate() {
    assert!(!dominates(&[1.0, 3.0], &[1.0, 3.0]));
}

#[test]
fn mutual_incomparability_returns_false() {
    // Each beats the other on one dim ⇒ neither dominates.
    assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0]));
    assert!(!dominates(&[1.0, 2.0], &[2.0, 1.0]));
}

#[test]
fn frontier_excludes_dominated_points() {
    let points: Vec<Vec<f64>> = vec![
        vec![1.0, 2.0], // 0
        vec![2.0, 1.0], // 1
        vec![1.5, 1.5], // 2
        vec![3.0, 3.0], // 3 — dominated by all of 0/1/2
    ];
    let f = frontier(&points);
    assert_eq!(f, vec![0, 1, 2]);
}
