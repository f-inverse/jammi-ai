//! The CPU-hermetic conformal-coverage tier: how well the engine's split
//! (inductive) conformal calibration holds its marginal coverage guarantee, as a
//! function of calibration-set size.
//!
//! This is the conformal analogue of [`crate::recall`]: where that tier measures
//! a *portable fraction* (set-intersection recall) and gates it against a
//! committed floor `measured − MARGIN`, this tier measures the *portable
//! fraction* of test points the engine's conformal sets / intervals cover and
//! gates it the same way. Coverage is portable — the
//! `⌈(n+1)(1-α)⌉` quantile and the `1[score ≤ q̂]` count are pure arithmetic over
//! the calibration scores, so any box re-derives the same number — so it carries
//! a real CI floor gate, not the same-box rate gate the GPU-bound tiers need.
//!
//! ## What drives the number — the real engine conformal path
//!
//! Every coverage number folds through [`ConformalModel`] (the engine's
//! [`jammi_ai::predict::conformal`] primitive) and is scored by the engine's own
//! [`jammi_numerics::calibration::coverage`] / [`interval_coverage`]. Three
//! families, one per verb the tier covers:
//!
//! * **LAC classification** (`conformalize`): [`ConformalModel::classification`]
//!   with [`ClassScore::Lac`], scored by `coverage` over whether the prediction
//!   set held the true class.
//! * **Absolute-residual regression** (`conformalize_interval`):
//!   [`ConformalModel::regression`] with [`IntervalScore::AbsoluteResidual`],
//!   scored by `interval_coverage`.
//! * **CQR regression** (`conformalize_cqr`): [`ConformalModel::regression`] with
//!   [`IntervalScore::Cqr`], scored by `interval_coverage`.
//!
//! A regression in the engine's quantile (e.g. dropping the `(n+1)` finite-sample
//! correction — the documented under-coverage failure) moves the measured
//! coverage *down*, below the committed floor: the gate has teeth in both
//! directions, exercised by the `cargo test` gate in this module.
//!
//! ## Why a committed *spec*, not committed scores
//!
//! The calibration and test data are drawn deterministically from a seeded LCG
//! (the same generator family [`crate::corpus`] and [`crate::train_scale`] use),
//! so the committed artifact is the *generation spec* (seeds, sizes, α, class
//! count, noise scale) plus the floor `measured − MARGIN` — never hand-written
//! coverage numbers. The gate regenerates the exact same exchangeable data from
//! the spec, re-folds it through the engine, and asserts the coverage clears the
//! floor. Committing the spec rather than the numbers is the conformal mirror of
//! committing the corpus parquet rather than a recall constant: the inputs travel
//! so the fold is re-derivable, the floor is a real measurement minus a margin.

use serde::{Deserialize, Serialize};

use jammi_ai::predict::conformal::{ClassScore, ConformalModel, IntervalScore};
use jammi_numerics::calibration::{coverage, interval_coverage};

use crate::report::{ConformalPoint, ConformalTier, CoverageGate, Measurement};

/// Safety margin subtracted from the committed measured coverage to set the
/// floor the gate asserts: `floor = measured − MARGIN`.
///
/// Mirrors [`crate::fixture::FLOOR_MARGIN`]'s discipline for the recall floor: the
/// gate asserts `coverage ≥ floor`, so the margin is the headroom the
/// finite-sample coverage has against f64-arithmetic or quantile-implementation
/// drift before tripping. Sized to absorb the finite-sample sampling spread of
/// the smallest calibration set (the noisiest point) without going vacuous — it
/// is never the bare measured number, nor an invented round value.
const FLOOR_MARGIN: f64 = 0.05;

/// The committed conformal spec: the generation parameters every coverage number
/// is folded from, plus the per-size committed floors. The on-disk
/// `baselines/conformal.json` the tier and its gate read.
///
/// Nothing here is a hand-written coverage number: the floors are
/// `measured − MARGIN` over the data the spec deterministically regenerates, and
/// the measured values travel alongside for the audit trail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConformalSpec {
    /// The nominal miscoverage level the thresholds target; coverage is gated
    /// against `1 − alpha − ε`, the ε absorbed by the floor margin.
    pub alpha: f64,
    /// Number of classes the synthetic LAC classification draws over.
    pub n_classes: usize,
    /// Held-out test-set size every coverage point is measured over.
    pub test_rows: usize,
    /// The margin subtracted from each committed measured coverage to set its
    /// floor.
    pub margin: f64,
    /// One committed floor record per calibration-set size, ascending.
    pub points: Vec<SpecPoint>,
}

/// One calibration-set size's committed floors — the measured coverage each
/// family achieved when the spec was cut, and the `measured − margin` floor the
/// gate asserts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecPoint {
    /// The calibration-set size this point calibrates over.
    pub cal_rows: usize,
    /// LAC-classification floor record.
    pub classification: FloorPair,
    /// Absolute-residual regression floor record.
    pub absolute_residual: FloorPair,
    /// CQR regression floor record.
    pub cqr: FloorPair,
}

/// A measured-coverage / derived-floor pair, the conformal mirror of the recall
/// fixture's per-k `{measured, floor}` entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloorPair {
    /// The coverage measured on the spec's data when it was cut.
    pub measured: f64,
    /// The floor the gate asserts: `measured − margin`, clamped at 0.
    pub floor: f64,
}

impl ConformalSpec {
    /// The crate-relative path to the committed conformal spec.
    pub fn path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("baselines")
            .join("conformal.json")
    }

    /// Load the committed spec from `baselines/conformal.json`.
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(Self::path())?;
        Ok(serde_json::from_str(&json)?)
    }
}

/// A pseudo-random generator: the Numerical-Recipes LCG the recall and training
/// tiers use, exposed here so the conformal data is drawn from the same
/// reproducible, no-rng-crate source.
///
/// Each call returns a fresh `u64`; the float helpers map it into the ranges the
/// synthetic exchangeable draws need.
struct Lcg {
    state: u64,
}

impl Lcg {
    /// Seed the generator. Distinct seeds give independent streams, so the
    /// calibration and test splits drawn from different seeds are disjoint draws
    /// of the *same* exchangeable distribution — the conformal contract.
    fn new(seed: u64) -> Self {
        Self {
            // Mix the seed into the LCG's initial state so seed 0 is not a fixed
            // point and nearby seeds give well-separated streams.
            state: seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407),
        }
    }

    /// The next raw `u64` in the stream.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// A uniform draw in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        // Top 53 bits → a double in [0, 1), the standard LCG-to-unit map.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard-normal draw via Box–Muller (one of the pair; the other is
    /// discarded — simplicity over throughput, the sizes here are small).
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Seed for the classification calibration stream.
const CLS_CAL_SEED: u64 = 0x0001_C0FFEE;
/// Seed for the classification test stream — distinct from calibration so the
/// two splits are disjoint exchangeable draws.
const CLS_TEST_SEED: u64 = 0x0002_C0FFEE;
/// Seed for the regression calibration stream.
const REG_CAL_SEED: u64 = 0x0003_C0FFEE;
/// Seed for the regression test stream.
const REG_TEST_SEED: u64 = 0x0004_C0FFEE;

/// A drawn classification split: per-row class probabilities and the realised
/// label, exchangeable by construction (the label is sampled *from* the row's
/// own softmax, so `(probs, label)` pairs are calibrated).
struct ClassSplit {
    probs: Vec<Vec<f64>>,
    labels: Vec<usize>,
}

/// Draw a softmax-and-sampled-label classification split of `n` rows over
/// `n_classes`, from the LCG seeded at `seed`.
///
/// Random logits → softmax → the label is drawn from that softmax. This is the
/// exchangeable, calibrated construction the engine's own conformal tests use; it
/// gives a model whose nominal `1 − α` coverage is attainable, so a *passing*
/// gate reflects the conformal arithmetic, not a degenerate always-cover model.
fn draw_classification(seed: u64, n: usize, n_classes: usize) -> ClassSplit {
    let mut rng = Lcg::new(seed);
    let mut probs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for _ in 0..n {
        let logits: Vec<f64> = (0..n_classes).map(|_| rng.normal() * 1.5).collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        let row: Vec<f64> = exp.iter().map(|e| e / sum).collect();

        // Sample the realised label from the row's own softmax.
        let u = rng.unit();
        let mut acc = 0.0;
        let mut label = n_classes - 1;
        for (c, p) in row.iter().enumerate() {
            acc += p;
            if u <= acc {
                label = c;
                break;
            }
        }
        probs.push(row);
        labels.push(label);
    }
    ClassSplit { probs, labels }
}

/// A drawn regression split: point predictions, the symmetric quantile band a
/// well-specified quantile predictor would emit, and the realised target.
///
/// The point/quantile predictor is well-specified (`ŷ` is the true mean, the band
/// is the central `1 − α` Gaussian band), so absolute-residual and CQR conformal
/// both have an attainable nominal coverage — a passing gate reflects the
/// conformal arithmetic over an exchangeable split, not a degenerate predictor.
struct RegressionSplit {
    predictions: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    observed: Vec<f64>,
}

/// The central `1 − α` Gaussian half-width in standard deviations: the `1 − α/2`
/// standard-normal quantile, so `[μ − z·σ, μ + z·σ]` is the well-specified
/// quantile band the regression predictor emits.
fn central_half_width_z(alpha: f64) -> f64 {
    // Acklam's rational approximation to the inverse standard-normal CDF, accurate
    // to ~1e-9 over the central region these α (0.05–0.2) land in — enough for a
    // well-specified band that conformal then corrects to exact coverage.
    let p = 1.0 - alpha / 2.0;
    inverse_standard_normal_cdf(p)
}

/// Acklam's inverse standard-normal CDF approximation. Pure arithmetic, no crate.
fn inverse_standard_normal_cdf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    let p_high = 1.0 - P_LOW;
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// Draw a heteroscedastic regression split of `n` rows from the LCG seeded at
/// `seed`, with the quantile band targeting central coverage `1 − α`.
fn draw_regression(seed: u64, n: usize, alpha: f64) -> RegressionSplit {
    let mut rng = Lcg::new(seed);
    let z = central_half_width_z(alpha);
    let mut predictions = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    for _ in 0..n {
        // A covariate in [0, 10) drives a heteroscedastic noise scale, so CQR's
        // adaptive band has something to adapt to (its width tracks σ(x)).
        let x = rng.unit() * 10.0;
        let mu = 2.0 * x; // the true mean the point predictor nails
        let sd = 0.5 + 0.4 * x; // heteroscedastic noise
        let y = mu + sd * rng.normal();
        predictions.push(mu);
        lower.push(mu - z * sd);
        upper.push(mu + z * sd);
        observed.push(y);
    }
    RegressionSplit {
        predictions,
        lower,
        upper,
        observed,
    }
}

/// Measure LAC-classification marginal coverage at calibration size `cal_rows`:
/// calibrate the engine's [`ConformalModel`] over the calibration split, then
/// score the fraction of the *test* split whose engine-emitted prediction set
/// held the true class via the engine's [`coverage`].
pub fn classification_coverage(
    cal_rows: usize,
    test_rows: usize,
    n_classes: usize,
    alpha: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cal = draw_classification(CLS_CAL_SEED, cal_rows, n_classes);
    let test = draw_classification(CLS_TEST_SEED, test_rows, n_classes);

    let model = ConformalModel::classification(&cal.probs, &cal.labels, ClassScore::Lac, alpha)?;
    let hits: Vec<bool> = test
        .probs
        .iter()
        .zip(test.labels.iter())
        .map(|(row, &y)| Ok(model.predict_set(row, None)?.contains(&y)))
        .collect::<Result<Vec<bool>, Box<dyn std::error::Error>>>()?;
    Ok(coverage(&hits)?)
}

/// Measure regression marginal coverage at calibration size `cal_rows` for the
/// given [`IntervalScore`]: calibrate the engine's [`ConformalModel`] over the
/// calibration split, emit each test row's interval, and score the fraction
/// covered via the engine's [`interval_coverage`].
pub fn regression_coverage(
    cal_rows: usize,
    test_rows: usize,
    alpha: f64,
    score: IntervalScore,
) -> Result<f64, Box<dyn std::error::Error>> {
    let cal = draw_regression(REG_CAL_SEED, cal_rows, alpha);
    let test = draw_regression(REG_TEST_SEED, test_rows, alpha);

    let model = ConformalModel::regression(
        &cal.predictions,
        &cal.lower,
        &cal.upper,
        &cal.observed,
        score,
        alpha,
    )?;

    // For absolute-residual the interval is built from the point prediction; for
    // CQR it is built from the lower/upper quantile estimates. `predict_interval`
    // ignores the irrelevant argument for each score, so pass all three.
    let mut lower = Vec::with_capacity(test_rows);
    let mut upper = Vec::with_capacity(test_rows);
    for i in 0..test.observed.len() {
        let (lo, hi) =
            model.predict_interval(test.predictions[i], test.lower[i], test.upper[i], None)?;
        lower.push(lo);
        upper.push(hi);
    }
    Ok(interval_coverage(&lower, &upper, &test.observed)?)
}

/// Build a [`CoverageGate`] from a measured coverage and a committed floor pair:
/// the measured value, the committed floor, and the `measured ≥ floor` verdict.
fn coverage_gate(measured: f64, floor: &FloorPair) -> CoverageGate {
    CoverageGate {
        measured: Measurement::measured(measured, "fraction"),
        floor: floor.floor,
        passed: measured >= floor.floor,
    }
}

/// Run the conformal-coverage tier against the committed spec: for each
/// calibration size in the spec, re-fold all three families through the engine
/// and gate each coverage against its committed floor.
///
/// This is the path the `conformal-scale` subcommand drives and the `cargo test`
/// gate asserts: every coverage is the engine's own conformal calibration scored
/// by the engine's own coverage kernel, gated `measured ≥ floor`.
pub fn run(spec: &ConformalSpec) -> Result<ConformalTier, Box<dyn std::error::Error>> {
    let mut points = Vec::with_capacity(spec.points.len());
    for sp in &spec.points {
        let cls = classification_coverage(sp.cal_rows, spec.test_rows, spec.n_classes, spec.alpha)?;
        let abs_res = regression_coverage(
            sp.cal_rows,
            spec.test_rows,
            spec.alpha,
            IntervalScore::AbsoluteResidual,
        )?;
        let cqr = regression_coverage(sp.cal_rows, spec.test_rows, spec.alpha, IntervalScore::Cqr)?;
        points.push(ConformalPoint {
            cal_rows: sp.cal_rows,
            test_rows: spec.test_rows,
            classification_coverage: coverage_gate(cls, &sp.classification),
            absolute_residual_coverage: coverage_gate(abs_res, &sp.absolute_residual),
            cqr_coverage: coverage_gate(cqr, &sp.cqr),
        });
    }
    Ok(ConformalTier {
        alpha: spec.alpha,
        points,
    })
}

/// Whether every coverage gate in a tier passed — the verdict the subcommand maps
/// to its exit code and the `cargo test` gate asserts.
pub fn all_gates_passed(tier: &ConformalTier) -> bool {
    tier.points.iter().all(|p| {
        p.classification_coverage.passed
            && p.absolute_residual_coverage.passed
            && p.cqr_coverage.passed
    })
}

/// Re-derive the committed spec's floors from a fresh measurement: for each size,
/// measure all three coverages and set `floor = measured − margin`. The off-box
/// one-shot that writes `baselines/conformal.json`; CI only ever loads and
/// re-folds it (the gate in [`run`]).
///
/// Returns the spec with measured values and derived floors filled in; the caller
/// serializes it to the committed path.
pub fn rebuild_spec(
    alpha: f64,
    n_classes: usize,
    test_rows: usize,
    cal_sizes: &[usize],
) -> Result<ConformalSpec, Box<dyn std::error::Error>> {
    let mut points = Vec::with_capacity(cal_sizes.len());
    for &cal_rows in cal_sizes {
        let cls = classification_coverage(cal_rows, test_rows, n_classes, alpha)?;
        let abs_res =
            regression_coverage(cal_rows, test_rows, alpha, IntervalScore::AbsoluteResidual)?;
        let cqr = regression_coverage(cal_rows, test_rows, alpha, IntervalScore::Cqr)?;
        let derive = |m: f64| FloorPair {
            measured: m,
            floor: (m - FLOOR_MARGIN).max(0.0),
        };
        points.push(SpecPoint {
            cal_rows,
            classification: derive(cls),
            absolute_residual: derive(abs_res),
            cqr: derive(cqr),
        });
    }
    Ok(ConformalSpec {
        alpha,
        n_classes,
        test_rows,
        margin: FLOOR_MARGIN,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    /// Load a spec from an arbitrary directory's `conformal.json` (test seam:
    /// reads a written copy without touching the committed file).
    fn load_spec_from(dir: &Path) -> Result<ConformalSpec, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(dir.join("conformal.json"))?;
        Ok(serde_json::from_str(&json)?)
    }

    /// The committed spec is well-formed and its floors are real (`measured −
    /// margin`, never an invented number): every floor equals its paired measured
    /// minus the spec margin, clamped at 0.
    #[test]
    fn committed_spec_floors_are_measured_minus_margin() {
        let spec = ConformalSpec::load().expect("baselines/conformal.json must be present");
        assert!(spec.alpha > 0.0 && spec.alpha < 1.0);
        assert!(!spec.points.is_empty(), "spec must carry at least one size");
        for sp in &spec.points {
            for pair in [&sp.classification, &sp.absolute_residual, &sp.cqr] {
                let expected = (pair.measured - spec.margin).max(0.0);
                assert!(
                    (pair.floor - expected).abs() < 1e-12,
                    "floor {} is not measured {} − margin {}",
                    pair.floor,
                    pair.measured,
                    spec.margin
                );
            }
        }
    }

    /// The teeth, FLOOR-CLEARS direction: re-folding the committed spec through
    /// the engine's real conformal path clears every committed floor. A
    /// regression in any conformal code path moves a measured coverage below its
    /// floor and trips this.
    #[test]
    fn measured_coverage_clears_committed_floor() {
        let spec = ConformalSpec::load().expect("baselines/conformal.json must be present");
        let tier = run(&spec).expect("conformal tier must run over the committed spec");
        assert!(
            all_gates_passed(&tier),
            "a committed coverage fell below its floor: {tier:?}"
        );
    }

    /// The teeth, GATE-FAILS direction (RC1: an assertion must be able to fail).
    ///
    /// The floor's job is to catch a *material* coverage regression — one larger
    /// than the [`FLOOR_MARGIN`] headroom. The class of conformal bug that
    /// produces one is a miscalibrated level: the engine's [`ConformalModel`]
    /// calibrated against a *wrong, inflated* α (e.g. a doubled-α bug, or reading
    /// the wrong column) targets `1 − α'` coverage well below the intended
    /// `1 − α`, dropping the achieved coverage past the margin and below the
    /// committed floor. This drives the SAME real engine path the gate does — only
    /// the level is regressed — so it proves the floor catches a real engine
    /// miscalibration, not a contrived non-engine number.
    ///
    /// (The `(n+1)` finite-sample correction's own ~`1/n` gap is, by contrast,
    /// far *smaller* than the margin at these calibration sizes — the margin is
    /// deliberately sized to ride over that finite-sample noise while still
    /// biting a structural miscalibration. So the regression the floor is built to
    /// catch is the inflated-α one, not the `(n+1)` gap.)
    #[test]
    fn miscalibrated_alpha_regression_trips_the_gate() {
        let spec = ConformalSpec::load().expect("baselines/conformal.json must be present");
        let sp = &spec.points[0];

        // The regressed level: a doubled α targets 1 − 2α coverage. At α = 0.1
        // that is ~0.80 — a ~0.10 drop, comfortably past the 0.05 margin, so the
        // floor must catch it. Calibrate the REAL engine model at the wrong level.
        let regressed_alpha = (2.0 * spec.alpha).min(0.9);
        let regressed =
            classification_coverage(sp.cal_rows, spec.test_rows, spec.n_classes, regressed_alpha)
                .unwrap();
        assert!(
            regressed < sp.classification.floor,
            "a miscalibrated (inflated-α) engine model must UNDER-cover below the committed \
             floor (else the gate is vacuous): regressed coverage {regressed} vs floor {}",
            sp.classification.floor
        );

        // And the engine at the CORRECT level clears the same floor on the same
        // data — the contrast that gives the gate its teeth.
        let correct =
            classification_coverage(sp.cal_rows, spec.test_rows, spec.n_classes, spec.alpha)
                .unwrap();
        assert!(
            correct >= sp.classification.floor,
            "the engine at the correct α must clear the floor the regressed one fails: \
             correct {correct} vs floor {}",
            sp.classification.floor
        );
    }

    /// A tampered spec — floors raised above the attainable coverage — fails the
    /// gate, proving [`run`] + [`all_gates_passed`] react to the committed floor,
    /// not just to the data. This is the gate-fails direction at the harness
    /// level (the previous test is at the quantile level).
    #[test]
    fn floor_above_attainable_coverage_fails_the_gate() {
        let mut spec = ConformalSpec::load().expect("baselines/conformal.json must be present");
        // Raise every floor to an unattainable 0.999 — no LAC set / interval at
        // these α covers that often, so the gate must fail.
        for sp in &mut spec.points {
            sp.classification.floor = 0.999;
            sp.absolute_residual.floor = 0.999;
            sp.cqr.floor = 0.999;
        }
        let tier = run(&spec).expect("tier still runs");
        assert!(
            !all_gates_passed(&tier),
            "an unattainable floor must trip the gate"
        );
    }

    /// `rebuild_spec` is the inverse of the gate: the floors it derives are
    /// exactly `measured − margin`, and re-running the gate over its output
    /// passes (the floor it just wrote is, by construction, below the measured
    /// coverage). This guards the off-box rebuilder against drifting from the
    /// committed-floor idiom.
    #[test]
    fn rebuild_spec_round_trips_through_the_gate() {
        let spec = ConformalSpec::load().expect("baselines/conformal.json must be present");
        let sizes: Vec<usize> = spec.points.iter().map(|p| p.cal_rows).collect();
        let rebuilt =
            rebuild_spec(spec.alpha, spec.n_classes, spec.test_rows, &sizes).expect("rebuild runs");
        for sp in &rebuilt.points {
            for pair in [&sp.classification, &sp.absolute_residual, &sp.cqr] {
                assert_eq!(pair.floor, (pair.measured - rebuilt.margin).max(0.0));
            }
        }
        let tier = run(&rebuilt).expect("tier runs over the rebuilt spec");
        assert!(
            all_gates_passed(&tier),
            "a freshly rebuilt spec must pass its own gate"
        );
    }

    /// The conformal data is deterministic across runs (the audit property): the
    /// same seed and size give byte-identical coverage, so the committed floor is
    /// a stable reference, not a moving target.
    #[test]
    fn coverage_is_deterministic_across_runs() {
        let a = classification_coverage(512, 2000, 5, 0.1).unwrap();
        let b = classification_coverage(512, 2000, 5, 0.1).unwrap();
        assert_eq!(a, b, "same seed/size must give identical coverage");
    }

    /// The `load_spec_from` seam reads a spec from an arbitrary directory — used
    /// by the rebuild round-trip to read a written copy without touching the
    /// committed file.
    #[test]
    fn load_spec_from_reads_a_written_copy() {
        let spec = ConformalSpec::load().expect("baselines/conformal.json must be present");
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("conformal.json"),
            serde_json::to_string_pretty(&spec).unwrap(),
        )
        .unwrap();
        let loaded = load_spec_from(dir.path()).unwrap();
        assert_eq!(loaded.points.len(), spec.points.len());
        assert_eq!(loaded.alpha, spec.alpha);
    }
}
