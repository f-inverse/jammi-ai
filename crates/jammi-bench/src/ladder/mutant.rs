//! Mutant columns: the proof that a paired rule can fail.
//!
//! A rule that has never been seen to fire is not known to work. A *mutant*
//! is the engine with one deliberate defect patched in; its legs stand in for
//! the upper rung of an edge and are judged by [`compare`] under that edge's
//! own rules against the same lower legs. The smallest defect the rule
//! detects is what the instrument resolves — the source of an equivalence
//! margin — and a defect built to degrade that is not detected is a finding
//! against the instrument.
//!
//! Two families of mutant are told apart by their label. `eps<x>` scales the
//! optimizer's update by `1 + x`: a signed, one-parameter dose family, read
//! as a ladder. `redproof-<name>` is any other defect, expected to degrade
//! outright.

use std::collections::BTreeSet;

use serde::Serialize;

use super::compare::{compare, CompareOptions};
use super::definition::{Edge, Rung, Workload};
use super::leg::{LegSet, RungLegs};
use super::premise::{LegPremise, TrainDirection};
use super::refusal::{Refusal, ReportedRefusal};
use super::verdict::{serde_plain, Direction, EdgeVerdict, OutcomeVerdict, Status};

const EPS_PREFIX: &str = "eps";
const RED_PROOF_PREFIX: &str = "redproof-";

/// The update scale `1 + eps` is zero at `eps = -1` and flips sign below it:
/// neither is a dose of this family.
const EPS_LOWER_EXCLUSIVE: f64 = -1.0;
const EPS_UPPER: f64 = 1.0;
/// Below this magnitude a dose is not a deliberate one.
const EPS_MIN_MAGNITUDE: f64 = 0.01;

/// The train-probe direction each red-proof patch is built to show, by the
/// patch's sha256. A patch absent here has no declared direction and its
/// column is refused; the label an operator types never decides it.
const RED_PROOF_DIRECTIONS: [(&str, TrainDirection); 2] = [
    (
        "c81d0ed59d45761bbd6487dbb23c5aaae22f30739c0e2e613d96c4901ad9b202",
        TrainDirection::Ascent,
    ),
    (
        "9b3c824dc041899c12c0e2d44d12a3ac8c7b86076ffc778638108925ba51bf4e",
        TrainDirection::Descent,
    ),
];

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DoseLabel {
    Eps(f64),
    RedProof(String),
}

/// One `--mutant LABEL:PATCH_SHA256`.
#[derive(Debug, Clone)]
pub struct MutantSpec {
    pub text: String,
    pub label: DoseLabel,
    pub patch_sha256: String,
}

impl MutantSpec {
    pub fn parse(spec: &str) -> Result<Self, Refusal> {
        let invalid = |reason: String| Refusal::MutantColumnInvalid {
            label: spec.to_owned(),
            reason,
        };
        let (text, sha) = spec
            .split_once(':')
            .ok_or_else(|| invalid("expected LABEL:PATCH_SHA256".into()))?;
        let patch_sha256 = sha.trim().to_ascii_lowercase();
        if patch_sha256.is_empty() {
            return Err(invalid("the patch sha256 is empty".into()));
        }
        let label = if let Some(name) = text.strip_prefix(RED_PROOF_PREFIX) {
            if name.trim().is_empty() {
                return Err(invalid(
                    "a red-proof label names its mutant after the prefix".into(),
                ));
            }
            DoseLabel::RedProof(name.to_owned())
        } else if let Some(number) = text.strip_prefix(EPS_PREFIX) {
            // The label is also the leg file's rung name, byte for byte, so
            // spellings a number parser forgives and a file name does not
            // are refused.
            if number.contains(char::is_whitespace) || number.contains('+') {
                return Err(invalid(
                    "the dose is written without whitespace or `+`".into(),
                ));
            }
            let eps: f64 = number
                .parse()
                .map_err(|_| invalid(format!("{number:?} is not a number")))?;
            let in_domain = eps.is_finite()
                && eps > EPS_LOWER_EXCLUSIVE
                && eps <= EPS_UPPER
                && eps.abs() >= EPS_MIN_MAGNITUDE;
            if !in_domain {
                return Err(invalid(format!(
                    "dose {eps} lies outside ({EPS_LOWER_EXCLUSIVE}, -{EPS_MIN_MAGNITUDE}] ∪ [{EPS_MIN_MAGNITUDE}, {EPS_UPPER}]"
                )));
            }
            DoseLabel::Eps(eps)
        } else {
            return Err(invalid(format!(
                "a label starts with {EPS_PREFIX:?} or {RED_PROOF_PREFIX:?}"
            )));
        };
        Ok(Self {
            text: text.to_owned(),
            label,
            patch_sha256,
        })
    }

    pub fn rung_name(&self) -> String {
        format!("mutant-{}", self.text)
    }
}

/// One dose, one label, one patch: two columns that share any of the three
/// are the same mutant measured twice, and their disagreement would be a
/// determinism question, not a sensitivity interval.
pub fn duplicates(specs: &[MutantSpec]) -> Vec<Refusal> {
    let mut refusals = vec![];
    for (i, a) in specs.iter().enumerate() {
        for b in &specs[i + 1..] {
            let shared = if a.text == b.text {
                Some("the same label".to_owned())
            } else if matches!((&a.label, &b.label), (DoseLabel::Eps(x), DoseLabel::Eps(y)) if x == y)
            {
                Some(format!("the same dose as {:?}", a.text))
            } else if a.patch_sha256 == b.patch_sha256 {
                Some(format!("the same patch as {:?}", a.text))
            } else {
                None
            };
            refusals.extend(shared.map(|reason| Refusal::MutantColumnInvalid {
                label: b.text.clone(),
                reason,
            }));
        }
    }
    refusals
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Detection {
    Undetected,
    Degradation,
    Improvement,
    Invalid,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoseColumn {
    pub label: String,
    pub dose: DoseLabel,
    pub patch_sha256: String,
    pub detected: Detection,
    pub verdict: EdgeVerdict,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoseFinding {
    pub label: String,
    pub finding: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct DoseLadder {
    pub columns: Vec<DoseColumn>,
    /// The adjacent pair of deflating doses, by magnitude, that straddles
    /// detection of a degradation: `(undetected, detected)`.
    pub sensitivity: Option<(String, String)>,
    /// What each inflating dose says about the prediction that inflation
    /// improves held-out loss.
    pub falsification: Vec<DoseFinding>,
    /// Deflating doses that *improved* held-out loss.
    pub anomalies: Vec<DoseFinding>,
    /// Whether some red-proof mutant was detected as a degradation; `None`
    /// when none was scheduled.
    pub red_proof_proven: Option<bool>,
}

/// Judge one mutant column by the edge's own rules.
pub fn column(
    workload: Workload,
    edge: &Edge<'_>,
    legs: &LegSet,
    spec: &MutantSpec,
    options: &CompareOptions,
) -> DoseColumn {
    let mut refusals = vec![];
    let direction = match &spec.label {
        DoseLabel::Eps(_) => TrainDirection::Descent,
        DoseLabel::RedProof(_) => RED_PROOF_DIRECTIONS
            .iter()
            .find(|(sha, _)| *sha == spec.patch_sha256)
            .map_or_else(
                || {
                    refusals.push(Refusal::MutantColumnInvalid {
                        label: spec.text.clone(),
                        reason: format!(
                            "patch {} has no declared train direction",
                            spec.patch_sha256
                        ),
                    });
                    TrainDirection::Descent
                },
                |(_, direction)| *direction,
            ),
    };
    let premises = edge
        .upper()
        .premises
        .iter()
        .map(|p| match p {
            LegPremise::LearningHappened(_) => LegPremise::LearningHappened(direction),
            other => other.clone(),
        })
        .chain([LegPremise::MutantStamp {
            patch_sha256: spec.patch_sha256.clone(),
        }])
        .collect();
    let rung = Rung {
        name: spec.rung_name(),
        layer: "a deliberate defect",
        premises,
        flat_host_memory: None,
    };
    let upper = legs.rung(&rung.name);
    // The column is measured against the lower rung's own legs, never a
    // second run of them, at the units the mutant ran and without the
    // controls the mutant has no counterpart for.
    let units: BTreeSet<_> = upper.measured_units().cloned().collect();
    let lower = RungLegs {
        units: legs
            .rung(&edge.lower().name)
            .units
            .into_iter()
            .filter(|(unit, _)| units.contains(unit))
            .collect(),
    };
    let mut options = options.clone();
    options.same_initial_probe = matches!(spec.label, DoseLabel::RedProof(_));
    options.controls = false;
    let mut verdict = compare(workload, &edge.with_upper(&rung), &lower, &upper, &options);
    verdict
        .refusals
        .extend(refusals.into_iter().map(ReportedRefusal::from));
    let direction = match &verdict.outcome {
        Some(OutcomeVerdict::SeededLoss { direction, .. }) => *direction,
        _ => Direction::None,
    };
    let detected = match (verdict.refusals.is_empty(), direction) {
        (false, _) => Detection::Invalid,
        (true, Direction::Degradation) => Detection::Degradation,
        (true, Direction::Improvement) => Detection::Improvement,
        (true, Direction::None) => Detection::Undetected,
    };
    if detected == Detection::Invalid {
        verdict.status = Status::Invalid;
    }
    DoseColumn {
        label: spec.text.clone(),
        dose: spec.label.clone(),
        patch_sha256: spec.patch_sha256.clone(),
        detected,
        verdict,
    }
}

impl DoseLadder {
    pub fn fold(columns: Vec<DoseColumn>) -> Self {
        let eps = |c: &DoseColumn| match c.dose {
            DoseLabel::Eps(eps) => Some(eps),
            DoseLabel::RedProof(_) => None,
        };
        // Sensitivity is read within the deflating doses only, ordered by
        // magnitude: a detection at an inflating dose answers a different
        // question, and the order doses were run in answers none.
        let mut deflating: Vec<(&DoseColumn, f64)> = columns
            .iter()
            .filter_map(|c| eps(c).filter(|e| *e < 0.0).map(|e| (c, e.abs())))
            .collect();
        deflating.sort_by(|a, b| a.1.total_cmp(&b.1));
        let sensitivity = deflating
            .windows(2)
            .find(|w| {
                w[0].0.detected == Detection::Undetected
                    && w[1].0.detected == Detection::Degradation
            })
            .map(|w| (w[0].0.label.clone(), w[1].0.label.clone()));
        let finding = |c: &DoseColumn, finding| DoseFinding {
            label: c.label.clone(),
            finding,
        };
        let falsification = columns
            .iter()
            .filter(|c| eps(c).is_some_and(|e| e > 0.0))
            .filter_map(|c| match c.detected {
                Detection::Degradation => Some(finding(
                    c,
                    "inflation degraded held-out loss: the improvement prediction is refuted",
                )),
                Detection::Improvement => Some(finding(
                    c,
                    "inflation improved held-out loss: the improvement prediction is confirmed",
                )),
                _ => None,
            })
            .collect();
        let anomalies = columns
            .iter()
            .filter(|c| eps(c).is_some_and(|e| e < 0.0) && c.detected == Detection::Improvement)
            .map(|c| finding(c, "deflation improved held-out loss"))
            .collect();
        let red_proofs: Vec<&DoseColumn> = columns.iter().filter(|c| eps(c).is_none()).collect();
        let red_proof_proven = (!red_proofs.is_empty()).then(|| {
            red_proofs
                .iter()
                .any(|c| c.detected == Detection::Degradation)
        });
        Self {
            sensitivity,
            falsification,
            anomalies,
            red_proof_proven,
            columns,
        }
    }

    /// What the dose ladder adds to the run's status, each with its name.
    pub fn causes(&self) -> Vec<(Status, String)> {
        let invalid: Vec<&str> = self
            .columns
            .iter()
            .filter(|c| c.detected == Detection::Invalid)
            .map(|c| c.label.as_str())
            .collect();
        let anomalous: Vec<&str> = self.anomalies.iter().map(|a| a.label.as_str()).collect();
        [
            (!invalid.is_empty()).then(|| {
                (
                    Status::Invalid,
                    format!("invalid mutant column(s) {invalid:?}"),
                )
            }),
            (!anomalous.is_empty()).then(|| {
                (
                    Status::RedForInvestigation,
                    format!("deflating dose(s) {anomalous:?} improved held-out loss"),
                )
            }),
            (self.red_proof_proven == Some(false)).then(|| {
                (
                    Status::Red,
                    "no red-proof mutant was detected as a degradation".to_owned(),
                )
            }),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    pub fn table(&self) -> Vec<String> {
        let mut lines = vec!["# mutant columns".to_owned()];
        for c in &self.columns {
            let summary = match &c.verdict.outcome {
                Some(OutcomeVerdict::SeededLoss {
                    sign_test: Some(s),
                    mean_d,
                    clean_units,
                    ..
                }) => format!(
                    "n_pos={} n_neg={} mean_d={:?} p_value={} clean_units={clean_units}",
                    s.n_pos, s.n_neg, mean_d, s.p_value
                ),
                _ => "no paired statistic".to_owned(),
            };
            lines.push(format!(
                "  {:<24} detected={:<14} {summary}",
                c.label,
                serde_plain(&c.detected)
            ));
            lines.extend(
                c.verdict
                    .refusals
                    .iter()
                    .map(|r| format!("    REFUSED: {}", r.message)),
            );
        }
        lines.push(format!("  sensitivity: {:?}", self.sensitivity));
        lines.extend(
            self.falsification
                .iter()
                .map(|f| format!("  {}: {}", f.label, f.finding)),
        );
        lines.extend(
            self.anomalies
                .iter()
                .map(|f| format!("  ANOMALY {}: {}", f.label, f.finding)),
        );
        if let Some(proven) = self.red_proof_proven {
            lines.push(format!(
                "  red-proof: {}",
                if proven { "proven" } else { "NOT proven" }
            ));
        }
        lines
    }
}
