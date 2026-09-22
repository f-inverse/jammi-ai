//! `compare(edge)`: the one operator, applied to every edge of every
//! ladder.
//!
//! It first decides whether the legs describe one comparison at all —
//! every leg present, every premise held, every identity field agreed — and
//! only then reads numbers off them. A unit whose legs fail is still measured
//! and reported; it is never counted.

use std::collections::{BTreeMap, BTreeSet};

use super::definition::{
    ControlRule, CrossStackOutcome, Edge, EdgeKind, RuleForce, ShapeRules, Workload,
    GRADIENTS_TAKE_FREE_FIELDS,
};
use super::leg::{Leg, RungLegs, Take, Unit};
use super::outcome::{self, Claims, CrossStackOptions, Pair};
use super::premise;
use super::refusal::Refusal;
use super::space;
use super::speed::{self, SpeedBound, TimeToQualityRule};
use super::verdict::{EdgeVerdict, OutcomeVerdict};

/// What a run measures. An axis left out is not judged; an axis asked for
/// whose hard rule finds nothing to measure is refused.
///
/// `shape` is not a fourth axis but the sweep under two of them: with it,
/// speed is also fitted as a line in work and space as a slope in work, which
/// needs legs at three sizes or more. A session at one size leaves it out.
#[derive(Debug, Clone, Copy)]
pub struct Axes {
    pub outcome: bool,
    pub speed: bool,
    pub space: bool,
    pub shape: bool,
}

#[derive(Debug, Clone)]
pub struct CompareOptions {
    pub axes: Axes,
    pub cross_stack: CrossStackOptions,
    /// Which of a seeded edge's claims are judged.
    pub claims: Claims,
    /// Require both rungs of a unit to have started from the same model: the
    /// train probe taken before any step must be equal, bit for bit.
    pub same_initial_probe: bool,
    /// A revision edge's A/A null: the legs of a second build of the lower
    /// side, measured in the same session.
    pub rebuilt: Option<RungLegs>,
}

impl CompareOptions {
    pub fn new(axes: Axes, cross_stack: CrossStackOptions) -> Self {
        Self {
            axes,
            cross_stack,
            claims: Claims::ALL,
            same_initial_probe: false,
            rebuilt: None,
        }
    }
}

/// The takes an edge declares beyond its repeats: the control of a seeded
/// edge, the gradient legs of a gradient edge.
#[derive(Debug, Clone, Copy, Default)]
struct DeclaredTakes<'a> {
    control: Option<&'a ControlRule>,
    gradients: Option<&'a str>,
}

impl<'a> DeclaredTakes<'a> {
    fn of(edge: &Edge<'a>) -> Self {
        match edge.kind() {
            EdgeKind::CrossStack(rules) => match &rules.outcome {
                CrossStackOutcome::SeededLoss { control, .. } => Self {
                    control: control.as_ref(),
                    gradients: None,
                },
                CrossStackOutcome::GradientAgreement { take, .. } => Self {
                    control: None,
                    gradients: Some(take),
                },
                _ => Self::default(),
            },
            EdgeKind::Exact(_) | EdgeKind::Revision(_) => Self::default(),
        }
    }

    fn declares(&self, take: &Take) -> bool {
        let Take::Control(tag) = take else {
            return false;
        };
        self.control.is_some_and(|rule| rule.take == tag)
            || self.gradients.is_some_and(|take| take == tag)
    }

    fn is_gradients(&self, take: &Take) -> bool {
        matches!(take, Take::Control(tag) if self.gradients.is_some_and(|take| take == tag))
    }
}

/// Group `legs` by their value of `field`; more than one group is a
/// disagreement.
fn disagreement(field: &str, legs: &[&Leg]) -> Option<Refusal> {
    let mut groups: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for leg in legs {
        if let Some(Some(value)) = leg.identity.get(field) {
            groups.entry(value).or_default().push(leg.name.to_string());
        }
    }
    (groups.len() > 1).then(|| Refusal::IdentityDisagreement {
        field: field.to_owned(),
        values: groups
            .into_iter()
            .map(|(v, legs)| (v.to_owned(), legs))
            .collect(),
    })
}

/// Whether the legs share a premise, and the units that do not.
///
/// A field absent from a leg is refused outright. A swept field names the
/// unit, so it must agree within each unit and is free across them. Every
/// other field must agree across every leg entering the comparison — not
/// merely between the two legs of one unit, or two halves of a sweep run
/// under different premises would each look consistent and be averaged as
/// one experiment. Two exceptions are the edge's own definition: the field a
/// control overrides must agree within the measured legs and within the
/// control legs, and differs between them; the fields a gradient take has no
/// use for are free on the gradient legs and must agree on every other.
fn identity(
    workload: Workload,
    declared: DeclaredTakes<'_>,
    legs: &[&Leg],
) -> (Vec<Refusal>, BTreeSet<Unit>) {
    let mut refusals = vec![];
    let mut unclean = BTreeSet::new();
    for (field, _) in workload.identity_fields() {
        let missing: Vec<&&Leg> = legs
            .iter()
            .filter(|leg| leg.identity.get(field) == Some(&None))
            .collect();
        if !missing.is_empty() {
            unclean.extend(missing.iter().map(|leg| leg.name.unit.clone()));
            refusals.push(Refusal::IdentityMissing {
                field: (*field).to_owned(),
                legs: missing.iter().map(|leg| leg.name.to_string()).collect(),
            });
        }
        if workload.swept_fields().contains(field) {
            let mut by_unit: BTreeMap<&Unit, Vec<&Leg>> = BTreeMap::new();
            for leg in legs {
                by_unit.entry(&leg.name.unit).or_default().push(leg);
            }
            for (unit, unit_legs) in by_unit {
                if let Some(refusal) = disagreement(field, &unit_legs) {
                    unclean.insert(unit.clone());
                    refusals.push(refusal);
                }
            }
        } else if declared.control.is_some_and(|rule| rule.field == *field) {
            let (controls, measured): (Vec<&Leg>, Vec<&Leg>) = legs
                .iter()
                .copied()
                .partition(|leg| matches!(leg.name.take, Take::Control(_)));
            refusals.extend(disagreement(field, &measured));
            refusals.extend(disagreement(field, &controls));
        } else if declared.gradients.is_some() && GRADIENTS_TAKE_FREE_FIELDS.contains(field) {
            let measured: Vec<&Leg> = legs
                .iter()
                .copied()
                .filter(|leg| !declared.is_gradients(&leg.name.take))
                .collect();
            refusals.extend(disagreement(field, &measured));
        } else {
            refusals.extend(disagreement(field, legs));
        }
    }
    (refusals, unclean)
}

/// An edge's sweep rules, when the run fits the sweep.
fn swept(rules: &Option<ShapeRules>, shape: bool) -> Option<&ShapeRules> {
    rules.as_ref().filter(|_| shape)
}

pub fn compare(
    workload: Workload,
    edge: &Edge<'_>,
    lower: &RungLegs,
    upper: &RungLegs,
    options: &CompareOptions,
) -> EdgeVerdict {
    let name = edge.name();
    let mut refusals = vec![];
    let mut unclean: BTreeSet<Unit> = BTreeSet::new();

    // Presence: a unit is comparable when both rungs carry its first repeat.
    let all_units: BTreeSet<&Unit> = lower
        .measured_units()
        .chain(upper.measured_units())
        .collect();
    for (rung, legs) in [(edge.lower_name(), lower), (edge.upper_name(), upper)] {
        let absent: Vec<&Unit> = all_units
            .iter()
            .copied()
            .filter(|u| legs.primary(u).is_none())
            .collect();
        refusals.extend(absent.iter().map(|unit| Refusal::MissingLeg {
            rung: rung.clone(),
            unit: unit.as_str().to_owned(),
            take: "r1".to_owned(),
        }));
        if all_units.is_empty() {
            refusals.push(Refusal::MissingLeg {
                rung,
                unit: "*".to_owned(),
                take: "r1".to_owned(),
            });
        }
    }
    let units: Vec<Unit> = all_units
        .into_iter()
        .filter(|u| lower.primary(u).is_some() && upper.primary(u).is_some())
        .cloned()
        .collect();

    // Premises: what each leg must show about itself.
    for (rung, legs) in [(edge.lower(), lower), (edge.upper(), upper)] {
        for leg in units.iter().flat_map(|unit| legs.repeats(unit)) {
            let violated = premise::violations(leg, &rung.premises);
            if !violated.is_empty() {
                unclean.insert(leg.name.unit.clone());
            }
            refusals.extend(violated);
        }
    }

    if options.same_initial_probe {
        for unit in &units {
            let initial = |legs: &RungLegs| {
                legs.primary(unit)
                    .and_then(|leg| leg.facts.train_probe_series.as_deref()?.first().copied())
            };
            let (below, above) = (initial(lower), initial(upper));
            if below.is_none() || below != above {
                unclean.insert(unit.clone());
                refusals.push(Refusal::PremiseViolated {
                    leg: format!("{}__{}__r1", edge.upper_name(), unit.as_str()),
                    premise: "same_initial_probe",
                    reason: format!(
                        "the untrained probe reads {above:?}; the lower rung's reads {below:?}"
                    ),
                });
            }
        }
    }

    // Identity: what the legs must agree on.
    let mut declared = DeclaredTakes::of(edge);
    if !options.claims.controls {
        declared.control = None;
    }
    let entering: Vec<&Leg> = [lower, upper]
        .into_iter()
        .flat_map(|legs| {
            units.iter().flat_map(|unit| legs.repeats(unit)).chain(
                legs.controls()
                    .filter(|leg| declared.declares(&leg.name.take)),
            )
        })
        .collect();
    let (identity_refusals, identity_unclean) = identity(workload, declared, &entering);
    refusals.extend(identity_refusals);
    unclean.extend(identity_unclean);

    let pair = Pair {
        edge: &name,
        lower,
        upper,
        units: &units,
        unclean: &unclean,
    };
    let mut judgements = vec![];

    let outcome = options.axes.outcome.then(|| match edge.kind() {
        EdgeKind::Exact(_) => outcome::digests(&pair, RuleForce::Hard),
        // Two revisions may legitimately change an artifact; the digests
        // are compared and reported, never refused.
        EdgeKind::Revision(_) => outcome::digests(&pair, RuleForce::Evidence),
        EdgeKind::CrossStack(rules) => outcome::cross_stack(
            &pair,
            &rules.outcome,
            &edge.lower().premises,
            &edge.upper().premises,
            &options.cross_stack,
            options.claims,
        ),
    });
    // The margin the outcome axis derived sets the time-to-quality target.
    let slack = outcome.as_ref().and_then(|axis| match &axis.verdict {
        Some(OutcomeVerdict::SeededLoss {
            assay: Some(assay), ..
        }) if assay.sensitive => Some(assay.delta),
        _ => None,
    });
    let speed = options.axes.speed.then(|| match edge.kind() {
        EdgeKind::Exact(rules) => speed::speed(
            &pair,
            SpeedBound::OverheadBudget(&rules.overhead_budget),
            swept(&rules.shape, options.axes.shape),
            None,
        ),
        EdgeKind::Revision(rules) => {
            speed::revision(&pair, rules.within_noise_band, options.rebuilt.as_ref())
        }
        EdgeKind::CrossStack(rules) => speed::speed(
            &pair,
            SpeedBound::NonInferiority(&rules.speed_bar),
            swept(&rules.shape, options.axes.shape),
            rules
                .time_to_quality_bar
                .as_ref()
                .map(|bar| TimeToQualityRule { bar, slack }),
        ),
    });
    let space = options.axes.space.then(|| {
        let rules = match edge.kind() {
            EdgeKind::Exact(rules) => &rules.space,
            EdgeKind::CrossStack(rules) => &rules.space,
            EdgeKind::Revision(rules) => &rules.space,
        };
        space::space(
            &pair,
            rules,
            edge.upper()
                .flat_host_memory
                .as_ref()
                .filter(|_| options.axes.shape),
        )
    });

    let outcome = outcome.and_then(|axis| {
        judgements.extend(axis.judgements);
        refusals.extend(axis.refusals);
        axis.verdict
    });
    let speed = speed.and_then(|axis| {
        judgements.extend(axis.judgements);
        refusals.extend(axis.refusals);
        axis.verdict
    });
    let space = space.and_then(|axis| {
        judgements.extend(axis.judgements);
        refusals.extend(axis.refusals);
        axis.verdict
    });

    EdgeVerdict::conclude(
        name.clone(),
        edge.difference(),
        units.iter().map(|u| u.as_str().to_owned()).collect(),
        (outcome, speed, space),
        judgements,
        refusals,
    )
}
