//! `compare(edge)`: the one operator, applied to every edge of every
//! ladder.
//!
//! It first decides whether the legs describe one comparison at all —
//! every leg present, every premise held, every identity field agreed — and
//! only then reads numbers off them. A unit whose legs fail is still measured
//! and reported; it is never counted.

use std::collections::{BTreeMap, BTreeSet};

use super::definition::{ControlRule, CrossStackOutcome, Edge, EdgeKind, Workload};
use super::leg::{Leg, RungLegs, Take, Unit};
use super::outcome::{self, CrossStackOptions, Pair};
use super::premise;
use super::refusal::Refusal;
use super::space;
use super::speed::{self, SpeedBound};
use super::verdict::EdgeVerdict;

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
    /// Judge the edge's control legs. Off for a stand-in upper rung, which
    /// has no controls of its own.
    pub controls: bool,
    /// Require both rungs of a unit to have started from the same model: the
    /// train probe taken before any step must be equal, bit for bit.
    pub same_initial_probe: bool,
}

impl CompareOptions {
    pub fn new(axes: Axes, cross_stack: CrossStackOptions) -> Self {
        Self {
            axes,
            cross_stack,
            controls: true,
            same_initial_probe: false,
        }
    }
}

fn control_rule<'a>(edge: &Edge<'a>) -> Option<&'a ControlRule> {
    match edge.kind() {
        EdgeKind::CrossStack(rules) => match &rules.outcome {
            CrossStackOutcome::SeededLoss { control, .. } => control.as_ref(),
            _ => None,
        },
        EdgeKind::Exact(_) => None,
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
/// one experiment. The field a control overrides is the single exception:
/// it must agree within the measured legs and within the control legs, and
/// differs between them by the control's own definition.
fn identity(
    workload: Workload,
    control: Option<&ControlRule>,
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
        } else if control.is_some_and(|rule| rule.field == *field) {
            let (controls, measured): (Vec<&Leg>, Vec<&Leg>) = legs
                .iter()
                .copied()
                .partition(|leg| matches!(leg.name.take, Take::Control(_)));
            refusals.extend(disagreement(field, &measured));
            refusals.extend(disagreement(field, &controls));
        } else {
            refusals.extend(disagreement(field, legs));
        }
    }
    (refusals, unclean)
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
    for (rung, legs) in [(edge.lower(), lower), (edge.upper(), upper)] {
        let absent: Vec<&Unit> = all_units
            .iter()
            .copied()
            .filter(|u| legs.primary(u).is_none())
            .collect();
        refusals.extend(absent.iter().map(|unit| Refusal::MissingLeg {
            rung: rung.name.clone(),
            unit: unit.as_str().to_owned(),
            take: "r1".to_owned(),
        }));
        if all_units.is_empty() {
            refusals.push(Refusal::MissingLeg {
                rung: rung.name.clone(),
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
                    leg: format!("{}__{}__r1", edge.upper().name, unit.as_str()),
                    premise: "same_initial_probe",
                    reason: format!(
                        "the untrained probe reads {above:?}; the lower rung's reads {below:?}"
                    ),
                });
            }
        }
    }

    // Identity: what the legs must agree on.
    let control = control_rule(edge).filter(|_| options.controls);
    let is_declared_control = |leg: &&Leg| {
        control.is_some_and(|rule| leg.name.take == Take::Control(rule.take.to_owned()))
    };
    let entering: Vec<&Leg> = [lower, upper]
        .into_iter()
        .flat_map(|legs| {
            units
                .iter()
                .flat_map(|unit| legs.repeats(unit))
                .chain(legs.controls().filter(is_declared_control))
        })
        .collect();
    let (identity_refusals, identity_unclean) = identity(workload, control, &entering);
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
        EdgeKind::Exact(_) => outcome::digests(&pair),
        EdgeKind::CrossStack(rules) => outcome::cross_stack(
            &pair,
            &rules.outcome,
            &edge.lower().premises,
            &edge.upper().premises,
            &options.cross_stack,
            options.controls,
        ),
    });
    let swept =
        |rules: &Option<super::definition::ShapeRules>| rules.filter(|_| options.axes.shape);
    let speed = options.axes.speed.then(|| match edge.kind() {
        EdgeKind::Exact(rules) => speed::speed(
            &pair,
            SpeedBound::OverheadBudget(rules.overhead_budget),
            swept(&rules.shape).as_ref(),
            None,
        ),
        EdgeKind::CrossStack(rules) => {
            let slack = match &rules.outcome {
                CrossStackOutcome::SeededLoss { delta, .. } => *delta,
                _ => None,
            };
            speed::speed(
                &pair,
                SpeedBound::NonInferiority(rules.speed_bar),
                swept(&rules.shape).as_ref(),
                rules.time_to_quality_bar.zip(slack),
            )
        }
    });
    let space = options.axes.space.then(|| {
        let rules = match edge.kind() {
            EdgeKind::Exact(rules) => &rules.space,
            EdgeKind::CrossStack(rules) => &rules.space,
        };
        space::space(
            &pair,
            rules,
            edge.upper().flat_host_memory.filter(|_| options.axes.shape),
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
        edge.upper().layer,
        units.iter().map(|u| u.as_str().to_owned()).collect(),
        (outcome, speed, space),
        judgements,
        refusals,
    )
}
