//! The operator against synthetic legs: every rule with a leg set that
//! passes it and one that must fail it, and the committed sessions as
//! oracles for the readings the ladder reproduces.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::report::Nullable;

use super::compare::{compare, Axes, CompareOptions};
use super::definition::{Budgets, Difference, Edge, Ladder, RuleForce, Workload};
use super::leg::{Leg, LegName, LegSet};
use super::mutant::{self, Detection, DoseColumn, DoseLabel, DoseLadder, MutantSpec};
use super::outcome::{CrossStackOptions, Pair};
use super::premise::tests::fused_facts;
use super::refusal::Refusal;
use super::speed;
use super::verdict::{Direction, EdgeVerdict, Held, Judgement, OutcomeVerdict, Status};
use super::{run_ladder, telescoping, Axis, LadderArgs};

// ── building legs ──────────────────────────────────────────────────────────

/// A block carrying every identity field of `workload`, then `fields` over
/// it.
fn block(workload: Workload, fields: Value) -> Value {
    let mut block: serde_json::Map<String, Value> = workload
        .identity_fields()
        .iter()
        .map(|(field, nullable)| {
            let value = match nullable {
                Nullable::NonNull => json!("same"),
                Nullable::NullMeans(_) => Value::Null,
            };
            ((*field).to_owned(), value)
        })
        .collect();
    block.extend(fields.as_object().cloned().unwrap_or_default());
    Value::Object(block)
}

fn leg_in(workload: Workload, name: &str, fields: Value, dir: &Path) -> Leg {
    let report = json!({ workload.tier_key(): block(workload, fields) });
    Leg::from_report(
        workload,
        LegName::parse(&format!("{name}.json")).unwrap(),
        &report,
        dir,
    )
    .unwrap()
}

fn leg(workload: Workload, name: &str, fields: Value) -> Leg {
    leg_in(workload, name, fields, Path::new("."))
}

/// A workload's ladder under the committed budgets.
fn committed_ladder(workload: Workload) -> Ladder {
    workload.ladder_with(&Budgets::committed())
}

fn set(legs: impl IntoIterator<Item = Leg>) -> LegSet {
    let mut set = LegSet::default();
    legs.into_iter().for_each(|leg| set.insert(leg));
    set
}

fn merged(mut base: Value, over: Value) -> Value {
    base.as_object_mut()
        .unwrap()
        .extend(over.as_object().cloned().unwrap());
    base
}

fn axes(outcome: bool, speed: bool, space: bool, shape: bool) -> CompareOptions {
    let axes = Axes {
        outcome,
        speed,
        space,
        shape,
    };
    CompareOptions::new(axes, CrossStackOptions::default())
}

fn outcome_only() -> CompareOptions {
    axes(true, false, false, false)
}

/// Every axis a session at one size can measure.
fn one_size() -> CompareOptions {
    axes(true, true, true, false)
}

fn all_axes() -> CompareOptions {
    axes(true, true, true, true)
}

/// Compare the edge `lower -> upper` of `ladder` over `legs`.
fn edge_verdict(
    ladder: &Ladder,
    lower: &str,
    upper: &str,
    legs: &LegSet,
    options: &CompareOptions,
) -> EdgeVerdict {
    let span = ladder.span(Some(lower), Some(upper)).unwrap();
    assert_eq!(span.len(), 1);
    compare(
        ladder.workload,
        &span[0],
        &legs.rung(lower),
        &legs.rung(upper),
        options,
    )
}

fn judgement<'a>(verdict: &'a EdgeVerdict, rule: &str) -> &'a Judgement {
    verdict
        .judgements
        .iter()
        .find(|j| j.rule == rule)
        .unwrap_or_else(|| panic!("no judgement {rule} in {:?}", verdict.judgements))
}

fn refused(verdict: &EdgeVerdict, matches: impl Fn(&Refusal) -> bool) -> bool {
    verdict.refusals.iter().any(|r| matches(&r.refusal))
}

// ── the torch edge of train-run: a seeded, paired outcome ──────────────────

const TORCH: &str = "torch";
const FUSED: &str = "resident";

/// What a torch leg states about its run: the schedule, the probe series,
/// the ties, the admission — no arm and no counters, since no kernel of the
/// engine's ran.
fn torch_facts() -> Value {
    json!({
        "schedule": "constant", "admission_is_dense": false, "tie_fraction": 0.0,
        "epochs": 3, "train_probe_series": [3.32, 2.88, 2.74, 2.52], "backbone_dtype": "bf16"
    })
}

fn train_leg(rung: &str, facts: Value, seed: usize, take: &str, fields: Value) -> Leg {
    let base = merged(facts, json!({"seed": seed, "lr": 2e-4}));
    leg(
        Workload::TrainRun,
        &format!("{rung}__seed{seed}__{take}"),
        merged(base, fields),
    )
}

fn control_legs(seed: usize) -> [Leg; 2] {
    let stalled = json!({"lr": 0.0, "train_probe_series": [3.32, 3.32, 3.32, 3.32], "held_out_example_mean": 3.3});
    [
        train_leg(TORCH, torch_facts(), seed, "lr0", stalled.clone()),
        train_leg(FUSED, fused_facts(), seed, "lr0", stalled),
    ]
}

/// What every synthetic reference run learns: its held-out loss falls by
/// this much from the untrained model to its lowest point. The margin the
/// ladder derives is half of it.
const LEARNING_EFFECT: f64 = 0.1;

/// A run whose held-out loss is lowest at `loss`, at the middle epoch of
/// three — never the final one — having started `LEARNING_EFFECT` above it.
fn learning(loss: f64) -> Value {
    json!({
        "held_out_at_init": loss + LEARNING_EFFECT,
        "held_out_example_mean": loss + 0.02,
        "trajectory": [
            {"epoch": 0, "held_out_mean": loss + 0.05},
            {"epoch": 1, "held_out_mean": loss},
            {"epoch": 2, "held_out_mean": loss + 0.02},
        ]
    })
}

/// A leg's measured block, re-emitted as fields.
fn measured_fields(leg: &Leg) -> Value {
    serde_json::to_value(&leg.measured).unwrap()
}

/// One seed per difference: the torch rung's loss, and the fused rung's loss
/// `d` above it at every epoch; controls at the first two seeds.
fn seeded_legs(d: &[f64]) -> Vec<Leg> {
    let measured = d.iter().enumerate().flat_map(|(i, d)| {
        let (seed, loss) = (i + 1, 3.0 + 0.01 * i as f64);
        [
            train_leg(TORCH, torch_facts(), seed, "r1", learning(loss)),
            train_leg(FUSED, fused_facts(), seed, "r1", learning(loss + d)),
        ]
    });
    measured
        .chain(control_legs(1))
        .chain(control_legs(2))
        .collect()
}

fn torch_verdict(legs: Vec<Leg>) -> EdgeVerdict {
    edge_verdict(
        &committed_ladder(Workload::TrainRun),
        TORCH,
        FUSED,
        &set(legs),
        &outcome_only(),
    )
}

fn alternating(magnitude: f64) -> Vec<f64> {
    (0..12)
        .map(|i| if i % 2 == 0 { magnitude } else { -magnitude })
        .collect()
}

#[test]
fn small_paired_differences_are_green_non_inferior_and_equivalent() {
    let verdict = torch_verdict(seeded_legs(&alternating(0.004)));
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    for rule in [
        "outcome_non_inferior",
        "equivalent_within_delta",
        "no_directional_difference",
    ] {
        assert_eq!(judgement(&verdict, rule).held, Held::Passed, "{rule}");
    }
}

#[test]
fn a_centred_but_wide_scatter_detects_nothing_and_establishes_nothing() {
    // No direction is detected, and none of that is evidence of "no worse":
    // the interval of the mean runs past the margin on both sides, so the
    // claim is not made.
    let verdict = torch_verdict(seeded_legs(&alternating(0.3)));
    assert_eq!(
        judgement(&verdict, "no_directional_difference").held,
        Held::Passed
    );
    let claim = judgement(&verdict, "outcome_non_inferior");
    assert_eq!((claim.held, claim.force), (Held::Failed, RuleForce::Hard));
    assert_eq!(verdict.status, Status::Red);
}

/// Better is not a failure of "as good as": an upper rung whose loss is lower
/// by more than the margin, without the concordance a direction needs, makes
/// the claim and is not equivalent — reported, never failed.
#[test]
fn an_upper_rung_better_by_more_than_the_margin_is_non_inferior_and_not_equivalent() {
    let d: Vec<f64> = (0..12)
        .map(|i| if i % 3 == 0 { 0.02 } else { -0.12 })
        .collect();
    let verdict = torch_verdict(seeded_legs(&d));
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    assert_eq!(
        judgement(&verdict, "outcome_non_inferior").held,
        Held::Passed
    );
    let equivalent = judgement(&verdict, "equivalent_within_delta");
    assert_eq!(
        (equivalent.held, equivalent.force),
        (Held::Failed, RuleForce::Evidence)
    );
    // The mirror image — worse by the same amount — fails the claim.
    let worse: Vec<f64> = d.iter().map(|d| -d).collect();
    let verdict = torch_verdict(seeded_legs(&worse));
    assert_eq!(
        judgement(&verdict, "outcome_non_inferior").held,
        Held::Failed
    );
    assert_eq!(verdict.status, Status::Red);
}

#[test]
fn a_concordant_degradation_is_red_and_an_improvement_is_investigated() {
    let worse = torch_verdict(seeded_legs(&[0.1; 12]));
    assert_eq!(worse.status, Status::Red);
    assert!(matches!(
        worse.outcome,
        Some(OutcomeVerdict::SeededLoss {
            direction: Direction::Degradation,
            critical_count: Some(11),
            ..
        })
    ));
    // A detected improvement is non-inferior by construction and is still
    // an anomaly to investigate.
    let better = torch_verdict(seeded_legs(&[-0.1; 12]));
    assert_eq!(
        judgement(&better, "outcome_non_inferior").held,
        Held::Passed
    );
    assert_eq!(better.status, Status::RedForInvestigation);
}

#[test]
fn ten_of_twelve_is_not_a_direction_and_eleven_is() {
    let signs = |positive: usize| {
        (0..12)
            .map(|i| if i < positive { 0.001 } else { -0.001 })
            .collect::<Vec<_>>()
    };
    assert_eq!(torch_verdict(seeded_legs(&signs(10))).status, Status::Green);
    assert_eq!(torch_verdict(seeded_legs(&signs(11))).status, Status::Red);
}

#[test]
fn a_seed_count_the_rule_is_not_stated_for_is_refused() {
    let verdict = torch_verdict(seeded_legs(&alternating(0.004)[..11]));
    assert_eq!(verdict.status, Status::Invalid);
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::WrongUnitCount {
            clean: 11,
            required: 12,
            ..
        }
    )));
}

#[test]
fn a_leg_that_fails_a_premise_is_measured_but_not_counted() {
    let mut legs = seeded_legs(&alternating(0.004));
    let flat = merged(
        learning(3.0),
        json!({"train_probe_series": [3.3, 3.3, 3.3, 3.3]}),
    );
    legs[1] = train_leg(FUSED, fused_facts(), 1, "r1", flat);
    let verdict = torch_verdict(legs);
    assert_eq!(verdict.status, Status::Invalid);
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::PremiseViolated {
            premise: "learning_happened",
            ..
        }
    )));
    let Some(OutcomeVerdict::SeededLoss {
        per_unit,
        clean_units,
        ..
    }) = &verdict.outcome
    else {
        panic!("no paired outcome");
    };
    assert_eq!(*clean_units, 11);
    assert!(per_unit[0].d.is_some() && !per_unit[0].clean);
}

#[test]
fn a_missing_leg_is_refused_by_name() {
    let mut legs = seeded_legs(&alternating(0.004));
    legs.remove(1);
    let verdict = torch_verdict(legs);
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::MissingLeg { rung, unit, .. } if rung == FUSED && unit == "seed1")
    ));
}

#[test]
fn legs_that_disagree_on_identity_are_refused_even_when_each_seed_agrees_with_itself() {
    // Seeds 7..12 ran against another held-out fixture, on both rungs: every
    // seed's own pair agrees, and the sweep is still two experiments.
    let legs = seeded_legs(&alternating(0.004))
        .into_iter()
        .map(|l| {
            let seed: usize = l.name.unit.as_str()[4..].parse().unwrap();
            if seed > 6 {
                let facts = if l.name.rung == FUSED {
                    fused_facts()
                } else {
                    torch_facts()
                };
                let fields = merged(
                    measured_fields(&l),
                    json!({"heldout_pairs_sha256": "other"}),
                );
                train_leg(&l.name.rung, facts, seed, "r1", fields)
            } else {
                l
            }
        })
        .collect();
    let verdict = torch_verdict(legs);
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::IdentityDisagreement { field, .. } if field == "heldout_pairs_sha256")
    ));
}

#[test]
fn an_identity_field_a_leg_does_not_state_is_refused() {
    let mut legs = seeded_legs(&alternating(0.004));
    let fields = merged(learning(3.0), json!({"lora_rank": null}));
    legs[0] = train_leg(TORCH, torch_facts(), 1, "r1", fields);
    let verdict = torch_verdict(legs);
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::IdentityMissing { field, .. } if field == "lora_rank")
    ));
}

#[test]
fn a_repeat_further_from_its_first_run_than_the_seeds_are_from_each_other_is_refused() {
    let mut legs = seeded_legs(&alternating(0.004));
    legs.push(train_leg(FUSED, fused_facts(), 3, "r2", learning(9.0)));
    let verdict = torch_verdict(legs);
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::RepeatExceedsSpread { unit, .. } if unit == "seed3")
    ));

    let mut legs = seeded_legs(&alternating(0.004));
    let same = measured_fields(&legs[5]);
    legs.push(train_leg(FUSED, fused_facts(), 3, "r2", same));
    assert_eq!(torch_verdict(legs).status, Status::Green);
}

/// The margin is not chosen: it is half the learning effect the reference
/// establishes in the session, and the effect is the lower confidence bound
/// of the reference's improvement from its untrained loss.
#[test]
fn the_margin_is_half_the_reference_rungs_established_learning_effect() {
    let verdict = torch_verdict(seeded_legs(&alternating(0.004)));
    let Some(OutcomeVerdict::SeededLoss {
        assay: Some(assay),
        per_unit,
        ..
    }) = &verdict.outcome
    else {
        panic!("no paired outcome");
    };
    assert!(per_unit.iter().all(|u| u
        .reference_improvement
        .is_some_and(|i| (i - LEARNING_EFFECT).abs() < 1e-12)));
    assert!((assay.established_effect - LEARNING_EFFECT).abs() < 1e-12);
    assert!((assay.delta - LEARNING_EFFECT / 2.0).abs() < 1e-12);
    assert!(assay.sensitive);
    assert_eq!(
        judgement(&verdict, "outcome_non_inferior").bound,
        super::verdict::Bound::Derived { value: assay.delta }
    );
}

/// Both rungs are read where the reference learned the most — the middle
/// epoch — so a fused rung that overfits harder at the end is still as good
/// at the judged point.
#[test]
fn the_judged_point_is_the_reference_rungs_minimum_not_its_final_epoch() {
    let legs = seeded_legs(&alternating(0.004))
        .into_iter()
        .map(|l| {
            if l.name.rung != FUSED || l.name.take.to_string() != "r1" {
                return l;
            }
            let seed: usize = l.name.unit.as_str()[4..].parse().unwrap();
            let mut fields = measured_fields(&l);
            fields["trajectory"][2]["held_out_mean"] = json!(9.0);
            fields["held_out_example_mean"] = json!(9.0);
            train_leg(FUSED, fused_facts(), seed, "r1", fields)
        })
        .collect();
    let verdict = torch_verdict(legs);
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    let Some(OutcomeVerdict::SeededLoss { per_unit, .. }) = &verdict.outcome else {
        panic!("no paired outcome");
    };
    assert!(per_unit.iter().all(|u| u.epoch == Some(1)));
}

#[test]
fn a_reference_that_did_not_record_its_untrained_loss_cannot_set_a_margin() {
    let legs = seeded_legs(&alternating(0.004))
        .into_iter()
        .map(|l| {
            if l.name.rung != TORCH || l.name.take.to_string() != "r1" {
                return l;
            }
            let seed: usize = l.name.unit.as_str()[4..].parse().unwrap();
            let mut fields = measured_fields(&l);
            fields.as_object_mut().unwrap().remove("held_out_at_init");
            train_leg(TORCH, torch_facts(), seed, "r1", fields)
        })
        .collect();
    let verdict = torch_verdict(legs);
    assert_eq!(verdict.status, Status::Invalid);
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::MeasurementMissing {
            measurement: "held_out_at_init",
            ..
        }
    )));
}

#[test]
fn the_control_must_be_run_must_be_a_control_and_must_not_learn() {
    let without: Vec<Leg> = seeded_legs(&alternating(0.004))
        .into_iter()
        .filter(|l| l.name.take.to_string() != "lr0")
        .collect();
    let verdict = torch_verdict(without.clone());
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::ControlMissing {
            found: 0,
            required: 2,
            ..
        }
    )));

    // Declared absent: not refused, and the verdict says so.
    let mut waived = outcome_only();
    waived.cross_stack.waive_control = true;
    let verdict = edge_verdict(
        &committed_ladder(Workload::TrainRun),
        TORCH,
        FUSED,
        &set(without.clone()),
        &waived,
    );
    assert_eq!(verdict.status, Status::Green);
    assert!(
        matches!(&verdict.outcome, Some(OutcomeVerdict::SeededLoss { control: Some(c), .. }) if c.waived)
    );

    let control = |fields: Value| {
        let mut legs = without.clone();
        legs.extend(control_legs(2));
        legs.push(train_leg(TORCH, torch_facts(), 1, "lr0", fields.clone()));
        legs.push(train_leg(FUSED, fused_facts(), 1, "lr0", fields));
        torch_verdict(legs)
    };
    let learned = control(json!({"lr": 0.0, "held_out_example_mean": 3.3}));
    assert!(refused(
        &learned,
        |r| matches!(r, Refusal::ControlInvalid { reason, .. } if reason.contains("cannot learn"))
    ));
    let never_a_control =
        control(json!({"train_probe_series": [3.3, 3.3, 3.3, 3.3], "held_out_example_mean": 3.3}));
    assert!(refused(
        &never_a_control,
        |r| matches!(r, Refusal::ControlInvalid { reason, .. } if reason.contains("lr is"))
    ));
}

/// A reference whose held-out loss at the judged point is no better than
/// untrained establishes no effect, so there is no margin to derive: the
/// edge is refused, never judged against a number.
#[test]
fn a_reference_that_established_no_learning_effect_is_refused_as_insensitive() {
    let workload = Workload::PredictorTrainRun;
    let legs = (1..=12).flat_map(|seed| {
        ["torch", "in-process"].map(|rung| {
            let loss = 0.5 + 0.01 * seed as f64;
            let fields = json!({
                "seed": seed, "schedule": "constant", "epochs": 2,
                "train_probe_series": [1.0, 0.8, 0.6],
                "held_out_at_init": loss + 0.001 * if seed % 2 == 0 { 1.0 } else { -1.0 },
                "trajectory": [{"epoch": 0, "held_out_mean": loss}, {"epoch": 1, "held_out_mean": loss + 0.01}]
            });
            leg(workload, &format!("{rung}__seed{seed}__r1"), fields)
        })
    });
    let verdict = edge_verdict(
        &committed_ladder(workload),
        "torch",
        "in-process",
        &set(legs),
        &outcome_only(),
    );
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::AssayInsensitive { lower_bound, .. } if *lower_bound <= 0.0
    )));
    assert!(verdict
        .judgements
        .iter()
        .all(|j| j.rule != "outcome_non_inferior"));
    assert_eq!(verdict.status, Status::Invalid);
}

// ── exact edges of encode: digests, overhead, shape, space ─────────────────

const PLAN: &str = "plan";
const PARTITIONED: &str = "plan-partitioned";
const PLACED: &str = "placed";

fn encode_leg(rung: &str, work: u64, take: &str, series: Vec<f64>, fields: Value) -> Leg {
    let base = json!({
        "rows": work, "corpus_sha256": format!("corpus-{work}"), "token_lengths_sha256": format!("lengths-{work}"),
        "tokens": work * 8, "work": work, "outcome_digest": format!("digest-{work}"),
        "iter_wall_s": series, "compute_precision": "f32",
        "peak_rss_bytes": {"value": 1.0e9, "unit": "bytes"},
        "peak_vram_bytes": {"value": 2.0e9, "unit": "bytes"}
    });
    leg(
        Workload::Encode,
        &format!("{rung}__rows{work}__{take}"),
        merged(base, fields),
    )
}

fn steady(seconds: f64) -> Vec<f64> {
    vec![seconds; 32]
}

/// A ladder whose rules carry the given bounds, for exercising a rule's
/// mechanics. The table is well-formed, not a measurement: every entry
/// names a file that exists, none an artifact that measured anything.
fn budgeted(workload: Workload, entries: &[(&str, &str, f64)]) -> Ladder {
    let budgets: Vec<Value> = entries
        .iter()
        .map(|(edge, rule, bound)| {
            json!({
                "workload": workload, "edge": edge, "rule": rule,
                "bound": bound, "measured_from": "Cargo.toml"
            })
        })
        .collect();
    let budgets: Budgets = serde_json::from_value(json!({ "budgets": budgets })).unwrap();
    workload.ladder_with(&budgets)
}

/// `encode`'s ladder with every rule of `plan -> plan-partitioned` bounded.
fn encode_test_ladder() -> Ladder {
    let edge = format!("{PLAN} -> {PARTITIONED}");
    budgeted(
        Workload::Encode,
        &[
            (&edge, "overhead_budget", 1.10),
            (&edge, "host_memory_ratio", 1.10),
            (&edge, "device_memory_ratio", 1.10),
            (&edge, "fixed_cost", 100.0),
            (&edge, "per_work_cost", 1.10),
        ],
    )
}

fn exact_verdict(lower: Leg, upper: Leg) -> EdgeVerdict {
    edge_verdict(
        &encode_test_ladder(),
        PLAN,
        PARTITIONED,
        &set([lower, upper]),
        &one_size(),
    )
}

/// A table that measures no bound for a rule: the rule is reported
/// unbudgeted, its judgement carries no pass or fail, and — being evidence —
/// it decides nothing. A hard rule left unbudgeted is a refusal.
#[test]
fn a_rule_with_no_measured_budget_is_reported_unbudgeted_and_never_judged() {
    let verdict = edge_verdict(
        &budgeted(Workload::Encode, &[]),
        PLAN,
        PARTITIONED,
        &set([
            encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
            encode_leg(PARTITIONED, 16, "r1", steady(3.0), json!({})),
        ]),
        &one_size(),
    );
    let overhead = judgement(&verdict, "overhead_budget");
    assert_eq!(overhead.held, Held::Unjudged);
    assert_eq!(overhead.bound, super::verdict::Bound::Unbudgeted);
    assert!(verdict.speed.as_ref().unwrap().cost.of_medians > 2.9);
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);

    let hard = EdgeVerdict::conclude(
        "edge".into(),
        &Difference::Layer { name: "a layer" },
        vec![],
        (None, None, None),
        vec![Judgement::new(
            "overhead_budget",
            RuleForce::Hard,
            super::verdict::Bound::Unbudgeted,
            None,
            "",
        )],
        vec![],
    );
    assert_eq!(hard.status, Status::Invalid);
    assert!(refused(
        &hard,
        |r| matches!(r, Refusal::Unbudgeted { rule, .. } if rule == "overhead_budget")
    ));
}

#[test]
fn equal_digests_pass_and_a_differing_digest_is_refused() {
    let lower = || encode_leg(PLAN, 16, "r1", steady(1.0), json!({}));
    let same = exact_verdict(
        lower(),
        encode_leg(PARTITIONED, 16, "r1", steady(1.0), json!({})),
    );
    assert_eq!(same.status, Status::Green, "{:?}", same.refusals);
    assert!(matches!(
        same.outcome,
        Some(OutcomeVerdict::Digest {
            units_equal: 1,
            units: 1
        })
    ));

    let differs = exact_verdict(
        lower(),
        encode_leg(
            PARTITIONED,
            16,
            "r1",
            steady(1.0),
            json!({"outcome_digest": "x"}),
        ),
    );
    assert_eq!(differs.status, Status::Invalid);
    assert!(refused(&differs, |r| matches!(
        r,
        Refusal::DigestMismatch { .. }
    )));

    let absent = exact_verdict(
        lower(),
        encode_leg(
            PARTITIONED,
            16,
            "r1",
            steady(1.0),
            json!({"outcome_digest": null}),
        ),
    );
    assert!(refused(&absent, |r| matches!(
        r,
        Refusal::MeasurementMissing {
            measurement: "outcome_digest",
            ..
        }
    )));
}

#[test]
fn an_overhead_a_hair_under_budget_passes_and_a_hair_over_fails() {
    let at = |upper: f64| {
        exact_verdict(
            encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
            encode_leg(PARTITIONED, 16, "r1", steady(upper), json!({})),
        )
    };
    let (under, over) = (at(1.0999), at(1.1001));
    assert_eq!(judgement(&under, "overhead_budget").held, Held::Passed);
    assert_eq!(under.status, Status::Green);
    // A budget is evidence, reported beside a verdict it does not decide.
    let budget = judgement(&over, "overhead_budget");
    assert_eq!(
        (budget.held, budget.force),
        (Held::Failed, RuleForce::Evidence)
    );
    assert_eq!(over.status, Status::Green);
}

#[test]
fn the_interval_is_judged_not_the_point() {
    // Medians put the cost at 1.08, under budget; the legs are noisy enough
    // that the interval is not.
    let noisy = |level: f64| {
        (0..32)
            .map(|i| level * (1.0 + 0.08 * ((i * 7) % 5) as f64))
            .collect::<Vec<_>>()
    };
    let verdict = exact_verdict(
        encode_leg(PLAN, 16, "r1", noisy(1.0), json!({})),
        encode_leg(PARTITIONED, 16, "r1", noisy(1.08), json!({})),
    );
    let speed = verdict.speed.as_ref().unwrap();
    assert!(
        speed.cost.of_medians < 1.10 && speed.cost.interval.upper > 1.10,
        "{:?}",
        speed.cost
    );
    assert_eq!(judgement(&verdict, "overhead_budget").held, Held::Failed);
}

#[test]
fn a_trending_series_and_a_short_one_are_refused() {
    let warming: Vec<f64> = (0..64).map(|i| 1.0 + 0.01 * i as f64).collect();
    let trending = exact_verdict(
        encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
        encode_leg(PARTITIONED, 16, "r1", warming, json!({})),
    );
    assert_eq!(trending.status, Status::Invalid);
    assert!(refused(
        &trending,
        |r| matches!(r, Refusal::NonStationary { leg, .. } if leg.starts_with(PARTITIONED))
    ));

    let short = exact_verdict(
        encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
        encode_leg(PARTITIONED, 16, "r1", vec![1.0; 5], json!({})),
    );
    assert!(refused(&short, |r| matches!(
        r,
        Refusal::TooFewSamples { got: 5, .. }
    )));
}

/// A run that starts slow and settles is cut where it settles: the cost is
/// read from the steady iterations, the transient named per leg.
#[test]
fn a_warming_start_is_cut_and_the_cost_read_from_what_settled() {
    let warming: Vec<f64> = (0..48)
        .map(|i| {
            if i < 12 {
                1.3
            } else {
                1.0 + 0.001 * (i % 3) as f64
            }
        })
        .collect();
    let verdict = exact_verdict(
        encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
        encode_leg(PARTITIONED, 16, "r1", warming, json!({})),
    );
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    let speed = verdict.speed.as_ref().unwrap();
    assert!(
        (speed.cost.of_medians - 1.001).abs() < 1e-9,
        "{}",
        speed.cost.of_medians
    );
    let cut = speed
        .settled
        .iter()
        .find(|l| l.leg.starts_with(PARTITIONED))
        .unwrap();
    assert_eq!(
        (cut.transient, cut.iterations, cut.at_limit),
        (12, 48, false)
    );
}

/// A run that drifts throughout is cut at the half-way limit and still
/// refused: what is left drifts too, and the trend test says so.
#[test]
fn a_run_that_never_settles_is_refused_on_what_is_left() {
    let drifting: Vec<f64> = (0..64).map(|i| 1.0 + 0.01 * i as f64).collect();
    let verdict = exact_verdict(
        encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
        encode_leg(PARTITIONED, 16, "r1", drifting, json!({})),
    );
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::NonStationary { leg, .. } if leg.starts_with(PARTITIONED))
    ));
}

#[test]
fn a_hard_speed_rule_with_nothing_to_measure_is_refused() {
    // The revision edge's band rule is hard; an evidence rule with nothing
    // to measure is only reported as unmeasured.
    let without_series =
        |name: &str| encode_leg(name, 16, "r1", steady(1.0), json!({"iter_wall_s": null}));
    let evidence = exact_verdict(
        encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
        without_series(PARTITIONED),
    );
    assert_eq!(judgement(&evidence, "overhead_budget").held, Held::Unjudged);
    assert_eq!(evidence.status, Status::Green, "{:?}", evidence.refusals);
    let hard = revision_verdict(vec![
        encode_leg("direct@base", 16, "r1", steady(1.0), json!({})),
        without_series("direct@revised"),
    ]);
    assert!(refused(&hard, |r| matches!(
        r,
        Refusal::MeasurementMissing {
            measurement: "speed_within_noise_band",
            ..
        }
    )));
    assert_eq!(hard.status, Status::Invalid);
}

#[test]
fn a_cost_inside_the_same_rung_noise_band_is_indistinguishable_from_one() {
    let jitter = |phase: usize| {
        (0..48)
            .map(|i| 1.0 + 0.02 * ((i + phase) % 4) as f64)
            .collect::<Vec<_>>()
    };
    let legs = [
        encode_leg(PLAN, 16, "r1", jitter(0), json!({})),
        encode_leg(PLAN, 16, "r2", jitter(1), json!({})),
        encode_leg(PARTITIONED, 16, "r1", jitter(2), json!({})),
        encode_leg(PARTITIONED, 16, "r2", jitter(3), json!({})),
    ];
    let verdict = edge_verdict(
        &committed_ladder(Workload::Encode),
        PLAN,
        PARTITIONED,
        &set(legs),
        &one_size(),
    );
    let speed = verdict.speed.unwrap();
    assert!(speed.noise_band.is_some_and(|band| band >= 1.0));
    assert_eq!(speed.indistinguishable_from_one, Some(true));
}

/// Four sizes on a line `fixed + per_work * work`, one leg each.
fn sweep(rung: &str, fixed: f64, per_work: f64, fields: impl Fn(u64) -> Value) -> Vec<Leg> {
    [16_u64, 256, 4096, 65536]
        .into_iter()
        .map(|work| {
            encode_leg(
                rung,
                work,
                "r1",
                steady(fixed + per_work * work as f64),
                fields(work),
            )
        })
        .collect()
}

#[test]
fn a_fixed_cost_regression_is_attributed_to_fixed_and_not_to_per_work() {
    let mut legs = sweep(PLAN, 0.001, 1e-5, |_| json!({}));
    legs.extend(sweep(PARTITIONED, 0.004, 1e-5, |_| json!({})));
    let verdict = edge_verdict(
        &encode_test_ladder(),
        PLAN,
        PARTITIONED,
        &set(legs),
        &all_axes(),
    );
    let shape = verdict.speed.as_ref().unwrap().shape.as_ref().unwrap();
    assert!(
        (shape.fixed_work_equivalent - 300.0).abs() < 1e-3,
        "{shape:?}"
    );
    assert!((shape.per_work_ratio - 1.0).abs() < 1e-9);
    assert_eq!(judgement(&verdict, "fixed_cost").held, Held::Failed);
    assert_eq!(judgement(&verdict, "per_work_cost").held, Held::Passed);

    // The mirror image: the same fixed cost, 30% more per unit of work.
    let mut legs = sweep(PLAN, 0.001, 1e-5, |_| json!({}));
    legs.extend(sweep(PARTITIONED, 0.001, 1.3e-5, |_| json!({})));
    let verdict = edge_verdict(
        &encode_test_ladder(),
        PLAN,
        PARTITIONED,
        &set(legs),
        &all_axes(),
    );
    assert_eq!(judgement(&verdict, "fixed_cost").held, Held::Passed);
    assert_eq!(judgement(&verdict, "per_work_cost").held, Held::Failed);
}

#[test]
fn time_that_is_not_a_line_in_work_is_refused_rather_than_fitted() {
    let mut legs = sweep(PLAN, 0.001, 1e-5, |_| json!({}));
    legs.extend([16_u64, 256, 4096, 65536].into_iter().map(|work| {
        encode_leg(
            PARTITIONED,
            work,
            "r1",
            steady(1e-3 * (work as f64).sqrt()),
            json!({}),
        )
    }));
    let verdict = edge_verdict(
        &committed_ladder(Workload::Encode),
        PLAN,
        PARTITIONED,
        &set(legs),
        &all_axes(),
    );
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::ShapeFitPoor { rung, .. } if rung == PARTITIONED)
    ));
}

/// Interleaved repeats share one process, so they carry no memory; each
/// rung's memory comes from its run alone, whose time pairs with nothing.
#[test]
fn memory_is_read_from_the_runs_alone_and_speed_from_the_repeats() {
    let shared = json!({
        "peak_rss_bytes": {"value": null, "unit": "bytes"},
        "peak_vram_bytes": {"value": null, "unit": "bytes"}
    });
    let alone = |rss: f64, seconds: f64| {
        json!({
            "peak_rss_bytes": {"value": rss, "unit": "bytes"},
            "peak_vram_bytes": {"value": 2.0e9, "unit": "bytes"},
            "iter_wall_s": steady(seconds)
        })
    };
    let verdict = edge_verdict(
        &encode_test_ladder(),
        PLAN,
        PARTITIONED,
        &set([
            encode_leg(PLAN, 16, "r1", steady(1.0), shared.clone()),
            encode_leg(PARTITIONED, 16, "r1", steady(1.0), shared),
            // Alone, the partitioned rung holds 5 % more memory — and its
            // lone session ran twice as slow, which no speed rule may read.
            encode_leg(PLAN, 16, "a1", steady(1.0), alone(1.0e9, 1.0)),
            encode_leg(PARTITIONED, 16, "a1", steady(1.0), alone(1.05e9, 2.0)),
        ]),
        &one_size(),
    );
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    let space = verdict.space.as_ref().unwrap();
    assert!((space.host_ratio.unwrap() - 1.05).abs() < 1e-9);
    assert!((verdict.speed.as_ref().unwrap().cost.of_medians - 1.0).abs() < 1e-9);
}

#[test]
fn memory_over_budget_and_unmeasured_memory_are_reported_as_evidence() {
    let lower = || encode_leg(PLAN, 16, "r1", steady(1.0), json!({}));
    let heavy = json!({"peak_rss_bytes": {"value": 1.2e9, "unit": "bytes"}});
    let over = exact_verdict(
        lower(),
        encode_leg(PARTITIONED, 16, "r1", steady(1.0), heavy),
    );
    let host = judgement(&over, "host_memory_ratio");
    assert_eq!((host.held, host.force), (Held::Failed, RuleForce::Evidence));
    assert_eq!(judgement(&over, "device_memory_ratio").held, Held::Passed);
    assert_eq!(over.status, Status::Green);

    let unmeasured = json!({"peak_vram_bytes": {"value": null, "unit": "bytes"}});
    let verdict = exact_verdict(
        lower(),
        encode_leg(PARTITIONED, 16, "r1", steady(1.0), unmeasured),
    );
    assert_eq!(
        judgement(&verdict, "device_memory_ratio").held,
        Held::Unjudged
    );
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
}

#[test]
fn host_memory_that_grows_with_the_training_set_breaks_the_streaming_rung() {
    let ladder = budgeted(
        Workload::TrainRun,
        &[("streamed", "host_memory_flat_in_work", 64.0)],
    );
    let legs = |bytes_per_row: f64| {
        let mut legs = vec![];
        for (seed, rows) in [(1_usize, 1000.0_f64), (2, 10_000.0), (3, 100_000.0)] {
            for rung in [FUSED, "streamed"] {
                let fields = json!({
                    "work": rows, "outcome_digest": format!("adapter-{seed}"),
                    "peak_rss_bytes": 4.0e9 + bytes_per_row * rows, "peak_vram_bytes": 1.0e9,
                    "held_out_example_mean": 3.0
                });
                legs.push(train_leg(rung, fused_facts(), seed, "r1", fields));
            }
        }
        set(legs)
    };
    let space_only = axes(true, false, true, true);
    let flat = edge_verdict(&ladder, FUSED, "streamed", &legs(1.0), &space_only);
    assert_eq!(
        judgement(&flat, "host_memory_flat_in_work").held,
        Held::Passed
    );
    assert_eq!(flat.status, Status::Green, "{:?}", flat.refusals);
    let growing = edge_verdict(&ladder, FUSED, "streamed", &legs(800.0), &space_only);
    assert_eq!(
        judgement(&growing, "host_memory_flat_in_work").held,
        Held::Failed
    );
}

// ── telescoping ────────────────────────────────────────────────────────────

fn ladder_of_three(placed: f64) -> (Vec<EdgeVerdict>, LegSet) {
    let legs = set([
        encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
        encode_leg(PARTITIONED, 16, "r1", steady(1.05), json!({})),
        encode_leg(PLACED, 16, "r1", steady(placed), json!({})),
    ]);
    let ladder = committed_ladder(Workload::Encode);
    let span = ladder.span(Some(PLAN), Some(PLACED)).unwrap();
    let edges = span
        .iter()
        .map(|e| {
            compare(
                Workload::Encode,
                e,
                &legs.rung(&e.lower().name),
                &legs.rung(&e.upper().name),
                &one_size(),
            )
        })
        .collect();
    (edges, legs)
}

#[test]
fn a_product_that_contradicts_the_direct_ratio_is_refused() {
    let ladder = committed_ladder(Workload::Encode);
    let span = ladder.span(Some(PLAN), Some(PLACED)).unwrap();
    let (edges, _) = ladder_of_three(1.10);
    let direct = |placed: f64| {
        set([
            encode_leg(PLAN, 16, "r1", steady(1.0), json!({})),
            encode_leg(PLACED, 16, "r1", steady(placed), json!({})),
        ])
    };
    let agreed = telescoping(&edges, &span, &direct(1.10)).unwrap().unwrap();
    assert!(
        (agreed.product.upper - 1.10).abs() < 1e-9
            && (agreed.direct.of_medians - 1.10).abs() < 1e-9
    );
    let contradicted = telescoping(&edges, &span, &direct(1.30)).unwrap_err();
    assert!(matches!(
        contradicted[0],
        Refusal::TelescopingContradiction { .. }
    ));
    // No direct session: nothing to contradict.
    assert!(telescoping(&edges, &span, &LegSet::default())
        .unwrap()
        .is_none());
}

// ── row agreement and law ──────────────────────────────────────────────────

fn write_vectors(dir: &Path, file: &str, rows: &[Vec<f32>]) {
    let bytes: Vec<u8> = rows
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(dir.join(file), bytes).unwrap();
}

fn row_verdict(workload: Workload, upper_rung: &str, perturbation: f32) -> EdgeVerdict {
    let dir = tempfile::tempdir().unwrap();
    let rows: Vec<Vec<f32>> = (0..8)
        .map(|r| (0..16).map(|c| ((r * 16 + c) as f32).sin() + 2.0).collect())
        .collect();
    let perturbed: Vec<Vec<f32>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(i, v)| v * (1.0 + perturbation * (i % 3) as f32))
                .collect()
        })
        .collect();
    write_vectors(dir.path(), "lower.f32", &rows);
    write_vectors(dir.path(), "upper.f32", &perturbed);
    let fields =
        |file: &str| json!({"vectors_file": file, "vector_dim": 16, "compute_precision": "f32"});
    let legs = set([
        leg_in(
            workload,
            "torch__rows8__r1",
            fields("lower.f32"),
            dir.path(),
        ),
        leg_in(
            workload,
            &format!("{upper_rung}__rows8__r1"),
            fields("upper.f32"),
            dir.path(),
        ),
    ]);
    edge_verdict(
        &committed_ladder(workload),
        "torch",
        upper_rung,
        &legs,
        &outcome_only(),
    )
}

#[test]
fn rows_within_the_precision_allowance_agree_and_rows_beyond_it_do_not() {
    for (workload, upper) in [
        (Workload::Encode, "direct"),
        (Workload::Propagate, "torch-geometric"),
    ] {
        let close = row_verdict(workload, upper, 1e-6);
        assert_eq!(
            judgement(&close, "row_agreement").held,
            Held::Passed,
            "{workload:?}"
        );
        let far = row_verdict(workload, upper, 1e-2);
        assert_eq!(
            judgement(&far, "row_agreement").held,
            Held::Failed,
            "{workload:?}"
        );
        assert!(matches!(
            far.outcome,
            Some(OutcomeVerdict::RowAgreement {
                rows: 8,
                rows_beyond_bound: 8,
                ..
            })
        ));
    }
}

#[test]
fn vectors_of_different_shape_are_refused() {
    let dir = tempfile::tempdir().unwrap();
    write_vectors(dir.path(), "a.f32", &vec![vec![1.0; 16]; 8]);
    write_vectors(dir.path(), "b.f32", &vec![vec![1.0; 16]; 7]);
    let fields =
        |file: &str| json!({"vectors_file": file, "vector_dim": 16, "compute_precision": "f32"});
    let legs = set([
        leg_in(
            Workload::Encode,
            "torch__rows8__r1",
            fields("a.f32"),
            dir.path(),
        ),
        leg_in(
            Workload::Encode,
            "direct__rows8__r1",
            fields("b.f32"),
            dir.path(),
        ),
    ]);
    let verdict = edge_verdict(
        &committed_ladder(Workload::Encode),
        "torch",
        "direct",
        &legs,
        &outcome_only(),
    );
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::VectorsMalformed { .. }
    )));
}

fn law_verdict(sampler_counts: Value, claimed_digest: Option<&str>) -> EdgeVerdict {
    let dir = tempfile::tempdir().unwrap();
    let law = br#"{"cells": [[0.2, 0.3, 0.5], [0.5, 0.5]]}"#;
    std::fs::write(dir.path().join("edges100.json"), law).unwrap();
    let digest = claimed_digest.map_or_else(|| hex::encode(Sha256::digest(law)), str::to_owned);
    let workload = Workload::GraphSample;
    let legs = set([
        leg(
            workload,
            "torch__edges100__r1",
            json!({"law_sha256": digest, "law_observed": [[2000, 3000, 5000], [5000, 5000]]}),
        ),
        leg(
            workload,
            "sampler__edges100__r1",
            json!({"law_sha256": digest, "law_observed": sampler_counts}),
        ),
    ]);
    let mut options = outcome_only();
    options.cross_stack.law_dir = Some(dir.path().to_owned());
    edge_verdict(
        &committed_ladder(workload),
        "torch",
        "sampler",
        &legs,
        &options,
    )
}

#[test]
fn a_sampler_that_follows_the_law_passes_and_a_biased_one_fails_hard() {
    let faithful = law_verdict(json!([[1990, 3020, 4990], [5030, 4970]]), None);
    assert_eq!(faithful.status, Status::Green, "{:?}", faithful.refusals);
    let biased = law_verdict(json!([[2600, 3000, 4400], [5000, 5000]]), None);
    assert_eq!(judgement(&biased, "law_goodness_of_fit").held, Held::Failed);
    assert_eq!(biased.status, Status::Red);
}

/// A state the sampler under-visited is pooled, not refused: the fit runs
/// over what can be tested and the edge is judged.
#[test]
fn a_sparse_state_is_pooled_and_the_law_still_judged() {
    // 20 visits of the first state expect (4, 6, 10): the 0.2 category is
    // merged with the 0.3 one: two categories pooled into one.
    let verdict = law_verdict(json!([[4, 6, 10], [5030, 4970]]), None);
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    assert!(judgement(&verdict, "law_goodness_of_fit")
        .detail
        .contains("2 categories pooled"));
}

/// A fit the statistics refuse is that refusal, once: the rule is not also
/// filed as unmeasured.
#[test]
fn a_refused_law_fit_is_filed_once_as_what_refused_it() {
    // No state of the sampler's counts can be tested.
    let verdict = law_verdict(json!([[1, 1, 1], [2, 2]]), None);
    assert_eq!(verdict.status, Status::Invalid);
    assert_eq!(
        judgement(&verdict, "law_goodness_of_fit").held,
        Held::Refused
    );
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::Statistics { .. }
    )));
    assert!(!refused(&verdict, |r| matches!(
        r,
        Refusal::MeasurementMissing { .. }
    )));
}

#[test]
fn a_law_the_legs_did_not_run_under_is_refused() {
    let verdict = law_verdict(
        json!([[2000, 3000, 5000], [5000, 5000]]),
        Some("not-the-law"),
    );
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::LawUnusable { .. }
    )));
    let cells = law_verdict(json!([[2000, 3000, 5000]]), None);
    assert!(refused(&cells, |r| matches!(r, Refusal::Statistics { .. })));
}

// ── gradient agreement ─────────────────────────────────────────────────────

const STEP_UNIT: &str = "b8s128d0";

/// A leg of the step ladder: a `torch` leg carries no counted facts, a
/// `fused` leg proves the fused arm and the flash cascade.
fn step_leg(rung: &str, take: &str, fields: Value) -> Leg {
    let facts = if rung == "fused" {
        fused_facts()
    } else {
        json!({"backbone_dtype": "bf16"})
    };
    let fields = merged(facts, fields);
    leg(
        Workload::TrainStep,
        &format!("{rung}__{STEP_UNIT}__{take}"),
        fields,
    )
}

/// A `grads` take leg: one forward at loaded weights, no warmup, no measured
/// step, no clip — free to differ from the timed repeats on exactly those.
fn grads_leg(rung: &str, tensors: &[(&str, &[f32], &[f32])]) -> Leg {
    let gradients: serde_json::Map<String, Value> = tensors
        .iter()
        .map(|(name, weight, grad)| {
            (
                (*name).to_owned(),
                json!({"shape": [grad.len()], "grad": grad, "weight": weight}),
            )
        })
        .collect();
    step_leg(
        rung,
        "grads",
        json!({"warmup": 0, "steps_measured": 0, "max_grad_norm": null, "gradients": gradients}),
    )
}

/// The edge's timed repeats, which the gradient legs are judged beside.
fn step_repeats() -> Vec<Leg> {
    let timed = || json!({"warmup": 5, "steps_measured": 20, "max_grad_norm": 1.0});
    vec![
        step_leg("torch", "r1", timed()),
        step_leg("fused", "r1", timed()),
    ]
}

const W: &[f32] = &[1.0, 2.0, 3.0];
const ZERO: &[f32] = &[0.0, 0.0, 0.0];
const G: &[f32] = &[0.1, 0.2, 0.3];

fn gradient_verdict(ladder: &Ladder, lower: Leg, upper: Leg) -> EdgeVerdict {
    let mut legs = step_repeats();
    legs.extend([lower, upper]);
    edge_verdict(ladder, "torch", "fused", &set(legs), &outcome_only())
}

#[test]
fn gradients_that_agree_at_shared_weights_pass_the_structure_and_a_zero_pair_is_vacuous() {
    let tensors: &[(&str, &[f32], &[f32])] = &[
        ("layer.0.Wqkv.lora_a", W, ZERO),
        ("layer.0.Wqkv.lora_b", W, G),
    ];
    let verdict = gradient_verdict(
        &committed_ladder(Workload::TrainStep),
        grads_leg("torch", tensors),
        grads_leg("fused", tensors),
    );
    assert_eq!(verdict.status, Status::Green, "{:?}", verdict.refusals);
    let structure = judgement(&verdict, "gradient_structure");
    assert_eq!(
        (structure.held, structure.force),
        (Held::Passed, RuleForce::Hard)
    );
    // No artifact has measured a cosine floor: the rule is reported
    // unbudgeted and decides nothing.
    let floor = judgement(&verdict, "gradient_cosine_floor");
    assert_eq!(floor.held, Held::Unjudged);
    assert_eq!(floor.bound, super::verdict::Bound::Unbudgeted);
    let Some(OutcomeVerdict::GradientAgreement {
        tensors,
        worst_cosine,
        ..
    }) = &verdict.outcome
    else {
        panic!("no gradient outcome");
    };
    assert_eq!(
        tensors.iter().map(|t| t.kind).collect::<Vec<_>>(),
        [
            super::verdict::TensorKind::Vacuous,
            super::verdict::TensorKind::Signal
        ]
    );
    assert!(worst_cosine.is_some_and(|c| (c - 1.0).abs() < 1e-12));
}

#[test]
fn a_measured_cosine_floor_judges_the_worst_tensor_as_evidence() {
    let ladder = budgeted(
        Workload::TrainStep,
        &[("torch -> fused", "gradient_cosine_floor", 0.999)],
    );
    let agreeing = gradient_verdict(
        &ladder,
        grads_leg("torch", &[("layer.0.Wqkv.lora_b", W, G)]),
        grads_leg("fused", &[("layer.0.Wqkv.lora_b", W, G)]),
    );
    assert_eq!(
        judgement(&agreeing, "gradient_cosine_floor").held,
        Held::Passed
    );
    let turned = gradient_verdict(
        &ladder,
        grads_leg("torch", &[("layer.0.Wqkv.lora_b", W, G)]),
        grads_leg("fused", &[("layer.0.Wqkv.lora_b", W, &[0.1, 0.2, -0.3])]),
    );
    let floor = judgement(&turned, "gradient_cosine_floor");
    assert_eq!(
        (floor.held, floor.force),
        (Held::Failed, RuleForce::Evidence)
    );
    assert_eq!(turned.status, Status::Green);
}

#[test]
fn a_one_sided_zero_or_differing_weights_break_the_structure_and_fail_the_edge() {
    let ladder = committed_ladder(Workload::TrainStep);
    let one_sided = gradient_verdict(
        &ladder,
        grads_leg("torch", &[("layer.0.Wqkv.lora_b", W, G)]),
        grads_leg("fused", &[("layer.0.Wqkv.lora_b", W, ZERO)]),
    );
    assert_eq!(
        judgement(&one_sided, "gradient_structure").held,
        Held::Failed
    );
    assert_eq!(one_sided.status, Status::Red);
    let other_weights = gradient_verdict(
        &ladder,
        grads_leg("torch", &[("layer.0.Wqkv.lora_b", W, G)]),
        grads_leg("fused", &[("layer.0.Wqkv.lora_b", &[1.0, 2.0, 3.0001], G)]),
    );
    let structure = judgement(&other_weights, "gradient_structure");
    assert_eq!(structure.held, Held::Failed);
    assert!(structure.detail.contains("same weights"));
    let missing_tensor = gradient_verdict(
        &ladder,
        grads_leg("torch", &[("layer.0.Wqkv.lora_b", W, G)]),
        grads_leg("fused", &[("layer.0.Wo.lora_b", W, G)]),
    );
    assert_eq!(missing_tensor.status, Status::Red);
}

#[test]
fn an_edge_asked_for_its_outcome_without_gradient_legs_is_refused() {
    let ladder = committed_ladder(Workload::TrainStep);
    let verdict = edge_verdict(
        &ladder,
        "torch",
        "fused",
        &set(step_repeats()),
        &outcome_only(),
    );
    assert!(refused(&verdict, |r| matches!(
        r,
        Refusal::MeasurementMissing {
            measurement: "gradient_structure",
            ..
        }
    )));
    assert_eq!(verdict.status, Status::Invalid);
    // One side only is named.
    let mut legs = step_repeats();
    legs.push(grads_leg("torch", &[("layer.0.Wqkv.lora_b", W, G)]));
    let verdict = edge_verdict(&ladder, "torch", "fused", &set(legs), &outcome_only());
    assert!(refused(
        &verdict,
        |r| matches!(r, Refusal::MissingLeg { rung, take, .. } if rung == "fused" && take == "grads")
    ));
}

/// A gradient leg differs from the timed repeats on warmup, measured steps
/// and clip by its nature; on anything else it is the same experiment or
/// none.
#[test]
fn a_gradient_leg_is_free_on_the_fields_a_single_forward_has_no_use_for_and_bound_on_the_rest() {
    let ladder = committed_ladder(Workload::TrainStep);
    let tensors: &[(&str, &[f32], &[f32])] = &[("layer.0.Wqkv.lora_b", W, G)];
    let free = gradient_verdict(
        &ladder,
        grads_leg("torch", tensors),
        grads_leg("fused", tensors),
    );
    assert!(
        !refused(&free, |r| matches!(r, Refusal::IdentityDisagreement { .. })),
        "{:?}",
        free.refusals
    );
    let other_rank = step_leg(
        "fused",
        "grads",
        json!({
            "lora_rank": "other", "warmup": 0, "steps_measured": 0, "max_grad_norm": null,
            "gradients": {}
        }),
    );
    let bound = gradient_verdict(&ladder, grads_leg("torch", tensors), other_rank);
    assert!(refused(
        &bound,
        |r| matches!(r, Refusal::IdentityDisagreement { field, .. } if field == "lora_rank")
    ));
}

// ── mutant columns ─────────────────────────────────────────────────────────

const PATCH: &str = "ab12";

#[test]
fn a_dose_label_is_a_signed_dose_or_a_named_red_proof() {
    let label = |spec: &str| MutantSpec::parse(spec).map(|s| s.label);
    assert_eq!(label("eps-0.50:AB12").unwrap(), DoseLabel::Eps(-0.5));
    assert_eq!(
        MutantSpec::parse("eps-0.50: AB12 ").unwrap().patch_sha256,
        "ab12"
    );
    assert_eq!(
        label("redproof-nobc:ab").unwrap(),
        DoseLabel::RedProof("nobc".into())
    );
    for bad in [
        "eps-0.50",
        "eps-0.50:",
        "eps+0.5:ab",
        "eps 0.5:ab",
        "eps0:ab",
        "eps-1.0:ab",
        "eps1.5:ab",
        "eps0.001:ab",
        "epsnan:ab",
        "redproof-:ab",
        "redproof-  :ab",
        "dose5:ab",
    ] {
        assert!(
            matches!(label(bad), Err(Refusal::MutantColumnInvalid { .. })),
            "{bad}"
        );
    }
}

#[test]
fn two_columns_may_not_share_a_label_a_dose_or_a_patch() {
    let specs = |list: &[&str]| {
        list.iter()
            .map(|s| MutantSpec::parse(s).unwrap())
            .collect::<Vec<_>>()
    };
    assert!(mutant::duplicates(&specs(&["eps-0.5:a", "eps-0.1:b", "redproof-x:c"])).is_empty());
    assert_eq!(
        mutant::duplicates(&specs(&["eps-0.5:a", "eps-0.5:b"])).len(),
        1
    );
    assert_eq!(
        mutant::duplicates(&specs(&["eps-0.1:a", "eps-0.10:b"])).len(),
        1
    );
    assert_eq!(
        mutant::duplicates(&specs(&["eps-0.1:a", "redproof-x:A"])).len(),
        1
    );
}

fn mutant_column(d: f64, stamped_patch: &str) -> DoseColumn {
    let mut legs = seeded_legs(&alternating(0.004));
    let stamp = json!({"mutant_id": "scaled-update", "mutant_base_sha": "base", "mutant_patch_sha256": stamped_patch});
    legs.extend((1..=12).map(|seed| {
        let loss = 3.0 + 0.01 * (seed - 1) as f64 + d;
        let fields = merged(stamp.clone(), learning(loss));
        train_leg("mutant-eps-0.50", fused_facts(), seed, "r1", fields)
    }));
    let ladder = committed_ladder(Workload::TrainRun);
    let span = ladder.span(Some(TORCH), Some(FUSED)).unwrap();
    let spec = MutantSpec::parse(&format!("eps-0.50:{PATCH}")).unwrap();
    mutant::column(
        Workload::TrainRun,
        &span[0],
        &set(legs),
        &spec,
        &outcome_only(),
    )
}

#[test]
fn a_mutant_is_judged_by_the_edges_own_rules_against_the_same_lower_legs() {
    assert_eq!(mutant_column(0.1, PATCH).detected, Detection::Degradation);
    assert_eq!(mutant_column(-0.1, PATCH).detected, Detection::Improvement);
    assert_eq!(
        mutant_column(0.0, PATCH).detected,
        Detection::Invalid,
        "all ties: the sign test refuses"
    );
    let mislabelled = mutant_column(0.1, "cd34");
    assert_eq!(mislabelled.detected, Detection::Invalid);
    assert!(refused(&mislabelled.verdict, |r| matches!(
        r,
        Refusal::PremiseViolated {
            premise: "mutant_stamp",
            ..
        }
    )));
}

fn synthetic(label: &str, detected: Detection) -> DoseColumn {
    let spec = MutantSpec::parse(&format!("{label}:{label}")).unwrap();
    DoseColumn {
        label: spec.text,
        dose: spec.label,
        patch_sha256: spec.patch_sha256,
        detected,
        verdict: EdgeVerdict::conclude(
            "edge".into(),
            &Difference::Layer { name: "a defect" },
            vec![],
            (None, None, None),
            vec![],
            vec![],
        ),
    }
}

#[test]
fn sensitivity_is_read_among_deflating_doses_by_magnitude() {
    use Detection::*;
    // Run order is large dose first; the straddle is found by magnitude.
    let ladder = DoseLadder::fold(vec![
        synthetic("eps-0.50", Degradation),
        synthetic("eps-0.10", Undetected),
        synthetic("eps0.50", Degradation),
    ]);
    assert_eq!(
        ladder.sensitivity,
        Some(("eps-0.10".into(), "eps-0.50".into()))
    );
    assert_eq!(ladder.falsification.len(), 1);
    assert!(ladder.anomalies.is_empty() && ladder.causes().is_empty());

    // An inflating detection is never a sensitivity bound.
    let none = DoseLadder::fold(vec![
        synthetic("eps-0.10", Undetected),
        synthetic("eps0.50", Degradation),
    ]);
    assert_eq!(none.sensitivity, None);
}

#[test]
fn anomalies_invalid_columns_and_an_unproven_red_proof_each_fail_the_run() {
    use Detection::*;
    let causes = |columns| {
        DoseLadder::fold(columns)
            .causes()
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        causes(vec![synthetic("eps-0.10", Improvement)]),
        [Status::RedForInvestigation]
    );
    assert_eq!(
        causes(vec![synthetic("eps-0.10", Invalid)]),
        [Status::Invalid]
    );
    assert_eq!(
        causes(vec![synthetic("redproof-x", Undetected)]),
        [Status::Red]
    );
    assert_eq!(
        causes(vec![synthetic("redproof-x", Improvement)]),
        [Status::Red]
    );
    assert!(causes(vec![
        synthetic("redproof-x", Degradation),
        synthetic("redproof-y", Undetected)
    ])
    .is_empty());
    // An inflating dose that improves is the prediction confirmed, not an anomaly.
    assert!(causes(vec![synthetic("eps0.50", Improvement)]).is_empty());
}

// ── the subcommand, over files ─────────────────────────────────────────────

fn args(workload: Workload, dir: &Path) -> LadderArgs {
    LadderArgs {
        workload,
        revision: None,
        legs_dir: dir.to_owned(),
        out: None,
        from: None,
        to: None,
        axes: vec![Axis::Outcome, Axis::Speed, Axis::Space],
        waive_control: false,
        law_dir: None,
        mutants: vec![],
    }
}

fn write_leg(dir: &Path, workload: Workload, name: &str, fields: Value) {
    let report = json!({"tiers": { workload.tier_key(): block(workload, fields) }});
    std::fs::write(
        dir.join(format!("{name}.json")),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
}

fn encode_fields(work: u64, seconds: f64) -> Value {
    json!({
        "rows": work, "corpus_sha256": format!("corpus-{work}"), "token_lengths_sha256": format!("lengths-{work}"),
        "tokens": work * 8, "work": work, "outcome_digest": "d",
        "iter_wall_s": steady(seconds), "peak_rss_bytes": 1.0e9, "peak_vram_bytes": 1.0e9
    })
}

#[test]
fn the_subcommand_reads_a_directory_and_refuses_what_it_cannot_place() {
    let dir = tempfile::tempdir().unwrap();
    for rung in [PLAN, PARTITIONED, PLACED] {
        write_leg(
            dir.path(),
            Workload::Encode,
            &format!("{rung}__rows16__r1"),
            encode_fields(16, 1.0),
        );
    }
    let mut span = args(Workload::Encode, dir.path());
    span.from = Some(PLAN.into());
    span.to = Some(PLACED.into());
    let verdict = run_ladder(&span).unwrap();
    assert_eq!(
        (verdict.status, verdict.edges.len()),
        (Status::Green, 2),
        "{:?}",
        verdict.causes
    );
    assert!(verdict.table().contains("status: GREEN"));

    // A rung this ladder does not have, a control no edge declares, a file
    // that is not a leg: each is named, none is skipped.
    write_leg(
        dir.path(),
        Workload::Encode,
        "plann__rows16__r1",
        encode_fields(16, 1.0),
    );
    write_leg(
        dir.path(),
        Workload::Encode,
        &format!("{PLAN}__rows16__lr0"),
        encode_fields(16, 1.0),
    );
    std::fs::write(dir.path().join("notes.json"), b"{}").unwrap();
    let verdict = run_ladder(&span).unwrap();
    assert_eq!(verdict.status, Status::Invalid);
    let kinds: Vec<&Refusal> = verdict.refusals.iter().map(|r| &r.refusal).collect();
    assert!(kinds
        .iter()
        .any(|r| matches!(r, Refusal::UnknownRung { rung, .. } if rung == "plann")));
    assert!(kinds
        .iter()
        .any(|r| matches!(r, Refusal::UnknownTake { take, .. } if take == "lr0")));
    assert!(kinds
        .iter()
        .any(|r| matches!(r, Refusal::LegNameMalformed { .. })));
}

#[test]
fn a_rung_in_the_span_with_no_legs_is_a_missing_leg() {
    let dir = tempfile::tempdir().unwrap();
    write_leg(
        dir.path(),
        Workload::Encode,
        &format!("{PLAN}__rows16__r1"),
        encode_fields(16, 1.0),
    );
    let mut span = args(Workload::Encode, dir.path());
    (span.from, span.to) = (Some(PLAN.into()), Some(PARTITIONED.into()));
    let verdict = run_ladder(&span).unwrap();
    assert_eq!(verdict.status, Status::Invalid);
    assert!(refused(
        &verdict.edges[0],
        |r| matches!(r, Refusal::MissingLeg { rung, .. } if rung == PARTITIONED)
    ));
}

// ── revision edges: one rung, two builds ───────────────────────────────────

const DIRECT: &str = "direct";

/// The four legs of a revision session at one size — base and revised, two
/// repeats each — and, with `rebuilt`, the A/A twin of the base.
fn revision_legs(base: f64, revised: f64, rebuilt: Option<f64>) -> Vec<Leg> {
    let jitter = |seconds: f64, k: u64| -> Vec<f64> {
        (0..32)
            .map(|i| seconds * (1.0 + 0.01 * (((i + k) % 5) as f64 - 2.0)))
            .collect()
    };
    let mut legs = vec![
        encode_leg("direct@base", 16, "r1", jitter(base, 0), json!({})),
        encode_leg("direct@base", 16, "r2", jitter(base, 1), json!({})),
        encode_leg("direct@revised", 16, "r1", jitter(revised, 2), json!({})),
        encode_leg("direct@revised", 16, "r2", jitter(revised, 3), json!({})),
    ];
    if let Some(rebuilt) = rebuilt {
        legs.push(encode_leg(
            "direct@rebuilt",
            16,
            "r1",
            jitter(rebuilt, 4),
            json!({}),
        ));
        legs.push(encode_leg(
            "direct@rebuilt",
            16,
            "r2",
            jitter(rebuilt, 0),
            json!({}),
        ));
    }
    legs
}

fn revision_verdict(legs: Vec<Leg>) -> EdgeVerdict {
    let ladder = committed_ladder(Workload::Encode);
    let rules = Workload::Encode.revision_rules(DIRECT, &Budgets::committed());
    let edge = Edge::revision(ladder.rung(DIRECT).unwrap(), &rules);
    let legs = set(legs);
    let mut options = axes(true, true, true, false);
    let rebuilt = legs.rung("direct@rebuilt");
    let has_rebuilt = rebuilt.all().next().is_some();
    options.rebuilt = has_rebuilt.then_some(rebuilt);
    compare(
        Workload::Encode,
        &edge,
        &legs.rung("direct@base"),
        &legs.rung("direct@revised"),
        &options,
    )
}

#[test]
fn a_revision_inside_its_own_noise_band_is_green_and_indistinguishable_from_one() {
    let verdict = revision_verdict(revision_legs(1.0, 1.005, None));
    assert_eq!(verdict.status, Status::Green, "{:#?}", verdict.refusals);
    assert_eq!(verdict.edge, "direct@base -> direct@revised");
    assert_eq!(verdict.difference, Difference::Revision);
    let speed = verdict.speed.as_ref().unwrap();
    assert_eq!(speed.indistinguishable_from_one, Some(true));
    assert!(speed.aa_null.is_none());
    // Digests are compared on a revision edge and reported, never refused.
    let digests = judgement(&verdict, "outcome_digests_equal");
    assert_eq!(
        (digests.held, digests.force),
        (Held::Passed, RuleForce::Evidence)
    );
}

#[test]
fn a_revision_outside_the_band_fails_slower_and_is_investigated_faster() {
    let slower = revision_verdict(revision_legs(1.0, 1.2, None));
    assert_eq!(slower.status, Status::Red, "{:#?}", slower.refusals);
    let rule = judgement(&slower, "speed_within_noise_band");
    assert_eq!(
        (rule.held, rule.direction),
        (Held::Failed, Direction::Degradation)
    );
    let faster = revision_verdict(revision_legs(1.0, 0.8, None));
    assert_eq!(faster.status, Status::RedForInvestigation);
    assert_eq!(
        judgement(&faster, "speed_within_noise_band").direction,
        Direction::Improvement
    );
}

/// A second build of the same revision that lands 10% away widens the band
/// to cover it: a revision 8% away is then inside, where against the
/// repeats alone it was outside.
#[test]
fn the_aa_null_widens_the_band_to_what_two_builds_of_one_revision_differ_by() {
    let against_repeats = revision_verdict(revision_legs(1.0, 1.08, None));
    assert_eq!(against_repeats.status, Status::Red);
    let with_null = revision_verdict(revision_legs(1.0, 1.08, Some(1.10)));
    assert_eq!(with_null.status, Status::Green, "{:#?}", with_null.refusals);
    let speed = with_null.speed.unwrap();
    assert!(speed.aa_null.is_some_and(|aa| aa.of_medians > 1.05));
    assert!(speed.noise_band.is_some_and(|b| b > 1.09));
    assert_eq!(speed.indistinguishable_from_one, Some(true));
}

#[test]
fn the_subcommand_judges_a_revision_edge_from_its_side_tagged_legs() {
    let dir = tempfile::tempdir().unwrap();
    for leg in revision_legs(1.0, 1.0, Some(1.0)) {
        let name = leg.name.to_string();
        let fields = json!({"iter_wall_s": leg.measured.iter_wall_s, "work": leg.measured.work, "outcome_digest": "d"});
        write_leg(dir.path(), Workload::Encode, &name, fields);
    }
    let mut args = args(Workload::Encode, dir.path());
    args.revision = Some(DIRECT.into());
    let verdict = run_ladder(&args).unwrap();
    assert_eq!(verdict.status, Status::Green, "{:#?}", verdict);
    assert_eq!(
        (verdict.from.as_str(), verdict.to.as_str()),
        ("direct@base", "direct@revised")
    );
    assert!(verdict.edges[0].speed.as_ref().unwrap().aa_null.is_some());
}

// ── oracle: the committed A/A null runs ────────────────────────────────────

fn artifacts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ci/artifacts")
}

/// A leg whose every timed iteration is the one summary the committed
/// artifact kept: with only a median per leg on record, the ladder's cost
/// is reproducible and its interval is not — a constant series has none.
fn constant_leg(workload: Workload, name: &str, seconds: f64, fields: Value) -> Leg {
    leg(
        workload,
        name,
        merged(json!({"iter_wall_s": vec![seconds; 16]}), fields),
    )
}

/// The five committed A/A runs — the parent sha built twice, four legs
/// `a1, b1, b2, a2` — read as revision edges of `encode`'s `direct` rung
/// with the same revision on both sides. The old comparator's combined
/// ratio (the mean of two adjacent-pair ratios) is reproduced to three
/// decimals by the ladder's ratio of pooled medians; against each build's
/// own repeat band, three of the five land outside it — two builds of one
/// sha differ by more than a build's repeats do, which is what the A/A twin
/// exists to measure.
#[test]
fn the_committed_aa_null_runs_reproduce_their_ratios_and_show_build_variance() {
    let mut readings = vec![];
    for stem in [
        "2026-08-30-pcie-p1",
        "2026-08-30-pcie-p2",
        "2026-08-30-pcie-p3",
        "2026-08-30-sxm4-r1",
        "2026-08-30-sxm4-r2",
    ] {
        let path = artifacts().join(format!("gpu-perf-aa-null/{stem}.json"));
        let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let p50 = |leg: &str| {
            report["legs"][leg]["measurements"]["embed"]["p50_ms"]["value"]
                .as_f64()
                .unwrap()
                / 1000.0
        };
        let legs = vec![
            constant_leg(
                Workload::Encode,
                "direct@base__rows256__r1",
                p50("a1"),
                json!({}),
            ),
            constant_leg(
                Workload::Encode,
                "direct@base__rows256__r2",
                p50("a2"),
                json!({}),
            ),
            constant_leg(
                Workload::Encode,
                "direct@revised__rows256__r1",
                p50("b1"),
                json!({}),
            ),
            constant_leg(
                Workload::Encode,
                "direct@revised__rows256__r2",
                p50("b2"),
                json!({}),
            ),
        ];
        let verdict = revision_verdict(legs);
        let speed = verdict.speed.as_ref().unwrap();
        let old = report["combined_embed_p50_ratio"].as_f64().unwrap();
        assert!(
            (speed.cost.of_medians - old).abs() < 3e-3,
            "{stem}: {} vs the old combined ratio {old}",
            speed.cost.of_medians
        );
        readings.push((
            stem,
            speed.cost.of_medians,
            speed.noise_band.unwrap(),
            speed.indistinguishable_from_one.unwrap(),
        ));
    }
    eprintln!("A/A null readings (cost, repeat band, inside): {readings:#?}");
    let inside: Vec<&str> = readings.iter().filter(|r| r.3).map(|r| r.0).collect();
    assert_eq!(inside, ["2026-08-30-pcie-p2", "2026-08-30-pcie-p3"]);
    let worst = readings.iter().map(|r| r.1.ln().abs()).fold(0.0, f64::max);
    // The old advisory band [0.75, 1.33] was 1.5 times the worst A/A
    // deviation, floored: exp(-1.5 * 0.139) = 0.81 was not it; the
    // committed derivation took the worst over its primary runs only.
    assert!((0.13..0.14).contains(&worst), "{worst}");
}

// ── oracle: the committed train-step sweep ─────────────────────────────────

/// The committed step sweep — six shapes, the `jammi-fused`/`torch-sdpa`
/// pair twice each in A,B,B,A order — as `train-step` legs: `torch` and
/// `fused` at each shape, each leg's series the one median the merged
/// report kept, so the ladder's point ratio is reproducible and its
/// interval is not. The sweep's `jammi-eager` arm had every family off,
/// which is no rung of this ladder (it ran out of memory at four of the six
/// shapes), so it files no leg.
fn committed_step_sweep() -> (Value, Vec<Leg>) {
    let path = artifacts()
        .join("finetune-ab-runs/2026-08-30-full-sweep-acce7b3d-a100-pcie/finetune_ab_report.json");
    let report: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut legs = vec![];
    for (slug, config) in report["configs"].as_object().unwrap() {
        let unit = slug.replace('-', "");
        let metrics = |leg: &str| -> Option<&Value> {
            let entry = config["legs"]
                .get(leg)
                .or_else(|| config["bar_second_run_legs"].get(leg))?;
            (entry["outcome"] == "OK").then(|| &entry["metrics"])
        };
        let p50 = |m: &Value| m["s_per_step_p50"].as_f64().unwrap();
        let vram = |m: &Value| json!({"peak_vram_bytes": m["vram_delta_bytes"]});
        for (rung, leg, take) in [
            ("torch", "torch-sdpa", "r1"),
            ("torch", "torch-sdpa-2", "r2"),
            ("fused", "jammi-fused", "r1"),
            ("fused", "jammi-fused-2", "r2"),
        ] {
            let m = metrics(leg).unwrap();
            let mut fields = merged(vram(m), json!({"backbone_dtype": "bf16"}));
            if rung == "fused" {
                fields = merged(fields, step_facts(m, "fused"));
            }
            legs.push(constant_leg(
                Workload::TrainStep,
                &format!("{rung}__{unit}__{take}"),
                p50(m),
                fields,
            ));
        }
    }
    (report, legs)
}

/// The arm facts of a jammi step leg, from the merged report's own record
/// of its dispatch pairs and disable lists.
fn step_facts(m: &Value, arm: &str) -> Value {
    let mut facts = json!({
        "arm": arm, "attention_arm": if arm == "fused" { "fused" } else { "eager" },
        "device_name": "NVIDIA A100 80GB PCIe", "build_features": ["cuda", "flash-attn"],
        "flash_compiled": m["flash_compiled"],
        "kernels_disabled_requested": m["kernels_disabled_requested"],
        "kernels_disabled_fired": m["kernels_disabled_fired"],
    });
    // The merged report did not keep the gelu pair: never dispatched on
    // this tower, counted as zero.
    facts["gelu_fused_dispatches"] = json!(0);
    facts["gelu_eager_dispatches"] = json!(0);
    for pair in m["dispatch_pairs"].as_array().unwrap() {
        let base = pair[0].as_str().unwrap();
        let fallback = if base == "attention_block_flash" {
            "declined"
        } else {
            "eager"
        };
        facts[format!("{base}_fused_dispatches")] = pair[1].clone();
        facts[format!("{base}_{fallback}_dispatches")] = pair[2].clone();
    }
    facts
}

/// Every reading the deleted merger reached, reproduced where the ladder
/// reaches it: PASS is a cost outside the repeat noise band with the
/// non-inferiority bound met; INDETERMINATE is a cost inside the band
/// (the two repeats disagree by more than the ratio is from 1); INVALID is
/// a refusal. The pair is also read directly at every shape, as a session of
/// its own, and the two readings agree.
#[test]
fn the_committed_step_sweep_reproduces_every_configs_reading() {
    let (report, legs) = committed_step_sweep();
    let legs = set(legs);
    let ladder = committed_ladder(Workload::TrainStep);
    let (torch, fused) = (legs.rung("torch"), legs.rung("fused"));
    let units: Vec<_> = torch.measured_units().cloned().collect();
    assert_eq!(units.len(), 6);
    let pair = Pair {
        edge: "torch -> fused (direct)",
        lower: &torch,
        upper: &fused,
        units: &units,
        unclean: &Default::default(),
    };
    let mut readings = vec![];
    for unit in &units {
        let one = [unit.clone()];
        let single = Pair {
            units: &one,
            ..pair
        };
        let measured = speed::measure_cost(&single).unwrap().unwrap();
        let (cost, band) = (measured.ratio, measured.band.unwrap());
        let inside = cost.of_medians.ln().abs() <= band.ln();
        let non_inferior = 1.0 / cost.of_medians > 0.9;
        let ladder_reading = if inside {
            "INDETERMINATE"
        } else if non_inferior {
            "PASS"
        } else {
            "FAIL"
        };
        let slug = report["configs"]
            .as_object()
            .unwrap()
            .keys()
            .find(|k| k.replace('-', "") == unit.as_str())
            .unwrap();
        let committed = report["configs"][slug]["verdict"]
            .as_str()
            .unwrap()
            .split(' ')
            .next()
            .unwrap()
            .to_owned();
        readings.push((
            slug.clone(),
            cost.of_medians,
            band,
            ladder_reading,
            committed,
        ));
    }
    eprintln!("step sweep readings (cost, repeat band, ladder, committed): {readings:#?}");
    for (slug, _, _, ladder_reading, committed) in &readings {
        assert_eq!(ladder_reading, committed, "{slug}");
    }
    assert_eq!(
        readings.iter().filter(|r| r.3 == "INDETERMINATE").count(),
        2
    );

    // The ladder over every shape: the one edge, torch to fused, read on
    // every unit. The committed sweep carries no gradient legs, so the
    // outcome axis is left out.
    let cost_axes = axes(false, true, true, false);
    let verdict = edge_verdict(&ladder, "torch", "fused", &legs, &cost_axes);
    assert!(verdict.refusals.is_empty(), "{:#?}", verdict.refusals);
    assert_eq!(verdict.units.len(), 6);
    let speed = verdict.speed.as_ref().unwrap();
    eprintln!(
        "torch -> fused: cost {:.4}, host {:?}, device {:?}",
        speed.cost.of_medians,
        verdict.space.as_ref().unwrap().host_ratio,
        verdict.space.as_ref().unwrap().device_ratio
    );
    assert_ne!(verdict.status, Status::Invalid);
}
