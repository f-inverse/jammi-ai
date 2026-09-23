//! The parity ladder: is a workload run through the engine's whole stack as
//! fast, as small, and as good as the same workload in a reference
//! framework, and what does each layer of the stack cost?
//!
//! A *workload* is a function from committed inputs to an artifact. A *rung*
//! is one implementation stack of it; rungs are ordered so adjacent rungs
//! differ by one layer. A *leg* is one run of one rung. An *edge* is an
//! adjacent pair of rungs, and one operator — [`compare::compare`] — gives
//! every edge a verdict on three axes: speed, space, outcome. A new layer in
//! the engine is a new rung in [`definition`], never a new comparison.
//!
//! Because adjacent rungs differ by one layer, an edge's cost *is* that
//! layer's cost and the costs telescope: their product is the end-to-end
//! cost, which the ladder also measures directly. A product that disagrees
//! with the direct measurement is a finding about the measurement, and the
//! verdict is refused.
//!
//! Legs live in one directory, `<rung>__<unit>__<take>.json`; the legs of
//! the directly measured end-to-end pair, a session of their own, live in
//! its `direct/` subdirectory. Producers emit legs and decide nothing.
//!
//! What differs between the two legs of an edge is one typed thing, a
//! [`definition::Difference`]: a reference framework, a kernel arm, an engine
//! layer, or — for a rung compared against itself across two builds
//! (`--revision`) — a revision of the engine.

pub mod compare;
pub mod definition;
pub mod leg;
pub mod mutant;
pub mod outcome;
pub mod premise;
pub mod refusal;
pub mod space;
pub mod speed;
pub mod verdict;

#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use jammi_numerics::stats::Interval;

use compare::{compare, Axes, CompareOptions};
use definition::{Budgets, CrossStackOutcome, Edge, EdgeKind, RevisionRules, Workload};
use leg::{LegSet, Take};
use mutant::{DoseLadder, MutantSpec};
use outcome::{CrossStackOptions, Pair};
use refusal::Refusal;
use verdict::{LadderVerdict, Status, Telescoping};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Axis {
    Outcome,
    Speed,
    Space,
    /// Fit speed and space against size; needs legs at three sizes or more.
    Shape,
}

/// `jammi-bench ladder`'s flags.
#[derive(Debug, clap::Args)]
pub struct LadderArgs {
    /// The workload whose ladder is compared.
    #[arg(value_enum)]
    pub workload: Workload,
    /// Directory of leg files, `<rung>__<unit>__<take>.json`.
    pub legs_dir: PathBuf,
    /// Write `ladder_verdict.json` and `ladder_table.txt` here instead of
    /// printing the verdict.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Lowest rung compared; the ladder's reference rung when omitted.
    #[arg(long)]
    pub from: Option<String>,
    /// Highest rung compared; the ladder's top rung when omitted.
    #[arg(long)]
    pub to: Option<String>,
    /// Axes to measure. An axis left out is not judged.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "outcome,speed,space,shape"
    )]
    pub axes: Vec<Axis>,
    /// State that an edge's control legs were not run. Recorded in the
    /// verdict; without it a missing control is refused.
    #[arg(long)]
    pub waive_control: bool,
    /// Directory holding a law workload's ground truth, `<unit>.json`.
    #[arg(long)]
    pub law_dir: Option<PathBuf>,
    /// A mutant column, `LABEL:PATCH_SHA256`, whose legs are filed under the
    /// rung name `mutant-LABEL`. It stands in for the upper rung of the
    /// highest cross-stack edge compared. Repeatable.
    #[arg(long = "mutant")]
    pub mutants: Vec<String>,
    /// Compare one rung against itself built from another revision: legs
    /// filed under `RUNG@base` and `RUNG@revised`, with a second build of
    /// the base under `RUNG@rebuilt` as the edge's own A/A null. The cost
    /// is judged against the wider of the repeat noise band and the A/A
    /// band. Excludes `--from`/`--to`.
    #[arg(long, conflicts_with_all = ["from", "to", "mutants"])]
    pub revision: Option<String>,
}

/// Legs that belong to nothing being compared: a rung this ladder does not
/// have, or a take no edge at that rung declares. Ignoring either would let
/// a misspelt file name remove a leg from the comparison silently.
fn strays(legs: &LegSet, span: &[Edge<'_>], mutant_rungs: &[String]) -> Vec<Refusal> {
    let known: Vec<String> = span
        .iter()
        .flat_map(|edge| {
            [
                Some(edge.lower_name()),
                Some(edge.upper_name()),
                edge.rebuilt_name(),
            ]
            .into_iter()
            .flatten()
        })
        .collect();
    let unknown = legs
        .rung_names()
        .filter(|name| {
            !known
                .iter()
                .chain(mutant_rungs)
                .any(|k| k.as_str() == *name)
        })
        .map(|name| Refusal::UnknownRung {
            rung: name.to_owned(),
            known: known.clone(),
        });
    let declared = |rung: &str, tag: &str| {
        span.iter().any(|edge| {
            let touches = edge.lower().name == rung || edge.upper().name == rung;
            let declares = matches!(
                edge.kind(),
                EdgeKind::CrossStack(rules) if match &rules.outcome {
                    CrossStackOutcome::SeededLoss { control: Some(c), .. } => c.take == tag,
                    CrossStackOutcome::GradientAgreement { take, .. } => *take == tag,
                    _ => false,
                }
            );
            touches && declares
        })
    };
    let undeclared = known
        .iter()
        .flat_map(|rung| legs.rung(rung).controls().cloned().collect::<Vec<_>>())
        .filter_map(|leg| match &leg.name.take {
            Take::Control(tag) if !declared(&leg.name.rung, tag) => Some(Refusal::UnknownTake {
                leg: leg.name.to_string(),
                take: tag.clone(),
            }),
            _ => None,
        });
    unknown.chain(undeclared).collect()
}

/// The product of the edge costs against the cost of the end-to-end pair
/// measured directly, in a session of its own.
fn telescoping(
    edges: &[verdict::EdgeVerdict],
    span: &[Edge<'_>],
    direct: &LegSet,
) -> Result<Option<Telescoping>, Vec<Refusal>> {
    let costs: Option<Vec<verdict::Ratio>> = edges
        .iter()
        .map(|e| e.speed.as_ref().map(|s| s.cost))
        .collect();
    let (Some(costs), [first, .., last]) = (costs, span) else {
        return Ok(None);
    };
    let (lower, upper) = (
        direct.rung(&first.lower_name()),
        direct.rung(&last.upper_name()),
    );
    let units: Vec<_> = lower
        .measured_units()
        .filter(|u| upper.primary(u).is_some())
        .cloned()
        .collect();
    let name = format!("{} -> {} (direct)", first.lower_name(), last.upper_name());
    let unclean = Default::default();
    let pair = Pair {
        edge: &name,
        lower: &lower,
        upper: &upper,
        units: &units,
        unclean: &unclean,
    };
    let Some((direct_cost, _)) = speed::measure_cost(&pair)? else {
        return Ok(None);
    };
    // Bounds multiplied: at least as wide as the product's own interval, so
    // two intervals this finds disjoint are disjoint.
    let product = costs.iter().fold(
        Interval {
            lower: 1.0,
            upper: 1.0,
        },
        |acc, c| Interval {
            lower: acc.lower * c.interval.lower,
            upper: acc.upper * c.interval.upper,
        },
    );
    if product.upper < direct_cost.interval.lower || direct_cost.interval.upper < product.lower {
        return Err(vec![Refusal::TelescopingContradiction {
            product,
            direct: direct_cost.interval,
        }]);
    }
    Ok(Some(Telescoping {
        product,
        direct: direct_cost,
    }))
}

/// Compare a span of a workload's ladder over the legs in `legs_dir`.
pub fn run_ladder(args: &LadderArgs) -> std::io::Result<LadderVerdict> {
    let budgets = Budgets::committed();
    let ladder = args.workload.ladder_with(&budgets);
    let revision_rules: Option<RevisionRules> = args
        .revision
        .as_deref()
        .map(|rung| args.workload.revision_rules(rung, &budgets));
    let mut refusals = vec![];
    let span: Vec<Edge<'_>> = match (&args.revision, &revision_rules) {
        (Some(rung), Some(rules)) => ladder
            .rung(rung)
            .map(|rung| vec![Edge::revision(rung, rules)])
            .unwrap_or_else(|| {
                refusals.push(Refusal::UnknownRung {
                    rung: rung.clone(),
                    known: ladder.rungs().map(|r| r.name.clone()).collect(),
                });
                vec![]
            }),
        _ => ladder
            .span(args.from.as_deref(), args.to.as_deref())
            .unwrap_or_else(|refusal| {
                refusals.push(refusal);
                vec![]
            }),
    };
    let (from, to) = match (span.first(), span.last()) {
        (Some(first), Some(last)) => (first.lower_name(), last.upper_name()),
        _ => (
            args.from.clone().unwrap_or_default(),
            args.to.clone().unwrap_or_default(),
        ),
    };

    let specs: Vec<MutantSpec> = args
        .mutants
        .iter()
        .filter_map(|spec| MutantSpec::parse(spec).map_err(|r| refusals.push(r)).ok())
        .collect();
    refusals.extend(mutant::duplicates(&specs));

    let mut legs = LegSet::read(args.workload, &args.legs_dir)?;
    let mut direct = LegSet::read(args.workload, &args.legs_dir.join("direct"))?;
    refusals.append(&mut legs.unreadable);
    refusals.append(&mut direct.unreadable);
    let mutant_rungs: Vec<String> = specs.iter().map(MutantSpec::rung_name).collect();
    refusals.extend(strays(&legs, &span, &mutant_rungs));

    let has = |axis| args.axes.contains(&axis);
    let mut options = CompareOptions::new(
        Axes {
            outcome: has(Axis::Outcome),
            speed: has(Axis::Speed),
            space: has(Axis::Space),
            shape: has(Axis::Shape),
        },
        CrossStackOptions {
            waive_control: args.waive_control,
            law_dir: args.law_dir.clone(),
        },
    );
    options.rebuilt = span
        .first()
        .and_then(Edge::rebuilt_name)
        .map(|name| legs.rung(&name))
        .filter(|rebuilt| rebuilt.all().next().is_some());
    let edges: Vec<verdict::EdgeVerdict> = span
        .iter()
        .map(|edge| {
            compare(
                ladder.workload,
                edge,
                &legs.rung(&edge.lower_name()),
                &legs.rung(&edge.upper_name()),
                &options,
            )
        })
        .collect();

    let telescoping = telescoping(&edges, &span, &direct).unwrap_or_else(|found| {
        refusals.extend(found);
        None
    });

    let mutated_edge = span
        .iter()
        .rev()
        .find(|e| matches!(e.kind(), EdgeKind::CrossStack(_)));
    let dose_ladder = match (specs.is_empty(), mutated_edge) {
        (true, _) => None,
        (false, Some(edge)) => Some(DoseLadder::fold(
            specs
                .iter()
                .map(|spec| mutant::column(ladder.workload, edge, &legs, spec, &options))
                .collect(),
        )),
        (false, None) => {
            refusals.push(Refusal::MutantColumnInvalid {
                label: "*".to_owned(),
                reason: "no cross-stack edge is being compared for a mutant to stand in on"
                    .to_owned(),
            });
            None
        }
    };

    let mut causes: Vec<(Status, String)> = edges
        .iter()
        .filter(|e| e.status != Status::Green)
        .map(|e| {
            (
                e.status,
                format!("edge {} is {}", e.edge, verdict::serde_plain(&e.status)),
            )
        })
        .collect();
    causes.extend(refusals.iter().map(|r| (Status::Invalid, r.to_string())));
    causes.extend(dose_ladder.iter().flat_map(DoseLadder::causes));
    let status = causes
        .iter()
        .map(|(s, _)| *s)
        .max()
        .unwrap_or(Status::Green);

    Ok(LadderVerdict {
        workload: args.workload,
        from,
        to,
        edges,
        telescoping,
        dose_ladder,
        refusals: refusals.into_iter().map(Into::into).collect(),
        causes: causes.into_iter().map(|(_, cause)| cause).collect(),
        status,
    })
}

const VERDICT_FILE: &str = "ladder_verdict.json";
const TABLE_FILE: &str = "ladder_table.txt";

fn write(out: &Path, json: &str, table: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(out)?;
    std::fs::write(out.join(VERDICT_FILE), json)?;
    std::fs::write(out.join(TABLE_FILE), format!("{table}\n"))
}

/// The `ladder` subcommand: one JSON verdict and a table. Exits non-zero on
/// a refusal or a failed hard rule; an evidence rule only reports.
pub fn run(args: &LadderArgs) -> std::process::ExitCode {
    let emitted = run_ladder(args).and_then(|verdict| {
        let json = serde_json::to_string_pretty(&verdict).map_err(std::io::Error::other)?;
        let table = verdict.table();
        match &args.out {
            Some(out) => {
                write(out, &json, &table)?;
                println!("{table}");
            }
            None => {
                println!("{json}");
                eprintln!("{table}");
            }
        }
        Ok(verdict)
    });
    match emitted {
        Ok(verdict) => verdict.exit_code(),
        Err(e) => {
            eprintln!("ladder: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
