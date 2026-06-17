//! Pure-`RecordBatch` tests for the as-of merge core — no catalog, no session.
//!
//! Each test builds two pre-sorted batches by hand and drives
//! [`merge_partition`] directly through the same [`SortedPartition`] resolution
//! and [`AsofJoinExec`]-shaped projection the operator uses, so the merge's
//! semantics are pinned independent of the planning/IO machinery: the four knobs
//! (direction, boundary, tolerance, tie-break) each shown changing the result,
//! the null/preservation rules (§5.4), the empty-`by` global group, and the
//! bit-reproducible determinism the tie-break guarantees.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use jammi_ai::pipeline::asof::merge::{merge_partition, output_schema, SortedPartition};
use jammi_ai::pipeline::asof::spec::{AsofError, MatchDirection};
use jammi_ai::pipeline::asof::{AsofJoinSpecBuilder, AsofKey, Boundary, TieBreak, Tolerance};

/// A spine batch: `sym` (by) + `t` (time, integer instant) + `row` (a probe
/// column carried through unchanged so the assertion can name each left row).
fn spine(rows: &[(&str, i64, &str)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("t", DataType::Int64, true),
        Field::new("row", DataType::Utf8, false),
    ]));
    let sym = StringArray::from_iter(rows.iter().map(|r| Some(r.0)));
    let t = Int64Array::from_iter(rows.iter().map(|r| Some(r.1)));
    let row = StringArray::from_iter_values(rows.iter().map(|r| r.2));
    RecordBatch::try_new(schema, vec![Arc::new(sym), Arc::new(t), Arc::new(row)]).unwrap()
}

/// A spine batch with nullable time, for the null-preservation case.
fn spine_nullable_time(rows: &[(&str, Option<i64>, &str)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("t", DataType::Int64, true),
        Field::new("row", DataType::Utf8, false),
    ]));
    let sym = StringArray::from_iter(rows.iter().map(|r| Some(r.0)));
    let t = Int64Array::from_iter(rows.iter().map(|r| r.1));
    let row = StringArray::from_iter_values(rows.iter().map(|r| r.2));
    RecordBatch::try_new(schema, vec![Arc::new(sym), Arc::new(t), Arc::new(row)]).unwrap()
}

/// A facts batch: `sym` (by) + `qt` (time) + `val` (the projected fact column).
/// `qt` nullable so the null-fact case can pass `None`.
fn facts(rows: &[(&str, Option<i64>, &str)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("qt", DataType::Int64, true),
        Field::new("val", DataType::Utf8, true),
    ]));
    let sym = StringArray::from_iter(rows.iter().map(|r| Some(r.0)));
    let qt = Int64Array::from_iter(rows.iter().map(|r| r.1));
    let val = StringArray::from_iter(rows.iter().map(|r| Some(r.2)));
    RecordBatch::try_new(schema, vec![Arc::new(sym), Arc::new(qt), Arc::new(val)]).unwrap()
}

/// Build a sym-keyed spec over (`t`, `qt`) projecting `val`, with the four knobs.
fn sym_spec(
    direction: MatchDirection,
    boundary: Boundary,
    tolerance: Option<Tolerance>,
    tie_break: TieBreak,
) -> jammi_ai::pipeline::asof::AsofJoinSpec {
    AsofJoinSpecBuilder::new(
        AsofKey {
            by: vec!["sym".into()],
            time: "t".into(),
        },
        AsofKey {
            by: vec!["sym".into()],
            time: "qt".into(),
        },
    )
    .direction(direction)
    .boundary(boundary)
    .tolerance(tolerance)
    .tie_break(tie_break)
    .project(vec!["val".into()])
    .build()
}

/// Resolve, project, and run the merge over two pre-sorted batches against a
/// caller-built spec — the single seam every case routes through. The tie-break
/// column lives only on the facts side (the spine resolves none), matching the
/// operator's left/right asymmetry; `val` is the single projected fact column.
fn run_spec(
    left: RecordBatch,
    right: RecordBatch,
    spec: &jammi_ai::pipeline::asof::AsofJoinSpec,
) -> Result<RecordBatch, AsofError> {
    let tie = match &spec.tie_break {
        TieBreak::ByColumnDesc(c) => Some(c.as_str()),
        TieBreak::Error => None,
    };
    let left_part =
        SortedPartition::resolve(left.clone(), &spec.left.by, &spec.left.time, None, "left")?;
    let right_part = SortedPartition::resolve(
        right.clone(),
        &spec.right.by,
        &spec.right.time,
        tie,
        "right",
    )?;
    let project = vec![right.schema().index_of("val").unwrap()];
    let out = output_schema(&left.schema(), &right.schema(), &project);
    merge_partition(&left_part, &right_part, spec, &project, &out)
}

/// The common case: a sym-keyed merge with the four knobs.
fn run_merge(
    left: RecordBatch,
    right: RecordBatch,
    direction: MatchDirection,
    boundary: Boundary,
    tolerance: Option<Tolerance>,
    tie_break: TieBreak,
) -> Result<RecordBatch, AsofError> {
    let spec = sym_spec(direction, boundary, tolerance, tie_break);
    run_spec(left, right, &spec)
}

/// The projected `val` column of the joined batch, as `Option<&str>` per row.
fn matched_vals(batch: &RecordBatch) -> Vec<Option<String>> {
    let col = batch
        .column(batch.schema().index_of("val").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    (0..col.len())
        .map(|i| (!col.is_null(i)).then(|| col.value(i).to_string()))
        .collect()
}

// ── Exit-crit #1: backward-inclusive correctness, left preserved ─────────────

#[test]
fn backward_inclusive_matches_most_recent_at_or_before() {
    // One group `a`: facts at 10, 20, 30. A spine at 25 matches the 20 fact;
    // a spine at 30 matches the 30 fact (inclusive); a spine at 5 (before any
    // fact) is preserved unmatched.
    let left = spine(&[("a", 5, "L0"), ("a", 25, "L1"), ("a", 30, "L2")]);
    let right = facts(&[
        ("a", Some(10), "f10"),
        ("a", Some(20), "f20"),
        ("a", Some(30), "f30"),
    ]);
    let out = run_merge(
        left.clone(),
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(
        out.num_rows(),
        3,
        "every spine row is preserved (left outer)"
    );
    assert_eq!(
        matched_vals(&out),
        vec![None, Some("f20".into()), Some("f30".into())]
    );
}

// ── Exit-crit #2: boundary distinguishes coincident facts ────────────────────

#[test]
fn boundary_inclusive_vs_exclusive_differs_on_coincident_fact() {
    // A spine at exactly 20 with a fact at exactly 20: Inclusive matches it,
    // Exclusive falls back to the prior fact (10).
    let left = spine(&[("a", 20, "L0")]);
    let right = facts(&[("a", Some(10), "f10"), ("a", Some(20), "f20")]);

    let inc = run_merge(
        left.clone(),
        right.clone(),
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(matched_vals(&inc), vec![Some("f20".into())]);

    let exc = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Exclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(
        matched_vals(&exc),
        vec![Some("f10".into())],
        "exclusive excludes the fact stamped at the spine instant"
    );
}

// ── Exit-crit #3: tolerance suppresses stale matches (Duration + Steps) ───────

#[test]
fn tolerance_suppresses_stale_match_just_outside_and_keeps_just_inside() {
    // Fact at 10; spine at 16 with a tolerance of 5 steps → 16-10=6 > 5 → no
    // match. Spine at 15 → 15-10=5 ≤ 5 → matches.
    let right = facts(&[("a", Some(10), "f10")]);

    let outside = run_merge(
        spine(&[("a", 16, "L0")]),
        right.clone(),
        MatchDirection::Backward,
        Boundary::Inclusive,
        Some(Tolerance::Steps(5)),
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(
        matched_vals(&outside),
        vec![None],
        "stale beyond tolerance is null"
    );

    let inside = run_merge(
        spine(&[("a", 15, "L0")]),
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        Some(Tolerance::Steps(5)),
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(matched_vals(&inside), vec![Some("f10".into())]);
}

#[test]
fn tolerance_duration_on_timestamp_key() {
    // The same tolerance gate over a real Timestamp(Microsecond) temporal key,
    // proving the Duration unit drives the microsecond-widened comparison.
    let left_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("t", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("row", DataType::Utf8, false),
    ]));
    let right_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("qt", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("val", DataType::Utf8, true),
    ]));
    // Fact at t=1_000_000µs; spine at 1_006_000µs; tolerance 5_000µs → 6_000 > 5_000 → null.
    let left = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_006_000])),
            Arc::new(StringArray::from(vec!["L0"])),
        ],
    )
    .unwrap();
    let right = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_000_000])),
            Arc::new(StringArray::from(vec!["f"])),
        ],
    )
    .unwrap();
    let out = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        Some(Tolerance::Duration(5_000)),
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(matched_vals(&out), vec![None]);
}

// ── Exit-crit #4: ambiguity is loud, tie-break is deterministic ──────────────

#[test]
fn duplicate_facts_with_error_tiebreak_fail_loud() {
    // Two facts at the same instant 20 in group `a`, spine at 25, no tie-break
    // column → AmbiguousMatch, never a silent pick.
    let left = spine(&[("a", 25, "L0")]);
    let right = facts(&[("a", Some(20), "f20a"), ("a", Some(20), "f20b")]);
    let err = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap_err();
    assert!(matches!(err, AsofError::AmbiguousMatch), "got {err:?}");
}

#[test]
fn tiebreak_by_column_picks_newest_deterministically() {
    // Two facts at instant 20 disambiguated by `seq` (ascending sort places the
    // max seq last → the merge takes it). The result is bit-reproducible:
    // running twice yields identical batches.
    let left_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("t", DataType::Int64, true),
        Field::new("row", DataType::Utf8, false),
    ]));
    let right_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("qt", DataType::Int64, true),
        Field::new("seq", DataType::Int64, true),
        Field::new("val", DataType::Utf8, true),
    ]));
    let left = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(StringArray::from(vec!["a"])),
            Arc::new(Int64Array::from(vec![25])),
            Arc::new(StringArray::from(vec!["L0"])),
        ],
    )
    .unwrap();
    // Already sorted ascending by (sym, qt, seq): seq 1 then 2 at the same qt.
    let right = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a"])),
            Arc::new(Int64Array::from(vec![20, 20])),
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["old", "new"])),
        ],
    )
    .unwrap();

    let spec = AsofJoinSpecBuilder::new(
        AsofKey {
            by: vec!["sym".into()],
            time: "t".into(),
        },
        AsofKey {
            by: vec!["sym".into()],
            time: "qt".into(),
        },
    )
    .tie_break(TieBreak::ByColumnDesc("seq".into()))
    .project(vec!["val".into()])
    .build();

    // Run twice through the shared seam (which resolves the tie-break on the
    // facts side only) and assert the two batches are bit-identical.
    let a = run_spec(left.clone(), right.clone(), &spec).unwrap();
    let b = run_spec(left, right, &spec).unwrap();
    assert_eq!(
        matched_vals(&a),
        vec![Some("new".into())],
        "newest seq wins"
    );
    assert_eq!(a, b, "the tie-break makes the output bit-reproducible");
}

// ── Exit-crit #5: null + preservation ────────────────────────────────────────

#[test]
fn null_time_spine_preserved_null_time_fact_never_matched() {
    // A null-time spine row is preserved with a null match; a null-time fact is
    // never a candidate (the spine at 25 falls back to the real fact at 10).
    let left = spine_nullable_time(&[("a", None, "L_nulltime"), ("a", Some(25), "L1")]);
    let right = facts(&[("a", Some(10), "f10"), ("a", None, "f_nulltime")]);
    let out = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 2, "spine never shrinks");
    assert_eq!(
        matched_vals(&out),
        vec![None, Some("f10".into())],
        "null-time spine is null; null-time fact is not a candidate"
    );
}

#[test]
fn null_by_key_never_matches_another_null() {
    // SQL NULL ≠ NULL: a null `sym` on the spine never matches a null `sym` fact.
    let left_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("t", DataType::Int64, true),
        Field::new("row", DataType::Utf8, false),
    ]));
    let right_schema: SchemaRef = Arc::new(Schema::new(vec![
        Field::new("sym", DataType::Utf8, true),
        Field::new("qt", DataType::Int64, true),
        Field::new("val", DataType::Utf8, true),
    ]));
    let left = RecordBatch::try_new(
        Arc::clone(&left_schema),
        vec![
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(Int64Array::from(vec![25])),
            Arc::new(StringArray::from(vec!["L0"])),
        ],
    )
    .unwrap();
    let right = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![
            Arc::new(StringArray::from(vec![None::<&str>])),
            Arc::new(Int64Array::from(vec![10])),
            Arc::new(StringArray::from(vec!["f"])),
        ],
    )
    .unwrap();
    let out = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(
        matched_vals(&out),
        vec![None],
        "a null by-key matches nothing, but the spine row is preserved"
    );
}

// ── Forward + Nearest directions ─────────────────────────────────────────────

#[test]
fn forward_matches_first_at_or_after() {
    // Spine at 15 takes the first fact at/after → 20; spine at 30 (after the
    // last fact) is unmatched.
    let left = spine(&[("a", 15, "L0"), ("a", 30, "L1")]);
    let right = facts(&[("a", Some(10), "f10"), ("a", Some(20), "f20")]);
    let out = run_merge(
        left,
        right,
        MatchDirection::Forward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(matched_vals(&out), vec![Some("f20".into()), None]);
}

#[test]
fn nearest_picks_closest_ties_toward_past() {
    // Spine at 15, facts at 10 and 20 are equidistant (5 each) → past (10) wins.
    // Spine at 18 → 20 is closer (2 vs 8) → 20.
    let left = spine(&[("a", 15, "L0"), ("a", 18, "L1")]);
    let right = facts(&[("a", Some(10), "f10"), ("a", Some(20), "f20")]);
    let out = run_merge(
        left,
        right,
        MatchDirection::Nearest,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(
        matched_vals(&out),
        vec![Some("f10".into()), Some("f20".into())],
        "equidistant resolves to the past; otherwise the nearer fact"
    );
}

// ── Multiple groups + empty-by global group ──────────────────────────────────

#[test]
fn groups_do_not_bleed_across_by_boundaries() {
    // Two groups a/b; b's facts must never match a's spine and vice versa.
    let left = spine(&[("a", 25, "La"), ("b", 25, "Lb")]);
    let right = facts(&[
        ("a", Some(20), "a20"),
        ("b", Some(10), "b10"),
        ("b", Some(30), "b30"),
    ]);
    let out = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    assert_eq!(
        matched_vals(&out),
        vec![Some("a20".into()), Some("b10".into())],
        "each group matches only its own facts"
    );
}

#[test]
fn empty_by_is_one_global_group() {
    // No equality key: a single global calendar of facts; every spine row matches
    // against all facts. Spine at 25 → fact at 20 regardless of `sym`.
    let left = spine(&[("a", 25, "L0"), ("b", 25, "L1")]);
    let right = facts(&[("z", Some(10), "g10"), ("y", Some(20), "g20")]);
    let spec = AsofJoinSpecBuilder::new(
        AsofKey {
            by: vec![],
            time: "t".into(),
        },
        AsofKey {
            by: vec![],
            time: "qt".into(),
        },
    )
    .project(vec!["val".into()])
    .build();
    let out = run_spec(left, right, &spec).unwrap();
    assert_eq!(
        matched_vals(&out),
        vec![Some("g20".into()), Some("g20".into())],
        "an empty by matches across the whole relation"
    );
}

// ── Output shape: left columns ride through, projection gathered nullable ─────

#[test]
fn output_schema_is_left_then_nullable_right_projection() {
    let left = spine(&[("a", 25, "L0")]);
    let right = facts(&[("a", Some(20), "f20")]);
    let out = run_merge(
        left,
        right,
        MatchDirection::Backward,
        Boundary::Inclusive,
        None,
        TieBreak::Error,
    )
    .unwrap();
    // Left columns sym/t/row, then the projected nullable `val`.
    let fields = out.schema();
    assert_eq!(
        fields
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        vec!["sym", "t", "row", "val"]
    );
    assert!(
        fields.field_with_name("val").unwrap().is_nullable(),
        "the projected fact column is nullable (left-outer)"
    );
    // The carried-through left probe column is intact.
    let row = out
        .column(out.schema().index_of("row").unwrap())
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(row.value(0), "L0");
}
