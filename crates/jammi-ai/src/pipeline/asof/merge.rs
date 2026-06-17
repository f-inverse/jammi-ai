//! The per-group sort-merge core of the as-of join.
//!
//! [`merge_partition`] consumes two whole sides — each already sorted by
//! (`by`..., `time`) and, when the tie-break is a secondary column, by that
//! column too — and emits one output row per left row: the left row followed by
//! the matched fact's projected columns (or nulls when nothing matched). The
//! merge advances a single fact cursor per group, so it never backtracks and is
//! O(n + m) per group after the sort.
//!
//! Everything reduces to two scalar comparisons the temporal-key type defines
//! once: the `by`-tuple equality that bounds a group (via Arrow's row encoding,
//! one comparison for every key arity and type) and the temporal "at or before"
//! ordering (every timestamp/date/integer width widened losslessly into `i128`,
//! one comparison for every temporal type). There are no per-type or per-key
//! branches in the merge body — `MatchDirection` parameterises one cursor.

use std::sync::Arc;

use arrow::array::{new_null_array, Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use arrow::row::{RowConverter, Rows, SortField};

use super::spec::{AsofError, AsofJoinSpec, Boundary, MatchDirection, TieBreak, Tolerance};

/// Lift an Arrow error into [`AsofError`] through the DataFusion arm. The
/// DataFusion 52 `ArrowError` variant boxes the inner error, so the box happens
/// here once rather than at every call site.
fn arrow_err(e: arrow::error::ArrowError) -> AsofError {
    AsofError::DataFusion(datafusion::error::DataFusionError::ArrowError(
        Box::new(e),
        None,
    ))
}

/// One side's data, already sorted into the merge's required order: ascending by
/// (`by`..., `time`), and — when the tie-break names a secondary column —
/// ascending by that column last so the maximal tie-break value within a
/// coincident-time run is the final row of the run.
///
/// Carries the concatenated batch plus the resolved column roles as indices, so
/// the merge reads typed values without re-resolving names per row.
pub struct SortedPartition {
    /// The sorted rows of this side.
    pub batch: RecordBatch,
    /// Column indices of the equality (`by`) keys, in declared order.
    pub by_indices: Vec<usize>,
    /// Column index of the temporal ordering key.
    pub time_index: usize,
    /// Column index of the tie-break secondary key, when one is declared.
    pub tie_break_index: Option<usize>,
}

impl SortedPartition {
    /// Resolve a spec side's column roles against `batch`'s schema into a
    /// [`SortedPartition`]. The names are validated up front by
    /// [`AsofJoinSpec::validate_against`]; an unresolved name here is a typed
    /// [`AsofError::MissingByKey`] rather than a panic.
    pub fn resolve(
        batch: RecordBatch,
        by: &[String],
        time: &str,
        tie_break: Option<&str>,
        side: &'static str,
    ) -> Result<Self, AsofError> {
        let schema = batch.schema();
        let index_of = |name: &str| {
            schema.index_of(name).map_err(|_| AsofError::MissingByKey {
                column: name.to_string(),
                side,
            })
        };
        let by_indices = by.iter().map(|c| index_of(c)).collect::<Result<_, _>>()?;
        let time_index = index_of(time)?;
        let tie_break_index = tie_break.map(index_of).transpose()?;
        Ok(Self {
            batch,
            by_indices,
            time_index,
            tie_break_index,
        })
    }
}

/// The output schema of the join: the spine schema verbatim followed by the
/// projected fact columns, every fact column made nullable (left-outer — an
/// unmatched spine row carries nulls there).
pub fn output_schema(
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    project: &[usize],
) -> SchemaRef {
    let mut fields: Vec<Arc<Field>> = left_schema.fields().iter().cloned().collect();
    for &idx in project {
        let f = right_schema.field(idx);
        fields.push(Arc::new(Field::new(f.name(), f.data_type().clone(), true)));
    }
    Arc::new(Schema::new(fields))
}

/// Merge one already-sorted left side against one already-sorted right side into
/// the joined batch. The two partitions span one hash partition, which may hold
/// several whole `by`-groups; the merge resets its fact cursor at each group
/// boundary, so groups never bleed into one another.
///
/// `project` is the resolved right-column indices to attach (already validated
/// non-empty-means-all by the caller). The temporal comparison and the group
/// boundary are the only data-dependent decisions, and each is one call.
pub fn merge_partition(
    left: &SortedPartition,
    right: &SortedPartition,
    spec: &AsofJoinSpec,
    project: &[usize],
    out_schema: &SchemaRef,
) -> Result<RecordBatch, AsofError> {
    let left_n = left.batch.num_rows();

    // The `by`-tuple equality is one comparison for every key arity/type: encode
    // each side's key columns into Arrow's row format and compare the encoded
    // rows. A null key compares unequal to everything (SQL `NULL ≠ NULL`) — the
    // encoder is configured so a null sorts distinctly and the explicit
    // null-aware equality below never treats two nulls as a match.
    let group_keys = GroupKeys::new(left, right)?;

    // The temporal keys, widened losslessly into `i128` (every timestamp/date/
    // integer width fits). `None` is a null instant: a null-time left row is
    // preserved with a null match, a null-time right row is never a candidate.
    let left_time = temporal_i128(left.batch.column(left.time_index))?;
    let right_time = temporal_i128(right.batch.column(right.time_index))?;
    let tolerance = spec.tolerance;

    // One advancing pointer per side; `matched[i]` is the right row matched to
    // left row `i`, or `None`. The cursor never backtracks within a group.
    let mut matched: Vec<Option<usize>> = Vec::with_capacity(left_n);
    let right_n = right.batch.num_rows();
    let mut r = 0usize;

    let mut l = 0usize;
    while l < left_n {
        // Bound the current group: the maximal run of left rows sharing one
        // `by`-tuple. The right cursor seeks the same group.
        let group_end = group_keys.left_group_end(l);
        // Advance the right cursor to the first right row whose group is not
        // strictly before the left group's, then bound that group.
        while r < right_n && group_keys.right_before_left(r, l) {
            r += 1;
        }
        let right_group_start = r;
        let right_group_end = if r < right_n && group_keys.same_group(l, r) {
            group_keys.right_group_end(r)
        } else {
            // No right group equal to this left group: every left row in it is
            // unmatched (null fact columns), the right cursor stays put.
            r
        };

        match spec.direction {
            MatchDirection::Backward | MatchDirection::Forward => merge_directional(
                spec.direction,
                spec.boundary,
                tolerance,
                &left_time[l..group_end],
                &right_time[right_group_start..right_group_end],
                right_group_start,
                &mut matched,
            ),
            MatchDirection::Nearest => merge_nearest(
                spec.boundary,
                tolerance,
                &left_time[l..group_end],
                &right_time[right_group_start..right_group_end],
                right_group_start,
                &mut matched,
            ),
        }

        // Ambiguity is loud: within the matched right group, a duplicate
        // (`by`, `time`) with no tie-break column is a true ambiguous match.
        if matches!(spec.tie_break, TieBreak::Error) {
            detect_ambiguous(&right_time[right_group_start..right_group_end])?;
        }

        l = group_end;
        r = right_group_end;
    }

    assemble_output(left, right, &matched, project, out_schema)
}

/// The directional (backward/forward) single-pointer merge over one group.
/// `left`/`right` are the group's temporal slices; `right_base` offsets a local
/// right index back to the global row index recorded in `matched`.
///
/// Backward: as the left instant advances, advance the right pointer over every
/// eligible fact (`<=` or `<` the instant) and remember the last one — the most
/// recent at/before. Forward mirrors it: the first fact at/after. Tolerance, when
/// set, rejects a remembered fact whose distance exceeds the limit.
fn merge_directional(
    direction: MatchDirection,
    boundary: Boundary,
    tolerance: Option<Tolerance>,
    left: &[Option<i128>],
    right: &[Option<i128>],
    right_base: usize,
    matched: &mut Vec<Option<usize>>,
) {
    let limit = tolerance_limit(tolerance);
    match direction {
        MatchDirection::Backward => {
            let mut last_eligible: Option<usize> = None;
            let mut cursor = 0usize;
            for &lt in left {
                let Some(lt) = lt else {
                    matched.push(None);
                    continue;
                };
                while cursor < right.len() {
                    match right[cursor] {
                        Some(rt) if eligible_at_or_before(rt, lt, boundary) => {
                            last_eligible = Some(cursor);
                            cursor += 1;
                        }
                        // A null fact-time is never a candidate; skip it without
                        // letting it terminate the scan.
                        None => cursor += 1,
                        _ => break,
                    }
                }
                matched.push(within_limit(last_eligible, right, right_base, lt, limit));
            }
        }
        MatchDirection::Forward => {
            // Mirror: the first fact at/after. Scan from the front, skipping
            // facts strictly before the instant, and take the first eligible.
            let mut cursor = 0usize;
            for &lt in left {
                let Some(lt) = lt else {
                    matched.push(None);
                    continue;
                };
                while cursor < right.len() {
                    match right[cursor] {
                        Some(rt) if !eligible_at_or_after(rt, lt, boundary) => cursor += 1,
                        None => cursor += 1,
                        _ => break,
                    }
                }
                let hit = (cursor < right.len() && right[cursor].is_some()).then_some(cursor);
                matched.push(within_limit(hit, right, right_base, lt, limit));
            }
        }
        MatchDirection::Nearest => unreachable!("nearest is merged by merge_nearest"),
    }
}

/// The `Nearest` merge over one group: for each left instant, the candidate of
/// smallest absolute temporal distance, equidistant ties resolved toward the
/// past (the backward candidate wins). Boundary still gates exact coincidence;
/// tolerance still caps the distance. Numeric temporal keys only (enforced at
/// validation), so the absolute distance is well-defined.
fn merge_nearest(
    boundary: Boundary,
    tolerance: Option<Tolerance>,
    left: &[Option<i128>],
    right: &[Option<i128>],
    right_base: usize,
    matched: &mut Vec<Option<usize>>,
) {
    let limit = tolerance_limit(tolerance);
    // A single forward-only scan suffices because both sides are time-sorted:
    // track the last fact at/before (backward candidate) and the first fact
    // at/after (forward candidate) as the left instant advances.
    let mut back: Option<usize> = None;
    let mut cursor = 0usize;
    for &lt in left {
        let Some(lt) = lt else {
            matched.push(None);
            continue;
        };
        while cursor < right.len() {
            match right[cursor] {
                Some(rt) if eligible_at_or_before(rt, lt, boundary) => {
                    back = Some(cursor);
                    cursor += 1;
                }
                None => cursor += 1,
                _ => break,
            }
        }
        // The forward candidate is the next non-null fact at/after the instant.
        let mut fwd_cursor = cursor;
        while fwd_cursor < right.len() && right[fwd_cursor].is_none() {
            fwd_cursor += 1;
        }
        let fwd = (fwd_cursor < right.len()
            && right[fwd_cursor].is_some_and(|rt| eligible_at_or_after(rt, lt, boundary)))
        .then_some(fwd_cursor);

        let pick = nearest_of(back, fwd, right, lt);
        matched.push(within_limit(pick, right, right_base, lt, limit));
    }
}

/// Pick the nearer of the backward and forward candidates, ties toward the past.
fn nearest_of(
    back: Option<usize>,
    fwd: Option<usize>,
    right: &[Option<i128>],
    lt: i128,
) -> Option<usize> {
    match (back, fwd) {
        (Some(b), Some(f)) => {
            let db = lt - right[b].expect("backward candidate is non-null");
            let df = right[f].expect("forward candidate is non-null") - lt;
            // Equidistant → past wins.
            Some(if df < db { f } else { b })
        }
        (Some(b), None) => Some(b),
        (None, Some(f)) => Some(f),
        (None, None) => None,
    }
}

/// Whether a fact instant is eligible under the at-or-before rule for the
/// boundary: `<=` for `Inclusive`, strict `<` for `Exclusive`.
fn eligible_at_or_before(fact: i128, instant: i128, boundary: Boundary) -> bool {
    match boundary {
        Boundary::Inclusive => fact <= instant,
        Boundary::Exclusive => fact < instant,
    }
}

/// Whether a fact instant is eligible under the at-or-after rule for the
/// boundary: `>=` for `Inclusive`, strict `>` for `Exclusive`.
fn eligible_at_or_after(fact: i128, instant: i128, boundary: Boundary) -> bool {
    match boundary {
        Boundary::Inclusive => fact >= instant,
        Boundary::Exclusive => fact > instant,
    }
}

/// The numeric tolerance limit, if any. Both `Duration` (microseconds) and
/// `Steps` are an absolute `i128` distance over the widened temporal key (the
/// validation pins the unit to the key type), so they collapse to one ceiling.
fn tolerance_limit(tolerance: Option<Tolerance>) -> Option<i128> {
    tolerance.map(|t| match t {
        Tolerance::Duration(d) => d as i128,
        Tolerance::Steps(s) => s as i128,
    })
}

/// Apply the tolerance ceiling to a candidate (local index): a candidate whose
/// absolute distance to the instant exceeds the limit becomes a no-match (the
/// spine row is preserved, fact columns null). Returns the global row index.
fn within_limit(
    candidate: Option<usize>,
    right: &[Option<i128>],
    right_base: usize,
    instant: i128,
    limit: Option<i128>,
) -> Option<usize> {
    let local = candidate?;
    let rt = right[local].expect("a matched candidate is non-null");
    if let Some(limit) = limit {
        if (instant - rt).abs() > limit {
            return None;
        }
    }
    Some(right_base + local)
}

/// Detect a true ambiguous match within a right group under `TieBreak::Error`: a
/// run of two facts sharing the same temporal instant. With no secondary column
/// to disambiguate, that is a loud failure, never a silent pick.
fn detect_ambiguous(right: &[Option<i128>]) -> Result<(), AsofError> {
    for pair in right.windows(2) {
        if let [Some(a), Some(b)] = pair {
            if a == b {
                return Err(AsofError::AmbiguousMatch);
            }
        }
    }
    Ok(())
}

/// The `by`-tuple grouping of both sides. With at least one `by` column it holds
/// the encoded tuples (one shared [`RowConverter`] so left and right encodings
/// are comparable) plus an explicit null mask so SQL `NULL ≠ NULL` holds at group
/// boundaries. With **no** `by` columns it is the single global group: every row
/// belongs to one group and there are no null keys (an empty tuple is never
/// null), so the row-encoding is skipped entirely (a 0-field converter has no
/// `.row(i)` to call).
enum GroupKeys {
    /// One or more equality keys: per-row encoded tuples + null masks.
    Keyed {
        left_rows: Rows,
        right_rows: Rows,
        left_null: Vec<bool>,
        right_null: Vec<bool>,
    },
    /// No equality keys: one global group spanning both whole sides.
    Global { left_n: usize, right_n: usize },
}

impl GroupKeys {
    fn new(left: &SortedPartition, right: &SortedPartition) -> Result<Self, AsofError> {
        if left.by_indices.is_empty() {
            return Ok(Self::Global {
                left_n: left.batch.num_rows(),
                right_n: right.batch.num_rows(),
            });
        }
        let fields: Vec<SortField> = left
            .by_indices
            .iter()
            .map(|&i| SortField::new(left.batch.column(i).data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields).map_err(arrow_err)?;
        let left_cols: Vec<ArrayRef> = left
            .by_indices
            .iter()
            .map(|&i| Arc::clone(left.batch.column(i)))
            .collect();
        let right_cols: Vec<ArrayRef> = right
            .by_indices
            .iter()
            .map(|&i| Arc::clone(right.batch.column(i)))
            .collect();
        let left_rows = converter.convert_columns(&left_cols).map_err(arrow_err)?;
        let right_rows = converter.convert_columns(&right_cols).map_err(arrow_err)?;
        let left_null = row_null_mask(&left_cols, left.batch.num_rows());
        let right_null = row_null_mask(&right_cols, right.batch.num_rows());
        Ok(Self::Keyed {
            left_rows,
            right_rows,
            left_null,
            right_null,
        })
    }

    /// The exclusive end of the left group starting at `start` — the maximal run
    /// of identical, non-null `by`-tuples. A null-key row is its own singleton
    /// group that matches nothing (SQL `NULL ≠ NULL`); the global group spans the
    /// whole side.
    fn left_group_end(&self, start: usize) -> usize {
        match self {
            Self::Global { left_n, .. } => *left_n,
            Self::Keyed {
                left_rows,
                left_null,
                ..
            } => {
                if left_null[start] {
                    return start + 1;
                }
                let mut end = start + 1;
                while end < left_rows.num_rows()
                    && !left_null[end]
                    && left_rows.row(end) == left_rows.row(start)
                {
                    end += 1;
                }
                end
            }
        }
    }

    /// The exclusive end of the right group starting at `start`. Mirror of
    /// [`Self::left_group_end`].
    fn right_group_end(&self, start: usize) -> usize {
        match self {
            Self::Global { right_n, .. } => *right_n,
            Self::Keyed {
                right_rows,
                right_null,
                ..
            } => {
                if right_null[start] {
                    return start + 1;
                }
                let mut end = start + 1;
                while end < right_rows.num_rows()
                    && !right_null[end]
                    && right_rows.row(end) == right_rows.row(start)
                {
                    end += 1;
                }
                end
            }
        }
    }

    /// Whether the right row at `r` belongs to a group strictly before the left
    /// group at `l` (so the right cursor must advance past it). A null key on
    /// either side never matches, so a null-key right row is always "before" in
    /// the sense that it can be skipped for this left group. In the global group
    /// no row is ever "before" — there is only one group.
    fn right_before_left(&self, r: usize, l: usize) -> bool {
        match self {
            Self::Global { .. } => false,
            Self::Keyed {
                left_rows,
                right_rows,
                left_null,
                right_null,
            } => {
                if right_null[r] {
                    return true;
                }
                if left_null[l] {
                    // The left group is a null singleton matching nothing; the
                    // right cursor need not advance for it.
                    return false;
                }
                right_rows.row(r) < left_rows.row(l)
            }
        }
    }

    /// Whether the left row at `l` and the right row at `r` share a non-null
    /// `by`-tuple — the group-equality the merge keys on. Two nulls never match;
    /// in the global group every left and right row share the one group.
    fn same_group(&self, l: usize, r: usize) -> bool {
        match self {
            Self::Global { .. } => true,
            Self::Keyed {
                left_rows,
                right_rows,
                left_null,
                right_null,
            } => {
                if left_null[l] || right_null[r] {
                    return false;
                }
                left_rows.row(l) == right_rows.row(r)
            }
        }
    }
}

/// A row's `by`-tuple is "null" for matching if ANY key column is null — SQL
/// equality over a tuple with a null component is never true.
fn row_null_mask(cols: &[ArrayRef], rows: usize) -> Vec<bool> {
    (0..rows)
        .map(|i| cols.iter().any(|c| c.is_null(i)))
        .collect()
}

/// Widen a temporal column into `Option<i128>` per row — one comparison domain
/// for every timestamp/date/integer width. A null is `None` (an unorderable
/// instant). The validation has already rejected non-temporal types, so an
/// unexpected type here is an internal invariant violation surfaced as an Arrow
/// error rather than a panic.
fn temporal_i128(array: &ArrayRef) -> Result<Vec<Option<i128>>, AsofError> {
    use arrow::array::{
        Date32Array, Date64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };

    macro_rules! widen {
        ($ty:ty) => {{
            let a = array
                .as_any()
                .downcast_ref::<$ty>()
                .expect("validated temporal type downcasts");
            Ok((0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i) as i128))
                .collect())
        }};
    }

    match array.data_type() {
        DataType::Int8 => widen!(Int8Array),
        DataType::Int16 => widen!(Int16Array),
        DataType::Int32 => widen!(Int32Array),
        DataType::Int64 => widen!(Int64Array),
        DataType::UInt8 => widen!(UInt8Array),
        DataType::UInt16 => widen!(UInt16Array),
        DataType::UInt32 => widen!(UInt32Array),
        DataType::UInt64 => widen!(UInt64Array),
        DataType::Date32 => widen!(Date32Array),
        DataType::Date64 => widen!(Date64Array),
        DataType::Timestamp(TimeUnit::Second, _) => widen!(TimestampSecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => widen!(TimestampMillisecondArray),
        DataType::Timestamp(TimeUnit::Microsecond, _) => widen!(TimestampMicrosecondArray),
        DataType::Timestamp(TimeUnit::Nanosecond, _) => widen!(TimestampNanosecondArray),
        other => Err(AsofError::UnorderedTimeKey {
            column: "<temporal>".to_string(),
            found: other.to_string(),
        }),
    }
}

/// Build the joined batch from the left rows and the per-left matched right row.
/// Left columns ride through unchanged; each projected right column is gathered
/// by the matched index (null where unmatched) via one `take` per column — the
/// vectorised gather, never a per-row copy.
fn assemble_output(
    left: &SortedPartition,
    right: &SortedPartition,
    matched: &[Option<usize>],
    project: &[usize],
    out_schema: &SchemaRef,
) -> Result<RecordBatch, AsofError> {
    let mut columns: Vec<ArrayRef> = left.batch.columns().to_vec();

    // The gather indices: the matched right row, or null for an unmatched spine
    // row. `take` with a null index yields a null in the output.
    let indices = UInt32Array::from_iter(matched.iter().map(|m| m.map(|i| i as u32)));

    for &col_idx in project {
        let source = right.batch.column(col_idx);
        if matched.is_empty() {
            columns.push(new_null_array(source.data_type(), 0));
        } else {
            let gathered = take(source.as_ref(), &indices, None).map_err(arrow_err)?;
            columns.push(gathered);
        }
    }

    RecordBatch::try_new(Arc::clone(out_schema), columns).map_err(arrow_err)
}
