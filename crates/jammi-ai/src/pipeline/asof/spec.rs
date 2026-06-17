//! The frozen `asof_join` contract: the typed descriptor an `asof_join` lowers
//! to, plus the column roles, the four pinned knobs, and the typed error set.
//!
//! Every type here makes an invalid as-of join unrepresentable: the temporal
//! key role is named (never inferred), the four semantic decisions
//! ([`MatchDirection`] / [`Boundary`] / [`Tolerance`] / [`TieBreak`]) are enums
//! rather than stringly-typed flags, and a float temporal key is rejected at
//! build time because NaN has no total order. The spec carries no domain
//! vocabulary — an entity id is a `by` column, an instant is a `time` column,
//! and what a consumer assembles from the result is the consumer's concern.

use arrow_schema::{DataType, SchemaRef};

/// One side's column roles for the as-of match.
///
/// The temporal key must be a totally-ordered Arrow type (any `Timestamp(..)`,
/// `Date32`/`Date64`, or a signed/unsigned integer); a float temporal key is
/// rejected — NaN has no total order, so "most recent at or before" would be
/// undefined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsofKey {
    /// Equality ("by") columns that partition the match into independent
    /// groups — e.g. an entity id, an instrument symbol, a subject id.
    ///
    /// May be empty: an empty `by` matches across the whole relation (a single
    /// global group), which is occasionally what a user wants (one global
    /// calendar of facts). Never silently defaulted.
    pub by: Vec<String>,
    /// The temporal ordering column. Required; exactly one.
    pub time: String,
}

/// Which side of the spine instant the matched fact may fall on.
///
/// `Backward` (the default and the only leakage-safe choice for past-keyed
/// assembly) takes the most recent fact at/before the instant; `Forward` takes
/// the first fact at/after; `Nearest` takes the smallest absolute distance,
/// resolving equidistant candidates toward the past. `Nearest` requires a
/// numeric temporal key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchDirection {
    /// Most recent fact at/before the spine instant.
    Backward,
    /// First fact at/after the spine instant.
    Forward,
    /// Smallest absolute distance; equidistant ties resolve toward the past.
    Nearest,
}

/// Whether a fact whose time exactly equals the spine instant is eligible.
///
/// `Inclusive` (default, `<=`/`>=`); `Exclusive` is strict (`<`/`>`). This is
/// the single most error-prone as-of decision — it is pinned on the spec, never
/// inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// A fact stamped exactly at the spine instant matches.
    Inclusive,
    /// A fact stamped exactly at the spine instant does not match.
    Exclusive,
}

/// Optional maximum look-back/forward distance; a candidate farther than the
/// limit is treated as no-match (the spine row is preserved, fact columns null).
///
/// `Duration` (microseconds) for temporal keys; `Steps` for integer keys. The
/// limit is measured relative to each spine instant, never wall-clock now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tolerance {
    /// Microsecond limit for a temporal key.
    Duration(i64),
    /// Step limit for an integer key.
    Steps(i64),
}

/// How coincident candidate facts are disambiguated into one match.
///
/// Silent non-determinism here is a known footgun; this engine refuses it. A
/// secondary descending column (newest wins) disambiguates late-arriving facts;
/// absent one, a true duplicate at the matched instant fails loudly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TieBreak {
    /// Break ties by a secondary column, the maximal value winning — the
    /// transaction-/created-time column. Event time bounds the join; this
    /// secondary time disambiguates facts coincident on the event time.
    ByColumnDesc(String),
    /// No secondary column. A true duplicate (same `by`, same `time`) fails with
    /// [`AsofError::AmbiguousMatch`] rather than a non-deterministic pick.
    Error,
}

/// The frozen descriptor an `asof_join` lowers to. Construct via
/// [`AsofJoinSpecBuilder`].
#[derive(Debug, Clone)]
pub struct AsofJoinSpec {
    /// The spine's column roles.
    pub left: AsofKey,
    /// The facts' column roles.
    pub right: AsofKey,
    /// Which side of the spine instant the matched fact may fall on.
    pub direction: MatchDirection,
    /// Whether a fact at exactly the spine instant matches.
    pub boundary: Boundary,
    /// Optional look-back/forward limit; a candidate outside it is no-match.
    pub tolerance: Option<Tolerance>,
    /// How coincident candidate facts are disambiguated.
    pub tie_break: TieBreak,
    /// Right-side columns to attach to the output. Empty = all non-key columns.
    pub project: Vec<String>,
}

/// Builder for an [`AsofJoinSpec`] — the spec has more than three parameters, so
/// construction goes through the builder (per the type-driven-design rule).
/// The two `AsofKey`s are required; every knob defaults to the leakage-safe,
/// least-surprising choice (`Backward` / `Inclusive` / no tolerance / loud
/// `Error` tie-break / all-columns projection).
#[derive(Debug, Clone)]
pub struct AsofJoinSpecBuilder {
    left: AsofKey,
    right: AsofKey,
    direction: MatchDirection,
    boundary: Boundary,
    tolerance: Option<Tolerance>,
    tie_break: TieBreak,
    project: Vec<String>,
}

impl AsofJoinSpecBuilder {
    /// Start from the two required column-role descriptors. Defaults:
    /// `Backward`, `Inclusive`, no tolerance, `TieBreak::Error`, project-all.
    pub fn new(left: AsofKey, right: AsofKey) -> Self {
        Self {
            left,
            right,
            direction: MatchDirection::Backward,
            boundary: Boundary::Inclusive,
            tolerance: None,
            tie_break: TieBreak::Error,
            project: Vec::new(),
        }
    }

    /// Set the match direction.
    pub fn direction(mut self, direction: MatchDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Set the boundary inclusivity.
    pub fn boundary(mut self, boundary: Boundary) -> Self {
        self.boundary = boundary;
        self
    }

    /// Set (or clear, with `None`) the look-back/forward tolerance.
    pub fn tolerance(mut self, tolerance: Option<Tolerance>) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the tie-break policy.
    pub fn tie_break(mut self, tie_break: TieBreak) -> Self {
        self.tie_break = tie_break;
        self
    }

    /// Set the right-side projection (empty keeps all non-key columns).
    pub fn project(mut self, project: Vec<String>) -> Self {
        self.project = project;
        self
    }

    /// Finish the spec. Pure: assembles the descriptor without touching a
    /// schema. Schema-dependent validation (key presence, temporal-key
    /// orderability, the `Nearest`-needs-numeric rule) is enforced by
    /// [`AsofJoinSpec::validate_against`] when the operator binds to its inputs,
    /// where both schemas are known.
    pub fn build(self) -> AsofJoinSpec {
        AsofJoinSpec {
            left: self.left,
            right: self.right,
            direction: self.direction,
            boundary: self.boundary,
            tolerance: self.tolerance,
            tie_break: self.tie_break,
            project: self.project,
        }
    }
}

impl AsofJoinSpec {
    /// Validate the spec against the two input schemas, the point where every
    /// schema-dependent invariant is decidable:
    ///
    /// * every `by` column exists on its side,
    /// * each temporal key is a totally-ordered type (rejecting floats),
    /// * the two temporal keys share a type, and
    /// * `Nearest` is used only with a numeric temporal key.
    ///
    /// Pure; no I/O. Returns the validated temporal [`DataType`] both sides
    /// share, which the operator threads into the merge.
    pub fn validate_against(
        &self,
        left_schema: &SchemaRef,
        right_schema: &SchemaRef,
    ) -> Result<DataType, AsofError> {
        for col in &self.left.by {
            if left_schema.field_with_name(col).is_err() {
                return Err(AsofError::MissingByKey {
                    column: col.clone(),
                    side: "left",
                });
            }
        }
        for col in &self.right.by {
            if right_schema.field_with_name(col).is_err() {
                return Err(AsofError::MissingByKey {
                    column: col.clone(),
                    side: "right",
                });
            }
        }

        let left_time = field_type(left_schema, &self.left.time, "left")?;
        let right_time = field_type(right_schema, &self.right.time, "right")?;

        if !is_totally_ordered(&left_time) {
            return Err(AsofError::UnorderedTimeKey {
                column: self.left.time.clone(),
                found: left_time.to_string(),
            });
        }
        if !is_totally_ordered(&right_time) {
            return Err(AsofError::UnorderedTimeKey {
                column: self.right.time.clone(),
                found: right_time.to_string(),
            });
        }
        if left_time != right_time {
            return Err(AsofError::TimeKeyTypeMismatch {
                left: left_time.to_string(),
                right: right_time.to_string(),
            });
        }
        if self.direction == MatchDirection::Nearest && !is_numeric(&left_time) {
            return Err(AsofError::NearestRequiresNumeric {
                column: self.left.time.clone(),
                found: left_time.to_string(),
            });
        }
        Ok(left_time)
    }
}

/// Resolve a column's [`DataType`] on a side, mapping an absent column to the
/// typed [`AsofError::MissingByKey`] for the temporal role.
fn field_type(schema: &SchemaRef, column: &str, side: &'static str) -> Result<DataType, AsofError> {
    schema
        .field_with_name(column)
        .map(|f| f.data_type().clone())
        .map_err(|_| AsofError::MissingByKey {
            column: column.to_string(),
            side,
        })
}

/// Whether a temporal key type carries a total order suitable for "at or before"
/// — every timestamp/date and every integer width. Floats are excluded (NaN has
/// no total order); everything else is not a temporal key.
fn is_totally_ordered(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Timestamp(_, _)
            | DataType::Date32
            | DataType::Date64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// Whether a temporal key type is numeric — the constraint `Nearest` adds on top
/// of total order (a timestamp/date is ordered but `Nearest`'s absolute-distance
/// arithmetic is defined here only over the integer widths, matching the
/// cross-engine string-key restriction).
fn is_numeric(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Timestamp(_, _)
            | DataType::Date32
            | DataType::Date64
    )
}

/// Errors raised assembling, validating, or executing an as-of join.
#[derive(Debug, thiserror::Error)]
pub enum AsofError {
    /// The temporal key is not a totally-ordered type.
    #[error(
        "temporal key `{column}` has type {found}, which is not totally ordered; \
         expected a timestamp, date, or integer"
    )]
    UnorderedTimeKey {
        /// The offending column.
        column: String,
        /// Its Arrow type.
        found: String,
    },
    /// An equality key column is absent from its side's schema.
    #[error("equality key `{column}` not found in {side} schema")]
    MissingByKey {
        /// The missing column.
        column: String,
        /// Which side it was expected on.
        side: &'static str,
    },
    /// The two temporal keys differ in type, so "at or before" is undefined
    /// across them.
    #[error("left and right temporal keys differ in type: {left} vs {right}")]
    TimeKeyTypeMismatch {
        /// The left temporal type.
        left: String,
        /// The right temporal type.
        right: String,
    },
    /// `Nearest` was requested over a non-numeric temporal key.
    #[error("`Nearest` direction requires a numeric temporal key; `{column}` is {found}")]
    NearestRequiresNumeric {
        /// The offending column.
        column: String,
        /// Its Arrow type.
        found: String,
    },
    /// A group held duplicate facts at the matched instant and no tie-break
    /// column was given.
    #[error(
        "ambiguous match: group has duplicate facts at the matched instant and \
         no tie-break column was given"
    )]
    AmbiguousMatch,
    /// A DataFusion / Arrow execution error surfaced through the operator.
    #[error(transparent)]
    DataFusion(#[from] datafusion::error::DataFusionError),
}
