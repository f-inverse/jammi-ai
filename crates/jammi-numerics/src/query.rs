//! The query vector's two checked states — a type-state, so "was this query
//! checked, and against what?" is decided by the compiler, not by a list of
//! call sites anyone maintains.
//!
//! A raw `&[f32]` can carry two faults into a kernel: a non-finite component
//! (which makes every distance it touches `NaN` — and a `NaN` distance is the
//! merge's sort key and the user-visible similarity), and the wrong width
//! (which reads past the stored vector or silently scores a prefix). The two
//! checks are two states:
//!
//! - [`FiniteQuery`] — every component is finite; the width is UNCHECKED. It
//!   exposes no component, no length and no slice, and no kernel or search
//!   entry accepts it. It is what an entry holds between receiving a vector
//!   and having a width authority in hand.
//! - [`ValidatedQuery`] — finite AND exactly as wide as the AUTHORITY its
//!   entry validated it against (the catalog's recorded width; for a table
//!   whose row records none, the width of the index or scan the entry loads
//!   before any search). Every distance kernel and every search consumer
//!   takes `&ValidatedQuery`.
//!
//! The one way from the first state to the second is
//! [`FiniteQuery::against_authority`], and [`validate_query`] is the two
//! steps composed for an entry that knows the width up front. There is no
//! width-less path to a `ValidatedQuery`: an entry that holds an authority
//! cannot reach a consumer without checking the query against it, and a new
//! producer that skips either check does not compile.
//!
//! [`QuerySource`] is the provenance the error class is decided by: a query
//! the CALLER supplied is the caller's fault (a schema-class error); a vector
//! read back from STORAGE (a query-by-example row, a neighbour-graph node, a
//! scanned corpus row) is a corrupt artifact, named by its table.
//!
//! **Whose fault a width mismatch is depends on WHERE it is discovered.**
//! The authority check — [`FiniteQuery::against_authority`] — attributes a
//! mismatch by the query's own provenance
//! ([`QueryValidationError::Width`]), because nothing but the query and the
//! authority has been consulted. Every width check a `ValidatedQuery` meets
//! afterwards — an index's declared dimensions, a scan's `FixedSizeList`
//! length, a stored vector's own length — is downstream of that check by
//! construction, so a mismatch there is the ARTIFACT's drift, never the
//! query's. [`ValidatedQuery::require_width`] is that downstream check: its
//! error ([`QueryValidationError::ArtifactMismatch`]) carries no
//! `QuerySource` at all, so no argument makes it express a caller fault, and
//! a `ValidatedQuery` has no method that can.
//!
//! A `FiniteQuery` cannot stand in for a `ValidatedQuery`:
//!
//! ```compile_fail,E0308
//! use jammi_numerics::distance::cosine_distance;
//! use jammi_numerics::query::{FiniteQuery, QuerySource};
//!
//! let finite = FiniteQuery::new(vec![1.0, 0.0], QuerySource::Caller).unwrap();
//! // A kernel takes `&ValidatedQuery`; the width is still unchecked here.
//! cosine_distance(&finite, &[1.0, 0.0]);
//! ```
//!
//! and it cannot be read as a slice:
//!
//! ```compile_fail,E0308
//! use jammi_numerics::query::{FiniteQuery, QuerySource};
//!
//! let finite = FiniteQuery::new(vec![1.0, 0.0], QuerySource::Caller).unwrap();
//! let components: &[f32] = &finite;
//! ```
//!
//! The same vector, once checked against its authority, is both:
//!
//! ```
//! use jammi_numerics::distance::cosine_distance;
//! use jammi_numerics::query::{FiniteQuery, QuerySource};
//!
//! let finite = FiniteQuery::new(vec![1.0, 0.0], QuerySource::Caller).unwrap();
//! let query = finite.against_authority(2).unwrap();
//! let components: &[f32] = &query;
//! assert_eq!(components, &[1.0, 0.0]);
//! assert!(cosine_distance(&query, &[1.0, 0.0]).abs() < 1e-6);
//! ```

use std::fmt;
use std::ops::Deref;

/// Where a query vector came from — decides whose fault a bad one is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuerySource {
    /// Supplied by the caller of a search (a request's vector, an encoder's
    /// output for the caller's text). A fault is the caller's: schema-class.
    Caller,
    /// Read back from a stored table (query-by-example, a graph node, a
    /// scanned row). A fault is a corrupt artifact, named by its table.
    Stored {
        /// The table the vector was read from.
        table: String,
    },
}

/// Why a query was refused: by [`FiniteQuery::new`], by
/// [`FiniteQuery::against_authority`], or by
/// [`ValidatedQuery::require_width`].
#[derive(Debug, Clone, PartialEq)]
pub enum QueryValidationError {
    /// A component is `NaN` or infinite.
    NonFinite {
        /// Index of the first non-finite component.
        index: usize,
        /// Its value.
        value: f32,
        /// Where the query came from.
        source: QuerySource,
    },
    /// The query is not as wide as the AUTHORITY its entry checked it
    /// against ([`FiniteQuery::against_authority`]). Nothing but the query and
    /// the authority has been consulted, so the fault is attributed by the
    /// query's OWN provenance — the one width error that can be the caller's.
    Width {
        /// The width required.
        expected: usize,
        /// The width the query has.
        actual: usize,
        /// Where the query came from.
        source: QuerySource,
    },
    /// The query disagreed with a downstream ARTIFACT it met at or after a
    /// consumer ([`ValidatedQuery::require_width`]) — an index's dimensions,
    /// a scan's `FixedSizeList` length, a stored vector's own length. Carries
    /// no [`QuerySource`] at all: unlike `Width`/`NonFinite`, this is never
    /// about where the query came from, only about what it disagreed with,
    /// so there is no field a reader could take for the query's own
    /// provenance.
    ArtifactMismatch {
        /// What disagreed with the query (an index, a segment, a scan) —
        /// never the query's own source table.
        artifact: String,
        /// The width the artifact required.
        expected: usize,
        /// The width the query has.
        actual: usize,
    },
}

impl QueryValidationError {
    /// The refused query's own provenance, for the two variants that have
    /// one. [`Self::ArtifactMismatch`] has none — it is never about the
    /// query's own source — so this returns `None` for it rather than
    /// forcing a caller to invent or borrow a provenance that would be
    /// false.
    pub fn source(&self) -> Option<&QuerySource> {
        match self {
            Self::NonFinite { source, .. } | Self::Width { source, .. } => Some(source),
            Self::ArtifactMismatch { .. } => None,
        }
    }
}

impl fmt::Display for QueryValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite {
                index,
                value,
                source,
            } => write!(
                f,
                "query vector component {index} is non-finite ({value:?}) [{}]",
                describe(source)
            ),
            Self::Width {
                expected,
                actual,
                source,
            } => write!(
                f,
                "query vector has {actual} dimensions, expected {expected} [{}]",
                describe(source)
            ),
            Self::ArtifactMismatch {
                artifact,
                expected,
                actual,
            } => write!(
                f,
                "query vector has {actual} dimensions, expected {expected} [checked against '{artifact}']"
            ),
        }
    }
}

impl std::error::Error for QueryValidationError {}

fn describe(source: &QuerySource) -> String {
    match source {
        QuerySource::Caller => "supplied by the caller".to_string(),
        QuerySource::Stored { table } => format!("read from table '{table}'"),
    }
}

/// A query vector every component of which is finite, and whose width is
/// UNCHECKED. Constructed only by [`FiniteQuery::new`]; its one transition is
/// [`FiniteQuery::against_authority`]. It exposes no component, no length
/// and no slice, so nothing can be computed from it until its width has been
/// checked against an authority.
#[derive(Debug, Clone, PartialEq)]
pub struct FiniteQuery {
    values: Vec<f32>,
    source: QuerySource,
}

impl FiniteQuery {
    /// Check that every component of `values` is finite. A non-finite
    /// component is attributed by `source`.
    pub fn new(values: Vec<f32>, source: QuerySource) -> Result<Self, QueryValidationError> {
        match values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
            Some((index, value)) => Err(QueryValidationError::NonFinite {
                index,
                value: *value,
                source,
            }),
            None => Ok(Self { values, source }),
        }
    }

    /// Where this query came from.
    pub fn source(&self) -> &QuerySource {
        &self.source
    }

    /// Check the width against `width` — the AUTHORITY the entry holds (the
    /// catalog's recorded width; for a table whose row records none, the
    /// width of the index or scan the entry loaded before any search). A
    /// mismatch is attributed by this query's own provenance: nothing
    /// downstream has been consulted yet, so a caller's wrong-width vector
    /// is the caller's fault and a stored one is its table's.
    pub fn against_authority(self, width: usize) -> Result<ValidatedQuery, QueryValidationError> {
        if self.values.len() != width {
            return Err(QueryValidationError::Width {
                expected: width,
                actual: self.values.len(),
                source: self.source,
            });
        }
        Ok(ValidatedQuery {
            values: self.values,
            source: self.source,
        })
    }
}

/// A query vector every component of which is finite and whose width matched
/// the authority its entry checked it against. Constructed ONLY through
/// [`FiniteQuery::against_authority`] (which [`validate_query`] composes);
/// dereferences to `&[f32]` for the kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedQuery {
    values: Vec<f32>,
    source: QuerySource,
}

/// Validate `values` as a query against the authority `width`: every
/// component finite ([`FiniteQuery::new`]) and exactly `width` wide
/// ([`FiniteQuery::against_authority`]) — the two checks composed, for an
/// entry that holds its authority when the vector arrives.
pub fn validate_query(
    values: Vec<f32>,
    width: usize,
    source: QuerySource,
) -> Result<ValidatedQuery, QueryValidationError> {
    FiniteQuery::new(values, source)?.against_authority(width)
}

impl ValidatedQuery {
    /// The components.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Where this query came from.
    pub fn source(&self) -> &QuerySource {
        &self.source
    }

    /// Enforce `expected` against a downstream ARTIFACT this query meets AT
    /// or AFTER a consumer (an index's dimensions, a scan's `FixedSizeList`
    /// length, a stored vector's own length): the width check every kernel
    /// and consumer reaches for. `artifact` names what disagreed (a segment,
    /// a table's scan, an index) for the resulting error.
    ///
    /// Does not read `self.source()` at all — it CANNOT attribute a mismatch
    /// to the caller, whichever [`QuerySource`] this query carries, because
    /// its error variant ([`QueryValidationError::ArtifactMismatch`]) has no
    /// `QuerySource` field to put one in. A `ValidatedQuery` has already
    /// matched its authority, so a mismatch discovered here is the
    /// artifact's own drift, never the query's.
    pub fn require_width(
        &self,
        expected: usize,
        artifact: impl Into<String>,
    ) -> Result<(), QueryValidationError> {
        if self.values.len() != expected {
            return Err(QueryValidationError::ArtifactMismatch {
                artifact: artifact.into(),
                expected,
                actual: self.values.len(),
            });
        }
        Ok(())
    }

    /// Give the components back (for a wire encoding).
    pub fn into_inner(self) -> Vec<f32> {
        self.values
    }
}

impl Deref for ValidatedQuery {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        &self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored() -> QuerySource {
        QuerySource::Stored {
            table: "docs".into(),
        }
    }

    #[test]
    fn a_finite_vector_at_the_authority_width_validates() {
        let q = validate_query(vec![1.0, 0.0, 0.5], 3, QuerySource::Caller).unwrap();
        assert_eq!(q.as_slice(), &[1.0, 0.0, 0.5]);
        assert_eq!(q.len(), 3);
        assert_eq!(q.source(), &QuerySource::Caller);
        assert!(q.require_width(3, "segment").is_ok());
    }

    /// The first state: finiteness is checked, the provenance is kept, and
    /// the width is not yet anyone's fault.
    #[test]
    fn the_finite_state_checks_finiteness_and_keeps_provenance() {
        let finite = FiniteQuery::new(vec![1.0, 0.0], stored()).unwrap();
        assert_eq!(finite.source(), &stored());
        // Any width is still open: the same finite query validates against
        // the authority it is eventually given, whatever that is.
        assert!(finite.clone().against_authority(2).is_ok());
        assert!(finite.against_authority(3).is_err());
        // An empty vector is finite; only an authority can refuse it.
        assert!(FiniteQuery::new(Vec::new(), QuerySource::Caller).is_ok());
    }

    #[test]
    fn every_non_finite_component_is_refused_with_its_index() {
        for poison in [f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            match FiniteQuery::new(vec![1.0, poison, 0.0], QuerySource::Caller).unwrap_err() {
                QueryValidationError::NonFinite { index, source, .. } => {
                    assert_eq!(index, 1);
                    assert_eq!(source, QuerySource::Caller);
                }
                other => panic!("{poison:?}: {other:?}"),
            }
        }
    }

    /// The transition attributes a mismatch by the query's own provenance:
    /// the caller's for a caller's vector, the table's for a stored one.
    #[test]
    fn the_authority_check_attributes_by_provenance() {
        let err = FiniteQuery::new(vec![1.0, 0.0, 0.5], QuerySource::Caller)
            .unwrap()
            .against_authority(4)
            .unwrap_err();
        assert_eq!(
            err,
            QueryValidationError::Width {
                expected: 4,
                actual: 3,
                source: QuerySource::Caller,
            }
        );
        assert_eq!(err.source(), Some(&QuerySource::Caller));
        assert!(err.to_string().contains("supplied by the caller"));

        let err = FiniteQuery::new(vec![1.0, 0.0, 0.5], stored())
            .unwrap()
            .against_authority(4)
            .unwrap_err();
        assert_eq!(err.source(), Some(&stored()));
        assert!(err.to_string().contains("read from table 'docs'"));

        // An empty query against an authority is a width fault.
        assert!(matches!(
            validate_query(Vec::new(), 3, QuerySource::Caller),
            Err(QueryValidationError::Width {
                expected: 3,
                actual: 0,
                ..
            })
        ));
    }

    /// `require_width` is the downstream, artifact-only check: whatever
    /// `QuerySource` the query carries, a mismatch it discovers is always
    /// `ArtifactMismatch`, which carries no `QuerySource` at all — never
    /// `Width { source: Caller, .. }` — and `.source()` returning `None` for
    /// it means a reader cannot even ask it for one.
    #[test]
    fn require_width_never_attributes_to_the_query() {
        for source in [QuerySource::Caller, stored()] {
            let q = validate_query(vec![1.0, 0.0, 0.5], 3, source).unwrap();
            let err = q.require_width(4, "segment 7").unwrap_err();
            assert_eq!(
                err,
                QueryValidationError::ArtifactMismatch {
                    artifact: "segment 7".into(),
                    expected: 4,
                    actual: 3,
                }
            );
            assert_eq!(err.source(), None);
            assert!(err.to_string().contains("checked against 'segment 7'"));
        }
    }

    #[test]
    fn finiteness_is_checked_before_width() {
        let err = validate_query(vec![f32::NAN], 3, QuerySource::Caller).unwrap_err();
        assert!(matches!(err, QueryValidationError::NonFinite { .. }));
    }

    #[test]
    fn provenance_rides_the_finiteness_error() {
        let err = FiniteQuery::new(vec![f32::NAN], stored()).unwrap_err();
        assert_eq!(err.source(), Some(&stored()));
        assert!(err.to_string().contains("read from table 'docs'"));
    }
}
