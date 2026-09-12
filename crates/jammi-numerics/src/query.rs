//! The validated query vector — the ONE type every distance kernel and every
//! search entry accepts, so that "was this query checked?" is decided by the
//! compiler, not by a list of call sites anyone maintains.
//!
//! A raw `&[f32]` can carry two faults into a kernel: a non-finite component
//! (which makes every distance it touches `NaN` — and a `NaN` distance is the
//! merge's sort key and the user-visible similarity), and the wrong width
//! (which reads past the stored vector or silently scores a prefix). Both
//! were guarded per producer in earlier rounds, and each round found producers
//! the previous list had missed. [`ValidatedQuery`] closes that class
//! structurally: its only constructor is [`validate_query`], which checks
//! finiteness and (when the width is known) the width, and every consumer
//! takes `&ValidatedQuery`. A new producer that skips validation does not
//! compile.
//!
//! [`QuerySource`] is the provenance the error class is decided by: a query
//! the CALLER supplied is the caller's fault (a schema-class error, the same
//! class as a width mismatch); a vector read back from STORAGE (a
//! query-by-example row, a neighbour-graph node, a scanned corpus row) is a
//! corrupt artifact, named by its table.

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

/// Why a query was refused by [`validate_query`] or
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
    /// The query is not as wide as the vectors it would be compared with.
    Width {
        /// The width required.
        expected: usize,
        /// The width the query has.
        actual: usize,
        /// Where the query came from.
        source: QuerySource,
    },
}

impl QueryValidationError {
    /// The provenance of the refused query.
    pub fn source(&self) -> &QuerySource {
        match self {
            Self::NonFinite { source, .. } | Self::Width { source, .. } => source,
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

/// A query vector every component of which is finite and whose width, when
/// the width was known at validation, matched. Constructed ONLY by
/// [`validate_query`]; dereferences to `&[f32]` for the kernels.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedQuery {
    values: Vec<f32>,
    source: QuerySource,
}

/// Validate `values` as a query: every component finite, and exactly
/// `expected_width` wide when that is `Some`. `None` is for an entry that
/// does not yet know the width (the width is then enforced downstream by
/// [`ValidatedQuery::require_width`] against the index or scan it meets).
pub fn validate_query(
    values: Vec<f32>,
    expected_width: Option<usize>,
    source: QuerySource,
) -> Result<ValidatedQuery, QueryValidationError> {
    if let Some((index, value)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(QueryValidationError::NonFinite {
            index,
            value: *value,
            source,
        });
    }
    if let Some(expected) = expected_width {
        if values.len() != expected {
            return Err(QueryValidationError::Width {
                expected,
                actual: values.len(),
                source,
            });
        }
    }
    Ok(ValidatedQuery { values, source })
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

    /// Enforce the width at a consumer that knows it (an index's dimensions,
    /// a scan's `FixedSizeList` length) — the check an entry validated with
    /// `expected_width = None` defers to here. Typed, provenance-carrying.
    pub fn require_width(&self, expected: usize) -> Result<(), QueryValidationError> {
        if self.values.len() != expected {
            return Err(QueryValidationError::Width {
                expected,
                actual: self.values.len(),
                source: self.source.clone(),
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

    fn caller(v: Vec<f32>, w: Option<usize>) -> Result<ValidatedQuery, QueryValidationError> {
        validate_query(v, w, QuerySource::Caller)
    }

    #[test]
    fn finite_and_right_width_validates() {
        let q = caller(vec![1.0, 0.0, 0.5], Some(3)).unwrap();
        assert_eq!(q.as_slice(), &[1.0, 0.0, 0.5]);
        assert_eq!(q.len(), 3);
        assert!(q.require_width(3).is_ok());
        assert!(matches!(
            q.require_width(4),
            Err(QueryValidationError::Width {
                expected: 4,
                actual: 3,
                ..
            })
        ));
    }

    #[test]
    fn every_non_finite_component_is_refused_with_its_index() {
        for poison in [f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = caller(vec![1.0, poison, 0.0], None).unwrap_err();
            match err {
                QueryValidationError::NonFinite { index, .. } => assert_eq!(index, 1),
                other => panic!("{poison:?}: {other:?}"),
            }
        }
    }

    #[test]
    fn width_is_checked_only_when_known() {
        assert!(caller(vec![1.0, 0.0], None).is_ok());
        assert!(matches!(
            caller(vec![1.0, 0.0], Some(3)),
            Err(QueryValidationError::Width {
                expected: 3,
                actual: 2,
                ..
            })
        ));
        // An empty query against a known width is a width fault.
        assert!(caller(Vec::new(), Some(3)).is_err());
    }

    #[test]
    fn finiteness_is_checked_before_width() {
        let err = caller(vec![f32::NAN], Some(3)).unwrap_err();
        assert!(matches!(err, QueryValidationError::NonFinite { .. }));
    }

    #[test]
    fn provenance_rides_the_error() {
        let err = validate_query(
            vec![f32::NAN],
            None,
            QuerySource::Stored {
                table: "docs".into(),
            },
        )
        .unwrap_err();
        assert_eq!(
            err.source(),
            &QuerySource::Stored {
                table: "docs".into()
            }
        );
        assert!(err.to_string().contains("read from table 'docs'"));
    }
}
