//! The per-row content hash an embedding table carries in its nullable
//! `_content_hash` column ([`crate::store::schema::embedding_table_schema`]).
//!
//! The hash is a domain-separated SHA-256 over the embedded columns of one
//! source row, in the descriptor's `columns` order, so an incremental refresh
//! can classify a row as unchanged / changed / added by comparing two hashes
//! instead of re-embedding it. Model, task and device are deliberately NOT
//! folded per row — they are the table's `definition_hash`, folded once.
//!
//! Encoding (`jammi.content_hash.v1`): the domain tag, then per column one
//! byte tag + `u64` little-endian length + the bytes —
//!
//! - `s` — a string cell (Utf8 / LargeUtf8 / Utf8View, or any other type once
//!   the caller has cast it to Utf8 with the runner's own kernel);
//! - `b` — a binary cell (Binary / LargeBinary / BinaryView / FixedSizeBinary);
//! - `n` — a null cell (zero-length payload).
//!
//! The pure fold ([`content_hash_row`]) is the single definition; the Arrow
//! kernel ([`content_hash_columns`]) applies it row-wise over already-rendered
//! columns. Rendering (casting a non-string column to text exactly as the
//! inference runner renders it) is the caller's — the `jammi_content_hash`
//! UDF in `jammi-ai` — so base, refresh and inference share one rendering by
//! construction.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, FixedSizeBinaryArray, LargeBinaryArray,
    LargeStringArray, StringArray, StringViewArray,
};
use arrow::datatypes::DataType;
use sha2::{Digest, Sha256};

use crate::error::{JammiError, Result};

/// The domain-separation tag every content hash starts with.
pub const CONTENT_HASH_DOMAIN: &[u8] = b"jammi.content_hash.v1";

/// One rendered cell of a source row, as the hash sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentValue<'a> {
    /// A string cell (tag `s`).
    Str(&'a str),
    /// A binary cell (tag `b`).
    Bytes(&'a [u8]),
    /// A null cell (tag `n`).
    Null,
}

/// A decoded 32-byte content hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// The lowercase hex rendering stored in the `_content_hash` column.
    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }

    /// Decode a stored hex value, refusing anything that is not exactly 64
    /// lowercase hex characters (K2: a reader never trusts a malformed cell).
    pub fn from_hex(s: &str) -> Result<Self> {
        if s.len() != 64 || !s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(JammiError::Schema {
                table: String::new(),
                column: "_content_hash".to_string(),
                expected: "64 lowercase hex characters".to_string(),
                actual: format!("{:?}", s),
            });
        }
        let mut out = [0u8; 32];
        hex::decode_to_slice(s, &mut out).map_err(|e| JammiError::Schema {
            table: String::new(),
            column: "_content_hash".to_string(),
            expected: "64 lowercase hex characters".to_string(),
            actual: format!("{s:?} ({e})"),
        })?;
        Ok(Self(out))
    }
}

/// The pure fold: hash one row's rendered cells in column order.
pub fn content_hash_row(values: &[ContentValue<'_>]) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(CONTENT_HASH_DOMAIN);
    for v in values {
        let (tag, bytes): (u8, &[u8]) = match v {
            ContentValue::Str(s) => (b's', s.as_bytes()),
            ContentValue::Bytes(b) => (b'b', b),
            ContentValue::Null => (b'n', &[]),
        };
        hasher.update([tag]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    ContentHash(hasher.finalize().into())
}

/// Read cell `i` of an already-rendered column as a [`ContentValue`].
fn cell(col: &ArrayRef, i: usize) -> Result<ContentValue<'_>> {
    if col.is_null(i) {
        return Ok(ContentValue::Null);
    }
    let any = col.as_any();
    let v = match col.data_type() {
        DataType::Utf8 => ContentValue::Str(any.downcast_ref::<StringArray>().unwrap().value(i)),
        DataType::LargeUtf8 => {
            ContentValue::Str(any.downcast_ref::<LargeStringArray>().unwrap().value(i))
        }
        DataType::Utf8View => {
            ContentValue::Str(any.downcast_ref::<StringViewArray>().unwrap().value(i))
        }
        DataType::Binary => {
            ContentValue::Bytes(any.downcast_ref::<BinaryArray>().unwrap().value(i))
        }
        DataType::LargeBinary => {
            ContentValue::Bytes(any.downcast_ref::<LargeBinaryArray>().unwrap().value(i))
        }
        DataType::BinaryView => {
            ContentValue::Bytes(any.downcast_ref::<BinaryViewArray>().unwrap().value(i))
        }
        DataType::FixedSizeBinary(_) => {
            ContentValue::Bytes(any.downcast_ref::<FixedSizeBinaryArray>().unwrap().value(i))
        }
        other => {
            return Err(JammiError::Schema {
                table: String::new(),
                column: "_content_hash".to_string(),
                expected: "a string- or binary-family column (render other types to Utf8 first)"
                    .to_string(),
                actual: format!("{other}"),
            })
        }
    };
    Ok(v)
}

/// The Arrow kernel: one hex hash per row over `columns` (all of equal length,
/// each already rendered to a string- or binary-family type). A column of any
/// other type is a typed refusal — the caller renders before hashing, never
/// this kernel, so the rendering stays the runner's.
pub fn content_hash_columns(columns: &[ArrayRef]) -> Result<StringArray> {
    if columns.is_empty() {
        return Err(JammiError::Schema {
            table: String::new(),
            column: "_content_hash".to_string(),
            expected: "at least one content column".to_string(),
            actual: "none".to_string(),
        });
    }
    let rows = columns[0].len();
    for c in columns {
        if c.len() != rows {
            return Err(JammiError::Schema {
                table: String::new(),
                column: "_content_hash".to_string(),
                expected: format!("{rows} rows in every column"),
                actual: format!("{} rows", c.len()),
            });
        }
    }
    let mut out: Vec<String> = Vec::with_capacity(rows);
    let mut cells: Vec<ContentValue<'_>> = Vec::with_capacity(columns.len());
    for i in 0..rows {
        cells.clear();
        for c in columns {
            cells.push(cell(c, i)?);
        }
        out.push(content_hash_row(&cells).to_hex());
    }
    Ok(StringArray::from(out))
}

/// A `_content_hash` column of `rows` nulls — the fifth column every
/// hand-built embedding batch carries (only the embedding pipeline writes
/// real hashes).
pub fn null_hash_column(rows: usize) -> ArrayRef {
    Arc::new(StringArray::new_null(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_hash_is_domain_separated_and_order_sensitive() {
        let a = content_hash_row(&[ContentValue::Str("x"), ContentValue::Str("y")]);
        let b = content_hash_row(&[ContentValue::Str("y"), ContentValue::Str("x")]);
        assert_ne!(a, b);
        // A null is not an empty string and a string is not the same bytes as
        // a binary cell.
        assert_ne!(
            content_hash_row(&[ContentValue::Null]),
            content_hash_row(&[ContentValue::Str("")])
        );
        assert_ne!(
            content_hash_row(&[ContentValue::Str("ab")]),
            content_hash_row(&[ContentValue::Bytes(b"ab")])
        );
        // Length-prefixing keeps "ab"+"c" distinct from "a"+"bc".
        assert_ne!(
            content_hash_row(&[ContentValue::Str("ab"), ContentValue::Str("c")]),
            content_hash_row(&[ContentValue::Str("a"), ContentValue::Str("bc")])
        );
    }

    #[test]
    fn hex_round_trips_and_rejects_malformed() {
        let h = content_hash_row(&[ContentValue::Str("x")]);
        let hex = h.to_hex();
        assert_eq!(hex.len(), 64);
        assert_eq!(ContentHash::from_hex(&hex).unwrap(), h);
        assert!(ContentHash::from_hex(&hex.to_uppercase()).is_err());
        assert!(ContentHash::from_hex(&hex[..63]).is_err());
        assert!(ContentHash::from_hex("zz").is_err());
    }

    #[test]
    fn kernel_matches_the_pure_fold_per_row() {
        let c1: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]));
        let c2: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(b"1".as_slice()),
            Some(b"2".as_slice()),
            None,
        ]));
        let hashes = content_hash_columns(&[c1, c2]).unwrap();
        assert_eq!(
            hashes.value(0),
            content_hash_row(&[ContentValue::Str("a"), ContentValue::Bytes(b"1")]).to_hex()
        );
        assert_eq!(
            hashes.value(1),
            content_hash_row(&[ContentValue::Null, ContentValue::Bytes(b"2")]).to_hex()
        );
        assert_eq!(
            hashes.value(2),
            content_hash_row(&[ContentValue::Str("c"), ContentValue::Null]).to_hex()
        );
    }

    #[test]
    fn kernel_refuses_an_unrendered_column() {
        let c: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1, 2]));
        assert!(matches!(
            content_hash_columns(&[c]),
            Err(JammiError::Schema { .. })
        ));
    }
}
