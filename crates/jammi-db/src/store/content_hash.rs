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
    /// lowercase hex characters (a reader never trusts a malformed cell).
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

/// A domain-separated SHA-256 fold: `SHA256(domain ++ parts[0] ++ parts[1]
/// ++ … )`. `domain_hash` adds NO framing of its own beyond concatenating
/// `domain` and `parts` in order — making two distinct ordered part
/// sequences unable to collide by a shifted boundary (`"ab"+"c"` vs
/// `"a"+"bc"`) is the CALLER's responsibility: this crate's one convention
/// is that every PART is already self-delimiting (a fixed-width type tag,
/// or an 8-byte little-endian length immediately followed by its payload —
/// see [`content_hash_row`]'s per-cell encoding below and
/// [`crate::index::peer::RendezvousPlacement`]'s per-field encoding). This
/// is NOT the one primitive every domain-separated hash in this crate folds
/// through — `crate::store::manifest`'s `definition_hash` and
/// `crate::store::version::Version::compute_identity` hand-roll the
/// identical length-prefixed-parts shape under their own domains
/// (`b"jammi.materialization.definition.v1"`, `b"jammi.version.identity.v1"`)
/// without calling this function; only [`content_hash_row`] and
/// [`crate::index::peer::RendezvousPlacement`]'s `rendezvous_score` route
/// through `domain_hash` itself.
///
/// **The domain itself is NOT length-prefixed or otherwise delimited from
/// `parts`.** Two DIFFERENT domain tags are guaranteed never to collide
/// (for ANY choice of parts on either side) if and only if NEITHER is a
/// byte-for-byte PREFIX of the other — when one is, `domain_hash(long,
/// [payload])` is byte-identical to `domain_hash(short, [long[short.len()..],
/// payload])`, since both fold the identical concatenated byte stream.
/// Executed, not asserted: `tests::domain_prefix_is_not_free_by_construction`
/// (below) constructs exactly this collision. Prefix-freedom of the SET of
/// domain tags this crate actually folds a hash under (`domain_hash`'s own
/// two callers above, plus the two hand-rolled folds this doc names) is
/// therefore the real, load-bearing invariant, and it is GATED —
/// `crates/jammi-db/tests/it/domain_hash_prefix_free_gate.rs` walks a real
/// `syn` parse of every tracked `.rs` file under `crates/` (this function is
/// `pub`, so a caller outside this crate is in scope, not just this crate's
/// own `src/`), finds every CALL to `domain_hash` (bare, path-qualified, or
/// reached through a `use ... as` alias), and resolves each call's first
/// argument — a byte-string literal, a string literal's `.as_bytes()`, or a
/// `const`/`static` reference resolved against every such item the same
/// scan finds — into the domain value it folds a hash under. The resolved
/// set's count is asserted so a new tag cannot silently join unreviewed; an
/// argument the scan cannot resolve at all (a local variable, a computed
/// slice) is its own tracked, reviewed finding rather than a silent miss;
/// and every pair of resolved domains is checked for prefix-freedom — never
/// reviewed by eye, never a text/regex scan over one spelling of the
/// literal, and never assumed from the current set's absence of a
/// counterexample.
///
/// Two callers: [`content_hash_row`] (domain [`CONTENT_HASH_DOMAIN`], one
/// already tag+length+payload-framed part per rendered cell — passed
/// VERBATIM as this crate's per-cell encoding:
/// `domain_hash(CONTENT_HASH_DOMAIN, parts)` folds exactly the bytes the
/// per-cell encoding produces, in order) and [`crate::index::peer::
/// RendezvousPlacement`] (domain `b"jammi.placement.v1"`, one length-prefixed
/// part each for `instance_id`, `table` and the segment id).
pub fn domain_hash(domain: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize().into()
}

/// The pure fold: hash one row's rendered cells in column order. Builds one
/// self-delimiting part per cell — `[tag: 1 byte][len(bytes): 8 bytes
/// little-endian][bytes]` — and folds them through [`domain_hash`] under
/// [`CONTENT_HASH_DOMAIN`]; byte-for-byte the same computation this function
/// ran before `domain_hash` was extracted from it (see that function's docs).
pub fn content_hash_row(values: &[ContentValue<'_>]) -> ContentHash {
    let framed: Vec<Vec<u8>> = values
        .iter()
        .map(|v| {
            let (tag, bytes): (u8, &[u8]) = match v {
                ContentValue::Str(s) => (b's', s.as_bytes()),
                ContentValue::Bytes(b) => (b'b', b),
                ContentValue::Null => (b'n', &[]),
            };
            let mut part = Vec::with_capacity(1 + 8 + bytes.len());
            part.push(tag);
            part.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            part.extend_from_slice(bytes);
            part
        })
        .collect();
    let parts: Vec<&[u8]> = framed.iter().map(Vec::as_slice).collect();
    ContentHash(domain_hash(CONTENT_HASH_DOMAIN, &parts))
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

    /// `content_hash_row`, built on the [`domain_hash`] primitive, is
    /// byte-identical to an independent fold reimplemented here (SHA-256 over
    /// the domain tag, then `[tag][len(bytes) as u64 LE][bytes]` per value,
    /// with NO call to `domain_hash`), over cases spanning every tag and several
    /// boundary-shift pairs.
    #[test]
    fn content_hash_row_is_byte_identical_to_the_pre_extraction_fold() {
        fn pre_extraction_fold(values: &[ContentValue<'_>]) -> ContentHash {
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

        let cases: Vec<Vec<ContentValue<'_>>> = vec![
            vec![ContentValue::Str("hello")],
            vec![ContentValue::Bytes(b"hello")],
            vec![ContentValue::Null],
            vec![ContentValue::Str("ab"), ContentValue::Str("c")],
            vec![ContentValue::Str("a"), ContentValue::Str("bc")],
            vec![
                ContentValue::Str("x"),
                ContentValue::Null,
                ContentValue::Bytes(b"y"),
            ],
            vec![],
        ];
        for case in &cases {
            assert_eq!(
                content_hash_row(case),
                pre_extraction_fold(case),
                "case {case:?} diverged after extracting domain_hash"
            );
        }
    }

    /// A third domain tag never collides with [`CONTENT_HASH_DOMAIN`] or
    /// RENDEZVOUS's `b"jammi.placement.v1"`, over the SAME parts, for these
    /// THREE CONCRETE domains — this is NOT a general "any two domains never
    /// collide" claim (see `domain_prefix_is_not_free_by_construction`
    /// immediately below for the general counterexample: two domains where
    /// one is a byte-prefix of the other DO collide, for a suitable choice
    /// of parts). What this test excludes is exactly the shape
    /// `crates/jammi-db/tests/it/domain_hash_prefix_free_gate.rs` verifies
    /// holds over every domain this crate actually folds a hash under: none
    /// of `CONTENT_HASH_DOMAIN`, `b"jammi.placement.v1"`, and an unrelated
    /// third literal is a prefix of another, so they cannot collide this way
    /// — pairwise prefix-freedom is the real, narrower, gated invariant.
    #[test]
    fn domain_hash_separates_by_domain_over_identical_parts() {
        let parts: &[&[u8]] = &[b"same", b"parts"];
        let a = domain_hash(CONTENT_HASH_DOMAIN, parts);
        let b = domain_hash(b"jammi.placement.v1", parts);
        let c = domain_hash(b"a.third.domain", parts);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    /// The GENERAL counterexample `domain_hash`'s own doc now names: when
    /// one domain is a byte-for-byte PREFIX of another, the two DO collide
    /// for a suitable choice of parts, because `domain_hash` adds no framing
    /// around the domain itself — `domain_hash(long, [rest, ...])` and
    /// `domain_hash(short, [rest_of_long, ...])` fold the identical byte
    /// stream when `long == short ++ rest_of_long`. `short_domain` is
    /// SLICED from `long_domain` at runtime (never a second `b"jammi...."`
    /// literal) so this fixture cannot itself be mistaken for a real domain
    /// by `domain_hash_prefix_free_gate.rs`'s source scan — it demonstrates
    /// the vulnerability without adding a new literal to the reviewed set.
    /// Executed, not asserted: this is exactly why prefix-freedom of the
    /// domain SET this crate actually uses is the real precondition, gated
    /// rather than reviewed by eye.
    #[test]
    fn domain_prefix_is_not_free_by_construction() {
        let long_domain: &[u8] = b"jammi.placement.v1";
        let short_domain: &[u8] = &long_domain[..long_domain.len() - 1]; // "jammi.placement.v"
        let extra = &long_domain[long_domain.len() - 1..]; // "1"
        let payload: &[u8] = b"\x31PAYLOAD";

        let via_long_domain = domain_hash(long_domain, &[payload]);
        // `short_domain ++ extra ++ payload == long_domain ++ payload` exactly,
        // since `long_domain == short_domain ++ extra`.
        let via_short_domain_and_extra_part = domain_hash(short_domain, &[extra, payload]);
        assert_eq!(
            via_long_domain, via_short_domain_and_extra_part,
            "a prefix-related domain pair must collide by construction — this is the \
             counterexample that refutes 'a third domain tag can never collide'"
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
