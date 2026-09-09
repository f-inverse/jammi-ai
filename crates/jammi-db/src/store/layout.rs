//! The tenant-prefixed key layout every result-table and sidecar object
//! lives under, and the derivation rules that keep a table's own siblings
//! co-located with it however the store root moves.
//!
//! Two rules, enforced in exactly one place each:
//!
//! 1. **Tenant attribution lives in the key.** A GLOBAL (untenanted) result
//!    table lands under `{root}/_global/{table}.parquet`; a tenant's table
//!    lands under `{root}/{tenant-uuid}/{table}.parquet`. [`TenantSegment`]
//!    is the only place that string is built ([`TenantSegment::of`]) or read
//!    back ([`TenantSegment::parse`]) — a listing pass (`reconcile`) that
//!    wants to attribute a raw object key to a tenant goes through the same
//!    parser a writer's key was built with, so the two can never drift.
//! 2. **A table's siblings derive from ITS OWN `parquet_path`, never from the
//!    store's current root.** [`segment_url`] and [`sidecar_url`] take the
//!    row's `parquet_path` and swap in a sibling name — so a table created
//!    under yesterday's root still resolves its segments and sidecars
//!    correctly even if the store's root configuration changes later.

use std::str::FromStr;

use crate::error::{JammiError, Result};
use crate::storage::StorageUrl;
use crate::tenant::TenantId;

/// The reserved segment naming a GLOBAL (untenanted) result table or artifact
/// prefix.
const GLOBAL_SEGMENT: &str = "_global";

/// The tenant-attribution segment of a result-table or artifact key.
///
/// [`Self::of`] and [`Self::parse`] are exact inverses on every value
/// [`Self::of`] can produce: `of(t)` followed by `parse` returns `Some(t)`
/// (K7/family M — doc-parity for a codec, not a status enum, but the same
/// round-trip discipline). [`Self::parse`] additionally REJECTS every UUID
/// spelling [`Self::of`] would never emit (braced, urn, unhyphenated
/// "simple") — a raw key found by `crate::storage::JammiObjectStore::list`
/// with a non-canonical UUID segment was never written by this engine's own
/// writer, so it is `unattributed`, not silently coerced into a tenant's
/// prefix.
pub struct TenantSegment;

impl TenantSegment {
    /// The key segment a `result_tables` / artifact row's tenant maps to:
    /// `_global` for `None` (a GLOBAL row), otherwise the tenant's canonical
    /// hyphenated lowercase UUID string (`TenantId`'s `Display`).
    pub fn of(tenant: Option<&TenantId>) -> String {
        match tenant {
            Some(t) => t.to_string(),
            None => GLOBAL_SEGMENT.to_string(),
        }
    }

    /// Parse a raw key segment back into the tenant it names, or `None` if
    /// the segment is not one [`Self::of`] could have produced.
    ///
    /// Returns `Some(None)` for exactly `_global`, `Some(Some(t))` for a
    /// segment that parses as a [`TenantId`] AND round-trips back to the
    /// identical string through [`TenantId`]'s `Display` (the canonical
    /// hyphenated lowercase form) — this is what rejects a braced
    /// (`{uuid}`), `urn:uuid:...`, or unhyphenated "simple" spelling that
    /// [`uuid::Uuid::from_str`] would otherwise happily parse: those forms
    /// never round-trip to themselves, so they parse to `None` here even
    /// though the bytes they name are a valid UUID. Anything else is `None`
    /// (not an attributable segment at all).
    pub fn parse(segment: &str) -> Option<Option<TenantId>> {
        if segment == GLOBAL_SEGMENT {
            return Some(None);
        }
        let parsed = TenantId::from_str(segment).ok()?;
        if parsed.to_string() == segment {
            Some(Some(parsed))
        } else {
            None
        }
    }
}

/// The key a new result table's Parquet object lands at:
/// `{root}/{seg}/{table_name}.parquet`. `seg` is a [`TenantSegment::of`]
/// value; the caller reads the tenant once (from the catalog binding in
/// force at `create_table`) and passes the same segment for both the row's
/// `tenant_id` and this key, so the two can never disagree.
pub fn result_table_url(root: &StorageUrl, seg: &str, table_name: &str) -> Result<StorageUrl> {
    let root_str = root.as_str().trim_end_matches('/');
    Ok(StorageUrl::parse(&format!(
        "{root_str}/{seg}/{table_name}.parquet"
    ))?)
}

/// A sibling URL of a result table's Parquet object: the same parent prefix,
/// a different object name. Every sidecar and segment key derives from the
/// row's own `parquet_path` this way — never from the store's current root
/// — so a table's objects stay co-located however the root moves.
pub fn sibling_url(parquet_url: &StorageUrl, name: &str) -> Result<StorageUrl> {
    let s = parquet_url.as_str();
    let (parent, _) = s.rsplit_once('/').ok_or_else(|| {
        JammiError::Config(format!("result-table URL '{s}' has no parent prefix"))
    })?;
    Ok(StorageUrl::parse(&format!("{parent}/{name}"))?)
}

/// The base object name of a result table's Parquet URL with the
/// `.parquet` extension stripped — the stem [`segment_url`] embeds so a
/// segment's bundle name carries its owning table's identity.
fn parquet_stem(parquet_url: &StorageUrl) -> Result<&str> {
    let s = parquet_url.as_str();
    let name = s.rsplit('/').next().unwrap_or(s);
    name.strip_suffix(".parquet").ok_or_else(|| {
        JammiError::Config(format!(
            "result-table URL '{s}' does not name a `.parquet` object"
        ))
    })
}

/// The base object name of ANY url with its trailing `.ext` stripped — the
/// stem [`sidecar_url`] swaps a different extension onto. Generic over
/// what the extension actually is (unlike [`parquet_stem`]) so it works on
/// both a Parquet URL (`table.parquet` → `table`) and a segment bundle URL
/// (`table__seg0.idx` → `table__seg0`), matching
/// [`crate::storage::JammiObjectStore::sibling_path`]'s own extension-swap
/// rule.
fn any_stem(url: &StorageUrl) -> Result<&str> {
    let s = url.as_str();
    let name = s.rsplit('/').next().unwrap_or(s);
    name.rsplit_once('.')
        .map(|(stem, _)| stem)
        .ok_or_else(|| JammiError::Config(format!("URL '{s}' has no extension to swap")))
}

/// The key of ANN segment `n`'s index bundle base, derived from the table's
/// own `parquet_path`: `{parent}/{table_name}__seg{n}.idx`.
pub fn segment_url(parquet_url: &StorageUrl, n: i64) -> Result<StorageUrl> {
    let base = parquet_stem(parquet_url)?;
    sibling_url(parquet_url, &format!("{base}__seg{n}.idx"))
}

/// The key of a sidecar object sharing a URL's own stem and parent prefix
/// but a different extension (e.g. a Parquet's `materialization.json`, or a
/// segment bundle's `usearch`/`rowmap`/`manifest.json`/…):
/// `{parent}/{stem}.{ext}`.
pub fn sidecar_url(url: &StorageUrl, ext: &str) -> Result<StorageUrl> {
    sibling_url(url, &format!("{}.{ext}", any_stem(url)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tenant(seed: u8) -> TenantId {
        let bytes = [seed; 16];
        TenantId::from_uuid(uuid::Builder::from_bytes(bytes).into_uuid()).unwrap()
    }

    #[test]
    fn global_segment_round_trips() {
        let seg = TenantSegment::of(None);
        assert_eq!(seg, "_global");
        assert_eq!(TenantSegment::parse(&seg), Some(None));
    }

    #[test]
    fn tenant_segment_round_trips() {
        let t = tenant(7);
        let seg = TenantSegment::of(Some(&t));
        assert_eq!(seg, t.to_string());
        assert_eq!(TenantSegment::parse(&seg), Some(Some(t)));
    }

    #[test]
    fn parse_rejects_braced_uuid_form() {
        let t = tenant(9);
        let braced = format!("{{{t}}}");
        assert_eq!(
            TenantSegment::parse(&braced),
            None,
            "a braced UUID is a valid Uuid::from_str input but never a segment `of` emits"
        );
    }

    #[test]
    fn parse_rejects_urn_uuid_form() {
        let t = tenant(11);
        let urn = format!("urn:uuid:{t}");
        assert_eq!(TenantSegment::parse(&urn), None);
    }

    #[test]
    fn parse_rejects_simple_unhyphenated_form() {
        let t = tenant(13);
        let simple = t.as_uuid().simple().to_string();
        assert_eq!(
            TenantSegment::parse(&simple),
            None,
            "the unhyphenated 'simple' UUID form is never emitted by `of`"
        );
    }

    #[test]
    fn parse_rejects_garbage() {
        assert_eq!(TenantSegment::parse("not-a-uuid-at-all"), None);
        assert_eq!(TenantSegment::parse(""), None);
    }

    #[test]
    fn result_table_url_lands_under_root_and_segment() {
        let root = StorageUrl::parse("memory:///jammi_db").unwrap();
        let url = result_table_url(&root, "_global", "articles__embed__x__ts").unwrap();
        assert_eq!(
            url.as_str(),
            "memory:///jammi_db/_global/articles__embed__x__ts.parquet"
        );
    }

    #[test]
    fn result_table_url_tolerates_trailing_slash_on_root() {
        let root = StorageUrl::parse("memory:///jammi_db/").unwrap();
        let url = result_table_url(&root, "_global", "t").unwrap();
        assert_eq!(url.as_str(), "memory:///jammi_db/_global/t.parquet");
    }

    #[test]
    fn segment_url_embeds_table_stem_and_segment_id() {
        let parquet = StorageUrl::parse("memory:///root/_global/table_a.parquet").unwrap();
        let seg = segment_url(&parquet, 3).unwrap();
        assert_eq!(seg.as_str(), "memory:///root/_global/table_a__seg3.idx");
    }

    #[test]
    fn sidecar_url_swaps_extension_keeping_stem() {
        let parquet = StorageUrl::parse("memory:///root/_global/table_a.parquet").unwrap();
        let sidecar = sidecar_url(&parquet, "materialization.json").unwrap();
        assert_eq!(
            sidecar.as_str(),
            "memory:///root/_global/table_a.materialization.json"
        );
    }

    #[test]
    fn sidecar_url_of_a_segment_url_derives_ann_siblings() {
        let parquet = StorageUrl::parse("memory:///root/t/table_a.parquet").unwrap();
        let seg = segment_url(&parquet, 0).unwrap();
        let usearch = sidecar_url(&seg, "usearch").unwrap();
        assert_eq!(usearch.as_str(), "memory:///root/t/table_a__seg0.usearch");
    }
}
