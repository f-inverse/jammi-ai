//! Opaque identifier for a unit of logical isolation inside one Jammi
//! engine process.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{JammiError, Result};

/// Opaque identifier for a unit of logical isolation inside one Jammi
/// engine process.
///
/// The engine never mints a `TenantId`. Auth, identity, billing, and quota
/// systems live above the engine and pass tenant identity in; the engine
/// stores and scopes by it.
///
/// `Uuid::nil()` is rejected at construction: the absent meaning is carried
/// by `Option<TenantId>`, not by a sentinel UUID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct TenantId(Uuid);

impl TenantId {
    /// Wrap an externally minted UUID.
    ///
    /// Returns an error if `uuid` is the nil UUID (all zeroes).
    pub fn from_uuid(uuid: Uuid) -> Result<Self> {
        if uuid.is_nil() {
            return Err(JammiError::Tenant(
                "tenant id must not be the nil UUID".into(),
            ));
        }
        Ok(Self(uuid))
    }

    /// Extract the underlying UUID.
    pub const fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl FromStr for TenantId {
    type Err = JammiError;

    fn from_str(s: &str) -> Result<Self> {
        let uuid =
            Uuid::from_str(s).map_err(|e| JammiError::Tenant(format!("tenant id '{s}': {e}")))?;
        Self::from_uuid(uuid)
    }
}

impl TryFrom<String> for TenantId {
    type Error = JammiError;

    fn try_from(s: String) -> Result<Self> {
        Self::from_str(&s)
    }
}

impl From<TenantId> for String {
    fn from(t: TenantId) -> String {
        t.to_string()
    }
}

/// Session-scoped tenant binding propagated through DataFusion's
/// `SessionConfig` extension store.
///
/// Read by [`crate::tenant_scope::TenantScopeAnalyzerRule`] during plan
/// analysis to inject `tenant_id = $current OR tenant_id IS NULL` predicates
/// on every scanned table whose schema declares the column; read by
/// [`crate::catalog::backend::Transaction::set_tenant`] to bind a tenant for
/// the duration of one transaction (used by the mutable-table sink to enforce
/// the write-side guard).
///
/// `Scoped(t)` filters to tenant `t` and also surfaces engine-default rows
/// (`tenant_id IS NULL`). `Unscoped` shows only `tenant_id IS NULL` rows —
/// the no-op identity on a pre-Phase-3 population.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TenantContext {
    /// No tenant binding; reads see only globally-scoped rows.
    Unscoped,
    /// Tenant binding active; reads see `tenant_id = t` ∪ `tenant_id IS NULL`.
    Scoped(TenantId),
}

impl TenantContext {
    /// Convert an optional tenant to the matching context value.
    pub fn from_option(t: Option<TenantId>) -> Self {
        match t {
            Some(t) => TenantContext::Scoped(t),
            None => TenantContext::Unscoped,
        }
    }

    /// The tenant id this context binds, if any.
    pub fn tenant(&self) -> Option<TenantId> {
        match self {
            TenantContext::Scoped(t) => Some(*t),
            TenantContext::Unscoped => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_uuid_accepts_non_nil() {
        let u = Uuid::parse_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
        assert_eq!(TenantId::from_uuid(u).unwrap().as_uuid(), u);
    }

    #[test]
    fn from_uuid_rejects_nil() {
        let err = TenantId::from_uuid(Uuid::nil()).unwrap_err();
        match err {
            JammiError::Tenant(m) => assert!(m.contains("nil")),
            other => panic!("expected Tenant error, got {other:?}"),
        }
    }

    #[test]
    fn from_str_rejects_nil_string() {
        let err = TenantId::from_str("00000000-0000-0000-0000-000000000000").unwrap_err();
        assert!(matches!(err, JammiError::Tenant(_)));
    }

    #[test]
    fn from_str_rejects_malformed() {
        assert!(TenantId::from_str("not-a-uuid").is_err());
    }

    #[test]
    fn display_roundtrip_via_from_str() {
        let original = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
        let s = original.to_string();
        let parsed = TenantId::from_str(&s).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn display_lowercase_hyphenated() {
        let t = TenantId::from_str("01906C83-D4C8-7E10-9C4F-3B6F7C5A8E9A").unwrap();
        assert_eq!(t.to_string(), "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a");
    }

    #[test]
    fn serde_roundtrip_through_string() {
        let t = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "\"01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a\"");
        let parsed: TenantId = serde_json::from_str(&json).unwrap();
        assert_eq!(t, parsed);
    }

    #[test]
    fn serde_deserialize_rejects_nil() {
        let r: std::result::Result<TenantId, _> =
            serde_json::from_str("\"00000000-0000-0000-0000-000000000000\"");
        assert!(r.is_err());
    }

    #[test]
    fn try_from_string_delegates_to_from_str() {
        let s = String::from("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a");
        let t: TenantId = s.clone().try_into().unwrap();
        assert_eq!(t.to_string(), s);
    }

    #[test]
    fn into_string_via_display() {
        let t = TenantId::from_str("01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a").unwrap();
        let s: String = t.into();
        assert_eq!(s, "01906c83-d4c8-7e10-9c4f-3b6f7c5a8e9a");
    }
}
