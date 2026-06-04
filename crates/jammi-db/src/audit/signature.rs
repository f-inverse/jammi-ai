//! HMAC-SHA256 signature scheme for audit records.
//!
//! - **Algorithm:** HMAC-SHA256 over [`super::record::canonical_serialize`].
//! - **Secret:** derived per-tenant from a server-held master key and the
//!   tenant id via HKDF-SHA256 with info `"jammi-audit-search-v1"`.
//! - **Master key source:** supplied by the session's
//!   [`SigningKeyStore`](super::SigningKeyStore) (env-backed by default via
//!   [`EnvSigningKeyStore`](super::EnvSigningKeyStore)). Absence of a usable key
//!   is fatal for any signing/verification operation.
//!
//! Derivation is fully determined by the master key and tenant id, so
//! signatures verify identically across server restarts.

use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::error::AuditError;
use super::key_store::SigningKeyStore;
use super::record::{self, PerQueryAudit};

type HmacSha256 = Hmac<Sha256>;

/// HKDF info string binding derivations to this audit scheme version.
const HKDF_INFO: &[u8] = b"jammi-audit-search-v1";

/// Startup check: verify the store can supply a usable master key.
///
/// Intended to be called once when a server process initializes so it can
/// refuse to start when the configured signing-key source has no usable key,
/// rather than failing on the first audit write.
pub fn ensure_master_key_present(store: &dyn SigningKeyStore) -> Result<(), AuditError> {
    store.master_key().map(|_| ())
}

/// Derive the per-tenant signing secret via HKDF-SHA256.
///
/// The tenant id is the HKDF salt and a fixed scheme string the info, so the
/// result is deterministic and isolated per tenant.
pub fn derive_tenant_secret(master: &[u8; 32], tenant_id: &str) -> Result<[u8; 32], AuditError> {
    let hk = Hkdf::<Sha256>::new(Some(tenant_id.as_bytes()), master);
    let mut secret = [0u8; 32];
    // `expand` only errors when the requested length exceeds 255*HashLen; 32
    // bytes is always valid for SHA-256, but we surface any error rather than
    // panic.
    hk.expand(HKDF_INFO, &mut secret)
        .map_err(|e| AuditError::MasterKey(format!("hkdf expand failed: {e}")))?;
    Ok(secret)
}

/// Compute the hex-encoded HMAC-SHA256 of `canonical` under `secret`.
pub fn hmac_sign(canonical: &[u8], secret: &[u8; 32]) -> String {
    // HMAC accepts a key of any length; from a fixed 32-byte array this never
    // fails.
    let mut mac = <HmacSha256 as Mac>::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(canonical);
    hex::encode(mac.finalize().into_bytes())
}

/// Sign a record in place, deriving the secret from its tenant binding.
///
/// The record's `tenant_id` must already be set. The master key is supplied by
/// `store`.
pub fn sign_record(
    record: &mut PerQueryAudit,
    store: &dyn SigningKeyStore,
) -> Result<(), AuditError> {
    let tenant = record
        .tenant_id
        .clone()
        .ok_or(AuditError::NoTenantBinding)?;
    let master = store.master_key()?;
    let secret = derive_tenant_secret(&master, &tenant)?;
    let canonical = record::canonical_serialize(record)?;
    record.signature = hmac_sign(&canonical, &secret);
    Ok(())
}

/// Verify a record's signature against an explicitly provided secret.
pub fn verify(record: &PerQueryAudit, secret: &[u8; 32]) -> Result<(), AuditError> {
    let canonical = record::canonical_serialize(record)?;
    let expected = hmac_sign(&canonical, secret);
    if !constant_time_eq(expected.as_bytes(), record.signature.as_bytes()) {
        return Err(AuditError::SignatureMismatch(record.query_id));
    }
    Ok(())
}

/// Verify a record by re-deriving the per-tenant secret from `store`.
///
/// Convenience for replay-time integrity checks where the caller has a
/// signing-key store and the record carries its `tenant_id`.
pub fn verify_with_store(
    record: &PerQueryAudit,
    store: &dyn SigningKeyStore,
) -> Result<(), AuditError> {
    let tenant = record
        .tenant_id
        .clone()
        .ok_or(AuditError::NoTenantBinding)?;
    let master = store.master_key()?;
    let secret = derive_tenant_secret(&master, &tenant)?;
    verify(record, &secret)
}

/// Constant-time byte comparison to avoid signature-timing side channels.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{EnvSigningKeyStore, MASTER_KEY_ENV};
    use std::sync::Mutex;
    use uuid::Uuid;

    // The master-key env var is process-global; serialize the tests that touch
    // it against each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    const TEST_KEY: &str = "0000000000000000000000000000000000000000000000000000000000000001";

    fn scoped() -> PerQueryAudit {
        let mut r = PerQueryAudit::new(
            Uuid::nil(),
            "m",
            "v",
            serde_json::json!({ "k": 1 }),
            vec!["a".into()],
            vec![0.5],
        )
        .unwrap();
        r.tenant_id = Some("tenant-a".into());
        r
    }

    #[test]
    fn sign_then_verify_roundtrips() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(MASTER_KEY_ENV, TEST_KEY);
        let store = EnvSigningKeyStore;
        let mut r = scoped();
        sign_record(&mut r, &store).unwrap();
        assert!(!r.signature.is_empty());
        verify_with_store(&r, &store).unwrap();
    }

    #[test]
    fn tampering_breaks_signature() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(MASTER_KEY_ENV, TEST_KEY);
        let store = EnvSigningKeyStore;
        let mut r = scoped();
        sign_record(&mut r, &store).unwrap();
        r.model_id = "tampered".into();
        assert!(matches!(
            verify_with_store(&r, &store),
            Err(AuditError::SignatureMismatch(_))
        ));
    }

    #[test]
    fn signing_is_deterministic_across_calls() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(MASTER_KEY_ENV, TEST_KEY);
        let master = EnvSigningKeyStore.master_key().unwrap();
        let secret = derive_tenant_secret(&master, "tenant-a").unwrap();
        let r = scoped();
        let canonical = record::canonical_serialize(&r).unwrap();
        assert_eq!(
            hmac_sign(&canonical, &secret),
            hmac_sign(&canonical, &secret)
        );
    }

    #[test]
    fn different_tenants_get_different_secrets() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(MASTER_KEY_ENV, TEST_KEY);
        let master = EnvSigningKeyStore.master_key().unwrap();
        assert_ne!(
            derive_tenant_secret(&master, "tenant-a").unwrap(),
            derive_tenant_secret(&master, "tenant-b").unwrap()
        );
    }

    #[test]
    fn missing_master_key_makes_present_check_fail() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var(MASTER_KEY_ENV);
        assert!(matches!(
            ensure_master_key_present(&EnvSigningKeyStore),
            Err(AuditError::MasterKey(_))
        ));
    }
}
