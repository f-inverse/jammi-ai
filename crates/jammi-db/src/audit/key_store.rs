//! Audit signing-key port and its env-backed default adapter.
//!
//! The port's single responsibility is to supply the 32-byte audit master key
//! material. HKDF derivation, HMAC computation, canonical serialization, and the
//! constant-time signature compare all stay engine-side in [`super::signature`];
//! the store hands back only the master key. This is the self-host seam: a
//! deployment that holds its master key in a KMS swaps the adapter without
//! reshaping the signing path.

use super::error::AuditError;

/// Environment variable holding the audit master key (32-byte hex).
pub const MASTER_KEY_ENV: &str = "JAMMI_AUDIT_MASTER_KEY";

/// Source of the audit master key.
///
/// Supplies the 32-byte material the engine derives per-tenant signing secrets
/// from. The read is synchronous because the default adapter reads a process
/// environment variable; adapters that must reach a remote key service are free
/// to block internally.
pub trait SigningKeyStore: Send + Sync + 'static {
    /// Return the 32-byte audit master key.
    ///
    /// Returns [`AuditError::MasterKey`] when the configured source has no
    /// usable key (unset, malformed, or wrong length).
    fn master_key(&self) -> Result<[u8; 32], AuditError>;
}

/// Default [`SigningKeyStore`]: reads the master key from `JAMMI_AUDIT_MASTER_KEY`.
///
/// Expects a 64-character hex string decoding to exactly 32 bytes. Absence or
/// malformation is reported as [`AuditError::MasterKey`].
pub struct EnvSigningKeyStore;

impl SigningKeyStore for EnvSigningKeyStore {
    fn master_key(&self) -> Result<[u8; 32], AuditError> {
        let hex_str = std::env::var(MASTER_KEY_ENV)
            .map_err(|_| AuditError::MasterKey(format!("{MASTER_KEY_ENV} is not set")))?;
        let bytes = hex::decode(hex_str.trim())
            .map_err(|e| AuditError::MasterKey(format!("not valid hex: {e}")))?;
        let arr: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
            AuditError::MasterKey(format!(
                "expected 32 bytes (64 hex chars), got {} bytes",
                bytes.len()
            ))
        })?;
        Ok(arr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // The master-key env var is process-global; serialize the tests that touch
    // it against each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    const TEST_KEY: &str = "0000000000000000000000000000000000000000000000000000000000000001";

    #[test]
    fn valid_key_decodes_to_32_bytes() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(MASTER_KEY_ENV, TEST_KEY);
        let key = EnvSigningKeyStore.master_key().unwrap();
        let mut expected = [0u8; 32];
        expected[31] = 1;
        assert_eq!(key, expected);
    }

    #[test]
    fn missing_master_key_is_error() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::remove_var(MASTER_KEY_ENV);
        assert!(matches!(
            EnvSigningKeyStore.master_key(),
            Err(AuditError::MasterKey(_))
        ));
    }

    #[test]
    fn bad_length_master_key_is_error() {
        let _g = ENV_LOCK.lock().unwrap();
        std::env::set_var(MASTER_KEY_ENV, "abcd");
        assert!(matches!(
            EnvSigningKeyStore.master_key(),
            Err(AuditError::MasterKey(_))
        ));
        std::env::remove_var(MASTER_KEY_ENV);
    }
}
