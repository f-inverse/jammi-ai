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

/// Test-only serialization for the process-global master-key environment
/// variable, declared beside the variable it guards.
///
/// Every test module that mutates [`MASTER_KEY_ENV`] must contend on *this*
/// mutex. A second lock declared in another module serializes that module
/// against itself while racing every other module — which reads as correct at
/// each individual call site and is not.
#[cfg(test)]
pub(crate) mod test_env {
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// A 32-byte all-zero-but-one master key in hex, shared by every test that
    /// needs a valid one.
    pub(crate) const TEST_KEY: &str =
        "0000000000000000000000000000000000000000000000000000000000000001";

    /// Exclusive access to [`super::MASTER_KEY_ENV`], held for the guard's
    /// lifetime.
    ///
    /// Mutation is reachable *only* through this guard's methods, so a test
    /// cannot touch the variable without first serializing on the lock — the
    /// property is enforced by construction rather than by every call site
    /// remembering to take a lock first. That is the whole point: the previous
    /// arrangement had two separate mutexes and each call site looked correct
    /// on its own.
    pub(crate) struct MasterKeyEnv(#[allow(dead_code)] MutexGuard<'static, ()>);

    impl MasterKeyEnv {
        /// Set the master key for the duration of this guard.
        pub(crate) fn set(&self, value: &str) {
            std::env::set_var(super::MASTER_KEY_ENV, value);
        }

        /// Unset the master key for the duration of this guard.
        pub(crate) fn clear(&self) {
            std::env::remove_var(super::MASTER_KEY_ENV);
        }
    }

    /// Acquire exclusive access to [`super::MASTER_KEY_ENV`].
    ///
    /// Lock poisoning is recovered rather than propagated: this mutex orders
    /// mutation of an environment variable and guards no invariant of its own,
    /// so a test that panics while holding it leaves nothing inconsistent
    /// behind. Propagating the poison would convert one genuine failure into a
    /// cascade across every test sharing the lock, burying the real one.
    pub(crate) fn lock() -> MasterKeyEnv {
        MasterKeyEnv(
            ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::test_env::{lock, TEST_KEY};
    use super::*;

    #[test]
    fn valid_key_decodes_to_32_bytes() {
        let env = lock();
        env.set(TEST_KEY);
        let key = EnvSigningKeyStore.master_key().unwrap();
        let mut expected = [0u8; 32];
        expected[31] = 1;
        assert_eq!(key, expected);
    }

    #[test]
    fn missing_master_key_is_error() {
        let env = lock();
        env.clear();
        assert!(matches!(
            EnvSigningKeyStore.master_key(),
            Err(AuditError::MasterKey(_))
        ));
    }

    #[test]
    fn bad_length_master_key_is_error() {
        let env = lock();
        env.set("abcd");
        assert!(matches!(
            EnvSigningKeyStore.master_key(),
            Err(AuditError::MasterKey(_))
        ));
        env.clear();
    }
}
