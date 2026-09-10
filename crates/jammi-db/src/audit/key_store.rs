//! Audit signing-key port and its env-backed default adapter.
//!
//! The port's single responsibility is to supply the 32-byte audit master key
//! material. HKDF derivation, HMAC computation, canonical serialization, and the
//! constant-time signature compare all stay engine-side in [`super::signature`];
//! the store hands back only the master key. This is the self-host seam: a
//! deployment that holds its master key in a KMS swaps the adapter without
//! reshaping the signing path.
//!
//! Two adapters ship: [`EnvSigningKeyStore`] (the `JAMMI_AUDIT_MASTER_KEY`
//! variable) and [`FileSigningKeyStore`] (a mounted secret file). Both expect
//! the same material — 64 hex characters decoding to 32 bytes — so a key
//! moved from one to the other signs identically.

use std::path::{Path, PathBuf};

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
        decode_master_key(&hex_str)
    }
}

/// [`SigningKeyStore`] reading the master key from a file — the shape a
/// container secret mount (`/run/secrets/…`) or a systemd credential hands
/// a process.
///
/// Selected by `[signing_key.file]` in the config
/// ([`crate::config::SigningKeyConfig::File`]). The file holds exactly what
/// [`EnvSigningKeyStore`] expects in its variable: a 64-character hex string
/// decoding to 32 bytes. It is read through the config's one file-secret
/// rule ([`crate::config::secret::read_secret_file`]) — UTF-8, one trailing
/// newline trimmed — and then decoded identically, so the same key bytes
/// produce the same signatures whichever store supplied them.
///
/// The read happens per [`SigningKeyStore::master_key`] call rather than
/// once at construction: a rotated mount takes effect without a restart, and
/// a missing or malformed file reports as [`AuditError::MasterKey`] at the
/// signing site, exactly as an unset variable does.
pub struct FileSigningKeyStore {
    path: PathBuf,
}

impl FileSigningKeyStore {
    /// A store reading the hex-encoded master key from `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// The file this store reads.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl SigningKeyStore for FileSigningKeyStore {
    fn master_key(&self) -> Result<[u8; 32], AuditError> {
        let secret = crate::config::secret::read_secret_file(&self.path)
            .map_err(|e| AuditError::MasterKey(e.to_string()))?;
        decode_master_key(secret.expose())
    }
}

/// The one decoding rule both stores share: surrounding whitespace trimmed,
/// hex-decoded, exactly 32 bytes.
fn decode_master_key(hex_str: &str) -> Result<[u8; 32], AuditError> {
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

    fn scoped_record() -> crate::audit::PerQueryAudit {
        let mut r = crate::audit::PerQueryAudit::new(
            uuid::Uuid::nil(),
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

    /// The same key bytes reached through the file store and the env store
    /// yield the same master key and the same signature over the same
    /// record — the file store is a source swap, not a different scheme.
    #[test]
    fn file_signing_key_store_signs_identically_to_env_store() {
        let env = lock();
        env.set(TEST_KEY);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("audit-master-key");
        // The trailing newline every `echo`/editor appends is tolerated.
        std::fs::write(&path, format!("{TEST_KEY}\n")).unwrap();
        let file_store = FileSigningKeyStore::new(&path);
        assert_eq!(file_store.path(), path.as_path());

        assert_eq!(
            file_store.master_key().unwrap(),
            EnvSigningKeyStore.master_key().unwrap()
        );

        let mut via_file = scoped_record();
        let mut via_env = scoped_record();
        crate::audit::sign_record(&mut via_file, &file_store).unwrap();
        crate::audit::sign_record(&mut via_env, &EnvSigningKeyStore).unwrap();
        assert!(!via_file.signature.is_empty());
        assert_eq!(via_file.signature, via_env.signature);
        crate::audit::verify_with_store(&via_file, &EnvSigningKeyStore).unwrap();
        crate::audit::verify_with_store(&via_env, &file_store).unwrap();
    }

    #[test]
    fn file_signing_key_store_missing_or_malformed_file_is_master_key_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("absent");
        match FileSigningKeyStore::new(&missing).master_key() {
            Err(AuditError::MasterKey(msg)) => {
                assert!(msg.contains(missing.to_str().unwrap()), "{msg}")
            }
            other => panic!("expected MasterKey error, got {other:?}"),
        }
        let short = dir.path().join("short");
        std::fs::write(&short, "abcd\n").unwrap();
        assert!(matches!(
            FileSigningKeyStore::new(&short).master_key(),
            Err(AuditError::MasterKey(_))
        ));
    }
}
