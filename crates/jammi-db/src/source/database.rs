//! What the database sources share: the URL a database source connects
//! through, the TLS mode it connects with, and the file its CA certificate is
//! read from.

use std::path::PathBuf;

use crate::error::{JammiError, Result};
use crate::source::{CaCertificate, SourceConnection};

/// The connection URL of a database source, which it cannot connect without.
pub(crate) fn url<'a>(source_id: &str, connection: &'a SourceConnection) -> Result<&'a str> {
    connection.url.as_deref().ok_or_else(|| JammiError::Source {
        source_id: source_id.into(),
        message: "a database source requires a connection URL".into(),
    })
}

/// A driver's TLS modes, as its URL's `sslmode` names them.
pub(crate) struct TlsModes {
    /// The mode a URL that names none connects with.
    pub(crate) default: &'static str,
    /// The modes that verify the server's certificate, the first of them the
    /// one a source holding a CA certificate connects with when its URL names
    /// no mode.
    pub(crate) verifying: &'static [&'static str],
}

impl TlsModes {
    /// The mode a source connects with: the one its URL names, or the
    /// driver's default. A CA certificate is there to verify the server, so a
    /// source holding one connects with a mode that does, and a URL naming
    /// one that does not is refused rather than connected unverified.
    pub(crate) fn mode(
        &self,
        named: Option<&str>,
        ca: bool,
    ) -> std::result::Result<String, String> {
        match (named, ca) {
            (Some(mode), true) if !self.verifying.contains(&mode) => Err(format!(
                "a CA certificate verifies the server and `sslmode={mode}` does not; name {} or no mode",
                self.verifying
                    .iter()
                    .map(|m| format!("`{m}`"))
                    .collect::<Vec<_>>()
                    .join(" or ")
            )),
            (Some(mode), _) => Ok(mode.into()),
            (None, true) => Ok(self.verifying[0].into()),
            (None, false) => Ok(self.default.into()),
        }
    }
}

/// The file the driver reads `ca` from: written under its content's digest in
/// this process's temporary directory, so every build of a source holding
/// that CA, and a driver that rereads it at each connection, finds the same
/// file while the process runs. Written whole, then moved into place.
pub(crate) async fn ca_file(source_id: &str, ca: &CaCertificate) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};

    let dir = std::env::temp_dir().join("jammi-source-ca");
    let path = dir.join(format!(
        "{}.pem",
        hex::encode(Sha256::digest(ca.pem().as_bytes()))
    ));
    let failed = |e: std::io::Error| JammiError::Source {
        source_id: source_id.into(),
        message: format!("writing its CA certificate to {}: {e}", path.display()),
    };
    tokio::fs::create_dir_all(&dir).await.map_err(failed)?;
    let scratch = dir.join(format!("{}.pem.part", uuid::Uuid::new_v4().simple()));
    tokio::fs::write(&scratch, ca.pem()).await.map_err(failed)?;
    tokio::fs::rename(&scratch, &path).await.map_err(failed)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODES: TlsModes = TlsModes {
        default: "prefer",
        verifying: &["verify-full", "verify-ca"],
    };

    #[test]
    fn a_ca_certificate_connects_verified() {
        assert_eq!(MODES.mode(None, false).unwrap(), "prefer");
        assert_eq!(MODES.mode(None, true).unwrap(), "verify-full");
        assert_eq!(MODES.mode(Some("verify-ca"), true).unwrap(), "verify-ca");
        assert_eq!(MODES.mode(Some("require"), false).unwrap(), "require");
        let refused = MODES.mode(Some("require"), true).unwrap_err();
        assert!(refused.contains("`sslmode=require`"), "{refused}");
    }
}
