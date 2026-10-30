//! What the database sources share: the URL a database source connects
//! through, the TLS mode it connects with, and the file its CA certificate is
//! read from.

use std::path::PathBuf;

use crate::config::secret::read_placeholders;
use crate::error::{JammiError, Result};
use crate::source::{url_writes_its_password, CaCertificate, SourceConnection};

/// The connection URL of a database source, which it cannot connect without:
/// the URL as registered, each `${NAME}` in it, a `JAMMI_SOURCE_` variable,
/// replaced by that variable of this process's environment, read now, at
/// every connection (#756). The catalog keeps the URL as registered, so it
/// never holds a password a placeholder names.
///
/// A URL registered before placeholders existed, whose password may hold
/// `$$`, connects as it always did; one that writes its password is the
/// catalog holding that password, so it still connects, and says so.
pub(crate) fn url(source_id: &str, connection: &SourceConnection) -> Result<String> {
    let url = connection
        .url
        .as_deref()
        .ok_or_else(|| JammiError::Source {
            source_id: source_id.into(),
            message: "a database source requires a connection URL".into(),
        })?;
    if url_writes_its_password(url) {
        tracing::warn!(
            source_id,
            "the catalog holds this source's password in its URL; register the source again \
             with the password named by a ${{JAMMI_SOURCE_<NAME>}} placeholder, and the \
             catalog will not"
        );
    }
    read_placeholders(url).map_err(|why| JammiError::Source {
        source_id: source_id.into(),
        message: format!("its URL's placeholder: {why}"),
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

    fn connection(url: &str) -> SourceConnection {
        SourceConnection {
            url: Some(url.into()),
            ..Default::default()
        }
    }

    #[test]
    fn a_placeholder_is_read_from_the_environment_and_an_unset_one_is_named() {
        let var = format!("JAMMI_SOURCE_DATABASE_URL_TEST_{}", std::process::id());
        std::env::set_var(&var, "s3cr%40t");
        let named = connection(&format!("postgresql://ada:${{{var}}}@db/sales"));
        assert_eq!(
            url("pg", &named).unwrap(),
            "postgresql://ada:s3cr%40t@db/sales"
        );
        // A URL with no placeholder is the URL, `$$` and all.
        assert_eq!(
            url("pg", &connection("postgresql://ada:pa$$word@db/sales")).unwrap(),
            "postgresql://ada:pa$$word@db/sales"
        );

        std::env::remove_var(&var);
        let unset = url("pg", &named).unwrap_err().to_string();
        assert!(unset.contains(&format!("`${{{var}}}`")), "{unset}");
        assert!(unset.contains("not set"), "{unset}");
    }

    #[test]
    fn a_variable_set_aside_for_no_source_is_never_read() {
        let var = format!("JAMMI_DATABASE_URL_TEST_{}", std::process::id());
        std::env::set_var(&var, "the-servers-own-secret");
        let named = connection(&format!("postgresql://ada:${{{var}}}@db/sales"));
        let refused = url("pg", &named).unwrap_err().to_string();
        std::env::remove_var(&var);
        assert!(refused.contains("JAMMI_SOURCE_"), "{refused}");
        assert!(!refused.contains("the-servers-own-secret"), "{refused}");
    }
}
