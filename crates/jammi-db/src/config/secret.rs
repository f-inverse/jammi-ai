//! Secret-valued configuration: the source spellings a deployer may write and
//! the redacted value the engine holds.
//!
//! A secret reaches the config in one of two shapes:
//!
//! - **inline** — the value itself: `url = "postgres://u:p@h/db"`;
//! - **file** — a table naming the file that holds it:
//!   `url = { file = "/run/secrets/pg-url" }`. The file is read at load and
//!   exactly one trailing newline is trimmed (the newline every editor and
//!   `echo` appends; a secret that legitimately ends in a newline keeps the
//!   rest of them).
//!
//! [`SecretSource`] is the parsed spelling; [`Secret`] is the resolved value.
//!
//! # Resolution happens at deserialization
//!
//! Every secret-typed field of [`crate::config::JammiConfig`] holds a
//! [`Secret`] directly. `Secret`'s `Deserialize` accepts the same shapes as
//! `SecretSource` and resolves the file form eagerly, so a missing or
//! unreadable secret file fails `JammiConfig::load` — the one place a
//! deployment error should surface — instead of the first Postgres pool, NATS
//! connect, or audit signature deep in a server startup. The alternative
//! (fields hold `SecretSource`, a parallel resolved view is built by the
//! loader) would let a config parse cleanly with a secret nobody can read; a
//! parse that succeeds is a config that works.
//!
//! # What never happens
//!
//! - A `Secret` never renders through `Debug`/`Display`: both print
//!   `Secret(***)`, so a `Debug` of the whole config, a tracing span, or a
//!   panic payload carries no secret. `Secret` itself has no blanket
//!   `Serialize` impl — the only way a `Secret` becomes plaintext again is
//!   [`Secret::expose`] or the standalone [`serialize_exposed`] function,
//!   and the latter is reachable only through an explicit
//!   `#[serde(serialize_with = "…")]` a field opts into by name (the
//!   persisted `crate::storage::config` credential fields, whose whole
//!   point is to round-trip through the catalog's `sources.options` JSON) —
//!   never by deriving `Serialize` on a struct that merely contains a
//!   `Secret`.
//! - A secret string that is *itself* the text `{ file = "…" }` is refused
//!   rather than passed along as an inline value: that is what an operator
//!   gets by quoting the file form in TOML or pasting it into a plain env
//!   var, and it would otherwise become a literal connection string that
//!   fails much later with an unrelated error. The refusal names the two
//!   spellings that work (the unquoted TOML table, or the `__FILE` env
//!   spelling).

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::de::{self, Deserialize, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde::Serializer;

use crate::error::{JammiError, Result};

/// The accepted shapes, worded once for every error that lists them.
const ACCEPTED_SHAPES: &str =
    "a string (the secret inline) or a table with exactly the key `file` naming a file to read";

/// Where a secret comes from, as written in the config.
///
/// Deserialises from a plain string (→ [`SecretSource::Inline`]) or a table
/// with exactly one key, `file` (→ [`SecretSource::File`]); every other shape
/// is a typed error naming the accepted ones. Call [`SecretSource::resolve`]
/// to read the file form into a [`Secret`].
///
/// `Debug` never prints an inline value — the source holds the secret before
/// resolution, so it is as sensitive as the resolved [`Secret`].
#[derive(Clone, PartialEq, Eq)]
pub enum SecretSource {
    /// The secret written directly in the config.
    Inline(String),
    /// The secret lives in this file; read at [`SecretSource::resolve`].
    File(PathBuf),
}

impl SecretSource {
    /// Produce the [`Secret`].
    ///
    /// `Inline` is taken verbatim. `File` reads the file as UTF-8 and trims
    /// exactly one trailing newline (`\n` or `\r\n`); a missing or unreadable
    /// file is [`JammiError::Config`] naming the path.
    pub fn resolve(&self) -> Result<Secret> {
        match self {
            Self::Inline(value) => Ok(Secret(value.clone())),
            Self::File(path) => read_secret_file(path),
        }
    }
}

impl fmt::Debug for SecretSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inline(_) => f.write_str("Inline(***)"),
            Self::File(path) => f.debug_tuple("File").field(path).finish(),
        }
    }
}

/// Read a file-backed secret: the one file-reading rule every file-backed
/// secret in the config shares (the `{ file = … }` form, and the file-backed
/// audit signing key).
///
/// Reads `path` as UTF-8 and trims exactly one trailing newline (`\n` or
/// `\r\n`). A missing, unreadable, or non-UTF-8 file is
/// [`JammiError::Config`] naming the path — the error a deployer sees at
/// `JammiConfig::load`, so the message carries the one thing they need to
/// fix it.
pub fn read_secret_file(path: &Path) -> Result<Secret> {
    let contents = std::fs::read_to_string(path).map_err(|e| {
        JammiError::Config(format!(
            "secret file `{}` could not be read: {e}",
            path.display()
        ))
    })?;
    Ok(Secret(trim_one_trailing_newline(contents)))
}

/// Drop exactly one trailing line terminator (`\n` or `\r\n`), nothing else.
fn trim_one_trailing_newline(mut value: String) -> String {
    if value.ends_with('\n') {
        value.pop();
        if value.ends_with('\r') {
            value.pop();
        }
    }
    value
}

/// A resolved secret. Renders as `Secret(***)`; read it with [`Secret::expose`].
///
/// Deliberately **not** `Serialize`: a secret that has entered the config
/// never leaves it as text through an ordinary derive. Deserialises from
/// the same shapes as [`SecretSource`], resolving the file form at parse
/// time (see the module docs for why). The one sanctioned exit is
/// [`serialize_exposed`], used solely by the persisted
/// `crate::storage::config` credential fields via
/// `#[serde(serialize_with = "…")]`.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wrap an already-resolved value (a test fixture, a value built in code).
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The secret itself. The only way to read it; every call site is a place
    /// the value is handed to the thing that needs it (a pool, a broker, a
    /// request header), never a log line.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

/// Serialize `secret` in exposed (plaintext) form — the one sanctioned exit
/// a `Secret` has to text, used ONLY as the `serialize_with` for the
/// persisted `crate::storage::config` credential fields
/// (`S3Config::secret_access_key`, `R2Config::secret_access_key`,
/// `GcsConfig::service_account_json`, `AzureConfig::{account_key,
/// sas_token, client_secret}`), whose whole point is to round-trip through
/// the catalog's `sources.options` JSON exactly as before this type
/// existed. Every other place a `Secret` sits in the config has no
/// `serialize_with` at all — a container can leak a `Secret` field only by
/// naming this function explicitly, field by field, never by deriving
/// `Serialize` over a struct that happens to contain one. Never call this
/// from a `Debug`/logging path; `Secret`'s own `Debug` stays redacted
/// regardless.
pub fn serialize_exposed<S: Serializer>(
    secret: &Option<Secret>,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    match secret {
        Some(s) => serializer.serialize_str(s.expose()),
        None => serializer.serialize_none(),
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(***)")
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for Secret {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        SecretSource::deserialize(deserializer)?
            .resolve()
            .map_err(de::Error::custom)
    }
}

impl<'de> Deserialize<'de> for SecretSource {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_any(SecretSourceVisitor)
    }
}

struct SecretSourceVisitor;

impl<'de> Visitor<'de> for SecretSourceVisitor {
    type Value = SecretSource;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(ACCEPTED_SHAPES)
    }

    fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
        if is_quoted_file_table(value) {
            return Err(E::custom(
                "a secret given as the string `{ file = \"…\" }` is not accepted: \
                 write `{ file = \"…\" }` as a TOML table (unquoted) or use the `__FILE` \
                 env spelling (`JAMMI_<SECTION>__<KEY>__FILE=/path`)",
            ));
        }
        Ok(SecretSource::Inline(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> std::result::Result<Self::Value, E> {
        self.visit_str(&value)
    }

    fn visit_map<A: MapAccess<'de>>(
        self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut file: Option<PathBuf> = None;
        let mut unknown = BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if key == "file" {
                if file.is_some() {
                    return Err(de::Error::duplicate_field("file"));
                }
                file = Some(map.next_value::<PathBuf>()?);
            } else {
                map.next_value::<IgnoredAny>()?;
                unknown.insert(key);
            }
        }
        if !unknown.is_empty() {
            let keys = unknown
                .iter()
                .map(|k| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(de::Error::custom(format!(
                "unexpected key(s) {keys} in a secret table; expected {ACCEPTED_SHAPES}"
            )));
        }
        file.map(SecretSource::File).ok_or_else(|| {
            de::Error::custom(format!("empty secret table; expected {ACCEPTED_SHAPES}"))
        })
    }

    // Every other primitive shape (integer, float, bool, sequence, unit, …)
    // falls through to serde's default `invalid_type` error, which quotes
    // `expecting` — the accepted shapes — so nothing is listed twice.
}

/// Does this string parse as a TOML inline table whose only key is `file`?
///
/// That is the text an operator produces by quoting the file form
/// (`url = "{ file = \"…\" }"`) or by pasting it into a plain env var. It is
/// never a plausible inline secret, so refusing it costs nothing and turns a
/// far-away "invalid connection string" into a load-time error that names the
/// fix.
fn is_quoted_file_table(value: &str) -> bool {
    let trimmed = value.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return false;
    }
    let Ok(doc) = toml::from_str::<toml::Table>(&format!("v = {trimmed}")) else {
        return false;
    };
    match doc.get("v") {
        Some(toml::Value::Table(inner)) => inner.len() == 1 && inner.contains_key("file"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, serde::Deserialize)]
    struct Holder {
        secret: Secret,
    }

    #[test]
    fn secret_file_form_loads_and_trims_one_trailing_newline() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pg-url");
        std::fs::write(&path, "postgres://u:hunter2@h/db\n").unwrap();
        let src = format!("secret = {{ file = {:?} }}", path.to_str().unwrap());
        let holder: Holder = toml::from_str(&src).unwrap();
        assert_eq!(holder.secret.expose(), "postgres://u:hunter2@h/db");

        // Exactly one: a second newline survives, and a value with none is
        // untouched.
        std::fs::write(&path, "two\n\n").unwrap();
        assert_eq!(
            SecretSource::File(path.clone()).resolve().unwrap().expose(),
            "two\n"
        );
        std::fs::write(&path, "bare").unwrap();
        assert_eq!(
            SecretSource::File(path.clone()).resolve().unwrap().expose(),
            "bare"
        );
        std::fs::write(&path, "crlf\r\n").unwrap();
        assert_eq!(SecretSource::File(path).resolve().unwrap().expose(), "crlf");
    }

    #[test]
    fn secret_missing_file_names_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist");
        let err = SecretSource::File(path.clone()).resolve().unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, JammiError::Config(_)), "{msg}");
        assert!(msg.contains(path.to_str().unwrap()), "{msg}");

        // The same error surfaces at parse time through `Secret`'s eager
        // resolution, still naming the path.
        let src = format!("secret = {{ file = {:?} }}", path.to_str().unwrap());
        let err = toml::from_str::<Holder>(&src).unwrap_err().to_string();
        assert!(err.contains(path.to_str().unwrap()), "{err}");
    }

    #[test]
    fn secret_debug_is_redacted() {
        let secret = Secret::new("hunter2");
        assert_eq!(format!("{secret:?}"), "Secret(***)");
        assert_eq!(format!("{secret}"), "Secret(***)");
        assert_eq!(format!("{secret:#?}"), "Secret(***)");
        assert_eq!(secret.expose(), "hunter2");

        let inline = SecretSource::Inline("hunter2".into());
        assert_eq!(format!("{inline:?}"), "Inline(***)");
        let file = SecretSource::File(PathBuf::from("/run/secrets/x"));
        assert_eq!(format!("{file:?}"), "File(\"/run/secrets/x\")");

        let holder: Holder = toml::from_str("secret = \"hunter2\"").unwrap();
        assert!(!format!("{holder:?}").contains("hunter2"));
    }

    #[test]
    fn secret_source_refuses_inline_table_string_pointing_at_file_spelling() {
        let err = toml::from_str::<Holder>(r#"secret = "{ file = \"/run/secrets/x\" }""#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("__FILE"), "{err}");
        assert!(err.contains("TOML table"), "{err}");

        // Only the single-`file`-key table shape is refused; other brace-y
        // strings are ordinary inline secrets.
        let json_like: Holder = toml::from_str(r#"secret = "{\"a\":1}""#).unwrap();
        assert_eq!(json_like.secret.expose(), "{\"a\":1}");
        let two_keys: Holder = toml::from_str(r#"secret = "{ file = \"x\", other = 1 }""#).unwrap();
        assert_eq!(two_keys.secret.expose(), "{ file = \"x\", other = 1 }");
    }

    #[test]
    fn secret_source_rejects_other_shapes_naming_the_accepted_ones() {
        for src in [
            "secret = 42",
            "secret = true",
            "secret = [\"a\"]",
            "secret = { path = \"/x\" }",
            "secret = { file = \"/x\", extra = 1 }",
            "secret = {}",
        ] {
            let err = toml::from_str::<Holder>(src).unwrap_err().to_string();
            assert!(err.contains("`file`"), "{src}: {err}");
        }
    }

    /// `serialize_exposed` is the one sanctioned way a `Secret` becomes
    /// plaintext again: only through an explicit `serialize_with` a field
    /// names, and only for the persisted-storage-config use it exists for.
    #[test]
    fn serialize_exposed_round_trips_the_plaintext_for_persistence() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Holder {
            #[serde(serialize_with = "serialize_exposed")]
            secret: Option<Secret>,
        }
        let holder = Holder {
            secret: Some(Secret::new("hunter2")),
        };
        let json = serde_json::to_string(&holder).unwrap();
        assert!(json.contains("hunter2"), "{json}");
        let round_tripped: Holder = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.secret.unwrap().expose(), "hunter2");

        let none_holder = Holder { secret: None };
        let json_none = serde_json::to_string(&none_holder).unwrap();
        assert_eq!(json_none, r#"{"secret":null}"#);
    }

    #[test]
    fn secret_equality_and_conversions() {
        assert_eq!(Secret::from("a"), Secret::new(String::from("a")));
        assert_ne!(Secret::from("a"), Secret::from("b"));
        assert_eq!(
            SecretSource::Inline("a".into()).resolve().unwrap(),
            Secret::from("a")
        );
    }
}
