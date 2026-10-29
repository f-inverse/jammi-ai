#[cfg(any(feature = "postgres", feature = "mysql"))]
mod database;
pub mod file_format;
pub mod mutable;
#[cfg(feature = "mysql")]
pub mod mysql;
#[cfg(feature = "postgres")]
pub mod postgres;
pub mod registry;
pub mod schema_provider;
// Note: SQLite external source via datafusion-table-providers is not supported
// because DTP's rusqlite version conflicts with our catalog's rusqlite (different
// libsqlite3-sys links versions). Users can query SQLite files via the File
// source after exporting to Parquet/CSV.

use serde::{Deserialize, Serialize};

use crate::storage::{CloudConfig, StorageUrl};

/// Supported data source backends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    /// File-shaped source (Parquet / CSV / JSON / JSONL) read through any
    /// `StorageUrl`-addressable backend — local disk, S3, GCS, Azure.
    File,
    /// PostgreSQL database.
    Postgres,
    /// MySQL / MariaDB database.
    Mysql,
}

/// File format for [`SourceType::File`] sources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileFormat {
    /// Apache Parquet columnar format.
    Parquet,
    /// Comma-separated values.
    Csv,
    /// Newline-delimited JSON.
    Json,
    /// Newline-delimited JSON, spelled `jsonl`/`ndjson`. Functionally
    /// identical to [`Self::Json`] (both use DataFusion's line-delimited
    /// `JsonFormat`); the distinct variant exists so a `.jsonl` corpus is
    /// neither rejected as an unknown format nor silently excluded by the
    /// `.json` directory glob.
    ///
    /// Directory listing tries `.jsonl` first; only when that has zero
    /// matches does it fall back to `.ndjson` (see
    /// [`crate::source::file_format::create_listing_table`]) — `.jsonl`
    /// always wins when a directory holds both. This is the only reachable
    /// path onto an `.ndjson` corpus: no wire or CLI surface names `.ndjson`
    /// directly. For a source registered through
    /// [`crate::session::JammiSession::add_source`], this resolution runs
    /// exactly ONCE, at registration, and is then PINNED into the persisted
    /// [`SourceConnection::file_extension`] — every later build of THAT
    /// source's providers (a restart, or another replica resolving it from
    /// the catalog row) passes the pinned value as an explicit override, so
    /// a directory change after registration (a `.jsonl` file added to a
    /// corpus that resolved to `.ndjson`, or vice versa) can never silently
    /// flip which files the source serves. Resolution from a row never
    /// backfills the pin, so a catalog row written some other way carries no
    /// such guarantee. An explicit `file_extension` override supplied by the
    /// caller up front disables the fallback from the start and is honoured
    /// literally.
    JsonLines,
}

impl FileFormat {
    /// The file extensions a source of this format lists, in the order they
    /// are tried: JSON lines are spelled `.jsonl` or `.ndjson`, whichever the
    /// directory holds; every other format has one.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Parquet => &[".parquet"],
            Self::Csv => &[".csv"],
            Self::Json => &[".json"],
            Self::JsonLines => &[".jsonl", ".ndjson"],
        }
    }
}

impl std::fmt::Display for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parquet => write!(f, "parquet"),
            Self::Csv => write!(f, "csv"),
            Self::Json => write!(f, "json"),
            Self::JsonLines => write!(f, "jsonl"),
        }
    }
}

impl std::str::FromStr for FileFormat {
    type Err = crate::error::JammiError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "parquet" => Ok(Self::Parquet),
            "csv" => Ok(Self::Csv),
            "json" => Ok(Self::Json),
            "jsonl" | "ndjson" => Ok(Self::JsonLines),
            other => Err(crate::error::JammiError::InvalidArgument {
                argument: "format".into(),
                reason: format!(
                    "unknown file format '{other}'; expected parquet, csv, json or jsonl"
                ),
            }),
        }
    }
}

/// The certificate authority a database source verifies its server against:
/// one or more PEM `CERTIFICATE` sections, as the text a client sent. A client
/// that cannot reach the engine's host hands over the certificate itself,
/// never a path on that host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CaCertificate(String);

impl CaCertificate {
    /// `pem` as a CA certificate: every `CERTIFICATE` section it holds must
    /// decode, and it must hold one.
    pub fn from_pem(pem: impl Into<String>) -> crate::error::Result<Self> {
        use rustls_pki_types::pem::PemObject;
        use rustls_pki_types::CertificateDer;

        let pem = pem.into();
        let refused = |reason: String| crate::error::JammiError::InvalidArgument {
            argument: "ca_certificate".into(),
            reason,
        };
        let certificates = CertificateDer::pem_slice_iter(pem.as_bytes())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| refused(format!("not a PEM certificate: {e}")))?;
        if certificates.is_empty() {
            return Err(refused("holds no PEM `CERTIFICATE` section".into()));
        }
        Ok(Self(pem))
    }

    /// The PEM text.
    pub fn pem(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for CaCertificate {
    type Error = crate::error::JammiError;

    fn try_from(pem: String) -> crate::error::Result<Self> {
        Self::from_pem(pem)
    }
}

impl From<CaCertificate> for String {
    fn from(ca: CaCertificate) -> Self {
        ca.0
    }
}

/// Connection parameters for a data source.
///
/// Structural equality is the identity a session's source cache is keyed on
/// (see [`SourceDefinition`]): two connections that compare equal build the
/// same providers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceConnection {
    /// Storage URL (file://, s3://, gs://, azure://) or external-source
    /// connection string (postgres://, mysql://).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// File format (required for [`SourceType::File`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<FileFormat>,

    /// Override the default file extension used during directory listing.
    /// `None` on a fresh [`FileFormat::JsonLines`] registration (no caller
    /// override) lets the engine adaptively choose `.jsonl` or `.ndjson`
    /// once at registration; the engine writes the winner back here before
    /// persisting, so every later build from the persisted row sees an
    /// explicit override and skips the adaptive resolution entirely — same
    /// persist-once-replay-forever contract [`Self::tenant_column`] has.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_extension: Option<String>,

    /// Per-cloud credentials. `None` for local / external sources and for
    /// cloud sources that rely on ambient credentials (instance profile,
    /// ADC, Managed Identity).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cloud: Option<CloudConfig>,

    /// The certificate authority a database source verifies its server's
    /// certificate against, when the server's certificate is not signed by
    /// one the engine's host trusts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ca_certificate: Option<CaCertificate>,

    /// Name of the tenant-discriminator column on this federated source, when
    /// its rows carry tenancy under a column other than `tenant_id`. Consulted
    /// by the tenant-scope analyzer to inject the discriminator predicate on a
    /// scan of this source. Persisted within the source's serialized options so
    /// it round-trips across session restarts (every resolution from the
    /// persisted row replays it into the in-process
    /// [`crate::tenant_scope::SourceTenantColumns`] lookup); a source with no
    /// tenant discriminator leaves this `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_column: Option<String>,
}

/// What a source IS: the pair the catalog's `sources` row persists and every
/// provider build reads. A session caches the providers it built for a
/// source together with the definition they were built from, and rebuilds
/// exactly when the row's definition no longer compares equal to the cached
/// one — a source re-registered under the same id with a different
/// connection resolves to its new providers on every replica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDefinition {
    /// The storage backend the source reads through.
    pub source_type: SourceType,
    /// The connection the providers are built from.
    pub connection: SourceConnection,
}

impl SourceConnection {
    /// Refuse the connection of a `source_type` source that writes a secret
    /// (#756): a database source's URL writing its password, or a source's
    /// cloud credentials writing one of theirs. The catalog keeps both, for
    /// every later build to connect with, so a secret in either would be
    /// written down with it; each names its secret with a `${NAME}`
    /// placeholder instead, read from the environment of the process that
    /// builds the source. That environment is the server's while the source
    /// is its caller's, so a placeholder naming a variable outside
    /// [`SOURCE_VARIABLE_PREFIX`](crate::config::secret::SOURCE_VARIABLE_PREFIX)
    /// is refused too. The refusal never repeats the URL or the secret.
    pub(crate) fn refuse_a_written_secret(
        &self,
        source_type: &SourceType,
    ) -> crate::error::Result<()> {
        use crate::config::secret::{unnamable_variable, SOURCE_VARIABLE_PREFIX as PREFIX};
        let refused = |argument: &str, reason: String| crate::error::JammiError::InvalidArgument {
            argument: argument.into(),
            reason,
        };
        let outside = |named: &str| {
            format!(
                "`${{{named}}}` names a variable no source may name, so that a source never \
                 reads the server's other secrets: a source's placeholder names a `{PREFIX}` \
                 variable"
            )
        };
        if let Some(cloud) = &self.cloud {
            if let Some(field) = cloud.written_secret() {
                return Err(refused(
                    "cloud",
                    format!(
                        "a source's cloud credentials name each secret with a \
                         `${{{PREFIX}<NAME>}}` placeholder, never the secret itself, so that the \
                         catalog never holds it: `{field}` holds one; set it to \
                         `${{{PREFIX}<NAME>}}`, and set that variable where the engine reads the \
                         source"
                    ),
                ));
            }
            if let Some((field, named)) = cloud.unnamable_variable() {
                return Err(refused("cloud", format!("`{field}`: {}", outside(&named))));
            }
        }
        let example = match source_type {
            SourceType::Postgres => "postgresql://user:${JAMMI_SOURCE_DB_PASSWORD}@host/database",
            SourceType::Mysql => "mysql://user:${JAMMI_SOURCE_DB_PASSWORD}@host/database",
            SourceType::File => return Ok(()),
        };
        let Some(url) = self.url.as_deref() else {
            return Ok(());
        };
        if url_writes_its_password(url) {
            return Err(refused(
                "url",
                format!(
                    "a database source's URL names its password with a `${{{PREFIX}<NAME>}}` \
                     placeholder, never the password itself, so that the catalog never holds it: \
                     write `{example}`, and set JAMMI_SOURCE_DB_PASSWORD where the engine connects"
                ),
            ));
        }
        match unnamable_variable(url) {
            Some(named) => Err(refused("url", outside(&named))),
            None => Ok(()),
        }
    }

    /// Build a connection from a [`StorageUrl`] + file format.
    ///
    /// Used by file-shaped sources regardless of scheme — local paths
    /// (parsed into `file://...`), S3, GCS, Azure. The URL has already
    /// been validated so this never fails.
    pub fn from_url(url: StorageUrl, format: FileFormat) -> Self {
        Self {
            url: Some(url.to_string()),
            format: Some(format),
            ..Default::default()
        }
    }

    /// Parse `path_or_url` into a [`StorageUrl`] and build a connection.
    ///
    /// Convenience wrapper used by CLI / Python callers that accept a free-
    /// form string. Returns the underlying [`crate::storage::StorageError`]
    /// when the input is unrecognisable.
    pub fn parse(
        path_or_url: &str,
        format: FileFormat,
    ) -> Result<Self, crate::storage::StorageError> {
        let url = StorageUrl::parse(path_or_url)?;
        Ok(Self::from_url(url, format))
    }
}

/// Whether `url`, a database source's URL, writes a password itself, in its
/// user info or as a Postgres `password` parameter, rather than naming it with
/// `${NAME}` placeholders alone (#756).
pub(crate) fn url_writes_its_password(url: &str) -> bool {
    password_of(url).is_some_and(|password| !crate::config::secret::names_only(&password))
}

/// The password `url` carries, decoded: as libpq reads a Postgres URL, else
/// the user info of any other.
fn password_of(url: &str) -> Option<String> {
    use crate::pg_uri::{Keyword, PgUri};

    if let Ok(uri) = PgUri::parse(url) {
        return uri
            .params()
            .find(|(keyword, _)| *keyword == Keyword::Password)
            .map(|(_, password)| password.to_string());
    }
    let parsed = url::Url::parse(url).ok()?;
    let password = parsed.password()?;
    Some(
        percent_encoding::percent_decode_str(password)
            .decode_utf8_lossy()
            .into_owned(),
    )
}

impl SourceDefinition {
    /// The source `url` names. A `postgres://` / `postgresql://` URL is a
    /// PostgreSQL database and a `mysql://` URL a MySQL / MariaDB database,
    /// each connected through the URL as given; any other URL or local path is
    /// a file read in `format`. A file needs its format; a database has none,
    /// so one given for it is refused rather than ignored.
    pub fn from_url(url: &str, format: Option<FileFormat>) -> crate::error::Result<Self> {
        let database = match url.split_once("://").map(|(scheme, _)| scheme) {
            Some("postgres" | "postgresql") => Some(SourceType::Postgres),
            Some("mysql") => Some(SourceType::Mysql),
            _ => None,
        };
        match (database, format) {
            (Some(source_type), None) => Ok(Self {
                source_type,
                connection: SourceConnection {
                    url: Some(url.to_string()),
                    ..Default::default()
                },
            }),
            (Some(source_type), Some(format)) => Err(crate::error::JammiError::InvalidArgument {
                argument: "format".into(),
                reason: format!(
                    "a {source_type:?} source is a database, not a file: it takes no format \
                 (got `{format}`)"
                ),
            }),
            (None, Some(format)) => Ok(Self {
                source_type: SourceType::File,
                connection: SourceConnection::parse(url, format)?,
            }),
            (None, None) => Err(crate::error::JammiError::InvalidArgument {
                argument: "format".into(),
                reason: format!(
                    "`{url}` names a file, which needs a format (parquet, csv, json, jsonl); a \
                 database source is a `postgres://` or `mysql://` URL"
                ),
            }),
        }
    }
}

/// A table a source serves: the source, the schema the table sits in, and
/// its name. A file's table and a MySQL database's tables sit in
/// [`SourceTable::PUBLIC`]; a Postgres database's sit in their own schemas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceTable {
    pub source: String,
    pub schema: String,
    pub table: String,
}

impl SourceTable {
    /// The schema a source with no schemas of its own serves its tables in.
    pub const PUBLIC: &'static str = "public";

    /// A table of `source` in `schema`.
    pub fn new(source: &str, schema: &str, table: &str) -> Self {
        Self {
            source: source.into(),
            schema: schema.into(),
            table: table.into(),
        }
    }

    /// The order a source lists its tables in, and the first of which a verb
    /// naming the source reads: the `public` schema first, then every other
    /// schema by name, and each schema's tables by name.
    pub(crate) fn order_key(&self) -> (bool, &str, &str) {
        (self.schema != Self::PUBLIC, &self.schema, &self.table)
    }
}

/// One table a source serves, with the columns it reads as.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTableDescription {
    pub table: SourceTable,
    pub columns: arrow::datatypes::SchemaRef,
}

/// One registered source described whole: its registry descriptor and every
/// table it serves, in the source's order ([`SourceTable`]'s).
#[derive(Debug, Clone)]
pub struct SourceDescription {
    pub descriptor: crate::catalog::source_repo::SourceDescriptor,
    pub tables: Vec<SourceTableDescription>,
}

/// The table as SQL names it: `"source"."schema"."table"`, each part quoted.
impl std::fmt::Display for SourceTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::sql::ident::quote_ident;
        write!(
            f,
            "{}.{}.{}",
            quote_ident(&self.source),
            quote_ident(&self.schema),
            quote_ident(&self.table)
        )
    }
}

/// The relation a verb reads its rows from: a source's table, or a table the
/// engine holds — a result table or a mutable table — under the name SQL
/// reads it by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputRelation {
    /// A source's table, or a mutable table (`mutable.public.<name>`).
    Table(SourceTable),
    /// A result table (`"jammi.<name>"`).
    Result(crate::store::RelationKey),
}

/// The relation as SQL names it.
impl std::fmt::Display for InputRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Table(table) => table.fmt(f),
            Self::Result(relation) => relation.fmt(f),
        }
    }
}

/// Derive a DataFusion table name from a URL by extracting the file stem.
///
/// Handles both Unix forward-slash paths and Windows backslash paths so that
/// `file://C:\Users\...\triplets_train.parquet` correctly yields `triplets_train`
/// rather than the full `C:\...\triplets_train` (which contains a colon that
/// breaks DataFusion's SQL parser).
pub(crate) fn table_name_from_url(url: &str) -> String {
    let path = url.rsplit(['/', '\\']).next().unwrap_or(url);
    path.split('.').next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_url_writes_its_password_unless_placeholders_alone_name_it() {
        for written in [
            "postgresql://ada:hunter2@db/sales",
            "postgres://ada@db/sales?password=hunter2",
            "mysql://ada:hunter2@db/sales",
            "postgresql://ada:${PG}x@db/sales",
            "postgresql://ada:pa$$word@db/sales",
            "postgresql://ada:p%40ss@db/sales",
        ] {
            assert!(url_writes_its_password(written), "{written}");
        }
        for named in [
            "postgresql://ada:${PG_PASSWORD}@db/sales",
            "postgresql://ada:${A}${B}@db/sales",
            "postgres://ada@db/sales?password=${PG_PASSWORD}",
            "mysql://ada:${MYSQL_PASSWORD}@db/sales",
            "postgresql://ada@db/sales",
            "postgresql://postgres:@/postgres?host=/tmp/pgdata",
            "${DATABASE_URL}",
        ] {
            assert!(!url_writes_its_password(named), "{named}");
        }
    }

    use std::str::FromStr;

    #[test]
    fn a_ca_certificate_is_pem_holding_a_certificate() {
        let pem = include_str!("../../../../tests/fixtures/isrg-root-x1.pem");
        assert_eq!(CaCertificate::from_pem(pem).unwrap().pem(), pem);
        for refused in [
            "/etc/ssl/certs/ca.pem",
            "-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----\n",
            "-----BEGIN CERTIFICATE-----\n!!!\n-----END CERTIFICATE-----\n",
        ] {
            let error = CaCertificate::from_pem(refused).unwrap_err();
            assert_eq!(
                error.class(),
                crate::error::ErrorClass::InvalidArgument,
                "{refused}: {error}"
            );
        }
    }

    #[test]
    fn file_format_round_trips_through_display_and_from_str() {
        for format in [
            FileFormat::Parquet,
            FileFormat::Csv,
            FileFormat::Json,
            FileFormat::JsonLines,
        ] {
            let rendered = format.to_string();
            let parsed = FileFormat::from_str(&rendered).expect("canonical format parses");
            assert_eq!(parsed, format, "round-trip must be identity for {format:?}");
        }
        assert_eq!(FileFormat::JsonLines.to_string(), "jsonl");
    }

    #[test]
    fn jsonl_and_ndjson_both_parse_to_json_lines() {
        assert_eq!(
            FileFormat::from_str("jsonl").unwrap(),
            FileFormat::JsonLines
        );
        assert_eq!(
            FileFormat::from_str("ndjson").unwrap(),
            FileFormat::JsonLines
        );
    }

    #[test]
    fn unknown_file_format_error_names_every_accepted_token() {
        let err = FileFormat::from_str("bogus").unwrap_err().to_string();
        assert!(
            err.contains("jsonl"),
            "error must name jsonl as an accepted format: {err}"
        );
        assert!(err.contains("parquet"));
        assert!(err.contains("csv"));
        assert!(err.contains("json"));
    }

    #[test]
    fn a_database_url_names_its_database_and_takes_no_format() {
        for (url, source_type) in [
            ("postgres://u@h/d", SourceType::Postgres),
            ("postgresql://u@h/d", SourceType::Postgres),
            ("mysql://u@h/d", SourceType::Mysql),
        ] {
            let definition = SourceDefinition::from_url(url, None).unwrap();
            assert_eq!(definition.source_type, source_type);
            assert_eq!(definition.connection.url.as_deref(), Some(url));
            assert_eq!(definition.connection.format, None);
            assert!(SourceDefinition::from_url(url, Some(FileFormat::Parquet)).is_err());
        }
    }

    #[test]
    fn any_other_url_is_a_file_which_needs_its_format() {
        let definition =
            SourceDefinition::from_url("/data/corpus.parquet", Some(FileFormat::Parquet)).unwrap();
        assert_eq!(definition.source_type, SourceType::File);
        assert_eq!(
            definition.connection.url.as_deref(),
            Some("file:///data/corpus.parquet")
        );
        assert_eq!(definition.connection.format, Some(FileFormat::Parquet));
        assert!(SourceDefinition::from_url("/data/corpus.parquet", None).is_err());
    }
}
