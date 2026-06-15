//! Catalog repository for evidence channels and their declared columns.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::{JammiError, Result};
use crate::evidence_channel::ChannelId;

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::Catalog;

/// A caller-facing failure of a channel-catalog operation (register a channel,
/// append columns). These are the conditions a remote caller can provoke and
/// must be able to distinguish — `map_engine_error` maps each onto its own gRPC
/// code. Catalog corruption on read-back (a stored dtype token or channel slug
/// that no longer parses) is NOT in this taxonomy: it is an engine invariant
/// failure routed to `Internal`, not a caller error.
#[derive(Debug, Error)]
pub enum ChannelCatalogError {
    /// A channel of this id is already registered for the bound tenant.
    #[error("channel '{0}': already exists")]
    AlreadyExists(String),

    /// An `add_columns` (or merged-schema) op named a channel that is not
    /// registered for the bound tenant.
    #[error("channel '{0}': not registered")]
    NotRegistered(String),

    /// `add_columns` re-declared an existing column with the SAME type — the
    /// append-only catalog rejects even an idempotent redeclaration.
    #[error("channel '{channel}': column '{column}' already declared")]
    ColumnAlreadyDeclared {
        /// The channel the column belongs to.
        channel: String,
        /// The column name being redeclared.
        column: String,
        /// The (matching) type of both the existing and requested declaration.
        ty: ChannelColumnType,
    },

    /// `add_columns` re-declared an existing column with a DIFFERENT type — a
    /// precondition conflict against the stored declaration.
    #[error(
        "channel '{channel}': column '{column}' was declared {existing}, \
         cannot redeclare as {requested}"
    )]
    ColumnConflict {
        /// The channel the column belongs to.
        channel: String,
        /// The column name being redeclared.
        column: String,
        /// The type the column was originally declared with.
        existing: ChannelColumnType,
        /// The conflicting type the caller asked to redeclare it as.
        requested: ChannelColumnType,
    },

    /// A caller-supplied channel id failed the slug rules.
    #[error("{0}")]
    InvalidId(String),

    /// A caller-supplied column-type token was not one of the closed set of
    /// PascalCase Arrow names (`Float32`, `Utf8`, …).
    #[error("unknown channel column type: '{0}'")]
    InvalidColumnType(String),
}

/// A column-type token (`"Float32"`, `"Utf8"`, …) that is not one of the closed
/// set of PascalCase Arrow names. A neutral parse outcome carrying the offending
/// token, with no caller-vs-corruption verdict baked in: [`ChannelColumnType::
/// from_sql_str`] returns it, and each call site decides whether the bad token
/// is a caller's input (→ [`ChannelCatalogError::InvalidColumnType`], a bad
/// request) or stored-catalog corruption (→ an internal fault).
#[derive(Debug)]
pub struct UnknownColumnTypeToken(pub String);

/// The closed set of Arrow types a channel column may declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ChannelColumnType {
    Float32,
    Float64,
    Int32,
    Int64,
    Utf8,
    Boolean,
}

impl std::fmt::Display for ChannelColumnType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ChannelColumnType {
    pub fn to_arrow(self) -> DataType {
        match self {
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::Utf8 => DataType::Utf8,
            Self::Boolean => DataType::Boolean,
        }
    }

    pub fn from_arrow(dt: &DataType) -> Result<Self> {
        match dt {
            DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            DataType::Int32 => Ok(Self::Int32),
            DataType::Int64 => Ok(Self::Int64),
            DataType::Utf8 => Ok(Self::Utf8),
            DataType::Boolean => Ok(Self::Boolean),
            other => Err(JammiError::Catalog(format!(
                "unsupported channel column type: {other:?}"
            ))),
        }
    }

    /// The canonical PascalCase variant name (`"Float32"`, `"Utf8"`, …). This is
    /// the string form shared with the catalog's stored representation, with
    /// [`Self::from_sql_str`], and with public-API callers (the Python binding's
    /// `register_channel` / `list_channels`, which round-trips this exact token).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float32 => "Float32",
            Self::Float64 => "Float64",
            Self::Int32 => "Int32",
            Self::Int64 => "Int64",
            Self::Utf8 => "Utf8",
            Self::Boolean => "Boolean",
        }
    }

    /// Parse a PascalCase variant name (`"Float32"`, `"Utf8"`, …) into a
    /// `ChannelColumnType`. The canonical string form is shared with the
    /// catalog's stored representation and with public-API callers (e.g.
    /// the Python binding's `register_channel(columns=[(name, dtype_str)])`).
    ///
    /// The token comes from two kinds of source — a caller's request and a
    /// read-back of the stored catalog — that fail differently (a bad request
    /// vs. catalog corruption). This parser stays neutral on that distinction:
    /// it returns the offending token as an [`UnknownColumnTypeToken`], and the
    /// call site decides which `JammiError` it becomes.
    pub fn from_sql_str(s: &str) -> std::result::Result<Self, UnknownColumnTypeToken> {
        match s {
            "Float32" => Ok(Self::Float32),
            "Float64" => Ok(Self::Float64),
            "Int32" => Ok(Self::Int32),
            "Int64" => Ok(Self::Int64),
            "Utf8" => Ok(Self::Utf8),
            "Boolean" => Ok(Self::Boolean),
            other => Err(UnknownColumnTypeToken(other.to_string())),
        }
    }
}

/// One declared column on a channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelColumn {
    pub name: String,
    pub data_type: ChannelColumnType,
}

/// The full declaration for one channel: identifier, priority, ordered columns.
#[derive(Debug, Clone)]
pub struct ChannelSpec {
    pub id: ChannelId,
    pub priority: i32,
    pub columns: Vec<ChannelColumn>,
}

/// Repository over `evidence_channels` and `evidence_channel_columns`.
///
/// Constructed via [`Catalog::channels`].
pub struct ChannelRepo<'a> {
    catalog: &'a Catalog,
}

fn map_constraint(e: BackendError, channel: &str) -> JammiError {
    match e {
        BackendError::Constraint { .. } => {
            JammiError::ChannelCatalog(ChannelCatalogError::AlreadyExists(channel.to_string()))
        }
        other => JammiError::BackendDriver(other),
    }
}

fn read_column_row(row: &Row<'_>) -> std::result::Result<(String, String), BackendError> {
    Ok((row.get("column_name")?, row.get("column_type")?))
}

/// A stored `column_type` token read back from the catalog that no longer parses
/// is corruption of the engine's own store, not a caller condition. It surfaces
/// as an internal [`JammiError::Catalog`] (→ gRPC `Internal`), never the
/// caller-facing [`ChannelCatalogError::InvalidColumnType`] (→ `InvalidArgument`).
fn stored_type_corruption(channel: &str, token: &UnknownColumnTypeToken) -> JammiError {
    JammiError::Catalog(format!(
        "channel '{channel}': stored column type '{}' is not a known channel column type",
        token.0
    ))
}

/// Does a parent `evidence_channels` row exist for `(tenant, channel)`?
///
/// Branches on the tenant to honour the `None -> IS NULL` semantics: SQL
/// `tenant_id = NULL` never matches, so a global lookup must use `IS NULL`
/// rather than binding `NULL` as a parameter (mirrors `mutable_repo`).
async fn channel_exists(
    tx: &mut super::backend::Transaction<'_>,
    tenant: Option<&str>,
    channel: &str,
) -> std::result::Result<bool, BackendError> {
    let found = if let Some(t) = tenant {
        tx.query_opt(
            "SELECT 1 AS one FROM evidence_channels \
             WHERE tenant_id = $1 AND channel_name = $2",
            &[
                SqlValue::TextOwned(t.to_string()),
                SqlValue::TextOwned(channel.to_string()),
            ],
            |row| row.get::<i32>("one"),
        )
        .await?
    } else {
        tx.query_opt(
            "SELECT 1 AS one FROM evidence_channels \
             WHERE tenant_id IS NULL AND channel_name = $1",
            &[SqlValue::TextOwned(channel.to_string())],
            |row| row.get::<i32>("one"),
        )
        .await?
    };
    Ok(found.is_some())
}

/// Highest declared ordinal on `(tenant, channel)`, or `None` if no columns.
async fn max_ordinal(
    tx: &mut super::backend::Transaction<'_>,
    tenant: Option<&str>,
    channel: &str,
) -> std::result::Result<Option<i32>, BackendError> {
    let found = if let Some(t) = tenant {
        tx.query_opt(
            "SELECT MAX(ordinal) AS m FROM evidence_channel_columns \
             WHERE tenant_id = $1 AND channel_name = $2",
            &[
                SqlValue::TextOwned(t.to_string()),
                SqlValue::TextOwned(channel.to_string()),
            ],
            |row| row.try_get::<i32>("m"),
        )
        .await?
    } else {
        tx.query_opt(
            "SELECT MAX(ordinal) AS m FROM evidence_channel_columns \
             WHERE tenant_id IS NULL AND channel_name = $1",
            &[SqlValue::TextOwned(channel.to_string())],
            |row| row.try_get::<i32>("m"),
        )
        .await?
    };
    Ok(found.flatten())
}

/// The stored `column_type` for `(tenant, channel, column)`, or `None`.
async fn column_type(
    tx: &mut super::backend::Transaction<'_>,
    tenant: Option<&str>,
    channel: &str,
    column: &str,
) -> std::result::Result<Option<String>, BackendError> {
    if let Some(t) = tenant {
        tx.query_opt(
            "SELECT column_type FROM evidence_channel_columns \
             WHERE tenant_id = $1 AND channel_name = $2 AND column_name = $3",
            &[
                SqlValue::TextOwned(t.to_string()),
                SqlValue::TextOwned(channel.to_string()),
                SqlValue::TextOwned(column.to_string()),
            ],
            |row| row.get::<String>("column_type"),
        )
        .await
    } else {
        tx.query_opt(
            "SELECT column_type FROM evidence_channel_columns \
             WHERE tenant_id IS NULL AND channel_name = $1 AND column_name = $2",
            &[
                SqlValue::TextOwned(channel.to_string()),
                SqlValue::TextOwned(column.to_string()),
            ],
            |row| row.get::<String>("column_type"),
        )
        .await
    }
}

/// `(channel_name, priority, columns)` row as returned by `list()`'s catalog
/// query. Aliased to keep the inner closure's local type readable.
type ChannelListing = (String, i32, Vec<(String, String)>);

impl<'a> ChannelRepo<'a> {
    pub(super) fn new(catalog: &'a Catalog) -> Self {
        Self { catalog }
    }

    /// Register a new channel and its columns atomically, scoped to the
    /// catalog's currently-bound tenant.
    ///
    /// The channel name is unique *per tenant*: tenant B may register a channel
    /// whose name already exists under tenant A without collision, but a
    /// re-registration within the same tenant is rejected. The explicit
    /// tenant-scoped existence check below yields the friendly "already exists"
    /// error; the database enforces the same uniqueness authoritatively via two
    /// constraints — the `UNIQUE (tenant_id, channel_name)` constraint for the
    /// non-NULL (tenant-scoped) case, and the partial unique index on
    /// `channel_name WHERE tenant_id IS NULL` for the global (`tenant = None`)
    /// case (both backends treat NULLs as distinct in a UNIQUE constraint, so
    /// the composite constraint alone would not catch a duplicate global
    /// channel). Either backstop surfaces through the same
    /// `BackendError::Constraint` → "already exists" path as the explicit check.
    pub async fn register(&self, spec: &ChannelSpec) -> Result<()> {
        let tenant = self.catalog.current_tenant().map(|t| t.to_string());
        let channel = spec.id.as_str().to_string();
        let channel_for_err = channel.clone();
        let priority = spec.priority as i64;
        let columns: Vec<(String, &'static str, i64)> = spec
            .columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), c.data_type.as_str(), i as i64))
            .collect();

        let res = self
            .catalog
            .backend()
            .transaction(TxOptions::default(), |tx| {
                let tenant = tenant.clone();
                Box::pin(async move {
                    // Explicit per-tenant existence check for a friendly error.
                    // The DB constraints (composite UNIQUE for tenant-scoped,
                    // partial unique index for global) are the authoritative
                    // backstop; this surfaces the same "already exists" message.
                    let exists = channel_exists(tx, tenant.as_deref(), &channel).await?;
                    if exists {
                        return Err(BackendError::Constraint {
                            table: "evidence_channels".to_string(),
                            detail: "channel_name already registered for tenant".to_string(),
                        });
                    }

                    tx.execute(
                        "INSERT INTO evidence_channels (tenant_id, channel_name, priority) \
                         VALUES ($1, $2, $3)",
                        &[
                            SqlValue::from(tenant.clone()),
                            SqlValue::TextOwned(channel.clone()),
                            SqlValue::Int(priority),
                        ],
                    )
                    .await?;

                    for (name, ty, ord) in columns {
                        tx.execute(
                            "INSERT INTO evidence_channel_columns \
                             (tenant_id, channel_name, column_name, column_type, ordinal) \
                             VALUES ($1, $2, $3, $4, $5)",
                            &[
                                SqlValue::from(tenant.clone()),
                                SqlValue::TextOwned(channel.clone()),
                                SqlValue::TextOwned(name),
                                SqlValue::Text(ty),
                                SqlValue::Int(ord),
                            ],
                        )
                        .await?;
                    }
                    Ok(())
                })
            })
            .await;

        res.map_err(|e| map_constraint(e, &channel_for_err))?;
        Ok(())
    }

    /// Append new columns to an already-registered channel. Append-only.
    /// Scoped to the catalog's currently-bound tenant: a channel is resolved,
    /// and its columns appended, only within the bound tenant's namespace.
    pub async fn add_columns(
        &self,
        channel: &ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        let tenant = self.catalog.current_tenant().map(|t| t.to_string());
        let channel_name = channel.as_str().to_string();
        let channel_for_err = channel_name.clone();
        let cols: Vec<(String, ChannelColumnType)> = new_columns
            .iter()
            .map(|c| (c.name.clone(), c.data_type))
            .collect();

        // The closure separates two failure kinds the transaction wrapper would
        // otherwise blur: the OUTER `BackendError` is a genuine DB fault (or
        // catalog corruption read back from the store — a stored type token that
        // no longer parses); the INNER `ChannelCatalogError` is a typed caller
        // validation outcome. After the transaction, `Ok(Ok(()))` is success,
        // `Ok(Err(cat))` a caller error, and `Err(be)` an internal fault — no
        // message-string matching to recover the kind.
        self.catalog
            .backend()
            .transaction(TxOptions::default(), |tx| {
                let tenant = tenant.clone();
                Box::pin(async move {
                    let tref = tenant.as_deref();
                    // Existence check on the parent row, tenant-scoped.
                    if !channel_exists(tx, tref, &channel_name).await? {
                        return Ok(Err(ChannelCatalogError::NotRegistered(channel_for_err)));
                    }

                    let max_ord = max_ordinal(tx, tref, &channel_name).await?;
                    let mut next = max_ord.unwrap_or(-1) + 1;

                    for (name, ty) in cols {
                        let existing = column_type(tx, tref, &channel_name, &name).await?;
                        if let Some(existing_type) = existing {
                            // A stored token that no longer parses is catalog
                            // corruption, not a caller condition: it stays the
                            // OUTER backend fault.
                            let existing = ChannelColumnType::from_sql_str(&existing_type)
                                .map_err(|t| {
                                    BackendError::Execution(format!(
                                        "channel '{channel_for_err}': stored column type '{}' \
                                         for '{name}' is not a known channel column type",
                                        t.0
                                    ))
                                })?;
                            if existing == ty {
                                return Ok(Err(ChannelCatalogError::ColumnAlreadyDeclared {
                                    channel: channel_for_err,
                                    column: name,
                                    ty,
                                }));
                            } else {
                                return Ok(Err(ChannelCatalogError::ColumnConflict {
                                    channel: channel_for_err,
                                    column: name,
                                    existing,
                                    requested: ty,
                                }));
                            }
                        }

                        tx.execute(
                            "INSERT INTO evidence_channel_columns \
                             (tenant_id, channel_name, column_name, column_type, ordinal) \
                             VALUES ($1, $2, $3, $4, $5)",
                            &[
                                SqlValue::from(tenant.clone()),
                                SqlValue::TextOwned(channel_name.clone()),
                                SqlValue::TextOwned(name),
                                SqlValue::Text(ty.as_str()),
                                SqlValue::Int(next as i64),
                            ],
                        )
                        .await?;
                        next += 1;
                    }
                    Ok(Ok(()))
                })
            })
            .await
            .map_err(JammiError::BackendDriver)?
            .map_err(JammiError::ChannelCatalog)?;
        Ok(())
    }

    /// Look up one channel's full spec, resolved in the catalog's currently-bound
    /// tenant namespace.
    ///
    /// A tenant resolves its OWN channel if it has registered one of that name,
    /// otherwise it falls back to the GLOBAL (`tenant_id IS NULL`) channel — the
    /// same own-shadows-global precedence the model catalog uses (#140). This is
    /// what lets every tenant resolve the global seed channels (`vector`,
    /// `inference`, `bm25`) while still owning a private channel of the same
    /// name. A `tenant = None` lookup resolves only global rows. Crucially, a
    /// tenant NEVER resolves another tenant's row. Once the owning `tenant_id` is
    /// resolved, the columns are read under that SAME `tenant_id`, so a tenant's
    /// channel and a global channel of the same name never blend their columns.
    pub async fn get(&self, channel: &ChannelId) -> Result<Option<ChannelSpec>> {
        let tenant = self.catalog.current_tenant().map(|t| t.to_string());
        let channel_name = channel.as_str().to_string();
        let id = channel.clone();
        let found = self
            .catalog
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let channel_name = channel_name.clone();
                    let tenant = tenant.clone();
                    Box::pin(async move {
                        let tref = tenant.as_deref();
                        // Resolve the owning row: own-tenant shadows global.
                        let resolved = if let Some(t) = tref {
                            tx.query_opt(
                                "SELECT tenant_id, priority FROM evidence_channels \
                                 WHERE (tenant_id = $1 OR tenant_id IS NULL) AND channel_name = $2 \
                                 ORDER BY (tenant_id IS NOT NULL) DESC LIMIT 1",
                                &[
                                    SqlValue::TextOwned(t.to_string()),
                                    SqlValue::TextOwned(channel_name.clone()),
                                ],
                                |row| {
                                    Ok((
                                        row.try_get::<String>("tenant_id")?,
                                        row.get::<i32>("priority")?,
                                    ))
                                },
                            )
                            .await?
                        } else {
                            tx.query_opt(
                                "SELECT tenant_id, priority FROM evidence_channels \
                                 WHERE tenant_id IS NULL AND channel_name = $1",
                                &[SqlValue::TextOwned(channel_name.clone())],
                                |row| {
                                    Ok((
                                        row.try_get::<String>("tenant_id")?,
                                        row.get::<i32>("priority")?,
                                    ))
                                },
                            )
                            .await?
                        };
                        let Some((row_tenant, priority)) = resolved else {
                            return Ok(None);
                        };
                        // Read columns under the SAME resolved tenant_id.
                        let cols = if let Some(rt) = row_tenant {
                            tx.query(
                                "SELECT column_name, column_type FROM evidence_channel_columns \
                                 WHERE tenant_id = $1 AND channel_name = $2 ORDER BY ordinal",
                                &[SqlValue::TextOwned(rt), SqlValue::TextOwned(channel_name)],
                                read_column_row,
                            )
                            .await?
                        } else {
                            tx.query(
                                "SELECT column_name, column_type FROM evidence_channel_columns \
                                 WHERE tenant_id IS NULL AND channel_name = $1 ORDER BY ordinal",
                                &[SqlValue::TextOwned(channel_name)],
                                read_column_row,
                            )
                            .await?
                        };
                        Ok(Some((priority, cols)))
                    })
                },
            )
            .await?;

        let Some((priority, raw_cols)) = found else {
            return Ok(None);
        };
        // A stored type token that no longer parses is catalog corruption, not a
        // caller condition — route it to an internal fault, never an
        // `InvalidColumnType` caller error.
        let columns: Result<Vec<ChannelColumn>> = raw_cols
            .into_iter()
            .map(|(name, ty)| {
                Ok(ChannelColumn {
                    name,
                    data_type: ChannelColumnType::from_sql_str(&ty)
                        .map_err(|t| stored_type_corruption(channel.as_str(), &t))?,
                })
            })
            .collect();
        Ok(Some(ChannelSpec {
            id,
            priority,
            columns: columns?,
        }))
    }

    /// List every registered channel resolved in the catalog's currently-bound
    /// tenant namespace, ordered by `(priority, channel_name)`.
    ///
    /// A tenant sees its OWN channels plus the GLOBAL (`tenant_id IS NULL`)
    /// channels it has not shadowed — its own channel of a given name takes
    /// precedence over a global one of the same name (the #140 own-shadows-global
    /// rule). It NEVER sees another tenant's channels. A `tenant = None` listing
    /// returns only global rows.
    pub async fn list(&self) -> Result<Vec<ChannelSpec>> {
        let tenant = self.catalog.current_tenant().map(|t| t.to_string());
        let entries = self
            .catalog
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    let tenant = tenant.clone();
                    Box::pin(async move {
                        let tref = tenant.as_deref();
                        // Candidate parents: own (if any) and global. Own rows
                        // sort first so the dedup-by-name keeps the tenant's row
                        // ahead of a global row of the same name.
                        let candidates: Vec<(Option<String>, String, i32)> = if let Some(t) = tref {
                            tx.query(
                                "SELECT tenant_id, channel_name, priority FROM evidence_channels \
                                 WHERE tenant_id = $1 OR tenant_id IS NULL \
                                 ORDER BY (tenant_id IS NOT NULL) DESC, priority, channel_name",
                                &[SqlValue::TextOwned(t.to_string())],
                                |row| {
                                    Ok((
                                        row.try_get::<String>("tenant_id")?,
                                        row.get::<String>("channel_name")?,
                                        row.get::<i32>("priority")?,
                                    ))
                                },
                            )
                            .await?
                        } else {
                            tx.query(
                                "SELECT tenant_id, channel_name, priority FROM evidence_channels \
                                 WHERE tenant_id IS NULL ORDER BY priority, channel_name",
                                &[],
                                |row| {
                                    Ok((
                                        row.try_get::<String>("tenant_id")?,
                                        row.get::<String>("channel_name")?,
                                        row.get::<i32>("priority")?,
                                    ))
                                },
                            )
                            .await?
                        };

                        // Dedup by channel_name, keeping the first occurrence
                        // (own shadows global because own rows are ordered first).
                        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                        let mut resolved: Vec<(Option<String>, String, i32)> =
                            Vec::with_capacity(candidates.len());
                        for (row_tenant, name, priority) in candidates {
                            if seen.insert(name.clone()) {
                                resolved.push((row_tenant, name, priority));
                            }
                        }

                        let mut out: Vec<ChannelListing> = Vec::with_capacity(resolved.len());
                        for (row_tenant, name, priority) in resolved {
                            let cols = if let Some(rt) = row_tenant {
                                tx.query(
                                    "SELECT column_name, column_type \
                                     FROM evidence_channel_columns \
                                     WHERE tenant_id = $1 AND channel_name = $2 ORDER BY ordinal",
                                    &[
                                        SqlValue::TextOwned(rt),
                                        SqlValue::TextOwned(name.clone()),
                                    ],
                                    read_column_row,
                                )
                                .await?
                            } else {
                                tx.query(
                                    "SELECT column_name, column_type \
                                     FROM evidence_channel_columns \
                                     WHERE tenant_id IS NULL AND channel_name = $1 ORDER BY ordinal",
                                    &[SqlValue::TextOwned(name.clone())],
                                    read_column_row,
                                )
                                .await?
                            };
                            out.push((name, priority, cols));
                        }
                        // Re-establish the documented (priority, channel_name)
                        // order after the own-first dedup pass.
                        out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                        Ok(out)
                    })
                },
            )
            .await?;

        let mut specs = Vec::with_capacity(entries.len());
        for (name, priority, raw_cols) in entries {
            // Both the channel slug and the column-type tokens here are READ
            // BACK from the store; a value that no longer parses is catalog
            // corruption, routed to an internal fault — never the caller-facing
            // `InvalidId` / `InvalidColumnType`, which `get()` (re-using caller
            // input) keeps.
            let id = ChannelId::new(&name).map_err(|_| {
                JammiError::Catalog(format!(
                    "stored channel name '{name}' is not a valid channel id"
                ))
            })?;
            let columns: Result<Vec<ChannelColumn>> = raw_cols
                .into_iter()
                .map(|(cname, ctype)| {
                    Ok(ChannelColumn {
                        name: cname,
                        data_type: ChannelColumnType::from_sql_str(&ctype)
                            .map_err(|t| stored_type_corruption(&name, &t))?,
                    })
                })
                .collect();
            specs.push(ChannelSpec {
                id,
                priority,
                columns: columns?,
            });
        }
        Ok(specs)
    }

    /// Build the Arrow schema produced by merging the given channels'
    /// declared columns in `(priority, ordinal)` order.
    pub async fn merged_schema(&self, participating: &[ChannelId]) -> Result<SchemaRef> {
        let mut specs: Vec<ChannelSpec> = Vec::with_capacity(participating.len());
        for id in participating {
            let spec = self.get(id).await?.ok_or_else(|| {
                JammiError::ChannelCatalog(ChannelCatalogError::NotRegistered(id.to_string()))
            })?;
            specs.push(spec);
        }
        specs.sort_by_key(|s| s.priority);

        let mut fields: Vec<Field> = Vec::new();
        for spec in specs {
            for col in spec.columns {
                fields.push(Field::new(&col.name, col.data_type.to_arrow(), true));
            }
        }
        Ok(Arc::new(Schema::new(fields)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn open_catalog() -> (tempfile::TempDir, Catalog) {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).await.unwrap();
        (dir, catalog)
    }

    #[test]
    fn type_to_and_from_arrow_round_trip() {
        for t in [
            ChannelColumnType::Float32,
            ChannelColumnType::Float64,
            ChannelColumnType::Int32,
            ChannelColumnType::Int64,
            ChannelColumnType::Utf8,
            ChannelColumnType::Boolean,
        ] {
            let arrow = t.to_arrow();
            assert_eq!(ChannelColumnType::from_arrow(&arrow).unwrap(), t);
        }
    }

    #[test]
    fn from_arrow_rejects_unsupported_type() {
        let err = ChannelColumnType::from_arrow(&DataType::UInt16).unwrap_err();
        assert!(matches!(err, JammiError::Catalog(_)));
    }

    #[test]
    fn from_sql_str_rejects_unknown_token() {
        assert!(ChannelColumnType::from_sql_str("Decimal").is_err());
    }

    #[test]
    fn serde_uses_pascal_case() {
        let json = serde_json::to_string(&ChannelColumnType::Float32).unwrap();
        assert_eq!(json, "\"Float32\"");
        let parsed: ChannelColumnType = serde_json::from_str("\"Utf8\"").unwrap();
        assert_eq!(parsed, ChannelColumnType::Utf8);
    }

    #[test]
    fn serde_rejects_unknown_variant_loudly() {
        let r: std::result::Result<ChannelColumnType, _> = serde_json::from_str("\"Decimal\"");
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn seed_channels_visible_via_list() {
        let (_dir, catalog) = open_catalog().await;
        let channels = catalog.channels().list().await.unwrap();
        let names: Vec<&str> = channels.iter().map(|c| c.id.as_str()).collect();
        assert!(names.contains(&"vector"));
        assert!(names.contains(&"inference"));
    }

    #[tokio::test]
    async fn vector_channel_has_similarity_column() {
        let (_dir, catalog) = open_catalog().await;
        let vector = catalog
            .channels()
            .get(&ChannelId::new("vector").unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(vector.columns.len(), 1);
        assert_eq!(vector.columns[0].name, "similarity");
        assert_eq!(vector.columns[0].data_type, ChannelColumnType::Float32);
    }

    #[tokio::test]
    async fn inference_channel_columns_ordered_by_ordinal() {
        let (_dir, catalog) = open_catalog().await;
        let inference = catalog
            .channels()
            .get(&ChannelId::new("inference").unwrap())
            .await
            .unwrap()
            .unwrap();
        let names: Vec<&str> = inference.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["inference_model", "inference_task", "inference_confidence"]
        );
    }

    #[tokio::test]
    async fn register_then_get_round_trip() {
        let (_dir, catalog) = open_catalog().await;
        let id = ChannelId::new("scored_by").unwrap();
        let spec = ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![
                ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Utf8,
                },
                ChannelColumn {
                    name: "rank_score".into(),
                    data_type: ChannelColumnType::Float32,
                },
            ],
        };
        catalog.channels().register(&spec).await.unwrap();

        let got = catalog.channels().get(&id).await.unwrap().unwrap();
        assert_eq!(got.priority, 3);
        assert_eq!(got.columns.len(), 2);
        assert_eq!(got.columns[0].name, "ranker");
        assert_eq!(got.columns[1].name, "rank_score");
    }

    #[tokio::test]
    async fn register_rejects_duplicate_channel() {
        let (_dir, catalog) = open_catalog().await;
        let id = ChannelId::new("scored_by").unwrap();
        let spec = ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        };
        catalog.channels().register(&spec).await.unwrap();
        let err = catalog.channels().register(&spec).await.unwrap_err();
        match err {
            JammiError::ChannelCatalog(ChannelCatalogError::AlreadyExists(c)) => {
                assert_eq!(c, "scored_by")
            }
            other => panic!("expected ChannelCatalog(AlreadyExists), got {other:?}"),
        }
    }

    /// A duplicate GLOBAL (unbound, `tenant = None`) registration is rejected the
    /// second time with the SAME "already exists" semantics as a tenant-scoped
    /// duplicate. `open_catalog` binds no tenant, so both `register` calls write
    /// `tenant_id IS NULL`; the partial unique index added in migration 020 is the
    /// DB-level backstop for this case (the composite `UNIQUE (tenant_id,
    /// channel_name)` treats NULLs as distinct and cannot catch it). Serial is
    /// sufficient to pin the constraint + error path — the concurrent Postgres
    /// race is closed by that same DB index, not by this test.
    #[tokio::test]
    async fn register_rejects_duplicate_global_channel() {
        let (_dir, catalog) = open_catalog().await;
        let spec = ChannelSpec {
            id: ChannelId::new("global_chan").unwrap(),
            priority: 5,
            columns: vec![ChannelColumn {
                name: "marker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        };
        catalog.channels().register(&spec).await.unwrap();
        let err = catalog.channels().register(&spec).await.unwrap_err();
        match err {
            JammiError::ChannelCatalog(ChannelCatalogError::AlreadyExists(c)) => assert_eq!(
                c, "global_chan",
                "global duplicate must reject with the same 'already exists' semantics"
            ),
            other => panic!("expected ChannelCatalog(AlreadyExists), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn add_columns_appends_with_continuous_ordinal() {
        let (_dir, catalog) = open_catalog().await;
        let id = ChannelId::new("scored_by").unwrap();
        let spec = ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        };
        catalog.channels().register(&spec).await.unwrap();

        catalog
            .channels()
            .add_columns(
                &id,
                &[ChannelColumn {
                    name: "rank_score".into(),
                    data_type: ChannelColumnType::Float32,
                }],
            )
            .await
            .unwrap();

        let got = catalog.channels().get(&id).await.unwrap().unwrap();
        assert_eq!(got.columns.len(), 2);
        assert_eq!(got.columns[0].name, "ranker");
        assert_eq!(got.columns[1].name, "rank_score");
    }

    #[tokio::test]
    async fn add_columns_rejects_redeclaration_with_same_type() {
        let (_dir, catalog) = open_catalog().await;
        let id = ChannelId::new("scored_by").unwrap();
        catalog
            .channels()
            .register(&ChannelSpec {
                id: id.clone(),
                priority: 3,
                columns: vec![ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Utf8,
                }],
            })
            .await
            .unwrap();

        let err = catalog
            .channels()
            .add_columns(
                &id,
                &[ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Utf8,
                }],
            )
            .await
            .unwrap_err();
        match err {
            JammiError::ChannelCatalog(ChannelCatalogError::ColumnAlreadyDeclared {
                column,
                ty,
                ..
            }) => {
                assert_eq!(column, "ranker");
                assert_eq!(ty, ChannelColumnType::Utf8);
            }
            other => panic!("expected ChannelCatalog(ColumnAlreadyDeclared), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn add_columns_rejects_redeclaration_with_different_type() {
        let (_dir, catalog) = open_catalog().await;
        let id = ChannelId::new("scored_by").unwrap();
        catalog
            .channels()
            .register(&ChannelSpec {
                id: id.clone(),
                priority: 3,
                columns: vec![ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Utf8,
                }],
            })
            .await
            .unwrap();

        let err = catalog
            .channels()
            .add_columns(
                &id,
                &[ChannelColumn {
                    name: "ranker".into(),
                    data_type: ChannelColumnType::Int32,
                }],
            )
            .await
            .unwrap_err();
        // The contiguous "cannot redeclare as <type>" substring is load-bearing
        // for the CLI / cookbook / db it-tests, so pin it on the Display too.
        assert!(err.to_string().contains("cannot redeclare as Int32"));
        match err {
            JammiError::ChannelCatalog(ChannelCatalogError::ColumnConflict {
                column,
                existing,
                requested,
                ..
            }) => {
                assert_eq!(column, "ranker");
                assert_eq!(existing, ChannelColumnType::Utf8);
                assert_eq!(requested, ChannelColumnType::Int32);
            }
            other => panic!("expected ChannelCatalog(ColumnConflict), got {other:?}"),
        }
    }

    #[tokio::test]
    async fn merged_schema_orders_by_priority_then_ordinal() {
        let (_dir, catalog) = open_catalog().await;
        let vector = ChannelId::new("vector").unwrap();
        let inference = ChannelId::new("inference").unwrap();
        let schema = catalog
            .channels()
            .merged_schema(&[inference.clone(), vector.clone()])
            .await
            .unwrap();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            names,
            vec![
                "similarity",
                "inference_model",
                "inference_task",
                "inference_confidence",
            ]
        );
    }

    #[tokio::test]
    async fn merged_schema_errors_on_unregistered_channel() {
        let (_dir, catalog) = open_catalog().await;
        let err = catalog
            .channels()
            .merged_schema(&[ChannelId::new("nonexistent").unwrap()])
            .await
            .unwrap_err();
        match err {
            JammiError::ChannelCatalog(ChannelCatalogError::NotRegistered(c)) => {
                assert_eq!(c, "nonexistent")
            }
            other => panic!("expected ChannelCatalog(NotRegistered), got {other:?}"),
        }
    }
}
