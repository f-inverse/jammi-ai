//! Catalog repository for evidence channels and their declared columns.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};

use crate::error::{JammiError, Result};
use crate::evidence_channel::ChannelId;

use super::backend::{BackendError, Row, SqlValue, TxOptions};
use super::Catalog;

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
            other => Err(JammiError::EvidenceChannel(format!(
                "unsupported channel column type: {other:?}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
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
    pub fn from_sql_str(s: &str) -> Result<Self> {
        match s {
            "Float32" => Ok(Self::Float32),
            "Float64" => Ok(Self::Float64),
            "Int32" => Ok(Self::Int32),
            "Int64" => Ok(Self::Int64),
            "Utf8" => Ok(Self::Utf8),
            "Boolean" => Ok(Self::Boolean),
            other => Err(JammiError::EvidenceChannel(format!(
                "unknown channel column type stored in catalog: '{other}'"
            ))),
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
            JammiError::EvidenceChannel(format!("channel '{channel}': already exists"))
        }
        other => JammiError::BackendDriver(other),
    }
}

fn read_column_row(row: &Row<'_>) -> std::result::Result<(String, String), BackendError> {
    Ok((row.get("column_name")?, row.get("column_type")?))
}

/// `(channel_name, priority, columns)` row as returned by `list()`'s catalog
/// query. Aliased to keep the inner closure's local type readable.
type ChannelListing = (String, i64, Vec<(String, String)>);

impl<'a> ChannelRepo<'a> {
    pub(super) fn new(catalog: &'a Catalog) -> Self {
        Self { catalog }
    }

    /// Register a new channel and its columns atomically.
    pub async fn register(&self, spec: &ChannelSpec) -> Result<()> {
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
                Box::pin(async move {
                    tx.execute(
                        "INSERT INTO evidence_channels (channel_name, priority) VALUES ($1, $2)",
                        &[
                            SqlValue::TextOwned(channel.clone()),
                            SqlValue::Int(priority),
                        ],
                    )
                    .await?;

                    for (name, ty, ord) in columns {
                        tx.execute(
                            "INSERT INTO evidence_channel_columns \
                             (channel_name, column_name, column_type, ordinal) \
                             VALUES ($1, $2, $3, $4)",
                            &[
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
    pub async fn add_columns(
        &self,
        channel: &ChannelId,
        new_columns: &[ChannelColumn],
    ) -> Result<()> {
        let channel_name = channel.as_str().to_string();
        let channel_for_err = channel_name.clone();
        let cols: Vec<(String, ChannelColumnType)> = new_columns
            .iter()
            .map(|c| (c.name.clone(), c.data_type))
            .collect();

        self.catalog
            .backend()
            .transaction(TxOptions::default(), |tx| {
                Box::pin(async move {
                    // Existence check on the parent row.
                    let parent_exists = tx
                        .query_opt(
                            "SELECT 1 AS one FROM evidence_channels WHERE channel_name = $1",
                            &[SqlValue::TextOwned(channel_name.clone())],
                            |row| row.get::<i64>("one"),
                        )
                        .await?
                        .is_some();
                    if !parent_exists {
                        return Err(BackendError::Execution(format!(
                            "channel '{channel_for_err}': not registered"
                        )));
                    }

                    let max_ord = tx
                        .query_opt(
                            "SELECT MAX(ordinal) AS m FROM evidence_channel_columns WHERE channel_name = $1",
                            &[SqlValue::TextOwned(channel_name.clone())],
                            |row| row.try_get::<i64>("m"),
                        )
                        .await?
                        .flatten();
                    let mut next = max_ord.unwrap_or(-1) + 1;

                    for (name, ty) in cols {
                        let existing = tx
                            .query_opt(
                                "SELECT column_type FROM evidence_channel_columns \
                                 WHERE channel_name = $1 AND column_name = $2",
                                &[
                                    SqlValue::TextOwned(channel_name.clone()),
                                    SqlValue::TextOwned(name.clone()),
                                ],
                                |row| row.get::<String>("column_type"),
                            )
                            .await?;
                        if let Some(existing_type) = existing {
                            let existing = ChannelColumnType::from_sql_str(&existing_type)
                                .map_err(|e| BackendError::Execution(e.to_string()))?;
                            if existing == ty {
                                return Err(BackendError::Execution(format!(
                                    "channel '{channel_for_err}': column '{name}' already declared"
                                )));
                            } else {
                                return Err(BackendError::Execution(format!(
                                    "channel '{channel_for_err}': column '{name}' was declared {}, \
                                     cannot redeclare as {}",
                                    existing.as_str(),
                                    ty.as_str(),
                                )));
                            }
                        }

                        tx.execute(
                            "INSERT INTO evidence_channel_columns \
                             (channel_name, column_name, column_type, ordinal) \
                             VALUES ($1, $2, $3, $4)",
                            &[
                                SqlValue::TextOwned(channel_name.clone()),
                                SqlValue::TextOwned(name),
                                SqlValue::Text(ty.as_str()),
                                SqlValue::Int(next),
                            ],
                        )
                        .await?;
                        next += 1;
                    }
                    Ok(())
                })
            })
            .await
            .map_err(|e| match e {
                BackendError::Execution(m) => JammiError::EvidenceChannel(m),
                other => JammiError::BackendDriver(other),
            })?;
        Ok(())
    }

    /// Look up one channel's full spec.
    pub async fn get(&self, channel: &ChannelId) -> Result<Option<ChannelSpec>> {
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
                    Box::pin(async move {
                        let priority = tx
                            .query_opt(
                                "SELECT priority FROM evidence_channels WHERE channel_name = $1",
                                &[SqlValue::TextOwned(channel_name.clone())],
                                |row| row.get::<i64>("priority"),
                            )
                            .await?;
                        let Some(priority) = priority else {
                            return Ok(None);
                        };
                        let cols = tx
                            .query(
                                "SELECT column_name, column_type FROM evidence_channel_columns \
                             WHERE channel_name = $1 ORDER BY ordinal",
                                &[SqlValue::TextOwned(channel_name)],
                                read_column_row,
                            )
                            .await?;
                        Ok(Some((priority, cols)))
                    })
                },
            )
            .await?;

        let Some((priority, raw_cols)) = found else {
            return Ok(None);
        };
        let columns: Result<Vec<ChannelColumn>> = raw_cols
            .into_iter()
            .map(|(name, ty)| {
                Ok(ChannelColumn {
                    name,
                    data_type: ChannelColumnType::from_sql_str(&ty)?,
                })
            })
            .collect();
        Ok(Some(ChannelSpec {
            id,
            priority: priority as i32,
            columns: columns?,
        }))
    }

    /// List every registered channel, ordered by `(priority, channel_name)`.
    pub async fn list(&self) -> Result<Vec<ChannelSpec>> {
        let entries = self
            .catalog
            .backend()
            .transaction(
                TxOptions {
                    read_only: true,
                    ..Default::default()
                },
                |tx| {
                    Box::pin(async move {
                        let parents: Vec<(String, i64)> = tx
                            .query(
                                "SELECT channel_name, priority FROM evidence_channels \
                             ORDER BY priority, channel_name",
                                &[],
                                |row| {
                                    Ok((
                                        row.get::<String>("channel_name")?,
                                        row.get::<i64>("priority")?,
                                    ))
                                },
                            )
                            .await?;
                        let mut out: Vec<ChannelListing> = Vec::with_capacity(parents.len());
                        for (name, priority) in parents {
                            let cols = tx
                            .query(
                                "SELECT column_name, column_type FROM evidence_channel_columns \
                                 WHERE channel_name = $1 ORDER BY ordinal",
                                &[SqlValue::TextOwned(name.clone())],
                                read_column_row,
                            )
                            .await?;
                            out.push((name, priority, cols));
                        }
                        Ok(out)
                    })
                },
            )
            .await?;

        let mut specs = Vec::with_capacity(entries.len());
        for (name, priority, raw_cols) in entries {
            let id = ChannelId::new(name)?;
            let columns: Result<Vec<ChannelColumn>> = raw_cols
                .into_iter()
                .map(|(cname, ctype)| {
                    Ok(ChannelColumn {
                        name: cname,
                        data_type: ChannelColumnType::from_sql_str(&ctype)?,
                    })
                })
                .collect();
            specs.push(ChannelSpec {
                id,
                priority: priority as i32,
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
                JammiError::EvidenceChannel(format!("channel '{id}': not registered"))
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
        assert!(matches!(err, JammiError::EvidenceChannel(_)));
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
            JammiError::EvidenceChannel(m) => assert!(m.contains("already exists")),
            other => panic!("expected EvidenceChannel(already exists), got {other:?}"),
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
            JammiError::EvidenceChannel(m) => assert!(m.contains("already declared")),
            other => panic!("expected EvidenceChannel(already declared), got {other:?}"),
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
        match err {
            JammiError::EvidenceChannel(m) => {
                assert!(m.contains("cannot redeclare"));
                assert!(m.contains("Utf8"));
                assert!(m.contains("Int32"));
            }
            other => panic!("expected EvidenceChannel(cannot redeclare), got {other:?}"),
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
            JammiError::EvidenceChannel(m) => assert!(m.contains("not registered")),
            other => panic!("expected EvidenceChannel(not registered), got {other:?}"),
        }
    }
}
