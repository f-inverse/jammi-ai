//! Catalog repository for evidence channels and their declared columns.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use serde::{Deserialize, Serialize};

use crate::error::{JammiError, Result};
use crate::evidence_channel::ChannelId;

use super::Catalog;

/// The closed set of Arrow types a channel column may declare.
///
/// New variants are added only when a third-tenant use case requires
/// them, never speculatively — see [`CLAUDE.md` — *Type-driven design*].
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
    /// Project to the corresponding Arrow `DataType`.
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

    /// Project an Arrow `DataType` back to the closed enum, rejecting any
    /// type a channel is not allowed to declare.
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

    /// SQL-stored token (matches `evidence_channel_columns.column_type`).
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

    fn from_sql_str(s: &str) -> Result<Self> {
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

impl<'a> ChannelRepo<'a> {
    pub(super) fn new(catalog: &'a Catalog) -> Self {
        Self { catalog }
    }

    /// Register a new channel and its columns atomically.
    ///
    /// Errors if the channel id is already registered (the SQL
    /// `PRIMARY KEY` on `evidence_channels.channel_name` enforces this).
    pub fn register(&self, spec: &ChannelSpec) -> Result<()> {
        let mut conn = self.catalog.conn()?;
        let tx = conn.transaction()?;

        tx.execute(
            "INSERT INTO evidence_channels (channel_name, priority) VALUES (?1, ?2)",
            rusqlite::params![spec.id.as_str(), spec.priority],
        )
        .map_err(|e| match e {
            rusqlite::Error::SqliteFailure(_, Some(ref m))
                if m.contains("UNIQUE constraint failed") =>
            {
                JammiError::EvidenceChannel(format!("channel '{}': already exists", spec.id))
            }
            other => JammiError::Sqlite(other),
        })?;

        for (ordinal, col) in spec.columns.iter().enumerate() {
            tx.execute(
                "INSERT INTO evidence_channel_columns \
                 (channel_name, column_name, column_type, ordinal) \
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![
                    spec.id.as_str(),
                    col.name,
                    col.data_type.as_str(),
                    ordinal as i64,
                ],
            )?;
        }

        tx.commit()?;
        Ok(())
    }

    /// Append new columns to an already-registered channel.
    ///
    /// Append-only: any attempt to redeclare an existing column name —
    /// whether with the same or a different dtype — is rejected with a
    /// typed `EvidenceChannel` error.
    pub fn add_columns(&self, channel: &ChannelId, new_columns: &[ChannelColumn]) -> Result<()> {
        let mut conn = self.catalog.conn()?;
        let tx = conn.transaction()?;

        // Existence check on the parent row.
        let parent_exists: bool = tx
            .query_row(
                "SELECT 1 FROM evidence_channels WHERE channel_name = ?1",
                rusqlite::params![channel.as_str()],
                |_| Ok(true),
            )
            .or_else(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => Ok(false),
                other => Err(JammiError::Sqlite(other)),
            })?;
        if !parent_exists {
            return Err(JammiError::EvidenceChannel(format!(
                "channel '{channel}': not registered"
            )));
        }

        // Look up the current max ordinal so the append produces a
        // contiguous sequence even after multiple add_columns calls.
        let max_ordinal: Option<i64> = tx.query_row(
            "SELECT MAX(ordinal) FROM evidence_channel_columns WHERE channel_name = ?1",
            rusqlite::params![channel.as_str()],
            |row| row.get::<_, Option<i64>>(0),
        )?;
        let mut next = max_ordinal.unwrap_or(-1) + 1;

        for col in new_columns {
            // Distinguish "already declared" from "different dtype" so
            // the caller gets a precise error.
            let existing: Option<String> = tx
                .query_row(
                    "SELECT column_type FROM evidence_channel_columns \
                     WHERE channel_name = ?1 AND column_name = ?2",
                    rusqlite::params![channel.as_str(), col.name],
                    |row| row.get::<_, String>(0),
                )
                .ok();
            if let Some(existing_type) = existing {
                let existing = ChannelColumnType::from_sql_str(&existing_type)?;
                if existing == col.data_type {
                    return Err(JammiError::EvidenceChannel(format!(
                        "channel '{channel}': column '{}' already declared",
                        col.name
                    )));
                } else {
                    return Err(JammiError::EvidenceChannel(format!(
                        "channel '{channel}': column '{}' was declared {}, \
                         cannot redeclare as {}",
                        col.name,
                        existing.as_str(),
                        col.data_type.as_str(),
                    )));
                }
            }

            tx.execute(
                "INSERT INTO evidence_channel_columns \
                 (channel_name, column_name, column_type, ordinal) \
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![channel.as_str(), col.name, col.data_type.as_str(), next],
            )?;
            next += 1;
        }

        tx.commit()?;
        Ok(())
    }

    /// Look up one channel's full spec.
    pub fn get(&self, channel: &ChannelId) -> Result<Option<ChannelSpec>> {
        let conn = self.catalog.conn()?;

        let priority: Option<i32> = conn
            .query_row(
                "SELECT priority FROM evidence_channels WHERE channel_name = ?1",
                rusqlite::params![channel.as_str()],
                |row| row.get(0),
            )
            .ok();
        let Some(priority) = priority else {
            return Ok(None);
        };

        let mut stmt = conn.prepare(
            "SELECT column_name, column_type FROM evidence_channel_columns \
             WHERE channel_name = ?1 ORDER BY ordinal",
        )?;
        let columns: Vec<ChannelColumn> = stmt
            .query_map(rusqlite::params![channel.as_str()], |row| {
                let name: String = row.get(0)?;
                let type_str: String = row.get(1)?;
                Ok((name, type_str))
            })?
            .map(|r| {
                let (name, type_str) = r?;
                Ok(ChannelColumn {
                    name,
                    data_type: ChannelColumnType::from_sql_str(&type_str)?,
                })
            })
            .collect::<Result<_>>()?;

        Ok(Some(ChannelSpec {
            id: channel.clone(),
            priority,
            columns,
        }))
    }

    /// List every registered channel, ordered by `(priority, ordinal)`.
    pub fn list(&self) -> Result<Vec<ChannelSpec>> {
        let conn = self.catalog.conn()?;

        let mut stmt = conn.prepare(
            "SELECT channel_name, priority FROM evidence_channels ORDER BY priority, channel_name",
        )?;
        let parents: Vec<(String, i32)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i32>(1)?))
            })?
            .collect::<rusqlite::Result<_>>()?;

        let mut specs = Vec::with_capacity(parents.len());
        for (name, priority) in parents {
            let id = ChannelId::new(name.clone())?;
            let mut col_stmt = conn.prepare(
                "SELECT column_name, column_type FROM evidence_channel_columns \
                 WHERE channel_name = ?1 ORDER BY ordinal",
            )?;
            let columns: Vec<ChannelColumn> = col_stmt
                .query_map(rusqlite::params![name], |row| {
                    let cname: String = row.get(0)?;
                    let ctype: String = row.get(1)?;
                    Ok((cname, ctype))
                })?
                .map(|r| {
                    let (cname, ctype) = r?;
                    Ok(ChannelColumn {
                        name: cname,
                        data_type: ChannelColumnType::from_sql_str(&ctype)?,
                    })
                })
                .collect::<Result<_>>()?;
            specs.push(ChannelSpec {
                id,
                priority,
                columns,
            });
        }

        Ok(specs)
    }

    /// Build the Arrow schema produced by merging the given channels'
    /// declared columns in `(priority, ordinal)` order. Unregistered
    /// ids produce an error.
    pub fn merged_schema(&self, participating: &[ChannelId]) -> Result<SchemaRef> {
        let mut specs: Vec<ChannelSpec> = Vec::with_capacity(participating.len());
        for id in participating {
            let spec = self.get(id)?.ok_or_else(|| {
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

    fn open_catalog() -> (tempfile::TempDir, Catalog) {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).unwrap();
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

    #[test]
    fn seed_channels_visible_via_list() {
        let (_dir, catalog) = open_catalog();
        let channels = catalog.channels().list().unwrap();
        let names: Vec<&str> = channels.iter().map(|c| c.id.as_str()).collect();
        assert!(names.contains(&"vector"));
        assert!(names.contains(&"inference"));
    }

    #[test]
    fn vector_channel_has_similarity_column() {
        let (_dir, catalog) = open_catalog();
        let vector = catalog
            .channels()
            .get(&ChannelId::new("vector").unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(vector.columns.len(), 1);
        assert_eq!(vector.columns[0].name, "similarity");
        assert_eq!(vector.columns[0].data_type, ChannelColumnType::Float32);
    }

    #[test]
    fn inference_channel_columns_ordered_by_ordinal() {
        let (_dir, catalog) = open_catalog();
        let inference = catalog
            .channels()
            .get(&ChannelId::new("inference").unwrap())
            .unwrap()
            .unwrap();
        let names: Vec<&str> = inference.columns.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["inference_model", "inference_task", "inference_confidence"]
        );
    }

    #[test]
    fn register_then_get_round_trip() {
        let (_dir, catalog) = open_catalog();
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
        catalog.channels().register(&spec).unwrap();

        let got = catalog.channels().get(&id).unwrap().unwrap();
        assert_eq!(got.priority, 3);
        assert_eq!(got.columns.len(), 2);
        assert_eq!(got.columns[0].name, "ranker");
        assert_eq!(got.columns[1].name, "rank_score");
    }

    #[test]
    fn register_rejects_duplicate_channel() {
        let (_dir, catalog) = open_catalog();
        let id = ChannelId::new("scored_by").unwrap();
        let spec = ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        };
        catalog.channels().register(&spec).unwrap();
        let err = catalog.channels().register(&spec).unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => assert!(m.contains("already exists")),
            other => panic!("expected EvidenceChannel(already exists), got {other:?}"),
        }
    }

    #[test]
    fn add_columns_appends_with_continuous_ordinal() {
        let (_dir, catalog) = open_catalog();
        let id = ChannelId::new("scored_by").unwrap();
        let spec = ChannelSpec {
            id: id.clone(),
            priority: 3,
            columns: vec![ChannelColumn {
                name: "ranker".into(),
                data_type: ChannelColumnType::Utf8,
            }],
        };
        catalog.channels().register(&spec).unwrap();

        catalog
            .channels()
            .add_columns(
                &id,
                &[ChannelColumn {
                    name: "rank_score".into(),
                    data_type: ChannelColumnType::Float32,
                }],
            )
            .unwrap();

        let got = catalog.channels().get(&id).unwrap().unwrap();
        assert_eq!(got.columns.len(), 2);
        assert_eq!(got.columns[0].name, "ranker");
        assert_eq!(got.columns[1].name, "rank_score");
    }

    #[test]
    fn add_columns_rejects_redeclaration_with_same_type() {
        let (_dir, catalog) = open_catalog();
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
            .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => assert!(m.contains("already declared")),
            other => panic!("expected EvidenceChannel(already declared), got {other:?}"),
        }
    }

    #[test]
    fn add_columns_rejects_redeclaration_with_different_type() {
        let (_dir, catalog) = open_catalog();
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

    #[test]
    fn merged_schema_orders_by_priority_then_ordinal() {
        let (_dir, catalog) = open_catalog();
        let vector = ChannelId::new("vector").unwrap();
        let inference = ChannelId::new("inference").unwrap();
        let schema = catalog
            .channels()
            .merged_schema(&[inference.clone(), vector.clone()])
            .unwrap();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        // vector (priority 1) → "similarity" first; inference (priority 2) → three columns next.
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

    #[test]
    fn merged_schema_errors_on_unregistered_channel() {
        let (_dir, catalog) = open_catalog();
        let err = catalog
            .channels()
            .merged_schema(&[ChannelId::new("nonexistent").unwrap()])
            .unwrap_err();
        match err {
            JammiError::EvidenceChannel(m) => assert!(m.contains("not registered")),
            other => panic!("expected EvidenceChannel(not registered), got {other:?}"),
        }
    }
}
