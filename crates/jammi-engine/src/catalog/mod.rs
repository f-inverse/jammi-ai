pub mod eval_repo;
pub mod fine_tune_repo;
pub mod migrations;
pub mod model_repo;
pub mod result_repo;
pub mod schema;
pub mod source_repo;
pub mod status;

use std::path::Path;

use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;

use crate::error::{JammiError, Result};

/// SQLite-backed artifact catalog for models, sources, and experiment metadata.
///
/// Provides connection-pooled access to a WAL-mode SQLite database stored
/// inside the configured artifact directory.
pub struct Catalog {
    pool: Pool<SqliteConnectionManager>,
}

impl Catalog {
    /// Open (or create) the catalog database in `artifact_dir`, running any pending migrations.
    pub fn open(artifact_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(artifact_dir)?;
        let db_path = artifact_dir.join("catalog.db");
        let manager = SqliteConnectionManager::file(&db_path).with_init(|conn| {
            conn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA busy_timeout=5000;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA foreign_keys=ON;",
            )
        });
        let pool = Pool::builder().max_size(8).build(manager)?;

        let mut conn = pool.get().map_err(|e| JammiError::Catalog(e.to_string()))?;
        migrations::run(&mut conn)?;

        Ok(Self { pool })
    }

    /// Acquire a pooled SQLite connection.
    pub(crate) fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().map_err(JammiError::Pool)
    }

    /// List registered evidence channel names, ordered by priority.
    pub fn evidence_channel_names(&self) -> Result<Vec<String>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT channel_name FROM evidence_channels ORDER BY priority")?;
        let rows = stmt.query_map([], |row| row.get(0))?;
        rows.map(|r| r.map_err(Into::into)).collect()
    }

    /// Fetch an evidence channel by name.
    pub fn get_evidence_channel(&self, name: &str) -> Result<EvidenceChannelRecord> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT channel_name, schema_json, priority FROM evidence_channels WHERE channel_name = ?1",
        )?;
        stmt.query_row(rusqlite::params![name], |row| {
            Ok(EvidenceChannelRecord {
                channel_name: row.get(0)?,
                schema_json: row.get(1)?,
                priority: row.get(2)?,
            })
        })
        .map_err(|e| JammiError::Catalog(format!("Evidence channel '{name}': {e}")))
    }
}

/// A row from the `evidence_channels` catalog table.
#[derive(Debug, Clone)]
pub struct EvidenceChannelRecord {
    pub channel_name: String,
    pub schema_json: String,
    pub priority: i32,
}
