pub mod migrations;
pub mod model_repo;
pub mod schema;
pub mod source_repo;

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
    pub fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool.get().map_err(JammiError::Pool)
    }
}
