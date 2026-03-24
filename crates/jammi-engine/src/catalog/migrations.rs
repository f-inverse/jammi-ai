use rusqlite_migration::{Migrations, M};

use super::schema;
use crate::error::Result;

static MIGRATIONS: &[M<'static>] = &[
    M::up(schema::MIGRATION_001_CORE_TABLES),
    M::up(schema::MIGRATION_002_RESULT_TABLES),
];

/// Build the full migration set for the catalog database.
pub(crate) fn migrations() -> Migrations<'static> {
    Migrations::from_slice(MIGRATIONS)
}

/// Apply all pending migrations to bring the database to the latest schema version.
pub(crate) fn run(conn: &mut rusqlite::Connection) -> Result<()> {
    migrations().to_latest(conn)?;
    Ok(())
}
