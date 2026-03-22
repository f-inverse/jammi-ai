use rusqlite_migration::{Migrations, M};

use super::schema;
use crate::error::Result;

static MIGRATIONS: &[M<'static>] = &[M::up(schema::MIGRATION_001_CORE_TABLES)];

pub fn migrations() -> Migrations<'static> {
    Migrations::from_slice(MIGRATIONS)
}

pub fn run(conn: &mut rusqlite::Connection) -> Result<()> {
    migrations().to_latest(conn)?;
    Ok(())
}
