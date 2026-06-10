//! `jammi query` / `jammi explain` subcommands.
//!
//! Both run SQL over the remote [`Session`]: `query` runs the statement as-is,
//! `explain` prepends `EXPLAIN`. The session drives the server's Flight SQL
//! lane, so the result batches come back over the wire.

use jammi_ai::Session;

pub async fn run(session: &Session, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
    let batches = session.sql(sql).await?;
    print_batches(&batches)?;
    if batches.is_empty() {
        println!("(0 rows)");
    }
    Ok(())
}

pub async fn explain(session: &Session, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
    let explain_sql = format!("EXPLAIN {sql}");
    let batches = session.sql(&explain_sql).await?;
    print_batches(&batches)?;
    Ok(())
}

fn print_batches(batches: &[arrow::array::RecordBatch]) -> Result<(), Box<dyn std::error::Error>> {
    for batch in batches {
        let formatted = arrow::util::pretty::pretty_format_batches(std::slice::from_ref(batch))?;
        println!("{formatted}");
    }
    Ok(())
}
