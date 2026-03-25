use jammi_ai::session::InferenceSession;
use jammi_engine::config::JammiConfig;

pub async fn run(config: JammiConfig, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
    let session = InferenceSession::new(config).await?;
    let batches = session.sql(sql).await?;

    for batch in &batches {
        let formatted = arrow::util::pretty::pretty_format_batches(std::slice::from_ref(batch))?;
        println!("{formatted}");
    }

    if batches.is_empty() {
        println!("(0 rows)");
    }

    Ok(())
}

pub async fn explain(config: JammiConfig, sql: &str) -> Result<(), Box<dyn std::error::Error>> {
    let session = InferenceSession::new(config).await?;
    let explain_sql = format!("EXPLAIN {sql}");
    let batches = session.sql(&explain_sql).await?;

    for batch in &batches {
        let formatted = arrow::util::pretty::pretty_format_batches(std::slice::from_ref(batch))?;
        println!("{formatted}");
    }

    Ok(())
}
