# Connect via Flight SQL

Run a query against a remote `jammi` server over Arrow Flight SQL.

**When to use this pattern.** You're connecting from a non-Python client
(Tableau, dbt, JDBC tools, Rust binaries), or you want to expose Jammi
to multiple readers without each one holding an embedded session. The
same protocol is what `dbt-flightsql`, the official Flight SQL JDBC
driver, and BI tools speak natively.

## What `example.py` does

1. Spawns `target/release/jammi serve` as a child process pointed at a
   temp `artifact_dir`
2. Polls the health endpoint (`http://127.0.0.1:8080/health`) until the
   server is ready (5 s budget)
3. Opens a `pyarrow.flight.FlightClient` against `grpc://127.0.0.1:8081`
4. Submits `SELECT 1 AS one` over Flight SQL and confirms the response
5. Tears down the server process cleanly

This recipe is gated out of the per-PR CI matrix — it depends on the
`jammi` binary being built (`cargo build --release -p jammi-cli`), and
the build cost dominates the test wall-clock. The nightly cookbook job
builds the binary and runs the recipe behind `JAMMI_COOKBOOK_SLOW=1`.

## Prerequisites

- `cargo build --release -p jammi-cli` — produces `target/release/jammi`
- `pip install pyarrow` (already a `jammi-ai` dependency)

The script auto-detects `JAMMI_BIN` (env var) or falls back to the
workspace's `target/release/jammi`.

## API surface exercised

- `pyarrow.flight.FlightClient.execute(query)` over the Flight SQL
  command dialect
- `jammi serve` — the OSS deployment-shape binary entrypoint

## Run it

```bash
cargo build --release -p jammi-cli      # one-time build
python cookbook/recipes/flight_sql/example.py
```

Exits 0 on success, prints the query result + `flight_sql: OK`.
