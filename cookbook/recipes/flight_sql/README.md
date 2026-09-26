# Connect via Flight SQL

Run a query against a remote `jammi-server` over Arrow Flight SQL.

**When to use this pattern.** You're connecting from a non-Python client
(Tableau, dbt, JDBC tools, Rust binaries), or you want to expose Jammi
to multiple readers without each one holding an embedded session. The
same protocol is what `dbt-flightsql`, the official Flight SQL JDBC
driver, and BI tools speak natively.

## What `example.py` does

1. Starts a `jammi-server` over a temp `artifact_dir` with
   `jammi.testing.LiveServer`, which binds kernel-assigned ports and waits
   until the server answers a handshake
2. Opens a `pyarrow.flight.FlightClient` against the server's endpoint
3. Submits `SELECT 1 AS one` over Flight SQL and confirms the response
4. Stops the server and waits for it to exit

## Prerequisites

- A `jammi-server` binary on PATH: `pip install jammi-server`, or
  `cargo build --release -p jammi-server` with `target/release` on PATH
- `pip install pyarrow` (already a `jammi-ai` dependency)

The script auto-detects `JAMMI_BIN` (env var) or falls back to the
workspace's `target/release/jammi-server`.

## API surface exercised

- `pyarrow.flight.FlightClient.execute(query)` over the Flight SQL
  command dialect
- `jammi-server` — the OSS deployment-shape binary entrypoint

## Run it

```bash
cargo build --release -p jammi-server      # one-time build
python cookbook/recipes/flight_sql/example.py
```

Exits 0 on success, prints the query result + `flight_sql: OK`.
