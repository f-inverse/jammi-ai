# Connect via Flight SQL

Run a query against a remote `jammi-server` over Arrow Flight SQL.

**When to use this pattern.** You're connecting from a non-Python client
(Tableau, dbt, JDBC tools, Rust binaries), or you want to expose Jammi
to multiple readers without each one holding an embedded session. The
same protocol is what `dbt-flightsql`, the official Flight SQL JDBC
driver, and BI tools speak natively.

The program uses `pyarrow.flight.FlightClient` alone: `get_flight_info` with
the encoded statement, then `do_get` on the ticket it returns. It needs a
`jammi-server` on PATH — `pip install jammi-server`, or
`cargo build --release -p jammi-server` with `target/release` on PATH.

## Run it

```bash
pip install jammi-server
python cookbook/recipes/flight_sql/example.py
```

It prints the one-row result of `SELECT 1 AS one`.
