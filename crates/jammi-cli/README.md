# jammi-cli

Command-line interface for [Jammi AI](https://github.com/f-inverse/jammi-ai).

A strict gRPC client: `jammi` talks to a running `jammi-server` over the wire
(`--target`) and never touches the catalog or storage in-process. Manage data
sources, run SQL queries, and drive the trigger / channel / mutable-table
surfaces from your terminal.

## Install

```bash
cargo install jammi-cli
```

## Usage

```bash
# Point at a running server (default --target is grpc://127.0.0.1:8081)
export TARGET=grpc://127.0.0.1:8081

# Report server capabilities and confirm reachability
jammi --target $TARGET status

# Register a data source
jammi --target $TARGET sources add patents --url /data/patents.parquet --format parquet

# Query with SQL
jammi --target $TARGET query "SELECT id, title, year FROM patents.public.patents WHERE year > 2020"

# Show execution plan
jammi --target $TARGET explain "SELECT * FROM patents.public.patents WHERE year > 2020"

# List sources and models
jammi --target $TARGET sources list
jammi --target $TARGET models list
```

Run the server itself with the `jammi-server` binary (or a container image),
not this CLI.

## Documentation

See the [Jammi AI Cookbook](https://f-inverse.github.io/jammi-ai/) for the full guide.

## License

Apache-2.0
