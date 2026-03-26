# jammi-cli

Command-line interface for [Jammi AI](https://github.com/f-inverse/jammi-ai).

Manage data sources, run SQL queries, and start the Arrow Flight SQL server from your terminal.

## Install

```bash
cargo install jammi-cli
```

## Usage

```bash
# Register a data source
jammi sources add patents --path /data/patents.parquet --format parquet

# Query with SQL
jammi query "SELECT id, title, year FROM patents.public.patents WHERE year > 2020"

# Show execution plan
jammi explain "SELECT * FROM patents.public.patents WHERE year > 2020"

# List sources and models
jammi sources list
jammi models list

# Start the Flight SQL server
jammi serve
```

## Documentation

See the [Jammi AI Cookbook](https://f-inverse.github.io/jammi-ai/) for the full guide.

## License

Apache-2.0
