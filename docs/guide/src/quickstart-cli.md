# Quickstart: CLI

The `jammi` CLI lets you manage sources, run queries, and start the server from your terminal.

## Register a source and query it

```bash
# Register a local Parquet file
jammi sources add patents --path /path/to/patents.parquet --format parquet

# List registered sources
jammi sources list

# Run a SQL query
jammi query "SELECT id, title, year FROM patents.public.patents WHERE year > 2020 LIMIT 5"

# Show the execution plan
jammi explain "SELECT * FROM patents.public.patents WHERE year > 2020"
```

## Start the server

```bash
# Start Flight SQL (port 8081) + health probe (port 8080)
jammi serve

# With a custom config
jammi --config jammi.toml serve
```

Once the server is running, any Arrow Flight SQL client can connect on port 8081 to query your data.

## Available commands

| Command | Description |
|---------|-------------|
| `jammi sources list` | List registered data sources |
| `jammi sources add <NAME> --path <PATH> --format <FMT>` | Register a local file |
| `jammi models list` | List registered models |
| `jammi query "<SQL>"` | Run a SQL query and print results |
| `jammi explain "<SQL>"` | Show the execution plan for a query |
| `jammi serve` | Start the Flight SQL server + health probe |

## Global options

```bash
jammi --config <PATH> <command>    # Use a specific config file
```

Without `--config`, Jammi looks for configuration in this order:
1. `$JAMMI_CONFIG` environment variable
2. `./jammi.toml` in the current directory
3. `~/.config/jammi/config.toml`
4. Built-in defaults

## Next steps

- [Deploy as a Server](./deploy-server.md) — Flight SQL server, configuration, preloading models
- [Configuration](./configuration.md) — full config reference
