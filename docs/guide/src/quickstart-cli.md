# Quickstart: CLI

The `jammi` CLI is a strict gRPC client: it talks to a running `jammi-server`
over the wire and never touches the catalog or storage in-process. Start a
server (see [Deploy as a Server](./deploy-server.md)), then point the CLI at it
with `--target`.

## Register a source and query it

```bash
# Register a remote source (the URL is resolved server-side)
jammi --target grpc://127.0.0.1:8081 \
  sources add patents --url /path/to/patents.parquet --format parquet

# List registered sources
jammi --target grpc://127.0.0.1:8081 sources list

# Run a SQL query
jammi --target grpc://127.0.0.1:8081 \
  query "SELECT id, title, year FROM patents.public.patents WHERE year > 2020 LIMIT 5"

# Show the execution plan
jammi --target grpc://127.0.0.1:8081 \
  explain "SELECT * FROM patents.public.patents WHERE year > 2020"
```

The default `--target` is `grpc://127.0.0.1:8081`, so a CLI talking to a local
server can omit the flag.

## Check the server

```bash
# Report version, compiled features, storage backends, and mounted services.
# A successful response also confirms reachability.
jammi status
```

## Available commands

| Command | Description |
|---------|-------------|
| `jammi status` | Report the server's capabilities and confirm reachability |
| `jammi sources list` | List registered data sources |
| `jammi sources add <NAME> --url <URL> --format <FMT>` | Register a source |
| `jammi models list` | List registered models |
| `jammi query "<SQL>"` | Run a SQL query and print results |
| `jammi explain "<SQL>"` | Show the execution plan for a query |
| `jammi channels …` | Manage evidence channels |
| `jammi mutable …` | Manage mutable companion tables |
| `jammi trigger …` | Manage trigger-stream topics |

## Global options

```bash
jammi --target <ENDPOINT> <command>   # Server endpoint (default grpc://127.0.0.1:8081)
jammi --tenant <UUID> <command>       # Bind a tenant scope for the session
```

`--target` accepts `grpc://host:port` (plaintext), `grpcs://host:port` (TLS),
`http(s)://host:port`, or a bare `host:port`. `--tenant` binds a tenant scope
before any verb runs, so every read and write is scoped to that tenant.

## Next steps

- [Deploy as a Server](./deploy-server.md) — `jammi-server`, configuration, preloading models
- [Configuration](./configuration.md) — full config reference
