# Quickstart: CLI

The `jammi` CLI is a strict gRPC client: it talks to a running `jammi-server`
over the wire and never touches the catalog or storage in-process. Start a
server (see [Deploy as a Server](./deploy-server.md)), then point the CLI at it
with `--target`.

## Register a source

```bash
# Register a remote source (the URL is resolved server-side)
jammi --target grpc://127.0.0.1:8081 \
  sources add patents --url /path/to/patents.parquet --format parquet

# List registered sources
jammi --target grpc://127.0.0.1:8081 sources list
```

The CLI is a control-plane tool — SQL itself runs over the server's Flight SQL
surface through the Rust or Python client, not through a CLI verb; see [Query
Your Data with SQL](./query-data.md).

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
| `jammi channels …` | Manage evidence channels |
| `jammi mutable …` | Manage mutable companion tables |
| `jammi trigger …` | Manage trigger-stream topics |
| `jammi train list` | List training jobs (lifecycle status; read-only) |
| `jammi train status <JOB_ID>` | Read one training job's lifecycle status |

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
