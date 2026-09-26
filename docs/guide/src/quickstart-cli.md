# Quickstart: CLI

> **Measured companion:** the [One program, four surfaces](https://f-inverse.github.io/jammi-ai/cookbook/chapters/28-surfaces/surfaces.html) chapter runs this CLI's `sources add` / `embed` / `search` beside Python, Rust and TypeScript and checks they return the same rows.

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

## Embed and search

```bash
# Embed a source's `content` column, one vector per `id`
jammi embed patents --model sentence-transformers/all-MiniLM-L6-v2 \
  --columns content --key id

# The three rows nearest row 7's vector, as JSON lines in rank order
jammi search patents --row-key 7 -k 3 --select id,title,similarity
```

`embed` prints the result table it wrote (name, rows, width, and whether a
ready table was `--reuse`d). `search` queries by example — the vector stored
for `--row-key` — and prints one JSON object per row, so a shell pipeline or a
script reads the rows exactly; `--filter`, `--exact` and `--embedding-table`
narrow it as in the other clients. SQL itself runs over the server's Flight SQL
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
| `jammi embed <SOURCE> --model <ID> --columns <COLS> --key <COL>` | Embed a source's columns |
| `jammi search <SOURCE> --row-key <KEY> [-k N]` | The rows nearest a row's vector, as JSON lines |
| `jammi channels …` | Manage evidence channels |
| `jammi mutable …` | Manage mutable companion tables |
| `jammi trigger …` | Manage trigger-stream topics |
| `jammi jobs list` | List jobs (lifecycle status; read-only) |
| `jammi jobs status <JOB_ID>` | Read one job's lifecycle status |
| `jammi jobs cancel <JOB_ID>` | Request cancellation of a job |
| `jammi jobs prune` | Delete terminal job rows past `[jobs] retention_days` |
| `jammi workers list` | List engine processes running the claim loop |

## Global options

```bash
jammi --target <ENDPOINT> <command>   # Server endpoint (default grpc://127.0.0.1:8081)
jammi --tenant <UUID> <command>       # Bind a tenant scope for the session
```

`--target` accepts `grpc://host:port`, `http://host:port`, or a bare
`host:port` — all plaintext h2. TLS termination is the consumer's runtime,
not the CLI's: put a TLS-terminating proxy in front and point `--target` at
it in plaintext. `--tenant` binds a tenant scope before any verb runs, so
every read and write is scoped to that tenant.

## Next steps

- [Deploy as a Server](./deploy-server.md) — `jammi-server`, configuration, preloading models
- [Configuration](./configuration.md) — full config reference
