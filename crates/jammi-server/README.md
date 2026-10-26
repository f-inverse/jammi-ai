# jammi-server

The server binary for [Jammi AI](https://github.com/f-inverse/jammi-ai).

## What it is

`jammi-server` serves the Jammi engine over the network: Arrow Flight SQL for
queries, typed gRPC for every other verb, and an HTTP side channel for health and
metrics. It is the runnable artifact behind `docker pull
ghcr.io/f-inverse/jammi-ai-server`, and the same binary serves every deployment
shape — one process, a multi-tenant server, or a disaggregated fleet over a
shared Postgres catalog — chosen by configuration (see [Reference
Topologies](https://f-inverse.github.io/jammi-ai/reference-topologies.html)).

## What it exposes

| Surface | Port | Wire format |
|---|---|---|
| Arrow Flight SQL | `8081` | gRPC (HTTP/2) |
| `CatalogService`, `EmbeddingService`, `InferenceService`, `PipelineService`, `AuditService`, `JobService` | `8081` | gRPC + gRPC-Web |
| `EvalService` (eval tier), `TriggerService` (event tier) | `8081` | gRPC + gRPC-Web |
| `PeerService`, `GangService` | `[server] peer_bind` | gRPC, replica-to-replica only |
| `/healthz`, `/readyz` | `8080` | HTTP/1.1 JSON |
| `/metrics` | `8080` | Prometheus text |

The gRPC services and Flight SQL share one listener, so a client that binds its
tenant with `CatalogService.SetTenant` (under a `jammi-session-id` header) sees
the same scope on its Flight SQL queries.

## Tenancy and authentication

The server authenticates no one. By default a client declares its session's
tenant cooperatively, so the deployer's network is the trust boundary; a
deployment that must prove who the caller is fronts the server with an
authenticating proxy, or composes its own `TenantResolver` through
`jammi_server::runtime::assemble_grpc_chain` (see [Bring your own
auth](https://f-inverse.github.io/jammi-ai/multi-tenant.html#bring-your-own-auth)).

## Quickstart (Docker)

```bash
docker run --rm \
  -p 127.0.0.1:8080:8080 -p 127.0.0.1:8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server:latest

# Liveness
curl http://localhost:8080/healthz
# {"status":"ok","version":"0.55.0"}

# Readiness
curl http://localhost:8080/readyz
# {"status":"ready"}

# Prometheus metrics
curl http://localhost:8080/metrics
# jammi_grpc_requests_total 0
# jammi_flight_queries_total 0
# ...

# Flight SQL roundtrip from Python.
python -c '
from pyarrow.flight import FlightClient, FlightDescriptor
client = FlightClient("grpc://localhost:8081")
info = client.get_flight_info(FlightDescriptor.for_command(b"SELECT 1 AS one"))
print(client.do_get(info.endpoints[0].ticket).read_all())
'
```

Both ports bind to `127.0.0.1`: the server performs no authentication of
its own (see [The identity seam](https://f-inverse.github.io/jammi-ai/deploy-server.html#the-identity-seam)),
so a loopback bind keeps the unauthenticated admin surface off the host's
public network until a terminator or reverse proxy is put in front of it.

## Quickstart (from source)

```bash
cargo run --release --bin jammi-server -- \
  --config crates/jammi-server/examples/jammi.toml
```

## Production deploy

See [Deploy as a Server](https://f-inverse.github.io/jammi-ai/deploy-server.html) in the guide for the full walkthrough. The short version:

1. Mount a persistent volume at `/var/lib/jammi` (or `chown 65532:65532` your bind mount — the distroless image runs as the nonroot user).
2. Provide a config file at `/etc/jammi/jammi.toml`. The sample at `examples/jammi.toml` is the starting point.
3. Point your load balancer's readiness check at `/readyz`. The probe pings the catalog backend; a 503 means traffic should be drained from this instance.
4. Scrape `/metrics` from Prometheus: gRPC request counts, Flight SQL query counts, eval invocation counts, a search-latency histogram, and the job-queue and worker gauges.

## License

Apache-2.0
