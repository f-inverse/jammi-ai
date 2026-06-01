# jammi-server

The OSS server binary for [Jammi AI](https://github.com/f-inverse/jammi-ai).

## What it is

`jammi-server` is the substrate-level service for Jammi: a single-process binary that exposes Jammi's data sources, mutable tables, trigger streams, and vector search over Arrow Flight SQL and typed gRPC. It is the runnable artifact behind `docker pull ghcr.io/f-inverse/jammi-ai-server`.

The OSS server is **single-tenant**. The deployer's network is the authentication boundary — run one server per tenant, or front the binary with a reverse proxy that enforces tenancy at the HTTP layer.

## What it exposes

| Surface | Port | Wire format |
|---|---|---|
| Arrow Flight SQL | `8081` | gRPC (HTTP/2) |
| `jammi.v1.session.SessionService` | `8081` | gRPC + gRPC-Web |
| `jammi.v1.trigger.TriggerService` | `8081` | gRPC + gRPC-Web |
| `/healthz` | `8080` | HTTP/1.1 JSON |
| `/readyz` | `8080` | HTTP/1.1 JSON |
| `/metrics` | `8080` | Prometheus text |

The gRPC chain and Flight SQL share one Tonic server so a client binding a tenant via `SessionService.SetTenant` (with a `jammi-session-id` header) sees the same scoping applied to its Flight SQL queries.

## What it is NOT

- Not multi-tenant. No tenant column on OSS-exclusive tables; no auth on the wire.
- Not clustered. Single-instance only; no leader election, no replication.
- Not authenticated. The OSS server speaks unauthenticated gRPC. Run it behind your own boundary.

## Quickstart (Docker)

```bash
docker run --rm \
  -p 8080:8080 -p 8081:8081 \
  -v jammi_data:/var/lib/jammi \
  ghcr.io/f-inverse/jammi-ai-server:latest

# Liveness
curl http://localhost:8080/healthz
# {"status":"ok","version":"0.8.0"}

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

## Quickstart (from source)

```bash
cargo run --release --bin jammi-server --features jetstream-broker -- \
  --config crates/jammi-server/examples/jammi.toml
```

## Production deploy

See [Deploy as a Server](https://f-inverse.github.io/jammi-ai/deploy-server.html) in the cookbook for the full guide. The short version:

1. Mount a persistent volume at `/var/lib/jammi` (or `chown 65532:65532` your bind mount — the distroless image runs as the nonroot user).
2. Provide a config file at `/etc/jammi/jammi.toml`. The sample at `examples/jammi.toml` is the starting point.
3. Point your load balancer's readiness check at `/readyz`. The probe pings the catalog backend; a 503 means traffic should be drained from this instance.
4. Scrape `/metrics` from Prometheus. The OSS server emits gRPC request counts, Flight SQL query counts, eval invocation counts, and a search-latency histogram.

## License

Apache-2.0
