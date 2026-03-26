# jammi-server

Arrow Flight SQL server and health probe for [Jammi AI](https://github.com/f-inverse/jammi-ai).

`jammi-server` exposes Jammi's data sources and embedding tables over the standard Arrow Flight SQL protocol. Any Arrow-compatible client (pyarrow, JDBC, DBeaver, Superset) can connect and query via SQL. A minimal HTTP health endpoint is included for container liveness probes.

## Usage

```bash
# Start the server
jammi serve

# Flight SQL on port 8081, health probe on port 8080
curl http://localhost:8080/health
# {"status": "ok"}
```

```python
from pyarrow.flight import FlightClient, FlightDescriptor

client = FlightClient("grpc://localhost:8081")
info = client.get_flight_info(
    FlightDescriptor.for_command(b"SELECT * FROM patents.public.patents LIMIT 10")
)
reader = client.do_get(info.endpoints[0].ticket)
table = reader.read_all()
```

## Documentation

See the [Jammi AI Cookbook](https://f-inverse.github.io/jammi-ai/) for the full guide, including [Deploy as a Server](https://f-inverse.github.io/jammi-ai/deploy-server.html).

## License

Apache-2.0
