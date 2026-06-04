# jammi-client

Pure-Python gRPC client for a remote [Jammi](https://github.com/f-inverse/jammi-ai)
engine — the **deploy** half of the develop→deploy journey.

`jammi-ai` runs an embedded engine in-process and bundles this client for its
remote targets. `jammi-client` is the lean variant: a universal
`py3-none-any` wheel that links no candle/ML stack — just `grpcio` + `protobuf`
+ `pyarrow` — for production where the engine runs behind a server.

## One front door

```python
import jammi_client

db = jammi_client.connect("https://engine.example.com")
db.add_source("patents", url="s3://corpus/patents.parquet", format="parquet")
db.generate_embeddings(source="patents", model="local:tiny_bert",
                       columns=["abstract"], key="id", modality="text")
q = db.encode_query(model="local:tiny_bert", query="quantum computing")
hits = db.search("patents", query=q, k=5)   # -> pyarrow.Table
```

`connect(target)` is the Python mirror of the Rust `Jammi::open(Target)`:
transport is config, not a code path.

## Authenticated channels

A bearer-protected endpoint is reached by attaching credentials to the channel
— the bearer rides the channel (attached once at connect), not threaded through
every call:

```python
from jammi_client import connect, BearerCredentials

db = connect("https://engine.example.com", credentials=BearerCredentials(token))
```

The same works on a plaintext `grpc://` target for local development. The
channel-level bearer covers the typed gRPC verbs; `db.sql()` (the Flight SQL
lane) does not yet carry it — tracked at
[issue #96](https://github.com/f-inverse/jammi-ai/issues/96).

| target | transport |
|---|---|
| `https://host` / `grpcs://host:8081` | secure remote |
| `http://host` / `grpc://host:8081` | plaintext remote |
| `file:///data` | local engine — **not in this build**; raises `NoEmbeddedEngineError` pointing at `pip install jammi-ai` |

Scaling local→remote is an env flip (`connect(os.environ["JAMMI_TARGET"])`)
with no code change. Productionising from the embed wheel to this lean client is
a one-line import swap (`import jammi_ai` → `import jammi_client`), `connect`
unchanged.

## Generated from the canonical proto

The wire stubs under `jammi_client/_generated/` are generated from the same
`jammi.v1` proto the engine, npm client, and embed wheel speak — one source, no
parallel schema. Regenerate after a proto change:

```sh
pip install -e '.[dev]'
make generate
```
