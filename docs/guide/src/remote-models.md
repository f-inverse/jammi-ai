# Use a Remote Model

A model the engine does not run itself can still run through the engine: a
hosted embeddings API, or an inference server on another machine. The
deployment declares it once, under `[models.remote.<name>]`, and every verb
that takes a model takes `remote:<name>`: SQL, the Rust and Python sessions,
and the gRPC surface. The plan is the same one a local model runs; only the
device a forward runs on differs.

## Declare it

```toml
[models.remote.hosted-encoder]
protocol = "openai_embeddings"
url = "https://api.example.com/v1/embeddings"
model = "text-embedding-small"
dimensions = 1536
revision = "2026-01"
headers = { Authorization = { file = "/run/secrets/embeddings-key" } }
```

| Key | Meaning |
|---|---|
| `protocol` | `openai_embeddings`: `POST {"input": [...], "model": ...}` answered by `{"data": [{"index", "embedding"}]}` — served by hosted APIs and self-hosted inference servers alike |
| `url` | The full request URL, e.g. the `/v1/embeddings` route itself |
| `model` | The model name the endpoint is asked for |
| `dimensions` | The width of every returned vector; any other width is refused by row |
| `revision` | Your pin of what `model` serves at `url` (see below) |
| `headers` | Sent with every request, each inline or `{ file = "…" }`; never logged, never recorded |
| `timeout_secs` | One request's limit, connection included. Default 60 |
| `max_in_flight` | Requests to this model at once, across every plan in the process. Default 4 |
| `max_retries` | Re-sends of a request refused with 429 or 5xx, or timed out, after the endpoint's `Retry-After` or an exponential backoff. Default 2 |

The declaration is configuration, not a catalog row: it carries credentials,
and credentials live where `[models]`' other secrets live. Referencing a name
no declaration carries is refused, naming the missing
`[models.remote.<name>]`. A remote model is only ever named explicitly with
`remote:`, so a declared name never shadows a Hub repository.

## Use it

```python
db.generate_embeddings(source="corpus", model="remote:hosted-encoder",
                       columns=["content"], key="id", modality="text")
vec = db.encode_query(model="remote:hosted-encoder", query="...")
db.search("corpus", query=vec, k=10)
```

The runnable version, with a local stand-in endpoint, is the
[`remote_model`](https://github.com/f-inverse/jammi-ai/tree/main/cookbook/recipes/remote_model)
recipe.

## How a forward runs

- **A chunk is one request.** The endpoint tokenizes, so a row costs one
  and `[inference] batch_size` bounds a request in rows; `batch_tokens`
  does not apply. Keep `batch_size` within the endpoint's own limit on
  inputs per request.
- **An empty or null row is never sent.** It is marked failed at its own
  row, as it is for a local model, and the request carries only the rows
  with text.
- **The endpoint is the device.** A remote model holds no device memory
  here; its forwards are admitted by `max_in_flight`, the same way a local
  model's are admitted by its device.
- **A response is placed by `index`.** A response that leaves an input out,
  answers one twice, names an index outside the request, or returns a
  vector of another width than `dimensions` fails the forward, naming the
  input or row. It is never paired with rows by position.
- **Refusals are not retried.** A non-2xx other than 429 or 5xx fails the
  forward with the status and the endpoint's own message (never a header).

## What a table records

A result table records the model run that produced it. For a remote model
that is the declaration without its headers: `protocol`, `url`, `model`,
`dimensions` and `revision`. Two runs share a definition only when all five
agree, so a table reused from the cache was produced by the same declared
model.

No endpoint exposes a digest of its weights, so a provider that changes the
model behind an unchanged name is not detected: `revision` is the pin that
tells the two apart. Change it when the model behind the name changes. A
remote endpoint's vectors are also not guaranteed bitwise identical from one
call to the next, so a recomputed table can differ from the original in the
last bits.

## What it does not do

The `openai_embeddings` protocol carries text embedding only. Any other
task on a remote model (classification, NER, regression, image or audio
embedding) is refused when it is described, before any request is sent. A
remote model has no weights here, so it cannot be fine-tuned.
