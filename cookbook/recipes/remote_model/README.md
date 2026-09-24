# Remote models

Embed and search with a model served at a remote endpoint: a hosted
embeddings API, or an inference server the engine does not run itself. The
deployment declares the endpoint once, and every verb that takes a model
takes `remote:<name>`.

Run it:

```bash
python cookbook/recipes/remote_model/example.py
```

The script serves its own stand-in endpoint (a hashed bag-of-words embedder
speaking the OpenAI-compatible embeddings protocol), so it runs with no
network and no key.

## Declare the model

```toml
[models.remote.hosted-encoder]
protocol = "openai_embeddings"
url = "https://api.example.com/v1/embeddings"
model = "text-embedding-small"
dimensions = 1536
revision = "2026-01"
headers = { Authorization = { file = "/run/secrets/embeddings-key" } }
# timeout_secs = 60, max_in_flight = 4, max_retries = 2
```

Then use `remote:hosted-encoder` wherever a model id goes:

```python
db.generate_embeddings(source="corpus", model="remote:hosted-encoder",
                       columns=["content"], key="id", modality="text")
vec = db.encode_query(model="remote:hosted-encoder", query="...")
db.search("corpus", query=vec, k=10)
```

## What to know

- **What a table records.** A result table's recorded model run is the
  declaration — URL, model name, width and revision. It never records the
  credentials. The endpoint exposes no digest of its weights, so
  `revision` is the pin that separates two models served under one name:
  change it when the provider changes the model.
- **Bounded concurrency.** At most `max_in_flight` requests to one model are in flight across
  every plan in the process. A 429 or 5xx is retried after the delay the
  endpoint names, up to `max_retries` times.
- **Refused, never guessed.** A response that leaves an input out, answers
  one twice, or returns a vector of another width fails the forward,
  naming the row. An empty or null row is never sent.
- **Text embedding only.** The OpenAI-compatible protocol carries text
  embedding; any other task on a remote model is refused.

See the guide's [Remote Models](../../../docs/guide/src/remote-models.md)
page for the full reference.
