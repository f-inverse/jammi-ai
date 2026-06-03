# Audio search

Run audio-to-audio similarity search over a corpus with a CLAP-format audio
model, measure retrieval quality, and domain-tune the audio embeddings on
caller-supplied triplets.

**When to use this pattern.** You have a corpus of sounds (clips, stems,
loops, recordings) and want to find the ones most similar to a query clip — and
a number that tells you how good the retrieval is. This is the audio
counterpart of the image `eval_embeddings` recipe; audio is simply the third
embedding modality the engine supports alongside text and images.

## Flow

1. **Load** a small audio corpus (inline audio bytes in a Parquet source)
2. **Generate** L2-normalized audio embeddings over the audio column
3. **Search** the index with an encoded audio query (cosine ANN)
4. **Eval** retrieval quality (Recall@K / MRR) against a held-out golden set
5. **Fine-tune** a projection head on audio triplets and re-eval (tuned ≠ base)

## Model

Any **HuggingFace CLAP** audio model works — its `config.json` declares
`model_type = "clap_audio_model"` (or lists `ClapModel` /
`ClapAudioModelWithProjection` in `architectures`), its checkpoint exposes the
`audio_model.audio_encoder.*` + `audio_projection.*` HTSAT-Swin tower keys, and
a `preprocessor_config.json` carries the feature-extractor geometry. The encoder
is auto-detected from that config, exactly as the image recipe auto-detects
OpenCLIP:

```bash
JAMMI_AUDIO_MODEL=<hf-repo-id-or-local-path> \
    python cookbook/recipes/audio_search/example.py
```

By **default** (no env var) the recipe runs against the hermetic
`cookbook/fixtures/htsat_clap_tiny` fixture so it runs offline in CI in under a
few seconds. That fixture has random weights, so its retrieval numbers are
meaningless — it exercises the full pipeline, not model quality. Point
`JAMMI_AUDIO_MODEL` at a real CLAP checkpoint for real numbers.

## What `example.py` does

1. Connects to a temporary artifact dir
2. Reads the 20 committed mono WAV clips under
   `cookbook/fixtures/tiny_audio_corpus/` into a Parquet `corpus` source
   (`clip_id`, `audio` bytes)
3. `db.generate_audio_embeddings(source="corpus", model=MODEL, audio_column="audio", key="clip_id")`
4. `db.encode_audio_query(MODEL, wav_bytes)` → `db.search("corpus", query=vec, k=5).run()`
5. Builds the audio-query golden source from `tiny_audio_golden.json` and calls
   `db.eval_embeddings(source="corpus", golden_source="golden.public.golden", k=5)`
6. Prints the base aggregate Recall@K / precision@K / MRR / nDCG and the
   per-query records. It **reports** the metrics; it does **not** assert a
   quality bar.
7. Builds synthetic `(anchor, positive, negative)` audio triplets from the
   corpus (positive = same timbre family, negative = a different family) and
   calls
   `db.fine_tune(source="triplets", base_model=MODEL, columns=["anchor","positive","negative"], method="lora", task="audio_embedding", ...)`.
   Empty `target_modules` ⇒ a trainable **projection head on the frozen CLAP
   audio tower** (the cheap, low-risk lightweight mode). It then re-embeds the
   corpus with the tuned model, re-evals, and prints base-vs-tuned metrics for
   narrative. For correctness it re-encodes the **same** query clip through the
   tuned model and asserts the **embedding vector** changed (max elementwise
   `|Δ| > 1e-4` versus the base encoding) — the real invariant fine-tuning
   guarantees, and a deterministic check. (Asserting on the coarse top-k metrics
   instead is flaky: on this tiny eval set the rankings rarely flip even when the
   vectors move.) It proves the adapter alters audio retrieval — not that it
   improves it; the random-weight fixture's direction is not meaningful, real
   lift comes from a real checkpoint.

The pairing semantics (what a "positive" *means*) are the caller's training
data, not the trainer's: the trainer only minimizes the contrastive triplet
loss over whatever clips you pair.

## Stepwise scripts

`example.py` runs every phase in one process (this is the version wired into
`tests/cookbook_smoke.py`). The numbered scripts decompose the search-and-eval
flow and share a persistent workdir, so run them in order:

```bash
python cookbook/recipes/audio_search/01-load-corpus.py
python cookbook/recipes/audio_search/02-generate-embeddings.py
python cookbook/recipes/audio_search/03-search.py
python cookbook/recipes/audio_search/04-eval.py
```

## API surface exercised

- `Database.generate_audio_embeddings(*, source, model, audio_column, key)`
- `Database.encode_audio_query(model_id, audio_bytes)` → `list[float]`
- `Database.search(source, *, query, k)` → `SearchBuilder` → `.run()`
- `Database.eval_embeddings(*, source, golden_source, model=None, k=10)`
- `Database.fine_tune(*, source, base_model, columns, method, task="audio_embedding", ...)` → `FineTuneJob`

### Audio triplet schema (fine-tune input)

| column     | type   | notes                                   |
|------------|--------|-----------------------------------------|
| `anchor`   | binary | encoded audio clip                      |
| `positive` | binary | a clip the caller deems related         |
| `negative` | binary | a clip the caller deems unrelated       |

Same column shape as text triplets — `task="audio_embedding"` is what tells the
loader to read the three columns as encoded audio rather than text.

## Input schema

| column    | type   | notes                                          |
|-----------|--------|------------------------------------------------|
| `clip_id` | utf8   | per-row key                                    |
| `audio`   | binary | raw WAV/FLAC/MP3/Ogg bytes (decoded by the encoder) |

Preprocessing (decode → resample to the model's sample rate → CLAP fusion
log-mel spectrogram → HTSAT-Swin tower → L2-normalized output) is handled inside
the encoder per the model's `preprocessor_config.json` feature-extractor
geometry. The audio column may also hold file-path strings instead of inline
bytes.

## Golden source shape (audio mode)

`eval_embeddings` switches to audio-query mode when the golden source carries a
`query_audio` (binary) column instead of `query_text` / `query_image`:

| column        | type   | example                          |
|---------------|--------|----------------------------------|
| `query_id`    | utf8   | `q_sine`                         |
| `query_audio` | binary | raw WAV bytes of the query clip  |
| `relevant_id` | utf8   | `clip_sine_0` (matches `clip_id`) |

## Fixtures

- `cookbook/fixtures/tiny_audio_corpus/` — 20 synthetic mono WAV clips in 5
  timbre families (sine / harmonic / square / saw / noise), 4 per family, plus
  a held-out query clip per family under `queries/`. Synthesised
  programmatically by `cookbook/fixtures/generate.py` — **no recorded audio**
  (licensing), no tenant data.
- `cookbook/fixtures/tiny_audio_golden.json` — per-query → expected corpus IDs
  (same timbre family).
- `cookbook/fixtures/htsat_clap_tiny/` — tiny offline HTSAT-Swin CLAP fixture
  used as the default CI model, generated by
  `tests/fixtures/generate_htsat_clap.py`.

## Run it

```bash
python cookbook/recipes/audio_search/example.py
```

Exits 0 on success, prints the top-K and the metrics dict + `audio_search: OK`.
