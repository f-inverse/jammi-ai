# Audio search

Run audio-to-audio similarity search over a corpus with a CLAP-format audio
model, measure retrieval quality, and domain-tune the audio embeddings on
caller-supplied triplets.

**When to use this pattern.** You have a corpus of sounds (clips, stems,
loops, recordings) and want to find the ones most similar to a query clip — and
a number that tells you how good the retrieval is. This is the audio
counterpart of the image `eval_embeddings` recipe; audio is simply the third
embedding modality the engine supports alongside text and images.

## Model

The recipe runs LAION's CLAP (`laion/clap-htsat-fused`: an HTSAT-Swin audio
tower and a text tower projected into one 512-dim space), downloaded from the
Hugging Face Hub on first use (about 600 MB) and cached. Any Hugging Face CLAP
checkpoint works the same way — its `config.json` declares
`model_type = "clap_audio_model"` (or lists `ClapModel` /
`ClapAudioModelWithProjection` in `architectures`), its weights carry the
`audio_model.audio_encoder.*` and `audio_projection.*` tower keys, and its
`preprocessor_config.json` holds the feature-extractor geometry. The encoder is
read from that config, as the image recipe reads OpenCLIP's; change `MODEL` in
the program to try another.

It takes a few minutes on a CPU, most of it the three training steps, and well
under one on a GPU.

## API surface exercised

- `Session.generate_embeddings(*, source, model, columns, key, modality="audio")`
- `Session.encode_query(*, model, query, modality="audio")` → `list[float]`
- `Session.search(source, *, query, k, filter=None, select=None)` → `pyarrow.Table`
- `Session.eval_embeddings(*, source, golden_source, model=None, k=10)`
- `Session.fine_tune(*, source, base_model, columns, method, task="audio_embedding", target_modules=[...], ...)` → `TrainingJob`
- `Session.describe_model(model_id)` → `dict | None`

### Audio triplet schema (fine-tune input)

| column     | type   | notes                                   |
|------------|--------|-----------------------------------------|
| `anchor`   | binary | encoded audio clip                      |
| `positive` | binary | a clip the caller deems related         |
| `negative` | binary | a clip the caller deems unrelated       |

Same column shape as text triplets — `task="audio_embedding"` is what tells the
loader to read the three columns as encoded audio rather than text.

### Audio-tower LoRA sites

`target_modules` names sites on **this** architecture. An **empty** list means
"no tower sites" and selects the projection-head mode instead. The HTSAT-Swin
audio tower offers `query`, `key`, `value`, `attention_output`,
`intermediate_dense`, `output_dense`, `reduction`, `linear1` and `linear2`;
`all-linear` selects every one. A selector matches a site name exactly or as a
suffix of it. A **non-empty** list matching nothing fails the job with a message
that echoes what you submitted and lists the tower's real names.

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

## Run it

```bash
python cookbook/recipes/audio_search/example.py
```

It prints the top five, the base and tuned retrieval metrics, how far each
adaptation moved the query's vector, and the refusal's message.
