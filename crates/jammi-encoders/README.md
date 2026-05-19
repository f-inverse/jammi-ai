# jammi-encoders

Candle-native BERT-family sentence encoders with built-in PEFT support via
[`jammi-lora`](../jammi-lora). One crate, three encoders: [`Bert`],
[`DistilBert`], [`ModernBert`]. All three take tokens in, produce pooled
embeddings out, and accept LoRA adapter injection at construction time.

## Status

`0.x` — pre-stable. The forward output (mean-pooled + L2-normalised by
default) is stable for the `0.x` line. The on-disk adapter format is owned by
`jammi-lora`.

## Quick start

```rust
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarMap;
use jammi_encoders::{Bert, Pooling};
use jammi_lora::LoraBuildConfig;

let device = Device::Cpu;
let fixture = PathBuf::from("tests/fixtures/tiny_bert");
let config_str = std::fs::read_to_string(fixture.join("config.json"))?;
let bert_config: jammi_encoders::BertConfig = serde_json::from_str(&config_str)?;

let varmap = VarMap::new();
let bert = Bert::builder()
    .pooling(Pooling::Mean)
    .lora(LoraBuildConfig::frozen())
    .backbone_dtype(DType::F32)
    .build(
        &[&fixture.join("model.safetensors")],
        &bert_config,
        &device,
        &varmap,
    )?;

// `input_ids: [batch, seq]`, `mask: [batch, seq]` → `[batch, hidden_size]`.
let embedding = bert.forward(&input_ids, &mask)?;
```

## Public API

- [`Bert`] / [`DistilBert`] / [`ModernBert`] — concrete encoders. Each has a
  `builder()` returning a fluent `*Builder<'static>` and the same forward
  surface (`forward`, `forward_hidden`, `max_seq_length`, `hidden_size`,
  `trainable_params`, `named_trainable_weights`, `set_training`,
  `load_weights`).
- [`AnyEncoder`] — closed enum holding any of the three concrete encoders for
  callers that need to dispatch generically without trait objects.
- [`Pooling`] — `Mean`, `Cls`, `Max`, `WeightedMean`. Mean is the
  sentence-transformer default.
- [`pool_and_normalize`] — exposed for callers reusing the pooling helper on
  pre-computed hidden states.
- [`EncoderError`] — single error type covering tensor, LoRA, I/O, and
  configuration failures.

## Why this exists

The candle ecosystem has `candle-transformers` for raw transformer
architectures but no candle-0.9.x-native sentence-embedding stack with PEFT
support. `jammi-encoders` fills that gap.

## When to use this vs. `candle-transformers`

Use `jammi-encoders` when you want LoRA-injectable BERT-family sentence
encoders on candle `0.9.x` with pooled, normalised output ready for similarity
search or fine-tuning.

Use `candle-transformers` when you need broader model coverage (Llama,
Whisper, Mistral, etc.) or raw hidden states from a wider set of
architectures.

## License

Apache-2.0 — same as the parent workspace.
