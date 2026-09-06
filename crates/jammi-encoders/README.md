# jammi-encoders

Candle-native embedding encoders across text, image, and audio, with
built-in PEFT support via [`jammi-lora`](../jammi-lora). Six concrete
encoders: [`Bert`], [`DistilBert`], [`ModernBert`] (BERT-family sentence
encoders), [`ClipText`] / [`OpenClipVisionTransformer`] (the OpenCLIP
text↔image tower pair), and [`HtsatAudio`] (the CLAP HTSAT-Swin audio
tower). All six accept LoRA adapter injection at construction time through
the same `MaybeLoraLinear` seam.

## Status

`0.x` — pre-stable. The BERT-family forward output (mean-pooled + L2-
normalised by default) is stable for the `0.x` line. The on-disk adapter
format is owned by `jammi-lora`.

## Quick start

```rust
use std::path::PathBuf;

use candle_core::{DType, Device};
use candle_nn::VarMap;
use jammi_encoders::{Bert, Pooling};
use jammi_lora::LoraBuildConfig;

let device = Device::Cpu;
let fixture = PathBuf::from("cookbook/fixtures/tiny_bert");
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

- [`Bert`] / [`DistilBert`] / [`ModernBert`] — BERT-family text encoders.
  Each has a `builder()` returning a fluent `*Builder<'static>` and the same
  forward surface (`forward`, `forward_hidden`, `max_seq_length`,
  `hidden_size`, `trainable_params`, `named_trainable_weights`,
  `set_training`, `load_weights`).
- [`ClipText`] / [`OpenClipVisionTransformer`] — the OpenCLIP text and vision
  towers, reached through [`ClipText::builder`] and
  [`OpenClipVisionTransformer::builder`]. Each produces a pooled
  `[batch, embed_dim]` shared-latent embedding for cross-modal text↔image
  search; their adapter key namespaces are disjoint (text at the checkpoint
  root, vision under `visual.`) so building both into one `VarMap` registers
  independent LoRA `Var`s.
- [`HtsatAudio`] — the CLAP HTSAT-Swin audio tower, reached through
  [`HtsatAudio::builder`]. Consumes a 4-channel fusion spectrogram and
  produces a pooled `[batch, projection_dim]` shared-latent embedding,
  compatible with a CLAP text tower for cross-modal text↔audio search.
- [`AnyEncoder`] — closed enum holding any of the six concrete encoders for
  callers that need to dispatch generically without trait objects. Key
  methods: [`AnyEncoder::forward_input`] (dispatches on an [`EncoderInput`],
  refusing a modality mismatch with a typed error), [`AnyEncoder::probe_input`]
  (materialises a minimal [`OwnedEncoderInput`] of the encoder's own
  modality), [`AnyEncoder::modality`], [`AnyEncoder::dtype`], plus the
  BERT-family-only `forward`/`forward_hidden` convenience methods and the
  shared training hooks (`trainable_params`, `named_trainable_weights`,
  `set_training`, `load_weights`).
- [`EncoderInput`] / [`OwnedEncoderInput`] — the one input vocabulary across
  all three modalities (`Text { input_ids, attention_mask }`,
  `Image { pixel_values }`, `Audio { input_features, is_longer }`).
  [`OwnedEncoderInput`] is the owning twin for a caller that must
  materialise and hand around a batch (e.g. what `probe_input` returns);
  borrow it back with `as_input`.
- [`Modality`] — `Text`, `Image`, `Audio`: which kind of input an encoder
  consumes and an [`EncoderInput`] carries.
- [`Pooling`] — `Mean`, `Cls`, `Max`, `WeightedMean`. Mean is the
  sentence-transformer default (BERT-family encoders only).
- [`pool_and_normalize`] — exposed for callers reusing the pooling helper on
  pre-computed hidden states.
- [`EncoderError`] — single error type covering tensor, LoRA, I/O,
  configuration, and modality-mismatch failures.

## Why this exists

The candle ecosystem has `candle-transformers` for raw transformer
architectures but no candle-native embedding stack spanning text, image, and
audio with PEFT support and a single closed-enum dispatch surface.
`jammi-encoders` fills that gap.

## When to use this vs. `candle-transformers`

Use `jammi-encoders` when you want LoRA-injectable text, image, or audio
encoders on candle `0.11` with pooled, normalised output ready for
similarity search or fine-tuning, dispatched through one [`AnyEncoder`]
regardless of modality.

Use `candle-transformers` when you need broader model coverage (Llama,
Whisper, Mistral, etc.) or raw hidden states from a wider set of
architectures.

## License

Apache-2.0 — same as the parent workspace.
