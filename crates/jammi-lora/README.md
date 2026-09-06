# jammi-lora

Static-dispatch PEFT-style LoRA primitives for [candle][candle]. Closed-enum
dispatch, immutable construction, native to candle `0.11`.

`jammi-lora` gives you everything you need to add LoRA adapters to a candle
model — `LoraLinear`, a `MaybeLoraLinear` wrapper, build helpers, adapter
metadata, and safetensors persistence — without proc-macros and without trait
objects in the forward path.

## Status

`0.x` — pre-stable. The on-disk adapter format (`adapter_config.json` +
`adapter.safetensors`) is stable for the `0.x` line.

## Quick start

```rust
use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use jammi_lora::{LoraBuildConfig, LoraInitMode, LoraLinear, MaybeLoraLinear};

let device = Device::Cpu;
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

// A frozen base linear from your model.
let base_weight = Tensor::zeros((16, 8), DType::F32, &device).unwrap();
let base = Linear::new(base_weight, None);

// Wrap it with a LoRA adapter (rank 4, alpha 8, ZerosB init).
let lora = LoraLinear::new_simple(base, 4, 8.0, &vb.pp("layer.0.query")).unwrap();
let layer = MaybeLoraLinear::Lora(lora);

let x = Tensor::randn(0f32, 1.0, (2, 5, 8), &device).unwrap();
let _y = layer.forward(&x).unwrap();
```

## Public API

- [`LoraLinear`] — single LoRA-augmented linear layer.
- [`MaybeLoraLinear`] — `enum { Frozen(FrozenBase), Lora(LoraLinear) }`.
  `FrozenBase` names what the frozen base weight is stored as: a plain dense
  `candle_nn::Linear` or a GGUF-quantized weight. Match on
  `MaybeLoraLinear` in your model so dispatch stays compile-time-inlined.
- [`LoraBuildConfig`] — borrowed-reference build configuration.
- [`should_apply_lora`] / [`effective_rank`] — module-name matching helpers
  that mirror HuggingFace PEFT's `target_modules` / `rank_pattern` semantics.
  `should_apply_lora` takes a `layer_idx: Option<usize>`: a `None` (an
  UNINDEXED site, e.g. a tower's head-side projection outside the numbered
  block stack) can never satisfy an active `layers_to_transform` filter and
  is excluded whenever one is set — matching PEFT's own
  `check_target_module_exists` rule exactly.
- [`AdapterConfig`] / [`Tower`] / [`ComputePrecision`] — persisted adapter
  metadata (`ComputePrecision` is the backbone dtype's candle-free
  vocabulary, re-exported from `jammi-numerics`).
  `AdapterConfig` carries an optional `tower: Option<Tower>` field naming
  which tower of a multi-tower checkpoint (`Tower::Text` / `Vision` /
  `Audio`) an adapter installs on; it is `None` for a single-tower family
  (BERT, DistilBERT, ModernBERT) and omitted from the serialised JSON
  entirely when `None`, so a single-tower `adapter_config.json` is
  byte-unchanged.
- [`save_adapter`] / [`load_adapter`] — safetensors-based directory
  persistence.
- [`LoraInitMode`] — A/B init strategy (`ZerosB` default, `Gaussian`).
- [`LoraError`] — single error type for the whole crate.

## When to use this crate

Use `jammi-lora` when you need LoRA adapters on candle `0.11` with:

- compile-time dispatch (no `Box<dyn …>` in the forward path),
- a small, focused surface (one enum, one struct, a builder helper),
- HuggingFace PEFT-compatible `target_modules` / `rank_pattern` semantics,
- a stable on-disk adapter format.

If you want broader model coverage, multi-adapter switching, or proc-macro
sugar, consider [candle-lora][candle-lora].

[candle]: https://github.com/huggingface/candle
[candle-lora]: https://github.com/EricLBuehler/candle-lora
