# Provenance — `finetune_run_golden/` fixtures

Every file here is ONE real report the compiled `jammi-bench finetune-run`
binary wrote, copied byte-for-byte (2-space `json.dumps(..., indent=2)`
shape) — no field added, removed, edited, or cross-sourced from a second
artifact. A hand-rolled fixture drifts from what the binary emits (a
counter pair the producer serializes but the fixture omits reaches the
merger as a `KeyError` on the first real leg); a producer-emitted golden
cannot.

The ladder's premise tests (`crates/jammi-bench/src/ladder/premise.rs`)
read the two ModernBERT goldens as real legs of the `resident` and
`resident-reference` rungs: each must clear its own arm's kernel-arm premise
and fail the other's, so a counter the producer emits but the premise
forgets, or the reverse, is caught against what the binary actually wrote.

## `bert_fused.json` — CPU-hermetic

Built with `cargo build --release -p jammi-bench` (no `cuda` feature) at
`ba80552a1345b1bb4459a33422377c10561c9a24` (`provenance.build_sha`), host
triple `aarch64-apple-darwin`, using the CLI shape
`crates/jammi-bench/tests/finetune_run_smoke.rs`'s own `base_command` builds
(a synthetic 4-train/2-heldout triplet set):

```
jammi-bench finetune-run --model-dir cookbook/fixtures/tiny_bert --arm fused \
  --train-jsonl <synthetic 4-triplet train.jsonl> \
  --heldout-ids <synthetic 2-pair heldout_ids.txt> \
  --heldout-jsonl <synthetic 2-triplet heldout.jsonl> \
  --seed 7 --epochs 2 --eval-cadence 1 --batch 2 --lr 0.001 \
  --schedule constant --warmup-steps 0 --weight-decay 0.0 --grad-accum 1 \
  --validation-fraction 0.0 --early-stopping-patience 10000 \
  --early-stopping-metric train_loss --max-grad-norm 0.0 --objective mnrl \
  --margin 0.3 --temperature 20.0 --lora-rank 2 --lora-alpha 4 \
  --lora-dropout 0.0 --target-modules query,value --backbone-dtype f32 \
  --max-seq-length 16 --work-dir <tmp>
```

Two back-to-back runs at the same CLI/seed agree on every field except
`train_run_wall_s` (wall-clock; not an identity field the ladder
compares).

Nonzero dispatch counters: `ln_fused: 12`, `softmax_fused: 4`,
`gelu_fused: 4` (`tiny_bert`'s dense-GELU FFN dispatches through
`gelu_erf_fused`), `lora_linear_fused: 8`, `adamw_fused: 16`,
`attention_block_eager: 4` and `attention_block_flash_declined: 4` (every
training-mode attention forward on this `head_dim == 16` fixture reaches the
attention cascade and is declined by the `head_dim == 64` domain predicate —
see `finetune_run.rs`'s
`fused_dispatch_proof_gate_passes_bert_counted_eager_head16_shape`). BERT has
no fused RoPE/GEGLU kernel, so those pairs read `0 / 0`.

## `modernbert_fused.json` / `modernbert_alloff.json` — A100

The `fused_r1`/`alloff_r1` legs of one `finetune_run_ab.sh` run
(`FINETUNE_RUN_AB_SEEDS=1`, `MODEL_DIR` = `answerdotai/ModernBERT-large`,
`head_dim == 64`), built `--features cuda,jammi-encoders/flash-attn`
(`flash_compiled: true` on both legs) at
`869c65f92aea21c6aa6a3ef12cc77f1132e3be80` (each leg's `provenance.build_sha`),
on an NVIDIA A100 80GB PCIe (driver 595.91.07, `x86_64-unknown-linux-gnu`).
For seed 1, r1 and r2 are bit-identical on both arms
(`determinism_floor.max_delta: 0.0`), so r1 is a representative leg. The
run's own report reads `INVALID` only because the pre-registered decision
rule renders a verdict at exactly 12 premise-clean seeds and this was a
deliberate one-seed producer probe; every leg reads `OK` and the seed's
premise is clean.

| golden file | sha256 (== the source leg's sha256) | source leg |
|---|---|---|
| `modernbert_fused.json` | `0f3a0cdefb0494136fbb2ae7a73660c693680c0e1b1b62dafa9f505cfdc1d412` | `raw/seed1__fused__r1.json` |
| `modernbert_alloff.json` | `66131ef2705f8681f0acf036bc997bc3170ff7f15f42b1b0467d8c7515b57e98` | `raw/seed1__alloff__r1.json` |

Dispatch counters (fused / declined-or-eager):

| base | `modernbert_fused.json` | `modernbert_alloff.json` |
|---|---|---|
| `ln` | 6669 / 0 | 6669 / 0 |
| `rope` | 0 / 0 (absorbed) | 0 / 0 (absorbed) |
| `softmax` | 0 / 0 (absorbed) | 0 / 0 (absorbed) |
| `geglu` | 3276 / 0 | 3276 / 0 |
| `gelu` | 0 / 0 (ModernBERT's GeGLU MLP never routes through `gelu_erf_fused`) | 0 / 0 |
| `lora_linear` | 13104 / 0 | 13104 / 0 |
| `attention_block` | 0 / 0 (absorbed by the flash cascade) | 3276 / 0 (the disabled flash cascade's fallthrough — this arm's positive training-path proof) |
| `attention_block_flash` | 3276 / 0 | 0 / 3276 |
| `adamw` | 26208 / 0 | 0 / 26208 |

The alloff leg's `kernels_disabled_requested == kernels_disabled_fired ==
["adamw_step_fused", "attention_block_flash"]`: `alloff` disables exactly
those two kernels, never every fused kernel the tier carries, which is why
`ln`/`geglu`/`lora_linear` still read fused on it. Both legs clear
the ladder's kernel-arm premise for their own arm.

## Regenerating

A producer change that adds a counter pair to `FinetuneRunTier` means all
three files are regenerated from real runs as above — never hand-edited.
The ModernBERT pair needs a CUDA A100 and the real ModernBERT-large
checkpoint; no CPU-hermetic run reproduces its device, checkpoint, or
dispatch-count fields.
