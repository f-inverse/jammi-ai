# Provenance — `finetune_run_golden/` fixtures

Unit-63 round-3 audit, docs-ci class fix: every merger regression against
the `finetune-run` tier so far traces to a HAND-ROLLED test fixture
(`test_ab_merge.py`'s own `_finetune_run_tier`/`_FINETUNE_RUN_DISPATCH_COUNTERS`)
that quietly diverged from what the real, compiled `jammi-bench finetune-run`
binary actually emits — `adamw_{fused,eager}_dispatches` fell out of both
`ab_merge.py::ALL_BASES` and this test suite's own hand-typed counters dict
even though `report.rs`/`finetune_step.rs` have emitted it on every real leg
for months, and `dispatch_pairs` raised `KeyError('adamw')` the moment a real
leg reached the merger. These two files close that class of drift at its
root: they are the REAL JSON `jammi-bench finetune-run` (built at
`1962891cede8b3beb73289f22de08dd4b47a99b3`, `cargo build --release -p
jammi-bench`, no `cuda` feature — CPU-hermetic, aarch64-apple-darwin) wrote
to stdout, copied byte-for-byte (reformatted with `python3 -m json.tool`
equivalent 2-space indent only — no field added, removed, or edited), never
a second hand-typed field list standing in for the compiled binary's own
`FinetuneRunTier` serde output.

`test_ab_merge.py`'s `_finetune_run_tier` loads `bert_fused.json`'s
`tiers.finetune_run` block as its STRUCTURAL base (every field the real
struct serializes is therefore present by construction — a future field the
producer adds shows up here automatically the next time this fixture is
regenerated, and `GoldenProducerAnchoredFieldSetTests` pins today's dispatch-
pair base set against it so a producer-side add REDs this suite instead of
raising `KeyError` on the next real leg), then overrides the identity/
provenance/premise/measurement fields with this suite's own literal,
predictable-for-testing values exactly as before this fix — the risk this
golden closes is a MISSING field name, never a specific numeric value.

## How to regenerate

Both runs use the exact CLI shape `crates/jammi-bench/tests/finetune_run_smoke.rs`'s
own `base_command` builds (a hand-written 4-train/2-heldout synthetic
triplet set, `--epochs 2 --eval-cadence 1 --batch 2`), built once via:

```
cargo build --release -p jammi-bench
```

### `bert_fused.json`

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

BERT architecture (`cookbook/fixtures/tiny_bert`, the SAME generic, committed
fixture `finetune_run_smoke.rs` itself uses) has no fused whole-attention-
block / LayerNorm / RoPE / softmax / GEGLU kernel at all (`report.rs`'s own
field docs) — this golden's `ln_*`/`rope_*`/`softmax_*`/`geglu_*`/
`attention_block_*` counters legitimately read `0` fused / `0` eager
forever; `adamw_fused_dispatches: 16` / `adamw_eager_dispatches: 0` and
`lora_linear_eager_dispatches: 8` are the two dispatch pairs a BERT LoRA
fine-tune step actually exercises on this fixture, both real, both from the
compiled binary, neither hand-typed.

### `modernbert_alloff.json`

```
JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused \
jammi-bench finetune-run --model-dir cookbook/fixtures/tiny_modernbert_ner --arm alloff \
  --train-jsonl <synthetic 4-triplet train.jsonl> \
  --heldout-ids <synthetic 2-pair heldout_ids.txt> \
  --heldout-jsonl <synthetic 2-triplet heldout.jsonl> \
  --seed 7 --epochs 2 --eval-cadence 1 --batch 2 --lr 0.001 \
  --schedule constant --warmup-steps 0 --weight-decay 0.0 --grad-accum 1 \
  --validation-fraction 0.0 --early-stopping-patience 10000 \
  --early-stopping-metric train_loss --max-grad-norm 0.0 --objective mnrl \
  --margin 0.3 --temperature 20.0 --lora-rank 2 --lora-alpha 4 \
  --lora-dropout 0.0 --target-modules Wqkv,Wo,Wi --backbone-dtype f32 \
  --max-seq-length 16 --work-dir <tmp>
```

ModernBERT architecture (`cookbook/fixtures/tiny_modernbert_ner`, a
committed generic fixture with a real `tokenizer.json`; `hidden_size=32`,
`num_attention_heads=2` ⇒ `head_dim=16`, NOT the fused whole-attention-block
kernel's fixed domain of 64 — `jammi_kernels::ops::ATTENTION_BLOCK_HEAD_DIM`)
mirrors `finetune_run_ab.sh`'s own documented `alloff` convention exactly
(`JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused` — main.rs's
own `FinetuneRunArgs::arm` doc: "the caller is responsible for setting
[this] itself before invoking this binary for the alloff arm"), run on a
CPU build (`flash_compiled: false`). The result is a REAL, producer-anchored
counter-example to two class-defect assumptions this test suite used to
hand-roll instead of check against a real leg:

1. `attention_block_flash_declined_dispatches: 4` /
   `attention_block_fused_dispatches: 0` /
   `attention_block_eager_dispatches: 4` — the flash cascade declines (both
   because it is named in `kernels_disabled_requested`/`kernels_disabled_fired`
   AND because `flash_compiled` is false on this CPU build) and falls
   through to the block arm, which itself declines by DOMAIN (`head_dim !=
   64`) to the eager composition — the real "training-mode attention path
   reached via the disabled-kernel fallback" shape Block 4's positive proof
   checks for on a modernbert-arch alloff leg.
2. `ln_fused_dispatches: 12`, `rope_fused_dispatches: 8`,
   `softmax_fused_dispatches: 4`, `geglu_fused_dispatches: 4`,
   `lora_linear_fused_dispatches: 16` all read FUSED (nonzero) despite
   `arm: "alloff"` — proof that `finetune_run_ab.sh`'s own `alloff` disables
   EXACTLY `attention_block_flash` and `adamw_step_fused`, never every fused
   kernel this tier carries. The pre-fix `finetune_run_dispatch_proof_violations`
   required EVERY dispatch pair to read `fused == 0` for an `alloff` leg —
   this golden is the real leg that would have failed that check on every
   single field above, which would have marked every real campaign `alloff`
   leg `INVALID` the day this producer went live. `finetune_run_dispatch_proof_violations`
   now checks only the bases a leg's own `kernels_disabled_requested` names
   (see that function's own doc), exactly the claim an `alloff` leg actually
   makes.

`adamw_eager_dispatches: 32` / `adamw_fused_dispatches: 0` is the real
counted fallback behind the disabled `adamw_step_fused` claim on this leg
(32 = 8 trainable LoRA tensors × 4 optimizer steps across 2 resume-cycled
epochs at `--batch 2` over 4 train triplets).
