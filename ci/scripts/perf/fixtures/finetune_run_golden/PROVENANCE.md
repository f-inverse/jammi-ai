# Provenance — `finetune_run_golden/` fixtures

Unit-63 round-3 audit, docs-ci class fix: every merger regression against
the `finetune-run` tier so far traces to a HAND-ROLLED test fixture
(`test_ab_merge.py`'s own `_finetune_run_tier`/`_FINETUNE_RUN_DISPATCH_COUNTERS`)
that quietly diverged from what the real, compiled `jammi-bench finetune-run`
binary actually emits — `adamw_{fused,eager}_dispatches` fell out of both
`ab_merge.py::ALL_BASES` and this test suite's own hand-typed counters dict
even though `report.rs`/`finetune_step.rs` have emitted it on every real leg
for months, and `dispatch_pairs` raised `KeyError('adamw')` the moment a real
leg reached the merger. These files close that class of drift at its root.
`bert_fused.json` is the REAL JSON `jammi-bench finetune-run` (built at
`1962891cede8b3beb73289f22de08dd4b47a99b3`, `cargo build --release -p
jammi-bench`, no `cuda` feature — CPU-hermetic, aarch64-apple-darwin) wrote
to stdout, copied byte-for-byte (reformatted with `python3 -m json.tool`
equivalent 2-space indent only — no field added, removed, or edited), never
a second hand-typed field list standing in for the compiled binary's own
`FinetuneRunTier` serde output. `modernbert_fused.json`/`modernbert_alloff.json`
are a DOCUMENTED COMPOSITE of a real CPU-hermetic `finetune-run` execution's
own identity/provenance/measurement fields with real dispatch-counter
fields copied from the committed CUDA artifacts named in "Coordinator
correction" below (no CUDA device or ModernBERT-large checkpoint exists in
this environment to run a genuine `flash_compiled: true` leg against) —
never hand-invented counts.

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

## How to regenerate the CPU-hermetic runs

`bert_fused.json` and the ORIGINAL (superseded) `modernbert_alloff.json`
CPU run below both use the exact CLI shape
`crates/jammi-bench/tests/finetune_run_smoke.rs`'s own `base_command`
builds (a hand-written 4-train/2-heldout synthetic triplet set,
`--epochs 2 --eval-cadence 1 --batch 2`), built once via:

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

### `modernbert_alloff.json` (superseded shape — kept for the historical
record below; see "Coordinator correction" for the file's CURRENT content)

The FIRST cut of this golden was run exactly like `bert_fused.json` above,
substituting `--model-dir cookbook/fixtures/tiny_modernbert_ner --arm
alloff` and `JAMMI_KERNELS_DISABLE=attention_block_flash,adamw_step_fused`.
ModernBERT architecture (`cookbook/fixtures/tiny_modernbert_ner`, a
committed generic fixture with a real `tokenizer.json`; `hidden_size=32`,
`num_attention_heads=2` ⇒ `head_dim=16`, NOT the fused whole-attention-block
kernel's fixed domain of 64 — `jammi_kernels::ops::ATTENTION_BLOCK_HEAD_DIM`)
on a CPU build (`flash_compiled: false`) produced a REAL, producer-anchored
counter-example to one class-defect assumption this test suite used to
hand-roll instead of check against a real leg: `ln_fused_dispatches: 12`,
`rope_fused_dispatches: 8`, `softmax_fused_dispatches: 4`,
`geglu_fused_dispatches: 4`, `lora_linear_fused_dispatches: 16` all read
FUSED (nonzero) despite `arm: "alloff"` — proof that `finetune_run_ab.sh`'s
own `alloff` disables EXACTLY `attention_block_flash` and `adamw_step_fused`,
never every fused kernel this tier carries. The pre-fix
`finetune_run_dispatch_proof_violations` required EVERY dispatch pair to
read `fused == 0` for an `alloff` leg — this golden is the real leg that
would have failed that check on every field above, which would have marked
every real campaign `alloff` leg `INVALID` the day this producer went live.
That fix (checking only the bases a leg's own `kernels_disabled_requested`
names) stays landed and is unaffected by the correction below.

BUT this first cut's `attention_block_fused_dispatches: 0` /
`attention_block_eager_dispatches: 4` (the block arm declining by DOMAIN,
`head_dim=16 != 64`, to the eager composition) is the WRONG shape for the
positive training-path proof this arm needs, and the tiny fixture's
`flash_compiled: false` is the WRONG shape for the `fused` arm's own
premise entirely — see "Coordinator correction" below.

## Coordinator correction (unit-63 round-3 audit)

CONTRACT 63 Frame pre-registers the arms as "fused cascade vs
ALLOFF=attention_block_flash,adamw_step_fused" — the A/B's own differential
IS whether the FlashAttention-2 cascade (and the fused multi-tensor AdamW
kernel) fired. This means:

  * The `fused` arm's own premise REQUIRES `flash_compiled: true` — a build
    that cannot compile flash in (this repo's CPU/no-`flash-attn` toolchain)
    can never exercise the pre-registered differential at all, making the
    experiment null on such a build. `finetune_run_ab.sh` now builds
    `--features cuda,jammi-encoders/flash-attn` (mirroring
    `fa2_ab.sh`'s/`stacked_sweep.sh`'s own flash-A/B build feature list)
    for exactly this reason — a CPU-hermetic golden can never carry a real
    `flash_compiled: true` leg's dispatch counters (this repo's own build
    environment has no CUDA device to run one against).
  * The `alloff` arm's positive training-path proof is `attention_block`'s
    own FUSED count (not eager): `attention_block` is NOT itself named in
    the alloff disable list, only `attention_block_flash` is, so on a REAL
    (`head_dim == 64`) checkpoint it remains an ACTIVE, undisabled fused
    kernel that the disabled flash cascade falls through to — the tiny
    `head_dim=16` fixture's own DOMAIN decline (to eager) was an accident
    of that fixture's shape, not the real production shape.

Neither shape is producible by a CPU-hermetic run in this environment (no
CUDA device, no ModernBERT-large checkpoint). Per instruction, the counter
FIELDS for both `modernbert_fused.json` (new) and `modernbert_alloff.json`
(this file, now REPLACED in place) are copied verbatim from the two
committed, REAL `jammi-bench finetune-step` artifacts below — the same
attention-block/flash/AdamW call sites `finetune_run.rs`'s own
`FinetuneRunTier` mirrors field-for-field from `FinetuneStepTier` (see
`report.rs`'s own doc) — never hand-invented. Every OTHER field (identity,
provenance, premise, measurement) in both files is UNCHANGED from the
original CPU-hermetic `modernbert_alloff.json` run documented above (a real
`finetune-run` execution's own real values for those fields; only the
dispatch-counter/`flash_compiled`/`kernels_disabled_*`/`arm`/`attention_arm`
fields are overridden with the cited real counts below) — this is a
DOCUMENTED COMPOSITE, not a second hand-rolled fixture.

### `modernbert_fused.json` — real counts from
`crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/s128_flash_on_1.json`
(ModernBERT-large, `head_dim=64`, `flash_compiled: true`, no disable
request — git_sha `b98f7e1de35b5cfcac00325089fd5eeaf7c6259a`, box `a100c`)
and
`crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_fused.r2.json.raw`
(the adamw fused leg, `kernels_disabled_requested: []`):

| field | value | source |
|---|---|---|
| `ln_fused_dispatches` / `ln_eager_dispatches` | 1710 / 0 | s128_flash_on_1.json |
| `rope_fused_dispatches` / `rope_eager_dispatches` | 0 / 0 | s128_flash_on_1.json (absorbed by the flash cascade) |
| `softmax_fused_dispatches` / `softmax_eager_dispatches` | 0 / 0 | s128_flash_on_1.json (absorbed) |
| `geglu_fused_dispatches` / `geglu_eager_dispatches` | 840 / 0 | s128_flash_on_1.json |
| `lora_epilogue_fused_dispatches` / `lora_epilogue_eager_dispatches` | 0 / 0 | s128_flash_on_1.json (permanently superseded by lora_linear) |
| `lora_linear_fused_dispatches` / `lora_linear_eager_dispatches` | 3360 / 0 | s128_flash_on_1.json |
| `attention_block_fused_dispatches` / `attention_block_eager_dispatches` | 0 / 0 | s128_flash_on_1.json (absorbed BY the flash cascade — its own `admit` call is never reached, see `report.rs`'s `attention_block_flash_fused_dispatches` field doc) |
| `attention_block_flash_fused_dispatches` / `..._declined_dispatches` | 840 / 0 | s128_flash_on_1.json — the pre-registered admitted branch |
| `adamw_fused_dispatches` / `adamw_eager_dispatches` | 6720 / 0 | b8_s512_fused.r2.json.raw |
| `flash_compiled` | `true` | s128_flash_on_1.json |
| `kernels_disabled_requested` / `kernels_disabled_fired` | `[]` / `[]` | s128_flash_on_1.json |

### `modernbert_alloff.json` (corrected, replaces the superseded shape
above) — real counts from
`crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-p6-b3-dense-raw-runs/s128_flash_off_1.json`
(the SAME checkpoint/shape as the fused golden, `attention_block_flash`
alone disabled — git_sha `b98f7e1de35b5cfcac00325089fd5eeaf7c6259a`, box
`a100c`) and
`crates/jammi-kernels/artifacts/cuda-runs/2026-08-25-adamw-d959805-a100-sxm4-raw-runs/a100b/b8_s512_disabled.r2.json.raw`
(the adamw-disabled leg, `kernels_disabled_requested: [adamw_step_fused]`).
Neither committed artifact disables BOTH ops in the same real run (the
finetune-step flash-vs-block A/B and the finetune-step multi-tensor-AdamW
A/B are two SEPARATE campaigns, run independently) — this golden composites
their two independently-real per-op counts, since the finetune-run tier's
own `alloff` arm is the first campaign to disable both at once and no
CPU-hermetic environment can produce a real leg for it:

| field | value | source |
|---|---|---|
| `ln_fused_dispatches` / `ln_eager_dispatches` | 1710 / 0 | s128_flash_off_1.json (unaffected by either disable) |
| `rope_fused_dispatches` / `rope_eager_dispatches` | 0 / 0 | s128_flash_off_1.json (absorbed by the block arm's own fused count below) |
| `softmax_fused_dispatches` / `softmax_eager_dispatches` | 0 / 0 | s128_flash_off_1.json (absorbed) |
| `geglu_fused_dispatches` / `geglu_eager_dispatches` | 840 / 0 | s128_flash_off_1.json |
| `lora_epilogue_fused_dispatches` / `lora_epilogue_eager_dispatches` | 0 / 0 | s128_flash_off_1.json |
| `lora_linear_fused_dispatches` / `lora_linear_eager_dispatches` | 3360 / 0 | s128_flash_off_1.json |
| `attention_block_fused_dispatches` / `attention_block_eager_dispatches` | 840 / 0 | s128_flash_off_1.json — the disabled flash cascade falls through to the block arm's own, still-ACTIVE fused kernel: the positive training-path proof for this arm |
| `attention_block_flash_fused_dispatches` / `..._declined_dispatches` | 0 / 840 | s128_flash_off_1.json — `kernels_disabled_requested == kernels_disabled_fired == ["attention_block_flash"]` on that real leg |
| `adamw_fused_dispatches` / `adamw_eager_dispatches` | 0 / 6720 | b8_s512_disabled.r2.json.raw |
| `flash_compiled` | `true` | both source artifacts |
| `kernels_disabled_requested` / `kernels_disabled_fired` | `["adamw_step_fused", "attention_block_flash"]` / same | composited (each source artifact independently confirms ITS OWN op firing as disabled; `finetune_run_ab.sh`'s own real invocation requests both at once) |
| `attention_arm` | `"eager"` | an attention base (`attention_block_flash`) is in `kernels_disabled_requested` ⇒ `"eager"` (`attention_arm`'s own field doc) |
