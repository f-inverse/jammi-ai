# Provenance — `finetune_run_golden/` fixtures

Every file here is ONE real report the compiled `jammi-bench finetune-run`
binary wrote, copied byte-for-byte (2-space `json.dumps(..., indent=2)`
shape) — no field added, removed, edited, or cross-sourced from a second
artifact. A hand-rolled fixture drifts from what the binary emits (a
counter pair the producer serializes but the fixture omits reaches the
merger as a `KeyError` on the first real leg); a producer-emitted golden
cannot.

`crates/jammi-bench/reference/test_torch_finetune_run_mirrors.py` reads
`bert_fused.json`'s `tiers.finetune_run` block to pin the torch twin's
held-out partition digest to the one a real jammi leg reports; the ladder's
premise checks (`crates/jammi-bench/src/ladder/premise.rs`) read the
ModernBERT pair as legs the real binary emitted, so the dispatch-proof
gate is tested against every counter the producer serializes rather than a
hand-typed set — the risk these goldens close is a MISSING field name,
never a specific numeric value.

## `bert_fused.json` — CPU-hermetic

Built with `cargo build --release -p jammi-bench` (no `cuda` feature) inside
the CI image (`ci/dev.sh`) at `fe874c02700bd741a2bd7ad1ed77b3eff3742461`
(`provenance.build_sha`), host triple `aarch64-unknown-linux-gnu`, using the
CLI shape
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

The leg is the `resident` rung (`rung`, `ran_on`: this process, role
`bench`), so it carries no job-path station. The measured cost fields —
`train_run_wall_s`, `epoch_walls`, `iter_wall_s`, each `trajectory` point's
cumulative walls, `peak_rss_bytes` — vary run to run and are not identity
fields the ladder compares. `peak_vram_bytes` reads `null`: the image has no
device-memory probe. The run wrote its untrained adapter into its work dir;
`initial_adapter_sha256` is that file's digest, and
`finetune_run_smoke.rs::finetune_run_emits_a_reproducible_pairing_surface`
pins it, and both token-batch digests, equal across two processes.

Nonzero dispatch counters: `ln_fused: 12`, `softmax_fused: 4`,
`gelu_fused: 4` (`tiny_bert`'s dense-GELU FFN dispatches through
`gelu_erf_fused`), `lora_linear_fused: 8`, `adamw_fused: 16`,
`attention_block_eager: 4` and `attention_block_flash_declined: 4` (every
training-mode attention forward on this `head_dim == 16` fixture reaches the
attention cascade and is declined by the `head_dim == 64` domain predicate —
see `finetune_run.rs`'s
`fused_dispatch_proof_gate_passes_bert_counted_eager_head16_shape`). BERT has
no fused RoPE/GEGLU kernel, so those pairs read `0 / 0`.

## `modernbert_fused.json` / `modernbert_eager.json` — A100

The `fused_r1`/`alloff_r1` legs of one `finetune_run_ab.sh` run (an `alloff` arm
that script had then: the flash cascade and the fused AdamW step requested off;
the leg's `arm` field reads `eager`, the label the binary emits for a process
with families off — the one field of `modernbert_eager.json` that differs from
the source leg)
(`FINETUNE_RUN_AB_SEEDS=1`, `MODEL_DIR` = `answerdotai/ModernBERT-large`,
`head_dim == 64`), built `--features cuda,jammi-encoders/flash-attn`
(`flash_compiled: true` on both legs) at
`869c65f92aea21c6aa6a3ef12cc77f1132e3be80` (each leg's `provenance.build_sha`),
on an NVIDIA A100 80GB PCIe (driver 595.91.07, `x86_64-unknown-linux-gnu`).
They carry that build's field set: the truncation cap under the name `seq`
(this build's `max_seq_length`), no `held_out_at_init`, `epoch_walls`, memory,
token-batch digest or `initial_adapter_sha256` fields, and a `steps_measured` of 234 —
that build's sum of each resume leg's absolute step counter, for a run of 117
optimizer steps (`adamw_fused_dispatches / 224` adapter tensors).
The ladder's premise checks read these two for their dispatch counters and
the dispatch-proof gate's inputs only.
For seed 1, r1 and r2 are bit-identical on both arms
(`determinism_floor.max_delta: 0.0`), so r1 is a representative leg. The
run's own report reads `INVALID` only because the pre-registered decision
rule renders a verdict at exactly 12 premise-clean seeds and this was a
deliberate one-seed producer probe; every leg reads `OK` and the seed's
premise is clean.

| golden file | sha256 (== the source leg's sha256) | source leg |
|---|---|---|
| `modernbert_fused.json` | `0f3a0cdefb0494136fbb2ae7a73660c693680c0e1b1b62dafa9f505cfdc1d412` | `raw/seed1__fused__r1.json` |
| `modernbert_eager.json` | `66131ef2705f8681f0acf036bc997bc3170ff7f15f42b1b0467d8c7515b57e98` (the source leg's) | `raw/seed1__alloff__r1.json` |

Dispatch counters (fused / declined-or-eager):

| base | `modernbert_fused.json` | `modernbert_eager.json` |
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
`ln`/`geglu`/`lora_linear` still read fused on it. Both legs clear the
ladder's dispatch-proof premise for their own arm.

## Regenerating

A producer change that adds a counter pair to the train-run leg means all
three files are regenerated from real runs as above — never hand-edited; any
other field change means `bert_fused.json` is. The ModernBERT pair needs a CUDA A100 and the
real ModernBERT-large checkpoint; no CPU-hermetic run reproduces its device,
checkpoint, or dispatch-count fields.
