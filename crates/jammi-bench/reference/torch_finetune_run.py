#!/usr/bin/env python3
"""PyTorch + PEFT twin of jammi-bench's `finetune-run` tier: one whole
fine-tune, end to end, emitted as a leg a reader can set beside a jammi leg.

`torch_finetune_step.py` twins ONE optimizer step (cost); `torch_grad_oracle.py`
twins ONE backward (gradient direction). This script twins the RUN: the same
checkpoint, the same committed rows in the same order, the same batch
partition, the same objective and hyperparameters, the same number of epochs
with early stopping off — trained with HuggingFace `transformers` + `peft`
instead of jammi's trainer. Like its two siblings it is an ORACLE, not a
dependency: pure Python, never a Cargo dependency, never vendored into a
crate, evidence for a reader and never a merge gate.

THE PAIRING PREMISE. Every jammi leg writes its untrained adapter to
`<work-dir>/initial_adapter.safetensors`; this script loads that file
(`--initial-adapter`), so for seed `s` both sides start from byte-identical
LoRA tensors, and both record the file's digest as `initial_adapter_sha256`. With `--lora-dropout 0`
on both sides no randomness is left unshared: the two runs differ by
arithmetic only, and their outcomes can be paired seed by seed. `--lora-init
peft` (PEFT's own draw) is kept for cost-only runs; it records `lora_init:
"peft"`, which no jammi leg carries, so such a leg can never be mistaken for
a paired one.

WHAT IS MEASURED, under the names `FinetuneRunTier` uses
(`crates/jammi-bench/src/report.rs`):

* LEARNING — `held_out_at_init` (the held-out example-mean at the untrained
  model: after the shared adapter is loaded, before step 1 — the origin a
  run's learning effect is measured from), `held_out_example_mean` (the
  final epoch's), `trajectory` (one point per evaluated
  epoch, each carrying `run_wall_s_cumulative` and `steps_wall_s_cumulative`,
  the seconds spent up to that epoch's end — so "how long until the loss
  first reached X" is answerable, and a faster step cannot hide slower
  convergence), and `train_probe_series`.
* SPEED — `train_run_wall_s` and `epoch_walls`, each epoch's wall whole
  (`run_s`) and by the phases jammi's trainer reports for itself: `steps_s`
  (training compute), `validation_s`, `checkpoint_s`. The spans are jammi's
  (see TIMED SPANS below); comparing `steps_s` compares training, without
  either side's checkpoint path riding along.
* SPACE — `peak_rss_bytes` (the kernel's resident-set high-water mark for the
  process) and `peak_vram_bytes` (whole-device memory above a baseline, from
  the same `nvidia-smi` poll at the same 25 ms interval under the same
  baseline rule as `crates/jammi-bench/src/vram.rs`). torch's own allocator
  counters are recorded under `provenance` and are NOT the comparable figure.

TRAINER BEHAVIOURS — every behaviour of `jammi_ai::fine_tune::trainer` (and of
the tier driving it) that can move an outcome or a cost, and what this script
does about it. REPRODUCED means the same rule is implemented here; DIFFERENT
names what could not be and why.

Data and batching
  1. Row order: file order, never shuffled — REPRODUCED (no sampler, no
     DataLoader; rows are sliced by index).
  2. Train/validation split: the LAST `round(n * validation_fraction)` rows
     are validation (`data::split_index`; Rust rounds half away from zero) —
     REPRODUCED (`split_index`).
  3. Batch partition: consecutive `batch`-sized chunks, the trailing short
     chunk KEPT (`ceil(n / batch)` batches) — REPRODUCED.
  4. One forward per batch over anchors then positives joined on the batch
     axis (`encode_groups`), split after pooling — REPRODUCED.
  5. Held-out rows scored in `--heldout-ids` order, text joined by
     `anchor_id`, every batch full or the run refuses — REPRODUCED.
  6. JSONL line splitting on `\\n` only (Rust `str::lines`), never on the
     Unicode separators Python's `splitlines` also honours — REPRODUCED.

Tokenization
  7. `tokenizer.json` as shipped, special tokens added, truncation to
     `min(max_seq_length, max_position_embeddings)` (longest-first, right
     side, stride 0) — REPRODUCED through the HuggingFace fast tokenizer.
  8. Right-padding to the batch's longest row with id 0 and mask 0 (jammi
     overrides the tokenizer's own padding with `PaddingParams::default()`,
     whose pad id is 0 whatever the vocabulary calls its pad token) —
     REPRODUCED: this script pads itself rather than through the tokenizer's
     `pad_token`.
  9. Training batches padded further, to the bucket ladder
     `{8, 16, 32, ...}` capped at the effective max length; evaluation batches
     left at natural width — REPRODUCED under `--width bucketed` (the default;
     `bucket_seq_len`). DIFFERENT, on purpose, under `--width natural`: see
     TWO WIDTHS below.
 10. Tokenization happens per batch, per epoch, inside the timed span —
     REPRODUCED.
     Items 7-10 are CHECKED, not assumed: both sides digest the token batches
     they fed their encoders (`train_token_ids_sha256`,
     `heldout_token_ids_sha256`), and the digests are identity fields.

Model and adapters
 11. Backbone frozen, at `--backbone-dtype` — REPRODUCED (`torch_dtype`).
 12. No backbone dropout in training mode (jammi's encoders have none) —
     REPRODUCED: every backbone dropout probability in the checkpoint's
     config is forced to 0 and the overridden values are recorded.
 13. LoRA on the same linears (`--target-modules`, suffix-matched by both
     stacks; `--layers-to-transform`), rank, `alpha / rank` scaling, no bias,
     no rsLoRA — REPRODUCED. The loaded adapter's tensor set must equal the
     model's trainable set exactly or the run refuses.
 14. Adapter tensors in f32 under a bf16 backbone, the LoRA arm computed in
     f32 and summed into the base output in the wider dtype — REPRODUCED
     (PEFT's `autocast_adapter_dtype`, which jammi's `LoraLinear` follows);
     the run refuses if any trainable tensor is not f32.
 15. Initial adapter tensors — REPRODUCED bit for bit under `--lora-init
     zeros_b|gaussian` (loaded from jammi's dump); DIFFERENT under `--lora-init
     peft` by design (PEFT's own draw, recorded as such).
 16. LoRA dropout mask — DIFFERENT whenever `--lora-dropout > 0`: jammi draws
     a counter-keyed Philox mask, torch its own generator's. The paired
     protocol runs both sides at 0, where no mask is drawn.
 17. Mean pooling over the attention mask, then L2 normalization with a
     dtype-exact floor, in the working dtype — REPRODUCED
     (`torch_finetune_step.pool_and_normalize`).

Objective and optimizer
 18. MNRL: rows re-normalized with a 1e-8 norm floor, similarity `A·Pᵀ`
     scaled by `--temperature`, symmetric (row and column cross-entropy
     averaged), in the embeddings' own dtype — REPRODUCED.
 19. Gradient accumulation: each micro-batch's loss divided by the window
     size, the epoch's trailing partial window by ITS size, and flushed as one
     extra step — REPRODUCED.
 20. Divergence guard: a micro-batch whose loss is NaN or > 100 is dropped
     without occupying a window slot; three in a row fail the run —
     REPRODUCED.
 21. Global-norm gradient clip with torch's `max_norm / (norm + 1e-6)` —
     REPRODUCED up to reduction order (jammi sums squares across tensors in
     name order; torch takes the norm of per-tensor norms). Off when
     `--max-grad-norm <= 0`.
 22. Non-finite gradient norm refusal on step 1, every 50th step, and the
     run's last step — REPRODUCED at that cadence (not every step, which
     would add a host sync jammi does not pay).
 23. AdamW, decoupled weight decay on every trainable tensor, betas
     (0.9, 0.999), eps 1e-8 — REPRODUCED. jammi's fused multi-tensor step has
     `--adamw-foreach` as its torch peer; the choice is recorded.
 24. Learning rate: linear warmup `lr * step / warmup_steps` from step 0 (so
     the first step runs at 0 when warmup is on), then constant, cosine or
     linear decay over the run's whole horizon (`ceil(batches / grad_accum) *
     epochs` steps), floored at 0 — REPRODUCED (`compute_lr`).
 24b. The zero-learning-rate control (`--zero-lr-control`): the same job,
     every optimizer step applied at rate 0 (`AppliedLearningRate::Zero`), so
     the whole loop runs and no tensor moves; the leg reports `lr: 0.0`. A
     non-positive `--lr` is refused, as it is for any jammi job — REPRODUCED.
 25. Early stopping disabled (`--early-stopping-patience >= 10000`) —
     REPRODUCED as the same refusal.

Evaluation
 26. Evaluation in eval mode, no gradient — REPRODUCED.
 27. Validation pass (when `--early-stopping-metric val_loss`): batch-mean
     loss per validation batch, averaged over batches, INSIDE the timed span —
     REPRODUCED.
 28. Held-out example loss: per-row NLL from the f32 logits on the host, a
     max-subtracted log-sum-exp folded left to right in f32, symmetric —
     REPRODUCED (`cross_entropy_per_row`), including the exact-zero tie floor.
 29. Held-out evaluation and the train-side probe (the first `batch` train
     rows) after every epoch, and both once before training at the untrained
     model (`held_out_at_init`, the probe's index 0), all OUTSIDE the timed
     span — REPRODUCED.
 30. `heldout_batch_partition_sha256`: sha256 of the compact JSON of the id
     batches — REPRODUCED.

Checkpointing (cost only; none of it changes a weight)
 31. Adapter weights written every `ceil(0.1 * horizon)` steps — REPRODUCED
     as a safetensors write at the same steps.
 32. At each epoch boundary: a finite-parameter check, the "best" adapter
     written, an epoch bundle (adapter + both AdamW moments + counters)
     written, the best adapter read back, and the final adapter + metadata
     written — REPRODUCED as the same reads and writes of the same payloads.
     Under the tier's one-epoch-per-leg cycle the best adapter IS the current
     one, so reading it back changes nothing on either side.
 33. jammi's epoch bundle also goes through its artifact store and a catalog
     row — DIFFERENT: no torch analogue; this script writes the bundle to
     local disk once. The cost lands in `checkpoint_s` on both sides and
     nowhere else, so it is visible and separable rather than absorbed.
 34. jammi's tier takes the run one epoch per `run()` call, rebuilding the
     model and restoring adapter, moments and counters from that bundle at
     every boundary — DIFFERENT in mechanism, identical in effect: each call
     keeps the run's full `epochs`, so schedule and cadences are the
     uninterrupted run's, and the restore is exact (jammi pins a sliced run
     byte-identical to an uninterrupted one), so this script keeps its model
     live. The rebuild sits outside every jammi span; the restore is charged
     to jammi's `checkpoint_s` and has no counterpart here.

TWO WIDTHS. jammi pads training batches up a bucket ladder because its
allocator wants few distinct shapes, and its variable-length attention path
does not pay for the padding. A PyTorch user has neither reason: they pad to
the batch's longest row, and padded attention pays for every padded column.
So, like `--attn eager|sdpa` on the step twin, this twin has two legs:
`--width bucketed` is the SEMANTIC twin — the token batches jammi feeds, digest
for digest — and the leg that pairs with a jammi leg on outcome; `--width
natural` pads to the batch's longest row and is the PRACTICAL BAR for speed
and space. `width` is an identity field of a torch leg: the two are different
computations, and a natural leg's `train_token_ids_sha256` differs from
jammi's by construction (its held-out digest does not — evaluation is at
natural width everywhere). The LOSS does not depend on the width beyond
rounding: padded positions are masked out of attention and out of the pooling
mean, and position encodings are absolute, so a row's embedding is the same
function of its real tokens at any padded width; only the shapes the kernels
reduce over change. `test_finetune_run_cross_producer_parity.py` holds a
natural leg to the bucketed leg and to jammi within the same 1e-5.

TIMED SPANS. `run_s` is the wall around one epoch's worth of what
`TrainingLoop::run` does — items 9-10, 18-24, 27 and 31-32 — and
`train_run_wall_s` its sum: the same inclusions as jammi (tokenization,
validation, every checkpoint write) and the same exclusions (model build,
held-out evaluation, probes). Inside it, `steps_s` is the batch loop with the
device synchronized before the clock stops and the step-checkpoint writes
taken out; `validation_s` the validation pass; `checkpoint_s` every
checkpoint read and write. They are disjoint and leave a small remainder,
exactly as jammi's `RunPhaseWall` does.

VRAM BASELINE. The window opens once the model and adapters are resident and
the optimizer is constructed, before anything runs through the model. jammi's
trainer builds its optimizer inside the timed call, after the baseline, and
torch allocates AdamW state on the first step — so the moments land inside
the window on both sides with no warm-up step, which a learning comparison
could not afford anyway (it would be an optimizer update the other side never
took).

Install: `torch`, `transformers>=4.48`, `peft`, `safetensors`, `numpy` — the
venv `ci/scripts/perf/torch_venv.py` names. No pin file ships beside this
script; the versions actually present are recorded in every leg's
`provenance`, which is the authority.

Usage (paired with a jammi leg at the same seed):
    jammi-bench finetune-run ... --seed 3 --lora-dropout 0 \\
        --work-dir jammi-work > jammi.json
    python3 torch_finetune_run.py --model-dir ... --seed 3 --lora-dropout 0 \\
        --initial-adapter jammi-work/initial_adapter.safetensors ... > torch.json

Usage (anywhere, no GPU, no checkpoint — a tiny random ModernBERT, a tiny
tokenizer and synthetic pairs, through the same code path):
    python3 crates/jammi-bench/reference/torch_finetune_run.py --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import struct
import subprocess
import sys
import tempfile
import threading
import time

import torch_finetune_step as tfs
import torch_grad_oracle as tgo

# `jammi_numerics::MIN_BUCKET_LEN`.
MIN_BUCKET_LEN = 8
# `fine_tune::optimizer::DEFAULT_NORM_CHECK_INTERVAL`.
NORM_CHECK_INTERVAL = 50
# The divergence guard's bound and strike count (`process_batch_loss`).
DIVERGENCE_LOSS_BOUND = 100.0
DIVERGENCE_STRIKES = 3
# `VramSampler`'s poll interval.
VRAM_POLL_INTERVAL_S = 0.025
# The tier's run protocol — `finetune_run::DEFAULT_LEARNING_RATE`,
# `DEFAULT_EPOCHS`, `DEFAULT_EVAL_CADENCE` (that constant's doc says why they
# are not the engine's defaults); `test_torch_finetune_run_mirrors.py` reads
# the Rust constants and holds these equal to them.
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_EPOCHS = 4
DEFAULT_EVAL_CADENCE = 1
# `jammi_wire::fine_tune::DEFAULT_MAX_SEQ_LENGTH`, the engine's own default
# truncation length and so `jammi-bench finetune-run`'s;
# `test_torch_finetune_run_mirrors.py` reads the Rust constant and holds this
# equal to it.
DEFAULT_MAX_SEQ_LENGTH = 512
# The never-stops patience the tier requires.
NEVER_STOPS_PATIENCE = 10_000
# `--schedule`'s spellings (jammi-bench's own) and the value a leg records for
# each: `format!("{:?}", LrSchedule).to_lowercase()`.
SCHEDULE_IDENTITY = {
    "constant": "constant",
    "cosine_decay": "cosinedecay",
    "linear_decay": "lineardecay",
}

# The identity tuple a jammi `finetune-run` leg carries, in
# `ci/scripts/perf/identity_fields.py::FINETUNE_RUN_IDENTITY_FIELDS` order.
# `test_torch_finetune_run_dry_run.py` holds this equal to that tuple, so the
# two producers cannot drift apart on what makes legs comparable.
RUN_IDENTITY_FIELDS = (
    "seed",
    "task",
    "batch",
    "max_seq_length",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_init",
    "margin",
    "target_modules",
    "layers_to_transform",
    "backbone_dtype",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "max_grad_norm",
    "warmup",
    "row_lengths",
    "epochs",
    "lr",
    "schedule",
    "warmup_steps",
    "weight_decay",
    "grad_accum",
    "validation_fraction",
    "train_pairs_file_sha256",
    "train_media_sha256",
    "heldout_ids_sha256",
    "heldout_pairs_sha256",
    "heldout_media_sha256",
    "heldout_batch_partition_sha256",
    "train_token_ids_sha256",
    "heldout_token_ids_sha256",
    "embedding_loss",
    "temperature",
    "matryoshka_dims",
    "early_stopping_patience",
    "early_stopping_metric",
    "eval_cadence",
)

# What a torch leg's identity carries beyond `RUN_IDENTITY_FIELDS`: two widths
# are two computations (see TWO WIDTHS in the module docstring).
TORCH_LEG_IDENTITY_FIELDS = ("width",)

# Every backbone dropout probability the supported architectures' configs
# carry. jammi's encoders apply none of them (behaviour 12).
BACKBONE_DROPOUT_FIELDS = (
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "embedding_dropout",
    "mlp_dropout",
    "attention_dropout",
)

class Refusal(Exception):
    """An input this twin will not train on: reproducing it is impossible or
    would silently measure something else. Reported on stderr, exit 2."""


# ─── data ────────────────────────────────────────────────────────────────────


def rust_lines(text: str):
    """`str::lines`: split on `\\n`, drop one trailing `\\r` per line."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line[:-1] if line.endswith("\r") else line for line in lines]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_rows(path: str):
    """`main.rs::load_train_jsonl`: rows in file order as `(id, anchor,
    positive)`, plus the sha256 of the file's bytes."""
    with open(path, "rb") as fh:
        data = fh.read()
    rows = []
    for number, line in enumerate(rust_lines(data.decode("utf-8")), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            rows.append((row["anchor_id"], row["anchor_text"], row["positive_text"]))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise Refusal(f"{path} line {number}: {exc!r}") from None
    return rows, sha256_hex(data)


def load_heldout(ids_path: str, jsonl_path: str):
    """`main.rs::load_heldout_fixture`: text joined to the committed id order
    by `anchor_id`; both files' digests measured off the bytes read."""
    with open(ids_path, "rb") as fh:
        ids_data = fh.read()
    text_rows, pairs_sha256 = load_rows(jsonl_path)
    by_anchor = {anchor_id: (anchor, positive) for anchor_id, anchor, positive in text_rows}
    rows = []
    for number, line in enumerate(rust_lines(ids_data.decode("utf-8")), start=1):
        if not line.strip():
            continue
        anchor_id = line.split("\t")[0]
        if anchor_id not in by_anchor:
            raise Refusal(
                f"{ids_path} line {number}: anchor_id {anchor_id!r} has no matching row in {jsonl_path}"
            )
        rows.append((anchor_id, *by_anchor[anchor_id]))
    return rows, sha256_hex(ids_data), pairs_sha256


def split_index(total: int, fraction: float) -> int:
    """`data::split_index`. Rust's `f64::round` rounds half away from zero;
    the product is never negative here, so that is `floor(x + 0.5)`."""
    return total - int(math.floor(total * fraction + 0.5))


def chunks(rows, size: int):
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def partition_sha256(id_batches) -> str:
    """`evaluate_held_out`'s partition digest: sha256 of `serde_json`'s
    compact encoding of `Vec<Vec<String>>`."""
    encoded = json.dumps(id_batches, separators=(",", ":"), ensure_ascii=False)
    return sha256_hex(encoded.encode("utf-8"))


# ─── tokenization ────────────────────────────────────────────────────────────


def bucket_seq_len(natural_len: int, max_seq_length: int) -> int:
    """`jammi_numerics::bucket_seq_len`."""
    if natural_len == 0 or max_seq_length == 0:
        return natural_len
    bucket = min(MIN_BUCKET_LEN, max_seq_length)
    while bucket < natural_len and bucket < max_seq_length:
        bucket = min(bucket * 2, max_seq_length)
    return bucket


class TokenBatcher:
    """The one place this script turns texts into `(input_ids, mask)` — the
    training loop, the evaluation passes and the token digests all go through
    it, so the digest describes what the model was fed."""

    def __init__(self, tokenizer_file: str, effective_max: int, width: str):
        from transformers import PreTrainedTokenizerFast

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self.effective_max = effective_max
        self.width = width

    def encode(self, texts, training: bool):
        """Rows of ids and masks: truncated by the tokenizer, right-padded
        with id 0 to the batch's longest row, then — for a training batch
        under `--width bucketed` — to the bucket ladder."""
        encoded = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            truncation=True,
            max_length=self.effective_max,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        natural = max((len(ids) for ids in encoded), default=0)
        bucketed = training and self.width == "bucketed"
        width = bucket_seq_len(natural, self.effective_max) if bucketed else natural
        input_ids = [ids + [0] * (width - len(ids)) for ids in encoded]
        mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in encoded]
        return input_ids, mask


def token_batches_sha256(batches) -> str:
    """`finetune_run::token_batches_sha256`: per batch `rows`, `cols`, every
    id row-major, every mask entry row-major — all little-endian `u32`."""
    hasher = hashlib.sha256()
    for input_ids, mask in batches:
        rows = len(input_ids)
        cols = len(input_ids[0]) if rows else 0
        hasher.update(struct.pack("<II", rows, cols))
        for plane in (input_ids, mask):
            for row in plane:
                hasher.update(struct.pack(f"<{len(row)}I", *row))
    return hasher.hexdigest()


def joined_texts(batch_rows):
    """Anchors, then positives — the order `encode_chunk` joins a pairs chunk
    in for its single forward."""
    return [row[1] for row in batch_rows] + [row[2] for row in batch_rows]


# ─── memory ──────────────────────────────────────────────────────────────────


def nvidia_smi_memory_used():
    """`vram::nvidia_smi_memory_used`: the first line of the whole-device
    `memory.used` query, MiB → bytes; `None` when the host cannot say."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return int(out.stdout.splitlines()[0].strip()) * 1024 * 1024
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


class VramWindow:
    """`vram::VramWindow`: a baseline read, then a background poll of the same
    probe every 25 ms; `close()` is the high-water mark above the baseline, or
    `None` when nothing could be sampled."""

    def __init__(self, probe=nvidia_smi_memory_used):
        self._probe = probe
        first = probe()
        self._baseline = first or 0
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None
        if first is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()

    def _poll(self):
        while not self._stop.is_set():
            used = self._probe()
            if used is not None:
                self._peak = max(self._peak, used)
            self._stop.wait(VRAM_POLL_INTERVAL_S)

    def close(self):
        if self._thread is None:
            return None
        self._stop.set()
        self._thread.join()
        return float(max(self._peak - self._baseline, 0))


def peak_rss():
    """The kernel's resident-set high-water mark for this process, in bytes,
    and which counter it came from: `/proc/self/status` `VmHWM` where it
    exists (jammi's source), otherwise `getrusage`'s `ru_maxrss` — the same
    kernel quantity (bytes on Darwin, KiB elsewhere)."""
    vm_hwm = tfs.peak_rss_bytes()
    if vm_hwm is not None:
        return vm_hwm, "proc_vm_hwm"
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(max_rss if sys.platform == "darwin" else max_rss * 1024), "getrusage_maxrss"


def device_name(device) -> str:
    """`finetune_step::device_name`."""
    if device.type != "cuda":
        return "cpu"
    return tfs.nvidia_smi_field("name") or "unknown"


# ─── model ───────────────────────────────────────────────────────────────────


def load_frozen_backbone(args):
    """The checkpoint at `--backbone-dtype`, every backbone dropout forced to
    zero (behaviour 12). Returns the model, its config, the attention
    implementation HF resolved, and the dropout values that were overridden."""
    from transformers import AutoModel

    import torch

    config = tfs.load_config_reference_compile_off(args.model_dir)
    overridden = {}
    for field in BACKBONE_DROPOUT_FIELDS:
        value = getattr(config, field, None)
        if value:
            overridden[field] = value
            setattr(config, field, 0.0)
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[args.backbone_dtype]
    model = AutoModel.from_pretrained(
        args.model_dir, config=config, attn_implementation=args.attn, torch_dtype=dtype
    )
    resolved_attn = getattr(model.config, "_attn_implementation", "absent")
    return model, config, resolved_attn, overridden


def wrap_lora(model, args):
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            layers_to_transform=args.layers_to_transform,
            bias="none",
        ),
    )


# jammi's adapter tensor names for a BERT checkpoint, by the PEFT module path
# they wrap. ModernBERT's table is `torch_grad_oracle`'s.
_BERT_SITE_BY_PEFT_PATH = {
    "attention.self.query": "query",
    "attention.self.key": "key",
    "attention.self.value": "value",
    "attention.output.dense": "dense",
    "intermediate.dense": "intermediate_dense",
    "output.dense": "output_dense",
}


def jammi_tensor_name(model_type: str, peft_name: str):
    """jammi's name for a PEFT LoRA parameter, or `None` when the table does
    not cover it (the caller refuses; nothing is skipped)."""
    if model_type == "modernbert":
        return tgo.translate_peft_name_to_jammi(peft_name)
    if model_type == "bert":
        prefix, suffix_a, suffix_b = "base_model.model.encoder.layer.", ".lora_A.default.weight", ".lora_B.default.weight"
        for suffix, leaf in ((suffix_a, "lora_a"), (suffix_b, "lora_b")):
            if peft_name.startswith(prefix) and peft_name.endswith(suffix):
                layer, _, path = peft_name[len(prefix) : -len(suffix)].partition(".")
                site = _BERT_SITE_BY_PEFT_PATH.get(path)
                if site is not None and layer.isdigit():
                    return f"layer.{layer}.{site}.{leaf}"
    return None


def trainable_by_jammi_name(model, model_type: str):
    named = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        jammi_name = jammi_tensor_name(model_type, name)
        if jammi_name is None:
            raise Refusal(
                f"trainable parameter {name!r} has no jammi tensor name for model_type "
                f"{model_type!r} — --target-modules selects a linear jammi does not wrap, so the "
                "two adapters could not be the same set"
            )
        named[jammi_name] = param
    return named


def load_initial_adapter(named, path: str, lora_init: str) -> str:
    """Copy jammi's dumped initial adapter into the PEFT parameters and return
    the file's sha256. The file's tensor set must equal the trainable set, and
    under `zeros_b` every `lora_b` must be exactly zero — otherwise the leg
    would carry a `lora_init` label its tensors do not honour."""
    from safetensors.torch import load_file

    import torch

    tensors = load_file(path)
    if set(tensors) != set(named):
        raise Refusal(
            f"{path} and the model disagree on the adapter's tensors — only in the file: "
            f"{sorted(set(tensors) - set(named))}; only in the model: {sorted(set(named) - set(tensors))}"
        )
    with torch.no_grad():
        for name, param in named.items():
            source = tensors[name]
            if tuple(source.shape) != tuple(param.shape):
                raise Refusal(f"{name}: {path} has shape {tuple(source.shape)}, the model {tuple(param.shape)}")
            if lora_init == "zeros_b" and name.endswith(".lora_b") and bool(source.ne(0).any()):
                raise Refusal(f"{name} in {path} is not zero, so this is not a zeros_b adapter")
            param.copy_(source.to(param.dtype))
    with open(path, "rb") as fh:
        return sha256_hex(fh.read())


def adapter_state(named):
    return {name: param.detach().cpu().contiguous() for name, param in named.items()}


# ─── objective ───────────────────────────────────────────────────────────────


def l2_normalize_rows(x):
    """`trainer::l2_normalize_rows` (the 1e-8 floor)."""
    return x / x.pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)


def mnrl_similarity(anchors, positives, scale: float):
    return (l2_normalize_rows(anchors) @ l2_normalize_rows(positives).t()) * scale


def mnrl_loss(anchors, positives, scale: float):
    """`trainer::mnrl_loss`, symmetric, no hard negatives."""
    import torch
    import torch.nn.functional as F

    sim = mnrl_similarity(anchors, positives, scale)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) * 0.5


def cross_entropy_per_row(logits):
    """`trainer::cross_entropy_per_row` with diagonal labels: f32 on the host,
    max-subtracted, summed left to right."""
    import numpy as np

    losses = []
    for index, row in enumerate(logits):
        peak = row.max()
        total = np.float32(0.0)
        for value in row:
            total = np.float32(total + np.exp(np.float32(value - peak), dtype=np.float32))
        losses.append(float(np.float32(np.float32(peak + np.log(total, dtype=np.float32)) - row[index])))
    return losses


def mnrl_loss_per_example(anchors, positives, scale: float):
    """`trainer::mnrl_loss_per_example`, symmetric."""
    sim = mnrl_similarity(anchors, positives, scale).float().cpu().numpy()
    rows = cross_entropy_per_row(sim)
    cols = cross_entropy_per_row(sim.T)
    return [0.5 * (r + c) for r, c in zip(rows, cols)]


# ─── the run ─────────────────────────────────────────────────────────────────


class Twin:
    """The model, its tokenizer and its optimizer, with the three passes the
    tier makes: a training epoch, a validation pass, a per-example pass."""

    def __init__(self, args, model, named, batcher, device):
        import torch

        self.args = args
        self.model = model
        self.named = named
        # Name order, the order jammi builds its optimizer and folds its clip in.
        self.trainable = [named[name] for name in sorted(named)]
        self.batcher = batcher
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.trainable,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=args.adamw_foreach,
        )
        self.global_step = 0
        self.divergence_count = 0

    def embed(self, batch_rows, training: bool):
        import torch

        input_ids, mask = self.batcher.encode(joined_texts(batch_rows), training)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention = torch.tensor(mask, dtype=torch.long, device=self.device)
        pooled = tfs.pool_and_normalize(tfs.forward_hidden(self.model, ids, attention), attention)
        rows = len(batch_rows)
        return pooled[:rows], pooled[rows:]

    def learning_rate(self, horizon: int) -> float:
        """`trainer::compute_lr`: the rate for the step about to run, over a
        `horizon`-step run."""
        args, step = self.args, self.global_step
        if args.zero_lr_control:
            return 0.0
        if step < args.warmup_steps:
            return args.lr * (step / max(args.warmup_steps, 1))
        decay_steps = max(horizon - args.warmup_steps, 0)
        progress = min(max((step - args.warmup_steps) / max(decay_steps, 1), 0.0), 1.0)
        rate = {
            "constant": args.lr,
            "cosine_decay": args.lr * 0.5 * (1.0 + math.cos(math.pi * progress)),
            "linear_decay": args.lr * (1.0 - progress),
        }[args.schedule]
        return max(rate, 0.0)

    def optimizer_step(self, grads, horizon: int):
        import torch

        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate(horizon)
        for param, grad in zip(self.trainable, grads):
            param.grad = grad
        step = self.global_step + 1
        if self.args.max_grad_norm > 0.0:
            checked = step == 1 or step % NORM_CHECK_INTERVAL == 0 or step == horizon
            torch.nn.utils.clip_grad_norm_(
                self.trainable, self.args.max_grad_norm, error_if_nonfinite=checked, foreach=False
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step = step

    def sync(self):
        import torch

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def write_step_checkpoint(self, scratch: str, walls: dict):
        from safetensors.torch import save_file

        started = time.perf_counter()
        save_file(adapter_state(self.named), os.path.join(scratch, f"step_{self.global_step}.safetensors"))
        walls["checkpoint_s"] += time.perf_counter() - started

    def train_epoch(self, train_rows, val_rows, epoch: int, scratch: str):
        """Everything `TrainingLoop::run` does for one epoch — the `run_s`
        span — with the phases jammi's `RunPhaseWall` names timed inside it.
        Returns `(avg_train_loss, avg_val_loss or None, walls)`."""
        import torch
        from safetensors.torch import load_file, save_file

        args = self.args
        run_started = time.perf_counter()
        walls = {"steps_s": 0.0, "validation_s": 0.0, "checkpoint_s": 0.0}
        batches = chunks(train_rows, args.batch)
        grad_accum = max(args.grad_accum, 1)
        # Every horizon is the WHOLE run's: the schedule, the step-checkpoint
        # cadence and the last step do not depend on which epoch this is.
        horizon = -(-len(batches) // grad_accum) * args.epochs
        checkpoint_interval = math.ceil(horizon * 0.1)
        partial_window = len(batches) % grad_accum

        steps_started = time.perf_counter()
        checkpoints_before_steps = walls["checkpoint_s"]
        self.model.train()
        batch_count = 0
        epoch_loss = 0.0
        accumulated = None
        for batch_rows in batches:
            anchors, positives = self.embed(batch_rows, training=True)
            loss = mnrl_loss(anchors, positives, args.temperature)
            candidate = batch_count + 1
            trailing = partial_window != 0 and candidate > len(batches) - partial_window
            scale = partial_window if trailing else grad_accum
            grads = torch.autograd.grad(loss / scale, self.trainable, allow_unused=True)
            loss_value = float(loss.detach().float().item())
            if math.isnan(loss_value) or loss_value > DIVERGENCE_LOSS_BOUND:
                self.divergence_count += 1
                if self.divergence_count >= DIVERGENCE_STRIKES:
                    raise Refusal("training diverged: loss was NaN or >100 for 3 consecutive batches")
                continue
            self.divergence_count = 0
            epoch_loss += loss_value
            batch_count = candidate
            grads = [torch.zeros_like(p) if g is None else g for p, g in zip(self.trainable, grads)]
            accumulated = grads if accumulated is None else [a + g for a, g in zip(accumulated, grads)]
            if batch_count % grad_accum == 0:
                self.optimizer_step(accumulated, horizon)
                accumulated = None
                if self.global_step % checkpoint_interval == 0:
                    self.write_step_checkpoint(scratch, walls)
        if accumulated is not None:
            self.optimizer_step(accumulated, horizon)
            if self.global_step % checkpoint_interval == 0:
                self.write_step_checkpoint(scratch, walls)
        # Close the step span on a synchronized device, less the checkpoint
        # writes made from inside it (two nested host spans).
        self.sync()
        walls["steps_s"] = (time.perf_counter() - steps_started) - (
            walls["checkpoint_s"] - checkpoints_before_steps
        )
        avg_train_loss = epoch_loss / max(batch_count, 1)

        avg_val_loss = None
        if args.early_stopping_metric == "val_loss":
            validation_started = time.perf_counter()
            avg_val_loss = self.validation_loss(val_rows)
            self.sync()
            walls["validation_s"] = time.perf_counter() - validation_started

        # The epoch boundary: finite-parameter check, then the checkpoint
        # I/O — the best adapter, the epoch bundle, the best adapter read
        # back, the final adapter.
        total = sum(float(p.detach().float().sum().item()) for p in self.trainable)
        if not math.isfinite(total):
            raise Refusal(f"a trainable parameter is non-finite after epoch {epoch}")
        boundary_started = time.perf_counter()
        best_path = os.path.join(scratch, "checkpoint_best.safetensors")
        save_file(adapter_state(self.named), best_path)
        bundle = adapter_state(self.named)
        for name, param in self.named.items():
            state = self.optimizer.state.get(param, {})
            for moment in ("exp_avg", "exp_avg_sq"):
                if moment in state:
                    bundle[f"{name}.{moment}"] = state[moment].detach().cpu().contiguous()
        save_file(bundle, os.path.join(scratch, f"epoch_{epoch}.safetensors"))
        with open(os.path.join(scratch, f"epoch_{epoch}.json"), "w") as fh:
            json.dump({"epoch": epoch, "global_step": self.global_step}, fh)
        with torch.no_grad():
            for name, tensor in load_file(best_path).items():
                self.named[name].copy_(tensor.to(self.named[name].dtype))
        save_file(adapter_state(self.named), os.path.join(scratch, "adapter_model.safetensors"))
        with open(os.path.join(scratch, "adapter_config.json"), "w") as fh:
            json.dump({"lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha}, fh)
        self.sync()
        walls["checkpoint_s"] += time.perf_counter() - boundary_started
        walls["run_s"] = time.perf_counter() - run_started
        return avg_train_loss, avg_val_loss, walls

    def validation_loss(self, val_rows) -> float:
        """`TrainingLoop::evaluate`: the mean over batches of each batch's
        mean loss."""
        import torch

        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_rows in chunks(val_rows, self.args.batch):
                anchors, positives = self.embed(batch_rows, training=False)
                losses.append(float(mnrl_loss(anchors, positives, self.args.temperature).float().item()))
        self.model.train()
        return sum(losses) / len(losses)

    def example_losses(self, rows):
        """`TrainingLoop::evaluate_held_out`: `(mean, tie_fraction,
        partition_sha256, count)` over full batches in the given order."""
        import torch

        if not rows or len(rows) % self.args.batch != 0:
            raise Refusal(f"{len(rows)} rows is not a nonzero multiple of --batch {self.args.batch}")
        self.model.eval()
        losses = []
        id_batches = []
        with torch.no_grad():
            for batch_rows in chunks(rows, self.args.batch):
                anchors, positives = self.embed(batch_rows, training=False)
                batch_losses = mnrl_loss_per_example(anchors, positives, self.args.temperature)
                for (example_id, _a, _p), value in zip(batch_rows, batch_losses):
                    if not math.isfinite(value):
                        raise Refusal(f"non-finite loss ({value}) for held-out example {example_id!r}")
                losses.extend(batch_losses)
                id_batches.append([row[0] for row in batch_rows])
        self.model.train()
        ties = sum(1 for value in losses if value == 0.0)
        return sum(losses) / len(losses), ties / len(losses), partition_sha256(id_batches), len(losses)


def validate(args):
    if args.early_stopping_patience < NEVER_STOPS_PATIENCE:
        raise Refusal(
            f"--early-stopping-patience {args.early_stopping_patience} is below {NEVER_STOPS_PATIENCE}, "
            "the never-stops setting — an early-stopped run has no final epoch to pair"
        )
    if args.epochs == 0:
        raise Refusal("--epochs 0 has no final epoch to measure")
    if not args.lr > 0.0:
        raise Refusal(
            f"--lr {args.lr} is not a positive learning rate: the negative control is the same job run "
            "with its updates nulled — pass the sweep's --lr together with --zero-lr-control"
        )
    if args.lora_init == "peft" and args.initial_adapter:
        raise Refusal("--initial-adapter names jammi's tensors; --lora-init peft draws PEFT's own")
    if args.lora_init != "peft" and not args.initial_adapter:
        raise Refusal(
            f"--lora-init {args.lora_init} is jammi's init: pass the initial_adapter.safetensors a jammi "
            "leg wrote into its --work-dir as --initial-adapter (or run --lora-init peft, which pairs "
            "with nothing)"
        )


def run(args) -> dict:
    import torch

    validate(args)
    fast_path_globals = tfs.pin_fast_path_globals()
    torch.manual_seed(args.seed)
    device = tfs.pick_device(args.cuda)

    train_rows_all, train_pairs_file_sha256 = load_rows(args.train_jsonl)
    heldout_rows, heldout_ids_sha256, heldout_pairs_sha256 = load_heldout(args.heldout_ids, args.heldout_jsonl)
    if len(train_rows_all) < args.batch:
        raise Refusal(f"{len(train_rows_all)} train pairs is fewer than --batch {args.batch}")
    boundary = split_index(len(train_rows_all), args.validation_fraction)
    train_rows, val_rows = train_rows_all[:boundary], train_rows_all[boundary:]
    if args.early_stopping_metric == "val_loss" and not val_rows:
        raise Refusal(
            f"early_stopping_metric=val_loss requires a non-empty validation split, but "
            f"validation_fraction={args.validation_fraction} over {len(train_rows_all)} row(s) holds out none"
        )
    probe_rows = train_rows_all[: args.batch]

    model, config, resolved_attn, dropout_overridden = load_frozen_backbone(args)
    checkpoint = tfs.checkpoint_identity(args.model_dir)
    tokenizer_file = os.path.join(args.model_dir, "tokenizer.json")
    with open(tokenizer_file, "rb") as fh:
        tokenizer_sha256 = sha256_hex(fh.read())
    batcher = TokenBatcher(
        tokenizer_file, min(args.max_seq_length, config.max_position_embeddings), args.width
    )

    model = wrap_lora(model, args)
    model.to(device)
    named = trainable_by_jammi_name(model, config.model_type)
    if not named:
        raise Refusal("no trainable LoRA tensors — --target-modules matched nothing")
    wrong_dtype = sorted(name for name, p in named.items() if p.dtype != torch.float32)
    if wrong_dtype:
        raise Refusal(f"adapter tensors are not f32 (jammi trains them in f32): {wrong_dtype}")
    initial_adapter_sha256 = None
    if args.initial_adapter:
        initial_adapter_sha256 = load_initial_adapter(named, args.initial_adapter, args.lora_init)

    # The token batches one epoch and one held-out pass feed the model.
    epoch_token_batches = [batcher.encode(joined_texts(b), training=True) for b in chunks(train_rows, args.batch)]
    if args.early_stopping_metric == "val_loss":
        epoch_token_batches += [batcher.encode(joined_texts(b), training=False) for b in chunks(val_rows, args.batch)]
    heldout_token_batches = [batcher.encode(joined_texts(b), training=False) for b in chunks(heldout_rows, args.batch)]

    twin = Twin(args, model, named, batcher, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    vram = VramWindow()

    # The untrained model, evaluated once before step 1: the held-out origin
    # the learning effect is measured from, and the probe series' index 0.
    held_out_at_init = twin.example_losses(heldout_rows)[0]
    train_probe_series = [twin.example_losses(probe_rows)[0]]
    trajectory = []
    epoch_walls = []
    train_run_wall_s = 0.0
    steps_wall_s = 0.0
    train_loss_curve, val_loss_curve = [], []
    held_out = None
    with tempfile.TemporaryDirectory(prefix="train-", dir=args.work_dir) as scratch:
        for epoch in range(args.epochs):
            avg_train_loss, avg_val_loss, walls = twin.train_epoch(train_rows, val_rows, epoch, scratch)
            epoch_walls.append(walls)
            train_run_wall_s += walls["run_s"]
            steps_wall_s += walls["steps_s"]
            train_loss_curve.append(avg_train_loss)
            if avg_val_loss is not None:
                val_loss_curve.append(avg_val_loss)

            is_final = epoch + 1 == args.epochs
            if is_final or (args.eval_cadence > 0 and (epoch + 1) % args.eval_cadence == 0):
                held_out = twin.example_losses(heldout_rows)
                trajectory.append(
                    {
                        "epoch": epoch,
                        "held_out_mean": held_out[0],
                        "held_out_tie_fraction": held_out[1],
                        "held_out_batch_partition_sha256": held_out[2],
                        "run_wall_s_cumulative": train_run_wall_s,
                        "steps_wall_s_cumulative": steps_wall_s,
                    }
                )
            train_probe_series.append(twin.example_losses(probe_rows)[0])

    peak_vram = vram.close()
    rss_bytes, rss_source = peak_rss()
    torch_allocator = None
    if device.type == "cuda":
        torch_allocator = {
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    monitored = val_loss_curve if args.early_stopping_metric == "val_loss" else train_loss_curve

    tier = {
        # identity — `FinetuneRunTier::IDENTITY_FIELDS`
        "seed": args.seed,
        "task": "text_embedding",
        "batch": args.batch,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_init": args.lora_init,
        "margin": None,
        "target_modules": args.target_modules,
        "layers_to_transform": args.layers_to_transform,
        "backbone_dtype": args.backbone_dtype,
        **checkpoint,
        "max_grad_norm": args.max_grad_norm if args.max_grad_norm > 0.0 else None,
        "warmup": None,
        "row_lengths": None,
        "epochs": args.epochs,
        # The rate the run was measured at: a control leg's is zero.
        "lr": 0.0 if args.zero_lr_control else args.lr,
        "schedule": SCHEDULE_IDENTITY[args.schedule],
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_accum": args.grad_accum,
        "validation_fraction": args.validation_fraction,
        "train_pairs_file_sha256": train_pairs_file_sha256,
        "train_media_sha256": None,
        "heldout_ids_sha256": heldout_ids_sha256,
        "heldout_pairs_sha256": heldout_pairs_sha256,
        "heldout_media_sha256": None,
        "heldout_batch_partition_sha256": held_out[2],
        "train_token_ids_sha256": token_batches_sha256(epoch_token_batches),
        "heldout_token_ids_sha256": token_batches_sha256(heldout_token_batches),
        "embedding_loss": "mnrl",
        "temperature": args.temperature,
        "matryoshka_dims": [],
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_metric": args.early_stopping_metric,
        "eval_cadence": args.eval_cadence,
        # provenance shared with a jammi leg
        "arm": "torch",
        # Identity of a torch leg, beyond the tuple it shares with jammi.
        "width": args.width,
        "device_name": device_name(device),
        "split_rule": "positional_fraction_split",
        "batched_forward": True,
        "steps_measured": twin.global_step,
        "initial_adapter_sha256": initial_adapter_sha256,
        # measured — the names `FinetuneRunTier` uses
        "tie_fraction": held_out[1],
        "final_epoch": args.epochs - 1,
        "held_out_at_init": held_out_at_init,
        "held_out_example_mean": held_out[0],
        "held_out_count": held_out[3],
        "final_loss_diagnostic": monitored[-1],
        "trajectory": trajectory,
        "train_probe_series": train_probe_series,
        "train_run_wall_s": train_run_wall_s,
        "epoch_walls": epoch_walls,
        "peak_rss_bytes": {"value": rss_bytes, "unit": "bytes"},
        "peak_vram_bytes": {"value": peak_vram, "unit": "bytes"},
    }
    return {
        "tool": "torch_finetune_run",
        "dry_run": args.dry_run,
        "tiers": {"finetune_run": tier},
        # Everything that is a fact about THIS stack rather than about the run.
        "provenance": {
            **tfs.provenance(device, fast_path_globals),
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizers_version": _version("tokenizers"),
            "safetensors_version": _version("safetensors"),
            "numpy_version": _version("numpy"),
            "attn_requested": args.attn,
            "attn_implementation": resolved_attn,
            "adamw_foreach": args.adamw_foreach,
            "backbone_dropout_overridden": dropout_overridden,
            "trainable_tensor_count": len(named),
            "peak_rss_source": rss_source,
            "torch_allocator": torch_allocator,
            "train_loss_curve": train_loss_curve,
            "val_loss_curve": val_loss_curve,
        },
    }


def _version(package: str):
    from importlib import metadata

    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


# ─── dry run ─────────────────────────────────────────────────────────────────

DRY_RUN_TRAIN_ROWS = 11
DRY_RUN_HELDOUT_ROWS = 4
DRY_RUN_WORDS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa")


def write_dry_run_inputs(root: str) -> dict:
    """A tiny random ModernBERT (the step twin's donor), a word-level
    tokenizer over a ten-word vocabulary, and synthetic pairs — enough to
    drive every pass of `run` on a CPU in seconds."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    model_dir = tfs.build_dry_run_checkpoint(os.path.join(root, "model"))
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    vocab.update({word: 4 + i for i, word in enumerate(DRY_RUN_WORDS)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(os.path.join(model_dir, "tokenizer.json"))

    def row(k: int) -> dict:
        words = [DRY_RUN_WORDS[(k + j) % len(DRY_RUN_WORDS)] for j in range(3 + k % 9)]
        return {
            "anchor_id": f"a{k}",
            "anchor_text": " ".join(words),
            "positive_id": f"p{k}",
            "positive_text": " ".join(reversed(words)),
            "negative_id": f"n{k}",
            "negative_text": DRY_RUN_WORDS[(k + 5) % len(DRY_RUN_WORDS)],
        }

    paths = {name: os.path.join(root, name) for name in ("train.jsonl", "heldout.jsonl", "heldout_ids.txt")}
    train = [row(k) for k in range(DRY_RUN_TRAIN_ROWS)]
    heldout = [row(100 + k) for k in range(DRY_RUN_HELDOUT_ROWS)]
    with open(paths["train.jsonl"], "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in train)
    with open(paths["heldout.jsonl"], "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in heldout)
    with open(paths["heldout_ids.txt"], "w") as fh:
        fh.writelines(f"{r['anchor_id']}\t{r['positive_id']}\t{r['negative_id']}\n" for r in heldout)
    return {
        "model_dir": model_dir,
        "train_jsonl": paths["train.jsonl"],
        "heldout_jsonl": paths["heldout.jsonl"],
        "heldout_ids": paths["heldout_ids.txt"],
        "work_dir": root,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    """`jammi-bench finetune-run`'s flags, name for name and default for
    default, minus the jammi-only ones (`--arm`, kernel and mutant flags) and
    plus the torch-only ones (`--attn`, `--adamw-foreach`, `--initial-adapter`,
    `--dry-run`). A value this twin cannot reproduce is not a choice."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir")
    p.add_argument("--train-jsonl")
    p.add_argument("--heldout-ids")
    p.add_argument("--heldout-jsonl")
    p.add_argument("--work-dir")
    p.add_argument("--task", choices=["text_embedding"], default="text_embedding")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--eval-cadence", type=int, default=DEFAULT_EVAL_CADENCE)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument(
        "--zero-lr-control",
        action="store_true",
        help="run this job as its own negative control: every optimizer step applied at learning rate zero",
    )
    p.add_argument("--schedule", choices=sorted(SCHEDULE_IDENTITY), default="constant")
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--early-stopping-patience", type=int, default=NEVER_STOPS_PATIENCE)
    p.add_argument("--early-stopping-metric", choices=["train_loss", "val_loss"], default="val_loss")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--objective", choices=["mnrl"], default="mnrl")
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-init", choices=["zeros_b", "gaussian", "peft"], default="zeros_b")
    p.add_argument("--target-modules", type=_csv, default=["Wqkv", "Wo", "Wi"])
    p.add_argument("--layers-to-transform", type=lambda v: [int(i) for i in _csv(v)] or None, default=None)
    p.add_argument("--backbone-dtype", choices=["f32", "bf16"], default="f32")
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--cuda", type=int, default=None)
    p.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    p.add_argument(
        "--width",
        choices=["bucketed", "natural"],
        default="bucketed",
        help="training-batch padding: jammi's bucket ladder (the semantic twin) or the batch's longest row "
        "(the practical bar for speed and space)",
    )
    p.add_argument("--adamw-foreach", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--initial-adapter",
        default=None,
        help="the initial_adapter.safetensors a jammi leg of the same seed wrote into its --work-dir",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.dry_run:
        missing = [f for f in ("model_dir", "train_jsonl", "heldout_ids", "heldout_jsonl", "work_dir") if not getattr(args, f)]
        if missing:
            p.error("required unless --dry-run: " + ", ".join("--" + f.replace("_", "-") for f in missing))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    scratch = tempfile.TemporaryDirectory() if args.dry_run else contextlib.nullcontext()
    try:
        with scratch as root:
            if args.dry_run:
                overrides = write_dry_run_inputs(root)
                overrides.update(
                    cuda=None, epochs=2, batch=2, max_seq_length=8, lora_rank=2, lora_alpha=4.0,
                    lora_dropout=0.0, lora_init="peft", initial_adapter=None, grad_accum=2, warmup_steps=1,
                    validation_fraction=0.2,
                )
                args = argparse.Namespace(**{**vars(args), **overrides})
            report = run(args)
    except Refusal as refusal:
        print(f"torch_finetune_run: refused — {refusal}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
