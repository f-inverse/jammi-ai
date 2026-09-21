#!/usr/bin/env python3
"""SHARED per-field canonicalizer table for every jammi-vs-torch cross-
producer IDENTITY check in this directory — extracted so
`compare_grad_oracle.py` (the gradient-oracle comparator) and `ab_merge.py`
(the finetune-step A/B merge stage) apply the SAME representational-gap
narrowing to the SAME field spellings, rather than each carrying its own,
independently-drifting copy.

WHY THIS MODULE EXISTS: both comparators face the identical class of
problem — two INDEPENDENT producers (jammi's Rust CLI, torch's Python
reference script) emit the same semantic identity field (`backbone_dtype`,
`target_modules`) through different serialization conventions (torch's bare
CLI-flag spelling `fp32` vs jammi's canonical `f32`; CLI-argument-order-
dependent list order vs an unordered `should_apply_lora` consumer). A
canonicalizer here narrows ONLY that representational gap, never widens what
counts as a match — the same non-widening discipline both comparators'
own test suites pin per field.

Stdlib-only, no jammi-bench/torch/numpy dependency — importable from either
comparator with zero extra setup, and directly by
`test_compare_grad_oracle.py`/`test_ab_merge.py`.
"""

from __future__ import annotations

import itertools
import math
import struct

# jammi's OWN CLI/interchange vocabulary
# (`crates/jammi-bench/src/main.rs`'s `--backbone-dtype` choices,
# `grad_oracle.rs`'s `format!("{:?}", ComputePrecision::F32).to_lowercase()`)
# is `f32`/`f16`/`bf16` -- the CANONICAL spelling both comparators normalize
# to, since it is the spelling used across every jammi entry point AND the
# weight-interchange file's own naming convention. Both
# `torch_grad_oracle.py`'s `run()` and `torch_finetune_step.py`'s `report`
# emit jammi's canonical spelling directly; this map covers a dump from any
# other producer that carries torch's bare CLI-flag spelling `fp32`.
LEGACY_BACKBONE_DTYPE_SPELLINGS = {
    "fp32": "f32",
}


def normalize_backbone_dtype(value):
    """Map a legacy `backbone_dtype` spelling to jammi's canonical one;
    anything not a recognized legacy spelling (including an already-
    canonical value, or a non-string/`None`) passes through UNCHANGED --
    this function only ever narrows two spellings of the SAME precision
    together, never widens what counts as a match.
    """
    if not isinstance(value, str):
        return value
    return LEGACY_BACKBONE_DTYPE_SPELLINGS.get(value, value)


def normalize_target_modules(value):
    """Canonicalize a `target_modules` run-identity value to an
    ORDER-INDEPENDENT representation before comparison.

    WHY ORDER IS NOT SEMANTICALLY MEANINGFUL: jammi's OWN consumer of this
    list, `jammi_lora::config::should_apply_lora`
    (`crates/jammi-lora/src/config.rs`), tests membership via
    `target_modules.iter().any(|t| module_name == t ||
    module_name.ends_with(t))` -- an UNORDERED existence check over the
    whole slice, never indexed by position. Every producer in this repo
    builds this field by literally splitting the operator's
    `--target-modules` CLI string on commas, preserving whatever order the
    operator typed -- so two operators who pass the SAME SET in a different
    order (a plausible, innocent difference: nobody agrees in advance on a
    comma-order convention for what is semantically a set) produce
    representationally different but semantically IDENTICAL
    `target_modules` values.

    Narrows ONLY order, never MEMBERSHIP: returns `tuple(sorted(value))`
    when `value` is a list -- duplicates are preserved and still compared
    (`["Wqkv", "Wqkv"]` vs `["Wqkv"]` remain different after sorting, since
    sorting a 2-element list does not collapse it to a 1-element one).
    Passes anything else (a non-list, `None`) through UNCHANGED, mirroring
    `normalize_backbone_dtype`'s own narrowing discipline.
    """
    if not isinstance(value, list):
        return value
    return tuple(sorted(value))


class _NotRepresentableAsF32:
    """Sentinel `_round_trip_f32` returns instead of a canonicalized
    `float` when the raw input cannot be trusted to describe a legitimate
    premise value in the engine's own `f32` storage. A bare
    `struct.pack('<f', ...)` raises `OverflowError` for any FINITE value
    outside `f32`'s representable range (e.g. `1e40`), which would crash the
    WHOLE merge over one malformed field rather than refuse one config. This
    sentinel turns that into an ORDINARY, catchable REFUSAL instead:
    `canonicalize_identity_field` returns it like any other value,
    `leg_premise_violations`/`generic_leg_premise_violations` compare it
    exactly like a real float (`va != vb`), and its own `__repr__` names the
    reason directly in the printed violation ("field X differs:
    jammi=<not representable as the engine's f32: 1e+40> torch=0.05").

    Covers BOTH the finite-but-out-of-range case (`OverflowError`) and the
    non-finite cases (`inf`/`-inf`/`nan`) — the latter pack into `f32`
    WITHOUT raising (IEEE-754 represents all three natively), but neither
    is a value EITHER real producer would ever validate a `lora_dropout`/
    `max_grad_norm` CLI argument to (`validate_max_grad_norm`'s own "must
    be finite and > 0.0" check, mirrored on torch's `parse_args`) — a
    report carrying one is already describing a premise this comparator
    cannot trust, not merely one it must round differently. `-0.0` is
    deliberately NOT covered (it is finite, in-range, round-trips cleanly,
    and Python's own `-0.0 == 0.0` already holds after the round-trip) —
    see this class's own test suite's negative control.

    `__eq__` ALWAYS returns `False` — including against another
    `_NotRepresentableAsF32`, even one wrapping the IDENTICAL raw value:
    neither side of a "cannot be represented" pair can be confirmed to
    describe the SAME premise, so two malformed inputs must refuse each
    other exactly as loudly as one malformed input against one clean
    value — never let two garbage values silently "cancel out" into an
    accidental match.

    `__repr__` is INSTANCE-UNIQUE
    (a per-instance sequence number folded in), not merely a function of
    `raw`. This is not cosmetic: `finetune_run_leg_identity_violations`
    (`ab_merge.py`, the cross-seed identity check) groups displayed values
    by `repr(display)` — a plain string KEY, never by `==` — precisely
    because a `dict` needs a hashable key and `_NotRepresentableAsF32`
    itself is deliberately not usefully hashable-by-value (see `__hash__`
    below). Two DIFFERENT `_NotRepresentableAsF32` instances that happened
    to wrap the SAME `raw` (e.g. two legs both reporting `1e40`, or both
    `nan`) would, with a `raw`-only `__repr__`, produce the IDENTICAL
    `repr()` string and collapse into ONE dict bucket — silently
    "agreeing" by string coincidence, the exact same class of accidental
    match `__eq__`'s own doc above forbids, just reached through a
    different (repr-keyed, not eq-keyed) grouping mechanism a SECOND
    caller happens to use. The sequence number makes that collision
    structurally impossible: no two instances, constructed at different
    times, can ever share a `repr()`.
    """

    __slots__ = ("raw", "_seq")

    _next_seq = itertools.count()

    def __init__(self, raw):
        self.raw = raw
        self._seq = next(_NotRepresentableAsF32._next_seq)

    def __repr__(self):
        return f"<not representable as the engine's f32 (#{self._seq}): {self.raw!r}>"

    def __eq__(self, other):
        return False

    def __hash__(self):
        return id(self)


def _round_trip_f32(value):
    """Round-trip a numeric `value` through IEEE-754 binary32 (the hardware
    default round-half-to-even), stdlib-only (`struct`, no numpy
    dependency — this module's own "Stdlib-only" doc stays true). Returns
    `value` UNCHANGED for anything that is not a real number (`None`, a
    `bool` -- `isinstance(True, int)` is `True` in Python, so `bool` is
    excluded explicitly -- a string, a list): this function only ever
    narrows the SAME numeric value's own two representations together,
    never widens what counts as a match for a non-numeric input it was
    never meant to touch. A non-finite or out-of-`f32`-range REAL number
    returns a `_NotRepresentableAsF32` sentinel instead — see that class's
    own doc for why (a REFUSAL, never a crash, never a silent pass-through).
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return _NotRepresentableAsF32(value)
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return _NotRepresentableAsF32(value)


def normalize_f32_stored_field(value):
    """Canonicalize `lora_dropout`/`max_grad_norm` — the TWO knobs jammi's
    own CLI stores at `f32` (`FinetuneStepParams::lora_dropout: f32`,
    `FinetuneStepParams::max_grad_norm: Option<f32>`), while torch's
    argparse/JSON round-trip for the SAME two `--lora-dropout`/
    `--max-grad-norm` flags carries the operator's literal at full `f64`
    precision. Null-safe: `None` (max_grad_norm's own "clip OFF" value,
    `identity_fields.FINETUNE_NULL_IS_A_VALUE_FIELDS`) passes through
    UNCHANGED, never coerced to a number.

    WHY ONLY THESE TWO, never "all floats": `lora_alpha` and `margin` are
    `f64` end-to-end on BOTH producers (jammi's `FinetuneStepParams::
    lora_alpha: f64`; torch's own `--lora-alpha`/`--margin` floats,
    unmodified before reaching the report) -- there is no representational
    gap between the two sides to narrow for either field, and adding a
    canonicalizer for them would WIDEN what counts as a match (silently
    accepting an f32-rounded value on ONE side against a genuine f64 value
    on the other, when neither producer ever produces that gap) --
    exactly the "narrows only the representational gap, never widens what
    counts as a match" discipline this module's own doc states.

    Round-trip, not a tolerance check: `0.05` stored as `f32` and read back
    as `f64` is `0.05000000074505806`, not `0.05` -- comparing the two raw
    `f64` values directly would refuse a config the operator asked for
    IDENTICALLY on both sides, purely because one producer's storage type
    rounds the input on the way in. Rounding BOTH sides through the SAME
    `f32` boundary (struct-packed here; a real jammi process does the
    identical rounding in hardware when the CLI float is stored into the
    `f32` field) makes the two values compare equal again without loosening
    the comparison for a genuine divergence (`0.05` vs `0.06` still
    differs after the SAME round-trip is applied to both).
    """
    return _round_trip_f32(value)


# THE finetune-step identity set — the ONE declaration every consumer
# derives from; no consumer keeps its own copy. The rule for membership: a
# field is IDENTITY when two legs differing in it are computing a DIFFERENT
# step, so their throughput/loss numbers are not comparable at all — any
# knob that changes what the step computes belongs here (a clip-ON leg
# against a clip-OFF leg, or an eager-attention leg against a fused one,
# must never merge to a PASS). It is NOT a place for provenance (recorded,
# never compared — torch's raw `attn_implementation` string, jammi's
# dispatch counters) or measurement.
#
# Both producers MUST emit every name here, at the level `ab_merge.py`'s
# `leg_identity_fields` reads it from:
#   * jammi — `crates/jammi-bench/src/report.rs`'s `FinetuneStepTier`
#     (`report["tiers"]["finetune_step"][field]`); that struct's own
#     `finetune_step_tier_emits_every_shared_identity_field` test reads THIS
#     tuple back out of THIS file and refuses a field it does not serialize.
#   * torch — `crates/jammi-bench/reference/torch_finetune_step.py`'s report
#     literal (`report["finetune_step"][field]`, or `report["args"][field]`
#     for the `ab_merge._TORCH_ARGS_LEVEL_FIELDS` trio); `test_ab_merge.py`'s
#     `SharedIdentityDeclarationTests` scans that literal for every name.
# `ab_merge.leg_premise_violations` refuses GENERICALLY on any member that
# is missing from either side or differs after `canonicalize_identity_field`
# — never through a per-field `if`.
FINETUNE_IDENTITY_FIELDS = (
    "seed",
    "batch",
    "seq",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "margin",
    "target_modules",
    "batched_forward",
    "backbone_dtype",
    "steps_measured",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    # The gradient clip's on/off + bound for this row: `null` (clip OFF)
    # or the positive
    # finite `max_norm` the PRODUCTION `clip_gradients` ran with (jammi:
    # `--max-grad-norm`; torch: `--max-grad-norm` →
    # `torch.nn.utils.clip_grad_norm_`). Two legs differing here run a
    # different step (one pays the `4n + 4` device ops and rescales its
    # gradients, the other does not), so the row's ratio is meaningless.
    # `null` is a VALUE here, not "missing" — see
    # `FINETUNE_NULL_IS_A_VALUE_FIELDS` below.
    "max_grad_norm",
    # Which attention REFERENCE CLASS the leg was ASKED to run — `"eager"`
    # (the materialised-scores softmax path; the semantic reference) or
    # `"fused"` (one fused attention kernel; the throughput reference).
    #   * torch: the class of the RESOLVED `_attn_implementation` (`eager`
    #     → "eager"; `sdpa` / flash / flex → "fused";
    #     `torch_finetune_step.py`'s `attention_arm_of`).
    #   * jammi: jammi has no `--attn` lever — `JAMMI_KERNELS_DISABLE` is the
    #     lever — so the value is the OPERATOR'S REQUEST: "eager" iff an
    #     attention base (`attention_block`, `attention_block_flash`, or the
    #     `all` wildcard) is in the resolved `kernels_disabled_requested`,
    #     else "fused" (`finetune_step.rs`'s `attention_arm`). It is
    #     deliberately NOT read off the `attention_block_*_dispatches`
    #     counters: those read eager whenever the fused predicate DECLINES
    #     BY DOMAIN (`head_dim != 64`, `seq > 4096`, dtype/contiguity/mask
    #     arms — documented as by-design in `report.rs`), so a legitimate
    #     jammi-fused leg on a non-64-head_dim checkpoint would read
    #     "eager", mismatch torch's "fused", and INVALIDate the row over a
    #     MEASUREMENT. Whether the fused arm actually ran lives (`ab_merge.fused_proof` + the counters); an identity
    #     field describes what was asked for.
    # This is the "two references, never mixed" rule (eager ↔ eager is the
    # semantic reference, sdpa ↔ fused the throughput one) made a CHECKED
    # premise instead of a leg-naming convention. A leg that is only a
    # FALLBACK (torch-sdpa OOM → torch-eager; jammi-fused failed →
    # jammi-eager) is "not comparable" — `ab_merge.build_report` skips the
    # identity check for that row and records why, rather than refusing a
    # documented non-gating outcome as an identity mismatch. The RAW value
    # each side ran with stays in provenance (torch: `attn_requested` /
    # `attn_implementation`; jammi: `kernels_disabled_requested` + the
    # `attention_block_*_dispatches` counters), recorded, never compared.
    "attention_arm",
    # Warmup iterations executed before the measured ones. Identity because
    # it changes what `clip_invocations` (pre-step + warmup + measured)
    # counts — two legs at different warmups are not comparable on that
    # counted fact. jammi: `FinetuneStepTier::warmup`; torch: `args.warmup`
    # (an `ab_merge._TORCH_ARGS_LEVEL_FIELDS` member).
    "warmup",
    "row_lengths",
)

# THE encode-step identity set — mirrors
# `crates/jammi-bench/src/report.rs`'s `EncodeStepTier::IDENTITY_FIELDS`
# EXACTLY, in the Rust const's own order (that const's own doc names this
# file's `ENCODE_IDENTITY_FIELDS` as its pinned mirror;
# `test_identity_fields_subset.py` holds the two sets equal and fails on a
# drift on either side).
#
# UNLIKE `FINETUNE_IDENTITY_FIELDS` above, this tuple is NOT a subset of a
# larger Rust const that also folds in provenance/dispatch facts —
# `EncodeStepTier` keeps its provenance (`partitions`, `model_dir`,
# `device_name`, `kernels_disabled_requested`, `kernels_disabled_fired`,
# `flash_compiled`, `build_features`, `chunk_size`, `attention_arm`) in its
# OWN, entirely DISJOINT `PROVENANCE_FIELDS` const. `ENCODE_IDENTITY_FIELDS` is
# therefore compared for SET EQUALITY against
# `EncodeStepTier::IDENTITY_FIELDS`, never a subset check — see
# `test_identity_fields_subset.py`'s own `EncodeStepIdentityFieldsTests` for
# the mechanical assertion.
#
# `attention_arm` is FORBIDDEN here: a dispatched arm is a POST-HOC fact, never
# knowable before compute, so it can never be a memoization key; it is also
# constant on this eval-only surface by construction (fused attention arms are
# training-only), which would make it a false determinant even if it were
# admitted. `partitions` is absent for the opposite reason: it is this
# surface's INDEPENDENT VARIABLE — the engine contracts it never to change the
# written bytes — and a comparator that paired legs only when it agreed could
# never set `partitions = 1` beside `partitions = N` (`EncodeStepTier`'s own
# doc; `encode_ab.py` checks each leg's recorded value against its label
# instead). This module carries no `ENCODE_PROVENANCE_FIELDS` tuple — the Rust
# `PROVENANCE_FIELDS` const is extracted directly by the test suite's regex
# scan rather than duplicated into a second Python list.
ENCODE_IDENTITY_FIELDS = (
    "seed",
    "rows",
    "batch_size",
    "corpus",
    "max_sequence_length",
    "compute_precision",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "checkpoint_tokenizer_sha256",
    "pooling",
    "normalize",
    "warmup",
    "iters_measured",
    "checkpoint_pooling_sha256",
    "device_requested",
)

# `checkpoint_pooling_sha256` is `Nullable::NullMeans("no 1_Pooling/config.json
# in this model dir")` on the Rust const: `null` there is the stated premise
# (the engine's mean-pooling fallback served), compared as a value, never
# folded into "this leg could not state its premise".
ENCODE_NULL_IS_A_VALUE_FIELDS = frozenset({"checkpoint_pooling_sha256"})

# The fields a jammi `encode-step` leg and the PyTorch reference
# (`crates/jammi-bench/reference/torch_encode.py`, whose `IDENTITY_FIELDS` this
# mirrors — `test_identity_fields_subset.py` holds the two equal) must agree on
# before their numbers are one comparison: `ENCODE_IDENTITY_FIELDS` minus
# `seed`. The reference reads the corpus from the file the jammi leg served
# and never generates one, so it has no seed to state; the corpus's own bytes
# (`corpus[*].corpus_sha256`) are the stronger anchor both sides carry.
ENCODE_TWIN_IDENTITY_FIELDS = tuple(f for f in ENCODE_IDENTITY_FIELDS if f != "seed")


# Identity fields for which a JSON `null` is a legitimate VALUE (compared as
# such, `null == null` matches) rather than the "present-but-unverifiable"
# state `ab_merge.leg_identity_fields` otherwise folds into MISSING
# (`serde_json` writes a NaN `f64` as `null`, so a null numeric identity field is normally a producer that could not state its
# premise). `max_grad_norm` is the exception BY CONSTRUCTION: both producers
# validate a supplied value as finite and `> 0.0` before running (jammi's
# `validate_max_grad_norm`, torch's `parse_args` check), so NaN can never
# reach the report — `null` there means exactly one thing, clip OFF. A key
# that is ABSENT entirely is still MISSING for these fields too (a producer
# that does not emit the field cannot state its premise).
FINETUNE_NULL_IS_A_VALUE_FIELDS = frozenset({"max_grad_norm"})


# Per-field canonicalizer table: every identity field NOT listed here is
# compared with NO canonicalization (the JSON-decoded value as-is), because
# it carries no known cross-producer representational gap — see
# `compare_grad_oracle.py`'s `RUN_IDENTITY_FIELDS` doc and
# `FINETUNE_IDENTITY_FIELDS` above for the full field-by-field determinant
# table each comparator maintains for ITS OWN field set (this table is
# shared machinery, not a duplicate of either). `lora_alpha`/`margin`
# deliberately have NO canonicalizer despite being numeric siblings of
# `lora_dropout`/`max_grad_norm`: both are `f64` end-to-end on BOTH
# producers (see `normalize_f32_stored_field`'s own doc for why exactly
# these two, never "all floats"). `attention_arm` also deliberately has NO
# canonicalizer: both producers emit the same vocabulary directly
# (`"eager"`/`"fused"`), and a canonicalizer that mapped torch's raw
# `"sdpa"` onto jammi's `"fused"` here would be WIDENING what counts as a
# match inside the comparator rather than each producer stating its own
# class honestly.
IDENTITY_FIELD_CANONICALIZERS = {
    "backbone_dtype": normalize_backbone_dtype,
    "target_modules": normalize_target_modules,
    "lora_dropout": normalize_f32_stored_field,
    "max_grad_norm": normalize_f32_stored_field,
}


# THE finetune-run identity set — mirrors
# `crates/jammi-bench/src/report.rs`'s `FinetuneRunTier::IDENTITY_FIELDS`
# EXACTLY, verbatim in the SAME order that const's own source lists them
# (order is not semantically load-bearing for a set-equality check, but
# keeping it identical makes a side-by-side diff against the Rust const
# trivial for a human reviewer). Like `ENCODE_IDENTITY_FIELDS` above and
# UNLIKE `FINETUNE_IDENTITY_FIELDS`'s superset-folding
# shape, `FinetuneRunTier` keeps its provenance (`arm`, `device_name`,
# `kernels_disabled_requested`, `kernels_disabled_fired`, `flash_compiled`,
# `build_features`, `attention_arm`, `split_rule`, `batched_forward`,
# `steps_measured`) in its OWN, entirely DISJOINT `PROVENANCE_FIELDS` const —
# see that struct's own doc for why `arm`/`attention_arm` are provenance here
# rather than identity (the CALLER'S request / the process-resolved
# reference class, neither a determinant of what the held-out loss itself
# computes). `FINETUNE_RUN_IDENTITY_FIELDS` is therefore compared for SET
# EQUALITY against `FinetuneRunTier::IDENTITY_FIELDS`, never a subset check
# — see `test_identity_fields_subset.py`'s own
# `FinetuneRunIdentityFieldsSubsetTests` for the mechanical assertion. This
# module carries no `FINETUNE_RUN_PROVENANCE_FIELDS` tuple, following
# `ENCODE_IDENTITY_FIELDS`'s own precedent: the Rust `PROVENANCE_FIELDS`
# const is extracted directly by the test suite's regex scan rather than
# duplicated into a second Python list nobody would keep in sync.
#
# `layers_to_transform` (`--layers-to-transform`'s own resolved value) is
# IDENTITY (not provenance) for the exact reason `target_modules` itself is:
# a `Some([..])` leg wraps a DIFFERENT set of linears than a `None` leg at
# the identical `target_modules`, so two legs agreeing on every other field
# but disagreeing here are not comparable. `Nullable::NullMeans("no
# restriction -- every layer matching target_modules gets a LoRA adapter")`
# on the Rust const -- `None` IS a meaningful, distinct value (every layer),
# never "unknown"/"not yet measured" -- so `layers_to_transform` is also a
# `FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS` member, mirrored below.
#
# Membership notes for fields a reader might expect elsewhere:
#   * `heldout_pairs_sha256` — sha256 of the `--heldout-jsonl` file's own
#     bytes, MEASURED at load; the held-out fixture's TEXT is a total
#     determinant of every per-example loss `d_i`, which the id ORDER alone
#     (`heldout_ids_sha256`) does not anchor.
#   * `train_pairs_file_sha256` — the `--train-jsonl` file's own raw bytes,
#     measured off the file this run actually read. Deliberately NOT named
#     `dataset_sha256`: the committed fixture manifest's own `dataset_sha256`
#     is a DIFFERENT quantity (a Merkle digest over per-pair content hashes,
#     built off-process), and the same spelling for both would anchor
#     neither.
#   * `split_rule`/`batched_forward` are provenance and there is no
#     `split_seed` — none of the three can vary independently of an
#     admitted field or a build-time constant: `split_rule` is a hardcoded
#     literal, `batched_forward` is always `true`, and `split()` takes no
#     seed of its own beyond `seed`. `heldout_batch_partition_sha256` is
#     KEPT despite also being a pure function of identity inputs (held-out
#     ids + `batch`) — see `FinetuneRunTier`'s own doc for why it earns its
#     slot (a genuine cross-arm equality guard against the partitioning
#     ALGORITHM diverging, not a redundant echo of inputs).
#   * `steps_measured` is provenance — a MEASURED OUTCOME of running
#     (cumulative optimizer steps), not a premise the run was configured
#     under.
FINETUNE_RUN_IDENTITY_FIELDS = (
    # FinetuneStepTier's fields minus attention_arm, `batched_forward`
    # and `steps_measured` — same order as the Rust const's own leading
    # block.
    "seed",
    # `--task`, the TOWER selector (`text_embedding` /
    # `image_embedding` / `audio_embedding`). Same position as the Rust
    # const's own listing, immediately after `seed`.
    "task",
    "batch",
    "seq",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    # `--lora-init` (`zeros_b` / `gaussian`). Same
    # position as the Rust const's own listing, immediately after
    # `lora_dropout`.
    "lora_init",
    "margin",
    "target_modules",
    # See the doc above this tuple -- same position as
    # the Rust const's own listing, immediately after `target_modules`.
    "layers_to_transform",
    "backbone_dtype",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "max_grad_norm",
    "warmup",
    "row_lengths",
    # The full-run fields a single step does not have.
    "epochs",
    "lr",
    "schedule",
    "warmup_steps",
    "weight_decay",
    "grad_accum",
    "validation_fraction",
    "train_pairs_file_sha256",
    # The media corpus CONTENT digests, in the Rust
    # const's own positions (each immediately after the MANIFEST digest it
    # completes).
    "train_media_sha256",
    "heldout_ids_sha256",
    "heldout_pairs_sha256",
    "heldout_media_sha256",
    "heldout_batch_partition_sha256",
    "embedding_loss",
    "temperature",
    "matryoshka_dims",
    "early_stopping_patience",
    "early_stopping_metric",
    "eval_cadence",
)

# Identity fields for which a JSON `null` is a legitimate VALUE — mirrors
# `FINETUNE_NULL_IS_A_VALUE_FIELDS`'s own doctrine, but for
# `FinetuneRunTier::IDENTITY_FIELDS`'s own `Nullable::NullMeans` entries
# (read verbatim off that const, see `report.rs`):
#   * `margin`        — NullMeans("objective is mnrl")
#   * `temperature`   — NullMeans("objective is triplet")
#   * `max_grad_norm` — NullMeans("no clip")
#   * `warmup`        — NullMeans("a full run has no discard-before-timing
#                        convention; see warmup_steps")
#   * `row_lengths`   — NullMeans("real text is variable-length; no single
#                        fixed row_lengths applies across a whole
#                        multi-epoch run")
#   * `layers_to_transform` — NullMeans("no restriction -- every layer
#                        matching target_modules gets a LoRA adapter") --
#                        see FINETUNE_RUN_IDENTITY_FIELDS's own doc above;
#                        `None` is the meaningful "all layers" value, never
#                        "unknown".
#   * `train_media_sha256`/`heldout_media_sha256` — NullMeans("text task —
#                        the {train,held-out} corpus content IS the
#                        manifest, digested by
#                        {train_pairs_file_sha256,heldout_pairs_sha256}").
#                        `None` is the meaningful "this
#                        leg has no media corpus" value on a text task,
#                        never "not measured": on a MEDIA task the manifest
#                        digests name PATHS only, so the content digest is
#                        the field that makes two media legs comparable at
#                        all.
# Every OTHER `FINETUNE_RUN_IDENTITY_FIELDS` member is `Nullable::NonNull`
# on the Rust const, so a present `null` there still folds to MISSING (the
# same "cannot verify this premise determinant" state `leg_identity_fields`
# already applies to `FINETUNE_IDENTITY_FIELDS`).
FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS = frozenset(
    {
        "margin",
        "temperature",
        "max_grad_norm",
        "warmup",
        "row_lengths",
        "layers_to_transform",
        "train_media_sha256",
        "heldout_media_sha256",
    }
)


# THE gpu-inference identity set — mirrors
# `crates/jammi-bench/src/report.rs`'s `GpuInferenceTier::IDENTITY_FIELDS`
# EXACTLY, in the SAME order that const's own source lists them.
# `test_identity_fields_subset.py`'s own `GpuInferenceIdentityFieldsSubsetTests`
# pins the cardinality on BOTH sides and fails on a drift on either one.
#
# `row_count` is identity because `p50_ms` moves LINEARLY with it (two legs
# at a different row count could manufacture a 2x "win"); `iters` because a
# differently-sized measured sample is not the same measurement; and
# `corpus_sha256` (a sha256 content hash over every committed sentence plus
# `corpus_seed`/`row_count`) closes the gap those two SCALARS alone cannot:
# a change that merely rewords a committed sentence, holding both scalars
# fixed, moves neither one.
#
# UNLIKE `FINETUNE_IDENTITY_FIELDS`, and LIKE `ENCODE_IDENTITY_FIELDS`, this
# tuple is NOT a subset of a larger Rust const that also folds in
# provenance/dispatch facts -- `GpuInferenceTier` keeps its provenance
# (`device_name`, `kernels_disabled_requested`, `flash_compiled`,
# `build_features`) in its OWN, entirely DISJOINT `PROVENANCE_FIELDS` const
# (the SAME disjoint shape `ENCODE_IDENTITY_FIELDS` follows, never
# `FINETUNE_IDENTITY_FIELDS`'s superset-folding one). `GPU_INFERENCE_IDENTITY_FIELDS`
# is therefore compared for SET EQUALITY against `GpuInferenceTier::IDENTITY_FIELDS`,
# never a subset check.
#
# `compute_precision` admits only the EMBED bundle's resolved precision to
# identity (never a second field for the classifier bundle) -- this tier
# states ONE pre-registered primary A/B endpoint (embed `p50_ms`, see
# `gpu_inference_ab.py`'s own module doc), and an identity field for a
# workload nothing gates would be a false determinant. `GpuInferenceTier`'s
# own doc has the full rationale.
GPU_INFERENCE_IDENTITY_FIELDS = (
    "corpus_seed",
    "row_count",
    "warmup",
    "iters",
    "corpus_sha256",
    "compute_precision",
    "embed_checkpoint_config_sha256",
    "embed_checkpoint_weights_sha256",
    "embed_checkpoint_tokenizer_sha256",
    "infer_checkpoint_config_sha256",
    "infer_checkpoint_weights_sha256",
    "infer_checkpoint_tokenizer_sha256",
)


def canonicalize_identity_field(field, value):
    """Apply `field`'s registered canonicalizer (see
    `IDENTITY_FIELD_CANONICALIZERS`'s own table), or return `value`
    unchanged if none is registered. The SINGLE dispatch point every
    identity-field comparison in this directory calls -- closing a
    representational gap for another field means registering one function in
    the table above, not writing a new `if field == ...` branch inline at
    each call site.
    """
    fn = IDENTITY_FIELD_CANONICALIZERS.get(field)
    return value if fn is None else fn(value)
