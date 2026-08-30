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
`test_compare_grad_oracle.py`/`test_ab_merge.py`/a future shared lattice
test.
"""

from __future__ import annotations

import itertools
import math
import struct

# B1 audit finding on PR #372: jammi's OWN CLI/interchange vocabulary
# (`crates/jammi-bench/src/main.rs`'s `--backbone-dtype` choices,
# `grad_oracle.rs`'s `format!("{:?}", ComputePrecision::F32).to_lowercase()`)
# is `f32`/`f16`/`bf16` -- picked as the CANONICAL spelling both comparators
# normalize to, since it is already the spelling used across every jammi
# entry point AND the weight-interchange file's own naming convention. Both
# `torch_grad_oracle.py`'s `run()` and `torch_finetune_step.py`'s `report`
# emit jammi's canonical spelling directly for a NEW dump; this map exists
# for LEGACY dumps only -- an older producer, or any external one, that
# still carries torch's bare CLI-flag spelling `fp32`.
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
    premise value in the engine's own `f32` storage — advisory (adversarial
    audit): an EARLIER version of this function let `struct.pack('<f',
    ...)` raise a bare `OverflowError` straight out of the comparator for
    any FINITE value outside `f32`'s representable range (e.g. `1e40`),
    crashing the WHOLE merge over one malformed field the same class of
    bug F1 fixed for `bar_ratio_classification` — never caught per-config,
    never even a violation, just a hard crash. This sentinel turns that
    into an ORDINARY, catchable REFUSAL instead: `canonicalize_identity_field`
    returns it like any other value, `leg_premise_violations`/
    `generic_leg_premise_violations` compare it exactly like a real float
    (`va != vb`), and its own `__repr__` names the reason directly in the
    printed violation ("field X differs: jammi=<not representable as the
    engine's f32: 1e+40> torch=0.05").

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

    Advisory (ii), round-2 adversarial audit: `__repr__` is INSTANCE-UNIQUE
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
    on the other, when today neither producer ever produces that gap) --
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
# derives from (PR #381 audit B1: `ab_merge.py` used to carry its own
# hand-kept 14-tuple, which lacked `max_grad_norm`, so a clip-ON jammi leg
# merged against a clip-OFF torch leg and printed PASS; the lead's class
# probe found the attention implementation absent from it for the same
# reason — "a knob that changes what the step computes is not an identity
# field" was the class, not one missing name). The rule for membership: a
# field is IDENTITY when two legs differing in it are computing a DIFFERENT
# step, so their throughput/loss numbers are not comparable at all. It is
# NOT a place for provenance (recorded, never compared — torch's raw
# `attn_implementation` string, jammi's dispatch counters) or measurement.
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
    # The gradient clip's on/off + bound for this row: `null` (clip OFF —
    # the step the tier measured before the flag existed) or the positive
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
    #     jammi-fused leg on a non-64-head_dim checkpoint would have read
    #     "eager", mismatched torch's "fused", and INVALIDated the row over a
    #     MEASUREMENT. Whether the fused arm actually ran stays where it
    #     already lives (`ab_merge.fused_proof` + the counters); an identity
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

# THE encode-step identity set (unit-62 E6, docs-ci domain) — mirrors
# `crates/jammi-bench/src/report.rs`'s `EncodeStepTier::IDENTITY_FIELDS`
# EXACTLY (that const's own doc names this file's `ENCODE_IDENTITY_FIELDS`
# as its pinned mirror; `test_identity_fields_subset.py` pins the cardinality
# on BOTH sides and REDs on a drift on either one). Grown 13 -> 15 (round-3
# audit F-5'/lead ruling): `checkpoint_pooling_sha256` (NullMeans — "no
# 1_Pooling/config.json in this model dir") and `device_requested` appended
# after the original 13, position-stable rather than re-ordered, mirroring
# the Rust const's own append order exactly.
#
# UNLIKE `FINETUNE_IDENTITY_FIELDS` above, this tuple is NOT a subset of a
# larger Rust const that also folds in provenance/dispatch facts —
# `EncodeStepTier` keeps its provenance (`device_name`,
# `kernels_disabled_requested`, `kernels_disabled_fired`, `flash_compiled`,
# `build_features`, `chunk_size`, `attention_arm`) in its OWN, entirely
# DISJOINT `PROVENANCE_FIELDS` const (unit-62 CONTRACT.md §E3 — a deliberate
# design choice, not the `FinetuneStepTier`/`REPORT_IDENTITY_FIELDS`
# superset-folding shape carried above). `ENCODE_IDENTITY_FIELDS` is
# therefore compared for SET EQUALITY against
# `EncodeStepTier::IDENTITY_FIELDS`, never a subset check — see
# `test_identity_fields_subset.py`'s own `EncodeStepIdentityFieldsTests` for
# the mechanical assertion.
#
# `attention_arm` is FORBIDDEN here (v2 reshape 3 of the unit-62 plan): a
# dispatched arm is a POST-HOC fact, never knowable before compute, so it can
# never be a memoization key (K7's own `definition_of`); it is also constant
# on this eval-only surface by construction (fused attention arms are
# training-only), which would make it a false determinant even if it were
# admitted. This module carries no `ENCODE_PROVENANCE_FIELDS` tuple — unlike
# the Rust side, this file's own existing convention has never declared a
# standalone provenance tuple for the finetune tier either (provenance is
# documented prose in `ab_merge.py`'s determinant table, never a
# machine-compared Python list here), so the encode mirror follows that same
# convention rather than inventing a new one.
ENCODE_IDENTITY_FIELDS = (
    "seed",
    "batch",
    "seq",
    "row_lengths",
    "compute_precision",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "checkpoint_tokenizer_sha256",
    "pooling",
    "normalize",
    "warmup",
    "iters_measured",
    # Round-3 audit additions (F-5'(b)/lead ruling), appended
    # position-stable rather than re-ordered into the original 13 — mirrors
    # `EncodeStepTier::IDENTITY_FIELDS`'s own append order exactly.
    "checkpoint_pooling_sha256",
    "device_requested",
)


# Identity fields for which a JSON `null` is a legitimate VALUE (compared as
# such, `null == null` matches) rather than the "present-but-unverifiable"
# state `ab_merge.leg_identity_fields` otherwise folds into MISSING (the
# round-4 PR #372 rule: `serde_json` writes a NaN `f64` as `null`, so a null
# numeric identity field is normally a producer that could not state its
# premise). `max_grad_norm` is the exception BY CONSTRUCTION: both producers
# validate a supplied value as finite and `> 0.0` before running (jammi's
# `validate_max_grad_norm`, torch's `parse_args` check), so NaN can never
# reach the report — `null` there means exactly one thing, clip OFF. A key
# that is ABSENT entirely is still MISSING for these fields too (a producer
# built before the field existed cannot state its premise).
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


# THE finetune-run identity set (unit 63, H4/H4a docs-ci mirror) — mirrors
# `crates/jammi-bench/src/report.rs`'s `FinetuneRunTier::IDENTITY_FIELDS`
# EXACTLY, verbatim in the SAME order that const's own source lists them
# (order is not semantically load-bearing for a set-equality check, but
# keeping it identical makes a side-by-side diff against the Rust const
# trivial for a human reviewer). Like `ENCODE_IDENTITY_FIELDS` above (unit
# 62's E3/E6 shape) and UNLIKE `FINETUNE_IDENTITY_FIELDS`'s superset-folding
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
# Unit-63 adversarial-audit finding 5 (identity-completeness) reshaped this
# set from its original 35 entries to 32:
#   (a) `heldout_pairs_sha256` ADDED — sha256 of the `--heldout-jsonl` file's
#       own bytes, MEASURED at load; the held-out fixture's TEXT is a total
#       determinant of every per-example loss `d_i` and was hashed nowhere
#       before this fix (only the id ORDER, via `heldout_ids_sha256`, was
#       anchored).
#   (b) `dataset_sha256` RENAMED to `train_pairs_file_sha256` — the old name
#       collided with the committed fixture manifest's OWN `dataset_sha256`
#       (a Merkle digest over per-pair content hashes, built off-process),
#       a DIFFERENT quantity under the SAME spelling, so neither anchored
#       the other. The new name states exactly what it hashes: the
#       `--train-jsonl` file's own raw bytes, measured off the file this
#       run actually read.
#   (c) `split_rule`/`batched_forward` MOVED to provenance, `split_seed`
#       DROPPED entirely — none of the three could vary independently of an
#       already-admitted field or a build-time constant: `split_rule` is a
#       hardcoded literal, `batched_forward` is always `true`, and
#       `split_seed` was a pure, literal duplicate of `seed` (`split()`
#       takes no separate seed parameter). `heldout_batch_partition_sha256`
#       is KEPT despite also being a pure function of already-identity
#       inputs (held-out ids + `batch`) — see `FinetuneRunTier`'s own doc
#       for why it earns its slot (a genuine cross-arm equality guard
#       against the partitioning ALGORITHM diverging, not a redundant echo
#       of inputs).
#   (d) `steps_measured` MOVED to provenance (advisory) — a MEASURED
#       OUTCOME of running (cumulative optimizer steps), not a premise the
#       run was configured under.
# Net: 35 − 4 (split_rule, split_seed, batched_forward, steps_measured) + 1
# (heldout_pairs_sha256) = 32.
FINETUNE_RUN_IDENTITY_FIELDS = (
    # FinetuneStepTier's 18, minus attention_arm and (finding 5(c))
    # `batched_forward`/`steps_measured` (15 entries) — carried over by
    # name, same order as the Rust const's own leading block.
    "seed",
    "batch",
    "seq",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "margin",
    "target_modules",
    "backbone_dtype",
    "checkpoint_config_sha256",
    "checkpoint_weights_sha256",
    "checkpoint_weights_size_bytes",
    "max_grad_norm",
    "warmup",
    "row_lengths",
    # New (18 entries in CONTRACT H4 v1/v2, minus `split_rule`/`split_seed`,
    # `dataset_sha256` renamed to `train_pairs_file_sha256`, plus
    # `heldout_pairs_sha256` added — finding 5(a)/(b)/(c)).
    "epochs",
    "lr",
    "schedule",
    "warmup_steps",
    "weight_decay",
    "grad_accum",
    "validation_fraction",
    "train_pairs_file_sha256",
    "heldout_ids_sha256",
    "heldout_pairs_sha256",
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
# Every OTHER `FINETUNE_RUN_IDENTITY_FIELDS` member is `Nullable::NonNull`
# on the Rust const, so a present `null` there still folds to MISSING (the
# same "cannot verify this premise determinant" state `leg_identity_fields`
# already applies to `FINETUNE_IDENTITY_FIELDS`).
FINETUNE_RUN_NULL_IS_A_VALUE_FIELDS = frozenset(
    {"margin", "temperature", "max_grad_norm", "warmup", "row_lengths"}
)


# THE gpu-inference identity set (issue #335, D4/K7-completeness) — mirrors
# `crates/jammi-bench/src/report.rs`'s `GpuInferenceTier::IDENTITY_FIELDS`
# EXACTLY, in the SAME order that const's own source lists them.
# `test_identity_fields_subset.py`'s own `GpuInferenceIdentityFieldsSubsetTests`
# pins the cardinality on BOTH sides and REDs on a drift on either one.
#
# Grown 9 -> 12 (round-1 adversarial audit B1, identity completeness):
# `row_count` (closes the "manufactured-2x attack" -- `p50_ms` moves
# LINEARLY with row_count, so two legs at a different row count are not
# comparable regardless of anything else they agree on), `iters` (was
# already EMITTED pre-#335 but never admitted to identity -- a differently-
# sized measured sample is not the same measurement), and `corpus_sha256`
# (a sha256 content hash over every committed sentence plus
# `corpus_seed`/`row_count` -- closes the residual gap those two SCALARS
# alone cannot: a PR that merely rewords a committed sentence, holding both
# scalars fixed, moves neither one).
#
# UNLIKE `FINETUNE_IDENTITY_FIELDS`, and LIKE `ENCODE_IDENTITY_FIELDS`, this
# tuple is NOT a subset of a larger Rust const that also folds in
# provenance/dispatch facts -- `GpuInferenceTier` keeps its provenance
# (`device_name`, `kernels_disabled_requested`, `flash_compiled`,
# `build_features`) in its OWN, entirely DISJOINT `PROVENANCE_FIELDS` const
# (the SAME E3 disjoint shape `ENCODE_IDENTITY_FIELDS` follows, never
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
    identity-field comparison in this directory calls -- closing a future
    representational gap for a NEW field means registering one function in
    the table above, not writing a new `if field == ...` branch inline at
    each call site.
    """
    fn = IDENTITY_FIELD_CANONICALIZERS.get(field)
    return value if fn is None else fn(value)
