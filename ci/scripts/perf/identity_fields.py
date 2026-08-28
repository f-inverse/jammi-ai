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
# on BOTH sides and REDs on a drift on either one).
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
    "pooling",
    "normalize",
    "warmup",
    "iters_measured",
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
# shared machinery, not a duplicate of either). `max_grad_norm` and
# `attention_arm` deliberately have NO canonicalizer: both producers emit
# the same vocabulary directly (a JSON number-or-null; `"eager"`/`"fused"`),
# and a canonicalizer that mapped torch's raw `"sdpa"` onto jammi's
# `"fused"` here would be WIDENING what counts as a match inside the
# comparator rather than each producer stating its own class honestly.
IDENTITY_FIELD_CANONICALIZERS = {
    "backbone_dtype": normalize_backbone_dtype,
    "target_modules": normalize_target_modules,
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
# `build_features`, `attention_arm`) in its OWN, entirely DISJOINT
# `PROVENANCE_FIELDS` const — see that struct's own doc for why `arm`/
# `attention_arm` are provenance here rather than identity (the CALLER'S
# request / the process-resolved reference class, neither a determinant of
# what the held-out loss itself computes). `FINETUNE_RUN_IDENTITY_FIELDS` is
# therefore compared for SET EQUALITY against
# `FinetuneRunTier::IDENTITY_FIELDS`, never a subset check — see
# `test_identity_fields_subset.py`'s own `FinetuneRunIdentityFieldsSubsetTests`
# for the mechanical assertion. This module carries no
# `FINETUNE_RUN_PROVENANCE_FIELDS` tuple, following `ENCODE_IDENTITY_FIELDS`'s
# own precedent: the Rust `PROVENANCE_FIELDS` const is extracted directly by
# the test suite's regex scan rather than duplicated into a second Python
# list nobody would keep in sync.
FINETUNE_RUN_IDENTITY_FIELDS = (
    # FinetuneStepTier's 18, minus attention_arm (17 entries) — carried over
    # by name, same order as the Rust const's own leading block.
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
    "max_grad_norm",
    "warmup",
    "row_lengths",
    # New (18 entries) — CONTRACT H4 v1/v2 + the objective-selection
    # amendment (embedding_loss/temperature/margin's objective-selected
    # nullness).
    "epochs",
    "lr",
    "schedule",
    "warmup_steps",
    "weight_decay",
    "grad_accum",
    "validation_fraction",
    "split_rule",
    "split_seed",
    "dataset_sha256",
    "heldout_ids_sha256",
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
