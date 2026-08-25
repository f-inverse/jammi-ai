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


# Per-field canonicalizer table: every identity field NOT listed here is
# compared with NO canonicalization (the JSON-decoded value as-is), because
# it carries no known cross-producer representational gap — see
# `compare_grad_oracle.py`'s `RUN_IDENTITY_FIELDS` doc and `ab_merge.py`'s
# `FINETUNE_IDENTITY_FIELDS` doc for the full field-by-field determinant
# table each comparator maintains for ITS OWN field set (this table is
# shared machinery, not a duplicate of either).
IDENTITY_FIELD_CANONICALIZERS = {
    "backbone_dtype": normalize_backbone_dtype,
    "target_modules": normalize_target_modules,
}


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
