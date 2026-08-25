#!/usr/bin/env python3
"""Merge + table stage for `ci/scripts/perf/finetune_ab.sh`'s #352 A/B sweep.

Extracted out of that script's own inline heredoc (B3: an inline heredoc has
ZERO automated coverage — `AB_DRY_RUN=1` only exercises the DRY_RUN arm,
never a real leg, so `fused_proof`/`dispatch_pairs`/the merge loop never saw
a real report shape in CI) into this importable module specifically so
`test_ab_merge.py` in this same directory can drive the REAL entry point
(`main`, exactly what `finetune_ab.sh` invokes) against fixture directories
shaped like `run_leg`'s own `.exit`/`.json`/`.stderr` output, never a
hand-rolled call to `fused_proof()` with literal tuples standing in for a
report.

Never imported by any Cargo crate, never a jammi-bench dependency — a
CI-adjacent script the sweep alone runs, same footing `finetune_ab.sh`
itself already has.
"""

from __future__ import annotations

import json
import os
import sys

LEGS = ["jammi-eager", "jammi-fused", "torch-eager", "torch-sdpa"]

# B2 — the DECLARED classification `fused_proof` checks a dispatch-counter
# pair against, replacing the old blanket "(fused, eager) == (0, 0) is
# always fine, for every pair" rule (which made a report where every real
# fused site read (0, 0) and only ONE unrelated pair read a positive fused
# count print `fused_proof YES` — a net loss of detection versus the
# pre-generalization check, which positively required ln/rope/softmax each
# `fused > 0`).
#
# F5 (PR #372 audit round): the FIRST generalization of this table fixed
# the "(0, 0) silently excluded" bug for `ln` only, and in doing so
# introduced the SAME bug with the polarity flipped for every OTHER base:
# `rope`/`softmax` being ENTIRELY ABSENT from a report's schema (a real
# field renamed/deleted/feature-gated-off regression, not merely reading
# `(0, 0)`) was silently `continue`d past rather than failed, and any base
# not named in ANY of the three sets below (`geglu` — required by nothing,
# despite `finetune_ab.sh`'s own header claiming otherwise) was never
# checked AT ALL, present or absent, zero or nonzero. `fused_proof([('ln',
# 9, 0)])` — EVERY other pair entirely missing from the report — used to
# return `True`. The invariant this table now enforces: EVERY base that
# `dispatch_pairs` discovers in a real report must be in EXACTLY ONE of the
# three sets below (`ALL_BASES` is their union); a discovered base outside
# `ALL_BASES` is a schema-drift ERROR (`dispatch_pairs` raises, same B6
# per-leg-loud/whole-merge-safe handling `build_report` already gives a
# solo counter), never a silent exemption. Within each set, ABSENCE from
# the report is now ALSO a hard fail for every member (not just the
# `REQUIRED_PAIRS` ones) — a classified base that vanishes from the schema
# is exactly the regression this proof exists to catch.
#
#   * REQUIRED_PAIRS — no fused block in this crate absorbs these; each
#     MUST be PRESENT and show its own `fused > 0` (and, like every pair,
#     `eager == 0`).
#       - `ln`: dispatches inside every layer's own norm call, never folded
#         into a whole-attention or whole-MLP kernel, and
#         `finetune_step.rs`'s own counter-delta test already asserts its
#         (fused+eager) total is nonzero on every run.
#       - `geglu`: same reasoning as `ln` — `ModernBertMlp::forward`'s
#         training arm calls `geglu_apply_training` unconditionally for
#         every layer's MLP (see that function's own doc); its admission
#         domain (F32/BF16, contiguous, nonzero-even last dim) holds for
#         every real ModernBERT MLP shape, so nothing legitimately
#         absorbs or exempts it the way `attention_block` absorbs
#         `rope`/`softmax`. This closes F4/F5's own reproduction (a
#         "deleted/feature-gated-off fused MLP" reading `geglu = (0, 0)`
#         used to still print `fused_proof YES`).
#       - `attention_block`: the whole-attention fused kernel itself. A
#         checkpoint whose `head_dim != 64` legitimately falls back to
#         eager here (`report.rs`'s `attention_block_eager_dispatches`
#         field doc) — that is ALREADY caught by rule 1 below (`eager >
#         0` anywhere is a hard fail), so requiring `fused > 0` here for
#         the cases rule 1 does not already reject adds detection without
#         changing behaviour on that documented domain-refusal case.
#   * ABSORBABLE_BY_ATTENTION_BLOCK — `rope`/`softmax` MUST be PRESENT; may
#     read `(0, 0)` IFF `attention_block`'s OWN `fused` count is `> 0` this
#     run: `ModernBertAttention::forward_training_attention`'s FUSED arm is
#     the whole RoPE+QKᵀ+mask+softmax+PV chain as one op and never calls
#     `rope_apply`/`softmax_apply_training` at all (see that method's own
#     doc), so their independent admission call sites are simply never
#     reached. When `attention_block` itself never goes fused (the eager
#     attention composition ran instead), that composition DOES call
#     `rope_apply`/`softmax_apply_training` — each independently
#     admission-gated — so they must clear the same `fused > 0` bar a
#     required pair does.
#   * LORA_SITE_EXCLUSIVE_GROUP — `lora_epilogue`/`lora_linear` MUST both be
#     PRESENT, and are genuinely exclusive with EACH OTHER, not with a
#     third pair: every training-arm LoRA-adapted forward routes through
#     EXACTLY ONE of these two call sites
#     (`jammi_lora::lora_linear::lora_linear_fused_counters`'s own doc —
#     today `lora_epilogue` is PERMANENTLY `(0, 0)`, superseded by the
#     fused whole-site kernel `lora_linear` now reports). So only the
#     GROUP's sum needs a `fused > 0` proof, never each member alone.
REQUIRED_PAIRS = frozenset({"ln", "geglu", "attention_block"})
ABSORBABLE_BY_ATTENTION_BLOCK = frozenset({"rope", "softmax"})
LORA_SITE_EXCLUSIVE_GROUP = frozenset({"lora_epilogue", "lora_linear"})
ALL_BASES = REQUIRED_PAIRS | ABSORBABLE_BY_ATTENTION_BLOCK | LORA_SITE_EXCLUSIVE_GROUP
assert (
    len(REQUIRED_PAIRS) + len(ABSORBABLE_BY_ATTENTION_BLOCK) + len(LORA_SITE_EXCLUSIVE_GROUP) == len(ALL_BASES)
), "REQUIRED_PAIRS / ABSORBABLE_BY_ATTENTION_BLOCK / LORA_SITE_EXCLUSIVE_GROUP must be pairwise disjoint -- every base gets exactly ONE class"

# B5 — bf16's ULP near a loss value around 0.30: 7 explicit mantissa bits,
# exponent bucket [0.25, 0.5) => 2^-9. Every real sweep leg runs
# --backbone-dtype/--dtype bf16 (see `run_jammi_leg`/`run_torch_leg` in
# `finetune_ab.sh`), so this is the resolution `loss_first`/`loss_last`
# entries actually carry — see `finetune_step.rs`'s `losses` field doc /
# `torch_finetune_step.py`'s `loss_note` for the same figure stated next to
# the field itself.
BF16_LOSS_ULP_NEAR_0P3 = 2.0**-9  # ~0.001953125


def load_leg(raw_dir, config_slug, leg):
    """Read one `run_leg`-produced `.exit`/`.json`/`.stderr` triple and
    classify its outcome. Never raises: a MISSING/FAIL/OOM/DRY_RUN leg is a
    normal row, not a script error.
    """
    base = os.path.join(raw_dir, f"{config_slug}__{leg}")
    exit_path, out_path, err_path = base + ".exit", base + ".json", base + ".stderr"
    if not os.path.exists(exit_path):
        return {"outcome": "MISSING", "err_tail": "", "report": None}

    with open(exit_path) as fh:
        exit_code = fh.read().strip()
    err_tail = ""
    if os.path.exists(err_path):
        with open(err_path, errors="replace") as fh:
            err_lines = fh.read().splitlines()
        err_tail = "\n".join(err_lines[-5:])

    report = None
    try:
        with open(out_path) as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError):
        report = None

    if report is not None and (report.get("tool") == "dry-run" or report.get("ab_dry_run") is True):
        return {"outcome": "DRY_RUN", "err_tail": "", "report": None}

    if exit_code != "0" or report is None:
        low = err_tail.lower()
        oom_markers = ("out of memory", "cuda_error_out_of_memory", "cublas_status_alloc_failed", "outofmemoryerror")
        outcome = "OOM" if any(m in low for m in oom_markers) else "FAIL"
        return {"outcome": outcome, "err_tail": err_tail, "report": None}

    return {"outcome": "OK", "err_tail": "", "report": report}


def finetune_block(report, leg):
    return report["tiers"]["finetune_step"] if leg.startswith("jammi") else report["finetune_step"]


def dispatch_pairs(fs):
    """Every `(base, fused_key, eager_key)` positive-proof pair PRESENT in
    this report's `finetune_step` block, discovered from the JSON keys
    themselves rather than a hardcoded name list — a hardcoded ln/rope/
    softmax trio would silently stop catching a NEW fused op (geglu,
    lora_epilogue, lora_linear, attention_block, and whatever lands next)
    the day it is added to `finetune_step.rs`'s `FinetuneStepTier` without
    this script being updated in lockstep. Every key ending in
    `_fused_dispatches` names a pair; its sibling is the same base with
    `_eager_dispatches`, which `finetune_step.rs`'s own struct guarantees
    always exists alongside the fused counter (every fused/eager counter in
    that struct is added as a pair, never solo).

    B6 SCHEMA STRICTNESS: this function stays LOUD (raises `KeyError`) on a
    solo counter — a fused key with no eager sibling is a genuine schema
    bug (a struct field added without its pair), never a config this script
    should silently skip. F5 extends the SAME loudness to a base
    `fused_proof`'s classification tables (`REQUIRED_PAIRS` /
    `ABSORBABLE_BY_ATTENTION_BLOCK` / `LORA_SITE_EXCLUSIVE_GROUP`, whose
    union is `ALL_BASES`) do not know about: a NEW fused kernel landing in
    `finetune_step.rs` without this module's classification tables being
    updated in lockstep is exactly the same class of schema drift as a
    solo counter — `fused_proof` would otherwise silently never require
    anything of it (the F5 bug this closes). `metrics()`'s two `.get()`
    reads for `loss_first`/`loss_last` are the OPPOSITE choice,
    deliberately: those two fields are optional/best-effort table
    decoration (present since the loss-trajectory unit landed, absent on
    an older report schema, and absence there changes nothing this proof
    depends on), while a dispatch pair — and its classification — is
    STRUCTURAL to `fused_proof`'s entire claim. The two windows are
    intentionally different; what changed in this revision is only WHERE
    this exception is caught — see `build_report`'s per-leg `try`/`except`,
    which stops one bad leg's solo-counter (or now, unclassified-base)
    `KeyError` from discarding the merged table for every other config
    (previously this raise propagated all the way to the top-level script
    and aborted the entire merge).
    """
    pairs = []
    for key in fs:
        if not key.endswith("_fused_dispatches"):
            continue
        base = key[: -len("_fused_dispatches")]
        eager_key = f"{base}_eager_dispatches"
        if eager_key not in fs:
            raise KeyError(
                f"'{key}' has no matching '{eager_key}' in the report — "
                "finetune_step.rs's fused/eager counters are supposed to "
                "always come in pairs; a solo counter is a schema bug, not "
                "a config this script should silently skip."
            )
        if base not in ALL_BASES:
            raise KeyError(
                f"dispatch-pair base {base!r} (from {key!r}) is not classified in ALL_BASES "
                f"({sorted(ALL_BASES)!r}) — a NEW fused kernel landed in finetune_step.rs "
                "without fused_proof's REQUIRED_PAIRS / ABSORBABLE_BY_ATTENTION_BLOCK / "
                "LORA_SITE_EXCLUSIVE_GROUP tables being updated to cover it. This is a "
                "schema-drift bug, not a base this script should silently leave unchecked "
                "(see F5's own fix note on the module-level classification tables)."
            )
        pairs.append((base, fs[key], fs[eager_key]))
    return pairs


def metrics(entry, leg):
    """Extract this leg's table/proof metrics from its raw report. Returns
    `None` when the leg itself did not produce a usable report (see
    `load_leg`); raises (never silently drops a field) when the report WAS
    produced but a STRUCTURAL piece — a dispatch pair — is malformed (see
    `dispatch_pairs`'s own doc for why that is the loud half of this
    module's B6 schema-strictness split).
    """
    if entry["outcome"] != "OK":
        return None
    fs = finetune_block(entry["report"], leg)
    m = {
        "s_per_step_p50": fs["s_per_step_p50"]["value"],
        "triplets_per_s": fs["triplets_per_s"]["value"],
        "loss_first": fs.get("loss_first"),
        "loss_last": fs.get("loss_last"),
    }
    if leg.startswith("jammi"):
        m["vram_delta_bytes"] = fs["peak_vram_bytes"]["value"]
        m["vram_absolute_bytes"] = None
        m["dispatch_pairs"] = dispatch_pairs(fs)
    else:
        m["vram_delta_bytes"] = fs["peak_vram_delta_bytes"]["value"]
        m["vram_absolute_bytes"] = fs["peak_vram_absolute_bytes"]["value"]
    return m


def fused_proof(m):
    """See the module-level `REQUIRED_PAIRS`/`ABSORBABLE_BY_ATTENTION_BLOCK`/
    `LORA_SITE_EXCLUSIVE_GROUP` (union `ALL_BASES`) doc for the
    classification this checks each pair against. Returns `True`/`False`/
    `None` (no `dispatch_pairs` at all — not a jammi leg, or the leg itself
    did not run). Raises (via `dispatch_pairs`, which `metrics()` already
    calls before this function ever sees `m` — see that function's own doc)
    if `m["dispatch_pairs"]` would ever contain a base outside `ALL_BASES`;
    `fused_proof` itself never receives an unclassified base to begin with.

    Rules, in order — EVERY base in `ALL_BASES` (not just `REQUIRED_PAIRS`)
    must be PRESENT in this report's pairs; absence is a hard fail for
    every classified base, never a silently-granted exemption (F5: the
    pre-fix code granted this exemption to every base except `ln`):
      1. ANY pair with `eager > 0` is a hard, unconditional fail — an
         admitted call site that actually fell back, on ANY pair, in ANY
         group. Never exempted.
      2. Every `REQUIRED_PAIRS` base must be PRESENT in this report's pairs
         (a required pair vanishing from the JSON entirely — the field
         renamed, deleted, or feature-gated off — is exactly the schema
         regression this proof exists to catch, never silently excluded)
         AND show `fused > 0`.
      3. Every `ABSORBABLE_BY_ATTENTION_BLOCK` member must be PRESENT (same
         "absence is a fail" rule as step 2 — F5's fix), and may read
         `(0, 0)` ONLY when `attention_block`'s own `fused` count is `> 0`
         in this SAME report; otherwise it must independently clear
         `fused > 0`.
      4. Every `LORA_SITE_EXCLUSIVE_GROUP` member must be PRESENT (same
         rule again), and the GROUP is then checked AS A GROUP: the SUM of
         their `fused` counts must be `> 0` (whichever member actually
         carries this run's dispatch — see the group's own doc).
      5. Overall: at least one pair ANYWHERE in the report must show
         `fused > 0` — a report where every single pair reads `(0, 0)`
         (e.g. a schema regression that dropped every counter) is NOT
         vacuously `True`. Steps 2/3/4 already make this true whenever
         `REQUIRED_PAIRS` is non-empty, but this stays a distinct,
         independently-stated check so the property holds even if
         `REQUIRED_PAIRS` were ever emptied.
    """
    if m is None:
        return None
    pairs = m.get("dispatch_pairs")
    if not pairs:
        return False
    by_base = {base: (fused, eager) for base, fused, eager in pairs}

    if any(eager > 0 for _fused, eager in by_base.values()):
        return False

    for base in REQUIRED_PAIRS:
        if base not in by_base:
            return False
        fused, _eager = by_base[base]
        if fused == 0:
            return False

    attention_block_fused = by_base.get("attention_block", (0, 0))[0]
    for base in ABSORBABLE_BY_ATTENTION_BLOCK:
        if base not in by_base:
            return False  # F5: absence is a schema regression, never silently excluded
        fused, _eager = by_base[base]
        if fused == 0 and attention_block_fused == 0:
            return False

    for base in LORA_SITE_EXCLUSIVE_GROUP:
        if base not in by_base:
            return False  # F5: absence is a schema regression, never silently excluded
    lora_group_fused = sum(by_base[base][0] for base in LORA_SITE_EXCLUSIVE_GROUP)
    if lora_group_fused == 0:
        return False

    return any(fused > 0 for fused, _eager in by_base.values())


def fmt(v, nd=4):
    return "n/a" if v is None else f"{v:.{nd}f}"


def fmt_loss(v):
    """B5: `loss_first`/`loss_last` are bf16-sourced on every real sweep
    leg (ULP ~0.00195 near 0.30 — see `BF16_LOSS_ULP_NEAR_0P3`). `fmt`'s
    default 4 decimal digits (resolution 0.0001) implies precision the
    dtype does not carry; 3 decimals (resolution 0.001) is still finer
    than the ULP without implying a 4th significant digit exists.
    """
    return "n/a" if v is None else f"{v:.3f}"


def fmt_bytes(v):
    return "n/a" if v is None else f"{int(v):,}"


def config_slugs(raw_dir):
    slugs = set()
    if os.path.isdir(raw_dir):
        for name in os.listdir(raw_dir):
            if name.endswith(".exit") and "__" in name:
                slugs.add(name.split("__", 1)[0])
    return sorted(slugs)


def build_report(raw_dir, steps, warmup, pass_ratio, torch_lora_init="peft"):
    """The merge stage itself: read every leg under `raw_dir`, extract
    metrics, compute the fused-dispatch proof / throughput ratio / loss
    ratio / verdict per config, and render both the merged JSON dict and
    the printed table string. Returns `(merged, table)`, or `(None, None)`
    if `raw_dir` has no leg output at all (an empty sweep — the caller
    treats this as a hard failure, unchanged from before this file's
    extraction).

    B6: a merge-stage error on ONE leg (`metrics()`/`dispatch_pairs()`
    raising — a solo dispatch counter, a missing report key) is caught
    HERE, per leg, so it produces a LOUD per-row error (visible in both the
    table and the JSON, under that leg's `outcome`/this config's
    `jammi_fused_dispatch_proof`) instead of discarding every OTHER
    config's row too. This is a change in WHERE the exception is caught,
    never in whether `dispatch_pairs` raises at all.
    """
    slugs = config_slugs(raw_dir)
    if not slugs:
        return None, None

    merged = {
        "steps": steps,
        "warmup": warmup,
        "pass_ratio_bar": pass_ratio,
        "lora_init": {
            "torch": torch_lora_init,
            "jammi": "jammi (LoraInitMode::ZerosB; not configurable via finetune-step's CLI)",
            "note": "B4: a loss-trajectory-equivalence comparison additionally requires "
            "torch_lora_init == 'jammi' (torch_finetune_step.py's --lora-init jammi re-draws "
            "A from jammi's own bound) — a throughput-only sweep (this script's default, "
            "'peft') does not need matched init at all. See torch_finetune_step.py's "
            "'LoRA INIT IS NOT A MATCH BY DEFAULT' section.",
        },
        "configs": {},
    }
    table_rows = []
    summary_rows = []

    for slug in slugs:
        entries = {leg: load_leg(raw_dir, slug, leg) for leg in LEGS}
        leg_metrics = {}
        leg_merge_errors = {}
        for leg in LEGS:
            try:
                leg_metrics[leg] = metrics(entries[leg], leg)
                leg_merge_errors[leg] = None
            except Exception as exc:  # noqa: BLE001 -- B6: LOUD, per-leg,
                # never silent, never fatal to the rest of the merge; see
                # this function's own doc and `dispatch_pairs`'s.
                leg_metrics[leg] = None
                leg_merge_errors[leg] = f"{type(exc).__name__}: {exc}"

        if leg_merge_errors["jammi-fused"] is not None:
            proof = f"ERROR: {leg_merge_errors['jammi-fused']}"
        else:
            proof = fused_proof(leg_metrics["jammi-fused"])

        for leg in LEGS:
            err_tail = entries[leg]["err_tail"]
            if leg_merge_errors[leg] is not None:
                err_tail = (err_tail + "\n" if err_tail else "") + f"[merge-stage] {leg_merge_errors[leg]}"
            table_rows.append(
                (
                    slug,
                    leg,
                    entries[leg]["outcome"],
                    leg_metrics[leg],
                    proof if leg == "jammi-fused" else None,
                    err_tail,
                )
            )

        fused_m, sdpa_m = leg_metrics["jammi-fused"], leg_metrics["torch-sdpa"]
        ratio = (
            fused_m["triplets_per_s"] / sdpa_m["triplets_per_s"]
            if (fused_m and sdpa_m and sdpa_m["triplets_per_s"])
            else None
        )

        # loss_final_ratio: jammi-fused's loss_last over torch-sdpa's
        # loss_last. SAME DATA, COST FIXTURE -- NOT A QUALITY RESULT (per
        # finetune_step.rs's own module doc's "Honesty about what is
        # measured", and torch_finetune_step.py's "LOSS TRAJECTORY"
        # section): the two stacks run different attention-kernel
        # arithmetic and different LoRA init distributions unless
        # torch_lora_init == "jammi", so a ratio far from 1.0 does NOT mean
        # either stack is wrong -- it means the loss values are not
        # comparable under these settings. Printed anyway so a large
        # divergence is VISIBLE to a human reader, never asserted against a
        # bar.
        loss_ratio = None
        if (
            fused_m
            and sdpa_m
            and fused_m.get("loss_last") is not None
            and sdpa_m.get("loss_last") is not None
            and sdpa_m["loss_last"] != 0.0
        ):
            loss_ratio = fused_m["loss_last"] / sdpa_m["loss_last"]

        any_dry_run = any(entries[leg]["outcome"] == "DRY_RUN" for leg in LEGS)
        torch_fits = entries["torch-sdpa"]["outcome"] == "OK"
        jammi_fused_fits = entries["jammi-fused"]["outcome"] == "OK"

        # The #352 bar is "no OOM where torch fits" -- it binds ONLY when
        # torch-sdpa itself succeeded. If torch-sdpa didn't fit, there is
        # no baseline to hold jammi-fused to and the bar does not apply --
        # that is NOT the same thing as jammi failing, and must not print
        # as FAIL.
        if any_dry_run:
            verdict = "N/A (dry-run)"
        elif not torch_fits:
            verdict = f"N/A (torch-sdpa itself did not fit: {entries['torch-sdpa']['outcome']} — bar does not apply)"
        elif not jammi_fused_fits:
            verdict = f"FAIL (OOM where torch fits: jammi-fused {entries['jammi-fused']['outcome']})"
        elif ratio is None:
            verdict = "FAIL (no ratio: triplets_per_s missing on an OK leg — investigate)"
        elif ratio < pass_ratio:
            verdict = f"FAIL (ratio {ratio:.3f} < {pass_ratio})"
        else:
            verdict = f"PASS (ratio {ratio:.3f})"

        # Advisory (iv), round-2 audit fix on PR #372: a failed/errored
        # `fused_proof` used to only APPEND a cosmetic "[WARN: ...]" suffix
        # to whatever ratio-based verdict was already computed above -- so a
        # config whose jammi-fused leg silently fell back to EAGER kernels
        # (the exact regression `fused_proof` exists to catch) could still
        # print `PASS (ratio 0.95x) [WARN: ...]`, and nothing downstream
        # (`main()`'s exit code, a human skimming the table for the string
        # "FAIL") ever noticed. This is a DIFFERENT class of problem than
        # the ratio-based PASS/FAIL bar this crate deliberately RECORDS,
        # never GATES, across a heterogeneous fleet (see
        # `finetune_ab.sh`'s own "script's own exit code reflects whether
        # the sweep RAN, not whether every [config] passed" doctrine): a
        # ratio below bar is a real, machine-dependent PERFORMANCE
        # observation; a failed fused_proof means the MEASUREMENT ITSELF is
        # not known to have exercised the code path it claims to -- the
        # ratio computed above could belong to a DIFFERENT kernel
        # composition entirely, making the PASS/FAIL classification
        # meaningless rather than merely unfavorable. INVALID therefore
        # REPLACES (never just annotates) whatever ratio-based verdict was
        # computed, and `main()` treats ANY `INVALID` verdict as a hard
        # sweep failure (non-zero exit) -- the one carve-out from the
        # record-don't-gate doctrine this crate makes, because this is a
        # correctness-of-measurement question, not a perf-number question.
        if proof is False or isinstance(proof, str):
            reason = (
                f"errored: {proof}" if isinstance(proof, str)
                else "checked and FAILED — see fused_proof column for the classification"
            )
            verdict = (
                f"INVALID (fused-dispatch proof {reason} — this leg's PASS/FAIL classification "
                f"cannot be trusted; the ratio-based verdict this would otherwise have been is "
                f"discarded, not merely annotated)"
            )

        summary_rows.append((slug, ratio, loss_ratio, verdict))
        merged["configs"][slug] = {
            "legs": {leg: {"outcome": entries[leg]["outcome"], "metrics": leg_metrics[leg]} for leg in LEGS},
            "jammi_fused_dispatch_proof": proof,
            "ratio_jammi_fused_over_torch_sdpa": ratio,
            "loss_final_ratio_jammi_fused_over_torch_sdpa": loss_ratio,
            "loss_final_ratio_note": "same data, cost fixture -- NOT a quality result "
            "(see finetune_step.rs's module doc / torch_finetune_step.py's LOSS "
            "TRAJECTORY section: different attention-kernel arithmetic and "
            "reduction order between the two stacks makes a loss VALUE comparison "
            "meaningless even given identical synthetic input ids, unless "
            "torch_lora_init == 'jammi'). Printed so a divergence is visible, never "
            "gated. loss values carry only bf16's ULP (~0.00195 near 0.30) of real "
            "precision -- see BF16_LOSS_ULP_NEAR_0P3.",
            "verdict": verdict,
        }

    lines = [
        "# finetune A/B -- jammi eager vs jammi fused vs torch eager vs torch sdpa",
        f"# steps={steps} warmup={warmup} pass_bar={pass_ratio}x torch-sdpa triplets/s, no OOM where torch fits",
        f"# torch --lora-init={torch_lora_init}; jammi always uses its own ZerosB init -- loss_final_ratio "
        "is only a loss-TRAJECTORY-equivalence signal when torch_lora_init == 'jammi'.",
        "# loss-trajectory equivalence (jammi-fused vs jammi-eager, real trainer, >=5 seeds) is a SEPARATE check -- not measured here.",
        "# loss_first->loss_last and loss_final_ratio below: SAME DATA, COST FIXTURE -- NOT A QUALITY RESULT. "
        "Values are bf16-sourced (ULP ~0.00195 near 0.30) -- printed to 3 decimals, never gated.",
        f"{'config':<16}{'leg':<13}{'outcome':<9}{'s/step_p50':<12}{'triplets/s':<12}"
        f"{'vram_delta(comparable)':<24}{'vram_absolute(torch only)':<27}{'fused_proof':<28}{'loss_first->last':<24}",
    ]
    for slug, leg, outcome, m, proof_val, err_tail in table_rows:
        p50 = fmt(m["s_per_step_p50"]) if m else "n/a"
        tps = fmt(m["triplets_per_s"]) if m else "n/a"
        vd = fmt_bytes(m["vram_delta_bytes"]) if m else "n/a"
        va = fmt_bytes(m["vram_absolute_bytes"]) if m else "n/a"
        if proof_val is None:
            proof_s = "n/a"
        elif isinstance(proof_val, str):
            proof_s = proof_val[:26]
        else:
            proof_s = "YES" if proof_val else "NO"
        loss_s = (
            "n/a"
            if not m or m.get("loss_first") is None or m.get("loss_last") is None
            else f"{fmt_loss(m['loss_first'])}->{fmt_loss(m['loss_last'])}"
        )
        lines.append(
            f"{slug:<16}{leg:<13}{outcome:<9}{p50:<12}{tps:<12}{vd:<24}{va:<27}{proof_s:<28}{loss_s:<24}"
        )
        if outcome not in ("OK", "DRY_RUN") and err_tail:
            last = err_tail.splitlines()[-1][:120] if err_tail.splitlines() else ""
            lines.append(f"    -> {last}")
        elif err_tail and "[merge-stage]" in err_tail:
            last = err_tail.splitlines()[-1][:120]
            lines.append(f"    -> {last}")

    lines.append("")
    lines.append(
        f"{'config':<16}{'ratio(fused/sdpa)':<20}{'loss_final_ratio(fused/sdpa,NOT-quality)':<42}{'verdict':<60}"
    )
    for slug, ratio, loss_ratio, verdict in summary_rows:
        ratio_s = "n/a" if ratio is None else f"{ratio:.3f}"
        loss_ratio_s = "n/a" if loss_ratio is None else f"{loss_ratio:.4f}"
        lines.append(f"{slug:<16}{ratio_s:<20}{loss_ratio_s:<42}{verdict:<60}")

    table = "\n".join(lines)
    return merged, table


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 5:
        print(
            "usage: ab_merge.py RAW_DIR OUT_DIR STEPS WARMUP PASS_RATIO [TORCH_LORA_INIT]",
            file=sys.stderr,
        )
        return 2
    raw_dir, out_dir, steps, warmup, pass_ratio_s = argv[:5]
    torch_lora_init = argv[5] if len(argv) > 5 else "peft"
    pass_ratio = float(pass_ratio_s)

    merged, table = build_report(raw_dir, steps, warmup, pass_ratio, torch_lora_init)
    if merged is None:
        print(f"finetune_ab: FAIL — no leg output found under {raw_dir}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "finetune_ab_report.json"), "w") as fh:
        json.dump(merged, fh, indent=2)
    print(table)
    with open(os.path.join(out_dir, "finetune_ab_table.txt"), "w") as fh:
        fh.write(table + "\n")

    # Advisory (iv), round-2 audit fix on PR #372: the ONE carve-out from
    # this crate's own record-don't-gate doctrine (see `finetune_ab.sh`'s
    # module doc and `build_report`'s own verdict-computation comment) --
    # an `INVALID` verdict (a failed/errored `fused_proof`) is a
    # correctness-of-MEASUREMENT problem, not a machine-dependent
    # performance number, so it is the one thing this sweep's own exit code
    # DOES gate on. An ordinary ratio-based `FAIL` row remains
    # record-only, unchanged.
    invalid_slugs = [
        slug for slug, cfg in merged["configs"].items() if str(cfg.get("verdict", "")).startswith("INVALID")
    ]
    if invalid_slugs:
        print(
            f"finetune_ab: FAIL — {len(invalid_slugs)} config(s) have an INVALID verdict "
            f"(fused-dispatch proof failed or errored, see the table above): {invalid_slugs}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
