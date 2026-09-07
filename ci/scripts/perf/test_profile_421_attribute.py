#!/usr/bin/env python3
"""Hermetic tests for `profile_421_attribute.py` (issue #421 post-export
attribution unit; CONTRACT `scratchpad/contract-421-profile.md` v2.5
`### Attribution` / `§D3`; pass 3 = the adversarial-audit fold).

No GPU, no pod, no `nsys`, no network. The kernel-name<->shape MAPPING
itself is tested against SMALL, REAL, committed fixtures cut byte-for-byte
from real nsys census exports (`fixtures/profile_421_clip_text_{a1,a2,d1,
d2}/`, `fixtures/profile_421_clip_vision_{a1,d1,d2}/` — see each
directory's `PROVENANCE.md`), never a hand-rolled census standing in for
what a real export actually contains. Per this crate's "never transcribe a
timing number into a test" convention, NOTHING here asserts a literal
`us_per_step`/`share`/`launches_per_step` value from a fixture as an
expected number — every assertion is STRUCTURAL: which chain a
`(kernel, grid, block)` row lands in, that chain shares partition
`gpu_kernel_us_per_step` exactly, that shares are `<= 1`, that every
declared chain is present-or-explicitly-absent, `decision_grade`'s own
threshold arithmetic, the two-sided decision rule evaluated on SYNTHETIC
share numbers (never a real leg's), and (pass 3) the DIRECTION/ORDERING a
D1-vs-D2 differential pair must move in — never a literal delta.

Run: `python3 ci/scripts/perf/test_profile_421_attribute.py`
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
ATTRIBUTE = PERF_DIR / "profile_421_attribute.py"
FIXTURE_KERNELS = PERF_DIR / "fixtures" / "profile_421_clip_text_a1" / "kernels.json"
FIXTURE_KERNELS_A2 = PERF_DIR / "fixtures" / "profile_421_clip_text_a2" / "kernels.json"
FIXTURE_KERNELS_D1 = PERF_DIR / "fixtures" / "profile_421_clip_text_d1" / "kernels.json"
FIXTURE_KERNELS_D2 = PERF_DIR / "fixtures" / "profile_421_clip_text_d2" / "kernels.json"
FIXTURE_KERNELS_VISION_A1 = PERF_DIR / "fixtures" / "profile_421_clip_vision_a1" / "kernels.json"
FIXTURE_KERNELS_VISION_D1 = PERF_DIR / "fixtures" / "profile_421_clip_vision_d1" / "kernels.json"
FIXTURE_KERNELS_VISION_D2 = PERF_DIR / "fixtures" / "profile_421_clip_vision_d2" / "kernels.json"
FIXTURE_KERNELS_HTSAT_A1 = PERF_DIR / "fixtures" / "profile_421_htsat_a1" / "kernels.json"
FIXTURE_KERNELS_HTSAT_D1 = PERF_DIR / "fixtures" / "profile_421_htsat_d1" / "kernels.json"

sys.path.insert(0, str(PERF_DIR))
import profile_421_attribute as attribute  # noqa: E402


# The REAL leg parameters `clip-text-A1`'s own `manifest.json` recorded
# (batch=8, max_seq_length=77, fusible_site_census.lora_sites_wrapped=48) —
# see `fixtures/profile_421_clip_text_a1/PROVENANCE.md`.
CLIP_TEXT_A1_MANIFEST_FIELDS = {
    "tower": "clip-text",
    "dtype": "f32",
    "batch": 8,
    "max_seq_length": 77,
    "kernels_disabled": [],
    "census_ok": True,
    "census_exit": 0,
    "fusible_site_census": {
        "n": {"lora_sites_wrapped": 48, "layer_norms": 25, "gelu_seam_calls_per_forward": 0},
        "m": {"lora_sites_wrapped": 48, "layer_norms": 25, "gelu_seam_calls_per_forward": 0},
    },
}

CLIP_TEXT_A2_MANIFEST_FIELDS = dict(CLIP_TEXT_A1_MANIFEST_FIELDS, dtype="bf16")

CLIP_TEXT_D1_MANIFEST_FIELDS = dict(
    CLIP_TEXT_A1_MANIFEST_FIELDS, kernels_disabled=["lora_linear_fused", "layer_norm_fused"]
)
CLIP_TEXT_D2_MANIFEST_FIELDS = dict(CLIP_TEXT_A1_MANIFEST_FIELDS, kernels_disabled=["lora_linear_fused"])

CLIP_VISION_A1_MANIFEST_FIELDS = {
    "tower": "clip-vision",
    "dtype": "f32",
    "batch": 8,
    "kernels_disabled": [],
    "census_ok": True,
    "census_exit": 0,
    "fusible_site_census": {
        "n": {"lora_sites_wrapped": 48, "layer_norms": 26, "gelu_seam_calls_per_forward": 0},
        "m": {"lora_sites_wrapped": 48, "layer_norms": 26, "gelu_seam_calls_per_forward": 0},
    },
}
CLIP_VISION_D1_MANIFEST_FIELDS = dict(
    CLIP_VISION_A1_MANIFEST_FIELDS, kernels_disabled=["lora_linear_fused", "layer_norm_fused"]
)
CLIP_VISION_D2_MANIFEST_FIELDS = dict(CLIP_VISION_A1_MANIFEST_FIELDS, kernels_disabled=["lora_linear_fused"])

# The REAL `htsat-A1`/`htsat-D1` legs' own `manifest.json` fields (pod
# `p421` run 2) — see `fixtures/profile_421_htsat_a1/PROVENANCE.md`.
HTSAT_A1_MANIFEST_FIELDS = {
    "tower": "htsat",
    "dtype": "f32",
    "batch": 8,
    "kernels_disabled": [],
    "census_ok": True,
    "census_exit": 0,
    "fusible_site_census": {
        "n": {"lora_sites_wrapped": 77, "layer_norms": 29, "gelu_seam_calls_per_forward": 12},
        "m": {"lora_sites_wrapped": 77, "layer_norms": 29, "gelu_seam_calls_per_forward": 12},
    },
}
HTSAT_D1_MANIFEST_FIELDS = dict(
    HTSAT_A1_MANIFEST_FIELDS,
    kernels_disabled=["lora_linear_fused", "layer_norm_fused", "gelu_erf_fused"],
)


def load_fixture_census(path: Path = FIXTURE_KERNELS) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def badd_launches_at_grid0(census: dict, grid0: int) -> float:
    """`launches_per_step` of the ONE `badd_f32` row in `census`'s own
    `by_kernel_and_grid` whose grid's first dimension is `grid0` — used to
    read the `badd_f32` launch-count ladder straight off a committed
    fixture, never a transcribed literal."""
    for row in census["by_kernel_and_grid"]:
        if row["kernel"] == "badd_f32" and row["grid"][0] == grid0:
            return row["launches_per_step"]
    raise AssertionError(f"no badd_f32 row at grid0={grid0} in this census")


def clip_text_signatures() -> attribute.TowerSignatures:
    return attribute.derive_signatures("clip-text", CLIP_TEXT_A1_MANIFEST_FIELDS)


def clip_vision_signatures() -> attribute.TowerSignatures:
    return attribute.derive_signatures("clip-vision", CLIP_VISION_A1_MANIFEST_FIELDS)


def htsat_signatures() -> attribute.HtsatSignatures:
    return attribute.derive_signatures("htsat", HTSAT_A1_MANIFEST_FIELDS)


class DeriveSignaturesTests(unittest.TestCase):
    def test_clip_text_witnessed_and_declared_fields(self):
        sig = clip_text_signatures()
        self.assertEqual(sig.rows.value, 24)
        self.assertIn("witnessed", sig.rows.source)
        self.assertEqual(sig.seq.value, 77)
        self.assertIn("witnessed", sig.seq.source)
        self.assertEqual(sig.width.value, 512)
        self.assertIn("declared_constant", sig.width.source)
        self.assertEqual(sig.heads.value, 8)
        self.assertEqual(sig.mlp_width.value, 2048)
        self.assertEqual(sig.layers.value, 12)
        self.assertIn("witnessed", sig.layers.source)

    def test_clip_vision_uses_declared_seq_constant(self):
        manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
        manifest["tower"] = "clip-vision"
        manifest.pop("max_seq_length")
        sig = attribute.derive_signatures("clip-vision", manifest)
        self.assertEqual(sig.seq.value, 50)
        self.assertIn("declared_constant", sig.seq.source)
        self.assertEqual(sig.width.value, 768)
        self.assertEqual(sig.heads.value, 12)

    def test_derived_element_counts_match_the_real_export(self):
        # These are the numbers `PROVENANCE.md` documents the
        # `clip-text-A1` fixture rows were chosen to sit exactly at.
        sig = clip_text_signatures()
        self.assertEqual(sig.gelu_shape_elements(), 1848 * 2048)
        self.assertEqual(sig.attn_softmax_rows(), 24 * 8 * 77)
        self.assertEqual(sig.attn_batch_count(), 24 * 8)
        self.assertEqual(sig.attn_shape_elements(), 24 * 8 * 77 * 77)
        self.assertEqual(sig.qkv_shape_elements(), 24 * 77 * 3 * 512)
        self.assertEqual(sig.out_shape_elements(), 24 * 77 * 512)
        self.assertEqual(sig.ln_row_count(), 24 * 77)

    def test_clip_vision_derived_element_counts_match_the_real_export(self):
        # `fixtures/profile_421_clip_vision_a1/PROVENANCE.md`'s own numbers.
        sig = clip_vision_signatures()
        self.assertEqual(sig.ln_row_count(), 1200)
        self.assertEqual(sig.gelu_shape_elements(), 3600 * 1024)
        self.assertEqual(sig.attn_softmax_rows(), 14400)
        self.assertEqual(sig.attn_batch_count(), 288)
        self.assertEqual(sig.attn_shape_elements(), 24 * 12 * 50 * 50)
        self.assertEqual(sig.out_shape_elements(), 900 * 1024)
        self.assertEqual(sig.qkv_shape_elements(), 2700 * 1024)

    def test_param_scale_elements_never_collide_with_activation_tiers(self):
        """The `OPTIMIZER` parameter-scale rule and the four activation
        tiers must partition cleanly for BOTH towers at their declared
        constants — a collision would make `classify_kernel`'s priority
        order (parameter-scale checked before the activation tiers)
        silently misroute a real activation-tensor row into `OPTIMIZER`."""
        for sig in (clip_text_signatures(), clip_vision_signatures()):
            activation_tiers = {
                sig.qkv_shape_elements(),
                sig.out_shape_elements(),
                sig.gelu_shape_elements(),
                sig.attn_shape_elements(),
                sig.ln_row_count(),
            }
            self.assertTrue(activation_tiers.isdisjoint(sig.param_scale_elements()))

    def test_unknown_tower_raises(self):
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_signatures("bert", CLIP_TEXT_A1_MANIFEST_FIELDS)

    def test_missing_batch_raises(self):
        manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
        del manifest["batch"]
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_signatures("clip-text", manifest)

    def test_non_multiple_lora_sites_raises(self):
        manifest = json.loads(json.dumps(CLIP_TEXT_A1_MANIFEST_FIELDS))
        manifest["fusible_site_census"]["n"]["lora_sites_wrapped"] = 49
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_signatures("clip-text", manifest)

    def test_flat_fusible_site_census_shape_also_works(self):
        manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
        manifest["fusible_site_census"] = {
            "lora_sites_wrapped": 48,
            "layer_norms": 25,
            "gelu_seam_calls_per_forward": 0,
        }
        sig = attribute.derive_signatures("clip-text", manifest)
        self.assertEqual(sig.layers.value, 12)


class IsKnownKernelNameTests(unittest.TestCase):
    """`is_known_kernel_name`'s TWO admission paths (pass 3, finding 3 —
    exact names only; no substring, no anonymous-name shortcut)."""

    def test_explicit_ground_truth_names_are_known(self):
        self.assertTrue(attribute.is_known_kernel_name("badd_f32"))

    def test_kernel2_is_no_longer_known_by_name(self):
        """Pass 3, finding 3: the literal `Kernel2` admission is REMOVED —
        an anonymous kernel is admitted ONLY by `classify_kernel`'s own
        grid-family rules now, never by an "we've seen this before" name
        entry."""
        self.assertFalse(attribute.is_known_kernel_name("Kernel2"))

    def test_bf16_only_explicit_names_are_known(self):
        for name in attribute.BF16_ONLY_KERNEL_NAMES:
            self.assertTrue(attribute.is_known_kernel_name(name))

    def test_evidenced_bf16_gemm_tile_names_are_hand_listed_not_substring_matched(self):
        """The eight real `ampere_bf16_s16816gemm_*` names observed on
        `clip-text-A2`/`clip-vision-A2` are hand-listed literally in
        `KNOWN_KERNEL_NAMES` now (pass 3) — NOT admitted via a `"gemm"`
        substring shortcut, which pass 3 removes."""
        name = "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nn"
        self.assertIn(name, attribute.KNOWN_KERNEL_NAMES)
        self.assertTrue(attribute.is_known_kernel_name(name))

    def test_an_unobserved_gemm_shaped_name_is_not_known_by_substring(self):
        """`is_known_kernel_name` admits ONLY exact, hand-listed names
        (plus the BF16-twin rule) — never a name via the `"gemm"`
        substring. A brand-new, never-observed tile-variant string is
        genuinely unknown, even though it looks GEMM-shaped."""
        name = "some_future_tile_variant_gemm_v9"
        self.assertNotIn(name, attribute.KNOWN_KERNEL_NAMES)
        self.assertFalse(attribute.is_known_kernel_name(name))
        self.assertFalse(attribute.is_known_kernel_name("magma_some_other_kernel"))

    def test_bf16_twin_of_a_known_f32_name_is_known(self):
        # `cast_bf16_f32`.replace("bf16", "f32") -> "cast_f32_f32", known.
        self.assertTrue(attribute.is_known_kernel_name("cast_bf16_f32"))
        self.assertTrue(attribute.is_known_kernel_name("layer_norm_fwd_bf16_biased"))

    def test_a_genuinely_unknown_name_is_not_known(self):
        self.assertFalse(attribute.is_known_kernel_name("totally_new_kernel_f32"))
        self.assertFalse(attribute.is_known_kernel_name("totally_new_kernel_bf16"))


class ClassifyKernelUnitTests(unittest.TestCase):
    """Per-row classification, isolated from the whole-census pass — one
    assertion per mapping decision documented in the module doc."""

    def setUp(self):
        self.sig = clip_text_signatures()

    def test_layer_norm_kernels_match_by_name_at_any_grid(self):
        for name in ("layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"):
            entry = {"kernel": name, "grid": [1, 1, 1], "block": [1, 1, 1]}
            self.assertEqual(
                attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_LN
            )

    def test_usigmoid_matches_gelu_at_any_grid(self):
        entry = {"kernel": "usigmoid_f32", "grid": [1, 1, 1], "block": [1, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_GELU
        )

    def test_affine_and_bmul_match_gelu_only_at_declared_shape(self):
        elements = self.sig.gelu_shape_elements()
        at_shape = {"kernel": "affine_f32", "grid": [3696, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(elements, 3696 * 1024)
        self.assertEqual(
            attribute.classify_kernel(at_shape, self.sig, "clip-text"), attribute.CHAIN_GELU
        )

    def test_bmul_at_the_attention_shape_falls_through_to_attn_not_gelu(self):
        """Pass-2 behavior change from pass 1: a GELU-shape-gated kernel
        NAME at a DIFFERENT declared shape (the attention tensor) now
        FALLS THROUGH to `C-ATTN-<tower>` rather than stopping at
        `UNATTRIBUTED` — see module doc, "fall through"."""
        wrong_shape = {"kernel": "bmul_f32", "grid": [1112, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(self.sig.attn_shape_elements(), 24 * 8 * 77 * 77)
        self.assertEqual(
            attribute.classify_kernel(wrong_shape, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

    def test_bmul_at_no_declared_shape_is_unattributed(self):
        # `block=[1,1,1]` (a single-thread launch) keeps `total_threads=1`
        # well clear of every declared activation/attention/parameter tier
        # for `clip-text`.
        entry = {"kernel": "bmul_f32", "grid": [1, 1, 1], "block": [1, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_badd_at_the_gelu_shape_lands_bias_residual_mlp_not_gelu(self):
        """`badd_f32` is never GELU-shape-gated (it is not in
        `GELU_SHAPE_KERNEL_NAMES`), so at the MLP tier it lands the
        tier-suffixed `BIAS/RESIDUAL-MLP` bucket (pass 3, finding 2) — the
        exclusion itself (bias-add is not the activation) is unchanged."""
        entry = {"kernel": "badd_f32", "grid": [3696, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.CHAIN_BIAS_RESIDUAL_MLP,
        )

    def test_attn_reduction_kernels_match_at_declared_row_count(self):
        rows = self.sig.attn_softmax_rows()
        self.assertEqual(rows, 14784)
        at_rows = {"kernel": "fast_max_f32", "grid": [14784, 1, 1], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(at_rows, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

    def test_fast_sum_at_neither_softmax_nor_ln_rows_is_loss_reduce(self):
        other_rows = {"kernel": "fast_sum_f32", "grid": [1, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(other_rows, self.sig, "clip-text"), attribute.CHAIN_LOSS_REDUCE
        )

    def test_fast_sum_at_ln_row_count_is_c_ln(self):
        entry = {"kernel": "fast_sum_f32", "grid": [self.sig.ln_row_count(), 1, 1], "block": [512, 1, 1]}
        self.assertEqual(attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_LN)

    def test_attn_reduction_requires_a_1d_grid(self):
        entry = {"kernel": "fast_max_f32", "grid": [14784, 2, 1], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_LOSS_REDUCE
        )

    def test_ln_eager_extended_kernels_require_ln_disabled(self):
        """Pass 3, finding 2: `usqrt_f32`/`urecip_f32`/`bsub_f32`/`usqr_f32`
        at the LN row-count shape only land `C-LN` when `ln_disabled=True`
        — on an LN-FUSED leg (the default), they fall through to whatever
        ELSE their shape matches (here: nothing else — `grid=[ln_row_count,
        1,1], block=[1,1,1]` is chosen to fall clear of every OTHER
        declared shape, including the parameter-scale tiers, so a
        `ln_disabled=False` miss is unambiguous)."""
        entry = {"kernel": "urecip_f32", "grid": [self.sig.ln_row_count(), 1, 1], "block": [1, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text", ln_disabled=True),
            attribute.CHAIN_LN,
        )
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text", ln_disabled=False))

    def test_ln_eager_extended_default_is_ln_disabled_false(self):
        """`ln_disabled` defaults to `False` — a caller that forgets to
        pass it never accidentally over-attributes to `C-LN`."""
        entry = {"kernel": "usqr_f32", "grid": [self.sig.ln_row_count(), 1, 1], "block": [1, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_ln_eager_extended_kernel_off_the_shape_falls_through_to_grad_bookkeeping(self):
        """A DIFFERENT grid for the same names (`grid=[2,1,1],
        block=[1024,1,1]`, `total_threads=2048`) happens to ALSO sit inside
        the `3*width=1536` parameter-scale tile's block-rounded range for
        `clip-text` — when `ln_disabled=False` this row legitimately falls
        through to the `GRAD-BOOKKEEPING` catch-all instead of `C-LN`, an
        honest outcome (not `UNATTRIBUTED`), not asserted as a bug. Pass 3
        (finding 8): this catch-all is `GRAD-BOOKKEEPING`, NOT `OPTIMIZER`
        — `OPTIMIZER` names `adamw_*` kernels only now."""
        entry = {"kernel": "urecip_f32", "grid": [2, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text", ln_disabled=False),
            attribute.CHAIN_GRAD_BOOKKEEPING,
        )

    def test_ampere_sgemm_matches_attn_when_grid_carries_rows_times_heads_at_position_2(self):
        batch = self.sig.attn_batch_count()
        self.assertEqual(batch, 192)
        entry = {"kernel": "ampere_sgemm_128x128_nt", "grid": [1, 1, 192], "block": [256, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

    def test_batch_count_at_grid_position_0_is_a_negative_control(self):
        """Pass 3, finding 1: the batched-attention rule is gated on grid
        POSITION 2 specifically — a row carrying `192` at `grid[0]` (a
        real value for some base-projection tile grids at OTHER shapes)
        must NOT be swept into `C-ATTN-<tower>` by a naive "192 anywhere in
        the grid" membership test."""
        entry = {"kernel": "some_base_projection_gemm", "grid": [192, 4, 1], "block": [128, 1, 1]}
        result = attribute.classify_kernel(entry, self.sig, "clip-text")
        self.assertNotEqual(result, attribute.chain_attn("clip-text"))
        # It IS still GEMM-family by name (contains "gemm") -> BASE-GEMM.
        self.assertEqual(result, attribute.CHAIN_BASE_GEMM)

    def test_anonymous_kernel_carrying_the_batch_count_at_position_2_is_attn(self):
        """The batched-grid rule is NAME-INDEPENDENT (module doc) — an
        anonymous `Kernel2` row carrying `rows*heads` at grid POSITION 2
        still lands `C-ATTN-<tower>`."""
        entry = {"kernel": "Kernel2", "grid": [2, 1, 192], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

    def test_anonymous_kernel_carrying_the_batch_count_at_position_0_is_a_negative_control(self):
        """Same negative control as
        `test_batch_count_at_grid_position_0_is_a_negative_control`, but
        for a NAME-INDEPENDENT anonymous kernel — the position-2 gate
        applies identically regardless of whether the kernel has a name."""
        entry = {"kernel": "Kernel2", "grid": [192, 1, 4], "block": [128, 1, 1]}
        result = attribute.classify_kernel(entry, self.sig, "clip-text")
        self.assertNotEqual(result, attribute.chain_attn("clip-text"))

    def test_ampere_sgemm_without_the_batch_count_is_base_gemm(self):
        """Pass-2 behavior change: a base/LoRA Linear projection's plain
        2-D matmul now lands the NAMED `BASE-GEMM` bucket instead of
        pass 1's `UNATTRIBUTED`."""
        entry = {"kernel": "ampere_sgemm_128x64_nn", "grid": [4, 29, 7], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_BASE_GEMM
        )

    def test_split_k_reduce_is_base_gemm(self):
        entry = {"kernel": "splitKreduce_kernel", "grid": [1, 116, 1], "block": [32, 16, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_BASE_GEMM
        )

    def test_anonymous_kernel_at_a_genuine_3d_tile_grid_is_base_gemm(self):
        """Pass 3, finding 3: an anonymous kernel with EVERY grid dimension
        `>1` (a plausible M-tile x N-tile x batch/split-K launch) is
        `BASE-GEMM` via the grid-family fallback alone."""
        entry = {"kernel": "Kernel2", "grid": [8, 2, 28], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_BASE_GEMM
        )

    def test_named_kernel_at_a_3d_tile_grid_never_lands_base_gemm_via_the_fallback(self):
        """The anonymous grid-family fallback is restricted to
        `Kernel<N>`/`magma_*` names ONLY — a NAMED kernel at a genuine 3-D
        tile grid (every dimension `>1`, product well above the magnitude
        floor) must NOT be routed here just because its grid happens to
        look tile-shaped. These are real shapes, never observed
        classifying this way."""
        for name, grid in (
            ("badd_f32", [2, 2, 278]),
            ("ucopy_f32", [2, 2, 231]),
            ("usqrt_f32", [3, 3, 8]),
        ):
            product = grid[0] * grid[1] * grid[2]
            self.assertGreaterEqual(product, attribute.GEMM_TILE_GRID_PRODUCT_FLOOR, (name, grid))
            entry = {"kernel": name, "grid": grid, "block": [1, 1, 1]}
            result = attribute.classify_kernel(entry, self.sig, "clip-text")
            self.assertNotEqual(result, attribute.CHAIN_BASE_GEMM, (name, grid))

    def test_anonymous_kernel_below_the_magnitude_floor_is_not_base_gemm(self):
        """Pass 3, finding 4: a magnitude floor (`GEMM_TILE_GRID_PRODUCT_
        FLOOR`) additionally excludes a near-scalar anonymous 3-D grid —
        `[2,2,2]` (product `8`, every dimension `>1`) is far below the
        smallest real evidence (`[8,2,28]`, product `448`)."""
        entry = {"kernel": "Kernel2", "grid": [2, 2, 2], "block": [1, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_anonymous_grid_fallback_runs_after_the_activation_tier_check(self):
        """Pass 3, finding 4: the anonymous grid-family fallback is placed
        AFTER the activation-tier checks — an anonymous kernel whose
        launch ALSO happens to cover an activation tier's own element
        count lands the TIER bucket, never `BASE-GEMM`, even though its
        grid is a genuine, floor-clearing 3-D tile (`[4,3,77]`,
        product=`924`, `924*1024` covers `out_shape_elements=946,176`)."""
        entry = {"kernel": "Kernel2", "grid": [4, 3, 77], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.CHAIN_ELEMENTWISE_OTHER_OUT,
        )

    def test_anonymous_kernel_with_a_degenerate_dimension_is_not_base_gemm(self):
        """Negative control for the above: a `1` in ANY position keeps an
        anonymous kernel OUT of the grid-family GEMM fallback (module doc:
        a genuine tile grid never degenerates to size 1 in any dimension)."""
        for grid in ([16, 1, 10], [4, 1, 24], [12, 1, 8], [128, 2, 1]):
            entry = {"kernel": "Kernel2", "grid": grid, "block": [128, 1, 1]}
            self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"), grid)

    def test_attn_tensor_elementwise_ops_land_attn(self):
        elements = self.sig.attn_shape_elements()
        entry = {"kernel": "badd_f32", "grid": [1112, 1, 1], "block": [1024, 1, 1]}
        self.assertLessEqual(elements, 1112 * 1024)
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.chain_attn("clip-text")
        )

    def test_cast_named_kernel_at_attn_shape_is_attn_not_cast(self):
        """Pass 3, finding 1: `CAST`'s own priority is now SHAPE-GATED — a
        cast row at the attention tensor's own element count lands
        `C-ATTN-<tower>`, not the generic `CAST` bucket (pass 2's bug: the
        name-only check returned before the shape check ever ran)."""
        entry = {"kernel": "cast_u8_f32", "grid": [1112, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.chain_attn("clip-text")
        )

    def test_cast_named_kernel_at_any_other_shape_is_cast(self):
        entry = {"kernel": "cast_f32_f32", "grid": [3696, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_CAST)

    def test_optimizer_by_name(self):
        entry = {"kernel": "adamw_moment_update_f32", "grid": [4, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_OPTIMIZER
        )

    def test_dropout_by_name_any_shape(self):
        entry = {"kernel": "dropout_fwd_f32", "grid": [3696, 1, 1], "block": [256, 1, 1]}
        self.assertEqual(attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_DROPOUT)

    def test_embed_gather_by_name_beats_activation_tier_fallback(self):
        entry = {"kernel": "is_u32_f32", "grid": [924, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_EMBED_GATHER
        )

    def test_grad_bookkeeping_parameter_scale_by_shape_requires_1d_grid(self):
        """Pass 3 (finding 8): the parameter-scale catch-all is named
        `GRAD-BOOKKEEPING`, not `OPTIMIZER` — `adamw_step_fused` is already
        admitted on A legs by NAME, so these shape-coincident rows are NOT
        the optimizer."""
        elements = sorted(self.sig.param_scale_elements())[0]
        entry = {"kernel": "usqrt_f32", "grid": [1, 1, 1], "block": [max(elements, 1024), 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_GRAD_BOOKKEEPING
        )

    def test_grad_bookkeeping_parameter_scale_negative_control_non_1d_grid(self):
        """Pass 3, finding 3: a kernel whose TOTAL THREAD COUNT coincides
        with a parameter-scale element count but whose grid is NOT 1-D
        (`grid=[ceil(N/b),1,1]`) must NOT land `GRAD-BOOKKEEPING` —
        evidenced by `clip-text-A2`'s anonymous `Kernel2 grid=[4,1,24]`
        (`total_threads=12,288 == LORA_RANK*3*width`, yet a 3-D tiled
        launch, not a bookkeeping op)."""
        elements = sorted(self.sig.param_scale_elements())[0]
        entry = {"kernel": "some_unknown_kernel", "grid": [1, 1, 4], "block": [max(elements, 1024), 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_permute_reshape_bucket_by_name_at_any_tier_shape(self):
        """Pass 3, finding 1/2: `ucopy_*`/`copy2d_*` get their OWN bucket
        (not tier-suffixed, and never `C-ATTN`) at any of the three
        activation-tier shapes."""
        for elements, name in (
            (self.sig.qkv_shape_elements(), "ucopy_f32"),
            (self.sig.out_shape_elements(), "copy2d_f32"),
            (self.sig.gelu_shape_elements(), "ucopy_bf16"),
        ):
            grid0 = -(-elements // 1024)
            entry = {"kernel": name, "grid": [grid0, 1, 1], "block": [1024, 1, 1]}
            self.assertEqual(
                attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_PERMUTE_RESHAPE
            )

    def test_bias_residual_tier_suffixed_buckets(self):
        cases = [
            (self.sig.qkv_shape_elements(), attribute.CHAIN_BIAS_RESIDUAL_QKV),
            (self.sig.out_shape_elements(), attribute.CHAIN_BIAS_RESIDUAL_OUT),
            (self.sig.gelu_shape_elements(), attribute.CHAIN_BIAS_RESIDUAL_MLP),
        ]
        for elements, expected in cases:
            grid0 = -(-elements // 1024)
            entry = {"kernel": "badd_f32", "grid": [grid0, 1, 1], "block": [1024, 1, 1]}
            self.assertEqual(attribute.classify_kernel(entry, self.sig, "clip-text"), expected)

    def test_elementwise_other_catch_all_tier_suffixed_buckets(self):
        cases = [
            (self.sig.qkv_shape_elements(), attribute.CHAIN_ELEMENTWISE_OTHER_QKV),
            (self.sig.out_shape_elements(), attribute.CHAIN_ELEMENTWISE_OTHER_OUT),
            (self.sig.gelu_shape_elements(), attribute.CHAIN_ELEMENTWISE_OTHER_MLP),
        ]
        for elements, expected in cases:
            grid0 = -(-elements // 1024)
            entry = {"kernel": "const_set_f32", "grid": [grid0, 1, 1], "block": [1024, 1, 1]}
            self.assertEqual(attribute.classify_kernel(entry, self.sig, "clip-text"), expected)

    def test_patch_embed_only_on_clip_vision(self):
        vision_sig = clip_vision_signatures()
        entry = {"kernel": "im2col_f32", "grid": [1, 1, 1], "block": [1, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, vision_sig, "clip-vision"), attribute.CHAIN_PATCH_EMBED
        )
        self.assertNotEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_PATCH_EMBED
        )

    def test_gemm_family_grid_rule_generalizes_to_a_second_tower(self):
        vision_sig = clip_vision_signatures()
        self.assertEqual(vision_sig.attn_batch_count(), 288)
        entry = {"kernel": "magma_sgemmEx_kernel", "grid": [1, 2, 288], "block": [8, 8, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, vision_sig, "clip-vision"),
            attribute.chain_attn("clip-vision"),
        )
        entry_no_batch = {"kernel": "magma_sgemmEx_kernel", "grid": [1, 2, 4], "block": [8, 8, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry_no_batch, vision_sig, "clip-vision"), attribute.CHAIN_BASE_GEMM
        )


class AttributeCensusRealFixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-text-A1`
    16-row cut."""

    def setUp(self):
        self.census = load_fixture_census()
        self.sig = clip_text_signatures()

    def test_chains_partition_busy_exactly(self):
        chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])
        total_in_rows = sum(r["us_per_step"] for r in self.census["by_kernel_and_grid"])
        attributed_total = sum(
            c.gpu_busy_us for c in chains.values() if c.gpu_busy_us is not None
        )
        self.assertAlmostEqual(attributed_total, total_in_rows, places=6)

    def test_every_declared_chain_is_present(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        for name in attribute.declared_chains_for_tower("clip-text"):
            self.assertIn(name, chains, name)
        self.assertIn(attribute.CHAIN_UNATTRIBUTED, chains)
        # `C-LORA` is NEVER a chain-partition member (pass 3, finding 4).
        self.assertNotIn(attribute.CHAIN_LORA, chains)

    def test_shares_are_bounded_by_one(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        for name, result in chains.items():
            if result.share_gpu_busy is not None:
                self.assertGreaterEqual(result.share_gpu_busy, 0.0, name)
                self.assertLessEqual(result.share_gpu_busy, 1.0, name)
            if result.share_wall is not None:
                self.assertGreaterEqual(result.share_wall, 0.0, name)

    def test_ln_kernel_names_are_exactly_the_two_ln_kernels(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(
            set(chains[attribute.CHAIN_LN].kernel_names),
            {"layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"},
        )

    def test_gelu_kernel_names_exclude_badd(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        gelu_names = set(chains[attribute.CHAIN_GELU].kernel_names)
        self.assertEqual(gelu_names, {"usigmoid_f32", "affine_f32", "bmul_f32"})
        self.assertNotIn("badd_f32", gelu_names)

    def test_attn_kernel_names(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        attn_names = set(chains[attribute.chain_attn("clip-text")].kernel_names)
        self.assertEqual(
            attn_names,
            {
                "fast_max_f32",
                "fast_sum_f32",
                "ampere_sgemm_128x128_nt",
                "ampere_sgemm_128x128_tn",
                "ampere_sgemm_128x128_nn",
                "bmul_f32",
            },
        )

    def test_badd_and_ampere_now_land_named_buckets_not_unattributed(self):
        """Pass-2 regression guard: the exact two rows pass 1's fixture
        used as UNATTRIBUTED negative controls now land NAMED buckets."""
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("badd_f32", chains[attribute.CHAIN_BIAS_RESIDUAL_MLP].kernel_names)
        self.assertIn("ampere_sgemm_128x64_nn", chains[attribute.CHAIN_BASE_GEMM].kernel_names)
        self.assertIn("dropout_fwd_f32", chains[attribute.CHAIN_DROPOUT].kernel_names)
        self.assertIn("adamw_moment_update_f32", chains[attribute.CHAIN_OPTIMIZER].kernel_names)

    def test_unattributed_share_is_small_on_the_real_fixture(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        unattributed = chains[attribute.CHAIN_UNATTRIBUTED]
        self.assertIsNotNone(unattributed.gpu_busy_us)
        self.assertGreaterEqual(unattributed.gpu_busy_us, 0.0)

    def test_empty_census_leaves_declared_chains_absent(self):
        empty = {
            "gpu_kernel_us_per_step": 0.0,
            "wall_s_per_step": 0.0,
            "by_kernel_and_grid": [],
        }
        chains, unknown, reasons = attribute.attribute_census(empty, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])
        for name in attribute.declared_chains_for_tower("clip-text"):
            self.assertEqual(chains[name].status, "absent", name)
            self.assertIsNone(chains[name].gpu_busy_us)
        self.assertEqual(chains[attribute.CHAIN_UNATTRIBUTED].status, "measured")
        self.assertEqual(chains[attribute.CHAIN_UNATTRIBUTED].gpu_busy_us, 0.0)

    def test_outside_signature_plausibly_attention_mirrors_permute_reshape(self):
        """This fixture carries a real `ucopy_f32` row at the `out` tier
        shape (added pass 3 — see `PROVENANCE.md`), so `PERMUTE/RESHAPE` is
        `measured`, not `absent`, here."""
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        permute = chains[attribute.CHAIN_PERMUTE_RESHAPE]
        self.assertEqual(permute.status, "measured")
        info = attribute.outside_signature_plausibly_attention(chains)
        self.assertEqual(info["busy_us"], permute.gpu_busy_us)
        self.assertEqual(info["share_wall"], permute.share_wall)

    def test_outside_signature_plausibly_attention_defaults_to_zero_when_absent(self):
        empty = {"gpu_kernel_us_per_step": 0.0, "wall_s_per_step": 1.0, "by_kernel_and_grid": []}
        chains, _u, _r = attribute.attribute_census(empty, self.sig, "clip-text")
        info = attribute.outside_signature_plausibly_attention(chains)
        self.assertEqual(info, {"busy_us": 0.0, "share_wall": 0.0})


class AttributeCensusBf16FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-text-A2`
    (BF16) 22-row cut — see `fixtures/profile_421_clip_text_a2/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_A2)
        self.sig = attribute.derive_signatures("clip-text", CLIP_TEXT_A2_MANIFEST_FIELDS)

    def test_chains_partition_busy_exactly(self):
        chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        total_in_rows = sum(r["us_per_step"] for r in self.census["by_kernel_and_grid"])
        attributed_total = sum(c.gpu_busy_us for c in chains.values() if c.gpu_busy_us is not None)
        self.assertAlmostEqual(attributed_total, total_in_rows, places=6)

    def test_ln_gelu_and_attn_bf16_names(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(
            set(chains[attribute.CHAIN_LN].kernel_names),
            {"layer_norm_fwd_bf16_biased", "layer_norm_bwd_dx_bf16"},
        )
        self.assertIn("usigmoid_bf16", chains[attribute.CHAIN_GELU].kernel_names)
        self.assertIn("affine_bf16", chains[attribute.CHAIN_GELU].kernel_names)
        self.assertIn("affine_bf16", chains[attribute.chain_attn("clip-text")].kernel_names)

    def test_loss_reduce_bucket_present(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("fast_sum_bf16", chains[attribute.CHAIN_LOSS_REDUCE].kernel_names)

    def test_anonymous_kernel2_splits_by_grid_and_leaves_one_unattributed(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("Kernel2", chains[attribute.chain_attn("clip-text")].kernel_names)
        self.assertIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_cast_bucket_includes_bf16_only_names_but_not_the_attn_shape_row(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        cast_names = set(chains[attribute.CHAIN_CAST].kernel_names)
        self.assertIn("cast_bf16_f32", cast_names)
        self.assertIn("cast_add_bf16", cast_names)
        self.assertIn("cast_scale_bf16_f32", cast_names)
        # Pass 3, finding 1: `cast_u8_bf16` at the attention shape now lands
        # `C-ATTN-clip-text`, NOT `CAST` — see the fixture's own PROVENANCE.
        self.assertNotIn("cast_u8_bf16", cast_names)
        self.assertIn("cast_u8_bf16", chains[attribute.chain_attn("clip-text")].kernel_names)

    def test_optimizer_and_dropout_are_dtype_independent(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("adamw_moment_update_f32", chains[attribute.CHAIN_OPTIMIZER].kernel_names)
        self.assertIn("dropout_fwd_f32", chains[attribute.CHAIN_DROPOUT].kernel_names)

    def test_embed_gather_bf16(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        embed_names = set(chains[attribute.CHAIN_EMBED_GATHER].kernel_names)
        self.assertIn("gather_u32_bf16", embed_names)
        self.assertIn("is_u32_bf16", embed_names)

    def test_bf16_gemm_tile_variant_lands_base_gemm(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn(
            "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nn",
            chains[attribute.CHAIN_BASE_GEMM].kernel_names,
        )


class UnknownKernelGateOnRealFixtureTests(unittest.TestCase):
    """Pass 3, finding 3: `clip-text-A2`'s own real anonymous `Kernel2`
    rows now INVALIDATE the leg (the `"gemm"`-substring/`Kernel2`-literal
    admission shortcuts are removed) — the exact, committed reproduction of
    the real leg's own measured outcome (module doc, "Consequence,
    measured")."""

    def test_a2_fixture_has_an_invalidating_unknown_kernel(self):
        census = load_fixture_census(FIXTURE_KERNELS_A2)
        sig = attribute.derive_signatures("clip-text", CLIP_TEXT_A2_MANIFEST_FIELDS)
        _chains, unknown, reasons = attribute.attribute_census(census, sig, "clip-text")
        unknown_names = {(u["kernel"], tuple(u["grid"])) for u in unknown}
        self.assertIn(("Kernel2", (16, 1, 10)), unknown_names)
        self.assertTrue(any("Kernel2" in r for r in reasons))
        self.assertTrue(any("1, 10" in r or "[16, 1, 10]" in r for r in reasons))


class Ln1VsD2DifferentialTests(unittest.TestCase):
    """The module doc's own "D1-vs-D2 differential: the measured split"
    section states the mechanism: eager LN's own `gamma*x_hat+beta` affine
    lands as an ordinary `badd_f32` at the `out` tier, whose
    `launches_per_step` forms a three-point ladder as fusion is
    progressively disabled — NOT run-to-run cuBLAS/launch-count session
    noise (see `test_badd_ladder_launches_per_step_by_tower`, which
    asserts each tower's own ladder point straight off the committed
    fixtures). This test asserts EXACTLY what that doc section states:
    toggling `layer_norm_fused` off (`D1`) moves MASS INTO `C-LN` relative
    to the fused twin (`D2`) — a positive delta — and BOTH leak buckets
    (`BIAS/RESIDUAL-OUT`, `ELEMENTWISE-OTHER-OUT`) are NAMED as also
    moving (also positive), with NO "less than `C-LN`" claim for either
    (they measurably move MORE). `BASE-GEMM`/`C-ATTN-<tower>` remain the
    honest LN-ORTHOGONAL negative control (their own rules never read
    `ln_disabled`, so their delta stays near zero relative to `C-LN`'s
    own, non-trivial move) — this IS still a "less than" comparison, but
    only for buckets this module's own rules guarantee are orthogonal, not
    for the two buckets known to leak."""

    def _chain_busy(self, chains: dict, name: str) -> float:
        entry = chains.get(name)
        if entry is None or entry.gpu_busy_us is None:
            return 0.0
        return entry.gpu_busy_us

    def _assert_differential(self, tower, sig, d1_census, d2_census):
        d1_chains, _u1, _r1 = attribute.attribute_census(d1_census, sig, tower, ln_disabled=True)
        d2_chains, _u2, _r2 = attribute.attribute_census(d2_census, sig, tower, ln_disabled=False)

        def delta(name: str) -> float:
            return self._chain_busy(d1_chains, name) - self._chain_busy(d2_chains, name)

        ln_delta = delta(attribute.CHAIN_LN)
        # Toggling `layer_norm_fused` off (D1) must move MASS INTO `C-LN`
        # relative to the fused twin (D2) — a positive delta.
        self.assertGreater(ln_delta, 0.0)

        # The two NAMED leak buckets also move — POSITIVE, never claimed
        # smaller than `ln_delta` (module doc: they measurably move MORE).
        self.assertGreater(delta(attribute.CHAIN_BIAS_RESIDUAL_OUT), 0.0)
        self.assertGreater(delta(attribute.CHAIN_ELEMENTWISE_OTHER_OUT), 0.0)

        # Genuinely LN-orthogonal buckets (their own rules never read
        # `ln_disabled`) move by LESS than `C-LN` itself.
        orthogonal_buckets = (attribute.CHAIN_BASE_GEMM, attribute.chain_attn(tower))
        for bucket in orthogonal_buckets:
            self.assertLess(abs(delta(bucket)), ln_delta, bucket)

    def test_clip_text_d1_vs_d2(self):
        sig = attribute.derive_signatures("clip-text", CLIP_TEXT_D1_MANIFEST_FIELDS)
        self._assert_differential(
            "clip-text", sig, load_fixture_census(FIXTURE_KERNELS_D1), load_fixture_census(FIXTURE_KERNELS_D2)
        )

    def test_clip_vision_d1_vs_d2(self):
        sig = attribute.derive_signatures("clip-vision", CLIP_VISION_D1_MANIFEST_FIELDS)
        self._assert_differential(
            "clip-vision",
            sig,
            load_fixture_census(FIXTURE_KERNELS_VISION_D1),
            load_fixture_census(FIXTURE_KERNELS_VISION_D2),
        )

    def test_badd_ladder_launches_per_step_by_tower(self):
        """`badd_f32`'s own `launches_per_step` at the `out` tier grid
        forms a three-point ladder (A1 -> D2 -> D1) as fusion is
        progressively disabled — read straight off the six committed
        fixture cuts, never a transcribed literal. `clip-text`'s own
        ladder point at `D1` and `clip-vision`'s own ladder point at `D1`
        are DIFFERENT numbers (874 vs 865) — this is real, per-tower
        architecture, never claimed identical across towers."""
        text_a1 = badd_launches_at_grid0(load_fixture_census(FIXTURE_KERNELS), 924)
        text_d2 = badd_launches_at_grid0(load_fixture_census(FIXTURE_KERNELS_D2), 924)
        text_d1 = badd_launches_at_grid0(load_fixture_census(FIXTURE_KERNELS_D1), 924)
        self.assertEqual((text_a1, text_d2, text_d1), (337.0, 633.0, 874.0))

        vision_a1 = badd_launches_at_grid0(load_fixture_census(FIXTURE_KERNELS_VISION_A1), 900)
        vision_d2 = badd_launches_at_grid0(load_fixture_census(FIXTURE_KERNELS_VISION_D2), 900)
        vision_d1 = badd_launches_at_grid0(load_fixture_census(FIXTURE_KERNELS_VISION_D1), 900)
        self.assertEqual((vision_a1, vision_d2, vision_d1), (337.0, 633.0, 865.0))


class AttributeCensusD1FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-text-D1`
    13-row cut — see `fixtures/profile_421_clip_text_d1/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_D1)
        self.sig = attribute.derive_signatures("clip-text", CLIP_TEXT_D1_MANIFEST_FIELDS)

    def _chains(self):
        chains, unknown, reasons = attribute.attribute_census(
            self.census, self.sig, "clip-text", ln_disabled=True
        )
        return chains, unknown, reasons

    def test_eager_ln_rows_land_c_ln(self):
        chains, _u, _r = self._chains()
        ln_names = set(chains[attribute.CHAIN_LN].kernel_names)
        self.assertIn("fast_sum_f32", ln_names)
        self.assertIn("usqrt_f32", ln_names)
        self.assertIn("urecip_f32", ln_names)

    def test_fast_sum_splits_three_ways_by_grid(self):
        chains, _u, _r = self._chains()
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LN].kernel_names)
        self.assertIn("fast_sum_f32", chains[attribute.chain_attn("clip-text")].kernel_names)
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LOSS_REDUCE].kernel_names)

    def test_d_leg_tile_variants_are_base_gemm_or_attn_by_grid(self):
        chains, _u, _r = self._chains()
        base_gemm_names = set(chains[attribute.CHAIN_BASE_GEMM].kernel_names)
        self.assertIn("ampere_sgemm_128x64_nt", base_gemm_names)
        self.assertIn("ampere_sgemm_32x32_sliced1x4_nt", base_gemm_names)
        self.assertIn("ampere_sgemm_128x128_nn", chains[attribute.chain_attn("clip-text")].kernel_names)

    def test_anonymous_3d_tile_grid_lands_base_gemm_not_unattributed(self):
        """Pass 3, finding 3 (corrects pass 2): `Kernel2 grid=[8,2,28]`
        (every dimension `>1`) now lands `BASE-GEMM` — corroborated by the
        IDENTICAL grid on the independent `clip-text-D2` fixture."""
        chains, _u, _r = self._chains()
        self.assertIn("Kernel2", chains[attribute.CHAIN_BASE_GEMM].kernel_names)
        self.assertNotIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_usqr_at_out_tier_lands_elementwise_other_out_not_c_ln(self):
        """`usqr_f32` at `grid=[924,1,1]` sits at the `out` ACTIVATION tier
        shape, not `ln_row_count` — it is NOT swept into `C-LN` by a
        launch-count coincidence (see `clip-vision-d1`'s fixture for the
        real evidence that `usqr` DOES land `C-LN` at the correct shape)."""
        chains, _u, _r = self._chains()
        self.assertIn("usqr_f32", chains[attribute.CHAIN_ELEMENTWISE_OTHER_OUT].kernel_names)
        self.assertNotIn("usqr_f32", chains[attribute.CHAIN_LN].kernel_names)

    def test_no_unknown_kernel_names(self):
        _chains, unknown, reasons = self._chains()
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])


class AttributeCensusD2FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-text-D2`
    13-row cut — `clip-text-D1`'s twin (`layer_norm_fused` FUSED here)."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_D2)
        self.sig = attribute.derive_signatures("clip-text", CLIP_TEXT_D2_MANIFEST_FIELDS)

    def _chains(self):
        return attribute.attribute_census(self.census, self.sig, "clip-text", ln_disabled=False)

    def test_ln_fused_rows_land_c_ln_by_name(self):
        chains, _u, _r = self._chains()
        self.assertEqual(
            set(chains[attribute.CHAIN_LN].kernel_names),
            {"layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"},
        )

    def test_shared_anonymous_grid_with_d1_is_base_gemm_here_too(self):
        chains, _u, _r = self._chains()
        self.assertIn("Kernel2", chains[attribute.CHAIN_BASE_GEMM].kernel_names)

    def test_no_unknown_kernel_names(self):
        _chains, unknown, reasons = self._chains()
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])


class AttributeCensusVisionA1FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-vision-A1`
    21-row cut — see `fixtures/profile_421_clip_vision_a1/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_VISION_A1)
        self.sig = clip_vision_signatures()

    def test_chains_partition_busy_exactly(self):
        chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-vision")
        total_in_rows = sum(r["us_per_step"] for r in self.census["by_kernel_and_grid"])
        attributed_total = sum(c.gpu_busy_us for c in chains.values() if c.gpu_busy_us is not None)
        self.assertAlmostEqual(attributed_total, total_in_rows, places=6)

    def test_patch_embed_bucket(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("im2col_f32", chains[attribute.CHAIN_PATCH_EMBED].kernel_names)

    def test_magma_gemm_lands_attn_via_grid_not_name(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("magma_sgemmEx_kernel", chains[attribute.chain_attn("clip-vision")].kernel_names)

    def test_badd_splits_three_bias_residual_tiers(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("badd_f32", chains[attribute.CHAIN_BIAS_RESIDUAL_OUT].kernel_names)
        self.assertIn("badd_f32", chains[attribute.CHAIN_BIAS_RESIDUAL_QKV].kernel_names)
        self.assertIn("badd_f32", chains[attribute.CHAIN_BIAS_RESIDUAL_MLP].kernel_names)
        self.assertIn("badd_f32", chains[attribute.chain_attn("clip-vision")].kernel_names)

    def test_urecip_off_ln_shape_lands_grad_bookkeeping_via_the_param_scale_catch_all(self):
        """Pass 3 (finding 8): the parameter-scale catch-all this
        `urecip_f32` row (`grid=[1,1,1]`, `total_threads` inside the
        `width=768` tier's block-rounded range) falls into is
        `GRAD-BOOKKEEPING`, not `OPTIMIZER` — `OPTIMIZER` names `adamw_*`
        kernels by name only now."""
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("urecip_f32", chains[attribute.CHAIN_GRAD_BOOKKEEPING].kernel_names)

    def test_every_declared_chain_present_including_patch_embed(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        for name in attribute.declared_chains_for_tower("clip-vision"):
            self.assertIn(name, chains, name)
        self.assertIn(attribute.CHAIN_PATCH_EMBED, chains)

    def test_kernel2_is_unknown_but_below_threshold_never_invalidates(self):
        """Pass 3, finding 3: `Kernel2 grid=[6,1,18]` (`dim[1]=1`, fails the
        anonymous-GEMM-tile rule) is genuinely unknown now, but its
        `share_gpu_busy` against the leg's REAL full busy total stays well
        under `UNKNOWN_KERNEL_SHARE_LIMIT` — recorded, never invalidating."""
        _chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["kernel"], "Kernel2")
        self.assertLess(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)
        self.assertEqual(reasons, [])


class AttributeCensusVisionD1FixtureTests(unittest.TestCase):
    """`clip-vision-D1`'s own fixture — the ONLY real evidence across all
    eight pulled CLIP legs for `usqr` at the LN row count (pass 3,
    finding 2)."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_VISION_D1)
        self.sig = attribute.derive_signatures("clip-vision", CLIP_VISION_D1_MANIFEST_FIELDS)

    def _chains(self):
        return attribute.attribute_census(self.census, self.sig, "clip-vision", ln_disabled=True)

    def test_usqr_at_ln_row_count_lands_c_ln(self):
        chains, _u, _r = self._chains()
        self.assertIn("usqr_f32", chains[attribute.CHAIN_LN].kernel_names)

    def test_bsub_usqrt_urecip_at_ln_row_count_land_c_ln(self):
        chains, _u, _r = self._chains()
        for name in ("bsub_f32", "usqrt_f32", "urecip_f32"):
            self.assertIn(name, chains[attribute.CHAIN_LN].kernel_names)

    def test_fast_sum_at_ln_row_count_lands_c_ln(self):
        chains, _u, _r = self._chains()
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LN].kernel_names)

    def test_magma_and_ampere_land_attn_via_grid_position_2(self):
        chains, _u, _r = self._chains()
        attn_names = set(chains[attribute.chain_attn("clip-vision")].kernel_names)
        self.assertIn("magma_sgemmEx_kernel", attn_names)
        self.assertIn("ampere_sgemm_128x128_nt", attn_names)

    def test_anonymous_row_with_a_degenerate_dimension_stays_unattributed(self):
        """A SECOND tower's evidence that the anonymous-GEMM-tile rule does
        not loosen: `Kernel2 grid=[6,1,18]` (`dim[1]=1`) stays
        `UNATTRIBUTED` here exactly as `clip-vision-d2`'s fixture shows."""
        chains, _u, _r = self._chains()
        self.assertIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_kernel2_is_unknown_but_below_threshold_never_invalidates(self):
        """`Kernel2 grid=[6,1,18]` is genuinely unknown now (pass 3,
        finding 3), but its `share_gpu_busy` against the leg's REAL full
        busy total is well under `UNKNOWN_KERNEL_SHARE_LIMIT` — recorded,
        never invalidating."""
        _chains, unknown, reasons = self._chains()
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["kernel"], "Kernel2")
        self.assertLess(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)
        self.assertEqual(reasons, [])


class AttributeCensusVisionD2FixtureTests(unittest.TestCase):
    """`clip-vision-D2`'s own fixture — `clip-vision-D1`'s LN-fused twin."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_VISION_D2)
        self.sig = attribute.derive_signatures("clip-vision", CLIP_VISION_D2_MANIFEST_FIELDS)

    def _chains(self):
        return attribute.attribute_census(self.census, self.sig, "clip-vision", ln_disabled=False)

    def test_ln_fused_rows_land_c_ln_by_name(self):
        chains, _u, _r = self._chains()
        self.assertEqual(
            set(chains[attribute.CHAIN_LN].kernel_names),
            {"layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"},
        )

    def test_no_row_lands_c_ln_via_the_eager_extended_names(self):
        """`usqr`/`bsub`/`usqrt`/`urecip` never appear at `ln_row_count` on
        this LN-FUSED leg's real export — nothing beyond the two by-name LN
        kernels should be in `C-LN`'s own `kernel_names`."""
        chains, _u, _r = self._chains()
        self.assertEqual(
            set(chains[attribute.CHAIN_LN].kernel_names),
            {"layer_norm_fwd_f32_biased", "layer_norm_bwd_dx_f32"},
        )

    def test_anonymous_row_with_a_degenerate_dimension_stays_unattributed(self):
        chains, _u, _r = self._chains()
        self.assertIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_kernel2_is_unknown_but_below_threshold_never_invalidates(self):
        _chains, unknown, reasons = self._chains()
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["kernel"], "Kernel2")
        self.assertLess(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)
        self.assertEqual(reasons, [])


class UnknownKernelGateTests(unittest.TestCase):
    def setUp(self):
        self.sig = clip_text_signatures()

    def _census_with(self, extra_row: dict) -> dict:
        rows = list(load_fixture_census()["by_kernel_and_grid"])
        rows.append(extra_row)
        total = sum(r["us_per_step"] for r in rows)
        return {
            "gpu_kernel_us_per_step": total,
            "wall_s_per_step": 0.1,
            "by_kernel_and_grid": rows,
        }

    def test_unknown_kernel_above_threshold_invalidates(self):
        census = self._census_with(
            {"kernel": "totally_new_kernel_f32", "grid": [1, 1, 1], "block": [1, 1, 1], "us_per_step": 1e9}
        )
        chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertTrue(any("totally_new_kernel_f32" in r for r in reasons))
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["kernel"], "totally_new_kernel_f32")
        self.assertGreater(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)
        self.assertIn("totally_new_kernel_f32", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_unknown_kernel_below_threshold_is_recorded_but_not_invalidating(self):
        census = self._census_with(
            {"kernel": "tiny_new_kernel_f32", "grid": [1, 1, 1], "block": [1, 1, 1], "us_per_step": 0.0001}
        )
        chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertEqual(reasons, [])
        self.assertEqual(len(unknown), 1)
        self.assertLess(unknown[0]["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)

    def test_a_bf16_twin_of_a_known_name_never_triggers_the_gate(self):
        census = self._census_with(
            {"kernel": "badd_bf16", "grid": [1, 1, 1], "block": [1, 1, 1], "us_per_step": 1e9}
        )
        _chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])

    def test_a_novel_gemm_named_kernel_still_classifies_by_name_regardless_of_admission(self):
        """Pass 3, finding 3 removes the `"gemm"`-substring ADMISSION
        shortcut from `is_known_kernel_name`, but `classify_kernel`'s own
        `GEMM_NAME_RE`-by-name CLASSIFICATION rule is UNCHANGED and
        unconditional on grid shape — an unrecognised GEMM-NAMED kernel
        still lands `BASE-GEMM` by name alone (classification and
        admission are different questions, module doc), so it NEVER
        reaches the unknown-kernel gate at all (that gate only sees rows
        `classify_kernel` could not place)."""
        census = self._census_with(
            {
                "kernel": "ampere_bf16_s16816gemm_bf16_999x999_ldg8_f2f_stages_99x9_nn",
                "grid": [1, 1, 1],
                "block": [1, 1, 1],
                "us_per_step": 1e9,
            }
        )
        chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertIn(
            "ampere_bf16_s16816gemm_bf16_999x999_ldg8_f2f_stages_99x9_nn",
            chains[attribute.CHAIN_BASE_GEMM].kernel_names,
        )
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])

    def test_an_unnamed_kernel_at_a_non_qualifying_grid_still_triggers_the_gate(self):
        """The REAL invalidating case (pass 3, finding 3): an ANONYMOUS
        name (no `"gemm"` substring to classify by, and a grid that fails
        the anonymous-tile-grid fallback) is genuinely unknown and, above
        the share threshold, invalidates — the committed reproduction of
        `clip-text-A2`'s own real `Kernel2` rows (see
        `UnknownKernelGateOnRealFixtureTests`)."""
        census = self._census_with(
            {"kernel": "Kernel2", "grid": [16, 1, 10], "block": [128, 1, 1], "us_per_step": 1e9}
        )
        _chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["kernel"], "Kernel2")
        self.assertTrue(reasons)


class LegRoleTests(unittest.TestCase):
    def test_a_leg_has_no_disabled_keys(self):
        self.assertEqual(attribute.leg_role([], "clip-text"), "A")

    def test_d2_leg_disables_only_lora_linear_fused(self):
        self.assertEqual(attribute.leg_role(["lora_linear_fused"], "clip-text"), "D2")

    def test_d1_leg_is_per_tower(self):
        self.assertEqual(
            attribute.leg_role(["lora_linear_fused", "layer_norm_fused"], "clip-text"), "D1"
        )
        self.assertEqual(
            attribute.leg_role(
                ["lora_linear_fused", "layer_norm_fused", "gelu_erf_fused"], "htsat"
            ),
            "D1",
        )
        self.assertEqual(
            attribute.leg_role(["lora_linear_fused", "layer_norm_fused"], "htsat"), "other"
        )

    def test_unrecognised_disable_set_is_other(self):
        self.assertEqual(attribute.leg_role(["some_other_key"], "clip-text"), "other")


def _chain_result(**kwargs) -> dict:
    return attribute.ChainResult(**kwargs).as_dict()


def _merge_row(leg_id: str, verdict: str = attribute.MERGE_VERDICT_VALID, wall_s_per_step: float | None = 0.1) -> dict:
    row = {"leg_id": leg_id, "verdict": verdict}
    if wall_s_per_step is not None:
        row["per_step"] = {"wall_s_per_step": wall_s_per_step}
    return row


class LegDecisionGradeTests(unittest.TestCase):
    def _row(self, verdict=attribute.VERDICT_VALID, unattributed_share=0.02):
        return {
            "verdict": verdict,
            "chains": {
                attribute.CHAIN_UNATTRIBUTED: _chain_result(
                    status="measured", gpu_busy_us=1.0, share_gpu_busy=unattributed_share
                )
            },
        }

    def test_valid_leg_under_the_bound_is_decision_grade(self):
        grade, reason = attribute.leg_decision_grade(
            self._row(unattributed_share=0.049), _merge_row("x")
        )
        self.assertTrue(grade)
        self.assertIsNone(reason)

    def test_valid_leg_at_exactly_the_bound_is_decision_grade(self):
        grade, _reason = attribute.leg_decision_grade(
            self._row(unattributed_share=attribute.UNATTRIBUTED_DECISION_GRADE_LIMIT), _merge_row("x")
        )
        self.assertTrue(grade)

    def test_valid_leg_over_the_bound_is_not_decision_grade(self):
        grade, reason = attribute.leg_decision_grade(self._row(unattributed_share=0.051), _merge_row("x"))
        self.assertFalse(grade)
        self.assertIn("0.0510", reason)

    def test_invalid_leg_is_never_decision_grade_even_with_a_tiny_share(self):
        grade, reason = attribute.leg_decision_grade(
            self._row(verdict=attribute.VERDICT_INVALID, unattributed_share=0.0), _merge_row("x")
        )
        self.assertFalse(grade)
        self.assertIn("INVALID", reason)

    def test_missing_chains_is_not_decision_grade(self):
        grade, reason = attribute.leg_decision_grade({"verdict": attribute.VERDICT_VALID}, _merge_row("x"))
        self.assertFalse(grade)
        self.assertIsNotNone(reason)

    def test_nan_unattributed_share_is_never_decision_grade(self):
        """Negative-control non-vacuity (family F): `NaN > c` is `False` in
        Python, so a naive `share > LIMIT` check would silently treat a
        diverged/NaN share as passing the gate."""
        row = self._row(unattributed_share=float("nan"))
        grade, reason = attribute.leg_decision_grade(row, _merge_row("x"))
        self.assertFalse(grade)
        self.assertIsNotNone(reason)

    def test_no_merge_row_is_never_decision_grade(self):
        """Pass 3, finding 5: `decision_grade` REQUIRES a corresponding
        `--merge-json` row — `merge_row=None` refuses, never assumes
        clean."""
        grade, reason = attribute.leg_decision_grade(self._row(), None)
        self.assertFalse(grade)
        self.assertIn("merge", reason.lower())

    def test_merge_row_not_valid_is_never_decision_grade(self):
        grade, reason = attribute.leg_decision_grade(self._row(), _merge_row("x", verdict="INVALID"))
        self.assertFalse(grade)
        self.assertIn("merge", reason.lower())

    def test_this_module_verdict_valid_but_merge_invalid_still_refuses(self):
        """Both conjuncts are required — a leg this module thinks is fine
        but the merge refused (counter equations, checkpoint identity, ...)
        is NOT decision-grade."""
        grade, _reason = attribute.leg_decision_grade(
            self._row(unattributed_share=0.0), _merge_row("x", verdict="INVALID")
        )
        self.assertFalse(grade)


class ComputeRealizedGainsTests(unittest.TestCase):
    """Pass 3, finding 4: `C-LORA`/`C-LN` are realized-gain NUMBERS, never
    chain-partition members — `compute_realized_gains` builds them as a
    SEPARATE top-level list."""

    def _leg(self, leg_id, tower, role, verdict, busy, dtype="f32"):
        return {
            "leg_id": leg_id,
            "tower": tower,
            "dtype": dtype,
            "verdict": verdict,
            "reasons": [],
            "chains": {attribute.CHAIN_LN: _chain_result(status="absent")},
            "_role": role,
            "_gpu_busy_us_per_step": busy,
        }

    def test_c_lora_is_a1_minus_d2_and_positive_when_eager_is_slower(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 78000.0),
        ]
        merge_by_leg_id = {
            "clip-text-A1": _merge_row("clip-text-A1", wall_s_per_step=0.10),
            "clip-text-D2": _merge_row("clip-text-D2", wall_s_per_step=0.13),
        }
        gains = attribute.compute_realized_gains(legs, merge_by_leg_id)
        lora = next(g for g in gains if g["chain"] == attribute.CHAIN_LORA and g["tower"] == "clip-text")
        self.assertAlmostEqual(lora["busy_delta_us_per_step"], 18000.0, places=6)
        self.assertGreater(lora["busy_delta_us_per_step"], 0.0)
        self.assertAlmostEqual(lora["wall_delta_s_per_step"], 0.03, places=6)
        self.assertGreater(lora["share_of_baseline_wall"], 0.0)
        self.assertIn("slower", lora["direction"])
        self.assertEqual(lora["eager_leg_id"], "clip-text-D2")
        self.assertEqual(lora["fused_leg_id"], "clip-text-A1")
        # `C-LORA`'s own fused leg (`A1`) IS the baseline leg — the two
        # denominators COINCIDE (pass 3, finding 5's own convention note).
        self.assertEqual(lora["baseline_leg_id"], "clip-text-A1")
        self.assertAlmostEqual(lora["baseline_wall_s"], lora["fused_twin_wall_s"], places=6)
        self.assertAlmostEqual(lora["share_of_baseline_wall"], lora["share_of_fused_twin_wall"], places=6)

    def test_c_ln_is_d1_minus_d2(self):
        legs = [
            self._leg("clip-text-D1", "clip-text", "D1", attribute.VERDICT_VALID, 89000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 79000.0),
        ]
        merge_by_leg_id = {
            "clip-text-D1": _merge_row("clip-text-D1", wall_s_per_step=0.15),
            "clip-text-D2": _merge_row("clip-text-D2", wall_s_per_step=0.13),
        }
        gains = attribute.compute_realized_gains(legs, merge_by_leg_id)
        ln = next(g for g in gains if g["chain"] == attribute.CHAIN_LN and g["tower"] == "clip-text")
        self.assertAlmostEqual(ln["busy_delta_us_per_step"], 10000.0, places=6)
        self.assertEqual(ln["eager_leg_id"], "clip-text-D1")
        self.assertEqual(ln["fused_leg_id"], "clip-text-D2")
        # Pass 3, finding 5: with no `A1` leg present, `share_of_baseline_wall`
        # cannot be computed (there is no shipped-A1 wall to divide by) — it
        # is `None`, never silently substituted with the fused leg's (`D2`'s)
        # own wall.
        self.assertIsNone(ln["baseline_wall_s"])
        self.assertIsNone(ln["share_of_baseline_wall"])
        self.assertAlmostEqual(ln["fused_twin_wall_s"], 0.13, places=6)
        self.assertAlmostEqual(ln["share_of_fused_twin_wall"], 0.02 / 0.13, places=6)

    def test_c_ln_share_of_baseline_wall_divides_by_a1_not_d2_pass3_finding5(self):
        """Pass 3, finding 5: `share_of_baseline_wall` is ALWAYS the
        tower's SHIPPED `A1` wall, even for `C-LN` (whose pair is `D1` vs
        `D2` — `D2` is NOT what ships). `fused_twin_wall_s`/
        `share_of_fused_twin_wall` divide by `D2`'s own wall instead, and
        the two denominators here are DELIBERATELY DIFFERENT so the test
        actually pins which is which."""
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D1", "clip-text", "D1", attribute.VERDICT_VALID, 89000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 79000.0),
        ]
        merge_by_leg_id = {
            "clip-text-A1": _merge_row("clip-text-A1", wall_s_per_step=0.10),
            "clip-text-D1": _merge_row("clip-text-D1", wall_s_per_step=0.15),
            "clip-text-D2": _merge_row("clip-text-D2", wall_s_per_step=0.13),
        }
        gains = attribute.compute_realized_gains(legs, merge_by_leg_id)
        ln = next(g for g in gains if g["chain"] == attribute.CHAIN_LN and g["tower"] == "clip-text")
        wall_delta = 0.15 - 0.13
        self.assertAlmostEqual(ln["baseline_wall_s"], 0.10, places=6)
        self.assertEqual(ln["baseline_leg_id"], "clip-text-A1")
        self.assertAlmostEqual(ln["share_of_baseline_wall"], wall_delta / 0.10, places=6)
        self.assertAlmostEqual(ln["fused_twin_wall_s"], 0.13, places=6)
        self.assertAlmostEqual(ln["share_of_fused_twin_wall"], wall_delta / 0.13, places=6)
        # The two denominators (and therefore the two shares) are distinct.
        self.assertNotAlmostEqual(ln["share_of_baseline_wall"], ln["share_of_fused_twin_wall"], places=6)

    def test_missing_twin_produces_no_entry_for_that_tower(self):
        legs = [self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0)]
        gains = attribute.compute_realized_gains(legs, {})
        self.assertEqual([g for g in gains if g["tower"] == "clip-text"], [])

    def test_invalid_leg_produces_no_entry(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_INVALID, 78000.0),
        ]
        gains = attribute.compute_realized_gains(legs, {})
        self.assertEqual([g for g in gains if g["chain"] == attribute.CHAIN_LORA], [])

    def test_bf16_a_leg_never_produces_a_lora_gain_d2_is_f32_only(self):
        legs = [
            self._leg("clip-text-A2", "clip-text", "A", attribute.VERDICT_VALID, 60000.0, dtype="bf16"),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 78000.0, dtype="f32"),
        ]
        gains = attribute.compute_realized_gains(legs, {})
        self.assertEqual([g for g in gains if g["chain"] == attribute.CHAIN_LORA], [])

    def test_missing_merge_wall_leaves_wall_fields_none_but_busy_still_computed(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 78000.0),
        ]
        gains = attribute.compute_realized_gains(legs, {})
        lora = next(g for g in gains if g["chain"] == attribute.CHAIN_LORA)
        self.assertAlmostEqual(lora["busy_delta_us_per_step"], 18000.0, places=6)
        self.assertIsNone(lora["wall_delta_s_per_step"])
        self.assertIsNone(lora["share_of_baseline_wall"])

    def test_realized_gains_never_appear_in_any_leg_chains_dict(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 78000.0),
        ]
        attribute.compute_realized_gains(legs, {})
        for leg in legs:
            self.assertNotIn(attribute.CHAIN_LORA, leg["chains"])

    def test_direction_string_reports_a_measured_negative_delta_honestly(self):
        """If a hypothetical leg pair ever produced a negative delta (eager
        FASTER than fused), the direction string states that honestly
        rather than silently flipping the sign."""
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 80000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 60000.0),
        ]
        gains = attribute.compute_realized_gains(legs, {})
        lora = next(g for g in gains if g["chain"] == attribute.CHAIN_LORA)
        self.assertLess(lora["busy_delta_us_per_step"], 0.0)
        self.assertIn("FASTER", lora["direction"])


class TwoSidedDecisionRuleTests(unittest.TestCase):
    """The contract's two-sided rule, evaluated ONLY on SYNTHETIC share
    numbers (never a real leg's)."""

    CHAIN_KEY = "C-ATTN-clip-text"

    def _leg(
        self, leg_id, dtype, s_wall, s_busy, u_wall, u_busy, role="A", verdict=attribute.VERDICT_VALID, reasons=None
    ):
        return {
            "leg_id": leg_id,
            "tower": "clip-text",
            "dtype": dtype,
            "verdict": verdict,
            "reasons": reasons if reasons is not None else [],
            "_role": role,
            "chains": {
                self.CHAIN_KEY: _chain_result(
                    status="measured", gpu_busy_us=1.0, share_wall=s_wall, share_gpu_busy=s_busy
                ),
                attribute.CHAIN_UNATTRIBUTED: _chain_result(
                    status="measured", gpu_busy_us=1.0, share_wall=u_wall, share_gpu_busy=u_busy
                ),
            },
        }

    def _merge_for(self, *leg_ids):
        return {leg_id: _merge_row(leg_id) for leg_id in leg_ids}

    def test_activate_when_a1_wall_share_at_or_above_ten_percent(self):
        legs = [self._leg("clip-text-A1", "f32", s_wall=0.10, s_busy=0.10, u_wall=0.0, u_busy=0.0)]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs, self._merge_for("clip-text-A1")
        )
        self.assertEqual(decision["verdict"], "ACTIVATE")

    def test_activate_when_only_a2_crosses_the_bar(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.02, s_busy=0.02, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.11, s_busy=0.11, u_wall=0.0, u_busy=0.0),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1", "clip-text-A2"),
        )
        self.assertEqual(decision["verdict"], "ACTIVATE")

    def test_decline_when_combined_share_under_five_percent_on_every_decision_grade_leg(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.01, u_busy=0.01),
            self._leg("clip-text-A2", "bf16", s_wall=0.02, s_busy=0.02, u_wall=0.01, u_busy=0.01),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1", "clip-text-A2"),
        )
        self.assertEqual(decision["verdict"], "DECLINE")

    def test_unresolved_in_the_five_to_ten_percent_band(self):
        legs = [self._leg("clip-text-A1", "f32", s_wall=0.07, s_busy=0.07, u_wall=0.0, u_busy=0.0)]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs, self._merge_for("clip-text-A1")
        )
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_unresolved_when_a1_missing(self):
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", [], {})
        self.assertEqual(decision["verdict"], "UNRESOLVED")
        self.assertIn("A1 not decision-grade", decision["reason"])

    def test_unresolved_without_a_merge_row_even_with_a_generous_share(self):
        """Pass 3, finding 5: a leg with NO corresponding `--merge-json`
        row can never reach decision-grade, so the two-sided rule reads
        UNRESOLVED honestly rather than ACTIVATE/DECLINE on an
        un-certified leg."""
        legs = [self._leg("clip-text-A1", "f32", s_wall=0.50, s_busy=0.50, u_wall=0.0, u_busy=0.0)]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs, {}
        )
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_unresolved_when_a1_not_decision_grade_even_if_a2_would_activate(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.06, u_busy=0.06),
            self._leg("clip-text-A2", "bf16", s_wall=0.20, s_busy=0.20, u_wall=0.0, u_busy=0.0),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1", "clip-text-A2"),
        )
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_decline_notes_f32_only_when_a2_is_not_decision_grade(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.01, s_busy=0.01, u_wall=0.06, u_busy=0.06),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1", "clip-text-A2"),
        )
        self.assertEqual(decision["verdict"], "DECLINE")
        self.assertIn("F32-only", decision["reason"])

    def test_f32_only_caveat_fires_when_a2_is_absent(self):
        """The F32-only caveat fires for EVERY way A2 can be
        not-decision-grade, not only "present but INVALID" — this is the
        ABSENT case (no A2 leg in `legs` at all)."""
        legs = [self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0)]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs, self._merge_for("clip-text-A1")
        )
        self.assertEqual(decision["verdict"], "DECLINE")
        self.assertIn("F32-only", decision["reason"])
        self.assertIn("no A2 (BF16) leg present", decision["reason"])

    def test_f32_only_caveat_names_the_invalid_reason_when_a2_is_present_but_invalid(self):
        """The SECOND way: A2 is present in `legs` but this module's own
        `verdict` for it is INVALID — evidenced by the real
        `clip-text-A2` (the 1%-unknown-kernel gate). The caveat must NAME
        the actual reason, not just point at "see this leg's own
        reasons" (pass 3, finding 1)."""
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
            self._leg(
                "clip-text-A2",
                "bf16",
                s_wall=0.01,
                s_busy=0.01,
                u_wall=0.0,
                u_busy=0.0,
                verdict=attribute.VERDICT_INVALID,
                reasons=["clip-text-A2: kernel 'Kernel2' (grid=[16, 1, 10]) is not a known kernel name"],
            ),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1", "clip-text-A2"),
        )
        self.assertEqual(decision["verdict"], "DECLINE")
        self.assertIn("F32-only", decision["reason"])
        self.assertIn("Kernel2", decision["reason"])
        self.assertIn("is not a known kernel name", decision["reason"])

    def test_f32_only_caveat_fires_when_a2_is_valid_but_has_no_merge_row(self):
        """The THIRD way: A2 is present and this module's own `verdict` is
        VALID, but `--merge-json` names no corresponding row for it — the
        merge cannot certify it, so it is still not decision-grade (pass
        3, finding 1)."""
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1"),  # no merge row for clip-text-A2
        )
        self.assertEqual(decision["verdict"], "DECLINE")
        self.assertIn("F32-only", decision["reason"])
        self.assertIn("no corresponding row in --merge-json", decision["reason"])

    def test_declines_only_when_every_decision_grade_leg_qualifies(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.04, s_busy=0.04, u_wall=0.03, u_busy=0.03),
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text",
            self.CHAIN_KEY,
            "clip-text",
            legs,
            self._merge_for("clip-text-A1", "clip-text-A2"),
        )
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_nan_unattributed_share_fails_the_leg_closed_at_decision_grade(self):
        legs = [
            self._leg(
                "clip-text-A1",
                "f32",
                s_wall=float("nan"),
                s_busy=float("nan"),
                u_wall=float("nan"),
                u_busy=float("nan"),
            )
        ]
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs, self._merge_for("clip-text-A1")
        )
        self.assertEqual(decision["verdict"], "UNRESOLVED")
        self.assertIn("A1 not decision-grade", decision["reason"])

    def test_nan_chain_share_alone_never_silently_activates_or_declines(self):
        leg = self._leg(
            "clip-text-A1", "f32", s_wall=float("nan"), s_busy=float("nan"), u_wall=0.01, u_busy=0.01
        )
        decision = attribute.decide_candidate_port(
            "C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", [leg], self._merge_for("clip-text-A1")
        )
        self.assertEqual(decision["verdict"], "DECLINE")

    def test_candidate_ports_for_tower_maps_c_mlp_to_gelu_chain(self):
        ports = attribute.candidate_ports_for_tower("clip-text")
        self.assertEqual(ports["C-MLP-clip-text"], attribute.CHAIN_GELU)
        self.assertEqual(ports["C-ATTN-clip-text"], attribute.chain_attn("clip-text"))

    def test_decide_all_candidates_covers_every_candidate_port(self):
        decisions = attribute.decide_all_candidates([], {})
        ports = {d["port"] for d in decisions}
        self.assertEqual(
            ports,
            {"C-ATTN-clip-text", "C-MLP-clip-text", "C-ATTN-clip-vision", "C-MLP-clip-vision"},
        )
        for decision in decisions:
            self.assertEqual(decision["verdict"], "UNRESOLVED")


def _write_leg_dir(base: Path, leg_id: str, manifest_extra: dict, census: dict) -> Path:
    leg_dir = base / leg_id
    leg_dir.mkdir(parents=True)
    manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
    manifest.update(manifest_extra)
    (leg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (leg_dir / "census.json").write_text(json.dumps(census), encoding="utf-8")
    return leg_dir


def _merge_report_for(*leg_ids: str) -> dict:
    return {
        "tool": "profile_421_merge",
        "schema": 1,
        "legs": [_merge_row(leg_id) for leg_id in leg_ids],
    }


class AttributeLegAndReportTests(unittest.TestCase):
    def test_end_to_end_on_the_real_fixture_is_valid_and_decision_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir, _merge_report_for("clip-text-A1"))
            self.assertEqual(report["summary"]["legs_total"], 1)
            self.assertEqual(report["summary"]["legs_valid"], 1)
            row = report["legs"][0]
            self.assertEqual(row["verdict"], attribute.VERDICT_VALID)
            self.assertNotIn("_role", row)
            self.assertNotIn("_gpu_busy_us_per_step", row)
            for name in (
                attribute.CHAIN_LN,
                attribute.CHAIN_GELU,
                attribute.chain_attn("clip-text"),
                attribute.CHAIN_BASE_GEMM,
            ):
                self.assertIn(name, row["chains"])
            self.assertNotIn(attribute.CHAIN_LORA, row["chains"])
            self.assertIn("decision_grade", row)
            self.assertIn("outside_signature_plausibly_attention", row)
            self.assertEqual([], report["realized_gains"])

    def test_missing_manifest_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "legs" / "clip-text-A1"
            leg_dir.mkdir(parents=True)
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)
            self.assertTrue(row["reasons"])
            self.assertFalse(row["decision_grade"])

    def test_missing_census_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = Path(tmp) / "legs" / "clip-text-A1"
            leg_dir.mkdir(parents=True)
            manifest = dict(CLIP_TEXT_A1_MANIFEST_FIELDS)
            (leg_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)

    def test_excluded_from_chain_attribution_is_invalid_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            census = load_fixture_census()
            census["excluded_from_chain_attribution"] = True
            leg_dir = _write_leg_dir(Path(tmp) / "legs", "clip-text-A1", {}, census)
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)
            self.assertTrue(any("excluded_from_chain_attribution" in r for r in row["reasons"]))

    def test_census_not_ok_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = _write_leg_dir(
                Path(tmp) / "legs",
                "clip-text-A1",
                {"census_ok": False, "census_exit": 1},
                load_fixture_census(),
            )
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_INVALID)

    def test_two_legs_a_and_d2_compose_a_full_report_with_lora_realized_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            d2_census = load_fixture_census()
            d2_census["gpu_kernel_us_per_step"] = load_fixture_census()["gpu_kernel_us_per_step"] + 5000.0
            _write_leg_dir(
                legs_dir,
                "clip-text-D2",
                {"kernels_disabled": ["lora_linear_fused"]},
                d2_census,
            )
            report = attribute.build_report(
                legs_dir, _merge_report_for("clip-text-A1", "clip-text-D2")
            )
            self.assertEqual(report["summary"]["legs_valid"], 2)
            by_id = {row["leg_id"]: row for row in report["legs"]}
            self.assertNotIn(attribute.CHAIN_LORA, by_id["clip-text-A1"]["chains"])
            self.assertNotIn(attribute.CHAIN_LORA, by_id["clip-text-D2"]["chains"])
            lora_gains = [g for g in report["realized_gains"] if g["chain"] == attribute.CHAIN_LORA]
            self.assertEqual(len(lora_gains), 1)
            self.assertAlmostEqual(lora_gains[0]["busy_delta_us_per_step"], 5000.0, places=6)
            self.assertEqual(lora_gains[0]["eager_leg_id"], "clip-text-D2")
            self.assertEqual(lora_gains[0]["fused_leg_id"], "clip-text-A1")

    def test_report_carries_candidate_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir, _merge_report_for("clip-text-A1"))
            self.assertIn("candidate_decisions", report)
            ports = {d["port"] for d in report["candidate_decisions"]}
            self.assertIn("C-ATTN-clip-text", ports)
            self.assertIn("C-MLP-clip-text", ports)

    def test_summary_carries_decision_grade_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir, _merge_report_for("clip-text-A1"))
            self.assertIn("legs_decision_grade", report["summary"])
            self.assertEqual(report["summary"]["legs_decision_grade"], 1)

    def test_a_leg_absent_from_the_merge_report_is_not_decision_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir, _merge_report_for())
            row = report["legs"][0]
            self.assertFalse(row["decision_grade"])
            self.assertIn("merge", row["decision_grade_reason"].lower())

    def test_manifest_less_directory_surfaces_as_invalid_not_dropped_silently(self):
        """A directory with no `manifest.json` is never silently dropped —
        `build_report` surfaces it as its own `INVALID` row, so a broken
        pull (a leg directory whose driver crashed before writing a
        manifest) shows up in `legs_total`/`legs_invalid`, never vanishing
        from the report's own count."""
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            (legs_dir / "clip-text-A2-broken").mkdir(parents=True)
            report = attribute.build_report(legs_dir, _merge_report_for("clip-text-A1"))
            self.assertEqual(report["summary"]["legs_total"], 2)
            self.assertEqual(report["summary"]["legs_invalid"], 1)
            by_id = {row["leg_id"]: row for row in report["legs"]}
            broken = by_id["clip-text-A2-broken"]
            self.assertEqual(broken["verdict"], attribute.VERDICT_INVALID)
            self.assertTrue(any("manifest.json could not be read" in r for r in broken["reasons"]))

    def test_p2_bf16_scratch_directory_is_still_excluded(self):
        """The ONE directory `_leg_dirs` still drops on purpose — the P2b
        BF16 driver's own scratch corpus directory, never a
        `profile_421_legs.sh` leg at all."""
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            (legs_dir / "p2-bf16").mkdir(parents=True)
            report = attribute.build_report(legs_dir, _merge_report_for("clip-text-A1"))
            self.assertEqual(report["summary"]["legs_total"], 1)
            self.assertNotIn("p2-bf16", {row["leg_id"] for row in report["legs"]})


class MemcpyMemsetInformationalTests(unittest.TestCase):
    def test_memcpy_memset_reported_but_not_in_any_chain_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            census = load_fixture_census()
            census["memcpy_per_step"] = {"count": 743.0, "us": 1351.6}
            census["memset_per_step"] = {"count": 900.0, "us": 2842.1}
            leg_dir = _write_leg_dir(Path(tmp) / "legs", "clip-text-A1", {}, census)
            row = attribute.attribute_leg(leg_dir)
            self.assertIn("memcpy_memset", row)
            self.assertEqual(row["memcpy_memset"]["memcpy_per_step"]["us"], 1351.6)
            total_chain_busy = sum(
                c["gpu_busy_us"] for c in row["chains"].values() if "gpu_busy_us" in c
            )
            total_in_rows = sum(r["us_per_step"] for r in census["by_kernel_and_grid"])
            self.assertAlmostEqual(total_chain_busy, total_in_rows, places=6)

    def test_absent_memcpy_memset_fields_do_not_add_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = _write_leg_dir(Path(tmp) / "legs", "clip-text-A1", {}, load_fixture_census())
            row = attribute.attribute_leg(leg_dir)
            self.assertNotIn("memcpy_memset", row)


class TableFormattingTests(unittest.TestCase):
    def test_format_table_lists_every_chain_row_realized_gains_and_candidate_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir, _merge_report_for("clip-text-A1"))
            table = attribute.format_table(report)
            self.assertIn("clip-text-A1", table)
            self.assertIn(attribute.CHAIN_LN, table)
            self.assertIn(attribute.chain_attn("clip-text"), table)
            self.assertIn("realized gains:", table)
            self.assertIn("candidate port decisions:", table)
            self.assertIn("C-ATTN-clip-text", table)


class HtsatStageSignatureSmokeTests(unittest.TestCase):
    """The name-only stage view (`htsat_stage_signatures`) — still a light
    smoke test, but the FULL per-stage element-count derivation
    (`derive_htsat_signatures`) is now cross-checked against the real
    `htsat-A1`/`htsat-D1` exports below (module doc, "HTSAT")."""

    def test_four_stages_with_positive_window_and_heads(self):
        stages = attribute.htsat_stage_signatures()
        self.assertEqual(len(stages), 4)
        for stage in stages:
            self.assertGreater(stage["depth"], 0)
            self.assertGreater(stage["heads"], 0)
            self.assertGreater(stage["window_size"], 0)

    def test_htsat_has_no_declared_windowing_or_front_fusion_chain(self):
        """`WINDOWING`/`FRONT-FUSION` remain UNDECLARED even with real
        HTSAT exports in hand (module doc, "HTSAT") — a window-partition
        copy sits at the IDENTICAL element count as a generic residual
        permute, and the one plausible front-fusion-activation candidate
        (`urelu_f32`) cannot be told apart from the audio projection
        head's own ReLU without a dedicated ablation. `GRAD-BOOKKEEPING`
        is likewise undeclared for `htsat` (no evidenced parameter-scale
        rule for its nine distinct target-module shapes)."""
        declared = attribute.declared_chains_for_tower("htsat")
        self.assertNotIn(attribute.CHAIN_WINDOWING, declared)
        self.assertNotIn(attribute.CHAIN_FRONT_FUSION, declared)
        self.assertNotIn(attribute.CHAIN_GRAD_BOOKKEEPING, declared)
        self.assertNotIn(attribute.CHAIN_BIAS_RESIDUAL_QKV, declared)
        self.assertIn(attribute.CHAIN_GELU_HTSAT, declared)
        self.assertIn(attribute.CHAIN_LN, declared)
        self.assertIn(attribute.chain_attn("htsat"), declared)


class HtsatSignatureDerivationTests(unittest.TestCase):
    """`derive_htsat_signatures`'s own per-stage element counts, checked
    against the table in `fixtures/profile_421_htsat_a1/PROVENANCE.md`
    (itself cross-checked against the real `htsat-A1` export)."""

    def setUp(self):
        self.hsig = htsat_signatures()

    def test_rows_witnessed_from_batch(self):
        self.assertEqual(self.hsig.rows.value, 24)

    def test_four_stages_with_declared_dims_doubling(self):
        dims = [s.dim for s in self.hsig.stages]
        self.assertEqual(dims, [96, 192, 384, 768])

    def test_tokens_quarter_each_stage(self):
        tokens = [s.tokens for s in self.hsig.stages]
        self.assertEqual(tokens, [4096, 1024, 256, 64])

    def test_windows_derived_from_tokens_over_window_size_squared(self):
        windows = [s.windows for s in self.hsig.stages]
        self.assertEqual(windows, [64, 16, 4, 1])

    def test_attn_batch_count_per_stage(self):
        counts = [s.attn_batch_count(self.hsig.rows.value) for s in self.hsig.stages]
        self.assertEqual(counts, [6144, 3072, 1536, 768])

    def test_attn_softmax_rows_per_stage(self):
        rows = [s.attn_softmax_rows(self.hsig.rows.value) for s in self.hsig.stages]
        self.assertEqual(rows, [393216, 196608, 98304, 49152])

    def test_mlp_shape_elements_per_stage(self):
        elements = [s.mlp_shape_elements(self.hsig.rows.value) for s in self.hsig.stages]
        self.assertEqual(elements, [37748736, 18874368, 9437184, 4718592])

    def test_out_shape_elements_per_stage(self):
        elements = [s.out_shape_elements(self.hsig.rows.value) for s in self.hsig.stages]
        self.assertEqual(elements, [9437184, 4718592, 2359296, 1179648])

    def test_known_collision_out_stage_equals_mlp_stage_plus_two(self):
        """Documented, not resolved (`fixtures/profile_421_htsat_a1/
        PROVENANCE.md`, "Known ambiguity") — `out(0)==mlp(2)` and
        `out(1)==mlp(3)` are a mathematical identity of this architecture,
        never an accident this module should paper over."""
        rows = self.hsig.rows.value
        self.assertEqual(self.hsig.stages[0].out_shape_elements(rows), self.hsig.stages[2].mlp_shape_elements(rows))
        self.assertEqual(self.hsig.stages[1].out_shape_elements(rows), self.hsig.stages[3].mlp_shape_elements(rows))

    def test_signature_error_on_non_positive_batch(self):
        with self.assertRaises(attribute.SignatureError):
            attribute.derive_htsat_signatures({"batch": 0})

    def test_gelu_seam_calls_matching_declared_depth_sum_succeeds(self):
        """`HTSAT_A1_MANIFEST_FIELDS`'s own witnessed
        `gelu_seam_calls_per_forward=12` equals `sum(HTSAT_DEPTHS)=12`
        (real, from the `htsat-A1` manifest) — `derive_htsat_signatures`
        does not raise."""
        self.assertEqual(sum(attribute.HTSAT_DEPTHS), 12)
        attribute.derive_htsat_signatures(HTSAT_A1_MANIFEST_FIELDS)

    def test_signature_error_on_gelu_seam_calls_mismatch(self):
        """A leg whose witnessed `gelu_seam_calls_per_forward` does not
        equal `sum(HTSAT_DEPTHS)` means this leg's own checkpoint does not
        match the declared Swin depth this module assumes — every
        per-stage element count would be silently wrong, so this module
        refuses rather than guessing. The error NAMES both values."""
        manifest = json.loads(json.dumps(HTSAT_A1_MANIFEST_FIELDS))
        manifest["fusible_site_census"]["n"]["gelu_seam_calls_per_forward"] = 11
        with self.assertRaises(attribute.SignatureError) as ctx:
            attribute.derive_htsat_signatures(manifest)
        self.assertIn("11", str(ctx.exception))
        self.assertIn("12", str(ctx.exception))


class HtsatAttributeCensusA1FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `htsat-A1` fixture
    cut (`kernels_disabled=[]`, every fusible seam admitted) — see
    `fixtures/profile_421_htsat_a1/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_HTSAT_A1)
        self.hsig = htsat_signatures()

    def _chains(self):
        return attribute.attribute_census(self.census, self.hsig, "htsat", ln_disabled=False)

    def test_chains_partition_busy_exactly(self):
        chains, _u, _r = self._chains()
        total_in_rows = sum(r["us_per_step"] for r in self.census["by_kernel_and_grid"])
        attributed_total = sum(c.gpu_busy_us for c in chains.values() if c.gpu_busy_us is not None)
        self.assertAlmostEqual(attributed_total, total_in_rows, places=6)

    def test_only_the_two_intentional_kernel2_rows_are_unknown(self):
        """This fixture's own negative controls (module doc,
        `test_anonymous_kernel2_degenerate_dims_stay_unattributed`) are the
        ONLY unknown rows, and neither invalidates the leg (each
        individually well under the 1% gate)."""
        _chains, unknown, reasons = self._chains()
        self.assertEqual({u["kernel"] for u in unknown}, {"Kernel2"})
        for u in unknown:
            self.assertLess(u["share_gpu_busy"], attribute.UNKNOWN_KERNEL_SHARE_LIMIT)
        self.assertEqual(reasons, [])

    def test_every_declared_chain_present_or_absent(self):
        chains, _u, _r = self._chains()
        for name in attribute.declared_chains_for_tower("htsat"):
            self.assertIn(name, chains, name)

    def test_ln_by_name(self):
        chains, _u, _r = self._chains()
        ln_names = set(chains[attribute.CHAIN_LN].kernel_names)
        self.assertIn("layer_norm_fwd_f32_biased", ln_names)
        self.assertIn("layer_norm_bwd_dx_f32", ln_names)

    def test_gelu_erf_fused_by_name(self):
        chains, _u, _r = self._chains()
        gelu_names = set(chains[attribute.CHAIN_GELU_HTSAT].kernel_names)
        self.assertIn("gelu_erf_fwd_f32", gelu_names)
        self.assertIn("gelu_erf_bwd_dx_f32", gelu_names)

    def test_attention_batched_gemm_two_different_libraries_two_stages(self):
        """`ampere_sgemm_128x128_nt grid=[1,1,6144]` (stage 0) and
        `magma_sgemmEx_kernel grid=[1,2,1536]` (stage 2) are TWO different
        GEMM libraries, both carrying `grid[2] == attn_batch_count(stage)`
        — the SAME name-independent relational rule generalizing across
        stages."""
        chains, _u, _r = self._chains()
        attn_names = set(chains[attribute.chain_attn("htsat")].kernel_names)
        self.assertIn("ampere_sgemm_128x128_nt", attn_names)
        self.assertIn("magma_sgemmEx_kernel", attn_names)

    def test_attention_elementwise_by_shape_two_stages(self):
        chains, _u, _r = self._chains()
        attn_names = set(chains[attribute.chain_attn("htsat")].kernel_names)
        self.assertIn("badd_f32", attn_names)  # stage 0 attn shape
        self.assertIn("eq_f32", attn_names)  # stage 2 attn shape

    def test_attention_softmax_reduction_block_64(self):
        chains, _u, _r = self._chains()
        self.assertIn("fast_max_f32", chains[attribute.chain_attn("htsat")].kernel_names)

    def test_bias_residual_out_unambiguous_stages(self):
        """Stages 2/3's own `out` tier grids (`2304`/`1152`) have no
        `mlp(stage+2)` to collide with — see PROVENANCE.md's own
        "Known ambiguity" note for the two grids this fixture deliberately
        avoids (`9216`/`4608`)."""
        chains, _u, _r = self._chains()
        self.assertIn("badd_f32", chains[attribute.CHAIN_BIAS_RESIDUAL_OUT].kernel_names)

    def test_permute_reshape_by_name(self):
        chains, _u, _r = self._chains()
        self.assertIn("ucopy_f32", chains[attribute.CHAIN_PERMUTE_RESHAPE].kernel_names)

    def test_optimizer_by_name(self):
        chains, _u, _r = self._chains()
        self.assertIn("adamw_moment_update_f32", chains[attribute.CHAIN_OPTIMIZER].kernel_names)

    def test_embed_gather_by_name(self):
        chains, _u, _r = self._chains()
        self.assertIn("is_u32_f32", chains[attribute.CHAIN_EMBED_GATHER].kernel_names)

    def test_patch_embed_by_name(self):
        chains, _u, _r = self._chains()
        self.assertIn("im2col_f32", chains[attribute.CHAIN_PATCH_EMBED].kernel_names)

    def test_dropout_by_name(self):
        chains, _u, _r = self._chains()
        self.assertIn("dropout_fwd_f32", chains[attribute.CHAIN_DROPOUT].kernel_names)

    def test_cast_at_non_attn_shape(self):
        chains, _u, _r = self._chains()
        self.assertIn("cast_f32_f32", chains[attribute.CHAIN_CAST].kernel_names)

    def test_loss_reduce_catch_all(self):
        chains, _u, _r = self._chains()
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LOSS_REDUCE].kernel_names)

    def test_base_gemm_by_name(self):
        chains, _u, _r = self._chains()
        self.assertIn("ampere_sgemm_128x64_nn", chains[attribute.CHAIN_BASE_GEMM].kernel_names)

    def test_anonymous_kernel2_degenerate_dims_stay_unattributed(self):
        """The two `Kernel2` rows this fixture carries (`[3,1,36]`,
        `[12,1,24]`) both have a `1` in grid position 1 — NOT a genuine
        3-D tile grid — so the anonymous grid-family fallback does not
        fire and they remain `UNATTRIBUTED` (counted toward the
        unknown-kernel gate, each individually under 1% of busy on the
        real leg — module doc)."""
        chains, unknown, _r = self._chains()
        self.assertIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)
        unknown_names = {u["kernel"] for u in unknown}
        self.assertIn("Kernel2", unknown_names)

    def test_ambiguous_out_mlp_collision_sums_the_real_colliding_rows(self):
        """This fixture's own `badd_f32` rows at `grid=[9216,...]`/
        `grid=[4608,...]` (real, byte-exact from the `htsat-A1` export —
        `fixtures/profile_421_htsat_a1/PROVENANCE.md`, "Known ambiguity")
        land `BIAS/RESIDUAL-OUT` via the tier fallback, at a shape that is
        SIMULTANEOUSLY `out_shape_elements(stage)` and `mlp_shape_elements
        (stage+2)` — `htsat_ambiguous_out_mlp_collision` sums exactly
        these two rows' own `us_per_step`, read straight off the census,
        never a transcribed literal."""
        result = attribute.htsat_ambiguous_out_mlp_collision(self.census, self.hsig, ln_disabled=False)
        expected_busy_us = sum(
            r["us_per_step"]
            for r in self.census["by_kernel_and_grid"]
            if r["kernel"] == "badd_f32" and r["grid"][0] in (9216, 4608)
        )
        self.assertGreater(expected_busy_us, 0.0)
        self.assertAlmostEqual(result["busy_us"], expected_busy_us, places=6)
        self.assertEqual(result["grids"], [4608, 9216])
        self.assertGreater(result["share_gpu_busy"], 0.0)

    def test_ambiguous_out_mlp_collision_excludes_name_classified_rows(self):
        """A row a NAME rule already resolves unambiguously (this
        fixture's own `gelu_erf_bwd_dx_f32`/`dropout_fwd_f32`/
        `cast_f32_f32`/`im2col_f32` rows, all real, all at
        `grid=[9216,...]` too) must NOT be counted — the ambiguity is
        specific to the tier-fallback classification, never to every
        kernel sharing a grid."""
        name_classified = ("gelu_erf_bwd_dx_f32", "dropout_fwd_f32", "cast_f32_f32", "im2col_f32")
        rows_at_ambiguous_grid = [
            r for r in self.census["by_kernel_and_grid"] if r["kernel"] in name_classified and r["grid"][0] == 9216
        ]
        # These rows really are present at the ambiguous grid, so this
        # test is not vacuously true.
        self.assertEqual({r["kernel"] for r in rows_at_ambiguous_grid}, set(name_classified))
        for row in rows_at_ambiguous_grid:
            chain = attribute.classify_htsat_kernel(row, self.hsig, ln_disabled=False)
            self.assertNotIn(chain, attribute._AMBIGUOUS_OUT_MLP_CHAINS, (row["kernel"], chain))

        result = attribute.htsat_ambiguous_out_mlp_collision(self.census, self.hsig, ln_disabled=False)
        excluded_busy_us = sum(r["us_per_step"] for r in rows_at_ambiguous_grid)
        self.assertGreater(excluded_busy_us, 0.0)
        total_at_ambiguous_grid = excluded_busy_us + sum(
            r["us_per_step"]
            for r in self.census["by_kernel_and_grid"]
            if r["kernel"] == "badd_f32" and r["grid"][0] in (9216, 4608)
        )
        self.assertLess(result["busy_us"], total_at_ambiguous_grid)


class HtsatAttributeCensusD1EagerTwinTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `htsat-D1` 8-row
    cut (`layer_norm_fused`/`gelu_erf_fused`/`lora_linear_fused` all
    disabled) — see `fixtures/profile_421_htsat_d1/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_HTSAT_D1)
        self.hsig = htsat_signatures()

    def _chains(self, ln_disabled=True):
        return attribute.attribute_census(self.census, self.hsig, "htsat", ln_disabled=ln_disabled)

    def test_no_unknown_kernel_names(self):
        _chains, unknown, reasons = self._chains()
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])

    def test_eager_ln_reduction_lands_c_ln(self):
        chains, _u, _r = self._chains(ln_disabled=True)
        ln_names = set(chains[attribute.CHAIN_LN].kernel_names)
        self.assertIn("fast_sum_f32", ln_names)

    def test_ln_disabled_false_does_not_misroute_eager_reduction_to_c_ln(self):
        """Negative control: with `ln_disabled=False` (as if this were an
        A/D2 leg), the SAME `fast_sum_f32 grid=[98304,...] block=[128,...]`
        row must NOT land `C-LN` (the eager LN chain literally cannot
        exist while `layer_norm_fused` is admitted)."""
        chains, _u, _r = self._chains(ln_disabled=False)
        self.assertNotIn("fast_sum_f32", chains[attribute.CHAIN_LN].kernel_names)

    def test_attn_softmax_reduction_block_64_not_swept_into_c_ln(self):
        """The attention-softmax `fast_sum_f32 grid=[393216,...]
        block=[64,...]` row is UNAFFECTED by the LN toggle — it classifies
        `C-ATTN-htsat` regardless of `ln_disabled` (module doc,
        `_is_attn_softmax_reduction_grid`'s own block-size discriminator),
        even though this SAME leg's `fast_sum_f32 grid=[98304,...]
        block=[128,...]` row (a DIFFERENT grid) legitimately lands `C-LN`
        — checked per-ROW (`classify_htsat_kernel` directly), since
        `kernel_names` aggregates by NAME and would show `fast_sum_f32` in
        BOTH buckets."""
        attn_row = {"kernel": "fast_sum_f32", "grid": [393216, 1, 1], "block": [64, 1, 1]}
        ln_row = {"kernel": "fast_sum_f32", "grid": [98304, 1, 1], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_htsat_kernel(attn_row, self.hsig, ln_disabled=True),
            attribute.chain_attn("htsat"),
        )
        self.assertEqual(
            attribute.classify_htsat_kernel(ln_row, self.hsig, ln_disabled=True),
            attribute.CHAIN_LN,
        )
        chains, _u, _r = self._chains(ln_disabled=True)
        self.assertIn("fast_sum_f32", chains[attribute.chain_attn("htsat")].kernel_names)
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LN].kernel_names)

    def test_eager_gelu_erf_trio_lands_c_gelu_htsat(self):
        """`ugelu_erf_f32`/`uerf_f32`/`uneg_f32` — candle's own three-kernel
        eager `gelu_erf` decomposition — all land `C-GELU-HTSAT`, the SAME
        chain the FUSED `gelu_erf_fwd_f32`/`gelu_erf_bwd_dx_f32` kernels
        land on `htsat-A1` (module doc)."""
        chains, _u, _r = self._chains()
        gelu_names = set(chains[attribute.CHAIN_GELU_HTSAT].kernel_names)
        self.assertIn("ugelu_erf_f32", gelu_names)
        self.assertIn("uerf_f32", gelu_names)
        self.assertIn("uneg_f32", gelu_names)

    def test_attention_elementwise_and_batched_gemm_still_attn(self):
        chains, _u, _r = self._chains()
        attn_names = set(chains[attribute.chain_attn("htsat")].kernel_names)
        self.assertIn("badd_f32", attn_names)
        self.assertIn("ampere_sgemm_128x128_nt", attn_names)


class HtsatAttributeLegEndToEndTests(unittest.TestCase):
    """`attribute_leg`/`declared_chains_for_tower` end to end for `htsat` —
    a real leg now reaches `VALID`, never `INVALID`-by-unrecognised-tower."""

    def test_htsat_a1_leg_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = _write_leg_dir(
                Path(tmp) / "legs", "htsat-A1", HTSAT_A1_MANIFEST_FIELDS, load_fixture_census(FIXTURE_KERNELS_HTSAT_A1)
            )
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["reasons"], [])
            self.assertEqual(row["verdict"], attribute.VERDICT_VALID)
            self.assertEqual(row["tower"], "htsat")

    def test_htsat_d1_leg_is_valid_and_ln_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            leg_dir = _write_leg_dir(
                Path(tmp) / "legs", "htsat-D1", HTSAT_D1_MANIFEST_FIELDS, load_fixture_census(FIXTURE_KERNELS_HTSAT_D1)
            )
            row = attribute.attribute_leg(leg_dir)
            self.assertEqual(row["verdict"], attribute.VERDICT_VALID)
            self.assertIn("fast_sum_f32", row["chains"][attribute.CHAIN_LN]["kernel_names"])


class HtsatRealizedGainTests(unittest.TestCase):
    """`compute_realized_gains` reuses the SAME machinery for `htsat` — no
    HTSAT-specific code path — but `C-LN`'s own entry for `htsat` carries a
    `note` since `htsat`'s `D1` disables `layer_norm_fused` AND
    `gelu_erf_fused` together (module doc, "HTSAT")."""

    def _leg(self, leg_id, role, verdict, busy, dtype="f32"):
        return {
            "leg_id": leg_id,
            "tower": "htsat",
            "dtype": dtype,
            "verdict": verdict,
            "reasons": [],
            "chains": {attribute.CHAIN_LN: _chain_result(status="absent")},
            "_role": role,
            "_gpu_busy_us_per_step": busy,
        }

    def test_c_lora_htsat_a1_vs_d2(self):
        legs = [
            self._leg("htsat-A1", "A", attribute.VERDICT_VALID, 234764.5),
            self._leg("htsat-D2", "D2", attribute.VERDICT_VALID, 260000.0),
        ]
        merge_by_leg_id = {
            "htsat-A1": _merge_row("htsat-A1", wall_s_per_step=1.55),
            "htsat-D2": _merge_row("htsat-D2", wall_s_per_step=1.65),
        }
        gains = attribute.compute_realized_gains(legs, merge_by_leg_id)
        lora = next(g for g in gains if g["chain"] == attribute.CHAIN_LORA and g["tower"] == "htsat")
        self.assertGreater(lora["busy_delta_us_per_step"], 0.0)
        self.assertNotIn("note", lora)

    def test_c_ln_htsat_d1_vs_d2_carries_joint_note(self):
        legs = [
            self._leg("htsat-A1", "A", attribute.VERDICT_VALID, 234764.5),
            self._leg("htsat-D1", "D1", attribute.VERDICT_VALID, 348814.4),
            self._leg("htsat-D2", "D2", attribute.VERDICT_VALID, 260000.0),
        ]
        merge_by_leg_id = {
            "htsat-A1": _merge_row("htsat-A1", wall_s_per_step=1.55),
            "htsat-D1": _merge_row("htsat-D1", wall_s_per_step=1.69),
            "htsat-D2": _merge_row("htsat-D2", wall_s_per_step=1.65),
        }
        gains = attribute.compute_realized_gains(legs, merge_by_leg_id)
        ln = next(g for g in gains if g["chain"] == attribute.CHAIN_LN and g["tower"] == "htsat")
        self.assertIn("note", ln)
        self.assertIn("JOINT", ln["note"])
        self.assertIn("C-GELU-HTSAT", ln["note"])

    def test_clip_ln_never_carries_the_joint_note(self):
        """The `note` is scoped to `htsat` ONLY — CLIP's own `C-LN` (D1
        disables `layer_norm_fused` alone) never carries it."""
        legs = [
            {
                "leg_id": "clip-text-D1",
                "tower": "clip-text",
                "dtype": "f32",
                "verdict": attribute.VERDICT_VALID,
                "reasons": [],
                "chains": {attribute.CHAIN_LN: _chain_result(status="absent")},
                "_role": "D1",
                "_gpu_busy_us_per_step": 89000.0,
            },
            {
                "leg_id": "clip-text-D2",
                "tower": "clip-text",
                "dtype": "f32",
                "verdict": attribute.VERDICT_VALID,
                "reasons": [],
                "chains": {attribute.CHAIN_LN: _chain_result(status="absent")},
                "_role": "D2",
                "_gpu_busy_us_per_step": 79000.0,
            },
        ]
        gains = attribute.compute_realized_gains(legs, {})
        ln = next(g for g in gains if g["chain"] == attribute.CHAIN_LN and g["tower"] == "clip-text")
        self.assertNotIn("note", ln)


class CliEndToEndTests(unittest.TestCase):
    def _write_merge_json(self, tmp: Path, *leg_ids: str) -> Path:
        merge_path = Path(tmp) / "merge.json"
        merge_path.write_text(json.dumps(_merge_report_for(*leg_ids)), encoding="utf-8")
        return merge_path

    def test_main_writes_a_report_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            merge_path = self._write_merge_json(tmp, "clip-text-A1")
            out_path = Path(tmp) / "out.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ATTRIBUTE),
                    "--legs-dir",
                    str(legs_dir),
                    "--merge-json",
                    str(merge_path),
                    "--out",
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["legs_valid"], 1)
            self.assertIn("clip-text-A1", result.stderr)

    def test_main_refuses_a_missing_legs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            merge_path = self._write_merge_json(tmp, "clip-text-A1")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ATTRIBUTE),
                    "--legs-dir",
                    "/nonexistent/path/xyz",
                    "--merge-json",
                    str(merge_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)

    def test_main_refuses_without_merge_json_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            result = subprocess.run(
                [sys.executable, str(ATTRIBUTE), "--legs-dir", str(legs_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_main_refuses_a_missing_merge_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            result = subprocess.run(
                [
                    sys.executable,
                    str(ATTRIBUTE),
                    "--legs-dir",
                    str(legs_dir),
                    "--merge-json",
                    "/nonexistent/merge.json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)

    def test_main_refuses_when_a_legs_dir_leg_has_no_merge_json_row(self):
        """Pass 3, finding 5: `--legs-dir`'s own leg set must be a SUBSET
        of `--merge-json`'s — a leg this run would attribute but the merge
        report never mentions can never be certified `decision_grade`, so
        `main` refuses rather than silently reading it as `False` and
        moving on."""
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            merge_path = self._write_merge_json(tmp, "clip-text-D2")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ATTRIBUTE),
                    "--legs-dir",
                    str(legs_dir),
                    "--merge-json",
                    str(merge_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("different legs set", result.stderr)
            self.assertIn("clip-text-A1", result.stderr)

    def test_main_accepts_a_merge_json_covering_extra_legs_not_in_legs_dir(self):
        """The merge report MAY name legs this run does NOT attribute
        (e.g. an HTSAT leg out of this pass's own scope) without tripping
        the refusal — only UNDER-coverage (a `--legs-dir` leg missing from
        the merge) is unsafe."""
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            merge_path = self._write_merge_json(tmp, "clip-text-A1", "htsat-A1")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ATTRIBUTE),
                    "--legs-dir",
                    str(legs_dir),
                    "--merge-json",
                    str(merge_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
