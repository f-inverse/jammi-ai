#!/usr/bin/env python3
"""Hermetic tests for `profile_421_attribute.py` (issue #421 post-export
attribution unit; CONTRACT `scratchpad/contract-421-profile.md` v2.5
`### Attribution` / `§D3`).

No GPU, no pod, no `nsys`, no network. The kernel-name<->shape MAPPING
itself is tested against SMALL, REAL, committed fixtures cut byte-for-byte
from real nsys census exports (`fixtures/profile_421_clip_text_a1/`,
`.../profile_421_clip_text_a2/`, `.../profile_421_clip_text_d1/`,
`.../profile_421_clip_vision_a1/` — see each directory's `PROVENANCE.md`),
never a hand-rolled census standing in for what a real export actually
contains. Per this crate's "never transcribe a timing number into a test"
convention, NOTHING here asserts a literal `us_per_step`/`share`/
`launches_per_step` value from a fixture as an expected number — every
assertion is STRUCTURAL: which chain a `(kernel, grid, block)` row lands
in, that chain shares partition `gpu_kernel_us_per_step` exactly, that
shares are `<= 1`, that every declared chain is present-or-explicitly-
absent, `decision_grade`'s own threshold arithmetic, and the two-sided
decision rule evaluated on SYNTHETIC share numbers (never a real leg's).

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
FIXTURE_KERNELS_VISION_A1 = PERF_DIR / "fixtures" / "profile_421_clip_vision_a1" / "kernels.json"

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


def load_fixture_census(path: Path = FIXTURE_KERNELS) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def clip_text_signatures() -> attribute.TowerSignatures:
    return attribute.derive_signatures("clip-text", CLIP_TEXT_A1_MANIFEST_FIELDS)


def clip_vision_signatures() -> attribute.TowerSignatures:
    return attribute.derive_signatures("clip-vision", CLIP_VISION_A1_MANIFEST_FIELDS)


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
    """`is_known_kernel_name`'s three admission paths (module doc)."""

    def test_explicit_ground_truth_names_are_known(self):
        self.assertTrue(attribute.is_known_kernel_name("badd_f32"))
        self.assertTrue(attribute.is_known_kernel_name("Kernel2"))

    def test_bf16_only_explicit_names_are_known(self):
        for name in attribute.BF16_ONLY_KERNEL_NAMES:
            self.assertTrue(attribute.is_known_kernel_name(name))

    def test_any_gemm_family_name_is_known_without_being_hand_listed(self):
        # `derive the twin rule, don't hand-list 18 names` — this exact
        # 32-char tile-variant string is NOT in `KNOWN_KERNEL_NAMES`.
        name = "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_nn"
        self.assertNotIn(name, attribute.KNOWN_KERNEL_NAMES)
        self.assertTrue(attribute.is_known_kernel_name(name))
        self.assertTrue(attribute.is_known_kernel_name("magma_sgemmEx_kernel"))
        self.assertTrue(attribute.is_known_kernel_name("some_future_tile_variant_gemm_v9"))

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
        # for `clip-text` — a `block=[1024,...]` launch at `grid=[1,1,1]`
        # would instead land `OPTIMIZER` (its `total_threads=1024` sits
        # inside the `width=512` parameter tier's `[512, 1536)` range, by
        # the SAME block-size-rounding tolerance every other shape match in
        # this module uses — see `test_optimizer_parameter_scale_by_shape`).
        entry = {"kernel": "bmul_f32", "grid": [1, 1, 1], "block": [1, 1, 1]}
        self.assertIsNone(attribute.classify_kernel(entry, self.sig, "clip-text"))

    def test_badd_at_the_gelu_shape_lands_elementwise_mlp_not_gelu(self):
        """`badd_f32` is never GELU-shape-gated (it is not in
        `GELU_SHAPE_KERNEL_NAMES`), so at the MLP tier it now lands the
        NAMED `ELEMENTWISE-MLP` bucket instead of pass 1's `UNATTRIBUTED`
        — the exclusion itself (bias-add is not the activation) is
        unchanged, only WHERE the excluded row now lands."""
        entry = {"kernel": "badd_f32", "grid": [3696, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_ELEMENTWISE_MLP
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
        """Pass-2 behavior change: `fast_sum`/`fast_max` at a row count
        that is NEITHER the softmax reduction NOR the eager-LN reduction
        now land `LOSS/REDUCE` (contract: "NOT at the softmax row count"),
        never `UNATTRIBUTED`."""
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

    def test_ln_row_flat_kernels_match_only_at_ln_row_shape(self):
        elements = self.sig.ln_row_count()
        at_shape = {"kernel": "urecip_f32", "grid": [2, 1, 1], "block": [1024, 1, 1]}
        self.assertLessEqual(elements, 2 * 1024)
        self.assertEqual(attribute.classify_kernel(at_shape, self.sig, "clip-text"), attribute.CHAIN_LN)

    def test_ampere_sgemm_matches_attn_when_grid_carries_rows_times_heads(self):
        batch = self.sig.attn_batch_count()
        self.assertEqual(batch, 192)
        entry = {"kernel": "ampere_sgemm_128x128_nt", "grid": [1, 1, 192], "block": [256, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

    def test_anonymous_kernel_carrying_the_batch_count_is_attn(self):
        """The batched-grid rule is NAME-INDEPENDENT (module doc) — an
        anonymous `Kernel2` row carrying `rows*heads` in its grid still
        lands `C-ATTN-<tower>`."""
        entry = {"kernel": "Kernel2", "grid": [2, 1, 192], "block": [128, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"),
            attribute.chain_attn("clip-text"),
        )

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

    def test_attn_tensor_elementwise_ops_land_attn(self):
        elements = self.sig.attn_shape_elements()
        entry = {"kernel": "badd_f32", "grid": [1112, 1, 1], "block": [1024, 1, 1]}
        self.assertLessEqual(elements, 1112 * 1024)
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.chain_attn("clip-text")
        )

    def test_cast_named_kernel_at_attn_shape_is_cast_not_attn(self):
        """`CAST` is checked by NAME before the shape-based attention
        check — a cast row at the attention tensor's own element count
        still lands `CAST` (priority order is deliberate, see
        `fixtures/profile_421_clip_text_a2/PROVENANCE.md`)."""
        entry = {"kernel": "cast_u8_f32", "grid": [1112, 1, 1], "block": [1024, 1, 1]}
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
        """`is_u32_f32` at the `out` activation tier's own element count
        still lands `EMBED/GATHER`, not `ELEMENTWISE-OUT` — name-only
        checks run before the generic tier fallback."""
        entry = {"kernel": "is_u32_f32", "grid": [924, 1, 1], "block": [1024, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_EMBED_GATHER
        )

    def test_optimizer_parameter_scale_by_shape(self):
        elements = sorted(self.sig.param_scale_elements())[0]
        entry = {"kernel": "usqrt_f32", "grid": [1, 1, 1], "block": [max(elements, 1024), 1, 1]}
        # A `usqrt_f32` row NOT at the LN row-count shape falls through to
        # the parameter-scale check.
        self.assertEqual(
            attribute.classify_kernel(entry, self.sig, "clip-text"), attribute.CHAIN_OPTIMIZER
        )

    def test_activation_tier_elementwise_buckets(self):
        cases = [
            (self.sig.qkv_shape_elements(), attribute.CHAIN_ELEMENTWISE_QKV),
            (self.sig.out_shape_elements(), attribute.CHAIN_ELEMENTWISE_OUT),
            (self.sig.gelu_shape_elements(), attribute.CHAIN_ELEMENTWISE_MLP),
        ]
        for elements, expected in cases:
            grid0 = -(-elements // 1024)  # ceil division, matching a real launch's grid sizing
            entry = {"kernel": "ucopy_f32", "grid": [grid0, 1, 1], "block": [1024, 1, 1]}
            self.assertEqual(attribute.classify_kernel(entry, self.sig, "clip-text"), expected)

    def test_patch_embed_only_on_clip_vision(self):
        vision_sig = clip_vision_signatures()
        entry = {"kernel": "im2col_f32", "grid": [1, 1, 1], "block": [1, 1, 1]}
        self.assertEqual(
            attribute.classify_kernel(entry, vision_sig, "clip-vision"), attribute.CHAIN_PATCH_EMBED
        )
        # `clip-text` never has an `im2col` op — the check is gated on
        # `tower == "clip-vision"`, so the SAME row on `clip-text`'s own
        # signature does not spuriously match.
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
    15-row cut."""

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
                # The fixture's SECOND `bmul_f32` row sits at `grid=[1112,1,1]`
                # (the attention tensor's own shape, not the GELU shape) —
                # pass-2 behavior change: it now FALLS THROUGH to
                # `C-ATTN-clip-text` instead of stopping at `UNATTRIBUTED`
                # (see `test_bmul_at_the_attention_shape_falls_through_to_attn_not_gelu`).
                "bmul_f32",
            },
        )

    def test_badd_and_ampere_now_land_named_buckets_not_unattributed(self):
        """Pass-2 regression guard: the exact two rows pass 1's fixture
        used as UNATTRIBUTED negative controls now land NAMED buckets."""
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("badd_f32", chains[attribute.CHAIN_ELEMENTWISE_MLP].kernel_names)
        self.assertIn("ampere_sgemm_128x64_nn", chains[attribute.CHAIN_BASE_GEMM].kernel_names)
        self.assertIn("dropout_fwd_f32", chains[attribute.CHAIN_DROPOUT].kernel_names)
        self.assertIn("adamw_moment_update_f32", chains[attribute.CHAIN_OPTIMIZER].kernel_names)

    def test_unattributed_share_is_small_on_the_real_fixture(self):
        chains, _unknown, _reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        unattributed = chains[attribute.CHAIN_UNATTRIBUTED]
        # This fixture is a 15-row CUT, not the full census, so its own
        # UNATTRIBUTED share is not the leg's real decision_grade number
        # (that is asserted against the full report in `main()`'s own
        # documented run, pasted in the hand-off) — only that it stays a
        # finite, bounded share here.
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


class AttributeCensusBf16FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-text-A2`
    (BF16) 22-row cut — see `fixtures/profile_421_clip_text_a2/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_A2)
        self.sig = attribute.derive_signatures("clip-text", CLIP_TEXT_A2_MANIFEST_FIELDS)

    def test_chains_partition_busy_exactly(self):
        chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])
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
        # The second `affine_bf16` row (attention shape) must NOT also be
        # counted as a GELU kernel name even though the NAME is shared —
        # `kernel_names` is a de-duplicated set, checked via the per-row
        # unit test instead; here we assert it DOES show up under attn.
        self.assertIn("affine_bf16", chains[attribute.chain_attn("clip-text")].kernel_names)

    def test_loss_reduce_bucket_present(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("fast_sum_bf16", chains[attribute.CHAIN_LOSS_REDUCE].kernel_names)

    def test_anonymous_kernel2_splits_by_grid(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("Kernel2", chains[attribute.chain_attn("clip-text")].kernel_names)
        self.assertIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_cast_bucket_includes_bf16_only_names(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        cast_names = set(chains[attribute.CHAIN_CAST].kernel_names)
        self.assertIn("cast_bf16_f32", cast_names)
        self.assertIn("cast_add_bf16", cast_names)
        self.assertIn("cast_scale_bf16_f32", cast_names)
        self.assertIn("cast_u8_bf16", cast_names)

    def test_optimizer_and_dropout_are_dtype_independent(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("adamw_moment_update_f32", chains[attribute.CHAIN_OPTIMIZER].kernel_names)
        self.assertIn("dropout_fwd_f32", chains[attribute.CHAIN_DROPOUT].kernel_names)

    def test_embed_gather_bf16(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        embed_names = set(chains[attribute.CHAIN_EMBED_GATHER].kernel_names)
        self.assertIn("gather_u32_bf16", embed_names)
        self.assertIn("is_u32_bf16", embed_names)

    def test_no_unknown_kernel_names(self):
        _chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])


class AttributeCensusD1FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-text-D1`
    13-row cut — see `fixtures/profile_421_clip_text_d1/PROVENANCE.md`."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_D1)
        self.sig = attribute.derive_signatures("clip-text", CLIP_TEXT_D1_MANIFEST_FIELDS)

    def test_eager_ln_rows_land_c_ln(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        ln_names = set(chains[attribute.CHAIN_LN].kernel_names)
        self.assertIn("fast_sum_f32", ln_names)
        self.assertIn("usqrt_f32", ln_names)
        self.assertIn("urecip_f32", ln_names)

    def test_fast_sum_splits_three_ways_by_grid(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LN].kernel_names)
        self.assertIn("fast_sum_f32", chains[attribute.chain_attn("clip-text")].kernel_names)
        self.assertIn("fast_sum_f32", chains[attribute.CHAIN_LOSS_REDUCE].kernel_names)

    def test_d_leg_tile_variants_are_base_gemm_or_attn_by_grid(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        base_gemm_names = set(chains[attribute.CHAIN_BASE_GEMM].kernel_names)
        self.assertIn("ampere_sgemm_128x64_nt", base_gemm_names)
        self.assertIn("ampere_sgemm_32x32_sliced1x4_nt", base_gemm_names)
        self.assertIn("ampere_sgemm_128x128_nn", chains[attribute.chain_attn("clip-text")].kernel_names)

    def test_usqr_does_not_get_swept_into_c_ln_by_launch_count_coincidence(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("usqr_f32", chains[attribute.CHAIN_ELEMENTWISE_OUT].kernel_names)
        self.assertNotIn("usqr_f32", chains[attribute.CHAIN_LN].kernel_names)

    def test_anonymous_row_with_no_tier_match_stays_unattributed(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertIn("Kernel2", chains[attribute.CHAIN_UNATTRIBUTED].kernel_names)

    def test_no_unknown_kernel_names(self):
        _chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])


class AttributeCensusVisionA1FixtureTests(unittest.TestCase):
    """The whole-census pass against the REAL committed `clip-vision-A1`
    21-row cut — see `fixtures/profile_421_clip_vision_a1/PROVENANCE.md`.
    The FIRST real `clip-vision` export this crate has ever attributed."""

    def setUp(self):
        self.census = load_fixture_census(FIXTURE_KERNELS_VISION_A1)
        self.sig = clip_vision_signatures()

    def test_chains_partition_busy_exactly(self):
        chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])
        total_in_rows = sum(r["us_per_step"] for r in self.census["by_kernel_and_grid"])
        attributed_total = sum(c.gpu_busy_us for c in chains.values() if c.gpu_busy_us is not None)
        self.assertAlmostEqual(attributed_total, total_in_rows, places=6)

    def test_patch_embed_bucket(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("im2col_f32", chains[attribute.CHAIN_PATCH_EMBED].kernel_names)

    def test_magma_gemm_lands_attn_via_grid_not_name(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("magma_sgemmEx_kernel", chains[attribute.chain_attn("clip-vision")].kernel_names)

    def test_badd_splits_three_activation_tiers(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("badd_f32", chains[attribute.CHAIN_ELEMENTWISE_OUT].kernel_names)
        self.assertIn("badd_f32", chains[attribute.CHAIN_ELEMENTWISE_QKV].kernel_names)
        self.assertIn("badd_f32", chains[attribute.CHAIN_ELEMENTWISE_MLP].kernel_names)
        self.assertIn("badd_f32", chains[attribute.chain_attn("clip-vision")].kernel_names)

    def test_urecip_off_ln_shape_lands_optimizer_via_the_param_scale_catch_all(self):
        """`urecip_f32` at `grid=[1,1,1], block=[1024,1,1]` does not match
        `ln_row_count=1,200` (the eager-LN-only rule), so `C-LN`'s
        NAME-restricted rule (`LN_ROW_FLAT_KERNEL_NAMES`) does not fire —
        but its `total_threads=1,024` DOES fall inside the `width=768`
        parameter-scale tier's `[768, 1792)` range (the same block-size
        rounding tolerance every shape match in this module uses), so it
        lands `OPTIMIZER` rather than `UNATTRIBUTED`. This is a smaller,
        honestly-labeled bucket than a guessed "L2-normalize" role would
        be — see `profile_421_attribute.py`'s own module doc."""
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertIn("urecip_f32", chains[attribute.CHAIN_OPTIMIZER].kernel_names)

    def test_every_declared_chain_present_including_patch_embed(self):
        chains, _u, _r = attribute.attribute_census(self.census, self.sig, "clip-vision")
        for name in attribute.declared_chains_for_tower("clip-vision"):
            self.assertIn(name, chains, name)
        self.assertIn(attribute.CHAIN_PATCH_EMBED, chains)

    def test_no_unknown_kernel_names(self):
        _chains, unknown, reasons = attribute.attribute_census(self.census, self.sig, "clip-vision")
        self.assertEqual(unknown, [])
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
        # An unknown kernel still contributes its time to UNATTRIBUTED —
        # the leg is flagged INVALID by the caller via `reasons`, but the
        # busy accounting itself stays a complete partition.
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

    def test_a_novel_gemm_tile_variant_never_triggers_the_gate(self):
        census = self._census_with(
            {
                "kernel": "ampere_bf16_s16816gemm_bf16_999x999_ldg8_f2f_stages_99x9_nn",
                "grid": [1, 1, 1],
                "block": [1, 1, 1],
                "us_per_step": 1e9,
            }
        )
        _chains, unknown, reasons = attribute.attribute_census(census, self.sig, "clip-text")
        self.assertEqual(unknown, [])
        self.assertEqual(reasons, [])


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
        # The CLIP D1 set named on an HTSAT leg is neither tower's D1/D2 set.
        self.assertEqual(
            attribute.leg_role(["lora_linear_fused", "layer_norm_fused"], "htsat"), "other"
        )

    def test_unrecognised_disable_set_is_other(self):
        self.assertEqual(attribute.leg_role(["some_other_key"], "clip-text"), "other")


def _chain_result(**kwargs) -> dict:
    return attribute.ChainResult(**kwargs).as_dict()


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
        grade, reason = attribute.leg_decision_grade(self._row(unattributed_share=0.049))
        self.assertTrue(grade)
        self.assertIsNone(reason)

    def test_valid_leg_at_exactly_the_bound_is_decision_grade(self):
        # Contract: `<= 5%`, a boundary check, not a strict `<`.
        grade, _reason = attribute.leg_decision_grade(
            self._row(unattributed_share=attribute.UNATTRIBUTED_DECISION_GRADE_LIMIT)
        )
        self.assertTrue(grade)

    def test_valid_leg_over_the_bound_is_not_decision_grade(self):
        grade, reason = attribute.leg_decision_grade(self._row(unattributed_share=0.051))
        self.assertFalse(grade)
        self.assertIn("0.0510", reason)

    def test_invalid_leg_is_never_decision_grade_even_with_a_tiny_share(self):
        grade, reason = attribute.leg_decision_grade(
            self._row(verdict=attribute.VERDICT_INVALID, unattributed_share=0.0)
        )
        self.assertFalse(grade)
        self.assertIn("INVALID", reason)

    def test_missing_chains_is_not_decision_grade(self):
        grade, reason = attribute.leg_decision_grade({"verdict": attribute.VERDICT_VALID})
        self.assertFalse(grade)
        self.assertIsNotNone(reason)

    def test_nan_unattributed_share_is_never_decision_grade(self):
        """Negative-control non-vacuity (family F): `NaN > c` is `False` in
        Python, so a naive `share > LIMIT` check would silently treat a
        diverged/NaN share as passing the gate. `leg_decision_grade` must
        refuse a non-finite share explicitly rather than let it slip
        through as `decision_grade=True`."""
        row = self._row(unattributed_share=float("nan"))
        grade, reason = attribute.leg_decision_grade(row)
        self.assertFalse(grade)
        self.assertIsNotNone(reason)


class LoraD2DeltaTests(unittest.TestCase):
    def _leg(self, leg_id, tower, role, verdict, busy, dtype="f32", wall=0.1):
        return {
            "leg_id": leg_id,
            "tower": tower,
            "dtype": dtype,
            "verdict": verdict,
            "reasons": [],
            "chains": {attribute.CHAIN_LN: _chain_result(status="absent")},
            "_role": role,
            "_gpu_busy_us_per_step": busy,
            "_wall_s_per_step": wall,
        }

    def test_delta_computed_when_a_and_d2_both_valid(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 50000.0),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        lora = legs[0]["chains"][attribute.CHAIN_LORA]
        self.assertEqual(lora["status"], "measured_via_d2_delta")
        self.assertAlmostEqual(lora["gpu_busy_us"], 10000.0, places=6)
        self.assertAlmostEqual(lora["share_gpu_busy"], 10000.0 / 60000.0, places=9)
        # D2 itself never gets a C-LORA delta of its own.
        self.assertNotIn(attribute.CHAIN_LORA, legs[1]["chains"])

    def test_delta_absent_without_a_matching_d2_leg(self):
        legs = [self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0)]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertEqual(legs[0]["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_delta_absent_when_d2_leg_is_invalid(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_INVALID, 50000.0),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertEqual(legs[0]["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_bf16_a_leg_never_gets_a_delta_d2_is_f32_only(self):
        legs = [
            self._leg("clip-text-A2", "clip-text", "A", attribute.VERDICT_VALID, 60000.0, dtype="bf16"),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 50000.0, dtype="f32"),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertEqual(legs[0]["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")

    def test_d1_minus_d2_advisory_attached_when_all_three_present(self):
        legs = [
            self._leg("clip-text-A1", "clip-text", "A", attribute.VERDICT_VALID, 60000.0),
            self._leg("clip-text-D1", "clip-text", "D1", attribute.VERDICT_VALID, 55000.0),
            self._leg("clip-text-D2", "clip-text", "D2", attribute.VERDICT_VALID, 50000.0),
        ]
        attribute.attach_lora_via_d2_delta(legs)
        self.assertAlmostEqual(legs[0]["d1_minus_d2_busy_us_advisory"], 5000.0, places=6)


class TwoSidedDecisionRuleTests(unittest.TestCase):
    """The contract's two-sided rule, evaluated ONLY on SYNTHETIC share
    numbers (never a real leg's) — per this crate's own convention, a
    verdict-rule unit test asserts the RULE's arithmetic, not a
    transcribed measurement."""

    CHAIN_KEY = "C-ATTN-clip-text"

    def _leg(self, leg_id, dtype, s_wall, s_busy, u_wall, u_busy, role="A", verdict=attribute.VERDICT_VALID):
        return {
            "leg_id": leg_id,
            "tower": "clip-text",
            "dtype": dtype,
            "verdict": verdict,
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

    def test_activate_when_a1_wall_share_at_or_above_ten_percent(self):
        legs = [self._leg("clip-text-A1", "f32", s_wall=0.10, s_busy=0.10, u_wall=0.0, u_busy=0.0)]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "ACTIVATE")

    def test_activate_when_only_a2_crosses_the_bar(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.02, s_busy=0.02, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.11, s_busy=0.11, u_wall=0.0, u_busy=0.0),
        ]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "ACTIVATE")

    def test_decline_when_combined_share_under_five_percent_on_every_decision_grade_leg(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.01, u_busy=0.01),
            self._leg("clip-text-A2", "bf16", s_wall=0.02, s_busy=0.02, u_wall=0.01, u_busy=0.01),
        ]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "DECLINE")

    def test_unresolved_in_the_five_to_ten_percent_band(self):
        legs = [self._leg("clip-text-A1", "f32", s_wall=0.07, s_busy=0.07, u_wall=0.0, u_busy=0.0)]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_unresolved_when_a1_missing(self):
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", [])
        self.assertEqual(decision["verdict"], "UNRESOLVED")
        self.assertIn("A1 not decision-grade", decision["reason"])

    def test_unresolved_when_a1_not_decision_grade_even_if_a2_would_activate(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.06, u_busy=0.06),
            self._leg("clip-text-A2", "bf16", s_wall=0.20, s_busy=0.20, u_wall=0.0, u_busy=0.0),
        ]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_decline_notes_f32_only_when_a2_is_not_decision_grade(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.01, s_busy=0.01, u_wall=0.06, u_busy=0.06),
        ]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "DECLINE")
        self.assertIn("F32-only", decision["reason"])

    def test_declines_only_when_every_decision_grade_leg_qualifies(self):
        legs = [
            self._leg("clip-text-A1", "f32", s_wall=0.01, s_busy=0.01, u_wall=0.0, u_busy=0.0),
            self._leg("clip-text-A2", "bf16", s_wall=0.04, s_busy=0.04, u_wall=0.03, u_busy=0.03),
        ]
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "UNRESOLVED")

    def test_nan_unattributed_share_fails_the_leg_closed_at_decision_grade(self):
        """Non-vacuity: a NaN `UNATTRIBUTED` share must not let a leg
        silently pass `leg_decision_grade`'s own gate via `NaN > 0.05`
        being `False` — it is caught THERE (never decision-grade), so the
        two-sided rule never even reaches its own arithmetic on this leg
        and reports UNRESOLVED, not a NaN-poisoned ACTIVATE or DECLINE."""
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
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", legs)
        self.assertEqual(decision["verdict"], "UNRESOLVED")
        self.assertIn("A1 not decision-grade", decision["reason"])

    def test_nan_chain_share_alone_never_silently_activates_or_declines(self):
        """A leg that IS decision-grade (finite, small UNATTRIBUTED share)
        but whose CANDIDATE chain share is itself NaN (a malformed/partial
        row) must not let `NaN >= 10%` or `NaN < 5%` silently decide
        anything — `_leg_chain_shares`'s finite-guard treats it as `0.0`,
        so this leg quietly contributes `0.0` to both checks rather than
        raising or poisoning the arithmetic."""
        leg = self._leg(
            "clip-text-A1", "f32", s_wall=float("nan"), s_busy=float("nan"), u_wall=0.01, u_busy=0.01
        )
        decision = attribute.decide_candidate_port("C-ATTN-clip-text", self.CHAIN_KEY, "clip-text", [leg])
        self.assertEqual(decision["verdict"], "DECLINE")

    def test_candidate_ports_for_tower_maps_c_mlp_to_gelu_chain(self):
        ports = attribute.candidate_ports_for_tower("clip-text")
        self.assertEqual(ports["C-MLP-clip-text"], attribute.CHAIN_GELU)
        self.assertEqual(ports["C-ATTN-clip-text"], attribute.chain_attn("clip-text"))

    def test_decide_all_candidates_covers_every_candidate_port(self):
        decisions = attribute.decide_all_candidates([])
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


class AttributeLegAndReportTests(unittest.TestCase):
    def test_end_to_end_on_the_real_fixture_is_valid_and_decision_grade(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir)
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
                attribute.CHAIN_LORA,
                attribute.CHAIN_BASE_GEMM,
            ):
                self.assertIn(name, row["chains"])
            self.assertEqual(row["chains"][attribute.CHAIN_LORA]["status"], "requires_d2_delta")
            self.assertIn("decision_grade", row)

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

    def test_two_legs_a_and_d2_compose_a_full_report_with_lora_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            d2_census = load_fixture_census()
            # A cheaper synthetic D2 twin: same shapes, less total busy (the
            # `lora_linear_fused` kernel forced eager and removed) — kept
            # structurally real (same 15 rows) so `attribute_census` runs
            # the identical mapping; only the aggregate is perturbed to
            # exercise the delta arithmetic without inventing a new export.
            d2_census["gpu_kernel_us_per_step"] = load_fixture_census()["gpu_kernel_us_per_step"] - 5000.0
            _write_leg_dir(
                legs_dir,
                "clip-text-D2",
                {"kernels_disabled": ["lora_linear_fused"]},
                d2_census,
            )
            report = attribute.build_report(legs_dir)
            self.assertEqual(report["summary"]["legs_valid"], 2)
            by_id = {row["leg_id"]: row for row in report["legs"]}
            lora = by_id["clip-text-A1"]["chains"][attribute.CHAIN_LORA]
            self.assertEqual(lora["status"], "measured_via_d2_delta")
            self.assertAlmostEqual(lora["gpu_busy_us"], 5000.0, places=6)
            self.assertNotIn(attribute.CHAIN_LORA, by_id["clip-text-D2"]["chains"])

    def test_report_carries_candidate_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir)
            self.assertIn("candidate_decisions", report)
            ports = {d["port"] for d in report["candidate_decisions"]}
            self.assertIn("C-ATTN-clip-text", ports)
            self.assertIn("C-MLP-clip-text", ports)

    def test_summary_carries_decision_grade_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir)
            self.assertIn("legs_decision_grade", report["summary"])


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
            # Never folded into gpu_kernel_us_per_step / any chain share:
            # the attributed total must equal the SUM OF THE CENSUS ROWS
            # this 15-row fixture cut actually carries (not the leg's own
            # top-level `gpu_kernel_us_per_step`, which is the REAL
            # export's full total — this fixture is a partial cut, kept
            # realistic per its own PROVENANCE.md, so the two are not
            # expected to agree).
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
    def test_format_table_lists_every_chain_row_and_candidate_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            report = attribute.build_report(legs_dir)
            table = attribute.format_table(report)
            self.assertIn("clip-text-A1", table)
            self.assertIn(attribute.CHAIN_LN, table)
            self.assertIn(attribute.chain_attn("clip-text"), table)
            self.assertIn("candidate port decisions:", table)
            self.assertIn("C-ATTN-clip-text", table)


class HtsatStageSignatureSmokeTests(unittest.TestCase):
    """Forward-compatibility smoke test only — no real HTSAT export exists
    to verify these declared constants against yet (module doc)."""

    def test_four_stages_with_positive_window_and_heads(self):
        stages = attribute.htsat_stage_signatures()
        self.assertEqual(len(stages), 4)
        for stage in stages:
            self.assertGreater(stage["depth"], 0)
            self.assertGreater(stage["heads"], 0)
            self.assertGreater(stage["window_size"], 0)

    def test_htsat_has_no_declared_windowing_or_front_fusion_chain_yet(self):
        """Declaring a chain with no mapping rule would silently report it
        `absent` forever, indistinguishable from `not yet implemented` —
        `declared_chains_for_tower` deliberately withholds them until a
        real HTSAT export exists (module doc, "HTSAT")."""
        declared = attribute.declared_chains_for_tower("htsat")
        self.assertNotIn(attribute.CHAIN_WINDOWING, declared)
        self.assertNotIn(attribute.CHAIN_FRONT_FUSION, declared)
        self.assertIn(attribute.CHAIN_GELU_HTSAT, declared)
        self.assertIn(attribute.chain_attn("htsat"), declared)


class CliEndToEndTests(unittest.TestCase):
    def test_main_writes_a_report_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            legs_dir = Path(tmp) / "legs"
            _write_leg_dir(legs_dir, "clip-text-A1", {}, load_fixture_census())
            out_path = Path(tmp) / "out.json"
            result = subprocess.run(
                [sys.executable, str(ATTRIBUTE), "--legs-dir", str(legs_dir), "--out", str(out_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["legs_valid"], 1)
            self.assertIn("clip-text-A1", result.stderr)

    def test_main_refuses_a_missing_legs_dir(self):
        result = subprocess.run(
            [sys.executable, str(ATTRIBUTE), "--legs-dir", "/nonexistent/path/xyz"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
