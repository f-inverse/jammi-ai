#!/usr/bin/env python3
"""Tests for `torch_grad_oracle.py`'s NAME TRANSLATION functions only.

These are the one piece of `torch_grad_oracle.py` this environment CAN
actually exercise without `torch`/`transformers`/`peft` installed (that
module's own `import torch_finetune_step as tfs` succeeds even without
torch present, because `torch_finetune_step.py` imports `torch` lazily
inside its functions, never at module scope — so importing the NAME
TRANSLATION functions, which touch no torch API at all, works here).
Everything else in `torch_grad_oracle.py` (`run`, `load_lora_weights_into_model`,
`dump_lora_weights_from_model`) is UNTESTED in this environment — see that
file's own PROVENANCE note.

This suite exists because a real bug was caught by exactly this kind of
check while writing `torch_grad_oracle.py`: an early
`translate_jammi_name_to_peft` used a `"mlp." in site` heuristic that
mis-routed `Wi` (an mlp site with NO "mlp." prefix in jammi's own naming —
only `mlp.Wo` carries the prefix, to disambiguate from attn's own `Wo`) to
`attn.Wi`, which does not exist in the model at all. Pinned here so it
cannot silently regress.

Run directly: `python3 crates/jammi-bench/reference/test_torch_grad_oracle_names.py`
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_grad_oracle as tgo  # noqa: E402


ALL_SITES = [
    ("attn", "Wqkv"),
    ("attn", "Wo"),
    ("mlp", "Wi"),
    ("mlp", "Wo"),
]


class NameTranslationTests(unittest.TestCase):
    def test_peft_to_jammi_matches_the_empirically_confirmed_convention(self):
        """`layer.0.Wqkv.lora_a` is not a guess -- it is the LITERAL key
        this round's own `jammi-bench grad-oracle` CLI run produced against
        the tiny fixture (see the dispatch verdict). Every OTHER site here
        follows the SAME `layer.{n}.{target_name}.lora_[ab]` shape by
        construction (`LoraSite::build`'s `target_name` argument, cited in
        `torch_grad_oracle.py`'s own module doc) -- this test pins the
        empirically-confirmed one plus the three siblings the source
        reading predicts.
        """
        cases = {
            "base_model.model.layers.0.attn.Wqkv.lora_A.default.weight": "layer.0.Wqkv.lora_a",
            "base_model.model.layers.0.attn.Wo.lora_B.default.weight": "layer.0.Wo.lora_b",
            "base_model.model.layers.3.mlp.Wi.lora_A.default.weight": "layer.3.Wi.lora_a",
            "base_model.model.layers.3.mlp.Wo.lora_B.default.weight": "layer.3.mlp.Wo.lora_b",
        }
        for peft_name, expected_jammi_name in cases.items():
            self.assertEqual(tgo.translate_peft_name_to_jammi(peft_name), expected_jammi_name)

    def test_non_lora_parameter_names_translate_to_none(self):
        self.assertIsNone(tgo.translate_peft_name_to_jammi("base_model.model.layers.0.attn.Wqkv.weight"))
        self.assertIsNone(tgo.translate_peft_name_to_jammi("base_model.model.embeddings.tok_embeddings.weight"))

    def test_round_trip_every_site_and_matrix(self):
        """Every `(mid, site) x {A, B} x {layer 0, layer 27}` combination
        must round-trip peft-name -> jammi-name -> peft-name EXACTLY. This
        is the test that would have caught the `Wi` mis-routing bug this
        module's own docstring describes -- `Wi` (an mlp site with no
        'mlp.' prefix in jammi's naming) is the one combination a naive
        'mlp.' in site heuristic gets wrong.
        """
        for mid, site in ALL_SITES:
            for ab in ("A", "B"):
                for layer in (0, 27):
                    peft_name = f"base_model.model.layers.{layer}.{mid}.{site}.lora_{ab}.default.weight"
                    jammi_name = tgo.translate_peft_name_to_jammi(peft_name)
                    self.assertIsNotNone(jammi_name, f"{peft_name} failed to translate at all")
                    back = tgo.translate_jammi_name_to_peft(jammi_name)
                    self.assertEqual(
                        back,
                        peft_name,
                        f"round trip broken for layer={layer} mid={mid} site={site} ab={ab}: "
                        f"{peft_name} -> {jammi_name} -> {back}",
                    )

    def test_wi_specifically_does_not_get_routed_to_attn(self):
        """The exact regression this suite exists to pin: `Wi` is an MLP
        site with NO 'mlp.' prefix in jammi's own naming -- a heuristic
        keyed on the STRING 'mlp.' would (and once did, in an earlier
        draft of this file) misroute it to `attn.Wi`, a parameter that
        does not exist in the model.
        """
        peft_name = "base_model.model.layers.4.mlp.Wi.lora_A.default.weight"
        jammi_name = tgo.translate_peft_name_to_jammi(peft_name)
        self.assertEqual(jammi_name, "layer.4.Wi.lora_a")
        back = tgo.translate_jammi_name_to_peft(jammi_name)
        self.assertIn(".mlp.Wi.", back, f"Wi must route back to the mlp sub-module, got {back!r}")
        self.assertNotIn(".attn.Wi.", back)

    def test_wo_disambiguates_attn_from_mlp_both_directions(self):
        """`Wo` is the one site name peft's own suffix-matching rule
        (documented in `torch_finetune_step.py`'s module doc) applies to
        BOTH `attn.Wo` and `mlp.Wo` — jammi's own naming disambiguates by
        keeping attn's bare (`Wo`) and prefixing mlp's (`mlp.Wo`); this
        test pins that BOTH directions preserve which sub-module a given
        `Wo` belongs to.
        """
        attn_jammi = tgo.translate_peft_name_to_jammi("base_model.model.layers.1.attn.Wo.lora_A.default.weight")
        mlp_jammi = tgo.translate_peft_name_to_jammi("base_model.model.layers.1.mlp.Wo.lora_A.default.weight")
        self.assertEqual(attn_jammi, "layer.1.Wo.lora_a")
        self.assertEqual(mlp_jammi, "layer.1.mlp.Wo.lora_a")
        self.assertNotEqual(attn_jammi, mlp_jammi)
        self.assertIn(".attn.Wo.", tgo.translate_jammi_name_to_peft(attn_jammi))
        self.assertIn(".mlp.Wo.", tgo.translate_jammi_name_to_peft(mlp_jammi))

    def test_malformed_jammi_name_returns_none_not_a_crash(self):
        self.assertIsNone(tgo.translate_jammi_name_to_peft("not.a.valid.name"))
        self.assertIsNone(tgo.translate_jammi_name_to_peft("layer.3.UnknownSite.lora_a"))


if __name__ == "__main__":
    unittest.main()
