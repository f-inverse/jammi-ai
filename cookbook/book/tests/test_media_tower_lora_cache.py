"""Cache-backed checks on the committed real-checkpoint media-tower LoRA vertical.

These run on CPU against the committed cache (no GPU, no recompute) and assert the
vertical's load-bearing invariants for each of the three real towers (OpenCLIP
vision, OpenCLIP text, CLAP HTSAT audio): the adapter measurably changes the served
embedding past a tolerance/floor DERIVED from an observed re-emit spread and that
tower's own same-input control (never a bare literal), the adapter round-trips
through a REAL `jammi-server` process restart (not merely a new connection to the
same process), a fixed probe through the OTHER OpenCLIP tower is bit-identical to
base after tuning ONE tower (the cross-tower selectivity oracle), and the
corrupt-media-row contract is recorded — for BOTH a `Binary` and a `Utf8`
(path-valued) input arm — with the error message's own named row index checked
against the row's actual Arrow position.

The committed cache (`artifacts/media_tower/`) is checked into the repo, not
gitignored, so a normal checkout always has it; these tests do not skip when it is
absent — `contracts.load_artifact`/`contracts.golden` raise loudly instead, which
is the correct outcome for a checked-out tree missing a committed artifact.
"""

from __future__ import annotations

import jammi

from jammi_cookbook import contracts

_TOWERS = ("vision", "text", "audio")
_CLIP_TOWERS = ("vision", "text")
_MEDIA_TYPES = ("image", "audio")
_INPUT_MODES = ("bytes", "path")


def test_installed_engine_identity_matches_the_recorded_provenance():
    """A stale committed cache under a newer/older engine must be detectable, not
    silently trusted. Neither `get_server_info()` nor the `jammi-ai` wheel expose a
    build sha (see the record's own `content_digest_via_shipped_surface: false`
    finding); the finest-grained identity the shipped surface exposes is the
    declared VERSION, so that is what this asserts — the same pattern
    `chapters/14-scale/scale.qmd` uses for `usearch.__version__`. FAILS (never
    skips) on a mismatch."""
    record = contracts.load_artifact("media_tower.record")
    provenance = record["provenance"]
    assert jammi.__version__ == provenance["installed_client_version"], (
        f"installed jammi {jammi.__version__} != the {provenance['installed_client_version']} "
        f"this cache was emitted against — a stale cache under a different engine build."
    )
    assert jammi.__version__ == record["engine_version"]
    assert provenance["content_digest_via_shipped_surface"] is False
    assert len(provenance["vision_checkpoint_sha256"]) == 64
    assert len(provenance["audio_checkpoint_sha256"]) == 64


def test_every_tower_changed_and_is_distinguishable_from_the_control():
    record = contracts.load_artifact("media_tower.record")
    towers = record["towers"]
    assert set(towers) == set(_TOWERS), f"expected all towers, got {sorted(towers)}"
    for name, t in towers.items():
        change = t["change_vs_base_max_abs_diff"]
        control = t["same_input_control_max_abs_diff"]
        floor = t["tolerance_derivation"]["change_floor"]
        assert change > floor, (name, change, floor)
        assert control < change, (name, control, change)
        # the recorded golden matches the committed row, at the DERIVED tolerance
        # (never a bare literal — see tolerance_derivation).
        assert contracts.golden(f"media_tower.{name}.change_vs_base_max_abs_diff").contains(change)
        assert contracts.golden(
            f"media_tower.{name}.same_input_control_max_abs_diff"
        ).contains(control)


def test_tolerance_derivation_is_from_an_observed_spread_not_a_bare_literal():
    record = contracts.load_artifact("media_tower.record")
    for name, t in record["towers"].items():
        td = t["tolerance_derivation"]
        assert td["repeats"] == len(t["repeats"]) >= 2, (name, td, t["repeats"])
        observed_spread = max(
            abs(a["change_vs_base_max_abs_diff"] - b["change_vs_base_max_abs_diff"])
            for a in t["repeats"] for b in t["repeats"]
        )
        assert round(observed_spread, 6) == td["observed_change_spread_max_abs_diff"], (
            name, observed_spread, td,
        )
        expected_tol = max(
            td["spread_multiplier_k"] * observed_spread, td["precision_floor"]
        )
        assert round(expected_tol, 6) == td["change_tol"], (name, expected_tol, td)
        expected_floor = max(
            td["control_floor_multiplier_k"] * t["same_input_control_max_abs_diff"],
            td["precision_floor"],
        )
        assert round(expected_floor, 6) == td["change_floor"], (name, expected_floor, td)


def test_every_tower_round_trips_through_a_real_process_restart():
    """A round-trip diff is not noisy like change-vs-base: the golden alone is
    NOT sufficient here, since a broken persistence bug would happily commit
    (and then trivially match) a golden reflecting the break. This asserts a
    tight, HARDCODED ceiling — independent of the golden — matching the
    ceiling the emit script itself refuses to emit a cache past."""
    record = contracts.load_artifact("media_tower.record")
    for name, t in record["towers"].items():
        diff = t["round_trip_max_abs_diff"]
        assert diff < 1e-5, (
            f"{name}: round-trip max|Δ| = {diff:.6f} is not near zero — the "
            "adapter did not survive a real process restart intact"
        )
        assert contracts.golden(f"media_tower.{name}.round_trip_max_abs_diff").contains(diff)
        assert t["model_id"].startswith("jammi:fine-tuned:"), t["model_id"]
        rt = t["round_trip"]
        assert rt["server_restarted"] is True, (name, rt)
        assert rt["before"]["pid"] != rt["after"]["pid"], (
            f"{name}: round trip did not cross a real OS process boundary: {rt}"
        )


def test_cross_tower_selectivity_is_bit_identical_on_the_untouched_tower():
    """After tuning ONE OpenCLIP tower, a fixed probe through the OTHER tower
    must be BIT-IDENTICAL to base (`max_abs_diff == 0.0`, exactly) — `task=`
    scoped the trainable LoRA sites to the tuned tower only."""
    record = contracts.load_artifact("media_tower.record")
    for name in _CLIP_TOWERS:
        sel = record["towers"][name]["cross_tower_selectivity"]
        assert sel["max_abs_diff"] == 0.0, (name, sel)
        assert contracts.golden(f"media_tower.{name}.cross_tower_selectivity_max_abs_diff"
                                 ).contains(sel["max_abs_diff"])
    assert record["towers"]["vision"]["cross_tower_selectivity"]["probed_task"] == "text_embedding"
    assert record["towers"]["text"]["cross_tower_selectivity"]["probed_task"] == "image_embedding"


def test_every_tower_uses_real_site_names_on_its_own_architecture():
    record = contracts.load_artifact("media_tower.record")
    towers = record["towers"]
    # OpenCLIP's shared ResidualAttentionBlock sites, on both the vision and
    # text towers of the SAME checkpoint.
    for name in _CLIP_TOWERS:
        assert set(towers[name]["target_modules"]) <= {"in_proj", "out_proj", "c_fc", "c_proj"}
        assert towers[name]["base_model"] == f"local:{record['vision_checkpoint']}"
    # HTSAT-Swin's own sites.
    assert set(towers["audio"]["target_modules"]) <= {"query", "value", "linear1"}


def test_corrupt_row_contract_is_recorded_honestly_for_both_media_types_and_input_arms():
    """A null row AND a corrupt (undecodable) row are BOTH per-row `_status`
    outcomes, on both media types AND both input arms (`Binary` bytes and a
    `Utf8` path column — the guide's other documented input shape) — the
    batch's other, valid row still reads `_status="ok"` either way. The error
    message's OWN row index is checked against the row's actual Arrow
    position, not just a substring match.
    """
    record = contracts.load_artifact("media_tower.record")
    for modality in _MEDIA_TYPES:
        for mode in _INPUT_MODES:
            c = record["corrupt_rows"][modality][mode]
            assert c["input_mode"] == mode
            assert c["good_row_still_ok"] is True, (modality, mode, c)
            assert c["null_row"]["status"] == "error", (modality, mode, c)
            assert "null" in c["null_row"]["error"].lower(), (modality, mode, c)
            assert c["corrupt_row"]["status"] == "error", (modality, mode, c)
            assert c["corrupt_row"]["row_index_matches_arrow_position"] is True, (
                modality, mode, c["corrupt_row"],
            )
            assert c["corrupt_row"]["row_index_in_message"] == 2, (modality, mode, c["corrupt_row"])


def test_committed_artifacts_match_contract():
    art = contracts.artifact("media_tower.record")
    record = contracts.load_artifact("media_tower.record")
    assert record["rows_per_modality"] > 0
    assert set(record) >= {
        "engine_version", "vision_checkpoint", "audio_checkpoint",
        "rows_per_modality", "towers", "corrupt_rows", "provenance",
    }
    assert art.filename == "record.json"
