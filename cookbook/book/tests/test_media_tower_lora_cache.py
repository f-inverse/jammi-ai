"""Cache-backed checks on the committed real-checkpoint media-tower LoRA vertical.

These run on CPU against the committed cache (no GPU, no recompute) and assert the
vertical's load-bearing invariants for each of the three real towers (OpenCLIP
vision, OpenCLIP text, CLAP HTSAT audio): the adapter measurably changes the served
embedding, the same-input control rules out noise as the source of that change, the
adapter round-trips through a brand-new connection, and the corrupt-media-row
contract is recorded honestly (a null row is a per-row outcome; a corrupt row is a
whole-call raise, never fabricated as a third per-row outcome).

If the emitted cache is absent the heavy artifacts are skipped, but the golden
metrics, once committed, are always asserted.
"""

from __future__ import annotations

import pytest

from jammi_cookbook import contracts

_MT = contracts._dataset_dir("media_tower")
_HAVE_CACHE = (_MT / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="media_tower cache not emitted")

_TOWERS = ("vision", "text", "audio")


@_needs_cache
def test_every_tower_changed_and_is_distinguishable_from_the_control():
    record = contracts.load_artifact("media_tower.record")
    towers = record["towers"]
    assert set(towers) == set(_TOWERS), f"expected all towers, got {sorted(towers)}"
    for name, t in towers.items():
        change = t["change_vs_base_max_abs_diff"]
        control = t["same_input_control_max_abs_diff"]
        assert change > 1e-4, (name, change)
        assert control < 1e-5, (name, control)
        assert control < change, (name, control, change)
        # the recorded golden matches the committed row.
        assert contracts.golden(f"media_tower.{name}.change_vs_base_max_abs_diff").contains(change)
        assert contracts.golden(
            f"media_tower.{name}.same_input_control_max_abs_diff"
        ).contains(control)


@_needs_cache
def test_every_tower_round_trips_through_a_new_connection():
    record = contracts.load_artifact("media_tower.record")
    for name, t in record["towers"].items():
        diff = t["round_trip_max_abs_diff"]
        assert diff < 1e-5, (name, diff)
        assert contracts.golden(f"media_tower.{name}.round_trip_max_abs_diff").contains(diff)
        assert t["model_id"].startswith("jammi:fine-tuned:"), t["model_id"]


@_needs_cache
def test_every_tower_uses_real_site_names_on_its_own_architecture():
    record = contracts.load_artifact("media_tower.record")
    towers = record["towers"]
    # OpenCLIP's shared ResidualAttentionBlock sites, on both the vision and
    # text towers of the SAME checkpoint.
    for name in ("vision", "text"):
        assert set(towers[name]["target_modules"]) <= {"in_proj", "out_proj", "c_fc", "c_proj"}
        assert towers[name]["base_model"] == f"local:{record['vision_checkpoint']}"
    # HTSAT-Swin's own sites.
    assert set(towers["audio"]["target_modules"]) <= {"query", "value", "linear1"}


@_needs_cache
def test_corrupt_row_contract_is_recorded_honestly_for_both_media_types():
    """A null row AND a corrupt (undecodable) row are BOTH per-row `_status`
    outcomes, on both media types — the batch's other, valid row still reads
    `_status="ok"` either way. Measured against the shipped fix, never
    assumed from the doc table.
    """
    record = contracts.load_artifact("media_tower.record")
    for modality in ("image", "audio"):
        c = record["corrupt_rows"][modality]
        assert c["good_row_still_ok"] is True, (modality, c)
        assert c["null_row"]["status"] == "error", (modality, c)
        assert "null" in c["null_row"]["error"].lower(), (modality, c)
        assert c["corrupt_row"]["status"] == "error", (modality, c)
        assert c["corrupt_row"]["message_contains_row_index"], (modality, c)


@_needs_cache
def test_committed_artifacts_match_contract():
    art = contracts.artifact("media_tower.record")
    record = contracts.load_artifact("media_tower.record")
    assert record["rows_per_modality"] > 0
    assert set(record) >= {
        "engine_version", "vision_checkpoint", "audio_checkpoint",
        "rows_per_modality", "towers", "corrupt_rows",
    }
    assert art.filename == "record.json"
