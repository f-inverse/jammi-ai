"""Cache-backed checks on the committed real-checkpoint media-tower LoRA vertical.

These run on CPU against the committed cache (no GPU, no recompute) and assert the
vertical's load-bearing invariants for each of the three real towers (OpenCLIP
vision, OpenCLIP text, CLAP HTSAT audio): the adapter measurably changes the served
embedding past a tolerance/floor DERIVED from an observed re-emit spread and that
tower's own same-input control (never a bare literal), the adapter round-trips
through a REAL `jammi-server` process restart (not merely a new connection to the
same process), a fixed probe through the OTHER OpenCLIP tower is bit-identical to
base after tuning ONE tower (the cross-tower selectivity oracle), and the
corrupt-media-row contract is recorded — for BOTH an inline-bytes arm and a
path-valued arm (each reaching the engine through a Parquet source, which
DataFusion materialises as `BinaryView`/`Utf8View` respectively under the
engine's default `schema_force_view_types=true`) — with the error message's own
named row index checked against the row's actual Arrow position.

The committed cache (`artifacts/media_tower/`) is checked into the repo, not
gitignored, so a normal checkout always has it; these tests do not skip when it is
absent — `contracts.load_artifact`/`contracts.golden` raise loudly instead, which
is the correct outcome for a checked-out tree missing a committed artifact.
"""

from __future__ import annotations

import hashlib

import jammi

from jammi_cookbook import contracts
from jammi_cookbook.contracts import (
    MEDIA_TOWER_CORRUPT_ARROW_POSITION as CORRUPT_ARROW_POSITION,
)
from jammi_cookbook.contracts import (
    MEDIA_TOWER_ROUND_TRIP_CEILING as ROUND_TRIP_CEILING,
)

# PRECISION_FLOOR / SPREAD_K / FLOOR_K are only consumed here (never by the
# qmd, which cannot import `scripts` — quarto executes each chapter with cwd
# set to the chapter's OWN directory, not `cookbook/book`), so they stay
# single-sourced in the emit script and this test reads them from there —
# the same pattern `tests/test_tenancy_h3_cache.py` uses for
# `scripts.build_tenancy_h3_cache`; pytest itself always runs with
# `cookbook/book` as pytest's rootdir (`pythonpath = ["."]` in
# pyproject.toml), where `scripts` IS importable.
from scripts.build_media_tower_lora_cache import FLOOR_K, PRECISION_FLOOR, SPREAD_K

_TOWERS = ("vision", "text", "audio")
_CLIP_TOWERS = ("vision", "text")
_MEDIA_TYPES = ("image", "audio")
_INPUT_MODES = ("bytes", "path")
_DIR = contracts._dataset_dir("media_tower")


def test_installed_engine_identity_matches_the_recorded_provenance():
    """A stale committed cache under a newer/older engine must be detectable, not
    silently trusted. Neither `get_server_info()` nor the `jammi-ai` wheel expose a
    build sha (see the record's own `content_digest_via_shipped_surface: false`
    finding); the finest-grained identity the shipped surface exposes is the
    declared VERSION, so that is what this asserts — the same pattern
    `chapters/14-scale/scale.qmd` uses for `usearch.__version__`. FAILS (never
    skips) on a mismatch. The checkpoint digests are pinned as goldens (not merely
    length-checked): a corrupted/substituted local checkpoint directory changes
    the digest, and this test would catch it."""
    record = contracts.load_artifact("media_tower.record")
    provenance = record["provenance"]
    assert jammi.__version__ == provenance["installed_client_version"], (
        f"installed jammi {jammi.__version__} != the {provenance['installed_client_version']} "
        f"this cache was emitted against — a stale cache under a different engine build."
    )
    assert jammi.__version__ == record["engine_version"]
    assert provenance["content_digest_via_shipped_surface"] is False
    assert provenance["vision_checkpoint_sha256"] == contracts.MEDIA_TOWER_VISION_CHECKPOINT_SHA256
    assert provenance["audio_checkpoint_sha256"] == contracts.MEDIA_TOWER_AUDIO_CHECKPOINT_SHA256


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


def test_tolerance_derivation_uses_the_scripts_own_constants_and_formula():
    """Proves the derivation, not merely that SOME value sits in `precision_floor`:
    the record's `precision_floor` / `spread_multiplier_k` /
    `control_floor_multiplier_k` must equal the emit script's own module
    constants (never a value the record is free to redefine), and `change_tol` /
    `change_floor` must equal `max(k * measured_quantity, PRECISION_FLOOR)`
    recomputed from the record's OWN measured spread/control — not merely read
    back from the record's own `precision_floor` field, which would pass for any
    literal a mutated emit script wrote there."""
    record = contracts.load_artifact("media_tower.record")
    for name, t in record["towers"].items():
        td = t["tolerance_derivation"]
        assert td["precision_floor"] == PRECISION_FLOOR, (name, td, PRECISION_FLOOR)
        assert td["spread_multiplier_k"] == SPREAD_K, (name, td, SPREAD_K)
        assert td["control_floor_multiplier_k"] == FLOOR_K, (name, td, FLOOR_K)

        assert td["repeats"] == len(t["repeats"]) >= 2, (name, td, t["repeats"])
        observed_spread = max(
            abs(a["change_vs_base_max_abs_diff"] - b["change_vs_base_max_abs_diff"])
            for a in t["repeats"] for b in t["repeats"]
        )
        assert round(observed_spread, 6) == td["observed_change_spread_max_abs_diff"], (
            name, observed_spread, td,
        )
        expected_tol = max(SPREAD_K * observed_spread, PRECISION_FLOOR)
        assert round(expected_tol, 6) == td["change_tol"], (name, expected_tol, td)
        expected_floor = max(
            FLOOR_K * t["same_input_control_max_abs_diff"], PRECISION_FLOOR
        )
        assert round(expected_floor, 6) == td["change_floor"], (name, expected_floor, td)


def test_every_tower_round_trips_through_a_real_process_restart():
    """A round-trip diff is not noisy like change-vs-base: the golden alone is
    NOT sufficient here, since a broken persistence bug would happily commit
    (and then trivially match) a golden reflecting the break. This asserts a
    tight ceiling — `contracts.MEDIA_TOWER_ROUND_TRIP_CEILING`, the ONE place
    this value lives (the emit script reads the same constant), imported here
    rather than re-typed — independent of the golden. `server_restarted` is re-derived
    from the pid pair and start-time ordering, not trusted as a recorded literal:
    a script that always wrote `server_restarted: True` regardless of the actual
    pids would fail this test."""
    record = contracts.load_artifact("media_tower.record")
    for name, t in record["towers"].items():
        diff = t["round_trip_max_abs_diff"]
        assert diff < ROUND_TRIP_CEILING, (
            f"{name}: round-trip max|Δ| = {diff:.6f} is not near zero — the "
            "adapter did not survive a real process restart intact"
        )
        assert contracts.golden(f"media_tower.{name}.round_trip_max_abs_diff").contains(diff)
        assert t["model_id"].startswith("jammi:fine-tuned:"), t["model_id"]
        rt = t["round_trip"]
        pid_changed = rt["before"]["pid"] != rt["after"]["pid"]
        later_start = rt["after"]["start_time"] > rt["before"]["start_time"]
        assert pid_changed, f"{name}: round trip did not cross a real OS process boundary: {rt}"
        assert later_start, f"{name}: the 'after' process did not start later than 'before': {rt}"
        assert rt["server_restarted"] == (pid_changed and later_start), (
            f"{name}: recorded server_restarted does not match the pid/start-time evidence: {rt}"
        )


def test_cross_tower_selectivity_is_bit_identical_on_the_untouched_tower():
    """After tuning ONE OpenCLIP tower, a fixed probe through the OTHER tower
    reads BIT-IDENTICAL to base (`max_abs_diff == 0.0`, exactly). This shows the
    adapter's effect did not reach the other tower's served embedding; it does
    not, by itself, distinguish whether that is because training touched only
    sites scoped to the tuned tower or because serving applies the adapter only
    under the tuned task — either mechanism produces this same result, and this
    test asserts only the observed outcome."""
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
    outcomes, on both media types AND both input arms — an inline-bytes column
    and a path column, each registered as a Parquet source and materialised by
    DataFusion as `BinaryView`/`Utf8View` respectively under the engine's
    default `schema_force_view_types=true` — the batch's other, valid row still
    reads `_status="ok"` either way. The error message's OWN row index is
    checked against the row's actual Arrow position
    (`contracts.MEDIA_TOWER_CORRUPT_ARROW_POSITION`, never a re-typed
    literal), not just a substring match.
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
            assert c["corrupt_row"]["row_index_in_message"] == CORRUPT_ARROW_POSITION, (
                modality, mode, c["corrupt_row"],
            )


def test_checksums_cover_every_committed_file_and_match():
    checksums = contracts.load_artifact("media_tower.checksums")
    for name in ("record.json", "golden_metrics.json"):
        assert name in checksums, name
        digest = hashlib.sha256((_DIR / name).read_bytes()).hexdigest()[:16]
        assert digest == checksums[name], f"{name}: on-disk file does not match committed checksum"


def test_committed_artifacts_match_contract():
    art = contracts.artifact("media_tower.record")
    record = contracts.load_artifact("media_tower.record")
    assert record["rows_per_modality"] > 0
    assert set(record) >= {
        "engine_version", "vision_checkpoint", "audio_checkpoint",
        "rows_per_modality", "towers", "corrupt_rows", "provenance",
    }
    assert art.filename == "record.json"
