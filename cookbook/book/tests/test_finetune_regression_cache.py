"""Cache-backed checks on the committed regression fine-tune vertical.

These run on CPU against the committed cache (no GPU, no recompute of the heavy
fine-tune) and assert the vertical's load-bearing invariants:

* every objective (beta_nll / gaussian_nll / crps / pinball) has a real held-out
  RMSE-in-years and a coverage in [0, 1], with the recorded golden matching the row;
* the metrics are NOT fabricated — re-folding RMSE/coverage from the committed
  held-out predictions reproduces methods.json (the read-the-cache auditability);
* the HONEST headline: the high-offset `year` target fits WITHOUT collapse — the
  Gaussian heads serve a real, non-degenerate std (orders of magnitude above the
  documented pre-0.26.2 ~0.001 variance collapse), and predicted means track the
  real year range.

No objective is privileged; the "best" loss is so only by the literal RMSE, and a
miscalibrated coverage (a short LoRA run) is reported as the measured truth, never
tuned away.
"""

from __future__ import annotations

import numpy as np
import pytest

from jammi_cookbook import contracts

_FR = contracts._dataset_dir("finetune_regression")
_HAVE_CACHE = (_FR / "golden_metrics.json").exists()
_needs_cache = pytest.mark.skipif(not _HAVE_CACHE, reason="regression cache not emitted")

_GAUSSIAN = ("beta_nll", "gaussian_nll", "crps")
_LOSSES = (*_GAUSSIAN, "pinball")
_Z_NOMINAL = 1.6448536269514722  # two-sided 90% std-normal quantile


def _rows() -> dict[str, dict]:
    return {r["loss"]: r for r in contracts.load_artifact("finetune_regression.methods")["losses"]}


@_needs_cache
def test_every_objective_has_a_real_rmse_and_coverage():
    """Each objective's held-out RMSE-in-years is real and finite; coverage in [0,1];
    the recorded golden matches the committed row."""
    rows = _rows()
    assert set(rows) == set(_LOSSES), f"expected all objectives, got {sorted(rows)}"
    for loss, r in rows.items():
        assert np.isfinite(r["rmse_years"]) and r["rmse_years"] >= 0.0
        assert 0.0 <= r["coverage_90"] <= 1.0
        assert contracts.golden(f"finetune_regression.{loss}.rmse_years").contains(
            r["rmse_years"])
        assert contracts.golden(f"finetune_regression.{loss}.coverage_90").contains(
            r["coverage_90"])


@_needs_cache
def test_metrics_refold_from_committed_predictions():
    """Anti-fabrication: re-fold RMSE/coverage from the committed held-out predictions
    and reproduce methods.json. Proves the recorded numbers come from real serve output,
    not a hand-written constant — the read-the-cache contract, end to end."""
    rows = _rows()
    for loss in _LOSSES:
        t = contracts.load_artifact(f"finetune_regression.pred_{loss}")
        true = np.asarray(t.column("true_year").to_pylist(), dtype=np.float64)
        if loss == "pinball":
            q05 = np.asarray(t.column("quantile_0.05").to_pylist(), dtype=np.float64)
            q50 = np.asarray(t.column("quantile_0.5").to_pylist(), dtype=np.float64)
            q95 = np.asarray(t.column("quantile_0.95").to_pylist(), dtype=np.float64)
            rmse = float(np.sqrt(np.mean((q50 - true) ** 2)))
            coverage = float(np.mean((true >= q05) & (true <= q95)))
        else:
            mean = np.asarray(t.column("predicted_mean").to_pylist(), dtype=np.float64)
            std = np.asarray(t.column("predicted_std").to_pylist(), dtype=np.float64)
            rmse = float(np.sqrt(np.mean((mean - true) ** 2)))
            coverage = float(np.mean(np.abs(mean - true) <= _Z_NOMINAL * std))
        # reproduce the recorded row (committed values are rounded to 4 dp).
        assert abs(rmse - rows[loss]["rmse_years"]) < 1e-3, (loss, rmse, rows[loss]["rmse_years"])
        assert abs(coverage - rows[loss]["coverage_90"]) < 1e-3, (loss, coverage)


@_needs_cache
def test_high_offset_target_fits_without_collapse():
    """The honest headline: the ~2018 high-offset target fits without variance collapse.

    The pre-0.26.2 failure was a degenerate head (std ~= 0.001, mean ~= 2163). On
    v0.29.0 every Gaussian head serves a real, non-degenerate std and predicted means
    that land inside the true year window — measured, not asserted by construction.
    """
    methods = contracts.load_artifact("finetune_regression.methods")
    assert methods["fits_without_collapse"] is True
    lo, hi = methods["target_year_lo"], methods["target_year_hi"]
    rows = _rows()
    for loss in _GAUSSIAN:
        r = rows[loss]
        # a real spread — far above the ~0.001 collapse, and not absurdly wide.
        assert r["std_mean"] > 0.1, (loss, r["std_mean"])
        # predicted means land within (a small margin of) the real year window.
        assert lo - 5 <= r["pred_mean_lo"] <= r["pred_mean_hi"] <= hi + 5, (loss, r)
    assert contracts.golden("finetune_regression.min_gaussian_std_mean").contains(
        methods["min_gaussian_std_mean"])


@_needs_cache
def test_best_loss_is_recorded_consistently():
    """`best_loss` is a real objective with the minimum held-out RMSE, and the golden
    best_rmse matches it — the chapter's measured verdict, internally consistent."""
    methods = contracts.load_artifact("finetune_regression.methods")
    rows = _rows()
    assert methods["best_loss"] in _LOSSES
    recomputed_best = min(rows.values(), key=lambda r: r["rmse_years"])
    assert methods["best_loss"] == recomputed_best["loss"]
    assert abs(methods["best_rmse_years"] - recomputed_best["rmse_years"]) < 1e-9
    assert contracts.golden("finetune_regression.best_rmse_years").contains(
        methods["best_rmse_years"])


@_needs_cache
def test_committed_prediction_artifacts_match_contract():
    """Each committed prediction parquet has the contracted columns and the held-out
    test row count."""
    methods = contracts.load_artifact("finetune_regression.methods")
    n_test = methods["n_test"]
    for loss in _LOSSES:
        name = f"finetune_regression.pred_{loss}"
        art = contracts.artifact(name)
        table = contracts.load_artifact(name)
        assert table.num_rows == n_test, (name, table.num_rows, n_test)
        for col in art.columns:
            assert col in table.column_names, f"{name} missing column {col}"
