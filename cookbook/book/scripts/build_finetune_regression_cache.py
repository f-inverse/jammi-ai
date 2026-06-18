#!/usr/bin/env python3
"""Emit the regression fine-tune cache — run ONCE on the GPU server.

The four regression objectives the engine exposes, measured side-by-side on ONE
high-offset target: predict an ogbn-arxiv paper's publication `year` (~2018, a
large-offset / low-relative-variance target) from its title+abstract via
`db.fine_tune(task="regression")`. Two honest, non-circular findings:

1. **The high-offset target fits WITHOUT collapse.** A raw mean-squared head on a
   target offset to ~2018 collapses (the documented pre-0.26.2 failure: predicted
   std ~= 0.001, mean ~= 2163 — see chapters/04-predict/predict.qmd). v0.29.0 trains
   the head against a *data-derived z-scored* target and de-standardizes at serve,
   so the served `predicted_std` is a real, non-collapsed spread and `predicted_mean`
   tracks the true year range. This chapter MEASURES that it now fits at all.
2. **Which objective wins on this target.** `beta_nll` (the default), `gaussian_nll`,
   `crps`, and `pinball` are genuinely distinct objectives; the winner (lowest
   held-out RMSE-in-years) is unknown until measured. That is the chapter's verdict.

The measurement is EXTERNAL and de-standardized: predictions are read back in raw
YEAR units (`predicted_mean`/`predicted_std`) and folded into held-out-test
RMSE-in-years, MAE, and nominal interval coverage — numbers that genuinely fail if
the z-space de-standardize, the scaler persistence, or the sigma path were broken.
"Equivalent under an affine rescale of the target" is arithmetic, not an engine
property, and is NOT the claim here.

This clones the `build_finetune_cache` pattern: connect to a running GPU
`jammi-server` over `grpc://`, reuse the committed 4000-paper ogbn-arxiv subset, hold
out a seeded representative test split (so RMSE/coverage are honest out-of-sample), run
each objective as a short LoRA fine-tune, infer the held-out test, and fold the metrics.
Emits `artifacts/finetune_regression/` (per-loss rows + golden_metrics.json +
checksums).

Determinism (K0 §3): committed subset ids, pinned ModernBERT + dtype, single-threaded
BLAS (applied by importing jammi_cookbook), `seed` passed to every fine-tune and
recorded, the train/test split a deterministic seeded shuffle, the metric folds pure
numpy, metrics asserted to tolerances downstream.

Usage::

    # 1. start the GPU server (clean artifact dir for a reproducible emit):
    #    JAMMI_ARTIFACT_DIR=/tmp/srv-ftreg-art JAMMI_GPU__DEVICE=0 \\
    #    JAMMI_GPU__REQUIRE_GPU=true JAMMI_SERVER__FLIGHT_LISTEN=127.0.0.1:50051 \\
    #    jammi-server &
    # 2. emit against it:
    python scripts/build_finetune_regression_cache.py --target grpc://127.0.0.1:50051
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jammi_ai
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import jammi_cookbook  # noqa: F401  # applies the determinism env on import
from jammi_cookbook import datasets, determinism

ENGINE_VERSION = "0.29.0"
EMBED_MODEL = "answerdotai/ModernBERT-base"
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts" / "finetune_regression"
SUBSET = 4000
TEST_FRACTION = 0.2  # held-out test split (seeded representative shuffle), honest OOS
EPOCHS = 3  # a few epochs to bound GPU time — small LoRA runs
BATCH = 32
TEXT_CLIP = 1500  # chars of title+abstract per example (bounds tokens)
NOMINAL = 0.90  # central interval coverage target
Z_NOMINAL = 1.6448536269514722  # the 95th std-normal quantile (two-sided 90%)
QUANTILE_LEVELS = [0.05, 0.5, 0.95]  # pinball head: a 90% central interval + median

# The Gaussian-head objectives serve (predicted_mean, predicted_std); pinball serves
# per-level quantile columns. `beta_nll` is the engine default (named here explicitly).
GAUSSIAN_LOSSES = ["beta_nll", "gaussian_nll", "crps"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _text(row: dict) -> str:
    return ((row["title"] or "") + ". " + (row["abstract"] or ""))[:TEXT_CLIP]


def _quantile_column(level: float) -> str:
    """The engine names a pinball head's columns `quantile_{level}` (the level with
    trailing zeros trimmed: 0.5 -> "quantile_0.5", 0.05 -> "quantile_0.05"). Python's
    `str(float)` trims the same way for these levels, matching Rust's `{}` f64 format.
    `quantile_metrics` checks presence against the served schema and raises clearly if
    the naming ever drifts, rather than failing with a silent KeyError."""
    return f"quantile_{level}"


def split_train_test(papers_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic, REPRESENTATIVE held-out split (seeded shuffle of the committed ids).

    A plain `paper_id` sort is temporal on this corpus (ids run roughly in time order),
    so a tail slice is a distribution-shifted test set — it would confound "does the
    high-offset target fit" with "does it extrapolate across a year shift". We instead
    shuffle the id-sorted papers with the recorded seed and take TEST_FRACTION, so train
    and test share the year distribution and RMSE/coverage measure fit and calibration,
    not shift. Deterministic (committed ids + seed), still genuinely out-of-sample.
    """
    ordered = sorted(papers_rows, key=lambda r: r["paper_id"])
    perm = np.random.default_rng(determinism.SEED).permutation(len(ordered))
    cut = int(round(len(ordered) * (1.0 - TEST_FRACTION)))
    train = [ordered[i] for i in perm[:cut]]
    test = [ordered[i] for i in perm[cut:]]
    return train, test


def materialize_sources(db, train_rows: list[dict], test_rows: list[dict]) -> dict[str, str]:
    """Register the train and test sources with the EXACT column contract.

    `task="regression"` detects supervision by columns literally named `text` and
    `target` (numeric); inference reads `text` keyed by `paper_id`. We materialize:
    * `reg_train` `(text, target)` — title+abstract clipped, target = `year` (int).
    * `reg_test`  `(paper_id, text)` — the held-out inputs to infer.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train_path = str(ARTIFACTS / "_train.parquet")
    test_path = str(ARTIFACTS / "_test.parquet")
    pq.write_table(
        pa.table({
            "text": [_text(r) for r in train_rows],
            "target": pa.array([int(r["year"]) for r in train_rows], type=pa.int64()),
        }),
        train_path,
    )
    pq.write_table(
        pa.table({
            "paper_id": [r["paper_id"] for r in test_rows],
            "text": [_text(r) for r in test_rows],
        }),
        test_path,
    )
    db.add_source("reg_train", url=train_path, format="parquet")
    db.add_source("reg_test", url=test_path, format="parquet")
    print(f"  sources: reg_train ({len(train_rows)}) / reg_test ({len(test_rows)})", flush=True)
    return {"train": "reg_train", "test": "reg_test"}


def _true_years(test_rows: list[dict]) -> dict[str, float]:
    return {r["paper_id"]: float(r["year"]) for r in test_rows}


def gaussian_metrics(table: pa.Table, true_years: dict[str, float]) -> dict:
    """Fold held-out RMSE-in-years / MAE / nominal coverage / std spread from a
    Gaussian head's de-standardized (predicted_mean, predicted_std) serve output."""
    ids = [str(x) for x in table.column("_row_id").to_pylist()]
    mean = np.asarray(table.column("predicted_mean").to_pylist(), dtype=np.float64)
    std = np.asarray(table.column("predicted_std").to_pylist(), dtype=np.float64)
    y = np.asarray([true_years[i] for i in ids], dtype=np.float64)
    err = mean - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    covered = np.abs(err) <= Z_NOMINAL * std
    coverage = float(np.mean(covered))
    return {
        "rmse_years": round(rmse, 4),
        "mae_years": round(mae, 4),
        "coverage_90": round(coverage, 4),
        "std_mean": round(float(np.mean(std)), 4),
        "std_min": round(float(np.min(std)), 6),
        "pred_mean_lo": round(float(np.min(mean)), 2),
        "pred_mean_hi": round(float(np.max(mean)), 2),
        "n_test": int(len(ids)),
    }


def quantile_metrics(table: pa.Table, true_years: dict[str, float]) -> dict:
    """Fold held-out metrics from a pinball head's per-level quantile columns:
    the median (q50) is the point prediction; coverage is the [q05, q95] band."""
    ids = [str(x) for x in table.column("_row_id").to_pylist()]
    cols = {lvl: _quantile_column(lvl) for lvl in QUANTILE_LEVELS}
    for lvl, name in cols.items():
        if name not in table.column_names:
            raise RuntimeError(
                f"pinball head missing quantile column {name!r} for level {lvl}; "
                f"served columns = {table.column_names}"
            )
    q = {lvl: np.asarray(table.column(name).to_pylist(), dtype=np.float64)
         for lvl, name in cols.items()}
    y = np.asarray([true_years[i] for i in ids], dtype=np.float64)
    err = q[0.5] - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    coverage = float(np.mean((y >= q[0.05]) & (y <= q[0.95])))
    return {
        "rmse_years": round(rmse, 4),
        "mae_years": round(mae, 4),
        "coverage_90": round(coverage, 4),
        "interval_width": round(float(np.mean(q[0.95] - q[0.05])), 4),
        "pred_med_lo": round(float(np.min(q[0.5])), 2),
        "pred_med_hi": round(float(np.max(q[0.5])), 2),
        "n_test": int(len(ids)),
    }


def _dump_predictions(out: pa.Table, true_years: dict[str, float], loss: str) -> str:
    """Commit the held-out test predictions so the chapter/test can RE-FOLD the
    RMSE/coverage on CPU and verify methods.json was not fabricated (the same
    read-the-cache auditability the Matryoshka emb dump gives the methods vertical)."""
    ids = [str(x) for x in out.column("_row_id").to_pylist()]
    cols: dict = {"paper_id": ids, "true_year": [true_years[i] for i in ids]}
    keep = (["predicted_mean", "predicted_std"] if loss != "pinball"
            else [_quantile_column(lvl) for lvl in QUANTILE_LEVELS])
    for c in keep:
        cols[c] = out.column(c).to_pylist()
    name = f"pred_{loss}.parquet"
    pq.write_table(pa.table(cols), ARTIFACTS / name)
    return name


def fine_tune_regression(db, *, loss: str, train_src: str, test_src: str,
                         true_years: dict[str, float], seed: int) -> dict:
    """Run one LoRA regression fine-tune, infer the held-out test, fold the metrics."""
    print(f"\n[{loss}] fine_tune task=regression", flush=True)
    kwargs: dict = {"regression_loss": loss}
    if loss == "pinball":
        kwargs["quantile_levels"] = QUANTILE_LEVELS
    job = db.fine_tune(source=train_src, base_model=EMBED_MODEL, columns=["text", "target"],
                       method="lora", task="regression", epochs=EPOCHS, batch_size=BATCH,
                       seed=seed, **kwargs)
    job.wait()
    if job.status() != "completed":
        raise RuntimeError(f"{loss} fine-tune did not complete: status={job.status()}")
    print(f"  model_id: {job.model_id}", flush=True)
    out = db.infer(source=test_src, model=job.model_id, columns=["text"],
                   task="regression", key="paper_id")
    metrics = (quantile_metrics if loss == "pinball" else gaussian_metrics)(out, true_years)
    metrics.update({"loss": loss, "model_id": job.model_id,
                    "head": "quantile" if loss == "pinball" else "gaussian",
                    "predictions": _dump_predictions(out, true_years, loss)})
    print(f"  rmse_years {metrics['rmse_years']}  mae {metrics['mae_years']}  "
          f"coverage@90 {metrics['coverage_90']}", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


def emit(db) -> None:
    info = db.get_server_info()
    print("server:", json.dumps(info), flush=True)
    if info.get("version") != ENGINE_VERSION:
        raise RuntimeError(
            f"server version {info.get('version')} != pinned {ENGINE_VERSION} — STOP")

    arxiv = datasets.load_ogbn_arxiv(db, subset=SUBSET)
    papers = arxiv.papers_source
    papers_rows = db.sql(
        f"SELECT paper_id, title, abstract, subject, year FROM {papers}.public.{papers}"
    ).to_pylist()
    years = np.asarray([int(r["year"]) for r in papers_rows], dtype=np.int64)
    print(f"papers: {len(papers_rows)}  year range [{years.min()}, {years.max()}] "
          f"mean {years.mean():.1f}", flush=True)

    train_rows, test_rows = split_train_test(papers_rows)
    src = materialize_sources(db, train_rows, test_rows)
    true_years = _true_years(test_rows)

    rows = [fine_tune_regression(db, loss=loss, train_src=src["train"], test_src=src["test"],
                                 true_years=true_years, seed=determinism.SEED)
            for loss in [*GAUSSIAN_LOSSES, "pinball"]]

    best = min(rows, key=lambda r: r["rmse_years"])
    # The no-collapse evidence: the Gaussian heads' served std is a real spread, orders
    # of magnitude above the documented pre-0.26.2 collapse (~0.001), and predicted
    # means span a meaningful slice of the true year range.
    gaussian = [r for r in rows if r["head"] == "gaussian"]
    min_std_mean = round(min(r["std_mean"] for r in gaussian), 4)
    target_std = round(float(np.std([float(r["year"]) for r in test_rows])), 4)

    methods = {
        "engine_version": ENGINE_VERSION,
        "base_model": EMBED_MODEL,
        "epochs": EPOCHS,
        "seed": determinism.SEED,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "target": "year",
        "target_year_lo": int(years.min()),
        "target_year_hi": int(years.max()),
        "target_std_years": target_std,
        "quantile_levels": QUANTILE_LEVELS,
        "losses": rows,
        "best_loss": best["loss"],
        "best_rmse_years": best["rmse_years"],
        "min_gaussian_std_mean": min_std_mean,
        "fits_without_collapse": bool(min_std_mean > 0.1),
    }
    (ARTIFACTS / "methods.json").write_text(json.dumps(methods, indent=2))

    # Golden metrics — asserted to tolerances by the chapter/test. RMSE/MAE in years
    # get an absolute tolerance; coverage an absolute tolerance; std a generous one.
    metrics: dict[str, dict[str, float]] = {
        "best_rmse_years": {"value": best["rmse_years"], "tol": 1.0},
        "min_gaussian_std_mean": {"value": min_std_mean, "tol": 0.5},
    }
    for r in rows:
        metrics[f"{r['loss']}.rmse_years"] = {"value": r["rmse_years"], "tol": 1.0}
        metrics[f"{r['loss']}.coverage_90"] = {"value": r["coverage_90"], "tol": 0.15}
    (ARTIFACTS / "golden_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True))

    # drop the intermediate supervised parquets — sources only, not committed cache.
    for tmp in ("_train.parquet", "_test.parquet"):
        (ARTIFACTS / tmp).unlink(missing_ok=True)

    _write_checksums()
    print("\n=== per-loss held-out regression (REAL) ===", flush=True)
    print(f"  target year [{years.min()},{years.max()}] std {target_std} "
          f"over {len(test_rows)} test rows", flush=True)
    for r in rows:
        print(f"  {r['loss']:<13} rmse {r['rmse_years']:.3f}y  mae {r['mae_years']:.3f}y  "
              f"cov@90 {r['coverage_90']:.3f}  "
              + (f"std_mean {r['std_mean']:.3f}" if r["head"] == "gaussian"
                 else f"width {r['interval_width']:.3f}"), flush=True)
    print(f"  best: {best['loss']} ({best['rmse_years']:.3f}y)  "
          f"fits_without_collapse={methods['fits_without_collapse']} "
          f"(min gaussian std_mean {min_std_mean})", flush=True)
    print("\nemitted cache:", flush=True)
    for f in sorted(ARTIFACTS.glob("*")):
        if f.is_file():
            print(f"  {f.name}  ({f.stat().st_size} bytes)", flush=True)


def _write_checksums() -> None:
    sums = {p.name: _checksum(p) for p in sorted(ARTIFACTS.glob("*"))
            if p.is_file() and p.name != "checksums.json"}
    (ARTIFACTS / "checksums.json").write_text(json.dumps(sums, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="grpc://127.0.0.1:50051",
                    help="connect() target — grpc://host:port for the GPU server.")
    args = ap.parse_args()
    db = jammi_ai.connect(args.target)
    emit(db)


if __name__ == "__main__":
    main()
