"""Measure classification accuracy / F1 against gold labels.

Run with `python cookbook/recipes/eval_inference/example.py`. Exits 0 on
success.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
CORPUS_PATH = FIXTURES / "tiny_corpus.parquet"
LABELS_PATH = FIXTURES / "tiny_labels.csv"
MODEL = f"local:{FIXTURES / 'tiny_modernbert_classifier'}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(
            artifact_dir=tmp,
            gpu_device=-1,
            inference_batch_size=8,
        )

        # 1. Register the corpus and the gold labels.
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")
        db.add_source("golden", url=str(LABELS_PATH), format="csv")

        # 2. Run inference + eval against the gold labels.
        metrics = db.eval_inference(
            model=MODEL,
            source="corpus",
            columns=["content"],
            task="classification",
            golden_source="golden.public.tiny_labels",
            label_column="label",
        )

        # 3. Sanity-check the aggregate metrics. `f1` is macro F1 averaged
        #    across classes.
        for key in ("accuracy", "f1"):
            value = metrics[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"

        # 4. Per-class metrics live under `per_class` — dict keyed by label.
        per_class = metrics.get("per_class", {})
        assert isinstance(per_class, dict), f"per_class shape: {type(per_class)}"

        print(f"accuracy:  {metrics['accuracy']:.4f}")
        print(f"macro_f1:  {metrics['f1']:.4f}")
        print("per_class:")
        for label, stats in per_class.items():
            print(
                f"  {label:<12} precision={stats['precision']:.4f}"
                f"  recall={stats['recall']:.4f}  f1={stats['f1']:.4f}"
            )

    print("eval_inference: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
