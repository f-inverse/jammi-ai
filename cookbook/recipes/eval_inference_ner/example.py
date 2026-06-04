"""Measure NER precision/recall/F1 against gold entity spans.

Run with `python cookbook/recipes/eval_inference_ner/example.py`. Exits 0
on success.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jammi_ai

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FIXTURES = REPO_ROOT / "cookbook" / "fixtures"
CORPUS_PATH = FIXTURES / "tiny_ner_corpus.parquet"
GOLD_PATH = FIXTURES / "tiny_ner_gold.csv"
MODEL = f"local:{FIXTURES / 'tiny_modernbert_ner'}"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        db = jammi_ai.connect(f"file://{tmp}")

        # 1. Register the NER corpus and the gold entity spans.
        db.add_source("corpus", url=str(CORPUS_PATH), format="parquet")
        db.add_source("golden", url=str(GOLD_PATH), format="csv")

        # 2. Run NER inference + eval against the gold spans.
        metrics = db.eval_inference(
            model=MODEL,
            source="corpus",
            columns=["text"],
            task="ner",
            golden_source="golden.public.tiny_ner_gold",
            label_column="label",
        )

        # 3. Sanity-check the aggregate metrics. Strict entity-level
        #    matching: a prediction counts as a true positive only when
        #    its (label, start, end) triple matches a gold span exactly.
        aggregate = metrics["aggregate"]
        assert aggregate["task"] == "ner", aggregate["task"]
        for key in ("precision", "recall", "f1"):
            value = aggregate[key]
            assert 0.0 <= value <= 1.0, f"{key} out of range: {value}"

        # 4. Per-type metrics live under `aggregate.per_type` — dict keyed
        #    by entity type (e.g. "PER", "ORG").
        per_type = aggregate.get("per_type", {})
        assert isinstance(per_type, dict), f"per_type shape: {type(per_type)}"

        # 5. Per-record predictions live under `per_record`. Each entry is
        #    tagged `"task": "ner"` and carries the predicted and gold
        #    entity-span lists for one row.
        per_record = metrics["per_record"]
        assert len(per_record) > 0, "per_record must carry one entry per aligned row"
        for entry in per_record:
            assert entry["task"] == "ner", entry["task"]
            assert isinstance(entry["predicted"], list)
            assert isinstance(entry["gold"], list)

        print(f"precision: {aggregate['precision']:.4f}")
        print(f"recall:    {aggregate['recall']:.4f}")
        print(f"f1:        {aggregate['f1']:.4f}")
        print("per_type:")
        for label, stats in per_type.items():
            print(
                f"  {label:<6} precision={stats['precision']:.4f}"
                f"  recall={stats['recall']:.4f}  f1={stats['f1']:.4f}"
                f"  support={stats['support']}"
            )
        print(f"per_record: {len(per_record)} predictions")

    print("eval_inference (ner): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
