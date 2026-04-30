"""
Generate ~100 medical diagnosis triplets from hix_data.csv.

ICD-10 codes are grouped by their chapter (first letter) so that:
  anchor   — a natural-language query about a diagnosis
  positive — a description from the SAME ICD chapter (similar body system)
  negative — a description from a DIFFERENT ICD chapter (unrelated)

Outputs:
  hcc_data/triplets_train.jsonl  — JSONL for train_embedding_model.py
  hcc_data/triplets_train.parquet — Parquet for Jammi fine-tuning
  hcc_data/triplets_val.jsonl     — JSONL val split (10 examples)
  hcc_data/triplets_val.parquet   — Parquet val split

ICD-10 chapter mapping (first letter):
  A/B → Infectious & parasitic diseases
  C   → Neoplasms
  D   → Blood & immune disorders
  E   → Endocrine, nutritional & metabolic
  F   → Mental & behavioural
  G   → Nervous system
  H   → Eye & ear
  I   → Circulatory system
  J   → Respiratory system
  K   → Digestive system
  L   → Skin & subcutaneous
  M   → Musculoskeletal
  N   → Genitourinary
  O   → Pregnancy & childbirth
  S/T → Injury & trauma
  Z   → Factors influencing health
"""

import csv
import json
import random
import os
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("pandas not installed — will only write JSONL files")

# ── paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA_CSV = HERE / "hix_data.csv"

# ── ICD chapter grouping ────────────────────────────────────────────────────────
CHAPTER_MAP = {
    "A": "infectious",
    "B": "infectious",
    "C": "neoplasm",
    "D": "blood",
    "E": "endocrine",
    "F": "mental",
    "G": "nervous",
    "H": "sensory",
    "I": "circulatory",
    "J": "respiratory",
    "K": "digestive",
    "L": "skin",
    "M": "musculoskeletal",
    "N": "genitourinary",
    "O": "pregnancy",
    "P": "perinatal",
    "Q": "congenital",
    "R": "symptoms",
    "S": "injury",
    "T": "injury",
    "Z": "factors",
}

# ── anchor query templates ─────────────────────────────────────────────────────
ANCHOR_TEMPLATES = [
    "Patient diagnosed with {desc}",
    "Clinical presentation consistent with {desc}",
    "ICD-10 diagnosis: {desc}",
    "Medical record documents {desc}",
    "Encounter for {desc}",
    "Assessment: {desc}",
    "Chief complaint related to {desc}",
]

random.seed(42)


def chapter(code: str) -> str:
    return CHAPTER_MAP.get(code[0].upper(), "other")


def load_codes() -> dict[str, list[dict]]:
    """Load CSV and group into chapters."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(DATA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["DX_CD"].strip()
            desc = row["DX_DSC"].strip()
            if code and desc:
                chap = chapter(code)
                groups[chap].append({"code": code, "desc": desc})
    return dict(groups)


def make_triplets(groups: dict[str, list[dict]], n: int = 110) -> list[dict]:
    """
    Build n triplets.  For each triplet:
      - Sample a chapter with >= 2 members.
      - Draw anchor and positive from that chapter (different entries).
      - Draw negative from a randomly chosen different chapter.
    """
    eligible = [ch for ch, items in groups.items() if len(items) >= 2]
    chapter_list = list(groups.keys())

    triplets = []
    while len(triplets) < n:
        pos_chapter = random.choice(eligible)
        a_entry, p_entry = random.sample(groups[pos_chapter], 2)

        # Pick a negative chapter that differs from positive
        neg_chapter = random.choice(chapter_list)
        while neg_chapter == pos_chapter or not groups[neg_chapter]:
            neg_chapter = random.choice(chapter_list)

        n_entry = random.choice(groups[neg_chapter])

        template = random.choice(ANCHOR_TEMPLATES)
        anchor_text = template.format(desc=a_entry["desc"])

        triplets.append(
            {
                "anchor": anchor_text,
                "positive": p_entry["desc"],
                "negative": n_entry["desc"],
                "anchor_code": a_entry["code"],
                "positive_code": p_entry["code"],
                "negative_code": n_entry["code"],
                "chapter": pos_chapter,
            }
        )

    random.shuffle(triplets)
    return triplets


def write_jsonl(triplets: list[dict], path: Path) -> None:
    """Write in train_embedding_model.py format:
       {"anchor": {"text": "..."}, "positive": {"text": "..."}, "negative": {"text": "..."}, ...}
    """
    with open(path, "w", encoding="utf-8") as f:
        for t in triplets:
            record = {
                "anchor":   {"text": t["anchor"]},
                "positive": {"text": t["positive"]},
                "negative": {"text": t["negative"]},
                "combination_type": t["chapter"],
                "difficulty": "medium",
            }
            f.write(json.dumps(record) + "\n")
    print(f"  Wrote {len(triplets)} lines -> {path}")


def write_parquet(triplets: list[dict], path: Path) -> None:
    """Write flat Parquet with columns anchor, positive, negative for Jammi."""
    if not HAS_PANDAS:
        print("  Skipping Parquet (no pandas) — install with: pip install pandas pyarrow")
        return
    df = pd.DataFrame(
        [{"anchor": t["anchor"], "positive": t["positive"], "negative": t["negative"]}
         for t in triplets]
    )
    df.to_parquet(path, index=False)
    print(f"  Wrote {len(triplets)} rows  -> {path}")


def main() -> None:
    print("Loading hix_data.csv ...")
    groups = load_codes()
    total_codes = sum(len(v) for v in groups.values())
    print(f"  Loaded {total_codes} diagnosis codes across {len(groups)} ICD chapters")
    for ch, items in sorted(groups.items()):
        print(f"    {ch:20s} {len(items):4d} codes")

    print("\nGenerating triplets ...")
    all_triplets = make_triplets(groups, n=110)

    # 100 train / 10 val
    train_triplets = all_triplets[:100]
    val_triplets   = all_triplets[100:]

    print(f"\nTrain: {len(train_triplets)}  Val: {len(val_triplets)}")

    # Print a few examples
    print("\nSample triplets:")
    for t in train_triplets[:3]:
        print(f"  ANCHOR   : {t['anchor']}")
        print(f"  POSITIVE : {t['positive']}  ({t['positive_code']})")
        print(f"  NEGATIVE : {t['negative']}  ({t['negative_code']})")
        print(f"  CHAPTER  : {t['chapter']}")
        print()

    # Write JSONL (for train_embedding_model.py)
    print("Writing JSONL files ...")
    write_jsonl(train_triplets, HERE / "triplets_train.jsonl")
    write_jsonl(val_triplets,   HERE / "triplets_val.jsonl")

    # Write Parquet (for Jammi)
    print("Writing Parquet files ...")
    write_parquet(train_triplets, HERE / "triplets_train.parquet")
    write_parquet(val_triplets,   HERE / "triplets_val.parquet")

    print("\nDone.  Files written to:", HERE)


if __name__ == "__main__":
    main()
