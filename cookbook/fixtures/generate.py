#!/usr/bin/env python3
"""Generate deterministic fixtures for the Jammi AI cookbook.

Produces:
- `tiny_corpus.parquet` — 20-row synthetic patent corpus (id, title, abstract,
  year, category, assignee_id)
- `tiny_golden.json` — relevance judgments for `eval_embeddings`
- `tiny_labels.csv`  — per-row gold labels for `eval_inference` classification
- `tiny_pairs.csv`   — contrastive text pairs for `fine_tune`

Re-run with `python cookbook/fixtures/generate.py` whenever the schema or row
shape needs to change. Output is fully deterministic — same content every run.

The corpus is synthetic public-domain text (no tenant data, no scraped content)
and the row count is intentionally small so the entire cookbook fixtures tree
stays under 5 MB.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

OUT = Path(__file__).resolve().parent


CORPUS: list[tuple[int, str, str, int, str, int]] = [
    (1, "Quantum error correction in superconducting qubits",
     "We present a novel approach to quantum error correction using surface codes on superconducting qubit architectures.",
     2021, "physics", 101),
    (2, "Deep reinforcement learning for robotic manipulation",
     "This paper introduces a deep RL framework for dexterous robotic grasping in unstructured environments.",
     2022, "cs", 102),
    (3, "CRISPR-Cas9 gene editing for sickle cell disease",
     "We demonstrate therapeutic gene editing using CRISPR-Cas9 to correct the sickle cell mutation in patient-derived stem cells.",
     2020, "biology", 103),
    (4, "Topological quantum computing with Majorana fermions",
     "Topological qubits based on Majorana zero modes offer inherent protection against local noise sources.",
     2021, "physics", 101),
    (5, "Transformer architectures for protein structure prediction",
     "We apply attention-based transformer models to predict three-dimensional protein structures from amino acid sequences.",
     2022, "cs", 104),
    (6, "Catalytic conversion of CO2 to methanol",
     "A copper-zinc oxide catalyst enables efficient hydrogenation of carbon dioxide to methanol under mild conditions.",
     2019, "chemistry", 105),
    (7, "Quantum advantage in combinatorial optimization",
     "We demonstrate quantum computational advantage for solving maximum independent set problems on unit-disk graphs.",
     2023, "physics", 106),
    (8, "Federated learning with differential privacy guarantees",
     "Our federated learning protocol provides formal differential privacy guarantees while maintaining model accuracy.",
     2021, "cs", 102),
    (9, "mRNA vaccine platform for rapid pandemic response",
     "The modular mRNA vaccine platform enables design-to-clinical-trial timelines under 60 days for novel pathogens.",
     2020, "biology", 107),
    (10, "Solid-state lithium battery with ceramic electrolyte",
     "A garnet-type ceramic electrolyte enables safe all-solid-state lithium batteries with high energy density.",
     2022, "engineering", 108),
    (11, "Quantum key distribution over metropolitan fiber networks",
     "We implement continuous-variable quantum key distribution over 50km of deployed metropolitan fiber.",
     2020, "physics", 101),
    (12, "Graph neural networks for molecular property prediction",
     "Message-passing neural networks operating on molecular graphs achieve state-of-the-art prediction of quantum chemical properties.",
     2021, "cs", 104),
    (13, "Synthetic biology for sustainable biofuel production",
     "Engineered cyanobacteria produce isobutanol directly from CO2 and sunlight at industrially relevant titers.",
     2019, "biology", 103),
    (14, "Perovskite-silicon tandem solar cells",
     "Monolithic perovskite-silicon tandem cells achieve 29.8 percent power conversion efficiency.",
     2023, "engineering", 108),
    (15, "Organocatalytic asymmetric synthesis of chiral amines",
     "A new organocatalyst enables highly enantioselective reductive amination of ketones.",
     2020, "chemistry", 109),
    (16, "Quantum simulation of lattice gauge theories",
     "We simulate a U(1) lattice gauge theory on a trapped-ion quantum processor with 20 qubits.",
     2022, "physics", 106),
    (17, "Self-supervised learning for medical image analysis",
     "Contrastive self-supervised pretraining on unlabeled chest X-rays improves downstream pneumonia detection.",
     2021, "cs", 110),
    (18, "CAR-T cell therapy for solid tumors",
     "Armored CAR-T cells engineered to resist the immunosuppressive tumor microenvironment show efficacy in pancreatic cancer models.",
     2022, "biology", 107),
    (19, "Electrochemical nitrogen fixation at ambient conditions",
     "A bismuth-based catalyst reduces dinitrogen to ammonia at ambient temperature and pressure with high faradaic efficiency.",
     2021, "chemistry", 105),
    (20, "Large language model compression via structured pruning",
     "Structured pruning removes 60 percent of transformer parameters with less than 2 percent accuracy loss on language benchmarks.",
     2023, "cs", 110),
]


def write_corpus() -> None:
    """20-row patent-shaped corpus used by every embedding/search recipe."""
    table = pa.table({
        "id": pa.array([r[0] for r in CORPUS], type=pa.int64()),
        "title": pa.array([r[1] for r in CORPUS]),
        "content": pa.array([r[2] for r in CORPUS]),
        "year": pa.array([r[3] for r in CORPUS], type=pa.int64()),
        "category": pa.array([r[4] for r in CORPUS]),
        "assignee_id": pa.array([r[5] for r in CORPUS], type=pa.int64()),
    })
    pq.write_table(table, OUT / "tiny_corpus.parquet")


def write_golden() -> None:
    """Per-query relevance judgments for `eval_embeddings`.

    Shape: list of {query_id, query_text, relevant_ids}. The recipe loads
    these into a CSV-shaped `golden_relevance` source via pyarrow + sql so
    the eval runner can consume them through the same path as user goldens.
    """
    golden = [
        {"query_id": "q1", "query_text": "quantum computing applications",
         "relevant_ids": [1, 4, 7, 11]},
        {"query_id": "q2", "query_text": "machine learning for science",
         "relevant_ids": [2, 5, 12]},
        {"query_id": "q3", "query_text": "gene therapy and editing",
         "relevant_ids": [3, 9, 18]},
        {"query_id": "q4", "query_text": "renewable energy technology",
         "relevant_ids": [10, 14]},
        {"query_id": "q5", "query_text": "chemical synthesis methods",
         "relevant_ids": [6, 15, 19]},
    ]
    (OUT / "tiny_golden.json").write_text(
        json.dumps(golden, indent=2, sort_keys=True) + "\n"
    )


def write_labels() -> None:
    """Per-row classification labels for `eval_inference`.

    Two-class collapse (physics+chemistry → 'physics', biology+cs+engineering
    → 'biology') keeps the label set inside the tiny_modernbert_classifier
    fixture's vocabulary (which only knows 'physics' and 'biology').
    """
    label_map = {
        "physics": "physics",
        "chemistry": "physics",
        "biology": "biology",
        "cs": "biology",
        "engineering": "biology",
    }
    with (OUT / "tiny_labels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for row in CORPUS:
            writer.writerow([row[0], label_map[row[4]]])


def write_pairs() -> None:
    """Contrastive (text_a, text_b, score) pairs for `fine_tune`.

    Mix of high-similarity positives and low-similarity negatives — same
    shape the LoRA fine-tune runner consumes for text_embedding tasks.
    """
    positives = [
        ("quantum error correction methods", "superconducting qubit stabilization", 0.88),
        ("topological quantum computing", "Majorana fermion braiding operations", 0.91),
        ("deep reinforcement learning agents", "policy gradient optimization methods", 0.85),
        ("transformer attention mechanisms", "self-attention in neural networks", 0.92),
        ("CRISPR gene editing tools", "genome modification techniques", 0.87),
        ("mRNA vaccine design", "nucleotide sequence optimization for immunology", 0.83),
        ("lithium battery cathode materials", "solid state electrolyte interfaces", 0.79),
        ("solar cell efficiency improvements", "photovoltaic perovskite research", 0.86),
        ("catalytic CO2 reduction", "electrochemical carbon capture", 0.81),
        ("protein structure prediction", "amino acid folding simulation", 0.90),
        ("federated learning protocols", "distributed model training privacy", 0.84),
        ("graph neural network architectures", "message passing on molecular graphs", 0.88),
        ("CAR-T immunotherapy advances", "engineered T cell cancer treatment", 0.93),
        ("synthetic biology metabolic engineering", "biofuel production from microorganisms", 0.82),
        ("quantum key distribution security", "quantum cryptography fiber networks", 0.89),
    ]
    negatives = [
        ("quantum error correction methods", "medieval poetry analysis techniques", 0.08),
        ("deep reinforcement learning agents", "geological survey mapping procedures", 0.05),
        ("CRISPR gene editing tools", "Renaissance painting restoration", 0.07),
        ("lithium battery cathode materials", "ancient Roman architecture", 0.04),
        ("transformer attention mechanisms", "ocean tidal pattern forecasting", 0.11),
        ("mRNA vaccine design", "18th century naval warfare tactics", 0.03),
        ("catalytic CO2 reduction", "competitive swimming stroke analysis", 0.06),
        ("protein structure prediction", "vintage automobile restoration", 0.09),
        ("solar cell efficiency improvements", "Baroque music composition theory", 0.05),
        ("federated learning protocols", "tropical rainforest ecology surveys", 0.07),
        ("graph neural network architectures", "Italian cuisine recipe development", 0.04),
        ("CAR-T immunotherapy advances", "arctic ice sheet geological mapping", 0.06),
        ("synthetic biology metabolic engineering", "modern jazz improvisation theory", 0.08),
        ("quantum key distribution security", "Victorian era fashion history", 0.03),
        ("topological quantum computing", "agricultural crop rotation planning", 0.05),
    ]
    with (OUT / "tiny_pairs.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text_a", "text_b", "score"])
        writer.writerows(positives + negatives)


def main() -> None:
    write_corpus()
    write_golden()
    write_labels()
    write_pairs()
    print(f"Cookbook fixtures regenerated in {OUT}")


if __name__ == "__main__":
    main()
