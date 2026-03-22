#!/usr/bin/env python3
"""Generate test fixtures for Jammi AI test suite."""

import pyarrow as pa
import pyarrow.parquet as pq
import csv
import os

OUT = os.path.dirname(os.path.abspath(__file__))


def patents():
    """20 patents across 5 categories with semantically meaningful text."""
    rows = [
        (1, "Quantum error correction in superconducting qubits",
         "We present a novel approach to quantum error correction using surface codes on superconducting qubit architectures.", 2021, "physics", 101),
        (2, "Deep reinforcement learning for robotic manipulation",
         "This paper introduces a deep RL framework for dexterous robotic grasping in unstructured environments.", 2022, "cs", 102),
        (3, "CRISPR-Cas9 gene editing for sickle cell disease",
         "We demonstrate therapeutic gene editing using CRISPR-Cas9 to correct the sickle cell mutation in patient-derived stem cells.", 2020, "biology", 103),
        (4, "Topological quantum computing with Majorana fermions",
         "Topological qubits based on Majorana zero modes offer inherent protection against local noise sources.", 2021, "physics", 101),
        (5, "Transformer architectures for protein structure prediction",
         "We apply attention-based transformer models to predict three-dimensional protein structures from amino acid sequences.", 2022, "cs", 104),
        (6, "Catalytic conversion of CO2 to methanol",
         "A copper-zinc oxide catalyst enables efficient hydrogenation of carbon dioxide to methanol under mild conditions.", 2019, "chemistry", 105),
        (7, "Quantum advantage in combinatorial optimization",
         "We demonstrate quantum computational advantage for solving maximum independent set problems on unit-disk graphs.", 2023, "physics", 106),
        (8, "Federated learning with differential privacy guarantees",
         "Our federated learning protocol provides formal differential privacy guarantees while maintaining model accuracy.", 2021, "cs", 102),
        (9, "mRNA vaccine platform for rapid pandemic response",
         "The modular mRNA vaccine platform enables design-to-clinical-trial timelines under 60 days for novel pathogens.", 2020, "biology", 107),
        (10, "Solid-state lithium battery with ceramic electrolyte",
         "A garnet-type ceramic electrolyte enables safe all-solid-state lithium batteries with high energy density.", 2022, "engineering", 108),
        (11, "Quantum key distribution over metropolitan fiber networks",
         "We implement continuous-variable quantum key distribution over 50km of deployed metropolitan fiber.", 2020, "physics", 101),
        (12, "Graph neural networks for molecular property prediction",
         "Message-passing neural networks operating on molecular graphs achieve state-of-the-art prediction of quantum chemical properties.", 2021, "cs", 104),
        (13, "Synthetic biology for sustainable biofuel production",
         "Engineered cyanobacteria produce isobutanol directly from CO2 and sunlight at industrially relevant titers.", 2019, "biology", 103),
        (14, "Perovskite-silicon tandem solar cells",
         "Monolithic perovskite-silicon tandem cells achieve 29.8 percent power conversion efficiency.", 2023, "engineering", 108),
        (15, "Organocatalytic asymmetric synthesis of chiral amines",
         "A new organocatalyst enables highly enantioselective reductive amination of ketones.", 2020, "chemistry", 109),
        (16, "Quantum simulation of lattice gauge theories",
         "We simulate a U(1) lattice gauge theory on a trapped-ion quantum processor with 20 qubits.", 2022, "physics", 106),
        (17, "Self-supervised learning for medical image analysis",
         "Contrastive self-supervised pretraining on unlabeled chest X-rays improves downstream pneumonia detection.", 2021, "cs", 110),
        (18, "CAR-T cell therapy for solid tumors",
         "Armored CAR-T cells engineered to resist the immunosuppressive tumor microenvironment show efficacy in pancreatic cancer models.", 2022, "biology", 107),
        (19, "Electrochemical nitrogen fixation at ambient conditions",
         "A bismuth-based catalyst reduces dinitrogen to ammonia at ambient temperature and pressure with high faradaic efficiency.", 2021, "chemistry", 105),
        (20, "Large language model compression via structured pruning",
         "Structured pruning removes 60 percent of transformer parameters with less than 2 percent accuracy loss on language benchmarks.", 2023, "cs", 110),
    ]

    table = pa.table({
        "id": pa.array([r[0] for r in rows], type=pa.int64()),
        "title": pa.array([r[1] for r in rows]),
        "abstract": pa.array([r[2] for r in rows]),
        "year": pa.array([r[3] for r in rows], type=pa.int64()),
        "category": pa.array([r[4] for r in rows]),
        "assignee_id": pa.array([r[5] for r in rows], type=pa.int64()),
    })
    pq.write_table(table, os.path.join(OUT, "patents.parquet"))


def patents_with_nulls():
    """10 rows, rows 3/7/9 have null abstracts."""
    titles = [
        "Valid patent one", "Valid patent two", "Null abstract patent",
        "Valid patent four", "Valid patent five", "Valid patent six",
        "Another null abstract", "Valid patent eight", "Third null abstract",
        "Valid patent ten",
    ]
    abstracts = [
        "Abstract for patent one.", "Abstract for patent two.", None,
        "Abstract for patent four.", "Abstract for patent five.", "Abstract for patent six.",
        None, "Abstract for patent eight.", None,
        "Abstract for patent ten.",
    ]
    table = pa.table({
        "id": pa.array(list(range(1, 11)), type=pa.int64()),
        "title": pa.array(titles),
        "abstract": pa.array(abstracts),
        "year": pa.array([2020] * 10, type=pa.int64()),
        "category": pa.array(["physics"] * 10),
        "assignee_id": pa.array([101] * 10, type=pa.int64()),
    })
    pq.write_table(table, os.path.join(OUT, "patents_with_nulls.parquet"))


def assignees():
    rows = [
        (101, "QuantumTech Inc", "US"),
        (102, "DeepMind Research", "UK"),
        (103, "GenomeWorks", "US"),
        (104, "MolecularAI Labs", "DE"),
        (105, "GreenChem Corp", "JP"),
        (106, "Quantum Innovations", "CA"),
        (107, "BioFrontier Therapeutics", "US"),
        (108, "EnergyNext", "CN"),
        (109, "ChiralSynth", "CH"),
        (110, "NeuralScale", "US"),
    ]
    with open(os.path.join(OUT, "assignees.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "company_name", "country"])
        w.writerows(rows)


def golden_relevance():
    """5 queries with known relevant patents."""
    rows = [
        ("q1", "quantum computing applications", 1),
        ("q1", "quantum computing applications", 4),
        ("q1", "quantum computing applications", 7),
        ("q1", "quantum computing applications", 11),
        ("q2", "machine learning for science", 2),
        ("q2", "machine learning for science", 5),
        ("q2", "machine learning for science", 12),
        ("q3", "gene therapy and editing", 3),
        ("q3", "gene therapy and editing", 9),
        ("q3", "gene therapy and editing", 18),
        ("q4", "renewable energy technology", 10),
        ("q4", "renewable energy technology", 14),
        ("q5", "chemical synthesis methods", 6),
        ("q5", "chemical synthesis methods", 15),
        ("q5", "chemical synthesis methods", 19),
    ]
    with open(os.path.join(OUT, "golden_relevance.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_text", "relevant_id"])
        w.writerows(rows)


def golden_labels():
    """Ground truth categories for classification eval."""
    t = pq.read_table(os.path.join(OUT, "patents.parquet"))
    rows = list(zip(t.column("id").to_pylist(), t.column("category").to_pylist()))
    with open(os.path.join(OUT, "golden_labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category"])
        w.writerows(rows)


def training_pairs():
    """30 contrastive pairs for embedding fine-tuning."""
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
    rows = positives + negatives
    with open(os.path.join(OUT, "training_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text_a", "text_b", "score"])
        w.writerows(rows)


def training_triplets():
    """15 triplets for triplet-loss fine-tuning."""
    rows = [
        ("quantum error correction", "surface code stabilizers", "medieval poetry analysis"),
        ("deep reinforcement learning", "policy gradient methods", "geological survey mapping"),
        ("CRISPR gene editing", "genome modification tools", "Renaissance painting restoration"),
        ("transformer architectures", "self-attention mechanisms", "ocean tidal forecasting"),
        ("mRNA vaccine platform", "nucleotide immunology design", "naval warfare tactics"),
        ("lithium battery research", "solid state electrolytes", "ancient Roman architecture"),
        ("catalytic CO2 conversion", "electrochemical reduction", "swimming stroke analysis"),
        ("protein folding prediction", "amino acid structure", "vintage car restoration"),
        ("solar cell technology", "perovskite photovoltaics", "Baroque music theory"),
        ("federated learning privacy", "distributed training", "rainforest ecology"),
        ("graph neural networks", "molecular message passing", "Italian cuisine"),
        ("CAR-T cell therapy", "engineered immunotherapy", "arctic geology"),
        ("synthetic biology", "metabolic engineering", "jazz improvisation"),
        ("quantum cryptography", "key distribution protocols", "Victorian fashion"),
        ("quantum simulation", "lattice gauge theory", "crop rotation"),
    ]
    with open(os.path.join(OUT, "training_triplets.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anchor", "positive", "negative"])
        w.writerows(rows)


def scores():
    with open(os.path.join(OUT, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "score"])
        w.writerows([(1, "alpha", 0.9), (2, "beta", 0.7), (3, "gamma", 0.5)])


def config():
    with open(os.path.join(OUT, "config_test.toml"), "w") as f:
        f.write("""artifact_dir = "/tmp/jammi-test-artifacts"

[engine]
execution_threads = 2
batch_size = 4096

[gpu]
device = -1

[inference]
batch_size = 8

[logging]
level = "debug"
""")


if __name__ == "__main__":
    patents()
    patents_with_nulls()
    assignees()
    golden_relevance()
    golden_labels()
    training_pairs()
    training_triplets()
    scores()
    config()
    print(f"Fixtures generated in {OUT}")
