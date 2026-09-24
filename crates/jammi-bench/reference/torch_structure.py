#!/usr/bin/env python3
"""The `torch` rung of the `structure` workload: an exact evaluation of the
engine's structure encoding from the engine's own seed rows.

The engine's operator (`crates/jammi-ai/src/pipeline/graph_structure.rs`):

    X⁽⁰⁾ = the seed rows        read from the unit's `x0.vectors.f32` — the engine
                                wrote them, so both stacks start from one set of bytes
    X⁽ᵏ⁾ = P · X⁽ᵏ⁻¹⁾           P = D̃⁻¹(A + I): the lazy walk over the self-loop-
                                augmented undirected graph, `k = 1..K`, no teleport
    out  = Σₖ wₖ · X⁽ᵏ⁾/‖X⁽ᵏ⁾‖₂  `k = 0..K`, row-wise norms; a zero row stays zero,
                                a zero weight reads nothing

| claim | status |
| --- | --- |
| operator | REPRODUCED — the mean over the augmented neighbourhood, `f64` throughout, one final `f32` cast |
| readout | REPRODUCED — each read block L2-normalised then weighed, summed; no final standardisation |
| seed rows | READ, not recomputed — the projection stream is the engine's, and the row file is the identity both legs carry |
| depth | `len(weights) − 1`, as the engine's request states it |

Filed as `torch__edges<N>__r<take>.json` beside the engine legs, the key-sorted
vectors beside it, for `jammi-bench ladder structure`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ladder_leg as ll

KEY = "structure"
RUNG = "torch"


def dim_of(input_dir: Path) -> int:
    keys = (input_dir / "x0.keys.txt").read_text(encoding="utf-8").splitlines()
    size = (input_dir / "x0.vectors.f32").stat().st_size
    if not keys or size % (4 * len(keys)) != 0:
        raise SystemExit(f"{input_dir}: {size} bytes is not whole rows over {len(keys)} keys")
    return size // (4 * len(keys))


def load_inputs(input_dir: Path):
    """Keys ascending, the seed rows in that order, and the undirected edge set
    as index pairs `(i, j)`, `i < j`."""
    keys, x0 = ll.read_vector_rows(input_dir / "x0.vectors.f32", dim_of(input_dir))
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    keys = [keys[i] for i in order]
    x0 = x0[torch.tensor(order)]
    index = {key: i for i, key in enumerate(keys)}
    pairs = set()
    for edge in ll.read_jsonl(input_dir / "edges.jsonl"):
        if edge["src"] not in index or edge["dst"] not in index:
            raise SystemExit(f"edge {edge} names a node with no seed row")
        i, j = index[edge["src"]], index[edge["dst"]]
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return keys, x0, sorted(pairs)


def lazy_walk(pairs, n: int):
    """`P = D̃⁻¹(A + I)` as a sparse CSR matrix in `f64`."""
    rows = [i for i, j in pairs] + [j for i, j in pairs] + list(range(n))
    cols = [j for i, j in pairs] + [i for i, j in pairs] + list(range(n))
    ones = torch.ones(len(rows), dtype=torch.float64)
    adjacency = torch.sparse_coo_tensor(torch.tensor([rows, cols]), ones, (n, n)).coalesce()
    degree = torch.zeros(n, dtype=torch.float64).index_add_(0, torch.tensor(rows), ones)
    scaled = torch.sparse_coo_tensor(
        adjacency.indices(), adjacency.values() / degree[adjacency.indices()[0]], (n, n)
    ).coalesce()
    return scaled.to_sparse_csr()


def encode(x0, pairs, weights: list[float]):
    """The blocks and their weighted, normalised sum; the fold's own wall."""
    n = x0.shape[0]
    walk = lazy_walk(pairs, n)
    t0 = time.perf_counter()
    block = x0.to(torch.float64)
    out = torch.zeros_like(block)
    for k, weight in enumerate(weights):
        if k > 0:
            block = torch.sparse.mm(walk, block)
        if weight == 0.0:
            continue
        norm = torch.linalg.vector_norm(block, dim=1, keepdim=True)
        unit = torch.where(norm > 0.0, block / norm, torch.zeros_like(block))
        out = out + weight * unit
    return out.to(torch.float32), time.perf_counter() - t0


def run(args, unit: str, take: int) -> list[str]:
    input_dir = args.legs_dir / "input" / unit
    stem = ll.leg_stem(RUNG, unit, take)
    iter_wall_s: list[float] = []
    fold_iteration_s: list[float] = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        keys, x0, pairs = load_inputs(input_dir)
        encoded, fold_s = encode(x0, pairs, args.weights)
        vectors, dim = ll.write_vector_rows(args.legs_dir, stem, keys, encoded)
        iter_wall_s.append(time.perf_counter() - t0)
        fold_iteration_s.append(fold_s)
    peak = ll.peak_rss_bytes()
    hops = len(args.weights) - 1

    block = {
        # Identity.
        "graph_edges_sha256": ll.sha256_file(input_dir / "edges.jsonl"),
        "seed_rows_sha256": ll.sha256_file(input_dir / "x0.vectors.f32"),
        "node_count": len(keys),
        "edge_count": len(pairs),
        "dimensions": dim,
        "weights": args.weights,
        "beta": args.beta,
        "sparsity": args.sparsity,
        "seed": args.seed,
        "weighting": "uniform",
        "direction": "undirected",
        "compute_precision": "f32",
        # Provenance.
        "rung": RUNG,
        "unit": unit,
        "take": take,
        "iters_measured": len(iter_wall_s),
        "operator": "torch.sparse.mm, f64 fold, per-block L2 readout",
        "fold_iteration_s": fold_iteration_s,
        **ll.provenance(),
        # Measured.
        "iter_wall_s": iter_wall_s,
        "work": len(pairs) * hops,
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "outcome_digest": ll.vector_rows_digest(keys, encoded),
        "vectors_file": f"{stem}.vectors.f32",
        "vector_dim": dim,
    }
    return [ll.file_leg(args.legs_dir, KEY, stem, block, "torch_structure.py")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--legs-dir", type=Path, required=True, help="the engine legs' directory: inputs under input/edges<N>/, legs filed beside them")
    parser.add_argument("--unit", action="append", help="a unit `edges<N>` to encode; every unit under input/ when omitted (one process each)")
    parser.add_argument("--weights", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 0.0, 1.0, 1.0, 1.0], help="the readout weight of each block 0..=K, comma-separated: the engine leg's identity")
    parser.add_argument("--beta", type=float, default=0.0, help="the engine leg's degree exponent — identity only; the seed rows already carry it")
    parser.add_argument("--sparsity", type=float, default=3.0, help="the engine leg's projection sparsity — identity only; the seed rows already carry it")
    parser.add_argument("--seed", type=int, default=0, help="the engine leg's projection seed — identity only; the seed rows already carry it")
    parser.add_argument("--iterations", type=int, default=32, help="iterations timed and filed; the ladder settles no fewer than 32")
    ll.add_take_argument(parser)
    args = parser.parse_args()

    units = args.unit or sorted(p.name for p in (args.legs_dir / "input").iterdir() if p.is_dir())
    points = [(unit, take) for unit in units for take in args.take]

    def argv_for(point) -> list[str]:
        unit, take = point
        argv = ["--legs-dir", str(args.legs_dir), "--unit", unit, "--take", str(take), "--weights", ",".join(str(w) for w in args.weights)]
        for flag in ("beta", "sparsity", "seed", "iterations"):
            argv += [f"--{flag}", str(getattr(args, flag))]
        return argv

    files = ll.legs_per_point(points, lambda point: run(args, *point), argv_for)
    print(json.dumps(files))


if __name__ == "__main__":
    main()
