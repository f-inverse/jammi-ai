#!/usr/bin/env python3
"""The `torch` and `torch-geometric` rungs of the `propagate` workload: APPNP /
SGC feature propagation over the unit input files a `jammi-bench propagate`
leg wrote (`<legs-dir>/input/edges<N>/x0.vectors.f32` + `x0.keys.txt` +
`edges.jsonl`), filed as legs the ladder pairs by row against the engine's
`plan` rung — the warm per-iteration series, the process's peak resident set,
the digest of the key-sorted propagated `f32` rows, and the rows themselves.

One iteration is what the engine rung times: load the inputs, build the
operator, propagate, persist the output. The fold alone is recorded beside it
as `fold_iteration_s`.

`--impl exact` is the `torch` rung and reproduces the engine's operator
(`crates/jammi-ai/src/pipeline/graph_propagation/`) with `torch.sparse`:

| aspect | status |
| --- | --- |
| adjacency | REPRODUCED — the edge list read undirected, a set of unordered pairs: a pair listed twice or in both directions is one edge, a listed self-edge is dropped |
| self-loops | REPRODUCED — `Ã = A + I`, exactly one per node |
| normalisation | REPRODUCED — `D̃^{-1/2} Ã D̃^{-1/2}` over `d̃ = deg + 1`, applied as the engine applies it: scale by `1/√d̃`, sum the neighbourhood, scale by `1/√d̃` again |
| recurrence | REPRODUCED — `X⁽ᵏ⁾ = α·X⁽⁰⁾ + (1−α)·Â·X⁽ᵏ⁻¹⁾`, in that operand order; `α = 0` is SGC's `Âᴷ·X` |
| hop count | REPRODUCED — `--hops` is the depth run; the engine clamps a request to `[1, 3]` and its leg's identity carries the clamped value, which is what to pass here |
| arithmetic | REPRODUCED — `f64` throughout from the `f32` input, one final `f32` cast (`--dtype f32` runs the fold in `f32` instead and is then DIFFERENT) |
| summation order | REPRODUCED in intent — nodes are indexed in ascending key order, so a CSR row lists a node's neighbourhood in the engine's `(group, neighbour)` fold order; whether `torch.sparse.mm` accumulates a row strictly left to right is the backend's business, and the rows are emitted so the difference is measured, not assumed |
| an edge endpoint with no vector | DIFFERENT — the engine counts it in the degree and sums nothing for it; this script refuses such an edge list |

`--impl pyg` is the `torch-geometric` rung, the practical bar:
`torch_geometric.nn.APPNP` (or `torch_geometric.nn.SGConv` with an identity
linear map when `α = 0`):

| aspect | status |
| --- | --- |
| adjacency, self-loops, normalisation, recurrence, hop count | REPRODUCED — `gcn_norm(add_self_loops=True)` is the same `D̃^{-1/2}(A+I)D̃^{-1/2}`, and `APPNP` iterates the same recurrence (dropout 0) |
| arithmetic | DIFFERENT — `f32` throughout, the normalisation folded into per-edge `f32` weights and summed by a scatter |
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ladder_leg as ll

KEY = "propagate"
RUNGS = {"exact": "torch", "pyg": "torch-geometric"}


def load_inputs(input_dir: Path):
    """Keys ascending, `X⁽⁰⁾` in that order, and the undirected edge set as index
    pairs `(i, j)`, `i < j`."""
    keys, x0 = ll.read_vector_rows(input_dir / "x0.vectors.f32", dim_of(input_dir))
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    keys = [keys[i] for i in order]
    x0 = x0[torch.tensor(order)]
    index = {key: i for i, key in enumerate(keys)}
    pairs = set()
    for edge in ll.read_jsonl(input_dir / "edges.jsonl"):
        if edge["src"] not in index or edge["dst"] not in index:
            raise SystemExit(f"edge {edge} names a node with no vector")
        i, j = index[edge["src"]], index[edge["dst"]]
        if i != j:
            pairs.add((min(i, j), max(i, j)))
    return keys, x0, sorted(pairs)


def dim_of(input_dir: Path) -> int:
    """The row width the vectors file holds, from its size and its key count."""
    keys = (input_dir / "x0.keys.txt").read_text(encoding="utf-8").splitlines()
    size = (input_dir / "x0.vectors.f32").stat().st_size
    if not keys or size % (4 * len(keys)) != 0:
        raise SystemExit(f"{input_dir}: {size} bytes is not whole rows over {len(keys)} keys")
    return size // (4 * len(keys))


def propagate_exact(x0, pairs, hops: int, alpha: float, dtype):
    n = x0.shape[0]
    rows = [i for i, j in pairs] + [j for i, j in pairs] + list(range(n))
    cols = [j for i, j in pairs] + [i for i, j in pairs] + list(range(n))
    adjacency = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.ones(len(rows), dtype=dtype), (n, n)).coalesce().to_sparse_csr()
    degree = torch.zeros(n, dtype=dtype).index_add_(0, torch.tensor(rows), torch.ones(len(rows), dtype=dtype))
    inv_sqrt = (1.0 / torch.sqrt(degree)).unsqueeze(1)

    initial = x0.to(dtype)
    t0 = time.perf_counter()
    current = initial
    for _ in range(hops):
        aggregated = torch.sparse.mm(adjacency, current * inv_sqrt) * inv_sqrt
        current = alpha * initial + (1.0 - alpha) * aggregated
    return current.to(torch.float32), time.perf_counter() - t0


def propagate_pyg(x0, pairs, hops: int, alpha: float):
    from torch_geometric.nn import APPNP, SGConv

    src = [i for i, j in pairs] + [j for i, j in pairs]
    dst = [j for i, j in pairs] + [i for i, j in pairs]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    if alpha == 0.0:
        conv = SGConv(x0.shape[1], x0.shape[1], K=hops, cached=False, bias=False)
        with torch.no_grad():
            conv.lin.weight.copy_(torch.eye(x0.shape[1]))
    else:
        conv = APPNP(K=hops, alpha=alpha, dropout=0.0, cached=False, add_self_loops=True, normalize=True)
    conv.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = conv(x0.float(), edge_index)
    return out, time.perf_counter() - t0


def run(args, unit: str, take: int) -> list[str]:
    input_dir = args.legs_dir / "input" / unit
    rung = RUNGS[args.impl]
    stem = ll.leg_stem(rung, unit, take)
    series = ll.IterationSeries(args.warmup, args.iterations)
    fold_series = ll.IterationSeries(args.warmup, args.iterations)
    dtype = torch.float64 if args.dtype == "f64" else torch.float32
    for _ in range(series.total):
        t0 = time.perf_counter()
        keys, x0, pairs = load_inputs(input_dir)
        if args.impl == "exact":
            propagated, fold_s = propagate_exact(x0, pairs, args.hops, args.alpha, dtype)
        else:
            propagated, fold_s = propagate_pyg(x0, pairs, args.hops, args.alpha)
        vectors, dim = ll.write_vector_rows(args.legs_dir, stem, keys, propagated)
        series.record(time.perf_counter() - t0)
        fold_series.record(fold_s)
    peak = ll.peak_rss_bytes()

    block = {
        # Identity.
        "graph_edges_sha256": ll.sha256_file(input_dir / "edges.jsonl"),
        "features_sha256": ll.sha256_file(input_dir / "x0.vectors.f32"),
        "node_count": len(keys),
        "edge_count": len(pairs),
        "dim": dim,
        "hops": args.hops,
        "alpha": args.alpha,
        "weighting": "degree_normalized",
        "compute_precision": "f32",
        # Provenance.
        "rung": rung,
        "unit": unit,
        "take": take,
        "warmup": args.warmup,
        "iters_measured": len(series.seconds),
        "operator": {"exact": f"torch.sparse.mm, {args.dtype} fold", "pyg": "torch_geometric.nn.SGConv" if args.alpha == 0.0 else "torch_geometric.nn.APPNP"}[args.impl],
        "fold_iteration_s": fold_series.seconds,
        **ll.provenance(),
        # Measured.
        "iter_wall_s": series.seconds,
        "work": len(pairs) * args.hops,
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "outcome_digest": ll.vector_rows_digest(keys, propagated),
        "vectors_file": f"{stem}.vectors.f32",
        "vector_dim": dim,
    }
    return [ll.file_leg(args.legs_dir, KEY, stem, block, "torch_propagate.py")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--legs-dir", type=Path, required=True, help="the engine legs' directory: inputs under input/edges<N>/, legs filed beside them")
    parser.add_argument("--unit", action="append", help="a unit `edges<N>` to propagate; every unit under input/ when omitted (one process each)")
    parser.add_argument("--impl", choices=["exact", "pyg"], required=True, help="exact: the `torch` rung; pyg: the `torch-geometric` rung")
    parser.add_argument("--dtype", choices=["f64", "f32"], default="f64", help="the exact fold's arithmetic")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--takes", type=int, default=1, help="measured repeats of each unit, each in its own process")
    parser.add_argument("--take", type=int, default=1, help="the take a single unit's run is filed as")
    args = parser.parse_args()

    units = args.unit or sorted(p.name for p in (args.legs_dir / "input").iterdir() if p.is_dir())
    points = [(unit, take) for unit in units for take in range(1, args.takes + 1)]

    def argv_for(point) -> list[str]:
        unit, take = point
        argv = ["--legs-dir", str(args.legs_dir), "--unit", unit, "--impl", args.impl, "--dtype", args.dtype, "--take", str(take)]
        for flag in ("hops", "alpha", "warmup", "iterations"):
            argv += [f"--{flag}", str(getattr(args, flag))]
        return argv

    files = ll.legs_per_point(points, lambda point: run(args, point[0], args.take if len(points) == 1 else point[1]), argv_for)
    print(json.dumps(files))


if __name__ == "__main__":
    main()
