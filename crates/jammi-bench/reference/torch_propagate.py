#!/usr/bin/env python3
"""The PyTorch rungs of the `propagate` workload: APPNP / SGC feature propagation
over the input files a `jammi-bench propagate` leg wrote (`x0.safetensors` +
`x0.keys.txt`, `edges.jsonl`), emitting the same leg fields — the warm
per-iteration series, the process's peak resident set, the digest of the
key-sorted propagated `f32` vectors, and the vectors file.

One iteration is what the engine rung times: load the inputs, build the
operator, propagate, persist the output. The fold alone is reported beside it
as `fold_iteration_s`.

`--impl exact` reproduces the engine's operator
(`crates/jammi-ai/src/pipeline/graph_propagation.rs`) with `torch.sparse`:

| aspect | status |
| --- | --- |
| adjacency | REPRODUCED — the edge list read undirected, a set of unordered pairs: a pair listed twice or in both directions is one edge, a listed self-edge is dropped |
| self-loops | REPRODUCED — `Ã = A + I`, exactly one per node |
| normalisation | REPRODUCED — `D̃^{-1/2} Ã D̃^{-1/2}` over `d̃ = deg + 1`, applied as the engine applies it: scale by `1/√d̃`, sum the neighbourhood, scale by `1/√d̃` again (`1/√d̃` is `1.0 / sqrt`, not `rsqrt`) |
| recurrence | REPRODUCED — `X⁽ᵏ⁾ = α·X⁽⁰⁾ + (1−α)·Â·X⁽ᵏ⁻¹⁾`, in that operand order; `α = 0` is SGC's `Âᴷ·X` |
| hop count | REPRODUCED — `--hops` is the depth run; the engine clamps a request to `[1, 3]` and its leg's identity carries the clamped value, which is what to pass here |
| arithmetic | REPRODUCED — `f64` throughout from the `f32` input, one final `f32` cast (`--dtype f32` runs the fold in `f32` instead and is then DIFFERENT) |
| summation order | REPRODUCED in intent — nodes are indexed in ascending key order, so a CSR row lists a node's neighbourhood in the engine's `(group, neighbour)` fold order. Whether `torch.sparse.mm` accumulates a row strictly left to right is the backend's business; the vectors are emitted so the difference is measured, not assumed |
| an edge endpoint with no vector | DIFFERENT — the engine counts it in the degree and sums nothing for it; this script refuses such an edge list |

`--impl pyg` is the practical bar, `torch_geometric.nn.APPNP` (or
`torch_geometric.nn.SGConv` with an identity linear map when `α = 0`):

| aspect | status |
| --- | --- |
| adjacency, self-loops, normalisation, recurrence, hop count | REPRODUCED — `gcn_norm(add_self_loops=True)` is the same `D̃^{-1/2}(A+I)D̃^{-1/2}`, and `APPNP` iterates the same recurrence (dropout 0) |
| arithmetic | DIFFERENT — `f32` throughout, the normalisation folded into per-edge `f32` weights and summed by a scatter |
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import ladder_leg as ll


def load_inputs(input_dir: Path):
    """Keys ascending, `X⁽⁰⁾` in that order, and the undirected edge set as index
    pairs `(i, j)`, `i < j`."""
    keys, x0 = ll.read_keyed_vectors(input_dir / "x0.safetensors")
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


def propagate_exact(x0, pairs, hops: int, alpha: float, dtype):
    n = x0.shape[0]
    rows = [i for i, j in pairs] + [j for i, j in pairs] + list(range(n))
    cols = [j for i, j in pairs] + [i for i, j in pairs] + list(range(n))
    adjacency = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.ones(len(rows), dtype=dtype), (n, n)
    ).coalesce().to_sparse_csr()
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


def run(args, input_dir: Path):
    out_dir = args.out / input_dir.parent.name / f"torch-{args.impl}"
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
        vectors = ll.write_keyed_vectors(out_dir, "propagated", keys, propagated)
        series.record(time.perf_counter() - t0)
        fold_series.record(fold_s)
    peak = ll.peak_rss_bytes()

    identity = {
        "x0_sha256": ll.artifact_of(input_dir / "x0.safetensors")["sha256"],
        "edges_sha256": ll.artifact_of(input_dir / "edges.jsonl")["sha256"],
        "nodes": len(keys),
        "edges": len(pairs),
        "dim": int(x0.shape[1]),
        "hops": args.hops,
        "alpha": args.alpha,
        "weighting": "degree_normalized",
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    operator = {
        "exact": f"torch.sparse.mm, {args.dtype} fold",
        "pyg": "torch_geometric.nn.SGConv" if args.alpha == 0.0 else "torch_geometric.nn.APPNP",
    }[args.impl]
    measured = {
        "iteration_s": series.seconds,
        "fold_iteration_s": fold_series.seconds,
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "digest": ll.keyed_vector_digest(keys, propagated),
        "vectors": vectors,
    }
    return ll.leg("propagate", identity, ll.provenance(operator=operator, device="cpu"), measured, rung=f"torch-{args.impl}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, action="append", required=True, help="a size's input directory (`<out>/n<nodes>/input`); repeat for a sweep (one process each)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--impl", choices=["exact", "pyg"], required=True)
    parser.add_argument("--dtype", choices=["f64", "f32"], default="f64", help="the exact fold's arithmetic")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    def argv_for(input_dir: Path) -> list[str]:
        argv = ["--input", str(input_dir), "--out", str(args.out), "--impl", args.impl, "--dtype", args.dtype]
        for flag in ("hops", "alpha", "warmup", "iterations"):
            argv += [f"--{flag}", str(getattr(args, flag))]
        return argv

    ll.emit("torch_propagate.py", ll.leg_per_point(args.input, lambda d: run(args, d), argv_for))


if __name__ == "__main__":
    main()
