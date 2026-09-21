#!/usr/bin/env python3
"""The PyTorch rung of the `graph-sample` workload: PyTorch Geometric's node2vec
random walks over the same graph directory `jammi-bench graph-sample` reads,
at the same walk length, walks per node, `p` and `q`, emitting the same leg
fields — the warm per-iteration series, the process's peak resident set, the
walks' raw second-order transition counts, and a pair file.

One iteration is every walk of the graph: `walks_per_node` walks of
`walk_length` steps from each node, the unit the engine rung times.

`--walker torch_cluster` (default) calls `torch_cluster.random_walk`, the
biased node2vec walker. `--walker node2vec` drives `torch_geometric.nn.Node2Vec`'s
own walker (`torch.ops.pyg.random_walk`, from `pyg-lib`) through the module's own
CSR; that walker samples uniformly only and refuses `p != 1` or `q != 1`
("Uniform sampling required for now"), so it is the DeepWalk point of the law
and nothing else. Which one ran is provenance.

Against the engine's sampler (`crates/jammi-ai/src/fine_tune/graph_sampler.rs`):

| aspect | status |
| --- | --- |
| transition law `π(x|t,v) ∝ α_pq(t,x)·w(v,x)`, first step uniform | REPRODUCED by `torch_cluster` (the engine samples it by roulette over the reweighted neighbours, `torch_cluster` by rejection sampling); `Node2Vec`'s `pyg-lib` walker REPRODUCES it at `p = q = 1` only and refuses any other `p`, `q`. The transition counts are emitted so the law is tested, not assumed |
| walk length | REPRODUCED — the engine's `walk_length` counts steps; `Node2Vec(walk_length=)` counts nodes, so this script passes `walk_length + 1` |
| walks per node, `p`, `q` | REPRODUCED |
| the graph | REPRODUCED — each `edges.jsonl` row is one directed edge for both; node index order is ascending node id, the engine's read order |
| "x adjacent to t" on a directed graph | DIFFERENT — the engine reads adjacency in either direction; these walkers test the one direction stored in the CSR. Identical on a symmetric edge list, which the leg's `edge_set_symmetric` records |
| a node with no out-edge | DIFFERENT — the engine ends the walk there; these walkers stay on the node for the remaining steps. Transition counts skip a stay that is not an edge, so they are over real steps only. Cannot occur on a symmetric edge list |
| repeated edge rows | DIFFERENT for `torch_cluster` — it coalesces the edge list, so a repeated row is not a heavier edge; the engine and `Node2Vec` keep the multiplicity. Cannot occur on a simple graph |
| seeded reproducibility | DIFFERENT — the engine's walk is a pure function of its seed on any machine; `torch.manual_seed` is set and recorded but the two generators are unrelated, so rows are never equal across rungs, only their law |
| pairs | the pair file applies the engine's rule to these walks (anchor = the walk's start, one row per distinct later node that is not the anchor, in visit order). `Node2Vec.pos_sample`'s context-window pairs are DIFFERENT and are not what is emitted |
| hard negatives | DIFFERENT — the engine mines structure-aware negatives outside the anchor's k-hop neighbourhood; `Node2Vec.neg_sample` draws uniform random nodes with no exclusion. This rung emits none, and its identity says `hard_negatives: 0` |
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import torch

import ladder_leg as ll


def load_graph(graph: Path):
    nodes = ll.read_jsonl(graph / "nodes.jsonl")
    edges = ll.read_jsonl(graph / "edges.jsonl")
    ids = sorted(n["id"] for n in nodes)
    index = {node_id: i for i, node_id in enumerate(ids)}
    text = {n["id"]: n["text"] for n in nodes}
    edge_index = torch.tensor(
        [[index[e["src"]] for e in edges], [index[e["dst"]] for e in edges]], dtype=torch.long
    )
    edge_set = {(e["src"], e["dst"]) for e in edges}
    symmetric = all((d, s) in edge_set for s, d in edge_set)
    return ids, text, edge_index, edge_set, symmetric


def build_walker(args, edge_index, num_nodes):
    """A function `start -> [len(start), walk_length + 1]` node-index walks."""
    if args.walker == "node2vec":
        if (args.return_p, args.in_out_q) != (1.0, 1.0):
            raise SystemExit(
                "--walker node2vec: torch_geometric.nn.Node2Vec walks through pyg-lib, which samples "
                "uniformly only; p != 1 or q != 1 needs --walker torch_cluster"
            )
        from torch_geometric.nn import Node2Vec

        model = Node2Vec(
            edge_index,
            embedding_dim=2,
            walk_length=args.walk_length + 1,
            context_size=args.walk_length + 1,
            walks_per_node=args.walks_per_node,
            p=args.return_p,
            q=args.in_out_q,
            num_nodes=num_nodes,
        )

        def walk(start):
            rw = model.random_walk_fn(model.rowptr, model.col, start, model.walk_length, model.p, model.q)
            return rw if isinstance(rw, torch.Tensor) else rw[0]

        return walk

    import torch_cluster

    row, col = edge_index

    def walk(start):
        return torch_cluster.random_walk(
            row, col, start, args.walk_length, p=args.return_p, q=args.in_out_q, num_nodes=num_nodes
        )

    return walk


def run(args, graph: Path):
    ids, text, edge_index, edge_set, symmetric = load_graph(graph)
    out = args.out / graph.name
    walker = build_walker(args, edge_index, len(ids))
    start = torch.arange(len(ids)).repeat_interleave(args.walks_per_node)

    series = ll.IterationSeries(args.warmup, args.iterations)
    for i in range(series.total):
        torch.manual_seed(args.seed + i)
        t0 = time.perf_counter()
        walker(start)
        series.record(time.perf_counter() - t0)
    peak = ll.peak_rss_bytes()

    torch.manual_seed(args.seed)
    first_walks = walker(start)

    # Untimed, after the peak is read: one observed pass per timed seed.
    counts: Counter = Counter()
    if args.transitions:
        for i in range(series.total):
            torch.manual_seed(args.seed + i)
            for nodes in walker(start).tolist():
                for j in range(len(nodes) - 1):
                    cur, nxt = ids[nodes[j]], ids[nodes[j + 1]]
                    if (cur, nxt) not in edge_set:
                        continue
                    prev = ids[nodes[j - 1]] if j > 0 else None
                    counts[(prev, cur, nxt)] += 1

    def pairs():
        ordinal = 0
        for nodes in first_walks.tolist():
            anchor, seen = ids[nodes[0]], set()
            for node in nodes[1:]:
                positive = ids[node]
                if positive == anchor or positive in seen:
                    continue
                seen.add(positive)
                yield {
                    "_ordinal": ordinal,
                    "anchor_id": anchor,
                    "anchor_text": text[anchor],
                    "positive_id": positive,
                    "positive_text": text[positive],
                }
                ordinal += 1

    pairs_file = ll.write_jsonl(out, "pairs.jsonl", pairs())
    measured = {
        "iteration_s": series.seconds,
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "walks": int(start.numel()),
        "sampled_pairs": len(ll.read_jsonl(Path(pairs_file["path"]))),
        "pairs": pairs_file,
    }
    if args.transitions:
        rows = sorted(counts.items(), key=lambda kv: (kv[0][0] is not None, kv[0][0] or "", kv[0][1], kv[0][2]))
        measured["transitions"] = ll.write_jsonl(
            out,
            "transitions.jsonl",
            ({"prev": p, "cur": c, "next": n, "value": v} for (p, c, n), v in rows),
        )

    identity = {
        "nodes_sha256": ll.artifact_of(graph / "nodes.jsonl")["sha256"],
        "edges_sha256": ll.artifact_of(graph / "edges.jsonl")["sha256"],
        "nodes": len(ids),
        "edges": int(edge_index.shape[1]),
        "edge_set_symmetric": symmetric,
        "walk_length": args.walk_length,
        "walks_per_node": args.walks_per_node,
        "return_p": args.return_p,
        "in_out_q": args.in_out_q,
        "hard_negatives": 0,
        "exclude_hops": None,
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
    }
    walker_name = "torch_geometric.nn.Node2Vec (torch.ops.pyg.random_walk)" if args.walker == "node2vec" else "torch_cluster.random_walk"
    return ll.leg("graph-sample", identity, ll.provenance(walker=walker_name, device="cpu"), measured)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", type=Path, action="append", required=True, help="a graph directory; repeat for a size sweep (one process each)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--walker", choices=["torch_cluster", "node2vec"], default="torch_cluster")
    parser.add_argument("--walk-length", type=int, default=4)
    parser.add_argument("--walks-per-node", type=int, default=4)
    parser.add_argument("--return-p", type=float, default=1.0)
    parser.add_argument("--in-out-q", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--transitions", action="store_true", help="count the walks' second-order transitions")
    args = parser.parse_args()

    def argv_for(graph: Path) -> list[str]:
        argv = ["--graph", str(graph), "--out", str(args.out), "--walker", args.walker]
        for flag in ("walk_length", "walks_per_node", "return_p", "in_out_q", "seed", "warmup", "iterations"):
            argv += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        return argv + (["--transitions"] if args.transitions else [])

    ll.emit("torch_graph_sample.py", ll.leg_per_point(args.graph, lambda graph: run(args, graph), argv_for))


if __name__ == "__main__":
    main()
