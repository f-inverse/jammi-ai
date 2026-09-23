#!/usr/bin/env python3
"""The `torch` rung of the `graph-sample` workload: node2vec random walks over
the same graph directory `jammi-bench graph-sample` reads, at the same walk
length, walks per node, `p` and `q`, filed as legs the ladder compares
against the engine's `sampler` rung — the warm per-iteration series, the
process's peak resident set, the walks' second-order transition counts
against the unit's law file, and a pair file.

One iteration is every walk of the graph: `walks_per_node` walks of
`walk_length` steps from each node, the unit the engine rung times.

`--walker torch_cluster` (default) calls `torch_cluster.random_walk`, the
biased node2vec walker. `--walker node2vec` drives `torch_geometric.nn.Node2Vec`'s
own walker (`torch.ops.pyg.random_walk`, from `pyg-lib`) through the module's own
CSR; that walker samples uniformly only and refuses `p != 1` or `q != 1`
("Uniform sampling required for now"), so it is the DeepWalk point of the law
and nothing else. Which one ran is provenance.

The law is the engine rung's: `<law-dir>/<unit>.json`,
`{"cells": [[probability, …], …], "observation_passes": N}`, one cell per walk
state `(previous, current)` in ascending order with the first-step states
(`previous` absent) first, the probabilities of the state's next nodes in
ascending order, and the untimed passes both rungs count their steps over.
This script recomputes the law from the graph to know that order, refuses if
its probabilities are not the file's, counts N passes' walks in it, and names
the file
by its sha256 as `law_sha256` — the ground truth is the committed file, never
a producer's claim.

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
| hard negatives | DIFFERENT — the engine mines structure-aware negatives outside the anchor's k-hop neighbourhood; `Node2Vec.neg_sample` draws uniform random nodes with no exclusion. This rung emits none; compare against an engine leg at `--hard-negatives 0` |
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

import ladder_leg as ll

RUNG = "torch"
KEY = "graph_sample"


def load_graph(graph: Path):
    nodes = ll.read_jsonl(graph / "nodes.jsonl")
    edges = ll.read_jsonl(graph / "edges.jsonl")
    ids = sorted(n["id"] for n in nodes)
    index = {node_id: i for i, node_id in enumerate(ids)}
    text = {n["id"]: n["text"] for n in nodes}
    edge_index = torch.tensor([[index[e["src"]] for e in edges], [index[e["dst"]] for e in edges]], dtype=torch.long)
    edge_set = {(e["src"], e["dst"]) for e in edges}
    symmetric = all((d, s) in edge_set for s, d in edge_set)
    return ids, text, edges, edge_index, edge_set, symmetric


def transition_law(edges: list[dict], p: float, q: float) -> list[tuple[tuple[str | None, str], list[tuple[str, float]]]]:
    """node2vec's law in the engine's cell order: states ascending with the
    first-step states first, a state's next nodes ascending."""
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    adjacent: set[tuple[str, str]] = set()
    for e in edges:
        out[e["src"]][e["dst"]] += 1.0
        adjacent.add((e["src"], e["dst"]))
        adjacent.add((e["dst"], e["src"]))

    def normalised(weights: dict[str, float]) -> list[tuple[str, float]]:
        total = sum(weights.values())
        return [(x, w / total) for x, w in sorted(weights.items())]

    law = []
    for v in sorted(out):
        law.append(((None, v), normalised(dict(out[v]))))
    for t in sorted(out):
        for v in sorted(out[t]):
            if v not in out:
                continue
            weights = {x: (1.0 / p if x == t else 1.0 if (t, x) in adjacent else 1.0 / q) * w for x, w in out[v].items()}
            law.append(((t, v), normalised(weights)))
    return law


def pair_digest(rows) -> str:
    """`graph_sample.rs`'s pair-table digest: FNV-1a over anchor, positive and
    negative texts with separator bytes."""
    mask = (1 << 64) - 1
    h = 0xCBF29CE484222325
    for anchor, positive, negatives in rows:
        for byte in anchor.encode() + b"\xff" + positive.encode() + b"\xfe" + b"".join(n.encode() + b"\xfd" for n in negatives) + b"\x00":
            h = ((h ^ byte) * 0x100000001B3) & mask
    return f"{h:016x}"


def build_walker(args, edge_index, num_nodes):
    """A function `start -> [len(start), walk_length + 1]` node-index walks."""
    if args.walker == "node2vec":
        if (args.return_p, args.in_out_q) != (1.0, 1.0):
            raise SystemExit(
                "--walker node2vec: torch_geometric.nn.Node2Vec walks through pyg-lib, which samples "
                "uniformly only; p != 1 or q != 1 needs --walker torch_cluster"
            )
        from torch_geometric.nn import Node2Vec

        model = Node2Vec(edge_index, embedding_dim=2, walk_length=args.walk_length + 1, context_size=args.walk_length + 1,
                         walks_per_node=args.walks_per_node, p=args.return_p, q=args.in_out_q, num_nodes=num_nodes)

        def walk(start):
            rw = model.random_walk_fn(model.rowptr, model.col, start, model.walk_length, model.p, model.q)
            return rw if isinstance(rw, torch.Tensor) else rw[0]

        return walk, "torch_geometric.nn.Node2Vec (torch.ops.pyg.random_walk)"

    import torch_cluster

    row, col = edge_index

    def walk(start):
        return torch_cluster.random_walk(row, col, start, args.walk_length, p=args.return_p, q=args.in_out_q, num_nodes=num_nodes)

    return walk, "torch_cluster.random_walk"


def run(args, graph: Path, take: int) -> list[str]:
    ids, text, edges, edge_index, edge_set, symmetric = load_graph(graph)
    unit = f"edges{len(edges)}"
    stem = ll.leg_stem(RUNG, unit, take)
    walker, walker_name = build_walker(args, edge_index, len(ids))
    start = torch.arange(len(ids)).repeat_interleave(args.walks_per_node)

    law_path = (args.law_dir or args.legs_dir / "law") / f"{unit}.json"
    law_file = json.loads(law_path.read_text(encoding="utf-8"))
    law_cells, observation_passes = law_file["cells"], law_file["observation_passes"]
    law = transition_law(edges, args.return_p, args.in_out_q)
    if len(law_cells) != len(law) or any(
        len(cell) != len(nexts) or any(abs(a - b) > 1e-12 for a, b in zip(cell, (p for _, p in nexts)))
        for cell, (_, nexts) in zip(law_cells, law)
    ):
        raise SystemExit(f"{law_path} is not node2vec's law of this graph at p={args.return_p}, q={args.in_out_q}")
    cell_index = {state: i for i, (state, _) in enumerate(law)}
    next_index = {state: {x: j for j, (x, _) in enumerate(nexts)} for state, nexts in law}

    iter_wall_s: list[float] = []
    for i in range(args.iterations):
        torch.manual_seed(args.seed + i)
        t0 = time.perf_counter()
        walker(start)
        iter_wall_s.append(time.perf_counter() - t0)
    peak = ll.peak_rss_bytes()

    # Untimed, after the peak is read: the passes the law file asks for, one
    # seed each.
    observed = [[0] * len(nexts) for _, nexts in law]
    for i in range(observation_passes):
        torch.manual_seed(args.seed + i)
        for nodes in walker(start).tolist():
            for j in range(len(nodes) - 1):
                cur, nxt = ids[nodes[j]], ids[nodes[j + 1]]
                if (cur, nxt) not in edge_set:
                    continue
                state = (ids[nodes[j - 1]] if j > 0 else None, cur)
                observed[cell_index[state]][next_index[state][nxt]] += 1

    torch.manual_seed(args.seed)
    first_walks = walker(start)

    def pairs():
        ordinal = 0
        for nodes in first_walks.tolist():
            anchor, seen = ids[nodes[0]], set()
            for node in nodes[1:]:
                positive = ids[node]
                if positive == anchor or positive in seen:
                    continue
                seen.add(positive)
                yield {"_ordinal": ordinal, "anchor_id": anchor, "anchor_text": text[anchor], "positive_id": positive, "positive_text": text[positive]}
                ordinal += 1

    pair_rows = list(pairs())
    pairs_file = ll.write_jsonl(args.legs_dir, f"{stem}.pairs.jsonl", pair_rows)
    block = {
        # Identity.
        "seed": args.seed,
        "graph_edges_sha256": ll.sha256_file(graph / "edges.jsonl"),
        "law_sha256": ll.sha256_file(law_path),
        "node_count": len(ids),
        "edge_count": len(edges),
        "walk_length": args.walk_length,
        "walks_per_node": args.walks_per_node,
        "return_p": args.return_p,
        "in_out_q": args.in_out_q,
        # Provenance.
        "rung": RUNG,
        "unit": unit,
        "take": take,
        "graph_nodes_sha256": ll.sha256_file(graph / "nodes.jsonl"),
        "edge_set_symmetric": symmetric,
        "hard_negatives": 0,
        "exclude_hops": None,
        "iters_measured": len(iter_wall_s),
        "walks": int(start.numel()),
        "observation_passes": observation_passes,
        "sampled_pairs": len(pair_rows),
        "pairs_file": pairs_file,
        "walker": walker_name,
        **ll.provenance(),
        # Measured.
        "iter_wall_s": iter_wall_s,
        "work": len(edges),
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "outcome_digest": pair_digest((r["anchor_text"], r["positive_text"], []) for r in pair_rows),
        "law_observed": observed,
    }
    return [ll.file_leg(args.legs_dir, KEY, stem, block, "torch_graph_sample.py")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--graph", type=Path, action="append", required=True, help="a graph directory; repeat for a size sweep (one process each)")
    parser.add_argument("--legs-dir", type=Path, required=True, help="where the legs are filed, torch__edges<N>__r<take>.json")
    parser.add_argument("--law-dir", type=Path, help="where the engine rung wrote each unit's law file; <legs-dir>/law by default")
    parser.add_argument("--walker", choices=["torch_cluster", "node2vec"], default="torch_cluster")
    parser.add_argument("--walk-length", type=int, default=4)
    parser.add_argument("--walks-per-node", type=int, default=4)
    parser.add_argument("--return-p", type=float, default=1.0)
    parser.add_argument("--in-out-q", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=32, help="passes timed and filed; the ladder settles no fewer than 32")
    parser.add_argument("--takes", type=int, default=1, help="measured repeats of each graph, each in its own process")
    parser.add_argument("--take", type=int, default=1, help="the take a single graph's run is filed as")
    args = parser.parse_args()

    points = [(graph, take) for graph in args.graph for take in range(1, args.takes + 1)]

    def argv_for(point) -> list[str]:
        graph, take = point
        argv = ["--graph", str(graph), "--legs-dir", str(args.legs_dir), "--walker", args.walker, "--take", str(take)]
        if args.law_dir:
            argv += ["--law-dir", str(args.law_dir)]
        for flag in ("walk_length", "walks_per_node", "return_p", "in_out_q", "seed", "iterations"):
            argv += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        return argv

    files = ll.legs_per_point(points, lambda point: run(args, point[0], args.take if len(points) == 1 else point[1]), argv_for)
    print(json.dumps(files))


if __name__ == "__main__":
    main()
