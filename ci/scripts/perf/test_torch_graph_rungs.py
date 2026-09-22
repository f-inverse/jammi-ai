#!/usr/bin/env python3
# lane: torch-host
# needs: torch-graph-venv
"""The PyTorch graph-learning rungs, run for real over tiny inputs, filed as
the ladder's legs.

Each reference producer under `crates/jammi-bench/reference/` claims to
reproduce an operator or a law. This suite holds each claim against an oracle
that shares no code with the producer, and holds each leg to the ladder's leg
contract — `<rung>__<unit>__r<take>.json` with the block under the workload's
key, `iter_wall_s` of the declared length, a measured `peak_rss_bytes`:

* `torch_graph_sample.py` — the walks step only along edges, the counts in
  `law_observed` follow the law file's cells and add up to the steps walked,
  the law file's sha256 is the leg's `law_sha256`, and the `Node2Vec` walker
  refuses a biased walk instead of silently walking uniformly;
* `torch_propagate.py` — the `torch` rung equals a dense `f64` evaluation of
  `X⁽ᵏ⁾ = α·X⁽⁰⁾ + (1−α)·D̃^{-1/2}(A+I)D̃^{-1/2}·X⁽ᵏ⁻¹⁾` written out here, a
  twice-listed edge and a listed self-edge change nothing, and the
  `torch-geometric` rung agrees with it to `f32` rounding;
* `torch_context_predictor.py` — for each of `Cnp`, `AttnCnp` and `Tnp`, the
  first step's loss equals the closed-form Gaussian CRPS of a head computed by
  an oracle forward written here without the script's code (explicit per-head
  loops, `math.erf`), the held-out trajectory has one point per epoch anchored
  at init, training moves the weights, and a weight file with a tensor missing
  or extra is refused.

REQUIRES the graph packages in the torch venv `torch_venv.py` resolves — the
`torch-graph-venv` need of this suite (`ci/needs.toml`), in the
`torch-host` lane.

Run: `python3 ci/scripts/run_guards.py --lane torch-host`
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch_venv  # noqa: E402

REFERENCE_DIR = torch_venv.REPO_ROOT / "crates" / "jammi-bench" / "reference"

# Runs under the venv's interpreter: fabricates the tiny inputs, which need
# torch and safetensors, and the oracles.
FIXTURE = r"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[1])
import ladder_leg as ll
root = Path(sys.argv[2])

# --- graph-sample: a 6-node graph exercising all three alpha branches, and its
# law written out by an independent computation.
undirected = [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("e", "c"), ("e", "f")]
edges = sorted({(s, d) for a, b in undirected for s, d in ((a, b), (b, a))})
graph = root / "graph"; graph.mkdir()
(graph / "nodes.jsonl").write_text("".join(json.dumps({"id": n, "text": f"text {n}"}) + "\n" for n in "abcdef"))
(graph / "edges.jsonl").write_text("".join(json.dumps({"src": s, "dst": d}) + "\n" for s, d in edges))
p, q = 0.25, 4.0
nbrs = defaultdict(set)
for s, d in edges:
    nbrs[s].add(d)
def alpha(t, x):
    return 1.0 / p if x == t else (1.0 if x in nbrs[t] else 1.0 / q)
cells = []
for v in sorted(nbrs):
    cells.append([1.0 / len(nbrs[v]) for _ in sorted(nbrs[v])])
for t in sorted(nbrs):
    for v in sorted(nbrs[t]):
        w = {x: alpha(t, x) for x in sorted(nbrs[v])}
        z = sum(w.values())
        cells.append([w[x] / z for x in sorted(nbrs[v])])
law = root / "gs" / "law"; law.mkdir(parents=True)
(law / f"edges{len(edges)}.json").write_text(json.dumps({"cells": cells}))

# --- propagate: 7 nodes, one edge listed twice, one in both directions, a
# self-edge; the dense reference beside.
keys = [f"n{i}" for i in range(7)]
torch.manual_seed(3)
x0 = torch.randn(7, 5)
inp = root / "prop" / "input" / "edges6"
ll.write_vector_rows(inp, "x0", keys, x0)
pairs = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (0, 1), (5, 4), (6, 6)]
ll.write_jsonl(inp, "edges.jsonl", ({"src": keys[a], "dst": keys[b]} for a, b in pairs))
a = torch.zeros(7, 7, dtype=torch.float64)
for i, j in pairs:
    if i != j:
        a[i, j] = a[j, i] = 1.0
a += torch.eye(7, dtype=torch.float64)
d = a.sum(1)
a_hat = a / torch.sqrt(d)[:, None] / torch.sqrt(d)[None, :]
x = x0.double()
for _ in range(2):
    x = 0.1 * x0.double() + 0.9 * (a_hat @ x)
ll.write_vector_rows(root / "prop", "dense_reference", keys, x.float())

# --- predictor: feature dim 3, hidden 4, two heads, two Tnp layers.
from safetensors.torch import save_file
g = torch.Generator().manual_seed(5)
def lin(name, out_dim, in_dim, bias=True):
    # in_dim 0 is a LayerNorm's affine pair: weight and bias both [out_dim]
    d = {f"{name}.weight": (out_dim,) if in_dim == 0 else (out_dim, in_dim)}
    if bias:
        d[f"{name}.bias"] = (out_dim,)
    return d
def mlp(name, in_dim, hidden, out_dim):
    return lin(f"{name}.fc1", hidden, in_dim) | lin(f"{name}.fc2", out_dim, hidden)
shapes = {
    "Cnp": mlp("phi", 4, 4, 4) | mlp("rho", 8, 4, 2),
    "AttnCnp": lin("query", 4, 3) | lin("key", 4, 3) | lin("value", 4, 4) | {"prior_key": (1, 1, 4), "prior_value": (1, 1, 4)} | mlp("rho", 7, 4, 2),
    "Tnp": lin("context_embed", 4, 4) | lin("target_embed", 4, 3) | {"query_marker": (1, 1, 4)} | mlp("head", 4, 4, 2) | lin("final_norm", 4, 0)
    | {k: v for n in range(2) for k, v in (lin(f"layer.{n}.q", 4, 4) | lin(f"layer.{n}.k", 4, 4, bias=False) | lin(f"layer.{n}.v", 4, 4)
                                          | lin(f"layer.{n}.attn_norm", 4, 0) | lin(f"layer.{n}.mlp_norm", 4, 0) | mlp(f"layer.{n}.mlp", 4, 4, 4)).items()},
}
episodes = {"train": {}, "heldout": {}}
for split, count in (("train", 3), ("heldout", 2)):
    for i in range(count):
        b, k = 4, 3
        presence = torch.ones(b, k)
        presence[0, 2] = 0.0
        presence[1, :] = 0.0
        episodes[split].update({
            f"{i}.target_x": torch.randn(b, 3, generator=g),
            f"{i}.context_x": torch.randn(b, k, 3, generator=g),
            f"{i}.context_y": torch.randn(b, k, 1, generator=g),
            f"{i}.presence": presence,
            f"{i}.target_y": torch.randn(b, generator=g),
        })
for arch, s in shapes.items():
    w = {n: torch.randn(sh, generator=g) * 0.3 + (1.0 if len(sh) == 1 and "norm" in n and n.endswith("weight") else 0.0) for n, sh in s.items()}
    unit = root / "cp" / "input" / arch / "seed7"; unit.mkdir(parents=True)
    save_file(w, str(unit / "initial_weights.safetensors"))
    save_file(episodes["train"], str(unit / "train_episodes.safetensors"))
    save_file(episodes["heldout"], str(unit / "heldout_episodes.safetensors"))
extra = dict(w); extra["layer.1.k.bias"] = torch.zeros(4)  # w is the Tnp set, the last built
unit = root / "cp" / "input" / "Tnp" / "seed8"; unit.mkdir(parents=True)
save_file(extra, str(unit / "initial_weights.safetensors"))
save_file(episodes["train"], str(unit / "train_episodes.safetensors"))
save_file(episodes["heldout"], str(unit / "heldout_episodes.safetensors"))
"""

READ_VECTORS = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ladder_leg as ll
keys, vectors = ll.read_vector_rows(Path(sys.argv[2]), int(sys.argv[3]))
print(json.dumps({"keys": keys, "vectors": vectors.double().tolist(), "digest": ll.vector_rows_digest(keys, vectors)}))
"""

FIRST_HEAD = r"""
import json, math, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
sys.path.insert(0, sys.argv[1])
import torch_context_predictor as tcp
W = load_file(sys.argv[2]); heads = int(sys.argv[4])
batch = tcp.load_episodes(Path(sys.argv[3]))[0]

# An oracle forward written without the script's model: explicit loops over
# episodes, members and heads; exact erf-GELU; the engine's additive mask.
def gelu(x): return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
def lin(name, x): return x @ W[name + ".weight"].T + (W[name + ".bias"] if name + ".bias" in W else 0.0)
def mlp(name, x): return lin(name + ".fc2", gelu(lin(name + ".fc1", x)))
def norm(name, x):
    mu = x.mean(-1, keepdim=True); var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + 1e-5) * W[name + ".weight"] + W[name + ".bias"]
def attend(q, K, V, present):
    hidden = q.shape[0]; d = hidden // heads; out = []
    for h in range(heads):
        sl = slice(h * d, (h + 1) * d)
        scores = torch.stack([q[sl] @ K[s, sl] for s in range(K.shape[0])]) / math.sqrt(d) + (present * 10000.0 - 10000.0)
        p = torch.softmax(scores, dim=0)
        out.append(sum(p[s] * V[s, sl] for s in range(K.shape[0])))
    return torch.cat(out)

heads_out = []
for e in range(batch["target_x"].shape[0]):
    tx, cx, cy, pr = batch["target_x"][e], batch["context_x"][e], batch["context_y"][e], batch["presence"][e]
    cxy = torch.cat([cx, cy], dim=1)
    if "phi.fc1.weight" in W:
        phi = mlp("phi", cxy); n = pr.sum()
        pooled = (phi * pr[:, None]).sum(0) / max(float(n), 1.0)
        heads_out.append(mlp("rho", torch.cat([pooled, tx, n[None]])))
    elif "prior_key" in W:
        K = torch.cat([W["prior_key"].reshape(1, -1), lin("key", cx)]); V = torch.cat([W["prior_value"].reshape(1, -1), lin("value", cxy)])
        att = attend(lin("query", tx), K, V, torch.cat([torch.ones(1), pr]))
        heads_out.append(mlp("rho", torch.cat([att, tx])))
    else:
        toks = torch.cat([(lin("target_embed", tx) + W["query_marker"].reshape(-1))[None], lin("context_embed", cxy)])
        present = torch.cat([torch.ones(1), pr]); n_layers = len({k.split(".")[1] for k in W if k.startswith("layer.")})
        for l in range(n_layers):
            nt = norm(f"layer.{l}.attn_norm", toks)
            q, k, v = (lin(f"layer.{l}.{p}", nt) for p in "qkv")
            toks = toks + torch.stack([attend(q[s], k, v, present) for s in range(toks.shape[0])])
            toks = toks + mlp(f"layer.{l}.mlp", norm(f"layer.{l}.mlp_norm", toks))
        heads_out.append(mlp("head", norm("final_norm", toks[0])))
print(json.dumps({"head": torch.stack(heads_out).double().tolist(), "target": batch["target_y"].double().tolist()}))
"""


def venv_python(code: str, *args: str) -> str:
    done = subprocess.run([str(torch_venv.TORCH_PY), "-c", code, str(REFERENCE_DIR), *args], capture_output=True, text=True, timeout=600, check=False)
    if done.returncode != 0:
        raise AssertionError(done.stdout + done.stderr)
    return done.stdout


def script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(torch_venv.TORCH_PY), str(REFERENCE_DIR / name), *args], capture_output=True, text=True, timeout=900, check=False)


def legs(name: str, key: str, legs_dir: Path, *args: str) -> list[dict]:
    done = script(name, *args)
    if done.returncode != 0:
        raise AssertionError(done.stdout + done.stderr)
    return [json.loads((legs_dir / file).read_text())[key] for file in json.loads(done.stdout)]


def crps(mean: float, raw: float, y: float) -> float:
    sigma = 1e-3 + math.log1p(math.exp(raw))
    z = (mean - y) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


class TorchGraphRungs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing_for_graphs():
            raise AssertionError(why)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        venv_python(FIXTURE, str(cls.root))
        cls.edges = {tuple(json.loads(line).values()) for line in (cls.root / "graph" / "edges.jsonl").read_text().splitlines()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_leg_shape(self, leg: dict, legs_dir: Path, name: str, series_length: int) -> None:
        self.assertTrue((legs_dir / name).is_file(), name)
        self.assertEqual(len(leg["iter_wall_s"]), series_length)
        self.assertTrue(all(s > 0.0 for s in leg["iter_wall_s"]))
        self.assertGreater(leg["peak_rss_bytes"]["value"], 0.0)
        self.assertIn("peak_vram_bytes", leg)
        self.assertIsNotNone(leg["packages"]["torch"])

    def test_graph_sample_walks_the_edges_and_counts_by_the_law(self):
        legs_dir = self.root / "gs"
        (leg,) = legs(
            "torch_graph_sample.py", "graph_sample", legs_dir,
            "--graph", str(self.root / "graph"), "--legs-dir", str(legs_dir),
            "--walk-length", "5", "--walks-per-node", "3", "--return-p", "0.25", "--in-out-q", "4",
            "--warmup", "1", "--iterations", "4",
        )
        unit = f"edges{len(self.edges)}"
        self.assert_leg_shape(leg, legs_dir, f"torch__{unit}__r1.json", 4)
        self.assertEqual(leg["unit"], unit)
        self.assertTrue(leg["edge_set_symmetric"])
        self.assertEqual(leg["hard_negatives"], 0)
        self.assertEqual(leg["work"], len(self.edges))
        law_file = legs_dir / "law" / f"{unit}.json"
        import hashlib
        self.assertEqual(leg["law_sha256"], hashlib.sha256(law_file.read_bytes()).hexdigest())
        cells = json.loads(law_file.read_text())["cells"]
        observed = leg["law_observed"]
        self.assertEqual([len(c) for c in observed], [len(c) for c in cells])
        # 6 nodes x 3 walks x 5 steps, over warm-up + measured iterations.
        self.assertEqual(sum(map(sum, observed)), 6 * 3 * 5 * 5)
        pairs = [json.loads(line) for line in Path(leg["pairs_file"]["path"]).read_text().splitlines()]
        self.assertEqual([p["_ordinal"] for p in pairs], list(range(len(pairs))))
        self.assertTrue(all(p["anchor_id"] != p["positive_id"] for p in pairs))

    def test_node2vec_walker_refuses_a_biased_walk(self):
        done = script("torch_graph_sample.py", "--graph", str(self.root / "graph"), "--legs-dir", str(self.root / "gs-refused"),
                      "--law-dir", str(self.root / "gs" / "law"), "--walker", "node2vec", "--in-out-q", "4")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("uniformly only", done.stderr)

    def test_propagate_torch_rung_is_the_stated_operator_and_pyg_agrees(self):
        legs_dir = self.root / "prop"
        common = ["--legs-dir", str(legs_dir), "--unit", "edges6", "--hops", "2", "--alpha", "0.1", "--warmup", "1", "--iterations", "2"]
        (exact,) = legs("torch_propagate.py", "propagate", legs_dir, *common, "--impl", "exact")
        (pyg,) = legs("torch_propagate.py", "propagate", legs_dir, *common, "--impl", "pyg")
        self.assert_leg_shape(exact, legs_dir, "torch__edges6__r1.json", 2)
        self.assert_leg_shape(pyg, legs_dir, "torch-geometric__edges6__r1.json", 2)
        identity = ("graph_edges_sha256", "features_sha256", "node_count", "edge_count", "dim", "hops", "alpha", "weighting", "compute_precision")
        self.assertEqual({k: exact[k] for k in identity}, {k: pyg[k] for k in identity})
        self.assertEqual(exact["edge_count"], 6, "a repeated pair is one edge and a self-edge is none")
        self.assertEqual(exact["work"], 12)

        reference = json.loads(venv_python(READ_VECTORS, str(legs_dir / "dense_reference.vectors.f32"), "5"))
        for leg, tolerance in ((exact, 0.0), (pyg, 1e-5)):
            got = json.loads(venv_python(READ_VECTORS, str(legs_dir / leg["vectors_file"]), str(leg["vector_dim"])))
            self.assertEqual(got["keys"], reference["keys"])
            self.assertEqual(got["digest"], leg["outcome_digest"])
            worst = max(abs(a - b) for ra, rb in zip(got["vectors"], reference["vectors"]) for a, b in zip(ra, rb))
            self.assertLessEqual(worst, tolerance, leg["rung"])

    def test_predictor_twins_score_crps_and_train(self):
        legs_dir = self.root / "cp"
        for arch in ("Cnp", "AttnCnp", "Tnp"):
            with self.subTest(arch=arch):
                (leg,) = legs(
                    "torch_context_predictor.py", "predictor_train_run", legs_dir,
                    "--legs-dir", str(legs_dir), "--arch", arch, "--seeds", "7", "--epochs", "4", "--learning-rate", "0.01",
                    "--grad-clip", "1.0", "--num-heads", "2", "--num-layers", "2", "--warmup-steps", "2",
                )
                self.assert_leg_shape(leg, legs_dir, "torch__seed7__r1.json", 4 * 3 - 2)
                self.assertEqual(leg["architecture"], arch)
                self.assertEqual((leg["context_k"], leg["feature_dim"], leg["value_dim"], leg["head_width"], leg["batch"]), (3, 3, 1, 2, 4))
                self.assertEqual(leg["schedule"], "constant")
                losses = leg["step_losses"]
                self.assertEqual(len(losses), 12)
                self.assertEqual([p["epoch"] for p in leg["trajectory"]], [1, 2, 3, 4])
                self.assertEqual(leg["held_out_example_mean"], leg["trajectory"][-1]["held_out_mean"])
                self.assertEqual(len(leg["train_probe_series"]), 5)
                self.assertIsNotNone(leg["held_out_at_init"])

                unit = legs_dir / "input" / arch / "seed7"
                first = json.loads(venv_python(FIRST_HEAD, str(unit / "initial_weights.safetensors"), str(unit / "train_episodes.safetensors"), "2"))
                expected = sum(crps(m, r, y) for (m, r), y in zip(first["head"], first["target"])) / len(first["target"])
                self.assertAlmostEqual(losses[0], expected, places=5)
                self.assertLess(sum(losses[-3:]), sum(losses[:3]), "four epochs over three batches lower the loss")
                self.assertNotEqual(leg["final_weights"]["sha256"], leg["initial_weights_sha256"])
                predictions = json.loads(venv_python(READ_VECTORS, str(legs_dir / leg["vectors_file"]), str(leg["vector_dim"])))
                self.assertEqual(predictions["keys"], [f"test{e}_{r}" for e in range(2) for r in range(4)])
                self.assertEqual(predictions["digest"], leg["outcome_digest"])

    def test_predictor_twin_refuses_a_foreign_tensor_set(self):
        done = script("torch_context_predictor.py", "--legs-dir", str(self.root / "cp"), "--arch", "Tnp", "--seeds", "8",
                      "--epochs", "1", "--learning-rate", "0.01", "--grad-clip", "1.0", "--num-heads", "2", "--num-layers", "2")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no member's parameter set", done.stderr)

if __name__ == "__main__":
    unittest.main()
