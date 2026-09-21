#!/usr/bin/env python3
"""The PyTorch graph-learning rungs, run for real over tiny inputs.

Each reference producer under `crates/jammi-bench/reference/` claims to
reproduce an operator or a law. This suite holds each claim against an oracle
that shares no code with the producer:

* `torch_graph_sample.py` — the walks step only along edges, the transition
  counts add up to the steps walked, and the `Node2Vec` walker refuses a biased
  walk instead of silently walking uniformly;
* `torch_propagate.py` — `--impl exact` equals a dense `f64` evaluation of
  `X⁽ᵏ⁾ = α·X⁽⁰⁾ + (1−α)·D̃^{-1/2}(A+I)D̃^{-1/2}·X⁽ᵏ⁻¹⁾` written out here, a
  twice-listed edge and a listed self-edge change nothing, and `--impl pyg`
  agrees with it to `f32` rounding;
* `torch_context_predictor.py` — the first step's loss equals the closed-form
  Gaussian CRPS of the initial head evaluated here with `math.erf`, every step
  has a loss and a timing, and training moves the weights.

Every leg is also held to the leg shape: `identity`, `provenance` and
`measured` blocks, a time series of the declared length, a measured peak
resident set.

REQUIRES the graph packages in the torch venv `torch_venv.py` resolves — the
`torch-graph-venv` need of this suite's guard in `ci/guards.toml`, in the
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
# torch and safetensors, and evaluates the oracles.
FIXTURE = r"""
import json, math, sys
from pathlib import Path
import torch
sys.path.insert(0, sys.argv[1])
import ladder_leg as ll
root = Path(sys.argv[2])

# propagate inputs: 7 nodes, one edge listed twice, one in both directions, a self-edge
keys = [f"n{i}" for i in range(7)]
torch.manual_seed(3)
x0 = torch.randn(7, 5)
inp = root / "prop" / "n7" / "input"
ll.write_keyed_vectors(inp, "x0", keys, x0)
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
ll.write_keyed_vectors(root / "prop", "dense_reference", keys, x.float())

# predictor inputs: a Cnp with feature dim 3, hidden 4
from safetensors.torch import save_file
g = torch.Generator().manual_seed(5)
shapes = {"phi.fc1.weight": (4, 4), "phi.fc1.bias": (4,), "phi.fc2.weight": (4, 4), "phi.fc2.bias": (4,),
          "rho.fc1.weight": (4, 8), "rho.fc1.bias": (4,), "rho.fc2.weight": (2, 4), "rho.fc2.bias": (2,)}
save_file({n: torch.randn(s, generator=g) * 0.3 for n, s in shapes.items()}, str(root / "initial_weights.safetensors"))
episodes = {}
for split, count in (("train", 3), ("test", 2)):
    for i in range(count):
        b, k = 4, 3
        presence = torch.ones(b, k)
        presence[0, 2] = 0.0
        presence[1, :] = 0.0
        episodes.update({
            f"{split}.{i}.target_x": torch.randn(b, 3, generator=g),
            f"{split}.{i}.context_x": torch.randn(b, k, 3, generator=g),
            f"{split}.{i}.context_y": torch.randn(b, k, 1, generator=g),
            f"{split}.{i}.presence": presence,
            f"{split}.{i}.target_y": torch.randn(b, generator=g),
        })
save_file(episodes, str(root / "episodes.safetensors"))
"""

READ_VECTORS = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import ladder_leg as ll
keys, vectors = ll.read_keyed_vectors(Path(sys.argv[2]))
print(json.dumps({"keys": keys, "vectors": vectors.double().tolist(), "digest": ll.keyed_vector_digest(keys, vectors)}))
"""

FIRST_HEAD = r"""
import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
sys.path.insert(0, sys.argv[1])
import torch_context_predictor as tcp
model = tcp.Cnp(load_file(sys.argv[2]))
batch = tcp.load_episodes(Path(sys.argv[3]))["train"][0]
with torch.no_grad():
    print(json.dumps({"head": model(batch).double().tolist(), "target": batch["target_y"].double().tolist()}))
"""


def venv_python(code: str, *args: str) -> str:
    done = subprocess.run(
        [str(torch_venv.TORCH_PY), "-c", code, str(REFERENCE_DIR), *args],
        capture_output=True, text=True, timeout=600, check=False,
    )
    if done.returncode != 0:
        raise AssertionError(done.stdout + done.stderr)
    return done.stdout


def legs(script: str, *args: str) -> list[dict]:
    return json.loads(torch_venv.run(REFERENCE_DIR / script, *args, timeout=900))["legs"]


def crps(mean: float, raw: float, y: float) -> float:
    sigma = 1e-3 + math.log1p(math.exp(raw))
    z = (mean - y) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


class TorchGraphRungs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if why := torch_venv.missing(torch_venv.GRAPH_PACKAGES):
            raise AssertionError(why)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        venv_python(FIXTURE, str(cls.root))
        graph = cls.root / "graph"
        graph.mkdir()
        undirected = [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("e", "c"), ("e", "f")]
        (graph / "nodes.jsonl").write_text("".join(json.dumps({"id": n, "text": f"text {n}"}) + "\n" for n in "abcdef"))
        cls.edges = {(s, d) for a, b in undirected for s, d in ((a, b), (b, a))}
        (graph / "edges.jsonl").write_text("".join(json.dumps({"src": s, "dst": d}) + "\n" for s, d in sorted(cls.edges)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_leg_shape(self, leg: dict, workload: str, series_length: int) -> None:
        self.assertEqual(leg["workload"], workload)
        for block in ("identity", "provenance", "measured"):
            self.assertIsInstance(leg[block], dict, block)
        self.assertEqual(len(leg["measured"]["iteration_s"]), series_length)
        self.assertTrue(all(s > 0.0 for s in leg["measured"]["iteration_s"]))
        self.assertGreater(leg["measured"]["peak_rss_bytes"]["value"], 0.0)
        self.assertIn("peak_vram_bytes", leg["measured"])
        self.assertIsNotNone(leg["provenance"]["packages"]["torch"])

    def test_graph_sample_walks_the_edges_and_counts_every_step(self):
        (leg,) = legs(
            "torch_graph_sample.py", "--graph", str(self.root / "graph"), "--out", str(self.root / "gs"),
            "--walk-length", "5", "--walks-per-node", "3", "--return-p", "0.25", "--in-out-q", "4",
            "--warmup", "1", "--iterations", "4", "--transitions",
        )
        self.assert_leg_shape(leg, "graph-sample", 4)
        self.assertTrue(leg["identity"]["edge_set_symmetric"])
        self.assertEqual(leg["identity"]["hard_negatives"], 0)
        rows = [json.loads(line) for line in Path(leg["measured"]["transitions"]["path"]).read_text().splitlines()]
        for row in rows:
            self.assertIn((row["cur"], row["next"]), self.edges)
            if row["prev"] is not None:
                self.assertIn((row["prev"], row["cur"]), self.edges)
        # 6 nodes x 3 walks x 5 steps, over warm-up + measured iterations.
        self.assertEqual(sum(row["value"] for row in rows), 6 * 3 * 5 * 5)
        pairs = [json.loads(line) for line in Path(leg["measured"]["pairs"]["path"]).read_text().splitlines()]
        self.assertEqual([p["_ordinal"] for p in pairs], list(range(len(pairs))))
        self.assertTrue(all(p["anchor_id"] != p["positive_id"] for p in pairs))

    def test_node2vec_walker_refuses_a_biased_walk(self):
        done = subprocess.run(
            [str(torch_venv.TORCH_PY), str(REFERENCE_DIR / "torch_graph_sample.py"), "--graph", str(self.root / "graph"),
             "--out", str(self.root / "gs-refused"), "--walker", "node2vec", "--in-out-q", "4"],
            capture_output=True, text=True, timeout=600, check=False,
        )
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("uniformly only", done.stderr)

    def test_propagate_exact_is_the_stated_operator_and_pyg_agrees(self):
        inp = self.root / "prop" / "n7" / "input"
        common = ["--input", str(inp), "--out", str(self.root / "prop-out"), "--hops", "2", "--alpha", "0.1", "--warmup", "1", "--iterations", "2"]
        (exact,) = legs("torch_propagate.py", *common, "--impl", "exact")
        (pyg,) = legs("torch_propagate.py", *common, "--impl", "pyg")
        self.assert_leg_shape(exact, "propagate", 2)
        self.assertEqual((exact["rung"], pyg["rung"]), ("torch-exact", "torch-pyg"))
        self.assertEqual(exact["identity"], pyg["identity"])
        self.assertEqual(exact["identity"]["edges"], 6, "a repeated pair is one edge and a self-edge is none")

        reference = json.loads(venv_python(READ_VECTORS, str(self.root / "prop" / "dense_reference.safetensors")))
        for leg, tolerance in ((exact, 0.0), (pyg, 1e-5)):
            got = json.loads(venv_python(READ_VECTORS, leg["measured"]["vectors"]["path"]))
            self.assertEqual(got["keys"], reference["keys"])
            self.assertEqual(got["digest"], leg["measured"]["digest"])
            worst = max(abs(a - b) for ra, rb in zip(got["vectors"], reference["vectors"]) for a, b in zip(ra, rb))
            self.assertLessEqual(worst, tolerance, leg["rung"])

    def test_predictor_twin_scores_crps_and_trains(self):
        args = ["--episodes", str(self.root / "episodes.safetensors"), "--initial-weights", str(self.root / "initial_weights.safetensors"),
                "--out", str(self.root / "cp"), "--epochs", "4", "--learning-rate", "0.01", "--grad-clip", "1.0", "--warmup-steps", "2"]
        (leg,) = legs("torch_context_predictor.py", *args)
        self.assert_leg_shape(leg, "predictor-train-run", 4 * 3 - 2)
        losses = leg["measured"]["step_losses"]
        self.assertEqual(len(losses), 12)

        first = json.loads(venv_python(FIRST_HEAD, str(self.root / "initial_weights.safetensors"), str(self.root / "episodes.safetensors")))
        expected = sum(crps(m, r, y) for (m, r), y in zip(first["head"], first["target"])) / len(first["target"])
        self.assertAlmostEqual(losses[0], expected, places=5)
        self.assertLess(sum(losses[-3:]), sum(losses[:3]), "four epochs over three batches lower the loss")
        self.assertNotEqual(leg["measured"]["final_weights"]["sha256"], leg["identity"]["initial_weights_sha256"])
        predictions = json.loads(venv_python(READ_VECTORS, leg["measured"]["predictions"]["path"]))
        self.assertEqual(predictions["keys"], [f"test{e}_{r}" for e in range(2) for r in range(4)])
        self.assertEqual(predictions["digest"], leg["measured"]["predictions_digest"])


if __name__ == "__main__":
    unittest.main()
