#!/usr/bin/env python3
"""The PyTorch rung of the `predictor-train-run` workload: the twin of the
engine's context-predictor meta-training, started from the initial weights and
trained over the episodes a `jammi-bench predictor-train-run` leg wrote
(`initial_weights.safetensors`, `episodes.safetensors`), emitting the same leg
fields — every optimizer step's wall-clock and loss, the process's peak
resident set, the trained weights, and the head's raw output on every held-out
test target with its digest.

Both stacks start from the same tensors and see the same batches in the same
order, and neither uses dropout, so the randomness between them is removed
rather than averaged over; what remains is numerics.

Against the engine (`crates/jammi-encoders/src/context/cnp.rs`,
`crates/jammi-ai/src/pipeline/{context_predictor,parallel_train}.rs`,
`crates/jammi-ai/src/fine_tune/{regression_loss,optimizer}.rs`):

| aspect | status |
| --- | --- |
| architecture | REPRODUCED for `Cnp` — φ and ρ are two-layer MLPs with exact (erf) GELU; φ over `(context_x ‖ context_y)`, a presence-masked mean pool (an empty context pools to zero), ρ over `(pooled ‖ target_x ‖ context_size)`. `AttnCnp` and `Tnp` have no twin here: the script reads the architecture off the weight names and refuses anything else |
| initial weights | REPRODUCED — loaded from the engine's file, tensor for tensor by name |
| episodes and their order | REPRODUCED — one step per train batch, in file order, `epochs` passes, no shuffling |
| objective | REPRODUCED — closed-form Gaussian CRPS of `(mean, σ = 1e-3 + softplus(raw))`, the mean over the batch's targets |
| optimiser | REPRODUCED — AdamW, the given learning rate, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0` |
| gradient clip | REPRODUCED — global L2 norm, `coef = min(1, max_norm / (norm + 1e-6))`, applied unconditionally (`torch.nn.utils.clip_grad_norm_`, which the engine's clip mirrors) |
| step loss | REPRODUCED — the batch's loss at the parameters the step started from |
| arithmetic | `f32` on both; reduction order and kernel fusion are each backend's own — the residual the paired comparison measures |
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import ladder_leg as ll

CNP_PARAMETERS = {f"{mlp}.{layer}.{kind}" for mlp in ("phi", "rho") for layer in ("fc1", "fc2") for kind in ("weight", "bias")}


class Cnp(torch.nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.params = torch.nn.ParameterDict({name.replace(".", "_"): torch.nn.Parameter(t.clone()) for name, t in weights.items()})

    def p(self, name: str) -> torch.Tensor:
        return self.params[name.replace(".", "_")]

    def mlp(self, which: str, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(F.linear(x, self.p(f"{which}.fc1.weight"), self.p(f"{which}.fc1.bias")))
        return F.linear(h, self.p(f"{which}.fc2.weight"), self.p(f"{which}.fc2.bias"))

    def forward(self, episode: dict[str, torch.Tensor]) -> torch.Tensor:
        presence = episode["presence"]
        phi = self.mlp("phi", torch.cat([episode["context_x"], episode["context_y"]], dim=2))
        count = presence.sum(dim=1, keepdim=True)
        pooled = (phi * presence.unsqueeze(2)).sum(dim=1) / count.clamp(min=1.0)
        return self.mlp("rho", torch.cat([pooled, episode["target_x"], count], dim=1))

    def state(self) -> dict[str, torch.Tensor]:
        return {name: self.p(name).detach().clone() for name in sorted(CNP_PARAMETERS)}


def crps_gaussian(head: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean, sigma = head[:, 0], F.softplus(head[:, 1]) + 1e-3
    z = (mean - target) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / 2.0**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2.0 * torch.pi) ** 0.5
    return (sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.pi**0.5)).mean()


def load_episodes(path: Path) -> dict[str, list[dict[str, torch.Tensor]]]:
    tensors = load_file(str(path))
    splits: dict[str, dict[int, dict[str, torch.Tensor]]] = {"train": {}, "test": {}}
    for name, tensor in tensors.items():
        split, index, field = name.split(".")
        splits[split].setdefault(int(index), {})[field] = tensor
    return {split: [batches[i] for i in sorted(batches)] for split, batches in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()

    weights = load_file(str(args.initial_weights))
    if set(weights) != CNP_PARAMETERS:
        raise SystemExit(f"only the Cnp architecture has a twin; the initial weights hold {sorted(weights)}")
    episodes = load_episodes(args.episodes)
    model = Cnp(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    total_steps = args.epochs * len(episodes["train"])
    series = ll.IterationSeries(args.warmup_steps, total_steps - args.warmup_steps)
    step_losses: list[float] = []
    for _ in range(args.epochs):
        for batch in episodes["train"]:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = crps_gaussian(model(batch), batch["target_y"])
            step_losses.append(float(loss))
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            series.record(time.perf_counter() - t0)
    peak = ll.peak_rss_bytes()

    args.out.mkdir(parents=True, exist_ok=True)
    final_path = args.out / "final_weights.safetensors"
    save_file(model.state(), str(final_path))
    with torch.no_grad():
        heads = [model(batch) for batch in episodes["test"]]
    keys = [f"test{e}_{row}" for e, head in enumerate(heads) for row in range(head.shape[0])]
    predictions = torch.cat(heads, dim=0)

    hidden_dim = int(weights["phi.fc1.weight"].shape[0])
    identity = {
        "episodes_sha256": ll.artifact_of(args.episodes)["sha256"],
        "initial_weights_sha256": ll.artifact_of(args.initial_weights)["sha256"],
        "architecture": "Cnp",
        "hidden_dim": hidden_dim,
        "objective": "gaussian-crps",
        "train_episodes": len(episodes["train"]),
        "test_episodes": len(episodes["test"]),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "warmup_steps": args.warmup_steps,
    }
    last_epoch = step_losses[-len(episodes["train"]) :]
    measured = {
        "iteration_s": series.seconds,
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "step_losses": step_losses,
        "final_loss": sum(last_epoch) / len(last_epoch),
        "final_weights": ll.artifact_of(final_path),
        "predictions": ll.write_keyed_vectors(args.out, "predictions", keys, predictions),
        "predictions_digest": ll.keyed_vector_digest(keys, predictions),
    }
    ll.emit(
        "torch_context_predictor.py",
        [ll.leg("predictor-train-run", identity, ll.provenance(trainer="torch.optim.AdamW over a Cnp twin", device="cpu"), measured)],
    )


if __name__ == "__main__":
    main()
