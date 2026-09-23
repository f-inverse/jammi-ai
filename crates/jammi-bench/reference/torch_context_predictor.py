#!/usr/bin/env python3
"""The `torch` rung of the `predictor-train-run` workload: exact twins of the
engine's three in-context predictors — `Cnp`, `AttnCnp`, `Tnp` — started from
the initial weights and trained over the episodes a `jammi-bench
predictor-train-run --arch <member>` leg wrote under
`<legs-dir>/input/<arch>/seed<N>/` (`initial_weights.safetensors`,
`train_episodes.safetensors`, `heldout_episodes.safetensors`), filed as legs
the ladder pairs by seed against the engine's `in-process` rung: every
optimizer step's wall-clock, the process's peak resident set, the held-out loss
at init and after every epoch, the train-side probe series, and the head's raw
output on every held-out target with its digest.

Both stacks start from the same tensors and see the same batches in the same
order, and neither uses dropout, so the randomness between them is removed
rather than averaged over; what remains is numerics.

The member is read off the initial weights' tensor names, which are the
engine's own. A name set that is not exactly one member's — a tensor missing,
a tensor extra, a layer index skipped — is refused. The identity fields the
files do not determine — the context width, the heads and layers the
configuration names, the epochs, the learning rate, the clip — are flags, the
engine leg's identity carrying their values.

Against the engine (`crates/jammi-encoders/src/context/{cnp,attncnp,tnp,
attention,mod}.rs`, `crates/jammi-encoders/src/{layer_norm,mask}.rs`,
`crates/jammi-ai/src/pipeline/{context_predictor,parallel_train}.rs`,
`crates/jammi-ai/src/fine_tune/{regression_loss,optimizer}.rs`), every
operation in the engine's order:

| behaviour | status |
| --- | --- |
| MLP (φ, ρ, a Tnp block's MLP, the Tnp head) | REPRODUCED — `fc2(gelu(fc1(x)))`, both linears biased, exact erf GELU (not the tanh approximation) |
| `Cnp` | REPRODUCED — φ over `(context_x ‖ context_y)`; mean over present members (an empty context pools to the zero row); ρ over `(pooled ‖ target_x ‖ context_size)` |
| `AttnCnp` projections | REPRODUCED — biased linears `query` on `target_x`, `key` on `context_x`, `value` on `(context_x ‖ context_y)`, all to `hidden_dim` |
| `AttnCnp` prior token | REPRODUCED — the learned `prior_key` / `prior_value` (`[1, 1, hidden]`) prepended as member 0, always present; an empty context attends the prior alone |
| `AttnCnp` decoder | REPRODUCED — ρ over `(attended ‖ target_x)`; no context size, no output projection after the attention |
| `Tnp` tokens | REPRODUCED — target token `target_embed(target_x) + query_marker` at position 0, then `context_embed(context_x ‖ context_y)`; no positional encoding |
| `Tnp` block | REPRODUCED — pre-norm: `attn_norm` (biased LayerNorm, eps `1e-5`, biased variance over the hidden axis) feeds biased `q`/`v` and bias-free `k`, attention, `tokens + attended`; then `mlp_norm` feeds the MLP, `tokens + mlp`; no output projection; padded tokens are still queries (and are updated) but never keys |
| `Tnp` read-out | REPRODUCED — `final_norm` on position 0 after the last block, then the `head` MLP |
| attention | REPRODUCED — split `hidden` into `num_heads` contiguous slices of `hidden / num_heads`; `QKᵀ`, divided by `√head_dim`, plus the additive key mask, softmax over the key axis in `f32`, times `V`; heads re-joined by concatenation. Written out with `matmul`, not `scaled_dot_product_attention`, whose fused kernels are a different sequence of operations |
| masking of an absent member | REPRODUCED — additive `presence · 10000 − 10000` on the key axis (`0` present, `−10000` absent), broadcast over heads and queries; never `−inf` |
| initial weights | REPRODUCED — loaded from the engine's file, tensor for tensor by name |
| episodes and their order | REPRODUCED — one step per train batch, in file order, `epochs` passes, no shuffling |
| objective | REPRODUCED — closed-form Gaussian CRPS of `(mean, σ = 1e-3 + softplus(raw))`, the mean over the batch's targets; the held-out and probe losses are the same score, each batch weighted by its target count, with no gradient |
| optimiser | REPRODUCED — AdamW, the given learning rate, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0` |
| gradient clip | REPRODUCED — global L2 norm over the parameters in name order, `coef = min(1, max_norm / (norm + 1e-6))`, applied unconditionally (`torch.nn.utils.clip_grad_norm_`, which the engine's clip mirrors) |
| held-out probe timing | REPRODUCED — after each epoch's last step, over the parameters as they then stand |
| arithmetic | DIFFERENT, irreducibly — `f32` on both, but each backend's matmul blocking, reduction order and `erf`/`exp` kernels are its own, so results agree to rounding and not to the bit. Measured on the committed spec over 180 steps: see the README, which also records how far each member amplifies one ulp |
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import ladder_leg as ll

MASKED_LOGIT = -10000.0
NORM_EPS = 1e-5


def linear_names(*layers: str) -> set[str]:
    return {f"{layer}.{kind}" for layer in layers for kind in ("weight", "bias")}


def mlp_names(prefix: str) -> set[str]:
    return linear_names(f"{prefix}.fc1", f"{prefix}.fc2")


def tnp_names(num_layers: int) -> set[str]:
    names = linear_names("context_embed", "target_embed", "final_norm") | {"query_marker"} | mlp_names("head")
    for n in range(num_layers):
        names |= linear_names(f"layer.{n}.q", f"layer.{n}.v", f"layer.{n}.attn_norm", f"layer.{n}.mlp_norm") | {f"layer.{n}.k.weight"} | mlp_names(f"layer.{n}.mlp")
    return names


def architecture_of(names: set[str]) -> tuple[str, int | None]:
    """The member whose parameter names are exactly `names`, and a Tnp's depth."""
    if names == mlp_names("phi") | mlp_names("rho"):
        return "Cnp", None
    if names == linear_names("query", "key", "value") | {"prior_key", "prior_value"} | mlp_names("rho"):
        return "AttnCnp", None
    depth = len({name.split(".")[1] for name in names if name.startswith("layer.")})
    if names == tnp_names(depth):
        return "Tnp", depth
    raise SystemExit(f"the initial weights are no member's parameter set: {sorted(names)}")


class ContextPredictor(torch.nn.Module):
    """One of the three members, holding the engine's tensors under the engine's
    names."""

    def __init__(self, weights: dict[str, torch.Tensor], num_heads: int | None) -> None:
        super().__init__()
        self.architecture, self.num_layers = architecture_of(set(weights))
        if self.architecture != "Cnp" and num_heads is None:
            raise SystemExit(f"{self.architecture} attends: --num-heads is required (the engine leg's identity.num_heads)")
        self.num_heads = num_heads
        self.names = sorted(weights)
        for name in self.names:
            self.register_parameter(name.replace(".", "__"), torch.nn.Parameter(weights[name].clone()))

    def p(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "__"))

    def ordered_parameters(self) -> list[torch.Tensor]:
        return [self.p(name) for name in self.names]

    def state(self) -> dict[str, torch.Tensor]:
        return {name: self.p(name).detach().clone() for name in self.names}

    def linear(self, layer: str, x: torch.Tensor) -> torch.Tensor:
        bias = f"{layer}.bias"
        return F.linear(x, self.p(f"{layer}.weight"), self.p(bias) if bias in self.names else None)

    def norm(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.shape[-1],), self.p(f"{name}.weight"), self.p(f"{name}.bias"), eps=NORM_EPS)

    def mlp(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        return self.linear(f"{prefix}.fc2", F.gelu(self.linear(f"{prefix}.fc1", x)))

    def attention(self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`softmax(QKᵀ/√head_dim + mask)·V` per head; `[B, S, hidden]` in and out,
        `mask` additive `[B, 1, 1, kv_len]`."""
        b, q_len, hidden = query.shape
        head_dim = hidden // self.num_heads

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(b, x.shape[1], self.num_heads, head_dim).transpose(1, 2).contiguous()

        q, k, v = split(query), split(keys), split(values)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5) + mask
        attended = torch.matmul(torch.softmax(scores, dim=-1), v)
        return attended.transpose(1, 2).contiguous().reshape(b, q_len, hidden)

    @staticmethod
    def key_mask(presence: torch.Tensor) -> torch.Tensor:
        """Additive `[B, 1, 1, 1 + k]` mask with an always-present member 0."""
        present = torch.cat([torch.ones(presence.shape[0], 1), presence], dim=1)
        return (present * -MASKED_LOGIT + MASKED_LOGIT)[:, None, None, :]

    def forward(self, episode: dict[str, torch.Tensor]) -> torch.Tensor:
        target_x, presence = episode["target_x"], episode["presence"]
        context_xy = torch.cat([episode["context_x"], episode["context_y"]], dim=2)
        b = target_x.shape[0]

        if self.architecture == "Cnp":
            phi = self.mlp("phi", context_xy)
            count = presence.sum(dim=1, keepdim=True)
            pooled = (phi * presence.unsqueeze(2)).sum(dim=1) / count.clamp(min=1.0)
            return self.mlp("rho", torch.cat([pooled, target_x, count], dim=1))

        if self.architecture == "AttnCnp":
            query = self.linear("query", target_x).unsqueeze(1)
            keys = torch.cat([self.p("prior_key").expand(b, -1, -1), self.linear("key", episode["context_x"])], dim=1)
            values = torch.cat([self.p("prior_value").expand(b, -1, -1), self.linear("value", context_xy)], dim=1)
            attended = self.attention(query, keys, values, self.key_mask(presence)).squeeze(1)
            return self.mlp("rho", torch.cat([attended, target_x], dim=1))

        target_token = self.linear("target_embed", target_x).unsqueeze(1) + self.p("query_marker")
        tokens = torch.cat([target_token, self.linear("context_embed", context_xy)], dim=1)
        mask = self.key_mask(presence)
        for n in range(self.num_layers):
            normed = self.norm(f"layer.{n}.attn_norm", tokens)
            q, k, v = (self.linear(f"layer.{n}.{proj}", normed) for proj in "qkv")
            tokens = tokens + self.attention(q, k, v, mask)
            tokens = tokens + self.mlp(f"layer.{n}.mlp", self.norm(f"layer.{n}.mlp_norm", tokens))
        return self.mlp("head", self.norm("final_norm", tokens[:, 0, :]))


def crps_gaussian(head: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean, sigma = head[:, 0], F.softplus(head[:, 1]) + 1e-3
    z = (mean - target) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / 2.0**0.5))
    pdf = torch.exp(-0.5 * z * z) / (2.0 * torch.pi) ** 0.5
    return (sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.pi**0.5)).mean()


def load_episodes(path: Path) -> list[dict[str, torch.Tensor]]:
    """The batches of one episode file, in index order: `{i}.target_x`,
    `.context_x`, `.context_y`, `.presence`, `.target_y`."""
    batches: dict[int, dict[str, torch.Tensor]] = {}
    for name, tensor in load_file(str(path)).items():
        index, field = name.split(".")
        batches.setdefault(int(index), {})[field] = tensor
    return [batches[i] for i in sorted(batches)]


def score_episodes(model, batches) -> float:
    """The objective over every target of `batches`, each batch weighted by its
    target count, with no gradient — `score_episodes` in the engine."""
    total, targets = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            count = int(batch["target_y"].shape[0])
            total += float(crps_gaussian(model(batch), batch["target_y"])) * count
            targets += count
    return total / targets


KEY = "predictor_train_run"
# The seeds the ladder's seeded-loss rule is stated for, `1..=SEEDED_LOSS_SEEDS`.
DEFAULT_SEEDS = list(range(1, 13))
RUNG = "torch"


def run(args, seed: int, take: int) -> list[str]:
    unit = f"seed{seed}"
    stem = ll.leg_stem(RUNG, unit, take)
    input_dir = args.legs_dir / "input" / args.arch / unit
    weights = load_file(str(input_dir / "initial_weights.safetensors"))
    train = load_episodes(input_dir / "train_episodes.safetensors")
    heldout = load_episodes(input_dir / "heldout_episodes.safetensors")
    model = ContextPredictor(weights, args.num_heads)
    if model.architecture != args.arch:
        raise SystemExit(f"--arch {args.arch} but the initial weights are a {model.architecture}'s")
    hidden_dim = int(weights["rho.fc1.weight" if model.architecture != "Tnp" else "head.fc1.weight"].shape[0])
    if model.architecture != "Cnp" and hidden_dim % args.num_heads != 0:
        raise SystemExit(f"--num-heads {args.num_heads} does not divide the hidden width {hidden_dim}")
    batch_sizes = {int(b["target_y"].shape[0]) for b in train}
    if len(batch_sizes) != 1:
        raise SystemExit("the train episodes do not share one target count, so `batch` names no one step")
    parameters = model.ordered_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    held_out_at_init = score_episodes(model, heldout)
    train_probe_series = [score_episodes(model, train)]
    trajectory = []
    iter_wall_s: list[float] = []
    step_losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        for batch in train:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = crps_gaussian(model(batch), batch["target_y"])
            step_losses.append(float(loss))
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            iter_wall_s.append(time.perf_counter() - t0)
        trajectory.append({"epoch": epoch, "held_out_mean": score_episodes(model, heldout), "run_wall_s_cumulative": time.perf_counter() - started})
        train_probe_series.append(score_episodes(model, train))
    peak = ll.peak_rss_bytes()

    final_path = args.legs_dir / f"{stem}.final_weights.safetensors"
    save_file(model.state(), str(final_path))
    with torch.no_grad():
        heads = [model(batch) for batch in heldout]
    keys = [f"test{e}_{row}" for e, head in enumerate(heads) for row in range(head.shape[0])]
    predictions = torch.cat(heads, dim=0)
    vectors, dim = ll.write_vector_rows(args.legs_dir, stem, keys, predictions)

    block = {
        # Identity.
        "seed": seed,
        "architecture": model.architecture,
        "context_k": int(train[0]["context_x"].shape[1]),
        "feature_dim": int(train[0]["target_x"].shape[1]),
        "value_dim": int(train[0]["context_y"].shape[2]),
        "hidden_dim": hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "head_width": dim,
        "initial_weights_sha256": ll.sha256_file(input_dir / "initial_weights.safetensors"),
        "train_episodes_sha256": ll.sha256_file(input_dir / "train_episodes.safetensors"),
        "heldout_episodes_sha256": ll.sha256_file(input_dir / "heldout_episodes.safetensors"),
        "epochs": args.epochs,
        "batch": batch_sizes.pop(),
        "lr": args.learning_rate,
        "weight_decay": 0.0,
        "schedule": "constant",
        "compute_precision": "f32",
        # Provenance.
        "rung": RUNG,
        "unit": unit,
        "take": take,
        "grad_clip": args.grad_clip,
        "objective": "gaussian-crps",
        "train_episodes": len(train),
        "heldout_episodes": len(heldout),
        "iters_measured": len(iter_wall_s),
        "final_weights": ll.artifact_of(final_path),
        "trainer": f"torch.optim.AdamW over a {model.architecture} twin",
        "step_losses": step_losses,
        **ll.provenance(),
        # Measured.
        "iter_wall_s": iter_wall_s,
        "peak_rss_bytes": peak,
        "peak_vram_bytes": ll.NOT_MEASURED_BYTES,
        "outcome_digest": ll.vector_rows_digest(keys, predictions),
        "held_out_example_mean": trajectory[-1]["held_out_mean"],
        "held_out_at_init": held_out_at_init,
        "trajectory": trajectory,
        "vectors_file": f"{stem}.vectors.f32",
        "vector_dim": dim,
        # Facts.
        "train_probe_series": train_probe_series,
    }
    return [ll.file_leg(args.legs_dir, KEY, stem, block, "torch_context_predictor.py")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--legs-dir", type=Path, required=True, help="the engine legs' directory: inputs under input/<arch>/seed<N>/, legs filed beside them")
    parser.add_argument("--arch", choices=["Cnp", "AttnCnp", "Tnp"], required=True)
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=DEFAULT_SEEDS, help="the seed units to train, comma-separated (one process each); the twelve the ladder's learning rule is stated for by default")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--grad-clip", type=float, required=True)
    parser.add_argument("--num-heads", type=int, required=True, help="the configuration's heads: the engine leg's identity.num_heads (a Cnp builds no attention)")
    parser.add_argument("--num-layers", type=int, required=True, help="the configuration's layers: the engine leg's identity.num_layers (only a Tnp builds them)")
    parser.add_argument("--takes", type=int, default=2, help="measured repeats of each seed, each in its own process; the ladder measures a rung against itself with two")
    parser.add_argument("--take", type=int, default=1, help="the take a single seed's run is filed as")
    args = parser.parse_args()

    points = [(seed, take) for seed in args.seeds for take in range(1, args.takes + 1)]

    def argv_for(point) -> list[str]:
        seed, take = point
        argv = ["--legs-dir", str(args.legs_dir), "--arch", args.arch, "--seeds", str(seed), "--take", str(take)]
        for flag in ("epochs", "learning_rate", "grad_clip", "num_heads", "num_layers"):
            argv += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
        return argv

    files = ll.legs_per_point(points, lambda point: run(args, point[0], args.take if len(points) == 1 else point[1]), argv_for)
    print(json.dumps(files))


if __name__ == "__main__":
    main()
