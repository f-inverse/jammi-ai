#!/usr/bin/env python3
"""The PyTorch rung of the `predictor-train-run` workload: exact twins of the
engine's three in-context predictors — `Cnp`, `AttnCnp`, `Tnp` — started from
the initial weights and trained over the episodes a `jammi-bench
predictor-train-run --arch <member>` leg wrote (`initial_weights.safetensors`,
`episodes.safetensors`), emitting the same leg fields: every optimizer step's
wall-clock and loss, the process's peak resident set, the trained weights, and
the head's raw output on every held-out test target with its digest.

Both stacks start from the same tensors and see the same batches in the same
order, and neither uses dropout, so the randomness between them is removed
rather than averaged over; what remains is numerics.

The member is read off the initial weights' tensor names, which are the
engine's own. A name set that is not exactly one member's — a tensor missing,
a tensor extra, a layer index skipped — is refused.

Against the engine (`crates/jammi-encoders/src/context/{cnp,attncnp,tnp,
attention,mod}.rs`, `crates/jammi-encoders/src/mask.rs`,
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
| head count | taken from `--num-heads`: it is not recoverable from the weights, and the engine leg's identity carries it |
| initial weights | REPRODUCED — loaded from the engine's file, tensor for tensor by name; the learned tokens start at zero, as the engine registers them |
| episodes and their order | REPRODUCED — one step per train batch, in file order, `epochs` passes, no shuffling |
| objective | REPRODUCED — closed-form Gaussian CRPS of `(mean, σ = 1e-3 + softplus(raw))`, the mean over the batch's targets |
| optimiser | REPRODUCED — AdamW, the given learning rate, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0` |
| gradient clip | REPRODUCED — global L2 norm over the parameters in name order, `coef = min(1, max_norm / (norm + 1e-6))`, applied unconditionally (`torch.nn.utils.clip_grad_norm_`, which the engine's clip mirrors) |
| step loss | REPRODUCED — the batch's loss at the parameters the step started from |
| arithmetic | DIFFERENT, irreducibly — `f32` on both, but each backend's matmul blocking, reduction order and `erf`/`exp` kernels are its own, so results agree to rounding and not to the bit. Measured on the committed spec over 180 steps: see the README, which also records how far each member amplifies one ulp |
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--num-heads", type=int, help="attention heads of an AttnCnp / Tnp (required for them): the engine leg's identity.num_heads")
    parser.add_argument("--warmup-steps", type=int, default=2)
    args = parser.parse_args()

    weights = load_file(str(args.initial_weights))
    episodes = load_episodes(args.episodes)
    model = ContextPredictor(weights, args.num_heads)
    attentive = model.architecture != "Cnp"
    hidden_dim = int(weights["rho.fc1.weight" if model.architecture != "Tnp" else "head.fc1.weight"].shape[0])
    if attentive and hidden_dim % args.num_heads != 0:
        raise SystemExit(f"--num-heads {args.num_heads} does not divide the hidden width {hidden_dim}")
    parameters = model.ordered_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

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
                torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
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

    identity = {
        "episodes_sha256": ll.artifact_of(args.episodes)["sha256"],
        "initial_weights_sha256": ll.artifact_of(args.initial_weights)["sha256"],
        "architecture": model.architecture,
        "hidden_dim": hidden_dim,
        "num_heads": args.num_heads if attentive else None,
        "num_layers": model.num_layers,
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
        [ll.leg("predictor-train-run", identity, ll.provenance(trainer=f"torch.optim.AdamW over a {model.architecture} twin", device="cpu"), measured)],
    )


if __name__ == "__main__":
    main()
