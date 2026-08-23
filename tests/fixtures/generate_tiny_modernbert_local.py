#!/usr/bin/env python3
"""Generate the ModernBERT sliding-window oracle: a tiny real-ModernBERT fixture
whose config exercises local attention, plus golden activations dumped from
HuggingFace `transformers`.

Every other ModernBERT fixture in this repo is `num_hidden_layers=1` /
`global_attn_every_n_layers=1`. Layer 0 is a *global* layer under ModernBERT's
own rule (`i % global_attn_every_n_layers != 0` selects sliding attention), so a
one-layer fixture computes identical output whether or not sliding-window
attention is implemented at all. That fixture set cannot express a
local-attention divergence, and no test built on it can.

This fixture can. With `num_hidden_layers=4` and `global_attn_every_n_layers=3`,
HuggingFace assigns:

    layer 0  full_attention      rope_theta = global_rope_theta  (160000)
    layer 1  sliding_attention   rope_theta = local_rope_theta   (10000)
    layer 2  sliding_attention   rope_theta = local_rope_theta   (10000)
    layer 3  full_attention      rope_theta = global_rope_theta  (160000)

`local_attention=16` gives `sliding_window = local_attention // 2 = 8`, so on a
sliding layer token i attends only to |j - i| <= 8. At the pinned sequence
length of 64 that band excludes most of the sequence, which is the point: an
implementation that ignores the window produces materially different output
rather than a rounding difference.

The fixture asserts its own discriminating power at generation time -- see
`QKV_LOGIT_GAIN` and `assert_fixture_discriminates`. A fixture that cannot tell
a correct implementation from a broken one is worse than no fixture, because it
reports green.

Two pinned inputs are dumped, and both matter:

  unpadded  attention_mask all ones. The additive padding mask is uniformly
            zero, so any masking effect observed can only come from the sliding
            window -- a fix that confuses the two cannot pass.
  padded    trailing PAD tokens. Proves the window and the padding mask compose;
            an implementation that applies the band but drops padding, or that
            lets a pad token contribute, fails here while passing `unpadded`.

Outputs, under cookbook/fixtures/tiny_modernbert_local/ (all committed; the
Rust test needs no torch and makes no network call):

    config.json          tiny ModernBertConfig with real key layout
    model.safetensors    tiny real-ModernBERT weights, `model.`-prefixed
    goldens.safetensors  pinned inputs + last_hidden_state + pooled embedding
    golden_manifest.json name -> shape -> dtype index of the goldens

Regenerate with:

    pip install "torch" "transformers" "safetensors"
    python tests/fixtures/generate_tiny_modernbert_local.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import ModernBertConfig, ModernBertModel

OUT = Path(__file__).resolve().parents[2] / "cookbook" / "fixtures" / "tiny_modernbert_local"

HIDDEN = 32
INTERMEDIATE = 64
HEADS = 2
LAYERS = 4
VOCAB = 256
MAX_POS = 512
GLOBAL_ATTN_EVERY_N_LAYERS = 3
LOCAL_ATTENTION = 16         # sliding_window = 8 -> |j - i| <= 8
GLOBAL_ROPE_THETA = 160_000.0
LOCAL_ROPE_THETA = 10_000.0
SEQ = 64
PAD_TAIL = 20                # trailing PAD positions in the `padded` case
SEED = 20260823
PAD_ID = 0

# Attention-logit gain applied to every Wqkv at fixture build time.
#
# This is load-bearing, not cosmetic. ModernBERT's default initializer draws
# Wqkv near zero, which makes every attention row nearly UNIFORM -- and over a
# uniform distribution, restricting which keys a query may attend to barely
# changes the result. A fixture built at the default scale therefore cannot
# discriminate a correct sliding-window implementation from one that ignores
# the window: measured at the default scale the whole divergence is ~5e-3 and
# the dual-RoPE half is ~1e-5, i.e. BELOW the parity tolerance, so a test built
# on it would pass while the bug is present.
#
# Real checkpoints have peaked attention. Scaling Wqkv puts the fixture in that
# regime, and `assert_fixture_discriminates` re-measures the margins on every
# regeneration so this can never silently drift back to degenerate.
QKV_LOGIT_GAIN = 8.0

# Minimum divergence a window-ignoring (resp. theta-ignoring) implementation
# must produce against the committed golden, in max-abs over the final hidden
# state. Both are >= 90x TOL_ABS (1e-4) in `golden_parity`-style tests.
MIN_WINDOW_DIVERGENCE = 1e-2
MIN_THETA_DIVERGENCE = 1e-3


def build_config(
    local_attention: int = LOCAL_ATTENTION,
    local_rope_theta: float = LOCAL_ROPE_THETA,
) -> ModernBertConfig:
    """Build the fixture config.

    The two window/theta parameters are constructor arguments rather than
    post-construction mutations because transformers folds them into
    `rope_parameters` during `__post_init__`; assigning to
    `cfg.local_rope_theta` afterwards is silently ignored.
    """
    return ModernBertConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_POS,
        layer_norm_eps=1e-5,
        global_attn_every_n_layers=GLOBAL_ATTN_EVERY_N_LAYERS,
        local_attention=local_attention,
        global_rope_theta=GLOBAL_ROPE_THETA,
        local_rope_theta=local_rope_theta,
        # Keep every special-token id inside this tiny vocab so the config is
        # self-consistent and transformers does not warn.
        pad_token_id=PAD_ID,
        bos_token_id=2,
        eos_token_id=3,
        cls_token_id=2,
        sep_token_id=3,
        attn_implementation="eager",
    )


def pinned_inputs() -> dict[str, torch.Tensor]:
    """Two deterministic (input_ids, attention_mask) pairs.

    Token ids avoid PAD_ID in the real positions so a real token can never be
    mistaken for padding.
    """
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(5, VOCAB, (1, SEQ), generator=g)

    unpadded_mask = torch.ones((1, SEQ), dtype=torch.long)

    padded_ids = ids.clone()
    padded_ids[0, SEQ - PAD_TAIL :] = PAD_ID
    padded_mask = unpadded_mask.clone()
    padded_mask[0, SEQ - PAD_TAIL :] = 0

    return {
        "unpadded.input_ids": ids,
        "unpadded.attention_mask": unpadded_mask,
        "padded.input_ids": padded_ids,
        "padded.attention_mask": padded_mask,
    }


def mean_pool_l2(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over unmasked tokens then L2-normalize.

    Mirrors the engine's `pool_and_normalize` for `Pooling::Mean`, so the golden
    is comparable to the encoder's own pooled `forward`, not just to its hidden
    states.
    """
    m = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1e-9)
    return F.normalize(summed / counts, p=2.0, dim=-1)


def assert_fixture_discriminates(state_dict, goldens) -> None:
    """Prove this fixture can tell a correct implementation from a broken one.

    Rebuilds the same weights under two deliberately wrong configurations and
    requires each to diverge from the committed golden by a real margin:

      window ignored  local_attention widened past the sequence, so every
                      sliding layer degenerates to full attention
      theta ignored   local_rope_theta set to the global value, so sliding
                      layers get the wrong RoPE base

    If either margin collapses, the fixture has drifted into a regime where the
    bug is unobservable and the parity test built on it would report green
    against a broken encoder. That is a generation-time failure, not something
    to be discovered later by trusting a passing test.
    """

    def rebuild(local_attention: int, local_theta: float) -> ModernBertModel:
        m = ModernBertModel(build_config(local_attention, local_theta)).eval()
        m.load_state_dict(state_dict)
        return m

    checks = (
        ("window", rebuild(SEQ * 64, LOCAL_ROPE_THETA), MIN_WINDOW_DIVERGENCE),
        ("theta", rebuild(LOCAL_ATTENTION, GLOBAL_ROPE_THETA), MIN_THETA_DIVERGENCE),
    )
    with torch.no_grad():
        for label, wrong, floor in checks:
            worst = 0.0
            for case in ("unpadded", "padded"):
                got = wrong(
                    input_ids=goldens[f"{case}.input_ids"],
                    attention_mask=goldens[f"{case}.attention_mask"],
                ).last_hidden_state
                delta = (got - goldens[f"{case}.last_hidden_state"]).abs().max().item()
                worst = max(worst, delta)
            print(f"  discrimination: {label + ' ignored':22s} max-abs = {worst:.5f} (floor {floor})")
            if worst < floor:
                raise SystemExit(
                    f"fixture cannot discriminate a {label}-ignoring implementation: "
                    f"max-abs {worst:.6g} < {floor}. The fixture is degenerate; a parity "
                    f"test built on it would pass against a broken encoder."
                )


def main() -> None:
    torch.manual_seed(SEED)
    cfg = build_config()
    model = ModernBertModel(cfg).eval()

    # Put attention in the peaked regime real checkpoints occupy; see
    # QKV_LOGIT_GAIN. Without this the fixture cannot observe its own subject.
    with torch.no_grad():
        for layer in model.layers:
            layer.attn.Wqkv.weight.mul_(QKV_LOGIT_GAIN)

    assert cfg.layer_types == [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ], f"fixture no longer exercises sliding attention: {cfg.layer_types}"
    assert cfg.sliding_window == LOCAL_ATTENTION // 2, cfg.sliding_window

    OUT.mkdir(parents=True, exist_ok=True)

    # The engine's loader expects the `model.`-prefixed key layout that the real
    # published checkpoints carry (ModernBertModel is the *base* model, so its
    # own state_dict omits the prefix a ForXxx wrapper would add).
    weights = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    save_file(weights, str(OUT / "model.safetensors"))

    with (OUT / "config.json").open("w") as f:
        json.dump(
            {
                "architectures": ["ModernBertModel"],
                "model_type": "modernbert",
                "hidden_size": HIDDEN,
                "num_hidden_layers": LAYERS,
                "num_attention_heads": HEADS,
                "intermediate_size": INTERMEDIATE,
                "vocab_size": VOCAB,
                "max_position_embeddings": MAX_POS,
                "layer_norm_eps": 1e-5,
                "pad_token_id": PAD_ID,
                "global_attn_every_n_layers": GLOBAL_ATTN_EVERY_N_LAYERS,
                "global_rope_theta": GLOBAL_ROPE_THETA,
                "local_attention": LOCAL_ATTENTION,
                "local_rope_theta": LOCAL_ROPE_THETA,
            },
            f,
            indent=2,
        )

    goldens = pinned_inputs()
    with torch.no_grad():
        for case in ("unpadded", "padded"):
            ids = goldens[f"{case}.input_ids"]
            mask = goldens[f"{case}.attention_mask"]
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            goldens[f"{case}.last_hidden_state"] = hidden.to(torch.float32)
            goldens[f"{case}.pooled"] = mean_pool_l2(hidden, mask).to(torch.float32)

    assert_fixture_discriminates(model.state_dict(), goldens)

    save_file({k: v.contiguous() for k, v in goldens.items()}, str(OUT / "goldens.safetensors"))

    with (OUT / "golden_manifest.json").open("w") as f:
        json.dump(
            {
                "source": "huggingface transformers ModernBertModel, eager attention",
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "seed": SEED,
                "layer_types": cfg.layer_types,
                "sliding_window": cfg.sliding_window,
                "tensors": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                    for k, v in sorted(goldens.items())
                },
            },
            f,
            indent=2,
        )

    print(f"wrote {OUT}")
    for k, v in sorted(goldens.items()):
        print(f"  {k:34s} {tuple(v.shape)} {str(v.dtype).replace('torch.', '')}")


if __name__ == "__main__":
    main()
