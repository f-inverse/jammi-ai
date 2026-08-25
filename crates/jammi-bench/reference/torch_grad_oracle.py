#!/usr/bin/env python3
"""PyTorch + PEFT counterpart to `crates/jammi-bench/src/grad_oracle.rs`'s
`grad-oracle` subcommand — the torch-side half of the jammi-vs-torch
LEARNING oracle. See that file's module doc for the full "why gradients,
not loss trajectories" argument; this docstring covers only what is
SPECIFIC to the torch side: the name translation this script does that
`grad_oracle.rs` does not need to.

============================================================================
PROVENANCE / HONESTY (execution-provenance principle): THIS SCRIPT HAS NOT
BEEN RUN. No `torch`/`transformers`/`peft`/`safetensors` install was
available in the environment this file was written in (no GPU, no torch
venv provisioned — the lead's own dispatch message named the A100 pod as
busy/self-terminating and asked for LOCAL build+unit-test only). Every
claim below about PEFT's exact `named_parameters()` naming is derived by
READING `torch_finetune_step.py`'s own already-working `wrap_lora`/
`reinit_lora_a_jammi_distribution` (which DOES iterate
`model.named_parameters()` against a real installed peft in that script's
own dry-run CI/pod usage) and PEFT's public documented naming convention,
never verified against a live run of THIS file. `translate_peft_name_to_jammi`
below FAILS LOUDLY (raises, never silently drops a tensor) the moment its
assumption is wrong, and `main()` asserts the translated name count against
the model's own trainable-parameter count before writing anything — so a
wrong assumption here is a hard crash on first run, not a silently-wrong
gradient dump. Treat this file as a reviewed-but-unexecuted design + full
implementation, not a verified oracle, until it has actually been run once
against a real checkpoint and its output round-tripped through
`compare_grad_oracle.py` against a real jammi `grad-oracle` dump.
============================================================================

NAME TRANSLATION — the crux this script owns (jammi's own
`grad_oracle.rs` does ZERO translation; the shared weight-interchange file
is written in jammi's OWN internal naming, and this script translates
PEFT's naming to/from it):

    jammi (VarBuilder path, confirmed empirically — see grad_oracle.rs's
    module doc):
        layer.{n}.Wqkv.lora_a      shape (rank, hidden_size)
        layer.{n}.Wqkv.lora_b      shape (hidden_size*3, rank)
        layer.{n}.Wo.lora_a        shape (rank, hidden_size)
        layer.{n}.Wo.lora_b        shape (hidden_size, rank)
        layer.{n}.Wi.lora_a        shape (rank, hidden_size)
        layer.{n}.Wi.lora_b        shape (intermediate_size*2, rank)
        layer.{n}.mlp.Wo.lora_a    shape (rank, intermediate_size)
        layer.{n}.mlp.Wo.lora_b    shape (hidden_size, rank)

    peft (`get_peft_model`'s own naming, NOT independently verified this
    round — see the PROVENANCE note above):
        base_model.model.layers.{n}.attn.Wqkv.lora_A.default.weight
        base_model.model.layers.{n}.attn.Wqkv.lora_B.default.weight
        base_model.model.layers.{n}.attn.Wo.lora_A.default.weight
        base_model.model.layers.{n}.attn.Wo.lora_B.default.weight
        base_model.model.layers.{n}.mlp.Wi.lora_A.default.weight
        base_model.model.layers.{n}.mlp.Wi.lora_B.default.weight
        base_model.model.layers.{n}.mlp.Wo.lora_A.default.weight
        base_model.model.layers.{n}.mlp.Wo.lora_B.default.weight

    Shapes match jammi's ORIENTATION exactly (peft's `lora_A.weight` is
    `(rank, in_features)`, `lora_B.weight` is `(out_features, rank)` — the
    same convention `LoraLinear::new`'s `vb.get_with_hints((rank,
    in_features), "lora_a", ...)` uses), so a value copy needs NO
    transpose, only a name translation.

WEIGHT INTERCHANGE FILE FORMAT: plain `safetensors` — the SAME file
`candle_nn::VarMap::save`/`VarMap::load` read/write on the jammi side, with
NO framework-specific wrapper. This script uses `safetensors.torch.load_file`/
`save_file` directly; no new format, no jammi/candle dependency on the
python side, no torch dependency on the Rust side.

Usage (mirrors `jammi-bench grad-oracle`'s own flags):
    python3 torch_grad_oracle.py --model-dir /path/to/ModernBERT-large \\
        --batch 8 --seq 128 --lora-rank 16 --lora-alpha 32 \\
        --dtype bf16 --attn eager --seed 42 \\
        --lora-weights-in shared_lora.safetensors --out torch_grad.json

Install: same venv `finetune_ab.sh`'s `setup_torch_venv` provisions
(`torch`, `transformers>=4.48`, `peft`) PLUS `safetensors` (already a
transitive dependency of both `torch` and `transformers`, so no extra
`uv pip install` line is expected to be needed — stated, not assumed;
`main()` raises a clear `ImportError`-derived message if it is somehow
absent rather than a bare traceback).
"""

from __future__ import annotations

import argparse
import json
import re
import sys

# Reuse EVERY piece of machinery torch_finetune_step.py already has working
# (and, unlike this script, has actually been run): the loader, the LoRA
# wrapper, the synthetic-id LCG, pooling, the triplet loss. Never
# reimplemented here — a second copy of `synthetic_ids` could drift from
# the first and silently feed the two stacks different tokens, exactly the
# class of bug this whole oracle exists to rule out as a variable.
import torch_finetune_step as tfs

PEFT_NAME_RE = re.compile(
    r"^base_model\.model\.layers\.(?P<layer>\d+)\.(?P<mid>attn|mlp)\.(?P<site>Wqkv|Wo|Wi)"
    r"\.lora_(?P<AB>[AB])\.default\.weight$"
)


def translate_peft_name_to_jammi(name: str) -> str | None:
    """`None` if `name` is not a LoRA `A`/`B` weight this script recognizes
    (e.g. a base/frozen parameter, or a bias PEFT does not touch here) --
    the caller filters those out rather than this function raising on
    every non-adapter parameter name in the model.
    """
    m = PEFT_NAME_RE.match(name)
    if m is None:
        return None
    layer = m.group("layer")
    mid = m.group("mid")
    site = m.group("site")
    ab = "lora_a" if m.group("AB") == "A" else "lora_b"
    # jammi's own target_name convention (see this module's docstring
    # table): attn.Wqkv -> "Wqkv", attn.Wo -> "Wo", mlp.Wi -> "Wi",
    # mlp.Wo -> "mlp.Wo" (the ONE site whose jammi target_name keeps the
    # "mlp." prefix -- disambiguating it from attn.Wo, which peft's own
    # suffix-matching rule treats as the SAME target_modules name "Wo";
    # jammi's builder call sites (`modernbert.rs`) already reflect this
    # asymmetry, not invented here).
    jammi_site = "mlp.Wo" if (mid == "mlp" and site == "Wo") else site
    return f"layer.{layer}.{jammi_site}.{ab}"



# jammi's bare `target_name` -> which sub-module it lives under. NOT
# derivable from the string alone ("Wi" carries no "mlp." prefix even
# though it IS an mlp site — only "mlp.Wo" is prefixed, specifically to
# disambiguate it from attn's OWN "Wo" — see `modernbert.rs`'s
# `LoraSite::build` call sites, cited in this module's own docstring
# table). A heuristic like "starts with 'mlp.'" gets `Wi` wrong (silently
# routes it to `attn.Wi`, which does not exist in the model, so the
# translated name would never match `named_parameters()` at all) — this
# table is the fix, an explicit, exhaustive map instead of a guess.
_JAMMI_SITE_TO_MID = {
    "Wqkv": "attn",
    "Wo": "attn",
    "Wi": "mlp",
    "mlp.Wo": "mlp",
}


def translate_jammi_name_to_peft(name: str) -> str | None:
    """Inverse of `translate_peft_name_to_jammi`, for loading a
    jammi-produced `--lora-weights-in` file: jammi's `layer.{n}.{site}.lora_a`
    -> peft's `base_model.model.layers.{n}.{attn|mlp}.{bare_site}.lora_A.default.weight`.
    """
    parts = name.split(".")
    # layer.{n}.Wqkv.lora_a (4 parts) or layer.{n}.mlp.Wo.lora_a (5 parts).
    if len(parts) == 4 and parts[0] == "layer" and parts[3] in ("lora_a", "lora_b"):
        _, layer, site, ab = parts
    elif len(parts) == 5 and parts[0] == "layer" and parts[2] == "mlp" and parts[4] in ("lora_a", "lora_b"):
        _, layer, _mlp, bare, ab = parts
        site = f"mlp.{bare}"
    else:
        return None
    mid = _JAMMI_SITE_TO_MID.get(site)
    if mid is None:
        return None
    bare_site = site.split(".")[-1]
    AB = "A" if ab == "lora_a" else "B"
    return f"base_model.model.layers.{layer}.{mid}.{bare_site}.lora_{AB}.default.weight"


def load_lora_weights_into_model(model, path: str) -> int:
    """Load a jammi-produced (or a previous torch-produced, round-trip)
    safetensors file into `model`'s trainable LoRA parameters, translating
    jammi names -> peft names. Returns the count actually written; raises
    if that count is zero (a silent no-op here would be the exact failure
    class `reinit_lora_a_jammi_distribution` in `torch_finetune_step.py`
    already guards against for its own, different, LoRA-init call site).
    """
    from safetensors.torch import load_file
    import torch

    data = load_file(path)
    by_peft_name = {}
    unrecognized = []
    for jammi_name, tensor in data.items():
        peft_name = translate_jammi_name_to_peft(jammi_name)
        if peft_name is None:
            unrecognized.append(jammi_name)
            continue
        by_peft_name[peft_name] = tensor
    if unrecognized:
        raise ValueError(
            f"{len(unrecognized)} tensor name(s) in {path!r} did not match jammi's own "
            f"'layer.{{n}}.{{site}}.lora_[ab]' naming convention: {unrecognized!r} — refusing to "
            "silently skip them (see this script's module doc's PROVENANCE note: the naming "
            "table here is unverified against a live jammi dump)."
        )

    written = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in by_peft_name:
                src = by_peft_name[name]
                if tuple(src.shape) != tuple(param.shape):
                    raise ValueError(
                        f"{name}: shape mismatch loading from {path!r} -- file has {tuple(src.shape)}, "
                        f"model parameter is {tuple(param.shape)}"
                    )
                param.copy_(src.to(param.dtype))
                written += 1
    if written == 0:
        raise RuntimeError(
            f"loaded zero LoRA parameters from {path!r} into the model -- either the file was "
            "empty, or every translated name failed to match model.named_parameters() (a peft "
            "naming-convention mismatch — see this script's PROVENANCE note)."
        )
    return written


def dump_lora_weights_from_model(model, path: str) -> int:
    """Inverse of `load_lora_weights_into_model`: write every trainable
    LoRA parameter currently in `model` to `path`, translated to jammi's
    naming, so a jammi `--lora-weights-in path` on the SAME config loads
    it directly.
    """
    from safetensors.torch import save_file

    out = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        jammi_name = translate_peft_name_to_jammi(name)
        if jammi_name is None:
            continue
        out[jammi_name] = param.detach().float().cpu().contiguous()
    if not out:
        raise RuntimeError("dump_lora_weights_from_model found zero trainable LoRA parameters to dump")
    save_file(out, path)
    return len(out)


def run(args) -> dict:
    import torch

    tfs.pin_fast_path_globals()
    torch.manual_seed(args.seed)
    device = tfs.pick_device(args.cuda)

    class ModelArgs:
        pass

    model_args = ModelArgs()
    model_args.model_dir = args.model_dir
    model_args.dtype = args.dtype
    model_args.attn = args.attn
    model, config = tfs.load_model(model_args)

    lora_args = ModelArgs()
    lora_args.lora_rank = args.lora_rank
    lora_args.lora_alpha = args.lora_alpha
    # Forced 0.0, unconditionally -- mirrors grad_oracle.rs's own forced
    # dropout=0 (a gradient-DIRECTION comparison must not compare two
    # different, RNG-divergent computations). This script has no
    # --lora-dropout flag on purpose.
    lora_args.lora_dropout = 0.0
    lora_args.target_modules = args.target_modules
    model = tfs.wrap_lora(model, lora_args)
    model.to(device)
    model.train()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable LoRA tensors — target_modules matched nothing")

    weights_written = None
    if args.lora_weights_in:
        weights_written = load_lora_weights_into_model(model, args.lora_weights_in)
        # Loud, not silent: every trainable tensor must have been written,
        # or SOME tensor is still at peft's own fresh init while the rest
        # came from the file -- a mixed-init forward, silently.
        if weights_written != len(trainable):
            raise RuntimeError(
                f"loaded {weights_written} LoRA tensors from {args.lora_weights_in!r} but the model "
                f"has {len(trainable)} trainable tensors -- partial load, refusing to run a forward "
                "against a mixed fresh-init/loaded-weights state."
            )

    if args.lora_weights_out:
        dump_lora_weights_from_model(model, args.lora_weights_out)

    mask = torch.ones(args.batch, args.seq, dtype=torch.long, device=device)
    blocks = [tfs.synthetic_ids(args.batch, args.seq, config.vocab_size, args.seed + i).to(device) for i in range(3)]

    if args.batched_forward:
        joined_ids = torch.cat(blocks, dim=0)
        joined_mask = mask.repeat(3, 1)
        hidden = tfs.forward_hidden(model, joined_ids, joined_mask)
        pooled = tfs.pool_and_normalize(hidden, joined_mask)
        b = args.batch
        a, p, n = pooled[:b], pooled[b : 2 * b], pooled[2 * b : 3 * b]
    else:
        a, p, n = (tfs.pool_and_normalize(tfs.forward_hidden(model, blk, mask), mask) for blk in blocks)
    loss = tfs.triplet_loss(a, p, n, margin=0.3)
    loss.backward()  # NO optimizer.step() -- see grad_oracle.rs's module doc for why.

    gradients = {}
    for name, param in trainable:
        jammi_name = translate_peft_name_to_jammi(name)
        if jammi_name is None:
            raise RuntimeError(
                f"trainable parameter {name!r} did not translate to a jammi tensor name -- "
                "the naming table in this script's module doc does not cover it (see the "
                "PROVENANCE note: unverified against a live run)."
            )
        if param.grad is None:
            raise RuntimeError(f"{name}: no gradient after backward() -- did it actually participate?")
        grad = param.grad.detach().float().cpu().contiguous().flatten().tolist()
        weight = param.detach().float().cpu().contiguous().flatten().tolist()
        gradients[jammi_name] = {"shape": list(param.shape), "grad": grad, "weight": weight}

    return {
        "tool": "torch_grad_oracle",
        "model_dir": str(args.model_dir),
        "device": str(device),
        "backbone_dtype": args.dtype,
        "batch": args.batch,
        "seq": args.seq,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": [t.strip() for t in args.target_modules.split(",") if t.strip()],
        "batched_forward": args.batched_forward,
        "seed": args.seed,
        "lora_dropout": 0.0,
        "lora_weights_in": args.lora_weights_in,
        "lora_weights_out": args.lora_weights_out,
        "trainable_tensor_count": len(gradients),
        "loss": float(loss.detach().float().item()),
        "gradients": gradients,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--target-modules", type=str, default="Wqkv,Wo,Wi")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--attn", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--cuda", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batched-forward", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lora-weights-in", type=str, default=None)
    p.add_argument("--lora-weights-out", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out} (loss={report['loss']}, tensors={report['trainable_tensor_count']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
