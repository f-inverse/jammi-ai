#!/usr/bin/env python3
"""PyTorch + PEFT counterpart to `crates/jammi-bench/src/grad_oracle.rs`'s
`grad-oracle` subcommand — the torch-side half of the jammi-vs-torch
LEARNING oracle. See that file's module doc for the full "why gradients,
not loss trajectories" argument; this docstring covers only what is
SPECIFIC to the torch side: the name translation this script does that
`grad_oracle.rs` does not need to.

============================================================================
PROVENANCE / HONESTY (execution-provenance principle): AS OF THE F2/F3
AUDIT-FIX ROUND ON PR #372, THIS SCRIPT HAS BEEN RUN — once, live, on an
A100 pod (ModernBERT-large, `--batch 8 --seq 128 --seed 42`, jammi tip
`e62c8a8`), reported by the lead who dispatched that pod job (not verified
locally by this fix round — no GPU was available here; see this repo's
`crates/jammi-bench` agent contract's "no GPU" disclosure). That run's
translated-name count DID match the model's own trainable-parameter count
(the `main()` assertion described below did not fire), and its output DID
round-trip through `compare_grad_oracle.py` against a real jammi
`grad-oracle` dump, producing (among others) these overall cosine
similarities: torch-eager vs torch-sdpa 0.825; torch-bf16 vs torch-f32
0.924; jammi-f32 vs torch-f32 0.9999998 (near-perfect, as expected — f32
has no bf16 rounding to diverge on). A separately-introduced defect on
that same run scored 0.30-0.53. Treat those five numbers as the empirical
anchor for picking a REAL `--cosine-floor` (see
`compare_grad_oracle.py`'s `derive_cosine_floor` doc for why its own
DERIVED worst-case bound, ~-0.40 at these dimensions, is far looser than
what real bf16 noise actually costs) — not the abstractly-derived floor.
That run ALSO surfaced that `dL/dA` is EXACTLY `0.0` on BOTH stacks for
every `lora_a` tensor at this fresh `LoraInitMode::ZerosB` init (112 of
224 matched tensors) — see "Structural limitation: a single fresh-init
call tests only `dL/dB`" below — and that the weight-identity check F3
added (`compare_grad_oracle.py`'s `_weight_mismatches`) held on that run
by actual agreement, not by luck of a loose bound: `max|w_jammi - w_torch|
= 1.86e-9` over 224 tensors -- orders of magnitude inside the ULP-relative
tolerance `compare_grad_oracle.py`'s `WEIGHT_MATCH_ULPS`/`_weight_element_tolerance`
derive (advisory ii, round-2 audit fix on PR #372: this was a fixed `1e-4`
absolute constant, now an f32-ULP-relative bound).

Everything ELSE about this file beyond that one confirmed run (arbitrary
checkpoints, other `target_modules` sets, other dtypes/ranks/batch/seq
combinations) remains UNVERIFIED against a live run — one successful
execution at one config is evidence the mechanism works, not a proof it
is correct at every config this script accepts. `translate_peft_name_to_jammi`
below still FAILS LOUDLY (raises, never silently drops a tensor) the
moment its naming assumption is wrong for a config not yet exercised, and
`main()` still asserts the translated name count against the model's own
trainable-parameter count before writing anything.
============================================================================

STRUCTURAL LIMITATION — a single fresh-init call tests ONLY `dL/dB`:
`LoraLinear`'s forward is `base(x) + scaling * dropout(x @ A^T @ B^T)`
(`lora_linear.rs`'s own doc). At `LoraInitMode::ZerosB` (both `grad_oracle.rs`
and this script use it — see `run()`'s `lora_args`/`lora.init_mode` above),
`B` starts at the exact zero matrix, so `dL/dA` — which the chain rule
routes through `B^T @ dL/d(output)` — is the EXACT zero vector for ANY
value of `A`, on BOTH stacks, REGARDLESS of whether either stack's
backward arithmetic is correct. Confirmed empirically on the one live run
described above: every `lora_a` tensor's gradient measured EXACTLY `0.0`
on both dumps. This means a single forward+backward at a fresh init
provides ZERO evidence about whether jammi's and torch's `dL/dA`
computations agree — a real defect specific to the `dL/dA` path (a
transposed axis, a dropped scale factor) could NOT be caught this way; it
would print an uninformative, structurally-guaranteed cosine of `0.0` on
both an agreeing and a disagreeing implementation alike.
`compare_grad_oracle.py`'s `is_vacuous_pair`/`vacuous_tensor_count`
classify and surface exactly this case rather than let it masquerade as
either a pass or a fail signal. Catching a real `dL/dA` defect requires AT
LEAST one optimizer step first (moving `B` away from zero) — the
N-step teacher-forced extension `grad_oracle.rs`'s own module doc scopes
under "What this tier does NOT do" is what would close this gap; not
implemented this round.

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
import contextlib
import json
import re
import sys
import tempfile

# Reuse EVERY piece of machinery torch_finetune_step.py already has working
# (and, unlike this script, has actually been run): the loader, the LoRA
# wrapper, the synthetic-id LCG, pooling, the triplet loss. Never
# reimplemented here — a second copy of `synthetic_ids` could drift from
# the first and silently feed the two stacks different tokens, exactly the
# class of bug this whole oracle exists to rule out as a variable.
import torch_finetune_step as tfs


class _ModelArgs:
    """A throwaway attribute-bag `tfs.load_model`/`tfs.wrap_lora` accept in
    place of a full `argparse.Namespace` — those two functions only ever
    read specific attributes off whatever object they are handed, never the
    object's type, so a minimal stand-in is enough. Module-level (not a
    class nested inside `run()`) so both `run()` (builds the `model_args`
    passed to `load_model`) and `_run_with_model()` (builds the separate
    `lora_args` passed to `wrap_lora`) can construct one.
    """


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


# jammi's OWN CLI/interchange dtype vocabulary (`main.rs`'s
# `--backbone-dtype` choices, `grad_oracle.rs`'s
# `format!("{:?}", ComputePrecision::F32).to_lowercase()`): `f32`/`f16`/
# `bf16`. This script's `--dtype` flag keeps torch's OWN spelling (`fp32`,
# matching `torch_finetune_step.py`'s convention) since it feeds
# `torch_finetune_step.load_model`'s dtype map, which is shared machinery
# this script does not own and has its own reasons (the `amp-fp16` case) to
# keep torch's spelling -- only the WRITTEN REPORT's `backbone_dtype` field
# is translated to jammi's canonical spelling (B1 audit finding on PR #372;
# see `run()`'s own comment at that field).
_DTYPE_FLAG_TO_JAMMI_SPELLING = {
    "fp32": "f32",
    "bf16": "bf16",
}


def translate_dtype_flag_to_jammi_spelling(dtype_flag: str) -> str:
    """`--dtype`'s own spelling -> jammi's canonical `backbone_dtype`
    spelling. Raises on an unrecognized flag rather than passing an
    un-translated value through silently -- this script's `--dtype` choices
    are a closed set (`argparse`'s own `choices=` already enforces that at
    parse time), so an unrecognized value here means this map itself has
    drifted out of sync with `parse_args`, not a caller error.
    """
    try:
        return _DTYPE_FLAG_TO_JAMMI_SPELLING[dtype_flag]
    except KeyError:
        raise ValueError(
            f"translate_dtype_flag_to_jammi_spelling: unrecognized --dtype {dtype_flag!r} -- "
            f"this script's own --dtype choices are {sorted(_DTYPE_FLAG_TO_JAMMI_SPELLING)}, this "
            "map has drifted out of sync with parse_args' choices=."
        ) from None


# round-4 audit fold-in on PR #372: THIS module's own `checkpoint_identity`
# used to be a second, independently-drifting copy (and non-streaming,
# `fh.read()` of the whole file) — `torch_finetune_step.py` is the file both
# this module and `torch_finetune_step.py`'s own `run()` now share the
# STREAMING implementation through (mirrors this module's own doc's
# "Reuse EVERY piece of machinery torch_finetune_step.py already has
# working" convention for `synthetic_ids`/`triplet_loss`/etc — never
# reimplemented here). A bare module-level alias, not a wrapper function,
# so `tgo.checkpoint_identity` (every existing call site, including
# `test_torch_grad_oracle_names.py::CheckpointIdentityTests`) keeps working
# unchanged.
checkpoint_identity = tfs.checkpoint_identity


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

    # Captured (not discarded, unlike an earlier draft of this function):
    # `tfs.provenance(device, fast_path_globals)` below needs this return
    # value -- `torch_finetune_step.py`'s own `run()` keeps it for the exact
    # same reason (that file's own call site, cited in this module's
    # determinant table).
    fast_path_globals = tfs.pin_fast_path_globals()
    # Seed BEFORE any model/adapter/checkpoint construction -- including the
    # --dry-run donor checkpoint below -- mirrors torch_finetune_step.py's
    # own `run()` ordering (see that function's comment for why: peft's
    # default LoRA init draws from torch's global generator at
    # `get_peft_model` time, so the generator must already be seeded when
    # that call happens).
    torch.manual_seed(args.seed)
    device = tfs.pick_device(None if args.dry_run else args.cuda)

    dry_run_tmp = tempfile.TemporaryDirectory() if args.dry_run else contextlib.nullcontext()
    with dry_run_tmp as tmp_dir:
        if args.dry_run:
            # Mirrors torch_finetune_step.py's --dry-run: hardcode small,
            # always-valid batch/seq knobs (the donor checkpoint's own
            # vocab/hidden size) so the smoke test never depends on a real
            # checkpoint; --dtype/--attn/--lora-rank/--lora-alpha/
            # --target-modules/--batched-forward are left as the caller
            # requested, exercising the SAME load_model/wrap_lora code the
            # real GPU path uses -- never a separate, untested dry-run-only
            # code path.
            args = argparse.Namespace(**vars(args))
            args.batch = tfs.DRY_RUN_BATCH
            args.seq = tfs.DRY_RUN_SEQ
            args.model_dir = tfs.build_dry_run_checkpoint(tmp_dir)

        model_args = _ModelArgs()
        model_args.model_dir = args.model_dir
        model_args.dtype = args.dtype
        model_args.attn = args.attn
        model, config = tfs.load_model(model_args)
        # RESOLVED, not requested -- what HF's `AutoModel.from_pretrained`
        # actually picked, which can differ from `args.attn` (e.g. falling
        # back off `flash_attention_2` when the package is not installed).
        # Captured HERE, before `wrap_lora` (mirrors
        # `torch_finetune_step.py`'s own `run()`, cited in this module's
        # determinant table -- PEFT-wrapping can obscure `.config` access on
        # some versions, so read it off the UNWRAPPED model).
        resolved_attn_implementation = getattr(model.config, "_attn_implementation", "absent")
        return _run_with_model(
            args, model, config, device, fast_path_globals, resolved_attn_implementation
        )


def _run_with_model(
    args, model, config, device, fast_path_globals, resolved_attn_implementation
) -> dict:
    """The part of `run()` that is IDENTICAL whether `model`/`config` came
    from a real `--model-dir` or a `--dry-run` donor checkpoint — split out
    so `--dry-run`'s `tempfile.TemporaryDirectory()` context (which must
    stay open only long enough for `load_model` to read the checkpoint off
    disk, not for the rest of the forward+backward) does not have to wrap
    this entire function body.
    """
    import torch

    # Base-checkpoint CONTENT identity, computed BEFORE the forward off the
    # exact bytes this run loaded -- `args.model_dir` is valid here whether
    # this call came from a real `--model-dir` or a `--dry-run` donor
    # checkpoint (`run()`'s `tempfile.TemporaryDirectory()` context is still
    # open for the whole duration of this call — see `run()`'s own comment).
    checkpoint_identity_fields = checkpoint_identity(args.model_dir)

    lora_args = _ModelArgs()
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

    mask = torch.ones(args.batch, args.seq, dtype=torch.long, device=device)
    blocks = [tfs.synthetic_ids(args.batch, args.seq, config.vocab_size, args.seed + i).to(device) for i in range(3)]
    # Same digest jammi's `grad_oracle.rs` reports as `batch_token_id_sums`
    # (see that module's field doc): `[sum(anchor ids), sum(positive ids),
    # sum(negative ids)]`, computed from the SAME `blocks` both the batched
    # and unbatched forward arms below consume, so `compare_grad_oracle.py`
    # can refuse a comparison whose two dumps ran different token content
    # even though `synthetic_ids` is meant to be bit-identical across the
    # two stacks for the same `(seed, vocab)`.
    batch_token_id_sums = [int(b.sum().item()) for b in blocks]

    def _forward_loss():
        if args.batched_forward:
            joined_ids = torch.cat(blocks, dim=0)
            joined_mask = mask.repeat(3, 1)
            hidden = tfs.forward_hidden(model, joined_ids, joined_mask)
            pooled = tfs.pool_and_normalize(hidden, joined_mask)
            b = args.batch
            a, p, n = pooled[:b], pooled[b : 2 * b], pooled[2 * b : 3 * b]
        else:
            a, p, n = (tfs.pool_and_normalize(tfs.forward_hidden(model, blk, mask), mask) for blk in blocks)
        return tfs.triplet_loss(a, p, n, margin=0.3)

    # `args.warmup_steps` REAL forward+backward+`AdamW.step()` iterations, at
    # `args.warmup_lr`, on the SAME `blocks`/`mask` the measured forward
    # below reuses ("on the same data" — mirrors `grad_oracle.rs`'s own
    # `warmup_steps`/`warmup_lr` fields and their doc). `weight_decay=0.01`
    # is the SAME value `finetune_step.rs`'s own `build_fixture` hardcodes
    # (and the jammi-side `ParamsAdamW` warmup loop this mirrors) — kept
    # identical so the two stacks' first-step update formula matches term
    # for term, not just in `lr`. Runs BEFORE `--lora-weights-out` (moved
    # below, past this loop) so a caller's shared-weights file captures the
    # POST-warmup state, mirroring `grad_oracle.rs::run`'s own comment at its
    # `lora_weights_out` save call site.
    if args.warmup_steps > 0:
        optimizer = torch.optim.AdamW(
            [p for _, p in trainable], lr=args.warmup_lr, weight_decay=0.01
        )
        for _ in range(args.warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            warmup_loss = _forward_loss()
            warmup_loss.backward()
            optimizer.step()

    if args.lora_weights_out:
        dump_lora_weights_from_model(model, args.lora_weights_out)

    loss = _forward_loss()
    loss.backward()  # NO optimizer.step() on the MEASURED forward -- see grad_oracle.rs's module doc for why.

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
        # `None` under --dry-run, mirroring torch_finetune_step.py's own
        # convention: the donor checkpoint lives in a `TemporaryDirectory`
        # that is deleted the moment `run()` returns, so reporting its path
        # here would name a directory that no longer exists by the time
        # anything reads this report.
        "model_dir": None if args.dry_run else str(args.model_dir),
        "dry_run": args.dry_run,
        "device": str(device),
        # Base-checkpoint CONTENT identity -- see `checkpoint_identity`'s own
        # doc and `grad_oracle.rs`'s module doc's determinant table. IDENTITY
        # (replaces the un-comparable `model_dir` path above, which stays
        # emitted for human debugging only).
        **checkpoint_identity_fields,
        # torch/transformers/peft versions, device NAME (vs. `device` above,
        # which is the device TYPE/ordinal), git rev, and the fast-path pin
        # state -- reuses `torch_finetune_step.py`'s own `provenance()`
        # UNCHANGED (never a second, drifting implementation). PROVENANCE:
        # recorded, never compared -- two producers legitimately run on
        # different boxes/software stacks.
        "provenance": tfs.provenance(device, fast_path_globals),
        # What `--attn` REQUESTED vs. what HF's `AutoModel.from_pretrained`
        # actually RESOLVED to (`model.config._attn_implementation`, which
        # can fall back off the request -- e.g. no `flash_attention_2`
        # package installed). PROVENANCE: jammi has no equivalent CLI lever
        # (its own analog, WHICH KERNEL COMPOSITION actually dispatched, is
        # the `*_fused_dispatches`/`*_eager_dispatches` MEASUREMENT fields on
        # the jammi side -- see `grad_oracle.rs`'s module doc's determinant
        # table for why these are not directly comparable to each other).
        "attn_requested": args.attn,
        "attn_implementation": resolved_attn_implementation,
        # B1 audit finding on PR #372: emit jammi's OWN canonical spelling
        # (`f32`/`bf16`, never `fp32`) here -- `--dtype` itself keeps torch's
        # bare CLI-flag spelling (`fp32`, matching `torch_finetune_step.py`'s
        # own `--dtype` convention this script's flags otherwise mirror; see
        # this module's usage docstring), but the WRITTEN report is what
        # `compare_grad_oracle.py`'s run-identity check actually reads, and
        # that check's premise is IDENTICAL configuration on both sides --
        # jammi's own producer (`grad_oracle.rs`) has emitted `f32`/`f16`/
        # `bf16` since day one (`main.rs`'s `--backbone-dtype` choices), so
        # this is the side that was wrong, not the comparator (see
        # `compare_grad_oracle.py`'s `normalize_backbone_dtype`, kept as a
        # legacy-spelling fallback for any OLDER dump still carrying `fp32`
        # here, not as a substitute for fixing the spelling at the source).
        "backbone_dtype": translate_dtype_flag_to_jammi_spelling(args.dtype),
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
        "warmup_steps": args.warmup_steps,
        "warmup_lr": args.warmup_lr,
        "trainable_tensor_count": len(gradients),
        "batch_token_id_sums": batch_token_id_sums,
        "loss": float(loss.detach().float().item()),
        "gradients": gradients,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=str, default=None, help="Required unless --dry-run.")
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
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "0 (default) or the number of real forward+backward+AdamW.step() iterations to run, "
            "at --warmup-lr, on the SAME batch the measured (no-step) forward reuses, BEFORE that "
            "measured forward+backward -- mirrors grad_oracle.rs's own --warmup-steps/warmup-lr "
            "fields; see this script's `_run_with_model` for the exact ordering."
        ),
    )
    p.add_argument(
        "--warmup-lr",
        type=float,
        default=2e-4,
        help="The AdamW learning rate --warmup-steps runs at -- 2e-4, the SAME reference value "
        "finetune_step.rs's own build_fixture hardcodes.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "CPU smoke test against a tiny random-init 2-layer ModernBERT built on the fly "
            "(tfs.build_dry_run_checkpoint) instead of --model-dir -- exercises the REAL "
            "load_model/wrap_lora/forward/backward/name-translation code path, just at a "
            "small, always-valid shape. See crates/jammi-bench/reference/README.md."
        ),
    )
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args(argv)
    if not args.dry_run and not args.model_dir:
        p.error("--model-dir is required unless --dry-run")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"wrote {args.out} (loss={report['loss']}, tensors={report['trainable_tensor_count']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
