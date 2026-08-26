#!/usr/bin/env python3
"""esc-045 round 6 (GH #374): torch-side per-layer/per-site activation-
gradient dump, the E3/E5 counterpart to jammi's
`esc045_round4_per_layer_activation_gradient_dump`
(`crates/jammi-encoders/src/modernbert.rs`).

Reuses `torch_finetune_step.py` (`tfs`) and `torch_grad_oracle.py` (`tgo`,
for LoRA weight-file name translation) UNCHANGED — never a second,
drifting copy of `load_model`/`wrap_lora`/`checkpoint_identity`/
`translate_jammi_name_to_peft` (see those modules' own docs).

SEED / TOKEN CONTENT: jammi's round-4 dump test builds its synthetic batch
via `round4_synthetic_ids` (`modernbert.rs`, tests module) — a SEPARATE,
self-contained SplitMix64-derived PCG, NOT `torch_finetune_step.synthetic_ids`
(the LCG `finetune_step.rs`'s own generator uses, ids in `[1, vocab)`).
`_round4_synthetic_ids` below is a bit-identical Python port of the FORMER
(ids in `[0, vocab)`, INCLUDING the pad id `0` -- the round-4 test always
runs an all-ones attention mask, so id `0` is content, never padding) --
using the wrong generator here would feed the two stacks DIFFERENT tokens
and confound every comparison this script exists to make.

LOSS: `L = sum(hidden_final ** 2)` in f32, matching the round-4 jammi dump
test's own loss EXACTLY (not the triplet-hinge `torch_grad_oracle.py`
uses) -- see that jammi test's doc for why (every activation gets a
nonzero gradient regardless of hinge sparsity).

CAPTURE POINTS:
  * boundary.{1..N}: dL/d(output of encoder layer i), i in 1..=num_layers
    -- `register_full_backward_hook` on `model.base_model.model.layers[i-1]`
    (peft's own `PeftModel.base_model.model` wrapping around
    `ModernBertModel`, confirmed empirically on this checkpoint). Matches
    jammi's `boundary.{i}` exactly (jammi's `boundary.0`, the embeddings
    output, structurally never fires its own tap -- see that module's doc
    -- so this script does not attempt to capture a torch analogue of it
    either, for the same reason: it depends on no LoRA `Var`/`Parameter`).
  * forward.{0..N}: the FORWARD hidden-state value at the same N+1
    boundaries (0 = embeddings output / layer-0 input, N = final layer's
    output, pre-`final_norm`) -- via `register_forward_hook` (embeddings
    hook + one hook per layer).
  * (--sublayer only, E5) per layer i, the 6 module-boundary points HF's
    `ModernBertEncoderLayer.forward` (source read on this box, transformers
    5.15.1) actually has:
        attn_norm_out   = self.attn_norm(hidden_states)
        attn_out        = self.attn(attn_norm_out, ...)[0]   (post-Wo, pre-add)
        resid1_out      = hidden_states + attn_out            (mid-layer residual)
        mlp_norm_out    = self.mlp_norm(resid1_out)
        wi_out          = self.mlp.Wi(mlp_norm_out)            (pre-chunk, pre-act)
        geglu_out       = self.mlp.act(input) * gate            (post-activation, pre-Wo)
        wo_out          = self.mlp.Wo(geglu_out)
        resid2_out      = resid1_out + wo_out                  (== boundary.i)
    `attn_norm`/`mlp_norm`/`mlp.Wi`/`mlp.Wo` are real submodules -- hooked
    directly. `resid1_out`/`geglu_out`/`resid2_out` are bare tensor
    expressions inside `forward`, not submodule outputs -- captured by
    MONKEY-PATCHING `ModernBertEncoderLayer.forward` and `ModernBertMLP.forward`
    with a byte-for-byte copy of the read source (see `_patched_layer_forward`/
    `_patched_mlp_forward` below) that calls `tensor.register_hook(...)` on
    each intermediate to get its OWN gradient (a plain forward hook cannot
    do this for a non-return-value intermediate) and stashes the forward
    VALUE in the same sink. `resid2_out` duplicates `boundary.i` under a
    second key (`sublayer.{i}.resid2`) -- a cheap self-consistency check
    (the two capture MECHANISMS -- module hook vs monkeypatched-forward
    tensor hook -- must agree to the bit, or one of them is wrong).
  * (--attn-probs-layer N) the layer-N attention weights (softmax output,
    `eager_attention_forward`'s own `attn_weights` return, shape
    `(batch, heads, seq, seq)`) via `output_attentions=True` --
    ONLY valid with `--attn eager` (HF's SDPA/FA2 paths do not materialize
    attention weights; `main()` refuses `--attn sdpa` with this flag
    rather than silently returning `None`).

OUTPUT: a `safetensors` file. Every dumped tensor is `f32` (the D2H read
widens storage; see `grad_oracle.rs`'s own `GradOracleTensor` doc for the
same convention). A companion `<out>.json` carries the run's identity
fields (mirrors `grad_oracle.rs`'s `GradOracleReport` provenance shape) --
NOT the same schema, this is a different tool, but the same discipline:
config identity recorded, never assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_finetune_step as tfs  # noqa: E402
import torch_grad_oracle as tgo  # noqa: E402

MASK64 = (1 << 64) - 1
SM64_MUL = 6364136223846793005
SM64_INC = 1442695040888963407
SM64_SEED_XOR = 0x9E3779B97F4A7C15


def round4_synthetic_ids(batch: int, seq: int, vocab: int, seed: int):
    """Bit-identical port of `modernbert.rs`'s `tests::round4_synthetic_ids`
    (esc-045 round 4, `crates/jammi-encoders/src/modernbert.rs`). See this
    module's own docstring for why this must NOT be
    `torch_finetune_step.synthetic_ids`."""
    import torch

    state = (seed ^ SM64_SEED_XOR) & MASK64
    ids = []
    for _ in range(batch * seq):
        state = (state * SM64_MUL + SM64_INC) & MASK64
        ids.append((state >> 33) % vocab)
    return torch.tensor(ids, dtype=torch.long).reshape(batch, seq)


class Sink:
    def __init__(self):
        self.grads: dict[str, "torch.Tensor"] = {}
        self.forwards: dict[str, "torch.Tensor"] = {}

    def put_grad(self, key, t):
        assert key not in self.grads, f"duplicate grad key {key!r}"
        self.grads[key] = t.detach().float().cpu().contiguous()

    def put_forward(self, key, t):
        assert key not in self.forwards, f"duplicate forward key {key!r}"
        self.forwards[key] = t.detach().float().cpu().contiguous()


def install_boundary_hooks(model, sink: Sink):
    """`register_forward_hook` (value) + `register_full_backward_hook`
    (dL/d(output)) on the embeddings module and every
    `ModernBertEncoderLayer` -- gives `forward.{0..N}` and
    `boundary.{1..N}` exactly matching jammi's own numbering (see this
    module's docstring)."""
    handles = []
    embeddings = model.base_model.model.embeddings
    layers = model.base_model.model.layers

    def emb_fwd_hook(_module, _inputs, output):
        sink.put_forward("forward.0", output)

    handles.append(embeddings.register_forward_hook(emb_fwd_hook))

    for i, layer in enumerate(layers, start=1):

        def fwd_hook(_module, _inputs, output, _i=i):
            sink.put_forward(f"forward.{_i}", output)

        def bwd_hook(_module, _grad_input, grad_output, _i=i):
            sink.put_grad(f"boundary.{_i}", grad_output[0])

        handles.append(layer.register_forward_hook(fwd_hook))
        handles.append(layer.register_full_backward_hook(bwd_hook))
    return handles


def install_sublayer_hooks(model, sink: Sink):
    """E5: monkeypatches `ModernBertEncoderLayer.forward` and
    `ModernBertMLP.forward` on every layer instance (bound methods, via
    `types.MethodType` -- this repo's other only-in-tests instrumentation,
    jammi's own `activation_capture`, uses the analogous `#[cfg(test)]`
    tap-insertion pattern) to capture the 6 sub-layer boundaries that are
    bare tensor expressions, not submodule outputs. See this module's
    docstring for the exact point list and why a plain forward hook cannot
    reach them."""
    import types
    import torch.nn as nn

    handles = []
    layers = model.base_model.model.layers

    def make_layer_forward(idx):
        def patched(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
            attn_norm_out = self.attn_norm(hidden_states)
            sink.put_forward(f"sublayer.{idx}.attn_norm.fwd", attn_norm_out)
            if attn_norm_out.requires_grad:
                attn_norm_out.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.attn_norm", g))

            attn_output, _ = self.attn(
                attn_norm_out, position_embeddings=position_embeddings, attention_mask=attention_mask, **kwargs
            )
            sink.put_forward(f"sublayer.{idx}.attn_out.fwd", attn_output)
            if attn_output.requires_grad:
                attn_output.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.attn_out", g))

            resid1 = hidden_states + attn_output
            sink.put_forward(f"sublayer.{idx}.resid1.fwd", resid1)
            if resid1.requires_grad:
                resid1.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.resid1", g))

            mlp_norm_out = self.mlp_norm(resid1)
            sink.put_forward(f"sublayer.{idx}.mlp_norm.fwd", mlp_norm_out)
            if mlp_norm_out.requires_grad:
                mlp_norm_out.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.mlp_norm", g))

            mlp_out = self.mlp(mlp_norm_out)
            resid2 = resid1 + mlp_out
            sink.put_forward(f"sublayer.{idx}.resid2.fwd", resid2)
            if resid2.requires_grad:
                resid2.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.resid2", g))
            return resid2

        return patched

    def make_mlp_forward(idx):
        def patched(self, hidden_states):
            wi_out = self.Wi(hidden_states)
            sink.put_forward(f"sublayer.{idx}.wi.fwd", wi_out)
            if wi_out.requires_grad:
                wi_out.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.wi", g))

            input_, gate = wi_out.chunk(2, dim=-1)
            geglu_out = self.act(input_) * gate
            sink.put_forward(f"sublayer.{idx}.geglu.fwd", geglu_out)
            if geglu_out.requires_grad:
                geglu_out.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.geglu", g))

            wo_out = self.Wo(self.drop(geglu_out))
            sink.put_forward(f"sublayer.{idx}.wo.fwd", wo_out)
            if wo_out.requires_grad:
                wo_out.register_hook(lambda g, _i=idx: sink.put_grad(f"sublayer.{_i}.wo", g))
            return wo_out

        return patched

    for i, layer in enumerate(layers, start=1):
        layer.forward = types.MethodType(make_layer_forward(i), layer)
        layer.mlp.forward = types.MethodType(make_mlp_forward(i), layer.mlp)
    return handles


def main(argv=None):
    import torch
    from safetensors.torch import save_file

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--target-modules", default="Wqkv,Wo,Wi")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--attn", choices=["eager", "sdpa"], default="eager")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-weights-in", required=True)
    p.add_argument("--sublayer", action="store_true", help="E5: also install sub-layer hooks")
    p.add_argument("--attn-probs-layer", type=int, default=None, help="1-indexed layer to dump attn probs for (--attn eager only)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if args.attn_probs_layer is not None and args.attn != "eager":
        raise SystemExit("--attn-probs-layer requires --attn eager (sdpa/FA2 do not materialize attn_weights)")

    torch.manual_seed(args.seed)
    device = tfs.pick_device(args.cuda)
    fast_path_globals = tfs.pin_fast_path_globals()

    class _A:
        pass

    margs = _A()
    margs.model_dir = args.model_dir
    margs.dtype = args.dtype
    margs.attn = args.attn
    model, config = tfs.load_model(margs)
    resolved_attn = getattr(model.config, "_attn_implementation", "absent")

    largs = _A()
    largs.lora_rank = args.lora_rank
    largs.lora_alpha = args.lora_alpha
    largs.lora_dropout = 0.0
    largs.target_modules = args.target_modules
    model = tfs.wrap_lora(model, largs)
    model.to(device)
    model.train()

    trainable = [(n, p_) for n, p_ in model.named_parameters() if p_.requires_grad]
    written = tgo.load_lora_weights_into_model(model, args.lora_weights_in)
    if written != len(trainable):
        raise RuntimeError(f"loaded {written} of {len(trainable)} trainable LoRA tensors -- partial load")

    sink = Sink()
    handles = install_boundary_hooks(model, sink)
    if args.sublayer:
        install_sublayer_hooks(model, sink)

    input_ids = round4_synthetic_ids(args.batch, args.seq, config.vocab_size, args.seed).to(device)
    mask = torch.ones(args.batch, args.seq, dtype=torch.long, device=device)
    batch_token_id_sum = int(input_ids.sum().item())

    attn_kwargs = {}
    if args.attn_probs_layer is not None:
        attn_kwargs["output_attentions"] = True

    out = model.base_model.model(input_ids=input_ids, attention_mask=mask, **attn_kwargs)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    attn_probs_tensor = None
    if args.attn_probs_layer is not None:
        idx0 = args.attn_probs_layer - 1
        attentions = getattr(out, "attentions", None)
        if attentions is None or attentions[idx0] is None:
            raise RuntimeError(
                f"--attn-probs-layer {args.attn_probs_layer}: model did not return attn_weights "
                f"for that layer (attn_implementation={resolved_attn!r})"
            )
        attn_probs_tensor = attentions[idx0].detach().float().cpu().contiguous()

    loss = hidden.float().pow(2).sum()
    loss_val = float(loss.detach().float().item())
    assert loss_val == loss_val and loss_val not in (float("inf"), float("-inf")), "loss must be finite"
    loss.backward()

    for h in handles:
        h.remove()

    n_layers = len(model.base_model.model.layers)
    assert len(sink.forwards) >= n_layers + 1, f"expected >= {n_layers + 1} forward captures, got {len(sink.forwards)}"
    for i in range(1, n_layers + 1):
        assert f"boundary.{i}" in sink.grads, f"missing boundary.{i}"
    nonzero = any(bool((g != 0).any().item()) for g in sink.grads.values())
    assert nonzero, "every captured gradient is exactly zero -- backward() never reached the captured tensors"

    collide = set(sink.grads) & set(sink.forwards)
    assert not collide, f"grad/forward key collision (would silently overwrite): {sorted(collide)}"
    out_tensors = dict(sink.grads)
    out_tensors.update(sink.forwards)
    if attn_probs_tensor is not None:
        out_tensors[f"attn_probs.{args.attn_probs_layer}"] = attn_probs_tensor
    save_file(out_tensors, args.out)

    meta = {
        "tool": "torch_round6_layer_dump",
        **tgo.checkpoint_identity(args.model_dir),
        "provenance": tfs.provenance(device, fast_path_globals),
        "attn_requested": args.attn,
        "attn_implementation": resolved_attn,
        "backbone_dtype": tgo.translate_dtype_flag_to_jammi_spelling(args.dtype),
        "batch": args.batch,
        "seq": args.seq,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": [t.strip() for t in args.target_modules.split(",") if t.strip()],
        "seed": args.seed,
        "lora_weights_in": args.lora_weights_in,
        "batch_token_id_sum": batch_token_id_sum,
        "loss": loss_val,
        "sublayer": args.sublayer,
        "attn_probs_layer": args.attn_probs_layer,
        "n_layers": n_layers,
        "n_grad_tensors": len(sink.grads),
        "n_forward_tensors": len(sink.forwards),
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"torch_round6_layer_dump: wrote {len(out_tensors)} tensors to {args.out} "
        f"(loss={loss_val}, dtype={args.dtype}, attn={args.attn}->resolved={resolved_attn}, "
        f"batch={args.batch}, seq={args.seq})"
    )


if __name__ == "__main__":
    main()
