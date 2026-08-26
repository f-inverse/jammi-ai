#!/usr/bin/env python3
"""esc-045 (GH#374) round 5 -- generate a torch-produced fixture for
`jammi-kernels/tests/softmax_bwd_dscores_matches_reference_call_path.rs`.

Reproduces `modeling_modernbert.py:180`'s exact call:
    nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
with attn_weights already BF16 (ModernBERT's training dtype), then computes
the backward w.r.t. attn_weights (the leaf) under a random, production-
amplitude, deliberately-skewed `dy` seed -- the SAME shape/amplitude/skew
convention this test's own Rust-side fixture already uses (LAST=512, 12
rows, AMPLITUDES 0.5..300, dy_amp in [0.01, 5.0) cubed within the row).

Saves x (bf16), dy (bf16), dx_torch (bf16, torch's real answer), y_torch
(bf16, torch's real BF16-rounded probabilities -- reference/debugging
only) and y_f32_true (F32, the UNROUNDED softmax probabilities
`_softmax_backward_data`'s `output` argument actually binds to) as a
safetensors file (`candle_core::safetensors::load` reads this with ZERO
new Cargo dependencies -- this project's leaf crates already depend on
candle_core, which already depends on the safetensors crate) plus a JSON
sidecar carrying full generation provenance.

Run with this project's pinned torch/transformers venv, e.g. on the
gpu-dev pod:
    .venv-torch-ref/bin/python3 crates/jammi-kernels/tests/fixtures/gen_softmax_bwd_torch_dx.py
"""
import json
import os
import struct

import torch
import torch.nn.functional as F

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

LAST = 512
AMPLITUDES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 96.0, 150.0, 220.0, 300.0]
ROWS = len(AMPLITUDES)

g = torch.Generator().manual_seed(20260825)  # CPU generator -- deterministic regardless of DEVICE

x_rows = []
dy_rows = []
for amp in AMPLITUDES:
    x_rows.append((torch.rand(LAST, generator=g) * 2.0 - 1.0) * amp)
    dy_amp = 0.01 + 4.99 * torch.rand(1, generator=g).item()
    u = torch.rand(LAST, generator=g) * 2.0 - 1.0
    dy_rows.append(dy_amp * u * u.abs())

x_f32 = torch.stack(x_rows, dim=0)
dy_f32 = torch.stack(dy_rows, dim=0)

x_bf16 = x_f32.to(torch.bfloat16).to(DEVICE)
dy_bf16 = dy_f32.to(torch.bfloat16).to(DEVICE)

x_var = x_bf16.clone().detach().requires_grad_(True)
# EXACTLY modeling_modernbert.py:180's call path -- kept as TWO separate
# steps (not one chained expression) so `softmax_f32` (the UNROUNDED f32
# intermediate _softmax(converted, dim, False) actually produces, and what
# `_softmax_backward_data`'s `output` argument actually binds to) stays a
# nameable, retainable non-leaf tensor rather than disappearing into an
# expression torch's autograd would otherwise free after backward.
softmax_f32 = F.softmax(x_var, dim=-1, dtype=torch.float32)
softmax_f32.retain_grad()
y = softmax_f32.to(x_var.dtype)
loss = (y * dy_bf16).sum()
loss.backward()
dx_torch = x_var.grad.detach().clone().to("cpu")
y_f32_true = softmax_f32.detach().clone().to("cpu")  # UNROUNDED -- the real `output` arg
x_bf16 = x_bf16.to("cpu")
dy_bf16 = dy_bf16.to("cpu")
y = y.to("cpu")

assert x_var.grad.dtype == torch.bfloat16
assert y.dtype == torch.bfloat16
assert y_f32_true.dtype == torch.float32

print("x_bf16", x_bf16.shape, x_bf16.dtype)
print("dy_bf16", dy_bf16.shape, dy_bf16.dtype)
print("dx_torch", dx_torch.shape, dx_torch.dtype)
print("y_bf16", y.shape, y.dtype)
nonzero = (dx_torch.float() != 0).sum().item()
print("dx_torch nonzero elements:", nonzero, "/", dx_torch.numel())


def save_safetensors_mixed(path, tensors: dict):
    """Hand-rolled safetensors writer for BF16/F32 tensors -- avoids adding
    the `safetensors` python package as a hard requirement beyond what
    this pod's `.venv-torch-ref` already has; format per
    https://github.com/huggingface/safetensors (8-byte LE header length,
    then a UTF-8 JSON header, then raw contiguous tensor bytes)."""
    header = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        t = t.contiguous()
        if t.dtype == torch.bfloat16:
            st_dtype = "BF16"
            raw = t.view(torch.uint16).numpy().tobytes()
        elif t.dtype == torch.float32:
            st_dtype = "F32"
            raw = t.numpy().tobytes()
        else:
            raise ValueError(f"unsupported dtype {t.dtype} for {name!r}")
        header[name] = {
            "dtype": st_dtype,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    header_json = json.dumps(header, sort_keys=True).encode("utf-8")
    # Pad header to 8-byte alignment per spec (optional but conventional).
    pad = (-len(header_json)) % 8
    header_json += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for b in blobs:
            f.write(b)


out_dir = os.path.dirname(os.path.abspath(__file__))
st_path = os.path.join(out_dir, "softmax_bwd_torch_dx.safetensors")
save_safetensors_mixed(
    st_path,
    {
        "x": x_bf16,
        "dy": dy_bf16,
        "dx_torch": dx_torch,
        "y_torch": y.detach(),
        "y_f32_true": y_f32_true,
    },
)
print("wrote", st_path, os.path.getsize(st_path), "bytes")

sidecar = {
    "schema_version": 1,
    "compute_device": DEVICE,
    "generator": "crates/jammi-kernels/tests/fixtures/gen_softmax_bwd_torch_dx.py (esc-045 round 5, GH#374); run with this project's pinned torch/transformers venv, e.g. `.venv-torch-ref/bin/python3 gen_softmax_bwd_torch_dx.py`",
    "purpose": (
        "Independent torch-produced ground truth for "
        "jammi-kernels/tests/softmax_bwd_dscores_matches_reference_call_path.rs "
        "-- torch's REAL dx from modeling_modernbert.py:180's exact call path "
        "(nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)"
        ".to(query.dtype)), not a hand-rederived formula."
    ),
    "torch_version": torch.__version__,
    "transformers_version_pinned_reference": "5.15.1",
    "shape": [ROWS, LAST],
    "amplitudes": AMPLITUDES,
    "dtype": "bfloat16",
    "seed_generator_manual_seed": 20260825,
    "torch_manual_seed": 0,
    "dx_torch_nonzero_elements": nonzero,
    "dx_torch_total_elements": dx_torch.numel(),
    "fields": {
        "x": "raw attn_weights (bf16), the leaf tensor softmax runs on",
        "dy": "upstream gradient (bf16), deliberately non-uniform/skewed per row",
        "dx_torch": "torch's real d(loss)/d(x) (bf16) -- the independent ground truth",
        "y_torch": "torch's real softmax(x) (bf16, downcast from the f32 upcast) -- for reference/debugging only, NOT what softmax_backward_data's own `output` argument binds to",
        "y_f32_true": "the UNROUNDED f32 intermediate _softmax(x.float(), dim, False) actually produces -- what softmax_backward_data's `output` argument really binds to (retain_grad()-ed before the .to(bf16) cast, never rounded); this is the correct 'same f32 y' to feed an isolated backward-formula check against dx_torch",
    },
}
sidecar_path = st_path + ".json"
with open(sidecar_path, "w") as f:
    json.dump(sidecar, f, indent=2, sort_keys=True)
    f.write("\n")
print("wrote", sidecar_path)
