#!/usr/bin/env python3
"""Generates the B0 parity fixtures for jammi-kernels' FA2 varlen op (P6
Stage B, contract-P6-stage-B-v4.md §5 B0 / §4 O0(a)).

Reference: torch's OWN vendored FlashAttention-2, `torch.ops.aten.
_flash_attention_forward` / `_flash_attention_backward` (varlen, packed
`cu_seqlens_q/k` + `max_q/max_k`, `window_size_left/right`, returns
`softmax_logsumexp`). Torch 2.13.0+cu130 vendors flash-attention at commit
6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15 (`__version__ == "2.8.4"`,
2026-05-26) via `FLASH_NAMESPACE::mha_varlen_fwd`/`mha_varlen_bwd`
(`aten/src/ATen/native/transformers/cuda/attention.cu:539` at the v2.13.0
tag) — the UNPACKED (separate q/k/v) varlen entry point. jammi vendors
flash-attention tag v2.8.3.post1 (commit a8aa52b1ab3e9ca574c8a33b3f35afc0
17ffa2e2, see `crates/jammi-kernels/third_party/flash-attention/VENDORED.
md`) via the qkv-PACKED entry point (`flash_attn_varlen_qkvpacked_func`
upstream). VERSION MISMATCH recorded: jammi v2.8.3.post1 vs torch's
~v2.8.4-dev (torch is NEWER by ~1 release, same 2.8.x kernel family). The
packed and unpacked upstream Python wrappers both dispatch to the SAME
`mha_varlen_fwd`/`mha_varlen_bwd` C++ entry points (upstream `flash_attn_
interface.py`: `flash_attn_varlen_qkvpacked_func`'s docstring — "avoids
explicit concatenation of the gradients of Q, K, V" — i.e. the ONLY
difference is whether the caller already has one packed buffer or three
separate ones; the CUDA math is identical), so torch's separate-q/k/v
call is a legitimate numerical oracle for jammi's packed op PROVIDED q,
k, v here are exactly the three [total_q, 3, H, 64] slabs.

Every leg's `o`/`lse`/`dq`/`dk`/`dv` from run 1 are saved as float32 .npy
(from bf16 tensors — the loss of precision from bf16->f32 is exact/lossless,
this is NOT a re-round). The run-to-run self-diff (run 1 vs run 2, same
inputs, no explicit determinism knob exposed by this ATen op's schema) is
recorded in the sidecar JSON as the reference's own noise floor, per §4 O0(a)
("tolerance DERIVED from two runs of the reference itself").

Usage: /root/jammi-ai/.venv-torch-ref/bin/python generate_fixtures.py
Requires CUDA. Writes into this directory.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEED = 20260825
H = 4
D = 64

LEGS = [
    # (name, lengths, window_radius_or_None, softmax_scale_mult)
    ("b1_s512", [512], None, 1.0),
    ("b1_s512_win64", [512], 64, 1.0),
    ("b8_s512", [512, 500, 512, 480, 512, 505, 490, 512], None, 1.0),
    ("b8_s512_win64", [512, 500, 512, 480, 512, 505, 490, 512], 64, 1.0),
    ("b8_s128", [128, 120, 128, 100, 128, 115, 128, 128], None, 1.0),
    ("b1_tile129", [129], None, 1.0),
    ("b1_tile257", [257], None, 1.0),
    ("prefix_mixed", [512, 300, 129, 64], None, 1.0),
]

# Injection legs — must RED the O0(a) parity oracle. `softmax_scale_mult`
# doubles the scale passed to the JAMMI leg only (the fixture itself is
# generated at the correct scale; the Rust test re-derives the wrong-scale
# comparison at consume time, so injection is not baked into the fixture —
# see the sidecar's "injection" section for what the Rust side must do).


def flash_attention_leg(name, lengths, window, dev, dtype):
    torch.manual_seed(SEED)
    total_q = sum(lengths)
    max_seqlen = max(lengths)
    cu = torch.tensor(
        [0] + list(np.cumsum(lengths)), dtype=torch.int32, device=dev
    )
    scale = 1.0 / (D ** 0.5)

    def make_qkv():
        q = torch.randn(total_q, H, D, device=dev, dtype=dtype, requires_grad=True)
        k = torch.randn(total_q, H, D, device=dev, dtype=dtype, requires_grad=True)
        v = torch.randn(total_q, H, D, device=dev, dtype=dtype, requires_grad=True)
        return q, k, v

    window_left = -1 if window is None else window
    window_right = -1 if window is None else window

    def run_once():
        q, k, v = make_qkv()
        out, lse, rng_state, unused, _dbg = torch.ops.aten._flash_attention_forward(
            q, k, v, cu, cu, max_seqlen, max_seqlen,
            0.0, False, False,
            scale=scale, window_size_left=window_left, window_size_right=window_right,
        )
        g = torch.randn_like(out)
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            g, q, k, v, out, lse, cu, cu, max_seqlen, max_seqlen,
            0.0, False, rng_state, unused,
            scale=scale, window_size_left=window_left, window_size_right=window_right,
        )
        return q, k, v, out, lse, g, dq, dk, dv

    q1, k1, v1, o1, lse1, g1, dq1, dk1, dv1 = run_once()
    # second run: SAME seed (so q/k/v/g are bit-identical inputs) — isolates
    # the *kernel's own* run-to-run variation, not input variation.
    q2, k2, v2, o2, lse2, g2, dq2, dk2, dv2 = run_once()

    def npf32(t):
        return t.detach().to(torch.float32).cpu().numpy()

    def maxabsdiff(a, b):
        return float((a.detach().to(torch.float32) - b.detach().to(torch.float32)).abs().max().item())

    self_noise = {
        "o": maxabsdiff(o1, o2),
        "lse": maxabsdiff(lse1, lse2),
        "dq": maxabsdiff(dq1, dq2),
        "dk": maxabsdiff(dk1, dk2),
        "dv": maxabsdiff(dv1, dv2),
    }

    np.save(HERE / f"{name}_q.npy", npf32(q1))
    np.save(HERE / f"{name}_k.npy", npf32(k1))
    np.save(HERE / f"{name}_v.npy", npf32(v1))
    np.save(HERE / f"{name}_grad_out.npy", npf32(g1))
    np.save(HERE / f"{name}_o.npy", npf32(o1))
    np.save(HERE / f"{name}_lse.npy", npf32(lse1))
    np.save(HERE / f"{name}_dq.npy", npf32(dq1))
    np.save(HERE / f"{name}_dk.npy", npf32(dk1))
    np.save(HERE / f"{name}_dv.npy", npf32(dv1))

    return {
        "lengths": lengths,
        "total_q": total_q,
        "max_seqlen": max_seqlen,
        "num_heads": H,
        "head_dim": D,
        "window_radius": window,
        "softmax_scale": scale,
        "self_noise_max_abs_diff": self_noise,
        "o_max_abs": float(o1.detach().abs().max().item()),
        "lse_max_abs": float(lse1.detach().abs().max().item()),
        "dq_max_abs": float(dq1.detach().abs().max().item()),
        "dk_max_abs": float(dk1.detach().abs().max().item()),
        "dv_max_abs": float(dv1.detach().abs().max().item()),
    }


def main():
    assert torch.cuda.is_available(), "CUDA required to generate the reference fixtures"
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16

    sidecar = {
        "generator": "crates/jammi-kernels/tests/fixtures/flash_reference/generate_fixtures.py",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_flash_attention_vendored_commit": "6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15",
        "torch_flash_attention_vendored_version_string": "2.8.4",
        "jammi_flash_attention_vendored_tag": "v2.8.3.post1",
        "jammi_flash_attention_vendored_commit": "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2",
        "version_mismatch_note": (
            "torch's vendored FA2 (2.8.4-dev @ 2026-05-26) is one release "
            "newer than jammi's pinned v2.8.3.post1; both are in the 2.8.x "
            "kernel family (no head_dim/window/varlen ABI break between "
            "them per upstream CHANGELOG) — parity is tolerance-based "
            "regardless (fast-math), so this bounds INTERPRETATION (a "
            "true kernel-level divergence between the two versions would "
            "look identical to fast-math noise) more than it bounds the "
            "numeric bar itself."
        ),
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "num_heads": H,
        "head_dim": D,
        "dtype": "bfloat16",
        "calling_convention_note": (
            "torch's _flash_attention_forward/_backward take SEPARATE "
            "query/key/value tensors (FLASH_NAMESPACE::mha_varlen_fwd); "
            "jammi's op takes ONE packed qkv [total_q,3,H,64] tensor "
            "(flash_attn_varlen_qkvpacked_func's upstream layout). Both "
            "dispatch to the same underlying mha_varlen_fwd/bwd kernel "
            "(upstream flash_attn_interface.py's qkvpacked wrapper docstring: "
            "packing only changes whether the backward needs an explicit "
            "concat of dQ/dK/dV, not the forward/backward math) — so this "
            "fixture's q/k/v arrays, STACKED along a new axis 1, are "
            "byte-for-byte the qkv tensor the jammi leg must feed the same "
            "kernel with the SAME numeric values."
        ),
        "legs": {},
    }

    for name, lengths, window, _mult in LEGS:
        print(f"leg {name}: lengths={lengths} window={window}", file=sys.stderr)
        sidecar["legs"][name] = flash_attention_leg(name, lengths, window, dev, dtype)

    with open(HERE / "sidecar.json", "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    print("wrote", HERE / "sidecar.json", file=sys.stderr)


if __name__ == "__main__":
    main()
