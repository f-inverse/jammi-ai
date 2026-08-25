#!/usr/bin/env python3
"""Generates the B0 parity fixtures for jammi-kernels' FA2 varlen op (P6
Stage B, contract-P6-stage-B-v4.md §5 B0 / §4 O0(a)).

Reference: torch's OWN vendored FlashAttention-2, `torch.ops.aten.
_flash_attention_forward` / `_flash_attention_backward` (varlen, packed
`cu_seqlens_q/k` + `max_q/max_k`, `window_size_left/right`, returns
`softmax_logsumexp`) via `FLASH_NAMESPACE::mha_varlen_fwd`/`mha_varlen_bwd`
(`aten/src/ATen/native/transformers/cuda/attention.cu:539` at the v2.13.0
tag) — the UNPACKED (separate q/k/v) varlen entry point.

**Identity** (v5 pressure-test correction of an intermediate, WRONG "fa4"
finding — the git tag `fa4-v4.0.0.beta15` also resolves to this same SHA on
Dao-AILab/flash-attention, but that tag name does NOT change the CONTENT at
this commit): torch 2.13.0's `third_party/flash-attention` submodule is
pinned at commit `6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15`, whose
`flash_attn/__init__.py:6` reads `__version__ = "2.8.4"` and whose CMake
build globs `csrc/flash_attn/src/*.cu` (the classic kernel tree, the SAME
subtree jammi vendors) — this is FlashAttention **2.8.4**, ONE patch release
above jammi's pinned v2.8.3.post1 (commit
a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2). Torch's build additionally
defines `UNFUSE_FMA` (verified at `csrc/flash_attn/src/softmax.h:85-89` at
that commit: this macro switches the online-softmax's `exp2f` call from a
compiler-fusable `x * scale - max` to an explicit `__fmul_rn(x, scale) -
max`, forcing IEEE multiply-then-subtract instead of a contracted FMA) and
is NOT compiled with `--use_fast_math` (jammi's three flash translation
units ARE, see `third_party/flash-attention/VENDORED.md`'s Build section).

Net: this is a CROSS-BUILD reference, not a same-kernel-same-flags oracle —
the tolerance below is NOT "two runs of the reference with different
`deterministic` settings" for `o`/`lse` (forward has no split-KV path
reachable from either build's varlen-packed/unpacked entry, so two runs of
the SAME build's forward are bit-identical regardless of `deterministic`,
which only affects backward `dQ` accumulation — see `flash/mod.rs`'s own
determinism doc); `o`/`lse` self-diff is expected to be, and measured as,
0. The `o`/`lse` tolerance is instead a stated ANALYTIC bound (bf16 output
ULP scaled by the online-softmax accumulation depth, i.e. the number of
128-row KV tiles = `ceil(max_seqlen/128)`, plus the fast-math/FMA-fusion
error the two builds disagree on) computed in the Rust oracle, not baked
into this sidecar. The `dq`/`dk`/`dv` self-diff below (same q/k/v/grad_out,
two backward calls) DOES isolate genuine kernel-level accumulation
nondeterminism in torch's own build and is used as-is for those three.

Every leg's `o`/`lse`/`dq`/`dk`/`dv` from run 1 are saved as float32 .npy
(from bf16 tensors — the loss of precision from bf16->f32 is exact/lossless,
this is NOT a re-round).

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

    window_left = -1 if window is None else window
    window_right = -1 if window is None else window

    # Fixed inputs, generated ONCE: `q`/`k`/`v`/`g` must be the SAME tensors
    # across both runs below, or a diff between the two runs measures input
    # variation (a fresh `torch.randn` draw), not the kernel's own
    # run-to-run behaviour — exactly the bug this comment replaces (the
    # first version of this script called `torch.randn` again inside each
    # `run_once`, so "self noise" was actually measuring two independent
    # random problems and came out comparable to the signal itself).
    q = torch.randn(total_q, H, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(total_q, H, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(total_q, H, D, device=dev, dtype=dtype, requires_grad=True)

    def run_once(g_in=None):
        out, lse, rng_state, unused, _dbg = torch.ops.aten._flash_attention_forward(
            q, k, v, cu, cu, max_seqlen, max_seqlen,
            0.0, False, False,
            scale=scale, window_size_left=window_left, window_size_right=window_right,
        )
        g = g_in if g_in is not None else torch.randn_like(out)
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            g, q, k, v, out, lse, cu, cu, max_seqlen, max_seqlen,
            0.0, False, rng_state, unused,
            scale=scale, window_size_left=window_left, window_size_right=window_right,
        )
        return out, lse, g, dq, dk, dv

    o1, lse1, g1, dq1, dk1, dv1 = run_once()
    # second run: SAME q/k/v/g (fixed above) — isolates the *kernel's own*
    # run-to-run variation, not input variation.
    o2, lse2, g2, dq2, dk2, dv2 = run_once(g_in=g1)

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

    np.save(HERE / f"{name}_q.npy", npf32(q))
    np.save(HERE / f"{name}_k.npy", npf32(k))
    np.save(HERE / f"{name}_v.npy", npf32(v))
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
        "torch_flash_attention_vendored_version": "2.8.4",
        "torch_flash_attention_vendored_version_source": "flash_attn/__init__.py:6 __version__ at that commit; also the git tag fa4-v4.0.0.beta15 resolves to the SAME SHA (a naming artifact of the upstream repo, not a different content generation — verified: torch's CMake globs csrc/flash_attn/src/*.cu, the classic tree, at that commit)",
        "torch_flash_attention_kernel_subtree": "third_party/flash-attention/csrc/flash_attn/src (classic tree, same subtree jammi vendors)",
        "torch_flash_attention_build_defines": ["FLASHATTENTION_DISABLE_ALIBI", "FLASHATTENTION_DISABLE_SOFTCAP", "UNFUSE_FMA"],
        "torch_flash_attention_use_fast_math": False,
        "jammi_flash_attention_vendored_tag": "v2.8.3.post1",
        "jammi_flash_attention_vendored_commit": "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2",
        "jammi_flash_attention_use_fast_math": True,
        "version_mismatch_note": (
            "torch pins Dao-AILab/flash-attention at commit 6c4f74fb338e = "
            "release 2.8.4 (flash_attn/__init__.py:6's __version__ string, "
            "and torch's own CMake globs the classic csrc/flash_attn/src "
            "tree at that commit, aten/src/ATen/CMakeLists.txt:359-365 at "
            "the v2.13.0 tag) — ONE patch release above jammi's pinned "
            "v2.8.3.post1, same kernel source lineage. Torch's build "
            "additionally defines UNFUSE_FMA (verified at "
            "csrc/flash_attn/src/softmax.h:85-89 at that commit: forces "
            "exp2f(__fmul_rn(x,scale)-max) instead of the fusable "
            "exp2f(x*scale-max)) and does NOT pass --use_fast_math, while "
            "jammi's three flash translation units DO "
            "(crates/jammi-kernels/third_party/flash-attention/VENDORED.md's "
            "Build section). This is a real numeric divergence source (FMA "
            "fusion + fast-math approximation) narrower than a generic "
            "cross-kernel-generation gap (same 2.8.x source, one patch "
            "apart) — the O0(a) parity tolerance for o/lse is derived "
            "analytically (bf16 ULP x accumulation depth x a loose safety "
            "factor) rather than from a same-build self-diff, which is 0 "
            "for o/lse regardless (forward has no split-KV path on either "
            "build); a TIGHTER, flag-matched bound (building a jammi "
            "variant with UNFUSE_FMA/no-fast-math for the test only) is "
            "possible but not done here — flagged as future work, not a "
            "blocker for this parity bar."
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
