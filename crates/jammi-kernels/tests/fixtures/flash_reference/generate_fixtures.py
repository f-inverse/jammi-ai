#!/usr/bin/env python3
"""Generates the B0 parity fixtures for jammi-kernels' FA2 varlen op (P6
Stage B, contract-P6-stage-B-v4.md §5 B0 / §4 O0(a)).

Reference: torch's OWN vendored FlashAttention-2, `torch.ops.aten.
_flash_attention_forward` / `_flash_attention_backward` (varlen, packed
`cu_seqlens_q/k` + `max_q/max_k`, `window_size_left/right`, returns
`softmax_logsumexp`) via `FLASH_NAMESPACE::mha_varlen_fwd`/`mha_varlen_bwd`
(`aten/src/ATen/native/transformers/cuda/attention.cu:539` at the v2.13.0
tag) — the UNPACKED (separate q/k/v) varlen entry point. See this
directory's `sidecar.json`'s `version_mismatch_note` for the reference's
exact identity (torch's vendored FA2 build is one patch release above
jammi's, defines `UNFUSE_FMA`, skips `--use_fast_math`).

# Fix round (`10b1f3b` audit, BLOCKING findings 1 and 5)

Two changes from the prior version:

1. **A TRUTH tensor per leg per output** (`o`/`lse`/`dq`/`dk`/`dv`),
   computed by a from-scratch, non-tiled eager attention implementation in
   **float64** (`truth_attention_fwd`/`truth_attention_bwd` below — plain
   `einsum`/`softmax`, no FlashAttention kernel of any generation involved)
   from the SAME bf16-exact inputs both `o1` (jammi) and this script's own
   `ref` (torch's FA2) consume. This is the higher-precision anchor
   `docs/maintainer/cuda-kernel-guide.md` §3.3 requires ("Agreement is not
   accuracy... anchor with a higher-precision reference"); the auditor
   computed one by hand with numpy for `b1_s512_win64` and found
   `|truth - ref_o|max = 0.0036`, `|truth - ref_lse|max = 7.7e-7` — the
   consuming Rust oracle (`tests/flash_torch_parity.rs`) now asserts
   `jammi`'s own distance to this SAME truth is bounded by a small multiple
   of torch FA2's distance to it, rather than an absolute ULP-derived
   constant (which the auditor showed a 5% `softmax_scale` injection
   PASSED through: 50-100x wider than the real divergence).
2. **Production shape/amplitude**: `H = 16` (was 4) and inputs are drawn
   with a spread matched to the decisive-timing harness's own convention
   (`tests/flash_decisive_timing.rs`'s `random_bf16`: production ModernBERT
   post-LN activations run roughly `[-18, 18]`), not `torch.randn`'s
   unclamped unit-variance draw.
3. **Storage**: `q`/`k`/`v`/`grad_out` (the bf16-exact INPUTS) are saved as
   raw `int16` `.npy` arrays — `tensor.view(torch.int16)`, a BIT-FOR-BIT
   reinterpretation of the bf16 payload, not a re-round — because candle's
   own `npy.rs` `Header::parse` (candle-core 0.11.0) has no numpy `descr`
   string mapped to `DType::BF16` at all (only `Tensor::from_reader`'s
   dtype-dispatch match arm exists; nothing ever constructs a `Header` with
   `descr: DType::BF16` via parsing), so a numpy file claiming a bf16 dtype
   is not readable through `Tensor::read_npy` regardless of what wrote it —
   this is a `no-hard-candle-dependency` constraint (memory), not something
   to work around by patching candle. `int16` is a supported `descr` on
   both sides (numpy `'<i2'` <-> candle `DType::I16`) and the SAME 2 bytes
   per element as bf16, so this halves input storage vs f32 with zero
   precision loss and no candle change. The Rust side bit-reinterprets
   `i16 -> u16 -> bf16::from_bits` (see `flash_torch_parity.rs`'s
   `load_bf16_exact`). Reference (`o`/`lse`/`dq`/`dk`/`dv`) and TRUTH
   outputs are saved as `float32` (never `float64` on disk — the f64
   TRUTH's precision matters only in the arithmetic that PRODUCES it; once
   produced, f32 already exceeds bf16's own precision by 16 mantissa bits
   and halves storage again vs f64).

Usage: /root/jammi-ai/.venv-torch-ref/bin/python generate_fixtures.py [--dtype bfloat16|float16]
Requires CUDA. Writes into this directory.

# fp16 twin (campaign #443, D2/D3)

`--dtype float16` generates the SAME `LEGS` sweep, at the SAME production
amplitude/spread and the SAME from-scratch f64 TRUTH derivation, for the fp16
twin of this op (`crate::flash::flash_varlen_{fwd,bwd}_f16`) — every file this
produces is prefixed `f16_` (`f16_{name}_q.npy`, ..., sidecar written to
`sidecar_f16.json`) so it NEVER collides with or overwrites the bf16 fixtures
above; the bf16 invocation (no `--dtype`, or `--dtype bfloat16`) is
byte-for-byte unchanged from before this parametrisation. Storage differs from
the bf16 legs' own `int16`-bit-pattern workaround: numpy has a NATIVE
`float16` dtype (`'<f2'`), and candle-core 0.11.0's `npy.rs` `Header::parse`
DOES map `"f2"`/`"e"` to `DType::F16` (unlike bf16, which has no descr mapping
at all — see this file's own "Storage" section above) — so fp16 inputs are
saved as plain `.astype(np.float16)` arrays, no bit-reinterpretation needed,
and `Tensor::read_npy` on the Rust side loads them directly as `DType::F16`.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SEED = 20260825
H = 16
D = 64
# Production amplitude: ModernBERT's typical post-LN activation range is
# roughly [-18, 18] (`tests/flash_decisive_timing.rs`'s own convention,
# `docs/maintainer/cuda-kernel-guide.md` §3.4's "test at production
# amplitude"). A scaled-and-clamped normal draw puts real mass near
# saturation rather than torch.randn's unclamped unit-variance draw (which
# every parity leg the 10b1f3b audit reproduced used, and which the guide
# calls out by name as "decoration" when production amplitude is far wider).
AMPLITUDE = 18.0
SPREAD = 9.0

LEGS = [
    # (name, lengths, window_radius_or_None)
    ("b1_s512", [512], None),
    ("b1_s512_win64", [512], 64),
    ("b8_s512", [512, 500, 512, 480, 512, 505, 490, 512], None),
    ("b8_s512_win64", [512, 500, 512, 480, 512, 505, 490, 512], 64),
    ("b8_s128", [128, 120, 128, 100, 128, 115, 128, 128], None),
    ("b1_tile129", [129], None),
    ("b1_tile257", [257], None),
    ("prefix_mixed", [512, 300, 129, 64], None),
]


def make_input(shape, dev, dtype):
    """Production-amplitude fill: `randn * SPREAD`, clamped to
    `[-AMPLITUDE, AMPLITUDE]`, then rounded to `dtype` (bf16) — the ROUNDING
    is what makes the input bf16-EXACT (every later f64/f32 read of this
    same tensor is lossless, since bf16 -> wider is always exact)."""
    x = torch.randn(shape, device=dev, dtype=torch.float32) * SPREAD
    x = x.clamp(-AMPLITUDE, AMPLITUDE)
    return x.to(dtype)


def truth_attention_fwd(q64, k64, v64, lengths, window, scale):
    """From-scratch f64 eager attention (NOT FlashAttention, any
    generation) — the higher-precision anchor
    (`docs/maintainer/cuda-kernel-guide.md` §3.3). Non-causal (matches the
    op's own domain: `is_causal=False` is passed to the aten call below),
    symmetric sliding window `|i - j| <= window` when `window is not None`
    (matches `crate::flash::VarlenConfig::window_sizes`' symmetric-pair
    convention). Returns `(o64 [total_q,H,D], lse64 [H,total_q])`."""
    total_q, Hh, Dd = q64.shape
    o = torch.zeros_like(q64)
    lse = torch.zeros(Hh, total_q, dtype=torch.float64, device=q64.device)
    start = 0
    for length in lengths:
        end = start + length
        qs, ks, vs = q64[start:end], k64[start:end], v64[start:end]
        scores = torch.einsum("ihd,jhd->hij", qs, ks) * scale
        if window is not None:
            idx = torch.arange(length, device=q64.device)
            dist = (idx[:, None] - idx[None, :]).abs()
            mask = dist > window
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        m = scores.max(dim=-1, keepdim=True).values
        exp_scores = (scores - m).exp()
        denom = exp_scores.sum(dim=-1, keepdim=True)
        probs = exp_scores / denom
        o[start:end] = torch.einsum("hij,jhd->ihd", probs, vs)
        lse[:, start:end] = (m.squeeze(-1) + denom.squeeze(-1).log())
        start = end
    return o, lse


def truth_attention_bwd(q64, k64, v64, lse64, g64, lengths, window, scale):
    """Standard softmax-attention backward (Tri Dao et al., FlashAttention
    paper eq. 4): `dS = P * (dP - rowsum(dP * P))`, recomputing `P` from the
    SAME `lse64` the forward produced (never re-deriving max/denom
    independently — this is the identical mathematical object, not an
    approximation of it). Returns `(dq64, dk64, dv64)`, each matching its
    input's shape."""
    total_q, Hh, Dd = q64.shape
    dq = torch.zeros_like(q64)
    dk = torch.zeros_like(k64)
    dv = torch.zeros_like(v64)
    start = 0
    for length in lengths:
        end = start + length
        qs, ks, vs, gs = q64[start:end], k64[start:end], v64[start:end], g64[start:end]
        scores = torch.einsum("ihd,jhd->hij", qs, ks) * scale
        if window is not None:
            idx = torch.arange(length, device=q64.device)
            dist = (idx[:, None] - idx[None, :]).abs()
            mask = dist > window
            scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        lse_local = lse64[:, start:end]
        probs = (scores - lse_local.unsqueeze(-1)).exp()
        dv[start:end] = torch.einsum("hij,ihd->jhd", probs, gs)
        dp = torch.einsum("ihd,jhd->hij", gs, vs)
        rowsum = (dp * probs).sum(dim=-1, keepdim=True)
        ds = probs * (dp - rowsum)
        dq[start:end] = torch.einsum("hij,jhd->ihd", ds, ks) * scale
        dk[start:end] = torch.einsum("hij,ihd->jhd", ds, qs) * scale
        start = end
    return dq, dk, dv


def flash_attention_leg(name, lengths, window, dev, dtype, file_prefix=""):
    """`file_prefix` (empty for the bf16 legs, `"f16_"` for the fp16 twin —
    see this file's module doc's "fp16 twin" section) is the ONLY thing that
    differs about the on-disk layout between the two dtypes; every other
    line of this function is dtype-PARAMETRIC already (`make_input`,
    `torch.ops.aten._flash_attention_forward/_backward` are dtype-generic on
    their `q`/`k`/`v` argument's own dtype)."""
    torch.manual_seed(SEED)
    total_q = sum(lengths)
    max_seqlen = max(lengths)
    cu = torch.tensor(
        [0] + list(np.cumsum(lengths)), dtype=torch.int32, device=dev
    )
    scale = 1.0 / (D ** 0.5)

    window_left = -1 if window is None else window
    window_right = -1 if window is None else window

    # Fixed inputs, generated ONCE at production amplitude (see
    # `make_input`) — `q`/`k`/`v`/`g` must be the SAME tensors across both
    # runs below, or a diff between the two runs measures input variation,
    # not the kernel's own run-to-run behaviour.
    q = make_input((total_q, H, D), dev, dtype)
    k = make_input((total_q, H, D), dev, dtype)
    v = make_input((total_q, H, D), dev, dtype)

    def run_once(g_in=None):
        out, lse, rng_state, unused, _dbg = torch.ops.aten._flash_attention_forward(
            q, k, v, cu, cu, max_seqlen, max_seqlen,
            0.0, False, False,
            scale=scale, window_size_left=window_left, window_size_right=window_right,
        )
        g = g_in if g_in is not None else make_input(out.shape, dev, dtype)
        dq, dk, dv = torch.ops.aten._flash_attention_backward(
            g, q, k, v, out, lse, cu, cu, max_seqlen, max_seqlen,
            0.0, False, rng_state, unused,
            scale=scale, window_size_left=window_left, window_size_right=window_right,
        )
        return out, lse, g, dq, dk, dv

    o1, lse1, g1, dq1, dk1, dv1 = run_once()
    # second run: SAME q/k/v/g (fixed above) — isolates the *kernel's own*
    # run-to-run variation, not input variation. REPORTED, not used as a
    # bound (see `flash_torch_parity.rs`'s module doc).
    o2, lse2, g2, dq2, dk2, dv2 = run_once(g_in=g1)

    def npf32(t):
        return t.detach().to(torch.float32).cpu().numpy()

    def npi16_bf16_bits(t):
        # Bit-for-bit reinterpretation of the bf16 payload as int16 — see
        # this file's module doc's "Storage" section for why (not a
        # re-round; candle's npy reader has no bf16 descr mapping at all).
        assert t.dtype == torch.bfloat16
        return t.detach().contiguous().view(torch.int16).cpu().numpy()

    def npf16_native(t):
        # Native numpy float16 — see this file's module doc's "fp16 twin"
        # section: unlike bf16, numpy DOES have a real float16 dtype and
        # candle's npy reader DOES map it (`"f2"`/`"e"` -> `DType::F16`), so
        # no bit-pattern workaround is needed here; this is a plain,
        # lossless dtype-preserving cast (the values are already fp16-exact
        # from `make_input`).
        assert t.dtype == torch.float16
        return t.detach().contiguous().to(torch.float16).cpu().numpy()

    def npin(t):
        # Dispatches to whichever of the two input-storage conventions
        # matches `dtype` — the ONE place this function's own dtype
        # parametrisation touches serialisation.
        return npf16_native(t) if dtype == torch.float16 else npi16_bf16_bits(t)

    def maxabsdiff(a, b):
        return float((a.detach().to(torch.float32) - b.detach().to(torch.float32)).abs().max().item())

    self_noise = {
        "o": maxabsdiff(o1, o2),
        "lse": maxabsdiff(lse1, lse2),
        "dq": maxabsdiff(dq1, dq2),
        "dk": maxabsdiff(dk1, dk2),
        "dv": maxabsdiff(dv1, dv2),
    }

    # --- TRUTH: from-scratch f64 eager attention on the SAME dtype-exact
    # inputs (upcast bf16/fp16 -> f64 is exact/lossless — the values were
    # already rounded to `dtype` by `make_input`).
    q64, k64, v64, g64 = (t.detach().to(torch.float64) for t in (q, k, v, g1))
    truth_o, truth_lse = truth_attention_fwd(q64, k64, v64, lengths, window, scale)
    truth_dq, truth_dk, truth_dv = truth_attention_bwd(
        q64, k64, v64, truth_lse, g64, lengths, window, scale
    )

    # --- Write: dtype-exact inputs (int16-bit-pattern for bf16, native
    # float16 for fp16 — half the size of f32 either way, zero precision
    # loss); ref + truth outputs as f32 (never f64 on disk — see module
    # doc's "Storage"). `file_prefix` keeps the two dtypes' fixtures in
    # disjoint namespaces within the SAME directory.
    fp = file_prefix
    np.save(HERE / f"{fp}{name}_q.npy", npin(q))
    np.save(HERE / f"{fp}{name}_k.npy", npin(k))
    np.save(HERE / f"{fp}{name}_v.npy", npin(v))
    np.save(HERE / f"{fp}{name}_grad_out.npy", npin(g1))
    np.save(HERE / f"{fp}{name}_o.npy", npf32(o1))
    np.save(HERE / f"{fp}{name}_lse.npy", npf32(lse1))
    np.save(HERE / f"{fp}{name}_dq.npy", npf32(dq1))
    np.save(HERE / f"{fp}{name}_dk.npy", npf32(dk1))
    np.save(HERE / f"{fp}{name}_dv.npy", npf32(dv1))
    np.save(HERE / f"{fp}{name}_truth_o.npy", npf32(truth_o))
    np.save(HERE / f"{fp}{name}_truth_lse.npy", npf32(truth_lse))
    np.save(HERE / f"{fp}{name}_truth_dq.npy", npf32(truth_dq))
    np.save(HERE / f"{fp}{name}_truth_dk.npy", npf32(truth_dk))
    np.save(HERE / f"{fp}{name}_truth_dv.npy", npf32(truth_dv))

    truth_vs_ref = {
        "o": maxabsdiff(truth_o, o1),
        "lse": maxabsdiff(truth_lse, lse1),
        "dq": maxabsdiff(truth_dq, dq1),
        "dk": maxabsdiff(truth_dk, dk1),
        "dv": maxabsdiff(truth_dv, dv1),
    }

    return {
        "lengths": lengths,
        "total_q": total_q,
        "max_seqlen": max_seqlen,
        "num_heads": H,
        "head_dim": D,
        "window_radius": window,
        "softmax_scale": scale,
        "self_noise_max_abs_diff": self_noise,
        "truth_minus_ref_max_abs_diff": truth_vs_ref,
        "o_max_abs": float(o1.detach().abs().max().item()),
        "lse_max_abs": float(lse1.detach().abs().max().item()),
        "dq_max_abs": float(dq1.detach().abs().max().item()),
        "dk_max_abs": float(dk1.detach().abs().max().item()),
        "dv_max_abs": float(dv1.detach().abs().max().item()),
    }


def main():
    assert torch.cuda.is_available(), "CUDA required to generate the reference fixtures"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help=(
            "bf16 (default, unchanged historical behaviour, files unprefixed) or "
            "float16 (campaign #443 D2/D3 fp16 twin, files prefixed f16_, sidecar "
            "written to sidecar_f16.json)."
        ),
    )
    args = parser.parse_args()
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    is_f16 = dtype == torch.float16
    file_prefix = "f16_" if is_f16 else ""
    sidecar_name = "sidecar_f16.json" if is_f16 else "sidecar.json"

    sidecar = {
        "generator": "crates/jammi-kernels/tests/fixtures/flash_reference/generate_fixtures.py",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_flash_attention_vendored_commit": "6c4f74fb338e0c3cdb07ac6f5eab5f54fc367c15",
        "torch_flash_attention_vendored_version": "2.8.4",
        "torch_flash_attention_vendored_version_source": "flash_attn/__init__.py:6 __version__ at that commit; also the git tag fa4-v4.0.0.beta15 resolves to the SAME SHA (a naming artifact of the upstream repo, not a different content generation — verified: torch's CMake globs csrc/flash_attn/src/*.cu, the classic tree, at that commit)",
        "torch_flash_attention_kernel_subtree": "third_party/flash-attn/csrc/flash_attn/src (classic tree, same subtree jammi vendors)",
        "torch_flash_attention_build_defines": ["FLASHATTENTION_DISABLE_ALIBI", "FLASHATTENTION_DISABLE_SOFTCAP", "UNFUSE_FMA"],
        "torch_flash_attention_use_fast_math": False,
        "jammi_flash_attention_vendored_tag": "v2.8.3.post1",
        "jammi_flash_attention_vendored_commit": "a8aa52b1ab3e9ca574c8a33b3f35afc017ffa2e2",
        "jammi_flash_attention_use_fast_math": True,
        "amplitude_note": (
            f"randn * {SPREAD}, clamped to [-{AMPLITUDE}, {AMPLITUDE}], production amplitude "
            "(matches tests/flash_decisive_timing.rs's own convention), not an unclamped "
            "unit-variance torch.randn draw."
        ),
        "truth_note": (
            "truth_{o,lse,dq,dk,dv} are a from-scratch f64 EAGER attention implementation "
            "(einsum + softmax, no FlashAttention kernel of any generation) on the SAME "
            f"{'fp16' if is_f16 else 'bf16'}-exact q/k/v/grad_out this leg's ref (torch FA2) "
            "also consumed — the higher-precision anchor "
            "docs/maintainer/cuda-kernel-guide.md §3.3 requires. truth_minus_ref_max_abs_diff "
            "in each leg below is |truth - ref|, computed HERE (numpy-equivalent, torch f64 "
            "CPU/GPU arithmetic) so the consuming Rust oracle "
            f"({'tests/flash_torch_parity_f16.rs' if is_f16 else 'tests/flash_torch_parity.rs'}) "
            "never has to trust an unverified bound. NOTE: neither consuming Rust test opens "
            "this sidecar file at run time — every number it asserts on (o/lse/dq/dk/dv, "
            "truth-relative bound) is recomputed live from the .npy fixtures each run (family "
            "F); this JSON is provenance/human-review documentation only, self_noise_max_abs_diff "
            "included, which is REPORTED here but never consumed as a live bound."
        ),
        "storage_note": (
            "q/k/v/grad_out are saved as int16 .npy — a bit-for-bit reinterpretation of the "
            "bf16 payload (tensor.view(torch.int16)), NOT a re-round: candle-core 0.11.0's "
            "npy.rs Header::parse has no descr mapping to DType::BF16 at all, so a numpy file "
            "claiming a bf16 dtype is not readable through Tensor::read_npy regardless of what "
            "wrote it (no-hard-candle-dependency: this works around it in the fixture format, "
            "not by patching candle). o/lse/dq/dk/dv (both ref and truth) are float32 on disk, "
            "never float64 — truth's f64 precision matters only in the arithmetic that PRODUCES "
            "it."
        ),
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
            "jammi's five flash translation units DO (the bf16 and fp16 "
            "fwd/bwd .cu sources plus the flash_api_jammi.cu wrapper -- "
            "campaign #443's fp16 twins widened this from three to five; "
            "see crates/jammi-kernels/build.rs's flash source list and "
            "crates/jammi-kernels/third_party/flash-attention/VENDORED.md's "
            "Build section)."
        ),
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "num_heads": H,
        "head_dim": D,
        "dtype": args.dtype,
        "legs": {},
    }
    if is_f16:
        # The fp16 twin's own storage convention differs from the bf16
        # note above (native float16, not an int16 bit-pattern) — see this
        # file's module doc's "fp16 twin" section.
        sidecar["storage_note"] = (
            "q/k/v/grad_out are saved as NATIVE float16 .npy ('<f2' descr) — "
            "unlike bf16, numpy has a real float16 dtype and candle-core "
            "0.11.0's npy.rs Header::parse DOES map 'f2'/'e' to DType::F16, "
            "so no bit-pattern workaround is needed (plain, lossless "
            "tensor.to(torch.float16).numpy()). o/lse/dq/dk/dv (both ref and "
            "truth) remain float32 on disk, never float64, exactly as the "
            "bf16 legs."
        )

    for name, lengths, window in LEGS:
        print(f"leg {name}: lengths={lengths} window={window} dtype={args.dtype}", file=sys.stderr)
        sidecar["legs"][name] = flash_attention_leg(
            name, lengths, window, dev, dtype, file_prefix=file_prefix
        )

    with open(HERE / sidecar_name, "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True)
    print("wrote", HERE / sidecar_name, file=sys.stderr)


if __name__ == "__main__":
    main()
