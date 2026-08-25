#!/usr/bin/env python3
"""PyTorch + PEFT reference for jammi-bench's `finetune-step` tier.

This is an ORACLE, not a dependency: it lives here so a human/CI operator can
run it on a rented GPU pod next to `jammi-bench finetune-step` and compare the
two numbers step-for-step. It is pure Python against the public HuggingFace
`transformers` + `peft` APIs — never a Cargo dependency, never invoked from
CI, never vendored into a crate.

It measures the SAME unit `crates/jammi-bench/src/finetune_step.rs` measures:
one LoRA optimizer step on ModernBERT — three encoder forwards (anchor,
positive, negative, all live on the tape at once, exactly as the trainer's
`encode_groups` keeps them), a cosine-margin triplet loss over L2-normalized
mean-pooled embeddings, one backward into the LoRA tensors, one AdamW step.
See that file's module doc for why the tier is shaped this way and why the
number it produces is RECORDED (a property of `code x device x box`), never
gated against a portable floor.

Every architectural choice below is pinned to what `finetune_step.rs` and
`crates/jammi-encoders/src/pooling.rs` actually do, not to what a generic
sentence-embedding recipe would do:

* Pooling — mean over the attention mask, THEN L2-normalize, in the tensor's
  own working dtype (no upcast to fp32 for the norm), with a dtype-exact
  epsilon floor: `1e-4` for fp16 (whose smallest useful subnormal is far
  above `1e-12`), `1e-12` for everything else (fp32, bf16). See
  `pool_and_normalize`/`l2_normalize` below — a literal port of
  `pooling.rs`'s `mean_pool`/`l2_normalize`.
* Loss — `mean(relu(margin - cos(a, p) + cos(a, n)))` over already-unit-norm
  rows, so the cosine is a row-wise dot product. Margin is `0.3`, matching
  `finetune_step.rs`'s hardcoded call (the Rust tier does not expose margin
  on its CLI; this script does, as `--margin`, defaulting to the same 0.3 —
  a deliberate widening, not a divergence in what gets measured by default).
* Synthetic token ids — the identical 64-bit LCG `finetune_step.rs`'s
  `synthetic_ids` uses, so for the same `--seed` and vocab size the two
  scripts feed literally the same integers to their respective encoders (this
  does not make the two frameworks' *arithmetic* bit-identical — it removes
  "the data was different" as a variable in why the numbers differ).
* LoRA targets — `Wqkv,Wo,Wi` (the default), matched by peft's own
  dot-boundary suffix rule (`key == target or key.endswith("." + target)`).
  On ModernBERT's real module names (`attn.Wqkv`, `attn.Wo`, `mlp.Wi`,
  `mlp.Wo`) this rule catches FOUR sites per layer, not three: `Wo` matches
  both `attn.Wo` (exact tail) and `mlp.Wo` (also ends in `.Wo`). This is not
  a script bug — `jammi_lora`'s own `should_apply_lora`
  (`crates/jammi-lora/src/config.rs`) has the identical suffix behaviour
  (module names there are already bare leaves like `"Wo"`/`"mlp.Wo"`, and
  `"mlp.Wo".ends_with("Wo")` is true), so the two frameworks select the same
  four linears per layer under the same `--target-modules` value. Verified
  empirically: 16 trainable tensors for a 2-layer tiny config (2 layers x 4
  sites x {A, B}).
* Optimizer — AdamW over ONLY the trainable (LoRA) tensors, `lr=2e-4`,
  `weight_decay=0.01`, PyTorch's default `betas=(0.9, 0.999)`, `eps=1e-8`
  (matching `candle_nn::ParamsAdamW`'s own defaults for the fields
  `finetune_step.rs` does not override), `foreach=False` — see the
  `--adamw-foreach` note below.
* Batched-vs-per-group forward — `--batched-forward` (default true) mirrors
  `FinetuneStepParams::batched_forward`: one forward over the three groups
  concatenated on the batch axis, split by row AFTER pooling+normalizing.

LOSS TRAJECTORY — `finetune_step.losses` (per measured step, warmup excluded)
plus `loss_first`/`loss_last` mirror `finetune_step.rs`'s own `losses` /
`loss_first` / `loss_last` fields, same placement convention (see
`_step_once`'s docstring), so the two stacks' per-step loss sequences line up
index-for-index IF the two runs used the same `--seed`/`--batch`/`--seq`/
vocab (the synthetic token ids are then bit-identical across the two stacks —
see `synthetic_ids` below, a literal LCG port of `finetune_step.rs`'s own).
Identical INPUT ids does NOT make the two stacks' loss VALUES a meaningful
apples-to-apples quality comparison by default: candle and torch run
different arithmetic (different attention composition, different fused-vs-
eager kernel paths, different reduction order), and — read the very next
section — the LoRA adapters are not equivalently initialized unless
`--lora-init jammi` is passed, and even then not bit-identically. Absent
`--lora-init jammi`, a "jammi trajectory diverges from torch trajectory"
observation conflates at least three variables (init distribution,
attention-kernel arithmetic, framework reduction order) and proves nothing
about correctness on its own; it is printed by `finetune_ab.sh`'s table as a
same-data, cost-fixture ratio precisely so a large divergence is VISIBLE, not
so it is read as a quality regression.

LoRA INIT IS NOT A MATCH BY DEFAULT — read before comparing loss curves.
peft's default init (`init_lora_weights=True`) draws `A` from PyTorch's
`kaiming_uniform_(a=sqrt(5))`, whose bound is `1 / sqrt(fan_in)`. jammi's
`LoraInitMode::ZerosB` draws `A` from `jammi_lora::seeded::kaiming_uniform_fill`,
whose bound is `sqrt(3 / fan_in)` — `sqrt(3)` (~1.73x) WIDER (empirically
confirmed: peft's `max|A|` was 0.1768 vs jammi's bound of 0.3062 at
`fan_in=32`). `B` is zero-initialized in both, so that half already matches.
`--lora-init {peft,jammi}` (default `peft`) controls this: `peft` is the
reference's own init (use for throughput/step-time rows, where the adapter's
initial values do not matter); `jammi` re-draws every `lora_A` matrix from
jammi's own bound (use for a loss-TRAJECTORY-equivalence comparison, where
the initial distribution matters). Even under `--lora-init jammi` the two
adapters are NOT bit-identical — jammi draws from a SplitMix64 stream keyed
by `(seed, fully-qualified parameter name)`
(`jammi_lora::seeded::seed_for_param`), independent of construction order;
this script draws from torch's own default generator, advanced sequentially
in `named_parameters()` order. Only the DISTRIBUTION (uniform, same bound)
is matched, never the bits. See `reinit_lora_a_jammi_distribution` below.

Two axes this script adds that `finetune_step.rs` does not have, because
they are properties of a torch/HF stack, not of jammi's own encoder:

* `--attn {eager,sdpa}` — the REQUESTED HF attention backend. The report's
  `finetune_step.attn_implementation` is read back from the loaded model's
  OWN `config._attn_implementation` (what HF actually resolved to, which can
  differ from the request — e.g. silently falling back to `eager` when
  `sdpa` isn't available for a given config/device), never echoed from
  `args`. Run both: `sdpa` is torch's best-case number (what the throughput
  bar in #352 compares against); `eager` is the semantic twin of jammi's own
  attention composition (no fused SDPA kernel), so state which row a
  headline ratio uses. On `--attn sdpa` and CUDA, `finetune_step.sdpa_backend_probe`
  RECORDS (never assumes) which torch SDPA kernel a real forward at OUR
  shapes actually dispatches to — see `sdpa_backend_probe`'s own docstring.
  This matters because torch's flash kernel is categorically ineligible with
  ANY non-null `attn_mask` (`sdp_utils_cpp.h::check_for_attn_mask`: "Flash
  Attention does not support non-null attn_mask"); on an A100, a call
  carrying a padding mask dispatches to `EFFICIENT_ATTENTION` instead (cuDNN
  is preferred only on sm90+). But this harness's synthetic batches are
  UNPADDED (`mask` is all-ones), and current `transformers`'
  `create_bidirectional_mask` (`masking_utils._ignore_bidirectional_mask_sdpa`)
  drops such a mask to `None` before it reaches sdpa — so the `sdpa` row here
  may genuinely dispatch to FLASH, a result a padded real-world batch would
  not reproduce. Never assume "torch's best" without reading this field.
* `--dtype {fp32,amp-fp16,bf16}` — NOTE the `amp-fp16` name, deliberately NOT
  called `f16` to flag a real divergence from jammi's `--backbone-dtype f16`
  lane: jammi's F16 casts the whole backbone (weights AND activations) to
  fp16 and runs unscaled — there is no loss-scaling concept in that path.
  Pure unscaled fp16 training is numerically fragile (AdamW's `eps=1e-8`
  underflows in fp16, gradients can flush to zero), so idiomatic PyTorch fp16
  training uses automatic mixed precision instead: fp32 master weights,
  `torch.autocast` picking the op-level compute dtype, and a `GradScaler` to
  keep small gradients representable through the backward pass. `amp-fp16`
  here is exactly that AMP recipe. It is intentionally NOT a bit-comparable
  peer of jammi's `f16` lane — read it as "the fp16 lane a PyTorch user would
  actually run", not as a replica of jammi's cast. `bf16` (no scaling needed,
  same dynamic range as fp32) and `fp32` ARE architecturally comparable to
  jammi's own `bf16`/`f32` lanes: both apply the dtype as a straight
  `torch_dtype=`/backbone-cast with no autocast machinery in between.
  `amp-fp16` REQUIRES a CUDA device (`torch.autocast(device_type="cuda", ...)`
  needs one) — requesting it on CPU (including under `--dry-run`) is a hard
  error, never a silent relabel to a dtype that did not run.

VRAM: two DIFFERENT fields, mapped to DIFFERENT jammi concepts, BOTH now
backed by torch's own CONTINUOUS allocator high-water mark (not a discrete
poll — an earlier draft of this script sampled `memory_allocated()` once per
step, which was measured to land at the deterministic TROUGH of each step
(after backward+step+the `.item()` sync, when every saved activation is
already freed): 403 KiB captured of a 9087 KiB in-step peak, ~4.4%,
systematically, on both measured steps. That per-step poll has been REMOVED;
both VRAM fields below now come from one `torch.cuda.reset_peak_memory_stats()`
call made once, before the warmup+measured loop starts, and one
`torch.cuda.max_memory_allocated()` read after it ends — a continuous
high-water mark that cannot miss an intra-step spike the way a discrete poll
(this script's old per-step read, or jammi's own 25ms `nvidia-smi` interval)
can.

jammi's `peak_vram_bytes` (`finetune_step.rs:112,:233`) is a whole-device
`nvidia-smi` poll, sampled every 25ms over the ENTIRE warmup+measured loop,
minus a baseline snapshot taken once right after the model+optimizer are
built (before the loop starts) — at which point candle's `AdamW::new` has
ALREADY allocated the (zero-initialized) first/second moment tensors, since
candle allocates them eagerly at construction. Torch's `AdamW`, by contrast,
allocates its `exp_avg`/`exp_avg_sq` state LAZILY on the first `step()` call
(measured: 0 state tensors before the first step, 48 after, on a tiny test
model) — so a baseline taken right after `torch.optim.AdamW(...)` returns
would NOT yet include the moments, and their first-step allocation would
land INSIDE the measured delta instead of being absorbed into the baseline
the way jammi's is. To make the baseline honestly comparable, this script
runs ONE UNTIMED optimizer step (forward + backward + `optimizer.step()`,
via `_step_once`, not counted in `--warmup`/`--steps` and not part of any
reported timing) BEFORE taking the baseline snapshot — forcing the lazy
moments into existence first, the honest equivalent of candle's eager
allocation. (Side effect, stated plainly: this means the model has already
taken one real gradient step before the officially-reported `--warmup`
step 0 begins. Irrelevant to cost/VRAM measurement — activation shapes and
optimizer-state sizes do not depend on the weights' actual values — and this
script never reports or interprets the loss value, so no reported number is
affected.)

* **`peak_vram_delta_bytes`** — the field COMPARABLE to jammi's
  `peak_vram_bytes` column. Same window (the entire warmup+measured loop,
  reset happens once before it starts), same baseline convention (a
  `torch.cuda.memory_allocated()` snapshot taken after model+optimizer
  construction AND after the one untimed moment-warmup step described
  above, recorded separately as `peak_vram_baseline_bytes`).
  `peak_vram_delta_bytes = max_memory_allocated() - peak_vram_baseline_bytes`
  after the loop. RESIDUAL ASYMMETRY, stated rather than papered over: this
  is now a CONTINUOUS allocator high-water mark; jammi's is a 25ms-interval
  discrete poll. A continuous tracker cannot miss an intra-step spike a
  25ms poll can straddle — so `peak_vram_delta_bytes` may legitimately read
  HIGHER than jammi's `peak_vram_bytes` even when the underlying activation
  footprint is identical, purely from the sampling-method difference, not
  from a real workload difference. Do not read a gap between the two as a
  regression without checking which side the sampling-method asymmetry
  would push it.
* **`peak_vram_absolute_bytes`** — the SAME continuous high-water mark, over
  the SAME window, WITHOUT the baseline subtraction: raw bytes live at the
  peak (model weights + LoRA adapters + optimizer moments + the peak
  activation footprint). No jammi analogue (jammi only ever reports the
  baseline-subtracted figure); useful on its own as "how much device memory
  did this configuration actually need", not as a substitute for
  `peak_vram_delta_bytes` in a jammi comparison.

FAST-PATH GLOBALS are pinned and RECORDED (never left at whatever a torch
build defaults to), so a `sdpa`/`eager` comparison is not accidentally
riding on TF32 or a JIT-compiled fast path jammi's own uncompiled kernels
never get to use: `torch.backends.cuda.matmul.allow_tf32 = False`,
`torch.backends.cudnn.allow_tf32 = False`, `torch.backends.cudnn.benchmark =
False`, `torch.set_float32_matmul_precision("highest")`, and
`reference_compile=False` passed to the HF config loader (HF self-enables
`torch.compile` on ModernBERT's MLP/embeddings when `triton` is importable;
an unrequested compiled reference vs jammi's uncompiled kernels would not be
an apples-to-apples throughput comparison). The config-loader kwarg is
guarded: on a `transformers` version that dropped the field entirely, this
script observed passing an unrecognized kwarg through
`AutoConfig.from_pretrained` can itself raise on some ModernBERT config
versions (a `transformers`-internal validation issue unrelated to this
script) — the guard falls back to the plain unmodified loader call rather
than propagating that error.

Whether the pin actually took is NEVER inferred from "the call didn't raise"
(measured: it can not-raise while silently dropping the kwarg — a `cfg`
built this way had `hasattr(cfg, "reference_compile") == False`, i.e. the
field does not exist on the installed `transformers` version at all, yet a
naive "accepted" flag would have reported `True`). Instead this script
reads the RESOLVED value back off the model's own config at two points and
reports both, never a boolean: `finetune_step.reference_compile_resolved`
(`getattr(model.config, "reference_compile", "absent")`, read right after
`AutoModel.from_pretrained` returns) and
`finetune_step.reference_compile_after_first_forward` (the same read,
repeated after one forward pass — on some `transformers` 4.48-4.5x releases
an internal `_maybe_set_compile` hook mutates this field at forward time,
so the pre-forward and post-forward readings can legitimately differ).
`"absent"` means the installed `transformers` version has no such field at
all on `ModernBertConfig` — which also means there is no compile-on-first-
forward risk to guard against on that version in the first place.

Install: developed and exercised against (via `uv`, a fresh venv, CPU-only,
`--dry-run`): `torch==2.13.0  transformers==5.15.1  peft==0.20.0`. Minimum
`transformers>=4.48.0` is REQUIRED for `ModernBertConfig`/`ModernBertModel`
to exist at all (ModernBERT landed in that release, 2025-01-10) — an earlier
docstring in this file claimed `transformers==4.44.2` (2024-08-22) as the
developed-against version, which predates ModernBERT and could not have
exercised this script; that claim was false and has been corrected here to
the versions actually run. No requirements-pinning file ships next to this
script on purpose (B2: a pinning file here is something CI could pick up and
start enforcing against a crate that has no Python toolchain; the versions
live in this docstring and in the README instead, and the `provenance` block
below records the versions actually present at run time — that block is the
authority, not this docstring).

Usage (on the pod, against a real ModernBERT-large checkpoint directory):
    uv run crates/jammi-bench/reference/torch_finetune_step.py \\
        --model-dir /path/to/ModernBERT-large --batch 8 --seq 128 \\
        --dtype bf16 --attn sdpa

Usage (anywhere, no GPU, no checkpoint — exercises the REAL loader path: a
tiny random-init 2-layer ModernBERT is `save_pretrained`'d to a temp dir and
then reloaded through the exact same `AutoConfig`/`AutoModel.from_pretrained`
+ dtype-map + `attn_implementation` code the GPU path uses):
    python3 crates/jammi-bench/reference/torch_finetune_step.py --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import subprocess
import sys
import tempfile
import time

MASK64 = (1 << 64) - 1
LCG_MUL = 6364136223846793005
LCG_INC = 1442695040888963407

# The tiny random-init ModernBERT the --dry-run donor checkpoint uses. Values
# chosen only to be small, fast, and internally consistent (hidden_size
# divisible by num_attention_heads); they carry no meaning beyond "a valid
# ModernBERT config".
DRY_RUN_VOCAB_SIZE = 1000
DRY_RUN_HIDDEN_SIZE = 32
DRY_RUN_INTERMEDIATE_SIZE = 64
DRY_RUN_NUM_LAYERS = 2
DRY_RUN_NUM_HEADS = 4
DRY_RUN_MAX_POSITION_EMBEDDINGS = 64
DRY_RUN_BATCH = 2
DRY_RUN_SEQ = 16
DRY_RUN_STEPS = 2
DRY_RUN_WARMUP = 0


def synthetic_ids(batch: int, seq: int, vocab: int, seed: int):
    """Bit-identical port of `finetune_step.rs::synthetic_ids`.

    A 64-bit LCG seeded from `seed`, emitting `batch * seq` ids uniform over
    `[1, vocab)` — never the pad id `0`. Same recurrence, same wrapping
    arithmetic (masked to 64 bits at every step, matching Rust's
    `wrapping_mul`/`wrapping_add`), so for the same `(seed, vocab)` this
    yields the exact same token ids `finetune_step.rs` feeds its encoder.
    """
    import torch

    if vocab < 2:
        raise ValueError(f"vocab must be >= 2 to leave room for a non-pad id, got {vocab}")
    s = (seed * LCG_MUL + 1) & MASK64
    ids = []
    for _ in range(batch * seq):
        s = (s * LCG_MUL + LCG_INC) & MASK64
        ids.append(1 + ((s >> 33) % (vocab - 1)))
    return torch.tensor(ids, dtype=torch.long).reshape(batch, seq)


def norm_floor(dtype) -> float:
    """Port of `pooling.rs::norm_floor`: fp16 gets its own dtype-exact floor
    (`1e-12` would underflow to `0.0` once cast to fp16), everything else
    (fp32, bf16 — matching Rust's `_ => 1e-12` arm) gets `1e-12`."""
    import torch

    return 1e-4 if dtype == torch.float16 else 1e-12


def pool_and_normalize(hidden, attention_mask):
    """Literal port of `pooling.rs::mean_pool` + `l2_normalize`.

    Mean over real (non-pad) tokens via the attention mask, in `hidden`'s own
    dtype (deliberately no upcast to fp32 — jammi's tier does not upcast
    either, and the comparison is meant to include whatever precision cost
    that choice has), then L2-normalize with the dtype-exact floor above so
    an all-padding row (none exist in this synthetic-uniform-mask harness,
    but the invariant is worth mirroring) is a finite zero, never a NaN.
    """
    import torch

    mask = attention_mask.to(torch.float32).unsqueeze(-1).expand(hidden.shape)
    masked = hidden * mask.to(hidden.dtype)
    summed = masked.sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / count.to(hidden.dtype)
    norm = pooled.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=norm_floor(pooled.dtype))
    return pooled / norm


def triplet_loss(a, p, n, margin: float):
    """Literal port of `finetune_step.rs::triplet_loss`:
    `mean(relu(margin - cos(a, p) + cos(a, n)))` over already-unit-norm rows,
    so the cosine is a row-wise dot product (`sum` over the last axis)."""
    pos = (a * p).sum(dim=-1)
    neg = (a * n).sum(dim=-1)
    raw = neg - pos + margin
    return raw.relu().mean()


def pick_device(cuda_ordinal):
    import torch

    if cuda_ordinal is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{cuda_ordinal}")
    return torch.device("cpu")


def pin_fast_path_globals():
    """Pin every fast-path global this script knows about to the OFF/strict
    setting, so a `--attn sdpa` row is not silently riding on TF32 or a
    cudnn-autotuned algorithm jammi's own kernels never get to use. Returns
    the dict recorded in `provenance.fast_path_globals`.

    These are process-wide `torch.backends`/`torch` globals — harmless to set
    on a CPU-only run (they simply have no effect there), so this always
    runs, not just on CUDA.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    return {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def peak_rss_bytes():
    """Peak resident set from `/proc/self/status` `VmHWM` — the exact same
    source and same "absent, not faked" convention as
    `finetune_step.rs::peak_rss_bytes` (which is Linux-only for the same
    reason: the field does not exist elsewhere)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    kb = float(line.split()[1])
                    return kb * 1024.0
    except (FileNotFoundError, OSError, IndexError, ValueError):
        pass
    return None


def nvidia_smi_field(query: str):
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    line = out.stdout.strip().splitlines()
    return line[0].strip() if line else None


def git_rev(repo_hint: str):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_hint,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(device, fast_path_globals):
    import torch

    try:
        import transformers
    except ImportError:
        transformers = None
    try:
        import peft
    except ImportError:
        peft = None

    info = {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_rev": git_rev(os.path.dirname(os.path.abspath(__file__))),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": getattr(transformers, "__version__", None),
        "peft_version": getattr(peft, "__version__", None),
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "nvidia_driver_version": None,
        "fast_path_globals": fast_path_globals,
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device)
        info["nvidia_driver_version"] = nvidia_smi_field("driver_version")
    return info


def build_dry_run_checkpoint(tmp_dir: str) -> str:
    """Materialize a tiny random-init 2-layer ModernBERT to `tmp_dir` via
    `save_pretrained`, so the `--dry-run` path can reload it through the
    IDENTICAL `load_model` code the real GPU path uses (`AutoConfig`/
    `AutoModel.from_pretrained` + the dtype map + `attn_implementation`) —
    the loader, the dtype dict, and the attention selector are exercised for
    real, not bypassed. The random weights themselves are never inspected;
    this checkpoint exists only to be loadable.
    """
    from transformers import ModernBertConfig, ModernBertModel

    config = ModernBertConfig(
        vocab_size=DRY_RUN_VOCAB_SIZE,
        hidden_size=DRY_RUN_HIDDEN_SIZE,
        intermediate_size=DRY_RUN_INTERMEDIATE_SIZE,
        num_hidden_layers=DRY_RUN_NUM_LAYERS,
        num_attention_heads=DRY_RUN_NUM_HEADS,
        max_position_embeddings=DRY_RUN_MAX_POSITION_EMBEDDINGS,
        pad_token_id=0,
    )
    model = ModernBertModel(config)
    model.save_pretrained(tmp_dir)
    return tmp_dir


def load_config_reference_compile_off(model_dir):
    """`AutoConfig.from_pretrained(model_dir, reference_compile=False)`,
    guarded: on a `transformers` version that dropped the field (or, as
    observed against `transformers==5.15.1`'s `ModernBertConfig`, where
    passing ANY unrecognized kwarg through this path can itself raise from
    that version's own strict-dataclass rope-parameter validation — a
    `transformers`-internal issue, not something this script can fix), fall
    back to the plain unmodified loader call. Never lets the fast-path pin
    itself be the reason the script fails to run.

    Whether the pin actually took is NOT this function's job to report —
    passing the kwarg can silently no-op (accepted without error, but
    dropped: `hasattr(cfg, "reference_compile")` was measured `False`
    immediately after a "successful", non-raising call). The caller reads
    the RESOLVED value back off `model.config` instead (see `run()`), never
    inferring it from whether this call raised.
    """
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_dir, reference_compile=False)
    except Exception:  # noqa: BLE001 - deliberately broad, see docstring
        return AutoConfig.from_pretrained(model_dir)


def load_model(args):
    """The ONE loader path both the real GPU run and `--dry-run` go through:
    `AutoConfig`/`AutoModel.from_pretrained` with the dtype map and
    `attn_implementation` exactly as requested. `--dry-run` points
    `args.model_dir` at a freshly `save_pretrained`'d tiny checkpoint
    (`build_dry_run_checkpoint`) before calling this, so this function itself
    never branches on `args.dry_run`.
    """
    from transformers import AutoModel
    import torch

    torch_dtype = {
        "fp32": torch.float32,
        "amp-fp16": torch.float32,  # fp32 master weights; autocast picks fp16 per-op
        "bf16": torch.bfloat16,
    }[args.dtype]
    config = load_config_reference_compile_off(args.model_dir)
    model = AutoModel.from_pretrained(
        args.model_dir,
        config=config,
        attn_implementation=args.attn,
        torch_dtype=torch_dtype,
    )
    return model, config


def wrap_lora(model, args):
    from peft import LoraConfig, get_peft_model

    target_modules = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, lora_config)


def reinit_lora_a_jammi_distribution(model):
    """Re-draw every LoRA `A` matrix from jammi's own Kaiming-uniform bound —
    see `jammi_lora::seeded::kaiming_uniform_fill`: `U(-bound, bound)` with
    `bound = sqrt(3 / fan_in)`. peft's default (`kaiming_uniform_(a=sqrt(5))`)
    bound is `1 / sqrt(fan_in)` — jammi's is `sqrt(3)` (~1.73x) WIDER.

    NOT a bit-identical replica: jammi draws each `A` from an independent
    SplitMix64 stream keyed by `(seed, fully-qualified parameter name)`
    (`seed_for_param`), invariant to construction order; this function draws
    from torch's own default generator (seeded by the caller's earlier
    `torch.manual_seed(seed)`), advanced sequentially through
    `named_parameters()` order — deterministic given `seed`, but not a
    cross-RNG match. Distribution-matched only: same family (uniform), same
    bound. `B` is left at peft's zero-init, already matching jammi's
    `ZerosB`.

    Returns the count of `lora_A` tensors it actually re-drew, and ASSERTS
    that count is `> 0` — a silent no-op (peft's parameter naming changing
    out from under the `.endswith("lora_A.default.weight")` match, on some
    future peft version) must be a loud failure, not a report that quietly
    still says `lora_init: "jammi"` while every `A` matrix is still at
    peft's own init.
    """
    import torch

    matched = 0
    for name, param in model.named_parameters():
        if not (name.endswith("lora_A.default.weight") and param.requires_grad):
            continue
        fan_in = param.shape[-1]
        bound = math.sqrt(3.0 / fan_in)
        with torch.no_grad():
            param.uniform_(-bound, bound)
        matched += 1
    assert matched > 0, (
        "reinit_lora_a_jammi_distribution matched zero lora_A tensors — the "
        "peft-init distribution would silently remain in place. Check peft's "
        "parameter naming (expected every trainable tensor's name to end in "
        "'lora_A.default.weight') against the installed peft version."
    )
    return matched


def forward_hidden(model, input_ids, attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]


def _step_once(model, optimizer, scaler, blocks, mask, args, use_amp, device):
    """One full forward + cosine-margin triplet loss + backward + optimizer
    step — the exact body the timed loop in `run()` executes, factored out
    so the SAME code can also be run once, untimed, before the timed loop
    starts (to force AdamW's lazily-allocated moment tensors into existence
    before the VRAM baseline is read, and to exercise one real forward for
    the `reference_compile_after_first_forward` readback). Ends with a CUDA
    sync (guarded: a no-op on CPU) so the caller can rely on the step having
    actually completed, not just been submitted.

    Returns the step's loss as a plain Python float — the same value read
    off `loss.detach().float().item()` below (the call this function already
    made to force the CUDA sync; no second read is added). PLACEMENT: this
    read happens AFTER `optimizer.step()` / `scaler.step(optimizer)` returns,
    exactly mirroring `finetune_step.rs`'s own placement (its comment at the
    loss-read call site states the identical convention). In both stacks the
    loss TENSOR was produced by the forward earlier in this same function,
    BEFORE the optimizer step — reading it after only decides when the host
    blocks on the (already-queued) device work, it does not recompute the
    loss against the just-updated weights. So the returned value is the
    PRE-UPDATE loss of this call's batch, not a re-evaluation after the
    step. `run()`'s timed loop below records this for every MEASURED
    (post-warmup) call, so `finetune_step.losses` here and
    `finetune_step.rs`'s `losses` field share the same per-step placement
    convention and are comparable step-for-step under that convention.
    """
    import torch

    optimizer.zero_grad(set_to_none=True)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        if args.batched_forward:
            joined_ids = torch.cat(blocks, dim=0)
            joined_mask = mask.repeat(3, 1)
            hidden = forward_hidden(model, joined_ids, joined_mask)
            pooled = pool_and_normalize(hidden, joined_mask)
            b = args.batch
            a, p, n = pooled[:b], pooled[b : 2 * b], pooled[2 * b : 3 * b]
        else:
            a, p, n = (
                pool_and_normalize(forward_hidden(model, blk, mask), mask) for blk in blocks
            )
        loss = triplet_loss(a, p, n, margin=args.margin)

    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # Force completion before returning: CUDA's queue is asynchronous, so
    # without a sync point the caller's clock (when this is called from the
    # timed loop) would measure submission time, not execution time — the
    # same caveat `finetune_step.rs` documents at its own `.to_scalar()`
    # call.
    loss_val = loss.detach().float().item()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return loss_val


def _probe_sdpa_backend_forward(model, backend, input_ids, attention_mask):
    """Try ONE real forward pass restricted to a single torch SDPA backend,
    via `torch.nn.attention.sdpa_kernel`. `"ok"` if the forward completes;
    `"ineligible: <message>"` if torch raises — a `RuntimeError` here IS the
    signal (e.g. `sdp_utils_cpp.h::check_for_attn_mask`'s categorical "Flash
    Attention does not support non-null attn_mask" check firing), never
    inferred from anything else. `torch.no_grad()` because this is a probe,
    not a training step — its output is discarded, only whether it raised
    matters.
    """
    import torch
    from torch.nn.attention import sdpa_kernel

    try:
        with torch.no_grad(), sdpa_kernel([backend]):
            model(input_ids=input_ids, attention_mask=attention_mask)
        return "ok"
    except RuntimeError as exc:
        return f"ineligible: {exc}"


def sdpa_backend_probe(model, model_config, blocks, mask, device, args):
    """RECORD (never assume) which torch SDPA backend a real forward at OUR
    shapes actually dispatches to. This exists because PyTorch's flash
    kernel is categorically ineligible with ANY non-null `attn_mask`
    (`sdp_utils_cpp.h::check_for_attn_mask`), and on an A100 a call carrying
    a padding mask dispatches to `EFFICIENT_ATTENTION` instead (`cuDNN` is
    only preferred on sm90+) — but whether OUR mask actually reaches sdpa as
    non-null is itself empirical, not assumed: this harness's synthetic
    batches are unpadded (`mask` is all-ones), and current `transformers`'
    `create_bidirectional_mask` (via
    `masking_utils._ignore_bidirectional_mask_sdpa`, verified by reading
    that function's source: it returns `True` — meaning "skip, pass `None`
    to sdpa" — whenever `padding_mask is None or padding_mask.all()`) drops
    an all-ones mask to `None` before `ModernBertModel`'s own
    `sdpa_attention_forward` call ever sees it. So the `sdpa` row here MAY
    genuinely dispatch to FLASH, while a padded real-world batch would not.
    The "torch's best" throughput claim must record the backend, not assume
    it — hence this probe, run twice, independently:

    1. Per-backend forward under `sdpa_kernel([<one backend>])`,
       try/except — see `_probe_sdpa_backend_forward`.
    2. `torch.backends.cuda.SDPAParams` built from representative
       `(q, k, v, mask, dropout_p, is_causal)` tensors shaped like what
       ModernBERT's `sdpa_attention_forward` actually passes at OUR
       `--batch`/`--seq`: `mask=None` (the same empirically-verified reason
       as above), `is_causal=False` (ModernBERT is bidirectional, never
       causal), `dropout_p=config.attention_dropout`, `enable_gqa=False`
       (`ModernBertConfig` has no `num_key_value_heads` field — this
       backbone has no grouped-query-attention path to enable) — fed to
       `can_use_flash_attention`/`can_use_efficient_attention`/
       `can_use_cudnn_attention`. `SDPAParams`'s constructor gained a
       trailing `enable_gqa` bool in a later torch release than some
       still-supported versions; the 7-arg (with `enable_gqa`) call is
       tried first (this is what torch 2.13.0 requires), falling back to
       the 6-arg (pre-`enable_gqa`) signature on `TypeError`, so this does
       not hard-fail across the version range.

    Only runs on CUDA with `--attn sdpa` — the combination the module
    docstring's flash/efficient/cudnn claim is actually about. Returns the
    plain string `"n/a (cpu)"` on CPU (including `--dry-run`), or an honest
    `"n/a (attn=...)"` string when `--attn eager` was requested (the probe
    is about sdpa backend selection specifically; forcing a backend via
    `sdpa_kernel` on a model that loaded the eager attention path would not
    measure anything the eager path itself does). Never a guessed value:
    every string here is either a literal not-applicable marker or the
    direct result of an empirical try/except or an empirical library call.
    """
    if device.type != "cuda":
        return "n/a (cpu)"
    if args.attn != "sdpa":
        return f"n/a (attn={args.attn}, probe only applies to --attn sdpa)"

    import torch
    from torch.nn.attention import SDPBackend

    probe_ids = blocks[0]
    result = {
        "flash": _probe_sdpa_backend_forward(model, SDPBackend.FLASH_ATTENTION, probe_ids, mask),
        "efficient": _probe_sdpa_backend_forward(
            model, SDPBackend.EFFICIENT_ATTENTION, probe_ids, mask
        ),
        "cudnn": _probe_sdpa_backend_forward(model, SDPBackend.CUDNN_ATTENTION, probe_ids, mask),
    }

    try:
        if not hasattr(torch.backends.cuda, "SDPAParams"):
            raise AttributeError("torch.backends.cuda.SDPAParams does not exist on this build")
        num_heads = model_config.num_attention_heads
        head_dim = model_config.hidden_size // num_heads
        dropout_p = float(getattr(model_config, "attention_dropout", 0.0))
        param_dtype = next(model.parameters()).dtype
        q = torch.zeros(
            args.batch, num_heads, args.seq, head_dim, dtype=param_dtype, device=device
        )
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        attn_mask = None  # verified empirically above: all-ones mask never reaches sdpa
        is_causal = False  # ModernBERT is bidirectional
        enable_gqa = False  # ModernBertConfig has no num_key_value_heads field
        try:
            params = torch.backends.cuda.SDPAParams(
                q, k, v, attn_mask, dropout_p, is_causal, enable_gqa
            )
        except TypeError:
            # Older torch: SDPAParams predates the trailing enable_gqa arg.
            params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal)
        result["sdpa_params_probe"] = {
            "flash_can_use": bool(torch.backends.cuda.can_use_flash_attention(params, True)),
            "efficient_can_use": bool(
                torch.backends.cuda.can_use_efficient_attention(params, True)
            ),
            "cudnn_can_use": (
                bool(torch.backends.cuda.can_use_cudnn_attention(params, True))
                if hasattr(torch.backends.cuda, "can_use_cudnn_attention")
                else "probe-unavailable: can_use_cudnn_attention not present on this torch build"
            ),
            "note": "debug=True only enables PyTorch's internal C++ LOG(WARNING) (a "
            "c10/glog-style sink), which writes to the process's native stderr fd "
            "outside Python's logging/warnings machinery -- not capturable as a string "
            "in this JSON field. Redirect the pod's own stderr if the specific "
            "ineligibility message text is needed; the booleans above are the "
            "reliably observable half of this probe.",
        }
    except Exception as exc:  # noqa: BLE001 - a probe must never crash the run; honesty
        # over a guess: record exactly what went wrong, never a fabricated value.
        result["sdpa_params_probe"] = f"probe-unavailable: {type(exc).__name__}: {exc}"

    return result


def run(args):
    import torch

    fast_path_globals = pin_fast_path_globals()

    device = pick_device(None if args.dry_run else args.cuda)
    is_cuda = device.type == "cuda"

    if args.dtype == "amp-fp16" and not is_cuda:
        raise ValueError(
            "--dtype amp-fp16 requires a CUDA device (torch.autocast('cuda', ...) "
            "needs one); use --dtype fp32 or bf16 for a CPU run or --dry-run. "
            "Never silently relabelled: reporting amp-fp16 without running it "
            "would misrepresent what ran."
        )

    dry_run_tmp = tempfile.TemporaryDirectory() if args.dry_run else contextlib.nullcontext()
    with dry_run_tmp as tmp_dir:
        if args.dry_run:
            # Hardcode small, always-valid shape/step knobs so the smoke test
            # never depends on a real checkpoint's vocab/hidden size; dtype,
            # attn, and every LoRA/optimizer knob are left as the caller
            # requested (subject to the amp-fp16-on-CPU guard above), so this
            # exercises the SAME load_model/wrap_lora code the GPU path uses.
            # The donor checkpoint itself (`build_dry_run_checkpoint`) is
            # built further below, AFTER `torch.manual_seed`, so its own
            # random init is deterministic given --seed too.
            args = argparse.Namespace(**vars(args))
            args.batch = DRY_RUN_BATCH
            args.seq = DRY_RUN_SEQ
            args.steps = DRY_RUN_STEPS
            args.warmup = DRY_RUN_WARMUP

        # Seed BEFORE any model/adapter/checkpoint construction — including
        # the --dry-run donor checkpoint below — so the whole random-draw
        # pipeline is deterministic given --seed, not just the parts that
        # happen to run after this line. peft's default LoRA init draws from
        # torch's global generator at `get_peft_model` time, so the
        # generator must already be seeded by `--seed` when that call
        # happens. (jammi's own seed is a pure function of `(seed, parameter
        # name)` — independent of call/construction order — which is a
        # different and stronger determinism guarantee than "seeded before
        # the first draw"; stated here, not claimed as equivalent.)
        torch.manual_seed(args.seed)

        if args.dry_run:
            args.model_dir = build_dry_run_checkpoint(tmp_dir)

        model, config = load_model(args)
        resolved_attn_implementation = getattr(model.config, "_attn_implementation", "absent")
        reference_compile_resolved = getattr(model.config, "reference_compile", "absent")

        model = wrap_lora(model, args)
        lora_a_tensors_reinitialized = None
        if args.lora_init == "jammi":
            lora_a_tensors_reinitialized = reinit_lora_a_jammi_distribution(model)
        model.to(device)
        model.train()

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "no trainable LoRA tensors — target_modules matched nothing "
                f"(target_modules={args.target_modules!r})"
            )
        # foreach=False pins AdamW to torch's per-tensor (non-multi-tensor)
        # update path, the closer structural peer of candle_nn::AdamW's own
        # per-tensor step loop — torch's multi-tensor `foreach` default is a
        # real fast path jammi's optimizer does not have, and belongs (if at
        # all) to a "torch's absolute best" row, out of scope here.
        optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01, foreach=False)
        # Read back what actually took, not the literal passed in — torch is
        # free to normalize/override a per-group default.
        adamw_foreach = optimizer.param_groups[0]["foreach"]

        use_amp = is_cuda and args.dtype == "amp-fp16"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        vocab = config.vocab_size
        mask = torch.ones(args.batch, args.seq, dtype=torch.long, device=device)
        blocks = [
            synthetic_ids(args.batch, args.seq, vocab, args.seed + i).to(device)
            for i in range(3)
        ]

        # ONE untimed step, BEFORE the timed loop and BEFORE the VRAM
        # baseline: forces torch's AdamW to allocate its (lazily-created)
        # exp_avg/exp_avg_sq moment tensors now, mirroring candle's own
        # eager allocation in `AdamW::new` — without this, the moments'
        # first-step allocation would land inside the measured VRAM delta
        # instead of the baseline (measured: 0 optimizer state tensors
        # before this step, 48 after, on a tiny test model). Also exercises
        # the model's first real forward, so `reference_compile_after_first_forward`
        # reflects whatever HF's own compile-selection hook did at forward
        # time (observed on some 4.48-4.5x releases: `_maybe_set_compile`
        # mutates `config.reference_compile` only once a forward runs).
        # Not counted in `--warmup`/`--steps`, never reported as a timed
        # sample. Runs regardless of device (CPU included), guarded
        # internally for CUDA-only calls.
        _step_once(model, optimizer, scaler, blocks, mask, args, use_amp, device)
        reference_compile_after_first_forward = getattr(
            model.config, "reference_compile", "absent"
        )

        # RECORD, never assume, which torch SDPA backend a real forward at
        # OUR shapes actually dispatches to — see `sdpa_backend_probe`'s own
        # docstring for why an unpadded synthetic batch can genuinely hit
        # FLASH even though torch's flash kernel is categorically ineligible
        # with any non-null attn_mask. Run AFTER the untimed warm-up step
        # (model+adapter+optimizer are fully constructed and resident by
        # this point) and BEFORE the VRAM baseline reset, so the probe's own
        # (discarded) forward passes cannot pollute the measured window.
        sdpa_backend_probe_result = sdpa_backend_probe(model, config, blocks, mask, device, args)

        vram_baseline_bytes = None
        if is_cuda:
            vram_baseline_bytes = torch.cuda.memory_allocated(device)
            # Single reset, right here — before the timed warmup+measured
            # loop starts, matching the window jammi's own background
            # sampler covers (finetune_step.rs starts its VramSampler right
            # after this same "model+optimizer resident" point). Continuous
            # high-water tracking from here on; no further resets, so no
            # intra-step spike can be missed the way a discrete per-step or
            # per-25ms poll could miss one.
            torch.cuda.reset_peak_memory_stats(device)

        times = []
        losses = []
        for step in range(args.warmup + args.steps):
            t0 = time.perf_counter()
            loss_val = _step_once(model, optimizer, scaler, blocks, mask, args, use_amp, device)
            elapsed = time.perf_counter() - t0
            if step >= args.warmup:
                times.append(elapsed)
                losses.append(loss_val)

        times.sort()
        p50 = times[len(times) // 2]
        mean = sum(times) / len(times)
        peak_vram_absolute = torch.cuda.max_memory_allocated(device) if is_cuda else None
        peak_vram_delta = (
            (peak_vram_absolute - vram_baseline_bytes) if is_cuda else None
        )

        report = {
            "tool": "torch_finetune_step",
            "provenance": provenance(device, fast_path_globals),
            "args": {
                "model_dir": None if args.dry_run else str(args.model_dir),
                "dry_run": args.dry_run,
                "batch": args.batch,
                "seq": args.seq,
                "steps": args.steps,
                "warmup": args.warmup,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_init": args.lora_init,
                "target_modules": args.target_modules,
                "dtype": args.dtype,
                "attn_requested": args.attn,
                "cuda": args.cuda,
                "seed": args.seed,
                "batched_forward": args.batched_forward,
                "margin": args.margin,
                "adamw_foreach": adamw_foreach,
                "moment_warmup_step_executed": True,
            },
            "finetune_step": {
                "device": str(device),
                "backbone_dtype": args.dtype,
                "attn_implementation": resolved_attn_implementation,
                "sdpa_backend_probe": sdpa_backend_probe_result,
                "reference_compile_resolved": reference_compile_resolved,
                "reference_compile_after_first_forward": reference_compile_after_first_forward,
                "batch": args.batch,
                "seq": args.seq,
                "lora_rank": args.lora_rank,
                "lora_dropout": args.lora_dropout,
                "lora_init": args.lora_init,
                "lora_a_tensors_reinitialized": lora_a_tensors_reinitialized,
                "target_modules": [
                    t.strip() for t in args.target_modules.split(",") if t.strip()
                ],
                "batched_forward": args.batched_forward,
                "trainable_tensors": len(trainable),
                "steps_measured": len(times),
                "losses": losses,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "loss_note": "cost-fixture data, not a quality result (synthetic uniform "
                "token ids) — see finetune_step.rs's module doc's 'Honesty about what is "
                "measured'. Each entry read after optimizer.step() returns; the value "
                "itself is the PRE-update loss of that step's batch (see _step_once's "
                "docstring for the placement convention shared with finetune_step.rs).",
                "s_per_step_p50": {"value": p50, "unit": "s"},
                "s_per_step_mean": {"value": mean, "unit": "s"},
                "steps_per_s": {"value": 1.0 / p50, "unit": "steps/s"},
                "triplets_per_s": {"value": args.batch / p50, "unit": "triplets/s"},
                "peak_rss_bytes": {"value": peak_rss_bytes(), "unit": "bytes"},
                "peak_vram_baseline_bytes": {
                    "value": float(vram_baseline_bytes)
                    if vram_baseline_bytes is not None
                    else None,
                    "unit": "bytes",
                    "note": "memory_allocated() snapshot taken after model+optimizer "
                    "construction AND after one untimed optimizer warm-up step (so "
                    "AdamW's lazily-allocated moments already exist) — the honest "
                    "equivalent of candle's eager moment allocation before jammi's own "
                    "baseline read.",
                },
                "peak_vram_absolute_bytes": {
                    "value": float(peak_vram_absolute)
                    if peak_vram_absolute is not None
                    else None,
                    "unit": "bytes",
                    "note": "torch's continuous allocator high-water mark "
                    "(reset_peak_memory_stats once before the warmup+measured loop, "
                    "max_memory_allocated read after it) over the SAME window as "
                    "peak_vram_delta_bytes, WITHOUT the baseline subtraction. No jammi "
                    "analogue.",
                },
                "peak_vram_delta_bytes": {
                    "value": float(peak_vram_delta) if peak_vram_delta is not None else None,
                    "unit": "bytes",
                    "note": "Comparable to jammi's peak_vram_bytes column: same window "
                    "(warmup+measured), same baseline convention (snapshot after "
                    "model+optimizer construction, moments already forced into "
                    "existence). RESIDUAL ASYMMETRY: this is a CONTINUOUS allocator "
                    "high-water mark; jammi's is a 25ms-interval discrete nvidia-smi "
                    "poll, which can miss an intra-step spike this cannot — so this "
                    "figure may legitimately read higher than jammi's for identical "
                    "underlying work.",
                },
            },
        }
        return report


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "PyTorch + PEFT reference for jammi-bench's finetune-step tier. "
            "Argument names mirror `jammi-bench finetune-step` where the "
            "underlying concept is the same; see the module docstring for "
            "every deliberate divergence (--dtype's amp-fp16 lane, --attn, "
            "--margin, --lora-init, VRAM fields)."
        ),
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory holding a HF ModernBERT checkpoint (config.json + weights). "
        "Required unless --dry-run.",
    )
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Default 16 per the C8 contract's reference adapter shape — NOTE this "
        "differs from jammi-bench's own CLI default of 8; pass --lora-rank to match "
        "whatever a specific jammi run used.",
    )
    p.add_argument(
        "--lora-alpha",
        type=float,
        default=32.0,
        help="Default 32 per the C8 contract (jammi-bench's own CLI default is 16).",
    )
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-init",
        choices=["peft", "jammi"],
        default="peft",
        help="'peft' (default): the reference's own kaiming_uniform_(a=sqrt(5)) init "
        "(bound 1/sqrt(fan_in)) — use for throughput/step-time rows. 'jammi': re-draw "
        "every lora_A matrix from jammi's own bound (sqrt(3/fan_in), ~1.73x wider) for "
        "a loss-trajectory-equivalence comparison. NOT bit-identical to jammi's init "
        "either way — see the module docstring.",
    )
    p.add_argument("--target-modules", type=str, default="Wqkv,Wo,Wi")
    p.add_argument(
        "--dtype",
        choices=["fp32", "amp-fp16", "bf16"],
        default="fp32",
        help="fp32/bf16 are straight torch_dtype casts, architecturally comparable to "
        "jammi's f32/bf16 lanes. amp-fp16 is torch.autocast + GradScaler, NOT a "
        "replica of jammi's unscaled f16 cast, and REQUIRES CUDA — see the module "
        "docstring.",
    )
    p.add_argument(
        "--attn",
        choices=["eager", "sdpa"],
        default="sdpa",
        help="Requested HF attention backend. The report's attn_implementation field "
        "is the RESOLVED value read back from the loaded model's own config, which "
        "can differ from this request. sdpa is torch's best-case number (what a #352 "
        "throughput ratio should compare jammi-fused against); eager is the semantic "
        "twin of jammi's own attention composition. Run both, state which is headline.",
    )
    p.add_argument("--cuda", type=int, default=None, help="CUDA ordinal; omit for CPU.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--batched-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Encode the three triplet groups in one forward (the trainer's shape). "
        "--no-batched-forward measures the three-forward shape for an A/B.",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Triplet margin. jammi's tier hardcodes 0.3 and does not expose this on "
        "its own CLI; the default here matches it.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build a tiny random-init 2-layer ModernBERT, save it to a temp dir, and "
        "reload it through the SAME AutoConfig/AutoModel.from_pretrained loader the "
        "real GPU path uses, then run 2 steps on CPU — so this script's own logic "
        "(including the loader, dtype map, and attention selector) is testable "
        "without a GPU or a real checkpoint. Overrides batch/seq/steps/warmup to "
        "small CPU-safe values; --dtype/--attn/--lora-* are honoured as given "
        "(subject to the amp-fp16-requires-CUDA guard).",
    )
    args = p.parse_args(argv)
    if not args.dry_run and not args.model_dir:
        p.error("--model-dir is required unless --dry-run")
    # Range guards apply UNCONDITIONALLY, including under --dry-run: a
    # nonsensical raw value (e.g. --lora-rank 0) should fail loudly at parse
    # time regardless of whether the dry-run path goes on to override
    # batch/seq/steps/warmup internally.
    if args.steps < 1:
        p.error("--steps must be >= 1")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.batch < 1:
        p.error("--batch must be >= 1")
    if args.seq < 1:
        p.error("--seq must be >= 1")
    if args.lora_rank < 1:
        p.error("--lora-rank must be >= 1")
    if not (0.0 <= args.lora_dropout < 1.0):
        p.error("--lora-dropout must be in [0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
