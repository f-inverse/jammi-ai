# The PyTorch/PEFT references

Four scripts, one per unit `jammi-bench` measures, each an ORACLE and not a
dependency — pure Python against the public `transformers` + `peft` APIs:
`torch_finetune_step.py` (one optimizer step's cost — this file's first
sections), `torch_finetune_run.py` (a whole fine-tune: learning, speed and
space — "the whole-run twin"), `torch_encode.py` (the serving side, the
`torch` rung of the `encode` ladder) and `torch_grad_oracle.py` (one
backward's direction). "What this is not" and "Install" hold for all four.

## `torch_finetune_step.py` — the step twin

`torch_finetune_step.py` measures the
same unit as `jammi-bench finetune-step`
(`crates/jammi-bench/src/finetune_step.rs`) — one LoRA optimizer step on
ModernBERT: three encoder forwards (anchor/positive/negative, all live on the
tape at once), a cosine-margin triplet loss over L2-normalized mean-pooled
embeddings, one backward into the LoRA tensors, one AdamW step — so the two
can be compared step-for-step on the same box.

## What this is not

* Not a Cargo dependency. `torch`/`transformers`/`peft` never appear in any
  crate's `Cargo.toml`, and these scripts are never invoked from CI (`torch` is
  not on the CI image).
* No requirements-pinning file lives next to it. A `requirements.txt` or
  `pyproject.toml` here is exactly the kind of file a CI job could later pick
  up and start enforcing against a crate that otherwise has no Python
  toolchain. The versions this script was developed against are recorded
  below and are also captured live in every report's `provenance` block
  (`torch_version`, `torch_cuda_version`, `transformers_version`,
  `peft_version`, `fast_path_globals`) — the report is the authority on what
  actually ran, not this file.
* Does not name or vendor any specific consumer's script. This reference is
  generic to jammi-bench's own `finetune-step` tier; it carries no
  consumer-specific data shape, model name default, or hyperparameter
  provenance beyond what `finetune_step.rs` itself hardcodes or exposes.

## Install

Developed against, and locally exercised via `--dry-run` in a fresh `uv`
venv (CPU only, no GPU/no real checkpoint available in that environment):

```
python3 ci/scripts/perf/torch_venv.py --provision
```

That verb is the one place the venv is made: it builds `TORCH_VENV` (default
`.venv-torch-ref`) from the interpreter running it and installs the reference
packages into it, reusing a venv that already imports them and refusing, by
name, an interpreter they cannot be installed for.

`pyarrow` and `usearch` are `torch_encode.py`'s: it reads and persists Parquet,
and `--ann-index` builds the graph the engine's sink builds.

Versions actually installed and run in that venv (recorded here because this
is what was verified, not a guess):

```
torch==2.13.0  transformers==5.15.1  peft==0.20.0
```

**Minimum requirement, not a suggestion:** `transformers >= 4.48.0` — that is
the release ModernBERT (`ModernBertConfig`/`ModernBertModel`) shipped in
(2025-01-10); no earlier `transformers` can run this script. Every report's
`provenance` block records the versions that
actually produced that report; treat that block as authoritative over
anything in this README or the script's own docstring.

On a real GPU run, pick the `torch` build that matches the pod's CUDA driver
by naming its index to pip the way pip reads it
(`PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 python3
ci/scripts/perf/torch_venv.py --provision`).

## Usage

Against a real ModernBERT-large checkpoint directory (config.json + weights,
loadable by `transformers.AutoModel.from_pretrained`), on a rented GPU pod:

```
python3 torch_finetune_step.py \
    --model-dir /path/to/ModernBERT-large \
    --batch 8 --seq 128 --steps 20 --warmup 5 \
    --lora-rank 16 --lora-alpha 32 --lora-dropout 0.05 \
    --target-modules Wqkv,Wo,Wi \
    --dtype bf16 --attn sdpa --seed 42
```

Run it twice per config — once with `--attn eager`, once with `--attn sdpa`
— and record both rows: `sdpa` is torch's best-case number (what a
throughput ratio should compare `jammi-fused` against); `eager` is the
semantic twin of jammi's own attention composition (no fused SDPA kernel).
The report's `finetune_step.attn_implementation` field is the RESOLVED
backend read back from the loaded model's own config — not an echo of
`--attn` — so a silent fallback (e.g. `sdpa` unavailable for a given
config/device, HF falls back to `eager`) shows up in the report rather than
being hidden behind the flag you passed.

**"torch's best" must be RECORDED, not assumed — flash requires a null
mask.** PyTorch's flash attention kernel is categorically ineligible with
ANY non-null `attn_mask` (`aten/src/ATen/native/transformers/sdp_utils_cpp.h`,
`check_for_attn_mask`: `"Flash Attention does not support non-null
attn_mask"`); on an A100, a call carrying a padding mask dispatches to
`EFFICIENT_ATTENTION` instead (cuDNN is only preferred on sm90+). This
script's synthetic batches are UNPADDED (`mask` is all-ones, no real
padding), and current `transformers`' `create_bidirectional_mask` — via
`masking_utils._ignore_bidirectional_mask_sdpa`, which returns `True`
("skip mask creation, pass `None` to sdpa") whenever the 2D padding mask is
`None` or `.all()` — drops that all-ones mask to `None` before
`ModernBertModel`'s own `sdpa_attention_forward` call ever sees it. So the
`sdpa` row here may genuinely dispatch to FLASH, a result a PADDED
real-world batch would not reproduce (it would land on `EFFICIENT_ATTENTION`
or, on sm90+, `CUDNN_ATTENTION`). `finetune_step.sdpa_backend_probe`
(CUDA + `--attn sdpa` only; `"n/a (cpu)"` elsewhere, `"n/a (attn=eager, ...)"`
under `--attn eager`) records this empirically, via two independent probes:
a single real forward wrapped in `torch.nn.attention.sdpa_kernel([<one
backend>])` per backend (`"ok"` if it completes, `"ineligible: <message>"`
if torch raises — the raise itself is the signal, never inferred), and a
`torch.backends.cuda.SDPAParams` built from representative
`(q, k, v, mask=None, dropout_p, is_causal=False, enable_gqa=False)` tensors
fed to `can_use_flash_attention`/`can_use_efficient_attention`/
`can_use_cudnn_attention(params, debug=True)`. Never read a "torch's best"
headline number without first reading this field to know which kernel
actually ran.

No GPU, no checkpoint, still exercises the REAL loader:

```
python3 torch_finetune_step.py --dry-run
```

This builds a tiny random-init 2-layer ModernBERT, `save_pretrained`s it to a
temp dir, and reloads it through the SAME `AutoConfig`/
`AutoModel.from_pretrained` + dtype-map + `attn_implementation` code path the
real GPU run uses (`load_model` never branches on `--dry-run`) — so the
loader, the dtype dict, and the attention selector are exercised for real,
not bypassed. `--dtype`/`--attn`/every `--lora-*` flag are honoured as given
under `--dry-run` (only `--batch`/`--seq`/`--steps`/`--warmup` are forced
small); `--dtype amp-fp16` under `--dry-run` is a hard error (see below), not
silently downgraded.

## The venv, the device and the checkpoint are checked before any leg

`ci/scripts/perf/torch_venv.py --provision` builds the venv from the host's own
interpreter and, on a box with an NVIDIA driver, installs torch from the newest
PyTorch wheel index whose CUDA version the driver supports (read from
`nvidia-smi`'s banner): a wheel built for a newer CUDA than the driver offers
imports cleanly, reports `torch.cuda.is_available() == False`, and would run
every reference leg on the CPU. On such a box a venv whose torch cannot see the
device is not usable, and provisioning ends with `--preflight`: one real
forward and backward of a tiny model on `cuda:0` through this directory's own
step script, which also exercises whatever the framework JIT-compiles on first
use — Triton's C shim needs the interpreter's development headers, which the
CUDA CI image carries (`python3.12-devel`).

Every script here takes its device from one function, `pick_device`: the CPU
when no `--cuda` was given, the asked-for device otherwise, and a refusal by
name — never the CPU — when that device cannot be used. `--dry-run` with
`--cuda N` runs its tiny random-init model on the device.

`ci/scripts/perf/checkpoint_files.py DIR` names what a checkpoint directory
lacks — `config.json`, `model.safetensors` (checked against its own header, so
an interrupted download is a finding) and `tokenizer.json` — and the ladder's
producers call it before any leg.

## Argument mapping and deliberate divergences from `finetune_step.rs`

| flag | mirrors | notes |
| --- | --- | --- |
| `--model-dir`, `--batch`, `--seq`, `--steps`, `--warmup`, `--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--target-modules`, `--cuda`, `--seed`, `--batched-forward` | same-named/same-shaped Rust CLI flags | argument-for-argument |
| `--dtype` | `--backbone-dtype` | **renamed on purpose.** `fp32`/`bf16` are straight `torch_dtype=` casts, architecturally comparable to jammi's `f32`/`bf16` lanes. `amp-fp16` is `torch.autocast` + `GradScaler` (fp32 master weights) — idiomatic PyTorch AMP, NOT a replica of jammi's `f16` lane, which casts the whole backbone (weights and activations) to fp16 and runs **unscaled**. Pure unscaled fp16 training is numerically fragile (AdamW's `eps=1e-8` underflows in fp16); no serious PyTorch training loop runs fp16 without loss scaling, so making the reference "match" jammi's unscaled cast would produce a number that misrepresents how anyone would actually run fp16 in torch. `amp-fp16` REQUIRES a CUDA device (`torch.autocast(device_type="cuda", ...)`); requesting it on CPU (including under `--dry-run`) is a hard `ValueError`, never a silent relabel to a dtype that did not run. This is a stated divergence, not a bug: `f16`-vs-`f16` is not a supported comparison between the two harnesses; `bf16`-vs-`bf16` and `fp32`-vs-`fp32` are. |
| `--attn` | *(none in jammi — new)* | `eager`/`sdpa`, the REQUESTED HF attention backend (recorded as `args.attn_requested`). The report's `finetune_step.attn_implementation` is the RESOLVED value read from `model.config._attn_implementation` after loading, falling back to the string `"absent"` (never to `args.attn`) if that attribute somehow does not exist — so a silent HF fallback, or a missing attribute, is visible in the report rather than papered over by echoing back the request. jammi's tier has no such axis (it has its own attention composition); run both, state which is headline. |
| `--margin` | *(none in jammi — new)* | jammi's tier hardcodes `0.3` in `triplet_loss(&a, &p, &n, 0.3)` and does not expose it on its own CLI. This script defaults to the same `0.3` so the default-vs-default comparison is unaffected; the flag exists so an operator can sweep it without editing the script. |
| `--lora-init` | *(none in jammi — new; see below)* | `peft` (default) or `jammi`. Controls the LoRA `A` matrix's initial distribution. See "LoRA init" below — this is NOT a cosmetic flag, the two inits differ by a ~1.73x bound factor. |
| `--max-grad-norm` | `--max-grad-norm` | See "The trainer-shaped step: `--max-grad-norm`" below. Both sides absent-by-default (clip OFF); when supplied, torch runs `torch.nn.utils.clip_grad_norm_(trainable, max_norm)` after `backward()` (after `scaler.unscale_` under AMP) and jammi runs the production `clip_gradients` at the same point. `max_grad_norm` is an identity field of the `train-step` workload (`TrainStepPayload::IDENTITY_FIELDS`, where `null` is a value meaning "off"), so the ladder refuses an edge whose two legs differ; each leg also reports `clip_invocations`, the counted number of clip calls. |
| `JAMMI_KERNELS_DISABLE=attention_block` (env) | `--attn` | `attention_arm` — the attention REFERENCE CLASS a leg was ASKED to run (`"eager"` or `"fused"`) — is a shared identity field too: torch derives it from the RESOLVED `_attn_implementation` (`eager` → `"eager"`, `sdpa`/flash/flex → `"fused"`), jammi from the operator's resolved `JAMMI_KERNELS_DISABLE` request (an attention base — `attention_block`, `attention_block_flash`, `all` — in `kernels_disabled_requested` → `"eager"`, else `"fused"`). Deliberately NOT the dispatch counters: those read eager on a by-design domain decline (`head_dim != 64`, `seq > 4096`, dtype/contiguity/mask), a measurement `fused_proof` already owns. The ladder refuses a jammi-eager ↔ torch-sdpa pairing — the "two references, never mixed" rule as a checked premise. The raw strings/counters stay in provenance. |
| *(n/a)* | `ln_fused_dispatches`/`ln_eager_dispatches`/`rope_fused_dispatches`/`rope_eager_dispatches`/`softmax_fused_dispatches`/`softmax_eager_dispatches`/`geglu_fused_dispatches`/`geglu_eager_dispatches`/`gelu_fused_dispatches`/`gelu_eager_dispatches`/`lora_epilogue_fused_dispatches`/`lora_epilogue_eager_dispatches`/`attention_block_fused_dispatches`/`attention_block_eager_dispatches`/`adamw_fused_dispatches`/`adamw_eager_dispatches` | Not reported here — those are jammi's own fused-kernel dispatch counters (`jammi_kernels::ops::LayerNormFused`/`RopeFused`/`SoftmaxLastDimFused`/`GegluFused`/`GeluErfFused`/`ScaledCastAdd`/`AttentionBlockFused`/`adamw_step_fused_t`); there is no equivalent concept on the torch side (`--attn` is the closest analogue for attention, and torch's own kernel dispatch inside `sdpa`/`eager` is not independently observable through the public API this script is restricted to). **Honest note on `attention_block_*`:** `AttentionBlockFused`'s domain is fixed at `head_dim == 64` (`jammi_kernels::ops::ATTENTION_BLOCK_HEAD_DIM`) — on any checkpoint whose `hidden_size / num_attention_heads != 64`, the admission predicate refuses by domain (`"head_dim_is_attention_block_fixed_head_dim"`) on every call, so the pair reads `attention_block_fused_dispatches: 0` / `attention_block_eager_dispatches: N` (`N` = the number of attention calls the step made) even on a run whose OTHER fused counters (`ln`/`rope`/`softmax`/`geglu`/`lora_epilogue`) are non-zero. That all-eager reading is the predicate working as designed, not a broken fused path — never read `0` fused dispatches here as evidence the kernel is unreachable in general; check the checkpoint's `head_dim` first — this restriction is shared: BERT and DistilBERT admit the SAME fused whole-attention-block kernel through the SAME `head_dim == 64` predicate, so this note applies identically across every architecture this tier supports, never by architecture name. **Honest note on `gelu_*`:** `GeluErfFused` (admit key `gelu_erf_fused`) is wired ONLY at BERT's and DistilBERT's FFN activation call sites — ModernBERT's FFN is GeGLU (counted in `geglu_*` above, whose internal `gelu_erf` composition step is a SEPARATE thing this pair never counts), so a ModernBert leg reads `gelu_fused_dispatches: 0` / `gelu_eager_dispatches: 0` by construction, not by domain decline. `adamw_fused_dispatches`/`adamw_eager_dispatches` are the forced-arm A/B's production switch: `JAMMI_KERNELS_DISABLE=adamw_step_fused` forces every `AdamW::step` call this run onto the eager arm (see `jammi_ai::fine_tune::adamw::AdamW::step`'s doc). |

## The trainer-shaped step: `--max-grad-norm`

A `finetune-step` row without this flag measures a step the product does
not run. The shipped trainer's `FineTuneConfig` defaults
`max_grad_norm` to `1.0`
(`crates/jammi-wire/src/fine_tune.rs`'s `default_max_grad_norm`), and
`fine_tune::trainer::TrainingLoop::process_batch_loss` always calls
`jammi_ai::fine_tune::optimizer::clip_and_step` — `clip_gradients` then the
optimizer step — at every accumulation boundary. `clip_gradients` computes
the global L2 norm entirely on device — a fixed left-to-right fold over
`trainable_vars` (`n` × `sqr` + `sum_all`, `n - 1` adds, then `sqrt` +
`affine` + `recip` + `affine` + `minimum` for the coefficient, then `n` ×
`broadcast_mul` to rescale every gradient: `4n + 4` device ops, zero
`to_scalar`/`to_vec` calls) — a device-op cost that never appeared in a
`finetune-step` row with `--max-grad-norm` omitted, because omitting it also
skipped those ops.

`--max-grad-norm <f32>` runs that same production `clip_gradients` at the
same point in the sequence the trainer does — after `backward()`, before the
optimizer step (`finetune_step.rs`'s step loop, mirroring
`trainer.rs`'s `process_batch_loss`). Omit the flag (the default) to measure
the no-clip step — the isolated no-clip reference point. Supply it
(`--max-grad-norm 1.0` to match the trainer's own default) to measure the
step the product actually runs; the delta between an on row and an off row
on the *same box* is the device-side clip's cost (the `4n + 4` device ops
above), measured rather than assumed.
Report field: `finetune_step.max_grad_norm` — `null` when the flag was
absent, the numeric value when present (never omitted from the JSON object
either way, so a report is never ambiguous about which step it measured).

A non-finite or `<= 0.0` `--max-grad-norm` is a typed refusal
(`InvalidMaxGradNorm`) rather than a silent no-clip run: `clip_gradients`
itself treats `max_norm <= 0.0` as "disable clipping" (its own documented
convention, shared with the trainer's config), which is correct for the
*absent* flag but would be a lie for a row an operator explicitly labeled
"clip on".

## LoRA init: NOT a match by default

peft's default (`init_lora_weights=True`, what `--lora-init peft` uses)
draws `A` from PyTorch's `kaiming_uniform_(a=sqrt(5))`, whose bound is
`1 / sqrt(fan_in)`. jammi's `LoraInitMode::ZerosB` (what `finetune_step.rs`
always builds with) draws `A` from
`jammi_lora::seeded::kaiming_uniform_fill`, whose bound is `sqrt(3 / fan_in)`
— **`sqrt(3)` (~1.73x) WIDER**. `B` is zero-initialized in both, so that half
already matches without any flag.

Measured directly (`fan_in = 32`, a tiny test config): peft's `max|A|` was
`0.1768`; jammi's bound is `0.3062` (and a re-drawn `A` under that bound
measured `max|A| = 0.3059`, consistent with a uniform draw near its bound).
Never claim these two inits are the same distribution without this flag.

`--lora-init jammi` re-draws every `lora_A.default.weight` tensor from
jammi's bound (`reinit_lora_a_jammi_distribution` in the script) — use this
for a loss-TRAJECTORY-equivalence comparison, where the adapter's starting
point matters. Use the default `peft` for throughput/step-time rows, where
the initial values are irrelevant to what is being measured. The function
ASSERTS it matched at least one `lora_A` tensor and returns the count,
reported as `finetune_step.lora_a_tensors_reinitialized` (`null` when
`--lora-init peft`, since nothing was re-drawn) — a silent no-op (peft
changing its own parameter-naming convention out from under the
`.endswith("lora_A.default.weight")` match) must fail loudly, not quietly
leave every `A` matrix at peft's own init while the report still says
`lora_init: "jammi"`.

**Even under `--lora-init jammi`, the two adapters are NOT bit-identical.**
jammi draws each `A` from an independent SplitMix64 stream keyed by
`(seed, fully-qualified parameter name)` (`jammi_lora::seeded::seed_for_param`)
— invariant to construction/iteration order. This script draws from torch's
own default generator (seeded once via `torch.manual_seed(seed)`, called
BEFORE model/adapter construction so the draw is deterministic given
`--seed`), advanced sequentially in whatever order `named_parameters()`
yields. Only the DISTRIBUTION (uniform family, same bound) is matched —
never the bits. Do not build a bit-identical-adapter test on top of this
flag; build a distribution/trajectory-equivalence test instead.

## Peak VRAM: one instrument for every rung, and torch's own counters beside it

Both producers report `peak_vram_bytes` the same way: a background `nvidia-smi
--query-gpu=memory.used` poll every 25 ms over the whole step loop, minus a
baseline read once after the model, adapter and optimizer are resident — the
ladder's space axis reads this one field on every rung. The script also
reports torch's own allocator counters (`peak_vram_delta_bytes`,
`peak_vram_absolute_bytes`, `peak_vram_baseline_bytes`) as provenance; the
rest of this section is what they mean and why they are not the comparable
figure.


`finetune_step.rs`'s `VramSampler` polls whole-device memory via `nvidia-smi`
(`nvidia_smi_memory_used`) on a background thread every 25ms
(`VramSampler::start`) over the ENTIRE step loop (warmup + measured), then
subtracts a baseline snapshot (`peak.saturating_sub(baseline)`,
`VramSampler::finish`) read once, right after the model+optimizer are built
(before the loop starts) — see `vram_baseline` in `run_with`.

**Sampling point matters.** Polling `torch.cuda.memory_allocated()` once per
step, at the same point the clock stops — i.e. AFTER `backward()` +
`optimizer.step()` + the `.item()` sync — reads the one instant in each step
where every saved activation has already been freed. Measured directly:
such a poll captured 403 KiB of a 9087 KiB in-step peak (~4.4%),
systematically, on every measured step — a discrete poll phase-locked to the
step's deterministic TROUGH, not its peak. This script does not poll per
step. A second, independent asymmetry: torch's `AdamW` allocates its
`exp_avg`/`exp_avg_sq` moment
tensors LAZILY, on the first `optimizer.step()` call (measured: 0 optimizer
state tensors before that first step, 48 after, on a tiny test model) —
while candle's `AdamW::new` allocates them EAGERLY, before jammi's own
baseline is read. A baseline taken right after `torch.optim.AdamW(...)`
returns (before any step) would therefore NOT yet include the moments, and
their one-time first-step allocation would land inside the measured delta
instead of being absorbed into the baseline the way jammi's is.

Both are handled the same way: this script runs ONE UNTIMED
optimizer step (forward + backward + `optimizer.step()`, via the internal
`_step_once` helper — not counted in `--warmup`/`--steps`, never part of any
reported timing) immediately after the model+optimizer are built, BEFORE
taking the VRAM baseline snapshot or resetting the peak tracker. This forces
torch's lazy moments into existence first, the honest equivalent of
candle's eager allocation. (Side effect, stated plainly: the model has
therefore already taken one real gradient step before the officially
reported `--warmup` step 0 begins. This does not affect any reported number
— activation shapes and optimizer-state sizes do not depend on the weights'
actual values, and this script never reports or interprets the loss value
itself.)

With that baseline point established, BOTH VRAM fields come from
torch's own CONTINUOUS allocator high-water mark — `torch.cuda.reset_peak_memory_stats()`
called ONCE right after the untimed warm-up step (i.e. right before the
timed warmup+measured loop starts, matching the window jammi's sampler
covers), then `torch.cuda.max_memory_allocated()` read once after the loop
ends. A continuous tracker cannot miss an intra-step spike the way ANY
discrete poll can — a per-step read, or jammi's own 25ms
`nvidia-smi` interval:

* **`peak_vram_delta_bytes`** — the field COMPARABLE to jammi's
  `peak_vram_bytes` column: same window (the entire warmup+measured loop),
  same baseline convention (`memory_allocated()` snapshot taken after
  model+optimizer construction AND after the one untimed moment-warmup step,
  recorded separately as `peak_vram_baseline_bytes`). Computed as
  `max_memory_allocated() - peak_vram_baseline_bytes` after the loop.
  **RESIDUAL ASYMMETRY, stated rather than papered over:** this is a
  CONTINUOUS allocator high-water mark; jammi's is a 25ms-interval discrete
  poll. `peak_vram_delta_bytes` may therefore legitimately read HIGHER than
  jammi's `peak_vram_bytes` even when the underlying activation footprint is
  identical — purely a sampling-method artifact, not a real workload
  difference. Do not read a gap between the two columns as a regression
  without first checking which direction this asymmetry would push it.
* **`peak_vram_absolute_bytes`** — the SAME continuous high-water mark, over
  the SAME window, WITHOUT the baseline subtraction: raw bytes live at the
  peak (model weights + LoRA adapters + optimizer moments + the peak
  activation footprint). No jammi analogue (jammi only ever reports the
  baseline-subtracted figure); useful on its own ("how much device memory
  did this configuration actually need"), never as a substitute for
  `peak_vram_delta_bytes` in a jammi comparison.

On CPU (including `--dry-run`), all three VRAM-family fields
(`peak_vram_baseline_bytes`, `peak_vram_absolute_bytes`,
`peak_vram_delta_bytes`) report `value: null` — every `torch.cuda.*` call in
this path is guarded behind `if is_cuda:`, so a CPU run never touches the
CUDA allocator API and never errors on a machine with no GPU.

Both are `torch.cuda.memory_allocated`-family figures (bytes the allocator
handed to live tensors), not `torch.cuda.memory_reserved` (bytes the caching
allocator holds, whether or not currently assigned to a tensor) — the closer
analogue to `nvidia-smi`'s whole-device reading would be
`max_memory_reserved`; if a future consumer needs that figure, add it as a
third field rather than replacing either of these.

## Fast-path globals: pinned and recorded

`pin_fast_path_globals()` runs at the start of every invocation (harmless on
CPU — these are no-ops there) and sets: `torch.backends.cuda.matmul.allow_tf32
= False`, `torch.backends.cudnn.allow_tf32 = False`,
`torch.backends.cudnn.benchmark = False`,
`torch.set_float32_matmul_precision("highest")`. The resulting state is read
back (not just assumed) into `provenance.fast_path_globals`. Without this, an
`sdpa` row could be silently riding on TF32 matmuls or a cudnn-autotuned
algorithm — fast paths jammi's own (uncompiled, non-TF32) kernels never get
to use — turning a kernel-fusion comparison into a "did torch's fast-math
flags happen to be on" comparison instead.

The HF config loader is also asked for `reference_compile=False` (HF
self-enables `torch.compile` on ModernBERT's MLP/embeddings when `triton` is
importable; an unrequested compiled reference vs. jammi's uncompiled kernels
would not be a fair comparison), via `load_config_reference_compile_off`,
guarded: on the `transformers==5.15.1` version this script was tested
against, passing an unrecognized kwarg through `AutoConfig.from_pretrained`
for `ModernBertConfig` specifically did NOT raise (the kwarg is silently
absorbed and ignored since it is not a declared field) — but the raw
`ModernBertConfig(...)` CONSTRUCTOR (not `from_pretrained`) was observed to
raise on an unrelated internal rope-parameter validation bug when passed
certain kwarg combinations, on this same transformers version. The guard
exists so that if a `transformers` version ever makes the
`from_pretrained(..., reference_compile=False)` call itself raise, this
script falls back to the plain call rather than failing the whole run.

**Whether the pin took is never inferred from "the call didn't raise."** A
boolean set because `from_pretrained` didn't raise reads `True` in exactly
the case where the kwarg was silently dropped
(`hasattr(cfg, "reference_compile")` is `False` immediately after that
"successful" call), which is precisely backwards. The report instead
carries two RESOLVED readbacks, taken directly off `model.config`:

* `finetune_step.reference_compile_resolved` — `getattr(model.config,
  "reference_compile", "absent")`, read right after `AutoModel.from_pretrained`
  returns.
* `finetune_step.reference_compile_after_first_forward` — the same read,
  repeated after the one untimed warm-up forward (see the VRAM section
  above) — on some `transformers` 4.48-4.5x releases an internal
  `_maybe_set_compile` hook mutates this field only once a forward actually
  runs, so the pre-forward and post-forward readings can legitimately
  differ.

Both fields can be `True`, `False`, or the string `"absent"` — `"absent"`
means the installed `transformers` version has no such field on
`ModernBertConfig` at all, which also means there is no compile-on-forward
risk to guard against on that version in the first place. On
`transformers==5.15.1` (this script's tested version) both fields read
`"absent"`.

`torch.optim.AdamW` is constructed with `foreach=False`, pinning it to
torch's per-tensor update path — the closer structural peer of
`candle_nn::AdamW`'s own per-tensor step loop. Torch's multi-tensor
(`foreach=True`) default is a real fast path jammi's optimizer does not have;
comparing against it belongs to a "torch's absolute best" row, out of scope
for the matched-work comparison this script targets. `args.adamw_foreach` in
every report is READ BACK from `optimizer.param_groups[0]["foreach"]` after
construction, not the literal `False` passed in — torch is free to normalize
or override a per-group default, so the report records what the optimizer
actually holds, not what was requested.

## Synthetic data

Token ids are generated by the identical 64-bit LCG `finetune_step.rs` uses
(`synthetic_ids` in both files) — for the same `--seed` and vocab size, the
two scripts feed literally the same integers to their respective encoders.
This does not make the two frameworks bit-identical (different backbone
implementations, different kernel libraries, different reduction orders);
it removes "the input data differed" as a variable in why any two numbers
differ. Attention masks are all-ones (`batch, seq` of real tokens, no
padding) — matching `finetune_step.rs`, which never exercises the
all-padding edge case this tier's `--seq`/`--batch` combination cannot
reach.

## Seeding order

`torch.manual_seed(args.seed)` runs BEFORE any model/adapter/checkpoint
construction — before `--dry-run`'s own donor-checkpoint build, before
`load_model`, before `wrap_lora` — because peft's default LoRA init draws
`A` from torch's global generator at `get_peft_model` time; seeding after
that call would leave the adapter
init unseeded by `--seed`; `--lora-init jammi`'s re-draw also depends on
this ordering for its own determinism. Seeding before the `--dry-run` donor
checkpoint's own random init means the WHOLE random-draw pipeline is
deterministic given `--seed`, not just the parts downstream of the model
load. This gives "deterministic given `--seed` and this call order" — a
weaker guarantee than jammi's, which is `(seed, parameter name)`-keyed and
provably independent of any construction or iteration order (see the LoRA
init section above); the two determinism guarantees are not equivalent and
this script does not claim otherwise.

## Report shape

The script prints one JSON document to stdout: `provenance` (GPU name via
`torch.cuda.get_device_name`, driver via `nvidia-smi --query-gpu=driver_version`,
`torch`/`torch.version.cuda`/`transformers`/`peft` versions, `fast_path_globals`,
UTC date, `git rev-parse HEAD` of this repo), `args` (every resolved CLI
argument, including `adamw_foreach` read back from the constructed optimizer
and `moment_warmup_step_executed`), and `finetune_step` (`p50`/`mean`
s/step, `steps/s`, `triplets/s`, peak RSS, the three peak-VRAM fields above,
the RESOLVED `attn_implementation` (or `"absent"`), `sdpa_backend_probe`
(see "torch's best" note above — the empirical flash/efficient/cudnn
eligibility probe, `"n/a (cpu)"` off CUDA, `"n/a (attn=...)"` off `--attn
sdpa`), the two `reference_compile_*` readbacks, and
`lora_a_tensors_reinitialized`) — field names chosen to line up with
`TrainStepPayload` in `crates/jammi-bench/src/report.rs` wherever the
concept is the same. No number in this report is asserted or gated inside
the script; it is a measurement to be read alongside jammi's own JSON
report by whatever process consumes both (e.g. an A/B table).

## Range guards

`--steps >= 1`, `--warmup >= 0`, `--batch >= 1`, `--seq >= 1`,
`--lora-rank >= 1`, `--lora-dropout` in `[0, 1)` are all checked
UNCONDITIONALLY in `parse_args` — including under `--dry-run`. `--dry-run`
overrides `--batch`/`--seq`/`--steps`/`--warmup` to small internal constants
AFTER argument parsing, inside `run()`; the guards reject a nonsensical raw
CLI value (e.g. `--dry-run --steps 0`) at parse time regardless of whether
that value would go on to be overridden, so a typo doesn't silently pass
just because `--dry-run` happened to make it irrelevant.

## `torch_encode.py` — the `torch` rung of the `encode` ladder

The `encode` workload is the engine's serving path: rows in a Parquet table →
one L2-normalized vector per key, persisted. `jammi-bench encode-step`
produces its engine rungs — `direct` (the loaded model called on the rows, no
plan), `plan` (the DataFusion plan at one partition), `plan-partitioned` (at N)
— and this script produces the `torch` rung. Every producer emits LEGS and
decides nothing; `jammi-bench ladder encode <legs-dir>` compares each adjacent
pair of rungs on speed, space and outcome. So this script's job is to do the
SAME work as the engine rung it sits beside, and to say exactly what it did.

### What every rung shares, and how that is checked

| | engine rungs | this script | on every leg |
| --- | --- | --- | --- |
| rows | write `corpus_<rows>.parquet` into `--exchange-dir`, serve from that file | read that file | `corpus_sha256`, the file's bytes |
| tokenizer, truncation | `TokenizerWrapper` over the checkpoint's `tokenizer.json`, batch-longest padding, truncated at the loaded model's `max_sequence_length` | `tokenizers.Tokenizer.from_file` — the same Rust library through its Python binding — same padding, truncated at `max_position_embeddings` | `token_lengths_sha256` (every row's real token count), `tokens`, `max_sequence_length` |
| checkpoint | `--model-dir`, or the compiled-in fixture left in `<exchange-dir>/model` | `--model-dir` | `checkpoint_{config,weights,tokenizer,pooling}_sha256`, `checkpoint_weights_size_bytes` |
| pooling, normalization | what the loaded model resolved (`resolved_pooling`), always L2-normalized | `1_Pooling/config.json` by `pooling_from_config`'s rules (absent → mean; an unrepresentable or ambiguous declaration is refused, as the engine refuses it), a port of `pooling.rs` | `pooling`, `normalize` |
| dtype, batch size | `--compute-precision` read back off the loaded model; `--batch-size` | `--dtype` (`f32`/`bf16`/`f16`, straight casts); `--batch-size` | `compute_precision`, `batch_size` |
| what one timed serve is | `plan`: one `generate_text_embeddings` call, source read → forward → **committed result table** (the Parquet object and the ANN segment the sink builds beside it); `direct`: rows → host vectors | corpus read → forward → Parquet written; with `--ann-index`, the same `usearch` graph (cosine, `f32`, default connectivity, one `add` per row on one thread) built and saved | `ann_index` on this script's legs |

These are `EncodeStepTier::IDENTITY_FIELDS` (`crates/jammi-bench/src/report.rs`),
the one declaration of what two legs must agree on; the comparator refuses legs
that differ on any of them. `seed` is carried through from `--seed` (this rung
reads the corpus, it does not generate one).

**Read `ann_index` before reading a ratio.** The engine never commits an
embedding table without its ANN segment, and for a small model that build is
the larger part of a big serve (the `plan` leg's `sink_phases` says how large).
A torch leg run without `--ann-index` stopped at "the vectors are in a file"
and did less work than the engine rung beside it.

### Two orders

`--order corpus` forwards the rows in key order, `--batch-size` at a time —
the chunks the engine's plan forwards, so the same padding: the semantic twin.
`--order length-sorted` forwards them longest-first (by character length, ties
in input order) and restores key order afterwards — what
`sentence-transformers`' `encode()` does by default, so the bar a user holds
the engine to. It pads far less; each leg's `padded_tokens` beside `tokens`
says how much. `encode_ab.sh` runs `corpus` with `--attn eager` and
`length-sorted` with `--attn sdpa` into two legs directories, so the comparator
judges the engine against each order as its own run; the RESOLVED
`attn_implementation` is on the leg.

### The leg contract

One JSON per (unit, take) under `--legs-dir`, named `torch__rows<N>__r<take>.json`,
the leg block at the top level under `encode_step`; a unit's first take has
its vectors beside it (`torch__rows<N>__r1.vectors.f32`: little-endian `f32`,
row-major, in key order — what the comparator's row-agreement outcome reads).
Every leg carries `iter_wall_s`, the wall seconds of every measured serve in
run order, never only a summary (`serve_ms_p50`/`serve_ms_min`, `rows_per_s`,
`tokens_per_s`, `model_load_ms`, `first_serve_ms` ride beside it for a reader);
`work` (its rows); `outcome_digest` (the same FNV fold `encode_step.rs`
applies, so two torch takes of one unit can be held equal); and a first and
last serve that digest differently is an error, never a leg.

### Space: one instrument per quantity, for every rung

Host memory is `peak_rss_bytes`, the kernel's resident-set high-water mark
(`VmHWM`) of the leg's own process — each (unit, take) runs in a fresh child of
this script, as each engine leg runs in a fresh child of `jammi-bench`.
Device memory is `peak_vram_bytes`, read by ONE external whole-device sampler
wrapped around the process — `jammi-bench sample-device`, which `--sampler-bin`
names — the same sampler, the same way, that wraps an engine leg. Without it
the field is unmeasured (`value: null`). Torch's own allocator high-water mark
(`peak_vram_allocator_bytes`, `max_memory_allocated` above the model-resident
baseline) is provenance: the continuous-vs-sampled asymmetry described above
for the fine-tune reference is exactly why it is never the compared quantity.

### Usage

```
jammi-bench encode-step --rung direct --rung plan --rung plan-partitioned \
    --model-dir /path/to/checkpoint --rows 16,1024,16384 --takes 2 \
    --batch-size 32 --compute-precision bf16 --cuda 0 \
    --exchange-dir /tmp/x --legs-dir /tmp/legs
python3 torch_encode.py --model-dir /path/to/checkpoint --exchange-dir /tmp/x \
    --legs-dir /tmp/legs --sampler-bin target/release/jammi-bench \
    --rows 16,1024,16384 --takes 2 --batch-size 32 --dtype bf16 --cuda 0 \
    --order corpus --attn eager --ann-index
jammi-bench ladder encode /tmp/legs
```

`ci/scripts/perf/encode_ab.sh` runs all of it: the three engine rungs
interleaved, then both torch orders, then each engine rung alone for its space
marks, and the comparator over each legs directory. `--cuda N` on a host where
torch sees no CUDA device is refused, never served on the CPU under a CUDA
leg's name.

`python3 torch_encode.py --dry-run` needs no checkpoint and no engine leg: it
serves the repository's own `cookbook/fixtures/tiny_bert` through the same
loader and code path over a small corpus it writes itself, two takes of each
of two units. The venv needs `pyarrow` (and `usearch` for `--ann-index`)
beside the packages above.

## `torch_finetune_run.py` — the whole-run twin

A THIRD script, twinning a third unit. `torch_finetune_step.py` measures one
optimizer step's cost; `torch_grad_oracle.py` one backward's direction;
`torch_finetune_run.py` trains a whole fine-tune and is the twin of
`jammi-bench finetune-run` (`crates/jammi-bench/src/finetune_run.rs`): the
same checkpoint, the same committed rows in the same order, the same batch
partition, the MNRL objective, the same hyperparameters and epoch count, early
stopping off — in `transformers` + `peft`. It writes a leg to standard output
whose `tiers.finetune_run` block carries `FinetuneRunTier`'s field names, so
one reader handles both producers.

### The pairing premise

A torch leg is paired with a jammi leg of the same seed, not merely run beside
it:

```
jammi-bench finetune-run ... --seed 3 --lora-dropout 0 \
    --work-dir jammi-work > jammi.json
python3 torch_finetune_run.py ... --seed 3 --lora-dropout 0 --lora-init zeros_b \
    --initial-adapter jammi-work/initial_adapter.safetensors > torch.json
```

Every jammi run writes its untrained adapter into its work dir as
`initial_adapter.safetensors` (safetensors, jammi's tensor names — the
interchange `grad-oracle --lora-weights-out` uses); `--initial-adapter` loads
it into the PEFT model, refusing unless the file's
tensor set equals the model's trainable set exactly, and, under `zeros_b`,
unless every `lora_b` in it is zero. Both legs record the file's digest as
`initial_adapter_sha256`. With LoRA dropout at 0 on both sides nothing random
is left unshared: the two runs differ by arithmetic only.

`--lora-init peft` keeps PEFT's own draw for cost-only runs. It records
`lora_init: "peft"`, a value no jammi leg carries, so the leg fails the
identity comparison against every jammi leg by construction.

### What is compared, and on what

* **Protocol.** `--lr`, `--epochs` and `--eval-cadence` default, on both
  producers, to the tier's own protocol (`finetune_run::DEFAULT_LEARNING_RATE`
  5e-5, `DEFAULT_EPOCHS` 4, `DEFAULT_EVAL_CADENCE` 1 — that constant's doc says
  why it is not the engine's 2e-4 over 3), the twin's literals held equal to
  the Rust constants by `test_torch_finetune_run_mirrors.py`.
* **Identity.** `--max-seq-length` defaults, on both producers, to the engine's
  own default truncation length (`jammi_wire::fine_tune::DEFAULT_MAX_SEQ_LENGTH`,
  512) and is recorded as the identity field `max_seq_length`. The torch leg carries all 39 `FINETUNE_RUN_IDENTITY_FIELDS`
  (`ci/scripts/perf/identity_fields.py`) under the same names with the same
  values, including the three realized-output digests: the held-out batch
  partition and the token batches each side fed its encoder
  (`train_token_ids_sha256`, `heldout_token_ids_sha256`). The last two are the
  token-id parity check — they cover ids, truncation, padding and bucketing,
  and they ride on the very legs being compared rather than on a separate
  run.
* **Learning.** `held_out_at_init` (the held-out example-mean at the untrained
  model, once before step 1 — the origin the learning effect is measured
  from); `held_out_example_mean`; `trajectory`, one point per evaluated
  epoch, each with `run_wall_s_cumulative` and `steps_wall_s_cumulative`
  (seconds up to that epoch's end, so time-to-a-given-loss is readable and a
  faster step cannot hide slower convergence); `train_probe_series`.
* **Speed.** `train_run_wall_s` and `epoch_walls` — each epoch's wall whole
  (`run_s`: everything `TrainingLoop::run` does for an epoch, nothing the tier
  does around it) and by the phases jammi's trainer reports for itself
  (`RunPhaseWall`): `steps_s`, the batch loop with the device synchronized at
  its end; `validation_s`; `checkpoint_s`, every checkpoint read and write.
  Comparing `steps_s` compares training compute without either side's
  checkpoint path riding along. `steps_measured` counts optimizer steps.
* **Space.** One instrument per axis, the same on both sides.
  `peak_rss_bytes` is the kernel's resident-set high-water mark for the
  process (`VmHWM`; `getrusage`'s `ru_maxrss` where `/proc` does not exist,
  with the source recorded). `peak_vram_bytes` is whole-device memory above a
  baseline, from the same `nvidia-smi --query-gpu=memory.used` poll at the
  same 25 ms interval as `crates/jammi-bench/src/vram.rs`, under the same
  baseline rule: read once the model and adapters are resident and the
  optimizer is constructed, before anything runs. AdamW state is allocated
  inside the window on both sides (jammi builds its optimizer inside the timed
  call; torch allocates lazily on the first step), so no warm-up step is
  needed — and none could be afforded, since it would be an update the other
  side never took. torch's `max_memory_allocated`/`max_memory_reserved` are
  recorded under `provenance.torch_allocator` and are not the comparable
  figure: an allocator's live-byte counter and a whole-device reading measure
  different things.

### Trainer behaviours: reproduced or different

Every behaviour of jammi's trainer, and of the tier driving it, that can move
an outcome or a cost. The script's module docstring carries the same list with
the function implementing each.

| # | Behaviour | Twin |
|---|---|---|
| 1 | Rows in file order, never shuffled | REPRODUCED |
| 2 | Validation split: the last `round(n · fraction)` rows, half rounding away from zero | REPRODUCED |
| 3 | Consecutive `batch`-sized chunks, the short last chunk kept | REPRODUCED |
| 4 | One forward per batch over anchors then positives, joined | REPRODUCED |
| 5 | Held-out rows in `--heldout-ids` order, every batch full or refuse | REPRODUCED |
| 6 | JSONL lines split on `\n` only | REPRODUCED |
| 7 | `tokenizer.json` as shipped, special tokens, truncation to `min(max_seq_length, max_position_embeddings)` | REPRODUCED |
| 8 | Right-padding with id 0 / mask 0 whatever the vocabulary's pad token is | REPRODUCED |
| 9 | Training batches padded up the bucket ladder `{8, 16, 32, …}`; evaluation at natural width | REPRODUCED under `--width bucketed` (default); DIFFERENT on purpose under `--width natural` — see "Two widths" |
| 10 | Tokenization per batch, per epoch, inside the timed span | REPRODUCED |
| 11 | Frozen backbone at `--backbone-dtype` | REPRODUCED |
| 12 | No backbone dropout in training mode | REPRODUCED — the checkpoint's dropout probabilities are forced to 0 and the overridden values recorded |
| 13 | LoRA on the same linears, rank, `alpha / rank`, no bias, no rsLoRA | REPRODUCED — refused unless the loaded adapter's tensors are exactly the trainable set |
| 14 | f32 adapter tensors under a bf16 backbone, summed in the wider dtype | REPRODUCED (PEFT's adapter autocast, which jammi's `LoraLinear` follows) |
| 15 | Initial adapter tensors | REPRODUCED bit for bit from jammi's dump; DIFFERENT under `--lora-init peft`, by design |
| 16 | LoRA dropout mask | DIFFERENT whenever `--lora-dropout > 0` (counter-keyed Philox vs torch's generator); the paired protocol runs at 0, where none is drawn |
| 17 | Mean pooling over the mask, then L2 normalization, in the working dtype | REPRODUCED |
| 18 | Symmetric MNRL with a 1e-8 norm floor, in the embeddings' dtype | REPRODUCED |
| 19 | Accumulation: loss / window size, the trailing partial window by its own size, flushed as a step | REPRODUCED |
| 20 | Divergence guard: a NaN or > 100 batch is dropped without taking a window slot; three in a row fail | REPRODUCED |
| 21 | Global-norm clip, `max_norm / (norm + 1e-6)` | REPRODUCED up to reduction order (sum of squares in name order vs norm of per-tensor norms) |
| 22 | Non-finite norm refusal on step 1, every 50th, and the run's last | REPRODUCED at that cadence |
| 23 | AdamW, decoupled decay on every trainable tensor, betas (0.9, 0.999), eps 1e-8 | REPRODUCED |
| 24 | Linear warmup from 0, then constant, cosine or linear decay over the whole run's horizon | REPRODUCED |
| 24b | The zero-learning-rate control (`--zero-lr-control`): the same job with every optimizer step applied at rate 0, reported as `lr: 0.0`; a non-positive `--lr` refused | REPRODUCED |
| 25 | Early stopping off (`patience >= 10000`) | REPRODUCED as the same refusal |
| 26 | Evaluation in eval mode, no gradient | REPRODUCED |
| 27 | Validation pass inside the timed span when monitoring `val_loss` | REPRODUCED |
| 28 | Held-out per-row NLL on the host in f32, left-to-right log-sum-exp | REPRODUCED |
| 29 | Held-out evaluation and train probe after every epoch, and both once before training at the untrained model (`held_out_at_init`), outside the span | REPRODUCED |
| 30 | `heldout_batch_partition_sha256` | REPRODUCED |
| 31 | Adapter written every `ceil(0.1 · horizon)` steps | REPRODUCED as a safetensors write at the same steps |
| 32 | Epoch boundary: finite check, best adapter, epoch bundle with both moments, best read back, final adapter | REPRODUCED as the same reads and writes |
| 33 | The epoch bundle also goes through jammi's artifact store and a catalog row | DIFFERENT — no torch analogue; written to local disk once. Charged to `checkpoint_s` on both sides and nowhere else |
| 34 | The tier takes the run one epoch per `run()` call (`epoch_limit`, full `epochs` every call), rebuilding the model and restoring adapter, moments and counters from that bundle | DIFFERENT in mechanism, identical in effect (the schedule is the uninterrupted run's and the restore is exact). The rebuild is outside every span; the restore is charged to jammi's `checkpoint_s` |

### Two widths

jammi pads training batches up a bucket ladder because its allocator wants few
distinct shapes, and its variable-length attention path does not pay for the
padding. A PyTorch user has neither reason — they pad to the batch's longest
row — and padded attention pays for every padded column. So, like `--attn
eager|sdpa` on the step twin, the run twin has two legs:

* `--width bucketed` (default): the SEMANTIC twin. The token batches are
  jammi's, digest for digest; this is the leg that pairs with a jammi leg on
  outcome.
* `--width natural`: pad to the batch's longest row — the PRACTICAL BAR for
  the speed and space axes.

`width` is an identity field of a torch leg. A natural leg's
`train_token_ids_sha256` differs from jammi's by construction (its held-out
digest does not: evaluation is at natural width everywhere), so it can never
be mistaken for the semantic twin. The loss does not depend on the width
beyond rounding — padded positions are masked out of attention and of the
pooling mean, and positions are absolute — and the cross-producer parity guard
holds a natural leg to the bucketed leg and to jammi within the same 1e-5.

### What holds it to jammi

* `test_torch_finetune_run_mirrors.py` (stdlib, beside this file): the rules
  re-implemented in Python — split boundary, bucket ladder, line splitting,
  both digests, adapter tensor names — against values jammi produced; the
  token-batch digests are the same three
  `finetune_run.rs::tests::token_batches_sha256_*` pin.
* `ci/scripts/perf/test_torch_finetune_run_dry_run.py` (torch venv): the leg a
  real `--dry-run` writes.
* `ci/scripts/perf/test_finetune_run_cross_producer_parity.py` (cargo + torch
  venv): both real producers over the committed held-out text on `tiny_bert`
  and `tiny_modernbert_classifier`, CPU, f32, the torch run started from the
  jammi run's adapter, under warmup, a cosine decay, accumulation with a
  trailing partial window, weight decay and the clip. It holds the identity
  tuple equal through the merger's own premise check, the token-batch and
  adapter digests equal, the step counts and the outcome/cost field names
  equal, and every held-out and probe loss within 1e-5 while both runs
  demonstrably learn — so an unreproduced trainer behaviour shows up as a gap,
  not as noise.

The name tables cover ModernBERT (the gradient oracle's) and BERT; the BERT
table is exercised on `query,value`. A `--target-modules` that reaches a
linear jammi does not wrap (PEFT's suffix match on `dense` also selects BERT's
pooler) is refused by the tensor-set check rather than trained.

### Producing paired legs

`ci/scripts/perf/finetune_run_ab.sh` with `FINETUNE_RUN_AB_TORCH=1` runs the
`torch` arm beside `fused` and `alloff`, order-balanced within each seed
(`fused r1, alloff r1, torch r1, torch-natural r1, torch-natural r2, torch r2, alloff r2, fused r2` — `torch` is `--width bucketed`, `torch-natural` is `--width natural`). It hands the
torch leg the same run flags it hands the jammi legs, from one array; points
it at the adapter the seed's first jammi leg wrote; and refuses before any leg
unless
`FINETUNE_RUN_AB_LORA_DROPOUT` is 0 and the objective is MNRL. The venv is the
one `ci/scripts/perf/torch_venv.py` resolves. `ab_merge.py finetune-run` reads
the `fused`/`alloff` legs only; the torch legs are evidence for a reader of
leg files and never a merge gate.

`--dry-run` builds a tiny random ModernBERT, a ten-word tokenizer and
synthetic pairs, and drives the same code path on a CPU in seconds.

## `torch_grad_oracle.py` — the jammi-vs-torch LEARNING oracle's torch side

A SEPARATE script, not a mode of `torch_finetune_step.py` (different
measurement: one forward+backward at IDENTICAL LoRA weights, no optimizer
step, no timing). See `crates/jammi-bench/src/grad_oracle.rs`'s module doc
for the full "why gradients, not loss trajectories" argument, and this
script's own module doc for the exact jammi<->PEFT tensor-name translation
table it owns (jammi's `grad-oracle` subcommand does zero translation — the
shared weight-interchange file is a plain `safetensors` file in jammi's OWN
internal naming; this script translates both directions).

**PROVENANCE — read before trusting this script's output**: it has been
run live on one config, on an A100 (ModernBERT-large, `--batch 8 --seq 128
--seed 42`, jammi at `e62c8a8`). See `torch_grad_oracle.py`'s own
module-doc PROVENANCE banner for the measured cosine similarities from that
run. Beyond that one confirmed config, everything else (other checkpoints,
`target_modules` sets, dtypes/ranks/batch/seq combinations) remains
UNVERIFIED against a live run — one successful execution is evidence the
mechanism works, not a proof it is correct everywhere this script accepts
flags for. Its NAME-TRANSLATION functions are, independently, locally
tested (`test_torch_grad_oracle_names.py`, stdlib-only, no torch needed —
that suite pins, among others, `Wi`: an MLP site whose jammi-side name
carries no `mlp.` prefix, which a naive string-prefix heuristic misroutes
to `attn.Wi`).

**Structural limitation, confirmed on that live run: a single fresh-init
call tests ONLY `dL/dB`, never `dL/dA`.** Both `grad_oracle.rs` and this
script run at `LoraInitMode::ZerosB` — `B` starts at the exact zero
matrix, and the LoRA forward's chain rule routes `dL/dA` through `B^T @
dL/d(output)`, which is IDENTICALLY zero whenever `B == 0`, for ANY value
of `A`, on BOTH stacks, REGARDLESS of whether either stack's `dL/dA`
arithmetic is actually correct. On the live A100 run, every `lora_a`
tensor's gradient measured EXACTLY `0.0` on both dumps (112 of 224
matched tensors) — a structural guarantee, not evidence the two stacks
agree on that path. See `grad_oracle.rs`'s own "Structural limitation"
doc section; the ladder's gradient-agreement outcome classifies such a
pair as *vacuous* — no evidence either way — rather than let a `0.0`
cosine there masquerade as either a pass or a fail. Catching a real
`dL/dA` defect needs at least one optimizer step first (moving `B` away
from zero), which neither script does.

Both producers emit a `train-step` leg — `tiers.finetune_step` in a
`jammi-bench` report — whose `gradients` block carries every trainable
tensor's gradient and the weight it was taken at, both as `f32`, under
jammi's tensor names. The legs are filed as
`torch__<unit>__grads.json` and `reference__<unit>__grads.json`, the
`grads` take of the `train-step` ladder's `torch -> reference` edge, and
`jammi-bench ladder train-step <legs-dir>` judges them as gradient
agreement: the two sides' weights must be the same bits (a premise, not a
tolerance); per tensor, both gradients zero is vacuous, exactly one zero
or a non-finite entry breaks the structure (a hard rule), and a real
pair's cosine is held to the `gradient_cosine_floor` budget — evidence,
and unbudgeted until a committed artifact measures one. A gradient leg
shares the edge's identity fields with its timed repeats and is free to
differ on exactly `warmup`, `steps_measured`, `lora_dropout` and
`max_grad_norm`, which a single forward has no use for. On the live A100
run above the weight premise held by actual agreement:
`max|w_jammi - w_torch| = 1.86e-9` over 224 tensors, and the jammi-f32 vs
torch-f32 cosine read 0.9999998; those readings are prose, no raw dump is
committed, so no floor has been measured yet.

## Legs for the parity ladder

A reference script is a *producer*: it emits legs and decides nothing. Every
comparison — identity, premises, speed, space, outcome — is
`jammi-bench ladder <workload> <legs-dir>`, one operator over every rung of a
workload (`docs/plans/69-parity-ladder/README.md` has the method and the leg
contract). A reference leg is filed as `torch__<unit>__<take>.json` beside the
engine's legs and carries its block at the top level under the workload's key
(`finetune_run`, `encode_step`, …) with the same field names the matching
`jammi-bench` tier emits: the workload's identity fields (declared once, in
Rust — `TrainRunPayload::IDENTITY_FIELDS`, `EncodePayload::IDENTITY_FIELDS`,
or the list in `crates/jammi-bench/src/ladder/definition.rs`; the ladder
refuses a leg that omits one or spells a value differently), `iter_wall_s`
(post-warmup seconds per timed iteration, in order), `peak_rss_bytes` and
`peak_vram_bytes` from the same instruments the engine's legs use, and the
outcome (`held_out_example_mean` with a `trajectory` of `held_out_mean` and
cumulative `train_wall_s`; or `vectors_file` + `vector_dim`, little-endian
`f32` rows in committed key order; or `law_observed` counts).

See that script's own module doc for the derived (never fitted) bf16
ULP-based cosine floor, and its `derive_cosine_floor` doc for why that
DERIVED worst-case bound (~-0.40 at ModernBERT-large's own default
`--num-layers`/`--hidden-size`) is far looser than what real bf16 noise
actually costs — the live run's measured overall cosines (torch-eager vs
torch-sdpa 0.825; torch-bf16 vs torch-f32 0.924; jammi-f32 vs torch-f32
0.9999998; a separately-introduced real defect on the same run scored
0.30-0.53) are the empirical anchor for picking a real `--cosine-floor`,
not the derived bound. See `ci/scripts/perf/test_compare_grad_oracle.py`
for its (numpy-optional) test suite.

# The graph-learning rungs — `torch_graph_sample.py`, `torch_propagate.py`, `torch_context_predictor.py`

Three more producers, one per graph-learning workload, each a PyTorch rung of a
workload whose engine rungs are `jammi-bench` leg subcommands
(`crates/jammi-bench/src/{graph_sample,propagate,context_predictor}.rs`). They
file legs exactly as the section above says — `<rung>__<unit>__r<take>.json`
under `--legs-dir`, the block at the top level under `graph_sample`,
`propagate` or `predictor_train_run`, the identity fields each payload declares
(`GraphSamplePayload`, `PropagatePayload`, `PredictorTrainRunPayload`),
`iter_wall_s`, `work`, `peak_rss_bytes` (`VmHWM`, or `getrusage`'s
`ru_maxrss` where there is no `/proc`), `peak_vram_bytes` (null: these rungs
use no device), and the outcome the workload pairs: `law_observed` against the
unit's law file, `vectors_file` + `vector_dim`, or `held_out_example_mean` with
a `trajectory` anchored by `held_out_at_init` and the `train_probe_series` the
learning premise reads. `ladder_leg.py` is the capture every script shares, the
twin of `src/capture.rs`. The comparison is `jammi-bench ladder <workload>
<legs-dir>`.

Every rung of a workload reads the **same input files**, which the engine rung
writes beside its legs. A composite workload is cut at its committed
intermediate artifact — a graph fine-tune at its pair table, a predictor
training at its episode set — so each comparison is of one thing, and sharing
the artifact across stacks removes the sampling randomness between them
instead of averaging over it.

Same rules as above: not a Cargo dependency, never invoked from CI, no
requirements file. Developed against and exercised on CPU with

```
torch==2.14.0  torch_geometric==2.8.0.post1  torch_cluster==1.6.3
pyg-lib==0.9.0+pt214  safetensors  numpy
```

`torch_cluster` and `pyg-lib` build or resolve against the installed torch, so
they go in after it:

```
uv pip install --python .venv-torch-ref/bin/python3 torch torch_geometric safetensors numpy setuptools wheel
uv pip install --python .venv-torch-ref/bin/python3 --no-build-isolation torch_cluster
uv pip install --python .venv-torch-ref/bin/python3 pyg-lib -f https://data.pyg.org/whl/torch-2.14.0+cpu.html
```

Every leg's `packages` records what actually ran; `torch_venv.py --graph`
probes the venv for them. `ci/scripts/perf/test_torch_graph_rungs.py` (the
`torch graph rungs` guard, `torch-host` lane) runs all three over tiny inputs
and holds each leg against an oracle that shares no code with it.

## `graph-sample` — node2vec walks

```
jammi-bench graph-fixture --out run/g64                      # the committed synthetic graph
jammi-bench graph-fixture --nodes-per 1024 --out run/g1024   # a larger point of a size sweep
jammi-bench graph-sample --graph run/g64 --graph run/g1024 --legs-dir run/legs \
    --walk-length 4 --walks-per-node 4 --return-p 1 --in-out-q 0.5
python3 torch_graph_sample.py --graph run/g64 --graph run/g1024 --legs-dir run/legs \
    --walk-length 4 --walks-per-node 4 --return-p 1 --in-out-q 0.5
jammi-bench ladder graph-sample run/legs --law-dir run/legs/law
```

Several `--graph` are a size sweep: each graph is sampled in its own process,
so no point inherits another's peak resident set, and cost and memory are
fitted against `edge_count`, the unit (`edges<N>`). The engine rung files
`sampler__edges<N>__r<take>.json` with its pair table beside it, and writes the
unit's law file `edges<N>.json` under `--law-dir` (`<legs-dir>/law` by
default): node2vec's law `π(x | t, v) ∝ α_pq(t, x) · w(v, x)` for that graph
(`graph_sample::node2vec_transition_law`), one cell per walk state in ascending
order with the first-step states first, the state's next nodes ascending. Every
leg counts its walks' steps in that order as `law_observed` and names the file
by its sha256 as `law_sha256` — the ground truth both rungs' counts are judged
against is the committed file, never a producer's claim; the torch rung
recomputes the law to know the order and refuses if the file is not it.

| aspect | status |
| --- | --- |
| transition law, first step uniform | REPRODUCED by `--walker torch_cluster` (the default): the engine samples the law by roulette over the reweighted neighbours, `torch_cluster` by rejection sampling |
| `torch_geometric.nn.Node2Vec`'s own walker | REPRODUCED at `p = q = 1` only. It walks through `pyg-lib`'s `random_walk`, which samples uniformly and refuses any other `p`, `q` ("Uniform sampling required for now"); `--walker node2vec` refuses likewise rather than walk a different law |
| walk length | REPRODUCED — the engine counts steps, `Node2Vec(walk_length=)` counts nodes (`+ 1`) |
| the graph | REPRODUCED — one `edges.jsonl` row is one directed edge for both rungs |
| "x adjacent to t" on a directed graph | DIFFERENT — the engine reads adjacency in either direction, the torch walkers the one direction in their CSR. Identical on a symmetric edge list; `identity.edge_set_symmetric` records which the graph is |
| a node with no out-edge | DIFFERENT — the engine ends the walk, the torch walkers stay in place. Cannot occur on a symmetric edge list |
| repeated edge rows | DIFFERENT for `torch_cluster`, which coalesces them; the engine and `Node2Vec` keep a repeated row as a heavier edge |
| seeds | DIFFERENT generators — rows never match across rungs, only their law |
| pairs | the torch pair file applies the engine's rule (anchor = walk start, one row per distinct later node) to the torch walks; `Node2Vec.pos_sample`'s context windows are DIFFERENT and not emitted |
| hard negatives | DIFFERENT — structure-aware k-hop-excluded mining has no PyG counterpart (`Node2Vec.neg_sample` is uniform). The torch rung emits none; compare against an engine leg at `--hard-negatives 0` |

`jammi-bench graph-pairs --graph <dir> --out <dir>` writes the pair table alone:
the rows a `fine_tune_graph` job at the same sampler configuration trains on, in
its `_ordinal` order, in the triplet row shape `finetune-run --train-jsonl`
reads. `cookbook/fixtures/tiny_citation_graph/` is a committed graph with
declared (citation) edges to cut one from.

## `propagate` — APPNP / SGC

```
jammi-bench propagate --nodes 1000,10000 --partitions 4 --hops 2 --alpha 0.1 --legs-dir run/legs
python3 torch_propagate.py --impl exact --legs-dir run/legs --hops 2 --alpha 0.1
python3 torch_propagate.py --impl pyg   --legs-dir run/legs --hops 2 --alpha 0.1
jammi-bench ladder propagate run/legs --to plan-partitioned
```

The engine rungs `plan` (one partition) and `plan-partitioned` (`--partitions`)
file `<rung>__edges<N>__r<take>.json` with the key-sorted propagated rows
beside (`<stem>.vectors.f32` + `.keys.txt`) and write the unit's inputs under
`input/edges<N>/` (`x0.vectors.f32` + `x0.keys.txt`, `edges.jsonl`); the torch
rungs read every unit under `input/` (or `--unit`) and file `torch__…` /
`torch-geometric__…` beside them. Pass the engine leg's `hops` — the depth
actually run, after the engine's clamp to its hop cap.

| aspect | `--impl exact` | `--impl pyg` |
| --- | --- | --- |
| adjacency (undirected set; a repeated or reversed row is one edge; a self-edge row is dropped) | REPRODUCED | REPRODUCED |
| self-loops `Ã = A + I`, `D̃^{-1/2} Ã D̃^{-1/2}` over `d̃ = deg + 1` | REPRODUCED, in the engine's order: scale, sum, scale | REPRODUCED (`gcn_norm`), folded into per-edge weights |
| recurrence `α·X⁽⁰⁾ + (1−α)·Â·X⁽ᵏ⁻¹⁾`, `α = 0` ≡ SGC | REPRODUCED | REPRODUCED (`APPNP`; `SGConv` with an identity map at `α = 0`) |
| arithmetic | REPRODUCED — `f64` fold, one final `f32` cast (`--dtype f32` is then DIFFERENT) | DIFFERENT — `f32` throughout |
| summation order | nodes are indexed in ascending key order so a CSR row is the engine's `(group, neighbour)` order; accumulation order inside `torch.sparse.mm` is the backend's | scatter order |
| an edge endpoint with no vector | DIFFERENT — the engine counts it in the degree; the script refuses the edge list | same refusal |

## `predictor-train-run` — the context predictor

```
jammi-bench predictor-train-run --arch Tnp --seeds 1,2,3,4,5,6,7,8,9,10,11,12 --legs-dir run/legs
python3 torch_context_predictor.py --legs-dir run/legs --arch Tnp --seeds 1,2,3,4,5,6,7,8,9,10,11,12 \
    --epochs 30 --learning-rate 0.005 --grad-clip 1.0 --num-heads 2 --num-layers 2
jammi-bench ladder predictor-train-run run/legs
```

The engine rung files `in-process__seed<N>__r<take>.json` per seed (a unit;
the seeded edge needs twelve) and writes the unit's inputs under
`input/<arch>/seed<N>/` — the train and held-out episodes and the seeded
initial weights; the twin reads the member off the weight file's tensor names
and refuses a name set that is not exactly one member's. The identity fields
the files do not determine (`--num-heads`, `--num-layers`, `--epochs`,
`--learning-rate`, `--grad-clip`) are the engine leg's. Both legs carry the
held-out loss at init and after every epoch (`held_out_at_init`, `trajectory`,
`held_out_example_mean`), the train-side probe series the learning premise
reads, every optimizer step's wall-clock, and the trained head's raw output on
every held-out target (`vectors_file`, keyed `test{episode}_{row}`).

| behaviour | status |
| --- | --- |
| MLP (φ, ρ, a Tnp block's MLP, the Tnp head) | REPRODUCED — `fc2(gelu(fc1(x)))`, both linears biased, exact erf-GELU |
| `Cnp` | REPRODUCED — φ over `(x ‖ y)`; mean over present members, an empty context pooling to zero; ρ over `(pooled ‖ target_x ‖ context_size)` |
| `AttnCnp` | REPRODUCED — biased `query`/`key`/`value` projections (query from `target_x`, key from `context_x`, value from `(x ‖ y)`); the learned `prior_key`/`prior_value` prepended as an always-present member 0; ρ over `(attended ‖ target_x)`, no output projection |
| `Tnp` | REPRODUCED — target token `target_embed(x) + query_marker` at position 0 then `context_embed(x ‖ y)`, no positional encoding; per block the Pre-LN pair — `attn_norm` (biased LayerNorm, eps `1e-5`) feeds biased `q`/`v` and bias-free `k`, attention, `tokens + attended`; `mlp_norm` feeds the MLP, `tokens + mlp` — no output projection; `final_norm` on position 0, then the `head` MLP |
| attention | REPRODUCED — `num_heads` contiguous slices of `hidden / num_heads`; `QKᵀ / √head_dim + mask`, softmax over keys in `f32`, `·V`, heads concatenated — written with `matmul`, never `scaled_dot_product_attention` (a fused kernel is a different operation order) |
| masking of an absent member | REPRODUCED — additive `presence · 10000 − 10000` on the key axis, never `−inf`; an absent token is still a query |
| initial weights, episodes, batch order | REPRODUCED — loaded from the engine's files; one step per train batch in file order, no shuffling, no dropout |
| objective, optimiser, clip | REPRODUCED — closed-form Gaussian CRPS of `(mean, σ = 1e-3 + softplus(raw))`; AdamW (betas `0.9, 0.999`, epsilon `1e-8`, no weight decay); global-L2 clip with `coef = min(1, max_norm / (norm + 1e-6))` |
| arithmetic | DIFFERENT, irreducibly — `f32` on both, each backend's matmul blocking, reduction order and `erf`/`exp` its own. Measured below |

Measured on the committed spec (6 train batches × 30 epochs = 180 steps, 2 test
batches; jammi from the CI image, torch 2.14.0 CPU), same `episodes_sha256` and
`initial_weights_sha256` on both legs:

| member | step-0 loss `\|d\|` | max `\|d\|` over 180 steps | final-epoch mean loss (jammi / torch) | held-out predictions, max abs / min cosine |
| --- | --- | --- | --- | --- |
| `Cnp` | 2.4e-7 | 6.7e-5 (step 176) | 0.295933 / 0.295958 | 2.9e-4 / 0.99999998 |
| `AttnCnp` (2 heads) | 6.0e-8 | 1.8e-7 (step 13) | 0.311394 / 0.311393 | 4.0e-6 / 1.0000000000 |
| `Tnp` (2 heads, 2 layers) | 6.0e-8 | 3.4e-3 (step 179) | 0.151816 / 0.152304 | 6.0e-2 / 0.99894 |

The residual on every row is rounding amplified by the member's own training
dynamics, measured the same way for each: the running maximum of `|d|` over the
180 steps, for jammi vs torch, for torch vs torch after **one ulp** in one
weight, and for torch `f32` vs torch `f64`, at the committed learning rate
`5e-3` and at two higher ones.

| member | jammi vs torch | torch vs one ulp | `f32` vs `f64` | one ulp at lr `2e-2` | one ulp at lr `5e-2` |
| --- | --- | --- | --- | --- | --- |
| `Cnp` | ×10 / 51 steps, 6.7e-5 at 179 | ×10 / 85, 9.7e-6 | ×10 / 43, 7.8e-5 | ×10 / 59, 1.3e-4 | ×10 / 28, 9.1e-3 |
| `AttnCnp` | flat, 1.8e-7 | flat, 1.2e-7 | flat, 1.6e-7 | ×10 / 47, 2.3e-5 | ×10 / 55, 3.1e-4 |
| `Tnp` | ×10 / 40, 3.4e-3 | ×10 / 40, 1.3e-3 | ×10 / 36, 2.2e-3 | ×10 / 30, 9.5e-2 | ×10 / 28, 1.4e-1 |

`Tnp`'s 37 weight tensors agree jammi-vs-torch to 2.4e-7 after 6 and after 30
steps, the same distance torch `f32` sits from its own `f64` run (2.8e-7 /
4.5e-7). The pre-norm placement is what makes the transformer member pairable
at all: the same two blocks without their norms amplified one ulp to 1.9e-1
within 179 steps (×10 every 23 steps, in `f32` and `f64` alike) and put the
weights 5.7e-5 apart after 30 steps — the measurement the norms are an
invariant against (`crates/jammi-encoders/src/context/tnp.rs`). The transformer
is still the most rounding-sensitive member at a high learning rate, as its
stacked residuals predict; the mean-pooled and attention-pooled members show the
same latent sensitivity only from lr `2e-2` up, and `AttnCnp` least of all.
