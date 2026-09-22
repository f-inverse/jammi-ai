# `torch_finetune_step.py` — the PyTorch/PEFT reference

This directory holds an ORACLE, not a dependency. `torch_finetune_step.py` is
pure Python against the public `transformers` + `peft` APIs. It measures the
same unit as `jammi-bench finetune-step`
(`crates/jammi-bench/src/finetune_step.rs`) — one LoRA optimizer step on
ModernBERT: three encoder forwards (anchor/positive/negative, all live on the
tape at once), a cosine-margin triplet loss over L2-normalized mean-pooled
embeddings, one backward into the LoRA tensors, one AdamW step — so the two
can be compared step-for-step on the same box.

## What this is not

* Not a Cargo dependency. `torch`/`transformers`/`peft` never appear in any
  crate's `Cargo.toml`, and this script is never invoked from CI (`torch` is
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
uv venv .venv-torch-ref
uv pip install --python .venv-torch-ref/bin/python torch transformers peft
```

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

Pick whatever `torch` build matches the pod's CUDA driver (`uv pip install
torch --index-url https://download.pytorch.org/whl/cu121` etc.) on a real
GPU run.

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

## Argument mapping and deliberate divergences from `finetune_step.rs`

| flag | mirrors | notes |
| --- | --- | --- |
| `--model-dir`, `--batch`, `--seq`, `--steps`, `--warmup`, `--lora-rank`, `--lora-alpha`, `--lora-dropout`, `--target-modules`, `--cuda`, `--seed`, `--batched-forward` | same-named/same-shaped Rust CLI flags | argument-for-argument |
| `--dtype` | `--backbone-dtype` | **renamed on purpose.** `fp32`/`bf16` are straight `torch_dtype=` casts, architecturally comparable to jammi's `f32`/`bf16` lanes. `amp-fp16` is `torch.autocast` + `GradScaler` (fp32 master weights) — idiomatic PyTorch AMP, NOT a replica of jammi's `f16` lane, which casts the whole backbone (weights and activations) to fp16 and runs **unscaled**. Pure unscaled fp16 training is numerically fragile (AdamW's `eps=1e-8` underflows in fp16); no serious PyTorch training loop runs fp16 without loss scaling, so making the reference "match" jammi's unscaled cast would produce a number that misrepresents how anyone would actually run fp16 in torch. `amp-fp16` REQUIRES a CUDA device (`torch.autocast(device_type="cuda", ...)`); requesting it on CPU (including under `--dry-run`) is a hard `ValueError`, never a silent relabel to a dtype that did not run. This is a stated divergence, not a bug: `f16`-vs-`f16` is not a supported comparison between the two harnesses; `bf16`-vs-`bf16` and `fp32`-vs-`fp32` are. |
| `--attn` | *(none in jammi — new)* | `eager`/`sdpa`, the REQUESTED HF attention backend (recorded as `args.attn_requested`). The report's `finetune_step.attn_implementation` is the RESOLVED value read from `model.config._attn_implementation` after loading, falling back to the string `"absent"` (never to `args.attn`) if that attribute somehow does not exist — so a silent HF fallback, or a missing attribute, is visible in the report rather than papered over by echoing back the request. jammi's tier has no such axis (it has its own attention composition); run both, state which is headline. |
| `--margin` | *(none in jammi — new)* | jammi's tier hardcodes `0.3` in `triplet_loss(&a, &p, &n, 0.3)` and does not expose it on its own CLI. This script defaults to the same `0.3` so the default-vs-default comparison is unaffected; the flag exists so an operator can sweep it without editing the script. |
| `--lora-init` | *(none in jammi — new; see below)* | `peft` (default) or `jammi`. Controls the LoRA `A` matrix's initial distribution. See "LoRA init" below — this is NOT a cosmetic flag, the two inits differ by a ~1.73x bound factor. |
| `--max-grad-norm` | `--max-grad-norm` | See "The trainer-shaped step: `--max-grad-norm`" below. Both sides absent-by-default (clip OFF); when supplied, torch runs `torch.nn.utils.clip_grad_norm_(trainable, max_norm)` after `backward()` (after `scaler.unscale_` under AMP) and jammi runs the production `clip_gradients` at the same point. `max_grad_norm` is a member of the SHARED identity set (`ci/scripts/perf/identity_fields.py`'s `FINETUNE_IDENTITY_FIELDS`, where `null` is a value meaning "off"), so `ab_merge.py` refuses a row whose two legs differ; each leg also reports `clip_invocations`, the counted number of clip calls, which `ab_merge.clip_fact_violations` checks against the request. |
| `JAMMI_KERNELS_DISABLE=attention_block` (env) | `--attn` | `attention_arm` — the attention REFERENCE CLASS a leg was ASKED to run (`"eager"` or `"fused"`) — is a shared identity field too: torch derives it from the RESOLVED `_attn_implementation` (`eager` → `"eager"`, `sdpa`/flash/flex → `"fused"`), jammi from the operator's resolved `JAMMI_KERNELS_DISABLE` request (an attention base — `attention_block`, `attention_block_flash`, `all` — in `kernels_disabled_requested` → `"eager"`, else `"fused"`). Deliberately NOT the dispatch counters: those read eager on a by-design domain decline (`head_dim != 64`, `seq > 4096`, dtype/contiguity/mask), a measurement `fused_proof` already owns. `ab_merge.py` refuses a jammi-eager ↔ torch-sdpa pairing — the "two references, never mixed" rule as a checked premise — and treats a FALLBACK leg (torch-sdpa OOM → torch-eager) as "not comparable", never as a mismatch. The raw strings/counters stay in provenance. |
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

## Peak VRAM: two fields, two different jammi mappings — read before comparing

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
`FinetuneStepTier` in `crates/jammi-bench/src/report.rs` wherever the
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
doc section and `compare_grad_oracle.py`'s `is_vacuous_pair`/
`vacuous_tensor_count`, which classify and surface this case explicitly
rather than let a `0.0` cosine there masquerade as either a pass or a
fail. Catching a real `dL/dA` defect needs at least one optimizer step
first (moving `B` away from zero), which neither script does.

`ci/scripts/perf/compare_grad_oracle.py` reads a jammi `grad-oracle` dump
and a `torch_grad_oracle.py` dump — SAME JSON schema on both sides,
INCLUDING `batch_token_id_sums` (both producers emit it; the comparator
refuses if either side omits it or the two disagree) — and reports gradient-DIRECTION
agreement (cosine similarity), never a loss comparison, ONLY after
verifying its own premise: that both dumps recorded a loaded
`--lora-weights-in` file, that their per-tensor `weight` arrays actually
agree, and that their run-identity fields (seed/batch/seq/lora_rank/
target_modules/batched_forward/backbone_dtype) and `batch_token_id_sums`
match — a mismatch on any of those REFUSES the comparison (never a silent
`PASS`) regardless of how well the gradients themselves happen to agree.
On the live A100 run above, this weight-identity check held by actual
agreement, not by luck of a loose bound: `max|w_jammi - w_torch| =
1.86e-9` over 224 tensors -- orders of magnitude inside the ULP-relative
tolerance `compare_grad_oracle.py`'s `WEIGHT_MATCH_ULPS`/`_weight_element_tolerance`
derive (an f32-ULP-relative bound, not a fixed absolute constant).

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

Three more oracles, one per graph-learning workload, each the PyTorch rung of a
workload whose engine rung is a `jammi-bench` leg producer
(`crates/jammi-bench/src/{graph_sample,propagate,context_predictor}.rs`). A
**leg** is one run of one implementation stack: `identity` (what two legs must
agree on to be comparable), `provenance` (recorded, never compared) and
`measured` — the warm per-iteration time series (`iteration_s`, never only a
summary), `peak_rss_bytes` (the kernel's high-water mark for the process —
`VmHWM`, or `getrusage`'s `ru_maxrss` where there is no `/proc`),
`peak_vram_bytes` (null: these rungs use no device) and the outcome (a digest
and the file it digests). A producer decides nothing; `ladder_leg.py` is the
capture every script shares, the twin of `src/leg.rs`.

Every rung of a workload reads the **same input files**, which the engine rung
writes. A composite workload is cut at its committed intermediate artifact — a
graph fine-tune at its pair table, a predictor training at its episode set — so
each comparison is of one thing, and sharing the artifact across stacks removes
the sampling randomness between them instead of averaging over it.

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

Every leg's `provenance.packages` records what actually ran.
`ci/scripts/perf/test_torch_graph_rungs.py` (the `torch graph rungs` guard,
`torch-host` lane) runs all three over tiny inputs and holds each against an
oracle that shares no code with it.

## `graph-sample` — node2vec walks

```
jammi-bench graph-fixture --out run/g64                      # the committed synthetic graph
jammi-bench graph-fixture --nodes-per 1024 --out run/g1024   # a larger point of a size sweep
jammi-bench graph-sample --graph run/g64 --graph run/g1024 --out run/jammi \
    --walk-length 4 --walks-per-node 4 --return-p 1 --in-out-q 0.5
python3 torch_graph_sample.py --graph run/g64 --graph run/g1024 --out run/torch \
    --walk-length 4 --walks-per-node 4 --return-p 1 --in-out-q 0.5
```

Several `--graph` are a size sweep: each graph is sampled in its own process, so
no point inherits another's peak resident set, and cost and memory can be fitted
against `identity.edges`. `--transitions` (meant for a small graph) adds
`transitions.jsonl` — for every `(prev, cur, next)` the number of times a walk
stepped `cur → next` having arrived from `prev`, `prev` null on a first step —
and, on the engine side, `expected_transitions.jsonl`: node2vec's analytic law
`π(x | t, v) ∝ α_pq(t, x) · w(v, x)` for that graph
(`graph_sample::node2vec_transition_law`), the ground truth both rungs' counts
are judged against.

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
jammi-bench propagate --nodes 1000,10000 --partitions 1,4 --hops 2 --alpha 0.1 --out run/prop
python3 torch_propagate.py --impl exact --input run/prop/n1000/input --input run/prop/n10000/input \
    --hops 2 --alpha 0.1 --out run/prop
python3 torch_propagate.py --impl pyg   --input run/prop/n1000/input --input run/prop/n10000/input \
    --hops 2 --alpha 0.1 --out run/prop
```

The engine rung writes `n<nodes>/input/{x0.safetensors, x0.keys.txt,
edges.jsonl}` and, per partition count, `n<nodes>/p<partitions>/propagated.*`;
the torch rungs read that `input/` and write `n<nodes>/torch-<impl>/`. Pass the
engine leg's `identity.hops` — the depth actually run, after the engine's clamp
to its hop cap.

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
jammi-bench predictor-train-run --arch Tnp --out run/cp/jammi --warmup-steps 2
python3 torch_context_predictor.py --episodes run/cp/jammi/episodes.safetensors \
    --initial-weights run/cp/jammi/initial_weights.safetensors --out run/cp/torch \
    --epochs 30 --learning-rate 0.005 --grad-clip 1.0 --warmup-steps 2 --num-heads 2
```

The engine rung samples the committed meta-dataset into episodes through the
engine, writes them and its seeded initial weights, and trains the member
`--arch` names (`Cnp`, `AttnCnp`, `Tnp`) with the engine's own fit; pass its
`identity.{epochs, learning_rate, grad_clip, warmup_steps, num_heads}` to the
twin, which reads the member off the weight file's tensor names and refuses a
name set that is not exactly one member's. Both legs carry every optimizer
step's loss (`step_losses`, the batch's loss at the parameters the step started
from) and the trained head's raw output on every held-out test target
(`predictions.*`, keyed `test{episode}_{row}`).

| behaviour | status |
| --- | --- |
| MLP (φ, ρ, a Tnp block's MLP, the Tnp head) | REPRODUCED — `fc2(gelu(fc1(x)))`, both linears biased, exact erf-GELU |
| `Cnp` | REPRODUCED — φ over `(x ‖ y)`; mean over present members, an empty context pooling to zero; ρ over `(pooled ‖ target_x ‖ context_size)` |
| `AttnCnp` | REPRODUCED — biased `query`/`key`/`value` projections (query from `target_x`, key from `context_x`, value from `(x ‖ y)`); the learned `prior_key`/`prior_value` prepended as an always-present member 0; ρ over `(attended ‖ target_x)`, no output projection |
| `Tnp` | REPRODUCED — target token `target_embed(x) + query_marker` at position 0 then `context_embed(x ‖ y)`, no positional encoding; per block biased `q`/`v` and bias-free `k`, attention, `tokens + attended`, `tokens + mlp(tokens)`, no layer norm, no output projection; `head` MLP on position 0 |
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
| `Tnp` (2 heads, 2 layers) | 2.4e-7 | 1.7e-1 (step 163) | 0.318137 / 0.279036 | 3.0 / −0.83 |

The `Tnp` row is not an unreproduced operation. Its 27 weight tensors agree to
1.3e-5 after 6 steps and 5.7e-5 after 30 (jammi vs torch `f32`), closer than
torch `f32` to its own `f64` run (4.9e-5 / 1.3e-4). What grows is rounding:
the running maximum of `|d|` rises ×10 every 25 steps for jammi vs torch — and
×10 every 23 steps for torch vs torch after **one ulp** in one weight (1.9e-1
by step 179), ×10 every 27 steps for torch `f32` vs torch `f64`, and ×10 every
27 steps for an `f64` run vs the same `f64` run with that one `f32` ulp. `Cnp`
amplifies far less (×10 every 51 steps) and `AttnCnp` not at all (flat at
~1e-7). The rate is the member's, not the stack's: the `Tnp` blocks carry no
normalisation, so residual growth compounds through both layers at the
committed learning rate `5e-3` (at `1e-3` the same ulp reaches 5.6e-2; at
`2e-4` it stays at 1e-6). A `Tnp` trajectory is therefore reproducible across
stacks step for step over a short horizon and in its weights, and reproducible
exactly only within one stack.
