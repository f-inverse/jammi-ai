# `torch_finetune_step.py` — the PyTorch/PEFT reference

This directory holds an ORACLE, not a dependency. `torch_finetune_step.py` is
pure Python against the public `transformers` + `peft` APIs. It measures the
same unit as `jammi-bench finetune-step`
(`crates/jammi-bench/src/finetune_step.rs`) — one LoRA optimizer step on
ModernBERT: three encoder forwards (anchor/positive/negative, all live on the
tape at once), a cosine-margin triplet loss over L2-normalized mean-pooled
embeddings, one backward into the LoRA tensors, one AdamW step — so the two
can be compared step-for-step on the same box.

## B2: what this is not

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
(2025-01-10). An earlier draft of this file pinned `transformers==4.44.2`
(2024-08-22) as "developed against"; that version predates ModernBERT
entirely and could not have run this script — that claim was false and has
been removed. Every report's `provenance` block records the versions that
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
— and record both rows: `sdpa` is torch's best-case number (what a `#352`
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
| *(none in this script yet)* | `--max-grad-norm` | See "The trainer-shaped step: `--max-grad-norm`" below. This script does not yet call `torch.nn.utils.clip_grad_norm_`, so there is no flag to mirror; a clip-on-vs-clip-off A/B today only exists on jammi's side of the comparison. Adding a matching flag here (and to `torch_finetune_step.py`, which this contract's `files_in_scope` does not cover) is a natural follow-up, not done in this change. |
| *(n/a)* | `ln_fused_dispatches`/`ln_eager_dispatches`/`rope_fused_dispatches`/`rope_eager_dispatches`/`softmax_fused_dispatches`/`softmax_eager_dispatches`/`geglu_fused_dispatches`/`geglu_eager_dispatches`/`lora_epilogue_fused_dispatches`/`lora_epilogue_eager_dispatches`/`attention_block_fused_dispatches`/`attention_block_eager_dispatches` | Not reported here — those are jammi's own fused-kernel dispatch counters (`jammi_kernels::ops::LayerNormFused`/`RopeFused`/`SoftmaxLastDimFused`/`GegluFused`/`ScaledCastAdd`/`AttentionBlockFused`); there is no equivalent concept on the torch side (`--attn` is the closest analogue for attention, and torch's own kernel dispatch inside `sdpa`/`eager` is not independently observable through the public API this script is restricted to). **Honest note on `attention_block_*`:** `AttentionBlockFused`'s domain is fixed at `head_dim == 64` (`jammi_kernels::ops::ATTENTION_BLOCK_HEAD_DIM`) — on any checkpoint whose `hidden_size / num_attention_heads != 64`, the admission predicate refuses by domain (`"head_dim_is_attention_block_fixed_head_dim"`) on every call, so the pair reads `attention_block_fused_dispatches: 0` / `attention_block_eager_dispatches: N` (`N` = the number of attention calls the step made) even on a run whose OTHER fused counters (`ln`/`rope`/`softmax`/`geglu`/`lora_epilogue`) are non-zero. That all-eager reading is the predicate working as designed, not a broken fused path — never read `0` fused dispatches here as evidence the kernel is unreachable in general; check the checkpoint's `head_dim` first. |

## The trainer-shaped step: `--max-grad-norm`

Every `finetune-step` row recorded before this flag existed measured a step
the product does not run. The shipped trainer's `FineTuneConfig` defaults
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
the step this tier always measured before — bit-identical to before this
flag existed, and useful as the isolated no-clip reference point. Supply it
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
(`device_memory_used_bytes`, finetune_step.rs:115) on a background thread
every 25ms (`std::thread::sleep`, finetune_step.rs:150) over the ENTIRE step
loop (warmup + measured), then subtracts a baseline snapshot
(`peak.saturating_sub(baseline)`, finetune_step.rs:166) read once, right
after the model+optimizer are built (before the loop starts) — see
`vram_baseline`, finetune_step.rs:540.

**An earlier draft of this script got the sampling point wrong.** It polled
`torch.cuda.memory_allocated()` once per step, at the same point the clock
stopped — i.e. AFTER `backward()` + `optimizer.step()` + the `.item()` sync,
the one instant in each step where every saved activation has already been
freed. Measured directly: that poll captured 403 KiB of a 9087 KiB in-step
peak (~4.4%), systematically, on every measured step — a discrete poll
phase-locked to the step's deterministic TROUGH, not its peak. That
per-step poll has been REMOVED. There was a second, independent asymmetry in
the same draft: torch's `AdamW` allocates its `exp_avg`/`exp_avg_sq` moment
tensors LAZILY, on the first `optimizer.step()` call (measured: 0 optimizer
state tensors before that first step, 48 after, on a tiny test model) —
while candle's `AdamW::new` allocates them EAGERLY, before jammi's own
baseline is read. A baseline taken right after `torch.optim.AdamW(...)`
returns (before any step) would therefore NOT yet include the moments, and
their one-time first-step allocation would land inside the measured delta
instead of being absorbed into the baseline the way jammi's is.

Both problems are fixed the same way: this script now runs ONE UNTIMED
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

With that fixed baseline point established, BOTH VRAM fields now come from
torch's own CONTINUOUS allocator high-water mark — `torch.cuda.reset_peak_memory_stats()`
called ONCE right after the untimed warm-up step (i.e. right before the
timed warmup+measured loop starts, matching the window jammi's sampler
covers), then `torch.cuda.max_memory_allocated()` read once after the loop
ends. A continuous tracker cannot miss an intra-step spike the way ANY
discrete poll can — this script's old per-step read, or jammi's own 25ms
`nvidia-smi` interval:

* **`peak_vram_delta_bytes`** — the field COMPARABLE to jammi's
  `peak_vram_bytes` column: same window (the entire warmup+measured loop),
  same baseline convention (`memory_allocated()` snapshot taken after
  model+optimizer construction AND after the one untimed moment-warmup step,
  recorded separately as `peak_vram_baseline_bytes`). Computed as
  `max_memory_allocated() - peak_vram_baseline_bytes` after the loop.
  **RESIDUAL ASYMMETRY, stated rather than papered over:** this is now a
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

**Whether the pin took is never inferred from "the call didn't raise."** An
earlier draft of this script recorded a boolean `..._accepted` flag set to
`True` purely because `from_pretrained` didn't raise — but it was measured
`True` in exactly the case where the kwarg was silently dropped
(`hasattr(cfg, "reference_compile")` was `False` immediately after that
"successful" call), which is precisely backwards. That boolean has been
replaced with two RESOLVED readbacks, taken directly off `model.config`,
never inferred from call success:

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
that call (an earlier draft of this script did) would leave the adapter
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
report by whatever process consumes both (a later contract's A/B table).

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
contract: one forward+backward at IDENTICAL LoRA weights, no optimizer
step, no timing). See `crates/jammi-bench/src/grad_oracle.rs`'s module doc
for the full "why gradients, not loss trajectories" argument, and this
script's own module doc for the exact jammi<->PEFT tensor-name translation
table it owns (jammi's `grad-oracle` subcommand does zero translation — the
shared weight-interchange file is a plain `safetensors` file in jammi's OWN
internal naming; this script translates both directions).

**PROVENANCE — read before trusting this script's output**: as of the
F2/F3 audit-fix round on PR #372, it HAS been run once, live, on an A100
pod (ModernBERT-large, `--batch 8 --seq 128 --seed 42`, jammi tip
`e62c8a8`) — reported by the lead who dispatched that pod job, not
verified locally by this fix round (no GPU was available here). See
`torch_grad_oracle.py`'s own module-doc PROVENANCE banner for the full
disclosure, including the measured cosine similarities from that run.
Beyond that one confirmed config, everything else (other checkpoints,
`target_modules` sets, dtypes/ranks/batch/seq combinations) remains
UNVERIFIED against a live run — one successful execution is evidence the
mechanism works, not a proof it is correct everywhere this script accepts
flags for. Its NAME-TRANSLATION functions are, independently, locally
tested (`test_torch_grad_oracle_names.py`, stdlib-only, no torch needed —
that suite caught and pinned a real bug in an early draft: `Wi`, an MLP
site whose jammi-side name carries no `mlp.` prefix, was misrouted to
`attn.Wi` by a naive string-prefix heuristic).

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
first (moving `B` away from zero); not implemented this round.

`ci/scripts/perf/compare_grad_oracle.py` reads a jammi `grad-oracle` dump
and a `torch_grad_oracle.py` dump — SAME JSON schema on both sides,
INCLUDING `batch_token_id_sums` (both producers emit it; the comparator
refuses if either side omits it or the two disagree — an earlier draft of
this script left `batch_token_id_sums` out of this dict entirely, which
this line used to describe as "same schema" while that gap existed; fixed
in the F3 audit round on PR #372) — and reports gradient-DIRECTION
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
derive (advisory ii, round-2 audit fix on PR #372: a fixed `1e-4` absolute
constant, now an f32-ULP-relative bound).

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
