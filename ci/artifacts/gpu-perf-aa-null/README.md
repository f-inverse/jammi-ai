# gpu-perf-aa-null — issue #335 D6 empirical-null campaign (2026-08-30)

This directory is the committed evidence `ci/scripts/perf/gpu_inference_ab.py`'s
own `PRE_REGISTERED_ADVISORY_BAND` is derived from — the D6 instrument named
in that module's own doc and in `gpu_inference_ab.sh`'s own `--aa-null`
section. Every file here is a merged `gpu_inference_ab.py::build_report`
output, produced by the committed producer (`ci/scripts/perf/gpu_inference_ab.sh`)
run with `GPU_INFERENCE_AB_AA_NULL=1` — never hand-edited after the fact
(committed artifacts are append-only evidence, the same discipline
`ci/scripts/perf/check_citations.py`'s own "Committed artifacts are
append-only evidence" section documents for this repo's other artifact
families). The machine-readable classification (which file is `primary`
vs `aux`, and why) lives in this directory's own `manifest.json`, checked
against a real re-derivation of the band by `ci/scripts/check_aa_null_band.py`
— see "Band derivation" below.

## Vocabulary warning: the committed JSONs carry PRE-FLIP fields

Every `*.json` in this directory was produced BEFORE the band-pre-registration
unit landed. Their own internal `advisory` object (`"band": [0.9, 1.1]`,
`"band_not_pre_registered": true`, `"classification":
"within_placeholder_band"`/`"outside_placeholder_band"`) reflects
`gpu_inference_ab.py`'s OLD placeholder-band vocabulary at record time, not
a classification against `PRE_REGISTERED_ADVISORY_BAND` — a reader must
never cite a committed file's own `advisory.classification` as if it were
evaluated against the current band. This README's own "Per-run table" and
"Band derivation" below always recompute against the CURRENT
`[0.75, 1.33]` band directly from each file's `adjacent_pair_ratios`, never
by trusting the stored `advisory` object.

## Campaign protocol

Each report was produced by ONE invocation of the committed producer,
`ci/scripts/perf/gpu_inference_ab.sh`, with `GPU_INFERENCE_AB_AA_NULL=1` set:

1. The producer resolves `PARENT_SHA` = `git merge-base origin/main HEAD`
   (at the time of the campaign, `main` itself — every report's own
   `a_sha`/`b_sha` and every leg's `provenance.build_sha` read
   `6980b8301b1bd104fbed2804af14115f2c0f3f2f`).
2. `--aa-null` makes `clone-b` check out the SAME `PARENT_SHA` as `clone-a`
   — never a PR sha — so BOTH clones measure the identical committed tree;
   any resulting ratio is pure build+measurement+pod noise, PROVIDED the
   invoking pod is not itself contended by a second, concurrent invocation
   — see the Disclosure section immediately below for the one pair of runs
   where that provision did NOT hold.
3. TWO independent, `--filter=blob:none` clones of that one sha
   (`clone_and_checkout`'s own `git clone --no-hardlinks --quiet
   --filter=blob:none`) — never one checkout reused for both builds (the
   producer's own doc: a single checkout whose ref moves between builds is
   unsound the instant the earlier binary is later run, since its baked
   fixture path resolves through whatever the checkout currently holds).
4. TWO full, independent, cold `cargo build --release -p jammi-bench
   --features cuda` builds, one per clone (`build_clone`), completed before
   any leg is measured.
5. Four legs run back to back on the SAME rented pod (i.e. the pod THIS ONE
   invocation rents/is given — see the Disclosure below for why "the SAME
   rented pod" does not imply "a pod exclusive to this one invocation" for
   every run in this directory), in the fixed order-balanced sequence
   **A, B, B, A** (`a1`, `b1`, `b2`, `a2` — `gpu_inference_ab.sh`'s own
   `run_leg` calls), each leg's `.started_at` timestamp machine-verified
   non-decreasing by `gpu_inference_ab.py::verify_recorded_order` before
   any ratio is trusted.
6. `gpu_inference_ab.py` merges the four legs: identity/premise refusal via
   `ab_merge.generic_leg_premise_violations` over
   `identity_fields.GPU_INFERENCE_IDENTITY_FIELDS`, then
   `combined_embed_p50_ratio` — the mean of the two adjacent-pair `b/a`
   ratios on `embed.p50_ms` (`a1/b1`, `b2/a2`).

Five runs were executed on rented RunPod pods spanning **both** A100 device
models on 2026-08-30, all against the SAME `main` tip
(`6980b8301b1bd104fbed2804af14115f2c0f3f2f`), all merging `status=GREEN`
(clean identity, clean A,B,B,A order) — no report in this directory carries
a leg-premise violation. TWO physical pods were rented in total for the
whole campaign (one SXM4 pod, one PCIe pod) — but "2 pods" is a count of
PHYSICAL HARDWARE rented, not a claim that every run had that hardware to
itself; see the Disclosure below.

## Disclosure: GPU contention between pcie-p1 and pcie-p2

`2026-08-30-pcie-p1.json` and `2026-08-30-pcie-p2.json` ran CONCURRENTLY on
the SAME single rented PCIe pod (`ezidhyckicgzpv`) — the p2 retry was
launched as a background task while the p1 task was still measuring, not
after p1 finished. This is a violation of the campaign protocol's own
isolation assumption ("the device and its conditions cancel by
construction" — `gpu_inference_ab.py`'s own module doc's opening line —
which presumes ONE producer invocation has the pod's GPU to itself for the
whole four-leg run): a run whose GPU is being time-shared with a SECOND,
independent measurement process is not measuring "pure build+measurement+pod
noise" any more, it is measuring build+measurement+pod noise PLUS
cross-process GPU contention, a qualitatively different (and unbounded)
noise source this campaign was never designed to characterize.

**Evidence, computed directly from the two committed reports' own
`recorded_order` fields** (never hand-adjusted): the two runs' `a1` legs
started `1.483729832s` apart (`pcie-p2.recorded_order.a1.value −
pcie-p1.recorded_order.a1.value = 1483729832` ns); by `b1` the gap had
narrowed to `0.630562161s`, by `b2` to `0.516805532s`, and by their final
(`a2`) legs it was `0.605235332s` — i.e. every one of p2's four legs started
within roughly half a second to a second and a half of p1's OWN
corresponding leg throughout the whole run, never the ~17-19s a genuinely
sequential pair of four-leg runs on the same pod would show (`pcie-p1`'s own
total `a1`-to-`a2` span is `18.791007409s`; `pcie-p2`'s is
`17.912512909s`). This is exactly the signature of two four-leg runs
executing side by side on the same GPU, not one after the other. Separately,
per the operator's own session records (task-launch/completion logs this
directory's committed JSONs do not themselves carry): the two tasks'
process-level outputs completed 27s apart, with p2's overall task window
nested inside p1's — consistent with, and additional to, the leg-timestamp
evidence above.

**Consequence**: BOTH `pcie-p1` and `pcie-p2` are demoted from primary to
**auxiliary** evidence (see "Per-run table" below and this directory's own
`manifest.json`) — `pcie-p1` for the contention alone (its own report
content and identity/order checks are otherwise clean), `pcie-p2` for the
contention AND its separately-observed anomalous driver return code (the
original, now-superseded rationale this file used to cite alone). The
PRIMARY evidence base this campaign's band is derived from is therefore
THREE runs / SIX pairs (`sxm4-r1`, `sxm4-r2`, `pcie-p3`), not four
runs/eight pairs — see "Band derivation" below for why this does not,
numerically, move the derived band.

**Structural fix**: `ci/scripts/perf/gpu_inference_ab.sh` now records pod
identity (`${RUNPOD_POD_ID:-$(hostname)}`) into each leg's own
`provenance.pod_id` field on every report it produces going forward — this
exact contamination class (two independent invocations sharing one physical
pod concurrently) is now ADJUDICABLE directly from a future committed
artifact's own content (two reports naming the same `pod_id` with
overlapping `recorded_order` windows), never requiring out-of-band operator
session records the way this disclosure currently does. The five reports
already committed here predate that field and carry no `pod_id` at all —
this disclosure is the one-time, by-hand reconstruction that field is meant
to make automatic for every future campaign.

## Per-run table

| file | device model | role | combined `embed.p50_ms` ratio (b/a) | pair `a1/b1` | pair `b2/a2` | within-run pair spread |
|---|---|---|---|---|---|---|
| `2026-08-30-sxm4-r1.json` | NVIDIA A100-SXM4-80GB | primary | 0.8707 | 0.8315 | 0.9098 | 0.0783 |
| `2026-08-30-sxm4-r2.json` | NVIDIA A100-SXM4-80GB | primary | 0.8822 | 0.8651 | 0.8993 | 0.0342 |
| `2026-08-30-pcie-p3.json` | NVIDIA A100 80GB PCIe | primary | 0.9933 | 1.0473 | 0.9392 | 0.1081 |
| `2026-08-30-pcie-p1.json` | NVIDIA A100 80GB PCIe | **auxiliary** | 1.0491 | 1.0376 | 1.0606 | 0.0229 |
| `2026-08-30-pcie-p2.json` | NVIDIA A100 80GB PCIe | **auxiliary** | 0.9475 | 0.8772 | 1.0179 | 0.1407 |

`2026-08-30-pcie-p1.json` and `2026-08-30-pcie-p2.json` are both committed
and both reports' own content is genuine — each `mode` field reads
`"aa-null"`, each `recorded_order` verifies as a clean, monotonic A,B,B,A
sequence, the same checks every other file here passes — but the two runs
executed CONCURRENTLY on the same physical PCIe pod (see "Disclosure"
above), and `pcie-p2`'s own driver process separately reported an anomalous
return code around that run, outside the producer's own recorded evidence.
Both are flagged **auxiliary** here and **excluded from the band derivation
below**; the three **primary** runs (`sxm4-r1`, `sxm4-r2`, `pcie-p3`) are the
campaign's own evidence base. `manifest.json` in this directory carries this
same classification machine-readably.

## Characterization findings

### (a) The effect is binary-level, and its sign follows the device model

The three primary runs' own `combined_embed_p50_ratio` values cluster by
**device model**: the two SXM4 runs both read **0.87–0.88** (0.8707,
0.8822), and the one primary PCIe run reads **0.9933** — close to 1.0, well
inside the band derived below. (The two AUXILIARY PCIe runs, `pcie-p1`
1.0491 and `pcie-p2` 0.9475, are consistent with the same rough PCIe range
but are contention-contaminated and excluded from any quantitative claim
here.) An alternating manual probe against the retained per-run binaries
(re-invoking `clone-a`'s and `clone-b`'s own built `jammi-bench` binaries
directly, off the SAME already-built artifacts these reports were measured
from, rather than rebuilding) confirmed the effect follows the **binary**,
not the invocation order: swapping which retained binary played the
`a`-role versus the `b`-role flipped which side read faster, reproducibly.
Since `--aa-null` builds the identical committed source tree twice, from two
independent clones, this is a genuine build-to-build (not code-to-code)
variance on this endpoint — the whole reason a real, evidence-derived band,
rather than a guessed one, is required before any enforcement.

### (b) Within-run pair spread

Each run's own two adjacent-pair ratios (`a1/b1`, `b2/a2`) are not
identical — `combined_embed_p50_ratio` is their mean, per
`gpu_inference_ab.py::combined_embed_p50_ratio`'s own doc. Across the three
PRIMARY runs, the spread between a single run's two pair ratios (max minus
min) ranges from 0.0342 (`sxm4-r2`) up to 0.1081 (`pcie-p3`). (The two
auxiliary runs' own spreads — `pcie-p1` 0.0229, `pcie-p2` 0.1407 — are
excluded from this range per their auxiliary flag above.) This within-run
spread is itself part of what the pre-registered band (below) has to
absorb, on top of the between-run, binary-level effect in (a).

### (c) Sensitivity: what this band can and cannot catch

With the upper edge at `1.33`, the smallest SLOWDOWN this band can catch on
an idealized (zero build-offset) pod is `> 33%` (`1.33 − 1`), not the
`≥ 25%` an earlier draft of this doc claimed — `25%` is the LOWER edge's
own distance from 1.0 (`1 − 0.75`), a different, asymmetric threshold that
does not describe the slowdown-catching side at all.

On a REAL pod, that nominal 33% figure is itself optimistic: this
campaign's own two primary SXM4 combined ratios (`0.8706549652288303`,
`0.8821655548443332`) show the binary-level build offset (finding (a)
above) already suppresses the observed ratio by roughly 12–13% on that
device model, working AGAINST detection of a real slowdown (a slowdown and
a favorable build offset partially cancel in the observed ratio). To still
cross the `1.33` upper edge despite that offset, a real slowdown needs to
overcome it first: `1.33 / 0.8706549652288303 − 1 ≈ 52.8%` under the worse
of the two observed SXM4 offsets, or `1.33 / 0.8821655548443332 − 1 ≈
50.8%` under the milder one — i.e. on a pod exhibiting an SXM4-like build
offset, the REAL smallest reliably-catchable slowdown is on the order of
**~51–53%**, not 33% and certainly not 25%. Tightening this requires
endpoint-precision work (more iters/rows, replicate medians, or
characterizing and correcting for the build-offset effect itself), never
band tuning alone.

## Band derivation

`ci/scripts/perf/gpu_inference_ab.py::PRE_REGISTERED_ADVISORY_BAND` is
derived from the three **primary** runs' own SIX adjacent-pair ratios above
(`sxm4-r1`: `a1/b1`, `b2/a2`; `sxm4-r2`: `a1/b1`, `b2/a2`; `pcie-p3`:
`a1/b1`, `b2/a2`) — mechanically checked against these same five committed
files by `ci/scripts/check_aa_null_band.py`, driven off this directory's own
`manifest.json` classification, wired into `ci.yml`'s Guard matrix (never
merely asserted in prose):

- The single largest `|log deviation|` from 1.0 among the six primary pair
  ratios is `sxm4-r1`'s `a1/b1` = `0.8315173238022384`
  (`|ln(0.8315173238022384)| = 0.18450314616782526`) — UNCHANGED from the
  earlier (four-run/eight-pair) derivation: excluding the two
  contention-contaminated `pcie-p1`/`pcie-p2` pairs did not remove the
  worst-deviation pair, since it was never one of theirs.
- `1.5 * 0.18450314616782526 = 0.27675471925173794`.
- Raw (unrounded) interval: `exp(∓0.27675471925173794)` =
  `[0.7582404560899295, 1.3188428445994143]`.
- Outward rounding (mechanical, reciprocal-symmetric, matches
  `check_aa_null_band.py`'s own `_round_band_outward` exactly): the LOWER
  edge is the raw value FLOORED to 2 decimal places —
  `floor(0.7582404560899295 * 100) / 100 = 0.75` (this direction is a
  simple floor: rounding a lower bound DOWN is always the conservative,
  band-widening direction). The UPPER edge is then set to the EXACT
  reciprocal of that already-rounded lower edge (`1 / 0.75 =
  1.3333333333333333` — keeping the band symmetric in ratio space,
  `lo * hi == 1`), itself FLOORED to 2 decimal places for a clean literal:
  `floor(1.3333333333333333 * 100) / 100 = 1.33`. `1.33 > 1.3188428445994143`
  (the raw upper edge), so this reciprocal-then-floor step is STILL outward
  (band-widening) on the upper side despite using floor, not ceiling,
  arithmetic — the reciprocal-of-a-floored-lower-edge is always larger than
  a direct ceiling of the raw upper edge would be, which is what makes this
  the actual (not merely asserted) rounding rule that reproduces `1.33`
  rather than the `1.32` a naive independent per-edge ceiling would give.
- Committed band: `[0.75, 1.33]`.

See `gpu_inference_ab.py`'s own module doc (the "ADVISORY classification: a
PRE-REGISTERED band, derived from the D6 empirical-null campaign" section)
for this derivation restated verbatim next to the constant itself, and
`ci/scripts/check_aa_null_band.py` for the mechanical re-derivation this
prose is checked against on every PR.
