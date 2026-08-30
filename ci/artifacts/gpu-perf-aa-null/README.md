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
families).

## Campaign protocol

Each report was produced by ONE invocation of the committed producer,
`ci/scripts/perf/gpu_inference_ab.sh`, with `GPU_INFERENCE_AB_AA_NULL=1` set:

1. The producer resolves `PARENT_SHA` = `git merge-base origin/main HEAD`
   (at the time of the campaign, `main` itself — every report's own
   `a_sha`/`b_sha` and every leg's `provenance.build_sha` read
   `6980b8301b1bd104fbed2804af14115f2c0f3f2f`).
2. `--aa-null` makes `clone-b` check out the SAME `PARENT_SHA` as `clone-a`
   — never a PR sha — so BOTH clones measure the identical committed tree;
   any resulting ratio is pure build+measurement+pod noise, never a real
   code difference.
3. TWO independent, `--filter=blob:none` clones of that one sha
   (`clone_and_checkout`'s own `git clone --no-hardlinks --quiet
   --filter=blob:none`) — never one checkout reused for both builds (the
   producer's own doc: a single checkout whose ref moves between builds is
   unsound the instant the earlier binary is later run, since its baked
   fixture path resolves through whatever the checkout currently holds).
4. TWO full, independent, cold `cargo build --release -p jammi-bench
   --features cuda` builds, one per clone (`build_clone`), completed before
   any leg is measured.
5. Four legs run back to back on the SAME rented pod, in the fixed
   order-balanced sequence **A, B, B, A** (`a1`, `b1`, `b2`, `a2` —
   `gpu_inference_ab.sh`'s own `run_leg` calls), each leg's
   `.started_at` timestamp machine-verified non-decreasing by
   `gpu_inference_ab.py::verify_recorded_order` before any ratio is trusted.
6. `gpu_inference_ab.py` merges the four legs: identity/premise refusal via
   `ab_merge.generic_leg_premise_violations` over
   `identity_fields.GPU_INFERENCE_IDENTITY_FIELDS`, then
   `combined_embed_p50_ratio` — the mean of the two adjacent-pair `b/a`
   ratios on `embed.p50_ms` (`a1/b1`, `b2/a2`).

Five runs were executed on rented RunPod pods spanning **both** A100 device
models on 2026-08-30, all against the SAME `main` tip
(`6980b8301b1bd104fbed2804af14115f2c0f3f2f`), all merging `status=GREEN`
(clean identity, clean A,B,B,A order) — no report in this directory carries
a leg-premise violation.

## Per-run table

| file | device model | role | combined `embed.p50_ms` ratio (b/a) | pair `a1/b1` | pair `b2/a2` | within-run pair spread |
|---|---|---|---|---|---|---|
| `2026-08-30-sxm4-r1.json` | NVIDIA A100-SXM4-80GB | primary | 0.8707 | 0.8315 | 0.9098 | 0.0783 |
| `2026-08-30-sxm4-r2.json` | NVIDIA A100-SXM4-80GB | primary | 0.8822 | 0.8651 | 0.8993 | 0.0342 |
| `2026-08-30-pcie-p1.json` | NVIDIA A100 80GB PCIe | primary | 1.0491 | 1.0376 | 1.0606 | 0.0229 |
| `2026-08-30-pcie-p3.json` | NVIDIA A100 80GB PCIe | primary | 0.9933 | 1.0473 | 0.9392 | 0.1081 |
| `2026-08-30-pcie-p2.json` | NVIDIA A100 80GB PCIe | **auxiliary** | 0.9475 | 0.8772 | 1.0179 | 0.1407 |

`2026-08-30-pcie-p2.json` is committed and its report content is genuine —
its `mode` field reads `"aa-null"` and its `recorded_order` verifies as a
clean, monotonic A,B,B,A sequence, the same checks every other file here
passes — but the pod's own driver process reported an anomalous return
code around that run outside the producer's own recorded evidence. It is
flagged **auxiliary** here and **excluded from the band derivation below**;
the four **primary** runs (`sxm4-r1`, `sxm4-r2`, `pcie-p1`, `pcie-p3`) are
the campaign's own evidence base.

## Characterization findings

### (a) The effect is binary-level, and its sign follows the device model

The four primary runs' own `combined_embed_p50_ratio` values cluster tightly
by **device model**, not by run order or wall-clock time: the two SXM4 runs
both read **0.87–0.88** (0.8707, 0.8822), and the two PCIe runs both read
**0.99–1.05** (0.9933, 1.0491) — the auxiliary PCIe run (0.9475) falls in
the same PCIe range. An alternating manual probe against the retained
per-run binaries (re-invoking `clone-a`'s and `clone-b`'s own built
`jammi-bench` binaries directly, off the SAME already-built artifacts these
reports were measured from, rather than rebuilding) confirmed the effect
follows the **binary**, not the invocation order: swapping which retained
binary played the `a`-role versus the `b`-role flipped which side read
faster, reproducibly, on both device models. Since `--aa-null` builds the
identical committed source tree twice, from two independent clones, this is
a genuine build-to-build (not code-to-code) variance on this endpoint — the
whole reason a real, evidence-derived band, rather than a guessed one, is
required before any enforcement.

### (b) Within-run pair spread

Each run's own two adjacent-pair ratios (`a1/b1`, `b2/a2`) are not
identical — `combined_embed_p50_ratio` is their mean, per
`gpu_inference_ab.py::combined_embed_p50_ratio`'s own doc. Across the four
primary runs, the spread between a single run's two pair ratios (max minus
min) ranges from 0.0229 (`pcie-p1`) up to 0.1081 (`pcie-p3`) — the auxiliary
`pcie-p2` run's own spread (0.1407) is wider still but is excluded from this
range per its auxiliary flag above. This within-run spread is itself part of
what the pre-registered band (below) has to absorb, on top of the
between-run, binary-level effect in (a).

## Band derivation

`ci/scripts/perf/gpu_inference_ab.py::PRE_REGISTERED_ADVISORY_BAND` is
derived from the four **primary** runs' own adjacent-pair ratios above:

- The single largest `|log deviation|` from 1.0 among all eight primary
  pair ratios is `sxm4-r1`'s `a1/b1` = 0.8315173238022384
  (`|ln(0.8315173238022384)| = 0.18450314616782526`).
- 1.5x that worst deviation: `1.5 * 0.18450314616782526 = 0.27675471925173794`.
- `exp(±0.27675471925173794)` ≈ `[0.7582404560899295, 1.3188428445994143]`.
- Rounded outward (never inward) to `[0.75, 1.33]` — the committed band.

See `gpu_inference_ab.py`'s own module doc (the "PRE-REGISTERED ADVISORY
BAND" section) for this derivation restated verbatim next to the constant
itself.
