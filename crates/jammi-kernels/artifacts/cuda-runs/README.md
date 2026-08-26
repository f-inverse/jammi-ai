# CUDA run artifacts

One JSON per proven branch tip: the machine-checked record that the crate's `--features cuda`
gates actually executed on hardware. No CI lane has a GPU, so a `cuda_device()`-gated test is not
coverage until a file here records it running — the file is the evidence, the commit message is a
pointer to it.

Each artifact records the exact `git_sha` of the checkout that ran, the device / driver / `nvcc`,
the gate outcomes (`cuda clippy → cuda_parity → per-crate tests with the feature on`), every parity
test by name with its status, and every bench leg's headline numbers with the per-op dispatch
counters (`fused > 0 && eager == 0` is the positive proof the fused path ran; a leg without it is
INVALID, not a datum). Produced by the pod job's `ci/scripts/perf/proof_artifact.py` from the
captured cargo logs; a run that parsed zero parity tests is written as `INVALID`, never as green.

Naming: `<date>-<unit>-<sha7>-<gpu>.json`. Append-only: a re-proof of the same tip on another box
is a new file, never an overwrite.

## Schema

`ci/scripts/check_cuda_run_artifacts.py` enforces this on every `*.json` under this directory
(recursing into any `*-raw-runs/` subdirectory too — a per-leg raw `jammi-bench` dump inherits
its parent artifact's `git_sha` and `producer`, it does not go schema-free just because it is a
leg rather than the top-level artifact). Required top-level fields:

- **`schema_version`** (int) — `1` today.
- **`git_sha`** (40 lowercase hex chars) — the exact commit the run measured, checked with
  `git merge-base --is-ancestor <git_sha> HEAD`. A green artifact whose sha is not an ancestor of
  the branch is evidence about the ORACLE, not the code (`docs/maintainer/cuda-kernel-guide.md`
  §4) — the gate fails it. **OR**, for a reviewed legacy artifact only, **`git_sha_unresolved`**
  (whatever short/malformed ref the pod session actually saw) paired with `producer.kind ==
  "none"` — never fabricate a 40-hex value that was not actually resolved.
- **`box`** (string) — the physical/pod identifier the run measured on.
- **`producer`** — `{path, kind, invocation, gating}`:
  - `kind`: `"cargo-test"` (a single named `#[test] fn`, invoked with `--exact <fn>` so the gate
    can statically confirm the fn exists, sits under `#[test]`, and carries the stated gating
    attribute) | `"script"` (a tracked producer script, e.g. this directory's own
    `proof_artifact.py`) | `"none"` (a reviewed legacy artifact predating this schema — the
    gate's own `LEGACY_NONE_ALLOWLIST` is closed; a NEW artifact may never default to this).
  - `gating`: `"#[ignore]"` | `"env:<VAR>"` | `"required-features"` | `"none"` — how the named
    test/script stays out of a plain `cargo test`/CI run.
- **`status`** (string).
- **`merged_as`** (40 lowercase hex chars, OPTIONAL) + **`merged_via_pr`** (int, OPTIONAL, only
  valid together with `merged_as`) — a branch tip a measurement ran on can be squash-merged, so
  `git_sha` is then legitimately never an ancestor of anything again. `merged_as` names the squash
  commit the SAME content landed on `main` as; `merged_via_pr` is the PR number that merged it.
  Rule (d) (`check_ancestry`) PASSES if EITHER `git_sha` is an ancestor of `HEAD`, OR `merged_as`
  is — `merged_as` is verified ONLY by `git merge-base --is-ancestor merged_as HEAD` (the same
  ancestry check `git_sha` itself gets), never by inspecting the squash commit's own diff. A record
  whose `merged_as` names a squash commit that introduced the artifact under a DIFFERENT path than
  it lives at today (e.g. a `git mv`'d baseline — this directory's own `p1-softmax-fold` and
  `finetune-step-reference` entries, moved here from `crates/jammi-bench/baselines/`) is legitimate
  and still passes: ancestry, not path/content reproduction, is the property this rule pins.
  `git_sha` is always kept verbatim (the tip that was actually measured) — `merged_as` only ever
  supplements it, never replaces it, and requires a resolved `git_sha` (never valid alongside
  `git_sha_unresolved`).

Run the gate: `python3 ci/scripts/check_cuda_run_artifacts.py`. Self-test (RED cases for every
rule, on a throwaway fixture repo — never this checkout):
`python3 ci/scripts/check_cuda_run_artifacts.py --self-test`. Falsifier for container drift:
`python3 ci/scripts/check_cuda_run_artifacts.py --census`.

## Schema v2 — rule (g), leg identity

Rules (a)-(f) above answer "who produced this file, at what sha, ancestor of `HEAD`" for every
`*.json` under this directory — but say nothing about whether one PARTICULAR number inside a
folded artifact carries a complete, machine-checkable identity (seed/shape/dtype/arm). Rule (g)
answers that, for any JSON object anywhere in the tree that opts in.

**Discriminator.** An object carrying `"leg_schema_version": 2` (or higher) is a v2 **leg** — found
by a recursive walk of every top-level artifact, plus every payload under its sibling
`*-raw-runs/` directory. An object WITHOUT the key is a v1 leg and is validated only by rules
(a)-(f), exactly as before this rule existed — v1 artifacts are never retroactively required to
carry it.

**Two leg shapes**, per `identity.leg_shape`:

- **`raw`** — the identity home. Carries `identity.{tier, producer_kind, leg_shape}` plus every
  entry of the identity TUPLE for `(tier, producer_kind)` (below), present and non-null unless the
  entry is declared `NullMeans("<reason>")`. `provenance.build_sha` (jammi) / `provenance.git_rev`
  (torch) must be a resolved 40-hex sha equal to the PARENT artifact's own `git_sha` — a GREEN v2
  leg with `unknown` or a `-dirty` suffix is RED, never silently accepted.
- **`folded`** — a reference. Carries `identity.{tier, producer_kind, leg_shape}` and `file` naming
  a raw leg under the sibling `*-raw-runs/` directory that exists and itself validates. A folded leg
  carries NO identity fields of its own — one home, never duplicated.

**Identity tuples** are never hand-typed in the gate: the jammi side is extracted by regex from
`FinetuneStepTier::IDENTITY_FIELDS` + `REPORT_IDENTITY_FIELDS` in `crates/jammi-bench/src/
report.rs`; the torch side is imported directly from `crates/jammi-bench/reference/
torch_finetune_step.py`'s own `TORCH_IDENTITY_FIELDS`/`TORCH_IDENTITY_FIELDS_NULL_MEANS`. A field
declared `NullMeans` may read JSON `null` with the declared meaning (e.g. torch's
`nvidia_driver_version` reading `null` off-CUDA); every other entry is `NonNull` — present AND
non-null, or the leg is RED. `ci/scripts/perf/test_identity_fields_subset.py` keeps these in sync
with the Python COMPARISON tuples (`ab_merge.py`/`compare_grad_oracle.py`), which stay UNCHANGED by
this schema (growing them would invalidate every existing jammi-vs-torch merge).

**Container census** (unit `perf-unification`, phase 2) — every leg-bearing container in this tree,
and its v2 decision:

| # | container | artifact(s) | v2 decision |
|---|---|---|---|
| 1 | `bench_legs[]` | p1, p2, p3 | folded shape defined; existing p1/p2/p3 stay v1 (their raw runs were never committed) |
| 2 | `shapes.<s>.legs.<leg>` | stacked | folded shape defined; existing stacked artifact stays v1 |
| 3a | `*-raw-runs/**` jammi raw = `Report` (`/tiers/finetune_step`) + stamp | stacked (24 jammi), p6-b3-dense (16), fa2-vram-attrib (8) | identity home for a jammi raw leg |
| 3b | `*-raw-runs/**` torch raw = `{tool, provenance, args, finetune_step}` + stamp | stacked (16 torch) | identity home for a torch raw leg |
| 4 | `legs.<name>` + `deltas` + `peak_vram_bytes_by_leg` | cast-w1 | v1 (`LEGACY_NONE_ALLOWLIST`) |
| 5 | `finetune_step_forced_arm_ab.*`, `a100b_full_step_ab_reference.summary` | adamw | v1; its raw legs are the 10 `.json.raw` files below |
| 6 | `legs.<name>.runs[]` + `summary` | p6-b3-dense | v1; raw legs are row 3a |
| 7 | `same_build_before_after_full_finetune_step.*`, `per_layer_vram_probe` | fa2-vram-attrib | v1; raw legs are row 3a |
| 8 | `measurements[].legs[]` / `legs[]` | p6-b1-flash-timing ×2 | v1, a different tier (`flash_kernel_timing`) |
| 9 | `legs[]` (test outcomes) | esc044-growth-oracle | not a bench leg |
| 10 | `healthy_oracle_measured`, `red_controls_measured` | flash-arm-encoder-oracle ×2 | not a bench leg |
| 11 | `clip_on_flash_leg.record` | #381 branch (device-clip-narrow) | v2 when it lands with `leg_schema_version: 2` |
| 12 | `combined_report.configs[]`, `paired_table.rows[]` | esc-045 branch | not committed (branch unmerged) |
| 13 | `additional_boxes.<box>.legs.<leg>.runs[]` | p6-b3-dense | v1, hand-folded (producer: an untracked pod script) |
| 14 | `optimizer_phase_wall_time_ms.{eager_arm,fused_arm}.run_N.{min_ms,median_ms,mean_ms}` | adamw | v1, a temporary uncommitted diagnostic |

**`LEGACY_RAW_NONJSON`** (closed, 10 entries) — bare pre-schema `Report` dumps committed as
`*.json.raw` (never `*.json`, so they predate rule (g)'s any-extension coverage AND rule (a)'s own
`*.json` glob), exempted from BOTH by relpath, one reason each, in the gate script itself. A
deletion must shrink this list in the same commit (`--self-test` proves every listed relpath
exists); growth is a gate edit, i.e. SWARM_GATE_TOUCHED by construction — this list cannot silently
absorb a new rename-to-dodge-the-glob file.

**`LEGACY_NONE_ALLOWLIST`** (closed, 7 entries) — `producer.kind == "none"` legacy artifacts. The
five pre-schema entries predate this gate entirely; the two newest (`2026-08-24-finetune-step-
reference-d361515-a100-pcie.json`, `2026-08-24-p1-softmax-fold-bf8e807-a100-sxm4.json`) are the two
baselines moved here from `crates/jammi-bench/baselines/` (unification contract C8) — both hand-
driven `jammi-bench finetune-step` runs with no tracked producer script. `producer.kind == "none"`
is legitimate ONLY when the artifact's own first-introduction commit (`git log --follow
--diff-filter=A`) is an ancestor of this gate's own introduction (`c7fd1df`) — a genuinely NEW
artifact can never satisfy that, so this list cannot grow again without a gate edit AND a history it
does not have.
