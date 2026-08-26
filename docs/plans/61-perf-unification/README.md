# 61 — perf-unification

> Plan group. One mechanism, in three phases, that makes the fine-tune performance track's own
> artifacts and the maintainer guide that summarizes them **derive from a single source of truth**,
> so a discrepancy between the two becomes a mechanical CI failure instead of a doc that quietly
> drifts. See [`DESIGN-STUDY.md`](./DESIGN-STUDY.md) for the mechanism, [`PRESSURE.md`](./PRESSURE.md)
> for what stood and what didn't at a second pass, and [`CONTRACT.md`](./CONTRACT.md) for the clauses
> an implementer builds against.
>
> **Status:** phases 1–3 are this unit; phase 4 (`ServerInfo` static-build-identity fields) and phase
> 5 (a held-out-loss acceptance statistic for the fine-tune arms) are descoped to their own follow-up
> units — each records its front-door precondition in `CONTRACT.md` (C16, C17) rather than being built
> here.

## Why this unit exists

`docs/maintainer/fine-tune-performance-guide.md` prints close to 200 numbers across nine tables,
summarizing a real performance track (`#352` → `#389`). Nothing mechanically checked that any of
those numbers still agreed with the artifacts they claim to summarize: a printed value could drift
from its source (several plausible estimators exist for the same delta — min, mean, r1, r2 — and the
tag was free to match whichever the doc already said), or could never have had a tracked producer at
all (a session-ledger number quoted from a `.jammi/ledger/*` row, never committed as JSON). The guide
read as equally authoritative either way. Separately, the harness's own `jammi-bench` binary carried
no build-time identity (`git_sha`/`target`/`profile`/enabled kernel features baked into the binary
itself), so a leg's provenance was assembled by hand at production time and never checked against
what actually ran.

## The three phases

| phase | what it builds | owner | file |
|---|---|---|---|
| 1 — bench provenance | `jammi-bench`'s own baked identity (`build_sha`/`target`/`profile`/`build_features`), the `Provenance` struct on `Report`, and the K7-completeness identity consts per `(tier, producer_kind, leg_shape)` | bench (+ numerics for one `admission.rs` const) | `crates/jammi-bench/build.rs`, `report.rs` |
| 2 — leg identity + producers | rule (g): `check_cuda_run_artifacts.py` gains a schema-v2 discriminator (`leg_schema_version >= 2`) that requires every v2 leg to carry a self-declaring identity tuple and a `build_sha` matching its artifact's own `git_sha`; every shell producer cross-checks the binary before it writes a leg; the two hand-carried baselines move under `crates/jammi-kernels/artifacts/cuda-runs/`; the swarm gate glob widens to cover a gate script under a subdirectory | docs-ci (+ numerics, bench per file) | `ci/scripts/check_cuda_run_artifacts.py`, `ci/scripts/perf/*.sh`, `.github/workflows/swarm.yml` |
| 3 — claim grammar + guide | rule (h): `ci/scripts/check_perf_claims.py`, a bounded six-form claim grammar that binds every numeric token in the guide's nine tables to a tracked artifact value at Decimal/`ROUND_HALF_EVEN` string equality, or escapes it into a committed, shrink-only ledger; the guide's own §7/§8 become citation tables into `cuda-kernel-guide.md` | docs-ci | `ci/scripts/check_perf_claims.py`, `ci/perf_claims_allowlist.txt`, `docs/maintainer/fine-tune-performance-guide.md` |

The lead's phase order is **1 → 2 → 3**, which reverses an earlier draft's "grammar first" — IN THAT
ORDER, the baselines move (phase 2) lands before any `claims:` tag is written (phase 3), so a tag
authored against the post-move guide would never need re-pointing. Phase 3 was, in practice, built as
its own branch off `main` before phase 2 merged (a real scheduling deviation from the idealized order,
not a hypothetical one), so its `claims:` tags point at the two baselines' PRE-move path
(`crates/jammi-bench/baselines/{finetune_step_reference,p1_softmax_scale_fold_ab}.json`) — 7 tags,
concretely, across the guide's T3 P1 and P3 rows. Whichever of phase 2 and phase 3 merges second must
re-point those 7 tags to the post-move path under `crates/jammi-kernels/artifacts/cuda-runs/` in the
SAME PR that rebases — `check_perf_claims.py`'s `Loader` resolves a pointer against `git ls-files`
membership at the exact path given, so a `git mv`'d file silently reds every tag still pointing at its
old location; a P2-move simulation against this phase's own tags confirms exactly this (see
`PRESSURE.md`).

## What the mechanism actually closes

Rule (h) does not claim to make every number in the guide a live citation. Roughly a quarter of the
guide's numeric tokens bind to a tracked cuda-run artifact or baseline (V, plus a small V-legacy class
for artifacts whose raw legs are deliberately outside the provenance gate, C7 row 14); the rest are
session-ledger, modeled/projected, or GH-issue numbers with no committed producer, and are escaped
into `ci/perf_claims_allowlist.txt` with one classification row each. What the mechanism buys is not a
higher bind rate by itself — it is that **every** number is now in exactly one of three mechanically
distinguishable buckets (bound, excluded-as-non-measurement, or ledgered-with-a-reason), an uncovered
number is a CI failure, and the ledger can only shrink. A number that drifts from its artifact, or a
new number added without a tag, is caught the same PR it lands in.

## Journey framing — a recorded exception

`docs/maintainer/fine-tune-performance-guide.md`'s opening line ("How jammi's Rust/candle training
step went from 0.44× to 1.07× of PyTorch on the same GPU (2026-08-23 → 08-26)...") is journey framing
— normally out of place in a doc that should describe the system as it is, not the path that produced
it (see `CLAUDE.md`). This guide is a deliberate, lead-approved exception: its own stated purpose (§0,
"What makes the track worth a guide is not the result but the path") is to teach the *process* of
proving a fused kernel faster and faithful, not only to state the current numbers, and every session
in the plan group that reviewed it agreed the guide is unusable for that purpose with the dates and
before/after framing stripped out. The exception is scoped to this one file; no other engine doc gets
the same latitude, and the guide's own §10 ("Where it stands") is still written as a status table, not
a diary.

## Concurrent-session strategy

Phases 1 and 2 touch `crates/jammi-bench`, `crates/jammi-kernels`, and `ci/scripts/perf/**` and are
each their own atomic PR (B6); phase 3 touches only `ci/scripts/check_perf_claims.py`, the two new
ledger files, `.github/workflows/ci.yml`'s `guard` matrix, and `docs/maintainer/**` plus this plan
directory. Phase 3's own pointer roots (`crates/jammi-kernels/artifacts/cuda-runs/**` and
`crates/jammi-bench/baselines/*.json`) already exist on `main` independent of phases 1–2 landing, so
phase 3 does not block on EITHER phase to open its own PR — but it DOES have one real, concrete
dependency at merge time: phase 3's tags for the two moved baselines (T3's P1 and P3 rows, 7 tags) are
written against their PRE-move path, since phase 3 was built before phase 2 merged. `POINTER_ROOTS`
listing BOTH `crates/jammi-kernels/artifacts/cuda-runs` and `crates/jammi-bench/baselines` as allowed
roots does NOT make a tag survive the move — `Loader` resolves a pointer against exact `git ls-files`
membership, and a `git mv` removes the old path from that set. Whichever of phase 2 and phase 3 merges
second re-points those 7 tags in the same PR (see the phase table above).
