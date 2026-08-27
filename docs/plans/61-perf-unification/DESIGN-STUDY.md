# DESIGN-STUDY — perf-unification

Design-only; phases 1–3 spend no GPU time. The unifying criterion (see `README.md`): one mechanism
makes at least two of the four pieces — the bench binary, the cuda-run artifact store, the maintainer
guide, and the CI gates that read them — derive from a single source of truth, such that a
discrepancy becomes a mechanical CI failure. Phases 1–3 satisfy it for harness-artifacts ↔ the
maintainer guide.

## 1. Rule (h) — the bounded claim grammar

Gate: `ci/scripts/check_perf_claims.py`. Hermetic — reads markdown and tracked JSON, shells out only
to `git ls-files`.

**Scope.** A pipe table under `docs/maintainer/**` or `docs/plans/61-perf-unification/**` is in scope
if its header row contains one of `s/step`, `VRAM`, `ratio`, `cosine`, `×`, `ms`, `GB`, `launches`.
Every numeric token in a body cell of an in-scope table must be covered by exactly one tag; an
uncovered token is a finding at `file:line:col`.

**Tag placement.** An HTML comment on the line immediately above the row: `<!-- claims: c1=<expr>;
c2=<expr>; ... -->`, one entry per non-excluded numeric token, left to right, cell by cell. Comments
do not render, so the table is visually unchanged for a reader.

**Grammar (closed).** A pointer `P` is `<tracked-path>#<json-pointer>` resolving to a JSON number.
`min(P1,P2,…)` / `mean(…)` / `max(…)` aggregate ≥2 pointers. `diff(A,B)` is `A − B`; `ratio(A,B)` is
`A / B`; `pct(A,B)` is `(A/B − 1)·100` — `A`,`B` each a pointer or an aggregate. `<form> as <unit>`
applies a fixed factor (`s→ms ×1000`, `bytes→GB ÷1e9`, `bytes→GiB ÷2^30`, `bytes→MiB ÷2^20`, `×100`
for `%`). `abs(<form>)` compares the printed cell's magnitude against `|evaluated|`, discarding sign
on both sides — this is what lets the doc print a saving as `−38.5 ms` while the artifact stores
`+39.57`. A token whose expression has no producer is marked `ledger` (a bare word, not an inline
reason string) and must have a matching entry in the committed allowlist; anything outside this
closed set is a parse error, never a silent pass.

**Artifact-field precedence.** For a `diff`/`ratio`/`pct` tag, if the operands' common ancestor JSON
object already carries its own computed `delta*`/`ratio*`/`speedup*`/`*_pct`/`spread*` NUMBER field,
the tag must bind to that field instead of recomputing a free aggregate — this closes the estimator-
shopping gap (min/mean/max/r1/r2 all "look right" over a 2-run pair, so a free aggregate can always be
tuned to match whatever the doc already prints).

**Equality.** The printed token is parsed as `Decimal` at its own precision (digits after the `.`);
the evaluated expression is quantized to that precision with `ROUND_HALF_EVEN` and compared as a
STRING, never a float — this is the only way a trailing zero (`1.2650` vs an evaluated `1.265`)
binds, since a float carries no notion of trailing-zero significance.

## 2. Provenance baked at build time (phase 1, not built in this unit's docs-ci scope but load-bearing
for what phase 3's tags can point at)

`jammi-bench`'s `build.rs` bakes `build_sha` (`JAMMI_BUILD_SHA` env, accepted only if 40-hex, else git
at build time, else `"unknown"`), `target`, `profile`, and `build_features` (linked-crate constants —
`jammi_kernels::admission::FLASH_COMPILED`/`CUDA_COMPILED`, never `CARGO_FEATURE_*`, which is a
build-script-only value) into a `Provenance` struct on every `Report`. This closes the `tip_sha`
defect: the pre-existing `grad_oracle.rs::tip_sha()` read `git rev-parse HEAD` at RUN time, so a
provenance value could silently drift from the binary that actually produced the numbers around it if
the checkout moved between build and run.

## 3. Rule (g) — leg identity on self-declaring v2 legs

`check_cuda_run_artifacts.py` gains a schema discriminator: any JSON object carrying
`leg_schema_version >= 2` is a v2 leg, walked for recursively under every artifact (plus every
`*-raw-runs/**` payload under a `schema_version >= 2` parent, any extension — a `.json.raw` payload
counts as a leg too, refused unless it is in a closed, non-growing `LEGACY_RAW_NONJSON` list). A v2
leg must carry its `(tier, producer_kind, leg_shape)` identity tuple, present and non-null unless
declared nullable with a stated meaning (`max_grad_norm: null` = no clip), and its own
`provenance.build_sha` must equal the parent artifact's `git_sha` — a leg with `unknown` or `-dirty`
can never be GREEN. This is what phase 3's `V` (as opposed to `V-legacy`) classification would key off
once it lands; phase 3 alone, without rule (i) live, treats every pointer bind as unclassified by v1/
v2 shape except the two cells this contract calls out by hand (`legacy(...)`, C12.2).

## 4. Honest bound on what closes

Rule (h) mechanically binds the fully self-consistent stacked-sweep table (all 8 shapes × 4 metrics +
the one measured spread) and the artifact-backed rows of the two other tables that cite a committed
producer. The remaining ~75% of the guide's numeric tokens are declared, per-token, `ledger` — a
session-ledger number, a GH-issue number, a modeled projection, or a number whose own committed
artifact prints a different value at the same site (estimator shopping caught red-handed, not
silently corrected). The unit does not claim to close doc-drift wholesale; it claims that every one of
those buckets is now enforced, not asserted in prose, and the ledger's own growth is a CI failure.
