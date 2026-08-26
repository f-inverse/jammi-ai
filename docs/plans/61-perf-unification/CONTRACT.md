# CONTRACT — perf-unification, phases 1–3

Binding order: `README.md` (lead resolutions) → `DESIGN-STUDY.md` → `PRESSURE.md`. Phase 4
(`ServerInfo` static-build-identity fields) and phase 5 (a held-out-loss acceptance statistic) are
NOT in this unit; this contract records their front-door preconditions (C15, C16) only.

## Frame

Single source of truth: the `jammi-bench` binary's own baked identity (`build_sha`, `target`,
`profile`, `build_features`) plus the runtime kernel facts `jammi_kernels::admission` owns
(`FLASH_COMPILED`, `disabled_ops_requested`, `disabled_ops_fired`), emitted through the existing
`Report` container. Three readers under `ci/` make disagreement a CI failure: rule (g) inside
`check_cuda_run_artifacts.py`, rule (h) as `check_perf_claims.py`, and a Python-⊆-Rust identity
subset suite. Every gate script under `ci/scripts/**/check_*.py` / `check_*.sh` is inside the
`SWARM_GATE_TOUCHED` glob and human-merged.

Phase order: **1 = bench provenance; 2 = rule (g) + producers + the two baselines move + swarm glob;
3 = rule (h) + the ledger + the guide.** The baselines move (phase 2) lands before any `claims:` tag
is written (phase 3), so no tag is ever re-pointed.

## C1 — Provenance is baked at build time (phase 1)

A new `crates/jammi-bench/build.rs`. Precedence: `JAMMI_BUILD_SHA` env (accepted only if
`^[0-9a-f]{40}$`, anything else — including a non-hex placeholder — rejected and falls through) → git
at build time → `"unknown"`. The git branch resolves HEAD via `git rev-parse --git-path HEAD` (never
the literal `<workspace>/.git/HEAD`, absent in a worktree) and emits `cargo:rerun-if-changed` for that
path and the ref it points at. Dirtiness is `git status --porcelain --untracked-files=no` restricted
to tracked paths. Output is `<sha>` or `<sha>-dirty`. A sha produced inside a `pull_request` CI
checkout resolves the MERGE ref and must never be accepted as provenance — no CI job in this unit
emits a leg, and rule (g) cross-checks `provenance.build_sha == parent.git_sha` so a merge-ref sha can
only ever land INVALID. Test shape: never assert 40-hex, never assert not-`unknown` (a dirty dev tree
is GREEN); assert run-time inertness (re-running with a different `JAMMI_BUILD_SHA` in the environment
yields a byte-identical `provenance` object).

## C2 — `Report::new` and the `Provenance` struct (phase 1)

`Provenance { build_sha, target, profile, build_features: Vec<&'static str>,
report_schema_version: u32 }` on `Report`; one constructor `Report::new(subcommand, tiers)` replaces
every literal `Report {` construction site. `build_features` are linked-crate constants —
`jammi_kernels::admission::FLASH_COMPILED` → `"flash-attn"`; a sibling `CUDA_COMPILED` const added
beside it → `"cuda"`; `cfg!(feature = "cuda")` of `jammi-bench` itself → `"bench-cuda"` — sorted,
deduplicated, never `CARGO_FEATURE_*` (a build-script-only value). A `jammi-bench provenance`
subcommand prints the struct so a shell producer can read it before any leg. The pre-existing
run-time `tip_sha()` (`git rev-parse HEAD` at RUN time) is deleted; the equivalent field is filled
from the baked `build_sha`.

## C3 — K7 identity consts (phase 1)

Per-`(tier, producer_kind, leg_shape)` `IDENTITY_FIELDS` consts, each a list of `(name, Nullable)`
pairs — `max_grad_norm` and (on the torch side) `nvidia_driver_version` are *present, may be null with
a declared meaning*; every other entry is non-null. A `#[test]` serializes a sample tier and asserts
every listed key is present.

## C4 — Python ⊆ Rust; the comparison tuple is UNCHANGED (phase 1)

The Python COMPARISON tuples (`ab_merge.py`'s `FINETUNE_IDENTITY_FIELDS`,
`compare_grad_oracle.py`'s `RUN_IDENTITY_FIELDS`) do not change in this unit — only the Rust
K7-completeness consts grow. A new stdlib-`unittest` suite asserts Python ⊆ Rust for each pair, and
that the two Python tuples keep their known cardinality (14, 11) — "unchanged" is a number, not a
promise.

## C5–C9 — Rule (g), producers, the two baselines move, the swarm glob (phase 2)

`check_cuda_run_artifacts.py` gains a v2-schema discriminator (`leg_schema_version >= 2`), covering
any file extension under a `-raw-runs/` directory (a closed, non-growing `LEGACY_RAW_NONJSON` list
exempts ten pre-existing `.json.raw` legs from the parse-as-JSON requirement). Every shell/Python
producer cross-checks `$BIN provenance`'s `build_sha` against the sha it is about to stamp before
writing a GREEN leg. `crates/jammi-bench/baselines/{finetune_step_reference.json,
p1_softmax_scale_fold_ab.json}` move under `crates/jammi-kernels/artifacts/cuda-runs/` with rules
(a)–(f) fields backfilled (the p1 record needs `merged_as`/`merged_via_pr` since its own `tip_ref` is
not an ancestor of `HEAD`). `swarm.yml`'s gate glob widens from `ci/scripts/check_*.py` to
`ci/scripts/**/check_*.py` (and the `.sh` twin) so a gate script under a subdirectory is covered too.

## C10 — Rule (h): `ci/scripts/check_perf_claims.py` (phase 3)

See `DESIGN-STUDY.md` §1 for the grammar. Implementation notes fixed here:

1. **Tokenizer.** Scope = pipe tables under `docs/maintainer/**` and
   `docs/plans/61-perf-unification/**` whose header row contains one of `s/step`, `VRAM`, `ratio`,
   `cosine`, `×`, `ms`, `GB`, `launches`. A numeric token is a maximal match of
   `[−\-+]?\d+(?:[.,]\d+)*` not immediately preceded by a word character or `.`.
2. **Lexical exclusion class**, applied before binding: shape labels (`b8s512`, `s512`, `b8`, `d0`/
   `d0.05`), issue/PR refs (`#377`) and the escape-ledger's own `esc-NNN` row id, version strings
   (`2.13.0+cu126`, `cu126`), ledger cites (`s2:89`, `row 5`, `cont row 11`, `fusion rows 30, 36`),
   dates (`2026-08-23`). A layer/tensor/launch count (`563 launches`) is explicitly NOT excluded.
3. **Artifact-field precedence**, VALUE-based (round-4 audit rewrite — this item went through two prior
   mechanisms, common-ancestor-subtree identity then whole-file identifier-token + unit-family, and an
   adversarial sweep of the real tracked tree found BOTH live-broken: the candidate field is almost
   always LESS token-specific than its operands, e.g. `delta_ms` carries no replicate suffix while the
   operands producing it do, so a token-subset test that requires the candidate to be AT LEAST as
   specific as the operands structurally cannot work — 172 free-bind evasions across 14 committed
   fields in the 79 tracked cuda-run JSONs, reproduced and closed). Identity/token reasoning is
   abandoned. For a `diff`/`ratio`/`pct` tag: evaluate the FREE result (the raw operation over the
   operands, in their native unit, before this tag's own `neg`/`as unit`/`legacy` wrapping); if all
   operand leaves share one tracked file, scan the WHOLE file for every NUMBER leaf whose key matches
   `delta*`/`ratio*`/`speedup*`/`*_pct`/`spread*` and test whether ITS VALUE equals the free result —
   same sign, opposite sign, or after a `×1000`/`÷1000`/`÷1e9`/`×1e9` unit rescale (`s<->ms`,
   `bytes<->GB` — round-5 advisory: the prior scale set was asymmetric, missing `×1e9`) — within a 5e-4
   RELATIVE tolerance. This check applies to EVERY non-pointer node in the tag's expr tree, not only an
   outermost `diff`/`ratio`/`pct` (round-5 fix, class A): a top-level or nested `min`/`mean`/`max` is
   itself a free computation — `mean` is literally the estimator-shopping form this contract has always
   named — and a real live evasion was exactly a bare, top-level `mean(...)` tag reproducing an unrelated
   shape's own `ratio_torch_over_stacked`. ANY match is a finding naming the matched field, regardless of
   which two leaves produced the free result or what either is called — the artifact already states
   this exact number under a name that says it is computed. A coincidental match (two genuinely
   unrelated quantities landing within tolerance by chance) is STILL reported, on purpose: the finding
   names the field so a human decides "point at it" or "ledger this cell, it's a coincidence," never a
   silent mechanical exemption for "probably unrelated." STRING-valued same-family fields are NEVER
   value-checked (a stated, deliberate narrowing — value equality cannot be tested against free text).
   THREE such fields exist in the tracked tree, not one (round-5 fix, class A — an earlier draft's "the
   one live case" was false): cast-w1's two `delta_gb` rows (`b8_s512`, `b8_s128`) and flash-arm-encoder-
   oracle's `delta_gb`. All three are in a closed registry (`ci/perf_claims_string_fields.txt`,
   `check_string_field_registry`, part of the bare run — REDs on any unregistered string-valued computed
   field in the tracked tree) with a reason each; a free expression whose operands live under a
   REGISTERED string field's own parent object is a finding too, even with no numeric twin to rescue it
   — the b8_s512 row stays ledgered as an editorial choice (C11), the other two were previously
   uncaught free binds with no guard at all, now closed by the registry-object check. Operands spanning
   more than one tracked file are UNDECIDABLE and the tag is a finding naming that explicitly — never a
   silent pass just because the mechanism could not determine an answer. A `ratio`/`pct` with a zero
   denominator (1,390 zero-valued leaves live in the tracked tree) is a finding naming the offending
   pointer, never a raw decimal exception (round-5 advisory). A regression oracle
   (`check_perf_claims.py --sweep`, required guard leg) enumerates every structurally plausible operand
   pair in the real tracked tree (two sibling containers sharing a leaf sub-path — the exact shape every
   `legs`/`runs`/`shapes`/`summary` container in this tree uses), now including `mean`/`min`/`max` forms
   (round-5 fix), via an INDEPENDENT reference matcher (round-5 fix, class B — never the production
   matcher, or a narrowed production matcher would shrink the enumerated population right alongside its
   catch rate and still look clean), and asserts every value-match is actually caught by the real
   runtime path AND that the population meets a pinned floor (`EXPECTED_SWEEP_MATCHES`).
4. **Equality.** `Decimal` at the token's own printed precision; `ROUND_HALF_EVEN` quantization;
   STRING comparison. A mismatch prints the evaluated value at full precision.
5. **Pointer roots.** Tracked `*.json` under `crates/jammi-kernels/artifacts/cuda-runs/**` and
   `crates/jammi-bench/baselines/*.json` only. A pointer into a non-`.json` payload, an untracked
   file, or a path outside these roots is a finding ("unprovenanced producer").
6. **The escape.** A token with no producer is marked `ledger` (a bare word — never an inline
   reason string) and MUST have a matching entry in `ci/perf_claims_allowlist.txt`, keyed
   `file:token:sha1(normalized line):col` (round-3 fix: `col`, the token's own column offset, makes
   the key injective — two identical tokens on one line, e.g. `112 / 112`, used to collapse into a
   single entry covering both). Every entry has one row in `ci/perf_claims_allowlist_classification.md`
   with a closed reason (`ledger-only | modeled | issue-text | superseded-run`) and a distinct note,
   verified mechanically by `check_classification_file` (a missing, duplicated, or orphaned row is a
   named problem, not a silent gap). `--check-allowlist-only-shrinks` fetches `origin/main` (fails
   CLOSED on a failed fetch or an unresolvable ref) and fails on any entry this branch ADDS; bootstrap
   (no allowlist file on `origin/main` yet) passes only on the introducing PR. **Known limitation,
   inherited from `check_doc_numbers_have_producers.py` (`ac2c5cb`) and out of scope to redesign here:**
   because the key includes `sha1(normalized line)`, an editorial edit to an already-ledgered row (a
   typo fix, a rewording that does not touch the token itself) changes the row's hash and therefore its
   key — the OLD key becomes an orphaned allowlist entry (a `check_classification_file` finding) and the
   token needs a NEW entry, which `--check-allowlist-only-shrinks` reads as an ADDITION and reds. The
   remedy is the same PR that makes the edit: remove the orphaned old entry, add the new one with the
   SAME classification row content, and justify the net-zero swap in the PR body — the ratchet cannot
   distinguish "editorial reword" from "new debt" mechanically, so a human states which one this is.
7. **`legacy(<form>)`** is the explicit, auditable marker for a pointer bind reported as `V-legacy`
   rather than `V` — used only for the two AdamW summary-block cells this contract names (below),
   standing in for what rule (g)'s v2-schema classification would infer automatically once it lands.

## C11 — Named cells

- T5 `s128 −16.5`: `ledger`, reason `ledger-only`, note stating the three candidates the
  `a100b_full_step_ab_reference` summary yields (r1 diff 16.408, mean diff 16.566, r2 diff 16.725 ms)
  and that none rounds to 16.5 — the guide is NOT silently corrected to −16.6 or −16.4.
- The AdamW `0.6759`/`0.6589`/`−16.4 ms` cells: `legacy(...)`, bound to
  `a100b_full_step_ab_reference/summary/s512/{disabled_eager_p50_r1_r2,fused_p50_r1_r2}` — reported
  `V-legacy` since the summary's raw legs are `.json.raw`, outside the provenance gate.
- `23.1 → 2.59 ms (8.9×)`: `ledger`, reason `ledger-only` — the artifact's own
  `optimizer_phase_wall_time_ms` fields print 23.8592/24.7481/24.7765 and 2.6004/2.6881/2.7097 (etc.);
  no pair prints 23.1 or 2.59 under any form.
- T5's FA2 row (`0.6756 → 0.4626`, `0.937×`, `−213 ms`, `39 measured`) and the equivalent T9 esc-044
  row: `ledger`, reason `ledger-only`, note naming the committed artifact whose own numbers differ
  (the b3-dense artifact reads block 0.7096 / flash 0.4992 at a different, uncommitted tip).
- T3's P2/P3 measured cells: `ledger`, note naming the committed p2/p3 artifacts and their own,
  different values (`b8_s512_d0` reads 0.7817 s / 39.58 GB, not the row's printed 0.780 s / 39.1 GB —
  a different session).
- Cast-w1's `−38.5 ms` cell is corrected to `−39.6 ms`, bound via `neg(P)` (sign-preserving; round-3
  fix — REPLACES an earlier `abs(P)`, which discarded sign on both sides and would have silently
  accepted a wrong-signed token as long as the magnitude matched) directly to the artifact's own
  `delta_ms` field (39.565658). Precedence (C10.3, round-4 value-based) forbids a free `diff(r1,r1)`
  here — the same-rep pair whose difference equals `delta_ms` bit-for-bit — but does NOT forbid the
  MEAN-based `diff(mean(r1,r2),mean(r1,r2))` a prior draft used to compute `−38.5 ms`: that mean
  genuinely differs from `delta_ms` by ~2.7%, outside the 5e-4 tolerance, so it is a real, different
  number, not a reproduction. Binding directly to `delta_ms` is still the right call — it is the
  artifact's own authoritative computed field for this exact quantity — but it is a documentation
  choice here, not a precedence-forced one.
- T8 (the torch-column table) is entirely `ledger`, reason `superseded-run`: its artifact lives on an
  unmerged branch until the sha is an ancestor of `HEAD` or carries `merged_as`/`merged_via_pr`.

## C12 — Guide edits, coexist

All nine tables are tagged per C10 (one HTML comment per row, one entry per non-excluded token,
left to right). §7 ("How to prove a fused kernel") becomes a citation table: rule →
`cuda-kernel-guide.md` §3.x → the mechanical oracle-standard check (`pending
check_kernel_oracles.py` until `ci/kernel-oracle-standard` lands) → the escape that paid for it — the
escape column is the guide's unique content and stays. §8 ("Measuring honestly"): the three facts
`cuda-kernel-guide.md` §4 lacked (exclusive box + timing lock; ratios travel across boxes, ms do not;
attribute by grid) move there; §8 keeps its track-specific bullets plus a pointer to §4. The `:226`
sentence about `ab_merge.py`'s identity refusal is rewritten to name the real file:line and to state
that the comparison tuple stays unchanged (C4) — pinned by the Python ⊆ Rust subset suite, not by
this doc alone. Journey framing at the guide's opening line is a recorded exception (`README.md`).
`check_doc_numbers_have_producers.py` is NOT widened to markdown in this unit; rule (h) is the
markdown gate and reuses only the ledger's SHAPE (C10.6).

## C13 — esc-045 torch-column artifact is not committed (all phases)

The torch-column artifact and its two producer scripts stay on their own branch until either the sha
is an ancestor of `HEAD` or a `merged_as`/`merged_via_pr` exists — `check_ancestry` would fail it
otherwise. T8 stays `ledger` (`superseded-run`) until then.

## C14 — Flash-attn is unreachable through a served build

Stated in every phase-3 PR body: no workspace member's closure reaches `jammi-kernels/flash-attn`
except through the feature itself, and the `cuda` lane stays CUTLASS-free — a served engine can never
report `flash-attn` among its compiled kernel features, so any future book-side attribution covers the
AdamW/LN/GeGLU arms only, never the FA2 lever.

## C15 — Phase 4 (`ServerInfo`) — NOT in this unit

Recorded for the follow-up's own front door: only genuinely static-per-process fields may be appended
to `ServerInfo` (`build_sha`, `build_target`, `kernel_features`, `device_name`, `driver_version`,
`kernels_disabled_requested`); `dispatch_counters` and `kernels_disabled_fired` NEVER go on the wire —
both are process-global, per-request-mutated state, so exposing either would falsify the tenant-
isolation allowlist's own written reason ("tenant-independent") while the structural test still
passes. Three fill sites exist, not two, one of which calls the storage-layer type directly.

## C16 — Phase 5 (how-well) — NOT in this unit

The result of the design study is the follow-up's precondition: no feasible how-well statistic exists
on the repo's data today. `recall@10` has no dynamic range (the fine-tune itself moves it by
0.002–0.0105 over a base of 0.538, tolerance 0.03). Held-out loss needs a new public per-pair
evaluation seam (`Trainer::evaluate` is private today), a non-default early-stopping metric (the
default reports TRAIN loss as `final_loss`), a non-hinge objective or a measured tie fraction (the
bench's hinge saturates to `loss_last == 0.0` on both arms at the shapes tried), and a calibration run
before any acceptance gate may be built.

## §6 — F1, the bind-rate floor (amended; lead decision, audit round 2 / A4)

The original phase-3 acceptance text ("≥ 50 V cells and ≥ 3 fully-bound tables") is NOT met by the
real artifacts this unit has to bind against, and the shortfall is not a defect to fix by tagging
harder: `check_perf_claims.py --report` shows exactly ONE fully-bound table (T6, the stacked-sweep
result — every one of its 33 cells is a live pointer into a single committed artifact). The other
eight tables are, by construction, ledger-derived: T1/T2/T7/T8 summarize GitHub-issue or session-ledger
numbers with no committed JSON producer at all; T3/T4/T5/T9 mix a handful of artifact-backed cells
(P1's baseline, the two P2/P3 §10 rows, the cast-w1 and AdamW cells) with projection-history and
census numbers that were never meant to be reproducible from a tracked artifact. Making a SECOND or
THIRD table fully bind would require re-running the underlying measurements against the current
tracked artifact set (e.g. producing a real p2/p3-shaped artifact for T3's own P2/P3 row, or a real
nsys-census artifact for T4) — a separate unit of GPU work, out of scope for a phase whose own standing
clause is "no GPU spend anywhere in phases 1–3."

**The amended acceptance bar is: `>= 1 fully-bound table AND V >= 50 AND the ledger may only shrink`.**
This is a STANDING RATCHET, not a PR-body number: `check_perf_claims.py --report --min-fully-bound=1
--min-v=50` is wired as a REQUIRED fourth `guard` leg beside the bare/self-test/allowlist-only-shrinks
legs (`ci.yml`), so a future PR that retags a bound cell to `ledger` without a compensating new bind
elsewhere — silently eroding V below 50, or un-binding T6's own cells below full coverage — REDs, the
same way `--check-allowlist-only-shrinks` REDs a ledger that grows. At the time of this amendment: V =
54, V-legacy = 5, one fully-bound table (T6), both floors cleared with headroom.

## Standing clauses

Every number in a doc cites its derivation; gates run blocking with real exit codes; SHAs not branch
names; own worktree/target per agent; gate edits (`ci/scripts/**/check_*.py`, `swarm.yml`) are
human-merged; no GPU spend anywhere in phases 1–3; the esc-045 artifact is not committed (C13); the
Python comparison tuples do not change (C4); the `no-producer:` inline escape does not exist in rule
(h) — the ONLY escape spelling is the bare `ledger` marker plus a committed allowlist entry (C10.6).
