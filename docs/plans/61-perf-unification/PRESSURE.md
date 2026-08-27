# PRESSURE — perf-unification (findings that shaped `CONTRACT.md`)

A second pass over `DESIGN-STUDY.md`, re-verifying every cited mechanism at source. What stood is
recorded here so an implementer building against `CONTRACT.md` knows *why* each clause exists, not
only what it says.

## The claim grammar (rule (h))

- **Mechanism verified.** Every falsifier cell the design study claimed re-derived independently:
  `−38.5 ms` = `abs(diff(mean(r1,r2),mean(r1,r2))) as ms`, `14.55 GB` = a byte count ÷1e9, `8.3%` =
  `pct(torch_r1,torch_r2)`. All measurement cells of the stacked-sweep table bind at printed
  precision, including trailing-zero forms (`1.2650`, `0.8300`) — which work only because equality is
  string-on-quantized-`Decimal`, never float.
- **Unit confusion.** The scope rule was stated per numeric *token*, but an early census counted
  *measurement numbers* (a coarser unit) — the two disagree once shape labels and version strings are
  in the mix. The grammar needs a closed lexical exclusion class (shape labels, issue/PR refs, version
  strings, ledger cites, dates) so a non-measurement digit has a defined disposition instead of being
  silently ignored or silently flagged.
- **Estimator shopping.** With min/mean/max/r1/r2 all legal over a 2-run pair, several candidate
  values exist per delta cell, and nothing stopped a tag from being written to match whatever the doc
  already printed — including a value that contradicts the artifact's own COMPUTED field (a committed
  artifact's `delta_ms` disagreeing with the doc's printed delta by a full millisecond). The remedy is
  mechanical precedence, not a style rule: when a computed field exists, the tag must point at it.
- **The remedy for a mismatched cell is not automatically "fix the doc."** Neither the artifact nor
  the doc is always wrong — a doc row can legitimately quote a session-ledger measurement the
  committed artifact never recorded (a different session, a since-superseded box). The correct
  disposition is `ledger`, with the classification note stating which candidates were considered and
  why none binds — never a silent pick of whichever candidate looks closest.
- **The escape needs a ratchet.** An inline `no-producer: <reason>` comment, with no central list, no
  shrink-only enforcement, and no per-entry classification, is strictly weaker than the mechanism this
  repo already ships for the identical class (`check_doc_numbers_have_producers.py`, landed for
  measurement-shaped numbers in Rust/CUDA doc comments). Reuse that ledger's exact shape — a committed
  `file:token:sha1(normalized-line)` allowlist plus a hand-classified reason table plus a
  `--check-allowlist-only-shrinks` leg that fetches `origin/main` — rather than build a second, weaker
  gate for the same failure mode.

## Provenance and leg identity (phases 1–2, load-bearing for what phase 3 can point at)

- **Worktree HEAD resolution.** A build script must resolve `.git`'s HEAD via `git rev-parse
  --git-path HEAD`, never the literal `<workspace>/.git/HEAD` — in a `git worktree add` checkout,
  `.git` is a FILE, not a directory, and the literal path does not exist.
- **Dirty-tree detection must exclude untracked files.** `git status --porcelain` alone includes
  untracked scratch; a build script that treats a dirty tree as invalidating a `-dirty` suffix (and
  rule (i) treats `-dirty` as always-INVALID) must scope the check to tracked paths only, or benign
  local scratch invalidates every leg.
- **A `pull_request` checkout resolves the MERGE ref**, never a commit any post-merge history
  contains — a CI-baked `build_sha` is provenance about the CI job, never about a shippable artifact,
  and must never be accepted as one.
- **The comparison tuple (Python) and the K7-completeness const (Rust) are different things and must
  stay different.** Growing the Python comparison tuple used to decide whether a jammi leg and a torch
  leg are comparable would make every existing merge INVALID (torch never emits several Rust-only
  identity fields) — only the Rust-side completeness const grows; the comparison tuple is unchanged,
  and a subset test pins the direction.

## Container census (rule (i), lettered (g) at design time)

- The container census undercounted: a hand-folded `additional_boxes.<box>.legs.<leg>.runs[]`
  container and a temporary, uncommitted `optimizer_phase_wall_time_ms` diagnostic were both missed on
  a first pass — the latter is the producer of a headline `8.9×` cell that, re-derived from the
  artifact's own fields, does not actually print `23.1`/`2.59` anywhere; the honest disposition is
  `ledger`, not a silent correction to whichever nearby value looks closest.
- Ten committed raw legs use a `.json.raw` extension specifically to dodge the schema gate's `*.json`
  glob (the artifact's own `provenance_note` states this in plain text) — the gate must cover any
  extension under a `-raw-runs/` directory, with the ten legacy `.json.raw` files named in a closed,
  non-growing allowlist rather than silently exempted by extension.

## What this means for phase 3 specifically

Phase 3 (rule (h) + the ledger + the guide) does not depend on phases 1–2 having landed first to OPEN
its own PR: its pointer roots (`crates/jammi-kernels/artifacts/cuda-runs/**`, `crates/jammi-bench/
baselines/*.json`) already exist and are already provenanced under rules (a)–(f), independent of rule
(g)'s v2 schema. The `legacy(...)` marker in the grammar is phase 3's explicit, auditable stand-in for
what rule (i) would otherwise classify automatically once it lands — used only for the two cells
`CONTRACT.md` C12.2 names by hand, not inferred.

**Round-2 audit finding (P2-move simulation).** Independence to OPEN a PR is not independence at MERGE
time. A `git mv` simulation of phase 2's baseline move (`crates/jammi-bench/baselines/{finetune_step_
reference,p1_softmax_scale_fold_ab}.json` → `crates/jammi-kernels/artifacts/cuda-runs/`) against phase
3's actual tags REDs 7 of them (T3's P1 row: 5 entries; T3's P3 row: 2 entries — both reuse the P1
baseline's own `base` fields). `POINTER_ROOTS` listing both the pre- and post-move directories as
allowed roots does NOT make a tag survive a rename: `Loader` resolves each pointer against exact `git
ls-files` membership at the path the tag names, and a `git mv` removes the old path from that set. An
earlier draft of `README.md` stated the opposite ("no re-pointing is required either way") — corrected;
whichever of phase 2 and phase 3 merges second re-points those 7 tags in the same PR.
