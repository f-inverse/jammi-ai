# Swarm rules — the tracked canonical source

**Authoritative. Human-amend-only, same as `CONSTITUTION.md`.** This file carries the
rule text `docs/swarm/CONSTITUTION.md` cites as its "canonical source" for the
invariants that are not themselves a `docs/guide/src/philosophy.md` anchor — the
swarm-facing operating rules, not the engine's design philosophy. Editing it trips
`CONSTITUTION_TOUCHED` in `.github/workflows/swarm.yml`, the same human-amend-only gate
`CONSTITUTION.md` and `philosophy.md` trip, so a rule cannot drift out from under the
invariant index unreviewed.

## Engine, not platform

Every capability that enters the engine passes the discipline test
(`philosophy.md#the-discipline-test`): a user who has never heard of any particular
consumer would still reach for it. The corollary this heading anchors (`CONSTITUTION.md`
row **B1**) is the swarm-operational half of that same rule — a card, a gate fixture, a
cookbook chapter, or a plan document that only makes sense by naming one consumer's
product, team, or deployment has failed the same test the engine code itself is held to;
the fix is the same in both places — generalize the primitive, file off the name, never
add a consumer-specific flag or branch to carry it.

## Atomic across the workspace

A behaviour change ships across every crate it touches in one PR; the workspace is never
left inconsistent between merges (`CONSTITUTION.md` row **B6**) — a caller crate is never
left compiling against a shape the callee crate no longer provides, even transiently
between two merged PRs. Every publishable crate ships at the same
`Cargo.toml:workspace.package.version` (`CONSTITUTION.md` row **K6**, lockstep) — a
version bump is a workspace-wide edit to the one `[workspace.package]` table
(`Cargo.toml:37`), never a per-crate override, so "which version is crate X at" never
needs a crate-by-crate answer.

## Migrations are append-only

Catalog migrations are append-only and monotonic (`CONSTITUTION.md` row **K5**): a new
migration is appended to the end of `crates/jammi-db/src/catalog/migrations.rs`'s
`MIGRATIONS` const list, and names are never reused or reordered.
`crates/jammi-db/tests/it/migrations.rs`'s `EXPECTED_MIGRATION_NAMES` is the oracle: it
pins the exact ordered name list, so a reorder, a rename, or a reused name is a diff
against a committed list, not a runtime surprise on an existing deployment that has
already applied the earlier numbering.

## Anti-Goodhart: propose, never self-weaken

The swarm may propose to tighten its own rules — this file, `CONSTITUTION.md`, and every
gate script under `ci/scripts/check_*.py` — only through a human-merged, human-reviewed
PR; no agent edit to any of them takes effect without that review
(`CONSTITUTION_TOUCHED` / `SWARM_GATE_TOUCHED` in `.github/workflows/swarm.yml` flag
exactly this touched set red). Every gate workflow always runs, with no `paths:` filter,
and detects its own touched set from inside the job (`git diff <base>...<head>`) rather
than being skipped by a path filter — a required check that a path filter can leave
un-run is not a check a PR can rely on.
