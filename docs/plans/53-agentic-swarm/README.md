# Jammi agentic swarm — design

Jammi already *practices* a rigorous agentic method: the rigor chain
(`plan → pressure-test → implement → independent adversarial audit → CI → merge`),
the coordinator-delegates-to-worktrees pattern, "never merge on green CI alone,"
and "a delegated agent's *done* is a claim to verify." Today that method lives as
**prose the lead must remember** — `CLAUDE.md`, `docs/PHILOSOPHY.md`, and
`docs/plans/51-marathon-learnings/AGENTIC-PLAYBOOK.md` — and is enacted ad-hoc.

This plan **mechanizes** it: turns discipline the lead carries in its head into
repo-committed, CI-enforced infrastructure. The design is inspired by the Lace
swarm (`~/git/lace/lace/.claude/`), adapted to Jammi's structure — a Rust engine
of generic primitives, a strict engine↔consumer boundary, and the
engine↔cookbook loop that is Jammi's real bug-discovery engine.

It is delivered capability-by-capability, each capability itself passing through
the rigor chain. **Capability 1 is doc-currency** — chosen first because it has a
proven, recent failure attached (below) and is self-contained.

---

## The failure this retires

Engine PR #243 added `ProducingDescriptor::External`, a 7th materialization
variant, and the 0.33.0 release shipped. Nothing bound the maintainer guide to
the code, so `docs/maintainer/MAINTAINER-GUIDE.md` silently went stale:

- Its `replay_descriptor` variant enumeration still listed **6** variants
  (`Inference, Embedding, NeighborGraph, GraphPropagation, ContextSet, AsofJoin`)
  while the real enum in `crates/jammi-db/src/store/manifest.rs` has **7**.
- Its recompute invariant — *"any new materialising verb must add a
  `ProducingDescriptor` variant **and** a `replay_descriptor` arm"* — became
  misleading, because `External` is the deliberate **exception**: a
  consumer-materialized table with *no* replay arm that intentionally returns
  `NotRecomputable`.

The gap was caught by a human reading code against the guide. It should have been
a mechanical tripwire. That tripwire is the `doc-parity` gate below.

---

## Capability 1 — doc-currency pipeline

A four-role pipeline that keeps the maintainer/architecture guides current with
the code, and a CI gate that surfaces drift red on every PR (and blocks merge once
it is required in branch protection).

```
build-graph ──▶ graph-navigator ──▶ doc-updater ──▶ doc-parity (CI oracle)
 (regenerate)     (query/surface)     (edit guides)   (bind guide ⇄ code)
```

### Roles

- **`build-graph`** — regenerates the symbol graph on demand. Wraps the existing
  `ci/scripts/build_graph_rich.sh` (`cargo build-graph build --rich --references`,
  post-verified genuinely rich) and `ci/scripts/gen_dep_dag.py` (the crate-level
  dep-dag block generated *into* the guide, already CI-freshness-gated by
  `.github/workflows/dep-dag.yml`). Deterministic, zero-token, local-only —
  the graph lives under `target/`, never committed.
- **`graph-navigator`** (the "graphify" role) — reads the rich graph to answer
  *"every site that enumerates `ProducingDescriptor` variants,"* *"who documents
  `replay_descriptor`,"* so the updater is **complete**, not spot-fix. Serves the
  graph via `python -m graphify.serve` (pip `graphifyy`) when the MCP nav server
  is available; falls back to reading `target/build-graph-rich/graph.json`
  directly. Communities are advisory; cite symbols/edges, never cluster ids.
- **`doc-updater`** — edits the hand-authored prose the generators can't: brings
  variant enumerations, invariant statements, and worked examples current with
  the code, following `CLAUDE.md`'s "docs reflect current state" (no journey
  markers). Read+Edit scoped to `docs/maintainer/**` and `docs/guide/**`.
- **`doc-parity`** — the oracle. A static check (`ci/scripts/check_doc_parity.py`)
  that fails CI when a documented enumeration diverges from the code it mirrors.
  Seeded with the `ProducingDescriptor` ⇄ guide binding; extensible to any
  enumerate-in-prose-of-a-code-enum pair. This is the mechanized anti-staleness —
  the Lace `hash-pin`/`ledger-keeper` idea, narrowed to Jammi's guide↔enum drift.

### Parity gate contract

`check_doc_parity.py` asserts, for each registered binding:

1. the set of variant names the guide enumerates == the set of variant names in
   the source enum (order-insensitive), and
2. any variant the guide flags as an *exception* to a stated invariant (e.g.
   `External` → no replay arm) is present and still an exception in code
   (its `recompute.rs` arm returns `NotRecomputable`).

Runs in CI on every PR and push to `main` (`.github/workflows/doc-parity.yml`);
fails closed (diff uncomputable → fail). To *gate merges* it must be added to the
branch's required-status-checks in GitHub branch protection — the same remaining
wiring step that makes `dep-dag` enforceable. Once required there, a new engine
variant surfaces the drift red on every PR and blocks merge until the guide is
updated or the binding is explicitly re-registered.

---

## Broader roster (roadmap — later capabilities)

Adapted from Lace, sized to Jammi (start lean, let the escape ledger justify
each addition — Lace's own anti-Goodhart rule applied to the swarm's growth):

- **Constitution, split in two.** Enumerate the load-bearing invariants already
  written in prose into citable lists: **engine** (the discipline test — *would a
  user who never heard of any consumer reach for this?* — one-way references,
  embeddings-through-`search`, one-binary-pluggable-backends, atomic-across-crates,
  `ProducingDescriptor`⇄`replay_descriptor` completeness) and **enterprise**
  (consumer-of-engine, `StorageUrl`-not-`PathBuf`, companion-tables-not-`enterprise.db`,
  `InferenceSession` borrowed-never-wrapped, tenant-isolation oracle). Human-amend-only.
- **`lead` + phase machine + self-failure-modes** — codify the AGENTIC-PLAYBOOK
  as a loadable pre-flight (trigger→symptom→root-cause→prevention).
- **Core verifiers** — `adversarial-audit` (already used ad-hoc; it caught the
  enterprise F1 pause-reason bug), `fix-verifier` (Rust red-green + the
  non-finite-control lesson from the playbook), `discipline-test-auditor`
  (engine-names-no-consumer, mechanizes `ci/scripts/check_dep_direction.py`),
  `oracle` hard-blocks (H4 API-freeze-guard, append-only migration numbering,
  lockstep `workspace.package.version`, tenant-iso seam).
- **Domain agents by crate/subsystem** — `db-catalog`, `db-store/materialization`,
  `ai-pipeline`, `finetune`, `eval`, `wire/server`, `cli`, `python`, `cookbook`
  (engine); `registry`, `experiment`, `gate`, `monitor`, `wire/grpc`, `licensing`
  (enterprise). Worktree-isolated writers with unique `CARGO_TARGET_DIR`.
- **Cookbook loop as a first-class workflow** (Jammi-unique; Lace has no analog) —
  a `cookbook-emit` role that re-runs chapters and treats divergence as a bug;
  for enterprise the triangular engine↔enterprise↔cookbook loop.
- **Escape ledger** — `.jammi/escapes.jsonl`, seeded from `CASE-STUDIES.md` and
  the enterprise F1; a new regression test cites an open escape.

## Jammi-specific constraints the design must honor

- **Two repos, one boundary.** engine + enterprise with one-way references; a
  domain agent may not cross the boundary. Enforced by the discipline-test
  oracle + `check_dep_direction.py`.
- **Build-env is harsher than Lace's.** Unique `CARGO_TARGET_DIR` per worktree;
  **never** override `RUSTC_WRAPPER`/`RUSTFLAGS` (a ~100-min sccache cache-miss
  incident); watch NVMe pressure. Higher-leverage hook material than any of
  Lace's hooks.
- **Enforcement is honest.** Like Lace, the real teeth are native per-agent
  `tools:` capabilities + a few fail-closed CI gates; the rest is discipline.
  Don't mistake elaborate role files for hard guarantees.
