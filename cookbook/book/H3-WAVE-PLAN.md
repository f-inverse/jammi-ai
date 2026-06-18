# H3 wave plan — operability & contracts → the 1.0 engineering bar

> **STATUS: H3 CLOSED (v0.30.0, 2026-06-16) — HISTORICAL.** All engine waves merged
> and `v0.30.0` published; the 3 cookbook H3 chapters (ch16 model catalog / ch17
> channel-error taxonomy / ch18 two-tenant isolation + BYO-auth) merged in cookbook
> **#21**. Boundary correction landed: model promotion/retirement/registry are
> governance functionality that lives above the open-core engine (removed from open-core in engine #203) — the §3.6 chapter was
> reframed to the catalog (`list`/`describe`/`delete`). Next horizon = **H4** (the 1.0
> engineering bar); this plan is kept for the record. The wave detail below is the
> as-derived H3 plan and is not maintained post-close.

Re-derived at the H3 Step-0 re-baseline (this session) against engine `jammi-ai`
`main` at **4 commits past `v0.29.0`** (`#193`–`#196`) and cookbook `main` at
**`#19` merged** (`db4d2c1`, pinned `jammi_ai==0.29.0`). Supersedes the earlier
`H3-WAVE-PLAN.md` referenced in memory (no longer present; engine H3 has advanced
past where it was scouted — `§3.8` channel-errors and all of `§3.7` have since
landed). Source of truth for *engine* workstreams: `jammi-ai`
`docs/plans/50-open-core-hardening-roadmap/ROADMAP.md` §3/§5. This doc is the
*cross-repo execution sequencing* — which engine PR unblocks which cookbook
loop-closer, in what release order.

"Completion" = drive the engine↔cookbook loop through **H3** then **H4** to the
**1.0 engineering bar**, shipped as a normal **0.x**. The **1.0 tag is deferred**
(user decision, 2026-06-14) — terminal-0.x is where this plan stops; the tag is a
separate post-adoption call.

Acceptance is fixed for every workstream: **a measured cookbook chapter + a CI
oracle**. No engine surface is "done" without its cookbook loop-closer.

---

## Engine H3 status snapshot (against `main`, end of this re-baseline)

| §  | Workstream | Status | Evidence |
|----|------------|--------|----------|
| 3.7 | Operability / production-readiness | **LANDED (unreleased)** | `#194` four dead metrics made live (one tower layer); `#195` gRPC+worker tracing spans (tenant/job-correlated); `#196` operability guide + failure-mode matrix + chaos-lane split. Acceptance (operability doc + chaos test in CI + metric/trace exemplars) met. |
| 3.8 | Channel error-taxonomy (carried from H1) | **LANDED (unreleased)** | `#193` typed `ChannelCatalogError` (6 variants) → correct gRPC codes (`AlreadyExists`/`NotFound`/`FailedPrecondition`/`InvalidArgument`), mirrors the `MutableTableError` precedent; closes the `Code::Internal`-for-everything defect. |
| 3.8 | Transport-parity collapse | **LANDED (unreleased)** | `#197` (`f301969`): the embedded-PyO3 ↔ pure-client request assembly collapsed onto one proto seam (`jammi_client/_assembly.py` + `jammi_ai::wire` decoders) — every later verb addition now meets at the proto once (`transport-parity-decision`). |
| 3.8 | API-stability staging (annotations + contract-test lift) | **OPEN — lands LAST in H3** | provisional/stable annotation on every public verb; lift the cookbook's grounded-API-reference contract test into engine CI; broaden the typed error taxonomy past channels/models. Sequenced **after** `§3.6`+`§3.5` so the annotation pass runs once over the feature-complete H3 surface (rather than re-touching it). The channel typed-error work (`#193`) + `§3.6`'s `ModelReferenced` already deliver the taxonomy incrementally. |
| 3.5 | Multi-tenant contract + BYO-auth seam | **PARTIAL** | LANDED: `with_tenant` → `set_tenant`/`tenant_scope` (`#60`/`#161`, 0.26.4). OPEN: standing isolation oracle across the full verb surface; measured multi-tenant chapter at the scale tier; BYO-auth seam (Flight SQL / gRPC interceptor) documented + worked example. |
| 3.6 | Catalog & lifecycle completeness | **OPEN** | hard-delete-when-unreferenced (typed `ModelReferenced` error); model version/promotion verbs; source schema evolution; the `§3.6` migration; migration load-tests on SQLite + Postgres at scale. Untouched since `v0.29.0`. |

Cookbook side: 17 chapters, caught up to **H2** (`#19` = the `fine_tune(task="regression")`
loop-closer). **No H3 engine surface is validated by a cookbook chapter yet** — the
`§3.7`+`§3.8`-channel batch above is merged but unreleased, so the cookbook (pinned
0.29.0) cannot import it until the engine cuts the release.

---

## Release model — ONE release at H3-close (decision, this session)

**H3 ships as a single `v0.30.0 = "operability & contracts"`, cut when all of H3
is done — not per workstream.** Rationale (user call, validated):
- The cookbook CI installs the engine via `pip install -e ".[book,dev]"`, which
  resolves the `jammi_ai==X` pin **from PyPI** — so a cookbook chapter can only
  *merge green* against a **published** wheel. That is the *only* driver toward an
  early release, and it does not require one.
- The engine↔cookbook loop's value (catching design/behavior bugs early) comes from
  **authoring + running** a chapter against the engine, not from *merging* it — and
  the emit/authoring path already runs against a **locally-built** engine (precedent:
  the GPU keystone emit ran against a local `jammi-server`, then pinned the published
  wheel for the committed CI path). So the loop stays tight against `main` builds
  while only one release is cut.
- No consumer is waiting (the deferred-1.0 posture), so releasing the already-landed
  `#193`–`#196` now buys nothing and adds re-pin churn. **Cleaner on fix-cost:** every
  H3 bug a chapter surfaces is fixed on `main` *before* release, so `v0.30.0` ships
  clean with no intermediate patch releases.
- **The one cost:** the cookbook H3 chapters are authored on branches against local
  builds and **merge as a batch** once `v0.30.0` publishes and the pin moves to
  `==0.30.0`. A workflow-tidiness tradeoff, not a correctness one.

## H3 execution — engine on `main` (serial), cookbook authored against local builds

`§3.7`, `§3.8`-channel (`#193`), and `§3.8` transport-parity (`#197`) are already
**landed on `main`, unreleased**. Remaining engine work runs **serial** (Rust, ≤2
build-heavy agents on this 8-core box) in the order **W1 `§3.6` → W2 `§3.5` → W3
`§3.8`-staging** (the `E1/E2/E3` labels below are superseded by this order — `§3.6`
before `§3.5` so the isolation oracle covers `§3.6`'s new verbs; `§3.8`-staging last so
the annotation pass runs once over the complete surface). Each cookbook chapter is
authored + run against a **local engine build** as its workstream lands, on a branch;
the batch merges after the single `v0.30.0` publish.

**PR sizing** (`pr-sizing-rule`): each workstream below = **one PR** (one
pressure-test / one audit / one CI gate / one merge), sliced into **commits** for
granularity — NOT fragmented into many PRs (agentic flow; PRs cost CI minutes +
latency, and correctness comes from the rigor chain, not diff size). So E1, E2, E3 are
three engine PRs; the cookbook H3 chapters are scoped the same way at `v0.30.0`.

### E1 — `§3.8` API-stability staging + the transport-parity collapse — DO FIRST
- **Engine (foundational — the parity collapse first):** collapse the embedded ↔ client
  per-transport verb duplication into one config-assembly above a thin transport
  primitive (every later verb addition otherwise doubles — `transport-parity-decision`);
  add a provisional/stable annotation to every public verb; **lift the cookbook's
  grounded-API contract test into engine CI** (the engine owns its own reference-contract
  gate); broaden the typed error taxonomy past channels.
- **Cookbook chapter (deferred-merge):** the **channel error-taxonomy chapter** —
  drive `register_channel` / `add_channel_columns` / `list_channels` on `grpc://`, assert
  each failure maps to its **typed gRPC code** (duplicate → `AlreadyExists`, unknown →
  `NotFound`, column conflict → `FailedPrecondition`, bad id/type → `InvalidArgument`),
  and a genuine DB fault still → `Internal` — the cross-transport validation of `#193`.
  Extends ch14. Plus a short "API stability contract" section (provisional vs stable for
  the verbs the book uses), coordinated with the lifted-in contract test.

### E2 — `§3.5` multi-tenant contract + BYO-auth seam
- **Engine:** a standing isolation oracle across the **full** verb surface (catalog
  reads, search, propagate, query, mutable, topics); the BYO-auth seam (Flight SQL /
  gRPC interceptor) where a consumer plugs identity — the engine ships the seam, never
  the auth.
- **Cookbook chapter (deferred-merge):** the measured **multi-tenant chapter at the
  scale tier** — listing isolation + discriminator-column isolation as **hard zeros** +
  the honest global-source caveat, across the verb surface — plus the **BYO-auth seam
  worked example**. Extends ch11. (No consumer vocabulary — `Names no consumer`.)

### E3 — `§3.6` catalog & lifecycle completeness
- **Engine:** hard-delete-when-unreferenced (typed `ModelReferenced` error); model
  version/promotion verbs; source schema evolution; the `§3.6` migration; migration
  load-tests on SQLite + Postgres at the scale tier.
- **Cookbook chapter (deferred-merge):** the **lifecycle chapter** — register → promote →
  retire → delete with referential-integrity assertions; migrations validated at scale on
  both backends.

### R — cut `v0.30.0` + merge the cookbook H3 batch — closes H3
- **Engine:** with `§3.7`/`§3.8`/`§3.5`/`§3.6` all on `main` and green, cut
  `v0.30.0 = "operability & contracts"` (workspace + PyPI `jammi-ai`/server); CHANGELOG
  `[Unreleased]` → `[0.30.0]`.
- **Cookbook:** re-pin `jammi_ai==0.30.0`; re-introspect the wheel
  (`scripts/check_api_reference.py`); merge the E1–E3 chapters as a batch, each green
  through the six-step gate; update EXECUTION-STATUS + this plan.
- **At merge: H3 CLOSED.**
- **Note:** if a natural mid-H3 checkpoint emerges (e.g. `§3.8` fully done and stable
  while `§3.5`/`§3.6` are still large), revisit splitting into two releases — but default
  is one.

---

## H4 — the 1.0 engineering bar (shipped as a terminal 0.x; tag deferred)

After H3, drive `§4` to the engineering bar:
1. **The breadth grid, fully green.** Every public verb has all four: (a) a measured
   cookbook chapter, (b) an adversarial oracle in CI, (c) a remote+embedded parity test,
   (d) a scale benchmark. The verb-coverage grid → 100% (`engine-cookbook-loop`). This is
   the bulk of H4 cookbook work — fill every uncovered cell.
2. **The harder guarantees, stated and proven.** Trigger-stream delivery (at-least-once /
   exactly-once, replay completeness after restart); transactional guarantees spanning
   catalog + result-table storage; mutable-companion-table crash-consistency — each with a
   property test / model-checked argument + a measured cookbook chapter.
3. **Performance SLOs** — documented, gated targets for the core verbs at named scales.
4. **Storage/wire format stability** — Parquet result-table + sidecar-index + proto
   schemas versioned, forward-compatible, documented upgrade path.
5. **Security posture (open-core)** — trusted-network boundary documented + published
   threat model + the stable BYO-auth seam (from Wave C).

Terminal 0.x when the grid is green and the guarantees proven. **Stop there** — the 1.0
tag is the user's deferred post-adoption decision.

---

## Sequencing & orchestration constraints

- **Engine chain is SERIAL** (Rust builds; ≤2 build-heavy agents on this 8-core box —
  `orchestration-model`). Cookbook/CPU loop-closers parallelize *once their engine
  release exists*.
- **Hard dependency direction:** each wave's engine release precedes its cookbook
  loop-closer (the cookbook re-pins to the release; it never imports unreleased engine
  code, and never patches the engine — `Don't touch engine repos`).
- **Rigor chain on every PR:** plan → adversarial pressure-test (+ citation research) →
  implement → independent audit → PR → watch CI → **merge on green** (manual
  merge-watcher; no branch protection — `marathon-decisions`).
- **Autonomous 0.x releases**, public repos, this box, self-paced loop.
- **Re-baseline at the start of every wave** (Step 0): reconcile trackers, re-derive this
  plan against engine `main`, green baseline before changing anything.

## Immediate next action (re-derived at Step 0, 2026-06-15 — supersedes the E1 ordering below)
**`§3.6` catalog lifecycle is the next engine workstream.** `§3.8` transport-parity
collapse is **DONE** (engine `#197`, on `main`); `§3.8`-channel-errors (`#193`) and
`§3.7` (`#194`–`#196`) are also landed. So the remaining engine chain runs **SERIAL** in
this order: **`§3.6` lifecycle → `§3.5` multi-tenant → `§3.8` API-stability staging**
(annotations + lift-the-contract-test land **last**, over the feature-complete surface, so
the provisional/stable pass is done once). The `§3.6`-before-`§3.5` order is binding: the
`§3.5` isolation oracle must cover `§3.6`'s new lifecycle verbs (engine
`H3-WAVE-PLAN.md` cross-workstream rule). The single `v0.30.0` is cut only at H3-close
(step R), once `§3.6`/`§3.5`/`§3.8`-staging are all on `main` and every H3 cookbook
chapter has run green against a **release-recipe** local build. The `§3.8`
channel-status-codes chapter (validating `#193`) can be authored against a local build now.
