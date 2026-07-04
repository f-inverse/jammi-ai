# Engine Constitution — the invariant index

**Authoritative. Human-amend-only.** This file is the canonical list of invariants the
swarm cites by ID. It is an **index, not a copy**: each row states an invariant in one
line and points at the canonical source that defines it and the code anchor that realises
it. It never re-states the philosophy — the source of truth for *why* is
[`docs/guide/src/philosophy.md`](../guide/src/philosophy.md); the source of truth for the
*rules* is [`CLAUDE.md`](../../CLAUDE.md). If this file and a canonical source disagree,
the canonical source wins and this row is the bug.

**Amendment.** No agent may edit this file. Any diff touching it fails closed at the
`CONSTITUTION_TOUCHED` CI gate and requires a human admin-merge. This is the
anti-Goodhart rule (ARCHITECTURE §2.7): the swarm may *tighten* itself autonomously but
may not *weaken* the constitution that governs it.

**Anchor freshness.** Every `code anchor` below is **typed** with a resolver kind —
`rust_symbol:<path>:<Symbol>`, `gate_script:<path>`, or `doc_heading:<path>#<heading>` —
that `ci/scripts/check_constitution_anchors.py` resolves on every PR. A row whose anchor no
longer resolves is a CI failure — so the constitution cannot silently drift from the code
it governs. Boundary invariants (their enforcement is a fail-closed gate, not a function)
carry a `gate_script` anchor to their live enforcer; a boundary invariant with no wired
gate is anchored `gate_script:UNENFORCED`, which the checker flags as an honest hole in the
enforcement surface. Correctness invariants carry `rust_symbol` anchors to the symbols that
realise them. The `checked by` column still says `discipline` where final enforcement is
human/agent judgement rather than the mechanical anchor.

The canonical-source anchors were verified against `docs/guide/src/philosophy.md` and
`CLAUDE.md`; the code anchors were verified to resolve on `feat/swarm-engine-core` at the
paths and symbols shown.

---

## Boundary invariants

The engine-not-platform line. These keep Jammi a set of generic primitives that names no
consumer. Most are enforced at the dep level by a gate and at the semantic level by
`discipline` (the `discipline-test-auditor` agent + human judgement) — a card that says
`discipline` has no mechanical teeth and says so plainly.

| ID | Statement | Canonical source | Code anchor | Checked by |
|---|---|---|---|---|
| **B1** | Before any capability enters the engine it must pass the discipline test: a user who has never heard of any particular consumer would still reach for it. | `docs/guide/src/philosophy.md#the-discipline-test`; `CLAUDE.md#engine-not-platform` | gate_script:ci/scripts/check_no_consumer_names.py | `check_no_consumer_names.py` (governance-verb-stem tripwire = mechanical half) + `check_dep_direction.py` (dep-level) + `discipline` (discipline-test-auditor) |
| **B2** | References point one way only: a consumer may depend on Jammi; Jammi depends on no consumer. A consumer's name anywhere in the engine repo is a bug. | `docs/guide/src/philosophy.md#the-one-rule-everything-else-follows-from` | gate_script:ci/scripts/check_no_consumer_names.py | `check_no_consumer_names.py` (name/leak-smell tripwire, swarm-wired) + `check_dep_direction.py` + `check_cookbook_one_way.sh` (dep-level, ci.yml) |
| **B3** | Embeddings are consumed through `search`, never through a dedicated vector-retrieval verb; the raw vector is reachable only as a SQL column. | `docs/guide/src/philosophy.md#how-embeddings-are-consumed-search` | gate_script:ci/scripts/check_no_consumer_names.py | `check_no_consumer_names.py` (`get_vector` raw-embedding leak-smell tripwire) + `discipline` (discipline-test-auditor) |
| **B4** | One binary serves every topology; a production topology is a backend-driver configuration change, never a topology-specific code path or server-only feature. | `docs/guide/src/philosophy.md#how-it-deploys-one-binary-pluggable-backends` | discipline:one-binary-no-clean-gate | `discipline` (no library-vs-server feature gate; pluggable-catalog seam `crates/jammi-db/src/catalog/backend.rs`) — enforced by `discipline-test-auditor` review; no clean mechanical gate exists |
| **B5** | Tenant session scope is a generic row/listing predicate; it names no consumer, and leak-guards keep semantics above the primitive (the stream/mutable-table/provenance guards). | `docs/guide/src/philosophy.md#leak-guards`; `docs/guide/src/philosophy.md#where-the-line-falls` | gate_script:ci/scripts/check_no_consumer_names.py | `check_no_consumer_names.py` (names-no-consumer, swarm-wired) + `check_dep_direction.py` + `discipline` (`crates/jammi-db/src/tenant_scope.rs:TenantScopeAnalyzerRule`) |
| **B6** | Behaviour changes ship atomically across every affected crate in one PR; the workspace is never left inconsistent between merges. | `CLAUDE.md#atomic-across-the-workspace` | discipline:workspace-atomicity-no-clean-gate | `discipline` (PR = one rigor-chain unit; lockstep `Cargo.toml:workspace.package` — see K6) — enforced by `adversarial-audit` review; atomicity has no clean mechanical gate (K6 covers the version-lockstep sub-part) |

## Correctness invariants

The compute-past-the-valid-domain guards and the identity/parity guards. These are the
recurring shapes of the marathon's real bugs (`docs/plans/51-marathon-learnings/`), pinned
as invariants so a future change cannot silently reintroduce them.

| ID | Statement | Canonical source | Code anchor | Checked by |
|---|---|---|---|---|
| **K1** | Every `ProducingDescriptor` variant has a matching `replay_descriptor` arm; the replay path is complete over the descriptor space. | ARCHITECTURE §6 (`db-materialization`); `CASE-STUDIES.md` §1 (one root cause, two homes) | rust_symbol:crates/jammi-db/src/store/manifest.rs:ProducingDescriptor; rust_symbol:crates/jammi-ai/src/pipeline/recompute.rs:replay_descriptor | compiler exhaustiveness + `crates/jammi-ai/tests/it/*`, `crates/jammi-db/tests/it/materialization.rs` + audit |
| **K2** | Validate / clamp / normalize at every numeric and catalog input edge; no operator computes confidently past its valid input domain. | `FIELD-NOTES.md#s1`; `CASE-STUDIES.md` §2 | rust_symbol:crates/jammi-ai/src/fine_tune/trainer.rs:compute_lr; rust_symbol:crates/jammi-ai/src/pipeline/graph_propagation.rs:augmented_degrees; rust_symbol:crates/jammi-db/src/index/exact.rs:BoundedTopK (exemplars; glob: every input edge) | boundary/degenerate oracle per op + adversarial sweep (`discipline`) |
| **K3** | Every trainable head on a high-offset/low-variance/large-magnitude target standardizes in data/representation space (a persisted scaler), never by rescaling the loss. | `CASE-STUDIES.md` §1; `FIELD-NOTES.md#o1` | rust_symbol:crates/jammi-ai/src/fine_tune/regression_loss.rs:TargetScaler; rust_symbol:crates/jammi-ai/src/fine_tune/target.rs:TrainingTarget | high-offset oracle per head, `fine_tune` correctness sweep (`discipline`) |
| **K4** | The remote (server/client) surface matches the embedded path byte-for-byte, not just "both respond". | `AGENTIC-PLAYBOOK.md` §2; ARCHITECTURE §5 (`oracle`) | rust_symbol:crates/jammi-server/tests/it/grpc_remote_session.rs:remote_round_trips_embeddings_and_search_like_local; rust_symbol:crates/jammi-wire/src/audit.rs:record_from_wire | server it-suite (gate) + oracle |
| **K5** | Catalog migrations are append-only and monotonic: a new migration is appended, names are never reused or reordered. | `CLAUDE.md` (Docs Reflect Current State — numbered append-only migrations) | rust_symbol:crates/jammi-db/src/catalog/migrations.rs:MIGRATIONS | `crates/jammi-db/tests/it/migrations.rs` + oracle (append-only numbering) |
| **K6** | Every publishable crate ships at the same `workspace.package.version` (lockstep). | `CLAUDE.md#atomic-across-the-workspace` | rust_symbol:Cargo.toml:workspace.package | oracle (lockstep version) |
| **K7** | A content-addressable identity folds the producer's **complete** output-affecting parameter set — audited for per-variant completeness, not just shape. | ARCHITECTURE §6 (`db-materialization`, hash completeness); memory: identity/hash completeness | rust_symbol:crates/jammi-db/src/store/manifest.rs:definition_hash; rust_symbol:crates/jammi-db/src/store/manifest.rs:DefinitionHash | `definition_hash_is_deterministic` test + per-variant completeness audit (`discipline`) |

---

## For the anchor gate

`check_constitution_anchors.py` reads each `code anchor` cell as one or more **typed**
anchors (`;`-separated) and resolves each by its kind:

- `rust_symbol:<path>:<Symbol>` — `<Symbol>` still appears in `<path>`.
- `gate_script:<path>` — the gate script exists **and is referenced in
  `.github/workflows/swarm.yml`** (an unwired gate enforces nothing). The sentinel
  `gate_script:UNENFORCED` marks a boundary invariant with no wired mechanical gate.
- `doc_heading:<path>#<heading>` — `<heading>` appears in `<path>`.

Every boundary invariant (B*) must carry a `gate_script` anchor. A `gate_script:UNENFORCED`
boundary invariant, an unresolvable anchor, or a parse failure is a non-zero exit naming
it — turning the check into a completeness check over the enforcement surface. Cells whose
`checked by` is `discipline` still carry a resolvable anchor so the mechanical check stays
meaningful; the `discipline` label marks that final enforcement is human/agent judgement,
not that the row is unpinned.

**Honest note (this tree):** B4 (one-binary / pluggable-backends) and B6 (atomic-across-the-
workspace) have **no clean mechanical gate** — a weak one would be theater — so they are
anchored `discipline:<rationale>`: a *conscious, human-gated* declaration (editing this file
trips `CONSTITUTION_TOUCHED`) that they are enforced by auditor review, not a fail-closed gate.
The anchor gate passes on a rationale-bearing `discipline:` anchor but still fails on a silent
gap (a boundary row with no `gate_script`/`discipline:`) or a claimed-but-missing gate
(`UNENFORCED`) — so enforcement can never be dropped unnoticed (ARCHITECTURE §7/§8). B1/B2/B3/B5 anchor to
`check_no_consumer_names.py`, the pure-Python name/governance tripwire wired into
`swarm.yml`; the dep-level hard gate `check_dep_direction.py` (which needs `cargo metadata`,
so it lives in `ci.yml`, not the toolchain-free `swarm.yml`) remains a real enforcer named in
the `checked by` column.
