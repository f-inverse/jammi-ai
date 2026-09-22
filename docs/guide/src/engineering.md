# Engineering Principles

[Design Philosophy](./philosophy.md) says what Jammi is and where its boundary falls. This
page says how its code is written. Both apply at every level — engine code, tests, fixtures,
CI scripts, docs.

## Code principles (non-negotiable)

These apply to every line of code written in this project. No exceptions. No "we'll clean it
up later."

### Clean, functional style — right abstractions, recursion, stack-safe

- **Find the right abstraction first.** Two things that are "the same thing at a different
  scale" are *one* thing — model them once and apply one operator at every level, never
  special-case the layers. A new special case is a smell that the abstraction is wrong; fix the
  abstraction, don't add the case. A typed refusal that names a topology, or says a capability
  arrives "later", is such a special case: a missing capability, not a finished one.
- **Think recursively.** When the structure is recursive (trees, composition graphs, nesting),
  express it as recursion over a well-founded structure — define the operator once and let it
  descend — not bespoke per-level code.
- **Stack-safe implementation.** Rust does not guarantee TCO, so a correct recursion that can
  blow the stack is a band-aid: use an accumulator, an explicit work-stack, or an iterative
  form, and keep it well-founded (bounded depth, DAG, no cycles). No unbounded recursion on
  unbounded input.
- **Separate deciding from doing.** A decision — a plan, a topology, an order — is a pure
  function of data, testable without I/O; the code that acts on it is a thin shell.
- Favor composition over inheritance.
- Use iterators, combinators, and pattern matching over imperative loops.
- Prefer pure functions — inputs in, outputs out, no side effects where avoidable.
- Use `Result` propagation (`?`) over panics. `unwrap()` only in tests.

### Clear boundaries, separation of concerns, right encapsulation

- **Encapsulate to the minimal correct surface.** A module exposes the smallest interface its
  consumers need and hides its representation; callers depend on behavior, not internals. Right
  encapsulation is the pair that makes the right abstraction safe to change without breaking
  callers.
- Every module has one responsibility. If you can't state it in one sentence, split it.
- Traits define boundaries. Concrete types live behind traits at module edges.
- No module may reach into another module's internals. Public API only.
- Lifecycle (load / cache / evict) and execution (infer / batch / adapt) are separate
  concerns — never mix them.

### DRY

- If logic appears twice, extract it. No copy-paste code — in tests and fixtures too.
- Shared behavior goes into traits or utility functions, not duplicated match arms.
- Configuration constants live in one place.

### No backwards compatibility

- This is a greenfield project. No shims, no deprecated paths, no "keep the old way around".
- If something needs to change, change it everywhere. Break and rebuild correctly.
- No `#[deprecated]`, no `_unused` renames, no compatibility re-exports.

### Type-driven design

- Use newtypes (`ModelId`, `GpuPermit`) to make invalid states unrepresentable.
- Use enums over stringly-typed parameters, and match them without a wildcard arm, so a new
  variant is a compile error at every site that must handle it.
- RAII for resource management (permits, guards, connections).
- Builder pattern where construction has more than 3 parameters.

### No band-aids

- Find the root cause. Don't paper over a symptom.
- Tell-signs to grep for and never accept: `#[allow(dead_code)]` to silence a warning instead
  of deleting the code, `let _ = expr` to discard a `Result` you don't want to handle,
  `unwrap_or_default()` over a value you don't understand, `// TODO: fix later` left in
  committed code, `#[cfg(any())]` or `#[ignore]` to disable a failing test, an `unreachable!()`
  for a case the types could have excluded, a comment justifying why a wrong primitive is
  "fine here".
- If a function's shape forces you toward a band-aid, the function shape is wrong. Reshape it.
- "I don't have time right now" is a band-aid scheduled for later. Do the right thing the
  first time.

### Engine, not platform

Jammi names no consumer — in code, config, docs, tests, fixtures, or scripts — and every
capability entering the engine passes the discipline test. The full statement, with the
boundary table and the deployment shapes, is [Design Philosophy](./philosophy.md); it is kept
in that one place. Generic fixtures (`patents.parquet`, synthetic triplets, small
public-domain text) live in `tests/fixtures/` or `crates/jammi-test-utils/`; a specific
consumer's data shape never does.

### Atomic across the workspace

- Behavior changes ship atomically across every affected crate. A trait change in `jammi-db`
  includes the corresponding updates in `jammi-ai`, `jammi-server`, `jammi-cli`, and
  `jammi-python` in the same PR.
- The lockstep `workspace.package.version` reflects this — every publishable crate ships at
  the same version.
- If a change is too large for one PR, split by *capability* (introduce the new trait first
  with no callers; migrate callers in a follow-up), never by *crate* (engine in PR1, ai in PR2
  leaves the workspace inconsistent between merges).

## Deciding at a fork

Derive the answer; do not pick by taste or by what is least work.

1. Apply the [discipline test](./philosophy.md#the-discipline-test) and the leak-guards.
2. Apply the principles on this page.
3. If it is still open, find how solid prior art decided it, and cite it in the change.

A design that needs many rounds to defend is usually answering the wrong question.

## Docs and comments reflect current state

User-facing docs (`README.md`, `docs/guide/`, rustdoc on public items, in-source comments)
describe the system as it IS, not the journey it took to get here. The phase plans under
`docs/plans/` are the only place journey-shaped writing belongs.

- No "added in PR #N", "since v0.2", "delivered in Phase X" markers in rustdoc or code comments.
- No `// removed once X lands`, `// legacy from arc-Y`, `// TODO: phase 4` comments.
- No `MIGRATION.md` at the repo root. Schema changes live as numbered append-only migrations;
  the runtime code does not narrate them.
- Comments explain hidden invariants — non-obvious constraints, surprising behavior, subtle
  ordering. Never history, and never the review or incident that produced the code.
- If a doc is wrong, fix it. Don't add a "(legacy — see X)" note alongside the new content.

## Test discipline

- Default `cargo test` must be fully hermetic. No live network calls. Live tests are gated
  behind a feature (`live-hub-tests` and its siblings).
- Every test exercises a real use case with realistic data — no fake column mappings or dummy
  inputs to dodge missing functionality. "Realistic" includes *to the model under test*: a
  fixture whose texts a tiny tokenizer maps to the same token is a dummy input.
- A test has proven nothing until it has been seen to fail for the reason it exists. Break the
  code it guards and watch it go red before trusting it green.

## Invariants every change keeps

| Invariant | What keeps it |
|---|---|
| Replay is complete: every `ProducingDescriptor` variant has a `replay_descriptor` arm. | compiler exhaustiveness (no `_` arm) |
| An identity hash folds the producer's **complete** output-affecting parameter set. | per-variant determinism tests on `definition_hash` |
| Validate, clamp or normalise at every numeric and catalog input edge; nothing computes confidently past its valid domain. | a boundary/degenerate test per operator |
| A trainable head on a high-offset or low-variance target standardises in data space (a persisted scaler), never by rescaling the loss. | a high-offset test per head |
| The remote surface matches the embedded path byte for byte, not merely "both respond". | the server parity suite |
| Migrations are append-only and monotonic: appended to `MIGRATIONS`, names never reused or reordered. | `EXPECTED_MIGRATION_NAMES` in `crates/jammi-db/tests/it/migrations.rs` |
| Jammi names no consumer, and dependencies point one way. | `ci/scripts/check_no_consumer_names.py`, `ci/scripts/check_dep_direction.py` |

## How a principle is enforced

In order of strength. Reach for the highest one that fits.

1. **The type system.** The wrong program does not compile.
2. **A test that can fail** — see Test discipline above.
3. **A CI check of a property of the product** — dependency direction, a release manifest, a
   doc that mirrors an enum. A check exists to protect a user or a contributor; one that only
   polices how the work was done does not belong in CI.
4. **Review**, for what nothing above can hold. Say so plainly where that is the case, so the
   gap is visible rather than assumed covered.

More checks are not more enforcement. Every check is code someone must maintain and every
contributor must pass; add one when it protects something, and delete it when it no longer does.

The checks that need no build live in one list, `ci/guards.toml`: each entry names its command,
the property it holds, and the paths that can affect it. One runner executes them, in CI and
locally alike, so a guard that passes on your machine passes in CI:

```bash
ci/dev.sh python3 ci/scripts/run_guards.py --base origin/main   # the guards your change can affect
ci/dev.sh python3 ci/scripts/run_guards.py                      # all of them
```

A guard runs its assertions or is not selected: it declares what it `needs` of its host, the runner
provides that before running it, and a need still missing fails the run by name. A guard whose
host no command can make of the CI image — a workspace build beside a PyTorch venv — declares a
`lane` too, and only a run on such a host selects it:

```bash
python3 ci/scripts/run_guards.py --lane torch-host
```

The lanes CI runs against a service — the Postgres arms, the distributed lane over Postgres and
the S3-class store — run locally the same way, with the backends the workflows declare provided for that run
alone and removed when it exits:

```bash
ci/dev.sh --with pg cargo test -p jammi-db --features live-postgres-tests --test it -- --test-threads=1
ci/dev.sh --with pg,s3 cargo test -p jammi-ballista --features live-distributed-tests --test distributed -- --test-threads=1
ci/dev.sh --gc          # remove whatever earlier runs left behind, keeping the build caches
```

## Self-check before completing any task

Before declaring work done, verify:

- [ ] No duplicated logic introduced
- [ ] Every new public type/trait has a clear single responsibility
- [ ] No module reaches into another module's internals
- [ ] No temporary APIs or compatibility shims
- [ ] New interfaces are downstream-driven (only what consumers need)
- [ ] Tests are in the right category (unit / contract / integration / live)
- [ ] Every test exercises a real use case with realistic data, and each new one has been
      seen to fail
- [ ] `cargo clippy` and `cargo fmt` pass — `ci/dev.sh` runs them as CI does
- [ ] Code reads as idiomatic Rust, not translated Java/Python
- [ ] Scaffolding earned — new directories only when there are multiple files; new files only
      when the code doesn't belong next to its producer or consumer
- [ ] Names match shape — typed errors, function names, and trait method names describe their
      actual parameter shape, not what they were originally intended for
- [ ] Idiom matches the surrounding file — recursion vs iteration, `?` vs `match`, owned vs
      borrowed all follow the conventions of the file the change lives in
- [ ] No band-aids — no new `#[allow(…)]`, no new `let _ = …`, no new `// TODO: later`, no
      comments justifying a wrong primitive
- [ ] No consumer/tenant name appears anywhere in this change (code, config, docs, tests,
      fixtures, scripts) — references point one way only
- [ ] Any new engine surface passes the discipline test — justifiable for unrelated
      hypothetical consumers, not a named one
- [ ] User-facing docs and comments describe the system as it IS, not the journey to it
- [ ] Changes ship atomically across every affected crate in the same PR

## Dodges that don't fly

These phrases are red flags in a PR description, a commit message, or self-defense to a
reviewer:

- *"Spec said it"* — the spec might be wrong; you read the spec, so catching the wrongness is
  your job.
- *"Out of scope for this PR"* — if it's a band-aid you noticed and didn't fix, file the
  follow-up issue in the same PR. Don't ship the band-aid with no trace.
- *"Minimal change"* — minimal does not mean *smallest diff that compiles*. Minimal means
  *smallest diff that's correct*. A two-line fix that introduces a wrong primitive is larger
  than a fifty-line fix that uses the right one.
- *"I'll clean it up later"* — there is no later. Do the right thing the first time.
- *"Existing code does it this way"* — existing code is not a constraint. If a test is right
  and the code is wrong, fix the code. If a function's shape makes a correct caller impossible,
  reshape the function.
- *"A limit of this version"* — a capability that works on one route and is refused on another
  is unfinished, however typed the refusal.
