# Engineering Principles

[Design Philosophy](./philosophy.md) says what Jammi is and where its boundary falls. This
page says how its code is written. Both apply at every level — engine code, tests, fixtures,
CI scripts, docs — and a change that satisfies a test while breaking one of these is not done.

## Greenfield first

Nothing that exists is a constraint. When a design turns out not to generalise, the existing
code is broken down and rebuilt the way it should have been — it is never worked around.
There is no backwards compatibility to keep: no shims, no deprecated paths, no `#[deprecated]`,
no `_unused` renames, no compatibility re-exports, no "keep the old way around". If something
needs to change, it changes everywhere, in one change.

The shape to watch for is a second path beside the first. If a capability works on one route
and needs a special case, a flag, or a refusal on another, the first route was the wrong
abstraction: generalise it and delete the other. A typed refusal that names a topology, or
says a capability arrives "later", is a missing capability, not a finished one — the
[one-binary rule](./philosophy.md#how-it-deploys-one-binary-pluggable-backends) already says
no feature exists in one deployment shape and not another.

## Deciding at a fork

Derive the answer; do not pick by taste or by what is least work.

1. Apply the [discipline test](./philosophy.md#the-discipline-test) and the leak-guards. Would
   a user who has never heard of any particular consumer reach for this?
2. Apply the principles on this page.
3. If it is still open, find how solid prior art decided it, and cite it in the change.

A design that needs many rounds to defend is usually answering the wrong question.

## Code principles

These apply to every line. No exceptions, and no "we'll clean it up later."

### Clean, functional style

- Favor composition over inheritance.
- Iterators, combinators and pattern matching over imperative loops.
- Prefer pure functions — inputs in, outputs out, no side effects where avoidable.
- **Separate deciding from doing.** A decision — a plan, a topology, an order, a view of a
  spec — is a pure function of data, testable without I/O. The code that acts on it is a thin
  shell. A function that both decides and acts gets split.
- `Result` propagation (`?`) over panics. `unwrap()` only in tests.

### Clear boundaries and separation of concerns

- Every module has one responsibility. If you can't state it in one sentence, split it. A
  source file that needs a table of contents is several modules.
- Traits define boundaries. Concrete types live behind traits at module edges.
- No module reaches into another module's internals. Public API only.
- Lifecycle (load / cache / evict) and execution (infer / batch / adapt) are separate
  concerns — never mixed.
- Callers consume a small value that says what they need, rather than matching on the
  variants of a large enum they do not own.

### DRY

- If logic appears twice, extract it. No copy-paste code — in tests and fixtures too: a
  shared fixture lives once, never pasted per crate.
- Shared behaviour goes into traits or functions, not duplicated match arms. Before adding a
  second arm or helper, find the first and generalise it. Producers may be plural; the reader
  is singular.
- Configuration constants live in one place.

### Type-driven design

- Newtypes (`ModelId`, `GpuPermit`) make invalid states unrepresentable. If a caller must not
  reach something, make it unreachable — a rule kept by a doc comment and review is a rule
  that will be broken.
- Enums over stringly-typed parameters.
- RAII for resources: permits, guards, connections.
- A builder where construction takes more than three parameters.

### Tests, comments and docs

- Default `cargo test` is fully hermetic: no live network calls. Live tests sit behind a
  feature (`live-hub-tests` and its siblings).
- Comments describe the code, for the next reader — not the review, incident or discussion
  that produced it. That history belongs in the commit message.
- Docs describe the system as it is, not the journey: no "added in PR #N", no "since v0.2".

## Invariants every change keeps

| Invariant | What keeps it |
|---|---|
| Replay is complete: every `ProducingDescriptor` variant has a `replay_descriptor` arm. | compiler exhaustiveness (no `_` arm) |
| An identity hash folds the producer's **complete** output-affecting parameter set. | per-variant determinism tests on `definition_hash` |
| Validate, clamp or normalise at every numeric and catalog input edge; nothing computes confidently past its valid domain. | a boundary/degenerate test per operator |
| A trainable head on a high-offset or low-variance target standardises in data space (a persisted scaler), never by rescaling the loss. | a high-offset test per head |
| The remote surface matches the embedded path byte for byte, not merely "both respond". | the server parity suite |
| Migrations are append-only and monotonic: appended to `MIGRATIONS`, names never reused or reordered. | `EXPECTED_MIGRATION_NAMES` in `crates/jammi-db/tests/it/migrations.rs` |
| A behaviour change ships across every crate it touches in one PR; every publishable crate shares `workspace.package.version`. | the workspace build; review |
| Jammi names no consumer, and dependencies point one way. | `ci/scripts/check_no_consumer_names.py`, `ci/scripts/check_dep_direction.py` |

## How a principle is enforced

In order of strength. Reach for the highest one that fits.

1. **The type system.** The wrong program does not compile. Exhaustive matches, private
   constructors, type-state, `compile_fail` doctests for a sealed surface.
2. **A test that can fail.** A test has proven nothing until it has been seen to fail for the
   reason it exists: break the code it guards and watch it go red. A fixture the model under
   test cannot distinguish, or a byte comparison nothing can perturb, is decoration.
3. **A CI check of a property of the product** — dependency direction, a release manifest, a
   doc that mirrors an enum. A check exists to protect a user or a contributor; one that only
   polices how the work was done does not belong in CI.
4. **Review**, for what nothing above can hold. Say so plainly where that is the case, so the
   gap is visible rather than assumed covered.

## Before calling a change done

- [ ] No duplicated logic introduced
- [ ] Every new public type or trait has one clear responsibility
- [ ] No module reaches into another module's internals
- [ ] No temporary APIs, compatibility shims, or refusals standing in for a capability
- [ ] New interfaces are downstream-driven — only what callers need
- [ ] Each test is in the right category (unit / contract / integration / live), and each new
      one has been seen to fail
- [ ] `cargo fmt` and `cargo clippy -- -D warnings` pass — `ci/dev.sh` runs them as CI does
- [ ] The code reads as idiomatic Rust, not translated Java or Python

More checks are not more enforcement. Every check is code someone must maintain and every
contributor must pass; add one when it protects something, and delete it when it no longer does.
