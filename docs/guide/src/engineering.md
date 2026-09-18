# Engineering Principles

[Design Philosophy](./philosophy.md) says what Jammi is and where its boundary falls. This
page says how its code is written. Both apply at every level — engine code, tests, fixtures,
CI scripts, docs — and a change that satisfies a test while breaking one of these is not done.

## Greenfield first

Nothing that exists is a constraint. When a design turns out not to generalise, the existing
code is broken down and rebuilt the way it should have been — it is never worked around.

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

## Functional by default

- **Separate deciding from doing.** A decision — a plan, a topology, an order, a view of a
  spec — is a pure function of data, testable without I/O. The code that acts on it is a thin
  shell. A function that both decides and acts gets split.
- **One concept, one home (DRY).** Before adding a second match, helper, or arm, find the first
  and generalise it. Producers may be plural; the reader is singular. This holds for tests:
  a shared fixture lives once, never pasted per crate.
- **Boundaries are types, not comments.** If a caller must not reach something, make it
  unreachable or unrepresentable. A rule stated in a doc comment and kept by review is a rule
  that will be broken. Callers consume a small value that says what they need, rather than
  matching on the variants of a large enum they do not own.
- **Separation of concerns is measured in files too.** A source file that needs a table of
  contents is several modules. Split along the seam a change exposes rather than adding to it.
- **Comments describe the code**, for the next reader — not the review, incident, or
  discussion that produced it. That history belongs in the commit message.

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

More checks are not more enforcement. Every check is code someone must maintain and every
contributor must pass; add one when it protects something, and delete it when it no longer does.
