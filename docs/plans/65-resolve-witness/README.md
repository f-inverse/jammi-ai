# 65 — Resolve-witness / cache-rekey (front-door record)

**Status:** SCOPED by the unit-62 fingerprint-subsystem design pressure-test (2026-08-28,
REFINE). NOT planned, NOT implemented. Sequenced after units 62/63/64 as its own unit.

## Why this unit exists

Unit 62's staleness subsystem enforces the NARROW contract its escape (esc-058) specified:
detect in-place mutation of the FILES the resolver selected under the recorded resolve
inputs. The design review proved the WIDE contract — "a warm serve equals what a cold
resolve+load would produce, in every respect" — is structurally unreachable by that
representation: several resolver inputs are not files, and two file inputs are outside the
expressible anchor. Those classes are THIS unit's scope:

1. **Fine-tuned retrain invisibility (sharpest member):** the catalog `artifact_path` is
   the commit pointer a reload resolves from (schema.rs:444-448); a retrain's finalize CAS
   rewrites it (training_repo.rs:294); `fetch_artifact` yields content-addressed IMMUTABLE
   dirs — so a warm entry's fingerprint can never change while a cold resolve serves the
   new adapter. esc-057's mutable-pointer defect recurring one layer up.
2. **Catalog-shadow divergence:** `ModelSource::Local(P)` shadowed by a catalog row with
   `artifact_path A != P` — the fingerprint reasons about A while cold-resolve fallthrough
   (resolver.rs:119-129) can succeed via P.
3. **Cache key narrower than the resolve key:** entries keyed by ModelId alone while
   resolve() takes (source, task, backend_hint); task mismatch is guarded by typed refusals
   (defence-in-depth landed in 62), backend_hint mismatch has NO guard. Candidate fix:
   rekey on (ModelId, ModelTask, Option<BackendType>) — the design point 62 deliberately
   declined to reach.
4. **HF revision witness:** hf-hub resolves via `refs/<rev> -> snapshots/<sha>` — the refs
   file is the mutable pointer, outside the fingerprint anchor, structurally inexpressible
   as an arm; snapshot blobs themselves are immutable (the subsystem is honestly a no-op
   for HF sources today, documented as such in 62).
5. **Remote-glob weights chains:** shard sets derive from a remote sibling listing — not
   expressible as a finite arm list.

## The design direction (from the review; the unit's own pressure-test refines)

- **Resolver-emitted witness:** `ResolvedModel` carries a `resolve_witness` produced BY the
  resolver (catalog row (artifact_path, backend, model_type, base_model_id, updated_at);
  HF refs commit; selected-arm provenance) — one enumeration, produced by the code that
  resolved, compared on probe. This terminates the mirror-the-resolver loop by
  construction.
- **Rekey the cache** by the full resolve key, demoting 62's typed-refusal guards to
  defence-in-depth.
- **Chain-const extraction** (CONFIG/WEIGHTS/TOKENIZER name lists shared by resolver and
  enumerator) — the cheap name-level half, possibly folded earlier.
- **Probe cadence** (per-entry debounce): free once 62's bounded-staleness doc lands;
  the knob belongs here, not in 62.

## Boundary with unit 62 (binding)

62 ships the narrow contract PINNED IN ITS DOCS with this unit's classes enumerated as
out-of-domain, plus the in-scope liveness/wedge fixes and honest doc corrections. Nothing
in this unit blocks 62's ship; item 1's harm window (warm process spanning a retrain)
is recorded there as a known residual with this unit as its fix vehicle.
