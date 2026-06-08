# K-rails — the three rails + the closed eval loop

**Status:** spec — draft (hand-off). Build after KV-arxiv is green; authored **against the committed cache** (no heavy re-execution). One PR.
**Purpose:** make the three rails — **provenance, tenancy, measurement** — first-class and legible across the book, and stand up the **closed eval loop** (`construct → propagate → learn → MEASURE`). The rails are the columns of the 4×3 grid; this spec ensures every tier recipe actually carries them rather than gesturing at them.

This deepens the thin `jammi_cookbook/rails.py` helpers from K1 and adds a dedicated "rails" chapter plus a closing "closed eval loop" chapter.

---

## Rail 1 — Provenance

- **What it is in Jammi:** evidence channels (`vector`, `inference`, `conformal`, `uncertainty`) merged at query time; the `uncertainty.context_ref` column recording *which rows informed a prediction*; the `source` fact (`ann|edges|hybrid`) on context-conditioned outputs.
- **Recipe content:** for a tier-04 prediction (from the cached arxiv predictor), show the full provenance: the `source` fact, the `context_ref` member keys, and (if a custom channel was registered) the declared evidence columns. Make the point: *a prediction here is auditable — you can see the exact rows and the assembly that produced it.*
- **Bridge note:** this is what neither the monograph (paper) nor a PyG notebook (throwaway) carries — provenance as a first-class, queryable property of every result.
- **Helper:** `rails.provenance(result)` extracts + displays the channel columns / `context_ref` from a result.

## Rail 2 — Tenancy (the two-tenant test)

- **What it is in Jammi:** `with_tenant(t)` scopes every catalog read + source predicate; the engine knows there is a tenant, nothing about who; isolation is enforced by the analyzer rule.
- **Recipe content (the showcase lives in KV-air, deepened here):** load a second cohort under a second tenant; show that construction, search, propagation, and prediction under tenant A never see tenant B's rows — including across declared edges (a cross-tenant edge endpoint is unmaterialisable). State the discipline-test framing: this is *generic isolation*, not a domain feature.
- **Bridge note:** the monograph + the graph-conformal literature assume a single global graph; tenancy is the axis they're silent on and the substrate's distinctive contribution.
- **Helper:** `rails.tenant(db, t)` wraps `with_tenant` + the cross-tenant leak assertion.

## Rail 3 — Measurement (R1/R2)

- **What it is in Jammi:** `eval_embeddings`/`eval_compare`/`eval_inference` (R1) + `eval_calibration` (R2), with per-query records and significance.
- **Recipe content:** every tier recipe ends in a measured verdict read from the cache + asserted against `golden_metrics.json`. This chapter consolidates the measurement discipline: a recipe without a real number (computed from committed artifacts) is not done — the no-deferral policy applied to numbers.
- **Helper:** `rails.measure(...)` dispatches to the right `eval_*` and asserts against `golden(metric)` with tolerance.

## The closed eval loop (the closing chapter)

- **Content:** run the whole spine as one measured loop on the cached arxiv artifacts — `construct → propagate → learn → predict & quantify → MEASURE` — and show the end-to-end story: did each stage move the metric, with significance, and did the tier-04 coverage claim hold after the inline repair. This is the end-to-end measured proof that the substrate delivers — one runnable loop from raw data to a calibrated, audited prediction.
- **Determinism:** reads committed artifacts only; asserts the full chain of golden metrics; CI-executed.

## Success criteria

The three rail helpers are real and used by every tier recipe (not just this chapter); the tenancy two-tenant test passes; every recipe carries provenance + a measured verdict; the closed-eval-loop chapter runs `--execute` green against the cache and asserts the full golden-metric chain; no recomputation of upstream; no band-aids.
