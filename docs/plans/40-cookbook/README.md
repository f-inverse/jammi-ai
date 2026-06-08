# 40-cookbook — The Theory↔Computation Cookbook (hand-off spec set)

**If you are a fresh session picking this up: read this file end-to-end, then `K0`, then `K1`, then build in the order under "Build order" below. Maintain `40-cookbook/EXECUTION-STATUS.md` as you go.**

---

## What this is

A **runnable book** that bridges *applied theory* (Ljubiša Stanković et al., *Data Analytics on Graphs*, Foundations & Trends in ML 13, 2020 — Part I graphs/spectra, Part II graph signal processing, Part III ML on graphs; **plus** the modern GNN canon: Hamilton *Graph Representation Learning*, GraphSAGE, the geometric-deep-learning framing) with *software and computation* (the `jammi_ai` Python package).

The **signature move** of every recipe: show that **one Jammi recipe = one equation in the monograph = one line in the GNN canon**, *executed and measured*. The book is organized "unified yet separate" like AWS Neptune (Database / Analytics / ML) but adds a **fourth tier Neptune structurally lacks** — calibrated, provenance-stamped, context-conditioned prediction.

It **reads** as **4 tiers × 3 rails**:

| | Construct (01) | Analyze (02) | Learn (03) | Predict & Quantify (04) |
|---|---|---|---|---|
| **provenance** | | | | |
| **tenancy** | | | | |
| **measurement** | | | | |

- Tier 01 Construct → `build_neighbor_graph` (monograph Part I, topology).
- Tier 02 Analyze → graph algorithms + `propagate_embeddings` (Part II, graph signal processing / low-pass filtering = SGC).
- Tier 03 Learn → `fine_tune` / `fine_tune_graph` (Part III, ML on graphs).
- Tier 04 Predict & Quantify → `train_context_predictor` / `predict_with_context_predictor` + conformal (no monograph **or** Neptune analogue — the moat).
- Rails (woven through every tier): **provenance** (evidence channels / `context_ref`), **tenancy** (`with_tenant`, the two-tenant test), **measurement** (`eval_*` — R1/R2; the book doubles as the closed eval loop `construct→propagate→learn→MEASURE`).

## This is NOT the S6 cookbook

A separate minimal cookbook lives **inside** `jammi-ai/cookbook/` (spec `10-substrate/S6-cookbook.md`). **This is a different, larger, consumer-facing book in its own repo** that *imports* `jammi_ai` as a dependency and **does not touch** `jammi-ai/cookbook/` or any jammi engine source — it only *calls* the `jammi_ai` API. Do not edit the engine repos to make a recipe work; a recipe that needs an engine change is a fork to escalate, not to patch (see "Fork-resolution").

## The artifact set (build these, in this order)

| ID | File | What it is |
|---|---|---|
| README | this file | START HERE: index + process discipline |
| **K0** | `K0-contracts-and-discipline.md` | artifact-contract registry (2 layers) · author-vs-execute · determinism template · philosophy gates |
| **K1** | `K1-scaffold-and-lib.md` | repo scaffold (Quarto), `jammi_cookbook` shared lib, CI `--execute` harness, the cookbook's own CLAUDE.md, the grounded `jammi_ai` API reference |
| **K2** | `K2-datasets.md` | Air Routes + ogbn-arxiv loaders; pinned versions/checksums; committed subset ID list |
| **KV-arxiv** | `KV-arxiv-vertical.md` | **keystone / de-risk slice** — ogbn-arxiv tiers 01–04 end-to-end; emits the committed artifacts + golden metrics |
| **KV-air** | `KV-air-vertical.md` | Air Routes tiers 01–02 (the Neptune-faithful on-ramp) |
| **K-rails** | `K-rails-and-eval-loop.md` | provenance/tenancy/measurement cross-cut + the closed eval loop |
| **K-bridge** | `K-bridge-chapters.md` | the equation=aggregate signature chapters + Neptune-contrast + the citation map |

## Build order (conservative, serial)

1. **K1** scaffold + **K0** contracts (schema layer). Foundation.
2. **K2** loaders (may be done together; trivially independent).
3. **KV-arxiv end-to-end** (the keystone): build the full arxiv vertical, get CI green, **emit the committed artifacts + `golden_metrics.json`**. This validates every API call (including the tier-04 crux) and populates the golden-sample contract layer.
4. **KV-air + K-rails + K-bridge**: authored **against the committed cache** (no heavy re-execution). A single session does these in sequence.
5. **Integration `--execute`** on the fixed subset + an opt-in full-scale run + `quarto render`.

**Parallelism is not required and not orchestrated.** The cookbook's bottleneck is notebook execution on one accelerator, so concurrent sessions would just queue on the GPU. Stay single-session and serial; the **artifact cache** (K0), not parallelism, is what makes this efficient.

---

## Process discipline (NON-NEGOTIABLE)

### 1. Per-spec lifecycle — every spec is one PR

```
plan → adversarial pressure-test (with research) → implement →
independent audit → remediate → open PR → watch CI to green → merge → release
```

- **Adversarial pressure-test:** before implementing, try to break the spec's approach against the philosophy harness and the real API; fold findings in. Research the relevant monograph/GNN-canon citations so the bridge notes are correct.
- **Independent audit (before the PR is opened):** run build + lint + `--execute` of the touched notebooks + the band-aid/principle scan (below) + a check against the spec's success criteria. **Remediate every finding before opening the PR.**
- **Watch CI to green:** poll `gh pr checks <N>` until **every** workflow passes (including the `--execute` notebook job). Never report a spec done while any job is pending or red.
- **Merge → release:** version-bump / tag / publish where applicable (the rendered book; any helper package). Record the merge in `EXECUTION-STATUS.md`.

### 2. No-deferral policy (greenfield)

**Every decision is build or cut — nothing is left in limbo.** Forbidden in committed work (the audit greps for these and fails on a hit):

```
TODO   FIXME   unimplemented!   todo!()   # type: ignore (unjustified)
"deferred"   "v1 later"   "for now"   commented-out code   <TBD>   placeholder numbers
```

If something cannot be built, **cut it with a written rationale** in the `EXECUTION-STATUS.md` decisions log — never half-ship it. A measured verdict is never a placeholder: if a number isn't real (computed from the committed artifacts), the recipe is not done.

### 3. Fork-resolution — decide from the harness, do not halt

When the spec set underdetermines a decision, **resolve it autonomously** by consulting, in order:

1. the **cookbook repo's `CLAUDE.md`** (created by K1 — the cookbook's own standards: determinism, read-the-cache, names-no-consumer, runnable+measured);
2. **jammi's philosophy doc** `jammi-ai/docs/guide/src/philosophy.md` (the discipline test; *transports/persists/merges, semantics live above*; *names no consumer*; *degrade governance, never service*);
3. the **source code itself** — the real `jammi_ai` Python API and current behaviour (grep + read; never guess a signature from memory).

Do **not** invent from training-data assumptions, and do **not** stop to ask the user on anything the harness can resolve. Record any non-obvious fork + its resolution in the `EXECUTION-STATUS.md` decisions log. (This mirrors the established autonomous-no-halts working mode on the jammi spec trains.)

---

## Keep the two axes separate

The book **reads** as 4 tiers × 3 rails (presentation). The work is **built** as a serial pipeline per dataset vertical + two cross-cut libraries (execution). Do not let the 4×3 grid tempt you into building four parallel "tier" chapters — within a dataset the tiers chain through artifacts (construct→graph→embeddings→learn→predict) and must be built as one serial pipeline. See K0.

## Maintain EXECUTION-STATUS.md

Create and keep `40-cookbook/EXECUTION-STATUS.md` (mirror the main one): a per-spec status table (spec · status · branch · PR · notes), a decisions log (every fork resolved + every cut with rationale), and a per-branch audit history (findings + remediations before each merge).
