# Did Structure Help? A Graph-ML Evaluation Recipe

When you produce a structure-aware embedding table — a fine-tuned model, a
propagated table, or any treatment that folds graph context into the
representation — the only question that matters is whether it *beats* plain
text embeddings on a held-out retrieval task. This recipe is the discipline
around `eval_compare` that turns a delta into a defensible conclusion.

Nothing here is new engine surface. `eval_compare` already computes
recall@k / precision@k / MRR / nDCG per table, the per-metric delta against a
baseline, and — paired by `query_id` over the per-query records — a
distribution-free significance result for each metric delta. The recipe is the
protocol: a clean split, the judgment-matched metric, multiple seeds for
trained treatments, and cohort slicing.

## The four steps

1. **Baseline.** Produce a plain text-embedding table with
   `generate_text_embeddings`.
2. **Treatment.** Produce each structure-aware table (a fine-tuned model's
   embeddings, a propagated table, etc.) over the **same source rows**.
3. **Compare.** Run `eval_compare` with the baseline table first. Read the
   per-metric delta *and* its paired significance.
4. **Conclude and slice.** Declare a win only when the judgment-matched metric
   improves with a significant paired test; then slice by cohort to see
   *where* structure helped.

## 1 & 2 — produce baseline and treatment tables

The baseline is a plain text-embedding table over your corpus. Each treatment
table must be built over the **same rows** so the comparison is apples-to-apples.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# use jammi_db::store::CachePolicy;
# async fn ex(session: &InferenceSession, baseline_model: &str, treatment_model: &str) -> jammi_db::error::Result<()> {
// Baseline: plain text embeddings.
let (baseline, _) = session
    .generate_text_embeddings("patents", baseline_model, &["abstract".into()], "id", CachePolicy::Bypass)
    .await?;

// Treatment: embeddings from a structure-aware model (e.g. a fine-tuned
// checkpoint), over the same source and key column.
let (treatment, _) = session
    .generate_text_embeddings("patents", treatment_model, &["abstract".into()], "id", CachePolicy::Bypass)
    .await?;

let baseline_table = baseline.table_name;
let treatment_table = treatment.table_name;
# let _ = (baseline_table, treatment_table);
# Ok(()) }
```

```python
baseline = db.generate_embeddings(
    source="patents", model=baseline_model,
    columns=["abstract"], key="id", modality="text",
)
treatment = db.generate_embeddings(
    source="patents", model=treatment_model,
    columns=["abstract"], key="id", modality="text",
)
```

> **Leakage contract (read before you build anything).** The graph and any
> graph-supervised training must use the **train split only**. The golden set
> — the `(query, judgments)` pairs you evaluate against — is **held out** and
> must never feed graph construction or training-pair selection. A
> structure-aware representation that has seen the eval rows will "win" by
> memorizing them, and the verdict is worthless. Split first, then build.

## 3 — compare on a held-out golden set

Run `eval_compare` with the baseline table **first**; every subsequent table
carries its `delta` against that baseline. The golden set is a registered
source of `(query_id, query_text, relevant_id[, relevance_grade])` rows — see
[Evaluate and Compare Models](./evaluation.md) for its schema.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession, baseline_table: String, treatment_table: String) -> jammi_db::error::Result<()> {
let comparison = session
    .eval_compare(
        &[baseline_table, treatment_table], // baseline FIRST
        "patents",
        "golden.public.golden_relevance",   // held-out golden set
        10,                                 // k for recall@k / precision@k
    )
    .await?;

for entry in comparison.per_table.iter().skip(1) {
    let delta = entry.delta.as_ref().expect("non-baseline entries carry a delta");
    println!(
        "{}: nDCG {:+.3} ({:+.1}%)",
        entry.table_name,
        delta.ndcg.absolute,
        delta.ndcg.relative * 100.0,
    );

    // The paired significance of each metric delta. `None` only when the two
    // runs share no `query_id` (nothing to pair).
    if let Some(sig) = delta.significance.as_ref() {
        let s = &sig.ndcg;
        println!(
            "  nDCG p={:.4}  95% CI [{:+.3}, {:+.3}]",
            s.p_value, s.ci_lower, s.ci_upper,
        );
    }
    # let _ = entry;
}
# Ok(()) }
```

```python
comparison = db.eval_compare(
    embedding_tables=[baseline_table, treatment_table],  # baseline FIRST
    source="patents",
    golden_source="golden.public.golden_relevance",
    k=10,
)
for entry in comparison["per_table"][1:]:
    delta = entry["delta"]
    d = delta["ndcg"]
    print(f"{entry['table_name']}: nDCG {d['absolute']:+.3f} ({d['relative']*100:+.1f}%)")
    sig = delta.get("significance")
    if sig is not None:
        s = sig["ndcg"]
        print(f"  nDCG p={s['p_value']:.4f}  95% CI [{s['ci_lower']:+.3f}, {s['ci_upper']:+.3f}]")
```

### Reading the significance

For each metric, `eval_compare` attaches a `MetricSignificance` carrying a
`p_value` and a `[ci_lower, ci_upper]` interval:

- **`ci_lower` / `ci_upper`** are a percentile **bootstrap confidence
  interval** on the mean paired difference (`treatment − baseline`), at the 95%
  level. A CI that lies **entirely above zero** is the resampling analogue of
  "the delta is real, not noise." A CI that **brackets zero** means you cannot
  distinguish the treatment from the baseline on this metric.
- **`p_value`** is the two-tailed **Mann–Whitney U** p-value comparing the
  baseline and treatment per-query distributions — distribution-free, and
  robust to the bounded, tie-heavy shape retrieval metrics have. Smaller is
  stronger evidence.

Both are **deterministic**: the bootstrap runs under a pinned seed and a fixed
iteration count, so the same inputs always yield the same interval. Two
identical runs collapse to a `[0, 0]` CI with `p ≈ 1`.

A delta of `+0.02` is a *headline*, not a conclusion. Report it only as
`+0.02, p=0.003, CI [+0.008, +0.031]` — the delta with its significance.

## Discipline contracts

These are the contracts a "did structure help?" claim must satisfy. State each
one explicitly when you report a result.

### Strict held-out split (the leakage contract)

The graph and any graph-supervised training use the **train split only**; the
golden set is held out and never feeds construction. This is the single most
important contract — restated here because it is the one that silently
invalidates a result. If you cannot point to the split boundary, you do not
have a clean number.

### Judgment-matched metric

Pick the metric from the **judgment type**, not by habit:

| Judgments | Metric | Why |
|-----------|--------|-----|
| **Graded** (`relevance_grade` > 1) | **nDCG** | Discounted cumulative gain uses the grade; recall would discard the ranking signal. |
| **Binary** (relevant / not) | **recall@k**, **MRR** | No grade to exploit; presence and first-hit rank are the right targets. |

`eval_compare` always computes all four metrics, but read the one that matches
your golden set. Using recall on graded judgments throws away signal you paid a
human to produce.

### ≥3 seeds for trained treatments

A trained treatment (a fine-tuned checkpoint, or any treatment whose
construction samples) varies by seed. **One lucky seed can fake a win.**
Run the treatment under **≥3 seeds**, compare each against the same baseline,
and report the **mean ± variance** of the delta plus the significance across
seeds — not a single run.

A deterministic treatment (e.g. a pure propagation with no sampling) does not
vary by seed, so it needs only the leakage and significance discipline — a
quiet advantage worth stating when it applies.

### Cohort slicing

The most *useful* output is **where** structure helped — by source family,
period, or any segment you care about. Tag each query with cohort labels at
eval time, then group the persisted per-query records.

Cohort tags are supplied per query to `eval_embeddings` (the per-table entry
point); `eval_compare` itself does not surface cohort tagging, so to slice a
comparison you run each table through `eval_embeddings` with the same cohort
map. Every per-query record — its metrics and its cohort tags — is persisted
to `_jammi_eval_per_query`, keyed by the run's `eval_run_id`, and read back
with `eval_per_query`.

```rust,no_run
# extern crate jammi_db;
# extern crate jammi_ai;
# extern crate tokio;
# use std::collections::{BTreeMap, HashMap};
# use jammi_ai::session::InferenceSession;
# async fn ex(session: &InferenceSession, treatment_table: &str) -> jammi_db::error::Result<()> {
// Tag each query with its cohort(s) at eval time.
let mut cohorts: HashMap<String, BTreeMap<String, String>> = HashMap::new();
cohorts.insert(
    "q1".into(),
    BTreeMap::from([("family".into(), "A".into())]),
);
// ... one entry per query_id ...

let report = session
    .eval_embeddings(
        "patents",
        Some(treatment_table),
        "golden.public.golden_relevance",
        10,
        &cohorts,
    )
    .await?;

// Read the persisted per-query rows back by run id; each row carries its
// metrics and cohort tags as JSON, ready to group by cohort.
let rows = session.eval_per_query(&report.eval_run_id).await?;
for row in &rows {
    println!("{}: cohorts={} metrics={}", row.query_id, row.cohorts_json, row.metrics_json);
}
# Ok(()) }
```

```python
cohorts = {"q1": {"family": "A"}, "q2": {"family": "B"}}  # one entry per query_id
report = db.eval_embeddings(
    source="patents",
    embedding_table=treatment_table,
    golden_source="golden.public.golden_relevance",
    k=10,
    cohorts=cohorts,
)
rows = db.eval_per_query(report["eval_run_id"])
# Group `rows` by their cohort tags and aggregate per-cohort metrics.
```

Then group by cohort and report **per-cohort `n` and a confidence interval** —
a 12-query cohort has a wide CI, and a swing inside it is not a finding. Small
cohorts go noisy; report `n` so a reader does not over-read them.

## What this recipe does *not* cover

- **New metrics.** Recall / precision / MRR / nDCG suffice; this recipe does
  not add others.
- **Online drift monitoring.** This is an *offline* held-out harness, not a
  production-drift monitor.
- **The golden set itself.** Constructing relevance judgments is the expensive
  human step — budget for it. The eval is cheap; the labels are not.

## See also

- [Evaluate and Compare Models](./evaluation.md) — the golden-set schema, the
  metric definitions, and the full `eval_compare` / `eval_embeddings` API.
- [Fine-Tune for Your Domain](./fine-tuning.md) — producing a trained treatment
  table to evaluate with this recipe.
