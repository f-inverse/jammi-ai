use std::cmp::Ordering;
use std::collections::BinaryHeap;

use arrow::array::{Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;

use jammi_numerics::distance::cosine_distance;

use crate::error::{JammiError, Result};
use crate::index::{distance_is_admissible, ValidatedQuery};
use crate::store::vectors::extend_with_fixed_size_list_f32;

/// Total order over scored candidates: ascending cosine distance, ties broken
/// by ascending `_row_id`.
///
/// `_row_id` is the table's primary key, so it is unique across the corpus;
/// the pair `(dist, _row_id)` is therefore a *total* order with no genuine
/// equalities. That totality is what makes the bounded top-k below return
/// exactly the same `k` elements — in exactly the same order — as a
/// collect-everything-then-sort pass: ties on `dist` are always resolved by
/// the unique row id, never by scan or batch arrival order.
///
/// `partial_cmp` on the distances falls back to [`Ordering::Equal`] for the
/// `NaN` case, and that fallback is LOAD-BEARING, not belt-and-braces:
/// `cosine_distance` guards zero MAGNITUDE (short-circuiting to `1.0`) but not
/// a non-finite COMPONENT — a `NaN`/`inf` inside a stored vector makes `denom`
/// `NaN`, `NaN < EPSILON` is false, and the distance is `NaN` (measured). The
/// earlier claim here that `cosine_distance` never produces `NaN` was false.
/// This is the one path with no index behind it, so the index-side
/// admissibility check ([`crate::index::distance_is_admissible`]) cannot reach
/// it; the comparator stays total for every `f32` instead.
fn candidate_order(a: &(String, f32), b: &(String, f32)) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

/// A scored candidate wrapped so that [`BinaryHeap`]'s max-ordering surfaces the
/// *worst* (largest under [`candidate_order`]) retained candidate at the top.
///
/// The bounded heap keeps the `k` best candidates: a new candidate is admitted
/// only when it orders before the current worst, which is then evicted. Both
/// the eviction comparison and the final drain-sort go through
/// [`candidate_order`], so the heap's notion of "worst" and the result's final
/// ordering are one definition.
struct Candidate((String, f32));

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        candidate_order(&self.0, &other.0) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        candidate_order(&self.0, &other.0)
    }
}

/// A bounded top-`k` collector over the [`candidate_order`] total order.
///
/// Retains at most `k` candidates, each a `(row_id, distance)` pair and nothing
/// more — the heap never holds a vector, so its footprint is `O(k)` independent
/// of the corpus size. `offer` admits a candidate when fewer than `k` are held,
/// or when it orders strictly before the current worst (which it then evicts).
struct BoundedTopK {
    k: usize,
    heap: BinaryHeap<Candidate>,
}

impl BoundedTopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    fn offer(&mut self, row_id: String, dist: f32) {
        if self.k == 0 {
            return;
        }
        let candidate = Candidate((row_id, dist));
        if self.heap.len() < self.k {
            self.heap.push(candidate);
        } else if let Some(worst) = self.heap.peek() {
            // `peek` is the largest retained candidate under `candidate_order`.
            // Admit the newcomer only if it orders strictly before that worst,
            // evicting the worst to keep the set at `k`.
            if candidate.cmp(worst) == Ordering::Less {
                self.heap.pop();
                self.heap.push(candidate);
            }
        }
    }

    /// Drain into a `Vec` sorted ascending under [`candidate_order`] — the same
    /// prefix a collect-everything-then-sort-then-`truncate(k)` pass produces.
    fn into_sorted(self) -> Vec<(String, f32)> {
        let mut out: Vec<(String, f32)> = self.heap.into_iter().map(|c| c.0).collect();
        out.sort_by(candidate_order);
        out
    }
}

/// Brute-force vector search over a registered Parquet table via DataFusion.
///
/// Computes cosine distance for every row, returns the `k` closest as
/// `(row_id, cosine_distance)` sorted by ascending distance, with ties broken
/// by ascending `_row_id` so equidistant candidates resolve deterministically
/// regardless of scan order.
///
/// The scan streams one [`arrow::array::RecordBatch`] at a time and folds each
/// into a bounded top-`k` heap, so only the current batch's vectors and at most
/// `k` `(row_id, distance)` pairs are resident at once. Peak memory is therefore
/// `O(k + batch_rows · d)`, independent of the corpus size `N`, rather than the
/// `O(N · d)` of materialising every vector before scoring.
/// `catalog_dimensions` is the catalog row's recorded width, when the caller
/// has one: it is a CROSS-CHECK against the scan's own `FixedSizeList`
/// width (the authority here — this is the one path with no index behind
/// it), and a disagreement is itself a typed, table-named error. The query's
/// width is enforced against the scan width before any distance is computed,
/// so the kernel's own width assert is unreachable from here.
pub async fn exact_vector_search(
    ctx: &SessionContext,
    table_name: &str,
    query: &ValidatedQuery,
    k: usize,
    catalog_dimensions: Option<usize>,
) -> Result<Vec<(String, f32)>> {
    let df = ctx
        .sql(&format!(
            "SELECT _row_id, vector FROM \"jammi.{table_name}\""
        ))
        .await?;
    // The scan schema's `FixedSizeList` width is the width every row below
    // has, by construction of the column type. A non-positive or
    // unconvertible length, a wrong Arrow type, or a missing column is a
    // CORRUPT scan schema — this table's own stored artifact, never anything
    // the caller supplied — so every arm below is `IncompatibleFormat`
    // (engine class), identically to the catalog-vs-scan disagreement four
    // lines below and every corrupt-artifact refusal in `sidecar.rs`. A
    // silent `0` would additionally refuse every non-empty query with a
    // confident wrong expectation ("expected 0 dimensions") instead of
    // naming the corrupt column, and this is the one path with no index
    // behind it, so nothing else catches it.
    let scan_width = match df.schema().field_with_unqualified_name("vector") {
        Ok(field) => match field.data_type() {
            DataType::FixedSizeList(_, n) => match usize::try_from(*n) {
                Ok(width) if width > 0 => width,
                _ => {
                    return Err(JammiError::IncompatibleFormat {
                        artifact: format!("{table_name}.vector"),
                        found: format!("{n}"),
                        supported: "a positive FixedSizeList width".into(),
                    })
                }
            },
            other => {
                return Err(JammiError::IncompatibleFormat {
                    artifact: format!("{table_name}.vector"),
                    found: format!("{other:?}"),
                    supported: "FixedSizeList<Float32>".into(),
                })
            }
        },
        Err(_) => {
            return Err(JammiError::IncompatibleFormat {
                artifact: format!("{table_name}.vector"),
                found: "missing".into(),
                supported: "FixedSizeList<Float32>".into(),
            })
        }
    };
    match catalog_dimensions {
        Some(catalog) => {
            if catalog != scan_width {
                return Err(JammiError::IncompatibleFormat {
                    artifact: format!("{table_name}.vector"),
                    found: format!("scan width {scan_width}"),
                    supported: format!("catalog width {catalog}"),
                });
            }
            // The catalog record is the authority a Caller-provenance query
            // was already checked against at construction (`QueryBuilder::
            // new`'s `expected_width`), before this call was ever reached —
            // downstream of that entry, so a disagreement against the scan's
            // own width (just proven equal to the catalog's) is this table's
            // data drifting from its schema, never the caller's.
            query.require_width(scan_width, table_name.to_string())?;
        }
        None => {
            // No catalog width is on record for this table, so nothing has
            // checked this query before it reached here: the scan's own
            // width is the ONLY authority this call has, exactly like the
            // placement entry's `require_authority_width` when the catalog
            // has no recorded width. A genuine caller mistake here is still
            // the caller's fault.
            query.require_authority_width(scan_width)?;
        }
    }
    // `execute_stream` yields a single merged stream over all partitions. The
    // `(dist, _row_id)` total order makes the partition layout irrelevant — the
    // retained set is identical regardless of how the scan is partitioned — so
    // the merged stream is preferred over the per-partition variant for its
    // simpler single-loop drain.
    let mut stream = df.execute_stream().await?;

    let mut top_k = BoundedTopK::new(k);
    // Reused across batches: cleared and refilled per batch so only one batch's
    // vectors are ever resident, never the whole corpus.
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    while let Some(batch) = stream.try_next().await? {
        // `_row_id` is a Utf8 column, but the parquet reader surfaces it as
        // `Utf8View` (`StringViewArray`) under DataFusion's default
        // `schema_force_view_types`, and could be `LargeUtf8` for a wide table.
        // Cast to `Utf8` so a single `StringArray` downcast covers every Utf8
        // family the scan can produce.
        let row_ids_col = batch
            .column_by_name("_row_id")
            .ok_or_else(|| JammiError::Other("Missing _row_id in exact search".into()))?;
        let row_ids_utf8 = cast(row_ids_col, &DataType::Utf8).map_err(|e| {
            JammiError::Other(format!("_row_id column could not be cast to Utf8: {e}"))
        })?;
        let row_ids = row_ids_utf8
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                JammiError::Other("_row_id column is not a Utf8-castable string type".into())
            })?;

        vectors.clear();
        extend_with_fixed_size_list_f32(&batch, table_name, "vector", &mut vectors)?;
        // `extend_with_fixed_size_list_f32` appends exactly one Vec<f32> per
        // row, so the batch's vectors map 1:1 with `row_ids`.
        for (offset, vec) in vectors.iter().enumerate() {
            let dist = cosine_distance(query, vec);
            // The SINK: a non-finite distance here means a corrupt stored
            // row (the query is finite by type). Typed and table-named —
            // never a silently dropped row, which would read as "fewer
            // matches exist", and never a top-k slot.
            if !distance_is_admissible(dist) {
                return Err(JammiError::IncompatibleFormat {
                    artifact: format!("{table_name}.vector"),
                    found: format!(
                        "row '{}' yields a non-finite distance ({dist:?})",
                        row_ids.value(offset)
                    ),
                    supported: "finite f32 components".into(),
                });
            }
            top_k.offer(row_ids.value(offset).to_string(), dist);
        }
    }

    Ok(top_k.into_sorted())
}
