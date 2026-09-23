//! Task adapters: the columns each task's output carries, and how a
//! model's raw output becomes them.

pub mod classification;
pub mod distribution;
pub mod embedding;
pub mod ner;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::Field;

pub use classification::ClassificationAdapter;
pub use distribution::{DistributionAdapter, DistributionForm};
pub use embedding::EmbeddingAdapter;

use crate::error::{Error, Result};
use crate::output::BackendOutput;
use crate::runtime::BoundModel;
use crate::task::ModelTask;

/// Converts raw backend output into Arrow arrays for a specific task.
pub trait OutputAdapter: Send + Sync {
    /// Arrow schema for this task's output columns (excluding common prefix).
    fn output_schema(&self) -> Vec<Field>;

    /// Convert raw backend output into Arrow arrays for one batch, taking
    /// the output so a head's buffer becomes the column without a copy.
    fn adapt(&self, output: BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>>;
}

/// Create an adapter for a given task with model-derived dimensions.
pub fn create_adapter(task: ModelTask, model: &dyn BoundModel) -> Result<Box<dyn OutputAdapter>> {
    match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding | ModelTask::AudioEmbedding => {
            Ok(Box::new(EmbeddingAdapter::new(model.embedding_dim())))
        }
        ModelTask::Classification => Ok(Box::new(ClassificationAdapter)),
        ModelTask::Ner => Ok(Box::new(ner::NerAdapter)),
        // A regression model serves the form its head was trained for, read
        // from the head's persisted `DistributionForm`: `Gaussian` →
        // `(predicted_mean, predicted_std)`, `Quantile { levels }` → one
        // `quantile_{level}` column per level. Selecting on the persisted form
        // (never on head width — a 2-level quantile head is also width 2) is
        // what stops a quantile-trained head being silently mis-decoded as a
        // Gaussian `(mean, std)` on the public `Infer` read path. A regression
        // head saved without a form (none today) falls back to Gaussian, the
        // density-bearing core form.
        ModelTask::Regression => match model.regression_form() {
            Some(DistributionForm::Quantile { levels }) => {
                Ok(Box::new(DistributionAdapter::quantile(levels.clone())?))
            }
            // A z-space-trained Gaussian head learns a z-scale σ (σ_z ≈ 1); the
            // adapter scales it back to raw units by σ_y (the persisted scaler's
            // std) on the post-softplus column. A head with no scaler (none today
            // for a trained regression model) serves σ unscaled (`std_scale = 1`).
            Some(DistributionForm::Gaussian) | None => Ok(Box::new(
                DistributionAdapter::gaussian_scaled(model.regression_std_scale().unwrap_or(1.0)),
            )),
        },
    }
}

/// Create an adapter for schema construction only (no model handle needed).
///
/// For embedding, pass the model's hidden_size as `embedding_dim`. For
/// regression, pass the head's persisted `regression_form` so the planned
/// output schema matches what the runtime adapter (built from the loaded model
/// via [`create_adapter`]) emits — a quantile head's schema is its level
/// columns, not the Gaussian default. `None` form falls back to Gaussian, and
/// a malformed persisted quantile form (it was validated at fine-tune time)
/// likewise falls back rather than failing schema planning.
pub(crate) fn create_adapter_for_schema(
    task: ModelTask,
    embedding_dim: Option<usize>,
    regression_form: Option<&DistributionForm>,
) -> Box<dyn OutputAdapter> {
    match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding | ModelTask::AudioEmbedding => {
            Box::new(EmbeddingAdapter::new(embedding_dim.unwrap_or(0)))
        }
        ModelTask::Classification => Box::new(ClassificationAdapter),
        ModelTask::Ner => Box::new(ner::NerAdapter),
        ModelTask::Regression => match regression_form {
            Some(DistributionForm::Quantile { levels }) => {
                DistributionAdapter::quantile(levels.clone()).map_or_else(
                    |_| Box::new(DistributionAdapter::gaussian()) as Box<dyn OutputAdapter>,
                    |a| Box::new(a) as Box<dyn OutputAdapter>,
                )
            }
            Some(DistributionForm::Gaussian) | None => Box::new(DistributionAdapter::gaussian()),
        },
    }
}

// ─── Shared null-handling helpers ────────────────────────────────────────────
//
// Both helpers share [`build_prefix_columns`](super::schema::build_prefix_columns)'s
// refuse-by-name policy: a `values`/`row_status` length that disagrees with the
// caller's `row_count` is a producer bug, never a per-row default. Defaulting a
// missing `row_status[i]` to `false` would mark an out-of-range row "errored"
// instead of surfacing the length mismatch, and letting the emitted column's
// length be governed by `values.len()` alone — with nothing pinning it to
// `row_count` — would return a column shorter than the rest of the batch
// instead of a refusal. `ClassificationAdapter`/`NerAdapter` pass their own
// `row_count` so this check runs regardless of adapter call order.

/// Build a nullable StringArray: rows where `row_status[i]` is false become
/// null. Refuses by name when `values` (if present) or `row_status` disagrees
/// in length with `row_count` — see the module-level note above.
pub(crate) fn nullify_strings(
    values: Option<&Vec<String>>,
    row_status: &[bool],
    row_count: usize,
) -> Result<StringArray> {
    match values {
        Some(v) => {
            if v.len() != row_count {
                return Err(Error::Inference(format!(
                    "nullify_strings: values has {} entries, expected one per row ({row_count})",
                    v.len()
                )));
            }
            if row_status.len() != row_count {
                return Err(Error::Inference(format!(
                    "nullify_strings: row_status has {} entries, expected one per row \
                     ({row_count})",
                    row_status.len()
                )));
            }
            Ok(v.iter()
                .enumerate()
                .map(|(i, s)| {
                    if row_status[i] {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                .collect())
        }
        None => {
            if row_status.len() != row_count {
                return Err(Error::Inference(format!(
                    "nullify_strings: row_status has {} entries, expected one per row \
                     ({row_count})",
                    row_status.len()
                )));
            }
            Ok(vec![None::<&str>; row_count].into_iter().collect())
        }
    }
}

/// Build a nullable Float32Array: rows where `row_status[i]` is false become
/// null. Refuses by name when `values` (if present) or `row_status` disagrees
/// in length with `row_count` — see the module-level note above.
pub(crate) fn nullify_floats(
    values: Option<&Vec<f32>>,
    row_status: &[bool],
    row_count: usize,
) -> Result<Float32Array> {
    match values {
        Some(v) => {
            if v.len() != row_count {
                return Err(Error::Inference(format!(
                    "nullify_floats: values has {} entries, expected one per row ({row_count})",
                    v.len()
                )));
            }
            if row_status.len() != row_count {
                return Err(Error::Inference(format!(
                    "nullify_floats: row_status has {} entries, expected one per row \
                     ({row_count})",
                    row_status.len()
                )));
            }
            Ok(v.iter()
                .enumerate()
                .map(|(i, &c)| if row_status[i] { Some(c) } else { None })
                .collect())
        }
        None => {
            if row_status.len() != row_count {
                return Err(Error::Inference(format!(
                    "nullify_floats: row_status has {} entries, expected one per row \
                     ({row_count})",
                    row_status.len()
                )));
            }
            Ok(vec![None::<f32>; row_count].into_iter().collect())
        }
    }
}

// ─── `BackendOutput` accessor oracles ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use arrow::array::Array;

    use super::*;

    /// A well-formed 3-row, dim-2 head where row 1 failed. Both accessors
    /// exercise their happy/refusal paths off this one fixture.
    fn three_rows_row1_failed() -> BackendOutput {
        BackendOutput {
            float_outputs: vec![vec![1.0, 2.0, 0.0, 0.0, 5.0, 6.0]],
            string_outputs: vec![],
            row_status: vec![true, false, true],
            row_errors: vec![
                String::new(),
                "row 1 decode failed".to_string(),
                String::new(),
            ],
            shapes: vec![(3, 2)],
        }
    }

    // -- `single_row_or_err` ---------------------------------------------

    #[test]
    fn single_row_or_err_returns_the_row_when_it_succeeded() {
        let out = three_rows_row1_failed();
        assert_eq!(out.single_row_or_err(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(out.single_row_or_err(2).unwrap(), &[5.0, 6.0]);
    }

    /// A caller reading `output.float_outputs[0][..dim]` directly turns a
    /// failed row 0 into the all-zero placeholder instead of an `Err`
    /// (`[0.0, 0.0]` is a valid slice). The checked accessor refuses it:
    /// replacing `single_row_or_err`'s body with
    /// `&self.float_outputs[0][row * dim..row * dim + dim]` fails this test
    /// (`Ok([0.0, 0.0])` vs the expected `Err`).
    #[test]
    fn single_row_or_err_refuses_a_failed_row_naming_it() {
        let out = three_rows_row1_failed();
        let err = out
            .single_row_or_err(1)
            .expect_err("a failed row must never resolve to its all-zero placeholder");
        assert!(
            err.to_string().contains("row 1 decode failed"),
            "must surface the row's own recorded message, got: {err}"
        );
    }

    #[test]
    fn single_row_or_err_refuses_a_failed_row_with_no_recorded_message() {
        let out = BackendOutput {
            float_outputs: vec![vec![0.0, 0.0]],
            string_outputs: vec![],
            row_status: vec![false],
            row_errors: vec![String::new()],
            shapes: vec![(1, 2)],
        };
        let err = out.single_row_or_err(0).unwrap_err();
        assert!(
            err.to_string().contains("Row 0"),
            "must name the row even with no recorded per-row message, got: {err}"
        );
    }

    #[test]
    fn single_row_or_err_refuses_a_row_index_out_of_range() {
        let out = three_rows_row1_failed();
        let err = out.single_row_or_err(3).unwrap_err();
        assert!(err.to_string().contains("3"));
    }

    // -- `all_rows_or_err` -------------------------------------------------

    #[test]
    fn all_rows_or_err_returns_the_flat_buffer_when_every_row_succeeded() {
        let out = BackendOutput {
            float_outputs: vec![vec![1.0, 2.0, 3.0, 4.0]],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 2)],
        };
        assert_eq!(out.all_rows_or_err().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    /// Pre-fix, `trainer.rs::project_frozen_embedding` read
    /// `&output.float_outputs[0]` directly — a corrupt training item's
    /// all-zero placeholder row was silently trained on instead of refusing
    /// the group. Verified by temporarily reverting `all_rows_or_err`'s body
    /// to the blind `&self.float_outputs[0]` read, which turns this RED (it
    /// returns `Ok` instead of `Err`).
    #[test]
    fn all_rows_or_err_with_two_bad_rows_names_the_lowest_index() {
        let out = BackendOutput {
            float_outputs: vec![vec![0.0; 8]],
            string_outputs: vec![],
            row_status: vec![true, false, true, false],
            row_errors: vec![
                String::new(),
                "row 1 failed".to_string(),
                String::new(),
                "row 3 failed".to_string(),
            ],
            shapes: vec![(4, 2)],
        };
        let err = out.all_rows_or_err().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("row 1 failed"), "got: {msg}");
        assert!(!msg.contains("row 3 failed"), "got: {msg}");
    }

    #[test]
    fn all_rows_or_err_refuses_a_zero_row_output() {
        let out = BackendOutput {
            float_outputs: vec![vec![]],
            string_outputs: vec![],
            row_status: vec![],
            row_errors: vec![],
            shapes: vec![(0, 4)],
        };
        assert!(out.all_rows_or_err().is_err());
    }

    // -- Shared consistency check (`checked_rows`), both accessors ---------
    //
    // A `.position(|ok| !ok)` scan alone fails OPEN on an empty/short
    // `row_status` (it finds nothing wrong and returns a truncated/garbage
    // slice as if every row succeeded). Both accessors refuse, by name, on
    // every one of these malformed states, via the one shared `checked_rows`
    // gate.

    #[test]
    fn both_accessors_refuse_when_shapes_is_empty() {
        let out = BackendOutput {
            float_outputs: vec![vec![1.0, 2.0]],
            string_outputs: vec![],
            row_status: vec![true],
            row_errors: vec![String::new()],
            shapes: vec![],
        };
        assert!(out.single_row_or_err(0).is_err());
        assert!(out.all_rows_or_err().is_err());
    }

    #[test]
    fn both_accessors_refuse_when_row_status_is_shorter_than_shapes_rows() {
        // Pre-fix: `all_rows_or_err`'s failed-row scan over an EMPTY
        // `row_status` finds no bad row and falls through to return
        // `float_outputs[0]` whole, as if 3 rows had all succeeded.
        let out = BackendOutput {
            float_outputs: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            string_outputs: vec![],
            row_status: vec![],
            row_errors: vec![],
            shapes: vec![(3, 2)],
        };
        let single_err = out
            .single_row_or_err(0)
            .expect_err("a short row_status must be a typed refusal, not an out-of-bounds read");
        assert!(single_err.to_string().contains("row_status"));
        let all_err = out
            .all_rows_or_err()
            .expect_err("a short row_status must never fail OPEN into returning every row");
        assert!(all_err.to_string().contains("row_status"));
    }

    #[test]
    fn both_accessors_refuse_when_row_status_is_longer_than_shapes_rows() {
        let out = BackendOutput {
            float_outputs: vec![vec![1.0, 2.0]],
            string_outputs: vec![],
            row_status: vec![true, true, true],
            row_errors: vec![String::new(), String::new(), String::new()],
            shapes: vec![(1, 2)],
        };
        assert!(out.single_row_or_err(0).is_err());
        assert!(out.all_rows_or_err().is_err());
    }

    #[test]
    fn both_accessors_refuse_when_the_flat_buffer_disagrees_with_rows_times_dim() {
        let out = BackendOutput {
            // shapes says 2 rows of dim 2 (4 values); the flat buffer is short.
            float_outputs: vec![vec![1.0, 2.0, 3.0]],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 2)],
        };
        let single_err = out.single_row_or_err(0).unwrap_err();
        assert!(
            single_err.to_string().contains("rows*dim") || single_err.to_string().contains("value")
        );
        assert!(out.all_rows_or_err().is_err());
    }

    #[test]
    fn both_accessors_refuse_when_row_errors_length_mismatches_shapes_rows() {
        let out = BackendOutput {
            float_outputs: vec![vec![1.0, 2.0]],
            string_outputs: vec![],
            row_status: vec![true],
            row_errors: vec![], // one short of rows(1)
            shapes: vec![(1, 2)],
        };
        let single_err = out.single_row_or_err(0).unwrap_err();
        assert!(
            single_err.to_string().contains("row_errors"),
            "{single_err}"
        );
        let all_err = out.all_rows_or_err().unwrap_err();
        assert!(all_err.to_string().contains("row_errors"), "{all_err}");
    }

    /// A producer with NO float head at all (`float_outputs` empty) but a
    /// shape entry claiming 2 rows. A flat-length check that reads
    /// `float_outputs.first()` treats the absent head as an empty (len-0)
    /// buffer, trivially satisfies `rows*dim == 0` at `dim == 0`, and lets
    /// both accessors panic indexing `float_outputs[0]`. Dropping the
    /// `float_outputs.len() != 1` check from `checked_rows` fails this test
    /// (a panic, not the `Err` asserted below).
    #[test]
    fn both_accessors_refuse_a_producer_with_no_float_head() {
        let out = BackendOutput {
            float_outputs: vec![],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 0)],
        };
        assert!(out.single_row_or_err(0).is_err());
        assert!(out.all_rows_or_err().is_err());
    }

    /// Isolates the `float_outputs.len() != 1` check from the `dim == 0`
    /// check above: a NONZERO-dim shape with no float head at all would
    /// still slip past a `dim == 0` guard alone and panic at
    /// `self.float_outputs[0]`. Dropping the `float_outputs.len() != 1` check
    /// from `checked_rows` (with the `dim == 0` check left in place) fails
    /// this test (a panic, not the `Err` asserted below), while the
    /// `dim == 0` fixture above would still pass off the OTHER check alone.
    #[test]
    fn both_accessors_refuse_a_producer_with_no_float_head_and_a_nonzero_dim() {
        let out = BackendOutput {
            float_outputs: vec![],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 3)],
        };
        assert!(out.single_row_or_err(0).is_err());
        assert!(out.all_rows_or_err().is_err());
    }

    /// A zero-dim head (`shapes[0].1 == 0`) with a present-but-empty
    /// `float_outputs[0]` must not return `Ok(&[])` — a vacuous "embedding"
    /// with no dimensions. Dropping the `dim == 0` check from `checked_rows`
    /// fails this test (`Ok([])` instead of the `Err` asserted below).
    #[test]
    fn both_accessors_refuse_a_zero_dim_head() {
        let out = BackendOutput {
            float_outputs: vec![vec![]],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 0)],
        };
        assert!(out.single_row_or_err(0).is_err());
        assert!(out.all_rows_or_err().is_err());
    }

    #[test]
    fn checked_rows_refuses_an_overflowing_rows_times_dim_without_panicking() {
        let out = BackendOutput {
            float_outputs: vec![vec![]],
            string_outputs: vec![],
            row_status: vec![],
            row_errors: vec![],
            shapes: vec![(usize::MAX, 2)],
        };
        let err = out.single_row_or_err(0).unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
        let err = out.all_rows_or_err().unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    // -- `single_head` constructor ------------------------------------------

    #[test]
    fn single_head_builds_a_consistent_output() {
        let out = BackendOutput::single_head(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
            vec![true, true],
            vec![String::new(), String::new()],
        )
        .unwrap();
        assert_eq!(out.shapes[0], (2, 2));
        assert_eq!(out.single_row_or_err(1).unwrap(), &[3.0, 4.0]);
    }

    /// `single_head(vec![], 2, 0, ..)` would otherwise succeed — `flat.len()
    /// == 0` trivially matches `rows(2) * dim(0) == 0`. `single_head` refuses
    /// a zero-dim head outright.
    #[test]
    fn single_head_refuses_a_zero_dim_head() {
        let err = BackendOutput::single_head(
            vec![],
            2,
            0,
            vec![true, true],
            vec![String::new(), String::new()],
        )
        .unwrap_err();
        assert!(err.to_string().contains("dim"), "{err}");
    }

    #[test]
    fn single_head_refuses_zero_rows() {
        let err = BackendOutput::single_head(vec![], 0, 4, vec![], vec![]).unwrap_err();
        assert!(err.to_string().contains("rows"), "{err}");
    }

    #[test]
    fn single_head_refuses_an_overflowing_rows_times_dim_without_panicking() {
        let err = BackendOutput::single_head(vec![], usize::MAX, 2, vec![], vec![]).unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    #[test]
    fn single_head_refuses_a_flat_buffer_that_disagrees_with_rows_times_dim() {
        let err = BackendOutput::single_head(
            vec![1.0, 2.0, 3.0], // one short of rows(2) * dim(2)
            2,
            2,
            vec![true, true],
            vec![String::new(), String::new()],
        )
        .unwrap_err();
        assert!(err.to_string().contains("rows*dim"));
    }

    #[test]
    fn single_head_refuses_a_row_status_length_mismatch() {
        let err = BackendOutput::single_head(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
            vec![true], // one short
            vec![String::new(), String::new()],
        )
        .unwrap_err();
        assert!(err.to_string().contains("row_status"));
    }

    #[test]
    fn single_head_refuses_a_row_errors_length_mismatch() {
        let err = BackendOutput::single_head(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            2,
            vec![true, true],
            vec![String::new()], // one short
        )
        .unwrap_err();
        assert!(err.to_string().contains("row_errors"));
    }

    // -- `nullify_strings` / `nullify_floats` -------------------------------
    //
    // `ClassificationAdapter`/`NerAdapter` are the two readers of these
    // helpers; both pass their own `row_count`, never the length of
    // `row_status`. Defaulting (`unwrap_or(false)`) every out-of-range row to
    // "errored" instead of refusing, and leaving a short `values` unchecked
    // against `row_count` (the emitted column's length coming straight from
    // `values.len()` instead) are exactly the two smells these tests pin
    // shut.

    #[test]
    fn nullify_strings_refuses_a_row_status_shorter_than_row_count() {
        // Verified by reverting the `row_status.len() != row_count` guard
        // (restoring `row_status.get(i).copied().unwrap_or(false)`): this
        // test goes RED — a 3-entry `values` with a 1-entry `row_status`
        // silently returns `Ok` with rows 1 and 2 defaulted to null instead
        // of the `Err` asserted below.
        let values = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let row_status = vec![true]; // one entry; row_count is 3
        let err = nullify_strings(Some(&values), &row_status, 3)
            .expect_err("a row_status shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("row_status"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    #[test]
    fn nullify_strings_refuses_values_shorter_than_row_count() {
        // Verified by removing the `v.len() != row_count` guard: this test
        // goes RED — `Ok` with a 1-entry column sitting next to the rest of
        // the batch's 3-entry columns, instead of the `Err` asserted below.
        let values = vec!["a".to_string()]; // one entry; row_count is 3
        let row_status = vec![true, true, true];
        let err = nullify_strings(Some(&values), &row_status, 3)
            .expect_err("values shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("values"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    #[test]
    fn nullify_strings_nulls_out_failed_rows_on_the_happy_path() {
        let values = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let row_status = vec![true, false, true];
        let array = nullify_strings(Some(&values), &row_status, 3)
            .expect("mutually consistent lengths must not be refused");
        assert_eq!(array.len(), 3);
        assert_eq!(array.value(0), "a");
        assert!(array.is_null(1));
        assert_eq!(array.value(2), "c");
    }

    #[test]
    fn nullify_strings_with_no_values_returns_an_all_null_column_of_row_count() {
        let row_status = vec![true, false];
        let array = nullify_strings(None, &row_status, 2).unwrap();
        assert_eq!(array.len(), 2);
        assert!(array.is_null(0));
        assert!(array.is_null(1));
    }

    #[test]
    fn nullify_strings_with_no_values_still_refuses_a_row_status_shorter_than_row_count() {
        // Verified by deleting the `row_status.len() != row_count` guard in
        // the `None` arm: this test goes RED — a 1-entry `row_status` against
        // `row_count = 3` silently returns `Ok` with an all-null 3-entry
        // column instead of the `Err` asserted below.
        let row_status = vec![true]; // one entry; row_count is 3
        let err = nullify_strings(None, &row_status, 3).expect_err(
            "a row_status shorter than row_count must be a typed refusal even with no values",
        );
        let msg = err.to_string();
        assert!(msg.contains("row_status"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    #[test]
    fn nullify_floats_refuses_a_row_status_shorter_than_row_count() {
        let values = vec![1.0_f32, 2.0, 3.0];
        let row_status = vec![true]; // one entry; row_count is 3
        let err = nullify_floats(Some(&values), &row_status, 3)
            .expect_err("a row_status shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("row_status"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    #[test]
    fn nullify_floats_refuses_values_shorter_than_row_count() {
        let values = vec![1.0_f32]; // one entry; row_count is 3
        let row_status = vec![true, true, true];
        let err = nullify_floats(Some(&values), &row_status, 3)
            .expect_err("values shorter than row_count must be a typed refusal");
        let msg = err.to_string();
        assert!(msg.contains("values"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }

    #[test]
    fn nullify_floats_nulls_out_failed_rows_on_the_happy_path() {
        let values = vec![1.0_f32, 2.0, 3.0];
        let row_status = vec![true, false, true];
        let array = nullify_floats(Some(&values), &row_status, 3)
            .expect("mutually consistent lengths must not be refused");
        assert_eq!(array.len(), 3);
        assert_eq!(array.value(0), 1.0);
        assert!(array.is_null(1));
        assert_eq!(array.value(2), 3.0);
    }

    #[test]
    fn nullify_floats_with_no_values_returns_an_all_null_column_of_row_count() {
        let row_status = vec![true, false];
        let array = nullify_floats(None, &row_status, 2).unwrap();
        assert_eq!(array.len(), 2);
        assert!(array.is_null(0));
        assert!(array.is_null(1));
    }

    #[test]
    fn nullify_floats_with_no_values_still_refuses_a_row_status_shorter_than_row_count() {
        // Verified by deleting the `row_status.len() != row_count` guard in
        // the `None` arm: this test goes RED — a 1-entry `row_status` against
        // `row_count = 3` silently returns `Ok` with an all-null 3-entry
        // column instead of the `Err` asserted below.
        let row_status = vec![true]; // one entry; row_count is 3
        let err = nullify_floats(None, &row_status, 3).expect_err(
            "a row_status shorter than row_count must be a typed refusal even with no values",
        );
        let msg = err.to_string();
        assert!(msg.contains("row_status"), "must name the field: {msg}");
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must name both the got (1) and expected (3) lengths: {msg}"
        );
    }
}
