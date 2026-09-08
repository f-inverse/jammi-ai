pub mod classification;
pub mod distribution;
pub mod embedding;
pub mod ner;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::Field;
use jammi_db::error::{JammiError, Result};

pub use classification::ClassificationAdapter;
pub use distribution::{DistributionAdapter, DistributionForm};
pub use embedding::EmbeddingAdapter;

use crate::model::{LoadedModel, ModelTask};

/// Raw output from a model backend, before task-specific adaptation.
///
/// # Output head 0's row-major invariant
///
/// [`Self::single_row_or_err`] and [`Self::all_rows_or_err`] read
/// `float_outputs[0]` as one flattened ROW-MAJOR `[rows, dim]` matrix, where
/// `rows` and `dim` are `shapes[0]` — never as one `Vec` per row. Every
/// producer of a single float head must uphold, and both accessors verify
/// before reading a single value (via the shared `checked_rows`):
///
/// - `shapes` has at least one entry (`shapes[0] = (rows, dim)`).
/// - `row_status.len() == rows` and `row_errors.len() == rows`.
/// - `float_outputs[0].len() == rows * dim`.
///
/// A producer that violates this (e.g. one `Vec<f32>` per row, as
/// [`crate::model::backend::http::HttpBackend`] built before this fold) is a
/// bug in the producer, not something a reader can safely reinterpret — both
/// accessors refuse by name rather than misread the buffer. Use
/// [`Self::single_head`] to construct a single-float-head `BackendOutput`
/// with this invariant checked at construction time.
#[derive(Debug)]
pub struct BackendOutput {
    /// Numeric output tensors flattened to 1-D (one vec per output head).
    pub float_outputs: Vec<Vec<f32>>,
    /// String output tensors (one vec per output head).
    pub string_outputs: Vec<Vec<String>>,
    /// Per-row success flag (`true` = inference succeeded).
    pub row_status: Vec<bool>,
    /// Per-row error message (empty string when status is `true`).
    pub row_errors: Vec<String>,
    /// Shape metadata for each float output as `(rows, cols)`.
    pub shapes: Vec<(usize, usize)>,
}

impl BackendOutput {
    /// Build a single-float-head `BackendOutput` from an already row-major
    /// flattened buffer, validating the row-major invariant documented on
    /// [`Self`] at construction time rather than leaving a malformed producer
    /// to be silently misread by [`Self::single_row_or_err`] /
    /// [`Self::all_rows_or_err`] later. `flat.len()` must equal `rows * dim`;
    /// `row_status`/`row_errors` must each carry exactly one entry per row.
    pub fn single_head(
        flat: Vec<f32>,
        rows: usize,
        dim: usize,
        row_status: Vec<bool>,
        row_errors: Vec<String>,
    ) -> Result<Self> {
        if flat.len() != rows * dim {
            return Err(JammiError::Inference(format!(
                "BackendOutput::single_head: flat buffer has {} value(s), expected rows*dim \
                 ({rows}*{dim})",
                flat.len()
            )));
        }
        if row_status.len() != rows {
            return Err(JammiError::Inference(format!(
                "BackendOutput::single_head: row_status has {} entries, expected one per row \
                 ({rows})",
                row_status.len()
            )));
        }
        if row_errors.len() != rows {
            return Err(JammiError::Inference(format!(
                "BackendOutput::single_head: row_errors has {} entries, expected one per row \
                 ({rows})",
                row_errors.len()
            )));
        }
        Ok(Self {
            float_outputs: vec![flat],
            string_outputs: Vec::new(),
            row_status,
            row_errors,
            shapes: vec![(rows, dim)],
        })
    }

    /// Build the typed refusal for a failed row, using the row's recorded
    /// message when present or a generic fallback otherwise. Shared by
    /// [`Self::single_row_or_err`] and [`Self::all_rows_or_err`] so both
    /// checked accessors report a failed row identically.
    fn row_error(&self, row: usize) -> JammiError {
        match self.row_errors.get(row).filter(|m| !m.is_empty()) {
            Some(msg) => JammiError::Inference(msg.clone()),
            None => JammiError::Inference(format!("Row {row} inference failed")),
        }
    }

    /// Derive output head 0's authoritative row count from `shapes[0].0` and
    /// verify `row_status` and the flattened `float_outputs[0]` buffer agree
    /// with it — the ONE consistency check both [`Self::single_row_or_err`]
    /// and [`Self::all_rows_or_err`] run before looking at any individual
    /// row's status, so an inconsistent producer is refused BY NAME on both
    /// paths rather than read.
    ///
    /// Deriving the row count from `shapes[0].0` (never from
    /// `row_status.len()`) is load-bearing: `all_rows_or_err`'s failed-row
    /// scan is a `.position(|ok| !ok)` over `row_status`, which finds nothing
    /// wrong on an EMPTY or SHORT `row_status` and would otherwise return a
    /// truncated (or, on a too-long `float_outputs[0]`, over-long) slice as
    /// if every row had succeeded — failing open exactly where
    /// `single_row_or_err`'s `row_status.get(row)` already failed closed.
    /// Reshaping both accessors onto this one check makes the two agree on
    /// every malformed state instead of diverging on which they happen to
    /// notice.
    fn checked_rows(&self) -> Result<usize> {
        let (rows, dim) = *self.shapes.first().ok_or_else(|| {
            JammiError::Inference("BackendOutput has no output-head shape".into())
        })?;
        if self.row_status.len() != rows {
            return Err(JammiError::Inference(format!(
                "BackendOutput row_status has {} entries, expected one per row ({rows})",
                self.row_status.len()
            )));
        }
        let flat_len = self.float_outputs.first().map(Vec::len).unwrap_or(0);
        if flat_len != rows * dim {
            return Err(JammiError::Inference(format!(
                "BackendOutput float_outputs[0] has {flat_len} value(s), expected rows*dim \
                 ({rows}*{dim})"
            )));
        }
        Ok(rows)
    }

    /// Returns output head 0's slice for `row`, or `row`'s typed error when
    /// `row_status[row]` is `false`.
    ///
    /// A per-row backend (`forward_image_embedding`, `forward_audio_embedding`)
    /// writes an all-zero placeholder into `float_outputs[0]` for a row whose
    /// decode/preprocess failed, so the rest of a multi-row batch can still
    /// embed. A caller that serves exactly one row per call (a single-item
    /// query) has no later row to recover with, so it MUST go through this
    /// accessor rather than reading `float_outputs[0]` directly — otherwise a
    /// corrupt input's refusal silently becomes a zero vector instead of an
    /// `Err`.
    ///
    /// Refuses (rather than reading) whenever `checked_rows` finds
    /// `shapes`/`row_status`/`float_outputs[0]` mutually inconsistent, or
    /// when `row` is out of range for the derived row count — see
    /// [`Self`]'s row-major invariant doc.
    pub fn single_row_or_err(&self, row: usize) -> Result<&[f32]> {
        let rows = self.checked_rows()?;
        if row >= rows {
            return Err(JammiError::Inference(format!(
                "BackendOutput row {row} is out of range for {rows} row(s)"
            )));
        }
        if !self.row_status[row] {
            return Err(self.row_error(row));
        }
        let dim = self.shapes[0].1;
        let start = row * dim;
        Ok(&self.float_outputs[0][start..start + dim])
    }

    /// Returns output head 0's full flattened matrix only when EVERY row
    /// succeeded; otherwise returns the lowest-index failed row's typed error
    /// (mirroring the trainer's decode-batch convention of surfacing the
    /// lowest-index error as a whole-step refusal).
    ///
    /// A batch consumer that forwards several items in one call (e.g. a
    /// fine-tune trainer projecting a frozen embedding for a training group)
    /// must never silently train on the all-zero placeholder a per-row
    /// backend substitutes for a decode/preprocess refusal — a corrupt
    /// training item is a refusal, not a row to skip.
    ///
    /// Refuses (rather than reading) whenever `checked_rows` finds
    /// `shapes`/`row_status`/`float_outputs[0]` mutually inconsistent — see
    /// [`Self`]'s row-major invariant doc. An intentionally empty (zero-row)
    /// output is also refused: there is no embedding to hand back.
    pub fn all_rows_or_err(&self) -> Result<&[f32]> {
        let rows = self.checked_rows()?;
        if let Some(bad) = self.row_status.iter().position(|ok| !ok) {
            return Err(self.row_error(bad));
        }
        if rows == 0 {
            return Err(JammiError::Inference("No embedding output".into()));
        }
        Ok(&self.float_outputs[0])
    }
}

/// Converts raw backend output into Arrow arrays for a specific task.
pub trait OutputAdapter: Send + Sync {
    /// Arrow schema for this task's output columns (excluding common prefix).
    fn output_schema(&self) -> Vec<Field>;

    /// Convert raw backend output into Arrow arrays for one batch.
    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>>;
}

/// Create an adapter for a given task with model-derived dimensions.
pub fn create_adapter(task: ModelTask, model: &LoadedModel) -> Result<Box<dyn OutputAdapter>> {
    match task {
        ModelTask::TextEmbedding | ModelTask::ImageEmbedding | ModelTask::AudioEmbedding => {
            let dim = model.embedding_dim().ok_or_else(|| {
                JammiError::Inference("Model does not report embedding dim".into())
            })?;
            Ok(Box::new(EmbeddingAdapter::new(dim)))
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

/// Build a nullable StringArray: rows where `row_status[i]` is false become null.
pub(crate) fn nullify_strings(
    values: Option<&Vec<String>>,
    row_status: &[bool],
    row_count: usize,
) -> StringArray {
    match values {
        Some(v) => v
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if row_status.get(i).copied().unwrap_or(false) {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect(),
        None => vec![None::<&str>; row_count].into_iter().collect(),
    }
}

/// Build a nullable Float32Array: rows where `row_status[i]` is false become null.
pub(crate) fn nullify_floats(
    values: Option<&Vec<f32>>,
    row_status: &[bool],
    row_count: usize,
) -> Float32Array {
    match values {
        Some(v) => v
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                if row_status.get(i).copied().unwrap_or(false) {
                    Some(c)
                } else {
                    None
                }
            })
            .collect(),
        None => vec![None::<f32>; row_count].into_iter().collect(),
    }
}

// ─── `BackendOutput` accessor oracles (#421 frontend follow-on, round 3) ────

#[cfg(test)]
mod tests {
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

    /// Pre-fix, `session.rs`'s `encode_image_query`/`encode_audio_query` read
    /// `output.float_outputs[0][..dim].to_vec()` directly — a failed row 0
    /// silently became the all-zero placeholder instead of an `Err`. This is
    /// the oracle that would stay GREEN on that blind read (`[0.0, 0.0]` is a
    /// valid slice) and only fails once the caller goes through the checked
    /// accessor — verified by temporarily reverting `single_row_or_err`'s body
    /// to `&self.float_outputs[0][row * dim..row * dim + dim]`, which turns
    /// this RED (`Ok([0.0, 0.0])` vs the expected `Err`).
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
    // BLOCK 2: `all_rows_or_err` pre-fix failed OPEN on an empty/short
    // `row_status` (its `.position(|ok| !ok)` scan finds nothing wrong and
    // would return a truncated/garbage slice as if every row succeeded),
    // while `single_row_or_err` failed CLOSED via `row_status.get(row)`. Both
    // must now refuse, by name, on every one of these malformed states.

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
}
