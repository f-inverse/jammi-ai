//! What a model's forward returns, before a task adapter shapes it into
//! columns.

use crate::error::{Error, Result};

/// Raw output from a model backend, before task-specific adaptation.
///
/// # Output head 0's row-major invariant
///
/// [`Self::single_row_or_err`] and [`Self::all_rows_or_err`] read a single
/// FLOAT-EMBEDDING head: `float_outputs[0]`, one flattened ROW-MAJOR `[rows,
/// dim]` matrix, where `rows` and `dim` are `shapes[0]` — never as one `Vec`
/// per row. This is the accessors' whole domain, checked as ONE consistency
/// gate (`checked_rows`) before either accessor reads a single value:
///
/// - `float_outputs` has exactly one head (`float_outputs.len() == 1`).
/// - `shapes` has at least one entry (`shapes[0] = (rows, dim)`), with
///   `rows >= 1` (there is nothing to hand back for an empty output) and
///   `dim >= 1` (a zero-width row carries no embedding).
/// - `row_status.len() == rows` and `row_errors.len() == rows`.
/// - `float_outputs[0].len() == rows * dim`, computed with a checked
///   multiply (`rows.checked_mul(dim)`) rather than a raw `rows * dim` that
///   could silently overflow on an adversarial shape.
///
/// A producer that violates this (e.g. one `Vec<f32>` per row, or a head
/// that has no real embedding at all) is a bug in the producer, not
/// something a reader can safely reinterpret — both accessors refuse by
/// name rather than misread or panic on the buffer. Use [`Self::single_head`]
/// to construct a single-float-head `BackendOutput` with this invariant
/// checked at construction time.
///
/// This invariant is scoped to a float-EMBEDDING head; it says nothing
/// about a producer with no embedding head at all. NER carries no float
/// head (`float_outputs` is empty and `shapes` is likewise empty), and
/// classification carries a width-1 float head (`shapes[0] = (rows, 1)`,
/// one confidence score per row) — neither goes through
/// [`Self::single_row_or_err`]/[`Self::all_rows_or_err`], whose task
/// adapters (`ClassificationAdapter`, `NerAdapter`) read `float_outputs`/
/// `string_outputs` directly instead.
///
/// # The empty-batch shape: `(0, 0)`
///
/// A zero-row batch is reported as `shapes[0] = (0, 0)` by every
/// float-embedding producer in this crate (`CandleModel::forward_embedding`
/// / `forward_image_embedding` / `forward_audio_embedding`, and
/// `HttpBackend::forward_embeddings`), never `(0, <a known dim>)`: `(rows,
/// 0)` is already this doc's own "no real embedding" shape, and `(0, 0)` is
/// the ONE such shape every producer can report honestly, including a
/// remote backend that has no model config and so cannot know an embedding
/// width before a request completes. This value is descriptive only —
/// `EmbeddingAdapter::adapt`'s own `row_count == 0` branch never reads
/// `shapes` at all, building the empty output off its own separately-known
/// `dimensions` field instead — but every producer still reports it
/// uniformly rather than each picking its own placeholder.
#[derive(Debug, Clone)]
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
    /// [`Self::all_rows_or_err`] later. `rows` and `dim` must both be `>= 1`
    /// (this constructor is for a real float-embedding head, never a
    /// zero-row or zero-width placeholder); `flat.len()` must equal
    /// `rows * dim`; `row_status`/`row_errors` must each carry exactly one
    /// entry per row.
    pub fn single_head(
        flat: Vec<f32>,
        rows: usize,
        dim: usize,
        row_status: Vec<bool>,
        row_errors: Vec<String>,
    ) -> Result<Self> {
        if rows == 0 {
            return Err(Error::Inference(
                "BackendOutput::single_head: rows must be >= 1 (a zero-row float-embedding \
                 head has no embedding to construct)"
                    .into(),
            ));
        }
        if dim == 0 {
            return Err(Error::Inference(
                "BackendOutput::single_head: dim must be >= 1 (a zero-width row carries no \
                 embedding)"
                    .into(),
            ));
        }
        let expected = rows.checked_mul(dim).ok_or_else(|| {
            Error::Inference(format!(
                "BackendOutput::single_head: rows*dim overflows (rows={rows}, dim={dim})"
            ))
        })?;
        if flat.len() != expected {
            return Err(Error::Inference(format!(
                "BackendOutput::single_head: flat buffer has {} value(s), expected rows*dim \
                 ({rows}*{dim})",
                flat.len()
            )));
        }
        if row_status.len() != rows {
            return Err(Error::Inference(format!(
                "BackendOutput::single_head: row_status has {} entries, expected one per row \
                 ({rows})",
                row_status.len()
            )));
        }
        if row_errors.len() != rows {
            return Err(Error::Inference(format!(
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
    fn row_error(&self, row: usize) -> Error {
        match self.row_errors.get(row).filter(|m| !m.is_empty()) {
            Some(msg) => Error::Inference(msg.clone()),
            None => Error::Inference(format!("Row {row} inference failed")),
        }
    }

    /// Verify the accessors' whole domain (see [`Self`]'s row-major
    /// invariant doc) and return output head 0's authoritative row count
    /// from `shapes[0].0` — the ONE consistency check both
    /// [`Self::single_row_or_err`] and [`Self::all_rows_or_err`] run before
    /// looking at any individual row's status, so an inconsistent producer
    /// is refused BY NAME, on both paths, before either ever indexes
    /// `float_outputs[0]` — never a panic, never a vacuous `Ok`.
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
        if self.float_outputs.len() != 1 {
            return Err(Error::Inference(format!(
                "BackendOutput has {} float head(s), expected exactly one (a float-embedding \
                 head)",
                self.float_outputs.len()
            )));
        }
        let (rows, dim) = *self
            .shapes
            .first()
            .ok_or_else(|| Error::Inference("BackendOutput has no output-head shape".into()))?;
        if rows == 0 {
            return Err(Error::Inference(
                "BackendOutput has zero rows (no embedding output)".into(),
            ));
        }
        if dim == 0 {
            return Err(Error::Inference(
                "BackendOutput's float-embedding head has dim 0 (no embedding output)".into(),
            ));
        }
        let expected = rows.checked_mul(dim).ok_or_else(|| {
            Error::Inference(format!(
                "BackendOutput shape rows*dim overflows (rows={rows}, dim={dim})"
            ))
        })?;
        if self.row_status.len() != rows {
            return Err(Error::Inference(format!(
                "BackendOutput row_status has {} entries, expected one per row ({rows})",
                self.row_status.len()
            )));
        }
        if self.row_errors.len() != rows {
            return Err(Error::Inference(format!(
                "BackendOutput row_errors has {} entries, expected one per row ({rows})",
                self.row_errors.len()
            )));
        }
        let flat_len = self.float_outputs[0].len();
        if flat_len != expected {
            return Err(Error::Inference(format!(
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
            return Err(Error::Inference(format!(
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
    /// Refuses (rather than reading) whenever `checked_rows` finds the
    /// domain violated — see [`Self`]'s row-major invariant doc; this
    /// includes an intentionally empty (zero-row) output, which
    /// `checked_rows` refuses by name since there is no embedding to hand
    /// back.
    pub fn all_rows_or_err(&self) -> Result<&[f32]> {
        let _rows = self.checked_rows()?;
        if let Some(bad) = self.row_status.iter().position(|ok| !ok) {
            return Err(self.row_error(bad));
        }
        Ok(&self.float_outputs[0])
    }
}
