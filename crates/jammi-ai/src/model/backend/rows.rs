//! The rows of one forward, as every backend sees them: which rows carry
//! input, and how the vectors computed for those rows are laid back out at
//! their Arrow positions. A backend owns only the computation between the
//! two.

use arrow::array::ArrayRef;
use jammi_datafusion::BackendOutput;
use jammi_db::error::{JammiError, Result};

use crate::inference::arrow_to_texts;

/// The rows of one text forward: every row's rendered text, and which rows
/// carry any. An empty or null text is marked at its row and never reaches
/// a model.
pub(crate) struct TextRows {
    pub(crate) texts: Vec<String>,
    pub(crate) valid: Vec<usize>,
    pub(crate) row_status: Vec<bool>,
    pub(crate) row_errors: Vec<String>,
}

impl TextRows {
    pub(crate) fn of(content: &[ArrayRef]) -> Result<Self> {
        let texts = arrow_to_texts(content)?;
        let (valid, row_status, row_errors) = texts.iter().enumerate().fold(
            (
                Vec::with_capacity(texts.len()),
                Vec::with_capacity(texts.len()),
                Vec::with_capacity(texts.len()),
            ),
            |(mut valid, mut status, mut errors), (i, text)| {
                if text.is_empty() {
                    status.push(false);
                    errors.push("Empty or null text input".to_string());
                } else {
                    valid.push(i);
                    status.push(true);
                    errors.push(String::new());
                }
                (valid, status, errors)
            },
        );
        Ok(Self {
            texts,
            valid,
            row_status,
            row_errors,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.texts.len()
    }

    pub(crate) fn valid_texts(&self) -> Vec<&str> {
        self.valid.iter().map(|&i| self.texts[i].as_str()).collect()
    }
}

/// The row count of a content column set, refusing an empty set the way
/// every content reader does.
pub(crate) fn content_row_count(content: &[ArrayRef]) -> Result<usize> {
    content
        .first()
        .map(|c| c.len())
        .ok_or_else(|| JammiError::Inference("No content columns provided".into()))
}

/// One float-embedding head over every row of a forward: `embedded[k]` is
/// the vector of Arrow row `valid[k]`, and a row that carried no input
/// (`row_status` false) holds zeros under its recorded error.
///
/// A zero-row forward reports the shape `(0, 0)` — `BackendOutput`'s
/// documented empty-batch shape, the one every producer reports the same
/// way. Every vector must be `dim` wide: a producer handing back another
/// width is refused naming the row, before anything is copied.
pub(crate) fn embedding_output(
    valid: &[usize],
    row_status: Vec<bool>,
    row_errors: Vec<String>,
    dim: usize,
    embedded: &[&[f32]],
) -> Result<BackendOutput> {
    let num_rows = row_status.len();
    if num_rows == 0 {
        return Ok(BackendOutput {
            float_outputs: vec![vec![]],
            string_outputs: vec![],
            row_status: vec![],
            row_errors: vec![],
            shapes: vec![(0, 0)],
        });
    }
    if embedded.len() != valid.len() {
        return Err(JammiError::Inference(format!(
            "{} vector(s) for {} row(s) with input",
            embedded.len(),
            valid.len()
        )));
    }
    if let Some((k, vector)) = embedded.iter().enumerate().find(|(_, v)| v.len() != dim) {
        return Err(JammiError::Inference(format!(
            "row {} has width {}, expected {dim}",
            valid[k],
            vector.len()
        )));
    }
    let total = num_rows.checked_mul(dim).ok_or_else(|| {
        JammiError::Inference(format!("rows*dim overflows (rows={num_rows}, dim={dim})"))
    })?;
    let mut flat = vec![0.0_f32; total];
    for (&row, vector) in valid.iter().zip(embedded) {
        flat[row * dim..(row + 1) * dim].copy_from_slice(vector);
    }
    Ok(BackendOutput {
        float_outputs: vec![flat],
        string_outputs: vec![],
        row_status,
        row_errors,
        shapes: vec![(num_rows, dim)],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors land at their Arrow rows; a row with no input holds zeros
    /// and keeps its error.
    #[test]
    fn vectors_land_at_their_rows_and_empty_rows_hold_zeros() {
        let out = embedding_output(
            &[0, 2],
            vec![true, false, true],
            vec![
                String::new(),
                "Empty or null text input".into(),
                String::new(),
            ],
            2,
            &[&[1.0, 2.0], &[3.0, 4.0]],
        )
        .unwrap();
        assert_eq!(out.float_outputs[0], vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        assert_eq!(out.shapes, vec![(3, 2)]);
        assert_eq!(out.row_errors[1], "Empty or null text input");
    }

    /// A vector of another width is refused naming its Arrow row, not
    /// spliced into its neighbour's slot.
    #[test]
    fn a_vector_of_another_width_is_refused_naming_its_row() {
        let err = embedding_output(
            &[0, 2],
            vec![true, false, true],
            vec![String::new(); 3],
            2,
            &[&[1.0, 2.0], &[3.0]],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("row 2 has width 1, expected 2"), "{err}");
    }

    #[test]
    fn a_zero_row_forward_has_the_empty_batch_shape() {
        let out = embedding_output(&[], vec![], vec![], 8, &[]).unwrap();
        assert_eq!(out.shapes, vec![(0, 0)]);
    }
}
