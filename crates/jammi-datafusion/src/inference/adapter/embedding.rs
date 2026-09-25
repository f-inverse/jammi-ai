use std::sync::Arc;

use crate::error::{Error, Result};
use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};

use super::OutputAdapter;
use crate::inference::output::BackendOutput;

/// Adapt raw float embeddings into a `FixedSizeList<Float32>` Arrow column.
pub struct EmbeddingAdapter {
    /// The width the model emits.
    native: usize,
    /// The width the adapter serves: `native`, or a Matryoshka prefix of it.
    dimensions: usize,
}

impl EmbeddingAdapter {
    /// Create an adapter serving embeddings at the model's own width.
    pub fn new(dimensions: usize) -> Self {
        Self {
            native: dimensions,
            dimensions,
        }
    }

    /// Create an adapter serving the leading `dimensions` coordinates of a
    /// model that emits `native`, each row L2-renormalised
    /// ([`matryoshka_prefix`]). A width of zero, or wider than the model's, is
    /// refused.
    pub fn serving(native: usize, dimensions: usize) -> Result<Self> {
        if dimensions == 0 || dimensions > native {
            return Err(Error::Inference(format!(
                "cannot serve {dimensions} dimensions of an embedding {native} wide"
            )));
        }
        Ok(Self { native, dimensions })
    }
}

/// The leading `width` coordinates of `row`, L2-renormalised: a Matryoshka
/// prefix served as an embedding of its own. A prefix with zero norm is
/// returned as is.
pub fn matryoshka_prefix(row: &[f32], width: usize) -> Vec<f32> {
    let prefix = &row[..width.min(row.len())];
    let norm = prefix.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        prefix.iter().map(|v| v / norm).collect()
    } else {
        prefix.to_vec()
    }
}

impl OutputAdapter for EmbeddingAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new_fixed_size_list(
            "vector",
            Field::new("item", DataType::Float32, false),
            self.dimensions as i32,
            true,
        )]
    }

    fn adapt(&self, output: BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        if row_count == 0 {
            let field = Arc::new(Field::new("item", DataType::Float32, false));
            let empty = FixedSizeListArray::new(
                field,
                self.dimensions as i32,
                Arc::new(Float32Array::from(Vec::<f32>::new())),
                None,
            );
            return Ok(vec![Arc::new(empty)]);
        }

        let BackendOutput {
            mut float_outputs,
            row_status,
            ..
        } = output;
        if float_outputs.is_empty() {
            return Err(Error::Inference(
                "embedding adapter: backend emitted no float head".into(),
            ));
        }
        let flat_values = float_outputs.swap_remove(0);
        // `row_count * self.native` with a raw multiply can silently
        // overflow on an adversarial `row_count`/`native` pair; the
        // checked multiply refuses by name instead (mirrors
        // `BackendOutput::checked_rows`'s `rows.checked_mul(dim)`).
        let expected = row_count.checked_mul(self.native).ok_or_else(|| {
            Error::Inference(format!(
                "embedding adapter: row_count*dim overflows (row_count={row_count}, \
                 dim={})",
                self.native
            ))
        })?;
        if flat_values.len() != expected {
            return Err(Error::Inference(format!(
                "embedding adapter: head has {} floats, expected rows({row_count}) * \
                 dim({})",
                flat_values.len(),
                self.native
            )));
        }
        let flat_values = if self.dimensions == self.native {
            flat_values
        } else {
            flat_values
                .chunks_exact(self.native)
                .flat_map(|row| matryoshka_prefix(row, self.dimensions))
                .collect()
        };
        // `FixedSizeListArray::new` panics (rather than returning an error)
        // when the null buffer's length disagrees with the values array's
        // row count. Refuse by name here, before construction, rather than
        // let a malformed `row_status` abort the process below.
        if row_status.len() != row_count {
            return Err(Error::Inference(format!(
                "embedding adapter: row_status has {} entries, expected one per row \
                 ({row_count})",
                row_status.len()
            )));
        }
        let values_array = Float32Array::from(flat_values);
        let nulls = NullBuffer::from(row_status);
        let field = Arc::new(Field::new("item", DataType::Float32, false));
        let array = FixedSizeListArray::new(
            field,
            self.dimensions as i32,
            Arc::new(values_array),
            Some(nulls),
        );
        Ok(vec![Arc::new(array)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;

    fn two_rows_dim3(status: Vec<bool>) -> BackendOutput {
        BackendOutput {
            float_outputs: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
            string_outputs: vec![],
            row_status: status,
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 3)],
        }
    }

    /// A `BackendOutput` with no float head at all (`float_outputs` empty)
    /// must be a named refusal, not an index into an absent element 0.
    /// Verified by temporarily reverting the `float_outputs.first()` guard to
    /// an unchecked `output.float_outputs[0]`: this test goes RED (a panic,
    /// not the `Err` asserted below).
    #[test]
    fn adapt_refuses_a_producer_with_no_float_head() {
        let out = BackendOutput {
            float_outputs: vec![],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 3)],
        };
        let err = EmbeddingAdapter::new(3).adapt(out.clone(), 2).unwrap_err();
        assert!(err.to_string().contains("no float head"), "{err}");
    }

    /// A flat head whose length disagrees with `row_count * dim` must be a
    /// named refusal, not a slice that reads past (or short of) the intended
    /// rows. Verified by temporarily removing the length check: this test
    /// goes RED (`Float32Array::from` builds a mis-shaped array instead of
    /// the `Err` asserted below).
    #[test]
    fn adapt_refuses_a_flat_length_that_disagrees_with_rows_times_dim() {
        let out = BackendOutput {
            // shapes/row_count say 2 rows of dim 3 (6 values); the flat
            // buffer is one short.
            float_outputs: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]],
            string_outputs: vec![],
            row_status: vec![true, true],
            row_errors: vec![String::new(), String::new()],
            shapes: vec![(2, 3)],
        };
        let err = EmbeddingAdapter::new(3).adapt(out.clone(), 2).unwrap_err();
        assert!(err.to_string().contains("rows"), "{err}");
    }

    /// `row_count * dim` computed with a raw multiply could silently overflow
    /// on an adversarial pair; the checked multiply must refuse by name
    /// instead of wrapping. Verified by temporarily reverting the checked
    /// multiply to a raw `row_count * self.dimensions`: this test goes RED
    /// (a debug-mode overflow panic, or a wrapped small `expected` in
    /// release, rather than the `Err` asserted below).
    #[test]
    fn adapt_refuses_an_overflowing_row_count_times_dim_without_panicking() {
        let out = BackendOutput {
            float_outputs: vec![vec![]],
            string_outputs: vec![],
            row_status: vec![],
            row_errors: vec![],
            shapes: vec![(usize::MAX, 3)],
        };
        let err = EmbeddingAdapter::new(3)
            .adapt(out.clone(), usize::MAX)
            .unwrap_err();
        assert!(err.to_string().contains("overflow"), "{err}");
    }

    /// `FixedSizeListArray::new` panics inside arrow when the null buffer's
    /// length disagrees with the values array's row count. A `row_status`
    /// shorter than `row_count` must be refused by name before construction,
    /// never left to arrow's internal `assert_eq!`. Verified by temporarily
    /// removing the `row_status.len() != row_count` guard: this test goes
    /// RED (a panic inside `FixedSizeListArray::new`, not the `Err` asserted
    /// below).
    #[test]
    fn adapt_refuses_a_row_status_length_mismatch_before_building_the_array() {
        let out = two_rows_dim3(vec![true]); // one short of row_count(2)
        let err = EmbeddingAdapter::new(3).adapt(out.clone(), 2).unwrap_err();
        assert!(err.to_string().contains("row_status"), "{err}");
    }

    #[test]
    fn adapt_builds_a_consistent_array_on_the_happy_path() {
        let out = two_rows_dim3(vec![true, false]);
        let cols = EmbeddingAdapter::new(3).adapt(out.clone(), 2).unwrap();
        let array = cols[0]
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(array.len(), 2);
        assert!(array.is_valid(0));
        assert!(array.is_null(1));
    }

    #[test]
    fn a_served_prefix_is_each_rows_leading_coordinates_renormalised() {
        let out = two_rows_dim3(vec![true, true]);
        let cols = EmbeddingAdapter::serving(3, 2)
            .unwrap()
            .adapt(out, 2)
            .unwrap();
        let array = cols[0]
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        assert_eq!(array.value_length(), 2);
        let values = array
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec();
        let (a, b) = (1.0 / 5f32.sqrt(), 2.0 / 5f32.sqrt());
        let (c, d) = (4.0 / 41f32.sqrt(), 5.0 / 41f32.sqrt());
        for (got, want) in values.iter().zip([a, b, c, d]) {
            assert!((got - want).abs() < 1e-6, "{values:?}");
        }
    }

    #[test]
    fn a_prefix_wider_than_the_model_or_empty_is_refused() {
        assert!(EmbeddingAdapter::serving(3, 4).is_err());
        assert!(EmbeddingAdapter::serving(3, 0).is_err());
        assert!(EmbeddingAdapter::serving(3, 3).is_ok());
    }

    #[test]
    fn a_zero_prefix_is_served_unscaled() {
        assert_eq!(matryoshka_prefix(&[0.0, 0.0, 1.0], 2), vec![0.0, 0.0]);
    }
}
