use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

/// Adapt raw float embeddings into a `FixedSizeList<Float32>` Arrow column.
pub struct EmbeddingAdapter {
    dimensions: usize,
}

impl EmbeddingAdapter {
    /// Create an adapter for embeddings of the given dimensionality.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
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

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
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

        let flat_values = &output.float_outputs[0];
        let values_array = Float32Array::from(flat_values.clone());
        let nulls = NullBuffer::from(output.row_status.clone());
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
