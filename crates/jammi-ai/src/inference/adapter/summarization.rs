use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{nullify_large_strings, BackendOutput, OutputAdapter};

/// Adapt summarization output into a `summary` LargeUtf8 column.
pub struct SummarizationAdapter;

impl OutputAdapter for SummarizationAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("summary", DataType::LargeUtf8, true)]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        Ok(vec![Arc::new(nullify_large_strings(
            output.string_outputs.first(),
            &output.row_status,
            row_count,
        ))])
    }
}
