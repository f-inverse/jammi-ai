use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{nullify_large_strings, nullify_strings, BackendOutput, OutputAdapter};

/// Adapt text generation output into `generated_text` and `finish_reason` columns.
pub struct TextGenerationAdapter;

impl OutputAdapter for TextGenerationAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![
            Field::new("generated_text", DataType::LargeUtf8, true),
            Field::new("finish_reason", DataType::Utf8, true),
        ]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        Ok(vec![
            Arc::new(nullify_large_strings(
                output.string_outputs.first(),
                &output.row_status,
                row_count,
            )),
            Arc::new(nullify_strings(
                output.string_outputs.get(1),
                &output.row_status,
                row_count,
            )),
        ])
    }
}
