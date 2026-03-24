use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{nullify_strings, BackendOutput, OutputAdapter};

/// NER adapter — serializes entity spans as JSON per row.
/// Full structured output (List<Struct{text, label, start, end, confidence}>)
/// lands when NER models are integrated.
pub struct NerAdapter;

impl OutputAdapter for NerAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("entities", DataType::Utf8, true)]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        Ok(vec![Arc::new(nullify_strings(
            output.string_outputs.first(),
            &output.row_status,
            row_count,
        ))])
    }
}
