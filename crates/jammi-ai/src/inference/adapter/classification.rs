use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{nullify_floats, nullify_strings, BackendOutput, OutputAdapter};

/// Adapt classification output into `label`, `confidence`, and `all_scores_json` columns.
pub struct ClassificationAdapter;

impl OutputAdapter for ClassificationAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![
            Field::new("label", DataType::Utf8, true),
            Field::new("confidence", DataType::Float32, true),
            Field::new("all_scores_json", DataType::Utf8, true),
        ]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        Ok(vec![
            Arc::new(nullify_strings(
                output.string_outputs.first(),
                &output.row_status,
                row_count,
            )),
            Arc::new(nullify_floats(
                output.float_outputs.first(),
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
