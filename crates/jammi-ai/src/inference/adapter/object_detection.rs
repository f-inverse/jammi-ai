use std::sync::Arc;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{nullify_strings, BackendOutput, OutputAdapter};

/// Object detection adapter — serializes detections as JSON per row.
/// Full structured output (List<Struct{label, confidence, bbox: FixedSizeList(4)}>)
/// lands when detection models are integrated.
pub struct ObjectDetectionAdapter;

impl OutputAdapter for ObjectDetectionAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("detections", DataType::Utf8, true)]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        Ok(vec![Arc::new(nullify_strings(
            output.string_outputs.first(),
            &output.row_status,
            row_count,
        ))])
    }
}
