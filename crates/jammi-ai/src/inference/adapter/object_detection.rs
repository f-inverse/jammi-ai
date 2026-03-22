use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

/// Object detection adapter — serializes detections as JSON per row.
/// Full structured output (List<Struct{label, confidence, bbox: FixedSizeList(4)}>)
/// lands when detection models are integrated.
pub struct ObjectDetectionAdapter;

impl OutputAdapter for ObjectDetectionAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("detections", DataType::Utf8, true)]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        let detections: StringArray = output
            .string_outputs
            .first()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(i, s)| {
                        if output.row_status.get(i).copied().unwrap_or(false) {
                            Some(s.as_str())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![None::<&str>; row_count].into_iter().collect());
        Ok(vec![Arc::new(detections)])
    }
}
