use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

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
        let labels: StringArray = output
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

        let confidences: Float32Array = output
            .float_outputs
            .first()
            .map(|v| {
                v.iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        if output.row_status.get(i).copied().unwrap_or(false) {
                            Some(c)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_else(|| vec![None::<f32>; row_count].into_iter().collect());

        let scores_json: StringArray = output
            .string_outputs
            .get(1)
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

        Ok(vec![
            Arc::new(labels),
            Arc::new(confidences),
            Arc::new(scores_json),
        ])
    }
}
