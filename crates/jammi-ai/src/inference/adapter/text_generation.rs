use std::sync::Arc;

use arrow::array::{ArrayRef, LargeStringArray, StringArray};
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

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
        let texts: LargeStringArray = output
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

        let reasons: StringArray = output
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

        Ok(vec![Arc::new(texts), Arc::new(reasons)])
    }
}
