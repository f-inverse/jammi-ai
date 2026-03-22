use std::sync::Arc;

use arrow::array::{ArrayRef, LargeStringArray};
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

pub struct SummarizationAdapter;

impl OutputAdapter for SummarizationAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("summary", DataType::LargeUtf8, true)]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        let summaries: LargeStringArray = output
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
        Ok(vec![Arc::new(summaries)])
    }
}
