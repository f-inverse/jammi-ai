use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

/// NER adapter — serializes entity spans as JSON per row.
/// Full structured output (List<Struct{text, label, start, end, confidence}>)
/// lands when NER models are integrated.
pub struct NerAdapter;

impl OutputAdapter for NerAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("entities", DataType::Utf8, true)]
    }

    fn adapt(&self, output: &BackendOutput, row_count: usize) -> Result<Vec<ArrayRef>> {
        let entities: StringArray = output
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
        Ok(vec![Arc::new(entities)])
    }
}
