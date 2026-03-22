use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

pub struct NerAdapter;

impl OutputAdapter for NerAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("entities", DataType::Utf8, true)]
    }

    fn adapt(&self, _output: &BackendOutput, _row_count: usize) -> Result<Vec<ArrayRef>> {
        todo!("NerAdapter::adapt")
    }
}
