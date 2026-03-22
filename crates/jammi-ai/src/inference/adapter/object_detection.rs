use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use jammi_engine::error::Result;

use super::{BackendOutput, OutputAdapter};

pub struct ObjectDetectionAdapter;

impl OutputAdapter for ObjectDetectionAdapter {
    fn output_schema(&self) -> Vec<Field> {
        vec![Field::new("detections", DataType::Utf8, true)]
    }

    fn adapt(&self, _output: &BackendOutput, _row_count: usize) -> Result<Vec<ArrayRef>> {
        todo!("ObjectDetectionAdapter::adapt")
    }
}
