use std::sync::Arc;

use arrow::array::{ArrayRef, ListArray, RecordBatch, StringArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};

use jammi_engine::error::{JammiError, Result};

/// Add `retrieved_by` and `annotated_by` `List<Utf8>` columns to result batches.
pub fn add_provenance(
    batches: &[RecordBatch],
    channels: &[String],
    has_annotation: bool,
) -> Result<Vec<RecordBatch>> {
    batches
        .iter()
        .map(|batch| {
            let row_count = batch.num_rows();

            // retrieved_by: channels that found each row (exclude "inference" if annotated)
            let retrieval_channels: Vec<&str> = channels
                .iter()
                .filter(|c| c.as_str() != "inference" || !has_annotation)
                .map(|c| c.as_str())
                .collect();
            let retrieved_by = build_list_column(row_count, &retrieval_channels)?;

            // annotated_by: channels that added evidence post-retrieval
            let annotated_by = if has_annotation {
                build_list_column(row_count, &["inference"])?
            } else {
                build_list_column(row_count, &[])?
            };

            // Append provenance columns to the batch
            let mut fields: Vec<Arc<Field>> = batch.schema().fields().to_vec();
            let mut columns: Vec<ArrayRef> = batch.columns().to_vec();

            let list_type = DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)));
            fields.push(Arc::new(Field::new(
                "retrieved_by",
                list_type.clone(),
                false,
            )));
            fields.push(Arc::new(Field::new("annotated_by", list_type, false)));
            columns.push(retrieved_by);
            columns.push(annotated_by);

            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|e| {
                jammi_engine::error::JammiError::Other(format!("Provenance assembly: {e}"))
            })
        })
        .collect()
}

/// Build a `List<Utf8>` column where every row has the same list of values.
fn build_list_column(row_count: usize, values: &[&str]) -> Result<ArrayRef> {
    let flat_values: Vec<&str> = (0..row_count)
        .flat_map(|_| values.iter().copied())
        .collect();
    let values_array = Arc::new(StringArray::from(flat_values));

    let offsets: Vec<i32> = (0..=row_count).map(|i| (i * values.len()) as i32).collect();

    let list = ListArray::try_new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        OffsetBuffer::new(offsets.into()),
        values_array,
        None,
    )
    .map_err(|e| JammiError::Other(format!("List construction: {e}")))?;

    Ok(Arc::new(list))
}
