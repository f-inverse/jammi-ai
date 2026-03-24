use std::fs::File;
use std::path::Path;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::error::Result;

/// Writes Arrow RecordBatches to a Parquet file with ZSTD compression
/// and 64K row groups.
pub struct ParquetResultWriter {
    writer: ArrowWriter<File>,
    row_count: usize,
}

impl ParquetResultWriter {
    /// Create a new writer at the given path.
    pub fn new(path: &Path, schema: SchemaRef) -> Result<Self> {
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .set_max_row_group_size(65_536)
            .build();
        let writer = ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| crate::error::JammiError::Other(format!("Parquet writer init: {e}")))?;
        Ok(Self {
            writer,
            row_count: 0,
        })
    }

    /// Write a batch to the Parquet file.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.writer
            .write(batch)
            .map_err(|e| crate::error::JammiError::Other(format!("Parquet write: {e}")))?;
        self.row_count += batch.num_rows();
        Ok(())
    }

    /// Flush and close the writer, returning the total row count.
    pub fn close(self) -> Result<usize> {
        let count = self.row_count;
        self.writer
            .close()
            .map_err(|e| crate::error::JammiError::Other(format!("Parquet close: {e}")))?;
        Ok(count)
    }
}
