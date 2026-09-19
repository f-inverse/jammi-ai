//! Forward chunks as a pure function of `_ordinal`.
//!
//! An embedding's bytes depend on the composition of the batch it was
//! forwarded in (padding to the longest row; the GEMM fold order), so the
//! rows that share one forward must be decided by the data alone — never by
//! how batches happen to arrive, which differs with the partition count, with
//! a re-batching exchange, and across a process boundary. The decision is one
//! expression, [`chunk_expr`]: `_ordinal / batch_size`. Rows with equal chunk
//! id are forwarded together; nothing else is.
//!
//! The same expression is the hash key of the exchange that fans a plan out
//! (`operator::inference_exec`), so a chunk is never divided between
//! partitions, and [`ChunkAssembler`] regroups a partition's rows into whole
//! chunks whatever batch boundaries they arrived with.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{col, lit, BinaryExpr};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::scalar::ScalarValue;
use jammi_db::error::{JammiError, Result};

use super::schema::ORDINAL_COLUMN;

/// Refuse a schema that does not carry `_ordinal` as `UInt64 NOT NULL` — the
/// column every model-facing input is numbered with
/// (`operator::numbered_input_exec`).
pub fn require_ordinal(schema: &Schema) -> DfResult<()> {
    let field = schema.field_with_name(ORDINAL_COLUMN).map_err(|_| {
        DataFusionError::Plan(format!(
            "inference input has no '{ORDINAL_COLUMN}' column: a model reads a numbered input"
        ))
    })?;
    if field.data_type() != &DataType::UInt64 || field.is_nullable() {
        return Err(DataFusionError::Plan(format!(
            "inference input column '{ORDINAL_COLUMN}' must be UInt64 NOT NULL, found {:?}{}",
            field.data_type(),
            if field.is_nullable() { " NULL" } else { "" }
        )));
    }
    Ok(())
}

/// The forward-chunk id of a row of `schema`: `_ordinal / batch_size`.
pub fn chunk_expr(schema: &Schema, batch_size: NonZeroUsize) -> DfResult<Arc<dyn PhysicalExpr>> {
    require_ordinal(schema)?;
    let ordinal = col(ORDINAL_COLUMN, schema)?;
    let size = lit(ScalarValue::UInt64(Some(batch_size.get() as u64)));
    Ok(Arc::new(BinaryExpr::new(ordinal, Operator::Divide, size)))
}

/// The rows of the chunk currently being gathered.
struct OpenChunk {
    id: u64,
    parts: Vec<RecordBatch>,
    rows: usize,
}

/// Regroups a stream of `_ordinal`-ascending batches into whole forward
/// chunks: consecutive rows with one [`chunk_expr`] value, at most
/// `batch_size` of them.
///
/// The chunk sequence it yields is a function of the rows alone, identical
/// under every re-batching of the same rows. A chunk closes when the chunk id
/// changes, when it holds `batch_size` rows, or at end of input. A row whose
/// chunk id is lower than one already seen means the input is not in
/// `_ordinal` order — a chunk could then be divided silently — and is refused.
pub struct ChunkAssembler {
    schema: SchemaRef,
    chunk_id: Arc<dyn PhysicalExpr>,
    capacity: usize,
    open: Option<OpenChunk>,
    last_id: Option<u64>,
}

impl ChunkAssembler {
    /// An assembler over batches of `schema`, chunked by `batch_size`.
    pub fn try_new(schema: SchemaRef, batch_size: NonZeroUsize) -> Result<Self> {
        let chunk_id = chunk_expr(schema.as_ref(), batch_size)?;
        Ok(Self {
            schema,
            chunk_id,
            capacity: batch_size.get(),
            open: None,
            last_id: None,
        })
    }

    /// Take `batch`'s rows, returning every chunk they complete, in order.
    pub fn push(&mut self, batch: &RecordBatch) -> Result<Vec<RecordBatch>> {
        let ids = self
            .chunk_id
            .evaluate(batch)?
            .into_array(batch.num_rows())?;
        let ids = ids
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                JammiError::Inference(format!(
                    "chunk id evaluated to {:?}, expected UInt64",
                    ids.data_type()
                ))
            })?
            .values();

        let mut complete = Vec::new();
        let mut start = 0;
        while start < ids.len() {
            let id = ids[start];
            if self.last_id.is_some_and(|last| id < last) {
                return Err(JammiError::Inference(format!(
                    "inference input is not in '{ORDINAL_COLUMN}' order: chunk {id} arrived \
                     after chunk {}",
                    self.last_id.unwrap_or(id)
                )));
            }
            self.last_id = Some(id);
            if self.open.as_ref().is_some_and(|open| open.id != id) {
                complete.extend(self.close()?);
            }
            let run = ids[start..].iter().take_while(|&&v| v == id).count();
            let open = self.open.get_or_insert_with(|| OpenChunk {
                id,
                parts: Vec::new(),
                rows: 0,
            });
            let take = run.min(self.capacity - open.rows);
            open.parts.push(batch.slice(start, take));
            open.rows += take;
            start += take;
            if open.rows == self.capacity {
                complete.extend(self.close()?);
            }
        }
        Ok(complete)
    }

    /// End of input: the last, possibly short, chunk.
    pub fn finish(&mut self) -> Result<Option<RecordBatch>> {
        self.close()
    }

    fn close(&mut self) -> Result<Option<RecordBatch>> {
        self.open
            .take()
            .map(|open| match <[RecordBatch; 1]>::try_from(open.parts) {
                Ok([whole]) => Ok(whole),
                Err(parts) => arrow::compute::concat_batches(&self.schema, &parts)
                    .map_err(|e| JammiError::Inference(format!("assembling a chunk: {e}"))),
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::Field;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new(ORDINAL_COLUMN, DataType::UInt64, false),
        ]))
    }

    /// Rows `ordinals`, each carrying text derived from its ordinal so a chunk
    /// holding the wrong rows is visible in the content, not only the ordinal.
    fn rows(ordinals: &[u64]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from_iter_values(ordinals.iter().map(|o| {
                    format!("passage {o} {}", "word ".repeat((o % 7) as usize))
                }))),
                Arc::new(UInt64Array::from(ordinals.to_vec())),
            ],
        )
        .unwrap()
    }

    /// `ordinals` cut into batches at `cuts` (ascending row offsets).
    fn rebatch(ordinals: &[u64], cuts: &[usize]) -> Vec<RecordBatch> {
        let bounds: Vec<usize> = std::iter::once(0)
            .chain(cuts.iter().copied())
            .chain(std::iter::once(ordinals.len()))
            .collect();
        bounds
            .windows(2)
            .filter(|w| w[1] > w[0])
            .map(|w| rows(&ordinals[w[0]..w[1]]))
            .collect()
    }

    /// The chunk sequence as `(ordinals, texts)` per chunk.
    fn chunks(batches: &[RecordBatch], batch_size: usize) -> Vec<(Vec<u64>, Vec<String>)> {
        let mut assembler =
            ChunkAssembler::try_new(schema(), NonZeroUsize::new(batch_size).unwrap()).unwrap();
        let mut out: Vec<RecordBatch> = Vec::new();
        for b in batches {
            out.extend(assembler.push(b).unwrap());
        }
        out.extend(assembler.finish().unwrap());
        out.iter()
            .map(|c| {
                let text = c.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                let ord = c.column(1).as_any().downcast_ref::<UInt64Array>().unwrap();
                (
                    ord.values().to_vec(),
                    text.iter().map(|t| t.unwrap().to_string()).collect(),
                )
            })
            .collect()
    }

    /// A small deterministic generator, so the re-batchings are random in
    /// shape but identical on every run.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, bound: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as usize) % bound
        }
    }

    fn random_cuts(rng: &mut Lcg, len: usize) -> Vec<usize> {
        let mut cuts: Vec<usize> = (0..rng.next(40)).map(|_| rng.next(len)).collect();
        cuts.sort_unstable();
        cuts.dedup();
        cuts
    }

    /// The same rows under every re-batching — one batch, a batch per row,
    /// and random cuts — yield the identical chunk sequence, and that
    /// sequence is `_ordinal / batch_size` with no chunk over `batch_size`.
    #[test]
    fn the_chunk_sequence_is_invariant_under_rebatching() {
        let ordinals: Vec<u64> = (0..1000).collect();
        for batch_size in [1usize, 7, 32, 1000, 4096] {
            let expected: Vec<Vec<u64>> =
                ordinals.chunks(batch_size).map(<[u64]>::to_vec).collect();
            let whole = chunks(&rebatch(&ordinals, &[]), batch_size);
            assert_eq!(
                whole.iter().map(|c| c.0.clone()).collect::<Vec<_>>(),
                expected,
                "batch_size={batch_size}: chunks are _ordinal / batch_size"
            );
            let per_row: Vec<usize> = (1..ordinals.len()).collect();
            assert_eq!(chunks(&rebatch(&ordinals, &per_row), batch_size), whole);
            let mut rng = Lcg(0x5eed ^ batch_size as u64);
            for _ in 0..50 {
                let cuts = random_cuts(&mut rng, ordinals.len());
                assert_eq!(
                    chunks(&rebatch(&ordinals, &cuts), batch_size),
                    whole,
                    "batch_size={batch_size} cuts={cuts:?}"
                );
            }
        }
    }

    /// One partition of a hash exchange keyed on the chunk id sees a
    /// subsequence of whole chunks. It assembles exactly those chunks, each
    /// identical to the chunk the unpartitioned stream forms, under every
    /// re-batching.
    #[test]
    fn a_partitions_share_assembles_the_same_chunks_as_the_whole() {
        let batch_size = 16usize;
        let ordinals: Vec<u64> = (0..500).collect();
        let whole = chunks(&rebatch(&ordinals, &[]), batch_size);
        for (partitions, partition) in [(2u64, 0u64), (2, 1), (4, 3)] {
            let share: Vec<u64> = ordinals
                .iter()
                .copied()
                .filter(|o| (o / batch_size as u64) % partitions == partition)
                .collect();
            let expected: Vec<_> = whole
                .iter()
                .filter(|c| (c.0[0] / batch_size as u64) % partitions == partition)
                .cloned()
                .collect();
            let mut rng = Lcg(partition + 1);
            for _ in 0..20 {
                let cuts = random_cuts(&mut rng, share.len());
                assert_eq!(chunks(&rebatch(&share, &cuts), batch_size), expected);
            }
        }
    }

    /// Rows out of `_ordinal` order could divide a chunk between two
    /// forwards without any error downstream, so the assembler refuses them.
    #[test]
    fn rows_out_of_ordinal_order_are_refused() {
        let mut assembler =
            ChunkAssembler::try_new(schema(), NonZeroUsize::new(4).unwrap()).unwrap();
        assembler.push(&rows(&[8, 9, 10])).unwrap();
        let err = assembler
            .push(&rows(&[2, 3]))
            .expect_err("a lower chunk after a higher one must refuse");
        assert!(err.to_string().contains("order"), "{err}");
    }

    /// The chunk expression refuses a schema whose `_ordinal` is absent,
    /// nullable, or not `UInt64`.
    #[test]
    fn the_chunk_expression_requires_a_numbered_schema() {
        let size = NonZeroUsize::new(8).unwrap();
        let none = Schema::new(vec![Field::new("text", DataType::Utf8, false)]);
        assert!(chunk_expr(&none, size).is_err());
        let nullable = Schema::new(vec![Field::new(ORDINAL_COLUMN, DataType::UInt64, true)]);
        assert!(chunk_expr(&nullable, size).is_err());
        let signed = Schema::new(vec![Field::new(ORDINAL_COLUMN, DataType::Int64, false)]);
        assert!(chunk_expr(&signed, size).is_err());
        assert!(chunk_expr(schema().as_ref(), size).is_ok());
    }
}
