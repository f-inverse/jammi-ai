//! Forward chunks: the rows one model forward takes, decided once, below
//! every exchange.
//!
//! An embedding's bytes depend on the composition of the batch it was
//! forwarded in (the padded width; the GEMM fold order), so the rows that
//! share one forward must be decided by the data alone — never by how
//! batches happen to arrive, which differs with the partition count, with a
//! re-batching exchange, and across a process boundary. The decision is made
//! in one place, the numbered input (`operator::numbered_input_exec`): rows
//! are ordered, and one pass of a [`ChunkCutter`] over their costs assigns
//! each its [`CHUNK_COLUMN`]. Rows with equal chunk id are forwarded
//! together; nothing else is.
//!
//! The rows leave the numbered input in chunk order, so a stream in that
//! order carries whole chunks as runs. `_chunk` is the hash key of the
//! exchange that fans a plan out (`operator::inference_exec`), so a chunk is
//! never divided between partitions, and [`ChunkAssembler`] regroups a
//! partition's rows into whole chunks whatever batch boundaries they arrived
//! with.
//!
//! The chunk's width is a rung of the model's
//! [`ShapeLadder`](jammi_numerics::ShapeLadder): the cutter budgets
//! `rows × width` on it, and the forward pads to the same rung, so the
//! budget describes the tensor the device actually runs.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::physical_expr::expressions::col;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use jammi_db::error::{JammiError, Result};
pub use jammi_numerics::ChunkCutter;

use super::schema::ORDINAL_COLUMN;
use crate::operator::numbered_input_exec::ASCENDING;

/// The forward-chunk id: a non-null `UInt64`, non-decreasing in the order
/// the rows leave the numbered input, which assigns it; read by
/// `InferenceExec`.
pub const CHUNK_COLUMN: &str = "_chunk";

/// Refuse a schema that does not carry `_ordinal` and `_chunk` as `UInt64
/// NOT NULL` — the columns every model-facing input is numbered with.
pub fn require_numbered(schema: &Schema) -> DfResult<()> {
    for name in [ORDINAL_COLUMN, CHUNK_COLUMN] {
        let field = schema.field_with_name(name).map_err(|_| {
            DataFusionError::Plan(format!(
                "inference input has no '{name}' column: a model reads a numbered input"
            ))
        })?;
        if field.data_type() != &DataType::UInt64 || field.is_nullable() {
            return Err(DataFusionError::Plan(format!(
                "inference input column '{name}' must be UInt64 NOT NULL, found {:?}{}",
                field.data_type(),
                if field.is_nullable() { " NULL" } else { "" }
            )));
        }
    }
    Ok(())
}

/// The forward-chunk id of a row of `schema`: its `_chunk` column.
pub fn chunk_expr(schema: &Schema) -> DfResult<Arc<dyn PhysicalExpr>> {
    require_numbered(schema)?;
    col(CHUNK_COLUMN, schema)
}

/// The ordering `[_chunk ASC]` over `schema`: the order a model reads its
/// input in, whole chunks as runs.
pub fn chunk_ordering(schema: &Schema) -> DfResult<LexOrdering> {
    let chunk = chunk_expr(schema)?;
    LexOrdering::new([PhysicalSortExpr::new(chunk, ASCENDING)])
        .ok_or_else(|| DataFusionError::Internal("an ordering of one expression is empty".into()))
}

/// The rows of the chunk currently being gathered.
struct OpenChunk {
    id: u64,
    parts: Vec<RecordBatch>,
}

/// Regroups a stream of `_chunk`-ascending batches into whole forward
/// chunks: the maximal runs of rows sharing one `_chunk` value.
///
/// The chunk sequence it yields is a function of the rows alone, identical
/// under every re-batching of the same rows. A chunk closes when the chunk id
/// changes or at end of input. A row whose chunk id is lower than one already
/// seen means the input is not in chunk order — a chunk could then be
/// divided silently — and is refused.
pub struct ChunkAssembler {
    schema: SchemaRef,
    chunk_id: Arc<dyn PhysicalExpr>,
    open: Option<OpenChunk>,
    last_id: Option<u64>,
}

impl ChunkAssembler {
    /// An assembler over batches of `schema`, which must be numbered.
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        let chunk_id = chunk_expr(schema.as_ref())?;
        Ok(Self {
            schema,
            chunk_id,
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
                    "inference input is not in '{CHUNK_COLUMN}' order: chunk {id} arrived \
                     after chunk {}",
                    self.last_id.unwrap_or(id)
                )));
            }
            self.last_id = Some(id);
            if self.open.as_ref().is_some_and(|open| open.id != id) {
                complete.extend(self.close()?);
            }
            let run = ids[start..].iter().take_while(|&&v| v == id).count();
            self.open
                .get_or_insert_with(|| OpenChunk {
                    id,
                    parts: Vec::new(),
                })
                .parts
                .push(batch.slice(start, run));
            start += run;
        }
        Ok(complete)
    }

    /// End of input: the last chunk.
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
    use std::num::NonZeroUsize;

    use arrow::array::StringArray;
    use arrow::datatypes::Field;
    use jammi_numerics::{ChunkBudget, ShapeLadder};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("text", DataType::Utf8, false),
            Field::new(ORDINAL_COLUMN, DataType::UInt64, false),
            Field::new(CHUNK_COLUMN, DataType::UInt64, false),
        ]))
    }

    /// The cost of the row numbered `o`: a spread of lengths that is not
    /// monotone in the ordinal, so a chunk cut by cost differs from one cut
    /// by count.
    fn cost(o: u64) -> u32 {
        (o * 37 % 23) as u32 + 3
    }

    /// Chunk ids for rows `0..n` under `budget`, cut in ordinal order.
    fn chunk_ids(n: u64, budget: ChunkBudget) -> Vec<u64> {
        let mut cutter = ChunkCutter::new(budget, ShapeLadder::new(128));
        (0..n).map(|o| cutter.push(cost(o))).collect()
    }

    fn budget(rows: usize, tokens: usize) -> ChunkBudget {
        ChunkBudget {
            rows: NonZeroUsize::new(rows).unwrap(),
            tokens: NonZeroUsize::new(tokens).unwrap(),
        }
    }

    /// Rows `ordinals` with their chunk ids from `ids`, each carrying text
    /// derived from its ordinal so a chunk holding the wrong rows is visible
    /// in the content, not only the ordinal.
    fn rows(ordinals: &[u64], ids: &[u64]) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from_iter_values(ordinals.iter().map(|o| {
                    format!("passage {o} {}", "word ".repeat(cost(*o) as usize))
                }))),
                Arc::new(UInt64Array::from(ordinals.to_vec())),
                Arc::new(UInt64Array::from(
                    ordinals
                        .iter()
                        .map(|&o| ids[o as usize])
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    }

    /// `ordinals` cut into batches at `cuts` (ascending row offsets).
    fn rebatch(ordinals: &[u64], ids: &[u64], cuts: &[usize]) -> Vec<RecordBatch> {
        let bounds: Vec<usize> = std::iter::once(0)
            .chain(cuts.iter().copied())
            .chain(std::iter::once(ordinals.len()))
            .collect();
        bounds
            .windows(2)
            .filter(|w| w[1] > w[0])
            .map(|w| rows(&ordinals[w[0]..w[1]], ids))
            .collect()
    }

    /// The chunk sequence as `(ordinals, texts)` per chunk.
    fn chunks(batches: &[RecordBatch]) -> Vec<(Vec<u64>, Vec<String>)> {
        let mut assembler = ChunkAssembler::try_new(schema()).unwrap();
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
    /// and random cuts — yield the identical chunk sequence: the maximal
    /// runs of one `_chunk` value, every chunk within its budget.
    #[test]
    fn the_chunk_sequence_is_invariant_under_rebatching() {
        let ordinals: Vec<u64> = (0..1000).collect();
        for (rows, tokens) in [(1usize, 4096usize), (7, 4096), (32, 256), (1000, 4096)] {
            let ids = chunk_ids(1000, budget(rows, tokens));
            let expected: Vec<Vec<u64>> = ids
                .chunk_by(|a, b| a == b)
                .scan(0u64, |next, run| {
                    let start = *next;
                    *next += run.len() as u64;
                    Some((start..*next).collect())
                })
                .collect();
            let whole = chunks(&rebatch(&ordinals, &ids, &[]));
            assert_eq!(
                whole.iter().map(|c| c.0.clone()).collect::<Vec<_>>(),
                expected,
                "rows={rows} tokens={tokens}: chunks are the runs of _chunk"
            );
            for chunk in &whole {
                assert!(chunk.0.len() <= rows);
                let longest = chunk.0.iter().map(|&o| cost(o) as usize).max().unwrap();
                let width = ShapeLadder::new(128).width(longest);
                assert!(chunk.0.len() == 1 || chunk.0.len() * width <= tokens);
            }
            let per_row: Vec<usize> = (1..ordinals.len()).collect();
            assert_eq!(chunks(&rebatch(&ordinals, &ids, &per_row)), whole);
            let mut rng = Lcg(0x5eed ^ rows as u64);
            for _ in 0..50 {
                let cuts = random_cuts(&mut rng, ordinals.len());
                assert_eq!(
                    chunks(&rebatch(&ordinals, &ids, &cuts)),
                    whole,
                    "rows={rows} tokens={tokens} cuts={cuts:?}"
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
        let ordinals: Vec<u64> = (0..500).collect();
        let ids = chunk_ids(500, budget(16, 512));
        let whole = chunks(&rebatch(&ordinals, &ids, &[]));
        for (partitions, partition) in [(2u64, 0u64), (2, 1), (4, 3)] {
            let share: Vec<u64> = ordinals
                .iter()
                .copied()
                .filter(|&o| ids[o as usize] % partitions == partition)
                .collect();
            let expected: Vec<_> = whole
                .iter()
                .filter(|c| ids[c.0[0] as usize] % partitions == partition)
                .cloned()
                .collect();
            let mut rng = Lcg(partition + 1);
            for _ in 0..20 {
                let cuts = random_cuts(&mut rng, share.len());
                assert_eq!(chunks(&rebatch(&share, &ids, &cuts)), expected);
            }
        }
    }

    /// Rows out of chunk order could divide a chunk between two forwards
    /// without any error downstream, so the assembler refuses them.
    #[test]
    fn rows_out_of_chunk_order_are_refused() {
        let ids = chunk_ids(20, budget(4, 4096));
        let mut assembler = ChunkAssembler::try_new(schema()).unwrap();
        assembler.push(&rows(&[8, 9, 10], &ids)).unwrap();
        let err = assembler
            .push(&rows(&[2, 3], &ids))
            .expect_err("a lower chunk after a higher one must refuse");
        assert!(err.to_string().contains("order"), "{err}");
    }

    /// The chunk expression refuses a schema whose `_ordinal` or `_chunk`
    /// is absent, nullable, or not `UInt64`.
    #[test]
    fn the_chunk_expression_requires_a_numbered_schema() {
        let text = Field::new("text", DataType::Utf8, false);
        let ordinal = Field::new(ORDINAL_COLUMN, DataType::UInt64, false);
        let none = Schema::new(vec![text.clone()]);
        assert!(chunk_expr(&none).is_err());
        let no_chunk = Schema::new(vec![text.clone(), ordinal.clone()]);
        assert!(chunk_expr(&no_chunk).is_err());
        let nullable = Schema::new(vec![
            ordinal.clone(),
            Field::new(CHUNK_COLUMN, DataType::UInt64, true),
        ]);
        assert!(chunk_expr(&nullable).is_err());
        let signed = Schema::new(vec![
            ordinal,
            Field::new(CHUNK_COLUMN, DataType::Int64, false),
        ]);
        assert!(chunk_expr(&signed).is_err());
        assert!(chunk_expr(schema().as_ref()).is_ok());
    }
}
