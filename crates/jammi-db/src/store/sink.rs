//! The result-table sink: the one writer every result-table materialization
//! streams its rows through.
//!
//! [`ResultStore::write_result_table`] and
//! [`ResultStore::write_version_fragment`] execute a plan and write its rows
//! as one result-table object under the caller's leased catalog row — the
//! table's Parquet, or a version's fragment — with the ANN segments and
//! checkpoints its [`SinkKind`] calls for, and report a [`SinkSummary`].
//!
//! Lifecycle is separate from the write, as types. The caller owns the
//! catalog row through its [`BuildingTable`] (or [`BuildingVersion`]) from
//! `create_table` to `finish`; the write borrows it, checks `is_live` at
//! every batch, and on failure aborts it in place — the row fails under the
//! caller's writer id and the bytes the write put under it are deleted — so
//! the caller's later drop is a no-op and a failed write leaves nothing a
//! reconcile must reap.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt32Array,
};
use arrow::datatypes::DataType;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{execute_stream, ExecutionPlan};
use futures::StreamExt;
use tracing::Instrument;

use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::segment::SegmentId;
use crate::index::sidecar::SidecarIndex;
use crate::storage::{ObjectParquetWriter, StorageUrl};
use crate::store::manifest::MaterializationEnv;
use crate::store::segment_builder::SegmentBuilder;
use crate::store::{BuildingTable, BuildingVersion, ResultStore};

/// The `tracing` target of the one event a finished write emits: where its
/// wall time went, as the nanosecond fields `SinkPhases` names. A
/// measurement harness subscribes to this target; nothing in the engine
/// reads it.
pub const SINK_PHASES_TARGET: &str = "jammi_db::store::sink::phases";

/// Where one sink write's wall time went. The wall phases are disjoint and
/// sequential on the writing task, so they sum to the write less its lease
/// and checkpoint bookkeeping; `index_build` is thread time on the segment
/// builders, overlapping the write and each other, and is not a slice of
/// that wall.
#[derive(Debug, Default, Clone, Copy)]
struct SinkPhases {
    /// Awaiting the plan's batches — everything beneath the sink.
    input: std::time::Duration,
    /// Filtering a batch to its ok rows and copying their vectors out.
    extract: std::time::Duration,
    /// Encoding and writing the object's Parquet, its close included.
    parquet: std::time::Duration,
    /// Handing vectors to the segment builder, which waits for a build slot
    /// once every one is taken, and awaiting the last builds after the rows
    /// have ended.
    index_wait: std::time::Duration,
    /// Persisting the built segments under the lease.
    segment: std::time::Duration,
    /// The summed thread time of the segment builds.
    index_build: std::time::Duration,
}

/// Run `work`, adding its wall time to `phase`.
async fn timed<T>(
    phase: &mut std::time::Duration,
    work: impl std::future::Future<Output = T>,
) -> T {
    let start = std::time::Instant::now();
    let out = work.await;
    *phase += start.elapsed();
    out
}

/// What the sink writes, and what its plan's rows are: the ANN-indexed,
/// ok-filtered embedding rows of an inference, every row as it is, or a
/// training set's rows in their committed order.
#[derive(Debug, Clone, PartialEq)]
pub enum SinkKind {
    /// An inference's output in the embedding table schema: only the rows
    /// whose `_status` is `ok` are written, their vectors built into ANN
    /// segments of `segment_rows` consecutive rows
    /// ([`crate::store::segment_builder`]), each appended under the lease;
    /// the table row's checkpoint is set every `checkpoint_interval` batches
    /// (`0`: never — a version row records no progress either way).
    Embeddings {
        /// The embedding width the schema and the index are built at.
        dimensions: usize,
        /// The HNSW knobs the segments are built with.
        ann: AnnIndexConfig,
        /// Rows per ANN segment.
        segment_rows: NonZeroUsize,
        /// Batches between two checkpoints on the table row; `0` never.
        checkpoint_interval: usize,
    },
    /// Every row of the plan, in the plan's own schema.
    Rows,
    /// Every row of the plan, asserted to arrive in the training set's
    /// committed order over `columns` ([`crate::store::TrainingSetSpec`]),
    /// and refused typed ([`JammiError::EmptyTrainingSet`], naming
    /// `source_query`) when the plan yields no row at all.
    TrainingSet {
        /// The order key the producer committed to.
        columns: Vec<String>,
        /// What the empty refusal names: the SQL text, or the source of a
        /// batch-fed input.
        source_query: String,
    },
}

/// The environment a process produces a materialization's bytes in: its
/// compute device, and the identity of every model the plan runs, so a
/// table records what produced it. Asked once the plan has run, when every
/// model it names is loaded.
#[async_trait::async_trait]
pub trait ProducingEnvironment: Send + Sync {
    async fn of(&self, plan: &Arc<dyn ExecutionPlan>) -> Result<MaterializationEnv>;
}

/// A process that runs no model: whatever plan it produces, its environment
/// is the model-free one.
#[derive(Debug, Clone, Copy)]
pub struct ModelFreeEnvironment;

#[async_trait::async_trait]
impl ProducingEnvironment for ModelFreeEnvironment {
    async fn of(&self, _plan: &Arc<dyn ExecutionPlan>) -> Result<MaterializationEnv> {
        Ok(MaterializationEnv::without_models())
    }
}

/// What one write reports to its caller: the rows the plan produced, the
/// rows written, the segments appended, and the environment that produced
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkSummary {
    /// Rows the plan produced, before any filter.
    pub input_rows: u64,
    /// Rows written to the object.
    pub rows: u64,
    /// The ANN segments appended under the lease, in the order of the rows
    /// they hold: empty unless a row realized under an embedding kind.
    pub segments: Vec<SegmentId>,
    /// The environment that ran the plan — what the table's manifest
    /// records.
    pub env: MaterializationEnv,
}

/// The leased row a write streams under: the table row or the version row.
/// Two rows, one lifecycle — the arms differ only where the rows do (a
/// table row checkpoints; a version row records no progress, its delta is
/// retried whole).
enum SinkRow<'a> {
    /// The `result_tables` row: the table's own Parquet is written.
    Table(&'a mut BuildingTable),
    /// A `result_table_versions` row: the version's fragment is written.
    Version(&'a mut BuildingVersion),
}

impl SinkRow<'_> {
    fn table_name(&self) -> &str {
        match self {
            Self::Table(t) => t.table_name(),
            Self::Version(v) => v.table_name(),
        }
    }

    /// The object the write produces: the table's Parquet, or the version's
    /// fragment.
    fn object_url(&self) -> Result<StorageUrl> {
        match self {
            Self::Table(t) => Ok(t.parquet_url().clone()),
            Self::Version(v) => v.fragment_url(),
        }
    }

    /// The precision the row was stamped with; every segment must be built
    /// at it.
    fn storage_precision(&self) -> StoragePrecision {
        match self {
            Self::Table(t) => t.storage_precision(),
            Self::Version(v) => v.storage_precision(),
        }
    }

    /// `true` while this process's keeper still owns the row.
    fn is_live(&self) -> bool {
        match self {
            Self::Table(t) => t.is_live(),
            Self::Version(v) => v.is_live(),
        }
    }

    /// Record that `batch` batches have landed: the table row's checkpoint,
    /// every `interval` batches (`0` never); nothing for a version row.
    async fn after_batch(&self, batch: usize, interval: usize) -> Result<()> {
        match self {
            Self::Table(t) if interval > 0 && batch.is_multiple_of(interval) => {
                t.set_checkpoint(batch).await
            }
            Self::Table(_) | Self::Version(_) => Ok(()),
        }
    }

    /// Persist `index` as a new segment under this row's lease.
    async fn append_segment(&self, index: &SidecarIndex) -> Result<SegmentId> {
        match self {
            Self::Table(t) => t.append_segment(index).await,
            Self::Version(v) => v.append_segment(index).await,
        }
    }

    /// The write failed: fail the row under its writer and delete what the
    /// write put under it.
    async fn abort(&mut self) -> Result<()> {
        match self {
            Self::Table(t) => t.abort_in_place().await,
            Self::Version(v) => v.abort_in_place().await,
        }
    }
}

/// The byte writer beneath the node: rows to the object's Parquet and, for
/// an embedding kind, ok rows only, their vectors into the segment builder
/// as each batch arrives.
struct ResultSink {
    writer: ObjectParquetWriter,
    segments: Option<SegmentBuilder>,
    input_rows: u64,
    phases: SinkPhases,
}

impl ResultSink {
    async fn write_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.input_rows += batch.num_rows() as u64;
        match &mut self.segments {
            Some(segments) => {
                let extract = std::time::Instant::now();
                let (ok_batch, row_ids, vectors) = filter_ok_and_extract_vectors(batch)?;
                self.phases.extract += extract.elapsed();
                if ok_batch.num_rows() > 0 {
                    timed(&mut self.phases.parquet, self.writer.write_batch(&ok_batch)).await?;
                    timed(&mut self.phases.index_wait, segments.push(row_ids, vectors)).await?;
                }
            }
            None => timed(&mut self.phases.parquet, self.writer.write_batch(batch)).await?,
        }
        Ok(())
    }

    /// Close the object and await the segment builds:
    /// `(input_rows, rows, segments, phases)`.
    async fn finalize(mut self) -> Result<(u64, u64, Vec<SidecarIndex>, SinkPhases)> {
        let rows = timed(&mut self.phases.parquet, self.writer.close()).await? as u64;
        let segments = match self.segments {
            Some(builder) => {
                let (segments, build_time) =
                    timed(&mut self.phases.index_wait, builder.finish()).await?;
                self.phases.index_build = build_time;
                segments
            }
            None => Vec::new(),
        };
        Ok((self.input_rows, rows, segments, self.phases))
    }
}

/// Filter an inference batch to its `ok` rows in the embedding table schema,
/// returning the batch with the `_row_id`s and vectors it kept.
///
/// Input schema: `_row_id, _ordinal, _source, _model, _status, _error,
/// _latency_ms, vector`, plus `_content_hash` when the plan passed it
/// through. Output schema: `_row_id, _source_id, _model_id, vector,
/// _content_hash` (no `_ordinal` — an embedding table's `_row_id` is unique
/// by construction; the hash is NULL when the input carried none).
pub fn filter_ok_and_extract_vectors(
    batch: &RecordBatch,
) -> Result<(RecordBatch, Vec<String>, Vec<Vec<f32>>)> {
    let status = batch
        .column_by_name("_status")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| JammiError::Inference("Missing _status column".into()))?;
    let keep: Vec<u32> = (0..status.len())
        .filter(|&i| status.value(i) == "ok")
        .map(|i| i as u32)
        .collect();
    let indices = UInt32Array::from(keep);

    let take = |name: &str| -> Result<arrow::array::ArrayRef> {
        let column = batch
            .column_by_name(name)
            .ok_or_else(|| JammiError::Inference(format!("Missing {name}")))?;
        arrow::compute::take(column.as_ref(), &indices, None)
            .map_err(|e| JammiError::Other(format!("Arrow take: {e}")))
    };
    let filtered_row_ids = take("_row_id")?;
    let filtered_source = take("_source")?;
    let filtered_model = take("_model")?;
    let filtered_vector = take("vector")?;
    let filtered_hash: arrow::array::ArrayRef =
        match batch.column_by_name(crate::store::schema::CONTENT_HASH_COLUMN) {
            Some(_) => {
                let taken = take(crate::store::schema::CONTENT_HASH_COLUMN)?;
                arrow::compute::cast(&taken, &DataType::Utf8)
                    .map_err(|e| JammiError::Other(format!("Arrow cast _content_hash: {e}")))?
            }
            None => crate::store::content_hash::null_hash_column(indices.len()),
        };

    let dims = filtered_vector
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .map(|fsl| fsl.value_length() as usize)
        .ok_or_else(|| JammiError::Inference("vector is not a FixedSizeList".into()))?;
    let ok_batch = RecordBatch::try_new(
        crate::store::schema::embedding_table_schema(dims),
        vec![
            Arc::clone(&filtered_row_ids),
            filtered_source,
            filtered_model,
            Arc::clone(&filtered_vector),
            filtered_hash,
        ],
    )
    .map_err(|e| JammiError::Other(format!("RecordBatch build: {e}")))?;

    let row_ids: Vec<String> = filtered_row_ids
        .as_any()
        .downcast_ref::<StringArray>()
        .map(|a| {
            a.iter()
                .map(|s| s.unwrap_or_default().to_string())
                .collect()
        })
        .ok_or_else(|| JammiError::Inference("_row_id is not Utf8".into()))?;
    let fsl = filtered_vector
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| JammiError::Inference("vector is not a FixedSizeList".into()))?;
    let vectors = (0..fsl.len())
        .map(|i| {
            let v = fsl.value(i);
            let a = v.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                JammiError::Inference("Expected Float32Array in vector column".into())
            })?;
            Ok(a.values().to_vec())
        })
        .collect::<Result<Vec<Vec<f32>>>>()?;
    Ok((ok_batch, row_ids, vectors))
}

/// Stream `input`'s rows into `row`'s object, then append the segments the
/// kind built — or abort the row in place on any failure.
async fn write(
    mut row: SinkRow<'_>,
    kind: &SinkKind,
    input: Arc<dyn ExecutionPlan>,
    store: &ResultStore,
    context: Arc<TaskContext>,
) -> Result<SinkSummary> {
    match write_under(&row, kind, input, store, context).await {
        Ok(summary) => Ok(summary),
        Err(e) => {
            if let Err(failed) = row.abort().await {
                tracing::warn!(
                    table = %row.table_name(),
                    error = %failed,
                    "result table sink: the failed write's row could not be failed; left for the \
                     lease reclaim"
                );
            }
            Err(e)
        }
    }
}

async fn write_under(
    row: &SinkRow<'_>,
    kind: &SinkKind,
    input: Arc<dyn ExecutionPlan>,
    store: &ResultStore,
    context: Arc<TaskContext>,
) -> Result<SinkSummary> {
    let url = row.object_url()?;
    let schema = match kind {
        SinkKind::Embeddings { dimensions, .. } => {
            crate::store::schema::embedding_table_schema(*dimensions)
        }
        SinkKind::Rows | SinkKind::TrainingSet { .. } => input.schema(),
    };
    let writer = store
        .open_writer(&url, schema)
        .instrument(tracing::debug_span!("sink.open_writer"))
        .await?;
    let (segments, checkpoint_interval) = match kind {
        SinkKind::Embeddings {
            dimensions,
            ann,
            segment_rows,
            checkpoint_interval,
        } => (
            Some(SegmentBuilder::new(
                *dimensions,
                *ann,
                row.storage_precision(),
                *segment_rows,
            )),
            *checkpoint_interval,
        ),
        SinkKind::Rows | SinkKind::TrainingSet { .. } => (None, 0),
    };
    let mut sink = ResultSink {
        writer,
        segments,
        input_rows: 0,
        phases: SinkPhases::default(),
    };
    let mut rows = execute_stream(Arc::clone(&input), context)?;
    if let SinkKind::TrainingSet { columns, .. } = kind {
        rows = crate::store::assert_batches_are_ordinal_sorted(rows, columns);
    }
    async {
        let mut batch_num = 0usize;
        while let Some(batch) = timed(&mut sink.phases.input, rows.next()).await {
            let batch = batch?;
            if !row.is_live() {
                return Err(JammiError::LeaseLost {
                    table: row.table_name().to_string(),
                });
            }
            batch_num += 1;
            sink.write_batch(&batch).await?;
            row.after_batch(batch_num, checkpoint_interval).await?;
        }
        Ok(())
    }
    .instrument(tracing::debug_span!("sink.rows"))
    .await?;
    let (input_rows, rows, built, mut phases) = sink
        .finalize()
        .instrument(tracing::debug_span!("sink.finalize"))
        .await?;
    if let (SinkKind::TrainingSet { source_query, .. }, 0) = (kind, rows) {
        return Err(JammiError::EmptyTrainingSet {
            source_query: source_query.clone(),
        });
    }
    let mut segments = Vec::with_capacity(built.len());
    for index in &built {
        segments.push(
            timed(
                &mut phases.segment,
                row.append_segment(index)
                    .instrument(tracing::debug_span!("sink.append_segment")),
            )
            .await?,
        );
    }
    tracing::info!(
        target: SINK_PHASES_TARGET,
        table = %row.table_name(),
        rows,
        input_ns = phases.input.as_nanos() as u64,
        extract_ns = phases.extract.as_nanos() as u64,
        parquet_ns = phases.parquet.as_nanos() as u64,
        index_wait_ns = phases.index_wait.as_nanos() as u64,
        segment_ns = phases.segment.as_nanos() as u64,
        index_build_ns = phases.index_build.as_nanos() as u64,
        "result table sink: phases"
    );
    Ok(SinkSummary {
        input_rows,
        rows,
        segments,
        env: store.producing_environment().of(&input).await?,
    })
}

impl ResultStore {
    /// Materialize `plan`'s rows under `building`'s row — the one call every
    /// producer of a result table makes between `create_table` and
    /// `finish`: the plan is executed under `context` and its rows written
    /// as the table's Parquet. A failed write has already aborted the row
    /// when this returns the error.
    pub async fn write_result_table(
        &self,
        building: &mut BuildingTable,
        kind: SinkKind,
        plan: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<SinkSummary> {
        write(SinkRow::Table(building), &kind, plan, self, context).await
    }

    /// [`Self::write_result_table`] for a version's fragment under
    /// `version`'s row.
    pub async fn write_version_fragment(
        &self,
        version: &mut BuildingVersion,
        kind: SinkKind,
        plan: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<SinkSummary> {
        write(SinkRow::Version(version), &kind, plan, self, context).await
    }
}
