//! The result-table sink: the one physical node every result-table
//! materialization roots in, carried by the compute plane.
//!
//! [`ResultTableSinkExec`] writes its child's rows as one result-table
//! object under a leased catalog row — the table's Parquet, or a version's
//! fragment — with the ANN segments and checkpoints its [`SinkKind`] calls
//! for, and emits ONE summary batch ([`SinkSummary`]) to whoever executed
//! it. Where it writes is decided when it is first polled: a node executed
//! under a context whose session carries a [`ComputePlane`] that holds the
//! plan is submitted to the plane WHOLE — the sink, the compute beneath it
//! and the write — and the summary streams back; the codec marks the node
//! `placed` as it crosses the wire, and a placed node writes, never
//! re-submits. With no plane, or a plane that cannot hold the plan, the
//! same node writes in this process under the same lifecycle: one execution
//! path, two locations.
//!
//! Lifecycle is separate from the write, as types. The submitter owns the
//! catalog row through its [`BuildingTable`] (or [`BuildingVersion`]) from
//! `create_table` to `finish`; the sink takes the row for the duration of
//! the write — [`SinkLease::take`], the transfer CAS from the submitter's
//! writer id to the writing process's own, under which the writing process's
//! own keeper renews it and every batch checks `is_live` — hands it back on
//! success ([`SinkLease::hand_back`], the transfer CAS in reverse) and fails
//! it under its own id on failure ([`SinkLease::fail`]). A writing process
//! killed mid-write leaves the row `building` under ITS writer id: its lease
//! expires and the lease reclaim retires the row exactly as it would a dead
//! writer's, while the submitter's stream errors. A second launch of the
//! same placed sink fails the transfer CAS typed — the row already moved.
//! In-process the transfer is from a writer to itself: a renewal.

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Int64Array, ListArray, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    execute_stream, DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::catalog::result_repo::ResultTableCas;
use crate::catalog::version_repo::VersionCas;
use crate::compute_plane::{ComputePlane, ComputePlaneSlot};
use crate::config::{AnnIndexConfig, StoragePrecision};
use crate::error::{JammiError, Result};
use crate::index::segment::SegmentId;
use crate::index::sidecar::SidecarIndex;
use crate::storage::{ObjectParquetWriter, StorageError, StorageUrl};
use crate::store::segment_builder::SegmentBuilder;
use crate::store::{layout, BuildingTable, BuildingVersion, ResultStore};
use crate::tenant::TenantId;

/// The line the sink logs, on the process that writes, as it starts a
/// table's write — the determinant a distributed oracle greps an executor's
/// log for. Carries `table = <name>` as a field.
pub const SINK_WRITE_LOG: &str = "result table sink: writing";

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
    /// Awaiting the child plan's batches — everything beneath the sink.
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

/// What the sink writes, and what its child's rows are: the ANN-indexed,
/// ok-filtered embedding rows of an inference, every row as it is, or a
/// training set's rows in their committed order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
        /// The HNSW knobs the segments are built with — the submitter's, so
        /// a placed write builds the index the in-process one would.
        ann: AnnIndexConfig,
        /// Rows per ANN segment — the submitter's, for the same reason.
        segment_rows: NonZeroUsize,
        /// Batches between two checkpoints on the table row; `0` never.
        checkpoint_interval: usize,
    },
    /// Every row of the child, in the child's own schema.
    Rows,
    /// Every row of the child, asserted to arrive in the training set's
    /// committed order over `columns` ([`crate::store::TrainingSetSpec`]),
    /// and refused typed ([`JammiError::EmptyTrainingSet`], naming
    /// `source_query`) when the child yields no row at all.
    TrainingSet {
        /// The order key the producer committed to.
        columns: Vec<String>,
        /// What the empty refusal names: the SQL text, or the source of a
        /// batch-fed input.
        source_query: String,
    },
}

/// Which leased row the sink writes under, and so which object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SinkLeaseKind {
    /// The `result_tables` row: the table's own Parquet is written.
    Table,
    /// A `result_table_versions` row: the version's fragment is written.
    Version {
        /// The version number the row and the fragment carry.
        version: i64,
    },
}

/// Everything the sink needs to write one result-table object under its
/// leased row, on whichever process executes it. Serialized as the node's
/// wire body; the store it writes through is the executing process's own.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResultTableSinkSpec {
    /// The `result_tables` primary key.
    pub table_name: String,
    /// The table's Parquet URL under the submitter's store root; a version
    /// sink writes the fragment derived from it.
    pub parquet_url: StorageUrl,
    /// The tenant the row belongs to (`None` for GLOBAL).
    pub tenant: Option<TenantId>,
    /// The submitter's writer id — the row's owner at submission, whom the
    /// lease is taken from and handed back to.
    pub writer_id: String,
    /// The precision the row was stamped with; every segment must be built
    /// at it.
    pub storage_precision: StoragePrecision,
    /// The leased row.
    pub lease: SinkLeaseKind,
    /// What is written.
    pub kind: SinkKind,
}

impl ResultTableSinkSpec {
    /// The object this sink writes: the table's Parquet, or the version's
    /// fragment.
    pub fn object_url(&self) -> Result<StorageUrl> {
        match self.lease {
            SinkLeaseKind::Table => Ok(self.parquet_url.clone()),
            SinkLeaseKind::Version { version } => {
                layout::version_fragment_url(&self.parquet_url, version)
            }
        }
    }

    /// The schema the object is written in: the embedding table schema at
    /// the kind's width, or the child's own.
    fn object_schema(&self, child: &SchemaRef) -> SchemaRef {
        match &self.kind {
            SinkKind::Embeddings { dimensions, .. } => {
                crate::store::schema::embedding_table_schema(*dimensions)
            }
            SinkKind::Rows | SinkKind::TrainingSet { .. } => Arc::clone(child),
        }
    }
}

/// What one sink reports back to its submitter: the rows its child
/// produced, the rows it wrote, and the segments it appended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SinkSummary {
    /// Rows the child produced, before any filter.
    pub input_rows: u64,
    /// Rows written to the object.
    pub rows: u64,
    /// The ANN segments appended under the lease, in the order of the rows
    /// they hold: empty unless a row realized under an embedding kind.
    pub segments: Vec<SegmentId>,
}

impl SinkSummary {
    /// The summary batch's schema: `input_rows`, `rows`, `segment_ids`.
    pub fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("input_rows", DataType::UInt64, false),
            Field::new("rows", DataType::UInt64, false),
            Field::new(
                "segment_ids",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, false))),
                false,
            ),
        ]))
    }

    fn to_batch(self) -> Result<RecordBatch> {
        let ids = ListArray::new(
            Arc::new(Field::new("item", DataType::Int64, false)),
            OffsetBuffer::from_lengths([self.segments.len()]),
            Arc::new(Int64Array::from_iter_values(
                self.segments.iter().map(|s| s.0),
            )),
            None,
        );
        RecordBatch::try_new(
            Self::schema(),
            vec![
                Arc::new(UInt64Array::from(vec![self.input_rows])),
                Arc::new(UInt64Array::from(vec![self.rows])),
                Arc::new(ids),
            ],
        )
        .map_err(|e| JammiError::Other(format!("sink summary batch: {e}")))
    }

    /// Read the one summary a sink's stream carried — exactly one row across
    /// `batches`, in [`Self::schema`].
    pub fn from_batches(batches: &[RecordBatch]) -> Result<Self> {
        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        let batch = batches.iter().find(|b| b.num_rows() > 0);
        let (Some(batch), 1) = (batch, rows) else {
            return Err(JammiError::Other(format!(
                "result table sink: expected exactly one summary row, got {rows}"
            )));
        };
        let column = |name: &str| {
            batch.column_by_name(name).ok_or_else(|| {
                JammiError::Other(format!("result table sink summary: missing column {name}"))
            })
        };
        let u64_at = |name: &str| -> Result<u64> {
            column(name)?
                .as_any()
                .downcast_ref::<UInt64Array>()
                .map(|a| a.value(0))
                .ok_or_else(|| {
                    JammiError::Other(format!("result table sink summary: {name} is not UInt64"))
                })
        };
        let ids = column("segment_ids")?
            .as_any()
            .downcast_ref::<ListArray>()
            .map(|list| list.value(0))
            .ok_or_else(|| {
                JammiError::Other("result table sink summary: segment_ids is not a list".into())
            })?;
        let ids = ids.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
            JammiError::Other("result table sink summary: segment_ids are not Int64".into())
        })?;
        Ok(Self {
            input_rows: u64_at("input_rows")?,
            rows: u64_at("rows")?,
            segments: ids.values().iter().map(|&id| SegmentId(id)).collect(),
        })
    }
}

/// The leased row the writing process holds for the duration of a sink's
/// write: the table row or the version row, taken from the submitter and
/// handed back. Two rows, one lifecycle — the arms differ only where the
/// rows do (a table row checkpoints; a version row records no progress,
/// its delta is retried whole).
pub enum SinkLease {
    /// The `result_tables` row.
    Table(BuildingTable),
    /// The `result_table_versions` row.
    Version(BuildingVersion),
}

impl SinkLease {
    /// Take `spec`'s row from its submitter for `store`'s writer: the
    /// transfer CAS from `spec.writer_id` to this writer (a renewal when
    /// the two are one), then this process's own handle over the row —
    /// keeper hold, `is_live` per batch — exactly as the row's creator
    /// holds it. A miss is the classified typed error: a second launch of
    /// the same placed sink reads [`JammiError::LeaseLost`].
    pub async fn take(store: &ResultStore, spec: &ResultTableSinkSpec) -> Result<Self> {
        let catalog = store.catalog();
        let lease = store.lease_intervals().lease();
        match spec.lease {
            SinkLeaseKind::Table => {
                let cas = ResultTableCas::writer(&spec.table_name, &spec.writer_id, spec.tenant);
                catalog
                    .transfer_building_lease(&cas, store.writer_id(), lease)
                    .await?;
                Ok(Self::Table(BuildingTable::adopt(
                    store.clone(),
                    spec.table_name.clone(),
                    spec.parquet_url.clone(),
                    spec.tenant,
                    store.writer_id().to_string(),
                    spec.storage_precision,
                )))
            }
            SinkLeaseKind::Version { version } => {
                let cas =
                    VersionCas::writer(&spec.table_name, version, &spec.writer_id, spec.tenant);
                catalog
                    .transfer_building_version_lease(&cas, store.writer_id(), lease)
                    .await?;
                let row = catalog
                    .get_result_table_version(&spec.table_name, version)
                    .await?
                    .ok_or_else(|| JammiError::RowGone {
                        table: spec.table_name.clone(),
                    })?;
                Ok(Self::Version(BuildingVersion::adopt(
                    store.clone(),
                    spec.table_name.clone(),
                    spec.parquet_url.clone(),
                    version,
                    row.parent_version,
                    StorageUrl::parse(&row.manifest_path)?,
                    spec.tenant,
                    store.writer_id().to_string(),
                    spec.storage_precision,
                )))
            }
        }
    }

    fn table_name(&self) -> &str {
        match self {
            Self::Table(t) => t.table_name(),
            Self::Version(v) => v.table_name(),
        }
    }

    /// `true` while this process's keeper still owns the row.
    pub fn is_live(&self) -> bool {
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

    /// Persist `index` as a new segment under this lease.
    pub async fn append_segment(&self, index: &SidecarIndex) -> Result<SegmentId> {
        match self {
            Self::Table(t) => t.append_segment(index).await,
            Self::Version(v) => v.append_segment(index).await,
        }
    }

    /// The write succeeded: hand the row back to `to_writer_id` and stop
    /// renewing.
    pub async fn hand_back(self, to_writer_id: &str) -> Result<()> {
        match self {
            Self::Table(t) => t.hand_back(to_writer_id).await,
            Self::Version(v) => v.hand_back(to_writer_id).await,
        }
    }

    /// The write failed: fail the row under this writer's own id and delete
    /// what this write put under it.
    pub async fn fail(self) -> Result<()> {
        match self {
            Self::Table(t) => t.abort().await,
            Self::Version(v) => v.abort().await,
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

/// The physical node every result-table materialization roots in — module
/// doc. One output partition carrying one [`SinkSummary`] batch.
pub struct ResultTableSinkExec {
    spec: ResultTableSinkSpec,
    input: Arc<dyn ExecutionPlan>,
    store: ResultStore,
    /// `true` once the node has crossed the wire: it writes here, never
    /// re-submits.
    placed: bool,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for ResultTableSinkExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResultTableSinkExec")
            .field("spec", &self.spec)
            .field("placed", &self.placed)
            .finish_non_exhaustive()
    }
}

impl ResultTableSinkExec {
    /// Root `input` in a sink writing `spec` through `store`, on the
    /// submitting process: where it writes is decided when it is polled.
    pub fn new(
        spec: ResultTableSinkSpec,
        input: Arc<dyn ExecutionPlan>,
        store: ResultStore,
    ) -> Self {
        Self::build(spec, input, store, false)
    }

    /// The node as it arrives on the process that will write: it writes
    /// here and never re-submits.
    pub fn placed(
        spec: ResultTableSinkSpec,
        input: Arc<dyn ExecutionPlan>,
        store: ResultStore,
    ) -> Self {
        Self::build(spec, input, store, true)
    }

    fn build(
        spec: ResultTableSinkSpec,
        input: Arc<dyn ExecutionPlan>,
        store: ResultStore,
        placed: bool,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(SinkSummary::schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Self {
            spec,
            input,
            store,
            placed,
            properties,
        }
    }

    /// What this sink writes, and where.
    pub fn spec(&self) -> &ResultTableSinkSpec {
        &self.spec
    }

    /// Whether this node has crossed the wire.
    pub fn is_placed(&self) -> bool {
        self.placed
    }

    /// Submit the whole plan — this node over its child — to `plane` when
    /// the plane holds it, or write here. The one decision the module doc
    /// describes; logged either way naming the root and the table.
    async fn run(
        spec: ResultTableSinkSpec,
        input: Arc<dyn ExecutionPlan>,
        store: ResultStore,
        placed: bool,
        plane: Option<Arc<dyn ComputePlane>>,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if let (false, Some(plane)) = (placed, plane) {
            let whole: Arc<dyn ExecutionPlan> =
                Arc::new(Self::new(spec.clone(), Arc::clone(&input), store.clone()));
            match plane.unheld(&whole).await? {
                None => {
                    let stream = plane.place(whole).await?;
                    tracing::info!(
                        root = "ResultTableSinkExec",
                        table = %spec.table_name,
                        "materialization placed on the compute plane"
                    );
                    return Ok(stream);
                }
                Some(why) => tracing::info!(
                    root = "ResultTableSinkExec",
                    table = %spec.table_name,
                    reason = %why,
                    "materialization runs in this process: the compute plane cannot hold it"
                ),
            }
        }
        let summary = write(&spec, input, &store, context).await?.to_batch()?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            SinkSummary::schema(),
            futures::stream::once(async move { Ok(summary) }),
        )))
    }
}

/// Write `spec` here: take the lease, stream the child through the byte
/// writer, append the segment, hand the lease back — or fail it.
async fn write(
    spec: &ResultTableSinkSpec,
    input: Arc<dyn ExecutionPlan>,
    store: &ResultStore,
    context: Arc<TaskContext>,
) -> Result<SinkSummary> {
    let lease = SinkLease::take(store, spec)
        .instrument(tracing::debug_span!("sink.take_lease"))
        .await?;
    tracing::info!(
        table = %spec.table_name,
        writer = %store.writer_id(),
        "{SINK_WRITE_LOG}"
    );
    match write_under(spec, input, store, context, &lease).await {
        Ok(summary) => {
            lease
                .hand_back(&spec.writer_id)
                .instrument(tracing::debug_span!("sink.hand_back"))
                .await?;
            Ok(summary)
        }
        Err(e) => {
            if let Err(failed) = lease.fail().await {
                tracing::warn!(
                    table = %spec.table_name,
                    error = %failed,
                    "result table sink: the failed write's row could not be failed under this \
                     writer; left for the lease reclaim"
                );
            }
            Err(e)
        }
    }
}

async fn write_under(
    spec: &ResultTableSinkSpec,
    input: Arc<dyn ExecutionPlan>,
    store: &ResultStore,
    context: Arc<TaskContext>,
    lease: &SinkLease,
) -> Result<SinkSummary> {
    let url = spec.object_url()?;
    let writer = store
        .open_writer(&url, spec.object_schema(&input.schema()))
        .instrument(tracing::debug_span!("sink.open_writer"))
        .await?;
    let (segments, checkpoint_interval) = match &spec.kind {
        SinkKind::Embeddings {
            dimensions,
            ann,
            segment_rows,
            checkpoint_interval,
        } => (
            Some(SegmentBuilder::new(
                *dimensions,
                *ann,
                spec.storage_precision,
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
    let mut rows = execute_stream(input, context)?;
    if let SinkKind::TrainingSet { columns, .. } = &spec.kind {
        rows = crate::store::assert_batches_are_ordinal_sorted(rows, columns);
    }
    async {
        let mut batch_num = 0usize;
        while let Some(batch) = timed(&mut sink.phases.input, rows.next()).await {
            let batch = batch?;
            if !lease.is_live() {
                return Err(JammiError::LeaseLost {
                    table: lease.table_name().to_string(),
                });
            }
            batch_num += 1;
            sink.write_batch(&batch).await?;
            lease.after_batch(batch_num, checkpoint_interval).await?;
        }
        Ok(())
    }
    .instrument(tracing::debug_span!("sink.rows"))
    .await?;
    let (input_rows, rows, built, mut phases) = sink
        .finalize()
        .instrument(tracing::debug_span!("sink.finalize"))
        .await?;
    if let (SinkKind::TrainingSet { source_query, .. }, 0) = (&spec.kind, rows) {
        return Err(JammiError::EmptyTrainingSet {
            source_query: source_query.clone(),
        });
    }
    let mut segments = Vec::with_capacity(built.len());
    for index in &built {
        segments.push(
            timed(
                &mut phases.segment,
                lease
                    .append_segment(index)
                    .instrument(tracing::debug_span!("sink.append_segment")),
            )
            .await?,
        );
    }
    tracing::info!(
        target: SINK_PHASES_TARGET,
        table = %spec.table_name,
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
    })
}

impl DisplayAs for ResultTableSinkExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ResultTableSinkExec: table={}, placed={}",
            self.spec.table_name, self.placed
        )
    }
}

impl ExecutionPlan for ResultTableSinkExec {
    fn name(&self) -> &str {
        "ResultTableSinkExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let [input] = <[Arc<dyn ExecutionPlan>; 1]>::try_from(children).map_err(|children| {
            DataFusionError::Internal(format!(
                "ResultTableSinkExec has exactly one child, got {}",
                children.len()
            ))
        })?;
        Ok(Arc::new(Self::build(
            self.spec.clone(),
            input,
            self.store.clone(),
            self.placed,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "ResultTableSinkExec has one output partition, partition {partition} was asked for"
            )));
        }
        let plane = context
            .session_config()
            .get_extension::<ComputePlaneSlot>()
            .and_then(|slot| slot.plane());
        let (spec, input, store, placed) = (
            self.spec.clone(),
            Arc::clone(&self.input),
            self.store.clone(),
            self.placed,
        );
        let stream = futures::stream::once(async move {
            Self::run(spec, input, store, placed, plane, context)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            SinkSummary::schema(),
            stream,
        )))
    }
}

impl ResultStore {
    /// Whether `url` lies under this store's root — the one object space a
    /// sink may write.
    pub fn holds_url(&self, url: &StorageUrl) -> bool {
        crate::store::reconcile::relative_to(&self.root, url).is_some()
    }

    /// Adopt a sink node that arrived from another process, to write
    /// through THIS store: refused typed when `spec`'s object is not under
    /// this store's root ([`StorageError::InvalidUrl`]) or its row is not
    /// one this catalog holds ([`JammiError::RowGone`]).
    pub async fn adopt_placed_sink(
        &self,
        spec: ResultTableSinkSpec,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<ResultTableSinkExec> {
        if !self.holds_url(&spec.parquet_url) {
            return Err(JammiError::Storage(StorageError::InvalidUrl {
                input: spec.parquet_url.as_str().to_string(),
                reason: format!(
                    "a result table sink writes only under this store's root {}",
                    self.root.as_str()
                ),
            }));
        }
        let known = match spec.lease {
            SinkLeaseKind::Table => self
                .catalog
                .get_result_table_for_tenant(&spec.table_name, spec.tenant)
                .await?
                .is_some(),
            SinkLeaseKind::Version { version } => self
                .catalog
                .get_result_table_version(&spec.table_name, version)
                .await?
                .is_some(),
        };
        if !known {
            return Err(JammiError::RowGone {
                table: spec.table_name,
            });
        }
        Ok(ResultTableSinkExec::placed(spec, input, self.clone()))
    }

    /// Materialize `plan`'s rows through the sink under `building`'s row —
    /// the one call every producer of a result table makes between
    /// `create_table` and `finish`: the sink is rooted over `plan`,
    /// executed under `context` (where the compute plane says), and its
    /// summary read back; the row's hold is then taken afresh, the loan
    /// being over.
    pub async fn write_result_table(
        &self,
        building: &mut BuildingTable,
        kind: SinkKind,
        plan: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<SinkSummary> {
        let spec = ResultTableSinkSpec {
            table_name: building.table_name().to_string(),
            parquet_url: building.parquet_url().clone(),
            tenant: building.tenant(),
            writer_id: building.writer_id().to_string(),
            storage_precision: building.storage_precision(),
            lease: SinkLeaseKind::Table,
            kind,
        };
        let summary = self.run_sink(spec, plan, context).await?;
        building.rehold();
        Ok(summary)
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
        let spec = ResultTableSinkSpec {
            table_name: version.table_name().to_string(),
            parquet_url: version.parquet_url().clone(),
            tenant: version.tenant(),
            writer_id: version.writer_id().to_string(),
            storage_precision: version.storage_precision(),
            lease: SinkLeaseKind::Version {
                version: version.version(),
            },
            kind,
        };
        let summary = self.run_sink(spec, plan, context).await?;
        version.rehold();
        Ok(summary)
    }

    async fn run_sink(
        &self,
        spec: ResultTableSinkSpec,
        plan: Arc<dyn ExecutionPlan>,
        context: Arc<TaskContext>,
    ) -> Result<SinkSummary> {
        let sink = ResultTableSinkExec::new(spec, plan, self.clone());
        // Both through the structural classifier (`JammiError::from`): a
        // typed refusal raised inside the plan, placed or not, reaches the
        // caller as that variant, never stringified.
        let stream = sink.execute(0, context).map_err(JammiError::from)?;
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .map_err(JammiError::from)?;
        SinkSummary::from_batches(&batches)
    }
}
