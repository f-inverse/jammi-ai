//! The hop state — the keyed relation a propagation carries from hop to hop —
//! and [`InitialStateExec`], which builds it from a propagation's initial
//! features.
//!
//! One row per node:
//!
//! | column | type | meaning |
//! |---|---|---|
//! | `key` | `Utf8` | the node key |
//! | `x` | `List<Float64>` | the current block `X⁽ᵏ⁾` |
//! | `x0` | `List<Float64>`, nullable | `X⁽⁰⁾`, carried only under a non-zero teleport |
//! | `acc` | `List<Float64>`, nullable | the running readout of blocks `0..k` |
//! | `deg` | `Int64` | the node's augmented degree `d̃`, computed once and carried |
//!
//! The vectors stay `f64` between hops: the single `f32` cast is the
//! readout's. Variable-width lists (not fixed-size ones) because a null
//! fixed-size list still occupies its lanes, and a hop's join nulls `x0`/`acc`
//! on every row but a node's own.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array, ListArray,
    RecordBatch, StringArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::StreamExt;

use super::seed::SeedSpec;

pub(crate) const KEY_COLUMN: &str = "key";
pub(crate) const X_COLUMN: &str = "x";
pub(crate) const X0_COLUMN: &str = "x0";
pub(crate) const ACC_COLUMN: &str = "acc";
/// The augmented degree column: on the initial input, on the state, and on a
/// hop's joined rows (the neighbour's).
pub(crate) const DEGREE_COLUMN: &str = "deg";

/// The node-key column [`InitialFeatures::Table`] reads.
pub(crate) const TABLE_KEY_COLUMN: &str = "_row_id";
/// The vector column [`InitialFeatures::Table`] reads.
pub(crate) const TABLE_VECTOR_COLUMN: &str = "vector";
/// The `List<Float64>` every state vector column is.
pub(crate) fn vector_type() -> DataType {
    DataType::List(Arc::new(Field::new_list_field(DataType::Float64, true)))
}

/// The hop state's schema.
pub(crate) fn state_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(KEY_COLUMN, DataType::Utf8, false),
        Field::new(X_COLUMN, vector_type(), false),
        Field::new(X0_COLUMN, vector_type(), true),
        Field::new(ACC_COLUMN, vector_type(), true),
        Field::new(DEGREE_COLUMN, DataType::Int64, false),
    ]))
}

/// The `Int64` column of `batch`, by name.
pub(crate) fn int64_column<'a>(batch: &'a RecordBatch, name: &str) -> DfResult<&'a Int64Array> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "graph propagation: `{name}` is not an Int64 column"
            ))
        })
}

/// An augmented degree read off a column: the adjacency gives every node its
/// self-loop, so a value below one is a broken input, not a node.
pub(crate) fn augmented_degree(key: &str, value: i64) -> DfResult<u64> {
    u64::try_from(value)
        .ok()
        .filter(|degree| *degree >= 1)
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "graph propagation: node '{key}' has augmented degree {value}, below its own \
                 self-loop"
            ))
        })
}

/// A `Utf8` column of `batch`, by name.
pub(crate) fn utf8_column<'a>(batch: &'a RecordBatch, name: &str) -> DfResult<&'a StringArray> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| {
            DataFusionError::Execution(format!("graph propagation: `{name}` is not a Utf8 column"))
        })
}

/// A borrowed view of a batch whose vector columns are `List<Float64>` — the
/// hop state, and the joined rows a hop folds.
pub(crate) struct HopState<'a> {
    batch: &'a RecordBatch,
    keys: &'a StringArray,
}

impl<'a> HopState<'a> {
    /// Refuses a schema that is not the hop state's, by column name and type.
    pub(crate) fn check(schema: &Schema) -> DfResult<()> {
        let expect = |name: &str, data_type: &DataType| match schema.field_with_name(name) {
            Ok(field) if field.data_type() == data_type => Ok(()),
            Ok(field) => Err(DataFusionError::Plan(format!(
                "graph propagation: state column `{name}` is {}, expected {data_type}",
                field.data_type()
            ))),
            Err(_) => Err(DataFusionError::Plan(format!(
                "graph propagation: the input has no state column `{name}`"
            ))),
        };
        expect(KEY_COLUMN, &DataType::Utf8)?;
        expect(DEGREE_COLUMN, &DataType::Int64)?;
        [X_COLUMN, X0_COLUMN, ACC_COLUMN]
            .iter()
            .try_for_each(|name| expect(name, &vector_type()))
    }

    pub(crate) fn read(batch: &'a RecordBatch) -> DfResult<Self> {
        Ok(Self {
            batch,
            keys: utf8_column(batch, KEY_COLUMN)?,
        })
    }

    pub(crate) fn key(&self, row: usize) -> &'a str {
        self.keys.value(row)
    }

    /// `column`'s vector at `row`, `None` when null.
    pub(crate) fn vector(&self, column: &str, row: usize) -> DfResult<Option<&'a [f64]>> {
        list_vector(self.batch, column, row)
    }
}

/// The `List<Float64>` cell of `column` at `row`, borrowed — `None` when null.
pub(crate) fn list_vector<'a>(
    batch: &'a RecordBatch,
    column: &str,
    row: usize,
) -> DfResult<Option<&'a [f64]>> {
    let mismatch = || {
        DataFusionError::Execution(format!(
            "graph propagation: `{column}` is not a List<Float64> column"
        ))
    };
    let list = batch
        .column_by_name(column)
        .and_then(|c| c.as_any().downcast_ref::<ListArray>())
        .ok_or_else(mismatch)?;
    if list.is_null(row) {
        return Ok(None);
    }
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(mismatch)?;
    let offsets = list.value_offsets();
    let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
    Ok(Some(&values.values()[start..end]))
}

/// One nullable `List<Float64>` column under construction.
#[derive(Default)]
struct VectorColumn {
    values: Vec<f64>,
    lengths: Vec<usize>,
    valid: Vec<bool>,
}

impl VectorColumn {
    fn push(&mut self, vector: Option<&[f64]>) {
        let vector = vector.filter(|v| !v.is_empty());
        self.lengths.push(vector.map_or(0, <[f64]>::len));
        self.valid.push(vector.is_some());
        self.values.extend_from_slice(vector.unwrap_or_default());
    }

    fn finish(self) -> ArrayRef {
        Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::Float64, true)),
            OffsetBuffer::from_lengths(self.lengths),
            Arc::new(Float64Array::from(self.values)),
            Some(NullBuffer::from(self.valid)),
        ))
    }
}

/// Accumulates hop state rows into batches of at most `batch_size` rows.
///
/// It holds at most one batch — the working set every streaming operator
/// has between one input batch and its output, which DataFusion's own
/// row-wise operators do not account to the pool either: the pool bounds
/// what can pile up, and a batch handed on as soon as it is full cannot. The
/// out-of-core context sizes that batch in bytes.
pub(crate) struct StateBatchBuilder {
    keys: Vec<String>,
    x: VectorColumn,
    x0: VectorColumn,
    acc: VectorColumn,
    degrees: Vec<i64>,
    batch_size: usize,
}

impl StateBatchBuilder {
    /// A builder that hands back a batch every `batch_size` rows.
    pub(crate) fn new(batch_size: usize) -> Self {
        Self {
            keys: Vec::new(),
            x: VectorColumn::default(),
            x0: VectorColumn::default(),
            acc: VectorColumn::default(),
            degrees: Vec::new(),
            batch_size: batch_size.max(1),
        }
    }

    /// Append one row; an empty `x0`/`acc` is stored null. Returns a full
    /// batch once `batch_size` rows have accumulated.
    pub(crate) fn push(
        &mut self,
        key: &str,
        x: &[f64],
        x0: Option<&[f64]>,
        acc: Option<&[f64]>,
        degree: u64,
    ) -> DfResult<Option<RecordBatch>> {
        self.keys.push(key.to_string());
        self.x.push(Some(x));
        self.x0.push(x0);
        self.acc.push(acc);
        self.degrees.push(degree as i64);
        if self.keys.len() >= self.batch_size {
            return self.flush();
        }
        Ok(None)
    }

    /// The rows buffered so far as a batch, `None` when there are none.
    pub(crate) fn flush(&mut self) -> DfResult<Option<RecordBatch>> {
        if self.keys.is_empty() {
            return Ok(None);
        }
        let keys = std::mem::take(&mut self.keys);
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(keys)),
            std::mem::take(&mut self.x).finish(),
            std::mem::take(&mut self.x0).finish(),
            std::mem::take(&mut self.acc).finish(),
            Arc::new(Int64Array::from(std::mem::take(&mut self.degrees))),
        ];
        Ok(Some(RecordBatch::try_new(state_schema(), columns)?))
    }
}

/// Where a propagation's block `X⁽⁰⁾` comes from. The two verbs that
/// propagate differ here and in their readout, and nowhere else. Either
/// input carries the node's augmented degree in `deg`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum InitialFeatures {
    /// An embedding table's rows: the input carries `_row_id` and a
    /// `FixedSizeList<Float32>` `vector`, widened losslessly to `f64`.
    Table,
    /// Generated from the graph: the input carries `key`, and each row is
    /// its [`SeedSpec`] seed row at its degree.
    StructuralSeed(SeedSpec),
}

/// What [`InitialStateExec`] is built from, and what crosses a process
/// boundary for it.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InitialStateSpec {
    pub(crate) features: InitialFeatures,
    /// Whether the state carries `x0` — only a non-zero teleport reads it.
    pub(crate) carry_x0: bool,
}

/// Builds hop state block `0` from a propagation's initial features. A
/// row-wise map — it keeps its input's partitioning.
#[derive(Debug)]
pub struct InitialStateExec {
    input: Arc<dyn ExecutionPlan>,
    spec: InitialStateSpec,
    properties: Arc<PlanProperties>,
}

impl InitialStateExec {
    /// Bind `spec` over `input`, refusing an input without the columns the
    /// spec's features read.
    pub fn try_new(input: Arc<dyn ExecutionPlan>, spec: InitialStateSpec) -> DfResult<Self> {
        let schema = input.schema();
        let required: &[&str] = match spec.features {
            InitialFeatures::Table => &[TABLE_KEY_COLUMN, TABLE_VECTOR_COLUMN, DEGREE_COLUMN],
            InitialFeatures::StructuralSeed(_) => &[KEY_COLUMN, DEGREE_COLUMN],
        };
        if let Some(missing) = required.iter().find(|c| schema.index_of(c).is_err()) {
            return Err(DataFusionError::Plan(format!(
                "InitialStateExec: the input has no `{missing}` column"
            )));
        }
        let properties = PlanProperties::new(
            EquivalenceProperties::new(state_schema()),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            spec,
            properties: Arc::new(properties),
        })
    }

    /// The spec this node was built from.
    pub fn spec(&self) -> &InitialStateSpec {
        &self.spec
    }
}

impl DisplayAs for InitialStateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        let features = match self.spec.features {
            InitialFeatures::Table => "table",
            InitialFeatures::StructuralSeed(_) => "structural_seed",
        };
        write!(f, "InitialStateExec: features={features}")
    }
}

impl ExecutionPlan for InitialStateExec {
    fn name(&self) -> &str {
        "InitialStateExec"
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
        Ok(Arc::new(Self::try_new(
            Arc::clone(&children[0]),
            self.spec.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let spec = self.spec.clone();
        let input = self.input.execute(partition, context)?;
        let stream = input.map(move |batch| {
            let batch = batch?;
            // One output batch per input batch.
            let mut rows = StateBatchBuilder::new(usize::MAX);
            initial_rows(&spec, &batch, &mut rows)?;
            Ok(rows
                .flush()?
                .unwrap_or_else(|| RecordBatch::new_empty(state_schema())))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            state_schema(),
            stream,
        )))
    }
}

/// Append `batch`'s initial state rows to `rows`.
fn initial_rows(
    spec: &InitialStateSpec,
    batch: &RecordBatch,
    rows: &mut StateBatchBuilder,
) -> DfResult<()> {
    let mut x: Vec<f64> = Vec::new();
    let degrees = int64_column(batch, DEGREE_COLUMN)?;
    let push = |rows: &mut StateBatchBuilder, key: &str, x: &[f64], row: usize| {
        let degree = augmented_degree(key, degrees.value(row))?;
        rows.push(key, x, spec.carry_x0.then_some(x), None, degree)
            .map(|_| ())
    };
    match &spec.features {
        InitialFeatures::Table => {
            let keys = utf8_column(batch, TABLE_KEY_COLUMN)?;
            let mismatch = || {
                DataFusionError::Execution(format!(
                    "InitialStateExec: `{TABLE_VECTOR_COLUMN}` is not a FixedSizeList<Float32>"
                ))
            };
            let vectors = batch
                .column_by_name(TABLE_VECTOR_COLUMN)
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(mismatch)?;
            let lanes = vectors
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(mismatch)?
                .values();
            let width = vectors.value_length() as usize;
            for row in (0..batch.num_rows()).filter(|row| vectors.is_valid(*row)) {
                x.clear();
                x.extend(
                    lanes[row * width..(row + 1) * width]
                        .iter()
                        .map(|v| f64::from(*v)),
                );
                push(rows, keys.value(row), &x, row)?;
            }
        }
        InitialFeatures::StructuralSeed(seed) => {
            let keys = utf8_column(batch, KEY_COLUMN)?;
            for row in 0..batch.num_rows() {
                let degree = augmented_degree(keys.value(row), degrees.value(row))?;
                x.clear();
                seed.row_into(keys.value(row), degree, &mut x);
                push(rows, keys.value(row), &x, row)?;
            }
        }
    }
    Ok(())
}
