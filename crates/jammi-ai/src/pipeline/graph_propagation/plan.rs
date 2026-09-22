//! The propagation's plans — one **stage** per hop, never the hops nested.
//!
//! ```text
//! stage k:   HopFoldExec(block k)
//!              SortExec(g, n, per partition)
//!                RepartitionExec(Hash(g))
//!                  join(state.key = adjacency.n)
//!                    state k−1   (a working table; InitialStateExec in stage 1)
//!                    adjacency   (the snapshot)
//! last:      SortPreservingMergeExec(_row_id) ← SortExec ← ReadoutExec ← stage K
//! ```
//!
//! A propagation of `K` hops is `K` plans run in order, each reading the
//! state the one before wrote and writing its own; the first builds block
//! `0` inline and the last reads out inline. It is the iterative form of the
//! recurrence, and it is that on purpose: a plan nesting `K` hops is `K`
//! times as deep, and every walker of a plan — DataFusion's optimizer, the
//! wire encoder and decoder that carry it to the compute plane, the plane's
//! own stage planner — recurses over that depth on a worker thread's fixed
//! stack. One hop per plan keeps the depth constant in `K`, the pool's floor
//! one hop's, and places each hop on its own.
//!
//! Everything relational — the adjacency, the join, the shuffle, the sorts —
//! is planned by DataFusion, under an
//! [out-of-core context](QueryContext::out_of_core) (partitioned sort-merge
//! joins, batches sized in bytes for `d`-wide rows): every blocking operator
//! of a hop holds a pool reservation and spills, so a graph larger than
//! `[engine] memory_limit` runs to completion, and a pool too small for the
//! hop's sort reservations (or a disk that cannot absorb the spill) ends in
//! the typed [`jammi_db::error::JammiError::ResourcesExhausted`], never an
//! out-of-memory kill. No step holds the edge set or the node set in process
//! memory: a hop has no aggregate in it.
//!
//! # The adjacency
//!
//! One relation `(g, n, w)` ([`adjacency_relation`]): the declared edges
//! oriented by the request's direction, declared self-edges dropped,
//! restricted to the node set (the embedding table's keys, or the edge
//! relation's distinct endpoints), collapsed to one row per pair — a pair
//! declared twice, or in both directions of an undirected read, is one edge,
//! its weight the larger (an order-free choice) — augmented with one
//! self-loop `(v, v)` per node (`Ã = A + I`), and sorted by `(n, g)`.
//!
//! The verb **snapshots it once**, as a working table it holds for its own
//! duration (`ResultTableKind::Adjacency`), and every hop — and the degrees —
//! read the snapshot: an edge source with no version surface can change while
//! a propagation runs, and hops that each read it afresh would propagate over
//! different graphs. The augmented degree `d̃` is a node's row count in the
//! snapshot; it is carried on every state row, so the fold reads `d̃_g` off a
//! group's own row and `d̃_n` off each joined row.
//!
//! The state is the join's streamed side and the adjacency its buffered
//! side: a sort-merge join holds the buffered rows of one key at a time under
//! a reservation sized by their batch, and a batch a sort emits is a slice of
//! one large array that reports the whole array's size. The narrow adjacency
//! rows are the cheap side to hold; the `d`-wide state rows stream through.

use std::sync::Arc;

use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::ScalarValue;
use datafusion::datasource::TableType;
use datafusion::functions::math::expr_fn::isnan;
use datafusion::functions_aggregate::expr_fn::{count, max};
use datafusion::logical_expr::{cast, lit, when, Expr};
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::expressions::col as physical_col;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties, Partitioning};
use datafusion::prelude::{col, DataFrame, JoinType};

use jammi_db::error::{JammiError, Result};
use jammi_db::session::QueryContext;

use super::hop::{HopFoldExec, HopSpec, GROUP_COLUMN, NEIGHBOUR_COLUMN, WEIGHT_COLUMN};
use super::readout::{BlockReadout, ReadoutExec, ReadoutSpec};
use super::seed::SeedSpec;
use super::state::{
    InitialFeatures, InitialStateExec, InitialStateSpec, ACC_COLUMN, DEGREE_COLUMN, KEY_COLUMN,
    TABLE_KEY_COLUMN, TABLE_VECTOR_COLUMN, X0_COLUMN, X_COLUMN,
};
use super::PropagationWeighting;
use crate::pipeline::graph_neighbourhood::EdgeDirection;

/// Where block `X⁽⁰⁾` and the node set come from.
pub enum FeatureSource {
    /// An embedding table's `(_row_id, vector)` rows. The node set is the
    /// table's keys, and the graph is the subgraph the edges induce on them:
    /// an edge with an endpoint the table does not hold is no edge.
    Table(Box<DataFrame>),
    /// Seed rows generated from the graph. The node set is every endpoint the
    /// edge relation names — the whole graph.
    StructuralSeed(SeedSpec),
}

/// The declared edges an adjacency is derived from, and how they are read.
pub struct EdgeRead {
    /// `_src`, `_dst` (`Utf8`) and, when `weighted`, `_weight` (`Float64`).
    pub edges: DataFrame,
    pub weighted: bool,
    pub direction: EdgeDirection,
    pub weighting: PropagationWeighting,
}

/// What one hop's plan reads its state from.
pub enum HopInput {
    /// The propagation's initial features: block `0` is built in this stage.
    Features(FeatureSource),
    /// The state an earlier stage wrote (the hop state's schema).
    State(Box<DataFrame>),
}

/// One hop's plan: at most one hop over the adjacency.
pub struct HopPlanSpec {
    /// The adjacency `(g, n, w)` — [`adjacency_relation`], as the verb
    /// snapshotted it.
    pub adjacency: DataFrame,
    pub weighting: PropagationWeighting,
    pub alpha: f64,
    pub readout: BlockReadout,
    /// The block this stage's hop produces (`k ≥ 1`), or `None` for a stage
    /// that only builds block `0` — a propagation of no hops.
    pub block: Option<usize>,
}

/// What the last stage's readout stamps on its rows.
pub struct Emit<'a> {
    /// The block width `d`.
    pub dimensions: usize,
    pub source_id: &'a str,
    pub model_id: &'a str,
}

const SRC: &str = "_src";
const DST: &str = "_dst";
const RAW_WEIGHT: &str = "_weight";

fn plan_error(step: &'static str) -> impl Fn(datafusion::error::DataFusionError) -> JammiError {
    move |e| match JammiError::from(e) {
        JammiError::DataFusion(inner) => {
            JammiError::Other(format!("graph propagation: {step}: {inner}"))
        }
        typed => typed,
    }
}

/// The order an adjacency's rows are committed in, `(n, g)` ascending: the
/// hop's join reads it by `n`.
pub fn adjacency_order() -> Vec<datafusion::logical_expr::SortExpr> {
    [NEIGHBOUR_COLUMN, GROUP_COLUMN]
        .iter()
        .map(|name| jammi_db::store::verbatim_column(name).sort(true, false))
        .collect()
}

/// The adjacency `(g, n, w)` of `read` over the node set `features` names:
/// the declared edges oriented, declared self-edges dropped, restricted to
/// the node set, collapsed to one row per pair (the larger declared weight —
/// an order-free choice), augmented with one self-loop per node (`Ã = A + I`)
/// and sorted by [`adjacency_order`]. The relation a propagation snapshots.
pub fn adjacency_relation(read: EdgeRead, features: &FeatureSource) -> Result<DataFrame> {
    let raw_weight = if read.weighted {
        col(RAW_WEIGHT)
    } else {
        lit(ScalarValue::Float64(None))
    };
    // The declared weight as `EdgeSimilarity` reads it: absent is a present,
    // full-strength edge; a negative similarity is anti-signal and a NaN no
    // signal, both clamped to zero rather than subtracted.
    let unit = lit(1.0_f64);
    let declared_weight = when(raw_weight.clone().is_null(), unit.clone())
        .when(isnan(raw_weight.clone()), lit(0.0_f64))
        .when(raw_weight.clone().lt(lit(0.0_f64)), lit(0.0_f64))
        .otherwise(raw_weight)
        .map_err(plan_error("edge weight"))?;
    let weight = match read.weighting {
        PropagationWeighting::EdgeSimilarity => declared_weight,
        PropagationWeighting::Uniform | PropagationWeighting::DegreeNormalized => unit.clone(),
    };
    let proper = read
        .edges
        .clone()
        .filter(
            col(SRC)
                .is_not_null()
                .and(col(DST).is_not_null())
                .and(col(SRC).not_eq(col(DST))),
        )
        .map_err(plan_error("edge filter"))?;
    let oriented = |group: &str, neighbour: &str| {
        proper
            .clone()
            .select(vec![
                col(group).alias(GROUP_COLUMN),
                col(neighbour).alias(NEIGHBOUR_COLUMN),
                weight.clone().alias(WEIGHT_COLUMN),
            ])
            .map_err(plan_error("edge orientation"))
    };
    // `Out`: `dst` is `src`'s neighbour, so `src` aggregates `dst`.
    let pairs = match read.direction {
        EdgeDirection::Out => oriented(SRC, DST)?,
        EdgeDirection::In => oriented(DST, SRC)?,
        EdgeDirection::Undirected => oriented(SRC, DST)?
            .union(oriented(DST, SRC)?)
            .map_err(plan_error("undirected union"))?,
    };

    let (pairs, nodes) = match features {
        FeatureSource::Table(table) => {
            let nodes = table
                .as_ref()
                .clone()
                .select(vec![
                    cast(col(TABLE_KEY_COLUMN), DataType::Utf8).alias(KEY_COLUMN)
                ])
                .map_err(plan_error("node keys"))?;
            let keyed = |alias: &str| {
                nodes
                    .clone()
                    .select(vec![col(KEY_COLUMN).alias(alias)])
                    .map_err(plan_error("node keys"))
            };
            let restricted = pairs
                .join(
                    keyed("_gk")?,
                    JoinType::Inner,
                    &[GROUP_COLUMN],
                    &["_gk"],
                    None,
                )
                .map_err(plan_error("group restriction"))?
                .join(
                    keyed("_nk")?,
                    JoinType::Inner,
                    &[NEIGHBOUR_COLUMN],
                    &["_nk"],
                    None,
                )
                .map_err(plan_error("neighbour restriction"))?;
            (restricted, nodes)
        }
        FeatureSource::StructuralSeed(_) => {
            let endpoint = |column: &str| {
                read.edges
                    .clone()
                    .select(vec![col(column).alias(KEY_COLUMN)])
                    .map_err(plan_error("endpoints"))
            };
            let nodes = endpoint(SRC)?
                .union(endpoint(DST)?)
                .map_err(plan_error("endpoint union"))?
                .filter(col(KEY_COLUMN).is_not_null())
                .map_err(plan_error("endpoint filter"))?
                .distinct()
                .map_err(plan_error("endpoint distinct"))?;
            (pairs, nodes)
        }
    };
    pairs
        .aggregate(
            vec![col(GROUP_COLUMN), col(NEIGHBOUR_COLUMN)],
            vec![max(col(WEIGHT_COLUMN)).alias(WEIGHT_COLUMN)],
        )
        .map_err(plan_error("pair collapse"))?
        .union(
            nodes
                .select(vec![
                    col(KEY_COLUMN).alias(GROUP_COLUMN),
                    col(KEY_COLUMN).alias(NEIGHBOUR_COLUMN),
                    unit.alias(WEIGHT_COLUMN),
                ])
                .map_err(plan_error("self-loops"))?,
        )
        .map_err(plan_error("self-loop union"))?
        .sort(adjacency_order())
        .map_err(plan_error("adjacency order"))
}

/// One hop's plan, producing hop state — the one plan-building site: the
/// two propagating verbs and anything that must carry the same plan (the
/// compute plane's codec) build it here, so they hold the same nodes by
/// construction. Read through `ctx`, a [`QueryContext::out_of_core`] context.
pub async fn hop_plan(
    ctx: &QueryContext,
    input: HopInput,
    spec: &HopPlanSpec,
) -> Result<Arc<dyn ExecutionPlan>> {
    let partitions = ctx.state().config().target_partitions().max(1);
    let state = match input {
        HopInput::State(state) => *state,
        HopInput::Features(features) => {
            let initial = initial_state_plan(features, spec).await?;
            if spec.block.is_none() {
                return Ok(initial);
            }
            ctx.read_table(Arc::new(PlanTable { plan: initial }))
                .map_err(plan_error("initial state"))?
        }
    };
    let Some(block) = spec.block else {
        return state
            .create_physical_plan()
            .await
            .map_err(plan_error("state plan"));
    };

    // Each neighbour's vector and degree for each group. A node's `X⁽⁰⁾` and
    // running readout ride its own row — the self-loop's — and are nulled on
    // every other row, where they are another node's: a null list costs
    // nothing through the shuffle and the sort.
    let own = |column: &str| -> Result<Expr> {
        Ok(
            when(col(GROUP_COLUMN).eq(col(NEIGHBOUR_COLUMN)), col(column))
                .end()
                .map_err(plan_error("own-row carry"))?
                .alias(column),
        )
    };
    let rows = state
        .join(
            spec.adjacency.clone(),
            JoinType::Inner,
            &[KEY_COLUMN],
            &[NEIGHBOUR_COLUMN],
            None,
        )
        .map_err(plan_error("hop join"))?
        .select(vec![
            col(GROUP_COLUMN),
            col(NEIGHBOUR_COLUMN),
            col(WEIGHT_COLUMN),
            col(X_COLUMN),
            own(X0_COLUMN)?,
            own(ACC_COLUMN)?,
            col(DEGREE_COLUMN),
        ])
        .map_err(plan_error("hop projection"))?
        .create_physical_plan()
        .await
        .map_err(plan_error("hop plan"))?;
    let grouped = sorted_within_partitions(
        hash_partitioned(rows, GROUP_COLUMN, partitions)?,
        &[GROUP_COLUMN, NEIGHBOUR_COLUMN],
    )?;
    Ok(Arc::new(
        HopFoldExec::try_new(
            grouped,
            HopSpec {
                weighting: spec.weighting,
                alpha: spec.alpha,
                block,
                readout: spec.readout.clone(),
            },
        )
        .map_err(plan_error("hop fold"))?,
    ))
}

/// Block `0`: the initial features with each node's augmented degree `d̃` —
/// its rows in the adjacency, its self-loop among them.
async fn initial_state_plan(
    features: FeatureSource,
    spec: &HopPlanSpec,
) -> Result<Arc<dyn ExecutionPlan>> {
    let degrees = spec
        .adjacency
        .clone()
        .aggregate(
            vec![col(GROUP_COLUMN)],
            vec![count(lit(1_i64)).alias(DEGREE_COLUMN)],
        )
        .map_err(plan_error("degrees"))?;
    let (input, features) = match features {
        FeatureSource::Table(table) => (
            table
                .select(vec![
                    cast(col(TABLE_KEY_COLUMN), DataType::Utf8).alias(TABLE_KEY_COLUMN),
                    col(TABLE_VECTOR_COLUMN),
                ])
                .map_err(plan_error("initial features"))?
                .join(
                    degrees,
                    JoinType::Inner,
                    &[TABLE_KEY_COLUMN],
                    &[GROUP_COLUMN],
                    None,
                )
                .map_err(plan_error("initial degrees"))?
                .select(vec![
                    col(TABLE_KEY_COLUMN),
                    col(TABLE_VECTOR_COLUMN),
                    col(DEGREE_COLUMN),
                ])
                .map_err(plan_error("initial projection"))?,
            InitialFeatures::Table,
        ),
        FeatureSource::StructuralSeed(seed) => (
            degrees
                .select(vec![
                    col(GROUP_COLUMN).alias(KEY_COLUMN),
                    col(DEGREE_COLUMN),
                ])
                .map_err(plan_error("seed input"))?,
            InitialFeatures::StructuralSeed(seed),
        ),
    };
    let input = input
        .create_physical_plan()
        .await
        .map_err(plan_error("initial features plan"))?;
    Ok(Arc::new(
        InitialStateExec::try_new(
            input,
            InitialStateSpec {
                features,
                carry_x0: spec.alpha != 0.0,
            },
        )
        .map_err(plan_error("initial state"))?,
    ))
}

/// The last stage's `state` read out into the rows the embedding sink
/// writes, in `_row_id` order through one merged partition — so the written
/// object, and the content digest a downstream producer anchors on, is the
/// same whatever the partitioning.
pub fn emit_plan(
    state: Arc<dyn ExecutionPlan>,
    readout: BlockReadout,
    emit: Emit<'_>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let readout: Arc<dyn ExecutionPlan> = Arc::new(
        ReadoutExec::try_new(
            state,
            ReadoutSpec {
                readout,
                dimensions: emit.dimensions,
                source_id: emit.source_id.to_string(),
                model_id: emit.model_id.to_string(),
            },
        )
        .map_err(plan_error("readout"))?,
    );
    let ordering = ascending(&readout.schema(), &["_row_id"])?;
    let sorted = sorted_within_partitions(readout, &["_row_id"])?;
    Ok(Arc::new(SortPreservingMergeExec::new(ordering, sorted)))
}

fn ascending(schema: &SchemaRef, columns: &[&str]) -> Result<LexOrdering> {
    let options = SortOptions {
        descending: false,
        nulls_first: false,
    };
    let exprs = columns
        .iter()
        .map(|name| {
            physical_col(name, schema.as_ref()).map(|expr| PhysicalSortExpr { expr, options })
        })
        .collect::<datafusion::error::Result<Vec<_>>>()
        .map_err(plan_error("sort key"))?;
    LexOrdering::new(exprs)
        .ok_or_else(|| JammiError::Other("graph propagation: an empty sort key".into()))
}

/// `plan` sorted by `columns` within each of its partitions.
fn sorted_within_partitions(
    plan: Arc<dyn ExecutionPlan>,
    columns: &[&str],
) -> Result<Arc<dyn ExecutionPlan>> {
    let ordering = ascending(&plan.schema(), columns)?;
    Ok(Arc::new(
        SortExec::new(ordering, plan).with_preserve_partitioning(true),
    ))
}

/// `plan` hash-partitioned on `column`, `partitions` ways — left alone where
/// it already is one partition and one is asked for.
fn hash_partitioned(
    plan: Arc<dyn ExecutionPlan>,
    column: &str,
    partitions: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    if partitions == 1 && plan.output_partitioning().partition_count() == 1 {
        return Ok(plan);
    }
    let key = physical_col(column, plan.schema().as_ref()).map_err(plan_error("partition key"))?;
    Ok(Arc::new(
        RepartitionExec::try_new(plan, Partitioning::Hash(vec![key], partitions))
            .map_err(plan_error("hash repartition"))?,
    ))
}

/// A physical plan as a relation, so the next hop's join is planned by
/// DataFusion over the previous hop's operator.
#[derive(Debug)]
struct PlanTable {
    plan: Arc<dyn ExecutionPlan>,
}

#[async_trait]
impl TableProvider for PlanTable {
    fn schema(&self) -> SchemaRef {
        self.plan.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        let Some(projection) = projection else {
            return Ok(Arc::clone(&self.plan));
        };
        let schema = self.plan.schema();
        let exprs = projection
            .iter()
            .map(|index| {
                let name = schema.field(*index).name();
                physical_col(name, schema.as_ref())
                    .map(|expr| (expr as Arc<dyn PhysicalExpr>, name.clone()))
            })
            .collect::<datafusion::error::Result<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new(
            exprs,
            Arc::clone(&self.plan),
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        Array, ArrayRef, FixedSizeListArray, Float32Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{Field, Schema};
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::prelude::{SessionConfig, SessionContext};

    use crate::pipeline::graph_propagation::PropagationOutput;

    const NODES: usize = 12_000;
    const DIMENSIONS: usize = 128;

    /// A ring of `NODES` accounts with two chords each — every node has the
    /// same small degree, so a hop's joined rows are `≈ 7 · NODES` vectors.
    /// `rotate` rotates the row order of the relation without changing the
    /// graph.
    fn ring_with_chords(rotate: usize) -> RecordBatch {
        let name = |i: usize| format!("acct-{:05}", i % NODES);
        let mut rows: Vec<(String, String)> = (0..NODES)
            .flat_map(|i| {
                [
                    (name(i), name(i + 1)),
                    (name(i), name(i + 37)),
                    (name(i), name(i + 401)),
                ]
            })
            .collect();
        rows.rotate_left(rotate);
        let column = |side: fn(&(String, String)) -> &String| -> ArrayRef {
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| side(row).as_str())
                    .collect::<Vec<_>>(),
            ))
        };
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(SRC, DataType::Utf8, false),
                Field::new(DST, DataType::Utf8, false),
            ])),
            vec![column(|row| &row.0), column(|row| &row.1)],
        )
        .unwrap()
    }

    struct Run {
        /// `(_row_id, vector bits)` in the order the plan emitted them.
        rows: Vec<(String, Vec<u32>)>,
        spills: usize,
    }

    /// Plan and run a three-hop structure encoding of the ring under a pool
    /// of `pool_bytes` and `partitions` partitions.
    async fn encode_ring(pool_bytes: usize, partitions: usize, rotate: usize) -> Result<Run> {
        // The session's pool at `pool_bytes`, over the default disk manager —
        // the runtime shape a session builds.
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(jammi_db::memory_pool::ActiveSpillPool::new(
                pool_bytes,
            )))
            .build_arc()
            .unwrap();
        let config = SessionConfig::new().with_target_partitions(partitions);
        let session = SessionContext::new_with_config_rt(config, runtime);
        let readout = BlockReadout::lower(
            &PropagationOutput::WeightedSum {
                weights: vec![0.0, 1.0, 1.0, 1.0],
            },
            3,
        )?;
        let ctx = QueryContext::from(session).out_of_core(readout.state_row_bytes(DIMENSIONS));
        let edges = ctx
            .read_table(Arc::new(
                datafusion::datasource::MemTable::try_new(
                    ring_with_chords(rotate).schema(),
                    vec![vec![ring_with_chords(rotate)]],
                )
                .unwrap(),
            ))
            .unwrap();
        let hops = readout.last_block();
        let seed = FeatureSource::StructuralSeed(SeedSpec::new(7, DIMENSIONS, 3.0, 0.0)?);
        let adjacency = adjacency_relation(
            EdgeRead {
                edges,
                weighted: false,
                direction: EdgeDirection::Undirected,
                weighting: PropagationWeighting::Uniform,
            },
            &seed,
        )?;
        let stage = |block| HopPlanSpec {
            adjacency: adjacency.clone(),
            weighting: PropagationWeighting::Uniform,
            alpha: 0.0,
            readout: readout.clone(),
            block: Some(block),
        };
        // The verb's shape: each stage's state handed to the next as a
        // relation — here a `MemTable`, there a working table.
        let mut spills = 0;
        let mut census: std::collections::BTreeMap<String, usize> = Default::default();
        let mut input = HopInput::Features(seed);
        let mut plans = Vec::new();
        for block in 1..hops {
            let state = hop_plan(&ctx, input, &stage(block)).await?;
            let batches = datafusion::physical_plan::collect(Arc::clone(&state), ctx.task_ctx())
                .await
                .map_err(JammiError::from)?;
            let held = datafusion::datasource::MemTable::try_new(state.schema(), vec![batches])
                .map_err(JammiError::from)?;
            input = HopInput::State(Box::new(
                ctx.read_table(Arc::new(held)).map_err(JammiError::from)?,
            ));
            plans.push(state);
        }
        let last = hop_plan(&ctx, input, &stage(hops)).await?;
        let plan = emit_plan(
            last,
            readout,
            Emit {
                dimensions: DIMENSIONS,
                source_id: "edges",
                model_id: "graph_structure",
            },
        )?;
        let batches = datafusion::physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .map_err(JammiError::from)?;
        plans.push(plan);

        let mut rows = Vec::new();
        for batch in &batches {
            let keys = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let vectors = batch
                .column_by_name("vector")
                .unwrap()
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            for row in 0..batch.num_rows() {
                let cell = vectors.value(row);
                let lanes = cell.as_any().downcast_ref::<Float32Array>().unwrap();
                rows.push((
                    keys.value(row).to_string(),
                    lanes.values().iter().map(|v| v.to_bits()).collect(),
                ));
            }
        }
        // Every operator's spill count over every stage — an explicit
        // work-stack per plan — and a census of the operators, which is what
        // the pool's floor is made of.
        let mut pending = plans;
        while let Some(node) = pending.pop() {
            spills += node.metrics().and_then(|m| m.spill_count()).unwrap_or(0);
            *census.entry(node.name().to_string()).or_default() +=
                node.output_partitioning().partition_count();
            pending.extend(node.children().into_iter().cloned());
        }
        println!("pool={pool_bytes} partitions={partitions}: operator partitions {census:?}");
        Ok(Run { rows, spills })
    }

    const ROOMY: usize = 1 << 30;
    /// A hop's joined rows are `≈ 7 · 12000 · 128 · 8 B ≈ 82 MiB`, more than
    /// the share a sort gets of this pool while its hop's other sorts and
    /// join hold memory too, so the sorts must spill to finish — while each
    /// share stays far above the 1 MiB merge reservation plus the batches a
    /// sort needs to make progress.
    const TIGHT: usize = 160 << 20;
    /// Below a single sort's merge reservation plus a batch, once the hop's
    /// other consumers hold theirs.
    const BELOW_FLOOR: usize = 4 << 20;

    #[tokio::test]
    async fn the_bits_do_not_depend_on_partitioning_row_order_or_spilling() {
        let reference = encode_ring(ROOMY, 1, 0).await.unwrap();
        assert_eq!(reference.rows.len(), NODES);
        assert!(
            reference.rows.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "rows leave in key order"
        );
        println!("roomy reference: {} spills", reference.spills);

        for (pool, partitions, rotate) in [(ROOMY, 4, 0), (ROOMY, 7, 911), (TIGHT, 2, 0)] {
            let run = encode_ring(pool, partitions, rotate).await.unwrap();
            println!(
                "pool={pool} partitions={partitions} rotate={rotate}: {} spills",
                run.spills
            );
            assert!(
                run.rows == reference.rows,
                "pool={pool} partitions={partitions} rotate={rotate}: the output differs"
            );
            if pool == TIGHT {
                assert!(
                    run.spills > reference.spills,
                    "the tight run must spill more than the roomy one to mean anything: {} vs {}",
                    run.spills,
                    reference.spills
                );
            }
        }
    }

    #[tokio::test]
    async fn a_pool_below_the_floor_is_the_typed_refusal() {
        let refused = encode_ring(BELOW_FLOOR, 2, 0)
            .await
            .err()
            .expect("4 MiB cannot hold a hop's sort reservations");
        assert!(
            matches!(refused, JammiError::ResourcesExhausted { .. }),
            "expected the typed pool refusal, got: {refused}"
        );
    }
}
