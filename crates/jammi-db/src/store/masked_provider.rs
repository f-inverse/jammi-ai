//! The SQL/exact read path of a versioned result table.
//!
//! [`MaskedTableProvider`] unions one `ListingTable` per fragment, each
//! wrapped in a [`MaskExec`] that drops the rows the version's deletion mask
//! hides for that fragment's stamped version, and projects back to the
//! caller's columns. Its `scan` contract: (1) the pushed-down projection is
//! remapped to always include `_row_id` (so `COUNT(*)` and `SELECT vector`,
//! which project no key, are still masked); (2) every fragment is scanned with
//! `limit = None` — `ListingTable` consumes a limit as file-list truncation
//! AND a scan fetch, so a fetch pushed under the mask would return fewer than
//! `LIMIT n` live rows; (3) `UnionExec` over the masked fragment scans; (4)
//! `ProjectionExec` back to the caller's columns. Filter pushdown is not
//! overridden (the `FilterExec` stays above), and [`MaskExec`] never overrides
//! `supports_limit_pushdown` (default `false`) nor `with_fetch` (default
//! `None`), so the outer fetch survives above the mask as a `LimitExec`.
//!
//! [`PlaceholderProvider`] stands in for a versioned table whose current
//! manifest cannot be resolved: it plans (schema off the catalog row) and
//! every scan fails with the typed `VersionUnavailable`, which the structural
//! error classifier restores for every SQL caller — registering nothing would
//! surface a stringly planner not-found, registering the base provider would
//! resurrect deleted rows.

use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{Array, BooleanArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::datasource::{TableProvider, TableType};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::Expr;
use datafusion::physical_expr::expressions::col;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
};
use futures::StreamExt;

use crate::error::JammiError;
use crate::store::deletes::DeletionMask;

/// One fragment: its provider and its stamped version.
pub struct MaskedFragment {
    pub provider: Arc<dyn TableProvider>,
    pub version: i64,
}

/// The union-of-masked-fragments provider. See the module doc.
pub struct MaskedTableProvider {
    table_name: String,
    fragments: Vec<MaskedFragment>,
    mask: Arc<DeletionMask>,
    schema: SchemaRef,
}

impl std::fmt::Debug for MaskedTableProvider {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaskedTableProvider")
            .field("table", &self.table_name)
            .field("fragments", &self.fragments.len())
            .field("mask_entries", &self.mask.len())
            .finish()
    }
}

impl MaskedTableProvider {
    /// `schema` is the base fragment's inferred schema, pinned on every
    /// fragment provider by the caller.
    pub fn new(
        table_name: String,
        fragments: Vec<MaskedFragment>,
        mask: Arc<DeletionMask>,
        schema: SchemaRef,
    ) -> Self {
        Self {
            table_name,
            fragments,
            mask,
            schema,
        }
    }

    /// The table's mask.
    pub fn mask(&self) -> &Arc<DeletionMask> {
        &self.mask
    }
}

#[async_trait]
impl TableProvider for MaskedTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let row_id_idx = self.schema.index_of("_row_id")?;
        // (1) the inner projection always carries `_row_id`.
        let inner: Vec<usize> = match projection {
            Some(p) => {
                let mut v = p.clone();
                if !v.contains(&row_id_idx) {
                    v.push(row_id_idx);
                }
                v.sort_unstable();
                v.dedup();
                v
            }
            None => (0..self.schema.fields().len()).collect(),
        };
        let inner_row_id = inner
            .iter()
            .position(|&i| i == row_id_idx)
            .expect("_row_id is in the inner projection by construction");

        // (2)+(3): every fragment with `limit = None`, masked per version.
        let mut scans: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(self.fragments.len());
        for fragment in &self.fragments {
            let plan = fragment
                .provider
                .scan(state, Some(&inner), &[], None)
                .await?;
            let plan: Arc<dyn ExecutionPlan> = if self.mask.masks_version(fragment.version) {
                Arc::new(MaskExec::new(
                    plan,
                    fragment.version,
                    Arc::clone(&self.mask),
                    inner_row_id,
                    self.table_name.clone(),
                ))
            } else {
                plan
            };
            scans.push(plan);
        }
        let unioned: Arc<dyn ExecutionPlan> = if scans.len() == 1 {
            scans.pop().expect("one scan")
        } else {
            UnionExec::try_new(scans)?
        };

        // (4) back to the caller's columns (in the caller's order).
        let Some(requested) = projection else {
            return Ok(unioned);
        };
        let inner_schema = unioned.schema();
        let exprs = requested
            .iter()
            .map(|&i| {
                let name = self.schema.field(i).name().clone();
                Ok((col(&name, inner_schema.as_ref())?, name))
            })
            .collect::<DfResult<Vec<_>>>()?;
        Ok(Arc::new(ProjectionExec::try_new(exprs, unioned)?))
    }
}

/// The arrow filter that drops a fragment's masked rows. Partitioning and
/// ordering are the input's; limit pushdown is NEVER supported (defaults).
#[derive(Debug)]
pub struct MaskExec {
    input: Arc<dyn ExecutionPlan>,
    version: i64,
    mask: Arc<DeletionMask>,
    row_id_index: usize,
    table_name: String,
    properties: Arc<PlanProperties>,
}

impl MaskExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        version: i64,
        mask: Arc<DeletionMask>,
        row_id_index: usize,
        table_name: String,
    ) -> Self {
        let properties = PlanProperties::new(
            input.equivalence_properties().clone(),
            input.output_partitioning().clone(),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Self {
            input,
            version,
            mask,
            row_id_index,
            table_name,
            properties: Arc::new(properties),
        }
    }

    /// The fragment version this node masks.
    pub fn version(&self) -> i64 {
        self.version
    }

    fn filter_batch(
        batch: &RecordBatch,
        row_id_index: usize,
        mask: &DeletionMask,
        version: i64,
    ) -> DfResult<RecordBatch> {
        let ids = batch.column(row_id_index);
        let ids = arrow::compute::cast(ids, &DataType::Utf8)?;
        let ids = ids
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| DataFusionError::Internal("_row_id is not a string column".into()))?;
        let keep: BooleanArray = (0..ids.len())
            .map(|i| Some(!(ids.is_valid(i) && mask.is_masked(ids.value(i), version))))
            .collect();
        Ok(arrow::compute::filter_record_batch(batch, &keep)?)
    }
}

impl DisplayAs for MaskExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "MaskExec: version={}", self.version)
    }
}

impl ExecutionPlan for MaskExec {
    fn name(&self) -> &str {
        "MaskExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.version,
            Arc::clone(&self.mask),
            self.row_id_index,
            self.table_name.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let schema = self.schema();
        let input = Arc::clone(&self.input);
        let mask = Arc::clone(&self.mask);
        let version = self.version;
        let row_id_index = self.row_id_index;
        let table = self.table_name.clone();
        // The input stream is opened AFTER the (test-only) park so a test can
        // delete the fragment object in the window before the drain.
        let stream = futures::stream::once(async move {
            #[cfg(feature = "test-hooks")]
            masked_scan_test_hooks::maybe_park_before_masked_scan_drain(&table).await;
            #[cfg(not(feature = "test-hooks"))]
            let _ = &table;
            match input.execute(partition, context) {
                Ok(inner) => inner
                    .map(move |b| {
                        b.and_then(|b| Self::filter_batch(&b, row_id_index, &mask, version))
                    })
                    .boxed(),
                Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
            }
        })
        .flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

/// A versioned table whose current manifest cannot be resolved: plans, never
/// scans. See the module doc.
#[derive(Debug)]
pub struct PlaceholderProvider {
    table_name: String,
    version: i64,
    schema: SchemaRef,
}

impl PlaceholderProvider {
    pub fn new(table_name: String, version: i64, schema: SchemaRef) -> Self {
        Self {
            table_name,
            version,
            schema,
        }
    }
}

#[async_trait]
impl TableProvider for PlaceholderProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::External(Box::new(
            JammiError::VersionUnavailable {
                table: self.table_name.clone(),
                version: self.version,
            },
        )))
    }
}

/// Test-only rendezvous for the "object vanished mid-scan" oracle (§6.19):
/// while a table is armed, EVERY `MaskExec` of that table parks before it
/// opens its input stream, and all of them wake on one release — so a test
/// can delete a fragment object in a window no scan has crossed yet.
#[cfg(feature = "test-hooks")]
pub mod masked_scan_test_hooks {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, PoisonError};

    use tokio::sync::Notify;

    struct ScanRace {
        table_name: String,
        arrivals: AtomicUsize,
        parked_notify: Notify,
        release: Notify,
        released: AtomicBool,
    }

    static ARM: Mutex<Option<Arc<ScanRace>>> = Mutex::new(None);

    /// The test's handle: wait for the first scan to park, then release
    /// every parked scan. Dropping the handle releases and disarms.
    pub struct MaskedScanRace {
        state: Arc<ScanRace>,
    }

    /// Arm the race for `table_name`. Panics if another table is armed.
    pub fn arm_masked_scan_drain(table_name: &str) -> MaskedScanRace {
        let state = Arc::new(ScanRace {
            table_name: table_name.to_string(),
            arrivals: AtomicUsize::new(0),
            parked_notify: Notify::new(),
            release: Notify::new(),
            released: AtomicBool::new(false),
        });
        let mut guard = ARM.lock().unwrap_or_else(PoisonError::into_inner);
        assert!(
            guard.is_none(),
            "masked-scan test hook: a race is already armed; release it before arming again"
        );
        *guard = Some(Arc::clone(&state));
        MaskedScanRace { state }
    }

    impl MaskedScanRace {
        /// Wait (bounded to 5s) until at least one scan of the armed table
        /// has parked.
        pub async fn wait_parked(&self) {
            let notified = self.state.parked_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.state.arrivals.load(Ordering::SeqCst) > 0 {
                return;
            }
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), notified).await;
        }

        /// How many scans have parked so far.
        pub fn parked_count(&self) -> usize {
            self.state.arrivals.load(Ordering::SeqCst)
        }

        /// Release every parked scan and disarm (idempotent).
        pub fn release(&self) {
            self.state.released.store(true, Ordering::SeqCst);
            self.state.release.notify_waiters();
            let mut guard = ARM.lock().unwrap_or_else(PoisonError::into_inner);
            if guard.as_ref().is_some_and(|s| Arc::ptr_eq(s, &self.state)) {
                *guard = None;
            }
        }
    }

    impl Drop for MaskedScanRace {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// Park if a race is armed for `table_name` (a no-op otherwise, and a
    /// no-op once released).
    pub(super) async fn maybe_park_before_masked_scan_drain(table_name: &str) {
        let state = {
            let guard = ARM.lock().unwrap_or_else(PoisonError::into_inner);
            guard
                .as_ref()
                .filter(|s| s.table_name == table_name)
                .map(Arc::clone)
        };
        let Some(state) = state else {
            return;
        };
        // Register interest before announcing the arrival so a release that
        // races the announcement cannot be missed.
        let released = state.release.notified();
        tokio::pin!(released);
        released.as_mut().enable();
        state.arrivals.fetch_add(1, Ordering::SeqCst);
        state.parked_notify.notify_waiters();
        if state.released.load(Ordering::SeqCst) {
            return;
        }
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), released).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;

    fn plan() -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "_row_id",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    /// The unit assertion §6.20 pins: a fetch is never pushed into the mask.
    #[test]
    fn mask_exec_never_accepts_a_pushed_limit() {
        let mask = Arc::new(DeletionMask::from_entries([("a".to_string(), 0)]));
        let exec = Arc::new(MaskExec::new(plan(), 0, mask, 0, "t".into()));
        assert!(!exec.supports_limit_pushdown());
        assert!(exec.with_fetch(Some(1)).is_none());
    }
}
