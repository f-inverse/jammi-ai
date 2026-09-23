//! The engine's model runtime: how `jammi-inference`'s operators bind and
//! run a model in this process. The model cache binds a model
//! ([`ModelRuntime`]), and a bound model is the cache's guard over it
//! ([`BoundModel`]) — resident for as long as the operator holds it,
//! its forwards admitted by the device the guard names.

use std::sync::Arc;

use arrow::array::ArrayRef;
use async_trait::async_trait;
use jammi_inference::adapter::DistributionForm;
use jammi_inference::{
    BackendOutput, BoundModel, ForwardError, ForwardPermit, ModelRuntime, ModelSource, ModelTask,
    Prepared,
};
use jammi_numerics::ShapeLadder;

use crate::model::cache::ModelCache;
use crate::model::oom::is_oom_message;
use crate::model::{ModelGuard, PreparedInput};

#[async_trait]
impl ModelRuntime for ModelCache {
    async fn bind(
        &self,
        source: &ModelSource,
        task: ModelTask,
    ) -> jammi_inference::Result<Arc<dyn BoundModel>> {
        let guard = self
            .get_or_load(source, task, None)
            .await
            .map_err(jammi_inference::Error::runtime)?;
        Ok(Arc::new(guard))
    }
}

/// A prepared chunk that is not this backend's: the operators only ever
/// hand back what [`BoundModel::prepare`] gave them, so this names a
/// runtime that mixed two models' halves.
fn foreign_prepared() -> ForwardError {
    ForwardError::Other(jammi_inference::Error::Inference(
        "a prepared chunk of another runtime reached the candle backend".into(),
    ))
}

#[async_trait]
impl BoundModel for ModelGuard {
    fn embedding_dim(&self) -> usize {
        self.model.description().embedding_dim()
    }

    fn regression_form(&self) -> Option<&DistributionForm> {
        self.model.description().regression_form()
    }

    fn regression_std_scale(&self) -> Option<f32> {
        self.model.regression_std_scale()
    }

    fn row_costs(
        &self,
        content: &[ArrayRef],
        task: ModelTask,
    ) -> jammi_inference::Result<Vec<u32>> {
        self.model
            .row_costs(content, task)
            .map_err(jammi_inference::Error::runtime)
    }

    fn shape_ladder(&self, task: ModelTask) -> jammi_inference::Result<ShapeLadder> {
        self.model
            .shape_ladder(task)
            .map_err(jammi_inference::Error::runtime)
    }

    fn prepare(&self, content: &[ArrayRef], task: ModelTask) -> jammi_inference::Result<Prepared> {
        let prepared = self
            .model
            .prepare(content, task)
            .map_err(jammi_inference::Error::runtime)?;
        Ok(Box::new(prepared))
    }

    async fn admit_forward(&self) -> jammi_inference::Result<ForwardPermit> {
        let permit = self
            .device()
            .admit_forward()
            .await
            .map_err(jammi_inference::Error::runtime)?;
        Ok(ForwardPermit::new(permit))
    }

    /// The device operation, with its failure classified here — the one
    /// place the engine's OOM spelling table (`model::oom`) meets the
    /// operators' typed recovery: a genuine out-of-memory failure takes
    /// the runner's batch-halving retry, every other failure propagates.
    fn forward(&self, prepared: Prepared) -> Result<BackendOutput, ForwardError> {
        let prepared = prepared
            .downcast::<PreparedInput>()
            .map_err(|_| foreign_prepared())?;
        self.model.forward_prepared(*prepared).map_err(|e| {
            if is_oom_message(&e.to_string().to_lowercase()) {
                ForwardError::OutOfMemory(e.to_string())
            } else {
                ForwardError::Other(jammi_inference::Error::runtime(e))
            }
        })
    }
}
