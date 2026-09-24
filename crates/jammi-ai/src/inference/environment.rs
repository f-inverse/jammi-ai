//! What a sink this process runs records as having produced a
//! materialization's bytes: its device and the models the plan ran.

use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use jammi_datafusion::inference_specs;
use jammi_db::store::manifest::{ComputeDevice, MaterializationEnv};

use crate::model::cache::ModelCache;

/// The environment a process running `InferenceExec` nodes produces a
/// materialization in: its compute device, and the identity of every model
/// the plan's inference nodes ran, read from this process's own model cache —
/// so a table records the models and the device that produced it wherever
/// its plan was placed.
pub struct InferenceEnvironment {
    /// The device this process runs models on.
    pub device: ComputeDevice,
    /// Where this process's models are resident.
    pub model_cache: Arc<ModelCache>,
}

#[async_trait::async_trait]
impl jammi_db::store::sink::ProducingEnvironment for InferenceEnvironment {
    async fn of(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> jammi_db::error::Result<MaterializationEnv> {
        let mut models = Vec::new();
        for spec in inference_specs(plan) {
            let guard = self
                .model_cache
                .get_or_load(&spec.source, spec.task)
                .await?;
            let identity = guard.model.description().identity();
            if !models.contains(&identity) {
                models.push(identity);
            }
        }
        Ok(MaterializationEnv::of_models(self.device.clone(), models))
    }
}
