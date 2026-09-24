//! The operators' wire form: what a `PhysicalExtensionCodec` writes for an
//! [`InferenceExec`] or a [`NumberedInputExec`] and rebuilds the node from
//! in another process. A codec that carries other nodes composes these
//! encoders under its own framing; only the descriptor crosses, and the
//! decoded node binds to the decoding process's own [`InferenceRuntime`].

use std::num::NonZeroUsize;
use std::sync::Arc;

use datafusion::physical_plan::ExecutionPlan;
use jammi_numerics::ChunkBudget;
use prost::Message;

use crate::device::ComputeDeviceKind;
use crate::error::{Error, Result};
use crate::inference::adapter::DistributionForm;
use crate::inference::exec::InferenceExec;
use crate::inference::numbered::NumberedInputExec;
use crate::inference::runtime::InferenceRuntime;
use crate::inference::spec::{InferenceSpec, RowOrder};
use crate::source::ModelSource;
use crate::task::ModelTask;

/// The crate's wire package (`jammi.inference.v1`, compiled by this crate's
/// `build.rs`). Public so a codec's own tests can construct a descriptor
/// the encoders never would and prove the decode refuses it.
#[allow(clippy::all, dead_code, missing_docs)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/jammi.inference.v1.rs"));
}

/// `spec` as the wire descriptor both nodes carry.
pub fn spec_to_proto(spec: &InferenceSpec) -> Result<pb::InferenceExecNode> {
    let source = match &spec.source {
        ModelSource::HuggingFace(id) => pb::model_source::Source::HuggingFace(id.clone()),
        ModelSource::Local(path) => {
            pb::model_source::Source::Local(path.to_string_lossy().into_owned())
        }
        ModelSource::Remote(name) => pb::model_source::Source::Remote(name.clone()),
    };
    Ok(pb::InferenceExecNode {
        source: Some(pb::ModelSource {
            source: Some(source),
        }),
        task: spec.task.as_str().to_string(),
        content_columns: spec.content_columns.clone(),
        key_column: spec.key_column.clone(),
        source_id: spec.source_id.clone(),
        batch_size: spec.chunk.rows.get() as u64,
        batch_tokens: spec.chunk.tokens.get() as u64,
        embedding_dim: spec.embedding_dim.map(|d| d as u64),
        regression_form_json: spec
            .regression_form
            .as_ref()
            .map(|form| serde_json::to_string(form).map_err(|e| Error::Decode(e.to_string())))
            .transpose()?,
        passthrough: spec.passthrough.clone(),
        // The wire carries exactly the constructed value — the codec never
        // invents or rewrites a device kind.
        device_kind: spec.device_kind.wire_str().to_string(),
        partitions: spec.partitions.get() as u64,
    })
}

/// A wire count that must be at least one.
fn non_zero(field: &str, value: u64) -> Result<NonZeroUsize> {
    usize::try_from(value)
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or_else(|| {
            Error::Decode(format!(
                "InferenceExecNode: {field} = {value} is not a count >= 1"
            ))
        })
}

/// The spec a wire descriptor carries.
pub fn spec_from_proto(msg: pb::InferenceExecNode) -> Result<InferenceSpec> {
    let source = match msg.source.and_then(|s| s.source) {
        Some(pb::model_source::Source::HuggingFace(id)) => ModelSource::hf(id),
        Some(pb::model_source::Source::Local(p)) => ModelSource::local(p),
        Some(pb::model_source::Source::Remote(name)) => ModelSource::remote(name),
        None => return Err(Error::Decode("InferenceExecNode: missing source".into())),
    };
    Ok(InferenceSpec {
        source,
        task: ModelTask::parse(&msg.task)?,
        content_columns: msg.content_columns,
        key_column: msg.key_column,
        source_id: msg.source_id,
        chunk: ChunkBudget {
            rows: non_zero("batch_size", msg.batch_size)?,
            tokens: non_zero("batch_tokens", msg.batch_tokens)?,
        },
        embedding_dim: msg.embedding_dim.map(|d| d as usize),
        regression_form: msg
            .regression_form_json
            .as_deref()
            .map(|json| {
                serde_json::from_str::<DistributionForm>(json)
                    .map_err(|e| Error::Decode(format!("regression_form: {e}")))
            })
            .transpose()?,
        passthrough: msg.passthrough,
        device_kind: ComputeDeviceKind::parse(&msg.device_kind)
            .ok_or_else(|| Error::UnknownDeviceKind(msg.device_kind.clone()))?,
        partitions: non_zero("partitions", msg.partitions)?,
    })
}

/// Encode an [`InferenceExec`]'s descriptor into `buf`.
pub fn encode_inference(exec: &InferenceExec, buf: &mut Vec<u8>) -> Result<()> {
    spec_to_proto(exec.spec())?
        .encode(buf)
        .map_err(|e| Error::Decode(e.to_string()))
}

/// Rebuild an [`InferenceExec`] from `body` over its one input, bound to
/// `runtime`.
pub fn decode_inference(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    runtime: InferenceRuntime,
) -> Result<Arc<dyn ExecutionPlan>> {
    let msg = pb::InferenceExecNode::decode(body).map_err(|e| Error::Decode(e.to_string()))?;
    let input = inputs
        .first()
        .cloned()
        .ok_or_else(|| Error::Decode("InferenceExecNode: no input".into()))?;
    // The same constructor `with_new_children` uses, bound to the DECODING
    // process's runtime.
    let exec = InferenceExec::bind(input, spec_from_proto(msg)?, runtime)
        .map_err(|e| Error::Decode(e.to_string()))?;
    Ok(Arc::new(exec))
}

/// Encode a [`NumberedInputExec`]'s descriptor into `buf`.
pub fn encode_numbered_input(exec: &NumberedInputExec, buf: &mut Vec<u8>) -> Result<()> {
    let (key_column, tie_breakers) = match exec.order() {
        RowOrder::Keyed {
            key_column,
            tie_breakers,
        } => (Some(key_column.clone()), tie_breakers.clone()),
        RowOrder::Arrival => (None, Vec::new()),
    };
    pb::NumberedInputExecNode {
        key_column,
        tie_breakers,
        spec: Some(spec_to_proto(exec.spec())?),
    }
    .encode(buf)
    .map_err(|e| Error::Decode(e.to_string()))
}

/// Rebuild a [`NumberedInputExec`] from `body` over its one input, bound
/// to `runtime`.
pub fn decode_numbered_input(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    runtime: InferenceRuntime,
) -> Result<Arc<dyn ExecutionPlan>> {
    let msg = pb::NumberedInputExecNode::decode(body).map_err(|e| Error::Decode(e.to_string()))?;
    let input = inputs
        .first()
        .cloned()
        .ok_or_else(|| Error::Decode("NumberedInputExecNode: no input".into()))?;
    let order = match msg.key_column {
        Some(key_column) => RowOrder::Keyed {
            key_column,
            tie_breakers: msg.tie_breakers,
        },
        None => RowOrder::Arrival,
    };
    let spec = msg
        .spec
        .ok_or_else(|| Error::Decode("NumberedInputExecNode: missing spec".into()))?;
    let exec = NumberedInputExec::try_new(input, order, spec_from_proto(spec)?, runtime)
        .map_err(|e| Error::Decode(e.to_string()))?;
    Ok(Arc::new(exec))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every model source crosses the wire as itself: a remote model's name
    /// is decoded as a remote source, for the decoding process to resolve
    /// against its own declaration — never as a Hub id it would fetch.
    #[test]
    fn every_model_source_round_trips_through_the_descriptor() {
        for source in [
            ModelSource::hf("owner/encoder"),
            ModelSource::local("/models/encoder"),
            ModelSource::remote("hosted-encoder"),
        ] {
            let spec = InferenceSpec {
                source: source.clone(),
                task: ModelTask::TextEmbedding,
                content_columns: vec!["text".into()],
                key_column: "id".into(),
                source_id: "src".into(),
                chunk: ChunkBudget {
                    rows: NonZeroUsize::new(32).unwrap(),
                    tokens: NonZeroUsize::new(16384).unwrap(),
                },
                embedding_dim: Some(384),
                regression_form: None,
                passthrough: vec![],
                device_kind: ComputeDeviceKind::Cpu,
                partitions: NonZeroUsize::new(1).unwrap(),
            };
            let decoded = spec_from_proto(spec_to_proto(&spec).unwrap()).unwrap();
            assert_eq!(decoded.source, source);
        }
    }
}
