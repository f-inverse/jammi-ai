//! `JammiCodec` — the `PhysicalExtensionCodec` that carries jammi's own
//! operators across a Ballista scheduler/executor boundary, delegating every
//! other node to Ballista's own codec (contract `feat_500-wave4` §2.2).
//!
//! Every buffer this codec WRITES starts with a 4-byte magic, `[0x07, b'J',
//! b'M', b'B']`. The first byte is deliberately an ILLEGAL prost tag: a
//! protobuf varint tag byte encodes `(field_number << 3) | wire_type`, field
//! numbers start at 1 and wire types run 0-5, so `0x07` (`field 0, wire type
//! 7`) can never be the first byte of ANY valid encoded protobuf message —
//! nothing Ballista's own codec ever writes can alias it. Pinned by
//! `tests/it/codec.rs`'s `ballista_oneof_tags_never_collide_with_magic`
//! against `BallistaPhysicalPlanNode`'s five oneof variants' first bytes
//! (`0x0A, 0x12, 0x1A, 0x22, 0x2A`,
//! `ballista-core-54.1.0/src/serde/generated/ballista.rs:31-54`), so an
//! upstream 6th variant landing on tag 7 (wire type 7 is still illegal, so
//! this can never actually happen, but the test makes that an EXECUTED
//! check, not an assumption) is caught.
//!
//! On decode, a buffer without the magic delegates whole to [`BallistaPhysicalExtensionCodec`]
//! (the ONLY way Ballista's own nodes cross the wire — its codec does not
//! delegate an unknown node, it returns a typed "Unsupported plan node"
//! error, `ballista-core-54.1.0/src/serde/mod.rs:709-730`). A magic-prefixed
//! buffer too short to carry a tag byte is a typed refusal of its own —
//! NEVER silently delegated (a delegate would misread the magic bytes as
//! Ballista's own prost varint tag and fail confusingly instead of naming
//! the truncation).
//!
//! Decode rebuilds each operator through its public constructor against the
//! DECODING process's own [`InferenceSession`] (a [`Weak`] reference — a
//! session outliving every plan built from it, never the reverse; a dead
//! `Weak` is a typed refusal naming the missing session, not a panic): the
//! model cache, result store, and DataFusion context are the decoding
//! session's, never serialized.

use std::sync::{Arc, Weak};

use datafusion::error::Result as DfResult;
use datafusion::execution::{FunctionRegistry, TaskContext};
use datafusion::logical_expr::ScalarUDF;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use prost::Message;

use ballista_core::serde::BallistaPhysicalExtensionCodec;

use jammi_ai::inference::adapter::DistributionForm;
use jammi_ai::model::{BackendType, ModelSource, ModelTask};
use jammi_ai::operator::ann_search_exec::AnnSearchExec;
use jammi_ai::operator::gang_exec::{GangDescriptor, GangExec};
use jammi_ai::operator::inference_exec::{InferenceExec, InferenceExecBuilder};
use jammi_ai::operator::key_check_exec::KeyCheckExec;
use jammi_ai::pipeline::asof::exec::AsofJoinExec;
use jammi_ai::pipeline::asof::spec::AsofJoinSpec;
use jammi_ai::session::InferenceSession;
use jammi_db::index::{validate_query, QuerySource};
use jammi_db::store::manifest::ComputeDeviceKind;
use jammi_db::TenantId;

use crate::error::Error;

#[allow(clippy::all, dead_code)]
mod pb {
    include!(concat!(env!("OUT_DIR"), "/jammi.ballista.v1.rs"));
}

/// The 4-byte magic every jammi-encoded buffer starts with: an illegal
/// prost tag byte (`0x07`, field 0 / wire type 7 — never legal) followed by
/// `JMB`. See the module doc for why this can never alias a Ballista buffer.
const MAGIC: [u8; 4] = [0x07, b'J', b'M', b'B'];

/// Node-type tag, the byte immediately after the magic.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeTag {
    Inference = 0,
    AnnSearch = 1,
    AsofJoin = 2,
    KeyCheck = 3,
    Gang = 4,
}

/// The codec `jammi-ballista`'s scheduler and executor roles both install.
pub struct JammiCodec {
    session: Weak<InferenceSession>,
    inner: BallistaPhysicalExtensionCodec,
}

impl std::fmt::Debug for JammiCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JammiCodec").finish_non_exhaustive()
    }
}

impl JammiCodec {
    /// Build a codec over a (weakly held) session. The session outlives
    /// every plan this codec decodes through it; a codec surviving its
    /// session is the typed-refusal case `try_decode` covers.
    pub fn new(session: &Arc<InferenceSession>) -> Self {
        Self {
            session: Arc::downgrade(session),
            inner: BallistaPhysicalExtensionCodec::default(),
        }
    }

    fn session(&self) -> Result<Arc<InferenceSession>, Error> {
        self.session.upgrade().ok_or(Error::SessionGone)
    }
}

/// Run an async catalog call from this synchronous trait-method call site.
/// `PhysicalExtensionCodec::try_decode` is not async (a fixed DataFusion/
/// Ballista trait signature), so a catalog re-read on decode must block the
/// calling worker thread. Requires a MULTI-THREADED tokio runtime (`Handle::
/// current()` inside `block_in_place` panics on a current-thread runtime) —
/// every jammi-server process runs one; a caller that does not is an
/// operational precondition this crate does not itself enforce (named in
/// this crate's contract file as a determinant, not silently assumed away).
fn block_on_catalog<F, T>(fut: F) -> Result<T, Error>
where
    F: std::future::Future<Output = jammi_db::error::Result<T>>,
{
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(fut))
        .map_err(Error::Catalog)
}

impl PhysicalExtensionCodec for JammiCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if buf.len() < 4 || buf[0..4] != MAGIC {
            return self.inner.try_decode(buf, inputs, ctx);
        }
        // Magic present: this buffer is ours. A missing tag byte is a
        // truncated buffer — refused typed, never delegated (delegating
        // would hand Ballista's codec our magic bytes as if they were its
        // own prost varint tag, a confusing foreign-looking failure instead
        // of naming the truncation).
        if buf.len() < 5 {
            return Err(Error::Decode(
                "truncated jammi operator buffer (magic, no tag byte)".into(),
            )
            .into_df_error());
        }
        let tag = buf[4];
        let body = &buf[5..];
        let session = self.session().map_err(Error::into_df_error)?;
        match tag {
            t if t == NodeTag::Inference as u8 => decode_inference(body, inputs, &session),
            t if t == NodeTag::AnnSearch as u8 => decode_ann_search(body, &session),
            t if t == NodeTag::AsofJoin as u8 => decode_asof(body, inputs),
            t if t == NodeTag::KeyCheck as u8 => decode_key_check(body, inputs),
            t if t == NodeTag::Gang as u8 => decode_gang(body),
            other => Err(Error::Decode(format!("unknown jammi node tag {other}")).into_df_error()),
        }
    }

    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> DfResult<()> {
        if let Some(exec) = node.downcast_ref::<InferenceExec>() {
            return encode_inference(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<AnnSearchExec>() {
            return encode_ann_search(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<AsofJoinExec>() {
            return encode_asof(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<KeyCheckExec>() {
            return encode_key_check(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<GangExec>() {
            return encode_gang(exec, buf);
        }
        // Not one of ours — delegate to Ballista's own codec (shuffle
        // reader/writer, unresolved shuffle, ...). A node NEITHER codec
        // knows (e.g. `MaskExec`, the named v1 cut) surfaces as the
        // delegate's own typed "Unsupported plan node" error naming it —
        // this codec adds no catch-all of its own.
        self.inner.try_encode(node, buf)
    }

    /// A jammi source scan's `_content_hash` projection (`build_embedding_
    /// plan`'s doc: "rides through to the sink as the table's fifth
    /// column") is a live `ScalarFunctionExpr` call to `jammi_content_hash`
    /// in the SUBMITTED physical plan, not a value already materialized
    /// before the wire — `datafusion-proto`'s own physical-expr encoding
    /// therefore asks THIS codec to encode/decode that (and every other
    /// jammi-registered) scalar UDF (`datafusion-proto-54.1.0/src/
    /// physical_plan/mod.rs:3859-3864`'s default `not_impl_err!` is what a
    /// codec that skips this override hits: "PhysicalExtensionCodec is not
    /// provided for scalar function …", discovered by executing
    /// `submit_physical_plan` of a `build_embedding_plan` plan end-to-end).
    /// Every jammi UDF is registered identically, by NAME, on every
    /// session (`InferenceSession::wrap`/`register_query_functions`), so
    /// there is nothing to serialize: encode writes zero bytes, decode
    /// looks the name up on the DECODING process's own session.
    fn try_encode_udf(&self, _node: &ScalarUDF, _buf: &mut Vec<u8>) -> DfResult<()> {
        Ok(())
    }

    fn try_decode_udf(&self, name: &str, _buf: &[u8]) -> DfResult<Arc<ScalarUDF>> {
        let session = self.session().map_err(Error::into_df_error)?;
        session.context().udf(name)
    }
}

fn to_json_string<T: serde::Serialize>(v: &T) -> DfResult<String> {
    serde_json::to_string(v).map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn from_json_str<T: serde::de::DeserializeOwned>(s: &str) -> DfResult<T> {
    serde_json::from_str(s).map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

/// `ComputeDeviceKind` <-> its canonical wire spelling. A plain match, not
/// `serde_json`: this field is compared byte-for-byte by `JammiExecutionEngine`
/// on every task, so it stays a bare string, never a quoted JSON scalar.
fn device_kind_str(kind: ComputeDeviceKind) -> &'static str {
    match kind {
        ComputeDeviceKind::Cpu => "cpu",
        ComputeDeviceKind::Cuda => "cuda",
        ComputeDeviceKind::Metal => "metal",
    }
}

fn device_kind_from_str(s: &str) -> DfResult<ComputeDeviceKind> {
    match s {
        "cpu" => Ok(ComputeDeviceKind::Cpu),
        "cuda" => Ok(ComputeDeviceKind::Cuda),
        "metal" => Ok(ComputeDeviceKind::Metal),
        other => Err(Error::Decode(format!("unknown device_kind '{other}'")).into_df_error()),
    }
}

fn encode_inference(exec: &InferenceExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let source = match exec.source() {
        ModelSource::HuggingFace(id) => pb::model_source::Source::HuggingFace(id.clone()),
        ModelSource::Local(path) => {
            pb::model_source::Source::Local(path.to_string_lossy().into_owned())
        }
    };
    let backend_json = exec.backend().map(|b| to_json_string(&b)).transpose()?;
    let regression_form_json = exec.regression_form().map(to_json_string).transpose()?;
    // The wire carries exactly the constructed value — the codec never
    // invents or rewrites a device kind (contract §9 B3).
    let device_kind = exec.device_kind();
    let msg = pb::InferenceExecNode {
        source: Some(pb::ModelSource {
            source: Some(source),
        }),
        task: exec.task().as_db_str().to_string(),
        content_columns: exec.content_columns().to_vec(),
        key_column: exec.key_column().to_string(),
        source_id: exec.source_id().to_string(),
        backend_json,
        batch_size: exec.batch_size() as u64,
        embedding_dim: exec.embedding_dim().map(|d| d as u64),
        regression_form_json,
        passthrough: exec.passthrough().to_vec(),
        device_kind: device_kind_str(device_kind).to_string(),
    };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::Inference as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_inference(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    session: &Arc<InferenceSession>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg = pb::InferenceExecNode::decode(body)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let input = inputs
        .first()
        .cloned()
        .ok_or_else(|| Error::Decode("InferenceExecNode: no input".into()).into_df_error())?;
    let source = match msg.source.and_then(|s| s.source) {
        Some(pb::model_source::Source::HuggingFace(id)) => ModelSource::hf(id),
        Some(pb::model_source::Source::Local(p)) => ModelSource::local(p),
        None => {
            return Err(Error::Decode("InferenceExecNode: missing source".into()).into_df_error())
        }
    };
    let task =
        ModelTask::try_from_db_str(&msg.task).map_err(|e| Error::Catalog(e).into_df_error())?;
    let backend = msg
        .backend_json
        .as_deref()
        .map(from_json_str::<BackendType>)
        .transpose()?;
    let regression_form = msg
        .regression_form_json
        .as_deref()
        .map(from_json_str::<DistributionForm>)
        .transpose()?;
    let node = InferenceExecBuilder::new(
        input,
        source,
        task,
        msg.content_columns,
        msg.key_column,
        msg.source_id,
        Arc::clone(session.model_cache()),
        device_kind_from_str(&msg.device_kind)?,
    )
    .batch_size(msg.batch_size as usize)
    .backend(backend)
    .embedding_dim(msg.embedding_dim.map(|d| d as usize))
    .regression_form(regression_form)
    .passthrough(msg.passthrough)
    .build()
    .map_err(|e| Error::Catalog(e).into_df_error())?;
    Ok(Arc::new(node))
}

fn encode_ann_search(exec: &AnnSearchExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let table = exec.table();
    let msg = pb::AnnSearchExecNode {
        table_name: table.table_name.clone(),
        tenant_id: table.tenant_id.clone(),
        query_vector: exec.query_vector().as_slice().to_vec(),
        k: exec.k() as u64,
        oversample_override: exec.oversample_override().map(|o| o as u64),
    };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::AnnSearch as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_ann_search(
    body: &[u8],
    session: &Arc<InferenceSession>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg = pb::AnnSearchExecNode::decode(body)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let tenant: Option<TenantId> = msg
        .tenant_id
        .map(TenantId::try_from)
        .transpose()
        .map_err(|e| Error::Catalog(e).into_df_error())?;
    let table = block_on_catalog(
        session
            .catalog()
            .get_result_table_for_tenant(&msg.table_name, tenant),
    )
    .map_err(Error::into_df_error)?
    .ok_or_else(|| {
        Error::Decode(format!("result table '{}' not found", msg.table_name)).into_df_error()
    })?;
    let query = validate_query(msg.query_vector, None, QuerySource::Caller)
        .map_err(|e| Error::Decode(format!("{e:?}")).into_df_error())?;
    let node = AnnSearchExec::new(
        table,
        query,
        msg.k as usize,
        msg.oversample_override.map(|o| o as usize),
        session.result_store(),
        session.context().clone(),
    )
    .map_err(|e| Error::Catalog(e).into_df_error())?;
    Ok(Arc::new(node))
}

fn encode_asof(exec: &AsofJoinExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let spec_json = serde_json::to_vec(exec.spec())
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let msg = pb::AsofJoinExecNode { spec_json };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::AsofJoin as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_asof(body: &[u8], inputs: &[Arc<dyn ExecutionPlan>]) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg = pb::AsofJoinExecNode::decode(body)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let spec: AsofJoinSpec = serde_json::from_slice(&msg.spec_json)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    if inputs.len() < 2 {
        return Err(Error::Decode(format!(
            "AsofJoinExecNode: expected 2 inputs, got {}",
            inputs.len()
        ))
        .into_df_error());
    }
    let node = AsofJoinExec::try_new(inputs[0].clone(), inputs[1].clone(), spec)
        .map_err(|e| Error::Decode(format!("{e:?}")).into_df_error())?;
    Ok(Arc::new(node))
}

fn encode_key_check(exec: &KeyCheckExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let msg = pb::KeyCheckExecNode {
        key_column: exec.key_column().to_string(),
    };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::KeyCheck as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_key_check(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg = pb::KeyCheckExecNode::decode(body)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let input = inputs
        .first()
        .cloned()
        .ok_or_else(|| Error::Decode("KeyCheckExecNode: no input".into()).into_df_error())?;
    let node = KeyCheckExec::try_new(input, &msg.key_column)?;
    Ok(Arc::new(node))
}

fn encode_gang(exec: &GangExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let d = exec.descriptor();
    let msg = pb::GangExecNode {
        job_id: d.job_id.clone(),
        attempt: d.attempt,
        world: d.world,
        submitter: d.submitter.clone(),
    };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::Gang as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_gang(body: &[u8]) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg =
        pb::GangExecNode::decode(body).map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let descriptor = GangDescriptor {
        job_id: msg.job_id,
        attempt: msg.attempt,
        world: msg.world,
        submitter: msg.submitter,
    };
    Ok(Arc::new(GangExec::new(descriptor)))
}
