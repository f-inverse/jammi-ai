//! `JammiCodec` — the `PhysicalExtensionCodec` that carries jammi's own
//! operators across a Ballista scheduler/executor boundary, delegating every
//! other node to Ballista's own codec.
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
//!
//! A partitioned inference plan (`jammi_datafusion::plan_inference`)
//! crosses whole: `InferenceExec` and `NumberedInputExec` are
//! `jammi-datafusion`'s, their wire forms that crate's own
//! (`jammi_datafusion::inference::wire`) under this codec's framing, and the exchange,
//! merge and coalesce between them are stock operators `datafusion-proto`
//! already carries. Ballista cuts a stage
//! at each of those three, so the numbered input runs as one task, the
//! `N`-partition `InferenceExec` as `N` tasks, and the merge as one.

use std::sync::{Arc, Weak};

use datafusion::error::DataFusionError;
use datafusion::error::Result as DfResult;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::ScalarUDF;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use prost::Message;

use ballista_core::serde::BallistaPhysicalExtensionCodec;

use jammi_ai::operator::vector_search_exec::VectorSearchExec;
use jammi_ai::pipeline::asof::exec::AsofJoinExec;
use jammi_ai::pipeline::asof::spec::AsofJoinSpec;
use jammi_ai::pipeline::graph_propagation::hop::{HopFoldExec, HopSpec};
use jammi_ai::pipeline::graph_propagation::readout::{ReadoutExec, ReadoutSpec};
use jammi_ai::pipeline::graph_propagation::state::{InitialStateExec, InitialStateSpec};
use jammi_ai::session::InferenceSession;
use jammi_datafusion::inference::key_check::KeyCheckExec;
use jammi_datafusion::InferenceExec;
use jammi_datafusion::NumberedInputExec;
use jammi_datafusion::{NoTrainingRunner, TrainingExec, TrainingRunner};
use jammi_db::error::JammiError;
use jammi_db::index::{FiniteQuery, QuerySource, SearchMethod};
use jammi_db::store::{ResultTableSinkExec, ResultTableSinkSpec};
use jammi_db::TenantId;

use crate::error::Error;

/// The crate's own wire package (`jammi.ballista.v1`, compiled by this
/// crate's `build.rs`; NOT part of the frozen `jammi.v1` surface). Public so
/// an oracle can construct a descriptor the codec never would — a plan
/// naming another tenant's table — and prove the decode refuses it.
#[allow(clippy::all, dead_code, missing_docs)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/jammi.ballista.v1.rs"));
}

/// The 4-byte magic every jammi-encoded buffer starts with: an illegal
/// prost tag byte (`0x07`, field 0 / wire type 7 — never legal) followed by
/// `JMB`. See the module doc for why this can never alias a Ballista buffer.
pub const MAGIC: [u8; 4] = [0x07, b'J', b'M', b'B'];

/// Node-type tag, the byte immediately after the magic.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeTag {
    Inference = 0,
    VectorSearch = 1,
    AsofJoin = 2,
    KeyCheck = 3,
    Training = 4,
    NumberedInput = 5,
    ResultTableSink = 6,
    InitialState = 7,
    HopFold = 8,
    Readout = 9,
}

/// The codec `jammi-ballista`'s scheduler and executor roles both install.
pub struct JammiCodec {
    session: Weak<InferenceSession>,
    /// What a decoded [`TrainingExec`] runs its job through: the executor
    /// role's runner on an executor, [`NoTrainingRunner`] everywhere else.
    training: Arc<dyn TrainingRunner>,
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
    ///
    /// A training stage this codec decodes runs no job: the codec binds
    /// [`NoTrainingRunner`] until [`Self::running_training`] names the
    /// process's runner.
    pub fn new(session: &Arc<InferenceSession>) -> Self {
        Self {
            session: Arc::downgrade(session),
            training: Arc::new(NoTrainingRunner),
            inner: BallistaPhysicalExtensionCodec::default(),
        }
    }

    /// This codec, binding every training stage it decodes to `runner` —
    /// the executor role's, the one process kind that runs training jobs.
    pub fn running_training(self, runner: Arc<dyn TrainingRunner>) -> Self {
        Self {
            training: runner,
            ..self
        }
    }

    fn session(&self) -> Result<Arc<InferenceSession>, Error> {
        self.session.upgrade().ok_or(Error::SessionGone)
    }
}

/// Run an async catalog-backed call (a row re-read, the width authority a
/// row resolves to) from this synchronous trait-method call site.
/// `PhysicalExtensionCodec::try_decode` is not async (a fixed DataFusion/
/// Ballista trait signature), so a catalog re-read on decode must block the
/// calling worker thread. Requires a MULTI-THREADED tokio runtime (`Handle::
/// current()` inside `block_in_place` panics on a current-thread runtime) —
/// every jammi-server process runs one; a caller that does not is an
/// operational precondition this crate does not itself enforce.
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
            t if t == NodeTag::VectorSearch as u8 => decode_vector_search(body, &session),
            t if t == NodeTag::AsofJoin as u8 => decode_asof(body, inputs),
            t if t == NodeTag::KeyCheck as u8 => decode_key_check(body, inputs),
            t if t == NodeTag::Training as u8 => decode_training(body, &self.training),
            t if t == NodeTag::NumberedInput as u8 => decode_numbered_input(body, inputs, &session),
            t if t == NodeTag::ResultTableSink as u8 => {
                decode_result_table_sink(body, inputs, &session)
            }
            t if t == NodeTag::InitialState as u8 => {
                decode_one_child::<pb::InitialStateExecNode, InitialStateSpec, _>(
                    body,
                    inputs,
                    "InitialStateExecNode",
                    |input, spec| Ok(Arc::new(InitialStateExec::try_new(input, spec)?)),
                )
            }
            t if t == NodeTag::HopFold as u8 => {
                decode_one_child::<pb::HopFoldExecNode, HopSpec, _>(
                    body,
                    inputs,
                    "HopFoldExecNode",
                    |input, spec| Ok(Arc::new(HopFoldExec::try_new(input, spec)?)),
                )
            }
            t if t == NodeTag::Readout as u8 => {
                decode_one_child::<pb::ReadoutExecNode, ReadoutSpec, _>(
                    body,
                    inputs,
                    "ReadoutExecNode",
                    |input, spec| Ok(Arc::new(ReadoutExec::try_new(input, spec)?)),
                )
            }
            other => Err(Error::Decode(format!("unknown jammi node tag {other}")).into_df_error()),
        }
    }

    fn try_encode(&self, node: Arc<dyn ExecutionPlan>, buf: &mut Vec<u8>) -> DfResult<()> {
        if let Some(exec) = node.downcast_ref::<InferenceExec>() {
            return encode_inference(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<VectorSearchExec>() {
            return encode_vector_search(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<AsofJoinExec>() {
            return encode_asof(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<KeyCheckExec>() {
            return encode_key_check(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<TrainingExec>() {
            return encode_training(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<NumberedInputExec>() {
            return encode_numbered_input(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<ResultTableSinkExec>() {
            return encode_result_table_sink(exec, buf);
        }
        if let Some(exec) = node.downcast_ref::<InitialStateExec>() {
            return encode_spec(NodeTag::InitialState, exec.spec(), buf, |spec_json| {
                pb::InitialStateExecNode { spec_json }
            });
        }
        if let Some(exec) = node.downcast_ref::<HopFoldExec>() {
            return encode_spec(NodeTag::HopFold, exec.spec(), buf, |spec_json| {
                pb::HopFoldExecNode { spec_json }
            });
        }
        if let Some(exec) = node.downcast_ref::<ReadoutExec>() {
            return encode_spec(NodeTag::Readout, exec.spec(), buf, |spec_json| {
                pb::ReadoutExecNode { spec_json }
            });
        }
        // Not one of ours — delegate to Ballista's own codec (shuffle
        // reader/writer, unresolved shuffle, ...). A node NEITHER codec
        // knows (e.g. `MaskExec`, a named v1 cut) surfaces as the delegate's
        // own typed "Unsupported plan node" error naming it — this codec
        // adds no catch-all of its own.
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
    /// session (`InferenceSession::wrap`/`install_query_functions`), so
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

/// The inference operators' wire forms are `jammi-datafusion`'s own
/// ([`jammi_datafusion::inference::wire`]); this codec frames them under its magic and
/// tag and binds a decoded node to the decoding session's runtime.
fn encode_inference(exec: &InferenceExec, buf: &mut Vec<u8>) -> DfResult<()> {
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::Inference as u8);
    jammi_datafusion::inference::wire::encode_inference(exec, buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_inference(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    session: &Arc<InferenceSession>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    jammi_datafusion::inference::wire::decode_inference(body, inputs, session.inference_runtime())
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn encode_numbered_input(exec: &NumberedInputExec, buf: &mut Vec<u8>) -> DfResult<()> {
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::NumberedInput as u8);
    jammi_datafusion::inference::wire::encode_numbered_input(exec, buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_numbered_input(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    session: &Arc<InferenceSession>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    jammi_datafusion::inference::wire::decode_numbered_input(
        body,
        inputs,
        session.inference_runtime(),
    )
    .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn encode_vector_search(exec: &VectorSearchExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let table = exec.table();
    let msg = pb::VectorSearchExecNode {
        table_name: table.table_name.clone(),
        tenant_id: table.tenant_id.clone(),
        query_vector: exec.query_vector().as_slice().to_vec(),
        k: exec.k() as u64,
        method: match exec.method() {
            SearchMethod::Approximate { oversample: None } => None,
            SearchMethod::Approximate {
                oversample: Some(o),
            } => Some(pb::vector_search_exec_node::Method::Oversample(o as u64)),
            SearchMethod::Exact => Some(pb::vector_search_exec_node::Method::Exact(
                pb::ExactSearch {},
            )),
        },
        query_stored_table: match exec.query_vector().source() {
            QuerySource::Caller => None,
            QuerySource::Stored { table } => Some(table.clone()),
        },
    };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::VectorSearch as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_vector_search(
    body: &[u8],
    session: &Arc<InferenceSession>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg = pb::VectorSearchExecNode::decode(body)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    // Every typed refusal below boxes the `JammiError` payload directly as
    // `DataFusionError::External`'s inner value (never this crate's own
    // `Error`, which `Error::into_df_error` would box instead, hiding the
    // `JammiError` a downcast needs) — so a caller inspecting the returned
    // `DataFusionError` IN THIS PROCESS (a same-process test, or a future
    // in-process consumer of this codec) recovers the typed variant via
    // `jammi_db::error`'s structural `DataFusionError` -> `JammiError`
    // classifier rather than the lossy catch-all `JammiError::DataFusion(e)`.
    //
    // This does NOT reach a client across a real distributed Ballista job.
    // `ballista-executor`'s task loop stringifies a failed task's error
    // (`e.to_string()`, `ballista-executor-54.1.0/src/lib.rs:138`) into
    // `TaskStatus`, and `ballista-core`'s job-status waiter does the same
    // for the job as a whole, rebuilding it as a bare
    // `DataFusionError::Execution(String)`
    // (`ballista-core-54.1.0/src/execution_plans/distributed_query.rs:519,685`)
    // — no boxed value survives that hop, so the classifier finds no typed
    // payload in the chain and falls to `JammiError::DataFusion(e)`, which
    // `jammi-server/src/grpc/wire.rs`'s classifier's catch-all
    // (`other => (Code::Internal, ..)` in `map_engine_error`) maps to `Internal`,
    // never `InvalidArgument`, regardless of which typed variant was boxed
    // here. The caller-class path for a REMOTE client is the coordinator's
    // own `QueryBuilder::new` check, which runs before any plan is ever
    // shipped; every check in this function is decode-time defense in depth
    // for a peer that skipped it, recoverable in-process but not over the
    // wire.
    let typed = |e: JammiError| DataFusionError::External(Box::new(e));
    let typed_lookup = |e: Error| match e {
        Error::Catalog(j) => typed(j),
        other => other.into_df_error(),
    };
    let tenant: Option<TenantId> = msg
        .tenant_id
        .map(TenantId::try_from)
        .transpose()
        .map_err(typed)?;
    let table = block_on_catalog(
        session
            .catalog()
            .get_result_table_for_tenant(&msg.table_name, tenant),
    )
    .map_err(typed_lookup)?
    .ok_or_else(|| {
        typed(JammiError::Other(format!(
            "result table '{}' not found",
            msg.table_name
        )))
    })?;
    // A decoded query is a fresh entry on THIS process: it is re-validated
    // with the provenance it was encoded with, against the same authority
    // every entry over this table uses (`ResultStore::query_width` — the
    // live catalog row's recorded width, else the artifact a search of it
    // reads). A refusal is boxed as the `JammiError` the validation `From`
    // impl classifies it into (`Schema` for a caller's vector, the table's
    // corrupt-artifact class for a stored one) — recoverable in-process
    // exactly as described above.
    let source = msg
        .query_stored_table
        .map_or(QuerySource::Caller, |table| QuerySource::Stored { table });
    let query = FiniteQuery::new(msg.query_vector, source).map_err(|e| typed(e.into()))?;
    let width = block_on_catalog(
        session
            .result_store()
            .query_width(session.context(), &table),
    )
    .map_err(typed_lookup)?;
    let query = query
        .against_authority(width)
        .map_err(|e| typed(e.into()))?;
    let node = VectorSearchExec::new(
        table,
        query,
        msg.k as usize,
        match msg.method {
            None => SearchMethod::default(),
            Some(pb::vector_search_exec_node::Method::Oversample(o)) => SearchMethod::Approximate {
                oversample: Some(o as usize),
            },
            Some(pb::vector_search_exec_node::Method::Exact(_)) => SearchMethod::Exact,
        },
        session.result_store(),
        session.context().clone(),
    )
    .map_err(typed)?;
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

/// Encode a node carried by its `serde` spec alone under `tag`: the spec as
/// JSON inside the message `wrap` builds.
fn encode_spec<M: prost::Message>(
    tag: NodeTag,
    spec: &impl serde::Serialize,
    buf: &mut Vec<u8>,
    wrap: impl FnOnce(Vec<u8>) -> M,
) -> DfResult<()> {
    let spec_json =
        serde_json::to_vec(spec).map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    buf.extend_from_slice(&MAGIC);
    buf.push(tag as u8);
    wrap(spec_json)
        .encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

/// The spec bytes a spec-only message carries.
trait SpecJson {
    fn spec_json(&self) -> &[u8];
}

impl SpecJson for pb::InitialStateExecNode {
    fn spec_json(&self) -> &[u8] {
        &self.spec_json
    }
}

impl SpecJson for pb::HopFoldExecNode {
    fn spec_json(&self) -> &[u8] {
        &self.spec_json
    }
}

impl SpecJson for pb::ReadoutExecNode {
    fn spec_json(&self) -> &[u8] {
        &self.spec_json
    }
}

/// Decode a one-child node carried by its `serde` spec alone: the message
/// `M`, its spec `S`, and the node `build` binds over the first input.
fn decode_one_child<M, S, B>(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    name: &str,
    build: B,
) -> DfResult<Arc<dyn ExecutionPlan>>
where
    M: prost::Message + Default + SpecJson,
    S: serde::de::DeserializeOwned,
    B: FnOnce(Arc<dyn ExecutionPlan>, S) -> DfResult<Arc<dyn ExecutionPlan>>,
{
    let msg = M::decode(body).map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let spec: S = serde_json::from_slice(msg.spec_json())
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let input = inputs
        .first()
        .cloned()
        .ok_or_else(|| Error::Decode(format!("{name}: no input")).into_df_error())?;
    build(input, spec)
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

fn encode_training(exec: &TrainingExec, buf: &mut Vec<u8>) -> DfResult<()> {
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::Training as u8);
    jammi_datafusion::training::wire::encode_training(exec, buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

fn decode_training(
    body: &[u8],
    runner: &Arc<dyn TrainingRunner>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    jammi_datafusion::training::wire::decode_training(body, Arc::clone(runner))
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

/// The sink crosses as its spec; the node that arrives is PLACED — it
/// writes on the process that decodes it and never re-submits.
fn encode_result_table_sink(exec: &ResultTableSinkExec, buf: &mut Vec<u8>) -> DfResult<()> {
    let msg = pb::ResultTableSinkExecNode {
        spec_json: to_json_string(exec.spec())?,
    };
    buf.extend_from_slice(&MAGIC);
    buf.push(NodeTag::ResultTableSink as u8);
    msg.encode(buf)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())
}

/// Rebuild the sink over the DECODING session's own result store
/// (`ResultStore::adopt_placed_sink`): a spec whose object is not under
/// this store's root, or whose row this catalog does not hold, is refused
/// typed — boxed as the `JammiError` it is, for the same reason
/// `decode_ann_search` boxes its refusals.
fn decode_result_table_sink(
    body: &[u8],
    inputs: &[Arc<dyn ExecutionPlan>],
    session: &Arc<InferenceSession>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let msg = pb::ResultTableSinkExecNode::decode(body)
        .map_err(|e| Error::Decode(e.to_string()).into_df_error())?;
    let input = inputs
        .first()
        .cloned()
        .ok_or_else(|| Error::Decode("ResultTableSinkExecNode: no input".into()).into_df_error())?;
    let spec: ResultTableSinkSpec = from_json_str(&msg.spec_json)?;
    let node = block_on_catalog(session.result_store().adopt_placed_sink(spec, input)).map_err(
        |e| match e {
            Error::Catalog(typed) => DataFusionError::External(Box::new(typed)),
            other => other.into_df_error(),
        },
    )?;
    Ok(Arc::new(node))
}
