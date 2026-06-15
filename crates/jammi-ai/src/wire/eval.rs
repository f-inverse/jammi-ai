//! `EvalService` request decoders: the embedding-retrieval, per-query-readback,
//! inference, compare, and calibration request shapes the embedded binding and
//! the gRPC handler share.
//!
//! The transport-neutral eval-task / calibration-shape / cohort-tag conversions
//! live on the wire substrate ([`jammi_wire`]) and are reused here, not
//! reimplemented; what stays in the engine crate are the request decoders that
//! return the engine's flat eval call args (`EvalEmbeddingsArgs`,
//! `EvalInferenceArgs`, `EvalCompareArgs`, `EvalCalibrationArgs`).
//!
//! The embedded binding builds each request with the same pure-Python assembly
//! the remote client uses, serializes it, and hands the bytes here — so the
//! in-process and remote eval paths decode the request through one shared seam.
//! Only the request is collapsed: the report responses stay transport-specific
//! (the gRPC handler encodes wire report messages, the embedded binding
//! serializes the report struct into a Python dict), so the response shapes are
//! untouched. The required-identity and task/shape validation runs at decode,
//! matching the gRPC handler's edge checks.

use std::collections::{BTreeMap, HashMap};

use prost::Message;
use tonic::Status;

use crate::eval::{EvalCalibrationShape, EvalTask};
use jammi_wire::proto::eval as pb;
use jammi_wire::{calibration_shape_from_proto, cohorts_from_proto, EvalTaskFromWire};

/// The engine's `query_id → {key: value}` cohort map. The substrate never
/// interprets these tags; the decode rebuilds the map verbatim.
type Cohorts = HashMap<String, BTreeMap<String, String>>;

// ─── EvalEmbeddings ──────────────────────────────────────────────────────────

/// The decoded retrieval-eval target a `EvalEmbeddings` request carries. The
/// engine method (`Session::eval_embeddings`) takes the source, the optional
/// embedding table, the golden source, the `k` cap, and the cohort map
/// separately, so the decode returns them as a struct the binding destructures.
/// `embedding_table` resolves the empty-string sentinel to `None` at decode (the
/// engine's "use the most recent table" marker).
pub struct EvalEmbeddingsArgs {
    pub source_id: String,
    pub embedding_table: Option<String>,
    pub golden_source: String,
    pub k: usize,
    pub cohorts: Cohorts,
}

/// Decode a serialized [`pb::EvalEmbeddingsRequest`] body into the engine
/// [`EvalEmbeddingsArgs`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote eval-embeddings paths decode through
/// one shared seam ([`eval_embeddings_from_proto`]). A body that is not a valid
/// `EvalEmbeddingsRequest` is a client error (`InvalidArgument`).
pub fn eval_embeddings_from_bytes(body: &[u8]) -> Result<EvalEmbeddingsArgs, Status> {
    let req = pb::EvalEmbeddingsRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed EvalEmbeddings request: {e}")))?;
    eval_embeddings_from_proto(req)
}

/// Decode a [`pb::EvalEmbeddingsRequest`] into the engine [`EvalEmbeddingsArgs`].
/// The required `source_id` / `golden_source` are validated at decode; the
/// `embedding_table` empty-string sentinel resolves to `None`; the `k` cap
/// widens to the engine's `usize`; and the cohort tags rebuild through the shared
/// [`cohorts_from_proto`] — matching the gRPC handler's edge checks.
pub fn eval_embeddings_from_proto(
    req: pb::EvalEmbeddingsRequest,
) -> Result<EvalEmbeddingsArgs, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    if req.golden_source.is_empty() {
        return Err(Status::invalid_argument("golden_source is required"));
    }
    Ok(EvalEmbeddingsArgs {
        source_id: req.source_id,
        embedding_table: optional(req.embedding_table),
        golden_source: req.golden_source,
        k: req.k as usize,
        cohorts: cohorts_from_proto(req.cohorts),
    })
}

// ─── EvalPerQuery ────────────────────────────────────────────────────────────

/// Decode a serialized [`pb::EvalPerQueryRequest`] body into the `eval_run_id`
/// the engine method (`Session::eval_per_query`) reads back. The embedded binding
/// builds the request with the same pure-Python assembly the remote client uses,
/// serializes it, and hands the bytes here — so the in-process and remote
/// per-query readback paths decode through one shared seam
/// ([`eval_per_query_from_proto`]). A body that is not a valid
/// `EvalPerQueryRequest` is a client error (`InvalidArgument`).
pub fn eval_per_query_from_bytes(body: &[u8]) -> Result<String, Status> {
    let req = pb::EvalPerQueryRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed EvalPerQuery request: {e}")))?;
    eval_per_query_from_proto(req)
}

/// Decode a [`pb::EvalPerQueryRequest`] into the `eval_run_id` to read back. The
/// required `eval_run_id` is validated at decode — matching the gRPC handler's
/// edge check. The wire `tenant_id` override is not read here: tenant scope is
/// the session's bound tenant, applied by the caller.
pub fn eval_per_query_from_proto(req: pb::EvalPerQueryRequest) -> Result<String, Status> {
    if req.eval_run_id.is_empty() {
        return Err(Status::invalid_argument("eval_run_id is required"));
    }
    Ok(req.eval_run_id)
}

// ─── EvalInference ───────────────────────────────────────────────────────────

/// The decoded inference-eval target a `EvalInference` request carries. The
/// engine method (`Session::eval_inference`) takes the model, the source, the
/// content columns, the decoded [`EvalTask`], the golden source, and the label
/// column separately, so the decode returns them as a struct the binding
/// destructures.
pub struct EvalInferenceArgs {
    pub model_id: String,
    pub source_id: String,
    pub columns: Vec<String>,
    pub task: EvalTask,
    pub golden_source: String,
    pub label_column: String,
}

/// Decode a serialized [`pb::EvalInferenceRequest`] body into the engine
/// [`EvalInferenceArgs`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote eval-inference paths decode through
/// one shared seam ([`eval_inference_from_proto`]). A body that is not a valid
/// `EvalInferenceRequest` is a client error (`InvalidArgument`).
pub fn eval_inference_from_bytes(body: &[u8]) -> Result<EvalInferenceArgs, Status> {
    let req = pb::EvalInferenceRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed EvalInference request: {e}")))?;
    eval_inference_from_proto(req)
}

/// Decode a [`pb::EvalInferenceRequest`] into the engine [`EvalInferenceArgs`].
/// The required identity fields (`model_id` / `source_id` / `golden_source` /
/// `label_column`) and a non-empty `columns` list are validated at decode, and
/// the task resolves through the shared [`EvalTaskFromWire`] conversion (an
/// unspecified or unknown task is rejected) — matching the gRPC handler's edge
/// checks.
pub fn eval_inference_from_proto(
    req: pb::EvalInferenceRequest,
) -> Result<EvalInferenceArgs, Status> {
    if req.model_id.is_empty() {
        return Err(Status::invalid_argument("model_id is required"));
    }
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    if req.golden_source.is_empty() {
        return Err(Status::invalid_argument("golden_source is required"));
    }
    if req.label_column.is_empty() {
        return Err(Status::invalid_argument("label_column is required"));
    }
    if req.columns.is_empty() {
        return Err(Status::invalid_argument("columns is required"));
    }
    Ok(EvalInferenceArgs {
        model_id: req.model_id,
        source_id: req.source_id,
        columns: req.columns,
        task: EvalTaskFromWire::try_from(req.task)?.0,
        golden_source: req.golden_source,
        label_column: req.label_column,
    })
}

// ─── EvalCompare ─────────────────────────────────────────────────────────────

/// The decoded compare-eval target a `EvalCompare` request carries. The engine
/// method (`Session::eval_compare`) takes the embedding tables, the source, the
/// golden source, and the `k` cap separately, so the decode returns them as a
/// struct the binding destructures.
pub struct EvalCompareArgs {
    pub embedding_tables: Vec<String>,
    pub source_id: String,
    pub golden_source: String,
    pub k: usize,
}

/// Decode a serialized [`pb::EvalCompareRequest`] body into the engine
/// [`EvalCompareArgs`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote eval-compare paths decode through
/// one shared seam ([`eval_compare_from_proto`]). A body that is not a valid
/// `EvalCompareRequest` is a client error (`InvalidArgument`).
pub fn eval_compare_from_bytes(body: &[u8]) -> Result<EvalCompareArgs, Status> {
    let req = pb::EvalCompareRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed EvalCompare request: {e}")))?;
    eval_compare_from_proto(req)
}

/// Decode a [`pb::EvalCompareRequest`] into the engine [`EvalCompareArgs`]. The
/// required `source_id` / `golden_source` are validated at decode, and
/// `embedding_tables` must name at least two tables (one baseline and one to
/// compare against it) — matching the gRPC handler's edge checks. The `k` cap
/// widens to the engine's `usize`.
pub fn eval_compare_from_proto(req: pb::EvalCompareRequest) -> Result<EvalCompareArgs, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    if req.golden_source.is_empty() {
        return Err(Status::invalid_argument("golden_source is required"));
    }
    if req.embedding_tables.len() < 2 {
        return Err(Status::invalid_argument(
            "embedding_tables requires at least two tables",
        ));
    }
    Ok(EvalCompareArgs {
        embedding_tables: req.embedding_tables,
        source_id: req.source_id,
        golden_source: req.golden_source,
        k: req.k as usize,
    })
}

// ─── EvalCalibration ─────────────────────────────────────────────────────────

/// The decoded calibration-eval target a `EvalCalibration` request carries. The
/// engine method (`InferenceSession::eval_calibration`) takes the source, the
/// golden source, the decoded [`EvalCalibrationShape`], and the cohort map
/// separately, so the decode returns them as a struct the binding destructures.
pub struct EvalCalibrationArgs {
    pub source_id: String,
    pub golden_source: String,
    pub shape: EvalCalibrationShape,
    pub cohorts: Cohorts,
}

/// Decode a serialized [`pb::EvalCalibrationRequest`] body into the engine
/// [`EvalCalibrationArgs`]. The embedded binding builds the request with the same
/// pure-Python assembly the remote client uses, serializes it, and hands the
/// bytes here — so the in-process and remote eval-calibration paths decode
/// through one shared seam ([`eval_calibration_from_proto`]). A body that is not
/// a valid `EvalCalibrationRequest` is a client error (`InvalidArgument`).
pub fn eval_calibration_from_bytes(body: &[u8]) -> Result<EvalCalibrationArgs, Status> {
    let req = pb::EvalCalibrationRequest::decode(body)
        .map_err(|e| Status::invalid_argument(format!("malformed EvalCalibration request: {e}")))?;
    eval_calibration_from_proto(req)
}

/// Decode a [`pb::EvalCalibrationRequest`] into the engine
/// [`EvalCalibrationArgs`]. The required `source_id` / `golden_source` are
/// validated at decode; the predictive shape resolves through the shared
/// [`calibration_shape_from_proto`] (an unspecified shape is rejected — it cannot
/// select the columns or scoring family); and the cohort tags rebuild through the
/// shared [`cohorts_from_proto`] — matching the gRPC handler's edge checks.
pub fn eval_calibration_from_proto(
    req: pb::EvalCalibrationRequest,
) -> Result<EvalCalibrationArgs, Status> {
    if req.source_id.is_empty() {
        return Err(Status::invalid_argument("source_id is required"));
    }
    if req.golden_source.is_empty() {
        return Err(Status::invalid_argument("golden_source is required"));
    }
    Ok(EvalCalibrationArgs {
        source_id: req.source_id,
        golden_source: req.golden_source,
        shape: calibration_shape_from_proto(req.shape)?,
        cohorts: cohorts_from_proto(req.cohorts),
    })
}

/// `""` → `None`, a non-empty string → `Some(String)`. Mirrors the engine's
/// "use the most recent table" sentinel for `embedding_table`.
fn optional(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}
