//! The remote [`Session`] transport: a gRPC client that drives an engine
//! hosted behind the `jammi.v1` wire surface.
//!
//! [`RemoteSession`] is the network peer of [`crate::LocalSession`]: it
//! implements the same verbs over the same request/result shapes, so the
//! [`Session`] enum dispatches to either arm and a consumer cannot tell which
//! transport backs it. Three properties make that interchangeability real:
//!
//! * **Faithful errors.** Every failure decodes the structured
//!   [`crate::wire::error_from_status`] detail the server attaches, so a remote
//!   call returns the *exact* [`JammiError`](jammi_db::error::JammiError)
//!   variant the in-process path would — never a lossy gRPC-code-category guess.
//! * **Tenant over the wire.** The tenant trio maps to
//!   `SessionService.SetTenant` / `GetTenant` / `ClearTenant`; the binding is
//!   keyed by an opaque per-session id carried in the
//!   [`SESSION_HEADER`](crate::wire::SESSION_HEADER), never in a request body.
//! * **Shared conversions.** Request encode / response decode reuse the
//!   [`crate::wire`] conversions the server's receive side uses — neither side
//!   reimplements a mapping.
//!
//! This stage wires the embeddings / encode-query / search / remove-source verbs
//! plus the tenant trio; the remaining verbs follow in a later stage.

use std::sync::Arc;

use arrow::array::{ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use tonic::codegen::InterceptedService;
use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Status};

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::TenantId;

use crate::wire::error_from_status;
use crate::wire::proto::embedding::embedding_service_client::EmbeddingServiceClient;
use crate::wire::proto::embedding::{
    encode_query_request::Input as ProtoEncodeInput, search_request::Query as ProtoSearchQuery,
    EncodeQueryRequest, GenerateEmbeddingsRequest, QueryVector, RemoveSourceRequest, SearchRequest,
    SearchResponse,
};
use crate::wire::proto::session::session_service_client::SessionServiceClient;
use crate::wire::proto::session::{SetTenantRequest, Tenant};
use crate::wire::{result_table_from_proto, SESSION_HEADER};
use crate::{Modality, QueryInput, SearchQuery, SearchRequest as SessionSearch};

/// Injects the [`SESSION_HEADER`] carrying this client's opaque session id on
/// every outbound request, so the server's tenant interceptor resolves the same
/// binding the tenant trio set. One per [`RemoteSession`]; cheap to clone (a
/// pre-parsed metadata value).
#[derive(Clone)]
struct SessionHeader {
    id: MetadataValue<tonic::metadata::Ascii>,
}

impl Interceptor for SessionHeader {
    fn call(&mut self, mut request: Request<()>) -> std::result::Result<Request<()>, Status> {
        request
            .metadata_mut()
            .insert(SESSION_HEADER, self.id.clone());
        Ok(request)
    }
}

/// A [`Session`] backed by a remote engine over gRPC.
///
/// [`Session`]: crate::Session
pub struct RemoteSession {
    channel: Channel,
    header: SessionHeader,
    /// The opaque session id this client minted, kept so [`Self::session_id`]
    /// can report the key the server binds tenant state against; the
    /// interceptor carries the parsed copy.
    session_id: String,
}

impl RemoteSession {
    /// Connect to a `jammi.v1` gRPC endpoint and mint a fresh session id.
    ///
    /// `endpoint` is any value an [`Endpoint`] accepts (e.g. an
    /// `"http://host:port"` string). The transport is native tonic here; the
    /// session id is a v4 UUID so two clients against one server never collide
    /// on a tenant binding.
    pub async fn connect(endpoint: impl Into<Endpoint>) -> Result<Self> {
        let channel = endpoint
            .into()
            .connect()
            .await
            .map_err(|e| JammiError::Config(format!("connect to jammi endpoint: {e}")))?;
        let session_id = uuid::Uuid::new_v4().to_string();
        let id: MetadataValue<tonic::metadata::Ascii> = session_id
            .parse()
            .map_err(|e| JammiError::Config(format!("session id metadata: {e}")))?;
        Ok(Self {
            channel,
            header: SessionHeader { id },
            session_id,
        })
    }

    /// The opaque session id this client minted. The server keys the tenant
    /// binding by it; two `RemoteSession`s over the same channel are isolated
    /// because each mints its own.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    fn embedding_client(
        &self,
    ) -> EmbeddingServiceClient<InterceptedService<Channel, SessionHeader>> {
        EmbeddingServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    fn session_client(&self) -> SessionServiceClient<InterceptedService<Channel, SessionHeader>> {
        SessionServiceClient::with_interceptor(self.channel.clone(), self.header.clone())
    }

    // --- sources ---------------------------------------------------------

    pub(crate) async fn remove_source(&self, source_id: &str) -> Result<()> {
        self.embedding_client()
            .remove_source(RemoveSourceRequest {
                source_id: source_id.to_string(),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    // --- embeddings ------------------------------------------------------

    pub(crate) async fn generate_embeddings(
        &self,
        source_id: &str,
        model_id: &str,
        columns: &[String],
        key_column: &str,
        modality: Modality,
    ) -> Result<ResultTableRecord> {
        let table = self
            .embedding_client()
            .generate_embeddings(GenerateEmbeddingsRequest {
                source_id: source_id.to_string(),
                model_id: model_id.to_string(),
                columns: columns.to_vec(),
                key_column: key_column.to_string(),
                modality: proto_modality(modality) as i32,
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(result_table_from_proto(table, modality))
    }

    pub(crate) async fn encode_query(
        &self,
        model_id: &str,
        input: QueryInput,
        modality: Modality,
    ) -> Result<Vec<f32>> {
        let input = match input {
            QueryInput::Text(text) => ProtoEncodeInput::Text(text),
            QueryInput::Bytes(bytes) => ProtoEncodeInput::Data(bytes),
        };
        let resp = self
            .embedding_client()
            .encode_query(EncodeQueryRequest {
                model_id: model_id.to_string(),
                modality: proto_modality(modality) as i32,
                input: Some(input),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        Ok(resp.embedding)
    }

    // --- search ----------------------------------------------------------

    pub(crate) async fn search(&self, request: SessionSearch) -> Result<Vec<RecordBatch>> {
        let SessionSearch {
            source_id,
            query,
            k,
            filter,
            select,
        } = request;
        let query = match query {
            SearchQuery::Vector(values) => ProtoSearchQuery::QueryVector(QueryVector { values }),
            SearchQuery::RowKey(key) => ProtoSearchQuery::RowKey(key),
        };
        let resp = self
            .embedding_client()
            .search(SearchRequest {
                source_id,
                query: Some(query),
                k: k as u32,
                filter,
                select: select.clone(),
            })
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        hits_to_batch(resp, &select)
    }

    // --- tenant ----------------------------------------------------------

    pub(crate) async fn bind_tenant(&self, t: TenantId) -> Result<()> {
        self.session_client()
            .set_tenant(SetTenantRequest {
                tenant: Some(Tenant { id: t.to_string() }),
            })
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn unbind_tenant(&self) -> Result<()> {
        self.session_client()
            .clear_tenant(())
            .await
            .map_err(|s| error_from_status(&s))?;
        Ok(())
    }

    pub(crate) async fn tenant(&self) -> Result<Option<TenantId>> {
        let resp = self
            .session_client()
            .get_tenant(())
            .await
            .map_err(|s| error_from_status(&s))?
            .into_inner();
        let id = resp.tenant.map(|t| t.id).unwrap_or_default();
        if id.is_empty() {
            return Ok(None);
        }
        id.parse()
            .map(Some)
            .map_err(|e| JammiError::Tenant(format!("invalid tenant id from server: {e}")))
    }
}

/// Map the engine [`Modality`] onto the wire enum. Encode is total (the engine
/// never holds an unspecified modality), so this is a plain `From`-shaped match
/// rather than the fallible decode the server side runs.
fn proto_modality(modality: Modality) -> crate::wire::proto::embedding::Modality {
    use crate::wire::proto::embedding::Modality as Pb;
    match modality {
        Modality::Text => Pb::Text,
        Modality::Image => Pb::Image,
        Modality::Audio => Pb::Audio,
    }
}

/// Rebuild the terminal `Vec<RecordBatch>` shape [`crate::Session::search`]
/// returns from the wire `SearchResponse`.
///
/// The wire surface carries each hit as `key` + `score` + a `columns` map of
/// stringified projections (the search-consumption shape edge runtimes reach),
/// so the remote arm rehydrates one batch with the `_row_id` (key) and
/// `similarity` (score) columns the local arm's hydrated batch carries, plus a
/// `Utf8` column per requested `select` name. Both arms therefore yield a batch
/// keyed and scored identically; the remote arm's projected columns are the
/// wire's stringified form (the server stringified them on the way out), which
/// is the only form that crosses this verb's wire.
fn hits_to_batch(resp: SearchResponse, select: &[String]) -> Result<Vec<RecordBatch>> {
    if resp.hits.is_empty() {
        return Ok(Vec::new());
    }
    let keys: Vec<&str> = resp.hits.iter().map(|h| h.key.as_str()).collect();
    let scores: Vec<f32> = resp.hits.iter().map(|h| h.score).collect();

    let mut fields: Vec<Field> = vec![
        Field::new("_row_id", DataType::Utf8, false),
        Field::new("similarity", DataType::Float32, false),
    ];
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(keys)),
        Arc::new(Float32Array::from(scores)),
    ];
    for name in select {
        let values: Vec<String> = resp
            .hits
            .iter()
            .map(|h| h.columns.get(name).cloned().unwrap_or_default())
            .collect();
        fields.push(Field::new(name, DataType::Utf8, false));
        arrays.push(Arc::new(StringArray::from(values)));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, arrays)
        .map_err(|e| JammiError::Other(format!("rebuild search batch: {e}")))?;
    Ok(vec![batch])
}
