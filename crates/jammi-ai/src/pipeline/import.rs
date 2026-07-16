//! Promotes precomputed per-row vectors into a first-class embedding table
//! without re-running any encoder — the import counterpart of the
//! [`EmbeddingPipeline`](crate::pipeline::embedding::EmbeddingPipeline) generate
//! path.
//!
//! A producer that already ran the model elsewhere (a remote vector upsert, a
//! migration off another store, an offline recompute-avoidance batch) hands the
//! engine a Parquet of `(_row_id, vector)` rows; this reads them and lands the
//! result through [`ResultStore::materialize_computed_embedding_table`], which
//! upholds the embedding-table invariants (every vector L2-normalized so the
//! cosine ANN sidecar is valid) and folds a content digest of the normalized
//! rows into the descriptor's `params`, so the result is searchable and
//! joinable exactly like a `GenerateEmbeddings` output.
//!
//! The table is recompute-inert: the engine did not compute it, so its
//! [`jammi_db::store::manifest::ProducingDescriptor::External`] descriptor
//! returns the typed [`jammi_db::error::JammiError::NotRecomputable`] refusal
//! on replay rather than a re-run guessed from the columns. The descriptor's
//! `params` fold every output-affecting scalar determinant — the content
//! digest of the normalized vectors is auto-folded in by the materialization
//! verb — so two distinct imports never collide on one definition hash.

use std::collections::BTreeMap;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{InputAnchor, MaterializationEnv, ModelIdentity};
use jammi_db::store::{ComputedEmbeddingProvenance, EmbeddingTableSpec, ResultStore};

use crate::model::ModelSource;
use crate::session::InferenceSession;

/// The generic label recorded as the imported table's `External` producer id.
/// Names a mechanism — a precomputed-vector import — never a consumer, so the
/// engine's one-way naming rule holds.
const IMPORT_PRODUCER_ID: &str = "external_import";

/// The backend identity recorded for an imported table. No inference backend
/// ran (the vectors were produced outside the engine), so the model identity's
/// backend is the import mechanism itself — an honest, deterministic value that
/// folds into the definition hash alongside the canonical model id.
const IMPORT_BACKEND: &str = "external_import";

/// Orchestrates a precomputed-vector import: read `(_row_id, vector)` rows →
/// materialize them through [`ResultStore::materialize_computed_embedding_table`],
/// which lands a ready embedding table with a content-complete
/// [`jammi_db::store::manifest::ProducingDescriptor::External`].
pub struct ImportPipeline<'a> {
    session: &'a InferenceSession,
    result_store: &'a ResultStore,
}

impl<'a> ImportPipeline<'a> {
    pub fn new(session: &'a InferenceSession, result_store: &'a ResultStore) -> Self {
        Self {
            session,
            result_store,
        }
    }

    /// Materialize the precomputed vectors at `vectors_url` as a ready
    /// `(source_id, TextEmbedding, model_id)` embedding table.
    ///
    /// `key_column` and `text_columns` are recorded as catalog provenance (which
    /// source column the `_row_id` keys and which content columns the vectors
    /// were computed from); the physical key stays `_row_id`. `dimensions` is
    /// the width every incoming vector is validated to. `model_id` is parsed to
    /// its canonical form and recorded as derivation provenance — it is never
    /// loaded, so the import runs GPU-free.
    ///
    /// `key_column` is required and must be non-empty ([`JammiError::Schema`]).
    /// It is recorded verbatim as the source-side provenance a reader later
    /// scans; the engine does not check that the source has that column, because
    /// an import resolves nothing at write time beyond the storage object it is
    /// handed. A `key_column` the source lacks therefore writes a ready table
    /// whose vectors search normally, and whose readers that follow the
    /// provenance back to the source — the context split and value-column
    /// hydration — fail when they scan for that column.
    ///
    /// Reads the whole input object into memory (a thin promotion of
    /// [`ResultStore::materialize_computed_embedding_table`], which takes the
    /// rows eagerly); a streaming variant is future work.
    pub async fn run(
        &self,
        source_id: &str,
        model_id: &str,
        vectors_url: &StorageUrl,
        key_column: &str,
        text_columns: &[String],
        dimensions: usize,
    ) -> Result<ResultTableRecord> {
        // An import always attributes its keys to a source column — that name is
        // both a hash determinant and the catalog provenance a reader joins on,
        // so an empty one would record two disagreeing stories about the same
        // table. Rejected here so every entry to the pipeline, not just the wire
        // decode, upholds it.
        if key_column.is_empty() {
            return Err(JammiError::Schema {
                table: source_id.to_string(),
                column: "key_column".to_string(),
                expected: "a source key-column name to attribute the imported keys to".to_string(),
                actual: "an empty key_column".to_string(),
            });
        }

        // GPU-free model validation: parse to the canonical encoder reference
        // (never load it — real loadability surfaces at query time), exactly as
        // the generate path canonicalizes at `pipeline/embedding.rs`.
        let canonical_model_id = ModelSource::parse(model_id).to_string();

        // Read the precomputed rows; the verb below upholds the embedding-table
        // invariants (width == dimensions, L2-normalized, rejecting a zero or
        // non-finite norm) and auto-folds a content digest of the normalized
        // rows into `params`, so no import-side duplication is needed here.
        let rows = self.session.read_keyed_vectors(vectors_url).await?;

        // The scalar identity of this import — every output-affecting
        // determinant *except* the content digest, which the verb folds in
        // itself. An omitted determinant here is a silent false hash match, so
        // this set must stay complete.
        let joined_text = text_columns.join(",");
        let mut params = BTreeMap::new();
        params.insert("source_id".to_string(), source_id.to_string());
        params.insert("model_id".to_string(), canonical_model_id.clone());
        params.insert("dimensions".to_string(), dimensions.to_string());
        params.insert("key_column".to_string(), key_column.to_string());
        params.insert("text_columns".to_string(), joined_text.clone());

        let env = MaterializationEnv::new(
            self.session.compute_device(),
            vec![ModelIdentity {
                model_id: canonical_model_id.clone(),
                backend: IMPORT_BACKEND.to_string(),
                // No inference ran (the vectors were produced outside the
                // engine), so there is no resolved compute precision to
                // report; the descriptor is `External` and therefore
                // recompute-inert (`NotRecomputable`) regardless, and the
                // content digest the verb auto-folds into `params` is what
                // actually distinguishes two imports, not this field. The
                // default is recorded rather than a fabricated non-default
                // value, matching `IMPORT_BACKEND`'s "honest mechanism
                // placeholder" stance.
                compute_precision: jammi_numerics::ComputePrecision::default(),
            }],
        );
        // The sole input is the external vector object, which exposes no version
        // surface — so it anchors `UnpinnedAtInstant`. The table is
        // recompute-inert regardless (`External` refuses replay), so this anchor
        // is provenance, not a reproducibility pin.
        let inputs = vec![InputAnchor::unpinned_at_instant(
            vectors_url.to_string(),
            chrono::Utc::now().to_rfc3339(),
        )];

        let text_columns = (!joined_text.is_empty()).then_some(joined_text.as_str());
        self.result_store
            .materialize_computed_embedding_table(
                self.session.context(),
                EmbeddingTableSpec {
                    source_id,
                    model_id: &canonical_model_id,
                    derived_from: None,
                    dimensions,
                    key_column: Some(key_column),
                    text_columns,
                },
                &rows,
                ComputedEmbeddingProvenance {
                    producer_id: IMPORT_PRODUCER_ID.to_string(),
                    params,
                    env,
                    inputs,
                },
            )
            .await
    }
}
