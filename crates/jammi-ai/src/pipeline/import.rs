//! Promotes precomputed per-row vectors into a first-class embedding table
//! without re-running any encoder — the import counterpart of the
//! [`EmbeddingPipeline`](crate::pipeline::embedding::EmbeddingPipeline) generate
//! path.
//!
//! A producer that already ran the model elsewhere (a remote vector upsert, a
//! migration off another store, an offline recompute-avoidance batch) hands the
//! engine a Parquet of `(_row_id, vector)` rows; this reads them, upholds the
//! embedding-table invariants (every vector L2-normalized so the cosine ANN
//! sidecar is valid), and lands the result through the single
//! [`ResultStore::materialize_embedding_table`] funnel so it is searchable and
//! joinable exactly like a `GenerateEmbeddings` output.
//!
//! The table is recompute-inert: the engine did not compute it, so its
//! [`ProducingDescriptor::External`] descriptor returns the typed
//! [`JammiError::NotRecomputable`] refusal on replay rather than a re-run
//! guessed from the columns. The
//! descriptor's `params` fold every output-affecting determinant — including a
//! content digest of the normalized vectors — so two distinct imports never
//! collide on one definition hash.

use std::collections::BTreeMap;

use jammi_db::catalog::result_repo::ResultTableRecord;
use jammi_db::error::{JammiError, Result};
use jammi_db::storage::StorageUrl;
use jammi_db::store::manifest::{
    ArtifactDigest, InputAnchor, Materialization, MaterializationEnv, ModelIdentity,
    ProducingDescriptor,
};
use jammi_db::store::{EmbeddingTableSpec, ResultStore};

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
/// L2-normalize (rejecting zero-norm) → land a ready embedding table through the
/// materialization funnel with a content-complete [`ProducingDescriptor::External`].
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

    /// Register the precomputed vectors at `vectors_url` as a ready
    /// `(source_id, TextEmbedding, model_id)` embedding table.
    ///
    /// `key_column` and `text_columns` are recorded as catalog provenance (which
    /// source column the `_row_id` keys and which content columns the vectors
    /// were computed from); the physical key stays `_row_id`. `dimensions` is
    /// the width every incoming vector is validated to. `model_id` is parsed to
    /// its canonical form and recorded as derivation provenance — it is never
    /// loaded, so the import runs GPU-free.
    ///
    /// Reads the whole input object into memory (a thin promotion of
    /// [`ResultStore::materialize_embedding_table`], which takes the rows
    /// eagerly); a streaming variant is future work.
    pub async fn run(
        &self,
        source_id: &str,
        model_id: &str,
        vectors_url: &StorageUrl,
        key_column: &str,
        text_columns: &[String],
        dimensions: usize,
    ) -> Result<ResultTableRecord> {
        // GPU-free model validation: parse to the canonical encoder reference
        // (never load it — real loadability surfaces at query time), exactly as
        // the generate path canonicalizes at `pipeline/embedding.rs`.
        let canonical_model_id = ModelSource::parse(model_id).to_string();

        // Read the precomputed rows, then uphold the embedding-table invariants:
        // every vector is `dimensions` wide and L2-normalized (the cosine ANN
        // sidecar assumes unit vectors); a zero-norm vector cannot be
        // cosine-searched, so it is rejected.
        let mut rows = self.session.read_keyed_vectors(vectors_url).await?;
        for (key, vector) in rows.iter_mut() {
            if vector.len() != dimensions {
                return Err(JammiError::Schema {
                    table: source_id.to_string(),
                    column: "vector".to_string(),
                    expected: format!("FixedSizeList<Float32> width {dimensions}"),
                    actual: format!("row '{key}' has width {}", vector.len()),
                });
            }
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(JammiError::Schema {
                    table: source_id.to_string(),
                    column: "vector".to_string(),
                    expected: "a non-zero-norm, L2-normalizable vector".to_string(),
                    actual: format!("row '{key}' has norm {norm}"),
                });
            }
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }

        // A content-complete `External` descriptor: `params` folds every
        // output-affecting determinant — the scalar identity AND a digest of the
        // normalized vectors — so two distinct imports under the same scalars
        // never serialise identically and collide on one definition hash.
        let joined_text = text_columns.join(",");
        let content_digest = content_digest(&rows);
        let mut params = BTreeMap::new();
        params.insert("source_id".to_string(), source_id.to_string());
        params.insert("model_id".to_string(), canonical_model_id.clone());
        params.insert("dimensions".to_string(), dimensions.to_string());
        params.insert("key_column".to_string(), key_column.to_string());
        params.insert("text_columns".to_string(), joined_text.clone());
        params.insert("content_digest".to_string(), content_digest);
        let descriptor = ProducingDescriptor::External {
            producer_id: IMPORT_PRODUCER_ID.to_string(),
            params,
        };

        let env = MaterializationEnv::new(
            self.session.compute_device(),
            vec![ModelIdentity {
                model_id: canonical_model_id.clone(),
                backend: IMPORT_BACKEND.to_string(),
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
            .materialize_embedding_table(
                self.session.context(),
                EmbeddingTableSpec {
                    source_id,
                    model_id: &canonical_model_id,
                    derived_from: None,
                    dimensions,
                    key_column,
                    text_columns,
                },
                &rows,
                Materialization::new(&descriptor, &env, inputs),
            )
            .await
    }
}

/// A stable content digest over the normalized rows: the hex of a hash folding
/// each row's key bytes and vector bytes in file order. Distinguishes two
/// imports that share every scalar determinant but carry different vectors, so
/// they do not alias on one definition hash.
fn content_digest(rows: &[(String, Vec<f32>)]) -> String {
    let mut buf = Vec::new();
    for (key, vector) in rows {
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&(vector.len() as u64).to_le_bytes());
        for x in vector {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    ArtifactDigest::of_bytes(&buf).0
}
