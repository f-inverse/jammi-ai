pub mod exact;
pub mod sidecar;

use crate::error::Result;

/// Trait for ANN vector indexes keyed by `_row_id`.
pub trait VectorIndex: Send + Sync {
    /// Add a vector with its row ID to the index.
    fn add(&mut self, row_id: &str, vector: &[f32]) -> Result<()>;

    /// Build the index graph. Must be called after all `add()` calls.
    fn build(&mut self) -> Result<()>;

    /// Search for the `k` nearest neighbors, returning `(row_id, cosine_distance)` sorted ascending.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>>;

    /// Persist the index to disk.
    fn save(&self, path: &std::path::Path) -> Result<()>;

    /// Load an index from disk.
    fn load(path: &std::path::Path) -> Result<Self>
    where
        Self: Sized;

    /// Number of vectors currently in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compute cosine distance between two vectors: 1.0 - cosine_similarity.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / denom)
}
