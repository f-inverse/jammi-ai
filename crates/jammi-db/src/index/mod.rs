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
