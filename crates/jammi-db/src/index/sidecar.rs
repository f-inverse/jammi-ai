use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use crate::config::AnnIndexConfig;
use crate::error::{JammiError, Result};
use crate::index::VectorIndex;

/// Current rowmap format version.
const ROWMAP_VERSION: u32 = 1;

/// Build the USearch index options for a sidecar of the given dimension.
///
/// This is the sole place USearch's field names appear: the public engine
/// surface speaks the HNSW primitive ([`AnnIndexConfig`]), and this function is
/// the one boundary that maps it onto `usearch::IndexOptions`. A `0` knob is
/// carried through verbatim — USearch treats `0` as "use the built-in default",
/// so an [`AnnIndexConfig::default`] reproduces the backend defaults exactly.
/// `..Default::default()` preserves the remaining options (notably `multi`).
fn index_options(dimensions: usize, ann: &AnnIndexConfig) -> usearch::IndexOptions {
    usearch::IndexOptions {
        dimensions,
        metric: usearch::MetricKind::Cos,
        quantization: usearch::ScalarKind::F32,
        connectivity: ann.connectivity,
        expansion_add: ann.build_expansion,
        expansion_search: ann.search_expansion,
        ..Default::default()
    }
}

/// Sidecar ANN index backed by USearch, with a Jammi-owned `_row_id` mapping
/// and a JSON manifest.
///
/// Files produced per embedding table:
/// - `.usearch` — USearch serialized graph
/// - `.rowmap` — row_id mapping (internal_id → _row_id string)
/// - `.manifest.json` — metadata (version, dimensions, count, backend, created_at)
pub struct SidecarIndex {
    dimensions: usize,
    index: usearch::Index,
    row_map: Vec<String>,
    /// Reverse of `row_map`: `_row_id` → internal USearch key, so a stored
    /// vector can be fetched back by id via [`SidecarIndex::get`] without the
    /// caller keeping a second copy of the vectors. Holds only the ids (the same
    /// strings already in `row_map`), never the embeddings.
    row_index: HashMap<String, u64>,
    built: bool,
}

impl SidecarIndex {
    /// Create a new empty sidecar index for vectors of the given dimension,
    /// tuned by the HNSW knobs in `ann`. The build-time knobs (`connectivity`,
    /// `build_expansion`) take effect as vectors are added; `search_expansion`
    /// governs queries against the resulting graph.
    pub fn new(dimensions: usize, ann: &AnnIndexConfig) -> Result<Self> {
        let index = usearch::Index::new(&index_options(dimensions, ann))
            .map_err(|e| JammiError::Other(format!("USearch index creation: {e}")))?;

        Ok(Self {
            dimensions,
            index,
            row_map: Vec::new(),
            row_index: HashMap::new(),
            built: false,
        })
    }

    /// Fetch the stored vector for `row_id`, or `None` if the id is not indexed.
    ///
    /// Reads the vector USearch already holds rather than asking the caller to
    /// keep its own id→vector map — the index is the single owner of the
    /// embeddings it was built over.
    pub fn get(&self, row_id: &str) -> Result<Option<Vec<f32>>> {
        let Some(&key) = self.row_index.get(row_id) else {
            return Ok(None);
        };
        let mut out = Vec::new();
        let found = self
            .index
            .export(key, &mut out)
            .map_err(|e| JammiError::Other(format!("USearch get: {e}")))?;
        if found == 0 {
            return Ok(None);
        }
        out.truncate(self.dimensions);
        Ok(Some(out))
    }

    /// Save the sidecar bundle (`.usearch` + `.rowmap` + `.manifest.json`).
    pub fn save(&self, base_path: &Path) -> Result<()> {
        // Save USearch index
        let usearch_path = base_path.with_extension("usearch");
        self.index
            .save(usearch_path.to_str().unwrap_or_default())
            .map_err(|e| JammiError::Other(format!("USearch save: {e}")))?;

        // Save rowmap: version (u32 LE) + entries (len_u32 LE + UTF-8 bytes)
        let rowmap_path = base_path.with_extension("rowmap");
        let mut file = std::fs::File::create(&rowmap_path)?;
        file.write_all(&ROWMAP_VERSION.to_le_bytes())?;
        for id in &self.row_map {
            let bytes = id.as_bytes();
            file.write_all(&(bytes.len() as u32).to_le_bytes())?;
            file.write_all(bytes)?;
        }

        // Save manifest
        let manifest_path = base_path.with_extension("manifest.json");
        let manifest = serde_json::json!({
            "version": 1,
            "dimensions": self.dimensions,
            "count": self.row_map.len(),
            "metric": "cosine",
            "backend": "usearch",
            "backend_version": crate::index::backend_version(),
            "files": {
                "index": usearch_path.file_name().and_then(|n| n.to_str()),
                "rowmap": rowmap_path.file_name().and_then(|n| n.to_str()),
            },
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        Ok(())
    }

    /// Load a sidecar bundle from disk, applying the query-time HNSW knob from
    /// `ann`.
    ///
    /// A loaded graph is frozen: `connectivity` was baked in at build time (and
    /// is repopulated from the serialized header), and `build_expansion` has no
    /// effect on an existing graph — so neither build knob is consequential
    /// here. `search_expansion` is the exception: USearch does not persist it in
    /// the serialized header, so the loaded handle carries the backend default
    /// until it is set explicitly. We re-apply it from `ann` when non-zero; a
    /// `0` leaves the backend default in place (today's behaviour).
    pub fn load(base_path: &Path, ann: &AnnIndexConfig) -> Result<Self> {
        // Load manifest to get dimensions
        let manifest_path = base_path.with_extension("manifest.json");
        let manifest_str = std::fs::read_to_string(&manifest_path)?;
        let manifest: serde_json::Value = serde_json::from_str(&manifest_str)?;
        let dimensions = manifest["dimensions"]
            .as_u64()
            .ok_or_else(|| JammiError::Other("Missing dimensions in manifest".into()))?
            as usize;

        // Load rowmap
        let rowmap_path = base_path.with_extension("rowmap");
        let mut file = std::fs::File::open(&rowmap_path)?;
        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != ROWMAP_VERSION {
            return Err(JammiError::Other(format!(
                "Unknown rowmap version {version}, expected {ROWMAP_VERSION}"
            )));
        }

        let mut row_map = Vec::new();
        loop {
            let mut len_bytes = [0u8; 4];
            match file.read_exact(&mut len_bytes) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            let mut buf = vec![0u8; len];
            file.read_exact(&mut buf)?;
            row_map.push(
                String::from_utf8(buf)
                    .map_err(|e| JammiError::Other(format!("Invalid UTF-8 in rowmap: {e}")))?,
            );
        }

        // Load USearch index. Build knobs are inert for a frozen graph, so the
        // handle is created with backend defaults; only `search_expansion` is
        // re-applied below.
        let index = usearch::Index::new(&index_options(dimensions, &AnnIndexConfig::default()))
            .map_err(|e| JammiError::Other(format!("USearch index creation for load: {e}")))?;

        let usearch_path = base_path.with_extension("usearch");
        index
            .load(usearch_path.to_str().unwrap_or_default())
            .map_err(|e| JammiError::Other(format!("USearch load: {e}")))?;

        if ann.search_expansion != 0 {
            index.change_expansion_search(ann.search_expansion);
        }

        let row_index = row_map
            .iter()
            .enumerate()
            .map(|(key, id)| (id.clone(), key as u64))
            .collect();

        Ok(Self {
            dimensions,
            index,
            row_map,
            row_index,
            built: true,
        })
    }
}

impl VectorIndex for SidecarIndex {
    fn add(&mut self, row_id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(JammiError::Other(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }
        let key = self.row_map.len() as u64;
        // Reserve space if needed
        if self.index.capacity() <= self.index.size() {
            let new_cap = (self.index.capacity() + 1).max(64);
            self.index
                .reserve(new_cap)
                .map_err(|e| JammiError::Other(format!("USearch reserve: {e}")))?;
        }
        self.index
            .add(key, vector)
            .map_err(|e| JammiError::Other(format!("USearch add: {e}")))?;
        self.row_map.push(row_id.to_string());
        self.row_index.insert(row_id.to_string(), key);
        Ok(())
    }

    fn build(&mut self) -> Result<()> {
        // USearch builds incrementally during add(), so build is a no-op.
        // We just mark it as built for correctness tracking.
        self.built = true;
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        if self.row_map.is_empty() {
            return Ok(Vec::new());
        }
        let actual_k = k.min(self.row_map.len());
        let matches = self
            .index
            .search(query, actual_k)
            .map_err(|e| JammiError::Other(format!("USearch search: {e}")))?;

        let results: Vec<(String, f32)> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&key, &dist)| {
                let idx = key as usize;
                self.row_map.get(idx).map(|id| (id.clone(), dist))
            })
            .collect();
        Ok(results)
    }

    fn save(&self, path: &Path) -> Result<()> {
        SidecarIndex::save(self, path)
    }

    fn len(&self) -> usize {
        self.row_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    // USearch's built-in HNSW defaults — what a `0` knob resolves to. Pinned
    // here deliberately: a backend bump that shifts a default trips these
    // assertions rather than silently changing every index's recall/cost.
    const USEARCH_DEFAULT_CONNECTIVITY: usize = 16;
    const USEARCH_DEFAULT_EXPANSION_ADD: usize = 128;
    const USEARCH_DEFAULT_EXPANSION_SEARCH: usize = 64;

    #[test]
    fn knobs_map_onto_a_freshly_built_graph() {
        // Non-default build/query knobs flow through `index_options` into the
        // backing graph at construction time.
        let ann = AnnIndexConfig {
            connectivity: 32,
            build_expansion: 200,
            search_expansion: 100,
        };
        let idx = SidecarIndex::new(8, &ann).unwrap();
        assert_eq!(idx.index.connectivity(), 32);
        assert_eq!(idx.index.expansion_add(), 200);
        assert_eq!(idx.index.expansion_search(), 100);
    }

    #[test]
    fn default_config_reproduces_backend_defaults() {
        // A zeroed config is the documented no-op: every knob resolves to the
        // backend's built-in default, so an unset deployment is unchanged.
        let idx = SidecarIndex::new(8, &AnnIndexConfig::default()).unwrap();
        assert_eq!(idx.index.connectivity(), USEARCH_DEFAULT_CONNECTIVITY);
        assert_eq!(idx.index.expansion_add(), USEARCH_DEFAULT_EXPANSION_ADD);
        assert_eq!(
            idx.index.expansion_search(),
            USEARCH_DEFAULT_EXPANSION_SEARCH
        );
    }

    #[test]
    fn load_reapplies_search_expansion_only() {
        // `search_expansion` is a query-time dial USearch does not persist in
        // the serialized header, so a load must re-apply it. The build knobs are
        // frozen into the graph and are NOT recovered from the config on load
        // (`connectivity` is read back from the header; `expansion_add` resets
        // to the backend default) — load honours `search_expansion` alone.
        let dir = tempdir().unwrap();
        let base = dir.path().join("knob_roundtrip");

        let build = AnnIndexConfig {
            connectivity: 32,
            build_expansion: 200,
            search_expansion: 0,
        };
        let mut idx = SidecarIndex::new(4, &build).unwrap();
        idx.add("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.build().unwrap();
        idx.save(&base).unwrap();

        // Load with a non-zero search_expansion → re-applied to the loaded graph.
        let loaded = SidecarIndex::load(
            &base,
            &AnnIndexConfig {
                search_expansion: 77,
                ..AnnIndexConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            loaded.index.expansion_search(),
            77,
            "search_expansion must be re-applied on load"
        );

        // Load with a default (0) search_expansion → the backend default, not 0.
        let loaded_default = SidecarIndex::load(&base, &AnnIndexConfig::default()).unwrap();
        assert_eq!(
            loaded_default.index.expansion_search(),
            USEARCH_DEFAULT_EXPANSION_SEARCH
        );
    }
}
