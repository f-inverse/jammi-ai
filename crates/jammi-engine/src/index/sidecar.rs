use std::io::{Read, Write};
use std::path::Path;

use crate::error::{JammiError, Result};
use crate::index::VectorIndex;

/// Current rowmap format version.
const ROWMAP_VERSION: u32 = 1;

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
    built: bool,
}

impl SidecarIndex {
    /// Create a new empty sidecar index for vectors of the given dimension.
    pub fn new(dimensions: usize) -> Result<Self> {
        let index = usearch::Index::new(&usearch::IndexOptions {
            dimensions,
            metric: usearch::MetricKind::Cos,
            quantization: usearch::ScalarKind::F32,
            ..Default::default()
        })
        .map_err(|e| JammiError::Other(format!("USearch index creation: {e}")))?;

        Ok(Self {
            dimensions,
            index,
            row_map: Vec::new(),
            built: false,
        })
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
            "files": {
                "index": usearch_path.file_name().and_then(|n| n.to_str()),
                "rowmap": rowmap_path.file_name().and_then(|n| n.to_str()),
            },
            "created_at": chrono::Utc::now().to_rfc3339(),
        });
        std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest)?)?;

        Ok(())
    }

    /// Load a sidecar bundle from disk.
    pub fn load(base_path: &Path) -> Result<Self> {
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

        // Load USearch index
        let index = usearch::Index::new(&usearch::IndexOptions {
            dimensions,
            metric: usearch::MetricKind::Cos,
            quantization: usearch::ScalarKind::F32,
            ..Default::default()
        })
        .map_err(|e| JammiError::Other(format!("USearch index creation for load: {e}")))?;

        let usearch_path = base_path.with_extension("usearch");
        index
            .load(usearch_path.to_str().unwrap_or_default())
            .map_err(|e| JammiError::Other(format!("USearch load: {e}")))?;

        Ok(Self {
            dimensions,
            index,
            row_map,
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

    fn load(path: &Path) -> Result<Self> {
        SidecarIndex::load(path)
    }

    fn len(&self) -> usize {
        self.row_map.len()
    }
}
