use jammi_engine::index::sidecar::SidecarIndex;
use jammi_engine::index::VectorIndex;
use tempfile::tempdir;

// ─── SidecarIndex: add, search, edge cases ───────────────────────────────────

#[test]
fn sidecar_add_search_and_edge_cases() {
    // Core: add vectors, search returns correct nearest neighbor
    let mut index = SidecarIndex::new(3).unwrap();
    index.add("row_a", &[1.0, 0.0, 0.0]).unwrap();
    index.add("row_b", &[0.0, 1.0, 0.0]).unwrap();
    index.add("row_c", &[0.9, 0.1, 0.0]).unwrap();
    index.build().unwrap();

    assert_eq!(index.len(), 3);

    let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "row_a", "Nearest should be row_a");
    assert!(
        results[0].1 < results[1].1,
        "Results sorted by distance ascending"
    );

    // Edge: k > count returns all
    let results = index.search(&[1.0, 0.0, 0.0], 100).unwrap();
    assert_eq!(results.len(), 3);

    // Edge: empty index
    let empty = SidecarIndex::new(3).unwrap();
    assert!(empty.search(&[1.0, 0.0, 0.0], 5).unwrap().is_empty());
    assert!(empty.is_empty());
}

// ─── Save/load roundtrip with manifest verification ──────────────────────────

#[test]
fn sidecar_save_load_roundtrip() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("test_index");

    let mut index = SidecarIndex::new(3).unwrap();
    index.add("id_1", &[1.0, 0.0, 0.0]).unwrap();
    index.add("id_2", &[0.0, 1.0, 0.0]).unwrap();
    index.add("id_3", &[0.0, 0.0, 1.0]).unwrap();
    index.build().unwrap();
    index.save(&base_path).unwrap();

    // Sidecar bundle produced
    assert!(base_path.with_extension("usearch").exists());
    assert!(base_path.with_extension("rowmap").exists());
    assert!(base_path.with_extension("manifest.json").exists());

    // Manifest has required fields
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(base_path.with_extension("manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["dimensions"], 3);
    assert_eq!(manifest["count"], 3);
    assert_eq!(manifest["metric"], "cosine");
    assert_eq!(manifest["backend"], "usearch");

    // Load and verify search still works (row_id mapping survives)
    let loaded = SidecarIndex::load(&base_path).unwrap();
    assert_eq!(loaded.len(), 3);
    let results = loaded.search(&[1.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].0, "id_1");
}

// ─── Corruption detection ────────────────────────────────────────────────────

#[test]
fn sidecar_load_rejects_corrupted_rowmap() {
    let dir = tempdir().unwrap();
    let base_path = dir.path().join("bad_version");

    let mut index = SidecarIndex::new(2).unwrap();
    index.add("r1", &[1.0, 0.0]).unwrap();
    index.build().unwrap();
    index.save(&base_path).unwrap();

    // Corrupt the rowmap version byte
    let map_path = base_path.with_extension("rowmap");
    let mut data = std::fs::read(&map_path).unwrap();
    data[0..4].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&map_path, data).unwrap();

    assert!(
        SidecarIndex::load(&base_path).is_err(),
        "Should reject unknown rowmap version"
    );
}
