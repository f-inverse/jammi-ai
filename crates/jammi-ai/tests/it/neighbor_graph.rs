//! Hermetic tests for similarity-graph materialization. Each builds an
//! embedding table over the shipped `patents` fixture, then materializes its
//! k-nearest-neighbour edge relation and asserts the contract: edge shape,
//! ranking, the cosine-similarity coupling, the post-filters
//! (`min_similarity` / `mutual`), direct-to-source endpoint joins, the
//! exact-driver determinism guarantee, tenancy isolation, and that traversal
//! is the caller's SQL — never the engine's.

use std::str::FromStr;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int32Array, StringArray};
use jammi_ai::pipeline::neighbor_graph::BuildNeighborGraph;
use jammi_ai::session::InferenceSession;
use jammi_db::catalog::result_repo::{ResultTableKind, ResultTableRecord};
use jammi_db::source::{FileFormat, SourceConnection, SourceType};
use jammi_db::TenantId;
use tempfile::TempDir;

use crate::common;

fn tiny_bert_model() -> String {
    "local:".to_string() + common::cookbook_fixture("tiny_bert").to_str().unwrap()
}

async fn session_with_embeddings() -> (Arc<InferenceSession>, TempDir) {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());
    session
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    (session, dir)
}

/// Collect an edge table into parallel `(src, dst, rank, similarity)` columns.
async fn collect_edges(
    session: &InferenceSession,
    edges: &ResultTableRecord,
) -> Vec<(String, String, i32, f32)> {
    let batches = session
        .sql(&format!(
            "SELECT src, dst, rank, similarity FROM \"jammi.{}\"",
            edges.table_name
        ))
        .await
        .unwrap();

    let mut out = Vec::new();
    for batch in &batches {
        let src = col_str(batch, "src");
        let dst = col_str(batch, "dst");
        let rank = batch
            .column_by_name("rank")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let sim = batch
            .column_by_name("similarity")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            out.push((
                src.value(i).to_string(),
                dst.value(i).to_string(),
                rank.value(i),
                sim.value(i),
            ));
        }
    }
    out
}

/// Read a string column into an owned `StringArray`. A registered Parquet
/// result table returns its string columns as `Utf8View` under DataFusion 52+,
/// so the column is cast to `Utf8` before the downcast rather than assuming one
/// Arrow string family — the same handling the production read path uses.
fn col_str(batch: &arrow::array::RecordBatch, name: &str) -> StringArray {
    let col = batch.column_by_name(name).unwrap();
    let utf8 = arrow::compute::cast(col.as_ref(), &arrow::datatypes::DataType::Utf8).unwrap();
    utf8.as_any().downcast_ref::<StringArray>().unwrap().clone()
}

// ─── Shape, ranking, and the cosine coupling ─────────────────────────────────

#[tokio::test]
async fn neighbor_graph_has_correct_shape_and_ranking() {
    let (session, _dir) = session_with_embeddings().await;

    let edges_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 5,
                exact: true, // deterministic so the assertions are stable
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    assert_eq!(edges_table.kind, ResultTableKind::NeighborGraph);
    assert!(
        edges_table.derived_from.is_some(),
        "an edge table records the embedding table it was derived from"
    );
    assert!(
        edges_table.index_path.is_none(),
        "an edge table carries no sidecar index"
    );

    let edges = collect_edges(&session, &edges_table).await;
    assert!(!edges.is_empty(), "the fixture yields edges");

    // Group edges by src to check per-node invariants.
    use std::collections::BTreeMap;
    let mut by_src: BTreeMap<String, Vec<(i32, f32, String)>> = BTreeMap::new();
    for (src, dst, rank, sim) in &edges {
        assert_ne!(src, dst, "no self-edges with self_exclude = true (default)");
        by_src
            .entry(src.clone())
            .or_default()
            .push((*rank, *sim, dst.clone()));
    }

    for (src, mut ranked) in by_src {
        assert!(
            ranked.len() <= 5,
            "node {src} has {} edges, exceeding k = 5",
            ranked.len()
        );
        ranked.sort_by_key(|(rank, _, _)| *rank);
        // rank is 1..=k contiguous and monotone; similarity descends with rank.
        for (i, (rank, sim, _)) in ranked.iter().enumerate() {
            assert_eq!(*rank, (i as i32) + 1, "ranks are 1..k contiguous for {src}");
            assert!(
                *sim >= -1.0 && *sim <= 1.0001,
                "similarity in cosine range for {src}, got {sim}"
            );
            if i > 0 {
                assert!(
                    ranked[i - 1].1 >= *sim,
                    "similarity descends with rank for {src}: {} < {sim}",
                    ranked[i - 1].1
                );
            }
        }
    }
}

// ─── Endpoints are source keys that join directly to source (§4.1) ───────────

#[tokio::test]
async fn neighbor_graph_endpoints_join_directly_to_source() {
    let (session, _dir) = session_with_embeddings().await;

    let edges_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 3,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    // `src` joins straight to the source's key column with no detour through
    // the embedding table — every edge resolves to a real patent title.
    let joined = session
        .sql(&format!(
            "SELECT e.src, e.dst, p.title \
             FROM \"jammi.{}\" e \
             JOIN patents.public.patents p ON CAST(p.id AS VARCHAR) = e.src",
            edges_table.table_name
        ))
        .await
        .unwrap();

    let total: usize = joined.iter().map(|b| b.num_rows()).sum();
    let edges = collect_edges(&session, &edges_table).await;
    assert_eq!(
        total,
        edges.len(),
        "every edge's src joins to exactly one source row"
    );
}

// ─── min_similarity floors weak edges ────────────────────────────────────────

#[tokio::test]
async fn neighbor_graph_min_similarity_floors_weak_edges() {
    let (session, _dir) = session_with_embeddings().await;

    let unfiltered = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 5,
                exact: true,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let all = collect_edges(&session, &unfiltered).await;

    // A floor strictly above the weakest surviving edge must drop at least it.
    let weakest = all
        .iter()
        .map(|(_, _, _, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let strongest = all
        .iter()
        .map(|(_, _, _, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        weakest < strongest,
        "fixture must have a spread of similarities for this test to be meaningful"
    );
    let floor = (weakest + strongest) / 2.0;

    let filtered_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 5,
                min_similarity: Some(floor),
                exact: true,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let filtered = collect_edges(&session, &filtered_table).await;

    assert!(
        filtered.len() < all.len(),
        "the floor must drop weak edges: {} unfiltered vs {} filtered",
        all.len(),
        filtered.len()
    );
    for (_, _, _, sim) in &filtered {
        assert!(*sim >= floor, "no edge below the floor survives, got {sim}");
    }
}

// ─── mutual = true yields a reciprocal subset ────────────────────────────────

#[tokio::test]
async fn neighbor_graph_mutual_is_a_reciprocal_subset() {
    let (session, _dir) = session_with_embeddings().await;

    let directed_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 3,
                exact: true,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let directed = collect_edges(&session, &directed_table).await;
    use std::collections::HashSet;
    let directed_pairs: HashSet<(String, String)> = directed
        .iter()
        .map(|(s, d, _, _)| (s.clone(), d.clone()))
        .collect();

    let mutual_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 3,
                mutual: true,
                exact: true,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let mutual = collect_edges(&session, &mutual_table).await;

    assert!(
        mutual.len() <= directed.len(),
        "mutual edges are a subset of all directed edges"
    );
    for (src, dst, _, _) in &mutual {
        assert!(
            directed_pairs.contains(&(src.clone(), dst.clone())),
            "every mutual edge is also a directed edge"
        );
        assert!(
            mutual.iter().any(|(s, d, _, _)| s == dst && d == src),
            "edge ({src} -> {dst}) survives mutual only if ({dst} -> {src}) is present"
        );
    }
}

// ─── Exact driver is deterministic across runs (§3.1) ────────────────────────

#[tokio::test]
async fn neighbor_graph_exact_is_deterministic_across_runs() {
    let (session, _dir) = session_with_embeddings().await;

    let build = || async {
        let table = session
            .build_neighbor_graph(
                "patents",
                None,
                &BuildNeighborGraph {
                    k: 5,
                    exact: true,
                    ..Default::default()
                },
                jammi_db::store::CachePolicy::Bypass,
            )
            .await
            .unwrap()
            .0;
        collect_edges(&session, &table).await
    };

    // The index-assisted driver is approximate and non-deterministic by
    // contract; only the exact driver guarantees byte-identical edge sets, so
    // determinism is asserted on `exact = true` alone.
    let run_one = build().await;
    let run_two = build().await;
    assert_eq!(
        run_one, run_two,
        "exact builds produce an identical edge set across runs"
    );
}

// ─── Index-assisted agrees with exact within a recall tolerance ──────────────

#[tokio::test]
async fn neighbor_graph_index_agrees_with_exact_within_tolerance() {
    let (session, _dir) = session_with_embeddings().await;

    let exact_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 5,
                exact: true,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let approx_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 5,
                exact: false,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    use std::collections::HashSet;
    let exact: HashSet<(String, String)> = collect_edges(&session, &exact_table)
        .await
        .into_iter()
        .map(|(s, d, _, _)| (s, d))
        .collect();
    let approx: HashSet<(String, String)> = collect_edges(&session, &approx_table)
        .await
        .into_iter()
        .map(|(s, d, _, _)| (s, d))
        .collect();

    let overlap = exact.intersection(&approx).count();
    let recall = overlap as f32 / exact.len() as f32;
    // On a tiny fixture HNSW recall is high; allow slack for the approximate,
    // non-deterministic contract rather than asserting equality.
    assert!(
        recall >= 0.6,
        "index-assisted recall vs exact should be reasonable, got {recall}"
    );
}

// ─── The similarity weight is a plain, non-null column (no evidence channel) ──

#[tokio::test]
async fn neighbor_graph_similarity_is_a_plain_non_null_column() {
    let (session, _dir) = session_with_embeddings().await;

    let edges_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 4,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    let batches = session
        .sql(&format!(
            "SELECT similarity FROM \"jammi.{}\"",
            edges_table.table_name
        ))
        .await
        .unwrap();
    for batch in &batches {
        let sim = batch
            .column_by_name("similarity")
            .unwrap()
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(sim.null_count(), 0, "similarity is non-null");
    }

    // The edge table carries exactly the four edge columns — no provenance
    // list columns (`retrieved_by` / `annotated_by`) that `search` results
    // would have.
    let one = &session
        .sql(&format!(
            "SELECT * FROM \"jammi.{}\" LIMIT 1",
            edges_table.table_name
        ))
        .await
        .unwrap()[0];
    let schema = one.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(names, vec!["src", "dst", "rank", "similarity"]);
}

// ─── Traversal is the caller's SQL, not the engine's (§7) ────────────────────

#[tokio::test]
async fn two_hop_expansion_is_plain_sql_over_the_edge_table() {
    let (session, _dir) = session_with_embeddings().await;

    let edges_table = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 3,
                exact: true,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;

    // A self-join expands one hop to two — entirely in the query, with no
    // traversal operator in the engine.
    let two_hop = session
        .sql(&format!(
            "SELECT a.src AS origin, b.dst AS two_hops \
             FROM \"jammi.{t}\" a JOIN \"jammi.{t}\" b ON b.src = a.dst",
            t = edges_table.table_name
        ))
        .await
        .unwrap();
    let total: usize = two_hop.iter().map(|b| b.num_rows()).sum();
    assert!(total > 0, "two-hop self-join yields paths");
}

// ─── Tenancy: a build is bound to its tenant's table (§6) ────────────────────

#[tokio::test]
async fn neighbor_graph_is_tenant_scoped() {
    let dir = TempDir::new().unwrap();
    let config = common::test_config(dir.path());
    let session = Arc::new(InferenceSession::new(config).await.unwrap());

    let alice = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e01").unwrap();
    let bob = TenantId::from_str("018f5a0e-c4c8-7e10-9c4f-3b6f7c5a8e02").unwrap();

    session
        .add_source(
            "patents",
            SourceType::File,
            SourceConnection {
                url: Some(common::fixture_url("patents.parquet")),
                format: Some(FileFormat::Parquet),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    // Alice embeds and builds the graph under her tenant.
    session.bind_tenant(alice);
    session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    let alice_edges = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 3,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap()
        .0;
    assert!(
        session
            .catalog()
            .get_result_table(&alice_edges.table_name)
            .await
            .unwrap()
            .is_some(),
        "Alice resolves her own edge table"
    );

    // Bob, same session/source, cannot build over Alice's embedding table — the
    // tenant-scoped resolution finds no embedding table he owns, so the build
    // errors rather than crossing into Alice's data.
    session.bind_tenant(bob);
    let bob_build = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 3,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await;
    assert!(
        bob_build.is_err(),
        "Bob cannot point the build at Alice's tenant-scoped embedding table"
    );
}

// ─── k = 0 is rejected ───────────────────────────────────────────────────────

#[tokio::test]
async fn neighbor_graph_rejects_zero_k() {
    let (session, _dir) = session_with_embeddings().await;
    let err = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 0,
                ..Default::default()
            },
            jammi_db::store::CachePolicy::Bypass,
        )
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("k >= 1"),
        "k = 0 is a typed error, got: {err}"
    );
}

// ─── Opt-in memoization (CachePolicy) ────────────────────────────────────────
//
// A neighbour-graph anchors on the immutable source-embedding-table digest
// (`ResultDigest`), so it is GENUINELY cacheable: the same build over the same
// parent yields the same edges. These prove the dial is opt-in, observable, and
// keyed on the COMPLETE descriptor.

use jammi_db::store::{CacheOutcome, CachePolicy};
use std::time::Instant;

#[tokio::test]
async fn cache_use_exact_hit_reuses_and_is_far_under_the_cold_build() {
    let (session, _dir) = session_with_embeddings().await;
    let params = BuildNeighborGraph {
        k: 5,
        exact: true,
        ..Default::default()
    };

    // Cold build (Use, but nothing cached yet) → Computed, and time it.
    let cold_start = Instant::now();
    let (first, first_outcome) = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Use)
        .await
        .unwrap();
    let cold = cold_start.elapsed();
    assert_eq!(
        first_outcome,
        CacheOutcome::Computed,
        "the first Use build has nothing to reuse — it computes"
    );

    // Second Use over the identical (definition, source digest) → Reused, and the
    // probe short-circuits before the expensive build, so it is far faster.
    let warm_start = Instant::now();
    let (second, second_outcome) = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Use)
        .await
        .unwrap();
    let warm = warm_start.elapsed();

    assert_eq!(
        second_outcome,
        CacheOutcome::Reused {
            table: first.table_name.clone()
        },
        "an exact (definition, source-digest) match reuses the prior table"
    );
    assert_eq!(
        second.table_name, first.table_name,
        "the reused record names the cached table, not a fresh one"
    );
    // The probe is a catalog lookup + an extant-bytes check — orders under the
    // cold compute. A 2x margin is a conservative, non-flaky floor for the
    // skip-the-whole-build win (the real ratio is far larger).
    assert!(
        warm * 2 < cold,
        "a cache hit ({warm:?}) must be far under the cold build ({cold:?})"
    );
}

#[tokio::test]
async fn cache_bypass_always_recomputes_a_distinct_table() {
    let (session, _dir) = session_with_embeddings().await;
    let params = BuildNeighborGraph {
        k: 5,
        exact: true,
        ..Default::default()
    };

    let (a, a_outcome) = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Bypass)
        .await
        .unwrap();
    let (b, b_outcome) = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Bypass)
        .await
        .unwrap();

    assert_eq!(a_outcome, CacheOutcome::Computed);
    assert_eq!(
        b_outcome,
        CacheOutcome::Computed,
        "the default Bypass never reuses — it always recomputes"
    );
    assert_ne!(
        a.table_name, b.table_name,
        "two Bypass builds materialise two distinct tables"
    );
}

#[tokio::test]
async fn cache_one_bit_param_change_recomputes_not_reuses() {
    // The probe keys on the COMPLETE ProducingDescriptor (PR-A's completeness).
    // A one-bit change to ANY output-affecting build param — here `k` 5 -> 6 —
    // is a different definition hash, so the second Use cannot reuse the first.
    let (session, _dir) = session_with_embeddings().await;

    let (first, _) = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 5,
                exact: true,
                ..Default::default()
            },
            CachePolicy::Use,
        )
        .await
        .unwrap();

    let (second, outcome) = session
        .build_neighbor_graph(
            "patents",
            None,
            &BuildNeighborGraph {
                k: 6, // one knob changed → a different definition → no reuse
                exact: true,
                ..Default::default()
            },
            CachePolicy::Use,
        )
        .await
        .unwrap();

    assert_eq!(
        outcome,
        CacheOutcome::Computed,
        "a one-bit param change is a different cache key — it must recompute, \
         proving the probe keys on the full descriptor, not a lossy subset"
    );
    assert_ne!(first.table_name, second.table_name);
}

#[tokio::test]
async fn cache_use_does_not_reuse_across_a_recomputed_parent() {
    // The neighbour-graph anchors on the parent embedding table's digest. A
    // SECOND, distinct embedding table (a recomputed parent, new digest) gives a
    // different input anchor, so a Use build over it cannot reuse the first
    // graph — reuse is sound only over the identical parent state.
    let (session, _dir) = session_with_embeddings().await;
    let params = BuildNeighborGraph {
        k: 5,
        exact: true,
        ..Default::default()
    };

    // First graph over the first embedding table.
    let (g1, _) = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Use)
        .await
        .unwrap();

    // Recompute the parent: a fresh embedding table with a new digest.
    let (parent2, _) = session
        .generate_text_embeddings(
            "patents",
            &tiny_bert_model(),
            &["abstract".to_string()],
            "id",
            CachePolicy::Bypass,
        )
        .await
        .unwrap();

    let (g2, outcome) = session
        .build_neighbor_graph(
            "patents",
            Some(&parent2.table_name),
            &params,
            CachePolicy::Use,
        )
        .await
        .unwrap();

    assert_eq!(
        outcome,
        CacheOutcome::Computed,
        "a different parent digest is a different input anchor — no reuse"
    );
    assert_ne!(g1.table_name, g2.table_name);
}

#[tokio::test]
async fn cache_hit_leaves_no_building_orphan_to_reap() {
    // The probe runs at the TOP of the producer, BEFORE `create_table` / any
    // Parquet write — so a cache hit short-circuits with NO `building` row and NO
    // orphaned bytes. There is therefore no new crash window: `recover()` (the
    // reaper) finds nothing to reconcile, exactly as for a clean state. This is
    // the design that makes the secondary in-funnel re-probe unnecessary.
    use jammi_db::catalog::status::ResultTableStatus;
    use jammi_db::store::CachePolicy;

    let (session, _dir) = session_with_embeddings().await;
    let params = BuildNeighborGraph {
        k: 5,
        exact: true,
        ..Default::default()
    };

    // Prime the cache, then take a cache hit.
    let _ = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Use)
        .await
        .unwrap()
        .0;
    let (_reused, outcome) = session
        .build_neighbor_graph("patents", None, &params, CachePolicy::Use)
        .await
        .unwrap();
    assert!(
        matches!(outcome, jammi_db::store::CacheOutcome::Reused { .. }),
        "expected a cache hit to exercise the short-circuit"
    );

    // No `building` orphan was left by the short-circuit.
    let orphans = session
        .catalog()
        .list_result_tables_by_status(ResultTableStatus::Building)
        .await
        .unwrap();
    assert!(
        orphans.is_empty(),
        "a top-of-producer cache hit leaves no building orphan, got {:?}",
        orphans.iter().map(|r| &r.table_name).collect::<Vec<_>>()
    );
}
