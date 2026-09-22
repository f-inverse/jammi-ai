//! What an ANN sidecar costs to build and to search as its rows are divided
//! among segments, measured on the engine against itself and printed under
//! `--nocapture`; asserted only for consistency.
//!
//! For each `(rows, dimensions)`: the serial build of one index over every
//! row; the same rows divided into `S` contiguous segments, each built on its
//! own thread in its own row order; and, per layout, the mean query latency
//! and recall@10 of [`SegmentedIndex::search_final`] against the exact
//! answer. Row counts come from `JAMMI_INDEX_MEASURE_ROWS` (comma-separated),
//! `4096` when unset.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use jammi_db::config::{AnnIndexConfig, StoragePrecision};
use jammi_db::index::segment::{SegmentId, SegmentedIndex};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;
use jammi_numerics::distance::cosine_distance;
use jammi_test_utils::vq;

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

/// `n` vectors around 64 cluster centres, so neighbours are meaningful.
fn corpus(n: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut rng = Lcg(0x5eed ^ (n * dims) as u64);
    let centres: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..dims).map(|_| rng.unit()).collect())
        .collect();
    (0..n)
        .map(|i| {
            centres[i % 64]
                .iter()
                .map(|c| c + 0.35 * rng.unit())
                .collect()
        })
        .collect()
}

fn build(rows: &[(usize, &Vec<f32>)], dims: usize) -> SidecarIndex {
    let mut index =
        SidecarIndex::new(dims, &AnnIndexConfig::default(), StoragePrecision::F32).unwrap();
    for (i, v) in rows {
        index.add(&format!("{i:08}"), v).unwrap();
    }
    index.build().unwrap();
    index
}

/// `vectors` divided into `segments` contiguous runs, each built on its own
/// thread: the wall of the whole build and the assembled set.
fn build_segmented(vectors: &[Vec<f32>], dims: usize, segments: usize) -> (Duration, SegmentedIndex) {
    let rows: Vec<(usize, &Vec<f32>)> = vectors.iter().enumerate().collect();
    let per = rows.len().div_ceil(segments);
    let start = Instant::now();
    let built: Vec<SidecarIndex> = std::thread::scope(|scope| {
        let handles: Vec<_> = rows
            .chunks(per)
            .map(|run| scope.spawn(move || build(run, dims)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let wall = start.elapsed();
    let set = SegmentedIndex::new(
        built
            .into_iter()
            .enumerate()
            .map(|(i, index)| (SegmentId(i as i64), index))
            .collect(),
    )
    .unwrap();
    (wall, set)
}

fn exact_top(vectors: &[Vec<f32>], query: &[f32], k: usize) -> HashSet<String> {
    let query = vq(query);
    let mut scored: Vec<(f32, usize)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_distance(&query, v), i))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored
        .into_iter()
        .take(k)
        .map(|(_, i)| format!("{i:08}"))
        .collect()
}

#[test]
fn index_build_and_search_cost_by_segment_count() {
    let sizes: Vec<usize> = std::env::var("JAMMI_INDEX_MEASURE_ROWS")
        .ok()
        .map(|v| v.split(',').map(|n| n.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![4096]);
    let k = 10;
    println!(
        "{:>8} {:>5} {:>9} {:>10} {:>12} {:>12} {:>9}",
        "rows", "dims", "segments", "build_ms", "us_per_row", "query_us", "recall@10"
    );
    for &n in &sizes {
        for dims in [32usize, 384] {
            let vectors = corpus(n, dims);
            let queries: Vec<&Vec<f32>> = vectors.iter().step_by((n / 64).max(1)).collect();
            let truth: Vec<HashSet<String>> =
                queries.iter().map(|q| exact_top(&vectors, q, k)).collect();
            for segments in [1usize, 2, 4, 8, 16] {
                let (wall, set) = build_segmented(&vectors, dims, segments);
                assert_eq!(set.len(), n);
                let start = Instant::now();
                let mut hits = 0usize;
                for (q, want) in queries.iter().zip(&truth) {
                    let got = set.search_final(&vq(q), k, 1).unwrap();
                    hits += got.iter().filter(|(id, _)| want.contains(id)).count();
                }
                let query = start.elapsed() / queries.len() as u32;
                println!(
                    "{n:>8} {dims:>5} {segments:>9} {:>10.1} {:>12.1} {:>12.1} {:>9.3}",
                    wall.as_secs_f64() * 1000.0,
                    wall.as_secs_f64() * 1e6 / n as f64,
                    query.as_secs_f64() * 1e6,
                    hits as f64 / (queries.len() * k) as f64
                );
            }
        }
    }
}
