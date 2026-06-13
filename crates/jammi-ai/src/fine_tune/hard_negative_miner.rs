//! Hard-negative mining over jammi's own ANN index.
//!
//! Random and in-batch negatives are mostly easy — far from the anchor, so they
//! supply little gradient. *Hard* negatives are near-misses: corpus rows the
//! current model ranks close to the anchor but that are not its positive. Mining
//! them is the ANCE quality lever ([Xiong et al. 2020]).
//!
//! jammi mines them with its own retrieval primitive: build the cosine
//! [`VectorIndex`] over the candidate corpus (the dogfooding story — the same
//! index `search` and the S9 neighbour graph use), then for each anchor retrieve
//! the top-`k` nearest candidates as hard negatives.
//!
//! # False-negative guard
//!
//! A retrieved "negative" can be a true-but-unlabelled positive — acute on
//! near-duplicate corpora, where the anchor's positive has near-identical
//! siblings. Training on such a row supplies a false-negative gradient that
//! pushes apart things that should be close. The miner excludes the anchor's own
//! positive **and the positive's `exclude_hops`-hop neighbourhood** in the
//! index: anything within a few hops of the positive is presumed too close to be
//! a safe negative.
//!
//! # Staleness / cost trade
//!
//! Mined negatives go stale as the model moves, but re-mining every step means
//! re-embedding and re-indexing the corpus every step. [`should_refresh`]
//! implements ANCE's asynchronous-refresh compromise: re-mine once every
//! `refresh_every` epochs, accepting slightly stale negatives between refreshes
//! for a large constant-factor cost saving.
//!
//! [Xiong et al. 2020]: https://arxiv.org/abs/2007.00808

use std::collections::HashSet;

use jammi_db::config::AnnIndexConfig;
use jammi_db::error::{JammiError, Result};
use jammi_db::index::sidecar::SidecarIndex;
use jammi_db::index::VectorIndex;

use super::HardNegativeConfig;

/// Ceiling on how many *excluded* neighbours the over-fetch budgets for.
///
/// `mine` over-fetches `k + excluded.len() + 1` so that, after the excluded
/// neighbourhood and the self-hit are dropped, `k` survivors remain. The term
/// that can blow up is `excluded.len()`: on a dense corpus the
/// `exclude_hops`-hop expansion can grow it to a large fraction of the corpus,
/// and budgeting for all of it would turn the ANN query into a near-full scan —
/// defeating the index. So the over-fetch budgets for at most this many excluded
/// ids. The `k + 1` base is never capped, so the query always asks for at least
/// `k` survivors plus the self-hit; the bound only limits headroom for an
/// extreme excluded set. If that headroom is genuinely exhausted (the top
/// `k + 1 + MAX_EXCLUDED_FETCH` neighbours are all excluded), the anchor falls
/// through to the existing "no usable negative" drop path rather than returning
/// a wrong (excluded) negative.
const MAX_EXCLUDED_FETCH: usize = 512;

/// How many neighbours to over-fetch for one anchor: `k` survivors, the self-hit,
/// and headroom for the excluded ids — the excluded headroom capped at
/// [`MAX_EXCLUDED_FETCH`] so a pathological excluded set cannot escalate the
/// query toward a full scan. The `k + 1` base is uncapped, so `k` itself is
/// never the limiting factor regardless of how large it is set.
fn capped_fetch(k: usize, excluded_len: usize) -> usize {
    k + 1 + excluded_len.min(MAX_EXCLUDED_FETCH)
}

/// One candidate row available to be mined as a hard negative: a stable id and
/// its current embedding under the model being trained.
pub struct Candidate {
    /// Stable identifier for the row (used to exclude the positive and to map a
    /// mined id back to its embedding).
    pub id: String,
    /// The row's embedding under the current model.
    pub embedding: Vec<f32>,
}

/// One anchor to mine negatives for: its embedding and the id of its positive,
/// which (with the positive's neighbourhood) is excluded from the negative pool.
pub struct AnchorQuery {
    /// The anchor's embedding under the current model.
    pub embedding: Vec<f32>,
    /// The id of the anchor's labelled positive, excluded from mining.
    pub positive_id: String,
}

/// Whether to re-mine at the start of `epoch` (0-based) given `refresh_every`.
///
/// Epoch 0 always mines (no negatives yet); thereafter mining happens every
/// `refresh_every` epochs. `refresh_every == 1` mines every epoch.
pub fn should_refresh(epoch: usize, refresh_every: usize) -> bool {
    refresh_every > 0 && epoch % refresh_every == 0
}

/// Mines hard negatives from a cosine ANN index built over the candidate
/// corpus. Holds the built index so multiple anchor batches can be mined
/// against one index, and so the positive's neighbourhood can be expanded by
/// querying the same index.
pub struct HardNegativeMiner {
    index: SidecarIndex,
    config: HardNegativeConfig,
}

impl HardNegativeMiner {
    /// Build the miner: index every candidate by its embedding. All candidate
    /// embeddings must share one dimension. An empty candidate set is a typed
    /// error — there is nothing to mine.
    pub fn build(candidates: &[Candidate], config: HardNegativeConfig) -> Result<Self> {
        let dim = candidates
            .first()
            .map(|c| c.embedding.len())
            .ok_or_else(|| {
                JammiError::FineTune("hard-negative mining needs at least one candidate".into())
            })?;

        // A transient in-memory index for mining; the deployment's HNSW tuning
        // is irrelevant to a scratch graph that never persists, so use defaults.
        let mut index = SidecarIndex::new(dim, &AnnIndexConfig::default())?;
        for cand in candidates {
            if cand.embedding.len() != dim {
                return Err(JammiError::FineTune(format!(
                    "candidate '{}' has embedding dim {}, expected {dim}",
                    cand.id,
                    cand.embedding.len()
                )));
            }
            index.add(&cand.id, &cand.embedding)?;
        }
        index.build()?;

        Ok(Self { index, config })
    }

    /// Mine up to `k` hard negatives for one anchor, returning their ids in
    /// ascending-distance order (hardest first). Excludes the anchor's positive
    /// and the positive's `exclude_hops`-hop neighbourhood.
    ///
    /// To leave room after the excluded ids are dropped, the index is queried
    /// for more than `k` neighbours; the excluded set is removed and the first
    /// `k` survivors returned.
    pub fn mine(&self, anchor: &AnchorQuery) -> Result<Vec<String>> {
        let excluded = self.excluded_neighbourhood(&anchor.positive_id)?;

        // Over-fetch so that after removing the excluded neighbourhood (and any
        // self/positive hit) at least `k` candidates remain when they exist. The
        // excluded headroom is capped (see `capped_fetch`) so a pathologically
        // large excluded set cannot turn the ANN query into a near-full scan; the
        // `k + 1` base is uncapped. If that headroom is genuinely exhausted, the
        // anchor falls through to the drop path below — it is never given a wrong
        // (excluded) negative.
        let fetch = capped_fetch(self.config.k, excluded.len());
        let neighbours = self.index.search(&anchor.embedding, fetch)?;

        let mut mined = Vec::with_capacity(self.config.k);
        for (id, _dist) in neighbours {
            if excluded.contains(&id) {
                continue;
            }
            mined.push(id);
            if mined.len() == self.config.k {
                break;
            }
        }
        Ok(mined)
    }

    /// The set of ids to exclude as unsafe negatives: the positive itself plus
    /// every id within `exclude_hops` hops of it in the index. Computed by a
    /// breadth-first expansion that queries the index for each frontier id's
    /// neighbours — the same `search` the anchor mining uses.
    fn excluded_neighbourhood(&self, positive_id: &str) -> Result<HashSet<String>> {
        let mut excluded = HashSet::new();
        excluded.insert(positive_id.to_string());

        if self.config.exclude_hops == 0 {
            return Ok(excluded);
        }

        // The positive's own embedding seeds the expansion. If it is not in the
        // index the only exclusion is the positive id itself (already inserted).
        let mut frontier = vec![positive_id.to_string()];
        for _hop in 0..self.config.exclude_hops {
            let mut next = Vec::new();
            for id in &frontier {
                let Some(vector) = self.embedding_of(id)? else {
                    continue;
                };
                // k + 1 to cover the self-hit, which is dropped by the excluded
                // check below.
                let neighbours = self.index.search(&vector, self.config.k + 1)?;
                for (nid, _dist) in neighbours {
                    if excluded.insert(nid.clone()) {
                        next.push(nid);
                    }
                }
            }
            frontier = next;
            if frontier.is_empty() {
                break;
            }
        }
        Ok(excluded)
    }

    /// Look up a row's embedding by id from the index's own stored vectors.
    fn embedding_of(&self, id: &str) -> Result<Option<Vec<f32>>> {
        self.index.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: &str, embedding: Vec<f32>) -> Candidate {
        Candidate {
            id: id.to_string(),
            embedding,
        }
    }

    /// Mined negatives are *harder* than random: the top index neighbour of an
    /// anchor is more similar to it than an arbitrary corpus row.
    #[test]
    fn mined_negatives_are_harder_than_random() {
        // A near-cluster around the anchor plus a far row. The anchor's positive
        // sits in the cluster; the hard negative should be the other cluster
        // member (close), never the far row.
        let candidates = vec![
            cand("pos", vec![1.0, 0.0, 0.0]),
            cand("near", vec![0.98, 0.02, 0.0]),
            cand("far", vec![0.0, 0.0, 1.0]),
        ];
        let config = HardNegativeConfig {
            mine: true,
            k: 1,
            exclude_hops: 0, // isolate the harder-than-random property
            refresh_every: 1,
        };
        let miner = HardNegativeMiner::build(&candidates, config).unwrap();

        let anchor = AnchorQuery {
            embedding: vec![1.0, 0.0, 0.0],
            positive_id: "pos".to_string(),
        };
        let mined = miner.mine(&anchor).unwrap();
        assert_eq!(
            mined,
            vec!["near".to_string()],
            "should mine the close row, not the far one"
        );
    }

    /// The k-hop exclusion removes the positive's neighbourhood, so a
    /// near-duplicate of the positive is never mined as a false negative.
    #[test]
    fn k_hop_exclusion_blocks_positive_neighbourhood() {
        // `dup` is a near-duplicate of the positive (a likely false negative);
        // `real` is a genuinely different but still-near row. With 1-hop
        // exclusion the positive's neighbour `dup` is excluded and only `real`
        // is mined.
        let candidates = vec![
            cand("pos", vec![1.0, 0.0, 0.0, 0.0]),
            cand("dup", vec![0.999, 0.001, 0.0, 0.0]),
            cand("real", vec![0.6, 0.8, 0.0, 0.0]),
        ];
        let config = HardNegativeConfig {
            mine: true,
            k: 1,
            exclude_hops: 1,
            refresh_every: 1,
        };
        let miner = HardNegativeMiner::build(&candidates, config).unwrap();

        let anchor = AnchorQuery {
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            positive_id: "pos".to_string(),
        };
        let mined = miner.mine(&anchor).unwrap();
        assert!(
            !mined.contains(&"dup".to_string()),
            "the positive's 1-hop near-duplicate must be excluded, got {mined:?}"
        );
        assert_eq!(
            mined,
            vec!["real".to_string()],
            "only the safe negative should be mined"
        );
    }

    /// `refresh_every` is honoured: epoch 0 always mines; with `refresh_every=2`
    /// mining happens on epochs 0, 2, 4 and is skipped on 1, 3.
    #[test]
    fn refresh_every_schedules_mining() {
        assert!(should_refresh(0, 2));
        assert!(!should_refresh(1, 2));
        assert!(should_refresh(2, 2));
        assert!(!should_refresh(3, 2));
        // Every-epoch refresh.
        assert!(should_refresh(0, 1));
        assert!(should_refresh(1, 1));
    }

    /// The k-hop exclusion re-queries each frontier id's embedding. With the
    /// redundant `id_to_embedding` copy removed, that embedding now comes from
    /// the index's own stored vectors via [`SidecarIndex::get`]. A 2-hop
    /// expansion exercises the index lookup on ids that are *not* the original
    /// positive (the second-hop frontier), proving the index-backed lookup
    /// works: with both `dup` (1-hop of `pos`) and `bridge` (1-hop of `dup`,
    /// 2-hop of `pos`) excluded, only the far-but-safe `real` is mined.
    #[test]
    fn two_hop_expansion_reads_embeddings_from_index() {
        // `dup` is `pos`'s nearest neighbour (1-hop); `bridge` is `dup`'s nearest
        // neighbour but *farther* from `pos` than `dup` is, so it is reachable
        // only by expanding from `dup` — i.e. only via `dup`'s embedding fetched
        // back from the index.
        let candidates = vec![
            cand("pos", vec![1.0, 0.0, 0.0, 0.0]),
            cand("dup", vec![0.9, 0.436, 0.0, 0.0]),
            cand("bridge", vec![0.8, 0.6, 0.0, 0.0]),
            cand("real", vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let config = HardNegativeConfig {
            mine: true,
            k: 1,
            exclude_hops: 2,
            refresh_every: 1,
        };
        let miner = HardNegativeMiner::build(&candidates, config).unwrap();

        // The excluded neighbourhood must reach the 2-hop `bridge`, which is only
        // discoverable by querying `dup`'s embedding fetched back from the index.
        let excluded = miner.excluded_neighbourhood("pos").unwrap();
        assert!(excluded.contains("pos"));
        assert!(excluded.contains("dup"), "1-hop neighbour excluded");
        assert!(
            excluded.contains("bridge"),
            "2-hop neighbour reached via index-fetched embedding, got {excluded:?}"
        );

        let anchor = AnchorQuery {
            embedding: vec![1.0, 0.0, 0.0, 0.0],
            positive_id: "pos".to_string(),
        };
        let mined = miner.mine(&anchor).unwrap();
        assert_eq!(
            mined,
            vec!["real".to_string()],
            "only the safe far row survives the 2-hop exclusion"
        );
    }

    /// The excluded headroom is capped regardless of how large the excluded set
    /// grows on a dense corpus, so the ANN query can never escalate into a
    /// near-full scan; below the cap it is the exact over-fetch. The `k + 1` base
    /// is never capped, so `k` survivors are always budgeted for even when `k`
    /// itself is large.
    #[test]
    fn fetch_is_capped_under_dense_exclusion() {
        // Small excluded set: exact over-fetch (k + excluded + self-hit).
        assert_eq!(capped_fetch(1, 3), 5);
        assert_eq!(capped_fetch(5, 10), 16);
        // A pathological excluded set (a large fraction of a dense corpus) caps
        // the excluded headroom at the ceiling instead of fetching the corpus —
        // but the k+1 base is still added on top, so survivors are budgeted for.
        assert_eq!(capped_fetch(1, 100_000), 1 + 1 + MAX_EXCLUDED_FETCH);
        assert_eq!(capped_fetch(3, usize::MAX - 10), 3 + 1 + MAX_EXCLUDED_FETCH);
        // A large `k` is never the limiter: the base scales with k, uncapped, so
        // the query always asks for at least `k` survivors plus the self-hit.
        assert_eq!(capped_fetch(1000, 0), 1001);
        assert_eq!(capped_fetch(1000, 100_000), 1000 + 1 + MAX_EXCLUDED_FETCH);
    }

    /// Correctness is unchanged by the working-set bounding: for a fixed
    /// deterministic fixture the miner selects the SAME hard negative it did
    /// before the index-backed lookup and the fetch cap. With `k = 1` the
    /// positive's 1-hop neighbour `dup` is excluded, and the hardest remaining
    /// survivor — not the excluded near-duplicate, not the far row — is mined.
    #[test]
    fn selection_unchanged_for_deterministic_fixture() {
        let candidates = vec![
            cand("pos", vec![1.0, 0.0, 0.0]),
            cand("dup", vec![0.999, 0.001, 0.0]),
            cand("near", vec![0.9, 0.44, 0.0]),
            cand("mid", vec![0.7, 0.71, 0.0]),
            cand("far", vec![0.0, 0.0, 1.0]),
        ];
        let config = HardNegativeConfig {
            mine: true,
            k: 1,
            exclude_hops: 1,
            refresh_every: 1,
        };
        let miner = HardNegativeMiner::build(&candidates, config).unwrap();

        let anchor = AnchorQuery {
            embedding: vec![1.0, 0.0, 0.0],
            positive_id: "pos".to_string(),
        };
        let mined = miner.mine(&anchor).unwrap();
        // `dup` is excluded (1-hop of `pos`); the hardest survivor is `near`,
        // ahead of `mid` and `far`.
        assert_eq!(mined, vec!["near".to_string()]);
    }
}
