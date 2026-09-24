//! What a [`super::trainer::TrainingLoop::run`] call trains from (the
//! production binding of the residency-bounded stream).
//!
//! [`TrainingSource`] is the type the worker hands `run`: either an already
//! in-memory [`TrainingDataLoader`] ([`TrainingSource::Resident`] — the
//! whole-set arms, precomputed test loaders, and every in-memory loader) or
//! a [`StreamedSet`] ([`TrainingSource::Streamed`] — everything else at
//! `W = 1`), which names a materialised table and a row window WITHOUT ever
//! reading a row back: the eager `Vec<RecordBatch>` collect
//! ([`super::training_set::read_back`]) never runs for a `Streamed` source.
//!
//! `whole_set_arm` is the ONE predicate that decides which arm a
//! configuration takes — the worker's source selection
//! (`worker.rs::run_spec`) and the trainer's own dispatch (`trainer.rs::run`,
//! which refuses a whole-set arm reached with a `Streamed` source) call
//! the SAME function, so the two can never disagree about which
//! configurations must stay resident.

use std::sync::Arc;

use jammi_db::store::TrainingSetTable;

use crate::session::InferenceSession;
use jammi_datafusion::ModelTask;

use super::data::TrainingDataLoader;
use super::decode::LabelVocabulary;
use super::stream::StreamConfig;
use super::{EmbeddingLoss, FineTuneConfig};

/// A materialised training-set table this run reads through a per-rank
/// [`super::stream::TrainingSetStream`] rather than an eager collect — see
/// the module doc.
///
/// `total_rows`/`train_count` are both row COUNTS: `total_rows` is the
/// catalog record's `row_count` (no scan — the worker reads it off the
/// materialised table's manifest), and `train_count` is
/// `super::data::split_index` applied to it, the SAME arithmetic
/// [`TrainingDataLoader::split`] uses over an already-resident row count.
#[derive(Clone)]
pub struct StreamedSet {
    pub(crate) session: Arc<InferenceSession>,
    pub(crate) table: TrainingSetTable,
    pub(crate) columns: Vec<String>,
    pub(crate) task: ModelTask,
    pub(crate) total_rows: usize,
    pub(crate) train_count: usize,
    pub(crate) batch: usize,
    pub(crate) stream_cfg: StreamConfig,
    /// The job's tenant scope, captured (`InferenceSession::tenant()`) at
    /// construction time — while the worker's `run_spec` is still running
    /// inside `with_tenant_scoped`'s task-local scope (`worker.rs`'s doc on
    /// that call). `TrainingLoop::open_streamed_source` runs on the
    /// `spawn_blocking` pool via `Handle::block_on`, a fresh top-level poll
    /// on a different OS thread that does NOT inherit a Tokio task-local —
    /// so every per-epoch stream open re-enters this scope explicitly rather
    /// than relying on inheritance. `None` for an unscoped
    /// run (the queue-drain worker's own claim, or a test session that never
    /// bound a tenant).
    pub(crate) tenant: Option<jammi_db::TenantId>,
    /// Built ONCE by the worker's own whole-table sweep — `None` for
    /// every non-classification task. Carries the FULL label→index
    /// assignment, not just its cardinality: every per-epoch training-window
    /// open AND the validation-window open need the actual vocabulary to
    /// decode a chunk's class indices (a per-step accumulator can never
    /// invent one of its own — `decode::LabelVocabulary`'s own doc), so this
    /// is cloned into each `TrainingSetStream::open` call
    /// (`TrainingLoop::open_streamed_source`) rather than re-derived from a
    /// second whole-table scan every time. [`Self::num_classes`] is the
    /// derived count the classification head is sized from — carrying the
    /// full vocabulary here is a strict superset of "carry `num_classes`":
    /// the count is always recoverable from it, never the reverse.
    pub(crate) label_vocab: Option<LabelVocabulary>,
}

impl StreamedSet {
    /// The classification head's output width — `None` for every
    /// non-classification task, `Some(vocab.num_classes())` otherwise. The
    /// SAME quantity the eager path's `TrainingFormat::Classification {
    /// num_classes }` carries, computed from the identical whole-table
    /// vocabulary.
    pub(crate) fn num_classes(&self) -> Option<usize> {
        self.label_vocab.as_ref().map(LabelVocabulary::num_classes)
    }
}

/// What a [`super::trainer::TrainingLoop::run`] call trains from: an
/// already-resident loader, or a streamed source that opens a fresh
/// [`super::stream::TrainingSetStream`] each epoch.
///
/// `Streamed` is boxed: `StreamedSet` carries an `Arc<InferenceSession>`
/// plus a `TrainingSetTable`/`columns`/an optional whole-table
/// `LabelVocabulary`, making it substantially larger than `Resident`'s bare
/// `TrainingDataLoader` — `clippy::large_enum_variant` flags the gap, and
/// this type is moved by value at every `TrainingSource` construction site
/// (the worker builds one per job, not per row), so the indirection costs
/// nothing observable.
pub enum TrainingSource {
    Resident(TrainingDataLoader),
    Streamed(Box<StreamedSet>),
}

impl TrainingSource {
    /// The same source for ANOTHER rank of an in-process gang
    /// (`worker.rs`'s local fan-out): a `Resident` loader is replicated
    /// over the same rows ([`TrainingDataLoader::replicate`]); a `Streamed`
    /// set is cloned — every rank opens its own per-epoch stream over the
    /// same committed table and window.
    pub(crate) fn replicate(&self) -> Self {
        match self {
            Self::Resident(loader) => Self::Resident(loader.replicate()),
            Self::Streamed(streamed) => Self::Streamed(streamed.clone()),
        }
    }
}

/// The two whole-table training arms a [`StreamedSet`] is exempted from
/// (the `Σ E` term of `stream.rs`'s module doc): mining scores every
/// candidate against the FULL corpus, and GradCache treats the whole
/// dataset as one in-batch-negative batch — both need every row resident
/// before the epoch can begin, so neither one can ever run against a
/// per-step stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WholeSetArm {
    Mining,
    GradCache,
}

fn is_in_batch_negative_objective(config: &FineTuneConfig) -> bool {
    matches!(
        config.embedding_loss,
        Some(EmbeddingLoss::MultipleNegativesRanking { .. })
    )
}

/// Whether a run mines hard negatives: `mine` is on, the objective is the
/// in-batch-negative one (mining only feeds that path), and a base model is
/// present to embed the corpus. The SAME predicate
/// [`super::trainer::TrainingLoop::mining_eligible`] calls (with
/// `has_base_model = self.base_model.is_some()`, the one piece of trainer
/// state this free function cannot read directly).
pub(crate) fn mining_eligible(config: &FineTuneConfig, has_base_model: bool) -> bool {
    has_base_model && config.hard_negatives.mine && is_in_batch_negative_objective(config)
}

/// Whether a run takes the GradCache path: `cached` is on, the objective is
/// the in-batch-negative one, and a base model is present. The SAME
/// predicate [`super::trainer::TrainingLoop::gradcache_eligible`] calls.
///
/// Independent of [`mining_eligible`] — a config with BOTH `mine` and
/// `cached` set is legal (mining replaces the epoch's data, which then
/// feeds the GradCache epoch): this function does not treat mining as
/// taking precedence over GradCache, or the reverse, matching
/// `trainer.rs::run`'s own per-epoch dispatch, which checks the two
/// conditions separately rather than through one mutually-exclusive choice.
pub(crate) fn gradcache_eligible(config: &FineTuneConfig, has_base_model: bool) -> bool {
    has_base_model && config.cached && is_in_batch_negative_objective(config)
}

/// Whether `config` (under a run that does/doesn't have a base model to
/// embed through) takes a whole-set arm AT ALL — the predicate
/// `worker.rs::run_spec`'s source selection calls: the
/// worker selects [`TrainingSource::Resident`] whenever this returns
/// `Some`, and `trainer.rs::run` refuses (typed) reaching a whole-set arm
/// with a `Streamed` source — so the two decisions can never come apart.
///
/// Reports [`WholeSetArm::Mining`] when [`mining_eligible`] holds
/// (regardless of whether [`gradcache_eligible`] ALSO holds — see that
/// function's own doc on the two being independent): this return value is
/// consumed only as "is some whole-set arm active" (the worker's
/// Resident-vs-Streamed choice) and as a diagnostic label (the trainer's
/// streamed-whole-set refusal message), never to pick which per-epoch branch runs — that
/// choice is `mining_eligible()`/`gradcache_eligible()`, called
/// independently.
pub(crate) fn whole_set_arm(config: &FineTuneConfig, has_base_model: bool) -> Option<WholeSetArm> {
    if mining_eligible(config, has_base_model) {
        Some(WholeSetArm::Mining)
    } else if gradcache_eligible(config, has_base_model) {
        Some(WholeSetArm::GradCache)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fine_tune::data::split_index;

    /// `split_index` (shared with `TrainingDataLoader::split`) is the
    /// SAME boundary a hand-rolled `total - round(total * fraction)`
    /// computes, for every `total` a training set could realistically hold
    /// and every fraction `FineTuneConfig::validate` admits.
    #[test]
    fn split_index_matches_the_resident_split_boundary() {
        for total in 0..=1000usize {
            for &fraction in &[0.0, 0.01, 0.05, 0.1, 0.2, 0.25, 0.33, 0.5, 0.9, 0.99] {
                let expected = {
                    let val_count = (total as f64 * fraction).round() as usize;
                    total - val_count
                };
                assert_eq!(
                    split_index(total, fraction),
                    expected,
                    "total={total} fraction={fraction}"
                );
                // The boundary is always a valid prefix length: never past
                // `total`, never negative (usize forbids that structurally).
                assert!(split_index(total, fraction) <= total);
            }
        }
    }

    fn base_config() -> FineTuneConfig {
        FineTuneConfig::default()
    }

    #[test]
    fn whole_set_arm_is_none_without_a_base_model() {
        let mut cfg = base_config();
        cfg.embedding_loss = Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        cfg.hard_negatives.mine = true;
        assert_eq!(whole_set_arm(&cfg, false), None);
    }

    #[test]
    fn whole_set_arm_is_none_for_a_non_in_batch_negative_objective() {
        let mut cfg = base_config();
        cfg.embedding_loss = Some(EmbeddingLoss::CosineMse);
        cfg.hard_negatives.mine = true;
        cfg.cached = true;
        assert_eq!(whole_set_arm(&cfg, true), None);
    }

    #[test]
    fn whole_set_arm_is_mining_when_mine_is_set() {
        let mut cfg = base_config();
        cfg.embedding_loss = Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        cfg.hard_negatives.mine = true;
        assert_eq!(whole_set_arm(&cfg, true), Some(WholeSetArm::Mining));
    }

    #[test]
    fn whole_set_arm_is_gradcache_when_cached_is_set() {
        let mut cfg = base_config();
        cfg.embedding_loss = Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        cfg.cached = true;
        assert_eq!(whole_set_arm(&cfg, true), Some(WholeSetArm::GradCache));
    }

    #[test]
    fn whole_set_arm_is_none_for_a_plain_in_batch_negative_run() {
        let mut cfg = base_config();
        cfg.embedding_loss = Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        assert_eq!(whole_set_arm(&cfg, true), None);
    }

    /// `whole_set_arm`'s label is `Mining` when BOTH are set (a diagnostic
    /// preference only — see its own doc) but [`mining_eligible`] AND
    /// [`gradcache_eligible`] are BOTH `true` underneath, independently:
    /// this is the property that actually matters for `trainer.rs::run`'s
    /// per-epoch dispatch, which checks the two conditions separately.
    #[test]
    fn whole_set_arm_prefers_mining_when_both_are_set_but_both_predicates_stay_independent() {
        let mut cfg = base_config();
        cfg.embedding_loss = Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        cfg.hard_negatives.mine = true;
        cfg.cached = true;
        assert_eq!(whole_set_arm(&cfg, true), Some(WholeSetArm::Mining));
        assert!(mining_eligible(&cfg, true));
        assert!(gradcache_eligible(&cfg, true));
    }

    #[test]
    fn mining_and_gradcache_eligible_are_independent_predicates() {
        let mut mining_only = base_config();
        mining_only.embedding_loss =
            Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        mining_only.hard_negatives.mine = true;
        assert!(mining_eligible(&mining_only, true));
        assert!(!gradcache_eligible(&mining_only, true));

        let mut gradcache_only = base_config();
        gradcache_only.embedding_loss =
            Some(EmbeddingLoss::MultipleNegativesRanking { temperature: 0.05 });
        gradcache_only.cached = true;
        assert!(!mining_eligible(&gradcache_only, true));
        assert!(gradcache_eligible(&gradcache_only, true));
    }
}
